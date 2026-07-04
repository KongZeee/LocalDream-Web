import io
import base64
import time
import json
import asyncio
import threading
import importlib
from pathlib import Path
from typing import Optional, AsyncGenerator

# ── Compatibility fix ───────────────────────────────────────────────────────
# peft >= 0.14 imports EncoderDecoderCache from transformers, but
# transformers 4.42.0 (C:\py_pkgs) does not ship it yet.
# Patch transformers before peft/diffusers loads so the import succeeds.
try:
    import transformers as _hf_tf
    if not hasattr(_hf_tf, "EncoderDecoderCache"):
        class _DummyEncoderDecoderCache:
            pass
        _hf_tf.EncoderDecoderCache = _DummyEncoderDecoderCache
    if not hasattr(_hf_tf, "DynamicCache"):
        class _DummyDynamicCache:
            pass
        _hf_tf.DynamicCache = _DummyDynamicCache
    importlib.reload(_hf_tf)
    del _hf_tf
except Exception:
    pass
# ──────────────────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F
import torch_directml
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
)

from app.config import MODELS_DIR, OUTPUT_DIR
from app.services.lora_manager import apply_loras, clear_all_loras, mark_pipeline, scan_loras

_EMBEDDING_PATCH_LOCK = threading.Lock()
_EMBEDDING_PATCH_DONE = False
_ORIG_EMBEDDING = F.embedding

# CPU VAE for SDXL (loaded on demand, fp32)
_cpu_vae_cache = None


def _get_cpu_vae_fp32():
    """Lazily load a fp32-float VAE for CPU decoding. Returns a cached instance."""
    global _cpu_vae_cache
    if _cpu_vae_cache is None:
        from diffusers import AutoencoderKL
        _cpu_vae_cache = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float32,
        )
        _cpu_vae_cache.to("cpu").float()
        print("[Generator] Loaded fp16-fix VAE (fp32) on CPU for SDXL decode")
    return _cpu_vae_cache


def _patch_nn_embedding(m: torch.nn.Module):
    """Patch a single nn.Embedding to fall back to CPU when called with DML tensors."""
    orig_fwd = m.forward

    def patched_fwd(input):
        i_dev = str(input.device)
        w_dev = str(m.weight.device)
        if i_dev == w_dev:
            return orig_fwd(input)
        if i_dev.startswith("privateuseone") or w_dev.startswith("privateuseone"):
            w_cpu = m.weight.detach().to("cpu")
            out = F.embedding(input.detach().to("cpu"), w_cpu, m.padding_idx,
                              m.max_norm, m.norm_type, m.scale_grad_by_freq, m.sparse)
            tgt = w_dev if not w_dev.startswith("privateuseone") else i_dev
            if tgt.startswith("privateuseone"):
                return out.to(tgt).to(m.weight.dtype)
            return out
        return orig_fwd(input)

    m.forward = patched_fwd


_ORIG_LINEAR = F.linear


def _patched_linear(input, weight, bias=None):
    """Global F.linear patch (registered for non-UNet call sites if any)."""
    i_dev = str(input.device)
    if not i_dev.startswith("privateuseone"):
        return _ORIG_LINEAR(input, weight, bias)
    try:
        return _ORIG_LINEAR(input, weight, bias)
    except Exception as e:
        err_str = str(e)
        if "INTERNAL ASSERT" in err_str or "unbox" in err_str.lower():
            w_cpu = weight.detach().to("cpu").float()
            b_cpu = bias.detach().to("cpu").float() if bias is not None else None
            i_cpu = input.detach().to("cpu").float()
            out = _ORIG_LINEAR(i_cpu, w_cpu, b_cpu)
            return out.to(i_dev).to(weight.dtype)
        raise


def _patch_nn_linear(m: torch.nn.Module):
    """Patch a single nn.Linear to fall back to CPU on DML internal assertion."""
    orig_fwd = m.forward

    def patched_fwd(input):
        i_dev = str(input.device)
        w_dev = str(m.weight.device)
        try:
            if i_dev != w_dev:
                return orig_fwd(input.to(m.weight.device))
            return orig_fwd(input)
        except Exception as e:
            err_str = str(e)
            if "INTERNAL ASSERT" in err_str or "unbox" in err_str.lower():
                w_cpu = m.weight.detach().to("cpu").float()
                b_cpu = m.bias.detach().to("cpu").float() if m.bias is not None else None
                i_cpu = input.detach().to("cpu").float()
                out = _ORIG_LINEAR(i_cpu, w_cpu, b_cpu)
                return out.to(w_dev).to(m.weight.dtype)
            raise

    m.forward = patched_fwd


def _patch_nn_conv2d(m: torch.nn.Module):
    """Patch a single nn.Conv2d to align device between input and weight.

    DirectML on RDNA 1 has two quirks:
    1. Some conv shapes (`_slow_conv2d_forward`) fall back to CPU. We detect
       the resulting CPU output and move it back to the weight device.
    2. Some DML tensors (e.g. ``sample`` at the UNet entry) are still on CPU
       while the conv weight is on DML. F.conv2d errors in that case, so we
       align the input to the weight device first.
    """
    orig_fwd = m.forward
    _n_fallback = [0]

    def patched_fwd(input):
        target = m.weight.device
        # Align input to weight device (handles CPU latents entering DML UNet).
        if input.device != target:
            input = input.to(target)
        out = orig_fwd(input)
        # Some conv2d paths (slow_conv2d) fall back to CPU even when input is
        # on DML. Move the output back to the weight device.
        if out.device != target:
            _n_fallback[0] += 1
            if _n_fallback[0] <= 3:
                print(f"[Generator] Conv2d CPU-fallback #{_n_fallback[0]} (stride={m.stride}, groups={m.groups}) {input.shape}->{out.shape}")
            out = out.to(target)
        return out

    m.forward = patched_fwd


def _patch_nn_groupnorm(m: torch.nn.Module):
    """Patch a single nn.GroupNorm to fall back to CPU on DML internal assertion.

    Always returns the tensor on the same device as the input.
    """
    orig_fwd = m.forward
    _n_fallback = [0]

    def patched_fwd(input):
        try:
            return orig_fwd(input)
        except Exception as e:
            err_str = str(e)
            if "INTERNAL ASSERT" in err_str or "unbox" in err_str.lower():
                _n_fallback[0] += 1
                if _n_fallback[0] <= 3:
                    print(f"[Generator] GroupNorm fallback #{_n_fallback[0]} ({m.num_groups} groups) shape={input.shape}")
                w_cpu = m.weight.detach().to("cpu") if m.weight is not None else None
                b_cpu = m.bias.detach().to("cpu") if m.bias is not None else None
                i_cpu = input.detach().to("cpu")
                out = F.group_norm(i_cpu, m.num_groups, w_cpu, b_cpu, m.eps)
                # Move back to the same device + dtype as the input.
                return out.to(input.device).to(input.dtype)
            raise

    m.forward = patched_fwd


def _install_embedding_patch(pipe):
    global _EMBEDDING_PATCH_DONE
    with _EMBEDDING_PATCH_LOCK:
        if _EMBEDDING_PATCH_DONE:
            return
        for name, module in pipe.text_encoder.named_modules():
            if isinstance(module, torch.nn.Embedding):
                _patch_nn_embedding(module)
        for name, module in pipe.text_encoder_2.named_modules():
            if isinstance(module, torch.nn.Embedding):
                _patch_nn_embedding(module)
        _EMBEDDING_PATCH_DONE = True
        print("[Generator] Installed DirectML embedding workaround")

SCHEDULER_MAP = {
    "dpm": DPMSolverMultistepScheduler,
    "dpm_karras": lambda c: DPMSolverMultistepScheduler.from_config(c, use_karras_sigmas=True),
    "euler": EulerDiscreteScheduler,
    "euler_karras": lambda c: EulerDiscreteScheduler.from_config(c, use_karras_sigmas=True),
    "euler_a": EulerAncestralDiscreteScheduler,
    "lcm": LCMScheduler,
}

SPEED_PRESETS = {
    "fast": {"steps": 10, "cfg": 4.0, "scheduler": "dpm"},
    "balanced": {"steps": 15, "cfg": 7.0, "scheduler": "dpm"},
    "quality": {"steps": 25, "cfg": 7.5, "scheduler": "dpm_karras"},
}

_pipelines: dict = {}
_pipeline_lock = threading.Lock()
_dml_device = None
_warmed_up: set = set()

# LRU eviction: keep at most 2 pipelines in VRAM
_MAX_CACHED_PIPELINES = 2
_pipeline_access_order: list[str] = []  # oldest → newest


def _touch(key: str) -> None:
    """Move key to end of access order (most-recently-used)."""
    global _pipeline_access_order
    _pipeline_access_order = [k for k in _pipeline_access_order if k != key]
    _pipeline_access_order.append(key)


def _report_vram(label: str = "") -> None:
    """Report current DML VRAM usage."""
    try:
        import torch_directml
        used = torch_directml.memory_allocated() / (1024**3)
        total = torch_directml.get_device_properties(0).total_memory / (1024**3)
        msg = f"[VRAM] {label} used={used:.1f}GB / total={total:.1f}GB ({used/total*100:.0f}%)"
    except Exception as e:
        msg = f"[VRAM] {label} query failed: {e}"
    try:
        import sys
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _empty_dml_cache() -> None:
    """Free all cached DML allocations (calls torch.cuda.empty_cache equivalent)."""
    try:
        import torch
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import torch_directml
        if hasattr(torch_directml, 'empty_cache'):
            torch_directml.empty_cache()
    except Exception:
        pass
    import gc; gc.collect()


def _evict_lru_unlocked() -> None:
    """Evict the oldest pipeline to free VRAM — caller must hold _pipeline_lock."""
    global _pipeline_access_order
    while len(_pipelines) >= _MAX_CACHED_PIPELINES and _pipeline_access_order:
        oldest = _pipeline_access_order[0]
        pipe = _pipelines.pop(oldest, None)
        _pipeline_access_order = _pipeline_access_order[1:]
        if pipe is not None:
            try:
                del pipe
            except Exception:
                pass
    if len(_pipelines) >= _MAX_CACHED_PIPELINES:
        import gc
        gc.collect()


def _cleanup_stale_lora_pipelines(model_id: str, mode: str, fresh_key: str) -> None:
    """Evict any previously-cached LoRA pipeline entries for this model that
    are not the freshly-created ``fresh_key``.

    Called after ``apply_loras`` so that switching LoRA combos (or removing
    them entirely) immediately drops the old pipeline+adapters from VRAM.
    The fresh pipeline itself is never touched.
    """
    with _pipeline_lock:
        prefix = f"{model_id}:{mode}:<"
        stale = [k for k in _pipelines if k.startswith(prefix) and k != fresh_key]
        if not stale:
            return
        print(f"[LoRA] evicting {len(stale)} stale pipeline(s): {stale}")
        for k in stale:
            pipe = _pipelines.pop(k, None)
            if pipe is not None:
                try:
                    del pipe
                except Exception:
                    pass
        # Also prune from LRU access order
        stale_set = set(stale)
        _pipeline_access_order = [k for k in _pipeline_access_order if k not in stale_set]
        import gc; gc.collect()


def _evict_lru() -> None:
    """Thread-safe wrapper: acquire lock then evict."""
    with _pipeline_lock:
        _evict_lru_unlocked()


def _get_pipeline(model_id: str, mode: str, is_sdxl: bool, loras: list | None = None):
    global _dml_device
    if _dml_device is None:
        _dml_device = torch_directml.device()

    cache_key = f"{model_id}:{mode}"
    if loras:
        lora_tag = ",".join(f"{s['id']}:{s.get('weight', 0.7)}" for s in sorted(loras, key=lambda x: x["id"]))
        cache_key = f"{model_id}:{mode}:<{lora_tag}>"

    with _pipeline_lock:
        if cache_key in _pipelines:
            _touch(cache_key)
            return _pipelines[cache_key]

        # Free VRAM if we already have the max number of pipelines cached.
        _evict_lru_unlocked()

        model_path = Path(MODELS_DIR) / model_id.replace("/", "--")
        is_local = model_path.exists()
        model_path_str = str(model_path) if is_local else model_id

        pipe_kwargs = {
            "torch_dtype": torch.float16,
            "safety_checker": None,
            "requires_safety_checker": False,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if is_local:
            pipe_kwargs["local_files_only"] = True

        if is_sdxl:
            if mode == "img2img":
                pipe_cls = StableDiffusionXLImg2ImgPipeline
            elif mode == "inpaint":
                pipe_cls = StableDiffusionXLInpaintPipeline
            else:
                pipe_cls = StableDiffusionXLPipeline
        else:
            if mode == "img2img":
                pipe_cls = StableDiffusionImg2ImgPipeline
            elif mode == "inpaint":
                pipe_cls = StableDiffusionInpaintPipeline
            else:
                pipe_cls = StableDiffusionPipeline

        pipe = pipe_cls.from_pretrained(model_path_str, **pipe_kwargs)

        if is_sdxl:
            try:
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=torch.float16,
                )
                pipe.vae = vae
                print("[Generator] Using sdxl-vae-fp16-fix VAE (fp16 for DML)")
            except Exception as e:
                print(f"[Generator] fp16-fix VAE load failed: {e}, using original VAE")

            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")

            # DirectML on RDNA 1 (RX 5700 XT) has internal assertion failures
            # in fp16 nn.Embedding, nn.Linear, and F.group_norm in SDXL UNet.
            # Keep UNet on DML fp16 (5GB) and patch the buggy ops to fall back
            # to CPU on the rare assertion failure. Text encoders on CPU.
            pipe.unet.to(_dml_device)
            pipe.unet.half()
            # Use channels_last memory format for DML conv compatibility.
            try:
                pipe.unet.to(memory_format=torch.channels_last)
            except Exception:
                pass
            # VAE on DML fp16 so that the pipeline's self.device (which reads
            # the first module's device) is DML. This makes _execution_device
            # DML, so prepare_latents creates latents on DML (matches the UNet
            # output and avoids the scheduler device mismatch).
            try:
                pipe.vae.to(_dml_device).half()
                print(f"[Generator] SDXL mode: UNet fp16 channels_last on DML, VAE fp16 on DML, text encoders on CPU")
            except Exception as e:
                print(f"[Generator] VAE -> DML failed ({e}), keeping on CPU")
                pipe.vae.to("cpu").float()

            # Patch text encoder nn.Embedding.
            _install_embedding_patch(pipe)

            # Patch UNet nn.Linear, nn.GroupNorm, and nn.Conv2d forward methods
            # to fall back / align devices on DirectML quirks.
            n_lin, n_gn, n_conv = 0, 0, 0
            for name, module in pipe.unet.named_modules():
                if isinstance(module, torch.nn.Linear):
                    _patch_nn_linear(module)
                    n_lin += 1
                elif isinstance(module, torch.nn.GroupNorm):
                    _patch_nn_groupnorm(module)
                    n_gn += 1
                elif isinstance(module, torch.nn.Conv2d):
                    _patch_nn_conv2d(module)
                    n_conv += 1
            print(f"[Generator] Patched {n_lin} nn.Linear, {n_gn} nn.GroupNorm, {n_conv} nn.Conv2d in UNet for DML RDNA 1")

            # Also patch VAE convs so DML VAE decode doesn't crash on slow_conv2d.
            for name, module in pipe.vae.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    _patch_nn_conv2d(module)
            print(f"[Generator] Patched VAE nn.Conv2d for DML RDNA 1")

            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            try:
                pipe.enable_attention_slicing("max")
            except Exception:
                pass
        else:
            pipe.to(_dml_device)
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

        # Patch the default scheduler.step so warmup (which uses the default
        # scheduler from the model config) does not crash on device mismatch.
        try:
            sched = pipe.scheduler
            if sched is not None and not getattr(sched, "_ld_dml_step_patched", False):
                _patch_scheduler_step(sched)
                sched._ld_dml_step_patched = True
        except Exception as e:
            print(f"[Generator] default scheduler patch failed: {e}")

        # Move scheduler sigmas / alphas_cumprod onto DML so all the
        # arithmetic inside scheduler.step runs on the same device as the
        # UNet output (avoids CPU<->DML ping-pong that can corrupt the
        # latent trajectory on RDNA 1 + directml).
        # NOTE: disabled — Euler scheduler calls numpy() on sigmas during
        # set_timesteps which fails on DML tensors.
        # try:
        #     _force_scheduler_to_dml(pipe)
        # except Exception as e:
        #     print(f"[Generator] _force_scheduler_to_dml failed: {e}")

        _pipelines[cache_key] = pipe
        mark_pipeline(pipe, model_id, mode, loras)
        apply_loras(pipe, loras)
        # Any previously-cached LoRA pipeline with a different combo is now stale —
        # evict it so VRAM is freed immediately on LoRA switch / remove.
        _cleanup_stale_lora_pipelines(model_id, mode, cache_key)
        _touch(cache_key)
        return pipe


def _warmup_model(model_id: str, loras: list | None = None) -> float:
    if model_id in _warmed_up:
        return 0.0
    _report_vram("warmup start")
    pipe = _get_pipeline(model_id, "txt2img", "xl" in model_id.lower(), loras)
    t0 = time.time()
    try:
        _ = pipe(
            prompt="warmup",
            num_inference_steps=2,
            guidance_scale=1.0,
            width=64,
            height=64,
            output_type="latent",
        )
        print(f"[Warmup] {model_id} OK in {time.time() - t0:.1f}s")
    except Exception as e:
        import traceback as _tb
        print(f"[Warmup] {model_id} FAILED: {e}")
        _tb.print_exc()
    elapsed = time.time() - t0
    _warmed_up.add(model_id)
    return elapsed


def _set_scheduler(pipe, scheduler_name: str):
    if scheduler_name in SCHEDULER_MAP:
        sched_cls = SCHEDULER_MAP[scheduler_name]
        if callable(sched_cls) and not isinstance(sched_cls, type):
            pipe.scheduler = sched_cls(pipe.scheduler.config)
        else:
            pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
        # Patch the scheduler's step() to align model_output/sample to whichever
        # device the scheduler's tensors live on. The DPMSolverMultistep
        # scheduler sets ``self.sigmas = self.sigmas.to("cpu")`` at the end of
        # set_timesteps() to "avoid too much CPU/GPU communication", so the
        # scheduler-internal sigma_t / alpha_t end up on CPU. Meanwhile the UNet
        # returns noise_pred on DML. Without this alignment, scheduler.step
        # crashes with "privateuseone:0 and cpu" in arithmetic like
        # ``sample - sigma_t * model_output``.
        sched = pipe.scheduler
        if not getattr(sched, "_ld_dml_step_patched", False):
            _patch_scheduler_step(sched)
            sched._ld_dml_step_patched = True


def _patch_scheduler_step(sched):
    """Wrap ``scheduler.step`` so model_output/sample follow the scheduler's
    device before arithmetic runs.

    Works in tandem with the explicit sigmas/alphas_cumprod relocation done
    after pipeline load (see ``_force_scheduler_to_dml``).
    """
    orig_step = sched.step
    orig_set_timesteps = sched.set_timesteps

    def set_timesteps_wrapper(*args, **kwargs):
        out = orig_set_timesteps(*args, **kwargs)
        # DPMSolverMultistep calls ``self.sigmas = self.sigmas.to("cpu")``
        # at the end of set_timesteps. Re-move them to the runtime device.
        target_dev = getattr(sched, "_ld_dml_runtime_device", None)
        if target_dev is not None:
            for attr in ("sigmas", "alphas_cumprod", "betas", "alphas"):
                t = getattr(sched, attr, None)
                if isinstance(t, torch.Tensor) and t.device != target_dev:
                    try:
                        setattr(sched, attr, t.to(target_dev))
                    except Exception:
                        pass
        return out

    def step_wrapper(model_output, timestep, sample, *args, **kwargs):
        # Cache the runtime device from sample (UNet output -> DML).
        if isinstance(sample, torch.Tensor):
            sched._ld_dml_runtime_device = sample.device
        # Align model_output to sample device (Euler keeps scheduler tensors
        # on CPU by default in some versions).
        if isinstance(model_output, torch.Tensor) and isinstance(sample, torch.Tensor):
            if model_output.device != sample.device:
                model_output = model_output.to(sample.device)
        return orig_step(model_output, timestep, sample, *args, **kwargs)

    sched.set_timesteps = set_timesteps_wrapper
    sched.step = step_wrapper


def _b64_to_image(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _image_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def _generate_preview(pipe, latents: torch.Tensor, is_sdxl: bool) -> str | None:
    """Quick VAE decode of latents to produce a preview image, used in callbacks."""
    try:
        # SDXL: use the dedicated CPU fp32 VAE to avoid dtype/device mismatch on DML.
        if is_sdxl:
            vae = _get_cpu_vae_fp32()
        else:
            vae = getattr(pipe, "vae", None)
        if vae is None:
            print("[Preview] No VAE on pipe")
            return None

        # latents shape: [B, C, H, W] or [C, H, W]
        if latents.dim() == 3:
            latents = latents.unsqueeze(0)

        # Always decode on CPU with fp32 latents for stability.
        lat_cpu = latents.to("cpu").float()
        print(f"[Preview] latents shape={latents.shape} device={latents.device}")

        # Move VAE to CPU fp32 as well, then restore after decode.
        vae_orig_device = next(vae.parameters()).device
        vae_orig_dtype = next(vae.parameters()).dtype
        vae_cpu = vae.to("cpu").float()

        with torch.no_grad():
            decoded = vae_cpu.decode(
                lat_cpu / vae_cpu.config.scaling_factor,
                return_dict=False,
            )[0]

        # Restore VAE to original device/dtype.
        vae.to(vae_orig_device).to(vae_orig_dtype)

        print(f"[Preview] decoded shape={decoded.shape} device={decoded.device}")

        # decoded: [B, C, H, W] -> [C, H, W] -> PIL
        img = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

        # Resize to max 256px for preview (fast)
        max_side = 256
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        result = _image_to_b64(img, "JPEG")
        print(f"[Preview] encoded len={len(result)}")
        return result
    except Exception as e:
        print(f"[Preview] ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def generate_image(
    prompt: str,
    negative_prompt: str,
    model_id: str,
    mode: str,
    steps: int,
    cfg: float,
    seed: int,
    width: int,
    height: int,
    scheduler: str,
    denoise_strength: float,
    image_b64: Optional[str],
    mask_b64: Optional[str],
    show_preview: bool,
    preview_stride: int,
    speed_mode: str = "balanced",
    loras: list | None = None,
) -> AsyncGenerator[dict, None]:
    start_time = time.time()
    first_step_time = 0
    progress_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    is_sdxl = "xl" in model_id.lower()

    if speed_mode in SPEED_PRESETS:
        preset = SPEED_PRESETS[speed_mode]
        if steps <= 0:
            steps = preset["steps"]
        if cfg <= 0:
            cfg = preset["cfg"]
        if not scheduler or scheduler == "dpm":
            scheduler = preset["scheduler"]

    _warmup_model(model_id, loras)

    _report_vram("before generate")
    pipe = _get_pipeline(model_id, mode, is_sdxl, loras)
    _report_vram("after get_pipeline")
    _set_scheduler(pipe, scheduler)

    torch.manual_seed(seed)

    ref_image = None
    mask_image = None
    if image_b64 and mode in ("img2img", "inpaint"):
        ref_image = _b64_to_image(image_b64).resize((width, height))
    if mask_b64 and mode == "inpaint":
        mask_image = _b64_to_image(mask_b64).resize((width, height)).convert("L")

    def callback(pipeline, step_idx: int, _timestep: int, callback_kwargs):
        nonlocal first_step_time
        if step_idx == 0:
            first_step_time = int((time.time() - start_time) * 1000)
        
        # Send step progress
        progress_event = {"type": "progress", "step": step_idx + 1, "total_steps": steps}
        
        # Generate preview image if enabled and on stride
        latents = callback_kwargs.get("latents")
        if (
            show_preview
            and latents is not None
            and isinstance(latents, torch.Tensor)
            and (step_idx + 1) % preview_stride == 0
        ):
            try:
                preview_b64 = _generate_preview(pipeline, latents, is_sdxl)
                if preview_b64:
                    progress_event["image"] = preview_b64
                    progress_event["preview_format"] = "jpeg"
            except Exception as e:
                print(f"[Preview] callback error: {e}")
                pass
        
        loop.call_soon_threadsafe(
            progress_queue.put_nowait,
            progress_event,
        )
        return callback_kwargs

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "output_type": "latent" if is_sdxl else "pil",
        "callback_on_step_end": callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

    if mode == "txt2img":
        kwargs["width"] = width
        kwargs["height"] = height
    elif mode == "img2img":
        kwargs["image"] = ref_image
        kwargs["strength"] = denoise_strength
    elif mode == "inpaint":
        kwargs["image"] = ref_image
        kwargs["mask_image"] = mask_image
        kwargs["strength"] = denoise_strength

    error_message = None
    result_latents = None
    _oom_retried = False

    def run_inference(retry: bool = False):
        nonlocal error_message, result_latents, first_step_time, _oom_retried, pipe
        try:
            import gc
            result = pipe(**kwargs)
            result_latents = result.images[0]
            gc.collect()
        except RuntimeError as e:
            if "not enough GPU video memory" in str(e) and not retry and not _oom_retried:
                _oom_retried = True
                print(f"[OOM] VRAM exhausted, freeing cache and retrying once…")
                _empty_dml_cache()
                # Re-fetch pipeline after freeing VRAM — update outer `pipe` via nonlocal
                pipe = _get_pipeline(model_id, mode, is_sdxl, loras)
                _set_scheduler(pipe, scheduler)
                try:
                    import gc as _gc2
                    result = pipe(**kwargs)
                    result_latents = result.images[0]
                    _gc2.collect()
                    return
                except RuntimeError as e2:
                    if "not enough GPU video memory" in str(e2):
                        error_message = "OOM after retry: VRAM still exhausted. Try reducing resolution or closing other apps."
                        return
                    error_message = f"{type(e2).__name__}: {e2}"
                    import traceback
                    error_message += "\n" + "".join(traceback.format_exc())
                    return
                except Exception as e2:
                    error_message = f"{type(e2).__name__}: {e2}"
                    import traceback
                    error_message += "\n" + "".join(traceback.format_exc())
                    return
            error_message = f"{type(e).__name__}: {e}"
            import traceback
            error_message += "\n" + "".join(traceback.format_exc())
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            import traceback
            error_message += "\n" + "".join(traceback.format_exc())

    yield {"type": "started", "total_steps": steps}

    future = loop.run_in_executor(None, run_inference)

    while not future.done():
        try:
            event = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
            yield event
        except asyncio.TimeoutError:
            pass

    await future

    while not progress_queue.empty():
        try:
            event = progress_queue.get_nowait()
            yield event
        except asyncio.QueueEmpty:
            break

    if error_message:
        yield {"type": "error", "message": error_message}
        return

    total_time = int((time.time() - start_time) * 1000)
    if first_step_time == 0:
        first_step_time = total_time

    # SDXL: decode latents with CPU VAE to avoid DML OOM on decode.
    # output_type="latent" returns latents as [C, H, W] (no batch dim).
    # vae.decode() expects [B, C, H, W] so we add the batch dim.
    if is_sdxl:
        latents = result_latents
        vae_cpu = _get_cpu_vae_fp32()
        lat_cpu = latents.to("cpu").float().unsqueeze(0)
        with torch.no_grad():
            decoded = vae_cpu.decode(
                lat_cpu / vae_cpu.config.scaling_factor,
                return_dict=False
            )[0]
        output_image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        del latents, lat_cpu, decoded
        import gc
        gc.collect()
    else:
        output_image = result_latents

    yield {
        "type": "complete",
        "image": _image_to_b64(output_image, "JPEG"),
        "format": "jpeg",
        "seed": seed,
        "width": width,
        "height": height,
        "generation_time_ms": total_time,
        "first_step_time_ms": first_step_time,
        "total_steps": steps,
    }


def get_loaded_model() -> Optional[str]:
    if _pipelines:
        return next(iter(_pipelines)).split(":")[0]
    return None


def get_loaded_models() -> list[str]:
    return list({k.split(":")[0] for k in _pipelines.keys()})


def unload_model(model_id: str) -> bool:
    keys_to_remove = [k for k in _pipelines if k.startswith(f"{model_id}:")]
    with _pipeline_lock:
        for k in keys_to_remove:
            pipe = _pipelines.pop(k, None)
            if pipe is not None:
                del pipe
    if keys_to_remove:
        import gc
        gc.collect()
        return True
    return False


def unload_all_models():
    with _pipeline_lock:
        _pipelines.clear()
    import gc
    gc.collect()