"""
LoRA lifecycle management for pipelines.

LoRAs are loaded as PEFT adapters into cached pipelines. The cache key
includes both model_id and the set of active LoRAs so that different LoRA
combinations get their own pipeline entry (avoids stale adapter contamination).

Example cache key: "aom3a1b:txt2img:<1990s,0.6>"
"""

import threading
from pathlib import Path

from app.config import MODELS_DIR

# ── types ────────────────────────────────────────────────────────────────────

LoraSpec = dict  # {"id": str, "name": str, "path": str, "filename": str, "weight": float}

# ── registry ─────────────────────────────────────────────────────────────

# Maps cache_key -> set of adapter names currently on that pipeline.
_pipeline_loras: dict[str, set[str]] = {}
_loras_lock = threading.Lock()


def _make_key(model_id: str, mode: str, loras: list | None) -> str:
    """Stable cache key that includes the LoRA combo."""
    if not loras:
        return f"{model_id}:{mode}"
    parts = ",".join(f"{s['id']}:{s['weight']}" for s in sorted(loras, key=lambda s: s["id"]))
    return f"{model_id}:{mode}:<{parts}>"


# ── apply / clear ─────────────────────────────────────────────────────────

def apply_loras(pipe, loras: list | None) -> None:
    """Load (or swap) LoRA adapters onto an already-loaded pipeline."""
    key = getattr(pipe, "_lora_cache_key", None)
    print(f"[LoRA] apply_loras: key={key}, loras={[s['id'] for s in loras] if loras else None}")
    if key is None:
        print("[LoRA] apply_loras: no _lora_cache_key, skipping")
        return  # No tracking tag; nothing to do.

    with _loras_lock:
        current_names = set(_pipeline_loras.get(key, set()))
    print(f"[LoRA] current_names={current_names}")

    if not loras:
        print("[LoRA] no loras requested, clearing")
        # Clear all adapters.
        if hasattr(pipe, "delete_adapters"):
            for name in list(current_names):
                try:
                    pipe.delete_adapters(name)
                except Exception:
                    pass
        with _loras_lock:
            _pipeline_loras[key] = set()
        return

    desired_names = {spec["id"] for spec in loras}
    print(f"[LoRA] desired_names={desired_names}")

    # Remove adapters that are no longer wanted.
    if hasattr(pipe, "delete_adapters"):
        for name in current_names - desired_names:
            try:
                pipe.delete_adapters(name)
            except Exception:
                pass

    # Load any adapters not yet in the pipeline.
    for spec in loras:
        name = spec["id"]
        weight = float(spec.get("weight", 0.7))
        if name not in current_names:
            print(f"[LoRA] load {spec['id']} (weight={weight})")
            try:
                pipe.load_lora_weights(
                    spec["path"],
                    adapter_name=name,
                )
            except TypeError:
                try:
                    pipe.load_lora_weights(
                        spec["path"],
                        weight_name=spec.get("filename", ""),
                    )
                except Exception as e2:
                    print(f"[LoRA] load failed for {spec['id']}: {e2}")
            except Exception as e:
                print(f"[LoRA] load failed for {spec['id']}: {e}")
        else:
            print(f"[LoRA] {name} already loaded, skipping")

    # Activate the desired adapter set with weights.
    weights = [float(s.get("weight", 0.7)) for s in loras]
    print(f"[LoRA] set_adapters: {[s['id'] for s in loras]} weights={weights}")
    pipe.set_adapters([s["id"] for s in loras], adapter_weights=weights)

    with _loras_lock:
        _pipeline_loras[key] = desired_names
    print(f"[LoRA] done, _pipeline_loras[{key}]={desired_names}")

def mark_pipeline(pipe, model_id: str, mode: str, loras: list | None) -> None:
    """Tag a newly-loaded pipeline with its LoRA-aware cache key."""
    key = _make_key(model_id, mode, loras)
    pipe._lora_cache_key = key
    with _loras_lock:
        _pipeline_loras.setdefault(key, set())
    print(f"[LoRA] Pipeline tagged cache_key={key}")


def clear_all_loras(pipe) -> None:
    """Remove all LoRA adapters from a pipeline."""
    key = getattr(pipe, "_lora_cache_key", None)
    if not key or not hasattr(pipe, "delete_adapters"):
        return
    with _loras_lock:
        names = list(_pipeline_loras.get(key, set()))
    for name in names:
        try:
            pipe.delete_adapters(name)
        except Exception:
            pass
    with _loras_lock:
        _pipeline_loras[key] = set()


# ── scan disk ─────────────────────────────────────────────────────────────

def scan_loras() -> list[dict]:
    """Return metadata for every LoRA file in data/models/Lora/."""
    lora_dir = Path(MODELS_DIR) / "Lora"
    if not lora_dir.exists():
        return []
    result = []
    for f in sorted(lora_dir.iterdir()):
        if f.suffix.lower() in (".safetensors", ".pt", ".ckpt"):
            size_mb = f.stat().st_size / (1024 * 1024)
            result.append({
                "id": f.stem,
                "name": f.stem,
                "filename": f.name,
                "path": str(f),
                "size_mb": round(size_mb, 1),
            })
    return result
