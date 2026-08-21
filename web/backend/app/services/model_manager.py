import os
import json
import asyncio
from pathlib import Path
from typing import Optional

from app.config import MODELS_DIR
from app.db import get_db

# Single-file checkpoint extensions recognized by the scanner.
CHECKPOINT_EXTS = (".safetensors", ".ckpt")


def peek_checkpoint_type(path) -> str:
    """Detect the architecture of a single-file checkpoint WITHOUT loading it.

    For ``.safetensors`` only the header (tensor names) is read, which is
    nearly instant. SDXL is identified by its adaln single blocks
    (``label_emb``) or its second text encoder (``conditioner.embedders.1``);
    SD 1.x/2.x has neither. ``.ckpt`` (pickle) cannot be peeked cheaply —
    fall back to the filename heuristic.
    """
    p = Path(path)
    if p.suffix.lower() == ".safetensors":
        try:
            from safetensors import safe_open
            with safe_open(str(p), framework="numpy") as f:
                keyset = " ".join(f.keys())
            if "label_emb" in keyset or "conditioner.embedders.1" in keyset:
                return "sdxl"
            return "sd15"
        except Exception:
            pass
    return "sdxl" if "xl" in p.stem.lower() else "sd15"


def detect_sdxl(model_id: str, local_dir: str | None = None) -> bool:
    """Detect whether a model is SDXL-based.

    Primary heuristic: name contains "xl" (covers e.g. "SDXL-1.0", "anima-xl").
    Local directory: read ``_class_name`` from ``model_index.json`` — this
    catches SDXL-architecture models whose name does not contain "xl"
    (Anima 2.9B, Pony, Illustrious, ...).
    Local file (single-file checkpoint): peek the safetensors header keys.
    """
    if "xl" in model_id.lower():
        return True
    if local_dir is None:
        local_dir = str(Path(MODELS_DIR) / model_id.replace("/", "--"))
    local = Path(local_dir)
    if local.is_file():
        return peek_checkpoint_type(local) == "sdxl"
    index_file = local / "model_index.json"
    if not index_file.is_file():
        return False
    try:
        with open(index_file, encoding="utf-8") as f:
            class_name = json.load(f).get("_class_name", "")
        return "XL" in class_name
    except Exception:
        return False


async def get_models() -> dict:
    # Merge local scan (ready models on disk) with model_cache rows (download
    # status). Previously only one source was shown: once anything had been
    # downloaded, manually placed models disappeared from the list.
    scanned = await _scan_local_models()
    scanned_by_id = {m["id"]: m for m in scanned}

    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM model_cache")
        rows = await cursor.fetchall()

        models = []
        for row in rows:
            model_id = row["model_id"]
            local = scanned_by_id.get(model_id)
            models.append({
                "id": model_id,
                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                "type": local["type"] if local else row["model_type"],
                # If the files are on disk and valid, the model is ready
                # regardless of what the cache row says (single-file entries
                # keep their own status).
                "status": local["status"] if local else row["status"],
                "size_mb": local["size_mb"] if local else row["size_mb"],
                "preview_url": row["preview_path"] or "",
            })

        cached_ids = {m["id"] for m in models}
        models.extend(m for m in scanned if m["id"] not in cached_ids)

        cursor = await db.execute("SELECT value FROM settings WHERE key='default_model'")
        default_rows = await cursor.fetchall()
        default_model = default_rows[0]["value"] if default_rows else (models[0]["id"] if models else "")

        return {"models": models, "default_model": default_model}
    finally:
        await db.close()


async def _scan_local_models() -> list:
    models = []
    if not os.path.exists(MODELS_DIR):
        return models

    _SKIP_DIRS = {".cache", "__pycache__"}

    for entry in os.scandir(MODELS_DIR):
        if entry.name in _SKIP_DIRS:
            continue

        if entry.is_file() and entry.name.lower().endswith(CHECKPOINT_EXTS):
            # Single-file checkpoint. Hidden once a converted Diffusers
            # directory for the same stem exists, or while a conversion is
            # running (the model_cache row for the stem shows the progress).
            stem = Path(entry.name).stem
            if (Path(MODELS_DIR) / stem / "model_index.json").exists():
                continue
            if stem in _active_conversions:
                continue
            model_type = peek_checkpoint_type(entry.path)
            try:
                size_mb = entry.stat().st_size // (1024 * 1024)
            except Exception:
                size_mb = 0
            models.append({
                "id": entry.name,
                "name": entry.name,
                "type": model_type,
                "status": "single_file",
                "size_mb": size_mb,
                "preview_url": "",
            })
            continue

        if not entry.is_dir() or entry.name.startswith("."):
            continue

        config_file = Path(entry.path) / "model_index.json"
        if not config_file.exists():
            continue

        model_id = entry.name.replace("--", "/")
        is_sdxl = detect_sdxl(model_id, entry.path)
        model_type = "sdxl" if is_sdxl else "sd15"

        try:
            total_size = sum(
                f.stat().st_size for f in Path(entry.path).rglob("*") if f.is_file()
            )
            size_mb = total_size // (1024 * 1024)
        except Exception:
            total_size = 0
            size_mb = 0

        models.append({
            "id": model_id,
            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
            "type": model_type,
            "status": "ready",
            "size_mb": size_mb,
            "preview_url": "",
        })

    return models


# Model IDs currently being downloaded in the background.
_active_downloads: set[str] = set()


async def download_model(model_id: str, model_type: str):
    """Mark the model as ``downloading`` and start the snapshot download in
    the background. Returns immediately — poll ``GET /api/models`` for status
    (the old implementation blocked the HTTP request for the whole download,
    which could hang for tens of minutes on SDXL repos).
    """
    local_dir = Path(MODELS_DIR) / model_id.replace("/", "--")
    local_dir.mkdir(parents=True, exist_ok=True)

    db = await get_db()
    try:
        await db.execute(
            "INSERT OR REPLACE INTO model_cache (model_id, model_type, local_path, status) VALUES (?, ?, ?, ?)",
            (model_id, model_type, str(local_dir), "downloading"),
        )
        await db.commit()
    finally:
        await db.close()

    if model_id in _active_downloads:
        return
    _active_downloads.add(model_id)
    asyncio.create_task(_run_download(model_id, local_dir))


async def _run_download(model_id: str, local_dir: Path):
    from huggingface_hub import snapshot_download

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            ),
        )
    except Exception as e:
        print(f"[Download] {model_id} failed: {e}")
        db = await get_db()
        try:
            await db.execute("UPDATE model_cache SET status='error' WHERE model_id=?", (model_id,))
            await db.commit()
        finally:
            await db.close()
        return
    finally:
        _active_downloads.discard(model_id)

    db = await get_db()
    try:
        total_size = sum(
            f.stat().st_size for f in local_dir.rglob("*") if f.is_file()
        )
        await db.execute(
            "UPDATE model_cache SET status='ready', size_mb=? WHERE model_id=?",
            (total_size // (1024 * 1024), model_id),
        )
        await db.commit()
    finally:
        await db.close()


async def delete_model(model_id: str):
    import shutil
    local = Path(MODELS_DIR) / model_id.replace("/", "--")
    if local.is_dir():
        shutil.rmtree(local)
    elif local.is_file():
        # Single-file checkpoint (and its optional cache row).
        local.unlink()

    db = await get_db()
    try:
        await db.execute("DELETE FROM model_cache WHERE model_id=?", (model_id,))
        await db.commit()
    finally:
        await db.close()


# Model stems currently being converted in the background.
_active_conversions: set[str] = set()


async def convert_model(model_id: str):
    """Convert a single-file checkpoint into a Diffusers directory layout.

    Runs in the background via ``DiffusionPipeline.from_single_file`` +
    ``save_pretrained`` (one-time cost; subsequent loads are much faster than
    re-parsing the checkpoint). The source file is KEPT — deleting it is up
    to the user. Returns immediately; poll ``GET /api/models`` for status.
    """
    src = Path(MODELS_DIR) / model_id.replace("/", "--")
    if not src.is_file():
        raise FileNotFoundError(f"single-file checkpoint not found: {model_id}")

    stem = src.stem
    if stem in _active_conversions:
        return

    model_type = peek_checkpoint_type(src)
    out_dir = Path(MODELS_DIR) / stem

    db = await get_db()
    try:
        await db.execute(
            "INSERT OR REPLACE INTO model_cache (model_id, model_type, local_path, status) VALUES (?, ?, ?, ?)",
            (stem, model_type, str(out_dir), "converting"),
        )
        await db.commit()
    finally:
        await db.close()

    _active_conversions.add(stem)
    asyncio.create_task(_run_conversion(str(src), out_dir, stem))


async def _run_conversion(src: str, out_dir: Path, stem: str):
    loop = asyncio.get_running_loop()

    def _do():
        import torch
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_single_file(src, torch_dtype=torch.float16)
        pipe.save_pretrained(str(out_dir))
        del pipe

    try:
        await loop.run_in_executor(None, _do)
    except Exception as e:
        import traceback
        print(f"[Convert] {stem} failed: {e}")
        traceback.print_exc()
        _active_conversions.discard(stem)
        db = await get_db()
        try:
            await db.execute("UPDATE model_cache SET status='error' WHERE model_id=?", (stem,))
            await db.commit()
        finally:
            await db.close()
        return
    finally:
        _active_conversions.discard(stem)

    db = await get_db()
    try:
        total_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
        await db.execute(
            "UPDATE model_cache SET status='ready', size_mb=? WHERE model_id=?",
            (total_size // (1024 * 1024), stem),
        )
        await db.commit()
    finally:
        await db.close()
    print(f"[Convert] {stem} -> Diffusers format OK ({out_dir})")


async def preload_model(model_id: str):
    """Load the pipeline into GPU memory and run a real warmup inference.

    ``_warmup_model`` loads the pipeline AND runs a tiny 64x64 inference pass,
    so shader compilation and memory-pool allocation are paid here instead of
    on the user's first generation. (Previously preload only loaded the
    pipeline and marked it warmed, deferring the real cost to the first run.)
    """
    def _do():
        from app.services.generator import _warmup_model
        _warmup_model(model_id, loras=None)
        return True

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do)