import os
import json
import asyncio
from pathlib import Path
from typing import Optional

from app.config import MODELS_DIR
from app.db import get_db


def detect_sdxl(model_id: str, local_dir: str | None = None) -> bool:
    """Detect whether a model is SDXL-based.

    Primary heuristic: name contains "xl" (covers e.g. "SDXL-1.0", "anima-xl").
    Fallback: read ``_class_name`` from the local ``model_index.json`` — this
    catches SDXL-architecture models whose name does not contain "xl"
    (Anima 2.9B, Pony, Illustrious, ...).
    """
    if "xl" in model_id.lower():
        return True
    if local_dir is None:
        local_dir = str(Path(MODELS_DIR) / model_id.replace("/", "--"))
    index_file = Path(local_dir) / "model_index.json"
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
                # regardless of what the cache row says.
                "status": "ready" if local else row["status"],
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
        if entry.is_dir() and entry.name.startswith("."):
            continue
        if entry.name in _SKIP_DIRS:
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
    local_dir = Path(MODELS_DIR) / model_id.replace("/", "--")
    if local_dir.exists():
        shutil.rmtree(local_dir)

    db = await get_db()
    try:
        await db.execute("DELETE FROM model_cache WHERE model_id=?", (model_id,))
        await db.commit()
    finally:
        await db.close()


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