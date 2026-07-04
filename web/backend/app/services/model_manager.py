import os
import json
from pathlib import Path
from typing import Optional

from app.config import MODELS_DIR
from app.db import get_db


async def get_models() -> dict:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM model_cache")
        rows = await cursor.fetchall()
        models = []
        for row in rows:
            models.append({
                "id": row["model_id"],
                "name": row["model_id"].split("/")[-1] if "/" in row["model_id"] else row["model_id"],
                "type": row["model_type"],
                "status": row["status"],
                "size_mb": row["size_mb"],
                "preview_url": row["preview_path"] or "",
            })

        if not models:
            models = await _scan_local_models()

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

    _SKIP_DIRS = {".cache", "anima", "__pycache__"}

    for entry in os.scandir(MODELS_DIR):
        if entry.is_dir() and entry.name.startswith("."):
            continue
        if entry.name in _SKIP_DIRS:
            continue

        config_file = Path(entry.path) / "model_index.json"
        if not config_file.exists():
            continue

        model_id = entry.name.replace("--", "/")
        is_sdxl = "xl" in entry.name.lower()
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


async def download_model(model_id: str, model_type: str):
    import asyncio
    from huggingface_hub import snapshot_download

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

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        ),
    )

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
    """Load the pipeline into GPU memory and warm it up, without generating.

    The model is loaded lazily on the first call and kept cached.
    Subsequent calls return immediately since the pipeline is already in memory.
    """
    import asyncio

    def _do():
        from app.services.generator import _get_pipeline, _warmed_up
        is_sdxl = "xl" in model_id.lower()
        # Trigger pipeline load (cached on subsequent calls).
        pipe = _get_pipeline(model_id, "txt2img", is_sdxl, loras=None)
        # Mark as warmed so real generation skips warmup.
        _warmed_up.add(model_id)
        return pipe is not None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do)