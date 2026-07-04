from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.model_manager import get_models, download_model, delete_model, preload_model
from app.services.generator import unload_model, unload_all_models, get_loaded_models, get_loaded_model

router = APIRouter(prefix="/api/models", tags=["models"])


class DownloadRequest(BaseModel):
    model_id: str
    model_type: str = "sd15"


class UnloadRequest(BaseModel):
    model_id: str


@router.get("")
async def list_models():
    return await get_models()


@router.get("/loaded")
async def loaded_models():
    return {
        "loaded_models": get_loaded_models(),
        "current_model": get_loaded_model(),
    }


@router.post("/unload")
async def unload(req: UnloadRequest):
    success = unload_model(req.model_id)
    if success:
        return {"status": "unloaded", "model_id": req.model_id}
    raise HTTPException(404, f"Model {req.model_id} not loaded")


@router.post("/unload-all")
async def unload_all():
    unload_all_models()
    return {"status": "all_unloaded"}


@router.post("/{model_id:path}/preload")
async def preload(model_id: str):
    """Pre-load a model into GPU memory and run a quick warmup inference pass.

    After this call the model is ready for fast generation. The model remains
    cached until explicitly unloaded via /api/models/unload or the server restarts.
    """
    try:
        await preload_model(model_id)
        return {"status": "loaded", "model_id": model_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/download")
async def download(req: DownloadRequest):
    try:
        await download_model(req.model_id, req.model_type)
        return {"status": "downloading", "model_id": req.model_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/{model_id:path}")
async def remove_model(model_id: str):
    try:
        await delete_model(model_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))