from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio

from app.services.generator import generate_image, get_loaded_model
from app.services.history import add_history

router = APIRouter(prefix="/api", tags=["generate"])


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model_id: str
    mode: str = "txt2img"
    steps: int = 20
    cfg: float = 7.5
    seed: int = 42
    width: int = 512
    height: int = 512
    scheduler: str = "dpm"
    denoise_strength: float = 0.6
    image: Optional[str] = None
    mask: Optional[str] = None
    show_preview: bool = True
    preview_stride: int = 1
    speed_mode: str = "balanced"
    loras: Optional[list[dict]] = None  # [{"id": "...", "name": "...", "path": "...", "filename": "...", "weight": 0.7}, ...]


@router.post("/generate")
async def generate(req: GenerateRequest):
    async def event_stream():
        try:
            async for event in generate_image(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                model_id=req.model_id,
                mode=req.mode,
                steps=req.steps,
                cfg=req.cfg,
                seed=req.seed,
                width=req.width,
                height=req.height,
                scheduler=req.scheduler,
                denoise_strength=req.denoise_strength,
                image_b64=req.image,
                mask_b64=req.mask,
                show_preview=req.show_preview,
                preview_stride=req.preview_stride,
                speed_mode=req.speed_mode,
                loras=req.loras,
            ):
                if event["type"] == "complete":
                    try:
                        await add_history(
                            prompt=req.prompt,
                            negative_prompt=req.negative_prompt,
                            seed=event["seed"],
                            steps=req.steps,
                            cfg=req.cfg,
                            width=req.width,
                            height=req.height,
                            model_id=req.model_id,
                            scheduler=req.scheduler,
                            mode=req.mode,
                            image_b64=event["image"],
                            image_format=event.get("format", "jpeg"),
                        )
                    except Exception as e:
                        import traceback
                        print(f"[History] add_history failed: {type(e).__name__}: {e}")
                        traceback.print_exc()

                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def health():
    from app.config import MODELS_DIR
    return {
        "status": "ok",
        "gpu_available": True,
        "gpu_name": "AMD GPU (DirectML)",
        "loaded_model": get_loaded_model(),
        "models_dir": MODELS_DIR,
    }