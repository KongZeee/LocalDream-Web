from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.services.upscaler import upscale_image

router = APIRouter(prefix="/api", tags=["upscale"])


@router.post("/upscale")
async def upscale(
    image: UploadFile = File(...),
    model_id: str = Form("RealESRGAN_x4plus"),
    tile_size: int = Form(512),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_data = await image.read()
    result = upscale_image(image_data, model_id, tile_size)
    return result