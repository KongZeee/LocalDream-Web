from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.services.history import get_history, get_history_image, delete_history

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("")
async def list_history(page: int = 1, page_size: int = 20):
    return await get_history(page, page_size)


@router.get("/{history_id}/image")
async def serve_image(history_id: int):
    data = await get_history_image(history_id, thumb=False)
    if data is None:
        raise HTTPException(404, "Image not found")
    return Response(content=data, media_type="image/jpeg")


@router.get("/{history_id}/thumbnail")
async def serve_thumbnail(history_id: int):
    data = await get_history_image(history_id, thumb=True)
    if data is None:
        raise HTTPException(404, "Thumbnail not found")
    return Response(content=data, media_type="image/jpeg")


@router.delete("/{history_id}")
async def remove_history(history_id: int):
    await delete_history(history_id)
    return {"status": "deleted"}