from fastapi import APIRouter
from app.services.lora_manager import scan_loras

router = APIRouter(prefix="/api/loras", tags=["loras"])


@router.get("")
async def list_loras():
    """Return all LoRA files found in data/models/Lora/."""
    return {"loras": scan_loras()}


@router.get("/names")
async def lora_names():
    """Lightweight endpoint: just returns LoRA IDs + names for the selector dropdown."""
    loras = scan_loras()
    return [{"id": l["id"], "name": l["name"]} for l in loras]
