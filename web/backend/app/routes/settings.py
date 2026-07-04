from fastapi import APIRouter
from app.db import get_db

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def get_settings():
    db = await get_db()
    try:
        cursor = await db.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}
    finally:
        await db.close()


@router.put("")
async def update_settings(settings: dict):
    db = await get_db()
    try:
        for key, value in settings.items():
            await db.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
        await db.commit()
        return {"status": "saved"}
    finally:
        await db.close()