import os
import io
import base64
from pathlib import Path
from typing import Optional

from app.config import OUTPUT_DIR
from app.db import get_db


async def add_history(
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    model_id: str,
    scheduler: str,
    mode: str,
    image_b64: str,
    image_format: str = "jpeg",
) -> int:
    print(f"[History] add_history called: model={model_id}, mode={mode}, size={len(image_b64) if image_b64 else 0}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    db = await get_db()
    print(f"[History] DB connected")
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) as c FROM history_items"
        )
        row = await cursor.fetchone()
        count = row["c"]
        image_id = count + 1

        image_path = str(Path(OUTPUT_DIR) / f"{image_id}.{image_format}")
        image_data = base64.b64decode(image_b64)
        with open(image_path, "wb") as f:
            f.write(image_data)

        from PIL import Image
        thumb_path = str(Path(OUTPUT_DIR) / f"{image_id}_thumb.jpg")
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail((256, 256))
        img.save(thumb_path, "JPEG", quality=80)

        print(f"[History] Inserting: prompt={prompt[:50]}, path={image_path}")
        await db.execute(
            """INSERT INTO history_items
               (prompt, negative_prompt, seed, steps, cfg, width, height,
                model_id, scheduler, mode, image_path, thumbnail_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prompt, negative_prompt, seed, steps, cfg, width, height,
             model_id, scheduler, mode, image_path, thumb_path),
        )
        await db.commit()
        print(f"[History] Insert committed, id={image_id}")
        return image_id
    finally:
        await db.close()


async def get_history(page: int = 1, page_size: int = 20) -> dict:
    db = await get_db()
    try:
        offset = (page - 1) * page_size

        cursor = await db.execute("SELECT COUNT(*) as c FROM history_items")
        total = (await cursor.fetchone())["c"]

        cursor = await db.execute(
            "SELECT * FROM history_items ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (page_size, offset),
        )
        rows = await cursor.fetchall()

        items = []
        for row in rows:
            items.append({
                "id": row["id"],
                "prompt": row["prompt"],
                "negative_prompt": row["negative_prompt"],
                "seed": row["seed"],
                "steps": row["steps"],
                "cfg": row["cfg"],
                "width": row["width"],
                "height": row["height"],
                "model_id": row["model_id"],
                "scheduler": row["scheduler"],
                "mode": row["mode"],
                "image_url": f"/api/history/{row['id']}/image",
                "thumbnail_url": f"/api/history/{row['id']}/thumbnail",
                "created_at": row["created_at"],
            })

        return {"items": items, "total": total, "page": page, "page_size": page_size}
    finally:
        await db.close()


async def get_history_image(history_id: int, thumb: bool = False) -> Optional[bytes]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT image_path, thumbnail_path FROM history_items WHERE id=?",
            (history_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        path = row["thumbnail_path"] if thumb else row["image_path"]
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None
    finally:
        await db.close()


async def delete_history(history_id: int):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT image_path, thumbnail_path FROM history_items WHERE id=?",
            (history_id,),
        )
        row = await cursor.fetchone()
        if row:
            for p in [row["image_path"], row["thumbnail_path"]]:
                if p and os.path.exists(p):
                    os.remove(p)

        await db.execute("DELETE FROM history_items WHERE id=?", (history_id,))
        await db.commit()
    finally:
        await db.close()