import aiosqlite
import json
from app.config import DB_PATH

async def get_db():
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db

async def init_db():
    db = await get_db()
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS history_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            negative_prompt TEXT DEFAULT '',
            seed INTEGER NOT NULL,
            steps INTEGER NOT NULL DEFAULT 20,
            cfg REAL NOT NULL DEFAULT 7.5,
            width INTEGER NOT NULL DEFAULT 512,
            height INTEGER NOT NULL DEFAULT 512,
            model_id TEXT NOT NULL,
            scheduler TEXT NOT NULL DEFAULT 'dpm',
            mode TEXT NOT NULL DEFAULT 'txt2img',
            image_path TEXT NOT NULL,
            thumbnail_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_history_created
            ON history_items(created_at DESC);

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_cache (
            model_id TEXT PRIMARY KEY,
            model_type TEXT NOT NULL,
            local_path TEXT NOT NULL,
            size_mb INTEGER DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            preview_path TEXT,
            downloaded_at DATETIME
        );

        INSERT OR IGNORE INTO settings (key, value) VALUES
            ('default_steps', '15'),
            ('default_cfg', '7.0'),
            ('default_width', '512'),
            ('default_height', '512'),
            ('default_scheduler', 'dpm'),
            ('default_negative_prompt', 'ugly, blurry, low quality, bad anatomy');
    """)
    await db.commit()
    await db.close()