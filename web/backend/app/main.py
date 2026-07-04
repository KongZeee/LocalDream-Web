import sys
sys.path.insert(0, r"C:\py_pkgs")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
from app.routes import generate, upscale, models, history, settings, loras, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Local Dream API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router)
app.include_router(upscale.router)
app.include_router(models.router)
app.include_router(history.router)
app.include_router(settings.router)
app.include_router(loras.router)
app.include_router(system.router)


@app.get("/")
async def root():
    return {"name": "Local Dream API", "version": "1.0.0"}