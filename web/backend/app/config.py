import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.environ.get("LOCAL_DREAM_DATA_DIR", str(BASE_DIR / "data"))
MODELS_DIR = os.environ.get("LOCAL_DREAM_MODELS_DIR", str(Path(DATA_DIR) / "models"))
OUTPUT_DIR = os.environ.get("LOCAL_DREAM_OUTPUT_DIR", str(Path(DATA_DIR) / "outputs"))
DB_PATH = os.environ.get("LOCAL_DREAM_DB_PATH", str(Path(DATA_DIR) / "localdream.db"))

HOST = os.environ.get("LOCAL_DREAM_HOST", "0.0.0.0")
PORT = int(os.environ.get("LOCAL_DREAM_PORT", "8081"))

DEFAULT_STEPS = 15
DEFAULT_CFG = 7.0
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_SCHEDULER = "dpm"
DEFAULT_NEGATIVE_PROMPT = "ugly, blurry, low quality, bad anatomy"
DEFAULT_SPEED_MODE = "balanced"

for d in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)