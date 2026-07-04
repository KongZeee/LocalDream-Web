import io
import base64
import time
import threading
from pathlib import Path
from typing import Optional

import torch
import torch_directml
from PIL import Image

from app.config import MODELS_DIR

_upscaler_pipeline = None
_upscaler_lock = threading.Lock()


def _get_upscale_device():
    try:
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def _load_esrgan_pipeline(model_path: str, device):
    global _upscaler_pipeline

    if _upscaler_pipeline is not None:
        return _upscaler_pipeline

    with _upscaler_lock:
        if _upscaler_pipeline is not None:
            return _upscaler_pipeline
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=False,
                device=device,
            )
            _upscaler_pipeline = upsampler
            return upsampler
        except ImportError:
            return None


def upscale_image(
    image_data: bytes,
    model_id: str,
    tile_size: int = 512,
) -> dict:
    start_time = time.time()

    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    orig_w, orig_h = img.size

    device = _get_upscale_device()

    realesrgan = _load_esrgan_pipeline(model_id, device)

    if realesrgan is not None:
        import numpy as np
        img_np = np.array(img)
        output_np, _ = realesrgan.enhance(img_np, outscale=4)
        output = Image.fromarray(output_np)
    else:
        new_w = orig_w * 4
        new_h = orig_h * 4
        output = img.resize((new_w, new_h), Image.LANCZOS)

    duration = int((time.time() - start_time) * 1000)

    buf = io.BytesIO()
    output.save(buf, format="JPEG", quality=95)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "image": image_b64,
        "format": "jpeg",
        "width": output.width,
        "height": output.height,
        "duration_ms": duration,
    }