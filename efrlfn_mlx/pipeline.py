"""EfRLFN inference: super-resolve an image."""
from __future__ import annotations
from pathlib import Path
import numpy as np, mlx.core as mx
from PIL import Image
from .config import EfRLFNConfig
from .model import EfRLFN

def load_model(weights: str | Path, config: EfRLFNConfig) -> EfRLFN:
    m = EfRLFN(config); m.load_weights(str(weights)); mx.eval(m.parameters()); return m

def upscale(model: EfRLFN, image_path: str | Path) -> np.ndarray:
    img = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0
    y = model(mx.array(img[None])); mx.eval(y)
    return (np.clip(np.array(y)[0], 0, 1) * 255).astype(np.uint8)

def upscale_to_file(model, image_path, out_path) -> str:
    Image.fromarray(upscale(model, image_path)).save(out_path); return str(out_path)
