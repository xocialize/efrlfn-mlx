"""Convert EfRLFN .pth to MLX. upsampler.0->upsampler; conv NCHW->NHWC; eca Conv1d (O,I,k)->(O,k,I)."""
from __future__ import annotations
import re
from typing import Dict
import mlx.core as mx

def _convert_key(k: str) -> str:
    return re.sub(r"^upsampler\.0\.", "upsampler.", k)

def convert_state_dict(sd: Dict) -> Dict[str, mx.array]:
    import numpy as np
    out = {}
    for k, v in sd.items():
        a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        nk = _convert_key(k)
        if a.ndim == 4:          # Conv2d weight (O,I,H,W) -> (O,H,W,I)
            a = a.transpose(0, 2, 3, 1)
        elif a.ndim == 3:        # eca Conv1d weight (O,I,k) -> (O,k,I)
            a = a.transpose(0, 2, 1)
        out[nk] = mx.array(a)
    return out

def load_pth(path: str) -> Dict:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck: ck = ck["state_dict"]
    if isinstance(ck, dict) and "params" in ck: ck = ck["params"]
    return ck
