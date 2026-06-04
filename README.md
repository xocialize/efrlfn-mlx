# efrlfn-mlx

Apple MLX port of **[EfRLFN](https://github.com/EvgeneyBogatyrev/EfRLFN)** — *Efficient Real-time
Super-Resolution* (ICLR 2026). Lightweight x2/x4 image upscaling on Apple Silicon.

Faithful NHWC port (all stock MLX ops): `conv → 6× ERLFB → conv → PixelShuffle`, where each
ERLFB is three 3×3 convs (tanh) + residual + 1×1 conv + ECA channel attention.

## Status
- [x] Architecture port (NHWC) + weight conversion (exact 60-param match)
- [x] PT-vs-MLX parity on a real image: **x2 1.07e-6 · x4 1.19e-6**
- [x] e2e SR verified — 270×480 → 1080×1920 in ~0.06s (realtime)
- [x] Published `mlx-community/EfRLFN-x{2,4}`

## Usage
```python
from efrlfn_mlx import EfRLFNConfig
from efrlfn_mlx.pipeline import load_model, upscale_to_file
m = load_model("EfRLFN-x4.safetensors", EfRLFNConfig.x4())
upscale_to_file(m, "lr.png", "sr.png")
```
License: MIT (derived from EvgeneyBogatyrev/EfRLFN, MIT). See `NOTICE`.
