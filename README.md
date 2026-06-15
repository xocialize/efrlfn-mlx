# efrlfn-mlx

Apple MLX port of **[EfRLFN](https://github.com/EvgeneyBogatyrev/EfRLFN)** — *Efficient
Real-time Super-Resolution* (ICLR 2026). Lightweight x2/x4 image upscaling on Apple Silicon.

Faithful NHWC port using stock MLX ops: `conv → 6× ERLFB → conv → PixelShuffle`. Each
ERLFB (Efficient Residual Local Feature Block) is three 3×3 convs (tanh) with an internal
residual add, then a 1×1 conv and an ECA channel-attention gate. Feature width is 52
channels (`feature_channels=52`).

## Status

- [x] Architecture port (NHWC) + weight conversion (exact 60-param match)
- [x] PT-vs-MLX parity on a real image: **x2 1.07e-6 · x4 1.19e-6**
- [x] e2e SR verified — 270×480 → 1080×1920 in ~0.06 s (realtime) for x4
- [x] Published `mlx-community/EfRLFN-x{2,4}`

## Install

```bash
pip install -e .                # runtime: mlx, numpy, safetensors, huggingface_hub, pillow
pip install -e ".[parity]"      # + torch, for PT-vs-MLX parity tests
pip install -e ".[dev]"         # parity + pytest + gdown (pulls upstream PT weights)
```

## Usage

```python
from efrlfn_mlx import EfRLFNConfig
from efrlfn_mlx.pipeline import load_model, upscale_to_file

# Each published variant is a directory containing model.safetensors + config.json.
m = load_model("EfRLFN-x4/model.safetensors", EfRLFNConfig.x4())
upscale_to_file(m, "lr.png", "sr.png")     # writes the upscaled image, returns the path
```

Pipeline API (`efrlfn_mlx/pipeline.py`):

- `load_model(weights, config) -> EfRLFN` — build the model and load safetensors weights.
- `upscale(model, image_path) -> np.ndarray` — return the upscaled image as an array.
- `upscale_to_file(model, image_path, out_path) -> str` — upscale and write to disk.

Configs: `EfRLFNConfig.x2()` and `EfRLFNConfig.x4()` (defaults `in_channels=3`,
`out_channels=3`, `feature_channels=52`, `num_blocks=6`).

## Architecture

```
EfRLFN
  conv_1            3×3 conv, 3 -> 52
  block_1 … block_6 ERLFB:
                      tanh(3×3) → tanh(3×3) → tanh(3×3) → (+ input residual)
                      → 1×1 conv → ECA channel attention
  conv_2            3×3 conv, 52 -> 52  (applied to blocks output + conv_1 feature)
  upsampler         3×3 conv, 52 -> 3·r²
  PixelShuffle      (C, r, r) channel grouping, scale r ∈ {2, 4}
```

ECA (Efficient Channel Attention): global average pool → `Conv1d` over the channel axis →
sigmoid gate. NHWC throughout (channel ops on the last axis); PixelShuffle follows
PyTorch's `(C, r, r)` grouping so weights are byte-compatible with upstream.

## Weight conversion

```bash
python scripts/convert_to_safetensors.py      # PT checkpoint -> MLX safetensors
```

Upstream `.pth` checkpoints are fetched from Google Drive (IDs pinned in
`efrlfn_mlx/config.py::GDRIVE_IDS`); `convert.py` premaps the 60 parameters into the
isomorphic MLX module tree. Published artifacts live under `dist/EfRLFN-x{2,4}/`
(`model.safetensors` + `config.json` + `NOTICE` + model card).

## Tests

```bash
pytest tests/smoke                 # forward-shape smoke (no weights)
pytest tests/parity                # PT-vs-MLX parity (needs [parity] extras + weights)
```

## Layout

```
efrlfn_mlx/
  __init__.py        # exports EfRLFNConfig, EfRLFN
  config.py          # EfRLFNConfig (.x2/.x4) + GDRIVE_IDS
  model.py           # EfRLFN, ERLFB, ECABlock, pixel_shuffle
  convert.py         # PT -> MLX weight premap
  pipeline.py        # load_model / upscale / upscale_to_file
scripts/convert_to_safetensors.py
dist/EfRLFN-x2/  dist/EfRLFN-x4/      # published MLX artifacts
tests/{smoke,parity}/
```

## License

MIT (derived from [EvgeneyBogatyrev/EfRLFN](https://github.com/EvgeneyBogatyrev/EfRLFN), MIT).
See `NOTICE`.
