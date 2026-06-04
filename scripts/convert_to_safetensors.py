import argparse, mlx.core as mx
from efrlfn_mlx.convert import load_pth, convert_state_dict
ap=argparse.ArgumentParser(); ap.add_argument("--pth",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args(); w=convert_state_dict(load_pth(a.pth))
for v in w.values(): mx.eval(v)
mx.save_safetensors(a.out, w); print(f"saved {len(w)} -> {a.out}")
