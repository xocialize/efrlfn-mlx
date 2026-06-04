"""PT vs MLX parity for EfRLFN x2/x4 on a real image (in-distribution). Gate <1e-3."""
import sys
from pathlib import Path
import numpy as np, pytest, mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "refs/EfRLFN"
IMG = REF / "images/input.png"
pytestmark = pytest.mark.skipif(not REF.exists() or not IMG.exists(),
                                reason="needs refs/EfRLFN (dev only)")

def _upstream(scale):
    import torch
    sys.path.insert(0, str(REF))            # local `code` package shadows stdlib
    for m in [k for k in sys.modules if k == "code" or k.startswith("code.")]:
        del sys.modules[m]                  # evict stdlib `code` so the local package resolves
    from code.model import EfRLFN as PT
    net = PT(upscale=scale); net.eval()
    sd = torch.load(str(ROOT / f"weights/EfRLFN-x{scale}.pth"), map_location="cpu", weights_only=False)
    net.load_state_dict(sd, strict=True)
    return net

@pytest.mark.parametrize("scale,factory", [(2, "x2"), (4, "x4")])
def test_parity(scale, factory):
    if not (ROOT / f"weights/EfRLFN-x{scale}.pth").exists():
        pytest.skip("weights missing")
    import torch
    from PIL import Image
    from mlx.utils import tree_unflatten
    from efrlfn_mlx import EfRLFN, EfRLFNConfig
    from efrlfn_mlx.convert import load_pth, convert_state_dict

    pt = _upstream(scale)
    img = (np.asarray(Image.open(IMG).convert("RGB"), np.float32)[:96, :96] / 255.0)[None]
    pt_out = pt(torch.from_numpy(img.transpose(0, 3, 1, 2))).detach().numpy()

    m = EfRLFN(getattr(EfRLFNConfig, factory)())
    m.update(tree_unflatten(list(convert_state_dict(load_pth(str(ROOT / f"weights/EfRLFN-x{scale}.pth"))).items())))
    mx.eval(m.parameters())
    mlx_out = np.array(m(mx.array(img))).transpose(0, 3, 1, 2)

    max_abs = float(np.max(np.abs(pt_out - mlx_out)))
    print(f"\nEfRLFN x{scale}: max_abs = {max_abs:.2e}")
    assert max_abs < 1e-3, f"x{scale} parity fail: {max_abs:.2e}"
