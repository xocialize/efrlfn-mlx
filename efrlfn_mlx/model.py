"""EfRLFN in MLX — faithful port of EvgeneyBogatyrev/EfRLFN (code/model.py, blocks.py).

Isomorphic class structure (EfRLFN, ERLFB, ECABlock). NHWC layout (MLX convs): channel
ops act on the last axis, the ECA channel-attention Conv1d runs over the channel axis as a
length-C / 1-feature sequence, PixelShuffle uses PyTorch's (C,r,r) channel grouping.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from .config import EfRLFNConfig


def pixel_shuffle(x: mx.array, r: int) -> mx.array:
    n, h, w, crr = x.shape
    c = crr // (r * r)
    x = x.reshape(n, h, w, c, r, r).transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(n, h * r, w * r, c)


class ECABlock(nn.Module):
    """Efficient Channel Attention: global pool -> Conv1d over channels -> sigmoid gate."""
    def __init__(self, k_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        n, _, _, c = x.shape
        y = mx.mean(x, axis=(1, 2), keepdims=True)     # [n,1,1,c]
        y = y.reshape(n, c, 1)                          # NLC: length c, 1 feature
        y = self.conv(y).reshape(n, 1, 1, c)
        return x * mx.sigmoid(y)


class ERLFB(nn.Module):
    """Efficient Residual Local Feature Block."""
    def __init__(self, c: int):
        super().__init__()
        self.c1_r = nn.Conv2d(c, c, 3, padding=1)
        self.c2_r = nn.Conv2d(c, c, 3, padding=1)
        self.c3_r = nn.Conv2d(c, c, 3, padding=1)
        self.c5 = nn.Conv2d(c, c, 1)
        self.eca = ECABlock()

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.tanh(self.c1_r(x))
        out = mx.tanh(self.c2_r(out))
        out = mx.tanh(self.c3_r(out))
        out = out + x
        return self.eca(self.c5(out))


class EfRLFN(nn.Module):
    def __init__(self, config: EfRLFNConfig | None = None):
        super().__init__()
        self.config = config = config or EfRLFNConfig()
        fc = config.feature_channels
        self.conv_1 = nn.Conv2d(config.in_channels, fc, 3, padding=1)
        # upstream names blocks block_1..block_6 (flat attributes)
        for i in range(1, config.num_blocks + 1):
            setattr(self, f"block_{i}", ERLFB(fc))
        self.conv_2 = nn.Conv2d(fc, fc, 3, padding=1)
        self.upscale = config.upscale
        self.upsampler = nn.Conv2d(fc, config.out_channels * config.upscale ** 2, 3, padding=1)
        self._nblk = config.num_blocks

    def __call__(self, x: mx.array) -> mx.array:
        """x: [N,H,W,3] in [0,1]. Returns upscaled [N, H*s, W*s, 3]."""
        feat = self.conv_1(x)
        out = feat
        for i in range(1, self._nblk + 1):
            out = getattr(self, f"block_{i}")(out)
        out = self.conv_2(out + feat)
        return pixel_shuffle(self.upsampler(out), self.upscale)
