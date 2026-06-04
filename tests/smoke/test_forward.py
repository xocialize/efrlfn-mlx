import mlx.core as mx
from efrlfn_mlx import EfRLFN, EfRLFNConfig
def test_x4_shape():
    y = EfRLFN(EfRLFNConfig.x4())(mx.zeros((1, 24, 32, 3))); mx.eval(y)
    assert y.shape == (1, 96, 128, 3)
def test_x2_shape():
    y = EfRLFN(EfRLFNConfig.x2())(mx.zeros((1, 24, 32, 3))); mx.eval(y)
    assert y.shape == (1, 48, 64, 3)
