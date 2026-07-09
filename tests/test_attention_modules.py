import pytest
import torch
from ultralytics.nn.modules import (
    CBAM, ECA, C2CBAM, C2ECA, C2MSCA, C2CA, CoordinateAttention,
    C2GC, GlobalContext, TripletAttention, C2Triplet, SimAM, C2SimAM,
    C2EMA, EMA, BiFormer, C2BiFormer,
)

CHANNELS = 256
X = torch.randn(2, CHANNELS, 32, 32)

MODULES = [
    ("CBAM", lambda: CBAM(c1=CHANNELS)),
    ("ECA", lambda: ECA(channels=CHANNELS)),
    ("C2CBAM", lambda: C2CBAM(c1=CHANNELS, c2=CHANNELS, n=1)),
    ("C2ECA", lambda: C2ECA(c1=CHANNELS, c2=CHANNELS, n=1)),
    ("C2MSCA", lambda: C2MSCA(c1=CHANNELS, c2=CHANNELS, n=1)),
    ("C2CA", lambda: C2CA(c1=CHANNELS, c2=CHANNELS, n=1)),
    ("CoordinateAttention", lambda: CoordinateAttention(channels=CHANNELS)),
    ("C2GC", lambda: C2GC(c1=CHANNELS, c2=CHANNELS, n=1)),
    ("GlobalContext", lambda: GlobalContext(CHANNELS)),
    ("TripletAttention", lambda: TripletAttention()),
    ("C2Triplet", lambda: C2Triplet(c1=CHANNELS, c2=CHANNELS, n=1)),
    ("SimAM", lambda: SimAM()),
    ("C2SimAM", lambda: C2SimAM(c1=CHANNELS, c2=CHANNELS, n=1)),
    ("EMA", lambda: EMA(CHANNELS)),
    ("C2EMA", lambda: C2EMA(c1=CHANNELS, c2=CHANNELS, n=1)),
    ("BiFormer", lambda: BiFormer(dim=CHANNELS, num_heads=8)),
    ("C2BiFormer", lambda: C2BiFormer(c1=CHANNELS, c2=CHANNELS, n=1)),
]

@pytest.mark.parametrize("name,factory", MODULES, ids=[m[0] for m in MODULES])
def test_forward_pass_preserves_shape(name, factory):
    module = factory()
    y = module(X)
    assert y.shape == X.shape, f"{name}: expected {X.shape}, got {y.shape}"

def test_simam_has_zero_parameters():
    # SimAM is parameter-free
    assert sum(p.numel() for p in SimAM().parameters()) == 0