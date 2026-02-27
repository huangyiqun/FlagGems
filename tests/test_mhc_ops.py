"""
Accuracy tests for mHC (Manifold Constrained Hyper-Connection) operators.

Tests both mhc_post and mhc_pre against PyTorch reference implementations,
and optionally compares with TileLang implementations.
"""

import sys
from itertools import product

import pytest
import torch

from flag_gems.fused.mhc.mhc_post import mhc_post, mhc_post_ref
from flag_gems.fused.mhc.mhc_pre import mhc_pre, mhc_pre_ref

# ─── Import TileLang versions for comparison ───
sys.path.insert(0, "/workspace/tilelang/examples/deepseek_mhc")
try:
    from example_mhc_post import mhc_post as mhc_post_tl
    from example_mhc_pre import mhc_pre as mhc_pre_tl

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False
    mhc_post_tl = None
    mhc_pre_tl = None


# ═══════════════════════════════════════════════════════════════
#  Test data generators
# ═══════════════════════════════════════════════════════════════


def generate_mhc_post_data(n: int, h: int, hc_mult: int = 4, device: str = "cuda"):
    torch.manual_seed(42)
    x = torch.randn((n, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n, hc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(
        (n, hc_mult, hc_mult), dtype=torch.float32, device=device
    )
    return dict(
        x=x, residual=residual, post_layer_mix=post_layer_mix, comb_res_mix=comb_res_mix
    )


def generate_mhc_pre_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    device: str = "cuda",
):
    torch.manual_seed(42)
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    residual = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return dict(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )


# ═══════════════════════════════════════════════════════════════
#  mhc_post tests
# ═══════════════════════════════════════════════════════════════

MHC_POST_CONFIGS = list(
    product(
        [4096],  # n (num_tokens)
        [1280, 2560, 7168],  # h (hidden_size)
    )
)


@pytest.mark.mhc_post
@pytest.mark.parametrize(
    "n, h", MHC_POST_CONFIGS, ids=[f"n{n}_h{h}" for n, h in MHC_POST_CONFIGS]
)
def test_mhc_post_vs_ref(n, h):
    """Test Triton mhc_post against PyTorch reference."""
    data = generate_mhc_post_data(n, h)
    out_triton = mhc_post(**data)
    out_ref = mhc_post_ref(**data)
    torch.testing.assert_close(out_triton, out_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.mhc_post
@pytest.mark.skipif(not HAS_TILELANG, reason="TileLang not available")
@pytest.mark.parametrize(
    "n, h", MHC_POST_CONFIGS, ids=[f"n{n}_h{h}" for n, h in MHC_POST_CONFIGS]
)
def test_mhc_post_vs_tilelang(n, h):
    """Test Triton mhc_post against TileLang implementation."""
    data = generate_mhc_post_data(n, h)
    out_triton = mhc_post(**data)
    out_tl = mhc_post_tl(**data)
    torch.testing.assert_close(out_triton, out_tl, rtol=1e-2, atol=1e-2)


# ═══════════════════════════════════════════════════════════════
#  mhc_pre tests
# ═══════════════════════════════════════════════════════════════

MHC_PRE_CONFIGS = list(
    product(
        [512, 1024, 2048, 8192],  # n
        [1280, 2560, 4096],  # hidden_size
        [4],  # hc_mult
    )
)


@pytest.mark.mhc_pre
@pytest.mark.parametrize(
    "n, hidden_size, hc_mult",
    MHC_PRE_CONFIGS,
    ids=[f"n{n}_h{h}_hc{hc}" for n, h, hc in MHC_PRE_CONFIGS],
)
def test_mhc_pre_vs_ref(n, hidden_size, hc_mult):
    """Test Triton mhc_pre against PyTorch reference."""
    data = generate_mhc_pre_data(n, hc_mult, hidden_size)
    post_triton, comb_triton, li_triton = mhc_pre(**data)
    post_ref, comb_ref, li_ref = mhc_pre_ref(**data)

    torch.testing.assert_close(post_triton, post_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(comb_triton, comb_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(li_triton, li_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.mhc_pre
@pytest.mark.skipif(not HAS_TILELANG, reason="TileLang not available")
@pytest.mark.parametrize(
    "n, hidden_size, hc_mult",
    MHC_PRE_CONFIGS,
    ids=[f"n{n}_h{h}_hc{hc}" for n, h, hc in MHC_PRE_CONFIGS],
)
def test_mhc_pre_vs_tilelang(n, hidden_size, hc_mult):
    """Test Triton mhc_pre against TileLang implementation."""
    data = generate_mhc_pre_data(n, hc_mult, hidden_size)
    post_triton, comb_triton, li_triton = mhc_pre(**data)
    post_tl, comb_tl, li_tl = mhc_pre_tl(**data)

    torch.testing.assert_close(post_triton, post_tl, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(comb_triton, comb_tl, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(li_triton, li_tl, rtol=1e-2, atol=1e-2)
