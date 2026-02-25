"""
Performance benchmark for mHC operators: Triton (FlagGems) vs TileLang vs PyTorch reference.

Usage:
    pytest benchmark/test_mhc_perf.py -v -s
"""

import sys

import pytest
import torch

from flag_gems.fused.mhc.mhc_post import mhc_post, mhc_post_ref
from flag_gems.fused.mhc.mhc_pre import mhc_pre, mhc_pre_ref

from .performance_utils import Benchmark

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
#  mhc_post benchmark
# ═══════════════════════════════════════════════════════════════


class MHCPostBenchmark(Benchmark):
    DEFAULT_SHAPE_DESC = "N, H"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (512, 1280),
            (1024, 2560),
            (4096, 1280),
            (4096, 2560),
            (4096, 7168),
            (8192, 7168),
        ]

    def get_input_iter(self, cur_dtype):
        for n, h in self.shapes:
            hc_mult = 4
            x = torch.randn((n, h), dtype=torch.bfloat16, device=self.device)
            residual = torch.randn(
                (n, hc_mult, h), dtype=torch.bfloat16, device=self.device
            )
            post_layer_mix = torch.randn(
                (n, hc_mult, 1), dtype=torch.float32, device=self.device
            )
            comb_res_mix = torch.randn(
                (n, hc_mult, hc_mult), dtype=torch.float32, device=self.device
            )
            yield x, residual, post_layer_mix, comb_res_mix


@pytest.mark.mhc_post
def test_perf_mhc_post():
    bench = MHCPostBenchmark(
        op_name="mhc_post",
        torch_op=mhc_post_ref,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(mhc_post)
    bench.run()


@pytest.mark.mhc_post
@pytest.mark.skipif(not HAS_TILELANG, reason="TileLang not available")
def test_perf_mhc_post_vs_tilelang():
    bench = MHCPostBenchmark(
        op_name="mhc_post_vs_tilelang",
        torch_op=mhc_post_tl,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(mhc_post)
    bench.run()


# ═══════════════════════════════════════════════════════════════
#  mhc_pre benchmark
# ═══════════════════════════════════════════════════════════════


class MHCPreBenchmark(Benchmark):
    DEFAULT_SHAPE_DESC = "N, hidden_size"

    def __init__(self, *args, hc_mult=4, sinkhorn_repeat=10, **kwargs):
        self.hc_mult = hc_mult
        self.sinkhorn_repeat = sinkhorn_repeat
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (512, 1280),
            (1024, 2560),
            (2048, 4096),
            (4096, 1280),
            (4096, 2560),
            (8192, 4096),
        ]

    def get_input_iter(self, cur_dtype):
        for n, hidden_size in self.shapes:
            hc_mult = self.hc_mult
            hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
            device = self.device

            torch.manual_seed(42)
            residual = (
                torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device)
                .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
                .bfloat16()
            )
            fn = (
                torch.randn(
                    (hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device
                )
                * 1e-4
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
            ).flatten(1, 2)
            hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
            hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

            yield (
                residual,
                fn,
                hc_scale,
                hc_base,
                1e-6,  # rms_eps
                1e-6,  # hc_pre_eps
                1e-6,  # hc_sinkhorn_eps
                1.0,  # hc_post_mult_value
                self.sinkhorn_repeat,
            )


@pytest.mark.mhc_pre
def test_perf_mhc_pre():
    bench = MHCPreBenchmark(
        op_name="mhc_pre",
        torch_op=mhc_pre_ref,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(mhc_pre)
    bench.run()


@pytest.mark.mhc_pre
@pytest.mark.skipif(not HAS_TILELANG, reason="TileLang not available")
def test_perf_mhc_pre_vs_tilelang():
    bench = MHCPreBenchmark(
        op_name="mhc_pre_vs_tilelang",
        torch_op=mhc_pre_tl,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(mhc_pre)
    bench.run()
