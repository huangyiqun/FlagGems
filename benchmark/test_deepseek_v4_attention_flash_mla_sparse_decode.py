import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_flash_mla_sparse_decode import (
    flash_mla_sparse_decode,
)

from . import base


class FlashMlaSparseDecodeBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "flash_mla_sparse_decode",
            flash_mla_sparse_decode,
            [torch.bfloat16],
            gems_op=flash_mla_sparse_decode,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(1, 1, 64, 576, 128)]

    def get_input_iter(self, dtype):
        for bsz, sq, h, dt, topk in self.shapes:
            block_size = 64
            rope_dim = 64
            nope_dim = dt - rope_dim
            scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
            token_data_size = nope_dim + rope_dim * 2
            block_stride = block_size * token_data_size + block_size * scale_slots
            q = torch.randn((bsz, sq, h, dt), device="cuda", dtype=dtype)
            k_cache = torch.zeros((2, block_stride), device="cuda", dtype=torch.uint8)
            indices = torch.arange(topk, device="cuda", dtype=torch.int32).view(
                bsz, sq, topk
            )
            attn_sink = torch.zeros((h,), device="cuda", dtype=torch.float32)
            topk_length = torch.full(
                (bsz * sq,), topk, device="cuda", dtype=torch.int32
            )
            out = torch.empty((bsz, sq, h, 512), device="cuda", dtype=dtype)
            yield (
                q,
                k_cache,
                indices,
                dt**-0.5,
                512,
                attn_sink,
                None,
                None,
                topk_length,
                None,
                out,
                block_size,
                rope_dim,
                nope_dim,
                scale_slots,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_flash_mla_sparse_decode_benchmark():
    FlashMlaSparseDecodeBenchmark().run()
