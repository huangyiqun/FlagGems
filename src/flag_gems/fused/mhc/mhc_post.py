"""
Triton implementation of mHC Post operator (optimized).

Computes:
    out[n, i, h] = post_layer_mix[n, i] * x[n, h] + sum_j(comb_res_mix[n, j, i] * residual[n, j, h])

Key optimization: load residual[n, j, h_block] once per h-tile and compute
all HC output streams simultaneously, avoiding HC× redundant reads.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def mhc_post_kernel(
    a_ptr,  # comb_res_mix: (N, hc, hc), float32 — a[n, j, i]
    b_ptr,  # residual:     (N, hc, H),  bfloat16
    c_ptr,  # post_layer_mix: (N, hc),   float32
    d_ptr,  # x:            (N, H),      bfloat16
    out_ptr,  # output:       (N, hc, H),  bfloat16
    a_stride_n,
    a_stride_i,
    a_stride_j,
    b_stride_n,
    b_stride_i,
    b_stride_h,
    c_stride_n,
    c_stride_i,
    d_stride_n,
    d_stride_h,
    out_stride_n,
    out_stride_i,
    out_stride_h,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Optimized: load residual & x once per h-tile, compute all HC outputs.
    Grid: (N, cdiv(H, BLOCK_H))
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Load x[n, h] once
    d_vals = tl.load(
        d_ptr + pid_n * d_stride_n + h_offsets * d_stride_h, mask=h_mask, other=0.0
    ).to(tl.float32)

    # Load residual[n, j, h] for all j once — key optimization!
    b0 = tl.load(
        b_ptr + pid_n * b_stride_n + 0 * b_stride_i + h_offsets * b_stride_h,
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)
    b1 = tl.load(
        b_ptr + pid_n * b_stride_n + 1 * b_stride_i + h_offsets * b_stride_h,
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)
    b2 = tl.load(
        b_ptr + pid_n * b_stride_n + 2 * b_stride_i + h_offsets * b_stride_h,
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)
    b3 = tl.load(
        b_ptr + pid_n * b_stride_n + 3 * b_stride_i + h_offsets * b_stride_h,
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)

    # Load all 16 comb_res_mix scalars + 4 post_layer_mix scalars for this token
    a_base = pid_n * a_stride_n
    c_base = pid_n * c_stride_n
    out_base = pid_n * out_stride_n

    # For output stream i=0
    c0 = tl.load(c_ptr + c_base + 0 * c_stride_i).to(tl.float32)
    a00 = tl.load(a_ptr + a_base + 0 * a_stride_i + 0 * a_stride_j).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 1 * a_stride_i + 0 * a_stride_j).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 2 * a_stride_i + 0 * a_stride_j).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 3 * a_stride_i + 0 * a_stride_j).to(tl.float32)
    acc0 = c0 * d_vals + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    tl.store(
        out_ptr + out_base + 0 * out_stride_i + h_offsets * out_stride_h,
        acc0.to(tl.bfloat16),
        mask=h_mask,
    )

    # For output stream i=1
    c1 = tl.load(c_ptr + c_base + 1 * c_stride_i).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 0 * a_stride_i + 1 * a_stride_j).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 1 * a_stride_i + 1 * a_stride_j).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 2 * a_stride_i + 1 * a_stride_j).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 3 * a_stride_i + 1 * a_stride_j).to(tl.float32)
    acc1 = c1 * d_vals + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    tl.store(
        out_ptr + out_base + 1 * out_stride_i + h_offsets * out_stride_h,
        acc1.to(tl.bfloat16),
        mask=h_mask,
    )

    # For output stream i=2
    c2 = tl.load(c_ptr + c_base + 2 * c_stride_i).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 0 * a_stride_i + 2 * a_stride_j).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 1 * a_stride_i + 2 * a_stride_j).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 2 * a_stride_i + 2 * a_stride_j).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 3 * a_stride_i + 2 * a_stride_j).to(tl.float32)
    acc2 = c2 * d_vals + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    tl.store(
        out_ptr + out_base + 2 * out_stride_i + h_offsets * out_stride_h,
        acc2.to(tl.bfloat16),
        mask=h_mask,
    )

    # For output stream i=3
    c3 = tl.load(c_ptr + c_base + 3 * c_stride_i).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 0 * a_stride_i + 3 * a_stride_j).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 1 * a_stride_i + 3 * a_stride_j).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 2 * a_stride_i + 3 * a_stride_j).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 3 * a_stride_i + 3 * a_stride_j).to(tl.float32)
    acc3 = c3 * d_vals + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3
    tl.store(
        out_ptr + out_base + 3 * out_stride_i + h_offsets * out_stride_h,
        acc3.to(tl.bfloat16),
        mask=h_mask,
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """
    mHC post-processing operator.

    Args:
        x: (N, H), bfloat16 — layer output
        residual: (N, hc_mult, H), bfloat16 — multi-head residual
        post_layer_mix: (N, hc_mult, 1), float32 — per-stream scale for x
        comb_res_mix: (N, hc_mult, hc_mult), float32 — combination matrix (applied transposed)

    Returns:
        out: (N, hc_mult, H), bfloat16
    """
    logger.debug(
        "GEMS MHC_POST FORWARD, x=%s, residual=%s, post_layer_mix=%s, comb_res_mix=%s",
        x.shape,
        residual.shape,
        post_layer_mix.shape,
        comb_res_mix.shape,
    )

    N, hc, H = residual.shape
    assert x.shape == (N, H)
    assert post_layer_mix.shape == (N, hc, 1)
    assert comb_res_mix.shape == (N, hc, hc)

    out = torch.empty_like(residual)

    c = post_layer_mix.squeeze(-1).contiguous()  # (N, hc)
    a = comb_res_mix.contiguous()  # (N, hc, hc)
    b = residual.contiguous()  # (N, hc, H)
    d = x.contiguous()  # (N, H)

    BLOCK_H = min(triton.next_power_of_2(H), 1024)

    grid = (N, triton.cdiv(H, BLOCK_H))

    mhc_post_kernel[grid](
        a,
        b,
        c,
        d,
        out,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        d.stride(0),
        d.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H=H,
        BLOCK_H=BLOCK_H,
    )
    return out


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference implementation."""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
