"""
Triton implementation of mHC Post operator.

Computes:
    out[n, i, h] = post_layer_mix[n, i] * x[n, h] + sum_j(comb_res_mix[n, j, i] * residual[n, j, h])

Equivalent PyTorch reference:
    term2 = bmm(comb_res_mix.mT, residual.float())
    out = (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def mhc_post_kernel(
    # --- pointers ---
    a_ptr,  # comb_res_mix: (N, hc, hc), float32
    b_ptr,  # residual:     (N, hc, H),  bfloat16
    c_ptr,  # post_layer_mix: (N, hc),   float32
    d_ptr,  # x:            (N, H),      bfloat16
    out_ptr,  # output:    (N, hc, H),  bfloat16
    # --- strides ---
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
    # --- dimensions ---
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Each program handles one token (pid_n) and one hidden-dim tile (pid_h).
    For each of the HC output streams, we compute:
        out[n, i, h] = c[n, i] * d[n, h] + sum_j a[n, j, i] * b[n, j, h]
    Note: a indexing uses [j, i] because we need comb_res_mix.mT, so a[n, j, i].
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Load d[n, h_offsets] -> (BLOCK_H,), bfloat16 -> float32
    d_vals = tl.load(d_ptr + pid_n * d_stride_n + h_offsets * d_stride_h, mask=h_mask, other=0.0).to(tl.float32)

    # For each output stream i in [0, HC):
    for i in tl.static_range(HC):
        # c[n, i] scalar
        c_val = tl.load(c_ptr + pid_n * c_stride_n + i * c_stride_i).to(tl.float32)

        # acc = c[n, i] * d[n, h]
        acc = c_val * d_vals

        # accumulate a[n, j, i] * b[n, j, h] for j in [0, HC)
        for j in tl.static_range(HC):
            a_val = tl.load(a_ptr + pid_n * a_stride_n + j * a_stride_i + i * a_stride_j).to(tl.float32)
            b_vals = tl.load(b_ptr + pid_n * b_stride_n + j * b_stride_i + h_offsets * b_stride_h, mask=h_mask, other=0.0).to(tl.float32)
            acc += a_val * b_vals

        # Store as bfloat16
        tl.store(
            out_ptr + pid_n * out_stride_n + i * out_stride_i + h_offsets * out_stride_h,
            acc.to(tl.bfloat16),
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
        x.shape, residual.shape, post_layer_mix.shape, comb_res_mix.shape,
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
        a, b, c, d, out,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        d.stride(0), d.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        H=H,
        HC=hc,
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
