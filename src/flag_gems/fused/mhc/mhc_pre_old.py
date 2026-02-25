"""
Triton implementation of mHC Pre operator.

The mHC pre block computes:
1. GEMM + sqrsum:  gemm_out = residual_flat @ fn.T, sqrsum = ||residual_flat||^2
2. RMS normalization of gemm_out
3. Split mixes → pre_mix (sigmoid + eps), post_mix (sigmoid * mult), comb_mix (Sinkhorn)
4. layer_input = sum_k(pre_mix_k * residual_k)  (weighted sum over streams)

Reference:
    mixes = RMSNorm(residual_flat @ fn.T)
    pre_mix  = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + pre_eps
    post_mix = sigmoid(mixes[:, hc:2hc] * scale[1] + base[hc:2hc]) * post_mult
    comb_mix = Sinkhorn(mixes[:, 2hc:].view(hc, hc), ...)
    layer_input = (residual * pre_mix.unsqueeze(-1)).sum(dim=-2)
"""

import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ─────────────────── Kernel 1: fused GEMM + sqrsum ───────────────────
# For the mHC pre block the "GEMM" is residual_flat @ fn.T where fn is small
# (hc_mult3 x hc_hidden_size, typically 24 x 5120). We implement it as a
# simple reduction kernel rather than using tensor-core matmul, because the
# N dimension (24) is too small for tiled GEMM.

@triton.jit
def mhc_pre_gemm_sqrsum_kernel(
    x_ptr,        # (num_tokens, hc_hidden_size), bfloat16
    fn_ptr,       # (hc_mult3, hc_hidden_size), float32
    out_ptr,      # (num_tokens, hc_mult3), float32
    sqrsum_ptr,   # (num_tokens,), float32
    x_stride_n,
    x_stride_h,
    fn_stride_m,
    fn_stride_h,
    out_stride_n,
    out_stride_m,
    hc_hidden_size,
    HC_MULT3: tl.constexpr,      # actual count of fn rows (e.g. 24)
    BLOCK_H: tl.constexpr,
):
    """Each program handles one token: compute dot products with all fn rows + sqrsum.
    Hardcoded for HC_MULT3 <= 24 using scalar accumulators to avoid power-of-2 issues.
    """
    pid_n = tl.program_id(0)

    # Use scalar accumulators for each of 24 outputs
    a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0; a4 = 0.0; a5 = 0.0
    a6 = 0.0; a7 = 0.0; a8 = 0.0; a9 = 0.0; a10 = 0.0; a11 = 0.0
    a12 = 0.0; a13 = 0.0; a14 = 0.0; a15 = 0.0; a16 = 0.0; a17 = 0.0
    a18 = 0.0; a19 = 0.0; a20 = 0.0; a21 = 0.0; a22 = 0.0; a23 = 0.0
    sq = 0.0

    for h_start in range(0, hc_hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hc_hidden_size

        # Load x[n, h] as float32
        x_vals = tl.load(x_ptr + pid_n * x_stride_n + h_offsets * x_stride_h, mask=h_mask, other=0.0).to(tl.float32)

        # sqrsum
        sq += tl.sum(x_vals * x_vals)

        # dot product with each fn row — unrolled scalar accumulators
        fn0 = tl.load(fn_ptr + 0 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a0 += tl.sum(x_vals * fn0)
        fn1 = tl.load(fn_ptr + 1 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a1 += tl.sum(x_vals * fn1)
        fn2 = tl.load(fn_ptr + 2 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a2 += tl.sum(x_vals * fn2)
        fn3 = tl.load(fn_ptr + 3 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a3 += tl.sum(x_vals * fn3)
        fn4 = tl.load(fn_ptr + 4 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a4 += tl.sum(x_vals * fn4)
        fn5 = tl.load(fn_ptr + 5 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a5 += tl.sum(x_vals * fn5)
        fn6 = tl.load(fn_ptr + 6 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a6 += tl.sum(x_vals * fn6)
        fn7 = tl.load(fn_ptr + 7 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a7 += tl.sum(x_vals * fn7)
        fn8 = tl.load(fn_ptr + 8 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a8 += tl.sum(x_vals * fn8)
        fn9 = tl.load(fn_ptr + 9 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a9 += tl.sum(x_vals * fn9)
        fn10 = tl.load(fn_ptr + 10 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a10 += tl.sum(x_vals * fn10)
        fn11 = tl.load(fn_ptr + 11 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a11 += tl.sum(x_vals * fn11)
        fn12 = tl.load(fn_ptr + 12 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a12 += tl.sum(x_vals * fn12)
        fn13 = tl.load(fn_ptr + 13 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a13 += tl.sum(x_vals * fn13)
        fn14 = tl.load(fn_ptr + 14 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a14 += tl.sum(x_vals * fn14)
        fn15 = tl.load(fn_ptr + 15 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a15 += tl.sum(x_vals * fn15)
        fn16 = tl.load(fn_ptr + 16 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a16 += tl.sum(x_vals * fn16)
        fn17 = tl.load(fn_ptr + 17 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a17 += tl.sum(x_vals * fn17)
        fn18 = tl.load(fn_ptr + 18 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a18 += tl.sum(x_vals * fn18)
        fn19 = tl.load(fn_ptr + 19 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a19 += tl.sum(x_vals * fn19)
        fn20 = tl.load(fn_ptr + 20 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a20 += tl.sum(x_vals * fn20)
        fn21 = tl.load(fn_ptr + 21 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a21 += tl.sum(x_vals * fn21)
        fn22 = tl.load(fn_ptr + 22 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a22 += tl.sum(x_vals * fn22)
        fn23 = tl.load(fn_ptr + 23 * fn_stride_m + h_offsets * fn_stride_h, mask=h_mask, other=0.0).to(tl.float32); a23 += tl.sum(x_vals * fn23)

    # Store results
    tl.store(sqrsum_ptr + pid_n, sq)
    base = pid_n * out_stride_n
    tl.store(out_ptr + base + 0 * out_stride_m, a0); tl.store(out_ptr + base + 1 * out_stride_m, a1)
    tl.store(out_ptr + base + 2 * out_stride_m, a2); tl.store(out_ptr + base + 3 * out_stride_m, a3)
    tl.store(out_ptr + base + 4 * out_stride_m, a4); tl.store(out_ptr + base + 5 * out_stride_m, a5)
    tl.store(out_ptr + base + 6 * out_stride_m, a6); tl.store(out_ptr + base + 7 * out_stride_m, a7)
    tl.store(out_ptr + base + 8 * out_stride_m, a8); tl.store(out_ptr + base + 9 * out_stride_m, a9)
    tl.store(out_ptr + base + 10 * out_stride_m, a10); tl.store(out_ptr + base + 11 * out_stride_m, a11)
    tl.store(out_ptr + base + 12 * out_stride_m, a12); tl.store(out_ptr + base + 13 * out_stride_m, a13)
    tl.store(out_ptr + base + 14 * out_stride_m, a14); tl.store(out_ptr + base + 15 * out_stride_m, a15)
    tl.store(out_ptr + base + 16 * out_stride_m, a16); tl.store(out_ptr + base + 17 * out_stride_m, a17)
    tl.store(out_ptr + base + 18 * out_stride_m, a18); tl.store(out_ptr + base + 19 * out_stride_m, a19)
    tl.store(out_ptr + base + 20 * out_stride_m, a20); tl.store(out_ptr + base + 21 * out_stride_m, a21)
    tl.store(out_ptr + base + 22 * out_stride_m, a22); tl.store(out_ptr + base + 23 * out_stride_m, a23)


# ─────────────── Kernel 2: fused norm + mix + sinkhorn + apply ───────────────

@triton.jit
def mhc_pre_fuse_kernel(
    gemm_out_ptr,   # (num_tokens, hc_mult3), float32
    sqrsum_ptr,     # (num_tokens,), float32
    hc_scale_ptr,   # (3,), float32
    hc_base_ptr,    # (hc_mult3,), float32
    residual_ptr,   # (num_tokens, hc_mult, hidden_size), bfloat16
    post_mix_ptr,   # (num_tokens, hc_mult), float32  — output
    comb_mix_ptr,   # (num_tokens, hc_mult*hc_mult), float32 — output
    layer_input_ptr, # (num_tokens, hidden_size), bfloat16 — output
    res_stride_n,
    res_stride_i,
    res_stride_h,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,  # hc_mult * hidden_size, used for RMS denom
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC: tl.constexpr,         # hc_mult (typically 4)
    HC_MULT3: tl.constexpr,   # hc_mult * (2 + hc_mult), typically 24
    HC_MULT3_PAD: tl.constexpr,  # next power-of-2 for Triton shapes
    BLOCK_H: tl.constexpr,
):
    """One program per token. Computes everything after GEMM+sqrsum.
    Hardcoded for HC=4 (hc_mult=4) to avoid non-power-of-2 tensor shapes.
    """
    pid_n = tl.program_id(0)

    # ── Stage 1: RMS normalize gemm_out ──
    sqrsum_val = tl.load(sqrsum_ptr + pid_n)
    rms_inv = tl.rsqrt(sqrsum_val / hc_hidden_size + rms_eps)

    # Load gemm_out as padded vector and normalize
    m_offsets = tl.arange(0, HC_MULT3_PAD)
    m_mask = m_offsets < HC_MULT3
    mixes = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + m_offsets, mask=m_mask, other=0.0) * rms_inv

    # Load scale and base
    scale_0 = tl.load(hc_scale_ptr + 0)
    scale_1 = tl.load(hc_scale_ptr + 1)
    scale_2 = tl.load(hc_scale_ptr + 2)
    base = tl.load(hc_base_ptr + m_offsets, mask=m_mask, other=0.0)

    # We extract individual scalar values from mixes and base using scalar loads
    # to avoid non-power-of-2 tensor indexing issues.

    # ── Stage 2a: pre_mix[k] = sigmoid(mixes[k] * scale_0 + base[k]) + pre_eps ──
    # ── Stage 2b: post_mix[k] = sigmoid(mixes[HC+k] * scale_1 + base[HC+k]) * mult ──
    # Use scalar loads for the small HC values
    pre_mix_0 = tl.sigmoid(tl.load(gemm_out_ptr + pid_n * HC_MULT3 + 0) * rms_inv * scale_0 + tl.load(hc_base_ptr + 0)) + hc_pre_eps
    pre_mix_1 = tl.sigmoid(tl.load(gemm_out_ptr + pid_n * HC_MULT3 + 1) * rms_inv * scale_0 + tl.load(hc_base_ptr + 1)) + hc_pre_eps
    pre_mix_2 = tl.sigmoid(tl.load(gemm_out_ptr + pid_n * HC_MULT3 + 2) * rms_inv * scale_0 + tl.load(hc_base_ptr + 2)) + hc_pre_eps
    pre_mix_3 = tl.sigmoid(tl.load(gemm_out_ptr + pid_n * HC_MULT3 + 3) * rms_inv * scale_0 + tl.load(hc_base_ptr + 3)) + hc_pre_eps

    post_0 = tl.sigmoid(tl.load(gemm_out_ptr + pid_n * HC_MULT3 + HC + 0) * rms_inv * scale_1 + tl.load(hc_base_ptr + HC + 0)) * hc_post_mult_value
    post_1 = tl.sigmoid(tl.load(gemm_out_ptr + pid_n * HC_MULT3 + HC + 1) * rms_inv * scale_1 + tl.load(hc_base_ptr + HC + 1)) * hc_post_mult_value
    post_2 = tl.sigmoid(tl.load(gemm_out_ptr + pid_n * HC_MULT3 + HC + 2) * rms_inv * scale_1 + tl.load(hc_base_ptr + HC + 2)) * hc_post_mult_value
    post_3 = tl.sigmoid(tl.load(gemm_out_ptr + pid_n * HC_MULT3 + HC + 3) * rms_inv * scale_1 + tl.load(hc_base_ptr + HC + 3)) * hc_post_mult_value

    tl.store(post_mix_ptr + pid_n * HC + 0, post_0)
    tl.store(post_mix_ptr + pid_n * HC + 1, post_1)
    tl.store(post_mix_ptr + pid_n * HC + 2, post_2)
    tl.store(post_mix_ptr + pid_n * HC + 3, post_3)

    # ── Stage 2c: comb_mix = Sinkhorn(mixes[2*HC:].reshape(HC,HC)) ──
    # Load 4x4 = 16 comb values as individual scalars
    comb_base = 2 * HC
    cm_00 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 0) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 0)
    cm_01 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 1) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 1)
    cm_02 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 2) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 2)
    cm_03 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 3) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 3)
    cm_10 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 4) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 4)
    cm_11 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 5) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 5)
    cm_12 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 6) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 6)
    cm_13 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 7) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 7)
    cm_20 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 8) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 8)
    cm_21 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 9) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 9)
    cm_22 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 10) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 10)
    cm_23 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 11) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 11)
    cm_30 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 12) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 12)
    cm_31 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 13) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 13)
    cm_32 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 14) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 14)
    cm_33 = tl.load(gemm_out_ptr + pid_n * HC_MULT3 + comb_base + 15) * rms_inv * scale_2 + tl.load(hc_base_ptr + comb_base + 15)

    # Softmax per row + eps
    # Row 0
    rm = tl.maximum(tl.maximum(cm_00, cm_01), tl.maximum(cm_02, cm_03))
    cm_00 = tl.exp(cm_00 - rm); cm_01 = tl.exp(cm_01 - rm); cm_02 = tl.exp(cm_02 - rm); cm_03 = tl.exp(cm_03 - rm)
    rs = cm_00 + cm_01 + cm_02 + cm_03
    cm_00 = cm_00 / rs + hc_sinkhorn_eps; cm_01 = cm_01 / rs + hc_sinkhorn_eps; cm_02 = cm_02 / rs + hc_sinkhorn_eps; cm_03 = cm_03 / rs + hc_sinkhorn_eps
    # Row 1
    rm = tl.maximum(tl.maximum(cm_10, cm_11), tl.maximum(cm_12, cm_13))
    cm_10 = tl.exp(cm_10 - rm); cm_11 = tl.exp(cm_11 - rm); cm_12 = tl.exp(cm_12 - rm); cm_13 = tl.exp(cm_13 - rm)
    rs = cm_10 + cm_11 + cm_12 + cm_13
    cm_10 = cm_10 / rs + hc_sinkhorn_eps; cm_11 = cm_11 / rs + hc_sinkhorn_eps; cm_12 = cm_12 / rs + hc_sinkhorn_eps; cm_13 = cm_13 / rs + hc_sinkhorn_eps
    # Row 2
    rm = tl.maximum(tl.maximum(cm_20, cm_21), tl.maximum(cm_22, cm_23))
    cm_20 = tl.exp(cm_20 - rm); cm_21 = tl.exp(cm_21 - rm); cm_22 = tl.exp(cm_22 - rm); cm_23 = tl.exp(cm_23 - rm)
    rs = cm_20 + cm_21 + cm_22 + cm_23
    cm_20 = cm_20 / rs + hc_sinkhorn_eps; cm_21 = cm_21 / rs + hc_sinkhorn_eps; cm_22 = cm_22 / rs + hc_sinkhorn_eps; cm_23 = cm_23 / rs + hc_sinkhorn_eps
    # Row 3
    rm = tl.maximum(tl.maximum(cm_30, cm_31), tl.maximum(cm_32, cm_33))
    cm_30 = tl.exp(cm_30 - rm); cm_31 = tl.exp(cm_31 - rm); cm_32 = tl.exp(cm_32 - rm); cm_33 = tl.exp(cm_33 - rm)
    rs = cm_30 + cm_31 + cm_32 + cm_33
    cm_30 = cm_30 / rs + hc_sinkhorn_eps; cm_31 = cm_31 / rs + hc_sinkhorn_eps; cm_32 = cm_32 / rs + hc_sinkhorn_eps; cm_33 = cm_33 / rs + hc_sinkhorn_eps

    # Col normalize
    cs0 = cm_00 + cm_10 + cm_20 + cm_30
    cs1 = cm_01 + cm_11 + cm_21 + cm_31
    cs2 = cm_02 + cm_12 + cm_22 + cm_32
    cs3 = cm_03 + cm_13 + cm_23 + cm_33
    cm_00 /= (cs0 + hc_sinkhorn_eps); cm_10 /= (cs0 + hc_sinkhorn_eps); cm_20 /= (cs0 + hc_sinkhorn_eps); cm_30 /= (cs0 + hc_sinkhorn_eps)
    cm_01 /= (cs1 + hc_sinkhorn_eps); cm_11 /= (cs1 + hc_sinkhorn_eps); cm_21 /= (cs1 + hc_sinkhorn_eps); cm_31 /= (cs1 + hc_sinkhorn_eps)
    cm_02 /= (cs2 + hc_sinkhorn_eps); cm_12 /= (cs2 + hc_sinkhorn_eps); cm_22 /= (cs2 + hc_sinkhorn_eps); cm_32 /= (cs2 + hc_sinkhorn_eps)
    cm_03 /= (cs3 + hc_sinkhorn_eps); cm_13 /= (cs3 + hc_sinkhorn_eps); cm_23 /= (cs3 + hc_sinkhorn_eps); cm_33 /= (cs3 + hc_sinkhorn_eps)

    # Sinkhorn iterations
    for _ in tl.static_range(sinkhorn_repeat - 1):
        # Row normalize
        rs0 = cm_00 + cm_01 + cm_02 + cm_03
        rs1 = cm_10 + cm_11 + cm_12 + cm_13
        rs2 = cm_20 + cm_21 + cm_22 + cm_23
        rs3 = cm_30 + cm_31 + cm_32 + cm_33
        cm_00 /= (rs0 + hc_sinkhorn_eps); cm_01 /= (rs0 + hc_sinkhorn_eps); cm_02 /= (rs0 + hc_sinkhorn_eps); cm_03 /= (rs0 + hc_sinkhorn_eps)
        cm_10 /= (rs1 + hc_sinkhorn_eps); cm_11 /= (rs1 + hc_sinkhorn_eps); cm_12 /= (rs1 + hc_sinkhorn_eps); cm_13 /= (rs1 + hc_sinkhorn_eps)
        cm_20 /= (rs2 + hc_sinkhorn_eps); cm_21 /= (rs2 + hc_sinkhorn_eps); cm_22 /= (rs2 + hc_sinkhorn_eps); cm_23 /= (rs2 + hc_sinkhorn_eps)
        cm_30 /= (rs3 + hc_sinkhorn_eps); cm_31 /= (rs3 + hc_sinkhorn_eps); cm_32 /= (rs3 + hc_sinkhorn_eps); cm_33 /= (rs3 + hc_sinkhorn_eps)
        # Col normalize
        cs0 = cm_00 + cm_10 + cm_20 + cm_30
        cs1 = cm_01 + cm_11 + cm_21 + cm_31
        cs2 = cm_02 + cm_12 + cm_22 + cm_32
        cs3 = cm_03 + cm_13 + cm_23 + cm_33
        cm_00 /= (cs0 + hc_sinkhorn_eps); cm_10 /= (cs0 + hc_sinkhorn_eps); cm_20 /= (cs0 + hc_sinkhorn_eps); cm_30 /= (cs0 + hc_sinkhorn_eps)
        cm_01 /= (cs1 + hc_sinkhorn_eps); cm_11 /= (cs1 + hc_sinkhorn_eps); cm_21 /= (cs1 + hc_sinkhorn_eps); cm_31 /= (cs1 + hc_sinkhorn_eps)
        cm_02 /= (cs2 + hc_sinkhorn_eps); cm_12 /= (cs2 + hc_sinkhorn_eps); cm_22 /= (cs2 + hc_sinkhorn_eps); cm_32 /= (cs2 + hc_sinkhorn_eps)
        cm_03 /= (cs3 + hc_sinkhorn_eps); cm_13 /= (cs3 + hc_sinkhorn_eps); cm_23 /= (cs3 + hc_sinkhorn_eps); cm_33 /= (cs3 + hc_sinkhorn_eps)

    # Store comb_mix (row-major: [i*HC+j])
    comb_base_out = pid_n * HC * HC
    tl.store(comb_mix_ptr + comb_base_out + 0, cm_00); tl.store(comb_mix_ptr + comb_base_out + 1, cm_01)
    tl.store(comb_mix_ptr + comb_base_out + 2, cm_02); tl.store(comb_mix_ptr + comb_base_out + 3, cm_03)
    tl.store(comb_mix_ptr + comb_base_out + 4, cm_10); tl.store(comb_mix_ptr + comb_base_out + 5, cm_11)
    tl.store(comb_mix_ptr + comb_base_out + 6, cm_12); tl.store(comb_mix_ptr + comb_base_out + 7, cm_13)
    tl.store(comb_mix_ptr + comb_base_out + 8, cm_20); tl.store(comb_mix_ptr + comb_base_out + 9, cm_21)
    tl.store(comb_mix_ptr + comb_base_out + 10, cm_22); tl.store(comb_mix_ptr + comb_base_out + 11, cm_23)
    tl.store(comb_mix_ptr + comb_base_out + 12, cm_30); tl.store(comb_mix_ptr + comb_base_out + 13, cm_31)
    tl.store(comb_mix_ptr + comb_base_out + 14, cm_32); tl.store(comb_mix_ptr + comb_base_out + 15, cm_33)

    # ── Stage 3: layer_input = sum_k(pre_mix_k * residual[n, k, :]) ──
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size

        r0 = tl.load(residual_ptr + pid_n * res_stride_n + 0 * res_stride_i + h_offsets * res_stride_h, mask=h_mask, other=0.0).to(tl.float32)
        r1 = tl.load(residual_ptr + pid_n * res_stride_n + 1 * res_stride_i + h_offsets * res_stride_h, mask=h_mask, other=0.0).to(tl.float32)
        r2 = tl.load(residual_ptr + pid_n * res_stride_n + 2 * res_stride_i + h_offsets * res_stride_h, mask=h_mask, other=0.0).to(tl.float32)
        r3 = tl.load(residual_ptr + pid_n * res_stride_n + 3 * res_stride_i + h_offsets * res_stride_h, mask=h_mask, other=0.0).to(tl.float32)

        acc = pre_mix_0 * r0 + pre_mix_1 * r1 + pre_mix_2 * r2 + pre_mix_3 * r3

        tl.store(
            layer_input_ptr + pid_n * li_stride_n + h_offsets * li_stride_h,
            acc.to(tl.bfloat16),
            mask=h_mask,
        )


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block (Triton version).

    Args:
        residual: (..., hc_mult, hidden_size), bfloat16
        fn: (hc_mult3, hc_mult * hidden_size), float32
        hc_scale: (3,), float32
        hc_base: (hc_mult3,), float32
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value: float scalars
        sinkhorn_repeat: int
        n_splits: unused, kept for API compatibility

    Returns:
        post_mix: (..., hc_mult, 1), float32
        comb_mix: (..., hc_mult, hc_mult), float32
        layer_input: (..., hidden_size), bfloat16
    """
    logger.debug(
        "GEMS MHC_PRE FORWARD, residual=%s, fn=%s", residual.shape, fn.shape,
    )

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    num_tokens = residual_flat.shape[0]

    device = residual.device

    # Allocate intermediate and output buffers
    gemm_out = torch.empty(num_tokens, hc_mult3, dtype=torch.float32, device=device)
    sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=device)
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    x_flat = residual_flat.reshape(num_tokens, hc_hidden_size).contiguous()
    fn_c = fn.contiguous()

    # ── Kernel 1: GEMM + sqrsum ──
    BLOCK_H_GEMM = min(triton.next_power_of_2(hc_hidden_size), 4096)
    HC_MULT3_PAD = triton.next_power_of_2(hc_mult3)
    mhc_pre_gemm_sqrsum_kernel[(num_tokens,)](
        x_flat, fn_c, gemm_out, sqrsum,
        x_flat.stride(0), x_flat.stride(1),
        fn_c.stride(0), fn_c.stride(1),
        gemm_out.stride(0), gemm_out.stride(1),
        hc_hidden_size,
        HC_MULT3=hc_mult3,
        BLOCK_H=BLOCK_H_GEMM,
    )

    # ── Kernel 2: fuse (norm + split_mixes + sinkhorn + apply_pre_mix) ──
    BLOCK_H_FUSE = min(triton.next_power_of_2(hidden_size), 1024)
    mhc_pre_fuse_kernel[(num_tokens,)](
        gemm_out, sqrsum,
        hc_scale, hc_base,
        residual_flat, post_mix, comb_mix, layer_input,
        residual_flat.stride(0), residual_flat.stride(1), residual_flat.stride(2),
        layer_input.stride(0), layer_input.stride(1),
        hidden_size,
        hc_hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        HC=hc_mult,
        HC_MULT3=hc_mult3,
        HC_MULT3_PAD=HC_MULT3_PAD,
        BLOCK_H=BLOCK_H_FUSE,
    )

    # Reshape outputs
    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


# ───────────────────────── Reference implementations ─────────────────────────

def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference (copied from tilelang example)."""
    hc_mult = residual.shape[-2]

    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()

    hc_scale_expanded = torch.cat([
        hc_scale[0].expand(hc_mult),
        hc_scale[1].expand(hc_mult),
        hc_scale[2].expand(hc_mult * hc_mult),
    ])
    mixes = mixes * hc_scale_expanded + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_mult_value).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult)
    res_mix = sinkhorn_normalize_ref(res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)
    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input
