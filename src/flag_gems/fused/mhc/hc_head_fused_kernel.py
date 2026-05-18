import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_K": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_K": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_K": 8192}, num_warps=8, num_stages=2),
    ],
    key=["K", "HC"],
)
@triton.jit
def _hc_head_apply_pre_mix_kernel(
    hs_ptr,
    fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    out_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    rms_eps,
    hc_eps,
    hs_stride_t,
    fn_stride_m,
    out_stride_t,
    HC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)

    if pid_t >= T:
        return

    hs_t_base = pid_t * hs_stride_t

    sqr_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    mix_acc0 = tl.zeros([BLOCK_K], dtype=tl.float32)
    mix_acc1 = tl.zeros([BLOCK_K], dtype=tl.float32)
    mix_acc2 = tl.zeros([BLOCK_K], dtype=tl.float32)
    mix_acc3 = tl.zeros([BLOCK_K], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        x_vals = tl.load(hs_ptr + hs_t_base + k_off, mask=k_mask, other=0.0).to(
            tl.float32
        )
        sqr_acc += x_vals * x_vals

        fn0 = tl.load(fn_ptr + 0 * fn_stride_m + k_off, mask=k_mask, other=0.0)
        fn1 = tl.load(fn_ptr + 1 * fn_stride_m + k_off, mask=k_mask, other=0.0)
        mix_acc0 += x_vals * fn0
        mix_acc1 += x_vals * fn1

        if HC > 2:
            fn2 = tl.load(fn_ptr + 2 * fn_stride_m + k_off, mask=k_mask, other=0.0)
            fn3 = tl.load(fn_ptr + 3 * fn_stride_m + k_off, mask=k_mask, other=0.0)
            mix_acc2 += x_vals * fn2
            mix_acc3 += x_vals * fn3

    sqr_total = tl.sum(sqr_acc)
    rsqrt = tl.math.rsqrt(sqr_total / K + rms_eps)

    hc_scale = tl.load(hc_scale_ptr)

    mix0 = tl.sum(mix_acc0)
    mix1 = tl.sum(mix_acc1)
    pre_mix0 = tl.sigmoid(mix0 * rsqrt * hc_scale + tl.load(hc_base_ptr + 0)) + hc_eps
    pre_mix1 = tl.sigmoid(mix1 * rsqrt * hc_scale + tl.load(hc_base_ptr + 1)) + hc_eps

    if HC > 2:
        mix2 = tl.sum(mix_acc2)
        mix3 = tl.sum(mix_acc3)
        pre_mix2 = (
            tl.sigmoid(mix2 * rsqrt * hc_scale + tl.load(hc_base_ptr + 2)) + hc_eps
        )
        pre_mix3 = (
            tl.sigmoid(mix3 * rsqrt * hc_scale + tl.load(hc_base_ptr + 3)) + hc_eps
        )

    out_t_base = pid_t * out_stride_t
    for h_start in range(0, H, BLOCK_K):
        h_off = h_start + tl.arange(0, BLOCK_K)
        h_mask = h_off < H

        r0 = tl.load(hs_ptr + hs_t_base + 0 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        r1 = tl.load(hs_ptr + hs_t_base + 1 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        out_vals = pre_mix0 * r0 + pre_mix1 * r1

        if HC > 2:
            r2 = tl.load(hs_ptr + hs_t_base + 2 * H + h_off, mask=h_mask, other=0.0).to(
                tl.float32
            )
            r3 = tl.load(hs_ptr + hs_t_base + 3 * H + h_off, mask=h_mask, other=0.0).to(
                tl.float32
            )
            out_vals += pre_mix2 * r2 + pre_mix3 * r3

        tl.store(out_ptr + out_t_base + h_off, out_vals, mask=h_mask)


def hc_head_fused_kernel_ref(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    if hs_flat.shape[0] == 0:
        return out
    x = hs_flat.reshape(hs_flat.shape[0], hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())

    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
    result = torch.sum(pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32), dim=1).to(
        out.dtype
    )
    out.copy_(result)
    return out


def hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    assert hs_flat.dtype in [torch.float32, torch.float16, torch.bfloat16]
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    num_tokens = hs_flat.shape[0]
    if num_tokens == 0:
        return out

    assert hs_flat.shape == (num_tokens, hc_mult, hidden_size)
    assert fn.shape == (hc_mult, hc_mult * hidden_size)
    assert hc_scale.shape == (1,)
    assert hc_base.shape == (hc_mult,)
    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == hs_flat.dtype
    assert hc_mult in (2, 4)

    if hs_flat.dtype == torch.float32:
        return hc_head_fused_kernel_ref(
            hs_flat,
            fn,
            hc_scale,
            hc_base,
            out,
            hidden_size,
            rms_eps,
            hc_eps,
            hc_mult,
        )

    if hs_flat.device.type != "cuda":
        return hc_head_fused_kernel_ref(
            hs_flat,
            fn,
            hc_scale,
            hc_base,
            out,
            hidden_size,
            rms_eps,
            hc_eps,
            hc_mult,
        )

    hs_flat_c = hs_flat.contiguous()
    fn_c = fn.contiguous()
    out_c = out.contiguous()

    _hc_head_apply_pre_mix_kernel[(num_tokens,)](
        hs_flat_c,
        fn_c,
        hc_scale,
        hc_base,
        out_c,
        num_tokens,
        hidden_size,
        hc_mult * hidden_size,
        rms_eps,
        hc_eps,
        hs_flat_c.stride(0),
        fn_c.stride(0),
        out_c.stride(0),
        HC=hc_mult,
    )

    if out.data_ptr() != out_c.data_ptr():
        out.copy_(out_c)
    return out
