import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_k = tle.program_id(1)
    pid_m = tle.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced

        # specialization does not improve performance inn this example, as tested
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cidv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(
            output_ptr.dtype.element_ty
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            # it is possible that there are -inf's in the input
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        # specialize the last iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            )
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)


# ------------------------  backward -------------------------------
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_non_inner"),
    key=[
        "M",
        "N",
        "K",
    ],
)
@triton.heuristics(runtime.get_heuristic_config("softmax_backward_non_inner"))
@triton.jit
def softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.sum(out_tile * out_grad_tile, axis=0)
        in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            scale += out_tile * out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale = tl.sum(scale, axis=0)  # (TILE_K)

        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_inner"),
    key=["M", "N"],
)
@triton.heuristics(
    values=runtime.get_heuristic_config("softmax_backward_inner"),
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_last"
            ).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first"
            ).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N


# -------------------------- op generate --------------------------
@triton.jit
def _softmax_kernel(
    x_ptr, out_ptr,
    # input sizes (unused in kernel, kept for symmetry/extension)
    s0, s1, s2, s3, s4, s5, s6, s7,
    # input strides (elements)
    is0, is1, is2, is3, is4, is5, is6, is7,
    # output strides (elements)
    os0, os1, os2, os3, os4, os5, os6, os7,
    # collapsed row products (elements)
    cp0, cp1, cp2, cp3, cp4, cp5, cp6, cp7,
    ndims, dim, K, stride_k_in, stride_k_out,
    BLOCK_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # row id in the collapsed view
    rid = pid.to(tl.int64)

    # Compute base offsets for this row for input and output
    base_in = tl.zeros([1], dtype=tl.int64)
    base_out = tl.zeros([1], dtype=tl.int64)
    base = rid

    # Unravel 'rid' into multi-index over dims != dim, and build input/output offsets
    for d in tl.static_range(0, 8):
        # select per-d scalars
        if d == 0:
            cp = cp0; istr = is0; ostr = os0
        elif d == 1:
            cp = cp1; istr = is1; ostr = os1
        elif d == 2:
            cp = cp2; istr = is2; ostr = os2
        elif d == 3:
            cp = cp3; istr = is3; ostr = os3
        elif d == 4:
            cp = cp4; istr = is4; ostr = os4
        elif d == 5:
            cp = cp5; istr = is5; ostr = os5
        elif d == 6:
            cp = cp6; istr = is6; ostr = os6
        else:
            cp = cp7; istr = is7; ostr = os7

        # masks for valid dimension and not the reduction dim
        m_valid = tl.where(ndims > d, 1, 0)
        m_proc = m_valid * tl.where(dim != d, 1, 0)

        # Compute index for this dimension
        # Ensure cp >= 1 to avoid div by zero, cp is provided as 1 for invalid dims.
        idx_d = (base // cp) * m_proc
        base = base - idx_d * cp

        base_in += (idx_d.to(tl.int64) * istr)
        base_out += (idx_d.to(tl.int64) * ostr)

    # Vector of K indices for this row
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    # Compute row-wise max in float32 for numerical stability
    x_ptrs = x_ptr + base_in + offs.to(tl.int64) * stride_k_in
    x_vals = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    x_vals_f32 = x_vals.to(tl.float32)
    row_max = tl.max(x_vals_f32, axis=0)

    # Compute denominator: sum(exp(x - max))
    x_vals2 = tl.load(x_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    num = tl.exp(x_vals2 - row_max)
    denom = tl.sum(num, axis=0)

    # Compute softmax and store
    y = num / denom
    out_ptrs = out_ptr + base_out + offs.to(tl.int64) * stride_k_out
    tl.store(out_ptrs, y.to(OUT_DTYPE), mask=mask)


def _torch_dtype_to_tl_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float64:
        return tl.float64
    raise ValueError(f"Unsupported output dtype for softmax: {dtype}")


def _normalize_dim(dim: int, ndims: int) -> int:
    if dim < 0:
        dim += ndims
    if not (0 <= dim < ndims):
        raise IndexError(f"dim {dim} out of range for tensor with {ndims} dims")
    return dim


def _infer_out_dtype(x: torch.Tensor, dtype: torch.dtype | None) -> torch.dtype:
    if dtype is not None:
        return dtype
    # Follow PyTorch convention: if input is floating, keep dtype, else promote to float32
    if x.dtype.is_floating_point:
        return x.dtype
    return torch.float32


def _prepare_meta(t: torch.Tensor, dim: int):
    shape = t.shape
    ndims = len(shape)
    if ndims > 8:
        raise NotImplementedError("softmax Triton kernel supports up to 8 dimensions")
    dim = _normalize_dim(dim, ndims)

    sizes = list(shape)
    in_strides = list(t.stride())  # in elements
    # Collapsed row products cp[d] = product of sizes[j] for j > d and j != dim
    cp = [1] * ndims
    for d in range(ndims):
        p = 1
        for j in range(d + 1, ndims):
            if j == dim:
                continue
            p *= sizes[j]
        cp[d] = max(p, 1)
    # Pad to 8
    def pad_list(lst, val, n=8):
        return lst + [val] * (n - len(lst))
    sizes = pad_list(sizes, 1)
    in_strides = pad_list(in_strides, 0)
    cp = pad_list(cp, 1)

    K = sizes[dim]
    stride_k_in = in_strides[dim]

    return dim, ndims, sizes, in_strides, cp, K, stride_k_in


def _launch_softmax(x: torch.Tensor, out: torch.Tensor, dim: int):
    assert x.is_cuda and out.is_cuda, "Tensors must be on CUDA device"
    assert x.shape == out.shape, "Input and output must have the same shape"
    # Meta for input
    dim, ndims, sizes, in_strides, cp, K, stride_k_in = _prepare_meta(x, dim)
    # Meta for output strides
    out_strides = list(out.stride())
    out_strides = out_strides + [0] * (8 - len(out_strides))
    stride_k_out = out_strides[dim]

    # Choose BLOCK_K = next power of two of K (cap at 4096)
    if K <= 0:
        return out
    bk = 1 << (K - 1).bit_length()
    BLOCK_K = min(bk, 4096)
    # num_warps heuristic
    num_warps = 4 if BLOCK_K <= 1024 else 8

    # Triton dtype for output
    tl_out_dtype = _torch_dtype_to_tl_dtype(out.dtype)

    # Number of rows M = product of sizes excluding dim
    M = 1
    for i, s in enumerate(sizes[:len(out.shape)]):
        if i == dim:
            continue
        M *= s

    grid = (M,)

    _softmax_kernel[grid](
        x, out,
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], sizes[7],
        in_strides[0], in_strides[1], in_strides[2], in_strides[3],
        in_strides[4], in_strides[5], in_strides[6], in_strides[7],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3],
        out_strides[4], out_strides[5], out_strides[6], out_strides[7],
        cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7],
        ndims, dim, K, stride_k_in, stride_k_out,
        BLOCK_K=BLOCK_K,
        OUT_DTYPE=tl_out_dtype,
        num_warps=num_warps,
    )
    return out


def softmax_int(self: torch.Tensor, dim: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    # print("这就是 softmax_int")
    out_dtype = _infer_out_dtype(self, dtype)
    out = torch.empty_like(self, dtype=out_dtype, device=self.device)
    return _launch_softmax(self, out, dim)


def softmax_Dimname(self: torch.Tensor, dim: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    print("这就是 softmax_Dimname")
    # Resolve named dim to index if available
    if not hasattr(self, 'names') or self.names is None:
        raise RuntimeError("Tensor has no names; cannot resolve Dimname")
    try:
        dim_index = self.names.index(dim)  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"Invalid dimension name: {dim}") from e
    out_dtype = _infer_out_dtype(self, dtype)
    out = torch.empty_like(self, dtype=out_dtype, device=self.device)
    return _launch_softmax(self, out, dim_index)


def softmax_int_out(self: torch.Tensor, dim: int, dtype: torch.dtype | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
    print("这就是 softmax_int_out")
    out_dtype = _infer_out_dtype(self, dtype)
    if out is None:
        out = torch.empty_like(self, dtype=out_dtype, device=self.device)
    else:
        if out.device != self.device:
            raise RuntimeError("out must be on the same device as input")
        if out.shape != self.shape:
            raise RuntimeError("out shape must match input shape")
        if out.dtype != out_dtype:
            raise RuntimeError(f"out dtype {out.dtype} does not match expected {out_dtype}")
    return _launch_softmax(self, out, dim)


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS SOFTMAX")
    print("这就是 原始的 softmax")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]  # pre_dim
    self = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(self, dtype=dtype)
    K = self.numel() // M // N  # post_dim

    with torch_device_fn.device(self.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_kernel_non_inner[grid](
                out,
                self,
                M,
                N,
                K,
            )
        else:
            grid = (M, 1, 1)
            softmax_kernel_inner[grid](
                out,
                self,
                M,
                N,
            )
    return out


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS SOFTMAX VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_backward_kernel_non_inner[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                K,
            )
        else:
            grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
            softmax_backward_kernel_inner[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
            )
    return in_grad
