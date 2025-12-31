import logging
from typing import Optional

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(src):
    return src


def _can_use_triton(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        # Preserve PyTorch's behaviour of warning when casting complex to real
        # by forcing the redispatch path, which issues the warning internally.
        return False
    return True


def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS COPY (functional)")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


# def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
#     if not isinstance(src, torch.Tensor):
#         raise TypeError("src must be a Tensor")

#     # this is the same as PyTorch's check
#     if dst._is_zerotensor():
#         raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
#     if src._is_zerotensor():
#         return dst.zero_()

#     if torch._C._is_alias_of(dst, src):
#         # Align with PyTorch: if metadata fully matches, this is a no-op.
#         if (
#             dst.storage_offset() == src.storage_offset()
#             and dst.stride() == src.stride()
#             and dst.size() == src.size()
#             and dst.dtype == src.dtype
#             and dst.device == src.device
#             and dst.is_conj() == src.is_conj()
#             and dst.is_neg() == src.is_neg()
#         ):
#             return dst
#         # Otherwise defer to PyTorch for well-defined semantics on overlapping writes.
#         return torch.ops.aten.copy_.default.redispatch(
#             _FALLBACK_KEYSET, dst, src, non_blocking
#         )

#     if not _can_use_triton(dst, src):
#         return torch.ops.aten.copy_.default.redispatch(
#             _FALLBACK_KEYSET, dst, src, non_blocking
#         )

#     if dst.numel() == 0:
#         # Respect PyTorch behaviour: empty tensors should still validate broadcast.
#         return torch.ops.aten.copy_.default.redispatch(
#             _FALLBACK_KEYSET, dst, src, non_blocking
#         )

#     logger.debug("GEMS COPY_")

#     try:
#         broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
#     except RuntimeError as exc:
#         raise RuntimeError(str(exc)) from exc

#     if torch.Size(broadcast_shape) != dst.shape:
#         raise RuntimeError(
#             f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
#         )

#     expanded_src = _expand_like(src, dst.shape)

#     overload = _copy_kernel.instantiate(expanded_src.ndim)
#     overload(expanded_src, out0=dst)
#     return dst


# -------------------------- op generate --------------------------
import triton.language as tl
@triton.jit
def copy_strided_broadcast_kernel(
    out_ptr,
    in_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    N_DIMS: tl.constexpr,
    MAX_DIMS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    OUT_SHAPE_0: tl.constexpr, OUT_STRIDE_0: tl.constexpr, IN_SHAPE_0: tl.constexpr, IN_STRIDE_0: tl.constexpr,
    OUT_SHAPE_1: tl.constexpr, OUT_STRIDE_1: tl.constexpr, IN_SHAPE_1: tl.constexpr, IN_STRIDE_1: tl.constexpr,
    OUT_SHAPE_2: tl.constexpr, OUT_STRIDE_2: tl.constexpr, IN_SHAPE_2: tl.constexpr, IN_STRIDE_2: tl.constexpr,
    OUT_SHAPE_3: tl.constexpr, OUT_STRIDE_3: tl.constexpr, IN_SHAPE_3: tl.constexpr, IN_STRIDE_3: tl.constexpr,
    OUT_SHAPE_4: tl.constexpr, OUT_STRIDE_4: tl.constexpr, IN_SHAPE_4: tl.constexpr, IN_STRIDE_4: tl.constexpr,
    OUT_SHAPE_5: tl.constexpr, OUT_STRIDE_5: tl.constexpr, IN_SHAPE_5: tl.constexpr, IN_STRIDE_5: tl.constexpr,
    OUT_SHAPE_6: tl.constexpr, OUT_STRIDE_6: tl.constexpr, IN_SHAPE_6: tl.constexpr, IN_STRIDE_6: tl.constexpr,
    OUT_SHAPE_7: tl.constexpr, OUT_STRIDE_7: tl.constexpr, IN_SHAPE_7: tl.constexpr, IN_STRIDE_7: tl.constexpr,
    OUT_SHAPE_8: tl.constexpr, OUT_STRIDE_8: tl.constexpr, IN_SHAPE_8: tl.constexpr, IN_STRIDE_8: tl.constexpr,
    OUT_SHAPE_9: tl.constexpr, OUT_STRIDE_9: tl.constexpr, IN_SHAPE_9: tl.constexpr, IN_STRIDE_9: tl.constexpr,
    OUT_SHAPE_10: tl.constexpr, OUT_STRIDE_10: tl.constexpr, IN_SHAPE_10: tl.constexpr, IN_STRIDE_10: tl.constexpr,
    OUT_SHAPE_11: tl.constexpr, OUT_STRIDE_11: tl.constexpr, IN_SHAPE_11: tl.constexpr, IN_STRIDE_11: tl.constexpr,
    OUT_SHAPE_12: tl.constexpr, OUT_STRIDE_12: tl.constexpr, IN_SHAPE_12: tl.constexpr, IN_STRIDE_12: tl.constexpr,
    OUT_SHAPE_13: tl.constexpr, OUT_STRIDE_13: tl.constexpr, IN_SHAPE_13: tl.constexpr, IN_STRIDE_13: tl.constexpr,
    OUT_SHAPE_14: tl.constexpr, OUT_STRIDE_14: tl.constexpr, IN_SHAPE_14: tl.constexpr, IN_STRIDE_14: tl.constexpr,
    OUT_SHAPE_15: tl.constexpr, OUT_STRIDE_15: tl.constexpr, IN_SHAPE_15: tl.constexpr, IN_STRIDE_15: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    linear = offsets

    out_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    in_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    for rk in tl.static_range(0, MAX_DIMS):
        if rk < N_DIMS:
            if rk == 0:
                size_k = OUT_SHAPE_0
                out_stride_k = OUT_STRIDE_0
                in_size_k = IN_SHAPE_0
                in_stride_k = IN_STRIDE_0
            elif rk == 1:
                size_k = OUT_SHAPE_1
                out_stride_k = OUT_STRIDE_1
                in_size_k = IN_SHAPE_1
                in_stride_k = IN_STRIDE_1
            elif rk == 2:
                size_k = OUT_SHAPE_2
                out_stride_k = OUT_STRIDE_2
                in_size_k = IN_SHAPE_2
                in_stride_k = IN_STRIDE_2
            elif rk == 3:
                size_k = OUT_SHAPE_3
                out_stride_k = OUT_STRIDE_3
                in_size_k = IN_SHAPE_3
                in_stride_k = IN_STRIDE_3
            elif rk == 4:
                size_k = OUT_SHAPE_4
                out_stride_k = OUT_STRIDE_4
                in_size_k = IN_SHAPE_4
                in_stride_k = IN_STRIDE_4
            elif rk == 5:
                size_k = OUT_SHAPE_5
                out_stride_k = OUT_STRIDE_5
                in_size_k = IN_SHAPE_5
                in_stride_k = IN_STRIDE_5
            elif rk == 6:
                size_k = OUT_SHAPE_6
                out_stride_k = OUT_STRIDE_6
                in_size_k = IN_SHAPE_6
                in_stride_k = IN_STRIDE_6
            elif rk == 7:
                size_k = OUT_SHAPE_7
                out_stride_k = OUT_STRIDE_7
                in_size_k = IN_SHAPE_7
                in_stride_k = IN_STRIDE_7
            elif rk == 8:
                size_k = OUT_SHAPE_8
                out_stride_k = OUT_STRIDE_8
                in_size_k = IN_SHAPE_8
                in_stride_k = IN_STRIDE_8
            elif rk == 9:
                size_k = OUT_SHAPE_9
                out_stride_k = OUT_STRIDE_9
                in_size_k = IN_SHAPE_9
                in_stride_k = IN_STRIDE_9
            elif rk == 10:
                size_k = OUT_SHAPE_10
                out_stride_k = OUT_STRIDE_10
                in_size_k = IN_SHAPE_10
                in_stride_k = IN_STRIDE_10
            elif rk == 11:
                size_k = OUT_SHAPE_11
                out_stride_k = OUT_STRIDE_11
                in_size_k = IN_SHAPE_11
                in_stride_k = IN_STRIDE_11
            elif rk == 12:
                size_k = OUT_SHAPE_12
                out_stride_k = OUT_STRIDE_12
                in_size_k = IN_SHAPE_12
                in_stride_k = IN_STRIDE_12
            elif rk == 13:
                size_k = OUT_SHAPE_13
                out_stride_k = OUT_STRIDE_13
                in_size_k = IN_SHAPE_13
                in_stride_k = IN_STRIDE_13
            elif rk == 14:
                size_k = OUT_SHAPE_14
                out_stride_k = OUT_STRIDE_14
                in_size_k = IN_SHAPE_14
                in_stride_k = IN_STRIDE_14
            else:
                size_k = OUT_SHAPE_15
                out_stride_k = OUT_STRIDE_15
                in_size_k = IN_SHAPE_15
                in_stride_k = IN_STRIDE_15

            idx_k = tl.where(size_k > 0, linear % size_k, 0)
            linear = tl.where(size_k > 0, linear // size_k, linear)
            in_idx_k = tl.where(in_size_k == 1, 0, idx_k)

            out_off += idx_k * out_stride_k
            in_off += in_idx_k * in_stride_k

    val = tl.load(in_ptr + in_off, mask=mask, other=0)
    val = tl.cast(val, OUT_DTYPE)
    tl.store(out_ptr + out_off, val, mask=mask)


def _torch_to_triton_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float64:
        return tl.float64
    if dtype == torch.int8:
        return tl.int8
    if dtype == torch.uint8:
        return tl.uint8
    if dtype == torch.int16:
        return tl.int16
    if dtype == torch.int32:
        return tl.int32
    if dtype == torch.int64:
        return tl.int64
    if dtype == torch.bool:
        return tl.int1
    return tl.float32


def _normalize_shapes_strides_lists(out: torch.Tensor, src_shape, src_strides):
    out_shape = list(out.shape)
    out_strides = list(out.stride())

    out_dims = len(out_shape)
    src_dims = len(src_shape)
    n_dims = max(1, max(out_dims, src_dims))

    pad_out = n_dims - out_dims
    pad_src = n_dims - src_dims

    out_shape = [1] * pad_out + out_shape
    out_strides = [0] * pad_out + out_strides

    src_shape = [1] * pad_src + list(src_shape)
    src_strides = [0] * pad_src + list(src_strides)

    in_strides_bcast = [0 if src_shape[i] == 1 else int(src_strides[i]) for i in range(n_dims)]

    return n_dims, out_shape, out_strides, src_shape, in_strides_bcast


def _launch_triton_copy(out: torch.Tensor, src: torch.Tensor):
    n_dims, out_shape, out_strides, in_shape, in_strides = _normalize_shapes_strides_lists(
        out, src.shape, src.stride()
    )
    n_elements = out.numel()
    if n_elements == 0:
        return out

    out_shape_rev = list(reversed(out_shape))
    out_strides_rev = list(reversed(out_strides))
    in_shape_rev = list(reversed(in_shape))
    in_strides_rev = list(reversed(in_strides))

    MAX_DIMS = 16

    def pad(lst, val, L):
        return lst + [val] * (L - len(lst))

    out_shape_rev = pad(out_shape_rev, 1, n_dims)
    out_strides_rev = pad(out_strides_rev, 0, n_dims)
    in_shape_rev = pad(in_shape_rev, 1, n_dims)
    in_strides_rev = pad(in_strides_rev, 0, n_dims)

    out_shape_rev_full = pad(out_shape_rev, 1, MAX_DIMS)
    out_strides_rev_full = pad(out_strides_rev, 0, MAX_DIMS)
    in_shape_rev_full = pad(in_shape_rev, 1, MAX_DIMS)
    in_strides_rev_full = pad(in_strides_rev, 0, MAX_DIMS)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    kwargs = {
        'BLOCK_SIZE': BLOCK_SIZE,
        'N_DIMS': n_dims,
        'MAX_DIMS': MAX_DIMS,
        'OUT_DTYPE': _torch_to_triton_dtype(out.dtype),
    }

    for i in range(MAX_DIMS):
        kwargs[f'OUT_SHAPE_{i}'] = int(out_shape_rev_full[i])
        kwargs[f'OUT_STRIDE_{i}'] = int(out_strides_rev_full[i])
        kwargs[f'IN_SHAPE_{i}'] = int(in_shape_rev_full[i])
        kwargs[f'IN_STRIDE_{i}'] = int(in_strides_rev_full[i])

    copy_strided_broadcast_kernel[grid](
        out,
        src,
        n_elements,
        **kwargs,
    )
    return out


def _launch_copy(out: torch.Tensor, src: torch.Tensor):
    if isinstance(src, torch.Tensor) and (not out.is_cuda) and (not src.is_cuda):
        if src.dtype != out.dtype:
            src = src.to(dtype=out.dtype)
        out_np = out.numpy()
        src_np = src.numpy()
        out_np[...] = src_np
        return out

    if isinstance(src, torch.Tensor) and out.is_cuda and src.is_cuda:
        return _launch_triton_copy(out, src)

    if not isinstance(src, torch.Tensor):
        out.fill_(src)
        return out

    if out.is_cuda != src.is_cuda and src.numel() == 1:
        return _launch_copy_scalar(out, src.item())

    return out


def _launch_copy_scalar(out: torch.Tensor, scalar_value):
    if not out.is_cuda:
        out.fill_(scalar_value)
        return out

    scalar_buf = torch.empty(1, dtype=out.dtype, device=out.device)
    scalar_buf.fill_(scalar_value)
    return _launch_triton_copy(out, scalar_buf)


def copy_(self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    return _launch_copy(self, src)


def copy__Tensor(self: torch.Tensor, other: torch.Tensor):
    return _launch_copy(self, other)


def copy__int(self: torch.Tensor, other: int):
    return _launch_copy_scalar(self, other)


def copy__float(self: torch.Tensor, other: float):
    return _launch_copy_scalar(self, other)
