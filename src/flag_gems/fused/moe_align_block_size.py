import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate
from triton.language.core import constexpr
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp


logger = logging.getLogger(__name__)


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def next_power_of_2(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


@triton.jit(do_not_specialize=["numel", "tokens_per_thread"])
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_expert: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    # Unroll loop by 4 for better instruction-level parallelism
    UNROLL: tl.constexpr = 4
    num_full_iters = tokens_per_thread // UNROLL

    for iter_idx in range(num_full_iters):
        base_i = iter_idx * UNROLL
        for unroll_i in range(UNROLL):
            i = base_i + unroll_i
            if start_idx + i < numel:
                idx = tl.load(topk_ids_ptr + start_idx + i)
                token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
                tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)

    for i in range(num_full_iters * UNROLL, tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2_vec(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    offset = tl.arange(0, num_experts) + 1
    token_cnt = tl.load(tokens_cnts_ptr + offset * num_experts + pid)
    cnt = tl.cumsum(token_cnt, axis=0)
    tl.store(tokens_cnts_ptr + offset * num_experts + pid, cnt)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    off_cnt = num_experts * num_experts

    expert_offsets = tl.arange(0, num_experts)
    token_cnts = tl.load(tokens_cnts_ptr + off_cnt + expert_offsets)
    aligned_cnts = tl.cdiv(token_cnts, block_size) * block_size

    cumsum_values = tl.cumsum(aligned_cnts, axis=0)
    tl.store(cumsum_ptr + 1 + expert_offsets, cumsum_values)

    total_tokens = tl.sum(aligned_cnts, axis=0)
    tl.store(total_tokens_post_pad_ptr, total_tokens)


@triton.jit(do_not_specialize=["numel"])
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    offset = tl.arange(0, tokens_per_thread) + start_idx
    mask = offset < numel
    expert_id = tl.load(topk_ids_ptr + offset, mask=mask)
    token_idx_in_expert = tl.atomic_add(
        tokens_cnts_ptr + off_t + expert_id, 1, mask=mask
    )
    rank_post_pad = token_idx_in_expert + tl.load(cumsum_ptr + expert_id, mask=mask)
    tl.store(sorted_token_ids_ptr + rank_post_pad, offset, mask=mask)


@gluon.jit(do_not_specialize=["numel", "tokens_per_thread"])
def moe_align_block_size_kernel(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: gl.constexpr,
    numel,
    numel_sorted_token_ids: gl.constexpr,
    numel_expert_ids: gl.constexpr,
    block_size_sorted: gl.constexpr,
    block_size_expert: gl.constexpr,
    tokens_per_thread,
    cumsum_ptr,
    block_size: gl.constexpr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    total_tokens_post_pad_ptr,
    sync_point_ptr,
):
    pid = gl.program_id(0)

    # allocate shared memory
    tokens_cnts_shared = gl.allocate_shared_memory(gl.int32, [num_experts, num_experts], mbarrier.MBarrierLayout())
    sync_point = gl.allocate_shared_memory(gl.int32, [1,], mbarrier.MBarrierLayout())



    # stage 1 # --------------------------------------------------------------------------

    if True:
        # init load_layout
        load_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
        # fill sorted_token_ids_ptr
        offsets_sorted = pid * block_size_sorted + gl.arange(0, block_size_sorted, layout=gl.SliceLayout(1, load_layout))
        mask_sorted = offsets_sorted < numel_sorted_token_ids
        gl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)
        # fill expert_ids_ptr
        offsets_expert = pid * block_size_expert + gl.arange(0, block_size_expert, layout=gl.SliceLayout(1, load_layout))
        mask_expert = offsets_expert < numel_expert_ids
        gl.store(expert_ids_ptr + offsets_expert, 0, mask=mask_expert)

        start_idx_0 = pid * tokens_per_thread
        off_c = (pid + 1) * num_experts
        for i in range(tokens_per_thread):
            if start_idx_0 + i < numel:
                idx = gl.load(topk_ids_ptr + start_idx_0 + i)
                token_cnt_0 = gl.load(tokens_cnts_ptr + off_c + idx)
                gl.store(tokens_cnts_ptr + off_c + idx, token_cnt_0 + 1)

    # stage 2
    # sync # --------------------------------------------------------------------------
    sync_offset = gl.arange(0, 1, layout=None)
    cp.async_copy_global_to_shared(sync_point, sync_point_ptr + sync_offset)
    cp.commit_group()
    cp.wait_group(0)

    stage1 = True
    if True:
        while stage1:
            stage1 = False

            last_cnt = 0
            for i in range(1, num_experts + 1):
                token_cnt_1 = gl.load(tokens_cnts_ptr + i * num_experts + pid)
                last_cnt = last_cnt + token_cnt_1
                gl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)

    # stage 3
    # sync # --------------------------------------------------------------------------
    cp.async_copy_global_to_shared(sync_point, sync_point_ptr + sync_offset)
    cp.commit_group()
    cp.wait_group(0)

    stage2 = True
    if True:
        while stage2:
            stage2 = False

            last_cumsum = 0
            off_cnt = num_experts * num_experts
            for i in range(1, num_experts + 1):
                token_cnt_2 = gl.load(tokens_cnts_ptr + off_cnt + i - 1)
                last_cumsum = last_cumsum + gl.cdiv(token_cnt_2, block_size) * block_size
                gl.store(cumsum_ptr + i, last_cumsum)
            gl.store(total_tokens_post_pad_ptr, last_cumsum)

    # stage 4
    # sync # --------------------------------------------------------------------------
    cp.async_copy_global_to_shared(sync_point, sync_point_ptr + sync_offset)
    cp.commit_group()
    cp.wait_group(0)

    stage3 = True
    if True:
        while stage3:
            stage3 = False

            start_idx_1 = gl.load(cumsum_ptr + pid)
            end_idx = gl.load(cumsum_ptr + pid + 1)

            for i in range(start_idx_1, end_idx, block_size):
                gl.store(expert_ids_ptr + i // block_size, pid)

            start_idx_1 = pid * tokens_per_thread
            off_t = pid * num_experts
            for i in range(start_idx_1, gl.minimum(start_idx_1 + tokens_per_thread, numel)):
                expert_id = gl.load(topk_ids_ptr + i)
                token_cnt_3 = gl.load(tokens_cnts_ptr + off_t + expert_id)
                rank_post_pad = token_cnt_3 + gl.load(cumsum_ptr + expert_id)
                gl.store(sorted_token_ids_ptr + rank_post_pad, i)
                gl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt_3 + 1)






def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    numel_sorted_token_ids = sorted_token_ids.numel()
    numel_expert_ids = expert_ids.numel()
    # The tensor needs to be padded before calculating IDs,
    # to prevent out-of-bounds address access.

    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = triton.next_power_of_2(ceil_div(numel, num_experts))

    block_size_sorted = triton.next_power_of_2(
        ceil_div(numel_sorted_token_ids, num_experts)
    )
    block_size_expert = triton.next_power_of_2(ceil_div(numel_expert_ids, num_experts))

    numel_sorted_token_ids = expert_ids.numel()
    numel_expert_ids = expert_ids.numel()

    block_size_sorted = next_power_of_2(ceil_div(numel, num_experts))
    block_size_expert = next_power_of_2(ceil_div(numel_expert_ids, num_experts))\

    sync_point = torch.zeros((1,), dtype=torch.int32, device=topk_ids.device)

    moe_align_block_size_kernel[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        numel_sorted_token_ids,
        numel_expert_ids,
        block_size_sorted,
        block_size_expert,
        tokens_per_thread,
        cumsum,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        sync_point,
    )

    # moe_align_block_size_stage1[grid](
    #     topk_ids,
    #     tokens_cnts,
    #     num_experts,
    #     numel,
    #     tokens_per_thread,
    # )
    # moe_align_block_size_stage2[grid](
    #     tokens_cnts,
    #     num_experts,
    # )
    # moe_align_block_size_stage3[(1,)](
    #     num_tokens_post_pad,
    #     tokens_cnts,
    #     cumsum,
    #     num_experts,
    #     block_size,
    # )
    # moe_align_block_size_stage4[grid](
    #     topk_ids,
    #     sorted_token_ids,
    #     expert_ids,
    #     tokens_cnts,
    #     cumsum,
    #     num_experts,
    #     block_size,
    #     numel,
    #     tokens_per_thread,
    # )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    logger.debug("GEMS MOE ALIGN BLOCK SIZE")
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad