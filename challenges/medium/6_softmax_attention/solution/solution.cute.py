"""Tensor Core tiled attention for contiguous FP32 Q/K/V and FP32 output.

Target: nvidia-cutlass-dsl==4.4.1, SM80+; 1 <= d <= 128 and M, N >= 1.

References:
  NVIDIA/cutlass: examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py
  Dao-AILab/flash-attention: flash_attn/cute/softmax.py
  NVIDIA/cutlass: examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm

The MMA, swizzled shared layouts, ldmatrix copies and accumulator views follow
CUTLASS's Ampere attention example. Online softmax keeps unscaled row maxima
and lane-local sums in FP32, as in FlashAttention.

Adaptation for this challenge's FP32 interface and 1e-4 tolerance: split each
FP32 operand into BF16 high and residual components and use three Tensor Core
products per GEMM, adapting CUTLASS FastFP32's three-product approach to BF16.
This omits the residual-residual product; it is an FP32
approximation, not an exact FP32 GEMM. It avoids a single FP16 conversion's
range loss and the rounding error that fails even the supplied basic example.
QK^T and PV both accumulate in FP32, without a global score workspace.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp

M_BLOCK_SIZE = 64
N_BLOCK_SIZE = 64
NUM_THREADS = 128
WARP_SIZE = 32
NUM_WARPS = NUM_THREADS // WARP_SIZE
LOG2_E = 1.4426950408889634

# CuTe naming: m/g/s/r/c/p = full/global/shared/register/coordinate/predicate.
# tQ/tK/tV = input copy partitions; tS/tO = score/output MMA partitions;
# taccO = output accumulator copy partitions. Suffix _lo marks the residual.


@cute.jit
def reshape_acc_to_mn(acc: cute.Tensor):
    # SM80 accumulator: ((2, 2), MMA_M, MMA_N) -> ((2, MMA_M), (2, MMA_N)).
    layout = cute.make_layout(acc.shape)
    layout_mn = cute.make_layout(
        ((layout.shape[0][1], layout.shape[1]), (layout.shape[0][0], layout.shape[2])),
        stride=(
            (layout.stride[0][1], layout.stride[1]),
            (layout.stride[0][0], layout.stride[2]),
        ),
    )
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, layout_mn))


@cute.jit
def reshape_acc_to_frgA(acc: cute.Tensor):
    # Reinterpret C's N pairs as A's K=16 operand, with no lane shuffles.
    layout = cute.logical_divide(acc.layout, (None, None, 2))
    layout_A = cute.make_layout(
        ((layout.shape[0], layout.shape[2][0]), layout.shape[1], layout.shape[2][1]),
        stride=((layout.stride[0], layout.stride[2][0]), layout.stride[1], layout.stride[2][1]),
    )
    return cute.make_tensor(acc.iterator, layout_A)


@cute.jit
def convert_to_bf16x2(src: cute.Tensor):
    rHi = cute.make_fragment_like(src, cutlass.BFloat16)
    rLo = cute.make_fragment_like(src, cutlass.BFloat16)
    rHi.store(src.load().to(cutlass.BFloat16))
    rLo.store((src.load() - rHi.load().to(cutlass.Float32)).to(cutlass.BFloat16))
    return rHi, rLo


@cute.jit
def gemm_bf16x3(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrA_lo: cute.Tensor,
    tCrB: cute.Tensor,
    tCrB_lo: cute.Tensor,
):
    # A @ B ~= A_hi @ B_lo + A_lo @ B_hi + A_hi @ B_hi.
    # Accumulate the smaller cross terms first. The low-low product is omitted.
    for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB_lo[None, None, k], acc)
        cute.gemm(tiled_mma, acc, tCrA_lo[None, None, k], tCrB[None, None, k], acc)
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def load_smem_operand(tiled_copy: cute.TiledCopy, src: cute.Tensor, dst: cute.Tensor):
    for k in cutlass.range_constexpr(cute.size(src, mode=[2])):
        cute.copy(tiled_copy, src[None, None, k], dst[None, None, k])


@cute.jit
def make_copy_predicate(coords: cute.Tensor, seqlen: cutlass.Int32, head_dim: cutlass.Int32):
    pred = cute.make_rmem_tensor(coords.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(coords)):
        m, k = coords[i]
        pred[i] = m < seqlen and k < head_dim
    return pred


@cute.jit
def load_gmem_tile(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    dst_lo: cute.Tensor,
    pred: cute.Tensor,
):
    # Predicated global -> register -> shared copies zero-fill padded entries.
    rmem = cute.make_rmem_tensor(src.shape, cutlass.Float32)
    rmem.fill(0.0)
    cute.copy(tiled_copy, src, rmem, pred=pred)
    rHi, rLo = convert_to_bf16x2(rmem)
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16)
    cute.copy(copy_atom, rHi, dst)
    cute.copy(copy_atom, rLo, dst_lo)


@cute.kernel
def attention_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    head_dim: cutlass.Int32,
):
    head_dim_padded = cute.ceil_div(mQ.shape[1], 32) * 32
    tidx, _, _ = cute.arch.thread_idx()
    m_block, _, _ = cute.arch.block_idx()
    softmax_scale_log2 = cute.math.rsqrt(cutlass.Float32(head_dim), fastmath=True) * LOG2_E

    smem = cutlass.utils.SmemAllocator()
    # 64-byte atoms support all padded head dimensions and avoid the DSL 4.4.1
    # lowering failure of the 128-byte swizzle with converted FP32 staging.
    smem_k_block_size = 32
    swizzle_bits = 2
    smem_layout_atom = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 3, 3),
        0,
        cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
    )
    sQ_layout = cute.tile_to_shape(smem_layout_atom, (M_BLOCK_SIZE, head_dim_padded), (0, 1))
    sKV_layout = cute.tile_to_shape(smem_layout_atom, (N_BLOCK_SIZE, head_dim_padded), (0, 1))
    # The FP32 epilogue buffer also holds both BF16 planes of Q in the mainloop.
    sO = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((M_BLOCK_SIZE, head_dim_padded), stride=(head_dim_padded, 1)),
        byte_alignment=128,
    )
    sQ_ptr = cute.recast_ptr(sO.iterator, dtype=cutlass.BFloat16)
    sQ = cute.make_tensor(sQ_ptr, sQ_layout)
    sQ_lo = cute.make_tensor(sQ_ptr + cute.cosize(sQ_layout), sQ_layout)
    sK = smem.allocate_tensor(cutlass.BFloat16, sKV_layout, byte_alignment=128)
    sK_lo = smem.allocate_tensor(cutlass.BFloat16, sKV_layout, byte_alignment=128)
    # K and V have disjoint lifetimes; CTA barriers protect their shared storage.
    sV, sV_lo = sK, sK_lo
    transpose_layout = cute.make_layout((head_dim_padded, N_BLOCK_SIZE), stride=(N_BLOCK_SIZE, 1))
    sVt = cute.composition(sV, transpose_layout)
    sVt_lo = cute.composition(sV_lo, transpose_layout)

    # Global -> shared copy partitions, including Q's bounds predicate.
    gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32),
        cute.make_layout((NUM_WARPS, WARP_SIZE), stride=(WARP_SIZE, 1)),
        cute.make_layout((1, 1)),
    )
    gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
    gQ = cute.local_tile(mQ, (M_BLOCK_SIZE, head_dim_padded), (m_block, 0))
    cQ = cute.local_tile(
        cute.make_identity_tensor(mQ.shape), (M_BLOCK_SIZE, head_dim_padded), (m_block, 0)
    )
    tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
    tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
    tQpQ = make_copy_predicate(tQcQ, seqlen_q, head_dim)
    tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
    tQsQ_lo = gmem_thr_copy_QKV.partition_D(sQ_lo)
    tKsK = gmem_thr_copy_QKV.partition_D(sK)
    tKsK_lo = gmem_thr_copy_QKV.partition_D(sK_lo)
    tVsV = gmem_thr_copy_QKV.partition_D(sV)
    tVsV_lo = gmem_thr_copy_QKV.partition_D(sV_lo)

    # The SM80 warp MMA and tiling follow CUTLASS's Ampere attention example.
    tiled_mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
        (NUM_WARPS, 1, 1),
        permutation_mnk=(NUM_WARPS * 16, 16, 16),
    )
    thr_mma = tiled_mma.get_slice(tidx)
    acc_shape_S = thr_mma.partition_shape_C((M_BLOCK_SIZE, N_BLOCK_SIZE))
    acc_shape_O = thr_mma.partition_shape_C((M_BLOCK_SIZE, head_dim_padded))
    acc_S = thr_mma.make_fragment_C(acc_shape_S)
    acc_O = thr_mma.make_fragment_C(acc_shape_O)
    acc_O.fill(0.0)
    acc_S_mn = reshape_acc_to_mn(acc_S)
    acc_O_mn = reshape_acc_to_mn(acc_O)
    cS = cute.make_identity_tensor((M_BLOCK_SIZE, N_BLOCK_SIZE))
    tScS = thr_mma.partition_C(cS)
    tScS_mn = reshape_acc_to_mn(tScS)
    num_rows = cute.size(acc_S_mn, mode=[0])
    num_cols = cute.size(acc_S_mn, mode=[1])
    tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
    tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
    tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

    tSrQ_lo = cute.make_fragment_like(tSrQ)
    tSrK_lo = cute.make_fragment_like(tSrK)
    tOrVt_lo = cute.make_fragment_like(tOrVt)

    # ldmatrix.x4 for Q/K, ldmatrix.trans.x4 for the transposed V operand.
    smem_copy_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16
    )
    smem_copy_atom_V = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.BFloat16
    )
    smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)
    smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
    smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
    smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
    smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
    smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
    tSsQ = smem_thr_copy_Q.partition_S(sQ)
    tSsQ_lo = smem_thr_copy_Q.partition_S(sQ_lo)
    tSsK = smem_thr_copy_K.partition_S(sK)
    tSsK_lo = smem_thr_copy_K.partition_S(sK_lo)
    tOsVt = smem_thr_copy_V.partition_S(sVt)
    tOsVt_lo = smem_thr_copy_V.partition_S(sVt_lo)
    tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
    tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
    tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)
    tSrQ_lo_copy_view = smem_thr_copy_Q.retile(tSrQ_lo)
    tSrK_lo_copy_view = smem_thr_copy_K.retile(tSrK_lo)
    tOrVt_lo_copy_view = smem_thr_copy_V.retile(tOrVt_lo)

    # Per-row online softmax state.
    row_max = cute.make_rmem_tensor(num_rows, cutlass.Float32)
    row_sum = cute.make_rmem_tensor(num_rows, cutlass.Float32)
    row_max.fill(-cutlass.Float32.inf)
    row_sum.fill(0.0)

    load_gmem_tile(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQsQ_lo, tQpQ)
    cute.arch.sync_threads()
    # Q is invariant across the K/V tiles. Keep its MMA operand in registers.
    load_smem_operand(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view)
    load_smem_operand(smem_tiled_copy_Q, tSsQ_lo, tSrQ_lo_copy_view)

    for n_block in range(cute.ceil_div(seqlen_k, N_BLOCK_SIZE)):
        row_scale = cute.make_fragment_like(row_max, cutlass.Float32)
        gK = cute.local_tile(mK, (N_BLOCK_SIZE, head_dim_padded), (n_block, 0))
        cKV = cute.local_tile(
            cute.make_identity_tensor(mK.shape),
            (N_BLOCK_SIZE, head_dim_padded),
            (n_block, 0),
        )
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
        tKVpKV = make_copy_predicate(tKVcKV, seqlen_k, head_dim)
        load_gmem_tile(gmem_tiled_copy_QKV, tKgK, tKsK, tKsK_lo, tKVpKV)
        cute.arch.sync_threads()

        acc_S.fill(0.0)
        load_smem_operand(smem_tiled_copy_K, tSsK, tSrK_copy_view)
        load_smem_operand(smem_tiled_copy_K, tSsK_lo, tSrK_lo_copy_view)
        gemm_bf16x3(tiled_mma, acc_S, tSrQ, tSrQ_lo, tSrK, tSrK_lo)

        # Online softmax follows flash_attn/cute/softmax.py: unscaled row_max,
        # exponentiated acc_S, and register-resident row_sum / row_scale.
        # For each row: m' = max(m, max(S)), alpha = exp((m - m') * scale),
        # l' = alpha * l + sum(exp((S - m') * scale)); O' = alpha * O + P @ V.
        # Here S is unscaled and exp2 uses scale * log2(e).
        # SM80's m16n8 accumulator distributes each row across four lanes.
        # Keep row_sum local to each lane and reduce it only in the epilogue.
        for r in cutlass.range_constexpr(num_rows):
            for c in cutlass.range_constexpr(num_cols):
                if n_block * N_BLOCK_SIZE + tScS_mn[r, c][1] >= seqlen_k:
                    acc_S_mn[r, c] = -cutlass.Float32.inf
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            for offset in [2, 1]:
                row_max_cur = cute.arch.fmax(
                    row_max_cur, cute.arch.shuffle_sync_bfly(row_max_cur, offset)
                )
            row_max_prev = row_max[r]
            row_max_cur = cute.arch.fmax(row_max_prev, row_max_cur)
            row_scale[r] = cute.math.exp2(
                (row_max_prev - row_max_cur) * softmax_scale_log2, fastmath=True
            )
            acc_S_row_exp = cute.math.exp2(
                (acc_S_row - row_max_cur) * softmax_scale_log2, fastmath=True
            )
            acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, 0.0, 0)
            row_sum[r] = row_sum[r] * row_scale[r] + acc_S_row_sum
            row_max[r] = row_max_cur
            acc_S_mn[r, None].store(acc_S_row_exp)
        gV = cute.local_tile(mV, (N_BLOCK_SIZE, head_dim_padded), (n_block, 0))
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        # Every warp must finish reading K before its storage is reused for V.
        cute.arch.sync_threads()
        load_gmem_tile(gmem_tiled_copy_QKV, tVgV, tVsV, tVsV_lo, tKVpKV)
        cute.arch.sync_threads()
        for r in cutlass.range_constexpr(num_rows):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])

        rP, rP_lo = convert_to_bf16x2(acc_S)
        tOrP = reshape_acc_to_frgA(rP)
        tOrP_lo = reshape_acc_to_frgA(rP_lo)
        load_smem_operand(smem_tiled_copy_V, tOsVt, tOrVt_copy_view)
        load_smem_operand(smem_tiled_copy_V, tOsVt_lo, tOrVt_lo_copy_view)
        gemm_bf16x3(tiled_mma, acc_O, tOrP, tOrP_lo, tOrVt, tOrVt_lo)
        cute.arch.sync_threads()

    for r in cutlass.range_constexpr(num_rows):
        row_sum_cur = row_sum[r]
        for offset in [2, 1]:
            row_sum_cur += cute.arch.shuffle_sync_bfly(row_sum_cur, offset)
        acc_O_mn[r, None].store(acc_O_mn[r, None].load() / row_sum_cur)

    # Redistribute MMA accumulators through shared memory to the coalesced
    # tiled-copy layout used for the final global store.
    rO = cute.make_fragment_like(acc_O, cutlass.Float32)
    rO.store(acc_O.load())
    smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
    smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
    smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
    taccOrO = smem_thr_copy_O.retile(rO)
    taccOsO = smem_thr_copy_O.partition_D(sO)
    cute.copy(smem_tiled_copy_O, taccOrO, taccOsO)
    cute.arch.sync_threads()
    gO = cute.local_tile(mO, (M_BLOCK_SIZE, head_dim_padded), (m_block, 0))
    cO = cute.local_tile(
        cute.make_identity_tensor(mO.shape), (M_BLOCK_SIZE, head_dim_padded), (m_block, 0)
    )
    # The output uses the same copy layout, with its own operand names.
    gmem_tiled_copy_O = gmem_tiled_copy_QKV
    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
    tOsO = gmem_thr_copy_O.partition_S(sO)
    tOgO = gmem_thr_copy_O.partition_D(gO)
    tOcO = gmem_thr_copy_O.partition_D(cO)
    tOpO = make_copy_predicate(tOcO, seqlen_q, head_dim)
    tOrO = cute.make_rmem_tensor(tOgO.shape, cutlass.Float32)
    cute.copy(gmem_tiled_copy_O, tOsO, tOrO)
    cute.copy(gmem_tiled_copy_O, tOrO, tOgO, pred=tOpO)


@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    output: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    d: cutlass.Int32,
):
    attention_kernel(Q, K, V, output, M, N, d).launch(
        grid=[cute.ceil_div(M, M_BLOCK_SIZE), 1, 1], block=[NUM_THREADS, 1, 1]
    )
