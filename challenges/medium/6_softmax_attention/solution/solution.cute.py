import cutlass
import cutlass.cute as cute

BM = 16
BN = 32
THREADS = 128
LOG2E = 1.4426950408889634


@cute.jit
def make_copy_predicate(
    coords: cute.Tensor,
    seqlen: cutlass.Int32,
    head_dim: cutlass.Int32,
):
    pred = cute.make_rmem_tensor_like(coords, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(coords)):
        m, n = coords[i]
        pred[i] = m < seqlen and n < head_dim
    return pred


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
    tidx, _, _ = cute.arch.thread_idx()
    m_block, _, _ = cute.arch.block_idx()
    head_dim_pad = cute.ceil_div(mQ.shape[1], 32) * 32
    scale_log2 = cute.math.rsqrt(cutlass.Float32(head_dim), fastmath=True) * LOG2E

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BM, head_dim_pad), stride=(head_dim_pad + 4, 1)),
    )
    sK = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BN, head_dim_pad), stride=(head_dim_pad + 4, 1)),
    )
    sV = smem.allocate_tensor(cutlass.Float32, sK.layout)
    sVt = cute.make_tensor(
        sV.iterator,
        cute.make_layout((sV.shape[1], sV.shape[0]), stride=(sV.stride[1], sV.stride[0])),
    )
    sP = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BM, BN), stride=(BN, 1)),
    )

    tiled_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), cutlass.Float32, num_bits_per_copy=32),
        cute.make_layout((4, 32), stride=(32, 1)),
        cute.make_layout((1, 1)),
    )
    thr_copy = tiled_copy.get_slice(tidx)
    tiled_mma = cute.make_tiled_mma(
        cute.make_mma_atom(cute.nvgpu.MmaUniversalOp(cutlass.Float32)),
        cute.make_layout(
            (BM, THREADS // BM, 1),
            stride=(THREADS // BM, 1, 0),
        ),
    )
    thr_mma = tiled_mma.get_slice(tidx)

    tQsQ = thr_copy.partition_D(sQ)
    tKsK = thr_copy.partition_D(sK)
    tVsV = thr_copy.partition_D(sV)
    tQsQ.fill(0.0)
    tKsK.fill(0.0)
    tVsV.fill(0.0)

    tSsQ = thr_mma.partition_A(sQ)
    tSrQ = thr_mma.make_fragment_A(tSsQ)
    tSsK = thr_mma.partition_B(sK)
    tSrK = thr_mma.make_fragment_B(tSsK)

    tSsP = thr_mma.partition_C(sP)
    tOsP = thr_mma.partition_A(sP)
    tOrP = thr_mma.make_fragment_A(tOsP)
    tOsVt = thr_mma.partition_B(sVt)
    tOrVt = thr_mma.make_fragment_B(tOsVt)

    gO = cute.local_tile(mO, (BM, head_dim_pad), (m_block, 0))
    cO = cute.local_tile(cute.make_identity_tensor(mO.shape), (BM, head_dim_pad), (m_block, 0))
    tOgO = thr_mma.partition_C(gO)
    tOcO = thr_mma.partition_C(cO)
    acc_O = thr_mma.make_fragment_C(tOgO)
    acc_O.fill(0.0)
    acc_O_mn = acc_O[0, None, None]

    acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((BM, BN)), cutlass.Float32)
    acc_S.fill(0.0)
    acc_S_mn = acc_S[0, None, None]
    tScS_mn = thr_mma.partition_C(cute.make_identity_tensor((BM, BN)))[0, None, None]

    num_rows = cute.size(acc_S_mn.shape[0])
    row_max = cute.make_rmem_tensor((num_rows,), cutlass.Float32)
    row_max.fill(-cutlass.Float32.inf)
    row_sum = cute.make_rmem_tensor_like(row_max)
    row_sum.fill(0.0)

    gQ = cute.local_tile(mQ, (BM, head_dim_pad), (m_block, 0))
    cQ = cute.local_tile(cute.make_identity_tensor(mQ.shape), (BM, head_dim_pad), (m_block, 0))
    gK = cute.local_tile(mK, (BN, head_dim_pad), (None, 0))
    gV = cute.local_tile(mV, (BN, head_dim_pad), (None, 0))
    cKV = cute.local_tile(cute.make_identity_tensor(mK.shape), (BN, head_dim_pad), (None, 0))
    tQgQ = thr_copy.partition_S(gQ)
    tQcQ = thr_copy.partition_S(cQ)
    tKgK = thr_copy.partition_S(gK)
    tVgV = thr_copy.partition_S(gV)
    tKVcKV = thr_copy.partition_S(cKV)
    cute.copy(
        tiled_copy,
        tQgQ,
        tQsQ,
        pred=make_copy_predicate(tQcQ, seqlen_q, head_dim),
    )
    cute.copy(
        tiled_copy,
        tKgK[None, None, None, 0],
        tKsK,
        pred=make_copy_predicate(tKVcKV[None, None, None, 0], seqlen_k, head_dim),
    )
    cute.arch.cp_async_commit_group()

    num_kv_tiles = cute.ceil_div(seqlen_k, BN)
    for n_block in range(num_kv_tiles):
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        cute.copy(
            tiled_copy,
            tVgV[None, None, None, n_block],
            tVsV,
            pred=make_copy_predicate(tKVcKV[None, None, None, n_block], seqlen_k, head_dim),
        )
        cute.arch.cp_async_commit_group()

        acc_S.fill(0.0)
        cute.autovec_copy(
            tSsQ[None, None, 0],
            tSrQ[None, None, 0],
        )
        cute.autovec_copy(
            tSsK[None, None, 0],
            tSrK[None, None, 0],
        )
        for k_block in cutlass.range_constexpr(cute.size(tSsQ.shape[2]), unroll_full=True):
            next_k_block = k_block + 1
            if cutlass.const_expr(next_k_block < cute.size(tSsQ.shape[2])):
                cute.autovec_copy(
                    tSsQ[None, None, next_k_block],
                    tSrQ[None, None, next_k_block],
                )
                cute.autovec_copy(
                    tSsK[None, None, next_k_block],
                    tSrK[None, None, next_k_block],
                )
            cute.gemm(
                tiled_mma,
                acc_S,
                tSrQ[None, None, k_block],
                tSrK[None, None, k_block],
                acc_S,
            )

        next_n_block = n_block + 1
        if next_n_block < num_kv_tiles:
            cute.arch.sync_threads()
            cute.copy(
                tiled_copy,
                tKgK[None, None, None, next_n_block],
                tKsK,
                pred=make_copy_predicate(
                    tKVcKV[None, None, None, next_n_block], seqlen_k, head_dim
                ),
            )
        cute.arch.cp_async_commit_group()

        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                if n_block * BN + tScS_mn[r, c][1] >= seqlen_k:
                    acc_S_mn[r, c] = -cutlass.Float32.inf

            s = acc_S_mn[r, None].load()
            row_max_cur = s.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            for offset in (4, 2, 1):
                row_max_cur = cute.arch.fmax(
                    row_max_cur,
                    cute.arch.shuffle_sync_bfly(row_max_cur, offset),
                )
            row_max_cur = cute.arch.fmax(row_max_cur, row_max[r])

            p = cute.math.exp2((s - row_max_cur) * scale_log2)
            row_sum_cur = p.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            for offset in (4, 2, 1):
                row_sum_cur = row_sum_cur + cute.arch.shuffle_sync_bfly(row_sum_cur, offset)

            alpha = cute.math.exp2((row_max[r] - row_max_cur) * scale_log2)
            row_sum[r] = row_sum[r] * alpha + row_sum_cur
            row_max[r] = row_max_cur
            acc_S_mn[r, None] = p
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * alpha

        cute.autovec_copy(acc_S, tSsP)
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

        cute.autovec_copy(
            tOsP[None, None, 0],
            tOrP[None, None, 0],
        )
        cute.autovec_copy(
            tOsVt[None, None, 0],
            tOrVt[None, None, 0],
        )
        for k_block in cutlass.range_constexpr(cute.size(tOsP.shape[2]), unroll_full=True):
            next_k_block = k_block + 1
            if cutlass.const_expr(next_k_block < cute.size(tOsP.shape[2])):
                cute.autovec_copy(
                    tOsP[None, None, next_k_block],
                    tOrP[None, None, next_k_block],
                )
                cute.autovec_copy(
                    tOsVt[None, None, next_k_block],
                    tOrVt[None, None, next_k_block],
                )
            cute.gemm(
                tiled_mma,
                acc_O,
                tOrP[None, None, k_block],
                tOrVt[None, None, k_block],
                acc_O,
            )

    cute.arch.cp_async_wait_group(0)

    for r in cutlass.range_constexpr(cute.size(row_sum)):
        acc_O_mn[r, None] = acc_O_mn[r, None].load() / row_sum[r]

    for i in cutlass.range_constexpr(cute.size(tOcO)):
        m, n = tOcO[i]
        if m < seqlen_q and n < head_dim:
            tOgO[i] = acc_O[i]


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
        grid=[cute.ceil_div(M, BM), 1, 1], block=[THREADS, 1, 1]
    )
