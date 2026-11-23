import cutlass
import cutlass.cute as cute

M_BLOCK_SIZE = 16
N_BLOCK_SIZE = 32
NUM_THREADS = 128
WARP_SIZE = 32
LOG2_E = 1.4426950408889634


@cute.jit
def make_acc_mn_view(acc: cute.Tensor) -> cute.Tensor:
    acc_layout = acc.layout
    mn_layout = cute.make_layout(
        ((acc_layout.shape[0], acc_layout.shape[1]), acc_layout.shape[2]),
        stride=((acc_layout.stride[0], acc_layout.stride[1]), acc_layout.stride[2]),
    )
    return cute.make_tensor(acc.iterator, mn_layout)


@cute.jit
def copy_qkv_tile(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    coords: cute.Tensor,
    seqlen: cutlass.Int32,
    head_dim: cutlass.Int32,
):
    pred = cute.make_rmem_tensor(coords.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(coords)):
        row, col = coords[i]
        pred[i] = row < seqlen and col < head_dim
    cute.copy(tiled_copy, src, dst, pred=pred)


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
    head_dim_padded = cute.ceil_div(mQ.shape[1], WARP_SIZE) * WARP_SIZE
    tidx, _, _ = cute.arch.thread_idx()
    m_block, _, _ = cute.arch.block_idx()
    softmax_scale_log2 = (
        cute.math.rsqrt(
            cutlass.Float32(head_dim),
            fastmath=True,
        )
        * LOG2_E
    )

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(
            (M_BLOCK_SIZE, head_dim_padded),
            stride=(head_dim_padded + 1, 1),
        ),
    )
    sKV_layout = cute.make_layout(
        (N_BLOCK_SIZE, head_dim_padded),
        stride=(head_dim_padded + 1, 1),
    )
    sK = smem.allocate_tensor(cutlass.Float32, sKV_layout)
    sV = smem.allocate_tensor(cutlass.Float32, sKV_layout)
    sP = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(
            (M_BLOCK_SIZE, N_BLOCK_SIZE),
            stride=(N_BLOCK_SIZE + 1, 1),
        ),
    )
    sVt = cute.make_tensor(
        sV.iterator,
        cute.make_layout(
            (head_dim_padded, N_BLOCK_SIZE),
            stride=(1, sKV_layout.stride[0]),
        ),
    )

    copy_thread_layout = cute.make_layout(
        (NUM_THREADS // WARP_SIZE, WARP_SIZE), stride=(WARP_SIZE, 1)
    )
    copy_value_layout = cute.make_layout((1, 1))
    gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.Float32,
            num_bits_per_copy=32,
        ),
        copy_thread_layout,
        copy_value_layout,
    )
    gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
    gQ = cute.local_tile(mQ, (M_BLOCK_SIZE, head_dim_padded), (m_block, 0))
    cQ = cute.local_tile(
        cute.make_identity_tensor(mQ.shape),
        (M_BLOCK_SIZE, head_dim_padded),
        (m_block, 0),
    )
    tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
    tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
    tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
    gK = cute.local_tile(mK, (N_BLOCK_SIZE, head_dim_padded), (None, 0))
    gV = cute.local_tile(mV, (N_BLOCK_SIZE, head_dim_padded), (None, 0))
    cKV = cute.local_tile(
        cute.make_identity_tensor(mK.shape), (N_BLOCK_SIZE, head_dim_padded), (None, 0)
    )
    tKgK = gmem_thr_copy_QKV.partition_S(gK)
    tVgV = gmem_thr_copy_QKV.partition_S(gV)
    tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
    tKsK = gmem_thr_copy_QKV.partition_D(sK)
    tVsV = gmem_thr_copy_QKV.partition_D(sV)

    tiled_mma = cute.make_tiled_mma(
        cute.make_mma_atom(cute.nvgpu.MmaUniversalOp(cutlass.Float32)),
        cute.make_layout(
            (M_BLOCK_SIZE, NUM_THREADS // M_BLOCK_SIZE, 1),
            stride=(NUM_THREADS // M_BLOCK_SIZE, 1, 0),
        ),
    )
    thr_mma = tiled_mma.get_slice(tidx)
    tSsQ = thr_mma.partition_A(sQ)
    tSsK = thr_mma.partition_B(sK)
    tSsP = thr_mma.partition_C(sP)
    tOsP = thr_mma.partition_A(sP)
    tOsVt = thr_mma.partition_B(sVt)
    cS = cute.make_identity_tensor((M_BLOCK_SIZE, N_BLOCK_SIZE))
    tScS = thr_mma.partition_C(cS)
    tScS_mn = make_acc_mn_view(tScS)
    acc_S = thr_mma.make_fragment_C(tSsP)
    acc_shape_O = thr_mma.partition_shape_C((M_BLOCK_SIZE, head_dim_padded))
    acc_O = thr_mma.make_fragment_C(acc_shape_O)
    acc_O.fill(0.0)

    acc_S_mn = make_acc_mn_view(acc_S)
    acc_O_mn = make_acc_mn_view(acc_O)
    row_max = cute.make_rmem_tensor((cute.size(acc_O_mn.shape[0]),), cutlass.Float32)
    row_max.fill(-cutlass.Float32.inf)
    row_sum = cute.make_rmem_tensor((cute.size(acc_O_mn.shape[0]),), cutlass.Float32)
    row_sum.fill(0.0)
    copy_qkv_tile(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, seqlen_q, head_dim)
    copy_qkv_tile(
        gmem_tiled_copy_QKV,
        tKgK[None, None, None, 0],
        tKsK,
        tKVcKV[None, None, None, 0],
        seqlen_k,
        head_dim,
    )
    cute.arch.cp_async_commit_group()

    num_kv_tiles = cute.ceil_div(seqlen_k, N_BLOCK_SIZE)
    for n_block in range(num_kv_tiles):
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        copy_qkv_tile(
            gmem_tiled_copy_QKV,
            tVgV[None, None, None, n_block],
            tVsV,
            tKVcKV[None, None, None, n_block],
            seqlen_k,
            head_dim,
        )
        cute.arch.cp_async_commit_group()

        acc_S.fill(0.0)
        tSrQ = thr_mma.make_fragment_A(tSsQ)
        tSrK = thr_mma.make_fragment_B(tSsK)
        for k_block in range(cute.size(tSsQ.shape[2]), unroll_full=True):
            cute.autovec_copy(tSsQ[None, None, k_block], tSrQ[None, None, k_block])
            cute.autovec_copy(tSsK[None, None, k_block], tSrK[None, None, k_block])
            cute.gemm(
                tiled_mma,
                acc_S,
                tSrQ[None, None, k_block],
                tSrK[None, None, k_block],
                acc_S,
            )

        cute.arch.sync_threads()
        next_n_block = n_block + 1
        if next_n_block < num_kv_tiles:
            copy_qkv_tile(
                gmem_tiled_copy_QKV,
                tKgK[None, None, None, next_n_block],
                tKsK,
                tKVcKV[None, None, None, next_n_block],
                seqlen_k,
                head_dim,
            )
        cute.arch.cp_async_commit_group()

        for r in cutlass.range_constexpr(cute.size(row_max)):
            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                if n_block * N_BLOCK_SIZE + tScS_mn[r, c][1] >= seqlen_k:
                    acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            for offset in (4, 2, 1):
                row_max_cur = cute.arch.fmax(
                    row_max_cur, cute.arch.shuffle_sync_bfly(row_max_cur, offset)
                )

            row_max_prev = row_max[r]
            row_max_cur = cute.arch.fmax(row_max_prev, row_max_cur)
            row_scale = cute.math.exp2(
                (row_max_prev - row_max_cur) * softmax_scale_log2, fastmath=True
            )
            acc_S_row_exp = cute.math.exp2(
                (acc_S_row - row_max_cur) * softmax_scale_log2, fastmath=True
            )
            row_sum[r] = row_sum[r] * row_scale + acc_S_row_exp.reduce(cute.ReductionOp.ADD, 0.0, 0)
            row_max[r] = row_max_cur
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * row_scale
            acc_S_mn[r, None] = acc_S_row_exp

        cute.autovec_copy(acc_S, tSsP)

        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

        tOrP = thr_mma.make_fragment_A(tOsP)
        tOrVt = thr_mma.make_fragment_B(tOsVt)
        for k_block in range(cute.size(tOsP.shape[2]), unroll_full=True):
            cute.autovec_copy(tOsP[None, None, k_block], tOrP[None, None, k_block])
            cute.autovec_copy(tOsVt[None, None, k_block], tOrVt[None, None, k_block])
            cute.gemm(
                tiled_mma,
                acc_O,
                tOrP[None, None, k_block],
                tOrVt[None, None, k_block],
                acc_O,
            )

    for r in cutlass.range_constexpr(cute.size(row_sum)):
        for offset in (4, 2, 1):
            row_sum[r] += cute.arch.shuffle_sync_bfly(row_sum[r], offset)
        acc_O_mn[r, None] = acc_O_mn[r, None].load() / row_sum[r]

    sO = sQ
    taccOsO = thr_mma.partition_C(sO)
    cute.autovec_copy(acc_O, taccOsO)
    cute.arch.sync_threads()
    gO = cute.local_tile(mO, (M_BLOCK_SIZE, head_dim_padded), (m_block, 0))
    cO = cute.local_tile(
        cute.make_identity_tensor(mO.shape), (M_BLOCK_SIZE, head_dim_padded), (m_block, 0)
    )
    gmem_tiled_copy_O = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32),
        copy_thread_layout,
        copy_value_layout,
    )
    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
    tOsO = gmem_thr_copy_O.partition_S(sO)
    tOgO = gmem_thr_copy_O.partition_D(gO)
    tOcO = gmem_thr_copy_O.partition_D(cO)
    tOpO = cute.make_rmem_tensor(tOcO.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tOcO)):
        m, k = tOcO[i]
        tOpO[i] = m < seqlen_q and k < head_dim
    tOrO = cute.make_rmem_tensor(tOgO.shape, cutlass.Float32)
    cute.autovec_copy(tOsO, tOrO)
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
