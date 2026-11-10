#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
// This kernel uses direct Cube instructions and explicit cross-core flags.
// The compiler's KFC message-pool initialization is unnecessary here.
#ifdef __MIX_CORE_MACRO__
#undef __MIX_CORE_MACRO__
#endif
#include "swi_glu_tiling.h"
#include "tiling_key_swi_glu.h"
#include "swi_glu_vector.h"

using namespace AscendC;

template <typename T, bool PACKED_N = false>
__aicore__ inline void CopyProjectionChunk(LocalTensor<T> a, LocalTensor<T> b,
        const GlobalTensor<T> &x, const GlobalTensor<T> &w1, const GlobalTensor<T> &w2,
        uint64_t xOffset, uint64_t wOffset, uint32_t chunk, uint32_t nk, uint32_t dim,
        uint32_t rows, uint32_t cols, uint32_t ma, uint32_t na, bool copyWeights) {
    Nd2NzParams aCopy{};
    aCopy.ndNum = 1;
    aCopy.nValue = rows;
    aCopy.dValue = nk;
    aCopy.srcDValue = dim;
    aCopy.dstNzC0Stride = ma;
    aCopy.dstNzNStride = 1;
    DataCopy(a, x[xOffset + chunk], aCopy);
    if (copyWeights) {
        Nd2NzParams bCopy = aCopy;
        bCopy.nValue = cols;
        bCopy.dstNzC0Stride = PACKED_N ? (2 * cols + 15) / 16 * 16 : 2 * na;
        DataCopy(b, w1[wOffset + chunk], bCopy);
        DataCopy(b[(PACKED_N ? cols : na) * 16], w2[wOffset + chunk], bCopy);
    }
}

template <typename T, bool PIPELINED = false, bool PREFETCH = false,
    bool SINGLE_TILE = false, uint32_t PIPE_K_STEP = 64, uint32_t STATIC_DIM = 0, uint32_t STATIC_ROWS = 0,
    bool BALANCED_TAIL = false, uint32_t STATIC_TAIL_ROWS = 0, bool COMPLETE_TAIL = false, bool LAZY_TILING = false>
__aicore__ inline void ComputeProjections(GM_ADDR input, GM_ADDR weight1,
        GM_ADDR weight2, GM_ADDR workspace, const SwiGluVectorTilingData &t, GM_ADDR tiling = nullptr) {
    if ASCEND_IS_AIC {
        constexpr bool UNIT_FLOW = !SINGLE_TILE;
        constexpr bool DOUBLE_GM = !SINGLE_TILE && (STATIC_DIM == 128 || STATIC_DIM == 256);
        constexpr bool FULL_ROW_TILE = STATIC_DIM == 256 && STATIC_ROWS == 128;
        constexpr bool ROW_PREFETCH = FULL_ROW_TILE;
        constexpr bool PACKED_N = (STATIC_DIM == 33 && STATIC_ROWS == 16) || FULL_ROW_TILE;
        const uint32_t tileRows = STATIC_ROWS == 0 ? t.tileRows : STATIC_ROWS;
        const uint32_t dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
        const uint32_t tileCols = STATIC_DIM == 0 ? t.tileCols : (STATIC_DIM < 128 ? (STATIC_DIM + 15) / 16 * 16 : 128);
        SetAtomicNone();
        if constexpr (UNIT_FLOW) SetMMRowMajor();
        SetLoadDataBoundary(0);
        SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        constexpr uint32_t K_STEP = PIPELINED ? PIPE_K_STEP : 128;
        constexpr uint32_t BUFFERS = PIPELINED ? 2 : 1;
        constexpr uint32_t L1_K_STEP = PREFETCH ? 256 : 512;
        constexpr uint32_t L1_BUFFERS = PREFETCH ? 2 : 1;
        LocalTensor<T> aL1(TPosition::A1, 0, L1_BUFFERS * tileRows * L1_K_STEP);
        LocalTensor<T> bL1(TPosition::A1, 131072, L1_BUFFERS * 2 * tileCols * L1_K_STEP);
        LocalTensor<T> aL0(TPosition::A2, 0, BUFFERS * tileRows * K_STEP);
        LocalTensor<T> bL0(TPosition::B2, 0, BUFFERS * 2 * tileCols * K_STEP);
        LocalTensor<float> cL0(TPosition::CO1, 0, 2 * tileRows * tileCols);
        GlobalTensor<T> x, w1, w2, first, second;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight2));
        const uint32_t span = tileRows * tileCols;
        const uint32_t core = GetBlockIdx();
        first.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + core * (DOUBLE_GM ? 4 : 2) * span);
        second.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + (core * (DOUBLE_GM ? 4 : 2) + 1) * span);
        const uint32_t nTiles = (dim + tileCols - 1) / tileCols;
        bool pending[2] = {false, false};
        if constexpr (PIPELINED) {
            // Each event denotes a free L0 slot. Prime both before the first
            // load; Mmad releases its slot for the next alternating load.
            SetFlag<HardEvent::M_MTE1>(3);
            SetFlag<HardEvent::M_MTE1>(4);
        }
        if constexpr (PREFETCH || ROW_PREFETCH) {
            SetFlag<HardEvent::MTE1_MTE2>(0);
            SetFlag<HardEvent::MTE1_MTE2>(1);
        }
        uint32_t cachedCol = UINT32_MAX;
        if constexpr (ROW_PREFETCH) {
            WaitFlag<HardEvent::MTE1_MTE2>(0);
            CopyProjectionChunk<T, PACKED_N>(aL1, bL1, x, w1, w2,
                static_cast<uint64_t>(core / nTiles * tileRows) * dim,
                static_cast<uint64_t>(core % nTiles * tileCols) * dim,
                0, dim, dim, tileRows, tileCols, tileRows, tileCols, true);
            SetFlag<HardEvent::MTE2_MTE1>(0);
        }
        // The large full-row class has at least four complete waves, so
        // its first X/weight prefetch does not depend on the total row count.
        // Read scheduling rows after issuing that asynchronous MTE2 work.
        const uint32_t totalRows = LAZY_TILING ?
            *reinterpret_cast<volatile __gm__ uint32_t *>(tiling) : t.rows;
        const uint32_t waveRows = t.cores / nTiles * tileRows;
        const uint32_t fullWaves = BALANCED_TAIL ? totalRows / waveRows : 0;
        const uint32_t remainingRows = BALANCED_TAIL ? totalRows - fullWaves * waveRows : 0;
        const uint32_t tailTileRows = STATIC_TAIL_ROWS != 0 ? STATIC_TAIL_ROWS :
            (BALANCED_TAIL && remainingRows != 0 ?
                ((remainingRows + t.cores / nTiles * 16 - 1) / (t.cores / nTiles * 16)) * 16 : tileRows);
        const uint32_t fullTiles = fullWaves * t.cores;
        const uint32_t tiles = BALANCED_TAIL ? fullTiles +
            ((remainingRows + tailTileRows - 1) / tailTileRows) * nTiles :
            ((totalRows + tileRows - 1) / tileRows) * nTiles;
        uint32_t iteration = 0;
        for (uint32_t tile = core; SINGLE_TILE || tile < tiles; tile += t.cores, ++iteration) {
            const bool tail = BALANCED_TAIL && tile >= fullTiles;
            const uint32_t row = tail ? fullWaves * waveRows + (tile - fullTiles) / nTiles * tailTileRows :
                tile / nTiles * tileRows;
            const uint32_t col = (FULL_ROW_TILE ? core % nTiles : tile % nTiles) * tileCols;
            const uint32_t activeTileRows = tail ? tailTileRows : tileRows;
            const uint32_t rows = COMPLETE_TAIL ? activeTileRows :
                (FULL_ROW_TILE && (!BALANCED_TAIL || (STATIC_TAIL_ROWS != 0 && !tail)) ? tileRows :
                    (totalRows - row < activeTileRows ? totalRows - row : activeTileRows));
            const uint32_t cols = FULL_ROW_TILE ? tileCols : (dim - col < tileCols ? dim - col : tileCols);
            const uint32_t ma = STATIC_TAIL_ROWS != 0 ? activeTileRows :
                (FULL_ROW_TILE && !BALANCED_TAIL ? tileRows : (STATIC_ROWS == 16 ? 16 : (rows + 15) / 16 * 16));
            const uint32_t na = (cols + 15) / 16 * 16;
            const uint32_t projectionCols = PACKED_N ? (2 * cols + 15) / 16 * 16 : 2 * na;
            const uint64_t xOffset = static_cast<uint64_t>(row) * dim;
            const uint64_t wOffset = static_cast<uint64_t>(col) * dim;
            const uint32_t rowSlot = ROW_PREFETCH ? (iteration & 1U) : 0;
            if constexpr (ROW_PREFETCH) {
                WaitFlag<HardEvent::MTE2_MTE1>(rowSlot);
            }
            if constexpr (PREFETCH) {
                for (uint32_t slot = 0; slot < 2; ++slot) {
                    const uint32_t chunk = slot * L1_K_STEP;
                    const uint32_t nk = dim - chunk < L1_K_STEP ? dim - chunk : L1_K_STEP;
                    WaitFlag<HardEvent::MTE1_MTE2>(slot);
                    CopyProjectionChunk<T, PACKED_N>(aL1[slot * tileRows * L1_K_STEP],
                        bL1[slot * 2 * tileCols * L1_K_STEP], x, w1, w2,
                        xOffset, wOffset, chunk, nk, dim, rows, cols, ma, na, true);
                    SetFlag<HardEvent::MTE2_MTE1>(slot);
                }
            }
            for (uint32_t chunk = 0; chunk < dim; chunk += L1_K_STEP) {
                const uint32_t nk = dim - chunk < L1_K_STEP ? dim - chunk : L1_K_STEP;
                const uint32_t l1k = (nk + 15) / 16 * 16;
                const uint32_t l1slot = PREFETCH ? (chunk / L1_K_STEP) % 2 : 0;
                auto currentA = aL1[ROW_PREFETCH ? rowSlot * tileRows * dim : l1slot * tileRows * L1_K_STEP];
                auto currentB = bL1[l1slot * 2 * tileCols * L1_K_STEP];
                if constexpr (!PREFETCH && !ROW_PREFETCH) {
                    if constexpr (!SINGLE_TILE) {
                        SetFlag<HardEvent::MTE1_MTE2>(0);
                        WaitFlag<HardEvent::MTE1_MTE2>(0);
                    }
                    // Reuse complete resident weights across row tiles.
                    CopyProjectionChunk<T, PACKED_N>(currentA, currentB, x, w1, w2, xOffset, wOffset,
                        chunk, nk, dim, rows, cols, ma, na,
                        dim > L1_K_STEP || cachedCol != col);
                    SetFlag<HardEvent::MTE2_MTE1>(0);
                }
                if constexpr (!ROW_PREFETCH) WaitFlag<HardEvent::MTE2_MTE1>(l1slot);
                for (uint32_t k = 0; k < l1k; k += K_STEP) {
                    const uint32_t ka = l1k - k < K_STEP ? l1k - k : K_STEP;
                    const uint32_t slot = PIPELINED ? (k / K_STEP) % 2 : 0;
                    if constexpr (PIPELINED) {
                        WaitFlag<HardEvent::M_MTE1>(3 + slot);
                    } else {
                        if constexpr (!SINGLE_TILE) {
                            SetFlag<HardEvent::M_MTE1>(3);
                            WaitFlag<HardEvent::M_MTE1>(3);
                        } else if (k != 0) {
                            SetFlag<HardEvent::M_MTE1>(3);
                            WaitFlag<HardEvent::M_MTE1>(3);
                        }
                    }
                    auto a = aL0[slot * tileRows * K_STEP];
                    auto b = bL0[slot * 2 * tileCols * K_STEP];
                    if constexpr (SINGLE_TILE && STATIC_DIM == 256) {
                        LoadData2dParams loadA{};
                        loadA.repeatTimes = ka / 16;
                        loadA.srcStride = ma / 16;
                        for (uint32_t m = 0; m < ma / 16; ++m) {
                            LoadData(a[m * ka * 16], currentA[k * ma + m * 256], loadA);
                        }
                    } else {
                        LoadData3DParamsV2<T> loadA{};
                        loadA.l1H = 1;
                        loadA.l1W = ma;
                        loadA.channelSize = l1k;
                        loadA.kExtension = ka;
                        loadA.mExtension = ma;
                        loadA.kStartPt = k;
                        loadA.strideW = 1;
                        loadA.strideH = 1;
                        loadA.filterW = 1;
                        loadA.filterH = 1;
                        loadA.dilationFilterW = 1;
                        loadA.dilationFilterH = 1;
                        LoadData(a, currentA, loadA);
                    }
                    LoadData2dParams load{};
                    load.repeatTimes = projectionCols * ka / 256;
                    load.srcStride = 1;
                    LoadData(b, currentB[k * projectionCols], load);
                    SetFlag<HardEvent::MTE1_M>(slot);
                    WaitFlag<HardEvent::MTE1_M>(slot);
                    MmadParams mad{};
                    mad.m = ma;
                    mad.n = projectionCols;
                    mad.k = ka;
                    mad.cmatrixInitVal = chunk == 0 && k == 0;
                    if constexpr (UNIT_FLOW) mad.unitFlag =
                        chunk + nk >= dim && k + ka >= l1k ? 3 : 2;
                    Mmad(cL0, a, b, mad);
                    if constexpr (PIPELINED) {
                        SetFlag<HardEvent::M_MTE1>(3 + slot);
                    }
                    if constexpr (ROW_PREFETCH) {
                        if (chunk == 0 && k == 0) {
                            const uint32_t nextTile = tile + t.cores;
                            if (nextTile < tiles) {
                                const bool nextTail = BALANCED_TAIL && nextTile >= fullTiles;
                                const uint32_t nextRow = nextTail ? fullWaves * waveRows +
                                    (nextTile - fullTiles) / nTiles * tailTileRows : nextTile / nTiles * tileRows;
                                const uint32_t nextTileRows = nextTail ? tailTileRows : tileRows;
                                const uint32_t nextRows = COMPLETE_TAIL ? nextTileRows : (totalRows - nextRow < nextTileRows ? totalRows - nextRow : nextTileRows);
                                const uint32_t nextSlot = rowSlot ^ 1;
                                WaitFlag<HardEvent::MTE1_MTE2>(nextSlot);
                                CopyProjectionChunk<T, PACKED_N>(aL1[nextSlot * tileRows * dim],
                                    bL1, x, w1, w2, static_cast<uint64_t>(nextRow) * dim, wOffset,
                                    0, dim, dim, nextRows, tileCols, STATIC_TAIL_ROWS != 0 ? nextTileRows : (nextRows + 15) / 16 * 16, tileCols, false);
                                SetFlag<HardEvent::MTE2_MTE1>(nextSlot);
                            }
                        }
                    }
                }
                if constexpr (PREFETCH) {
                    SetFlag<HardEvent::MTE1_MTE2>(l1slot);
                    const uint32_t nextChunk = chunk + 2 * L1_K_STEP;
                    if (nextChunk < dim) {
                        const uint32_t nextK = dim - nextChunk < L1_K_STEP ?
                            dim - nextChunk : L1_K_STEP;
                        WaitFlag<HardEvent::MTE1_MTE2>(l1slot);
                        CopyProjectionChunk<T, PACKED_N>(currentA, currentB, x, w1, w2, xOffset, wOffset,
                            nextChunk, nextK, dim, rows, cols, ma, na, true);
                        SetFlag<HardEvent::MTE2_MTE1>(l1slot);
                    }
                }
            }
            if constexpr (ROW_PREFETCH) SetFlag<HardEvent::MTE1_MTE2>(rowSlot);
            cachedCol = col;
            if constexpr (!UNIT_FLOW) {
                SetFlag<HardEvent::M_FIX>(0);
                WaitFlag<HardEvent::M_FIX>(0);
            }
            // Only the GM result buffer needs the vector reader's release.
            // Compute the next tile while the previous tile is consumed.
            const uint32_t outputSlot = DOUBLE_GM ? (iteration & 1U) : 0;
            if (pending[outputSlot]) CrossCoreWaitFlag(9 + 2 * outputSlot);
            FixpipeParamsV220 fix{};
            fix.nSize = PACKED_N ? projectionCols : na;
            fix.mSize = ma;
            fix.srcStride = ma;
            fix.dstStride = PACKED_N ? projectionCols : tileCols;
            fix.ndNum = PACKED_N ? 1 : 2;
            fix.srcNdStride = na * ma / 256;
            fix.dstNdStride = span;
            fix.quantPre = QuantMode_t::F322F16;
            fix.unitFlag = UNIT_FLOW ? 3 : 0;
            Fixpipe(first[outputSlot * 2 * span], cL0, fix);
            if constexpr (!SINGLE_TILE && !UNIT_FLOW) {
                SetFlag<HardEvent::FIX_M>(0);
                WaitFlag<HardEvent::FIX_M>(0);
            }
            CrossCoreSetFlag<2, PIPE_FIX>(8 + 2 * outputSlot);
            pending[outputSlot] = true;
            if constexpr (SINGLE_TILE) break;
        }
        if constexpr (!SINGLE_TILE) {
            if (pending[0]) CrossCoreWaitFlag(9);
            if constexpr (DOUBLE_GM) {
                if (pending[1]) CrossCoreWaitFlag(11);
            }
        }
        if constexpr (PIPELINED) {
            WaitFlag<HardEvent::M_MTE1>(3);
            WaitFlag<HardEvent::M_MTE1>(4);
        }
        if constexpr (PREFETCH || ROW_PREFETCH) {
            WaitFlag<HardEvent::MTE1_MTE2>(0);
            WaitFlag<HardEvent::MTE1_MTE2>(1);
        }
        if constexpr (!SINGLE_TILE) PipeBarrier<PIPE_ALL>();
        if constexpr (UNIT_FLOW) SetMMColumnMajor();
    }
}

template <typename T,bool FULL_ROW_SINGLE_VECTOR=false>
__aicore__ inline void ComputeSplitFullK(GM_ADDR input, GM_ADDR weight1,
        GM_ADDR weight2, GM_ADDR workspace, const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIC {
        constexpr uint32_t D = 256, N = 128;
        const uint32_t core = GetBlockIdx();
        const uint32_t row = core / 2 * t.tileRows;
        const uint32_t col = core % 2 * N;
        const uint32_t rows = FULL_ROW_SINGLE_VECTOR ? t.tileRows : (t.rows - row < t.tileRows ? t.rows - row : t.tileRows);
        const uint32_t ma = (rows + 15) / 16 * 16;
        const uint32_t span = t.tileRows * N;
        GlobalTensor<T> x, w1, w2, result;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight2));
        result.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + core * 2 * span);
        LocalTensor<T> aL1(TPosition::A1, 0, t.tileRows * D);
        LocalTensor<T> bL1(TPosition::A1, 131072, 2 * N * D);
        LocalTensor<T> aL0(TPosition::A2, 0, t.tileRows * D);
        LocalTensor<T> bL0(TPosition::B2, 0, N * D);
        LocalTensor<float> cL0(TPosition::CO1, 0, t.tileRows * N);
        SetAtomicNone();
        SetMMRowMajor();
        SetLoadDataBoundary(0);
        SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = rows;
        copy.dValue = D;
        copy.srcDValue = D;
        copy.dstNzC0Stride = ma;
        copy.dstNzNStride = 1;
        DataCopy(aL1, x[static_cast<uint64_t>(row) * D], copy);
        copy.nValue = N;
        copy.dstNzC0Stride = N;
        DataCopy(bL1, w1[col * D], copy);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        DataCopy(bL1[N * D], w2[col * D], copy);
        SetFlag<HardEvent::MTE2_MTE1>(1);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData3DParamsV2<T> loadA{};
        loadA.l1H = 1;
        loadA.l1W = ma;
        loadA.channelSize = D;
        loadA.kExtension = D;
        loadA.mExtension = ma;
        loadA.strideW = loadA.strideH = 1;
        loadA.filterW = loadA.filterH = 1;
        loadA.dilationFilterW = loadA.dilationFilterH = 1;
        LoadData(aL0, aL1, loadA);
        for (uint32_t projection = 0; projection < 2; ++projection) {
            if (projection != 0) {
                WaitFlag<HardEvent::MTE2_MTE1>(1);
                SetFlag<HardEvent::M_MTE1>(0);
                WaitFlag<HardEvent::M_MTE1>(0);
            }
            LoadData2dParams loadB{};
            loadB.repeatTimes = N * D / 256;
            loadB.srcStride = 1;
            LoadData(bL0, bL1[projection * N * D], loadB);
            SetFlag<HardEvent::MTE1_M>(0);
            WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams mad{};
            mad.m = ma;
            mad.n = N;
            mad.k = D;
            mad.unitFlag = 3;
            mad.cmatrixInitVal = true;
            Mmad(cL0, aL0, bL0, mad);
            FixpipeParamsV220 fix{};
            fix.nSize = N;
            fix.mSize = ma;
            fix.srcStride = ma;
            fix.dstStride = N;
            fix.ndNum = 1;
            fix.quantPre = QuantMode_t::F322F16;
            fix.unitFlag = 3;
            Fixpipe(result[projection * span], cL0, fix);
            if(projection==0) CrossCoreSetFlag<2,PIPE_FIX>(8);
            else CrossCoreSetFlag<2,PIPE_FIX>(10);
        }
        PipeBarrier<PIPE_ALL>();
        SetMMColumnMajor();
    }
}

template <typename T,bool FULL_ROW_SINGLE_VECTOR=false>
__aicore__ inline void GateSplitFullK(GM_ADDR bias1,GM_ADDR bias2,GM_ADDR output,
        GM_ADDR workspace,const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIV {
        constexpr uint32_t D=256,N=128;
        const uint32_t core=FULL_ROW_SINGLE_VECTOR?GetBlockIdx():GetBlockIdx()/GetTaskRation();
        const uint32_t row=core/2*t.tileRows;
        const uint32_t col=core%2*N;
        const uint32_t rows=FULL_ROW_SINGLE_VECTOR?t.tileRows:(t.rows-row<t.tileRows?t.rows-row:t.tileRows);
        const uint32_t halfRows=FULL_ROW_SINGLE_VECTOR?rows:(rows+GetTaskRation()-1)/GetTaskRation();
        const uint32_t begin=FULL_ROW_SINGLE_VECTOR?0:GetSubBlockIdx()*halfRows;
        const uint32_t nr=FULL_ROW_SINGLE_VECTOR?rows:(begin>=rows?0:(rows-begin<halfRows?rows-begin:halfRows));
        const uint32_t count=nr*N;
        const uint32_t span=t.tileRows*N;
        GlobalTensor<T> first,second,b1,b2,y;
        first.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*span);
        second.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*span+span);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        y.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        LocalTensor<T> a(TPosition::VECCALC,0,16384);
        LocalTensor<T> b(TPosition::VECCALC,32768,16384);
        LocalTensor<T> out(TPosition::VECCALC,65536,16384);
        LocalTensor<T> bias(TPosition::VECCALC,98304,256);
        DataCopyPadExtParams<T> noPad{false,0,0,0};
        BinaryRepeatParams broadcast{1,1,1,8,8,0};
        UnaryRepeatParams unary{1,1,8,8};
        BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();
        if(nr) CopyTensorPair(bias,N,b1[col],b2[col],N,noPad);
        CrossCoreWaitFlag(8);
        if(nr) {
            DataCopyPad(a,first[begin*N],DataCopyExtParams{1,count*2,0,0,0},noPad);
            SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            Add(a,a,bias,N,nr,broadcast);
            PipeBarrier<PIPE_V>();
            if(count>128) {
                SetMaskCount(); SetVectorMask<T,MaskMode::COUNTER>(count);
            } else SetVectorMask<T>(count);
            Muls<T,false>(out,a,static_cast<T>(-1),MASK_PLACEHOLDER,1,unary);
            PipeBarrier<PIPE_V>();
            Exp<T,false>(out,out,MASK_PLACEHOLDER,1,unary);
            PipeBarrier<PIPE_V>();
            Adds<T,false>(out,out,static_cast<T>(1),MASK_PLACEHOLDER,1,unary);
            PipeBarrier<PIPE_V>();
            Div<T,false>(out,a,out,MASK_PLACEHOLDER,1,binary);
            // B is independent of this division. Its later Add/PIPE_V fence
            // joins both operations before Mul consumes their results.
            if(count>128) SetMaskNorm();
        }
        CrossCoreWaitFlag(10);
        if(nr) {
            DataCopyPad(b,second[begin*N],DataCopyExtParams{1,count*2,0,0,0},noPad);
            SetFlag<HardEvent::MTE2_V>(1); WaitFlag<HardEvent::MTE2_V>(1);
            Add(b,b,bias[N],N,nr,broadcast);
            PipeBarrier<PIPE_V>();
            if(count>128) {
                SetMaskCount(); SetVectorMask<T,MaskMode::COUNTER>(count);
            } else SetVectorMask<T>(count);
            Mul<T,false>(out,out,b,MASK_PLACEHOLDER,1,binary);
            if(count>128) SetMaskNorm();
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(y[static_cast<uint64_t>(row+begin)*D+col],out,
                DataCopyExtParams{static_cast<uint16_t>(nr),N*2,0,(D-N)*2,0});
        }
    }
}

template <typename T,uint32_t D=256,uint32_t N=128>
__aicore__ inline void ComputeSingleRowWave(GM_ADDR input, GM_ADDR weight1,
        GM_ADDR weight2, GM_ADDR workspace, const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIC {
        const uint32_t core = GetBlockIdx();
        const uint32_t row = core / (D/N) * t.tileRows;
        const uint32_t col = core % (D/N) * N;
        const uint32_t rows = t.rows - row < t.tileRows ? t.rows - row : t.tileRows;
        const uint32_t ma = (rows + 15) / 16 * 16;
        const uint32_t span = t.tileRows * N;
        GlobalTensor<T> x, w1, w2, result;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight2));
        result.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + core * 2 * span);
        LocalTensor<T> aL1(TPosition::A1, 0, t.tileRows * D);
        LocalTensor<T> bL1(TPosition::A1, 131072, 2 * N * D);
        LocalTensor<T> aL0(TPosition::A2, 0, t.tileRows * D);
        LocalTensor<T> bL0(TPosition::B2, 0, N * D);
        LocalTensor<float> cL0(TPosition::CO1, 0, t.tileRows * N);
        SetAtomicNone();
        SetMMRowMajor();
        SetLoadDataBoundary(0);
        SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = rows;
        copy.dValue = D;
        copy.srcDValue = D;
        copy.dstNzC0Stride = ma;
        copy.dstNzNStride = 1;
        DataCopy(aL1, x[static_cast<uint64_t>(row) * D], copy);
        copy.nValue = N;
        copy.dstNzC0Stride = N;
        DataCopy(bL1, w1[col * D], copy);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        DataCopy(bL1[N * D], w2[col * D], copy);
        SetFlag<HardEvent::MTE2_MTE1>(1);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData3DParamsV2<T> loadA{};
        loadA.l1H = 1;
        loadA.l1W = ma;
        loadA.channelSize = D;
        loadA.kExtension = D;
        loadA.mExtension = ma;
        loadA.strideW = loadA.strideH = 1;
        loadA.filterW = loadA.filterH = 1;
        loadA.dilationFilterW = loadA.dilationFilterH = 1;
        LoadData(aL0, aL1, loadA);
        for (uint32_t projection = 0; projection < 2; ++projection) {
            if (projection != 0) {
                WaitFlag<HardEvent::MTE2_MTE1>(1);
                SetFlag<HardEvent::M_MTE1>(0);
                WaitFlag<HardEvent::M_MTE1>(0);
            }
            LoadData2dParams loadB{};
            loadB.repeatTimes = N * D / 256;
            loadB.srcStride = 1;
            LoadData(bL0, bL1[projection * N * D], loadB);
            SetFlag<HardEvent::MTE1_M>(0);
            WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams mad{};
            mad.m = ma;
            mad.n = N;
            mad.k = D;
            mad.unitFlag = 3;
            mad.cmatrixInitVal = true;
            Mmad(cL0, aL0, bL0, mad);
            FixpipeParamsV220 fix{};
            fix.nSize = N;
            fix.mSize = ma;
            fix.srcStride = ma;
            fix.dstStride = N;
            fix.ndNum = 1;
            fix.quantPre = QuantMode_t::F322F16;
            fix.unitFlag = 3;
            Fixpipe(result[projection * span], cL0, fix);
            if(projection==0) CrossCoreSetFlag<2,PIPE_FIX>(8);
            else CrossCoreSetFlag<2,PIPE_FIX>(10);
        }
        PipeBarrier<PIPE_ALL>();
        SetMMColumnMajor();
    }
}

template <typename T,uint32_t D=256,uint32_t N=128>
__aicore__ inline void GateSingleRowWave(GM_ADDR bias1,GM_ADDR bias2,GM_ADDR output,
        GM_ADDR workspace,const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIV {
        const uint32_t core=GetBlockIdx()/2;
        const uint32_t row=core/(D/N)*t.tileRows;
        const uint32_t col=core%(D/N)*N;
        const uint32_t rows=t.rows-row<t.tileRows?t.rows-row:t.tileRows;
        const uint32_t halfRows=(rows+1)/2;
        const uint32_t begin=GetSubBlockIdx()*halfRows;
        const uint32_t nr=begin>=rows?0:(rows-begin<halfRows?rows-begin:halfRows);
        const uint32_t count=nr*N;
        const uint32_t span=t.tileRows*N;
        GlobalTensor<T> first,second,b1,b2,y;
        first.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*span);
        second.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*span+span);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        y.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        LocalTensor<T> a(TPosition::VECCALC,0,16384);
        LocalTensor<T> b(TPosition::VECCALC,32768,16384);
        LocalTensor<T> out(TPosition::VECCALC,65536,16384);
        LocalTensor<T> bias(TPosition::VECCALC,98304,256);
        DataCopyPadExtParams<T> noPad{false,0,0,0};
        BinaryRepeatParams broadcast{1,1,1,8,8,0};
        UnaryRepeatParams unary{1,1,8,8};
        BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();
        if(nr) CopyTensorPair(bias,N,b1[col],b2[col],N,noPad);
        CrossCoreWaitFlag(8);
        if(nr) {
            DataCopyPad(a,first[begin*N],DataCopyExtParams{1,count*2,0,0,0},noPad);
            SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            Add(a,a,bias,N,nr,broadcast);
            PipeBarrier<PIPE_V>();
            if(count>128) {
                SetMaskCount(); SetVectorMask<T,MaskMode::COUNTER>(count);
            } else SetVectorMask<T>(count);
            Muls<T,false>(out,a,static_cast<T>(-1),MASK_PLACEHOLDER,1,unary);
            PipeBarrier<PIPE_V>();
            Exp<T,false>(out,out,MASK_PLACEHOLDER,1,unary);
            PipeBarrier<PIPE_V>();
            Adds<T,false>(out,out,static_cast<T>(1),MASK_PLACEHOLDER,1,unary);
            PipeBarrier<PIPE_V>();
            Div<T,false>(out,a,out,MASK_PLACEHOLDER,1,binary);
            // B is independent of this division. Its later Add/PIPE_V fence
            // joins both operations before Mul consumes their results.
            if(count>128) SetMaskNorm();
        }
        CrossCoreWaitFlag(10);
        if(nr) {
            DataCopyPad(b,second[begin*N],DataCopyExtParams{1,count*2,0,0,0},noPad);
            SetFlag<HardEvent::MTE2_V>(1); WaitFlag<HardEvent::MTE2_V>(1);
            Add(b,b,bias[N],N,nr,broadcast);
            PipeBarrier<PIPE_V>();
            if(count>128) {
                SetMaskCount(); SetVectorMask<T,MaskMode::COUNTER>(count);
            } else SetVectorMask<T>(count);
            Mul<T,false>(out,out,b,MASK_PLACEHOLDER,1,binary);
            if(count>128) SetMaskNorm();
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            if constexpr(D==N) DataCopyPad(y[static_cast<uint64_t>(row+begin)*D],out,DataCopyExtParams{1,count*2,0,0,0});
            else DataCopyPad(y[static_cast<uint64_t>(row+begin)*D+col],out,
                DataCopyExtParams{static_cast<uint16_t>(nr),N*2,0,(D-N)*2,0});
        }
    }
}

template <typename T>
__aicore__ inline void ComputeDualC0TwoWave(GM_ADDR input,GM_ADDR weight1,GM_ADDR weight2,
        GM_ADDR workspace,const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIC {
        constexpr uint32_t D=128,M=128,SPAN=M*D;
        const uint32_t core=GetBlockIdx();
        const uint32_t remaining=t.rows-t.cores*M;
        const uint32_t tailRows=(remaining+t.cores*16-1)/(t.cores*16)*16;
        const uint32_t tailStart=t.cores*M+core*tailRows;
        const uint32_t tailNr=tailStart<t.rows?(t.rows-tailStart<tailRows?t.rows-tailStart:tailRows):0;
        const bool hasTail=tailNr!=0;
        GlobalTensor<T> x,w1,w2,result;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight2));
        result.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*4*SPAN);
        LocalTensor<T> aL1(TPosition::A1,0,2*SPAN);
        LocalTensor<T> bL1(TPosition::A1,131072,2*D*D);
        LocalTensor<T> aL0(TPosition::A2,0,2*SPAN);
        LocalTensor<T> bL0(TPosition::B2,0,2*D*D);
        LocalTensor<float> cL0(TPosition::CO1,0,2*SPAN);
        SetAtomicNone(); SetMMColumnMajor();
        SetLoadDataBoundary(0); SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        Nd2NzParams copy{};
        copy.ndNum=1; copy.nValue=M; copy.dValue=D; copy.srcDValue=D;
        copy.dstNzC0Stride=M; copy.dstNzNStride=1;
        DataCopy(aL1,x[static_cast<uint64_t>(core*M)*D],copy);
        copy.nValue=D; copy.dstNzC0Stride=D;
        DataCopy(bL1,w1,copy); DataCopy(bL1[D*D],w2,copy);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        if(hasTail) {
            copy.nValue=tailNr; copy.dstNzC0Stride=(tailNr+15)/16*16;
            DataCopy(aL1[SPAN],x[static_cast<uint64_t>(tailStart)*D],copy);
            SetFlag<HardEvent::MTE2_MTE1>(1);
        }
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData2dParams loadB{};
        loadB.repeatTimes=2*D*D/256; loadB.srcStride=1;
        LoadData(bL0,bL1,loadB);
        for(uint32_t projection=0;projection<2;++projection) SetFlag<HardEvent::FIX_M>(6+projection);
        for(uint32_t wave=0;wave<(hasTail?2u:1u);++wave) {
            const uint32_t rows=wave?tailNr:M;
            const uint32_t ma=(rows+15)/16*16;
            if(wave) WaitFlag<HardEvent::MTE2_MTE1>(1);
            LoadData3DParamsV2<T> loadA{};
            loadA.l1H=1; loadA.l1W=ma; loadA.channelSize=D;
            loadA.kExtension=D; loadA.mExtension=ma;
            loadA.strideW=loadA.strideH=1;
            loadA.filterW=loadA.filterH=1;
            loadA.dilationFilterW=loadA.dilationFilterH=1;
            LoadData(aL0[wave*SPAN],aL1[wave*SPAN],loadA);
            SetFlag<HardEvent::MTE1_M>(0); WaitFlag<HardEvent::MTE1_M>(0);
            for(uint32_t projection=0;projection<2;++projection) {
                // Each C0 projection is reused only by the second wave.
                WaitFlag<HardEvent::FIX_M>(6+projection);
                MmadParams mad{};
                mad.m=ma; mad.n=D; mad.k=D; mad.cmatrixInitVal=true;
                Mmad(cL0[projection*SPAN],aL0[wave*SPAN],bL0[projection*D*D],mad);
                SetFlag<HardEvent::M_FIX>(6+projection); WaitFlag<HardEvent::M_FIX>(6+projection);
                FixpipeParamsV220 fix{};
                fix.nSize=D; fix.mSize=ma; fix.srcStride=ma; fix.dstStride=D;
                fix.ndNum=1; fix.quantPre=QuantMode_t::F322F16;
                Fixpipe(result[(2*wave+projection)*SPAN],cL0[projection*SPAN],fix);
                SetFlag<HardEvent::FIX_M>(6+projection);
                if(!wave) {
                    if(!projection) CrossCoreSetFlag<2,PIPE_FIX>(8);
                    else CrossCoreSetFlag<2,PIPE_FIX>(10);
                } else {
                    if(!projection) CrossCoreSetFlag<2,PIPE_FIX>(12);
                    else CrossCoreSetFlag<2,PIPE_FIX>(14);
                }
            }
        }
        for(uint32_t projection=0;projection<2;++projection) WaitFlag<HardEvent::FIX_M>(6+projection);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void GateDualC0TwoWave(GM_ADDR bias1,GM_ADDR bias2,GM_ADDR output,
        GM_ADDR workspace,const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIV {
        constexpr uint32_t D=128,M=128,SPAN=M*D;
        const uint32_t core=GetBlockIdx()/GetTaskRation();
        const uint32_t remaining=t.rows-t.cores*M;
        const uint32_t tailRows=(remaining+t.cores*16-1)/(t.cores*16)*16;
        GlobalTensor<T> source,b1,b2,y;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*4*SPAN);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        y.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        LocalTensor<T> bias(TPosition::VECCALC,98304,256);
        DataCopyPadExtParams<T> pad{false,0,0,0};
        CopyTensorPair(bias,D,b1,b2,D,pad);
        BinaryRepeatParams broadcast{1,1,1,8,8,0};
        UnaryRepeatParams unary{1,1,8,8};
        BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();
        const uint32_t tailStart=t.cores*M+core*tailRows;
        const bool hasTail=tailStart<t.rows;
        for(uint32_t wave=0;wave<(hasTail?2u:1u);++wave) {
            const uint32_t row=wave?tailStart:core*M;
            const uint32_t limit=wave?tailRows:M;
            const uint32_t rows=t.rows-row<limit?t.rows-row:limit;
            const uint32_t halfRows=(rows+GetTaskRation()-1)/GetTaskRation();
            const uint32_t begin=GetSubBlockIdx()*halfRows;
            const uint32_t nr=begin>=rows?0:(rows-begin<halfRows?rows-begin:halfRows);
            const uint32_t count=nr*D;
            LocalTensor<T> a(TPosition::VECCALC,wave*16384,8192);
            LocalTensor<T> b(TPosition::VECCALC,32768+wave*16384,8192);
            LocalTensor<T> out(TPosition::VECCALC,65536+wave*16384,8192);
            if(!wave) CrossCoreWaitFlag(8); else CrossCoreWaitFlag(12);
            if(nr) {
                DataCopyPad(a,source[2*wave*SPAN+begin*D],DataCopyExtParams{1,count*2,0,0,0},pad);
                SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
                Add(a,a,bias,D,nr,broadcast); PipeBarrier<PIPE_V>();
                if(count>128) {SetMaskCount(); SetVectorMask<T,MaskMode::COUNTER>(count);}
                else SetVectorMask<T>(count);
                Muls<T,false>(out,a,static_cast<T>(-1),MASK_PLACEHOLDER,1,unary); PipeBarrier<PIPE_V>();
                Exp<T,false>(out,out,MASK_PLACEHOLDER,1,unary); PipeBarrier<PIPE_V>();
                Adds<T,false>(out,out,static_cast<T>(1),MASK_PLACEHOLDER,1,unary); PipeBarrier<PIPE_V>();
                Div<T,false>(out,a,out,MASK_PLACEHOLDER,1,binary); PipeBarrier<PIPE_V>();
                if(count>128) SetMaskNorm();
            }
            if(!wave) CrossCoreWaitFlag(10); else CrossCoreWaitFlag(14);
            if(nr) {
                DataCopyPad(b,source[(2*wave+1)*SPAN+begin*D],DataCopyExtParams{1,count*2,0,0,0},pad);
                SetFlag<HardEvent::MTE2_V>(1); WaitFlag<HardEvent::MTE2_V>(1);
                Add(b,b,bias[D],D,nr,broadcast); PipeBarrier<PIPE_V>();
                if(count>128) {SetMaskCount(); SetVectorMask<T,MaskMode::COUNTER>(count);}
                else SetVectorMask<T>(count);
                Mul<T,false>(out,out,b,MASK_PLACEHOLDER,1,binary);
                if(count>128) SetMaskNorm();
                SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(y[static_cast<uint64_t>(row+begin)*D],out,DataCopyExtParams{1,count*2,0,0,0});
            }
        }
    }
}

template <typename T, bool SINGLE_TILE = false, uint32_t STATIC_DIM = 0, uint32_t STATIC_ROWS = 0,
    bool BALANCED_TAIL = false, uint32_t STATIC_TAIL_ROWS = 0, bool COMPLETE_TAIL = false, bool LAZY_TILING = false>
class GateProjections {
    static constexpr bool DOUBLE_GM = !SINGLE_TILE && (STATIC_DIM == 128 || STATIC_DIM == 256);
    static constexpr bool FULL_ROW_TILE = STATIC_DIM == 256 && STATIC_ROWS == 128;
public:
    __aicore__ inline void Init(GM_ADDR bias1, GM_ADDR bias2, GM_ADDR output,
            GM_ADDR workspace, const SwiGluVectorTilingData &data, GM_ADDR tiling = nullptr) {
        t = data;
        tilingRows = tiling;
        const uint32_t tileRows = STATIC_ROWS == 0 ? t.tileRows : STATIC_ROWS;
        const uint32_t tileCols = STATIC_DIM == 0 ? t.tileCols : (STATIC_DIM < 128 ? (STATIC_DIM + 15) / 16 * 16 : 128);
        const uint32_t span = tileRows * tileCols;
        core = GetBlockIdx() / GetTaskRation();
        sub = GetSubBlockIdx();
        first.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + core * (DOUBLE_GM ? 4 : 2) * span);
        const uint32_t secondOffset = FULL_ROW_TILE ? tileCols : (STATIC_DIM == 33 && STATIC_ROWS == 16 ? STATIC_DIM : span);
        second.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + core * (DOUBLE_GM ? 4 : 2) * span + secondOffset);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
        if constexpr (STATIC_DIM == 128 || STATIC_DIM == 256) {
            y.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        stepRows = STATIC_DIM == 256 && SINGLE_TILE ? 128 : 64;
        if constexpr (LAZY_TILING) {
            // A full-row core retains one column stripe for all waves.
            LocalTensor<T> bias(TPosition::VECCALC, 98304, 2 * tileCols);
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            const uint32_t col = core % 2 * tileCols;
            CopyTensorPair(bias, tileCols, b1[col], b2[col], tileCols, pad);
        }
    }
    __aicore__ inline void Process() {
        // A previous kernel can leave counter-mask mode active on an AIV.
        // The row-wise bias adds below use normal repeat masks.
        SetMaskNorm();
        const uint32_t tileRows = STATIC_ROWS == 0 ? t.tileRows : STATIC_ROWS;
        const uint32_t dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
        const uint32_t tileCols = STATIC_DIM == 0 ? t.tileCols : (STATIC_DIM < 128 ? (STATIC_DIM + 15) / 16 * 16 : 128);
        const uint32_t sourcePitch = FULL_ROW_TILE ? 2 * tileCols : (STATIC_DIM == 33 && STATIC_ROWS == 16 ? 80 : tileCols);
        const uint32_t nTiles = (dim + tileCols - 1) / tileCols;
        const uint32_t totalRows = LAZY_TILING ?
            *reinterpret_cast<volatile __gm__ uint32_t *>(tilingRows) : t.rows;
        const uint32_t waveRows = t.cores / nTiles * tileRows;
        const uint32_t fullWaves = BALANCED_TAIL ? totalRows / waveRows : 0;
        const uint32_t remainingRows = BALANCED_TAIL ? totalRows - fullWaves * waveRows : 0;
        const uint32_t tailTileRows = STATIC_TAIL_ROWS != 0 ? STATIC_TAIL_ROWS :
            (BALANCED_TAIL && remainingRows != 0 ?
                ((remainingRows + t.cores / nTiles * 16 - 1) / (t.cores / nTiles * 16)) * 16 : tileRows);
        const uint32_t fullTiles = fullWaves * t.cores;
        const uint32_t tiles = BALANCED_TAIL ? fullTiles +
            ((remainingRows + tailTileRows - 1) / tailTileRows) * nTiles :
            ((totalRows + tileRows - 1) / tileRows) * nTiles;
        uint32_t cachedCol = LAZY_TILING ? core % nTiles * tileCols : UINT32_MAX;
        uint32_t iteration = 0;
        for (uint32_t tile = core; SINGLE_TILE || tile < tiles; tile += t.cores, ++iteration) {
            const bool tail = BALANCED_TAIL && tile >= fullTiles;
            const uint32_t row = tail ? fullWaves * waveRows + (tile - fullTiles) / nTiles * tailTileRows :
                tile / nTiles * tileRows;
            const uint32_t col = (FULL_ROW_TILE ? core % nTiles : tile % nTiles) * tileCols;
            const uint32_t activeTileRows = tail ? tailTileRows : tileRows;
            const uint32_t rows = COMPLETE_TAIL ? activeTileRows :
                (FULL_ROW_TILE && (!BALANCED_TAIL || (STATIC_TAIL_ROWS != 0 && !tail)) ? tileRows :
                    (totalRows - row < activeTileRows ? totalRows - row : activeTileRows));
            const uint32_t cols = FULL_ROW_TILE ? tileCols : (dim - col < tileCols ? dim - col : tileCols);
            const uint32_t halfRows = COMPLETE_TAIL ? activeTileRows / GetTaskRation() :
                (FULL_ROW_TILE && (!BALANCED_TAIL || (STATIC_TAIL_ROWS != 0 && !tail)) ? 128 / GetTaskRation() : (rows + GetTaskRation() - 1) / GetTaskRation());
            const uint32_t begin = sub * halfRows;
            const uint32_t end = COMPLETE_TAIL ? begin + halfRows : (begin + halfRows < rows ? begin + halfRows : rows);
            LocalTensor<T> bias(TPosition::VECCALC, 98304, 2 * tileCols);
            DataCopyExtParams bcopy{1, cols * 2, 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            if (cachedCol != col) {
                CopyTensorPair(bias, tileCols, b1[col], b2[col], cols, pad);
                cachedCol = col;
            }
            const uint32_t inputSlot = DOUBLE_GM ? (iteration & 1U) : 0;
            const uint32_t inputOffset = inputSlot * 2 * tileRows * tileCols;
            CrossCoreWaitFlag(8 + 2 * inputSlot);
            for (uint32_t rr = begin; (FULL_ROW_TILE && COMPLETE_TAIL) || rr < end; rr += stepRows) {
                const uint32_t nr = COMPLETE_TAIL ? activeTileRows / GetTaskRation() :
                    (FULL_ROW_TILE && (!BALANCED_TAIL || (STATIC_TAIL_ROWS != 0 && !tail)) ? 128 / GetTaskRation() : (end - rr < stepRows ? end - rr : stepRows));
                const uint32_t count = nr * tileCols;
                if (rr != begin) PipeBarrier<PIPE_ALL>();
                LocalTensor<T> pair(TPosition::VECCALC, 0, 2 * stepRows * tileCols);
                auto a = pair;
                auto b = pair[FULL_ROW_TILE ? tileCols : stepRows * tileCols];
                if constexpr (FULL_ROW_TILE) {
                    DataCopyExtParams flat{1, 2 * count * 2, 0, 0, 0};
                    DataCopyPad(pair, first[inputOffset + rr * sourcePitch], flat, pad);
                } else {
                    DataCopyExtParams inputCopy{static_cast<uint16_t>(nr), cols * 2,
                        (sourcePitch - cols) * 2,
                        (tileCols - ((cols + 15) / 16) * 16) / 16, 0};
                    if (cols == tileCols) {
                        inputCopy.blockCount = 1;
                        inputCopy.blockLen = count * 2;
                    }
                    DataCopyPad(a, first[inputOffset + rr * sourcePitch], inputCopy, pad);
                    DataCopyPad(b, second[inputOffset + rr * sourcePitch], inputCopy, pad);
                }
                // One chunk belongs to each vector subcore. Once both
                // reads finish, the cube may reuse GM while activation runs.
                if constexpr (!SINGLE_TILE) CrossCoreSetFlag<2, PIPE_MTE2>(9 + 2 * inputSlot);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                a = pair;
                b = pair[FULL_ROW_TILE ? tileCols : stepRows * tileCols];
                for (uint32_t c = 0; c < cols; c += 128) {
                    uint32_t width = cols - c < 128 ? cols - c : 128;
                    BinaryRepeatParams params{1, 1, 1,
                        static_cast<uint8_t>((FULL_ROW_TILE ? 2 * tileCols : tileCols) / 16),
                        static_cast<uint8_t>((FULL_ROW_TILE ? 2 * tileCols : tileCols) / 16), 0};
                    Add(a[c], a[c], bias[c], width, nr, params);
                    Add(b[c], b[c], bias[tileCols + c], width, nr, params);
                }
                PipeBarrier<PIPE_V>();
                LocalTensor<T> out(TPosition::VECCALC, 65536, stepRows * tileCols);
                if constexpr (FULL_ROW_TILE) HalfGateInterleaved128(out, pair, nr);
                else HalfGateSwiGlu(out, a, b, count);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyExtParams outCopy{static_cast<uint16_t>(nr), cols * 2,
                    (tileCols - ((cols + 15) / 16) * 16) / 16,
                    (dim - cols) * 2, 0};
                if (cols == tileCols && cols == dim) {
                    outCopy.blockCount = 1;
                    outCopy.blockLen = count * 2;
                }
                DataCopyPad(y[static_cast<uint64_t>(row + rr) * dim + col], out, outCopy);
                if constexpr (FULL_ROW_TILE) break;
            }
            if constexpr (!SINGLE_TILE) {
                // The next input may arrive while the current output is stored.
                SetFlag<HardEvent::V_MTE2>(1);
                WaitFlag<HardEvent::V_MTE2>(1);
                SetFlag<HardEvent::MTE3_V>(1);
                WaitFlag<HardEvent::MTE3_V>(1);
            }
            if constexpr (!SINGLE_TILE && !COMPLETE_TAIL) {
                if (begin >= end) CrossCoreSetFlag<2, PIPE_MTE2>(9 + 2 * inputSlot);
            }
            if constexpr (SINGLE_TILE) break;
        }
    }
private:
    SwiGluVectorTilingData t;
    GM_ADDR tilingRows;
    uint32_t core, sub, stepRows;
    GlobalTensor<T> first, second, b1, b2, y;
};

// Complete full-K row tiles: two aligned waves use disjoint A0/B0/C0 slots.
template <typename T, uint32_t SLOT, uint16_t FLAG, bool REUSE, bool RELEASE>
__aicore__ inline void EmitProjectionTile(AscendC::LocalTensor<float> c,
        AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b,
        AscendC::GlobalTensor<T> result, uint32_t rows, uint32_t offset) {
    using namespace AscendC;
    if ASCEND_IS_AIC {
        if constexpr (REUSE) WaitFlag<HardEvent::FIX_M>(SLOT);
        MmadParams mad{};
        mad.m=rows;mad.n=128;mad.k=128;
        mad.unitFlag=3;mad.cmatrixInitVal=true;
        Mmad(c,a,b,mad);
        // Paired unitFlag3 provides MMAD-to-Fixpipe data readiness per512B block.
        FixpipeParamsV220 fix{};
        fix.nSize=128;fix.mSize=rows;fix.srcStride=rows;fix.dstStride=128;
        fix.ndNum=1;fix.quantPre=QuantMode_t::F322F16;fix.unitFlag=3;
        Fixpipe(result[offset],c,fix);
        CrossCoreSetFlag<2,PIPE_FIX>(FLAG);
        if constexpr (RELEASE) SetFlag<HardEvent::FIX_M>(SLOT);
    }
}

template <typename T>
__aicore__ inline void ComputeProjectionWaves(GM_ADDR input,GM_ADDR weight1,
        GM_ADDR weight2,GM_ADDR workspace) {
    using namespace AscendC;
    if ASCEND_IS_AIC {
        constexpr uint32_t D=128,STEP=256,ASLOT=128*D,CSLOT=128*D,SPAN=STEP*D;
        static_assert(2*ASLOT*sizeof(T)<=65536);
        static_assert(2*D*D*sizeof(T)<=65536);
        static_assert(2*CSLOT*sizeof(float)<=131072);
        const uint32_t core=GetBlockIdx(),row=core*STEP;
        // Tiling selects only completeSTEP-row tiles; both waves are full.
        constexpr uint32_t r0=128,r1=128;
        GlobalTensor<T> x,w1,w2,result;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight2));
        result.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*SPAN);
        LocalTensor<T> aL1(TPosition::A1,0,2*ASLOT);
        LocalTensor<T> bL1(TPosition::A1,131072,2*D*D);
        LocalTensor<T> aL0(TPosition::A2,0,2*ASLOT);
        LocalTensor<T> bL0(TPosition::B2,0,2*D*D);
        LocalTensor<float> cL0(TPosition::CO1,0,2*CSLOT);
        SetAtomicNone();SetMMRowMajor();SetLoadDataBoundary(0);
        SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        Nd2NzParams copy{};
        copy.ndNum=1;copy.nValue=r0;copy.dValue=D;copy.srcDValue=D;
        copy.dstNzC0Stride=r0;copy.dstNzNStride=1;
        DataCopy(aL1,x[static_cast<uint64_t>(row)*D],copy);
        copy.nValue=D;copy.dstNzC0Stride=D;
        DataCopy(bL1,w1,copy);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        DataCopy(bL1[D*D],w2,copy);
        SetFlag<HardEvent::MTE2_MTE1>(1);
        copy.nValue=r1;copy.dstNzC0Stride=r1;
        DataCopy(aL1[ASLOT],x[static_cast<uint64_t>(row+r0)*D],copy);
        SetFlag<HardEvent::MTE2_MTE1>(2);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData3DParamsV2<T> loadA{};
        loadA.l1H=1;loadA.l1W=r0;loadA.channelSize=D;
        loadA.kExtension=D;loadA.mExtension=r0;
        loadA.strideW=loadA.strideH=1;loadA.filterW=loadA.filterH=1;
        loadA.dilationFilterW=loadA.dilationFilterH=1;
        LoadData(aL0,aL1,loadA);
        LoadData2dParams loadB{};
        loadB.repeatTimes=D*D/256;loadB.srcStride=1;
        LoadData(bL0,bL1,loadB);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        EmitProjectionTile<T,0,8,false,true>(cL0,aL0,bL0,result,r0,0);
        // The upper B0 half is independent of the first MMAD's live lower half.
        WaitFlag<HardEvent::MTE2_MTE1>(1);
        LoadData(bL0[D*D],bL1[D*D],loadB);
        SetFlag<HardEvent::MTE1_M>(1);WaitFlag<HardEvent::MTE1_M>(1);
        EmitProjectionTile<T,1,10,false,true>(cL0[CSLOT],aL0,bL0[D*D],result,r0,SPAN);
        // The second A0 slot is likewise disjoint from the first row wave.
        WaitFlag<HardEvent::MTE2_MTE1>(2);
        loadA.l1W=r1;loadA.mExtension=r1;
        LoadData(aL0[ASLOT],aL1[ASLOT],loadA);
        SetFlag<HardEvent::MTE1_M>(2);WaitFlag<HardEvent::MTE1_M>(2);
        EmitProjectionTile<T,0,12,true,false>(cL0,aL0[ASLOT],bL0,result,r1,r0*D);
        EmitProjectionTile<T,1,14,true,false>(cL0[CSLOT],aL0[ASLOT],bL0[D*D],result,r1,SPAN+r0*D);
        PipeBarrier<PIPE_ALL>();SetMMColumnMajor();
    }
}

template <typename T, uint32_t WAVE>
__aicore__ inline void GateProjectionWave(AscendC::GlobalTensor<T> first,
        AscendC::GlobalTensor<T> second,AscendC::GlobalTensor<T> y,
        AscendC::LocalTensor<T> bias,uint32_t row,uint32_t begin,uint32_t nr) {
    using namespace AscendC;
    constexpr uint32_t D=128;
    const uint32_t count=nr*D;
    LocalTensor<T> a(TPosition::VECCALC,WAVE*16384,8192);
    LocalTensor<T> b(TPosition::VECCALC,32768+WAVE*16384,8192);
    LocalTensor<T> out(TPosition::VECCALC,65536+WAVE*16384,8192);
    DataCopyPadExtParams<T> pad{false,0,0,0};
    BinaryRepeatParams broadcast{1,1,1,8,8,0},binary{1,1,1,8,8,8};
    UnaryRepeatParams unary{1,1,8,8};
    CrossCoreWaitFlag(8+4*WAVE);
    DataCopyPad(a,first[begin*D],DataCopyExtParams{1,count*2,0,0,0},pad);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    Add(a,a,bias,D,nr,broadcast);PipeBarrier<PIPE_V>();
    SetMaskCount();SetVectorMask<T,MaskMode::COUNTER>(count);
    Muls<T,false>(out,a,static_cast<T>(-1),MASK_PLACEHOLDER,1,unary);PipeBarrier<PIPE_V>();
    Exp<T,false>(out,out,MASK_PLACEHOLDER,1,unary);PipeBarrier<PIPE_V>();
    Adds<T,false>(out,out,static_cast<T>(1),MASK_PLACEHOLDER,1,unary);PipeBarrier<PIPE_V>();
    Div<T,false>(out,a,out,MASK_PLACEHOLDER,1,binary);
    // B addition is independent; its fence joins with the preceding division.
    SetMaskNorm();
    CrossCoreWaitFlag(10+4*WAVE);
    DataCopyPad(b,second[begin*D],DataCopyExtParams{1,count*2,0,0,0},pad);
    SetFlag<HardEvent::MTE2_V>(1);WaitFlag<HardEvent::MTE2_V>(1);
    Add(b,b,bias[D],D,nr,broadcast);PipeBarrier<PIPE_V>();
    SetMaskCount();SetVectorMask<T,MaskMode::COUNTER>(count);
    Mul<T,false>(out,out,b,MASK_PLACEHOLDER,1,binary);SetMaskNorm();
    SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyPad(y[static_cast<uint64_t>(row+begin)*D],out,DataCopyExtParams{1,count*2,0,0,0});
}

template <typename T>
__aicore__ inline void GateProjectionWaves(GM_ADDR bias1,GM_ADDR bias2,
        GM_ADDR output,GM_ADDR workspace) {
    using namespace AscendC;
    if ASCEND_IS_AIV {
        constexpr uint32_t D=128,STEP=256,SPAN=STEP*D;
        const uint32_t core=GetBlockIdx()/2,row=core*STEP,sub=GetSubBlockIdx();
        // Tiling selects only completeSTEP-row tiles; both waves are full.
        constexpr uint32_t r0=128,r1=128;
        GlobalTensor<T> first,second,b1,b2,y;
        first.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*SPAN);
        second.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*SPAN+SPAN);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        y.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        LocalTensor<T> bias(TPosition::VECCALC,98304,256);
        DataCopyPadExtParams<T> noPad{false,0,0,0};
        SetMaskNorm();CopyTensorPair(bias,D,b1,b2,D,noPad);
        // Each AV consumes flags8/10/12/14 exactly once. The two output slots
        // are disjoint, so first-wave MTE3 may overlap the second-wave gate.
        GateProjectionWave<T,0>(first,second,y,bias,row,sub*(r0/2),r0/2);
        GateProjectionWave<T,1>(first,second,y,bias,row,r0+sub*(r1/2),r1/2);
    }
}



template <typename DT_INPUT_X, uint32_t CALC_MODE, uint32_t DIM_TAG, uint32_t TAIL_CODE, uint32_t PAD_BLOCKS, uint32_t ROW_WAVES>
__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
        GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    if ASCEND_IS_AIV { SetMaskNorm(); }
    if constexpr (CALC_MODE == 7) {
        GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, data, tiling);
        PairVectorSwiGlu<DT_INPUT_X, DIM_TAG == 1,
            DIM_TAG == 13 ? 224 : (DIM_TAG == 14 ? 256 : 0)> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data);
        op.Process();
    } else if constexpr (CALC_MODE == 6 && DIM_TAG >= 25 && DIM_TAG <= 31) {
        if (GetBlockIdx() < DIM_TAG - 23) {
            const SwiGluVectorTilingData data{7,13,7,16,1};
            BatchedVectorSwiGlu<DT_INPUT_X,7,true,false,13,true,true> op;
            op.Init(input_x,weight_1,bias_1,weight_2,bias_2,output,data,tiling);
            op.Process();
        }
    } else if constexpr (CALC_MODE == 6 && DIM_TAG >= 16 && DIM_TAG <= 22) {
        if (GetBlockIdx() < GetBlockNum() - (23 - DIM_TAG)) {
            const SwiGluVectorTilingData data{3, 13, 3, 16, 1};
            BatchedVectorSwiGlu<DT_INPUT_X, 3, true, false, 13, true, true> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
            op.Process();
        }
    } else if constexpr (CALC_MODE == 6 && DIM_TAG == 14) {
        // A complete local tile descriptor; total rows follow the launch
        // grid, with the remainder and inactive lanes selected by tiling.
        const SwiGluVectorTilingData data{8,33,8,40,1};
        BatchedVectorSwiGlu<DT_INPUT_X,8,true,false,33,true,TAIL_CODE==0,40,TAIL_CODE,true,PAD_BLOCKS> op;
        op.Init(input_x,weight_1,bias_1,weight_2,bias_2,output,data,tiling);
        op.Process();
    } else if constexpr (CALC_MODE == 6 && DIM_TAG == 13) {
        const SwiGluVectorTilingData data{2, 33, 2, 40, 1};
        BatchedVectorSwiGlu<DT_INPUT_X, 2, true, false, 33, true, false, 40, 1> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
        op.Process();
    } else if constexpr (CALC_MODE == 6 && DIM_TAG == 11) {
        const SwiGluVectorTilingData data{3, 33, 3, 40, 1};
        BatchedVectorSwiGlu<DT_INPUT_X, 3, true, false, 33, true, false, 40, 1> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
        op.Process();
    } else if constexpr (CALC_MODE == 6) {
        SwiGluVectorTilingData data;
        if constexpr (DIM_TAG == 12 || DIM_TAG == 15) {
            // Complete row groups derive their input/output offsets from block IDs.
            data = {DIM_TAG == 12 ? 1U : 3U, 13, DIM_TAG == 12 ? 1U : 3U, 16, 1};
        } else {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, dynamicData, tiling);
            data = dynamicData;
        }
        BatchedVectorSwiGlu<DT_INPUT_X, DIM_TAG == 12 ? 1 : (DIM_TAG == 15 || DIM_TAG == 23 ? 3 :
            (DIM_TAG == 7 ? 32 : (DIM_TAG == 6 ? 16 : (DIM_TAG == 3 || DIM_TAG == 9 ? 8 : 4)))),
            DIM_TAG == 2 || DIM_TAG == 3 || DIM_TAG == 5 || DIM_TAG == 6 ||
                DIM_TAG == 7 || DIM_TAG == 12 || DIM_TAG == 15 || DIM_TAG == 23,
            DIM_TAG == 4 || DIM_TAG == 5 || DIM_TAG == 17 || DIM_TAG == 18,
            DIM_TAG == 8 || DIM_TAG == 9 ? 16 : (DIM_TAG == 10 ? 32 :
                (DIM_TAG == 11 ? 48 : (DIM_TAG == 12 ? 13 :
                    (DIM_TAG == 15 ? 13 :
                        (DIM_TAG == 17 ? 112 : (DIM_TAG == 18 ? 128 :
                            (DIM_TAG == 23 ? 13 : 0))))))), DIM_TAG == 12 || DIM_TAG == 15 || DIM_TAG == 23,
            DIM_TAG == 12 || DIM_TAG == 15> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
        op.Process();
    } else if constexpr (CALC_MODE == 1 && DIM_TAG == 7) {
        if (GetBlockIdx() == 0) {
            ScalarVectorSwiGlu<DT_INPUT_X> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output);
            op.Process();
        }
    } else if constexpr (CALC_MODE == 2 && (DIM_TAG == 24 || DIM_TAG == 16)) {
        if (DIM_TAG != 16 || GetBlockIdx() == 0) {
            TinyVectorSwiGlu<DT_INPUT_X, 4> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling);
            op.Process();
        }
    } else if constexpr(CALC_MODE==3 && DIM_TAG==19) {
        if ASCEND_IS_AIC ComputeProjectionWaves<DT_INPUT_X>(input_x,weight_1,weight_2,workspace);
        if ASCEND_IS_AIV GateProjectionWaves<DT_INPUT_X>(bias_1,bias_2,output,workspace);
    } else if constexpr(CALC_MODE==3 && DIM_TAG==18) {
        constexpr uint32_t paddedTailRows = TAIL_CODE == 8 || TAIL_CODE == 10 ? 64 : (TAIL_CODE == 9 || TAIL_CODE == 11 ? 128 : 0);
        // Geometry follows generic schedule parameters, not a list of input
        // shapes. The same body is compiled for all bounded wave/tail plans.
        constexpr uint32_t scheduledRows = ROW_WAVES * (20 / 2 * 128) +
            (20 / 2 - PAD_BLOCKS) * paddedTailRows;
        SwiGluVectorTilingData data{ROW_WAVES != 0 ? scheduledRows : 0,256,128,128,20};
        if constexpr (TAIL_CODE == 0) {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, dynamicData, tiling);
            data.rows = dynamicData.rows; data.cores = dynamicData.cores;
        }
        if ASCEND_IS_AIC ComputeProjections<DT_INPUT_X,true,false,false,64,256,128,true,paddedTailRows,(TAIL_CODE==10 || TAIL_CODE==11),(TAIL_CODE!=0 && ROW_WAVES==0)>(input_x,weight_1,weight_2,workspace,data,tiling);
        if ASCEND_IS_AIV {
            GateProjections<DT_INPUT_X,false,256,128,true,paddedTailRows,(TAIL_CODE==10 || TAIL_CODE==11),(TAIL_CODE!=0 && ROW_WAVES==0)> op;
            op.Init(bias_1,bias_2,output,workspace,data,tiling); op.Process();
        }
    } else if constexpr(CALC_MODE==3 && DIM_TAG==21) {
        // This path owns one complete64-row tile perN128pair. TotalM and
        // corecount follow the launch grid, and do not specialize a test shape.
        const uint32_t cores=static_cast<uint32_t>(GetBlockNum());
        const SwiGluVectorTilingData data{cores/2*64,256,64,128,cores};
        if ASCEND_IS_AIC ComputeSplitFullK<DT_INPUT_X,true>(input_x,weight_1,weight_2,workspace,data);
        if ASCEND_IS_AIV GateSplitFullK<DT_INPUT_X,true>(bias_1,bias_2,output,workspace,data);
    } else if constexpr(CALC_MODE==3 && DIM_TAG==20) {
        GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, data, tiling);
        if ASCEND_IS_AIC ComputeDualC0TwoWave<DT_INPUT_X>(input_x,weight_1,weight_2,workspace,data);
        if ASCEND_IS_AIV GateDualC0TwoWave<DT_INPUT_X>(bias_1,bias_2,output,workspace,data);
    } else if constexpr(CALC_MODE==3 && DIM_TAG==22) {
        GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData,data,tiling);
        if ASCEND_IS_AIC {
            ComputeSplitFullK<DT_INPUT_X>(input_x,weight_1,weight_2,workspace,data);
        }
        if ASCEND_IS_AIV {
            GateSplitFullK<DT_INPUT_X>(bias_1,bias_2,output,workspace,data);
        }
    } else if constexpr(CALC_MODE==3 && DIM_TAG==23) {
        GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData,data,tiling);
        if ASCEND_IS_AIC ComputeDualC0TwoWave<DT_INPUT_X>(input_x,weight_1,weight_2,workspace,data);
        if ASCEND_IS_AIV GateDualC0TwoWave<DT_INPUT_X>(bias_1,bias_2,output,workspace,data);
    } else if constexpr (CALC_MODE == 0 || CALC_MODE >= 3) {
        // CANN passes the user region after its reserved workspace prefix.
        GM_ADDR userWorkspace = workspace;
        if ASCEND_IS_AIC {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, data, tiling);
            if constexpr (DIM_TAG >= 26 && DIM_TAG <= 29) {
                data.dim = DIM_TAG == 26 || DIM_TAG == 28 ? 128 : 256;
                data.tileCols = 128;
            }
            ComputeProjections<DT_INPUT_X, CALC_MODE == 3 || CALC_MODE == 4,
                CALC_MODE >= 4, DIM_TAG == 25 || DIM_TAG == 28 || DIM_TAG == 29 || DIM_TAG == 30 || DIM_TAG == 31, 64,
                DIM_TAG == 26 || DIM_TAG == 28 ? 128 :
                    (DIM_TAG == 14 || DIM_TAG == 17 || DIM_TAG == 27 || DIM_TAG == 29 ? 256 : ((DIM_TAG == 30 || DIM_TAG == 31) ? 33 : 0)),
                DIM_TAG == 14 || DIM_TAG == 17 ? 128 : (DIM_TAG == 31 ? 16 : 0), DIM_TAG == 17>(input_x,
                weight_1, weight_2, userWorkspace, data);
        }
        if ASCEND_IS_AIV {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, data, tiling);
            if constexpr (DIM_TAG >= 26 && DIM_TAG <= 29) {
                data.dim = DIM_TAG == 26 || DIM_TAG == 28 ? 128 : 256;
                data.tileCols = 128;
            }
            GateProjections<DT_INPUT_X, DIM_TAG == 25 || DIM_TAG == 28 || DIM_TAG == 29 || DIM_TAG == 30 || DIM_TAG == 31,
                DIM_TAG == 26 || DIM_TAG == 28 ? 128 :
                    (DIM_TAG == 14 || DIM_TAG == 17 || DIM_TAG == 27 || DIM_TAG == 29 ? 256 : ((DIM_TAG == 30 || DIM_TAG == 31) ? 33 : 0)),
                DIM_TAG == 14 || DIM_TAG == 17 ? 128 : (DIM_TAG == 31 ? 16 : 0), DIM_TAG == 17> op;
            op.Init(bias_1, bias_2, output, userWorkspace, data);
            op.Process();
        }
    } else {
        SwiGluVectorTilingData data;
        if constexpr ((DIM_TAG >= 4 && DIM_TAG <= 6) || (CALC_MODE == 1 && DIM_TAG == 7) ||
            (CALC_MODE == 2 && (DIM_TAG == 19 || DIM_TAG == 3))) {
            // Block IDs identify each row and column tile. The one-tile vector
            // classes do not need a device-side read of scheduling fields.
            constexpr uint32_t dim = CALC_MODE == 2 && DIM_TAG == 19 ? 32 : (DIM_TAG == 4 ? 3 :
                (DIM_TAG == 5 ? 13 : (DIM_TAG == 3 || DIM_TAG == 6 ? 64 : 1)));
            data = {1, dim, 128, 16,
                1U};
        } else {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, dynamicData, tiling);
            data = dynamicData;
            if constexpr (DIM_TAG >= 1 && DIM_TAG <= 3) {
                data.dim = DIM_TAG == 1 ? 3 : (DIM_TAG == 2 ? 13 : 64);
                data.tileCols = 16;
            }
        }
        if (CALC_MODE != 2 || DIM_TAG != 19 || GetBlockIdx() == 0) {
            VectorSwiGlu<DT_INPUT_X, CALC_MODE, CALC_MODE == 2 && (DIM_TAG >= 4 || DIM_TAG == 3),
                CALC_MODE == 2 && DIM_TAG == 19 ? 32 :
                    (DIM_TAG == 4 ? 3 : (DIM_TAG == 5 ? 13 : (DIM_TAG == 3 || DIM_TAG == 6 ? 64 : 0)))> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
            op.Process();
        }
    }
}

// HardSwish small-shape method: wrapper emits this call after including the
// user translation unit. Only these workspace-free vector tiling keys skip
// system-workspace initialization. Cube/all other keys retain the SDK call.
#if !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON)
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 33281UL || TILING_KEY_VAR == 6657UL || TILING_KEY_VAR == 39425UL || TILING_KEY_VAR == 14593UL || TILING_KEY_VAR == 49665UL)
namespace AscendC {
__aicore__ inline void SwiGluIgnoreUnusedSysWorkspace(GM_ADDR) {}
}
#define SetSysWorkspaceForce SwiGluIgnoreUnusedSysWorkspace
#endif
#endif
