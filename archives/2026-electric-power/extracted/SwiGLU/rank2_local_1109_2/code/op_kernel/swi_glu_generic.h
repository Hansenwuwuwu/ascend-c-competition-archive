#pragma once

#include "kernel_operator.h"
#include "swi_glu_tiling.h"

namespace swiglu {
using namespace AscendC;

class KernelFullK {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                               GM_ADDR b2, GM_ADDR y, GM_ADDR workspace, uint32_t dim, uint32_t rows,
                               TPipe *ppipe) {
        M = rows;
        N = K = dim;
        NAlign = KAlign = CeilAlign(N, 16);
        xGm.SetGlobalBuffer((__gm__ half *)x);
        w1Gm.SetGlobalBuffer((__gm__ half *)w1);
        b1Gm.SetGlobalBuffer((__gm__ half *)b1);
        w2Gm.SetGlobalBuffer((__gm__ half *)w2);
        b2Gm.SetGlobalBuffer((__gm__ half *)b2);
        yGm.SetGlobalBuffer((__gm__ half *)y);

        // 两路投影各有[core][slot=2][TILE_L1_BASEM*TILE_L1_BASEN]的Half GM空间。
        auto ws = (__gm__ half *)GetUserWorkspace(workspace);
        z1Gm.SetGlobalBuffer(ws);
        z2Gm.SetGlobalBuffer(ws + GetBlockNum() * 2 * TILE_L1_BASEM * TILE_L1_BASEN);

        pipe = ppipe;
        if ASCEND_IS_AIC {
            pipe->InitBuffer(qA1, 2, TILE_L1_BASEM * TILE_L1_BASEK * sizeof(half));
            pipe->InitBuffer(qB1, 2, TILE_L1_BASEN * TILE_L1_BASEK * sizeof(half));
            pipe->InitBuffer(qA2, 2, TILE_L1_BASEM * TILE_L2_BASEK * sizeof(half));
            pipe->InitBuffer(qB2, 2, TILE_L2_BASEN * TILE_L2_BASEK * sizeof(half));
            pipe->InitBuffer(cBuf, 2 * TILE_L1_BASEM * TILE_L1_BASEN * sizeof(float));
        }
        if ASCEND_IS_AIV {
            // 两路投影和激活均使用half，各队列保留独立的双缓冲。
            pipe->InitBuffer(q1, 2, TILE_L1_BASEN * TILE_L1_BASEM / 2 * sizeof(half));
            pipe->InitBuffer(q2, 2, TILE_L1_BASEN * TILE_L1_BASEM / 2 * sizeof(half));
            pipe->InitBuffer(qOut, 2, TILE_L1_BASEN * TILE_L1_BASEM / 2 * sizeof(half));
            pipe->InitBuffer(tmpBuf, TILE_L1_BASEM * TILE_L1_BASEN / 2 *  sizeof(half));
            pipe->InitBuffer(qBias, 1, 2 * TILE_L1_BASEN * sizeof(half));
            pipe->InitBuffer(biasBuf, 2 * TILE_L1_BASEN * TILE_L1_BASEM / 2 * sizeof(half));
        }
    }

    __aicore__ inline void Process() {
        // 核间只分M；一个A块复用两路投影的全部N块，MMAD一次覆盖完整K=D。
        uint32_t core = GetBlockIdx();
        if ASCEND_IS_AIV { core /= 2; }
        uint32_t mTiles = (M + TILE_L1_BASEM - 1) / TILE_L1_BASEM;
        uint32_t iteration = 0;
        if ASCEND_IS_AIC {
            LoadFullWeights();
            Prefetch(core * TILE_L1_BASEM, Min(TILE_L1_BASEM, M - core * TILE_L1_BASEM));
        }
        if ASCEND_IS_AIV { FullBias(); }
        for (uint32_t mt = core; mt < mTiles; mt += GetBlockNum(), ++iteration) {
            uint32_t row = mt * TILE_L1_BASEM, rows = Min(TILE_L1_BASEM, M - row);
            uint32_t slot = iteration & 1;
            if ASCEND_IS_AIC {
                if (mt + GetBlockNum() < mTiles) {
                    uint32_t nextRow = (mt + GetBlockNum()) * TILE_L1_BASEM;
                    Prefetch(nextRow, Min(TILE_L1_BASEM, M - nextRow));
                }
                auto panel = qA1.DeQue<half>();
                auto a = qA2.AllocTensor<half>();
                LoadData2DParams ld;
                ld.repeatTimes = CeilDiv(K, 16); ld.srcStride = CeilDiv(rows, 16);
                for (uint32_t m = 0; m < CeilDiv(rows, 16); ++m) {
                    ld.startIndex = m;
                    LoadData(a[m * KAlign * 16], panel, ld);
                }
                qA2.EnQue(a); qA1.FreeTensor(panel);
                a = qA2.DeQue<half>();
                LoadFullB(0, 0);
                for (uint32_t col = 0; col < N; col += TILE_L2_BASEN) {
                    CubeFullK(a, rows, col, Min(TILE_L2_BASEN, N - col));
                }
                qA2.FreeTensor(a);
                if (iteration >= 2) CrossCoreWaitFlag(slot + 2);
                StoreFullCube((core * 2 + slot) * TILE_L1_BASEM * TILE_L1_BASEN, rows);
                CrossCoreSetFlag<2, PIPE_FIX>(slot);
            }
            if ASCEND_IS_AIV { FullVector(row, rows, core, slot); }
        }
        if ASCEND_IS_AIC {
            qB1.FreeTensor(residentB);
            for (uint32_t slot = 0; slot < Min(iteration, 2); ++slot) CrossCoreWaitFlag(slot + 2);
        }
    }

private:
    // 向上整除
    __aicore__ inline uint32_t CeilDiv(uint32_t numerator, uint32_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }
    // 向上对齐
    __aicore__ inline uint32_t CeilAlign(uint32_t numerator, uint32_t denominator)
    {
        return (numerator + denominator - 1) / denominator * denominator;
    }

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

    __aicore__ inline void LoadFullWeights() {
        // 两路完整权重按FullN列块分别转NZ，一次驻留L1后复用所有M块。
        residentB = qB1.AllocTensor<half>();
        Nd2NzParams cp;
        cp.ndNum = 1; cp.dValue = K;
        cp.srcNdMatrixStride = 0; cp.srcDValue = K;
        cp.dstNzNStride = 1; cp.dstNzMatrixStride = 0;
        uint32_t span = TILE_L1_BASEN * KAlign;
        for (uint32_t col = 0; col < N; col += TILE_L2_BASEN) {
            cp.nValue = Min(TILE_L2_BASEN, N - col);
            cp.dstNzC0Stride = CeilAlign(cp.nValue, 16);
            DataCopy(residentB[col * KAlign], w1Gm[static_cast<uint64_t>(col) * K], cp);
            DataCopy(residentB[span + col * KAlign], w2Gm[static_cast<uint64_t>(col) * K], cp);
        }
        qB1.EnQue(residentB);
        residentB = qB1.DeQue<half>();
    }
    __aicore__ inline void LoadFullB(uint32_t col, uint32_t projection) {
        auto b = qB2.AllocTensor<half>();
        uint32_t nBlocks = CeilDiv(Min(TILE_L2_BASEN, N - col), 16);
        uint32_t span = TILE_L1_BASEN * KAlign;
        auto source = residentB[projection * span + col * KAlign];
        LoadData2DParams ld;
        ld.repeatTimes = CeilDiv(KAlign, 16) * nBlocks; ld.srcStride = 1;
        LoadData(b, source, ld);
        qB2.EnQue(b);
    }
    __aicore__ inline void CubeFullK(LocalTensor<half> a, uint32_t rows,
                                    uint32_t col, uint32_t cols) {
        auto c = cBuf.Get<float>();
        MmadParams mp;
        mp.m = rows; mp.n = cols; mp.k = K;
        mp.cmatrixInitVal = true; mp.unitFlag = 3;
        for (uint32_t projection = 0; projection < 2; ++projection) {
            auto b = qB2.DeQue<half>();
            // Each L0B slot holds one projection: the other slot can load ahead.
            if (projection == 0) LoadFullB(col, 1);
            else if (col + cols < N) LoadFullB(col + cols, 0);
            Mmad(c[projection * TILE_L1_BASEM * TILE_L1_BASEN + col * CeilAlign(rows, 16)], a, b, mp);
            qB2.FreeTensor(b);
        }
    }

    __aicore__ inline void StoreFullCube(uint32_t ws, uint32_t rows) {
        auto c = cBuf.Get<float>();
        FixpipeParamsV220 fp;
        fp.mSize = CeilAlign(rows, 16); fp.nSize = NAlign;
        fp.srcStride = CeilAlign(rows, 16);
        fp.dstStride = NAlign; fp.ndNum = 1; fp.unitFlag = 3;
        // Fixpipe完成第一次FP16舍入；加bias后的第二次舍入由AIV完成。
        fp.quantPre = QuantMode_t::F322F16;
        Fixpipe(z1Gm[ws], c, fp);
        Fixpipe(z2Gm[ws], c[TILE_L1_BASEM * TILE_L1_BASEN], fp);
    }

    __aicore__ inline void FullBias() {
        // Bias rows use padded K stride; real-D lanes alone are written to GM.
        auto bias = qBias.AllocTensor<half>();
        DataCopy(bias, b1Gm, NAlign);
        DataCopy(bias[NAlign], b2Gm, NAlign);
        qBias.EnQue(bias); bias = qBias.DeQue<half>();
        auto bias1 = biasBuf.Get<half>();
        auto bias2 = bias1[TILE_L1_BASEN * TILE_L1_BASEM / 2];
        for( uint32_t i = 0; i < TILE_L1_BASEM / 2; i++ ) {
            DataCopy(bias1[i * NAlign], bias, NAlign);
            DataCopy(bias2[i * NAlign], bias[NAlign], NAlign);
        }
        qBias.FreeTensor(bias);
    }

    __aicore__ inline void FullVector(uint32_t row, uint32_t rows, uint32_t core,
                                      uint32_t slot) {
        CrossCoreWaitFlag(slot);
        // 每个AIV负责BM/2行，先把至多两份BM/4行读入私有UB，再释放GM槽。
        uint32_t vecore0 = rows / 2;
        uint32_t vecore1 = rows - vecore0;
        uint32_t first = GetSubBlockIdx() * vecore0;
        uint32_t total = GetSubBlockIdx() ? vecore1 : vecore0;
        for (uint32_t part = 0; part < total; part += TILE_L1_BASEM / 4) {
            uint32_t count = Min(TILE_L1_BASEM / 4, total - part);
            uint32_t offset = (core * 2 + slot) * TILE_L1_BASEM * TILE_L1_BASEN + (first + part) * NAlign;
            auto a = q1.AllocTensor<half>();
            DataCopy(a, z1Gm[offset], count * KAlign);
            q1.EnQue(a);
            auto b = q2.AllocTensor<half>();
            DataCopy(b, z2Gm[offset], count * KAlign);
            q2.EnQue(b);
        }
        // Both chunks are in private UB before this GM slot can be overwritten.
        CrossCoreSetFlag<2, PIPE_MTE2>(slot + 2);
        auto bias1 = biasBuf.Get<half>();
        auto bias2 = bias1[TILE_L1_BASEN * TILE_L1_BASEM / 2];
        for (uint32_t part = 0; part < total; part += TILE_L1_BASEM / 4) {
            uint32_t count = Min(TILE_L1_BASEM / 4, total - part), elements = count * NAlign;
            auto a = q1.DeQue<half>();
            Add(a, a, bias1, elements);
            auto silu = tmpBuf.Get<half>();
            Silu(silu, a, elements);
            auto b = q2.DeQue<half>();
            Add(b, b, bias2, elements);
            auto out = qOut.AllocTensor<half>();
            Mul(out, silu, b, elements);
            q1.FreeTensor(a); q2.FreeTensor(b);
            qOut.EnQue(out); out = qOut.DeQue<half>();
            DataCopyPad(yGm[static_cast<uint64_t>(row + first + part) * N], out,
                        {static_cast<uint16_t>(count), N * 2, 0, 0, 0});
            qOut.FreeTensor(out);
        }
    }



    __aicore__ inline void Prefetch(uint32_t row, uint32_t rows) {
        auto a = qA1.AllocTensor<half>();
        Nd2NzParams cp;
        cp.ndNum = 1; cp.nValue = rows; cp.dValue = K;
        cp.srcNdMatrixStride = 0; cp.srcDValue = K;
        rows == 1 ? cp.dstNzC0Stride = 1 : cp.dstNzC0Stride = CeilAlign(rows, 16);
        cp.dstNzNStride = 1; cp.dstNzMatrixStride = 0;
        DataCopy(a, xGm[static_cast<uint64_t>(row) * K], cp);
        qA1.EnQue(a);
    }

    TPipe *pipe;
    TQue<TPosition::A1, 2> qA1;
    TQue<TPosition::B1, 2> qB1;
    TQue<TPosition::A2, 2> qA2;
    TQue<TPosition::B2, 2> qB2;
    TQue<TPosition::VECIN, 1> qBias;
    TBuf<TPosition::CO1> cBuf;
    TQue<TPosition::VECIN, 2> q1, q2;
    TQue<TPosition::VECOUT, 2> qOut;
    TBuf<TPosition::VECCALC> tmpBuf, biasBuf;
    LocalTensor<half> residentB;

    GlobalTensor<half> xGm, w1Gm, b1Gm, w2Gm, b2Gm, yGm;
    GlobalTensor<half> z1Gm, z2Gm;

    uint32_t M, N, K, KAlign, NAlign;
};

class KernelSwiGluSplitK {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                               GM_ADDR b2, GM_ADDR y, GM_ADDR workspace,
                               uint32_t dim, uint32_t rows, TPipe *p) {
        pipe = p;
        M = rows; D = dim;
        xGm.SetGlobalBuffer((__gm__ half *)x);
        w1Gm.SetGlobalBuffer((__gm__ half *)w1);
        w2Gm.SetGlobalBuffer((__gm__ half *)w2);
        b1Gm.SetGlobalBuffer((__gm__ half *)b1);
        b2Gm.SetGlobalBuffer((__gm__ half *)b2);
        yGm.SetGlobalBuffer((__gm__ half *)y);
        auto ws = (__gm__ half *)GetUserWorkspace(workspace);
        gateGm.SetGlobalBuffer(ws);
        upGm.SetGlobalBuffer(ws + GetBlockNum() * 2 * BM * BN);
        if ASCEND_IS_AIC {
            pipe->InitBuffer(qA1, 2, BM * BK * sizeof(half));
            pipe->InitBuffer(qB1, 2, BN * BK * sizeof(half));
            pipe->InitBuffer(qA2, 2, BM * BK * sizeof(half));
            pipe->InitBuffer(qB2, 2, BN * BK * sizeof(half));
            pipe->InitBuffer(cBuf, 4 * BM * BN * sizeof(float));
        }
        if ASCEND_IS_AIV {
            pipe->InitBuffer(qGate, 2, VM * BN * sizeof(half));
            pipe->InitBuffer(qUp, 2, VM * BN * sizeof(half));
            pipe->InitBuffer(qOut, 2, VM * BN * sizeof(half));
            pipe->InitBuffer(qBias, 1, 2 * BN * sizeof(half));
            pipe->InitBuffer(biasBuf, 2 * VM * BN * sizeof(half));
            pipe->InitBuffer(siluBuf, VM * BN * sizeof(half));
        }
    }

    __aicore__ inline void Process() {
        uint32_t core = GetBlockIdx();
        if ASCEND_IS_AIV { core /= 2; }
        uint32_t nTiles = (D + BN - 1) / BN;
        uint32_t tiles = (M + BM - 1) / BM * nTiles;
        uint32_t iteration = 0;
        for (uint32_t tile = core; tile < tiles; tile += GetBlockNum(), ++iteration) {
            uint32_t row = tile / nTiles * BM, col = tile % nTiles * BN;
            uint32_t rows = Min(BM, M - row), cols = Min(BN, D - col);
            uint32_t slot = iteration & 1;
            if ASCEND_IS_AIC { Cube(row, col, rows, cols, core, slot, iteration); }
            if ASCEND_IS_AIV { Vector(row, col, rows, cols, core, slot); }
        }
        if ASCEND_IS_AIC {
            for (uint32_t slot = 0; slot < Min(iteration, 2); ++slot) {
                CrossCoreWaitFlag(slot + 2);
            }
        }
    }

private:
    static constexpr uint32_t BM = TILE_L1_BASEM, BN = TILE_L2_BASEN;
    static constexpr uint32_t BK = TILE_L1_BASEK, VM = BM / 2;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
    __aicore__ inline uint32_t Align(uint32_t value) { return (value + 15) / 16 * 16; }

    __aicore__ inline void PadRows(LocalTensor<half> dst, uint32_t rows, uint32_t count) {
        if (rows % 16 == 0) return;
        // 只清零M/N补齐行；K尾块由ND2NZ补零。
        InitConstValueParams<half> pad;
        pad.repeatTimes = Align(count) / 16;
        pad.blockNum = Align(rows) - rows; pad.dstGap = rows; pad.initValue = 0;
        InitConstValue(dst[rows * 16], pad);
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void CopyA(uint32_t row, uint32_t k, uint32_t rows, uint32_t count) {
        auto a = qA1.AllocTensor<half>();
        PadRows(a, rows, count);
        Nd2NzParams cp;
        cp.ndNum = 1; cp.nValue = rows; cp.dValue = count;
        cp.srcNdMatrixStride = 0; cp.srcDValue = D;
        cp.dstNzC0Stride = Align(rows); cp.dstNzNStride = 1; cp.dstNzMatrixStride = 0;
        DataCopy(a, xGm[static_cast<uint64_t>(row) * D + k], cp);
        qA1.EnQue(a);
    }

    __aicore__ inline void CopyB(uint32_t col, uint32_t k, uint32_t cols,
                                uint32_t count, uint32_t projection) {
        auto b = qB1.AllocTensor<half>();
        PadRows(b, cols, count);
        Nd2NzParams cp;
        cp.ndNum = 1; cp.nValue = cols; cp.dValue = count;
        cp.srcNdMatrixStride = 0; cp.srcDValue = D;
        cp.dstNzC0Stride = Align(cols); cp.dstNzNStride = 1; cp.dstNzMatrixStride = 0;
        uint64_t offset = static_cast<uint64_t>(col) * D + k;
        if (projection == 0) DataCopy(b, w1Gm[offset], cp);
        else DataCopy(b, w2Gm[offset], cp);
        qB1.EnQue(b);
    }

    __aicore__ inline LocalTensor<half> LoadA(uint32_t rows, uint32_t count) {
        auto source = qA1.DeQue<half>();
        auto a = qA2.AllocTensor<half>();
        LoadData2DParams ld;
        ld.repeatTimes = Align(count) / 16; ld.srcStride = Align(rows) / 16;
        for (uint32_t m = 0; m < (rows + 15) / 16; ++m) {
            ld.startIndex = m;
            LoadData(a[m * Align(count) * 16], source, ld);
        }
        qA2.EnQue(a); qA1.FreeTensor(source);
        return qA2.DeQue<half>();
    }

    __aicore__ inline void LoadB(uint32_t cols, uint32_t count) {
        auto source = qB1.DeQue<half>();
        auto b = qB2.AllocTensor<half>();
        LoadData2DParams ld;
        ld.repeatTimes = Align(cols) * Align(count) / 256; ld.srcStride = 1;
        LoadData(b, source, ld);
        qB2.EnQue(b); qB1.FreeTensor(source);
    }

    __aicore__ inline void Store(uint32_t rows, uint32_t cols, uint32_t core,
                                uint32_t slot, uint32_t projection) {
        FixpipeParamsV220 fp;
        fp.mSize = Align(rows); fp.nSize = Align(cols);
        fp.srcStride = Align(rows); fp.dstStride = BN;
        fp.unitFlag = 3; fp.quantPre = QuantMode_t::F322F16;
        uint32_t offset = (core * 2 + slot) * BM * BN;
        auto c = cBuf.Get<float>()[(slot * 2 + projection) * BM * BN];
        if (projection == 0) Fixpipe(gateGm[offset], c, fp);
        else Fixpipe(upGm[offset], c, fp);
    }

    __aicore__ inline void Cube(uint32_t row, uint32_t col, uint32_t rows,
                               uint32_t cols, uint32_t core, uint32_t slot,
                               uint32_t iteration) {
        for (uint32_t k = 0; k < D; k += BK) {
            // 同一L1槽复用前，等待上一轮ND2NZ完成。
            PipeBarrier<PIPE_MTE2>();
            uint32_t count = Min(BK, D - k);
            bool last = k + count == D;
            CopyA(row, k, rows, count);
            CopyB(col, k, cols, count, 0);
            CopyB(col, k, cols, count, 1);
            auto a = LoadA(rows, count);
            LoadB(cols, count);
            for (uint32_t projection = 0; projection < 2; ++projection) {
                auto b = qB2.DeQue<half>();
                if (projection == 0) LoadB(cols, count);
                MmadParams mp;
                // M=1也使用普通Cube分形布局，避免自动GEMV布局切换。
                mp.m = Align(rows); mp.n = Align(cols); mp.k = Align(count);
                mp.cmatrixInitVal = k == 0;
                // 中间K块保留FP32累加，仅最后一块允许Fixpipe读取。
                mp.unitFlag = last ? 3 : 2;
                Mmad(cBuf.Get<float>()[(slot * 2 + projection) * BM * BN], a, b, mp);
                qB2.FreeTensor(b);
                if (last) {
                    if (projection == 0 && iteration >= 2) CrossCoreWaitFlag(slot + 2);
                    Store(rows, cols, core, slot, projection);
                }
            }
            qA2.FreeTensor(a);
        }
        CrossCoreSetFlag<2, PIPE_FIX>(slot);
    }

    __aicore__ inline void LoadBias(uint32_t col, uint32_t cols) {
        uint32_t colsAlign = Align(cols);
        auto input = qBias.AllocTensor<half>();
        DataCopyPad(input, b1Gm[col], {1, cols * 2, 0, 0, 0},
                    {true, 0, static_cast<uint8_t>(colsAlign - cols), half(0)});
        DataCopyPad(input[BN], b2Gm[col], {1, cols * 2, 0, 0, 0},
                    {true, 0, static_cast<uint8_t>(colsAlign - cols), half(0)});
        qBias.EnQue(input); input = qBias.DeQue<half>();
        auto bias = biasBuf.Get<half>();
        Copy(bias, input, colsAlign, VM, {1, 1, static_cast<uint16_t>(colsAlign / 16), 0});
        Copy(bias[VM * BN], input[BN], colsAlign, VM,
             {1, 1, static_cast<uint16_t>(colsAlign / 16), 0});
        qBias.FreeTensor(input);
    }

    __aicore__ inline void Vector(uint32_t row, uint32_t col, uint32_t rows,
                                 uint32_t cols, uint32_t core, uint32_t slot) {
        uint32_t first = GetSubBlockIdx() * (rows / 2);
        uint32_t count = GetSubBlockIdx() ? rows - first : rows / 2;
        if (count != 0) LoadBias(col, cols);
        CrossCoreWaitFlag(slot);
        if (count == 0) {
            CrossCoreSetFlag<2, PIPE_MTE2>(slot + 2);
            return;
        }
        uint32_t colsAlign = Align(cols), elements = count * colsAlign;
        uint32_t offset = (core * 2 + slot) * BM * BN + first * BN;
        auto gate = qGate.AllocTensor<half>();
        auto up = qUp.AllocTensor<half>();
        DataCopyPad(gate, gateGm[offset],
                    {static_cast<uint16_t>(count), colsAlign * 2, (BN - colsAlign) * 2, 0, 0}, {});
        DataCopyPad(up, upGm[offset],
                    {static_cast<uint16_t>(count), colsAlign * 2, (BN - colsAlign) * 2, 0, 0}, {});
        qGate.EnQue(gate); qUp.EnQue(up);
        // 两路结果已进入私有UB后，Cube才可复用对应GM槽。
        CrossCoreSetFlag<2, PIPE_MTE2>(slot + 2);
        gate = qGate.DeQue<half>(); up = qUp.DeQue<half>();
        auto bias = biasBuf.Get<half>();
        Add(gate, gate, bias, elements);
        auto silu = siluBuf.Get<half>();
        Silu(silu, gate, elements);
        Add(up, up, bias[VM * BN], elements);
        auto out = qOut.AllocTensor<half>();
        Mul(out, silu, up, elements);
        qGate.FreeTensor(gate); qUp.FreeTensor(up);
        qOut.EnQue(out); out = qOut.DeQue<half>();
        DataCopyPad(yGm[static_cast<uint64_t>(row + first) * D + col], out,
                    {static_cast<uint16_t>(count), cols * 2, 0, (D - cols) * 2, 0});
        qOut.FreeTensor(out);
    }

    TPipe *pipe;
    uint32_t M, D;
    TQue<TPosition::A1, 2> qA1;
    TQue<TPosition::B1, 2> qB1;
    TQue<TPosition::A2, 2> qA2;
    TQue<TPosition::B2, 2> qB2;
    TQue<TPosition::VECIN, 1> qBias;
    TQue<TPosition::VECIN, 2> qGate, qUp;
    TQue<TPosition::VECOUT, 2> qOut;
    TBuf<TPosition::CO1> cBuf;
    TBuf<TPosition::VECCALC> biasBuf, siluBuf;
    GlobalTensor<half> xGm, w1Gm, w2Gm, b1Gm, b2Gm, yGm, gateGm, upGm;
};

} // namespace swiglu