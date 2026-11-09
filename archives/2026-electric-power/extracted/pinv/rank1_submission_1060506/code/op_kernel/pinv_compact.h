#pragma once
#include "pinv_cube_residual.h"
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;
template <bool TAIL, uint32_t PAD = 0>
class Pinv {
    TBuf<TPosition::VECCALC> mem, precision, vectorPrecision, inverseLow, batchPrecision, parallelElimination;
    LocalTensor<float> batchCalc[8], batchU, batchV, batchH, batchL;
    LocalTensor<float> tileHigh, tileLow;
    LocalTensor<float> lowMatrix;
    LocalTensor<float> calc[8];
    LocalTensor<float> gh, gl, ih, il;
    LocalTensor<float> matrix, x, y, u, v, prod, tmp, red, gather, norms, output;
    GlobalTensor<float> input, result, work;
    __gm__ float *workBase;
    LinalgPinvTilingData t;
    uint32_t r8, k8, s, stride, rowLength;
    __aicore__ inline void VS() { SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0); }
    __aicore__ inline void SV() { SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0); }
    __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t count) {
        DataCopyPad(dst, src, DataCopyExtParams{1, count * 4, 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - count % 8) % 8), 0});
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
    }
    __aicore__ inline void Store(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t count) {
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopyPad(dst, src, DataCopyExtParams{1, count * 4, 0, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
    __aicore__ inline LocalTensor<float> Row(uint32_t i) { return matrix[i * stride]; }
    __aicore__ inline void ReadRow(LocalTensor<float> dst, uint32_t i) {
        if (t.local) { DataCopy(dst, Row(i), rowLength); PipeBarrier<PIPE_V>(); }
        else Load(dst, work[i * stride], rowLength);
    }
    __aicore__ inline void SaveRow(LocalTensor<float> src, uint32_t i) {
        if (t.local) { DataCopy(Row(i), src, rowLength); PipeBarrier<PIPE_V>(); }
        else Store(work[i * stride], src, rowLength);
    }
    __aicore__ inline float Dot(LocalTensor<float> a, LocalTensor<float> b) {
        Mul(prod, a, b, r8); PipeBarrier<PIPE_V>();
        if (r8 <= 64) WholeReduceSum(red, prod, r8, 1, 1, 1, 8);
        else ReduceSum(red, prod, tmp, r8);
        VS(); return red.GetValue(0);
    }
    __aicore__ inline void CacheNorms() {
        uint32_t first = t.parallel ? GetBlockIdx() : 0;
        uint32_t step = t.parallel ? PinvVectorBlockCount() : 1;
        for (uint32_t i = first; i < t.cols; i += step) {
            ReadRow(x, i); float a = Dot(x, x); x.SetValue(s, a);
            if (t.local) SV();
            SaveRow(x, i);
        }
        if (t.parallel) SyncAll();
    }
    __aicore__ inline float Root(float value) {
        red.SetValue(0, value); SV(); Sqrt(red, red, 1); VS(); return red.GetValue(0);
    }
    __aicore__ inline float AbsScalar(float value) { return value < 0 ? -value : value; }
#include "pinv_wide.h"
    __aicore__ inline void AccurateFull(uint32_t offset, float scale) {
        float power = 1.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        for (uint32_t i = 0; i < t.cols; ++i) {
            Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
            if (t.m >= t.n) {
                DataCopyPad(gather, input[offset + i], DataCopyExtParams{static_cast<uint16_t>(t.m), 4, (t.n - 1) * 4, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
                SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0); VS();
                for (uint32_t j = 0; j < t.rows; ++j) x.SetValue(j, gather.GetValue(j * 8));
                SV();
            } else Load(x, input[offset + i * t.n], t.n);
            Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); SaveRow(x, i);
        }
        for (uint32_t i = 0; i < t.cols; ++i) {
            ReadRow(x, i);
            for (uint32_t j = 0; j < t.cols; ++j) {
                ReadRow(y, j); VS(); Wide sum = {0, 0};
                for (uint32_t k = 0; k < t.rows; ++k) sum = Plus(sum, Times({x.GetValue(k), 0}, {y.GetValue(k), 0}));
                SaveEntry(gh, gl, i * k8 + j, sum);
                SaveEntry(ih, il, i * k8 + j, {i == j ? 1.0f : 0.0f, 0});
            }
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            uint32_t pivot = p; float best = AbsScalar(gh.GetValue(p * k8 + p));
            for (uint32_t i = p + 1; i < t.cols; ++i) {
                float value = AbsScalar(gh.GetValue(i * k8 + p));
                if (value > best) { best = value; pivot = i; }
            }
            if (pivot != p) for (uint32_t j = 0; j < t.cols; ++j) {
                Wide a = Entry(gh, gl, p * k8 + j), b = Entry(gh, gl, pivot * k8 + j);
                SaveEntry(gh, gl, p * k8 + j, b); SaveEntry(gh, gl, pivot * k8 + j, a);
                a = Entry(ih, il, p * k8 + j); b = Entry(ih, il, pivot * k8 + j);
                SaveEntry(ih, il, p * k8 + j, b); SaveEntry(ih, il, pivot * k8 + j, a);
            }
            Wide divisor = Entry(gh, gl, p * k8 + p);
            for (uint32_t j = 0; j < t.cols; ++j) {
                SaveEntry(gh, gl, p * k8 + j, Divide(Entry(gh, gl, p * k8 + j), divisor));
                SaveEntry(ih, il, p * k8 + j, Divide(Entry(ih, il, p * k8 + j), divisor));
            }
            for (uint32_t i = 0; i < t.cols; ++i) if (i != p) {
                Wide factor = Entry(gh, gl, i * k8 + p);
                for (uint32_t j = 0; j < t.cols; ++j) {
                    SaveEntry(gh, gl, i * k8 + j, Minus(Entry(gh, gl, i * k8 + j), Times(factor, Entry(gh, gl, p * k8 + j))));
                    SaveEntry(ih, il, i * k8 + j, Minus(Entry(ih, il, i * k8 + j), Times(factor, Entry(ih, il, p * k8 + j))));
                }
            }
        }
        for (uint32_t row = 0; row < t.n; ++row) {
            for (uint32_t col = 0; col < t.m; ++col) {
                Wide sum = {0, 0};
                for (uint32_t i = 0; i < t.cols; ++i) {
                    ReadRow(x, i); VS();
                    Wide inv = Entry(ih, il, (t.m >= t.n ? row : col) * k8 + i);
                    sum = Plus(sum, Times(inv, {x.GetValue(t.m >= t.n ? col : row), 0}));
                }
                output.SetValue(col, (sum.h + sum.l) / power);
            }
            Store(result[offset + row * t.m], output, t.m);
        }
    }


#include "pinv_vector_precision.h"

    __aicore__ inline void BatchEliminate(uint32_t count) {
        auto qh = batchCalc[0], ql = batchCalc[1], bh = batchCalc[2], bl = batchCalc[3], e = batchCalc[4], part = batchCalc[5], product = batchCalc[6], aux = batchCalc[7];
        Muls(bh, batchH, pinv::kSplitFactor, count); PipeBarrier<PIPE_V>();
        Sub(bl, bh, batchH, count); PipeBarrier<PIPE_V>(); Sub(bh, bh, bl, count); PipeBarrier<PIPE_V>();
        Sub(bl, batchH, bh, count); Muls(qh, batchU, pinv::kSplitFactor, count); PipeBarrier<PIPE_V>();
        Sub(ql, qh, batchU, count); PipeBarrier<PIPE_V>(); Sub(qh, qh, ql, count); PipeBarrier<PIPE_V>();
        Sub(ql, batchU, qh, count); Mul(product, batchU, batchH, count); PipeBarrier<PIPE_V>();
        Mul(e, qh, bh, count); PipeBarrier<PIPE_V>(); Sub(e, e, product, count); PipeBarrier<PIPE_V>();
        Mul(part, ql, bh, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, qh, bl, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, ql, bl, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, batchV, batchH, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Mul(part, batchU, batchL, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Add(qh, product, e, count); PipeBarrier<PIPE_V>(); Sub(ql, qh, product, count); PipeBarrier<PIPE_V>(); Sub(ql, e, ql, count); PipeBarrier<PIPE_V>();
        Sub(bh, x, qh, count); PipeBarrier<PIPE_V>(); Sub(bl, bh, x, count); PipeBarrier<PIPE_V>();
        Sub(e, bh, bl, count); PipeBarrier<PIPE_V>(); Sub(e, x, e, count); Muls(part, qh, -1.0f, count); PipeBarrier<PIPE_V>();
        Sub(part, part, bl, count); PipeBarrier<PIPE_V>(); Add(e, e, part, count); PipeBarrier<PIPE_V>();
        Add(e, e, y, count); PipeBarrier<PIPE_V>(); Sub(e, e, ql, count); PipeBarrier<PIPE_V>();
        Add(x, bh, e, count); PipeBarrier<PIPE_V>(); Sub(aux, x, bh, count); PipeBarrier<PIPE_V>(); Sub(y, e, aux, count); PipeBarrier<PIPE_V>();
    }
    template <uint32_t LANES>
    __aicore__ inline bool CompactInverseLegacyLanes(uint32_t offset, LocalTensor<float> bx, LocalTensor<float> by,
                                            LocalTensor<float> bu, LocalTensor<float> bv, bool tryQuick = false) {
        const uint32_t n = LANES ? LANES : t.cols;
        auto highData = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        auto lowData = reinterpret_cast<__ubuf__ float *>(lowMatrix.GetPhyAddr());
        uint32_t count = n * stride;
        float scale = 0.0f;
        Duplicate(matrix, 0.0f, count); Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(matrix, input[offset], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, 0, (stride - r8) / 8, 0}, DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(r8 - n), 0});
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        Abs(lowMatrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceMax(red, lowMatrix, tmp, count, false); VS();
        scale = red.GetValue(0);

        if (!(scale > 0.0f)) return false;
        bool quick = scale >= 0.1f;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        float total = 0.0f;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS(); total = red.GetValue(0);
        Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < n; ++i) highData[i * stride + r8 + i] = 1.0f;
        SV();

        bool vectorQuick = quick && n >= 8;
        auto indices = gather.ReinterpretCast<uint32_t>();
        if (vectorQuick) {
            for (uint32_t i = 0; i < k8; ++i) indices.SetValue(i, i < n ? i * stride * sizeof(float) : 0);
            Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>(); SV();
        }
        for (uint32_t p = 0; p < n; ++p) {
            uint32_t pivot = p; float best = 0.0f;
            if (vectorQuick) {
                Gather(norms, matrix, indices, p * sizeof(float), n); PipeBarrier<PIPE_V>();
                Abs(prod, norms, k8); PipeBarrier<PIPE_V>();
                uint64_t pivotMask[2] = {((uint64_t(1) << n) - 1) & ~((uint64_t(1) << p) - 1), 0};
                WholeReduceMax(red, prod, pivotMask, 1, 1, 1, 8); VS();
                best = red.GetValue(0); pivot = red.ReinterpretCast<uint32_t>().GetValue(1);
            } else {
                if (p == 0 || !quick) VS();
                for (uint32_t i = p; i < n; ++i) {
                    float value = AbsScalar(highData[i * stride + p]);
                    if (value > best) { best = value; pivot = i; }
                }
            }
            if (!(best > 0.0f)) return false;
            if (pivot != p) {
                DataCopy(bx, Row(p), rowLength); if (!quick) DataCopy(by, lowMatrix[p * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(p), Row(pivot), rowLength); if (!quick) DataCopy(lowMatrix[p * stride], lowMatrix[pivot * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(pivot), bx, rowLength); if (!quick) DataCopy(lowMatrix[pivot * stride], by, rowLength); PipeBarrier<PIPE_V>();
            }
            if (quick) {
                if (pivot != p) VS();
                auto pivotLocal = matrix[p * stride];
                float divisor = highData[p * stride + p];
                Muls(pivotLocal, pivotLocal, 1.0f / divisor, s); PipeBarrier<PIPE_V>();
                DataCopy(bu, pivotLocal, stride); PipeBarrier<PIPE_V>();
                if (vectorQuick) {
                    if (pivot != p) norms.SetValue(pivot, norms.GetValue(p));
                    norms.SetValue(p, 0.0f); SV(); Muls(norms, norms, -1.0f, k8); PipeBarrier<PIPE_V>();
                    if constexpr (TAIL) {
                        Brcb(lowMatrix, norms, k8 / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
                    } else {
                        Brcb(lowMatrix, norms, n / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
                    }
                    PipeBarrier<PIPE_V>();
                    MulAddDst(matrix, bu, lowMatrix, static_cast<uint64_t>(stride), static_cast<uint8_t>(n), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
                    PipeBarrier<PIPE_V>();
                    continue;
                }
                for (uint32_t first = 0; first < n; first += t.inverseTile) {
                    uint32_t rows = n - first < t.inverseTile ? n - first : t.inverseTile;
                    for (uint32_t row = 0; row < 8; ++row) {
                        uint32_t i = first + row;
                        float factor = row >= rows || i == p ? 0.0f : highData[i * stride + p];
                        red.SetValue(row, -factor);
                    }
                    SV(); Brcb(batchH, red, 1, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)}); PipeBarrier<PIPE_V>();
                    auto target = matrix[first * stride];
                    MulAddDst(target, bu, batchH, static_cast<uint64_t>(stride), static_cast<uint8_t>(rows), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
                    PipeBarrier<PIPE_V>(); VS();
                }
                continue;
            }
            x = Row(p); y = lowMatrix[p * stride]; u = bu; v = bv; VS();
            Wide divisor = {highData[p * stride + p], lowData[p * stride + p]}; SV();
            VectorNormalize(divisor); VS(); u = x; v = y;
            for (uint32_t first = 0; first < n; first += t.inverseTile) {
                uint32_t rows = n - first < t.inverseTile ? n - first : t.inverseTile;
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = first + row;
                    float fh = i == p ? 0.0f : highData[i * stride + p];
                    float fl = i == p ? 0.0f : lowData[i * stride + p];
                    Duplicate(batchH[row * stride], fh, stride);
                    DataCopy(batchU[row * stride], u, stride);
                    Duplicate(batchL[row * stride], fl, stride); DataCopy(batchV[row * stride], v, stride);
                }
                PipeBarrier<PIPE_V>();
                x = Row(first); y = lowMatrix[first * stride];
                BatchEliminate(rows * stride); VS();
            }
        }
        float inverseTotal = 0.0f;
        if (!vectorQuick) { Add(matrix, matrix, lowMatrix, count); PipeBarrier<PIPE_V>(); }
        Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix[r8], matrix[r8], static_cast<uint64_t>(r8), static_cast<uint8_t>(n), {1, 1, 1, static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8)});
        PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS(); inverseTotal = red.GetValue(0);

        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < pinv::kDirectGateSquared)) return false;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        bool packedOutput = n * n <= 8 * s;
        auto stored = matrix[r8];
        if (packedOutput) {
            auto pattern = red.ReinterpretCast<uint32_t>();
            pattern.SetValue(0, (uint32_t(1) << n) - 1);
            for (uint32_t i = 1; i < 8; ++i) pattern.SetValue(i, 0);
            SV();
            auto packed = vectorPrecision.Get<float>();
            uint64_t retained = 0;
            GatherMask(packed, matrix[r8], pattern, false, 0,
                       {1, static_cast<uint8_t>(n), static_cast<uint16_t>(stride / 8), 0}, retained);
            PipeBarrier<PIPE_V>(); stored = packed;
        }
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        if (packedOutput) DataCopyPad(result[offset], stored, DataCopyExtParams{1, n * n * 4, 0, 0, 0});
        else DataCopyPad(result[offset], matrix[r8], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, (stride - r8) / 8, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
        return true;
    }
    __aicore__ inline bool CompactInverseLegacyBody(uint32_t offset, LocalTensor<float> bx, LocalTensor<float> by,
                                            LocalTensor<float> bu, LocalTensor<float> bv, bool tryQuick = false) {
        if constexpr (PAD > 0) {
            if (t.cols % 8 == 7) return CompactInverseLegacyLanes<PAD - 1>(offset, bx, by, bu, bv, tryQuick);
        }
        return CompactInverseLegacyLanes<0>(offset, bx, by, bu, bv, tryQuick);
    }
    __aicore__ inline bool CompactInverseBody(uint32_t offset, LocalTensor<float> bx, LocalTensor<float> by,
                                            LocalTensor<float> bu, LocalTensor<float> bv, bool tryQuick = false) {
        auto highData = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        auto lowData = reinterpret_cast<__ubuf__ float *>(lowMatrix.GetPhyAddr());
        uint32_t count = t.cols * stride;
        float scale = 0.0f;
        Duplicate(matrix, 0.0f, count); Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(matrix, input[offset], DataCopyExtParams{static_cast<uint16_t>(t.cols), t.n * 4, 0, (stride - r8) / 8, 0}, DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(r8 - t.n), 0});
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        Abs(lowMatrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceMax(red, lowMatrix, tmp, count, false); VS();
        scale = red.GetValue(0);

        if (!(scale > 0.0f)) return false;
        bool quick = scale >= 0.1f;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        float total = 0.0f;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix, matrix, count); PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS(); total = red.GetValue(0);
        Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < t.cols; ++i) highData[i * stride + r8 + i] = 1.0f;
        SV();

        bool vectorQuick = quick && t.cols >= 8;
        bool logicalPivot = vectorQuick && !TAIL;
        auto indices = gather.ReinterpretCast<uint32_t>();
        auto pivotOrder = output.ReinterpretCast<uint32_t>();
        uint64_t available = (uint64_t(1) << t.cols) - 1;
        if (vectorQuick) {
            for (uint32_t i = 0; i < k8; ++i) indices.SetValue(i, i < t.cols ? i * stride * sizeof(float) : 0);
            Duplicate(norms, 0.0f, k8); PipeBarrier<PIPE_V>(); SV();
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            uint32_t pivot = p; float best = 0.0f;
            if (vectorQuick) {
                Gather(norms, matrix, indices, p * sizeof(float), t.cols); PipeBarrier<PIPE_V>();
                Abs(prod, norms, k8); PipeBarrier<PIPE_V>();
                uint64_t pivotMask[2] = {logicalPivot ? available : (((uint64_t(1) << t.cols) - 1) & ~((uint64_t(1) << p) - 1)), 0};
                WholeReduceMax(red, prod, pivotMask, 1, 1, 1, 8); VS();
                best = red.GetValue(0); pivot = red.ReinterpretCast<uint32_t>().GetValue(1);
            } else {
                if (p == 0 || !quick) VS();
                for (uint32_t i = p; i < t.cols; ++i) {
                    float value = AbsScalar(highData[i * stride + p]);
                    if (value > best) { best = value; pivot = i; }
                }
            }
            if (!(best > 0.0f)) return false;
            if (!logicalPivot && pivot != p) {
                DataCopy(bx, Row(p), rowLength); if (!quick) DataCopy(by, lowMatrix[p * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(p), Row(pivot), rowLength); if (!quick) DataCopy(lowMatrix[p * stride], lowMatrix[pivot * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(pivot), bx, rowLength); if (!quick) DataCopy(lowMatrix[pivot * stride], by, rowLength); PipeBarrier<PIPE_V>();
            }
            if (quick) {
                if (!logicalPivot && pivot != p) VS();
                uint32_t physicalPivot = logicalPivot ? pivot : p;
                auto pivotLocal = matrix[physicalPivot * stride];
                float divisor = highData[physicalPivot * stride + p];
                Muls(pivotLocal, pivotLocal, 1.0f / divisor, s); PipeBarrier<PIPE_V>();
                DataCopy(bu, pivotLocal, stride); PipeBarrier<PIPE_V>();
                if (vectorQuick) {
                    if (logicalPivot) {
                        available &= ~(uint64_t(1) << pivot);
                        pivotOrder.SetValue(pivot, p);
                        norms.SetValue(pivot, 0.0f);
                    } else {
                        if (pivot != p) norms.SetValue(pivot, norms.GetValue(p));
                        norms.SetValue(p, 0.0f);
                    }
                    SV(); Muls(norms, norms, -1.0f, k8); PipeBarrier<PIPE_V>();
                    if constexpr (TAIL) {
                        Brcb(lowMatrix, norms, k8 / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
                    } else {
                        Brcb(lowMatrix, norms, t.cols / 8, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)});
                    }
                    PipeBarrier<PIPE_V>();
                    MulAddDst(matrix, bu, lowMatrix, static_cast<uint64_t>(stride), static_cast<uint8_t>(t.cols), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
                    PipeBarrier<PIPE_V>();
                    continue;
                }
                for (uint32_t first = 0; first < t.cols; first += t.inverseTile) {
                    uint32_t rows = t.cols - first < t.inverseTile ? t.cols - first : t.inverseTile;
                    for (uint32_t row = 0; row < 8; ++row) {
                        uint32_t i = first + row;
                        float factor = row >= rows || i == p ? 0.0f : highData[i * stride + p];
                        red.SetValue(row, -factor);
                    }
                    SV(); Brcb(batchH, red, 1, {static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride)}); PipeBarrier<PIPE_V>();
                    auto target = matrix[first * stride];
                    MulAddDst(target, bu, batchH, static_cast<uint64_t>(stride), static_cast<uint8_t>(rows), {1, 1, 0, static_cast<uint8_t>(stride / 8), 0, static_cast<uint8_t>(stride / 8)});
                    PipeBarrier<PIPE_V>(); VS();
                }
                continue;
            }
            x = Row(p); y = lowMatrix[p * stride]; u = bu; v = bv; VS();
            Wide divisor = {highData[p * stride + p], lowData[p * stride + p]}; SV();
            VectorNormalize(divisor); VS(); u = x; v = y;
            for (uint32_t first = 0; first < t.cols; first += t.inverseTile) {
                uint32_t rows = t.cols - first < t.inverseTile ? t.cols - first : t.inverseTile;
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = first + row;
                    float fh = i == p ? 0.0f : highData[i * stride + p];
                    float fl = i == p ? 0.0f : lowData[i * stride + p];
                    Duplicate(batchH[row * stride], fh, stride);
                    DataCopy(batchU[row * stride], u, stride);
                    Duplicate(batchL[row * stride], fl, stride); DataCopy(batchV[row * stride], v, stride);
                }
                PipeBarrier<PIPE_V>();
                x = Row(first); y = lowMatrix[first * stride];
                BatchEliminate(rows * stride); VS();
            }
        }
        float inverseTotal = 0.0f;
        if (!vectorQuick) { Add(matrix, matrix, lowMatrix, count); PipeBarrier<PIPE_V>(); }
        Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix[r8], matrix[r8], static_cast<uint64_t>(r8), static_cast<uint8_t>(t.cols), {1, 1, 1, static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8), static_cast<uint8_t>(stride / 8)});
        PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS(); inverseTotal = red.GetValue(0);

        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < pinv::kDirectGateSquared)) return false;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        auto ordered = matrix;
        if (logicalPivot) {
            for (uint32_t row = 0; row < t.cols; ++row) {
                uint32_t target = pivotOrder.GetValue(row);
                DataCopy(lowMatrix[target * stride], matrix[row * stride], stride);
            }
            PipeBarrier<PIPE_V>(); ordered = lowMatrix;
        }
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopyPad(result[offset], ordered[r8], DataCopyExtParams{static_cast<uint16_t>(t.n), t.m * 4, (stride - r8) / 8, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
        return true;
    }
    __aicore__ inline bool AccurateLocalInverseBody(uint32_t offset, LocalTensor<float> bx, LocalTensor<float> by,
                                            LocalTensor<float> bu, LocalTensor<float> bv) {
        float scale = 0.0f;
        for (uint32_t i = 0; i < t.cols; ++i) {
            x = Row(i); y = lowMatrix[i * stride];
            Duplicate(x, 0.0f, rowLength); Duplicate(y, 0.0f, rowLength); PipeBarrier<PIPE_V>();
            Load(x, input[offset + i * t.n], t.n);
            Abs(prod, x, r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
            float mx = red.GetValue(0); scale = mx > scale ? mx : scale;
            x.SetValue(r8 + i, 1.0f); SV();
        }
        if (!(scale > 0.0f)) return false;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        float total = 0.0f;
        for (uint32_t i = 0; i < t.cols; ++i) {
            x = Row(i); Muls(x, x, 1.0f / power, r8); PipeBarrier<PIPE_V>(); total += Dot(x, x);
        }
        for (uint32_t p = 0; p < t.cols; ++p) {
            VS(); uint32_t pivot = p; float best = 0.0f;
            for (uint32_t i = p; i < t.cols; ++i) {
                float value = AbsScalar(Row(i).GetValue(p));
                if (value > best) { best = value; pivot = i; }
            }
            if (!(best > 0.0f)) return false;
            if (pivot != p) {
                DataCopy(bx, Row(p), rowLength); DataCopy(by, lowMatrix[p * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(p), Row(pivot), rowLength); DataCopy(lowMatrix[p * stride], lowMatrix[pivot * stride], rowLength); PipeBarrier<PIPE_V>();
                DataCopy(Row(pivot), bx, rowLength); DataCopy(lowMatrix[pivot * stride], by, rowLength); PipeBarrier<PIPE_V>();
            }
            x = Row(p); y = lowMatrix[p * stride]; u = bu; v = bv; VS();
            Wide divisor = {x.GetValue(p), y.GetValue(p)}; SV(); VectorNormalize(divisor);
            VS(); u = x; v = y;
            for (uint32_t first = 0; first < t.cols; first += t.inverseTile) {
                uint32_t rows = t.cols - first < t.inverseTile ? t.cols - first : t.inverseTile;
                for (uint32_t row = 0; row < rows; ++row) {
                    uint32_t i = first + row;
                    float fh = i == p ? 0.0f : Row(i).GetValue(p);
                    float fl = i == p ? 0.0f : lowMatrix[i * stride].GetValue(p);
                    Duplicate(batchH[row * stride], fh, stride); Duplicate(batchL[row * stride], fl, stride);
                    DataCopy(batchU[row * stride], u, stride); DataCopy(batchV[row * stride], v, stride);
                }
                PipeBarrier<PIPE_V>();
                x = Row(first); y = lowMatrix[first * stride];
                BatchEliminate(rows * stride); VS();
            }
        }
        float inverseTotal = 0.0f;
        for (uint32_t i = 0; i < t.cols; ++i) { x = Row(i); inverseTotal += Dot(x[r8], x[r8]); }
        float cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < pinv::kDirectGateSquared)) return false;
        for (uint32_t i = 0; i < t.n; ++i) {
            x = Row(i); y = lowMatrix[i * stride];
            Add(output, x[r8], y[r8], r8); PipeBarrier<PIPE_V>(); Muls(output, output, 1.0f / power, r8); PipeBarrier<PIPE_V>();
            Store(result[offset + i * t.m], output, t.m);
        }
        return true;
    }
    template <uint32_t LANES>
    __aicore__ inline bool CompactQuickLogical(uint32_t offset) {
        constexpr uint32_t n = LANES, width = 2 * PAD, pitch = 2 * PAD + 8;
        constexpr uint32_t count = n * pitch;
        Duplicate(matrix, 0.0f, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(matrix, input[offset], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, 0, (pitch - PAD) / 8, 0}, DataCopyPadExtParams<float>{TAIL, 0, static_cast<uint8_t>(PAD - n), 0});
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        Abs(lowMatrix, matrix, count); PipeBarrier<PIPE_V>();
        ReduceMax(red, lowMatrix, tmp, count, false); VS();
        float scale = red.GetValue(0);
        if (!(scale >= 0.1f && scale < pinv::kScaleLimit)) return false;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix, matrix, count); PipeBarrier<PIPE_V>();
        ReduceSum(red, lowMatrix, tmp, count); VS();
        float total = red.GetValue(0);
        auto values = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        for (uint32_t i = 0; i < n; ++i) values[i * pitch + PAD + i] = 1.0f;
        auto indices = gather.ReinterpretCast<uint32_t>();
        uint32_t order[PAD];
        for (uint32_t i = 0; i < PAD; ++i) indices.SetValue(i, i < n ? i * pitch * sizeof(float) : 0);
        if constexpr (TAIL) { Duplicate(norms, 0.0f, PAD); PipeBarrier<PIPE_V>(); }
        uint64_t available = (uint64_t(1) << n) - 1;
        SV();
        for (uint32_t column = 0; column < n; ++column) {
            Gather(norms, matrix, indices, column * sizeof(float), n); PipeBarrier<PIPE_V>();
            Abs(prod, norms, PAD); PipeBarrier<PIPE_V>();
            uint64_t mask[2] = {available, 0};
            WholeReduceMax(red, prod, mask, 1, 1, 1, 8); VS();
            float best;uint32_t pivot;
            if constexpr (TAIL) {
                float pivotBits;GetReduceRepeatMaxMinSpr<float>(best,pivotBits);
                pivot=*reinterpret_cast<uint32_t*>(&pivotBits);
            } else {
                best=red.GetValue(0);pivot=red.ReinterpretCast<uint32_t>().GetValue(1);
            }
            if (!(best > 0.0f)) return false;
            available &= ~(uint64_t(1) << pivot);
            order[pivot] = column;
            Muls(matrix[pivot * pitch], matrix[pivot * pitch], -1.0f / values[pivot * pitch + column], width); PipeBarrier<PIPE_V>();
            if constexpr (TAIL) { DataCopy(u, matrix[pivot * pitch], pitch); PipeBarrier<PIPE_V>(); }
            norms.SetValue(pivot, 0.0f); SV();
            Brcb(lowMatrix, norms, PAD / 8, {static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch)}); PipeBarrier<PIPE_V>();
            MulAddDst(matrix, TAIL ? u : matrix[pivot * pitch], lowMatrix, static_cast<uint64_t>(width), static_cast<uint8_t>(n), {1, 1, 0, static_cast<uint8_t>(pitch / 8), 0, static_cast<uint8_t>(pitch / 8)});
            PipeBarrier<PIPE_V>();
        }
        Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix[PAD], matrix[PAD], static_cast<uint64_t>(PAD), static_cast<uint8_t>(n), {1, 1, 1, static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch / 8)});
        PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS();
        float inverseTotal = red.GetValue(0), cutoff = Root(total) * t.rtol;
        float absolute = t.atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < pinv::kDirectGateSquared)) return false;
        Muls(matrix, matrix, -1.0f / power, count); PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < n; ++row) {
            uint32_t target = order[row];
            DataCopy(lowMatrix[target * pitch], matrix[row * pitch], pitch);
        }
        PipeBarrier<PIPE_V>();
        if constexpr (TAIL) {
        uint64_t kept = 0;
        GatherMask(matrix, lowMatrix[PAD], uint8_t(7), true, n, {1, static_cast<uint8_t>(n), static_cast<uint16_t>(pitch / 8), 0}, kept);
        PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        if constexpr (TAIL) DataCopyPad(result[offset], matrix, DataCopyExtParams{1, n * n * 4, 0, 0, 0});
        else DataCopyPad(result[offset],lowMatrix[PAD],DataCopyExtParams{static_cast<uint16_t>(n),n*4,(pitch-n)/8,0,0});
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
        return true;
    }
    __aicore__ inline bool LocalInverse(uint32_t offset) {
        if constexpr (PAD > 0) {
            if constexpr (TAIL) { if (t.cols % 8 == 7 && CompactQuickLogical<PAD - 1>(offset)) return true; }
            else { if (CompactQuickLogical<PAD>(offset)) return true; }
        }
        auto bx = x, by = y, bu = u, bv = v;
        bool accepted;
        if constexpr (TAIL) accepted = stride <= 64 ? CompactInverseLegacyBody(offset, bx, by, bu, bv) : AccurateLocalInverseBody(offset, bx, by, bu, bv);
        else accepted = stride <= 64 ? CompactInverseBody(offset, bx, by, bu, bv) : AccurateLocalInverseBody(offset, bx, by, bu, bv);
        x = bx; y = by; u = bu; v = bv;
        return accepted;
    }
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR out, GM_ADDR ws, const LinalgPinvTilingData &data, TPipe *pipe) {
        t = data; r8 = (t.rows + 7) / 8 * 8; k8 = (t.cols + 7) / 8 * 8; s = r8 + k8; stride = t.stride; rowLength = s + 8;
        if constexpr (PAD > 0) {
            r8 = PAD; k8 = PAD; s = 2 * PAD; stride = 2 * PAD + 8; rowLength = stride;
            if constexpr (!TAIL) { t.m = PAD; t.n = PAD; t.rows = PAD; t.cols = PAD; }
        }
        uint32_t bufferStride = t.parallel ? (rowLength + 63) / 64 * 64 : stride;
        uint32_t size = (t.local ? t.cols * stride : 0) + 4 * bufferStride + 4 * r8 + 8 * t.rows + k8 + 32;
        pipe->InitBuffer(mem, size * 4);
        auto all = mem.Get<float>(); uint32_t off = 0;
        matrix = all; off += t.local ? t.cols * stride : 0;
        x = all[off]; off += bufferStride; y = all[off]; off += bufferStride; u = all[off]; off += bufferStride; v = all[off]; off += bufferStride;
        prod = all[off]; off += r8; tmp = all[off]; off += r8; output = all[off]; off += r8;
        red = all[off]; off += 32; gather = all[off]; off += 8 * t.rows;
        norms = all[off];
        if (t.parallel || t.localInverse) {
            pipe->InitBuffer(vectorPrecision, 8 * s * sizeof(float));
            auto area = vectorPrecision.Get<float>();
            for (uint32_t i = 0; i < 8; ++i) calc[i] = area[i * s];
        }
        if (t.localInverse) {
            pipe->InitBuffer(inverseLow, (TAIL ? k8 : t.cols) * stride * sizeof(float));
            lowMatrix = inverseLow.Get<float>();
        }
        if (t.localInverse || (t.parallel && t.m == t.n)) {
            uint32_t areaSize = t.inverseTile * stride;
            pipe->InitBuffer(batchPrecision, 12 * areaSize * sizeof(float));
            auto area = batchPrecision.Get<float>();
            for (uint32_t i = 0; i < 8; ++i) batchCalc[i] = area[i * areaSize];
            batchU = area[8 * areaSize]; batchV = area[9 * areaSize]; batchH = area[10 * areaSize]; batchL = area[11 * areaSize];
            if (t.parallel) {
                pipe->InitBuffer(parallelElimination, 2 * areaSize * sizeof(float));
                tileHigh = parallelElimination.Get<float>(); tileLow = tileHigh[areaSize];
            }
        }
        if (t.rows <= 32) {
            pipe->InitBuffer(precision, 4 * k8 * k8 * sizeof(float));
            gh = precision.Get<float>(); gl = gh[k8 * k8]; ih = gl[k8 * k8]; il = ih[k8 * k8];
        }
        input.SetGlobalBuffer((__gm__ float *)a); result.SetGlobalBuffer((__gm__ float *)out);
        workBase = (__gm__ float *)ws; work.SetGlobalBuffer(workBase);
    }
    __aicore__ inline void Process() {
        for (uint32_t b = GetBlockIdx(); b < t.batch; b += PinvVectorBlockCount()) {
            uint32_t offset = b * t.m * t.n;
            if (t.localInverse && LocalInverse(offset)) continue;
            if (!t.local) work.SetGlobalBuffer(workBase + uint64_t(b) * t.cols * stride);
            float scale = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                Duplicate(x, 0.0f, rowLength); PipeBarrier<PIPE_V>();
                if (t.m >= t.n) {
                    DataCopyPad(gather, input[offset + i], DataCopyExtParams{static_cast<uint16_t>(t.m), 4, (t.n - 1) * 4, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
                    SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0); VS();
                    for (uint32_t j = 0; j < t.rows; ++j) x.SetValue(j, gather.GetValue(j * 8));
                } else Load(x, input[offset + i * t.n], t.n);
                VS(); x.SetValue(r8 + i, 1.0f); SV();
                Abs(prod, x, r8); PipeBarrier<PIPE_V>(); ReduceMax(red, prod, tmp, r8, false); VS();
                float mx = red.GetValue(0); scale = mx > scale ? mx : scale;
                SaveRow(x, i);
            }
            if (scale == 0) {
                Duplicate(output, 0.0f, r8); PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < t.n; ++i) Store(result[offset + i * t.m], output, t.m);
                continue;
            }
            float total = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                ReadRow(x, i); Muls(x, x, 1.0f / scale, r8); PipeBarrier<PIPE_V>();
                total += Dot(x, x); SaveRow(x, i);
            }
            float absTol = t.atol / scale;
            float discard = absTol * absTol;
            float rel = t.rtol * t.rtol * total;
            discard = (discard > rel ? discard : rel) * pinv::kJacobiDiscardMargin;
            float roundoffFloor = total * pinv::kJacobiRoundoffFloor;
            discard = discard < roundoffFloor ? discard : roundoffFloor;
            bool cached = false;
            for (uint32_t sweep = 0; sweep < pinv::kJacobiSweeps; ++sweep) {
                uint32_t changed = 0;
                for (uint32_t p = 0; p < t.cols; ++p) {
                    ReadRow(x, p);
                    for (uint32_t q = p + 1; q < t.cols; ++q) {
                        ReadRow(y, q);
                        float a, d, h;
                        if (cached) { h = Dot(x, y); a = x.GetValue(s); d = y.GetValue(s); }
                        else { a = Dot(x, x); d = Dot(y, y); h = Dot(x, y); }
                        if (a <= discard || d <= discard || h == 0 || h * h <= pinv::kFp32OrthogonalitySquared * (a * d)) continue;
                        float tau = (d - a) / (2 * h);
                        float at = AbsScalar(tau);
                        float z = (tau < 0 ? -1.0f : 1.0f) / (at + Root(1 + at * at));
                        float c = 1.0f / Root(1 + z * z), sn = c * z;
                        Muls(u, x, c, s); Muls(v, y, -sn, s); PipeBarrier<PIPE_V>(); Add(u, u, v, s); PipeBarrier<PIPE_V>();
                        Muls(v, x, sn, s); Muls(y, y, c, s); PipeBarrier<PIPE_V>(); Add(y, y, v, s); PipeBarrier<PIPE_V>();
                        if (cached) {
                            float na = Dot(u, u), nd = Dot(y, y);
                            u.SetValue(s, na); y.SetValue(s, nd); SV();
                        }
                        DataCopy(x, u, stride); PipeBarrier<PIPE_V>(); SaveRow(y, q); ++changed;
                    }
                    SaveRow(x, p);
                }
                if (changed == 0) break;
                if (!cached && changed < pinv::kJacobiNormCacheRatio * t.cols * (t.cols - 1)) { CacheNorms(); cached = true; }
            }
            float maxNorm = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                ReadRow(x, i); float a = Dot(x, x); norms.SetValue(i, a); maxNorm = a > maxNorm ? a : maxNorm;
            }
            float threshold = absTol * absTol;
            float relative = maxNorm * t.rtol * t.rtol;
            threshold = threshold > relative ? threshold : relative;
            uint32_t kept = 0;
            for (uint32_t i = 0; i < t.cols; ++i) {
                float a = norms.GetValue(i); if (a > threshold) ++kept; norms.SetValue(i, a > threshold ? 1.0f / (a * scale) : 0.0f);
            }
            if (t.rows <= 32 && kept == t.cols && scale < 0.001f) { AccurateFull(offset, scale); continue; }
            for (uint32_t row = 0; row < t.n; ++row) {
                Duplicate(output, 0.0f, r8); PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < t.cols; ++i) {
                    float inv = norms.GetValue(i); if (inv == 0) continue;
                    ReadRow(x, i); VS();
                    float factor = x.GetValue(t.m >= t.n ? r8 + row : row) * inv;
                    Muls(prod, t.m >= t.n ? x : x[r8], factor, t.m >= t.n ? r8 : k8); PipeBarrier<PIPE_V>();
                    Add(output, output, prod, t.m >= t.n ? r8 : k8); PipeBarrier<PIPE_V>();
                }
                Store(result[offset + row * t.m], output, t.m);
            }
        }

    }
};
