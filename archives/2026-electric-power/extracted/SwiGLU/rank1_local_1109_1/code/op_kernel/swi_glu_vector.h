#pragma once
#include "kernel_operator.h"

template <typename T>
__aicore__ inline void CopyTensorPair(AscendC::LocalTensor<T> dst, uint32_t dstStep,
        AscendC::GlobalTensor<T> first, AscendC::GlobalTensor<T> second,
        uint32_t elements, AscendC::DataCopyPadExtParams<T> pad) {
    using namespace AscendC;
    const uint64_t firstAddress = reinterpret_cast<uint64_t>(first.GetPhyAddr());
    const uint64_t secondAddress = reinterpret_cast<uint64_t>(second.GetPhyAddr());
    const uint32_t bytes = elements * sizeof(T);
    const uint32_t rounded = (bytes + 31) / 32 * 32;
    const uint32_t dstBytes = dstStep * sizeof(T);
    const uint64_t distance = secondAddress - firstAddress;
    // A two-block DMA is valid only when both source and destination strides fit.
    if (secondAddress >= firstAddress && distance >= bytes &&
            distance - bytes <= 0xffffffffULL && dstBytes >= rounded &&
            (dstBytes - rounded) % 32 == 0) {
        DataCopyExtParams copy{2, bytes, static_cast<uint32_t>(distance - bytes),
            (dstBytes - rounded) / 32, 0};
        DataCopyPad(dst, first, copy, pad);
    } else {
        DataCopyExtParams copy{1, bytes, 0, 0, 0};
        DataCopyPad(dst, first, copy, pad);
        DataCopyPad(dst[dstStep], second, copy, pad);
    }
}

// Read in increasing GM address order and preserve the logical projection order.
template <typename T>
__aicore__ inline bool CopyOrderedTensorPair(AscendC::LocalTensor<T> dst, uint32_t dstStep,
        AscendC::GlobalTensor<T> first, AscendC::GlobalTensor<T> second,
        uint32_t elements, AscendC::DataCopyPadExtParams<T> pad) {
    using namespace AscendC;
    const uint64_t firstAddress = reinterpret_cast<uint64_t>(first.GetPhyAddr());
    const uint64_t secondAddress = reinterpret_cast<uint64_t>(second.GetPhyAddr());
    const bool reverse = secondAddress < firstAddress;
    const uint64_t distance = reverse ? firstAddress - secondAddress : secondAddress - firstAddress;
    const uint32_t bytes = elements * sizeof(T);
    const uint32_t rounded = (bytes + 31) / 32 * 32;
    const uint32_t dstBytes = dstStep * sizeof(T);
    if (distance >= bytes && distance - bytes <= 0xffffffffULL &&
            dstBytes >= rounded && (dstBytes - rounded) % 32 == 0) {
        DataCopyExtParams copy{2, bytes, static_cast<uint32_t>(distance - bytes),
            (dstBytes - rounded) / 32, 0};
        if (reverse) DataCopyPad(dst, second, copy, pad);
        else DataCopyPad(dst, first, copy, pad);
        return reverse;
    }
    DataCopyExtParams copy{1, bytes, 0, 0, 0};
    DataCopyPad(dst, first, copy, pad);
    DataCopyPad(dst[dstStep], second, copy, pad);
    return false;
}

__aicore__ inline void MakeGatherOffsets(AscendC::LocalTensor<uint32_t> offsets,
        uint32_t dim, uint32_t pitch, uint32_t rows) {
    using namespace AscendC;
    auto indices = offsets.ReinterpretCast<int32_t>();
    for (uint32_t i = 0; i < 8 && i < pitch; ++i) indices.SetValue(i, 2 * i);
    SetFlag<HardEvent::S_V>(5);
    WaitFlag<HardEvent::S_V>(5);
    for (uint32_t n = 8; n < pitch; n *= 2) {
        Adds(indices[n], indices, static_cast<int32_t>(n * 2),
            pitch - n < n ? pitch - n : n);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t n = 1; n < rows; n *= 2) {
        Adds(indices[n * pitch], indices, static_cast<int32_t>(n * dim * 2),
            (rows - n < n ? rows - n : n) * pitch);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void HalfGateSwiGlu(AscendC::LocalTensor<T> out,
        AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b, uint32_t count) {
    if (count <= 128) {
        AscendC::SetVectorMask<T>(count);
    } else {
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(count);
    }
    AscendC::UnaryRepeatParams unary{1, 1, 8, 8};
    AscendC::BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    AscendC::Muls<T, false>(out, a, static_cast<T>(-1.0f), AscendC::MASK_PLACEHOLDER, 1, unary);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<T, false>(out, out, AscendC::MASK_PLACEHOLDER, 1, unary);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<T, false>(out, out, static_cast<T>(1.0f), AscendC::MASK_PLACEHOLDER, 1, unary);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<T, false>(out, a, out, AscendC::MASK_PLACEHOLDER, 1, binary);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<T, false>(out, out, b, AscendC::MASK_PLACEHOLDER, 1, binary);
    if (count > 128) AscendC::SetMaskNorm();
}

// Every row contains two consecutive 128-element projections.
template <typename T>
__aicore__ inline void HalfGateInterleaved128(AscendC::LocalTensor<T> out,
        AscendC::LocalTensor<T> pair, uint8_t rows) {
    using namespace AscendC;
    SetVectorMask<T>(128);
    UnaryRepeatParams fromPair{1, 1, 8, 16};
    UnaryRepeatParams self{1, 1, 8, 8};
    BinaryRepeatParams divide{1, 1, 1, 8, 16, 8};
    BinaryRepeatParams multiply{1, 1, 1, 8, 8, 16};
    Muls<T, false>(out, pair, static_cast<T>(-1.0f), MASK_PLACEHOLDER, rows, fromPair);
    PipeBarrier<PIPE_V>();
    Exp<T, false>(out, out, MASK_PLACEHOLDER, rows, self);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(out, out, static_cast<T>(1.0f), MASK_PLACEHOLDER, rows, self);
    PipeBarrier<PIPE_V>();
    Div<T, false>(out, pair, out, MASK_PLACEHOLDER, rows, divide);
    PipeBarrier<PIPE_V>();
    Mul<T, false>(out, out, pair[128], MASK_PLACEHOLDER, rows, multiply);
}

template <typename T, uint32_t DIM>
class TinyVectorSwiGlu {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
            GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output, GM_ADDR tiling) {
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight2));
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    }

    __aicore__ inline void Process() {
        using namespace AscendC;
        static_assert(DIM == 4);
        const uint32_t row = GetBlockIdx();
        LocalTensor<T> packed(TPosition::VECCALC, 0, 48);
        LocalTensor<T> bias(TPosition::VECCALC, 192, 32);
        LocalTensor<T> pair(TPosition::VECCALC, 288, 32);
        LocalTensor<T> out(TPosition::VECCALC, 384, 16);
        LocalTensor<float> weights(TPosition::VECCALC, 512, 48);
        auto input = weights[32];
        LocalTensor<float> repeated(TPosition::VECCALC, 1344, 32);
        LocalTensor<float> products(TPosition::VECCALC, 1632, 32);
        LocalTensor<float> partial(TPosition::VECCALC, 1824, 32);
        LocalTensor<uint32_t> offsets(TPosition::VECCALC, 2048, 32);
        LocalTensor<float> sums(TPosition::VECCALC, 2304, 64);
        DataCopyPadExtParams<T> noPad{false, 0, 0, 0};
        const bool reverseWeights = CopyOrderedTensorPair(packed, 16, w1, w2, 16, noPad);
        const bool reverseBias = reverseWeights;
        DataCopyExtParams vectorCopy{1, DIM * sizeof(T), 0, 0, 0};
        DataCopyPad(bias[reverseWeights ? 16 : 0], b1, vectorCopy, noPad);
        DataCopyPad(bias[reverseWeights ? 0 : 16], b2, vectorCopy, noPad);
        DataCopyPadExtParams<T> inputPad{true, 0, 12, 0};
        DataCopyPad(packed[32], x[row * DIM], vectorCopy, inputPad);

        auto pattern = offsets.ReinterpretCast<int32_t>();
        Duplicate(pattern, static_cast<int32_t>(0x0f0f0f0f), 2);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        UnaryRepeatParams widen{1, 1, 8, 4};
        Cast(weights, packed, RoundMode::CAST_NONE, 48, 1, widen);
        PipeBarrier<PIPE_V>();
        uint64_t gathered = 0;
        GatherMask(repeated, input, offsets, false, 0,
            GatherMaskParams{0, 1, 0, 0}, gathered);
        PipeBarrier<PIPE_V>();
        Mul(products, weights, repeated, 32, 1, BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        PairReduceSum(partial, products, 1, 32, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        PairReduceSum(sums, partial, 2, 8, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Cast(pair, sums, RoundMode::CAST_RINT, DIM, 2, UnaryRepeatParams{1, 1, 1, 4});
        PipeBarrier<PIPE_V>();
        if (reverseWeights == reverseBias) {
            Add(pair, pair, bias, DIM, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});
        } else {
            Add(pair, pair, bias[16], DIM);
            Add(pair[16], pair[16], bias, DIM);
        }
        PipeBarrier<PIPE_V>();
        HalfGateSwiGlu(out, pair[reverseWeights ? 16 : 0], pair[reverseWeights ? 0 : 16], DIM);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y[row * DIM], out, vectorCopy);
    }

private:
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};

// One scalar output needs no row scheduling or large vector tile buffers.
template <typename T>
class ScalarVectorSwiGlu {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
            GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output) {
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight2));
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    }
    __aicore__ inline void Process() {
        using namespace AscendC;
        LocalTensor<T> coeff(TPosition::VECCALC, 0, 80);
        LocalTensor<T> input(TPosition::VECCALC, 192, 16);
        LocalTensor<T> pair(TPosition::VECCALC, 256, 32);
        LocalTensor<T> out(TPosition::VECCALC, 352, 16);
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyExtParams one{1, sizeof(T), 0, 0, 0};
        const uint64_t xa = reinterpret_cast<uint64_t>(x.GetPhyAddr());
        const uint64_t wa = reinterpret_cast<uint64_t>(w1.GetPhyAddr());
        const uint64_t ba = reinterpret_cast<uint64_t>(b1.GetPhyAddr());
        const uint64_t wb = reinterpret_cast<uint64_t>(w2.GetPhyAddr());
        const uint64_t bb = reinterpret_cast<uint64_t>(b2.GetPhyAddr());
        const uint64_t delta = wa - xa;
        const uint64_t mismatch = (ba - wa - delta) | (wb - ba - delta) | (bb - wb - delta);
        const bool forward = wa >= xa;
        const uint64_t step = forward ? delta : xa - wa;
        auto first = pair;
        auto second = pair[16];
        SetMaskNorm();
        SetVectorMask<T>(1);
        if (mismatch == 0 && step >= sizeof(T) && step - sizeof(T) <= 0xffffffffULL) {
            DataCopyExtParams all{5, sizeof(T), static_cast<uint32_t>(step - sizeof(T)), 0, 0};
            if (forward) DataCopyPad(coeff, x, all, pad);
            else DataCopyPad(coeff, b2, all, pad);
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
            const T scalarX = coeff.GetValue(forward ? 0 : 64);
            Axpy<T, T, false>(coeff[forward ? 32 : 0], coeff[16], scalarX, MASK_PLACEHOLDER, 2,
                UnaryRepeatParams{1, 1, 2, 2});
            PipeBarrier<PIPE_V>();
            first = coeff[32];
            second = coeff[forward ? 64 : 0];
        } else {
            DataCopyPad(input, x, one, pad);
            DataCopyPad(coeff, w1, one, pad);
            DataCopyPad(coeff[16], w2, one, pad);
            DataCopyPad(pair, b1, one, pad);
            DataCopyPad(pair[16], b2, one, pad);
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
            const T scalarX = input.GetValue(0);
            Axpy<T, T, false>(pair, coeff, scalarX, MASK_PLACEHOLDER, 2,
                UnaryRepeatParams{1, 1, 1, 1});
            PipeBarrier<PIPE_V>();
        }
        UnaryRepeatParams unary{1, 1, 8, 8};
        BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        Muls<T, false>(out, first, static_cast<T>(-1.0f), MASK_PLACEHOLDER, 1, unary);
        PipeBarrier<PIPE_V>();
        Exp<T, false>(out, out, MASK_PLACEHOLDER, 1, unary);
        PipeBarrier<PIPE_V>();
        Adds<T, false>(out, out, static_cast<T>(1.0f), MASK_PLACEHOLDER, 1, unary);
        PipeBarrier<PIPE_V>();
        Div<T, false>(out, first, out, MASK_PLACEHOLDER, 1, binary);
        PipeBarrier<PIPE_V>();
        Mul<T, false>(out, out, second, MASK_PLACEHOLDER, 1, binary);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y, out, one);
    }
private:
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};

template <typename T, uint32_t MODE, bool ONE_TILE = false, uint32_t STATIC_DIM = 0>
class VectorSwiGlu {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
            GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output,
            const SwiGluVectorTilingData &data, GM_ADDR tiling) {
        t = data;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight2));
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    }
    __aicore__ inline void Process() {
        using namespace AscendC;
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if constexpr (MODE == 1) {
            LocalTensor<T> input(TPosition::VECCALC, 0, 16384);
            LocalTensor<T> a(TPosition::VECCALC, 32800, 16384);
            LocalTensor<T> b(TPosition::VECCALC, 65600, 16384);
            LocalTensor<T> out(TPosition::VECCALC, 98400, 16384);
            LocalTensor<T> coeff(TPosition::VECCALC, 131200, 80);
            DataCopyExtParams one{1, 2, 0, 0, 0};
            constexpr bool reverseWeights = false, reverseBias = false;
            DataCopyPad(coeff, w1, one, pad);
            DataCopyPad(coeff[16], w2, one, pad);
            DataCopyPad(coeff[32], b1, one, pad);
            DataCopyPad(coeff[48], b2, one, pad);
            if (t.rows == 1) {
                DataCopyPad(input, x, one, pad);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                SetVectorMask<T>(1);
                BinaryRepeatParams binary{1, 1, 1, 1, 1, 0};
                Mul<T, false>(a, coeff, input, MASK_PLACEHOLDER, 2, binary);
                PipeBarrier<PIPE_V>();
                binary.src1RepStride = 1;
                if (reverseWeights == reverseBias) {
                    Add<T, false>(a, a, coeff[32], MASK_PLACEHOLDER, 2, binary);
                } else {
                    Add(a, a, coeff[48], 1);
                    Add(a[16], a[16], coeff[32], 1);
                }
                PipeBarrier<PIPE_V>();
                HalfGateSwiGlu(out, a[reverseWeights ? 16 : 0], a[reverseWeights ? 0 : 16], 1);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(y, out, one);
                return;
            }
            T ww1 = 0, ww2 = 0, bb1 = 0, bb2 = 0;
            bool first = true;
            for (uint32_t start = GetBlockIdx() * t.tileRows; start < t.rows; start += t.cores * t.tileRows) {
                if (!first) PipeBarrier<PIPE_ALL>();
                const uint32_t count = t.rows - start < t.tileRows ? t.rows - start : t.tileRows;
                DataCopyPad(input, x[start], DataCopyExtParams{1, count * 2, 0, 0, 0}, pad);
                if (first) {
                    SetFlag<HardEvent::MTE2_S>(0);
                    WaitFlag<HardEvent::MTE2_S>(0);
                    ww1 = coeff.GetValue(reverseWeights ? 16 : 0);
                    ww2 = coeff.GetValue(reverseWeights ? 0 : 16);
                    bb1 = coeff.GetValue(reverseBias ? 48 : 32);
                    bb2 = coeff.GetValue(reverseBias ? 32 : 48);
                    first = false;
                }
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                Muls(a, input, ww1, count);
                Muls(b, input, ww2, count);
                PipeBarrier<PIPE_V>();
                Adds(a, a, bb1, count);
                Adds(b, b, bb2, count);
                PipeBarrier<PIPE_V>();
                HalfGateSwiGlu(out, a, b, count);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(y[start], out, DataCopyExtParams{1, count * 2, 0, 0, 0});
            }
        } else {
            const uint32_t dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
            const uint32_t capacity = STATIC_DIM == 32 ? 32 : (STATIC_DIM == 0 ? t.tileCols : 16);
            const uint32_t pitch = (dim + 15) / 16 * 16;
            const uint32_t tileColumns = capacity;
            const uint32_t nTiles = (dim + tileColumns - 1) / tileColumns;
            LocalTensor<T> input(TPosition::VECCALC, 0, (2 * capacity + 1) * pitch + 2 * capacity);
            LocalTensor<float> wf(TPosition::VECCALC, 36864, (2 * capacity + 1) * pitch + 8 * capacity);
            auto sums = wf[(2 * capacity + 1) * pitch];
            LocalTensor<T> a(TPosition::VECCALC, 114688, 2 * capacity);
            auto b = a[capacity];
            LocalTensor<T> out(TPosition::VECCALC, 114944, capacity);
            const uint32_t gatherRows = dim < capacity ? dim : capacity;
            const bool gatherWeights = dim > 3 && pitch != dim &&
                gatherRows * pitch <= SWIGLU_GATHER_CAPACITY;
            LocalTensor<T> raw(TPosition::VECCALC, 131072, 2 * capacity * dim);
            LocalTensor<uint32_t> offsets(TPosition::VECCALC, 163840, capacity * pitch);
            bool first = true;
            for (uint32_t tile = GetBlockIdx(); ONE_TILE || tile < t.rows * nTiles; tile += t.cores) {
                const bool firstTile = first;
                if (!first) PipeBarrier<PIPE_ALL>();
                first = false;
                const uint32_t row = tile / nTiles, col = tile % nTiles * tileColumns;
                const uint32_t cols = STATIC_DIM != 0 && STATIC_DIM % tileColumns == 0 ?
                    tileColumns : (dim - col < tileColumns ? dim - col : tileColumns);
                const uint64_t weightOffset = static_cast<uint64_t>(col) * dim;
                DataCopyExtParams matrixCopy{static_cast<uint16_t>(cols), dim * 2, 0, 0, 0};
                if (pitch == dim) {
                    matrixCopy.blockCount = 1;
                    matrixCopy.blockLen = cols * dim * 2;
                }
                DataCopyPadExtParams<T> matrixPad{true, 0, static_cast<uint8_t>(pitch - dim), 0};
                bool reverseWeights = false;
                if (gatherWeights) {
                    reverseWeights = CopyOrderedTensorPair(raw, capacity * dim, w1[weightOffset], w2[weightOffset],
                        cols * dim, pad);
                } else if (pitch == dim) {
                    if constexpr (STATIC_DIM == 32 || STATIC_DIM == 64) {
                        DataCopyPad(input, w1[weightOffset], matrixCopy, matrixPad);
                        DataCopyPad(input[cols * pitch], w2[weightOffset], matrixCopy, matrixPad);
                    } else {
                        reverseWeights = CopyOrderedTensorPair(input, cols * pitch, w1[weightOffset], w2[weightOffset],
                            cols * dim, matrixPad);
                    }
                } else {
                    DataCopyPad(input, w1[weightOffset], matrixCopy, matrixPad);
                    DataCopyPad(input[cols * pitch], w2[weightOffset], matrixCopy, matrixPad);
                }
                DataCopyPad(input[2 * cols * pitch], x[static_cast<uint64_t>(row) * dim],
                    DataCopyExtParams{1, dim * 2, 0, 0, 0}, matrixPad);
                DataCopyExtParams biasCopy{1, cols * 2, 0, 0, 0};
                bool reverseBias = false;
                if constexpr (STATIC_DIM == 64) {
                    DataCopyPad(input[(2 * capacity + 1) * pitch], b1[col], biasCopy, pad);
                    DataCopyPad(input[(2 * capacity + 1) * pitch + capacity], b2[col], biasCopy, pad);
                } else if constexpr (STATIC_DIM == 32) {
                    reverseBias = reverseWeights;
                    DataCopyPad(input[(2 * capacity + 1) * pitch + (reverseWeights ? capacity : 0)],
                        b1[col], biasCopy, pad);
                    DataCopyPad(input[(2 * capacity + 1) * pitch + (reverseWeights ? 0 : capacity)],
                        b2[col], biasCopy, pad);
                } else {
                    reverseBias = CopyOrderedTensorPair(input[(2 * capacity + 1) * pitch], capacity,
                        b1[col], b2[col], cols, pad);
                }
                SetFlag<HardEvent::MTE2_V>(0);
                if (firstTile && gatherWeights) MakeGatherOffsets(offsets, dim, pitch, gatherRows);
                WaitFlag<HardEvent::MTE2_V>(0);
                if (gatherWeights) {
                    Gather(input, raw, offsets, 0, cols * pitch);
                    Gather(input[cols * pitch], raw[capacity * dim], offsets, 0, cols * pitch);
                    PipeBarrier<PIPE_V>();
                }
                auto xf = wf[2 * cols * pitch];
                if constexpr (STATIC_DIM == 32) {
                    Cast(wf, input, RoundMode::CAST_NONE, 2 * cols * pitch);
                    Cast(xf, input[2 * cols * pitch], RoundMode::CAST_NONE, 32, 2,
                        UnaryRepeatParams{1, 1, 4, 0});
                } else {
                    Cast(wf, input, RoundMode::CAST_NONE, (2 * cols + 1) * pitch);
                }
                PipeBarrier<PIPE_V>();
                BinaryRepeatParams mulParams{1, 1, 1,
                    static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch / 8), 0};
                if constexpr (STATIC_DIM == 32) {
                    Mul(wf, wf, xf, 64, cols, BinaryRepeatParams{1, 1, 1, 8, 8, 0});
                } else {
                    for (uint32_t c = 0; c < dim; c += 64) {
                        const uint32_t width = dim - c < 64 ? dim - c : 64;
                        Mul(wf[c], wf[c], xf[c], width, 2 * cols, mulParams);
                    }
                }
                PipeBarrier<PIPE_V>();
                if constexpr (STATIC_DIM == 32) {
                    LocalTensor<float> partial(TPosition::VECCALC, 65536, 4 * capacity);
                    BlockReduceSum(sums, wf, cols, 64, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                    PairReduceSum(partial, sums, 2 * cols / 16, 64, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                    PairReduceSum(sums, partial, cols / 16, 64, 1, 1, 8);
                } else if constexpr (STATIC_DIM == 64) {
                    LocalTensor<float> blockSums(TPosition::VECCALC, 65536, 256);
                    LocalTensor<float> partial(TPosition::VECCALC, 69632, 128);
                    BlockReduceSum(blockSums, wf, 32, 64, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                    BlockReduceSum(sums, blockSums, 4, 64, 1, 1, 8);
                } else {
                    for (uint32_t c = 0; c < dim; c += 64) {
                        const uint32_t width = dim - c < 64 ? dim - c : 64;
                        if (cols == capacity) {
                            WholeReduceSum(sums[c / 64 * 2 * capacity], wf[c], width, 2 * capacity, 1, 1, pitch / 8);
                        } else {
                            WholeReduceSum(sums[c / 64 * 2 * capacity], wf[c], width, cols, 1, 1, pitch / 8);
                            WholeReduceSum(sums[c / 64 * 2 * capacity + capacity], wf[cols * pitch + c], width,
                                           cols, 1, 1, pitch / 8);
                        }
                    }
                }
                PipeBarrier<PIPE_V>();
                for (uint32_t c = 64; c < dim; c += 64) {
                    Add(sums, sums, sums[c / 64 * 2 * capacity], 2 * capacity);
                    PipeBarrier<PIPE_V>();
                }
                Cast(a, sums, RoundMode::CAST_RINT, 2 * capacity);
                PipeBarrier<PIPE_V>();
                if (reverseWeights == reverseBias) {
                    Add(a, a, input[(2 * capacity + 1) * pitch], 2 * capacity);
                } else {
                    Add(a, a, input[(2 * capacity + 1) * pitch + capacity], capacity);
                    Add(b, b, input[(2 * capacity + 1) * pitch], capacity);
                }
                PipeBarrier<PIPE_V>();
                HalfGateSwiGlu(out, reverseWeights ? b : a, reverseWeights ? a : b, cols);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(y[static_cast<uint64_t>(row) * dim + col], out,
                    DataCopyExtParams{1, cols * 2, 0, 0, 0});
                if constexpr (ONE_TILE) break;
            }
        }
        if constexpr (!ONE_TILE) PipeBarrier<PIPE_ALL>();
    }
private:
    SwiGluVectorTilingData t;
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};

// The current counter mask is shared by independent gather instructions.
template <typename T>
__aicore__ inline void GatherMaskCurrentMode(AscendC::LocalTensor<T> dst,
        AscendC::LocalTensor<T> src,AscendC::LocalTensor<uint16_t> pattern) {
    AscendC::GatherMask(dst.template ReinterpretCast<uint16_t>(),
        src.template ReinterpretCast<uint16_t>(),pattern,static_cast<uint8_t>(0),
        AscendC::GatherMaskParams{1,1,8,0});
}

template <typename T, uint32_t ROWS, bool PADDED = false, bool LONG_K = false,
    uint32_t STATIC_DIM = 0, bool ONE_TILE = false, bool FULL_ROWS = false,
    uint32_t COL_TILE = 16, uint32_t LAST_ROWS = 0, bool FAST33 = false, uint32_t PADDED_BLOCKS = 0>
class BatchedVectorSwiGlu {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
            GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output,
            const SwiGluVectorTilingData &data, GM_ADDR tiling) {
        t = data;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight2));
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    }
    __aicore__ inline void Process() {
        using namespace AscendC;
        static_assert(ROWS == 1 || ROWS == 2 || ROWS == 3 || ROWS == 4 || ROWS == 7 || ROWS == 8 || ROWS == 16 || ROWS == 32);
        static_assert(!LONG_K || ROWS == 4);
        if constexpr(FAST33 && ROWS==8) { if(GetBlockIdx()>=GetBlockNum()-PADDED_BLOCKS) return; }
        constexpr uint32_t COLS = COL_TILE;
        constexpr uint32_t OUT_PITCH = (COLS + 15) / 16 * 16;
        constexpr bool WIDE = COLS > 16;
        constexpr bool SPLIT33 = STATIC_DIM == 33 && WIDE;
        const uint32_t dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
        const uint32_t pitch = SPLIT33 ? 32 : (PADDED ? (dim + 15) / 16 * 16 : dim);
        const uint32_t nTiles = PADDED ? (dim + COLS - 1) / COLS : dim / COLS;
        const uint32_t tiles = ((t.rows + ROWS - 1) / ROWS) * nTiles;
        const uint32_t weightCount = 2 * COLS * pitch;
        LocalTensor<T> input(TPosition::VECCALC, 0, (2 * COLS + ROWS) * pitch);
        LocalTensor<T> bias(TPosition::VECCALC, LONG_K ? 12288 : 8192, 2 * OUT_PITCH);
        LocalTensor<float> weights(TPosition::VECCALC, 16384, weightCount);
        LocalTensor<float> inputs(TPosition::VECCALC, LONG_K ? 104512 : (WIDE ? 36864 : 24640), ROWS * pitch);
        LocalTensor<float> products(TPosition::VECCALC, WIDE ? 49152 : 32768, ROWS * weightCount);
        LocalTensor<float> sums(TPosition::VECCALC, FAST33 && ROWS==8 ? 132096 : 98304, ROWS * 2 * COLS * (LONG_K ? 2 : 1));
        LocalTensor<T> a(TPosition::VECCALC, FAST33 && ROWS==8 ? 135168 : 114688, ROWS * OUT_PITCH);
        LocalTensor<T> b(TPosition::VECCALC, (FAST33 && ROWS==8 ? 135168 : 114688) + ROWS * OUT_PITCH * sizeof(T), ROWS * OUT_PITCH);
        LocalTensor<T> out(TPosition::VECCALC, (FAST33 && ROWS==8 ? 135232 : 114752) + 2 * ROWS * OUT_PITCH * sizeof(T), ROWS * OUT_PITCH);
        const uint32_t rawPlane = (COLS * dim + 15) / 16 * 16;
        LocalTensor<T> raw(TPosition::VECCALC, 131072, 2 * rawPlane + (FAST33 ? (ROWS*33+15)/16*16 : ROWS*pitch));
        LocalTensor<uint32_t> offsets(TPosition::VECCALC, 163840, COLS * pitch);
        LocalTensor<uint64_t> packMask64(TPosition::VECCALC,179200,FAST33 && ROWS==8 ? 6 : 4);
        if constexpr (!(FAST33 && ROWS==8) && STATIC_DIM == 33 && (ROWS == 2 || ROWS == 3 || (FAST33 && ROWS == 8)) && WIDE && ONE_TILE) {
            if constexpr(FAST33 && ROWS==8) {
                packMask64.SetValue(0,0xffff0001ffffffffULL);
                packMask64.SetValue(1,0xffffffff0001ffffULL);
                packMask64.SetValue(2,0x0001ffffffff0001ULL);
                packMask64.SetValue(3,0xffff0001ffffffffULL);
                packMask64.SetValue(4,0xffffffff0001ffffULL);
                packMask64.SetValue(5,0x0001ffffffff0001ULL);
            } else {
            // Select 33 valid values from each padded 48-element row.
            packMask64.SetValue(0,0xffff0001ffffffffULL);
            packMask64.SetValue(1,ROWS == 2 ? 0x000000000001ffffULL : 0xffffffff0001ffffULL);
            if constexpr (ROWS == 3) {
                packMask64.SetValue(2,1ULL);
                packMask64.SetValue(3,0ULL);
            }
            }
            SetFlag<HardEvent::S_V>(6);
        }
        LocalTensor<uint32_t> tailIndices(TPosition::VECCALC,184320,COLS);
        LocalTensor<uint64_t> mainMask64(TPosition::VECCALC,163840,24);
        LocalTensor<uint64_t> tailMask64(TPosition::VECCALC,164096,24);
        LocalTensor<T> tailHalf(TPosition::VECCALC,FAST33 && ROWS==8 ? 41728 : 94208,2*OUT_PITCH+16);
        LocalTensor<float> tailWeights(TPosition::VECCALC,FAST33 && ROWS==8 ? 40960 : 86016,2*COLS);
        LocalTensor<float> tailProducts(TPosition::VECCALC,81920,ROWS*2*COLS);
        LocalTensor<float> tailX(TPosition::VECCALC,FAST33 && ROWS==8 ? 41280 : 91136,8);
        LocalTensor<float> tailBroadcast(TPosition::VECCALC,FAST33 && ROWS==8 ? 41472 : 90112,64);
        uint32_t cachedCol = UINT32_MAX;
        bool reverseWeights = false, reverseBias = false;
        bool first = true;
        for (uint32_t tile = GetBlockIdx(); ONE_TILE || tile < tiles; tile += t.cores) {
            const bool firstTile = first;
            if (!first) PipeBarrier<PIPE_ALL>();
            first = false;
            const uint32_t row = tile / nTiles * ROWS;
            const uint32_t col = tile % nTiles * COLS;
            const uint32_t nr = LAST_ROWS != 0 ?
                ((FAST33 && ROWS==8 ? GetBlockIdx()+1==GetBlockNum()-PADDED_BLOCKS : GetBlockIdx()+1==GetBlockNum()) ? LAST_ROWS : ROWS) :
                (FULL_ROWS ? ROWS : (t.rows - row < ROWS ? t.rows - row : ROWS));
            const uint32_t cols = PADDED && dim - col < COLS ? dim - col : COLS;
            const bool reload = cachedCol != col;
            if constexpr (PADDED) {
                DataCopyPadExtParams<T> noPad{false, 0, 0, 0};
                if (reload) {
                    reverseWeights = CopyOrderedTensorPair(raw, rawPlane, w1[static_cast<uint64_t>(col) * dim],
                        w2[static_cast<uint64_t>(col) * dim], cols * dim, noPad);
                    DataCopyPadExtParams<T> biasPad{true, 0, static_cast<uint8_t>(OUT_PITCH - cols), 0};
                    reverseBias = CopyOrderedTensorPair(bias, OUT_PITCH, b1[col], b2[col], cols, biasPad);
                }
                if constexpr (STATIC_DIM == 13 && !((ROWS == 3 || ROWS == 7) && FULL_ROWS)) {
                    DataCopyExtParams xCopy{static_cast<uint16_t>(nr), dim * 2, 0, 0, 0};
                    DataCopyPadExtParams<T> xPad{true, 0,
                        static_cast<uint8_t>(pitch - dim), 0};
                    DataCopyPad(input[weightCount], x[static_cast<uint64_t>(row) * dim],
                        xCopy, xPad);
                } else {
                    DataCopyPad(raw[2 * rawPlane], x[static_cast<uint64_t>(row) * dim],
                        DataCopyExtParams{1, nr * dim * 2, 0, 0, 0}, noPad);
                }
            } else {
                if (reload) {
                    DataCopy(input, w1[static_cast<uint64_t>(col) * dim], COLS * dim);
                    DataCopy(input[COLS * dim], w2[static_cast<uint64_t>(col) * dim], COLS * dim);
                    DataCopy(bias, b1[col], COLS);
                    DataCopy(bias[COLS], b2[col], COLS);
                }
                DataCopy(input[weightCount], x[static_cast<uint64_t>(row) * dim], nr * dim);
            }
            SetFlag<HardEvent::MTE2_V>(0);
            if constexpr(FAST33 && ROWS==8) {
            if constexpr(FAST33 && ROWS==8) {
                packMask64.SetValue(0,0xffff0001ffffffffULL);
                packMask64.SetValue(1,0xffffffff0001ffffULL);
                packMask64.SetValue(2,0x0001ffffffff0001ULL);
                packMask64.SetValue(3,0xffff0001ffffffffULL);
                packMask64.SetValue(4,0xffffffff0001ffffULL);
                packMask64.SetValue(5,0x0001ffffffff0001ULL);
            }
            SetFlag<HardEvent::S_V>(6);
            }
            if constexpr (PADDED) {
                if (firstTile) {
                    const uint32_t weightRows = dim < COLS ? dim : COLS;
                    const uint32_t gatherRows = weightRows > ROWS ? weightRows : ROWS;
                    if constexpr (SPLIT33) {
                        // Main K=32 and final K=1 are complementary selections
                        // from contiguous rows of length33. No index-vector construction.
                        if constexpr(FAST33) {
                            // 33 rows*33 elements need 18 uint64 pattern words.
                        tailMask64.SetValue(0,0x0000000100000000ULL);
                        tailMask64.SetValue(1,0x0000000400000002ULL);
                        tailMask64.SetValue(2,0x0000001000000008ULL);
                        tailMask64.SetValue(3,0x0000004000000020ULL);
                        tailMask64.SetValue(4,0x0000010000000080ULL);
                        tailMask64.SetValue(5,0x0000040000000200ULL);
                        tailMask64.SetValue(6,0x0000100000000800ULL);
                        tailMask64.SetValue(7,0x0000400000002000ULL);
                        tailMask64.SetValue(8,0x0001000000008000ULL);
                        tailMask64.SetValue(9,0x0004000000020000ULL);
                        tailMask64.SetValue(10,0x0010000000080000ULL);
                        tailMask64.SetValue(11,0x0040000000200000ULL);
                        tailMask64.SetValue(12,0x0100000000800000ULL);
                        tailMask64.SetValue(13,0x0400000002000000ULL);
                        tailMask64.SetValue(14,0x1000000008000000ULL);
                        tailMask64.SetValue(15,0x4000000020000000ULL);
                        tailMask64.SetValue(16,0x0000000080000000ULL);
                        tailMask64.SetValue(17,0x0000000200000001ULL);
                        } else {
                        mainMask64.SetValue(0,0xfffffffeffffffffULL);
                        tailMask64.SetValue(0,0x0000000100000000ULL);
                        mainMask64.SetValue(1,0xfffffffbfffffffdULL);
                        tailMask64.SetValue(1,0x0000000400000002ULL);
                        mainMask64.SetValue(2,0xffffffeffffffff7ULL);
                        tailMask64.SetValue(2,0x0000001000000008ULL);
                        mainMask64.SetValue(3,0xffffffbfffffffdfULL);
                        tailMask64.SetValue(3,0x0000004000000020ULL);
                        mainMask64.SetValue(4,0xfffffeffffffff7fULL);
                        tailMask64.SetValue(4,0x0000010000000080ULL);
                        mainMask64.SetValue(5,0xfffffbfffffffdffULL);
                        tailMask64.SetValue(5,0x0000040000000200ULL);
                        mainMask64.SetValue(6,0xffffeffffffff7ffULL);
                        tailMask64.SetValue(6,0x0000100000000800ULL);
                        mainMask64.SetValue(7,0xffffbfffffffdfffULL);
                        tailMask64.SetValue(7,0x0000400000002000ULL);
                        mainMask64.SetValue(8,0xfffeffffffff7fffULL);
                        tailMask64.SetValue(8,0x0001000000008000ULL);
                        mainMask64.SetValue(9,0xfffbfffffffdffffULL);
                        tailMask64.SetValue(9,0x0004000000020000ULL);
                        mainMask64.SetValue(10,0xffeffffffff7ffffULL);
                        tailMask64.SetValue(10,0x0010000000080000ULL);
                        mainMask64.SetValue(11,0xffbfffffffdfffffULL);
                        tailMask64.SetValue(11,0x0040000000200000ULL);
                        mainMask64.SetValue(12,0xfeffffffff7fffffULL);
                        tailMask64.SetValue(12,0x0100000000800000ULL);
                        mainMask64.SetValue(13,0xfbfffffffdffffffULL);
                        tailMask64.SetValue(13,0x0400000002000000ULL);
                        mainMask64.SetValue(14,0xeffffffff7ffffffULL);
                        tailMask64.SetValue(14,0x1000000008000000ULL);
                        mainMask64.SetValue(15,0xbfffffffdfffffffULL);
                        tailMask64.SetValue(15,0x4000000020000000ULL);
                        mainMask64.SetValue(16,0xffffffff7fffffffULL);
                        tailMask64.SetValue(16,0x0000000080000000ULL);
                        mainMask64.SetValue(17,0xfffffffdfffffffeULL);
                        tailMask64.SetValue(17,0x0000000200000001ULL);
                        mainMask64.SetValue(18,0xfffffff7fffffffbULL);
                        tailMask64.SetValue(18,0x0000000800000004ULL);
                        mainMask64.SetValue(19,0xffffffdfffffffefULL);
                        tailMask64.SetValue(19,0x0000002000000010ULL);
                        mainMask64.SetValue(20,0xffffff7fffffffbfULL);
                        tailMask64.SetValue(20,0x0000008000000040ULL);
                        }
                        SetFlag<HardEvent::S_V>(7); WaitFlag<HardEvent::S_V>(7);
                        if constexpr(FAST33) {
                            Not(mainMask64.template ReinterpretCast<uint16_t>(),
                                tailMask64.template ReinterpretCast<uint16_t>(),72);
                            PipeBarrier<PIPE_V>();
                        }
                    } else {
                        MakeGatherOffsets(offsets, dim, pitch, gatherRows);
                    }
                }
            }
            WaitFlag<HardEvent::MTE2_V>(0);
            if constexpr (PADDED) {
                if constexpr (SPLIT33) {
                    auto mainMask=mainMask64.template ReinterpretCast<uint16_t>();
                    auto tailMask=tailMask64.template ReinterpretCast<uint16_t>();
                    SetMaskCount();
                    if(reload) {
                        SetVectorMask<uint16_t,MaskMode::COUNTER>(cols*33);
                        GatherMaskCurrentMode(input,raw,mainMask);
                        GatherMaskCurrentMode(input[COLS*pitch],raw[rawPlane],mainMask);
                        GatherMaskCurrentMode(tailHalf,raw,tailMask);
                        GatherMaskCurrentMode(tailHalf[OUT_PITCH],raw[rawPlane],tailMask);
                    }
                    SetVectorMask<uint16_t,MaskMode::COUNTER>(nr*33);
                    GatherMaskCurrentMode(input[weightCount],raw[2*rawPlane],mainMask);
                    GatherMaskCurrentMode(tailHalf[2*OUT_PITCH],raw[2*rawPlane],tailMask);
                    SetMaskNorm();
                } else {
                    if (reload) {
                        Gather(input, raw, offsets, 0, cols * pitch);
                        Gather(input[COLS * pitch], raw[rawPlane], offsets, 0, cols * pitch);
                    }
                    if constexpr (STATIC_DIM != 13 || ((ROWS == 3 || ROWS == 7) && FULL_ROWS)) {
                        Gather(input[weightCount], raw[2 * rawPlane], offsets, 0, nr * pitch);
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            if (reload) {
                Cast(weights, input, RoundMode::CAST_NONE, weightCount);
                cachedCol = col;
            }
            if constexpr (STATIC_DIM == 13) {
                LocalTensor<float> expanded(TPosition::VECCALC, 28672, ROWS * 64);
                for (uint32_t rr = 0; rr < nr; ++rr) {
                    Cast(expanded[rr * 64], input[weightCount + rr * pitch], RoundMode::CAST_NONE,
                        16, 4, UnaryRepeatParams{1, 1, 2, 0});
                }
            } else if constexpr (SPLIT33) {
                LocalTensor<float> expanded(TPosition::VECCALC,FAST33 && ROWS==8 ? 36864 : 102400,ROWS*64);
                if(reload) {
                    Cast(tailWeights,tailHalf,RoundMode::CAST_NONE,COLS);
                    Cast(tailWeights[COLS],tailHalf[OUT_PITCH],RoundMode::CAST_NONE,COLS);
                }
                Cast(tailX,tailHalf[2*OUT_PITCH],RoundMode::CAST_NONE,nr);
                if constexpr(FAST33 && ROWS==8) {
                    // Two row-strided casts produce [Xrow,Xrow] for each row.
                    Cast(expanded,input[weightCount],RoundMode::CAST_NONE,
                        32,nr,UnaryRepeatParams{1,1,8,2});
                    Cast(expanded[32],input[weightCount],RoundMode::CAST_NONE,
                        32,nr,UnaryRepeatParams{1,1,8,2});
                } else {
                for(uint32_t rr=0;rr<nr;++rr) {
                    Cast(expanded[rr*64],input[weightCount+rr*pitch],RoundMode::CAST_NONE,
                        32,2,UnaryRepeatParams{1,1,4,0});
                }
                }
            } else {
                Cast(inputs, input[weightCount], RoundMode::CAST_NONE, nr * pitch);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (SPLIT33) {
                Brcb(tailBroadcast,tailX,1,BrcbRepeatParams{1,8});
                PipeBarrier<PIPE_V>();
            }
            BinaryRepeatParams mulParams{1, 1, 1, static_cast<uint8_t>(pitch / 8),
                static_cast<uint8_t>(pitch / 8), 0};
            if constexpr (LONG_K) {
                for (uint32_t rr = 0; rr < nr; ++rr) {
                    for (uint32_t c = 0; c < dim; c += 64) {
                        const uint32_t width = dim - c < 64 ? dim - c : 64;
                        Mul(products[rr * weightCount + c], weights[c], inputs[rr * pitch + c],
                            width, 2 * COLS, mulParams);
                    }
                }
                PipeBarrier<PIPE_V>();
                for (uint32_t c = 0; c < dim; c += 64) {
                    const uint32_t width = dim - c < 64 ? dim - c : 64;
                    WholeReduceSum(sums[c / 64 * ROWS * 2 * COLS], products[c], width,
                        nr * 2 * COLS, 1, 1, pitch / 8);
                }
                PipeBarrier<PIPE_V>();
                // Match the original vector path's two 64-element partial sums.
                Add(sums, sums, sums[ROWS * 2 * COLS], nr * 2 * COLS);
                PipeBarrier<PIPE_V>();
            } else {
                if constexpr (STATIC_DIM == 13) {
                    LocalTensor<float> expanded(TPosition::VECCALC, 28672, ROWS * 64);
                    for (uint32_t rr = 0; rr < nr; ++rr) {
                        Mul(products[rr * weightCount], weights, expanded[rr * 64], 64, 8,
                            BinaryRepeatParams{1, 1, 1, 8, 8, 0});
                    }
                } else if constexpr (SPLIT33) {
                    LocalTensor<float> expanded(TPosition::VECCALC,FAST33 && ROWS==8 ? 36864 : 102400,ROWS*64);
                    for(uint32_t rr=0;rr<nr;++rr) {
                        Mul(products[rr*weightCount],weights,expanded[rr*64],64,COLS,
                            BinaryRepeatParams{1,1,1,8,8,0});
                        if constexpr(!FAST33) {
                        Mul(tailProducts[rr*2*COLS],tailWeights,tailBroadcast[rr*8],COLS,2,
                            BinaryRepeatParams{1,1,0,COLS/8,COLS/8,0});
                        }
                    }
                } else {
                    for (uint32_t rr = 0; rr < nr; ++rr) {
                        Mul(products[rr * weightCount], weights, inputs[rr * pitch], dim, 2 * COLS, mulParams);
                    }
                }
                PipeBarrier<PIPE_V>();
                if constexpr (STATIC_DIM == 13) {
                    LocalTensor<float> blockSums(TPosition::VECCALC, 65536, ROWS * 64);
                    // Ignore the three padded K lanes in each 16-element row
                    // directly in the reduction, avoiding a clearing pass.
                    uint64_t validMask[2] = {0x1fff1fff1fff1fffULL, 0};
                    BlockReduceSum(blockSums, products, nr * 8, validMask, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                    PairReduceSum(sums, blockSums, nr, 64, 1, 1, 8);
                } else if constexpr (SPLIT33) {
                    // Each 32-product dot yields four adjacent 8-lane sums.
                    LocalTensor<float> blockPartial(TPosition::VECCALC,FAST33 && ROWS==8 ? 135168 : 131072,ROWS*8*COLS);
                    LocalTensor<float> pairPartial(TPosition::VECCALC,FAST33 && ROWS==8 ? 147456 : 139264,ROWS*4*COLS);
                    if constexpr(FAST33 && ROWS==8) {
                        // At most6rows*40=240 repeats per instruction.
                        const uint32_t first=nr>6?6:nr;
                        BlockReduceSum(blockPartial,products,first*COLS,64,1,1,8);
                        if(nr>6) BlockReduceSum(blockPartial[6*8*COLS],
                            products[6*weightCount],(nr-6)*COLS,64,1,1,8);
                    } else {
                    BlockReduceSum(blockPartial,products,nr*COLS,64,1,1,8);
                    }
                    PipeBarrier<PIPE_V>();
                    PairReduceSum(pairPartial,blockPartial,nr*COLS/8,64,1,1,8);
                    PipeBarrier<PIPE_V>();
                    const uint32_t pairCount=nr*4*COLS;
                    PairReduceSum(sums,pairPartial,pairCount/64,64,1,1,8);
                    if(pairCount%64) {
                        PairReduceSum(sums[pairCount/64*32],pairPartial[pairCount/64*64],1,pairCount%64,1,1,8);
                    }
                    PipeBarrier<PIPE_V>();
                    if constexpr(FAST33) {
                        // Each tail product is exactly representable in FP32:
                        // its operands are half values converted to float.
                        if constexpr(ROWS==8) {
                            // Each Brcb row has eight copies of the same X33.
                            // Advance output rows, reuse all80 tail weights.
                            const BinaryRepeatParams rowMac{1,1,0,2*COLS/8,0,1};
                            MulAddDst<float,float>(sums,tailWeights,tailBroadcast,
                                64,nr,rowMac);
                            MulAddDst<float,float>(sums[64],tailWeights[64],tailBroadcast,
                                2*COLS-64,nr,rowMac);
                        } else {
                        for(uint32_t rr=0;rr<nr;++rr) {
                            MulAddDst<float,float>(sums[rr*2*COLS],tailWeights,
                                tailBroadcast[rr*8],COLS,2,
                                BinaryRepeatParams{1,1,0,COLS/8,COLS/8,0});
                        }
                        }
                    } else {
                    Add(sums,sums,tailProducts,nr*2*COLS);
                    }
                } else if constexpr (ROWS == 4) {
                    WholeReduceSum(sums, products, dim, nr * 2 * COLS, 1, 1, pitch / 8);
                } else {
                    // The instruction accepts at most 255 repeats.
                    for (uint32_t rr = 0; rr < nr; rr += (WIDE ? 3 : 4)) {
                        const uint32_t count = nr - rr < (WIDE ? 3 : 4) ? nr - rr : (WIDE ? 3 : 4);
                        WholeReduceSum(sums[rr * 2 * COLS], products[rr * weightCount], dim,
                            count * 2 * COLS, 1, 1, pitch / 8);
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            UnaryRepeatParams castParams{1, 1, OUT_PITCH / 16, 2 * COLS / 8};
            Cast(a, sums, RoundMode::CAST_RINT, COLS, nr, castParams);
            Cast(b, sums[COLS], RoundMode::CAST_RINT, COLS, nr, castParams);
            PipeBarrier<PIPE_V>();
            BinaryRepeatParams biasParams{1, 1, 1, OUT_PITCH / 16, OUT_PITCH / 16, 0};
            const bool swapBias = reverseWeights != reverseBias;
            Add(a, a, bias[swapBias ? OUT_PITCH : 0], COLS, nr, biasParams);
            Add(b, b, bias[swapBias ? 0 : OUT_PITCH], COLS, nr, biasParams);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint16_t> packMask(TPosition::VECCALC, 163840, 16);
            if constexpr (STATIC_DIM == 13 && (ROWS == 3 || ROWS == 7) && FULL_ROWS) {
                Duplicate(packMask, static_cast<uint16_t>(0x1fff), 16);
            }
            if constexpr(FAST33) {
                LocalTensor<T> packedA(TPosition::VECCALC,ROWS==8 ? 153600 : 131072,ROWS*48);
                LocalTensor<T> packedB(TPosition::VECCALC,ROWS==8 ? 154400 : 131392,ROWS*48);
                WaitFlag<HardEvent::S_V>(6);
                auto pattern=packMask64.template ReinterpretCast<uint16_t>();
                SetMaskCount();SetVectorMask<uint16_t,MaskMode::COUNTER>(nr*48);
                GatherMaskCurrentMode(packedA,a,pattern);
                GatherMaskCurrentMode(packedB,b,pattern);
                SetMaskNorm();PipeBarrier<PIPE_V>();
                HalfGateSwiGlu(out,reverseWeights?packedB:packedA,
                    reverseWeights?packedA:packedB,nr*33);
                SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(y[static_cast<uint64_t>(row)*33],out,
                    DataCopyExtParams{1,nr*66,0,0,0});
            } else {
            HalfGateSwiGlu(out, reverseWeights ? b : a, reverseWeights ? a : b, nr * OUT_PITCH);
            if constexpr (STATIC_DIM == 13 && (ROWS == 3 || ROWS == 7) && FULL_ROWS) {
                // Compress padded rows into one contiguous output copy.
                LocalTensor<T> packed(TPosition::VECCALC, 131072, ROWS*16);
                PipeBarrier<PIPE_V>();
                uint64_t selected=0;
                GatherMask(packed,out,packMask,true,ROWS*16,GatherMaskParams{1,1,8,0},selected);
                SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(y[static_cast<uint64_t>(row)*13],packed,DataCopyExtParams{1,ROWS*26,0,0,0});
            } else if constexpr (STATIC_DIM == 33 && (ROWS == 2 || ROWS == 3) && WIDE && ONE_TILE) {
                LocalTensor<T> packed(TPosition::VECCALC,131072,ROWS*48);
                auto packMask=packMask64.template ReinterpretCast<uint16_t>();
                PipeBarrier<PIPE_V>(); WaitFlag<HardEvent::S_V>(6);
                uint64_t selected=0;
                GatherMask(packed,out,packMask,true,nr*48,GatherMaskParams{1,1,8,0},selected);
                SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(y[static_cast<uint64_t>(row)*33],packed,DataCopyExtParams{1,nr*66,0,0,0});
            } else {
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyExtParams outCopy{static_cast<uint16_t>(nr), cols * 2, 0, (dim - cols) * 2, 0};
                DataCopyPad(y[static_cast<uint64_t>(row) * dim + col], out, outCopy);
            }
            }
            if constexpr (ONE_TILE) break;
        }
        if constexpr (!ONE_TILE) PipeBarrier<PIPE_ALL>();
    }
private:
    SwiGluVectorTilingData t;
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};

// Two rows share each converted weight tile for the remaining vector-small
// dimensions. Products are reduced in 64-element chunks and accumulated in
// the same order as VectorSwiGlu.
template <typename T, bool PADDED, uint32_t STATIC_DIM = 0>
class PairVectorSwiGlu {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
            GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output,
            const SwiGluVectorTilingData &data) {
        t = data;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight2));
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    }

    __aicore__ inline void Process() {
        using namespace AscendC;
        constexpr uint32_t COLS = 16, ROWS = 2;
        const uint32_t dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
        const uint32_t pitch = (dim + 15) / 16 * 16;
        const uint32_t parts = (dim + 63) / 64;
        const uint32_t nTiles = (dim + COLS - 1) / COLS;
        const uint32_t tiles = ((t.rows + 1) / 2) * nTiles;
        const uint32_t weightCount = 2 * COLS * pitch;
        LocalTensor<T> input(TPosition::VECCALC, 0, (2 * COLS + ROWS) * pitch);
        LocalTensor<T> bias(TPosition::VECCALC, 18432, 2 * COLS);
        LocalTensor<float> weights(TPosition::VECCALC, 32768, weightCount);
        LocalTensor<float> products(TPosition::VECCALC, 65536, ROWS * weightCount);
        LocalTensor<float> inputs(TPosition::VECCALC, 131072, ROWS * pitch);
        LocalTensor<float> sums(TPosition::VECCALC, 135168, 4 * ROWS * 2 * COLS);
        LocalTensor<T> a(TPosition::VECCALC, 139264, ROWS * COLS);
        LocalTensor<T> b(TPosition::VECCALC, 139328, ROWS * COLS);
        LocalTensor<T> out(TPosition::VECCALC, 139392, ROWS * COLS);
        uint32_t cachedCol = UINT32_MAX;
        bool first = true;
        for (uint32_t tile = GetBlockIdx(); tile < tiles; tile += t.cores) {
            if (!first) PipeBarrier<PIPE_ALL>();
            first = false;
            const uint32_t row = tile / nTiles * ROWS;
            const uint32_t col = tile % nTiles * COLS;
            const uint32_t nr = t.rows - row < ROWS ? t.rows - row : ROWS;
            const uint32_t cols = dim - col < COLS ? dim - col : COLS;
            const bool reload = cachedCol != col;
            if (reload) {
                if constexpr (PADDED) {
                    DataCopyExtParams matrixCopy{static_cast<uint16_t>(cols), dim * 2, 0, 0, 0};
                    DataCopyPadExtParams<T> matrixPad{true, 0,
                        static_cast<uint8_t>(pitch - dim), 0};
                    DataCopyPad(input, w1[static_cast<uint64_t>(col) * dim], matrixCopy, matrixPad);
                    DataCopyPad(input[COLS * pitch], w2[static_cast<uint64_t>(col) * dim],
                        matrixCopy, matrixPad);
                } else {
                    DataCopy(input, w1[static_cast<uint64_t>(col) * dim], cols * dim);
                    DataCopy(input[COLS * pitch], w2[static_cast<uint64_t>(col) * dim],
                        cols * dim);
                }
                DataCopyExtParams biasCopy{1, cols * 2, 0, 0, 0};
                DataCopyPadExtParams<T> biasPad{true, 0, static_cast<uint8_t>(COLS - cols), 0};
                DataCopyPad(bias, b1[col], biasCopy, biasPad);
                DataCopyPad(bias[COLS], b2[col], biasCopy, biasPad);
            }
            if constexpr (PADDED) {
                DataCopyExtParams xCopy{static_cast<uint16_t>(nr), dim * 2, 0, 0, 0};
                DataCopyPadExtParams<T> xPad{true, 0, static_cast<uint8_t>(pitch - dim), 0};
                DataCopyPad(input[weightCount], x[static_cast<uint64_t>(row) * dim], xCopy, xPad);
            } else {
                DataCopy(input[weightCount], x[static_cast<uint64_t>(row) * dim], nr * dim);
            }
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            if (reload) {
                Cast(weights, input, RoundMode::CAST_NONE, weightCount);
                cachedCol = col;
            }
            Cast(inputs, input[weightCount], RoundMode::CAST_NONE, nr * pitch);
            PipeBarrier<PIPE_V>();
            BinaryRepeatParams mulParams{1, 1, 1, static_cast<uint8_t>(pitch / 8),
                static_cast<uint8_t>(pitch / 8), 0};
            for (uint32_t rr = 0; rr < nr; ++rr) {
                for (uint32_t c = 0; c < dim; c += 64) {
                    const uint32_t width = dim - c < 64 ? dim - c : 64;
                    Mul(products[rr * weightCount + c], weights[c], inputs[rr * pitch + c],
                        width, 2 * COLS, mulParams);
                }
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < dim; c += 64) {
                const uint32_t width = dim - c < 64 ? dim - c : 64;
                WholeReduceSum(sums[c / 64 * ROWS * 2 * COLS], products[c], width,
                    nr * 2 * COLS, 1, 1, pitch / 8);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t part = 1; part < parts; ++part) {
                Add(sums, sums, sums[part * ROWS * 2 * COLS], nr * 2 * COLS);
                PipeBarrier<PIPE_V>();
            }
            UnaryRepeatParams castParams{1, 1, COLS / 16, 2 * COLS / 8};
            Cast(a, sums, RoundMode::CAST_RINT, COLS, nr, castParams);
            Cast(b, sums[COLS], RoundMode::CAST_RINT, COLS, nr, castParams);
            PipeBarrier<PIPE_V>();
            BinaryRepeatParams biasParams{1, 1, 1, COLS / 16, COLS / 16, 0};
            Add(a, a, bias, COLS, nr, biasParams);
            Add(b, b, bias[COLS], COLS, nr, biasParams);
            PipeBarrier<PIPE_V>();
            HalfGateSwiGlu(out, a, b, nr * COLS);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyExtParams outCopy{static_cast<uint16_t>(nr), cols * 2, 0,
                (dim - cols) * 2, 0};
            DataCopyPad(y[static_cast<uint64_t>(row) * dim + col], out, outCopy);
        }
        PipeBarrier<PIPE_ALL>();
    }
private:
    SwiGluVectorTilingData t;
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};
