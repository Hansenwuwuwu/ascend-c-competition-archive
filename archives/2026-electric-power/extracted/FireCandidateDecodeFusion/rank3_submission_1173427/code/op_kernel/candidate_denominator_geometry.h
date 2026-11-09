#pragma once

template <uint32_t Pitch, uint32_t Tile>
__aicore__ inline void DecodeCandidateGeometry(LocalTensor<float> c, uint32_t count)
{
    if (count * 2 <= Tile) {
        for (uint32_t j = 0; j < 4; ++j)
            Muls(c[j * Pitch], c[j * Pitch], 2.0f, count);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < 2; ++j)
            Adds(c[j * Pitch], c[j * Pitch], -0.5f, count);
        Mul(c[2 * Pitch], c[2 * Pitch], c[2 * Pitch], count);
        Mul(c[3 * Pitch], c[3 * Pitch], c[3 * Pitch], count);
        Mul(c[4 * Pitch], c[4 * Pitch], c[5 * Pitch], count);
        PipeBarrier<PIPE_V>();
        Add(c, c, c[6 * Pitch], count);
        Add(c[Pitch], c[Pitch], c[7 * Pitch], count);
        Mul(c[2 * Pitch], c[2 * Pitch], c[9 * Pitch], count);
        Mul(c[3 * Pitch], c[3 * Pitch], c[10 * Pitch], count);
        PipeBarrier<PIPE_V>();
    } else {
        Muls(c, c, 2.0f, Pitch * 4);
        PipeBarrier<PIPE_V>();
        Adds(c, c, -0.5f, Pitch * 2);
        Mul(c[2 * Pitch], c[2 * Pitch], c[2 * Pitch], Pitch * 2);
        Mul(c[4 * Pitch], c[4 * Pitch], c[5 * Pitch], count);
        PipeBarrier<PIPE_V>();
        Add(c, c, c[6 * Pitch], Pitch * 2);
        Mul(c[2 * Pitch], c[2 * Pitch], c[9 * Pitch], Pitch * 2);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t j = 0; j < 4; ++j)
        Mul(c[j * Pitch], c[j * Pitch], c[8 * Pitch], count);
    PipeBarrier<PIPE_V>();
}

template <uint32_t Tile, uint32_t Pitch, bool FastConfidence = false, bool ReplicatedStride = false, bool TransformedPriors = false>
__aicore__ inline void DecodeCandidateDenominators(LocalTensor<float> columns,
    LocalTensor<float> two, LocalTensor<float> one, LocalTensor<float> firstStride)
{
    static_assert(Tile >= 1024 && Tile % 64 == 0 && (Pitch == Tile + 8 || Pitch == Tile + 16));
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(Tile);
    Mul<float, false>(columns[4 * Pitch], columns[4 * Pitch], columns[5 * Pitch],
        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    SetVectorMask<float, MaskMode::COUNTER>(Pitch * 4);
    Div<float, false>(columns, two, columns, MASK_PLACEHOLDER, 1, {1, 0, 1, 8, 0, 8});
    SetVectorMask<float, MaskMode::COUNTER>(Tile);
    if constexpr (FastConfidence) {
        Reciprocal<float, false>(columns[4 * Pitch], columns[4 * Pitch],
            MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    } else {
        Div<float, false>(columns[4 * Pitch], one, columns[4 * Pitch],
            MASK_PLACEHOLDER, 1, {1, 0, 1, 8, 0, 8});
    }
    SetVectorMask<float, MaskMode::COUNTER>(Pitch * 2);
    if constexpr (TransformedPriors) {
        Mul<float, false>(columns, columns, firstStride,
            MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], columns[2 * Pitch],
            MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        Add<float, false>(columns, columns, columns[6 * Pitch],
            MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], columns[9 * Pitch],
            MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    } else {
    Adds<float, false>(columns, columns, -0.5f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], columns[2 * Pitch],
        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    Add<float, false>(columns, columns, columns[6 * Pitch],
        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], columns[9 * Pitch],
        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    if constexpr (ReplicatedStride) {
        SetVectorMask<float, MaskMode::COUNTER>(Pitch * 4);
        Mul<float, false>(columns, columns, firstStride,
            MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    } else {
        SetVectorMask<float, MaskMode::COUNTER>(Tile);
        Mul<float, false>(columns, columns, firstStride,
            MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        for (uint32_t field = 1; field < 4; ++field)
            Mul<float, false>(columns[field * Pitch], columns[field * Pitch], columns[8 * Pitch],
                MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    }
    }
    SetMaskNorm();
    ResetMask();
}

template <uint32_t Tile, uint32_t Pitch, bool FastConfidence = false>
__aicore__ inline void DecodeCandidateDenominators(LocalTensor<float> columns,
    LocalTensor<float> two, LocalTensor<float> one)
{
    DecodeCandidateDenominators<Tile, Pitch, FastConfidence>(columns, two, one, columns[8 * Pitch]);
}

template <uint32_t Rows>
__aicore__ inline void ClearCandidatePriorTail(LocalTensor<float> prior, uint32_t count)
{
    if (count == Rows) return;
    const uint32_t valid = count * 5;
    const uint32_t begin = valid / 8 * 8;
    const uint32_t remaining = Rows * 5 - begin;
    const uint32_t first = remaining < 64 ? remaining : 64;
    const uint64_t active = first == 64 ? ~uint64_t(0) : (uint64_t(1) << first) - 1;
    uint64_t mask[2] = {active & ~((uint64_t(1) << (valid - begin)) - 1), 0};
    Duplicate(prior[begin], 0.0f, mask, uint8_t(1), uint16_t(1), uint8_t(8));
    if (remaining > 64) Duplicate(prior[begin + 64], 0.0f, remaining - 64);
    PipeBarrier<PIPE_V>();
}

template <uint32_t Tile, uint32_t Pitch, uint32_t Rows>
__aicore__ inline bool CanTransformCandidatePriors(LocalTensor<float> prior,
    LocalTensor<float> columns, LocalTensor<float> result,
    LocalTensor<float> temporary)
{
    constexpr uint32_t PriorBytes = ((Rows * 5 / 8 + 31) / 32) * 32;
    constexpr uint32_t StrideBytes = ((Tile / 8 + 31) / 32) * 32;
    constexpr uint32_t Words = Rows * 5 / 32;
    constexpr uint32_t Repeats = (Words + 63) / 64;
    static_assert(Rows * 5 % 64 == 0 && Tile % 64 == 0);
    auto bytes = temporary.template ReinterpretCast<uint8_t>();
    auto priorMask = bytes;
    auto strideMask = bytes[PriorBytes];
    auto flags = bytes[PriorBytes + StrideBytes].template ReinterpretCast<float>();
    Duplicate(flags, 0.0f, Repeats * 64);
    PipeBarrier<PIPE_V>();
    CompareScalar(priorMask, prior, 1.0e18f, CMPMODE::GT, Rows * 5);
    CompareScalar(strideMask, columns[8 * Pitch], 32.0f, CMPMODE::GT, Tile);
    PipeBarrier<PIPE_V>();
    Or(priorMask.template ReinterpretCast<uint16_t>(),
        priorMask.template ReinterpretCast<uint16_t>(),
        strideMask.template ReinterpretCast<uint16_t>(), Tile / 16);
    PipeBarrier<PIPE_V>();
    Cast(flags, priorMask.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, Words);
    Abs(flags, flags, Repeats * 64);
    WholeReduceMax(result[8], flags, 64, Repeats, 1, 1, 8,
        ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(result, result[8], Repeats, 1, 1, 1, 8,
        ReduceOrder::ORDER_ONLY_VALUE);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    return result.GetValue(0) == 0.0f;
}

template <uint32_t Pitch>
__aicore__ inline void TransformCandidatePriors(LocalTensor<float> columns,
    LocalTensor<float> stride)
{
    Adds(columns[6 * Pitch], columns[6 * Pitch], -0.5f, Pitch * 2);
    Mul(columns[6 * Pitch], columns[6 * Pitch], stride, Pitch * 2);
    Mul(columns[9 * Pitch], columns[9 * Pitch], stride, Pitch * 2);
}

namespace HistoricalNative {
template <uint32_t Tile, uint32_t Pitch, bool FastConfidence = false, bool ReplicatedStride = false>
__aicore__ inline void DecodeCandidateDenominators(LocalTensor<float> columns,
    LocalTensor<float> two, LocalTensor<float> one, LocalTensor<float> firstStride)
{
    static_assert(Tile >= 1024 && Tile % 64 == 0 && (Pitch == Tile + 8 || Pitch == Tile + 16));
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(Tile);
    Mul<float, false>(columns[4 * Pitch], columns[4 * Pitch], columns[5 * Pitch],
        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    SetVectorMask<float, MaskMode::COUNTER>(Pitch * 4);
    Div<float, false>(columns, two, columns, MASK_PLACEHOLDER, 1, {1, 0, 1, 8, 0, 8});
    PipeBarrier<PIPE_V>();
    SetVectorMask<float, MaskMode::COUNTER>(Tile);
    if constexpr (FastConfidence) {
        Reciprocal<float, false>(columns[4 * Pitch], columns[4 * Pitch],
            MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    } else {
        Div<float, false>(columns[4 * Pitch], one, columns[4 * Pitch],
            MASK_PLACEHOLDER, 1, {1, 0, 1, 8, 0, 8});
    }
    SetVectorMask<float, MaskMode::COUNTER>(Pitch * 2);
    Adds<float, false>(columns, columns, -0.5f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], columns[2 * Pitch],
        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Add<float, false>(columns, columns, columns[6 * Pitch],
        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], columns[9 * Pitch],
        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    if constexpr (ReplicatedStride) {
        SetVectorMask<float, MaskMode::COUNTER>(Pitch * 4);
        Mul<float, false>(columns, columns, firstStride,
            MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    } else {
        SetVectorMask<float, MaskMode::COUNTER>(Tile);
        Mul<float, false>(columns, columns, firstStride,
            MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        for (uint32_t field = 1; field < 4; ++field)
            Mul<float, false>(columns[field * Pitch], columns[field * Pitch], columns[8 * Pitch],
                MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    }
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <uint32_t Tile, uint32_t Pitch, bool FastConfidence = false>
__aicore__ inline void DecodeCandidateDenominators(LocalTensor<float> columns,
    LocalTensor<float> two, LocalTensor<float> one)
{
    DecodeCandidateDenominators<Tile, Pitch, FastConfidence>(columns, two, one, columns[8 * Pitch]);
}
}
