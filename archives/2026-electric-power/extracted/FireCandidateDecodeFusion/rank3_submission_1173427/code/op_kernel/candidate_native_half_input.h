#pragma once

template <uint32_t Tile, uint32_t Pitch, uint32_t SourceAddr,
    uint32_t ColumnAddr, uint32_t Phase>
__aicore__ inline void NativeHalfCandidatePhase()
{
    static_assert(Tile % 128 == 0 && Pitch % 16 == 0);
    uint64_t sources[16], targets[16];
    #pragma unroll
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const uint32_t field = (Phase * 16 + lane) % 6;
        const uint32_t stripe = (Phase * 16 + lane) / 6;
        sources[lane] = SourceAddr + lane * 96 + Phase * 32;
        targets[lane] = ColumnAddr + (field * Pitch + stripe * (Tile / 8)) * 2;
    }
    TransDataTo5HD<half>(targets, sources,
        TransDataTo5HDParams{false, false, uint8_t(Tile / 128), 1, 48});
}

template <uint32_t Tile, uint32_t Pitch, uint32_t SourceAddr, uint32_t ColumnAddr>
__aicore__ inline void GatherNativeHalfCandidates()
{
    NativeHalfCandidatePhase<Tile, Pitch, SourceAddr, ColumnAddr, 0>();
    NativeHalfCandidatePhase<Tile, Pitch, SourceAddr, ColumnAddr, 1>();
    NativeHalfCandidatePhase<Tile, Pitch, SourceAddr, ColumnAddr, 2>();
}

template <uint32_t Elements, bool DenominatorsOnly>
__aicore__ inline void ActivateNativeHalfColumns(LocalTensor<float> values,
    LocalTensor<float> numerator)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(Elements);
    Muls<float, false>(values, values, -1.0f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    Exp<float, false>(values, values, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    Adds<float, false>(values, values, 1.0f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    if constexpr (!DenominatorsOnly) {
        Div<float, false>(values, numerator, values, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 0, 8});
    }
    SetMaskNorm();
    ResetMask();
}

namespace HistoricalNative {
template <uint32_t Tile, uint32_t Pitch, uint32_t SourceAddr,
    uint32_t ColumnAddr, uint32_t Phase>
__aicore__ inline void NativeHalfCandidatePhase()
{
    static_assert(Tile % 128 == 0 && Pitch % 16 == 0);
    uint64_t sources[16], targets[16];
    #pragma unroll
    for (uint32_t lane = 0; lane < 16; ++lane) {
        const uint32_t field = (Phase * 16 + lane) % 6;
        const uint32_t stripe = (Phase * 16 + lane) / 6;
        sources[lane] = SourceAddr + lane * 96 + Phase * 32;
        targets[lane] = ColumnAddr + (field * Pitch + stripe * (Tile / 8)) * 2;
    }
    TransDataTo5HD<half>(targets, sources,
        TransDataTo5HDParams{false, false, uint8_t(Tile / 128), 1, 48});
}

template <uint32_t Tile, uint32_t Pitch, uint32_t SourceAddr, uint32_t ColumnAddr>
__aicore__ inline void GatherNativeHalfCandidates()
{
    NativeHalfCandidatePhase<Tile, Pitch, SourceAddr, ColumnAddr, 0>();
    NativeHalfCandidatePhase<Tile, Pitch, SourceAddr, ColumnAddr, 1>();
    NativeHalfCandidatePhase<Tile, Pitch, SourceAddr, ColumnAddr, 2>();
}

template <uint32_t Elements, bool DenominatorsOnly>
__aicore__ inline void ActivateNativeHalfColumns(LocalTensor<float> values,
    LocalTensor<float> numerator)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(Elements);
    Muls<float, false>(values, values, -1.0f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    Exp<float, false>(values, values, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    Adds<float, false>(values, values, 1.0f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    if constexpr (!DenominatorsOnly) {
        Div<float, false>(values, numerator, values, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 0, 8});
        PipeBarrier<PIPE_V>();
    }
    SetMaskNorm();
    ResetMask();
}
}
