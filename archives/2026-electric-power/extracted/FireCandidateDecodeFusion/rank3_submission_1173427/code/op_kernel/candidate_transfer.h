#pragma once

template <typename T, bool Aligned>
__aicore__ inline void CopyCandidateInput(LocalTensor<T> local, GlobalTensor<T> global,
    uint32_t index, uint16_t groups, uint32_t count, uint32_t localPitch, uint32_t globalPitch)
{
    const uint32_t bytes = count * uint32_t(sizeof(T));
    if constexpr (Aligned) {
        DataCopy(local, global[index], DataCopyParams{groups, uint16_t(bytes / 32),
            uint16_t(groups == 1 ? 0 : (globalPitch - count) * sizeof(T) / 32),
            uint16_t(groups == 1 ? 0 : (localPitch - count) * sizeof(T) / 32)});
    } else {
        DataCopyPad(local, global[index],
            DataCopyExtParams{groups, bytes,
                groups == 1 ? 0 : (globalPitch - count) * uint32_t(sizeof(T)),
                groups == 1 ? 0 : (localPitch * uint32_t(sizeof(T)) - ((bytes + 31) / 32) * 32) / 32, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
    }
}

template <typename T, bool Aligned>
__aicore__ inline void CopyCandidateOutput(GlobalTensor<T> global, LocalTensor<T> local,
    uint32_t index, uint16_t groups, uint32_t count, uint32_t localPitch, uint32_t globalPitch)
{
    const uint32_t bytes = count * uint32_t(sizeof(T));
    if constexpr (Aligned) {
        DataCopy(global[index], local, DataCopyParams{groups, uint16_t(bytes / 32),
            uint16_t((localPitch - count) * sizeof(T) / 32),
            uint16_t((globalPitch - count) * sizeof(T) / 32)});
    } else {
        DataCopyPad(global[index], local,
            DataCopyExtParams{groups, bytes,
                (localPitch * uint32_t(sizeof(T)) - ((bytes + 31) / 32) * 32) / 32,
                (globalPitch - count) * uint32_t(sizeof(T)), 0});
    }
}

template <typename T, uint32_t Pitch, uint32_t Tile>
__aicore__ inline void DecodeGroupedCandidateTail(
    LocalTensor<T> raw, LocalTensor<float> prior, LocalTensor<float> sig,
    LocalTensor<float> c, LocalTensor<float> ones, LocalTensor<uint32_t> ri,
    LocalTensor<uint32_t> pi, LocalTensor<uint32_t> packIndex,
    LocalTensor<float> packed, LocalTensor<T> dst, uint32_t batch,
    uint32_t count, uint32_t rawStride, uint32_t dstStride, TPipe *pipe)
{
    const uint32_t rows = batch * count;
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < count; ++i) {
            ri.SetValue(b * count + i, (b * rawStride + i * 6) * 4);
            pi.SetValue(b * count + i, i * 20);
        }
        for (uint32_t i = 0; i < dstStride; ++i) {
            const uint32_t index = i < count * 5 ?
                (i % 5) * Pitch * 4 + (b * count + i / 5) * 4 : 0;
            packIndex.SetValue(b * dstStride + i, index);
        }
    }
    const auto ready = pipe->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(ready);
    WaitFlag<HardEvent::S_V>(ready);
    const uint32_t elements = batch * rawStride;
    if constexpr (sizeof(T) == 2) Cast(sig, raw, RoundMode::CAST_NONE, elements);
    PipeBarrier<PIPE_V>();
    Muls(sig, sig, -1.0f, elements);
    PipeBarrier<PIPE_V>();
    Exp(sig, sig, elements);
    PipeBarrier<PIPE_V>();
    Adds(sig, sig, 1.0f, elements);
    PipeBarrier<PIPE_V>();
    Div(sig, ones, sig, uint64_t(64), uint8_t((elements + 63) / 64),
        {1, 1, 1, 8, 0, 8});
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < 6; ++j)
        Gather(c[j * Pitch], sig, ri, j * 4, uint64_t(64), uint8_t((rows + 63) / 64), 8);
    for (uint32_t j = 0; j < 5; ++j)
        Gather(c[(j + 6) * Pitch], prior, pi, j * 4,
            uint64_t(64), uint8_t((rows + 63) / 64), 8);
    PipeBarrier<PIPE_V>();
    DecodeCandidateGeometry<Pitch, Tile>(c, rows);
    const uint32_t outputElements = batch * dstStride;
    Gather(packed, c, packIndex, 0, uint64_t(64),
        uint8_t((outputElements + 63) / 64), 8);
    PipeBarrier<PIPE_V>();
    if constexpr (sizeof(T) == 2) Cast(dst, packed, RoundMode::CAST_RINT, outputElements);
}
