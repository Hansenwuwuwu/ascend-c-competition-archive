#pragma once

template <uint32_t Rows>
__aicore__ inline void StoreNativeHalfSmallCandidates(GlobalTensor<half> output,
    LocalTensor<float> columns, uint32_t begin, uint32_t count = Rows)
{
    LocalTensor<half> casted(TPosition::VECCALC, 4352, 256);
    LocalTensor<half> transposed(TPosition::VECOUT, 5120, 256);
    LocalTensor<uint16_t> pattern(TPosition::VECCALC, 4608, 16);
    Cast(casted, columns, RoundMode::CAST_RINT, 80);
    if constexpr (Rows == 8) Duplicate(pattern, uint16_t(0x001f), 16);
    else {
        Duplicate(pattern, uint16_t(0), 16);
        PipeBarrier<PIPE_V>();
        Duplicate(pattern, uint16_t(0x001f), Rows);
    }
    PipeBarrier<PIPE_V>();
    Transpose<half>(transposed, casted);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> packed(TPosition::VECOUT, 4352, 64);
    uint64_t selected;
    GatherMask(packed, transposed, pattern, false, 0, {1, 1, 8, 0}, selected);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyPad(output[begin * 5], packed,
        DataCopyExtParams{1, count * 10, 0, 0, 0});
}
