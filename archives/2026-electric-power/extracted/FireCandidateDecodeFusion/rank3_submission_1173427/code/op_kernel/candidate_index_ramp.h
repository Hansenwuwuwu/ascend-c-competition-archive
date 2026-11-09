#pragma once

__aicore__ inline void InitCandidateIndexRamp(LocalTensor<int32_t> cache, TPipe *pipe)
{
    for (uint32_t i = 0; i < 8; ++i) cache.SetValue(i, int32_t(i));
    const auto ready = pipe->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(ready);
    WaitFlag<HardEvent::S_V>(ready);
    auto lanes = cache[8];
    Brcb(lanes, cache, uint8_t(1), BrcbRepeatParams{1, 8});
    PipeBarrier<PIPE_V>();
    Muls(lanes, lanes, int32_t(8), 64);
    PipeBarrier<PIPE_V>();
    Add(lanes, lanes, cache, uint64_t(64), uint8_t(1), {1, 1, 0, 8, 8, 0});
    PipeBarrier<PIPE_V>();
    auto steps = cache[72];
    DataCopy(steps, lanes, 64);
    Adds(steps[64], lanes, int32_t(64), uint64_t(64), uint8_t(1), {1, 1, 8, 8});
    Adds(steps[128], lanes, int32_t(128), uint64_t(64), uint8_t(1), {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    Muls(steps, steps, int32_t(64), 192);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FillCandidateIndexRamp(LocalTensor<int32_t> target, uint32_t elements,
                                            LocalTensor<int32_t> cache)
{
    const uint32_t repeats = elements / 64;
    auto spread = cache[264];
    Brcb(spread, cache[72], uint8_t((repeats + 7) / 8), BrcbRepeatParams{1, 8});
    PipeBarrier<PIPE_V>();
    Brcb(target, spread, uint8_t(repeats), BrcbRepeatParams{1, 8});
    PipeBarrier<PIPE_V>();
    Add(target, target, cache[8], uint64_t(64), uint8_t(repeats), {1, 1, 1, 8, 8, 0});
    PipeBarrier<PIPE_V>();
}

template <uint32_t Elements, bool StaticEvents = false>
__aicore__ inline void BuildDoublingCandidateIndex(LocalTensor<int32_t> sequence,
    LocalTensor<int32_t> seed, TPipe *pipe)
{
    for (uint32_t i = 0; i < 8; ++i) seed.SetValue(i, int32_t(i));
    int32_t ready = 0;
    if constexpr (!StaticEvents) ready = pipe->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(ready);
    WaitFlag<HardEvent::S_V>(ready);
    Brcb(sequence, seed, uint8_t(1), BrcbRepeatParams{1, 8});
    PipeBarrier<PIPE_V>();
    Muls(sequence, sequence, int32_t(8), 64);
    PipeBarrier<PIPE_V>();
    Add(sequence, sequence, seed, uint64_t(64), uint8_t(1), {1, 1, 0, 8, 8, 0});
    PipeBarrier<PIPE_V>();
    #pragma unroll
    for (uint32_t filled = 64; filled < Elements; filled *= 2) {
        const uint32_t count = filled * 2 <= Elements ? filled : Elements - filled;
        Adds(sequence[filled], sequence, int32_t(filled), count);
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t Elements>
__aicore__ inline void ExpandCandidatePatternIndex(LocalTensor<int32_t> index)
{
    #pragma unroll
    for (uint32_t filled = 320; filled < Elements; filled *= 2) {
        const uint32_t count = filled * 2 <= Elements ? filled : Elements - filled;
        Adds(index[filled], index, int32_t((filled / 5) * 4), count);
        PipeBarrier<PIPE_V>();
    }
}
