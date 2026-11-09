#pragma once

template <typename T>
__aicore__ inline void ProcessFixedSmallBatch(GM_ADDR raw, GM_ADDR prior, GM_ADDR out,
    const FireCandidateDecodeFusionTilingData &data)
{
    const uint32_t groups = (data.candidates + 7) / 8;
    const uint32_t task = uint32_t(GetBlockIdx());
    const uint32_t batch = task / groups;
    const uint32_t begin = (task - batch * groups) * 8;
    const uint32_t count = data.candidates - begin < 8 ? data.candidates - begin : 8;
    ConfigNativeSmallCandidateDecode<T, 8> op;
    op.RunRange(raw + uint64_t(batch * data.candidates) * 6 * sizeof(T), prior,
        out + uint64_t(batch * data.candidates) * 5 * sizeof(T), begin, count);
}
