#pragma once
#include "historical_paired_candidate_decode.h"

template <typename T, bool Adaptive>
__aicore__ inline void RunFourBatchMediumCandidate(GM_ADDR raw, GM_ADDR prior, GM_ADDR out,
    const FireCandidateDecodeFusionTilingData &data)
{
    InitSocState();
    if constexpr (!Adaptive) {
        PairedBatchCandidateDecode<T, 512, false, false, true, false, true, true, 4, true, true, true> op;
        op.Init(raw, prior, out, data, nullptr);
        op.Process();
    } else {
        const uint32_t groups = (data.candidates + 15) / 16;
        const uint32_t begin = (groups * GetBlockIdx() / data.cores) * 16;
        uint32_t end = (groups * (GetBlockIdx() + 1) / data.cores) * 16;
        if (end > data.candidates) end = data.candidates;
        if (end - begin <= 128) {
            PairedBatchCandidateDecode<T, 512, false, false, true, true, true, true, 4, true, true, true> op;
            op.Init(raw, prior, out, data, nullptr);
            op.Process();
        } else {
            PairedBatchCandidateDecode<T, 640, true, false, true, true, true, true, 4, true, true, true> op;
            op.Init(raw, prior, out, data, nullptr);
            op.Process();
        }
    }
}


template <typename T, uint32_t Tile, bool Preload, bool NativePrior, bool AlignedTransfers = false>
__aicore__ inline void RunLargeCandidatePath(GM_ADDR raw, GM_ADDR prior, GM_ADDR out,
    const FireCandidateDecodeFusionTilingData &data)
{
    InitSocState();
    if constexpr (NativePrior && Tile >= 1408 && sizeof(T) == 2) {
    HistoricalNative::PairedBatchCandidateDecode<T, Tile, false, true,
        sizeof(T) == 4, false, true, true, 2, true, true, Preload, NativePrior, true, sizeof(T) == 2 && NativePrior, AlignedTransfers, true, NativePrior> op;
    op.Init(raw, prior, out, data, nullptr);
    op.Process();
    } else {
    PairedBatchCandidateDecode<T, Tile, false, true,
        sizeof(T) == 4, false, true, true, 2, true, true, Preload, NativePrior, true, sizeof(T) == 2 && NativePrior, AlignedTransfers, true, NativePrior> op;
    op.Init(raw, prior, out, data, nullptr);
    op.Process();
    }

}

template <typename T, uint32_t Tile>
__aicore__ inline void RunNativeLargeCandidate(GM_ADDR raw, GM_ADDR prior, GM_ADDR out,
    const FireCandidateDecodeFusionTilingData &data)
{
    if (data.candidates % 16 == 0) {
        RunLargeCandidatePath<T, Tile, true, true, true>(raw, prior, out, data);
        return;
    }
    RunLargeCandidatePath<T, Tile, true, true>(raw, prior, out, data);
}



template <typename T, bool FuseConfidence = false>
__aicore__ inline void RunCoarseMediumCandidate(GM_ADDR raw, GM_ADDR prior, GM_ADDR out,
    const FireCandidateDecodeFusionTilingData &data)
{
    InitSocState();
    const uint32_t groups = (data.candidates + 15) / 16;
    const uint32_t left = groups * GetBlockIdx();
    const uint32_t right = groups * (GetBlockIdx() + 1);
    const uint32_t begin = (data.segments < 32 ? left >> data.segments : left / data.cores) * 16;
    uint32_t end = (data.segments < 32 ? right >> data.segments : right / data.cores) * 16;
    if (end > data.candidates) end = data.candidates;
    if (end - begin <= 256) {
        PairedBatchCandidateDecode<T, 1024, false, false, true, true, true, true, 4, true, true, true, true, FuseConfidence, sizeof(T) == 2> op;
        op.Init(raw, prior, out, data, nullptr, begin, end);
        op.Process();
    } else {
        PairedBatchCandidateDecode<T, 1280, false, false, true, true, true, true, 4, true, true, true, true, FuseConfidence, sizeof(T) == 2> op;
        op.Init(raw, prior, out, data, nullptr, begin, end);
        op.Process();
    }
}
