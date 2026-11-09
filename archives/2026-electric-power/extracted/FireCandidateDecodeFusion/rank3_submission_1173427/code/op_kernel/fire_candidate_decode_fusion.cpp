#include "kernel_operator.h"
#include "fire_candidate_decode_fusion_tiling.h"
#include "tiling_key_fire_candidate_decode_fusion.h"

using namespace AscendC;
#include "candidate_index_ramp.h"
#include "candidate_denominator_geometry.h"
#include "candidate_transfer.h"

#include "small_candidate_decode.h"
#include "native_small_candidate_decode.h"
#include "medium_candidate_decode.h"
#include "generic_candidate_decode.h"

#include "paired_batch_candidate_decode.h"
#include "candidate_fixed_small_batch.h"
#include "grouped_candidate_entry.h"
#include "single_group_medium_candidate_decode.h"
#include "historical_medium_candidate_decode.h"
#include "contiguous_candidate_decode.h"
#include "paired_small_batch_candidate_decode.h"

template <typename DT_RAW_PRED, uint32_t PATH>
__global__ __aicore__ void fire_candidate_decode_fusion(GM_ADDR raw_pred,
    GM_ADDR priors, GM_ADDR decoded, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionTilingData);
    if constexpr (PATH == 51 || PATH == 54) {
        constexpr uint32_t factor = PATH == 51 ? 2 :
            (sizeof(DT_RAW_PRED) == 4 ? CandidateSmallLaunchFloat : CandidateSmallLaunchHalf);
        const uint32_t active = uint32_t(GetBlockNum()) / factor;
        if (uint32_t(GetBlockIdx()) >= active) return;
        if constexpr (PATH == 54 && sizeof(DT_RAW_PRED) == 4) {
            InitSocState();
            WideSingleSmallCandidateDecode<DT_RAW_PRED> op;
            op.Run(raw_pred, priors, decoded, uint32_t(GetBlockIdx()) * 16);
        } else {
        ConfigNativeSmallCandidateDecode<DT_RAW_PRED, 8> op;
        op.Run(raw_pred, priors, decoded);
        }
    } else if constexpr (PATH == 15) {
        InitSocState();
        NativeSmallCandidateDecode<DT_RAW_PRED> op;
        op.Run(raw_pred, priors, decoded);
    } else if constexpr (PATH == 65) {
        InitSocState();
        constexpr uint32_t Rows = CandidateMediumRows;
        const uint64_t bound = uint64_t(GetBlockNum()) * Rows;
        const uint32_t candidates = uint32_t(uint64_t(1) << (63 - ScalarCountLeadingZero(bound)));
        const uint32_t begin = uint32_t(GetBlockIdx()) * Rows;
        const uint32_t count = candidates - begin < Rows ? candidates - begin : Rows;
        SingleGroupMediumCandidateDecode<DT_RAW_PRED, Rows * 4, true> op;
        op.Run(raw_pred, priors, decoded, candidates, begin, count);
    } else if constexpr (PATH == 34 || PATH == 35) {
        InitSocState();
        const uint32_t candidates = uint32_t(GetBlockNum()) * 256 + (PATH == 35 ? 1 : 0);
        const uint32_t begin = uint32_t(GetBlockIdx()) * 256;
        if constexpr (PATH == 35) {
            if (uint32_t(GetBlockIdx()) + 1 == uint32_t(GetBlockNum())) {
                HistoricalNative::SingleGroupMediumCandidateDecode<DT_RAW_PRED, 1280> op;
                op.Run(raw_pred, priors, decoded, candidates, begin, 257);
            } else {
                HistoricalNative::SingleGroupMediumCandidateDecode<DT_RAW_PRED, 1024> op;
                op.Run(raw_pred, priors, decoded, candidates, begin, 256);
            }
        } else {
            SingleGroupMediumCandidateDecode<DT_RAW_PRED, 1024, true> op;
            op.Run(raw_pred, priors, decoded, candidates, begin, 256);
        }
    } else if constexpr (PATH >= 36 && PATH <= 43) {
        constexpr bool Tail = PATH >= 40;
        constexpr uint32_t Key = Tail ? PATH - 4 : PATH;
        constexpr uint32_t Tile = Key < 38 ? 1408 : 1664;
        const uint32_t cores = uint32_t(GetBlockNum());
        const uint64_t bound = uint64_t(cores) * (Tile / 2);
        const uint32_t candidates = uint32_t(uint64_t(1) << (63 - ScalarCountLeadingZero(bound))) + (Tail ? 1u : 0u);
        const FireCandidateDecodeFusionTilingData data{Key % 2 == 0 ? 4u : 8u, candidates, cores, 0};
        RunNativeLargeCandidate<DT_RAW_PRED, Tile>(raw_pred, priors, decoded, data);
    } else if constexpr (PATH == 44) {
        if constexpr (sizeof(DT_RAW_PRED) == 4) {
            const uint32_t groups = uint32_t(GetBlockNum()) / 2;
            if (uint32_t(GetBlockIdx()) >= groups) return;
            InitSocState();
            PairedSmallBatchCandidateDecode<DT_RAW_PRED> op;
            op.Run(raw_pred, priors, decoded, (groups - 1) * 8 + 1, uint32_t(GetBlockIdx()) * 8);
        } else {
            InitSocState();
            const uint32_t cores = uint32_t(GetBlockNum());
            const FireCandidateDecodeFusionTilingData data{2, (cores / 2 - 1) * 8 + 1, cores, 0};
            ProcessFixedSmallBatch<DT_RAW_PRED>(raw_pred, priors, decoded, data);
        }
    } else if constexpr (PATH >= 45 && PATH <= 50) {
        InitSocState();
        constexpr uint32_t Batch = PATH - 42;
        const uint32_t cores = uint32_t(GetBlockNum());
        const uint32_t candidates = (cores / Batch - 1) * 8 + 1;
        const FireCandidateDecodeFusionTilingData data{Batch, candidates, cores, 0};
        ProcessFixedSmallBatch<DT_RAW_PRED>(raw_pred, priors, decoded, data);
    } else {
    GET_TILING_DATA_WITH_STRUCT(FireCandidateDecodeFusionTilingData, data, tiling);
    if constexpr (PATH == 52 || PATH == 53) {
        InitSocState();
        constexpr uint32_t Tile = PATH == 52 ? 512 : 128;
        const uint32_t begin = uint32_t(GetBlockIdx()) * Tile;
        const uint32_t count = data.candidates - begin < Tile ? data.candidates - begin : Tile;
        ContiguousCandidateDecode<DT_RAW_PRED, Tile> op;
        op.Run(raw_pred, priors, decoded, data.candidates, 0, begin, count);
    } else if constexpr (PATH == 66) {
        InitSocState();
        constexpr uint32_t Rows = CandidateMediumRows;
        const uint32_t begin = uint32_t(GetBlockIdx()) * Rows;
        const uint32_t count = data.candidates - begin < Rows ? data.candidates - begin : Rows;
        SingleGroupMediumCandidateDecode<DT_RAW_PRED, Rows * 4, true> op;
        op.Run(raw_pred, priors, decoded, data.candidates, begin, count);
    } else if constexpr (PATH == 33) {
        RunCoarseMediumCandidate<DT_RAW_PRED, sizeof(DT_RAW_PRED) == 2>(raw_pred, priors, decoded, data);
    } else if constexpr (PATH == 31 || PATH == 32) {
        RunFourBatchMediumCandidate<DT_RAW_PRED, PATH == 32>(raw_pred, priors, decoded, data);
    } else if constexpr (PATH == 26) {
        InitSocState();
        ProcessFixedSmallBatch<DT_RAW_PRED>(raw_pred, priors, decoded, data);
    } else if constexpr (PATH == 24) {
        SmallCandidateDecode<DT_RAW_PRED, 8, false, true> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else if constexpr (PATH == 21) {
        const uint32_t groups = (data.candidates + 15) / 16;
        const uint32_t begin = (groups * GetBlockIdx() / data.cores) * 16;
        uint32_t end = (groups * (GetBlockIdx() + 1) / data.cores) * 16;
        if (end > data.candidates) end = data.candidates;
        InitSocState();
        if (end - begin <= 128) {
            PairedBatchCandidateDecode<DT_RAW_PRED, 256, false, false, true, true, true, true> op;
            op.Init(raw_pred, priors, decoded, data, nullptr);
            op.Process();
        } else {
            PairedBatchCandidateDecode<DT_RAW_PRED, 320, true, false, true, true, true, true> op;
            op.Init(raw_pred, priors, decoded, data, nullptr);
            op.Process();
        }
    } else if constexpr (PATH == 19) {
        RunNativeLargeCandidate<DT_RAW_PRED, 1664>(raw_pred, priors, decoded, data);
    } else if constexpr (PATH == 18) {
        RunNativeLargeCandidate<DT_RAW_PRED, 1408>(raw_pred, priors, decoded, data);
    } else if constexpr (PATH == 14) {
        SmallCandidateDecode<DT_RAW_PRED, 4, true> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else if constexpr (PATH == 12 || PATH == 13) {
        InitSocState();
        PairedBatchCandidateDecode<DT_RAW_PRED, PATH == 12 ? 256 : 320, PATH == 13, false, true, false, true, true> op;
        op.Init(raw_pred, priors, decoded, data, nullptr);
        op.Process();
    } else if constexpr (PATH == 10) {
        SmallCandidateDecode<DT_RAW_PRED, 4> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else if constexpr (PATH == 8 || PATH == 9) {
        TPipe pipe;
        PairedBatchCandidateDecode<DT_RAW_PRED, PATH == 8 ? 1408 : 1664> op;
        op.Init(raw_pred, priors, decoded, data, &pipe);
        op.Process();
    } else if constexpr (PATH == 0) {
        SmallCandidateDecode<DT_RAW_PRED, 1> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else if constexpr (PATH == 5) {
        SmallCandidateDecode<DT_RAW_PRED, 8> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else if constexpr (PATH == 3) {
        SmallCandidateDecode<DT_RAW_PRED, 2> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else if constexpr (PATH == 6) {
        KernelFireCandidateDecodeFusion<DT_RAW_PRED, 768, false, true> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else if constexpr (PATH == 7) {
        KernelFireCandidateDecodeFusion<DT_RAW_PRED, 1024, false, true> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else if constexpr (PATH == 4) {
        MediumCandidateDecodeFusion<DT_RAW_PRED, 576, true> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else if constexpr (PATH == 2) {
        MediumCandidateDecodeFusion<DT_RAW_PRED, 512> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    } else {
        KernelFireCandidateDecodeFusion<DT_RAW_PRED, 1024> op;
        op.Init(raw_pred, priors, decoded, data);
        op.Process();
    }
    }
}
