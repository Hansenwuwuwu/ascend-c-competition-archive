#pragma once
#include "candidate_native_five_pack.h"
#include "candidate_native_prior.h"
#include "candidate_denominator_geometry.h"
#include "candidate_native_half_input.h"
#include "candidate_transfer.h"

template <typename T, uint32_t Tile, bool AlignedTransfers = false>
class SingleGroupMediumCandidateDecode {
    static constexpr uint32_t BatchRows = Tile / 4;
    static constexpr uint32_t Pitch = Tile + (sizeof(T) == 2 ? 16 : 8);
    static constexpr bool Reuse = sizeof(T) == 4;
    static constexpr uint32_t FloatAddr = Tile * 6 * sizeof(T);
    static constexpr uint32_t SharedAddr = FloatAddr + (Reuse ? 0 : Tile * 24);
    static constexpr uint32_t ColumnAddr = SharedAddr + Tile * 5 * sizeof(T);
    static constexpr uint32_t OneAddr = ColumnAddr + Pitch * 44;

    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t address, uint32_t count,
        TPosition position = TPosition::VECCALC)
    {
        return LocalTensor<U>(position, address, count);
    }

public:
    __aicore__ inline void Run(GM_ADDR rawAddress, GM_ADDR priorAddress, GM_ADDR outAddress,
        uint32_t candidates, uint32_t begin, uint32_t count)
    {
        GlobalTensor<T> rawGlobal, outGlobal;
        GlobalTensor<float> priorGlobal;
        rawGlobal.SetGlobalBuffer((__gm__ T *)rawAddress);
        priorGlobal.SetGlobalBuffer((__gm__ float *)priorAddress);
        outGlobal.SetGlobalBuffer((__gm__ T *)outAddress);
        auto prior = At<float>(SharedAddr, BatchRows * 5, TPosition::VECIN);
        auto raw = At<T>(0, Tile * 6, TPosition::VECIN);
        CopyCandidateInput<float, AlignedTransfers>(prior, priorGlobal, begin * 5,
            1, count * 5, count * 5, count * 5);
        SetFlag<HardEvent::MTE2_V>(0);
        CopyCandidateInput<T, AlignedTransfers>(raw, rawGlobal, begin * 6,
            4, count * 6, BatchRows * 6, candidates * 6);
        SetFlag<HardEvent::MTE2_V>(1);
        Duplicate(At<float>(OneAddr, 64), 2.0f, 64);
        Duplicate(At<float>(OneAddr + 256, 64), 1.0f, 64);
        WaitFlag<HardEvent::MTE2_V>(0);
        GatherNativePriorCandidates<Tile, Pitch, SharedAddr, ColumnAddr, 4, sizeof(T) == 2>();
        auto columns = At<float>(ColumnAddr, Pitch * 11);
        if constexpr (sizeof(T) == 2) {
            #pragma unroll
            for (uint32_t field = 0; field < 5; ++field)
                DataCopy(columns[(field + 6) * Pitch + Tile / 16], columns[(field + 6) * Pitch],
                    DataCopyParams{8, uint16_t(Tile / 128), uint16_t(Tile / 128), uint16_t(Tile / 128)});
        } else {
            DataCopy(columns[6 * Pitch + Tile / 2], columns[6 * Pitch],
                DataCopyParams{5, uint16_t(Tile / 16),
                    uint16_t((Pitch - Tile / 2) / 8), uint16_t((Pitch - Tile / 2) / 8)});
        }
        WaitFlag<HardEvent::MTE2_V>(1);
        if constexpr (sizeof(T) == 2) {
            GatherNativeHalfCandidates<Tile, Pitch, 0, FloatAddr>();
            Cast(columns, At<half>(FloatAddr, Pitch * 6), RoundMode::CAST_NONE, Pitch * 6);
            ActivateNativeHalfColumns<Pitch * 6, true>(columns, At<float>(OneAddr, 64));
        } else {
        auto sigmoid = raw;
        Muls(sigmoid, sigmoid, -1.0f, uint64_t(64), uint8_t(Tile * 6 / 64), {1, 1, 8, 8});
        Exp(sigmoid, sigmoid, uint64_t(64), uint8_t(Tile * 6 / 64), {1, 1, 8, 8});
        Adds(sigmoid, sigmoid, 1.0f, uint64_t(64), uint8_t(Tile * 6 / 64), {1, 1, 8, 8});
        GatherNativeEightCandidates<Tile, Pitch, 0, ColumnAddr>();
        }
        DecodeCandidateDenominators<Tile, Pitch>(columns,
            At<float>(OneAddr, 64), At<float>(OneAddr + 256, 64));
        PackNativeFiveCandidates<Tile, Pitch, ColumnAddr, Reuse ? SharedAddr : FloatAddr, sizeof(T) == 2>();
        auto output = At<T>(SharedAddr, Tile * 5, TPosition::VECOUT);
        if constexpr (!Reuse) {
            Cast(output, At<float>(FloatAddr, Tile * 5), RoundMode::CAST_RINT, Tile * 5);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyCandidateOutput<T, AlignedTransfers>(outGlobal, output, begin * 5,
            4, count * 5, BatchRows * 5, candidates * 5);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
};
