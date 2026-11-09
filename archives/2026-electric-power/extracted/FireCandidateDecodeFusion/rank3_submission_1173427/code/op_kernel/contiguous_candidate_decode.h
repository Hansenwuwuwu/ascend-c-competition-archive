#pragma once
#include "candidate_native_five_pack.h"

template <uint32_t Tile, uint32_t Pitch, uint32_t PriorAddr,
    uint32_t ColumnAddr, uint32_t Phase>
__aicore__ inline void TransposeContiguousCandidatePriors()
{
    constexpr uint32_t Rows = Tile / 2;
    constexpr uint32_t Quarter = Rows / 8;
    uint64_t sources[16], targets[16];
    #pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t field = (Phase * 8 + j) % 5;
        const uint32_t stripe = (Phase * 8 + j) / 5;
        sources[j] = PriorAddr + j * 160 + Phase * 32;
        sources[j + 8] = sources[j] + Rows * 20;
        targets[j * 2] = ColumnAddr + ((field + 6) * Pitch + stripe * Quarter) * 4;
        targets[j * 2 + 1] = targets[j * 2] + Rows * 4;
    }
    TransDataTo5HD<float>(targets, sources,
        TransDataTo5HDParams{false, false, uint8_t(Rows / 64),
            uint16_t(Rows == 64 ? 0 : 1), uint16_t(Rows == 64 ? 0 : 40)});
}

template <typename T, uint32_t Tile = 512>
class ContiguousCandidateDecode {
    static constexpr uint32_t Pitch = Tile + 8;
    static constexpr bool Reuse = sizeof(T) == 4;
    static constexpr uint32_t FloatAddr = Tile * 6 * sizeof(T);
    static constexpr uint32_t PriorAddr = FloatAddr + (Reuse ? 0 : Tile * 24);
    static constexpr uint32_t ColumnAddr = PriorAddr + Tile * 20;
    static constexpr uint32_t OneAddr = ColumnAddr + Pitch * 44;
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t addr, uint32_t count = Tile * 6,
        TPosition pos = TPosition::VECCALC)
    {
        return LocalTensor<U>(pos, addr, count);
    }
public:
    __aicore__ inline void Run(GM_ADDR rawAddress, GM_ADDR priorAddress, GM_ADDR outAddress,
        uint32_t n, uint32_t b, uint32_t start, uint32_t count)
    {
        GlobalTensor<T> rawGlobal, outGlobal;
        GlobalTensor<float> priorGlobal;
        rawGlobal.SetGlobalBuffer((__gm__ T *)rawAddress);
        priorGlobal.SetGlobalBuffer((__gm__ float *)priorAddress);
        outGlobal.SetGlobalBuffer((__gm__ T *)outAddress);
        auto raw = At<T>(0, Tile * 6, TPosition::VECIN);
        auto prior = At<float>(PriorAddr, Tile * 5, TPosition::VECIN);
        DataCopyPad(raw, rawGlobal[(b * n + start) * 6],
            DataCopyExtParams{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        DataCopyPad(prior, priorGlobal[start * 5],
            DataCopyExtParams{1, count * 20, 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0);
        auto two = At<float>(OneAddr, 64);
        Duplicate(two, 2.0f, 64);
        WaitFlag<HardEvent::MTE2_V>(0);
        LocalTensor<float> sig;
        if constexpr (Reuse) sig = raw;
        else {
            sig = At<float>(FloatAddr);
            Cast(sig, raw, RoundMode::CAST_NONE, Tile * 6);
            PipeBarrier<PIPE_V>();
        }
        Muls(sig, sig, -1.0f, Tile * 6);
        PipeBarrier<PIPE_V>();
        Exp(sig, sig, Tile * 6);
        PipeBarrier<PIPE_V>();
        Adds(sig, sig, 1.0f, Tile * 6);
        PipeBarrier<PIPE_V>();
        Div(sig, two, sig, uint64_t(64), uint8_t(Tile * 6 / 64), {1, 1, 1, 8, 0, 8});
        PipeBarrier<PIPE_V>();
        GatherNativeEightCandidates<Tile, Pitch, Reuse ? 0 : FloatAddr, ColumnAddr>();
        TransposeContiguousCandidatePriors<Tile, Pitch, PriorAddr, ColumnAddr, 0>();
        TransposeContiguousCandidatePriors<Tile, Pitch, PriorAddr, ColumnAddr, 1>();
        TransposeContiguousCandidatePriors<Tile, Pitch, PriorAddr, ColumnAddr, 2>();
        TransposeContiguousCandidatePriors<Tile, Pitch, PriorAddr, ColumnAddr, 3>();
        TransposeContiguousCandidatePriors<Tile, Pitch, PriorAddr, ColumnAddr, 4>();
        PipeBarrier<PIPE_V>();
        auto c = At<float>(ColumnAddr, Pitch * 11);
        Adds(c, c, -0.5f, Pitch * 2);
        Mul(c[2 * Pitch], c[2 * Pitch], c[2 * Pitch], Pitch * 2);
        Mul(c[4 * Pitch], c[4 * Pitch], c[5 * Pitch], Tile);
        PipeBarrier<PIPE_V>();
        Add(c, c, c[6 * Pitch], Pitch * 2);
        Mul(c[2 * Pitch], c[2 * Pitch], c[9 * Pitch], Pitch * 2);
        Muls(c[4 * Pitch], c[4 * Pitch], 0.25f, Tile);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < 4; ++j)
            Mul(c[j * Pitch], c[j * Pitch], c[8 * Pitch], Tile);
        PipeBarrier<PIPE_V>();
        PackNativeFiveCandidates<Tile, Pitch, ColumnAddr, Reuse ? PriorAddr : FloatAddr>();
        auto output = At<T>(PriorAddr, Tile * 5, TPosition::VECOUT);
        if constexpr (!Reuse) {
            PipeBarrier<PIPE_V>();
            Cast(output, At<float>(FloatAddr), RoundMode::CAST_RINT, Tile * 5);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(outGlobal[(b * n + start) * 5], output,
            DataCopyExtParams{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }
};
