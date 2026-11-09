#pragma once
#include "candidate_native_half_output.h"

template <uint32_t Source, uint32_t RowBytes, uint32_t Target>
__aicore__ inline void TransposeFourCandidateRows()
{
    uint64_t source[16], target[16];
    #pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
        source[j] = Source + (j % 4) * RowBytes;
        source[j + 8] = source[j];
        target[j * 2] = Target + j * 64;
        target[j * 2 + 1] = target[j * 2] + 32;
    }
    TransDataTo5HD<float>(target, source, TransDataTo5HDParams{false, false, 1, 0, 0});
}

template <typename T>
class NativeSmallCandidateDecode {
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t address, uint32_t count = 128,
        TPosition position = TPosition::VECCALC)
    {
        return LocalTensor<U>(position, address, count);
    }
public:
    __aicore__ inline void Run(GM_ADDR rawAddress, GM_ADDR priorAddress, GM_ADDR outAddress)
    {
        constexpr uint32_t Pitch = 16;
        GlobalTensor<T> rawGlobal, outGlobal;
        GlobalTensor<float> priorGlobal;
        rawGlobal.SetGlobalBuffer((__gm__ T *)rawAddress);
        priorGlobal.SetGlobalBuffer((__gm__ float *)priorAddress);
        outGlobal.SetGlobalBuffer((__gm__ T *)outAddress);
        const uint32_t begin = uint32_t(GetBlockIdx()) * 4;
        auto raw = At<T>(0, 128, TPosition::VECIN);
        auto prior = At<float>(256, 128, TPosition::VECIN);
        DataCopyPad(raw, rawGlobal[begin * 6],
            DataCopyExtParams{4, 6 * uint32_t(sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        DataCopyPad(prior, priorGlobal[begin * 5],
            DataCopyExtParams{4, 20, 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0);
        auto index = At<uint32_t>(4608);
        #pragma unroll
        for (uint32_t i = 0; i < 20; ++i)
            index.SetValue(i, (i / 5 * 8 + i % 5) * 4);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        auto ones = At<float>(4864);
        Duplicate(ones, 1.0f, 64);
        WaitFlag<HardEvent::MTE2_V>(0);
        LocalTensor<float> activation;
        if constexpr (sizeof(T) == 2) {
            activation = At<float>(512);
            Cast(activation, raw, RoundMode::CAST_NONE, 64);
            PipeBarrier<PIPE_V>();
        } else activation = raw;
        constexpr uint64_t activationCount = sizeof(T) == 2 ? 64 : 32;
        Muls(activation, activation, -1.0f, activationCount, uint8_t(1), {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Exp(activation, activation, activationCount, uint8_t(1), {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Adds(activation, activation, 1.0f, activationCount, uint8_t(1), {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Div(activation, ones, activation, activationCount, uint8_t(1), {1, 1, 1, 8, 0, 8});
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == 2) TransposeFourCandidateRows<512, 64, 1024>();
        else TransposeFourCandidateRows<0, 32, 1024>();
        TransposeFourCandidateRows<256, 32, 2048>();
        PipeBarrier<PIPE_V>();
        auto columns = At<float>(1024);
        auto priors = At<float>(2048);
        Muls(columns, columns, 2.0f, Pitch * 4);
        PipeBarrier<PIPE_V>();
        Adds(columns, columns, -0.5f, Pitch * 2);
        Mul(columns[2 * Pitch], columns[2 * Pitch], columns[2 * Pitch], Pitch * 2);
        Mul(columns[4 * Pitch], columns[4 * Pitch], columns[5 * Pitch], Pitch);
        PipeBarrier<PIPE_V>();
        Add(columns, columns, priors, Pitch * 2);
        Mul(columns[2 * Pitch], columns[2 * Pitch], priors[3 * Pitch], Pitch * 2);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < 4; ++j)
            Mul(columns[j * Pitch], columns[j * Pitch], priors[2 * Pitch], Pitch);
        PipeBarrier<PIPE_V>();
        uint64_t sources[16], targets[16];
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            sources[j] = 1024 + j * 64;
            sources[j + 8] = sources[j] + 32;
            targets[j * 2] = 3072 + j * 32;
            targets[j * 2 + 1] = 3072 + (j + 8) * 32;
        }
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams{false, false, 1, 0, 0});
        PipeBarrier<PIPE_V>();
        auto packed = At<float>(4096);
        Gather(packed, At<float>(3072), index, 0, 20);
        LocalTensor<T> result;
        if constexpr (sizeof(T) == 2) {
            PipeBarrier<PIPE_V>();
            result = At<T>(4352, 64, TPosition::VECOUT);
            Cast(result, packed, RoundMode::CAST_RINT, 20);
        } else result = packed;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(outGlobal[begin * 5], result,
            DataCopyExtParams{1, 20 * uint32_t(sizeof(T)), 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }
};

template <uint32_t Rows, uint32_t Source, uint32_t RowBytes>
__aicore__ inline void ConfigTransposeCandidateInputs()
{
    uint64_t source[16], target[16];
    #pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
        source[j] = Source + (j % Rows) * RowBytes;
        source[j + 8] = 256 + (j % Rows) * 32;
        target[j * 2] = 1024 + j * 64;
        target[j * 2 + 1] = target[j * 2] + 32;
    }
    TransDataTo5HD<float>(target, source, TransDataTo5HDParams{false, false, 1, 0, 0});
}

template <typename T, uint32_t Rows = 4, bool ScaledSigmoid = true>
class ConfigNativeSmallCandidateDecode {
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t address, uint32_t count = 128,
        TPosition position = TPosition::VECCALC)
    {
        return LocalTensor<U>(position, address, count);
    }
public:
    __aicore__ inline void Run(GM_ADDR rawAddress, GM_ADDR priorAddress, GM_ADDR outAddress)
    {
        RunRange(rawAddress, priorAddress, outAddress, uint32_t(GetBlockIdx()) * Rows, Rows);
    }

    __aicore__ inline void RunRange(GM_ADDR rawAddress, GM_ADDR priorAddress, GM_ADDR outAddress,
        uint32_t begin, uint32_t count)
    {
        constexpr uint32_t Pitch = 16;
        GlobalTensor<T> rawGlobal, outGlobal;
        GlobalTensor<float> priorGlobal;
        rawGlobal.SetGlobalBuffer((__gm__ T *)rawAddress);
        priorGlobal.SetGlobalBuffer((__gm__ float *)priorAddress);
        outGlobal.SetGlobalBuffer((__gm__ T *)outAddress);
        auto raw = At<T>(0, 128, TPosition::VECIN);
        auto prior = At<float>(256, 128, TPosition::VECIN);
        DataCopyPad(raw, rawGlobal[begin * 6],
            DataCopyExtParams{uint16_t(count), 6 * uint32_t(sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0);
        DataCopyPad(prior, priorGlobal[begin * 5],
            DataCopyExtParams{uint16_t(count), 20, 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(1);
        auto index = At<uint32_t>(4608);
        if constexpr (sizeof(T) == 4 && Rows == 8) {
        Duplicate(index, uint32_t(0x1f1f1f1f), 2);
        }
        auto ones = At<float>(4864);
        Duplicate(ones, ScaledSigmoid ? 2.0f : 1.0f, 64);
        WaitFlag<HardEvent::MTE2_V>(0);
        LocalTensor<float> activation;
        if constexpr (sizeof(T) == 2) {
            activation = At<float>(512);
            Cast(activation, raw, RoundMode::CAST_NONE, Rows * 16);
        } else activation = raw;
        constexpr uint64_t activationCount = Rows == 8 ? 64 : (sizeof(T) == 2 ? 64 : 32);
        constexpr uint8_t activationRepeat = Rows == 8 && sizeof(T) == 2 ? 2 : 1;
        Muls(activation, activation, -1.0f, activationCount, activationRepeat, {1, 1, 8, 8});
        Exp(activation, activation, activationCount, activationRepeat, {1, 1, 8, 8});
        Adds(activation, activation, 1.0f, activationCount, activationRepeat, {1, 1, 8, 8});
        Div(activation, ones, activation, activationCount, activationRepeat, {1, 1, 1, 8, 0, 8});
        WaitFlag<HardEvent::MTE2_V>(1);
        if constexpr (sizeof(T) == 2) ConfigTransposeCandidateInputs<Rows, 512, 64>();
        else ConfigTransposeCandidateInputs<Rows, 0, 32>();
        auto columns = At<float>(1024);
        auto priors = At<float>(1056);
        constexpr bool IsSetMask = sizeof(T) == 4;
        constexpr uint64_t GeometryMask = IsSetMask ? 8 : MASK_PLACEHOLDER;
        if constexpr (!IsSetMask) {
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(8);
        }
        if constexpr (ScaledSigmoid) {
        Muls<float, IsSetMask>(columns[5 * Pitch], columns[5 * Pitch], 0.25f, GeometryMask, uint8_t(1), {1, 1, 2, 2});
        } else {
            Muls<float, IsSetMask>(columns, columns, 2.0f, GeometryMask, uint8_t(4), {1, 1, 2, 2});
            }
        Adds<float, IsSetMask>(columns, columns, -0.5f, GeometryMask, uint8_t(2), {1, 1, 2, 2});
        Mul<float, IsSetMask>(columns[2 * Pitch], columns[2 * Pitch], columns[2 * Pitch], GeometryMask, uint8_t(2), {1, 1, 1, 2, 2, 2});
        if constexpr (!ScaledSigmoid)
        Mul<float, IsSetMask>(columns[4 * Pitch], columns[4 * Pitch], columns[5 * Pitch], GeometryMask, uint8_t(1), {1, 1, 1, 2, 2, 2});
        if constexpr (ScaledSigmoid)
        Mul<float, IsSetMask>(columns[4 * Pitch], columns[4 * Pitch], columns[5 * Pitch], GeometryMask, uint8_t(1), {1, 1, 1, 2, 2, 2});
        Add<float, IsSetMask>(columns, columns, priors, GeometryMask, uint8_t(2), {1, 1, 1, 2, 2, 2});
        Mul<float, IsSetMask>(columns[2 * Pitch], columns[2 * Pitch], priors[3 * Pitch], GeometryMask, uint8_t(2), {1, 1, 1, 2, 2, 2});
        if constexpr (ScaledSigmoid) {
            Mul<float, IsSetMask>(columns, columns, priors[2 * Pitch], GeometryMask,
                uint8_t(4), {1, 1, 1, 2, 2, 0});
        } else {
            for (uint32_t j = 0; j < 4; ++j)
                Mul<float, IsSetMask>(columns[j * Pitch], columns[j * Pitch], priors[2 * Pitch], GeometryMask, uint8_t(1), {1, 1, 1, 2, 2, 2});
        }
        if constexpr (!IsSetMask) ResetMask();
        if constexpr (sizeof(T) == 2) {
            StoreNativeHalfSmallCandidates<Rows>(outGlobal, columns, begin, count);
        } else {
        uint64_t sources[16], targets[16];
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            sources[j] = 1024 + j * 64;
            sources[j + 8] = sources[j] + 32;
            targets[j * 2] = 3072 + j * 32;
            targets[j * 2 + 1] = 3072 + (j + 8) * 32;
        }
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams{false, false, 1, 0, 0});
        auto packed = At<float>(4096);
        uint64_t selected;
        GatherMask(packed, At<float>(3072), index, false, 0, {1, 1, 8, 0}, selected);
        LocalTensor<T> result;
        if constexpr (sizeof(T) == 2) {
                result = At<T>(4352, 64, TPosition::VECOUT);
            Cast(result, packed, RoundMode::CAST_RINT, Rows * 5);
        } else result = packed;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(outGlobal[begin * 5], result,
            DataCopyExtParams{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0});
        }
    }
};
