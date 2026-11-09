#pragma once

template <typename T>
class PairedSmallBatchCandidateDecode {
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t address, uint32_t count = 256,
        TPosition position = TPosition::VECCALC)
    {
        return LocalTensor<U>(position, address, count);
    }
public:
    __aicore__ inline void Run(GM_ADDR rawAddress, GM_ADDR priorAddress, GM_ADDR outAddress,
        uint32_t candidates, uint32_t begin)
    {
        constexpr uint32_t Pitch = 16;
        const uint32_t count = candidates - begin < 8 ? candidates - begin : 8;
        GlobalTensor<T> rawGlobal, outGlobal;
        GlobalTensor<float> priorGlobal;
        rawGlobal.SetGlobalBuffer((__gm__ T *)rawAddress);
        priorGlobal.SetGlobalBuffer((__gm__ float *)priorAddress);
        outGlobal.SetGlobalBuffer((__gm__ T *)outAddress);
        auto raw = At<T>(0, 256, TPosition::VECIN);
        auto prior = At<float>(1024, 128, TPosition::VECIN);
        DataCopyPad(raw, rawGlobal[begin * 6],
            DataCopyExtParams{uint16_t(count), 6 * uint32_t(sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        DataCopyPad(raw[256 / sizeof(T)], rawGlobal[(candidates + begin) * 6],
            DataCopyExtParams{uint16_t(count), 6 * uint32_t(sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0);
        DataCopyPad(prior, priorGlobal[begin * 5],
            DataCopyExtParams{uint16_t(count), 20, 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(1);
        auto two = At<float>(9216, 64);
        auto pattern = At<uint32_t>(9728, 8);
        Duplicate(two, 2.0f, 64);
        Duplicate(pattern, uint32_t(0x1f1f1f1f), 2);
        WaitFlag<HardEvent::MTE2_V>(0);
        LocalTensor<float> activation;
        if constexpr (sizeof(T) == 2) {
            activation = At<float>(2048, 256);
            Cast(activation, raw, RoundMode::CAST_NONE, 256);
        } else activation = raw;
        constexpr uint8_t repeats = sizeof(T) == 2 ? 4 : 2;
        Muls(activation, activation, -1.0f, uint64_t(64), repeats, {1, 1, 8, 8});
        Exp(activation, activation, uint64_t(64), repeats, {1, 1, 8, 8});
        Adds(activation, activation, 1.0f, uint64_t(64), repeats, {1, 1, 8, 8});
        Div(activation, two, activation, uint64_t(64), repeats, {1, 1, 1, 8, 0, 8});
        WaitFlag<HardEvent::MTE2_V>(1);
        uint64_t sources[16], targets[16];
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            constexpr uint32_t base = sizeof(T) == 2 ? 2048 : 0;
            constexpr uint32_t rowBytes = sizeof(T) == 2 ? 64 : 32;
            sources[j] = base + j * rowBytes;
            sources[j + 8] = base + (j + 8) * rowBytes;
            targets[j * 2] = 4096 + j * 64;
            targets[j * 2 + 1] = targets[j * 2] + 32;
        }
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams{false, false, 1, 0, 0});
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            sources[j] = 1024 + j * 32;
            sources[j + 8] = sources[j];
            targets[j * 2] = 4608 + j * 64;
            targets[j * 2 + 1] = targets[j * 2] + 32;
        }
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams{false, false, 1, 0, 0});
        auto columns = At<float>(4096, 256);
        auto priors = At<float>(4608, 256);
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(16);
        Muls<float, false>(columns[5 * Pitch], columns[5 * Pitch], 0.25f,
            MASK_PLACEHOLDER, uint8_t(1), {1, 1, 2, 2});
        Adds<float, false>(columns, columns, -0.5f, MASK_PLACEHOLDER, uint8_t(2), {1, 1, 2, 2});
        Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], columns[2 * Pitch],
            MASK_PLACEHOLDER, uint8_t(2), {1, 1, 1, 2, 2, 2});
        Mul<float, false>(columns[4 * Pitch], columns[4 * Pitch], columns[5 * Pitch],
            MASK_PLACEHOLDER, uint8_t(1), {1, 1, 1, 2, 2, 2});
        Add<float, false>(columns, columns, priors, MASK_PLACEHOLDER, uint8_t(2), {1, 1, 1, 2, 2, 2});
        Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], priors[3 * Pitch],
            MASK_PLACEHOLDER, uint8_t(2), {1, 1, 1, 2, 2, 2});
        Mul<float, false>(columns, columns, priors[2 * Pitch],
            MASK_PLACEHOLDER, uint8_t(4), {1, 1, 1, 2, 2, 0});
        ResetMask();
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            sources[j] = 4096 + j * 64;
            sources[j + 8] = sources[j] + 32;
            targets[j * 2] = 6144 + j * 32;
            targets[j * 2 + 1] = 6144 + (j + 8) * 32;
        }
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams{false, false, 1, 0, 0});
        auto packed = At<float>(8192, 128, TPosition::VECOUT);
        uint64_t selected;
        GatherMask(packed, At<float>(6144), pattern, false, 0, {1, 2, 8, 0}, selected);
        LocalTensor<T> output;
        constexpr uint32_t outputPitch = sizeof(T) == 2 ? 96 : 160;
        if constexpr (sizeof(T) == 2) {
            output = At<T>(0, 128, TPosition::VECOUT);
            Cast(output, packed, RoundMode::CAST_RINT, 40);
            Cast(output[48], packed[40], RoundMode::CAST_RINT, 40);
        } else output = packed;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const uint32_t bytes = count * 5 * uint32_t(sizeof(T));
        DataCopyPad(outGlobal[begin * 5], output,
            DataCopyExtParams{2, bytes, (outputPitch - ((bytes + 31) / 32) * 32) / 32,
                (candidates - count) * 5 * uint32_t(sizeof(T)), 0});
    }
};

template <typename T>
class WideSingleSmallCandidateDecode {
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t address, uint32_t count = 256,
        TPosition position = TPosition::VECCALC)
    {
        return LocalTensor<U>(position, address, count);
    }
public:
    __aicore__ inline void Run(GM_ADDR rawAddress, GM_ADDR priorAddress, GM_ADDR outAddress,
        uint32_t begin)
    {
        constexpr uint32_t Pitch = 16;
        constexpr uint32_t count = 16;
        GlobalTensor<T> rawGlobal, outGlobal;
        GlobalTensor<float> priorGlobal;
        rawGlobal.SetGlobalBuffer((__gm__ T *)rawAddress);
        priorGlobal.SetGlobalBuffer((__gm__ float *)priorAddress);
        outGlobal.SetGlobalBuffer((__gm__ T *)outAddress);
        auto raw = At<T>(0, 256, TPosition::VECIN);
        auto prior = At<float>(1024, 128, TPosition::VECIN);
        DataCopyPad(raw, rawGlobal[begin * 6],
            DataCopyExtParams{uint16_t(count), 6 * uint32_t(sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0);
        ICachePreLoad(1);
        DataCopyPad(prior, priorGlobal[begin * 5],
            DataCopyExtParams{uint16_t(count), 20, 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(1);
        auto two = At<float>(9216, 64);
        auto pattern = At<uint32_t>(9728, 8);
        Duplicate(two, 2.0f, 64);
        Duplicate(pattern, uint32_t(0x1f1f1f1f), 2);
        WaitFlag<HardEvent::MTE2_V>(0);
        LocalTensor<float> activation;
        if constexpr (sizeof(T) == 2) {
            activation = At<float>(2048, 256);
            Cast(activation, raw, RoundMode::CAST_NONE, 256);
        } else activation = raw;
        constexpr uint8_t repeats = sizeof(T) == 2 ? 4 : 2;
        Muls(activation, activation, -1.0f, uint64_t(64), repeats, {1, 1, 8, 8});
        Exp(activation, activation, uint64_t(64), repeats, {1, 1, 8, 8});
        Adds(activation, activation, 1.0f, uint64_t(64), repeats, {1, 1, 8, 8});
        Div(activation, two, activation, uint64_t(64), repeats, {1, 1, 1, 8, 0, 8});
        WaitFlag<HardEvent::MTE2_V>(1);
        uint64_t sources[16], targets[16];
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            constexpr uint32_t base = sizeof(T) == 2 ? 2048 : 0;
            constexpr uint32_t rowBytes = sizeof(T) == 2 ? 64 : 32;
            sources[j] = base + j * rowBytes;
            sources[j + 8] = base + (j + 8) * rowBytes;
            targets[j * 2] = 4096 + j * 64;
            targets[j * 2 + 1] = targets[j * 2] + 32;
        }
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams{false, false, 1, 0, 0});
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            sources[j] = 1024 + j * 32;
            sources[j + 8] = 1024 + (j + 8) * 32;
            targets[j * 2] = 4608 + j * 64;
            targets[j * 2 + 1] = targets[j * 2] + 32;
        }
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams{false, false, 1, 0, 0});
        auto columns = At<float>(4096, 256);
        auto priors = At<float>(4608, 256);
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(16);
        Muls<float, false>(columns[5 * Pitch], columns[5 * Pitch], 0.25f,
            MASK_PLACEHOLDER, uint8_t(1), {1, 1, 2, 2});
        Adds<float, false>(columns, columns, -0.5f, MASK_PLACEHOLDER, uint8_t(2), {1, 1, 2, 2});
        Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], columns[2 * Pitch],
            MASK_PLACEHOLDER, uint8_t(2), {1, 1, 1, 2, 2, 2});
        Mul<float, false>(columns[4 * Pitch], columns[4 * Pitch], columns[5 * Pitch],
            MASK_PLACEHOLDER, uint8_t(1), {1, 1, 1, 2, 2, 2});
        Add<float, false>(columns, columns, priors, MASK_PLACEHOLDER, uint8_t(2), {1, 1, 1, 2, 2, 2});
        Mul<float, false>(columns[2 * Pitch], columns[2 * Pitch], priors[3 * Pitch],
            MASK_PLACEHOLDER, uint8_t(2), {1, 1, 1, 2, 2, 2});
        Mul<float, false>(columns, columns, priors[2 * Pitch],
            MASK_PLACEHOLDER, uint8_t(4), {1, 1, 1, 2, 2, 0});
        ResetMask();
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            sources[j] = 4096 + j * 64;
            sources[j + 8] = sources[j] + 32;
            targets[j * 2] = 6144 + j * 32;
            targets[j * 2 + 1] = 6144 + (j + 8) * 32;
        }
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams{false, false, 1, 0, 0});
        auto packed = At<float>(8192, 128, TPosition::VECOUT);
        uint64_t selected;
        GatherMask(packed, At<float>(6144), pattern, false, 0, {1, 2, 8, 0}, selected);
        LocalTensor<T> output;
        constexpr uint32_t outputPitch = sizeof(T) == 2 ? 96 : 160;
        if constexpr (sizeof(T) == 2) {
            output = At<T>(0, 128, TPosition::VECOUT);
            Cast(output, packed, RoundMode::CAST_RINT, 40);
            Cast(output[48], packed[40], RoundMode::CAST_RINT, 40);
        } else output = packed;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(outGlobal[begin * 5], output,
            DataCopyExtParams{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0});
    }
};
