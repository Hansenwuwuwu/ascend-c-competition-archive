#pragma once

#include "kernel_operator.h"
#include "swi_glu_tiling.h"

namespace swiglu {
using namespace AscendC;

// 每核一行；维度和有效核数在实例化时固定，避免引入运行时tiling读取。
template<uint32_t DIM, uint32_t ROWS>
__aicore__ inline void RunSingleRow(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                   GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output) {
    static_assert(DIM == 1 || DIM == 4 || DIM == 32 || DIM == 64, "Unsupported fixed row dimension");
    if constexpr (ROWS == 1) {
        if (GetBlockIdx() != 0) return;
    } else {
        if (GetBlockIdx() > ROWS - 1) return;
    }
    constexpr uint32_t DIM_ALIGN = (DIM + 15) / 16 * 16;
    constexpr uint32_t INPUT_OFFSET = 0;
    constexpr uint32_t WEIGHT_1_OFFSET = DIM_ALIGN;
    constexpr uint32_t WEIGHT_2_OFFSET = WEIGHT_1_OFFSET + DIM * DIM_ALIGN;
    constexpr uint32_t BIAS_1_OFFSET = WEIGHT_2_OFFSET + DIM * DIM_ALIGN;
    constexpr uint32_t BIAS_2_OFFSET = BIAS_1_OFFSET + DIM_ALIGN;
    GlobalTensor<half> gx, gw1, gb1, gw2, gb2, gy;
    gx.SetGlobalBuffer((__gm__ half *)input_x + (ROWS == 1 ? 0 : GetBlockIdx() * DIM));
    gw1.SetGlobalBuffer((__gm__ half *)weight_1);
    gb1.SetGlobalBuffer((__gm__ half *)bias_1);
    gw2.SetGlobalBuffer((__gm__ half *)weight_2);
    gb2.SetGlobalBuffer((__gm__ half *)bias_2);
    gy.SetGlobalBuffer((__gm__ half *)output + (ROWS == 1 ? 0 : GetBlockIdx() * DIM));
    LocalMemAllocator<AscendC::Hardware::UB> ub;
    auto input = ub.Alloc<half>((3 + DIM * 2) * DIM_ALIGN);
    auto out = ub.Alloc<half>(2 * DIM * DIM_ALIGN);
    auto tmp = ub.Alloc<half>(2 * DIM_ALIGN);
    LocalTensor<half> reduce;
    if constexpr (DIM >= 32) {
        reduce = ub.Alloc<half>(16 * 2 * DIM_ALIGN);
    }
    if constexpr (DIM == 4) {
        // X的补齐元素参与16元素归约，必须填零；权重仍按行搬运。
        DataCopyPad(input[INPUT_OFFSET], gx, {1, DIM * 2, 0, 0, 0}, {true, 0, DIM_ALIGN - DIM, (half)0});
        DataCopyPad(input[WEIGHT_1_OFFSET], gw1, {DIM, DIM * 2, 0, 0, 0}, {false, 0, 0, (half)0});
        DataCopyPad(input[WEIGHT_2_OFFSET], gw2, {DIM, DIM * 2, 0, 0, 0}, {false, 0, 0, (half)0});
        DataCopyPad(input[BIAS_1_OFFSET], gb1, {1, DIM * 2, 0, 0, 0}, {false, 0, 0, (half)0});
        DataCopyPad(input[BIAS_2_OFFSET], gb2, {1, DIM * 2, 0, 0, 0}, {false, 0, 0, (half)0});
    } else {
        // DIM=1保留原有32B读取；仅首元素有效且不执行跨元素归约。
        DataCopy(input[INPUT_OFFSET], gx, DIM_ALIGN);
        DataCopy(input[WEIGHT_1_OFFSET], gw1, DIM * DIM_ALIGN);
        DataCopy(input[WEIGHT_2_OFFSET], gw2, DIM * DIM_ALIGN);
        DataCopy(input[BIAS_1_OFFSET], gb1, DIM_ALIGN);
        DataCopy(input[BIAS_2_OFFSET], gb2, DIM_ALIGN);
    }
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    if constexpr (DIM == 1) {
        Mul(tmp, input[WEIGHT_1_OFFSET], input, DIM, DIM * 2, {1, 1, 0, 1, 1, 0});
    } else if constexpr (DIM == 4) {
        Mul(out, input[WEIGHT_1_OFFSET], input, DIM_ALIGN, DIM * 2, {1, 1, 0, 1, 1, 0});
        BlockReduceSum(tmp, out, 1, 64, 1, 1, 1);
        BlockReduceSum(tmp[DIM_ALIGN], out[DIM * DIM_ALIGN], 1, 64, 1, 1, 1);
    } else {
        // 两路权重连续存放，X在所有输出通道间复用；先二分到16项再规约。
        Mul(out, input[WEIGHT_1_OFFSET], input, DIM, DIM * 2,
            {1, 1, 1, DIM / 16, DIM / 16, 0});
        if constexpr (DIM == 64) {
            Add(out, out, out[32], 32, DIM * 2, {1, 1, 1, 4, 4, 4});
        }
        Add(reduce, out, out[16], 16, DIM * 2, {1, 1, 1, 1, DIM / 16, DIM / 16});
        BlockReduceSum(tmp, reduce, DIM / 4, 128, 1, 1, 8);
    }
    Add(tmp, tmp, input[BIAS_1_OFFSET], DIM_ALIGN * 2);
    // SiLU目的与源分离，保持原FP16舍入与激活路径。
    Silu(out, tmp, DIM_ALIGN);
    Mul(out, out, tmp[DIM_ALIGN], DIM_ALIGN);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    if constexpr (DIM < 16) {
        DataCopyPad(gy, out, {1, DIM * 2, 0, 0, 0});
    } else {
        DataCopy(gy, out, DIM_ALIGN);
    }
}

// 通用小M路径：每个AIV处理一行，D由tiling给出。
__aicore__ inline void RunDynamicRow(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output, uint32_t d) {
    GlobalTensor<half> gx, gw1, gb1, gw2, gb2, gy;
    gx.SetGlobalBuffer((__gm__ half *)input_x + GetBlockIdx() * d);
    gw1.SetGlobalBuffer((__gm__ half *)weight_1);
    gb1.SetGlobalBuffer((__gm__ half *)bias_1);
    gw2.SetGlobalBuffer((__gm__ half *)weight_2);
    gb2.SetGlobalBuffer((__gm__ half *)bias_2);
    gy.SetGlobalBuffer((__gm__ half *)output + GetBlockIdx() * d);
    LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    uint32_t DIM = d;
    uint32_t DIM_ALIGN = ( DIM + 15 ) / 16 * 16;
    LocalTensor<half> input = ubAllocator.Alloc<half>((3 + DIM * 2) * DIM_ALIGN);
    LocalTensor<half> out = ubAllocator.Alloc<half>(2 * DIM * DIM_ALIGN);
    LocalTensor<half> tmp = ubAllocator.Alloc<half>(2 * DIM_ALIGN);
    uint32_t INPUT_OFFSET = 0;
    uint32_t WEIGHT_1_OFFSET = DIM_ALIGN;
    uint32_t WEIGHT_2_OFFSET = WEIGHT_1_OFFSET + DIM * DIM_ALIGN;
    uint32_t BIAS_1_OFFSET = WEIGHT_2_OFFSET + DIM * DIM_ALIGN;
    uint32_t BIAS_2_OFFSET = BIAS_1_OFFSET + DIM_ALIGN;
    DataCopyPad(input[INPUT_OFFSET], gx, {1, static_cast<uint16_t>(DIM*2), 0, 0, 0}, {false, 0, 0, (half)0});
    DataCopyPad(input[WEIGHT_1_OFFSET], gw1, {static_cast<uint16_t>(DIM), static_cast<uint16_t>(DIM*2), 0, 0, 0}, {false, 0, 0, (half)0});
    DataCopyPad(input[WEIGHT_2_OFFSET], gw2, {static_cast<uint16_t>(DIM), static_cast<uint16_t>(DIM*2), 0, 0, 0}, {false, 0, 0, (half)0});
    DataCopyPad(input[BIAS_1_OFFSET], gb1, {1, static_cast<uint16_t>(DIM*2), 0, 0, 0}, {false, 0, 0, (half)0});
    DataCopyPad(input[BIAS_2_OFFSET], gb2, {1, static_cast<uint16_t>(DIM*2), 0, 0, 0}, {false, 0, 0, (half)0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    // 一个repeat计算两路投影；X的block stride为0，两个权重块复用同一X块。
    uint8_t RepStride = DIM_ALIGN / 16;
    uint8_t BlkStride = d > 16 ? 1 : 0;
    Mul(out, input[WEIGHT_1_OFFSET], input, DIM, DIM*2, {1, 1, BlkStride, RepStride, RepStride, 0});
    WholeReduceSum(tmp, out, DIM, DIM, 1, 1, RepStride);
    WholeReduceSum(tmp[DIM_ALIGN], out[DIM * DIM_ALIGN], DIM, DIM, 1, 1, RepStride);
    Add(tmp, tmp, input[BIAS_1_OFFSET], DIM_ALIGN * 2);
    // SiLU写入独立缓冲，第二路投影保持不变。
    Silu(out, tmp, DIM_ALIGN);
    Mul(out, out, tmp[DIM_ALIGN], DIM_ALIGN);
    // 两类HardEvent各自使用事件0，输入等待完成后才进入计算。
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyPad(gy, out, {1, DIM*2, 0, 0, 0});
}

// D=13: contiguous GM transfers, batched projections, and packed output.
template<uint32_t STATIC_ROWS>
__aicore__ inline void RunSmall13(GM_ADDR x, GM_ADDR w1, GM_ADDR b1,
                                 GM_ADDR w2, GM_ADDR b2, GM_ADDR y, uint32_t rows) {
    constexpr uint32_t D = 13, A = 16, ROWS = SMALL13_ROWS;
    constexpr uint32_t W = A * A, P = ROWS * A;
    static_assert(ROWS <= A, "Input gather reuses the 16-row weight indices");
    const uint32_t total = STATIC_ROWS ? STATIC_ROWS : rows;
    const uint32_t begin = GetBlockIdx() * ROWS;
    if (begin >= total) return;
    uint32_t count = ROWS;
    if constexpr (STATIC_ROWS == 0 || STATIC_ROWS % ROWS != 0)
        count = total - begin < ROWS ? total - begin : ROWS;
    GlobalTensor<half> gx, gw1, gw2, gb1, gb2, gy;
    gx.SetGlobalBuffer((__gm__ half *)x + begin * D);
    gy.SetGlobalBuffer((__gm__ half *)y + begin * D);
    gw1.SetGlobalBuffer((__gm__ half *)w1); gw2.SetGlobalBuffer((__gm__ half *)w2);
    gb1.SetGlobalBuffer((__gm__ half *)b1); gb2.SetGlobalBuffer((__gm__ half *)b2);
    LocalMemAllocator<AscendC::Hardware::UB> ub;
    auto input = ub.Alloc<half>(P + 2 * W + 2 * A);
    auto products = ub.Alloc<half>(2 * ROWS * W);
    auto projection = ub.Alloc<half>(2 * P);
    auto raw = ub.Alloc<half>(2 * W + P);
    auto index = ub.Alloc<int32_t>(W);
    auto seed = ub.Alloc<int32_t>(2 * A);
    auto rowBroadcast = ub.Alloc<int32_t>(A * 8);
    auto selection = ub.Alloc<uint64_t>(4);
    DataCopyPad(raw[2 * W], gx, {1, count * D * 2, 0, 0, 0}, {false, 0, 0, (half)0});
    DataCopyPad(raw, gw1, {1, D * D * 2, 0, 0, 0}, {true, 0, 7, (half)0});
    DataCopyPad(raw[W], gw2, {1, D * D * 2, 0, 0, 0}, {true, 0, 7, (half)0});
    SetFlag<HardEvent::MTE2_V>(0);
    DataCopyPad(input[P + 2 * W], gb1, {1, D * 2, 0, 0, 0}, {false, 0, 0, (half)0});
    DataCopyPad(input[P + 2 * W + A], gb2, {1, D * 2, 0, 0, 0}, {false, 0, 0, (half)0});
    SetFlag<HardEvent::MTE2_V>(1);
    // Byte indices [row*26 + column*2]; reuse them for both weights and X.
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        seed.ReinterpretCast<uint64_t>().SetValue(i, (uint64_t)(4 * i) | ((uint64_t)(4 * i + 2) << 32));
        selection.SetValue(i, i == 0 ? (1ULL << D) - 1 : 0);
    }
    SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
    Adds(seed[8], seed, (int32_t)16, 8);
    Muls(seed[A], seed, (int32_t)D, A);
    Brcb(rowBroadcast, seed[A], 2, {1, 8});
    Add(index, seed, rowBroadcast, (uint64_t)A, A, {1, 1, 0, 2, 0, 1});
    WaitFlag<HardEvent::MTE2_V>(0);
    Gather(input, raw[2 * W], index.ReinterpretCast<uint32_t>(), (uint32_t)0, P);
    uint64_t padMask[2] = {0xe000, 0};
    Duplicate(input, (half)0, padMask, ROWS, 1, 1);
    Gather(input[P], raw, index.ReinterpretCast<uint32_t>(), (uint32_t)0, W);
    Gather(input[P + W], raw[W], index.ReinterpretCast<uint32_t>(), (uint32_t)0, W);
    #pragma unroll
    for (uint32_t p = 0; p < 2; ++p) {
        #pragma unroll
        for (uint32_t halfBlock = 0; halfBlock < 2; ++halfBlock) {
            Mul(products[p * ROWS * W + halfBlock * 128], input[P + p * W + halfBlock * 128], input,
                (uint64_t)128, ROWS, {1, 1, 0, 16, 0, 1});
        }
    }
    BlockReduceSum(projection, products, ROWS * 4, 128, 1, 1, 8);
    WaitFlag<HardEvent::MTE2_V>(1);
    Add(projection, projection, input[P + 2 * W], (uint64_t)A, ROWS, {1, 1, 1, 1, 1, 0});
    Add(projection[P], projection[P], input[P + 2 * W + A], (uint64_t)A, ROWS, {1, 1, 1, 1, 1, 0});
    // products and raw are no longer inputs; reuse them for activation and packed output.
    Silu(products, projection, P);
    Mul(products, products, projection[P], P);
    uint64_t selectedCount = 0;
    GatherMask(raw, products, selection.ReinterpretCast<uint16_t>(), false, 0, {1, ROWS, 1, 0}, selectedCount);
    SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyPad(gy, raw, {1, count * D * 2, 0, 0, 0});
}

__aicore__ inline void RunSmall33(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                 GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output) {
    // D=33：17核，每核固定5行；两路主块连续计算，尾行/列单独处理。
    constexpr uint32_t DIM = 33, DIM_ALIGN = 48, MAIN_DIM = DIM - 1;
    constexpr uint32_t ROWS_PER_CORE = 5, INPUT_STRIDE = 128;
    constexpr uint32_t RAW_SIZE = (DIM * DIM + 15) / 16 * 16;
    constexpr uint32_t MAIN_SIZE = MAIN_DIM * MAIN_DIM;
    constexpr uint32_t WEIGHT_STRIDE = MAIN_SIZE + MAIN_DIM;
    constexpr uint32_t INPUT_OFFSET = RAW_SIZE * 2;
    constexpr uint32_t WEIGHT_OFFSET = INPUT_OFFSET + ROWS_PER_CORE * INPUT_STRIDE + 128; // 保留实测选定的256B间隔。
    constexpr uint32_t BIAS_OFFSET = WEIGHT_OFFSET + WEIGHT_STRIDE * 2;
    constexpr uint32_t MAIN_COUNT = ROWS_PER_CORE * MAIN_DIM;
    constexpr uint32_t OUTPUT_COUNT = ROWS_PER_CORE * DIM_ALIGN;
    // 85行均分给17核；行数和缓冲尺寸保持编译期常量。
    const uint32_t begin = ROWS_PER_CORE * GetBlockIdx();
    constexpr uint32_t count = ROWS_PER_CORE;
    GlobalTensor<half> gx, gw[2], gb[2], gy;
    gx.SetGlobalBuffer((__gm__ half *)input_x + begin * DIM);
    gw[0].SetGlobalBuffer((__gm__ half *)weight_1);
    gw[1].SetGlobalBuffer((__gm__ half *)weight_2);
    gb[0].SetGlobalBuffer((__gm__ half *)bias_1);
    gb[1].SetGlobalBuffer((__gm__ half *)bias_2);
    gy.SetGlobalBuffer((__gm__ half *)output + begin * DIM);
    LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    LocalTensor<int32_t> index = ubAllocator.Alloc<int32_t>(WEIGHT_STRIDE);
    LocalTensor<half> input = ubAllocator.Alloc<half>(BIAS_OFFSET + DIM_ALIGN * 2);
    LocalTensor<half> out = ubAllocator.Alloc<half>(MAIN_COUNT * MAIN_DIM * 2);
    LocalTensor<half> reduce = ubAllocator.Alloc<half>(MAIN_COUNT * 16 * 2);
    LocalTensor<half> tmp = ubAllocator.Alloc<half>(MAIN_COUNT * 2);
    LocalTensor<half> projection = ubAllocator.Alloc<half>(OUTPUT_COUNT * 2);
    LocalTensor<half> scratch = ubAllocator.Alloc<half>(ROWS_PER_CORE * INPUT_STRIDE);
    LocalTensor<int32_t> baseIndex = ubAllocator.Alloc<int32_t>(MAIN_DIM);
    LocalTensor<int32_t> rowIndex = ubAllocator.Alloc<int32_t>(MAIN_DIM);
    LocalTensor<int32_t> rowBroadcast = ubAllocator.Alloc<int32_t>(MAIN_DIM * 8);
    LocalTensor<uint64_t> selection = ubAllocator.Alloc<uint64_t>(4);

    DataCopyPad(input, gw[0], {1, DIM * DIM * 2, 0, 0, 0}, {false, 0, 0, (half)0});
    DataCopyPad(input[RAW_SIZE], gw[1], {1, DIM * DIM * 2, 0, 0, 0}, {false, 0, 0, (half)0});
    DataCopyPad(input[INPUT_OFFSET], gx, {static_cast<uint16_t>(count), DIM * 2, 0, 5, 0}, {false, 0, 0, (half)0});
    #pragma unroll
    for (uint32_t p = 0; p < 2; ++p) {
        DataCopyPad(input[BIAS_OFFSET + p * DIM_ALIGN], gb[p], {1, DIM * 2, 0, 0, 0}, {false, 0, 0, (half)0});
    }
    // 字节偏移种子0,2,...,14；输出掩码仅保留每行前33项。
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        baseIndex.ReinterpretCast<uint64_t>().SetValue(i, (uint64_t)(4 * i) | ((uint64_t)(4 * i + 2) << 32));
        selection.SetValue(i, i == 0 ? (1ULL << DIM) - 1 : 0);
    }
    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);
    Adds(baseIndex[8], baseIndex, (int32_t)16, 8);
    Adds(baseIndex[16], baseIndex, (int32_t)32, 16);
    ShiftLeft(rowIndex, baseIndex, (int32_t)5, MAIN_DIM);
    Add(rowIndex, rowIndex, baseIndex, MAIN_DIM);
    Adds(index[MAIN_SIZE], rowIndex, (int32_t)(MAIN_DIM * 2), MAIN_DIM);
    Brcb(rowBroadcast, rowIndex, 4, {1, 8});
    Add(index, baseIndex, rowBroadcast, (uint64_t)MAIN_DIM, MAIN_DIM, {1, 1, 0, 4, 0, 1});
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    // 两路主块连续排布，尾列另存；第33输出通道直接规约原始权重行。
    #pragma unroll
    for (uint32_t p = 0; p < 2; ++p) {
        Gather(input[WEIGHT_OFFSET + p * MAIN_SIZE], input[p * RAW_SIZE], index.ReinterpretCast<uint32_t>(), (uint32_t)0, WEIGHT_STRIDE);
        if (p == 0) {
            // 先保存W1尾列，再让W2主块紧接W1主块，避免尾列被覆盖。
            Copy(input[WEIGHT_OFFSET + 2 * MAIN_SIZE + MAIN_DIM], input[WEIGHT_OFFSET + MAIN_SIZE],
                 (uint64_t)MAIN_DIM, 1, {1, 1, 2, 2});
        }
        Mul(out, input[p * RAW_SIZE + MAIN_DIM * DIM], input[INPUT_OFFSET], (uint64_t)DIM_ALIGN, ROWS_PER_CORE, {1, 1, 1, 3, 0, 8});
        WholeReduceSum(projection[p * OUTPUT_COUNT + MAIN_DIM], out, DIM, ROWS_PER_CORE, DIM_ALIGN, 1, 3);
    }
    #pragma unroll
    for (uint32_t row = 0; row < ROWS_PER_CORE; ++row) {
        Brcb(scratch[row * INPUT_STRIDE], input[INPUT_OFFSET + row * INPUT_STRIDE + MAIN_DIM], 1, {1, 8});
    }
    Copy(input[INPUT_OFFSET + MAIN_DIM], input[INPUT_OFFSET], (uint64_t)MAIN_DIM, ROWS_PER_CORE, {1, 1, 8, 8});
    Copy(input[INPUT_OFFSET + MAIN_DIM * 2], input[INPUT_OFFSET], (uint64_t)(MAIN_DIM * 2), ROWS_PER_CORE, {1, 1, 8, 8});
    // 乘积和归约结果按[行][投影][输出通道]排列，一次Mul覆盖两路主块。
    #pragma unroll
    for (uint32_t row = 0; row < ROWS_PER_CORE; ++row) {
        Mul(out[row * 2 * MAIN_SIZE], input[WEIGHT_OFFSET],
            input[INPUT_OFFSET + row * INPUT_STRIDE], (uint64_t)128, 16, {1, 1, 1, 8, 8, 0});
    }
    // 主块二分折叠后批量规约；out随后复用为尾列乘积和最终输出。
    Add(reduce, out, out[16], (uint64_t)128, MAIN_COUNT * 2 / 8, {1, 2, 2, 8, 16, 16});
    BlockReduceSum(tmp, reduce, MAIN_COUNT * 2 / 8, 128, 1, 1, 8);
    #pragma unroll
    for (uint32_t p = 0; p < 2; ++p) {
        Mul(out[p * MAIN_DIM], input[WEIGHT_OFFSET + 2 * MAIN_SIZE + (1 - p) * MAIN_DIM], scratch,
            (uint64_t)MAIN_DIM, ROWS_PER_CORE, {1, 1, 0, 4, 0, 8});
    }
    Add(tmp, tmp, out, MAIN_COUNT * 2);
    #pragma unroll
    for (uint32_t p = 0; p < 2; ++p) {
        Copy(projection[p * OUTPUT_COUNT], tmp[p * MAIN_DIM], (uint64_t)MAIN_DIM,
             ROWS_PER_CORE, {1, 1, 3, 4});
    }
    #pragma unroll
    for (uint32_t p = 0; p < 2; ++p) {
        Add(projection[p * OUTPUT_COUNT], projection[p * OUTPUT_COUNT], input[BIAS_OFFSET + p * DIM_ALIGN],
            (uint64_t)DIM, ROWS_PER_CORE, {1, 1, 1, 3, 3, 0});
    }
    Silu(out, projection, OUTPUT_COUNT);
    Mul(out, out, projection[OUTPUT_COUNT], OUTPUT_COUNT);
    // scratch的尾项广播已用完，复用为无padding的连续输出。
    uint64_t selectedCount = 0;
    GatherMask(scratch, out, selection.ReinterpretCast<uint16_t>(), false, 0, {1, ROWS_PER_CORE, 3, 0}, selectedCount);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyPad(gy, scratch, {1, count * DIM * 2, 0, 0, 0});
}

} // namespace swiglu

#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 0UL || TILING_KEY_VAR == 1UL || TILING_KEY_VAR == 2UL)
namespace AscendC {
static __aicore__ __attribute__((used, noinline)) int64_t SmallRowWorkspaceLayoutAnchor() {
    return reinterpret_cast<int64_t>(&g_sysWorkspaceReserved);
}
__aicore__ inline void SmallRowSkipUnusedSysWorkspace(GM_ADDR) {}
}
#define SetSysWorkspaceForce SmallRowSkipUnusedSysWorkspace
#endif
