#pragma once

#include "kernel_operator.h"

namespace swiglu {
using namespace AscendC;

// 输入为负Gate，输出的负号由负Up在最终乘法中抵消。
__aicore__ inline void NegativeSilu(LocalTensor<half> gate, LocalTensor<half> input,
                                   uint32_t count) {
    const UnaryRepeatParams unary;
    const BinaryRepeatParams binary;
    SetMaskCount();
    SetVectorMask<half>(0, count);
    Exp<half, false>(gate, input, MASK_PLACEHOLDER, 1, unary);
    PipeBarrier<PIPE_V>();
    Adds<half, false>(gate, gate, half(1), MASK_PLACEHOLDER, 1, unary);
    PipeBarrier<PIPE_V>();
    Div<half, false>(gate, input, gate, MASK_PLACEHOLDER, 1, binary);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

} // namespace swiglu
