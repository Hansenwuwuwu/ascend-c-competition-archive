#pragma once
#include "spline_conv_sparse_output.h"

struct SplineWideTile {
    static constexpr uint32_t C=128;
    template <uint32_t Slot>
    __aicore__ static inline void Load(const LocalTensor<float>& local,const GlobalTensor<float>& global) {
        DataCopy(local[Slot*C*16],global[Slot*C*16],C*16);
        SetFlag<HardEvent::MTE2_V>(TEventID(Slot));
    }
    __aicore__ static inline void Prefetch(const LocalTensor<float>& local,const GlobalTensor<float>& global) {
        Load<0>(local,global);Load<1>(local,global);
        Load<2>(local,global);Load<3>(local,global);
    }
    __aicore__ static inline void Interpolate(const LocalTensor<float>& dot,const LocalTensor<float>& weights,
            float f0,float f1,float f2,float f3) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);Muls(dot,weights,f0,C*16);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);Axpy(dot,weights[C*16],f1,C*16);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);Axpy(dot,weights[2*C*16],f2,C*16);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);Axpy(dot,weights[3*C*16],f3,C*16);
    }
    template <uint32_t Rows>
    __aicore__ static inline void Reduce(const LocalTensor<float>& dot) {
        for(uint32_t r=Rows;r>1;r/=2) Add(dot,dot,dot[r/2*16],r/2*16);
    }
    __aicore__ static inline void FeatureDot(const LocalTensor<float>& dot,const LocalTensor<float>& features,
            const LocalTensor<float>& spread) {
        Brcb(spread,features,uint8_t(C/8),BrcbRepeatParams{2,16});
        Brcb(spread[8],features,uint8_t(C/8),BrcbRepeatParams{2,16});
        Mul(dot,dot,spread,C*16);Reduce<C>(dot);
    }
    __aicore__ static inline void GroupDot(const LocalTensor<float>& dot,const LocalTensor<float>& weights,
            const LocalTensor<float>& agg,const LocalTensor<float>& spread) {
        Brcb(spread,agg,uint8_t(4*C/8),BrcbRepeatParams{1,8});
        for(uint32_t row=0;row<4*C;row+=255) {
            uint32_t count=4*C-row;if(count>255) count=255;
            Mul(dot[row*16],weights[row*16],spread[row*8],uint64_t(16),uint8_t(count),BinaryRepeatParams{1,1,0,2,2,1});
        }
        Reduce<4*C>(dot);
    }
};
