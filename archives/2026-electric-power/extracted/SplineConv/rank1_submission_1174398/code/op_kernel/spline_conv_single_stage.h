#pragma once
#include "spline_conv_single_interpolate.h"
#include "spline_conv_sparse_output.h"

template <bool Closed0,bool Closed1,bool Tail>
__aicore__ inline void RunSingleStaged(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out) {
    uint32_t n=2*GetBlockNum()-(Tail?1:0),block=GetBlockIdx();
    uint32_t aligned=n/2*2,active=(n+15)/16;
    if constexpr(Tail) active=(aligned+15)/16+1;
    if(block>=active) return;
    GlobalTensor<float> xg,pg,wg,yg;GlobalTensor<int64_t> eg;
    xg.SetGlobalBuffer((__gm__ float*)x);pg.SetGlobalBuffer((__gm__ float*)pseudo);
    wg.SetGlobalBuffer((__gm__ float*)weight);yg.SetGlobalBuffer((__gm__ float*)out);
    eg.SetGlobalBuffer((__gm__ int64_t*)edges);
    LocalTensor<float> weights(TPosition::VECCALC,0,512);
    LocalTensor<float> features(TPosition::VECCALC,2048,8);
    LocalTensor<float> spread(TPosition::VECCALC,2080,64);
    LocalTensor<float> dot(TPosition::VECCALC,2336,128);
    LocalTensor<float> result(TPosition::VECCALC,2848,256);
    DataCopyPadExtParams<float> pad{false,0,0,0.0f};
    DataCopyPad(weights,wg,DataCopyExtParams{1,2048,0,0,0},pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint32_t start=block*16,limit=n;
    if constexpr(Tail) {
        if(block==(aligned+15)/16) start=aligned;
        else limit=aligned;
    }
    uint32_t rows=limit-start;if(rows>16) rows=16;
    Duplicate(result,0.0f,rows*16);
    int64_t source=eg.GetValue(0),target=eg.GetValue(1);
    bool valid=source>=0 && source<n && target>=start && target<int64_t(start+rows);
    float q0=0.0f,q1=0.0f;
    if(valid) {
        DataCopyPad(features,xg[uint32_t(source)*8],DataCopyExtParams{1,32,0,0,0},pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }
    if(valid) {q0=pg.GetValue(0);q1=pg.GetValue(1);}
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if(valid) {
        SplineSingleInterpolate<Closed0,Closed1>(dot,weights,q0,q1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Brcb(spread,features,1,BrcbRepeatParams{1,8});
        Mul(dot,dot,spread,uint64_t(16),uint8_t(8),BinaryRepeatParams{1,1,0,2,2,1});
        Add(dot,dot,dot[64],64);Add(dot,dot,dot[32],32);Add(dot,dot,dot[16],16);
        DataCopy(result[(uint32_t(target)-start)*16],dot,16);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    spline_output::CopyRows(yg,result,start,rows);
}
