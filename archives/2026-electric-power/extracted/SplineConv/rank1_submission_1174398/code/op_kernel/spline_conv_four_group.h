#pragma once
#include "spline_conv_single_interpolate.h"
#include "spline_conv_sparse_output.h"

template <uint32_t Nodes,bool Closed0,bool Closed1>
__aicore__ __attribute__((noinline)) void SplineFourGroupDot(GM_ADDR x,GM_ADDR edges,
        GM_ADDR pseudo,GM_ADDR out,GM_ADDR tiling) {
    uint32_t block=GetBlockIdx();
    block=(block>>1)|((block&1)<<2);
    GlobalTensor<float> xg,pg,yg;GlobalTensor<int64_t> eg;
    xg.SetGlobalBuffer((__gm__ float*)x);pg.SetGlobalBuffer((__gm__ float*)pseudo);
    yg.SetGlobalBuffer((__gm__ float*)out);eg.SetGlobalBuffer((__gm__ int64_t*)edges);
    int64_t target=eg.GetValue(4+block);
    if(target<0 || target>=Nodes) return;
    for(uint32_t e=0;e<block;++e) if(eg.GetValue(4+e)==target) return;
    LocalTensor<float> weights(TPosition::VECCALC,2304,2048);
    LocalTensor<float> features(TPosition::VECCALC,32768,2048);
    LocalTensor<float> aggregate(TPosition::VECCALC,40960,128);
    LocalTensor<float> spread(TPosition::VECCALC,41472,1024);
    LocalTensor<float> dot(TPosition::VECCALC,45568,2048);
    DataCopy(features,xg,Nodes*32);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Duplicate(aggregate,0.0f,128);PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint32_t valid=0;
    for(uint32_t e=0;e<4;++e) {
        if(eg.GetValue(4+e)!=target) continue;
        int64_t source=eg.GetValue(e);if(source<0 || source>=Nodes) continue;
        float factors[4];SplineLinearFactors<Closed0,Closed1>(pg.GetValue(e*2),pg.GetValue(e*2+1),factors);
        for(uint32_t slot=0;slot<4;++slot) {
            Axpy(aggregate[slot*32],features[uint32_t(source)*32],factors[slot],32);
            PipeBarrier<PIPE_V>();
        }
        ++valid;
    }
    if(valid) {
        Brcb(spread,aggregate,16,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        Mul(dot,weights,spread,uint64_t(16),uint8_t(128),BinaryRepeatParams{1,1,0,2,2,1});
        PipeBarrier<PIPE_V>();
        for(uint32_t rows=128;rows>1;rows/=2) {
            Add(dot,dot,dot[rows/2*16],rows/2*16);PipeBarrier<PIPE_V>();
        }
        if(valid>1 && ((const volatile __gm__ SplineConvTilingData*)tiling)->mean) {
            Muls(dot,dot,1.0f/valid,16);PipeBarrier<PIPE_V>();
        }
    } else {Duplicate(dot,0.0f,16);PipeBarrier<PIPE_V>();}
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    spline_output::CopyRows(yg,dot,uint32_t(target),1);
    PipeBarrier<PIPE_ALL>();
}
