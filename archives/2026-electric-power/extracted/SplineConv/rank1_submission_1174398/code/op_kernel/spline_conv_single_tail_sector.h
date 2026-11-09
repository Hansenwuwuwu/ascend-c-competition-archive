#pragma once
#include "spline_conv_single_interpolate.h"
#include "spline_conv_sparse_output.h"

template <bool Closed0,bool Closed1>
__aicore__ inline void RunSingleTailSector(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,
        GM_ADDR weight,GM_ADDR out,uint32_t n) {
    if(GetBlockIdx()!=0) return;
    GlobalTensor<float> xg,pg,wg,yg;GlobalTensor<int64_t> eg;
    xg.SetGlobalBuffer((__gm__ float*)x);pg.SetGlobalBuffer((__gm__ float*)pseudo);
    wg.SetGlobalBuffer((__gm__ float*)weight);yg.SetGlobalBuffer((__gm__ float*)out);
    eg.SetGlobalBuffer((__gm__ int64_t*)edges);
    GlobalTensor<uint64_t> edgeCache,pseudoCache;
    edgeCache.SetGlobalBuffer((__gm__ uint64_t*)edges);
    pseudoCache.SetGlobalBuffer((__gm__ uint64_t*)pseudo);
    DataCachePreload(edgeCache,int64_t(0));DataCachePreload(pseudoCache,int64_t(0));
    LocalTensor<float> weights(TPosition::VECCALC,0,512);
    LocalTensor<float> features(TPosition::VECCALC,2048,512);
    LocalTensor<float> spread(TPosition::VECCALC,4096,64);
    LocalTensor<float> dot(TPosition::VECCALC,4352,128);
    LocalTensor<float> zeros(TPosition::VECCALC,4864,512);
    LocalTensor<float> result(TPosition::VECCALC,6912,128);
    DataCopyPad(weights,wg,DataCopyExtParams{1,2048,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0f});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopy(features,xg,n*8);SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Duplicate(zeros,0.0f,n*16);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    spline_output::CopyRows(yg,zeros,0,n);
    Duplicate(result,0.0f,128);
    int64_t source=eg.GetValue(0),target=eg.GetValue(1);
    bool valid=source>=0 && source<n && target>=0 && target<n;
    uint32_t begin=valid?(uint32_t(target)/8)*8:0;
    uint32_t rows=n-begin;if(rows>8) rows=8;
    float q0=0.0f,q1=0.0f;
    if(valid) {q0=pg.GetValue(0);q1=pg.GetValue(1);}
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if(valid) SplineSingleInterpolate<Closed0,Closed1,false>(dot,weights,q0,q1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    if(!valid) return;
    Brcb(spread,features[uint32_t(source)*8],1,BrcbRepeatParams{1,8});
    Mul(dot,dot,spread,uint64_t(16),uint8_t(8),BinaryRepeatParams{1,1,0,2,2,1});
    Add(dot,dot,dot[64],64);Add(dot,dot,dot[32],32);
    if(rows==8) Add(result[(uint32_t(target)-begin)*16],dot,dot[16],16);
    else Add(dot,dot,dot[16],16);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    PipeBarrier<PIPE_MTE3>();
    if(rows==8) spline_output::CopyRows(yg,result,begin,rows);
    else spline_output::CopyRows(yg,dot,uint32_t(target),1);
}
