#pragma once
#include "spline_conv_single_interpolate.h"
#include "spline_conv_sparse_output.h"
#include "spline_conv_single_tail_sector.h"

template <bool Closed0,bool Closed1,bool LinearReduce=false,bool ExplicitBarrier=!LinearReduce>
__aicore__ inline void RunSingleClearAhead(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,
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
    DataCopyPad(weights,wg,DataCopyExtParams{1,2048,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0f});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopy(features,xg,n*8);SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Duplicate(zeros,0.0f,n*16);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    spline_output::CopyRows(yg,zeros,0,n);
    int64_t source=eg.GetValue(0),target=eg.GetValue(1);
    bool valid=source>=0 && source<n && target>=0 && target<n;
    float q0=0.0f,q1=0.0f;
    if(valid) {q0=pg.GetValue(0);q1=pg.GetValue(1);}
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if(valid) SplineSingleInterpolate<Closed0,Closed1,ExplicitBarrier>(dot,weights,q0,q1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    if(!valid) return;
    Brcb(spread,features[uint32_t(source)*8],1,BrcbRepeatParams{1,8});
    Mul(dot,dot,spread,uint64_t(16),uint8_t(8),BinaryRepeatParams{1,1,0,2,2,1});
    if constexpr(LinearReduce)
        Add(dot,dot[16],dot,uint64_t(16),uint8_t(7),BinaryRepeatParams{1,1,1,0,2,0});
    else {Add(dot,dot,dot[64],64);Add(dot,dot,dot[32],32);Add(dot,dot,dot[16],16);}
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    PipeBarrier<PIPE_MTE3>();
    spline_output::CopyRows(yg,dot,uint32_t(target),1);
}

template <bool Closed0,bool Closed1,bool Tail>
__aicore__ inline void RunSingleBulkStaged(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out) {
    uint32_t n=2*GetBlockNum()-(Tail?1:0),block=GetBlockIdx();
    if(n<=32) {RunSingleClearAhead<Closed0,Closed1,!Tail>(x,edges,pseudo,weight,out,n);return;}
    uint32_t aligned=n/2*2,active=(n+15)/16;
    if constexpr(Tail) active=(aligned+15)/16+1;
    if(block>=active) return;
    GlobalTensor<float> xg,pg,wg,yg;GlobalTensor<int64_t> eg;
    xg.SetGlobalBuffer((__gm__ float*)x);pg.SetGlobalBuffer((__gm__ float*)pseudo);
    wg.SetGlobalBuffer((__gm__ float*)weight);yg.SetGlobalBuffer((__gm__ float*)out);
    eg.SetGlobalBuffer((__gm__ int64_t*)edges);
    GlobalTensor<uint64_t> edgeCache,pseudoCache;
    edgeCache.SetGlobalBuffer((__gm__ uint64_t*)edges);
    pseudoCache.SetGlobalBuffer((__gm__ uint64_t*)pseudo);
    DataCachePreload(edgeCache,int64_t(0));
    DataCachePreload(pseudoCache,int64_t(0));
    LocalTensor<float> weights(TPosition::VECCALC,0,512);
    LocalTensor<float> features(TPosition::VECCALC,2048,512);
    LocalTensor<float> spread(TPosition::VECCALC,4096,64);
    LocalTensor<float> dot(TPosition::VECCALC,4352,128);
    LocalTensor<float> result(TPosition::VECCALC,4864,256);
    DataCopyPadExtParams<float> pad{false,0,0,0.0f};
    DataCopyPad(weights,wg,DataCopyExtParams{1,2048,0,0,0},pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopy(features,xg,n*8);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
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
        q0=pg.GetValue(0);q1=pg.GetValue(1);
    }
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if(valid) SplineSingleInterpolate<Closed0,Closed1>(dot,weights,q0,q1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    if(valid) {
        Brcb(spread,features[uint32_t(source)*8],1,BrcbRepeatParams{1,8});
        Mul(dot,dot,spread,uint64_t(16),uint8_t(8),BinaryRepeatParams{1,1,0,2,2,1});
        Add(dot,dot,dot[64],64);Add(dot,dot,dot[32],32);Add(dot,dot,dot[16],16);
        DataCopy(result[(uint32_t(target)-start)*16],dot,16);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    spline_output::CopyRows(yg,result,start,rows);
}

#include "spline_conv_single_stage.h"
#include "spline_conv_clear_mask.h"

template <bool Closed0,bool Closed1>
__aicore__ inline void RunSingleTailBulk(GM_ADDR x,GM_ADDR edges,
        GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out) {
    uint32_t blocks=GetBlockNum();
    if(blocks<2) {RunSingleStaged<Closed0,Closed1,true>(x,edges,pseudo,weight,out);return;}
    uint32_t n=2*blocks-1,block=GetBlockIdx();
    if(n<=32) {RunSingleClearAhead<Closed0,Closed1>(x,edges,pseudo,weight,out,n);return;}
    if(block>=2) return;
    GlobalTensor<uint64_t> graphCache;
    graphCache.SetGlobalBuffer((__gm__ uint64_t*)edges);
    DataCachePreload(graphCache,int64_t(0));
    GlobalTensor<int64_t> eg;
    GlobalTensor<float> yg;
    eg.SetGlobalBuffer((__gm__ int64_t*)edges);
    yg.SetGlobalBuffer((__gm__ float*)out);
    if(block==0) {
        LocalTensor<float> zeros(TPosition::VECCALC,0,64*16);
        Duplicate(zeros,0.0f,n*16);
        int64_t source=eg.GetValue(0),target=eg.GetValue(1);
        bool valid=source>=0 && source<n && target>=0 && target<n;
        uint64_t occupied=valid?(uint64_t(1)<<uint32_t(target)):0;
        SplineClearMask128(yg,zeros,0,n,occupied,0);
        return;
    }
    GlobalTensor<float> xg,pg,wg;
    GlobalTensor<uint64_t> pseudoCache;
    pseudoCache.SetGlobalBuffer((__gm__ uint64_t*)pseudo);
    DataCachePreload(pseudoCache,int64_t(0));
    xg.SetGlobalBuffer((__gm__ float*)x);
    pg.SetGlobalBuffer((__gm__ float*)pseudo);
    wg.SetGlobalBuffer((__gm__ float*)weight);
    LocalTensor<float> weights(TPosition::VECCALC,0,512);
    LocalTensor<float> features(TPosition::VECCALC,2048,512);
    LocalTensor<float> spread(TPosition::VECCALC,4096,64);
    LocalTensor<float> dot(TPosition::VECCALC,4352,128);
    DataCopyPadExtParams<float> pad{false,0,0,0.0f};
    DataCopyPad(weights,wg,DataCopyExtParams{1,2048,0,0,0},pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopy(features,xg,n*8);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    int64_t source=eg.GetValue(0),target=eg.GetValue(1);
    bool valid=source>=0 && source<n && target>=0 && target<n;
    float q0=0.0f,q1=0.0f;
    if(valid) {q0=pg.GetValue(0);q1=pg.GetValue(1);}
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if(valid) SplineSingleInterpolate<Closed0,Closed1>(dot,weights,q0,q1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    if(!valid) return;
    Brcb(spread,features[uint32_t(source)*8],1,BrcbRepeatParams{1,8});
    Mul(dot,dot,spread,uint64_t(16),uint8_t(8),BinaryRepeatParams{1,1,0,2,2,1});
    Add(dot,dot,dot[64],64);Add(dot,dot,dot[32],32);Add(dot,dot,dot[16],16);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    spline_output::CopyRows(yg,dot,uint32_t(target),1);
}
