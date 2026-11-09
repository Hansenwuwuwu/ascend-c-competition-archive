#pragma once
#include "spline_conv_edge_owner.h"
#include "spline_conv_four_group.h"

template <uint32_t Nodes,bool Bulk,bool Closed0,bool Closed1>
__aicore__ inline void SplineFourOwnerFallback(GM_ADDR x,GM_ADDR edges,
        GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,GM_ADDR tiling) {
    SplineFourGroupDot<Nodes,Closed0,Closed1>(x,edges,pseudo,out,tiling);
}

template <uint32_t Nodes,bool Bulk,bool Closed0,bool Closed1>
__aicore__ inline void RunFourEdgeOwner(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,
        GM_ADDR weight,GM_ADDR out,GM_ADDR tiling) {
    uint32_t block=GetBlockIdx();
    block=(block>>1)|((block&1)<<2);
    GlobalTensor<float> xg,pg,wg,yg;GlobalTensor<int64_t> eg;
    xg.SetGlobalBuffer((__gm__ float*)x);pg.SetGlobalBuffer((__gm__ float*)pseudo);
    wg.SetGlobalBuffer((__gm__ float*)weight);yg.SetGlobalBuffer((__gm__ float*)out);
    eg.SetGlobalBuffer((__gm__ int64_t*)edges);
    LocalTensor<int64_t> graph(TPosition::VECCALC,0,8);
    LocalTensor<float> zeros(TPosition::VECCALC,128,512);
    if(block>=4) {
        SplineClearOwner<4,8>(eg,yg,Nodes,block,graph,zeros);return;
    }
    LocalTensor<float> weights(TPosition::VECCALC,2304,2048);
    LocalTensor<float> features(TPosition::VECCALC,10496,2048);
    LocalTensor<float> spread(TPosition::VECCALC,24832,512);
    LocalTensor<float> reduced(TPosition::VECCALC,32768,512);
    LocalTensor<float> dot(TPosition::VECCALC,19712,512);
    GlobalTensor<uint64_t> edgeCache,pseudoCache;
    edgeCache.SetGlobalBuffer((__gm__ uint64_t*)edges);
    pseudoCache.SetGlobalBuffer((__gm__ uint64_t*)pseudo);
    DataCachePreload(edgeCache,int64_t(0));DataCachePreload(pseudoCache,int64_t(0));
    DataCopy(weights,wg,512);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopy(weights[512],wg[512],512);SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    DataCopy(weights[1024],wg[1024],512);SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    DataCopy(weights[1536],wg[1536],512);SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
    if constexpr(Bulk) DataCopy(features,xg,Nodes*32);
    int64_t source=eg.GetValue(block),target=eg.GetValue(4+block);
    bool validSource=source>=0 && source<Nodes;
    if constexpr(!Bulk) {
        if(validSource) DataCopy(features,xg[uint32_t(source)*32],32);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
    uint32_t bits=uint32_t(eg.GetValue(4)==target) | (uint32_t(eg.GetValue(5)==target)<<1) |
        (uint32_t(eg.GetValue(6)==target)<<2) | (uint32_t(eg.GetValue(7)==target)<<3);
    if((bits&(bits-1))!=0) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        PipeBarrier<PIPE_ALL>();
        SplineFourOwnerFallback<Nodes,Bulk,Closed0,Closed1>(x,edges,pseudo,weight,out,tiling);return;
    }
    if(target<0 || target>=Nodes) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);return;
    }
    float q0=pg.GetValue(block*2),q1=pg.GetValue(block*2+1);
    if constexpr(Closed0) {q0*=2.0f;q0=q0>1.0f?2.0f-q0:q0;}
    if constexpr(Closed1) {q1*=2.0f;q1=q1>1.0f?2.0f-q1:q1;}
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if(validSource) {
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(64);
        Muls<float,false>(dot,weights,(1.0f-q0)*(1.0f-q1),uint64_t(64),uint8_t(8),UnaryRepeatParams{1,1,8,8});
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Axpy<float,float,false>(dot,weights[512],q0*(1.0f-q1),uint64_t(64),uint8_t(8),UnaryRepeatParams{1,1,8,8});
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        Axpy<float,float,false>(dot,weights[1024],(1.0f-q0)*q1,uint64_t(64),uint8_t(8),UnaryRepeatParams{1,1,8,8});
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        Axpy<float,float,false>(dot,weights[1536],q0*q1,uint64_t(64),uint8_t(8),UnaryRepeatParams{1,1,8,8});
    } else {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    }
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
    if(validSource) {
        uint32_t offset=Bulk?uint32_t(source)*32:0;
        Brcb(spread,features[offset],4,BrcbRepeatParams{2,16});
        Brcb(spread[8],features[offset],4,BrcbRepeatParams{2,16});
        Mul<float,false>(dot,dot,spread,uint64_t(64),uint8_t(8),BinaryRepeatParams{1,1,1,8,8,8});
        Add<float,false>(dot,dot[64],dot,uint64_t(64),uint8_t(7),BinaryRepeatParams{1,1,1,0,8,0});
        Add<float,false>(reduced,dot,dot[32],uint64_t(64),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});
        Add<float,false>(dot,reduced,reduced[16],uint64_t(64),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});
    } else Duplicate(dot,0.0f,16);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    spline_output::CopyRows(yg,dot,uint32_t(target),1);
}

template <uint32_t Pad,bool Closed0,bool Closed1>
__aicore__ inline void RunPartialEdgeOwner(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,
        GM_ADDR weight,GM_ADDR out,GM_ADDR tiling) {
    RunFourEdgeOwner<64-Pad,true,Closed0,Closed1>(x,edges,pseudo,weight,out,tiling);
}
