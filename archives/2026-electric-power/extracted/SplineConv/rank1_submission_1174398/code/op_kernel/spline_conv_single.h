#pragma once
#include "spline_conv_single_interpolate.h"

template <typename T,bool Raw=false,bool One=false,bool Closed0=false,bool Closed1=false,bool Chunk=false,bool Tail=false>
__aicore__ inline bool RunSingleStatic(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,const SplineConvTilingData &td,GM_ADDR tiling=nullptr) {
    constexpr bool Narrow=Raw && Chunk && sizeof(T)==4;
    if constexpr(Chunk) {
        uint32_t active=(GetBlockNum()+15)/16;
        if constexpr(Tail) {
            uint32_t aligned=(GetBlockNum()-1)/(128/(16*sizeof(T)))*(128/(16*sizeof(T)));
            active=(aligned+15)/16+1;
        }
        if(GetBlockIdx()>=active) return true;
    }
    SplineConvTilingData t;
    if constexpr(!Raw) {
        t=td;
        if(t.ci!=8 || t.co!=16 || t.e!=1 || t.nodesPerBlock>8 ||
           t.streaming || t.hasRoot || t.hasBias || t.dim!=2 || t.degree!=1 || t.ks[0]!=2 || t.ks[1]!=2) return false;
    }
    constexpr uint32_t WO=64,WF=sizeof(T)==2?WO+512*sizeof(T):WO;
    constexpr uint32_t XO=WF+512*4,XF=sizeof(T)==2?XO+32:XO;
    constexpr uint32_t AO=XF+32,BO=AO+128,DO=BO+1024,YO=DO+2048,END=YO+(Chunk?16:8)*16*sizeof(T);
    LocalTensor<int64_t> el(TPosition::VECCALC,0,4);LocalTensor<T> pr(TPosition::VECCALC,32,32/sizeof(T));
    LocalTensor<T> wr(TPosition::VECCALC,WO,512);LocalTensor<float> weights(TPosition::VECCALC,WF,512);
    LocalTensor<T> xr(TPosition::VECCALC,XO,32/sizeof(T));LocalTensor<float> xv(TPosition::VECCALC,XF,8);
    LocalTensor<float> agg(TPosition::VECCALC,AO,32),broadcast(TPosition::VECCALC,BO,256);
    LocalTensor<float> dot(TPosition::VECCALC,DO,512);LocalTensor<T> yr(TPosition::VECCALC,YO,Chunk?256:128);
    GlobalTensor<int64_t> eg;eg.SetGlobalBuffer((__gm__ int64_t*)edges);
    GlobalTensor<T> xg,pg,wg,yg;xg.SetGlobalBuffer((__gm__ T*)x);pg.SetGlobalBuffer((__gm__ T*)pseudo);
    wg.SetGlobalBuffer((__gm__ T*)weight);yg.SetGlobalBuffer((__gm__ T*)out);
    DataCopyExtParams ep{1,16,0,0,0};DataCopyPadExtParams<int64_t> ez{false,0,0,0};DataCopyPad(el,eg,ep,ez);
    if constexpr(Tail) {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        DataCopyPadExtParams<T> pad{false,0,0,{}};
        DataCopyExtParams wp{1,512*uint32_t(sizeof(T)),0,0,0};DataCopyPad(wr,wg,wp,pad);
        DataCopyExtParams pp{1,2*uint32_t(sizeof(T)),0,0,0};DataCopyPad(pr,pg,pp,pad);
    }
    if constexpr(Raw && One) {
        t.n=GetBlockNum();t.blocks=t.n;t.scale[0]=Closed0?2:1;t.scale[1]=Closed1?2:1;
    } else if constexpr(Raw && Chunk) {
        t.n=GetBlockNum()-(Tail?1:0);t.blocks=t.n;t.scale[0]=Closed0?2:1;t.scale[1]=Closed1?2:1;
    } else if constexpr(Raw) {
        const volatile __gm__ SplineConvTilingData *raw=(const volatile __gm__ SplineConvTilingData*)tiling;
        t.n=raw->n;t.blocks=raw->blocks;t.scale[0]=raw->scale[0];t.scale[1]=raw->scale[1];
    }
    uint32_t start=Chunk?GetBlockIdx()*16:GetBlockIdx(),step=Chunk?1:t.blocks;
    bool oneRow=One || (!Chunk && t.n==step);
    uint32_t rows=oneRow?1:(t.n-start+step-1)/step;
    if constexpr(Chunk) {
        uint32_t limit=t.n;
        if constexpr(Tail) {
            uint32_t aligned=t.n/(128/(16*sizeof(T)))*(128/(16*sizeof(T)));
            if(GetBlockIdx()==(aligned+15)/16) start=aligned;
            else limit=aligned;
        }
        rows=limit-start;if(rows>16) rows=16;
    }
    uint32_t end=Chunk?start+rows:t.n;
    Duplicate(yr.template ReinterpretCast<int32_t>(),int32_t(0),rows*16*sizeof(T)/4);
    if constexpr(Tail) WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    else if constexpr(Narrow) {SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);}
    else PipeBarrier<PIPE_ALL>();
    int64_t src=el.GetValue(0),target=el.GetValue(1);
    bool owns=oneRow?target==int64_t(start):(target>=start && target<end && (uint32_t(target)-start)%step==0);
    if(owns && src>=0 && src<t.n) {
        DataCopyPadExtParams<T> pad{false,0,0,{}};
        if constexpr(!Tail) {
            DataCopyExtParams wp{1,512*uint32_t(sizeof(T)),0,0,0};DataCopyPad(wr,wg,wp,pad);
        }
        DataCopyExtParams xp{1,8*uint32_t(sizeof(T)),0,0,0};DataCopyPad(xr,xg[uint32_t(src)*8],xp,pad);
        if constexpr(!Tail) {
            DataCopyExtParams pp{1,2*uint32_t(sizeof(T)),0,0,0};DataCopyPad(pr,pg,pp,pad);
        }
        if constexpr(Narrow) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        } else PipeBarrier<PIPE_ALL>();
        float p0,p1;
        if constexpr(sizeof(T)==2) {
            Cast(weights,wr,RoundMode::CAST_NONE,512);Cast(xv,xr,RoundMode::CAST_NONE,8);
            auto pb=pr.template ReinterpretCast<uint16_t>();union {uint32_t u;float f;} b;
            b.u=uint32_t(pb.GetValue(0))<<16;p0=b.f;b.u=uint32_t(pb.GetValue(1))<<16;p1=b.f;
        } else {p0=pr.GetValue(0);p1=pr.GetValue(1);}
        if constexpr(Narrow) WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        if constexpr(Raw && (One || Chunk)) {
            SplineSingleInterpolate<Closed0,Closed1>(dot,weights,p0,p1);
        } else {
        p0*=int32_t(t.scale[0]);p1*=int32_t(t.scale[1]);int32_t b0=int32_t(p0),b1=int32_t(p1);
        p0-=float(b0);p1-=float(b1);
        float factors[4]={(1-p0)*(1-p1),p0*(1-p1),(1-p0)*p1,p0*p1};PipeBarrier<PIPE_V>();
        uint32_t wi=(b0&1)+2*(b1&1);
        Muls(dot,weights[wi*128],factors[0],128);PipeBarrier<PIPE_V>();
        for(uint32_t slot=1;slot<4;++slot) {
            wi=((b0+(slot&1))&1)+2*((b1+(slot>>1))&1);
            Axpy(dot,weights[wi*128],factors[slot],128);PipeBarrier<PIPE_V>();
        }
        }
        Brcb(broadcast,xv,1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        Mul(dot,dot,broadcast,uint64_t(16),uint8_t(8),BinaryRepeatParams{1,1,0,2,2,1});PipeBarrier<PIPE_V>();
        for(uint32_t rows=8;rows>1;rows/=2) {Add(dot,dot,dot[rows/2*16],rows/2*16);PipeBarrier<PIPE_V>();}
        uint32_t row=oneRow?0:(uint32_t(target)-start)/step;
        if constexpr(sizeof(T)==2) Cast(yr[row*16],dot,RoundMode::CAST_RINT,16);else DataCopy(yr[row*16],dot,16);
    }
    if constexpr(Narrow) {SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);}
    else PipeBarrier<PIPE_ALL>();
    if((oneRow || Chunk) && (uint64_t(yg.GetPhyAddr())&31)==0) DataCopy(yg[start*16],yr,rows*16);
    else {
        DataCopyExtParams yp{uint16_t(rows),16*uint32_t(sizeof(T)),0,(step-1)*16*uint32_t(sizeof(T)),0};
        DataCopyPad(yg[start*16],yr,yp);
    }
    if constexpr(!Narrow) PipeBarrier<PIPE_ALL>();
    return true;
}
