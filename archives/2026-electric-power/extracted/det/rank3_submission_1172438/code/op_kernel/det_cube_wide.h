#pragma once
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"
namespace DetWide {
using namespace AscendC;
using namespace matmul;
using AB = MatmulType<TPosition::GM, CubeFormat::ND, float>;
class CubeLUVector {
public:
    __aicore__ inline CubeLUVector(TPipe &external):pipe(external){}
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR det,GM_ADDR ws,const LinalgDetConfig &cfg) {
        t=cfg;uint32_t logical=GetBlockIdx()%t.coresPerMatrix;sub=0;q=logical;
        group=GetBlockIdx()/t.coresPerMatrix;first=q*32;
        input.SetGlobalBuffer((__gm__ float*)A); output.SetGlobalBuffer((__gm__ float*)det);
        allShared.SetGlobalBuffer((__gm__ float*)ws);
        shared=allShared[group*t.robustOffset];
        comm=shared[t.paddedN*32];own=shared[2*(t.paddedN*32+128)+q*(t.paddedN*32+2048)];prodGm=own[1024];
        pipe.InitBuffer(panelBuf,2*t.paddedN*32*4);panel=panelBuf.Get<float>();corr=panel[t.paddedN*32];
        pipe.InitBuffer(rawBuf,t.paddedN*8*4); raw=rawBuf.Get<float>();
        pipe.InitBuffer(vecBuf,8*t.paddedN*4); col=vecBuf.Get<float>(); ab=col[t.paddedN]; tmp=ab[t.paddedN];
        row=tmp[t.paddedN]; old=row[t.paddedN]; v=old[t.paddedN]; meta=v[t.paddedN]; pivots=meta[t.paddedN].ReinterpretCast<uint32_t>();
        pipe.InitBuffer(topBuf,32*32*4);top=topBuf.Get<float>();
        pipe.InitBuffer(idxBuf,t.paddedN*4); idx=idxBuf.Get<uint32_t>();
        CreateVecIndex(idx.ReinterpretCast<int32_t>(),0,t.paddedN); PipeBarrier<PIPE_V>();
        ShiftLeft(idx,idx,uint32_t(7),t.paddedN); PipeBarrier<PIPE_V>();
    }
    template<HardEvent E> __aicore__ inline void Fence(){auto e=pipe.FetchEventID(E);SetFlag<E>(e);WaitFlag<E>(e);}
    __aicore__ inline uint32_t Min(uint32_t a,uint32_t b){return a<b?a:b;}
    __aicore__ inline float Pow2(int32_t e){union{uint32_t u;float f;}x;x.u=uint32_t(e+127)<<23;return x.f;}
    __aicore__ inline void Acc(float p,float &m,int32_t &e){
        union{uint32_t u;float f;}x;x.f=p;e+=int32_t((x.u>>23)&255)-127;
        x.u=(x.u&0x807fffffU)|(127U<<23);m*=x.f;
        x.f=m;e+=int32_t((x.u>>23)&255)-127;x.u=(x.u&0x807fffffU)|(127U<<23);m=x.f;
    }
    __aicore__ inline void Swap(uint32_t a,uint32_t b){
        if(a==b)return;
        Adds(row,panel[a*32],0.0f,32);Adds(old,panel[b*32],0.0f,32);PipeBarrier<PIPE_V>();
        Adds(panel[a*32],old,0.0f,32);Adds(panel[b*32],row,0.0f,32);PipeBarrier<PIPE_V>();
        Adds(row,corr[a*32],0.0f,32);Adds(old,corr[b*32],0.0f,32);PipeBarrier<PIPE_V>();
        Adds(corr[a*32],old,0.0f,32);Adds(corr[b*32],row,0.0f,32);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Factor(uint32_t start){
        float m=1;int32_t e=0;uint32_t local=start-first;bool risky=!(magnitude>1.e-20f&&magnitude<1.e20f);
        if(local==0) {
            Add(panel,panel,corr,t.paddedN*32);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<32;++j)Gather(corr[j*t.paddedN],panel,idx,j*4,t.paddedN);
            PipeBarrier<PIPE_V>();
        }
        auto rowIndices=panel[t.paddedN].ReinterpretCast<uint32_t>();
        CreateVecIndex(rowIndices.ReinterpretCast<int32_t>(),0,32);PipeBarrier<PIPE_V>();
        Muls(rowIndices.ReinterpretCast<int32_t>(),rowIndices.ReinterpretCast<int32_t>(),
             static_cast<int32_t>(t.paddedN*4),32);PipeBarrier<PIPE_V>();
        auto inverseMap=panel.ReinterpretCast<uint32_t>();
        CreateVecIndex(inverseMap.ReinterpretCast<int32_t>(),0,t.paddedN);
        CreateVecIndex(idx.ReinterpretCast<int32_t>(),0,t.paddedN);
        Duplicate(old,1.0f,t.paddedN);PipeBarrier<PIPE_V>();
        ShiftLeft(idx,idx,uint32_t(2),t.paddedN);
        if(start)Duplicate(old,0.0f,start);
        PipeBarrier<PIPE_V>();
        const uint32_t begin=start,rows=t.paddedN-begin;
        uint64_t maxMask[2]={0,0};for(uint32_t c=0;c<(rows+63)/64;++c)maxMask[0]|=1ULL<<(8*c);
        for(uint32_t i=0;!risky&&i<32;++i){
            uint32_t k=start+i,c=local+i;
            auto current=corr[c*t.paddedN];
            Abs(ab,current[begin],rows);PipeBarrier<PIPE_V>();
            Mul(ab,ab,old[begin],rows);PipeBarrier<PIPE_V>();
            WholeReduceMax(tmp,ab,64,rows/64,4,1,8);
            if(rows%64)WholeReduceMax(tmp[rows/64*8],ab[rows/64*64],rows%64,1,4,1,8);
            PipeBarrier<PIPE_V>();WholeReduceMax(meta,tmp,maxMask,1,1,1,8);PipeBarrier<PIPE_V>();
            uint32_t r=meta.ReinterpretCast<uint32_t>().GetValue(1);
            r=begin+r/8*64+tmp.ReinterpretCast<uint32_t>().GetValue(r+1);float p=current.GetValue(r);
            const float ap=p<0?-p:p;
            if(!(ap>=0.0625f*magnitude&&ap<1.e20f)){risky=true;break;}
            old.SetValue(r,0.0f);
            Fence<HardEvent::S_V>();
            Muls(current[begin],current[begin],-1.0f/p,rows);PipeBarrier<PIPE_V>();
            Mul(current[begin],current[begin],old[begin],rows);PipeBarrier<PIPE_V>();
            Gather(row,corr,rowIndices,r*4,32);PipeBarrier<PIPE_V>();
            Brcb(raw,row,4,{1,8});PipeBarrier<PIPE_V>();
            if(c+1<32) for(uint32_t off=start;off<t.paddedN;off+=64)
                MulAddDst(corr[(c+1)*t.paddedN+off],current[off],raw[(c+1)*8],Min(64,t.paddedN-off),32-c-1,
                          {1,1,0,static_cast<uint8_t>(t.paddedN/8),0,1});
            Acc(p,m,e);
            const uint32_t logical=inverseMap.GetValue(r);
            const uint32_t physical=idx.GetValue(k)/4;
            if(logical!=k)m=-m;
            pivots.SetValue(i,logical);
            idx.SetValue(k,r*4);idx.SetValue(logical,physical*4);
            inverseMap.SetValue(r,k);inverseMap.SetValue(physical,logical);
            PipeBarrier<PIPE_V>();
        }
        Muls(corr,corr,-1.0f,32*t.paddedN);PipeBarrier<PIPE_V>();
        for(uint32_t j=0;j<32;++j)Gather(panel[j*t.paddedN],corr[j*t.paddedN],idx,0,t.paddedN);
        PipeBarrier<PIPE_V>();Adds(corr,panel,0.0f,32*t.paddedN);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();meta.SetValue(0,m);meta.ReinterpretCast<int32_t>().SetValue(1,e);meta.SetValue(2,risky?1.0f:0.0f);
        Fence<HardEvent::S_MTE3>();Fence<HardEvent::V_MTE3>();
        DataCopy(shared,corr[local*t.paddedN],32*t.paddedN);
        DataCopy(comm,meta,8);DataCopy(comm[8],pivots.ReinterpretCast<float>(),32);Fence<HardEvent::MTE3_MTE2>();
    }
    // Fast block factorization without row pivoting.  The caller discards the
    // whole result and runs the compensated partial-pivot path when a diagonal
    // pivot is too small, so this path is only used on benign panels.
    __aicore__ inline void FactorNoPivot(uint32_t start){
        float m=1.0f;int32_t e=0;uint32_t local=start-first;
        bool risky=!(magnitude>1.e-20f&&magnitude<1.e20f);
        if(local==0){
            Add(panel,panel,corr,t.paddedN*32);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<32;++j)Gather(corr[j*t.paddedN],panel,idx,j*4,t.paddedN);
            PipeBarrier<PIPE_V>();
        }
        auto rowIndices=panel[t.paddedN].ReinterpretCast<uint32_t>();
        CreateVecIndex(rowIndices.ReinterpretCast<int32_t>(),0,32);PipeBarrier<PIPE_V>();
        Muls(rowIndices.ReinterpretCast<int32_t>(),rowIndices.ReinterpretCast<int32_t>(),
             static_cast<int32_t>(t.paddedN*4),32);PipeBarrier<PIPE_V>();
        const uint32_t begin=start,rows=t.paddedN-begin;
        for(uint32_t i=0;!risky&&i<32;++i){
            const uint32_t k=start+i,c=local+i;
            auto current=corr[c*t.paddedN];
            const float p=current.GetValue(k);
            const float ap=p<0?-p:p;
            if(!(ap>=0.0001f*magnitude&&ap<1.e20f)){risky=true;break;}
            Muls(current[begin],current[begin],-1.0f/p,rows);PipeBarrier<PIPE_V>();
            Duplicate(current[begin],0.0f,k+1-begin);PipeBarrier<PIPE_V>();
            Gather(row,corr,rowIndices,k*4,32);PipeBarrier<PIPE_V>();
            Brcb(raw,row,4,{1,8});PipeBarrier<PIPE_V>();
            if(c+1<32)for(uint32_t off=start;off<t.paddedN;off+=64)
                MulAddDst(corr[(c+1)*t.paddedN+off],current[off],raw[(c+1)*8],Min(64,t.paddedN-off),32-c-1,
                          {1,1,0,static_cast<uint8_t>(t.paddedN/8),0,1});
            Acc(p,m,e);PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_S>();meta.SetValue(0,m);meta.ReinterpretCast<int32_t>().SetValue(1,e);meta.SetValue(2,risky?1.0f:0.0f);
        Fence<HardEvent::S_MTE3>();Fence<HardEvent::V_MTE3>();
        DataCopy(shared,corr[local*t.paddedN],32*t.paddedN);
        DataCopy(comm,meta,8);Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void Solve(uint32_t start){
        DataCopyExtParams topCopy{32,128,(t.paddedN-32)*4,0,0};
        DataCopyPadExtParams<float> topPad{false,0,0,0};
        DataCopyPad(top,shared[start],topCopy,topPad);DataCopy(pivots.ReinterpretCast<float>(),comm[8],32);
        Fence<HardEvent::MTE2_S>();Fence<HardEvent::MTE2_V>();
        for(uint32_t i=0;i<32;++i)Swap(start+i,pivots.GetValue(i));

        for(uint32_t i=0;i<32;++i){
            Adds(col,top[i*32],0.0f,32);PipeBarrier<PIPE_V>();Duplicate(col,0.0f,i+1);PipeBarrier<PIPE_V>();
            Muls(col,col,-1.0f,32);PipeBarrier<PIPE_V>();Brcb(raw,col,4,{1,8});
            Adds(row,panel[(start+i)*32],0.0f,32);PipeBarrier<PIPE_V>();
            MulAddDst(panel[start*32],row,raw,32,32,{1,1,0,4,0,1});PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_MTE3>();DataCopy(own,panel[start*32],1024);Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void SolveNoPivot(uint32_t start){
        DataCopyExtParams topCopy{32,128,(t.paddedN-32)*4,0,0};
        DataCopyPadExtParams<float> topPad{false,0,0,0};
        DataCopyPad(top,shared[start],topCopy,topPad);Fence<HardEvent::MTE2_V>();
        for(uint32_t i=0;i<32;++i){
            Brcb(raw,top[i*32],4,{1,8});
            if(i+1<32)MulAddDst(panel[(start+i+1)*32],panel[(start+i)*32],raw[(i+1)*8],32,31-i,
                                {1,1,0,4,0,1});
            PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_MTE3>();DataCopy(own,panel[start*32],1024);Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void Update(uint32_t start){
        for(uint32_t r=start+32;r<t.paddedN;r+=t.paddedN/4){
            uint32_t len=Min(t.paddedN/4,t.paddedN-r)*32;
            DataCopy(raw,prodGm[(r-start-32)*32],len);Fence<HardEvent::MTE2_V>();
            auto hi=panel[r*32];auto lo=corr[r*32];
            Add(raw,raw,lo,len);PipeBarrier<PIPE_V>();
            Add(lo,hi,raw,len);PipeBarrier<PIPE_V>();
            Sub(hi,lo,hi,len);PipeBarrier<PIPE_V>();
            Sub(raw,hi,raw,len);PipeBarrier<PIPE_V>();
            Adds(hi,lo,0.0f,len);PipeBarrier<PIPE_V>();
            Muls(lo,raw,-1.0f,len);Fence<HardEvent::V_MTE2>();
        }
    }
    __aicore__ inline bool ProcessOne(uint32_t batch){
            CreateVecIndex(idx.ReinterpretCast<int32_t>(),0,t.paddedN);PipeBarrier<PIPE_V>();
            ShiftLeft(idx,idx,uint32_t(7),t.paddedN);PipeBarrier<PIPE_V>();
            Duplicate(panel,0.0f,t.paddedN*32);Duplicate(corr,0.0f,t.paddedN*32);Fence<HardEvent::V_S>();
            for(uint32_t r=batch<t.batch?t.n:0;r<t.paddedN;++r)if(sub==0&&r>=first&&r<first+32)panel.SetValue(r*32+r-first,1.0f);
            Fence<HardEvent::S_MTE2>();
            if(sub==0&&first<t.n&&batch<t.batch){
                uint32_t width=Min(32,t.n-first),aligned=(width+7)/8*8;
                DataCopyExtParams cp{uint16_t(t.n),width*4,(t.n-width)*4,(32-aligned)/8,0};
                DataCopyPadExtParams<float> pad{true,0,uint8_t(aligned-width),0};
                DataCopyPad(panel,input[uint64_t(batch)*t.n*t.n+first],cp,pad);Fence<HardEvent::MTE2_V>();
            }
            magnitude=1.0f;
            float m=1;int32_t e=0;bool groupSafe=true;
            for(uint32_t start=0;start<t.paddedN;start+=32){
                uint32_t owner=start/32;
                shared=allShared[group*t.robustOffset+(owner%2)*(t.paddedN*32+128)];comm=shared[t.paddedN*32];
                if(sub==0&&q==owner){
                    if(groupSafe)FactorNoPivot(start);
                    else {Fence<HardEvent::V_S>();meta.SetValue(0,1.0f);meta.ReinterpretCast<int32_t>().SetValue(1,0);meta.SetValue(2,1.0f);Fence<HardEvent::S_MTE3>();DataCopy(comm,meta,8);Fence<HardEvent::MTE3_MTE2>();}
                }
                SyncAll<true>();
                DataCopy(meta,comm,8);Fence<HardEvent::MTE2_S>();
                const bool safe=meta.GetValue(2)==0.0f;groupSafe=groupSafe&&safe;
                if(sub==0&&q==0){e+=meta.ReinterpretCast<int32_t>().GetValue(1);Acc(meta.GetValue(0),m,e);}
                if(sub==0&&safe&&q>owner)SolveNoPivot(start);
                CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);
                if(sub==0&&safe&&q>owner)Update(start);
            }
            const uint32_t finalOffset=2*(t.paddedN*32+128)-8;
            if(sub==0&&q==0){Fence<HardEvent::V_S>();meta.SetValue(2,groupSafe?0.0f:1.0f);Fence<HardEvent::S_MTE3>();DataCopy(allShared[group*t.robustOffset+finalOffset],meta,8);Fence<HardEvent::MTE3_MTE2>();}
            SyncAll<true>();
            bool allSafe=true;
            for(uint32_t g=0;g<t.groups;++g){DataCopy(meta,allShared[g*t.robustOffset+finalOffset],8);Fence<HardEvent::MTE2_S>();allSafe=allSafe&&meta.GetValue(2)==0.0f;}
            SyncAll<true>();
            if(!allSafe)return false;
            if(sub==0&&q==0&&batch<t.batch){
                while(e>120){m*=Pow2(120);e-=120;}while(e< -120){m*=Pow2(-120);e+=120;}
                Fence<HardEvent::V_S>();meta.SetValue(0,m*Pow2(e));Fence<HardEvent::S_MTE3>();
                DataCopyExtParams cp{1,4,0,0,0};DataCopyPad(output[batch],meta,cp);Fence<HardEvent::MTE3_S>();
            }
        return true;
    }
private:
    TPipe &pipe;TBuf<TPosition::VECCALC> panelBuf,rawBuf,vecBuf,topBuf,idxBuf;
    LocalTensor<float> panel,corr,raw,col,ab,tmp,row,old,v,meta,top;
    LocalTensor<uint32_t> idx,pivots;
    GlobalTensor<float> input,output,shared,comm,own,prodGm,allShared;
    LinalgDetConfig t;uint32_t q,sub,first,group;float magnitude;
};
__aicore__ inline void RunLUCube(const LinalgDetTilingData &t, GM_ADDR ws){
    if ASCEND_IS_AIC {
        TPipe pipe;MatmulImpl<AB,AB,AB,AB> mm;mm.SetSubBlockIdx(0);mm.Init(&t.cube,&pipe);mm.SetHF32(false);
        GlobalTensor<float> shared,own,prod,risk;
        uint32_t q=GetBlockIdx()%t.coresPerMatrix, group=GetBlockIdx()/t.coresPerMatrix;
        shared.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset);
        own=shared[2*(t.paddedN*32+128)+q*(t.paddedN*32+2048)];prod=own[1024];risk=shared[t.paddedN*32+2];
        for(uint32_t batch=0;batch<(t.batch+t.groups-1)/t.groups;++batch)for(uint32_t start=0;start<t.paddedN;start+=32){
            shared.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+(start/32%2)*(t.paddedN*32+128));risk=shared[t.paddedN*32+2];
            CrossCoreWaitFlag(8);
            DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE>(risk);
            const bool safe=risk.GetValue(0)==0.0f;
            if(safe&&q>start/32){
                mm.SetTail(t.paddedN-start-32,32,32);mm.SetTensorA(shared[start+32],true);mm.SetTensorB(own);
                mm.IterateAll(prod);mm.End();
            }
            CrossCoreSetFlag<2,PIPE_FIX>(9);
        }
    }
}

}
