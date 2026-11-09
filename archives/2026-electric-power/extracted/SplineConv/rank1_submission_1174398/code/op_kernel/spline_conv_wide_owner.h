#pragma once
#include "spline_conv_sparse_output.h"
#include "spline_conv_clear_mask.h"
#include "spline_conv_wide_tile.h"

template <bool Closed0,bool Closed1,bool DedicatedClear=false,bool EarlyClear=false,bool CachedMeta=false,bool Eager=false>
class SplineWideEdgeOwner {
    static constexpr uint32_t C=128;
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,uint32_t n,uint32_t e,uint32_t mean) {
        nodes=n;edgeCount=e;average=mean;
        xg.SetGlobalBuffer((__gm__ float*)x);eg.SetGlobalBuffer((__gm__ int64_t*)edges);
        pg.SetGlobalBuffer((__gm__ float*)pseudo);wg.SetGlobalBuffer((__gm__ float*)weight);yg.SetGlobalBuffer((__gm__ float*)out);
        el=LocalTensor<int64_t>(TPosition::VECCALC,0,128);
        pr=LocalTensor<float>(TPosition::VECCALC,1024,128);
        constexpr uint32_t zeroRows=DedicatedClear?128:32;
        constexpr uint32_t weightBase=1536+zeroRows*16*4;
        zeros=LocalTensor<float>(TPosition::VECCALC,1536,zeroRows*16);
        weights=LocalTensor<float>(TPosition::VECCALC,weightBase,4*C*16);
        features=LocalTensor<float>(TPosition::VECCALC,weightBase+4*C*16*4,64*C);
        agg=LocalTensor<float>(TPosition::VECCALC,weightBase+4*C*16*4+64*C*4,4*C);
        spread=LocalTensor<float>(TPosition::VECCALC,weightBase+4*C*16*4+64*C*4+4*C*4,4*C*8);
        dot=LocalTensor<float>(TPosition::VECCALC,weightBase+4*C*16*4+64*C*4+4*C*4+4*C*8*4+256,4*C*16);
    }
    __aicore__ inline void PrefetchGraph() {
        if constexpr(CachedMeta) {
            GlobalTensor<uint64_t> graphCache,pseudoCache;
            graphCache.SetGlobalBuffer((__gm__ uint64_t*)eg.GetPhyAddr());
            pseudoCache.SetGlobalBuffer((__gm__ uint64_t*)pg.GetPhyAddr());
            DataCachePreload(graphCache,int64_t(0));DataCachePreload(pseudoCache,int64_t(0));
            SplineWideTile::Prefetch(weights,wg);
            if(edgeCount>4) DataCachePreload(graphCache,int64_t(64));
        } else {
            DataCopyPadExtParams<int64_t> epad{false,0,0,0};DataCopyPadExtParams<float> pad{false,0,0,0.0f};
            DataCopyPad(el,eg,DataCopyExtParams{1,edgeCount*16,0,0,0},epad);
            DataCopyPad(pr,pg,DataCopyExtParams{1,edgeCount*8,0,0,0},pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SplineWideTile::Prefetch(weights,wg);
        }
    }
    __aicore__ inline void SetShape(uint32_t n,uint32_t mean) {nodes=n;average=mean;}
    __aicore__ inline void PrefetchClearGraph() {
        DataCopyPad(el,eg,DataCopyExtParams{1,edgeCount*16,0,0,0},
            DataCopyPadExtParams<int64_t>{false,0,0,0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    __aicore__ inline void ClearOnly(uint32_t worker,uint32_t workers) {
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        uint32_t bound=(nodes+workers-1)/workers,begin=worker*bound;
        uint32_t end=begin+bound;if(end>nodes) end=nodes;
        if(begin>=end) return;
        uint32_t rows=end-begin,zeroRows=rows>128?128:rows;
        Duplicate(zeros,0.0f,zeroRows*16);
        uint64_t m0=0,m1=0,m2=0,m3=0;
        for(uint32_t e=0;e<edgeCount;++e) {
            int64_t dst=GraphValue(edgeCount+e);
            if(dst>=begin && dst<int64_t(end)) {
                uint32_t bit=uint32_t(dst)-begin;
                if(bit<64) m0|=uint64_t(1)<<bit;
                else if(bit<128) m1|=uint64_t(1)<<(bit-64);
                else if(bit<192) m2|=uint64_t(1)<<(bit-128);
                else m3|=uint64_t(1)<<(bit-192);
            }
        }
        SplineClearMask256(yg,zeros,begin,rows,m0,m1,m2,m3);
    }
    __aicore__ inline void Process(GM_ADDR attributes=nullptr) {
        uint32_t block=GetBlockIdx(),blocks=GetBlockNum();
        uint32_t zeroBegin=0,zeroRows=0,occupied=0;
        if constexpr(EarlyClear && !DedicatedClear) {
            uint32_t bound=(nodes+blocks-1)/blocks;zeroBegin=block*bound;
            zeroRows=zeroBegin<nodes?nodes-zeroBegin:0;if(zeroRows>bound) zeroRows=bound;
            if(zeroRows) {
                Duplicate(zeros,0.0f,zeroRows*16);
                if constexpr(Eager) SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
        }
        if constexpr(!CachedMeta) WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        int64_t src=GraphValue(block);
        if(src>=0 && src<nodes) DataCopy(features[block*C],xg[uint32_t(src)*C],C);
        int64_t target=block<edgeCount?GraphValue(edgeCount+block):-1;
        bool owner=target>=0 && target<nodes;
        bool prepared=false;
        if constexpr(Eager) {
            if(owner && src>=0 && src<nodes) {
                float f[4];uint32_t wi;Factors(block,f,wi);
                SplineWideTile::Interpolate(dot,weights,f[0],f[1],f[2],f[3]);
                prepared=true;
            }
        }
        uint32_t degree=0;
        for(uint32_t e=0;e<edgeCount;++e) {
            int64_t dst=GraphValue(edgeCount+e);
            if constexpr(EarlyClear && !DedicatedClear) {
                if(dst>=zeroBegin && dst<int64_t(zeroBegin+zeroRows)) occupied|=1u<<(uint32_t(dst)-zeroBegin);
            }
            if(dst==target) {
                if(e<block) owner=false;
                ++degree;
                if constexpr(C==128) {
                    if(e!=block) {
                        int64_t src=GraphValue(e);
                        if(src>=0 && src<nodes) DataCopy(features[e*C],xg[uint32_t(src)*C],C);
                    }
                }
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
        if constexpr(EarlyClear && !DedicatedClear) {
            if(zeroRows) spline_output::ClearInactiveRows<float,Eager>(yg,zeros,zeroBegin,zeroRows,occupied);
        }
        if(!owner) {
            if(!prepared) WaitWeights();WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
            if constexpr(C==128 && !DedicatedClear && !EarlyClear) {
                Clear(block,blocks);
            }
            return;
        }
        if(degree==1) Single(block,prepared);
        else {
            if(!prepared) WaitWeights();WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
            Duplicate(agg,0.0f,4*C);PipeBarrier<PIPE_ALL>();uint32_t valid=0;
            for(uint32_t e=0;e<edgeCount;++e) {
                if(GraphValue(edgeCount+e)!=target) continue;
                int64_t src=GraphValue(e);if(src<0 || src>=nodes) continue;
                float f[4];uint32_t wi;Factors(e,f,wi);
                for(uint32_t slot=0;slot<4;++slot) {
                    uint32_t index=wi^((slot&1)+2*(slot>>1));
                    Axpy(agg[index*C],features[e*C],f[slot],C);
                }
                ++valid;
            }
            if(!valid) {
                Duplicate(dot,0.0f,16);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                spline_output::CopyRows(yg,dot,uint32_t(target),1);
                return;
            }
            SplineWideTile::GroupDot(dot,weights,agg,spread);
            uint32_t mean=average;
            if(attributes && valid!=1) mean=((const volatile __gm__ SplineConvTilingData*)attributes)->mean;
            if(mean && valid!=1) Muls(dot,dot,1.0f/valid,16);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        spline_output::CopyRows(yg,dot,uint32_t(target),1);
        if constexpr(C==128 && !DedicatedClear && !EarlyClear) {
            Clear(block,blocks);
        }
    }
private:
    __aicore__ inline int64_t GraphValue(uint32_t i) {
        if constexpr(CachedMeta) return eg.GetValue(i);
        else return el.GetValue(i);
    }
    __aicore__ inline float PseudoValue(uint32_t i) {
        if constexpr(CachedMeta) return pg.GetValue(i);
        else return pr.GetValue(i);
    }
    __aicore__ inline void Clear(uint32_t block,uint32_t blocks) {
        uint32_t bound=(nodes+blocks-1)/blocks,begin=block*bound;
        uint32_t rows=begin<nodes?nodes-begin:0;if(rows>bound) rows=bound;
        if(!rows) return;
        Duplicate(zeros,0.0f,rows*16);
        uint32_t occupied=0;
        for(uint32_t e=0;e<edgeCount;++e) {
            int64_t dst=GraphValue(edgeCount+e);
            if(dst>=begin && dst<int64_t(begin+rows)) occupied|=1u<<(uint32_t(dst)-begin);
        }
        spline_output::ClearInactiveRows(yg,zeros,begin,rows,occupied);
    }
    __aicore__ inline void WaitWeights() {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    }
    __aicore__ inline void Factors(uint32_t e,float *f,uint32_t &wi) {
        float q0=PseudoValue(e*2),q1=PseudoValue(e*2+1);
        if constexpr(Closed0) {q0*=2.0f;q0=q0>1.0f?2.0f-q0:q0;}
        if constexpr(Closed1) {q1*=2.0f;q1=q1>1.0f?2.0f-q1:q1;}
        f[0]=(1-q0)*(1-q1);f[1]=q0*(1-q1);f[2]=(1-q0)*q1;f[3]=q0*q1;
        wi=0;
    }
    __aicore__ inline void Single(uint32_t e,bool prepared=false) {
        int64_t src=GraphValue(e);
        if(src<0 || src>=nodes) {WaitWeights();WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);Duplicate(dot,0.0f,16);return;}
        if(!prepared) {
            float q0=PseudoValue(e*2),q1=PseudoValue(e*2+1);
            if constexpr(Closed0) {q0*=2.0f;q0=q0>1.0f?2.0f-q0:q0;}
            if constexpr(Closed1) {q1*=2.0f;q1=q1>1.0f?2.0f-q1:q1;}
            float f0=(1-q0)*(1-q1),f1=q0*(1-q1),f2=(1-q0)*q1,f3=q0*q1;
            SplineWideTile::Interpolate(dot,weights,f0,f1,f2,f3);
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        SplineWideTile::FeatureDot(dot,features[e*C],spread);
    }
    uint32_t nodes,edgeCount,average;
    GlobalTensor<float> xg,pg,wg,yg;GlobalTensor<int64_t> eg;
    LocalTensor<int64_t> el;LocalTensor<float> pr,zeros,weights,features,agg,spread,dot;
};

template <bool Closed0,bool Closed1>
__aicore__ inline void RunWideEarlyOwner(GM_ADDR x,GM_ADDR edges,
        GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,GM_ADDR tiling) {
    const __gm__ SplineConvTilingData* td=(const __gm__ SplineConvTilingData*)tiling;
    SplineWideEdgeOwner<Closed0,Closed1,false,true,false,true> op;
    op.Init(x,edges,pseudo,weight,out,0,GetBlockNum(),0);
    op.PrefetchGraph();op.SetShape(td->n,td->mean);op.Process();
}
