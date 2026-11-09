#pragma once
#include "spline_conv_sparse_output.h"
#include "spline_conv_clear_owner.h"
#include "spline_conv_single_interpolate.h"

template <uint32_t C,bool Closed0,bool Closed1,uint32_t FixedBlocks=0,bool DedicatedClear=false,bool BulkFeatures=false>
class SplineEdgeOwner {
    static constexpr bool PerEdgeFeatures=DedicatedClear && !BulkFeatures;
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,uint32_t n,uint32_t e,uint32_t mean) {
        nodes=n;edgeCount=e;average=mean;
        xg.SetGlobalBuffer((__gm__ float*)x);eg.SetGlobalBuffer((__gm__ int64_t*)edges);
        pg.SetGlobalBuffer((__gm__ float*)pseudo);wg.SetGlobalBuffer((__gm__ float*)weight);yg.SetGlobalBuffer((__gm__ float*)out);
        el=LocalTensor<int64_t>(TPosition::VECCALC,0,128);
        pr=LocalTensor<float>(TPosition::VECCALC,1024,128);
        zeros=LocalTensor<float>(TPosition::VECCALC,1536,32*16);
        weights=LocalTensor<float>(TPosition::VECCALC,3584,4*C*16);
        features=LocalTensor<float>(TPosition::VECCALC,3584+4*C*16*4,64*C);
        agg=LocalTensor<float>(TPosition::VECCALC,3584+4*C*16*4+64*C*4,4*C);
        spread=LocalTensor<float>(TPosition::VECCALC,3584+4*C*16*4+64*C*4+4*C*4,4*C*8);
        dot=LocalTensor<float>(TPosition::VECCALC,3584+4*C*16*4+64*C*4+4*C*4+4*C*8*4,4*C*16);
    }
    __aicore__ inline void Process(GM_ADDR attributes=nullptr) {
        uint32_t block=GetBlockIdx(),blocks=FixedBlocks?FixedBlocks:GetBlockNum();
        if constexpr(C==32 && FixedBlocks==8 && DedicatedClear && BulkFeatures)
            block=(block>>1)|((block&1)<<2);
        if constexpr(DedicatedClear) {
            if(block>=4) {SplineClearOwner<4,8>(eg,yg,nodes,block,el,zeros);return;}
        }
        uint32_t zeroBound=(nodes+blocks-1)/blocks,zeroBegin=block*zeroBound;
        uint32_t zeroRows=zeroBegin<nodes?nodes-zeroBegin:0;if(zeroRows>zeroBound) zeroRows=zeroBound;
        DataCopyPadExtParams<int64_t> epad{false,0,0,0};DataCopyPadExtParams<float> pad{false,0,0,0.0f};
        DataCopyPad(el,eg,DataCopyExtParams{1,edgeCount*16,0,0,0},epad);
        DataCopyPad(pr,pg,DataCopyExtParams{1,edgeCount*8,0,0,0},pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        DataCopy(weights,wg,4*C*16);
        if constexpr(DedicatedClear) SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr(!PerEdgeFeatures) DataCopy(features,xg,nodes*C);
        if constexpr(!DedicatedClear) Duplicate(zeros,0.0f,zeroRows*16);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        bool groupedHead=false;
        if constexpr(PerEdgeFeatures) {
            groupedHead=el.GetValue(4)==el.GetValue(5);
            if(!groupedHead) {
                int64_t src=el.GetValue(block);
                if(src>=0 && src<nodes) DataCopy(features[block*C],xg[uint32_t(src)*C],C);
            }
        }
        uint32_t occupied=0;
        int64_t target=block<edgeCount?el.GetValue(edgeCount+block):-1;
        bool owner=target>=0 && target<nodes;
        bool uniqueTarget=false;
        if constexpr(C==32 && FixedBlocks==8 && PerEdgeFeatures) {
            uint32_t targetBits=SplineTargetBits4(el,target);
            owner=owner && (targetBits&((1u<<block)-1))==0;
            uniqueTarget=(targetBits&(targetBits-1))==0;
            if constexpr(PerEdgeFeatures) {
                if(owner && (groupedHead || !uniqueTarget)) {
                    for(uint32_t e=0;e<4;++e) {
                        if((targetBits&(1u<<e)) && (groupedHead || e!=block)) {
                            int64_t src=el.GetValue(e);
                            if(src>=0 && src<nodes) DataCopy(features[e*C],xg[uint32_t(src)*C],C);
                        }
                    }
                }
            }
        } else {
            uint32_t degree=0;
            for(uint32_t e=0;e<edgeCount;++e) {
                int64_t dst=el.GetValue(edgeCount+e);
                if constexpr(!DedicatedClear) {
                    if(dst>=zeroBegin && dst<int64_t(zeroBegin+zeroRows)) occupied|=1u<<(uint32_t(dst)-zeroBegin);
                }
                if(dst==target) {
                    if(e<block) owner=false;
                    ++degree;
                    if constexpr(PerEdgeFeatures) {
                        if(owner && (groupedHead || e!=block)) {
                            int64_t src=el.GetValue(e);
                            if(src>=0 && src<nodes) DataCopy(features[e*C],xg[uint32_t(src)*C],C);
                        }
                    }
                }
            }
            uniqueTarget=degree==1;
        }
        if constexpr(!DedicatedClear) {
            if(zeroRows) spline_output::ClearInactiveRows(yg,zeros,zeroBegin,zeroRows,occupied);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        if constexpr(!DedicatedClear) WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        if(!owner) {WaitData();return;}
        if(uniqueTarget) Single(block);
        else {
            WaitData();
            Duplicate(agg,0.0f,4*C);PipeBarrier<PIPE_ALL>();uint32_t valid=0;
            for(uint32_t e=0;e<edgeCount;++e) {
                if(el.GetValue(edgeCount+e)!=target) continue;
                int64_t src=el.GetValue(e);if(src<0 || src>=nodes) continue;
                float f[4];uint32_t wi;Factors(e,f,wi);
                for(uint32_t slot=0;slot<4;++slot) {
                    uint32_t index=wi^((slot&1)+2*(slot>>1));
                    Axpy(agg[index*C],features[(PerEdgeFeatures?e:uint32_t(src))*C],f[slot],C);
                }
                ++valid;
            }
            if(!valid) return;
            Brcb(spread,agg,uint8_t(4*C/8),BrcbRepeatParams{1,8});
            for(uint32_t row=0;row<4*C;row+=255) {
                uint32_t count=4*C-row;if(count>255) count=255;
                Mul(dot[row*16],weights[row*16],spread[row*8],uint64_t(16),uint8_t(count),BinaryRepeatParams{1,1,0,2,2,1});
            }
            for(uint32_t r=4*C;r>1;r/=2) Add(dot,dot,dot[r/2*16],r/2*16);
            uint32_t mean=average;
            if(attributes && valid!=1) mean=((const volatile __gm__ SplineConvTilingData*)attributes)->mean;
            if(mean && valid!=1) Muls(dot,dot,1.0f/valid,16);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        spline_output::CopyRows(yg,dot,uint32_t(target),1);
    }
private:
    __aicore__ inline void WaitData() {
        if constexpr(DedicatedClear) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
    }
    __aicore__ inline void Factors(uint32_t e,float *f,uint32_t &wi) {
        SplineLinearFactors<Closed0,Closed1>(pr.GetValue(e*2),pr.GetValue(e*2+1),f);
        wi=0;
    }
    __aicore__ inline void Single(uint32_t e) {
        int64_t src=el.GetValue(e);
        if(src<0 || src>=nodes) {WaitData();Duplicate(dot,0.0f,16);return;}
        float f[4];uint32_t wi;Factors(e,f,wi);
        if constexpr(DedicatedClear) WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Muls(dot,weights[wi*C*16],f[0],C*16);
        for(uint32_t slot=1;slot<4;++slot) Axpy(dot,weights[(wi^slot)*C*16],f[slot],C*16);
        if constexpr(DedicatedClear) WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Brcb(spread,features[(PerEdgeFeatures?e:uint32_t(src))*C],uint8_t(C/8),BrcbRepeatParams{1,8});
        Mul(dot,dot,spread,uint64_t(16),uint8_t(C),BinaryRepeatParams{1,1,0,2,2,1});
        if constexpr(C==32) {
            Add(dot,dot[64],dot,uint64_t(64),uint8_t(7),BinaryRepeatParams{1,1,1,0,8,0});
            Add(dot,dot,dot[32],uint64_t(32),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});
            Add(dot,dot,dot[16],uint64_t(16),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});
        } else {
            for(uint32_t r=C;r>1;r/=2) Add(dot,dot,dot[r/2*16],r/2*16);
        }
    }
    uint32_t nodes,edgeCount,average;
    GlobalTensor<float> xg,pg,wg,yg;GlobalTensor<int64_t> eg;
    LocalTensor<int64_t> el;LocalTensor<float> pr,zeros,weights,features,agg,spread,dot;
};

template <uint32_t Pad,bool Closed0,bool Closed1>
__aicore__ inline void RunFixedEdgeOwner(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,
        GM_ADDR weight,GM_ADDR out,GM_ADDR tiling) {
    uint32_t blocks=GetBlockNum();
    if(blocks==8) {
        SplineEdgeOwner<32,Closed0,Closed1,8,true,true> op;
        op.Init(x,edges,pseudo,weight,out,64-Pad,4,0);
        op.Process(tiling);return;
    }
    uint32_t nodes=16*(blocks-4)-Pad;
    SplineEdgeOwner<32,Closed0,Closed1> op;
    op.Init(x,edges,pseudo,weight,out,nodes,4,0);
    op.Process(tiling);
}
