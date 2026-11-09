#pragma once

template <typename T> class SplineStreamingKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edge, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR root, GM_ADDR bias, GM_ADDR out, GM_ADDR work, const SplineConvTilingData &td) {
        t=td;
        xg.SetGlobalBuffer((__gm__ T*)x); xb.SetGlobalBuffer((__gm__ uint16_t*)x);
        pg.SetGlobalBuffer((__gm__ T*)pseudo); pb.SetGlobalBuffer((__gm__ uint16_t*)pseudo);
        eg.SetGlobalBuffer((__gm__ int64_t*)edge); wg.SetGlobalBuffer((__gm__ T*)weight); wb.SetGlobalBuffer((__gm__ uint16_t*)weight);
        rg.SetGlobalBuffer((__gm__ T*)root); bg.SetGlobalBuffer((__gm__ T*)bias); yg.SetGlobalBuffer((__gm__ T*)out);
        next.SetGlobalBuffer((__gm__ int32_t*)work+GetBlockIdx()*((t.e+15)/16*16));
        pipe.InitBuffer(headBuf,((t.nodesPerBlock+7)/8*8)*4);
        pipe.InitBuffer(floatBuf,128*24); pipe.InitBuffer(rawBuf,128*sizeof(T)); pipe.InitBuffer(outBuf,128*sizeof(T));
        h=headBuf.Get<int32_t>(); w=floatBuf.Get<float>(); y=w[128]; comp=w[256]; prod=w[384]; z=w[512]; sum=w[640];
        raw=rawBuf.Get<T>(); yr=outBuf.Get<T>();
    }
    __aicore__ inline float XValue(uint32_t offset) {
        if constexpr(sizeof(T)==2) {union {uint32_t u;float f;} bits;bits.u=uint32_t(xb.GetValue(offset))<<16;return bits.f;}
        else return xg.GetValue(offset);
    }
    __aicore__ inline void Basis(uint32_t edge, float b[8][4], uint32_t base[8]) {
        for(uint32_t d=0;d<t.dim;++d) {
            float pv;
            uint32_t offset=t.pseudoTrans?d*t.e+edge:edge*t.dim+d;
            if constexpr(sizeof(T)==2) {union {uint32_t u;float f;} bits;bits.u=uint32_t(pb.GetValue(offset))<<16;pv=bits.f;}
            else pv=pg.GetValue(offset);
            float v=pv*int32_t(t.scale[d]); int32_t integral=int32_t(v);
            if(v<float(integral)) --integral;
            v-=float(integral); int32_t wrapped=integral%int32_t(t.ks[d]);
            base[d]=wrapped<0?wrapped+int32_t(t.ks[d]):wrapped;
            if(t.degree==1) {b[d][0]=1-v;b[d][1]=v;}
            else if(t.degree==2) {b[d][0]=0.5f*v*v-v+0.5f;b[d][1]=-v*v+v+0.5f;b[d][2]=0.5f*v*v;}
            else if(t.degree==0) b[d][0]=1.0f;
            else {float v2=v*v,v3=v2*v,u=1-v;b[d][0]=u*u*u/6;b[d][1]=(3*v3-6*v2+4)/6;b[d][2]=(-3*v3+3*v2+3*v+1)/6;b[d][3]=v3/6;}
        }
    }
    __aicore__ inline void LoadWeight(const GlobalTensor<T> &g, uint64_t offset, uint32_t count, uint32_t padded) {
        DataCopyExtParams p{1,count*uint32_t(sizeof(T)),0,0,0}; DataCopyPadExtParams<T> pad{true,0,uint8_t(padded-count),{}};
        DataCopyPad(raw,g[offset],p,pad); PipeBarrier<PIPE_ALL>();
        if constexpr(sizeof(T)==2) Cast(w,raw,RoundMode::CAST_NONE,padded); else DataCopy(w,raw,padded);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Accumulate(float factor, uint32_t count) {
        Muls(prod,w,factor,count); PipeBarrier<PIPE_V>();
        Sub(z,prod,comp,count); PipeBarrier<PIPE_V>();
        Add(sum,y,z,count); PipeBarrier<PIPE_V>();
        Sub(comp,sum,y,count); PipeBarrier<PIPE_V>();
        Sub(comp,comp,z,count); PipeBarrier<PIPE_V>();
        DataCopy(y,sum,count); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadKernelWeight(uint32_t kernel, uint32_t channel, uint32_t col, uint32_t count, uint32_t padded) {
        if(!t.weightTrans) {LoadWeight(wg,(uint64_t(kernel)*t.ci+channel)*t.co+col,count,padded);return;}
        for(uint32_t j=0;j<padded;++j) {
            float value=0;
            if(j<count) {
                uint64_t offset=(uint64_t(kernel)*t.co+col+j)*t.ci+channel;
                if constexpr(sizeof(T)==2) {union {uint32_t u;float f;} bits;bits.u=uint32_t(wb.GetValue(offset))<<16;value=bits.f;}
                else value=wg.GetValue(offset);
            }
            w.SetValue(j,value);
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Process() {
        uint32_t start=GetBlockIdx()*t.nodesPerBlock, end=start+t.nodesPerBlock; if(end>t.n) end=t.n;
        for(uint32_t j=0;j<t.nodesPerBlock;++j) h.SetValue(j,-1);
        for(uint32_t edge=0;edge<t.e;++edge) {
            int64_t dst=eg.GetValue(t.edgeTrans?2*edge+1:t.e+edge);
            if(dst>=start && dst<end) {uint32_t j=uint32_t(dst)-start;next.SetValue(edge,h.GetValue(j));h.SetValue(j,edge);}
        }
        for(uint32_t node=start;node<end;++node) {
            uint32_t degree=0;
            for(int32_t edge=h.GetValue(node-start);edge>=0;edge=next.GetValue(edge)) {int64_t src=eg.GetValue(t.edgeTrans?2*edge:edge);if(src>=0 && src<t.n) ++degree;}
            for(uint32_t col=0;col<t.co;col+=128) {
                uint32_t count=t.co-col; if(count>128) count=128;
                uint32_t align=sizeof(T)==2?16:8,padded=(count+align-1)/align*align;
                Duplicate(y,0.0f,padded); Duplicate(comp,0.0f,padded); PipeBarrier<PIPE_ALL>();
                for(int32_t edge=h.GetValue(node-start);edge>=0;edge=next.GetValue(edge)) {
                    int64_t src=eg.GetValue(t.edgeTrans?2*edge:edge); if(src<0 || src>=t.n) continue;
                    float b[8][4];uint32_t base[8];Basis(edge,b,base);
                    for(uint32_t slot=0;slot<t.slots;++slot) {
                        uint32_t q=slot,wi=0,stride=1;float factor=1;
                        for(uint32_t d=0;d<t.dim;++d) {uint32_t j=q%(t.degree+1);q/=t.degree+1;wi+=((base[d]+j)%t.ks[d])*stride;stride*=t.ks[d];factor*=b[d][j];}
                        if(factor==0) continue;
                        for(uint32_t c=0;c<t.ci;++c) {
                            float a=factor*XValue(uint32_t(src)*t.ci+c);
                            if(a!=0) {LoadKernelWeight(wi,c,col,count,padded);Accumulate(a,padded);}
                        }
                    }
                }
                if(t.mean && degree>0) {Muls(y,y,1.0f/degree,padded);PipeBarrier<PIPE_V>();}
                if(t.hasRoot) {
                    Duplicate(comp,0.0f,padded);PipeBarrier<PIPE_V>();
                    for(uint32_t c=0;c<t.ci;++c) {float a=XValue(node*t.ci+c);if(a!=0) {LoadWeight(rg,uint64_t(c)*t.co+col,count,padded);Accumulate(a,padded);}}
                }
                if(t.hasBias) {LoadWeight(bg,col,count,padded);Add(y,y,w,padded);PipeBarrier<PIPE_V>();}
                if constexpr(sizeof(T)==2) Cast(yr,y,RoundMode::CAST_RINT,padded); else DataCopy(yr,y,padded);
                PipeBarrier<PIPE_ALL>();DataCopyExtParams p{1,count*uint32_t(sizeof(T)),0,0,0};DataCopyPad(yg[node*t.co+col],yr,p);PipeBarrier<PIPE_ALL>();
            }
        }
    }
private:
    SplineConvTilingData t; TPipe pipe;
    TBuf<TPosition::VECCALC> headBuf,floatBuf,rawBuf,outBuf;
    LocalTensor<int32_t> h; LocalTensor<float> w,y,comp,prod,z,sum; LocalTensor<T> raw,yr;
    GlobalTensor<T> xg,pg,wg,rg,bg,yg;GlobalTensor<uint16_t> xb,pb,wb;GlobalTensor<int64_t> eg;GlobalTensor<int32_t> next;
};
