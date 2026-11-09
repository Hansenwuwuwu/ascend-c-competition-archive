#pragma once

template <typename T> class SplineSmallKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edge, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR root, GM_ADDR bias, GM_ADDR out, const SplineConvTilingData &td, TPipe *pipe) {
        t=td;
        xg.SetGlobalBuffer((__gm__ T*)x); wg.SetGlobalBuffer((__gm__ T*)weight);
        pg.SetGlobalBuffer((__gm__ T*)pseudo); eg.SetGlobalBuffer((__gm__ int64_t*)edge);
        rg.SetGlobalBuffer((__gm__ T*)root); bg.SetGlobalBuffer((__gm__ T*)bias); yg.SetGlobalBuffer((__gm__ T*)out);
        pipe->InitBuffer(xRawBuf,((t.n*8*sizeof(T)+31)/32)*32);
        pipe->InitBuffer(wRawBuf,256*sizeof(T)); pipe->InitBuffer(pBuf,((t.e*2*sizeof(T)+31)/32)*32);
        pipe->InitBuffer(eBuf,((t.e*16+31)/32)*32);
        pipe->InitBuffer(rRawBuf,64*sizeof(T)); pipe->InitBuffer(bRawBuf,32);
        pipe->InitBuffer(yBuf,64); pipe->InitBuffer(yRawBuf,32);
        xr=xRawBuf.Get<T>(); wr=wRawBuf.Get<T>(); pr=pBuf.Get<T>(); el=eBuf.Get<int64_t>();
        rr=rRawBuf.Get<T>(); br=bRawBuf.Get<T>(); y=yBuf.Get<float>(); yr=yRawBuf.Get<T>();
        if constexpr(sizeof(T)==2) {
            pipe->InitBuffer(xBuf,t.n*8*4); pipe->InitBuffer(wBuf,256*4);
            pipe->InitBuffer(rBuf,64*4); pipe->InitBuffer(bBuf,16*4);
            xv=xBuf.Get<float>(); wv=wBuf.Get<float>(); rv=rBuf.Get<float>(); bv=bBuf.Get<float>();
        } else {
            xv=xr.template ReinterpretCast<float>(); wv=wr.template ReinterpretCast<float>();
            rv=rr.template ReinterpretCast<float>(); bv=br.template ReinterpretCast<float>();
        }
    }
    __aicore__ inline void Load(LocalTensor<T> local, GlobalTensor<T> global, uint32_t count) {
        DataCopyExtParams p{1,count*uint32_t(sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{false,0,0,{}};
        DataCopyPad(local,global,p,pad);
    }
    __aicore__ inline float Pseudo(uint32_t i) {
        if constexpr(sizeof(T)==2) {
            union {uint32_t u;float f;} value;
            value.u=uint32_t(pr.template ReinterpretCast<uint16_t>().GetValue(i))<<16;
            return value.f;
        } else return pr.GetValue(i);
    }
    __aicore__ inline void Process() {
        Load(xr,xg,t.n*8); Load(wr,wg,256); Load(pr,pg,t.e*2);
        DataCopyExtParams ep{1,t.e*16,0,0,0}; DataCopyPadExtParams<int64_t> ez{false,0,0,0};
        DataCopyPad(el,eg,ep,ez);
        if(t.hasRoot) Load(rr,rg,64);
        if(t.hasBias) Load(br,bg,8);
        PipeBarrier<PIPE_ALL>();
        if constexpr(sizeof(T)==2) {
            Cast(xv,xr,RoundMode::CAST_NONE,t.n*8); Cast(wv,wr,RoundMode::CAST_NONE,256);
            if(t.hasRoot) Cast(rv,rr,RoundMode::CAST_NONE,64);
            if(t.hasBias) Cast(bv,br,RoundMode::CAST_NONE,8);
            PipeBarrier<PIPE_ALL>();
        }
        uint32_t start=GetBlockIdx()*t.nodesPerBlock, end=start+t.nodesPerBlock; if(end>t.n) end=t.n;
        for(uint32_t node=start;node<end;++node) {
            float a[32]={}, correction[32]={}; uint32_t count=0;
            for(uint32_t e=0;e<t.e;++e) if(el.GetValue(t.e+e)==node) {
                int64_t src=el.GetValue(e); if(src<0 || src>=t.n) continue;
                float v0=Pseudo(e*2)*t.scale[0], v1=Pseudo(e*2+1)*t.scale[1];
                int32_t base0=int32_t(v0), base1=int32_t(v1);
                v0-=float(base0); v1-=float(base1); float factors[4];
                factors[(base0&1)+2*(base1&1)]=(1-v0)*(1-v1);
                factors[((base0+1)&1)+2*(base1&1)]=v0*(1-v1);
                factors[(base0&1)+2*((base1+1)&1)]=(1-v0)*v1;
                factors[((base0+1)&1)+2*((base1+1)&1)]=v0*v1;
                ++count;
                for(uint32_t c=0;c<8;++c) {
                    float value=xv.GetValue(uint32_t(src)*8+c);
                    for(uint32_t k=0;k<4;++k) {
                        uint32_t idx=k*8+c; float term=value*factors[k]-correction[idx], updated=a[idx]+term;
                        correction[idx]=(updated-a[idx])-term; a[idx]=updated;
                    }
                }
            }
            for(uint32_t c=0;c<16;++c) y.SetValue(c,0.0f);
            for(uint32_t col=0;col<8;++col) {
                float value=0;
                for(uint32_t row=0;row<32;++row) value+=a[row]*wv.GetValue(row*8+col);
                if(t.mean && count>0) value/=float(int32_t(count));
                if(t.hasRoot) for(uint32_t row=0;row<8;++row) value+=xv.GetValue(node*8+row)*rv.GetValue(row*8+col);
                if(t.hasBias) value+=bv.GetValue(col);
                y.SetValue(col,value);
            }
            PipeBarrier<PIPE_ALL>();
            if constexpr(sizeof(T)==2) Cast(yr,y,RoundMode::CAST_RINT,16);
            else DataCopy(yr,y,8);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams yp{1,8*uint32_t(sizeof(T)),0,0,0}; DataCopyPad(yg[node*8],yr,yp);
            PipeBarrier<PIPE_ALL>();
        }
    }
private:
    SplineConvTilingData t;
    TBuf<TPosition::VECCALC> xRawBuf,wRawBuf,pBuf,eBuf,rRawBuf,bRawBuf,yBuf,yRawBuf,xBuf,wBuf,rBuf,bBuf;
    LocalTensor<T> xr,wr,pr,rr,br,yr; LocalTensor<int64_t> el; LocalTensor<float> xv,wv,rv,bv,y;
    GlobalTensor<T> xg,wg,pg,rg,bg,yg; GlobalTensor<int64_t> eg;
};
