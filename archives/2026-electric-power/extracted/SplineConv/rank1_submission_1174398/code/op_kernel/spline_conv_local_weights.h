#pragma once

template <typename T,uint32_t C>
class SplineSparseLocalWeights {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,const SplineConvTilingData &td,TPipe *pipe) {
        t=td; xg.SetGlobalBuffer((__gm__ T*)x); eg.SetGlobalBuffer((__gm__ int64_t*)edges);
        pg.SetGlobalBuffer((__gm__ T*)pseudo); wg.SetGlobalBuffer((__gm__ T*)weight); yg.SetGlobalBuffer((__gm__ T*)out);
        uint32_t off=0;
        uint32_t eo=Reserve(off,t.e*16),po=Reserve(off,t.e*2*sizeof(T)),ho=Reserve(off,t.nodesPerBlock*4),no=Reserve(off,t.e*4);
        uint32_t ro=C==128?Reserve(off,t.e*4):0;
        uint32_t wo=Reserve(off,4*C*16*sizeof(T));
        uint32_t wf=sizeof(T)==2?Reserve(off,4*C*16*4):wo;
        uint32_t xo=Reserve(off,C*sizeof(T)),xf=sizeof(T)==2?Reserve(off,C*4):xo;
        uint32_t ao=Reserve(off,4*C*4),bo=Reserve(off,4*C*8*4),doff=Reserve(off,4*C*16*4),yo=Reserve(off,t.nodesPerBlock*16*sizeof(T));
        pipe->InitBuffer(buf,off);
        el=buf.Get<int64_t>()[eo/8]; pr=buf.Get<T>()[po/sizeof(T)]; pb=pr.template ReinterpretCast<uint16_t>();
        head=buf.Get<int32_t>()[ho/4]; links=buf.Get<int32_t>()[no/4];
        if constexpr(C==128) occupied=buf.Get<uint32_t>()[ro/4];
        wr=buf.Get<T>()[wo/sizeof(T)]; weights=buf.Get<float>()[wf/4];
        xr=buf.Get<T>()[xo/sizeof(T)]; xv=buf.Get<float>()[xf/4];
        agg=buf.Get<float>()[ao/4]; broadcast=buf.Get<float>()[bo/4]; dot=buf.Get<float>()[doff/4]; outLocal=buf.Get<T>()[yo/sizeof(T)];
    }
    __aicore__ inline uint32_t Reserve(uint32_t &off,uint32_t bytes) {
        uint32_t result=off; off+=(bytes+31)/32*32; return result;
    }
    __aicore__ inline float Pseudo(uint32_t i) {
        if constexpr(sizeof(T)==2) {union {uint32_t u;float f;} b;b.u=uint32_t(pb.GetValue(i))<<16;return b.f;}
        else return pr.GetValue(i);
    }
    __aicore__ inline bool SingleDot(uint32_t e) {
        int64_t src=el.GetValue(e);if(src<0 || src>=t.n) return false;
        float p0=Pseudo(e*2)*int32_t(t.scale[0]),p1=Pseudo(e*2+1)*int32_t(t.scale[1]);
        int32_t b0=int32_t(p0),b1=int32_t(p1);p0-=float(b0);p1-=float(b1);
        float f[4]={(1-p0)*(1-p1),p0*(1-p1),(1-p0)*p1,p0*p1};
        DataCopyPadExtParams<T> pad{false,0,0,{}};
        DataCopyExtParams xp{1,C*uint32_t(sizeof(T)),0,0,0};
        DataCopyPad(xr,xg[uint32_t(src)*C],xp,pad);
        DataCopyExtParams wp{1,C*16*uint32_t(sizeof(T)),0,0,0};
        for(uint32_t slot=0;slot<4;++slot) {
            uint32_t wi=(uint32_t(b0)+(slot&1))%t.ks[0]+t.ks[0]*((uint32_t(b1)+(slot>>1))%t.ks[1]);
            DataCopyPad(wr[slot*C*16],wg[wi*C*16],wp,pad);
        }
        PipeBarrier<PIPE_ALL>();
        if constexpr(sizeof(T)==2) {Cast(xv,xr,RoundMode::CAST_NONE,C);Cast(weights,wr,RoundMode::CAST_NONE,4*C*16);PipeBarrier<PIPE_V>();}
        Muls(dot,weights,f[0],C*16);PipeBarrier<PIPE_V>();
        for(uint32_t slot=1;slot<4;++slot) {Axpy(dot,weights[slot*C*16],f[slot],C*16);PipeBarrier<PIPE_V>();}
        Brcb(broadcast,xv,uint8_t(C/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        Mul(dot,dot,broadcast,uint64_t(16),uint8_t(C),BinaryRepeatParams{1,1,0,2,2,1});PipeBarrier<PIPE_V>();
        for(uint32_t rows=C;rows>1;rows/=2) {Add(dot,dot,dot[rows/2*16],rows/2*16);PipeBarrier<PIPE_V>();}
        return true;
    }
    __aicore__ inline void Process() {
        uint32_t start=GetBlockIdx(),step=t.blocks,rows=(t.n-start+step-1)/step;
        DataCopyExtParams ep{1,t.e*16,0,0,0};DataCopyPadExtParams<int64_t> ez{false,0,0,0};DataCopyPad(el,eg,ep,ez);
        DataCopyExtParams pp{1,t.e*2*uint32_t(sizeof(T)),0,0,0};DataCopyPadExtParams<T> pz{false,0,0,{}};
        DataCopyPad(pr,pg,pp,pz);
        Duplicate(outLocal.template ReinterpretCast<int32_t>(),int32_t(0),rows*16*sizeof(T)/4); PipeBarrier<PIPE_ALL>();
        if constexpr(C!=128) for(uint32_t i=0;i<rows;++i) head.SetValue(i,-1);
        uint32_t activeCount=0; bool found=false;
        for(uint32_t e=0;e<t.e;++e) {
            int64_t dst=el.GetValue(t.e+e);
            if(dst>=start && dst<t.n && (uint32_t(dst)-start)%step==0) {
                uint32_t j=(uint32_t(dst)-start)/step;
                if constexpr(C==128) {
                    uint32_t index=0;
                    while(index<activeCount && occupied.GetValue(index)!=j) ++index;
                    if(index==activeCount) {occupied.SetValue(index,j); head.SetValue(index,-1); ++activeCount;}
                    j=index;
                }
                links.SetValue(e,head.GetValue(j));head.SetValue(j,e);found=true;
            }
        }
        if(found) {
            for(uint32_t j=0;j<activeCount;++j) {
                uint32_t count=0;
                Duplicate(agg,0.0f,16);PipeBarrier<PIPE_V>();
                for(int32_t e=head.GetValue(j);e>=0;e=links.GetValue(e)) {
                    if(SingleDot(uint32_t(e))) {Add(agg,agg,dot,16);PipeBarrier<PIPE_V>();++count;}
                }
                if(count==0) continue;
                if(t.mean && count>1) {Muls(agg,agg,1.0f/count,16);PipeBarrier<PIPE_V>();}
                uint32_t row=occupied.GetValue(j);
                if constexpr(sizeof(T)==2) Cast(outLocal[row*16],agg,RoundMode::CAST_RINT,16);
                else DataCopy(outLocal[row*16],agg,16);
                PipeBarrier<PIPE_ALL>();
            }
        }
        DataCopyExtParams yp{uint16_t(rows),16*uint32_t(sizeof(T)),0,(step-1)*16*uint32_t(sizeof(T)),0};
        PipeBarrier<PIPE_ALL>();DataCopyPad(yg[start*16],outLocal,yp);PipeBarrier<PIPE_ALL>();
    }
private:
    SplineConvTilingData t;TBuf<TPosition::VECCALC> buf;
    LocalTensor<int64_t> el;LocalTensor<int32_t> head,links;LocalTensor<uint32_t> occupied;
    LocalTensor<T> pr,wr,xr,outLocal;LocalTensor<uint16_t> pb;
    LocalTensor<float> weights,xv,agg,broadcast,dot;
    GlobalTensor<T> xg,pg,wg,yg;GlobalTensor<int64_t> eg;
};
template <typename T>
__aicore__ inline bool RunSparseLocalWeights(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,const SplineConvTilingData &t) {
    if(t.ci!=128 || t.co!=16 || t.kernels<=4 || t.dim!=2 || t.degree!=1 ||
       t.ks[0]<2 || t.ks[0]>4 || t.ks[1]<2 || t.ks[1]>4 || t.e==0 || t.e>16 || t.e>=t.n ||
       t.nodesPerBlock>128 || t.hasRoot || t.hasBias || t.streaming || t.edgeTrans || t.pseudoTrans || t.weightTrans) return false;
    TPipe pipe;SplineSparseLocalWeights<T,128> op;op.Init(x,edges,pseudo,weight,out,t,&pipe);op.Process();
    return true;
}
