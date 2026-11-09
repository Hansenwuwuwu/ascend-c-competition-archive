#pragma once

template <typename T, bool LargeKernelCount=false, uint32_t FixedCi=0> class SplineKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edge, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR root, GM_ADDR bias, GM_ADDR out, GM_ADDR work, const SplineConvTilingData &td) {
        t=td;
        if constexpr(FixedCi!=0) {
            t.ci=FixedCi; t.ciPad=FixedCi; t.co=16; t.coPad=16;
            t.dim=2; t.degree=1; t.slots=4; t.kernels=4;
            t.hasRoot=0; t.hasBias=0; t.cacheWeights=1; t.cacheGraph=1; t.fusedDot=1;
        }
        xg.SetGlobalBuffer((__gm__ T*)x); eg.SetGlobalBuffer((__gm__ int64_t*)edge);
        pg.SetGlobalBuffer((__gm__ T*)pseudo); pbits.SetGlobalBuffer((__gm__ uint16_t*)pseudo); wg.SetGlobalBuffer((__gm__ T*)weight);
        rg.SetGlobalBuffer((__gm__ T*)root); bg.SetGlobalBuffer((__gm__ T*)bias); yg.SetGlobalBuffer((__gm__ T*)out);
        next.SetGlobalBuffer((__gm__ int32_t*)work+GetBlockIdx()*((t.e+15)/16*16));
        if(t.cacheGraph) {
            pipe.InitBuffer(edgeBuf,((2*t.e+3)/4*4)*8);
            pipe.InitBuffer(pseudoBuf,((t.e*t.dim+15)/16*16)*sizeof(T));
            pipe.InitBuffer(nextBuf,((t.e+7)/8*8)*4);
            edgeLocal=edgeBuf.Get<int64_t>(); pseudoLocal=pseudoBuf.Get<T>();
            pseudoBits=pseudoLocal.template ReinterpretCast<uint16_t>(); nextLocal=nextBuf.Get<int32_t>();
        }
        if(t.vectorDot || t.fusedDot) {
            uint32_t dotRows=t.fusedDot?t.kernels*t.ciPad:t.ciPad;
            pipe.InitBuffer(dotBuf,dotRows*t.coPad*4); pipe.InitBuffer(broadcastBuf,t.fusedDot?dotRows*32:t.ciPad*t.coPad*4);
            dot=dotBuf.Get<float>(); broadcast=broadcastBuf.Get<float>();
        }
        cacheSize=t.kernels*t.ci*t.coPad; useCache=t.cacheWeights!=0;
        if(useCache) {
            pipe.InitBuffer(cacheRawBuf,cacheSize*sizeof(T)); cacheRaw=cacheRawBuf.Get<T>();
            if constexpr(sizeof(T)==2) {pipe.InitBuffer(cacheBuf,cacheSize*4); cached=cacheBuf.Get<float>();}
            else cached=cacheRaw.template ReinterpretCast<float>();
        }
        pipe.InitBuffer(headBuf, ((t.nodesPerBlock+7)/8*8)*4);
        if constexpr(LargeKernelCount) {pipe.InitBuffer(activeBuf,((t.kernels+7)/8*8)*4); activeFlags=activeBuf.Get<uint32_t>();}
        pipe.InitBuffer(aggBuf,t.kernels*t.ciPad*4);
        if constexpr(sizeof(T)==2) pipe.InitBuffer(wBuf,t.ci*t.coPad*4);
        pipe.InitBuffer(compBuf,t.kernels*t.ciPad*4); pipe.InitBuffer(tempBuf,t.ciPad*12);
        pipe.InitBuffer(rawBuf,t.ci*t.coPad*sizeof(T));
        pipe.InitBuffer(xBuf,t.ciPad*4); pipe.InitBuffer(xRawBuf,t.ciPad*sizeof(T));
        pipe.InitBuffer(yBuf,t.coPad*4); pipe.InitBuffer(yRawBuf,t.coPad*sizeof(T));
        h=headBuf.Get<int32_t>(); agg=aggBuf.Get<float>(); raw=rawBuf.Get<T>();
        if constexpr(sizeof(T)==2) mat=wBuf.Get<float>(); else mat=raw.template ReinterpretCast<float>();
        comp=compBuf.Get<float>(); prod=tempBuf.Get<float>(); z=prod[t.ciPad]; sum=prod[2*t.ciPad];
        xv=xBuf.Get<float>(); xr=xRawBuf.Get<T>(); y=yBuf.Get<float>(); yr=yRawBuf.Get<T>();
    }
    __aicore__ inline int64_t EdgeValue(uint32_t i) {return t.cacheGraph?edgeLocal.GetValue(i):eg.GetValue(i);}
    __aicore__ inline int32_t NextValue(uint32_t i) {return t.cacheGraph?nextLocal.GetValue(i):next.GetValue(i);}
    __aicore__ inline void NextAssign(uint32_t i,int32_t v) {if(t.cacheGraph) nextLocal.SetValue(i,v); else next.SetValue(i,v);}
    __aicore__ inline void LoadRow(const GlobalTensor<T> &g, uint32_t offset) {
        DataCopyExtParams p{1,t.ci*uint32_t(sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{true,0,uint8_t(t.ciPad-t.ci),{}};
        DataCopyPad(xr,g[offset],p,pad); PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T)==2) Cast(xv,xr,RoundMode::CAST_NONE,t.ciPad);
        else DataCopy(xv,xr,t.ciPad);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadMatrix(const GlobalTensor<T> &g, uint32_t offset) {
        uint32_t rows=t.co==t.coPad?1:t.ci;
        uint32_t width=t.co==t.coPad?t.ci*t.co:t.co;
        DataCopyExtParams p{uint16_t(rows),width*uint32_t(sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{true,0,uint8_t(t.coPad-t.co),{}};
        DataCopyPad(raw,g[offset],p,pad); PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T)==2) Cast(mat,raw,RoundMode::CAST_NONE,t.ci*t.coPad);

        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Basis(uint32_t e, float b[8][4], uint32_t base[8]) {
        for(uint32_t d=0;d<t.dim;++d) {
            float pv; if constexpr(sizeof(T)==2) { union {uint32_t u; float f;} bits; bits.u=uint32_t((t.cacheGraph?pseudoBits.GetValue(e*t.dim+d):pbits.GetValue(e*t.dim+d)))<<16; pv=bits.f; } else pv=(t.cacheGraph?pseudoLocal.GetValue(e*t.dim+d):pg.GetValue(e*t.dim+d));
            float v=pv*int32_t(t.scale[d]);
            if(int32_t(t.scale[d])<0) {
                int32_t integral=int32_t(v); if(v<float(integral)) --integral;
                v-=float(integral); int32_t wrapped=integral%int32_t(t.ks[d]);
                base[d]=wrapped<0?wrapped+int32_t(t.ks[d]):wrapped;
            } else {base[d]=int32_t(v); v-=float(int32_t(base[d]));}
            if(t.degree==1) { b[d][0]=1-v; b[d][1]=v; }
            else if(t.degree==2) { b[d][0]=0.5f*v*v-v+0.5f; b[d][1]=-v*v+v+0.5f; b[d][2]=0.5f*v*v; }
            else if(t.degree==0) b[d][0]=1.0f;
            else { float v2=v*v,v3=v2*v,u=1-v; b[d][0]=u*u*u/6; b[d][1]=(3*v3-6*v2+4)/6; b[d][2]=(-3*v3+3*v2+3*v+1)/6; b[d][3]=v3/6; }
        }
    }
    template <uint32_t D, uint32_t Degree>
    __aicore__ inline void BasisSpecial(uint32_t e, float b[8][4], uint32_t base[8]) {
        for(uint32_t d=0;d<D;++d) {
            float pv; if constexpr(sizeof(T)==2) { union {uint32_t u; float f;} bits; bits.u=uint32_t((t.cacheGraph?pseudoBits.GetValue(e*D+d):pbits.GetValue(e*D+d)))<<16; pv=bits.f; } else pv=(t.cacheGraph?pseudoLocal.GetValue(e*D+d):pg.GetValue(e*D+d));
            float v=pv*int32_t(t.scale[d]);
            if(int32_t(t.scale[d])<0) {
                int32_t integral=int32_t(v); if(v<float(integral)) --integral;
                v-=float(integral); int32_t wrapped=integral%int32_t(t.ks[d]);
                base[d]=wrapped<0?wrapped+int32_t(t.ks[d]):wrapped;
            } else {base[d]=int32_t(v); v-=float(int32_t(base[d]));}
            if(Degree==1) { b[d][0]=1-v; b[d][1]=v; }
            else if(Degree==2) { b[d][0]=0.5f*v*v-v+0.5f; b[d][1]=-v*v+v+0.5f; b[d][2]=0.5f*v*v; }
            else if(Degree==0) b[d][0]=1.0f;
            else { float v2=v*v,v3=v2*v,u=1-v; b[d][0]=u*u*u/6; b[d][1]=(3*v3-6*v2+4)/6; b[d][2]=(-3*v3+3*v2+3*v+1)/6; b[d][3]=v3/6; }
        }
    }
    template <uint32_t D, uint32_t Degree>
    __aicore__ inline void AccumulateEdge(uint32_t e, bool stable, bool active[64]) {
        constexpr uint32_t Slots = D==2 ? (Degree+1)*(Degree+1) : (Degree+1)*(Degree+1)*(Degree+1);
        float b[8][4]; uint32_t base[8];
        BasisSpecial<D,Degree>(e,b,base);
        if(!stable) {SetMaskCount(); SetVectorMask<float,MaskMode::COUNTER>(t.ciPad);}
                for(uint32_t s=0;s<Slots;++s) {
                    uint32_t q=s,wi=0,stride=1; float f=1;
                    for(uint32_t d=0;d<D;++d) {uint32_t k=q%(Degree+1); q/=Degree+1; wi+=((base[d]+k)%t.ks[d])*stride; stride*=t.ks[d]; f*=b[d][k];}
                    if(f!=0) {
                        if constexpr(LargeKernelCount) activeFlags.SetValue(wi,1); else active[wi]=true;
                        uint32_t off=wi*t.ciPad;
                        if(stable) {
                            Muls(prod,xv,f,t.ciPad); PipeBarrier<PIPE_V>();
                            Sub(z,prod,comp[off],t.ciPad); PipeBarrier<PIPE_V>();
                            Add(sum,agg[off],z,t.ciPad); PipeBarrier<PIPE_V>();
                            Sub(comp[off],sum,agg[off],t.ciPad); PipeBarrier<PIPE_V>();
                            Sub(comp[off],comp[off],z,t.ciPad); PipeBarrier<PIPE_V>();
                            DataCopy(agg[off],sum,t.ciPad); PipeBarrier<PIPE_V>();
                        } else {Axpy<float,float,false>(agg[off],xv,f,MASK_PLACEHOLDER,1,UnaryRepeatParams{1,1,8,8}); PipeBarrier<PIPE_V>();}
                    }
                }
        if(!stable) {SetMaskNorm(); ResetMask();}
    }
    __aicore__ inline void DotMatrix(const LocalTensor<float> &weights,const LocalTensor<float> &values) {
        Brcb(broadcast,values,uint8_t(t.ciPad/8),BrcbRepeatParams{uint16_t(t.coPad/8),uint16_t(t.coPad)});
        PipeBarrier<PIPE_V>();
        uint32_t first=t.coPad<64?t.coPad:64;
        BinaryRepeatParams params{1,1,0,uint8_t(t.coPad/8),uint8_t(t.coPad/8),uint8_t(t.coPad/8)};
        Mul(dot,weights,broadcast,uint64_t(first),uint8_t(t.ci),params);
        if(t.coPad>64) Mul(dot[64],weights[64],broadcast,uint64_t(t.coPad-64),uint8_t(t.ci),params);
        PipeBarrier<PIPE_V>();
        uint32_t rows=t.ci;
        while(rows>1) {
            uint32_t half=(rows+1)/2;
            Add(dot,dot,dot[half*t.coPad],(rows-half)*t.coPad); PipeBarrier<PIPE_V>();
            rows=half;
        }
        Add(y,y,dot,t.coPad); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FusedDotMatrix(const LocalTensor<float> &weights,const LocalTensor<float> &values,uint32_t rows) {
        Brcb(broadcast,values,uint8_t((rows+7)/8),BrcbRepeatParams{1,8}); PipeBarrier<PIPE_V>();
        BinaryRepeatParams params{1,1,0,uint8_t(t.coPad/8),uint8_t(t.coPad/8),1};
        uint32_t first=t.coPad<64?t.coPad:64;
        for(uint32_t r=0;r<rows;r+=255) {
            uint32_t count=rows-r; if(count>255) count=255;
            Mul(dot[r*t.coPad],weights[r*t.coPad],broadcast[r*8],uint64_t(first),uint8_t(count),params);
            if(t.coPad>64) Mul(dot[r*t.coPad+64],weights[r*t.coPad+64],broadcast[r*8],uint64_t(t.coPad-64),uint8_t(count),params);
        }
        PipeBarrier<PIPE_V>();
        while(rows>1) {
            uint32_t half=(rows+1)/2;
            Add(dot,dot,dot[half*t.coPad],(rows-half)*t.coPad); PipeBarrier<PIPE_V>();
            rows=half;
        }
        Add(y,y,dot,t.coPad); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        if(t.cacheGraph) {
            DataCopyExtParams ep{1,t.e*16,0,0,0}; DataCopyPadExtParams<int64_t> ez{false,0,0,0};
            DataCopyPad(edgeLocal,eg,ep,ez);
            DataCopyExtParams pp{1,t.e*t.dim*uint32_t(sizeof(T)),0,0,0}; DataCopyPadExtParams<T> pz{false,0,0,{}};
            DataCopyPad(pseudoLocal,pg,pp,pz); PipeBarrier<PIPE_ALL>();
        }
        if(useCache) {
            uint32_t rows=t.co==t.coPad?1:t.kernels*t.ci;
            uint32_t width=t.co==t.coPad?cacheSize:t.co;
            DataCopyExtParams p{uint16_t(rows),width*uint32_t(sizeof(T)),0,0,0};
            DataCopyPadExtParams<T> pad{true,0,uint8_t(t.coPad-t.co),{}};
            DataCopyPad(cacheRaw,wg,p,pad); PipeBarrier<PIPE_ALL>();
            if constexpr(sizeof(T)==2) {Cast(cached,cacheRaw,RoundMode::CAST_NONE,cacheSize); PipeBarrier<PIPE_ALL>();}
        }
        bool cyclic=t.e<t.n && !t.hasRoot && t.nodesPerBlock>2;
        uint32_t step=cyclic?t.blocks:1;
        uint32_t start=cyclic?GetBlockIdx():GetBlockIdx()*t.nodesPerBlock;
        uint32_t end=cyclic?t.n:start+t.nodesPerBlock; if(end>t.n) end=t.n;
        for(uint32_t j=0;j<t.nodesPerBlock;++j) h.SetValue(j,-1);
        for(uint32_t e=0;e<t.e;++e) {
            int64_t dst=EdgeValue(t.e+e);
            if(dst>=start && dst<end && (!cyclic || (uint32_t(dst)-start)%step==0)) {
                uint32_t j=(uint32_t(dst)-start)/step; NextAssign(e,h.GetValue(j)); h.SetValue(j,e);
            }
        }
        bool sparseZero=t.e<t.n && !t.hasRoot && !t.hasBias && t.co==t.coPad && t.nodesPerBlock<=t.ci;
        if(sparseZero && start<end) {
            uint32_t rows=(end-start+step-1)/step;
            Duplicate(raw.template ReinterpretCast<int32_t>(),int32_t(0),rows*t.co*sizeof(T)/4);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams zeros{uint16_t(rows),t.co*uint32_t(sizeof(T)),0,(step-1)*t.co*uint32_t(sizeof(T)),0};
            DataCopyPad(yg[start*t.co],raw,zeros); PipeBarrier<PIPE_ALL>();
        }
        for(uint32_t node=start;node<end;node+=step) {
            if(sparseZero && h.GetValue((node-start)/step)<0) continue;
            Duplicate(agg,0.0f,t.kernels*t.ciPad); Duplicate(y,0.0f,t.coPad); PipeBarrier<PIPE_ALL>();
            uint32_t count=0,indegree=0; bool active[64]={};
            if constexpr(LargeKernelCount) for(uint32_t k=0;k<t.kernels;++k) activeFlags.SetValue(k,0);
            for(int32_t e=h.GetValue((node-start)/step);e>=0;e=NextValue(e)) ++indegree;
            bool stable=indegree>128; if(stable) {Duplicate(comp,0.0f,t.kernels*t.ciPad); PipeBarrier<PIPE_V>();}
            for(int32_t e=h.GetValue((node-start)/step);e>=0;e=NextValue(e)) {
                int64_t src=EdgeValue(e); if(src<0 || src>=t.n) continue;
                LoadRow(xg,uint32_t(src)*t.ci); ++count;
                if(t.dim==2 && t.degree==1) AccumulateEdge<2,1>(e,stable,active);
                else if(t.dim==2 && t.degree==2) AccumulateEdge<2,2>(e,stable,active);
                else if(t.dim==2 && t.degree==3) AccumulateEdge<2,3>(e,stable,active);
                else if(t.dim==3 && t.degree==1) AccumulateEdge<3,1>(e,stable,active);
                else if(t.dim==3 && t.degree==2) AccumulateEdge<3,2>(e,stable,active);
                else if(t.dim==3 && t.degree==3) AccumulateEdge<3,3>(e,stable,active);
                else {
                float b[8][4]; uint32_t base[8]; Basis(e,b,base);
                if(!stable) {SetMaskCount(); SetVectorMask<float,MaskMode::COUNTER>(t.ciPad);}
                for(uint32_t s=0;s<t.slots;++s) {
                    uint32_t q=s,wi=0,stride=1; float f=1;
                    for(uint32_t d=0;d<t.dim;++d) {uint32_t k=q%(t.degree+1); q/=t.degree+1; wi+=((base[d]+k)%t.ks[d])*stride; stride*=t.ks[d]; f*=b[d][k];}
                    if(f!=0) {
                        if constexpr(LargeKernelCount) activeFlags.SetValue(wi,1); else active[wi]=true;
                        uint32_t off=wi*t.ciPad;
                        if(stable) {
                            Muls(prod,xv,f,t.ciPad); PipeBarrier<PIPE_V>();
                            Sub(z,prod,comp[off],t.ciPad); PipeBarrier<PIPE_V>();
                            Add(sum,agg[off],z,t.ciPad); PipeBarrier<PIPE_V>();
                            Sub(comp[off],sum,agg[off],t.ciPad); PipeBarrier<PIPE_V>();
                            Sub(comp[off],comp[off],z,t.ciPad); PipeBarrier<PIPE_V>();
                            DataCopy(agg[off],sum,t.ciPad); PipeBarrier<PIPE_V>();
                        } else {Axpy<float,float,false>(agg[off],xv,f,MASK_PLACEHOLDER,1,UnaryRepeatParams{1,1,8,8}); PipeBarrier<PIPE_V>();}
                    }
                }
                if(!stable) {SetMaskNorm(); ResetMask();}
                }
            }
            PipeBarrier<PIPE_ALL>();
            if(t.fusedDot) {if(count>0) FusedDotMatrix(cached,agg,t.kernels*t.ci);}
            else for(uint32_t k=0;k<t.kernels;++k) if((!LargeKernelCount && active[k]) || (LargeKernelCount && activeFlags.GetValue(k)!=0)) {
                if(!useCache) LoadMatrix(wg,k*t.ci*t.co);
                LocalTensor<float> wk=useCache?cached[k*t.ci*t.coPad]:mat;
                if(t.vectorDot) DotMatrix(wk,agg[k*t.ciPad]);
                else {
                SetMaskCount(); SetVectorMask<float,MaskMode::COUNTER>(t.coPad);
                for(uint32_t c=0;c<t.ci;++c) {float a=agg.GetValue(k*t.ciPad+c); Axpy<float,float,false>(y,wk[c*t.coPad],a,MASK_PLACEHOLDER,1,UnaryRepeatParams{1,1,8,8}); PipeBarrier<PIPE_V>();}
                SetMaskNorm(); ResetMask();
                }
            }
            if(t.mean && count>0) {Muls(y,y,1.0f/count,t.coPad); PipeBarrier<PIPE_V>();}
            if(t.hasRoot) {LoadRow(xg,node*t.ci); LoadMatrix(rg,0); if(t.fusedDot) FusedDotMatrix(mat,xv,t.ci); else if(t.vectorDot) DotMatrix(mat,xv); else for(uint32_t c=0;c<t.ci;++c) {Axpy(y,mat[c*t.coPad],xv.GetValue(c),t.coPad); PipeBarrier<PIPE_V>();}}
            if(t.hasBias) { DataCopyExtParams p{1,t.co*uint32_t(sizeof(T)),0,0,0}; DataCopyPadExtParams<T> pad{true,0,uint8_t(t.coPad-t.co),{}}; DataCopyPad(raw,bg,p,pad); PipeBarrier<PIPE_ALL>(); if constexpr(sizeof(T)==2) Cast(mat,raw,RoundMode::CAST_NONE,t.coPad);  PipeBarrier<PIPE_ALL>(); Add(y,y,mat,t.coPad); PipeBarrier<PIPE_V>(); }
            PipeBarrier<PIPE_ALL>();
            if constexpr(sizeof(T)==2) Cast(yr,y,RoundMode::CAST_RINT,t.coPad); else DataCopy(yr,y,t.coPad);
            PipeBarrier<PIPE_ALL>(); DataCopyExtParams p{1,t.co*uint32_t(sizeof(T)),0,0,0}; DataCopyPad(yg[node*t.co],yr,p); PipeBarrier<PIPE_ALL>();
        }
    }
private:
    SplineConvTilingData t;
    bool useCache; uint32_t cacheSize;
    TBuf<TPosition::VECCALC> dotBuf,broadcastBuf;
    LocalTensor<float> dot,broadcast;
    TPipe pipe;
    TBuf<TPosition::VECCALC> edgeBuf,pseudoBuf,nextBuf;
    LocalTensor<int64_t> edgeLocal; LocalTensor<int32_t> nextLocal; LocalTensor<T> pseudoLocal; LocalTensor<uint16_t> pseudoBits;
    TBuf<TPosition::VECCALC> headBuf,aggBuf,wBuf,rawBuf,xBuf,xRawBuf,yBuf,yRawBuf,compBuf,tempBuf,cacheBuf,cacheRawBuf,activeBuf;
    LocalTensor<uint32_t> activeFlags;
    LocalTensor<int32_t> h; LocalTensor<float> agg,mat,xv,y,comp,prod,z,sum,cached; LocalTensor<T> raw,xr,yr,cacheRaw;
    GlobalTensor<T> xg,pg,wg,rg,bg,yg; GlobalTensor<uint16_t> pbits; GlobalTensor<int64_t> eg; GlobalTensor<int32_t> next;
};
