#pragma once
// Process the live rows of each pivot pair, preserving 32-byte vector alignment.
template<uint32_t N,bool PAIRED=false> class DetStaticSequentialVector {
public:
    __aicore__ inline DetStaticSequentialVector(){}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR det, GM_ADDR ws,
                                const LinalgDetConfig &cfg) {
        t=cfg;
        const uint32_t logical=(GetBlockIdx()/(PAIRED?2:1))%((N+63)/64);
        q=logical;
        group=GetBlockIdx()/((PAIRED?2:1)*((N+63)/64)); first=q*64;
        input.SetGlobalBuffer((__gm__ float*)A);
        output.SetGlobalBuffer((__gm__ float*)det);
        allShared.SetGlobalBuffer((__gm__ float*)ws);
        shared=allShared[group*t.robustOffset];
        comm=shared[((N+31)/32*32)*64];
        own=allShared[group*t.robustOffset+2*(((N+31)/32*32)*64+128)+q*(((N+31)/32*32)*64+8192)];
        prodGm=own[4096];
        constexpr uint32_t PN=(N+31)/32*32,VP=(PN+63)/64*64;
        constexpr uint32_t panelBytes=(PN+32)*104*4,rawBytes=PN*72*4,vecBytes=5*VP*4;
        panel=LocalTensor<float>(TPosition::VECCALC,0,panelBytes/4);
        raw=LocalTensor<float>(TPosition::VECCALC,panelBytes,rawBytes/4);
        col=LocalTensor<float>(TPosition::VECCALC,panelBytes+rawBytes,5*VP);
        row=col[VP];meta=row[VP];temp=meta[VP];
        top=LocalTensor<float>(TPosition::VECCALC,panelBytes+rawBytes+vecBytes,32*32);
        idx=LocalTensor<uint32_t>(TPosition::VECCALC,panelBytes+rawBytes+vecBytes+32*32*4,VP);
    }
    template<HardEvent E> __aicore__ inline void Fence(){
        SetFlag<E>(0); WaitFlag<E>(0);
    }
    __aicore__ inline uint32_t Min(uint32_t a,uint32_t b){return a<b?a:b;}
    __aicore__ inline void NativeMask(uint32_t n){set_vector_mask(0,n>=64?~uint64_t(0):(uint64_t(1)<<n)-1);}
    __aicore__ inline float Pow2(int32_t e){
        union{uint32_t u;float f;}x; x.u=uint32_t(e+127)<<23; return x.f;
    }
    __aicore__ inline void Acc(float p,float &m,int32_t &e){
        union{uint32_t u;float f;}x; x.f=p;
        e+=int32_t((x.u>>23)&255)-127; x.u=(x.u&0x807fffffU)|(127U<<23); m*=x.f;
        x.f=m; e+=int32_t((x.u>>23)&255)-127; x.u=(x.u&0x807fffffU)|(127U<<23); m=x.f;
    }
    __aicore__ inline void NativeGather(__ubuf__ float *dst,uint32_t base,uint32_t rows){
        auto ip=(__ubuf__ uint32_t*)idx.GetPhyAddr();const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vgather((__ubuf__ uint32_t*)dst,ip,base,8,uint8_t(full));}
        if(tail){NativeMask(tail);vgather((__ubuf__ uint32_t*)(dst+full*64),ip+full*64,base,8,1);}
    }
    __aicore__ inline void NativeMuls(__ubuf__ float *dst,__ubuf__ float *src,float value,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vmuls(dst,src,float(value),uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vmuls(dst+full*64,src+full*64,float(value),1,1,1,8,8);}
    }
    __aicore__ inline void NativeAxpy(__ubuf__ float *dst,__ubuf__ float *src,float value,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vaxpy(dst,src,float(value),uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vaxpy(dst+full*64,src+full*64,float(value),1,1,1,8,8);}
    }
    __aicore__ inline void NativeCopy(__ubuf__ float *dst,__ubuf__ float *src,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vadds(dst,src,0.0f,uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vadds(dst+full*64,src+full*64,0.0f,1,1,1,8,8);}
    }
    __aicore__ inline void GatherPair(__ubuf__ float *d0,__ubuf__ float *d1,uint32_t base,uint32_t rows){
        auto ip=(__ubuf__ uint32_t*)idx.GetPhyAddr();NativeMask(64);
        vgather((__ubuf__ uint32_t*)d0,ip,base,8,uint8_t((rows+63)/64));
        vgather((__ubuf__ uint32_t*)d1,ip,base+4,8,uint8_t((rows+63)/64));
    }
    __aicore__ inline void MulsPair(__ubuf__ float *d0,__ubuf__ float *d1,__ubuf__ float *src,float v0,float v1,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vmuls(d0,src,v0,uint8_t(full),1,1,8,8);vmuls(d1,src,v1,uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vmuls(d0+full*64,src+full*64,v0,1,1,1,8,8);vmuls(d1+full*64,src+full*64,v1,1,1,1,8,8);}
    }
    __aicore__ inline void AxpyPair(__ubuf__ float *d0,__ubuf__ float *d1,__ubuf__ float *src,float v0,float v1,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vaxpy(d0,src,v0,uint8_t(full),1,1,8,8);vaxpy(d1,src,v1,uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vaxpy(d0+full*64,src+full*64,v0,1,1,1,8,8);vaxpy(d1+full*64,src+full*64,v1,1,1,1,8,8);}
    }
    __aicore__ inline void NarrowMla(__ubuf__ float *dst,__ubuf__ float *upper,__ubuf__ float *weights,uint32_t rows,uint32_t columns){
        if(columns>8){NativeMask(columns);vmla(dst,upper,weights,uint8_t(rows),1,1,0,13,0,13);return;}
        const uint32_t full=rows/8,tail=rows%8;
        if(full){
            NativeMask(64);
            #pragma unroll 1
            for(uint32_t c=0;c<columns;c+=8)
                vmla(dst+c,upper+c,weights,uint8_t(full),13,0,13,104,0,104);
        }
        // Remaining rows use the original row-major mode in one instruction.
        if(tail){
            NativeMask(columns);
            vmla(dst+full*8*104,upper,weights+full*8*104,uint8_t(tail),1,1,0,13,0,13);
        }
    }
    __aicore__ inline void FactorCoop200(uint32_t start){
        constexpr bool COOPERATIVE=true;
        SetMaskNorm();
        float m=1.0f;int32_t e=0;uint32_t riskMask=0;
        const uint32_t rowEnd=COOPERATIVE?start+32:t.n;
        const uint32_t rows=rowEnd-start,steps=COOPERATIVE?32U:Min((t.n==256&&start==192)?64U:32U,t.n-start),local=start%64;
        const bool saveLower=COOPERATIVE||start+steps<t.n;
        const bool splitColumns=local==0&&saveLower;
        auto pp=(__ubuf__ float*)panel.GetPhyAddr();auto c0=(__ubuf__ float*)col.GetPhyAddr();
        auto c1=(__ubuf__ float*)row.GetPhyAddr();
        auto lower=(__ubuf__ float*)raw.GetPhyAddr();auto expanded=(__ubuf__ uint32_t*)(pp+start*104+64);
        for(uint32_t i=0;i<steps;i+=2){
            const uint32_t k=start+i;
            auto v0=lower+i*t.paddedN+start;auto second=lower+(i+1)*t.paddedN+start;
            if(i==0)Fence<HardEvent::V_S>();else WaitFlag<HardEvent::V_S>(0);
            float p;bool pivotReady=false;
            if(i+1==steps){
                p=pp[k*104+local+i];union{float f;uint32_t u;}bits;bits.f=p;const uint32_t ap=bits.u&0x7fffffffU;
                riskMask|=uint32_t(ap<0x38d1b717U)|uint32_t(ap>=0x60ad78ecU);
            }
            else{
                const bool needLower=saveLower||i+2<steps;
                if(needLower){GatherPair(c0,c1,uint32_t(uint64_t(pp)+(start*104+local+i)*4),rows);PipeBarrier<PIPE_V>();}
                const float a=pp[k*104+local+i],b=pp[k*104+local+i+1],c=pp[(k+1)*104+local+i],d=pp[(k+1)*104+local+i+1];
                p=a*d-b*c;union{float f;uint32_t u;}bits;bits.f=p;const uint32_t ap=bits.u&0x7fffffffU;
                riskMask|=uint32_t(ap<0x3d800000U)|uint32_t(ap>=0x7149f2caU);
                if(needLower){
                    const float inv=1.0f/p;
                    if(rows>64&&rows%64!=0){
                        SetMaskCount();set_vector_mask(0,uint64_t(rows));
                        vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(second,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
                        vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(second,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
                        SetMaskNorm();
                    }else{
                        MulsPair(v0,second,c0,-d*inv,b*inv,rows);PipeBarrier<PIPE_V>();
                        AxpyPair(v0,second,c1,c*inv,-a*inv,rows);PipeBarrier<PIPE_V>();
                    }
                }
                if(i+2<steps){
                    set_vector_mask(~uint64_t(0),~uint64_t(0));vbrcb(expanded,(__ubuf__ uint32_t*)v0,13,104,(rows+7)/8);
                    vbrcb(expanded+8,(__ubuf__ uint32_t*)second,13,104,(rows+7)/8);PipeBarrier<PIPE_V>();
                    const uint32_t off=(local+i+2)/8*8,boundary=Min(start+steps,t.n),upper=boundary-k-2;
                    if(splitColumns){
                        NativeMask(32-off);
                        vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(i+2)*104,rowEnd-k-2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(i+2)*104,rowEnd-k-2,1,1,0,13,0,13);
                        SetFlag<HardEvent::V_S>(0);pivotReady=true;
                        NativeMask(32);
                        vmla(pp+(k+2)*104+32,pp+k*104+32,(__ubuf__ float*)expanded+(i+2)*104,upper,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+32,pp+(k+1)*104+32,(__ubuf__ float*)expanded+8+(i+2)*104,upper,1,1,0,13,0,13);
                    }else if(local==32){
                        NativeMask(64-off);
                        vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(i+2)*104,rowEnd-k-2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(i+2)*104,rowEnd-k-2,1,1,0,13,0,13);
                    }else{
                        if(upper){
                            NativeMask(64-off);
                            vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(i+2)*104,upper,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                            vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(i+2)*104,upper,1,1,0,13,0,13);
                            SetFlag<HardEvent::V_S>(0);pivotReady=true;
                        }
                        if(boundary<rowEnd&&off<local+32){
                            NativeMask(local+32-off);
                            vmla(pp+boundary*104+off,pp+k*104+off,(__ubuf__ float*)expanded+32*104,rowEnd-boundary,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                            vmla(pp+boundary*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+32*104,rowEnd-boundary,1,1,0,13,0,13);
                        }
                    }
                }
            }
            m*=p;
            if((i&7)==6||i+2>=steps){
                union{uint32_t u;float f;}x;x.f=m;const uint32_t am=x.u&0x7fffffffU;
                riskMask|=uint32_t(am<=0x0da24260U)|uint32_t(am>=0x7149f2caU);
                e+=int32_t((x.u>>23)&255)-127;x.u=(x.u&0x807fffffU)|(127U<<23);m=x.f;
            }
            if(pivotReady){if(!splitColumns)PipeBarrier<PIPE_V>();}
            else SetFlag<HardEvent::V_S>(0);
        }
        WaitFlag<HardEvent::V_S>(0);
        const bool risky=riskMask!=0;
        SetMaskNorm();ResetMask();ownRisk=ownRisk||risky;ownedM*=m;ownedE+=e;Fence<HardEvent::V_S>();meta.SetValue(0,ownedM);meta.ReinterpretCast<int32_t>().SetValue(1,ownedE);meta.SetValue(2,ownRisk?1.0f:0.0f);meta.SetValue(3,ownRisk?1.0f:0.0f);
        if(saveLower){
            if(COOPERATIVE){
                DataCopyParams pack{uint16_t(t.n-start),4,9,0};
                DataCopy(raw[32*t.paddedN],panel[start*104+local],pack);
            }
            Fence<HardEvent::V_MTE3>();
            if(COOPERATIVE){
                DataCopyExtParams upper{32,128,(t.paddedN-32)/8,(t.paddedN-32)*4,0};
                DataCopyPad(shared[start],raw[start],upper);
                DataCopy(shared[32*t.paddedN+start*32],raw[32*t.paddedN],(t.n-start)*32);
            }else {const uint32_t rows=t.n-start;DataCopyExtParams copy{32,rows*4,(t.paddedN-(rows+7)/8*8)/8,(t.paddedN-rows)*4,0};DataCopyPad(shared[start],raw[start],copy);}
            if(local==0){DataCopyExtParams cp{32,256,5,0,0};DataCopyPad(own,panel[start*104],cp);}
        }
        Fence<HardEvent::S_MTE3>();DataCopy(comm,meta,8);
        if(start+steps>=Min(first+64,t.n))DataCopy(allShared[group*t.robustOffset+t.paddedN*64+64+q*8],meta,8);
        if constexpr(COOPERATIVE){
            auto signal=SignalBuffer();Duplicate(signal,0,32);Fence<HardEvent::V_MTE3>();
            DataCopy(LowerSignalGM(0),signal,32);
        }
    }
    __aicore__ inline LocalTensor<int32_t> SignalBuffer(){
        // Helper meta reduction uses at most160 floats;192..223 are disjoint.
        
        return col[4*((t.paddedN+63)/64*64)].ReinterpretCast<int32_t>();
    }
    __aicore__ inline GlobalTensor<int32_t> LowerSignalGM(uint32_t qoffset){
        GlobalTensor<int32_t> gm;gm.SetGlobalBuffer((__gm__ int32_t*)comm[96+qoffset*8].GetPhyAddr());return gm;
    }
    __aicore__ inline void PublishLowerReady(){
        Fence<HardEvent::MTE3_S>();auto signal=SignalBuffer();signal.SetValue(0,1);
        Fence<HardEvent::S_MTE3>();DataCopy(LowerSignalGM(q),signal,8);
    }
    __aicore__ inline void WaitLowerReady(){
        auto signal=SignalBuffer();Fence<HardEvent::MTE3_MTE2>();
        do{DataCopy(signal,LowerSignalGM(0),32);Fence<HardEvent::MTE2_S>();}
        while(signal.GetValue(0)!=1||signal.GetValue(8)!=1||signal.GetValue(16)!=1||signal.GetValue(24)!=1);
    }
    __aicore__ inline void Factor(uint32_t start){
        SetMaskNorm();
        float m=1.0f;int32_t e=0;uint32_t riskMask=0;
        const uint32_t steps=Min(32U,N-start),local=start%64;
        const bool saveLower=start+32<N;
        const bool splitColumns=local==0&&saveLower;
        auto pp=(__ubuf__ float*)panel.GetPhyAddr();auto c0=(__ubuf__ float*)col.GetPhyAddr();
        auto c1=(__ubuf__ float*)row.GetPhyAddr();
        auto lower=(__ubuf__ float*)raw.GetPhyAddr();
        #pragma unroll 1
        for(uint32_t i=0;i<steps;i+=2){
            const uint32_t k=start+i,rowBase=(k+2)/8*8,rows=N-rowBase;
            auto expanded=(__ubuf__ uint32_t*)(pp+rowBase*104+64);
            auto v0=lower+i*((N+31)/32*32)+rowBase;auto second=lower+(i+1)*((N+31)/32*32)+rowBase;
            if(i==0)Fence<HardEvent::V_S>();else WaitFlag<HardEvent::V_S>(0);
            float p;bool pivotReady=false;
            if(i+1==steps){
                p=pp[k*104+local+i];union{float f;uint32_t u;}bits;bits.f=p;const uint32_t ap=bits.u&0x7fffffffU;
                riskMask|=uint32_t(ap<0x38d1b717U)|uint32_t(ap>=0x60ad78ecU);
            }
            else{
                const bool needLower=saveLower||i+2<steps;
                if(needLower){GatherPair(c0,c1,uint32_t(uint64_t(pp)+(rowBase*104+local+i)*4),rows);PipeBarrier<PIPE_V>();}
                const float a=pp[k*104+local+i],b=pp[k*104+local+i+1],c=pp[(k+1)*104+local+i],d=pp[(k+1)*104+local+i+1];
                p=a*d-b*c;union{float f;uint32_t u;}bits;bits.f=p;const uint32_t ap=bits.u&0x7fffffffU;
                riskMask|=uint32_t(ap<0x3d800000U)|uint32_t(ap>=0x7149f2caU);
                if(needLower){
                    const float inv=1.0f/p;
                    SetMaskCount();set_vector_mask(0,uint64_t(rows));
                    vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(second,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
                    vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(second,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
                    SetMaskNorm();
                    if(saveLower&&(i&7)==6){
                        // Publish each finished8-column chunk while the following pairs compute.
                        // Separate event IDs1..4 avoid reusing an unconsumed signal.
                        const uint8_t event=uint8_t(1+i/8);
                        SetFlag<HardEvent::V_MTE3>(event);WaitFlag<HardEvent::V_MTE3>(event);
                        DataCopy(shared[(i-6)*((N+31)/32*32)],raw[(i-6)*((N+31)/32*32)],8*((N+31)/32*32));
                    }
                }
                if(i+2<steps){
                    set_vector_mask(~uint64_t(0),~uint64_t(0));vbrcb(expanded,(__ubuf__ uint32_t*)v0,13,104,(rows+7)/8);
                    vbrcb(expanded+8,(__ubuf__ uint32_t*)second,13,104,(rows+7)/8);PipeBarrier<PIPE_V>();
                    const uint32_t off=(local+i+2)/8*8,boundary=Min(start+32,N),upper=boundary-k-2;
                    if(splitColumns){
                        NarrowMla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(k+2-rowBase)*104,N-k-2,32-off);PipeBarrier<PIPE_V>();
                        NarrowMla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(k+2-rowBase)*104,N-k-2,32-off);
                        SetFlag<HardEvent::V_S>(0);pivotReady=true;
                        NativeMask(32);
                        vmla(pp+(k+2)*104+32,pp+k*104+32,(__ubuf__ float*)expanded+(k+2-rowBase)*104,upper,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+32,pp+(k+1)*104+32,(__ubuf__ float*)expanded+8+(k+2-rowBase)*104,upper,1,1,0,13,0,13);
                    }else if(local==32){
                        NarrowMla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(k+2-rowBase)*104,N-k-2,64-off);PipeBarrier<PIPE_V>();
                        NarrowMla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(k+2-rowBase)*104,N-k-2,64-off);
                    }else{
                        if(upper){
                            NativeMask(64-off);
                            vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(k+2-rowBase)*104,upper,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                            vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(k+2-rowBase)*104,upper,1,1,0,13,0,13);
                            SetFlag<HardEvent::V_S>(0);pivotReady=true;
                        }
                        if(boundary<N&&off<local+32){
                            NativeMask(local+32-off);
                            vmla(pp+boundary*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(start+32-rowBase)*104,N-boundary,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                            vmla(pp+boundary*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(start+32-rowBase)*104,N-boundary,1,1,0,13,0,13);
                        }
                    }
                }
            }
            m*=p;
            if((i&7)==6||i+2>=steps){
                union{uint32_t u;float f;}x;x.f=m;const uint32_t am=x.u&0x7fffffffU;
                riskMask|=uint32_t(am<=0x0da24260U)|uint32_t(am>=0x7149f2caU);
                e+=int32_t((x.u>>23)&255)-127;x.u=(x.u&0x807fffffU)|(127U<<23);m=x.f;
            }
            if(pivotReady){if(!splitColumns)PipeBarrier<PIPE_V>();}
            else SetFlag<HardEvent::V_S>(0);
        }
        WaitFlag<HardEvent::V_S>(0);
        const bool risky=riskMask!=0;
        SetMaskNorm();ResetMask();ownRisk=ownRisk||risky;ownedM*=m;ownedE+=e;
        if(saveLower){
            Fence<HardEvent::V_MTE3>();
            DataCopyExtParams cp{32,256,5,0,0};DataCopyPad(own,panel[start*104],cp);
        }
        if(start+32>=Min(first+64,N)){
            Fence<HardEvent::V_S>();meta.SetValue(0,ownedM);meta.ReinterpretCast<int32_t>().SetValue(1,ownedE);meta.SetValue(2,ownRisk?1.0f:0.0f);meta.SetValue(3,ownRisk?1.0f:0.0f);
            Fence<HardEvent::S_MTE3>();DataCopy(allShared[group*t.robustOffset+((N+31)/32*32)*64+64+q*8],meta,8);
        }
    }
    __aicore__ inline void Finish72(uint32_t start){
        SetMaskNorm();float m=1.0f;int32_t e=0;uint32_t risk=0;
        auto p=(__ubuf__ float*)panel.GetPhyAddr();auto c0=(__ubuf__ float*)col.GetPhyAddr();auto c1=(__ubuf__ float*)row.GetPhyAddr();auto lower=(__ubuf__ float*)raw.GetPhyAddr();
        #pragma unroll 1
        for(uint32_t i=0;i<72;i+=2){
            const uint32_t k=start+i,rowBase=(k+2)/8*8,rows=N-rowBase;
            if(i==0)Fence<HardEvent::V_S>();else WaitFlag<HardEvent::V_S>(0);
            const float a=p[k*104+i],b=p[k*104+i+1],c=p[(k+1)*104+i],d=p[(k+1)*104+i+1];
            const float pivot=a*d-b*c;union{float f;uint32_t u;}bits;bits.f=pivot;const uint32_t mag=bits.u&0x7fffffffU;
            risk|=uint32_t(mag<0x3d800000U)|uint32_t(mag>=0x7149f2caU);
            if(i+2<72){
                GatherPair(c0,c1,uint32_t(uint64_t(p)+(rowBase*104+i)*4),rows);PipeBarrier<PIPE_V>();
                auto v0=lower+i*((N+31)/32*32)+rowBase;auto v1=v0+((N+31)/32*32);const float inv=1.0f/pivot;
                SetMaskCount();set_vector_mask(0,uint64_t(rows));
                vmuls(v0,c0,-d*inv,1,1,1,8,8);vmuls(v1,c0,b*inv,1,1,1,8,8);PipeBarrier<PIPE_V>();
                vaxpy(v0,c1,c*inv,1,1,1,8,8);vaxpy(v1,c1,-a*inv,1,1,1,8,8);PipeBarrier<PIPE_V>();SetMaskNorm();
                auto ex=(__ubuf__ uint32_t*)(p+rowBase*104+80);set_vector_mask(~uint64_t(0),~uint64_t(0));
                vbrcb(ex,(__ubuf__ uint32_t*)v0,13,104,(rows+7)/8);vbrcb(ex+8,(__ubuf__ uint32_t*)v1,13,104,(rows+7)/8);PipeBarrier<PIPE_V>();
                // Only the first three pivot pairs span both 64 and 8 columns.
                // All later pairs retain one mask across early rows and tail.
                const uint32_t off=(i+2)/8*8;
                if(off==0){
                    NativeMask(64);
                    vmla(p+(k+2)*104,p+k*104,(__ubuf__ float*)ex+(k+2-rowBase)*104,2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                    vmla(p+(k+2)*104,p+(k+1)*104,(__ubuf__ float*)ex+8+(k+2-rowBase)*104,2,1,1,0,13,0,13);
                    NativeMask(8);
                    vmla(p+(k+2)*104+64,p+k*104+64,(__ubuf__ float*)ex+(k+2-rowBase)*104,2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                    vmla(p+(k+2)*104+64,p+(k+1)*104+64,(__ubuf__ float*)ex+8+(k+2-rowBase)*104,2,1,1,0,13,0,13);
                    SetFlag<HardEvent::V_S>(0);
                    if(k+4<N){
                        NativeMask(64);
                        vmla(p+(k+4)*104,p+k*104,(__ubuf__ float*)ex+(k+4-rowBase)*104,N-k-4,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(p+(k+4)*104,p+(k+1)*104,(__ubuf__ float*)ex+8+(k+4-rowBase)*104,N-k-4,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        NativeMask(8);
                        vmla(p+(k+4)*104+64,p+k*104+64,(__ubuf__ float*)ex+(k+4-rowBase)*104,N-k-4,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(p+(k+4)*104+64,p+(k+1)*104+64,(__ubuf__ float*)ex+8+(k+4-rowBase)*104,N-k-4,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                    }
                }else{
                    NativeMask(72-off);
                    vmla(p+(k+2)*104+off,p+k*104+off,(__ubuf__ float*)ex+(k+2-rowBase)*104,2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                    vmla(p+(k+2)*104+off,p+(k+1)*104+off,(__ubuf__ float*)ex+8+(k+2-rowBase)*104,2,1,1,0,13,0,13);
                    SetFlag<HardEvent::V_S>(0);
                    if(k+4<N){
                        vmla(p+(k+4)*104+off,p+k*104+off,(__ubuf__ float*)ex+(k+4-rowBase)*104,N-k-4,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(p+(k+4)*104+off,p+(k+1)*104+off,(__ubuf__ float*)ex+8+(k+4-rowBase)*104,N-k-4,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                    }
                }
            }
            if(i+2>=72)SetFlag<HardEvent::V_S>(0);m*=pivot;
            if((i&7)==6||i+2>=72){union{uint32_t u;float f;}v;v.f=m;const uint32_t am=v.u&0x7fffffffU;risk|=uint32_t(am<=0x0da24260U)|uint32_t(am>=0x7149f2caU);e+=int32_t((v.u>>23)&255)-127;v.u=(v.u&0x807fffffU)|(127U<<23);m=v.f;}
        }
        WaitFlag<HardEvent::V_S>(0);ownRisk=ownRisk||risk!=0;ownedM*=m;ownedE+=e;ResetMask();meta.SetValue(0,ownedM);meta.ReinterpretCast<int32_t>().SetValue(1,ownedE);meta.SetValue(2,risk?1.0f:0.0f);meta.SetValue(3,ownRisk?1.0f:0.0f);
        Fence<HardEvent::S_MTE3>();DataCopy(allShared[group*t.robustOffset+((N+31)/32*32)*64+64+q*8],meta,8);
    }
    __aicore__ inline void Solve(uint32_t start){
        DataCopyExtParams cp{32,128,(((N+31)/32*32)-32)*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(top,shared[start],cp,pad);Fence<HardEvent::MTE2_V>();
        auto pp=(__ubuf__ float*)panel.GetPhyAddr();auto lower=(__ubuf__ uint32_t*)top.GetPhyAddr();auto expanded=(__ubuf__ uint32_t*)(pp+start*104+64);
        SetMaskNorm();
        for(uint32_t i=0;i+2<32;i+=2){
            set_vector_mask(~uint64_t(0),~uint64_t(0));
            vbrcb(expanded,lower+i*32,13,104,4);vbrcb(expanded+8,lower+(i+1)*32,13,104,4);PipeBarrier<PIPE_V>();NativeMask(64);
            vmla(pp+(start+i+2)*104,pp+(start+i)*104,(__ubuf__ float*)expanded+(i+2)*104,30-i,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
            vmla(pp+(start+i+2)*104,pp+(start+i+1)*104,(__ubuf__ float*)expanded+8+(i+2)*104,30-i,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();ResetMask();Fence<HardEvent::V_MTE3>();DataCopyExtParams store{32,256,5,0,0};DataCopyPad(own,panel[start*104],store);Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void Update(uint32_t start){
        const uint32_t off=q==start/64?start%64+32:0;
        if(off>=64||start+32>=N)return;
        const uint32_t rows=N-start-32;
        DataCopy(raw,prodGm,rows*64);Fence<HardEvent::MTE2_V>();
        Add(panel[(start+32)*104+off],panel[(start+32)*104+off],raw[off],64-off,rows,{1,1,1,13,13,8});Fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline bool ProcessOne(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        const uint32_t vectorPitch=(((N+31)/32*32)+63)/64*64;
        SetMaskNorm();auto indexWords=(__ubuf__ int32_t*)idx.GetPhyAddr();
        #pragma unroll
        for(uint32_t j=0;j<8;++j)indexWords[j]=int32_t(j*416);
        Fence<HardEvent::S_V>();
        #pragma unroll
        for(uint32_t step=8;step<vectorPitch;step*=2){
            NativeMask(Min(step,64U));
            vadds(indexWords+step,indexWords,int32_t(step*416),uint8_t((step+63)/64),1,1,8,8);PipeBarrier<PIPE_V>();
        }
        Duplicate(panel[((N+31)/32*32)*104],0.0f,32*104);
        if(batch>=t.batch||first+64>N)Duplicate(panel,0.0f,((N+31)/32*32)*104);
        else if(((N+31)/32*32)>N)Duplicate(panel[N*104],0.0f,(((N+31)/32*32)-N)*104);
        Fence<HardEvent::V_S>();
        for(uint32_t r=batch<t.batch?N:0;r<((N+31)/32*32);++r)
            if(r>=first&&r<first+64)panel.SetValue(r*104+r-first,1.0f);
        Fence<HardEvent::S_MTE2>();
        if(first<N&&batch<t.batch){
            const uint32_t width=Min(64,N-first),aligned=(width+7)/8*8;
            DataCopyExtParams cp{uint16_t(N),width*4,(N-width)*4,(104-aligned)/8,0};
            DataCopyPadExtParams<float> pad{true,0,uint8_t(aligned-width),0};
            DataCopyPad(panel,input[uint64_t(batch)*N*N+first],cp,pad);
            Fence<HardEvent::MTE2_V>();
        }
        magnitude=1.0f;
        float m=1.0f;int32_t e=0;ownRisk=false;ownedM=1.0f;ownedE=0;
        #pragma unroll
        for(uint32_t start=0;start<N;start+=32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(((N+31)/32*32)*64+128)];comm=shared[((N+31)/32*32)*64];
            if constexpr(N==200){
                if(start==128){
                    if(q==3){
                        Fence<HardEvent::V_MTE3>();DataCopyExtParams tail{72,32,12,0,0};DataCopyPad(own,panel[128*104],tail);
                        Fence<HardEvent::V_S>();meta.SetValue(0,1.0f);meta.ReinterpretCast<int32_t>().SetValue(1,0);meta.SetValue(2,0.0f);meta.SetValue(3,ownRisk?1.0f:0.0f);Fence<HardEvent::S_MTE3>();
                        DataCopy(allShared[group*t.robustOffset+((N+31)/32*32)*64+64+q*8],meta,8);
                    }
                    CrossCoreSetFlag<0,PIPE_MTE3>(12);CrossCoreWaitFlag(12);
                    if(q==2){
                        auto peer=allShared[group*t.robustOffset+2*(((N+31)/32*32)*64+128)+3*(((N+31)/32*32)*64+8192)];
                        DataCopyExtParams tail{72,32,0,12,0};DataCopyPadExtParams<float> pad{false,0,0,0};DataCopyPad(panel[128*104+64],peer,tail,pad);Fence<HardEvent::MTE2_V>();
                    }
                }
            }
            if(q==owner){if constexpr(N==200){if(start==128)Finish72(start);else if constexpr(PAIRED){if(start<128)FactorCoop200(start);else Factor(start);}else Factor(start);}else Factor(start);}
            {CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
            if constexpr(N==200){if(start==128)break;}
            if(start+32<N){
                const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
                // Factor's global MTE3 barrier has published the immutable negative L.
                // Only A can be fetched now; B remains protected by flag8 after Solve.
                if constexpr(N==200){if(cubeNeeded)CrossCoreSetFlag<2,PIPE_MTE3>(11);}
                if(q>owner)Solve(start);
                if(cubeNeeded){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Update(start);}
            }
        }
        DataCopyExtParams flags{uint16_t(t.groups),((N+63)/64)*32,t.robustOffset*4-((N+63)/64)*32,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(meta,allShared[((N+31)/32*32)*64+64],flags,pad);Fence<HardEvent::MTE2_S>();
        bool safe=true;for(uint32_t c=0;c<t.groups*((N+63)/64);++c)safe=safe&&meta.GetValue(c*8+3)==0.0f;
        if(!safe)return false;
        if(q==0&&batch<t.batch){
            m=1.0f;e=0;
            for(uint32_t c=0;c<(N+63)/64;++c){const uint32_t at=(group*((N+63)/64)+c)*8;m*=meta.GetValue(at);e+=meta.ReinterpretCast<int32_t>().GetValue(at+1);}
            while(e>120){m*=Pow2(120);e-=120;} while(e< -120){m*=Pow2(-120);e+=120;}
            Fence<HardEvent::V_S>(); meta.SetValue(0,m*Pow2(e)); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams cp{1,4,0,0,0}; DataCopyPad(output[batch],meta,cp);
            Fence<HardEvent::MTE3_S>();
        }
        return true;
    }
private:
    LocalTensor<float> panel,raw,col,row,meta,temp,top;
    LocalTensor<uint32_t> idx;
    GlobalTensor<float> input,output,shared,comm,own,prodGm,allShared;
    LinalgDetConfig t; uint32_t q,first,group; float magnitude;bool ownRisk;float ownedM;int32_t ownedE;
};


