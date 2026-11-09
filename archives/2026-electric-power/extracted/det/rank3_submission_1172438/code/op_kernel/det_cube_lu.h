#pragma once
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"
using namespace AscendC;
using namespace matmul;
using AB = MatmulType<TPosition::GM, CubeFormat::ND, float>;
__aicore__ constexpr MatmulConfig DetFixedCubeConfig(){
    auto cfg=GetNormalConfig();
    cfg.singleCoreM=256;cfg.basicM=256;cfg.singleCoreN=64;cfg.basicN=64;cfg.singleCoreK=32;cfg.basicK=32;
    cfg.enableSetBias=false;cfg.enableQuantVector=false;cfg.enableSetDefineData=false;
    cfg.iterateMode=IterateMode::ITERATE_MODE_ALL;return cfg;
}
constexpr MatmulConfig DET_FIXED_CUBE_CFG=DetFixedCubeConfig();


class CubeLUVector {
public:
    __aicore__ inline CubeLUVector(TPipe &external):pipe(external){}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR det, GM_ADDR ws,
                                const LinalgDetConfig &cfg) {
        t=cfg;
        const uint32_t logical=GetBlockIdx()%t.coresPerMatrix;
        q=logical;
        group=GetBlockIdx()/t.coresPerMatrix; first=q*64;
        input.SetGlobalBuffer((__gm__ float*)A);
        output.SetGlobalBuffer((__gm__ float*)det);
        allShared.SetGlobalBuffer((__gm__ float*)ws);
        shared=allShared[group*t.robustOffset];
        comm=shared[t.paddedN*64];
        own=allShared[group*t.robustOffset+2*(t.paddedN*64+128)+q*(t.paddedN*64+8192)];
        prodGm=own[4096];
        pipe.InitBuffer(panelBuf,(t.paddedN+32)*104*4); panel=panelBuf.Get<float>();
        pipe.InitBuffer(rawBuf,t.paddedN*64*4); raw=rawBuf.Get<float>();
        const uint32_t vectorPitch=(t.paddedN+63)/64*64;
        pipe.InitBuffer(vecBuf,5*vectorPitch*4); col=vecBuf.Get<float>();
        row=col[vectorPitch]; meta=row[vectorPitch]; temp=meta[vectorPitch];
        pipe.InitBuffer(topBuf,32*32*4); top=topBuf.Get<float>();
        pipe.InitBuffer(idxBuf,vectorPitch*4); idx=idxBuf.Get<uint32_t>();
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
    template<bool COOPERATIVE> __aicore__ inline void Factor(uint32_t start){
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
        SetMaskNorm();ResetMask();Fence<HardEvent::V_S>();meta.SetValue(0,m);meta.ReinterpretCast<int32_t>().SetValue(1,e);meta.SetValue(2,risky?1.0f:0.0f);ownRisk=ownRisk||risky;meta.SetValue(3,ownRisk?1.0f:0.0f);
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
    }
    __aicore__ inline void Solve(uint32_t start){
        DataCopyExtParams cp{32,128,(t.paddedN-32)*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
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
    __aicore__ inline void FactorTail(uint32_t start){
        const uint32_t remaining=t.n-start-32;
        const uint32_t pitch=((remaining+t.coresPerMatrix-1)/t.coresPerMatrix+31)/32*32;
        const uint32_t begin=start+32+q*pitch;
        if(begin>=t.n)return;
        const uint32_t rows=Min(pitch,t.n-begin),columnPitch=pitch + 8;
        auto scratch=raw;
        auto lower=(__ubuf__ float*)scratch.GetPhyAddr();
        auto matrix=lower+32*pitch;
        const uint32_t packedOffset=32*pitch+32*columnPitch;
        auto packed=lower+packedOffset;
        auto expanded=(__ubuf__ uint32_t*)(packed+32*pitch);
        auto u=(__ubuf__ float*)top.GetPhyAddr();
        auto offsets=(__ubuf__ uint32_t*)temp.GetPhyAddr();
        DataCopy(scratch[packedOffset],shared[32*t.paddedN+begin*32],rows*32);
        DataCopy(top,shared[32*t.paddedN+start*32],32*32);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::V_S>();SetMaskNorm();NativeMask(rows);
        for(uint32_t c=0;c<32;++c)
            vgather((__ubuf__ uint32_t*)(matrix+c*columnPitch),offsets,uint32_t(uint64_t(packed)+c*4),8,1);
        PipeBarrier<PIPE_V>();
        for(uint32_t i=0;i<32;i+=2){
            NativeMask(rows);
            auto c0=matrix+i*columnPitch;auto c1=c0+columnPitch;
            const float a=u[i*32+i],b=u[i*32+i+1],c=u[(i+1)*32+i],d=u[(i+1)*32+i+1];
            const float inv=1.0f/(a*d-b*c);
            auto v0=lower+i*pitch;auto v1=v0+pitch;
            vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(v1,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(v1,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            if(i+2<32){
                set_vector_mask(~uint64_t(0),~uint64_t(0));
                vbrcb(expanded,(__ubuf__ uint32_t*)(u+i*32),1,8,4);
                vbrcb(expanded+256,(__ubuf__ uint32_t*)(u+(i+1)*32),1,8,4);PipeBarrier<PIPE_V>();
                NativeMask(rows);
                vmla(matrix+(i+2)*columnPitch,v0,(__ubuf__ float*)expanded+(i+2)*8,30-i,1,1,0,columnPitch/8,0,1);PipeBarrier<PIPE_V>();
                vmla(matrix+(i+2)*columnPitch,v1,(__ubuf__ float*)expanded+256+(i+2)*8,30-i,1,1,0,columnPitch/8,0,1);PipeBarrier<PIPE_V>();
            }
        }
        ResetMask();Fence<HardEvent::V_MTE3>();
        const uint32_t alignedRows=(rows+7)/8*8;
        DataCopyExtParams write{32,rows*4,(pitch-alignedRows)/8,(t.paddedN-rows)*4,0};
        DataCopyPad(shared[begin],scratch,write);
    }
    __aicore__ inline void Update(uint32_t start){
        const uint32_t off=q==start/64?start%64+32:0;
        if(off>=64||start+32>=t.n)return;
        const uint32_t rows=t.n-start-32;
        DataCopy(raw,prodGm,rows*64);Fence<HardEvent::MTE2_V>();
        Add(panel[(start+32)*104+off],panel[(start+32)*104+off],raw[off],64-off,rows,{1,1,1,13,13,8});Fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline bool ProcessOne(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        const uint32_t vectorPitch=(t.paddedN+63)/64*64;
        CreateVecIndex(idx.ReinterpretCast<int32_t>(),0,vectorPitch);PipeBarrier<PIPE_V>();
        Muls(idx.ReinterpretCast<int32_t>(),idx.ReinterpretCast<int32_t>(),int32_t(416),vectorPitch);PipeBarrier<PIPE_V>();
        CreateVecIndex(temp.ReinterpretCast<int32_t>(),0,64);PipeBarrier<PIPE_V>();
        Muls(temp.ReinterpretCast<int32_t>(),temp.ReinterpretCast<int32_t>(),int32_t(128),64);PipeBarrier<PIPE_V>();
        Duplicate(panel[t.paddedN*104],0.0f,32*104);
        if(batch>=t.batch||first+64>t.n)Duplicate(panel,0.0f,t.paddedN*104);
        else if(t.paddedN>t.n)Duplicate(panel[t.n*104],0.0f,(t.paddedN-t.n)*104);
        Fence<HardEvent::V_S>();
        for(uint32_t r=batch<t.batch?t.n:0;r<t.paddedN;++r)
            if(r>=first&&r<first+64)panel.SetValue(r*104+r-first,1.0f);
        Fence<HardEvent::S_MTE2>();
        if(first<t.n&&batch<t.batch){
            const uint32_t width=Min(64,t.n-first),aligned=(width+7)/8*8;
            DataCopyExtParams cp{uint16_t(t.n),width*4,(t.n-width)*4,(104-aligned)/8,0};
            DataCopyPadExtParams<float> pad{true,0,uint8_t(aligned-width),0};
            DataCopyPad(panel,input[uint64_t(batch)*t.n*t.n+first],cp,pad);
            Fence<HardEvent::MTE2_V>();
        }
        magnitude=1.0f;
        float m=1.0f;int32_t e=0;ownRisk=false;
        // Finish the final 64 columns locally when the matrix ends on a 64-column boundary.
        for(uint32_t start=0;start<t.n;start+=(t.n==256&&start==192)?64:32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(t.paddedN*64+128)];comm=shared[t.paddedN*64];
            const bool cooperative=start+32<t.n&&t.n-start-32>=t.coresPerMatrix*32;
            if(q==owner){
                if(cooperative)Factor<true>(start);else Factor<false>(start);
            }
            {CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
            if(q==0){
                if(q!=owner){DataCopy(meta,comm,8);Fence<HardEvent::MTE2_S>();}
                e+=meta.ReinterpretCast<int32_t>().GetValue(1);m*=meta.GetValue(0);
            }
            if(start+32<t.n&&!(t.n==256&&start==192)){
                if(cooperative){
                    FactorTail(start);
                    CrossCoreSetFlag<0,PIPE_MTE3>(12);
                }
                if(q>owner){
                    if(cooperative)Fence<HardEvent::V_MTE2>();
                    Solve(start);
                }
                if(cooperative)CrossCoreWaitFlag(12);
                const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
                if(cubeNeeded){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Update(start);}
            }
        }
        DataCopyExtParams flags{uint16_t(t.groups),t.coresPerMatrix*32,t.robustOffset*4-t.coresPerMatrix*32,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(meta,allShared[t.paddedN*64+64],flags,pad);Fence<HardEvent::MTE2_S>();
        bool safe=true;for(uint32_t c=0;c<t.groups*t.coresPerMatrix;++c)safe=safe&&meta.GetValue(c*8+3)==0.0f;
        if(!safe)return false;
        if(q==0&&batch<t.batch){
            while(e>120){m*=Pow2(120);e-=120;} while(e< -120){m*=Pow2(-120);e+=120;}
            Fence<HardEvent::V_S>(); meta.SetValue(0,m*Pow2(e)); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams cp{1,4,0,0,0}; DataCopyPad(output[batch],meta,cp);
            Fence<HardEvent::MTE3_S>();
        }
        return true;
    }
private:
    TPipe &pipe;
    TBuf<TPosition::VECCALC> panelBuf,rawBuf,vecBuf,topBuf,idxBuf;
    LocalTensor<float> panel,raw,col,row,meta,temp,top;
    LocalTensor<uint32_t> idx;
    GlobalTensor<float> input,output,shared,comm,own,prodGm,allShared;
    LinalgDetConfig t; uint32_t q,first,group; float magnitude;bool ownRisk;
};

class CubeLUVectorHelpers {
public:
    __aicore__ inline CubeLUVectorHelpers(TPipe &external):pipe(external){}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR det, GM_ADDR ws,
                                const LinalgDetConfig &cfg) {
        t=cfg;
        const uint32_t logical=(GetBlockIdx()/2)%t.coresPerMatrix;
        q=logical;
        group=GetBlockIdx()/(2*t.coresPerMatrix); first=q*64;
        if(GetBlockIdx()&1){
            // Helpers only transpose/factor the lower rows and inspect risk flags.
            allShared.SetGlobalBuffer((__gm__ float*)ws);
            shared=allShared[group*t.robustOffset];comm=shared[t.paddedN*64];
            pipe.InitBuffer(rawBuf,6912*4);raw=rawBuf.Get<float>();
            pipe.InitBuffer(vecBuf,(256+64)*4);col=vecBuf.Get<float>();meta=col;temp=col[256];
            pipe.InitBuffer(topBuf,32*32*4);top=topBuf.Get<float>();
            return;
        }

        input.SetGlobalBuffer((__gm__ float*)A);
        output.SetGlobalBuffer((__gm__ float*)det);
        allShared.SetGlobalBuffer((__gm__ float*)ws);
        shared=allShared[group*t.robustOffset];
        comm=shared[t.paddedN*64];
        own=allShared[group*t.robustOffset+2*(t.paddedN*64+128)+q*(t.paddedN*64+8192)];
        prodGm=own[4096];
        pipe.InitBuffer(panelBuf,(t.paddedN+32)*104*4); panel=panelBuf.Get<float>();
        pipe.InitBuffer(rawBuf,t.paddedN*64*4); raw=rawBuf.Get<float>();
        const uint32_t vectorPitch=(t.paddedN+63)/64*64;
        pipe.InitBuffer(vecBuf,5*vectorPitch*4); col=vecBuf.Get<float>();
        row=col[vectorPitch]; meta=row[vectorPitch]; temp=meta[vectorPitch];
        pipe.InitBuffer(topBuf,32*32*4); top=topBuf.Get<float>();
        pipe.InitBuffer(idxBuf,vectorPitch*4); idx=idxBuf.Get<uint32_t>();
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
    template<bool COOPERATIVE> __aicore__ inline void Factor(uint32_t start){
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
        SetMaskNorm();ResetMask();Fence<HardEvent::V_S>();meta.SetValue(0,m);meta.ReinterpretCast<int32_t>().SetValue(1,e);meta.SetValue(2,risky?1.0f:0.0f);ownRisk=ownRisk||risky;meta.SetValue(3,ownRisk?1.0f:0.0f);
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
    __aicore__ inline void Solve(uint32_t start){
        DataCopyExtParams cp{32,128,(t.paddedN-32)*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
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
    __aicore__ inline void FactorTail(uint32_t start){
        const uint32_t remaining=t.n-start-32;
        const uint32_t pitch=((remaining+t.coresPerMatrix-1)/t.coresPerMatrix+31)/32*32;
        const uint32_t begin=start+32+q*pitch;
        if(begin>=t.n)return;
        const uint32_t rows=Min(pitch,t.n-begin),columnPitch=pitch + 8;
        auto scratch=raw;
        auto lower=(__ubuf__ float*)scratch.GetPhyAddr();
        auto matrix=lower+32*pitch;
        const uint32_t packedOffset=32*pitch+32*columnPitch;
        auto packed=lower+packedOffset;
        // The input packed tile is dead after the transpose barrier below.
        // Reuse it for up to8 immutable U-coefficient planes,2048 floats.
        auto expanded=(__ubuf__ uint32_t*)packed;
        auto u=(__ubuf__ float*)top.GetPhyAddr();
        auto offsets=(__ubuf__ uint32_t*)temp.GetPhyAddr();
        DataCopy(scratch[packedOffset],shared[32*t.paddedN+begin*32],rows*32);
        DataCopy(top,shared[32*t.paddedN+start*32],32*32);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::V_S>();SetMaskNorm();NativeMask(rows);
        if((t.n==256||t.n==200)&&rows%16==0){
            // Repeat over four8-column groups; each call transposes16 rows.
            TransDataTo5HDParams cv{};cv.repeatTimes=4;
            cv.srcRepStride=1;cv.dstRepStride=columnPitch;
            for(uint32_t rb=0;rb<rows;rb+=16){
                uint64_t srcList[16],dstList[16];
                #pragma unroll
                for(uint32_t rr=0;rr<16;++rr)srcList[rr]=uint64_t(packed+(rb+rr)*32);
                #pragma unroll
                for(uint32_t j=0;j<8;++j){dstList[2*j]=uint64_t(matrix+j*columnPitch+rb);dstList[2*j+1]=uint64_t(matrix+j*columnPitch+rb+8);}
                TransDataTo5HD<float>(dstList,srcList,cv);
            }
        }else{
            for(uint32_t c=0;c<32;++c)
                vgather((__ubuf__ uint32_t*)(matrix+c*columnPitch),offsets,uint32_t(uint64_t(packed)+c*4),8,1);
        }

        PipeBarrier<PIPE_V>();
        NativeMask(rows);
        for(uint32_t i=0;i<32;i+=2){
            auto c0=matrix+i*columnPitch;auto c1=c0+columnPitch;
            const float a=u[i*32+i],b=u[i*32+i+1],c=u[(i+1)*32+i],d=u[(i+1)*32+i+1];
            const float inv=1.0f/(a*d-b*c);
            auto v0=lower+i*pitch;auto v1=v0+pitch;
            vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(v1,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(v1,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            if(i+2<32){
                if((i&7)==0){
                    set_vector_mask(~uint64_t(0),~uint64_t(0));
                    // Eight rows for0/8/16; only six still-used rows for24.
                    vbrcb(expanded,(__ubuf__ uint32_t*)(u+i*32),1,8,i==24?24:32);
                    PipeBarrier<PIPE_V>();
                    NativeMask(rows);
                }
                vmla(matrix+(i+2)*columnPitch,v0,(__ubuf__ float*)expanded+((i&7)*32+i+2)*8,30-i,1,1,0,columnPitch/8,0,1);PipeBarrier<PIPE_V>();
                vmla(matrix+(i+2)*columnPitch,v1,(__ubuf__ float*)expanded+(((i&7)+1)*32+i+2)*8,30-i,1,1,0,columnPitch/8,0,1);PipeBarrier<PIPE_V>();
            }
        }
        ResetMask();Fence<HardEvent::V_MTE3>();
        const uint32_t alignedRows=(rows+7)/8*8;
        DataCopyExtParams write{32,rows*4,(pitch-alignedRows)/8,(t.paddedN-rows)*4,0};
        DataCopyPad(shared[begin],scratch,write);
    }
    __aicore__ inline void Update(uint32_t start){
        const uint32_t off=q==start/64?start%64+32:0;
        if(off>=64||start+32>=t.n)return;
        const uint32_t rows=t.n-start-32;
        DataCopy(raw,prodGm,rows*64);Fence<HardEvent::MTE2_V>();
        Add(panel[(start+32)*104+off],panel[(start+32)*104+off],raw[off],64-off,rows,{1,1,1,13,13,8});Fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline LocalTensor<int32_t> SignalBuffer(){
        // Helper meta reduction uses at most160 floats;192..223 are disjoint.
        if(GetBlockIdx()&1)return col[192].ReinterpretCast<int32_t>();
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
    __aicore__ inline bool ProcessOne(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        const uint32_t vectorPitch=(t.paddedN+63)/64*64;
        CreateVecIndex(idx.ReinterpretCast<int32_t>(),0,vectorPitch);PipeBarrier<PIPE_V>();
        Muls(idx.ReinterpretCast<int32_t>(),idx.ReinterpretCast<int32_t>(),int32_t(416),vectorPitch);PipeBarrier<PIPE_V>();
        CreateVecIndex(temp.ReinterpretCast<int32_t>(),0,64);PipeBarrier<PIPE_V>();
        Muls(temp.ReinterpretCast<int32_t>(),temp.ReinterpretCast<int32_t>(),int32_t(128),64);PipeBarrier<PIPE_V>();
        Duplicate(panel[t.paddedN*104],0.0f,32*104);
        if(batch>=t.batch||first+64>t.n)Duplicate(panel,0.0f,t.paddedN*104);
        else if(t.paddedN>t.n)Duplicate(panel[t.n*104],0.0f,(t.paddedN-t.n)*104);
        Fence<HardEvent::V_S>();
        for(uint32_t r=batch<t.batch?t.n:0;r<t.paddedN;++r)
            if(r>=first&&r<first+64)panel.SetValue(r*104+r-first,1.0f);
        Fence<HardEvent::S_MTE2>();
        if(first<t.n&&batch<t.batch){
            const uint32_t width=Min(64,t.n-first),aligned=(width+7)/8*8;
            DataCopyExtParams cp{uint16_t(t.n),width*4,(t.n-width)*4,(104-aligned)/8,0};
            DataCopyPadExtParams<float> pad{true,0,uint8_t(aligned-width),0};
            DataCopyPad(panel,input[uint64_t(batch)*t.n*t.n+first],cp,pad);
            Fence<HardEvent::MTE2_V>();
        }
        magnitude=1.0f;
        float m=1.0f;int32_t e=0;ownRisk=false;
        // Finish the final 64 columns locally when the matrix ends on a 64-column boundary.
        for(uint32_t start=0;start<t.n;start+=(t.n==256&&start==192)?64:32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(t.paddedN*64+128)];comm=shared[t.paddedN*64];
            const bool cooperative=start+32<t.n&&t.n-start-32>=t.coresPerMatrix*32;
            if(q==owner){
                if(cooperative)Factor<true>(start);else Factor<false>(start);
            }
            {CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
            if(q==0){
                if(q!=owner){DataCopy(meta,comm,8);Fence<HardEvent::MTE2_S>();}
                e+=meta.ReinterpretCast<int32_t>().GetValue(1);m*=meta.GetValue(0);
            }
            if(start+32<t.n&&!(t.n==256&&start==192)){
                // Only noncooperative n256 stages have complete lower A at global10.
                const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
                if(t.n==256&&!cooperative&&cubeNeeded)CrossCoreSetFlag<2,PIPE_MTE3>(11);
                // Keep the original helper/Solve overlap in cooperative stages.
                if(q>owner){
                    if(cooperative)Fence<HardEvent::V_MTE2>();
                    Solve(start);
                }

                if(cubeNeeded){if(cooperative&&!(t.n==256&&t.coresPerMatrix==4))WaitLowerReady();CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Update(start);}
            }
        }
        DataCopyExtParams flags{uint16_t(t.groups),t.coresPerMatrix*32,t.robustOffset*4-t.coresPerMatrix*32,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(meta,allShared[t.paddedN*64+64],flags,pad);Fence<HardEvent::MTE2_S>();
        bool safe=true;for(uint32_t c=0;c<t.groups*t.coresPerMatrix;++c)safe=safe&&meta.GetValue(c*8+3)==0.0f;
        if(!safe)return false;
        if(q==0&&batch<t.batch){
            while(e>120){m*=Pow2(120);e-=120;} while(e< -120){m*=Pow2(-120);e+=120;}
            Fence<HardEvent::V_S>(); meta.SetValue(0,m*Pow2(e)); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams cp{1,4,0,0,0}; DataCopyPad(output[batch],meta,cp);
            Fence<HardEvent::MTE3_S>();
        }
        return true;
    }
    // Paired helper for n200: all four K32 prefix stages cooperate.
    __aicore__ inline bool ProcessHelper200(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        CreateVecIndex(temp.ReinterpretCast<int32_t>(),0,64);PipeBarrier<PIPE_V>();
        Muls(temp.ReinterpretCast<int32_t>(),temp.ReinterpretCast<int32_t>(),int32_t(128),64);PipeBarrier<PIPE_V>();
        for(uint32_t start=0;start<=128;start+=32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(t.paddedN*64+128)];comm=shared[t.paddedN*64];
            if(start==128){CrossCoreSetFlag<0,PIPE_MTE3>(12);CrossCoreWaitFlag(12);}
            CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);
            if(start==128)break;
            if(start<128){FactorTail(start);PublishLowerReady();}
            const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
            if(cubeNeeded){
                CrossCoreSetFlag<2,PIPE_MTE3>(11);
                CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);
            }
        }
        DataCopyExtParams flags{uint16_t(t.groups),t.coresPerMatrix*32,t.robustOffset*4-t.coresPerMatrix*32,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(meta,allShared[t.paddedN*64+64],flags,pad);Fence<HardEvent::MTE2_S>();
        bool safe=true;for(uint32_t c=0;c<t.groups*t.coresPerMatrix;++c)safe=safe&&meta.GetValue(c*8+3)==0.0f;
        return safe;
    }
    __aicore__ inline bool ProcessHelper(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        CreateVecIndex(temp.ReinterpretCast<int32_t>(),0,64);PipeBarrier<PIPE_V>();
        Muls(temp.ReinterpretCast<int32_t>(),temp.ReinterpretCast<int32_t>(),int32_t(128),64);PipeBarrier<PIPE_V>();
        for(uint32_t start=0;start<t.n;start+=(t.n==256&&start==192)?64:32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(t.paddedN*64+128)];comm=shared[t.paddedN*64];
            const bool cooperative=start+32<t.n&&t.n-start-32>=t.coresPerMatrix*32;
            CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);
            if(start+32<t.n&&!(t.n==256&&start==192)){
                if(cooperative){
                    FactorTail(start);
                    PublishLowerReady();
                }
                const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
                // Both AIV subblocks must publish11 before the paired Cube proceeds.
                if(t.n==256&&!cooperative&&cubeNeeded)CrossCoreSetFlag<2,PIPE_MTE3>(11);
                if(cubeNeeded){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);}
            }
        }
        DataCopyExtParams flags{uint16_t(t.groups),t.coresPerMatrix*32,t.robustOffset*4-t.coresPerMatrix*32,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(meta,allShared[t.paddedN*64+64],flags,pad);Fence<HardEvent::MTE2_S>();
        bool safe=true;for(uint32_t c=0;c<t.groups*t.coresPerMatrix;++c)safe=safe&&meta.GetValue(c*8+3)==0.0f;
        return safe;
    }
private:
    TPipe &pipe;
    TBuf<TPosition::VECCALC> panelBuf,rawBuf,vecBuf,topBuf,idxBuf;
    LocalTensor<float> panel,raw,col,row,meta,temp,top;
    LocalTensor<uint32_t> idx;
    GlobalTensor<float> input,output,shared,comm,own,prodGm,allShared;
    LinalgDetConfig t; uint32_t q,first,group; float magnitude;bool ownRisk;
};

class CubeLUSequentialVector {
public:
    __aicore__ inline CubeLUSequentialVector(TPipe &external):pipe(external){}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR det, GM_ADDR ws,
                                const LinalgDetConfig &cfg) {
        t=cfg;
        const uint32_t logical=GetBlockIdx()%t.coresPerMatrix;
        q=logical;
        group=GetBlockIdx()/t.coresPerMatrix; first=q*64;
        input.SetGlobalBuffer((__gm__ float*)A);
        output.SetGlobalBuffer((__gm__ float*)det);
        allShared.SetGlobalBuffer((__gm__ float*)ws);
        shared=allShared[group*t.robustOffset];
        comm=shared[t.paddedN*64];
        own=allShared[group*t.robustOffset+2*(t.paddedN*64+128)+q*(t.paddedN*64+8192)];
        prodGm=own[4096];
        pipe.InitBuffer(panelBuf,(t.paddedN+32)*104*4); panel=panelBuf.Get<float>();
        pipe.InitBuffer(rawBuf,t.paddedN*64*4); raw=rawBuf.Get<float>();
        const uint32_t vectorPitch=(t.paddedN+63)/64*64;
        pipe.InitBuffer(vecBuf,5*vectorPitch*4); col=vecBuf.Get<float>();
        row=col[vectorPitch]; meta=row[vectorPitch]; temp=meta[vectorPitch];
        pipe.InitBuffer(topBuf,32*32*4); top=topBuf.Get<float>();
        pipe.InitBuffer(idxBuf,vectorPitch*4); idx=idxBuf.Get<uint32_t>();
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
    __aicore__ inline void Factor(uint32_t start){
        SetMaskNorm();
        float m=1.0f;int32_t e=0;uint32_t riskMask=0;
        const uint32_t rows=t.n-start,steps=Min(32U,t.n-start),local=start%64;
        const bool saveLower=start+32<t.n;
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
                    const uint32_t off=(local+i+2)/8*8,boundary=Min(start+32,t.n),upper=boundary-k-2;
                    if(splitColumns){
                        NativeMask(32-off);
                        vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(i+2)*104,t.n-k-2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(i+2)*104,t.n-k-2,1,1,0,13,0,13);
                        SetFlag<HardEvent::V_S>(0);pivotReady=true;
                        NativeMask(32);
                        vmla(pp+(k+2)*104+32,pp+k*104+32,(__ubuf__ float*)expanded+(i+2)*104,upper,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+32,pp+(k+1)*104+32,(__ubuf__ float*)expanded+8+(i+2)*104,upper,1,1,0,13,0,13);
                    }else if(local==32){
                        NativeMask(64-off);
                        vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(i+2)*104,t.n-k-2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(i+2)*104,t.n-k-2,1,1,0,13,0,13);
                    }else{
                        if(upper){
                            NativeMask(64-off);
                            vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(i+2)*104,upper,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                            vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(i+2)*104,upper,1,1,0,13,0,13);
                            SetFlag<HardEvent::V_S>(0);pivotReady=true;
                        }
                        if(boundary<t.n&&off<local+32){
                            NativeMask(local+32-off);
                            vmla(pp+boundary*104+off,pp+k*104+off,(__ubuf__ float*)expanded+32*104,t.n-boundary,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                            vmla(pp+boundary*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+32*104,t.n-boundary,1,1,0,13,0,13);
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
        SetMaskNorm();ResetMask();Fence<HardEvent::V_S>();meta.SetValue(0,m);meta.ReinterpretCast<int32_t>().SetValue(1,e);meta.SetValue(2,risky?1.0f:0.0f);ownRisk=ownRisk||risky;meta.SetValue(3,ownRisk?1.0f:0.0f);
        if(saveLower){
            Fence<HardEvent::V_MTE3>();DataCopy(shared,raw,32*t.paddedN);
            DataCopyExtParams cp{32,256,5,0,0};DataCopyPad(own,panel[start*104],cp);
        }
        Fence<HardEvent::S_MTE3>();DataCopy(comm,meta,8);
        if(start+32>=Min(first+64,t.n))DataCopy(allShared[group*t.robustOffset+t.paddedN*64+64+q*8],meta,8);
    }
    __aicore__ inline void Solve(uint32_t start){
        DataCopyExtParams cp{32,128,(t.paddedN-32)*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
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
        if(off>=64||start+32>=t.n)return;
        const uint32_t rows=t.n-start-32;
        DataCopy(raw,prodGm,rows*64);Fence<HardEvent::MTE2_V>();
        Add(panel[(start+32)*104+off],panel[(start+32)*104+off],raw[off],64-off,rows,{1,1,1,13,13,8});Fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline bool ProcessOne(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        const uint32_t vectorPitch=(t.paddedN+63)/64*64;
        CreateVecIndex(idx.ReinterpretCast<int32_t>(),0,vectorPitch);PipeBarrier<PIPE_V>();
        Muls(idx.ReinterpretCast<int32_t>(),idx.ReinterpretCast<int32_t>(),int32_t(416),vectorPitch);PipeBarrier<PIPE_V>();
        Duplicate(panel[t.paddedN*104],0.0f,32*104);
        if(batch>=t.batch||first+64>t.n)Duplicate(panel,0.0f,t.paddedN*104);
        else if(t.paddedN>t.n)Duplicate(panel[t.n*104],0.0f,(t.paddedN-t.n)*104);
        Fence<HardEvent::V_S>();
        for(uint32_t r=batch<t.batch?t.n:0;r<t.paddedN;++r)
            if(r>=first&&r<first+64)panel.SetValue(r*104+r-first,1.0f);
        Fence<HardEvent::S_MTE2>();
        if(first<t.n&&batch<t.batch){
            const uint32_t width=Min(64,t.n-first),aligned=(width+7)/8*8;
            DataCopyExtParams cp{uint16_t(t.n),width*4,(t.n-width)*4,(104-aligned)/8,0};
            DataCopyPadExtParams<float> pad{true,0,uint8_t(aligned-width),0};
            DataCopyPad(panel,input[uint64_t(batch)*t.n*t.n+first],cp,pad);
            Fence<HardEvent::MTE2_V>();
        }
        magnitude=1.0f;
        float m=1.0f;int32_t e=0;ownRisk=false;
        for(uint32_t start=0;start<t.n;start+=32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(t.paddedN*64+128)];comm=shared[t.paddedN*64];
            if(q==owner)Factor(start);
            {CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
            if(q==0){
                if(q!=owner){DataCopy(meta,comm,8);Fence<HardEvent::MTE2_S>();}
                e+=meta.ReinterpretCast<int32_t>().GetValue(1);m*=meta.GetValue(0);
            }
            if(start+32<t.n){
                if(q>owner)Solve(start);
                const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
                if(cubeNeeded){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Update(start);}
            }
        }
        DataCopyExtParams flags{uint16_t(t.groups),t.coresPerMatrix*32,t.robustOffset*4-t.coresPerMatrix*32,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(meta,allShared[t.paddedN*64+64],flags,pad);Fence<HardEvent::MTE2_S>();
        bool safe=true;for(uint32_t c=0;c<t.groups*t.coresPerMatrix;++c)safe=safe&&meta.GetValue(c*8+3)==0.0f;
        if(!safe)return false;
        if(q==0&&batch<t.batch){
            while(e>120){m*=Pow2(120);e-=120;} while(e< -120){m*=Pow2(-120);e+=120;}
            Fence<HardEvent::V_S>(); meta.SetValue(0,m*Pow2(e)); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams cp{1,4,0,0,0}; DataCopyPad(output[batch],meta,cp);
            Fence<HardEvent::MTE3_S>();
        }
        return true;
    }
private:
    TPipe &pipe;
    TBuf<TPosition::VECCALC> panelBuf,rawBuf,vecBuf,topBuf,idxBuf;
    LocalTensor<float> panel,raw,col,row,meta,temp,top;
    LocalTensor<uint32_t> idx;
    GlobalTensor<float> input,output,shared,comm,own,prodGm,allShared;
    LinalgDetConfig t; uint32_t q,first,group; float magnitude;bool ownRisk;
};


template<const MatmulConfig &CFG> __aicore__ inline void RunLUCubeImpl(const LinalgDetTilingData &t, GM_ADDR ws){
    if ASCEND_IS_AIC {
        TPipe pipe;MatmulImpl<AB,AB,AB,AB,CFG> mm;mm.SetSubBlockIdx(0);mm.Init(&t.cube,&pipe);mm.SetHF32(false);mm.SetOrgShape(t.paddedN,64,32);
        GlobalTensor<float> shared,own,prod;
        const uint32_t q=GetBlockIdx()%t.coresPerMatrix,group=GetBlockIdx()/t.coresPerMatrix;
        own.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+2*(t.paddedN*64+128)+q*(t.paddedN*64+8192));prod=own[4096];
        for(uint32_t batch=0;batch<(t.batch+t.groups-1)/t.groups;++batch)
          for(uint32_t start=0;start+32<t.n&&!(t.n==256&&start==192)&&!(t.n==200&&start>=128)&&start<=q*64;start+=32){
            shared.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+(start/32%2)*(t.paddedN*64+128));
            CrossCoreWaitFlag(8);
            mm.SetTail(t.n-start-32,64,32);mm.SetTensorA(shared[start+32],true);mm.SetTensorB(own);mm.IterateAll(prod);mm.End();
            CrossCoreSetFlag<2,PIPE_FIX>(9);
          }
    }
}

// CANN9/arch2201 implementation of early GM->L1 A loading.
// One buffer is reused only after the prior Fixpipe/Vector update/global barrier.
__aicore__ inline void RunLUCubePrefetch200(const LinalgDetTilingData &t,GM_ADDR ws){
    if ASCEND_IS_AIC {
        // Basic API matches the qualified SDK9 Matmul Mmad/Fixpipe control fields.
        TPipe pipe;
        LocalTensor<float> a1(TPosition::A1,0,256*32);
        LocalTensor<float> b1(TPosition::B1,256*32*4,64*32);
        LocalTensor<float> a2(TPosition::A2,0,256*32);
        LocalTensor<float> b2(TPosition::B2,0,64*32);
        LocalTensor<float> c(TPosition::CO1,0,256*64);
        SetHF32Mode(0);SetHF32TransMode(0);
        const uint8_t pad[4]={0,0,0,0};
        SetFmatrix(1,32,pad,FmatrixMode::FMATRIX_LEFT);
        SetFmatrix(1,32,pad,FmatrixMode::FMATRIX_RIGHT);
        GlobalTensor<float> shared,own,prod;
        const uint32_t q=GetBlockIdx()%4,group=GetBlockIdx()/4;
        own.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+2*(t.paddedN*64+128)+q*(t.paddedN*64+8192));prod=own[4096];
        for(uint32_t batch=0;batch<(t.batch+t.groups-1)/t.groups;++batch)
          for(uint32_t start=0;start<128&&start<=q*64;start+=32){
            const uint32_t rows=200-start-32,ma=(rows+15)/16*16;
            shared.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+(start/32%2)*(t.paddedN*64+128));
            CrossCoreWaitFlag(11);
            // Keep256's paired11 and allfour volatile helper-ready reads.
            volatile __gm__ int32_t *ready=(volatile __gm__ int32_t*)shared[t.paddedN*64+96].GetPhyAddr();
            do{
                dcci((__gm__ uint64_t*)ready,cache_line_t::SINGLE_CACHE_LINE,dcci_dst_t::CACHELINE_OUT);
                dcci((__gm__ uint64_t*)(ready+16),cache_line_t::SINGLE_CACHE_LINE,dcci_dst_t::CACHELINE_OUT);
            }while(ready[0]!=1||ready[8]!=1||ready[16]!=1||ready[24]!=1);
            Nd2NzParams ap{};ap.ndNum=1;ap.nValue=32;ap.dValue=rows;
            ap.srcDValue=t.paddedN;ap.dstNzC0Stride=32;ap.dstNzNStride=1;
            InitConstValue(a1[rows*32],{1,32,0,0U});
            DataCopy(a1,shared[start+32],ap);
            CrossCoreWaitFlag(8);
            Nd2NzParams bp{};bp.ndNum=1;bp.nValue=32;bp.dValue=64;
            bp.srcDValue=64;bp.dstNzC0Stride=32;bp.dstNzNStride=1;
            DataCopy(b1,own,bp);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData3DParamsV2Pro la{};la.channelSize=ma;
            la.extConfig=(uint64_t(32)<<16)|ma;la.enTranspose=true;
            LoadData(a2,a1,la);
            LoadData3DParamsV2Pro lb{};lb.channelSize=64;
            lb.extConfig=(uint64_t(32)<<16)|64;lb.fMatrixCtrl=true;
            LoadData(b2,b1,lb);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams mp{};mp.m=rows;mp.n=64;mp.k=32;
            mp.cmatrixInitVal=true;mp.cmatrixSource=false;mp.unitFlag=3;mp.kDirectionAlign=1;
            Mmad(c,a2,b2,mp);
            // Unit flag3 waits on Cube completion; original9 protects all later reuse.
            FixpipeParamsV220 fp{};fp.mSize=rows;fp.nSize=64;fp.srcStride=ma;fp.dstStride=64;
            fp.quantPre=QuantMode_t::NoQuant;fp.unitFlag=3;
            Fixpipe<float,float,CFG_ROW_MAJOR>(prod,c,fp);
            CrossCoreSetFlag<2,PIPE_FIX>(9);
          }
        PipeBarrier<PIPE_ALL>();
    }
}


// Early A is legal only at starts128/160; earlier A still depends on all helpers.
__aicore__ inline void RunLUCubePrefetch256(const LinalgDetTilingData &t,GM_ADDR ws){
    if ASCEND_IS_AIC {
        // Basic API matches the qualified SDK9 Matmul Mmad/Fixpipe control fields.
        TPipe pipe;
        LocalTensor<float> a1(TPosition::A1,0,256*32);
        LocalTensor<float> b1(TPosition::B1,256*32*4,64*32);
        LocalTensor<float> a2(TPosition::A2,0,256*32);
        LocalTensor<float> b2(TPosition::B2,0,64*32);
        LocalTensor<float> c(TPosition::CO1,0,256*64);
        SetHF32Mode(0);SetHF32TransMode(0);
        const uint8_t pad[4]={0,0,0,0};
        SetFmatrix(1,32,pad,FmatrixMode::FMATRIX_LEFT);
        SetFmatrix(1,32,pad,FmatrixMode::FMATRIX_RIGHT);
        GlobalTensor<float> shared,own,prod;
        const uint32_t q=GetBlockIdx()%4,group=GetBlockIdx()/4;
        own.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+2*(256*64+128)+q*(256*64+8192));prod=own[4096];
        for(uint32_t batch=0;batch<(t.batch+t.groups-1)/t.groups;++batch)
          for(uint32_t start=0;start<192&&start<=q*64;start+=32){
            const uint32_t rows=256-start-32;
            shared.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+(start/32%2)*(256*64+128));
            Nd2NzParams ap{};ap.ndNum=1;ap.nValue=32;ap.dValue=rows;
            ap.srcDValue=256;ap.dstNzC0Stride=32;ap.dstNzNStride=1;
            Nd2NzParams bp{};bp.ndNum=1;bp.nValue=32;bp.dValue=64;
            bp.srcDValue=64;bp.dstNzC0Stride=32;bp.dstNzNStride=1;
            if(start>=128){
                CrossCoreWaitFlag(11);DataCopy(a1,shared[start+32],ap);
                CrossCoreWaitFlag(8);DataCopy(b1,own,bp);
            }else{
                // Paired8 orders this core's complete upperB and local helper.
                CrossCoreWaitFlag(8);DataCopy(b1,own,bp);
                // Volatile reads must remain inside the poll: ordinary GetValue
                // loads were hoisted before the DCCI loop by the compiler.
                volatile __gm__ int32_t *ready=(volatile __gm__ int32_t*)shared[256*64+96].GetPhyAddr();
                // Same cache invalidation primitive as qualified ReadDecision.
                // Invalidate both SDK64B cache lines covering allfour32B flagrecords.
                do{
                    dcci((__gm__ uint64_t*)ready,cache_line_t::SINGLE_CACHE_LINE,dcci_dst_t::CACHELINE_OUT);
                    dcci((__gm__ uint64_t*)(ready+16),cache_line_t::SINGLE_CACHE_LINE,dcci_dst_t::CACHELINE_OUT);
                }while(ready[0]!=1||ready[8]!=1||ready[16]!=1||ready[24]!=1);
                DataCopy(a1,shared[start+32],ap);
            }
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData3DParamsV2Pro la{};la.channelSize=rows;
            la.extConfig=(uint64_t(32)<<16)|rows;la.enTranspose=true;
            LoadData(a2,a1,la);
            LoadData3DParamsV2Pro lb{};lb.channelSize=64;
            lb.extConfig=(uint64_t(32)<<16)|64;lb.fMatrixCtrl=true;
            LoadData(b2,b1,lb);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams mp{};mp.m=rows;mp.n=64;mp.k=32;
            mp.cmatrixInitVal=true;mp.cmatrixSource=false;mp.unitFlag=3;mp.kDirectionAlign=1;
            Mmad(c,a2,b2,mp);
            // Unit flag3 waits on Cube completion; original9 protects all later reuse.
            FixpipeParamsV220 fp{};fp.mSize=rows;fp.nSize=64;fp.srcStride=rows;fp.dstStride=64;
            fp.quantPre=QuantMode_t::NoQuant;fp.unitFlag=3;
            Fixpipe<float,float,CFG_ROW_MAJOR>(prod,c,fp);
            CrossCoreSetFlag<2,PIPE_FIX>(9);
          }
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void RunLUCube(const LinalgDetTilingData &t,GM_ADDR ws){
    if(t.n==256&&t.coresPerMatrix==4){RunLUCubePrefetch256(t,ws);return;}
    if(t.n==200&&t.coresPerMatrix==4){RunLUCubePrefetch200(t,ws);return;}
    if(t.cubeMode==2)RunLUCubeImpl<DET_FIXED_CUBE_CFG>(t,ws);
    else RunLUCubeImpl<CFG_NORM>(t,ws);
}
