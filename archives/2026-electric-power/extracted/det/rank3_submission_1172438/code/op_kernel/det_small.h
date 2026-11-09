#pragma once
#include "kernel_operator.h"
#include "linalg_det_tiling.h"
using namespace AscendC;
template<HardEvent E> __aicore__ inline void SmallFence(){SetFlag<E>(0);WaitFlag<E>(0);}
__aicore__ inline uint32_t SmallMin(uint32_t a,uint32_t b){return a<b?a:b;}
__aicore__ inline uint32_t SmallMagnitudeBits(float value){
    union{float f;uint32_t u;}bits;bits.f=value;return bits.u&0x7fffffffU;
}
__aicore__ inline void SmallNativeMask(uint32_t n){
    SetMaskNorm();set_vector_mask(0,n>=64?~uint64_t(0):(uint64_t(1)<<n)-1);
}
template<uint32_t ALIGNED_N,bool FIXED=false,uint32_t STATIC_N=0,bool FINAL_OUTPUT=false> __aicore__ inline bool SmallDet(GM_ADDR a,GM_ADDR det,uint32_t realN,uint32_t batch,GM_ADDR indexData) {
    GlobalTensor<float> input,output;input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)det);
    constexpr bool NORMALIZE=ALIGNED_N>32;
    constexpr uint32_t q=ALIGNED_N<=8?8:ALIGNED_N<=16?16:ALIGNED_N<=32?32:64,p=q*(2*q+8);
    constexpr uint64_t maskQ=~uint64_t(0)>>(64-q),maskN=~uint64_t(0)>>(64-ALIGNED_N);
    SetMaskNorm();
    __builtin_assume(realN>=3&&realN<=ALIGNED_N&&realN>ALIGNED_N-8);
    if constexpr(FIXED)__builtin_assume(realN==ALIGNED_N);
    LocalTensor<float> panel(TPosition::VECCALC,0,2*p+41*q+8+(ALIGNED_N<=16?ALIGNED_N*ALIGNED_N:0));auto panelLo=panel[p];
    auto v=panel[2*p];auto raw=panel[2*p+q];
    auto indices=panel[2*p+9*q].ReinterpretCast<uint32_t>();
    auto meta=panel[2*p+11*q];auto s0=meta[8];
    auto ph=s0[8*q];auto pl=ph[q];
        const uint32_t n = STATIC_N?STATIC_N:(FIXED?ALIGNED_N:realN);
        constexpr uint32_t pitch=ALIGNED_N<=16?ALIGNED_N:(ALIGNED_N==56?56:2*ALIGNED_N+8),period=ALIGNED_N==56?16:((ALIGNED_N/4)&(0U-ALIGNED_N/4)),bankMask=period-1;
        const uint32_t len = n * pitch;
        auto index = indices.template ReinterpretCast<int32_t>();
        auto indexPtr=(__ubuf__ uint32_t*)indices.GetPhyAddr();
        auto panelPtr=(__ubuf__ float*)panel.GetPhyAddr();
        auto colPtr=(__ubuf__ float*)s0.GetPhyAddr();
        auto vectorPtr=(__ubuf__ float*)v.GetPhyAddr();
        const uint64_t rawBase=uint64_t(panel[2*p+21*q+8].GetPhyAddr());
        const uint32_t bank=uint32_t(rawBase>>5)&bankMask;
        auto copyIndex=(__ubuf__ uint32_t*)(indices[q].GetPhyAddr());
        GlobalTensor<uint32_t> gmIndices;gmIndices.SetGlobalBuffer((__gm__ uint32_t*)indexData);
        if constexpr(ALIGNED_N<=16){
            #pragma unroll
            for(uint32_t j=0;j<q;++j)indexPtr[j]=j*pitch*4;
            SmallFence<HardEvent::S_V>();
        }else if constexpr(!(FIXED&&ALIGNED_N==64))DataCopy(indices,gmIndices,2*q);
        if(n%8){
            auto repack=panel[2*p+41*q+8].template ReinterpretCast<uint32_t>();
            if constexpr(ALIGNED_N<=16)DataCopy(repack,gmIndices[128],(n*pitch+63)/64*64);
            DataCopyExtParams cp{1,n*n*4,0,0,0};DataCopyPadExtParams<float> pad{true,0,uint8_t((8-n*n%8)%8),0};
            DataCopyPad(panelLo,input[uint64_t(batch)*n*n],cp,pad);
            if constexpr(FIXED&&ALIGNED_N==64){
                #pragma unroll
                for(uint32_t i=0;i<q;++i)indexPtr[i]=i*pitch*4;
                SmallFence<HardEvent::S_V>();
            }
            if constexpr(FIXED)SmallFence<HardEvent::MTE2_S>();else SmallFence<HardEvent::MTE2_V>();
            if constexpr(ALIGNED_N<=16){
                set_vector_mask(0,~uint64_t(0));
                vgather((__ubuf__ uint32_t*)panelPtr,(__ubuf__ uint32_t*)repack.GetPhyAddr(),uint32_t(panelLo.GetPhyAddr()),8,(n*pitch+63)/64);
            }else{
                set_vector_mask(0,maskQ>>(q-n));
                for(uint32_t r=0;r<n;++r)
                    vgather((__ubuf__ uint32_t*)(panelPtr+r*pitch),copyIndex,uint32_t(panelLo.GetPhyAddr()+r*n*4),8,1);
            }
            PipeBarrier<PIPE_V>();
        }else{
            DataCopyExtParams cp{uint16_t(n),n*4,0,(pitch-ALIGNED_N)/8,0};DataCopyPadExtParams<float> pad{true,0,uint8_t(ALIGNED_N-n),0};
            if constexpr(ALIGNED_N<=16)DataCopy(panel,input[uint64_t(batch)*n*n],n*n);
            else DataCopyPad(panel,input[uint64_t(batch)*n*n],cp,pad);
            
            if constexpr(FIXED&&ALIGNED_N==64){
                #pragma unroll
                for(uint32_t i=0;i<q;++i)indexPtr[i]=i*pitch*4;
                SmallFence<HardEvent::S_V>();
            }
            if constexpr(FIXED)SmallFence<HardEvent::MTE2_S>();else SmallFence<HardEvent::MTE2_V>();
        }
        // Input repacking has finished; panelLo is dead until fallback reloads input.
        auto pairIndex=(__ubuf__ int32_t*)panelLo.GetPhyAddr();
        if constexpr(ALIGNED_N>16){
            set_vector_mask(0,maskQ);
            vadds(pairIndex,(__ubuf__ int32_t*)indexPtr,int32_t(0),1,1,1,8,8);
            vadds(pairIndex+64,(__ubuf__ int32_t*)indexPtr,int32_t(4),1,1,1,8,8);
            PipeBarrier<PIPE_V>();
        }
        const float magnitude=1.0f;
        bool safe=true;uint32_t riskMask=0;
        float product = 1.0f;
        int32_t exponent = 0;
        auto col2Ptr=colPtr+q;
        
        auto secondPtr=(__ubuf__ float*)ph.GetPhyAddr();
        #pragma unroll (ALIGNED_N<=16?8:4)
        for(uint32_t k=0;k<n;k+=2){
            const uint32_t remaining=n-k;
            // The aligned input DMA has already completed before the first pivot.
            if constexpr(FIXED){if(k)SmallFence<HardEvent::V_S>();}
            else SmallFence<HardEvent::V_S>();
                if(remaining==4&&ALIGNED_N>16){
                    auto z=panelPtr+k*pitch+k;
                    const float a=z[0],b=z[1],c=z[2],d=z[3];
                    const float e=z[pitch],f=z[pitch+1],g=z[pitch+2],h=z[pitch+3];
                    const float i=z[2*pitch],j=z[2*pitch+1],l=z[2*pitch+2],m=z[2*pitch+3];
                    const float p=z[3*pitch],qv=z[3*pitch+1],r=z[3*pitch+2],s=z[3*pitch+3];
                    const float s0=a*f-e*b,s1=a*g-e*c,s2=a*h-e*d;
                    const float s3=b*g-f*c,s4=b*h-f*d,s5=c*h-g*d;
                    const float c0=i*qv-p*j,c1=i*r-p*l,c2=i*s-p*m;
                    const float c3=j*r-qv*l,c4=j*s-qv*m,c5=l*s-r*m;
                    const float tail=((s0*c5-s1*c4)+(s2*c3+s3*c2))+(-s4*c1+s5*c0);
                    const uint32_t at=SmallMagnitudeBits(tail);
                    riskMask|=uint32_t(at<=0x38d1b717U)|uint32_t(at>=0x7149f2caU);
                    product*=tail;
                    if constexpr(NORMALIZE){
                        const uint32_t ap=SmallMagnitudeBits(product);
                        riskMask|=uint32_t(ap<=0x0da24260U)|uint32_t(ap>=0x7149f2caU);
                        union{uint32_t u;float f;}value;value.f=product;
                        exponent+=int32_t((value.u>>23)&255)-127;
                        value.u=(value.u&0x807fffffU)|(127U<<23);product=value.f;
                    }
                    break;
                }
                if(remaining==4){
                    auto z=panelPtr+k*pitch+k;
                    const float a=z[0],b=z[1],c=z[pitch],d=z[pitch+1];
                    const float pivot=a*d-b*c,inv=1.0f/pivot;
                    const float u=-d*inv,v=c*inv,w=b*inv,x=-a*inv;
                    const float l20=z[2*pitch]*u+z[2*pitch+1]*v;
                    const float l21=z[2*pitch]*w+z[2*pitch+1]*x;
                    const float l30=z[3*pitch]*u+z[3*pitch+1]*v;
                    const float l31=z[3*pitch]*w+z[3*pitch+1]*x;
                    const float s22=(z[2*pitch+2]+z[2]*l20)+z[pitch+2]*l21;
                    const float s23=(z[2*pitch+3]+z[3]*l20)+z[pitch+3]*l21;
                    const float s32=(z[3*pitch+2]+z[2]*l30)+z[pitch+2]*l31;
                    const float s33=(z[3*pitch+3]+z[3]*l30)+z[pitch+3]*l31;
                    const float tail=s22*s33-s23*s32;
                    const uint32_t ap=SmallMagnitudeBits(pivot),at=SmallMagnitudeBits(tail);
                    riskMask|=uint32_t(ap<=(ALIGNED_N>16?0x3d800000U:0x38d1b717U))|uint32_t(ap>=0x7149f2caU);
                    riskMask|=uint32_t(at<=0x322bcc77U)|uint32_t(at>=0x7149f2caU);
                    product*=pivot;
                    if constexpr(NORMALIZE){
                        if((k&7)==6){
                            const uint32_t ap=SmallMagnitudeBits(product);
                            riskMask|=uint32_t(ap<=0x0da24260U)|uint32_t(ap>=0x7149f2caU);
                            union{uint32_t u;float f;}value;value.f=product;
                            exponent+=int32_t((value.u>>23)&255)-127;
                            value.u=(value.u&0x807fffffU)|(127U<<23);product=value.f;
                        }
                    }
                    product*=tail;
                    if constexpr(NORMALIZE){
                        const uint32_t ap=SmallMagnitudeBits(product);
                        riskMask|=uint32_t(ap<=0x0da24260U)|uint32_t(ap>=0x7149f2caU);
                        union{uint32_t u;float f;}value;value.f=product;
                        exponent+=int32_t((value.u>>23)&255)-127;
                        value.u=(value.u&0x807fffffU)|(127U<<23);product=value.f;
                    }
                    break;
                }
            if constexpr(!NORMALIZE){
                if(remaining==3){
                    auto tail=panelPtr+k*pitch+k;
                    const float a0=tail[0],a1=tail[1],a2=tail[2];
                    const float b0=tail[pitch],b1=tail[pitch+1],b2=tail[pitch+2];
                    const float c0=tail[2*pitch],c1=tail[2*pitch+1],c2=tail[2*pitch+2];
                    const float d=a0*(b1*c2-b2*c1)-a1*(b0*c2-b2*c0)+a2*(b0*c1-b1*c0);
                    product*=d;break;
                }
            }
            if(remaining>2){
                set_vector_mask(0,maskQ>>(q-remaining));
                if constexpr(ALIGNED_N>16){
                    vgather((__ubuf__ uint32_t*)colPtr,(__ubuf__ uint32_t*)pairIndex,uint32_t(uint64_t(panelPtr)+(k*pitch+k)*4),q/8,2);
                }else{
                    vgather((__ubuf__ uint32_t*)colPtr,indexPtr,uint32_t(uint64_t(panelPtr)+(k*pitch+k)*4),8,1);
                    vgather((__ubuf__ uint32_t*)col2Ptr,indexPtr,uint32_t(uint64_t(panelPtr)+(k*pitch+k+1)*4),8,1);
                }
            }
            float pivot;
            if(remaining==1){
                pivot=panelPtr[k*pitch+k];
                const uint32_t ap=SmallMagnitudeBits(pivot);
                riskMask|=uint32_t(ap<=0x38d1b717U)|uint32_t(ap>=0x7f800000U);
            }else{
                const float aa=panelPtr[k*pitch+k],cc=panelPtr[(k+1)*pitch+k],bb=panelPtr[k*pitch+k+1],dd=panelPtr[(k+1)*pitch+k+1];
                pivot=aa*dd-bb*cc;
                const uint32_t ap=SmallMagnitudeBits(pivot);
                riskMask|=uint32_t(ap<=(ALIGNED_N>16?0x3d800000U:0x38d1b717U))|uint32_t(ap>=0x7149f2caU);
                if(remaining>2){
                    const float inv=1.0f/pivot;
                    PipeBarrier<PIPE_V>();
                    vmuls(vectorPtr,colPtr,-dd*inv,1,1,1,8,8);
                    vmuls(secondPtr,colPtr,bb*inv,1,1,1,8,8);
                    PipeBarrier<PIPE_V>();
                    vaxpy(vectorPtr,col2Ptr,cc*inv,1,1,1,8,8);
                    vaxpy(secondPtr,col2Ptr,-aa*inv,1,1,1,8,8);
                    PipeBarrier<PIPE_V>();
                    const uint32_t shift=(((ALIGNED_N==56?k*(pitch/8):k)+ALIGNED_N/8-bank)&bankMask)*32;
                    auto rawPtr=(__ubuf__ float*)(rawBase+shift);
                    auto raw2Ptr=(__ubuf__ float*)(rawBase+10*q*4+shift);
                    set_vector_mask(~uint64_t(0),~uint64_t(0));
                    vbrcb((__ubuf__ uint32_t*)rawPtr,(__ubuf__ uint32_t*)vectorPtr,1,8,(remaining+7)/8);
                    vbrcb((__ubuf__ uint32_t*)raw2Ptr,(__ubuf__ uint32_t*)secondPtr,1,8,(remaining+7)/8);
                    PipeBarrier<PIPE_V>();
                    const uint32_t offset=(k+2)/8*8;
                    set_vector_mask(0,maskN>>offset);
                    vmla(panelPtr+(k+2)*pitch+offset,panelPtr+k*pitch+offset,rawPtr+16,remaining-2,1,1,0,pitch/8,0,1);
                    PipeBarrier<PIPE_V>();
                    vmla(panelPtr+(k+2)*pitch+offset,panelPtr+(k+1)*pitch+offset,raw2Ptr+16,remaining-2,1,1,0,pitch/8,0,1);
                }
            }
            product*=pivot;
            if constexpr(NORMALIZE){
                if((k&7)==6||k+2>=n){
                    const uint32_t ap=SmallMagnitudeBits(product);
                    riskMask|=uint32_t(ap<=0x0da24260U)|uint32_t(ap>=0x7149f2caU);
                    union{uint32_t u;float f;}value;value.f=product;
                    exponent+=int32_t((value.u>>23)&255)-127;
                    value.u=(value.u&0x807fffffU)|(127U<<23);product=value.f;
                }
            }
            PipeBarrier<PIPE_V>();
        }
        const uint32_t productBits=SmallMagnitudeBits(product);
        safe=riskMask==0&&productBits>0&&productBits<0x7149f2caU;
        if constexpr(NORMALIZE){
            while(exponent>120){product*=0x1p120f;exponent-=120;}
            while(exponent< -120){product*=0x1p-120f;exponent+=120;}
            union{uint32_t u;float f;}scale;scale.u=uint32_t(exponent+127)<<23;product*=scale.f;
        }
        if (safe) {
            if constexpr(FINAL_OUTPUT&&ALIGNED_N<=16){
#if __NPU_ARCH__ == 2201
                auto address=(__gm__ uint32_t*)det+batch;
                if((reinterpret_cast<uint64_t>(address)&127U)==0){
                    union{float f;uint32_t u;}value;value.f=product;
                    WriteGmByPassDCache(address,value.u);
                    return safe;
                }
#endif
            }
            PipeBarrier<PIPE_V>();
            meta.SetValue(0, product);
            SmallFence<HardEvent::S_MTE3>();
            DataCopyExtParams cp{1, sizeof(float), 0, 0, 0};
            DataCopyPad(output[batch], meta, cp);
            if constexpr(!FINAL_OUTPUT)SmallFence<HardEvent::MTE3_S>();
        }
        return safe;
    }

