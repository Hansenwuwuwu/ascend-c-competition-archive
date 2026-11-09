#pragma once
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & PINV_KEY_CUBE)
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#endif
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & PINV_KEY_CUBE) && (TILING_KEY_VAR & PINV_KEY_WINDOW)
__aicore__ inline __gm__ float *PinvCubeArea(GM_ADDR ws,const LinalgPinvTilingData &t){
    uint64_t count=pinv::CubeOffsetFloats(t.batch,t.cols,t.stride);
    return reinterpret_cast<__gm__ float*>(ws)+count;
}
__aicore__ inline void PinvCubeService(GM_ADDR ws,GM_ADDR inverse,const LinalgPinvCubeTilingData &data){
    auto base=PinvCubeArea(ws,data.base);uint32_t n=data.base.cols;
    GlobalTensor<int32_t> control;control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base));
    for(;;){
        CrossCoreWaitFlag<2>(8);
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
        int32_t command=control.GetValue(0);
        if(command==0){PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(10);return;}
        if(command==1){
            GlobalTensor<int8_t> a,b;GlobalTensor<int32_t> c;
            a.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+pinv::kCubeControlFloats));
            b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+pinv::kCubeControlFloats)+pinv::CubePlaneElements(n));
            c.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base+pinv::kCubeControlFloats)+pinv::CubeProductOffset(n));
            TPipe pipe;
            using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
            using B=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
            using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
            matmul::Matmul<A,B,C,C,CFG_MDL> mm;mm.Init(&data.cube,&pipe);
            uint32_t extent=pinv::kCubePlanes*n,tile=((extent+GetBlockNum()-1)/GetBlockNum()+15)/16*16;
            uint32_t row=GetBlockIdx()*tile;
            if(row<extent){
                mm.SetOrgShape(extent,extent,n);mm.SetSingleShape(extent-row<tile?extent-row:tile,extent,n);
                mm.SetTensorA(a[row*n]);mm.SetTensorB(b);
                mm.IterateAll(c[row*extent]);mm.End();
            }
            PipeBarrier<PIPE_ALL>();pipe.Destroy();
        }else{
            uint32_t batch=static_cast<uint32_t>(control.GetValue(1));
            GlobalTensor<float> a,b,c;
            a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inverse)+uint64_t(batch)*n*n);
            b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws)+uint64_t(data.base.batch+batch)*n*data.base.stride);
            c.SetGlobalBuffer(base+pinv::kCubeControlFloats+pinv::CubeProductOffset(n));
            TPipe pipe;
            using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,float>;
            using B=matmul::MatmulType<TPosition::GM,CubeFormat::ND,float>;
            using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,float>;
            matmul::Matmul<A,B,C,C,CFG_MDL> mm;mm.Init(&data.correction,&pipe);mm.SetHF32(false);
            uint32_t tile=((n+GetBlockNum()-1)/GetBlockNum()+15)/16*16,row=GetBlockIdx()*tile;
            if(row<n){
                mm.SetOrgShape(n,n,n);mm.SetSingleShape(n-row<tile?n-row:tile,n,n);
                mm.SetTensorA(a[row*n]);mm.SetTensorB(b);mm.IterateAll(c[row*n]);mm.End();
            }
            PipeBarrier<PIPE_ALL>();pipe.Destroy();
        }
        PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(10);
    }
}
__aicore__ inline void PinvCubeCorrect(GM_ADDR ws,const LinalgPinvTilingData &t,uint32_t batch,
    LocalTensor<float> out,uint32_t owned,uint32_t core,uint32_t cores){
    auto base=PinvCubeArea(ws,t);uint32_t n=t.cols;GlobalTensor<int32_t> control;GlobalTensor<float> product;
    control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base));product.SetGlobalBuffer(base+pinv::kCubeControlFloats+pinv::CubeProductOffset(n));
    if(core==0){
        control.SetValue(0,2);control.SetValue(1,static_cast<int32_t>(batch));
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
    }
    SyncAll();PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag<2>(10);SyncAll();
    SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
    DataCopyPad(out,product[core*n],DataCopyExtParams{static_cast<uint16_t>(owned),n*4,(cores-1)*n*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
}
__aicore__ inline void PinvCubeStop(GM_ADDR ws,const LinalgPinvTilingData &data){
    GlobalTensor<int32_t> control;control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(PinvCubeArea(ws,data)));
    if(GetBlockIdx()==0){
        control.SetValue(0,0);
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
    }
    SyncAll();PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag<2>(10);
}
__aicore__ inline float PinvCubePeakScale(__gm__ float *base,LocalTensor<float> cache,
    LocalTensor<float> scratch[],uint32_t count,uint32_t core,uint32_t cores,float fallback){
    Abs(scratch[0],cache,count);PipeBarrier<PIPE_V>();ReduceMax(scratch[2],scratch[0],scratch[1],count,false);
    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    auto statistic=scratch[7];statistic.SetValue(0,scratch[2].GetValue(0));
    GlobalTensor<float> statistics;statistics.SetGlobalBuffer(base);
    SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyPad(statistics[36+core*8],statistic,DataCopyExtParams{1,4,0,0,0});SyncAll();
    uint32_t chunks=(cores+7)/8;
    Duplicate(statistic,0.0f,chunks*64);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);DataCopy(statistic,statistics[32],cores*8);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    uint64_t selected[2]={uint64_t(0x1010101010101010),0};
    WholeReduceMax(scratch[0],statistic,selected,static_cast<uint8_t>(chunks),1,1,8);PipeBarrier<PIPE_V>();
    uint64_t summary[2]={uint64_t(0x5555)&((uint64_t(1)<<(chunks*2))-1),0};
    WholeReduceMax(scratch[2],scratch[0],summary,1,1,1,8);
    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    float peak,bits;GetReduceRepeatMaxMinSpr<float>(peak,bits);
    if(!(peak>0 && peak<1.0e20f))return fallback;
    float scale=1.0f,target=4.0f*peak;
    while(scale<target)scale*=2;while(scale*.5f>=target)scale*=.5f;
    return scale;
}
template<bool RIGHT>
__aicore__ inline float PinvCubeQuantize(LocalTensor<float> values,LocalTensor<float> scratch[],
    GlobalTensor<int8_t> digits,uint32_t n,uint32_t owned,uint32_t core,uint32_t cores,float scale){
    uint32_t count=owned*n;
    auto integers=scratch[0].ReinterpretCast<int32_t>();auto temp=scratch[0],rounded=scratch[1],carry=scratch[2];
    LocalTensor<half> planes[pinv::kCubePlanes];
    for(uint32_t limb=0;limb<pinv::kCubePlanes;++limb)planes[limb]=scratch[4+limb/2].ReinterpretCast<half>()[(limb%2)*count];
    Muls(values,values,pinv::kCubeRadix/scale,count);PipeBarrier<PIPE_V>();
    for(uint32_t limb=0;limb<pinv::kCubePlanes;++limb){
        Cast(integers,values,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
        Cast(rounded,integers,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
        Sub(values,values,rounded,count);PipeBarrier<PIPE_V>();Muls(values,values,pinv::kCubeRadix,count);PipeBarrier<PIPE_V>();
        Cast(planes[limb],rounded,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
    }
    for(uint32_t limb=pinv::kCubePlanes-1;limb>0;--limb){
        Cast(rounded,planes[limb],RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        Adds(carry,rounded,-127.0f,count);PipeBarrier<PIPE_V>();Maxs(carry,carry,0.0f,count);PipeBarrier<PIPE_V>();
        Mins(carry,carry,1.0f,count);PipeBarrier<PIPE_V>();
        Muls(temp,carry,pinv::kCubeRadix,count);PipeBarrier<PIPE_V>();Sub(rounded,rounded,temp,count);PipeBarrier<PIPE_V>();
        Cast(planes[limb],rounded,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        Cast(rounded,planes[limb-1],RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        Add(rounded,rounded,carry,count);PipeBarrier<PIPE_V>();
        Cast(planes[limb-1],rounded,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
    }
    auto small=scratch[3].ReinterpretCast<int8_t>();
    for(uint32_t limb=0;limb<pinv::kCubePlanes;++limb){
        Cast(small,planes[limb],RoundMode::CAST_RINT,count);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        uint32_t destination=RIGHT?core*pinv::kCubePlanes*n+limb*n:limb*n*n+core*n;
        uint32_t step=RIGHT?(cores*pinv::kCubePlanes-1)*n:(cores-1)*n;
        DataCopyPad(digits[destination],small,DataCopyExtParams{static_cast<uint16_t>(owned),n,0,step,0});
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
    }
    Mul(scratch[0],values,values,count);PipeBarrier<PIPE_V>();
    ReduceSum(scratch[2],scratch[0],scratch[1],count);
    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    return scratch[2].GetValue(0);
}

__aicore__ inline void PinvCubeBuildResidual(GM_ADDR ws,const LinalgPinvTilingData &data,
    GlobalTensor<float> inverse,uint32_t offset,LocalTensor<float> cache,
    LocalTensor<float> high,LocalTensor<float> low,LocalTensor<float> scratch[],
    uint32_t owned,uint32_t core,uint32_t cores,float &inverseScale,float &sumErrorA,float &sumErrorX){
    uint32_t n=data.cols,count=owned*n;auto base=PinvCubeArea(ws,data);
    GlobalTensor<int8_t> a,b;GlobalTensor<int32_t> c,control;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+pinv::kCubeControlFloats));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+pinv::kCubeControlFloats)+pinv::CubePlaneElements(n));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base+pinv::kCubeControlFloats)+pinv::CubeProductOffset(n));
    control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base));
    float localErrorA=PinvCubeQuantize<false>(cache,scratch,a,n,owned,core,cores,pinv::kCubeInputScale);
    SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
    DataCopyPad(cache,inverse[offset+core*n],
        DataCopyExtParams{static_cast<uint16_t>(owned),n*4,(cores-1)*n*4,0,0},
        DataCopyPadExtParams<float>{false,0,0,0});
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    inverseScale=PinvCubePeakScale(base,cache,scratch,count,core,cores,inverseScale);
    float localErrorX=PinvCubeQuantize<true>(cache,scratch,b,n,owned,core,cores,inverseScale);
    auto statistic=scratch[7];statistic.SetValue(0,localErrorA);statistic.SetValue(1,localErrorX);
    GlobalTensor<float> statistics;statistics.SetGlobalBuffer(base);
    SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyPad(statistics[32+core*8],statistic,DataCopyExtParams{1,8,0,0,0});
    SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
    if(core==0){
        control.SetValue(0,1);
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
    }
    SyncAll();PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag<2>(10);SyncAll();
    SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
    DataCopy(statistic,statistics[32],cores*8);
    SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
    sumErrorA=0.0f;sumErrorX=0.0f;
    for(uint32_t peer=0;peer<cores;++peer){sumErrorA+=statistic.GetValue(peer*8);sumErrorX+=statistic.GetValue(peer*8+1);}
    auto raw=scratch[0].ReinterpretCast<int32_t>();auto part=scratch[1],sum=scratch[2],bv=scratch[3];
    auto av=scratch[4],err=scratch[5],alt=scratch[6];
    float rowWeight=pinv::kCubeInputScale*inverseScale*pinv::kCubeLeadingWeight;
    for(uint32_t i=0;i<pinv::kCubePlanes;++i){
        float weight=rowWeight;
        for(uint32_t j=0;j<pinv::kCubePlanes;++j){
            SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(raw,c[(i*n+core)*pinv::kCubePlanes*n+j*n],
                DataCopyExtParams{static_cast<uint16_t>(owned),n*4,(cores*pinv::kCubePlanes-1)*n*4,0,0},
                DataCopyPadExtParams<int32_t>{false,0,0,0});
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            Cast(part,raw,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
            Muls(part,part,-weight,count);PipeBarrier<PIPE_V>();
            Add(sum,high,part,count);PipeBarrier<PIPE_V>();
            Sub(bv,sum,high,count);PipeBarrier<PIPE_V>();
            Sub(av,sum,bv,count);PipeBarrier<PIPE_V>();
            Sub(err,high,av,count);Sub(alt,part,bv,count);PipeBarrier<PIPE_V>();
            Add(err,err,alt,count);PipeBarrier<PIPE_V>();
            Add(low,low,err,count);DataCopy(high,sum,count);PipeBarrier<PIPE_V>();
            weight*=pinv::kCubeReciprocalRadix;
        }
        rowWeight*=pinv::kCubeReciprocalRadix;
    }
}
#endif
template<bool ENABLE>
class PinvCubeGuard{
    GM_ADDR ws;LinalgPinvTilingData data;
public:
    __aicore__ inline PinvCubeGuard(GM_ADDR workspace,const LinalgPinvTilingData &t){
        if constexpr(ENABLE){ws=workspace;data=t;}
    }
    __aicore__ inline ~PinvCubeGuard(){
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & PINV_KEY_CUBE) && (TILING_KEY_VAR & PINV_KEY_WINDOW)
        if constexpr(ENABLE)PinvCubeStop(ws,data);
#endif
    }
};
// AIV uses two logical blocks per physical block in paired mixed kernels.
// Cube services always call AscendC::GetBlockNum directly.
__aicore__ inline uint32_t PinvVectorBlockCount() {
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & PINV_KEY_CUBE)
    return AscendC::GetBlockNum() * 2;
#else
    return AscendC::GetBlockNum();
#endif
}
