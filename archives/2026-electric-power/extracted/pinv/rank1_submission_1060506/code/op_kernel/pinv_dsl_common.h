#pragma once
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#pragma once
#include "kernel_operator.h"
using namespace AscendC;
namespace tl::ascend {
// These adapters contain individual transfers/instructions, never a solver.
__aicore__ inline void pv_fill(LocalTensor<float> d, LocalTensor<float>, LocalTensor<float>, float v,int n){Duplicate(d,v,n);}
__aicore__ inline void pv_abs(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float>,int n){Abs(d,a,n);}
__aicore__ inline void pv_scale(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float>,float v,int n){Muls(d,a,v,n);}
__aicore__ inline void pv_mul(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b,int n){Mul(d,a,b,n);}
__aicore__ inline void pv_add(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b,int n){Add(d,a,b,n);}
__aicore__ inline void pv_div(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b,int n){Div(d,a,b,n);}
__aicore__ inline void pv_copy(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float>,int n){DataCopy(d,a,n);}
__aicore__ inline void pv_copy_ub_rows(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int rows,int width,int sourceGap,int destinationGap){DataCopy(d,a,DataCopyParams{uint16_t(rows),uint16_t(width/8),uint16_t(sourceGap/8),uint16_t(destinationGap/8)});}
__aicore__ inline void pv_sqrt(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float>,int n){Sqrt(d,a,n);}
__aicore__ inline void pv_reduce_max(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> w,int n){ReduceMax(d,a,w,n,false);}
__aicore__ inline void pv_reduce_sum(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> w,int n){ReduceSum(d,a,w,n);}
__aicore__ inline void pv_load_rows(LocalTensor<float> d, GlobalTensor<float> a, LocalTensor<float>,int rows,int cols,int sourceGap,int destinationGap,bool padding=true){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*4),uint32_t(sourceGap*4),uint32_t(destinationGap/8),0},DataCopyPadExtParams<float>{padding,0,uint8_t((8-cols%8)%8),0});}
__aicore__ inline void pv_store_rows(GlobalTensor<float> d,LocalTensor<float> a,GlobalTensor<float>,int rows,int cols,int sourceGap,int destinationGap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*4),uint32_t(sourceGap/8),uint32_t(destinationGap*4),0});}
__aicore__ inline void pv_gather(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<uint32_t> offsets,uint32_t base,int n){Gather(d,a,offsets,base,n);}
__aicore__ inline void pv_max_available(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,uint32_t bits){uint64_t mask[2]={bits,0};WholeReduceMax(d,a,mask,1,1,1,8);}
__aicore__ inline void pv_max_available64(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,uint64_t bits){uint64_t mask[2]={bits,0};WholeReduceMax(d,a,mask,1,1,1,8);}
__aicore__ inline void pv_max_two32(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,uint32_t lo,uint32_t hi){uint64_t mask[2]={uint64_t(lo)|(uint64_t(hi)<<32),0};WholeReduceMax(d,a,mask,1,1,1,8);}
__aicore__ inline void pv_mask_init(LocalTensor<uint32_t> d,LocalTensor<uint32_t>,LocalTensor<uint32_t>,int n){uint64_t bits=n==64?~uint64_t(0):(uint64_t(1)<<n)-1;d.SetValue(0,uint32_t(bits));d.SetValue(1,uint32_t(bits>>32));}
__aicore__ inline void pv_mask_clear(LocalTensor<uint32_t> d,LocalTensor<uint32_t>,LocalTensor<uint32_t>,int bit){uint32_t index=uint32_t(bit)/32;d.SetValue(index,d.GetValue(index)&~(uint32_t(1)<<(uint32_t(bit)%32)));}
__aicore__ inline void pv_max_mask_buffer(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<uint32_t> bits){uint64_t mask[2]={uint64_t(bits.GetValue(0))|(uint64_t(bits.GetValue(1))<<32),0};WholeReduceMax(d,a,mask,1,1,1,8);}
// The DSL stores raw offset words in the shared flat pool at native offsets.
__aicore__ inline void pv_gather_bits(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float> offsets,uint32_t base,int n){Gather(d,a,offsets.ReinterpretCast<uint32_t>(),base,n);}
struct PvLaneMask {uint64_t bits;__aicore__ inline explicit PvLaneMask(int n):bits(n==64?~uint64_t(0):(uint64_t(1)<<n)-1){}};
__aicore__ inline void PvMaskReduce(LocalTensor<float> d,LocalTensor<float> a,const PvLaneMask &state){uint64_t mask[2]={state.bits,0};WholeReduceMax(d,a,mask,1,1,1,8);}
__aicore__ inline void PvMaskRemove(PvLaneMask &state,int bit){state.bits&=~(uint64_t(1)<<bit);}
__aicore__ inline void pv_reduce_registers(LocalTensor<float> d,LocalTensor<float>,LocalTensor<float>){float best,bits;GetReduceRepeatMaxMinSpr<float>(best,bits);d.SetValue(0,best);d.SetValue(1,bits);}
// Scalar-output ABI for the same SDK instruction. Allocation, lifetime and
// TileLang alloc_var controls synchronization of these scalar values.
__aicore__ inline void pv_reduce_scalars(LocalTensor<float>,LocalTensor<float>,LocalTensor<float>,float &best,int32_t &position){float bits;GetReduceRepeatMaxMinSpr<float>(best,bits);union {float value;uint32_t word;} index;index.value=bits;position=int32_t(index.word);}
__aicore__ inline void pv_brcb_rows(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int n,int pitch){Brcb(d,a,uint8_t(n/8),BrcbRepeatParams{uint8_t(pitch/8),uint8_t(pitch)});}
__aicore__ inline void pv_gj_update(LocalTensor<float> d,LocalTensor<float> pivot,LocalTensor<float> factors,int width,int n,int pitch){MulAddDst(d,pivot,factors,uint64_t(width),uint8_t(n),BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,uint8_t(pitch/8)});}
__aicore__ inline void pv_square_rows(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int n,int pitch){Mul(d,a,a,uint64_t(n),uint8_t(n),BinaryRepeatParams{1,1,1,uint8_t(pitch/8),uint8_t(pitch/8),uint8_t(pitch/8)});}
__aicore__ inline void pv_square_padded_rows(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int width,int rows,int pitch){Mul(d,a,a,uint64_t(width),uint8_t(rows),BinaryRepeatParams{1,1,1,uint8_t(pitch/8),uint8_t(pitch/8),uint8_t(pitch/8)});}
__aicore__ inline void pv_pack15(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int rows,int pitch){uint64_t kept=0;GatherMask(d,a,uint8_t(7),true,rows,{1,uint8_t(rows),uint16_t(pitch/8),0},kept);}
__aicore__ inline void pv_mad_strided(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float> b,int mask,int repeats,int db,int ab,int bb,int dr,int ar,int br){MulAddDst(d,a,b,uint64_t(mask),uint8_t(repeats),BinaryRepeatParams{uint8_t(db),uint8_t(ab),uint8_t(bb),uint8_t(dr),uint8_t(ar),uint8_t(br)});}
}
#define pv_mask_begin(d,a,b,n) PvLaneMask pvLaneMask(n)
#define pv_mask_reduce(d,a,b) PvMaskReduce(d,a,pvLaneMask)
#define pv_mask_remove(d,a,b,pivot) PvMaskRemove(pvLaneMask,pivot)


namespace tl::ascend {
__aicore__ inline void bl_index_brcb(LocalTensor<float>d,LocalTensor<uint32_t>a,LocalTensor<float>,int n){Brcb(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),uint8_t(n/8),BrcbRepeatParams{1,8});}
__aicore__ inline void bl_index_matrix(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<uint32_t>b,int width,int rows){Add(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),uint64_t(width),uint8_t(rows),BinaryRepeatParams{1,0,1,uint8_t(width/8),1,0});}
__aicore__ inline void bl_gather(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<float>indices,int n){Gather(d,a,indices.ReinterpretCast<uint32_t>(),uint32_t(0),n);}
}

#pragma once
#include "kernel_operator.h"
namespace tl::ascend {
// A uint64 register transports two FP32 words; no FP64 arithmetic is used.
struct PvWide { float h,l; };
union PvWideWord { PvWide pair; uint64_t word; };
__aicore__ inline uint64_t pv_dd_pack(float h,float l){PvWideWord x;x.pair={h,l};return x.word;}
__aicore__ inline float pv_dd_h(uint64_t a){PvWideWord x;x.word=a;return x.pair.h;}
__aicore__ inline float pv_dd_l(uint64_t a){PvWideWord x;x.word=a;return x.pair.l;}
__aicore__ inline uint64_t pv_dd_plus(uint64_t left,uint64_t right){
    PvWideWord x,y;x.word=left;y.word=right;PvWide a=x.pair,b=y.pair;
    volatile float sum=a.h+b.h;
    volatile float bv=sum-a.h,av=sum-bv,br=b.h-bv,ar=a.h-av;
    volatile float e=ar+br;e=e+a.l;e=e+b.l;
    volatile float h=sum+e,z=h-sum;
    return pv_dd_pack(h,e-z);
}
__aicore__ inline uint64_t pv_dd_minus(uint64_t a,uint64_t b){return pv_dd_plus(a,pv_dd_pack(-pv_dd_h(b),-pv_dd_l(b)));}
__aicore__ inline uint64_t pv_dd_times(uint64_t left,uint64_t right){
    PvWideWord x,y;x.word=left;y.word=right;PvWide a=x.pair,b=y.pair;
    volatile float ahc=4097.0f*a.h,bhc=4097.0f*b.h;
    volatile float ahv=ahc-a.h,bhv=bhc-b.h;
    volatile float ah=ahc-ahv,bh=bhc-bhv,al=a.h-ah,bl=b.h-bh;
    volatile float p=a.h*b.h,e=ah*bh;e=e-p;
    volatile float x1=ah*bl;e=e+x1;
    volatile float x2=al*bh;e=e+x2;
    volatile float x3=al*bl;e=e+x3;
    volatile float x4=a.h*b.l;e=e+x4;
    volatile float x5=a.l*b.h;e=e+x5;
    return pv_dd_plus(pv_dd_pack(p,0),pv_dd_pack(e,0));
}
__aicore__ inline uint64_t pv_dd_divide(uint64_t a,uint64_t b){
    float q=pv_dd_h(a)/pv_dd_h(b);
    uint64_t residual=pv_dd_minus(a,pv_dd_times(b,pv_dd_pack(q,0)));
    return pv_dd_plus(pv_dd_pack(q,0),pv_dd_pack(pv_dd_h(residual)/pv_dd_h(b),0));
}
}
// Scalar arithmetic results use explicit FP32-pair buffers because the
// pinned backend's custom instruction hook is a statement, not an expression.
namespace tl::ascend {
__aicore__ inline uint64_t PvReadPair(AscendC::LocalTensor<float> a){return pv_dd_pack(a.GetValue(0),a.GetValue(1));}
__aicore__ inline void PvWritePair(AscendC::LocalTensor<float> d,uint64_t a){d.SetValue(0,pv_dd_h(a));d.SetValue(1,pv_dd_l(a));}
__aicore__ inline void pv_pair_pack(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,float h,float l){d.SetValue(0,h);d.SetValue(1,l);}
__aicore__ inline void pv_pair_plus(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b){PvWritePair(d,pv_dd_plus(PvReadPair(a),PvReadPair(b)));}
__aicore__ inline void pv_pair_minus(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b){PvWritePair(d,pv_dd_minus(PvReadPair(a),PvReadPair(b)));}
__aicore__ inline void pv_pair_times(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b){PvWritePair(d,pv_dd_times(PvReadPair(a),PvReadPair(b)));}
__aicore__ inline void pv_pair_divide(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b){PvWritePair(d,pv_dd_divide(PvReadPair(a),PvReadPair(b)));}
}

#pragma once
// Elementary compensated dot/scale/add primitives from ec5ec7a. All
// decomposition, truncation, certification and reconstruction control is DSL.
// This type has no inverse, SVD, Jacobi sweep or operator entry point.
#include "kernel_operator.h"
namespace tl::ascend {
class PvArithmetic {
public:
    using Wide=PvWide;
    AscendC::LocalTensor<float> pool,calc[8],red,bh,bl,certH,certL,certT;
    __aicore__ inline void VS(){AscendC::SetFlag<AscendC::HardEvent::V_S>(0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);}
    __aicore__ inline void SV(){AscendC::SetFlag<AscendC::HardEvent::S_V>(0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);}
    __aicore__ inline Wide Unpack(uint64_t a){PvWideWord x;x.word=a;return x.pair;}
    __aicore__ inline uint64_t Pack(Wide a){return pv_dd_pack(a.h,a.l);}
    __aicore__ inline Wide Plus(Wide a,Wide b){return Unpack(pv_dd_plus(Pack(a),Pack(b)));}
    __aicore__ inline Wide Minus(Wide a,Wide b){return Unpack(pv_dd_minus(Pack(a),Pack(b)));}
    __aicore__ inline Wide Times(Wide a,Wide b){return Unpack(pv_dd_times(Pack(a),Pack(b)));}
    __aicore__ inline Wide Divide(Wide a,Wide b){return Unpack(pv_dd_divide(Pack(a),Pack(b)));}
    __aicore__ inline float Root(float x){red.SetValue(0,x>0?x:0);SV();AscendC::Sqrt(red,red,1);VS();return red.GetValue(0);}
    __aicore__ inline Wide RootPair(Wide x){float h=Root(x.h);if(!(h>0))return {0,0};return Plus({h,0},Divide(Minus(x,Times({h,0},{h,0})),{2*h,0}));}
    __aicore__ inline void Init(AscendC::LocalTensor<float> p,int width){
        pool=p;bh=p[6*width];bl=p[7*width];
        for(int i=0;i<8;++i)calc[i]=p[(8+i)*width];
        certH=p[16*width];certL=p[17*width];certT=p[18*width];red=p[20*width];
    }
    __aicore__ inline void PairAdd(LocalTensor<float> h,LocalTensor<float> l,
        LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        auto sum=calc[0],bv=calc[1],av=calc[2],e=calc[3],v=calc[4];
        Add(sum,a,c,count);PipeBarrier<PIPE_V>();Sub(bv,sum,a,count);PipeBarrier<PIPE_V>();
        Sub(av,sum,bv,count);Sub(e,c,bv,count);PipeBarrier<PIPE_V>();
        Sub(v,a,av,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Add(e,e,b,count);PipeBarrier<PIPE_V>();Add(e,e,d,count);PipeBarrier<PIPE_V>();
        Add(h,sum,e,count);PipeBarrier<PIPE_V>();Sub(v,h,sum,count);PipeBarrier<PIPE_V>();
        Sub(l,e,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Scale(LocalTensor<float> h,LocalTensor<float> l,
        LocalTensor<float> a,LocalTensor<float> b,Wide factor,uint32_t count){
        auto hi=calc[0],lo=calc[1],p=calc[2],e=calc[3],v=calc[4],z=calc[5];
        float split=factor.h*4097.0f,fh=split-(split-factor.h),fl=factor.h-fh;
        Muls(hi,a,4097.0f,count);PipeBarrier<PIPE_V>();Sub(lo,hi,a,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);PipeBarrier<PIPE_V>();Sub(lo,a,hi,count);
        Muls(p,a,factor.h,count);Muls(e,hi,fh,count);PipeBarrier<PIPE_V>();Sub(e,e,p,count);PipeBarrier<PIPE_V>();
        Muls(v,hi,fl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,fh,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,fl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Muls(v,a,factor.l,count);Muls(z,b,factor.h,count);PipeBarrier<PIPE_V>();
        Add(e,e,v,count);PipeBarrier<PIPE_V>();Add(e,e,z,count);PipeBarrier<PIPE_V>();
        Muls(v,b,factor.l,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Add(h,p,e,count);PipeBarrier<PIPE_V>();Sub(v,h,p,count);PipeBarrier<PIPE_V>();Sub(l,e,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline Wide Dot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        auto p=calc[0],e=calc[1],hi=calc[2],lo=calc[3],ch=calc[4],cl=calc[5],v=calc[6],z=calc[7];
        Muls(hi,a,4097.0f,count);Muls(ch,c,4097.0f,count);PipeBarrier<PIPE_V>();
        Sub(lo,hi,a,count);Sub(cl,ch,c,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);Sub(ch,ch,cl,count);PipeBarrier<PIPE_V>();
        Sub(lo,a,hi,count);Sub(cl,c,ch,count);Mul(p,a,c,count);PipeBarrier<PIPE_V>();
        Mul(e,hi,ch,count);PipeBarrier<PIPE_V>();Sub(e,e,p,count);PipeBarrier<PIPE_V>();
        Mul(v,hi,cl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,ch,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,cl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Mul(v,a,d,count);Mul(z,b,c,count);PipeBarrier<PIPE_V>();
        Add(e,e,v,count);PipeBarrier<PIPE_V>();Add(e,e,z,count);PipeBarrier<PIPE_V>();
        Mul(v,b,d,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        // Pairwise TwoSum keeps the FP32 reduction error in the low component.
        for(uint32_t size=count;size>8;size/=2){
            uint32_t half=size/2;
            Add(hi,p,p[half],half);PipeBarrier<PIPE_V>();Sub(lo,hi,p,half);PipeBarrier<PIPE_V>();
            Sub(ch,hi,lo,half);Sub(cl,p[half],lo,half);PipeBarrier<PIPE_V>();
            Sub(v,p,ch,half);PipeBarrier<PIPE_V>();Add(cl,cl,v,half);PipeBarrier<PIPE_V>();
            Add(cl,cl,e,half);PipeBarrier<PIPE_V>();Add(cl,cl,e[half],half);PipeBarrier<PIPE_V>();
            Add(p,hi,cl,half);PipeBarrier<PIPE_V>();Sub(v,p,hi,half);PipeBarrier<PIPE_V>();
            Sub(e,cl,v,half);PipeBarrier<PIPE_V>();
        }
        VS();Wide sum={0,0};for(uint32_t i=0;i<8;++i)sum=Plus(sum,{p.GetValue(i),e.GetValue(i)});
        return sum;
    }
// Three-component arithmetic is confined to posterior residual measurements.
// Error-free FP32 products and TwoSum leave a third-order accumulation remainder.
    __aicore__ inline void CertTwoSum(LocalTensor<float> sum,LocalTensor<float> error,
        LocalTensor<float> a,LocalTensor<float> b,uint32_t count){
        auto v=calc[6],w=calc[7];
        Add(sum,a,b,count);PipeBarrier<PIPE_V>();Sub(v,sum,a,count);PipeBarrier<PIPE_V>();
        Sub(w,sum,v,count);PipeBarrier<PIPE_V>();Sub(w,a,w,count);Sub(v,b,v,count);PipeBarrier<PIPE_V>();
        Add(error,w,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CertAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,
        LocalTensor<float> value,uint32_t count){
        CertTwoSum(calc[0],calc[1],a,value,count);
        CertTwoSum(calc[2],calc[3],b,calc[1],count);
        CertTwoSum(calc[4],calc[5],c,calc[3],count);
        CertTwoSum(a,calc[2],calc[0],calc[2],count);
        CertTwoSum(b,c,calc[2],calc[4],count);
    }
    __aicore__ inline void CertProduct(LocalTensor<float> p,LocalTensor<float> error,
        LocalTensor<float> a,LocalTensor<float> b,uint32_t count){
        auto hi=calc[0],lo=calc[1],ch=calc[2],cl=calc[3],v=calc[4];
        Muls(hi,a,4097.0f,count);Muls(ch,b,4097.0f,count);PipeBarrier<PIPE_V>();
        Sub(lo,hi,a,count);Sub(cl,ch,b,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);Sub(ch,ch,cl,count);PipeBarrier<PIPE_V>();
        Sub(lo,a,hi,count);Sub(cl,b,ch,count);Mul(p,a,b,count);PipeBarrier<PIPE_V>();
        Mul(error,hi,ch,count);PipeBarrier<PIPE_V>();Sub(error,error,p,count);PipeBarrier<PIPE_V>();
        Mul(v,hi,cl,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,ch,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,cl,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CertProductScalar(LocalTensor<float> p,LocalTensor<float> error,
        LocalTensor<float> a,float b,uint32_t count){
        auto hi=calc[0],lo=calc[1],v=calc[2];
        volatile float split=b*4097.0f;
        volatile float temp=split-b;
        volatile float ch=split-temp;
        volatile float cl=b-ch;
        Muls(hi,a,4097.0f,count);PipeBarrier<PIPE_V>();Sub(lo,hi,a,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);PipeBarrier<PIPE_V>();Sub(lo,a,hi,count);Muls(p,a,b,count);PipeBarrier<PIPE_V>();
        Muls(error,hi,static_cast<float>(ch),count);PipeBarrier<PIPE_V>();Sub(error,error,p,count);PipeBarrier<PIPE_V>();
        Muls(v,hi,static_cast<float>(cl),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,static_cast<float>(ch),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,static_cast<float>(cl),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline Wide CertScalarSum(float a,float b){
        volatile float sum=a+b;
        volatile float v=sum-a;
        volatile float w=sum-v;
        volatile float ar=a-w;
        volatile float br=b-v;
        return {sum,ar+br};
    }
    __aicore__ inline void CertScalarAdd(float &a,float &b,float &c,float value){
        Wide first=CertScalarSum(a,value),second=CertScalarSum(b,first.l),third=CertScalarSum(c,second.l);
        first=CertScalarSum(first.h,second.h);
        second=CertScalarSum(first.l,third.h);
        a=first.h;b=second.h;c=second.l;
    }
    __aicore__ inline void CertScale(LocalTensor<float> a,Wide factor,uint32_t count){
        Duplicate(certH,0.0f,count);Duplicate(certL,0.0f,count);Duplicate(certT,0.0f,count);PipeBarrier<PIPE_V>();
        CertProductScalar(bh,bl,a,factor.h,count);
        CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
        CertProductScalar(bh,bl,a,factor.l,count);
        CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
    }
    __aicore__ inline Wide CertDot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        Duplicate(certH,0.0f,count);Duplicate(certL,0.0f,count);Duplicate(certT,0.0f,count);PipeBarrier<PIPE_V>();
        LocalTensor<float> left[2]={a,b},right[2]={c,d};
        for(uint32_t i=0;i<2;++i)for(uint32_t j=0;j<2;++j){
            CertProduct(bh,bl,left[i],right[j],count);
            CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
        }
        for(uint32_t size=count;size>8;size/=2){
            uint32_t half=size/2;
            CertAdd(certH,certL,certT,certH[half],half);
            CertAdd(certH,certL,certT,certL[half],half);
            CertAdd(certH,certL,certT,certT[half],half);
        }
        VS();float sh=0,sl=0,st=0;
        for(uint32_t i=0;i<8;++i){
            CertScalarAdd(sh,sl,st,certH.GetValue(i));
            CertScalarAdd(sh,sl,st,certL.GetValue(i));
            CertScalarAdd(sh,sl,st,certT.GetValue(i));
        }
        return Plus({sh,sl},{st,0});
    }

    __aicore__ inline void Eliminate(LocalTensor<float> x,LocalTensor<float> y,LocalTensor<float> u,LocalTensor<float> v,Wide factor,uint32_t s) {
        auto qh = calc[0], ql = calc[1], bh = calc[2], bl = calc[3], e = calc[4], part = calc[5], product = calc[6], aux = calc[7];
        volatile float ac = factor.h * 4097.0f;
        volatile float az = ac - factor.h;
        volatile float ah = ac - az;
        volatile float al = factor.h - ah;
        Muls(bh, u, 4097.0f, s); PipeBarrier<PIPE_V>();
        Sub(bl, bh, u, s); PipeBarrier<PIPE_V>(); Sub(bh, bh, bl, s); PipeBarrier<PIPE_V>();
        Sub(bl, u, bh, s); Muls(product, u, factor.h, s); PipeBarrier<PIPE_V>();
        Muls(e, bh, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Sub(e, e, product, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bh, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, v, factor.h, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, u, factor.l, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(qh, product, e, s); PipeBarrier<PIPE_V>(); Sub(ql, qh, product, s); PipeBarrier<PIPE_V>(); Sub(ql, e, ql, s); PipeBarrier<PIPE_V>();
        Sub(bh, x, qh, s); PipeBarrier<PIPE_V>(); Sub(bl, bh, x, s); PipeBarrier<PIPE_V>();
        Sub(e, bh, bl, s); PipeBarrier<PIPE_V>(); Sub(e, x, e, s); Muls(part, qh, -1.0f, s); PipeBarrier<PIPE_V>();
        Sub(part, part, bl, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(e, e, y, s); PipeBarrier<PIPE_V>(); Sub(e, e, ql, s); PipeBarrier<PIPE_V>();
        Add(x, bh, e, s); PipeBarrier<PIPE_V>(); Sub(aux, x, bh, s); PipeBarrier<PIPE_V>(); Sub(y, e, aux, s); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Normalize(LocalTensor<float> x,LocalTensor<float> y,LocalTensor<float> u,LocalTensor<float> v,Wide divisor,uint32_t s) {
        float reciprocal = 1.0f / divisor.h;
        Muls(u, x, reciprocal, s); Duplicate(v, 0.0f, s); PipeBarrier<PIPE_V>();
        Eliminate(x,y,u,v,divisor,s);
        Muls(calc[0], x, reciprocal, s); PipeBarrier<PIPE_V>();
        Add(x, u, calc[0], s); PipeBarrier<PIPE_V>();
        Sub(y, x, u, s); PipeBarrier<PIPE_V>();
        Sub(y, calc[0], y, s); PipeBarrier<PIPE_V>();
    }

};

__aicore__ inline void PvDotStore(PvArithmetic& p,AscendC::LocalTensor<float> dst,int ah,int al,int bh,int bl,int n){PvWritePair(dst,p.Pack(p.Dot(p.pool[ah],p.pool[al],p.pool[bh],p.pool[bl],n)));}
__aicore__ inline void PvCertDotStore(PvArithmetic& p,AscendC::LocalTensor<float> dst,int ah,int al,int bh,int bl,int n){PvWritePair(dst,p.Pack(p.CertDot(p.pool[ah],p.pool[al],p.pool[bh],p.pool[bl],n)));}
__aicore__ inline void PvRootPairStore(PvArithmetic& p,AscendC::LocalTensor<float> dst,AscendC::LocalTensor<float> a){PvWritePair(dst,p.Pack(p.RootPair(p.Unpack(PvReadPair(a)))));}
__aicore__ inline void PvRootStore(PvArithmetic& p,AscendC::LocalTensor<float> dst,float a){dst.SetValue(0,p.Root(a));}
__aicore__ inline void PvScaleAt(PvArithmetic& p,int h,int l,int a,int b,float fh,float fl,int n){p.Scale(p.pool[h],p.pool[l],p.pool[a],p.pool[b],{fh,fl},n);}
__aicore__ inline void PvAddAt(PvArithmetic& p,int h,int l,int a,int b,int c,int d,int n){p.PairAdd(p.pool[h],p.pool[l],p.pool[a],p.pool[b],p.pool[c],p.pool[d],n);}
__aicore__ inline void PvCertScaleAt(PvArithmetic& p,int a,float fh,float fl,int n){p.CertScale(p.pool[a],{fh,fl},n);}
__aicore__ inline void PvCertAddAt(PvArithmetic& p,int a,int b,int c,int value,int n){p.CertAdd(p.pool[a],p.pool[b],p.pool[c],p.pool[value],n);}
__aicore__ inline void PvNormalizeAt(PvArithmetic& p,int h,int l,int u,int v,float fh,float fl,int n){p.Normalize(p.pool[h],p.pool[l],p.pool[u],p.pool[v],{fh,fl},n);}
__aicore__ inline void PvEliminateAt(PvArithmetic& p,int h,int l,int u,int v,float fh,float fl,int n){p.Eliminate(p.pool[h],p.pool[l],p.pool[u],p.pool[v],{fh,fl},n);}
}
#define pv_arith_init(d,a,b,width) PvArithmetic pvArithmetic; pvArithmetic.Init(d,width)
#define pv_pair_dot(d,a,b,...) PvDotStore(pvArithmetic,d,__VA_ARGS__)
#define pv_pair_cert_dot(d,a,b,...) PvCertDotStore(pvArithmetic,d,__VA_ARGS__)
#define pv_pair_root(d,a,b) PvRootPairStore(pvArithmetic,d,a)
#define pv_root(d,a,b,value) PvRootStore(pvArithmetic,d,value)
#define pv_pair_scale(d,a,b,...) PvScaleAt(pvArithmetic,__VA_ARGS__)
#define pv_pair_add(d,a,b,...) PvAddAt(pvArithmetic,__VA_ARGS__)
#define pv_cert_scale(d,a,b,...) PvCertScaleAt(pvArithmetic,__VA_ARGS__)
#define pv_cert_add(d,a,b,...) PvCertAddAt(pvArithmetic,__VA_ARGS__)

#define pv_pair_normalize(d,a,b,...) PvNormalizeAt(pvArithmetic,__VA_ARGS__)
#define pv_pair_eliminate(d,a,b,...) PvEliminateAt(pvArithmetic,__VA_ARGS__)

#pragma once
#include "kernel_operator.h"
namespace tl::ascend {
// Individual transfers, reductions and hardware synchronization only.
template<class D,class A,class B> __aicore__ inline void pv_all_barrier(D,A,B){AscendC::PipeBarrier<PIPE_ALL>();}
template<class D,class A,class B> __aicore__ inline void pv_sync_all(D,A,B){AscendC::SyncAll();}
__aicore__ inline void PvSetSyncBase(uint64_t address){if(address)AscendC::SetSyncBaseAddr(address);}
template<class D,class A,class B> __aicore__ inline void pv_cross_set0(D,A,B,int flag){AscendC::CrossCoreSetFlag<0,PIPE_MTE3>(flag);}
template<class D,class A,class B> __aicore__ inline void pv_cross_set1(D,A,B,int flag){AscendC::CrossCoreSetFlag<1,PIPE_MTE3>(flag);}
template<class D,class A,class B> __aicore__ inline void pv_cross_set2(D,A,B,int flag){AscendC::CrossCoreSetFlag<2,PIPE_MTE3>(flag);}
template<class D,class A,class B> __aicore__ inline void pv_cross_wait(D,A,B,int flag){AscendC::CrossCoreWaitFlag(flag);}
template<class D,class A,class B> __aicore__ inline void pv_cross_wait2(D,A,B,int flag){AscendC::CrossCoreWaitFlag<2>(flag);}
__aicore__ inline void pv_gather_offsets(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> offsets,int base,int n){AscendC::Gather(d,a,offsets.ReinterpretCast<uint32_t>(),uint32_t(base),n);}
__aicore__ inline void pv_iota(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int n){AscendC::CreateVecIndex(d.ReinterpretCast<int32_t>(),int32_t(0),n);}
__aicore__ inline void pv_scale_indices(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float>,int scale,int n){AscendC::Muls(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),scale,n);}
__aicore__ inline void pv_store_u32(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int index,uint32_t value){d.ReinterpretCast<uint32_t>().SetValue(index,value);}
// Each binding emits one reduction; its chunking and dependency schedule live
// in reduction.py. The mask adapter only encodes the hardware's even lanes.
__aicore__ inline void pv_whole_max(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float>,int mask,int repeats){AscendC::WholeReduceMax(d,a,uint64_t(mask),uint8_t(repeats),1,1,8);}
__aicore__ inline void pv_even_max(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float>,int count){uint64_t mask[2]={uint64_t(0x5555)&((uint64_t(1)<<(count*2))-1),0};AscendC::WholeReduceMax(d,a,mask,1,1,1,8);}

}
#define pv_sync_init(d,a,b) PvSetSyncBase(fftsAddr)

namespace tl::ascend {
__aicore__ inline void pp_clear_history(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int count,int pitch){AscendC::Duplicate(d,0.0f,uint64_t(8),uint8_t(count),1,uint8_t(pitch));}
}

using uint=unsigned int;using uchar=unsigned char;using ushort=unsigned short;
namespace tl::ascend {
__aicore__ inline void PvDslPipelineRows( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, uint32_t n, uint32_t stride, uint32_t all_batch, uint32_t job, uint32_t core, uint32_t cores, uint32_t window, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  uint32_t k = (uint32_t)0;
  uint32_t wide = (uint32_t)0;
  uint32_t owned = (uint32_t)0;
  uint32_t factor_rows = (uint32_t)0;
  uint32_t packet_stride = (uint32_t)0;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  int valid = 0;
  float power = 0.000000e+00f;
  uint32_t window_end = (uint32_t)0;
  float inverse = 0.000000e+00f;
  float peak = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    k = ((((n + (uint32_t)8) - (uint32_t)1) / (uint32_t)8) * (uint32_t)8);
    wide = ((((k + (uint32_t)64) - (uint32_t)1) / (uint32_t)64) * (uint32_t)64);
    owned = ((((n + cores) - (uint32_t)1) - core) / cores);
    factor_rows = ((((owned + (uint32_t)8) - (uint32_t)1) / (uint32_t)8) * (uint32_t)8);
    packet_stride = (k + (uint32_t)16);
    uint32_t packets = (owned * stride);
    uint32_t output = (packets + ((uint32_t)16 * packet_stride));
    uint32_t factors = (output + max((owned * k), (uint32_t)512));
    uint32_t meta = (factors + ((uint32_t)8 * max(factor_rows, (uint32_t)16)));
    uint32_t norms = (meta + (uint32_t)64);
    uint32_t vector = (norms + wide);
    uint32_t product = (vector + wide);
    uint32_t accumulator = (product + wide);
    uint32_t scratch = (accumulator + wide);
    uint32_t red = (scratch + wide);
    uint32_t row_indices = (red + (uint32_t)32);
    uint32_t packet_indices = (row_indices + factor_rows);
    uint32_t mapping = (packet_indices + (uint32_t)16);
    uint64_t work = ((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)stride));
    uint64_t metadata = ((((((uint64_t)all_batch) * (uint64_t)2) * ((uint64_t)n)) * ((uint64_t)stride)) + (((uint64_t)job) * (uint64_t)8192));
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (owned * stride));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[0], A[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(core * n)))], pool[0], owned, n, ((cores - (uint32_t)1) * n), (stride - k));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::pv_fill(pool[norms], pool[norms], pool[norms], 1.000000e+00f, k);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    for (uint32_t i = (uint32_t)0; i < factor_rows; ++i) {
      uint32_t condval;
      if ((i < owned)) {
        condval = ((i * stride) * (uint32_t)4);
      } else {
        condval = (uint32_t)0;
      }
      uint32_t index = condval;
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (row_indices + i), index);
    }
    for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (packet_indices + ((uint32_t)i_1)), ((((uint32_t)i_1) * packet_stride) * (uint32_t)4));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    valid = 1;
    power = 1.000000e+00f;
    window_end = n;
    if ((uint32_t)0 < window) {
      window_end = min(n, window);
    }
    for (uint32_t panel = (uint32_t)0; panel < (((n + (uint32_t)16) - (uint32_t)1) / (uint32_t)16); ++panel) {
      uint32_t first = (panel * (uint32_t)16);
      uint32_t rows = min((n - first), (uint32_t)16);
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_load_rows(pool[meta], W[(metadata + ((uint64_t)(panel * (uint32_t)64)))], pool[meta], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      if (pool.GetValue((meta + (uint32_t)34)) <= 5.000000e-01f) {
        valid = 0;
      }
      power = pool.GetValue((meta + (uint32_t)32));
      uint32_t condval_1;
      if (((uint32_t)0 < window)) {
        condval_1 = (first * packet_stride);
      } else {
        condval_1 = (uint32_t)0;
      }
      uint32_t packet_begin = condval_1;
      if (valid != 0) {
        if (first == (uint32_t)0) {
          for (uint32_t chunk = (uint32_t)0; chunk < ((((owned * stride) + (uint32_t)16320) - (uint32_t)1) / (uint32_t)16320); ++chunk) {
            uint32_t begin = (chunk * (uint32_t)16320);
            tl::ascend::pv_scale(pool[begin], pool[begin], pool[begin], (1.000000e+00f / power), min(((owned * stride) - begin), (uint32_t)16320));
          }
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        for (uint32_t column = (uint32_t)0; column < rows; ++column) {
          int32_t selected = ((int32_t)pool.GetValue((meta + column)));
          pool.SetValue((norms + ((uint32_t)selected)), (((float)((int32_t)((first + column) + (uint32_t)1))) * -1.000000e+00f));
          if ((((uint32_t)selected) % cores) == core) {
            pool.SetValue((((((((uint32_t)selected) / cores) * stride) + k) + first) + column), 1.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(W[((work + ((uint64_t)packet_begin)) + ((uint64_t)(column * packet_stride)))], pool[(((((uint32_t)selected) / cores) * stride) + first)], W[((work + ((uint64_t)packet_begin)) + ((uint64_t)(column * packet_stride)))], 1, packet_stride, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          }
        }
      }
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
      if (valid != 0) {
        tl::ascend::pv_load_rows(pool[packets], W[(work + ((uint64_t)packet_begin))], pool[packets], 1, (rows * packet_stride), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        for (uint32_t column_1 = (uint32_t)0; column_1 < rows; ++column_1) {
          tl::ascend::pv_gather_offsets(pool[product], pool[packets], pool[packet_indices], (column_1 * (uint32_t)4), rows);
          float reciprocal = pool.GetValue(((meta + (uint32_t)16) + column_1));
          tl::ascend::pv_scale(pool[(packets + (column_1 * packet_stride))], pool[(packets + (column_1 * packet_stride))], pool[(packets + (column_1 * packet_stride))], reciprocal, packet_stride);
          if ((column_1 + (uint32_t)1) < rows) {
            tl::ascend::pv_scale(pool[product], pool[product], pool[product], -1.000000e+00f, 16);
            tl::ascend::pv_brcb_rows(pool[factors], pool[product], pool[factors], 16, 8);
            for (uint32_t chunk_1 = (uint32_t)0; chunk_1 < (((packet_stride + (uint32_t)64) - (uint32_t)1) / (uint32_t)64); ++chunk_1) {
              uint32_t col = (chunk_1 * (uint32_t)64);
              tl::ascend::pv_mad_strided(pool[((packets + ((column_1 + (uint32_t)1) * packet_stride)) + col)], pool[((packets + (column_1 * packet_stride)) + col)], pool[(factors + ((column_1 + (uint32_t)1) * (uint32_t)8))], min((packet_stride - col), (uint32_t)64), ((rows - column_1) - (uint32_t)1), 1, 1, 0, (packet_stride / (uint32_t)8), 0, 1);
            }
          }
        }
        for (uint32_t column_2 = (uint32_t)0; column_2 < rows; ++column_2) {
          int32_t selected_1 = ((int32_t)pool.GetValue((meta + column_2)));
          if ((((uint32_t)selected_1) % cores) == core) {
            tl::ascend::pv_copy(pool[(((((uint32_t)selected_1) / cores) * stride) + first)], pool[(packets + (column_2 * packet_stride))], pool[(((((uint32_t)selected_1) / cores) * stride) + first)], packet_stride);
          }
          tl::ascend::pv_gather_offsets(pool[vector], pool[0], pool[row_indices], ((first + column_2) * (uint32_t)4), factor_rows);
          if ((((uint32_t)selected_1) % cores) == core) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            pool.SetValue((vector + (((uint32_t)selected_1) / cores)), 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          }
          tl::ascend::pv_scale(pool[vector], pool[vector], pool[vector], -1.000000e+00f, factor_rows);
          tl::ascend::pv_brcb_rows(pool[factors], pool[vector], pool[factors], factor_rows, 8);
          uint32_t pivot_column = (first + column_2);
          bool consumed = (((pivot_column % (uint32_t)8) == (uint32_t)7) || ((pivot_column + (uint32_t)1) == n));
          int32_t condval_2;
          if (consumed) {
            condval_2 = 8;
          } else {
            condval_2 = 0;
          }
          uint32_t live_begin = (((pivot_column / (uint32_t)8) * (uint32_t)8) + ((uint32_t)condval_2));
          int32_t condval_3;
          if (consumed) {
            condval_3 = 0;
          } else {
            condval_3 = 8;
          }
          uint32_t live_limit = ((live_begin + k) + ((uint32_t)condval_3));
          for (uint32_t chunk_2 = (uint32_t)0; chunk_2 < ((((live_limit - live_begin) + (uint32_t)64) - (uint32_t)1) / (uint32_t)64); ++chunk_2) {
            uint32_t col_1 = (live_begin + (chunk_2 * (uint32_t)64));
            tl::ascend::pv_mad_strided(pool[col_1], pool[(((packets + (column_2 * packet_stride)) + col_1) - first)], pool[factors], min((live_limit - col_1), (uint32_t)64), owned, 1, 1, 0, (stride / (uint32_t)8), 0, 1);
          }
        }
      }
      if ((((uint32_t)0 < window) && ((first + rows) == window_end)) && (window_end < n)) {
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 3);
        if (valid != 0) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          uint32_t next_columns = min(window, (n - window_end));
          tl::ascend::pv_store_rows(W[((work + ((uint64_t)(core * k))) + ((uint64_t)window_end))], pool[window_end], W[((work + ((uint64_t)(core * k))) + ((uint64_t)window_end))], owned, next_columns, (stride - next_columns), ((cores * k) - next_columns));
        }
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
        window_end = min(n, (window_end + window));
      }
    }
    inverse = 0.000000e+00f;
    peak = 0.000000e+00f;
    if (valid != 0) {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      for (uint32_t j = (uint32_t)0; j < n; ++j) {
        int32_t index_1 = ((((int32_t)(pool.GetValue((norms + j)) * -1.000000e+00f)) * 4) - 4);
        tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (mapping + j), ((uint32_t)index_1));
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      tl::ascend::pv_fill(pool[accumulator], pool[accumulator], pool[accumulator], 0.000000e+00f, wide);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t i_2 = (uint32_t)0; i_2 < owned; ++i_2) {
        tl::ascend::pv_gather_offsets(pool[(output + (i_2 * k))], pool[((i_2 * stride) + k)], pool[mapping], 0, n);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mul(pool[product], pool[(output + (i_2 * k))], pool[(output + (i_2 * k))], n);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_add(pool[accumulator], pool[accumulator], pool[product], n);
        AscendC::PipeBarrier<PIPE_V>();
      }
      tl::ascend::pv_reduce_sum(pool[red], pool[accumulator], pool[scratch], k);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      inverse = pool.GetValue(red);
      for (uint32_t i_3 = (uint32_t)0; i_3 < owned; ++i_3) {
        tl::ascend::pv_scale(pool[(output + (i_3 * k))], pool[(output + (i_3 * k))], pool[(output + (i_3 * k))], (1.000000e+00f / power), k);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t target = (((int32_t)(pool.GetValue(((norms + core) + (i_3 * cores))) * -1.000000e+00f)) - 1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(X[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(((uint32_t)target) * n)))], pool[(output + (i_3 * k))], X[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(((uint32_t)target) * n)))], 1, n, 0, 0);
      }
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    }
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, inverse);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], pool[0], W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], 1, 2, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}

__aicore__ inline void PvDslPipelinePeer( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, int32_t n, int32_t stride, int32_t all_batch, int32_t job, int32_t core, int32_t cores, int32_t window, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int role = 0;
  int k = 0;
  int wide = 0;
  int half = 0;
  int own_width = 0;
  int pitch = 0;
  int capacity = 0;
  int actual_width = 0;
  int prefix = 0;
  int source_column = 0;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  float local_scale = 0.000000e+00f;
  float local_total = 0.000000e+00f;
  float local_power = 0.000000e+00f;
  int valid = 0;
  float power = 0.000000e+00f;
  int size = 0;
  int header_issued = 0;
  int pending_ready = 0;
  int owner = 0;
  int drop = 0;
  int old_prefix = 0;
  int new_width = 0;
  int next_owner = 0;
  int new_prefix = 0;
  int next_pitch = 0;
  float best = 0.000000e+00f;
  int chunk = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    role = (core - cores);
    k = (((n + 7) / 8) * 8);
    wide = (((k + 63) / 64) * 64);
    half = (((n + 31) / 32) * 16);
    own_width = half;
    if (role != 0) {
      own_width = ((((n + 7) - half) / 8) * 8);
    }
    pitch = own_width;
    if (role != 0) {
      pitch = (pitch + 16);
    }
    if (pitch < 16) {
      pitch = 16;
    }
    capacity = (n * pitch);
    int32_t scratch = capacity;
    int32_t vector = (((k * 8) + scratch) + 4096);
    int32_t product = (vector + wide);
    int32_t temporary = (product + wide);
    int32_t permutation = ((temporary + k) + 64);
    int32_t records = (permutation + k);
    actual_width = half;
    prefix = 0;
    if (role != 0) {
      actual_width = (n - half);
      prefix = 16;
    }
    source_column = 0;
    if (role != 0) {
      source_column = half;
    }
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[prefix], A[(((job * n) * n) + source_column)], pool[prefix], n, actual_width, (n - actual_width), prefix);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    local_scale = 0.000000e+00f;
    local_total = 0.000000e+00f;
    local_power = 1.000000e+00f;
    for (int32_t chunk_1 = 0; chunk_1 < ((capacity + 4095) / 4096); ++chunk_1) {
      int32_t count = min((capacity - (chunk_1 * 4096)), 4096);
      tl::ascend::pv_abs(pool[scratch], pool[(chunk_1 * 4096)], pool[scratch], count);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[(temporary + 32)], pool[scratch], pool[product], count);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      local_scale = max(local_scale, pool.GetValue((((int64_t)temporary) + (int64_t)32)));
    }
    if ((0.000000e+00f < local_scale) && (local_scale < 1.701412e+38f)) {
      while (1) {
        if (!((local_power < local_scale))) { break; }
        local_power = (local_power * 2.000000e+00f);
      }
      while (1) {
        if (!((local_scale <= (local_power * 5.000000e-01f)))) { break; }
        local_power = (local_power * 5.000000e-01f);
      }
    }
    for (int32_t chunk_2 = 0; chunk_2 < ((capacity + 16319) / 16320); ++chunk_2) {
      tl::ascend::pv_scale(pool[(chunk_2 * 16320)], pool[(chunk_2 * 16320)], pool[(chunk_2 * 16320)], (1.000000e+00f / local_power), min((capacity - (chunk_2 * 16320)), 16320));
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t chunk_3 = 0; chunk_3 < ((capacity + 4095) / 4096); ++chunk_3) {
      int32_t count_1 = min((capacity - (chunk_3 * 4096)), 4096);
      tl::ascend::pv_mul(pool[scratch], pool[(chunk_3 * 4096)], pool[(chunk_3 * 4096)], count_1);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(pool[(temporary + 32)], pool[scratch], pool[product], count_1);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      local_total = (local_total + pool.GetValue((((int64_t)temporary) + (int64_t)32)));
    }
    pool.SetValue(((int64_t)scratch), local_scale);
    pool.SetValue((((int64_t)scratch) + (int64_t)1), local_total);
    pool.SetValue((((int64_t)scratch) + (int64_t)2), local_power);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((job * 8192) + (role * 8)) + (((all_batch * n) * stride) * 2)) + 3072)], pool[scratch], W[((((job * 8192) + (role * 8)) + (((all_batch * n) * stride) * 2)) + 3072)], 1, 8, 0, 0);
    tl::ascend::pv_cross_set1(pool[0], pool[0], pool[0], 4);
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    tl::ascend::pv_load_rows(pool[scratch], W[(((job * 8192) + (((all_batch * n) * stride) * 2)) + 3072)], pool[scratch], 1, 16, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    float magnitude = max(pool.GetValue(((int64_t)scratch)), pool.GetValue((((int64_t)scratch) + (int64_t)8)));
    float sum0 = pool.GetValue((((int64_t)scratch) + (int64_t)1));
    float sum1 = pool.GetValue((((int64_t)scratch) + (int64_t)9));
    float power0 = pool.GetValue((((int64_t)scratch) + (int64_t)2));
    float power1 = pool.GetValue((((int64_t)scratch) + (int64_t)10));
    tl::ascend::pv_cross_set1(pool[0], pool[0], pool[0], 5);
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 5);
    valid = 1;
    if ((magnitude < 1.000000e-01f) || (1.701412e+38f <= magnitude)) {
      valid = 0;
    }
    power = 1.000000e+00f;
    if (valid != 0) {
      while (1) {
        if (!((power < magnitude))) { break; }
        power = (power * 2.000000e+00f);
      }
      while (1) {
        if (!((magnitude <= (power * 5.000000e-01f)))) { break; }
        power = (power * 5.000000e-01f);
      }
    }
    float ratio0 = (power0 / power);
    float ratio1 = (power1 / power);
    float total = ((((sum0 * ratio0) * ratio0) + ((sum1 * ratio1) * ratio1)) * 1.000010e+00f);
    for (int32_t chunk_4 = 0; chunk_4 < ((capacity + 16319) / 16320); ++chunk_4) {
      tl::ascend::pv_scale(pool[(chunk_4 * 16320)], pool[(chunk_4 * 16320)], pool[(chunk_4 * 16320)], (local_power / power), min((capacity - (chunk_4 * 16320)), 16320));
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t row = 0; row < k; ++row) {
      pool.SetValue((((int64_t)permutation) + ((int64_t)row)), ((float)row));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    size = n;
    header_issued = 0;
    pending_ready = 0;
    for (int32_t panel = 0; panel < ((n + 15) / 16); ++panel) {
      if (valid != 0) {
        if (header_issued != 0) {
          if (pending_ready != 0) {
            tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 7);
          }
          tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 9);
          header_issued = 0;
        }
        owner = 0;
        if (half <= (panel * 16)) {
          owner = 1;
        }
        if (role == owner) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(W[(((job * 8192) + (((all_batch * n) * stride) * 2)) + 3072)], pool[0], W[(((job * 8192) + (((all_batch * n) * stride) * 2)) + 3072)], size, 16, (pitch - 16), 0);
          tl::ascend::pv_cross_set1(pool[0], pool[0], pool[0], 7);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          pending_ready = 1;
        } else {
          tl::ascend::pv_cross_set1(pool[0], pool[0], pool[0], 7);
          tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 7);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
          tl::ascend::pv_load_rows(pool[0], W[(((job * 8192) + (((all_batch * n) * stride) * 2)) + 3072)], pool[0], size, 16, 0, (pitch - 16));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          pending_ready = 0;
        }
        tl::ascend::pv_cross_set1(pool[0], pool[0], pool[0], 9);
        header_issued = 1;
        tl::ascend::pv_iota(pool[(temporary + 64)], pool[(temporary + 64)], pool[(temporary + 64)], size);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_scale_indices(pool[(temporary + 64)], pool[(temporary + 64)], pool[(temporary + 64)], (pitch * 4), size);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t block = min(size, 16);
        for (int32_t pivot = 0; pivot < block; ++pivot) {
          int32_t remaining = (size - pivot);
          tl::ascend::pv_gather_offsets(pool[vector], pool[0], pool[(temporary + 64)], ((pivot * (pitch + 1)) * 4), remaining);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_abs(pool[product], pool[vector], pool[product], remaining);
          AscendC::PipeBarrier<PIPE_V>();
          if (63 < remaining) {
            tl::ascend::pv_whole_max(pool[(temporary + 32)], pool[product], pool[(temporary + 32)], 64, (remaining / 64));
          }
          if (0 < (remaining % 64)) {
            tl::ascend::pv_whole_max(pool[((((remaining / 64) * 2) + temporary) + 32)], pool[(((remaining / 64) * 64) + product)], pool[((((remaining / 64) * 2) + temporary) + 32)], (remaining % 64), 1);
          }
          AscendC::PipeBarrier<PIPE_V>();
          int32_t condval;
          if ((0 < (remaining % 64))) {
            condval = 1;
          } else {
            condval = 0;
          }
          tl::ascend::pv_even_max(pool[temporary], pool[(temporary + 32)], pool[temporary], ((remaining / 64) + condval));
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best, chunk);
            float v_ = pool.GetValue(((((int64_t)temporary) + ((int64_t)chunk)) + (int64_t)33));
          int32_t selected = ((((chunk / 2) * 64) + pivot) + ((int32_t)(*(uint32_t *)(&(v_)))));
          if (best <= 1.000000e-06f) {
            valid = 0;
            break;
          }
          float divisor = pool.GetValue(((((int64_t)vector) + ((int64_t)selected)) - ((int64_t)pivot)));
          if (selected != pivot) {
            float old = pool.GetValue((((int64_t)permutation) + ((int64_t)pivot)));
            pool.SetValue((((int64_t)permutation) + ((int64_t)pivot)), pool.GetValue((((int64_t)permutation) + ((int64_t)selected))));
            pool.SetValue((((int64_t)permutation) + ((int64_t)selected)), old);
          }
          pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)), pool.GetValue((((int64_t)permutation) + ((int64_t)pivot))));
          pool.SetValue(((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)) + (int64_t)16), (1.000000e+00f / divisor));
          if (1 < remaining) {
            pool.SetValue(((((int64_t)vector) + ((int64_t)selected)) - ((int64_t)pivot)), 0.000000e+00f);
            tl::ascend::pv_copy(pool[scratch], pool[(selected * pitch)], pool[scratch], pitch);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_scale(pool[scratch], pool[scratch], pool[scratch], (-1.000000e+00f / divisor), pitch);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            tl::ascend::pv_brcb_rows(pool[(scratch + 4096)], pool[vector], pool[(scratch + 4096)], (((remaining + 7) / 8) * 8), 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t chunk_5 = 0; chunk_5 < (((pitch + 63) - ((pivot / 8) * 8)) / 64); ++chunk_5) {
              for (int32_t row_chunk = 0; row_chunk < ((remaining + 254) / 255); ++row_chunk) {
                tl::ascend::pv_mad_strided(pool[(((chunk_5 * 64) + ((pivot / 8) * 8)) + (((row_chunk * 255) + pivot) * pitch))], pool[(((chunk_5 * 64) + ((pivot / 8) * 8)) + scratch)], pool[(((row_chunk * 2040) + scratch) + 4096)], min(((pitch - ((pivot / 8) * 8)) - (chunk_5 * 64)), 64), min((remaining - (row_chunk * 255)), 255), 1, 1, 0, (pitch / 8), 0, 1);
              }
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (selected != pivot) {
              tl::ascend::pv_copy(pool[(selected * pitch)], pool[(pivot * pitch)], pool[(selected * pitch)], pitch);
              AscendC::PipeBarrier<PIPE_V>();
            }
          }
        }
        int32_t remaining_1 = (size - block);
        drop = 0;
        old_prefix = 16;
        if (role == owner) {
          drop = block;
          old_prefix = 0;
        }
        new_width = (own_width - drop);
        next_owner = 0;
        if (half <= ((panel * 16) + block)) {
          next_owner = 1;
        }
        new_prefix = 16;
        if (role == next_owner) {
          new_prefix = 0;
        }
        next_pitch = max(16, (new_prefix + new_width));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        if ((0 < remaining_1) && (0 < new_width)) {
          for (int32_t row_chunk_1 = 0; row_chunk_1 < ((remaining_1 + 15) / 16); ++row_chunk_1) {
            tl::ascend::pv_copy_ub_rows(pool[(((row_chunk_1 * next_pitch) * 16) + new_prefix)], pool[(((((row_chunk_1 * 16) + block) * pitch) + old_prefix) + drop)], pool[(((row_chunk_1 * next_pitch) * 16) + new_prefix)], min((remaining_1 - (row_chunk_1 * 16)), 16), new_width, (pitch - new_width), (next_pitch - new_width));
            AscendC::PipeBarrier<PIPE_V>();
          }
        }
        if (0 < remaining_1) {
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          tl::ascend::pv_copy(pool[scratch], pool[(permutation + block)], pool[scratch], (((remaining_1 + 7) / 8) * 8));
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_copy(pool[permutation], pool[scratch], pool[permutation], (((remaining_1 + 7) / 8) * 8));
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        size = remaining_1;
        own_width = new_width;
        pitch = next_pitch;
      }
      if (0 < panel) {
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
      }
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)32), power);
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)33), total);
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)34), ((float)valid));
      if (role == 1) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        tl::ascend::pv_store_rows(W[(((job * 8192) + (panel * 64)) + (((all_batch * n) * stride) * 2))], pool[((panel * 64) + records)], W[(((job * 8192) + (panel * 64)) + (((all_batch * n) * stride) * 2))], 1, 64, 0, 0);
      }
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
    }
    if (header_issued != 0) {
      if (pending_ready != 0) {
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 7);
      }
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 9);
    }
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], pool[0], W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], 1, 2, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}

struct PvPipelineBuffers { AscendC::GlobalTensor<float> inputTensor,outputTensor,workspaceTensor; };
}
#define pv_pipeline_bind(d,a,b) PvPipelineBuffers pvPipelineBuffers{d,a,b}
#define pv_pipeline_rows(d,a,b,...) PvDslPipelineRows((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
#define pv_pipeline_peer(d,a,b,...) PvDslPipelinePeer((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)



using uint=unsigned int;using uchar=unsigned char;using ushort=unsigned short;
namespace tl::ascend {
__aicore__ static __attribute__((noinline)) void PvDslAccurate( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t job, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 176544);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto accepted = ascend_ub.GetWithOffset<int>(8, 0);
  int pad = 0;
  int width = 0;
  int pitch = 0;
  auto pool = ascend_ub.GetWithOffset<float>(44128, 32);
  float magnitude = 0.000000e+00f;
  int valid = 0;
  float power = 0.000000e+00f;
  float total = 0.000000e+00f;
  float inverse_total = 0.000000e+00f;
  int pivot = 0;
  float best = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    accepted.SetValue(0, 0);
    pad = (((n + 7) / 8) * 8);
    width = (pad * 2);
    pitch = (width + 8);
    int32_t matrix = ((width * 20) + 32);
    int32_t low = (matrix + (n * pitch));
    int32_t bx = (low + (n * pitch));
    int32_t by = (bx + pitch);
    int32_t bu = (by + pitch);
    int32_t bv = (bu + pitch);
    int32_t red = (width * 20);
    int32_t prod = (width * 4);
    int32_t tmp = (width * 5);
    tl::ascend::pv_arith_init(pool[0], pool[0], pool[0], width);
    magnitude = 0.000000e+00f;
    valid = 1;
    power = 1.000000e+00f;
    total = 0.000000e+00f;
    for (int32_t i = 0; i < n; ++i) {
      tl::ascend::pv_fill(pool[(matrix + (i * pitch))], pool[(matrix + (i * pitch))], pool[(matrix + (i * pitch))], 0.000000e+00f, pitch);
      tl::ascend::pv_fill(pool[(low + (i * pitch))], pool[(low + (i * pitch))], pool[(low + (i * pitch))], 0.000000e+00f, pitch);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
      tl::ascend::pv_load_rows(pool[(matrix + (i * pitch))], A[(n * ((job * m) + i))], pool[(matrix + (i * pitch))], 1, n, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      tl::ascend::pv_abs(pool[prod], pool[(matrix + (i * pitch))], pool[prod], pad);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[red], pool[prod], pool[tmp], pad);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      magnitude = max(magnitude, pool.GetValue(((int64_t)red)));
      pool.SetValue((((((int64_t)matrix) + (((int64_t)i) * ((int64_t)pitch))) + ((int64_t)pad)) + ((int64_t)i)), 1.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    }
    if (magnitude <= 0.000000e+00f) {
      valid = 0;
    }
    if (valid != 0) {
      while (1) {
        if (!(((power < magnitude) && (power < 1.701412e+38f)))) { break; }
        power = (power * 2.000000e+00f);
      }
      while (1) {
        if (!((magnitude <= (power * 5.000000e-01f)))) { break; }
        power = (power * 5.000000e-01f);
      }
      for (int32_t i_1 = 0; i_1 < n; ++i_1) {
        tl::ascend::pv_scale(pool[(matrix + (i_1 * pitch))], pool[(matrix + (i_1 * pitch))], pool[(matrix + (i_1 * pitch))], (1.000000e+00f / power), pad);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mul(pool[prod], pool[(matrix + (i_1 * pitch))], pool[(matrix + (i_1 * pitch))], pad);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_reduce_sum(pool[red], pool[prod], pool[tmp], pad);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        total = (total + pool.GetValue(((int64_t)red)));
      }
      for (int32_t p = 0; p < n; ++p) {
        if (valid != 0) {
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          pivot = p;
          best = 0.000000e+00f;
          for (int32_t i_2 = p; i_2 < n; ++i_2) {
            float entry = pool.GetValue(((((int64_t)matrix) + (((int64_t)i_2) * ((int64_t)pitch))) + ((int64_t)p)));
            float absolute = max(entry, (entry * -1.000000e+00f));
            if (best < absolute) {
              best = absolute;
              pivot = i_2;
            }
          }
          if (best <= 0.000000e+00f) {
            valid = 0;
          } else {
            if (pivot != p) {
              tl::ascend::pv_copy(pool[bx], pool[(matrix + (p * pitch))], pool[bx], pitch);
              tl::ascend::pv_copy(pool[by], pool[(low + (p * pitch))], pool[by], pitch);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_copy(pool[(matrix + (p * pitch))], pool[(matrix + (pivot * pitch))], pool[(matrix + (p * pitch))], pitch);
              tl::ascend::pv_copy(pool[(low + (p * pitch))], pool[(low + (pivot * pitch))], pool[(low + (p * pitch))], pitch);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_copy(pool[(matrix + (pivot * pitch))], pool[bx], pool[(matrix + (pivot * pitch))], pitch);
              tl::ascend::pv_copy(pool[(low + (pivot * pitch))], pool[by], pool[(low + (pivot * pitch))], pitch);
              AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            float dh = pool.GetValue(((((int64_t)matrix) + (((int64_t)p) * ((int64_t)pitch))) + ((int64_t)p)));
            float dl = pool.GetValue(((((int64_t)low) + (((int64_t)p) * ((int64_t)pitch))) + ((int64_t)p)));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            tl::ascend::pv_pair_normalize(pool[0], pool[0], pool[0], (matrix + (p * pitch)), (low + (p * pitch)), bu, bv, dh, dl, width);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            for (int32_t i_3 = 0; i_3 < n; ++i_3) {
              if (i_3 != p) {
                float fh = pool.GetValue(((((int64_t)matrix) + (((int64_t)i_3) * ((int64_t)pitch))) + ((int64_t)p)));
                float fl = pool.GetValue(((((int64_t)low) + (((int64_t)i_3) * ((int64_t)pitch))) + ((int64_t)p)));
                tl::ascend::pv_pair_eliminate(pool[0], pool[0], pool[0], (matrix + (i_3 * pitch)), (low + (i_3 * pitch)), (matrix + (p * pitch)), (low + (p * pitch)), fh, fl, width);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              }
            }
          }
        }
      }
      if (valid != 0) {
        inverse_total = 0.000000e+00f;
        for (int32_t i_4 = 0; i_4 < n; ++i_4) {
          tl::ascend::pv_mul(pool[prod], pool[((matrix + (i_4 * pitch)) + pad)], pool[((matrix + (i_4 * pitch)) + pad)], pad);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_reduce_sum(pool[red], pool[prod], pool[tmp], pad);
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          inverse_total = (inverse_total + pool.GetValue(((int64_t)red)));
        }
        pool.SetValue(((int64_t)red), total);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        tl::ascend::pv_sqrt(pool[red], pool[red], pool[red], 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        float cutoff = max((pool.GetValue(((int64_t)red)) * rtol), (atol / power));
        if ((inverse_total <= 0.000000e+00f) || (8.100000e-01f <= ((inverse_total * cutoff) * cutoff))) {
          valid = 0;
        } else {
          for (int32_t i_5 = 0; i_5 < n; ++i_5) {
            tl::ascend::pv_add(pool[bx], pool[((matrix + (i_5 * pitch)) + pad)], pool[((low + (i_5 * pitch)) + pad)], pad);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_scale(pool[bx], pool[bx], pool[bx], (1.000000e+00f / power), pad);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(X[(n * ((job * m) + i_5))], pool[bx], X[(n * ((job * m) + i_5))], 1, n, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          }
        }
      }
    }
    accepted.SetValue(0, valid);
    S.SetValue((((int64_t)job) * (int64_t)32), ((float)accepted.GetValue(0)));
  }
  pipe.Destroy();
}

__aicore__ static __attribute__((noinline)) void PvDslRectangular( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t job, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 105536);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto accepted = ascend_ub.GetWithOffset<int>(8, 0);
  int head = 0;
  int basis = 0;
  auto pool = ascend_ub.GetWithOffset<float>(26144, 32);
  float power = 0.000000e+00f;
  float peak = 0.000000e+00f;
  int sweep = 0;
  int changed = 0;
  auto _pair_temp_0 = ascend_ub.GetWithOffset<float>(8, 104704);
  auto _pair_temp_2 = ascend_ub.GetWithOffset<float>(8, 104736);
  auto _pair_temp_1 = ascend_ub.GetWithOffset<float>(8, 104768);
  float maximum = 0.000000e+00f;
  float minimum = 0.000000e+00f;
  float total = 0.000000e+00f;
  int rank = 0;
  auto w_norm = ascend_ub.GetWithOffset<float>(8, 105056);
  auto w_ov = ascend_ub.GetWithOffset<float>(8, 105088);
  auto w_ou = ascend_ub.GetWithOffset<float>(8, 105120);
  auto w_r1 = ascend_ub.GetWithOffset<float>(8, 105152);
  auto w_r2 = ascend_ub.GetWithOffset<float>(8, 105184);
  float discarded_gram = 0.000000e+00f;
  float v_gram_total = 0.000000e+00f;
  float u_gram_total = 0.000000e+00f;
  float norm = 0.000000e+00f;
  int head_depth = 0;
  int basis_depth = 0;
  int size = 0;
  int certified = 0;
  float error = 0.000000e+00f;
  int valid = 0;
  auto w_a = ascend_ub.GetWithOffset<float>(8, 104608);
  auto w_d = ascend_ub.GetWithOffset<float>(8, 104640);
  auto w_h = ascend_ub.GetWithOffset<float>(8, 104672);
  auto w_tau = ascend_ub.GetWithOffset<float>(8, 104800);
  auto w_z = ascend_ub.GetWithOffset<float>(8, 104832);
  auto w_root = ascend_ub.GetWithOffset<float>(8, 104864);
  auto w_absolute = ascend_ub.GetWithOffset<float>(8, 104896);
  auto _pair_temp_3 = ascend_ub.GetWithOffset<float>(8, 104928);
  auto _pair_temp_4 = ascend_ub.GetWithOffset<float>(8, 104960);
  auto w_c = ascend_ub.GetWithOffset<float>(8, 104992);
  auto w_sn = ascend_ub.GetWithOffset<float>(8, 105024);
  auto w_ni = ascend_ub.GetWithOffset<float>(8, 105216);
  float row_bound = 0.000000e+00f;
  auto w_vv = ascend_ub.GetWithOffset<float>(8, 105248);
  auto w_v = ascend_ub.GetWithOffset<float>(8, 105280);
  auto w_cc = ascend_ub.GetWithOffset<float>(8, 105312);
  auto w_nj = ascend_ub.GetWithOffset<float>(8, 105344);
  auto w_uu = ascend_ub.GetWithOffset<float>(8, 105376);
  auto w_u = ascend_ub.GetWithOffset<float>(8, 105408);
  auto w_vij = ascend_ub.GetWithOffset<float>(8, 105440);
  auto w_rhs = ascend_ub.GetWithOffset<float>(8, 105472);
  auto w_factor = ascend_ub.GetWithOffset<float>(8, 105504);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    accepted.SetValue(0, 0);
    head = 8;
    basis = 8;
    while (1) {
      if (!(((head < m) || (head < n)))) { break; }
      head = (head * 2);
    }
    while (1) {
      if (!(((basis < m) && (basis < n)))) { break; }
      basis = (basis * 2);
    }
    int32_t width = (head + basis);
    int32_t sigma_low = ((((width * 20) + (max(m, n) * 8)) + basis) + 32);
    int32_t retained = ((((width * 20) + (max(m, n) * 8)) + (basis * 2)) + 32);
    tl::ascend::pv_arith_init(pool[0], pool[0], pool[0], width);
    accepted.SetValue(0, 0);
    power = 1.000000e+00f;
    peak = 0.000000e+00f;
    for (int32_t i = 0; i < min(m, n); ++i) {
      tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, width);
      tl::ascend::pv_fill(pool[width], pool[width], pool[width], 0.000000e+00f, width);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
      if (n <= m) {
        tl::ascend::pv_load_rows(pool[((width * 20) + 32)], A[(((job * m) * n) + i)], pool[((width * 20) + 32)], max(m, n), 1, (n - 1), 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        for (int32_t j = 0; j < max(m, n); ++j) {
          pool.SetValue(j, pool.GetValue((((((int64_t)width) * (int64_t)20) + (((int64_t)j) * (int64_t)8)) + (int64_t)32)));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      } else {
        tl::ascend::pv_load_rows(pool[0], A[(n * ((job * m) + i))], pool[0], 1, n, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      }
      tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power), head);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_abs(pool[(width * 4)], pool[0], pool[(width * 4)], head);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[(width * 20)], pool[(width * 4)], pool[(width * 8)], head);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      peak = max(peak, pool.GetValue((((int64_t)width) * (int64_t)20)));
    }
    if (peak == 0.000000e+00f) {
      tl::ascend::pv_fill(pool[(width * 6)], pool[(width * 6)], pool[(width * 6)], 0.000000e+00f, width);
      AscendC::PipeBarrier<PIPE_V>();
      for (int32_t output_row = 0; output_row < n; ++output_row) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        tl::ascend::pv_store_rows(X[(((job * m) * n) + (output_row * m))], pool[(width * 6)], X[(((job * m) * n) + (output_row * m))], 1, m, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
      }
      accepted.SetValue(0, 1);
    } else {
      while (1) {
        if (!(((power < peak) && (power < 1.701412e+38f)))) { break; }
        power = (power * 2.000000e+00f);
      }
      while (1) {
        if (!((peak <= (power * 5.000000e-01f)))) { break; }
        power = (power * 5.000000e-01f);
      }
      for (int32_t attempt = 0; attempt < 2; ++attempt) {
        if (accepted.GetValue(0) == 0) {
          for (int32_t i_1 = 0; i_1 < min(m, n); ++i_1) {
            tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, width);
            tl::ascend::pv_fill(pool[width], pool[width], pool[width], 0.000000e+00f, width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            if (n <= m) {
              tl::ascend::pv_load_rows(pool[((width * 20) + 32)], A[(((job * m) * n) + i_1)], pool[((width * 20) + 32)], max(m, n), 1, (n - 1), 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
              for (int32_t j_1 = 0; j_1 < max(m, n); ++j_1) {
                pool.SetValue(j_1, pool.GetValue((((((int64_t)width) * (int64_t)20) + (((int64_t)j_1) * (int64_t)8)) + (int64_t)32)));
              }
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            } else {
              tl::ascend::pv_load_rows(pool[0], A[(n * ((job * m) + i_1))], pool[0], 1, n, 0, 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            }
            tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power), head);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            pool.SetValue((((int64_t)head) + ((int64_t)i_1)), 1.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(W[((((job * min(m, n)) * width) * 2) + (i_1 * width))], pool[0], W[((((job * min(m, n)) * width) * 2) + (i_1 * width))], 1, width, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (i_1 * width))], pool[width], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (i_1 * width))], 1, width, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          }
          sweep = 0;
          changed = 1;
          float condval;
          if ((attempt == 0)) {
            condval = 1.000000e-06f;
          } else {
            condval = 2.000000e-13f;
          }
          float orth = condval;
          while (1) {
            int32_t condval_1;
            if ((attempt == 0)) {
              condval_1 = 40;
            } else {
              condval_1 = 120;
            }
            if (!(((sweep < condval_1) && (changed != 0)))) { break; }
            changed = 0;
            for (int32_t p = 0; p < min(m, n); ++p) {
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              tl::ascend::pv_load_rows(pool[0], W[((((job * min(m, n)) * width) * 2) + (p * width))], pool[0], 1, width, 0, 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              tl::ascend::pv_load_rows(pool[width], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (p * width))], pool[width], 1, width, 0, 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
              for (int32_t q = (p + 1); q < min(m, n); ++q) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                tl::ascend::pv_load_rows(pool[(width * 2)], W[((((job * min(m, n)) * width) * 2) + (q * width))], pool[(width * 2)], 1, width, 0, 0);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                tl::ascend::pv_load_rows(pool[(width * 3)], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (q * width))], pool[(width * 3)], 1, width, 0, 0);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                tl::ascend::pv_pair_dot(w_a[0], pool[0], w_a[0], 0, width, 0, width, head);
                tl::ascend::pv_pair_dot(w_d[0], pool[0], w_d[0], (width * 2), (width * 3), (width * 2), (width * 3), head);
                tl::ascend::pv_pair_dot(w_h[0], pool[0], w_h[0], 0, width, (width * 2), (width * 3), head);
                tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_a.GetValue(0) * w_d.GetValue(0)));
                float norm_product = _pair_temp_0.GetValue(0);
                if (((0.000000e+00f < w_a.GetValue(0)) && (0.000000e+00f < w_d.GetValue(0))) && ((orth * norm_product) < max((w_h.GetValue(0) + w_h.GetValue(1)), ((w_h.GetValue(0) + w_h.GetValue(1)) * -1.000000e+00f)))) {
                  tl::ascend::pv_pair_minus(_pair_temp_0[0], w_d[0], w_a[0]);
                  tl::ascend::pv_pair_pack(_pair_temp_2[0], _pair_temp_2[0], _pair_temp_2[0], 2.000000e+00f, 0.000000e+00f);
                  tl::ascend::pv_pair_times(_pair_temp_1[0], _pair_temp_2[0], w_h[0]);
                  tl::ascend::pv_pair_divide(w_tau[0], _pair_temp_0[0], _pair_temp_1[0]);
                  if (1.000000e+08f < max(w_tau.GetValue(0), (w_tau.GetValue(0) * -1.000000e+00f))) {
                    tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], 5.000000e-01f, 0.000000e+00f);
                    tl::ascend::pv_pair_divide(w_z[0], _pair_temp_0[0], w_tau[0]);
                  } else {
                    tl::ascend::pv_pair_pack(_pair_temp_1[0], _pair_temp_1[0], _pair_temp_1[0], 1.000000e+00f, 0.000000e+00f);
                    tl::ascend::pv_pair_times(_pair_temp_2[0], w_tau[0], w_tau[0]);
                    tl::ascend::pv_pair_plus(_pair_temp_0[0], _pair_temp_1[0], _pair_temp_2[0]);
                    tl::ascend::pv_pair_root(w_root[0], _pair_temp_0[0], pool[0]);
                    tl::ascend::pv_pair_pack(w_absolute[0], w_absolute[0], w_absolute[0], w_tau.GetValue(0), w_tau.GetValue(1));
                    if (w_tau.GetValue(0) < 0.000000e+00f) {
                      tl::ascend::pv_pair_pack(w_absolute[0], w_absolute[0], w_absolute[0], (w_tau.GetValue(0) * -1.000000e+00f), (w_tau.GetValue(1) * -1.000000e+00f));
                    }
                    float condval_2;
                    if ((w_tau.GetValue(0) < 0.000000e+00f)) {
                      condval_2 = -1.000000e+00f;
                    } else {
                      condval_2 = 1.000000e+00f;
                    }
                    tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], condval_2, 0.000000e+00f);
                    tl::ascend::pv_pair_plus(_pair_temp_1[0], w_absolute[0], w_root[0]);
                    tl::ascend::pv_pair_divide(w_z[0], _pair_temp_0[0], _pair_temp_1[0]);
                  }
                  tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], 1.000000e+00f, 0.000000e+00f);
                  tl::ascend::pv_pair_pack(_pair_temp_3[0], _pair_temp_3[0], _pair_temp_3[0], 1.000000e+00f, 0.000000e+00f);
                  tl::ascend::pv_pair_times(_pair_temp_4[0], w_z[0], w_z[0]);
                  tl::ascend::pv_pair_plus(_pair_temp_2[0], _pair_temp_3[0], _pair_temp_4[0]);
                  tl::ascend::pv_pair_root(_pair_temp_1[0], _pair_temp_2[0], pool[0]);
                  tl::ascend::pv_pair_divide(w_c[0], _pair_temp_0[0], _pair_temp_1[0]);
                  tl::ascend::pv_pair_times(w_sn[0], w_c[0], w_z[0]);
                  if (0 < attempt) {
                    tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 4), (width * 5), 0, width, w_c.GetValue(0), w_c.GetValue(1), width);
                    tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 6), (width * 7), (width * 2), (width * 3), (w_sn.GetValue(0) * -1.000000e+00f), (w_sn.GetValue(1) * -1.000000e+00f), width);
                    tl::ascend::pv_pair_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 4), (width * 5), (width * 6), (width * 7), width);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                    tl::ascend::pv_store_rows(W[((((job * min(m, n)) * width) * 2) + (p * width))], pool[(width * 4)], W[((((job * min(m, n)) * width) * 2) + (p * width))], 1, width, 0, 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                    tl::ascend::pv_store_rows(W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (p * width))], pool[(width * 5)], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (p * width))], 1, width, 0, 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 4), (width * 5), 0, width, w_sn.GetValue(0), w_sn.GetValue(1), width);
                    tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 6), (width * 7), (width * 2), (width * 3), w_c.GetValue(0), w_c.GetValue(1), width);
                    tl::ascend::pv_pair_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 4), (width * 5), (width * 6), (width * 7), width);
                  } else {
                    tl::ascend::pv_scale(pool[(width * 4)], pool[0], pool[(width * 4)], w_c.GetValue(0), width);
                    tl::ascend::pv_scale(pool[(width * 6)], pool[(width * 2)], pool[(width * 6)], (w_sn.GetValue(0) * -1.000000e+00f), width);
                    AscendC::PipeBarrier<PIPE_V>();
                    tl::ascend::pv_add(pool[(width * 4)], pool[(width * 4)], pool[(width * 6)], width);
                    tl::ascend::pv_fill(pool[(width * 5)], pool[(width * 5)], pool[(width * 5)], 0.000000e+00f, width);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                    tl::ascend::pv_store_rows(W[((((job * min(m, n)) * width) * 2) + (p * width))], pool[(width * 4)], W[((((job * min(m, n)) * width) * 2) + (p * width))], 1, width, 0, 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                    tl::ascend::pv_store_rows(W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (p * width))], pool[(width * 5)], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (p * width))], 1, width, 0, 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    tl::ascend::pv_scale(pool[(width * 4)], pool[0], pool[(width * 4)], w_sn.GetValue(0), width);
                    tl::ascend::pv_scale(pool[(width * 6)], pool[(width * 2)], pool[(width * 6)], w_c.GetValue(0), width);
                    AscendC::PipeBarrier<PIPE_V>();
                    tl::ascend::pv_add(pool[(width * 4)], pool[(width * 4)], pool[(width * 6)], width);
                    AscendC::PipeBarrier<PIPE_V>();
                  }
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                  tl::ascend::pv_store_rows(W[((((job * min(m, n)) * width) * 2) + (q * width))], pool[(width * 4)], W[((((job * min(m, n)) * width) * 2) + (q * width))], 1, width, 0, 0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                  tl::ascend::pv_store_rows(W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (q * width))], pool[(width * 5)], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (q * width))], 1, width, 0, 0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                  tl::ascend::pv_load_rows(pool[0], W[((((job * min(m, n)) * width) * 2) + (p * width))], pool[0], 1, width, 0, 0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                  tl::ascend::pv_load_rows(pool[width], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (p * width))], pool[width], 1, width, 0, 0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                  changed = (changed + 1);
                }
              }
            }
            sweep = (sweep + 1);
          }
          maximum = 0.000000e+00f;
          minimum = 1.701412e+38f;
          total = 0.000000e+00f;
          rank = 0;
          for (int32_t i_2 = 0; i_2 < min(m, n); ++i_2) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[0], W[((((job * min(m, n)) * width) * 2) + (i_2 * width))], pool[0], 1, width, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[width], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (i_2 * width))], pool[width], 1, width, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            tl::ascend::pv_pair_cert_dot(_pair_temp_0[0], pool[0], _pair_temp_0[0], 0, width, 0, width, head);
            tl::ascend::pv_pair_root(w_norm[0], _pair_temp_0[0], pool[0]);
            float magnitude = (w_norm.GetValue(0) + w_norm.GetValue(1));
            pool.SetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)i_2)) + (int64_t)32), w_norm.GetValue(0));
            pool.SetValue((((int64_t)sigma_low) + ((int64_t)i_2)), w_norm.GetValue(1));
            maximum = max(maximum, magnitude);
            total = (total + (magnitude * magnitude));
          }
          float cutoff = max((atol / power), (rtol * maximum));
          for (int32_t i_3 = 0; i_3 < min(m, n); ++i_3) {
            float magnitude_1 = (pool.GetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)i_3)) + (int64_t)32)) + pool.GetValue((((int64_t)sigma_low) + ((int64_t)i_3))));
            if (cutoff < magnitude_1) {
              pool.SetValue((((int64_t)retained) + ((int64_t)i_3)), 1.000000e+00f);
            } else {
              pool.SetValue((((int64_t)retained) + ((int64_t)i_3)), 0.000000e+00f);
            }
            if (cutoff < magnitude_1) {
              rank = (rank + 1);
              minimum = min(minimum, magnitude_1);
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          tl::ascend::pv_pair_pack(w_ov[0], w_ov[0], w_ov[0], 0.000000e+00f, 0.000000e+00f);
          tl::ascend::pv_pair_pack(w_ou[0], w_ou[0], w_ou[0], 0.000000e+00f, 0.000000e+00f);
          tl::ascend::pv_pair_pack(w_r1[0], w_r1[0], w_r1[0], 0.000000e+00f, 0.000000e+00f);
          tl::ascend::pv_pair_pack(w_r2[0], w_r2[0], w_r2[0], 0.000000e+00f, 0.000000e+00f);
          discarded_gram = 0.000000e+00f;
          v_gram_total = 0.000000e+00f;
          u_gram_total = 0.000000e+00f;
          for (int32_t i_4 = 0; i_4 < min(m, n); ++i_4) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[0], W[((((job * min(m, n)) * width) * 2) + (i_4 * width))], pool[0], 1, width, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[width], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (i_4 * width))], pool[width], 1, width, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            tl::ascend::pv_pair_pack(w_ni[0], w_ni[0], w_ni[0], pool.GetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)i_4)) + (int64_t)32)), pool.GetValue((((int64_t)sigma_low) + ((int64_t)i_4))));
            row_bound = 0.000000e+00f;
            for (int32_t j_2 = 0; j_2 < min(m, n); ++j_2) {
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              tl::ascend::pv_load_rows(pool[(width * 2)], W[((((job * min(m, n)) * width) * 2) + (j_2 * width))], pool[(width * 2)], 1, width, 0, 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              tl::ascend::pv_load_rows(pool[(width * 3)], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (j_2 * width))], pool[(width * 3)], 1, width, 0, 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
              tl::ascend::pv_pair_cert_dot(w_vv[0], pool[0], w_vv[0], head, (width + head), ((width * 2) + head), ((width * 3) + head), basis);
              if (i_4 == j_2) {
                v_gram_total = (v_gram_total + (w_vv.GetValue(0) + w_vv.GetValue(1)));
              }
              float condval_3;
              if ((i_4 == j_2)) {
                condval_3 = 1.000000e+00f;
              } else {
                condval_3 = 0.000000e+00f;
              }
              tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], condval_3, 0.000000e+00f);
              tl::ascend::pv_pair_minus(w_v[0], w_vv[0], _pair_temp_0[0]);
              tl::ascend::pv_pair_times(_pair_temp_0[0], w_v[0], w_v[0]);
              tl::ascend::pv_pair_plus(w_ov[0], w_ov[0], _pair_temp_0[0]);
              tl::ascend::pv_pair_cert_dot(w_cc[0], pool[0], w_cc[0], 0, width, (width * 2), (width * 3), head);
              if ((pool.GetValue((((int64_t)retained) + ((int64_t)i_4))) != 0.000000e+00f) && (pool.GetValue((((int64_t)retained) + ((int64_t)j_2))) != 0.000000e+00f)) {
                tl::ascend::pv_pair_pack(w_nj[0], w_nj[0], w_nj[0], pool.GetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)j_2)) + (int64_t)32)), pool.GetValue((((int64_t)sigma_low) + ((int64_t)j_2))));
                tl::ascend::pv_pair_times(_pair_temp_0[0], w_ni[0], w_nj[0]);
                tl::ascend::pv_pair_divide(w_uu[0], w_cc[0], _pair_temp_0[0]);
                if (i_4 == j_2) {
                  u_gram_total = (u_gram_total + (w_uu.GetValue(0) + w_uu.GetValue(1)));
                }
                float condval_4;
                if ((i_4 == j_2)) {
                  condval_4 = 1.000000e+00f;
                } else {
                  condval_4 = 0.000000e+00f;
                }
                tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], condval_4, 0.000000e+00f);
                tl::ascend::pv_pair_minus(w_u[0], w_uu[0], _pair_temp_0[0]);
                tl::ascend::pv_pair_times(_pair_temp_0[0], w_u[0], w_u[0]);
                tl::ascend::pv_pair_plus(w_ou[0], w_ou[0], _pair_temp_0[0]);
              } else {
                if ((pool.GetValue((((int64_t)retained) + ((int64_t)i_4))) == 0.000000e+00f) && (pool.GetValue((((int64_t)retained) + ((int64_t)j_2))) == 0.000000e+00f)) {
                  row_bound = (row_bound + max((w_cc.GetValue(0) + w_cc.GetValue(1)), ((w_cc.GetValue(0) + w_cc.GetValue(1)) * -1.000000e+00f)));
                }
              }
            }
            discarded_gram = max(discarded_gram, row_bound);
            tl::ascend::pv_fill(pool[(width * 4)], pool[(width * 4)], pool[(width * 4)], 0.000000e+00f, width);
            tl::ascend::pv_fill(pool[(width * 5)], pool[(width * 5)], pool[(width * 5)], 0.000000e+00f, width);
            tl::ascend::pv_fill(pool[(width * 19)], pool[(width * 19)], pool[(width * 19)], 0.000000e+00f, width);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t j_3 = 0; j_3 < min(m, n); ++j_3) {
              tl::ascend::pv_fill(pool[(width * 2)], pool[(width * 2)], pool[(width * 2)], 0.000000e+00f, width);
              tl::ascend::pv_fill(pool[(width * 3)], pool[(width * 3)], pool[(width * 3)], 0.000000e+00f, width);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              if (n <= m) {
                tl::ascend::pv_load_rows(pool[((width * 20) + 32)], A[(((job * m) * n) + j_3)], pool[((width * 20) + 32)], max(m, n), 1, (n - 1), 0);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                for (int32_t j_4 = 0; j_4 < max(m, n); ++j_4) {
                  pool.SetValue(((((int64_t)width) * (int64_t)2) + ((int64_t)j_4)), pool.GetValue((((((int64_t)width) * (int64_t)20) + (((int64_t)j_4) * (int64_t)8)) + (int64_t)32)));
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              } else {
                tl::ascend::pv_load_rows(pool[(width * 2)], A[(n * ((job * m) + j_3))], pool[(width * 2)], 1, n, 0, 0);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              }
              tl::ascend::pv_scale(pool[(width * 2)], pool[(width * 2)], pool[(width * 2)], (1.000000e+00f / power), head);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              tl::ascend::pv_pair_pack(w_vij[0], w_vij[0], w_vij[0], pool.GetValue((((int64_t)head) + ((int64_t)j_3))), pool.GetValue(((((int64_t)width) + ((int64_t)head)) + ((int64_t)j_3))));
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_cert_scale(pool[0], pool[0], pool[0], (width * 2), w_vij.GetValue(0), w_vij.GetValue(1), head);
              tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 16), head);
              tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 17), head);
              tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 18), head);
              if (pool.GetValue((((int64_t)retained) + ((int64_t)i_4))) != 0.000000e+00f) {
                tl::ascend::pv_pair_cert_dot(_pair_temp_1[0], pool[0], _pair_temp_1[0], (width * 2), (width * 3), 0, width, head);
                tl::ascend::pv_pair_divide(_pair_temp_0[0], _pair_temp_1[0], w_ni[0]);
                tl::ascend::pv_pair_times(_pair_temp_2[0], w_vij[0], w_ni[0]);
                tl::ascend::pv_pair_minus(w_rhs[0], _pair_temp_0[0], _pair_temp_2[0]);
                tl::ascend::pv_pair_times(_pair_temp_0[0], w_rhs[0], w_rhs[0]);
                tl::ascend::pv_pair_plus(w_r2[0], w_r2[0], _pair_temp_0[0]);
              }
            }
            tl::ascend::pv_scale(pool[(width * 6)], pool[0], pool[(width * 6)], -1.000000e+00f, head);
            tl::ascend::pv_scale(pool[(width * 7)], pool[width], pool[(width * 7)], -1.000000e+00f, head);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 6), head);
            tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 7), head);
            tl::ascend::pv_add(pool[(width * 5)], pool[(width * 5)], pool[(width * 19)], head);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_pair_cert_dot(_pair_temp_0[0], pool[0], _pair_temp_0[0], (width * 4), (width * 5), (width * 4), (width * 5), head);
            tl::ascend::pv_pair_plus(w_r1[0], w_r1[0], _pair_temp_0[0]);
          }
          tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], total);
          norm = (_pair_temp_0.GetValue(0) * 1.001000e+00f);
          head_depth = 0;
          basis_depth = 0;
          size = head;
          while (1) {
            if (!((1 < size))) { break; }
            head_depth = (head_depth + 1);
            size = (size / 2);
          }
          size = basis;
          while (1) {
            if (!((1 < size))) { break; }
            basis_depth = (basis_depth + 1);
            size = (size / 2);
          }
          float dot_gamma0 = (1.136868e-13f + (1.694066e-21f * ((float)(head_depth + 4))));
          float basis_gamma0 = (2.842171e-14f + (1.694066e-21f * ((float)(basis_depth + 4))));
          float norm_gamma0 = (1.136868e-13f + (1.694066e-21f * ((float)(head_depth + 8))));
          float product_gamma0 = (1.694066e-21f * ((float)(min(m, n) + 4)));
          float output_gamma = (2.842171e-14f * ((float)(min(m, n) + 4)));
          float dot_gamma = ((dot_gamma0 / (1.000000e+00f - dot_gamma0)) * 1.001000e+00f);
          float basis_gamma = ((basis_gamma0 / (1.000000e+00f - basis_gamma0)) * 1.001000e+00f);
          float norm_gamma = ((norm_gamma0 / (1.000000e+00f - norm_gamma0)) * 1.001000e+00f);
          float product_gamma = ((product_gamma0 / (1.000000e+00f - product_gamma0)) * 1.001000e+00f);
          tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_ov.GetValue(0) + w_ov.GetValue(1)));
          float eta_v = ((_pair_temp_0.GetValue(0) * 1.001000e+00f) + ((basis_gamma * v_gram_total) * 1.001000e+00f));
          tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_ou.GetValue(0) + w_ou.GetValue(1)));
          float eta_u = ((_pair_temp_0.GetValue(0) * 1.001000e+00f) + ((norm_gamma * u_gram_total) * 1.001000e+00f));
          certified = 0;
          error = 0.000000e+00f;
          if ((eta_v < 2.500000e-01f) && (eta_u < 2.500000e-01f)) {
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (1.000000e+00f - eta_v));
            float dv = (eta_v / (1.000000e+00f + _pair_temp_0.GetValue(0)));
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (1.000000e+00f - eta_u));
            float du = (eta_u / (1.000000e+00f + _pair_temp_0.GetValue(0)));
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_r1.GetValue(0) + w_r1.GetValue(1)));
            float first_residual = (_pair_temp_0.GetValue(0) * 1.001000e+00f);
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], ((float)min(m, n)));
            float root_cols = _pair_temp_0.GetValue(0);
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (1.000000e+00f - eta_v));
            float basis_denominator = (_pair_temp_0.GetValue(0) - (product_gamma * root_cols));
            if (0.000000e+00f < basis_denominator) {
              norm = (((norm + first_residual) / basis_denominator) * 1.001000e+00f);
              float e1 = (first_residual + ((product_gamma * norm) * root_cols));
              tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], discarded_gram);
              float discarded = (((_pair_temp_0.GetValue(0) * 1.001000e+00f) + e1) + (norm * dv));
              tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], ((float)rank));
              float root_rank = _pair_temp_0.GetValue(0);
              tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_r2.GetValue(0) + w_r2.GetValue(1)));
              float delta = (((e1 + (_pair_temp_0.GetValue(0) * 1.001000e+00f)) + ((dot_gamma * norm) * root_rank)) + ((norm + maximum) * (du + dv)));
              float lower = ((maximum * 9.999999e-01f) - delta);
              float upper = ((maximum * 1.000000e+00f) + delta);
              float threshold_low = max((atol / power), (rtol * lower));
              float threshold_high = max((atol / power), (rtol * upper));
              if (rank == 0) {
                if ((discarded + delta) <= threshold_low) {
                  certified = 1;
                }
              } else {
                float smallest = ((minimum * 9.999999e-01f) - delta);
                float condval_5;
                if ((rank == min(m, n))) {
                  condval_5 = 0.000000e+00f;
                } else {
                  condval_5 = (discarded + delta);
                }
                float discarded_upper = condval_5;
                if (((threshold_high < smallest) && ((rank == min(m, n)) || (discarded_upper <= threshold_low))) && (discarded_upper < smallest)) {
                  float lipschitz = (1.000000e+00f / (smallest * (smallest - discarded_upper)));
                  error = (((((lipschitz * delta) + ((du + dv) / minimum)) + ((output_gamma * root_rank) / minimum)) / power) * 1.010000e+00f);
                  if ((0.000000e+00f <= error) && (error < 1.701412e+38f)) {
                    certified = 1;
                  }
                }
              }
            }
          }
          if (certified != 0) {
            valid = 1;
            int32_t condval_6;
            if ((n <= m)) {
              condval_6 = head;
            } else {
              condval_6 = basis;
            }
            int32_t count = condval_6;
            int32_t condval_7;
            if ((n <= m)) {
              condval_7 = 0;
            } else {
              condval_7 = head;
            }
            int32_t first = condval_7;
            for (int32_t output_row_1 = 0; output_row_1 < n; ++output_row_1) {
              tl::ascend::pv_fill(pool[(width * 4)], pool[(width * 4)], pool[(width * 4)], 0.000000e+00f, width);
              tl::ascend::pv_fill(pool[(width * 5)], pool[(width * 5)], pool[(width * 5)], 0.000000e+00f, width);
              AscendC::PipeBarrier<PIPE_V>();
              for (int32_t i_5 = 0; i_5 < min(m, n); ++i_5) {
                if (pool.GetValue((((int64_t)retained) + ((int64_t)i_5))) != 0.000000e+00f) {
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                  tl::ascend::pv_load_rows(pool[0], W[((((job * min(m, n)) * width) * 2) + (i_5 * width))], pool[0], 1, width, 0, 0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                  tl::ascend::pv_load_rows(pool[width], W[(((((job * min(m, n)) * width) * 2) + (min(m, n) * width)) + (i_5 * width))], pool[width], 1, width, 0, 0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                  tl::ascend::pv_pair_pack(w_norm[0], w_norm[0], w_norm[0], pool.GetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)i_5)) + (int64_t)32)), pool.GetValue((((int64_t)sigma_low) + ((int64_t)i_5))));
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  int32_t condval_8;
                  if ((n <= m)) {
                    condval_8 = (head + output_row_1);
                  } else {
                    condval_8 = output_row_1;
                  }
                  int32_t position = condval_8;
                  tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], pool.GetValue(((int64_t)position)), pool.GetValue((((int64_t)width) + ((int64_t)position))));
                  tl::ascend::pv_pair_times(_pair_temp_1[0], w_norm[0], w_norm[0]);
                  tl::ascend::pv_pair_divide(w_factor[0], _pair_temp_0[0], _pair_temp_1[0]);
                  AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                  tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 6), (width * 7), first, (width + first), w_factor.GetValue(0), w_factor.GetValue(1), count);
                  tl::ascend::pv_pair_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 4), (width * 5), (width * 6), (width * 7), count);
                }
              }
              tl::ascend::pv_scale(pool[(width * 4)], pool[(width * 4)], pool[(width * 4)], (1.000000e+00f / power), count);
              tl::ascend::pv_scale(pool[(width * 5)], pool[(width * 5)], pool[(width * 5)], (1.000000e+00f / power), count);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_add(pool[(width * 6)], pool[(width * 4)], pool[(width * 5)], count);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              for (int32_t col = 0; col < m; ++col) {
                float output_value = pool.GetValue(((((int64_t)width) * (int64_t)6) + ((int64_t)col)));
                float bound = (error + (5.960465e-08f * max(output_value, (output_value * -1.000000e+00f))));
                float budget = ((1.000000e-04f + (1.000000e-04f * max(output_value, (output_value * -1.000000e+00f)))) * 9.999000e-01f);
                if (budget < bound) {
                  valid = 0;
                }
              }
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
              AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
              tl::ascend::pv_store_rows(X[(((job * m) * n) + (output_row_1 * m))], pool[(width * 6)], X[(((job * m) * n) + (output_row_1 * m))], 1, m, 0, 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            }
            if (valid != 0) {
              accepted.SetValue(0, 1);
            }
          }
        }
      }
      if (accepted.GetValue(0) == 0) {
        tl::ascend::pv_fill(pool[(width * 6)], pool[(width * 6)], pool[(width * 6)], 0.000000e+00f, width);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_div(pool[(width * 6)], pool[(width * 6)], pool[(width * 6)], width);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t output_row_2 = 0; output_row_2 < n; ++output_row_2) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
          tl::ascend::pv_store_rows(X[(((job * m) * n) + (output_row_2 * m))], pool[(width * 6)], X[(((job * m) * n) + (output_row_2 * m))], 1, m, 0, 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
      }
    }
    S.SetValue((((int64_t)job) * (int64_t)32), ((float)accepted.GetValue(0)));
  }
  pipe.Destroy();
}

__aicore__ static __attribute__((noinline)) void PvDslResidentAccurate( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  int32_t &module_status, int32_t blocks, int32_t job, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 82208);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto accepted = ascend_ub.GetWithOffset<int>(8, 0);
  int pad = 0;
  int width = 0;
  int pitch = 0;
  auto pool = ascend_ub.GetWithOffset<float>(20544, 32);
  float magnitude = 0.000000e+00f;
  int valid = 0;
  float power = 0.000000e+00f;
  float total = 0.000000e+00f;
  float inverse_total = 0.000000e+00f;
  int pivot = 0;
  float best = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    accepted.SetValue(0, 0);
    pad = (((n + 7) / 8) * 8);
    width = (pad * 2);
    pitch = (width + 8);
    int32_t matrix = ((width * 20) + 32);
    int32_t low = (matrix + (n * pitch));
    int32_t bx = (low + (n * pitch));
    int32_t by = (bx + pitch);
    int32_t bu = (by + pitch);
    int32_t bv = (bu + pitch);
    int32_t red = (width * 20);
    int32_t prod = (width * 4);
    int32_t tmp = (width * 5);
    tl::ascend::pv_arith_init(pool[0], pool[0], pool[0], width);
    magnitude = 0.000000e+00f;
    valid = 1;
    power = 1.000000e+00f;
    total = 0.000000e+00f;
    for (int32_t i = 0; i < n; ++i) {
      tl::ascend::pv_fill(pool[(matrix + (i * pitch))], pool[(matrix + (i * pitch))], pool[(matrix + (i * pitch))], 0.000000e+00f, pitch);
      tl::ascend::pv_fill(pool[(low + (i * pitch))], pool[(low + (i * pitch))], pool[(low + (i * pitch))], 0.000000e+00f, pitch);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
      tl::ascend::pv_load_rows(pool[(matrix + (i * pitch))], A[(n * ((job * m) + i))], pool[(matrix + (i * pitch))], 1, n, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      tl::ascend::pv_abs(pool[prod], pool[(matrix + (i * pitch))], pool[prod], pad);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[red], pool[prod], pool[tmp], pad);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      magnitude = max(magnitude, pool.GetValue(((int64_t)red)));
      pool.SetValue((((((int64_t)matrix) + (((int64_t)i) * ((int64_t)pitch))) + ((int64_t)pad)) + ((int64_t)i)), 1.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    }
    if (magnitude <= 0.000000e+00f) {
      valid = 0;
    }
    if (valid != 0) {
      while (1) {
        if (!(((power < magnitude) && (power < 1.701412e+38f)))) { break; }
        power = (power * 2.000000e+00f);
      }
      while (1) {
        if (!((magnitude <= (power * 5.000000e-01f)))) { break; }
        power = (power * 5.000000e-01f);
      }
      for (int32_t i_1 = 0; i_1 < n; ++i_1) {
        tl::ascend::pv_scale(pool[(matrix + (i_1 * pitch))], pool[(matrix + (i_1 * pitch))], pool[(matrix + (i_1 * pitch))], (1.000000e+00f / power), pad);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mul(pool[prod], pool[(matrix + (i_1 * pitch))], pool[(matrix + (i_1 * pitch))], pad);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_reduce_sum(pool[red], pool[prod], pool[tmp], pad);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        total = (total + pool.GetValue(((int64_t)red)));
      }
      for (int32_t p = 0; p < n; ++p) {
        if (valid != 0) {
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          pivot = p;
          best = 0.000000e+00f;
          for (int32_t i_2 = p; i_2 < n; ++i_2) {
            float entry = pool.GetValue(((((int64_t)matrix) + (((int64_t)i_2) * ((int64_t)pitch))) + ((int64_t)p)));
            float absolute = max(entry, (entry * -1.000000e+00f));
            if (best < absolute) {
              best = absolute;
              pivot = i_2;
            }
          }
          if (best <= 0.000000e+00f) {
            valid = 0;
          } else {
            if (pivot != p) {
              tl::ascend::pv_copy(pool[bx], pool[(matrix + (p * pitch))], pool[bx], pitch);
              tl::ascend::pv_copy(pool[by], pool[(low + (p * pitch))], pool[by], pitch);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_copy(pool[(matrix + (p * pitch))], pool[(matrix + (pivot * pitch))], pool[(matrix + (p * pitch))], pitch);
              tl::ascend::pv_copy(pool[(low + (p * pitch))], pool[(low + (pivot * pitch))], pool[(low + (p * pitch))], pitch);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_copy(pool[(matrix + (pivot * pitch))], pool[bx], pool[(matrix + (pivot * pitch))], pitch);
              tl::ascend::pv_copy(pool[(low + (pivot * pitch))], pool[by], pool[(low + (pivot * pitch))], pitch);
              AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            float dh = pool.GetValue(((((int64_t)matrix) + (((int64_t)p) * ((int64_t)pitch))) + ((int64_t)p)));
            float dl = pool.GetValue(((((int64_t)low) + (((int64_t)p) * ((int64_t)pitch))) + ((int64_t)p)));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            tl::ascend::pv_pair_normalize(pool[0], pool[0], pool[0], (matrix + (p * pitch)), (low + (p * pitch)), bu, bv, dh, dl, width);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            for (int32_t i_3 = 0; i_3 < n; ++i_3) {
              if (i_3 != p) {
                float fh = pool.GetValue(((((int64_t)matrix) + (((int64_t)i_3) * ((int64_t)pitch))) + ((int64_t)p)));
                float fl = pool.GetValue(((((int64_t)low) + (((int64_t)i_3) * ((int64_t)pitch))) + ((int64_t)p)));
                tl::ascend::pv_pair_eliminate(pool[0], pool[0], pool[0], (matrix + (i_3 * pitch)), (low + (i_3 * pitch)), (matrix + (p * pitch)), (low + (p * pitch)), fh, fl, width);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              }
            }
          }
        }
      }
      if (valid != 0) {
        inverse_total = 0.000000e+00f;
        for (int32_t i_4 = 0; i_4 < n; ++i_4) {
          tl::ascend::pv_mul(pool[prod], pool[((matrix + (i_4 * pitch)) + pad)], pool[((matrix + (i_4 * pitch)) + pad)], pad);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_reduce_sum(pool[red], pool[prod], pool[tmp], pad);
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          inverse_total = (inverse_total + pool.GetValue(((int64_t)red)));
        }
        pool.SetValue(((int64_t)red), total);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        tl::ascend::pv_sqrt(pool[red], pool[red], pool[red], 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        float cutoff = max((pool.GetValue(((int64_t)red)) * rtol), (atol / power));
        if ((inverse_total <= 0.000000e+00f) || (8.100000e-01f <= ((inverse_total * cutoff) * cutoff))) {
          valid = 0;
        } else {
          for (int32_t i_5 = 0; i_5 < n; ++i_5) {
            tl::ascend::pv_add(pool[bx], pool[((matrix + (i_5 * pitch)) + pad)], pool[((low + (i_5 * pitch)) + pad)], pad);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_scale(pool[bx], pool[bx], pool[bx], (1.000000e+00f / power), pad);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(X[(n * ((job * m) + i_5))], pool[bx], X[(n * ((job * m) + i_5))], 1, n, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          }
        }
      }
    }
    accepted.SetValue(0, valid);
    module_status = static_cast<int32_t>(((float)accepted.GetValue(0)));
  }
  pipe.Destroy();
}

__aicore__ static __attribute__((noinline)) void PvDslResidentRectangular( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  int32_t &module_status, int32_t blocks, int32_t job, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 79680);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto accepted = ascend_ub.GetWithOffset<int>(8, 0);
  int head = 0;
  int basis = 0;
  auto pool = ascend_ub.GetWithOffset<float>(3296, 32);
  float power = 0.000000e+00f;
  float peak = 0.000000e+00f;
  auto resident_work = ascend_ub.GetWithOffset<float>(16384, 13216);
  int sweep = 0;
  int changed = 0;
  auto _pair_temp_0 = ascend_ub.GetWithOffset<float>(8, 78848);
  auto _pair_temp_2 = ascend_ub.GetWithOffset<float>(8, 78880);
  auto _pair_temp_1 = ascend_ub.GetWithOffset<float>(8, 78912);
  float maximum = 0.000000e+00f;
  float minimum = 0.000000e+00f;
  float total = 0.000000e+00f;
  int rank = 0;
  auto w_norm = ascend_ub.GetWithOffset<float>(8, 79200);
  auto w_ov = ascend_ub.GetWithOffset<float>(8, 79232);
  auto w_ou = ascend_ub.GetWithOffset<float>(8, 79264);
  auto w_r1 = ascend_ub.GetWithOffset<float>(8, 79296);
  auto w_r2 = ascend_ub.GetWithOffset<float>(8, 79328);
  float discarded_gram = 0.000000e+00f;
  float v_gram_total = 0.000000e+00f;
  float u_gram_total = 0.000000e+00f;
  float norm = 0.000000e+00f;
  int head_depth = 0;
  int basis_depth = 0;
  int size = 0;
  int certified = 0;
  float error = 0.000000e+00f;
  int valid = 0;
  auto w_a = ascend_ub.GetWithOffset<float>(8, 78752);
  auto w_d = ascend_ub.GetWithOffset<float>(8, 78784);
  auto w_h = ascend_ub.GetWithOffset<float>(8, 78816);
  auto w_tau = ascend_ub.GetWithOffset<float>(8, 78944);
  auto w_z = ascend_ub.GetWithOffset<float>(8, 78976);
  auto w_root = ascend_ub.GetWithOffset<float>(8, 79008);
  auto w_absolute = ascend_ub.GetWithOffset<float>(8, 79040);
  auto _pair_temp_3 = ascend_ub.GetWithOffset<float>(8, 79072);
  auto _pair_temp_4 = ascend_ub.GetWithOffset<float>(8, 79104);
  auto w_c = ascend_ub.GetWithOffset<float>(8, 79136);
  auto w_sn = ascend_ub.GetWithOffset<float>(8, 79168);
  auto w_ni = ascend_ub.GetWithOffset<float>(8, 79360);
  float row_bound = 0.000000e+00f;
  auto w_vv = ascend_ub.GetWithOffset<float>(8, 79392);
  auto w_v = ascend_ub.GetWithOffset<float>(8, 79424);
  auto w_cc = ascend_ub.GetWithOffset<float>(8, 79456);
  auto w_nj = ascend_ub.GetWithOffset<float>(8, 79488);
  auto w_uu = ascend_ub.GetWithOffset<float>(8, 79520);
  auto w_u = ascend_ub.GetWithOffset<float>(8, 79552);
  auto w_vij = ascend_ub.GetWithOffset<float>(8, 79584);
  auto w_rhs = ascend_ub.GetWithOffset<float>(8, 79616);
  auto w_factor = ascend_ub.GetWithOffset<float>(8, 79648);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    accepted.SetValue(0, 0);
    head = 8;
    basis = 8;
    while (1) {
      if (!(((head < m) || (head < n)))) { break; }
      head = (head * 2);
    }
    while (1) {
      if (!(((basis < m) && (basis < n)))) { break; }
      basis = (basis * 2);
    }
    int32_t width = (head + basis);
    int32_t sigma_low = ((((width * 20) + (max(m, n) * 8)) + basis) + 32);
    int32_t retained = ((((width * 20) + (max(m, n) * 8)) + (basis * 2)) + 32);
    tl::ascend::pv_arith_init(pool[0], pool[0], pool[0], width);
    accepted.SetValue(0, 0);
    power = 1.000000e+00f;
    peak = 0.000000e+00f;
    for (int32_t i = 0; i < min(m, n); ++i) {
      tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, width);
      tl::ascend::pv_fill(pool[width], pool[width], pool[width], 0.000000e+00f, width);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
      if (n <= m) {
        tl::ascend::pv_load_rows(pool[((width * 20) + 32)], A[(((job * m) * n) + i)], pool[((width * 20) + 32)], max(m, n), 1, (n - 1), 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        for (int32_t j = 0; j < max(m, n); ++j) {
          pool.SetValue(j, pool.GetValue((((((int64_t)width) * (int64_t)20) + (((int64_t)j) * (int64_t)8)) + (int64_t)32)));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      } else {
        tl::ascend::pv_load_rows(pool[0], A[(n * ((job * m) + i))], pool[0], 1, n, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      }
      tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power), head);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_abs(pool[(width * 4)], pool[0], pool[(width * 4)], head);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[(width * 20)], pool[(width * 4)], pool[(width * 8)], head);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      peak = max(peak, pool.GetValue((((int64_t)width) * (int64_t)20)));
    }
    if (peak == 0.000000e+00f) {
      tl::ascend::pv_fill(pool[(width * 6)], pool[(width * 6)], pool[(width * 6)], 0.000000e+00f, width);
      AscendC::PipeBarrier<PIPE_V>();
      for (int32_t output_row = 0; output_row < n; ++output_row) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        tl::ascend::pv_store_rows(X[(((job * m) * n) + (output_row * m))], pool[(width * 6)], X[(((job * m) * n) + (output_row * m))], 1, m, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
      }
      accepted.SetValue(0, 1);
    } else {
      while (1) {
        if (!(((power < peak) && (power < 1.701412e+38f)))) { break; }
        power = (power * 2.000000e+00f);
      }
      while (1) {
        if (!((peak <= (power * 5.000000e-01f)))) { break; }
        power = (power * 5.000000e-01f);
      }
      for (int32_t attempt = 0; attempt < 2; ++attempt) {
        if (accepted.GetValue(0) == 0) {
          for (int32_t i_1 = 0; i_1 < min(m, n); ++i_1) {
            tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, width);
            tl::ascend::pv_fill(pool[width], pool[width], pool[width], 0.000000e+00f, width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            if (n <= m) {
              tl::ascend::pv_load_rows(pool[((width * 20) + 32)], A[(((job * m) * n) + i_1)], pool[((width * 20) + 32)], max(m, n), 1, (n - 1), 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
              for (int32_t j_1 = 0; j_1 < max(m, n); ++j_1) {
                pool.SetValue(j_1, pool.GetValue((((((int64_t)width) * (int64_t)20) + (((int64_t)j_1) * (int64_t)8)) + (int64_t)32)));
              }
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            } else {
              tl::ascend::pv_load_rows(pool[0], A[(n * ((job * m) + i_1))], pool[0], 1, n, 0, 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            }
            tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power), head);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            pool.SetValue((((int64_t)head) + ((int64_t)i_1)), 1.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            tl::ascend::pv_copy(resident_work[(i_1 * width)], pool[0], resident_work[(i_1 * width)], width);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_copy(resident_work[(width * (min(m, n) + i_1))], pool[width], resident_work[(width * (min(m, n) + i_1))], width);
            AscendC::PipeBarrier<PIPE_V>();
          }
          sweep = 0;
          changed = 1;
          float condval;
          if ((attempt == 0)) {
            condval = 1.000000e-06f;
          } else {
            condval = 2.000000e-13f;
          }
          float orth = condval;
          while (1) {
            int32_t condval_1;
            if ((attempt == 0)) {
              condval_1 = 40;
            } else {
              condval_1 = 120;
            }
            if (!(((sweep < condval_1) && (changed != 0)))) { break; }
            changed = 0;
            for (int32_t p = 0; p < min(m, n); ++p) {
              tl::ascend::pv_copy(pool[0], resident_work[(p * width)], pool[0], width);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              tl::ascend::pv_copy(pool[width], resident_work[(width * (min(m, n) + p))], pool[width], width);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              for (int32_t q = (p + 1); q < min(m, n); ++q) {
                tl::ascend::pv_copy(pool[(width * 2)], resident_work[(q * width)], pool[(width * 2)], width);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::pv_copy(pool[(width * 3)], resident_work[(width * (min(m, n) + q))], pool[(width * 3)], width);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::pv_pair_dot(w_a[0], pool[0], w_a[0], 0, width, 0, width, head);
                tl::ascend::pv_pair_dot(w_d[0], pool[0], w_d[0], (width * 2), (width * 3), (width * 2), (width * 3), head);
                tl::ascend::pv_pair_dot(w_h[0], pool[0], w_h[0], 0, width, (width * 2), (width * 3), head);
                tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_a.GetValue(0) * w_d.GetValue(0)));
                float norm_product = _pair_temp_0.GetValue(0);
                if (((0.000000e+00f < w_a.GetValue(0)) && (0.000000e+00f < w_d.GetValue(0))) && ((orth * norm_product) < max((w_h.GetValue(0) + w_h.GetValue(1)), ((w_h.GetValue(0) + w_h.GetValue(1)) * -1.000000e+00f)))) {
                  tl::ascend::pv_pair_minus(_pair_temp_0[0], w_d[0], w_a[0]);
                  tl::ascend::pv_pair_pack(_pair_temp_2[0], _pair_temp_2[0], _pair_temp_2[0], 2.000000e+00f, 0.000000e+00f);
                  tl::ascend::pv_pair_times(_pair_temp_1[0], _pair_temp_2[0], w_h[0]);
                  tl::ascend::pv_pair_divide(w_tau[0], _pair_temp_0[0], _pair_temp_1[0]);
                  if (1.000000e+08f < max(w_tau.GetValue(0), (w_tau.GetValue(0) * -1.000000e+00f))) {
                    tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], 5.000000e-01f, 0.000000e+00f);
                    tl::ascend::pv_pair_divide(w_z[0], _pair_temp_0[0], w_tau[0]);
                  } else {
                    tl::ascend::pv_pair_pack(_pair_temp_1[0], _pair_temp_1[0], _pair_temp_1[0], 1.000000e+00f, 0.000000e+00f);
                    tl::ascend::pv_pair_times(_pair_temp_2[0], w_tau[0], w_tau[0]);
                    tl::ascend::pv_pair_plus(_pair_temp_0[0], _pair_temp_1[0], _pair_temp_2[0]);
                    tl::ascend::pv_pair_root(w_root[0], _pair_temp_0[0], pool[0]);
                    tl::ascend::pv_pair_pack(w_absolute[0], w_absolute[0], w_absolute[0], w_tau.GetValue(0), w_tau.GetValue(1));
                    if (w_tau.GetValue(0) < 0.000000e+00f) {
                      tl::ascend::pv_pair_pack(w_absolute[0], w_absolute[0], w_absolute[0], (w_tau.GetValue(0) * -1.000000e+00f), (w_tau.GetValue(1) * -1.000000e+00f));
                    }
                    float condval_2;
                    if ((w_tau.GetValue(0) < 0.000000e+00f)) {
                      condval_2 = -1.000000e+00f;
                    } else {
                      condval_2 = 1.000000e+00f;
                    }
                    tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], condval_2, 0.000000e+00f);
                    tl::ascend::pv_pair_plus(_pair_temp_1[0], w_absolute[0], w_root[0]);
                    tl::ascend::pv_pair_divide(w_z[0], _pair_temp_0[0], _pair_temp_1[0]);
                  }
                  tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], 1.000000e+00f, 0.000000e+00f);
                  tl::ascend::pv_pair_pack(_pair_temp_3[0], _pair_temp_3[0], _pair_temp_3[0], 1.000000e+00f, 0.000000e+00f);
                  tl::ascend::pv_pair_times(_pair_temp_4[0], w_z[0], w_z[0]);
                  tl::ascend::pv_pair_plus(_pair_temp_2[0], _pair_temp_3[0], _pair_temp_4[0]);
                  tl::ascend::pv_pair_root(_pair_temp_1[0], _pair_temp_2[0], pool[0]);
                  tl::ascend::pv_pair_divide(w_c[0], _pair_temp_0[0], _pair_temp_1[0]);
                  tl::ascend::pv_pair_times(w_sn[0], w_c[0], w_z[0]);
                  if (0 < attempt) {
                    tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 4), (width * 5), 0, width, w_c.GetValue(0), w_c.GetValue(1), width);
                    tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 6), (width * 7), (width * 2), (width * 3), (w_sn.GetValue(0) * -1.000000e+00f), (w_sn.GetValue(1) * -1.000000e+00f), width);
                    tl::ascend::pv_pair_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 4), (width * 5), (width * 6), (width * 7), width);
                    tl::ascend::pv_copy(resident_work[(p * width)], pool[(width * 4)], resident_work[(p * width)], width);
                    AscendC::PipeBarrier<PIPE_V>();
                    tl::ascend::pv_copy(resident_work[(width * (min(m, n) + p))], pool[(width * 5)], resident_work[(width * (min(m, n) + p))], width);
                    AscendC::PipeBarrier<PIPE_V>();
                    tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 4), (width * 5), 0, width, w_sn.GetValue(0), w_sn.GetValue(1), width);
                    tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 6), (width * 7), (width * 2), (width * 3), w_c.GetValue(0), w_c.GetValue(1), width);
                    tl::ascend::pv_pair_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 4), (width * 5), (width * 6), (width * 7), width);
                  } else {
                    tl::ascend::pv_scale(pool[(width * 4)], pool[0], pool[(width * 4)], w_c.GetValue(0), width);
                    tl::ascend::pv_scale(pool[(width * 6)], pool[(width * 2)], pool[(width * 6)], (w_sn.GetValue(0) * -1.000000e+00f), width);
                    AscendC::PipeBarrier<PIPE_V>();
                    tl::ascend::pv_add(pool[(width * 4)], pool[(width * 4)], pool[(width * 6)], width);
                    tl::ascend::pv_fill(pool[(width * 5)], pool[(width * 5)], pool[(width * 5)], 0.000000e+00f, width);
                    AscendC::PipeBarrier<PIPE_V>();
                    tl::ascend::pv_copy(resident_work[(p * width)], pool[(width * 4)], resident_work[(p * width)], width);
                    AscendC::PipeBarrier<PIPE_V>();
                    tl::ascend::pv_copy(resident_work[(width * (min(m, n) + p))], pool[(width * 5)], resident_work[(width * (min(m, n) + p))], width);
                    AscendC::PipeBarrier<PIPE_V>();
                    tl::ascend::pv_scale(pool[(width * 4)], pool[0], pool[(width * 4)], w_sn.GetValue(0), width);
                    tl::ascend::pv_scale(pool[(width * 6)], pool[(width * 2)], pool[(width * 6)], w_c.GetValue(0), width);
                    AscendC::PipeBarrier<PIPE_V>();
                    tl::ascend::pv_add(pool[(width * 4)], pool[(width * 4)], pool[(width * 6)], width);
                    AscendC::PipeBarrier<PIPE_V>();
                  }
                  tl::ascend::pv_copy(resident_work[(q * width)], pool[(width * 4)], resident_work[(q * width)], width);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_copy(resident_work[(width * (min(m, n) + q))], pool[(width * 5)], resident_work[(width * (min(m, n) + q))], width);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_copy(pool[0], resident_work[(p * width)], pool[0], width);
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  tl::ascend::pv_copy(pool[width], resident_work[(width * (min(m, n) + p))], pool[width], width);
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  changed = (changed + 1);
                }
              }
            }
            sweep = (sweep + 1);
          }
          maximum = 0.000000e+00f;
          minimum = 1.701412e+38f;
          total = 0.000000e+00f;
          rank = 0;
          for (int32_t i_2 = 0; i_2 < min(m, n); ++i_2) {
            tl::ascend::pv_copy(pool[0], resident_work[(i_2 * width)], pool[0], width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            tl::ascend::pv_copy(pool[width], resident_work[(width * (min(m, n) + i_2))], pool[width], width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            tl::ascend::pv_pair_cert_dot(_pair_temp_0[0], pool[0], _pair_temp_0[0], 0, width, 0, width, head);
            tl::ascend::pv_pair_root(w_norm[0], _pair_temp_0[0], pool[0]);
            float magnitude = (w_norm.GetValue(0) + w_norm.GetValue(1));
            pool.SetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)i_2)) + (int64_t)32), w_norm.GetValue(0));
            pool.SetValue((((int64_t)sigma_low) + ((int64_t)i_2)), w_norm.GetValue(1));
            maximum = max(maximum, magnitude);
            total = (total + (magnitude * magnitude));
          }
          float cutoff = max((atol / power), (rtol * maximum));
          for (int32_t i_3 = 0; i_3 < min(m, n); ++i_3) {
            float magnitude_1 = (pool.GetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)i_3)) + (int64_t)32)) + pool.GetValue((((int64_t)sigma_low) + ((int64_t)i_3))));
            if (cutoff < magnitude_1) {
              pool.SetValue((((int64_t)retained) + ((int64_t)i_3)), 1.000000e+00f);
            } else {
              pool.SetValue((((int64_t)retained) + ((int64_t)i_3)), 0.000000e+00f);
            }
            if (cutoff < magnitude_1) {
              rank = (rank + 1);
              minimum = min(minimum, magnitude_1);
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          tl::ascend::pv_pair_pack(w_ov[0], w_ov[0], w_ov[0], 0.000000e+00f, 0.000000e+00f);
          tl::ascend::pv_pair_pack(w_ou[0], w_ou[0], w_ou[0], 0.000000e+00f, 0.000000e+00f);
          tl::ascend::pv_pair_pack(w_r1[0], w_r1[0], w_r1[0], 0.000000e+00f, 0.000000e+00f);
          tl::ascend::pv_pair_pack(w_r2[0], w_r2[0], w_r2[0], 0.000000e+00f, 0.000000e+00f);
          discarded_gram = 0.000000e+00f;
          v_gram_total = 0.000000e+00f;
          u_gram_total = 0.000000e+00f;
          for (int32_t i_4 = 0; i_4 < min(m, n); ++i_4) {
            tl::ascend::pv_copy(pool[0], resident_work[(i_4 * width)], pool[0], width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            tl::ascend::pv_copy(pool[width], resident_work[(width * (min(m, n) + i_4))], pool[width], width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            tl::ascend::pv_pair_pack(w_ni[0], w_ni[0], w_ni[0], pool.GetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)i_4)) + (int64_t)32)), pool.GetValue((((int64_t)sigma_low) + ((int64_t)i_4))));
            row_bound = 0.000000e+00f;
            for (int32_t j_2 = 0; j_2 < min(m, n); ++j_2) {
              tl::ascend::pv_copy(pool[(width * 2)], resident_work[(j_2 * width)], pool[(width * 2)], width);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              tl::ascend::pv_copy(pool[(width * 3)], resident_work[(width * (min(m, n) + j_2))], pool[(width * 3)], width);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              tl::ascend::pv_pair_cert_dot(w_vv[0], pool[0], w_vv[0], head, (width + head), ((width * 2) + head), ((width * 3) + head), basis);
              if (i_4 == j_2) {
                v_gram_total = (v_gram_total + (w_vv.GetValue(0) + w_vv.GetValue(1)));
              }
              float condval_3;
              if ((i_4 == j_2)) {
                condval_3 = 1.000000e+00f;
              } else {
                condval_3 = 0.000000e+00f;
              }
              tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], condval_3, 0.000000e+00f);
              tl::ascend::pv_pair_minus(w_v[0], w_vv[0], _pair_temp_0[0]);
              tl::ascend::pv_pair_times(_pair_temp_0[0], w_v[0], w_v[0]);
              tl::ascend::pv_pair_plus(w_ov[0], w_ov[0], _pair_temp_0[0]);
              tl::ascend::pv_pair_cert_dot(w_cc[0], pool[0], w_cc[0], 0, width, (width * 2), (width * 3), head);
              if ((pool.GetValue((((int64_t)retained) + ((int64_t)i_4))) != 0.000000e+00f) && (pool.GetValue((((int64_t)retained) + ((int64_t)j_2))) != 0.000000e+00f)) {
                tl::ascend::pv_pair_pack(w_nj[0], w_nj[0], w_nj[0], pool.GetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)j_2)) + (int64_t)32)), pool.GetValue((((int64_t)sigma_low) + ((int64_t)j_2))));
                tl::ascend::pv_pair_times(_pair_temp_0[0], w_ni[0], w_nj[0]);
                tl::ascend::pv_pair_divide(w_uu[0], w_cc[0], _pair_temp_0[0]);
                if (i_4 == j_2) {
                  u_gram_total = (u_gram_total + (w_uu.GetValue(0) + w_uu.GetValue(1)));
                }
                float condval_4;
                if ((i_4 == j_2)) {
                  condval_4 = 1.000000e+00f;
                } else {
                  condval_4 = 0.000000e+00f;
                }
                tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], condval_4, 0.000000e+00f);
                tl::ascend::pv_pair_minus(w_u[0], w_uu[0], _pair_temp_0[0]);
                tl::ascend::pv_pair_times(_pair_temp_0[0], w_u[0], w_u[0]);
                tl::ascend::pv_pair_plus(w_ou[0], w_ou[0], _pair_temp_0[0]);
              } else {
                if ((pool.GetValue((((int64_t)retained) + ((int64_t)i_4))) == 0.000000e+00f) && (pool.GetValue((((int64_t)retained) + ((int64_t)j_2))) == 0.000000e+00f)) {
                  row_bound = (row_bound + max((w_cc.GetValue(0) + w_cc.GetValue(1)), ((w_cc.GetValue(0) + w_cc.GetValue(1)) * -1.000000e+00f)));
                }
              }
            }
            discarded_gram = max(discarded_gram, row_bound);
            tl::ascend::pv_fill(pool[(width * 4)], pool[(width * 4)], pool[(width * 4)], 0.000000e+00f, width);
            tl::ascend::pv_fill(pool[(width * 5)], pool[(width * 5)], pool[(width * 5)], 0.000000e+00f, width);
            tl::ascend::pv_fill(pool[(width * 19)], pool[(width * 19)], pool[(width * 19)], 0.000000e+00f, width);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t j_3 = 0; j_3 < min(m, n); ++j_3) {
              tl::ascend::pv_fill(pool[(width * 2)], pool[(width * 2)], pool[(width * 2)], 0.000000e+00f, width);
              tl::ascend::pv_fill(pool[(width * 3)], pool[(width * 3)], pool[(width * 3)], 0.000000e+00f, width);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              if (n <= m) {
                tl::ascend::pv_load_rows(pool[((width * 20) + 32)], A[(((job * m) * n) + j_3)], pool[((width * 20) + 32)], max(m, n), 1, (n - 1), 0);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                for (int32_t j_4 = 0; j_4 < max(m, n); ++j_4) {
                  pool.SetValue(((((int64_t)width) * (int64_t)2) + ((int64_t)j_4)), pool.GetValue((((((int64_t)width) * (int64_t)20) + (((int64_t)j_4) * (int64_t)8)) + (int64_t)32)));
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              } else {
                tl::ascend::pv_load_rows(pool[(width * 2)], A[(n * ((job * m) + j_3))], pool[(width * 2)], 1, n, 0, 0);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              }
              tl::ascend::pv_scale(pool[(width * 2)], pool[(width * 2)], pool[(width * 2)], (1.000000e+00f / power), head);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              tl::ascend::pv_pair_pack(w_vij[0], w_vij[0], w_vij[0], pool.GetValue((((int64_t)head) + ((int64_t)j_3))), pool.GetValue(((((int64_t)width) + ((int64_t)head)) + ((int64_t)j_3))));
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_cert_scale(pool[0], pool[0], pool[0], (width * 2), w_vij.GetValue(0), w_vij.GetValue(1), head);
              tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 16), head);
              tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 17), head);
              tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 18), head);
              if (pool.GetValue((((int64_t)retained) + ((int64_t)i_4))) != 0.000000e+00f) {
                tl::ascend::pv_pair_cert_dot(_pair_temp_1[0], pool[0], _pair_temp_1[0], (width * 2), (width * 3), 0, width, head);
                tl::ascend::pv_pair_divide(_pair_temp_0[0], _pair_temp_1[0], w_ni[0]);
                tl::ascend::pv_pair_times(_pair_temp_2[0], w_vij[0], w_ni[0]);
                tl::ascend::pv_pair_minus(w_rhs[0], _pair_temp_0[0], _pair_temp_2[0]);
                tl::ascend::pv_pair_times(_pair_temp_0[0], w_rhs[0], w_rhs[0]);
                tl::ascend::pv_pair_plus(w_r2[0], w_r2[0], _pair_temp_0[0]);
              }
            }
            tl::ascend::pv_scale(pool[(width * 6)], pool[0], pool[(width * 6)], -1.000000e+00f, head);
            tl::ascend::pv_scale(pool[(width * 7)], pool[width], pool[(width * 7)], -1.000000e+00f, head);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 6), head);
            tl::ascend::pv_cert_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 19), (width * 7), head);
            tl::ascend::pv_add(pool[(width * 5)], pool[(width * 5)], pool[(width * 19)], head);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_pair_cert_dot(_pair_temp_0[0], pool[0], _pair_temp_0[0], (width * 4), (width * 5), (width * 4), (width * 5), head);
            tl::ascend::pv_pair_plus(w_r1[0], w_r1[0], _pair_temp_0[0]);
          }
          tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], total);
          norm = (_pair_temp_0.GetValue(0) * 1.001000e+00f);
          head_depth = 0;
          basis_depth = 0;
          size = head;
          while (1) {
            if (!((1 < size))) { break; }
            head_depth = (head_depth + 1);
            size = (size / 2);
          }
          size = basis;
          while (1) {
            if (!((1 < size))) { break; }
            basis_depth = (basis_depth + 1);
            size = (size / 2);
          }
          float dot_gamma0 = (1.136868e-13f + (1.694066e-21f * ((float)(head_depth + 4))));
          float basis_gamma0 = (2.842171e-14f + (1.694066e-21f * ((float)(basis_depth + 4))));
          float norm_gamma0 = (1.136868e-13f + (1.694066e-21f * ((float)(head_depth + 8))));
          float product_gamma0 = (1.694066e-21f * ((float)(min(m, n) + 4)));
          float output_gamma = (2.842171e-14f * ((float)(min(m, n) + 4)));
          float dot_gamma = ((dot_gamma0 / (1.000000e+00f - dot_gamma0)) * 1.001000e+00f);
          float basis_gamma = ((basis_gamma0 / (1.000000e+00f - basis_gamma0)) * 1.001000e+00f);
          float norm_gamma = ((norm_gamma0 / (1.000000e+00f - norm_gamma0)) * 1.001000e+00f);
          float product_gamma = ((product_gamma0 / (1.000000e+00f - product_gamma0)) * 1.001000e+00f);
          tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_ov.GetValue(0) + w_ov.GetValue(1)));
          float eta_v = ((_pair_temp_0.GetValue(0) * 1.001000e+00f) + ((basis_gamma * v_gram_total) * 1.001000e+00f));
          tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_ou.GetValue(0) + w_ou.GetValue(1)));
          float eta_u = ((_pair_temp_0.GetValue(0) * 1.001000e+00f) + ((norm_gamma * u_gram_total) * 1.001000e+00f));
          certified = 0;
          error = 0.000000e+00f;
          if ((eta_v < 2.500000e-01f) && (eta_u < 2.500000e-01f)) {
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (1.000000e+00f - eta_v));
            float dv = (eta_v / (1.000000e+00f + _pair_temp_0.GetValue(0)));
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (1.000000e+00f - eta_u));
            float du = (eta_u / (1.000000e+00f + _pair_temp_0.GetValue(0)));
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_r1.GetValue(0) + w_r1.GetValue(1)));
            float first_residual = (_pair_temp_0.GetValue(0) * 1.001000e+00f);
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], ((float)min(m, n)));
            float root_cols = _pair_temp_0.GetValue(0);
            tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (1.000000e+00f - eta_v));
            float basis_denominator = (_pair_temp_0.GetValue(0) - (product_gamma * root_cols));
            if (0.000000e+00f < basis_denominator) {
              norm = (((norm + first_residual) / basis_denominator) * 1.001000e+00f);
              float e1 = (first_residual + ((product_gamma * norm) * root_cols));
              tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], discarded_gram);
              float discarded = (((_pair_temp_0.GetValue(0) * 1.001000e+00f) + e1) + (norm * dv));
              tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], ((float)rank));
              float root_rank = _pair_temp_0.GetValue(0);
              tl::ascend::pv_root(_pair_temp_0[0], pool[0], _pair_temp_0[0], (w_r2.GetValue(0) + w_r2.GetValue(1)));
              float delta = (((e1 + (_pair_temp_0.GetValue(0) * 1.001000e+00f)) + ((dot_gamma * norm) * root_rank)) + ((norm + maximum) * (du + dv)));
              float lower = ((maximum * 9.999999e-01f) - delta);
              float upper = ((maximum * 1.000000e+00f) + delta);
              float threshold_low = max((atol / power), (rtol * lower));
              float threshold_high = max((atol / power), (rtol * upper));
              if (rank == 0) {
                if ((discarded + delta) <= threshold_low) {
                  certified = 1;
                }
              } else {
                float smallest = ((minimum * 9.999999e-01f) - delta);
                float condval_5;
                if ((rank == min(m, n))) {
                  condval_5 = 0.000000e+00f;
                } else {
                  condval_5 = (discarded + delta);
                }
                float discarded_upper = condval_5;
                if (((threshold_high < smallest) && ((rank == min(m, n)) || (discarded_upper <= threshold_low))) && (discarded_upper < smallest)) {
                  float lipschitz = (1.000000e+00f / (smallest * (smallest - discarded_upper)));
                  error = (((((lipschitz * delta) + ((du + dv) / minimum)) + ((output_gamma * root_rank) / minimum)) / power) * 1.010000e+00f);
                  if ((0.000000e+00f <= error) && (error < 1.701412e+38f)) {
                    certified = 1;
                  }
                }
              }
            }
          }
          if (certified != 0) {
            valid = 1;
            int32_t condval_6;
            if ((n <= m)) {
              condval_6 = head;
            } else {
              condval_6 = basis;
            }
            int32_t count = condval_6;
            int32_t condval_7;
            if ((n <= m)) {
              condval_7 = 0;
            } else {
              condval_7 = head;
            }
            int32_t first = condval_7;
            for (int32_t output_row_1 = 0; output_row_1 < n; ++output_row_1) {
              tl::ascend::pv_fill(pool[(width * 4)], pool[(width * 4)], pool[(width * 4)], 0.000000e+00f, width);
              tl::ascend::pv_fill(pool[(width * 5)], pool[(width * 5)], pool[(width * 5)], 0.000000e+00f, width);
              AscendC::PipeBarrier<PIPE_V>();
              for (int32_t i_5 = 0; i_5 < min(m, n); ++i_5) {
                if (pool.GetValue((((int64_t)retained) + ((int64_t)i_5))) != 0.000000e+00f) {
                  tl::ascend::pv_copy(pool[0], resident_work[(i_5 * width)], pool[0], width);
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  tl::ascend::pv_copy(pool[width], resident_work[(width * (min(m, n) + i_5))], pool[width], width);
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  tl::ascend::pv_pair_pack(w_norm[0], w_norm[0], w_norm[0], pool.GetValue(((((((int64_t)width) * (int64_t)20) + (max(((int64_t)m), ((int64_t)n)) * (int64_t)8)) + ((int64_t)i_5)) + (int64_t)32)), pool.GetValue((((int64_t)sigma_low) + ((int64_t)i_5))));
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  int32_t condval_8;
                  if ((n <= m)) {
                    condval_8 = (head + output_row_1);
                  } else {
                    condval_8 = output_row_1;
                  }
                  int32_t position = condval_8;
                  tl::ascend::pv_pair_pack(_pair_temp_0[0], _pair_temp_0[0], _pair_temp_0[0], pool.GetValue(((int64_t)position)), pool.GetValue((((int64_t)width) + ((int64_t)position))));
                  tl::ascend::pv_pair_times(_pair_temp_1[0], w_norm[0], w_norm[0]);
                  tl::ascend::pv_pair_divide(w_factor[0], _pair_temp_0[0], _pair_temp_1[0]);
                  AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                  tl::ascend::pv_pair_scale(pool[0], pool[0], pool[0], (width * 6), (width * 7), first, (width + first), w_factor.GetValue(0), w_factor.GetValue(1), count);
                  tl::ascend::pv_pair_add(pool[0], pool[0], pool[0], (width * 4), (width * 5), (width * 4), (width * 5), (width * 6), (width * 7), count);
                }
              }
              tl::ascend::pv_scale(pool[(width * 4)], pool[(width * 4)], pool[(width * 4)], (1.000000e+00f / power), count);
              tl::ascend::pv_scale(pool[(width * 5)], pool[(width * 5)], pool[(width * 5)], (1.000000e+00f / power), count);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_add(pool[(width * 6)], pool[(width * 4)], pool[(width * 5)], count);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              for (int32_t col = 0; col < m; ++col) {
                float output_value = pool.GetValue(((((int64_t)width) * (int64_t)6) + ((int64_t)col)));
                float bound = (error + (5.960465e-08f * max(output_value, (output_value * -1.000000e+00f))));
                float budget = ((1.000000e-04f + (1.000000e-04f * max(output_value, (output_value * -1.000000e+00f)))) * 9.999000e-01f);
                if (budget < bound) {
                  valid = 0;
                }
              }
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
              AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
              tl::ascend::pv_store_rows(X[(((job * m) * n) + (output_row_1 * m))], pool[(width * 6)], X[(((job * m) * n) + (output_row_1 * m))], 1, m, 0, 0);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            }
            if (valid != 0) {
              accepted.SetValue(0, 1);
            }
          }
        }
      }
      if (accepted.GetValue(0) == 0) {
        tl::ascend::pv_fill(pool[(width * 6)], pool[(width * 6)], pool[(width * 6)], 0.000000e+00f, width);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_div(pool[(width * 6)], pool[(width * 6)], pool[(width * 6)], width);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t output_row_2 = 0; output_row_2 < n; ++output_row_2) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
          tl::ascend::pv_store_rows(X[(((job * m) * n) + (output_row_2 * m))], pool[(width * 6)], X[(((job * m) * n) + (output_row_2 * m))], 1, m, 0, 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
      }
    }
    module_status = static_cast<int32_t>(((float)accepted.GetValue(0)));
  }
  pipe.Destroy();
}

struct PvModuleBuffers { AscendC::GlobalTensor<float> inputTensor,outputTensor,workspaceTensor; };
}
#define pv_module_bind(d,a,b) PvModuleBuffers pvModuleBuffers{d,a,b}
#define pv_module_accurate(d,a,b,...) PvDslAccurate((GM_ADDR)pvModuleBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.workspaceTensor.GetPhyAddr(),(GM_ADDR)d.GetPhyAddr(),__VA_ARGS__,fftsAddr)
#define pv_module_rectangular(d,a,b,...) PvDslRectangular((GM_ADDR)pvModuleBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.workspaceTensor.GetPhyAddr(),(GM_ADDR)d.GetPhyAddr(),__VA_ARGS__,fftsAddr)
#define pv_module_resident_accurate(d,a,b,status,...) PvDslResidentAccurate((GM_ADDR)pvModuleBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.workspaceTensor.GetPhyAddr(),status,__VA_ARGS__,fftsAddr)
#define pv_module_resident_rectangular(d,a,b,status,...) PvDslResidentRectangular((GM_ADDR)pvModuleBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.workspaceTensor.GetPhyAddr(),status,__VA_ARGS__,fftsAddr)

#define CATLASS_DEVICE __aicore__ inline
namespace tl::ascend {
template <typename T1, typename T2, uint32_t len>
CATLASS_DEVICE void copy_ub_to_ub(LocalTensor<T1> dstTensor,
                                  LocalTensor<T2> srcTensor) {
  if constexpr (std::is_same_v<T1, T2>) {
    AscendC::DataCopy(dstTensor, srcTensor, len);
  } else {
    if constexpr ((std::is_same_v<T1, float> && std::is_same_v<T2, half>) ||
                  (std::is_same_v<T1, float> && std::is_same_v<T2, int16_t>) ||
                  (std::is_same_v<T1, half> && std::is_same_v<T2, int8_t>) ||
                  (std::is_same_v<T1, int16_t> &&
                   std::is_same_v<T2, int32_t>)) {
      AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_NONE, len);
    } else {
      AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_RINT, len);
    }
  }
}

}
#undef CATLASS_DEVICE

namespace tl::ascend {
__aicore__ inline void lv_abs_nomask(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>){AscendC::Abs<float,false>(d,a,uint64_t(64),1,AscendC::UnaryRepeatParams{1,1,8,8});}
__aicore__ inline void lv_scale_nomask(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,float scale){AscendC::Muls<float,false>(d,a,scale,uint64_t(64),1,AscendC::UnaryRepeatParams{1,1,8,8});}
__aicore__ inline void lv_mad_nomask(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int rows,int pitch){AscendC::MulAddDst<float,float,false>(d,a,b,uint64_t(64),uint8_t(rows),AscendC::BinaryRepeatParams{1,1,0,uint8_t(pitch),0,1});}
}

namespace tl::ascend {
__aicore__ inline void ns_sum(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int n){AscendC::SetMaskCount();AscendC::SetVectorMask<float>(0,uint64_t(n));vcadd((__ubuf__ float*)d.GetPhyAddr(),(__ubuf__ float*)a.GetPhyAddr(),1,1,1,8,1);AscendC::SetMaskNorm();AscendC::ResetMask();}
__aicore__ inline void ns_acc(AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,float &value){value=AscendC::GetReduceRepeatSumSpr<float>();}
__aicore__ inline void sr_max(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int mask,int repeats){AscendC::WholeReduceMax<float>(d,a,mask,repeats,1,1,8,AscendC::ReduceOrder::ORDER_ONLY_VALUE);}
}


namespace tl::ascend {
__aicore__ inline void lv_index_iota(AscendC::LocalTensor<uint32_t>d,AscendC::LocalTensor<uint32_t>,AscendC::LocalTensor<uint32_t>,int n){AscendC::CreateVecIndex(d.ReinterpretCast<int32_t>(),int32_t(0),n);}
__aicore__ inline void lv_index_scale(AscendC::LocalTensor<uint32_t>d,AscendC::LocalTensor<uint32_t>a,AscendC::LocalTensor<uint32_t>,int step,int n){AscendC::Muls(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),int32_t(step),n);}
}

namespace tl::ascend {
__aicore__ inline void lv_square_packed_rows(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int width,int rows,int pitch){AscendC::Mul(d,a,a,uint64_t(width),uint8_t(rows),AscendC::BinaryRepeatParams{1,1,1,uint8_t(width/8),uint8_t(pitch/8),uint8_t(pitch/8)});}
}

namespace tl::ascend {
__aicore__ inline void pf_shift(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int bias,int n){AscendC::Adds(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),int32_t(bias),n);}
__aicore__ inline void pf_indices(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int width,int rows,int pitch){AscendC::Add(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),uint64_t(width),uint8_t(rows),AscendC::BinaryRepeatParams{1,0,1,uint8_t(pitch/8),1,0});}
}

namespace tl::ascend {
__aicore__ inline void li_abs_packed(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int width,int rows,int pitch){AscendC::Abs(d,a,uint64_t(width),uint8_t(rows),AscendC::UnaryRepeatParams{1,1,uint8_t(width/8),uint8_t(pitch/8)});}
__aicore__ inline void li_scale_rows(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,float scale,int width,int rows,int pitch){AscendC::Muls(d,a,scale,uint64_t(width),uint8_t(rows),AscendC::UnaryRepeatParams{1,1,uint8_t(pitch/8),uint8_t(pitch/8)});}
}
