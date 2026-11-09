#include "kernel_operator.h"
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"
using namespace AscendC;
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

static __aicore__ inline float PivotProduct(LocalTensor<float> diag,LocalTensor<float> col,LocalTensor<float> temp,LocalTensor<float> signs,LocalTensor<float> weights,LocalTensor<float> sc,uint32_t r) {
    auto mantissa=signs;
    Duplicate(col.ReinterpretCast<uint32_t>(),uint32_t(0x7fffffff),r);PipeBarrier<PIPE_V>();
    And(temp.ReinterpretCast<uint16_t>(),diag.ReinterpretCast<uint16_t>(),col.ReinterpretCast<uint16_t>(),2*r);
    Duplicate(col.ReinterpretCast<uint32_t>(),uint32_t(0x807fffff),r);PipeBarrier<PIPE_V>();
    ShiftRight(temp.ReinterpretCast<uint32_t>(),temp.ReinterpretCast<uint32_t>(),uint32_t(23),r);
    And(mantissa.ReinterpretCast<uint16_t>(),diag.ReinterpretCast<uint16_t>(),col.ReinterpretCast<uint16_t>(),2*r);PipeBarrier<PIPE_V>();
    Adds(mantissa.ReinterpretCast<int32_t>(),mantissa.ReinterpretCast<int32_t>(),int32_t(0x3f800000),r);
    Cast(temp,temp.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,r);PipeBarrier<PIPE_V>();
    for(uint32_t c=0;c<r;c+=64)WholeReduceSum(sc[c/8],temp[c],uint64_t(r-c<64?r-c:64),1,1,1,8);PipeBarrier<PIPE_V>();
    Brcb(weights,mantissa,r/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
    for(uint32_t step=1;step<r && step<64;step*=2){
        Mul(weights,weights,weights[step*8],uint64_t(8),uint8_t(r/(2*step)),BinaryRepeatParams{1,1,1,uint8_t(2*step),uint8_t(2*step),uint8_t(2*step)});PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    int exponent=-int32_t(127*r);for(uint32_t c=0;c<r;c+=64)exponent+=int32_t(sc.GetValue(c/8));float value=1.0f;
    union {float f;uint32_t u;} bits,scale;
    for(uint32_t block=0;block<r;block+=64){
        bits.f=weights.GetValue(block*8);int e=int((bits.u>>23)&255)-127;
        scale.u=uint32_t(127-e)<<23;value*=bits.f*scale.f;exponent+=e;
    }
    bits.f=value;int e=int((bits.u>>23)&255)-127;scale.u=uint32_t(127-e)<<23;value*=scale.f;exponent+=e;
    if(exponent>127){bits.u=(bits.u&0x80000000u)|0x7f800000u;value=bits.f;}
    else if(exponent<-149)value=0.0f;
    else{scale.u=exponent>=-126?uint32_t(exponent+127)<<23:1u<<uint32_t(exponent+149);value*=scale.f;}
    return value;
}

// Rare path, defined after DetKernel: recompute one matrix with partial pivoting inside this call.
static __aicore__ __attribute__((noinline)) void PivotedFallback(GM_ADDR a,GM_ADDR out,GM_ADDR workspace,GM_ADDR tiling,uint32_t batch);
// Unpivoted elimination loses accuracy when a pivot is small. Above order 16 its result is kept only when it
// overflowed, where nothing but the sign is left to compare; a finite, zero or NaN result is recomputed with
// partial pivoting. Up to order 16 a zero, NaN or infinite result is recomputed.
static __aicore__ inline bool DetAbnormal(float value,bool finiteExpected){
    union{float f;uint32_t u;} bits;bits.f=value;const uint32_t magnitude=bits.u&0x7fffffffu;
    return finiteExpected?(magnitude==0u || magnitude>=0x7f800000u):magnitude!=0x7f800000u;
}
static __aicore__ inline float DetMagnitude(float value){union{float f;uint32_t u;} bits;bits.f=value;bits.u&=0x7fffffffu;return bits.f;}
template<uint32_t P,bool Guard=false> static __aicore__ inline float TerminalDet(LocalTensor<float> mat,uint32_t q,uint32_t count,float *ratio=nullptr){
    if(count==1)return mat.GetValue(q*(P+1));
    if(count==2)return mat.GetValue(q*(P+1))*mat.GetValue((q+1)*(P+1))-mat.GetValue(q*P+q+1)*mat.GetValue((q+1)*P+q);
    if(count==3){
        float a=mat.GetValue(q*P+q),b=mat.GetValue(q*P+q+1),c=mat.GetValue(q*P+q+2);
        float d=mat.GetValue((q+1)*P+q),e=mat.GetValue((q+1)*P+q+1),f=mat.GetValue((q+1)*P+q+2);
        float g=mat.GetValue((q+2)*P+q),h=mat.GetValue((q+2)*P+q+1),i=mat.GetValue((q+2)*P+q+2);
        return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    }
    float a00=mat.GetValue((q+0)*P+q+0),a01=mat.GetValue((q+0)*P+q+1),a02=mat.GetValue((q+0)*P+q+2),a03=mat.GetValue((q+0)*P+q+3);
    float a10=mat.GetValue((q+1)*P+q+0),a11=mat.GetValue((q+1)*P+q+1),a12=mat.GetValue((q+1)*P+q+2),a13=mat.GetValue((q+1)*P+q+3);
    float a20=mat.GetValue((q+2)*P+q+0),a21=mat.GetValue((q+2)*P+q+1),a22=mat.GetValue((q+2)*P+q+2),a23=mat.GetValue((q+2)*P+q+3);
    float a30=mat.GetValue((q+3)*P+q+0),a31=mat.GetValue((q+3)*P+q+1),a32=mat.GetValue((q+3)*P+q+2),a33=mat.GetValue((q+3)*P+q+3);
    float s0=a00*a11-a10*a01,s1=a00*a12-a10*a02,s2=a00*a13-a10*a03;
    float s3=a01*a12-a11*a02,s4=a01*a13-a11*a03,s5=a02*a13-a12*a03;
    float c5=a22*a33-a32*a23,c4=a21*a33-a31*a23,c3=a21*a32-a31*a22;
    float c2=a20*a33-a30*a23,c1=a20*a32-a30*a22,c0=a20*a31-a30*a21;
    const float result=s0*c5-s1*c4+s2*c3+s3*c2-s4*c1+s5*c0;
    if constexpr(Guard){
        float bs0=DetMagnitude(a00*a11)+DetMagnitude(a10*a01),bs1=DetMagnitude(a00*a12)+DetMagnitude(a10*a02),bs2=DetMagnitude(a00*a13)+DetMagnitude(a10*a03);
        float bs3=DetMagnitude(a01*a12)+DetMagnitude(a11*a02),bs4=DetMagnitude(a01*a13)+DetMagnitude(a11*a03),bs5=DetMagnitude(a02*a13)+DetMagnitude(a12*a03);
        float bc5=DetMagnitude(a22*a33)+DetMagnitude(a32*a23),bc4=DetMagnitude(a21*a33)+DetMagnitude(a31*a23),bc3=DetMagnitude(a21*a32)+DetMagnitude(a31*a22);
        float bc2=DetMagnitude(a20*a33)+DetMagnitude(a30*a23),bc1=DetMagnitude(a20*a32)+DetMagnitude(a30*a22),bc0=DetMagnitude(a20*a31)+DetMagnitude(a30*a21);
        float scale=bs0*bc5+bs1*bc4+bs2*bc3+bs3*bc2+bs4*bc1+bs5*bc0;
        *ratio=(scale>0.0f && DetMagnitude(result)>1.0e-4f*scale)?1.0f:0.0f;
    }
    return result;
}

template<uint32_t P> static __aicore__ inline float TerminalSix(LocalTensor<float> mat,uint32_t n){
        const uint32_t q=n-6,base=q*(P+1);float product=1.0f;
        float a00=mat.GetValue(base+0*P+0),a01=mat.GetValue(base+0*P+1),a02=mat.GetValue(base+0*P+2),a03=mat.GetValue(base+0*P+3),a04=mat.GetValue(base+0*P+4),a05=mat.GetValue(base+0*P+5);
        float a10=mat.GetValue(base+1*P+0),a11=mat.GetValue(base+1*P+1),a12=mat.GetValue(base+1*P+2),a13=mat.GetValue(base+1*P+3),a14=mat.GetValue(base+1*P+4),a15=mat.GetValue(base+1*P+5);
        float a20=mat.GetValue(base+2*P+0),a21=mat.GetValue(base+2*P+1),a22=mat.GetValue(base+2*P+2),a23=mat.GetValue(base+2*P+3),a24=mat.GetValue(base+2*P+4),a25=mat.GetValue(base+2*P+5);
        float a30=mat.GetValue(base+3*P+0),a31=mat.GetValue(base+3*P+1),a32=mat.GetValue(base+3*P+2),a33=mat.GetValue(base+3*P+3),a34=mat.GetValue(base+3*P+4),a35=mat.GetValue(base+3*P+5);
        float a40=mat.GetValue(base+4*P+0),a41=mat.GetValue(base+4*P+1),a42=mat.GetValue(base+4*P+2),a43=mat.GetValue(base+4*P+3),a44=mat.GetValue(base+4*P+4),a45=mat.GetValue(base+4*P+5);
        float a50=mat.GetValue(base+5*P+0),a51=mat.GetValue(base+5*P+1),a52=mat.GetValue(base+5*P+2),a53=mat.GetValue(base+5*P+3),a54=mat.GetValue(base+5*P+4),a55=mat.GetValue(base+5*P+5);
        product*=a00;
        float r0=-1.0f/a00;
        float u01=a01*r0,u02=a02*r0,u03=a03*r0,u04=a04*r0,u05=a05*r0;
        a11+=a10*u01;a12+=a10*u02;a13+=a10*u03;a14+=a10*u04;a15+=a10*u05;
        a21+=a20*u01;a22+=a20*u02;a23+=a20*u03;a24+=a20*u04;a25+=a20*u05;
        a31+=a30*u01;a32+=a30*u02;a33+=a30*u03;a34+=a30*u04;a35+=a30*u05;
        a41+=a40*u01;a42+=a40*u02;a43+=a40*u03;a44+=a40*u04;a45+=a40*u05;
        a51+=a50*u01;a52+=a50*u02;a53+=a50*u03;a54+=a50*u04;a55+=a50*u05;
        product*=a11;
        float r1=-1.0f/a11;
        float u12=a12*r1,u13=a13*r1,u14=a14*r1,u15=a15*r1;
        a22+=a21*u12;a23+=a21*u13;a24+=a21*u14;a25+=a21*u15;
        a32+=a31*u12;a33+=a31*u13;a34+=a31*u14;a35+=a31*u15;
        a42+=a41*u12;a43+=a41*u13;a44+=a41*u14;a45+=a41*u15;
        a52+=a51*u12;a53+=a51*u13;a54+=a51*u14;a55+=a51*u15;
        product*=a22;
        float r2=-1.0f/a22;
        float u23=a23*r2,u24=a24*r2,u25=a25*r2;
        a33+=a32*u23;a34+=a32*u24;a35+=a32*u25;
        a43+=a42*u23;a44+=a42*u24;a45+=a42*u25;
        a53+=a52*u23;a54+=a52*u24;a55+=a52*u25;
        product*=a33;
        float r3=-1.0f/a33;
        float u34=a34*r3,u35=a35*r3;
        a44+=a43*u34;a45+=a43*u35;
        a54+=a53*u34;a55+=a53*u35;
        product*=a44;
        float r4=-1.0f/a44;
        float u45=a45*r4;
        a55+=a54*u45;
        product*=a55;
        return product;
    }

// The determinant needs the pivots, not the triangular factors. Keep every
// elimination dependency in the vector pipe while the active matrix fits UB.
template<uint32_t N> class ReplicatedTiny {
public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR y,const LinalgDetTilingData &t,GM_ADDR tiling){
        constexpr uint32_t C=16,P=136;
        LocalTensor<float> arena(TPosition::VECCALC,0,C*P+768);auto mat=arena,raw=mat[C*P],pivot=raw[256],sc=pivot[128];
        GlobalTensor<float> input,output;input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);const uint32_t n=N?N:t.n;
        for(uint32_t batch=GetBlockIdx();batch<(N?GetBlockIdx()+1:t.batches);batch+=(N?1:t.blocks)){
            DataCopyPad(raw,input[(uint64_t)batch*n*n],DataCopyExtParams{uint16_t(n),n*4,0,uint32_t((16-(n+7)/8*8)/8),0},DataCopyPadExtParams<float>{true,0,uint8_t((8-n%8)%8),0.0f});
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            for(uint32_t r=0;r<n;++r)Brcb(mat[r*P],raw[r*16],2,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j+1<n;++j){
                Div(pivot,mat[j*P],mat[j*(P+8)],uint64_t(64),2,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
                Muls(pivot,pivot,-1.0f,128);PipeBarrier<PIPE_V>();
                for(uint32_t col=(j+1)/8;col<2;++col){
                    const uint32_t skip=j+1>col*8?j+1-col*8:0;
                    uint64_t mask[2]={skip<8?(~uint64_t(0)<<(skip*8)):0,0};
                    MulAddDst(mat[(j+1)*P+col*64],pivot[col*64],mat[(j+1)*P+j*8],mask,uint8_t(n-j-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                }
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);float value=1.0f;
            for(uint32_t j=0;j<n;++j)value*=mat.GetValue(j*(P+8));
            sc.SetValue(0,value);SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);DataCopyPad(output[batch],sc,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
};

// All pivot diagonals are complete at the existing final V_S boundary.
// A balanced Scalar product uses the same values without new Vector work.
template<uint32_t P,uint32_t First,uint32_t Count>
static __aicore__ inline float TinyScalarPivotProduct(LocalTensor<float> mat){
    if constexpr(Count==0)return 1.0f;
    else if constexpr(Count==1)return mat.GetValue(First*(P+1));
    else return TinyScalarPivotProduct<P,First,Count/2>(mat)*
        TinyScalarPivotProduct<P,First+Count/2,Count-Count/2>(mat);
}


// Manual synchronization island: same instructions as the float,false APIs.
static __aicore__ inline void DetNativeMuls(LocalTensor<float> dst,LocalTensor<float> src,
    float scalar,uint64_t mask,uint8_t repeat,const UnaryRepeatParams &p){
    vmuls((__ubuf__ float*)dst.GetPhyAddr(),(__ubuf__ float*)src.GetPhyAddr(),scalar,
        repeat,uint16_t(p.dstBlkStride),uint16_t(p.srcBlkStride),uint16_t(p.dstRepStride),uint16_t(p.srcRepStride));
}
static __aicore__ inline void DetNativeMla(LocalTensor<float> dst,LocalTensor<float> a,
    LocalTensor<float> b,uint64_t mask,uint8_t repeat,const BinaryRepeatParams &p){
    vmla((__ubuf__ float*)dst.GetPhyAddr(),(__ubuf__ float*)a.GetPhyAddr(),(__ubuf__ float*)b.GetPhyAddr(),
        repeat,p.dstBlkStride,p.src0BlkStride,p.src1BlkStride,p.dstRepStride,p.src0RepStride,p.src1RepStride);
}


static __aicore__ inline void DetNativeMulsCount(LocalTensor<float> dst,LocalTensor<float> src,float scalar,uint32_t count){
    SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,count);
    vmuls((__ubuf__ float*)dst.GetPhyAddr(),(__ubuf__ float*)src.GetPhyAddr(),scalar,1,1,1,8,8);
    ResetMask();SetMaskNorm();
}
static __aicore__ inline void DetNativeMlaMasked(LocalTensor<float> dst,LocalTensor<float> a,
    LocalTensor<float> b,uint64_t count,uint8_t repeat,const BinaryRepeatParams &p){
    SetVectorMask<float>(0,count==64?~uint64_t(0):(uint64_t(1)<<count)-1);
    DetNativeMla(dst,a,b,count,repeat,p);
}

template<uint32_t N> class TinyResident {
    static constexpr uint32_t C=16,P=(N>8 && N%8)?16:24;
    static __aicore__ inline float TinyBlockPairs(LocalTensor<float> mat,LocalTensor<float> col,
        LocalTensor<float> weights,LocalTensor<uint32_t> ix,uint32_t n){
        #pragma clang fp contract(on)
        LocalTensor<uint32_t> pairIx(TPosition::VECCALC,24576,32);
        auto p0=mat[416],p1=mat[448];float product=1.0f;uint32_t j=0;
        SetMaskNorm();SetVectorMask<float>(0,uint64_t(0xffffu));
        #pragma unroll
        for(;j+5<n;j+=2){
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
            float a00=mat.GetValue(j*(P+1)),a01=mat.GetValue(j*P+j+1);
            float a10=mat.GetValue((j+1)*P+j),a11=mat.GetValue((j+1)*(P+1));
            float delta=a00*a11-a10*a01,inverse=1.0f/delta;product*=delta;
            float c00=-a11*inverse,c01=a01*inverse,c10=a10*inverse,c11=-a00*inverse;
            SetVectorMask<float>(0,uint64_t(0xffffffffu));
            vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)pairIx.GetPhyAddr(),uint32_t((uint64_t)mat[j].GetPhyAddr()),8,1);
            SetVectorMask<float>(0,uint64_t(0xffffu));
            DetNativeMuls(p0,mat[j*P],c00,16,1,UnaryRepeatParams{1,1,8,8});
            DetNativeMuls(p1,mat[(j+1)*P],c11,16,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SetVectorMask<float>(0,~uint64_t(0));
            vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,4);
            SetVectorMask<float>(0,uint64_t(0xffffu));
            vaxpy((__ubuf__ float*)p0.GetPhyAddr(),(__ubuf__ float*)mat[(j+1)*P].GetPhyAddr(),c01,1,1,1,8,8);
            vaxpy((__ubuf__ float*)p1.GetPhyAddr(),(__ubuf__ float*)mat[j*P].GetPhyAddr(),c10,1,1,1,8,8);PipeBarrier<PIPE_V>();
            DetNativeMla(mat[(j+2)*P],p0,weights[(j+2)*8],16,uint8_t(n-j-2),BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
            DetNativeMla(mat[(j+2)*P],p1,weights[128+(j+2)*8],16,uint8_t(n-j-2),BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
        }
        if(j+4<n){
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
            float d=mat.GetValue(j*(P+1));product*=d;
            vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[j].GetPhyAddr()),8,1);
            DetNativeMuls(p0,mat[j*P],-1.0f/d,16,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SetVectorMask<float>(0,~uint64_t(0));
            vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,2);
            SetVectorMask<float>(0,uint64_t(0xffffu));PipeBarrier<PIPE_V>();
            DetNativeMla(mat[(j+1)*P],p0,weights[(j+1)*8],16,uint8_t(n-j-1),BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
        }
        return product;
    }

    static __aicore__ inline void Prepare(LocalTensor<uint32_t> ix,LocalTensor<uint32_t> packing,uint32_t n,bool packed){
        LocalTensor<int32_t> columns(TPosition::VECCALC,16384,32),rows(TPosition::VECCALC,16512,16),spread(TPosition::VECCALC,16640,128);
        for(uint32_t k=0;k<8;++k)columns.SetValue(k,k*4);
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        Adds(columns[8],columns,int32_t(32),8);PipeBarrier<PIPE_V>();
        Muls(ix.ReinterpretCast<int32_t>(),columns,int32_t(P),C);PipeBarrier<PIPE_V>();
        if(packed){
            uint64_t paddingMask[2]={(uint64_t(0xffffffffu)<<n)&uint64_t(0xffffffffu),0};
            Duplicate(columns,int32_t(0),paddingMask,1,1,4);PipeBarrier<PIPE_V>();
            Muls(rows,columns,int32_t(n),C);PipeBarrier<PIPE_V>();
            Brcb(spread,rows,C/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Add(packing.ReinterpretCast<int32_t>(),columns,spread,uint64_t(P),uint8_t(C),BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
        }
    }
public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR y,const LinalgDetTilingData &t,GM_ADDR workspace,GM_ADDR tiling){
        constexpr bool DirectInput=N && N%8==0;
        constexpr bool Expanded=N>4;
        LocalTensor<float> arena(TPosition::VECCALC,0,1408);
        LocalTensor<float> raw(TPosition::VECCALC,8192,256);
        LocalTensor<uint32_t> packing(TPosition::VECCALC,12288,C*P);
        auto mat=arena,col=mat[384],row=mat[416],weights=mat[512],sc=mat[Expanded?768:704];auto ix=sc[64].ReinterpretCast<uint32_t>(),expanded=ix[C];
        bool layoutReady=false;
        GlobalTensor<float> input,output;input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);uint32_t n=N?N:t.n;
        for(uint32_t batch=GetBlockIdx();batch<(N?GetBlockIdx()+1:t.batches);batch+=(N?1:t.blocks)){
            float value=1.0f,pairValue=1.0f;
            if(n>=3){

                DataCopyPad(raw,input[(uint64_t)batch*n*n],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0f});
                if(!layoutReady){
                    Prepare(ix,packing,n,!DirectInput);
                    if constexpr(N>4){LocalTensor<uint32_t> pairIx(TPosition::VECCALC,24576,32);DataCopy(pairIx,ix,C);Adds(pairIx[C].ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(4),C);PipeBarrier<PIPE_V>();}
                    layoutReady=true;
                }
                SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
                if constexpr(DirectInput)DataCopy(mat,raw,DataCopyParams{uint16_t(n),uint16_t(n/8),0,uint16_t((P-n)/8)});
                else Gather(mat,raw,packing,0,C*P);
                PipeBarrier<PIPE_V>();
            }else{
                DataCopyPad(mat,input[(uint64_t)batch*n*n],DataCopyExtParams{uint16_t(n),n*4,0,uint32_t((P-(n+7)/8*8)/8),0},DataCopyPadExtParams<float>{true,0,uint8_t((8-n%8)%8),0.0f});
                if(n>2 && !layoutReady){Prepare(ix,packing,n,false);layoutReady=true;}
                SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            }
            if(n>4){SetMaskNorm();SetVectorMask<float>(0,uint64_t(0xffffu));}
            if constexpr(N>4){value=TinyBlockPairs(mat,col,weights,ix,n);}else
            {
            // Keep the compact matrix and every pivot dependency in Vector.
            // Earlier pivot rows are never modified again, so their diagonals
            // remain available for one final Scalar product.
            #pragma unroll
            for(uint32_t j=0;j+4<n;++j){
                Muls<float,false>(col,mat[j*P],-1.0f,uint64_t(C),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                if constexpr(Expanded){
                    SetVectorMask<float>(0,~uint64_t(0));
                    vgather((__ubuf__ uint32_t*)weights[j*8].GetPhyAddr(),
                        (__ubuf__ uint32_t*)expanded[j*8].GetPhyAddr(),
                        uint32_t((uint64_t)mat[j].GetPhyAddr()),8,uint8_t((n-j+7)/8));
                    SetVectorMask<float>(0,uint64_t(0xffffu));
                }else{
                    vgather((__ubuf__ uint32_t*)row.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),
                        uint32_t((uint64_t)mat[j].GetPhyAddr()),8,1);
                    PipeBarrier<PIPE_V>();
                    SetVectorMask<float>(0,~uint64_t(0));
                    vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)row.GetPhyAddr(),1,8,C/8);
                    SetVectorMask<float>(0,uint64_t(0xffffu));
                }
                PipeBarrier<PIPE_V>();
                Div<float,false>(col,col,weights[j*8],uint64_t(C),uint8_t(1),BinaryRepeatParams{1,1,0,8,8,0});
                PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(mat[(j+1)*P],col,weights[(j+1)*8],
                    uint64_t(C),uint8_t(n-j-1),BinaryRepeatParams{1,1,0,P/8,0,1});
                PipeBarrier<PIPE_V>();
            }
            }
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
            if constexpr(N<=4){
            if constexpr(N){
                value=TinyScalarPivotProduct<P,0,(N>4?N-4:0)>(mat);
            }else{
                #pragma unroll
                for(uint32_t j=0;j+4<n;++j)value*=mat.GetValue(j*(P+1));
            }
            }
            value*=TerminalDet<P>(mat,n<4?0:n-4,n<4?n:4);
            sc.SetValue(0,value);SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);DataCopyPad(output[batch],sc,DataCopyExtParams{1,4,0,0,0});const bool abnormal=DetAbnormal(value,true);PipeBarrier<PIPE_MTE3>();
            if(abnormal){PivotedFallback(a,y,workspace,tiling,batch);layoutReady=false;}
        }
    }
};

template<uint32_t P> static __aicore__ inline float ReverseTerminalSix(LocalTensor<float> mat){
        float product=1.0f;
        float a00=mat.GetValue(5*P+5),a01=mat.GetValue(5*P+4),a02=mat.GetValue(5*P+3),a03=mat.GetValue(5*P+2),a04=mat.GetValue(5*P+1),a05=mat.GetValue(5*P+0);
        float a10=mat.GetValue(4*P+5),a11=mat.GetValue(4*P+4),a12=mat.GetValue(4*P+3),a13=mat.GetValue(4*P+2),a14=mat.GetValue(4*P+1),a15=mat.GetValue(4*P+0);
        float a20=mat.GetValue(3*P+5),a21=mat.GetValue(3*P+4),a22=mat.GetValue(3*P+3),a23=mat.GetValue(3*P+2),a24=mat.GetValue(3*P+1),a25=mat.GetValue(3*P+0);
        float a30=mat.GetValue(2*P+5),a31=mat.GetValue(2*P+4),a32=mat.GetValue(2*P+3),a33=mat.GetValue(2*P+2),a34=mat.GetValue(2*P+1),a35=mat.GetValue(2*P+0);
        float a40=mat.GetValue(1*P+5),a41=mat.GetValue(1*P+4),a42=mat.GetValue(1*P+3),a43=mat.GetValue(1*P+2),a44=mat.GetValue(1*P+1),a45=mat.GetValue(1*P+0);
        float a50=mat.GetValue(0*P+5),a51=mat.GetValue(0*P+4),a52=mat.GetValue(0*P+3),a53=mat.GetValue(0*P+2),a54=mat.GetValue(0*P+1),a55=mat.GetValue(0*P+0);
        product*=a00;
        float r0=-1.0f/a00;
        float u01=a01*r0,u02=a02*r0,u03=a03*r0,u04=a04*r0,u05=a05*r0;
        a11+=a10*u01;a12+=a10*u02;a13+=a10*u03;a14+=a10*u04;a15+=a10*u05;
        a21+=a20*u01;a22+=a20*u02;a23+=a20*u03;a24+=a20*u04;a25+=a20*u05;
        a31+=a30*u01;a32+=a30*u02;a33+=a30*u03;a34+=a30*u04;a35+=a30*u05;
        a41+=a40*u01;a42+=a40*u02;a43+=a40*u03;a44+=a40*u04;a45+=a40*u05;
        a51+=a50*u01;a52+=a50*u02;a53+=a50*u03;a54+=a50*u04;a55+=a50*u05;
        product*=a11;
        float r1=-1.0f/a11;
        float u12=a12*r1,u13=a13*r1,u14=a14*r1,u15=a15*r1;
        a22+=a21*u12;a23+=a21*u13;a24+=a21*u14;a25+=a21*u15;
        a32+=a31*u12;a33+=a31*u13;a34+=a31*u14;a35+=a31*u15;
        a42+=a41*u12;a43+=a41*u13;a44+=a41*u14;a45+=a41*u15;
        a52+=a51*u12;a53+=a51*u13;a54+=a51*u14;a55+=a51*u15;
        product*=a22;
        float r2=-1.0f/a22;
        float u23=a23*r2,u24=a24*r2,u25=a25*r2;
        a33+=a32*u23;a34+=a32*u24;a35+=a32*u25;
        a43+=a42*u23;a44+=a42*u24;a45+=a42*u25;
        a53+=a52*u23;a54+=a52*u24;a55+=a52*u25;
        product*=a33;
        float r3=-1.0f/a33;
        float u34=a34*r3,u35=a35*r3;
        a44+=a43*u34;a45+=a43*u35;
        a54+=a53*u34;a55+=a53*u35;
        product*=a44;
        float r4=-1.0f/a44;
        float u45=a45*r4;
        a55+=a54*u45;
        product*=a55;
        return product;
    }

// Packed unaligned input, mirrored in both axes; logical pivot order is unchanged.
template<uint32_t N,uint32_t P> class ReverseResidentDet {
    static constexpr uint32_t C=64;
    template<uint32_t Width> static __aicore__ inline void Step(
        LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<float> pivot,
        LocalTensor<float> weights,LocalTensor<uint32_t> ix,uint32_t q,float inverse){
        vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),
            uint32_t((uint64_t)mat[q].GetPhyAddr()),8,1);
        Muls<float,false>(pivot,mat[q*P],inverse,uint64_t(Width),uint8_t(1),UnaryRepeatParams{1,1,8,8});
        if constexpr(Width==64){
            vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),P/8,P,8);
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(mat,pivot,mat[64],uint64_t(64),uint8_t(q),BinaryRepeatParams{1,1,0,P/8,0,P/8});
        }else{
            SetVectorMask<float>(0,~uint64_t(0));
            vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,4);
            SetVectorMask<float>(0,uint64_t(0xffffffffu));PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(mat,pivot,weights,uint64_t(32),uint8_t(q),BinaryRepeatParams{1,1,0,P/8,0,1});
        }
        PipeBarrier<PIPE_V>();
    }
    template<uint32_t W0,uint32_t W1> static __aicore__ inline void Pair(
        LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<float> pivot,
        LocalTensor<float> weights,LocalTensor<uint32_t> ix,uint32_t q,float &value){
        #pragma clang fp contract(on)
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        float d0=mat.GetValue(q*P+q),a01=mat.GetValue(q*P+q-1);
        float a10=mat.GetValue((q-1)*P+q),a11=mat.GetValue((q-1)*P+q-1);
        float inv0=-1.0f/d0;value*=d0;
        Step<W0>(mat,col,pivot,weights,ix,q,inv0);
        float d1Base=a11+a10*a01*inv0;
        float d1=d1Base,inv1=-1.0f/d1;value*=d1;
        if constexpr(W0!=W1)SetVectorMask<float>(0,uint64_t(0xffffffffu));
        Step<W1>(mat,col,pivot,weights,ix,q-1,inv1);
    }
    template<bool Batched> static __aicore__ inline void PairCoupled(
        LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<float> pivot,
        LocalTensor<float> weights,LocalTensor<uint32_t> ix,uint32_t q,float &value){
        #pragma clang fp contract(on)
        
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        if constexpr(Batched){
            LocalTensor<uint32_t> pairIx(TPosition::VECCALC,172032,64);
            SetVectorMask<float>(0,~uint64_t(0));
            vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)pairIx.GetPhyAddr(),uint32_t((uint64_t)mat[q-1].GetPhyAddr()),8,1);
            SetVectorMask<float>(0,uint64_t(0xffffffffu));
        }else{
            vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[q].GetPhyAddr()),8,1);
            vgather((__ubuf__ uint32_t*)col[32].GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[q-1].GetPhyAddr()),8,1);
        }

        float d0=mat.GetValue(q*P+q),a01=mat.GetValue(q*P+q-1);
        float a10=mat.GetValue((q-1)*P+q),a11=mat.GetValue((q-1)*P+q-1);

        float delta=d0*a11-a10*a01,reciprocal=1.0f/delta;
        float c00=-a11*reciprocal,c01=a01*reciprocal,c10=a10*reciprocal,c11=-d0*reciprocal;
        value*=delta;

        DetNativeMuls(pivot,mat[q*P],c00,uint64_t(32),1,UnaryRepeatParams{1,1,8,8});
        DetNativeMuls(pivot[P],mat[(q-1)*P],c11,uint64_t(32),1,UnaryRepeatParams{1,1,8,8});
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0,~uint64_t(0));
        vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,8);
        SetVectorMask<float>(0,uint64_t(0xffffffffu));
        vaxpy((__ubuf__ float*)pivot.GetPhyAddr(),(__ubuf__ float*)mat[(q-1)*P].GetPhyAddr(),c01,1,1,1,8,8);
        vaxpy((__ubuf__ float*)pivot[P].GetPhyAddr(),(__ubuf__ float*)mat[q*P].GetPhyAddr(),c10,1,1,1,8,8);
        PipeBarrier<PIPE_V>();
        DetNativeMla(mat,pivot,weights,uint64_t(32),uint8_t(q-1),BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
        DetNativeMla(mat,pivot[P],weights[256],uint64_t(32),uint8_t(q-1),BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
    }
    static __aicore__ inline void BlockWide(LocalTensor<float> mat,LocalTensor<float> pivot,uint32_t q,float &value){
        #pragma clang fp contract(on)
        LocalTensor<uint32_t> wideIx(TPosition::VECCALC,173056,128);
        LocalTensor<float> col(TPosition::VECCALC,174080,128);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)wideIx.GetPhyAddr(),uint32_t((uint64_t)mat[q-1].GetPhyAddr()),8,2);
        float d0=mat.GetValue(q*P+q),a01=mat.GetValue(q*P+q-1);
        float a10=mat.GetValue((q-1)*P+q),a11=mat.GetValue((q-1)*P+q-1);
        
        float delta=d0*a11-a10*a01,reciprocal=1.0f/delta;
        float c00=-a11*reciprocal,c01=a01*reciprocal,c10=a10*reciprocal,c11=-d0*reciprocal;
        value*=delta;

        DetNativeMuls(pivot,mat[q*P],c00,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});
        DetNativeMuls(pivot[P],mat[(q-1)*P],c11,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});
        PipeBarrier<PIPE_V>();
        vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),P/8,P,8);
        vbrcb((__ubuf__ uint32_t*)mat[72].GetPhyAddr(),(__ubuf__ uint32_t*)col[64].GetPhyAddr(),P/8,P,8);
        vaxpy((__ubuf__ float*)pivot.GetPhyAddr(),(__ubuf__ float*)mat[(q-1)*P].GetPhyAddr(),c01,1,1,1,8,8);
        vaxpy((__ubuf__ float*)pivot[P].GetPhyAddr(),(__ubuf__ float*)mat[q*P].GetPhyAddr(),c10,1,1,1,8,8);
        PipeBarrier<PIPE_V>();
        DetNativeMla(mat,pivot,mat[64],uint64_t(64),uint8_t(q-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});PipeBarrier<PIPE_V>();
        DetNativeMla(mat,pivot[P],mat[72],uint64_t(64),uint8_t(q-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});PipeBarrier<PIPE_V>();

    }

public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR y,const LinalgDetTilingData &t,GM_ADDR workspace,GM_ADDR tiling){
        LocalTensor<float> arena(TPosition::VECCALC,0,C*P+24*C+P+256);
        auto mat=arena,col=mat[C*P],weights=col[C],pivot=weights[8*C+64],diag=pivot[P],temp=diag[C],signs=temp[C],sc=signs[C];
        auto rows=signs.ReinterpretCast<int32_t>(),columns=temp.ReinterpretCast<int32_t>();
        auto ix=sc[128].ReinterpretCast<uint32_t>();
        LocalTensor<float> raw(TPosition::VECCALC,128*1024,4096);
        LocalTensor<uint32_t> packing(TPosition::VECCALC,144*1024,4096);
        GlobalTensor<float> input,output;input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);
        bool prepared=false;
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks){
            DataCopyPad(raw,input[uint64_t(batch)*N*N],DataCopyExtParams{1,N*N*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0f});
            if(!prepared){
                for(uint32_t k=0;k<8;++k)columns.SetValue(k,int32_t((N-1-k)*4));
                SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
                for(uint32_t k=8;k<C;k*=2){Adds(columns[k],columns,-int32_t(k*4),k);PipeBarrier<PIPE_V>();}
                uint64_t padding[2]={~((uint64_t(1)<<N)-1),0};
                Duplicate(columns,int32_t(0),padding,1,1,8);PipeBarrier<PIPE_V>();
                Muls(ix.ReinterpretCast<int32_t>(),columns,-int32_t(P),C);
                Muls(rows,columns,int32_t(N),C);PipeBarrier<PIPE_V>();
                Adds(ix.ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t((N-1)*P*4),C);
                Brcb(weights.ReinterpretCast<int32_t>(),rows,C/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                Add(packing.ReinterpretCast<int32_t>(),columns,weights.ReinterpretCast<int32_t>(),uint64_t(C),uint8_t(N),BinaryRepeatParams{1,1,0,8,0,1});
                PipeBarrier<PIPE_V>();prepared=true;
            }
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            Gather(mat,raw,packing,0,uint64_t(N),uint8_t(N),P/8);PipeBarrier<PIPE_V>();
            float value=1.0f;uint32_t q=N-1;
            SetMaskNorm();SetVectorMask<float>(0,~uint64_t(0));
if constexpr(P>=80){LocalTensor<uint32_t> wideIx(TPosition::VECCALC,173056,128);Adds(wideIx.ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(4),64);DataCopy(wideIx[64],ix,64);PipeBarrier<PIPE_V>();}
            #pragma unroll
            for(;q>=33;q-=2){if constexpr(P>=80)BlockWide(mat,pivot,q,value);else Pair<64,64>(mat,col,pivot,weights,ix,q,value);}
            if(q==32){Pair<64,32>(mat,col,pivot,weights,ix,q,value);q-=2;}
            LocalTensor<uint32_t> pairIx(TPosition::VECCALC,172032,64);
            Adds(pairIx.ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(4),32);
            DataCopy(pairIx[32],ix,32);PipeBarrier<PIPE_V>();
            SetVectorMask<float>(0,uint64_t(0xffffffffu));
            #pragma unroll
            for(;q>=7;q-=2)PairCoupled<true>(mat,col,pivot,weights,ix,q,value);
            if(q==6){
                SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                float diagonal=mat.GetValue(q*P+q),inverse=-1.0f/diagonal;value*=diagonal;
                Step<32>(mat,col,pivot,weights,ix,q,inverse);
            }
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
            value*=ReverseTerminalSix<P>(mat);
            sc.SetValue(0,value);SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
            DataCopyPad(output[batch],sc,DataCopyExtParams{1,4,0,0,0});const bool abnormal=DetAbnormal(value,false);PipeBarrier<PIPE_MTE3>();
            if(abnormal){PivotedFallback(a,y,workspace,tiling,batch);prepared=false;}
        }
    }
};


template<uint32_t P> static __aicore__ inline float BlockColumnTerminalSix(LocalTensor<float> mat,uint32_t n){
        const uint32_t base=(n-6)*P+n-6;float product=1.0f;
        float a00=mat.GetValue(base+0*P+0),a01=mat.GetValue(base+0*P+1),a02=mat.GetValue(base+0*P+2),a03=mat.GetValue(base+0*P+3),a04=mat.GetValue(base+0*P+4),a05=mat.GetValue(base+0*P+5);
        float a10=mat.GetValue(base+1*P+0),a11=mat.GetValue(base+1*P+1),a12=mat.GetValue(base+1*P+2),a13=mat.GetValue(base+1*P+3),a14=mat.GetValue(base+1*P+4),a15=mat.GetValue(base+1*P+5);
        float a20=mat.GetValue(base+2*P+0),a21=mat.GetValue(base+2*P+1),a22=mat.GetValue(base+2*P+2),a23=mat.GetValue(base+2*P+3),a24=mat.GetValue(base+2*P+4),a25=mat.GetValue(base+2*P+5);
        float a30=mat.GetValue(base+3*P+0),a31=mat.GetValue(base+3*P+1),a32=mat.GetValue(base+3*P+2),a33=mat.GetValue(base+3*P+3),a34=mat.GetValue(base+3*P+4),a35=mat.GetValue(base+3*P+5);
        float a40=mat.GetValue(base+4*P+0),a41=mat.GetValue(base+4*P+1),a42=mat.GetValue(base+4*P+2),a43=mat.GetValue(base+4*P+3),a44=mat.GetValue(base+4*P+4),a45=mat.GetValue(base+4*P+5);
        float a50=mat.GetValue(base+5*P+0),a51=mat.GetValue(base+5*P+1),a52=mat.GetValue(base+5*P+2),a53=mat.GetValue(base+5*P+3),a54=mat.GetValue(base+5*P+4),a55=mat.GetValue(base+5*P+5);
        product*=a00;
        float r0=-1.0f/a00;
        float u01=a01*r0,u02=a02*r0,u03=a03*r0,u04=a04*r0,u05=a05*r0;
        a11+=a10*u01;a12+=a10*u02;a13+=a10*u03;a14+=a10*u04;a15+=a10*u05;
        a21+=a20*u01;a22+=a20*u02;a23+=a20*u03;a24+=a20*u04;a25+=a20*u05;
        a31+=a30*u01;a32+=a30*u02;a33+=a30*u03;a34+=a30*u04;a35+=a30*u05;
        a41+=a40*u01;a42+=a40*u02;a43+=a40*u03;a44+=a40*u04;a45+=a40*u05;
        a51+=a50*u01;a52+=a50*u02;a53+=a50*u03;a54+=a50*u04;a55+=a50*u05;
        product*=a11;
        float r1=-1.0f/a11;
        float u12=a12*r1,u13=a13*r1,u14=a14*r1,u15=a15*r1;
        a22+=a21*u12;a23+=a21*u13;a24+=a21*u14;a25+=a21*u15;
        a32+=a31*u12;a33+=a31*u13;a34+=a31*u14;a35+=a31*u15;
        a42+=a41*u12;a43+=a41*u13;a44+=a41*u14;a45+=a41*u15;
        a52+=a51*u12;a53+=a51*u13;a54+=a51*u14;a55+=a51*u15;
        product*=a22;
        float r2=-1.0f/a22;
        float u23=a23*r2,u24=a24*r2,u25=a25*r2;
        a33+=a32*u23;a34+=a32*u24;a35+=a32*u25;
        a43+=a42*u23;a44+=a42*u24;a45+=a42*u25;
        a53+=a52*u23;a54+=a52*u24;a55+=a52*u25;
        product*=a33;
        float r3=-1.0f/a33;
        float u34=a34*r3,u35=a35*r3;
        a44+=a43*u34;a45+=a43*u35;
        a54+=a53*u34;a55+=a53*u35;
        product*=a44;
        float r4=-1.0f/a44;
        float u45=a45*r4;
        a55+=a54*u45;
        product*=a55;
        return product;
    }

// Reverse eight-column blocks during input loading; keep rows and logical pivot order.
template<uint32_t N,uint32_t P,bool LayoutOnly=false> class BlockColumnResidentDet {
    static constexpr uint32_t C=64;
    template<uint32_t Width> static __aicore__ inline void Step(LocalTensor<float> mat,
        LocalTensor<float> col,LocalTensor<float> pivot,LocalTensor<float> weights,
        LocalTensor<uint32_t> ix,uint32_t j,uint32_t q,float inverse){
        constexpr uint32_t firstRow=Width==64?0:N-32;
        vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),
            uint32_t((uint64_t)mat[firstRow*P+q].GetPhyAddr()),8,1);
        Muls<float,false>(pivot,mat[j*P+(Width==64?0:N-32)],inverse,uint64_t(Width),uint8_t(1),UnaryRepeatParams{1,1,8,8});
        if constexpr(Width==64){
            vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),P/8,P,8);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(mat[(j+1)*P],pivot,mat[(j+1)*P+64],uint64_t(64),uint8_t(N-j-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
        }else{
            SetVectorMask<float>(0,~uint64_t(0));
            vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,4);
            SetVectorMask<float>(0,uint64_t(0xffffffffu));PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(mat[(j+1)*P+N-32],pivot,weights[(j+1-firstRow)*8],uint64_t(32),uint8_t(N-j-1),BinaryRepeatParams{1,1,0,P/8,0,1});
        }
        PipeBarrier<PIPE_V>();
    }
    template<uint32_t Width> static __aicore__ inline void Pair(LocalTensor<float> mat,
        LocalTensor<float> col,LocalTensor<float> pivot,LocalTensor<float> weights,
        LocalTensor<uint32_t> ix,uint32_t j,float &value){
        #pragma clang fp contract(on)
        const uint32_t q=j;
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        float d0=mat.GetValue(j*P+q),a01=mat.GetValue(j*P+q+1);
        float a10=mat.GetValue((j+1)*P+q),a11=mat.GetValue((j+1)*P+q+1);
        float inv0=-1.0f/d0;value*=d0;Step<Width>(mat,col,pivot,weights,ix,j,q,inv0);
        float d1Base=a11+a10*a01*inv0,d1=d1Base,inv1=-1.0f/d1;
        value*=d1;Step<Width>(mat,col,pivot,weights,ix,j+1,q+1,inv1);
    }
    template<bool Batched> static __aicore__ inline void PairCoupled(
        LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<float> pivot,
        LocalTensor<float> weights,LocalTensor<uint32_t> ix,uint32_t j,float &value,uint32_t activeRows=N){
        #pragma clang fp contract(on)
        const uint32_t q=j;
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        if constexpr(Batched){
            LocalTensor<uint32_t> pairIx(TPosition::VECCALC,172032,64);
            SetVectorMask<float>(0,~uint64_t(0));
            vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)pairIx.GetPhyAddr(),uint32_t((uint64_t)mat[(N-32)*P+q].GetPhyAddr()),8,1);
            SetVectorMask<float>(0,uint64_t(0xffffffffu));
        }else{
            vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[(N-32)*P+q].GetPhyAddr()),8,1);
            vgather((__ubuf__ uint32_t*)col[32].GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[(N-32)*P+q+1].GetPhyAddr()),8,1);
        }

        float d0=mat.GetValue(j*P+q),a01=mat.GetValue(j*P+q+1);
        float a10=mat.GetValue((j+1)*P+q),a11=mat.GetValue((j+1)*P+q+1);

        float delta=d0*a11-a10*a01,reciprocal=1.0f/delta;
        float c00=-a11*reciprocal,c01=a01*reciprocal,c10=a10*reciprocal,c11=-d0*reciprocal;
        value*=delta;

        DetNativeMuls(pivot,mat[j*P+N-32],c00,uint64_t(32),1,UnaryRepeatParams{1,1,8,8});
        DetNativeMuls(pivot[P],mat[(j+1)*P+N-32],c11,uint64_t(32),1,UnaryRepeatParams{1,1,8,8});
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0,~uint64_t(0));
        vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,8);
        SetVectorMask<float>(0,uint64_t(0xffffffffu));
        vaxpy((__ubuf__ float*)pivot.GetPhyAddr(),(__ubuf__ float*)mat[(j+1)*P+N-32].GetPhyAddr(),c01,1,1,1,8,8);
        vaxpy((__ubuf__ float*)pivot[P].GetPhyAddr(),(__ubuf__ float*)mat[j*P+N-32].GetPhyAddr(),c10,1,1,1,8,8);
        PipeBarrier<PIPE_V>();
        DetNativeMla(mat[(j+2)*P+N-32],pivot,weights[(j+2-(N-32))*8],uint64_t(32),uint8_t(activeRows-j-2),BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
        DetNativeMla(mat[(j+2)*P+N-32],pivot[P],weights[256+(j+2-(N-32))*8],uint64_t(32),uint8_t(activeRows-j-2),BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
    }
    static __aicore__ inline void BlockWide(LocalTensor<float> mat,LocalTensor<float> pivot,uint32_t j,float &value,uint32_t activeRows=N){
        #pragma clang fp contract(on)
        LocalTensor<uint32_t> wideIx(TPosition::VECCALC,173056,128);
        LocalTensor<float> col(TPosition::VECCALC,174080,128);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)wideIx.GetPhyAddr(),uint32_t((uint64_t)mat[j].GetPhyAddr()),8,2);
        float d0=mat.GetValue(j*P+j),a01=mat.GetValue(j*P+j+1);
        float a10=mat.GetValue((j+1)*P+j),a11=mat.GetValue((j+1)*P+j+1);
        
        float delta=d0*a11-a10*a01,reciprocal=1.0f/delta;
        float c00=-a11*reciprocal,c01=a01*reciprocal,c10=a10*reciprocal,c11=-d0*reciprocal;
        value*=delta;

        DetNativeMuls(pivot,mat[j*P],c00,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});
        DetNativeMuls(pivot[P],mat[(j+1)*P],c11,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});
        PipeBarrier<PIPE_V>();
        vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),P/8,P,8);
        vbrcb((__ubuf__ uint32_t*)mat[72].GetPhyAddr(),(__ubuf__ uint32_t*)col[64].GetPhyAddr(),P/8,P,8);
        vaxpy((__ubuf__ float*)pivot.GetPhyAddr(),(__ubuf__ float*)mat[(j+1)*P].GetPhyAddr(),c01,1,1,1,8,8);
        vaxpy((__ubuf__ float*)pivot[P].GetPhyAddr(),(__ubuf__ float*)mat[j*P].GetPhyAddr(),c10,1,1,1,8,8);
        PipeBarrier<PIPE_V>();
        DetNativeMla(mat[(j+2)*P],pivot,mat[(j+2)*P+64],uint64_t(64),uint8_t(activeRows-j-2),BinaryRepeatParams{1,1,0,P/8,0,P/8});PipeBarrier<PIPE_V>();
        DetNativeMla(mat[(j+2)*P],pivot[P],mat[(j+2)*P+72],uint64_t(64),uint8_t(activeRows-j-2),BinaryRepeatParams{1,1,0,P/8,0,P/8});PipeBarrier<PIPE_V>();

    }

public:

    template<uint32_t ActualN> static __aicore__ inline float FactorOnly(
        LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<float> pivot,
        LocalTensor<float> weights,LocalTensor<uint32_t> ix,float &condition){
        static_assert(ActualN%2==0 && ActualN>32 && ActualN<=N && N-ActualN<8);
        LocalTensor<uint32_t> wideIx(TPosition::VECCALC,173056,128);
        LocalTensor<uint32_t> pairIx(TPosition::VECCALC,172032,64);
        DataCopy(wideIx,ix,64);
        Adds(wideIx[64].ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(4),64);
        DataCopy(pairIx,ix,32);
        Adds(pairIx[32].ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(4),32);
        PipeBarrier<PIPE_V>();SetMaskNorm();SetVectorMask<float>(0,~uint64_t(0));
        float value=1.0f;uint32_t j=0;
        #pragma unroll 1
        for(;j<N-32;j+=2)BlockWide(mat,pivot,j,value,ActualN);
        SetVectorMask<float>(0,uint64_t(0xffffffffu));
        #pragma unroll 1
        for(;j+1<ActualN-4;j+=2)PairCoupled<true>(mat,col,pivot,weights,ix,j,value,ActualN);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        return value*TerminalDet<P,true>(mat,ActualN-4,4,&condition);
    }

    __aicore__ inline void Run(GM_ADDR a,GM_ADDR y,const LinalgDetTilingData &t,GM_ADDR workspace,GM_ADDR tiling){
        static_assert(N%8==0 && N>32 && N<=64);
        LocalTensor<float> arena(TPosition::VECCALC,0,C*P+24*C+P+256);
        auto mat=arena,col=mat[C*P],weights=col[C],pivot=weights[8*C+64],diag=pivot[P],temp=diag[C],signs=temp[C],sc=signs[C];
        auto ramp=temp.ReinterpretCast<int32_t>(),rows=signs.ReinterpretCast<int32_t>();auto ix=sc[128].ReinterpretCast<uint32_t>();
        LocalTensor<float> raw(TPosition::VECCALC,128*1024,4096);
        LocalTensor<uint32_t> packing(TPosition::VECCALC,144*1024,512);
        GlobalTensor<float> input,output;input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);
        bool prepared=false;
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks){
            DataCopyPad(raw,input[uint64_t(batch)*N*N],DataCopyExtParams{1,N*N*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0f});
            if(!prepared){
                for(uint32_t k=0;k<8;++k){ramp.SetValue(k,k*4);col.ReinterpretCast<int32_t>().SetValue(k,k<N/8?int32_t((N/8-1-k)*32):0);}
                SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
                for(uint32_t k=8;k<C;k*=2){Adds(ramp[k],ramp,int32_t(k*4),k);PipeBarrier<PIPE_V>();}
                Muls(ix.ReinterpretCast<int32_t>(),ramp,int32_t(P),C);Muls(rows,ramp,int32_t(N),C);PipeBarrier<PIPE_V>();
                Brcb(weights.ReinterpretCast<int32_t>(),rows,C/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                Add(packing.ReinterpretCast<int32_t>(),weights.ReinterpretCast<int32_t>(),col.ReinterpretCast<int32_t>(),uint64_t(64),uint8_t(N/8),BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();prepared=true;
            }
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            Muls(mat,raw,1.0f,uint64_t(N),uint8_t(N),UnaryRepeatParams{1,1,uint8_t(P/8),uint8_t(N/8)});PipeBarrier<PIPE_V>();
            float value=1.0f;
            if constexpr(LayoutOnly){
                SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);int32_t errors=0;
                for(uint32_t r=0;r<N;++r)for(uint32_t c=0;c<N;++c){
                    const uint32_t original=N-8-(c/8)*8+c%8;
                    errors+=mat.ReinterpretCast<uint32_t>().GetValue(r*P+c)!=raw.ReinterpretCast<uint32_t>().GetValue(r*N+original);
                }
                sc.SetValue(0,float(errors));
            }else{
                uint32_t j=0;SetMaskNorm();SetVectorMask<float>(0,~uint64_t(0));
if constexpr(P>=80){LocalTensor<uint32_t> wideIx(TPosition::VECCALC,173056,128);DataCopy(wideIx,ix,64);Adds(wideIx[64].ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(4),64);PipeBarrier<PIPE_V>();}
                #pragma unroll
                for(;j<N-32;j+=2){if constexpr(P>=80)BlockWide(mat,pivot,j,value);else Pair<64>(mat,col,pivot,weights,ix,j,value);}
                LocalTensor<uint32_t> pairIx(TPosition::VECCALC,172032,64);
                DataCopy(pairIx,ix,32);
                Adds(pairIx[32].ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(4),32);PipeBarrier<PIPE_V>();
                SetVectorMask<float>(0,uint64_t(0xffffffffu));
                #pragma unroll
                for(;j<N-6;j+=2)PairCoupled<true>(mat,col,pivot,weights,ix,j,value);
                SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                value*=BlockColumnTerminalSix<P>(mat,N);sc.SetValue(0,value);
            }
            SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);DataCopyPad(output[batch],sc,DataCopyExtParams{1,4,0,0,0});const bool abnormal=DetAbnormal(value,false);PipeBarrier<PIPE_MTE3>();
            if(abnormal){PivotedFallback(a,y,workspace,tiling,batch);prepared=false;}
        }
    }
};

template<uint32_t Capacity=0,uint32_t Order=0,uint32_t Pitch=0> class ResidentDet {
public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR y,const LinalgDetTilingData &t,GM_ADDR tiling,GM_ADDR workspace) {
        const uint32_t inputN=Order?Order:t.n,n=Order?Order:(Capacity && Capacity<=32?Capacity:inputN),r=Capacity?Capacity:t.rows,p=Pitch?Pitch:(Capacity?Capacity+8:(n+7)/8*8+8);
        LocalTensor<float> arena(TPosition::VECCALC,0,(Capacity?Capacity:n)*p+24*r+p+256);
        auto mat=arena,col=mat[(Capacity?Capacity:n)*p],weights=col[r],pivot=weights[8*r+64],diag=pivot[p];
        auto temp=diag[r],signs=temp[r],sc=signs[r];
        auto ix=sc[128].ReinterpretCast<uint32_t>(),dix=ix[r],expanded=dix[r];
        bool offsetsReady=false,tablesReady=false;
        LocalTensor<float> raw(TPosition::VECCALC,128*1024,4096);LocalTensor<uint32_t> packing(TPosition::VECCALC,144*1024,4096);
        GlobalTensor<float> input,output;input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            float tinyValue=1.0f;
            if constexpr(Capacity!=64){if(!tablesReady){
                GlobalTensor<uint32_t> metadata;metadata.SetGlobalBuffer((__gm__ uint32_t*)(tiling+32));
                if constexpr(Capacity==16)DataCopy(ix,metadata,32);else{DataCopy(ix,metadata,r);DataCopy(dix,metadata[64],r);}
                tablesReady=true;
            }}
            if constexpr(Capacity && Capacity<=32){
                if(!Order && inputN<Capacity){
                    Duplicate(mat[inputN*p],0.0f,(Capacity-inputN)*p);PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    for(uint32_t q=inputN;q<Capacity;++q)mat.SetValue(q*(p+1),1.0f);
                    SetFlag<HardEvent::S_MTE2>(0);WaitFlag<HardEvent::S_MTE2>(0);
                }
            }
            if(Capacity==64 && n%8){
                DataCopyPad(raw,input[(uint64_t)batch*n*n],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0f});
                if(n%8){
                    auto columns=temp.ReinterpretCast<int32_t>(),rows=signs.ReinterpretCast<int32_t>();
                    for(uint32_t k=0;k<8;++k)columns.SetValue(k,k*4);SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
                    for(uint32_t k=8;k<64;k*=2){Adds(columns[k],columns,int32_t(k*4),k);PipeBarrier<PIPE_V>();}
                    if(!offsetsReady){Muls(ix.ReinterpretCast<int32_t>(),columns,int32_t(p),64);offsetsReady=true;}
                    Muls(rows,columns,int32_t(n),64);PipeBarrier<PIPE_V>();Brcb(weights.ReinterpretCast<int32_t>(),rows,8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                    Add(packing.ReinterpretCast<int32_t>(),columns,weights.ReinterpretCast<int32_t>(),uint64_t(64),uint8_t(n),BinaryRepeatParams{1,1,0,8,0,1});PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
                if(n%8)Gather(mat,raw,packing,0,uint64_t(n),uint8_t(n),p/8);
                else DataCopy(mat,raw,DataCopyParams{uint16_t(n),uint16_t(n/8),0,uint16_t((p-n)/8)});
                PipeBarrier<PIPE_V>();
            }else{
                DataCopyPad(mat,input[(uint64_t)batch*inputN*inputN],DataCopyExtParams{uint16_t(inputN),inputN*4,0,uint32_t((p-(inputN+7)/8*8)/8),0},DataCopyPadExtParams<float>{true,0,uint8_t((8-inputN%8)%8),0.0f});
                if constexpr(Capacity==64){if(!offsetsReady){
                    for(uint32_t k=0;k<8;++k)ix.SetValue(k,k*p*4);
                    SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
                    for(uint32_t k=8;k<r;k*=2){Adds(ix[k].ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(k*p*4),k);PipeBarrier<PIPE_V>();}
                    offsetsReady=true;
                }}
                SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            }
            if constexpr(Capacity && Capacity<=32){Brcb(expanded,ix,r/8,BrcbRepeatParams{1,8});Duplicate(expanded[8*r],uint32_t(0),64);PipeBarrier<PIPE_V>();}
            const uint32_t tail=Capacity>32 && n>=6?6:1;
            if constexpr(Capacity==64){
                #pragma clang fp contract(off)
                const uint32_t origin=((n+7)/8)*8-32;
                uint64_t columnMask[2]={n==64?~uint64_t(0):(uint64_t(1)<<n)-1,0};
                {
                SetMaskNorm();SetVectorMask<float>(0,~uint64_t(0));
                const uint32_t stop=(n-tail<origin?n-tail:origin);uint32_t j=0;
                #pragma unroll 1
                for(;j+1<stop;j+=2){
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    float d0=mat.GetValue(j*p+j),a01=mat.GetValue(j*p+j+1);
                    float a10=mat.GetValue((j+1)*p+j),a11=mat.GetValue((j+1)*p+j+1);
                    float inv0=-1.0f/d0;tinyValue*=d0;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[j].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j)*p+0],inv0,uint64_t(64),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),uint16_t(p/8),uint16_t(p),uint8_t(8));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1)*p+0],pivot,mat[(j+1)*p+64],uint64_t(64),uint8_t(n-(j)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,uint8_t(p/8)});
                    PipeBarrier<PIPE_V>();
                    float u01=a01*inv0,correction=a10*u01;
                    float d1=a11+correction,inv1=-1.0f/d1;tinyValue*=d1;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[j+1].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j+1)*p+0],inv1,uint64_t(64),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),uint16_t(p/8),uint16_t(p),uint8_t(8));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1+1)*p+0],pivot,mat[(j+1+1)*p+64],uint64_t(64),uint8_t(n-(j+1)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,uint8_t(p/8)});
                    PipeBarrier<PIPE_V>();
                }
                if(j<stop){
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    float d=mat.GetValue(j*p+j),inverse=-1.0f/d;tinyValue*=d;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[j].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j)*p+0],inverse,uint64_t(64),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),uint16_t(p/8),uint16_t(p),uint8_t(8));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1)*p+0],pivot,mat[(j+1)*p+64],uint64_t(64),uint8_t(n-(j)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,uint8_t(p/8)});
                    PipeBarrier<PIPE_V>();
                }
                }
                {
                SetMaskNorm();SetVectorMask<float>(0,uint64_t(0xffffffffu));
                const uint32_t stop=n-tail;uint32_t j=origin;
                #pragma unroll 1
                for(;j+1<stop;j+=2){
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    float d0=mat.GetValue(j*p+j),a01=mat.GetValue(j*p+j+1);
                    float a10=mat.GetValue((j+1)*p+j),a11=mat.GetValue((j+1)*p+j+1);
                    float inv0=-1.0f/d0;tinyValue*=d0;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[origin*p+(j)].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j)*p+origin],inv0,uint64_t(32),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    SetVectorMask<float>(0,~uint64_t(0));
                    vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,4);
                    SetVectorMask<float>(0,uint64_t(0xffffffffu));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1)*p+origin],pivot,weights[(j+1-origin)*8],uint64_t(32),uint8_t(n-(j)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,1});
                    PipeBarrier<PIPE_V>();
                    float u01=a01*inv0,correction=a10*u01;
                    float d1=a11+correction,inv1=-1.0f/d1;tinyValue*=d1;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[origin*p+(j+1)].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j+1)*p+origin],inv1,uint64_t(32),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    SetVectorMask<float>(0,~uint64_t(0));
                    vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,4);
                    SetVectorMask<float>(0,uint64_t(0xffffffffu));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1+1)*p+origin],pivot,weights[(j+1+1-origin)*8],uint64_t(32),uint8_t(n-(j+1)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,1});
                    PipeBarrier<PIPE_V>();
                }
                if(j<stop){
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    float d=mat.GetValue(j*p+j),inverse=-1.0f/d;tinyValue*=d;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[origin*p+(j)].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j)*p+origin],inverse,uint64_t(32),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    SetVectorMask<float>(0,~uint64_t(0));
                    vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,4);
                    SetVectorMask<float>(0,uint64_t(0xffffffffu));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1)*p+origin],pivot,weights[(j+1-origin)*8],uint64_t(32),uint8_t(n-(j)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,1});
                    PipeBarrier<PIPE_V>();
                }
                }
            }else{
            #pragma unroll 1
            for(uint32_t j=0;j+tail<n;++j) {
                const uint32_t first=Capacity?0:j/8*8,width=Capacity?Capacity:((n+7)/8*8-first);
                if constexpr(Capacity && Capacity<=32){
                    Gather(weights[j*8],mat[j],expanded[j*8],0,uint64_t(64),uint8_t((n-j+7)/8),8);PipeBarrier<PIPE_V>();
                    Div(pivot,mat[j*p+first],weights[j*8],uint64_t(width),1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();Muls(pivot,pivot,-1.0f,width);PipeBarrier<PIPE_V>();
                }else{
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);float d=mat.GetValue(j*p+j),inverse=-1.0f/d;tinyValue*=d;
                    Gather(col,mat[j],ix,0,n);Muls(pivot,mat[j*p+first],inverse,width);PipeBarrier<PIPE_V>();
                    Brcb(weights,col,r/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                }
                for(uint32_t c=0;c<width;c+=64)
                    MulAddDst(mat[(j+1)*p+first+c],pivot[c],weights[(j+1)*8],uint64_t(width-c<64?width-c:64),uint8_t(n-j-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,1});
                PipeBarrier<PIPE_V>();
            }
            }
            float value;if constexpr(Capacity && Capacity<=32){Duplicate(diag,1.0f,r);PipeBarrier<PIPE_V>();Gather(diag,mat,dix,0,n);PipeBarrier<PIPE_V>();}
            if constexpr(Capacity && Capacity<=32){
                Brcb(weights,diag,r/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t step=1;step<r;step*=2){Mul(weights,weights,weights[step*8],uint64_t(8),uint8_t(r/(2*step)),BinaryRepeatParams{1,1,1,uint8_t(2*step),uint8_t(2*step),uint8_t(2*step)});PipeBarrier<PIPE_V>();}
                SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);value=weights.GetValue(0);
            }else{SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);value=tinyValue*(tail==6?TerminalSix<Pitch?Pitch:Capacity+8>(mat,n):mat.GetValue((n-1)*(p+1)));}
            sc.SetValue(0,value);SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
            DataCopyPad(output[batch],sc,DataCopyExtParams{1,4,0,0,0});const bool abnormal=DetAbnormal(value,false);PipeBarrier<PIPE_MTE3>();
            if(abnormal){PivotedFallback(a,y,workspace,tiling,batch);offsetsReady=false;tablesReady=false;}
        }
    }
};



// Block LU stores only the two solved side panels. The cubic trailing update
// is a Cube product; the last diagonal block uses its actual remaining order.
template<uint32_t Block> static __aicore__ constexpr MatmulConfig BlockConfiguration(){auto c=GetNormalConfig();c.basicM=64;c.basicN=64;c.basicK=Block<32?Block:32;c.singleCoreM=64;c.singleCoreN=64;c.singleCoreK=Block;c.enableSetBias=false;return c;}

template<bool Publish> static __aicore__ inline void AugStep(LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<uint32_t> ix,uint32_t q,uint32_t width,uint32_t own,float inverse){
    constexpr uint32_t P=136,B=64;const uint32_t first=q/8*8,count=B+own-first;
    if(q+1<width){
        SetVectorMask<float>(0,~uint64_t(0));
        vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[q].GetPhyAddr()),8,1);
    }
    // The diagonal suffix and RHS share the same scale and row multiplier.
    DetNativeMulsCount(mat[q*P+first],mat[q*P+first],inverse,count);PipeBarrier<PIPE_V>();
    if(q+1<width){
        vbrcb((__ubuf__ uint32_t*)mat[120].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),P/8,P,8);PipeBarrier<PIPE_V>();
        DetNativeMlaMasked(mat[(q+1)*P+first],mat[q*P+first],mat[(q+1)*P+120],uint64_t(count<64?count:64),uint8_t(width-q-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
        // This first chunk contains the complete active diagonal suffix.
        if constexpr(Publish)SetFlag<HardEvent::V_S>(0);
        if(count>64){DetNativeMlaMasked(mat[(q+1)*P+first+64],mat[q*P+first+64],mat[(q+1)*P+120],uint64_t(count-64),uint8_t(width-q-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});}
        PipeBarrier<PIPE_V>();
    }
}
template<uint32_t Width=0> static __aicore__ inline float AugFactor(LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<uint32_t> ix,uint32_t liveWidth,uint32_t own){
    constexpr uint32_t P=136;const uint32_t width=Width?Width:liveWidth;
    SetMaskNorm();SetFlag<HardEvent::V_S>(0);float value=1.0f;uint32_t q=0;
    for(;q+1<width;q+=2){
        #pragma clang fp contract(off)
        WaitFlag<HardEvent::V_S>(0);
        float d0=mat.GetValue(q*P+q),a01=mat.GetValue(q*P+q+1),a10=mat.GetValue((q+1)*P+q),a11=mat.GetValue((q+1)*P+q+1);
        float inv0=-1.0f/d0;value*=d0;AugStep<false>(mat,col,ix,q,width,own,inv0);
        float u01=a01*inv0,correction=a10*u01,d1=a11+correction,inv1=-1.0f/d1;value*=d1;
        AugStep<true>(mat,col,ix,q+1,width,own,inv1);
    }
    if(q<width){WaitFlag<HardEvent::V_S>(0);float d=mat.GetValue(q*P+q);value*=d;AugStep<false>(mat,col,ix,q,width,own,-1.0f/d);}
    return value;
}

template<bool Publish> static __aicore__ inline void AugLeftStep(LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<uint32_t> ix,uint32_t q,uint32_t width,uint32_t own,float inverse){
    constexpr uint32_t P=136,B=64;const uint32_t first=q/8*8,count=B+own-first;
    if(q+1<width){
        SetVectorMask<float>(0,~uint64_t(0));
        vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[q].GetPhyAddr()),8,1);
        PipeBarrier<PIPE_V>();DetNativeMuls(col,col,inverse,uint64_t(B),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        vbrcb((__ubuf__ uint32_t*)mat[120].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),P/8,P,8);PipeBarrier<PIPE_V>();
        DetNativeMlaMasked(mat[(q+1)*P+first],mat[q*P+first],mat[(q+1)*P+120],uint64_t(count<64?count:64),uint8_t(width-q-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
        if constexpr(Publish)SetFlag<HardEvent::V_S>(0);
        if(count>64){DetNativeMlaMasked(mat[(q+1)*P+first+64],mat[q*P+first+64],mat[(q+1)*P+120],uint64_t(count-64),uint8_t(width-q-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});}
        PipeBarrier<PIPE_V>();
    }
}
template<uint32_t Width=0> static __aicore__ inline float AugLeftFactor(LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<uint32_t> ix,uint32_t liveWidth,uint32_t own){
    constexpr uint32_t P=136;const uint32_t width=Width?Width:liveWidth;
    SetMaskNorm();SetFlag<HardEvent::V_S>(0);float value=1.0f;uint32_t q=0;
    for(;q+1<width;q+=2){
        #pragma clang fp contract(off)
        WaitFlag<HardEvent::V_S>(0);
        float d0=mat.GetValue(q*P+q),a01=mat.GetValue((q+1)*P+q),a10=mat.GetValue(q*P+q+1),a11=mat.GetValue((q+1)*P+q+1);
        float inv0=-1.0f/d0;value*=d0;AugLeftStep<false>(mat,col,ix,q,width,own,inv0);
        float u01=a01*inv0,correction=a10*u01,d1=a11+correction,inv1=-1.0f/d1;value*=d1;
        AugLeftStep<true>(mat,col,ix,q+1,width,own,inv1);
    }
    if(q<width){WaitFlag<HardEvent::V_S>(0);float d=mat.GetValue(q*P+q);value*=d;AugLeftStep<false>(mat,col,ix,q,width,own,-1.0f/d);}
    return value;
}
// A diagonal-only panel needs no side solve. Reuse the fixed 64/32-lane
// resident schedule, preserving the original four-by-four terminal formula.
// Block-diagonal 2x2 LU convention: the right side uses -T^-1 B;
// the transposed left side uses -H^-T C^T, with A=-T H and diag(H)=-I2.
// Only live rows below the pair are updated. Right pivot-side rows are saved
// after use, beside the final update; the retired diagonal rows need no store.
static __aicore__ inline void DetNativeAxpyCount(LocalTensor<float> dst,LocalTensor<float> src,float scalar,uint32_t count){
    SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,count);
    vaxpy((__ubuf__ float*)dst.GetPhyAddr(),(__ubuf__ float*)src.GetPhyAddr(),scalar,1,1,1,8,8);
    ResetMask();SetMaskNorm();
}

static __aicore__ inline void DetRowMulsPair(LocalTensor<float> d0,LocalTensor<float> d1,
    LocalTensor<float> s0,LocalTensor<float> s1,float c0,float c1,uint32_t count){
    SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,count);
    vmuls((__ubuf__ float*)d0.GetPhyAddr(),(__ubuf__ float*)s0.GetPhyAddr(),c0,1,1,1,8,8);
    vmuls((__ubuf__ float*)d1.GetPhyAddr(),(__ubuf__ float*)s1.GetPhyAddr(),c1,1,1,1,8,8);
    ResetMask();SetMaskNorm();
}
static __aicore__ inline void DetRowAxpyPair(LocalTensor<float> d0,LocalTensor<float> d1,
    LocalTensor<float> s0,LocalTensor<float> s1,float c0,float c1,uint32_t count){
    SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,count);
    vaxpy((__ubuf__ float*)d0.GetPhyAddr(),(__ubuf__ float*)s0.GetPhyAddr(),c0,1,1,1,8,8);
    vaxpy((__ubuf__ float*)d1.GetPhyAddr(),(__ubuf__ float*)s1.GetPhyAddr(),c1,1,1,1,8,8);
    ResetMask();SetMaskNorm();
}
static __aicore__ inline void DetRowMulsRounded(LocalTensor<float> dst,LocalTensor<float> src,float c,uint32_t count){
    vmuls((__ubuf__ float*)dst.GetPhyAddr(),(__ubuf__ float*)src.GetPhyAddr(),c,uint8_t((count+63)/64),1,1,8,8);
}
static __aicore__ inline void DetRowAxpyRounded(LocalTensor<float> dst,LocalTensor<float> src,float c,uint32_t count){
    vaxpy((__ubuf__ float*)dst.GetPhyAddr(),(__ubuf__ float*)src.GetPhyAddr(),c,uint8_t((count+63)/64),1,1,8,8);
}
template<uint32_t Width,bool Left> static __aicore__ inline float AugBlockFactor(
    LocalTensor<float> mat,LocalTensor<float> unused,LocalTensor<uint32_t> ix,uint32_t liveWidth,uint32_t own){
    #pragma clang fp contract(on)
    constexpr uint32_t P=136,B=64;
    const uint32_t width=Width?Width:liveWidth;
    LocalTensor<float> columns(TPosition::VECCALC,160*1024,256);
    LocalTensor<float> factors(TPosition::VECCALC,172*1024,1024);
    LocalTensor<uint32_t> pairIx(TPosition::VECCALC,162*1024,128);
    LocalTensor<float> row0(TPosition::VECCALC,164*1024,128),row1(TPosition::VECCALC,165*1024,128);
    DataCopy(pairIx,ix,64);
    Adds(pairIx[64].ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(4),64);
    PipeBarrier<PIPE_V>();SetMaskNorm();SetFlag<HardEvent::V_S>(0);
    float value=1.0f;uint32_t q=0;
    for(;q+1<width;q+=2){
        WaitFlag<HardEvent::V_S>(0);
        float a00=mat.GetValue(q*P+q),a01=mat.GetValue(q*P+q+1);
        float a10=mat.GetValue((q+1)*P+q),a11=mat.GetValue((q+1)*P+q+1);
        SetVectorMask<float>(0,~uint64_t(0));
        if(q+2<width){
            vgather((__ubuf__ uint32_t*)columns.GetPhyAddr(),(__ubuf__ uint32_t*)pairIx.GetPhyAddr(),uint32_t((uint64_t)mat[q].GetPhyAddr()),8,2);
        }
        float delta=a00*a11-a10*a01,reciprocal=1.0f/delta;
        float c00=-a11*reciprocal,c01=a01*reciprocal,c10=a10*reciprocal,c11=-a00*reciprocal;
        // abs(a00*c00)=abs(a00*a11/delta), the inverse amplification
        // already implicit in the required coefficient. This avoids a
        // separate product-scale calculation before issuing the reciprocal.
        // 2^10 corresponds to losing about11 FP32 bits by cancellation.
        if(!(DetMagnitude(a00*c00)<1024.0f)){
            // The speculative gather reads the matrix that sequential repair
            // is about to modify. Drain it only on this rare repair branch.
            PipeBarrier<PIPE_V>();
            float inv0=-1.0f/a00;value*=a00;
            if constexpr(Left)AugLeftStep<false>(mat,unused,ix,q,width,own,inv0);
            else AugStep<false>(mat,unused,ix,q,width,own,inv0);
            // Read the actual Vector-rounded successor pivot, not a Scalar
            // fused prediction with different intermediate rounding.
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
            float d1=mat.GetValue((q+1)*P+q+1),inv1=-1.0f/d1;value*=d1;
            if constexpr(Left)AugLeftStep<true>(mat,unused,ix,q+1,width,own,inv1);
            else AugStep<true>(mat,unused,ix,q+1,width,own,inv1);
            continue;
        }
        value*=delta;
        const uint32_t first=q/8*8,count=B+own-first;
        if constexpr(Left){
            if(q+2<width){
                PipeBarrier<PIPE_V>();
                DetNativeMuls(columns[128],columns,c00,64,1,UnaryRepeatParams{1,1,8,8});
                DetNativeMuls(columns[192],columns[64],c11,64,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                vaxpy((__ubuf__ float*)columns[128].GetPhyAddr(),(__ubuf__ float*)columns[64].GetPhyAddr(),c10,1,1,1,8,8);
                vaxpy((__ubuf__ float*)columns[192].GetPhyAddr(),(__ubuf__ float*)columns.GetPhyAddr(),c01,1,1,1,8,8);PipeBarrier<PIPE_V>();
                vbrcb((__ubuf__ uint32_t*)factors.GetPhyAddr(),(__ubuf__ uint32_t*)columns[128].GetPhyAddr(),1,8,16);PipeBarrier<PIPE_V>();
            }
        }else{
            DetRowMulsRounded(row0,mat[q*P+first],c00,count);
            DetRowMulsRounded(row1,mat[(q+1)*P+first],c11,count);PipeBarrier<PIPE_V>();
            if(q+2<width){
                vbrcb((__ubuf__ uint32_t*)factors.GetPhyAddr(),(__ubuf__ uint32_t*)columns.GetPhyAddr(),1,8,16);
            }
            DetRowAxpyRounded(row0,mat[(q+1)*P+first],c01,count);
            DetRowAxpyRounded(row1,mat[q*P+first],c10,count);PipeBarrier<PIPE_V>();
        }
        if(q+2<width){
            auto p0=Left?mat[q*P+first]:row0;
            auto p1=Left?mat[(q+1)*P+first]:row1;
            auto dst=mat[(q+2)*P+first];
            auto w0=factors[(q+2)*8],w1=factors[512+(q+2)*8];
            DetNativeMlaMasked(dst,p0,w0,uint64_t(count<64?count:64),uint8_t(width-q-2),BinaryRepeatParams{1,1,0,P/8,0,1});
            if(count>64)DetNativeMlaMasked(dst[64],p0[64],w0,uint64_t(count-64),uint8_t(width-q-2),BinaryRepeatParams{1,1,0,P/8,0,1});
            PipeBarrier<PIPE_V>();
            DetNativeMlaMasked(dst,p1,w1,uint64_t(count<64?count:64),uint8_t(width-q-2),BinaryRepeatParams{1,1,0,P/8,0,1});
            // All next-pair diagonal values are in this first chunk.
            SetFlag<HardEvent::V_S>(0);
            if(count>64)DetNativeMlaMasked(dst[64],p1[64],w1,uint64_t(count-64),uint8_t(width-q-2),BinaryRepeatParams{1,1,0,P/8,0,1});
        }
        if constexpr(!Left){
            // own is a multiple of eight in the existing shard geometry.
            copy_ubuf_to_ubuf((__ubuf__ void*)mat[q*P+B].GetPhyAddr(),(__ubuf__ void*)row0[B-first].GetPhyAddr(),0,1,own/8,0,0);
            copy_ubuf_to_ubuf((__ubuf__ void*)mat[(q+1)*P+B].GetPhyAddr(),(__ubuf__ void*)row1[B-first].GetPhyAddr(),0,1,own/8,0,0);
        }
        PipeBarrier<PIPE_V>();
    }
    if(q<width){
        WaitFlag<HardEvent::V_S>(0);float d=mat.GetValue(q*P+q);value*=d;
        if constexpr(Left)AugLeftStep<false>(mat,unused,ix,q,width,own,-1.0f/d);
        else AugStep<false>(mat,unused,ix,q,width,own,-1.0f/d);
    }
    return value;
}

template<uint32_t Width,uint32_t P> static __aicore__ inline float FinalResidentFactor(
    LocalTensor<float> mat,LocalTensor<float> col,LocalTensor<float> pivot,
    LocalTensor<float> weights,LocalTensor<uint32_t> ix,uint32_t liveWidth,float &condition){
    #pragma clang fp contract(off)
    if constexpr(Width>32 && Width%2==0){
        return BlockColumnResidentDet<(Width+7)/8*8,P>::template FactorOnly<Width>(mat,col,pivot,weights,ix,condition);
    }
    const uint32_t n=Width?Width:liveWidth,p=P,tail=4;
    float tinyValue=1.0f;
                const uint32_t origin=((n+7)/8)*8-32;
                
                {
                SetMaskNorm();SetVectorMask<float>(0,~uint64_t(0));
                const uint32_t stop=(n-tail<origin?n-tail:origin);uint32_t j=0;
                #pragma unroll 1
                for(;j+1<stop;j+=2){
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    float d0=mat.GetValue(j*p+j),a01=mat.GetValue(j*p+j+1);
                    float a10=mat.GetValue((j+1)*p+j),a11=mat.GetValue((j+1)*p+j+1);
                    float inv0=-1.0f/d0;tinyValue*=d0;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[j].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j)*p+0],inv0,uint64_t(64),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                    vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),uint16_t(p/8),uint16_t(p),uint8_t(8));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1)*p+0],pivot,mat[(j+1)*p+64],uint64_t(64),uint8_t(n-(j)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,uint8_t(p/8)});
                    PipeBarrier<PIPE_V>();
                    float u01=a01*inv0,correction=a10*u01;
                    float d1=a11+correction,inv1=-1.0f/d1;tinyValue*=d1;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[j+1].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j+1)*p+0],inv1,uint64_t(64),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                    vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),uint16_t(p/8),uint16_t(p),uint8_t(8));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1+1)*p+0],pivot,mat[(j+1+1)*p+64],uint64_t(64),uint8_t(n-(j+1)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,uint8_t(p/8)});
                    PipeBarrier<PIPE_V>();
                }
                if(j<stop){
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    float d=mat.GetValue(j*p+j),inverse=-1.0f/d;tinyValue*=d;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[j].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j)*p+0],inverse,uint64_t(64),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                    vbrcb((__ubuf__ uint32_t*)mat[64].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),uint16_t(p/8),uint16_t(p),uint8_t(8));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1)*p+0],pivot,mat[(j+1)*p+64],uint64_t(64),uint8_t(n-(j)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,uint8_t(p/8)});
                    PipeBarrier<PIPE_V>();
                }
                }
                {
                SetMaskNorm();SetVectorMask<float>(0,uint64_t(0xffffffffu));
                const uint32_t stop=n-tail;uint32_t j=origin;
                #pragma unroll 1
                for(;j+1<stop;j+=2){
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    float d0=mat.GetValue(j*p+j),a01=mat.GetValue(j*p+j+1);
                    float a10=mat.GetValue((j+1)*p+j),a11=mat.GetValue((j+1)*p+j+1);
                    float inv0=-1.0f/d0;tinyValue*=d0;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[origin*p+(j)].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j)*p+origin],inv0,uint64_t(32),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                    SetVectorMask<float>(0,~uint64_t(0));
                    vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,4);
                    SetVectorMask<float>(0,uint64_t(0xffffffffu));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1)*p+origin],pivot,weights[(j+1-origin)*8],uint64_t(32),uint8_t(n-(j)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,1});
                    PipeBarrier<PIPE_V>();
                    float u01=a01*inv0,correction=a10*u01;
                    float d1=a11+correction,inv1=-1.0f/d1;tinyValue*=d1;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[origin*p+(j+1)].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j+1)*p+origin],inv1,uint64_t(32),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                    SetVectorMask<float>(0,~uint64_t(0));
                    vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,4);
                    SetVectorMask<float>(0,uint64_t(0xffffffffu));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1+1)*p+origin],pivot,weights[(j+1+1-origin)*8],uint64_t(32),uint8_t(n-(j+1)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,1});
                    PipeBarrier<PIPE_V>();
                }
                if(j<stop){
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    float d=mat.GetValue(j*p+j),inverse=-1.0f/d;tinyValue*=d;

                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[origin*p+(j)].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot,mat[(j)*p+origin],inverse,uint64_t(32),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                    SetVectorMask<float>(0,~uint64_t(0));
                    vbrcb((__ubuf__ uint32_t*)weights.GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),1,8,4);
                    SetVectorMask<float>(0,uint64_t(0xffffffffu));PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[(j+1)*p+origin],pivot,weights[(j+1-origin)*8],uint64_t(32),uint8_t(n-(j)-1),BinaryRepeatParams{1,1,0,uint8_t(p/8),0,1});
                    PipeBarrier<PIPE_V>();
                }
                }

    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    return tinyValue*TerminalDet<P,true>(mat,n-4,4,&condition);
}

template<uint32_t Block=64,uint32_t Family=0,uint32_t StorageBlock=Block> class BlockDet {
    using F=MatmulType<TPosition::GM,CubeFormat::ND,float>;
    using A=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;
    static constexpr auto cubeConfig=GetMatmulApiTiling<A,F,F,F>(BlockConfiguration<Block>());
    template<int Role,uint32_t Width=0,uint32_t Pitch=72> static __aicore__ inline float Factor(LocalTensor<float> mat,LocalTensor<float> side,LocalTensor<float> col,LocalTensor<float> pivot,LocalTensor<float> weights,LocalTensor<float> upper,LocalTensor<uint32_t> ix,uint32_t liveWidth,uint32_t own,float &condition){
        const uint32_t width=Width?Width:liveWidth;
        constexpr uint32_t B=Block,P=Pitch;
        if constexpr(Role==0 && B==64 && Family==1){
            return FinalResidentFactor<64,P>(mat,col,pivot,weights,ix,64,condition);
        }
        if constexpr(Role==0 && B==64 && Family>1){
            return FinalResidentFactor<Family,P>(mat,col,pivot,weights,ix,Family,condition);
        }
        if constexpr(Role==0 && B==64){SetMaskNorm();SetVectorMask<float>(0,~uint64_t(0));}
        SetFlag<HardEvent::V_S>(0);
        float determinant=1.0f;const uint32_t steps=Role==0 && width>=4?width-4:width;
        uint32_t q=0;
        for(;q+1<steps;q+=2){
            #pragma clang fp contract(off)
            WaitFlag<HardEvent::V_S>(0);
            float d0=mat.GetValue(q*P+q),a01=mat.GetValue(q*P+q+1);
            float a10=mat.GetValue((q+1)*P+q),a11=mat.GetValue((q+1)*P+q+1);
            float inv0=-1.0f/d0;determinant*=d0;

            if constexpr(Role==1)Muls(side[(q)*P],side[(q)*P],inv0,own);
            if((q)+1<width){
                const uint32_t first=(Role==0 && B==64)?0:(q)/8*8,active=B-first;
                if constexpr(Role==0 && B==64){
                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[q].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot[first],mat[(q)*P+first],inv0,uint64_t(active),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                    vbrcb((__ubuf__ uint32_t*)mat[B].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),P/8,P,B/8);PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[((q)+1)*P+first],pivot[first],mat[((q)+1)*P+B],uint64_t(active),uint8_t(width-(q)-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                }else{
                Gather(col,mat[q],ix,0,B);Muls(pivot[first],mat[(q)*P+first],inv0,uint64_t(active),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                Brcb(mat[B],col,B/8,BrcbRepeatParams{P/8,P});if constexpr(Role==2)Brcb(side[B],pivot,B/8,BrcbRepeatParams{P/8,P});PipeBarrier<PIPE_V>();
                MulAddDst(mat[((q)+1)*P+first],pivot[first],mat[((q)+1)*P+B],uint64_t(active),uint8_t(width-(q)-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                }
                
                if constexpr(Role)MulAddDst(side[((q)+1)*P],side[(q)*P],(Role==2?side[((q)+1)*P+B]:mat[((q)+1)*P+B]),uint64_t(own),uint8_t(width-(q)-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                PipeBarrier<PIPE_V>();
            }

            float u01=a01*inv0,correction=a10*u01;
            float d1=a11+correction,inv1=-1.0f/d1;determinant*=d1;

            if constexpr(Role==1)Muls(side[(q+1)*P],side[(q+1)*P],inv1,own);
            if((q+1)+1<width){
                const uint32_t first=(Role==0 && B==64)?0:(q+1)/8*8,active=B-first;
                if constexpr(Role==0 && B==64){
                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[q+1].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot[first],mat[(q+1)*P+first],inv1,uint64_t(active),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                    vbrcb((__ubuf__ uint32_t*)mat[B].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),P/8,P,B/8);PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[((q+1)+1)*P+first],pivot[first],mat[((q+1)+1)*P+B],uint64_t(active),uint8_t(width-(q+1)-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                }else{
                Gather(col,mat[q+1],ix,0,B);Muls(pivot[first],mat[(q+1)*P+first],inv1,uint64_t(active),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                Brcb(mat[B],col,B/8,BrcbRepeatParams{P/8,P});if constexpr(Role==2)Brcb(side[B],pivot,B/8,BrcbRepeatParams{P/8,P});PipeBarrier<PIPE_V>();
                MulAddDst(mat[((q+1)+1)*P+first],pivot[first],mat[((q+1)+1)*P+B],uint64_t(active),uint8_t(width-(q+1)-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                }
                SetFlag<HardEvent::V_S>(0);
                if constexpr(Role)MulAddDst(side[((q+1)+1)*P],side[(q+1)*P],(Role==2?side[((q+1)+1)*P+B]:mat[((q+1)+1)*P+B]),uint64_t(own),uint8_t(width-(q+1)-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                PipeBarrier<PIPE_V>();
            }
        }
        if(q<steps){
            WaitFlag<HardEvent::V_S>(0);float d=mat.GetValue(q*P+q),inverse=-1.0f/d;determinant*=d;

            if constexpr(Role==1)Muls(side[(q)*P],side[(q)*P],inverse,own);
            if((q)+1<width){
                const uint32_t first=(Role==0 && B==64)?0:(q)/8*8,active=B-first;
                if constexpr(Role==0 && B==64){
                    vgather((__ubuf__ uint32_t*)col.GetPhyAddr(),(__ubuf__ uint32_t*)ix.GetPhyAddr(),uint32_t((uint64_t)mat[q].GetPhyAddr()),8,1);
                    Muls<float,false>(pivot[first],mat[(q)*P+first],inverse,uint64_t(active),uint8_t(1),UnaryRepeatParams{1,1,8,8});
                    vbrcb((__ubuf__ uint32_t*)mat[B].GetPhyAddr(),(__ubuf__ uint32_t*)col.GetPhyAddr(),P/8,P,B/8);PipeBarrier<PIPE_V>();
                    MulAddDst<float,float,false>(mat[((q)+1)*P+first],pivot[first],mat[((q)+1)*P+B],uint64_t(active),uint8_t(width-(q)-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                }else{
                Gather(col,mat[q],ix,0,B);Muls(pivot[first],mat[(q)*P+first],inverse,uint64_t(active),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                Brcb(mat[B],col,B/8,BrcbRepeatParams{P/8,P});if constexpr(Role==2)Brcb(side[B],pivot,B/8,BrcbRepeatParams{P/8,P});PipeBarrier<PIPE_V>();
                MulAddDst(mat[((q)+1)*P+first],pivot[first],mat[((q)+1)*P+B],uint64_t(active),uint8_t(width-(q)-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                }
                SetFlag<HardEvent::V_S>(0);
                if constexpr(Role)MulAddDst(side[((q)+1)*P],side[(q)*P],(Role==2?side[((q)+1)*P+B]:mat[((q)+1)*P+B]),uint64_t(own),uint8_t(width-(q)-1),BinaryRepeatParams{1,1,0,P/8,0,P/8});
                PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(Role==0){if(width>=4){WaitFlag<HardEvent::V_S>(0);determinant*=TerminalDet<P,true>(mat,width-4,4,&condition);}}
        return determinant;
    }
    static __aicore__ inline void LoadStoredTile(LocalTensor<float> packed,
        GlobalTensor<float> matrix,uint64_t offset){
        constexpr uint32_t S=StorageBlock;
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(packed,matrix[offset],DataCopyExtParams{1,S*S*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0f});
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    }
    static __aicore__ inline void RunStoredCube(GM_ADDR a,GM_ADDR workspace,const LinalgDetTilingData &t){
        constexpr uint32_t S=StorageBlock,K=Block;
        const uint32_t G=t.team*S,W=Family;
        const uint32_t id=GetBlockIdx(),group=id/t.team,lane=id%t.team;
        GlobalTensor<float> input,matrix,left,right;
        input.SetGlobalBuffer((__gm__ float*)a);matrix.SetGlobalBuffer((__gm__ float*)workspace);
        left.SetGlobalBuffer((__gm__ float*)workspace+t.groups*G*G);
        right.SetGlobalBuffer((__gm__ float*)workspace+t.groups*(G*G+K*G));
        TPipe pipe;TBuf<TPosition::A1> l1;TBuf<TPosition::A2> a2;
        TBuf<TPosition::B2> b2;TBuf<TPosition::CO1> c0;
        const uint32_t stagingWords=K*G>2*K*K?K*G:2*K*K;
        pipe.InitBuffer(l1,stagingWords*4);pipe.InitBuffer(a2,K*K*4);
        pipe.InitBuffer(b2,K*K*4);pipe.InitBuffer(c0,K*K*4);
        auto stage=l1.Get<float>(),lhs=stage,rhs=stage[K*K];
        auto la=a2.Get<float>(),rb=b2.Get<float>(),acc=c0.Get<float>();
        SetHF32Mode(false);SetHF32TransMode(false);
        for(uint32_t batch=group;batch<t.batches;batch+=t.groups){
            InitConstValue(stage,InitConstValueParams<float>{1,uint16_t(K*G/8),0,0.0f});PipeBarrier<PIPE_MTE2>();
            for(uint32_t columnBlock=0;columnBlock<t.team;++columnBlock){
                Nd2NzParams q{};q.ndNum=1;q.nValue=W;q.dValue=W;q.srcDValue=t.n;q.dstNzC0Stride=K;q.dstNzNStride=1;
                DataCopy(stage[columnBlock*S*K],input[uint64_t(batch)*t.n*t.n+lane*W*t.n+columnBlock*W],q);
            }
            SetFlag<HardEvent::MTE2_MTE3>(0);WaitFlag<HardEvent::MTE2_MTE3>(0);
            for(uint32_t columnBlock=0;columnBlock<t.team;++columnBlock){
                const uint32_t tile=group*G*G+(lane*t.team+columnBlock)*S*S;
                for(uint32_t column=0;column<S;column+=8)
                    DataCopy(matrix[tile+column],stage[columnBlock*S*K+column*K],DataCopyParams{S,1,0,S/8-1});
            }
            SetFlag<HardEvent::MTE3_S>(0);WaitFlag<HardEvent::MTE3_S>(0);
            const uint32_t diagonal=group*G*G+(lane*t.team+lane)*S*S;
            for(uint32_t row=W;row<S;++row){
                matrix.SetValue(diagonal+row*(S+1),1.0f);
                DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(matrix[diagonal+row*(S+1)]);
            }
            PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_MTE3>(3);
            // All later loads have the same64xS shape. L1 columns S..63 stay zero.
            InitConstValue(stage,InitConstValueParams<float>{1,uint16_t(2*K*K/8),0,0.0f});PipeBarrier<PIPE_MTE2>();
            for(uint32_t panel=0;panel+1<t.team;++panel){
                CrossCoreWaitFlag<2>(0);
                uint32_t lastRow=0xffffffffu,lastColumn=0xffffffffu;
                const uint32_t tiles=t.team-panel-1,critical=2*tiles-1;
                bool sent=lane>=critical;if(sent)CrossCoreSetFlag<2,PIPE_FIX>(2);
                for(uint32_t tile=lane;tile<tiles*tiles;tile+=t.team){
                    uint32_t rowTile,colTile;
                    if(tile<tiles){rowTile=0;colTile=tile;}
                    else if(tile<critical){rowTile=tile-tiles+1;colTile=0;}
                    else{uint32_t inner=tile-critical;rowTile=1+inner/(tiles-1);colTile=1+inner%(tiles-1);}
                    const uint32_t rowBlock=panel+1+rowTile,columnBlock=panel+1+colTile;
                    Nd2NzParams q{};q.ndNum=1;q.nValue=K;q.dValue=S;q.srcDValue=S;q.dstNzC0Stride=K;q.dstNzNStride=1;
                    const bool loadLeft=rowBlock!=lastRow,loadRight=columnBlock!=lastColumn;
                    if(tile==lane){
                        DataCopy(lhs,left[group*K*G+rowBlock*K*S],q);
                        DataCopy(rhs,right[group*K*G+columnBlock*K*S],q);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
                    uint8_t padding[4]={};SetFmatrix(1,K,padding,FmatrixMode::FMATRIX_LEFT);
                    LoadData3DParamsV2Pro ld{};ld.channelSize=K;ld.extConfig=(uint64_t(K)<<16)|K;
                    if(loadLeft){ld.enTranspose=true;LoadData<float>(la,lhs,ld);}
                    if(loadRight){ld.enTranspose=false;LoadData<float>(rb,rhs,ld);}
                    lastRow=rowBlock;lastColumn=columnBlock;
                    SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
                    MmadParams m{};m.m=K;m.n=K;m.k=K;m.cmatrixInitVal=true;Mmad(acc,la,rb,m);
                    const uint32_t nextTile=tile+t.team;
                    if(nextTile<tiles*tiles){
                        uint32_t nr,nc;
                        if(nextTile<tiles){nr=0;nc=nextTile;}
                        else if(nextTile<critical){nr=nextTile-tiles+1;nc=0;}
                        else{uint32_t inner=nextTile-critical;nr=1+inner/(tiles-1);nc=1+inner%(tiles-1);}
                        nr+=panel+1;nc+=panel+1;
                        // Current MMAD reads L0; L1 is reusable after its L0 loads.
                        SetFlag<HardEvent::MTE1_MTE2>(0);WaitFlag<HardEvent::MTE1_MTE2>(0);
                        if(nr!=rowBlock)DataCopy(lhs,left[group*K*G+nr*K*S],q);
                        if(nc!=columnBlock)DataCopy(rhs,right[group*K*G+nc*K*S],q);
                    }
                    SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
                    FixpipeParamsV220 f{};f.mSize=S;f.nSize=S;f.srcStride=K;f.dstStride=S;
                    SetAtomicAdd<float>();Fixpipe<float,float,CFG_ROW_MAJOR>(matrix[group*G*G+(rowBlock*t.team+columnBlock)*S*S],acc,f);
                    PipeBarrier<PIPE_ALL>();SetAtomicNone();
                    if(!sent && tile+t.team>=critical){CrossCoreSetFlag<2,PIPE_FIX>(2);sent=true;}
                }
                CrossCoreSetFlag<2,PIPE_FIX>(1);
            }
        }
    }

    static __aicore__ inline bool RunVector(GM_ADDR a,GM_ADDR y,GM_ADDR workspace,const LinalgDetTilingData &t){
        constexpr uint32_t B=Block,P=136,S=StorageBlock;
        bool retryAny=false;
        const uint32_t n=t.n,G=(n+StorageBlock-1)/StorageBlock*StorageBlock,r=B,W=Family>1?Family:(Family==1?Block:(Block<64?Block:(n+G/Block-1)/(G/Block)));
        const bool remappedStage=Family>1 || (Family==0 && Block==64 && n!=G && t.groups==t.batches);
        const bool directStage=Family!=0 || ((n==G || remappedStage) && t.groups==t.batches);
        const bool alignedStage=Family==0?n==G:Family==1;
        GlobalTensor<float> input,output,matrix,left,right;
        input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);
        matrix.SetGlobalBuffer((__gm__ float*)workspace);
        left.SetGlobalBuffer((__gm__ float*)workspace+t.groups*G*G);
        right.SetGlobalBuffer((__gm__ float*)workspace+t.groups*(G*G+B*G));

            TPipe pipe;TBuf<TPosition::VECCALC> buf;
            constexpr uint32_t panelWords=(2*B+64)*P+15*B+1024;
            constexpr uint32_t stagingWords=32*((512+B-1)/B*B);
            constexpr uint32_t words=panelWords>stagingWords?panelWords:stagingWords;
            static_assert(words*4<=192*1024);
            pipe.InitBuffer(buf,words*4);auto mat=buf.Get<float>(),separate=mat[B*P],raw=separate[B*P],weights=raw[64*P],upper=weights[8*r],col=upper[8*B],pivot=col[r],diag=pivot[B],temp=diag[r],signs=temp[r],sc=signs[r];auto ix=sc[128].ReinterpretCast<uint32_t>(),dix=ix[r];auto side=mat[B];
            const uint32_t id=GetBlockIdx(),group=id/(2*t.team),lane=id%(2*t.team);
            uint64_t src[4][16],dst[4][16];
            if(lane&1)for(uint32_t g=0;g<4;++g){for(uint32_t q=0;q<16;++q)src[g][q]=(uint64_t)raw[(g*16+q)*P].GetPhyAddr();for(uint32_t q=0;q<8;++q){dst[g][2*q]=(uint64_t)side[q*P+g*16].GetPhyAddr();dst[g][2*q+1]=(uint64_t)side[q*P+g*16+8].GetPhyAddr();}}

            uint64_t diagSrc[4][16],diagDst[4][16];
            if(lane&1)for(uint32_t g=0;g<4;++g){
                for(uint32_t q=0;q<16;++q)diagSrc[g][q]=(uint64_t)separate[(g*16+q)*P].GetPhyAddr();
                for(uint32_t q=0;q<8;++q){diagDst[g][2*q]=(uint64_t)mat[q*P+g*16].GetPhyAddr();diagDst[g][2*q+1]=(uint64_t)mat[q*P+g*16+8].GetPhyAddr();}
            }
            TransDataTo5HDParams transpose;transpose.dstHighHalf=false;transpose.srcHighHalf=false;transpose.repeatTimes=B/8;transpose.dstRepStride=P;transpose.srcRepStride=1;
            for(uint32_t batch=group;batch<t.batches;batch+=t.groups){
                float value=1.0f,condition=1.0f;
                if(!alignedStage && !remappedStage && lane*32<G){
                const uint32_t row=lane*32,logicalRow=Block<64?row:((row/64)*W+row%64);
                const uint32_t liveRows=Block<64?(row<n?(n-row<32?n-row:32):0):(row%64<W && logicalRow<n)?(n-logicalRow<32?n-logicalRow:(W-row%64<32?W-row%64:32)):0;
                Duplicate(mat,0.0f,32*G);SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
                if constexpr(Block<64){if(liveRows)DataCopyPad(mat,input[(uint64_t)batch*n*n+row*n],DataCopyExtParams{uint16_t(liveRows),n*4,0,uint32_t((G-(n+7)/8*8)/8),0},DataCopyPadExtParams<float>{true,0,uint8_t((8-n%8)%8),0.0f});}
                else if(liveRows)for(uint32_t block=0;block<G/64;++block){
                    const uint32_t column=block*W,liveColumns=n-column<W?n-column:W;
                    DataCopyPad(mat[block*64],input[(uint64_t)batch*n*n+logicalRow*n+column],DataCopyExtParams{uint16_t(liveRows),liveColumns*4,(n-liveColumns)*4,uint32_t((G-(liveColumns+7)/8*8)/8),0},DataCopyPadExtParams<float>{true,0,uint8_t((8-liveColumns%8)%8),0.0f});
                }
                PipeBarrier<PIPE_ALL>();for(uint32_t q=liveRows;q<32;++q)mat.SetValue(q*G+row+q,1.0f);
                SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);DataCopy(matrix[group*G*G+row*G],mat,32*G);
                }
                if(!directStage)SyncAll<false>();
                CreateVecIndex(ix.ReinterpretCast<int32_t>(),int32_t(0),B);PipeBarrier<PIPE_V>();Muls(ix.ReinterpretCast<int32_t>(),ix.ReinterpretCast<int32_t>(),int32_t(P*4),B);PipeBarrier<PIPE_V>();
                Duplicate(diag,1.0f,r);PipeBarrier<PIPE_ALL>();
                for(uint32_t start=0;start<G;start+=S){
                    if(start){CrossCoreWaitFlag<2>(2);SyncAll<true>();}
                    const uint32_t logicalStart=start/S*W,width=Family>1?Family:(Family==1?Block:(n-logicalStart<W?n-logicalStart:W)),remain=G-start-S;
                    const uint32_t shard=StorageBlock<64?S:(remain+t.team*8-1)/(t.team*8)*8;
                    const uint32_t first=start+S+(lane/2)*shard;
                    const uint32_t own=StorageBlock<64?(first<G?S:0):(first<G?(G-first<shard?G-first:shard):0);
                    const bool direct=directStage && start==0;
                    const uint64_t sourceBase=(direct?uint64_t(batch):uint64_t(group))*G*G;
                    if(own || lane==0){
                        if(direct && remappedStage){
                            const uint64_t base=(uint64_t)batch*n*n;
                            Duplicate(mat,0.0f,3*B*P);
                            SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
                            DataCopyPad((lane&1)?separate:mat,input[base],DataCopyExtParams{uint16_t(W),W*4,(n-W)*4,uint32_t((P-(W+7)/8*8)/8),0},DataCopyPadExtParams<float>{true,0,uint8_t((8-W%8)%8),0.0f});
                            for(uint32_t consumed=0;consumed<own;){
                                const uint32_t physical=first+consumed,block=physical/S,offset=physical%S;
                                const uint32_t span=own-consumed<S-offset?own-consumed:S-offset;
                                const uint32_t logical=block*W+offset,capacity=n-block*W<W?n-block*W:W;
                                const uint32_t live=offset<capacity?(capacity-offset<span?capacity-offset:span):0;
                                if(live){
                                    if((lane&1)==0)DataCopyPad(side[consumed],input[base+logical],DataCopyExtParams{uint16_t(W),live*4,(n-live)*4,uint32_t((P-(live+7)/8*8)/8),0},DataCopyPadExtParams<float>{true,0,uint8_t((8-live%8)%8),0.0f});
                                    else DataCopyPad(raw[consumed*P],input[base+logical*n],DataCopyExtParams{uint16_t(live),W*4,(n-W)*4,uint32_t((P-(W+7)/8*8)/8),0},DataCopyPadExtParams<float>{true,0,uint8_t((8-W%8)%8),0.0f});
                                }
                                consumed+=span;
                            }
                        }else if constexpr(StorageBlock<64){
                            const uint32_t panel=start/S;
                            auto packed=(lane&1)?mat:separate;
                            LoadStoredTile(packed,matrix,uint64_t(group)*G*G+(panel*t.team+panel)*S*S);
                            DataCopy((lane&1)?separate:mat,packed,DataCopyParams{S,S/8,0,(P-S)/8});PipeBarrier<PIPE_V>();
                            if(own){
                                const uint32_t block=panel+1+lane/2;
                                const uint32_t rowBlock=(lane&1)?block:panel,columnBlock=(lane&1)?panel:block;
                                LoadStoredTile(packed,matrix,uint64_t(group)*G*G+(rowBlock*t.team+columnBlock)*S*S);
                                DataCopy((lane&1)?raw:side,packed,DataCopyParams{S,S/8,0,(P-S)/8});PipeBarrier<PIPE_V>();
                            }
                        }else{
                        DataCopyPad((lane&1)?separate:mat,(direct?input:matrix)[sourceBase+start*G+start],DataCopyExtParams{S,S*4,(G-S)*4,(P-S)/8,0},DataCopyPadExtParams<float>{false,0,0,0.0f});
                        if(own){if((lane&1)==0)DataCopyPad(side,(direct?input:matrix)[sourceBase+start*G+first],DataCopyExtParams{S,own*4,(G-own)*4,(P-own)/8,0},DataCopyPadExtParams<float>{false,0,0,0.0f});
                            else DataCopyPad(raw,(direct?input:matrix)[sourceBase+first*G+start],DataCopyExtParams{uint16_t(own),S*4,(G-S)*4,(P-S)/8,0},DataCopyPadExtParams<float>{false,0,0,0.0f});}
                        }
                        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
                        if(own && (lane&1)){for(uint32_t g=0;g*16<own;++g)TransDataTo5HD<float>(dst[g],src[g],transpose);for(uint32_t g=0;g<4;++g)TransDataTo5HD<float>(diagDst[g],diagSrc[g],transpose);PipeBarrier<PIPE_V>();}
                        if constexpr(Family>1){
                            if(!own)value*=Factor<0,Family,P>(mat,side,col,pivot,weights,upper,ix,width,own,condition);
                            else if(lane&1)value*=AugBlockFactor<Family,true>(mat,col,ix,width,own);
                            else value*=AugBlockFactor<Family,false>(mat,col,ix,width,own);
                        }else if(width==64){
                            if(!own)value*=Factor<0,64,P>(mat,side,col,pivot,weights,upper,ix,width,own,condition);
                            else if(lane&1)value*=AugBlockFactor<64,true>(mat,col,ix,width,own);
                            else value*=AugBlockFactor<64,false>(mat,col,ix,width,own);
                        }else{
                            if(!own)value*=Factor<0,0,P>(mat,side,col,pivot,weights,upper,ix,width,own,condition);
                            else if(lane&1)value*=AugBlockFactor<0,true>(mat,col,ix,width,own);
                            else value*=AugBlockFactor<0,false>(mat,col,ix,width,own);
                        }
                    }
                    // The previous product may still consume these panels.
                    if(start)CrossCoreWaitFlag<2>(1);
                    if(!remain)break;
                    if(own){
                        if constexpr(StorageBlock<64){
                            // The old input diagonal is dead; this buffer is disjoint from side.
                            PipeBarrier<PIPE_V>();
                            DataCopy(separate,side,DataCopyParams{B,S/8,(P-S)/8,0});PipeBarrier<PIPE_V>();
                            SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
                            DataCopyPad((lane&1?left:right)[group*B*G+(start/S+1+lane/2)*B*S],separate,
                                DataCopyExtParams{1,B*S*4,0,0,0});
                        }else{
                            SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
                            DataCopyPad((lane&1?left:right)[group*B*G+first],side,DataCopyExtParams{B,own*4,(P-own)/8,(G-own)*4,0});
                        }
                    }
                    if(direct)CrossCoreWaitFlag<2>(3);
                    SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(0);
                }
                if(lane==0){
                    const bool risky=!(condition>1.0e-4f) || DetAbnormal(value,false);retryAny|=risky;
                    union{uint32_t u;float f;} marker;marker.u=0x7fc00000u;
                    sc.SetValue(0,risky?marker.f:value);SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);DataCopyPad(output[batch],sc,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();}
            }
            return retryAny;
    }
public:
    static __aicore__ inline void Run(GM_ADDR a,GM_ADDR y,GM_ADDR workspace,const LinalgDetTilingData &t,GM_ADDR tiling,bool &retryAny){
        constexpr uint32_t B=StorageBlock,P=72,K=Block;
        const uint32_t n=t.n,G=(n+StorageBlock-1)/StorageBlock*StorageBlock,r=t.rows,W=Family>1?Family:(Family==1?Block:(Block<64?Block:(n+G/Block-1)/(G/Block)));
        const bool remappedStage=Family>1 || (Family==0 && Block==64 && n!=G && t.groups==t.batches);
        const bool directStage=Family!=0 || ((n==G || remappedStage) && t.groups==t.batches);
        const bool alignedStage=Family==0?n==G:Family==1;
        GlobalTensor<float> input,output,matrix,left,right;
        input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);
        matrix.SetGlobalBuffer((__gm__ float*)workspace);
        left.SetGlobalBuffer((__gm__ float*)workspace+t.groups*G*G);
        right.SetGlobalBuffer((__gm__ float*)workspace+t.groups*(G*G+K*G));
        if ASCEND_IS_AIC {
            if constexpr(StorageBlock<64){RunStoredCube(a,workspace,t);}
            else{
            TPipe pipe;Matmul<A,F,F,F,cubeConfig> mm;
            REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,(AscendC::tiling::TCubeTiling*)nullptr);
            mm.SetHF32(false);mm.SetOrgShape(G,G,K);mm.SetTail(64,64,K);
            const uint32_t id=GetBlockIdx(),group=id/t.team,lane=id%t.team;
            for(uint32_t batch=group;batch<t.batches;batch+=t.groups){
                if(alignedStage){
                    LocalTensor<float> staging(TPosition::A1,0,64*G);uint32_t first=lane*64;
                    if(first<G)DataCopy(staging,input[(uint64_t)batch*n*n+first*G],(G-first<64?G-first:64)*G);SetFlag<HardEvent::MTE2_MTE3>(0);WaitFlag<HardEvent::MTE2_MTE3>(0);
                    if(first<G)DataCopy(matrix[group*G*G+first*G],staging,(G-first<64?G-first:64)*G);PipeBarrier<PIPE_ALL>();
                    if(directStage)CrossCoreSetFlag<2,PIPE_MTE3>(3);
                }else if(remappedStage){
                    LocalTensor<float> staging(TPosition::A1,0,Block*G);
                    const uint32_t logicalRow=lane*W,liveRows=n-logicalRow<W?n-logicalRow:W;
                    InitConstValue(staging,InitConstValueParams<float>{1,uint16_t(Block*G/8),0,0.0f});
                    PipeBarrier<PIPE_MTE2>();
                    for(uint32_t block=0;block<G/B;++block){
                        const uint32_t column=block*W,liveColumns=n-column<W?n-column:W;
                        Nd2NzParams q{};q.ndNum=1;q.nValue=liveRows;q.dValue=liveColumns;q.srcDValue=n;q.dstNzC0Stride=1;q.dstNzNStride=G/8;
                        DataCopy(staging[block*B],input[(uint64_t)batch*n*n+logicalRow*n+column],q);
                    }
                    SetFlag<HardEvent::MTE2_MTE3>(0);WaitFlag<HardEvent::MTE2_MTE3>(0);
                    DataCopy(matrix[group*G*G+lane*B*G],staging,B*G);
                    SetFlag<HardEvent::MTE3_S>(0);WaitFlag<HardEvent::MTE3_S>(0);
                    for(uint32_t row=liveRows;row<B;++row){
                        const uint32_t offset=group*G*G+(lane*B+row)*(G+1);
                        matrix.SetValue(offset,1.0f);
                        DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(matrix[offset]);
                    }
                    PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_MTE3>(3);
                }
                if(!directStage)SyncAll<false>();
                for(uint32_t start=0;start+B<G;start+=B){
                    CrossCoreWaitFlag<2>(0);
                    const uint32_t first=start+B,tiles=(G-first+63)/64,critical=2*tiles-1;
                    bool sent=lane>=critical;if(sent)CrossCoreSetFlag<2,PIPE_FIX>(2);
                    for(uint32_t tile=lane;tile<tiles*tiles;tile+=t.team){
                        uint32_t rowTile,colTile;
                        if(tile<tiles){rowTile=0;colTile=tile;}
                        else if(tile<critical){rowTile=tile-tiles+1;colTile=0;}
                        else{uint32_t inner=tile-critical;rowTile=1+inner/(tiles-1);colTile=1+inner%(tiles-1);}
                        uint32_t row=first+rowTile*64,col=first+colTile*64;
                        mm.SetTail(G-row<64?G-row:64,G-col<64?G-col:64,K);
                        mm.SetTensorA(left[group*K*G+row],true);mm.SetTensorB(right[group*K*G+col]);
                        mm.IterateAll(matrix[group*G*G+row*G+col],1);mm.End();
                        if(!sent && tile+t.team>=critical){CrossCoreSetFlag<2,PIPE_FIX>(2);sent=true;}
                    }
                    CrossCoreSetFlag<2,PIPE_FIX>(1);
                }
            }
            }
        }
        if ASCEND_IS_AIV {
            retryAny=RunVector(a,y,workspace,t);
        }
    }
};

class DetKernel {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    LocalTensor<float> matrix,column,weights,pivot,candidates,active,scalar,scratch;
    LocalTensor<uint32_t> offsets;
    GlobalTensor<float> input,output,factors;
    uint32_t chosen[512],bits[16];
    __aicore__ inline int Exponent(float x) {
        union {float f;uint32_t u;} v;v.f=x;return int((v.u>>23)&255)-127;
    }
    __aicore__ inline float Power(int e) {
        union {float f;uint32_t u;} v;v.u=uint32_t(e+127)<<23;return v.f;
    }
    __aicore__ inline void Update(uint32_t n,uint32_t width,uint32_t j,uint32_t firstColumn) {
        Brcb(weights,column,(n+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        for(uint32_t start=j+1;start<n;start+=255) {
            uint8_t rows=n-start<255?n-start:255;
            for(uint32_t c=firstColumn;c<width;c+=64)
                MulAddDst(matrix[start*width+c],pivot[c],weights[start*8],
                    (uint64_t)(width-c<64?width-c:64),rows,
                    BinaryRepeatParams{1,1,0,(uint8_t)(width/8),0,1});
        }
        PipeBarrier<PIPE_V>();
    }
public:
    template<bool Recovery=false> __aicore__ inline void Run(GM_ADDR a,GM_ADDR out,GM_ADDR workspace,const LinalgDetTilingData &t,
        uint32_t batchStart=0,uint32_t batchStep=1,bool force=false) {
        const uint32_t n=t.n,padded=(n+7)/8*8,width=t.panel;
        // Matrix + broadcast coefficients + six vectors + index/reduction area.
        const uint32_t words=n*width+8*padded+6*padded+width+128;
        pipe.InitBuffer(buffer,words*4);auto arena=buffer.Get<float>();
        matrix=arena;column=matrix[n*width];weights=column[padded];pivot=weights[8*padded];
        candidates=pivot[width];active=candidates[padded];scalar=active[padded];
        scratch=scalar[128];offsets=scratch[2*padded].ReinterpretCast<uint32_t>();
        input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)out);
        factors.SetGlobalBuffer((__gm__ float*)workspace+uint64_t(Recovery?0:GetBlockIdx())*n*padded);
        for(uint32_t batch=Recovery?batchStart:GetBlockIdx();batch<t.batches;batch+=Recovery?batchStep:t.blocks) {
            if constexpr(Recovery){if(!force){
                DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(output[batch]);
                union{float f;uint32_t u;} marker;marker.f=output.GetValue(batch);
                if((marker.u&0x7fffffffu)<=0x7f800000u)continue;
            }}
            for(uint32_t i=0;i<16;++i)bits[i]=0;
            for(uint32_t r=0;r<padded;++r)offsets.SetValue(r,r<n?r*width*4:0);
            Duplicate(active.ReinterpretCast<uint32_t>(),uint32_t(0x7fffffff),padded);PipeBarrier<PIPE_ALL>();
            float determinant=1.0f;int exponent=0;bool singular=false;
            for(uint32_t first=0;first<n;first+=width) {
                const uint32_t live=n-first<width?n-first:width;
                DataCopyPad(matrix,input[(uint64_t)batch*n*n+first],
                    DataCopyExtParams{(uint16_t)n,live*4,(n-live)*4,(width-(live+7)/8*8)/8,0},
                    DataCopyPadExtParams<float>{true,0,(uint8_t)((8-live%8)%8),0.0f});
                PipeBarrier<PIPE_ALL>();
                // Replaying earlier panels is a triangular solve and Schur
                // update, using exactly the same pivot permutation and L.
                for(uint32_t j=0;j<first;++j) {
                    DataCopy(column,factors[(uint64_t)j*padded],padded);PipeBarrier<PIPE_ALL>();
                    DataCopy(pivot,matrix[chosen[j]*width],width);PipeBarrier<PIPE_V>();
                    if(chosen[j]!=j)DataCopy(matrix[chosen[j]*width],matrix[j*width],width);
                    PipeBarrier<PIPE_V>();Update(n,width,j,0);
                }
                for(uint32_t k=0;k<live;++k) {
                    uint32_t j=first+k;
                    Gather(column,matrix[k],offsets,0,n);PipeBarrier<PIPE_V>();
                    Abs(candidates,column,n);PipeBarrier<PIPE_V>();
                    if(j)Duplicate(candidates,0.0f,j);
                    PipeBarrier<PIPE_V>();
                    SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,n);
                    WholeReduceMax<float,false>(scalar,candidates,MASK_PLACEHOLDER,1,1,1,8);SetMaskNorm();ResetMask();
                    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                    uint32_t bestBits,selected;GetReduceRepeatMaxMinSpr<float>(bestBits,selected);
                    if(!bestBits || selected>=n){singular=true;break;}
                    chosen[j]=selected;
                    if(selected!=j)determinant=-determinant;
                    float diagonal=column.GetValue(selected);int e=Exponent(diagonal);e=e>126?126:e<-126?-126:e;
                    determinant*=diagonal*Power(-e);exponent+=e;
                    e=Exponent(determinant);determinant*=Power(-e);exponent+=e;

                    if(j+1==n)break;
                    float inverse=-1.0f/diagonal;
                    if(selected!=j)column.SetValue(selected,column.GetValue(j));
                    SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
                    Muls(column,column,inverse,n);
                    DataCopy(pivot,matrix[selected*width],width);PipeBarrier<PIPE_V>();
                    if(selected!=j)DataCopy(matrix[selected*width],matrix[j*width],width);
                    PipeBarrier<PIPE_V>();
                    if(first+width<n) {
                        PipeBarrier<PIPE_ALL>();DataCopy(factors[(uint64_t)j*padded],column,padded);PipeBarrier<PIPE_ALL>();
                    }
                    Update(n,width,j,k/8*8);
                }
                if(singular)break;
            }
            if(singular)determinant=0.0f;
            else {
                while(exponent>126){determinant*=Power(126);exponent-=126;}
                while(exponent<-126){determinant*=Power(-126);exponent+=126;}
                determinant*=Power(exponent);
            }
            PipeBarrier<PIPE_ALL>();scalar.SetValue(0,determinant);PipeBarrier<PIPE_ALL>();
            DataCopyPad(output[batch],scalar,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
};

static __aicore__ __attribute__((noinline)) void CorrectnessRecovery(GM_ADDR a,GM_ADDR out,GM_ADDR factors,
    const LinalgDetTilingData &t,uint32_t group){
    DetKernel pivot;pivot.Run<true>(a,out,factors,t,group,t.groups);
}

static __aicore__ __attribute__((noinline)) void PivotedFallback(GM_ADDR a,GM_ADDR out,GM_ADDR workspace,GM_ADDR tiling,uint32_t batch){
    LinalgDetTilingData t{};
    const __gm__ uint32_t *words=(__gm__ uint32_t*)tiling;
    t.n=words[0];t.rows=words[1];t.batches=words[2];t.blocks=words[3];t.panel=words[4];t.team=words[5];t.groups=words[6];
    DetKernel pivot;pivot.Run<true>(a,out,GetUserWorkspace(workspace),t,batch,t.batches,true);
}

template<typename DT_A,uint32_t KERNEL_PATH,uint32_t ORDER> __global__ __aicore__ void linalg_det(GM_ADDR A,GM_ADDR det,GM_ADDR workspace,GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);InitSocState();
    REGISTER_TILING_DEFAULT(LinalgDetMetadata);
    LinalgDetTilingData t{};
    if constexpr(ORDER && KERNEL_PATH==3){
        t.n=ORDER;t.rows=16;t.blocks=GetBlockNum();
        t.batches=((__gm__ uint32_t*)tiling)[2];
        if(GetBlockIdx()>=t.batches)return;
    }
    else if constexpr(ORDER && (KERNEL_PATH==5 || KERNEL_PATH==7)){
        t.n=ORDER;t.blocks=GetBlockNum();t.batches=((__gm__ uint32_t*)tiling)[2];
        if(GetBlockIdx()>=t.batches)return;
    }
    else{
        const __gm__ uint32_t *words=(__gm__ uint32_t*)tiling;
        t.n=words[0];t.rows=words[1];t.batches=words[2];t.blocks=words[3];t.panel=words[4];t.team=words[5];t.groups=words[6];
    }
    if constexpr(KERNEL_PATH==3){TinyResident<ORDER> op;op.Run(A,det,t,workspace,tiling);}
    else if constexpr(KERNEL_PATH==4){ResidentDet<32> op;op.Run(A,det,t,tiling,workspace);}
    else if constexpr(KERNEL_PATH==5){
        if constexpr(ORDER && ORDER%8){ReverseResidentDet<ORDER,72> op;op.Run(A,det,t,workspace,tiling);}
        else{ResidentDet<64,ORDER> op;op.Run(A,det,t,tiling,workspace);}
    }
    else if constexpr(KERNEL_PATH==7){
        if constexpr(ORDER && ORDER%8){ReverseResidentDet<ORDER,104> op;op.Run(A,det,t,workspace,tiling);}
        else if constexpr(ORDER){BlockColumnResidentDet<ORDER,104,false> op;op.Run(A,det,t,workspace,tiling);}
        else{ResidentDet<64,ORDER,104> op;op.Run(A,det,t,tiling,workspace);}
    }
    else if constexpr(KERNEL_PATH==6){
        GM_ADDR user=GetUserWorkspace(workspace);bool retryAny=false;
        BlockDet<64,ORDER>::Run(A,det,user,t,tiling,retryAny);
        if ASCEND_IS_AIV {if(retryAny){
            constexpr uint32_t slot=64;
            const uint32_t group=GetBlockIdx()/(2*t.team),g=(t.n+slot-1)/slot*slot;
            CorrectnessRecovery(A,det,user+uint64_t(group)*g*g*sizeof(float),t,group);
        }}
    }
    else{DetKernel op;op.Run(A,det,GetUserWorkspace(workspace),t);}
}
