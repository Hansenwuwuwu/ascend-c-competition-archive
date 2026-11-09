#define TREE_NATIVE_FLOAT_TOKEN_float 1
#define TREE_NATIVE_FLOAT_TOKEN_half 0
#define TREE_NATIVE_FLOAT_TOKEN_IMPL(token) TREE_NATIVE_FLOAT_TOKEN_##token
#define TREE_NATIVE_FLOAT_TOKEN(token) TREE_NATIVE_FLOAT_TOKEN_IMPL(token)
#define TREE_NATIVE_HALF_TOKEN_float 0
#define TREE_NATIVE_HALF_TOKEN_half 1
#define TREE_NATIVE_HALF_TOKEN_IMPL(token) TREE_NATIVE_HALF_TOKEN_##token
#define TREE_NATIVE_HALF_TOKEN(token) TREE_NATIVE_HALF_TOKEN_IMPL(token)
#include "kernel_operator.h"

#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

using namespace AscendC;
#if defined(DTYPE_TREE_POINTS)
// All arithmetic payloads are sixteen-lane vector packets. Scalar loops
// traverse radix digits, axes and segment packets, never coordinate values.
class TreeFloatRational16 {
    LocalTensor<float> f_;
    uint32_t l_, baseExp_;
    static constexpr uint32_t V=16;
    __aicore__ inline LocalTensor<float> F(uint32_t p) { return f_[p*V]; }
    __aicore__ inline LocalTensor<float> S(uint32_t p) { return F(51*l_+p); }
    __aicore__ inline LocalTensor<uint8_t> M(uint32_t p) { return S(16).ReinterpretCast<uint8_t>()[p*32]; }
    __aicore__ inline void CmpScalar(LocalTensor<uint8_t> dst,LocalTensor<float> x,float value,CMPMODE mode) {
        CompareScalar<float,uint8_t,true>(dst,x,value,mode,uint64_t(V),uint8_t(1),UnaryRepeatParams(1,1,8,8));
    }
    __aicore__ inline void Cmp(LocalTensor<uint8_t> dst,LocalTensor<float> x,LocalTensor<float> y,CMPMODE mode) {
        Compare<float,uint8_t,true>(dst,x,y,mode,uint64_t(V),uint8_t(1),BinaryRepeatParams(1,1,1,8,8,8));
    }
    __aicore__ inline void Split(LocalTensor<float> dst) {
        Muls(S(4),S(0),1.0F/4096.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(3).ReinterpretCast<int32_t>(),S(4),RoundMode::CAST_FLOOR,V);PipeBarrier<PIPE_V>();
        Cast(S(1),S(3).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Muls(S(4),S(1),4096.0F,V);PipeBarrier<PIPE_V>();
        Sub(dst,S(0),S(4),V);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void MaskAnd(uint32_t d,uint32_t a,uint32_t b) {
        And(M(d).ReinterpretCast<uint16_t>(),M(a).ReinterpretCast<uint16_t>(),M(b).ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void MaskOr(uint32_t d,uint32_t a,uint32_t b) {
        Or(M(d).ReinterpretCast<uint16_t>(),M(a).ReinterpretCast<uint16_t>(),M(b).ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    }
public:
    static constexpr uint32_t MAX_PLANES=51*24+24;
    static constexpr uint32_t ARENA_BYTES=MAX_PLANES*16*4;
    __aicore__ inline TreeFloatRational16(uint32_t address,uint32_t limbs,uint32_t baseExp)
      : f_(TPosition::VECCALC,address,MAX_PLANES*16),l_(limbs),baseExp_(baseExp) {}
    __aicore__ inline LocalTensor<float> Plane(uint32_t p) {return F(p);}
    __aicore__ inline void Init() {
        Duplicate(S(12),0.0F,V);Duplicate(S(13),1.0F,V);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Zero(uint32_t dst,uint32_t n) {
        for(uint32_t k=0;k<n;++k)Duplicate(F(dst+k),0.0F,V);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CopyNumber(uint32_t dst,uint32_t src,uint32_t n) {
        for(uint32_t k=0;k<n;++k)Copy(F(dst+k),F(src+k),V,1,CopyRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void NegateIf(uint32_t dst,uint32_t n,uint32_t mask) {
        Duplicate(S(1),1.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<n;++k) {
            Muls(S(0),F(dst+k),-1.0F,V);PipeBarrier<PIPE_V>();
            Adds(S(0),S(0),4095.0F,V);PipeBarrier<PIPE_V>();
            Add(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();
            Split(S(5));
            Select(F(dst+k),M(mask),S(5),F(dst+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
        }
    }
    // Decode IEEE bits. No float multiply by 2^149, denormal arithmetic, or
    // approximate exponential is involved in the exact integer encoding.
    __aicore__ inline void Encode(uint32_t dst,LocalTensor<float> x) {
        auto bits=x.ReinterpretCast<uint32_t>();
        ShiftRight(S(2).ReinterpretCast<uint32_t>(),bits,uint32_t(31),V);PipeBarrier<PIPE_V>();
        Cast(S(6),S(2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        CmpScalar(M(10),S(6),0.0F,CMPMODE::GT);
        ShiftRight(S(2).ReinterpretCast<uint32_t>(),bits,uint32_t(23),V);
        Duplicate(S(3).ReinterpretCast<uint32_t>(),uint32_t(255),V);PipeBarrier<PIPE_V>();
        And(S(2).ReinterpretCast<uint16_t>(),S(2).ReinterpretCast<uint16_t>(),S(3).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
        Cast(S(6),S(2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);
        Duplicate(S(3).ReinterpretCast<uint32_t>(),uint32_t(0x7fffff),V);PipeBarrier<PIPE_V>();
        And(S(2).ReinterpretCast<uint16_t>(),bits.ReinterpretCast<uint16_t>(),S(3).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
        Cast(S(7),S(2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);
        CmpScalar(M(0),S(6),0.0F,CMPMODE::GT);PipeBarrier<PIPE_V>();
        Adds(S(8),S(7),8388608.0F,V);PipeBarrier<PIPE_V>();
        Select(S(7),M(0),S(8),S(7),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);
        Select(S(6),M(0),S(6),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,V);PipeBarrier<PIPE_V>();
        Adds(S(6),S(6),-float(int32_t(baseExp_)),V);PipeBarrier<PIPE_V>();
        // Zero may carry a negative shift when every nonzero input is normal.
        // Its mantissa is zero, so clamping that unused exponent is exact.
        Maxs(S(6),S(6),0.0F,V);PipeBarrier<PIPE_V>();
        Muls(S(8),S(6),1.0F/12.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(2).ReinterpretCast<int32_t>(),S(8),RoundMode::CAST_FLOOR,V);PipeBarrier<PIPE_V>();
        Cast(S(9),S(2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Muls(S(8),S(9),12.0F,V);PipeBarrier<PIPE_V>();
        Sub(S(8),S(6),S(8),V);PipeBarrier<PIPE_V>();
        Adds(S(8),S(8),127.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(2).ReinterpretCast<int32_t>(),S(8),RoundMode::CAST_RINT,V);PipeBarrier<PIPE_V>();
        Muls(S(2).ReinterpretCast<int32_t>(),S(2).ReinterpretCast<int32_t>(),int32_t(8388608),V);PipeBarrier<PIPE_V>();
        // Split the 24-bit mantissa before multiplying by 2^remainder.
        Muls(S(8),S(7),1.0F/4096.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(3).ReinterpretCast<int32_t>(),S(8),RoundMode::CAST_FLOOR,V);PipeBarrier<PIPE_V>();
        Cast(S(11),S(3).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Muls(S(8),S(11),4096.0F,V);PipeBarrier<PIPE_V>();
        Sub(S(7),S(7),S(8),V);PipeBarrier<PIPE_V>();
        Mul(S(0),S(7),S(2),V);PipeBarrier<PIPE_V>();Split(S(6));
        Mul(S(0),S(11),S(2),V);PipeBarrier<PIPE_V>();
        Add(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();Split(S(7));
        Copy(S(8),S(1),V,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<l_;++k) {
            CmpScalar(M(0),S(9),float(int32_t(k)),CMPMODE::EQ);
            CmpScalar(M(1),S(9),float(int32_t(k)-1),CMPMODE::EQ);
            CmpScalar(M(2),S(9),float(int32_t(k)-2),CMPMODE::EQ);PipeBarrier<PIPE_V>();
            Select(F(dst+k),M(0),S(6),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,V);PipeBarrier<PIPE_V>();
            Select(F(dst+k),M(1),S(7),F(dst+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
            Select(F(dst+k),M(2),S(8),F(dst+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
        }
        NegateIf(dst,l_,10);
    }
    __aicore__ inline void AddNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(S(1),0.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<n;++k) {
            Add(S(0),F(a+k),F(b+k),V);PipeBarrier<PIPE_V>();
            Add(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();Split(F(dst+k));
        }
    }
    __aicore__ inline void SubNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(S(1),0.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<n;++k) {
            Sub(S(0),F(a+k),F(b+k),V);PipeBarrier<PIPE_V>();
            Sub(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();
            CmpScalar(M(0),S(0),0.0F,CMPMODE::LT);PipeBarrier<PIPE_V>();
            Select(S(1),M(0),S(13),S(12),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
            Muls(S(4),S(1),4096.0F,V);PipeBarrier<PIPE_V>();
            Add(F(dst+k),S(0),S(4),V);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Multiply(uint32_t dst,uint32_t a,uint32_t na,uint32_t b,uint32_t nb,uint32_t n) {
        Zero(dst,n);
        for(uint32_t i=0;i<na;++i) {
            Duplicate(S(1),0.0F,V);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<nb && i+j<n;++j) {
                Mul(S(0),F(a+i),F(b+j),V);PipeBarrier<PIPE_V>();
                Add(S(0),S(0),F(dst+i+j),V);PipeBarrier<PIPE_V>();
                Add(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();Split(F(dst+i+j));
            }
            if(i+nb<n){Copy(F(dst+i+nb),S(1),V,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
        }
    }
    __aicore__ inline void CompareNumber(uint32_t a,uint32_t b,uint32_t n,uint32_t less,uint32_t equal) {
        Duplicate(M(less).ReinterpretCast<uint16_t>(),uint16_t(0),1);
        Duplicate(M(equal).ReinterpretCast<uint16_t>(),uint16_t(65535),1);PipeBarrier<PIPE_V>();
        for(int32_t k=int32_t(n)-1;k>=0;--k) {
            Cmp(M(0),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::LT);
            Cmp(M(1),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::EQ);PipeBarrier<PIPE_V>();
            MaskAnd(0,0,equal);MaskOr(less,less,0);MaskAnd(equal,equal,1);
        }
    }
    __aicore__ inline void SelectNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n,uint32_t mask) {
        for(uint32_t k=0;k<n;++k)Select(F(dst+k),M(mask),F(a+k),F(b+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void MakeScores(LocalTensor<float> point,LocalTensor<float> segments,LocalTensor<uint32_t> offsets,uint32_t segmentCount,uint32_t baseSegment=0) {
        Zero(25*l_,8*l_);
        for(uint32_t axis=0;axis<3;++axis) {
            Gather(S(14),segments,offsets,axis*4,V);
            Gather(S(15),segments,offsets,(axis+3)*4,V);PipeBarrier<PIPE_V>();
            Encode(33*l_,point[axis*V]);Encode(34*l_,S(14));Encode(35*l_,S(15));
            SubNumber(36*l_,33*l_,34*l_,l_);SubNumber(37*l_,35*l_,34*l_,l_);SubNumber(38*l_,33*l_,35*l_,l_);
            CmpScalar(M(8),F(37*l_-1),2048.0F,CMPMODE::GE);
            CmpScalar(M(9),F(38*l_-1),2048.0F,CMPMODE::GE);PipeBarrier<PIPE_V>();
            MaskAnd(0,8,9);MaskOr(11,8,9);
            Not(M(0).ReinterpretCast<uint16_t>(),M(0).ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();MaskAnd(11,11,0);
            NegateIf(36*l_,l_,8);NegateIf(37*l_,l_,9);
            CmpScalar(M(8),F(39*l_-1),2048.0F,CMPMODE::GE);PipeBarrier<PIPE_V>();NegateIf(38*l_,l_,8);
            Multiply(39*l_,36*l_,l_,36*l_,l_,2*l_);AddNumber(27*l_,27*l_,39*l_,2*l_);
            Multiply(39*l_,37*l_,l_,37*l_,l_,2*l_);AddNumber(29*l_,29*l_,39*l_,2*l_);
            Multiply(39*l_,38*l_,l_,38*l_,l_,2*l_);AddNumber(31*l_,31*l_,39*l_,2*l_);
            Multiply(39*l_,36*l_,l_,37*l_,l_,2*l_);NegateIf(39*l_,2*l_,11);AddNumber(25*l_,25*l_,39*l_,2*l_);
        }
        Multiply(43*l_,27*l_,2*l_,29*l_,2*l_,4*l_);
        Multiply(47*l_,25*l_,2*l_,25*l_,2*l_,4*l_);
        SubNumber(0,43*l_,47*l_,4*l_);CopyNumber(4*l_,29*l_,2*l_);
        CompareNumber(25*l_,29*l_,2*l_,6,7);
        Not(M(12).ReinterpretCast<uint16_t>(),M(6).ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
        CmpScalar(M(13),F(27*l_-1),2048.0F,CMPMODE::GE);PipeBarrier<PIPE_V>();
        Zero(39*l_,4*l_);CompareNumber(25*l_,39*l_,2*l_,6,7);MaskOr(13,13,7);
        CompareNumber(29*l_,39*l_,2*l_,6,7);MaskOr(13,13,7);
        CopyNumber(39*l_,31*l_,2*l_);SelectNumber(0,39*l_,0,4*l_,12);
        Zero(39*l_,4*l_);CopyNumber(39*l_,27*l_,2*l_);SelectNumber(0,39*l_,0,4*l_,13);
        MaskOr(12,12,13);Zero(39*l_,2*l_);Duplicate(F(39*l_),1.0F,V);PipeBarrier<PIPE_V>();
        SelectNumber(4*l_,39*l_,4*l_,2*l_,12);
        Cast(F(24*l_+3),offsets.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Muls(F(24*l_+3),F(24*l_+3),1.0F/24.0F,V);PipeBarrier<PIPE_V>();
        Cast(F(24*l_+2).ReinterpretCast<int32_t>(),F(24*l_+3),RoundMode::CAST_RINT,V);PipeBarrier<PIPE_V>();
        Cast(F(24*l_+3),F(24*l_+2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Adds(F(24*l_+3),F(24*l_+3),float(int32_t(baseSegment)),V);PipeBarrier<PIPE_V>();
        CmpScalar(M(12),F(24*l_+3),float(int32_t(segmentCount)),CMPMODE::GE);PipeBarrier<PIPE_V>();
        Zero(39*l_,4*l_);Duplicate(F(39*l_),1.0F,V);PipeBarrier<PIPE_V>();SelectNumber(0,39*l_,0,4*l_,12);
        Zero(39*l_,2*l_);SelectNumber(4*l_,39*l_,4*l_,2*l_,12);CopyNumber(24*l_,24*l_+3,1);
    }
    __aicore__ inline void ChooseGathered() {
        Multiply(12*l_,6*l_,4*l_,4*l_,2*l_,6*l_);
        Multiply(18*l_,0,4*l_,10*l_,2*l_,6*l_);
        CompareNumber(12*l_,18*l_,6*l_,6,7);
        Cmp(M(0),F(24*l_+1),F(24*l_),CMPMODE::LT);PipeBarrier<PIPE_V>();
        MaskAnd(0,0,7);MaskOr(6,6,0);
        SelectNumber(0,6*l_,0,6*l_,6);SelectNumber(24*l_,24*l_+1,24*l_,1,6);
    }
    __aicore__ inline void Tournament(LocalTensor<uint32_t> laneOffsets) {
        // laneOffsets initially contains 0..15, independent of global segment ID.
        for(uint32_t stage=0;stage<4;++stage) {
            Duplicate(S(10).ReinterpretCast<uint32_t>(),uint32_t(1U<<stage),V);PipeBarrier<PIPE_V>();
            And(S(9).ReinterpretCast<uint16_t>(),laneOffsets.ReinterpretCast<uint16_t>(),S(10).ReinterpretCast<uint16_t>(),2*V);
            Or(F(24*l_+2).ReinterpretCast<uint16_t>(),laneOffsets.ReinterpretCast<uint16_t>(),S(10).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
            Not(S(9).ReinterpretCast<uint16_t>(),S(9).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
            And(F(24*l_+2).ReinterpretCast<uint16_t>(),F(24*l_+2).ReinterpretCast<uint16_t>(),S(9).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
            Muls(F(24*l_+2).ReinterpretCast<int32_t>(),F(24*l_+2).ReinterpretCast<int32_t>(),int32_t(4),V);PipeBarrier<PIPE_V>();
            for(uint32_t k=0;k<6*l_;++k)Gather(F(6*l_+k),F(k),F(24*l_+2).ReinterpretCast<uint32_t>(),0U,V);
            Gather(F(24*l_+1),F(24*l_),F(24*l_+2).ReinterpretCast<uint32_t>(),0U,V);PipeBarrier<PIPE_V>();ChooseGathered();
        }
    }
    __aicore__ inline void Merge(LocalTensor<float> best,bool first) {
        if(!first) {
            for(uint32_t k=0;k<6*l_;++k)Copy(F(6*l_+k),best[k*V],V,1,CopyRepeatParams(1,1,8,8));
            Copy(F(24*l_+1),best[6*l_*V],V,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();ChooseGathered();
        }
        for(uint32_t k=0;k<6*l_;++k)Copy(best[k*V],F(k),V,1,CopyRepeatParams(1,1,8,8));
        Copy(best[6*l_*V],F(24*l_),V,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Normalized(uint32_t src,uint32_t digits,LocalTensor<float> value,LocalTensor<float> exponent,LocalTensor<uint32_t> lanes) {
        Duplicate(exponent,0.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<digits;++k) {
            CmpScalar(M(0),F(src+k),0.0F,CMPMODE::GT);
            Duplicate(S(0),float(int32_t(k)),V);PipeBarrier<PIPE_V>();
            Select(exponent,M(0),S(0),exponent,SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
        }
        Duplicate(value,0.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<3;++k) {
            Adds(S(0),exponent,-float(int32_t(k)),V);PipeBarrier<PIPE_V>();
            CmpScalar(M(0),S(0),0.0F,CMPMODE::GE);
            Maxs(S(0),S(0),0.0F,V);PipeBarrier<PIPE_V>();
            Cast(S(1).ReinterpretCast<int32_t>(),S(0),RoundMode::CAST_RINT,V);PipeBarrier<PIPE_V>();
            Muls(S(1).ReinterpretCast<int32_t>(),S(1).ReinterpretCast<int32_t>(),int32_t(V),V);PipeBarrier<PIPE_V>();
            Add(S(1).ReinterpretCast<int32_t>(),S(1).ReinterpretCast<int32_t>(),lanes.ReinterpretCast<int32_t>(),V);PipeBarrier<PIPE_V>();
            Muls(S(1).ReinterpretCast<int32_t>(),S(1).ReinterpretCast<int32_t>(),int32_t(4),V);PipeBarrier<PIPE_V>();
            Gather(S(2),F(src),S(1).ReinterpretCast<uint32_t>(),0U,V);PipeBarrier<PIPE_V>();
            Select(S(2),M(0),S(2),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,V);PipeBarrier<PIPE_V>();
            Muls(S(2),S(2),k==0?1.0F:k==1?0.000244140625F:5.960464477539063e-8F,V);PipeBarrier<PIPE_V>();
            Add(value,value,S(2),V);PipeBarrier<PIPE_V>();
        }
    }
    // Recover distance from the winning exact rational using its leading
    // three radix digits. Indices remain exact; FP32 distance uses the normal
    // operator tolerance. Split powers avoid avoidable intermediate overflow.
    __aicore__ inline void WriteDistance(LocalTensor<float> dst,LocalTensor<uint32_t> lanes) {
        Normalized(0,4*l_,S(6),S(7),lanes);
        Normalized(4*l_,2*l_,S(8),S(9),lanes);
        Div(S(0),S(6),S(8),V);PipeBarrier<PIPE_V>();Sqrt(S(0),S(0),V);PipeBarrier<PIPE_V>();
        Sub(S(1),S(7),S(9),V);PipeBarrier<PIPE_V>();
        Muls(S(1),S(1),6.0F,V);PipeBarrier<PIPE_V>();
        Adds(S(1),S(1),float(int32_t(baseExp_)-150),V);PipeBarrier<PIPE_V>();
        Maxs(S(2),S(1),-126.0F,V);PipeBarrier<PIPE_V>();Mins(S(2),S(2),127.0F,V);PipeBarrier<PIPE_V>();
        Sub(S(3),S(1),S(2),V);PipeBarrier<PIPE_V>();
        Maxs(S(3),S(3),-126.0F,V);PipeBarrier<PIPE_V>();Mins(S(3),S(3),127.0F,V);PipeBarrier<PIPE_V>();
        Adds(S(2),S(2),127.0F,V);Adds(S(3),S(3),127.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(2).ReinterpretCast<int32_t>(),S(2),RoundMode::CAST_RINT,V);
        Cast(S(3).ReinterpretCast<int32_t>(),S(3),RoundMode::CAST_RINT,V);PipeBarrier<PIPE_V>();
        Muls(S(2).ReinterpretCast<int32_t>(),S(2).ReinterpretCast<int32_t>(),int32_t(8388608),V);
        Muls(S(3).ReinterpretCast<int32_t>(),S(3).ReinterpretCast<int32_t>(),int32_t(8388608),V);PipeBarrier<PIPE_V>();
        Mul(S(0),S(0),S(2),V);PipeBarrier<PIPE_V>();Mul(dst,S(0),S(3),V);PipeBarrier<PIPE_V>();
    }
};


#endif
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
// Called after refinement and index remapping, when all six work planes are dead.
// Columns, compactP, prefetched rawP, and output rows remain outside this arena.
__aicore__ __attribute__((noinline)) inline void RefineWideFloatRowsBody(
    GM_ADDR segmentAddress,uint32_t segments,uint32_t workAddress,
    uint32_t pointAddress,uint32_t indexAddress,uint32_t distanceAddress,
    uint32_t liveRows,uint64_t wideRows) {
    LocalTensor<float> work(TPosition::VECCALC,workAddress,25344);
    LocalTensor<float> points(TPosition::VECCALC,pointAddress,192);
    LocalTensor<int32_t> indices(TPosition::VECCALC,indexAddress,64);
    LocalTensor<float> distances(TPosition::VECCALC,distanceAddress,64);
    GlobalTensor<float> sg;sg.SetGlobalBuffer((__gm__ float*)segmentAddress);
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

    // 79,872 B arithmetic + 9,280 B best score + a 384 B streamed packet.
    // Range detection temporarily stages all coefficients before arithmetic starts.
    const uint32_t padded=(6*segments+63)/64*64;
    auto sf=work[12000],point=work[22528],packet=work[22656],temp=work[22800];
    auto lane=work[22912].ReinterpretCast<int32_t>();
    auto offsets=work[22928].ReinterpretCast<int32_t>();
    for(uint32_t row=0;row<liveRows;++row) {
        if((wideRows&(uint64_t(1)<<row))==0)continue;
        PipeBarrier<PIPE_ALL>();
        Duplicate(sf,0.0F,padded+64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        DataCopyPad(sf,sg,DataCopyExtParams{1,segments*24,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        for(uint32_t axis=0;axis<3;++axis) {
            Brcb(temp,points[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Copy(point[axis*16],temp[(row%8)*8],16,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
        }
        Copy(sf[padded],point,48,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        const uint32_t coords=padded+64;
        auto exponent=work,scratch=work[6208],record=work[8192],reduced=work[8320];
        auto mask=work[7000].ReinterpretCast<uint8_t>();
        ShiftRight(exponent.ReinterpretCast<uint32_t>(),sf.ReinterpretCast<uint32_t>(),uint32_t(23),coords);
        Duplicate(scratch.ReinterpretCast<uint32_t>(),uint32_t(255),64);PipeBarrier<PIPE_V>();
        // Broadcast bit mask to all coordinates without scalar payload reads.
        for(uint32_t k=0;k<coords;k+=64)And(exponent[k].ReinterpretCast<uint16_t>(),exponent[k].ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),128);
        PipeBarrier<PIPE_V>();Cast(exponent,exponent.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,coords);PipeBarrier<PIPE_V>();
        Maxs(exponent,exponent,1.0F,coords);PipeBarrier<PIPE_V>();
        Duplicate(record,0.0F,128);PipeBarrier<PIPE_V>();
        WholeReduceMax(record,exponent,64,coords/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(reduced,record,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(reduced,reduced,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Cast(reduced[8].ReinterpretCast<int32_t>(),reduced,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);const uint32_t maxExp=uint32_t(reduced[8].ReinterpretCast<int32_t>().GetValue(0));
        // Magnitude bits distinguish subnormals from signed zeros without FTZ.
        Duplicate(scratch.ReinterpretCast<uint32_t>(),uint32_t(0x7fffffff),64);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<coords;k+=64) {
            And(work[6272].ReinterpretCast<uint16_t>(),sf[k].ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            CompareScalar(mask,work[6272].ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            Not(mask.ReinterpretCast<uint16_t>(),mask.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Select(exponent[k],mask,exponent[k],255.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        }
        Duplicate(record,255.0F,128);PipeBarrier<PIPE_V>();
        WholeReduceMin(record,exponent,64,coords/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(reduced,record,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(reduced,reduced,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Cast(reduced[8].ReinterpretCast<int32_t>(),reduced,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);uint32_t minExp=uint32_t(reduced[8].ReinterpretCast<int32_t>().GetValue(0));
        if(minExp>maxExp)minExp=maxExp;
        const uint32_t required=(maxExp-minExp+28+11)/12,limbs=required<4?4:required;

        TreeFloatRational16 exact(workAddress,limbs,minExp);exact.Init();
        Duplicate(lane,int32_t(0),16);PipeBarrier<PIPE_V>();
        const uint64_t laneMasks[4]={0xaaaa,0xcccc,0xf0f0,0xff00};
        for(uint32_t k=0;k<4;++k) {
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),laneMasks[k]);
            Adds<int32_t,false>(lane,lane,int32_t(1U<<k),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
        Muls(offsets,lane,int32_t(24),16);PipeBarrier<PIPE_V>();
        for(uint32_t start=0;start<segments;start+=16) {
            const uint32_t count=segments-start<16?segments-start:16;
            Duplicate(packet,0.0F,96);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            DataCopyPad(packet,sg[start*6],DataCopyExtParams{1,count*24,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            exact.MakeScores(point,packet,offsets.ReinterpretCast<uint32_t>(),segments,start);
            exact.Tournament(lane.ReinterpretCast<uint32_t>());exact.Merge(work[20000],start==0);
        }
        exact.WriteDistance(temp,lane.ReinterpretCast<uint32_t>());
        Cast(temp[16].ReinterpretCast<int32_t>(),exact.Plane(24*limbs),RoundMode::CAST_RINT,16);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(0),uint64_t(1)<<(row%8));
        Copy<float,false>(distances[(row/8)*8],temp,uint64_t(64),1,CopyRepeatParams(1,0,8,0));
        Copy<int32_t,false>(indices[(row/8)*8],temp[16].ReinterpretCast<int32_t>(),uint64_t(64),1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    }
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}

__aicore__ __attribute__((always_inline)) inline void RefineWideFloatRows(
    GM_ADDR segmentAddress,uint32_t segments,uint32_t workAddress,
    uint32_t pointAddress,uint32_t indexAddress,uint32_t distanceAddress,
    uint32_t liveRows,bool coefficientSafe) {
    LocalTensor<float> work(TPosition::VECCALC,workAddress,25344);
    LocalTensor<float> points(TPosition::VECCALC,pointAddress,192);
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    const uint64_t active=(~uint64_t(0))>>(64-liveRows);
    uint64_t wideRows=coefficientSafe?uint64_t(0):active;
    if(coefficientSafe) {
        // Three contiguous coordinate planes share the same vector predicate pass.
        // Integer zero equality retains nonzero subnormals; one V_S joins all axes.
        auto masks=work[384].ReinterpretCast<uint8_t>();
        Duplicate(work[192].ReinterpretCast<uint32_t>(),uint32_t(0x7fffffff),192);PipeBarrier<PIPE_V>();
        And(work.ReinterpretCast<uint16_t>(),points.ReinterpretCast<uint16_t>(),work[192].ReinterpretCast<uint16_t>(),384);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,true>(masks,work,1099511627776.0F,CMPMODE::GT,uint64_t(64),uint8_t(3),UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,true>(masks[32],work,9.5367431640625e-7F,CMPMODE::LT,uint64_t(64),uint8_t(3),UnaryRepeatParams(1,1,8,8));
        CompareScalar(masks[64],work.ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,192);PipeBarrier<PIPE_V>();
        Not(masks[64].ReinterpretCast<uint16_t>(),masks[64].ReinterpretCast<uint16_t>(),12);PipeBarrier<PIPE_V>();
        And(masks[32].ReinterpretCast<uint16_t>(),masks[32].ReinterpretCast<uint16_t>(),masks[64].ReinterpretCast<uint16_t>(),12);PipeBarrier<PIPE_V>();
        Or(masks.ReinterpretCast<uint16_t>(),masks.ReinterpretCast<uint16_t>(),masks[32].ReinterpretCast<uint16_t>(),12);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto packed=masks.ReinterpretCast<uint64_t>();
        wideRows=(packed.GetValue(0)|packed.GetValue(1)|packed.GetValue(2))&active;
    }

    if(wideRows)RefineWideFloatRowsBody(segmentAddress,segments,workAddress,
        pointAddress,indexAddress,distanceAddress,liveRows,wideRows);
}


#endif // TREE_STREAM_EXACT

#if defined(DTYPE_TREE_POINTS)
template<uint32_t DIGITS=4,uint32_t SCALE_EXP=24>
class TreePacketRational64 {
    static_assert((DIGITS==4 && SCALE_EXP==24) || (DIGITS==2 && SCALE_EXP==14),"Certified rational formats");
    LocalTensor<float> f_;
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> F(uint32_t p) { return f_[p*64]; }
    __aicore__ __attribute__((always_inline)) inline LocalTensor<uint8_t> M(uint32_t p) { return F(220).template ReinterpretCast<uint8_t>()[p*32]; }
    __aicore__ __attribute__((always_inline)) inline void Split(LocalTensor<float> dst) {
        Muls(F(208),F(204),0.000244140625F,64);PipeBarrier<PIPE_V>();
        Cast(F(207).template ReinterpretCast<int32_t>(),F(208),RoundMode::CAST_FLOOR,64);PipeBarrier<PIPE_V>();
        Cast(F(205),F(207).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(F(208),F(205),4096.0F,64);PipeBarrier<PIPE_V>();
        Sub(dst,F(204),F(208),64);PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ __attribute__((always_inline)) inline TreePacketRational64(uint32_t address)
      : f_(TPosition::VECCALC,address,224*64) {}
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> Plane(uint32_t p) {return F(p);}

    __aicore__ __attribute__((always_inline)) inline void Init() {
        Duplicate(F(216),0.0F,64);Duplicate(F(217),1.0F,64);PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void Zero(uint32_t dst,uint32_t n) {
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k)Duplicate(F(dst+k),0.0F,64);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void CopyNumber(uint32_t dst,uint32_t src,uint32_t n) {
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k)Copy(F(dst+k),F(src+k),64,1,CopyRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void NegateIf(uint32_t dst,uint32_t n,uint32_t mask) {
        Duplicate(F(205),1.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Muls(F(204),F(dst+k),-1.0F,64);PipeBarrier<PIPE_V>();
            Adds(F(204),F(204),4095.0F,64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(209));
            Select(F(dst+k),M(mask),F(209),F(dst+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ __attribute__((always_inline)) inline void Encode(uint32_t dst,LocalTensor<float> x) {
        CompareScalar(M(10),x,0.0F,CMPMODE::LT,64);
        Abs(F(210),x,64);PipeBarrier<PIPE_V>();
        Muls(F(210),F(210),SCALE_EXP==14?16384.0F:16777216.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(int32_t k=int32_t(DIGITS)-1;k>=0;--k) {
            const float inverse=k==3?1.4551915228366852e-11F:k==2?5.960464477539063e-8F:k==1?0.000244140625F:1.0F;
            const float power=k==3?68719476736.0F:k==2?16777216.0F:k==1?4096.0F:1.0F;
            Muls(F(211),F(210),inverse,64);PipeBarrier<PIPE_V>();
            Cast(F(207).template ReinterpretCast<int32_t>(),F(211),RoundMode::CAST_FLOOR,64);PipeBarrier<PIPE_V>();
            Cast(F(dst+uint32_t(k)),F(207).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Muls(F(211),F(dst+uint32_t(k)),power,64);PipeBarrier<PIPE_V>();
            Sub(F(210),F(210),F(211),64);PipeBarrier<PIPE_V>();
        }
        NegateIf(dst,DIGITS,10);
    }
    __aicore__ __attribute__((always_inline)) inline void AddNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(F(205),0.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Add(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(dst+k));
        }
    }
    __aicore__ __attribute__((always_inline)) inline void SubNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(F(205),0.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Sub(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Sub(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            CompareScalar(M(0),F(204),0.0F,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            Select(F(205),M(0),F(217),F(216),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
            Muls(F(208),F(205),4096.0F,64);PipeBarrier<PIPE_V>();
            Add(F(dst+k),F(204),F(208),64);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ __attribute__((always_inline)) inline void Multiply(uint32_t dst,uint32_t a,uint32_t na,uint32_t b,uint32_t nb,uint32_t n) {
        Zero(dst,n);
        #pragma unroll 1
        for(uint32_t i=0;i<na;++i) {
            Duplicate(F(205),0.0F,64);PipeBarrier<PIPE_V>();
            #pragma unroll 1
            for(uint32_t j=0;j<nb && i+j<n;++j) {
                if constexpr(DIGITS==2) {
                    Add(F(204),F(dst+i+j),F(205),64);PipeBarrier<PIPE_V>();
                    MulAddDst(F(204),F(a+i),F(b+j),64);PipeBarrier<PIPE_V>();
                } else {
                    Mul(F(204),F(a+i),F(b+j),64);PipeBarrier<PIPE_V>();
                    Add(F(204),F(204),F(dst+i+j),64);PipeBarrier<PIPE_V>();
                    Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
                }
                Split(F(dst+i+j));
            }
            if(i+nb<n) {Copy(F(dst+i+nb),F(205),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
        }
    }
    // Return masks a<b and a==b; callers retain any signed predicates elsewhere.
    __aicore__ __attribute__((always_inline)) inline void CompareNumber(uint32_t a,uint32_t b,uint32_t n,uint32_t less,uint32_t equal) {
        Duplicate(M(less).template ReinterpretCast<uint16_t>(),uint16_t(0),4);
        Duplicate(M(equal).template ReinterpretCast<uint16_t>(),uint16_t(65535),4);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(int32_t k=int32_t(n)-1;k>=0;--k) {
            Compare(M(0),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::LT,64);
            Compare(M(1),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),M(equal).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(less).template ReinterpretCast<uint16_t>(),M(less).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);
            And(M(equal).template ReinterpretCast<uint16_t>(),M(equal).template ReinterpretCast<uint16_t>(),M(1).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ __attribute__((always_inline)) inline void SelectNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n,uint32_t mask) {
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k)Select(F(dst+k),M(mask),F(a+k),F(b+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        PipeBarrier<PIPE_V>();
    }
    template<bool PACKET=false>
    __aicore__ __attribute__((always_inline)) inline void MakeScores(uint32_t row,uint32_t pointAddress,uint32_t pointPitch,uint32_t sfAddress,uint32_t soAddress,uint32_t segments,
        uint32_t indexAddress=0,uint64_t validLanes=0) {
        const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
        const LocalTensor<float> sf(TPosition::VECCALC,sfAddress,PACKET?768:384);
        const LocalTensor<uint32_t> so(TPosition::VECCALC,soAddress,64);
        if constexpr(DIGITS!=4)Zero(0,24); // Keep unused numerator/denominator digits zero.
        Zero(100,32); // dot, |P-A|^2, |C-A|^2, |P-C|^2
        #pragma unroll 1
        for(uint32_t axis=0;axis<3;++axis) {
            if constexpr(PACKET) {
                Copy(F(218),sf[axis*128],64,1,CopyRepeatParams(1,1,8,8));
                Copy(F(219),sf[(axis+3)*128],64,1,CopyRepeatParams(1,1,8,8));
            } else {
                Gather(F(218),sf,so,axis*4,64);Gather(F(219),sf,so,(axis+3)*4,64);
            }
            PipeBarrier<PIPE_V>();
            Encode(132,points[axis*pointPitch+row*64]);Encode(136,F(218));Encode(140,F(219));
            SubNumber(144,132,136,DIGITS);SubNumber(148,140,136,DIGITS);SubNumber(152,132,140,DIGITS);
            CompareScalar(M(8),F(144+DIGITS-1),2048.0F,CMPMODE::GE,64);
            CompareScalar(M(9),F(148+DIGITS-1),2048.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(8).template ReinterpretCast<uint16_t>(),M(9).template ReinterpretCast<uint16_t>(),4);
            Or(M(11).template ReinterpretCast<uint16_t>(),M(8).template ReinterpretCast<uint16_t>(),M(9).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Not(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(M(11).template ReinterpretCast<uint16_t>(),M(11).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            NegateIf(144,DIGITS,8);NegateIf(148,DIGITS,9);
            CompareScalar(M(8),F(152+DIGITS-1),2048.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();NegateIf(152,DIGITS,8);
            Multiply(156,144,DIGITS,144,DIGITS,2*DIGITS);AddNumber(108,108,156,2*DIGITS);
            Multiply(156,148,DIGITS,148,DIGITS,2*DIGITS);AddNumber(116,116,156,2*DIGITS);
            Multiply(156,152,DIGITS,152,DIGITS,2*DIGITS);AddNumber(124,124,156,2*DIGITS);
            Multiply(156,144,DIGITS,148,DIGITS,2*DIGITS);NegateIf(156,2*DIGITS,11);AddNumber(100,100,156,2*DIGITS);
        }
        Multiply(172,108,2*DIGITS,116,2*DIGITS,4*DIGITS);Multiply(188,100,2*DIGITS,100,2*DIGITS,4*DIGITS);
        SubNumber(0,172,188,4*DIGITS);CopyNumber(16,116,2*DIGITS);
        CompareNumber(100,116,2*DIGITS,6,7);
        Not(M(12).template ReinterpretCast<uint16_t>(),M(6).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>(); // dot>=vv
        CompareScalar(M(13),F(100+2*DIGITS-1),2048.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>(); // dot<0
        Zero(156,16);CompareNumber(100,156,2*DIGITS,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CompareNumber(116,156,2*DIGITS,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CopyNumber(156,124,2*DIGITS);SelectNumber(0,156,0,16,12);
        Zero(156,16);CopyNumber(156,108,2*DIGITS);SelectNumber(0,156,0,16,13);
        Or(M(12).template ReinterpretCast<uint16_t>(),M(12).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Zero(156,8);Duplicate(F(156),1.0F,64);PipeBarrier<PIPE_V>();SelectNumber(16,156,16,8,12);
        if constexpr(PACKET) {
            const LocalTensor<float> ids(TPosition::VECCALC,indexAddress,64);
            Copy(F(99),so.template ReinterpretCast<float>(),64,1,CopyRepeatParams(1,1,8,8));
            Copy(F(96),ids,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            M(12).template ReinterpretCast<uint64_t>().SetValue(0,~validLanes);
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        } else {
        Cast(F(99),so.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(F(99),F(99),1.0F/24.0F,64);PipeBarrier<PIPE_V>();
        // The fixed offset vector consists exactly of 0,24,...,1512.
        Cast(F(98).template ReinterpretCast<int32_t>(),F(99),RoundMode::CAST_RINT,64);PipeBarrier<PIPE_V>();
        Cast(F(99),F(98).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(M(12),F(99),float(int32_t(segments)),CMPMODE::GE,64);PipeBarrier<PIPE_V>();
        }
        Zero(156,16);Duplicate(F(156),1.0F,64);PipeBarrier<PIPE_V>();SelectNumber(0,156,0,16,12);
        Zero(156,8);SelectNumber(16,156,16,8,12);if constexpr(!PACKET)CopyNumber(96,99,1);
    }
    template<uint32_t STAGES=6>
    __aicore__ __attribute__((always_inline)) inline void Tournament() {
        #pragma unroll 1
        for(uint32_t stage=0;stage<STAGES;++stage) {
            Cast(F(98).template ReinterpretCast<int32_t>(),F(99),RoundMode::CAST_RINT,64);
            Duplicate(F(214).template ReinterpretCast<int32_t>(),int32_t(1U<<stage),64);PipeBarrier<PIPE_V>();
            And(F(213).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(214).template ReinterpretCast<uint16_t>(),128);
            Or(F(98).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(214).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Not(F(213).template ReinterpretCast<uint16_t>(),F(213).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            And(F(98).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(213).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(F(98).template ReinterpretCast<int32_t>(),F(98).template ReinterpretCast<int32_t>(),int32_t(4),64);PipeBarrier<PIPE_V>();
            #pragma unroll 1
            for(uint32_t k=0;k<24;++k)Gather(F(24+k),F(k),F(98).template ReinterpretCast<uint32_t>(),0U,64);
            Gather(F(97),F(96),F(98).template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            Multiply(48,24,4*DIGITS,16,2*DIGITS,6*DIGITS);Multiply(72,0,4*DIGITS,40,2*DIGITS,6*DIGITS);
            CompareNumber(48,72,6*DIGITS,6,7);
            Compare(M(0),F(97),F(96),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(6).template ReinterpretCast<uint16_t>(),M(6).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            SelectNumber(0,24,0,24,6);SelectNumber(96,97,96,1,6);
        }
    }
};

template<uint32_t DIGITS=2,bool SIGNED_COORDINATES=true>
class TreePacketIntegerRational64 {
    static_assert(DIGITS==2 && SIGNED_COORDINATES,"Signed base-32768 rational format");
    // With |coordinate| < 128 and scale 2^20, dot products fit 59 signed
    // bits, Gram numerators fit 120 bits, and cross products fit 180 bits.
    static constexpr uint32_t NUMERATOR_DIGITS=8U;
    static constexpr uint32_t CROSS_DIGITS=NUMERATOR_DIGITS+2*DIGITS;
    LocalTensor<int32_t> f_;
    __aicore__ __attribute__((always_inline)) inline LocalTensor<int32_t> F(uint32_t p) { return f_[p*64]; }
    __aicore__ __attribute__((always_inline)) inline LocalTensor<uint8_t> M(uint32_t p) { return F(220).template ReinterpretCast<uint8_t>()[p*32]; }
    // Each multiply-add is at most 2^30-1, safely within signed int32.
    __aicore__ __attribute__((always_inline)) inline void Split(LocalTensor<int32_t> dst) {
        ShiftRight(F(205),F(204),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(F(208),F(205),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(dst,F(204),F(208),64);PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ __attribute__((always_inline)) inline TreePacketIntegerRational64(uint32_t address)
      : f_(TPosition::VECCALC,address,224*64) {}
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> Plane(uint32_t p) {return F(p).template ReinterpretCast<float>();}

    __aicore__ __attribute__((always_inline)) inline void Init() {
        Duplicate(F(216),int32_t(0),64);Duplicate(F(217),int32_t(1),64);PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void Zero(uint32_t dst,uint32_t n) {
        if constexpr(DIGITS==2) {
            Duplicate(F(dst),int32_t(0),n*64);
        } else {
            #pragma unroll 1
            for(uint32_t k=0;k<n;++k)Duplicate(F(dst+k),int32_t(0),64);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void CopyNumber(uint32_t dst,uint32_t src,uint32_t n) {
        if constexpr(DIGITS==2) {
            Copy(F(dst),F(src),64,uint8_t(n),CopyRepeatParams(1,1,8,8));
        } else {
            #pragma unroll 1
            for(uint32_t k=0;k<n;++k)Copy(F(dst+k),F(src+k),64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
    }
    // Digit values are in [0,32767]. Bias only their representation so VSEL
    // sees normal float values, then undo the bias using exact int32 arithmetic.
    __aicore__ __attribute__((always_inline)) inline void SelectDigit(
        LocalTensor<int32_t> dst,LocalTensor<int32_t> a,LocalTensor<int32_t> b,uint32_t mask) {
        Adds(F(206),a,int32_t(1073741824),64);
        Adds(F(212),b,int32_t(1073741824),64);PipeBarrier<PIPE_V>();
        Select(F(215).template ReinterpretCast<float>(),M(mask),F(206).template ReinterpretCast<float>(),F(212).template ReinterpretCast<float>(),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        Adds(dst,F(215),int32_t(-1073741824),64);PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void NegateIf(uint32_t dst,uint32_t n,uint32_t mask) {
        Duplicate(F(205),int32_t(1),64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Muls(F(204),F(dst+k),int32_t(-1),64);PipeBarrier<PIPE_V>();
            Adds(F(204),F(204),int32_t(32767),64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(209));
            SelectDigit(F(dst+k),F(209),F(dst+k),mask);
        }
    }
    __aicore__ __attribute__((always_inline)) inline void Encode(uint32_t dst,LocalTensor<float> x) {
        if constexpr(SIGNED_COORDINATES) {
        // The certificate preserves every admitted coordinate. Bound speculative
        // rejected lanes as well, so all signed digit intermediates stay in range.
        Maxs(F(210).template ReinterpretCast<float>(),x,-128.0F,64);PipeBarrier<PIPE_V>();
        Mins(F(210).template ReinterpretCast<float>(),F(210).template ReinterpretCast<float>(),128.0F,64);PipeBarrier<PIPE_V>();
        Muls(F(210).template ReinterpretCast<float>(),F(210).template ReinterpretCast<float>(),1048576.0F,64);PipeBarrier<PIPE_V>();
        Cast(F(207),F(210).template ReinterpretCast<float>(),RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
        ShiftRight(F(dst+1),F(207),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(F(211),F(dst+1),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(F(dst),F(207),F(211),64);PipeBarrier<PIPE_V>();
    
        } else {
        // Convert the signed value once, then form its two base32768 digits.
        // All finite Half values times2^14 fit int32; admitted values are exact.
        Muls(F(210).template ReinterpretCast<float>(),x,16384.0F,64);PipeBarrier<PIPE_V>();
        Cast(F(207),F(210).template ReinterpretCast<float>(),RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
        ShiftRight(F(dst+1),F(207),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(F(211),F(dst+1),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(F(dst),F(207),F(211),64);PipeBarrier<PIPE_V>();
        ShiftRight(F(205),F(dst+1),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(F(211),F(205),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(F(dst+1),F(dst+1),F(211),64);PipeBarrier<PIPE_V>();
    
        }
    }
    __aicore__ __attribute__((always_inline)) inline void SubCoordinate(uint32_t dst,uint32_t a,uint32_t b) {
        Sub(F(204),F(a),F(b),64);PipeBarrier<PIPE_V>();
        Split(F(dst));
        Sub(F(dst+1),F(a+1),F(b+1),64);PipeBarrier<PIPE_V>();
        Add(F(dst+1),F(dst+1),F(205),64);PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void CoordinateProduct(uint32_t dst,uint32_t a,uint32_t b) {
        // Low digits are unsigned15-bit; differences have |high digit| <= 8192.
        Mul(F(204),F(a),F(b),64);PipeBarrier<PIPE_V>();
        Split(F(dst));
        Mul(F(210),F(a),F(b+1),64);
        Mul(F(211),F(a+1),F(b),64);PipeBarrier<PIPE_V>();
        Add(F(204),F(210),F(211),64);PipeBarrier<PIPE_V>();
        Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
        Split(F(dst+1));
        Mul(F(204),F(a+1),F(b+1),64);PipeBarrier<PIPE_V>();
        Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
        Split(F(dst+2));
        Copy(F(204),F(205),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        Split(F(dst+3));
    }
    __aicore__ __attribute__((always_inline)) inline void AddNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(F(205),int32_t(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Add(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(dst+k));
        }
    }
    __aicore__ __attribute__((always_inline)) inline void SubNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        if constexpr(SIGNED_COORDINATES) {
        Duplicate(F(205),int32_t(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Sub(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            // Signed carry is0 or-1. Split normalizes every digit exactly.
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(dst+k));
        }
    
        } else {
        Duplicate(F(205),int32_t(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Sub(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Sub(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Cast(F(206).template ReinterpretCast<float>(),F(204),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(M(0),F(206).template ReinterpretCast<float>(),0.0F,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            SelectDigit(F(205),F(217),F(216),0);
            Muls(F(208),F(205),int32_t(32768),64);PipeBarrier<PIPE_V>();
            Add(F(dst+k),F(204),F(208),64);PipeBarrier<PIPE_V>();
        }
    
        }
    }
    __aicore__ __attribute__((always_inline)) inline void Multiply(uint32_t dst,uint32_t a,uint32_t na,uint32_t b,uint32_t nb,uint32_t n) {
        Zero(dst,n);
        #pragma unroll 1
        for(uint32_t i=0;i<na;++i) {
            Duplicate(F(205),int32_t(0),64);PipeBarrier<PIPE_V>();
            #pragma unroll 1
            for(uint32_t j=0;j<nb && i+j<n;++j) {
                Mul(F(204),F(a+i),F(b+j),64);PipeBarrier<PIPE_V>();
                Add(F(204),F(204),F(dst+i+j),64);PipeBarrier<PIPE_V>();
                Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
                Split(F(dst+i+j));
            }
            if(i+nb<n) {Copy(F(dst+i+nb),F(205),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
        }
    }
    // Return masks a<b and a==b; callers retain any signed predicates elsewhere.
    __aicore__ __attribute__((always_inline)) inline void CompareNumber(uint32_t a,uint32_t b,uint32_t n,uint32_t less,uint32_t equal) {
        Duplicate(M(less).template ReinterpretCast<uint16_t>(),uint16_t(0),4);
        Duplicate(M(equal).template ReinterpretCast<uint16_t>(),uint16_t(65535),4);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(int32_t k=int32_t(n)-1;k>=0;--k) {
            // int32 Compare supports EQ only on this device. The biased
            // normal float encodings preserve the order of all15-bit digits.
            Adds(F(206),F(a+uint32_t(k)),int32_t(1073741824),64);
            Adds(F(212),F(b+uint32_t(k)),int32_t(1073741824),64);PipeBarrier<PIPE_V>();
            Compare(M(0),F(206).template ReinterpretCast<float>(),F(212).template ReinterpretCast<float>(),CMPMODE::LT,64);
            Compare(M(1),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),M(equal).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(less).template ReinterpretCast<uint16_t>(),M(less).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);
            And(M(equal).template ReinterpretCast<uint16_t>(),M(equal).template ReinterpretCast<uint16_t>(),M(1).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ __attribute__((always_inline)) inline void SelectNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n,uint32_t mask) {
        if(dst==96) {
            // Lane IDs already contain normal float indices or exact zero.
            Select(F(96).template ReinterpretCast<float>(),M(mask),F(a).template ReinterpretCast<float>(),F(b).template ReinterpretCast<float>(),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        } else {
            #pragma unroll 1
            for(uint32_t k=0;k<n;++k)SelectDigit(F(dst+k),F(a+k),F(b+k),mask);
        }
    }
    template<bool PACKET=false>
    __aicore__ __attribute__((always_inline)) inline void MakeScores(uint32_t row,uint32_t pointAddress,uint32_t pointPitch,uint32_t sfAddress,uint32_t soAddress,uint32_t segments,
        uint32_t indexAddress=0,uint64_t validLanes=0) {
        const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
        const LocalTensor<float> sf(TPosition::VECCALC,sfAddress,PACKET?768:384);
        const LocalTensor<uint32_t> so(TPosition::VECCALC,soAddress,64);
        if constexpr(DIGITS!=4)Zero(0,24); // Keep unused numerator/denominator digits zero.
        Zero(100,32); // dot, |P-A|^2, |C-A|^2, |P-C|^2
        #pragma unroll 1
        for(uint32_t axis=0;axis<3;++axis) {
            if constexpr(PACKET) {
                Copy(F(218),sf[axis*128].template ReinterpretCast<int32_t>(),64,1,CopyRepeatParams(1,1,8,8));
                Copy(F(219),sf[(axis+3)*128].template ReinterpretCast<int32_t>(),64,1,CopyRepeatParams(1,1,8,8));
            } else {
                Gather(F(218),sf.template ReinterpretCast<int32_t>(),so,axis*4,64);Gather(F(219),sf.template ReinterpretCast<int32_t>(),so,(axis+3)*4,64);
            }
            PipeBarrier<PIPE_V>();
            Encode(132,points[axis*pointPitch+row*64]);Encode(136,F(218).template ReinterpretCast<float>());Encode(140,F(219).template ReinterpretCast<float>());
            if constexpr(SIGNED_COORDINATES) {
            SubCoordinate(144,132,136);SubCoordinate(148,140,136);SubCoordinate(152,132,140);
            CoordinateProduct(156,144,144);AddNumber(108,108,156,2*DIGITS);
            CoordinateProduct(156,148,148);AddNumber(116,116,156,2*DIGITS);
            CoordinateProduct(156,152,152);AddNumber(124,124,156,2*DIGITS);
            CoordinateProduct(156,144,148);AddNumber(100,100,156,2*DIGITS);
            } else {
            SubNumber(144,132,136,DIGITS);SubNumber(148,140,136,DIGITS);SubNumber(152,132,140,DIGITS);
            Cast(F(206).template ReinterpretCast<float>(),F(144+DIGITS-1),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(M(8),F(206).template ReinterpretCast<float>(),16384.0F,CMPMODE::GE,64);
            Cast(F(206).template ReinterpretCast<float>(),F(148+DIGITS-1),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(M(9),F(206).template ReinterpretCast<float>(),16384.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(8).template ReinterpretCast<uint16_t>(),M(9).template ReinterpretCast<uint16_t>(),4);
            Or(M(11).template ReinterpretCast<uint16_t>(),M(8).template ReinterpretCast<uint16_t>(),M(9).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Not(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(M(11).template ReinterpretCast<uint16_t>(),M(11).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            NegateIf(144,DIGITS,8);NegateIf(148,DIGITS,9);
            Cast(F(206).template ReinterpretCast<float>(),F(152+DIGITS-1),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(M(8),F(206).template ReinterpretCast<float>(),16384.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();NegateIf(152,DIGITS,8);
            Multiply(156,144,DIGITS,144,DIGITS,2*DIGITS);AddNumber(108,108,156,2*DIGITS);
            Multiply(156,148,DIGITS,148,DIGITS,2*DIGITS);AddNumber(116,116,156,2*DIGITS);
            Multiply(156,152,DIGITS,152,DIGITS,2*DIGITS);AddNumber(124,124,156,2*DIGITS);
            Multiply(156,144,DIGITS,148,DIGITS,2*DIGITS);NegateIf(156,2*DIGITS,11);AddNumber(100,100,156,2*DIGITS);
            }
        }
        Multiply(172,108,2*DIGITS,116,2*DIGITS,NUMERATOR_DIGITS);Multiply(188,100,2*DIGITS,100,2*DIGITS,NUMERATOR_DIGITS);
        SubNumber(0,172,188,NUMERATOR_DIGITS);CopyNumber(16,116,2*DIGITS);
        CompareNumber(100,116,2*DIGITS,6,7);
        Not(M(12).template ReinterpretCast<uint16_t>(),M(6).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>(); // dot>=vv
        Cast(F(206).template ReinterpretCast<float>(),F(100+2*DIGITS-1),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(M(13),F(206).template ReinterpretCast<float>(),16384.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>(); // dot<0
        Zero(156,NUMERATOR_DIGITS);CompareNumber(100,156,2*DIGITS,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CompareNumber(116,156,2*DIGITS,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CopyNumber(156,124,2*DIGITS);SelectNumber(0,156,0,NUMERATOR_DIGITS,12);
        Zero(156,NUMERATOR_DIGITS);CopyNumber(156,108,2*DIGITS);SelectNumber(0,156,0,NUMERATOR_DIGITS,13);
        Or(M(12).template ReinterpretCast<uint16_t>(),M(12).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Zero(156,2*DIGITS);Duplicate(F(156),int32_t(1),64);PipeBarrier<PIPE_V>();SelectNumber(16,156,16,2*DIGITS,12);
        if constexpr(PACKET) {
            const LocalTensor<float> ids(TPosition::VECCALC,indexAddress,64);
            Copy(F(99),so.template ReinterpretCast<int32_t>(),64,1,CopyRepeatParams(1,1,8,8));
            Copy(F(96),ids.template ReinterpretCast<int32_t>(),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            M(12).template ReinterpretCast<uint64_t>().SetValue(0,~validLanes);
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        } else {
        Cast(F(99).template ReinterpretCast<float>(),so.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(F(99).template ReinterpretCast<float>(),F(99).template ReinterpretCast<float>(),1.0F/24.0F,64);PipeBarrier<PIPE_V>();
        // The fixed offset vector consists exactly of 0,24,...,1512.
        Cast(F(98),F(99).template ReinterpretCast<float>(),RoundMode::CAST_RINT,64);PipeBarrier<PIPE_V>();
        Cast(F(99).template ReinterpretCast<float>(),F(98),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(M(12),F(99).template ReinterpretCast<float>(),float(int32_t(segments)),CMPMODE::GE,64);PipeBarrier<PIPE_V>();
        }
        Zero(156,NUMERATOR_DIGITS);Duplicate(F(156),int32_t(1),64);PipeBarrier<PIPE_V>();SelectNumber(0,156,0,NUMERATOR_DIGITS,12);
        Zero(156,2*DIGITS);SelectNumber(16,156,16,2*DIGITS,12);if constexpr(!PACKET)CopyNumber(96,99,1);
    }
    template<uint32_t STAGES=6>
    __aicore__ __attribute__((always_inline)) inline void Tournament() {
        #pragma unroll 1
        for(uint32_t stage=0;stage<STAGES;++stage) {
            Cast(F(98),F(99).template ReinterpretCast<float>(),RoundMode::CAST_RINT,64);
            Duplicate(F(214).template ReinterpretCast<int32_t>(),int32_t(1U<<stage),64);PipeBarrier<PIPE_V>();
            And(F(213).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(214).template ReinterpretCast<uint16_t>(),128);
            Or(F(98).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(214).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Not(F(213).template ReinterpretCast<uint16_t>(),F(213).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            And(F(98).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(213).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(F(98).template ReinterpretCast<int32_t>(),F(98).template ReinterpretCast<int32_t>(),int32_t(4),64);PipeBarrier<PIPE_V>();
            if constexpr(DIGITS==2) {
                #pragma unroll 1
                for(uint32_t k=0;k<NUMERATOR_DIGITS;++k)Gather(F(24+k),F(k),F(98).template ReinterpretCast<uint32_t>(),0U,64);
                #pragma unroll 1
                for(uint32_t k=0;k<4;++k)Gather(F(40+k),F(16+k),F(98).template ReinterpretCast<uint32_t>(),0U,64);
            } else {
                #pragma unroll 1
                for(uint32_t k=0;k<24;++k)Gather(F(24+k),F(k),F(98).template ReinterpretCast<uint32_t>(),0U,64);
            }
            Gather(F(97),F(96),F(98).template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            Multiply(48,24,NUMERATOR_DIGITS,16,2*DIGITS,CROSS_DIGITS);Multiply(72,0,NUMERATOR_DIGITS,40,2*DIGITS,CROSS_DIGITS);
            CompareNumber(48,72,CROSS_DIGITS,6,7);
            Compare(M(0),F(97).template ReinterpretCast<float>(),F(96).template ReinterpretCast<float>(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(6).template ReinterpretCast<uint16_t>(),M(6).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            if constexpr(DIGITS==2) {
                SelectNumber(0,24,0,NUMERATOR_DIGITS,6);SelectNumber(16,40,16,4,6);
            } else {
                SelectNumber(0,24,0,24,6);
            }
            SelectNumber(96,97,96,1,6);
        }
    }
};
#endif

// A2 vector multiply-add rounds its product before accumulation. Split both
// operands into twelve-bit significands to recover the exact product error.
// Callers certify normal, finite operands and products before accepting zero.
template<uint32_t COUNT>
__aicore__ __attribute__((always_inline)) inline void ExactProductError(
    const LocalTensor<float>& error,const LocalTensor<float>& x,
    const LocalTensor<float>& y,const LocalTensor<float>& leading,
    const LocalTensor<float>& scratch) {
    auto xh=scratch,xl=scratch[64],yh=scratch[2*64],yl=scratch[3*64];
    auto lowProduct=scratch[4*64],cross=scratch[5*64];
    Duplicate(cross.template ReinterpretCast<uint32_t>(),uint32_t(0xfffff000U),COUNT);PipeBarrier<PIPE_V>();
    And(xh.template ReinterpretCast<uint16_t>(),x.template ReinterpretCast<uint16_t>(),cross.template ReinterpretCast<uint16_t>(),2*COUNT);
    And(yh.template ReinterpretCast<uint16_t>(),y.template ReinterpretCast<uint16_t>(),cross.template ReinterpretCast<uint16_t>(),2*COUNT);PipeBarrier<PIPE_V>();
    Sub(xl,x,xh,COUNT);Sub(yl,y,yh,COUNT);PipeBarrier<PIPE_V>();
    Mul(error,xh,yh,COUNT);Mul(lowProduct,xl,yl,COUNT);PipeBarrier<PIPE_V>();
    Sub(error,leading,error,COUNT);PipeBarrier<PIPE_V>();
    Mul(cross,xl,yh,COUNT);PipeBarrier<PIPE_V>();
    Sub(error,error,cross,COUNT);PipeBarrier<PIPE_V>();
    Mul(cross,xh,yl,COUNT);PipeBarrier<PIPE_V>();
    Sub(error,error,cross,COUNT);PipeBarrier<PIPE_V>();
    Sub(error,lowProduct,error,COUNT);PipeBarrier<PIPE_V>();
}

// Raw coordinate comparisons certify geometric zero without a tolerance.
// A rounded projection can otherwise leave a residual on an axis-aligned line.
template<bool BROADCAST_ENDPOINTS=false>
__aicore__ __attribute__((noinline)) void CorrectAxialZeroScores(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t scratchAddress,uint32_t scratchPitch,
    uint32_t predicateAddress,uint32_t predicatePitch,
    uint32_t count) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,count);
    const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
    const LocalTensor<float> endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch);
    const LocalTensor<float> scratch(TPosition::VECCALC,scratchAddress,3*scratchPitch);
    const LocalTensor<uint8_t> predicates(TPosition::VECCALC,predicateAddress,3*predicatePitch);
    SetMaskNorm();
    auto a=predicates,b=predicates[predicatePitch];
    auto low=scratch,high=scratch[scratchPitch],zero=scratch[2*scratchPitch];
    const uint8_t repeats=count/64;
    const uint8_t coefficientStride=BROADCAST_ENDPOINTS?0:8;
    const BinaryRepeatParams both{1,1,1,8,coefficientStride,coefficientStride};
    const BinaryRepeatParams coefficient{1,1,1,8,8,coefficientStride};
    Duplicate(zero,0.0F,count);PipeBarrier<PIPE_V>();
    #pragma unroll
    for(uint32_t axis=0;axis<3;++axis) {
        const uint32_t u=(axis+1)%3,v=(axis+2)%3;
        Compare(a,endpoints[u*endpointPitch],endpoints[(u+3)*endpointPitch],CMPMODE::EQ,uint64_t(64),repeats,both);
        Compare(b,endpoints[v*endpointPitch],endpoints[(v+3)*endpointPitch],CMPMODE::EQ,uint64_t(64),repeats,both);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Compare(b,points[u*pointPitch],endpoints[u*endpointPitch],CMPMODE::EQ,uint64_t(64),repeats,coefficient);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Compare(b,points[v*pointPitch],endpoints[v*endpointPitch],CMPMODE::EQ,uint64_t(64),repeats,coefficient);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Min(low,endpoints[axis*endpointPitch],endpoints[(axis+3)*endpointPitch],uint64_t(64),repeats,both);
        Max(high,endpoints[axis*endpointPitch],endpoints[(axis+3)*endpointPitch],uint64_t(64),repeats,both);PipeBarrier<PIPE_V>();
        Compare(b,points[axis*pointPitch],low,CMPMODE::GE,count);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Compare(b,points[axis*pointPitch],high,CMPMODE::LE,count);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Select(scores,a,zero,scores,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();
    }
    SetMaskNorm();
}

// Exact floating expansions are used only when a result can be geometric zero.
// Binary16 coordinates and their differences have a 2^-24 quantum, so products
// and their exact product errors stay in the normal binary32 range. Wider binary32 inputs
// receive an explicit operand-range certificate before a zero can be accepted.
template<bool HALF_INPUT,bool BOUNDS_CERTIFIED=false,bool REFINE_NEAR=HALF_INPUT && !BOUNDS_CERTIFIED>
__aicore__ __attribute__((noinline)) void CorrectCollinearZeroScores(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t limitAddress,
    uint32_t predicateAddress,uint32_t count,bool broadcastEndpoints=false,uint64_t candidateBits=~uint64_t(0)) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,count);
    const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
    const LocalTensor<float> endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch);
    const LocalTensor<float> storage(TPosition::VECCALC,storageAddress,45*64);
    const LocalTensor<float> limit(TPosition::VECCALC,limitAddress,64);
    const LocalTensor<uint8_t> predicates(TPosition::VECCALC,predicateAddress,96);
    auto geometry=storage,delta=storage[9*64],tmp=storage[21*64],expansion=storage[29*64];
    auto certified=predicates,test=predicates[32],other=predicates[64];
    const BinaryRepeatParams same{1,1,1,8,8,8};
    const UnaryRepeatParams unary{1,1,8,8};
    for(uint32_t start=0;start<count;start+=64) {
        SetMaskNorm();
        if constexpr(BOUNDS_CERTIFIED) {
            Duplicate(certified.ReinterpretCast<uint32_t>(),uint32_t(0),2);PipeBarrier<PIPE_V>();
            Duplicate(certified.ReinterpretCast<uint32_t>(),uint32_t(candidateBits),1);PipeBarrier<PIPE_V>();
        } else {
        WholeReduceMin(tmp,scores[start],64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Compare(test,tmp,limit,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(!(test.GetValue(0)&1U))continue;
        CompareScalar(certified,scores[start],0.0F,CMPMODE::GT,64);
        CompareScalar(test,scores[start],__builtin_inff(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
        And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(!certified.ReinterpretCast<uint64_t>().GetValue(0))continue;
        }
        SetMaskNorm();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Copy(geometry[axis*64],points[axis*pointPitch+start],64,1,CopyRepeatParams(1,1,8,8));
            Copy(geometry[(axis+3)*64],endpoints[axis*endpointPitch+(broadcastEndpoints?0:start)],64,1,CopyRepeatParams(1,1,8,8));
            Copy(geometry[(axis+6)*64],endpoints[(axis+3)*endpointPitch+(broadcastEndpoints?0:start)],64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        if constexpr(!BOUNDS_CERTIFIED && !REFINE_NEAR) {
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Min(tmp,geometry[(axis+3)*64],geometry[(axis+6)*64],64);
            Max(tmp[64],geometry[(axis+3)*64],geometry[(axis+6)*64],64);PipeBarrier<PIPE_V>();
            Compare(test,geometry[axis*64],tmp,CMPMODE::GE,64);
            Compare(other,geometry[axis*64],tmp[64],CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            And(test.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),other.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        }
        // A point outside the segment's coordinate bounds cannot be collinear
        // within that segment. Subsequent predicates only clear these bits.
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(!certified.ReinterpretCast<uint64_t>().GetValue(0))continue;
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            #pragma unroll
            for(uint32_t kind=0;kind<2;++kind) {
                auto x=geometry[(kind==0?axis:axis+6)*64],y=geometry[(axis+3)*64];
                auto hi=delta[(kind*3+axis)*64],lo=delta[(6+kind*3+axis)*64];
                Sub(hi,x,y,64);PipeBarrier<PIPE_V>();
                Sub(tmp,x,hi,64);PipeBarrier<PIPE_V>();
                Add(tmp[64],hi,tmp,64);Sub(lo,tmp,y,64);PipeBarrier<PIPE_V>();
                Sub(tmp[64],x,tmp[64],64);PipeBarrier<PIPE_V>();
                Add(lo,lo,tmp[64],64);PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(REFINE_NEAR) {
            // Test the projection against both endpoints without subtracting
            // two large, almost equal squared lengths.
            Duplicate(tmp,0.0F,64);Duplicate(tmp[64],0.0F,64);PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Mul(tmp[2*64],delta[axis*64],delta[(3+axis)*64],64);
                Mul(tmp[3*64],delta[(6+axis)*64],delta[(3+axis)*64],64);PipeBarrier<PIPE_V>();
                MulAddDst(tmp[2*64],delta[axis*64],delta[(9+axis)*64],64);
                MulAddDst(tmp[3*64],delta[(6+axis)*64],delta[(9+axis)*64],64);PipeBarrier<PIPE_V>();
                Add(tmp[2*64],tmp[2*64],tmp[3*64],64);PipeBarrier<PIPE_V>();
                Add(tmp,tmp,tmp[2*64],64);
                Sub(tmp[4*64],geometry[axis*64],geometry[(6+axis)*64],64);PipeBarrier<PIPE_V>();
                Sub(tmp[6*64],geometry[axis*64],tmp[4*64],64);PipeBarrier<PIPE_V>();
                Add(tmp[7*64],tmp[4*64],tmp[6*64],64);
                Sub(tmp[5*64],tmp[6*64],geometry[(6+axis)*64],64);PipeBarrier<PIPE_V>();
                Sub(tmp[7*64],geometry[axis*64],tmp[7*64],64);PipeBarrier<PIPE_V>();
                Add(tmp[5*64],tmp[5*64],tmp[7*64],64);PipeBarrier<PIPE_V>();
                Mul(tmp[2*64],tmp[4*64],delta[(3+axis)*64],64);
                Mul(tmp[3*64],tmp[5*64],delta[(3+axis)*64],64);PipeBarrier<PIPE_V>();
                MulAddDst(tmp[2*64],tmp[4*64],delta[(9+axis)*64],64);
                MulAddDst(tmp[3*64],tmp[5*64],delta[(9+axis)*64],64);PipeBarrier<PIPE_V>();
                Add(tmp[2*64],tmp[2*64],tmp[3*64],64);PipeBarrier<PIPE_V>();
                Add(tmp[64],tmp[64],tmp[2*64],64);PipeBarrier<PIPE_V>();
            }
            CompareScalar(test,tmp,0.0F,CMPMODE::GE,64);
            CompareScalar(other,tmp[64],0.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),other.ReinterpretCast<uint16_t>(),4);
            Mul(tmp,delta[3*64],delta[3*64],64);PipeBarrier<PIPE_V>();
            MulAddDst(tmp,delta[4*64],delta[4*64],64);PipeBarrier<PIPE_V>();
            MulAddDst(tmp,delta[5*64],delta[5*64],64);PipeBarrier<PIPE_V>();
            CompareScalar(test,tmp,0.0F,CMPMODE::GT,64);PipeBarrier<PIPE_V>();
            And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        Abs(tmp,delta[6*64],6*64);PipeBarrier<PIPE_V>();
        // Only surviving candidates affect the expansion's component counts.
        #pragma unroll
        for(uint32_t axis=0;axis<6;++axis)
            Select(tmp[axis*64],certified,tmp[axis*64],0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        WholeReduceMax(tmp[6*64],tmp,64,3,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax(tmp[6*64+8],tmp[3*64],64,3,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(tmp[7*64],tmp[6*64],3,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax(tmp[7*64+8],tmp[6*64+8],3,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        CompareScalar(test,tmp[7*64],0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t componentBits=test.ReinterpretCast<uint16_t>().GetValue(0);
        const uint32_t pointComponents=(componentBits&1U)?1:2;
        const uint32_t directionComponents=(componentBits&256U)?1:2;
        if constexpr(!HALF_INPUT) {
            const uint32_t checkedColumns=6+3*(pointComponents-1)+3*(directionComponents-1);
            for(uint32_t k=0;k<checkedColumns;++k) {
                const uint32_t col=k>=6 && pointComponents==1?k+3:k;
                Abs(tmp,delta[col*64],64);PipeBarrier<PIPE_V>();
                CompareScalar(test,tmp,9.313225746154785e-10F,CMPMODE::GE,64);
                CompareScalar(other,tmp,0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
                Or(test.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),other.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);
                CompareScalar(test,tmp,9.223372036854776e18F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
                And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t safeCandidates=certified.ReinterpretCast<uint64_t>().GetValue(0);
        if(!safeCandidates)continue;
        if(pointComponents==1 && directionComponents==1) {
            // With exact coordinate differences, each cross-product has two
            // products. Equality of both rounded products and exact errors
            // is an exact equality certificate, avoiding expansion merging.
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                const uint32_t next=(axis+1)%3;
                Mul(tmp,delta[axis*64],delta[(3+next)*64],64);
                Mul(tmp[2*64],delta[next*64],delta[(3+axis)*64],64);PipeBarrier<PIPE_V>();
                ExactProductError<64>(tmp[64],delta[axis*64],delta[(3+next)*64],tmp,expansion);
                ExactProductError<64>(tmp[3*64],delta[next*64],delta[(3+axis)*64],tmp[2*64],expansion);
                if constexpr(BOUNDS_CERTIFIED || REFINE_NEAR) {
                    Sub(geometry[axis*64],tmp,tmp[2*64],64);
                    Sub(tmp[4*64],tmp[64],tmp[3*64],64);PipeBarrier<PIPE_V>();
                    Add(geometry[axis*64],geometry[axis*64],tmp[4*64],64);PipeBarrier<PIPE_V>();
                }
                Compare(test,tmp,tmp[2*64],CMPMODE::EQ,64);
                Compare(other,tmp[64],tmp[3*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
                And(test.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),other.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            }
        } else {
        for(uint32_t axis=0;axis<3;++axis) {
            const uint32_t next=(axis+1)%3;
            uint32_t limbs=0;
            for(uint32_t sign=0;sign<2;++sign)
                for(uint32_t al=0;al<pointComponents;++al)
                    for(uint32_t dl=0;dl<directionComponents;++dl) {
                        auto x=delta[(al*6+(sign?next:axis))*64];
                        auto y=delta[(3+dl*6+(sign?axis:next))*64];
                        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                        Mul<float,false>(tmp,x,y,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                        ExactProductError<64>(tmp[64],x,y,tmp,tmp[2*64]);
                        if(sign) {
                            Muls<float,false>(tmp,tmp,-1.0F,uint64_t(64),1,unary);
                            Muls<float,false>(tmp[64],tmp[64],-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                        }
                        for(uint32_t part=0;part<2;++part) {
                            Copy(tmp[2*64],tmp[(1-part)*64],64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                            for(uint32_t k=0;k<limbs;++k) {
                                Add<float,false>(tmp[3*64],tmp[2*64],expansion[k*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                                Sub<float,false>(tmp[4*64],tmp[3*64],tmp[2*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                                Sub<float,false>(tmp[5*64],tmp[3*64],tmp[4*64],uint64_t(64),1,same);
                                Sub<float,false>(tmp[6*64],expansion[k*64],tmp[4*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                                Sub<float,false>(tmp[7*64],tmp[2*64],tmp[5*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                                Add<float,false>(expansion[k*64],tmp[7*64],tmp[6*64],uint64_t(64),1,same);
                                Copy(tmp[2*64],tmp[3*64],64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                            }
                            Copy(expansion[limbs*64],tmp[2*64],64,1,CopyRepeatParams(1,1,8,8));++limbs;PipeBarrier<PIPE_V>();
                        }
                    }
            for(uint32_t k=0;k<limbs;++k) {
                CompareScalar(test,expansion[k*64],0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
                And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            }
            if constexpr(BOUNDS_CERTIFIED || REFINE_NEAR) {
                Duplicate(geometry[axis*64],0.0F,64);PipeBarrier<PIPE_V>();
                for(uint32_t k=0;k<limbs;++k) {
                    Add(geometry[axis*64],geometry[axis*64],expansion[k*64],64);PipeBarrier<PIPE_V>();
                }
            }
        }
        }
        Duplicate(tmp,0.0F,64);PipeBarrier<PIPE_V>();
        Select(scores[start],certified,tmp,scores[start],SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        if constexpr(BOUNDS_CERTIFIED || REFINE_NEAR) {
            constexpr uint32_t scoreLanes=REFINE_NEAR?64:8;
            // Interior projection predicates permit the compensated cross
            // product avoids subtracting a rounded closest point from P.
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Add(tmp[(2+axis)*64],delta[(3+axis)*64],delta[(9+axis)*64],scoreLanes);
            PipeBarrier<PIPE_V>();
            Mul(tmp,tmp[2*64],tmp[2*64],scoreLanes);
            Mul(tmp[64],geometry,geometry,scoreLanes);PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=1;axis<3;++axis) {
                MulAddDst(tmp,tmp[(2+axis)*64],tmp[(2+axis)*64],scoreLanes);
                MulAddDst(tmp[64],geometry[axis*64],geometry[axis*64],scoreLanes);PipeBarrier<PIPE_V>();
            }
            CompareScalar(test,tmp,0.0F,CMPMODE::GT,64);PipeBarrier<PIPE_V>();
            Select(tmp,test,tmp,1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,scoreLanes);PipeBarrier<PIPE_V>();
            Div(tmp[64],tmp[64],tmp,scoreLanes);PipeBarrier<PIPE_V>();
            Duplicate(other.ReinterpretCast<uint32_t>(),uint32_t(0),2);PipeBarrier<PIPE_V>();
            Duplicate(other.ReinterpretCast<uint32_t>(),uint32_t(safeCandidates),REFINE_NEAR?2:1);PipeBarrier<PIPE_V>();
            if constexpr(REFINE_NEAR) {
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(2));
                Duplicate<uint32_t,false>(other.ReinterpretCast<uint32_t>(),uint32_t(safeCandidates>>32),uint64_t(64),1,1,8);
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();
            }
            CompareScalar(test,tmp[64],__builtin_inff(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            And(other.ReinterpretCast<uint16_t>(),other.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),REFINE_NEAR?4:1);PipeBarrier<PIPE_V>();
            Select(scores[start],other,tmp[64],scores[start],SELMODE::VSEL_TENSOR_TENSOR_MODE,scoreLanes);PipeBarrier<PIPE_V>();
        }
    }
    SetMaskNorm();
}

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
__aicore__ __attribute__((noinline)) inline void CorrectExactLineScores(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t limitAddress,
    uint32_t predicateAddress,uint32_t count,bool broadcastEndpoints=false,uint64_t candidateBits=~uint64_t(0)) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,count);
    const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
    const LocalTensor<float> endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch);
    const LocalTensor<float> storage(TPosition::VECCALC,storageAddress,45*64);
    const LocalTensor<float> limit(TPosition::VECCALC,limitAddress,64);
    const LocalTensor<uint8_t> predicates(TPosition::VECCALC,predicateAddress,96);
    auto geometry=storage,delta=storage[9*64],tmp=storage[21*64],expansion=storage[29*64];
    auto certified=predicates,test=predicates[32],other=predicates[64];
    const BinaryRepeatParams same{1,1,1,8,8,8};
    const UnaryRepeatParams unary{1,1,8,8};
    for(uint32_t start=0;start<count;start+=64) {
        SetMaskNorm();
        {
            Duplicate(certified.ReinterpretCast<uint32_t>(),uint32_t(0),2);PipeBarrier<PIPE_V>();
            Duplicate(certified.ReinterpretCast<uint32_t>(),uint32_t(candidateBits),1);PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Copy(geometry[axis*64],points[axis*pointPitch+start],64,1,CopyRepeatParams(1,1,8,8));
            Copy(geometry[(axis+3)*64],endpoints[axis*endpointPitch+(broadcastEndpoints?0:start)],64,1,CopyRepeatParams(1,1,8,8));
            Copy(geometry[(axis+6)*64],endpoints[(axis+3)*endpointPitch+(broadcastEndpoints?0:start)],64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        // Incoming certified lanes are narrowed by exact line predicates.
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(!certified.ReinterpretCast<uint64_t>().GetValue(0))continue;
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            #pragma unroll
            for(uint32_t kind=0;kind<2;++kind) {
                auto x=geometry[(kind==0?axis:axis+6)*64],y=geometry[(axis+3)*64];
                auto hi=delta[(kind*3+axis)*64],lo=delta[(6+kind*3+axis)*64];
                Sub(hi,x,y,64);PipeBarrier<PIPE_V>();
                Sub(tmp,x,hi,64);PipeBarrier<PIPE_V>();
                Add(tmp[64],hi,tmp,64);Sub(lo,tmp,y,64);PipeBarrier<PIPE_V>();
                Sub(tmp[64],x,tmp[64],64);PipeBarrier<PIPE_V>();
                Add(lo,lo,tmp[64],64);PipeBarrier<PIPE_V>();
            }
        }
        Abs(tmp,delta[6*64],6*64);PipeBarrier<PIPE_V>();
        // Only surviving candidates affect the expansion's component counts.
        #pragma unroll
        for(uint32_t axis=0;axis<6;++axis)
            Select(tmp[axis*64],certified,tmp[axis*64],0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        WholeReduceMax(tmp[6*64],tmp,64,3,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax(tmp[6*64+8],tmp[3*64],64,3,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(tmp[7*64],tmp[6*64],3,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax(tmp[7*64+8],tmp[6*64+8],3,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        CompareScalar(test,tmp[7*64],0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t componentBits=test.ReinterpretCast<uint16_t>().GetValue(0);
        const uint32_t pointComponents=(componentBits&1U)?1:2;
        const uint32_t directionComponents=(componentBits&256U)?1:2;
        {
            const uint32_t checkedColumns=6+3*(pointComponents-1)+3*(directionComponents-1);
            for(uint32_t k=0;k<checkedColumns;++k) {
                const uint32_t col=k>=6 && pointComponents==1?k+3:k;
                Abs(tmp,delta[col*64],64);PipeBarrier<PIPE_V>();
                CompareScalar(test,tmp,9.313225746154785e-10F,CMPMODE::GE,64);
                CompareScalar(other,tmp,0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
                Or(test.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),other.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);
                CompareScalar(test,tmp,9.223372036854776e18F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
                And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t safeCandidates=certified.ReinterpretCast<uint64_t>().GetValue(0);
        if(!safeCandidates)continue;
        if(pointComponents==1 && directionComponents==1) {
            // With exact coordinate differences, each cross-product has two
            // products. Equality of both rounded products and exact errors
            // is an exact equality certificate, avoiding expansion merging.
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                const uint32_t next=(axis+1)%3;
                Mul(tmp,delta[axis*64],delta[(3+next)*64],64);
                Mul(tmp[2*64],delta[next*64],delta[(3+axis)*64],64);PipeBarrier<PIPE_V>();
                ExactProductError<64>(tmp[64],delta[axis*64],delta[(3+next)*64],tmp,expansion);
                ExactProductError<64>(tmp[3*64],delta[next*64],delta[(3+axis)*64],tmp[2*64],expansion);
                {
                    Sub(geometry[axis*64],tmp,tmp[2*64],64);
                    Sub(tmp[4*64],tmp[64],tmp[3*64],64);PipeBarrier<PIPE_V>();
                    Add(geometry[axis*64],geometry[axis*64],tmp[4*64],64);PipeBarrier<PIPE_V>();
                }
                Compare(test,tmp,tmp[2*64],CMPMODE::EQ,64);
                Compare(other,tmp[64],tmp[3*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
                And(test.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),other.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            }
        } else {
        for(uint32_t axis=0;axis<3;++axis) {
            const uint32_t next=(axis+1)%3;
            uint32_t limbs=0;
            for(uint32_t sign=0;sign<2;++sign)
                for(uint32_t al=0;al<pointComponents;++al)
                    for(uint32_t dl=0;dl<directionComponents;++dl) {
                        auto x=delta[(al*6+(sign?next:axis))*64];
                        auto y=delta[(3+dl*6+(sign?axis:next))*64];
                        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                        Mul<float,false>(tmp,x,y,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                        ExactProductError<64>(tmp[64],x,y,tmp,tmp[2*64]);
                        if(sign) {
                            Muls<float,false>(tmp,tmp,-1.0F,uint64_t(64),1,unary);
                            Muls<float,false>(tmp[64],tmp[64],-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                        }
                        for(uint32_t part=0;part<2;++part) {
                            Copy(tmp[2*64],tmp[(1-part)*64],64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                            for(uint32_t k=0;k<limbs;++k) {
                                Add<float,false>(tmp[3*64],tmp[2*64],expansion[k*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                                Sub<float,false>(tmp[4*64],tmp[3*64],tmp[2*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                                Sub<float,false>(tmp[5*64],tmp[3*64],tmp[4*64],uint64_t(64),1,same);
                                Sub<float,false>(tmp[6*64],expansion[k*64],tmp[4*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                                Sub<float,false>(tmp[7*64],tmp[2*64],tmp[5*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                                Add<float,false>(expansion[k*64],tmp[7*64],tmp[6*64],uint64_t(64),1,same);
                                Copy(tmp[2*64],tmp[3*64],64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                            }
                            Copy(expansion[limbs*64],tmp[2*64],64,1,CopyRepeatParams(1,1,8,8));++limbs;PipeBarrier<PIPE_V>();
                        }
                    }
            for(uint32_t k=0;k<limbs;++k) {
                CompareScalar(test,expansion[k*64],0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
                And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            }
            {
                Duplicate(geometry[axis*64],0.0F,64);PipeBarrier<PIPE_V>();
                for(uint32_t k=0;k<limbs;++k) {
                    Add(geometry[axis*64],geometry[axis*64],expansion[k*64],64);PipeBarrier<PIPE_V>();
                }
            }
        }
        }
        Duplicate(tmp,0.0F,64);PipeBarrier<PIPE_V>();
        Select(scores[start],certified,tmp,scores[start],SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();

    }
    SetMaskNorm();
}

// Equal supporting lines with strictly interior projections have the same
// exact distance. Only certified lanes reuse the incumbent's computed score.
// Reuse the original necessary-condition rejection before entering the exact
// certificate. A surviving predicate always executes the complete correction.
__aicore__ __attribute__((always_inline)) inline bool MayHaveCollinearInteriorTies8(
    uint32_t scoreAddress,uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t predicateAddress) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,8);
    const LocalTensor<float> endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch);
    const LocalTensor<float> storage(TPosition::VECCALC,storageAddress,65*64);
    const LocalTensor<uint8_t> predicates(TPosition::VECCALC,predicateAddress,96);
    auto winnerEndpoints=storage[45*64],tmp=storage[51*64],record=storage[64*64];
    auto certified=predicates,test=predicates[32],other=predicates[64];
    PipeBarrier<PIPE_V>();SetMaskNorm();
    WholeReduceMin(record,scores,8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t winner=record.ReinterpretCast<uint32_t>().GetValue(1);
    if(winner>=8){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return false;}
    CompareScalar<float,uint8_t,true>(certified,scores,__builtin_inff(),CMPMODE::LT,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    Duplicate(tmp[12*64].ReinterpretCast<uint32_t>(),winner*4,64);PipeBarrier<PIPE_V>();
    #pragma unroll
    for(uint32_t col=0;col<6;++col)if(col!=2 && col!=5)
        Gather(winnerEndpoints[col*64],endpoints[col*endpointPitch],tmp[12*64].ReinterpretCast<uint32_t>(),0U,64);
    PipeBarrier<PIPE_V>();
    // A necessary common-line condition only. The exact certificate below
    // still decides equality; a rounded cross product never accepts a tie.
    Sub(tmp,winnerEndpoints[3*64],winnerEndpoints,8);
    Sub(tmp[64],winnerEndpoints[4*64],winnerEndpoints[64],8);
    Sub(tmp[2*64],endpoints,winnerEndpoints,8);
    Sub(tmp[3*64],endpoints[endpointPitch],winnerEndpoints[64],8);PipeBarrier<PIPE_V>();
    Mul(tmp[4*64],tmp[2*64],tmp[64],8);
    Mul(tmp[5*64],tmp[3*64],tmp,8);PipeBarrier<PIPE_V>();
    Sub(tmp[6*64],tmp[4*64],tmp[5*64],8);
    Abs(tmp[4*64],tmp[4*64],8);Abs(tmp[5*64],tmp[5*64],8);PipeBarrier<PIPE_V>();
    Abs(tmp[6*64],tmp[6*64],8);
    Add(tmp[7*64],tmp[4*64],tmp[5*64],8);PipeBarrier<PIPE_V>();
    Muls(tmp[7*64],tmp[7*64],3.0517578125e-5F,8);PipeBarrier<PIPE_V>();
    Compare<float,uint8_t,true>(test,tmp[6*64],tmp[7*64],CMPMODE::LE,uint64_t(8),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
    And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t possible=certified.ReinterpretCast<uint16_t>().GetValue(0)&255U;
    const bool possibleTie=(possible&(1U<<winner)) && (possible&(possible-1));
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return possibleTie;
}

__aicore__ __attribute__((always_inline)) inline bool MayHaveCollinearInteriorTies8NormalMask(
    uint32_t scoreAddress,uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t predicateAddress) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,8);
    const LocalTensor<float> endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch);
    const LocalTensor<float> storage(TPosition::VECCALC,storageAddress,65*64);
    const LocalTensor<uint8_t> predicates(TPosition::VECCALC,predicateAddress,96);
    auto winnerEndpoints=storage[45*64],tmp=storage[51*64],record=storage[64*64];
    auto certified=predicates,test=predicates[32],other=predicates[64];
    PipeBarrier<PIPE_V>();SetMaskNorm();
    WholeReduceMin(record,scores,8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t winner=record.ReinterpretCast<uint32_t>().GetValue(1);
    if(winner>=8){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return false;}
    CompareScalar<float,uint8_t,true>(certified,scores,__builtin_inff(),CMPMODE::LT,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    Duplicate(tmp[12*64].ReinterpretCast<uint32_t>(),winner*4,64);PipeBarrier<PIPE_V>();
    #pragma unroll
    for(uint32_t col=0;col<6;++col)if(col!=2 && col!=5)
        Gather(winnerEndpoints[col*64],endpoints[col*endpointPitch],tmp[12*64].ReinterpretCast<uint32_t>(),0U,64);
    PipeBarrier<PIPE_V>();
    // A necessary common-line condition only. The exact certificate below
    // still decides equality; a rounded cross product never accepts a tie.
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
    const BinaryRepeatParams binary{1,1,1,8,8,8};
    const UnaryRepeatParams unary{1,1,8,8};
    Sub<float,false>(tmp,winnerEndpoints[3*64],winnerEndpoints,uint64_t(8),1,binary);
    Sub<float,false>(tmp[64],winnerEndpoints[4*64],winnerEndpoints[64],uint64_t(8),1,binary);
    Sub<float,false>(tmp[2*64],endpoints,winnerEndpoints,uint64_t(8),1,binary);
    Sub<float,false>(tmp[3*64],endpoints[endpointPitch],winnerEndpoints[64],uint64_t(8),1,binary);PipeBarrier<PIPE_V>();
    Mul<float,false>(tmp[4*64],tmp[2*64],tmp[64],uint64_t(8),1,binary);
    Mul<float,false>(tmp[5*64],tmp[3*64],tmp,uint64_t(8),1,binary);PipeBarrier<PIPE_V>();
    Sub<float,false>(tmp[6*64],tmp[4*64],tmp[5*64],uint64_t(8),1,binary);
    Abs<float,false>(tmp[4*64],tmp[4*64],uint64_t(8),1,unary);Abs<float,false>(tmp[5*64],tmp[5*64],uint64_t(8),1,unary);PipeBarrier<PIPE_V>();
    Abs<float,false>(tmp[6*64],tmp[6*64],uint64_t(8),1,unary);
    Add<float,false>(tmp[7*64],tmp[4*64],tmp[5*64],uint64_t(8),1,binary);PipeBarrier<PIPE_V>();
    Muls<float,false>(tmp[7*64],tmp[7*64],3.0517578125e-5F,uint64_t(8),1,unary);PipeBarrier<PIPE_V>();
    Compare<float,uint8_t,false>(test,tmp[6*64],tmp[7*64],CMPMODE::LE,uint64_t(8),1,binary);PipeBarrier<PIPE_V>();
    SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(1));
    And<uint16_t,false>(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),uint64_t(1),1,binary);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t possible=certified.ReinterpretCast<uint16_t>().GetValue(0)&255U;
    const bool possibleTie=(possible&(1U<<winner)) && (possible&(possible-1));
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return possibleTie;
}

__aicore__ __attribute__((noinline)) inline void CorrectCollinearInteriorTies8(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t predicateAddress) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,8);
    const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
    const LocalTensor<float> endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch);
    const LocalTensor<float> storage(TPosition::VECCALC,storageAddress,65*64);
    const LocalTensor<uint8_t> predicates(TPosition::VECCALC,predicateAddress,96);
    auto winnerEndpoints=storage[45*64],tmp=storage[51*64],record=storage[64*64];
    auto certified=predicates,test=predicates[32],other=predicates[64];
    PipeBarrier<PIPE_V>();SetMaskNorm();
    WholeReduceMin(record,scores,8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t winner=record.ReinterpretCast<uint32_t>().GetValue(1);
    if(winner>=8){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return;}
    CompareScalar<float,uint8_t,true>(certified,scores,__builtin_inff(),CMPMODE::LT,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    Duplicate(tmp[12*64].ReinterpretCast<uint32_t>(),winner*4,64);PipeBarrier<PIPE_V>();
    #pragma unroll
    for(uint32_t col=0;col<6;++col)if(col!=2 && col!=5)
        Gather(winnerEndpoints[col*64],endpoints[col*endpointPitch],tmp[12*64].ReinterpretCast<uint32_t>(),0U,64);
    PipeBarrier<PIPE_V>();
    // A necessary common-line condition only. The exact certificate below
    // still decides equality; a rounded cross product never accepts a tie.
    Sub(tmp,winnerEndpoints[3*64],winnerEndpoints,8);
    Sub(tmp[64],winnerEndpoints[4*64],winnerEndpoints[64],8);
    Sub(tmp[2*64],endpoints,winnerEndpoints,8);
    Sub(tmp[3*64],endpoints[endpointPitch],winnerEndpoints[64],8);PipeBarrier<PIPE_V>();
    Mul(tmp[4*64],tmp[2*64],tmp[64],8);
    Mul(tmp[5*64],tmp[3*64],tmp,8);PipeBarrier<PIPE_V>();
    Sub(tmp[6*64],tmp[4*64],tmp[5*64],8);
    Abs(tmp[4*64],tmp[4*64],8);Abs(tmp[5*64],tmp[5*64],8);PipeBarrier<PIPE_V>();
    Abs(tmp[6*64],tmp[6*64],8);
    Add(tmp[7*64],tmp[4*64],tmp[5*64],8);PipeBarrier<PIPE_V>();
    Muls(tmp[7*64],tmp[7*64],3.0517578125e-5F,8);PipeBarrier<PIPE_V>();
    Compare<float,uint8_t,true>(test,tmp[6*64],tmp[7*64],CMPMODE::LE,uint64_t(8),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
    And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t possible=certified.ReinterpretCast<uint16_t>().GetValue(0)&255U;
    if(!(possible&(1U<<winner)) || !(possible&(possible-1))){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return;}
    #pragma unroll
    for(uint32_t axis=0;axis<3;++axis) {
        Sub(tmp[axis*64],endpoints[(axis+3)*endpointPitch],endpoints[axis*endpointPitch],8);
        Sub(tmp[(axis+3)*64],points[axis*pointPitch],endpoints[axis*endpointPitch],8);
    }
    PipeBarrier<PIPE_V>();
    // The normal-operand range keeps every product and rounding bound finite
    // or rejects the lane; it also avoids underflow in the dot certificate.
    #pragma unroll
    for(uint32_t k=0;k<6;++k) {
        Abs(tmp[9*64],tmp[k*64],8);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,true>(test,tmp[9*64],9.313225746154785e-10F,CMPMODE::GE,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});
        CompareScalar<float,uint8_t,true>(other,tmp[9*64],0.0F,CMPMODE::EQ,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Or(test.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),other.ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
        And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),1);
        CompareScalar<float,uint8_t,true>(test,tmp[9*64],9.223372036854776e18F,CMPMODE::LE,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    }
    Mul(tmp[6*64],tmp[3*64],tmp,8);
    Mul(tmp[7*64],tmp,tmp,8);PipeBarrier<PIPE_V>();
    #pragma unroll
    for(uint32_t axis=1;axis<3;++axis) {
        MulAddDst(tmp[6*64],tmp[(axis+3)*64],tmp[axis*64],8);
        MulAddDst(tmp[7*64],tmp[axis*64],tmp[axis*64],8);PipeBarrier<PIPE_V>();
    }
    #pragma unroll
    for(uint32_t k=0;k<6;++k)Abs(tmp[k*64],tmp[k*64],8);
    PipeBarrier<PIPE_V>();
    Mul(tmp[8*64],tmp[3*64],tmp,8);PipeBarrier<PIPE_V>();
    #pragma unroll
    for(uint32_t axis=1;axis<3;++axis) {
        MulAddDst(tmp[8*64],tmp[(axis+3)*64],tmp[axis*64],8);PipeBarrier<PIPE_V>();
    }
    Add(tmp[10*64],tmp[8*64],tmp[7*64],8);
    Sub(tmp[9*64],tmp[7*64],tmp[6*64],8);PipeBarrier<PIPE_V>();
    Muls(tmp[10*64],tmp[10*64],3.0517578125e-5F,8);PipeBarrier<PIPE_V>();
    Compare<float,uint8_t,true>(test,tmp[6*64],tmp[10*64],CMPMODE::GT,uint64_t(8),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});
    Compare<float,uint8_t,true>(other,tmp[9*64],tmp[10*64],CMPMODE::GT,uint64_t(8),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
    And(test.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),other.ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t bits=certified.ReinterpretCast<uint16_t>().GetValue(0)&255U;
    if(!(bits&(1U<<winner)) || !(bits&(bits-1))){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return;}
    // The XY rejection and projection certificate do not read winner z.
    // Its byte-offset vector remains live in tmp[12*64] until this point.
    Gather(winnerEndpoints[2*64],endpoints[2*endpointPitch],tmp[12*64].ReinterpretCast<uint32_t>(),0U,64);
    Gather(winnerEndpoints[5*64],endpoints[5*endpointPitch],tmp[12*64].ReinterpretCast<uint32_t>(),0U,64);
    PipeBarrier<PIPE_V>();
    for(uint32_t end=0;end<2;++end) {
        Duplicate(tmp,0.0F,3*64);Duplicate(tmp[11*64],1.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Copy(tmp[axis*64],endpoints[(end*3+axis)*endpointPitch],8,1,CopyRepeatParams(1,1,1,1));
        PipeBarrier<PIPE_V>();
        CorrectExactLineScores(uint32_t(tmp[11*64].GetPhyAddr()),uint32_t(tmp.GetPhyAddr()),64,
            uint32_t(winnerEndpoints.GetPhyAddr()),64,storageAddress,uint32_t(record.GetPhyAddr()),
            predicateAddress,64,false,uint64_t(bits));
        CompareScalar<float,uint8_t,true>(test,tmp[11*64],0.0F,CMPMODE::EQ,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        bits&=test.ReinterpretCast<uint16_t>().GetValue(0)&255U;
        if(!(bits&(bits-1))){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return;}
    }
    Duplicate(certified.ReinterpretCast<uint32_t>(),bits,1);PipeBarrier<PIPE_V>();
    Brcb(tmp[3*64],record,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    Select(scores,certified,tmp[3*64],scores,SELMODE::VSEL_TENSOR_TENSOR_MODE,8);PipeBarrier<PIPE_V>();
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}
#endif

template<bool HALF_INPUT,uint32_t RECORDS=8>
__aicore__ __attribute__((noinline)) void CorrectTinyZeroScores(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t aAddress,uint32_t cAddress,
    uint32_t workAddress,uint32_t geometryAddress,uint32_t exactAddress,
    uint32_t predicateAddress,uint32_t offsetAddress,uint64_t activePairs) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,64);
    const LocalTensor<float> p(TPosition::VECCALC,pointAddress,64);
    const LocalTensor<float> a(TPosition::VECCALC,aAddress,64);
    const LocalTensor<float> c(TPosition::VECCALC,cAddress,64);
    const LocalTensor<float> work(TPosition::VECCALC,workAddress,8*64);
    const LocalTensor<float> geometry(TPosition::VECCALC,geometryAddress,9*64);
    const LocalTensor<float> exact(TPosition::VECCALC,exactAddress,45*64);
    const LocalTensor<uint8_t> masks(TPosition::VECCALC,predicateAddress,512);
    const LocalTensor<int32_t> offsets(TPosition::VECCALC,offsetAddress,64);
    constexpr uint32_t coordinates=RECORDS*8;
                // Certify axis-aligned zero and coordinate bounds directly
                // from the existing eight-coordinate records.
                Min(work[0*64],a,c,coordinates);Max(work[1*64],a,c,coordinates);
                CompareScalar(masks,scores,0.0F,CMPMODE::GT,64);
                CompareScalar(masks[320],scores,__builtin_inff(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
                Compare(masks[64],p,work[0*64],CMPMODE::GE,coordinates);
                Compare(masks[128],p,work[1*64],CMPMODE::LE,coordinates);
                Compare(masks[192],a,c,CMPMODE::EQ,coordinates);
                Compare(masks[256],p,a,CMPMODE::EQ,coordinates);PipeBarrier<PIPE_V>();
                And(masks[64].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),4);
                And(masks[192].template ReinterpretCast<uint16_t>(),masks[192].template ReinterpretCast<uint16_t>(),masks[256].template ReinterpretCast<uint16_t>(),4);
                And(masks.template ReinterpretCast<uint16_t>(),masks.template ReinterpretCast<uint16_t>(),masks[320].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const uint64_t coordinateBounds=masks[64].template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t fixedCoordinates=masks[192].template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t positiveFinite=masks.template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t bounded=((coordinateBounds&(coordinateBounds>>1)&(coordinateBounds>>2))&0x0101010101010101ULL);
                const uint64_t axial=((fixedCoordinates&(fixedCoordinates>>1))|(fixedCoordinates&(fixedCoordinates>>2))|((fixedCoordinates>>1)&(fixedCoordinates>>2)))&bounded;
                const uint64_t boundedPairs=(bounded*0x0102040810204080ULL)>>56;
                const uint64_t axialPairs=(axial*0x0102040810204080ULL)>>56;
                Duplicate(masks[256].template ReinterpretCast<uint32_t>(),uint32_t(~axialPairs),2);PipeBarrier<PIPE_V>();
                Select(scores,masks[256],scores,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                const uint64_t candidates=positiveFinite&~axialPairs&activePairs&(HALF_INPUT?~uint64_t(0):boundedPairs);
                if(candidates) {
                    Duplicate(geometry,0.0F,9*64);PipeBarrier<PIPE_V>();
                    Duplicate(offsets,int32_t(0),8);PipeBarrier<PIPE_V>();
                    SetMaskNorm();
                    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaa));
                    Adds<int32_t,false>(offsets,offsets,int32_t(1),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xcc));
                    Adds<int32_t,false>(offsets,offsets,int32_t(2),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xf0));
                    Adds<int32_t,false>(offsets,offsets,int32_t(4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();Muls(offsets,offsets,int32_t(32),8);PipeBarrier<PIPE_V>();
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis) {
                        Gather(geometry[axis*64],p,offsets.template ReinterpretCast<uint32_t>(),axis*4,RECORDS);
                        Gather(geometry[(axis+3)*64],a,offsets.template ReinterpretCast<uint32_t>(),axis*4,RECORDS);
                        Gather(geometry[(axis+6)*64],c,offsets.template ReinterpretCast<uint32_t>(),axis*4,RECORDS);
                    }
                    PipeBarrier<PIPE_V>();
                    CorrectCollinearZeroScores<HALF_INPUT,true,HALF_INPUT>(uint32_t(scores.GetPhyAddr()),uint32_t(geometry.GetPhyAddr()),64,
                        uint32_t(geometry[3*64].GetPhyAddr()),64,uint32_t(exact.GetPhyAddr()),uint32_t(work[5*64].GetPhyAddr()),
                        uint32_t(masks.GetPhyAddr()),64,false,candidates);
                }
                Select(scores,masks[384],scores,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
    SetMaskNorm();
}

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
// Finite binary16 coordinates are integral after multiplication by 2^24.
// Base4096 digits fit exact binary32 multiply/add packets: the maximum
// normalized product plus destination and carry is 2^24-1.
class TreeHalfRational64 {
    LocalTensor<float> f_;
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> F(uint32_t p) { return f_[p*64]; }
    __aicore__ __attribute__((always_inline)) inline LocalTensor<uint8_t> M(uint32_t p) { return F(220).ReinterpretCast<uint8_t>()[p*32]; }
    __aicore__ __attribute__((always_inline)) inline void Split(LocalTensor<float> dst) {
        Muls(F(208),F(204),0.000244140625F,64);PipeBarrier<PIPE_V>();
        Cast(F(207).ReinterpretCast<int32_t>(),F(208),RoundMode::CAST_FLOOR,64);PipeBarrier<PIPE_V>();
        Cast(F(205),F(207).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(F(208),F(205),4096.0F,64);PipeBarrier<PIPE_V>();
        Sub(dst,F(204),F(208),64);PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ __attribute__((always_inline)) inline TreeHalfRational64(uint32_t address)
      : f_(TPosition::VECCALC,address,224*64) {}
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> Plane(uint32_t p) {return F(p);}

    __aicore__ __attribute__((always_inline)) inline void Init() {
        Duplicate(F(216),0.0F,64);Duplicate(F(217),1.0F,64);PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void Zero(uint32_t dst,uint32_t n) {
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k)Duplicate(F(dst+k),0.0F,64);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void CopyNumber(uint32_t dst,uint32_t src,uint32_t n) {
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k)Copy(F(dst+k),F(src+k),64,1,CopyRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void NegateIf(uint32_t dst,uint32_t n,uint32_t mask) {
        Duplicate(F(205),1.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Muls(F(204),F(dst+k),-1.0F,64);PipeBarrier<PIPE_V>();
            Adds(F(204),F(204),4095.0F,64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(209));
            Select(F(dst+k),M(mask),F(209),F(dst+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ __attribute__((always_inline)) inline void Encode(uint32_t dst,LocalTensor<float> x) {
        CompareScalar(M(10),x,0.0F,CMPMODE::LT,64);
        Abs(F(210),x,64);PipeBarrier<PIPE_V>();
        Muls(F(210),F(210),16777216.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(int32_t k=3;k>=0;--k) {
            const float inverse=k==3?1.4551915228366852e-11F:k==2?5.960464477539063e-8F:k==1?0.000244140625F:1.0F;
            const float power=k==3?68719476736.0F:k==2?16777216.0F:k==1?4096.0F:1.0F;
            Muls(F(211),F(210),inverse,64);PipeBarrier<PIPE_V>();
            Cast(F(207).ReinterpretCast<int32_t>(),F(211),RoundMode::CAST_FLOOR,64);PipeBarrier<PIPE_V>();
            Cast(F(dst+uint32_t(k)),F(207).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Muls(F(211),F(dst+uint32_t(k)),power,64);PipeBarrier<PIPE_V>();
            Sub(F(210),F(210),F(211),64);PipeBarrier<PIPE_V>();
        }
        NegateIf(dst,4,10);
    }
    __aicore__ __attribute__((always_inline)) inline void AddNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(F(205),0.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Add(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(dst+k));
        }
    }
    __aicore__ __attribute__((always_inline)) inline void SubNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(F(205),0.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Sub(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Sub(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            CompareScalar(M(0),F(204),0.0F,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            Select(F(205),M(0),F(217),F(216),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
            Muls(F(208),F(205),4096.0F,64);PipeBarrier<PIPE_V>();
            Add(F(dst+k),F(204),F(208),64);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ __attribute__((always_inline)) inline void Multiply(uint32_t dst,uint32_t a,uint32_t na,uint32_t b,uint32_t nb,uint32_t n) {
        Zero(dst,n);
        #pragma unroll 1
        for(uint32_t i=0;i<na;++i) {
            Duplicate(F(205),0.0F,64);PipeBarrier<PIPE_V>();
            #pragma unroll 1
            for(uint32_t j=0;j<nb && i+j<n;++j) {
                Mul(F(204),F(a+i),F(b+j),64);PipeBarrier<PIPE_V>();
                Add(F(204),F(204),F(dst+i+j),64);PipeBarrier<PIPE_V>();
                Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
                Split(F(dst+i+j));
            }
            if(i+nb<n) {Copy(F(dst+i+nb),F(205),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
        }
    }
    // Return masks a<b and a==b; callers retain any signed predicates elsewhere.
    __aicore__ __attribute__((always_inline)) inline void CompareNumber(uint32_t a,uint32_t b,uint32_t n,uint32_t less,uint32_t equal) {
        Duplicate(M(less).ReinterpretCast<uint16_t>(),uint16_t(0),4);
        Duplicate(M(equal).ReinterpretCast<uint16_t>(),uint16_t(65535),4);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(int32_t k=int32_t(n)-1;k>=0;--k) {
            Compare(M(0),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::LT,64);
            Compare(M(1),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(M(0).ReinterpretCast<uint16_t>(),M(0).ReinterpretCast<uint16_t>(),M(equal).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(less).ReinterpretCast<uint16_t>(),M(less).ReinterpretCast<uint16_t>(),M(0).ReinterpretCast<uint16_t>(),4);
            And(M(equal).ReinterpretCast<uint16_t>(),M(equal).ReinterpretCast<uint16_t>(),M(1).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ __attribute__((always_inline)) inline void SelectNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n,uint32_t mask) {
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k)Select(F(dst+k),M(mask),F(a+k),F(b+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void MakeScores(uint32_t row,uint32_t pointAddress,uint32_t pointPitch,uint32_t sfAddress,uint32_t soAddress,uint32_t segments) {
        const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
        const LocalTensor<float> sf(TPosition::VECCALC,sfAddress,384);
        const LocalTensor<uint32_t> so(TPosition::VECCALC,soAddress,64);
        Zero(100,32); // dot, |P-A|^2, |C-A|^2, |P-C|^2
        #pragma unroll 1
        for(uint32_t axis=0;axis<3;++axis) {
            Gather(F(218),sf,so,axis*4,64);Gather(F(219),sf,so,(axis+3)*4,64);PipeBarrier<PIPE_V>();
            Encode(132,points[axis*pointPitch+row*64]);Encode(136,F(218));Encode(140,F(219));
            SubNumber(144,132,136,4);SubNumber(148,140,136,4);SubNumber(152,132,140,4);
            CompareScalar(M(8),F(147),2048.0F,CMPMODE::GE,64);
            CompareScalar(M(9),F(151),2048.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
            And(M(0).ReinterpretCast<uint16_t>(),M(8).ReinterpretCast<uint16_t>(),M(9).ReinterpretCast<uint16_t>(),4);
            Or(M(11).ReinterpretCast<uint16_t>(),M(8).ReinterpretCast<uint16_t>(),M(9).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Not(M(0).ReinterpretCast<uint16_t>(),M(0).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(M(11).ReinterpretCast<uint16_t>(),M(11).ReinterpretCast<uint16_t>(),M(0).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            NegateIf(144,4,8);NegateIf(148,4,9);
            CompareScalar(M(8),F(155),2048.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();NegateIf(152,4,8);
            Multiply(156,144,4,144,4,8);AddNumber(108,108,156,8);
            Multiply(156,148,4,148,4,8);AddNumber(116,116,156,8);
            Multiply(156,152,4,152,4,8);AddNumber(124,124,156,8);
            Multiply(156,144,4,148,4,8);NegateIf(156,8,11);AddNumber(100,100,156,8);
        }
        Multiply(172,108,8,116,8,16);Multiply(188,100,8,100,8,16);
        SubNumber(0,172,188,16);CopyNumber(16,116,8);
        CompareNumber(100,116,8,6,7);
        Not(M(12).ReinterpretCast<uint16_t>(),M(6).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>(); // dot>=vv
        CompareScalar(M(13),F(107),2048.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>(); // dot<0
        Zero(156,16);CompareNumber(100,156,8,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).ReinterpretCast<uint16_t>(),M(13).ReinterpretCast<uint16_t>(),M(7).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CompareNumber(116,156,8,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).ReinterpretCast<uint16_t>(),M(13).ReinterpretCast<uint16_t>(),M(7).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CopyNumber(156,124,8);SelectNumber(0,156,0,16,12);
        Zero(156,16);CopyNumber(156,108,8);SelectNumber(0,156,0,16,13);
        Or(M(12).ReinterpretCast<uint16_t>(),M(12).ReinterpretCast<uint16_t>(),M(13).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Zero(156,8);Duplicate(F(156),1.0F,64);PipeBarrier<PIPE_V>();SelectNumber(16,156,16,8,12);
        Cast(F(99),so.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(F(99),F(99),1.0F/24.0F,64);PipeBarrier<PIPE_V>();
        // The fixed offset vector consists exactly of 0,24,...,1512.
        Cast(F(98).ReinterpretCast<int32_t>(),F(99),RoundMode::CAST_RINT,64);PipeBarrier<PIPE_V>();
        Cast(F(99),F(98).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(M(12),F(99),float(int32_t(segments)),CMPMODE::GE,64);PipeBarrier<PIPE_V>();
        Zero(156,16);Duplicate(F(156),1.0F,64);PipeBarrier<PIPE_V>();SelectNumber(0,156,0,16,12);
        Zero(156,8);SelectNumber(16,156,16,8,12);CopyNumber(96,99,1);
    }
    __aicore__ __attribute__((always_inline)) inline void Tournament() {
        #pragma unroll 1
        for(uint32_t stage=0;stage<6;++stage) {
            Cast(F(98).ReinterpretCast<int32_t>(),F(99),RoundMode::CAST_RINT,64);
            Duplicate(F(214).ReinterpretCast<int32_t>(),int32_t(1U<<stage),64);PipeBarrier<PIPE_V>();
            And(F(213).ReinterpretCast<uint16_t>(),F(98).ReinterpretCast<uint16_t>(),F(214).ReinterpretCast<uint16_t>(),128);
            Or(F(98).ReinterpretCast<uint16_t>(),F(98).ReinterpretCast<uint16_t>(),F(214).ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Not(F(213).ReinterpretCast<uint16_t>(),F(213).ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            And(F(98).ReinterpretCast<uint16_t>(),F(98).ReinterpretCast<uint16_t>(),F(213).ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(F(98).ReinterpretCast<int32_t>(),F(98).ReinterpretCast<int32_t>(),int32_t(4),64);PipeBarrier<PIPE_V>();
            #pragma unroll 1
            for(uint32_t k=0;k<24;++k)Gather(F(24+k),F(k),F(98).ReinterpretCast<uint32_t>(),0U,64);
            Gather(F(97),F(96),F(98).ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            Multiply(48,24,16,16,8,24);Multiply(72,0,16,40,8,24);
            CompareNumber(48,72,24,6,7);
            Compare(M(0),F(97),F(96),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            And(M(0).ReinterpretCast<uint16_t>(),M(0).ReinterpretCast<uint16_t>(),M(7).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(6).ReinterpretCast<uint16_t>(),M(6).ReinterpretCast<uint16_t>(),M(0).ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            SelectNumber(0,24,0,24,6);SelectNumber(96,97,96,1,6);
        }
    }
};

__aicore__ __attribute__((always_inline)) inline uint32_t TreeHalfPendingRows(
    uint32_t scoreAddress,uint32_t storageAddress,uint32_t validRows,uint32_t limitAddress,
    uint32_t recordsAddress,uint32_t offsetsAddress) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,32*64);
    const LocalTensor<float> storage(TPosition::VECCALC,storageAddress,8*2080);
    const LocalTensor<float> limit(TPosition::VECCALC,limitAddress,64);
    const LocalTensor<float> records(TPosition::VECCALC,recordsAddress,32*8+64);
    const LocalTensor<uint32_t> offsets(TPosition::VECCALC,offsetsAddress,32+64);
    auto masks=storage[220*64].ReinterpretCast<uint8_t>();
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    Gather(storage[2048],records,offsets,0U,32);PipeBarrier<PIPE_V>();
    Muls(storage[2048],storage[2048],1.00006103515625F,32);
    Muls(storage[2400],limit,4.0F,32);PipeBarrier<PIPE_V>();
    Add(storage[2048],storage[2048],storage[2400],32);PipeBarrier<PIPE_V>();
    Brcb(storage[2080],storage[2048],4,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    Compare(masks,scores,storage[2080],CMPMODE::LE,uint64_t(64),uint8_t(32),
        BinaryRepeatParams(1,1,0,1,8,1));PipeBarrier<PIPE_V>();
    auto bits=masks.ReinterpretCast<int32_t>();
    auto minus=storage.ReinterpretCast<int32_t>();
    Adds(minus,bits,int32_t(-1),64);PipeBarrier<PIPE_V>();
    And(storage[64].ReinterpretCast<uint16_t>(),bits.ReinterpretCast<uint16_t>(),minus.ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
    auto empty=storage[128].ReinterpretCast<uint8_t>();
    auto one=storage[136].ReinterpretCast<uint8_t>();
    auto signOnly=storage[144].ReinterpretCast<uint8_t>();
    CompareScalar(empty,bits,int32_t(0),CMPMODE::EQ,64);
    CompareScalar(one,storage[64].ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);
    // INT_MIN is a singleton. Exclude it independently of integer overflow mode.
    CompareScalar(signOnly,bits,int32_t(-2147483647-1),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint64_t any=~empty.ReinterpretCast<uint64_t>().GetValue(0);
    const uint64_t multiple=~one.ReinterpretCast<uint64_t>().GetValue(0)&~signOnly.ReinterpretCast<uint64_t>().GetValue(0);
    uint64_t packed=(multiple|(multiple>>1)|(any&(any>>1)))&uint64_t(0x5555555555555555ULL);
    packed=(packed|(packed>>1))&uint64_t(0x3333333333333333ULL);
    packed=(packed|(packed>>2))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
    packed=(packed|(packed>>4))&uint64_t(0x00ff00ff00ff00ffULL);
    packed=(packed|(packed>>8))&uint64_t(0x0000ffff0000ffffULL);
    packed=(packed|(packed>>16))&uint64_t(0x00000000ffffffffULL);
    uint32_t pending=uint32_t(packed);
    pending&=validRows==32?~uint32_t(0):(uint32_t(1)<<validRows)-1U;
    return pending;
}

__aicore__ __attribute__((noinline)) inline bool TreeHalfRefineNearTies(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t sfAddress,uint32_t soAddress,uint32_t storageAddress,
    uint32_t indexAddress,uint32_t validRows,uint32_t segments,uint32_t pending) {
    const LocalTensor<int32_t> indices(TPosition::VECCALC,indexAddress,32);
    if(!pending)return false;
    TreeHalfRational64 exact(storageAddress);exact.Init();
    while(pending) {
        const uint32_t row=uint32_t(__builtin_ctz(pending));pending&=pending-1U;
        exact.MakeScores(row,pointAddress,pointPitch,sfAddress,soAddress,segments);exact.Tournament();
        Cast(exact.Plane(214).ReinterpretCast<int32_t>(),exact.Plane(96),RoundMode::CAST_RINT,64);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(1U<<(row&7U)));
        Copy<int32_t,false>(indices[(row/8)*8],exact.Plane(214).ReinterpretCast<int32_t>(),uint64_t(8),1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    }
    return true;
}
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T>
__aicore__ __attribute__((noinline,aligned(128))) void TreeGeneralExactFallback(
    GM_ADDR pointAddress,GM_ADDR segmentAddress,GM_ADDR distanceAddress,GM_ADDR indexAddress,uint32_t segments);
#endif

template <class T, uint32_t MODE, bool SINGLE_TILE=false,uint32_t FIXED_ROWS=0,uint32_t FIXED_STRIDE=0>
class KernelTreePairs {
    static constexpr bool NATIVE_HALF_ENDPOINT_ORDER=SINGLE_TILE && sizeof(T)==2 &&
        ((MODE==4 && (FIXED_ROWS==0 || FIXED_ROWS==16 || FIXED_ROWS==32)) ||
         (MODE==1 && FIXED_STRIDE==32 && FIXED_ROWS>0 && FIXED_ROWS<=8));
    static constexpr bool TILE_COEFFICIENTS=MODE==1 && SINGLE_TILE;
    static constexpr bool PACKED_HALF_ROW_OUTPUT=sizeof(T)==2 && TILE_COEFFICIENTS && FIXED_STRIDE==32 && FIXED_ROWS>0 && FIXED_ROWS<=8;
    static constexpr bool DIRECT_HALF_ROW_POINTS=sizeof(T)==2 && MODE==4 && SINGLE_TILE && FIXED_ROWS==32 && FIXED_STRIDE==64;
    static constexpr bool DIRECT_SINGLE_QUERY_POINTS=sizeof(T)==2 && MODE==8 && FIXED_ROWS==1 && FIXED_STRIDE==1024;
    static constexpr bool SHORT_FP32_METADATA=sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool MAPPED_STARTS=(MODE==1 && SINGLE_TILE && FIXED_ROWS==4 && FIXED_STRIDE==32 && SHORT_FP32_METADATA) ||
        (MODE==4 && SINGLE_TILE && sizeof(T)==2
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#endif
        );
    static constexpr bool OWNED_AXES8=MODE==2 && SINGLE_TILE && SHORT_FP32_METADATA;
    static constexpr bool NATIVE_SINGLE_SEGMENT_GEOMETRY=MODE==3 && sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool BATCH_COORDINATE_AXES=(MAPPED_STARTS && SHORT_FP32_METADATA) || OWNED_AXES8;
    static constexpr bool PACKED_HALF_GEOMETRY_AXES=MODE==4 && SINGLE_TILE && sizeof(T)==2
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#endif
        ;
public:
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                               const TreeLineMinDistanceTilingData &t) {
        t_ = t;
        exactDifferences_=false;needsExact_=false;
        stride_ = FIXED_STRIDE ? FIXED_STRIDE : MODE == 2 ? 8 : MODE == 4 ? 64 : t.paddedSegments;
        rows_ = FIXED_ROWS ? FIXED_ROWS : MODE == 2 ? 8 : MODE == 4 ? 32 : MODE == 8 ? 1 : t.pointTile;
        count_ = stride_ * rows_;
        allocCount_ = count_ < 64 ? 64 : count_;
        count_ = allocCount_;
        group_ = (MODE == 4 || MODE == 8) ? 1 : stride_ > 64 ? 32 : stride_ == 64 ? 16 : 1;
        allocCount_ += 32;
        pg_.SetGlobalBuffer((__gm__ T *)p);
        sg_.SetGlobalBuffer((__gm__ T *)s);
        dg_.SetGlobalBuffer((__gm__ T *)d);
        ig_.SetGlobalBuffer((__gm__ int32_t *)i);
        uint32_t address = 0;
        sf_ = Allocate<float>(address, stride_ * 6);
        if constexpr(sizeof(T)==4)rawS_=sf_.template ReinterpretCast<T>();
        else rawS_=Allocate<T>(address,stride_*6);
        columns_ = Allocate<float>(address, (MODE==3?104:MODE==4?96:allocCount_) * 13);
        pf_ = Allocate<float>(address, rows_ * group_ * 3 + 64);
        if constexpr(sizeof(T)==4)rawP_=pf_.template ReinterpretCast<T>();
        else rawP_=Allocate<T>(address,rows_*group_*3+64);
        if constexpr (MODE != 3) pointSlots_ = Allocate<float>(address, (rows_ * group_ * 3 + 64) * 8);
        xyz_ = Allocate<float>(address, allocCount_ * 3);
        work_ = Allocate<float>(address, allocCount_ * 9);
        po_ = Allocate<int32_t>(address, allocCount_);
        so_ = Allocate<int32_t>(address, allocCount_);
        mask_ = Allocate<uint8_t>(address, (allocCount_ + 255) / 256 * 32);
        valid_ = Allocate<uint8_t>(address, (allocCount_ + 255) / 256 * 32);
        records_ = Allocate<float>(address, rows_ * 8 + 64);
        reduceWork_ = Allocate<float>(address, allocCount_);
        outputOffsets_ = Allocate<int32_t>(address, rows_ + 64);
        out_ = Allocate<float>(address, rows_ + 64);
        stageIdx_ = Allocate<int32_t>(address, ((rows_ + 7) / 8 * 8) * group_);
        outIdx_ = stageIdx_;
        stageOut_ = Allocate<T>(address, ((rows_ * sizeof(T) + 31) / 32 * 32 / sizeof(T)) * group_);
        outT_ = stageOut_;
        if constexpr(MODE!=3 || sizeof(T)==4) {
            reduceOffsets_ = Allocate<int32_t>(address, rows_ * 16 + 64);
            reduceBases_ = Allocate<int32_t>(address, rows_ + 64);
        }
        if constexpr(MODE!=3 || sizeof(T)==2) {
            zeroBound_=Allocate<float>(address,96);
            zeroExact_=Allocate<float>(address,45*64);
        }
        if constexpr(MODE==3 || (MODE==4 && !SINGLE_TILE)) {
        if constexpr (MODE == 3) {
            // Point gather offsets are prepared while the first input DMA is in flight.
        } else if constexpr(MODE == 4) {
            MakeIndices(so_,int32_t(0),64);PipeBarrier<PIPE_V>();
            if constexpr((!SINGLE_TILE || sizeof(T)==2) && !PACKED_HALF_ROW_OUTPUT)Muls(outputOffsets_,so_,int32_t(8),rows_);
            Muls(so_,so_,int32_t(24),64);
            const uint32_t high=t_.segments==64?~uint32_t(0):(uint32_t(1)<<(t_.segments-32))-1;
            Duplicate(valid_.template ReinterpretCast<uint32_t>(),~uint32_t(0),count_/32);PipeBarrier<PIPE_V>();
            if(high!=~uint32_t(0)) {
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL &
                    (count_/32==64?~uint64_t(0):((uint64_t(1)<<(count_/32))-1))));
                Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),high,uint64_t(64),1,1,8);
                SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
            }
        } else if constexpr(MODE == 2) {
            MakeIndices(outputOffsets_,int32_t(0),8);PipeBarrier<PIPE_V>();
            Muls(so_,outputOffsets_,int32_t(24),8);
            if constexpr(!SINGLE_TILE || sizeof(T)==2)Muls(outputOffsets_,outputOffsets_,int32_t(8),8);
            PipeBarrier<PIPE_V>();
            Copy(so_[8],so_,8,7,CopyRepeatParams(1,1,1,0));
            const uint32_t bits=((1U<<t_.segments)-1U)*0x01010101U;
            Duplicate(valid_.template ReinterpretCast<uint32_t>(),bits,2);
        } else if constexpr(MODE==8) {
            MakeIndices(po_,int32_t(0),count_);PipeBarrier<PIPE_V>();
            Muls(so_,po_,int32_t(24),count_);
            Cast(W(8),po_,RoundMode::CAST_NONE,count_);
            Duplicate(outputOffsets_,int32_t(0),8);
            Muls(reduceOffsets_,po_,int32_t(8),32);
            Duplicate(reduceBases_,int32_t(4),8);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(8),float(int32_t(t_.segments)),CMPMODE::LT,count_);
        } else {
        auto index = W(6).template ReinterpretCast<int32_t>();
        auto tmp = W(7).template ReinterpretCast<int32_t>();
        if constexpr(sizeof(T)==4) {
        if(stride_<=64) {
            MakeIndices(so_,int32_t(0),stride_);PipeBarrier<PIPE_V>();
            if constexpr((!SINGLE_TILE || sizeof(T)==2) && !PACKED_HALF_ROW_OUTPUT)Muls(outputOffsets_,so_,int32_t(8),rows_);
            if(count_>stride_)Copy(so_[stride_],so_,stride_,count_/stride_-1,CopyRepeatParams(1,1,stride_/8,0));
            PipeBarrier<PIPE_V>();
            Muls(so_,so_,int32_t(24),count_);
            if(stride_<=32) {
                uint32_t bits=t_.segments==32?~uint32_t(0):(uint32_t(1)<<t_.segments)-1;
                if(stride_==8)bits*=0x01010101U;
                else if(stride_==16)bits|=bits<<16;
                Duplicate(valid_.template ReinterpretCast<uint32_t>(),bits,count_/32);
            } else {
                const uint32_t high=t_.segments==64?~uint32_t(0):(uint32_t(1)<<(t_.segments-32))-1;
                Duplicate(valid_.template ReinterpretCast<uint32_t>(),~uint32_t(0),count_/32);PipeBarrier<PIPE_V>();
                if(high!=~uint32_t(0)) {
                    SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL &
                        (count_/32==64?~uint64_t(0):((uint64_t(1)<<(count_/32))-1))));
                    Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),high,uint64_t(64),1,1,8);
                    SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
                }
            }
        } else {
            MakeIndices(index,int32_t(0),count_);PipeBarrier<PIPE_V>();
            ShiftRight(po_.template ReinterpretCast<uint32_t>(),index.template ReinterpretCast<uint32_t>(),t_.segmentLog,count_);PipeBarrier<PIPE_V>();
            Muls(tmp,po_,static_cast<int32_t>(stride_),count_);PipeBarrier<PIPE_V>();
            Sub(so_,index,tmp,count_);PipeBarrier<PIPE_V>();
            Cast(W(8),so_,RoundMode::CAST_NONE,count_);
            Muls(so_,so_,int32_t(24),count_);Muls(po_,po_,int32_t(12),count_);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(8),static_cast<float>(static_cast<int32_t>(t_.segments)),CMPMODE::LT,count_);
            Muls(outputOffsets_,index,int32_t(8),rows_);
        }
        } else {
        MakeIndices(index, int32_t(0), count_);
        PipeBarrier<PIPE_V>();
        ShiftRight(po_.template ReinterpretCast<uint32_t>(), index.template ReinterpretCast<uint32_t>(),
                   t_.segmentLog, count_);
        PipeBarrier<PIPE_V>();
        Muls(tmp, po_, static_cast<int32_t>(stride_), count_);
        PipeBarrier<PIPE_V>();
        Sub(so_, index, tmp, count_);
        PipeBarrier<PIPE_V>();
        Cast(W(8), so_, RoundMode::CAST_NONE, count_);
        Muls(so_, so_, int32_t(24), count_);
        Muls(po_, po_, int32_t(12), count_);
        PipeBarrier<PIPE_V>();
        if constexpr(MODE==8) {
            Copy(po_,index,64,count_/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        }
        CompareScalar(valid_, W(8), static_cast<float>(static_cast<int32_t>(t_.segments)), CMPMODE::LT, count_);
        Muls(outputOffsets_, index, int32_t(8), rows_);
        }
        if (stride_ > 64) {
            const uint32_t parts = stride_ / 64;
            const uint32_t partStride = parts < 8 ? 8 : parts;
            if(parts>=8) {
                Muls(reduceOffsets_,index,int32_t(8),rows_*partStride);
            } else {
                Duplicate(reduceOffsets_, int32_t(0), rows_ * partStride);
                PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < rows_; ++row)
                    MakeIndices(reduceOffsets_[row * partStride], static_cast<int32_t>(row * parts), parts);
                PipeBarrier<PIPE_V>();
                Muls(reduceOffsets_, reduceOffsets_, int32_t(8), rows_ * partStride);
            }
            Muls(reduceBases_, index, static_cast<int32_t>(parts * 8), rows_);
            PipeBarrier<PIPE_V>();
            Adds(reduceBases_, reduceBases_, int32_t(4), rows_);
        }
        }
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Process() {
        if constexpr(SINGLE_TILE) {
            const uint32_t tiles=(t_.points+rows_-1)/rows_;
            const uint32_t batch=GetBlockIdx()/tiles;
            const uint32_t start=(GetBlockIdx()-batch*tiles)*rows_;
            const uint32_t validRows=t_.points-start<rows_?t_.points-start:rows_;
            validComputeRows_=validRows;needsExact_=false;
            LoadSegments(batch,start,validRows,true);
            if constexpr(!DIRECT_HALF_ROW_POINTS) {
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)
                    Copy(P(axis),pointSlots_[axis*8],stride_,rows_,CopyRepeatParams(1,0,stride_/8,3));
                PipeBarrier<PIPE_V>();
            }
            if constexpr((TILE_COEFFICIENTS || MODE==4) && sizeof(T)==2) {
                if(exactDifferences_)Compute<true>();else Compute<false>();
            } else if constexpr(SHORT_FP32_METADATA && SINGLE_TILE && MODE==2) {
                Compute<true>();
                if(needsExact_) {
                    needsExact_=false;
                    Compute<false>();
                }
            } else Compute();
            DataCopyPad(dg_[batch*t_.points+start],outT_,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
            DataCopyPad(ig_[batch*t_.points+start],outIdx_,DataCopyExtParams{1,validRows*4,0,0,0});
            if(needsExact_) {
                // The exact repair overwrites the same GM interval. Ordinary
                // single-tile drains have no subsequent UB writer or GM overwrite.
                SetFlag<HardEvent::MTE3_S>(EVENT_ID3);WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
                RepairRange(batch,start,validRows);
            }
        } else if constexpr(MODE==8) {
            const uint32_t batch=GetBlockIdx();
            validComputeRows_=1;needsExact_=false;
            LoadSegments(batch,0,1,true);
            if constexpr(!DIRECT_SINGLE_QUERY_POINTS) {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                if(stride_<=64)Copy(P(axis),pointSlots_[axis*8],stride_,1,CopyRepeatParams(1,0,8,0));
                else Copy(P(axis),pointSlots_[axis*8],64,stride_/64,CopyRepeatParams(1,0,8,0));
            }
            }
            PipeBarrier<PIPE_V>();SelectCandidates();
            if constexpr(sizeof(T)==2
#ifdef DTYPE_TREE_POINTS
                && sizeof(DTYPE_TREE_POINTS)==2
#endif
            ) {
                if(count_==64)Compute<false,true>();
                else Compute();
            } else Compute();
            DataCopyPad(dg_[batch],outT_,DataCopyExtParams{1,uint32_t(sizeof(T)),0,0,0});
            DataCopyPad(ig_[batch],outIdx_,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
            if(needsExact_)RepairRange(batch,0,1);
        } else {
        const uint32_t tiles = (t_.points + rows_ - 1) / rows_;
        const uint32_t total = t_.batches * tiles;
        const uint32_t begin = total * GetBlockIdx() / t_.blocks;
        const uint32_t end = total * (GetBlockIdx() + 1) / t_.blocks;
        uint32_t cached = 0xffffffffU;
        for (uint32_t task = begin; task < end;) {
            const uint32_t batch = task / tiles;
            const uint32_t start = task % tiles * rows_;
            uint32_t groups = end - task < group_ ? end - task : group_;
            if (groups > tiles - task % tiles) groups = tiles - task % tiles;
            const uint32_t validRows = t_.points - start < rows_ * groups ? t_.points - start : rows_ * groups;
            needsExact_=false;
            const bool changed = batch != cached;
            if(changed) { LoadSegments(batch,start,validRows,cached==0xffffffffU);cached=batch; }
            if(!changed)LoadPoints(batch,start,validRows);
            const uint32_t distanceStride = (rows_ * sizeof(T) + 31) / 32 * 32 / sizeof(T);
            const uint32_t indexStride = (rows_ + 7) / 8 * 8;
            for (uint32_t slot = 0; slot < groups; ++slot) {
                const uint32_t offset=slot*rows_;
                validComputeRows_=validRows-offset<rows_?validRows-offset:rows_;
                outT_ = stageOut_[slot * distanceStride];
                outIdx_ = stageIdx_[slot * indexStride];
                #pragma unroll
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    if constexpr(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) {
                        uint64_t selected=0;
                        GatherMask(P(axis),pf_,po_[axis*(3U*FIXED_ROWS/32U)].template ReinterpretCast<uint32_t>(),
                            true,3U*count_,GatherMaskParams(1,1,8,1),selected);
                    } else if constexpr (MODE == 3) {
                        Gather(P(axis), pf_, po_.template ReinterpretCast<uint32_t>(), axis * 4, count_);
                    } else if (stride_ <= 64) {
                        Copy(P(axis), pointSlots_[(slot * rows_ * 3 + axis) * 8],
                             stride_, rows_, CopyRepeatParams(1, 0, stride_ / 8, 3));
                    } else {
                        for (uint32_t row = 0; row < rows_; ++row)
                            Copy(P(axis)[row * stride_], pointSlots_[((slot * rows_ + row) * 3 + axis) * 8],
                                 64, stride_ / 64, CopyRepeatParams(1, 0, 8, 0));
                    }
                }
                PipeBarrier<PIPE_V>();
                if constexpr(MODE==4 && sizeof(T)==2) {
                    if(exactDifferences_)Compute<true>();else Compute();
                } else Compute();
            }
            const uint32_t base = batch * t_.points + start;
            const uint32_t full = validRows / rows_;
            const uint32_t tail = validRows % rows_;
            if (full) {
                DataCopyPad(dg_[base], stageOut_, DataCopyExtParams{static_cast<uint16_t>(full), rows_ * uint32_t(sizeof(T)), 0, 0, 0});
                DataCopyPad(ig_[base], stageIdx_, DataCopyExtParams{static_cast<uint16_t>(full), rows_ * 4, 0, 0, 0});
            }
            if (tail) {
                DataCopyPad(dg_[base + full * rows_], stageOut_[full * distanceStride], DataCopyExtParams{1, tail * uint32_t(sizeof(T)), 0, 0, 0});
                DataCopyPad(ig_[base + full * rows_], stageIdx_[full * indexStride], DataCopyExtParams{1, tail * 4, 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
            if(needsExact_) {
                RepairRange(batch,start,validRows);
                if(task+groups<end) {
                    const TreeLineMinDistanceTilingData t=t_;
                    Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
                         (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
                    cached=0xffffffffU;
                }
            }
            task += groups;
        }
        }
    }
private:
    __aicore__ inline void RepairRange(uint32_t batch,uint32_t start,uint32_t rows) {
#if defined(DTYPE_TREE_POINTS)
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=0;row<rows;++row) {
            const uint32_t pos=batch*t_.points+start+row;
            TreeGeneralExactFallback<T>((GM_ADDR)pg_[pos*3].GetPhyAddr(),
                (GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),
                (GM_ADDR)dg_[pos].GetPhyAddr(),(GM_ADDR)ig_[pos].GetPhyAddr(),t_.segments);
        }
#endif
    }
    template<bool FAST_FLOAT_DIFFERENCES=false>
    __aicore__ inline bool PackedRowsAreUnique(LocalTensor<uint8_t> possible) {
        if constexpr(PACKED_HALF_ROW_OUTPUT) {
            // Only the first eight predicate words represent this resident arena.
            // Keep one normal vector mask across arithmetic and comparisons.
            auto safe=zeroExact_[128].template ReinterpretCast<uint8_t>();
            auto combined=possible.template ReinterpretCast<int32_t>();
            const UnaryRepeatParams unary{1,1,8,8};
            const BinaryRepeatParams binary{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Adds<int32_t,false>(W(6).template ReinterpretCast<int32_t>(),combined,int32_t(-1),uint64_t(64),1,unary);
            PipeBarrier<PIPE_V>();
            And<uint16_t,false>(W(6).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),combined.template ReinterpretCast<uint16_t>(),uint64_t(64),1,binary);
            PipeBarrier<PIPE_V>();
            CompareScalar<int32_t,uint8_t,false>(safe,W(6).template ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,uint64_t(64),1,unary);
            Compare<float,uint8_t,false>(mask_,out_,zeroBound_[32],CMPMODE::GT,uint64_t(64),1,binary);
            PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(4));
            And<uint16_t,false>(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),uint64_t(4),1,binary);
            PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t live=(uint64_t(1)<<validComputeRows_)-1U;
            return (safe.template ReinterpretCast<uint64_t>().GetValue(0)&live)==live;
        } else {
        auto source=possible.template ReinterpretCast<int32_t>();
        auto combined=source;
        if(stride_==64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            GatherMask(W(6),possible.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,1,8,0));
            GatherMask(W(7),possible.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Or(W(8).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),W(7).template ReinterpretCast<uint16_t>(),128);
            And(W(7).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),W(7).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            combined=W(8).template ReinterpretCast<int32_t>();
        }
        Adds(W(6).template ReinterpretCast<int32_t>(),combined,int32_t(-1),64);PipeBarrier<PIPE_V>();
        And(W(6).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),combined.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        if(stride_==64) {
            Or(W(6).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),W(7).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        }
        auto safe=zeroExact_[128].template ReinterpretCast<uint8_t>();
        CompareScalar(safe,W(6).template ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);
        if constexpr(sizeof(T)==4) {
            CompareScalar(mask_,combined,int32_t(0),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            Not(mask_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        if constexpr(FAST_FLOAT_DIFFERENCES) {
            Muls(W(7),zeroBound_[32],134217728.0F,64);PipeBarrier<PIPE_V>();
            Compare(mask_,out_,W(7),CMPMODE::GT,64);
        } else Compare(mask_,out_,zeroBound_[32],CMPMODE::GT,64);
        PipeBarrier<PIPE_V>();
        And(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        // Finite binary16 endpoints satisfy norm <= 3*(2*65504)^2,
        // hence norm*2^-39 < 1. The wider float domain still needs its check.
        if constexpr(sizeof(T)==4) {
            CompareScalar(mask_,zeroBound_[32],2097152.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            And(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t live=validComputeRows_==64?~uint64_t(0):(uint64_t(1)<<validComputeRows_)-1;
        return (safe.template ReinterpretCast<uint64_t>().GetValue(0)&live)==live;
            }
}
    template<bool FAST_FLOAT_DIFFERENCES=false>
    __aicore__ inline void CheckNumericalRisk() {
#if defined(DTYPE_TREE_POINTS)
        if constexpr(MODE==3 && sizeof(T)==4) {
            CompareScalar(mask_,W(0),1152921504606846976.0F,CMPMODE::LE,count_);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            auto bits=mask_.template ReinterpretCast<uint64_t>();
            for(uint32_t first=0;first<validComputeRows_;first+=64) {
                const uint32_t count=validComputeRows_-first<64?validComputeRows_-first:64;
                const uint64_t active=count==64?~uint64_t(0):(uint64_t(1)<<count)-1;
                if((bits.GetValue(first/64)&active)!=active)needsExact_=true;
            }
            const float norm=C(9).GetValue(0);
            if(!(norm>=0.0F && norm<=1152921504606846976.0F))needsExact_=true;
        } else if constexpr(MODE!=3) {
            if constexpr(MODE==8) {
                if(selectedCount_==1) {
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const float score=out_.GetValue(0),norm=C(9).GetValue(0);
                    if(!(score>=0.0F && score<=1152921504606846976.0F &&
                         norm>=0.0F && norm<=1152921504606846976.0F))needsExact_=true;
                    return;
                }
            }
            auto possible=zeroExact_.template ReinterpretCast<uint8_t>();
            if constexpr(MODE!=8) {
                // Enlarging both one-sided score intervals gives a conservative
                // upper threshold: the old candidate set is a subset of this one.
                Muls(W(7),out_,1.000244140625F,64);
                Muls(W(6),zeroBound_[32],262272.0F,64);PipeBarrier<PIPE_V>();
                Add(W(7),W(7),W(6),64);PipeBarrier<PIPE_V>();
                if constexpr(MODE==2) {
                    Brcb(W(8),W(7),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                } else if(stride_<=64) {
                    Brcb(W(6),W(7),(rows_+7U)/8U,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                    Copy(W(8),W(6),stride_,rows_,CopyRepeatParams(1,0,stride_/8U,1));PipeBarrier<PIPE_V>();
                } else {
                    MakeIndices(W(6).template ReinterpretCast<int32_t>(),0,count_);PipeBarrier<PIPE_V>();
                    uint32_t shift=0;for(uint32_t n=stride_;n>1;n>>=1)++shift;
                    ShiftRight(W(6).template ReinterpretCast<uint32_t>(),W(6).template ReinterpretCast<uint32_t>(),shift,count_);PipeBarrier<PIPE_V>();
                    Muls(W(6).template ReinterpretCast<int32_t>(),W(6).template ReinterpretCast<int32_t>(),int32_t(4),count_);PipeBarrier<PIPE_V>();
                    Gather(W(8),W(7),W(6).template ReinterpretCast<uint32_t>(),0U,count_);PipeBarrier<PIPE_V>();
                }
                Compare(possible,W(0),W(8),CMPMODE::LE,count_);PipeBarrier<PIPE_V>();
                if(stride_==32 || stride_==64) {
                    if(!PackedRowsAreUnique<FAST_FLOAT_DIFFERENCES>(possible))needsExact_=true;
                    return;
                }
                auto safe=zeroExact_[128].template ReinterpretCast<uint8_t>();
                if constexpr(FAST_FLOAT_DIFFERENCES) {
                    // Require squared distance > maximum segment norm / 4096.
                    // Near/cancellation-sensitive results rerun the unchanged
                    // compensated vector calculation before any GM output write.
                    Muls(W(8),zeroBound_[32],134217728.0F,64);PipeBarrier<PIPE_V>();
                    Compare(safe,out_,W(8),CMPMODE::GT,64);
                } else Compare(safe,out_,zeroBound_[32],CMPMODE::GT,64);
                if constexpr(sizeof(T)==4) {
                    CompareScalar(mask_,zeroBound_[32],2097152.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
                    And(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const uint64_t live=validComputeRows_==64?~uint64_t(0):(uint64_t(1)<<validComputeRows_)-1;
                if((safe.template ReinterpretCast<uint64_t>().GetValue(0)&live)!=live)needsExact_=true;
                auto bits=possible.template ReinterpretCast<uint64_t>();
                if constexpr(MODE==2) {
                    uint64_t counts=bits.GetValue(0);
                    counts-=((counts>>1)&uint64_t(0x5555555555555555ULL));
                    counts=(counts&uint64_t(0x3333333333333333ULL))+((counts>>2)&uint64_t(0x3333333333333333ULL));
                    counts=(counts+(counts>>4))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
                    const uint64_t active=validComputeRows_==8?~uint64_t(0):(uint64_t(1)<<(validComputeRows_*8))-1;
                    if(((counts^uint64_t(0x0101010101010101ULL))&active)!=0)needsExact_=true;
                } else {
                    for(uint32_t row=0;row<validComputeRows_;++row) {
                        uint32_t candidates=0;
                        if(stride_<=64) {
                            const uint32_t first=row*stride_;
                            const uint64_t active=stride_==64?~uint64_t(0):(uint64_t(1)<<stride_)-1;
                            const uint64_t selected=(bits.GetValue(first/64)>>(first%64))&active;
                            candidates=selected!=0 && (selected&(selected-1))==0?1:0;
                        } else for(uint32_t k=0;k<stride_/64;++k)candidates+=uint32_t(__builtin_popcountll(bits.GetValue(row*(stride_/64)+k)));
                        if(candidates!=1)needsExact_=true;
                    }
                }
                return;
            }

            if constexpr(MODE==2) {
                Brcb(W(8),out_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            } else if constexpr(MODE==8) {
                Brcb(W(6),out_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                Brcb(W(7),W(6),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                Copy(W(8),W(7),64,count_/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
            } else {
                MakeIndices(W(6).template ReinterpretCast<int32_t>(),0,count_);PipeBarrier<PIPE_V>();
                uint32_t shift=0;for(uint32_t s=stride_;s>1;s>>=1)++shift;
                ShiftRight(W(6).template ReinterpretCast<uint32_t>(),W(6).template ReinterpretCast<uint32_t>(),shift,count_);PipeBarrier<PIPE_V>();
                Muls(W(6).template ReinterpretCast<int32_t>(),W(6).template ReinterpretCast<int32_t>(),int32_t(4),count_);PipeBarrier<PIPE_V>();
                Gather(W(8),out_,W(6).template ReinterpretCast<uint32_t>(),0U,count_);PipeBarrier<PIPE_V>();
            }
            const uint8_t repeats=count_/64;
            const UnaryRepeatParams unary{1,1,8,8};
            const BinaryRepeatParams repeatBound{1,1,1,8,8,0};
            Muls<float,false>(W(7),W(0),0.999935150146484375F,uint64_t(64),repeats,unary);
            Muls<float,false>(W(8),W(8),1.000064849853515625F,uint64_t(64),repeats,unary);
            Muls(W(6),zeroBound_[32],131072.0F,64);PipeBarrier<PIPE_V>();
            Sub<float,false>(W(7),W(7),W(6),uint64_t(64),repeats,repeatBound);
            Add<float,false>(W(8),W(8),W(6),uint64_t(64),repeats,repeatBound);PipeBarrier<PIPE_V>();
            Compare(possible,W(7),W(8),CMPMODE::LE,count_);
            if constexpr(MODE!=8)Compare(mask_,out_,zeroBound_[32],CMPMODE::LE,64);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if constexpr(MODE!=8) {
                const uint64_t live=validComputeRows_==64?~uint64_t(0):(uint64_t(1)<<validComputeRows_)-1;
                if(mask_.template ReinterpretCast<uint64_t>().GetValue(0)&live)needsExact_=true;
            }
            auto bits=possible.template ReinterpretCast<uint64_t>();
            const float normBound=zeroBound_[32].GetValue(0);
            if(!(normBound>=0.0F && normBound<=2097152.0F))needsExact_=true;
            if constexpr(MODE==2) {
                uint64_t counts=bits.GetValue(0);
                counts-=((counts>>1)&uint64_t(0x5555555555555555ULL));
                counts=(counts&uint64_t(0x3333333333333333ULL))+((counts>>2)&uint64_t(0x3333333333333333ULL));
                counts=(counts+(counts>>4))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
                const uint64_t active=validComputeRows_==8?~uint64_t(0):
                    (uint64_t(1)<<(validComputeRows_*8))-1;
                if(((counts^uint64_t(0x0101010101010101ULL))&active)!=0)needsExact_=true;
            } else if constexpr(MODE==8) {
                uint64_t candidateCount=0;
                GatherMask(W(6),po_.template ReinterpretCast<float>(),possible.template ReinterpretCast<uint32_t>(),
                           true,count_,GatherMaskParams(1,1,8,1),candidateCount);
                PipeBarrier<PIPE_V>();
                if(candidateCount!=1)needsExact_=true;
            } else {
            for(uint32_t row=0;row<validComputeRows_;++row) {
                uint32_t candidates=0;
                if(stride_<=64) {
                    const uint32_t first=row*stride_;
                    const uint64_t active=stride_==64?~uint64_t(0):(uint64_t(1)<<stride_)-1;
                    const uint64_t selected=(bits.GetValue(first/64)>>(first%64))&active;
                    candidates=selected!=0 && (selected&(selected-1))==0?1:0;
                } else {
                    for(uint32_t k=0;k<stride_/64;++k)
                        candidates+=uint32_t(__builtin_popcountll(bits.GetValue(row*(stride_/64)+k)));
                }
                if(candidates!=1)needsExact_=true;
            }
            }
        }
#endif
    }
    __aicore__ inline void PrepareMetadata() {
        if constexpr (MODE == 3) {
            MakeIndices(po_,int32_t(0),count_);PipeBarrier<PIPE_V>();
            Muls(po_,po_,int32_t(12),count_);
        } else if constexpr(MODE == 4) {
            MakeIndices(so_,int32_t(0),64);PipeBarrier<PIPE_V>();
            if constexpr((!SINGLE_TILE || sizeof(T)==2) && !PACKED_HALF_ROW_OUTPUT)Muls(outputOffsets_,so_,int32_t(8),rows_);
            Muls(so_,so_,int32_t(24),64);
            const uint32_t high=t_.segments==64?~uint32_t(0):(uint32_t(1)<<(t_.segments-32))-1;
            Duplicate(valid_.template ReinterpretCast<uint32_t>(),~uint32_t(0),count_/32);PipeBarrier<PIPE_V>();
            if(high!=~uint32_t(0)) {
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL & (count_/32==64?~uint64_t(0):((uint64_t(1)<<(count_/32))-1))));
                Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),high,uint64_t(64),1,1,8);
                SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
            }
        } else if constexpr(MODE == 2) {
            MakeIndices(outputOffsets_,int32_t(0),8);PipeBarrier<PIPE_V>();
            Muls(so_,outputOffsets_,int32_t(24),8);
            if constexpr(!SINGLE_TILE || sizeof(T)==2)Muls(outputOffsets_,outputOffsets_,int32_t(8),8);
            PipeBarrier<PIPE_V>();
            Copy(so_[8],so_,8,7,CopyRepeatParams(1,1,1,0));
            const uint32_t bits=((1U<<t_.segments)-1U)*0x01010101U;
            Duplicate(valid_.template ReinterpretCast<uint32_t>(),bits,2);
        } else if constexpr(MODE==8) {
            MakeIndices(po_,int32_t(0),count_);PipeBarrier<PIPE_V>();
            Muls(so_,po_,int32_t(24),count_);
            Cast(W(8),po_,RoundMode::CAST_NONE,count_);
            Duplicate(outputOffsets_,int32_t(0),8);
            Muls(reduceOffsets_,po_,int32_t(8),32);
            Duplicate(reduceBases_,int32_t(4),8);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(8),float(int32_t(t_.segments)),CMPMODE::LT,count_);
        } else {
        auto index = W(6).template ReinterpretCast<int32_t>();
        auto tmp = W(7).template ReinterpretCast<int32_t>();
        if constexpr(sizeof(T)==4 || (TILE_COEFFICIENTS && sizeof(T)==2 && FIXED_STRIDE==32 && FIXED_ROWS<=8)) {
        if(stride_<=64) {
            MakeIndices(so_,int32_t(0),stride_);PipeBarrier<PIPE_V>();
            if constexpr((!SINGLE_TILE || sizeof(T)==2) && !PACKED_HALF_ROW_OUTPUT)Muls(outputOffsets_,so_,int32_t(8),rows_);
            if(count_>stride_)Copy(so_[stride_],so_,stride_,count_/stride_-1,CopyRepeatParams(1,1,stride_/8,0));
            PipeBarrier<PIPE_V>();
            Muls(so_,so_,int32_t(24),count_);
            if(stride_<=32) {
                uint32_t bits=t_.segments==32?~uint32_t(0):(uint32_t(1)<<t_.segments)-1;
                if(stride_==8)bits*=0x01010101U;
                else if(stride_==16)bits|=bits<<16;
                Duplicate(valid_.template ReinterpretCast<uint32_t>(),bits,count_/32);
            } else {
                const uint32_t high=t_.segments==64?~uint32_t(0):(uint32_t(1)<<(t_.segments-32))-1;
                Duplicate(valid_.template ReinterpretCast<uint32_t>(),~uint32_t(0),count_/32);PipeBarrier<PIPE_V>();
                if(high!=~uint32_t(0)) {
                    SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL &
                        (count_/32==64?~uint64_t(0):((uint64_t(1)<<(count_/32))-1))));
                    Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),high,uint64_t(64),1,1,8);
                    SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
                }
            }
        } else {
            MakeIndices(index,int32_t(0),count_);PipeBarrier<PIPE_V>();
            ShiftRight(po_.template ReinterpretCast<uint32_t>(),index.template ReinterpretCast<uint32_t>(),t_.segmentLog,count_);PipeBarrier<PIPE_V>();
            Muls(tmp,po_,static_cast<int32_t>(stride_),count_);PipeBarrier<PIPE_V>();
            Sub(so_,index,tmp,count_);PipeBarrier<PIPE_V>();
            Cast(W(8),so_,RoundMode::CAST_NONE,count_);
            Muls(so_,so_,int32_t(24),count_);Muls(po_,po_,int32_t(12),count_);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(8),static_cast<float>(static_cast<int32_t>(t_.segments)),CMPMODE::LT,count_);
            Muls(outputOffsets_,index,int32_t(8),rows_);
        }
        } else {
        MakeIndices(index, int32_t(0), count_);
        PipeBarrier<PIPE_V>();
        ShiftRight(po_.template ReinterpretCast<uint32_t>(), index.template ReinterpretCast<uint32_t>(),
                   t_.segmentLog, count_);
        PipeBarrier<PIPE_V>();
        Muls(tmp, po_, static_cast<int32_t>(stride_), count_);
        PipeBarrier<PIPE_V>();
        Sub(so_, index, tmp, count_);
        PipeBarrier<PIPE_V>();
        Cast(W(8), so_, RoundMode::CAST_NONE, count_);
        Muls(so_, so_, int32_t(24), count_);
        Muls(po_, po_, int32_t(12), count_);
        PipeBarrier<PIPE_V>();
        if constexpr(MODE==8) {
            Copy(po_,index,64,count_/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        }
        CompareScalar(valid_, W(8), static_cast<float>(static_cast<int32_t>(t_.segments)), CMPMODE::LT, count_);
        Muls(outputOffsets_, index, int32_t(8), rows_);
        }
        if (stride_ > 64) {
            const uint32_t parts = stride_ / 64;
            const uint32_t partStride = parts < 8 ? 8 : parts;
            if(parts>=8) {
                Muls(reduceOffsets_,index,int32_t(8),rows_*partStride);
            } else {
                Duplicate(reduceOffsets_, int32_t(0), rows_ * partStride);
                PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < rows_; ++row)
                    MakeIndices(reduceOffsets_[row * partStride], static_cast<int32_t>(row * parts), parts);
                PipeBarrier<PIPE_V>();
                Muls(reduceOffsets_, reduceOffsets_, int32_t(8), rows_ * partStride);
            }
            Muls(reduceBases_, index, static_cast<int32_t>(parts * 8), rows_);
            PipeBarrier<PIPE_V>();
            Adds(reduceBases_, reduceBases_, int32_t(4), rows_);
        }
        }
    }
    __aicore__ inline void MakeIndices(const LocalTensor<int32_t>& values, int32_t first, uint32_t count) {
        if constexpr((MODE==1 && SINGLE_TILE && FIXED_ROWS==4 && FIXED_STRIDE==32 && SHORT_FP32_METADATA) || OWNED_AXES8 ||
            (sizeof(T)==2 && SINGLE_TILE && ((MODE==1 && FIXED_STRIDE==32) || (MODE==4 && FIXED_STRIDE==64)))) {
            // Input-independent gather metadata, with no scalar data-processing loop.
            auto seed=values.template ReinterpretCast<uint64_t>();
            seed.SetValue(0,(uint64_t(uint32_t(first+1))<<32)|uint32_t(first));
            seed.SetValue(1,(uint64_t(uint32_t(first+3))<<32)|uint32_t(first+2));
            seed.SetValue(2,(uint64_t(uint32_t(first+5))<<32)|uint32_t(first+4));
            seed.SetValue(3,(uint64_t(uint32_t(first+7))<<32)|uint32_t(first+6));
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        } else {
        Duplicate(values, first, 8);
        SetMaskNorm();
        SetVectorMask<int32_t, MaskMode::NORMAL>(uint64_t(0), uint64_t(0xaa));
        Adds<int32_t, false>(values, values, int32_t(1), uint64_t(64), 1, UnaryRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t, MaskMode::NORMAL>(uint64_t(0), uint64_t(0xcc));
        Adds<int32_t, false>(values, values, int32_t(2), uint64_t(64), 1, UnaryRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t, MaskMode::NORMAL>(uint64_t(0), uint64_t(0xf0));
        Adds<int32_t, false>(values, values, int32_t(4), uint64_t(64), 1, UnaryRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t, MaskMode::NORMAL>(uint64_t(64));
        }
        for (uint32_t filled = 8; filled < count; filled *= 2) {
            const uint32_t n = count-filled < filled ? count-filled : filled;
            Adds(values[filled], values, static_cast<int32_t>(filled), n);
            PipeBarrier<PIPE_V>();
        }
    }
    template <class U>
    __aicore__ inline LocalTensor<U> Allocate(uint32_t &address, uint32_t elements) {
        const uint32_t bytes = (elements * sizeof(U) + 31) / 32 * 32;
        LocalTensor<U> result(TPosition::VECCALC, address, bytes / sizeof(U));
        address += bytes;
        return result;
    }
    __aicore__ inline LocalTensor<float> RawC(uint32_t col) { return columns_[col * (MODE==3?104:MODE==4?96:allocCount_)]; }
    __aicore__ inline LocalTensor<float> C(uint32_t col) {
        if constexpr(MAPPED_STARTS) {
            constexpr uint32_t columns[13]={6,7,8,3,4,5,0,1,2,9,10,11,12};
            return RawC(columns[col]);
        } else return columns_[col * (MODE==3?104:MODE==4?96:allocCount_)];
    }
    __aicore__ inline LocalTensor<float> W(uint32_t col) { return work_[col * allocCount_]; }
    __aicore__ inline LocalTensor<float> P(uint32_t col) {
        if constexpr(DIRECT_HALF_ROW_POINTS || DIRECT_SINGLE_QUERY_POINTS)return pointSlots_[col*8];
        else return xyz_[col*allocCount_];
    }
    __aicore__ inline void LoadSegments(uint32_t batch,uint32_t pointStart,uint32_t validRows,bool prepareMetadata) {
        if constexpr(MODE==3 && sizeof(T)==2) validPointRows_=validRows;
        const uint32_t expandedCount=count_;
        if constexpr(MODE==3 || MODE==4 || SINGLE_TILE)count_=64;
        else if(stride_<=64 && count_>=(sizeof(T)==4?128:256))count_=64;
        Duplicate(rawS_, T(0), stride_ * 6);
        // A native Half32-row tile fits in the existing 160-element arena.
        // Zero padding to 128 coordinates keeps the range certificate unchanged.
        constexpr bool fixedHalfCoordinatePacket=MODE==4 && SINGLE_TILE && sizeof(T)==2 &&
            (FIXED_ROWS==0 || FIXED_ROWS==32);
        const uint32_t transferCount=fixedHalfCoordinatePacket?128U:(validRows*3+63)/64*64;
        Duplicate(rawP_,T(0),transferCount);
        if constexpr(SINGLE_TILE) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        } else PipeBarrier<PIPE_ALL>();
        DataCopyPad(rawS_, sg_[batch * t_.segments * 6],
                    DataCopyExtParams{1, t_.segments * 6 * uint32_t(sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, T(0)});
        DataCopyPad(rawP_,pg_[(batch*t_.points+pointStart)*3],
                        DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},
                        DataCopyPadExtParams<T>{false,0,0,T(0)});
        if constexpr(SINGLE_TILE)SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if constexpr(MODE==3) {
            if(prepareMetadata) {
                const uint32_t coefficientCount=count_;
                count_=expandedCount;
                if constexpr(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) {
                    // Each bitmap selects one coordinate in an arbitrary AoS3
                    // query array. No input values are involved in its construction.
                    auto bitmap=po_.template ReinterpretCast<uint32_t>();
                    constexpr uint32_t maskWords=3U*FIXED_ROWS/32U;
                    constexpr uint32_t words[3]={0x49249249U,0x92492492U,0x24924924U};
                    constexpr uint64_t lanes[3]={0x249249ULL,0x492492ULL,0x924924ULL};
                    SetMaskNorm();
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis) {
                        #pragma unroll
                        for(uint32_t phase=0;phase<3;++phase) {
                            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),lanes[phase]);
                            Duplicate<uint32_t,false>(bitmap[axis*maskWords],words[(axis+phase)%3],uint64_t(64),1,1,8);
                        }
                    }
                    PipeBarrier<PIPE_V>();
                    SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis)
                        Copy(bitmap[axis*maskWords+24],bitmap[axis*maskWords],24,3,CopyRepeatParams(1,1,3,0));
                    PipeBarrier<PIPE_V>();
                } else {
                    MakeIndices(po_,int32_t(0),count_);PipeBarrier<PIPE_V>();
                    Muls(po_,po_,int32_t(12),count_);
                }
                if constexpr(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) {
                    // With one segment, every valid nearest-segment index is0.
                    // The index buffer has no other writer in this specialization.
                    Duplicate(stageIdx_,int32_t(0),rows_);
                }
                count_=coefficientCount;
            }
        } else if constexpr(MODE!=4 || SINGLE_TILE) {
            if(prepareMetadata) {
                const uint32_t coefficientCount=count_;count_=expandedCount;
                PrepareMetadata();count_=coefficientCount;
            }
        }
        if constexpr(SINGLE_TILE)WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        else PipeBarrier<PIPE_ALL>();
        if constexpr(sizeof(T)==2)Cast(pf_,rawP_,RoundMode::CAST_NONE,transferCount);
        if constexpr (sizeof(T) == 2) Cast(sf_, rawS_, RoundMode::CAST_NONE, stride_ * 6);
        PipeBarrier<PIPE_V>();
        if constexpr (MODE == 3) {
            Brcb(W(0),sf_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<6;++axis)
                Copy(C(axis),W(0)[axis*8],64,count_/64,CopyRepeatParams(1,0,8,0));
        } else {
            #pragma unroll
            for (uint32_t axis = 0; axis < 6; ++axis)
                if constexpr(MAPPED_STARTS)
                    Gather(RawC(axis), sf_, so_.template ReinterpretCast<uint32_t>(), axis * 4, count_);
                else Gather(C(axis), sf_, so_.template ReinterpretCast<uint32_t>(), axis * 4, count_);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(MODE==4 && sizeof(T)==2)CheckExactDifferences(transferCount);
        if constexpr(TILE_COEFFICIENTS && sizeof(T)==2)CheckExactTileDifferences(transferCount);
        if constexpr(SINGLE_TILE)PrepareTileGeometry();else PrepareGeometry<MODE!=8>();
        if(count_!=expandedCount) {
            if constexpr(MODE != 3 && MODE != 4 && !TILE_COEFFICIENTS) {
                #pragma unroll
                for(uint32_t col=0;col<13;++col)
                    Copy(C(col),C(col),64,expandedCount/64,CopyRepeatParams(1,1,8,0));
            }
            PipeBarrier<PIPE_V>();
            count_=expandedCount;
        }
        if constexpr(MODE != 3)Brcb(pointSlots_,pf_,(validRows*3+7)/8,BrcbRepeatParams(1,8));
        if constexpr(SINGLE_TILE)PipeBarrier<PIPE_V>();else PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void CheckExactTileDifferences(uint32_t pointCount) {
        if constexpr(PACKED_HALF_ROW_OUTPUT) {
            const uint32_t segmentCount=6*stride_;
            const uint32_t values=segmentCount+pointCount,repeats=values/128U;
            auto packet=W(0).template ReinterpretCast<T>();packet.SetSize(values);
            auto reduced=W(3).template ReinterpretCast<T>();
            auto extrema=W(5).template ReinterpretCast<T>();
            Abs(packet,rawS_,segmentCount);Abs(packet[segmentCount],rawP_,pointCount);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(extrema,packet,128,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(reduced,extrema,repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
            CompareScalar(mask_,packet,T(0),CMPMODE::GT,values);PipeBarrier<PIPE_V>();
            Select(packet,mask_,packet,T(__builtin_inff()),SELMODE::VSEL_TENSOR_SCALAR_MODE,values);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(extrema,packet,128,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(reduced[16],extrema,repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
            Compare<float,uint8_t,false>(W(0).template ReinterpretCast<uint8_t>(),
                C(0),C(3),CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t largest=reduced.template ReinterpretCast<uint16_t>().GetValue(0);
            const uint32_t smallest=reduced.template ReinterpretCast<uint16_t>().GetValue(16);
            const uint32_t highField=largest>>10U,lowField=smallest>>10U;
            // Normal Half exponents translate by 112. A nonzero subnormal
            // mantissa m translates to 103+floor(log2(m)); OR 1 defines zero.
            const uint32_t highExponent=highField==31U?255U:highField?highField+112U:134U-uint32_t(__builtin_clz(largest|1U));
            const uint32_t lowExponent=lowField==31U?255U:lowField?lowField+112U:134U-uint32_t(__builtin_clz(smallest|1U));
            exactDifferences_=largest==0U || highExponent-lowExponent<=13U;
            const uint32_t validBits=t_.segments==32U?~uint32_t(0):(uint32_t(1)<<t_.segments)-1U;
            nativeHalfXTies_=(W(0).template ReinterpretCast<uint32_t>().GetValue(0)&validBits)!=0;
        } else {
        // This runtime certificate covers all input coordinates in the tile.
        // For two eleven-bit significands, gaps <=12 fit 24 bits even with
        // carry. At gap 13 the smaller operand is below the larger operand's
        // gap to its next power of two, so the signed difference still fits.
        // Effective subnormal significands are zero-padded subsets of this rule.
        const uint32_t segmentCount=6*stride_;
        const uint32_t values=(segmentCount+pointCount+63)/64*64,repeats=values/64;
        auto coordinates=W(0);coordinates.SetSize(values);
        if constexpr(PACKED_HALF_ROW_OUTPUT) {
            // Abs writes the complete packed packet directly from the two
            // resident inputs; no intermediate copy or clearing is needed.
            Abs(coordinates,sf_,segmentCount);
            Abs(coordinates[segmentCount],pf_,pointCount);PipeBarrier<PIPE_V>();
        } else {
        Duplicate(coordinates,0.0F,values);PipeBarrier<PIPE_V>();
        if(segmentCount>=64)Copy(coordinates,sf_,64,segmentCount/64,CopyRepeatParams(1,1,8,8));
        if(segmentCount%64)Copy(coordinates[(segmentCount/64)*64],sf_[(segmentCount/64)*64],segmentCount%64,1,CopyRepeatParams(1,1,8,8));
        Copy(coordinates[segmentCount],pf_,64,pointCount/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        Abs(coordinates,coordinates,values);PipeBarrier<PIPE_V>();
        }
        WholeReduceMax(W(3),coordinates,64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(W(4),W(3),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
        CompareScalar(mask_,coordinates,0.0F,CMPMODE::GT,values);PipeBarrier<PIPE_V>();
        Select(coordinates,mask_,coordinates,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,values);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(3),coordinates,64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(5),W(3),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER) {
            // W0 is dead after the extrema reduction. Share its predicate
            // and scalar synchronization with the exact-difference certificate.
            Compare<float,uint8_t,false>(W(0).template ReinterpretCast<uint8_t>(),
                C(0),C(3),CMPMODE::EQ,uint64_t(64),1,
                BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t largest=W(4).template ReinterpretCast<uint32_t>().GetValue(0);
        const uint32_t smallest=W(5).template ReinterpretCast<uint32_t>().GetValue(0);
        const uint32_t highExponent=(largest>>23)&255U,lowExponent=(smallest>>23)&255U;
        exactDifferences_=largest==0 || highExponent-lowExponent<=13;
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER) {
            const uint32_t validBits=t_.segments==32?~uint32_t(0):(uint32_t(1)<<t_.segments)-1U;
            nativeHalfXTies_=(W(0).template ReinterpretCast<uint32_t>().GetValue(0)&validBits)!=0;
        }
            }
}
    __aicore__ inline void CheckExactDifferences(uint32_t pointCount) {
        if constexpr(DIRECT_HALF_ROW_POINTS) {
            // The resident inputs are Half. Reduce their magnitudes before
            // promotion, then keep the original FP32 ratio certificate.
            const uint32_t values=384U+pointCount;
            const uint32_t padded=(values+127U)/128U*128U,repeats=padded/128U;
            auto packet=W(0).template ReinterpretCast<T>();packet.SetSize(padded);
            auto extrema=W(1).template ReinterpretCast<T>();
            auto reduced=W(2).template ReinterpretCast<T>();
            Abs(packet,rawS_,384U);Abs(packet[384],rawP_,pointCount);
            if(padded>values)Duplicate(packet[values],T(0),padded-values);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(extrema,packet,128,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(reduced,extrema,repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
            CompareScalar(mask_,packet,T(0),CMPMODE::GT,padded);PipeBarrier<PIPE_V>();
            Select(packet,mask_,packet,T(__builtin_inff()),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(extrema,packet,128,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(reduced[16],extrema,repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            Cast(W(3),reduced,RoundMode::CAST_NONE,1);
            Cast(W(4),reduced[16],RoundMode::CAST_NONE,1);PipeBarrier<PIPE_V>();
            Muls(W(4),W(4),4096.0F,1);PipeBarrier<PIPE_V>();
            Compare(mask_,W(3),W(4),CMPMODE::LE,1);
            Compare<float,uint8_t,false>(W(0).template ReinterpretCast<uint8_t>(),
                RawC(0),RawC(3),CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            exactDifferences_=(mask_.GetValue(0)&1U)!=0;
            const uint64_t validBits=t_.segments==64U?~uint64_t(0):(uint64_t(1)<<t_.segments)-1U;
            nativeHalfXTies_=(W(0).template ReinterpretCast<uint64_t>().GetValue(0)&validBits)!=0;
        } else {
        // A binary16 value has at most eleven significant bits. An exponent
        // spread of at most twelve makes every signed difference fit binary32.
        Copy(W(0),sf_,64,6,CopyRepeatParams(1,1,8,8));
        Copy(W(0)[384],pf_,64,pointCount/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        const uint32_t values=384+pointCount,repeats=values/64;
        Abs(W(0),W(0),values);PipeBarrier<PIPE_V>();
        WholeReduceMax(W(1),W(0),64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(W(2),W(1),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
        CompareScalar(mask_,W(0),0.0F,CMPMODE::GT,values);PipeBarrier<PIPE_V>();
        Select(W(0),mask_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,values);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(1),W(0),64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(3),W(1),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(W(3),W(3),4096.0F,1);PipeBarrier<PIPE_V>();
        Compare(mask_,W(2),W(3),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER) {
            // Endpoint equality is exact for the promoted Half coordinates.
            // W(0) is dead after the min reduction and remains the equality mask
            // until PrepareTileGeometry consumes it. Share the existing V_S wait.
            Compare<float,uint8_t,false>(W(0).template ReinterpretCast<uint8_t>(),
                RawC(0),RawC(3),CMPMODE::EQ,uint64_t(64),1,
                BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        exactDifferences_=(mask_.GetValue(0)&1U)!=0;
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER) {
            const uint64_t validBits=t_.segments==64?~uint64_t(0):(uint64_t(1)<<t_.segments)-1U;
            nativeHalfXTies_=(W(0).template ReinterpretCast<uint64_t>().GetValue(0)&validBits)!=0;
        }
            }
}
    __aicore__ __attribute__((always_inline)) inline void PrepareTileGeometry() {
        constexpr uint64_t lanes=64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const uint8_t axisPitch=uint8_t(MODE==4?12:allocCount_/8);
        const BinaryRepeatParams axes{1,1,1,axisPitch,axisPitch,axisPitch};
        // The score certificate handles arbitrary endpoint orientation.
        // Keep mapped layouts canonical, because their raw endpoint storage aliases directions.
        if constexpr(!(MODE==2 && SINGLE_TILE && !MAPPED_STARTS)) {
        auto keep=W(8).template ReinterpretCast<uint8_t>();
        auto equal=W(0).template ReinterpretCast<uint8_t>();
        auto less=W(1).template ReinterpretCast<uint8_t>();
        auto equalY=W(2).template ReinterpretCast<uint8_t>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(lanes);
        if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(keep,RawC(0),RawC(3),CMPMODE::LT,lanes,1,same);
        else Compare<float,uint8_t,false>(keep,C(0),C(3),CMPMODE::LT,lanes,1,same);
        if constexpr(!NATIVE_HALF_ENDPOINT_ORDER) {
            if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(equal,RawC(0),RawC(3),CMPMODE::EQ,lanes,1,same);
            else Compare<float,uint8_t,false>(equal,C(0),C(3),CMPMODE::EQ,lanes,1,same);
        }
        bool hasXTies=true;
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER)hasXTies=nativeHalfXTies_;
        if constexpr(sizeof(T)==2 && TILE_COEFFICIENTS && FIXED_STRIDE==32 && !NATIVE_HALF_ENDPOINT_ORDER) {
            // Only equal first coordinates require the remaining lexicographic axes.
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t validBits=t_.segments==32?~uint32_t(0):(uint32_t(1)<<t_.segments)-1U;
            hasXTies=(equal.template ReinterpretCast<uint32_t>().GetValue(0)&validBits)!=0;
        }
        if constexpr(sizeof(T)==4) {
            if constexpr(MODE==1 && SINGLE_TILE && FIXED_ROWS==4 && FIXED_STRIDE==32 && SHORT_FP32_METADATA) {
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const uint32_t validBits=t_.segments==32?~uint32_t(0):(uint32_t(1)<<t_.segments)-1U;
                hasXTies=(equal.template ReinterpretCast<uint32_t>().GetValue(0)&validBits)!=0;
            } else {
                PipeBarrier<PIPE_V>();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(4));
                And<uint16_t,false>(equal.template ReinterpretCast<uint16_t>(),equal.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                hasXTies=equal.template ReinterpretCast<uint64_t>().GetValue(0)!=0;
                SetVectorMask<float,MaskMode::NORMAL>(lanes);
            }
        }
        if(hasXTies) {
        if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(less,RawC(1),RawC(4),CMPMODE::LT,lanes,1,same);
        else Compare<float,uint8_t,false>(less,C(1),C(4),CMPMODE::LT,lanes,1,same);
        if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(equalY,RawC(1),RawC(4),CMPMODE::EQ,lanes,1,same);
        else Compare<float,uint8_t,false>(equalY,C(1),C(4),CMPMODE::EQ,lanes,1,same);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(4));
        And<uint16_t,false>(less.template ReinterpretCast<uint16_t>(),less.template ReinterpretCast<uint16_t>(),equal.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);
        And<uint16_t,false>(equal.template ReinterpretCast<uint16_t>(),equal.template ReinterpretCast<uint16_t>(),equalY.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep.template ReinterpretCast<uint16_t>(),keep.template ReinterpretCast<uint16_t>(),less.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);
        SetVectorMask<float,MaskMode::NORMAL>(lanes);
        if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(less,RawC(2),RawC(5),CMPMODE::LE,lanes,1,same);
        else Compare<float,uint8_t,false>(less,C(2),C(5),CMPMODE::LE,lanes,1,same);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(4));
        And<uint16_t,false>(less.template ReinterpretCast<uint16_t>(),less.template ReinterpretCast<uint16_t>(),equal.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep.template ReinterpretCast<uint16_t>(),keep.template ReinterpretCast<uint16_t>(),less.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float,MaskMode::NORMAL>(lanes);SetCmpMask(keep);
        if constexpr(MAPPED_STARTS) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(0),RawC(0),RawC(3),3,axes);
        } else if constexpr(PACKED_HALF_GEOMETRY_AXES) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(0),C(0),C(3),3,BinaryRepeatParams(1,1,1,8,12,12));
        } else if constexpr(OWNED_AXES8 || PACKED_HALF_ROW_OUTPUT) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(0),C(0),C(3),3,axes);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(axis),C(axis),C(axis+3),1,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(MAPPED_STARTS) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(3),RawC(3),RawC(0),3,axes);
        } else if constexpr(PACKED_HALF_GEOMETRY_AXES) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(3),C(3),C(0),3,BinaryRepeatParams(1,1,1,12,12,12));
        } else if constexpr(OWNED_AXES8 || PACKED_HALF_ROW_OUTPUT) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(3),C(3),C(0),3,axes);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(axis+3),C(axis+3),C(axis),1,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(!MAPPED_STARTS) {
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Adds<float,false>(C(0),W(0),0.0F,lanes,3,UnaryRepeatParams(1,1,12,8));
            } else if constexpr(OWNED_AXES8 || PACKED_HALF_ROW_OUTPUT) {
                Adds<float,false>(C(0),W(0),0.0F,lanes,3,UnaryRepeatParams(1,1,axisPitch,axisPitch));
            } else {
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Adds<float,false>(C(axis),W(axis),0.0F,lanes,1,unary);
            }
            PipeBarrier<PIPE_V>();
        }
        }
        if constexpr(PACKED_HALF_GEOMETRY_AXES) {
            Sub<float,false>(C(6),C(3),C(0),lanes,3,BinaryRepeatParams(1,1,1,12,12,12));
        } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
            Sub<float,false>(C(6),C(3),C(0),lanes,3,axes);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(C(axis+6),C(axis+3),C(axis),lanes,1,same);
        }
        PipeBarrier<PIPE_V>();
        if(!(sizeof(T)==2 && exactDifferences_)) {
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Sub<float,false>(W(0),C(3),C(6),lanes,3,BinaryRepeatParams(1,1,1,8,12,12));
            } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
                Sub<float,false>(W(0),C(3),C(6),lanes,3,axes);
            } else {
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),C(axis+3),C(axis+6),lanes,1,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Add<float,false>(W(3),C(6),W(0),lanes,3,BinaryRepeatParams(1,1,1,8,12,8));
                Sub<float,false>(C(10),W(0),C(0),lanes,3,BinaryRepeatParams(1,1,1,12,8,12));
            } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
                Add<float,false>(W(3),C(6),W(0),lanes,3,axes);
                Sub<float,false>(C(10),W(0),C(0),lanes,3,axes);
            } else {
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis) {
                    Add<float,false>(W(axis+3),C(axis+6),W(axis),lanes,1,same);
                    Sub<float,false>(C(axis+10),W(axis),C(axis),lanes,1,same);
                }
            }
            PipeBarrier<PIPE_V>();
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Sub<float,false>(W(3),C(3),W(3),lanes,3,BinaryRepeatParams(1,1,1,8,12,8));
            } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
                Sub<float,false>(W(3),C(3),W(3),lanes,3,axes);
            } else {
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis+3),C(axis+3),W(axis+3),lanes,1,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Add<float,false>(C(10),C(10),W(3),lanes,3,BinaryRepeatParams(1,1,1,12,12,8));
            } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
                Add<float,false>(C(10),C(10),W(3),lanes,3,axes);
            } else {
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Add<float,false>(C(axis+10),C(axis+10),W(axis+3),lanes,1,same);
            }
        } else {
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Duplicate<float,false>(C(10),0.0F,lanes,3,1,12);
            } else {
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Duplicate<float,false>(C(axis+10),0.0F,lanes,1,1,8);
            }
        }
        Mul<float,false>(C(9),C(6),C(6),lanes,1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(7),C(7),lanes,1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(8),C(8),lanes,1,same);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(mask_,C(9),0.0F,CMPMODE::GT,lanes,1,unary);PipeBarrier<PIPE_V>();
        Select<float,uint8_t,false>(C(9),mask_,C(9),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,lanes,1,same);PipeBarrier<PIPE_V>();
        // One complete coefficient register needs one reduction, without
        // reducing the single result a second time.
        WholeReduceMax(zeroBound_,C(9),64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(zeroBound_,zeroBound_,1.8189894035458565e-12F,1);PipeBarrier<PIPE_V>();
        Brcb(zeroBound_[8],zeroBound_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(zeroBound_[32],zeroBound_[8],64,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) inline void PrepareSingleSegmentGeometryNative() {
        constexpr uint64_t lanes=64;
        const uint8_t workPitch=uint8_t(allocCount_/8);
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(lanes);
        Sub<float,false>(C(6),C(3),C(0),lanes,3,BinaryRepeatParams(1,1,1,13,13,13));
        PipeBarrier<PIPE_V>();
        Sub<float,false>(W(0),C(3),C(6),lanes,3,BinaryRepeatParams(1,1,1,workPitch,13,13));
        PipeBarrier<PIPE_V>();
        Add<float,false>(W(3),C(6),W(0),lanes,3,BinaryRepeatParams(1,1,1,workPitch,13,workPitch));
        Sub<float,false>(C(10),W(0),C(0),lanes,3,BinaryRepeatParams(1,1,1,13,workPitch,13));
        PipeBarrier<PIPE_V>();
        Sub<float,false>(W(3),C(3),W(3),lanes,3,BinaryRepeatParams(1,1,1,workPitch,13,workPitch));
        PipeBarrier<PIPE_V>();
        Add<float,false>(C(10),C(10),W(3),lanes,3,BinaryRepeatParams(1,1,1,13,13,workPitch));
        Mul<float,false>(C(9),C(6),C(6),lanes,1,same);
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(7),C(7),lanes,1,same);
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(8),C(8),lanes,1,same);
        PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(mask_,C(9),0.0F,CMPMODE::GT,lanes,1,unary);
        PipeBarrier<PIPE_V>();
        Select<float,uint8_t,false>(C(9),mask_,C(9),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,lanes,1,same);
    }
    template<bool EXACT> __aicore__ inline void PrepareGeometry(bool canonicalize=true) {
        if constexpr(EXACT && NATIVE_SINGLE_SEGMENT_GEOMETRY) {
            if(count_==64 && allocCount_/8<=255) {
                PrepareSingleSegmentGeometryNative();
                return;
            }
        }
        if constexpr(EXACT) {
        if constexpr (MODE != 3) {
        if(canonicalize) {
            // A segment and its reversed representation must follow identical
            // floating-point operations, including the exact-index tie rule.
            auto keep = W(8).template ReinterpretCast<uint8_t>();
            auto equal = W(0).template ReinterpretCast<uint8_t>();
            auto less = W(1).template ReinterpretCast<uint8_t>();
            auto equalY = W(2).template ReinterpretCast<uint8_t>();
            Compare(keep, C(0), C(3), CMPMODE::LT, count_);
            Compare(equal, C(0), C(3), CMPMODE::EQ, count_);
            Compare(less, C(1), C(4), CMPMODE::LT, count_);
            Compare(equalY, C(1), C(4), CMPMODE::EQ, count_);
            PipeBarrier<PIPE_V>();
            And(less.template ReinterpretCast<uint16_t>(), less.template ReinterpretCast<uint16_t>(),
                equal.template ReinterpretCast<uint16_t>(), count_/16);
            And(equal.template ReinterpretCast<uint16_t>(), equal.template ReinterpretCast<uint16_t>(),
                equalY.template ReinterpretCast<uint16_t>(), count_/16);
            PipeBarrier<PIPE_V>();
            Or(keep.template ReinterpretCast<uint16_t>(), keep.template ReinterpretCast<uint16_t>(),
               less.template ReinterpretCast<uint16_t>(), count_/16);
            Compare(less, C(2), C(5), CMPMODE::LE, count_);
            PipeBarrier<PIPE_V>();
            And(less.template ReinterpretCast<uint16_t>(), less.template ReinterpretCast<uint16_t>(),
                equal.template ReinterpretCast<uint16_t>(), count_/16);
            PipeBarrier<PIPE_V>();
            Or(keep.template ReinterpretCast<uint16_t>(), keep.template ReinterpretCast<uint16_t>(),
               less.template ReinterpretCast<uint16_t>(), count_/16);
            PipeBarrier<PIPE_V>();
            if(count_==64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));SetCmpMask(keep);
                const BinaryRepeatParams same{1,1,1,8,8,8};
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)
                    Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(axis),C(axis),C(axis+3),1,same);
                PipeBarrier<PIPE_V>();
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)
                    Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(axis+3),C(axis+3),C(axis),1,same);
                PipeBarrier<PIPE_V>();
            } else {
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)
                    Select(W(axis),keep,C(axis),C(axis+3),SELMODE::VSEL_TENSOR_TENSOR_MODE,count_);
                PipeBarrier<PIPE_V>();
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)
                    Select(C(axis+3),keep,C(axis+3),C(axis),SELMODE::VSEL_TENSOR_TENSOR_MODE,count_);
                PipeBarrier<PIPE_V>();
            }
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Adds(C(axis), W(axis), 0.0F, count_);
            PipeBarrier<PIPE_V>();
        }
        }
        }
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub(C(axis+6), C(axis+3), C(axis), count_);
        PipeBarrier<PIPE_V>();
        if constexpr(EXACT) {
        if(!((MODE==4 || TILE_COEFFICIENTS) && sizeof(T)==2 && exactDifferences_)) {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub(W(axis), C(axis+3), C(axis+6), count_);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            Add(W(axis+3), C(axis+6), W(axis), count_);
            Sub(C(axis+10), W(axis), C(axis), count_);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub(W(axis+3), C(axis+3), W(axis+3), count_);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add(C(axis+10), C(axis+10), W(axis+3), count_);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Duplicate(C(axis+10),0.0F,64);
        }
        }
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(EXACT && MODE==8 && sizeof(T)==2) {
            Adds(C(9),W(7),0.0F,count_);
            PipeBarrier<PIPE_V>();
        } else {
#endif
        Mul(C(9), C(6), C(6), count_);
        PipeBarrier<PIPE_V>();
        MulAddDst(C(9), C(7), C(7), count_);
        PipeBarrier<PIPE_V>();
        MulAddDst(C(9), C(8), C(8), count_);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask_, C(9), 0.0F, CMPMODE::GT, count_);
        PipeBarrier<PIPE_V>();
        Select(C(9), mask_, C(9), 1.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, count_);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        }
#endif
        if constexpr(EXACT && (MODE!=3 || sizeof(T)==2)) {
            if(!(MODE==8 && selectedCount_==1)) {
            PipeBarrier<PIPE_V>();
            WholeReduceMax(W(6),C(9),64,count_/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMax(zeroBound_,W(6),count_/64,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            Muls(zeroBound_,zeroBound_,1.8189894035458565e-12F,1);PipeBarrier<PIPE_V>();
            Brcb(zeroBound_[8],zeroBound_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Copy(zeroBound_[32],zeroBound_[8],64,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void LoadPoints(uint32_t batch, uint32_t start, uint32_t validRows) {
        if constexpr(MODE==3 && sizeof(T)==2) validPointRows_=validRows;
        const uint32_t transferCount=(validRows*3+63)/64*64;
        Duplicate(rawP_, T(0), transferCount);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(rawP_, pg_[(batch * t_.points + start) * 3],
                    DataCopyExtParams{1, validRows * 3 * uint32_t(sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, T(0)});
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T) == 2) Cast(pf_, rawP_, RoundMode::CAST_NONE, transferCount);
        PipeBarrier<PIPE_V>();
        if constexpr(MODE==4 && sizeof(T)==2)CheckExactDifferences(transferCount);
        if constexpr (MODE != 3) Brcb(pointSlots_, pf_, (validRows * 3 + 7) / 8, BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
    }
    template<bool MAP_ORIGINAL=false> __aicore__ inline void ReducePairs() {
        if constexpr (MODE == 3) {
            out_=W(0);
            if constexpr(!(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024))
                Duplicate(outIdx_, int32_t(0), rows_);
        } else if (stride_ <= 64) {
            WholeReduceMin(records_, W(0), static_cast<int32_t>(stride_), rows_, 1, 1, stride_ / 8,
                           ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            if constexpr(SINGLE_TILE && (sizeof(T)==4 || MODE==4 || PACKED_HALF_ROW_OUTPUT)) {
                // Reduction scratch is dead until the single GM index write.
                outIdx_=reduceWork_.template ReinterpretCast<int32_t>();
                GatherMask(outIdx_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));
                GatherMask(out_,records_,uint8_t(1),GatherMaskParams(1,1,8,0));
            } else Gather(outIdx_, records_.template ReinterpretCast<int32_t>(),
                          outputOffsets_.template ReinterpretCast<uint32_t>(), 4U, rows_);
        } else {
            const uint32_t parts = stride_ / 64;
            const uint32_t partStride = parts < 8 ? 8 : parts;
            WholeReduceMin(reduceWork_, W(0), 64, count_ / 64, 1, 1, 8,
                           ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            Gather(W(6), reduceWork_, reduceOffsets_.template ReinterpretCast<uint32_t>(), 0U,
                   rows_ * partStride);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(records_, W(6), static_cast<int32_t>(parts), rows_, 1, 1, partStride / 8,
                           ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            auto block = W(7).template ReinterpretCast<int32_t>();
            Gather(block, records_.template ReinterpretCast<int32_t>(),
                   outputOffsets_.template ReinterpretCast<uint32_t>(), 4U, rows_);
            PipeBarrier<PIPE_V>();
            Muls(outIdx_, block, int32_t(8), rows_);
            Muls(block, block, int32_t(64), rows_);
            PipeBarrier<PIPE_V>();
            auto chosenOffsets = W(8).template ReinterpretCast<int32_t>();
            Add(chosenOffsets, outIdx_, reduceBases_, rows_);
            PipeBarrier<PIPE_V>();
            Gather(outIdx_, reduceWork_.template ReinterpretCast<int32_t>(),
                   chosenOffsets.template ReinterpretCast<uint32_t>(), 0U, rows_);
            PipeBarrier<PIPE_V>();
            Add(outIdx_, outIdx_, block, rows_);
        }
        if constexpr (MODE != 3) {
            if constexpr(!SINGLE_TILE || (sizeof(T)==2 && MODE!=4 && !PACKED_HALF_ROW_OUTPUT)) Gather(out_,records_,outputOffsets_.template ReinterpretCast<uint32_t>(),0U,rows_);
            else if(stride_>64) Gather(out_,records_,outputOffsets_.template ReinterpretCast<uint32_t>(),0U,rows_);
        }
        if constexpr(!(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) || MAP_ORIGINAL)
            PipeBarrier<PIPE_V>();
        if constexpr(MAP_ORIGINAL) {
            Muls(W(6).template ReinterpretCast<int32_t>(),outIdx_,int32_t(4),1);PipeBarrier<PIPE_V>();
            Gather(outIdx_,so_,W(6).template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void SelectCandidates() {
        const uint8_t repeats=count_/64;
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),P(axis),C(axis),uint64_t(64),repeats,BinaryRepeatParams(1,DIRECT_SINGLE_QUERY_POINTS?0:1,1,8,DIRECT_SINGLE_QUERY_POINTS?0:8,8));
        PipeBarrier<PIPE_V>();
        Mul<float,false>(W(3),W(0),C(6),uint64_t(64),repeats,binary);
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(W(3),W(1),C(7),uint64_t(64),repeats,binary);
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(W(3),W(2),C(8),uint64_t(64),repeats,binary);
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        Div<float,false>(W(6),W(3),C(9),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        Maxs<float,false>(W(6),W(6),0.0F,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Mins<float,false>(W(6),W(6),1.0F,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Mul<float,false>(W(7),W(6),C(9),uint64_t(64),repeats,binary);
        Muls<float,false>(W(3),W(3),-2.0F,uint64_t(64),repeats,unary);
        Muls<float,false>(W(5),W(4),1.52587890625e-5F,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(W(7),W(7),W(3),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(W(4),W(6),W(7),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        Sub<float,false>(W(0),W(4),W(5),uint64_t(64),repeats,binary);
        Add<float,false>(W(5),W(4),W(5),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        // The minimum of conservative upper bounds is itself an incumbent.
        // Every possible winner has a lower bound no greater than this value.
        // Padding must be excluded from both sides of the interval reduction.
        Select(W(5),valid_,W(5),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);PipeBarrier<PIPE_V>();
        WholeReduceMin(records_,W(5),64,count_/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(6),records_,count_/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
#else
        ReducePairs<false>();
        Muls(W(6).template ReinterpretCast<int32_t>(),outIdx_,int32_t(4),1);PipeBarrier<PIPE_V>();
        Gather(W(6),W(5),W(6).template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
#endif
        Brcb(W(7),W(6),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(W(8),W(7),64,count_/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
        Compare(mask_,W(0),W(8),CMPMODE::LE,count_);
        Duplicate(so_,int32_t(0),count_);PipeBarrier<PIPE_V>();
        uint64_t kept=0;
        GatherMask(so_.template ReinterpretCast<float>(),po_.template ReinterpretCast<float>(),
                   mask_.template ReinterpretCast<uint32_t>(),true,stride_,GatherMaskParams(1,1,8,1),kept);
        PipeBarrier<PIPE_V>();
        uint32_t selected=uint32_t(kept);
        if(selected==0 || selected>t_.segments) {
            Copy(so_,po_,64,count_/64,CopyRepeatParams(1,1,8,8));selected=t_.segments;PipeBarrier<PIPE_V>();
        }
        selectedCount_=selected;
        stride_=8;while(stride_<selected)stride_*=2;
        count_=stride_<64?64:stride_;
        Muls(W(6).template ReinterpretCast<int32_t>(),so_,int32_t(24),count_);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<6;++axis)
            Gather(C(axis),sf_,W(6).template ReinterpretCast<uint32_t>(),axis*4,count_);
        PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(MODE==8 && sizeof(T)==2) {
            Muls(W(6).template ReinterpretCast<int32_t>(),so_,int32_t(4),count_);PipeBarrier<PIPE_V>();
            Gather(W(7),C(9),W(6).template ReinterpretCast<uint32_t>(),0U,count_);PipeBarrier<PIPE_V>();
        }
#endif
        PrepareGeometry<true>(selected>1);
        Cast(W(8),po_,RoundMode::CAST_NONE,count_);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,W(8),float(int32_t(selected)),CMPMODE::LT,count_);PipeBarrier<PIPE_V>();
    }
    template<bool EXACT_DIFFERENCES=false,bool HALF_SELECTED_PACKET=false>
    __aicore__ __attribute__((always_inline)) inline void Compute() {
        const uint8_t repeats = HALF_SELECTED_PACKET ? 1 : count_ / 64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        const BinaryRepeatParams coefficient{1,1,1,8,8,(MODE == 3 || MODE == 4 || TILE_COEFFICIENTS) ? 0 : 8};
        // One 64-lane segment packet corresponds to one query. Broadcast
        // its existing eight-value slot directly, with three slots per query.
        const BinaryRepeatParams pointCoefficient{1,(DIRECT_HALF_ROW_POINTS || DIRECT_SINGLE_QUERY_POINTS)?0:1,1,8,DIRECT_SINGLE_QUERY_POINTS?0:DIRECT_HALF_ROW_POINTS?3:8,(MODE==3 || MODE==4 || TILE_COEFFICIENTS)?0:8};
        const BinaryRepeatParams pointBinary{1,(DIRECT_HALF_ROW_POINTS || DIRECT_SINGLE_QUERY_POINTS)?0:1,1,8,DIRECT_SINGLE_QUERY_POINTS?0:DIRECT_HALF_ROW_POINTS?3:8,8};
        const uint8_t axisPitch=uint8_t(allocCount_/8);
        const BinaryRepeatParams axes{1,1,1,axisPitch,axisPitch,axisPitch};
        const BinaryRepeatParams pointAxes{1,DIRECT_SINGLE_QUERY_POINTS?0:1,1,axisPitch,uint8_t(DIRECT_SINGLE_QUERY_POINTS?1:axisPitch),axisPitch};
        const BinaryRepeatParams projectionAxes{1,1,1,axisPitch,0,axisPitch};
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(64));
        // Error-free subtraction preserves short distances at large coordinates.
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Sub<float,false>(W(0),P(0),C(0),uint64_t(64),3,pointAxes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(0)[64],P(0)[64],C(0),uint64_t(64),3,axes);
        } else {
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis), P(axis), C(axis), uint64_t(64), repeats, pointCoefficient);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(!EXACT_DIFFERENCES) {
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Sub<float,false>(W(3),P(0),W(0),uint64_t(64),3,pointAxes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(3)[64],P(0)[64],W(0)[64],uint64_t(64),3,axes);
        } else {
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+3), P(axis), W(axis), uint64_t(64), repeats, pointBinary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Add<float,false>(W(6),W(0),W(3),uint64_t(64),3,axes);
            if constexpr(MAPPED_STARTS) Add<float,false>(W(6)[64],W(0)[64],W(3)[64],uint64_t(64),3,axes);
            Sub<float,false>(W(3),W(3),C(0),uint64_t(64),3,axes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(3)[64],W(3)[64],C(0),uint64_t(64),3,axes);
        } else {
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) {
                Add<float, false>(W(axis+6), W(axis), W(axis+3), uint64_t(64), repeats, binary);
                Sub<float, false>(W(axis+3), W(axis+3), C(axis), uint64_t(64), repeats, coefficient);
            }
        }
        PipeBarrier<PIPE_V>();
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Sub<float,false>(W(6),P(0),W(6),uint64_t(64),3,pointAxes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(6)[64],P(0)[64],W(6)[64],uint64_t(64),3,axes);
        } else {
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+6), P(axis), W(axis+6), uint64_t(64), repeats, pointBinary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Add<float,false>(W(3),W(3),W(6),uint64_t(64),3,axes);
            if constexpr(MAPPED_STARTS) Add<float,false>(W(3)[64],W(3)[64],W(6)[64],uint64_t(64),3,axes);
        } else {
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(W(axis+3), W(axis+3), W(axis+6), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        }
        Mul<float, false>(W(8), W(0), C(6), uint64_t(64), repeats, coefficient);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(8), W(1), C(7), uint64_t(64), repeats, coefficient);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(8), W(2), C(8), uint64_t(64), repeats, coefficient);
        PipeBarrier<PIPE_V>();
        Div<float, false>(W(8), W(8), C(9), uint64_t(64), repeats, coefficient);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(W(8), W(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(W(8), W(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        if constexpr (MODE != 3 && !EXACT_DIFFERENCES) CompareScalar<float, uint8_t, false>(mask_, W(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(W(8), W(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            MulAddDst<float,float,false>(W(0),W(8),C(6),uint64_t(64),3,projectionAxes);
            if constexpr(MAPPED_STARTS) MulAddDst<float,float,false>(W(0)[64],W(8)[64],C(6),uint64_t(64),3,projectionAxes);
            if constexpr(!EXACT_DIFFERENCES) {
                MulAddDst<float,float,false>(W(3),W(8),C(10),uint64_t(64),3,projectionAxes);
                if constexpr(MAPPED_STARTS) MulAddDst<float,float,false>(W(3)[64],W(8)[64],C(10),uint64_t(64),3,projectionAxes);
            }
        } else {
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) {
                MulAddDst<float, float, false>(W(axis), W(8), C(axis+6), uint64_t(64), repeats, coefficient);
                if constexpr(!EXACT_DIFFERENCES)
                    MulAddDst<float,float,false>(W(axis+3),W(8),C(axis+10),uint64_t(64),repeats,coefficient);
            }
        }
        PipeBarrier<PIPE_V>();
        if constexpr(!EXACT_DIFFERENCES) {
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Add<float,false>(W(0),W(0),W(3),uint64_t(64),3,axes);
            if constexpr(MAPPED_STARTS) Add<float,false>(W(0)[64],W(0)[64],W(3)[64],uint64_t(64),3,axes);
        } else {
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(W(axis), W(axis), W(axis+3), uint64_t(64), repeats, binary);
        }
        }
        PipeBarrier<PIPE_V>();
        if constexpr (MODE != 3 && !EXACT_DIFFERENCES) {
        // When all coordinate differences are exact, t=1 gives P-C in one FMA.
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Sub<float,false>(W(3),P(0),C(3),uint64_t(64),3,pointAxes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(3)[64],P(0)[64],C(3),uint64_t(64),3,axes);
        } else {
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+3), P(axis), C(axis+3), uint64_t(64), repeats, pointCoefficient);
        }
        PipeBarrier<PIPE_V>();
        if(count_==64) {
            SetCmpMask(mask_);
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(axis),W(axis),W(axis+3),1,binary);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,uint8_t,false>(W(axis),mask_,W(axis),W(axis+3),SELMODE::VSEL_TENSOR_TENSOR_MODE,uint64_t(64),repeats,binary);
        }
        PipeBarrier<PIPE_V>();
        }
        Mul<float, false>(W(0), W(0), W(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(0), W(1), W(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(0), W(2), W(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr (MODE != 3) Select<float, uint8_t, false>(W(0), valid_, W(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
        if constexpr(MODE==3 && sizeof(T)==2) {
            // Cached gather offsets identify real point lanes. Padded lanes
            // must not contribute NaNs to the near-distance reduction.
            Cast(W(6),po_,RoundMode::CAST_NONE,count_);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(6),float(int32_t(validPointRows_*12U)),CMPMODE::LT,count_);PipeBarrier<PIPE_V>();
            Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);
        }
        // The native M1 final score FMA already has its V barrier, and its
        // alias-only reduction adds no vector instruction before the square root.
        if constexpr(!(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024))
            PipeBarrier<PIPE_V>();
        if constexpr(MODE==8) {
            if(selectedCount_==1) {
                out_=W(0);
                Copy(outIdx_,so_,8,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
            } else ReducePairs<true>();
        } else ReducePairs<false>();
        if constexpr(MODE==3 && sizeof(T)==2) {
            WholeReduceMin(W(6),W(0),64,count_/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMin(zeroBound_[8],W(6),count_/64,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            Compare(mask_,zeroBound_[8],zeroBound_[32],CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if(mask_.GetValue(0)&1U) {
                CorrectAxialZeroScores<true>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),allocCount_,
                    uint32_t(columns_.GetPhyAddr()),104,uint32_t(W(1).GetPhyAddr()),allocCount_,
                    uint32_t(outputOffsets_.GetPhyAddr()),(count_+255)/256*32,count_);
                CorrectCollinearZeroScores<true>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),allocCount_,
                    uint32_t(columns_.GetPhyAddr()),104,uint32_t(zeroExact_.GetPhyAddr()),
                    uint32_t(zeroBound_[32].GetPhyAddr()),uint32_t(outputOffsets_.GetPhyAddr()),count_,true);
            }
        }
        if constexpr(MODE==8) {
            if(!(MODE==8 && selectedCount_==1)) {
            Compare(mask_,out_,zeroBound_[32],CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t nearZero=mask_.template ReinterpretCast<uint64_t>().GetValue(0);
            if(nearZero&((uint64_t(1)<<rows_)-1)) {
                if constexpr(MAPPED_STARTS) {
                    // All ordinary pair scores have been computed. Direction
                    // columns are dead; restore the raw-endpoint ABI only for
                    // the exact geometric-zero helpers, before the final store.
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis)
                        Copy(RawC(axis),C(axis),64,1,CopyRepeatParams(1,1,8,8));
                    PipeBarrier<PIPE_V>();
                }
                if constexpr(DIRECT_SINGLE_QUERY_POINTS) {
                    // Geometric-zero helpers retain their expanded-point ABI.
                    // Materialize that view only on entry to those helpers.
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis)
                        Copy(xyz_[axis*allocCount_],pointSlots_[axis*8],64,count_/64,CopyRepeatParams(1,0,8,0));
                    PipeBarrier<PIPE_V>();
                }
                const uint32_t predicatePitch=(count_+255)/256*32;
                CorrectAxialZeroScores<MODE==4 || TILE_COEFFICIENTS>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),allocCount_,
                    uint32_t(columns_.GetPhyAddr()),MODE==4?96:allocCount_,uint32_t(W(1).GetPhyAddr()),allocCount_,
                    uint32_t(po_.GetPhyAddr()),predicatePitch,count_);
                CorrectCollinearZeroScores<sizeof(T)==2>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),allocCount_,
                    uint32_t(columns_.GetPhyAddr()),MODE==4?96:allocCount_,uint32_t(zeroExact_.GetPhyAddr()),
                    uint32_t(zeroBound_[32].GetPhyAddr()),uint32_t(po_.GetPhyAddr()),count_,MODE==4 || TILE_COEFFICIENTS);
                Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);PipeBarrier<PIPE_V>();
                if constexpr(MODE==8) {
                    if(selectedCount_==1)out_=W(0);else ReducePairs<true>();
                } else ReducePairs<false>();
            }
            }
        }
        CheckNumericalRisk<EXACT_DIFFERENCES && SHORT_FP32_METADATA && SINGLE_TILE && MODE==2>();
        if constexpr(sizeof(T)==2) {
            Sqrt(out_,out_,rows_);PipeBarrier<PIPE_V>();
            Cast(outT_,out_,RoundMode::CAST_RINT,rows_);
        } else if constexpr(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) {
            // The same1024 values use16 complete repeats with the existing64 mask.
            // Avoid counter-mask setup/restoration for this fixed-capacity drain.
            Sqrt<float,false>(outT_,out_,uint64_t(64),repeats,unary);
        } else Sqrt(outT_,out_,rows_);
        if constexpr(SINGLE_TILE) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        } else PipeBarrier<PIPE_ALL>();
    }
    TreeLineMinDistanceTilingData t_;
    uint32_t stride_, rows_, count_, allocCount_, group_, selectedCount_, validPointRows_, validComputeRows_;
    bool exactDifferences_, nativeHalfXTies_, needsExact_;
    GlobalTensor<T> pg_, sg_, dg_;
    GlobalTensor<int32_t> ig_;
    LocalTensor<T> rawS_, rawP_, outT_, stageOut_;
    LocalTensor<float> sf_, pf_, pointSlots_, columns_, xyz_, work_, records_, reduceWork_, out_, zeroBound_, zeroExact_;
    LocalTensor<int32_t> po_, so_, outputOffsets_, outIdx_, reduceOffsets_, reduceBases_, stageIdx_;
    LocalTensor<uint8_t> mask_, valid_;
};

template<class T>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyExactFallback(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,
    uint32_t rows,uint32_t segmentCount);

template <class T,uint32_t POINT_ROWS=1,uint32_t PAIR_STRIDE=8,bool CONTIG_SEGMENTS=false,bool BLOCK_INDICES=false,bool SINGLE_SEGMENT=false,bool OPTIMIZE_FOUR_ROW=false,uint32_t FIXED_SEGMENTS=0,bool PAIR_SINGLE_POINT=false,bool DEFER_SOC_INIT=false>
class KernelTreeTiny {
    // Zero encodes one point per batch; physical row capacity remains one.
    // Positive values retain the original flattened/partial-row ownership.
    static constexpr uint32_t ROWS=POINT_ROWS?POINT_ROWS:1;
    static constexpr bool STATIC_ONE_POINT=POINT_ROWS==0;
    static constexpr bool THIN_SINGLE_FP32=ROWS==1 && SINGLE_SEGMENT && sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool NATIVE_SINGLE_POINT=PAIR_SINGLE_POINT && THIN_SINGLE_FP32;
    static constexpr bool NATIVE_ONE_ROW=ROWS==1 && !SINGLE_SEGMENT &&
#ifdef DTYPE_TREE_POINTS
        ((sizeof(T)==4 && sizeof(DTYPE_TREE_POINTS)==4) ||
         (STATIC_ONE_POINT && sizeof(T)==2 && sizeof(DTYPE_TREE_POINTS)==2))
#else
        sizeof(T)==4
#endif
        ;
    static constexpr bool NATIVE_FOUR_ROW_SINGLE=OPTIMIZE_FOUR_ROW && ROWS==4 && PAIR_STRIDE==1 && SINGLE_SEGMENT && sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool NATIVE_HALF_PACKET=FIXED_SEGMENTS==2 && ROWS==1 && NATIVE_ONE_ROW && sizeof(T)==2;
    static constexpr bool COMPLETE_ENDPOINT_LOAD=NATIVE_FOUR_ROW_SINGLE ||
        (FIXED_SEGMENTS==2 && ROWS==4 && !CONTIG_SEGMENTS && sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
         && sizeof(DTYPE_TREE_POINTS)==4
#endif
        );
    static constexpr bool MAPPED_TINY_STARTS=NATIVE_ONE_ROW;
public:
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                               const TreeLineMinDistanceTilingData& t) {
        t_ = t; rows_ = ROWS; pairs_ = ROWS==1?8:ROWS*PAIR_STRIDE; count_ = ROWS==4 && PAIR_STRIDE==1?32:64;
        pg_.SetGlobalBuffer((__gm__ T*)p); sg_.SetGlobalBuffer((__gm__ T*)s);
        dg_.SetGlobalBuffer((__gm__ T*)d); ig_.SetGlobalBuffer((__gm__ int32_t*)i);
        uint32_t address = 0;
        rawP_ = Alloc<T>(address, 256 / sizeof(T));
        rawS_ = Alloc<T>(address, 512 / sizeof(T));
        if constexpr (sizeof(T) == 2) {
            fp_ = Alloc<float>(address, 128); fs_ = Alloc<float>(address, 256);
        } else { fp_ = rawP_.template ReinterpretCast<float>(); fs_ = rawS_.template ReinterpretCast<float>(); }
        p_ = Alloc<float>(address, count_); a_ = Alloc<float>(address, count_); c_ = Alloc<float>(address, count_);
        work_ = Alloc<float>(address, count_ * 8);
        if constexpr(MAPPED_TINY_STARTS) originalA_=a_;
        pointOffsets_ = Alloc<int32_t>(address, count_);
        segmentOffsets_ = Alloc<int32_t>(address, count_);
        lengths_ = Alloc<float>(address, 64); projection_ = Alloc<float>(address, 64);
        scores_ = Alloc<float>(address, 64); records_ = Alloc<float>(address, 64);
        out_ = Alloc<float>(address, 64); outT_ = Alloc<T>(address, 64);
        indices_ = Alloc<int32_t>(address, 64);
        masks_ = Alloc<uint8_t>(address, 512);
        zeroGeometry_=Alloc<float>(address,9*64);
        zeroExact_=Alloc<float>(address,45*64);
        if constexpr(!DEFER_SOC_INIT) InitializeVectorState();
    }
    __aicore__ inline void Process(uint32_t logicalOwner=GetBlockIdx()) {
        const uint32_t pointCount=STATIC_ONE_POINT || (ROWS==1 && SINGLE_SEGMENT)?1U:t_.points;
        const uint32_t segmentCount=SINGLE_SEGMENT?1U:FIXED_SEGMENTS?FIXED_SEGMENTS:t_.segments;
        uint32_t batch,pointPosition,validRows;
        if constexpr(ROWS==1) {
            // One-row tiles own consecutive flattened point/output positions.
            batch=pointCount==1?logicalOwner:logicalOwner/pointCount;
            pointPosition=logicalOwner;
            validRows=1;
        } else {
            const uint32_t tiles=(pointCount+rows_-1)/rows_;
            batch=logicalOwner/tiles;
            const uint32_t start=logicalOwner%tiles*rows_;
            pointPosition=batch*pointCount+start;
            validRows=pointCount-start<rows_?pointCount-start:rows_;
        }
        // The native single-segment load defines both padded endpoint
        // records; no preceding vector write is needed for this domain.
        if constexpr(ROWS!=1 && !COMPLETE_ENDPOINT_LOAD)Duplicate(rawS_,T(0),CONTIG_SEGMENTS?64:512/sizeof(T));
        if constexpr(ROWS!=1 && !COMPLETE_ENDPOINT_LOAD) {SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);}
        DataCopyPad(rawP_,pg_[pointPosition*3],DataCopyExtParams{static_cast<uint16_t>(validRows),3U*sizeof(T),0,0,0},
                    DataCopyPadExtParams<T>{false,0,0,T(0)});
        if constexpr(CONTIG_SEGMENTS) {
            DataCopyPad(rawS_,sg_[batch*segmentCount*6],DataCopyExtParams{1,segmentCount*6U*sizeof(T),0,0,0},
                        DataCopyPadExtParams<T>{false,0,0,T(0)});
            if constexpr(DEFER_SOC_INIT) { InitSocState(); InitializeVectorState(); }
            // i = 8*segment+coordinate; its packed source offset is 4*i-8*segment.
            if constexpr(BLOCK_INDICES) {
                if constexpr(FIXED_SEGMENTS==2 && ROWS==1 && NATIVE_ONE_ROW && sizeof(T)==4) {
                    // Construct byte offsets directly with the original vector seed.
                    MakeIndices<4>(pointOffsets_,8);PipeBarrier<PIPE_V>();
                } else {
                MakeIndices(pointOffsets_,8);PipeBarrier<PIPE_V>();
                Muls(pointOffsets_,pointOffsets_,int32_t(4),8);PipeBarrier<PIPE_V>();
                }
                Adds(pointOffsets_[8],pointOffsets_,int32_t(24),8);PipeBarrier<PIPE_V>();
                if(segmentCount>2) {Adds(pointOffsets_[16],pointOffsets_,int32_t(48),16);PipeBarrier<PIPE_V>();}
                if(segmentCount>4) {Adds(pointOffsets_[32],pointOffsets_,int32_t(96),32);PipeBarrier<PIPE_V>();}
            } else {
                MakeIndices(pointOffsets_,64);PipeBarrier<PIPE_V>();
                ShiftRight(segmentOffsets_,pointOffsets_,int32_t(3),64);
                Muls(pointOffsets_,pointOffsets_,int32_t(4),64);PipeBarrier<PIPE_V>();
                Muls(segmentOffsets_,segmentOffsets_,int32_t(-8),64);PipeBarrier<PIPE_V>();
                Add(segmentOffsets_,segmentOffsets_,pointOffsets_,64);PipeBarrier<PIPE_V>();
            }
        } else {
            DataCopyPad(rawS_,sg_[batch*segmentCount*6],DataCopyExtParams{static_cast<uint16_t>(2*segmentCount),3U*sizeof(T),0,0,0},
                        DataCopyPadExtParams<T>{false,0,0,T(0)});
            if constexpr(DEFER_SOC_INIT) { InitSocState(); InitializeVectorState(); }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if constexpr(NATIVE_HALF_PACKET) {
            auto absoluteInputs=zeroGeometry_.template ReinterpretCast<T>();
            SetVectorMask<T,MaskMode::NORMAL>(uint64_t(16));
            Abs<T,false>(absoluteInputs,rawP_,uint64_t(128),2,UnaryRepeatParams(1,1,1,8));
        }
        if constexpr (sizeof(T)==2) {
            if constexpr(NATIVE_HALF_PACKET) {
                // Repeat strides address the two independent padded DMA records.
                const LocalTensor<float> convertedInputs(TPosition::VECCALC,uint32_t(fp_.GetPhyAddr()),256);
                const LocalTensor<T> rawInputs(TPosition::VECCALC,uint32_t(rawP_.GetPhyAddr()),256);
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Cast<float,T,false>(convertedInputs,rawInputs,RoundMode::CAST_NONE,uint64_t(64),2,UnaryRepeatParams(1,1,16,8));
            } else {
                Cast(fp_,rawP_,RoundMode::CAST_NONE,validRows*slot_);
                Cast(fs_,rawS_,RoundMode::CAST_NONE,CONTIG_SEGMENTS?((6*segmentCount+7)/8*8):(2*segmentCount*slot_));
            }
            PipeBarrier<PIPE_V>();
        }
        if constexpr(NATIVE_HALF_PACKET) {
            auto absoluteInputs=zeroGeometry_.template ReinterpretCast<T>();
            auto belowMaximum=zeroGeometry_[72].template ReinterpretCast<uint8_t>();
            SetVectorMask<T,MaskMode::NORMAL>(uint64_t(32));
            CompareScalar<T,uint8_t,false>(belowMaximum,absoluteInputs,T(2.0F),CMPMODE::LT,uint64_t(128),1,UnaryRepeatParams(1,1,8,8));
        }
        if constexpr(NATIVE_SINGLE_POINT) {
            // Padded native DMA records already form complete coordinate vectors.
            p_=fp_;a_=fs_;c_=fs_[slot_];
        } else {
        Copy(p_,fp_,THIN_SINGLE_FP32?8:PAIR_STRIDE*8,rows_,CopyRepeatParams(1,0,PAIR_STRIDE,slot_/8));
        if constexpr(CONTIG_SEGMENTS) {
            auto sourceOffsets=BLOCK_INDICES?pointOffsets_:segmentOffsets_;
            const uint32_t coordinateCount=BLOCK_INDICES?(segmentCount<=2?16:segmentCount<=4?32:64):64;
            Gather(a_,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),0U,coordinateCount);
            Gather(c_,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),12U,coordinateCount);
        } else if constexpr(NATIVE_FOUR_ROW_SINGLE) {
            const LocalTensor<float> endpointPairs(TPosition::VECCALC,uint32_t(a_.GetPhyAddr()),64);
            Copy(endpointPairs,fs_,uint64_t(32),2,CopyRepeatParams(1,0,4,1));
        } else {
            Copy(a_,fs_,THIN_SINGLE_FP32?8:PAIR_STRIDE*8,rows_,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
            Copy(c_,fs_[slot_],THIN_SINGLE_FP32?8:PAIR_STRIDE*8,rows_,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
        }
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_ONE_ROW || NATIVE_FOUR_ROW_SINGLE ||
                     (COMPLETE_ENDPOINT_LOAD && FIXED_SEGMENTS==2)) {
            // Axis alignment is a geometric property of the loaded endpoints.
            // The exact lattice/bound checks below guard the direct AABB result.
            const uint32_t pairs=ROWS==1?segmentCount:validRows*PAIR_STRIDE;
            const uint32_t active=(1U<<pairs)-1U;
            auto axisFlags=zeroExact_.template ReinterpretCast<uint8_t>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs*8U));
            Compare<float,uint8_t,false>(axisFlags,a_,c_,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t fixed=axisFlags.template ReinterpretCast<uint64_t>().GetValue(0);
            const uint64_t axis=((fixed&(fixed>>1))|(fixed&(fixed>>2))|((fixed>>1)&(fixed>>2)))&0x0101010101010101ULL;
            const uint32_t aligned=uint32_t((axis*0x0102040810204080ULL)>>56);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            bool directReady=false;
            if((aligned&active)==active) {
                if constexpr(NATIVE_FOUR_ROW_SINGLE) directReady=TryAxisSingle(validRows);
                else directReady=TryAxisDyadicGeometry(validRows,segmentCount);
            }
            if(directReady) {
                WriteTinyResult(pointPosition,validRows);
                return;
            }
        }
        if constexpr(NATIVE_HALF_PACKET) {
            // Coordinate range flags reuse the original input-copy barrier.
        }
        if constexpr(NATIVE_HALF_PACKET) {
            auto rangeFlags=zeroGeometry_.template ReinterpretCast<uint32_t>();
            const uint32_t accepted=rangeFlags.GetValue(72);
            if((accepted&0x0fff0007U)==0x0fff0007U) {
                // Bounded Half coordinates keep endpoint dot and norm errors below the guard.
                const BinaryRepeatParams same{1,1,1,8,8,8};
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Sub<float,false>(W(0),c_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(2),p_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(3),p_,c_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                const LocalTensor<float> endpointPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),128);
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(zeroGeometry_[128].GetPhyAddr()),128);
                Mul<float,false>(products,endpointPairs,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,8,8,0));PipeBarrier<PIPE_V>();
                WholeReduceSum(zeroGeometry_[256],zeroGeometry_[128],3,2,1,1,1);
                WholeReduceSum(zeroGeometry_[320],zeroGeometry_[192],3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(masks_,zeroGeometry_[256],-0.001953125F,CMPMODE::LT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
                CompareScalar<float,uint8_t,false>(masks_[64],zeroGeometry_[320],0.001953125F,CMPMODE::GT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const uint32_t atStart=masks_.template ReinterpretCast<uint32_t>().GetValue(0);
                const uint32_t atEnd=masks_[64].template ReinterpretCast<uint32_t>().GetValue(0);
                if(((atStart|atEnd)&3U)==3U) {
                    // Only reduced predicates choose addresses; every residual is vector-computed.
                    const auto residual0=(atStart&1U)?W(2):W(3);
                    const auto residual1=(atStart&2U)?W(2)[8]:W(3)[8];
                    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Duplicate<float,false>(scores_,__builtin_inff(),uint64_t(64),1,1,8);
                    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
                    Mul<float,false>(zeroGeometry_[128],residual0,residual0,uint64_t(64),1,same);
                    Mul<float,false>(zeroGeometry_[136],residual1,residual1,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                    WholeReduceSum(scores_,zeroGeometry_[128],3,2,1,1,1);PipeBarrier<PIPE_V>();
                    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    ReduceTiny();
                    // Bounded endpoint arithmetic still needs an index certificate.
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const auto scoreBits=scores_.template ReinterpretCast<uint32_t>();
                    const uint32_t score0=scoreBits.GetValue(0),score1=scoreBits.GetValue(1);
                    // Nonnegative finite endpoint norms differ by far more than
                    // the bounded binary32 accumulation error at 128 ULPs.
                    const uint32_t separation=score0>score1?score0-score1:score1-score0;
                    if(separation<=128U && !TryExactDyadicGeometry(validRows,segmentCount)) {
                        ExactFallback(pointPosition,batch,validRows,segmentCount);return;
                    }
                    WriteTinyResult(pointPosition,validRows);
                    // No subsequent UB reuse or GM overwrite follows this terminal write.
                    return;
                }
            }
        }
        if constexpr(NATIVE_ONE_ROW && STATIC_ONE_POINT && FIXED_SEGMENTS==2 && sizeof(T)==4) {
            // Certify bounded differences and residuals after vector geometry.
            auto bounds=zeroExact_.template ReinterpretCast<uint8_t>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Sub<float,false>(W(0),c_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(2),p_,a_,uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),192);
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(zeroGeometry_[128].GetPhyAddr()),128);
                Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,8,16,0));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(zeroGeometry_[256],zeroGeometry_[128],3,2,1,1,1);
                WholeReduceSum<float,false>(lengths_,zeroGeometry_[192],3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(bounds[64],lengths_,0.000244140625F,CMPMODE::GE,uint64_t(64),1,unary);
                CompareScalar<float,uint8_t,false>(bounds[320],lengths_,64.0F,CMPMODE::LE,uint64_t(64),1,unary);
                Maxs<float,false>(lengths_,lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Div<float,false>(projection_,zeroGeometry_[256],lengths_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Muls<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(zeroGeometry_[256],projection_,lengths_,uint64_t(64),1,same);
                Muls<float,false>(zeroGeometry_[384],lengths_,0.00000095367431640625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Abs<float,false>(zeroGeometry_[256],zeroGeometry_[256],uint64_t(64),1,unary);
                Maxs<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Compare<float,uint8_t,false>(bounds[128],zeroGeometry_[256],zeroGeometry_[384],CMPMODE::LE,uint64_t(64),1,same);
                Mins<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(scores_,W(2),3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(bounds[384],scores_,64.0F,CMPMODE::LE,uint64_t(64),1,unary);
                WholeReduceMin<float,false>(records_,scores_,2,1,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);
                WholeReduceMax<float,false>(zeroGeometry_[384],scores_,2,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                out_=records_;
                GatherMask(indices_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
                Sub<float,false>(zeroGeometry_[384],zeroGeometry_[384],out_,uint64_t(64),1,same);
                CompareScalar<float,uint8_t,false>(bounds[192],out_,0.015625F,CMPMODE::GE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                CompareScalar<float,uint8_t,false>(bounds[256],zeroGeometry_[384],0.00390625F,CMPMODE::GT,uint64_t(64),1,unary);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                auto flags=bounds.template ReinterpretCast<uint32_t>();
                if((((flags.GetValue(16)&flags.GetValue(32)&flags.GetValue(80)&flags.GetValue(96))&3U)==3U &&
                   ((flags.GetValue(48)&flags.GetValue(64))&1U)==1U) ||
                   TryExactDyadicGeometry(validRows,segmentCount)) {
                    Sqrt<float,false>(out_,out_,uint64_t(64),1,unary);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyPad(dg_[pointPosition],out_,DataCopyExtParams{1,4,0,0,0});
                    DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,4,0,0,0});
                    // No subsequent UB reuse or GM overwrite follows this terminal write.
                    return;
                }
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        if constexpr(NATIVE_ONE_ROW && !STATIC_ONE_POINT && FIXED_SEGMENTS==2 && sizeof(T)==4) {
            // Certify bounded differences and residuals after vector geometry.
            auto bounds=zeroExact_.template ReinterpretCast<uint8_t>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Sub<float,false>(W(0),c_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(2),p_,a_,uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),192);
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(zeroGeometry_[128].GetPhyAddr()),128);
                Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,8,16,0));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(zeroGeometry_[256],zeroGeometry_[128],3,2,1,1,1);
                WholeReduceSum<float,false>(lengths_,zeroGeometry_[192],3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(bounds[64],lengths_,0.000244140625F,CMPMODE::GE,uint64_t(64),1,unary);
                CompareScalar<float,uint8_t,false>(bounds[320],lengths_,256.0F,CMPMODE::LE,uint64_t(64),1,unary);
                Maxs<float,false>(lengths_,lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Div<float,false>(projection_,zeroGeometry_[256],lengths_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Muls<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(zeroGeometry_[256],projection_,lengths_,uint64_t(64),1,same);
                Muls<float,false>(zeroGeometry_[384],lengths_,0.0000002384185791015625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Abs<float,false>(zeroGeometry_[256],zeroGeometry_[256],uint64_t(64),1,unary);
                Maxs<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Compare<float,uint8_t,false>(bounds[128],zeroGeometry_[256],zeroGeometry_[384],CMPMODE::LE,uint64_t(64),1,same);
                Mins<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(scores_,W(2),3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(bounds[384],scores_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);
                WholeReduceMin<float,false>(records_,scores_,2,1,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);
                WholeReduceMax<float,false>(zeroGeometry_[384],scores_,2,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                out_=records_;
                GatherMask(indices_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
                Sub<float,false>(zeroGeometry_[384],zeroGeometry_[384],out_,uint64_t(64),1,same);
                CompareScalar<float,uint8_t,false>(bounds[192],out_,0.25F,CMPMODE::GE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                CompareScalar<float,uint8_t,false>(bounds[256],zeroGeometry_[384],0.03125F,CMPMODE::GT,uint64_t(64),1,unary);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                auto flags=bounds.template ReinterpretCast<uint32_t>();
                if((((flags.GetValue(16)&flags.GetValue(32)&flags.GetValue(80)&flags.GetValue(96))&3U)==3U &&
                   ((flags.GetValue(48)&flags.GetValue(64))&1U)==1U) ||
                   TryExactDyadicGeometry(validRows,segmentCount)) {
                    Sqrt<float,false>(out_,out_,uint64_t(64),1,unary);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyPad(dg_[pointPosition],out_,DataCopyExtParams{1,4,0,0,0});
                    DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,4,0,0,0});
                    // No subsequent UB reuse or GM overwrite follows this terminal write.
                    return;
                }
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        // The bounded certificate below permits ordinary FP32 differences;
        // difficult exponents, cancellation and ties use exact vector packets.
        if constexpr(THIN_SINGLE_FP32)count_=8;
        else if constexpr(ROWS==1 && sizeof(T)==2 && !NATIVE_ONE_ROW)count_=SINGLE_SEGMENT?8:(segmentCount<=2?16:segmentCount<=4?32:64);
        else if constexpr(NATIVE_HALF_PACKET)count_=16;
        else if constexpr(ROWS==4 && PAIR_STRIDE==2 && FIXED_SEGMENTS==2 && sizeof(T)==4)count_=64;
        else if constexpr(ROWS>1 && !SINGLE_SEGMENT)count_=validRows*PAIR_STRIDE*8;
        Sub(W(0),c_,a_,count_);
        Sub(W(2),p_,a_,count_);
        PipeBarrier<PIPE_V>();
        if constexpr(THIN_SINGLE_FP32) {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
            if constexpr(NATIVE_SINGLE_POINT) {
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(W(4).GetPhyAddr()),16);
                const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),24);
                Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,1,2,0));
            } else {
                Mul<float,false>(W(4),W(0),W(0),uint64_t(64),1,same);
                Mul<float,false>(W(5),W(2),W(0),uint64_t(64),1,same);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(NATIVE_SINGLE_POINT?5:4),3,1,1,1,1);
            WholeReduceSum(projection_,W(NATIVE_SINGLE_POINT?4:5),3,1,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
            // The certificate tests the original norm; clamping only the
            // divisor cannot admit an underflowed or degenerate segment.
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
            Muls<float,false>(W(6),W(6),-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,1,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else if constexpr(NATIVE_ONE_ROW) {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(count_));
            const LocalTensor<float> productPairs(TPosition::VECCALC,uint32_t(originalA_.GetPhyAddr()),512);
            const LocalTensor<float> pointDirectionPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),192);
            Mul<float,false>(productPairs,pointDirectionPairs,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,56,16,0));
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Duplicate<float,false>(lengths_,1.0F,uint64_t(64),1,1,8);
            Duplicate<float,false>(projection_,0.0F,uint64_t(64),1,1,8);
            if constexpr(!SINGLE_SEGMENT)Duplicate<float,false>(scores_,__builtin_inff(),uint64_t(64),1,1,8);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(5),3,SINGLE_SEGMENT?1:segmentCount,1,1,1);
            WholeReduceSum(projection_,W(4),3,SINGLE_SEGMENT?1:segmentCount,1,1,1);PipeBarrier<PIPE_V>();
            // Keep the original norm for the numerical certificate. Only the
            // divisor is clamped; small/degenerate norms retain the exact fallback.
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(count_));
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(count_));
            if constexpr(!SINGLE_SEGMENT)CompareScalar(masks_,W(6),1.0F,CMPMODE::LT,64);
            Muls<float,false>(W(6),W(6),-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            const LocalTensor<float> residualPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),128);
            const LocalTensor<float> directionPairs(TPosition::VECCALC,uint32_t(W(0).GetPhyAddr()),128);
            MulAddDst<float,float,false>(residualPairs,W(6),directionPairs,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,0,8));PipeBarrier<PIPE_V>();
            if constexpr(!SINGLE_SEGMENT)Sub<float,false>(W(4),p_,c_,uint64_t(64),1,same);
            PipeBarrier<PIPE_V>();
            if constexpr(!SINGLE_SEGMENT) {
                SetCmpMask(masks_);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(2),W(2),W(4),1,same);PipeBarrier<PIPE_V>();
            }
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,SINGLE_SEGMENT?1:segmentCount,1,1,1);PipeBarrier<PIPE_V>();
        } else if constexpr(NATIVE_FOUR_ROW_SINGLE) {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            // One repeat forms each independent set of dot/norm products.
            const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),96);
            const LocalTensor<float> products(TPosition::VECCALC,uint32_t(W(4).GetPhyAddr()),64);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
            Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,4,8,0));
            Duplicate(projection_,0.0F,8);PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(5),3,4,1,1,1);
            WholeReduceSum(projection_,W(4),3,4,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(4));
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Muls<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
            MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,4,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else if constexpr(ROWS==4 && PAIR_STRIDE==2 && FIXED_SEGMENTS==2 && sizeof(T)==4) {
            // Eight independent padded point/segment pairs fit one native vector packet.
            // Only the first validRows outputs are stored; inactive rows cannot join an argmin.
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),1,same);
            Mul<float,false>(W(5),W(2),W(0),uint64_t(64),1,same);
            Duplicate<float,false>(lengths_,1.0F,uint64_t(64),1,1,8);
            Duplicate<float,false>(projection_,0.0F,uint64_t(64),1,1,8);
            Duplicate<float,false>(scores_,__builtin_inff(),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
            WholeReduceSum<float,false>(lengths_,W(4),3,8,1,1,1);
            WholeReduceSum<float,false>(projection_,W(5),3,8,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            CompareScalar<float,uint8_t,false>(masks_,lengths_,0.0F,CMPMODE::GT,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Select(lengths_,masks_,lengths_,1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,lengths_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            CompareScalar<float,uint8_t,false>(masks_,W(6),1.0F,CMPMODE::LT,uint64_t(64),1,unary);
            Muls<float,false>(W(6),W(6),-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);
            Sub<float,false>(W(4),p_,c_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            SetCmpMask(masks_);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(2),W(2),W(4),1,same);PipeBarrier<PIPE_V>();
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
            WholeReduceSum<float,false>(scores_,W(2),3,8,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Select(scores_,masks_[384],scores_,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        } else {
        Mul(W(4),W(0),W(0),count_); Mul(W(5),W(2),W(0),count_);
            Duplicate(lengths_,1.0F,64); Duplicate(projection_,0.0F,64);
            if constexpr(!NATIVE_FOUR_ROW_SINGLE)Duplicate(scores_,__builtin_inff(),64);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(4),3,ROWS==1?(SINGLE_SEGMENT?1:segmentCount):pairs_,1,1,1);
            WholeReduceSum(projection_,W(5),3,ROWS==1?(SINGLE_SEGMENT?1:segmentCount):pairs_,1,1,1);
            PipeBarrier<PIPE_V>();
            CompareScalar(masks_,lengths_,0.0F,CMPMODE::GT,64);
            PipeBarrier<PIPE_V>();
            Select(lengths_,masks_,lengths_,1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
            PipeBarrier<PIPE_V>();
            Div(projection_,projection_,lengths_,64);
            PipeBarrier<PIPE_V>();
            Maxs(projection_,projection_,0.0F,64);
            PipeBarrier<PIPE_V>();
            Mins(projection_,projection_,1.0F,64);
            PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,(pairs_+7)/8,BrcbRepeatParams(1,8));
            PipeBarrier<PIPE_V>();
            if(!SINGLE_SEGMENT && segmentCount>1) CompareScalar(masks_,W(6),1.0F,CMPMODE::LT,64);
            Muls(W(6),W(6),-1.0F,count_);
            PipeBarrier<PIPE_V>();
            if constexpr(NATIVE_FOUR_ROW_SINGLE) {
                const LocalTensor<float> residualPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),64);
                const LocalTensor<float> directionPairs(TPosition::VECCALC,uint32_t(W(0).GetPhyAddr()),64);
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
                MulAddDst<float,float,false>(residualPairs,W(6),directionPairs,uint64_t(64),1,BinaryRepeatParams(1,1,1,4,0,4));
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            } else {
                MulAddDst(W(2),W(6),W(0),count_);
            }
            PipeBarrier<PIPE_V>();
            if(!SINGLE_SEGMENT && segmentCount>1) Sub(W(4),p_,c_,count_);
            PipeBarrier<PIPE_V>();
            if(!SINGLE_SEGMENT && segmentCount>1) Select(W(2),masks_,W(2),W(4),SELMODE::VSEL_CMPMASK_SPR,count_);
            PipeBarrier<PIPE_V>();
            Mul(W(2),W(2),W(2),count_);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,ROWS==1?(SINGLE_SEGMENT?1:segmentCount):pairs_,1,1,1);
            PipeBarrier<PIPE_V>();
            if constexpr(ROWS!=1 && !SINGLE_SEGMENT)Select(scores_,masks_[384],scores_,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
            PipeBarrier<PIPE_V>();
        }
        ReduceTiny();
        if(!CertifiedTiny(validRows,segmentCount) && !TryExactDyadicGeometry(validRows,segmentCount)) {
            ExactFallback(pointPosition,batch,validRows,segmentCount);return;
        }
        WriteTinyResult(pointPosition,validRows);
        // No subsequent UB reuse or GM overwrite follows this terminal write.
    }
private:
    __aicore__ inline void WriteTinyResult(uint32_t pointPosition,uint32_t validRows) {
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(ROWS));
        Sqrt<float,false>(out_,out_,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        if constexpr(sizeof(T)==2) {
            PipeBarrier<PIPE_V>();
            Cast<T,float,false>(outT_,out_,RoundMode::CAST_RINT,uint64_t(64),1,UnaryRepeatParams(1,1,4,8));
        } else outT_=out_;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dg_[pointPosition],outT_,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
        DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,validRows*4,0,0,0});
    }
    __aicore__ inline void InitializeVectorState() {
        // Count-form APIs restore normal mode; native masked operations preserve it.
        SetMaskNorm();
        if constexpr(!SINGLE_SEGMENT && !NATIVE_ONE_ROW) {
            const uint32_t segmentCount=FIXED_SEGMENTS?FIXED_SEGMENTS:t_.segments;
            const uint32_t bits=(SINGLE_SEGMENT?1U:((1U<<segmentCount)-1U))*(ROWS==1?0x01010101U:0x55555555U);
            Duplicate(masks_[384].template ReinterpretCast<uint32_t>(),bits,2);
        }
        if constexpr(SINGLE_SEGMENT && sizeof(T)==2) {
            Duplicate(masks_[384].template ReinterpretCast<uint32_t>(),uint32_t((1U<<ROWS)-1U),2);
        }
        if constexpr(ROWS!=1 && !SINGLE_SEGMENT)Duplicate(segmentOffsets_,int32_t(1),64);
    }
    __aicore__ inline void DyadicLatticeFlags(const LocalTensor<float>& left,
            const LocalTensor<float>& right,uint32_t coordinates,bool packedInputs=false) {
        auto scaled=zeroExact_[1280],rounded=zeroExact_[1472],restored=zeroExact_[1664];
        auto flags=zeroExact_[880].template ReinterpretCast<uint8_t>();
        if(packedInputs && FIXED_SEGMENTS==2) {
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Muls<float,false>(scaled,p_,16.0F,uint64_t(64),3,UnaryRepeatParams(1,1,8,8));
        } else {
            Muls(scaled,p_,16.0F,coordinates);
            Muls(scaled[64],left,16.0F,coordinates);
            Muls(scaled[128],right,16.0F,coordinates);
        }
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        const UnaryRepeatParams blocks{1,1,8,8};
        // Round in FP32: scaled coordinates on the dyadic lattice are already integers.
        // Large FP32 values are also grid-aligned; downstream finite residual bounds remain required.
        Cast<float,float,false>(restored,scaled,RoundMode::CAST_RINT,uint64_t(64),3,blocks);PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Compare<float,uint8_t,false>(flags,scaled,restored,CMPMODE::EQ,uint64_t(64),1,same);
        Compare<float,uint8_t,false>(flags[64],scaled[64],restored[64],CMPMODE::EQ,uint64_t(64),1,same);
        Compare<float,uint8_t,false>(flags[128],scaled[128],restored[128],CMPMODE::EQ,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool TryAxisDyadicGeometry(uint32_t validRows,uint32_t segments) {
        // Axis alignment was established from original endpoints. On the 2^-4
        // lattice, bounded residuals and their three-term squared norm are exact.
        const uint32_t pairs=ROWS==1?segments:validRows*PAIR_STRIDE;
        const uint32_t coordinates=pairs*8U;
        const uint64_t prefix=coordinates==64?~uint64_t(0):((uint64_t(1)<<coordinates)-1U);
        const uint64_t live=prefix&0x0707070707070707ULL;
        auto lower=zeroExact_[704],upper=zeroExact_[768],residual=zeroExact_[192];
        auto magnitude=zeroExact_[320];
        auto flags=zeroExact_[832].template ReinterpretCast<uint8_t>();
        DyadicLatticeFlags(a_,c_,coordinates,count_==64 && FIXED_SEGMENTS==2);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        Min<float,false>(lower,a_,c_,uint64_t(64),1,same);Max<float,false>(upper,a_,c_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Max<float,false>(lower,p_,lower,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Min<float,false>(lower,lower,upper,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Sub<float,false>(residual,p_,lower,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Abs<float,false>(magnitude,residual,uint64_t(64),1,unary);
        Mul<float,false>(residual,residual,residual,uint64_t(64),1,same);
        // Fixed reductions read exactly the written segment scores; inactive rows are independent.
        if constexpr(FIXED_SEGMENTS==0) Duplicate(scores_,__builtin_inff(),64);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        CompareScalar<float,uint8_t,false>(flags,magnitude,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        WholeReduceSum(scores_,residual,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        ReduceTiny();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto words=flags.template ReinterpretCast<uint64_t>();
        const uint64_t admitted=words.GetValue(24)&words.GetValue(32)&words.GetValue(40)&words.GetValue(0);
        return (admitted&live)==live;
    }
    __aicore__ inline bool TryAxisSingle(uint32_t validRows) {
        // A single axis-aligned segment has no argmin ambiguity. AABB clipping
        // avoids projection cancellation, including zero-length/zero-distance.
        // Finite residuals bounded by 2^30 keep squares and sums far from overflow;
        // subnormal residual loss is far below the contract's absolute tolerance.
        const uint32_t coordinates=validRows*8U;
        auto clipped=zeroExact_[704],upper=zeroExact_[768],residual=zeroExact_[192];
        auto flags=zeroExact_[832].template ReinterpretCast<uint8_t>();
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        Min<float,false>(clipped,a_,c_,uint64_t(64),1,same);Max<float,false>(upper,a_,c_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Max<float,false>(clipped,p_,clipped,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Min<float,false>(clipped,clipped,upper,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Sub<float,false>(residual,p_,clipped,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Abs<float,false>(upper,residual,uint64_t(64),1,unary);
        Mul<float,false>(residual,residual,residual,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(flags,upper,1073741824.0F,CMPMODE::LE,uint64_t(64),1,unary);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
        WholeReduceSum<float,false>(out_,residual,3,validRows,1,1,1);
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(8));
        Duplicate<int32_t,false>(indices_,int32_t(0),uint64_t(64),1,1,8);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t live=((uint64_t(1)<<coordinates)-1)&0x0707070707070707ULL;
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        return (flags.template ReinterpretCast<uint64_t>().GetValue(0)&live)==live;
    }
    template<bool INPUT_READY>
    __aicore__ __attribute__((noinline)) static bool ExactDyadicPrepared(uint32_t validRows,uint32_t segments) {
        // Reconstruct compile-time UB views locally instead of spilling the caller object.
        // Input DMA and its completion event remain owned by the caller.
        KernelTreeTiny local;
        const TreeLineMinDistanceTilingData resource{0,ROWS,segments,8,ROWS,0,0,3};
        local.Init(nullptr,nullptr,nullptr,nullptr,resource);
        if constexpr(NATIVE_SINGLE_POINT) {
            local.p_=local.fp_;local.a_=local.fs_;local.c_=local.fs_[slot_];
        }
        return local.ComputeExactDyadicGeometry(validRows,segments,INPUT_READY);
    }
    __aicore__ inline bool TryExactDyadicGeometry(uint32_t validRows,uint32_t segments,bool inputReady=false) {
        const bool accepted=inputReady?ExactDyadicPrepared<true>(validRows,segments):
                                         ExactDyadicPrepared<false>(validRows,segments);
        if constexpr(SINGLE_SEGMENT && (ROWS==1 || PAIR_STRIDE==1))out_=scores_;
        else if constexpr(ROWS==1)out_=records_;
        return accepted;
    }
    __aicore__ inline bool ComputeExactDyadicGeometry(uint32_t validRows,uint32_t segments,bool inputReady=false) {
        // Matched integer reconstruction certifies a 2^-4 coordinate lattice.
        // A difference bounded by 64 is exact: small operands fit binary32,
        // and larger nearby same-sign operands satisfy Sterbenz's lemma.
        // Relative/direction bounds make dots and squared norms exact (22 bits).
        // Endpoint projections and axis-aligned segments therefore admit an
        // exact argmin, including zero length, zero distance and true ties.
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const uint32_t pairs=ROWS==1?segments:validRows*PAIR_STRIDE;
        const uint32_t coordinates=pairs*8;
        const uint64_t prefix=coordinates==64?~uint64_t(0):((uint64_t(1)<<coordinates)-1);
        const uint64_t liveCoordinates=prefix&((ROWS>1 && PAIR_STRIDE==2 && segments==1)?
            0x0007000700070007ULL:0x0707070707070707ULL);
        const uint32_t livePairs=((1U<<pairs)-1U)&
            ((ROWS>1 && PAIR_STRIDE==2 && segments==1)?0x55U:0xffU);
        auto left=zeroExact_,right=zeroExact_[64];
        auto delta=zeroExact_[128],relA=zeroExact_[192],relC=zeroExact_[256];
        auto temp=zeroExact_[320],rounded=zeroExact_[384],product=zeroExact_[448];
        auto norm=zeroExact_[512],dot=zeroExact_[576],endpoint=zeroExact_[640];
        auto box=zeroExact_[704],upper=zeroExact_[768];
        auto flags=zeroExact_[832].template ReinterpretCast<uint8_t>();
        if(inputReady) {left=a_;right=c_;}
        else {
        if constexpr(CONTIG_SEGMENTS) {
            auto sourceOffsets=BLOCK_INDICES?pointOffsets_:segmentOffsets_;
            Gather(left,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),0U,coordinates);
            Gather(right,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),12U,coordinates);
        } else {
            Copy(left,fs_,PAIR_STRIDE*8,validRows,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
            Copy(right,fs_[slot_],PAIR_STRIDE*8,validRows,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
        }
        PipeBarrier<PIPE_V>();
        }
        DyadicLatticeFlags(left,right,coordinates,inputReady && count_==64);
        Sub(delta,right,left,coordinates);Sub(relA,p_,left,coordinates);Sub(relC,p_,right,coordinates);PipeBarrier<PIPE_V>();
        if constexpr(FIXED_SEGMENTS==2) {
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Abs<float,false>(temp,delta,uint64_t(64),3,UnaryRepeatParams(1,1,8,8));
        } else {
            Abs(temp,delta,coordinates);Abs(rounded,relA,coordinates);Abs(product,relC,coordinates);
        }
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        CompareScalar<float,uint8_t,false>(flags[96],temp,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,false>(flags[128],rounded,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,false>(flags[160],product,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        Compare<float,uint8_t,false>(flags,left,right,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const auto words=flags.template ReinterpretCast<uint64_t>();
        const uint64_t admitted=words.GetValue(24)&words.GetValue(32)&words.GetValue(40);
        const uint64_t bounded=words.GetValue(12)&words.GetValue(16)&words.GetValue(20);
        if(((admitted&bounded)&liveCoordinates)!=liveCoordinates)return false;
        const uint64_t fixed=words.GetValue(0);
        const uint64_t axis=((fixed&(fixed>>1))|(fixed&(fixed>>2))|((fixed>>1)&(fixed>>2)))&0x0101010101010101ULL;
        const uint32_t axisPairs=inputReady?livePairs:uint32_t((axis*0x0102040810204080ULL)>>56);
        if((axisPairs&livePairs)==livePairs) {
            // An axis-aligned segment equals its AABB. The verified 2^-4
            // lattice and bounded differences make these squared norms exact.
            Min(box,left,right,coordinates);Max(product,left,right,coordinates);PipeBarrier<PIPE_V>();
            Max(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
            Min(box,box,product,coordinates);PipeBarrier<PIPE_V>();
            Sub(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
            Mul(box,box,box,coordinates);Duplicate(scores_,__builtin_inff(),64);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,box,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            if(livePairs!=((1U<<pairs)-1U)) {
                Duplicate(flags.template ReinterpretCast<uint32_t>(),livePairs,2);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Select(scores_,flags,scores_,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            }
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            ReduceTiny();return true;
        }
        Mul(product,delta,delta,coordinates);Mul(temp,relA,delta,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(norm,product,3,pairs,1,1,1);
        WholeReduceSum(dot,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
        CompareScalar<float,uint8_t,false>(flags[32],dot,0.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        Compare<float,uint8_t,false>(flags[64],dot,norm,CMPMODE::GE,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t endpoints=uint32_t(words.GetValue(4)|words.GetValue(8));
        if(((axisPairs|endpoints)&livePairs)!=livePairs) {
            // On this lattice every cross-product subtraction is exact. A
            // certified zero beats every nonzero distance without any division.
            auto u=zeroExact_[896],v=zeroExact_[960],cross=zeroExact_[1024],crossNorm=zeroExact_[1088];
            auto rotate1=zeroExact_[1152].template ReinterpretCast<int32_t>();
            auto rotate2=zeroExact_[1216].template ReinterpretCast<int32_t>();
            MakeIndices(rotate1,64);PipeBarrier<PIPE_V>();
            ShiftRight(rotate1,rotate1,int32_t(3),64);PipeBarrier<PIPE_V>();
            Muls(rotate1,rotate1,int32_t(32),64);PipeBarrier<PIPE_V>();
            Adds(rotate2,rotate1,int32_t(8),64);Adds(rotate1,rotate1,int32_t(4),64);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x0202020202020202ULL));
            Adds<int32_t,false>(rotate1,rotate1,int32_t(4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Adds<int32_t,false>(rotate2,rotate2,int32_t(-8),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x0404040404040404ULL));
            Adds<int32_t,false>(rotate1,rotate1,int32_t(-4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Adds<int32_t,false>(rotate2,rotate2,int32_t(-4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Gather(u,relA,rotate1.template ReinterpretCast<uint32_t>(),0U,coordinates);
            Gather(v,delta,rotate2.template ReinterpretCast<uint32_t>(),0U,coordinates);PipeBarrier<PIPE_V>();
            Mul(cross,u,v,coordinates);PipeBarrier<PIPE_V>();
            Gather(u,relA,rotate2.template ReinterpretCast<uint32_t>(),0U,coordinates);
            Gather(v,delta,rotate1.template ReinterpretCast<uint32_t>(),0U,coordinates);PipeBarrier<PIPE_V>();
            Mul(u,u,v,coordinates);PipeBarrier<PIPE_V>();
            Sub(cross,cross,u,coordinates);PipeBarrier<PIPE_V>();
            Mul(cross,cross,cross,coordinates);PipeBarrier<PIPE_V>();
            WholeReduceSum(crossNorm,cross,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            Mul(product,relA,relA,coordinates);PipeBarrier<PIPE_V>();
            WholeReduceSum(endpoint,product,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
            CompareScalar<float,uint8_t,false>(flags,crossNorm,0.0F,CMPMODE::EQ,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            CompareScalar<float,uint8_t,false>(flags[32],dot,0.0F,CMPMODE::GE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Compare<float,uint8_t,false>(flags[64],dot,norm,CMPMODE::LE,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            CompareScalar<float,uint8_t,false>(flags[96],norm,0.0F,CMPMODE::GT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            CompareScalar<float,uint8_t,false>(flags[128],endpoint,0.0F,CMPMODE::EQ,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t zeroPairs=uint32_t(words.GetValue(0)&words.GetValue(4)&words.GetValue(8)&
                (words.GetValue(12)|words.GetValue(16)))&livePairs;
            const uint32_t firstPairs=ROWS==1?1U:(PAIR_STRIDE==2?(livePairs&0x55U):livePairs);
            const bool allRowsZero=ROWS==1?zeroPairs!=0U:
                (PAIR_STRIDE==2?(((zeroPairs|(zeroPairs>>1))&firstPairs)==firstPairs):zeroPairs==livePairs);
            if(allRowsZero) {
                Duplicate(endpoint,0.0F,64);
                Duplicate(flags.template ReinterpretCast<uint32_t>(),zeroPairs,2);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                ReduceTiny();return true;
            }
            // Equal undirected segments have equal mathematical distances for
            // every point. Compare original endpoints before forcing index zero.
            Copy(u,left,uint64_t(8),pairs,CopyRepeatParams(1,1,1,0));
            Copy(v,right,uint64_t(8),pairs,CopyRepeatParams(1,1,1,0));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Compare<float,uint8_t,false>(flags,left,u,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[32],right,v,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[64],left,v,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[96],right,u,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t forward=words.GetValue(0)&words.GetValue(4),reverse=words.GetValue(8)&words.GetValue(12);
            const uint64_t identical=((forward&(forward>>1)&(forward>>2))|(reverse&(reverse>>1)&(reverse>>2)))&0x0101010101010101ULL;
            const uint32_t identicalPairs=uint32_t((identical*0x0102040810204080ULL)>>56);
            if((identicalPairs&livePairs)!=livePairs)return false;
            // Positive cross norms avoid projection cancellation; float
            // multiply/sum/divide contributes far less than the declared tolerance.
            Mul(temp,relC,relC,coordinates);PipeBarrier<PIPE_V>();
            WholeReduceSum(upper,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            Min(endpoint,endpoint,upper,pairs);Maxs(norm,norm,0.00390625F,pairs);PipeBarrier<PIPE_V>();
            Div(upper,crossNorm,norm,pairs);PipeBarrier<PIPE_V>();
            Duplicate(flags.template ReinterpretCast<uint32_t>(),endpoints,2);PipeBarrier<PIPE_V>();
            SetCmpMask(flags);SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(endpoint,endpoint,upper,1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
            Duplicate(flags.template ReinterpretCast<uint32_t>(),firstPairs,2);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            ReduceTiny();return true;
        }
        Mul(product,relA,relA,coordinates);Mul(temp,relC,relC,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(endpoint,product,3,pairs,1,1,1);
        WholeReduceSum(upper,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        Min(box,left,right,coordinates);Max(product,left,right,coordinates);PipeBarrier<PIPE_V>();
        Max(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
        Min(box,box,product,coordinates);Min(endpoint,endpoint,upper,pairs);PipeBarrier<PIPE_V>();
        Sub(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
        Mul(box,box,box,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(upper,box,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        Duplicate(flags.template ReinterpretCast<uint32_t>(),axisPairs,2);PipeBarrier<PIPE_V>();
        SetCmpMask(flags);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(endpoint,upper,endpoint,1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
        Duplicate(flags.template ReinterpretCast<uint32_t>(),livePairs,2);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        ReduceTiny();
        return true;
    }
    __aicore__ inline void ExactFallback(uint32_t position,uint32_t batch,uint32_t rows,uint32_t segments) {
        TreeTinyExactFallback<T>((GM_ADDR)pg_[position*3].GetPhyAddr(),
            (GM_ADDR)sg_[batch*segments*6].GetPhyAddr(),(GM_ADDR)dg_[position].GetPhyAddr(),
            (GM_ADDR)ig_[position].GetPhyAddr(),rows,segments);
    }
    __aicore__ inline bool CertifiedTiny(uint32_t validRows,uint32_t segments) {
        // In this bounded domain, the norm error is below 2^-13 and the
        // squared-score error below 2^-6. A gap > 2^-5 proves the winner;
        // the admitted residual range also leaves the specified FP32 tolerance margin.
        if constexpr(ROWS==1 && SINGLE_SEGMENT) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t norm=lengths_.template ReinterpretCast<uint32_t>().GetValue(0);
            const uint32_t score=scores_.template ReinterpretCast<uint32_t>().GetValue(0);
            // Ordered positive IEEE fields certify the same inclusive bounds.
            return norm>=0x39800000U && norm<=0x44800000U &&
                   score>=0x3f800000U && score<=0x44800000U;
        }
        auto flags=zeroGeometry_.template ReinterpretCast<uint8_t>();
        const uint32_t n=SINGLE_SEGMENT?validRows:ROWS==1?segments:validRows*PAIR_STRIDE;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(n));
        CompareScalar<float,uint8_t,false>(flags,lengths_,0.000244140625F,CMPMODE::GE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[32],lengths_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[64],scores_,ROWS>1 && !SINGLE_SEGMENT?0.25F:1.0F,CMPMODE::GE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[96],scores_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        if constexpr(!SINGLE_SEGMENT) {
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            if constexpr(ROWS==1) {
                Brcb(W(6),out_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
                Adds<float,false>(W(6),W(6),0.03125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Compare<float,uint8_t,false>(flags[128],scores_,W(6),CMPMODE::LE,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            } else {
                // The pair reduction already supplies each row's larger score.
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(validRows));
                Adds<float,false>(W(7),out_,0.03125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Compare<float,uint8_t,false>(flags[128],W(6),W(7),CMPMODE::GT,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto words=flags.template ReinterpretCast<uint32_t>();
        const uint32_t active=(1U<<n)-1U;
        const uint32_t safe=words.GetValue(0)&words.GetValue(8)&words.GetValue(16)&words.GetValue(24);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        if((safe&active)!=active)return false;
        if constexpr(!SINGLE_SEGMENT) {
            const uint32_t close=words.GetValue(32)&active;
            if constexpr(ROWS==1)return close && !(close&(close-1U));
            else return (close&((1U<<validRows)-1U))==((1U<<validRows)-1U);
        }
        return true;
    }
    __aicore__ inline void ReduceTiny() {
        if constexpr(SINGLE_SEGMENT) {
            if constexpr(ROWS==1 || PAIR_STRIDE==1)out_=scores_;
            else GatherMask(out_,scores_,uint8_t(1),GatherMaskParams(1,1,8,0));
            Duplicate(indices_,int32_t(0),ROWS);PipeBarrier<PIPE_V>();
        } else if constexpr(ROWS==1) {
            WholeReduceMin(records_,scores_,FIXED_SEGMENTS?FIXED_SEGMENTS:8,rows_,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            out_=records_;
            GatherMask(indices_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));
            PipeBarrier<PIPE_V>();
        } else if constexpr(ROWS==4 && PAIR_STRIDE==2 && FIXED_SEGMENTS==2 && sizeof(T)==4) {
            // Gather overwrites every live result lane; four rows fit a single packet.
            GatherMask(out_,scores_,uint8_t(1),GatherMaskParams(1,1,8,0));
            GatherMask(W(4),scores_,uint8_t(2),GatherMaskParams(1,1,8,0));
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(8));
            Duplicate<int32_t,false>(indices_,int32_t(0),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(4));
            Compare<float>(out_,W(4),CMPMODE::LE,uint64_t(4),same);PipeBarrier<PIPE_V>();
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(indices_.template ReinterpretCast<float>(),indices_.template ReinterpretCast<float>(),segmentOffsets_.template ReinterpretCast<float>(),1,same);
            Max<float,false>(W(6),out_,W(4),uint64_t(64),1,same);
            Min<float,false>(out_,out_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else {
            Duplicate(out_,__builtin_inff(),64);
            Duplicate(indices_,int32_t(0),64);PipeBarrier<PIPE_V>();
            GatherMask(out_,scores_,uint8_t(1),GatherMaskParams(1,1,8,0));
            GatherMask(W(4),scores_,uint8_t(2),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            
            Compare<float>(out_,W(4),CMPMODE::LE,uint64_t(64),same);PipeBarrier<PIPE_V>();
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(indices_.template ReinterpretCast<float>(),indices_.template ReinterpretCast<float>(),segmentOffsets_.template ReinterpretCast<float>(),1,same);
            Max(W(6),out_,W(4),32);
            Min(out_,out_,W(4),32);PipeBarrier<PIPE_V>();
        }
    }
    template<class U> __aicore__ inline LocalTensor<U> Alloc(uint32_t& address,uint32_t n) {
        uint32_t bytes=(n*sizeof(U)+31)/32*32;
        LocalTensor<U> v(TPosition::VECCALC,address,bytes/sizeof(U));address+=bytes;return v;
    }
    __aicore__ inline LocalTensor<float> W(uint32_t k) {
        if constexpr(NATIVE_HALF_PACKET) {
            if(k<4)return work_[(k^2U)*64U];
        } else if constexpr(NATIVE_ONE_ROW || NATIVE_FOUR_ROW_SINGLE || NATIVE_SINGLE_POINT) {
            if(k<4)return work_[(k^2U)*count_];
        }
        if constexpr(MAPPED_TINY_STARTS) { if(k==4)return originalA_; }
        return work_[k*(NATIVE_HALF_PACKET?64U:count_)];
    }
    template<int32_t STEP=1> __aicore__ inline void MakeIndices(const LocalTensor<int32_t>& v,uint32_t n) {
        Duplicate(v,int32_t(0),8);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaa));
        Adds<int32_t,false>(v,v,int32_t(STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xcc));
        Adds<int32_t,false>(v,v,int32_t(2*STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xf0));
        Adds<int32_t,false>(v,v,int32_t(4*STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        for(uint32_t filled=8;filled<n;filled*=2){Adds(v[filled],v,int32_t(filled*STEP),filled);PipeBarrier<PIPE_V>();}
    }
    __aicore__ inline void Difference(const LocalTensor<float>& x,const LocalTensor<float>& y,
                                      const LocalTensor<float>& hi,const LocalTensor<float>& lo) {
        Sub(hi,x,y,count_);PipeBarrier<PIPE_V>();
        Sub(W(6),x,hi,count_);PipeBarrier<PIPE_V>();
        Add(W(7),hi,W(6),count_);Sub(lo,W(6),y,count_);PipeBarrier<PIPE_V>();
        Sub(W(7),x,W(7),count_);PipeBarrier<PIPE_V>();
        Add(lo,lo,W(7),count_);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Canonicalize() {
        auto less=masks_.template ReinterpretCast<uint16_t>();
        auto eq=masks_[64].template ReinterpretCast<uint16_t>();
        auto keep=masks_[128].template ReinterpretCast<uint16_t>();
        auto tmp=masks_[192].template ReinterpretCast<uint16_t>();
        auto ones=masks_[256].template ReinterpretCast<uint16_t>();
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(NATIVE_HALF_PACKET?16:64));
        Compare<float,uint8_t,false>(masks_,a_,c_,CMPMODE::LT,uint64_t(64),1,binary);
        Compare<float,uint8_t,false>(masks_[64],a_,c_,CMPMODE::EQ,uint64_t(64),1,binary);PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_HALF_PACKET) {
            // Float metadata bounds reuse the original canonical comparison barrier.
            // Two packed eight-lane records use only the first predicate word.
            // Merge predicate metadata after the vector comparisons complete.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t lt=less.GetValue(0),equal=eq.GetValue(0);
            const uint16_t packet=uint16_t((lt|((lt>>1)&equal)|((lt>>2)&equal&(equal>>1)))&0x0101U);
            keep.SetValue(0,uint16_t(packet*255U));
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        } else if constexpr(NATIVE_ONE_ROW && sizeof(T)==4) {
            // Eight coordinate records occupy one 64-bit predicate word.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if constexpr(FIXED_SEGMENTS==2) {
                // Only the two real coordinate records feed this bounded resource.
                const uint32_t lt=less.template ReinterpretCast<uint32_t>().GetValue(0);
                const uint32_t equal=eq.template ReinterpretCast<uint32_t>().GetValue(0);
                const uint32_t packet=(lt|((lt>>1)&equal)|((lt>>2)&equal&(equal>>1)))&0x0101U;
                keep.template ReinterpretCast<uint64_t>().SetValue(0,uint64_t(packet*255U));
            } else {
                const uint64_t lt=less.template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t equal=eq.template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t packet=(lt|((lt>>1)&equal)|((lt>>2)&equal&(equal>>1)))&0x0101010101010101ULL;
                keep.template ReinterpretCast<uint64_t>().SetValue(0,packet*255ULL);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        } else {
        // Packed predicate words share one mask throughout the lexicographic merge.
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(NATIVE_HALF_PACKET?1:4));
        Duplicate<uint16_t,false>(ones,uint16_t(0x0101),uint64_t(128),1,1,8);PipeBarrier<PIPE_V>();
        ShiftRight<uint16_t,false>(tmp,less,uint16_t(1),uint64_t(128),1,unary,false);PipeBarrier<PIPE_V>();
        And<uint16_t,false>(tmp,tmp,eq,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep,less,tmp,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        ShiftRight<uint16_t,false>(tmp,keep,uint16_t(1),uint64_t(128),1,unary,false);PipeBarrier<PIPE_V>();
        And<uint16_t,false>(tmp,tmp,eq,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep,less,tmp,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        And<uint16_t,false>(keep,keep,ones,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        Muls<int16_t,false>(keep.template ReinterpretCast<int16_t>(),keep.template ReinterpretCast<int16_t>(),int16_t(255),uint64_t(128),1,unary);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(NATIVE_HALF_PACKET?16:64));SetCmpMask(masks_[128]);
        if constexpr(MAPPED_TINY_STARTS)
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(work_[4*count_],a_,c_,1,binary);
        else Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(4),a_,c_,1,binary);
        PipeBarrier<PIPE_V>();
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(c_,c_,a_,1,binary);PipeBarrier<PIPE_V>();
        if constexpr(MAPPED_TINY_STARTS) {
            // Original A is dead after canonical B consumes it. Keep canonical
            // A in its selected plane and reuse old A for ordinary W4 scratch.
            // CorrectTinyZeroScores receives the explicit A address and only
            // uses work planes 0, 1 and 5, so it cannot overwrite canonical A.
            a_=work_[4*count_];
        } else {Copy(a_,W(4),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
    }
    static constexpr uint32_t slot_=32/sizeof(T);
    TreeLineMinDistanceTilingData t_;uint32_t rows_,pairs_,count_;
    GlobalTensor<T> pg_,sg_,dg_;GlobalTensor<int32_t> ig_;
    LocalTensor<T> rawP_,rawS_,outT_;
    LocalTensor<float> fp_,fs_,p_,a_,c_,work_,lengths_,projection_,scores_,records_,out_,zeroGeometry_,zeroExact_;
    LocalTensor<float> originalA_;
    LocalTensor<int32_t> pointOffsets_,segmentOffsets_,indices_;
    LocalTensor<uint8_t> masks_;
};

// Independent vector implementation for the single-point, single-segment resource.
template <class T,uint32_t POINT_ROWS=1,uint32_t PAIR_STRIDE=8,bool CONTIG_SEGMENTS=false,bool BLOCK_INDICES=false,bool SINGLE_SEGMENT=false,bool OPTIMIZE_FOUR_ROW=false,uint32_t FIXED_SEGMENTS=0,bool PAIR_SINGLE_POINT=false,bool DEFER_SOC_INIT=false>
class KernelTreeSinglePointSingleSegment {
    // Zero encodes one point per batch; physical row capacity remains one.
    // Positive values retain the original flattened/partial-row ownership.
    static constexpr uint32_t ROWS=POINT_ROWS?POINT_ROWS:1;
    static constexpr bool STATIC_ONE_POINT=POINT_ROWS==0;
    static constexpr bool THIN_SINGLE_FP32=ROWS==1 && SINGLE_SEGMENT && sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool NATIVE_SINGLE_POINT=PAIR_SINGLE_POINT && THIN_SINGLE_FP32;
    static constexpr bool NATIVE_ONE_ROW=ROWS==1 && !SINGLE_SEGMENT &&
#ifdef DTYPE_TREE_POINTS
        ((sizeof(T)==4 && sizeof(DTYPE_TREE_POINTS)==4) ||
         (STATIC_ONE_POINT && sizeof(T)==2 && sizeof(DTYPE_TREE_POINTS)==2))
#else
        sizeof(T)==4
#endif
        ;
    static constexpr bool NATIVE_FOUR_ROW_SINGLE=OPTIMIZE_FOUR_ROW && ROWS==4 && PAIR_STRIDE==1 && SINGLE_SEGMENT && sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool NATIVE_HALF_PACKET=FIXED_SEGMENTS==2 && ROWS==1 && NATIVE_ONE_ROW && sizeof(T)==2;
    static constexpr bool COMPLETE_ENDPOINT_LOAD=NATIVE_FOUR_ROW_SINGLE ||
        (FIXED_SEGMENTS==2 && ROWS==4 && !CONTIG_SEGMENTS && sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
         && sizeof(DTYPE_TREE_POINTS)==4
#endif
        );
    static constexpr bool MAPPED_TINY_STARTS=NATIVE_ONE_ROW;
public:
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                               const TreeLineMinDistanceTilingData& t) {
        t_ = t; rows_ = ROWS; pairs_ = ROWS==1?8:ROWS*PAIR_STRIDE; count_ = ROWS==4 && PAIR_STRIDE==1?32:64;
        pg_.SetGlobalBuffer((__gm__ T*)p); sg_.SetGlobalBuffer((__gm__ T*)s);
        dg_.SetGlobalBuffer((__gm__ T*)d); ig_.SetGlobalBuffer((__gm__ int32_t*)i);
        uint32_t address = 0;
        rawP_ = Alloc<T>(address, 256 / sizeof(T));
        rawS_ = Alloc<T>(address, 512 / sizeof(T));
        if constexpr (sizeof(T) == 2) {
            fp_ = Alloc<float>(address, 128); fs_ = Alloc<float>(address, 256);
        } else { fp_ = rawP_.template ReinterpretCast<float>(); fs_ = rawS_.template ReinterpretCast<float>(); }
        p_ = Alloc<float>(address, count_); a_ = Alloc<float>(address, count_); c_ = Alloc<float>(address, count_);
        work_ = Alloc<float>(address, count_ * 8);
        if constexpr(MAPPED_TINY_STARTS) originalA_=a_;
        pointOffsets_ = Alloc<int32_t>(address, count_);
        segmentOffsets_ = Alloc<int32_t>(address, count_);
        lengths_ = Alloc<float>(address, 64); projection_ = Alloc<float>(address, 64);
        scores_ = Alloc<float>(address, 64); records_ = Alloc<float>(address, 64);
        out_ = Alloc<float>(address, 64); outT_ = Alloc<T>(address, 64);
        indices_ = Alloc<int32_t>(address, 64);
        masks_ = Alloc<uint8_t>(address, 512);
        zeroGeometry_=Alloc<float>(address,9*64);
        zeroExact_=Alloc<float>(address,45*64);
        if constexpr(!SINGLE_SEGMENT && !NATIVE_ONE_ROW) {
            const uint32_t segmentCount=FIXED_SEGMENTS?FIXED_SEGMENTS:t.segments;
            const uint32_t bits=(SINGLE_SEGMENT?1U:((1U<<segmentCount)-1U))*(ROWS==1?0x01010101U:0x55555555U);
            Duplicate(masks_[384].template ReinterpretCast<uint32_t>(),bits,2);
        }
        if constexpr(SINGLE_SEGMENT && sizeof(T)==2) {
            Duplicate(masks_[384].template ReinterpretCast<uint32_t>(),uint32_t((1U<<ROWS)-1U),2);
        }
        if constexpr(ROWS!=1 && !SINGLE_SEGMENT)Duplicate(segmentOffsets_,int32_t(1),64);
    }
    __aicore__ inline void Process() {
        const uint32_t pointCount=STATIC_ONE_POINT || (ROWS==1 && SINGLE_SEGMENT)?1U:t_.points;
        const uint32_t segmentCount=SINGLE_SEGMENT?1U:FIXED_SEGMENTS?FIXED_SEGMENTS:t_.segments;
        uint32_t batch,pointPosition,validRows;
        if constexpr(ROWS==1) {
            // One-row tiles own consecutive flattened point/output positions.
            batch=pointCount==1?GetBlockIdx():GetBlockIdx()/pointCount;
            pointPosition=GetBlockIdx();
            validRows=1;
        } else {
            const uint32_t tiles=(pointCount+rows_-1)/rows_;
            batch=GetBlockIdx()/tiles;
            const uint32_t start=GetBlockIdx()%tiles*rows_;
            pointPosition=batch*pointCount+start;
            validRows=pointCount-start<rows_?pointCount-start:rows_;
        }
        // The native single-segment load defines both padded endpoint
        // records; no preceding vector write is needed for this domain.
        if constexpr(ROWS!=1 && !COMPLETE_ENDPOINT_LOAD)Duplicate(rawS_,T(0),CONTIG_SEGMENTS?64:512/sizeof(T));
        if constexpr(ROWS!=1 && !COMPLETE_ENDPOINT_LOAD) {SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);}
        DataCopyPad(rawP_,pg_[pointPosition*3],DataCopyExtParams{static_cast<uint16_t>(validRows),3U*sizeof(T),0,0,0},
                    DataCopyPadExtParams<T>{false,0,0,T(0)});
        if constexpr(CONTIG_SEGMENTS) {
            DataCopyPad(rawS_,sg_[batch*segmentCount*6],DataCopyExtParams{1,segmentCount*6U*sizeof(T),0,0,0},
                        DataCopyPadExtParams<T>{false,0,0,T(0)});
            if constexpr(DEFER_SOC_INIT) InitSocState();
            // i = 8*segment+coordinate; its packed source offset is 4*i-8*segment.
            if constexpr(BLOCK_INDICES) {
                if constexpr(FIXED_SEGMENTS==2 && ROWS==1 && NATIVE_ONE_ROW && sizeof(T)==4) {
                    // Construct byte offsets directly with the original vector seed.
                    MakeIndices<4>(pointOffsets_,8);PipeBarrier<PIPE_V>();
                } else {
                MakeIndices(pointOffsets_,8);PipeBarrier<PIPE_V>();
                Muls(pointOffsets_,pointOffsets_,int32_t(4),8);PipeBarrier<PIPE_V>();
                }
                Adds(pointOffsets_[8],pointOffsets_,int32_t(24),8);PipeBarrier<PIPE_V>();
                if(segmentCount>2) {Adds(pointOffsets_[16],pointOffsets_,int32_t(48),16);PipeBarrier<PIPE_V>();}
                if(segmentCount>4) {Adds(pointOffsets_[32],pointOffsets_,int32_t(96),32);PipeBarrier<PIPE_V>();}
            } else {
                MakeIndices(pointOffsets_,64);PipeBarrier<PIPE_V>();
                ShiftRight(segmentOffsets_,pointOffsets_,int32_t(3),64);
                Muls(pointOffsets_,pointOffsets_,int32_t(4),64);PipeBarrier<PIPE_V>();
                Muls(segmentOffsets_,segmentOffsets_,int32_t(-8),64);PipeBarrier<PIPE_V>();
                Add(segmentOffsets_,segmentOffsets_,pointOffsets_,64);PipeBarrier<PIPE_V>();
            }
        } else {
            DataCopyPad(rawS_,sg_[batch*segmentCount*6],DataCopyExtParams{static_cast<uint16_t>(2*segmentCount),3U*sizeof(T),0,0,0},
                        DataCopyPadExtParams<T>{false,0,0,T(0)});
            if constexpr(DEFER_SOC_INIT) InitSocState();
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if constexpr(NATIVE_HALF_PACKET) {
            auto absoluteInputs=zeroGeometry_.template ReinterpretCast<T>();
            SetMaskNorm();SetVectorMask<T,MaskMode::NORMAL>(uint64_t(16));
            Abs<T,false>(absoluteInputs,rawP_,uint64_t(128),2,UnaryRepeatParams(1,1,1,8));
        }
        if constexpr (sizeof(T)==2) {
            if constexpr(NATIVE_HALF_PACKET) {
                // Repeat strides address the two independent padded DMA records.
                const LocalTensor<float> convertedInputs(TPosition::VECCALC,uint32_t(fp_.GetPhyAddr()),256);
                const LocalTensor<T> rawInputs(TPosition::VECCALC,uint32_t(rawP_.GetPhyAddr()),256);
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Cast<float,T,false>(convertedInputs,rawInputs,RoundMode::CAST_NONE,uint64_t(64),2,UnaryRepeatParams(1,1,16,8));
            } else {
                Cast(fp_,rawP_,RoundMode::CAST_NONE,validRows*slot_);
                Cast(fs_,rawS_,RoundMode::CAST_NONE,CONTIG_SEGMENTS?((6*segmentCount+7)/8*8):(2*segmentCount*slot_));
            }
            PipeBarrier<PIPE_V>();
        }
        if constexpr(NATIVE_HALF_PACKET) {
            auto absoluteInputs=zeroGeometry_.template ReinterpretCast<T>();
            auto belowMaximum=zeroGeometry_[72].template ReinterpretCast<uint8_t>();
            SetMaskNorm();SetVectorMask<T,MaskMode::NORMAL>(uint64_t(32));
            CompareScalar<T,uint8_t,false>(belowMaximum,absoluteInputs,T(2.0F),CMPMODE::LT,uint64_t(128),1,UnaryRepeatParams(1,1,8,8));
        }
        if constexpr(NATIVE_SINGLE_POINT) {
            // Padded native DMA records already form complete coordinate vectors.
            p_=fp_;a_=fs_;c_=fs_[slot_];
        } else {
        Copy(p_,fp_,THIN_SINGLE_FP32?8:PAIR_STRIDE*8,rows_,CopyRepeatParams(1,0,PAIR_STRIDE,slot_/8));
        if constexpr(CONTIG_SEGMENTS) {
            auto sourceOffsets=BLOCK_INDICES?pointOffsets_:segmentOffsets_;
            const uint32_t coordinateCount=BLOCK_INDICES?(segmentCount<=2?16:segmentCount<=4?32:64):64;
            Gather(a_,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),0U,coordinateCount);
            Gather(c_,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),12U,coordinateCount);
        } else if constexpr(NATIVE_FOUR_ROW_SINGLE) {
            const LocalTensor<float> endpointPairs(TPosition::VECCALC,uint32_t(a_.GetPhyAddr()),64);
            Copy(endpointPairs,fs_,uint64_t(32),2,CopyRepeatParams(1,0,4,1));
        } else {
            Copy(a_,fs_,THIN_SINGLE_FP32?8:PAIR_STRIDE*8,rows_,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
            Copy(c_,fs_[slot_],THIN_SINGLE_FP32?8:PAIR_STRIDE*8,rows_,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
        }
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_ONE_ROW || NATIVE_FOUR_ROW_SINGLE ||
                     (COMPLETE_ENDPOINT_LOAD && FIXED_SEGMENTS==2)) {
            // Axis alignment is a geometric property of the loaded endpoints.
            // The exact lattice/bound checks below guard the direct AABB result.
            const uint32_t pairs=ROWS==1?segmentCount:validRows*PAIR_STRIDE;
            const uint32_t active=(1U<<pairs)-1U;
            auto axisFlags=zeroExact_.template ReinterpretCast<uint8_t>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs*8U));
            Compare<float,uint8_t,false>(axisFlags,a_,c_,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t fixed=axisFlags.template ReinterpretCast<uint64_t>().GetValue(0);
            const uint64_t axis=((fixed&(fixed>>1))|(fixed&(fixed>>2))|((fixed>>1)&(fixed>>2)))&0x0101010101010101ULL;
            const uint32_t aligned=uint32_t((axis*0x0102040810204080ULL)>>56);
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            bool directReady=false;
            if((aligned&active)==active) {
                if constexpr(NATIVE_FOUR_ROW_SINGLE) directReady=TryAxisSingle(validRows);
                else directReady=TryExactDyadicGeometry(validRows,segmentCount,true);
            }
            if(directReady) {
                Sqrt(out_,out_,rows_);
                if constexpr(sizeof(T)==2) {PipeBarrier<PIPE_V>();Cast(outT_,out_,RoundMode::CAST_RINT,rows_);}
                else outT_=out_;
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(dg_[pointPosition],outT_,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
                DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,validRows*4,0,0,0});
                return;
            }
        }
        if constexpr(NATIVE_HALF_PACKET) {
            // Coordinate range flags reuse the original input-copy barrier.
        }
        if constexpr(NATIVE_HALF_PACKET) {
            auto rangeFlags=zeroGeometry_.template ReinterpretCast<uint32_t>();
            const uint32_t accepted=rangeFlags.GetValue(72);
            if((accepted&0x0fff0007U)==0x0fff0007U) {
                // Bounded Half coordinates keep endpoint dot and norm errors below the guard.
                const BinaryRepeatParams same{1,1,1,8,8,8};
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Sub<float,false>(W(0),c_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(2),p_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(3),p_,c_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                const LocalTensor<float> endpointPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),128);
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(zeroGeometry_[128].GetPhyAddr()),128);
                Mul<float,false>(products,endpointPairs,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,8,8,0));PipeBarrier<PIPE_V>();
                WholeReduceSum(zeroGeometry_[256],zeroGeometry_[128],3,2,1,1,1);
                WholeReduceSum(zeroGeometry_[320],zeroGeometry_[192],3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(masks_,zeroGeometry_[256],-0.001953125F,CMPMODE::LT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
                CompareScalar<float,uint8_t,false>(masks_[64],zeroGeometry_[320],0.001953125F,CMPMODE::GT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const uint32_t atStart=masks_.template ReinterpretCast<uint32_t>().GetValue(0);
                const uint32_t atEnd=masks_[64].template ReinterpretCast<uint32_t>().GetValue(0);
                if(((atStart|atEnd)&3U)==3U) {
                    // Only reduced predicates choose addresses; every residual is vector-computed.
                    const auto residual0=(atStart&1U)?W(2):W(3);
                    const auto residual1=(atStart&2U)?W(2)[8]:W(3)[8];
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Duplicate<float,false>(scores_,__builtin_inff(),uint64_t(64),1,1,8);
                    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
                    Mul<float,false>(zeroGeometry_[128],residual0,residual0,uint64_t(64),1,same);
                    Mul<float,false>(zeroGeometry_[136],residual1,residual1,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                    WholeReduceSum(scores_,zeroGeometry_[128],3,2,1,1,1);PipeBarrier<PIPE_V>();
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    ReduceTiny();
                    // Bounded endpoint arithmetic still needs an index certificate.
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const auto scoreBits=scores_.template ReinterpretCast<uint32_t>();
                    const uint32_t score0=scoreBits.GetValue(0),score1=scoreBits.GetValue(1);
                    // Nonnegative finite endpoint norms differ by far more than
                    // the bounded binary32 accumulation error at 128 ULPs.
                    const uint32_t separation=score0>score1?score0-score1:score1-score0;
                    if(separation<=128U && !TryExactDyadicGeometry(validRows,segmentCount)) {
                        ExactFallback(pointPosition,batch,validRows,segmentCount);return;
                    }
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Sqrt(out_,out_,rows_);PipeBarrier<PIPE_V>();
                    Cast(outT_,out_,RoundMode::CAST_RINT,rows_);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyPad(dg_[pointPosition],outT_,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
                    DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,validRows*4,0,0,0});
                    // No subsequent UB reuse or GM overwrite follows this terminal write.
                    return;
                }
            }
        }
        if constexpr(NATIVE_ONE_ROW && STATIC_ONE_POINT && FIXED_SEGMENTS==2 && sizeof(T)==4) {
            // Certify bounded differences and residuals after vector geometry.
            auto bounds=zeroExact_.template ReinterpretCast<uint8_t>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetMaskNorm();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Sub<float,false>(W(0),c_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(2),p_,a_,uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),192);
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(zeroGeometry_[128].GetPhyAddr()),128);
                Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,8,16,0));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(zeroGeometry_[256],zeroGeometry_[128],3,2,1,1,1);
                WholeReduceSum<float,false>(lengths_,zeroGeometry_[192],3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(bounds[64],lengths_,0.000244140625F,CMPMODE::GE,uint64_t(64),1,unary);
                CompareScalar<float,uint8_t,false>(bounds[320],lengths_,64.0F,CMPMODE::LE,uint64_t(64),1,unary);
                Maxs<float,false>(lengths_,lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Div<float,false>(projection_,zeroGeometry_[256],lengths_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Muls<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(zeroGeometry_[256],projection_,lengths_,uint64_t(64),1,same);
                Muls<float,false>(zeroGeometry_[384],lengths_,0.00000095367431640625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Abs<float,false>(zeroGeometry_[256],zeroGeometry_[256],uint64_t(64),1,unary);
                Maxs<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Compare<float,uint8_t,false>(bounds[128],zeroGeometry_[256],zeroGeometry_[384],CMPMODE::LE,uint64_t(64),1,same);
                Mins<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(scores_,W(2),3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(bounds[384],scores_,64.0F,CMPMODE::LE,uint64_t(64),1,unary);
                WholeReduceMin<float,false>(records_,scores_,2,1,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                out_=records_;
                GatherMask(indices_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                WholeReduceMax<float,false>(zeroGeometry_[384],scores_,2,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
                Sub<float,false>(zeroGeometry_[384],zeroGeometry_[384],out_,uint64_t(64),1,same);
                CompareScalar<float,uint8_t,false>(bounds[192],out_,0.015625F,CMPMODE::GE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                CompareScalar<float,uint8_t,false>(bounds[256],zeroGeometry_[384],0.00390625F,CMPMODE::GT,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                auto flags=bounds.template ReinterpretCast<uint32_t>();
                if((((flags.GetValue(16)&flags.GetValue(32)&flags.GetValue(80)&flags.GetValue(96))&3U)==3U &&
                   ((flags.GetValue(48)&flags.GetValue(64))&1U)==1U) ||
                   TryExactDyadicGeometry(validRows,segmentCount)) {
                    Sqrt<float,false>(out_,out_,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyPad(dg_[pointPosition],out_,DataCopyExtParams{1,4,0,0,0});
                    DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,4,0,0,0});
                    // No subsequent UB reuse or GM overwrite follows this terminal write.
                    return;
                }
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        if constexpr(NATIVE_ONE_ROW && !STATIC_ONE_POINT && FIXED_SEGMENTS==2 && sizeof(T)==4) {
            // Certify bounded differences and residuals after vector geometry.
            auto bounds=zeroExact_.template ReinterpretCast<uint8_t>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetMaskNorm();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Sub<float,false>(W(0),c_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(2),p_,a_,uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),192);
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(zeroGeometry_[128].GetPhyAddr()),128);
                Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,8,16,0));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(zeroGeometry_[256],zeroGeometry_[128],3,2,1,1,1);
                WholeReduceSum<float,false>(lengths_,zeroGeometry_[192],3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(bounds[64],lengths_,0.000244140625F,CMPMODE::GE,uint64_t(64),1,unary);
                CompareScalar<float,uint8_t,false>(bounds[320],lengths_,256.0F,CMPMODE::LE,uint64_t(64),1,unary);
                Maxs<float,false>(lengths_,lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Div<float,false>(projection_,zeroGeometry_[256],lengths_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Muls<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(zeroGeometry_[256],projection_,lengths_,uint64_t(64),1,same);
                Muls<float,false>(zeroGeometry_[384],lengths_,0.0000002384185791015625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Abs<float,false>(zeroGeometry_[256],zeroGeometry_[256],uint64_t(64),1,unary);
                Maxs<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Compare<float,uint8_t,false>(bounds[128],zeroGeometry_[256],zeroGeometry_[384],CMPMODE::LE,uint64_t(64),1,same);
                Mins<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(scores_,W(2),3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(bounds[384],scores_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);
                WholeReduceMin<float,false>(records_,scores_,2,1,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                out_=records_;
                GatherMask(indices_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                WholeReduceMax<float,false>(zeroGeometry_[384],scores_,2,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
                Sub<float,false>(zeroGeometry_[384],zeroGeometry_[384],out_,uint64_t(64),1,same);
                CompareScalar<float,uint8_t,false>(bounds[192],out_,0.25F,CMPMODE::GE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                CompareScalar<float,uint8_t,false>(bounds[256],zeroGeometry_[384],0.03125F,CMPMODE::GT,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                auto flags=bounds.template ReinterpretCast<uint32_t>();
                if((((flags.GetValue(16)&flags.GetValue(32)&flags.GetValue(80)&flags.GetValue(96))&3U)==3U &&
                   ((flags.GetValue(48)&flags.GetValue(64))&1U)==1U) ||
                   TryExactDyadicGeometry(validRows,segmentCount)) {
                    Sqrt<float,false>(out_,out_,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyPad(dg_[pointPosition],out_,DataCopyExtParams{1,4,0,0,0});
                    DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,4,0,0,0});
                    // No subsequent UB reuse or GM overwrite follows this terminal write.
                    return;
                }
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        // The bounded certificate below permits ordinary FP32 differences;
        // difficult exponents, cancellation and ties use exact vector packets.
        if constexpr(THIN_SINGLE_FP32)count_=8;
        else if constexpr(ROWS==1 && sizeof(T)==2 && !NATIVE_ONE_ROW)count_=SINGLE_SEGMENT?8:(segmentCount<=2?16:segmentCount<=4?32:64);
        else if constexpr(NATIVE_HALF_PACKET)count_=16;
        else if constexpr(ROWS>1 && !SINGLE_SEGMENT)count_=validRows*PAIR_STRIDE*8;
        Sub(W(0),c_,a_,count_);
        Sub(W(2),p_,a_,count_);
        PipeBarrier<PIPE_V>();
        if constexpr(THIN_SINGLE_FP32) {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
            if constexpr(NATIVE_SINGLE_POINT) {
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(W(4).GetPhyAddr()),16);
                const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),24);
                Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,1,2,0));
            } else {
                Mul<float,false>(W(4),W(0),W(0),uint64_t(64),1,same);
                Mul<float,false>(W(5),W(2),W(0),uint64_t(64),1,same);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(NATIVE_SINGLE_POINT?5:4),3,1,1,1,1);
            WholeReduceSum(projection_,W(NATIVE_SINGLE_POINT?4:5),3,1,1,1,1);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
            // The certificate tests the original norm; clamping only the
            // divisor cannot admit an underflowed or degenerate segment.
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
            Muls<float,false>(W(6),W(6),-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,1,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else if constexpr(NATIVE_ONE_ROW) {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(count_));
            const LocalTensor<float> productPairs(TPosition::VECCALC,uint32_t(originalA_.GetPhyAddr()),512);
            const LocalTensor<float> pointDirectionPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),192);
            Mul<float,false>(productPairs,pointDirectionPairs,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,56,16,0));
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Duplicate<float,false>(lengths_,1.0F,uint64_t(64),1,1,8);
            Duplicate<float,false>(projection_,0.0F,uint64_t(64),1,1,8);
            if constexpr(!SINGLE_SEGMENT)Duplicate<float,false>(scores_,__builtin_inff(),uint64_t(64),1,1,8);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(5),3,SINGLE_SEGMENT?1:segmentCount,1,1,1);
            WholeReduceSum(projection_,W(4),3,SINGLE_SEGMENT?1:segmentCount,1,1,1);PipeBarrier<PIPE_V>();
            // Keep the original norm for the numerical certificate. Only the
            // divisor is clamped; small/degenerate norms retain the exact fallback.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(count_));
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(count_));
            if constexpr(!SINGLE_SEGMENT)CompareScalar(masks_,W(6),1.0F,CMPMODE::LT,64);
            Muls<float,false>(W(6),W(6),-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            const LocalTensor<float> residualPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),128);
            const LocalTensor<float> directionPairs(TPosition::VECCALC,uint32_t(W(0).GetPhyAddr()),128);
            MulAddDst<float,float,false>(residualPairs,W(6),directionPairs,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,0,8));PipeBarrier<PIPE_V>();
            if constexpr(!SINGLE_SEGMENT)Sub<float,false>(W(4),p_,c_,uint64_t(64),1,same);
            PipeBarrier<PIPE_V>();
            if constexpr(!SINGLE_SEGMENT) {
                SetCmpMask(masks_);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(2),W(2),W(4),1,same);PipeBarrier<PIPE_V>();
            }
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,SINGLE_SEGMENT?1:segmentCount,1,1,1);PipeBarrier<PIPE_V>();
        } else if constexpr(NATIVE_FOUR_ROW_SINGLE) {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            // One repeat forms each independent set of dot/norm products.
            const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),96);
            const LocalTensor<float> products(TPosition::VECCALC,uint32_t(W(4).GetPhyAddr()),64);
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
            Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,4,8,0));
            Duplicate(projection_,0.0F,8);PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(5),3,4,1,1,1);
            WholeReduceSum(projection_,W(4),3,4,1,1,1);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(4));
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Muls<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
            MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,4,1,1,1);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else {
        Mul(W(4),W(0),W(0),count_); Mul(W(5),W(2),W(0),count_);
            Duplicate(lengths_,1.0F,64); Duplicate(projection_,0.0F,64);
            if constexpr(!NATIVE_FOUR_ROW_SINGLE)Duplicate(scores_,__builtin_inff(),64);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(4),3,ROWS==1?(SINGLE_SEGMENT?1:segmentCount):pairs_,1,1,1);
            WholeReduceSum(projection_,W(5),3,ROWS==1?(SINGLE_SEGMENT?1:segmentCount):pairs_,1,1,1);
            PipeBarrier<PIPE_V>();
            CompareScalar(masks_,lengths_,0.0F,CMPMODE::GT,64);
            PipeBarrier<PIPE_V>();
            Select(lengths_,masks_,lengths_,1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
            PipeBarrier<PIPE_V>();
            Div(projection_,projection_,lengths_,64);
            PipeBarrier<PIPE_V>();
            Maxs(projection_,projection_,0.0F,64);
            PipeBarrier<PIPE_V>();
            Mins(projection_,projection_,1.0F,64);
            PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,(pairs_+7)/8,BrcbRepeatParams(1,8));
            PipeBarrier<PIPE_V>();
            if(!SINGLE_SEGMENT && segmentCount>1) CompareScalar(masks_,W(6),1.0F,CMPMODE::LT,64);
            Muls(W(6),W(6),-1.0F,count_);
            PipeBarrier<PIPE_V>();
            if constexpr(NATIVE_FOUR_ROW_SINGLE) {
                const LocalTensor<float> residualPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),64);
                const LocalTensor<float> directionPairs(TPosition::VECCALC,uint32_t(W(0).GetPhyAddr()),64);
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
                MulAddDst<float,float,false>(residualPairs,W(6),directionPairs,uint64_t(64),1,BinaryRepeatParams(1,1,1,4,0,4));
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            } else {
                MulAddDst(W(2),W(6),W(0),count_);
            }
            PipeBarrier<PIPE_V>();
            if(!SINGLE_SEGMENT && segmentCount>1) Sub(W(4),p_,c_,count_);
            PipeBarrier<PIPE_V>();
            if(!SINGLE_SEGMENT && segmentCount>1) Select(W(2),masks_,W(2),W(4),SELMODE::VSEL_CMPMASK_SPR,count_);
            PipeBarrier<PIPE_V>();
            Mul(W(2),W(2),W(2),count_);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,ROWS==1?(SINGLE_SEGMENT?1:segmentCount):pairs_,1,1,1);
            PipeBarrier<PIPE_V>();
            if constexpr(ROWS!=1 && !SINGLE_SEGMENT)Select(scores_,masks_[384],scores_,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
            PipeBarrier<PIPE_V>();
        }
        ReduceTiny();
        if(!CertifiedTiny(validRows,segmentCount) && !TryExactDyadicGeometry(validRows,segmentCount)) {
            ExactFallback(pointPosition,batch,validRows,segmentCount);return;
        }
        if constexpr(NATIVE_ONE_ROW && sizeof(T)==4 && STATIC_ONE_POINT && FIXED_SEGMENTS==2) {
            // Static single-point packets consume exactly output lane0.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
            Sqrt<float,false>(out_,out_,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            
            
        } else Sqrt(out_,out_,rows_);
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T)==2) Cast(outT_,out_,RoundMode::CAST_RINT,rows_);
        else outT_=out_;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dg_[pointPosition],outT_,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
        DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,validRows*4,0,0,0});
        // No subsequent UB reuse or GM overwrite follows this terminal write.
    }
private:
    __aicore__ inline void DyadicLatticeFlags(const LocalTensor<float>& left,
            const LocalTensor<float>& right,uint32_t coordinates,bool packedInputs=false) {
        auto scaled=zeroExact_[1280],rounded=zeroExact_[1472],restored=zeroExact_[1664];
        auto flags=zeroExact_[880].template ReinterpretCast<uint8_t>();
        if(packedInputs && FIXED_SEGMENTS==2) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Muls<float,false>(scaled,p_,16.0F,uint64_t(64),3,UnaryRepeatParams(1,1,8,8));
        } else {
            Muls(scaled,p_,16.0F,coordinates);
            Muls(scaled[64],left,16.0F,coordinates);
            Muls(scaled[128],right,16.0F,coordinates);
        }
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        const UnaryRepeatParams blocks{1,1,8,8};
        Cast<int32_t,float,false>(rounded.template ReinterpretCast<int32_t>(),scaled,RoundMode::CAST_RINT,uint64_t(64),3,blocks);PipeBarrier<PIPE_V>();
        Cast<float,int32_t,false>(restored,rounded.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,uint64_t(64),3,blocks);PipeBarrier<PIPE_V>();
        Muls<float,false>(restored,restored,0.0625F,uint64_t(64),3,blocks);PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Compare<float,uint8_t,false>(flags,p_,restored,CMPMODE::EQ,uint64_t(64),1,same);
        Compare<float,uint8_t,false>(flags[64],left,restored[64],CMPMODE::EQ,uint64_t(64),1,same);
        Compare<float,uint8_t,false>(flags[128],right,restored[128],CMPMODE::EQ,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool TryAxisSingle(uint32_t validRows) {
        // A single axis-aligned segment has no argmin ambiguity. AABB clipping
        // avoids projection cancellation, including zero-length/zero-distance.
        // Finite residuals bounded by 2^30 keep squares and sums far from overflow;
        // subnormal residual loss is far below the contract's absolute tolerance.
        const uint32_t coordinates=validRows*8U;
        auto clipped=zeroExact_[704],upper=zeroExact_[768],residual=zeroExact_[192];
        auto flags=zeroExact_[832].template ReinterpretCast<uint8_t>();
        Min(clipped,a_,c_,coordinates);Max(upper,a_,c_,coordinates);PipeBarrier<PIPE_V>();
        Max(clipped,p_,clipped,coordinates);PipeBarrier<PIPE_V>();
        Min(clipped,clipped,upper,coordinates);PipeBarrier<PIPE_V>();
        Sub(residual,p_,clipped,coordinates);PipeBarrier<PIPE_V>();
        Abs(upper,residual,coordinates);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        CompareScalar<float,uint8_t,false>(flags,upper,1073741824.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t live=((uint64_t(1)<<coordinates)-1)&0x0707070707070707ULL;
        if((flags.template ReinterpretCast<uint64_t>().GetValue(0)&live)!=live)return false;
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Mul(residual,residual,residual,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(out_,residual,3,validRows,1,1,1);
        Duplicate(indices_,int32_t(0),8);PipeBarrier<PIPE_V>();
        return true;
    }
    __aicore__ inline bool TryExactDyadicGeometry(uint32_t validRows,uint32_t segments,bool inputReady=false) {
        // Matched integer reconstruction certifies a 2^-4 coordinate lattice.
        // A difference bounded by 64 is exact: small operands fit binary32,
        // and larger nearby same-sign operands satisfy Sterbenz's lemma.
        // Relative/direction bounds make dots and squared norms exact (22 bits).
        // Endpoint projections and axis-aligned segments therefore admit an
        // exact argmin, including zero length, zero distance and true ties.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const uint32_t pairs=ROWS==1?segments:validRows*PAIR_STRIDE;
        const uint32_t coordinates=pairs*8;
        const uint64_t prefix=coordinates==64?~uint64_t(0):((uint64_t(1)<<coordinates)-1);
        const uint64_t liveCoordinates=prefix&((ROWS>1 && PAIR_STRIDE==2 && segments==1)?
            0x0007000700070007ULL:0x0707070707070707ULL);
        const uint32_t livePairs=((1U<<pairs)-1U)&
            ((ROWS>1 && PAIR_STRIDE==2 && segments==1)?0x55U:0xffU);
        auto left=zeroExact_,right=zeroExact_[64];
        auto delta=zeroExact_[128],relA=zeroExact_[192],relC=zeroExact_[256];
        auto temp=zeroExact_[320],rounded=zeroExact_[384],product=zeroExact_[448];
        auto norm=zeroExact_[512],dot=zeroExact_[576],endpoint=zeroExact_[640];
        auto box=zeroExact_[704],upper=zeroExact_[768];
        auto flags=zeroExact_[832].template ReinterpretCast<uint8_t>();
        if(inputReady) {left=a_;right=c_;}
        else {
        if constexpr(CONTIG_SEGMENTS) {
            auto sourceOffsets=BLOCK_INDICES?pointOffsets_:segmentOffsets_;
            Gather(left,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),0U,coordinates);
            Gather(right,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),12U,coordinates);
        } else {
            Copy(left,fs_,PAIR_STRIDE*8,validRows,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
            Copy(right,fs_[slot_],PAIR_STRIDE*8,validRows,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
        }
        PipeBarrier<PIPE_V>();
        }
        DyadicLatticeFlags(left,right,coordinates,inputReady && count_==64);
        Sub(delta,right,left,coordinates);Sub(relA,p_,left,coordinates);Sub(relC,p_,right,coordinates);PipeBarrier<PIPE_V>();
        if constexpr(FIXED_SEGMENTS==2) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Abs<float,false>(temp,delta,uint64_t(64),3,UnaryRepeatParams(1,1,8,8));
        } else {
            Abs(temp,delta,coordinates);Abs(rounded,relA,coordinates);Abs(product,relC,coordinates);
        }
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        CompareScalar<float,uint8_t,false>(flags[96],temp,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,false>(flags[128],rounded,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,false>(flags[160],product,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        Compare<float,uint8_t,false>(flags,left,right,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const auto words=flags.template ReinterpretCast<uint64_t>();
        const uint64_t admitted=words.GetValue(24)&words.GetValue(32)&words.GetValue(40);
        const uint64_t bounded=words.GetValue(12)&words.GetValue(16)&words.GetValue(20);
        if(((admitted&bounded)&liveCoordinates)!=liveCoordinates)return false;
        const uint64_t fixed=words.GetValue(0);
        const uint64_t axis=((fixed&(fixed>>1))|(fixed&(fixed>>2))|((fixed>>1)&(fixed>>2)))&0x0101010101010101ULL;
        const uint32_t axisPairs=inputReady?livePairs:uint32_t((axis*0x0102040810204080ULL)>>56);
        if((axisPairs&livePairs)==livePairs) {
            // An axis-aligned segment equals its AABB. The verified 2^-4
            // lattice and bounded differences make these squared norms exact.
            Min(box,left,right,coordinates);Max(product,left,right,coordinates);PipeBarrier<PIPE_V>();
            Max(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
            Min(box,box,product,coordinates);PipeBarrier<PIPE_V>();
            Sub(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
            Mul(box,box,box,coordinates);Duplicate(scores_,__builtin_inff(),64);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,box,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            if(livePairs!=((1U<<pairs)-1U)) {
                Duplicate(flags.template ReinterpretCast<uint32_t>(),livePairs,2);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Select(scores_,flags,scores_,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            ReduceTiny();return true;
        }
        Mul(product,delta,delta,coordinates);Mul(temp,relA,delta,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(norm,product,3,pairs,1,1,1);
        WholeReduceSum(dot,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
        CompareScalar<float,uint8_t,false>(flags[32],dot,0.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        Compare<float,uint8_t,false>(flags[64],dot,norm,CMPMODE::GE,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t endpoints=uint32_t(words.GetValue(4)|words.GetValue(8));
        if(((axisPairs|endpoints)&livePairs)!=livePairs) {
            // On this lattice every cross-product subtraction is exact. A
            // certified zero beats every nonzero distance without any division.
            auto u=zeroExact_[896],v=zeroExact_[960],cross=zeroExact_[1024],crossNorm=zeroExact_[1088];
            auto rotate1=zeroExact_[1152].template ReinterpretCast<int32_t>();
            auto rotate2=zeroExact_[1216].template ReinterpretCast<int32_t>();
            MakeIndices(rotate1,64);PipeBarrier<PIPE_V>();
            ShiftRight(rotate1,rotate1,int32_t(3),64);PipeBarrier<PIPE_V>();
            Muls(rotate1,rotate1,int32_t(32),64);PipeBarrier<PIPE_V>();
            Adds(rotate2,rotate1,int32_t(8),64);Adds(rotate1,rotate1,int32_t(4),64);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x0202020202020202ULL));
            Adds<int32_t,false>(rotate1,rotate1,int32_t(4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Adds<int32_t,false>(rotate2,rotate2,int32_t(-8),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x0404040404040404ULL));
            Adds<int32_t,false>(rotate1,rotate1,int32_t(-4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Adds<int32_t,false>(rotate2,rotate2,int32_t(-4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Gather(u,relA,rotate1.template ReinterpretCast<uint32_t>(),0U,coordinates);
            Gather(v,delta,rotate2.template ReinterpretCast<uint32_t>(),0U,coordinates);PipeBarrier<PIPE_V>();
            Mul(cross,u,v,coordinates);PipeBarrier<PIPE_V>();
            Gather(u,relA,rotate2.template ReinterpretCast<uint32_t>(),0U,coordinates);
            Gather(v,delta,rotate1.template ReinterpretCast<uint32_t>(),0U,coordinates);PipeBarrier<PIPE_V>();
            Mul(u,u,v,coordinates);PipeBarrier<PIPE_V>();
            Sub(cross,cross,u,coordinates);PipeBarrier<PIPE_V>();
            Mul(cross,cross,cross,coordinates);PipeBarrier<PIPE_V>();
            WholeReduceSum(crossNorm,cross,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            Mul(product,relA,relA,coordinates);PipeBarrier<PIPE_V>();
            WholeReduceSum(endpoint,product,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
            CompareScalar<float,uint8_t,false>(flags,crossNorm,0.0F,CMPMODE::EQ,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            CompareScalar<float,uint8_t,false>(flags[32],dot,0.0F,CMPMODE::GE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Compare<float,uint8_t,false>(flags[64],dot,norm,CMPMODE::LE,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            CompareScalar<float,uint8_t,false>(flags[96],norm,0.0F,CMPMODE::GT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            CompareScalar<float,uint8_t,false>(flags[128],endpoint,0.0F,CMPMODE::EQ,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t zeroPairs=uint32_t(words.GetValue(0)&words.GetValue(4)&words.GetValue(8)&
                (words.GetValue(12)|words.GetValue(16)))&livePairs;
            const uint32_t firstPairs=ROWS==1?1U:(PAIR_STRIDE==2?(livePairs&0x55U):livePairs);
            const bool allRowsZero=ROWS==1?zeroPairs!=0U:
                (PAIR_STRIDE==2?(((zeroPairs|(zeroPairs>>1))&firstPairs)==firstPairs):zeroPairs==livePairs);
            if(allRowsZero) {
                Duplicate(endpoint,0.0F,64);
                Duplicate(flags.template ReinterpretCast<uint32_t>(),zeroPairs,2);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                ReduceTiny();return true;
            }
            // Equal undirected segments have equal mathematical distances for
            // every point. Compare original endpoints before forcing index zero.
            Copy(u,left,uint64_t(8),pairs,CopyRepeatParams(1,1,1,0));
            Copy(v,right,uint64_t(8),pairs,CopyRepeatParams(1,1,1,0));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Compare<float,uint8_t,false>(flags,left,u,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[32],right,v,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[64],left,v,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[96],right,u,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t forward=words.GetValue(0)&words.GetValue(4),reverse=words.GetValue(8)&words.GetValue(12);
            const uint64_t identical=((forward&(forward>>1)&(forward>>2))|(reverse&(reverse>>1)&(reverse>>2)))&0x0101010101010101ULL;
            const uint32_t identicalPairs=uint32_t((identical*0x0102040810204080ULL)>>56);
            if((identicalPairs&livePairs)!=livePairs)return false;
            // Positive cross norms avoid projection cancellation; float
            // multiply/sum/divide contributes far less than the declared tolerance.
            Mul(temp,relC,relC,coordinates);PipeBarrier<PIPE_V>();
            WholeReduceSum(upper,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            Min(endpoint,endpoint,upper,pairs);Maxs(norm,norm,0.00390625F,pairs);PipeBarrier<PIPE_V>();
            Div(upper,crossNorm,norm,pairs);PipeBarrier<PIPE_V>();
            Duplicate(flags.template ReinterpretCast<uint32_t>(),endpoints,2);PipeBarrier<PIPE_V>();
            SetCmpMask(flags);SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(endpoint,endpoint,upper,1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
            Duplicate(flags.template ReinterpretCast<uint32_t>(),firstPairs,2);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            ReduceTiny();return true;
        }
        Mul(product,relA,relA,coordinates);Mul(temp,relC,relC,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(endpoint,product,3,pairs,1,1,1);
        WholeReduceSum(upper,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        Min(box,left,right,coordinates);Max(product,left,right,coordinates);PipeBarrier<PIPE_V>();
        Max(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
        Min(box,box,product,coordinates);Min(endpoint,endpoint,upper,pairs);PipeBarrier<PIPE_V>();
        Sub(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
        Mul(box,box,box,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(upper,box,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        Duplicate(flags.template ReinterpretCast<uint32_t>(),axisPairs,2);PipeBarrier<PIPE_V>();
        SetCmpMask(flags);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(endpoint,upper,endpoint,1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
        Duplicate(flags.template ReinterpretCast<uint32_t>(),livePairs,2);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        ReduceTiny();
        return true;
    }
    __aicore__ inline void ExactFallback(uint32_t position,uint32_t batch,uint32_t rows,uint32_t segments) {
        TreeTinyExactFallback<T>((GM_ADDR)pg_[position*3].GetPhyAddr(),
            (GM_ADDR)sg_[batch*segments*6].GetPhyAddr(),(GM_ADDR)dg_[position].GetPhyAddr(),
            (GM_ADDR)ig_[position].GetPhyAddr(),rows,segments);
    }
    __aicore__ inline bool CertifiedTiny(uint32_t validRows,uint32_t segments) {
        // In this bounded domain, the norm error is below 2^-13 and the
        // squared-score error below 2^-6. A gap > 2^-5 proves the winner;
        // the admitted residual range also leaves the specified FP32 tolerance margin.
        if constexpr(ROWS==1 && SINGLE_SEGMENT) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t norm=lengths_.template ReinterpretCast<uint32_t>().GetValue(0);
            const uint32_t score=scores_.template ReinterpretCast<uint32_t>().GetValue(0);
            // Ordered positive IEEE fields certify the same inclusive bounds.
            return norm>=0x39800000U && norm<=0x44800000U &&
                   score>=0x3f800000U && score<=0x44800000U;
        }
        auto flags=zeroGeometry_.template ReinterpretCast<uint8_t>();
        const uint32_t n=SINGLE_SEGMENT?validRows:ROWS==1?segments:validRows*PAIR_STRIDE;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(n));
        CompareScalar<float,uint8_t,false>(flags,lengths_,0.000244140625F,CMPMODE::GE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[32],lengths_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[64],scores_,ROWS>1 && !SINGLE_SEGMENT?0.25F:1.0F,CMPMODE::GE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[96],scores_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        if constexpr(!SINGLE_SEGMENT) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            if constexpr(ROWS==1) {
                Brcb(W(6),out_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            } else {
                Brcb(W(6),out_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                GatherMask(W(7),W(6),uint8_t(1),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
                GatherMask(W(6),W(7),uint8_t(1),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
            Adds<float,false>(W(6),W(6),0.03125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Compare<float,uint8_t,false>(flags[128],scores_,W(6),CMPMODE::LE,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto words=flags.template ReinterpretCast<uint32_t>();
        const uint32_t active=(1U<<n)-1U;
        const uint32_t safe=words.GetValue(0)&words.GetValue(8)&words.GetValue(16)&words.GetValue(24);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        if((safe&active)!=active)return false;
        if constexpr(!SINGLE_SEGMENT) {
            const uint32_t close=words.GetValue(32)&active;
            if constexpr(ROWS==1)return close && !(close&(close-1U));
            else return ((close^(close>>1))&0x5555U)==(active&0x5555U);
        }
        return true;
    }
    __aicore__ inline void ReduceTiny() {
        if constexpr(SINGLE_SEGMENT) {
            if constexpr(ROWS==1 || PAIR_STRIDE==1)out_=scores_;
            else GatherMask(out_,scores_,uint8_t(1),GatherMaskParams(1,1,8,0));
            Duplicate(indices_,int32_t(0),ROWS);PipeBarrier<PIPE_V>();
        } else if constexpr(ROWS==1) {
            WholeReduceMin(records_,scores_,8,rows_,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            out_=records_;
            GatherMask(indices_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));
            PipeBarrier<PIPE_V>();
        } else {
            Duplicate(out_,__builtin_inff(),64);
            Duplicate(indices_,int32_t(0),64);PipeBarrier<PIPE_V>();
            GatherMask(out_,scores_,uint8_t(1),GatherMaskParams(1,1,8,0));
            GatherMask(W(4),scores_,uint8_t(2),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            SetMaskNorm();
            Compare<float>(out_,W(4),CMPMODE::LE,uint64_t(64),same);PipeBarrier<PIPE_V>();
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(indices_.template ReinterpretCast<float>(),indices_.template ReinterpretCast<float>(),segmentOffsets_.template ReinterpretCast<float>(),1,same);
            Min(out_,out_,W(4),32);PipeBarrier<PIPE_V>();
        }
    }
    template<class U> __aicore__ inline LocalTensor<U> Alloc(uint32_t& address,uint32_t n) {
        uint32_t bytes=(n*sizeof(U)+31)/32*32;
        LocalTensor<U> v(TPosition::VECCALC,address,bytes/sizeof(U));address+=bytes;return v;
    }
    __aicore__ inline LocalTensor<float> W(uint32_t k) {
        if constexpr(NATIVE_HALF_PACKET) {
            if(k<4)return work_[(k^2U)*64U];
        } else if constexpr(NATIVE_ONE_ROW || NATIVE_FOUR_ROW_SINGLE || NATIVE_SINGLE_POINT) {
            if(k<4)return work_[(k^2U)*count_];
        }
        if constexpr(MAPPED_TINY_STARTS) { if(k==4)return originalA_; }
        return work_[k*(NATIVE_HALF_PACKET?64U:count_)];
    }
    template<int32_t STEP=1> __aicore__ inline void MakeIndices(const LocalTensor<int32_t>& v,uint32_t n) {
        Duplicate(v,int32_t(0),8);PipeBarrier<PIPE_V>();SetMaskNorm();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaa));
        Adds<int32_t,false>(v,v,int32_t(STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xcc));
        Adds<int32_t,false>(v,v,int32_t(2*STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xf0));
        Adds<int32_t,false>(v,v,int32_t(4*STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        for(uint32_t filled=8;filled<n;filled*=2){Adds(v[filled],v,int32_t(filled*STEP),filled);PipeBarrier<PIPE_V>();}
    }
    __aicore__ inline void Difference(const LocalTensor<float>& x,const LocalTensor<float>& y,
                                      const LocalTensor<float>& hi,const LocalTensor<float>& lo) {
        Sub(hi,x,y,count_);PipeBarrier<PIPE_V>();
        Sub(W(6),x,hi,count_);PipeBarrier<PIPE_V>();
        Add(W(7),hi,W(6),count_);Sub(lo,W(6),y,count_);PipeBarrier<PIPE_V>();
        Sub(W(7),x,W(7),count_);PipeBarrier<PIPE_V>();
        Add(lo,lo,W(7),count_);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Canonicalize() {
        auto less=masks_.template ReinterpretCast<uint16_t>();
        auto eq=masks_[64].template ReinterpretCast<uint16_t>();
        auto keep=masks_[128].template ReinterpretCast<uint16_t>();
        auto tmp=masks_[192].template ReinterpretCast<uint16_t>();
        auto ones=masks_[256].template ReinterpretCast<uint16_t>();
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(NATIVE_HALF_PACKET?16:64));
        Compare<float,uint8_t,false>(masks_,a_,c_,CMPMODE::LT,uint64_t(64),1,binary);
        Compare<float,uint8_t,false>(masks_[64],a_,c_,CMPMODE::EQ,uint64_t(64),1,binary);PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_HALF_PACKET) {
            // Float metadata bounds reuse the original canonical comparison barrier.
            // Two packed eight-lane records use only the first predicate word.
            // Merge predicate metadata after the vector comparisons complete.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t lt=less.GetValue(0),equal=eq.GetValue(0);
            const uint16_t packet=uint16_t((lt|((lt>>1)&equal)|((lt>>2)&equal&(equal>>1)))&0x0101U);
            keep.SetValue(0,uint16_t(packet*255U));
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        } else if constexpr(NATIVE_ONE_ROW && sizeof(T)==4) {
            // Eight coordinate records occupy one 64-bit predicate word.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if constexpr(FIXED_SEGMENTS==2) {
                // Only the two real coordinate records feed this bounded resource.
                const uint32_t lt=less.template ReinterpretCast<uint32_t>().GetValue(0);
                const uint32_t equal=eq.template ReinterpretCast<uint32_t>().GetValue(0);
                const uint32_t packet=(lt|((lt>>1)&equal)|((lt>>2)&equal&(equal>>1)))&0x0101U;
                keep.template ReinterpretCast<uint64_t>().SetValue(0,uint64_t(packet*255U));
            } else {
                const uint64_t lt=less.template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t equal=eq.template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t packet=(lt|((lt>>1)&equal)|((lt>>2)&equal&(equal>>1)))&0x0101010101010101ULL;
                keep.template ReinterpretCast<uint64_t>().SetValue(0,packet*255ULL);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        } else {
        // Packed predicate words share one mask throughout the lexicographic merge.
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(NATIVE_HALF_PACKET?1:4));
        Duplicate<uint16_t,false>(ones,uint16_t(0x0101),uint64_t(128),1,1,8);PipeBarrier<PIPE_V>();
        ShiftRight<uint16_t,false>(tmp,less,uint16_t(1),uint64_t(128),1,unary,false);PipeBarrier<PIPE_V>();
        And<uint16_t,false>(tmp,tmp,eq,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep,less,tmp,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        ShiftRight<uint16_t,false>(tmp,keep,uint16_t(1),uint64_t(128),1,unary,false);PipeBarrier<PIPE_V>();
        And<uint16_t,false>(tmp,tmp,eq,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep,less,tmp,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        And<uint16_t,false>(keep,keep,ones,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        Muls<int16_t,false>(keep.template ReinterpretCast<int16_t>(),keep.template ReinterpretCast<int16_t>(),int16_t(255),uint64_t(128),1,unary);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(NATIVE_HALF_PACKET?16:64));SetCmpMask(masks_[128]);
        if constexpr(MAPPED_TINY_STARTS)
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(work_[4*count_],a_,c_,1,binary);
        else Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(4),a_,c_,1,binary);
        PipeBarrier<PIPE_V>();
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(c_,c_,a_,1,binary);PipeBarrier<PIPE_V>();
        if constexpr(MAPPED_TINY_STARTS) {
            // Original A is dead after canonical B consumes it. Keep canonical
            // A in its selected plane and reuse old A for ordinary W4 scratch.
            // CorrectTinyZeroScores receives the explicit A address and only
            // uses work planes 0, 1 and 5, so it cannot overwrite canonical A.
            a_=work_[4*count_];
        } else {Copy(a_,W(4),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
    }
    static constexpr uint32_t slot_=32/sizeof(T);
    TreeLineMinDistanceTilingData t_;uint32_t rows_,pairs_,count_;
    GlobalTensor<T> pg_,sg_,dg_;GlobalTensor<int32_t> ig_;
    LocalTensor<T> rawP_,rawS_,outT_;
    LocalTensor<float> fp_,fs_,p_,a_,c_,work_,lengths_,projection_,scores_,records_,out_,zeroGeometry_,zeroExact_;
    LocalTensor<float> originalA_;
    LocalTensor<int32_t> pointOffsets_,segmentOffsets_,indices_;
    LocalTensor<uint8_t> masks_;
};

#define TREE_LINE_MIN_DISTANCE_DEVICE_BODY 1
#include "tree_line_min_distance_tiling.h"
#undef TREE_LINE_MIN_DISTANCE_DEVICE_BODY

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
constexpr uint32_t SORT_CAP=4096;
__aicore__ inline void SpatialIndices(LocalTensor<int32_t> values,uint32_t count) {
    Duplicate(values,int32_t(0),8);PipeBarrier<PIPE_V>();SetMaskNorm();
    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaa));
    Adds<int32_t,false>(values,values,int32_t(1),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xcc));
    Adds<int32_t,false>(values,values,int32_t(2),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xf0));
    Adds<int32_t,false>(values,values,int32_t(4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
    for(uint32_t filled=8;filled<count;filled*=2) {
        uint32_t n=count-filled<filled?count-filled:filled;
        Adds(values[filled],values,int32_t(filled),n);PipeBarrier<PIPE_V>();
    }
}
__aicore__ __attribute__((always_inline)) inline void SpatialSort(LocalTensor<float> pairs,LocalTensor<float> scratch,
    LocalTensor<float> key,LocalTensor<uint32_t> index,uint32_t count) {
    Sort32(pairs,key,index,count/32);PipeBarrier<PIPE_V>();
    bool inPairs=true;
    for(uint32_t length=32;length<count;length*=4) {
        auto src=inPairs?pairs:scratch;auto dst=inPairs?scratch:pairs;
        const uint32_t lists=count/length<4?count/length:4;
        const uint16_t sizes[4]={uint16_t(length),uint16_t(length),uint16_t(lists>2?length:0),uint16_t(lists>3?length:0)};
        for(uint32_t begin=0;begin<count;begin+=4*length) {
            MrgSortSrcList<float> sources(src[2*begin],src[2*(begin+length)],
                src[2*(begin+(lists>2?2:0)*length)],src[2*(begin+(lists>3?3:0)*length)]);
            MrgSort(dst[2*begin],sources,MrgSort4Info(sizes,false,lists==4?15:3,1));
        }
        PipeBarrier<PIPE_V>();inPairs=!inPairs;
    }
    if(!inPairs) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));Copy<float,false>(pairs,scratch,uint64_t(64),count/32,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
}
// Hilbert axes-to-transpose, using exact bounded integer vector arithmetic.
// Algorithm reference: Skilling; https://github.com/galtay/hilbertcurve .
__aicore__ __attribute__((always_inline)) inline void SpatialBitAndValue(
    const LocalTensor<int32_t>& dst,const LocalTensor<int32_t>& src,int32_t value,
    const LocalTensor<float>& constants,uint32_t count) {
    Duplicate(constants.ReinterpretCast<int32_t>(),value,64);PipeBarrier<PIPE_V>();
    SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
    And<uint16_t,false>(dst.ReinterpretCast<uint16_t>(),src.ReinterpretCast<uint16_t>(),constants.ReinterpretCast<uint16_t>(),uint64_t(128),count/64,BinaryRepeatParams(1,1,1,8,8,0));PipeBarrier<PIPE_V>();
}

// The 24-state Hilbert automaton is shared by both sorting strategies.
__aicore__ __attribute__((always_inline)) inline void InitHilbertConstants(
    const LocalTensor<uint64_t>& hilbertConstants) {
    hilbertConstants.SetValue(0,uint64_t(0x0000001100000008ULL));
    hilbertConstants.SetValue(1,uint64_t(0x000000020000001bULL));
    hilbertConstants.SetValue(2,uint64_t(0x0000002e00000027ULL));
    hilbertConstants.SetValue(3,uint64_t(0x0000000500000034ULL));
    hilbertConstants.SetValue(4,uint64_t(0x0000004700000038ULL));
    hilbertConstants.SetValue(5,uint64_t(0x0000005600000049ULL));
    hilbertConstants.SetValue(6,uint64_t(0x000000140000005bULL));
    hilbertConstants.SetValue(7,uint64_t(0x0000000d0000000aULL));
    hilbertConstants.SetValue(8,uint64_t(0x0000000100000030ULL));
    hilbertConstants.SetValue(9,uint64_t(0x0000006e00000067ULL));
    hilbertConstants.SetValue(10,uint64_t(0x0000001200000073ULL));
    hilbertConstants.SetValue(11,uint64_t(0x000000150000000cULL));
    hilbertConstants.SetValue(12,uint64_t(0x000000810000007eULL));
    hilbertConstants.SetValue(13,uint64_t(0x0000001a0000001dULL));
    hilbertConstants.SetValue(14,uint64_t(0x000000500000004fULL));
    hilbertConstants.SetValue(15,uint64_t(0x000000030000008cULL));
    hilbertConstants.SetValue(16,uint64_t(0x0000002b00000094ULL));
    hilbertConstants.SetValue(17,uint64_t(0x0000002200000025ULL));
    hilbertConstants.SetValue(18,uint64_t(0x000000800000007fULL));
    hilbertConstants.SetValue(19,uint64_t(0x000000510000004eULL));
    hilbertConstants.SetValue(20,uint64_t(0x0000002d0000009cULL));
    hilbertConstants.SetValue(21,uint64_t(0x0000002a00000023ULL));
    hilbertConstants.SetValue(22,uint64_t(0x000000060000001fULL));
    hilbertConstants.SetValue(23,uint64_t(0x00000069000000a0ULL));
    hilbertConstants.SetValue(24,uint64_t(0x0000005700000048ULL));
    hilbertConstants.SetValue(25,uint64_t(0x000000040000008bULL));
    hilbertConstants.SetValue(26,uint64_t(0x0000004600000039ULL));
    hilbertConstants.SetValue(27,uint64_t(0x0000003500000032ULL));
    hilbertConstants.SetValue(28,uint64_t(0x000000ab00000000ULL));
    hilbertConstants.SetValue(29,uint64_t(0x0000004c0000006fULL));
    hilbertConstants.SetValue(30,uint64_t(0x0000003a00000031ULL));
    hilbertConstants.SetValue(31,uint64_t(0x0000003d00000066ULL));
    hilbertConstants.SetValue(32,uint64_t(0x0000008f000000b4ULL));
    hilbertConstants.SetValue(33,uint64_t(0x000000b800000053ULL));
    hilbertConstants.SetValue(34,uint64_t(0x0000003600000045ULL));
    hilbertConstants.SetValue(35,uint64_t(0x0000006100000042ULL));
    hilbertConstants.SetValue(36,uint64_t(0x0000007b00000010ULL));
    hilbertConstants.SetValue(37,uint64_t(0x0000004a00000009ULL));
    hilbertConstants.SetValue(38,uint64_t(0x0000003c0000002fULL));
    hilbertConstants.SetValue(39,uint64_t(0x0000004d00000026ULL));
    hilbertConstants.SetValue(40,uint64_t(0x0000005f00000084ULL));
    hilbertConstants.SetValue(41,uint64_t(0x0000000e00000055ULL));
    hilbertConstants.SetValue(42,uint64_t(0x0000009000000043ULL));
    hilbertConstants.SetValue(43,uint64_t(0x0000002100000052ULL));
    hilbertConstants.SetValue(44,uint64_t(0x000000370000008eULL));
    hilbertConstants.SetValue(45,uint64_t(0x00000060000000b9ULL));
    hilbertConstants.SetValue(46,uint64_t(0x000000740000005dULL));
    hilbertConstants.SetValue(47,uint64_t(0x0000000b0000005aULL));
    hilbertConstants.SetValue(48,uint64_t(0x0000006b000000bcULL));
    hilbertConstants.SetValue(49,uint64_t(0x000000b0000000afULL));
    hilbertConstants.SetValue(50,uint64_t(0x0000006200000065ULL));
    hilbertConstants.SetValue(51,uint64_t(0x000000410000003eULL));
    hilbertConstants.SetValue(52,uint64_t(0x0000006d000000a4ULL));
    hilbertConstants.SetValue(53,uint64_t(0x0000001600000077ULL));
    hilbertConstants.SetValue(54,uint64_t(0x0000006a00000063ULL));
    hilbertConstants.SetValue(55,uint64_t(0x0000002900000098ULL));
    hilbertConstants.SetValue(56,uint64_t(0x000000b1000000aeULL));
    hilbertConstants.SetValue(57,uint64_t(0x000000400000003fULL));
    hilbertConstants.SetValue(58,uint64_t(0x0000007200000075ULL));
    hilbertConstants.SetValue(59,uint64_t(0x000000130000005cULL));
    hilbertConstants.SetValue(60,uint64_t(0x0000007d0000001eULL));
    hilbertConstants.SetValue(61,uint64_t(0x0000007a000000a1ULL));
    hilbertConstants.SetValue(62,uint64_t(0x000000ac00000007ULL));
    hilbertConstants.SetValue(63,uint64_t(0x0000004b00000068ULL));
    hilbertConstants.SetValue(64,uint64_t(0x0000001900000082ULL));
    hilbertConstants.SetValue(65,uint64_t(0x000000a600000085ULL));
    hilbertConstants.SetValue(66,uint64_t(0x00000088000000b3ULL));
    hilbertConstants.SetValue(67,uint64_t(0x000000bf00000054ULL));
    hilbertConstants.SetValue(68,uint64_t(0x0000000f0000005eULL));
    hilbertConstants.SetValue(69,uint64_t(0x0000001c0000008dULL));
    hilbertConstants.SetValue(70,uint64_t(0x0000002000000091ULL));
    hilbertConstants.SetValue(71,uint64_t(0x000000330000008aULL));
    hilbertConstants.SetValue(72,uint64_t(0x0000009b00000092ULL));
    hilbertConstants.SetValue(73,uint64_t(0x0000002400000095ULL));
    hilbertConstants.SetValue(74,uint64_t(0x0000001800000089ULL));
    hilbertConstants.SetValue(75,uint64_t(0x000000a7000000beULL));
    hilbertConstants.SetValue(76,uint64_t(0x0000009d0000009aULL));
    hilbertConstants.SetValue(77,uint64_t(0x0000002c00000093ULL));
    hilbertConstants.SetValue(78,uint64_t(0x000000b6000000a9ULL));
    hilbertConstants.SetValue(79,uint64_t(0x0000008700000078ULL));
    hilbertConstants.SetValue(80,uint64_t(0x000000a5000000a2ULL));
    hilbertConstants.SetValue(81,uint64_t(0x0000008600000079ULL));
    hilbertConstants.SetValue(82,uint64_t(0x0000006c000000bbULL));
    hilbertConstants.SetValue(83,uint64_t(0x000000b7000000a8ULL));
    hilbertConstants.SetValue(84,uint64_t(0x000000ad00000076ULL));
    hilbertConstants.SetValue(85,uint64_t(0x0000007c00000017ULL));
    hilbertConstants.SetValue(86,uint64_t(0x000000aa00000099ULL));
    hilbertConstants.SetValue(87,uint64_t(0x0000003b00000028ULL));
    hilbertConstants.SetValue(88,uint64_t(0x00000071000000b2ULL));
    hilbertConstants.SetValue(89,uint64_t(0x0000005800000083ULL));
    hilbertConstants.SetValue(90,uint64_t(0x0000009e000000b5ULL));
    hilbertConstants.SetValue(91,uint64_t(0x0000009700000044ULL));
    hilbertConstants.SetValue(92,uint64_t(0x000000a3000000baULL));
    hilbertConstants.SetValue(93,uint64_t(0x0000007000000059ULL));
    hilbertConstants.SetValue(94,uint64_t(0x00000064000000bdULL));
    hilbertConstants.SetValue(95,uint64_t(0x0000009f00000096ULL));
}
// Hilbert keys use complete64-word tiles; no point arithmetic is changed.
__aicore__ __attribute__((always_inline)) inline void SpatialKeyAndNormal(
    const LocalTensor<uint32_t>& dst,const LocalTensor<uint32_t>& src,uint32_t value,
    const LocalTensor<uint32_t>& constants,uint8_t repeats) {
    Duplicate<uint32_t,false>(constants,value,uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
    SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
    And<uint16_t,false>(dst.ReinterpretCast<uint16_t>(),src.ReinterpretCast<uint16_t>(),
        constants.ReinterpretCast<uint16_t>(),uint64_t(128),repeats,
        BinaryRepeatParams(1,1,1,8,8,0));PipeBarrier<PIPE_V>();
    SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
}
__aicore__ __attribute__((always_inline)) inline void SpatialHilbertNormalPackets(
    const LocalTensor<int32_t>& quantized,const LocalTensor<uint32_t>& state,
    const LocalTensor<uint32_t>& keyInteger,const LocalTensor<uint32_t>& digit,
    const LocalTensor<uint32_t>& ids,const LocalTensor<uint32_t>& mask,
    const LocalTensor<float>& hilbertTable,uint32_t count,uint32_t lowest) {
    const uint8_t repeats=uint8_t(count/64);
    const UnaryRepeatParams unary{1,1,8,8};
    const BinaryRepeatParams binary{1,1,1,8,8,8};
    SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
    Duplicate<uint32_t,false>(state,uint32_t(0),uint64_t(64),repeats,1,8);
    Duplicate<uint32_t,false>(keyInteger,uint32_t(0),uint64_t(64),repeats,1,8);PipeBarrier<PIPE_V>();
    for(uint32_t remaining=7;remaining>lowest;--remaining) {
        Duplicate<uint32_t,false>(mask,uint32_t(0),uint64_t(64),repeats,1,8);PipeBarrier<PIPE_V>();
        for(uint32_t axis=0;axis<3;++axis) {
            ShiftRight<uint32_t,false>(digit,quantized[axis*SORT_CAP].ReinterpretCast<uint32_t>(),remaining-1,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
            SpatialKeyAndNormal(digit,digit,1,ids,repeats);
            ShiftLeft<uint32_t,false>(digit,digit,2-axis,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
            Add<int32_t,false>(mask.ReinterpretCast<int32_t>(),mask.ReinterpretCast<int32_t>(),digit.ReinterpretCast<int32_t>(),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        }
        ShiftLeft<uint32_t,false>(ids,state,uint32_t(3),uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),mask.ReinterpretCast<int32_t>(),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        ShiftLeft<uint32_t,false>(ids,ids,uint32_t(2),uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Gather(mask.ReinterpretCast<float>(),hilbertTable,ids,uint32_t(0),count);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
        ShiftRight<uint32_t,false>(state,mask,uint32_t(3),uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        SpatialKeyAndNormal(digit,mask,7,ids,repeats);
        ShiftLeft<uint32_t,false>(keyInteger,keyInteger,uint32_t(3),uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(keyInteger.ReinterpretCast<int32_t>(),keyInteger.ReinterpretCast<int32_t>(),digit.ReinterpretCast<int32_t>(),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
    }
}
template<bool NORMAL_KEY_PACKETS=false,bool INPUT_SOA=false>
__aicore__ __attribute__((always_inline)) inline void TreeSpatialArrange(
    GM_ADDR input,GM_ADDR workspace,uint32_t valid,uint32_t count) {
    GlobalTensor<float> gx,gy;gx.SetGlobalBuffer((__gm__ float*)input);gy.SetGlobalBuffer((__gm__ float*)workspace);
    LocalTensor<float> raw(TPosition::VECCALC,0,3*SORT_CAP);
    LocalTensor<float> pairs(TPosition::VECCALC,49152,2*SORT_CAP),scratch(TPosition::VECCALC,81920,2*SORT_CAP);
    LocalTensor<float> key(TPosition::VECCALC,114688,SORT_CAP),temp(TPosition::VECCALC,147456,SORT_CAP);
    LocalTensor<uint32_t> ids(TPosition::VECCALC,131072,SORT_CAP),mask(TPosition::VECCALC,163840,SORT_CAP);
    LocalTensor<float> record(TPosition::VECCALC,180224,128),reduced(TPosition::VECCALC,180736,64);
    Duplicate(raw,0.0F,3*SORT_CAP);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if constexpr(INPUT_SOA) {
        // The global bucket already stores these exact three coordinate planes.
        // Zero padding matches the previous AoS reload; live values are bit copies.
        for(uint32_t axis=0;axis<3;++axis)
            DataCopyPad(raw[axis*SORT_CAP],gx[axis*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    } else {
        DataCopyPad(raw,gx,DataCopyExtParams{1,3*valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SpatialIndices(ids.ReinterpretCast<int32_t>(),count);
    Muls(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),int32_t(12),count);PipeBarrier<PIPE_V>();
    for(uint32_t axis=0;axis<3;++axis) {
        if constexpr(INPUT_SOA)
            Copy(temp,raw[axis*SORT_CAP],64,count/64,CopyRepeatParams(1,1,8,8));
        else Gather(temp,raw,ids,axis*4,count);
        PipeBarrier<PIPE_V>();auto coord=temp;
        WholeReduceMin(record,coord,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);float low=reduced.GetValue(0);
        WholeReduceMax(record,coord,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);float high=reduced.GetValue(0);
        const float span=high-low;
        float scale=span>=9.5367431640625e-7F?127.0F/span:0.0F;
        Adds(temp,coord,-low,count);PipeBarrier<PIPE_V>();Muls(temp,temp,scale,count);PipeBarrier<PIPE_V>();
        Maxs(temp,temp,0.0F,count);PipeBarrier<PIPE_V>();Mins(temp,temp,127.0F,count);PipeBarrier<PIPE_V>();
        Cast(mask.ReinterpretCast<int32_t>(),temp,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
        LocalTensor<float> quantized(TPosition::VECCALC,49152,3*SORT_CAP);
        Copy(quantized[axis*SORT_CAP],mask.ReinterpretCast<float>(),64,count/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    }
    LocalTensor<int32_t> quantized(TPosition::VECCALC,49152,3*SORT_CAP);
    LocalTensor<uint32_t> state(TPosition::VECCALC,98304,SORT_CAP);
    // Fixed Hilbert orientation automaton:24 states,8 input octants per state.
    // The packed output is3 key bits plus the next orientation state.
    // These stores initialize192 algorithm constants, never input elements.
    LocalTensor<uint64_t> hilbertConstants(TPosition::VECCALC,NORMAL_KEY_PACKETS?181248U:180224U,96);
    if constexpr(!NORMAL_KEY_PACKETS) {
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    InitHilbertConstants(hilbertConstants);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    auto hilbertTable=hilbertConstants.ReinterpretCast<float>();
    auto keyInteger=key.ReinterpretCast<uint32_t>(),digit=temp.ReinterpretCast<uint32_t>();
    if constexpr(NORMAL_KEY_PACKETS) {
        SpatialHilbertNormalPackets(quantized,state,keyInteger,digit,ids,mask,hilbertTable,count,3);
    } else {
    Duplicate(state,uint32_t(0),count);Duplicate(keyInteger,uint32_t(0),count);PipeBarrier<PIPE_V>();
    for(uint32_t remaining=7;remaining>3;--remaining) {
        Duplicate(mask,uint32_t(0),count);PipeBarrier<PIPE_V>();
        for(uint32_t axis=0;axis<3;++axis) {
            ShiftRight(digit,quantized[axis*SORT_CAP].ReinterpretCast<uint32_t>(),remaining-1,count);PipeBarrier<PIPE_V>();
            SpatialBitAndValue(digit.ReinterpretCast<int32_t>(),digit.ReinterpretCast<int32_t>(),1,ids.ReinterpretCast<float>(),count);
            ShiftLeft(digit,digit,2-axis,count);PipeBarrier<PIPE_V>();
            Add(mask.ReinterpretCast<int32_t>(),mask.ReinterpretCast<int32_t>(),digit.ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
        }
        ShiftLeft(ids,state,uint32_t(3),count);PipeBarrier<PIPE_V>();
        Add(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),mask.ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
        ShiftLeft(ids,ids,uint32_t(2),count);PipeBarrier<PIPE_V>();
        Gather(mask.ReinterpretCast<float>(),hilbertTable,ids,uint32_t(0),count);PipeBarrier<PIPE_V>();
        ShiftRight(state,mask,uint32_t(3),count);PipeBarrier<PIPE_V>();
        SpatialBitAndValue(digit.ReinterpretCast<int32_t>(),mask.ReinterpretCast<int32_t>(),7,ids.ReinterpretCast<float>(),count);
        ShiftLeft(keyInteger,keyInteger,uint32_t(3),count);PipeBarrier<PIPE_V>();
        Add(keyInteger.ReinterpretCast<int32_t>(),keyInteger.ReinterpretCast<int32_t>(),digit.ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
    }
    }
    Cast(key,keyInteger.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
    SpatialIndices(ids.ReinterpretCast<int32_t>(),count);
    if(valid<count) {
        Cast(temp,ids.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        CompareScalar(mask.ReinterpretCast<uint8_t>(),temp,float(int32_t(valid)),CMPMODE::LT,count);PipeBarrier<PIPE_V>();
        Select(key,mask.ReinterpretCast<uint8_t>(),key,-__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count);PipeBarrier<PIPE_V>();
    }
    SpatialSort(pairs,scratch,key,ids,count);
    GatherMask(ids.ReinterpretCast<float>(),pairs,uint8_t(2),GatherMaskParams(1,count/32,8,0));PipeBarrier<PIPE_V>();
    Muls(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),int32_t(INPUT_SOA?4:12),count);PipeBarrier<PIPE_V>();
    for(uint32_t axis=0;axis<3;++axis) {
        Gather(temp,raw[INPUT_SOA?axis*SORT_CAP:0],ids,INPUT_SOA?0:axis*4,count);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(gy[axis*SORT_CAP],temp,count);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    GatherMask(ids.ReinterpretCast<float>(),pairs,uint8_t(2),GatherMaskParams(1,count/32,8,0));PipeBarrier<PIPE_V>();
    Cast(key,ids.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();Muls(key,key,-1.0F,count);PipeBarrier<PIPE_V>();
    SpatialIndices(ids.ReinterpretCast<int32_t>(),count);SpatialSort(pairs,scratch,key,ids,count);
    GatherMask(ids.ReinterpretCast<float>(),pairs,uint8_t(2),GatherMaskParams(1,count/32,8,0));PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(gy[5*SORT_CAP],ids.ReinterpretCast<float>(),count);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

    // The sorting arena is now dead. Reuse 144 KiB for the vector gather
    // from sorted SoA to the existing operator's contiguous AoS input format.
    PipeBarrier<PIPE_ALL>();
    for(uint32_t axis=0;axis<3;++axis)DataCopy(raw[axis*SORT_CAP],gy[axis*SORT_CAP],count);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    LocalTensor<float> aos(TPosition::VECCALC,49152,3*SORT_CAP);
    LocalTensor<int32_t> gatherOffsets(TPosition::VECCALC,98304,3*SORT_CAP);
    SpatialIndices(gatherOffsets,3*count);
    Cast(aos,gatherOffsets,RoundMode::CAST_NONE,3*count);PipeBarrier<PIPE_V>();
    Muls(aos,aos,0.3333333432674408F,3*count);PipeBarrier<PIPE_V>();
    Cast(aos.ReinterpretCast<int32_t>(),aos,RoundMode::CAST_FLOOR,3*count);PipeBarrier<PIPE_V>();
    // 4*((i-3*q)*SORT_CAP+q) = (4*SORT_CAP)*i + (4-12*SORT_CAP)*q.
    Muls(gatherOffsets,gatherOffsets,int32_t(4*SORT_CAP),3*count);
    Muls(aos.ReinterpretCast<int32_t>(),aos.ReinterpretCast<int32_t>(),int32_t(4)-int32_t(12*SORT_CAP),3*count);PipeBarrier<PIPE_V>();
    Add(gatherOffsets,gatherOffsets,aos.ReinterpretCast<int32_t>(),3*count);PipeBarrier<PIPE_V>();
    Gather(aos,raw,gatherOffsets.ReinterpretCast<uint32_t>(),uint32_t(0),3*count);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gy,aos,DataCopyExtParams{1,3*valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
}
__aicore__ __attribute__((always_inline)) inline void TreeSpatialRestore(
    GM_ADDR workspace,GM_ADDR distance,GM_ADDR index,uint32_t valid) {
    GlobalTensor<float> gx,gd,gi;gx.SetGlobalBuffer((__gm__ float*)workspace);
    gd.SetGlobalBuffer((__gm__ float*)distance);gi.SetGlobalBuffer((__gm__ float*)index);
    GlobalTensor<uint32_t> gxInverse;gxInverse.SetGlobalBuffer((__gm__ uint32_t*)workspace+5*SORT_CAP);
    LocalTensor<float> raw(TPosition::VECCALC,0,2*SORT_CAP),outD(TPosition::VECCALC,32768,SORT_CAP),outI(TPosition::VECCALC,49152,SORT_CAP);
    LocalTensor<uint32_t> inverse(TPosition::VECCALC,65536,SORT_CAP);
    DataCopyPad(raw,gx[3*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    DataCopyPad(raw[SORT_CAP],gx[4*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    DataCopyPad(inverse,gxInverse,DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<uint32_t>{false,0,0,uint32_t(0)});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    ShiftLeft(inverse,inverse,uint32_t(2),valid);PipeBarrier<PIPE_V>();
    Gather(outD,raw,inverse,uint32_t(0),valid);Gather(outI,raw[SORT_CAP],inverse,uint32_t(0),valid);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gd,outD,DataCopyExtParams{1,valid*4,0,0,0});DataCopyPad(gi,outI,DataCopyExtParams{1,valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
}
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))

// Batch-wide geometric buckets reduce the spatial diameter of a query tile.
// Admission is derived from available cores, UB capacity and measured occupancy.
constexpr uint32_t TREE_BUCKET_CAPACITY=sizeof(DTYPE_TREE_POINTS)==2?16U:8U;
constexpr uint32_t TREE_BUCKET_BOUND_BASE=188160U+4U*TREE_BUCKET_CAPACITY;
__aicore__ __attribute__((noinline)) inline void TreeBucketKeys(uint32_t count,uint32_t owners) {
    const LocalTensor<float> low(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE,3),scale(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE+64,3);
    LocalTensor<float> raw(TPosition::VECCALC,0,3*SORT_CAP);
    LocalTensor<float> key(TPosition::VECCALC,114688,SORT_CAP),temp(TPosition::VECCALC,147456,SORT_CAP);
    LocalTensor<uint32_t> ids(TPosition::VECCALC,131072,SORT_CAP),mask(TPosition::VECCALC,163840,SORT_CAP);
    LocalTensor<float> record(TPosition::VECCALC,180224,128);
    SpatialIndices(ids.ReinterpretCast<int32_t>(),count);
    Muls(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),int32_t(12),count);PipeBarrier<PIPE_V>();
    LocalTensor<float> axes(TPosition::VECCALC,49152,3*SORT_CAP);
    for(uint32_t axis=0;axis<3;++axis) {
        Gather(temp,raw,ids,axis*4,count);PipeBarrier<PIPE_V>();
        Adds(temp,temp,-low.GetValue(axis),count);PipeBarrier<PIPE_V>();
        Muls(temp,temp,scale.GetValue(axis),count);PipeBarrier<PIPE_V>();
        Maxs(temp,temp,0.0F,count);PipeBarrier<PIPE_V>();Mins(temp,temp,127.0F,count);PipeBarrier<PIPE_V>();
        Cast(axes[axis*SORT_CAP].ReinterpretCast<int32_t>(),temp,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
    }
    LocalTensor<int32_t> quantized(TPosition::VECCALC,49152,3*SORT_CAP);
    LocalTensor<uint32_t> state(TPosition::VECCALC,98304,SORT_CAP);
    // Fixed Hilbert orientation automaton:24 states,8 input octants per state.
    // The packed output is3 key bits plus the next orientation state.
    // These stores initialize192 algorithm constants, never input elements.
    LocalTensor<uint64_t> hilbertConstants(TPosition::VECCALC,181248,96);
    {
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    InitHilbertConstants(hilbertConstants);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    auto hilbertTable=hilbertConstants.ReinterpretCast<float>();
    auto keyInteger=key.ReinterpretCast<uint32_t>(),digit=temp.ReinterpretCast<uint32_t>();
    const uint32_t lowest=((owners & (owners-1))==0 && (sizeof(DTYPE_TREE_POINTS)==4 || owners<=8))?6:4;
    SpatialHilbertNormalPackets(quantized,state,keyInteger,digit,ids,mask,hilbertTable,count,lowest);
    Cast(key,keyInteger.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
}

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
__aicore__ __attribute__((always_inline)) inline void TreeBucketSoAToAoS(
    GM_ADDR workspace,uint32_t valid,uint32_t count) {
    GlobalTensor<float> gy;gy.SetGlobalBuffer((__gm__ float*)workspace);
    LocalTensor<float> raw(TPosition::VECCALC,0,3*SORT_CAP),aos(TPosition::VECCALC,49152,3*SORT_CAP);
    LocalTensor<int32_t> offsets(TPosition::VECCALC,98304,3*SORT_CAP);
    PipeBarrier<PIPE_ALL>();
    for(uint32_t axis=0;axis<3;++axis)DataCopy(raw[axis*SORT_CAP],gy[axis*SORT_CAP],count);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SpatialIndices(offsets,3*count);Cast(aos,offsets,RoundMode::CAST_NONE,3*count);PipeBarrier<PIPE_V>();
    Muls(aos,aos,0.3333333432674408F,3*count);PipeBarrier<PIPE_V>();
    Cast(aos.ReinterpretCast<int32_t>(),aos,RoundMode::CAST_FLOOR,3*count);PipeBarrier<PIPE_V>();
    Muls(offsets,offsets,int32_t(4*SORT_CAP),3*count);
    Muls(aos.ReinterpretCast<int32_t>(),aos.ReinterpretCast<int32_t>(),int32_t(4)-int32_t(12*SORT_CAP),3*count);PipeBarrier<PIPE_V>();
    Add(offsets,offsets,aos.ReinterpretCast<int32_t>(),3*count);PipeBarrier<PIPE_V>();
    Gather(aos,raw,offsets.ReinterpretCast<uint32_t>(),uint32_t(0),3*count);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gy,aos,DataCopyExtParams{1,3*valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
}

#endif

__aicore__ __attribute__((always_inline)) inline void TreeBucketCopy(
    const LocalTensor<float>& destination,const LocalTensor<float>& input,uint32_t count) {
    const uint32_t full=count/64,tail=count%64;
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    if(full)Copy<float,false>(destination,input,uint64_t(64),full,CopyRepeatParams(1,1,8,8));
    if(tail) {
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(tail));
        Copy<float,false>(destination[full*64],input[full*64],uint64_t(tail),1,CopyRepeatParams(1,1,8,8));
    }
    PipeBarrier<PIPE_V>();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}

__aicore__ __attribute__((always_inline)) inline void TreeBucketRestore(
    GM_ADDR workspace,uint32_t valid) {
    GlobalTensor<float> gx;gx.SetGlobalBuffer((__gm__ float*)workspace);
    LocalTensor<float> raw(TPosition::VECCALC,0,2*SORT_CAP),outD(TPosition::VECCALC,32768,SORT_CAP),outI(TPosition::VECCALC,49152,SORT_CAP);
    LocalTensor<uint32_t> inverse(TPosition::VECCALC,65536,SORT_CAP);
    LocalTensor<float> constant(TPosition::VECCALC,81920,64);
    DataCopyPad(raw,gx[3*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    DataCopyPad(raw[SORT_CAP],gx[4*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    DataCopyPad(inverse.ReinterpretCast<float>(),gx[5*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SpatialBitAndValue(inverse.ReinterpretCast<int32_t>(),inverse.ReinterpretCast<int32_t>(),4095,constant,(valid+63)/64*64);
    ShiftLeft(inverse,inverse,uint32_t(2),valid);PipeBarrier<PIPE_V>();
    Gather(outD,raw,inverse,uint32_t(0),valid);Gather(outI,raw[SORT_CAP],inverse,uint32_t(0),valid);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gx[3*SORT_CAP],outD,DataCopyExtParams{1,valid*4,0,0,0});
    DataCopyPad(gx[4*SORT_CAP],outI,DataCopyExtParams{1,valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
}

__aicore__ __attribute__((always_inline)) inline void TreeBucketWriteOriginalOrder(
    GM_ADDR workspace,GM_ADDR distance,GM_ADDR index,const LocalTensor<uint32_t>& bucketCount,
    uint32_t batch,uint32_t owners,uint32_t owner,uint32_t points) {
    constexpr uint32_t stride=6*SORT_CAP+14*1056+1024;
    const uint32_t begin=points*owner/owners,end=points*(owner+1)/owners;
    LocalTensor<float> assembledD(TPosition::VECCALC,0,SORT_CAP),assembledI(TPosition::VECCALC,16384,SORT_CAP),assembledIds(TPosition::VECCALC,32768,SORT_CAP);
    LocalTensor<float> sourceD(TPosition::VECCALC,49152,SORT_CAP),sourceI(TPosition::VECCALC,65536,SORT_CAP);
    LocalTensor<uint32_t> encoded(TPosition::VECCALC,81920,SORT_CAP),ordinal(TPosition::VECCALC,131072,SORT_CAP);
    LocalTensor<float> originalId(TPosition::VECCALC,98304,SORT_CAP),packed(TPosition::VECCALC,114688,SORT_CAP);
    LocalTensor<uint8_t> predicate(TPosition::VECCALC,147456,512),upper(TPosition::VECCALC,147968,512),validMask(TPosition::VECCALC,148480,512);
    Duplicate(assembledIds,__builtin_inff(),SORT_CAP);PipeBarrier<PIPE_V>();uint32_t position=0;
    for(uint32_t other=0;other<owners;++other) {
        const uint32_t valid=bucketCount.GetValue(other);uint32_t count=(valid+63)/64*64;
        GlobalTensor<float> gs;gs.SetGlobalBuffer((__gm__ float*)(workspace+(batch*owners+other)*stride*4));
        DataCopyPad(sourceD,gs[3*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopyPad(sourceI,gs[4*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopyPad(encoded.ReinterpretCast<float>(),gs[5*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ShiftRight(encoded,encoded,uint32_t(12),count);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        Cast(originalId,encoded.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        const uint32_t gatherExtent=valid;
#else
        Cast(originalId,encoded.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);SpatialIndices(ordinal.ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
        Cast(packed,ordinal.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        CompareScalar(validMask,packed,float(int32_t(valid)),CMPMODE::LT,count);
        const uint32_t gatherExtent=count;
#endif
        CompareScalar(predicate,originalId,float(int32_t(begin)),CMPMODE::GE,count);
        CompareScalar(upper,originalId,float(int32_t(end)),CMPMODE::LT,count);PipeBarrier<PIPE_V>();
        And(predicate.ReinterpretCast<uint16_t>(),predicate.ReinterpretCast<uint16_t>(),upper.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
#if !defined(DTYPE_TREE_POINTS) || !TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        And(predicate.ReinterpretCast<uint16_t>(),predicate.ReinterpretCast<uint16_t>(),validMask.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
#endif
        uint64_t kept=0;GatherMask(packed,sourceD,predicate.ReinterpretCast<uint32_t>(),true,gatherExtent,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);uint32_t live=uint32_t(kept);
        if(live) {
            TreeBucketCopy(assembledD[position],packed,live);
            GatherMask(packed,sourceI,predicate.ReinterpretCast<uint32_t>(),true,gatherExtent,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            TreeBucketCopy(assembledI[position],packed,live);
            GatherMask(packed,originalId,predicate.ReinterpretCast<uint32_t>(),true,gatherExtent,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            TreeBucketCopy(assembledIds[position],packed,live);position+=(live+7)/8*8;
        }
    }
    LocalTensor<float> pairs(TPosition::VECCALC,49152,2*SORT_CAP),scratch(TPosition::VECCALC,81920,2*SORT_CAP);
    LocalTensor<float> key(TPosition::VECCALC,114688,SORT_CAP),outD(TPosition::VECCALC,147456,SORT_CAP),outI(TPosition::VECCALC,163840,SORT_CAP);
    LocalTensor<uint32_t> ids(TPosition::VECCALC,131072,SORT_CAP);uint32_t count=64;while(count<position)count*=2;
    Muls(key,assembledIds,-1.0F,count);SpatialIndices(ids.ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
    SpatialSort(pairs,scratch,key,ids,count);GatherMask(ids.ReinterpretCast<float>(),pairs,uint8_t(2),GatherMaskParams(1,count/32,8,0));PipeBarrier<PIPE_V>();
    ShiftLeft(ids,ids,uint32_t(2),count);PipeBarrier<PIPE_V>();
    Gather(outD,assembledD,ids,uint32_t(0),end-begin);Gather(outI,assembledI,ids,uint32_t(0),end-begin);PipeBarrier<PIPE_V>();
    GlobalTensor<float> gd,gi;gd.SetGlobalBuffer((__gm__ float*)distance+batch*points+begin);gi.SetGlobalBuffer((__gm__ float*)index+batch*points+begin);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gd,outD,DataCopyExtParams{1,(end-begin)*4,0,0,0});DataCopyPad(gi,outI,DataCopyExtParams{1,(end-begin)*4,0,0,0});PipeBarrier<PIPE_ALL>();
}

__aicore__ __attribute__((always_inline)) inline void TreeGlobalSpatialProcess(
    GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,GM_ADDR workspace,const TreeLineMinDistanceTilingData& t) {
    constexpr uint32_t stride=6*SORT_CAP+14*1056+1024;
    const uint32_t owners=t.blocks/t.batches,batch=GetBlockIdx()/owners,owner=GetBlockIdx()%owners;
    GlobalTensor<float> input,store;input.SetGlobalBuffer((__gm__ float*)p+batch*t.points*3);
    GM_ADDR local=workspace+GetBlockIdx()*stride*4;store.SetGlobalBuffer((__gm__ float*)local);
    LocalTensor<float> raw(TPosition::VECCALC,0,3*SORT_CAP),axes(TPosition::VECCALC,49152,3*SORT_CAP),originalId(TPosition::VECCALC,98304,SORT_CAP);
    LocalTensor<float> key(TPosition::VECCALC,114688,SORT_CAP),temp(TPosition::VECCALC,147456,SORT_CAP);
    LocalTensor<uint32_t> ids(TPosition::VECCALC,131072,SORT_CAP);
    LocalTensor<float> record(TPosition::VECCALC,180224,128),reduced(TPosition::VECCALC,180736,64);
    // Native R64 search uses UB bytes[0,188128); retain bounded metadata
    // in the aligned padding below184 KiB, outside all payload arenas.
    LocalTensor<uint32_t> bucketCount(TPosition::VECCALC,188160,TREE_BUCKET_CAPACITY);
    // Bounds are dead after key generation. Reuse their reserved UB padding
    // for one producer's bounded counts, outside the search payload arena.
    LocalTensor<uint32_t> pieceCounts(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE,TREE_BUCKET_CAPACITY);
    LocalTensor<float> low(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE,3),high(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE+32,3),scale(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE+64,3);
    PipeBarrier<PIPE_ALL>();
    #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
        for(uint32_t axis=0;axis<3;++axis){low.SetValue(axis,__builtin_inff());high.SetValue(axis,-__builtin_inff());}
    for(uint32_t bucket=0;bucket<TREE_BUCKET_CAPACITY;++bucket)bucketCount.SetValue(bucket,uint32_t(0));
    // Quantization is only a spatial ordering heuristic. Each key producer
    // may normalize its own disjoint interval; all consumers read that key.
    const uint32_t boundBegin=t.points*owner/owners;
    const uint32_t boundEnd=t.points*(owner+1)/owners;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_5
    bool localPointCoordinatesSafe=true;
#else // TREE_HALF_COORD_CERT_5
#endif // TREE_HALF_COORD_CERT_5
    {
        const uint32_t start=boundBegin;
        uint32_t valid=boundEnd-start;uint32_t count=64;while(count<valid)count*=2;
        Duplicate(raw,0.0F,3*SORT_CAP);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(raw,input[3*start],DataCopyExtParams{1,3*valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SpatialIndices(ids.ReinterpretCast<int32_t>(),count);Muls(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),int32_t(12),count);PipeBarrier<PIPE_V>();
        #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
        for(uint32_t axis=0;axis<3;++axis) {
            Gather(temp,raw,ids,axis*4,count);PipeBarrier<PIPE_V>();
            WholeReduceMin(record,temp,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMin(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);float lo=reduced.GetValue(0);if(lo<low.GetValue(axis))low.SetValue(axis,lo);
            WholeReduceMax(record,temp,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMax(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);float hi=reduced.GetValue(0);if(hi>high.GetValue(axis))high.SetValue(axis,hi);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_7
            // Finite inputs are required by the operator contract. Existing
            // axis bounds certify the upper magnitude; zero is admissible.
            auto nonzero=key.ReinterpretCast<uint8_t>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            // Test original bits before Abs: FTZ must not turn a nonzero
            // Float subnormal into an admissible mathematical zero. Negative
            // zero conservatively rejects the certificate and uses fallback.
            CompareScalar(nonzero,temp.ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,count);PipeBarrier<PIPE_V>();
            Not(nonzero.ReinterpretCast<uint16_t>(),nonzero.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
            Abs(temp,temp,count);PipeBarrier<PIPE_V>();
#else
            Abs(temp,temp,count);PipeBarrier<PIPE_V>();
            CompareScalar(nonzero,temp,0.0F,CMPMODE::NE,count);PipeBarrier<PIPE_V>();
#endif
            Select(temp,nonzero,temp,9.5367431640625e-7F,SELMODE::VSEL_TENSOR_SCALAR_MODE,count);PipeBarrier<PIPE_V>();
            WholeReduceMin(record,temp,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMin(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if(!(reduced.GetValue(0)>=9.5367431640625e-7F))localPointCoordinatesSafe=false;
#else // TREE_HALF_COORD_CERT_7
#endif // TREE_HALF_COORD_CERT_7
        }
    }
    bool localKeySafe=true;
    #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
        for(uint32_t axis=0;axis<3;++axis) {
        float lo=low.GetValue(axis),hi=high.GetValue(axis);
        if(!(lo>=-1099511627776.0F && hi<=1099511627776.0F))localKeySafe=false;
        float span=hi-lo;scale.SetValue(axis,span>=9.5367431640625e-7F?127.0F/span:0.0F);
    }
    bool eligible=true;
    // Unsafe producers publish a negative key for each real query. After
    // the existing key barrier every consumer observes the same missing
    // bucket membership, so the original total-count check rejects globally.
    {
        const uint32_t valid=boundEnd-boundBegin;
        const uint32_t count=(valid+63U)&~63U;
        if(localKeySafe) TreeBucketKeys(count,owners);
        else {Duplicate(key,-1.0F,count);PipeBarrier<PIPE_V>();}
        // Compute this producer's counts once. The same multiplication and
        // FLOOR cast define bucket membership in all original consumers.
        Muls(temp,key,float(int32_t(owners))*(((owners & (owners-1))==0 && (sizeof(DTYPE_TREE_POINTS)==4 || owners<=8))?0.125F:0.001953125F),count);PipeBarrier<PIPE_V>();
        Cast(ids.ReinterpretCast<int32_t>(),temp,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
        Cast(temp,ids.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        // Publish the canonical bucket ID once, using the original FP32
        // multiplication and FLOOR cast. Consumers need no reconstruction.
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(store[6*SORT_CAP],temp,DataCopyExtParams{1,valid*4,0,0,0});
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        for(uint32_t bucket=0;bucket<owners;++bucket) {
            CompareScalar(record.ReinterpretCast<uint8_t>(),temp,float(int32_t(bucket)),CMPMODE::EQ,count);PipeBarrier<PIPE_V>();
            // Counter mode excludes the padded tail directly.
            uint64_t kept=0;GatherMask(raw,temp,record.ReinterpretCast<uint32_t>(),true,valid,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            bucketCount.SetValue(bucket,uint32_t(kept));
        }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_9
        // Counts are at most SORT_CAP. Bit31 of the first word carries a
        // conservative certificate without changing the metadata transfer size.
        bucketCount.SetValue(0,bucketCount.GetValue(0)|((localPointCoordinatesSafe && localKeySafe)?0x80000000U:0U));
#else // TREE_HALF_COORD_CERT_9
#endif // TREE_HALF_COORD_CERT_9
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_11
        // The unchanged second global barrier precedes coefficient-cache reuse.
#else // TREE_HALF_COORD_CERT_11
        // This temporary count slot follows the complete key region. The
        // unchanged second global barrier precedes coefficient-cache reuse.
#endif // TREE_HALF_COORD_CERT_11
        DataCopy(store[7*SORT_CAP],bucketCount.ReinterpretCast<float>(),TREE_BUCKET_CAPACITY);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        for(uint32_t bucket=0;bucket<TREE_BUCKET_CAPACITY;++bucket)bucketCount.SetValue(bucket,uint32_t(0));
    }
    PipeBarrier<PIPE_ALL>();SyncAll<true>();
    SetMaskNorm();ResetMask();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_13
    bool allPointCoordinatesSafe=true;
#else // TREE_HALF_COORD_CERT_13
#endif // TREE_HALF_COORD_CERT_13
    if(eligible) {
        // Traverse the original point order, reading each producer interval.
        // Grouping boundaries change; stable per-bucket point order does not.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // Every producer uses a prefix of these unchanged gather offsets.
        const uint32_t producerMax=((t.points+owners-1U)/owners+63U)&~63U;
        SpatialIndices(ids.ReinterpretCast<int32_t>(),producerMax);PipeBarrier<PIPE_V>();
        Cast(originalId,ids.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,producerMax);PipeBarrier<PIPE_V>();
        Muls(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),int32_t(12),producerMax);PipeBarrier<PIPE_V>();
        uint32_t previousStart=0U;
#endif
        for(uint32_t piece=0;piece<owners;++piece) {
            const uint32_t start=t.points*piece/owners;
            const uint32_t valid=t.points*(piece+1)/owners-start;
            const uint32_t count=(valid+63U)&~63U;
            GlobalTensor<float> cachedKeys;
            cachedKeys.SetGlobalBuffer((__gm__ float*)(workspace+(batch*owners+piece)*stride*4)+6*SORT_CAP);
            Duplicate(raw,0.0F,3*SORT_CAP);Duplicate(key,0.0F,count);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopyPad(raw,input[3*start],DataCopyExtParams{1,3*valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
            DataCopyPad(key,cachedKeys,DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
            DataCopy(pieceCounts.ReinterpretCast<float>(),cachedKeys[SORT_CAP],TREE_BUCKET_CAPACITY);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
#if !defined(DTYPE_TREE_POINTS) || !TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            SpatialIndices(ids.ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
            Cast(temp,ids.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
#endif
            // Every payload GatherMask uses counter-mode valid; padded
            // lanes need no explicit validity Compare/Select.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            // All IDs stay below16*4096+4096, so every integer increment is exact in FP32.
            if(start!=previousStart) {
                Adds(originalId,originalId,float(int32_t(start-previousStart)),producerMax);PipeBarrier<PIPE_V>();
                previousStart=start;
            }
#else
            Adds(originalId,temp,float(int32_t(start)),count);Muls(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),int32_t(12),count);PipeBarrier<PIPE_V>();
#endif
            #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
        for(uint32_t axis=0;axis<3;++axis)Gather(axes[axis*SORT_CAP],raw,ids,axis*4,count);PipeBarrier<PIPE_V>();
            {
                CompareScalar(record.ReinterpretCast<uint8_t>(),key,float(int32_t(owner)),CMPMODE::EQ,count);PipeBarrier<PIPE_V>();
                uint64_t kept=0;
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_15
                allPointCoordinatesSafe=allPointCoordinatesSafe && ((pieceCounts.GetValue(0)&0x80000000U)!=0);
                uint32_t live=pieceCounts.GetValue(owner)&0x7fffffffU,prefix=bucketCount.GetValue(owner);
#else // TREE_HALF_COORD_CERT_15
                uint32_t live=pieceCounts.GetValue(owner),prefix=bucketCount.GetValue(owner);
#endif // TREE_HALF_COORD_CERT_15
                for(uint32_t bucket=0;bucket<owners;++bucket)
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_17
                    bucketCount.SetValue(bucket,bucketCount.GetValue(bucket)+(pieceCounts.GetValue(bucket)&0x7fffffffU));
#else // TREE_HALF_COORD_CERT_17
                    bucketCount.SetValue(bucket,bucketCount.GetValue(bucket)+pieceCounts.GetValue(bucket));
#endif // TREE_HALF_COORD_CERT_17
                if(live && prefix+live<=SORT_CAP) {
                    // All coordinate gathers consumed raw before this point.
                    // Its three disjoint planes can hold the compacted payload.
                    #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
        for(uint32_t axis=0;axis<3;++axis) {
                        GatherMask(raw[axis*SORT_CAP],axes[axis*SORT_CAP],record.ReinterpretCast<uint32_t>(),true,valid,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
                    }
                    GatherMask(temp,originalId,record.ReinterpretCast<uint32_t>(),true,valid,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
        for(uint32_t axis=0;axis<3;++axis)
                        DataCopyPad(store[axis*SORT_CAP+prefix],raw[axis*SORT_CAP],DataCopyExtParams{1,live*4,0,0,0});
                    DataCopyPad(store[4*SORT_CAP+prefix],temp,DataCopyExtParams{1,live*4,0,0,0});
                    // Drain before the next producer clears raw or reuses temp.
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                }
            }
        }
        uint32_t sum=0;
        for(uint32_t bucket=0;bucket<owners;++bucket) {uint32_t live=bucketCount.GetValue(bucket);sum+=live;if(live>SORT_CAP || live<t.points/(2*owners))eligible=false;}
        if(sum!=t.points)eligible=false;
    }
    // Complete every key read before any owner reuses its coefficient cache.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    AscendC::ICachePreLoad(int64_t(6));
#endif
    PipeBarrier<PIPE_ALL>();SyncAll<true>();
    SetMaskNorm();ResetMask();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    KernelTreeBatch<float,64,1024,1024,false,false> op;
    if(eligible) {
        uint32_t valid=bucketCount.GetValue(owner),count=64;while(count<valid)count*=2;
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // Consume the existing coordinate planes directly; preserve sorting and fallback.
        TreeSpatialArrange<true,true>(local,local,valid,count);
#else
        TreeSpatialArrange<true,true>(local,local,valid,count);
#endif
        LocalTensor<float> originals(TPosition::VECCALC,0,SORT_CAP),inverse(TPosition::VECCALC,16384,SORT_CAP);
        DataCopyPad(originals,store[4*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopyPad(inverse,store[5*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(originals.ReinterpretCast<int32_t>(),originals,RoundMode::CAST_FLOOR,valid);PipeBarrier<PIPE_V>();
        ShiftLeft(originals.ReinterpretCast<uint32_t>(),originals.ReinterpretCast<uint32_t>(),uint32_t(12),valid);PipeBarrier<PIPE_V>();
        Add(inverse.ReinterpretCast<int32_t>(),inverse.ReinterpretCast<int32_t>(),originals.ReinterpretCast<int32_t>(),valid);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopyPad(store[5*SORT_CAP],inverse,DataCopyExtParams{1,valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
        auto part=t;part.batches=1;part.points=valid;part.blocks=1;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_19
        op.Init(local,s+batch*t.segments*24,local+3*SORT_CAP*4,local+4*SORT_CAP*4,part);op.Process<true>(0,valid,local,allPointCoordinatesSafe);
#else // TREE_HALF_COORD_CERT_19
        op.Init(local,s+batch*t.segments*24,local+3*SORT_CAP*4,local+4*SORT_CAP*4,part);op.Process<true>(0,valid,local);
#endif // TREE_HALF_COORD_CERT_19
        TreeBucketRestore(local,valid);
    } else {op.Init(p,s,d,i,t);op.Process<true>();}
    SyncAll<true>();SetMaskNorm();ResetMask();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    if(eligible)TreeBucketWriteOriginalOrder(workspace,d,i,bucketCount,batch,owners,owner,t.points);
}

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
__aicore__ __attribute__((noinline,aligned(128))) inline void TreeGlobalSpatialCompactProcess(
    GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,GM_ADDR workspace,GM_ADDR tiling) {
    // Launch-owned metadata remains immutable in GM until the kernel ends.
    const auto words=(__gm__ uint32_t*)tiling;
    TreeLineMinDistanceTilingData t;
    t.batches=words[0];t.points=words[1];t.segments=words[2];t.paddedSegments=words[3];
    t.pointTile=words[4];t.blocks=words[5];t.schedule=words[6];t.segmentLog=words[7];
    TreeGlobalSpatialProcess(p,s,d,i,workspace,t);
}
#endif

__aicore__ __attribute__((always_inline)) inline void TreeSpatialProcess(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,
    GM_ADDR workspace,const TreeLineMinDistanceTilingData& t) {
    KernelTreeBatch<float,64,1024,1024,false,false> op;
    op.Init(p,s,d,i,t);
    const uint32_t total=t.batches*t.points,begin=total*GetBlockIdx()/t.blocks,end=total*(GetBlockIdx()+1)/t.blocks;
    if(!op.SpatialSortPointCoordinatesSafe(begin,end)) {op.Process();}
    else {
        constexpr uint32_t stride=6*SORT_CAP+14*1056+1024;
        GM_ADDR local=workspace+GetBlockIdx()*stride*4;
        for(uint32_t position=begin;position<end;) {
            uint32_t batch=position/t.points,start=position%t.points;
            uint32_t valid=end-position;if(valid>t.points-start)valid=t.points-start;if(valid>SORT_CAP)valid=SORT_CAP;
            uint32_t count=64;while(count<valid)count*=2;
            TreeSpatialArrange(p+position*12,local,valid,count);
            auto part=t;part.batches=1;part.points=valid;part.blocks=1;
            op.Init(local,s+batch*t.segments*24,local+3*SORT_CAP*4,local+4*SORT_CAP*4,part);
            op.Process(0,valid,local);
            TreeSpatialRestore(local,d+position*4,i+position*4,valid);
            position+=valid;
        }
    }
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
// Promotion is exact for every binary16 value; only the final distance narrows.
// The intermediate GM planes are private to the owner and always computed here.
__aicore__ __attribute__((always_inline)) inline void PromoteSpatialHalf(
    GM_ADDR input,GM_ADDR output,uint32_t count) {
    GlobalTensor<half> source;source.SetGlobalBuffer((__gm__ half*)input);
    GlobalTensor<float> destination;destination.SetGlobalBuffer((__gm__ float*)output);
    LocalTensor<half> raw(TPosition::VECCALC,32768U,4096U);
    LocalTensor<float> promoted(TPosition::VECCALC,0U,4096U);
    SetMaskNorm();ResetMask();
    for(uint32_t start=0;start<count;start+=4096U) {
        const uint32_t live=count-start<4096U?count-start:4096U;
        const uint32_t aligned=(live+63U)/64U*64U;
        Duplicate(raw,half(0),aligned);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(raw,source[start],DataCopyExtParams{1,live*2U,0,0,0},DataCopyPadExtParams<half>{false,0,0,half(0)});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(promoted,raw,RoundMode::CAST_NONE,aligned);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(destination[start],promoted,DataCopyExtParams{1,live*4U,0,0,0});
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
}
__aicore__ __attribute__((always_inline)) inline void WriteSpatialHalf(
    GM_ADDR sourceDistance,GM_ADDR sourceIndex,GM_ADDR outputDistance,GM_ADDR outputIndex,
    uint32_t count,bool copyIndex) {
    GlobalTensor<float> source;source.SetGlobalBuffer((__gm__ float*)sourceDistance);
    GlobalTensor<half> destination;destination.SetGlobalBuffer((__gm__ half*)outputDistance);
    GlobalTensor<int32_t> indices,indexOut;
    indices.SetGlobalBuffer((__gm__ int32_t*)sourceIndex);indexOut.SetGlobalBuffer((__gm__ int32_t*)outputIndex);
    LocalTensor<float> raw(TPosition::VECCALC,0U,4096U);
    LocalTensor<half> rounded(TPosition::VECCALC,32768U,4096U);
    LocalTensor<int32_t> indexBuffer(TPosition::VECCALC,65536U,4096U);
    SetMaskNorm();ResetMask();
    for(uint32_t start=0;start<count;start+=4096U) {
        const uint32_t live=count-start<4096U?count-start:4096U;
        DataCopyPad(raw,source[start],DataCopyExtParams{1,live*4U,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        if(copyIndex)DataCopyPad(indexBuffer,indices[start],DataCopyExtParams{1,live*4U,0,0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(rounded,raw,RoundMode::CAST_RINT,live);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(destination[start],rounded,DataCopyExtParams{1,live*2U,0,0,0});
        if(copyIndex)DataCopyPad(indexOut[start],indexBuffer,DataCopyExtParams{1,live*4U,0,0,0});
        PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ __attribute__((noinline)) inline void TreeHalfGlobalSpatialProcess(
    GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,GM_ADDR workspace,const TreeLineMinDistanceTilingData& t) {
    constexpr uint32_t spatialWords=6*SORT_CAP+14*1056+1024;
    const uint32_t total=t.batches*t.points;
    const uint32_t owners=t.blocks/t.batches,batch=GetBlockIdx()/owners,owner=GetBlockIdx()%owners;
    const uint32_t begin=total*GetBlockIdx()/t.blocks,end=total*(GetBlockIdx()+1)/t.blocks;
    GM_ADDR promotedPoints=workspace+t.blocks*spatialWords*4U;
    GM_ADDR promotedSegments=promotedPoints+total*12U;
    GM_ADDR promotedDistances=promotedSegments+t.batches*t.segments*24U;
    PromoteSpatialHalf(p+begin*6U,promotedPoints+begin*12U,(end-begin)*3U);
    if(owner==0)PromoteSpatialHalf(s+batch*t.segments*12U,
        promotedSegments+batch*t.segments*24U,t.segments*6U);
    PipeBarrier<PIPE_ALL>();SyncAll<true>();SetMaskNorm();ResetMask();
    TreeGlobalSpatialProcess(promotedPoints,promotedSegments,promotedDistances,i,workspace,t);
    // Global restore writes this owner's original interval before it returns.
    WriteSpatialHalf(promotedDistances+begin*4U,i+begin*4U,d+begin*2U,i+begin*4U,end-begin,false);
}
#endif

template<class T,uint32_t ROWS,uint32_t CAPACITY,uint32_t COEFFICIENTS,bool COMPACT_TAIL,bool SINGLE_TAIL>
class KernelTreeBatchPacked : public KernelTreeBatch<T,ROWS,CAPACITY,COEFFICIENTS,COMPACT_TAIL,SINGLE_TAIL> {
    using Base=KernelTreeBatch<T,ROWS,CAPACITY,COEFFICIENTS,COMPACT_TAIL,SINGLE_TAIL>;
    using Base::t_;
    using Base::m_;
    using Base::cp_;
    using Base::wp_;
    using Base::pp_;
    using Base::count_;
    using Base::liveRows_;
    using Base::pg_;
    using Base::sg_;
    using Base::dg_;
    using Base::ig_;
    using Base::rawP_;
    using Base::pf_;
    using Base::compactP_;
    using Base::refined_;
    using Base::chunkLower_;
    using Base::record_;
    using Base::record2_;
    using Base::hi_;
    using Base::second_;
    using Base::idx_;
    using Base::valid_;
    using Base::smallMask_;
    using Base::mask_;
    using Base::tileSecond_;
    using Base::offset_;
    using Base::pointOffset_;
    using Base::outputOffset_;
    using Base::tileLo_;
    using Base::tileIdx_;
    using Base::segmentIndices_;
    using Base::work_;
    using Base::columns_;
    using Base::xyz_;
    using Base::outT_;
    using Base::Smallest;
    using Base::C;
    using Base::W;
    using Base::P;
    using Base::LoadSegments;
    using Base::Search;
    using Base::Compensated;
    using Base::MakeIndices;
public:
    __aicore__ inline void Process() {
        const uint32_t total=t_.batches*t_.points;
        const uint32_t begin=total*GetBlockIdx()/t_.blocks,end=total*(GetBlockIdx()+1)/t_.blocks;
        uint32_t cached=0xffffffffU;
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_PACKED_EXACT
        bool coefficientSafe=false;
#endif // TREE_PACKED_EXACT
        for(uint32_t position=begin;position<end;) {
            const uint32_t batch=position/t_.points,start=position%t_.points;
            const uint32_t validRows=Smallest(Smallest(end-position,t_.points-start),ROWS);
            liveRows_=validRows;
            count_=COEFFICIENTS;
            if(batch!=cached){
                wp_=m_+32;
                if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)
                    this->LoadPrefixResidentSegments((GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),t_.segments);
                else LoadSegments(batch);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_PACKED_EXACT
                if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)
                    coefficientSafe=this->template CoarseCoordinatesSafe<true>();
#endif // TREE_PACKED_EXACT
                wp_=ROWS*64+128;cached=batch;
            }
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
                // The prior iteration ended at MTE3_S; coefficient setup also drains.
                DataCopyPad(rawP_,pg_[(batch*t_.points+start)*3],DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID3);WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
            } else {
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(rawP_,pg_[(batch*t_.points+start)*3],DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            PipeBarrier<PIPE_ALL>();
            }
            if constexpr(sizeof(T)==2)Cast(pf_,rawP_,RoundMode::CAST_NONE,192);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Gather(compactP_[axis*64],pf_,pointOffset_.template ReinterpretCast<uint32_t>(),axis*4,liveRows_);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Brcb(P(axis),compactP_[axis*64],(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            }
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) count_=this->SearchPrefixOutlined(liveRows_,t_.segments);
            else Search();
            Compare(smallMask_,second_,hi_,CMPMODE::GT,64);
            PipeBarrier<PIPE_V>();
            And(smallMask_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            uint64_t certified=smallMask_.template ReinterpretCast<uint64_t>().GetValue(0);
            const uint64_t active=(~uint64_t(0))>>(64-validRows);
            uint64_t exactRows=active&~certified,uniqueRows=0;
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
                if((certified&active)!=active) {
                    // Search predicates survive subsequent point broadcasts and exact scratch reuse.
                    #pragma unroll
                    for(uint32_t group=0;group<3;++group)
                        Copy(work_[14*1056+group*512],xyz_[group*pp_+ROWS*8],64,8,CopyRepeatParams(1,1,8,8));
                    PipeBarrier<PIPE_V>();
                }
            }
            constexpr uint32_t refinementRows=64;
            Muls(offset_,idx_,int32_t(4),64);PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t col=0;col<13;++col)
                if(col<6 || col>9)Gather(refined_[col*64],C(col),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
            Gather(refined_[9*64],C(13),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
            PipeBarrier<PIPE_V>();
            auto originalColumns=columns_,originalPoints=xyz_;
            const uint32_t originalCp=cp_,originalPp=pp_;
            columns_=refined_;cp_=64;xyz_=compactP_;pp_=64;count_=refinementRows;wp_=96;
            if constexpr(!Base::NATIVE_FLOAT_PREFIX_SPARSE) {
                Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),128);PipeBarrier<PIPE_V>();
            }
            Compensated();
            Copy(tileSecond_,W(0),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            columns_=originalColumns;cp_=originalCp;xyz_=originalPoints;pp_=originalPp;count_=m_;wp_=m_+32;
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
                if((certified&active)!=active)
                    certified|=this->RefineSparseRows(active&~certified,validRows,t_.segments,uniqueRows);
            }
            bool packed=false;
            if constexpr(CAPACITY==1024 && COEFFICIENTS<=576 && sizeof(T)==4) {
                if((certified & active) != active) {
                    if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)packed=RefinePrefixPacked(certified,active);
                    else packed=RefinePacked(certified,active);
                }
            }
            if((certified & active) != active && !packed) {
                if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)RefinePrefixUnpackedRows(certified,validRows);
                else {
                for(uint32_t row=0;row<validRows;++row) if(!(certified&(uint64_t(1)<<row))) {
                    // Only chunks whose conservative lower bound overlaps the incumbent can win.
                    Muls(offset_,outputOffset_,int32_t(32),16);PipeBarrier<PIPE_V>();
                    Adds(offset_,offset_,int32_t(row*4),16);
                    Duplicate(W(0),__builtin_inff(),64);
                    Brcb(W(1),hi_[(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                    Copy(W(2),W(1)[(row%8)*8],64,1,CopyRepeatParams(1,0,8,0));
                    Gather(W(0),chunkLower_,offset_.template ReinterpretCast<uint32_t>(),0U,(t_.segments+63)/64);PipeBarrier<PIPE_V>();
                    Compare(smallMask_,W(0),W(2),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const uint32_t chunks=smallMask_.template ReinterpretCast<uint32_t>().GetValue(0)&((1U<<((t_.segments+63)/64))-1);
                    uint32_t first=0,second=0,hits=0;
                    for(uint32_t chunk=0;chunk<(t_.segments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
                        if(hits==0)first=chunk;else if(hits==1)second=chunk;++hits;
                    }
                    const bool sparse=hits>0&&hits<=2;
                    const uint32_t candidates=sparse?hits*64:COEFFICIENTS;
                    auto searchColumns=columns_;const uint32_t searchCp=cp_;
                    if(sparse) {
                        auto selectedColumns=work_[16*1056];
                        #pragma unroll
                        for(uint32_t col=0;col<14;++col)if(col<6||col>9) {
                            Copy(selectedColumns[col*128],C(col)[first*64],64,1,CopyRepeatParams(1,1,8,8));
                            if(hits==2)Copy(selectedColumns[col*128+64],C(col)[second*64],64,1,CopyRepeatParams(1,1,8,8));
                        }
                        PipeBarrier<PIPE_V>();columns_=selectedColumns;cp_=128;
                        Duplicate(record2_.template ReinterpretCast<int32_t>(),int32_t(first*64),8);
                        Duplicate(record2_.template ReinterpretCast<int32_t>()[8],int32_t(second*64),8);
                    }
                    count_=candidates;wp_=sparse?160:m_+32;
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis) {
                        Brcb(W(0),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                        Copy(P(axis),W(0)[(row%8)*8],64,candidates/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
                    }
                    if(sparse) {
                        Adds(W(7),segmentIndices_,float(int32_t(first*64)),64);
                        if(hits==2)Adds(W(7)[64],segmentIndices_,float(int32_t(second*64)),64);
                    } else {
                        MakeIndices(W(6).template ReinterpretCast<int32_t>(),0,COEFFICIENTS);PipeBarrier<PIPE_V>();
                        Cast(W(7),W(6).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,COEFFICIENTS);
                    }
                    PipeBarrier<PIPE_V>();
                    CompareScalar(valid_,W(7),float(int32_t(t_.segments)),CMPMODE::LT,candidates);PipeBarrier<PIPE_V>();
                    this->template Compensated<true>();
                    WholeReduceMin(W(6),W(0),64,candidates/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(W(7),W(6),outputOffset_.template ReinterpretCast<uint32_t>(),0U,candidates/64);PipeBarrier<PIPE_V>();
                    WholeReduceMin(record_,W(7),candidates/64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(offset_,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
                    if(sparse) {
                        Muls(tileIdx_,offset_,int32_t(32),1);PipeBarrier<PIPE_V>();
                        Gather(tileIdx_,record2_.template ReinterpretCast<int32_t>(),tileIdx_.template ReinterpretCast<uint32_t>(),0U,1);
                    } else Muls(tileIdx_,offset_,int32_t(64),1);
                    Muls(offset_,offset_,int32_t(8),1);PipeBarrier<PIPE_V>();
                    Gather(offset_,W(6).template ReinterpretCast<int32_t>(),offset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
                    Add(tileIdx_,tileIdx_,offset_,1);PipeBarrier<PIPE_V>();
                    columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
                    this->CommitRefinedCandidate(row);
                }
            
                }
            }
            Sqrt(tileSecond_,tileSecond_,64);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_PACKED_EXACT
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
                RefineWideFloatRows((GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),t_.segments,
                    uint32_t(work_.GetPhyAddr()),uint32_t(compactP_.GetPhyAddr()),
                    uint32_t(idx_.GetPhyAddr()),uint32_t(tileSecond_.GetPhyAddr()),liveRows_,coefficientSafe);
            }
#endif // TREE_PACKED_EXACT
            auto selectedOutput=outT_;
            if constexpr(sizeof(T)==2)Cast(selectedOutput,tileSecond_,RoundMode::CAST_RINT,64);
            else selectedOutput=tileSecond_.template ReinterpretCast<T>();
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
            } else PipeBarrier<PIPE_ALL>();
            DataCopyPad(dg_[batch*t_.points+start],selectedOutput,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
            DataCopyPad(ig_[batch*t_.points+start],idx_,DataCopyExtParams{1,validRows*4,0,0,0});
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
                SetFlag<HardEvent::MTE3_S>(EVENT_ID3);WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
            } else PipeBarrier<PIPE_ALL>();
#if defined(DTYPE_TREE_POINTS)
            exactRows&=~uniqueRows;
            if(exactRows) {
                PipeBarrier<PIPE_ALL>();
                for(uint32_t row=0;row<validRows;++row)if(exactRows&(uint64_t(1)<<row)) {
                    const uint32_t pos=position+row;
                    TreeGeneralExactFallback<T>((GM_ADDR)pg_[pos*3].GetPhyAddr(),
                        (GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),
                        (GM_ADDR)dg_[pos].GetPhyAddr(),(GM_ADDR)ig_[pos].GetPhyAddr(),t_.segments);
                }
                if(position+validRows<end) {
                    if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)this->RestartBoundedAfterExact();
                    else {
                        const TreeLineMinDistanceTilingData t=t_;
                        this->Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
                                   (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
                    }
                    cached=0xffffffffU;
                }
            }
#endif
            wp_=ROWS*64+128;
            position+=validRows;
        }
    }
private:
    __aicore__ __attribute__((noinline,aligned(128))) void RefinePrefixUnpackedRows(uint64_t certified,uint32_t validRows) {
        static_assert(Base::NATIVE_FLOAT_PREFIX_SPARSE,"Native Float prefix cache required");

                for(uint32_t row=0;row<validRows;++row) if(!(certified&(uint64_t(1)<<row))) {
                    // Only chunks whose conservative lower bound overlaps the incumbent can win.
                    Muls(offset_,outputOffset_,int32_t(32),16);PipeBarrier<PIPE_V>();
                    Adds(offset_,offset_,int32_t(row*4),16);
                    Duplicate(W(0),__builtin_inff(),64);
                    Brcb(W(1),hi_[(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                    Copy(W(2),W(1)[(row%8)*8],64,1,CopyRepeatParams(1,0,8,0));
                    Gather(W(0),chunkLower_,offset_.template ReinterpretCast<uint32_t>(),0U,(t_.segments+63)/64);PipeBarrier<PIPE_V>();
                    Compare(smallMask_,W(0),W(2),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const uint32_t chunks=smallMask_.template ReinterpretCast<uint32_t>().GetValue(0)&((1U<<((t_.segments+63)/64))-1);
                    uint32_t first=0,second=0,hits=0;
                    for(uint32_t chunk=0;chunk<(t_.segments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
                        if(hits==0)first=chunk;else if(hits==1)second=chunk;++hits;
                    }
                    const bool sparse=hits>0&&hits<=2;
                    const uint32_t candidates=sparse?hits*64:COEFFICIENTS;
                    auto searchColumns=columns_;const uint32_t searchCp=cp_;
                    if(sparse) {
                        auto selectedColumns=work_[16*1056];
                        #pragma unroll
                        for(uint32_t col=0;col<14;++col)if(col<6||col>9) {
                            Copy(selectedColumns[col*128],C(col)[first*64],64,1,CopyRepeatParams(1,1,8,8));
                            if(hits==2)Copy(selectedColumns[col*128+64],C(col)[second*64],64,1,CopyRepeatParams(1,1,8,8));
                        }
                        PipeBarrier<PIPE_V>();columns_=selectedColumns;cp_=128;
                        Duplicate(record2_.template ReinterpretCast<int32_t>(),int32_t(first*64),8);
                        Duplicate(record2_.template ReinterpretCast<int32_t>()[8],int32_t(second*64),8);
                    }
                    count_=candidates;wp_=sparse?160:m_+32;
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis) {
                        Brcb(W(0),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                        Copy(P(axis),W(0)[(row%8)*8],64,candidates/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
                    }
                    if(sparse) {
                        Adds(W(7),segmentIndices_,float(int32_t(first*64)),64);
                        if(hits==2)Adds(W(7)[64],segmentIndices_,float(int32_t(second*64)),64);
                    } else {
                        MakeIndices(W(6).template ReinterpretCast<int32_t>(),0,COEFFICIENTS);PipeBarrier<PIPE_V>();
                        Cast(W(7),W(6).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,COEFFICIENTS);
                    }
                    PipeBarrier<PIPE_V>();
                    CompareScalar(valid_,W(7),float(int32_t(t_.segments)),CMPMODE::LT,candidates);PipeBarrier<PIPE_V>();
                    this->template Compensated<true>();
                    WholeReduceMin(W(6),W(0),64,candidates/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(W(7),W(6),outputOffset_.template ReinterpretCast<uint32_t>(),0U,candidates/64);PipeBarrier<PIPE_V>();
                    WholeReduceMin(record_,W(7),candidates/64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(offset_,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
                    if(sparse) {
                        Muls(tileIdx_,offset_,int32_t(32),1);PipeBarrier<PIPE_V>();
                        Gather(tileIdx_,record2_.template ReinterpretCast<int32_t>(),tileIdx_.template ReinterpretCast<uint32_t>(),0U,1);
                    } else Muls(tileIdx_,offset_,int32_t(64),1);
                    Muls(offset_,offset_,int32_t(8),1);PipeBarrier<PIPE_V>();
                    Gather(offset_,W(6).template ReinterpretCast<int32_t>(),offset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
                    Add(tileIdx_,tileIdx_,offset_,1);PipeBarrier<PIPE_V>();
                    columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
                    this->CommitRefinedCandidate(row);
                }
            
    }
    __aicore__ __attribute__((noinline,aligned(128))) bool RefinePrefixPacked(uint64_t certified,uint64_t active) {
        static_assert(Base::NATIVE_FLOAT_PREFIX_SPARSE,"Native Float prefix cache required");
        return RefinePacked(certified,active);
    }
    __aicore__ inline bool RefinePacked(uint64_t certified,uint64_t active) {
        // Non-finite search bounds keep the complete per-point fallback.
        CompareScalar(smallMask_,hi_,__builtin_inff(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if((smallMask_.template ReinterpretCast<uint64_t>().GetValue(0)&active)!=active)return false;
        const uint64_t uncertain=active&~certified;
        const uint32_t chunks=(t_.segments+63)/64;
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetMaskNorm();
        SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
        Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(uncertain),uint64_t(64),1,1,8);
        SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL));
        Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(uncertain>>32),uint64_t(64),1,1,8);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Duplicate(W(0),__builtin_inff(),64);PipeBarrier<PIPE_V>();
        SetCmpMask(valid_);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,W(0),tileSecond_,1,same);PipeBarrier<PIPE_V>();
        auto originalColumns=columns_,originalPoints=xyz_;
        const uint32_t originalCp=cp_,originalPp=pp_;
        for(uint32_t chunkBegin=0;chunkBegin<chunks;chunkBegin+=12) {
            Duplicate(mask_.template ReinterpretCast<uint16_t>(),uint16_t(0),64);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,uint8_t,false>(mask_,chunkLower_,hi_,CMPMODE::LE,uint64_t(64),chunks,BinaryRepeatParams(1,1,1,8,8,0));PipeBarrier<PIPE_V>();
            Or(valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),mask_[64].template ReinterpretCast<uint16_t>(),32);PipeBarrier<PIPE_V>();
            Or(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),valid_[32].template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            Duplicate(W(0).template ReinterpretCast<int32_t>(),int32_t(31),8);
            ShiftRight(offset_,outputOffset_,int32_t(1),8);PipeBarrier<PIPE_V>();
            Adds(offset_,offset_,int32_t(16),8);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),offset_.template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            Gather(W(1).template ReinterpretCast<uint32_t>(),valid_.template ReinterpretCast<uint32_t>(),offset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Or(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),W(1).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            ShiftRight(offset_,outputOffset_,int32_t(1),8);PipeBarrier<PIPE_V>();
            Adds(offset_,offset_,int32_t(8),8);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),offset_.template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            Gather(W(1).template ReinterpretCast<uint32_t>(),valid_.template ReinterpretCast<uint32_t>(),offset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Or(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),W(1).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t covered=valid_.template ReinterpretCast<uint64_t>().GetValue(0);
            const uint64_t missing=uncertain&~covered;
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
            Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(missing),uint64_t(64),1,1,8);
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL));
            Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(missing>>32),uint64_t(64),1,1,8);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();
            Or(mask_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),chunks*4);PipeBarrier<PIPE_V>();
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
            Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(uncertain),uint64_t(64),1,1,8);
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL));
            Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(uncertain>>32),uint64_t(64),1,1,8);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();
            And(mask_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),chunks*4);PipeBarrier<PIPE_V>();
            const uint32_t groupChunks=Smallest(chunks-chunkBegin,12);
            MakeIndices(work_.template ReinterpretCast<int32_t>(),int32_t(chunkBegin*64),groupChunks*64);PipeBarrier<PIPE_V>();
            uint64_t kept=0;
            GatherMask(refined_,work_,mask_[chunkBegin*8].template ReinterpretCast<uint32_t>(),true,groupChunks*64,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            for(uint32_t first=0;first<uint32_t(kept);first+=8) {
                const uint32_t queries=Smallest(uint32_t(kept)-first,8),pairs=queries*64;
                wp_=544;
                Copy(W(0),refined_[first],8,1,CopyRepeatParams(1,1,1,1));
                Duplicate(W(2).template ReinterpretCast<uint32_t>(),uint32_t(63),8);PipeBarrier<PIPE_V>();
                ShiftRight(W(1).template ReinterpretCast<uint32_t>(),W(0).template ReinterpretCast<uint32_t>(),uint32_t(6),8);
                And(W(3).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(2).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
                ShiftLeft(W(1).template ReinterpretCast<uint32_t>(),W(1).template ReinterpretCast<uint32_t>(),uint32_t(8),8);
                ShiftLeft(W(3).template ReinterpretCast<uint32_t>(),W(3).template ReinterpretCast<uint32_t>(),uint32_t(2),8);PipeBarrier<PIPE_V>();
                Brcb(W(2),W(1),1,BrcbRepeatParams(1,8));
                Brcb(W(4),W(3),1,BrcbRepeatParams(1,8));
                ShiftRight(W(5).template ReinterpretCast<int32_t>(),outputOffset_,int32_t(1),64);PipeBarrier<PIPE_V>();
                Copy(W(6),W(2),64,queries,CopyRepeatParams(1,0,8,1));
                Copy(W(7),W(4),64,queries,CopyRepeatParams(1,0,8,1));PipeBarrier<PIPE_V>();
                Add<int32_t,false>(W(6).template ReinterpretCast<int32_t>(),W(5).template ReinterpretCast<int32_t>(),W(6).template ReinterpretCast<int32_t>(),uint64_t(64),queries,BinaryRepeatParams(1,1,1,8,0,8));PipeBarrier<PIPE_V>();
                auto packedColumns=work_[8*1056];
                #pragma unroll
                for(uint32_t col=0;col<14;++col)if(col<6||col>9)
                    Gather(packedColumns[col*512],C(col),W(6).template ReinterpretCast<uint32_t>(),0U,pairs);
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)
                    Gather(P(axis),compactP_[axis*64],W(7).template ReinterpretCast<uint32_t>(),0U,pairs);
                PipeBarrier<PIPE_V>();
                ShiftRight(W(6).template ReinterpretCast<int32_t>(),W(6).template ReinterpretCast<int32_t>(),int32_t(2),pairs);PipeBarrier<PIPE_V>();
                Cast(W(5),W(6).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,pairs);PipeBarrier<PIPE_V>();
                CompareScalar(valid_,W(5),float(int32_t(t_.segments)),CMPMODE::LT,pairs);PipeBarrier<PIPE_V>();
                columns_=packedColumns;cp_=512;count_=pairs;
                this->template Compensated<true>();
                WholeReduceMin(record_,W(0),64,queries,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,1,8,0));
                GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
                columns_=originalColumns;cp_=originalCp;xyz_=originalPoints;pp_=originalPp;count_=m_;wp_=m_+32;
                Brcb(W(0),tileLo_,1,BrcbRepeatParams(1,8));
                Brcb(W(1),tileIdx_.template ReinterpretCast<float>(),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for(uint32_t query=0;query<queries;++query) {
                    const uint32_t gid=refined_.template ReinterpretCast<uint32_t>().GetValue(first+query);
                    Copy(W(2),W(0)[query*8],64,1,CopyRepeatParams(1,0,8,0));
                    Copy(W(3),W(1)[query*8],64,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
                    Adds(W(3).template ReinterpretCast<int32_t>(),W(3).template ReinterpretCast<int32_t>(),int32_t(gid&~uint32_t(63)),64);
                    CompareScalar(smallMask_,segmentIndices_,float(int32_t(gid&63)),CMPMODE::EQ,64);
                    Compare(mask_,W(2),tileSecond_,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
                    And(smallMask_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                    SetCmpMask(smallMask_);
                    Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,W(2),tileSecond_,1,same);
                    Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),W(3),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
                }
            }
        }
        return true;
    }
};

template <typename T, uint32_t FIXED_ROWS>
__aicore__ __attribute__((noinline)) void TreeM1FallbackTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    GM_ADDR tilingAddress) {
    TreeLineMinDistanceTilingData tiling;
    const auto words=reinterpret_cast<__gm__ const uint32_t*>(tilingAddress);
    tiling.batches=words[0];tiling.points=words[1];
    tiling.segments=1;tiling.paddedSegments=1;
    tiling.pointTile=words[4];tiling.blocks=words[5];
    tiling.schedule=3;tiling.segmentLog=0;
    // Keep each complete multi-tile implementation outside the bounded owner entry.
    KernelTreePairs<T,3,false,FIXED_ROWS,(FIXED_ROWS ? 1 : 0)> op;
    op.Init(points,segments,distance,indices,tiling);op.Process();
    if constexpr(sizeof(T)==4 && FIXED_ROWS==0) {
        
    }
}

// Fixed-capacity M1 resource. All addresses match KernelTreePairs<float,3,true,1024,1>.
class TreeM1FixedArena {
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> C(uint32_t col) {
        const uint32_t offset=col*104U+(col==9?16U:0U);
        return LocalTensor<float>(TPosition::VECCALC,32U+offset*4U,1352U-offset);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> W(uint32_t col) {
        return LocalTensor<float>(TPosition::VECCALC,30656U+col*4096U,9504U-col*1024U);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> P(uint32_t col) {
        return LocalTensor<float>(TPosition::VECCALC,17984U+col*4096U,3168U-col*1024U);
    }
public:
    __aicore__ __attribute__((always_inline)) static inline void Process(
        GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,uint32_t pointCount) {
        
        const uint32_t tiles=(pointCount+1023U)/1024U;
        const uint32_t owner=GetBlockIdx();
        uint32_t batch,begin;
        if(owner<tiles) {batch=0;begin=owner*1024U;}
        else {batch=owner/tiles;begin=(owner-batch*tiles)*1024U;}
        const uint32_t validRows=pointCount-begin<1024U?pointCount-begin:1024U;
        GlobalTensor<float> pg,sg,dg;GlobalTensor<int32_t> ig;
        pg.SetGlobalBuffer((__gm__ float*)points);sg.SetGlobalBuffer((__gm__ float*)segments);
        dg.SetGlobalBuffer((__gm__ float*)distance);ig.SetGlobalBuffer((__gm__ int32_t*)indices);
        LocalTensor<float> sf(TPosition::VECCALC,0U,8U),pf(TPosition::VECCALC,5440U,3136U);
        LocalTensor<uint32_t> bitmap(TPosition::VECCALC,68672U,1056U);
        LocalTensor<uint8_t> mask(TPosition::VECCALC,77120U,160U);
        LocalTensor<int32_t> outIdx(TPosition::VECCALC,123392U,1024U);
        LocalTensor<float> out(TPosition::VECCALC,127488U,1024U);
        LocalTensor<float> projection(TPosition::VECCALC,127552U,1024U);
        const BinaryRepeatParams same{1,1,1,8,8,8},coefficient{1,1,1,8,8,0};
        const UnaryRepeatParams unary{1,1,8,8};
        // All six segment coordinates are overwritten by the following DMA.
        // Only a partial point tile has a vector initialization to order before MTE2.
        if(validRows!=1024U) {
            Duplicate(pf,0.0F,(validRows*3U+63U)/64U*64U);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        }
        DataCopyPad(sf,sg[batch*6U],DataCopyExtParams{1,24U,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        DataCopyPad(pf,pg[(batch*pointCount+begin)*3U],DataCopyExtParams{1,validRows*12U,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        constexpr uint32_t words[3]={0x49249249U,0x92492492U,0x24924924U};
        constexpr uint64_t lanes[3]={0x249249ULL,0x492492ULL,0x924924ULL};
        SetMaskNorm();
        #pragma unroll
        for(uint32_t phase=0;phase<3;++phase) {
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),lanes[phase]);
            Duplicate<uint32_t,false>(bitmap,words[phase],uint64_t(64),1,1,8);
        }
        PipeBarrier<PIPE_V>();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
        Copy(bitmap[24],bitmap,24,4,CopyRepeatParams(1,1,3,0));PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);PipeBarrier<PIPE_V>();
        Brcb(C(0),sf,1,BrcbRepeatParams(13,104));PipeBarrier<PIPE_V>();
        Copy(C(0)[8],C(0),56,6,CopyRepeatParams(1,0,13,13));
        PipeBarrier<PIPE_V>();SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Sub<float,false>(C(6),C(3),C(0),uint64_t(64),3,BinaryRepeatParams(1,1,1,13,13,13));PipeBarrier<PIPE_V>();
        Sub<float,false>(W(0),C(3),C(6),uint64_t(64),3,BinaryRepeatParams(1,1,1,132,13,13));PipeBarrier<PIPE_V>();
        Add<float,false>(W(3),C(6),W(0),uint64_t(64),3,BinaryRepeatParams(1,1,1,132,13,132));
        Sub<float,false>(C(10),W(0),C(0),uint64_t(64),3,BinaryRepeatParams(1,1,1,13,132,13));PipeBarrier<PIPE_V>();
        Sub<float,false>(W(3),C(3),W(3),uint64_t(64),3,BinaryRepeatParams(1,1,1,132,13,132));PipeBarrier<PIPE_V>();
        Add<float,false>(C(10),C(10),W(3),uint64_t(64),3,BinaryRepeatParams(1,1,1,13,13,132));
        Mul<float,false>(C(9),C(6),C(6),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(7),C(7),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(8),C(8),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(mask,C(9),0.0F,CMPMODE::GT,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        Select<float,uint8_t,false>(C(9),mask,C(9),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(64),1,same);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        SetFlag<HardEvent::V_S>(EVENT_ID0);
#endif
        PipeBarrier<PIPE_V>();WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);PipeBarrier<PIPE_V>();
        SetMaskCount();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            SetVectorMask<float,MaskMode::COUNTER>(uint64_t(3072));
            GatherMask(P(axis),pf,bitmap[axis==0?0U:(axis==1?16U:8U)].ReinterpretCast<float>(),
                uint8_t(0),GatherMaskParams(1,1,8,1));
        }
        PipeBarrier<PIPE_V>();SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Copy<float,false>(pf[axis*1024U],C(axis),uint64_t(64),16,CopyRepeatParams(1,0,8,0));
        PipeBarrier<PIPE_V>();
        Sub<float,false>(W(0),P(0),pf,uint64_t(64),48,same);
        Sub<float,false>(W(3),P(0),W(0),uint64_t(64),48,same);
        Mul<float,false>(projection,W(0),C(6),uint64_t(64),16,coefficient);
        Add<float,false>(W(6),W(0),W(3),uint64_t(64),48,same);
        MulAddDst<float,float,false>(projection,W(1),C(7),uint64_t(64),16,coefficient);
        Sub<float,false>(W(3),W(3),pf,uint64_t(64),48,same);
        MulAddDst<float,float,false>(projection,W(2),C(8),uint64_t(64),16,coefficient);
        Sub<float,false>(W(6),P(0),W(6),uint64_t(64),48,same);
        Div<float,false>(projection,projection,C(9),uint64_t(64),16,coefficient);
        Add<float,false>(W(3),W(3),W(6),uint64_t(64),48,same);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t errorBits=C(10).ReinterpretCast<uint32_t>().GetValue(0)
            |C(11).ReinterpretCast<uint32_t>().GetValue(0)
            |C(12).ReinterpretCast<uint32_t>().GetValue(0);
        const bool directionErrorZero=(errorBits&0x7fffffffU)==0;
#endif
        PipeBarrier<PIPE_V>();
        Maxs<float,false>(projection,projection,0.0F,uint64_t(64),16,unary);PipeBarrier<PIPE_V>();
        Mins<float,false>(projection,projection,1.0F,uint64_t(64),16,unary);PipeBarrier<PIPE_V>();
        // Negate cached directions once instead of every query projection.
        // For the following FMAs, t*(-u) has the same signed product as (-t)*u.
        Muls<float,false>(C(6),C(6),-1.0F,uint64_t(64),3,UnaryRepeatParams(1,1,13,13));
        Muls<float,false>(C(10),C(10),-1.0F,uint64_t(64),3,UnaryRepeatParams(1,1,13,13));
        PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        if(directionErrorZero) {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                MulAddDst<float,float,false>(W(axis),projection,C(axis+6U),uint64_t(64),16,coefficient);
        } else
#endif
        {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                MulAddDst<float,float,false>(W(axis),projection,C(axis+6U),uint64_t(64),16,coefficient);
                MulAddDst<float,float,false>(W(axis+3U),projection,C(axis+10U),uint64_t(64),16,coefficient);
            }
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Add<float,false>(W(axis),W(axis),W(axis+3U),uint64_t(64),16,same);
        PipeBarrier<PIPE_V>();
        Mul<float,false>(P(0),W(0),W(0),uint64_t(64),16,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(P(0),W(1),W(1),uint64_t(64),16,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(P(0),W(2),W(2),uint64_t(64),16,same);PipeBarrier<PIPE_V>();
        CompareScalar(mask,P(0),1152921504606846976.0F,CMPMODE::LE,1024);
        Sqrt<float,false>(out,P(0),uint64_t(64),16,unary);
        Duplicate<int32_t,false>(outIdx,int32_t(0),uint64_t(64),16,1,8);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float norm=C(9).GetValue(0);
        bool needsExact=!(norm>=0.0F && norm<=1152921504606846976.0F);
        auto safe=mask.ReinterpretCast<uint64_t>();
        for(uint32_t first=0;first<validRows;first+=64U) {
            const uint32_t count=validRows-first<64U?validRows-first:64U;
            const uint64_t active=count==64U?~uint64_t(0):(uint64_t(1)<<count)-1U;
            if((safe.GetValue(first/64U)&active)!=active)needsExact=true;
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        DataCopyPad(dg[batch*pointCount+begin],out,DataCopyExtParams{1,validRows*4U,0,0,0});
        DataCopyPad(ig[batch*pointCount+begin],outIdx,DataCopyExtParams{1,validRows*4U,0,0,0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
        if(needsExact) {
            PipeBarrier<PIPE_ALL>();
            for(uint32_t row=0;row<validRows;++row) {
                const uint32_t pos=batch*pointCount+begin+row;
                TreeGeneralExactFallback<float>((GM_ADDR)pg[pos*3U].GetPhyAddr(),
                    (GM_ADDR)sg[batch*6U].GetPhyAddr(),(GM_ADDR)dg[pos].GetPhyAddr(),
                    (GM_ADDR)ig[pos].GetPhyAddr(),1U);
            }
        }
    }
};

template <typename T>
__aicore__ __attribute__((noinline)) void TreeM1SingleTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    uint32_t pointCount) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    if constexpr(sizeof(T)==4)AscendC::ICachePreLoad(int64_t(1));
#endif
    TreeM1FixedArena::Process(points,segments,distance,indices,pointCount);
    
}

template <typename T>
__aicore__ __attribute__((noinline)) void TreeTinyPairTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    const TreeLineMinDistanceTilingData &tiling) {
    // Preserve the existing one-query, two-segment vector resource in a helper.
    KernelTreeTiny<T,1,8,true,true,false,false,2> op;
    op.Init(points,segments,distance,indices,tiling);op.Process();
}

template <typename T>
__aicore__ __attribute__((noinline)) void TreeMidSingleTailTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    const TreeLineMinDistanceTilingData &tiling) {
    // Keep the bounded coefficient-cache resource in a separate device function.
    KernelTreeBatch<T,64,256,192,true,true> op;
    op.Init(points,segments,distance,indices,tiling);op.Process();
}

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template <typename T, uint32_t FIXED_POINTS=0>
__aicore__ __attribute__((noinline,aligned(128))) void TreeShortPairTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    uint32_t pointCount, uint32_t segmentCount) {
    // Fixed resource: only the actual N and M are live in this instantiation.
    // Process uses the existing bounded owner mapping; neither B nor blocks
    // is read by the single-tile class. All generic resources stay in the entry.
    TreeLineMinDistanceTilingData resource{0,FIXED_POINTS?FIXED_POINTS:pointCount,segmentCount,32,4,0,1,5};
    KernelTreePairs<T,1,true,4,32> op;
    op.Init(points,segments,distance,indices,resource);op.Process();
}
#endif

#if defined(DTYPE_TREE_POINTS)
template<typename T,uint32_t OWNERS=0>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyOneTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices) {
    AscendC::ICachePreLoad(int64_t(1));
    const TreeLineMinDistanceTilingData resource{0,1,2,8,1,0,16,3};
    KernelTreeTiny<T,0,8,(sizeof(T)!=4),true,false,false,2,false,true> op;
    op.Init(points,segments,distance,indices,resource);op.Process(OWNERS==1U?0U:GetBlockIdx());
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<typename T>
__aicore__ __attribute__((noinline)) void TreeOneQueryTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,
    uint32_t batches,uint32_t segmentsCount,uint32_t blocks) {
    const TreeLineMinDistanceTilingData tiling{batches,1,segmentsCount,1024,1,blocks,8,10};
    KernelTreePairs<T,8,false,1,1024> op;
    op.Init(points,segments,distance,indices,tiling);op.Process();
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<typename T,uint32_t FIXED_POINTS=0>
__aicore__ __attribute__((always_inline)) inline void TreeFewPointTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,uint32_t pointCount) {
    const TreeLineMinDistanceTilingData resource{0,FIXED_POINTS?FIXED_POINTS:pointCount,1,8,4,0,15,3};
    KernelTreeTiny<T,4,1,false,false,true,true,0,false,true> op;
    op.Init(points,segments,distance,indices,resource);op.Process(FIXED_POINTS>0U && FIXED_POINTS<=4U?0U:GetBlockIdx());
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<typename T,uint32_t FIXED_POINTS=0>
__aicore__ __attribute__((noinline,aligned(128))) void TreeNativePairPointTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,uint32_t pointCount) {
    AscendC::ICachePreLoad(int64_t(1));
    const TreeLineMinDistanceTilingData resource{0,FIXED_POINTS?FIXED_POINTS:pointCount,2,8,4,0,23,3};
    KernelTreeTiny<T,4,2,false,false,false,false,2,false,true> op;
    op.Init(points,segments,distance,indices,resource);op.Process(FIXED_POINTS>0U && FIXED_POINTS<=4U?0U:GetBlockIdx());
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<typename T,uint32_t FIXED_SEGMENTS=0>
__aicore__ __attribute__((noinline,aligned(128))) void TreeBoundedSingleQueryTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,uint32_t segmentCount) {
    const TreeLineMinDistanceTilingData resource{0,1,FIXED_SEGMENTS?FIXED_SEGMENTS:segmentCount,8,1,0,24,3};
    KernelTreeTiny<T,0,8,false,true,false,false,FIXED_SEGMENTS,false,true> op;
    op.Init(points,segments,distance,indices,resource);op.Process(FIXED_SEGMENTS>0U?0U:GetBlockIdx());
}
#endif

#if defined(DTYPE_TREE_POINTS)
template<uint32_t RESERVED>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyExactCore(
    GM_ADDR distanceAddress,GM_ADDR indexAddress,uint32_t rows,uint32_t segments,bool halfOutput);
template<class T>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyExactFallback(
    GM_ADDR pointAddress,GM_ADDR segmentAddress,GM_ADDR distanceAddress,GM_ADDR indexAddress,
    uint32_t rows,uint32_t segments) {
    // All previous Tiny buffers are dead. This disjoint arena holds exact
    // radix-4096 rational packets; no host numerical fallback is involved.
    PipeBarrier<PIPE_ALL>();SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    LocalTensor<T> raw(TPosition::VECCALC,100000,128);
    LocalTensor<float> pf(TPosition::VECCALC,100512,64),sf(TPosition::VECCALC,100768,128);
    LocalTensor<float> point(TPosition::VECCALC,101280,48),temp(TPosition::VECCALC,101536,64);
    LocalTensor<int32_t> lanes(TPosition::VECCALC,101792,16),offsets(TPosition::VECCALC,101856,16);
    LocalTensor<float> exponents(TPosition::VECCALC,101920,192),scratch(TPosition::VECCALC,102688,64);
    LocalTensor<float> record(TPosition::VECCALC,102944,64);
    LocalTensor<uint8_t> mask(TPosition::VECCALC,103200,256);
    GlobalTensor<T> pg,sg,dg;GlobalTensor<int32_t> ig;
    pg.SetGlobalBuffer((__gm__ T*)pointAddress);sg.SetGlobalBuffer((__gm__ T*)segmentAddress);
    dg.SetGlobalBuffer((__gm__ T*)distanceAddress);ig.SetGlobalBuffer((__gm__ int32_t*)indexAddress);
    Duplicate(pf,0.0F,64);Duplicate(sf,0.0F,128);
    if constexpr(sizeof(T)==2)Duplicate(raw,T(0),128);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    if constexpr(sizeof(T)==2) {
        DataCopyPad(raw,pg,DataCopyExtParams{1,rows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
        DataCopyPad(raw[64],sg,DataCopyExtParams{1,segments*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
    } else {
        DataCopyPad(pf,pg,DataCopyExtParams{1,rows*12,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopyPad(sf,sg,DataCopyExtParams{1,segments*24,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    if constexpr(sizeof(T)==2) {
        Cast(pf,raw,RoundMode::CAST_NONE,64);Cast(sf,raw[64],RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
    }
    TreeTinyExactCore<0>(distanceAddress,indexAddress,rows,segments,sizeof(T)==2);
}
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T>
__aicore__ __attribute__((noinline,aligned(128))) void TreeGeneralExactFallback(
    GM_ADDR pointAddress,GM_ADDR segmentAddress,GM_ADDR distanceAddress,GM_ADDR indexAddress,
    uint32_t segments) {
    PipeBarrier<PIPE_ALL>();SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    LocalTensor<float> work(TPosition::VECCALC,0,25344);
    auto sf=work[12000],point=work[22528],packet=work[22656],temp=work[22800];
    auto lane=work[22912].template ReinterpretCast<int32_t>();
    auto offsets=work[22928].template ReinterpretCast<int32_t>();
    auto pointRaw=work[23104].template ReinterpretCast<T>();
    auto packetRaw=work[23232].template ReinterpretCast<T>();
    GlobalTensor<T> pg,sg,dg;GlobalTensor<int32_t> ig;
    pg.SetGlobalBuffer((__gm__ T*)pointAddress);sg.SetGlobalBuffer((__gm__ T*)segmentAddress);
    dg.SetGlobalBuffer((__gm__ T*)distanceAddress);ig.SetGlobalBuffer((__gm__ int32_t*)indexAddress);
    const uint32_t padded=(6*segments+63)/64*64;
    Duplicate(sf,0.0F,padded+64);Duplicate(pointRaw,T(0),64);
    if constexpr(sizeof(T)==2)Duplicate(work[8496].template ReinterpretCast<T>(),T(0),padded);
    PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    DataCopyPad(pointRaw,pg,DataCopyExtParams{1,3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
    if constexpr(sizeof(T)==2)
        DataCopyPad(work[8496].template ReinterpretCast<T>(),sg,DataCopyExtParams{1,segments*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
    else
        DataCopyPad(sf,sg,DataCopyExtParams{1,segments*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    if constexpr(sizeof(T)==2) {
        Cast(work[23136],pointRaw,RoundMode::CAST_NONE,64);
        Cast(sf,work[8496].template ReinterpretCast<T>(),RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
    }
    auto pointFloat=sizeof(T)==2?work[23136]:work[23104];
    Brcb(temp,pointFloat,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    for(uint32_t axis=0;axis<3;++axis)
        Copy(point[axis*16],temp[axis*8],16,1,CopyRepeatParams(1,0,8,0));
    PipeBarrier<PIPE_V>();
    Copy(sf[padded],point,48,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();

    const uint32_t coords=padded+64;
    auto exponent=work,scratch=work[6208],record=work[8192],reduced=work[8320];
    auto mask=work[7000].template ReinterpretCast<uint8_t>();
    ShiftRight(exponent.template ReinterpretCast<uint32_t>(),sf.template ReinterpretCast<uint32_t>(),uint32_t(23),coords);
    Duplicate(scratch.template ReinterpretCast<uint32_t>(),uint32_t(255),64);PipeBarrier<PIPE_V>();
    for(uint32_t k=0;k<coords;k+=64)
        And(exponent[k].template ReinterpretCast<uint16_t>(),exponent[k].template ReinterpretCast<uint16_t>(),scratch.template ReinterpretCast<uint16_t>(),128);
    PipeBarrier<PIPE_V>();Cast(exponent,exponent.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,coords);PipeBarrier<PIPE_V>();
    Maxs(exponent,exponent,1.0F,coords);PipeBarrier<PIPE_V>();
    Duplicate(record,0.0F,128);PipeBarrier<PIPE_V>();
    WholeReduceMax(record,exponent,64,coords/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,record,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,reduced,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Cast(reduced[8].template ReinterpretCast<int32_t>(),reduced,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t maxExp=uint32_t(reduced[8].template ReinterpretCast<int32_t>().GetValue(0));
    Duplicate(scratch.template ReinterpretCast<uint32_t>(),uint32_t(0x7fffffff),64);PipeBarrier<PIPE_V>();
    for(uint32_t k=0;k<coords;k+=64) {
        And(work[6272].template ReinterpretCast<uint16_t>(),sf[k].template ReinterpretCast<uint16_t>(),scratch.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        CompareScalar(mask,work[6272].template ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        Not(mask.template ReinterpretCast<uint16_t>(),mask.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Select(exponent[k],mask,exponent[k],255.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
    }
    Duplicate(record,255.0F,128);PipeBarrier<PIPE_V>();
    WholeReduceMin(record,exponent,64,coords/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMin(reduced,record,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMin(reduced,reduced,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Cast(reduced[8].template ReinterpretCast<int32_t>(),reduced,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t minExp=uint32_t(reduced[8].template ReinterpretCast<int32_t>().GetValue(0));
    if(minExp>maxExp)minExp=maxExp;
    const uint32_t required=(maxExp-minExp+39)/12,limbs=required<4?4:required;
    TreeFloatRational16 exact(0,limbs,minExp);exact.Init();
    Duplicate(lane,int32_t(0),16);PipeBarrier<PIPE_V>();
    const uint64_t laneMasks[4]={0xaaaa,0xcccc,0xf0f0,0xff00};
    for(uint32_t k=0;k<4;++k) {
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),laneMasks[k]);
        Adds<int32_t,false>(lane,lane,int32_t(1U<<k),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    }
    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
    Muls(offsets,lane,int32_t(24),16);PipeBarrier<PIPE_V>();
    for(uint32_t start=0;start<segments;start+=16) {
        const uint32_t count=segments-start<16?segments-start:16;
        Duplicate(packet,0.0F,96);
        if constexpr(sizeof(T)==2)Duplicate(packetRaw,T(0),96);
        PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        if constexpr(sizeof(T)==2)
            DataCopyPad(packetRaw,sg[start*6],DataCopyExtParams{1,count*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
        else
            DataCopyPad(packet,sg[start*6],DataCopyExtParams{1,count*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if constexpr(sizeof(T)==2){Cast(packet,packetRaw,RoundMode::CAST_NONE,96);PipeBarrier<PIPE_V>();}
        exact.MakeScores(point,packet,offsets.template ReinterpretCast<uint32_t>(),segments,start);
        exact.Tournament(lane.template ReinterpretCast<uint32_t>());exact.Merge(work[20000],start==0);
    }
    exact.WriteDistance(temp,lane.template ReinterpretCast<uint32_t>());
    Cast(temp[16].template ReinterpretCast<int32_t>(),exact.Plane(24*limbs),RoundMode::CAST_RINT,16);PipeBarrier<PIPE_V>();
    LocalTensor<T> out=temp.template ReinterpretCast<T>();
    if constexpr(sizeof(T)==2){out=packetRaw;Cast(out,temp,RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();}
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(dg,out,DataCopyExtParams{1,uint32_t(sizeof(T)),0,0,0});
    DataCopyPad(ig,temp[16].template ReinterpretCast<int32_t>(),DataCopyExtParams{1,4,0,0,0});
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
#endif

template <typename DT_TREE_POINTS, uint32_t SCHEDULE, uint32_t RESIDENT_OWNERS>
 __global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments, GM_ADDR min_distance, GM_ADDR nearest_segment_idx, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    const auto words=reinterpret_cast<__gm__ const uint32_t*>(tiling);
    if constexpr(SCHEDULE==1 && sizeof(DT_TREE_POINTS)==4 && RESIDENT_OWNERS>0
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        if(GetBlockIdx()<words[5]) {
            InitSocState();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            TreeShortPairTile<DT_TREE_POINTS,RESIDENT_OWNERS>(tree_points,line_segments,min_distance,
                nearest_segment_idx,RESIDENT_OWNERS,words[2]);
#endif
        }
    } else if constexpr(SCHEDULE>=25 && SCHEDULE<=32
#ifdef DTYPE_TREE_POINTS
        && sizeof(DT_TREE_POINTS)==sizeof(DTYPE_TREE_POINTS)
#else
        && false
#endif
        ) {
        constexpr uint32_t residentOwners=SCHEDULE-24U;
        if(GetBlockIdx()<residentOwners) {
#if defined(DTYPE_TREE_POINTS)
            TreeTinyOneTile<DT_TREE_POINTS,residentOwners>(tree_points,line_segments,min_distance,nearest_segment_idx);
#endif
        }
    } else if constexpr(SCHEDULE==15 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        if constexpr(RESIDENT_OWNERS>0) {
            constexpr uint32_t pointCount=RESIDENT_OWNERS,owners=(pointCount+3U)/4U;
            if(GetBlockIdx()<owners) {
                // Native packet initializes Vector state after the input DMA is queued.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                TreeFewPointTile<DT_TREE_POINTS,RESIDENT_OWNERS>(tree_points,line_segments,min_distance,nearest_segment_idx,pointCount);
#endif
            }
        } else {
        const uint32_t pointCount=words[1],owners=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        if(GetBlockIdx()<owners) {
            // Native packet initializes Vector state after the input DMA is queued.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            TreeFewPointTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,pointCount);
#endif
        }
        }
    } else if constexpr(SCHEDULE==16
#ifdef DTYPE_TREE_POINTS
        && sizeof(DT_TREE_POINTS)==sizeof(DTYPE_TREE_POINTS)
#else
        && false
#endif
        ) {
        const uint32_t owners=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        if(GetBlockIdx()<owners) {
#if defined(DTYPE_TREE_POINTS)
            TreeTinyOneTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx);
#endif
        }
    } else if constexpr(SCHEDULE==23 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        if constexpr(RESIDENT_OWNERS>0) {
            // A complete single-batch resident packet encodes N and owns four points per core.
            if(GetBlockIdx()<(RESIDENT_OWNERS+3U)/4U) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                TreeNativePairPointTile<DT_TREE_POINTS,RESIDENT_OWNERS>(tree_points,line_segments,min_distance,nearest_segment_idx,RESIDENT_OWNERS);
#endif
            }
        } else {
        const uint32_t pointCount=words[1],owners=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        if(GetBlockIdx()<owners) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            TreeNativePairPointTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,pointCount);
#endif
        }
        }
    } else if constexpr(SCHEDULE==24 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        if constexpr(RESIDENT_OWNERS>0) {
            // The complete single-batch point resource has exactly one owner.
            if(GetBlockIdx()==0) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                TreeBoundedSingleQueryTile<DT_TREE_POINTS,RESIDENT_OWNERS>(tree_points,line_segments,min_distance,nearest_segment_idx,RESIDENT_OWNERS);
#endif
            }
        } else {
        const uint32_t segmentCount=words[2],owners=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        if(GetBlockIdx()<owners) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            TreeBoundedSingleQueryTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,segmentCount);
#endif
        }
        }
    } else {
    TreeLineMinDistanceTilingData tiling_data;
    tiling_data.batches=words[0];tiling_data.points=words[1];
    if constexpr((SCHEDULE==18 || SCHEDULE==19) && sizeof(DT_TREE_POINTS)==2) {
        tiling_data.segments=words[2];tiling_data.paddedSegments=SCHEDULE==18?32U:64U;
        tiling_data.pointTile=SCHEDULE==18?8U:32U;tiling_data.blocks=words[5];
        tiling_data.schedule=SCHEDULE;tiling_data.segmentLog=SCHEDULE==18?5U:6U;
    } else if constexpr(SCHEDULE==17 && sizeof(DT_TREE_POINTS)==4) {
        // The one-tail 192-coefficient resource contains two full 64-lane
        // groups and one live tail. B, N and the active grid remain dynamic.
        tiling_data.segments=2U*64U+1U;tiling_data.paddedSegments=256U;
        tiling_data.pointTile=16U;tiling_data.blocks=words[5];
        tiling_data.schedule=17U;tiling_data.segmentLog=8U;
    } else if constexpr(SCHEDULE==3 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        // Schedule 3 is the host's single-segment domain. These four fields
        // are invariants of that domain for every admitted shape and value.
        tiling_data.segments=1;tiling_data.paddedSegments=1;
        tiling_data.pointTile=words[4];tiling_data.blocks=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        tiling_data.schedule=3;tiling_data.segmentLog=0;
    } else {
        tiling_data.segments=words[2];tiling_data.paddedSegments=words[3];
        tiling_data.pointTile=words[4];tiling_data.blocks=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        tiling_data.schedule=words[6];tiling_data.segmentLog=words[7];
    }
    if (GetBlockIdx() < tiling_data.blocks) {
    if constexpr (SCHEDULE == 21) {
        InitSocState();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        TreeGlobalSpatialCompactProcess(tree_points,line_segments,min_distance,nearest_segment_idx,GetUserWorkspace(workspace),tiling);
        
        
#elif defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        TreeHalfGlobalSpatialProcess(tree_points,line_segments,min_distance,nearest_segment_idx,GetUserWorkspace(workspace),tiling_data);
#endif
    } else if constexpr (SCHEDULE == 7 || SCHEDULE == 10 || SCHEDULE == 12) {
        InitSocState();
        if(tiling_data.paddedSegments==128) {
            KernelTreeBatch<DT_TREE_POINTS,64,128,128,(SCHEDULE==10||SCHEDULE==12),SCHEDULE==10> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else if(tiling_data.paddedSegments==256) {
            if(tiling_data.segments<=192) {
                    if constexpr(sizeof(DT_TREE_POINTS)==4)AscendC::ICachePreLoad(int64_t(6));
                    KernelTreeBatch<DT_TREE_POINTS,64,256,192,(SCHEDULE==10||SCHEDULE==12),SCHEDULE==10> op;
                    op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            } else {
                KernelTreeBatch<DT_TREE_POINTS,64,256,256,(SCHEDULE==10||SCHEDULE==12),SCHEDULE==10> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else {
            KernelTreeBatch<DT_TREE_POINTS,64,512,512,(SCHEDULE==10||SCHEDULE==12),SCHEDULE==10> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else if constexpr (SCHEDULE == 6 || SCHEDULE == 11 || SCHEDULE == 13) {
        InitSocState();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(SCHEDULE==6 && sizeof(DT_TREE_POINTS)==4) {
            // Emit the complete-cache resource before the bounded prefix resource.
            if(tiling_data.segments>576) {
                if(tiling_data.segments==1024 && tiling_data.batches*tiling_data.points/tiling_data.blocks>=1024)
                    TreeSpatialProcess(tree_points,line_segments,min_distance,nearest_segment_idx,GetUserWorkspace(workspace),tiling_data);
                else {
                    KernelTreeBatch<DT_TREE_POINTS,64,1024,1024,false,false> op;
                    op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
                }
            } else {
                KernelTreeBatchPacked<DT_TREE_POINTS,64,1024,576,false,false> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else
#endif
        if constexpr(SCHEDULE==6 && sizeof(DT_TREE_POINTS)==2
#ifdef DTYPE_TREE_POINTS
            && sizeof(DTYPE_TREE_POINTS)==2
#endif
        ) {
            if(tiling_data.segments>576) {
                KernelTreeBatch<DT_TREE_POINTS,64,1024,1024,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            } else {
                KernelTreeBatch<DT_TREE_POINTS,64,1024,576,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else {
        if(tiling_data.segments<=576) {
            if constexpr(sizeof(DT_TREE_POINTS)==4) {
                AscendC::ICachePreLoad(int64_t(6));
                KernelTreeBatchPacked<DT_TREE_POINTS,64,1024,576,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            } else {
                KernelTreeBatch<DT_TREE_POINTS,64,1024,576,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            if constexpr(SCHEDULE==6 && sizeof(DT_TREE_POINTS)==4) {
                if(tiling_data.segments==1024 && tiling_data.batches*tiling_data.points/tiling_data.blocks>=1024)
                    TreeSpatialProcess(tree_points,line_segments,min_distance,nearest_segment_idx,GetUserWorkspace(workspace),tiling_data);
                else {
                    KernelTreeBatch<DT_TREE_POINTS,64,1024,1024,false,false> op;
                    op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
                }
            } else
#endif
            {
            KernelTreeBatch<DT_TREE_POINTS,64,1024,1024,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);
            op.Process();
            }
        }
        }
    } else if constexpr (SCHEDULE == 9) {
        InitSocState();
        KernelTreePairs<DT_TREE_POINTS,2,true> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr (SCHEDULE == 14) {
        InitSocState();
        KernelTreeTiny<DT_TREE_POINTS,0,8,true,true> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr (SCHEDULE == 15) {
        InitSocState();
        KernelTreeTiny<DT_TREE_POINTS,4,1,false,false,true,true> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr (SCHEDULE == 16) {
        InitSocState();
        KernelTreeTiny<DT_TREE_POINTS,0,8,true,true,false,false,2> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr (SCHEDULE == 17) {
        InitSocState();
        if constexpr(sizeof(DT_TREE_POINTS)==4) {
            AscendC::ICachePreLoad(int64_t(4));
            KernelTreeBatch<DT_TREE_POINTS,64,256,192,true,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            KernelTreeTiny<DT_TREE_POINTS,1,8,false,false,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else if constexpr(SCHEDULE==18 && sizeof(DT_TREE_POINTS)==2) {
        InitSocState();
        KernelTreePairs<DT_TREE_POINTS,1,true,8,32> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr(SCHEDULE==19 && sizeof(DT_TREE_POINTS)==2) {
        InitSocState();
        KernelTreePairs<DT_TREE_POINTS,4,true,32,64> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr(SCHEDULE==18 || SCHEDULE==19 || SCHEDULE==22) {
        InitSocState();
        AscendC::ICachePreLoad(int64_t(2));
        KernelTreeBatchPacked<DT_TREE_POINTS,64,1024,576,SCHEDULE!=18,SCHEDULE==19> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr (SCHEDULE == 20) {
        if constexpr(sizeof(DT_TREE_POINTS)==4) {
            InitSocState();
            KernelTreeSinglePointSingleSegment<DT_TREE_POINTS,1,8,false,false,true,false,0,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            InitSocState();
            KernelTreeTiny<DT_TREE_POINTS,1,8,false,false,true,false,0,true,false> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);
            op.Process(RESIDENT_OWNERS==1U?0U:GetBlockIdx());
        }
    } else if constexpr(SCHEDULE==8 && sizeof(DT_TREE_POINTS)==2
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#endif
        ) {
        InitSocState();
        if(tiling_data.paddedSegments==1024U) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
            TreeOneQueryTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data.batches,tiling_data.segments,tiling_data.blocks);
#endif
        } else {
            KernelTreePairs<DT_TREE_POINTS,8> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else if constexpr (SCHEDULE == 5) {
        InitSocState();
        if(tiling_data.pointTile==4) {
            if(tiling_data.segments==1) {
                KernelTreeTiny<DT_TREE_POINTS,4,1,false,false,true> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            } else if(tiling_data.segments==2 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
                      && sizeof(DTYPE_TREE_POINTS)==4
#endif
                      ) {
                KernelTreeTiny<DT_TREE_POINTS,4,2,false,false,false,false,2> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            } else {
                KernelTreeTiny<DT_TREE_POINTS,4,2> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else if(tiling_data.segments==2 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
                  && sizeof(DTYPE_TREE_POINTS)==4
#endif
                  ) {
            TreeTinyPairTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);
        } else if(tiling_data.segments>1) {
            KernelTreeTiny<DT_TREE_POINTS,1,8,true,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            KernelTreeTiny<DT_TREE_POINTS,1,8,false,false,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else if constexpr (SCHEDULE == 1) {
        InitSocState();
        uint32_t tiles;
        if constexpr(sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
                     && sizeof(DTYPE_TREE_POINTS)==4
#endif
                     ) {
            tiles=tiling_data.points<=tiling_data.pointTile?1U:
                (tiling_data.points+tiling_data.pointTile-1)/tiling_data.pointTile;
        } else tiles=(tiling_data.points+tiling_data.pointTile-1)/tiling_data.pointTile;
        if(tiling_data.paddedSegments<=32 && tiling_data.batches*tiles<=tiling_data.blocks) {
            if(tiling_data.pointTile==4 && tiling_data.paddedSegments==32) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                if constexpr(sizeof(DT_TREE_POINTS)==4) {
                    TreeShortPairTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,
                        tiling_data.points,tiling_data.segments);
                } else {
#endif
                KernelTreePairs<DT_TREE_POINTS,1,true,4,32> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                }
#endif
            } else {
                KernelTreePairs<DT_TREE_POINTS,1,true> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else {
            KernelTreePairs<DT_TREE_POINTS,1> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else if constexpr(SCHEDULE==4 && sizeof(DT_TREE_POINTS)==2
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#endif
        ) {
        InitSocState();
        const uint32_t tiles=(tiling_data.points+tiling_data.pointTile-1U)/tiling_data.pointTile;
        if(tiling_data.pointTile==16U && tiling_data.batches*tiles<=tiling_data.blocks) {
            KernelTreePairs<DT_TREE_POINTS,4,true,16,64> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
        if(tiling_data.batches*tiles<=tiling_data.blocks) {
            KernelTreePairs<DT_TREE_POINTS,4,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            KernelTreePairs<DT_TREE_POINTS,4> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
        }
        
        
    } else if constexpr(SCHEDULE==3 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        InitSocState();
        // A bounded1K-query arena makes all UB pitches constant. The class
        // covers a continuous range of query tile sizes, with a dynamic fallback.
        if(tiling_data.pointTile>512U && tiling_data.pointTile<=1024U) {
            const uint32_t tiles=(tiling_data.points+1023U)/1024U;
            const uint32_t owners=tiling_data.batches*tiles;
            if(owners<=tiling_data.blocks) {
                if(GetBlockIdx()<owners) {
                    TreeM1SingleTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data.points);
                }
            } else {
                KernelTreePairs<DT_TREE_POINTS,3,false,1024,1> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else {
            KernelTreePairs<DT_TREE_POINTS,3> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else {
        InitSocState();
        KernelTreePairs<DT_TREE_POINTS, SCHEDULE> op;
        op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, tiling_data);
        op.Process();
    }
}
}
}

#undef TREE_NATIVE_HALF_TOKEN
#undef TREE_NATIVE_HALF_TOKEN_IMPL
#undef TREE_NATIVE_HALF_TOKEN_half
#undef TREE_NATIVE_HALF_TOKEN_float

#undef TREE_NATIVE_FLOAT_TOKEN
#undef TREE_NATIVE_FLOAT_TOKEN_IMPL
#undef TREE_NATIVE_FLOAT_TOKEN_half
#undef TREE_NATIVE_FLOAT_TOKEN_float

#if defined(DTYPE_TREE_POINTS)
template<uint32_t RESERVED>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyExactCore(
    GM_ADDR distanceAddress,GM_ADDR indexAddress,uint32_t rows,uint32_t segments,bool halfOutput) {
    LocalTensor<half> raw(TPosition::VECCALC,100000,128);
    LocalTensor<float> pf(TPosition::VECCALC,100512,64),sf(TPosition::VECCALC,100768,128);
    LocalTensor<float> point(TPosition::VECCALC,101280,48),temp(TPosition::VECCALC,101536,64);
    LocalTensor<int32_t> lanes(TPosition::VECCALC,101792,16),offsets(TPosition::VECCALC,101856,16);
    LocalTensor<float> exponents(TPosition::VECCALC,101920,192),scratch(TPosition::VECCALC,102688,64);
    LocalTensor<float> record(TPosition::VECCALC,102944,64);
    LocalTensor<uint8_t> mask(TPosition::VECCALC,103200,256);
    GlobalTensor<float> dg;GlobalTensor<half> dh;GlobalTensor<int32_t> ig;
    dg.SetGlobalBuffer((__gm__ float*)distanceAddress);dh.SetGlobalBuffer((__gm__ half*)distanceAddress);
    ig.SetGlobalBuffer((__gm__ int32_t*)indexAddress);
    Copy(sf[64],pf,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    ShiftRight(exponents.ReinterpretCast<uint32_t>(),sf.ReinterpretCast<uint32_t>(),uint32_t(23),128);
    Duplicate(scratch.ReinterpretCast<uint32_t>(),uint32_t(255),64);PipeBarrier<PIPE_V>();
    for(uint32_t k=0;k<128;k+=64)And(exponents[k].ReinterpretCast<uint16_t>(),exponents[k].ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),128);
    PipeBarrier<PIPE_V>();Cast(exponents,exponents.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
    Maxs(exponents,exponents,1.0F,128);PipeBarrier<PIPE_V>();
    WholeReduceMax(record,exponents,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMax(temp,record,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Cast(temp[8].ReinterpretCast<int32_t>(),temp,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t maxExp=uint32_t(temp[8].ReinterpretCast<int32_t>().GetValue(0));
    Duplicate(scratch.ReinterpretCast<uint32_t>(),uint32_t(0x7fffffff),64);PipeBarrier<PIPE_V>();
    for(uint32_t k=0;k<128;k+=64) {
        And(temp.ReinterpretCast<uint16_t>(),sf[k].ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        CompareScalar(mask,temp.ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        Not(mask.ReinterpretCast<uint16_t>(),mask.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Select(exponents[k],mask,exponents[k],255.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
    }
    WholeReduceMin(record,exponents,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMin(temp,record,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Cast(temp[8].ReinterpretCast<int32_t>(),temp,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t minExp=uint32_t(temp[8].ReinterpretCast<int32_t>().GetValue(0));
    if(minExp>maxExp)minExp=maxExp;
    const uint32_t required=(maxExp-minExp+39)/12,limbs=required<4?4:required;
    TreeFloatRational16 exact(0,limbs,minExp);exact.Init();
    Duplicate(lanes,int32_t(0),16);PipeBarrier<PIPE_V>();
    const uint64_t bits[4]={0xaaaa,0xcccc,0xf0f0,0xff00};
    for(uint32_t k=0;k<4;++k) {
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),bits[k]);
        Adds<int32_t,false>(lanes,lanes,int32_t(1U<<k),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    }
    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    Muls(offsets,lanes,int32_t(24),16);PipeBarrier<PIPE_V>();
    for(uint32_t row=0;row<rows;++row) {
        for(uint32_t axis=0;axis<3;++axis) {
            const uint32_t pos=row*3+axis;
            Brcb(temp,pf[(pos/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Copy(point[axis*16],temp[(pos%8)*8],16,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
        }
        exact.MakeScores(point,sf,offsets.ReinterpretCast<uint32_t>(),segments);
        exact.Tournament(lanes.ReinterpretCast<uint32_t>());
        exact.WriteDistance(temp,lanes.ReinterpretCast<uint32_t>());
        Cast(temp[16].ReinterpretCast<int32_t>(),exact.Plane(24*limbs),RoundMode::CAST_RINT,16);PipeBarrier<PIPE_V>();
        if(halfOutput)Cast(raw,temp,RoundMode::CAST_RINT,1);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if(halfOutput)DataCopyPad(dh[row],raw,DataCopyExtParams{1,2,0,0,0});
        else DataCopyPad(dg[row],temp,DataCopyExtParams{1,4,0,0,0});
        DataCopyPad(ig[row],temp[16].ReinterpretCast<int32_t>(),DataCopyExtParams{1,4,0,0,0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}
#endif
