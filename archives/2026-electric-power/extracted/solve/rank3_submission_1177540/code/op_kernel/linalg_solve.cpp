#include "kernel_operator.h"
#include "linalg_solve_tiling.h"
#ifndef SOLVE_DIRECT_BENCH
#include "tiling_key_linalg_solve.h"
#endif
using namespace AscendC;

constexpr uint32_t kRowsPerRegion = 512;
constexpr uint32_t kScratchRegions = 32;
constexpr uint32_t kMaxPaddedRows = 512;
// Columns moved by one DataCopy pair. Consecutive columns are contiguous in the
// workspace, so a panel is a single copy; the per-column work stays identical.
constexpr uint32_t kPanel = 8;
// Above this order the strided input copy is cheaper than n scalar global reads.
constexpr uint32_t kInputDmaThreshold = 224;
static_assert(kMaxPaddedRows % 8 == 0, "padded order must keep 8-float alignment");
static_assert(kRowsPerRegion >= kMaxPaddedRows, "region cannot hold a padded row");
static_assert(kPanel >= 2, "panel batching needs at least two columns");
static_assert(2*kPanel*kMaxPaddedRows*sizeof(float) <= 64*1024,
        "panel buffers must stay inside the UB budget");
// Region map (all regions hold kRowsPerRegion floats; rows <= kMaxPaddedRows).
//   0 qh   1 ql   2 ah   3 al   4 dh   5 dl   6 ph   7 pl
//   8 ah2  9 al2 10 dh2 11 dl2 12 ph2 13 pl2
//  14 xh  15 xl  16..23 s[0..7]  24..29 s2[0..5]  30 q1  31 q2
static_assert(kScratchRegions == 32, "region map above must match the allocation in Run");

// A number is hi+lo. QR and back substitution both run on the device.
struct SolvePair {float hi,lo;};
class SolveKernel {
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuf,packBuf,panelBuf;
    LocalTensor<float> qh,ql,ah,al,dh,dl,ph,pl,q1,q2,xh,xl,s[8],packed;
    // Second lane for the two-column interleaved path. Disjoint buffers, same
    // arithmetic per column, issued alternately so the dependent chains of the
    // two columns can overlap instead of serialising on one instruction stream.
    LocalTensor<float> ah2,al2,dh2,dl2,ph2,pl2,s2[6];
    // Panel of kPanel consecutive columns, loaded and stored with one copy each.
    LocalTensor<float> panelH,panelL;
    GlobalTensor<float> inputA,inputB,outputX,high,low,columnSolutions;
    LinalgSolveTilingData t;
    uint32_t rhsRows;
    template<HardEvent event> __aicore__ inline void Sync() {
        const auto id=pipe.FetchEventID(event);SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void WriteRows(uint32_t batch) {
        Sync<HardEvent::MTE3_S>();
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(columnSolutions);
        for(uint32_t row=0;row<t.n;++row) {
            for(uint32_t rhs=0;rhs<t.k;++rhs)ph.SetValue(rhs,columnSolutions.GetValue((uint64_t)rhs*t.n+row));
            Sync<HardEvent::S_MTE3>();
            DataCopyPad(outputX[(uint64_t)batch*t.n*t.k+row*t.k],ph,DataCopyExtParams{1,t.k*4,0,0,0});
            Sync<HardEvent::MTE3_S>();
        }
    }
    __aicore__ inline SolvePair AddPair(SolvePair a,SolvePair b) {
        // Scalar pipe: same rounding sequence as the vector form on one element.
        return AddScalarPair(a,b);
    }
    // Same rounding sequence as AddVector(...,1), executed on the scalar pipe:
    // no vector instruction is issued, so the reduction tail costs no PIPE_ALL.
    static __aicore__ inline SolvePair AddScalarPair(SolvePair a,SolvePair b) {
        const float s0=a.hi+b.hi;
        const float s1=s0-a.hi;
        const float t0=s0-s1;
        const float e0=a.hi-t0;
        const float t1=b.hi-s1;
        float e=e0+t1;
        e=e+a.lo;
        e=e+b.lo;
        const float hi=s0+e;
        const float u0=hi-s0;
        const float u1=hi-u0;
        const float u2=s0-u1;
        const float u3=e-u0;
        return {hi,u2+u3};
    }
    static __aicore__ inline SolvePair Neg(SolvePair a) {return {-a.hi,-a.lo};}
    // Dekker split of a scalar. volatile keeps each rounding separate so the
    // compiler cannot contract the sequence into an FMA.
    static __aicore__ inline void SplitScalar(float c,float &c1,float &c2) {
        volatile float m=4097.0f*c; const float s=m;
        const float t=s-c;
        volatile float n=s-t; c1=n;
        c2=c-c1;
    }
    // Two-product via Dekker split on the scalar pipe. Every multiplication is
    // forced through volatile so the compiler cannot fuse it into an FMA: the
    // split depends on each product being rounded separately.
    static __aicore__ inline SolvePair Product(float a,float b) {
        volatile float m;
        m=4097.0f*a; const float c1=m;
        const float t1=c1-a;
        const float a1=c1-t1;
        const float a2=a-a1;
        m=4097.0f*b; const float c2=m;
        const float t2=c2-b;
        const float b1=c2-t2;
        const float b2=b-b1;
        m=a*b; const float p=m;
        m=a1*b1; const float p11=m;
        const float d1=p11-p;
        m=a1*b2; const float p12=m;
        const float s1=d1+p12;
        m=a2*b1; const float p21=m;
        const float s2=s1+p21;
        m=a2*b2; const float p22=m;
        const float e=s2+p22;
        return {p,e};
    }
    __aicore__ inline SolvePair MulPair(SolvePair a,SolvePair b) {
        SolvePair p=Product(a.hi,b.hi);
        return AddPair(p,{a.hi*b.lo+a.lo*b.hi,a.lo*b.lo});
    }
    __aicore__ inline SolvePair DivPair(SolvePair a,SolvePair b) {
        float h=a.hi/b.hi;
        SolvePair remainder=AddPair(a,Neg(MulPair(b,{h,0})));
        return AddPair({h,0},{(remainder.hi+remainder.lo)/b.hi,0});
    }
    __aicore__ inline SolvePair Root(SolvePair a) {
        s[7].SetValue(0,a.hi);PipeBarrier<PIPE_ALL>();
        Sqrt(s[7],s[7],1);PipeBarrier<PIPE_ALL>();
        const float h=s[7].GetValue(0);
        if(h==0)return {0,0};
        const SolvePair r=AddPair(a,Neg(Product(h,h)));
        return AddPair({h,0},{(r.hi+r.lo)/(2*h),0});
    }
    __aicore__ inline void VAdd(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Add(o,a,b,n);PipeBarrier<PIPE_V>();}
    __aicore__ inline void VSub(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Sub(o,a,b,n);PipeBarrier<PIPE_V>();}
    __aicore__ inline void VMul(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Mul(o,a,b,n);PipeBarrier<PIPE_V>();}
    __aicore__ inline void VScale(LocalTensor<float> o,LocalTensor<float> a,float b,uint32_t n) {Muls(o,a,b,n);PipeBarrier<PIPE_V>();}
    __aicore__ inline void Fill(LocalTensor<float> o,float v,uint32_t n) {Duplicate(o,v,n);PipeBarrier<PIPE_ALL>();}
    // In-place output on a is safe; scratch never overlaps an input or output.
    __aicore__ inline void AddVector(LocalTensor<float> oh,LocalTensor<float> ol,
            LocalTensor<float> a,LocalTensor<float> ae,LocalTensor<float> b,LocalTensor<float> be,uint32_t n) {
        // Knuth two-sum in 6 ops, then fold both low parts and renormalise once.
        VAdd(s[0],a,b,n);
        VSub(s[1],s[0],a,n);VSub(s[2],s[0],s[1],n);
        VSub(s[3],a,s[2],n);VSub(s[4],b,s[1],n);VAdd(s[3],s[3],s[4],n);
        VAdd(s[3],s[3],ae,n);VAdd(s[3],s[3],be,n);
        VAdd(oh,s[0],s[3],n);
        VSub(s[4],s[0],oh,n);VAdd(ol,s[4],s[3],n);
    }
    __aicore__ inline void SplitA(LocalTensor<float> a1,LocalTensor<float> a2,
            LocalTensor<float> a,uint32_t n) {
        VScale(a1,a,4097.0f,n);VSub(a2,a1,a,n);VSub(a1,a1,a2,n);VSub(a2,a,a1,n);
    }
    __aicore__ inline void MulVector(LocalTensor<float> oh,LocalTensor<float> ol,
            LocalTensor<float> a,LocalTensor<float> ae,LocalTensor<float> b,LocalTensor<float> be,uint32_t n) {
        VMul(oh,a,b,n);
        VScale(s[0],a,4097.0f,n);VSub(s[1],s[0],a,n);VSub(s[0],s[0],s[1],n);VSub(s[1],a,s[0],n);
        VScale(s[2],b,4097.0f,n);VSub(s[3],s[2],b,n);VSub(s[2],s[2],s[3],n);VSub(s[3],b,s[2],n);
        VMul(s[4],s[0],s[2],n);VSub(s[4],s[4],oh,n);
        VMul(s[5],s[0],s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],s[1],s[2],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],s[1],s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],a,be,n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],ae,b,n);VAdd(ol,s[4],s[5],n);
    }
    // double-float vector times double-float scalar, with `a` split already known.
    // Term order matches MulVectorPre, so the result is bit-identical.
    __aicore__ inline void MulVectorScalar(LocalTensor<float> oh,LocalTensor<float> ol,
            LocalTensor<float> a,LocalTensor<float> ae,LocalTensor<float> a1,LocalTensor<float> a2,
            float c,float ce,uint32_t n) {
        float c1,c2;SplitScalar(c,c1,c2);
        VScale(oh,a,c,n);
        VScale(s[4],a1,c1,n);VSub(s[4],s[4],oh,n);
        VScale(s[5],a1,c2,n);VAdd(s[4],s[4],s[5],n);
        VScale(s[5],a2,c1,n);VAdd(s[4],s[4],s[5],n);
        VScale(s[5],a2,c2,n);VAdd(s[4],s[4],s[5],n);
        VScale(s[5],a,ce,n);VAdd(s[4],s[4],s[5],n);
        VScale(s[5],ae,c,n);VAdd(ol,s[4],s[5],n);
    }
    __aicore__ inline void MulVectorPre(LocalTensor<float> oh,LocalTensor<float> ol,
            LocalTensor<float> a,LocalTensor<float> ae,LocalTensor<float> b,LocalTensor<float> be,
            LocalTensor<float> a1,LocalTensor<float> a2,uint32_t n) {
        VMul(oh,a,b,n);
        VScale(s[2],b,4097.0f,n);VSub(s[3],s[2],b,n);VSub(s[2],s[2],s[3],n);VSub(s[3],b,s[2],n);
        VMul(s[4],a1,s[2],n);VSub(s[4],s[4],oh,n);
        VMul(s[5],a1,s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],a2,s[2],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],a2,s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],a,be,n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],ae,b,n);VAdd(ol,s[4],s[5],n);
    }
    // ---- two-column interleaved path -------------------------------------
    // Identical arithmetic and identical per-column operation order, but the
    // statements of lane 0 and lane 1 are issued alternately. A PIPE_V barrier
    // inside a lane would put both chains back in series, so the NB variants
    // below omit it; ordering is preserved by construction (each lane only
    // touches its own buffers, qh/ql/q1/q2 are read-only here).
    __aicore__ inline void VAddNB(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Add(o,a,b,n);}
    __aicore__ inline void VSubNB(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Sub(o,a,b,n);}
    __aicore__ inline void VMulNB(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Mul(o,a,b,n);}
    __aicore__ inline void VScaleNB(LocalTensor<float> o,LocalTensor<float> a,float b,uint32_t n) {Muls(o,a,b,n);}
    // Same 11-op compensated add as AddVector, for both lanes, interleaved.
    __aicore__ inline void AddVec2(LocalTensor<float> o0,LocalTensor<float> l0,
            LocalTensor<float> a0,LocalTensor<float> ae0,LocalTensor<float> b0,LocalTensor<float> be0,
            LocalTensor<float> o1,LocalTensor<float> l1,
            LocalTensor<float> a1,LocalTensor<float> ae1,LocalTensor<float> b1,LocalTensor<float> be1,
            uint32_t n) {
        VAddNB(s[0],a0,b0,n);VAddNB(s2[0],a1,b1,n);
        VSubNB(s[1],s[0],a0,n);VSubNB(s2[1],s2[0],a1,n);
        VSubNB(s[2],s[0],s[1],n);VSubNB(s2[2],s2[0],s2[1],n);
        VSubNB(s[3],a0,s[2],n);VSubNB(s2[3],a1,s2[2],n);
        VSubNB(s[4],b0,s[1],n);VSubNB(s2[4],b1,s2[1],n);
        VAddNB(s[3],s[3],s[4],n);VAddNB(s2[3],s2[3],s2[4],n);
        VAddNB(s[3],s[3],ae0,n);VAddNB(s2[3],s2[3],ae1,n);
        VAddNB(s[3],s[3],be0,n);VAddNB(s2[3],s2[3],be1,n);
        VAddNB(o0,s[0],s[3],n);VAddNB(o1,s2[0],s2[3],n);
        VSubNB(s[4],s[0],o0,n);VSubNB(s2[4],s2[0],o1,n);
        VAddNB(l0,s[4],s[3],n);VAddNB(l1,s2[4],s2[3],n);
    }
    // Same 17-op double-double product as MulVectorPre, for both lanes.
    __aicore__ inline void MulVecPre2(LocalTensor<float> o0,LocalTensor<float> l0,
            LocalTensor<float> b0,LocalTensor<float> be0,
            LocalTensor<float> o1,LocalTensor<float> l1,
            LocalTensor<float> b1,LocalTensor<float> be1,uint32_t n) {
        VMulNB(o0,qh,b0,n);VMulNB(o1,qh,b1,n);
        VScaleNB(s[2],b0,4097.0f,n);VScaleNB(s2[2],b1,4097.0f,n);
        VSubNB(s[3],s[2],b0,n);VSubNB(s2[3],s2[2],b1,n);
        VSubNB(s[2],s[2],s[3],n);VSubNB(s2[2],s2[2],s2[3],n);
        VSubNB(s[3],b0,s[2],n);VSubNB(s2[3],b1,s2[2],n);
        VMulNB(s[4],q1,s[2],n);VMulNB(s2[4],q1,s2[2],n);
        VSubNB(s[4],s[4],o0,n);VSubNB(s2[4],s2[4],o1,n);
        VMulNB(s[5],q1,s[3],n);VMulNB(s2[5],q1,s2[3],n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VMulNB(s[5],q2,s[2],n);VMulNB(s2[5],q2,s2[2],n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VMulNB(s[5],q2,s[3],n);VMulNB(s2[5],q2,s2[3],n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VMulNB(s[5],qh,be0,n);VMulNB(s2[5],qh,be1,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VMulNB(s[5],ql,b0,n);VMulNB(s2[5],ql,b1,n);
        VAddNB(l0,s[4],s[5],n);VAddNB(l1,s2[4],s2[5],n);
    }
    // Same 13-op scalar-multiplied product as MulVectorScalar, per-lane scalar.
    __aicore__ inline void MulVecScalar2(float c,float ce,float cL,float ceL,uint32_t n) {
        float c1,c2,c1L,c2L;
        SplitScalar(c,c1,c2);SplitScalar(cL,c1L,c2L);
        VScaleNB(dh,qh,c,n);VScaleNB(dh2,qh,cL,n);
        VScaleNB(s[4],q1,c1,n);VScaleNB(s2[4],q1,c1L,n);
        VSubNB(s[4],s[4],dh,n);VSubNB(s2[4],s2[4],dh2,n);
        VScaleNB(s[5],q1,c2,n);VScaleNB(s2[5],q1,c2L,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VScaleNB(s[5],q2,c1,n);VScaleNB(s2[5],q2,c1L,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VScaleNB(s[5],q2,c2,n);VScaleNB(s2[5],q2,c2L,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VScaleNB(s[5],qh,ce,n);VScaleNB(s2[5],qh,ceL,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VScaleNB(s[5],ql,c,n);VScaleNB(s2[5],ql,cL,n);
        VAddNB(dl,s[4],s[5],n);VAddNB(dl2,s2[4],s2[5],n);
    }
    // Balanced tail shared by the sequential and interleaved paths.
    static __aicore__ inline SolvePair Tail8(LocalTensor<float> h,LocalTensor<float> l,uint32_t n) {
        if(n!=8) {
            SolvePair total{0,0};
            for(uint32_t i=0;i<n;++i)total=AddScalarPair(total,{h.GetValue(i),l.GetValue(i)});
            return total;
        }
        const SolvePair p0{h.GetValue(0),l.GetValue(0)};
        const SolvePair p1{h.GetValue(1),l.GetValue(1)};
        const SolvePair p2{h.GetValue(2),l.GetValue(2)};
        const SolvePair p3{h.GetValue(3),l.GetValue(3)};
        const SolvePair p4{h.GetValue(4),l.GetValue(4)};
        const SolvePair p5{h.GetValue(5),l.GetValue(5)};
        const SolvePair p6{h.GetValue(6),l.GetValue(6)};
        const SolvePair p7{h.GetValue(7),l.GetValue(7)};
        const SolvePair s0=AddScalarPair(p0,p1);
        const SolvePair s1=AddScalarPair(p2,p3);
        const SolvePair s2v=AddScalarPair(p4,p5);
        const SolvePair s3v=AddScalarPair(p6,p7);
        return AddScalarPair(AddScalarPair(s0,s1),AddScalarPair(s2v,s3v));
    }
    __aicore__ inline void DotPre2(SolvePair &r0,SolvePair &r1) {
        uint32_t n=t.rows;
        MulVecPre2(ph,pl,ah,al,ph2,pl2,ah2,al2,n);
        // Tree stops at 8 elements: smaller offsets would violate the 32-byte
        // alignment that UB vector access requires (n=4 -> 16 B, n=2 -> 8 B).
        while(n>8) {n/=2;AddVec2(ph,pl,ph,pl,ph[n],pl[n],ph2,pl2,ph2,pl2,ph2[n],pl2[n],n);}
        PipeBarrier<PIPE_ALL>();
        r0=Tail8(ph,pl,n);r1=Tail8(ph2,pl2,n);
    }
    __aicore__ inline void Transform2(uint32_t col0,uint32_t col1,SolvePair tau) {
        DataCopy(ah,high[(uint64_t)col0*t.rows],t.rows);DataCopy(al,low[(uint64_t)col0*t.rows],t.rows);
        DataCopy(ah2,high[(uint64_t)col1*t.rows],t.rows);DataCopy(al2,low[(uint64_t)col1*t.rows],t.rows);
        PipeBarrier<PIPE_ALL>();
        SolvePair d0,d1;DotPre2(d0,d1);
        const SolvePair f0=MulPair(tau,d0);
        const SolvePair f1=MulPair(tau,d1);
        MulVecScalar2(-f0.hi,-f0.lo,-f1.hi,-f1.lo,t.rows);
        AddVec2(ph,pl,ah,al,dh,dl,ph2,pl2,ah2,al2,dh2,dl2,t.rows);
        PipeBarrier<PIPE_ALL>();
        DataCopy(high[(uint64_t)col0*t.rows],ph,t.rows);DataCopy(low[(uint64_t)col0*t.rows],pl,t.rows);
        DataCopy(high[(uint64_t)col1*t.rows],ph2,t.rows);DataCopy(low[(uint64_t)col1*t.rows],pl2,t.rows);
        PipeBarrier<PIPE_ALL>();
    }
    // kPanel consecutive columns per copy pair. The workspace stores columns
    // back to back, so the panel is contiguous; only the number of DataCopy and
    // barrier instances changes, never the per-column arithmetic.
    __aicore__ inline void TransformPanel(uint32_t start,SolvePair tau) {
        DataCopy(panelH,high[(uint64_t)start*t.rows],kPanel*t.rows);
        DataCopy(panelL,low[(uint64_t)start*t.rows],kPanel*t.rows);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t p=0;p<kPanel;++p) {
            LocalTensor<float> h=panelH[p*t.rows],l=panelL[p*t.rows];
            const SolvePair d=DotPre(qh,ql,h,l,q1,q2,t.rows);
            const SolvePair f=MulPair(tau,d);
            MulVectorScalar(dh,dl,qh,ql,q1,q2,-f.hi,-f.lo,t.rows);
            AddVector(h,l,h,l,dh,dl,t.rows);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(high[(uint64_t)start*t.rows],panelH,kPanel*t.rows);
        DataCopy(low[(uint64_t)start*t.rows],panelL,kPanel*t.rows);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline SolvePair Dot(LocalTensor<float> a,LocalTensor<float> ae,
            LocalTensor<float> b,LocalTensor<float> be,uint32_t n) {
        MulVector(ph,pl,a,ae,b,be,n);
        // Tree stops at 8 elements: smaller offsets would violate the 32-byte
        // alignment that UB vector access requires (n=4 -> 16 B, n=2 -> 8 B).
        static_assert(kRowsPerRegion >= kMaxPaddedRows, "region cannot hold a padded row");
        while(n>8) {n/=2;AddVector(ph,pl,ph,pl,ph[n],pl[n],n);}
        // Measured: one PIPE_ALL here beats V_S + S_V by ~0.6% on large cases, so
        // the full drain stays (see interleaved-lanes record).
        PipeBarrier<PIPE_ALL>();
        return Tail8(ph,pl,n);
    }
    __aicore__ inline SolvePair DotPre(LocalTensor<float> a,LocalTensor<float> ae,
            LocalTensor<float> b,LocalTensor<float> be,
            LocalTensor<float> a1,LocalTensor<float> a2,uint32_t n) {
        MulVectorPre(ph,pl,a,ae,b,be,a1,a2,n);
        // Tree stops at 8 elements: smaller offsets would violate the 32-byte
        // alignment that UB vector access requires (n=4 -> 16 B, n=2 -> 8 B).
        static_assert(kRowsPerRegion >= kMaxPaddedRows, "region cannot hold a padded row");
        while(n>8) {n/=2;AddVector(ph,pl,ph,pl,ph[n],pl[n],n);}
        PipeBarrier<PIPE_ALL>();
        return Tail8(ph,pl,n);
    }
    __aicore__ inline void Load(uint32_t col,LocalTensor<float> h,LocalTensor<float> l) {
        DataCopy(h,high[(uint64_t)col*t.rows],t.rows);DataCopy(l,low[(uint64_t)col*t.rows],t.rows);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Store(uint32_t col,LocalTensor<float> h,LocalTensor<float> l) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(high[(uint64_t)col*t.rows],h,t.rows);DataCopy(low[(uint64_t)col*t.rows],l,t.rows);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void InputColumn(GlobalTensor<float> input,uint64_t base,uint32_t width,uint32_t col) {
        Fill(ah,0,t.rows);Fill(al,0,t.rows);
        // Measured crossover: at n=256 one strided copy beats n global scalar reads
        // by ~7%, while at n<=200 the extra barrier costs more than it saves. The
        // threshold sits above the largest measured loser, so it can only help.
        if(t.n>=kInputDmaThreshold) {
            DataCopyPad(packed,input[base+col],DataCopyExtParams{(uint16_t)t.n,4,(width-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            for(uint32_t row=0;row<t.n;++row)ah.SetValue(row,packed.GetValue(row*8));
        } else {
            for(uint32_t row=0;row<t.n;++row)ah.SetValue(row,input.GetValue(base+(uint64_t)row*width+col));
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Transform(uint32_t col,SolvePair tau) {
        Load(col,ah,al);const SolvePair factor=MulPair(tau,DotPre(qh,ql,ah,al,q1,q2,t.rows));
        MulVectorScalar(dh,dl,qh,ql,q1,q2,-factor.hi,-factor.lo,t.rows);
        AddVector(ph,pl,ah,al,dh,dl,t.rows);Store(col,ph,pl);
    }
    __aicore__ inline void Solve(uint32_t rhs) {
        Load(t.n+rhs,ah,al);
        Load(t.n+rhs,qh,ql);
        Fill(xh,0,rhsRows);Fill(xl,0,rhsRows);
        // Rows only replace [0,n); the complement never changes, so it is zeroed
        // once per right-hand side instead of on every row.
        Fill(ah,0,rhsRows);Fill(al,0,rhsRows);
        for(int32_t row=t.n-1;row>=0;--row) {
            DataCopyPad(packed,high[row],DataCopyExtParams{(uint16_t)t.n,4,(t.rows-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<t.n;++j)ah.SetValue(j,packed.GetValue(j*8));
            DataCopyPad(packed,low[row],DataCopyExtParams{(uint16_t)t.n,4,(t.rows-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<t.n;++j)al.SetValue(j,packed.GetValue(j*8));
            PipeBarrier<PIPE_ALL>();
            const SolvePair diag{ah.GetValue(row),al.GetValue(row)};
            const SolvePair value{qh.GetValue(row),ql.GetValue(row)};
            const SolvePair result=DivPair(AddPair(value,Neg(Dot(ah,al,xh,xl,rhsRows))),diag);
            xh.SetValue(row,result.hi);xl.SetValue(row,result.lo);PipeBarrier<PIPE_ALL>();
        }
        Add(ph,xh,xl,t.n);PipeBarrier<PIPE_V>();Sync<HardEvent::V_MTE3>();
        DataCopyPad(columnSolutions[(uint64_t)rhs*t.n],ph,DataCopyExtParams{1,t.n*4,0,0,0});
        Sync<HardEvent::MTE3_S>();
    }
public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR workspace,const LinalgSolveTilingData &data) {
        t=data;rhsRows=8;while(rhsRows<t.n)rhsRows*=2;
        pipe.InitBuffer(dataBuf,kScratchRegions*kRowsPerRegion*sizeof(float));pipe.InitBuffer(packBuf,8*1024*4);
        pipe.InitBuffer(panelBuf,2*kPanel*kMaxPaddedRows*sizeof(float));
        auto all=dataBuf.Get<float>();
        // Offsets follow the region map asserted at the top of this file.
        qh=all[0];ql=all[1*kRowsPerRegion];ah=all[2*kRowsPerRegion];al=all[3*kRowsPerRegion];
        dh=all[4*kRowsPerRegion];dl=all[5*kRowsPerRegion];ph=all[6*kRowsPerRegion];pl=all[7*kRowsPerRegion];
        ah2=all[8*kRowsPerRegion];al2=all[9*kRowsPerRegion];
        dh2=all[10*kRowsPerRegion];dl2=all[11*kRowsPerRegion];
        ph2=all[12*kRowsPerRegion];pl2=all[13*kRowsPerRegion];
        xh=all[14*kRowsPerRegion];xl=all[15*kRowsPerRegion];
        for(uint32_t i=0;i<8;++i)s[i]=all[(16+i)*kRowsPerRegion];
        for(uint32_t i=0;i<6;++i)s2[i]=all[(24+i)*kRowsPerRegion];
        q1=all[30*kRowsPerRegion];q2=all[31*kRowsPerRegion];
        packed=packBuf.Get<float>();
        auto panels=panelBuf.Get<float>();
        panelH=panels;panelL=panels[kPanel*kMaxPaddedRows];
        inputA.SetGlobalBuffer((__gm__ float*)a);inputB.SetGlobalBuffer((__gm__ float*)b);
        outputX.SetGlobalBuffer((__gm__ float*)x);
        const uint64_t words=(uint64_t)(t.n+t.k)*t.rows;
        high.SetGlobalBuffer((__gm__ float*)workspace+GetBlockIdx()*2*words);
        low.SetGlobalBuffer((__gm__ float*)workspace+(GetBlockIdx()*2+1)*words);
        columnSolutions.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.blocks*2*words+(uint64_t)GetBlockIdx()*((t.n*t.k+7)/8*8));
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            float largest=0;
            for(uint32_t col=0;col<t.n;++col) {
                InputColumn(inputA,(uint64_t)batch*t.n*t.n,t.n,col);
                for(uint32_t row=0;row<t.n;++row){float v=ah.GetValue(row);v=v<0?-v:v;if(v>largest)largest=v;}
                Store(col,ah,al);
            }
            union {float value;uint32_t bits;} exponent;exponent.value=largest;
            uint32_t power=254-((exponent.bits>>23)&255);if(power>253)power=253;if(power<1)power=1;
            exponent.bits=power<<23;const float scale=exponent.value;
            for(uint32_t col=0;col<t.n;++col){Load(col,ah,al);VScale(ah,ah,scale,t.rows);Store(col,ah,al);}
            for(uint32_t rhs=0;rhs<t.k;++rhs){InputColumn(inputB,(uint64_t)(t.broadcastB?0:batch)*t.n*t.k,t.k,rhs);VScale(ah,ah,scale,t.rows);Store(t.n+rhs,ah,al);}
            for(uint32_t j=0;j<t.n;++j) {
                Load(j,qh,ql);
                for(uint32_t i=0;i<j;++i){qh.SetValue(i,0);ql.SetValue(i,0);}
                PipeBarrier<PIPE_ALL>();
                SolvePair alpha=Root(Dot(qh,ql,qh,ql,t.rows));
                if(qh.GetValue(j)>=0)alpha=Neg(alpha);
                const SolvePair v=AddPair({qh.GetValue(j),ql.GetValue(j)},Neg(alpha));
                qh.SetValue(j,v.hi);ql.SetValue(j,v.lo);PipeBarrier<PIPE_ALL>();
                SplitA(q1,q2,qh,t.rows);
                const SolvePair normal=DotPre(qh,ql,qh,ql,q1,q2,t.rows);
                const SolvePair tau=DivPair({2,0},normal);
                // Widest batching first, then pairs, then a single column. No column
                // is skipped because each step advances by exactly the batch width.
                uint32_t col=j+1,end=t.n+t.k;
                for(;col+kPanel<=end;col+=kPanel)TransformPanel(col,tau);
                for(;col+1<end;col+=2)Transform2(col,col+1,tau);
                if(col<end)Transform(col,tau);
                Load(j,ah,al);ah.SetValue(j,alpha.hi);al.SetValue(j,alpha.lo);
                for(uint32_t i=j+1;i<t.rows;++i){ah.SetValue(i,0);al.SetValue(i,0);}
                Store(j,ah,al);
            }
            for(uint32_t rhs=0;rhs<t.k;++rhs)Solve(rhs);
            WriteRows(batch);
        }
    }
};
// Single-RHS 16-row capacity: contiguous input, column elimination, deferred scaling.
class TinyColumns {
    template<uint32_t Count>
    __aicore__ inline void StaticIndex(LocalTensor<int32_t> dst) {
        static_assert(Count==24 || Count==256);
        // Match dav_c220 CreateVecIndexCalc for these two aligned counts.
        // Its initial eight scalar stores need S->V, but no TPipe lookup.
        for(int32_t i=0;i<8;++i)dst.SetValue(i,static_cast<float>(i));
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        UnaryRepeatParams params;
        constexpr uint32_t first=Count<64?Count:64;
        for(uint32_t i=0;i<first/8-1;++i){
            Adds(dst[(i+1)*8],dst[i*8],(int32_t)8,(uint64_t)8,1,params);PipeBarrier<PIPE_V>();
        }
        if constexpr(Count>64)for(uint32_t i=0;i<Count/64-1;++i){
            Adds(dst[(i+1)*64],dst[i*64],(int32_t)64,(uint64_t)64,1,params);PipeBarrier<PIPE_V>();
        }
    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        constexpr uint32_t P=16,W=24;
        LocalTensor<float> mat(TPosition::VECCALC,0,2048);
        auto input=mat[P*W],pivot=input[P*P],candidates=pivot[W],maximum=candidates[P];
        auto scales=maximum[64],weights=scales[P];auto offsets=weights[8*W].ReinterpretCast<int32_t>();
        auto transposeIx=offsets[W],tempIx=transposeIx[P*P];auto output=tempIx[P*P].ReinterpretCast<float>();auto outputIx=output[P].ReinterpretCast<uint32_t>();
        GlobalTensor<float> ga,gb,gx;ga.SetGlobalBuffer((__gm__ float*)aa);gb.SetGlobalBuffer((__gm__ float*)bb);gx.SetGlobalBuffer((__gm__ float*)xx);
        auto transposeSrc=transposeIx.ReinterpretCast<uint64_t>();
        auto transposeDst=transposeIx[32].ReinterpretCast<uint64_t>();
        if(t.n==16) {
            constexpr uint64_t broadcast4=0x0001000100010001ULL;
            const uint64_t srcBase=((uint64_t)input.GetPhyAddr()>>5)&0x1fff;
            const uint64_t dstBase=((uint64_t)mat.GetPhyAddr()>>5)&0x1fff;
            for(uint32_t q=0;q<4;++q) {
                transposeSrc.SetValue(q,(srcBase+q*8)*broadcast4+(2ULL<<16)+(4ULL<<32)+(6ULL<<48));
                transposeDst.SetValue(q,(dstBase+q*4)*broadcast4+(1ULL<<16)+(2ULL<<32)+(3ULL<<48));
            }
            SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        } else {
        StaticIndex<P*P>(transposeIx);PipeBarrier<PIPE_V>();ShiftRight(tempIx,transposeIx,(int32_t)4,P*P);PipeBarrier<PIPE_V>();
        Muls(transposeIx,transposeIx,(int32_t)t.n,P*P);Muls(tempIx,tempIx,(int32_t)(t.n*P-1),P*P);PipeBarrier<PIPE_V>();Sub(transposeIx,transposeIx,tempIx,P*P);PipeBarrier<PIPE_V>();Muls(transposeIx,transposeIx,(int32_t)4,P*P);PipeBarrier<PIPE_V>();
        }
        StaticIndex<W>(offsets);PipeBarrier<PIPE_V>();Muls(offsets,offsets,(int32_t)(P*4),W);PipeBarrier<PIPE_ALL>();
        constexpr uint32_t vs=0;
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            Duplicate(input,0.0f,P*P);PipeBarrier<PIPE_ALL>();
            DataCopyPad(input,ga[(uint64_t)batch*t.n*t.n],DataCopyExtParams{1,t.n*t.n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            DataCopyPad(mat[t.n*P],gb[(uint64_t)(t.broadcastB?0:batch)*t.n],DataCopyExtParams{1,t.n*4,0,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-t.n%8)%8),0});PipeBarrier<PIPE_ALL>();
            if(t.n==16) {
                TransDataTo5HDParams params;params.dstHighHalf=false;params.srcHighHalf=false;
                params.repeatTimes=2;params.dstRepStride=P;params.srcRepStride=1;
                TransDataTo5HDVldVaRegImpl<float>((__ubuf__ uint64_t*)transposeDst.GetPhyAddr(),
                    (__ubuf__ uint64_t*)transposeSrc.GetPhyAddr(),params);PipeBarrier<PIPE_V>();
            } else {
                Gather(mat,input,transposeIx.ReinterpretCast<uint32_t>(),0,t.n*P);PipeBarrier<PIPE_V>();
            }
            uint64_t liveMask[2]={(1ULL<<t.n)-1,0};
            for(uint32_t j=0;j+3<t.n;++j) {
                auto col=mat[j*P];Abs(candidates,col,P);PipeBarrier<PIPE_V>();
                WholeReduceMax(maximum,candidates,liveMask,1,1,1,8);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint32_t bits,row;GetReduceRepeatMaxMinSpr<float>(bits,row);
                const float inverse=1.0f/col.GetValue(row);liveMask[0]&=~(1ULL<<row);
                outputIx.SetValue(j,(t.n*P+row)*4);scales.SetValue(j,inverse);
                uint64_t pivotMask[2]={1ULL<<(row%8),0};Duplicate(col[(row/8)*8],0.0f,pivotMask,1,1,8);
                const uint32_t width=t.n-j;
                Gather(pivot,mat,offsets.ReinterpretCast<uint32_t>(),((j+1)*P+row)*4,width);PipeBarrier<PIPE_V>();
                Muls(pivot,pivot,-inverse,width);PipeBarrier<PIPE_V>();Brcb(weights,pivot,(width+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                MulAddDst(mat[(j+1)*P],col,weights,(uint64_t)P,(uint8_t)width,BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
            }
            // Solve the final 3x3 Schur complement with partial pivoting.
            {
                const uint32_t j=t.n-3;
                auto col0=mat[j*P],col1=mat[(j+1)*P],col2=mat[(j+2)*P],rhs=mat[t.n*P];
                SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint64_t tailMask=liveMask[0];
                uint32_t row0=63-static_cast<uint32_t>(CountLeadingZero(tailMask));tailMask^=1ULL<<row0;
                uint32_t row1=63-static_cast<uint32_t>(CountLeadingZero(tailMask));tailMask^=1ULL<<row1;
                uint32_t row2=63-static_cast<uint32_t>(CountLeadingZero(tailMask));
                float a0=col0.GetValue(row0),a1=col0.GetValue(row1),a2=col0.GetValue(row2);
                float mag0=a0<0?-a0:a0,mag1=a1<0?-a1:a1,mag2=a2<0?-a2:a2;
                if(mag1>mag0 || (mag1==mag0 && row1<row0)) {
                    const uint32_t r=row0;row0=row1;row1=r;
                    const float v=a0;a0=a1;a1=v;mag0=mag1;
                }
                if(mag2>mag0 || (mag2==mag0 && row2<row0)) {
                    const uint32_t r=row0;row0=row2;row2=r;
                    const float v=a0;a0=a2;a2=v;
                }
                const float b0=col1.GetValue(row0),c0=col2.GetValue(row0),y0=rhs.GetValue(row0);
                float b1=col1.GetValue(row1),c1=col2.GetValue(row1),y1=rhs.GetValue(row1);
                float b2=col1.GetValue(row2),c2=col2.GetValue(row2),y2=rhs.GetValue(row2);
                const float inverse0=1.0f/a0,upperB=b0*inverse0,upperC=c0*inverse0,first=y0*inverse0;
                b1-=a1*upperB;c1-=a1*upperC;y1-=a1*first;
                b2-=a2*upperB;c2-=a2*upperC;y2-=a2*first;
                const float magB1=b1<0?-b1:b1,magB2=b2<0?-b2:b2;
                if(magB2>magB1 || (magB2==magB1 && row2<row1)) {
                    const float b=b1,c=c1,y=y1;b1=b2;c1=c2;y1=y2;b2=b;c2=c;y2=y;
                }
                const float inverse1=1.0f/b1,upper=c1*inverse1,middle=y1*inverse1;
                const float diagonal=c2-b2*upper,second=y2-b2*middle;
                const float x2=second/diagonal,x1=middle-upper*x2,x0=first-upperB*x1-upperC*x2;
                Axpy(rhs,col0,-x0,P);PipeBarrier<PIPE_V>();
                Axpy(rhs,col1,-x1,P);PipeBarrier<PIPE_V>();
                Axpy(rhs,col2,-x2,P);PipeBarrier<PIPE_V>();
                constexpr uint32_t unitOffset=P*W+P*P+W+P+64+P;
                weights.SetValue(0,1.0f);
                outputIx.SetValue(j,unitOffset*4);outputIx.SetValue(j+1,unitOffset*4);outputIx.SetValue(j+2,unitOffset*4);
                scales.SetValue(j,x0);scales.SetValue(j+1,x1);scales.SetValue(j+2,x2);
                SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
            }
            PipeBarrier<PIPE_ALL>();Gather(output,mat,outputIx,0,t.n);PipeBarrier<PIPE_V>();Mul(output,output,scales,t.n);PipeBarrier<PIPE_ALL>();
            DataCopyPad(gx[(uint64_t)batch*t.n],output,DataCopyExtParams{1,t.n*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
    }
};
class SmallColumns {
    template<uint32_t Count>
    __aicore__ inline void StaticIndex(LocalTensor<int32_t> dst) {
        static_assert(Count==64 || Count==72);
        for(int32_t i=0;i<8;++i)dst.SetValue(i,i);
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        UnaryRepeatParams params;
        for(uint32_t i=1;i<8;++i){
            Adds(dst[i*8],dst[(i-1)*8],(int32_t)8,(uint64_t)8,1,params);PipeBarrier<PIPE_V>();
        }
        if constexpr(Count>64){
            Adds(dst[64],dst,(int32_t)64,(uint64_t)8,1,params);PipeBarrier<PIPE_V>();
        }
    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        constexpr uint32_t P=64,W=72,L=P+8;
        LocalTensor<float> mat(TPosition::VECCALC,0,24000);
        auto input=mat[(P+8)*W],rhs=input[P*P],pivot=rhs[P*8];auto colIx=pivot[W].ReinterpretCast<int32_t>();
        auto candidates=colIx[P].ReinterpretCast<float>(),maximum=candidates[P],scales=maximum[64],weights=scales[P];
        auto offsets=weights[8*W].ReinterpretCast<int32_t>(),address=offsets[W];
        auto transposeIx=address[W],tempIx=transposeIx[P*P];auto output=tempIx[P*P].ReinterpretCast<float>();auto outputIx=output[P*8].ReinterpretCast<uint32_t>();
        GlobalTensor<float> ga,gb,gx;ga.SetGlobalBuffer((__gm__ float*)aa);gb.SetGlobalBuffer((__gm__ float*)bb);gx.SetGlobalBuffer((__gm__ float*)xx);
        // Four 16-bit block addresses per word, as required by CANN 9.0 dav_c220.
        // This region holds packed A only on the mutually exclusive unaligned path.
        auto transposeSrc=transposeIx.ReinterpretCast<uint64_t>();
        auto transposeDst=transposeIx[32].ReinterpretCast<uint64_t>();
        if(t.n%8==0) {
            constexpr uint64_t broadcast4=0x0001000100010001ULL;
            const uint64_t srcBase=((uint64_t)input.GetPhyAddr()>>5)&0x1fff;
            const uint64_t dstBase=((uint64_t)mat.GetPhyAddr()>>5)&0x1fff;
            for(uint32_t group=0;group<4;++group) {
                for(uint32_t q=0;q<4;++q) {
                    transposeSrc.SetValue(group*4+q,(srcBase+group*128+q*32)*broadcast4+
                        (8ULL<<16)+(16ULL<<32)+(24ULL<<48));
                    transposeDst.SetValue(group*4+q,(dstBase+group*2+q*18)*broadcast4+
                        (1ULL<<16)+(9ULL<<32)+(10ULL<<48));
                }
            }
            constexpr uint32_t transposeSv=0;
            SetFlag<HardEvent::S_V>(transposeSv);WaitFlag<HardEvent::S_V>(transposeSv);
        }
        TransDataTo5HDParams transposeParams;
        transposeParams.dstHighHalf=false;transposeParams.srcHighHalf=false;
        transposeParams.repeatTimes=(t.n+7)/8;
        transposeParams.dstRepStride=L;transposeParams.srcRepStride=1;
        StaticIndex<W>(offsets);DataCopy(colIx,offsets,P);PipeBarrier<PIPE_V>();
        // The old transpose-index scratch is free with the native transpose.
        auto rawInput=transposeIx.ReinterpretCast<float>();
        if(t.n%8) {
            auto rows=input.ReinterpretCast<int32_t>();
            auto columns=weights[8*P].ReinterpretCast<int32_t>();
            Muls(rows,colIx,(int32_t)(t.n*4),P);
            Muls(columns,colIx,(int32_t)4,P);PipeBarrier<PIPE_V>();
            Brcb(weights.ReinterpretCast<int32_t>(),columns,P/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Add(tempIx,rows,weights.ReinterpretCast<int32_t>(),(uint64_t)P,(uint8_t)t.n,BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
        }
        Muls(offsets,offsets,(int32_t)(L*4),W);Muls(colIx,colIx,(int32_t)(t.k*4),P);PipeBarrier<PIPE_ALL>();
        constexpr uint32_t vs=0;
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            Duplicate(mat,0.0f,(P+8)*W);if(t.n%8==0)Duplicate(input,0.0f,P*P);Duplicate(rhs,0.0f,P*8);Duplicate(scales,1.0f,P);PipeBarrier<PIPE_ALL>();
            if(t.n%8) {
                DataCopyPad(rawInput,ga[(uint64_t)batch*t.n*t.n],DataCopyExtParams{1,t.n*t.n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            } else {
            DataCopyPad(input,ga[(uint64_t)batch*t.n*t.n],DataCopyExtParams{(uint16_t)t.n,t.n*4,0,(P-(t.n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-t.n%8)%8),0});
            }
            DataCopyPad(rhs,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k],DataCopyExtParams{1,t.n*t.k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
            if(t.n%8) {
                Gather(mat,rawInput,tempIx.ReinterpretCast<uint32_t>(),0,(uint64_t)t.n,(uint8_t)t.n,(uint16_t)(L/8));PipeBarrier<PIPE_V>();
            } else {
                for(uint32_t group=0;group*16<t.n;++group) {
                    TransDataTo5HDVldVaRegImpl<float>(
                        (__ubuf__ uint64_t*)transposeDst[group*4].GetPhyAddr(),
                        (__ubuf__ uint64_t*)transposeSrc[group*4].GetPhyAddr(),transposeParams);
                    PipeBarrier<PIPE_V>();
                }
            }
            for(uint32_t k=0;k<t.k;++k)Gather(mat[(t.n+k)*L],rhs,colIx.ReinterpretCast<uint32_t>(),k*4,P);PipeBarrier<PIPE_V>();
            uint64_t liveMask[2]={t.n==64?~0ULL:((1ULL<<t.n)-1),0};
            const uint32_t tail=t.k<=2?2:0;
            for(uint32_t j=0;j+tail<t.n;++j) {
                auto col=mat[j*L];Abs(candidates,col,P);PipeBarrier<PIPE_V>();
                WholeReduceMax(maximum,candidates,liveMask,1,1,1,8);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint32_t bits,row;GetReduceRepeatMaxMinSpr<float>(bits,row);
                const float inverse=1.0f/col.GetValue(row);liveMask[0]&=~(1ULL<<row);
                uint64_t pivotMask[2]={1ULL<<(row%8),0};
                Duplicate(col[(row/8)*8],0.0f,pivotMask,1,1,8);
                Duplicate(scales[(row/8)*8],inverse,pivotMask,1,1,8);
                for(uint32_t k=0;k<t.k;++k)outputIx.SetValue(j*t.k+k,((t.n+k)*L+row)*4);
                PipeBarrier<PIPE_V>();
                const uint32_t width=t.n+t.k-j-1;
                const uint32_t firstColumn=(j+1)/8*8,gatheredWidth=t.n+t.k-firstColumn;
                Gather(pivot,mat,offsets.ReinterpretCast<uint32_t>(),(firstColumn*L+row)*4,gatheredWidth);PipeBarrier<PIPE_V>();
                Muls(pivot,pivot,-inverse,gatheredWidth);PipeBarrier<PIPE_V>();
                // Aligned eight-column groups stay within the72 reserved columns.
                // Consumers use only the original live columns and rows0..63.
                Brcb(mat[firstColumn*L+P],pivot,(gatheredWidth+7)/8,BrcbRepeatParams{L/8,L});PipeBarrier<PIPE_V>();
                MulAddDst(mat[(j+1)*L],col,mat[(j+1)*L+P],(uint64_t)P,(uint8_t)width,BinaryRepeatParams{1,1,0,L/8,0,L/8});PipeBarrier<PIPE_V>();
            }
            if(tail) {
                const uint32_t j=t.n-2;
                auto col0=mat[j*L],col1=mat[(j+1)*L];
                SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint32_t row0=63-static_cast<uint32_t>(CountLeadingZero(liveMask[0]));
                uint32_t row1=63-static_cast<uint32_t>(CountLeadingZero(liveMask[0]^(1ULL<<row0)));
                float a0=col0.GetValue(row0),a1=col0.GetValue(row1);
                const float mag0=a0<0?-a0:a0,mag1=a1<0?-a1:a1;
                if(mag1>mag0 || (mag1==mag0 && row1<row0)) {
                    const uint32_t r=row0;row0=row1;row1=r;
                    const float v=a0;a0=a1;a1=v;
                }
                const float b0=col1.GetValue(row0),b1=col1.GetValue(row1);
                const float inverse0=1.0f/a0,upper=b0*inverse0;
                const float inverse1=1.0f/(b1-a1*upper);
                // Reuse the now-dead broadcast region for already-solved values.
                // Final Gather reads these directly, outside the scaled RHS area.
                constexpr uint32_t valueOffset=(P+8)*W+P*P+P*8+W+P+P+64+P;
                for(uint32_t k=0;k<t.k;++k) {
                    auto rhsCol=mat[(t.n+k)*L];
                    const float first=rhsCol.GetValue(row0)*inverse0;
                    const float second=rhsCol.GetValue(row1)-a1*first;
                    const float x1=second*inverse1,x0=first-upper*x1;
                    Axpy(rhsCol,col0,-x0,P);PipeBarrier<PIPE_V>();
                    Axpy(rhsCol,col1,-x1,P);PipeBarrier<PIPE_V>();
                    weights.SetValue(k,x0);weights.SetValue(8+k,x1);
                    outputIx.SetValue(j*t.k+k,(valueOffset+k)*4);
                    outputIx.SetValue((j+1)*t.k+k,(valueOffset+8+k)*4);
                }
                constexpr uint32_t sv=0;
                SetFlag<HardEvent::S_V>(sv);WaitFlag<HardEvent::S_V>(sv);
            }
            for(uint32_t k=0;k<t.k;++k)Mul(mat[(t.n+k)*L],mat[(t.n+k)*L],scales,P);PipeBarrier<PIPE_ALL>();
            Gather(output,mat,outputIx,0,t.n*t.k);PipeBarrier<PIPE_ALL>();DataCopyPad(gx[(uint64_t)batch*t.n*t.k],output,DataCopyExtParams{1,t.n*t.k*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
    }
};

template<uint32_t P> class SmallSquare {
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        const uint32_t n=t.n,k=t.k,p=P,q=(k+7)/8*8,L=P+q;
        // Keep the measured scratch spacing and initialization; compacting it
        // regressed the small-matrix path in the paired experiments.
        const uint32_t oldCap=n*L+L+22*p+64+n*q;pipe.InitBuffer(buf,(oldCap+24*n+8+8*((n+7)/8*8))*4);
        auto mat=buf.Get<float>(),pivot=mat[n*L],col=pivot[L],scratchGap=col[p],candidates=scratchGap[p],weights=candidates[p];
        auto tmp=weights[p*8],scalar=tmp[2*p];auto indices=tmp[p].ReinterpretCast<uint32_t>();
        auto rhsInput=mat[oldCap],rhsExpanded=rhsInput[24*n+8];auto rhsIndices=rhsInput[8*n+8].ReinterpretCast<int32_t>(),rhsRows=rhsIndices[8*n];
        auto result=scalar[64];auto outputIndices=result[n*q].ReinterpretCast<uint32_t>();
        const auto vs=pipe.FetchEventID(HardEvent::V_S);
        GlobalTensor<float> ga,gb,gx;
        ga.SetGlobalBuffer((__gm__ float*)a);gb.SetGlobalBuffer((__gm__ float*)b);
        gx.SetGlobalBuffer((__gm__ float*)x);
        for(uint32_t i=0;i<n;++i)indices.SetValue(i,i*L*4);PipeBarrier<PIPE_ALL>();
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            uint32_t labels[64];uint64_t liveMask[2]={n==64?~0ULL:((1ULL<<n)-1),0};
            Duplicate(mat,0.0f,n*L);Duplicate(scratchGap,1.0f,p);Duplicate(col,0.0f,p);PipeBarrier<PIPE_ALL>();
            DataCopyPad(mat,ga[(uint64_t)batch*n*n],DataCopyExtParams{(uint16_t)n,n*4,0,(L-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
            if(k<=8) {
                Duplicate(rhsInput,0.0f,8*n+8);PipeBarrier<PIPE_ALL>();
                DataCopyPad(rhsInput,gb[(uint64_t)(t.broadcastB?0:batch)*n*k],DataCopyExtParams{1,n*k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                if(k==1) {
                    PipeBarrier<PIPE_ALL>();Brcb(rhsExpanded,rhsInput,(n+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                } else {
                CreateVecIndex(rhsIndices,(int32_t)0,8*n);PipeBarrier<PIPE_V>();
                ShiftRight(rhsRows,rhsIndices,(int32_t)3,8*n);PipeBarrier<PIPE_V>();
                Muls(rhsRows,rhsRows,(int32_t)(4*(8-k)),8*n);Muls(rhsIndices,rhsIndices,(int32_t)4,8*n);PipeBarrier<PIPE_V>();
                Sub(rhsIndices,rhsIndices,rhsRows,8*n);PipeBarrier<PIPE_ALL>();
                Gather(rhsExpanded,rhsInput,rhsIndices.ReinterpretCast<uint32_t>(),0,8*n);PipeBarrier<PIPE_V>();
                }
                DataCopy(mat[p],rhsExpanded,DataCopyParams{(uint16_t)n,1,0,(uint16_t)(p/8)});
            } else {
            DataCopyPad(mat[p],gb[(uint64_t)(t.broadcastB?0:batch)*n*k],DataCopyExtParams{(uint16_t)n,k*4,0,p/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(q-k),0});
            }
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<n;++j) {
                Gather(col,mat[j],indices,0,n);PipeBarrier<PIPE_V>();
                Abs(candidates,col,n);PipeBarrier<PIPE_V>();
                WholeReduceMax(tmp,candidates,liveMask,1,1,1,8);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint32_t maximumBits,row;GetReduceRepeatMaxMinSpr<float>(maximumBits,row);
                if(k>8)labels[row]=j;float inv=1.0f/col.GetValue(row);
                liveMask[0]&=~(1ULL<<row);
                const uint32_t first=j/8*8;
                // Include the pivot row in the update, then restore it. No scalar
                // UB store or S_V handoff is needed inside the pivot loop.
                Muls(pivot,mat[row*L+first],-inv,L-first);
                Brcb(weights,col,p/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                BinaryRepeatParams update{1,1,0,(uint8_t)(L/8),0,1};
                for(uint32_t off=first;off<L;off+=64)
                    MulAddDst(mat[off],pivot[off-first],weights,(uint64_t)(L-off<64?L-off:64),(uint8_t)n,update);
                PipeBarrier<PIPE_V>();
                Muls(mat[row*L+first],pivot,-1.0f,L-first);PipeBarrier<PIPE_V>();
                if(k<=8)for(uint32_t rhs=0;rhs<k;++rhs)outputIndices.SetValue(j*k+rhs,(row*L+p+rhs)*4);
            }
            PipeBarrier<PIPE_ALL>();
            if(k<=8) {
                Gather(result,mat,outputIndices,0,n*k);PipeBarrier<PIPE_ALL>();
                DataCopyPad(gx[(uint64_t)batch*n*k],result,DataCopyExtParams{1,n*k*4,0,0,0});
            } else {
                for(uint32_t row=0;row<n;++row)
                    DataCopyPad(gx[(uint64_t)batch*n*k+labels[row]*k],mat[row*L+p],DataCopyExtParams{1,k*4,0,0,0});
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
};

// Deferred row normalization avoids scattered pivot-row stores in column layout.
template<uint32_t N,uint32_t PITCH=N,uint32_t WIDTH=16> struct ColumnPanel {
    static constexpr uint32_t B=WIDTH,SCRATCH=3*N+112+8*B;
    static __aicore__ inline void Run(LocalTensor<float> panel,LocalTensor<float> result,
        LocalTensor<float> active,LocalTensor<float> scratch,uint32_t width,uint32_t *rows,TPipe &pipe) {
        auto pivot=scratch,candidates=scratch[32+N],maximum=scratch[32+2*N],scales=scratch[48+2*N];
        auto offsets=scratch[48+3*N].template ReinterpretCast<int32_t>(),address=scratch[80+3*N].template ReinterpretCast<int32_t>();
        auto coefficients=scratch[112+3*N];
        Duplicate(scales,1.0f,N);CreateVecIndex(offsets,(int32_t)0,32);PipeBarrier<PIPE_V>();Muls(offsets,offsets,(int32_t)(PITCH*4),32);PipeBarrier<PIPE_V>();
        const auto vs=pipe.FetchEventID(HardEvent::V_S),sv=pipe.FetchEventID(HardEvent::S_V);
        for(uint32_t j=0;j<width;++j) {
            // The consumed input column is dead after this pivot; reuse it in place.
            auto col=panel[j*PITCH];Abs(candidates,col,N);PipeBarrier<PIPE_V>();Mul(candidates,candidates,active,N);PipeBarrier<PIPE_V>();
            if constexpr(N%64==0)WholeReduceMax(maximum,candidates,64,N/64,1,1,8);
            else {SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,N);WholeReduceMax<float,false>(maximum,candidates,MASK_PLACEHOLDER,1,1,1,8);SetMaskNorm();ResetMask();}
            SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);uint32_t bits,row;GetReduceRepeatMaxMinSpr<float>(bits,row);
            const float inverse=1.0f/col.GetValue(row);rows[j]=row;col.SetValue(row,0.0f);active.SetValue(row,0.0f);scales.SetValue(row,inverse);panel.SetValue((B+j)*PITCH+row,1.0f);
            SetFlag<HardEvent::S_V>(sv);WaitFlag<HardEvent::S_V>(sv);
            // Only the remaining input columns and the first j+1 transform columns
            // are live. Their contiguous window is always exactly B columns.
            Adds(address,offsets,(int32_t)(((j+1)*PITCH+row)*4),B);PipeBarrier<PIPE_V>();Gather(pivot,panel,address.template ReinterpretCast<uint32_t>(),0,B);PipeBarrier<PIPE_V>();
            Muls(pivot,pivot,-inverse,B);PipeBarrier<PIPE_V>();Brcb(coefficients,pivot,B/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            const BinaryRepeatParams update{1,1,0,PITCH/8,0,1};
            for(uint32_t off=0;off<N;off+=64)MulAddDst(panel[(j+1)*PITCH+off],col[off],coefficients,(uint64_t)(N-off<64?N-off:64),(uint8_t)B,update);
            PipeBarrier<PIPE_V>();
        }
        // Selected rows stay unnormalized during elimination; apply their saved
        // inverse pivot once when publishing the transform.
        for(uint32_t j=0;j<B;++j)Mul(result[j*N],panel[(B+j)*PITCH],scales,N);
        PipeBarrier<PIPE_ALL>();
    }
};
template<uint32_t N,uint32_t C=32,uint32_t Steps=1,bool SharedRhs=false> class RefinedSquare {
    static constexpr uint32_t P=256,T=16,SZ=P*T,CP=(N+7)/8*8;
    static constexpr uint32_t B=16,PL=2*B+8;
    static constexpr uint32_t ELIM_CAP=N*(2*C+8+(SharedRhs?8:0))+N*(PL+B)+B*(2*C+(SharedRhs?8:0));
    static constexpr uint32_t ARRAYS=11;
    static constexpr uint32_t CAP0=ELIM_CAP>ARRAYS*SZ+T*8?ELIM_CAP:ARRAYS*SZ+T*8;
    static constexpr uint32_t RHS_CAP=SharedRhs?N*(2*C+16+64):2*N*64;
    static constexpr uint32_t CAP=CAP0>RHS_CAP?CAP0:RHS_CAP;
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
    LocalTensor<float> a,x,ah,al,xh,xl,h,l,s[6],raw,ahi,alo;
    LocalTensor<uint32_t> ixBase,rhsIx;
    template<HardEvent event> __aicore__ inline void Sync() {
        const auto id=pipe.FetchEventID(event);SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void A(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t z) {Add(o,u,v,z);PipeBarrier<PIPE_V>();}
    __aicore__ inline void D(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t z) {Sub(o,u,v,z);PipeBarrier<PIPE_V>();}
    __aicore__ inline void M(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t z) {Mul(o,u,v,z);PipeBarrier<PIPE_V>();}
    __aicore__ inline void Scale(LocalTensor<float> o,LocalTensor<float> u,float v,uint32_t z) {Muls(o,u,v,z);PipeBarrier<PIPE_V>();}
    template<bool sub=false> __aicore__ inline void Rows(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t rows,uint32_t width) {
        const BinaryRepeatParams step{1,1,1,P/8,P/8,P/8};
        for(uint32_t off=0;off<width;off+=64) {
            uint64_t mask=width-off<64?width-off:64;
            if constexpr(sub)Sub(o[off],u[off],v[off],mask,(uint8_t)rows,step);
            else Add(o[off],u[off],v[off],mask,(uint8_t)rows,step);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DensePairSum(uint32_t count) {
        A(s[0],a,x,count);D(s[1],s[0],a,count);D(s[2],s[0],s[1],count);
        D(s[2],a,s[2],count);D(s[3],x,s[1],count);A(s[2],s[2],s[3],count);
        A(s[2],s[2],a[64],count);A(l,s[2],x[64],count);
        Scale(h,s[0],1.0f,count);
    }
    __aicore__ inline void AlignedRowSum(LocalTensor<float> out,LocalTensor<float> input,uint32_t rows) {
        for(uint32_t half=P/2;half>=8;half/=2)Rows(input,input,input[half],rows,half);
        WholeReduceSum(a,input,8,rows,1,1,P/8);PipeBarrier<PIPE_V>();
        Scale(out,a,1.0f,rows);
    }
    __aicore__ inline void AccurateDot(uint32_t rows) {
        const uint32_t count=rows*P;
        M(h,raw,x,count);
        Scale(xh,x,4097.0f,count);D(xl,xh,x,count);D(xh,xh,xl,count);D(xl,x,xh,count);
        M(l,ahi,xh,count);D(l,l,h,count);
        M(a,ahi,xl,count);A(l,l,a,count);
        M(a,alo,xh,count);A(l,l,a,count);
        M(a,alo,xl,count);A(l,l,a,count);
        Abs(a,h,count);PipeBarrier<PIPE_V>();
        WholeReduceMax(s[0],a,64,count/64,1,1,8);Sync<HardEvent::V_S>();
        uint32_t maximumBits,maximumIndex;GetReduceRepeatMaxMinSpr<float>(maximumBits,maximumIndex);
        const uint32_t exponent=(maximumBits>>23)&255;
        if(exponent>10 && exponent<240) {
            constexpr uint32_t logP=P==512?9:(P==256?8:7);
            union ScaleBits {uint32_t bits;float value;};
            ScaleBits q0,q1;q0.bits=((exponent+logP+1)<<23)|0x400000;
            q1.bits=((exponent+2*logP-22)<<23)|0x400000;
            Adds(s[0],h,q0.value,count);PipeBarrier<PIPE_V>();Adds(s[0],s[0],-q0.value,count);PipeBarrier<PIPE_V>();
            D(s[1],h,s[0],count);
            Adds(s[2],s[1],q1.value,count);PipeBarrier<PIPE_V>();Adds(s[2],s[2],-q1.value,count);PipeBarrier<PIPE_V>();
            D(s[3],s[1],s[2],count);
            AlignedRowSum(s[0],s[0],rows);AlignedRowSum(s[1],s[2],rows);
            AlignedRowSum(s[2],s[3],rows);AlignedRowSum(s[3],l,rows);
            A(a,s[2],s[3],rows);
            A(x,s[1],a,rows);D(h,x,s[1],rows);D(l,x,h,rows);D(l,s[1],l,rows);D(h,a,h,rows);A(l,l,h,rows);
            A(h,s[0],x,rows);D(s[1],h,s[0],rows);D(s[2],h,s[1],rows);D(s[2],s[0],s[2],rows);D(s[1],x,s[1],rows);A(a,s[2],s[1],rows);A(l,a,l,rows);
            return;
        }
        for(uint32_t half=P/2;half>=8;half/=2) {
            Rows(s[0],h,h[half],rows,half);Rows<true>(s[1],s[0],h,rows,half);
            Rows<true>(s[2],s[0],s[1],rows,half);Rows<true>(s[2],h,s[2],rows,half);
            Rows<true>(s[3],h[half],s[1],rows,half);Rows(s[2],s[2],s[3],rows,half);
            Rows(s[2],s[2],l,rows,half);Rows(l,s[2],l[half],rows,half);
            for(uint32_t off=0;off<half;off+=64)
                Muls(h[off],s[0][off],1.0f,(uint64_t)(half-off<64?half-off:64),(uint8_t)rows,UnaryRepeatParams{1,1,P/8,P/8});
            PipeBarrier<PIPE_V>();
        }
        for(uint32_t half=4;half;half/=2) {
            auto ix0=ixBase[4*T*(4-half)],ix1=ix0[T*half];
            Gather(a,h,ix0,0,rows*half);Gather(x,h,ix1,0,rows*half);
            Gather(a[64],l,ix0,0,rows*half);Gather(x[64],l,ix1,0,rows*half);
            PipeBarrier<PIPE_V>();
            DensePairSum(rows*half);
        }
    }
    __aicore__ inline void FastDot(uint32_t rows) {
        M(h,a,x,rows*P);
        for(uint32_t half=P/2;half>=8;half/=2)Rows(h,h,h[half],rows,half);
        WholeReduceSum(s[0],h,8,rows,1,1,P/8);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadTile(GlobalTensor<float> matrix,uint64_t offset,uint32_t rows,LocalTensor<float> tile,uint32_t width=N) {
        Duplicate(tile,0.0f,T*P);PipeBarrier<PIPE_ALL>();
        DataCopyPad(tile,matrix[offset],DataCopyExtParams{(uint16_t)rows,width*4,0,(P-((width+7)/8*8))/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-width%8)%8),0});
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadColumn(GlobalTensor<float> column,uint64_t offset,uint32_t rows) {
        DataCopy(x,column[offset],P);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=1;r<rows;++r)DataCopy(x[r*P],x,P);
        PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        const uint32_t lane=GetBlockIdx()%t.teams,group=GetBlockIdx()/t.teams,groups=t.blocks/t.teams;
        const bool rhs=lane==t.teams-1;
        const uint32_t W=SharedRhs?2*C+(rhs?8:0):(rhs?64:2*C),L=SharedRhs?W+8:(rhs?64:2*C+8),CTL=N*B+B;
        pipe.InitBuffer(buf,(CAP+128+14*CP+64)*4);
        auto mat=buf.Get<float>(),ordered=mat[SharedRhs?N*(2*C+16):N*64],pivot=mat[CAP],col=pivot[128],scalar=col[CP],active=col[CP+8],candidates=active[CP],weights=candidates[CP];
        auto tmp=weights[CP*8];auto indices=tmp[CP].template ReinterpretCast<uint32_t>();
        ixBase=indices;rhsIx=col.template ReinterpretCast<uint32_t>();
        a=mat;x=mat[SZ];
        h=mat[2*SZ];l=mat[3*SZ];ah=mat[4*SZ];al=mat[5*SZ];xh=mat[6*SZ];xl=mat[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=mat[(4+i)*SZ];
        raw=mat[8*SZ];ahi=mat[9*SZ];alo=mat[10*SZ];
        auto outputTile=mat[ARRAYS*SZ];
        GlobalTensor<float> ga,gb,gx,ctl,inverse,initial,residual;
        ga.SetGlobalBuffer((__gm__ float*)aa);gb.SetGlobalBuffer((__gm__ float*)bb);
        gx.SetGlobalBuffer((__gm__ float*)xx);ctl.SetGlobalBuffer((__gm__ float*)workspace);
        const uint64_t invStart=(uint64_t)groups*2*CTL,initialStart=invStart+(uint64_t)groups*N*N,residualStart=initialStart+(uint64_t)groups*t.k*P;
        inverse.SetGlobalBuffer((__gm__ float*)workspace+invStart);
        initial.SetGlobalBuffer((__gm__ float*)workspace+initialStart);
        residual.SetGlobalBuffer((__gm__ float*)workspace+residualStart);
        for(uint32_t round=0;round<(t.batches+groups-1)/groups;++round) {
            const uint32_t batch=round*groups+group;const bool live=batch<t.batches;
            uint32_t labels[N];
            for(uint32_t i=0;i<N;++i)labels[i]=i;
            for(uint32_t i=0;i<N;++i)indices.SetValue(i,i*PL*4);
            Duplicate(mat,0.0f,CAP);Duplicate(active,0.0f,CP);PipeBarrier<PIPE_V>();Duplicate(active,1.0f,t.n);Duplicate(col,0.0f,CP);PipeBarrier<PIPE_ALL>();
            if(live) {
                if(!rhs || SharedRhs) {
                    const uint32_t width=N-lane*C<C?N-lane*C:C;
                    const uint32_t realWidth=lane*C<t.n?(t.n-lane*C<C?t.n-lane*C:C):0;
                    if(realWidth)DataCopyPad(mat,ga[(uint64_t)batch*t.n*t.n+lane*C],DataCopyExtParams{(uint16_t)t.n,realWidth*4,(t.n-realWidth)*4,(L-((realWidth+7)/8*8))/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-realWidth%8)%8),0});
                    PipeBarrier<PIPE_ALL>();
                    for(uint32_t i=0;i<width;++i) {
                        mat.SetValue((lane*C+i)*L+C+i,1.0f);
                        if(lane*C+i>=t.n)mat.SetValue((lane*C+i)*L+i,1.0f);
                    }
                }
                if(rhs) {
                    const uint32_t qp=(t.k+7)/8*8,rhsOffset=SharedRhs?2*C:0;
                    DataCopyPad(mat[rhsOffset],gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k],DataCopyExtParams{(uint16_t)t.n,t.k*4,0,(L-qp)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(qp-t.k),0});
                }
                PipeBarrier<PIPE_ALL>();
            }
            {
                auto panel=mat[N*L],transform=panel[N*PL],selected=transform[N*B];
                for(uint32_t base=0;base<t.n;base+=B) {
                    const uint32_t owner=base/C,panelWidth=t.n-base<B?t.n-base:B;
                    const uint64_t at=(uint64_t)group*2*CTL+((base/B)&1)*CTL;
                    uint32_t panelRows[B];
                    if(live && lane==owner) {
                        if constexpr(true) {
                            auto panelIndex=selected[B*W].template ReinterpretCast<int32_t>();
                            auto temporary=selected[B*W+N*B];
                            static_assert(ELIM_CAP+N*B+ColumnPanel<N>::SCRATCH<=CAP);
                            auto rowBytes=temporary.template ReinterpretCast<int32_t>();
                            CreateVecIndex(rowBytes,(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(rowBytes,rowBytes,(int32_t)(L*4),N);PipeBarrier<PIPE_V>();
                            for(uint32_t j=0;j<B;++j)Adds(panelIndex[j*N],rowBytes,(int32_t)(j*4),N);
                            PipeBarrier<PIPE_V>();Gather(panel,mat[base%C],panelIndex.template ReinterpretCast<uint32_t>(),0,N*B);Duplicate(panel[N*B],0.0f,N*B);PipeBarrier<PIPE_V>();
                            ColumnPanel<N>::Run(panel,transform,active,temporary,panelWidth,panelRows,pipe);
                        } else {
                        Duplicate(panel,0.0f,N*PL);PipeBarrier<PIPE_V>();
                        DataCopy(panel,mat[base%C],DataCopyParams{(uint16_t)N,(uint16_t)(B/8),(uint16_t)((L-B)/8),(uint16_t)((PL-B)/8)});
                        PipeBarrier<PIPE_ALL>();
                        for(uint32_t j=0;j<panelWidth;++j) {
                            Gather(col,panel[j],indices,0,N);PipeBarrier<PIPE_V>();
                            Abs(candidates,col,N);PipeBarrier<PIPE_V>();Mul(candidates,candidates,active,N);PipeBarrier<PIPE_V>();
                            if constexpr(N%64==0)WholeReduceMax(tmp,candidates,64,N/64,1,1,8);
                            else {
                                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,N);
                                WholeReduceMax<float,false>(tmp,candidates,MASK_PLACEHOLDER,1,1,1,8);
                                SetMaskNorm();ResetMask();
                            }
                            Sync<HardEvent::V_S>();
                            uint32_t maximumBits,row;GetReduceRepeatMaxMinSpr<float>(maximumBits,row);
                            const float diagonal=col.GetValue(row);panelRows[j]=row;
                            panel.SetValue(row*PL+B+j,1.0f);Sync<HardEvent::S_V>();
                            Duplicate(tmp,diagonal,2*B);PipeBarrier<PIPE_V>();
                            Div(pivot,panel[row*PL],tmp,2*B);PipeBarrier<PIPE_V>();
                            DataCopy(panel[row*PL],pivot,2*B);
                            Muls(col,col,-1.0f,N);Sync<HardEvent::V_S>();col.SetValue(row,0);active.SetValue(row,0);Sync<HardEvent::S_V>();
                            Brcb(weights,col,CP/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                            const BinaryRepeatParams update{1,1,0,PL/8,0,1};
                            MulAddDst(panel,pivot,weights,(uint64_t)(2*B),(uint8_t)128,update);
                            if constexpr(N>128)MulAddDst(panel[128*PL],pivot,weights[128*8],(uint64_t)(2*B),(uint8_t)(N-128),update);
                            PipeBarrier<PIPE_V>();
                        }
                        for(uint32_t j=0;j<panelWidth;++j)Gather(transform[j*N],panel[B+j],indices,0,N);
                        }
                        PipeBarrier<PIPE_ALL>();DataCopy(ctl[at],transform,N*B);
                        for(uint32_t j=0;j<B;++j)pivot.template ReinterpretCast<uint32_t>().SetValue(j,j<panelWidth?panelRows[j]:0);
                        PipeBarrier<PIPE_ALL>();DataCopy(ctl[at+N*B],pivot,B);PipeBarrier<PIPE_ALL>();
                    }
                    SyncAll<true>();
                    if(live) {
                        if(lane!=owner) {
                            DataCopy(transform,ctl[at],N*B);
                            DataCopy(pivot,ctl[at+N*B],B);PipeBarrier<PIPE_ALL>();
                            for(uint32_t j=0;j<panelWidth;++j)panelRows[j]=pivot.template ReinterpretCast<uint32_t>().GetValue(j);
                        }
                        for(uint32_t j=0;j<panelWidth;++j) {
                            const uint32_t row=panelRows[j];labels[row]=base+j;active.SetValue(row,0);
                            DataCopy(selected[j*W],mat[row*L],W);
                        }
                        PipeBarrier<PIPE_ALL>();
                        for(uint32_t j=0;j<panelWidth;++j)Duplicate(mat[panelRows[j]*L],0.0f,W);
                        PipeBarrier<PIPE_V>();
                        // Columns already pivoted are never consumed again; retain all inverse/RHS columns.
                        const uint32_t done=base+panelWidth;
                        const uint32_t first=(!rhs || SharedRhs)?(done<=lane*C?0:(done-lane*C<C?done-lane*C:C)):0;
                        const BinaryRepeatParams update{1,1,0,(uint8_t)(L/8),0,1};
                        for(uint32_t j=0;j<panelWidth;++j) {
                            Brcb(weights,transform[j*N],CP/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                            for(uint32_t off=first;off<W;off+=64) {
                                MulAddDst(mat[off],selected[j*W+off],weights,(uint64_t)(W-off<64?W-off:64),(uint8_t)128,update);
                                if constexpr(N>128)MulAddDst(mat[128*L+off],selected[j*W+off],weights[128*8],(uint64_t)(W-off<64?W-off:64),(uint8_t)(N-128),update);
                            }
                            PipeBarrier<PIPE_V>();
                        }
                        PipeBarrier<PIPE_ALL>();
                    }
                }
                for(uint32_t i=0;i<N;++i)indices.SetValue(i,i*(rhs?64:L)*4);PipeBarrier<PIPE_ALL>();
            }
            if(live) {
                if(!rhs || SharedRhs) {
                    for(uint32_t r=0;r<N;++r)DataCopy(mat[labels[r]*L],mat[r*L+C],C);
                    PipeBarrier<PIPE_ALL>();
                    const uint32_t width=N-lane*C<C?N-lane*C:C;
                    DataCopyPad(inverse[(uint64_t)group*N*N+lane*C],mat,DataCopyExtParams{(uint16_t)N,width*4,(L-width)/8,(N-width)*4,0});
                    PipeBarrier<PIPE_ALL>();
                }
                if(rhs) {
                    for(uint32_t r=0;r<N;++r)DataCopy(ordered[labels[r]*64],mat[r*L+(SharedRhs?2*C:0)],8);
                    PipeBarrier<PIPE_ALL>();
                    for(uint32_t k=0;k<t.k;++k) {
                        Duplicate(mat,0.0f,P);PipeBarrier<PIPE_V>();Gather(mat,ordered[k],indices,0,N);PipeBarrier<PIPE_ALL>();
                        DataCopy(initial[((uint64_t)group*t.k+k)*P],mat,P);PipeBarrier<PIPE_ALL>();
                    }
                    Duplicate(mat,0.0f,P);PipeBarrier<PIPE_ALL>();
                    for(uint32_t k=0;k<t.k;++k)DataCopy(residual[((uint64_t)group*t.k+k)*P],mat,P);
                }
                PipeBarrier<PIPE_ALL>();
            }
            SyncAll<true>();
            const uint32_t chunk=(N+t.teams-1)/t.teams,begin=lane*chunk,end=begin+chunk<t.n?begin+chunk:t.n;
            for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
                auto ix0=ixBase[4*T*(4-half)],ix1=ix0[T*half];
                for(uint32_t row=0;row<T;++row)for(uint32_t j=0;j<half;++j) {
                    ix0.SetValue(row*half+j,(row*stride+j)*4);
                    ix1.SetValue(row*half+j,(row*stride+half+j)*4);
                }
            }
            PipeBarrier<PIPE_ALL>();
            for(uint32_t iteration=0;iteration<Steps;++iteration) {
                if(live)for(uint32_t row=begin;row<end;row+=T) {
                    const uint32_t rows=end-row<T?end-row:T;
                    LoadTile(ga,(uint64_t)batch*t.n*t.n+row*t.n,rows,raw,t.n);
                    Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
                    for(uint32_t k=0;k<t.k;++k) {
                        LoadColumn(initial,((uint64_t)group*t.k+k)*P,rows);
                        AccurateDot(rows);PipeBarrier<PIPE_ALL>();
                        DataCopyPad(a,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+row*t.k],DataCopyExtParams{(uint16_t)rows,t.k*4,0,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-t.k),0});
                        PipeBarrier<PIPE_ALL>();
                        for(uint32_t i=0;i<rows;++i)rhsIx.SetValue(i,(i*8+k)*4);
                        PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,0,rows);PipeBarrier<PIPE_V>();
                        D(h,x,h,rows);D(h,h,l,rows);PipeBarrier<PIPE_ALL>();
                        DataCopyPad(residual[((uint64_t)group*t.k+k)*P+row],h,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
                    }
                }
                SyncAll<true>();
                if(live)for(uint32_t row=begin;row<end;row+=T) {
                    const uint32_t rows=end-row<T?end-row:T;
                    LoadTile(inverse,(uint64_t)group*N*N+row*N,rows,a);
                    for(uint32_t k=0;k<t.k;++k) {
                        LoadColumn(residual,((uint64_t)group*t.k+k)*P,rows);FastDot(rows);PipeBarrier<PIPE_ALL>();
                        DataCopyPad(x,initial[((uint64_t)group*t.k+k)*P+row],DataCopyExtParams{1,rows*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
                        A(x,x,s[0],rows);PipeBarrier<PIPE_ALL>();
                        for(uint32_t i=0;i<rows;++i)outputTile.SetValue(i*8+k,x.GetValue(i));
                        if constexpr(Steps>1)if(iteration+1<Steps) {
                            PipeBarrier<PIPE_ALL>();DataCopyPad(initial[((uint64_t)group*t.k+k)*P+row],x,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
                        }
                    }
                    PipeBarrier<PIPE_ALL>();
                    DataCopyPad(gx[(uint64_t)batch*t.n*t.k+row*t.k],outputTile,DataCopyExtParams{(uint16_t)rows,t.k*4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
                SyncAll<true>();
            }
            SyncAll<true>();
        }
    }
};


#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
template<uint32_t N,bool MatrixRhs=false,bool SplitResidual=false> class CubePanel {
protected:
    static constexpr uint32_t P=N,T=N<=256?16:8,SZ=P*T,CP=(N+7)/8*8,IT=(T+7)/8*8;
    static constexpr uint32_t B=N<=256?32:16,PL=2*B+8,PP=N+8,ARRAYS=11;
    static constexpr uint32_t INPUT_INDEX=2*N*B+2*PP*B;
    static constexpr uint32_t PANEL_SCRATCH=INPUT_INDEX+PP*B;
    static constexpr uint32_t PANEL_CAP=PANEL_SCRATCH+ColumnPanel<N,PP,B>::SCRATCH+2*N;
    static constexpr uint32_t DOT_CAP=ARRAYS*SZ+T*8;
    static constexpr uint32_t CAP=PANEL_CAP>DOT_CAP?PANEL_CAP:DOT_CAP;
    static_assert(N*PL+((B+3)/4)*(2*P)<=CAP-2*P);
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
    LocalTensor<float> a,x,ah,al,xh,xl,h,l,s[6],raw,ahi,alo;
    LocalTensor<uint32_t> ixBase,rhsIx;
    template<HardEvent event> __aicore__ inline void Sync() {
        const auto id=pipe.FetchEventID(event);SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void A(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t z) {Add(o,u,v,z);PipeBarrier<PIPE_V>();}
    __aicore__ inline void D(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t z) {Sub(o,u,v,z);PipeBarrier<PIPE_V>();}
    __aicore__ inline void M(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t z) {Mul(o,u,v,z);PipeBarrier<PIPE_V>();}
    __aicore__ inline void Scale(LocalTensor<float> o,LocalTensor<float> u,float v,uint32_t z) {Muls(o,u,v,z);PipeBarrier<PIPE_V>();}
    template<bool sub=false> __aicore__ inline void Rows(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t rows,uint32_t width) {
        const BinaryRepeatParams step{1,1,1,P/8,P/8,P/8};
        for(uint32_t off=0;off<width;off+=64) {
            uint64_t mask=width-off<64?width-off:64;
            if constexpr(sub)Sub(o[off],u[off],v[off],mask,(uint8_t)rows,step);
            else Add(o[off],u[off],v[off],mask,(uint8_t)rows,step);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DensePairSum(uint32_t count) {
        A(s[0],a,x,count);D(s[1],s[0],a,count);D(s[2],s[0],s[1],count);
        D(s[2],a,s[2],count);D(s[3],x,s[1],count);A(s[2],s[2],s[3],count);
        A(s[2],s[2],a[64],count);A(l,s[2],x[64],count);
        Scale(h,s[0],1.0f,count);
    }
    __aicore__ inline void AlignedRowSum(LocalTensor<float> out,LocalTensor<float> input,uint32_t rows) {
        for(uint32_t half=P==384?256:P/2;half>=8;half/=2)Rows(input,input,input[half],rows,half<P-half?half:P-half);
        WholeReduceSum(a,input,8,rows,1,1,P/8);PipeBarrier<PIPE_V>();
        Scale(out,a,1.0f,rows);
    }
    __aicore__ inline void AccurateDot(uint32_t rows) {
        const uint32_t count=rows*P;
        M(h,raw,x,count);
        Scale(xh,x,4097.0f,count);D(xl,xh,x,count);D(xh,xh,xl,count);D(xl,x,xh,count);
        M(l,ahi,xh,count);D(l,l,h,count);
        M(a,ahi,xl,count);A(l,l,a,count);
        M(a,alo,xh,count);A(l,l,a,count);
        M(a,alo,xl,count);A(l,l,a,count);
        Abs(a,h,count);PipeBarrier<PIPE_V>();
        WholeReduceMax(s[0],a,64,count/64,1,1,8);Sync<HardEvent::V_S>();
        uint32_t maximumBits,maximumIndex;GetReduceRepeatMaxMinSpr<float>(maximumBits,maximumIndex);
        const uint32_t exponent=(maximumBits>>23)&255;
        if(exponent>10 && exponent<240) {
            constexpr uint32_t logP=P>256?9:(P==256?8:7);
            union ScaleBits {uint32_t bits;float value;};
            ScaleBits q0,q1;q0.bits=((exponent+logP+1)<<23)|0x400000;
            q1.bits=((exponent+2*logP-22)<<23)|0x400000;
            Adds(s[0],h,q0.value,count);PipeBarrier<PIPE_V>();Adds(s[0],s[0],-q0.value,count);PipeBarrier<PIPE_V>();
            D(s[1],h,s[0],count);
            Adds(s[2],s[1],q1.value,count);PipeBarrier<PIPE_V>();Adds(s[2],s[2],-q1.value,count);PipeBarrier<PIPE_V>();
            D(s[3],s[1],s[2],count);
            AlignedRowSum(s[0],s[0],rows);AlignedRowSum(s[1],s[2],rows);
            AlignedRowSum(s[2],s[3],rows);AlignedRowSum(s[3],l,rows);
            A(a,s[2],s[3],rows);
            A(x,s[1],a,rows);D(h,x,s[1],rows);D(l,x,h,rows);D(l,s[1],l,rows);D(h,a,h,rows);A(l,l,h,rows);
            A(h,s[0],x,rows);D(s[1],h,s[0],rows);D(s[2],h,s[1],rows);D(s[2],s[0],s[2],rows);D(s[1],x,s[1],rows);A(a,s[2],s[1],rows);A(l,a,l,rows);
            return;
        }
        for(uint32_t half=P==384?256:P/2;half>=8;half/=2) {
            const uint32_t width=half<P-half?half:P-half;
            Rows(s[0],h,h[half],rows,width);Rows<true>(s[1],s[0],h,rows,width);
            Rows<true>(s[2],s[0],s[1],rows,width);Rows<true>(s[2],h,s[2],rows,width);
            Rows<true>(s[3],h[half],s[1],rows,width);Rows(s[2],s[2],s[3],rows,width);
            Rows(s[2],s[2],l,rows,width);Rows(l,s[2],l[half],rows,width);
            for(uint32_t off=0;off<width;off+=64)
                Muls(h[off],s[0][off],1.0f,(uint64_t)(width-off<64?width-off:64),(uint8_t)rows,UnaryRepeatParams{1,1,P/8,P/8});
            PipeBarrier<PIPE_V>();
        }
        for(uint32_t half=4;half;half/=2) {
            auto ix0=ixBase[4*T*(4-half)],ix1=ix0[T*half];
            Gather(a,h,ix0,0,rows*half);Gather(x,h,ix1,0,rows*half);
            Gather(a[64],l,ix0,0,rows*half);Gather(x[64],l,ix1,0,rows*half);
            PipeBarrier<PIPE_V>();
            DensePairSum(rows*half);
        }
    }
    __aicore__ inline void FastDot(uint32_t rows) {
        M(h,a,x,rows*P);
        for(uint32_t half=P==384?256:P/2;half>=8;half/=2)Rows(h,h,h[half],rows,half<P-half?half:P-half);
        WholeReduceSum(s[0],h,8,rows,1,1,P/8);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadTile(GlobalTensor<float> matrix,uint64_t offset,uint32_t rows,LocalTensor<float> tile) {
        Duplicate(tile,0.0f,T*P);PipeBarrier<PIPE_ALL>();
        DataCopyPad(tile,matrix[offset],DataCopyExtParams{(uint16_t)rows,N*4,0,(P-N)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadInput(GlobalTensor<float> matrix,uint64_t offset,uint32_t rows,uint32_t realRows,uint32_t width,LocalTensor<float> tile) {
        Duplicate(tile,0.0f,T*P);PipeBarrier<PIPE_ALL>();
        if(realRows)DataCopyPad(tile,matrix[offset],DataCopyExtParams{(uint16_t)realRows,width*4,0,(P-(width+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-width%8)%8),0});
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadColumn(GlobalTensor<float> column,uint64_t offset,uint32_t rows) {
        DataCopy(x,column[offset],P);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=1;r<rows;++r)DataCopy(x[r*P],x,P);
        PipeBarrier<PIPE_V>();
    }
    // Four base-256 signed digits. Each half-input integer dot is exact:
    // 512 terms of magnitude at most 128*128 sum to at most 2^23.
    __aicore__ inline void SplitRow(GlobalTensor<half> digits,GlobalTensor<float> exponents,uint32_t row,uint32_t rowCount) {
        auto w=buf.Get<float>(),v=w,q=w[N],ab=w[2*N],mx=w[3*N];
        auto hd=w[4*N].template ReinterpretCast<half>();
        Abs(ab,v,N);PipeBarrier<PIPE_V>();WholeReduceMax(mx,ab,64,N/64,1,1,8);Sync<HardEvent::V_S>();
        uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);
        const int32_t e=(int32_t)((mb>>23)&255)-127;
        int32_t exponent=e-6;
        if((mb&0x7fffffff)==0) {exponent=0;Duplicate(v,0.0f,N);}
        else if(e<-100 || e>100) {exponent=1000;Duplicate(v,0.0f,N);}
        else {
            union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;
            Scale(v,v,scale.value,N);
        }
        for(uint32_t d=0;d<4;++d) {
            Adds(q,v,12582912.0f,N);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,N);PipeBarrier<PIPE_V>();
            Cast(hd[d*N],q,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();
            D(v,v,q,N);Scale(v,v,256.0f,N);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(digits[(uint64_t)row*N],hd,DataCopyExtParams{4,N*2,0,(rowCount-1)*N*2,0});
        mx.SetValue(0,(float)exponent);PipeBarrier<PIPE_ALL>();
        DataCopyPad(exponents[row],mx,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
    }
    template<bool Minimum> __aicore__ inline int32_t ExponentExtreme(LocalTensor<float> src,LocalTensor<float> scratch,uint32_t count) {
        int32_t result=Minimum?10000:-10000;
        uint32_t index;union {uint32_t bits;float value;} value;
        if(count>=64) {
            if constexpr(Minimum)WholeReduceMin(scratch,src,64,(uint8_t)(count/64),1,1,8);
            else WholeReduceMax(scratch,src,64,(uint8_t)(count/64),1,1,8);
            Sync<HardEvent::V_S>();GetReduceRepeatMaxMinSpr<float>(value.bits,index);result=(int32_t)value.value;
        }
        if(count%64) {
            if constexpr(Minimum)WholeReduceMin(scratch,src[count/64*64],(uint64_t)(count%64),1,1,1,8);
            else WholeReduceMax(scratch,src[count/64*64],(uint64_t)(count%64),1,1,1,8);
            Sync<HardEvent::V_S>();GetReduceRepeatMaxMinSpr<float>(value.bits,index);
            const int32_t tail=(int32_t)value.value;
            result=Minimum?(tail<result?tail:result):(tail>result?tail:result);
        }
        return result;
    }
    __aicore__ inline bool SplitAssemble(GlobalTensor<float> pairs,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> gb,GlobalTensor<float> residual,const LinalgSolveTilingData &t,uint32_t batch,uint32_t lane,uint32_t teams) {
        auto w=buf.Get<float>(),ae=w,xe=w[N],high=w[2*N],low=w[3*N],gs=w[4*N],tmp=w[5*N],term=w[6*N],u=w[7*N],v=w[8*N],factor=w[9*N],br=w[10*N];
        auto ix=w[18*N].template ReinterpretCast<int32_t>();
        auto integerSum=w[11*N].template ReinterpretCast<int32_t>(),rounded=w[12*N].template ReinterpretCast<int32_t>(),remainder=w[13*N].template ReinterpretCast<int32_t>();
        auto groupLow=w[14*N];
        DataCopy(ae,expA,N);DataCopy(xe,expX,N);PipeBarrier<PIPE_ALL>();
        // Every lane reads all exponents, so the fallback decision is uniform.
        int32_t amin=10000,amax=-10000,xmin=10000,xmax=-10000;
        for(uint32_t r=0;r<N;++r) {
            const int32_t av=(int32_t)ae.GetValue(r),xv=(int32_t)xe.GetValue(r);
            amin=av<amin?av:amin;amax=av>amax?av:amax;
            if(r<t.k) {xmin=xv<xmin?xv:xmin;xmax=xv>xmax?xv:xmax;}
        }
        if(amax>=1000 || xmax>=1000 || amin+xmin<-50 || amax+xmax>100)return false;
        CreateVecIndex(ix,(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)32,N);PipeBarrier<PIPE_V>();
        for(uint32_t k=lane;k<t.k;k+=teams) {
            Duplicate(br,0.0f,N*8);Duplicate(low,0.0f,N);PipeBarrier<PIPE_ALL>();
            DataCopyPad(br,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+k],DataCopyExtParams{(uint16_t)t.n,4,(t.k-1)*4,0,0},DataCopyPadExtParams<float>{true,0,7,0});
            PipeBarrier<PIPE_ALL>();Gather(high,br,ix.template ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_V>();
            const int32_t xq=(int32_t)xe.GetValue(k);
            for(uint32_t d=0;d<4;++d) {
                if(d<2)Duplicate(gs,0.0f,N);else Duplicate(integerSum,(int32_t)0,N);PipeBarrier<PIPE_V>();
                const uint32_t count=d+1;
                // Same-weight pairs are three matrix planes apart. Fetch them
                // together, after the previous vector reads of tmp complete.
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(tmp,pairs[((uint64_t)d*t.k+k)*N],DataCopyExtParams{(uint16_t)count,N*4,(3*t.k-1)*N*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
                if(d<2)for(uint32_t i=0;i<count;++i)A(gs,gs,tmp[i*N],N);
                else {
                    // A weight group may exceed 2^24. Sum in int32, then
                    // retain the float conversion's remainder as a low part.
                    for(uint32_t i=0;i<count;++i) {
                        Cast(rounded,tmp[i*N],RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();Add(integerSum,integerSum,rounded,N);PipeBarrier<PIPE_V>();
                    }
                    Cast(gs,integerSum,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();Cast(rounded,gs,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();
                    Sub(remainder,integerSum,rounded,N);PipeBarrier<PIPE_V>();Cast(groupLow,remainder,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();
                }
                Adds(factor,ae,(float)(xq-8*(int32_t)d+127),N);PipeBarrier<PIPE_V>();
                Cast(factor.template ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();
                ShiftLeft(factor.template ReinterpretCast<int32_t>(),factor.template ReinterpretCast<int32_t>(),(int32_t)23,N);PipeBarrier<PIPE_V>();
                M(term,gs,factor,N);Scale(term,term,-1.0f,N);
                A(tmp,high,term,N);D(u,tmp,high,N);D(v,tmp,u,N);D(v,high,v,N);D(u,term,u,N);A(u,u,v,N);A(low,low,u,N);Scale(high,tmp,1.0f,N);
                if(d>=2) {M(term,groupLow,factor,N);Scale(term,term,-1.0f,N);A(low,low,term,N);}
            }
            A(high,high,low,N);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N],high,N);PipeBarrier<PIPE_ALL>();
        }
        return true;
    }
    template<uint32_t Weights=4,bool VectorBounds=false> __aicore__ inline bool SplitAssembleRows(GlobalTensor<float> pairs,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> gb,GlobalTensor<float> residual,const LinalgSolveTilingData &t,uint32_t batch,uint32_t lane,uint32_t teams) {
        static_assert(N==256);constexpr uint32_t C=128,R=N/C;
        auto w=buf.Get<float>(),ae=w,xe=w[N],high=w[2*N],low=w[3*N],gs=w[4*N],tmp=w[5*N],term=w[6*N],u=w[7*N],v=w[8*N],factor=w[9*N],br=w[10*N];
        auto ix=w[18*N].template ReinterpretCast<int32_t>();
        auto integerSum=w[11*N].template ReinterpretCast<int32_t>(),rounded=w[12*N].template ReinterpretCast<int32_t>(),remainder=w[13*N].template ReinterpretCast<int32_t>();
        auto groupLow=w[14*N];
        DataCopy(ae,expA,N);DataCopy(xe,expX,N);PipeBarrier<PIPE_ALL>();
        // Every lane reads all exponents, so the fallback decision is uniform.
        int32_t amin,amax,xmin,xmax;
        if constexpr(VectorBounds) {
            amin=ExponentExtreme<true>(ae,high,N);amax=ExponentExtreme<false>(ae,high,N);
            xmin=ExponentExtreme<true>(xe,high,t.k);xmax=ExponentExtreme<false>(xe,high,t.k);
        } else {
        amin=10000;amax=-10000;xmin=10000;xmax=-10000;
        for(uint32_t r=0;r<N;++r) {
            const int32_t av=(int32_t)ae.GetValue(r),xv=(int32_t)xe.GetValue(r);
            amin=av<amin?av:amin;amax=av>amax?av:amax;
            if(r<t.k) {xmin=xv<xmin?xv:xmin;xmax=xv>xmax?xv:xmax;}
        }
        }
        if(amax>=1000 || xmax>=1000 || amin+xmin<-50 || amax+xmax>100)return false;
        CreateVecIndex(ix,(int32_t)0,C);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)(t.k*4),C);PipeBarrier<PIPE_V>();
        for(uint32_t task=lane;task<t.k*R;task+=teams) {
            const uint32_t k=task/R,row=task%R*C,valid=row<t.n?(t.n-row<C?t.n-row:C):0;
            Duplicate(br,0.0f,C*t.k);Duplicate(low,0.0f,C);PipeBarrier<PIPE_ALL>();
            if(valid)DataCopyPad(br,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+row*t.k],DataCopyExtParams{1,valid*t.k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();Gather(high,br,ix.template ReinterpretCast<uint32_t>(),k*4,C);PipeBarrier<PIPE_V>();
            const int32_t xq=(int32_t)xe.GetValue(k);
            for(uint32_t d=0;d<Weights;++d) {
                if(d<2)Duplicate(gs,0.0f,C);else Duplicate(integerSum,(int32_t)0,C);PipeBarrier<PIPE_V>();
                const uint32_t first=d>3?d-3:0,count=(d<3?d:3)-first+1;
                // Same-weight pairs are three matrix planes apart. Fetch them
                // together, after the previous vector reads of tmp complete.
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(tmp,pairs[((uint64_t)(d+3*first)*t.k+k)*N+row],DataCopyExtParams{(uint16_t)count,C*4,(3*t.k*N-C)*4,(N-C)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
                if(d<2)for(uint32_t i=0;i<count;++i)A(gs,gs,tmp[i*N],C);
                else {
                    // A weight group may exceed 2^24. Sum in int32, then
                    // retain the float conversion's remainder as a low part.
                    for(uint32_t i=0;i<count;++i) {
                        Cast(rounded,tmp[i*N],RoundMode::CAST_RINT,C);PipeBarrier<PIPE_V>();Add(integerSum,integerSum,rounded,C);PipeBarrier<PIPE_V>();
                    }
                    Cast(gs,integerSum,RoundMode::CAST_NONE,C);PipeBarrier<PIPE_V>();Cast(rounded,gs,RoundMode::CAST_RINT,C);PipeBarrier<PIPE_V>();
                    Sub(remainder,integerSum,rounded,C);PipeBarrier<PIPE_V>();Cast(groupLow,remainder,RoundMode::CAST_NONE,C);PipeBarrier<PIPE_V>();
                }
                Adds(factor,ae[row],(float)(xq-8*(int32_t)d+127),C);PipeBarrier<PIPE_V>();
                Cast(factor.template ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,C);PipeBarrier<PIPE_V>();
                ShiftLeft(factor.template ReinterpretCast<int32_t>(),factor.template ReinterpretCast<int32_t>(),(int32_t)23,C);PipeBarrier<PIPE_V>();
                M(term,gs,factor,C);Scale(term,term,-1.0f,C);
                A(tmp,high,term,C);D(u,tmp,high,C);D(v,tmp,u,C);D(v,high,v,C);D(u,term,u,C);A(u,u,v,C);A(low,low,u,C);Scale(high,tmp,1.0f,C);
                if(d>=2) {M(term,groupLow,factor,C);Scale(term,term,-1.0f,C);A(low,low,term,C);}
            }
            A(high,high,low,C);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N+row],high,C);PipeBarrier<PIPE_ALL>();
        }
        return true;
    }
public:
    static __aicore__ inline void Service(GM_ADDR workspace,GM_ADDR system,const tiling::TCubeTiling &tc,const tiling::TCubeTiling &ht,GM_ADDR rhs,uint32_t batches,uint32_t order,uint32_t rhsCount,uint32_t broadcastB) {
        using F=MatmulType<TPosition::GM,CubeFormat::ND,float>;
        using H=MatmulType<TPosition::GM,CubeFormat::ND,half>;
        TPipe cubePipe;Matmul<F,F,F> mm;Matmul<H,H,F> hm;
        if constexpr(SplitResidual) {REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc,hm,&ht);}
        else {REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc);}
        mm.SetHF32(false);
        const uint64_t coeffStart=(uint64_t)batches*N*2*N,selectedStart=coeffStart+(uint64_t)batches*N*B;
        GlobalTensor<float> matrix,coefficient,selected;
        matrix.SetGlobalBuffer((__gm__ float*)workspace);
        coefficient.SetGlobalBuffer((__gm__ float*)workspace+coeffStart);
        selected.SetGlobalBuffer((__gm__ float*)workspace+selectedStart);
        const uint32_t tm=tc.singleCoreM,tn=tc.singleCoreN,mr=(N+tm-1)/tm,nr=(2*N+tn-1)/tn,per=mr*nr;
        for(uint32_t base=0;base<order;base+=B) {
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<batches*per;task+=GetBlockNum()) {
                const uint32_t batch=task/per,tile=task%per,row=tile/nr*tm,column=tile%nr*tn;
                // G is stored as B x N; the logical tile starts at column row.
                if constexpr(MatrixRhs) {mm.SetOrgShape(N,2*N,B);mm.SetSingleShape(tm,tn,B);}
                mm.SetTensorA(coefficient[(uint64_t)batch*N*B+row],true);
                mm.SetTensorB(selected[(uint64_t)batch*B*2*N+column]);
                if constexpr(MatrixRhs)mm.SetTail(N-row<tm?N-row:tm,2*N-column<tn?2*N-column:tn,B);
                else mm.SetTail(N-row<tm?N-row:tm,2*N-column<tn?2*N-column:tn);
                mm.IterateAll(matrix[(uint64_t)batch*N*2*N+row*2*N+column],1);mm.End();
            }
            SyncAll<false>();
        }
        if constexpr(MatrixRhs) {
            const uint64_t invStart=selectedStart+(uint64_t)batches*B*2*N+batches*B;
            const uint64_t initialStart=invStart+(uint64_t)batches*N*N+(uint64_t)batches*rhsCount*P;
            const uint64_t residualStart=initialStart+(uint64_t)batches*rhsCount*P;
            GlobalTensor<float> gb,inverse,initial,residual;
            gb.SetGlobalBuffer((__gm__ float*)rhs);inverse.SetGlobalBuffer((__gm__ float*)workspace+invStart);
            initial.SetGlobalBuffer((__gm__ float*)workspace+initialStart);residual.SetGlobalBuffer((__gm__ float*)workspace+residualStart);
            const uint32_t rblocks=(rhsCount+tm-1)/tm,cblocks=(N+tn-1)/tn,tiles=rblocks*cblocks;
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<batches*tiles;task+=GetBlockNum()) {
                const uint32_t batch=task/tiles,tile=task%tiles,row=tile/cblocks*tm,column=tile%cblocks*tn;
                mm.SetOrgShape(rhsCount,N,order,N,N);mm.SetSingleShape(tm,tn,order);
                mm.SetTensorA(gb[(uint64_t)(broadcastB?0:batch)*order*rhsCount+row],true);
                mm.SetTensorB(inverse[(uint64_t)batch*N*N+column*N],true);
                mm.SetTail(rhsCount-row<tm?rhsCount-row:tm,N-column<tn?N-column:tn,order);
                mm.IterateAll(initial[((uint64_t)batch*rhsCount+row)*P+column],0);mm.End();
            }
            SyncAll<false>();
            if constexpr(SplitResidual) {
                const uint64_t splitStart=residualStart+(uint64_t)batches*rhsCount*N;
                const uint64_t adWords=(uint64_t)batches*2*N*N,xdWords=(uint64_t)batches*2*rhsCount*N;
                GlobalTensor<half> ad,xd;GlobalTensor<float> products;
                ad.SetGlobalBuffer((__gm__ half*)((__gm__ float*)workspace+splitStart));
                xd.SetGlobalBuffer((__gm__ half*)((__gm__ float*)workspace+splitStart+adWords));
                products.SetGlobalBuffer((__gm__ float*)workspace+splitStart+adWords+xdWords);
                const uint32_t hmM=ht.singleCoreM,hmN=ht.singleCoreN,hmR=(rhsCount+hmM-1)/hmM,hmC=(N+hmN-1)/hmN,hmTiles=hmR*hmC;
                SyncAll<false>();
                for(uint32_t task=GetBlockIdx();task<batches*10*hmTiles;task+=GetBlockNum()) {
                    const uint32_t bt=task/(10*hmTiles),tile=task%hmTiles,row=tile/hmC*hmM,column=tile%hmC*hmN;
                    const uint32_t pair=task/hmTiles%10;
                    uint32_t i=0,j=pair;
                    while(j>=4-i) {j-=4-i;++i;}
                    hm.SetOrgShape(rhsCount,N,N);hm.SetSingleShape(hmM,hmN,N);
                    hm.SetTensorA(xd[((uint64_t)bt*4+j)*rhsCount*N+row*N],false);
                    hm.SetTensorB(ad[((uint64_t)bt*4+i)*N*N+column*N],true);
                    hm.SetTail(rhsCount-row<hmM?rhsCount-row:hmM,N-column<hmN?N-column:hmN,N);
                    hm.IterateAll(products[(((uint64_t)bt*16+i*4+j)*rhsCount+row)*N+column],0);hm.End();
                }
                SyncAll<false>();
            }
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<batches*tiles;task+=GetBlockNum()) {
                const uint32_t batch=task/tiles,tile=task%tiles,row=tile/cblocks*tm,column=tile%cblocks*tn;
                mm.SetOrgShape(rhsCount,N,N);mm.SetSingleShape(tm,tn,N);
                mm.SetTensorA(residual[((uint64_t)batch*rhsCount+row)*P],false);
                mm.SetTensorB(inverse[(uint64_t)batch*N*N+column*N],true);
                mm.SetTail(rhsCount-row<tm?rhsCount-row:tm,N-column<tn?N-column:tn,N);
                mm.IterateAll(initial[((uint64_t)batch*rhsCount+row)*P+column],1);mm.End();
            }
            SyncAll<false>();
        }
    }
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        const uint32_t cores=GetBlockNum()*2,id=GetBlockIdx(),batch=id%t.batches,lane=id/t.batches;
        const uint32_t teams=(cores+t.batches-1-batch)/t.batches,chunk=(N+teams-1)/teams,begin=lane*chunk,end=begin+chunk<N?begin+chunk:N;
        const uint32_t L=2*N,cw=(L+teams*8-1)/(teams*8)*8,firstColumn=lane*cw,width=firstColumn<L?(L-firstColumn<cw?L-firstColumn:cw):0;
        pipe.InitBuffer(buf,(CAP+128+2*CP+28*IT)*4);
        auto mat=buf.Get<float>(),pivot=mat[CAP],col=pivot[128],active=col[CP];
        auto indices=active[CP].template ReinterpretCast<uint32_t>();
        ixBase=indices;rhsIx=col.template ReinterpretCast<uint32_t>();
        a=mat;x=mat[SZ];h=mat[2*SZ];l=mat[3*SZ];ah=mat[4*SZ];al=mat[5*SZ];xh=mat[6*SZ];xl=mat[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=mat[(4+i)*SZ];
        raw=mat[8*SZ];ahi=mat[9*SZ];alo=mat[10*SZ];auto outputTile=mat[ARRAYS*SZ];
        GlobalTensor<float> ga,gb,gx,matrix,coefficient,selectedGM,ids,inverse,bcolumns,initial,residual;
        ga.SetGlobalBuffer((__gm__ float*)aa);gb.SetGlobalBuffer((__gm__ float*)bb);
        gx.SetGlobalBuffer((__gm__ float*)xx);
        const uint64_t coeffStart=(uint64_t)t.batches*N*L,selectedStart=coeffStart+(uint64_t)t.batches*N*B;
        const uint64_t idsStart=selectedStart+(uint64_t)t.batches*B*L,invStart=idsStart+t.batches*B;
        const uint64_t inputStart=invStart+(uint64_t)t.batches*N*N,initialStart=inputStart+(uint64_t)t.batches*t.k*P,residualStart=initialStart+(uint64_t)t.batches*t.k*P;
        auto gm=(__gm__ float*)workspace;
        matrix.SetGlobalBuffer(gm);coefficient.SetGlobalBuffer(gm+coeffStart);selectedGM.SetGlobalBuffer(gm+selectedStart);ids.SetGlobalBuffer(gm+idsStart);
        inverse.SetGlobalBuffer(gm+invStart);bcolumns.SetGlobalBuffer(gm+inputStart);initial.SetGlobalBuffer(gm+initialStart);residual.SetGlobalBuffer(gm+residualStart);
        uint32_t labels[N];for(uint32_t i=0;i<N;++i)labels[i]=i;
        for(uint32_t row=begin;row<end;row+=T) {
            const uint32_t rows=end-row<T?end-row:T;
            Duplicate(mat,0.0f,rows*L);PipeBarrier<PIPE_ALL>();
            const uint32_t realRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;
            if(realRows)DataCopyPad(mat,ga[(uint64_t)batch*t.n*t.n+row*t.n],DataCopyExtParams{(uint16_t)realRows,t.n*4,0,(L-(t.n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-t.n%8)%8),0});
            PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<rows;++r) {
                mat.SetValue(r*L+N+row+r,1.0f);
                if(row+r>=t.n)mat.SetValue(r*L+row+r,1.0f);
            }
            PipeBarrier<PIPE_ALL>();DataCopy(matrix[(uint64_t)batch*N*L+row*L],mat,rows*L);PipeBarrier<PIPE_ALL>();
        }
        auto matrixZeros=mat[CAP-L];
        static_assert(N*PL+B*(L/4)<=CAP-L);
        Duplicate(matrixZeros,0.0f,L);Duplicate(active,0.0f,CP);PipeBarrier<PIPE_V>();Duplicate(active,1.0f,t.n);Duplicate(col,0.0f,CP);
        PipeBarrier<PIPE_ALL>();SyncAll<true>();
        auto panel=mat[N*B],selected=mat[N*PL];
        auto transform=mat[N*B+2*PP*B];
        auto inputIndex=mat[INPUT_INDEX].ReinterpretCast<int32_t>();
        auto panelScratch=mat[PANEL_SCRATCH];
        // Selected-row storage is reused after the panel/transform is published.
        // Persistent indices and helper scratch stay above both phase layouts.
        static_assert(2*N*B+2*PP*B<=INPUT_INDEX);
        static_assert(N*PL+B*(L/4)<=INPUT_INDEX);
        static_assert(INPUT_INDEX+PP*B<=PANEL_SCRATCH);
        static_assert(PANEL_SCRATCH+ColumnPanel<N,PP,B>::SCRATCH<=CAP-L);
        if(lane==0) {
            auto rowIndex=panelScratch.ReinterpretCast<int32_t>();
            CreateVecIndex(rowIndex,(int32_t)0,N);PipeBarrier<PIPE_V>();
            Muls(rowIndex,rowIndex,(int32_t)(B*4),N);PipeBarrier<PIPE_V>();
            Duplicate(inputIndex,(int32_t)0,PP*B);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<B;++j)Adds(inputIndex[j*PP],rowIndex,(int32_t)(j*4),N);
        }
        for(uint32_t base=0;base<t.n;base+=B) {
            uint32_t panelRows[B]={};
            const uint32_t panelWidth=t.n-base<B?t.n-base:B;
            if(lane==0) {
                DataCopyPad(mat,matrix[(uint64_t)batch*N*L+base],DataCopyExtParams{(uint16_t)N,B*4,(L-B)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
                Gather(panel,mat,inputIndex.ReinterpretCast<uint32_t>(),0,PP*B);
                Duplicate(panel[PP*B],0.0f,PP*B);PipeBarrier<PIPE_V>();
                ColumnPanel<N,PP,B>::Run(panel,transform,active,panelScratch,t.n-base<B?t.n-base:B,panelRows,pipe);
                DataCopy(coefficient[(uint64_t)batch*N*B],transform,N*B);
                for(uint32_t j=0;j<B;++j)pivot.template ReinterpretCast<uint32_t>().SetValue(j,j<panelWidth?panelRows[j]:0);
                PipeBarrier<PIPE_ALL>();DataCopy(ids[batch*B],pivot,B);
            }
            SyncAll<true>();
            if(lane!=0) {
                DataCopy(pivot,ids[batch*B],B);PipeBarrier<PIPE_ALL>();
                for(uint32_t j=0;j<B;++j)panelRows[j]=pivot.template ReinterpretCast<uint32_t>().GetValue(j);
            }
            for(uint32_t j=0;j<panelWidth;++j)labels[panelRows[j]]=base+j;
            for(uint32_t j=lane;j<B;j+=teams) {
                if(j<panelWidth)DataCopy(selected[(j/teams)*L],matrix[(uint64_t)batch*N*L+panelRows[j]*L],L);
                else Duplicate(selected[(j/teams)*L],0.0f,L);
            }
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=lane;j<B;j+=teams) {
                DataCopy(selectedGM[((uint64_t)batch*B+j)*L],selected[(j/teams)*L],L);
                if(j<panelWidth)DataCopy(matrix[(uint64_t)batch*N*L+panelRows[j]*L],matrixZeros,L);
            }
            SyncAll<false>();SyncAll<false>();
        }
        for(uint32_t row=begin;row<end;row+=T) {
            const uint32_t rows=end-row<T?end-row:T;
            DataCopyPad(a,matrix[(uint64_t)batch*N*L+row*L+N],DataCopyExtParams{(uint16_t)rows,N*4,(L-N)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            for(uint32_t r=0;r<rows;++r)DataCopy(inverse[(uint64_t)batch*N*N+labels[row+r]*N],a[r*N],N);
            PipeBarrier<PIPE_ALL>();
            const uint32_t realRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;
            if constexpr(!MatrixRhs)for(uint32_t rb=0;rb<t.k;rb+=8) {
                const uint32_t width=t.k-rb<8?t.k-rb:8;
                Duplicate(a,0.0f,T*8);PipeBarrier<PIPE_ALL>();
                if(realRows)DataCopyPad(a,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+row*t.k+rb],DataCopyExtParams{(uint16_t)realRows,width*4,(t.k-width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-width),0});
                PipeBarrier<PIPE_ALL>();
                for(uint32_t k=rb;k<rb+width;++k) {
                    for(uint32_t r=0;r<rows;++r)rhsIx.SetValue(r,(r*8+k-rb)*4);
                    PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,0,rows);PipeBarrier<PIPE_ALL>();
                    DataCopyPad(bcolumns[((uint64_t)batch*t.k+k)*P+row],x,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
            }
        }
        SyncAll<true>();
        for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
            auto ix0=ixBase[4*IT*(4-half)],ix1=ix0[IT*half];
            for(uint32_t row=0;row<T;++row)for(uint32_t j=0;j<half;++j) {
                ix0.SetValue(row*half+j,(row*stride+j)*4);ix1.SetValue(row*half+j,(row*stride+half+j)*4);
            }
        }
        PipeBarrier<PIPE_ALL>();
        if constexpr(MatrixRhs) {SyncAll<false>();SyncAll<false>();}
        else {
        for(uint32_t row=begin;row<end;row+=T) {
            const uint32_t rows=end-row<T?end-row:T;
            LoadTile(inverse,(uint64_t)batch*N*N+row*N,rows,a);
            for(uint32_t k=0;k<t.k;++k) {
                LoadColumn(bcolumns,((uint64_t)batch*t.k+k)*P,rows);FastDot(rows);PipeBarrier<PIPE_ALL>();
                DataCopyPad(initial[((uint64_t)batch*t.k+k)*P+row],s[0],DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
        }
        }
        SyncAll<true>();
        bool assembled=false;
        if constexpr(SplitResidual) {
            const uint64_t splitStart=residualStart+(uint64_t)t.batches*t.k*N;
            const uint64_t adWords=(uint64_t)t.batches*2*N*N,xdWords=(uint64_t)t.batches*2*t.k*N,pairWords=(uint64_t)t.batches*16*t.k*N;
            GlobalTensor<half> ad,xd;GlobalTensor<float> products,expA,expX;
            ad.SetGlobalBuffer((__gm__ half*)(gm+splitStart)+(uint64_t)batch*4*N*N);
            xd.SetGlobalBuffer((__gm__ half*)(gm+splitStart+adWords)+(uint64_t)batch*4*t.k*N);
            products.SetGlobalBuffer(gm+splitStart+adWords+xdWords+(uint64_t)batch*16*t.k*N);
            expA.SetGlobalBuffer(gm+splitStart+adWords+xdWords+pairWords+(uint64_t)batch*2*N);
            expX.SetGlobalBuffer(gm+splitStart+adWords+xdWords+pairWords+(uint64_t)batch*2*N+N);
            for(uint32_t r=lane;r<N;r+=teams) {
                Duplicate(mat,0.0f,N);PipeBarrier<PIPE_ALL>();
                if(r<t.n)DataCopyPad(mat,ga[((uint64_t)batch*t.n+r)*t.n],DataCopyExtParams{1,t.n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();SplitRow(ad,expA,r,N);
                if(r<t.k) {
                    DataCopy(mat,initial[((uint64_t)batch*t.k+r)*N],N);PipeBarrier<PIPE_ALL>();SplitRow(xd,expX,r,t.k);
                } else {
                    mat.SetValue(0,0.0f);PipeBarrier<PIPE_ALL>();DataCopyPad(expX[r],mat,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
            }
            SyncAll<false>();SyncAll<false>();
            assembled=SplitAssemble(products,expA,expX,gb,residual,t,batch,lane,teams);
        }
        if(!assembled)for(uint32_t row=begin;row<end;row+=T) {
            const uint32_t rows=end-row<T?end-row:T;
            LoadInput(ga,(uint64_t)batch*t.n*t.n+row*t.n,rows,row<t.n?(t.n-row<rows?t.n-row:rows):0,t.n,raw);
            Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
            for(uint32_t k=0;k<t.k;++k) {
                LoadColumn(initial,((uint64_t)batch*t.k+k)*P,rows);AccurateDot(rows);PipeBarrier<PIPE_ALL>();
                const uint32_t rb=k/8*8,width=t.k-rb<8?t.k-rb:8;
                Duplicate(a,0.0f,T*8);PipeBarrier<PIPE_ALL>();
                const uint32_t validRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;
                if(validRows)DataCopyPad(a,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+row*t.k+rb],DataCopyExtParams{(uint16_t)validRows,width*4,(t.k-width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-width),0});PipeBarrier<PIPE_ALL>();
                for(uint32_t i=0;i<rows;++i)rhsIx.SetValue(i,(i*8+k-rb)*4);
                PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,0,rows);PipeBarrier<PIPE_V>();D(h,x,h,rows);D(h,h,l,rows);PipeBarrier<PIPE_ALL>();
                DataCopyPad(residual[((uint64_t)batch*t.k+k)*P+row],h,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
        }
        SyncAll<true>();
        if constexpr(MatrixRhs) {
            SyncAll<false>();SyncAll<false>();
            constexpr uint32_t OW=32;
            auto packed=mat[OW*IT];auto transposeIx=mat[2*OW*IT].template ReinterpretCast<int32_t>();
            CreateVecIndex(transposeIx,(int32_t)0,OW);PipeBarrier<PIPE_V>();
            Muls(transposeIx,transposeIx,(int32_t)(IT*4),OW);PipeBarrier<PIPE_V>();
            for(uint32_t r=1;r<T;++r)Adds(transposeIx[r*OW],transposeIx,(int32_t)(r*4),OW);
            PipeBarrier<PIPE_ALL>();
            for(uint32_t row=begin;row<end;row+=T) {
                const uint32_t rows=end-row<T?end-row:T;
                const uint32_t validRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;
                if(!validRows)continue;
                for(uint32_t rb=0;rb<t.k;rb+=OW) {
                    const uint32_t width=t.k-rb<OW?t.k-rb:OW;
                    Duplicate(a,0.0f,OW*IT);PipeBarrier<PIPE_ALL>();
                    DataCopyPad(a,initial[((uint64_t)batch*t.k+rb)*P+row],DataCopyExtParams{(uint16_t)width,rows*4,(P-rows)*4,(IT-(rows+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-rows%8)%8),0});
                    PipeBarrier<PIPE_ALL>();Gather(packed,a,transposeIx.template ReinterpretCast<uint32_t>(),0,rows*OW);PipeBarrier<PIPE_ALL>();
                    DataCopyPad(gx[(uint64_t)batch*t.n*t.k+row*t.k+rb],packed,DataCopyExtParams{(uint16_t)validRows,width*4,(OW-(width+7)/8*8)/8,(t.k-width)*4,0});PipeBarrier<PIPE_ALL>();
                }
            }
        } else {
        for(uint32_t row=begin;row<end;row+=T) {
            const uint32_t rows=end-row<T?end-row:T;
            LoadTile(inverse,(uint64_t)batch*N*N+row*N,rows,a);
            for(uint32_t rb=0;rb<t.k;rb+=8) {
                const uint32_t width=t.k-rb<8?t.k-rb:8;
                for(uint32_t k=rb;k<rb+width;++k) {
                LoadColumn(residual,((uint64_t)batch*t.k+k)*P,rows);FastDot(rows);PipeBarrier<PIPE_ALL>();
                DataCopyPad(x,initial[((uint64_t)batch*t.k+k)*P+row],DataCopyExtParams{1,rows*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
                A(x,x,s[0],rows);PipeBarrier<PIPE_ALL>();
                for(uint32_t i=0;i<rows;++i)outputTile.SetValue(i*8+k-rb,x.GetValue(i));
            }
                PipeBarrier<PIPE_ALL>();
                const uint32_t validRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;
                if(validRows)DataCopyPad(gx[(uint64_t)batch*t.n*t.k+row*t.k+rb],outputTile,DataCopyExtParams{(uint16_t)validRows,width*4,0,(t.k-width)*4,0});PipeBarrier<PIPE_ALL>();
            }
        }
        }

    }
};


// Apply the same per-panel elimination maps to the correction RHS.
// All factor data is produced from this invocation's A in workspace.
template<uint32_t N,uint32_t DotN=N,bool SplitResidual=false> class DirectPanel : public CubePanel<DotN> {
    using Base=CubePanel<DotN>;
    static_assert(!SplitResidual || (N<=256 && DotN==256));
    using Base::SplitRow;using Base::SplitAssembleRows;
    using Base::P;using Base::T;using Base::SZ;using Base::CP;using Base::IT;
    static constexpr uint32_t B=Base::B,PL=2*B+8,PP=Base::PP;using Base::ARRAYS;
    static constexpr uint32_t FactorCols=(N+B-1)/B*B;
    static constexpr uint32_t INPUT_INDEX=Base::INPUT_INDEX,PANEL_SCRATCH=Base::PANEL_SCRATCH;using Base::CAP;
    static_assert(N*PL+((B+3)/4)*(2*P)<=CAP-2*P);
    using Base::pipe;using Base::buf;using Base::a;using Base::x;
    using Base::ah;using Base::al;using Base::xh;using Base::xl;using Base::h;using Base::l;using Base::s;
    using Base::raw;using Base::ahi;using Base::alo;using Base::ixBase;using Base::rhsIx;
    using Base::Scale;using Base::D;using Base::LoadInput;using Base::LoadColumn;using Base::AccurateDot;
    template<HardEvent event> __aicore__ inline void Sync() {Base::template Sync<event>();}
    __aicore__ inline void ApplyFactors(GlobalTensor<float> factors,GlobalTensor<float> ids,GlobalTensor<float> initial,GlobalTensor<float> residual,
        const LinalgSolveTilingData &t,uint32_t batch,uint32_t lane,uint32_t teams,const uint32_t *order) {
        static_assert(2*PP*B+4*N+10*B<=CAP);
        if(lane>=t.k)return;
        auto w=buf.template Get<float>(),input=w,prod=w[PP*B],sol=w[2*PP*B],sums=sol[N],picked=sums[N],coeff=picked[B];
        auto perm=coeff[8*B].template ReinterpretCast<int32_t>();
        for(uint32_t row=0;row<N;++row)perm.SetValue(row,order[row]*4);PipeBarrier<PIPE_ALL>();
        if constexpr(FactorCols>N){Duplicate(perm[N],(int32_t)0,FactorCols-N);PipeBarrier<PIPE_ALL>();}
        for(uint32_t k=lane;k<t.k;k+=teams) {
            Duplicate(prod,0.0f,PP*B);PipeBarrier<PIPE_ALL>();
            DataCopy(sol,residual[((uint64_t)batch*t.k+k)*P],N);PipeBarrier<PIPE_ALL>();
            for(uint32_t base=0;base<t.n;base+=B) {
                const uint32_t pw=t.n-base<B?t.n-base:B;
                DataCopyPad(input,factors[((uint64_t)batch*FactorCols+base)*N],DataCopyExtParams{B,N*4,0,(PP-N)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
                Gather(picked,sol,perm[base].template ReinterpretCast<uint32_t>(),0,B);PipeBarrier<PIPE_V>();
                Brcb(coeff,picked,B/8,BrcbRepeatParams{1,8});
                Sync<HardEvent::V_S>();for(uint32_t j=0;j<pw;++j)sol.SetValue(order[base+j],0.0f);Sync<HardEvent::S_V>();
                for(uint32_t off=0;off<N;off+=64)Mul(prod[off],input[off],coeff,(uint64_t)64,(uint8_t)B,BinaryRepeatParams{1,1,0,PP/8,PP/8,1});
                PipeBarrier<PIPE_V>();
                for(uint32_t half=B/2;half;half/=2){Add(prod,prod,prod[half*PP],half*PP);PipeBarrier<PIPE_V>();}
                Add(sol,sol,prod,N);PipeBarrier<PIPE_ALL>();
            }
            Gather(sums,sol,perm.template ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_ALL>();
            DataCopy(sol,initial[((uint64_t)batch*t.k+k)*P],N);PipeBarrier<PIPE_ALL>();
            Add(sol,sol,sums,N);PipeBarrier<PIPE_ALL>();DataCopy(initial[((uint64_t)batch*t.k+k)*P],sol,N);PipeBarrier<PIPE_ALL>();
        }
    }

public:
    static __aicore__ inline void Service(GM_ADDR workspace,GM_ADDR system,const tiling::TCubeTiling &tc,uint32_t batches,uint32_t order,uint32_t rhsCount,const tiling::TCubeTiling *halfTiling=nullptr) {
        using Fp=MatmulType<TPosition::GM,CubeFormat::ND,float>;
        using Hp=MatmulType<TPosition::GM,CubeFormat::ND,half>;
        TPipe cubePipe;Matmul<Fp,Fp,Fp> mm;Matmul<Hp,Hp,Fp> hm;
        if constexpr(SplitResidual){REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc,hm,halfTiling);}else{REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc);}mm.SetHF32(false);
        const uint32_t L=2*P,usedColumns=P+(rhsCount+7)/8*8;

        GlobalTensor<float> matrix,factors,selected;
        matrix.SetGlobalBuffer((__gm__ float*)workspace);
        factors.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batches*N*L);
        selected.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batches*(N*L+FactorCols*N));
        const uint32_t tm=tc.singleCoreM,tn=tc.singleCoreN,mr=(N+tm-1)/tm;
        for(uint32_t base=0;base<order;base+=B) {

            SyncAll<false>();
            const uint32_t first=N>=512?(base+B)/tn*tn:base+B,nr=(usedColumns-first+tn-1)/tn,per=mr*nr;
            for(uint32_t task=GetBlockIdx();task<batches*per;task+=GetBlockNum()) {
                const uint32_t batch=task/per,tile=task%per,row=tile/nr*tm,column=first+tile%nr*tn;
                mm.SetOrgShape(N,L,B);mm.SetSingleShape(tm,tn,B);
                mm.SetTensorA(factors[((uint64_t)batch*FactorCols+base)*N+row],true);
                mm.SetTensorB(selected[(uint64_t)batch*B*L+column],false);
                mm.SetTail(N-row<tm?N-row:tm,usedColumns-column<tn?usedColumns-column:tn,B);
                mm.IterateAll(matrix[(uint64_t)batch*N*L+row*L+column],1);mm.End();
            }
            SyncAll<false>();
        }
        if constexpr(SplitResidual) {
            const uint64_t splitStart=(uint64_t)batches*(N*L+FactorCols*N+B*L+FactorCols+2*rhsCount*P);
            const uint64_t adWords=(uint64_t)batches*2*P*P,xdWords=(uint64_t)batches*2*rhsCount*P;
            GlobalTensor<half> ad,xd;GlobalTensor<float> products;
            ad.SetGlobalBuffer((__gm__ half*)((__gm__ float*)workspace+splitStart));
            xd.SetGlobalBuffer((__gm__ half*)((__gm__ float*)workspace+splitStart+adWords));
            products.SetGlobalBuffer((__gm__ float*)workspace+splitStart+adWords+xdWords);
            const auto &ht=*halfTiling;const uint32_t hmM=ht.singleCoreM,hmN=ht.singleCoreN,hmR=(rhsCount+hmM-1)/hmM,hmC=(P+hmN-1)/hmN,hmTiles=hmR*hmC;
            constexpr uint32_t pairs=N<P?13:10;
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<batches*pairs*hmTiles;task+=GetBlockNum()) {
                const uint32_t bt=task/(pairs*hmTiles),tile=task%hmTiles,row=tile/hmC*hmM,column=tile%hmC*hmN,pair=task/hmTiles%pairs;
                uint32_t i=0,j=pair;while(j>=(N<P?(i<2?4:5-i):4-i)){j-=(N<P?(i<2?4:5-i):4-i);++i;}
                hm.SetOrgShape(rhsCount,P,P);hm.SetSingleShape(hmM,hmN,P);
                hm.SetTensorA(xd[((uint64_t)bt*4+j)*rhsCount*P+row*P],false);
                hm.SetTensorB(ad[((uint64_t)bt*4+i)*P*P+column*P],true);
                hm.SetTail(rhsCount-row<hmM?rhsCount-row:hmM,P-column<hmN?P-column:hmN,P);
                hm.IterateAll(products[(((uint64_t)bt*16+i*4+j)*rhsCount+row)*P+column],0);hm.End();
            }
            SyncAll<false>();
        }
    }
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        const uint32_t cores=GetBlockNum()*2,id=GetBlockIdx(),batch=id%t.batches,lane=id/t.batches;
        const uint32_t teams=(cores+t.batches-1-batch)/t.batches,chunk=(N+teams-1)/teams,begin=lane*chunk,end=begin+chunk<N?begin+chunk:N;
        const uint32_t L=2*P,cw=(L+teams*8-1)/(teams*8)*8,firstColumn=lane*cw,width=firstColumn<L?(L-firstColumn<cw?L-firstColumn:cw):0;
        pipe.InitBuffer(buf,(CAP+128+2*CP+28*IT)*4);
        auto mat=buf.template Get<float>(),pivot=mat[CAP],col=pivot[128],active=col[CP];
        ixBase=active[CP].template ReinterpretCast<uint32_t>();rhsIx=col.template ReinterpretCast<uint32_t>();
        a=mat;x=mat[SZ];h=mat[2*SZ];l=mat[3*SZ];ah=mat[4*SZ];al=mat[5*SZ];xh=mat[6*SZ];xl=mat[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=mat[(4+i)*SZ];raw=mat[8*SZ];ahi=mat[9*SZ];alo=mat[10*SZ];auto outputTile=mat[ARRAYS*SZ];
        GlobalTensor<float> ga,gb,gx,matrix,factors,selectedGM,ids,initial,residual;
        ga.SetGlobalBuffer((__gm__ float*)aa);gb.SetGlobalBuffer((__gm__ float*)bb);gx.SetGlobalBuffer((__gm__ float*)xx);
        const uint64_t factorStart=(uint64_t)t.batches*N*L,selectedStart=factorStart+(uint64_t)t.batches*FactorCols*N;
        const uint64_t idsStart=selectedStart+(uint64_t)t.batches*B*L,initialStart=idsStart+t.batches*FactorCols,residualStart=initialStart+(uint64_t)t.batches*t.k*P;
        auto gm=(__gm__ float*)workspace;matrix.SetGlobalBuffer(gm);factors.SetGlobalBuffer(gm+factorStart);selectedGM.SetGlobalBuffer(gm+selectedStart);ids.SetGlobalBuffer(gm+idsStart);initial.SetGlobalBuffer(gm+initialStart);residual.SetGlobalBuffer(gm+residualStart);
        GlobalTensor<half> ad,xd;GlobalTensor<float> products,expA,expX;
        if constexpr(SplitResidual) {
            const uint64_t splitStart=residualStart+(uint64_t)t.batches*t.k*P;
            const uint64_t adWords=(uint64_t)t.batches*2*P*P,xdWords=(uint64_t)t.batches*2*t.k*P,pairWords=(uint64_t)t.batches*16*t.k*P;
            ad.SetGlobalBuffer((__gm__ half*)(gm+splitStart)+(uint64_t)batch*4*P*P);
            xd.SetGlobalBuffer((__gm__ half*)(gm+splitStart+adWords)+(uint64_t)batch*4*t.k*P);
            products.SetGlobalBuffer(gm+splitStart+adWords+xdWords+(uint64_t)batch*16*t.k*P);
            expA.SetGlobalBuffer(gm+splitStart+adWords+xdWords+pairWords+(uint64_t)batch*2*P);
            expX.SetGlobalBuffer(gm+splitStart+adWords+xdWords+pairWords+(uint64_t)batch*2*P+P);
            if(lane==0){Duplicate(mat,0.0f,P);PipeBarrier<PIPE_ALL>();DataCopy(expX,mat,P);PipeBarrier<PIPE_ALL>();}
        }

        uint32_t labels[N],order[N];for(uint32_t i=0;i<N;++i){labels[i]=i;order[i]=i;}
        for(uint32_t row=begin;row<end;row+=T) {
            const uint32_t rows=end-row<T?end-row:T,realRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;
            Duplicate(mat,0.0f,rows*L);PipeBarrier<PIPE_ALL>();
            if(realRows) {
                DataCopyPad(mat,ga[(uint64_t)batch*t.n*t.n+row*t.n],DataCopyExtParams{(uint16_t)realRows,t.n*4,0,(L-(t.n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-t.n%8)%8),0});
                DataCopyPad(mat[P],gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+row*t.k],DataCopyExtParams{(uint16_t)realRows,t.k*4,0,(L-(t.k+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-t.k%8)%8),0});
            }
            PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<rows;++r)if(row+r>=t.n)mat.SetValue(r*L+row+r,1.0f);
            PipeBarrier<PIPE_ALL>();DataCopy(matrix[(uint64_t)batch*N*L+row*L],mat,rows*L);PipeBarrier<PIPE_ALL>();
        }
        auto matrixZeros=mat[CAP-2*P];Duplicate(matrixZeros,0.0f,L);Duplicate(active,0.0f,CP);PipeBarrier<PIPE_V>();Duplicate(active,1.0f,t.n);PipeBarrier<PIPE_ALL>();SyncAll<true>();
        auto panel=mat[N*B],selected=mat[N*PL],transform=mat[N*B+2*PP*B];
        auto inputIndex=mat[INPUT_INDEX].template ReinterpretCast<int32_t>();auto panelScratch=mat[PANEL_SCRATCH];
        if(lane==0) {

            auto rowIndex=panelScratch.template ReinterpretCast<int32_t>();CreateVecIndex(rowIndex,(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(rowIndex,rowIndex,(int32_t)(B*4),N);PipeBarrier<PIPE_V>();
            Duplicate(inputIndex,(int32_t)0,PP*B);PipeBarrier<PIPE_V>();for(uint32_t j=0;j<B;++j)Adds(inputIndex[j*PP],rowIndex,(int32_t)(j*4),N);

        }
        for(uint32_t base=0;base<t.n;base+=B) {
            uint32_t panelRows[B]={};const uint32_t pw=t.n-base<B?t.n-base:B;

            if(lane==0) {
                DataCopyPad(mat,matrix[(uint64_t)batch*N*L+base],DataCopyExtParams{(uint16_t)N,B*4,(L-B)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
                Gather(panel,mat,inputIndex.template ReinterpretCast<uint32_t>(),0,PP*B);Duplicate(panel[PP*B],0.0f,PP*B);PipeBarrier<PIPE_V>();
                ColumnPanel<N,PP,B>::Run(panel,transform,active,panelScratch,pw,panelRows,pipe);
                DataCopy(factors[((uint64_t)batch*FactorCols+base)*N],transform,N*B);
                for(uint32_t j=0;j<B;++j)pivot.template ReinterpretCast<uint32_t>().SetValue(j,j<pw?panelRows[j]:0);
                PipeBarrier<PIPE_ALL>();DataCopy(ids[(uint64_t)batch*FactorCols+base],pivot,B);
            }
            if constexpr(SplitResidual)if(lane>0) {
                const uint32_t panels=(t.n+B-1)/B,step=(teams-1)*panels,first=(base/B)*(teams-1)+lane-1;
                for(uint32_t r=first;r<P;r+=step) {
                    Duplicate(mat,0.0f,P);PipeBarrier<PIPE_ALL>();
                    if(r<t.n)DataCopyPad(mat,ga[((uint64_t)batch*t.n+r)*t.n],DataCopyExtParams{1,t.n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                    PipeBarrier<PIPE_ALL>();SplitRow(ad,expA,r,P);
                }
            }
            SyncAll<true>();
            if(lane!=0){DataCopy(pivot,ids[(uint64_t)batch*FactorCols+base],B);PipeBarrier<PIPE_ALL>();for(uint32_t j=0;j<B;++j)panelRows[j]=pivot.template ReinterpretCast<uint32_t>().GetValue(j);}

            for(uint32_t j=0;j<pw;++j){labels[panelRows[j]]=base+j;order[base+j]=panelRows[j];}
            for(uint32_t j=lane;j<B;j+=teams) {
                if(j<pw)DataCopy(selected[(j/teams)*L],matrix[(uint64_t)batch*N*L+panelRows[j]*L],L);
                else Duplicate(selected[(j/teams)*L],0.0f,L);
            }
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=lane;j<B;j+=teams) {
                DataCopy(selectedGM[((uint64_t)batch*B+j)*L],selected[(j/teams)*L],L);
                if(j<pw)DataCopy(matrix[(uint64_t)batch*N*L+panelRows[j]*L],matrixZeros,L);
            }
            SyncAll<false>();SyncAll<false>();
        }
        // Reorder the directly transformed RHS into variable order.
        for(uint32_t k=lane;k<t.k;k+=teams) {
            for(uint32_t r=0;r<N;++r)rhsIx.SetValue(r,r*8*4);PipeBarrier<PIPE_ALL>();
            DataCopyPad(a,matrix[(uint64_t)batch*N*L+P+k],DataCopyExtParams{N,4,(L-1)*4,0,0},DataCopyPadExtParams<float>{true,0,7,0});PipeBarrier<PIPE_ALL>();
            Gather(x,a,rhsIx,0,N);PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<N;++r)rhsIx.SetValue(r,order[r]*4);PipeBarrier<PIPE_ALL>();
            Gather(a,x,rhsIx,0,N);if constexpr(P>N)Duplicate(a[N],0.0f,P-N);PipeBarrier<PIPE_ALL>();DataCopy(initial[((uint64_t)batch*t.k+k)*P],a,P);PipeBarrier<PIPE_ALL>();
            if constexpr(SplitResidual)SplitRow(xd,expX,k,t.k);
        }
        SyncAll<true>();
        for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
            auto ix0=ixBase[4*IT*(4-half)],ix1=ix0[IT*half];for(uint32_t row=0;row<T;++row)for(uint32_t j=0;j<half;++j){ix0.SetValue(row*half+j,(row*stride+j)*4);ix1.SetValue(row*half+j,(row*stride+half+j)*4);}
        }
        PipeBarrier<PIPE_ALL>();
        bool assembled=false;
        if constexpr(SplitResidual){SyncAll<false>();SyncAll<false>();assembled=Base::template SplitAssembleRows<(N<P?5:4)>(products,expA,expX,gb,residual,t,batch,lane,teams);}
        if(!assembled)for(uint32_t row=begin;row<end;row+=T) {
            const uint32_t rows=end-row<T?end-row:T;
            LoadInput(ga,(uint64_t)batch*t.n*t.n+row*t.n,rows,row<t.n?(t.n-row<rows?t.n-row:rows):0,t.n,raw);Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
            for(uint32_t k=0;k<t.k;++k) {
                LoadColumn(initial,((uint64_t)batch*t.k+k)*P,rows);AccurateDot(rows);PipeBarrier<PIPE_ALL>();
                const uint32_t rb=k/8*8,kw=t.k-rb<8?t.k-rb:8,validRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;
                Duplicate(a,0.0f,T*8);PipeBarrier<PIPE_ALL>();if(validRows)DataCopyPad(a,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+row*t.k+rb],DataCopyExtParams{(uint16_t)validRows,kw*4,(t.k-kw)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-kw),0});PipeBarrier<PIPE_ALL>();
                for(uint32_t i=0;i<rows;++i)rhsIx.SetValue(i,(i*8+k-rb)*4);PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,0,rows);PipeBarrier<PIPE_V>();D(h,x,h,rows);D(h,h,l,rows);PipeBarrier<PIPE_ALL>();DataCopyPad(residual[((uint64_t)batch*t.k+k)*P+row],h,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
        }
        SyncAll<true>();ApplyFactors(factors,ids,initial,residual,t,batch,lane,teams,order);SyncAll<true>();
        for(uint32_t row=begin;row<end;row+=T) {
            const uint32_t rows=end-row<T?end-row:T;
            for(uint32_t rb=0;rb<t.k;rb+=8) {
                const uint32_t kw=t.k-rb<8?t.k-rb:8;
                for(uint32_t k=rb;k<rb+kw;++k){DataCopyPad(x,initial[((uint64_t)batch*t.k+k)*P+row],DataCopyExtParams{1,rows*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();for(uint32_t i=0;i<rows;++i)outputTile.SetValue(i*8+k-rb,x.GetValue(i));}
                PipeBarrier<PIPE_ALL>();const uint32_t validRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;if(validRows)DataCopyPad(gx[(uint64_t)batch*t.n*t.k+row*t.k+rb],outputTile,DataCopyExtParams{(uint16_t)validRows,kw*4,0,(t.k-kw)*4,0});PipeBarrier<PIPE_ALL>();
            }
        }
    }
};

// Fixed-block normal-equation inverse: 256 narrow RHS and 512-capacity
// single-batch wide RHS. Padding and workspace follow the shared capacity.
// Normalize A by a power of two, form C^T C, and eliminate fixed SPD blocks
// against C^T. Apply the resulting C inverse to scaled RHS and residuals.
// The first correction uses FP32 residuals; the last uses compensated products.
template<uint32_t N,uint32_t B=32,uint32_t Refine=2,bool Grouped=(N>256)> class NormalInverse : public CubePanel<N> {
    static_assert((N==256 || N==384 || N==512) && B==32 && Refine==2);
    using Base=CubePanel<N>;
    using Base::SplitRow;
    using Base::pipe;using Base::buf;using Base::CAP;using Base::P;using Base::T;using Base::SZ;using Base::CP;using Base::IT;using Base::ARRAYS;
    using Base::a;using Base::x;using Base::h;using Base::l;using Base::ah;using Base::al;using Base::xh;using Base::xl;using Base::s;
    using Base::raw;using Base::ahi;using Base::alo;using Base::ixBase;using Base::rhsIx;
    using Base::A;using Base::Scale;using Base::D;using Base::LoadInput;using Base::LoadColumn;using Base::AccurateDot;
    __aicore__ inline void Invert(GlobalTensor<float> gram,GlobalTensor<float> inv,uint32_t batch,uint32_t base) {
        constexpr uint32_t R=2*B+8;
        auto w=buf.template Get<float>(),panel=w,pivot=w[B*R],col=pivot[2*B],weights=col[B];
        auto ix=weights[8*B].template ReinterpretCast<int32_t>();
        Duplicate(panel,0.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(panel,gram[((uint64_t)batch*N+base)*N+base],DataCopyExtParams{B,B*4,(N-B)*4,(R-B)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r)panel.SetValue(r*R+B+r,1.0f);
        for(uint32_t r=0;r<B;++r)ix.SetValue(r,r*R*4);PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            const float d=panel.GetValue(j*R+j);
            Muls(pivot,panel[j*R],-1.0f/d,(uint64_t)(2*B),1,UnaryRepeatParams{1,1,8,8});Gather(col,panel[j],ix.template ReinterpretCast<uint32_t>(),0,(uint64_t)B,1,8);PipeBarrier<PIPE_V>();
            Brcb(weights,col,B/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<2*B;c+=64)MulAddDst(panel[c],pivot[c],weights,(uint64_t)(2*B-c<64?2*B-c:64),(uint8_t)B,BinaryRepeatParams{1,1,0,R/8,0,1});
            PipeBarrier<PIPE_V>();Muls(panel[j*R],pivot,-1.0f,(uint64_t)(2*B),1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_ALL>();
        }
        Muls(panel,panel,-1.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(inv[((uint64_t)batch*N+base)*N+base],panel[B],DataCopyExtParams{B,B*4,(R-B)/8,(N-B)*4,0});PipeBarrier<PIPE_ALL>();
        // Subtract I from the current Gram block. Its product with -Gjj^-1
        // then gives Gjj^-1-I on the pivot rows, so the following accumulated
        // updates need no separate row-zeroing or coefficient override.
        DataCopyPad(panel,gram[((uint64_t)batch*N+base)*N+base],DataCopyExtParams{B,B*4,(N-B)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<B;++r)panel.SetValue(r*B+r,panel.GetValue(r*B+r)-1.0f);PipeBarrier<PIPE_ALL>();
        DataCopyPad(gram[((uint64_t)batch*N+base)*N+base],panel,DataCopyExtParams{B,B*4,0,(N-B)*4,0});PipeBarrier<PIPE_ALL>();
    }
    // Four signed base-128 digits. A stores ascending digits per row;
    // each X prefix is reversed so one int8 GEMM sums a common weight.
    template<bool Rhs,bool Chunk=false> __aicore__ inline void SplitRowInt8(GlobalTensor<int8_t> digits,GlobalTensor<float> exponents,uint32_t row,uint32_t rowCount,uint32_t chunkRow=0) {
        static_assert(!Chunk || (!Rhs && 40*N+16<=CAP));
        auto w=buf.template Get<float>(),v=Chunk?w[8*N+chunkRow*N]:w,q=w[N],ab=w[2*N],mx=w[3*N];
        auto hd=w[4*N].template ReinterpretCast<half>();auto bd=w[7*N].template ReinterpretCast<int8_t>();
        Abs(ab,v,N);PipeBarrier<PIPE_V>();WholeReduceMax(mx,ab,64,N/64,1,1,8);Base::template Sync<HardEvent::V_S>();
        uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);const int32_t e=(int32_t)((mb>>23)&255)-127;int32_t exponent=e-5;
        if((mb&0x7fffffff)==0){exponent=0;Duplicate(v,0.0f,N);}
        else if(e<-100 || e>100){exponent=1000;Duplicate(v,0.0f,N);}
        else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;Scale(v,v,scale.value,N);}
        for(uint32_t d=0;d<4;++d){
            Adds(q,v,12582912.0f,N);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,N);PipeBarrier<PIPE_V>();
            Cast(hd,q,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();Cast(bd[(Rhs?3-d:d)*N],hd,RoundMode::CAST_TRUNC,N);PipeBarrier<PIPE_V>();
            D(v,v,q,N);Scale(v,v,128.0f,N);
        }
        PipeBarrier<PIPE_ALL>();
        if constexpr(Chunk) {
            DataCopy(w[24*N].template ReinterpretCast<int8_t>()[chunkRow*4*N],bd,4*N);
            w[40*N].SetValue(chunkRow,(float)exponent);PipeBarrier<PIPE_ALL>();
        } else {
            if constexpr(Rhs){for(uint32_t d=0;d<4;++d)DataCopy(digits[((uint64_t)d*(d+1)/2*rowCount+(uint64_t)row*(d+1))*N],bd[(3-d)*N],(d+1)*N);}
            else DataCopy(digits[(uint64_t)row*4*N],bd,4*N);
            PipeBarrier<PIPE_ALL>();mx.SetValue(0,(float)exponent);PipeBarrier<PIPE_ALL>();DataCopyPad(exponents[row],mx,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    template<uint32_t G> __aicore__ inline bool SplitAssembleGrouped(GlobalTensor<int32_t> pairs,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> gb,GlobalTensor<float> residual,const LinalgSolveTilingData &t,uint32_t batch,uint32_t lane,uint32_t teams,float inputScale) {
        constexpr uint32_t L=G*N;static_assert(19*L<=CAP);
        auto w=buf.template Get<float>(),ae=w,xe=w[L],high=w[2*L],low=w[3*L],gs=w[4*L],tmp=w[5*L],term=w[6*L],u=w[7*L],v=w[8*L],factor=w[9*L],br=w[10*L];
        auto ix=w[18*L].template ReinterpretCast<int32_t>();
        auto integerSum=w[11*L].template ReinterpretCast<int32_t>(),rounded=w[12*L].template ReinterpretCast<int32_t>(),remainder=w[13*L].template ReinterpretCast<int32_t>();
        auto groupLow=w[14*L];
        DataCopy(ae,expA,N);DataCopy(xe,expX,N);PipeBarrier<PIPE_ALL>();
        const int32_t amin=Base::template ExponentExtreme<true>(ae,high,N),amax=Base::template ExponentExtreme<false>(ae,high,N);
        const int32_t xmin=Base::template ExponentExtreme<true>(xe,high,t.k),xmax=Base::template ExponentExtreme<false>(xe,high,t.k);
        if(amax>=1000 || xmax>=1000 || amin+xmin<-50 || amax+xmax>100)return false;
        for(uint32_t c=1;c<G;++c)DataCopy(ae[c*N],ae,N);PipeBarrier<PIPE_ALL>();
        CreateVecIndex(ix,(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)32,N);PipeBarrier<PIPE_V>();
        for(uint32_t c=1;c<G;++c)Adds(ix[c*N],ix,(int32_t)(c*4),N);PipeBarrier<PIPE_V>();
        for(uint32_t task=lane;task<(t.k+G-1)/G;task+=teams) {
            const uint32_t k=task*G,width=t.k-k<G?t.k-k:G;
            Duplicate(br,0.0f,N*8);Duplicate(low,0.0f,L);PipeBarrier<PIPE_ALL>();
            DataCopyPad(br,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+k],DataCopyExtParams{(uint16_t)t.n,width*4,(t.k-width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-width),0});
            PipeBarrier<PIPE_ALL>();Gather(high,br,ix.template ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();Muls(high,high,inputScale,L);PipeBarrier<PIPE_V>();
            for(uint32_t d=0;d<4;++d) {
                if(width<G){Duplicate(integerSum,(int32_t)0,L);PipeBarrier<PIPE_ALL>();}
                DataCopy(integerSum,pairs[((uint64_t)d*t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();
                // At most 4*512*64^2 < 2^24: conversion preserves the exact integer sum.
                Cast(gs,integerSum,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();
                for(uint32_t c=0;c<G;++c)Adds(factor[c*N],ae[c*N],xe.GetValue(k+c)-7.0f*d+127.0f,N);PipeBarrier<PIPE_V>();
                Cast(factor.template ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,L);PipeBarrier<PIPE_V>();
                ShiftLeft(factor.template ReinterpretCast<int32_t>(),factor.template ReinterpretCast<int32_t>(),(int32_t)23,L);PipeBarrier<PIPE_V>();
                Mul(term,gs,factor,L);PipeBarrier<PIPE_V>();Scale(term,term,-1.0f,L);
                A(tmp,high,term,L);D(u,tmp,high,L);D(v,tmp,u,L);D(v,high,v,L);D(u,term,u,L);A(u,u,v,L);A(low,low,u,L);Scale(high,tmp,1.0f,L);
            }
            A(high,high,low,L);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N],high,width*N);PipeBarrier<PIPE_ALL>();
        }
        return true;
    }
public:
    static __aicore__ inline void Service(GM_ADDR workspace,GM_ADDR system,const tiling::TCubeTiling &tc,uint32_t batches,uint32_t order,uint32_t rhsCount,const tiling::TCubeTiling *halfTiling=nullptr,GM_ADDR sourceA=nullptr) {
        using Fp=MatmulType<TPosition::GM,CubeFormat::ND,float>;
        using Hp=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
        using Ip=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
        TPipe cubePipe;Matmul<Fp,Fp,Fp> mm;Matmul<Hp,Hp,Ip> hm;REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc,hm,halfTiling);mm.SetHF32(false);
        auto gm=(__gm__ float*)workspace;
        GlobalTensor<float> input,gram,factors,inv,initial,residual,normalRhs;
        input.SetGlobalBuffer(gm);gram.SetGlobalBuffer(gm+(uint64_t)batches*N*N);factors.SetGlobalBuffer(gm+(uint64_t)batches*2*N*N);inv.SetGlobalBuffer(gm+(uint64_t)batches*3*N*N);
        initial.SetGlobalBuffer(gm+(uint64_t)batches*4*N*N);residual.SetGlobalBuffer(gm+(uint64_t)batches*(4*N*N+rhsCount*N));normalRhs.SetGlobalBuffer(gm+(uint64_t)batches*(4*N*N+2*rhsCount*N));
        GlobalTensor<float> selectedGram,selectedInverse;
        selectedGram.SetGlobalBuffer(gm+(uint64_t)batches*(4*N*N+3*rhsCount*N));selectedInverse.SetGlobalBuffer(gm+(uint64_t)batches*(4*N*N+3*rhsCount*N+B*N));
        const uint64_t splitStart=(uint64_t)batches*(4*N*N+3*rhsCount*N+2*B*N),adWords=(uint64_t)batches*4*N*N/4,xdWords=(uint64_t)batches*10*rhsCount*N/4;
        GlobalTensor<int8_t> ad,xd;GlobalTensor<int32_t> products;
        ad.SetGlobalBuffer((__gm__ int8_t*)(gm+splitStart));xd.SetGlobalBuffer((__gm__ int8_t*)(gm+splitStart+adWords));products.SetGlobalBuffer((__gm__ int32_t*)(gm+splitStart+adWords+xdWords));
        const auto &ht=*halfTiling;const uint32_t hmM=ht.singleCoreM,hmN=ht.singleCoreN,hmR=(rhsCount+hmM-1)/hmM,hmC=(N+hmN-1)/hmN,hmTiles=hmR*hmC;
        GlobalTensor<float> originalA;originalA.SetGlobalBuffer(order<N?gm+(uint64_t)batches*N*N:(__gm__ float*)sourceA);
        const uint32_t tm=tc.singleCoreM,tn=tc.singleCoreN,mr=(N+tm-1)/tm,nr=(N+tn-1)/tn;
        SyncAll<false>();
        for(uint32_t task=GetBlockIdx();task<batches*mr*nr;task+=GetBlockNum()) {
            const uint32_t batch=task/(mr*nr),tile=task%(mr*nr),row=tile/nr*tm,col=tile%nr*tn;
            mm.SetOrgShape(N,N,N);mm.SetSingleShape(tm,tn,N);mm.SetTensorA(input[(uint64_t)batch*N*N+row],true);mm.SetTensorB(input[(uint64_t)batch*N*N+col],false);mm.SetTail(N-row<tm?N-row:tm,N-col<tn?N-col:tn,N);mm.IterateAll(gram[((uint64_t)batch*N+row)*N+col],0);mm.End();
        }
        SyncAll<false>();
        for(uint32_t base=0;base<order;base+=B) {
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<batches*mr;task+=GetBlockNum()) {
                const uint32_t batch=task/mr,row=task%mr*tm;
                mm.SetOrgShape(N,N,N);mm.SetSingleShape(tm,tn,B);mm.SetTensorA(gram[((uint64_t)batch*N+row)*N+base],false);mm.SetTensorB(inv[((uint64_t)batch*N+base)*N+base],false);mm.SetTail(N-row<tm?N-row:tm,B,B);mm.IterateAll(factors[((uint64_t)batch*N+row)*N+base],0);mm.End();
            }
            SyncAll<false>();
            const uint32_t first=base+B<N?(base+B)/tn*tn:N,cc=(N-first+tn-1)/tn;
            for(uint32_t task=GetBlockIdx();task<batches*mr*cc;task+=GetBlockNum()) {
                const uint32_t batch=task/(mr*cc),tile=task%(mr*cc),row=tile/cc*tm,col=first+tile%cc*tn;
                mm.SetOrgShape(N,N,N);mm.SetSingleShape(tm,tn,B);mm.SetTensorA(factors[((uint64_t)batch*N+row)*N+base],false);mm.SetTensorB(selectedGram[(uint64_t)batch*B*N+col],false);mm.SetTail(N-row<tm?N-row:tm,N-col<tn?N-col:tn,B);mm.IterateAll(gram[((uint64_t)batch*N+row)*N+col],1);mm.End();
            }
            for(uint32_t task=GetBlockIdx();task<batches*mr*nr;task+=GetBlockNum()) {
                const uint32_t batch=task/(mr*nr),tile=task%(mr*nr),row=tile/nr*tm,col=tile%nr*tn;
                mm.SetOrgShape(N,N,N);mm.SetSingleShape(tm,tn,B);mm.SetTensorA(factors[((uint64_t)batch*N+row)*N+base],false);mm.SetTensorB(selectedInverse[(uint64_t)batch*B*N+col],false);mm.SetTail(N-row<tm?N-row:tm,N-col<tn?N-col:tn,B);mm.IterateAll(input[((uint64_t)batch*N+row)*N+col],1);mm.End();
            }
            SyncAll<false>();
        }
        const uint32_t kr=(rhsCount+tm-1)/tm;
        for(uint32_t pass=0;pass<=Refine;++pass) {
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<batches*kr*nr;task+=GetBlockNum()) {
                const uint32_t batch=task/(kr*nr),tile=task%(kr*nr),row=tile/nr*tm,col=tile%nr*tn;
                mm.SetOrgShape(rhsCount,N,N);mm.SetSingleShape(tm,tn,N);mm.SetTensorA(residual[((uint64_t)batch*rhsCount+row)*N],false);mm.SetTensorB(input[((uint64_t)batch*N+col)*N],true);mm.SetTail(rhsCount-row<tm?rhsCount-row:tm,N-col<tn?N-col:tn,N);mm.IterateAll(normalRhs[((uint64_t)batch*rhsCount+row)*N+col],0);mm.End();
            }
            SyncAll<false>();
            if(pass<Refine) {
                SyncAll<false>();
                if(pass==0) {
                    for(uint32_t task=GetBlockIdx();task<batches*kr*nr;task+=GetBlockNum()) {
                        const uint32_t batch=task/(kr*nr),tile=task%(kr*nr),row=tile/nr*tm,col=tile%nr*tn;
                        mm.SetOrgShape(rhsCount,N,N);mm.SetSingleShape(tm,tn,N);mm.SetTensorA(initial[((uint64_t)batch*rhsCount+row)*N],false);mm.SetTensorB(originalA[((uint64_t)batch*N+col)*N],true);mm.SetTail(rhsCount-row<tm?rhsCount-row:tm,N-col<tn?N-col:tn,N);mm.IterateAll(normalRhs[((uint64_t)batch*rhsCount+row)*N+col],0);mm.End();
                    }
                } else for(uint32_t task=GetBlockIdx();task<batches*4*hmTiles;task+=GetBlockNum()) {
                    const uint32_t bt=task/(4*hmTiles),tile=task%hmTiles,row=tile/hmC*hmM,column=tile%hmC*hmN,d=task/hmTiles%4,K=(d+1)*N;
                    hm.SetOrgShape(rhsCount,N,K,4*N);hm.SetSingleShape(hmM,hmN,K);
                    hm.SetTensorA(xd[((uint64_t)bt*10*rhsCount+(uint64_t)d*(d+1)/2*rhsCount+(uint64_t)row*(d+1))*N],false);
                    hm.SetTensorB(ad[((uint64_t)bt*N+column)*4*N],true);
                    hm.SetTail(rhsCount-row<hmM?rhsCount-row:hmM,N-column<hmN?N-column:hmN,K);hm.IterateAll(products[((uint64_t)bt*4*rhsCount+(uint64_t)d*rhsCount+row)*N+column],0);hm.End();
                }
                SyncAll<false>();
            }
        }
    }
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        const uint32_t cores=GetBlockNum()*2,id=GetBlockIdx(),batch=id%t.batches,lane=id/t.batches;
        const uint32_t teams=(cores+t.batches-1-batch)/t.batches,tiles=(N+T-1)/T;
        pipe.InitBuffer(buf,(CAP+128+2*CP+28*IT)*4);
        auto mat=buf.template Get<float>(),pivot=mat[CAP],col=pivot[128],active=col[CP];
        ixBase=active[CP].template ReinterpretCast<uint32_t>();rhsIx=col.template ReinterpretCast<uint32_t>();
        a=mat;x=mat[SZ];h=mat[2*SZ];l=mat[3*SZ];ah=mat[4*SZ];al=mat[5*SZ];xh=mat[6*SZ];xl=mat[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=mat[(4+i)*SZ];raw=mat[8*SZ];ahi=mat[9*SZ];alo=mat[10*SZ];auto outputTile=mat[ARRAYS*SZ];
        auto gm=(__gm__ float*)workspace;
        GlobalTensor<float> ga,gb,gx,input,gram,factors,inv,initial,residual,normalRhs;
        ga.SetGlobalBuffer((__gm__ float*)aa);gb.SetGlobalBuffer((__gm__ float*)bb);gx.SetGlobalBuffer((__gm__ float*)xx);
        input.SetGlobalBuffer(gm);gram.SetGlobalBuffer(gm+(uint64_t)t.batches*N*N);factors.SetGlobalBuffer(gm+(uint64_t)t.batches*2*N*N);inv.SetGlobalBuffer(gm+(uint64_t)t.batches*3*N*N);
        initial.SetGlobalBuffer(gm+(uint64_t)t.batches*4*N*N);residual.SetGlobalBuffer(gm+(uint64_t)t.batches*(4*N*N+t.k*N));normalRhs.SetGlobalBuffer(gm+(uint64_t)t.batches*(4*N*N+2*t.k*N));
        GlobalTensor<float> selectedGram,selectedInverse;
        selectedGram.SetGlobalBuffer(gm+(uint64_t)t.batches*(4*N*N+3*t.k*N));selectedInverse.SetGlobalBuffer(gm+(uint64_t)t.batches*(4*N*N+3*t.k*N+B*N));
        const uint64_t splitStart=(uint64_t)t.batches*(4*N*N+3*t.k*N+2*B*N),adWords=(uint64_t)t.batches*4*N*N/4,xdWords=(uint64_t)t.batches*10*t.k*N/4,pairWords=(uint64_t)t.batches*4*t.k*N;
        GlobalTensor<int8_t> ad,xd;GlobalTensor<int32_t> products;GlobalTensor<float> expA,expX;
        ad.SetGlobalBuffer((__gm__ int8_t*)(gm+splitStart)+(uint64_t)batch*4*N*N);xd.SetGlobalBuffer((__gm__ int8_t*)(gm+splitStart+adWords)+(uint64_t)batch*10*t.k*N);
        products.SetGlobalBuffer((__gm__ int32_t*)(gm+splitStart+adWords+xdWords)+(uint64_t)batch*4*t.k*N);expA.SetGlobalBuffer(gm+splitStart+adWords+xdWords+pairWords+(uint64_t)batch*2*N);expX.SetGlobalBuffer(gm+splitStart+adWords+xdWords+pairWords+(uint64_t)batch*2*N+N);
        if(lane==0){Duplicate(mat,0.0f,N);PipeBarrier<PIPE_ALL>();DataCopy(expX,mat,N);PipeBarrier<PIPE_ALL>();}
        uint32_t laneMax=0;
        for(uint32_t tile=lane;tile<tiles;tile+=teams) {
            const uint32_t row=tile*T;LoadInput(ga,(uint64_t)batch*t.n*t.n+row*t.n,T,row<t.n?(t.n-row<T?t.n-row:T):0,t.n,mat);
            DataCopy(input[((uint64_t)batch*N+row)*N],mat,T*N);PipeBarrier<PIPE_ALL>();
            Abs(x,mat,T*N);PipeBarrier<PIPE_V>();WholeReduceMax(h,x,64,T*N/64,1,1,8);Base::template Sync<HardEvent::V_S>();
            uint32_t bits,index;GetReduceRepeatMaxMinSpr<float>(bits,index);laneMax=bits>laneMax?bits:laneMax;
        }
        Duplicate(h,0.0f,8);PipeBarrier<PIPE_ALL>();h.template ReinterpretCast<uint32_t>().SetValue(0,laneMax);PipeBarrier<PIPE_ALL>();
        DataCopy(factors[(uint64_t)batch*N*N+lane*8],h,8);PipeBarrier<PIPE_ALL>();SyncAll<true>();
        DataCopy(a,factors[(uint64_t)batch*N*N],teams*8);PipeBarrier<PIPE_ALL>();
        uint32_t maxBits=0;for(uint32_t r=0;r<teams;++r){uint32_t value=a.template ReinterpretCast<uint32_t>().GetValue(r*8);maxBits=value>maxBits?value:maxBits;}
        int32_t exponent=(int32_t)((maxBits>>23)&255)-127;exponent=exponent>126?126:(exponent<-126?-126:exponent);
        union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;const float inputScale=scale.value;
        for(uint32_t tile=lane;tile<tiles;tile+=teams){const uint32_t row=tile*T;DataCopy(a,input[((uint64_t)batch*N+row)*N],T*N);PipeBarrier<PIPE_ALL>();Muls(a,a,inputScale,T*N);PipeBarrier<PIPE_ALL>();if(t.n<N){for(uint32_t r=0;r<T;++r)if(row+r>=t.n)a.SetValue(r*N+row+r,1.0f);PipeBarrier<PIPE_ALL>();}DataCopy(input[((uint64_t)batch*N+row)*N],a,T*N);PipeBarrier<PIPE_ALL>();}
        CreateVecIndex(rhsIx.template ReinterpretCast<int32_t>(),(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(rhsIx.template ReinterpretCast<int32_t>(),rhsIx.template ReinterpretCast<int32_t>(),(int32_t)(Grouped?32:t.k*4),N);PipeBarrier<PIPE_ALL>();
        constexpr uint32_t RG=8;
        auto groupIx=mat[3*SZ].template ReinterpretCast<int32_t>();
        if constexpr(!Grouped) {
        for(uint32_t k=lane;k<t.k;k+=teams) {
            DataCopy(mat,gb[(uint64_t)(t.broadcastB?0:batch)*N*t.k],N*t.k);PipeBarrier<PIPE_ALL>();Gather(x,mat,rhsIx,k*4,N);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N],x,N);PipeBarrier<PIPE_ALL>();
        }
        } else {
        for(uint32_t c=0;c<RG;++c)Adds(groupIx[c*N],rhsIx.template ReinterpretCast<int32_t>(),(int32_t)(c*4),N);PipeBarrier<PIPE_ALL>();
        for(uint32_t task=lane;task<(t.k+RG-1)/RG;task+=teams) {
            const uint32_t k=task*RG,width=t.k-k<RG?t.k-k:RG;
            Duplicate(mat,0.0f,N*8);PipeBarrier<PIPE_ALL>();
            DataCopyPad(mat,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+k],DataCopyExtParams{(uint16_t)t.n,width*4,(t.k-width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-width),0});
            PipeBarrier<PIPE_ALL>();Gather(x,mat,groupIx.template ReinterpretCast<uint32_t>(),0,RG*N);PipeBarrier<PIPE_ALL>();
            DataCopy(residual[((uint64_t)batch*t.k+k)*N],x,width*N);PipeBarrier<PIPE_ALL>();
        }
        }
        SyncAll<false>();SyncAll<false>();
        auto tileIndices=mat[4*SZ].template ReinterpretCast<int32_t>();
        CreateVecIndex(col.template ReinterpretCast<int32_t>(),(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(col.template ReinterpretCast<int32_t>(),col.template ReinterpretCast<int32_t>(),(int32_t)(T*4),N);PipeBarrier<PIPE_V>();
        for(uint32_t r=0;r<T;++r)Adds(tileIndices[r*N],col.template ReinterpretCast<int32_t>(),(int32_t)(r*4),N);PipeBarrier<PIPE_ALL>();
        for(uint32_t tile=lane;tile<tiles;tile+=teams) {
            const uint32_t row=tile*T;
            const uint32_t width=row<t.n?(t.n-row<T?t.n-row:T):0,paddedWidth=(width+7)/8*8;
            if(t.n<N){Duplicate(a,0.0f,T*N);PipeBarrier<PIPE_ALL>();}
            if(width)DataCopyPad(a,ga[(uint64_t)batch*t.n*t.n+row],DataCopyExtParams{(uint16_t)t.n,width*4,(t.n-width)*4,(T-paddedWidth)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(paddedWidth-width),0});PipeBarrier<PIPE_ALL>();
            Gather(x,a,tileIndices.template ReinterpretCast<uint32_t>(),0,T*N);PipeBarrier<PIPE_V>();Muls(x,x,inputScale,T*N);PipeBarrier<PIPE_ALL>();
            if(t.n<N){for(uint32_t r=0;r<T;++r)if(row+r>=t.n)x.SetValue(r*N+row+r,1.0f);PipeBarrier<PIPE_ALL>();}
            DataCopy(input[((uint64_t)batch*N+row)*N],x,T*N);PipeBarrier<PIPE_ALL>();
        }
        SyncAll<true>();
        for(uint32_t base=0;base<t.n;base+=B){
            if(lane==0)Invert(gram,inv,batch,base);
            else {
                const uint32_t share=(B+teams-2)/(teams-1),start=(lane-1)*share,count=start<B?(B-start<share?B-start:share):0;
                if(count){
                    // A four-lane team publishes up to ceil(32/3) rows per lane.
                    // Keep the two row groups disjoint at the larger capacities.
                    auto selectedData=mat[N>256?2*SZ:SZ];
                    DataCopy(a,gram[((uint64_t)batch*N+base+start)*N],count*N);DataCopy(selectedData,input[((uint64_t)batch*N+base+start)*N],count*N);PipeBarrier<PIPE_ALL>();
                    DataCopy(selectedGram[((uint64_t)batch*B+start)*N],a,count*N);DataCopy(selectedInverse[((uint64_t)batch*B+start)*N],selectedData,count*N);PipeBarrier<PIPE_ALL>();
                }
            }
            if(lane>0) {
                const uint32_t slots=((t.n+B-1)/B)*(teams-1),group=(N+slots-1)/slots;
                const uint32_t start=((base/B)*(teams-1)+lane-1)*group,count=start<N?(N-start<group?N-start:group):0;
                if(count) {
                    auto staged=mat[8*N],exponents=mat[40*N];auto packed=mat[24*N].template ReinterpretCast<int8_t>();
                    const uint32_t validRows=start<t.n?(t.n-start<count?t.n-start:count):0;
                    Duplicate(staged,0.0f,count*N);PipeBarrier<PIPE_ALL>();
                    if(validRows)DataCopyPad(staged,ga[((uint64_t)batch*t.n+start)*t.n],DataCopyExtParams{(uint16_t)validRows,t.n*4,0,(N-(t.n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-t.n%8)%8),0});
                    PipeBarrier<PIPE_ALL>();if constexpr(Grouped){Muls(staged,staged,inputScale,count*N);PipeBarrier<PIPE_V>();}
                    for(uint32_t row=0;row<count;++row)SplitRowInt8<false,true>(ad,expA,start+row,N,row);
                    PipeBarrier<PIPE_ALL>();DataCopy(ad[(uint64_t)start*4*N],packed,count*4*N);
                    DataCopyPad(expA[start],exponents,DataCopyExtParams{1,count*4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
            }
            SyncAll<false>();SyncAll<false>();SyncAll<false>();
        }
        // Gram is no longer needed after factorization; reuse it for padded A
        // in the first FP32 residual, avoiding an extra workspace matrix.
        if(t.n<N)for(uint32_t tile=lane;tile<tiles;tile+=teams) {
            const uint32_t row=tile*T;LoadInput(ga,(uint64_t)batch*t.n*t.n+row*t.n,T,row<t.n?(t.n-row<T?t.n-row:T):0,t.n,mat);
            DataCopy(gram[((uint64_t)batch*N+row)*N],mat,T*N);PipeBarrier<PIPE_ALL>();
        }
        for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
            auto ix0=ixBase[4*IT*(4-half)],ix1=ix0[IT*half];for(uint32_t row=0;row<T;++row)for(uint32_t j=0;j<half;++j){ix0.SetValue(row*half+j,(row*stride+j)*4);ix1.SetValue(row*half+j,(row*stride+half+j)*4);}
        }
        PipeBarrier<PIPE_ALL>();
        bool scaledResidual=false;
        for(uint32_t pass=0;pass<=Refine;++pass) {
            // Residual columns are assembled by multiple lanes. Finish every
            // segment before the column owner scales it in place.
            SyncAll<true>();
            if(!scaledResidual) {
            if constexpr(!Grouped) {
            for(uint32_t k=lane;k<t.k;k+=teams){DataCopy(a,residual[((uint64_t)batch*t.k+k)*N],N);PipeBarrier<PIPE_ALL>();Muls(a,a,inputScale,N);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N],a,N);PipeBarrier<PIPE_ALL>();}
            } else {
            for(uint32_t task=lane;task<(t.k+RG-1)/RG;task+=teams){const uint32_t k=task*RG,width=t.k-k<RG?t.k-k:RG;DataCopy(a,residual[((uint64_t)batch*t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();Muls(a,a,inputScale,width*N);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N],a,width*N);PipeBarrier<PIPE_ALL>();}
            }
            }
            SyncAll<false>();SyncAll<false>();
            if constexpr(!Grouped) {
            for(uint32_t k=lane;k<t.k;k+=teams) {
                DataCopy(a,normalRhs[((uint64_t)batch*t.k+k)*N],N);PipeBarrier<PIPE_ALL>();
                if(pass){DataCopy(x,initial[((uint64_t)batch*t.k+k)*N],N);PipeBarrier<PIPE_ALL>();Add(a,a,x,N);PipeBarrier<PIPE_ALL>();}
                DataCopy(initial[((uint64_t)batch*t.k+k)*N],a,N);PipeBarrier<PIPE_ALL>();
            }
            } else {
            for(uint32_t task=lane;task<(t.k+RG-1)/RG;task+=teams) {
                const uint32_t k=task*RG,width=t.k-k<RG?t.k-k:RG;
                DataCopy(a,normalRhs[((uint64_t)batch*t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();
                if(pass){DataCopy(x,initial[((uint64_t)batch*t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();Add(a,a,x,width*N);PipeBarrier<PIPE_ALL>();}
                DataCopy(initial[((uint64_t)batch*t.k+k)*N],a,width*N);PipeBarrier<PIPE_ALL>();
            }
            }
            SyncAll<true>();
            if(pass==Refine)break;
            if(pass>0)for(uint32_t k=lane;k<t.k;k+=teams){DataCopy(mat,initial[((uint64_t)batch*t.k+k)*N],N);PipeBarrier<PIPE_ALL>();SplitRowInt8<true>(xd,expX,k,t.k);}
            SyncAll<false>();SyncAll<false>();
            bool assembled=false;
            if(pass==0) {
                CreateVecIndex(rhsIx.template ReinterpretCast<int32_t>(),(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(rhsIx.template ReinterpretCast<int32_t>(),rhsIx.template ReinterpretCast<int32_t>(),(int32_t)(Grouped?32:t.k*4),N);PipeBarrier<PIPE_ALL>();
                if constexpr(!Grouped) {
                for(uint32_t k=lane;k<t.k;k+=teams){DataCopy(a,gb[(uint64_t)(t.broadcastB?0:batch)*N*t.k],N*t.k);DataCopy(h,normalRhs[((uint64_t)batch*t.k+k)*N],N);PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,k*4,N);PipeBarrier<PIPE_V>();Sub(x,x,h,N);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N],x,N);PipeBarrier<PIPE_ALL>();}
                } else {
                for(uint32_t c=0;c<RG;++c)Adds(groupIx[c*N],rhsIx.template ReinterpretCast<int32_t>(),(int32_t)(c*4),N);PipeBarrier<PIPE_ALL>();
                for(uint32_t task=lane;task<(t.k+RG-1)/RG;task+=teams) {
                    const uint32_t k=task*RG,width=t.k-k<RG?t.k-k:RG;
                    Duplicate(a,0.0f,N*8);PipeBarrier<PIPE_ALL>();
                    DataCopyPad(a,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+k],DataCopyExtParams{(uint16_t)t.n,width*4,(t.k-width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-width),0});
                    DataCopy(h,normalRhs[((uint64_t)batch*t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();
                    Gather(x,a,groupIx.template ReinterpretCast<uint32_t>(),0,RG*N);PipeBarrier<PIPE_V>();Sub(x,x,h,width*N);PipeBarrier<PIPE_ALL>();
                    DataCopy(residual[((uint64_t)batch*t.k+k)*N],x,width*N);PipeBarrier<PIPE_ALL>();
                }
                }
                assembled=true;
            } else assembled=SplitAssembleGrouped<4>(products,expA,expX,gb,residual,t,batch,lane,teams,Grouped?inputScale:1.0f);
            if(!assembled)for(uint32_t tile=lane;tile<tiles;tile+=teams) {
                const uint32_t row=tile*T,rows=T;
                LoadInput(ga,(uint64_t)batch*t.n*t.n+row*t.n,rows,row<t.n?(t.n-row<rows?t.n-row:rows):0,t.n,raw);Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
                for(uint32_t k=0;k<t.k;++k) {
                    LoadColumn(initial,((uint64_t)batch*t.k+k)*P,rows);AccurateDot(rows);PipeBarrier<PIPE_ALL>();
                    const uint32_t rb=k/8*8,kw=t.k-rb<8?t.k-rb:8,validRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;
                    Duplicate(a,0.0f,T*8);PipeBarrier<PIPE_ALL>();if(validRows)DataCopyPad(a,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+row*t.k+rb],DataCopyExtParams{(uint16_t)validRows,kw*4,(t.k-kw)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-kw),0});PipeBarrier<PIPE_ALL>();
                    for(uint32_t i=0;i<rows;++i)rhsIx.SetValue(i,(i*8+k-rb)*4);PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,0,rows);PipeBarrier<PIPE_V>();D(h,x,h,rows);D(h,h,l,rows);PipeBarrier<PIPE_ALL>();DataCopyPad(residual[((uint64_t)batch*t.k+k)*P+row],h,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
            }
            scaledResidual=Grouped && pass>0 && assembled;
        }
        if constexpr(!Grouped) {
        for(uint32_t tile=lane;tile<tiles;tile+=teams) {
            const uint32_t row=tile*T,rows=T;
            for(uint32_t rb=0;rb<t.k;rb+=8) {
                const uint32_t kw=t.k-rb<8?t.k-rb:8;
                for(uint32_t k=rb;k<rb+kw;++k){DataCopyPad(x,initial[((uint64_t)batch*t.k+k)*P+row],DataCopyExtParams{1,rows*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();for(uint32_t i=0;i<rows;++i)outputTile.SetValue(i*8+k-rb,x.GetValue(i));}
                PipeBarrier<PIPE_ALL>();const uint32_t validRows=row<t.n?(t.n-row<rows?t.n-row:rows):0;if(validRows)DataCopyPad(gx[(uint64_t)batch*t.n*t.k+row*t.k+rb],outputTile,DataCopyExtParams{(uint16_t)validRows,kw*4,0,(t.k-kw)*4,0});PipeBarrier<PIPE_ALL>();
            }
        }
        } else {
        constexpr uint32_t OW=32;
        auto packed=mat[OW*IT];auto transposeIx=mat[2*OW*IT].template ReinterpretCast<int32_t>();
        CreateVecIndex(transposeIx,(int32_t)0,OW);PipeBarrier<PIPE_V>();Muls(transposeIx,transposeIx,(int32_t)(IT*4),OW);PipeBarrier<PIPE_V>();
        for(uint32_t r=1;r<T;++r)Adds(transposeIx[r*OW],transposeIx,(int32_t)(r*4),OW);PipeBarrier<PIPE_ALL>();
        for(uint32_t tile=lane;tile<tiles;tile+=teams) {
            const uint32_t row=tile*T,rows=T,validRows=row<t.n?(t.n-row<T?t.n-row:T):0;if(!validRows)continue;
            for(uint32_t rb=0;rb<t.k;rb+=OW) {
                const uint32_t width=t.k-rb<OW?t.k-rb:OW;
                Duplicate(a,0.0f,OW*IT);PipeBarrier<PIPE_ALL>();
                DataCopyPad(a,initial[((uint64_t)batch*t.k+rb)*P+row],DataCopyExtParams{(uint16_t)width,rows*4,(P-rows)*4,(IT-(rows+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-rows%8)%8),0});
                PipeBarrier<PIPE_ALL>();Gather(packed,a,transposeIx.template ReinterpretCast<uint32_t>(),0,rows*OW);PipeBarrier<PIPE_ALL>();
                DataCopyPad(gx[(uint64_t)batch*t.n*t.k+row*t.k+rb],packed,DataCopyExtParams{(uint16_t)validRows,width*4,(OW-(width+7)/8*8)/8,(t.k-width)*4,0});PipeBarrier<PIPE_ALL>();
            }
        }
        }
    }
};


#include "linalg_solve_tiling.h"
// FP32 GEMM closure. Load3D handles the float transposes that generic Load2D cannot.
#include "kernel_operator.h"
using namespace AscendC;
__aicore__ inline void SyncCube() {
    PipeBarrier<PIPE_ALL>();
    ffts_cross_core_sync(PIPE_FIX,GetffstMsg(0x0,SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
}
namespace NarrowOrthogonal {
constexpr uint32_t CT=0,D=65536,H=131072,AN=196608,G=262144,GI=278528,
 BP=294912,XP=303104,R=311296,DX=319488,REDUCE=327680,AD=327936,XD=426240,PRODUCTS=447744,EA=472320,EX=472576,NT=472832,NH=489216,STRIDE=kNarrowOrthogonalWorkspaceFloats;
static_assert(NH+49152==STRIDE);
constexpr uint32_t FORWARD_MAP=G+6144,FORWARD_IDENTITY=AN+32768;
}

class NativeMM {
    TPipe pipe;
    TBuf<TPosition::A1> l1buf;
    TBuf<TPosition::A2> abuf;
    TBuf<TPosition::B2> bbuf;
    TBuf<TPosition::CO1> cbuf;
public:
    __aicore__ inline void Init() {
        pipe.InitBuffer(l1buf,262144);pipe.InitBuffer(abuf,65536);
        pipe.InitBuffer(bbuf,65536);pipe.InitBuffer(cbuf,131072);
        SetHF32Mode(false);SetHF32TransMode(false);
    }
    template<uint32_t BM,uint32_t BN,uint32_t BK,bool TA,bool TB,bool Accumulate>
    __aicore__ inline void Tile(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,
        uint32_t K,uint32_t as,uint32_t bs,uint32_t cs,uint32_t skip=0) {
        auto la=l1buf.Get<float>(),lb=la[16384],a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        uint8_t pad[4]={};SetFmatrix(1,BK,pad,FmatrixMode::FMATRIX_LEFT);
        for(uint32_t k=0;k<K;k+=BK) {
            PipeBarrier<PIPE_ALL>();
            Nd2NzParams p{};p.ndNum=1;p.nValue=TA?BK:BM;p.dValue=TA?BM:BK;p.srcDValue=as;p.dstNzC0Stride=TA?BK:BM;p.dstNzNStride=1;
            DataCopy(la,A[TA?k*as:k],p);
            p.nValue=TB?BN:BK;p.dValue=TB?BK:BN;p.srcDValue=bs;p.dstNzC0Stride=TB?BN:BK;
            DataCopy(lb,B[TB?k:k*bs],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            if constexpr(TA) {
                LoadData3DParamsV2Pro q{};q.channelSize=BM;q.extConfig=((uint64_t)BK<<16)|BM;q.enTranspose=true;
                LoadData<float>(a,la,q);
            } else {
                LoadData2DParams q{};q.repeatTimes=BK/8;q.srcStride=BM/16;
                for(uint32_t r=0;r<BM/16;++r)LoadData(a[r*BK*16],la[r*128],q);
            }
            if constexpr(TB) {
                LoadData2DParams q{};q.repeatTimes=BK*BN/128;q.srcStride=1;LoadData(b,lb,q);
            } else {
                LoadData3DParamsV2Pro q{};q.channelSize=BN;q.extConfig=((uint64_t)BK<<16)|BN;
                LoadData<float>(b,lb,q);
            }
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=BM;m.n=BN;m.k=BK;m.cmatrixInitVal=k==0;Mmad(c,a,b,m);
        }
        SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN-skip;f.srcStride=BM;f.dstStride=cs;
        if constexpr(Accumulate)SetAtomicAdd<float>();
        Fixpipe<float,float,CFG_ROW_MAJOR>(C[skip],c[skip*BM],f);PipeBarrier<PIPE_ALL>();
        if constexpr(Accumulate)SetAtomicNone();
    }
    template<uint32_t M,uint32_t N,uint32_t K,bool TA=false,bool TB=false,bool Accumulate=false>
    __aicore__ inline void Run(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,
        uint32_t batches,uint32_t astride,uint32_t bstride,uint32_t cstride,
        uint32_t as=TA?M:K,uint32_t bs=TB?K:N,uint32_t cs=N) {
        constexpr uint32_t BM=(M==128&&N>=128)?64:(M>=128?((M==192 || (N==32 && K>=128))?64:128):(M==64 && N>=128?64:32));
        constexpr uint32_t BN=N>=128?(M<=128?64:128):32;
        constexpr uint32_t BK=K==192 && BM<=64 && BN<=64?192:(K>=256 && BM<=64 && BN<=64?256:(K>=128?128:(K>=64?64:32)));
        constexpr uint32_t nr=M/BM,nc=N/BN;
        for(uint32_t task=GetBlockIdx();task<batches*nr*nc;task+=GetBlockNum()) {
            uint32_t batch=task/(nr*nc),r=task/nc%nr*BM,col=task%nc*BN;
            Tile<BM,BN,BK,TA,TB,Accumulate>(A[batch*astride+(TA?r:r*as)],B[batch*bstride+(TB?col*bs:col)],C[batch*cstride+r*cs+col],K,as,bs,cs);
        }
    }
    template<uint32_t K> __aicore__ inline void TileInt8(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t as,uint32_t bs) {
        constexpr uint32_t BM=K>1024?32:64,BN=16,BK=K;
        auto la=l1buf.Get<int8_t>(),lb=la[65536],a=abuf.Get<int8_t>(),b=bbuf.Get<int8_t>();auto c=cbuf.Get<int32_t>();
        for(uint32_t k=0;k<K;k+=BK) {
            PipeBarrier<PIPE_ALL>();Nd2NzParams p{};p.ndNum=1;p.nValue=BM;p.dValue=BK;p.srcDValue=as;p.dstNzC0Stride=BM;p.dstNzNStride=1;DataCopy(la,A[k],p);
            p.nValue=BN;p.dValue=BK;p.srcDValue=bs;p.dstNzC0Stride=BN;DataCopy(lb,B[k],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams q{};q.repeatTimes=BK/32;q.srcStride=BM/16;
            for(uint32_t r=0;r<BM/16;++r)LoadData(a[r*BK*16],la[r*512],q);
            q.repeatTimes=BK*BN/512;q.srcStride=1;LoadData(b,lb,q);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=BM;m.n=BN;m.k=BK;m.cmatrixInitVal=k==0;Mmad(c,a,b,m);
        }
        SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN;f.srcStride=BM;f.dstStride=16;
        Fixpipe<int32_t,int32_t,CFG_ROW_MAJOR>(C,c,f);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t K> __aicore__ inline void RunInt8(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t batches,uint32_t astride,uint32_t bstride,uint32_t cstride,uint32_t as,uint32_t bs) {
        constexpr uint32_t BM=K>1024?32:64,tiles=256/BM;
        for(uint32_t task=GetBlockIdx();task<batches*tiles;task+=GetBlockNum()) {uint32_t batch=task/tiles,row=task%tiles*BM;TileInt8<K>(A[batch*astride+row*as],B[batch*bstride],C[batch*cstride+row*16],as,bs);}
    }
    template<uint32_t K> __aicore__ inline void Int8Task(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t task,uint32_t stride) {
        constexpr uint32_t BM=K>1024?32:64,tiles=256/BM;
        uint32_t batch=task/tiles,row=task%tiles*BM;
        TileInt8<K>(A[batch*stride*4+row*1536],B[batch*stride*4],C[batch*stride+row*16],1536,K);
    }
    __aicore__ inline void Int8Packed4(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t batches,uint32_t stride) {
        constexpr uint32_t BM=32,BN=32,K=1536;
        auto la=l1buf.Get<int8_t>(),lb=la[65536],a=abuf.Get<int8_t>(),b=bbuf.Get<int8_t>();auto c=cbuf.Get<int32_t>();
        for(uint32_t task=GetBlockIdx();task<batches*8;task+=GetBlockNum()) {
            uint32_t batch=task/8,row=task%8*32;PipeBarrier<PIPE_ALL>();
            Nd2NzParams p{};p.ndNum=1;p.nValue=32;p.dValue=K;p.srcDValue=K;p.dstNzC0Stride=32;p.dstNzNStride=1;
            DataCopy(la,A[batch*stride*4+row*K],p);DataCopy(lb,B[batch*stride*4],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams q{};q.repeatTimes=K/32;q.srcStride=2;
            for(uint32_t r=0;r<2;++r)LoadData(a[r*K*16],la[r*512],q);
            q.repeatTimes=K*BN/512;q.srcStride=1;LoadData(b,lb,q);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=BM;m.n=BN;m.k=K;m.cmatrixInitVal=true;Mmad(c,a,b,m);
            SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN;f.srcStride=BM;f.dstStride=BN;
            Fixpipe<int32_t,int32_t,CFG_ROW_MAJOR>(C[batch*stride+row*BN],c,f);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Int8All(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*32;task+=GetBlockNum()) {
            if(task<batches*4)Int8Task<256>(A,B,C,task,stride);
            else if(task<batches*8)Int8Task<512>(A,B[4096],C[4096],task-batches*4,stride);
            else if(task<batches*12)Int8Task<768>(A,B[12288],C[8192],task-batches*8,stride);
            else if(task<batches*16)Int8Task<1024>(A,B[24576],C[12288],task-batches*12,stride);
            else if(task<batches*24)Int8Task<1280>(A,B[40960],C[16384],task-batches*16,stride);
            else Int8Task<1536>(A,B[61440],C[20480],task-batches*24,stride);
        }
    }

    __aicore__ inline void Local64(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum())
            Tile<64,64,64,false,false,false>(A[batch*stride],B[batch*stride],C[batch*stride],64,64,64,64);
    }

    template<uint32_t K,uint32_t AS,uint32_t BS,bool Cached=false,bool Accumulate=false>
    __aicore__ inline void HalfProduct(LocalTensor<half> la,LocalTensor<half> lb,LocalTensor<float> c) {
        PipeBarrier<PIPE_ALL>();auto a=abuf.Get<half>()[Cached?0:8192],b=bbuf.Get<half>();
        uint8_t pad[4]={};SetFmatrix(1,BS,pad,FmatrixMode::FMATRIX_LEFT);
        if constexpr(!Cached) {
            LoadData2DParams q{};q.repeatTimes=K/16;q.srcStride=AS/16;
            for(uint32_t r=0;r<4;++r)LoadData(a[r*K*16],la[r*256],q);
        }
        LoadData3DParamsV2Pro z{};z.channelSize=64;z.extConfig=((uint64_t)K<<16)|64;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=64;m.n=64;m.k=K;m.cmatrixInitVal=!Accumulate;Mmad(c,a,b,m);
        PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t DS> __aicore__ inline void HalfStore(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=DS;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot,uint32_t AS>
    __aicore__ inline void LoadPair64(LocalTensor<half> la,LocalTensor<half> lb) {
        auto a=abuf.Get<half>()[Slot*4096],b=bbuf.Get<half>()[Slot*4096];
        uint8_t pad[4]={};SetFmatrix(1,64,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=4;q.srcStride=AS/16;
        for(uint32_t row=0;row<4;++row)LoadData(a[row*1024],la[row*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=64;z.extConfig=((uint64_t)64<<16)|64;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(Slot);WaitFlag<HardEvent::MTE1_M>(Slot);
    }
    __aicore__ inline void NeumannPair64(LocalTensor<half> x,LocalTensor<half> r,LocalTensor<float> y,LocalTensor<float> tmp,bool square) {
        LoadPair64<0,192>(x,r);auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        MmadParams m{};m.m=64;m.n=64;m.k=64;m.cmatrixInitVal=false;Mmad(y,a,b,m);SetFlag<HardEvent::M_FIX>(0);
        if(square){LoadPair64<1,64>(r,r);m.cmatrixInitVal=true;Mmad(tmp,a[4096],b[4096],m);SetFlag<HardEvent::M_FIX>(1);}
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=192;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(x,y,f);
        if(square){WaitFlag<HardEvent::M_FIX>(1);f.dstStride=64;Fixpipe<half,float,CFG_NZ>(r,tmp,f);}
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Newton64(GlobalTensor<half> packed,GlobalTensor<float> result,GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        auto weights=l1buf.Get<half>(),values=weights[12288],x=values[1024],r=weights[24576],low=weights[8192];auto y=cbuf.Get<float>(),tmp=y[4096];
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            Nd2NzParams p{};p.ndNum=1;p.nValue=64;p.dValue=64;p.srcDValue=64;p.dstNzC0Stride=64;p.dstNzNStride=1;
            DataCopy(weights,packed[batch*stride*2+20480],p);DataCopy(weights[4096],packed[batch*stride*2],p);DataCopy(low,packed[batch*stride*2+16384],p);DataCopy(r,packed[batch*stride*2+12288],p);
            p.dstNzC0Stride=192;
            DataCopy(values,packed[batch*stride*2+8192],p);DataCopy(x,packed[batch*stride*2+4096],p);PipeBarrier<PIPE_ALL>();
            HalfProduct<64,192,192>(x,values,y);
            for(uint32_t iter=0;iter+1<steps;++iter) {
                NeumannPair64(x,r,y,tmp,iter+2<steps);
            }
            // Recompute the last Newton correction from G to remove residual-power drift.
            auto cachedA=abuf.Get<half>();LoadData2DParams q{};q.repeatTimes=8;q.srcStride=4;
            for(uint32_t row=0;row<4;++row)LoadData(cachedA[row*2048],weights[row*256],q);PipeBarrier<PIPE_ALL>();
            HalfProduct<128,64,192,true>(weights,values,tmp);HalfStore<64>(r,tmp);
            HalfProduct<64,192,64>(x,r,y);HalfStore<192>(x,y);
            // [I,G_hi,G_lo]*[I;Y;Y], then [Y,Y]*[I;E].
            HalfStore<192>(x[1024],y);
            q.repeatTimes=4;q.srcStride=12;
            for(uint32_t row=0;row<4;++row)LoadData(cachedA[row*3072],values[row*256],q);
            q.repeatTimes=8;q.srcStride=4;
            for(uint32_t row=0;row<4;++row)LoadData(cachedA[row*3072+1024],weights[4096+row*256],q);
            PipeBarrier<PIPE_ALL>();HalfProduct<192,64,192,true>(weights,values,tmp);
            q.repeatTimes=4;q.srcStride=12;
            for(uint32_t row=0;row<4;++row) {
                LoadData(cachedA[row*2048],x[row*256],q);
                LoadData(cachedA[row*2048+1024],x[row*256],q);
            }
            PipeBarrier<PIPE_ALL>();HalfStore<192>(x,tmp);
            HalfProduct<128,64,192,true>(weights,values,y);
            FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=64;
            Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);
            Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
        }
    }


    __aicore__ inline void TrailingH32(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,uint32_t batches,uint32_t stride,uint32_t cut) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,col=task%4*64;if(col+64<=cut)continue;
            Tile<32,64,256,false,true,false>(A[batch*stride],B[batch*stride+col*256],C[batch*stride+col],256,256,256,256,col<cut?cut-col:0);
        }
    }


    template<bool Add=false>
    __aicore__ inline void Product128(LocalTensor<half> la,LocalTensor<half> lb,LocalTensor<float> c) {
        PipeBarrier<PIPE_ALL>();auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        uint8_t pad[4]={};SetFmatrix(1,128,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=8;q.srcStride=8;
        for(uint32_t r=0;r<8;++r)LoadData(a[r*2048],la[r*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=128;z.extConfig=((uint64_t)128<<16)|128;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=128;m.n=128;m.k=128;m.cmatrixInitVal=!Add;Mmad(c,a,b,m);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Store128(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=128;f.nSize=128;f.srcStride=128;f.dstStride=128;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot>
    __aicore__ inline void LoadPair128(LocalTensor<half> la,LocalTensor<half> lb) {
        auto a=abuf.Get<half>()[Slot*16384],b=bbuf.Get<half>()[Slot*16384];
        uint8_t pad[4]={};SetFmatrix(1,128,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=8;q.srcStride=8;
        for(uint32_t row=0;row<8;++row)LoadData(a[row*2048],la[row*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=128;z.extConfig=((uint64_t)128<<16)|128;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(Slot);WaitFlag<HardEvent::MTE1_M>(Slot);
    }
    __aicore__ inline void NeumannPair128(LocalTensor<half> x,LocalTensor<half> r,LocalTensor<float> y,LocalTensor<float> tmp,bool square) {
        LoadPair128<0>(x,r);auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        MmadParams m{};m.m=128;m.n=128;m.k=128;m.cmatrixInitVal=false;Mmad(y,a,b,m);SetFlag<HardEvent::M_FIX>(0);
        if(square){LoadPair128<1>(r,r);m.cmatrixInitVal=true;Mmad(tmp,a[16384],b[16384],m);SetFlag<HardEvent::M_FIX>(1);}
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=128;f.nSize=128;f.srcStride=128;f.dstStride=128;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(x,y,f);
        if(square){WaitFlag<HardEvent::M_FIX>(1);Fixpipe<half,float,CFG_NZ>(r,tmp,f);}
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Newton128(GlobalTensor<half> packed,GlobalTensor<float> result,GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        auto g=l1buf.Get<half>(),low=g[16384],x=g[32768],eye=g[49152],two=g[65536],r=g[81920];auto y=cbuf.Get<float>(),tmp=y[16384];
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            Nd2NzParams p{};p.ndNum=1;p.nValue=128;p.dValue=128;p.srcDValue=128;p.dstNzC0Stride=128;p.dstNzNStride=1;
            auto src=packed[batch*stride*2];DataCopy(g,src,p);DataCopy(low,src[65536],p);DataCopy(x,src[16384],p);DataCopy(eye,src[32768],p);DataCopy(two,src[49152],p);DataCopy(r,src[81920],p);PipeBarrier<PIPE_ALL>();
            Product128(x,eye,y);
            for(uint32_t iter=0;iter+1<steps;++iter) {
                NeumannPair128(x,r,y,tmp,iter+2<steps);
            }
            Product128(eye,two,tmp);Product128<true>(g,x,tmp);Store128(r,tmp);
            Product128(x,r,y);Store128(x,y);
            Product128(eye,eye,tmp);Product128<true>(g,x,tmp);Product128<true>(low,x,tmp);Store128(r,tmp);
            Product128(x,eye,y);Product128<true>(x,r,y);
            FixpipeParamsV220 f{};f.mSize=128;f.nSize=128;f.srcStride=128;f.dstStride=128;
            Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
        }
    }
    template<uint32_t N,bool Add=false>
    __aicore__ inline void LocalRun(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum())Tile<N,N,N,false,false,Add>(A[batch*stride],B[batch*stride],C[batch*stride],N,128,128,128);
    }
    __aicore__ inline void TrailingH128(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*64,col=128+task%2*64;
            Tile<64,64,256,false,true,false>(A[batch*stride+row*256],B[batch*stride+col*256],C[batch*stride+row*256+col],256,256,256,256);
        }
    }
    __aicore__ inline void BacksolveMixed(GlobalTensor<float> H,GlobalTensor<float> X,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            auto h=H[batch*stride],x=X[batch*stride];
            Tile<32,32,64,false,false,true>(h[160*256+192],x[192*32],x[160*32],64,256,32,32);
            Tile<32,32,128,false,false,true>(h[128*256+128],x[128*32],x[128*32],128,256,32,32);
            Tile<128,32,128,false,false,true>(h[128],x[128*32],x,128,256,32,32);
        }
    }    template<bool Add=false>
    __aicore__ inline void Product96(LocalTensor<half> la,LocalTensor<half> lb,LocalTensor<float> c) {
        PipeBarrier<PIPE_ALL>();auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        uint8_t pad[4]={};SetFmatrix(1,96,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=6;q.srcStride=6;
        for(uint32_t r=0;r<6;++r)LoadData(a[r*1536],la[r*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=96;z.extConfig=((uint64_t)96<<16)|96;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=96;m.n=96;m.k=96;m.cmatrixInitVal=!Add;Mmad(c,a,b,m);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Store96(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=96;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot>
    __aicore__ inline void LoadPair96(LocalTensor<half> la,LocalTensor<half> lb) {
        auto a=abuf.Get<half>()[Slot*9216],b=bbuf.Get<half>()[Slot*9216];
        uint8_t pad[4]={};SetFmatrix(1,96,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=6;q.srcStride=6;
        for(uint32_t row=0;row<6;++row)LoadData(a[row*1536],la[row*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=96;z.extConfig=((uint64_t)96<<16)|96;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(Slot);WaitFlag<HardEvent::MTE1_M>(Slot);
    }
    __aicore__ inline void NeumannPair96(LocalTensor<half> x,LocalTensor<half> r,LocalTensor<float> y,LocalTensor<float> tmp,bool square) {
        LoadPair96<0>(x,r);auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        MmadParams m{};m.m=96;m.n=96;m.k=96;m.cmatrixInitVal=false;Mmad(y,a,b,m);SetFlag<HardEvent::M_FIX>(0);
        if(square){LoadPair96<1>(r,r);m.cmatrixInitVal=true;Mmad(tmp,a[9216],b[9216],m);SetFlag<HardEvent::M_FIX>(1);}
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=96;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(x,y,f);
        if(square){WaitFlag<HardEvent::M_FIX>(1);Fixpipe<half,float,CFG_NZ>(r,tmp,f);}
        PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot>
    __aicore__ inline void RefineLoadA96(LocalTensor<half> src) {
        static_assert(Slot<3);
        auto dst=abuf.Get<half>()[Slot*9216];
        LoadData2DParams q{};q.repeatTimes=6;q.srcStride=6;
        for(uint32_t row=0;row<6;++row)LoadData(dst[row*1536],src[row*256],q);
    }
    template<uint32_t Slot>
    __aicore__ inline void RefineLoadB96(LocalTensor<half> src) {
        static_assert(Slot<3);
        auto dst=bbuf.Get<half>()[Slot*9216];uint8_t pad[4]={};
        SetFmatrix(1,96,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData3DParamsV2Pro z{};z.channelSize=96;z.extConfig=((uint64_t)96<<16)|96;
        LoadData<half>(dst,src,z);
    }
    template<uint32_t AS,uint32_t BS,bool Add>
    __aicore__ inline void RefineProduct96(LocalTensor<float> dst) {
        MmadParams m{};m.m=96;m.n=96;m.k=96;m.cmatrixInitVal=!Add;
        Mmad(dst,abuf.Get<half>()[AS*9216],bbuf.Get<half>()[BS*9216],m);
    }
    __aicore__ inline void RefineStore96(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=96;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);
    }
    template<bool Second>
    __aicore__ __attribute__((noinline)) void RefineResident96() {
        auto g=l1buf.Get<half>(),low=g[9216],x=g[18432],eye=g[27648],two=g[36864],r=g[46080],minus=g[55296];
        auto y=cbuf.Get<float>(),tmp=y[9216];
        if constexpr(Second) {
        RefineLoadA96<1>(eye);
        RefineLoadB96<2>(two);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,2,false>(tmp);
        RefineLoadA96<0>(g);
        RefineLoadB96<1>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,1,true>(tmp);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(r,tmp);
        RefineLoadA96<2>(x);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(r);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,false>(y);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(x,y);
        RefineLoadB96<0>(eye);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,0,false>(tmp);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<1>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,1,true>(tmp);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        RefineLoadA96<1>(low);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,1,true>(tmp);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(r,tmp);
        RefineLoadA96<2>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,0,false>(y);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(r);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,true>(y);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(x,y);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadA96<2>(x);
        RefineLoadB96<2>(minus);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,true>(y);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(two,y);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        RefineLoadA96<2>(eye);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,0,false>(tmp);
        RefineLoadB96<1>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,1,true>(tmp);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,1,true>(tmp);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(two);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,2,true>(tmp);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,2,true>(tmp);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(r,tmp);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        RefineLoadA96<2>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,0,false>(y);
        RefineLoadA96<1>(two);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,0,true>(y);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(r);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,true>(y);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,2,true>(y);
        } else {
        RefineLoadA96<1>(eye);
        RefineLoadB96<2>(two);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,2,false>(tmp);
        RefineLoadA96<0>(g);
        RefineLoadB96<1>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,1,true>(tmp);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(r,tmp);
        RefineLoadA96<2>(x);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(r);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,false>(y);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(x,y);
        RefineLoadB96<0>(eye);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,0,false>(tmp);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<1>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,1,true>(tmp);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        RefineLoadA96<1>(low);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,1,true>(tmp);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(r,tmp);
        RefineLoadA96<2>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,0,false>(y);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(r);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,true>(y);
        }
        PipeBarrier<PIPE_ALL>();
    }
    template<bool Second=false>
    __aicore__ inline void Newton96(GlobalTensor<half> packed,GlobalTensor<float> result,GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        auto g=l1buf.Get<half>(),low=g[9216],x=g[18432],eye=g[27648],two=g[36864],r=g[46080],minus=g[55296];auto y=cbuf.Get<float>(),tmp=y[9216];
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            Nd2NzParams p{};p.ndNum=1;p.nValue=96;p.dValue=96;p.srcDValue=96;p.dstNzC0Stride=96;p.dstNzNStride=1;
            auto src=packed[batch*stride*2];DataCopy(g,src,p);DataCopy(low,src[36864],p);DataCopy(x,src[9216],p);DataCopy(eye,src[18432],p);DataCopy(two,src[27648],p);DataCopy(r,src[46080],p);if constexpr(Second)DataCopy(minus,src[55296],p);PipeBarrier<PIPE_ALL>();
            Product96(x,eye,y);
            for(uint32_t iter=0;iter+1<steps;++iter) {
                NeumannPair96(x,r,y,tmp,iter+2<steps);
            }
            RefineResident96<Second>();
            FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=128;
            Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
        }
    }

    // Helpers form the actual updated-panel Gram while owners iterate on the seed.
    // Split it in Cube: high=half(G), low=half(G-high*(-(-I))).
    __aicore__ __attribute__((noinline)) void RefreshSecondGram96(GlobalTensor<float> mem,uint32_t batches) {
        using namespace NarrowOrthogonal;
        constexpr uint32_t stride=STRIDE;
        auto ct=mem[CT];GlobalTensor<half> packed,truth;
        packed.SetGlobalBuffer((__gm__ half*)mem[NH].GetPhyAddr());
        truth.SetGlobalBuffer((__gm__ half*)mem[PRODUCTS].GetPhyAddr());
        uint32_t worker=GetBlockIdx()-batches,workers=GetBlockNum()-batches;
        auto la=l1buf.Get<float>(),lb=la[16384];
        auto hi=l1buf.Get<half>()[65536],minus=l1buf.Get<half>()[69632];
        auto a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        for(uint32_t task=worker;task<batches*6;task+=workers) {
            const uint32_t batch=task/6,row=task%6/2*32,col=task%2*48;
            PipeBarrier<PIPE_ALL>();
            uint8_t pad[4]={};SetFmatrix(1,208,pad,FmatrixMode::FMATRIX_LEFT);
            Nd2NzParams p{};p.ndNum=1;p.nValue=32;p.dValue=208;p.srcDValue=256;p.dstNzC0Stride=32;p.dstNzNStride=1;
            DataCopy(la,ct[batch*stride+(96+row)*256],p);
            p.nValue=48;p.dstNzC0Stride=48;
            DataCopy(lb,ct[batch*stride+(96+col)*256],p);
            p.nValue=48;p.dValue=48;p.srcDValue=96;p.dstNzC0Stride=48;
            DataCopy(minus,packed[batch*stride*2+55296],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams q{};q.repeatTimes=26;q.srcStride=2;
            for(uint32_t r=0;r<2;++r)LoadData(a[r*3328],la[r*128],q);
            q.repeatTimes=78;q.srcStride=1;LoadData(b,lb,q);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=32;m.n=48;m.k=208;m.cmatrixInitVal=true;Mmad(c,a,b,m);
            SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
            FixpipeParamsV220 f{};f.mSize=32;f.nSize=48;f.srcStride=32;f.dstStride=96;f.quantPre=QuantMode_t::F322F16;
            Fixpipe<half,float,CFG_ROW_MAJOR>(truth[batch*stride*2+row*96+col],c,f);
            f.dstStride=32;Fixpipe<half,float,CFG_NZ>(hi,c,f);
            PipeBarrier<PIPE_ALL>();
            auto ah=abuf.Get<half>(),bh=bbuf.Get<half>();
            SetFmatrix(1,48,pad,FmatrixMode::FMATRIX_LEFT);
            q.repeatTimes=3;q.srcStride=2;
            for(uint32_t r=0;r<2;++r)LoadData(ah[r*768],hi[r*256],q);
            LoadData3DParamsV2Pro z{};z.channelSize=48;z.extConfig=((uint64_t)48<<16)|48;LoadData<half>(bh,minus,z);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            m.k=48;m.cmatrixInitVal=false;Mmad(c,ah,bh,m);
            SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
            f.dstStride=96;Fixpipe<half,float,CFG_ROW_MAJOR>(truth[batch*stride*2+9216+row*96+col],c,f);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void SecondNewtonRefreshed96(GlobalTensor<float> mem,uint32_t batches) {
        using namespace NarrowOrthogonal;
        constexpr uint32_t stride=STRIDE;
        const uint32_t batch=GetBlockIdx();
        if(batch<batches) {
            auto base=l1buf.Get<half>(),x=base[18432],eye=base[27648],two=base[36864],r=base[46080],minus=base[55296];
            auto y=cbuf.Get<float>(),tmp=y[9216];
            GlobalTensor<half> packed;packed.SetGlobalBuffer((__gm__ half*)mem[NH].GetPhyAddr());
            Nd2NzParams p{};p.ndNum=1;p.nValue=96;p.dValue=96;p.srcDValue=96;p.dstNzC0Stride=96;p.dstNzNStride=1;
            auto src=packed[batch*stride*2];
            DataCopy(x,src[9216],p);DataCopy(eye,src[18432],p);DataCopy(two,src[27648],p);DataCopy(r,src[46080],p);DataCopy(minus,src[55296],p);PipeBarrier<PIPE_ALL>();
            Product96(x,eye,y);
            for(uint32_t iter=0;iter<13;++iter)NeumannPair96(x,r,y,tmp,iter<12);
        } else {
            RefreshSecondGram96(mem,batches);
            CrossGramFuture208(mem[CT],mem[AN],batches,stride,96);
        }
        SyncCube();
        if(batch<batches) {
            // L1/L0 contents persist; descriptors need not live through helper calls.
            auto g=l1buf.Get<half>(),low=g[9216],x=g[18432],eye=g[27648],two=g[36864],r=g[46080],minus=g[55296];
            auto y=cbuf.Get<float>(),tmp=y[9216];
            auto result=mem[GI],defect=mem[NT];
            GlobalTensor<half> truth;truth.SetGlobalBuffer((__gm__ half*)mem[PRODUCTS].GetPhyAddr());
            Nd2NzParams p{};p.ndNum=1;p.nValue=96;p.dValue=96;p.srcDValue=96;p.dstNzC0Stride=96;p.dstNzNStride=1;
            DataCopy(g,truth[batch*stride*2],p);DataCopy(low,truth[batch*stride*2+9216],p);PipeBarrier<PIPE_ALL>();
            RefineResident96<true>();
            FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=128;
            Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
        }
    }

    template<uint32_t Active=256>
    __aicore__ inline void Gram96(GlobalTensor<float> ct,GlobalTensor<float> g,uint32_t batches,uint32_t stride,uint32_t panel) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*48,col=task%2*48;
            Tile<48,48,Active,false,true,false>(ct[batch*stride+(panel+row)*256],ct[batch*stride+(panel+col)*256],g[batch*stride+row*128+col],Active,256,256,128);
        }
    }
    __aicore__ inline void Factor96(GlobalTensor<float> gi,GlobalTensor<float> ct,GlobalTensor<float> d,uint32_t batches,uint32_t stride,uint32_t panel) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*48,col=task%2*128;
            Tile<48,128,96,false,false,false>(gi[batch*stride+row*128],ct[batch*stride+panel*256+col],d[batch*stride+(panel+row)*256+col],96,128,256,256);
        }
    }
    template<uint32_t Active=256>
    __aicore__ inline void TrailingH96(GlobalTensor<float> d,GlobalTensor<float> ct,GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*48;
            if constexpr(Active==208) {
                if(panel==0) {
                    if(task%2==0)Tile<48,64,208,false,true,false>(d[batch*stride+row*256],ct[batch*stride+96*256],h[batch*stride+row*256+96],208,256,256,256);
                    else Tile<48,64,208,false,true,false>(d[batch*stride+row*256],ct[batch*stride+160*256],h[batch*stride+row*256+160],208,256,256,256);
                } else if(task%2==0) {
                    Tile<48,32,208,false,true,false>(d[batch*stride+(panel+row)*256],ct[batch*stride+192*256],h[batch*stride+(panel+row)*256+192],208,256,256,256);
                }
            } else {
                if(panel==0) {
                    uint32_t col=96+task%2*80;
                    Tile<48,80,128,false,true,false>(d[batch*stride+row*256],ct[batch*stride+col*256],h[batch*stride+row*256+col],256,256,256,256);
                } else {
                    uint32_t col=192+task%2*32;
                    Tile<48,32,256,false,true,false>(d[batch*stride+(panel+row)*256],ct[batch*stride+col*256],h[batch*stride+(panel+row)*256+col],256,256,256,256);
                }
            }
        }
    }
    template<uint32_t Active=256>
    __aicore__ inline void Update96(GlobalTensor<float> h,GlobalTensor<float> ct,uint32_t batches,uint32_t stride,uint32_t panel) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,col=task%2*128;
            if constexpr(Active==208) {
                if(panel==0) {
                    if(task%4<2)Tile<64,128,96,true,false,true>(h[batch*stride+96],ct[batch*stride+col],ct[batch*stride+96*256+col],96,256,256,256);
                    else Tile<64,128,96,true,false,true>(h[batch*stride+160],ct[batch*stride+col],ct[batch*stride+160*256+col],96,256,256,256);
                } else if(task%4<2) {
                    Tile<32,128,96,true,false,true>(h[batch*stride+panel*256+192],ct[batch*stride+panel*256+col],ct[batch*stride+192*256+col],96,256,256,256);
                }
            } else {
                if(panel==0) {
                    uint32_t row=96+task%4/2*80;
                    Tile<80,128,96,true,false,true>(h[batch*stride+row],ct[batch*stride+col],ct[batch*stride+row*256+col],96,256,256,256);
                } else {
                    uint32_t row=192+task%4/2*32;
                    Tile<32,128,96,true,false,true>(h[batch*stride+panel*256+row],ct[batch*stride+panel*256+col],ct[batch*stride+row*256+col],96,256,256,256);
                }
            }
        }
    }
    // Keep strides and the full 256-column factor; only omit zero products.
    __aicore__ inline void CrossGramFuture208(GlobalTensor<float> ct,GlobalTensor<float> cross,uint32_t batches,uint32_t stride,uint32_t panel) {
        uint32_t worker=GetBlockIdx();if(worker<batches)return;worker-=batches;
        uint32_t workers=GetBlockNum()-batches;
        if(panel==0) {
            for(uint32_t task=worker;task<batches*6;task+=workers) {
                uint32_t batch=task/6,future=96+task%6/3*64,row=task%3*32;
                Tile<64,32,208,false,true,false>(ct[batch*stride+future*256],ct[batch*stride+row*256],cross[batch*stride+future*128+row],208,256,256,128);
            }
        } else {
            for(uint32_t task=worker;task<batches*3;task+=workers) {
                uint32_t batch=task/3,row=task%3*32;
                Tile<32,32,208,false,true,false>(ct[batch*stride+192*256],ct[batch*stride+(panel+row)*256],cross[batch*stride+192*128+row],208,256,256,128);
            }
        }
    }
    __aicore__ inline void CrossAndSecondGram208(GlobalTensor<float> ct,GlobalTensor<float> cross,GlobalTensor<float> gram,uint32_t batches,uint32_t stride) {
        uint32_t worker=GetBlockIdx();if(worker<batches)return;worker-=batches;
        const uint32_t workers=GetBlockNum()-batches;
        auto la=l1buf.Get<float>(),lb0=la[16384],lb1=la[24576];
        auto a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        for(uint32_t task=worker;task<batches*6;task+=workers) {
            const uint32_t batch=task/6,future=96+task%6/3*64,row=task%3*32;
            PipeBarrier<PIPE_ALL>();
            uint8_t pad[4]={};SetFmatrix(1,208,pad,FmatrixMode::FMATRIX_LEFT);
            Nd2NzParams p{};p.ndNum=1;p.nValue=64;p.dValue=208;p.srcDValue=256;p.dstNzC0Stride=64;p.dstNzNStride=1;
            DataCopy(la,ct[batch*stride+future*256],p);
            p.nValue=32;p.dstNzC0Stride=32;
            DataCopy(lb0,ct[batch*stride+row*256],p);
            DataCopy(lb1,ct[batch*stride+(96+row)*256],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams q{};q.repeatTimes=26;q.srcStride=4;
            for(uint32_t r=0;r<4;++r)LoadData(a[r*3328],la[r*128],q);
            q.repeatTimes=52;q.srcStride=1;
            LoadData(b,lb0,q);LoadData(b[8192],lb1,q);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=64;m.n=32;m.k=208;m.cmatrixInitVal=true;
            Mmad(c,a,b,m);Mmad(c[2048],a,b[8192],m);
            SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
            FixpipeParamsV220 f{};f.mSize=64;f.nSize=32;f.srcStride=64;f.dstStride=128;
            Fixpipe<float,float,CFG_ROW_MAJOR>(cross[batch*stride+future*128+row],c,f);
            // Extra bottom32 cache rows are replaced by PadGram96 before use.
            Fixpipe<float,float,CFG_ROW_MAJOR>(gram[batch*stride+(future-96)*128+row],c[2048],f);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void SchurSecondGram96(GlobalTensor<float> cross,GlobalTensor<float> h,GlobalTensor<float> gram,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            const uint32_t batch=task/4,row=task%4/2*48,col=task%2*48;
            Tile<48,48,96,false,false,true>(cross[batch*stride+(96+row)*128],h[batch*stride+96+col],gram[batch*stride+row*128+col],96,128,256,128);
        }
    }
    template<uint32_t N>
    __aicore__ inline void HWithResidentInverse(GlobalTensor<float> cross,GlobalTensor<float> h) {
        auto lb=l1buf.Get<float>()[16384],a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        PipeBarrier<PIPE_ALL>();
        Nd2NzParams p{};p.ndNum=1;p.nValue=N;p.dValue=96;p.srcDValue=128;p.dstNzC0Stride=N;p.dstNzNStride=1;
        DataCopy(lb,cross,p);
        SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData2DParams q{};q.repeatTimes=96*N/128;q.srcStride=1;LoadData(b,lb,q);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=48;m.n=N;m.k=96;m.cmatrixInitVal=true;Mmad(c,a,b,m);
        SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=48;f.nSize=N;f.srcStride=48;f.dstStride=256;
        Fixpipe<float,float,CFG_ROW_MAJOR>(h,c,f);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void HFromCross208(GlobalTensor<float> gi,GlobalTensor<float> cross,GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        // key26 uses exactly four Cube tasks per batch. Factor96 just loaded
        // this core's GI row tile into L0A, including on the fallback path.
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*48;
            if(panel==0) {
                uint32_t col=96+task%2*64;
                HWithResidentInverse<64>(cross[batch*stride+col*128],h[batch*stride+row*256+col]);
            } else if(task%2==0) {
                HWithResidentInverse<32>(cross[batch*stride+192*128],h[batch*stride+(panel+row)*256+192]);
            }
        }
    }
    __aicore__ inline void TailCoefficients224(GlobalTensor<float> d,GlobalTensor<float> ct,GlobalTensor<float> e,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*3;task+=GetBlockNum()) {
            uint32_t batch=task/3,row=task%3*64;
            Tile<64,32,208,false,true,false>(d[batch*stride+row*256],ct[batch*stride+192*256],e[batch*stride+row*32],208,256,256,32);
        }
    }
    __aicore__ inline void ReprojectTail224(GlobalTensor<float> e,GlobalTensor<float> ct,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,col=task%4*64;
            Tile<32,64,192,true,false,true>(e[batch*stride],ct[batch*stride+col],ct[batch*stride+192*256+col],192,32,256,256);
        }
    }
    __aicore__ inline void ApplyFactor208(GlobalTensor<float> d,GlobalTensor<float> y,GlobalTensor<float> x,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4*64;
            Tile<64,16,208,true,false,false>(d[batch*stride+row],y[batch*stride],x[batch*stride+row*16],208,256,16,16);
        }
    }
    __aicore__ inline void CrossGramImplicit256(GlobalTensor<float> ct,GlobalTensor<float> cross,uint32_t batches,uint32_t stride,uint32_t panel) {
        uint32_t worker=GetBlockIdx();if(worker<batches)return;worker-=batches;
        uint32_t workers=GetBlockNum()-batches,tiles=(192-panel)/64;
        for(uint32_t task=worker;task<batches*tiles;task+=workers) {
            uint32_t batch=task/tiles,future=panel+64+task%tiles*64;
            Tile<64,64,256,false,true,false>(ct[batch*stride+future*256],ct[batch*stride+panel*256],cross[batch*stride+future*64],256,256,256,64);
        }
    }
    __aicore__ inline void HImplicit256(GlobalTensor<float> inverse,GlobalTensor<float> cross,GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        uint32_t tiles=(192-panel)/64;
        for(uint32_t task=GetBlockIdx();task<batches*tiles;task+=GetBlockNum()) {
            uint32_t batch=task/tiles,future=panel+64+task%tiles*64;
            Tile<64,64,64,false,true,false>(inverse[batch*stride],cross[batch*stride+future*64],h[batch*stride+panel*256+future],64,64,64,256);
        }
    }
    __aicore__ inline void ForwardApplyInverse64(GlobalTensor<float> H,GlobalTensor<float> inverses,GlobalTensor<float> input,GlobalTensor<float> output,uint32_t batches,uint32_t stride) {
        // One owner per RHS block. Forward reads immutable input; the same
        // owner then applies its local inverse after Tile's completion fence.
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,block=task%4;
            auto h=H[batch*stride+block*64],x=input[batch*stride],y=output[batch*stride+block*2048];
            if(block==1)Tile<64,32,64,true,false,true>(h,x,y,64,256,32,32);
            else if(block==2)Tile<64,32,128,true,false,true>(h,x,y,128,256,32,32);
            else if(block==3)Tile<64,32,192,true,false,true>(h,x,y,192,256,32,32);
            Tile<64,32,64,true,false,false>(inverses[batch*stride+block*4096],y,y,64,64,32,32);
        }
    }
    __aicore__ inline void Schur32Parallel(GlobalTensor<float> inverse,GlobalTensor<float> gram,GlobalTensor<float> q,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*2;task+=GetBlockNum()) {
            uint32_t batch=task/2;auto j=inverse[batch*stride],g=gram[batch*stride],tmp=q[batch*stride];
            if(task%2==0) {
                Tile<32,32,32,false,false,false>(j,g[32],g[32],32,64,64,64);
                Tile<32,32,32,false,false,true>(g[2048],g[32],g[2080],32,64,64,64);
            } else {
                // Preserve g[2048] until the other owner has consumed it.
                Tile<32,32,32,false,false,false>(g[2048],j,tmp,32,64,64,32);
            }
        }
    }
    __aicore__ inline void Assemble32Parallel(GlobalTensor<float> inverse,GlobalTensor<float> gram,GlobalTensor<float> q,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*2;task+=GetBlockNum()) {
            uint32_t batch=task/2;auto j=inverse[batch*stride],g=gram[batch*stride],tmp=q[batch*stride];
            if(task%2==0) {
                Tile<32,32,32,false,false,false>(j[2080],tmp,j[2048],32,64,32,64);
                Tile<32,32,32,false,false,true>(g[32],j[2048],j,32,64,64,64);
            } else {
                Tile<32,32,32,false,false,false>(g[32],j[2080],j[32],32,64,64,64);
            }
        }
    }
    __aicore__ inline void Forward96(GlobalTensor<float> H,GlobalTensor<float> Y,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            auto h=H[batch*stride],y=Y[batch*stride];
            Tile<160,32,96,true,false,true>(h[96],y,y[96*32],96,256,32,32);
            Tile<64,32,96,true,false,true>(h[96*256+192],y[96*32],y[192*32],96,256,32,32);
        }
    }
    __aicore__ inline void ComposeForward96(GlobalTensor<float> H,GlobalTensor<float> E,GlobalTensor<float> identity,GlobalTensor<float> F,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*2;task+=GetBlockNum()) {
            uint32_t batch=task/2;auto h=H[batch*stride],e=E[batch*stride],i=identity[batch*stride],f=F[batch*stride];
            if(task%2==0) {
                Tile<96,32,96,false,false,false>(h[96],h[96*256+192],f,96,256,256,32);
                Tile<96,32,96,false,false,true>(h[96],e[96*32],f,96,256,32,32);
                Tile<96,32,96,false,false,true>(i,h[192],f,96,96,256,32);
                Tile<96,32,96,false,false,true>(i,e,f,96,96,32,32);
            } else {
                Tile<96,32,96,false,false,false>(i,h[96*256+192],f[96*32],96,96,256,32);
                Tile<96,32,96,false,false,true>(i,e[96*32],f[96*32],96,96,32,32);
            }
        }
    }
    __aicore__ inline void Forward96Parallel(GlobalTensor<float> H,GlobalTensor<float> F,GlobalTensor<float> input,GlobalTensor<float> output,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,block=task%4;auto x=input[batch*stride],y=output[batch*stride];
            if(block<3)Tile<32,16,96,true,false,true>(H[batch*stride+96+block*32],x,y[(96+block*32)*16],96,256,16,16);
            else Tile<32,16,192,true,false,true>(F[batch*stride],x,y[192*16],192,32,16,16);
        }
    }
    __aicore__ inline void Forward96Tail224(GlobalTensor<float> H,GlobalTensor<float> E,GlobalTensor<float> Y,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            auto h=H[batch*stride],y=Y[batch*stride];
            Tile<128,32,96,true,false,true>(h[96],y,y[96*32],96,256,32,32);
            Tile<32,32,96,true,false,true>(h[96*256+192],y[96*32],y[192*32],96,256,32,32);
            Tile<32,32,192,true,false,true>(E[batch*stride],y,y[192*32],192,32,32,32);
        }
    }
    __aicore__ inline void Backsolve96(GlobalTensor<float> H,GlobalTensor<float> X,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            auto h=H[batch*stride],x=X[batch*stride];
            Tile<96,32,64,false,false,true>(h[96*256+192],x[192*32],x[96*32],64,256,32,32);
            Tile<96,32,160,false,false,true>(h[96],x[96*32],x,160,256,32,32);
        }
    }

};

#include <type_traits>
namespace Orthogonal {
constexpr uint32_t CT=0,D=65536,H=131072,AN=196608,G=262144,Q32=266240,
 BP=270336,XP=278528,R=286720,DX=294912,REDUCE=303104,NT=303360,NH=307456,STRIDE=kOrthogonalWorkspaceFloats;
static_assert(NH+12288==STRIDE);
class Vector {
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
    LocalTensor<float> w;
    uint32_t lane,teams;float inputScale=1.0f;
public:
    uint32_t batch;
    __aicore__ inline void Init(uint32_t batches) {
        uint32_t id=GetBlockIdx();batch=id%batches;lane=id/batches;teams=(2*GetBlockNum()+batches-1-batch)/batches;
        pipe.InitBuffer(buf,160*1024);w=buf.Get<float>();
        if(lane<8) {
            auto identity=w[36864];Duplicate(identity,0.0f,512);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<8;++j)identity.SetValue(j*64+lane*8+j,1.0f);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Residual(GlobalTensor<float> dst,GlobalTensor<float> lhs,GlobalTensor<float> rhs,GlobalTensor<float> mirror) {
        for(uint32_t tile=lane;tile<8;tile+=teams) {
            uint32_t off=tile*1024;DataCopy(w,lhs[off],1024);DataCopy(w[4096],rhs[off],1024);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],1024);PipeBarrier<PIPE_ALL>();DataCopy(dst[off],w,1024);DataCopy(mirror[off],w,1024);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PackCachedB(GlobalTensor<float> dst,GlobalTensor<float> initial,uint32_t k) {
        // Normalize retained the unscaled complete RHS at w[8448].
        // Offset r*32+c maps to r*k+c; only c<k is gathered.
        auto ix=w[16384].ReinterpretCast<int32_t>(),rows=w[16896].ReinterpretCast<int32_t>();
        CreateVecIndex(ix,(int32_t)0,512);PipeBarrier<PIPE_V>();
        ShiftRight(rows,ix,(int32_t)5,512);PipeBarrier<PIPE_V>();
        Muls(rows,rows,(int32_t)(32-k),512);PipeBarrier<PIPE_V>();
        Sub(ix,ix,rows,512);PipeBarrier<PIPE_V>();ShiftLeft(ix,ix,(int32_t)2,512);PipeBarrier<PIPE_V>();
        uint64_t columns=(1ULL<<k)-1,mask[2]={columns|(columns<<32),0};
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            Duplicate(w,0.0f,512);PipeBarrier<PIPE_V>();
            Gather(w,w[8448],ix.ReinterpretCast<uint32_t>(),row*k*4,mask,8,8);PipeBarrier<PIPE_V>();
            Muls(w,w,-inputScale,512);PipeBarrier<PIPE_ALL>();
            DataCopy(dst[row*32],w,512);DataCopy(initial[row*32],w,512);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Normalize(GlobalTensor<float> normalized,GlobalTensor<float> current,GlobalTensor<float> src,GlobalTensor<float> rhs,uint32_t k) {
        uint32_t count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams){DataCopy(w[count],src[tile*4096],4096);count+=4096;}
        PipeBarrier<PIPE_ALL>();
        // Every lane independently derives one scale from identical A/B data.
        Duplicate(w[8192],0.0f,2304);PipeBarrier<PIPE_ALL>();
        DataCopy(w[8192],src,256);DataCopy(w[8448],rhs,256*k);
        PipeBarrier<PIPE_ALL>();Abs(w[12288],w[8192],2304);PipeBarrier<PIPE_V>();
        WholeReduceMax(w[16384],w[12288],64,4,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        uint32_t aBits,mi;GetReduceRepeatMaxMinSpr<float>(aBits,mi);
        WholeReduceMax(w[16416],w[12544],64,32,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        uint32_t bBits;GetReduceRepeatMaxMinSpr<float>(bBits,mi);
        int32_t aExponent=(int32_t)((aBits>>23)&255)-127;
        int32_t bExponent=(int32_t)((bBits>>23)&255)-127;
        int32_t scaleExponent=-aExponent;
        const int32_t rhsLimit=126-bExponent;
        scaleExponent=scaleExponent<rhsLimit?scaleExponent:rhsLimit;
        scaleExponent=scaleExponent>126?126:(scaleExponent<-126?-126:scaleExponent);
        union {uint32_t bits;float value;} scale;
        scale.bits=(uint32_t)(127+scaleExponent)<<23;inputScale=scale.value;
        if(count){Muls(w,w,inputScale,count);PipeBarrier<PIPE_ALL>();}
        count=0;for(uint32_t tile=lane;tile<16;tile+=teams){DataCopy(normalized[tile*4096],w[count],4096);DataCopy(current[tile*4096],w[count],4096);count+=4096;}PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Finish(GlobalTensor<float> dst,GlobalTensor<float> x,GlobalTensor<float> dx,uint32_t k) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*32+c)*4);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            DataCopy(w,x[row*32],512);DataCopy(w[4096],dx[row*32],512);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],512);PipeBarrier<PIPE_ALL>();
            Gather(w[8192],w,ix,0,16*k);PipeBarrier<PIPE_ALL>();DataCopy(dst[row*k],w[8192],16*k);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Invert32(GlobalTensor<float> dst,GlobalTensor<float> src,bool negative=false) {
        if(lane)return;
        constexpr uint32_t B=32,R=72;
        auto panel=w,pivot=w[B*R],col=pivot[64],weights=col[B];auto ix=weights[256].ReinterpretCast<int32_t>();
        auto recip=w[4160],minusOne=w[4224];
        Duplicate(minusOne,-1.0f,8);
        Duplicate(panel,0.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(panel,src,DataCopyExtParams{B,128,128,5,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r){panel.SetValue(r*R+B+r,1.0f);ix.SetValue(r,r*R*4);}PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            Gather(col,panel[j],ix.ReinterpretCast<uint32_t>(),0,(uint64_t)32,1,8);PipeBarrier<PIPE_V>();
            Brcb(weights,col,4,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Div(recip,minusOne,weights[j*8],(uint64_t)8,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Mul(pivot,panel[j*R],recip,(uint64_t)64,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            MulAddDst(panel,pivot,weights,(uint64_t)64,(uint8_t)32,BinaryRepeatParams{1,1,0,9,0,1});PipeBarrier<PIPE_V>();
            Muls(panel[j*R],pivot,-1.0f,(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        if(negative){Muls(panel,panel,-1.0f,B*R);PipeBarrier<PIPE_ALL>();}
        DataCopyPad(dst,panel[B],DataCopyExtParams{B,128,5,128,0});PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void NegateBlock(GlobalTensor<float> src,uint32_t owner=1) {
        if(lane!=owner)return;
        DataCopyPad(w,src,DataCopyExtParams{32,128,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        Muls(w,w,-1.0f,1024);PipeBarrier<PIPE_ALL>();DataCopyPad(src,w,DataCopyExtParams{32,128,0,128,0});PipeBarrier<PIPE_ALL>();
    }


    __aicore__ inline void NewtonPrepare(GlobalTensor<half> packed,GlobalTensor<float> gram) {
        if(lane>=8)return;
        auto value=w,ab=w[4096],sum=w[8192],eye=w[12288];auto h=w[16384].ReinterpretCast<half>();
        DataCopy(value,gram,4096);PipeBarrier<PIPE_ALL>();Abs(ab,value,4096);PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,ab,64,64,1,1,8);PipeBarrier<PIPE_V>();WholeReduceMax(sum[128],sum,64,1,1,1,8);PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_ALL>();float alpha=1.8f/sum.GetValue(128);
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            DataCopy(value,gram[row*64],512);PipeBarrier<PIPE_ALL>();
            Cast(h,value,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
            Cast(ab,h,RoundMode::CAST_NONE,512);PipeBarrier<PIPE_V>();Sub(ab,value,ab,512);PipeBarrier<PIPE_V>();Cast(h[2048],ab,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
            Muls(eye,w[36864],1.0f,512);PipeBarrier<PIPE_V>();
            Cast(h[1024],eye,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();Muls(ab,value,-alpha,512);PipeBarrier<PIPE_V>();Add(ab,eye,ab,512);PipeBarrier<PIPE_V>();Cast(h[1536],ab,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();Muls(value,eye,-alpha,512);PipeBarrier<PIPE_V>();Cast(h[512],value,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
            Muls(value,eye,2.0f,512);PipeBarrier<PIPE_V>();Cast(h[2560],value,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_ALL>();DataCopy(packed[row*64],h,DataCopyParams{4,32,0,224});DataCopy(packed[16384+row*64],h[2048],512);DataCopy(packed[20480+row*64],h[2560],512);PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ZeroDecision(GlobalTensor<int32_t> decision) {
        if(batch || lane)return;auto flag=w.ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NewtonGuard(GlobalTensor<float> dst,GlobalTensor<int32_t> decision) {
        auto ab=w[4096],sum=w[8192];
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            DataCopy(w,dst[row*64],512);PipeBarrier<PIPE_ALL>();
            Abs(ab,w,512);PipeBarrier<PIPE_V>();WholeReduceSum(sum,ab,64,8,1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceMax(sum[128],sum,8,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(sum[192],sum,8,1,1,1,8);PipeBarrier<PIPE_ALL>();
            float bound=sum.GetValue(128),total=sum.GetValue(192);
            if(!(bound<=0.015625f && total>=0.0f && total<=2.0f)) {
                auto flag=sum[128].ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);PipeBarrier<PIPE_ALL>();
                SetAtomicAdd<int32_t>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();SetAtomicNone();
            }
        }
    }
    __aicore__ inline void NewtonDefect(GlobalTensor<float> dst) {
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            DataCopy(w,dst[row*64],512);PipeBarrier<PIPE_ALL>();Muls(w,w,-1.0f,512);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<8;++j){uint32_t off=j*64+row+j;w.SetValue(off,2.0f+w.GetValue(off));}
            PipeBarrier<PIPE_ALL>();DataCopy(dst[row*64],w,512);PipeBarrier<PIPE_ALL>();
        }
    }

};
template<bool Cube> __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,uint32_t batches,uint32_t k,bool broadcast) {
    GlobalTensor<float> mem,A,B,X;
    mem.SetGlobalBuffer((__gm__ float*)workspace);A.SetGlobalBuffer((__gm__ float*)aa);B.SetGlobalBuffer((__gm__ float*)bb);X.SetGlobalBuffer((__gm__ float*)xx);
    std::conditional_t<Cube,NativeMM,Vector> op;
    uint32_t batch=0;
    if constexpr(Cube)op.Init();else {op.Init(batches);batch=op.batch;mem=mem[(uint64_t)batch*STRIDE];A=A[(uint64_t)batch*65536];B=B[(uint64_t)(broadcast?0:batch)*256*k];X=X[(uint64_t)batch*256*k];}
    GlobalTensor<half> halfMem;halfMem.SetGlobalBuffer((__gm__ half*)workspace+((uint64_t)batch*STRIDE+NH)*2);
    if constexpr(!Cube){op.Normalize(mem[AN],mem[CT],A,B,k);op.PackCachedB(mem[BP],mem[DX],k);}
    SyncAll<false>();
    for(uint32_t p=0;p<256;p+=64) {
        const uint32_t giBase=D+p*64;
        if constexpr(Cube)op.template Run<64,64,256,false,true>(mem[CT+p*256],mem[CT+p*256],mem[G],batches,STRIDE,STRIDE,STRIDE);
        SyncAll<false>();
        bool useExact=p==192;
        if(p<192) {
            GlobalTensor<int32_t> decisions;decisions.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+128+(p/64)*32);
            if constexpr(!Cube)op.ZeroDecision(decisions);
            if constexpr(!Cube)op.NewtonPrepare(halfMem,mem[G]);
            SyncAll<false>();
            if constexpr(Cube){op.Newton64(halfMem,mem[giBase],mem[NT],batches,STRIDE,p==0?6:(p==64?7:9));op.CrossGramImplicit256(mem[CT],mem[D+16384],batches,STRIDE,p);}
            SyncAll<false>();
            if constexpr(!Cube)op.NewtonGuard(mem[NT],decisions);
            if constexpr(Cube)op.HImplicit256(mem[giBase],mem[D+16384],mem[H],batches,STRIDE,p);
            SyncAll<false>();
            DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decisions);
            useExact=decisions.GetValue(0)!=0;
        }
        if(useExact) {
        if constexpr(!Cube){op.Invert32(mem[giBase],mem[G]);op.NegateBlock(mem[G+2048]);}
        if constexpr(Cube){if(p==192)op.template Run<64,128,64,false,false,true>(mem[H+64],mem[H+64*256+128],mem[H+128],batches,STRIDE,STRIDE,STRIDE,256,256,256);}
        SyncAll<false>();
        if constexpr(Cube)op.Schur32Parallel(mem[giBase],mem[G],mem[Q32],batches,STRIDE);
        SyncAll<false>();
        if constexpr(!Cube){op.Invert32(mem[giBase+2080],mem[G+2080],true);op.NegateBlock(mem[G+32]);op.NegateBlock(mem[giBase],2);}
        if constexpr(Cube){if(p==192)op.template Run<128,64,64,false,false,true>(mem[H+128],mem[H+128*256+192],mem[H+192],batches,STRIDE,STRIDE,STRIDE,256,256,256);}
        SyncAll<false>();
        if constexpr(Cube)op.Assemble32Parallel(mem[giBase],mem[G],mem[Q32],batches,STRIDE);
        if constexpr(Cube)SyncCube();
        if constexpr(Cube){if(p<192){op.HImplicit256(mem[giBase],mem[D+16384],mem[H],batches,STRIDE,p);SyncCube();}}
        }
        if(p+64<256) {
            if constexpr(Cube) {
                if(p==0)op.template Run<192,256,64,true,false,true>(mem[H+64],mem[CT],mem[CT+64*256],batches,STRIDE,STRIDE,STRIDE,256,256,256);
                else if(p==64)op.template Run<128,256,64,true,false,true>(mem[H+p*256+128],mem[CT+p*256],mem[CT+128*256],batches,STRIDE,STRIDE,STRIDE,256,256,256);
                else op.template Run<64,256,64,true,false,true>(mem[H+p*256+192],mem[CT+p*256],mem[CT+192*256],batches,STRIDE,STRIDE,STRIDE,256,256,256);
            }
            if constexpr(Cube)SyncCube();
        }
    }
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[BP],mem[DX],batches,STRIDE);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true>(mem[CT],mem[DX],mem[XP],batches,STRIDE,STRIDE,STRIDE);
    if constexpr(Cube)SyncCube();
    if constexpr(Cube)op.template Run<256,32,256>(mem[AN],mem[XP],mem[R],batches,STRIDE,STRIDE,STRIDE);
    SyncAll<false>();
    if constexpr(!Cube)op.Residual(mem[R],mem[BP],mem[R],mem[DX]);
    SyncAll<false>();
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[R],mem[DX],batches,STRIDE);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true>(mem[CT],mem[DX],mem[R],batches,STRIDE,STRIDE,STRIDE);
    SyncAll<false>();
    if constexpr(!Cube)op.Finish(X,mem[XP],mem[R],k);
}
}

namespace NarrowOrthogonal {
class Vector {
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
    LocalTensor<float> w;
    uint32_t lane,teams;float inputScale=1.0f;
public:
    uint32_t batch;
    __aicore__ inline void Init(uint32_t batches) {
        uint32_t id=GetBlockIdx();batch=id%batches;lane=id/batches;teams=(2*GetBlockNum()+batches-1-batch)/batches;
        pipe.InitBuffer(buf,160*1024);w=buf.Get<float>();
        if(lane<6) {
            auto identity=w[36864];Duplicate(identity,0.0f,1536);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<16;++j)identity.SetValue(j*96+lane*16+j,1.0f);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PackCachedB(GlobalTensor<float> dst,GlobalTensor<float> initial,uint32_t k) {
        // NormalizeSeed retained unscaled RHS, zero-padded to 256 rows, at w[8448].
        // Offset r*16+c maps to r*k+c; only c<k is gathered.
        auto ix=w[16384].ReinterpretCast<int32_t>(),rows=w[16896].ReinterpretCast<int32_t>();
        CreateVecIndex(ix,(int32_t)0,256);PipeBarrier<PIPE_V>();
        ShiftRight(rows,ix,(int32_t)4,256);PipeBarrier<PIPE_V>();
        Muls(rows,rows,(int32_t)(16-k),256);PipeBarrier<PIPE_V>();
        Sub(ix,ix,rows,256);PipeBarrier<PIPE_V>();ShiftLeft(ix,ix,(int32_t)2,256);PipeBarrier<PIPE_V>();
        uint64_t columns=(1ULL<<k)-1,mask[2]={columns|(columns<<16)|(columns<<32)|(columns<<48),0};
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            Duplicate(w,0.0f,256);PipeBarrier<PIPE_V>();
            Gather(w,w[8448],ix.ReinterpretCast<uint32_t>(),row*k*4,mask,4,8);PipeBarrier<PIPE_V>();
            Muls(w,w,-inputScale,256);PipeBarrier<PIPE_ALL>();
            DataCopy(dst[row*16],w,256);DataCopy(initial[row*16],w,256);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PackB(GlobalTensor<float> dst,GlobalTensor<float> initial,GlobalTensor<float> src,uint32_t k,uint32_t n) {
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            Duplicate(w,0.0f,512);PipeBarrier<PIPE_ALL>();
            if(row<n)DataCopyPad(w,src[row*k],DataCopyExtParams{(uint16_t)(n-row<16?n-row:16),k*4,0,3,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-k),0});
            PipeBarrier<PIPE_ALL>();Muls(w,w,-inputScale,512);PipeBarrier<PIPE_ALL>();
            DataCopy(dst[row*32],w,512);DataCopy(initial[row*32],w,512);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Normalize(GlobalTensor<float> normalized,GlobalTensor<float> src,GlobalTensor<float> tmp,uint32_t n) {
        uint32_t count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams){
            uint32_t row=tile*16,rows=row<n?(n-row<16?n-row:16):0;
            Duplicate(w[count],0.0f,4096);PipeBarrier<PIPE_ALL>();
            if(rows)DataCopyPad(w[count],src[row*n],DataCopyExtParams{(uint16_t)rows,n*4,0,(256-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});count+=4096;
        }
        PipeBarrier<PIPE_ALL>();uint32_t mb=0,mi=0;
        if(count) {
            Abs(w[8192],w,count);PipeBarrier<PIPE_V>();WholeReduceMax(w[16384],w[8192],64,(uint8_t)(count/64),1,1,8);
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);GetReduceRepeatMaxMinSpr<float>(mb,mi);
        }
        w[20000].ReinterpretCast<uint32_t>().SetValue(0,mb);PipeBarrier<PIPE_ALL>();DataCopy(tmp[lane*8],w[20000],8);PipeBarrier<PIPE_ALL>();
        SyncAll<true>();DataCopy(w[22000],tmp,teams*8);PipeBarrier<PIPE_ALL>();
        uint32_t maxBits=0;for(uint32_t i=0;i<teams;++i){uint32_t v=w[22000].ReinterpretCast<uint32_t>().GetValue(i*8);maxBits=v>maxBits?v:maxBits;}
        int32_t exponent=(int32_t)((maxBits>>23)&255)-127;exponent=exponent>126?126:(exponent<-126?-126:exponent);
        union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;inputScale=scale.value;
        if(count){Muls(w,w,inputScale,count);PipeBarrier<PIPE_ALL>();}
        count=0;for(uint32_t tile=lane;tile<16;tile+=teams){
            for(uint32_t row=tile*16;row<tile*16+16;++row)if(row>=n)w.SetValue(count+(row%16)*256+row,1.0f);
            PipeBarrier<PIPE_ALL>();DataCopy(normalized[tile*4096],w[count],4096);count+=4096;}PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void NormalizeSeed(GlobalTensor<float> normalized,GlobalTensor<float> src,GlobalTensor<float> rhs,uint32_t k,uint32_t n) {
        uint32_t count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams){
            uint32_t row=tile*16,rows=row<n?(n-row<16?n-row:16):0;
            Duplicate(w[count],0.0f,4096);PipeBarrier<PIPE_ALL>();
            if(rows)DataCopyPad(w[count],src[row*n],DataCopyExtParams{(uint16_t)rows,n*4,0,(256-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});count+=4096;
        }
        PipeBarrier<PIPE_ALL>();
        // The common seed row and complete RHS are identical on all eight lanes.
        Duplicate(w[8192],0.0f,1280);PipeBarrier<PIPE_ALL>();
        DataCopyPad(w[8192],src,DataCopyExtParams{1,n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        DataCopyPad(w[8448],rhs,DataCopyExtParams{1,n*k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();Abs(w[10000],w[8192],1280);PipeBarrier<PIPE_V>();
        WholeReduceMax(w[16384],w[10000],64,4,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        uint32_t aBits,mi;GetReduceRepeatMaxMinSpr<float>(aBits,mi);
        WholeReduceMax(w[16416],w[10256],64,16,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        uint32_t bBits;GetReduceRepeatMaxMinSpr<float>(bBits,mi);
        int32_t aExponent=(int32_t)((aBits>>23)&255)-127;
        int32_t bExponent=(int32_t)((bBits>>23)&255)-127;
        int32_t scaleExponent=-aExponent;
        const int32_t rhsLimit=126-bExponent;
        scaleExponent=scaleExponent<rhsLimit?scaleExponent:rhsLimit;
        scaleExponent=scaleExponent>126?126:(scaleExponent<-126?-126:scaleExponent);
        union {uint32_t bits;float value;} scale;
        scale.bits=(uint32_t)(127+scaleExponent)<<23;inputScale=scale.value;
        if(count){Muls(w,w,inputScale,count);PipeBarrier<PIPE_ALL>();}
        count=0;for(uint32_t tile=lane;tile<16;tile+=teams){
            for(uint32_t row=tile*16;row<tile*16+16;++row)if(row>=n)w.SetValue(count+(row%16)*256+row,1.0f);
            PipeBarrier<PIPE_ALL>();DataCopy(normalized[tile*4096],w[count],4096);count+=4096;}PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Finish(GlobalTensor<float> dst,GlobalTensor<float> x,GlobalTensor<float> dx,uint32_t k,uint32_t n) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*32+c)*4);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<n;row+=teams*16) {
            DataCopy(w,x[row*32],512);DataCopy(w[4096],dx[row*32],512);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],512);PipeBarrier<PIPE_ALL>();
            uint32_t rows=n-row<16?n-row:16;
            Gather(w[8192],w,ix,0,rows*k);PipeBarrier<PIPE_ALL>();
            DataCopyPad(dst[row*k],w[8192],DataCopyExtParams{1,rows*k*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Finish16(GlobalTensor<float> dst,GlobalTensor<float> x,GlobalTensor<float> dx,uint32_t k,uint32_t n) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*16+c)*4);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<n;row+=teams*16) {
            DataCopy(w,x[row*16],256);DataCopy(w[4096],dx[row*16],256);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],256);PipeBarrier<PIPE_ALL>();
            uint32_t rows=n-row<16?n-row:16;
            Gather(w[8192],w,ix,0,rows*k);PipeBarrier<PIPE_ALL>();
            DataCopyPad(dst[row*k],w[8192],DataCopyExtParams{1,rows*k*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    template<bool Rhs> __aicore__ inline float Split6(LocalTensor<float> v,LocalTensor<int8_t> digits) {
        auto q=w[256],ab=w[512],mx=w[768];auto hd=w[1024].ReinterpretCast<half>();
        Abs(ab,v,256);PipeBarrier<PIPE_V>();WholeReduceMax(mx,ab,64,4,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);
        int32_t exponent=(int32_t)((mb>>23)&255)-127-5;
        if((mb&0x7fffffff)==0){exponent=0;Duplicate(v,0.0f,256);PipeBarrier<PIPE_V>();}
        else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;Muls(v,v,scale.value,256);PipeBarrier<PIPE_V>();}
        for(uint32_t d=0;d<6;++d) {
            Adds(q,v,12582912.0f,256);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,256);PipeBarrier<PIPE_V>();
            Cast(hd,q,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();Cast(digits[(Rhs?5-d:d)*256],hd,RoundMode::CAST_TRUNC,256);PipeBarrier<PIPE_V>();
            Sub(v,v,q,256);PipeBarrier<PIPE_V>();Muls(v,v,128.0f,256);PipeBarrier<PIPE_V>();
        }
        return (float)exponent;
    }
    __aicore__ inline float Split6Batched(LocalTensor<float> v,LocalTensor<int8_t> digits) {
        auto allDigits=w[2048],ab=w[512],mx=w[768];auto hd=w[4096].ReinterpretCast<half>();
        Abs(ab,v,256);PipeBarrier<PIPE_V>();WholeReduceMax(mx,ab,64,4,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);
        int32_t exponent=(int32_t)((mb>>23)&255)-127-5;
        if((mb&0x7fffffff)==0){exponent=0;Duplicate(v,0.0f,256);PipeBarrier<PIPE_V>();}
        else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;Muls(v,v,scale.value,256);PipeBarrier<PIPE_V>();}
        for(uint32_t d=0;d<6;++d) {
            auto q=allDigits[(5-d)*256];
            Adds(q,v,12582912.0f,256);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,256);PipeBarrier<PIPE_V>();
            Sub(v,v,q,256);PipeBarrier<PIPE_V>();Muls(v,v,128.0f,256);PipeBarrier<PIPE_V>();
        }
        Cast(hd,allDigits,RoundMode::CAST_NONE,1536);PipeBarrier<PIPE_V>();
        Cast(digits,hd,RoundMode::CAST_TRUNC,1536);PipeBarrier<PIPE_V>();
        return (float)exponent;
    }
    __aicore__ __attribute__((noinline)) void SplitAChunk(GlobalTensor<int8_t> digits,GlobalTensor<float> exponents,GlobalTensor<float> matrix,uint32_t panel,uint32_t n) {
        if(lane==0)return;
        constexpr uint32_t L=2048;auto v=w,q=w[2048],ab=w[4096],mx=w[6144],e=w[12000];auto hd=w[8192].ReinterpretCast<half>();auto bd=w[10000].ReinterpretCast<int8_t>();
        uint32_t start=((panel/32)*(teams-1)+lane-1)*8;
        if(start<256) {
            uint32_t rows=start<n?(n-start<8?n-start:8):0;
            Duplicate(v,0.0f,L);PipeBarrier<PIPE_ALL>();
            if(rows)DataCopyPad(v,matrix[start*n],DataCopyExtParams{(uint16_t)rows,n*4,0,(256-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
            PipeBarrier<PIPE_ALL>();Muls(v,v,inputScale,L);PipeBarrier<PIPE_ALL>();
            for(uint32_t r=0;r<8;++r)if(start+r>=n)v.SetValue(r*256+start+r,1.0f);
            PipeBarrier<PIPE_ALL>();Abs(ab,v,L);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<8;++r) {
                WholeReduceMax(mx,ab[r*256],64,4,1,1,8);SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);int32_t exponent=(int32_t)((mb>>23)&255)-132;
                if((mb&0x7fffffff)==0){exponent=0;Duplicate(v[r*256],0.0f,256);}
                else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;Muls(v[r*256],v[r*256],scale.value,256);}
                PipeBarrier<PIPE_ALL>();e.SetValue(r,(float)exponent);
            }
            PipeBarrier<PIPE_ALL>();DataCopy(exponents[start],e,8);PipeBarrier<PIPE_ALL>();
            for(uint32_t d=0;d<6;++d) {
                Adds(q,v,12582912.0f,L);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,L);PipeBarrier<PIPE_V>();
                Cast(hd,q,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();Cast(bd,hd,RoundMode::CAST_TRUNC,L);PipeBarrier<PIPE_V>();
                Sub(v,v,q,L);PipeBarrier<PIPE_V>();Muls(v,v,128.0f,L);PipeBarrier<PIPE_ALL>();
                DataCopyPad(digits[start*1536+d*256],bd,DataCopyExtParams{8,256,0,1280,0});PipeBarrier<PIPE_ALL>();
            }
        }
    }
    __aicore__ __attribute__((noinline)) void SplitXPacked4(GlobalTensor<int8_t> digits,GlobalTensor<float> exponents,GlobalTensor<float> solution,uint32_t k) {
        auto input=w[10000],e=w[8000];auto d=w[1536].ReinterpretCast<int8_t>();auto ix=w[19000].ReinterpretCast<int32_t>();auto packed=w[20000].ReinterpretCast<int8_t>();
        CreateVecIndex(ix,(int32_t)0,256);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)64,256);PipeBarrier<PIPE_ALL>();
        if(lane<k){DataCopy(input,solution,4096);PipeBarrier<PIPE_ALL>();}
        for(uint32_t c=lane;c<4;c+=teams) {
            if(c<k)Gather(w,input[c],ix.ReinterpretCast<uint32_t>(),0,256);else Duplicate(w,0.0f,256);PipeBarrier<PIPE_ALL>();
            float exponent=0;if(c<k)exponent=Split6Batched(w,d);else Duplicate(d.ReinterpretCast<int32_t>(),(int32_t)0,384);PipeBarrier<PIPE_ALL>();
            Duplicate(packed.ReinterpretCast<int32_t>(),(int32_t)0,3072);PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<6;++weight)DataCopy(packed[weight*1536],d[(5-weight)*256],(weight+1)*256);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(digits[c*1536],packed,DataCopyExtParams{8,1536,0,4608,0});PipeBarrier<PIPE_ALL>();
            e.SetValue(0,exponent);e.SetValue(8,0.0f);PipeBarrier<PIPE_ALL>();
            DataCopyPad(exponents[c],e,DataCopyExtParams{1,4,0,0,0});DataCopyPad(exponents[c+4],e[8],DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void SplitX(GlobalTensor<int8_t> digits,GlobalTensor<float> exponents,GlobalTensor<float> solution,uint32_t k) {
        auto input=w[10000],e=w[8000];auto d=w[1536].ReinterpretCast<int8_t>();auto ix=w[19000].ReinterpretCast<int32_t>();
        CreateVecIndex(ix,(int32_t)0,256);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)128,256);PipeBarrier<PIPE_ALL>();
        if(lane<k){DataCopy(input,solution,8192);PipeBarrier<PIPE_ALL>();}
        for(uint32_t c=lane;c<16;c+=teams) {
            if(c<k)Gather(w,input[c],ix.ReinterpretCast<uint32_t>(),0,256);else Duplicate(w,0.0f,256);PipeBarrier<PIPE_ALL>();
            float exponent=0;
            if(c<k)exponent=Split6<true>(w,d);else Duplicate(d.ReinterpretCast<int32_t>(),(int32_t)0,384);PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<6;++weight)DataCopy(digits[(weight*(weight+1)/2*16+c*(weight+1))*256],d[(5-weight)*256],(weight+1)*256);
            e.SetValue(0,exponent);PipeBarrier<PIPE_ALL>();DataCopyPad(exponents[c],e,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void Assemble6Packed4(GlobalTensor<float> dst,GlobalTensor<float> initial,GlobalTensor<int32_t> products,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> rhs) {
        constexpr uint32_t L=128;
        auto ae=w,xe=w[256],gs=w[512],high=w[1024],low=w[1536],term=w[2048],tmp=w[2560],u=w[3072],v=w[3584],factor=w[4096],ea=w[5120],ex=w[5632];auto packed=w[7000],allProducts=w[9000],rhsTile=w[10400];auto ix=w[4608].ReinterpretCast<int32_t>();
        DataCopy(ae,expA,256);DataCopy(xe,expX,16);PipeBarrier<PIPE_ALL>();
        CreateVecIndex(ix,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(ix,ix,(int32_t)2,L);PipeBarrier<PIPE_V>();ShiftLeft(ix,ix,(int32_t)2,L);PipeBarrier<PIPE_ALL>();
        auto index=w[11000].ReinterpretCast<int32_t>();auto columns=w[11504].ReinterpretCast<int32_t>();
        CreateVecIndex(index,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(columns,index,(int32_t)2,L);PipeBarrier<PIPE_V>();ShiftLeft(columns,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();Sub(columns,index,columns,L);PipeBarrier<PIPE_V>();
        ShiftLeft(index,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();Gather(ex,xe,index.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
        auto outputIndex=w[16000].ReinterpretCast<int32_t>(),outputRows=w[16512].ReinterpretCast<int32_t>(),rhsIndex=w[17024].ReinterpretCast<int32_t>();
        CreateVecIndex(rhsIndex,(int32_t)0,L);PipeBarrier<PIPE_V>();Add(rhsIndex,rhsIndex,ix,L);PipeBarrier<PIPE_V>();ShiftLeft(rhsIndex,rhsIndex,(int32_t)2,L);PipeBarrier<PIPE_V>();
        CreateVecIndex(outputIndex,(int32_t)0,512);PipeBarrier<PIPE_V>();ShiftRight(outputRows,outputIndex,(int32_t)4,512);PipeBarrier<PIPE_V>();
        Muls(outputRows,outputRows,(int32_t)12,512);PipeBarrier<PIPE_V>();Sub(outputIndex,outputIndex,outputRows,512);PipeBarrier<PIPE_V>();ShiftLeft(outputIndex,outputIndex,(int32_t)2,512);PipeBarrier<PIPE_ALL>();
        uint64_t outputMask[2]={0x000f000f000f000fULL,0};
        for(uint32_t tile=lane;tile<8;tile+=teams) {
            uint32_t row=tile*32;Gather(ea,ae[row],ix.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Duplicate(low,0.0f,L);DataCopyPad(rhsTile,rhs[row*16],DataCopyExtParams{32,32,32,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();Gather(high,rhsTile,rhsIndex.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            DataCopy(allProducts.ReinterpretCast<int32_t>(),products[row*32],1024);PipeBarrier<PIPE_ALL>();
            CreateVecIndex(columns,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(index,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();ShiftLeft(gs.ReinterpretCast<int32_t>(),index,(int32_t)2,L);PipeBarrier<PIPE_V>();Sub(columns,columns,gs.ReinterpretCast<int32_t>(),L);PipeBarrier<PIPE_V>();ShiftLeft(index,index,(int32_t)7,L);PipeBarrier<PIPE_V>();ShiftLeft(columns,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();Add(index,index,columns,L);PipeBarrier<PIPE_ALL>();
            auto q0=w[14000].ReinterpretCast<int32_t>(),q1=w[14256].ReinterpretCast<int32_t>(),q2=w[14512].ReinterpretCast<int32_t>(),q3=w[14768].ReinterpretCast<int32_t>(),q4=w[15024].ReinterpretCast<int32_t>(),q5=w[15280].ReinterpretCast<int32_t>();
            Adds(columns,index,(int32_t)(0*16),L);PipeBarrier<PIPE_V>();Gather(q0,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(1*16),L);PipeBarrier<PIPE_V>();Gather(q1,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(2*16),L);PipeBarrier<PIPE_V>();Gather(q2,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(3*16),L);PipeBarrier<PIPE_V>();Gather(q3,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(4*16),L);PipeBarrier<PIPE_V>();Gather(q4,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(5*16),L);PipeBarrier<PIPE_V>();Gather(q5,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            ShiftLeft(q0,q0,(int32_t)7,L);PipeBarrier<PIPE_V>();Add(q0,q0,q1,L);PipeBarrier<PIPE_V>();
            ShiftLeft(q2,q2,(int32_t)7,L);PipeBarrier<PIPE_V>();Add(q2,q2,q3,L);PipeBarrier<PIPE_V>();
            ShiftLeft(q4,q4,(int32_t)7,L);PipeBarrier<PIPE_V>();Add(q4,q4,q5,L);PipeBarrier<PIPE_V>();
            // Canonical base-16384 digits. All intermediate magnitudes are below 2^30.
            ShiftRight(q5,q4,(int32_t)14,L);PipeBarrier<PIPE_V>();Add(q2,q2,q5,L);PipeBarrier<PIPE_V>();ShiftLeft(q5,q5,(int32_t)14,L);PipeBarrier<PIPE_V>();Sub(q4,q4,q5,L);PipeBarrier<PIPE_V>();
            ShiftRight(q3,q2,(int32_t)14,L);PipeBarrier<PIPE_V>();Add(q0,q0,q3,L);PipeBarrier<PIPE_V>();ShiftLeft(q3,q3,(int32_t)14,L);PipeBarrier<PIPE_V>();Sub(q2,q2,q3,L);PipeBarrier<PIPE_V>();
            // Split the leading int32 at its FP32 rounding boundary; pack its exact remainder with the middle digit.
            Cast(gs,q0,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();Cast(q1,gs,RoundMode::CAST_RINT,L);PipeBarrier<PIPE_V>();Sub(q1,q0,q1,L);PipeBarrier<PIPE_V>();ShiftLeft(q1,q1,(int32_t)14,L);PipeBarrier<PIPE_V>();Add(q1,q1,q2,L);PipeBarrier<PIPE_V>();
            Cast(q2.ReinterpretCast<float>(),q1,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();Cast(q4.ReinterpretCast<float>(),q4,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();
            Add(factor,ea,ex,L);PipeBarrier<PIPE_V>();Adds(factor,factor,120.0f,L);PipeBarrier<PIPE_V>();Cast(factor.ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,L);PipeBarrier<PIPE_V>();ShiftLeft(factor.ReinterpretCast<int32_t>(),factor.ReinterpretCast<int32_t>(),(int32_t)23,L);PipeBarrier<PIPE_V>();
            Mul(term,gs,factor,L);PipeBarrier<PIPE_V>();
            Add(tmp,high,term,L);PipeBarrier<PIPE_V>();Sub(u,tmp,high,L);PipeBarrier<PIPE_V>();Sub(v,tmp,u,L);PipeBarrier<PIPE_V>();Sub(v,high,v,L);PipeBarrier<PIPE_V>();Sub(u,term,u,L);PipeBarrier<PIPE_V>();Add(u,u,v,L);PipeBarrier<PIPE_V>();Add(low,low,u,L);PipeBarrier<PIPE_V>();Muls(high,tmp,1.0f,L);PipeBarrier<PIPE_V>();
            Muls(factor,factor,0.00006103515625f,L);PipeBarrier<PIPE_V>();
            Mul(term,q2.ReinterpretCast<float>(),factor,L);PipeBarrier<PIPE_V>();
            Add(tmp,high,term,L);PipeBarrier<PIPE_V>();Sub(u,tmp,high,L);PipeBarrier<PIPE_V>();Sub(v,tmp,u,L);PipeBarrier<PIPE_V>();Sub(v,high,v,L);PipeBarrier<PIPE_V>();Sub(u,term,u,L);PipeBarrier<PIPE_V>();Add(u,u,v,L);PipeBarrier<PIPE_V>();Add(low,low,u,L);PipeBarrier<PIPE_V>();Muls(high,tmp,1.0f,L);PipeBarrier<PIPE_V>();
            Muls(factor,factor,0.00006103515625f,L);PipeBarrier<PIPE_V>();
            Mul(term,q4.ReinterpretCast<float>(),factor,L);PipeBarrier<PIPE_V>();
            Add(tmp,high,term,L);PipeBarrier<PIPE_V>();Sub(u,tmp,high,L);PipeBarrier<PIPE_V>();Sub(v,tmp,u,L);PipeBarrier<PIPE_V>();Sub(v,high,v,L);PipeBarrier<PIPE_V>();Sub(u,term,u,L);PipeBarrier<PIPE_V>();Add(u,u,v,L);PipeBarrier<PIPE_V>();Add(low,low,u,L);PipeBarrier<PIPE_V>();Muls(high,tmp,1.0f,L);PipeBarrier<PIPE_V>();
            Add(high,high,low,L);PipeBarrier<PIPE_V>();Duplicate(packed,0.0f,512);PipeBarrier<PIPE_V>();Gather(packed,high,outputIndex.ReinterpretCast<uint32_t>(),0,outputMask,8,8);PipeBarrier<PIPE_ALL>();DataCopy(dst[row*16],packed,512);DataCopy(initial[row*16],packed,512);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void Assemble6(GlobalTensor<float> dst,GlobalTensor<int32_t> products,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> rhs) {
        constexpr uint32_t L=256;
        auto ae=w,xe=w[256],gs=w[512],high=w[1024],low=w[1536],term=w[2048],tmp=w[2560],u=w[3072],v=w[3584],factor=w[4096],ea=w[5120],ex=w[5632];auto packed=w[7000];auto ix=w[4608].ReinterpretCast<int32_t>();
        DataCopy(ae,expA,256);DataCopy(xe,expX,16);PipeBarrier<PIPE_ALL>();
        CreateVecIndex(ix,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(ix,ix,(int32_t)4,L);PipeBarrier<PIPE_V>();ShiftLeft(ix,ix,(int32_t)2,L);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<16;++r)DataCopy(ex[r*16],xe,16);PipeBarrier<PIPE_ALL>();
        for(uint32_t tile=lane;tile<16;tile+=teams) {
            uint32_t row=tile*16;Gather(ea,ae[row],ix.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Duplicate(low,0.0f,L);DataCopyPad(high,rhs[row*32],DataCopyExtParams{16,64,64,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<6;++weight) {
                DataCopy(gs.ReinterpretCast<int32_t>(),products[weight*4096+row*16],L);PipeBarrier<PIPE_ALL>();
                Cast(gs,gs.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();
                Add(factor,ea,ex,L);PipeBarrier<PIPE_V>();Adds(factor,factor,127.0f-7.0f*weight,L);PipeBarrier<PIPE_V>();
                Cast(factor.ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,L);PipeBarrier<PIPE_V>();ShiftLeft(factor.ReinterpretCast<int32_t>(),factor.ReinterpretCast<int32_t>(),(int32_t)23,L);PipeBarrier<PIPE_V>();
                Mul(term,gs,factor,L);PipeBarrier<PIPE_V>();
                Add(tmp,high,term,L);PipeBarrier<PIPE_V>();Sub(u,tmp,high,L);PipeBarrier<PIPE_V>();Sub(v,tmp,u,L);PipeBarrier<PIPE_V>();Sub(v,high,v,L);PipeBarrier<PIPE_V>();Sub(u,term,u,L);PipeBarrier<PIPE_V>();Add(u,u,v,L);PipeBarrier<PIPE_V>();Add(low,low,u,L);PipeBarrier<PIPE_V>();Muls(high,tmp,1.0f,L);PipeBarrier<PIPE_V>();
            }
            Add(high,high,low,L);PipeBarrier<PIPE_ALL>();Duplicate(packed,0.0f,512);PipeBarrier<PIPE_V>();Muls(packed,high,1.0f,(uint64_t)16,(uint8_t)16,UnaryRepeatParams{1,1,4,2});PipeBarrier<PIPE_ALL>();DataCopy(dst[row*32],packed,512);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void SaveForwardIdentity(GlobalTensor<float> dst) {
        // Init prepared these six disjoint 16x96 identity row tiles.
        if(lane<6){DataCopy(dst[lane*1536],w[36864],1536);PipeBarrier<PIPE_ALL>();}
    }
    __aicore__ inline void ZeroUnusedFactorRows(GlobalTensor<float> dst) {
        if(lane<4) {Duplicate(w,0.0f,4096);PipeBarrier<PIPE_ALL>();DataCopy(dst[(12+lane)*4096],w,4096);PipeBarrier<PIPE_ALL>();}
    }
    __aicore__ inline void Defect(GlobalTensor<float> dst,GlobalTensor<float> src) {
        if(lane)return;
        DataCopy(w,src,1024);PipeBarrier<PIPE_ALL>();Muls(w,w,-1.0f,1024);PipeBarrier<PIPE_ALL>();
        for(uint32_t i=0;i<32;++i)w.SetValue(i*33,1.0f+w.GetValue(i*33));
        PipeBarrier<PIPE_ALL>();DataCopy(dst,w,1024);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void ReprojectCoefficients(GlobalTensor<float> h,GlobalTensor<float> coefficient,uint32_t cut) {
        for(uint32_t row=lane*16;row<cut;row+=teams*16) {
            DataCopy(w,coefficient[row*32],512);
            DataCopyPad(w[4096],h[row*256+cut],DataCopyExtParams{16,128,896,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
            Add(w[4096],w[4096],w,512);PipeBarrier<PIPE_ALL>();
            DataCopy(coefficient[row*32],w,512);DataCopyPad(h[row*256+cut],w[4096],DataCopyExtParams{16,128,0,896,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void TailQR(GlobalTensor<float> dst,GlobalTensor<float> src,uint32_t real) {
        if(lane)return;
        auto panel=w,triangular=w[8192],product=w[9216],partial=w[13312],sum=w[13568],coeff=w[13632],weights=w[13760];
        auto ix=w[14080].ReinterpretCast<int32_t>();
        DataCopy(panel,src,real*256);Duplicate(triangular,0.0f,1024);Duplicate(partial,0.0f,256);PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<real;++j) {
            Mul(product,panel[j*256],panel[j*256],256);PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,product,64,4,1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,sum,4,1,1,1,8);PipeBarrier<PIPE_V>();Sqrt(sum,sum,1);PipeBarrier<PIPE_ALL>();
            float norm=sum.GetValue(0);triangular.SetValue(j*32+31,norm);PipeBarrier<PIPE_ALL>();
            Muls(panel[j*256],panel[j*256],1.0f/norm,256);PipeBarrier<PIPE_V>();
            uint32_t rows=real-j-1;
            if(rows)for(uint32_t pass=0;pass<2;++pass) {
                for(uint32_t q=0;q<4;++q)Mul(product[q*64],panel[(j+1)*256+q*64],panel[j*256+q*64],(uint64_t)64,(uint8_t)rows,BinaryRepeatParams{1,1,1,32,32,0});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(partial,product,64,rows*4,2,1,8);PipeBarrier<PIPE_V>();
                WholeReduceSum(coeff,partial,8,rows,1,1,1);PipeBarrier<PIPE_V>();
                Add(triangular[j*32],triangular[j*32],coeff,rows);PipeBarrier<PIPE_V>();
                Muls(coeff,coeff,-1.0f,rows);PipeBarrier<PIPE_V>();Brcb(weights,coeff,(uint8_t)((rows+7)/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(panel[(j+1)*256+q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)rows,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_ALL>();
            }
        }
        for(int32_t j=(int32_t)real-1;j>=0;--j) {
            float norm=triangular.GetValue(j*32+31);
            Muls(panel[j*256],panel[j*256],1.0f/norm,256);PipeBarrier<PIPE_V>();
            if(j) {
                for(int32_t r=0;r<j;++r)ix.SetValue(r,(r*32+j-r-1)*4);PipeBarrier<PIPE_ALL>();
                Gather(coeff,triangular,ix.ReinterpretCast<uint32_t>(),0,(uint32_t)j);PipeBarrier<PIPE_V>();
                Muls(coeff,coeff,-1.0f,(uint32_t)j);PipeBarrier<PIPE_V>();Brcb(weights,coeff,(uint8_t)((j+7)/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(panel[q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)j,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_V>();Muls(panel,panel,-1.0f,real*256);PipeBarrier<PIPE_ALL>();DataCopy(dst,panel,real*256);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void Invert(GlobalTensor<float> dst,GlobalTensor<float> src) {
        if(lane)return;
        constexpr uint32_t B=32,R=72;
        auto panel=w,pivot=w[B*R],col=pivot[64],weights=col[B];auto ix=weights[256].ReinterpretCast<int32_t>();
        Duplicate(panel,0.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(panel,src,DataCopyExtParams{B,128,0,5,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r){panel.SetValue(r*R+B+r,-1.0f);ix.SetValue(r,r*R*4);}PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            float d=panel.GetValue(j*R+j);
            Muls(pivot,panel[j*R],-1.0f/d,(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});
            Gather(col,panel[j],ix.ReinterpretCast<uint32_t>(),0,(uint64_t)32,1,8);PipeBarrier<PIPE_V>();
            Brcb(weights,col,4,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(panel,pivot,weights,(uint64_t)64,(uint8_t)32,BinaryRepeatParams{1,1,0,9,0,1});PipeBarrier<PIPE_V>();
            Muls(panel[j*R],pivot,-1.0f,(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(dst,panel[B],DataCopyExtParams{B,128,5,0,0});PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void Invert32(GlobalTensor<float> dst,GlobalTensor<float> src,bool negative=false) {
        if(lane)return;
        constexpr uint32_t B=32,R=72;
        auto panel=w,pivot=w[B*R],col=pivot[64],weights=col[B];auto ix=weights[256].ReinterpretCast<int32_t>();
        Duplicate(panel,0.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(panel,src,DataCopyExtParams{B,128,384,5,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r){panel.SetValue(r*R+B+r,1.0f);ix.SetValue(r,r*R*4);}PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            float d=panel.GetValue(j*R+j);
            Muls(pivot,panel[j*R],-1.0f/d,(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});
            Gather(col,panel[j],ix.ReinterpretCast<uint32_t>(),0,(uint64_t)32,1,8);PipeBarrier<PIPE_V>();
            Brcb(weights,col,4,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(panel,pivot,weights,(uint64_t)64,(uint8_t)32,BinaryRepeatParams{1,1,0,9,0,1});PipeBarrier<PIPE_V>();
            Muls(panel[j*R],pivot,-1.0f,(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_ALL>();
        }
        if(negative){Muls(panel,panel,-1.0f,B*R);PipeBarrier<PIPE_ALL>();}
        DataCopyPad(dst,panel[B],DataCopyExtParams{B,128,5,384,0});PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t N> __aicore__ inline void NegateBlock(GlobalTensor<float> src) {
        if(lane!=1)return;
        DataCopyPad(w,src,DataCopyExtParams{N,N*4,(128-N)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        Muls(w,w,-1.0f,N*N);PipeBarrier<PIPE_ALL>();DataCopyPad(src,w,DataCopyExtParams{N,N*4,0,(128-N)*4,0});PipeBarrier<PIPE_ALL>();
    }


    __aicore__ inline void PadGram96(GlobalTensor<float> gram) {
        uint32_t row=lane*16;
        if(row<96) {
            Duplicate(w,0.0f,512);PipeBarrier<PIPE_ALL>();
            DataCopyPad(gram[row*128+96],w,DataCopyExtParams{16,128,0,384,0});
        } else if(row<128) {
            Duplicate(w,0.0f,2048);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<16;++j)w.SetValue(j*128+row+j,1.0f);
            PipeBarrier<PIPE_ALL>();DataCopy(gram[row*128],w,2048);
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void NewtonPrepare96(GlobalTensor<half> packed,GlobalTensor<float> gram) {
        if(lane>=6)return;
        auto value=w,ab=w[16384],sum=w[32768];DataCopy(value,gram,8192);PipeBarrier<PIPE_ALL>();Abs(ab,value,8192);Duplicate(sum,0.0f,128);PipeBarrier<PIPE_V>();
        Add(ab,ab,ab[64],(uint64_t)64,(uint8_t)64,BinaryRepeatParams{1,1,1,16,16,16});PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,ab,64,64,1,1,16);PipeBarrier<PIPE_V>();WholeReduceMax(sum[256],sum,64,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceMax(sum[512],sum[256],2,1,1,1,8);PipeBarrier<PIPE_ALL>();
        float alpha=1.8f/sum.GetValue(512);auto temp=w[4096],eye=w[12288];auto h=w[16384].ReinterpretCast<half>();
        uint32_t row=lane*16;
        DataCopyPad(value,gram[row*128],DataCopyExtParams{16,384,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();Cast(h,value,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Cast(temp,h,RoundMode::CAST_NONE,1536);PipeBarrier<PIPE_V>();Sub(temp,value,temp,1536);PipeBarrier<PIPE_V>();Cast(h[6144],temp,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(eye,w[36864],1.0f,1536);PipeBarrier<PIPE_V>();Cast(h[3072],eye,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(temp,value,-alpha,1536);PipeBarrier<PIPE_V>();Add(temp,temp,eye,1536);PipeBarrier<PIPE_V>();Cast(h[7680],temp,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(value,eye,-alpha,1536);PipeBarrier<PIPE_V>();Cast(h[1536],value,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(value,eye,2.0f,1536);PipeBarrier<PIPE_V>();Cast(h[4608],value,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(value,eye,-1.0f,1536);PipeBarrier<PIPE_V>();Cast(h[9216],value,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_ALL>();
        DataCopy(packed[row*96],h,DataCopyParams{7,96,0,480});PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SaveFallbackGram96(GlobalTensor<float> dst,GlobalTensor<float> src) {
        if(lane>=6)return;
        const uint32_t row=lane*16;
        DataCopyPad(w,src[row*128],DataCopyExtParams{16,384,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        DataCopyPad(dst[row*128],w,DataCopyExtParams{16,384,0,128,0});PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void FallbackDefect96(GlobalTensor<float> dst) {
        if(lane>=6)return;
        const uint32_t row=lane*16;
        DataCopyPad(w,dst[row*128],DataCopyExtParams{16,384,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        Add(w,w,w[36864],1536);PipeBarrier<PIPE_ALL>();
        DataCopyPad(dst[row*128],w,DataCopyExtParams{16,384,0,128,0});PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ZeroDecision(GlobalTensor<int32_t> decision) {
        if(batch || lane)return;auto flag=w.ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NewtonGuard96(GlobalTensor<float> dst,GlobalTensor<int32_t> decision) {
        if(lane>=6)return;auto ab=w[4096],sum=w[8192];uint32_t row=lane*16;
        DataCopyPad(w,dst[row*128],DataCopyExtParams{16,384,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();Abs(ab,w,1536);PipeBarrier<PIPE_V>();
        Add(ab,ab,ab[32],(uint64_t)32,(uint8_t)16,BinaryRepeatParams{1,1,1,12,12,12});PipeBarrier<PIPE_V>();
        Add(ab,ab,ab[64],(uint64_t)32,(uint8_t)16,BinaryRepeatParams{1,1,1,12,12,12});PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,ab,32,16,1,1,12);PipeBarrier<PIPE_V>();WholeReduceMax(sum[128],sum,16,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(sum[192],sum,16,1,1,1,8);PipeBarrier<PIPE_ALL>();
        float bound=sum.GetValue(128),total=sum.GetValue(192);
        if(!(bound<=0.0625f && total>=0.0f && total<=2.0f)) {
            auto flag=sum[128].ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);PipeBarrier<PIPE_ALL>();
            SetAtomicAdd<int32_t>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();SetAtomicNone();
        }
    }
    __aicore__ inline void NewtonDefect(GlobalTensor<float> dst) {
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            DataCopy(w,dst[row*64],512);PipeBarrier<PIPE_ALL>();Muls(w,w,-1.0f,512);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<8;++j){uint32_t off=j*64+row+j;w.SetValue(off,2.0f+w.GetValue(off));}
            PipeBarrier<PIPE_ALL>();DataCopy(dst[row*64],w,512);PipeBarrier<PIPE_ALL>();
        }
    }

};
template<uint32_t N,bool Cube,class Op>
__aicore__ __attribute__((noinline)) void Inverse(Op &op,GlobalTensor<float> dst,GlobalTensor<float> src,uint32_t batches) {
    if constexpr(N==32) {
        if constexpr(!Cube)op.Invert32(dst,src);SyncAll<false>();
    } else {
        constexpr uint32_t H=N/2,LO=H*128,DIAG=LO+H;
        if constexpr(!Cube)op.template NegateBlock<H>(src[LO]);
        Inverse<H,Cube>(op,dst,src,batches);
        if constexpr(Cube) {
            op.template LocalRun<H>(dst,src[H],src[H],batches,STRIDE);
            op.template LocalRun<H,true>(src[LO],src[H],src[DIAG],batches,STRIDE);
            op.template LocalRun<H>(src[LO],dst,src[LO],batches,STRIDE);
        }
        SyncAll<false>();if constexpr(!Cube)op.template NegateBlock<H>(src[H]);
        Inverse<H,Cube>(op,dst[DIAG],src[DIAG],batches);
        if constexpr(Cube) {
            op.template LocalRun<H>(src[H],dst[DIAG],dst[H],batches,STRIDE);
            op.template LocalRun<H>(dst[DIAG],src[LO],dst[LO],batches,STRIDE);
            op.template LocalRun<H,true>(src[H],dst[LO],dst,batches,STRIDE);
        }
        SyncAll<false>();
    }
}
// Keep additional accuracy work outside the normal Newton path.
template<bool Cube,class Op>
__aicore__ __attribute__((noinline)) void InverseFallback96(Op &op,GlobalTensor<float> mem,uint32_t panel,uint32_t batches) {
    auto inverse=mem[GI],gram=mem[panel==96?XD:G],saved=mem[NT];
    if constexpr(!Cube)op.SaveFallbackGram96(saved,gram);
    SyncAll<false>();
    Inverse<128,Cube>(op,inverse,gram,batches);
    if constexpr(!Cube)op.template NegateBlock<128>(inverse);
    SyncAll<false>();
    // Factor consumes inverse*Gram; refine this side to avoid amplifying its defect.
    if constexpr(Cube)op.template LocalRun<96>(inverse,saved,gram,batches,STRIDE);
    SyncAll<false>();
    if constexpr(!Cube)op.FallbackDefect96(gram);
    SyncAll<false>();
    if constexpr(Cube){op.template LocalRun<96,true>(gram,inverse,inverse,batches,STRIDE);SyncCube();}
}
// Every Cube participates in the global producer/consumer join.
// Mode2 exchanges only the data-ready notification with its paired two AIVs.
template<bool Cube> __aicore__ inline void CubeToVector() {
    if constexpr(Cube) {
        SyncCube();
        ffts_cross_core_sync(PIPE_FIX,GetffstMsg(0x2,1));
    } else {
        PipeBarrier<PIPE_ALL>();wait_flag_dev(1);
    }
}
template<bool Cube> __aicore__ inline void VectorToCube() {
    if constexpr(Cube) {
        wait_flag_dev(0);SyncCube();
    } else {
        PipeBarrier<PIPE_ALL>();
        ffts_cross_core_sync(PIPE_MTE3,GetffstMsg(0x2,0));
    }
}
template<bool Cube,bool Packed=false> __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,uint32_t batches,uint32_t k,bool broadcast,uint32_t n) {
    GlobalTensor<float> mem,A,B,X;
    mem.SetGlobalBuffer((__gm__ float*)workspace);A.SetGlobalBuffer((__gm__ float*)aa);B.SetGlobalBuffer((__gm__ float*)bb);X.SetGlobalBuffer((__gm__ float*)xx);
    std::conditional_t<Cube,NativeMM,Vector> op;
    uint32_t batch=0;
    if constexpr(Cube)op.Init();else {op.Init(batches);batch=op.batch;mem=mem[(uint64_t)batch*STRIDE];A=A[(uint64_t)batch*n*n];B=B[(uint64_t)(broadcast?0:batch)*n*k];X=X[(uint64_t)batch*n*k];}
    GlobalTensor<int8_t> ad,xd;GlobalTensor<int32_t> products;
    ad.SetGlobalBuffer((__gm__ int8_t*)mem[AD].GetPhyAddr());xd.SetGlobalBuffer((__gm__ int8_t*)mem[XD].GetPhyAddr());products.SetGlobalBuffer((__gm__ int32_t*)mem[PRODUCTS].GetPhyAddr());
    GlobalTensor<half> halfMem;halfMem.SetGlobalBuffer((__gm__ half*)workspace+((uint64_t)batch*STRIDE+NH)*2);
    if constexpr(!Cube){if constexpr(Packed){op.NormalizeSeed(mem[CT],A,B,k,n);op.PackCachedB(mem[BP],mem[DX],k);}else{op.Normalize(mem[CT],A,mem[REDUCE],n);op.PackB(mem[BP],mem[DX],B,k,n);}if constexpr(Packed)op.SaveForwardIdentity(mem[FORWARD_IDENTITY]);op.ZeroUnusedFactorRows(mem[D]);}
    if constexpr(Packed)VectorToCube<Cube>();else SyncAll<false>();
    for(uint32_t p=0;p<n;p+=(p<192?96:32)) {
        if(p==192) {
            // G is dead after both local inverses; retain E here through both RHS applications.
            if constexpr(Cube){if constexpr(Packed)op.TailCoefficients224(mem[D],mem[CT],mem[G],batches,STRIDE);else op.template Run<192,32,256,false,true>(mem[D],mem[CT+p*256],mem[R],batches,STRIDE,STRIDE,STRIDE);}
            if constexpr(Packed) {
                if constexpr(Cube)SyncCube();
            } else {
                SyncAll<false>();
                if constexpr(!Cube)op.ReprojectCoefficients(mem[H],mem[R],p);
                SyncAll<false>();
            }
            if constexpr(Cube){if constexpr(Packed)op.ReprojectTail224(mem[G],mem[CT],batches,STRIDE);else op.template Run<32,256,192,true,false,true>(mem[R],mem[CT],mem[CT+p*256],batches,STRIDE,STRIDE,STRIDE);}
            if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
            if constexpr(!Cube){op.TailQR(mem[D+p*256],mem[CT+p*256],n-p);op.SplitAChunk(ad,mem[EA],A,96,n);op.SplitAChunk(ad,mem[EA],A,128,n);}
            if constexpr(Cube && Packed)op.ComposeForward96(mem[H],mem[G],mem[FORWARD_IDENTITY],mem[FORWARD_MAP],batches,STRIDE);
            // Forward uses H/F/BP only; it can overlap TailQR's stores to D.
            if constexpr(Packed){if constexpr(Cube)SyncCube();}else SyncAll<false>();
        } else {
            // XD is unused until initial X is split; a padded128 Gram fits here.
            auto gram=mem[(Packed && p==96)?XD:G];
            if constexpr(Cube) {
                if constexpr(Packed) {
                    if(p==96)op.SchurSecondGram96(mem[AN],mem[H],gram,batches,STRIDE);
                    else op.template Gram96<208>(mem[CT],gram,batches,STRIDE,p);
                } else op.template Gram96<256>(mem[CT],gram,batches,STRIDE,p);
            } else op.PadGram96(gram);
            SyncAll<false>();
            GlobalTensor<int32_t> decisions;decisions.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+128);
            if constexpr(!Cube){op.ZeroDecision(decisions);op.NewtonPrepare96(halfMem,gram);}
            if constexpr(Cube && Packed) {
                if(p==96)op.template Update96<208>(mem[H],mem[CT],batches,STRIDE,0);
            }
            if constexpr(Packed)VectorToCube<Cube>();else SyncAll<false>();
            if constexpr(Cube) {
                if(p==0)op.template Newton96<false>(halfMem,mem[GI],mem[NT],batches,STRIDE,8);
                else {
                    if constexpr(Packed) {
                        op.SecondNewtonRefreshed96(mem,batches);
                    } else op.template Newton96<true>(halfMem,mem[GI],mem[NT],batches,STRIDE,14);
                }
                if constexpr(Packed) {
                    if(p==0)op.CrossAndSecondGram208(mem[CT],mem[AN],mem[XD],batches,STRIDE);

                }
            } else {
                if(p==0)op.SplitAChunk(ad,mem[EA],A,0,n);
                else {op.SplitAChunk(ad,mem[EA],A,32,n);op.SplitAChunk(ad,mem[EA],A,64,n);}
            }
            if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
            if constexpr(!Cube)op.NewtonGuard96(mem[NT],decisions);
            if constexpr(Cube){op.Factor96(mem[GI],mem[CT],mem[D],batches,STRIDE,p);if constexpr(Packed)op.HFromCross208(mem[GI],mem[AN],mem[H],batches,STRIDE,p);}
            SyncAll<false>();DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decisions);
            if(decisions.GetValue(0)!=0) {
                if constexpr(Packed) {
                    if(p==96) {
                        // The fallback retains the original explicit-C1 Gram target.
                        if constexpr(Cube)op.template Gram96<208>(mem[CT],gram,batches,STRIDE,p);
                        SyncAll<false>();
                    }
                }
                if constexpr(Packed)InverseFallback96<Cube>(op,mem,p,batches);
                else {
                    // Preserve the Packed call graph; both wide-RHS panels use G.
                    if constexpr(!Cube)op.SaveFallbackGram96(mem[NT],gram);
                    SyncAll<false>();
                    Inverse<128,Cube>(op,mem[GI],gram,batches);
                    if constexpr(!Cube)op.template NegateBlock<128>(mem[GI]);SyncAll<false>();
                    if constexpr(Cube)op.template LocalRun<96>(mem[GI],mem[NT],gram,batches,STRIDE);
                    SyncAll<false>();
                    if constexpr(!Cube)op.FallbackDefect96(gram);
                    SyncAll<false>();
                    if constexpr(Cube){op.template LocalRun<96,true>(gram,mem[GI],mem[GI],batches,STRIDE);SyncCube();}
                }
                if constexpr(Cube){op.Factor96(mem[GI],mem[CT],mem[D],batches,STRIDE,p);if constexpr(Packed)op.HFromCross208(mem[GI],mem[AN],mem[H],batches,STRIDE,p);}SyncAll<false>();
            }
        }
        if(p<192) {
            if constexpr(Cube && !Packed)op.template TrailingH96<256>(mem[D],mem[CT],mem[H],batches,STRIDE,p);
            if constexpr(Cube && !Packed)SyncCube();
            if constexpr(Cube) {
                if constexpr(Packed) {
                    if(p!=0){op.template Update96<208>(mem[H],mem[CT],batches,STRIDE,p);SyncCube();}
                } else {op.template Update96<256>(mem[H],mem[CT],batches,STRIDE,p);SyncCube();}
            }
        }
    }
    if constexpr(Cube){if constexpr(Packed)op.Forward96Parallel(mem[H],mem[FORWARD_MAP],mem[BP],mem[DX],batches,STRIDE);else op.Forward96(mem[H],mem[DX],batches,STRIDE);}
    if constexpr(Cube)SyncCube();
    if constexpr(Packed)VectorToCube<Cube>();
    if constexpr(Cube){if constexpr(Packed)op.ApplyFactor208(mem[D],mem[DX],mem[XP],batches,STRIDE);else op.template Run<256,32,256,true>(mem[D],mem[DX],mem[XP],batches,STRIDE,STRIDE,STRIDE);}
    if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
    if constexpr(!Cube){if constexpr(Packed)op.SplitXPacked4(xd,mem[EX],mem[XP],k);else op.SplitX(xd,mem[EX],mem[XP],k);}
    if constexpr(Packed)VectorToCube<Cube>();else SyncAll<false>();
    if constexpr(Cube){if constexpr(Packed)op.Int8Packed4(ad,xd,products,batches,STRIDE);else op.Int8All(ad,xd,products,batches,STRIDE);}
    if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
    if constexpr(!Cube){if constexpr(Packed)op.Assemble6Packed4(mem[R],mem[DX],products,mem[EA],mem[EX],mem[BP]);else op.Assemble6(mem[R],products,mem[EA],mem[EX],mem[BP]);}
    if constexpr(Packed)VectorToCube<Cube>();else SyncAll<false>();
    if constexpr(Cube){if constexpr(Packed)op.Forward96Parallel(mem[H],mem[FORWARD_MAP],mem[R],mem[DX],batches,STRIDE);else op.Forward96(mem[H],mem[R],batches,STRIDE);}
    if constexpr(Cube)SyncCube();
    if constexpr(Cube){if constexpr(Packed)op.ApplyFactor208(mem[D],mem[DX],mem[R],batches,STRIDE);else op.template Run<256,32,256,true>(mem[D],mem[R],mem[DX],batches,STRIDE,STRIDE,STRIDE);}
    if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
    if constexpr(!Cube){if constexpr(Packed)op.Finish16(X,mem[XP],mem[R],k,n);else op.Finish(X,mem[XP],mem[DX],k,n);}
}
}

#ifndef SOLVE_DIRECT_BENCH
 template <typename DT_A,uint32_t KERNEL_PATH>
__global__ __aicore__ void linalg_solve(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    InitSocState();
    REGISTER_TILING_DEFAULT(LinalgSolveCubeTilingData);
    if constexpr(KERNEL_PATH==27) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData,t,tiling);
        TinyColumns op;op.Run(A,B,X,GetUserWorkspace(workspace),t);
    } else if constexpr(KERNEL_PATH==28) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData,t,tiling);
        SmallColumns op;op.Run(A,B,X,GetUserWorkspace(workspace),t);
    } else if constexpr(KERNEL_PATH==25 || KERNEL_PATH==26) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        if ASCEND_IS_AIC {NarrowOrthogonal::Run<true,KERNEL_PATH==26>(A,B,X,GetUserWorkspace(workspace),cube.base.batches,cube.base.k,cube.base.broadcastB,cube.base.n);}
        if ASCEND_IS_AIV {NarrowOrthogonal::Run<false,KERNEL_PATH==26>(A,B,X,GetUserWorkspace(workspace),cube.base.batches,cube.base.k,cube.base.broadcastB,cube.base.n);}
    } else if constexpr(KERNEL_PATH==24) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        if ASCEND_IS_AIC {Orthogonal::Run<true>(A,B,X,GetUserWorkspace(workspace),cube.base.batches,cube.base.k,cube.base.broadcastB);}
        if ASCEND_IS_AIV {Orthogonal::Run<false>(A,B,X,GetUserWorkspace(workspace),cube.base.batches,cube.base.k,cube.base.broadcastB);}
    } else if constexpr(KERNEL_PATH==23) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        GM_ADDR user=GetUserWorkspace(workspace);GM_ADDR system=user-cube.systemWorkspaceBytes;
        AscendC::tiling::TCubeTiling tc{};
        if ASCEND_IS_AIC {auto words=(uint32_t*)&tc;for(uint32_t i=0;i<sizeof(tc)/4;++i)words[i]=cube.cubeWords[i];}
        const uint32_t resident=GetBlockNum()/2;
        for(uint32_t first=0;first<cube.base.batches;first+=resident) {
            auto t=cube.base;t.batches=cube.base.batches-first<resident?cube.base.batches-first:resident;
            GM_ADDR aa=A+(uint64_t)first*t.n*t.n*4;
            GM_ADDR bb=B+(uint64_t)(t.broadcastB?0:first)*t.n*t.k*4;
            GM_ADDR xx=X+(uint64_t)first*t.n*t.k*4;
            if ASCEND_IS_AIC {DirectPanel<128>::Service(user,system,tc,t.batches,t.n,t.k);}
            if ASCEND_IS_AIV {DirectPanel<128> op;op.Run(aa,bb,xx,user,t);}
            if(first+resident<cube.base.batches)SyncAll<false>();
        }
    } else if constexpr(KERNEL_PATH==22) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        GM_ADDR user=GetUserWorkspace(workspace);GM_ADDR system=user-cube.systemWorkspaceBytes;
        AscendC::tiling::TCubeTiling tc{},ht{};
        if ASCEND_IS_AIC {auto words=(uint32_t*)&tc;for(uint32_t i=0;i<sizeof(tc)/4;++i)words[i]=cube.cubeWords[i];}
        if ASCEND_IS_AIC {auto words=(uint32_t*)&ht;for(uint32_t i=0;i<sizeof(ht)/4;++i)words[i]=cube.halfWords[i];}
        const uint32_t resident=GetBlockNum()/2;
        for(uint32_t first=0;first<cube.base.batches;first+=resident) {
            auto t=cube.base;t.batches=cube.base.batches-first<resident?cube.base.batches-first:resident;
            GM_ADDR aa=A+(uint64_t)first*t.n*t.n*4;
            GM_ADDR bb=B+(uint64_t)(t.broadcastB?0:first)*t.n*t.k*4;
            GM_ADDR xx=X+(uint64_t)first*t.n*t.k*4;
            if ASCEND_IS_AIC {NormalInverse<256,32,2,true>::Service(user,system,tc,t.batches,t.n,t.k,&ht,aa);}
            if ASCEND_IS_AIV {NormalInverse<256,32,2,true> op;op.Run(aa,bb,xx,user,t);}
            if(first+resident<cube.base.batches)SyncAll<false>();
        }
    } else if constexpr(KERNEL_PATH==18 || KERNEL_PATH==21) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        GM_ADDR user=GetUserWorkspace(workspace);GM_ADDR system=user-cube.systemWorkspaceBytes;
        AscendC::tiling::TCubeTiling tc{},ht{};
        if ASCEND_IS_AIC {auto words=(uint32_t*)&tc;for(uint32_t i=0;i<sizeof(tc)/4;++i)words[i]=cube.cubeWords[i];}
        if ASCEND_IS_AIC {auto words=(uint32_t*)&ht;for(uint32_t i=0;i<sizeof(ht)/4;++i)words[i]=cube.halfWords[i];}
        constexpr uint32_t N=KERNEL_PATH==21?384:512;
        const uint32_t resident=GetBlockNum()/2;
        for(uint32_t first=0;first<cube.base.batches;first+=resident) {
            auto t=cube.base;t.batches=cube.base.batches-first<resident?cube.base.batches-first:resident;
            GM_ADDR aa=A+(uint64_t)first*t.n*t.n*4;
            GM_ADDR bb=B+(uint64_t)(t.broadcastB?0:first)*t.n*t.k*4;
            GM_ADDR xx=X+(uint64_t)first*t.n*t.k*4;
            if ASCEND_IS_AIC {NormalInverse<N,32,2>::Service(user,system,tc,t.batches,t.n,t.k,&ht,aa);}
            if ASCEND_IS_AIV {NormalInverse<N,32,2> op;op.Run(aa,bb,xx,user,t);}
            if(first+resident<cube.base.batches)SyncAll<false>();
        }
    } else if constexpr(KERNEL_PATH==20) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        GM_ADDR user=GetUserWorkspace(workspace);GM_ADDR system=user-cube.systemWorkspaceBytes;
        AscendC::tiling::TCubeTiling tc{},ht{};
        if ASCEND_IS_AIC {auto words=(uint32_t*)&tc;for(uint32_t i=0;i<sizeof(tc)/4;++i)words[i]=cube.cubeWords[i];}
        if ASCEND_IS_AIC {auto words=(uint32_t*)&ht;for(uint32_t i=0;i<sizeof(ht)/4;++i)words[i]=cube.halfWords[i];}
        if ASCEND_IS_AIC {NormalInverse<256,32,2>::Service(user,system,tc,cube.base.batches,cube.base.n,cube.base.k,&ht,A);}
        if ASCEND_IS_AIV {NormalInverse<256,32,2> op;op.Run(A,B,X,user,cube.base);}
    } else if constexpr(KERNEL_PATH==19) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        GM_ADDR user=GetUserWorkspace(workspace);GM_ADDR system=user-cube.systemWorkspaceBytes;
        AscendC::tiling::TCubeTiling tc{},ht{};
        if ASCEND_IS_AIC {auto words=(uint32_t*)&tc;for(uint32_t i=0;i<sizeof(tc)/4;++i)words[i]=cube.cubeWords[i];}
        if ASCEND_IS_AIC {auto words=(uint32_t*)&ht;for(uint32_t i=0;i<sizeof(ht)/4;++i)words[i]=cube.halfWords[i];}
        const uint32_t resident=GetBlockNum()/2;
        for(uint32_t first=0;first<cube.base.batches;first+=resident) {
            auto t=cube.base;t.batches=cube.base.batches-first<resident?cube.base.batches-first:resident;
            GM_ADDR aa=A+(uint64_t)first*t.n*t.n*4;
            GM_ADDR bb=B+(uint64_t)(t.broadcastB?0:first)*t.n*t.k*4;
            GM_ADDR xx=X+(uint64_t)first*t.n*t.k*4;
            if ASCEND_IS_AIC {DirectPanel<208,256,true>::Service(user,system,tc,t.batches,t.n,t.k,&ht);}
            if ASCEND_IS_AIV {DirectPanel<208,256,true> op;op.Run(aa,bb,xx,user,t);}
            if(first+resident<cube.base.batches)SyncAll<false>();
        }
    } else if constexpr(KERNEL_PATH==9 || KERNEL_PATH==10 || KERNEL_PATH==11 || KERNEL_PATH==13 || KERNEL_PATH==15) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        GM_ADDR user=GetUserWorkspace(workspace);GM_ADDR system=user-cube.systemWorkspaceBytes;
        AscendC::tiling::TCubeTiling tc{},ht{};
        if ASCEND_IS_AIC {auto words=(uint32_t*)&tc;for(uint32_t i=0;i<sizeof(tc)/4;++i)words[i]=cube.cubeWords[i];}
        if constexpr(KERNEL_PATH==15) {if ASCEND_IS_AIC {auto words=(uint32_t*)&ht;for(uint32_t i=0;i<sizeof(ht)/4;++i)words[i]=cube.halfWords[i];}}
        constexpr uint32_t N=(KERNEL_PATH==9 || KERNEL_PATH==13)?128:256;
        const uint32_t resident=GetBlockNum()/2;
        for(uint32_t first=0;first<cube.base.batches;first+=resident) {
            auto t=cube.base;t.batches=cube.base.batches-first<resident?cube.base.batches-first:resident;
            GM_ADDR aa=A+(uint64_t)first*t.n*t.n*4;
            GM_ADDR bb=B+(uint64_t)(t.broadcastB?0:first)*t.n*t.k*4;
            GM_ADDR xx=X+(uint64_t)first*t.n*t.k*4;
            if constexpr(KERNEL_PATH==10) {
                if ASCEND_IS_AIC {DirectPanel<N>::Service(user,system,tc,t.batches,t.n,t.k);}
                if ASCEND_IS_AIV {DirectPanel<N> op;op.Run(aa,bb,xx,user,t);}
            } else {
                if ASCEND_IS_AIC {CubePanel<N,(KERNEL_PATH>=11),(KERNEL_PATH==15)>::Service(user,system,tc,ht,bb,t.batches,t.n,t.k,t.broadcastB);}
                if ASCEND_IS_AIV {CubePanel<N,(KERNEL_PATH>=11),(KERNEL_PATH==15)> op;op.Run(aa,bb,xx,user,t);}
            }
            if(first+resident<cube.base.batches)SyncAll<false>();
        }
    } else {
    GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData,t,tiling);
    if constexpr(KERNEL_PATH==5) {RefinedSquare<208,16,1,true> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==6) {RefinedSquare<256,16,1,true> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==7) {RefinedSquare<160,16,1,true> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==2) {RefinedSquare<128,32> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==3) {RefinedSquare<256,32> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==1) {
        // Capacities follow vector/alignment boundaries, not individual orders.
        if(t.n<=16) {SmallSquare<16> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
        else if(t.n<=32) {SmallSquare<32> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
        else if(t.n<=56) {SmallSquare<56> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
        else {SmallSquare<64> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    }
    else {SolveKernel op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    }
}
#endif
