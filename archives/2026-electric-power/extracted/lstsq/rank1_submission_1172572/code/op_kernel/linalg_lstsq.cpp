#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "linalg_lstsq_tiling.h"
#ifndef LSTSQ_DIRECT_BENCH
#include "tiling_key_linalg_lstsq.h"
#endif
using namespace AscendC;

// A number is hi+lo. QR and back substitution both run on the device.
struct LstsqPair {float hi,lo;};
class Lstsq {
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuf,packBuf;
    LocalTensor<float> qh,ql,ah,al,dh,dl,ph,pl,xh,xl,s[8],packed;
    GlobalTensor<float> inputA,inputB,outputX,outputR,high,low,columnSolutions;
    LinalgLstsqTilingData t;
    uint32_t rhsRows;
    bool hasResiduals;
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
    __aicore__ inline LstsqPair AddPair(LstsqPair a,LstsqPair b) {
        // Vector instructions preserve each required FP32 rounding step.
        PipeBarrier<PIPE_ALL>();
        s[6].SetValue(16,a.hi);s[6].SetValue(24,a.lo);
        s[6].SetValue(32,b.hi);s[6].SetValue(40,b.lo);PipeBarrier<PIPE_ALL>();
        AddVector(s[6],s[6][8],s[6][16],s[6][24],s[6][32],s[6][40],1);
        PipeBarrier<PIPE_ALL>();return {s[6].GetValue(0),s[6].GetValue(8)};
    }
    static __aicore__ inline LstsqPair Neg(LstsqPair a) {return {-a.hi,-a.lo};}
    __aicore__ inline LstsqPair Product(float a,float b) {
        PipeBarrier<PIPE_ALL>();
        s[6].SetValue(16,a);s[6].SetValue(24,0);
        s[6].SetValue(32,b);s[6].SetValue(40,0);PipeBarrier<PIPE_ALL>();
        MulVector(s[6],s[6][8],s[6][16],s[6][24],s[6][32],s[6][40],1);
        PipeBarrier<PIPE_ALL>();return {s[6].GetValue(0),s[6].GetValue(8)};
    }
    __aicore__ inline LstsqPair MulPair(LstsqPair a,LstsqPair b) {
        LstsqPair p=Product(a.hi,b.hi);
        return AddPair(p,{a.hi*b.lo+a.lo*b.hi,a.lo*b.lo});
    }
    __aicore__ inline LstsqPair DivPair(LstsqPair a,LstsqPair b) {
        float h=a.hi/b.hi;
        LstsqPair remainder=AddPair(a,Neg(MulPair(b,{h,0})));
        return AddPair({h,0},{(remainder.hi+remainder.lo)/b.hi,0});
    }
    __aicore__ inline LstsqPair Root(LstsqPair a) {
        s[7].SetValue(0,a.hi);PipeBarrier<PIPE_ALL>();
        Sqrt(s[7],s[7],1);PipeBarrier<PIPE_ALL>();
        const float h=s[7].GetValue(0);
        if(h==0)return {0,0};
        const LstsqPair r=AddPair(a,Neg(Product(h,h)));
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
        VAdd(s[0],a,b,n);VSub(s[1],s[0],a,n);VSub(s[2],s[0],s[1],n);
        VSub(s[2],a,s[2],n);VSub(s[3],b,s[1],n);VAdd(s[2],s[2],s[3],n);
        VAdd(s[2],s[2],ae,n);VAdd(s[2],s[2],be,n);
        VAdd(oh,s[0],s[2],n);VSub(s[3],oh,s[0],n);VSub(s[4],oh,s[3],n);
        VSub(s[4],s[0],s[4],n);VSub(s[5],s[2],s[3],n);VAdd(ol,s[4],s[5],n);
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
        VMul(s[5],ae,b,n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],ae,be,n);VAdd(ol,s[4],s[5],n);
    }
    __aicore__ inline LstsqPair Dot(LocalTensor<float> a,LocalTensor<float> ae,
            LocalTensor<float> b,LocalTensor<float> be,uint32_t n) {
        MulVector(ph,pl,a,ae,b,be,n);
        while(n>8) {n/=2;AddVector(ph,pl,ph,pl,ph[n],pl[n],n);}
        PipeBarrier<PIPE_ALL>();
        LstsqPair total{0,0};
        for(uint32_t i=0;i<n;++i)total=AddPair(total,{ph.GetValue(i),pl.GetValue(i)});
        return total;
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
        for(uint32_t row=0;row<t.m;++row)ah.SetValue(row,input.GetValue(base+(uint64_t)row*width+col));
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Transform(uint32_t col,LstsqPair tau) {
        Load(col,ah,al);const LstsqPair factor=MulPair(tau,Dot(qh,ql,ah,al,t.rows));
        Fill(ph,-factor.hi,t.rows);Fill(pl,-factor.lo,t.rows);
        MulVector(dh,dl,qh,ql,ph,pl,t.rows);
        AddVector(ph,pl,ah,al,dh,dl,t.rows);Store(col,ph,pl);
    }
    __aicore__ inline void Solve(uint32_t batch,uint32_t rhs,float scale) {
        Load(t.n+rhs,ah,al);
        if(hasResiduals) {
            for(uint32_t i=0;i<t.n;++i){ah.SetValue(i,0);al.SetValue(i,0);}
            PipeBarrier<PIPE_ALL>();
            LstsqPair norm=Dot(ah,al,ah,al,t.rows);
            norm=MulPair(MulPair(norm,{1/scale,0}),{1/scale,0});
            s[7].SetValue(8,norm.hi+norm.lo);Sync<HardEvent::S_MTE3>();
            DataCopyPad(outputR[(uint64_t)batch*t.k+rhs],s[7][8],DataCopyExtParams{1,4,0,0,0});
            Sync<HardEvent::MTE3_S>();
        }
        Load(t.n+rhs,qh,ql);
        Fill(xh,0,rhsRows);Fill(xl,0,rhsRows);
        for(int32_t row=t.n-1;row>=0;--row) {
            DataCopyPad(packed,high[row],DataCopyExtParams{(uint16_t)t.n,4,(t.rows-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();Fill(ah,0,rhsRows);
            for(uint32_t j=0;j<t.n;++j)ah.SetValue(j,packed.GetValue(j*8));
            DataCopyPad(packed,low[row],DataCopyExtParams{(uint16_t)t.n,4,(t.rows-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();Fill(al,0,rhsRows);
            for(uint32_t j=0;j<t.n;++j)al.SetValue(j,packed.GetValue(j*8));
            PipeBarrier<PIPE_ALL>();
            const LstsqPair diag{ah.GetValue(row),al.GetValue(row)};
            const LstsqPair value{qh.GetValue(row),ql.GetValue(row)};
            const LstsqPair result=DivPair(AddPair(value,Neg(Dot(ah,al,xh,xl,rhsRows))),diag);
            xh.SetValue(row,result.hi);xl.SetValue(row,result.lo);PipeBarrier<PIPE_ALL>();
        }
        Add(ph,xh,xl,t.n);PipeBarrier<PIPE_V>();Sync<HardEvent::V_MTE3>();
        DataCopyPad(columnSolutions[(uint64_t)rhs*t.n],ph,DataCopyExtParams{1,t.n*4,0,0,0});
        Sync<HardEvent::MTE3_S>();
    }
public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR r,GM_ADDR workspace,const LinalgLstsqTilingData &data) {
        t=data;hasResiduals=r!=nullptr && t.residualElements>=(uint64_t)t.batches*t.k;rhsRows=8;while(rhsRows<t.n)rhsRows*=2;
        pipe.InitBuffer(dataBuf,18*1024*4);pipe.InitBuffer(packBuf,8*1024*4);
        auto all=dataBuf.Get<float>();
        qh=all;ql=all[1024];ah=all[2048];al=all[3072];dh=all[4096];dl=all[5120];ph=all[6144];pl=all[7168];
        xh=all[8192];xl=all[9216];for(uint32_t i=0;i<8;++i)s[i]=all[(10+i)*1024];
        packed=packBuf.Get<float>();
        inputA.SetGlobalBuffer((__gm__ float*)a);inputB.SetGlobalBuffer((__gm__ float*)b);
        outputX.SetGlobalBuffer((__gm__ float*)x);outputR.SetGlobalBuffer((__gm__ float*)r);
        const uint64_t words=(uint64_t)(t.n+t.k)*t.rows;
        high.SetGlobalBuffer((__gm__ float*)workspace+GetBlockIdx()*2*words);
        low.SetGlobalBuffer((__gm__ float*)workspace+(GetBlockIdx()*2+1)*words);
        columnSolutions.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.blocks*2*words+(uint64_t)GetBlockIdx()*((t.n*t.k+7)/8*8));
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            float largest=0;
            for(uint32_t col=0;col<t.n;++col) {
                InputColumn(inputA,(uint64_t)batch*t.m*t.n,t.n,col);
                for(uint32_t row=0;row<t.m;++row){float v=ah.GetValue(row);v=v<0?-v:v;if(v>largest)largest=v;}
                Store(col,ah,al);
            }
            union {float value;uint32_t bits;} exponent;exponent.value=largest;
            uint32_t power=254-((exponent.bits>>23)&255);if(power>253)power=253;if(power<1)power=1;
            exponent.bits=power<<23;const float scale=exponent.value;
            for(uint32_t col=0;col<t.n;++col){Load(col,ah,al);VScale(ah,ah,scale,t.rows);Store(col,ah,al);}
            for(uint32_t rhs=0;rhs<t.k;++rhs){InputColumn(inputB,(uint64_t)batch*t.m*t.k,t.k,rhs);VScale(ah,ah,scale,t.rows);Store(t.n+rhs,ah,al);}
            for(uint32_t j=0;j<t.n;++j) {
                Load(j,qh,ql);
                for(uint32_t i=0;i<j;++i){qh.SetValue(i,0);ql.SetValue(i,0);}
                PipeBarrier<PIPE_ALL>();
                LstsqPair alpha=Root(Dot(qh,ql,qh,ql,t.rows));
                if(qh.GetValue(j)>=0)alpha=Neg(alpha);
                const LstsqPair v=AddPair({qh.GetValue(j),ql.GetValue(j)},Neg(alpha));
                qh.SetValue(j,v.hi);ql.SetValue(j,v.lo);PipeBarrier<PIPE_ALL>();
                const LstsqPair tau=DivPair({2,0},Dot(qh,ql,qh,ql,t.rows));
                for(uint32_t col=j+1;col<t.n+t.k;++col)Transform(col,tau);
                Load(j,ah,al);ah.SetValue(j,alpha.hi);al.SetValue(j,alpha.lo);
                for(uint32_t i=j+1;i<t.rows;++i){ah.SetValue(i,0);al.SetValue(i,0);}
                Store(j,ah,al);
            }
            for(uint32_t rhs=0;rhs<t.k;++rhs)Solve(batch,rhs,scale);
            WriteRows(batch);
        }
    }
};

// Gauss-Jordan elimination updates all rows in one vector repeat group.
template<uint32_t P,uint32_t T=8> class AccurateRows {
    static constexpr uint32_t SZ=P*T,PAIR=4*T;
public:
    static constexpr uint32_t FLOATS=9*SZ+14*T;
    LocalTensor<float> a,x,h,l,ah,al,xh,xl,s[4],raw,ahi,alo;
    LocalTensor<uint32_t> ixBase;
    __aicore__ inline void Init(LocalTensor<float> mem) {
        a=mem;x=mem[SZ];h=mem[2*SZ];l=mem[3*SZ];ah=mem[4*SZ];al=mem[5*SZ];xh=mem[6*SZ];xl=mem[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=mem[(4+i)*SZ];raw=mem[8*SZ];ahi=mem[4*SZ];alo=mem[5*SZ];ixBase=mem[9*SZ].template ReinterpretCast<uint32_t>();
    }
    __aicore__ inline void Split(uint32_t rows) {
        Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
    }
    template<HardEvent event> __aicore__ inline void Sync() {
        SetFlag<event>(0);WaitFlag<event>(0);
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
        A(s[2],s[2],a[PAIR],count);A(l,s[2],x[PAIR],count);
        Scale(h,s[0],1.0f,count);
    }
    __aicore__ inline void AlignedRowSum(LocalTensor<float> out,LocalTensor<float> input,uint32_t rows) {
        for(uint32_t half=P/2;half>=8;half/=2)Rows(input,input,input[half],rows,half);
        WholeReduceSum(a,input,8,rows,1,1,P/8);PipeBarrier<PIPE_V>();
        Scale(out,a,1.0f,rows);
    }
    __aicore__ inline void AccurateDot(uint32_t rows) {
        Split(rows);
        const uint32_t count=rows*P;
        M(h,raw,x,count);
        Scale(xh,x,4097.0f,count);D(xl,xh,x,count);D(xh,xh,xl,count);D(xl,x,xh,count);
        M(l,ahi,xh,count);D(l,l,h,count);
        M(a,ahi,xl,count);A(l,l,a,count);
        M(a,alo,xh,count);A(l,l,a,count);
        M(a,alo,xl,count);A(l,l,a,count);
        Abs(a,h,count);PipeBarrier<PIPE_V>();
        if constexpr(P==64) {
            WholeReduceMax(s[0],a,64,rows,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMax(s[0][128],s[0],(uint64_t)rows,1,1,1,8);
        } else if constexpr(P>64)WholeReduceMax(s[0],a,64,count/64,1,1,8);
        else {
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,count);
            WholeReduceMax<float,false>(s[0],a,MASK_PLACEHOLDER,1,1,1,8);SetMaskNorm();ResetMask();
        }
        Sync<HardEvent::V_S>();
        uint32_t maximumBits,maximumIndex;GetReduceRepeatMaxMinSpr<float>(maximumBits,maximumIndex);
        const uint32_t exponent=(maximumBits>>23)&255;
        if(exponent>10 && exponent<240) {
            constexpr uint32_t logP=P==512?9:(P==256?8:(P==128?7:(P==64?6:4)));
            union ScaleBits {uint32_t bits;float value;};
            ScaleBits q0,q1;q0.bits=((exponent+logP+1)<<23)|0x400000;
            q1.bits=((exponent+2*logP-22)<<23)|0x400000;
            Adds(s[0],h,q0.value,count);PipeBarrier<PIPE_V>();Adds(s[0],s[0],-q0.value,count);PipeBarrier<PIPE_V>();
            D(s[1],h,s[0],count);
            if constexpr(P==64) {
                // The coarse lattice keeps its row sum exact. Sum the remaining
                // product and correction together, then retain the final TwoSum error.
                A(s[1],s[1],l,count);
                AlignedRowSum(s[0],s[0],rows);AlignedRowSum(s[1],s[1],rows);
                A(h,s[0],s[1],rows);D(a,h,s[0],rows);D(x,h,a,rows);
                D(x,s[0],x,rows);D(a,s[1],a,rows);A(l,x,a,rows);
                return;
            }
            Adds(s[2],s[1],q1.value,count);PipeBarrier<PIPE_V>();Adds(s[2],s[2],-q1.value,count);PipeBarrier<PIPE_V>();
            D(s[3],s[1],s[2],count);
            AlignedRowSum(s[0],s[0],rows);AlignedRowSum(s[1],s[2],rows);
            AlignedRowSum(s[2],s[3],rows);AlignedRowSum(s[3],l,rows);
            A(a,s[2],s[3],rows);
            A(x,s[1],a,rows);D(h,x,s[1],rows);D(l,x,h,rows);D(l,s[1],l,rows);D(h,a,h,rows);A(l,l,h,rows);
            A(h,s[0],x,rows);D(s[1],h,s[0],rows);D(s[2],h,s[1],rows);D(s[2],s[0],s[2],rows);D(s[1],x,s[1],rows);A(a,s[2],s[1],rows);A(l,a,l,rows);
            return;
        }
        for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
            auto i0=ixBase[4*T*(4-half)],i1=i0[T*half];
            for(uint32_t r=0;r<T;++r)for(uint32_t j=0;j<half;++j){i0.SetValue(r*half+j,(r*stride+j)*4);i1.SetValue(r*half+j,(r*stride+half+j)*4);}
        }
        PipeBarrier<PIPE_ALL>();
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
            Gather(a[PAIR],l,ix0,0,rows*half);Gather(x[PAIR],l,ix1,0,rows*half);
            PipeBarrier<PIPE_V>();
            DensePairSum(rows*half);
        }
    }
};

template<uint32_t P,bool Parallel=false,bool InverseRows=false,bool InitialOwner=false,bool InitialOnly=false> class ResidentElimination {
    static constexpr uint32_t L=P+8,W=P+8,R=P;
    static_assert(!InverseRows || (Parallel && P==64));
    static_assert(!InitialOwner || InverseRows);
    static_assert(!InitialOnly || (P==64 && !Parallel && !InverseRows));
    __aicore__ inline void Index(LocalTensor<int32_t> dst) {
        for(int32_t i=0;i<8;++i)dst.SetValue(i,i);
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        UnaryRepeatParams params;
        for(uint32_t i=0;i<P/8-1;++i){Adds(dst[(i+1)*8],dst[i*8],(int32_t)8,(uint64_t)8,1,params);PipeBarrier<PIPE_V>();}
    }
    __aicore__ inline void ApplyInverseRows(LocalTensor<float> result,LocalTensor<float> inverse,
        LocalTensor<float> value,LocalTensor<float> products,uint32_t width) {
        Mul(products,inverse,value,(uint64_t)width,(uint8_t)8,BinaryRepeatParams{1,1,1,P/8,P/8,0});PipeBarrier<PIPE_V>();
        WholeReduceSum(result,products,(uint64_t)width,8,1,1,P/8);PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR rr,GM_ADDR workspace,const LinalgLstsqTilingData &t) {
        const uint32_t n=t.n,k=t.k;
        const bool packedInput=P==16 || ((Parallel || InitialOnly) && n%8!=0);
        constexpr uint32_t BASE=W*L+3*P*8+W+P+64+P+8*W+P+8*P+W;
        LocalTensor<float> mat(TPosition::VECCALC,0,BASE+AccurateRows<P,R>::FLOATS+256);
        auto rhs=mat[W*L],solution=rhs[P*8],residual=solution[P*8];
        auto pivot=residual[P*8],candidates=pivot[W],maximum=candidates[P],scales=maximum[64],weights=scales[P];
        auto rowIx=weights[8*W].template ReinterpretCast<uint32_t>();auto output=rowIx[P].template ReinterpretCast<float>();auto offsets=output[8*P].template ReinterpretCast<int32_t>();
        AccurateRows<P,R> dot;dot.Init(mat[BASE]);
        auto input=dot.raw,product=dot.h;auto ix=dot.a.template ReinterpretCast<int32_t>(),tmpIx=dot.x.template ReinterpretCast<int32_t>();
        static_assert(!InitialOnly || 14*R>=P*8);
        auto outputIndices=dot.ixBase;
        GlobalTensor<float> ga,gb,gx,gr;ga.SetGlobalBuffer((__gm__ float*)aa);gb.SetGlobalBuffer((__gm__ float*)bb);gx.SetGlobalBuffer((__gm__ float*)xx);gr.SetGlobalBuffer((__gm__ float*)rr);
        constexpr uint32_t vs=0;
        const uint32_t lane=Parallel?GetBlockIdx()%t.teams:0,group=Parallel?GetBlockIdx()/t.teams:GetBlockIdx();
        const uint32_t groups=Parallel?t.blocks/t.teams:t.blocks;
        const bool ownsInverse=InverseRows && (!InitialOwner || lane!=0);
        const uint32_t inverseLane=InitialOwner?(lane?lane-1:0):lane,augmented=ownsInverse?8:k;
        const bool nativeInput=InitialOnly && n%8==0;
        uint64_t transposeSrc[4][16],transposeDst[4][16];
        TransDataTo5HDParams transposeParams;
        if constexpr(InitialOnly)if(nativeInput) {
            for(uint32_t group=0;group<4;++group) {
                for(uint32_t row=0;row<16;++row)
                    transposeSrc[group][row]=(uint64_t)input[(group*16+row)*P].GetPhyAddr();
                for(uint32_t col=0;col<8;++col) {
                    transposeDst[group][2*col]=(uint64_t)mat[col*L+group*16].GetPhyAddr();
                    transposeDst[group][2*col+1]=(uint64_t)mat[col*L+group*16+8].GetPhyAddr();
                }
            }
            transposeParams.dstHighHalf=false;transposeParams.srcHighHalf=false;
            transposeParams.repeatTimes=n/8;transposeParams.dstRepStride=L;transposeParams.srcRepStride=1;
        }
        GlobalTensor<float> shared;shared.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)group*2*P*8);
        for(uint32_t batch=group;batch<t.batches;batch+=groups) {
            uint32_t pivotRows[P],residualRow=0,residualRows=0;bool refine=P!=16 && !InitialOnly;
            if constexpr(Parallel) {
                const uint32_t rowTeams=(t.teams-(InitialOwner?1:0))/k,chunk=((P+rowTeams-1)/rowTeams+7)/8*8;
                residualRow=inverseLane/k*chunk;
                if((!InitialOwner || ownsInverse) && inverseLane<rowTeams*k && residualRow<n)residualRows=n-residualRow<chunk?n-residualRow:chunk;
            }
            {
            // A transposed solve gives each lane eight independent inverse rows.
            // The original resident path instead replicates the initial solution.
            if constexpr(P==16) {
                // DMA and transpose-index construction use disjoint UB regions.
                if(n<P){Duplicate(input,0.0f,P*P);PipeBarrier<PIPE_ALL>();}
                DataCopyPad(input,ga[(uint64_t)batch*n*n],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                DataCopyPad(rhs,gb[(uint64_t)batch*n*k],DataCopyExtParams{1,n*k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            }
            Index(tmpIx);
            Muls(offsets,tmpIx,(int32_t)4,P);PipeBarrier<PIPE_V>();
            if(!nativeInput) {
                Muls(tmpIx,tmpIx,(int32_t)((packedInput?n:P)*4),P);PipeBarrier<PIPE_V>();
                if(ownsInverse) {
                    Brcb(weights.template ReinterpretCast<int32_t>(),tmpIx,P/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                    Add(ix,offsets,weights.template ReinterpretCast<int32_t>(),(uint64_t)P,(uint8_t)P,BinaryRepeatParams{1,1,0,P/8,0,1});
                } else {
                    Brcb(weights.template ReinterpretCast<int32_t>(),offsets,P/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                    Add(ix,tmpIx,weights.template ReinterpretCast<int32_t>(),(uint64_t)P,(uint8_t)P,BinaryRepeatParams{1,1,0,P/8,0,1});
                }
            }
            PipeBarrier<PIPE_ALL>();
            if constexpr(P!=16)if(n<P)Duplicate(input,0.0f,P*P);
            Duplicate(solution,0.0f,P*k);Duplicate(scales,1.0f,P);PipeBarrier<PIPE_ALL>();
            if constexpr(P!=16) {
                if(packedInput)DataCopyPad(input,ga[(uint64_t)batch*n*n],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                else DataCopyPad(input,ga[(uint64_t)batch*n*n],DataCopyExtParams{(uint16_t)n,n*4,0,(P-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
                DataCopyPad(rhs,gb[(uint64_t)batch*n*k],DataCopyExtParams{1,n*k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                if constexpr(InitialOnly)if(k>1) {
                    // Row byte offsets already exist here; prepare the final
                    // output mapping while input DMA targets disjoint buffers.
                    if((k&(k-1))==0) {
                        auto rowBytes=dot.l.template ReinterpretCast<int32_t>();
                        auto columnBytes=dot.h.template ReinterpretCast<int32_t>();
                        const uint32_t shift=k==2?1:(k==4?2:3);
                        ShiftRight(rowBytes,offsets,(int32_t)(shift+2),P);PipeBarrier<PIPE_V>();
                        ShiftLeft(rowBytes,rowBytes,(int32_t)2,P);PipeBarrier<PIPE_V>();
                        Muls(columnBytes,rowBytes,(int32_t)k,P);PipeBarrier<PIPE_V>();
                        Sub(columnBytes,offsets,columnBytes,P);PipeBarrier<PIPE_V>();
                        Muls(columnBytes,columnBytes,(int32_t)P,P);PipeBarrier<PIPE_V>();
                        Add(outputIndices.template ReinterpretCast<int32_t>(),columnBytes,rowBytes,P);PipeBarrier<PIPE_V>();
                        for(uint32_t c=1;c<k;++c)Adds(outputIndices[c*P].template ReinterpretCast<int32_t>(),outputIndices.template ReinterpretCast<int32_t>(),(int32_t)(c*(P/k)*4),P);
                    } else {
                        for(uint32_t r=0;r<n;++r)for(uint32_t c=0;c<k;++c)outputIndices.SetValue(r*k+c,(c*P+r)*4);
                        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
                    }
                }
                PipeBarrier<PIPE_ALL>();
            }
            if constexpr(P==16) {
                Gather(product,input,ix.template ReinterpretCast<uint32_t>(),0,n*P);PipeBarrier<PIPE_V>();
                DataCopy(mat,product,DataCopyParams{(uint16_t)n,P/8,0,(L-P)/8});
            } else {
                if(nativeInput) {
                    // Include padded rows so the complete 64-row capacity matches
                    // the original Gather; RHS columns are filled afterward.
                    for(uint32_t group=0;group<4;++group) {
                        TransDataTo5HD<float>(transposeDst[group],transposeSrc[group],transposeParams);
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    if(packedInput){Duplicate(mat,0.0f,W*L);PipeBarrier<PIPE_V>();}
                    Gather(mat,input,ix.template ReinterpretCast<uint32_t>(),0,(uint64_t)(packedInput?n:P),(uint8_t)n,(uint16_t)(L/8));
                }
            }
            PipeBarrier<PIPE_V>();
            Muls(tmpIx,offsets,(int32_t)k,P);PipeBarrier<PIPE_V>();
            Muls(offsets,offsets,(int32_t)L,P);PipeBarrier<PIPE_V>();
            Adds(offsets[P],offsets[P-8],(int32_t)(8*L*4),(uint64_t)8,1,UnaryRepeatParams{});PipeBarrier<PIPE_ALL>();
            if(ownsInverse) {
                Duplicate(mat[n*L],0.0f,8*L);PipeBarrier<PIPE_ALL>();
                for(uint32_t c=0;c<8;++c)if(inverseLane*8+c<n)mat.SetValue((n+c)*L+inverseLane*8+c,1.0f);
                PipeBarrier<PIPE_ALL>();
            } else {
                for(uint32_t c=0;c<k;++c)Gather(mat[(n+c)*L],rhs,tmpIx.template ReinterpretCast<uint32_t>(),c*4,P);PipeBarrier<PIPE_V>();
            }
            uint32_t pivotFloorBits=0;
            if constexpr(P==64 && !Parallel && !InitialOnly) {
                Abs(product,input,P*P);PipeBarrier<PIPE_V>();
                if constexpr(P==64) {
                    WholeReduceMax(dot.a,product,64,P,1,1,8);PipeBarrier<PIPE_V>();
                    WholeReduceMax(maximum,dot.a,P,1,1,1,8);
                } else {
                    // Collapse the 16-row input to one reduction without
                    // mixing index fields into the magnitude bound.
                    Max(product,product,product[128],128);PipeBarrier<PIPE_V>();
                    Max(product,product,product[64],64);PipeBarrier<PIPE_V>();
                    WholeReduceMax(maximum,product,64,1,1,1,8);
                }
                SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint32_t maximumBits,index;GetReduceRepeatMaxMinSpr<float>(maximumBits,index);
                pivotFloorBits=maximumBits>(3U<<23)?maximumBits-(3U<<23):0;
            }
            uint64_t live[2]={n==64?~0ULL:(1ULL<<n)-1,0};
            for(uint32_t j=0;j<n;++j) {
                auto col=mat[j*L];uint32_t bits,row;float pivotValue;
                if constexpr(P==16 || (P==64 && (Parallel || InitialOnly))) {
                    Abs(candidates,col,P);PipeBarrier<PIPE_V>();
                    WholeReduceMax(maximum,candidates,live,1,1,1,8);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                    GetReduceRepeatMaxMinSpr<float>(bits,row);
                    if constexpr(P!=16)pivotValue=col.GetValue(row);
                } else {
                    row=63-static_cast<uint32_t>(CountLeadingZero(live[0]&(~live[0]+1ULL)));
                    SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                    union {float value;uint32_t bits;} first;first.value=col.GetValue(row);
                    if((first.bits&0x7fffffffU)<pivotFloorBits || (first.bits&0x7fffffffU)==0) {
                        Abs(candidates,col,P);PipeBarrier<PIPE_V>();WholeReduceMax(maximum,candidates,live,1,1,1,8);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                        GetReduceRepeatMaxMinSpr<float>(bits,row);pivotValue=col.GetValue(row);
                    } else pivotValue=first.value;
                }
                if constexpr(P==16) {
                    live[0]&=~(1ULL<<row);pivotRows[j]=row;rowIx.SetValue(j,row*4);
                    uint64_t mask[2]={1ULL<<(row%8),0};
                    const uint32_t width=n+augmented-j-1;
                    // Fetch the signed divisor with the remaining pivot row.
                    // The first broadcast block supplies all divisor lanes.
                    Gather(pivot,mat,offsets.template ReinterpretCast<uint32_t>(),(j*L+row)*4,width+1);PipeBarrier<PIPE_V>();
                    Brcb(weights,pivot,(width+8)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                    Muls(scales[(row/8)*8],weights,1.0f,mask,1,UnaryRepeatParams{1,1,8,8});
                    Duplicate(col[(row/8)*8],0.0f,mask,1,1,8);
                    Muls(weights,weights,-1.0f,8);PipeBarrier<PIPE_V>();
                    Div(col,col,weights,(uint64_t)P,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
                    MulAddDst(mat[(j+1)*L],col,weights[8],(uint64_t)P,(uint8_t)width,BinaryRepeatParams{1,1,0,L/8,0,1});PipeBarrier<PIPE_V>();
                } else {
                    const float scaleValue=1.0f/pivotValue;live[0]&=~(1ULL<<row);
                    pivotRows[j]=row;rowIx.SetValue(j,row*4);uint64_t mask[2]={1ULL<<(row%8),0};
                    Duplicate(col[(row/8)*8],0.0f,mask,1,1,8);Duplicate(scales[(row/8)*8],scaleValue,mask,1,1,8);
                    const uint32_t width=n+augmented-j-1;
                    const uint32_t firstColumn=InitialOnly?((j+1)/8*8):(j+1);
                    const uint32_t gatheredWidth=n+augmented-firstColumn;
                    Gather(pivot,mat,offsets.template ReinterpretCast<uint32_t>(),(firstColumn*L+row)*4,gatheredWidth);PipeBarrier<PIPE_V>();
                    Muls(col,col,-scaleValue,P);PipeBarrier<PIPE_V>();
                    auto producer=weights,consumer=weights;
                    constexpr uint32_t coefficientStride=InitialOnly?L/8:1;
                    if constexpr(InitialOnly) {
                        // Aligned groups stay inside the reserved column capacity.
                        // Only the still-live columns consume their padding slots.
                        producer=mat[firstColumn*L+P];consumer=mat[(j+1)*L+P];
                    }
                    Brcb(producer,pivot,(gatheredWidth+7)/8,BrcbRepeatParams{coefficientStride,8*coefficientStride});PipeBarrier<PIPE_V>();
                    MulAddDst(mat[(j+1)*L],col,consumer,(uint64_t)P,(uint8_t)width,BinaryRepeatParams{1,1,0,L/8,0,coefficientStride});PipeBarrier<PIPE_V>();
                }
            }
            for(uint32_t c=n;c<n+augmented;++c) {
                if constexpr(P==16)Div(mat[c*L],mat[c*L],scales,P);
                else Mul(mat[c*L],mat[c*L],scales,P);
            }
            PipeBarrier<PIPE_ALL>();
            if constexpr(InverseRows) {
                if(ownsInverse) {
                    Duplicate(output,0.0f,8*P);PipeBarrier<PIPE_V>();
                    for(uint32_t c=0;c<8;++c)Gather(output[c*P],mat[(n+c)*L],rowIx,0,n);PipeBarrier<PIPE_V>();
                    if constexpr(!InitialOwner)for(uint32_t c=0;c<k;++c) {
                        Gather(residual,rhs,tmpIx.template ReinterpretCast<uint32_t>(),c*4,n);PipeBarrier<PIPE_V>();
                        ApplyInverseRows(pivot,output,residual,product,n);PipeBarrier<PIPE_ALL>();
                        DataCopy(shared[c*P+inverseLane*8],pivot,8);PipeBarrier<PIPE_ALL>();
                    }
                } else {
                    for(uint32_t c=0;c<k;++c)Gather(solution[c*P],mat[(n+c)*L],rowIx,0,n);PipeBarrier<PIPE_ALL>();
                    DataCopy(shared,solution,P*k);PipeBarrier<PIPE_ALL>();
                }
                SyncAll<true>();DataCopy(solution,shared,P*k);PipeBarrier<PIPE_ALL>();
            } else {
                for(uint32_t c=0;c<k;++c)Gather(solution[c*P],mat[(n+c)*L],rowIx,0,n);PipeBarrier<PIPE_ALL>();
            }
            }
            if constexpr(Parallel) {
                // Decide from the preliminary solution's numerical magnitude.
                Abs(product,solution,P*k);PipeBarrier<PIPE_V>();
                WholeReduceMax(maximum,product,(uint64_t)P,(uint8_t)k,1,1,P/8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                WholeReduceMax(maximum[32],maximum,(uint64_t)k,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_ALL>();
                refine=!(maximum.GetValue(32)<=10.0f);
            }
            if constexpr(Parallel) {
                if(refine) {
                if(residualRows) {
                    const uint32_t row=residualRow,rows=residualRows;
                    if(packedInput) {
                        DataCopyPad(input,ga[(uint64_t)batch*n*n+row*n],DataCopyExtParams{(uint16_t)rows,n*4,0,(P-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});PipeBarrier<PIPE_ALL>();
                    } else if(row){DataCopy(input,input[row*P],rows*P);PipeBarrier<PIPE_ALL>();}
                    {
                        const uint32_t c=inverseLane%k;
                        Muls(dot.x,solution[c*P],1.0f,(uint64_t)P,(uint8_t)rows,UnaryRepeatParams{1,1,P/8,0});PipeBarrier<PIPE_V>();dot.AccurateDot(rows);
                        for(uint32_t r=0;r<rows;++r)tmpIx.SetValue(r,(r*k+c)*4);PipeBarrier<PIPE_ALL>();
                        Gather(dot.a,rhs[row*k],tmpIx.template ReinterpretCast<uint32_t>(),0,rows);PipeBarrier<PIPE_V>();
                        dot.D(dot.h,dot.a,dot.h,rows);dot.D(residual[c*P],dot.h,dot.l,rows);PipeBarrier<PIPE_ALL>();
                        DataCopyPad(shared[P*8+c*P+row],residual[c*P],DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
                    }
                }
                }
                SyncAll<true>();
                if(refine && (ownsInverse || (!InverseRows && lane==0))){DataCopy(residual,shared[P*8],P*k);PipeBarrier<PIPE_ALL>();}
            } else if(refine) {
            for(uint32_t row=0;row<n;row+=R) {
                const uint32_t rows=n-row<R?n-row:R;
                PipeBarrier<PIPE_ALL>();
                for(uint32_t c=0;c<k;++c) {
                    Muls(dot.x,solution[c*P],1.0f,(uint64_t)P,(uint8_t)rows,UnaryRepeatParams{1,1,P/8,0});PipeBarrier<PIPE_V>();dot.AccurateDot(rows);
                    for(uint32_t r=0;r<rows;++r)tmpIx.SetValue(r,(r*k+c)*4);PipeBarrier<PIPE_ALL>();
                    Gather(dot.a,rhs[row*k],tmpIx.template ReinterpretCast<uint32_t>(),0,rows);PipeBarrier<PIPE_V>();dot.D(dot.h,dot.a,dot.h,rows);dot.D(residual[c*P+row],dot.h,dot.l,rows);
                }
            }
            }
            if constexpr(InverseRows) {
                const uint32_t first=inverseLane*8,rows=ownsInverse && first<n?(n-first<8?n-first:8):0;
                if(rows) {
                    for(uint32_t c=0;c<k;++c) {
                        ApplyInverseRows(pivot,output,residual[c*P],product,n);
                        Add(pivot,pivot,solution[c*P+first],8);PipeBarrier<PIPE_V>();
                        DataCopy(mat[c*8],pivot,8);PipeBarrier<PIPE_V>();
                    }
                    for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<k;++c)tmpIx.SetValue(r*k+c,(c*8+r)*4);
                    PipeBarrier<PIPE_ALL>();Gather(product,mat,tmpIx.template ReinterpretCast<uint32_t>(),0,rows*k);PipeBarrier<PIPE_ALL>();
                    DataCopyPad(gx[(uint64_t)batch*n*k+first*k],product,DataCopyExtParams{1,rows*k*4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
                if(lane==0 && rr && t.residualElements>=(uint64_t)t.batches*k){Duplicate(pivot,0.0f,8);PipeBarrier<PIPE_ALL>();DataCopyPad(gr[(uint64_t)batch*k],pivot,DataCopyExtParams{1,k*4,0,0,0});PipeBarrier<PIPE_ALL>();}
            } else if(lane==0) {
            if(refine) {
            if(k>1) {
                for(uint32_t c=0;c<k;++c)offsets.SetValue(c,c*P*4);PipeBarrier<PIPE_ALL>();
                for(uint32_t j=0;j<n;++j) {
                    Gather(pivot,residual,offsets.template ReinterpretCast<uint32_t>(),pivotRows[j]*4,k);PipeBarrier<PIPE_V>();
                    Brcb(weights,pivot,1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                    MulAddDst(residual,mat[j*L],weights,(uint64_t)P,(uint8_t)k,BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
                }
            } else {
                for(uint32_t j=0;j<n;++j) {
                    const uint32_t row=pivotRows[j];
                    Brcb(weights,residual[(row/8)*8],1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                    MulAddDst(residual,mat[j*L],weights[(row%8)*8],(uint64_t)P,1,BinaryRepeatParams{1,1,0,P/8,L/8,0});PipeBarrier<PIPE_V>();
                }
            }
            for(uint32_t c=0;c<k;++c) {
                auto corr=residual[c*P];
                Mul(corr,corr,scales,P);PipeBarrier<PIPE_V>();
                Gather(pivot,corr,rowIx,0,n);PipeBarrier<PIPE_V>();Add(solution[c*P],solution[c*P],pivot,n);PipeBarrier<PIPE_V>();
            }
            }
            if(k==1) {
                PipeBarrier<PIPE_ALL>();DataCopyPad(gx[(uint64_t)batch*n],solution,DataCopyExtParams{1,n*4,0,0,0});PipeBarrier<PIPE_ALL>();
            } else {
                if constexpr(!InitialOnly) {
                if((k&(k-1))==0) {
                    // Flatten (row,rhs) into column-contiguous solution offsets.
                    // The explicit-event index helper does not need a TPipe.
                    Index(tmpIx);const uint32_t shift=k==2?1:(k==4?2:3);
                    ShiftRight(ix,tmpIx,(int32_t)shift,P);PipeBarrier<PIPE_V>();
                    Muls(output.template ReinterpretCast<int32_t>(),ix,(int32_t)k,P);PipeBarrier<PIPE_V>();
                    Sub(tmpIx,tmpIx,output.template ReinterpretCast<int32_t>(),P);PipeBarrier<PIPE_V>();
                    Muls(ix,ix,(int32_t)4,P);Muls(tmpIx,tmpIx,(int32_t)(P*4),P);PipeBarrier<PIPE_V>();
                    Add(tmpIx,tmpIx,ix,P);PipeBarrier<PIPE_V>();
                    for(uint32_t c=1;c<k;++c)Adds(tmpIx[c*P],tmpIx,(int32_t)(c*(P/k)*4),P);
                    PipeBarrier<PIPE_V>();
                } else {
                    for(uint32_t r=0;r<n;++r)for(uint32_t c=0;c<k;++c)tmpIx.SetValue(r*k+c,(c*P+r)*4);PipeBarrier<PIPE_ALL>();
                }
                }
                auto resultOffsets=InitialOnly?outputIndices:tmpIx.template ReinterpretCast<uint32_t>();
                Gather(output,solution,resultOffsets,0,n*k);PipeBarrier<PIPE_ALL>();DataCopyPad(gx[(uint64_t)batch*n*k],output,DataCopyExtParams{1,n*k*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
            if(rr && t.residualElements>=(uint64_t)t.batches*k){Duplicate(pivot,0.0f,8);PipeBarrier<PIPE_ALL>();DataCopyPad(gr[(uint64_t)batch*k],pivot,DataCopyExtParams{1,k*4,0,0,0});PipeBarrier<PIPE_ALL>();}
            }
            if constexpr(Parallel)if(batch+groups<t.batches)SyncAll<true>();
        }
    }
};

// Deferred row normalization avoids scattered pivot-row writes in column layout.
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
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR rr,GM_ADDR workspace,const LinalgLstsqTilingData &t) {
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
                    DataCopyPad(mat[rhsOffset],gb[(uint64_t)batch*t.n*t.k],DataCopyExtParams{(uint16_t)t.n,t.k*4,0,(L-qp)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(qp-t.k),0});
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
                        const uint32_t first=((!rhs || SharedRhs)?(done<=lane*C?0:(done-lane*C<C?done-lane*C:C)):0)/8*8;
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
                        DataCopyPad(a,gb[(uint64_t)batch*t.n*t.k+row*t.k],DataCopyExtParams{(uint16_t)rows,t.k*4,0,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-t.k),0});
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
            if(live && lane==0 && rr && t.residualElements>=(uint64_t)t.batches*t.k) {
                GlobalTensor<float> gr;gr.SetGlobalBuffer((__gm__ float*)rr);
                Duplicate(mat,0.0f,8);PipeBarrier<PIPE_ALL>();
                DataCopyPad(gr[(uint64_t)batch*t.k],mat,DataCopyExtParams{1,t.k*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
            SyncAll<true>();
        }
    }
};


__aicore__ inline void SyncCube() {
    PipeBarrier<PIPE_ALL>();
    ffts_cross_core_sync(PIPE_FIX,GetffstMsg(0x0,SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
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
        uint32_t K,uint32_t as,uint32_t bs,uint32_t cs,uint32_t skip=0,uint32_t mirrorOffset=0) {
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
        Fixpipe<float,float,CFG_ROW_MAJOR>(C[skip],c[skip*BM],f);
        if(mirrorOffset)Fixpipe<float,float,CFG_ROW_MAJOR>(C[mirrorOffset+skip],c[skip*BM],f);
        PipeBarrier<PIPE_ALL>();
        if constexpr(Accumulate)SetAtomicNone();
    }
    template<uint32_t M,uint32_t N,uint32_t K,bool TA=false,bool TB=false,bool Accumulate=false>
    __aicore__ inline void Run(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,
        uint32_t batches,uint32_t astride,uint32_t bstride,uint32_t cstride,
        uint32_t as=TA?M:K,uint32_t bs=TB?K:N,uint32_t cs=N,uint32_t mirrorOffset=0) {
        constexpr uint32_t BM=(M==128&&N>=128)?64:(M>=128?((M==192 || (N==32 && K>=128))?64:128):(M==64 && N>=128?64:(M>=32?32:16)));
        constexpr uint32_t BN=N>=128?(M<=128?64:128):(N>=32?32:16);
        constexpr uint32_t BK=K==192 && BM<=64 && BN<=64?192:(K>=256 && BM<=64 && BN<=64?256:(K>=128?128:(K>=64?64:(K>=32?32:16))));
        constexpr uint32_t nr=M/BM,nc=N/BN;
        for(uint32_t task=GetBlockIdx();task<batches*nr*nc;task+=GetBlockNum()) {
            uint32_t batch=task/(nr*nc),r=task/nc%nr*BM,col=task%nc*BN;
            Tile<BM,BN,BK,TA,TB,Accumulate>(A[batch*astride+(TA?r:r*as)],B[batch*bstride+(TB?col*bs:col)],C[batch*cstride+r*cs+col],K,as,bs,cs,0,mirrorOffset);
        }
    }
    __aicore__ inline void ReprojectTail(GlobalTensor<float> ct,GlobalTensor<float> inverse,GlobalTensor<float> h,GlobalTensor<float> cross,GlobalTensor<float> correction,uint32_t batches,uint32_t stride,uint32_t panel) {
        if(panel==128)Run<128,64,256,false,true,false>(ct,ct[panel*256],cross,batches,stride,stride,stride,256,256,64);
        else Run<192,64,256,false,true,false>(ct,ct[panel*256],cross,batches,stride,stride,stride,256,256,64);
        SyncCube();
        const uint32_t blocks=panel/64;
        for(uint32_t task=GetBlockIdx();task<batches*blocks;task+=GetBlockNum()) {
            const uint32_t batch=task/blocks,block=task%blocks;
            Tile<64,64,64,false,false,false>(inverse[batch*stride+block*4096],cross[batch*stride+block*4096],correction[batch*stride+block*4096],64,64,64,64);
            auto acc=cbuf.Get<float>();FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=256;
            SetAtomicAdd<float>();Fixpipe<float,float,CFG_ROW_MAJOR>(h[batch*stride+block*64*256+panel],acc,f);PipeBarrier<PIPE_ALL>();SetAtomicNone();
        }
        SyncCube();
        if(panel==128)Run<64,256,128,true,false,true>(correction,ct,ct[panel*256],batches,stride,stride,stride,64,256,256);
        else Run<64,256,192,true,false,true>(correction,ct,ct[panel*256],batches,stride,stride,stride,64,256,256);
        SyncCube();
    }
    __aicore__ inline void TailProducts32(GlobalTensor<float> ct,GlobalTensor<float> gram,GlobalTensor<float> cross,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*2;task+=GetBlockNum()) {
            const uint32_t batch=task/2;
            if(task%2==0)Tile<32,32,256,false,true,false>(ct[batch*stride],ct[batch*stride],gram[batch*stride],256,256,256,64);
            else Tile<32,32,256,false,true,false>(ct[batch*stride],ct[batch*stride+32*256],cross[batch*stride],256,256,256,32);
        }
    }
    __aicore__ inline void ComposeForward(GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        Run<64,128,64,false,false,true>(h[64],h[64*256+128],h[128],batches,stride,stride,stride,256,256,256);SyncCube();
        if(panel==192){Run<128,64,64,false,false,true>(h[128],h[128*256+192],h[192],batches,stride,stride,stride,256,256,256);SyncCube();}
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
    __aicore__ inline void ForwardApplyInverse64(GlobalTensor<float> H,GlobalTensor<float> inverses,GlobalTensor<float> input,GlobalTensor<float> output,uint32_t batches,uint32_t stride,uint32_t activeBlocks) {
        // One owner per RHS block. Forward reads immutable input; the same
        // owner then applies its local inverse after Tile's completion fence.
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,block=task%4;if(block>=activeBlocks)continue;
            auto h=H[batch*stride+block*64],x=input[batch*stride],y=output[batch*stride+block*2048];
            if(block==1)Tile<64,32,64,true,false,true>(h,x,y,64,256,32,32);
            else if(block==2)Tile<64,32,128,true,false,true>(h,x,y,128,256,32,32);
            else if(block==3)Tile<64,32,192,true,false,true>(h,x,y,192,256,32,32);
            if(block+1<activeBlocks)Tile<64,32,64,true,false,false>(inverses[batch*stride+block*4096],y,y,64,64,32,32);
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
};

#include <type_traits>
namespace Orthogonal {
constexpr uint32_t CT=0,D=65536,H=131072,AN=196608,G=262144,Q32=266240,
 BP=270336,XP=278528,R=286720,DX=294912,REDUCE=303104,NT=303360,NH=307456,AD=319744,XD=418048,PRODUCTS=430336,EA=438528,EX=438784,STRIDE=kOrthogonalWorkspaceFloats;
// Phase measurements put one packing chunk near four direct-QR rows.
// Move only the uncovered packing work into the longest prior Cube phases.
__aicore__ inline uint32_t EarlyPackingChunks(uint32_t n) {
    const uint32_t directRows=(n-1)%16+1,available=(n-1)/64;
    const uint32_t wanted=(24-directRows+7)/8;
    return wanted<available?wanted:available;
}
static_assert(NH+12288==AD);
static_assert(EX+16==STRIDE);
class Vector {
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
    LocalTensor<float> w;
    uint32_t lane,teams;float inputScale=1.0f;
public:
    uint32_t batch;
    __aicore__ inline void Init(uint32_t batches) {
        uint32_t id=GetBlockIdx();batch=id%batches;lane=id/batches;teams=(2*GetBlockNum()+batches-1-batch)/batches;
        pipe.InitBuffer(buf,192*1024);w=buf.Get<float>();
        if(lane<8) {
            auto identity=w[36864];Duplicate(identity,0.0f,512);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<8;++j)identity.SetValue(j*64+lane*8+j,1.0f);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void TailGuard32(GlobalTensor<float> defect,GlobalTensor<int32_t> decision) {
        if(lane)return;
        DataCopy(w,defect,1024);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<32;++r)w.SetValue(r*32+r,1.0f+w.GetValue(r*32+r));PipeBarrier<PIPE_ALL>();
        Abs(w[1024],w,1024);PipeBarrier<PIPE_V>();WholeReduceSum(w[2048],w[1024],32,32,1,1,4);PipeBarrier<PIPE_V>();
        WholeReduceMax(w[2112],w[2048],32,1,1,1,8);WholeReduceSum(w[2120],w[2048],32,1,1,1,8);PipeBarrier<PIPE_ALL>();
        const float bound=w.GetValue(2112),total=w.GetValue(2120);
        if(!(bound<=0.015625f && total>=0.0f && total<=0.5f)) {
            auto flag=w[2176].ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);PipeBarrier<PIPE_ALL>();
            SetAtomicAdd<int32_t>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();SetAtomicNone();
        }
    }
    __aicore__ inline void TailGuard16(GlobalTensor<float> defect,GlobalTensor<int32_t> decision) {
        if(lane)return;
        DataCopy(w,defect,256);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<16;++r)w.SetValue(r*16+r,1.0f+w.GetValue(r*16+r));PipeBarrier<PIPE_ALL>();
        Abs(w[1024],w,256);PipeBarrier<PIPE_V>();WholeReduceSum(w[2048],w[1024],16,16,1,1,2);PipeBarrier<PIPE_V>();
        WholeReduceMax(w[2112],w[2048],16,1,1,1,8);WholeReduceSum(w[2120],w[2048],16,1,1,1,8);PipeBarrier<PIPE_ALL>();
        const float bound=w.GetValue(2112),total=w.GetValue(2120);
        if(!(bound<=0.015625f && total>=0.0f && total<=0.25f)) {
            auto flag=w[2176].ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);PipeBarrier<PIPE_ALL>();
            SetAtomicAdd<int32_t>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();SetAtomicNone();
        }
    }
    __aicore__ inline void PackTail16(GlobalTensor<float> inverse,GlobalTensor<float> cross) {
        if(lane!=1)return;
        DataCopy(w,cross,256);PipeBarrier<PIPE_ALL>();DataCopyPad(inverse[16],w,DataCopyExtParams{16,64,0,64,0});PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void PackTail32(GlobalTensor<float> inverse,GlobalTensor<float> cross) {
        if(lane!=1)return;
        DataCopy(w,cross,1024);PipeBarrier<PIPE_ALL>();DataCopyPad(inverse[32],w,DataCopyExtParams{32,128,0,128,0});PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void TailQR(GlobalTensor<float> dst,GlobalTensor<float> src,uint32_t real) {
        if(lane)return;
        auto panel=w,triangular=w[16384],product=w[20480],partial=w[36864],sum=w[37376],coeff=w[37440],weights=w[37504];
        auto ix=w[38016].ReinterpretCast<int32_t>();
        DataCopy(panel,src,real*256);Duplicate(triangular,0.0f,4096);Duplicate(partial,0.0f,512);PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<real;++j) {
            Mul(product,panel[j*256],panel[j*256],256);PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,product,64,4,1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,sum,4,1,1,1,8);PipeBarrier<PIPE_V>();Sqrt(sum,sum,1);PipeBarrier<PIPE_ALL>();
            float norm=sum.GetValue(0);triangular.SetValue(j*64+63,norm);PipeBarrier<PIPE_ALL>();
            Muls(panel[j*256],panel[j*256],1.0f/norm,256);PipeBarrier<PIPE_V>();
            uint32_t rows=real-j-1;
            if(rows)for(uint32_t pass=0;pass<2;++pass) {
                for(uint32_t q=0;q<4;++q)Mul(product[q*64],panel[(j+1)*256+q*64],panel[j*256+q*64],(uint64_t)64,(uint8_t)rows,BinaryRepeatParams{1,1,1,32,32,0});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(partial,product,64,rows*4,2,1,8);PipeBarrier<PIPE_V>();
                WholeReduceSum(coeff,partial,8,rows,1,1,1);PipeBarrier<PIPE_V>();
                Add(triangular[j*64],triangular[j*64],coeff,rows);PipeBarrier<PIPE_V>();
                Muls(coeff,coeff,-1.0f,rows);PipeBarrier<PIPE_V>();Brcb(weights,coeff,(uint8_t)((rows+7)/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(panel[(j+1)*256+q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)rows,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_ALL>();
            }
        }
        for(int32_t j=(int32_t)real-1;j>=0;--j) {
            float norm=triangular.GetValue(j*64+63);
            Muls(panel[j*256],panel[j*256],1.0f/norm,256);PipeBarrier<PIPE_V>();
            if(j) {
                for(int32_t r=0;r<j;++r)ix.SetValue(r,(r*64+j-r-1)*4);PipeBarrier<PIPE_ALL>();
                Gather(coeff,triangular,ix.ReinterpretCast<uint32_t>(),0,(uint32_t)j);PipeBarrier<PIPE_V>();
                Muls(coeff,coeff,-1.0f,(uint32_t)j);PipeBarrier<PIPE_V>();Brcb(weights,coeff,(uint8_t)((j+7)/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(panel[q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)j,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_V>();Muls(panel,panel,-1.0f,real*256);PipeBarrier<PIPE_ALL>();DataCopy(dst,panel,real*256);PipeBarrier<PIPE_ALL>();
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
        CreateVecIndex(ix,(int32_t)0,256);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)128,256);PipeBarrier<PIPE_ALL>();
        if(lane<k){DataCopy(input,solution,8192);PipeBarrier<PIPE_ALL>();}
        for(uint32_t c=lane;c<4;c+=teams) {
            if(c<k)Gather(w,input[c],ix.ReinterpretCast<uint32_t>(),0,256);else Duplicate(w,0.0f,256);PipeBarrier<PIPE_ALL>();
            float exponent=0;if(c<k)exponent=Split6<true>(w,d);else Duplicate(d.ReinterpretCast<int32_t>(),(int32_t)0,384);PipeBarrier<PIPE_ALL>();
            Duplicate(packed.ReinterpretCast<int32_t>(),(int32_t)0,3072);PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<6;++weight)DataCopy(packed[weight*1536],d[(5-weight)*256],(weight+1)*256);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(digits[c*1536],packed,DataCopyExtParams{8,1536,0,4608,0});PipeBarrier<PIPE_ALL>();
            e.SetValue(0,exponent);e.SetValue(8,0.0f);PipeBarrier<PIPE_ALL>();
            DataCopyPad(exponents[c],e,DataCopyExtParams{1,4,0,0,0});DataCopyPad(exponents[c+4],e[8],DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void Assemble6Packed4(GlobalTensor<float> dst,GlobalTensor<float> initial,GlobalTensor<float> reserve,GlobalTensor<float> next,GlobalTensor<int32_t> products,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> rhs) {
        constexpr uint32_t L=128;
        auto ae=w,xe=w[256],gs=w[512],high=w[1024],low=w[1536],term=w[2048],tmp=w[2560],u=w[3072],v=w[3584],factor=w[4096],ea=w[5120],ex=w[5632];auto packed=w[7000],allProducts=w[9000],rhsTile=w[10400];auto ix=w[4608].ReinterpretCast<int32_t>();
        DataCopy(ae,expA,256);DataCopy(xe,expX,16);PipeBarrier<PIPE_ALL>();
        CreateVecIndex(ix,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(ix,ix,(int32_t)2,L);PipeBarrier<PIPE_V>();ShiftLeft(ix,ix,(int32_t)2,L);PipeBarrier<PIPE_ALL>();
        auto index=w[11000].ReinterpretCast<int32_t>();auto columns=w[11504].ReinterpretCast<int32_t>();
        CreateVecIndex(index,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(columns,index,(int32_t)2,L);PipeBarrier<PIPE_V>();ShiftLeft(columns,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();Sub(columns,index,columns,L);PipeBarrier<PIPE_V>();
        ShiftLeft(index,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();Gather(ex,xe,index.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
        auto outputIndex=w[32000].ReinterpretCast<int32_t>(),outputRows=w[33024].ReinterpretCast<int32_t>(),rhsIndex=w[34048].ReinterpretCast<int32_t>();
        CreateVecIndex(rhsIndex,(int32_t)0,L);PipeBarrier<PIPE_V>();Add(rhsIndex,rhsIndex,ix,L);PipeBarrier<PIPE_V>();ShiftLeft(rhsIndex,rhsIndex,(int32_t)2,L);PipeBarrier<PIPE_V>();
        CreateVecIndex(outputIndex,(int32_t)0,1024);PipeBarrier<PIPE_V>();ShiftRight(outputRows,outputIndex,(int32_t)5,1024);PipeBarrier<PIPE_V>();
        Muls(outputRows,outputRows,(int32_t)28,1024);PipeBarrier<PIPE_V>();Sub(outputIndex,outputIndex,outputRows,1024);PipeBarrier<PIPE_V>();ShiftLeft(outputIndex,outputIndex,(int32_t)2,1024);PipeBarrier<PIPE_ALL>();
        uint64_t outputMask[2]={0x0000000f0000000fULL,0};
        for(uint32_t tile=lane;tile<8;tile+=teams) {
            uint32_t row=tile*32;Gather(ea,ae[row],ix.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Duplicate(low,0.0f,L);DataCopyPad(rhsTile,rhs[row*32],DataCopyExtParams{32,32,96,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();Gather(high,rhsTile,rhsIndex.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
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
            Add(high,high,low,L);PipeBarrier<PIPE_V>();Duplicate(packed,0.0f,1024);PipeBarrier<PIPE_V>();Gather(packed,high,outputIndex.ReinterpretCast<uint32_t>(),0,outputMask,16,8);PipeBarrier<PIPE_ALL>();DataCopy(dst[row*32],packed,1024);DataCopy(initial[row*32],packed,1024);DataCopy(reserve[row*32],packed,1024);DataCopy(next[row*32],packed,1024);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void AccurateResidual(GlobalTensor<float> dst,GlobalTensor<float> matrix,GlobalTensor<float> solution,GlobalTensor<float> rhs,GlobalTensor<float> mirror,GlobalTensor<float> reserve,GlobalTensor<float> next,uint32_t k) {
        AccurateRows<256,16> dot;dot.Init(w);
        auto packedX=w[45312],rhsTile=w[47648],result=w[48160];auto xix=w[47360].ReinterpretCast<uint32_t>(),bix=w[47616].ReinterpretCast<uint32_t>();
        DataCopyPad(packedX,solution,DataCopyExtParams{256,k*4,(32-k)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-k),0});for(uint32_t j=0;j<256;++j)xix.SetValue(j,j*8*4);PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            DataCopy(dot.raw,matrix[row*256],4096);DataCopy(rhsTile,rhs[row*32],512);Duplicate(result,0.0f,512);PipeBarrier<PIPE_ALL>();
            for(uint32_t c=0;c<k;++c) {
                Gather(dot.x,packedX,xix,c*4,256);PipeBarrier<PIPE_V>();
                for(uint32_t r=1;r<16;++r)DataCopy(dot.x[r*256],dot.x,256);PipeBarrier<PIPE_ALL>();dot.AccurateDot(16);
                for(uint32_t r=0;r<16;++r)bix.SetValue(r,(r*32+c)*4);PipeBarrier<PIPE_ALL>();Gather(dot.a,rhsTile,bix,0,16);PipeBarrier<PIPE_V>();
                dot.A(dot.h,dot.h,dot.a,16);dot.A(dot.h,dot.h,dot.l,16);PipeBarrier<PIPE_ALL>();
                for(uint32_t r=0;r<16;++r)result.SetValue(r*32+c,dot.h.GetValue(r));PipeBarrier<PIPE_ALL>();
            }
            DataCopy(dst[row*32],result,512);DataCopy(mirror[row*32],result,512);DataCopy(reserve[row*32],result,512);DataCopy(next[row*32],result,512);PipeBarrier<PIPE_ALL>();
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
    __aicore__ inline void Normalize(GlobalTensor<float> normalized,GlobalTensor<float> current,GlobalTensor<float> src,GlobalTensor<float> rhs,uint32_t k,uint32_t n) {
        uint32_t count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams) {
            Duplicate(w[count],0.0f,4096);PipeBarrier<PIPE_ALL>();
            const uint32_t row=tile*16,realRows=row<n?(n-row<16?n-row:16):0;
            if(realRows)DataCopyPad(w[count],src[row*n],DataCopyExtParams{(uint16_t)realRows,n*4,0,(256-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
            count+=4096;
        }
        PipeBarrier<PIPE_ALL>();
        // Every lane independently derives one scale from identical A/B data.
        Duplicate(w[8192],0.0f,2304);PipeBarrier<PIPE_ALL>();
        DataCopyPad(w[8192],src,DataCopyExtParams{1,n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        DataCopyPad(w[8448],rhs,DataCopyExtParams{1,n*k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
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
        count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams) {
            for(uint32_t r=0;r<16;++r)if(tile*16+r>=n)w.SetValue(count+r*256+tile*16+r,1.0f);
            PipeBarrier<PIPE_ALL>();
            DataCopy(normalized[tile*4096],w[count],4096);DataCopy(current[tile*4096],w[count],4096);count+=4096;
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Finish(GlobalTensor<float> dst,GlobalTensor<float> x,GlobalTensor<float> dx,uint32_t k,uint32_t n,GM_ADDR rr,bool hasResidual) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*32+c)*4);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<n;row+=teams*16) {
            DataCopy(w,x[row*32],512);DataCopy(w[4096],dx[row*32],512);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],512);PipeBarrier<PIPE_ALL>();
            Gather(w[8192],w,ix,0,16*k);PipeBarrier<PIPE_ALL>();DataCopyPad(dst[row*k],w[8192],DataCopyExtParams{1,(n-row<16?n-row:16)*k*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
        if(lane==0 && hasResidual) {
            GlobalTensor<float> gr;gr.SetGlobalBuffer((__gm__ float*)rr);
            Duplicate(w,0.0f,8);PipeBarrier<PIPE_ALL>();
            DataCopyPad(gr[(uint64_t)batch*k],w,DataCopyExtParams{1,k*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    template<uint32_t B=32,uint32_t S=64> __aicore__ inline void InvertSmall(GlobalTensor<float> dst,GlobalTensor<float> src,bool negative=false) {
        if(lane)return;
        constexpr uint32_t R=2*B+8;
        auto panel=w,pivot=w[B*R],col=pivot[64],weights=col[B];auto ix=weights[256].ReinterpretCast<int32_t>();
        auto recip=w[4160],minusOne=w[4224];
        Duplicate(minusOne,-1.0f,8);
        Duplicate(panel,0.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(panel,src,DataCopyExtParams{B,B*4,(S-B)*4,(R-B)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r){panel.SetValue(r*R+B+r,1.0f);ix.SetValue(r,r*R*4);}PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            Gather(col,panel[j],ix.ReinterpretCast<uint32_t>(),0,(uint64_t)B,1,8);PipeBarrier<PIPE_V>();
            Brcb(weights,col,B/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Div(recip,minusOne,weights[j*8],(uint64_t)8,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Mul(pivot,panel[j*R],recip,(uint64_t)(2*B),1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            MulAddDst(panel,pivot,weights,(uint64_t)(2*B),(uint8_t)B,BinaryRepeatParams{1,1,0,R/8,0,1});PipeBarrier<PIPE_V>();
            Muls(panel[j*R],pivot,-1.0f,(uint64_t)(2*B),1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        if(negative){Muls(panel,panel,-1.0f,B*R);PipeBarrier<PIPE_ALL>();}
        DataCopyPad(dst,panel[B],DataCopyExtParams{B,B*4,(R-B)/8,(S-B)*4,0});PipeBarrier<PIPE_ALL>();
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

    __aicore__ inline void NewtonGuard(GlobalTensor<float> dst,GlobalTensor<int32_t> decision,bool product=false) {
        auto ab=w[4096],sum=w[8192];
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            DataCopy(w,dst[row*64],512);PipeBarrier<PIPE_ALL>();
            if(product){Add(w,w,w[36864],512);PipeBarrier<PIPE_V>();}
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
template<bool Cube> __aicore__ __attribute__((noinline)) bool CheckRefined(
    std::conditional_t<Cube,NativeMM,Vector> &op,GlobalTensor<float> mem,
    GlobalTensor<int32_t> decisions,uint32_t batches,uint32_t giBase) {
    // The first decision remains immutable while other cores read it.
    auto actual=decisions[8];
    if constexpr(!Cube)op.ZeroDecision(actual);
    if constexpr(Cube)op.template Run<64,64,64>(mem[G],mem[giBase],mem[NT],batches,STRIDE,STRIDE,STRIDE);
    SyncAll<false>();
    if constexpr(!Cube)op.NewtonGuard(mem[NT],actual,true);
    SyncAll<false>();
    DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(actual);
    return actual.GetValue(0)==0;
}
template<bool Cube> __aicore__ inline void Tail(std::conditional_t<Cube,NativeMM,Vector> &op,
    GlobalTensor<float> mem,GlobalTensor<float> A,GlobalTensor<int8_t> ad,GM_ADDR workspace,
    uint32_t offset,uint32_t real,uint32_t n,uint32_t k,uint32_t batches) {
    constexpr uint32_t INV=D+32768,GRAM=D+33792,CROSS=D+34304,DEFECT=D+34560;
    bool split=false;
    if(real>16 && real<=32) {
        GlobalTensor<int32_t> decision;decision.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+224);
        if constexpr(!Cube)op.ZeroDecision(decision);
        if constexpr(Cube)op.template Run<16,32,256,false,true,false>(mem[CT+offset*256],mem[CT+offset*256],mem[GRAM],batches,STRIDE,STRIDE,STRIDE,256,256,32);
        SyncAll<false>();
        if constexpr(!Cube)op.template InvertSmall<16,32>(mem[INV],mem[GRAM],true);
        SyncAll<false>();
        if constexpr(Cube)op.template Run<16,16,16,false,false,false>(mem[GRAM],mem[INV],mem[DEFECT],batches,STRIDE,STRIDE,STRIDE,32,32,16);
        SyncAll<false>();
        if constexpr(!Cube)op.TailGuard16(mem[DEFECT],decision);
        SyncAll<false>();
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decision);
        split=decision.GetValue(0)==0;
        if(split) {
            if constexpr(Cube)op.template Run<16,16,16,false,false,false>(mem[INV],mem[GRAM+16],mem[CROSS],batches,STRIDE,STRIDE,STRIDE,32,32,16);
            if constexpr(Cube)SyncCube();
            if constexpr(Cube)op.template Run<16,256,16,true,false,true>(mem[CROSS],mem[CT+offset*256],mem[CT+(offset+16)*256],batches,STRIDE,STRIDE,STRIDE,16,256,256);
            SyncAll<false>();
        }
    }
    if constexpr(!Cube) {
        op.TailQR(mem[CT+(offset+(split?16:0))*256],mem[CT+(offset+(split?16:0))*256],real-(split?16:0));
        if(split)op.PackTail16(mem[INV],mem[CROSS]);
        if(k<=4)for(uint32_t panel=EarlyPackingChunks(n)*32;panel<160;panel+=32)op.SplitAChunk(ad,mem[EA],A,panel,n);
    }
    SyncAll<false>();
    if(split) {
        if constexpr(Cube)op.template Run<16,256,32,false,false,false>(mem[INV],mem[CT+offset*256],mem[CT+offset*256],batches,STRIDE,STRIDE,STRIDE,32,256,256);
        SyncAll<false>();
    }
}
template<bool Cube> __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgLstsqTilingData &t,GM_ADDR rr) {
    const uint32_t batches=t.batches,k=t.k,n=t.n;
    GlobalTensor<float> mem,A,B,X;
    mem.SetGlobalBuffer((__gm__ float*)workspace);A.SetGlobalBuffer((__gm__ float*)aa);B.SetGlobalBuffer((__gm__ float*)bb);X.SetGlobalBuffer((__gm__ float*)xx);
    std::conditional_t<Cube,NativeMM,Vector> op;
    uint32_t batch=0;
    if constexpr(Cube)op.Init();else {op.Init(batches);batch=op.batch;mem=mem[(uint64_t)batch*STRIDE];A=A[(uint64_t)batch*n*n];B=B[(uint64_t)batch*n*k];X=X[(uint64_t)batch*n*k];}
    GlobalTensor<half> halfMem;halfMem.SetGlobalBuffer((__gm__ half*)workspace+((uint64_t)batch*STRIDE+NH)*2);
    GlobalTensor<int8_t> ad,xd;GlobalTensor<int32_t> products;
    ad.SetGlobalBuffer((__gm__ int8_t*)workspace+((uint64_t)batch*STRIDE+AD)*4);
    xd.SetGlobalBuffer((__gm__ int8_t*)workspace+((uint64_t)batch*STRIDE+XD)*4);
    products.SetGlobalBuffer((__gm__ int32_t*)workspace+(uint64_t)batch*STRIDE+PRODUCTS);
    if constexpr(!Cube){op.Normalize(mem[AN],mem[CT],A,B,k,n);op.PackCachedB(mem[BP],mem[DX],k);}
    SyncAll<false>();
    const uint32_t lastPanel=(n-1)/64*64;
    // The fixed capacity has at most three regular panels followed by one real tail.
    for(uint32_t p=0;p<lastPanel && p<192;p+=64) {
        const uint32_t giBase=D+p*64;
        if constexpr(Cube)op.template Run<64,64,256,false,true>(mem[CT+p*256],mem[CT+p*256],mem[G],batches,STRIDE,STRIDE,STRIDE);
        SyncAll<false>();
        bool useExact=false;
        {
            GlobalTensor<int32_t> decisions;decisions.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+128+(p/64)*32);
            if constexpr(!Cube)op.ZeroDecision(decisions);
            if constexpr(!Cube)op.NewtonPrepare(halfMem,mem[G]);
            SyncAll<false>();
            if constexpr(Cube){// Narrower remaining row space makes the Gram spectrum less favorable.
            // This is an iteration budget only; the same defect guard decides use.
            const uint32_t remaining=n-p,steps=remaining>=256?6:(remaining>=192?7:(remaining>=128?9:13));
            op.Newton64(halfMem,mem[giBase],mem[NT],batches,STRIDE,steps);op.CrossGramImplicit256(mem[CT],mem[D+16384],batches,STRIDE,p);}
            if constexpr(!Cube)if(k<=4) {
                const uint32_t first=(n-1)/64-EarlyPackingChunks(n);
                if(p/64>=first)op.SplitAChunk(ad,mem[EA],A,(p/64-first)*32,n);
            }
            SyncAll<false>();
            if constexpr(!Cube)op.NewtonGuard(mem[NT],decisions);
            if constexpr(Cube)op.HImplicit256(mem[giBase],mem[D+16384],mem[H],batches,STRIDE,p);
            SyncAll<false>();
            DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decisions);
            useExact=decisions.GetValue(0)!=0;
            // The first guard precedes the final refinement. Validate the
            // returned inverse with the same bound before exact elimination.
            if(useExact)useExact=!CheckRefined<Cube>(op,mem,decisions,batches,giBase);
        }
        if(useExact) {
        if constexpr(!Cube){op.template InvertSmall<>(mem[giBase],mem[G]);op.NegateBlock(mem[G+2048]);}
        SyncAll<false>();
        if constexpr(Cube)op.Schur32Parallel(mem[giBase],mem[G],mem[Q32],batches,STRIDE);
        SyncAll<false>();
        if constexpr(!Cube){op.template InvertSmall<>(mem[giBase+2080],mem[G+2080],true);op.NegateBlock(mem[G+32]);op.NegateBlock(mem[giBase],2);}
        SyncAll<false>();
        if constexpr(Cube)op.Assemble32Parallel(mem[giBase],mem[G],mem[Q32],batches,STRIDE);
        if constexpr(Cube)SyncCube();
        if constexpr(Cube){op.HImplicit256(mem[giBase],mem[D+16384],mem[H],batches,STRIDE,p);SyncCube();}
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
    {
        const uint32_t p=lastPanel;
            if constexpr(Cube)op.ReprojectTail(mem[CT],mem[D],mem[H],mem[D+28672],mem[D+16384],batches,STRIDE,p);
            SyncAll<false>();
            const uint32_t real=n-p;
            if(real>32) {
                GlobalTensor<int32_t> decision;decision.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+224);
                if constexpr(!Cube)op.ZeroDecision(decision);
                if constexpr(Cube)op.TailProducts32(mem[CT+p*256],mem[G],mem[Q32],batches,STRIDE);
                SyncAll<false>();
                if constexpr(!Cube)op.template InvertSmall<>(mem[D+12288],mem[G],true);
                if constexpr(Cube)op.ComposeForward(mem[H],batches,STRIDE,p);
                SyncAll<false>();
                if constexpr(Cube)op.template Run<32,32,32,false,false,false>(mem[G],mem[D+12288],mem[NT],batches,STRIDE,STRIDE,STRIDE,64,64,32);
                SyncAll<false>();
                if constexpr(!Cube)op.TailGuard32(mem[NT],decision);
                SyncAll<false>();
                DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decision);
                const bool fallback=decision.GetValue(0)!=0;
                if(fallback) {
                    Tail<Cube>(op,mem,A,ad,workspace,p,real,n,k,batches);
                } else {
                    if constexpr(Cube)op.template Run<32,32,32,false,false,false>(mem[D+12288],mem[Q32],mem[Q32],batches,STRIDE,STRIDE,STRIDE,64,32,32);
                    if constexpr(Cube)SyncCube();
                    if constexpr(Cube)op.template Run<32,256,32,true,false,true>(mem[Q32],mem[CT+p*256],mem[CT+(p+32)*256],batches,STRIDE,STRIDE,STRIDE,32,256,256);
                    SyncAll<false>();
                    Tail<Cube>(op,mem,A,ad,workspace,p+32,real-32,n,k,batches);
                    if constexpr(!Cube)op.PackTail32(mem[D+12288],mem[Q32]);
                    SyncAll<false>();
                    // [negative inverse, negative projection] applies [W0; negative tail factor].
                    // Each output tile reads and overwrites only its own columns of W0.
                    if constexpr(Cube)op.template Run<32,256,64,false,false,false>(mem[D+12288],mem[CT+p*256],mem[CT+p*256],batches,STRIDE,STRIDE,STRIDE,64,256,256);
                    if constexpr(Cube)SyncCube();
                }
            } else {
                if constexpr(Cube)op.ComposeForward(mem[H],batches,STRIDE,p);
                Tail<Cube>(op,mem,A,ad,workspace,p,real,n,k,batches);
            }

            }
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[BP],mem[DX],batches,STRIDE,(n+63)/64);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true>(mem[CT],mem[DX],mem[XP],batches,STRIDE,STRIDE,STRIDE);
    if constexpr(Cube)SyncCube();
    constexpr uint32_t R0=D+32768,R1=D+49152;
    SyncAll<false>();
    if(k<=4) {
        if constexpr(!Cube)op.SplitXPacked4(xd,mem[EX],mem[XP],k);
        SyncAll<false>();
        if constexpr(Cube)op.Int8Packed4(ad,xd,products,batches,STRIDE);
        SyncAll<false>();
        if constexpr(!Cube)op.Assemble6Packed4(mem[R],mem[DX],mem[R0],mem[R1],products,mem[EA],mem[EX],mem[BP]);
    } else {
        if constexpr(!Cube)op.AccurateResidual(mem[R],mem[AN],mem[XP],mem[BP],mem[DX],mem[R0],mem[R1],k);
    }
    SyncAll<false>();
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[R],mem[DX],batches,STRIDE,(n+63)/64);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true>(mem[CT],mem[DX],mem[R],batches,STRIDE,STRIDE,STRIDE);
    if constexpr(Cube)SyncCube();
    // Both copies retain q0 while the first forward solve consumes DX.
    // One product updates immutable input R0 and the next forward destination R1.
    if constexpr(Cube)op.template Run<256,32,256,false,false,true>(mem[AN],mem[R],mem[R0],batches,STRIDE,STRIDE,STRIDE,256,32,32,R1-R0);
    if constexpr(Cube)SyncCube();
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[R0],mem[R1],batches,STRIDE,(n+63)/64);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true,false,true>(mem[CT],mem[R1],mem[R],batches,STRIDE,STRIDE,STRIDE);
    SyncAll<false>();
    if constexpr(!Cube)op.Finish(X,mem[XP],mem[R],k,n,rr,rr && t.residualElements>=(uint64_t)batches*k);
}
}

// Row Gram panels, adapted from solve bfdd2d4, with accurate residual refinement.
namespace NarrowOrthogonal {
constexpr uint32_t CT=0,D=65536,H=131072,AN=196608,G=262144,GI=278528,
 BP=294912,XP=303104,R=311296,DX=319488,REDUCE=327680,AD=327936,XD=426240,PRODUCTS=447744,EA=472320,EX=472576,NT=472832,NH=489216,CROSS=538368,STRIDE=kNarrowOrthogonalWorkspaceFloats;
static_assert(NH+49152==CROSS && CROSS+256*128<=STRIDE);
// Five operand digits still feed all six retained convolution coefficients.
constexpr uint32_t kOperandDigits=5,kResidualWidth=256*kOperandDigits;
template<uint32_t Panel> class NativeMM {
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
        uint32_t K,uint32_t as,uint32_t bs,uint32_t cs,uint32_t skip=0,uint32_t mirrorOffset=0) {
        static_assert(BM*BK*sizeof(float)<=65536 && BN*BK*sizeof(float)<=65536);
        static_assert(BM*BN*sizeof(float)<=131072);
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
        Fixpipe<float,float,CFG_ROW_MAJOR>(C[skip],c[skip*BM],f);
        if(mirrorOffset)Fixpipe<float,float,CFG_ROW_MAJOR>(C[mirrorOffset+skip],c[skip*BM],f);
        PipeBarrier<PIPE_ALL>();
        if constexpr(Accumulate)SetAtomicNone();
    }
    template<uint32_t M,uint32_t N,uint32_t K,bool TA=false,bool TB=false,bool Accumulate=false>
    __aicore__ inline void Run(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,
        uint32_t batches,uint32_t astride,uint32_t bstride,uint32_t cstride,
        uint32_t as=TA?M:K,uint32_t bs=TB?K:N,uint32_t cs=N,uint32_t mirrorOffset=0) {
        constexpr uint32_t BM=(M==128&&N>=128)?64:(M>=128?((M==192 || (N<=32 && K>=128))?64:128):(M==64 && N>=128?64:(M>=32?32:16)));
        constexpr uint32_t BN=N>=128?(M<=128?64:128):(N>=32?32:16);
        constexpr uint32_t BK=K==192 && BM<=64 && BN<=64?192:(K>=256 && BM<=64 && BN<=64?256:(K>=128?128:(K>=64?64:(K>=32?32:16))));
        constexpr uint32_t nr=M/BM,nc=N/BN;
        for(uint32_t task=GetBlockIdx();task<batches*nr*nc;task+=GetBlockNum()) {
            uint32_t batch=task/(nr*nc),r=task/nc%nr*BM,col=task%nc*BN;
            Tile<BM,BN,BK,TA,TB,Accumulate>(A[batch*astride+(TA?r:r*as)],B[batch*bstride+(TB?col*bs:col)],C[batch*cstride+r*cs+col],K,as,bs,cs,0,mirrorOffset);
        }
    }
    __aicore__ inline void Int8Packed4(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t batches,uint32_t stride) {
        constexpr uint32_t Rows=64,First=48,BN=48,K=kResidualWidth;
        static_assert(First*K<=65536 && BN*K<=65536);
        static_assert(Rows*K<=131072 && 131072+BN*K<=262144);
        auto la=l1buf.Get<int8_t>(),lb=la[131072],a=abuf.Get<int8_t>(),b=bbuf.Get<int8_t>();auto c=cbuf.Get<int32_t>();
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4*Rows;PipeBarrier<PIPE_ALL>();
            Nd2NzParams p{};p.ndNum=1;p.nValue=Rows;p.dValue=K;p.srcDValue=K;p.dstNzC0Stride=Rows;p.dstNzNStride=1;
            DataCopy(la,A[batch*stride*4+row*K],p);
            p.nValue=BN;p.dstNzC0Stride=BN;DataCopy(lb,B[batch*stride*4],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams right{};right.repeatTimes=K*BN/512;right.srcStride=1;LoadData(b,lb,right);
            // Both row groups reuse the complete right operand in L0B.
            for(uint32_t part=0;part<2;++part) {
                const uint32_t first=part?First:0,rows=part?Rows-First:First;
                LoadData2DParams q{};q.repeatTimes=K/32;q.srcStride=Rows/16;
                for(uint32_t r=0;r<rows/16;++r)LoadData(a[r*K*16],la[(first/16+r)*512],q);
                SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
                MmadParams m{};m.m=rows;m.n=BN;m.k=K;m.cmatrixInitVal=true;Mmad(c,a,b,m);
                SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);FixpipeParamsV220 f{};f.mSize=rows;f.nSize=BN;f.srcStride=rows;f.dstStride=BN;
                Fixpipe<int32_t,int32_t,CFG_ROW_MAJOR>(C[batch*stride+(row+first)*BN],c,f);PipeBarrier<PIPE_ALL>();
            }
        }
    }
    template<uint32_t N,bool Add=false>
    __aicore__ inline void LocalRun(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum())Tile<N,N,N,false,false,Add>(A[batch*stride],B[batch*stride],C[batch*stride],N,128,128,128);
    }    template<bool Add=false>
    __aicore__ inline void ProductPanel(LocalTensor<half> la,LocalTensor<half> lb,LocalTensor<float> c) {
        PipeBarrier<PIPE_ALL>();auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        uint8_t pad[4]={};SetFmatrix(1,Panel,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=Panel/16;q.srcStride=Panel/16;
        for(uint32_t r=0;r<Panel/16;++r)LoadData(a[r*(16*Panel)],la[r*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=Panel;z.extConfig=((uint64_t)Panel<<16)|Panel;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=Panel;m.n=Panel;m.k=Panel;m.cmatrixInitVal=!Add;Mmad(c,a,b,m);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void StorePanel(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=Panel;f.nSize=Panel;f.srcStride=Panel;f.dstStride=Panel;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void ReplicatedNeumannPrefix(GlobalTensor<half> packed,
        GlobalTensor<half> output,uint32_t stride,uint32_t steps) {
        static_assert(Panel==96 || Panel==112);
        const uint32_t batch=GetBlockIdx()/4,lane=GetBlockIdx()%4,row=lane*32;
        if(row<Panel) {
            const uint32_t rows=Panel-row<32?Panel-row:32;
            auto g=l1buf.Get<half>(),x=g[2*Panel*Panel],eye=g[3*Panel*Panel],r=g[5*Panel*Panel];
            auto a=abuf.Get<half>(),b=bbuf.Get<half>();auto y=cbuf.Get<float>(),tmp=y[Panel*Panel];
            auto src=packed[batch*stride*2];
            Nd2NzParams p{};p.ndNum=1;p.nValue=Panel;p.dValue=Panel;p.srcDValue=Panel;p.dstNzC0Stride=Panel;p.dstNzNStride=1;
            DataCopy(eye,src[2*Panel*Panel],p);DataCopy(r,src[5*Panel*Panel],p);
            p.nValue=rows;p.dstNzC0Stride=rows;DataCopy(x,src[Panel*Panel+row*Panel],p);PipeBarrier<PIPE_ALL>();
            uint8_t pad[4]={};SetFmatrix(1,Panel,pad,FmatrixMode::FMATRIX_LEFT);
            LoadData2DParams left{};left.repeatTimes=Panel/16;left.srcStride=rows/16;
            for(uint32_t q=0;q<rows/16;++q)LoadData(a[q*Panel*16],x[q*256],left);
            LoadData3DParamsV2Pro right{};right.channelSize=Panel;right.extConfig=((uint64_t)Panel<<16)|Panel;LoadData<half>(b,eye,right);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=rows;m.n=Panel;m.k=Panel;m.cmatrixInitVal=true;Mmad(y,a,b,m);PipeBarrier<PIPE_ALL>();
            for(uint32_t iter=0;iter+1<steps;++iter) {
                left.srcStride=rows/16;
                for(uint32_t q=0;q<rows/16;++q)LoadData(a[q*Panel*16],x[q*256],left);
                LoadData<half>(b,r,right);SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
                m.m=rows;m.cmatrixInitVal=false;Mmad(y,a,b,m);SetFlag<HardEvent::M_FIX>(0);
                const bool square=iter+2<steps;
                if(square) {
                    left.srcStride=Panel/16;
                    for(uint32_t q=0;q<Panel/16;++q)LoadData(a[Panel*Panel+q*Panel*16],r[q*256],left);
                    SetFlag<HardEvent::MTE1_M>(1);WaitFlag<HardEvent::MTE1_M>(1);
                    m.m=Panel;m.cmatrixInitVal=true;Mmad(tmp,a[Panel*Panel],b,m);SetFlag<HardEvent::M_FIX>(1);
                }
                WaitFlag<HardEvent::M_FIX>(0);
                FixpipeParamsV220 local{};local.mSize=rows;local.nSize=Panel;local.srcStride=rows;local.dstStride=rows;local.quantPre=QuantMode_t::F322F16;
                Fixpipe<half,float,CFG_NZ>(x,y,local);
                if(square) {
                    WaitFlag<HardEvent::M_FIX>(1);
                    FixpipeParamsV220 retain{};retain.mSize=Panel;retain.nSize=Panel;retain.srcStride=Panel;retain.dstStride=Panel;retain.quantPre=QuantMode_t::F322F16;
                    Fixpipe<half,float,CFG_NZ>(r,tmp,retain);
                }
                PipeBarrier<PIPE_ALL>();
            }
            // Publish only the owned X rows once. NT is not a defect buffer yet.
            FixpipeParamsV220 f{};f.mSize=rows;f.nSize=Panel;f.srcStride=rows;f.dstStride=Panel;f.quantPre=QuantMode_t::F322F16;
            Fixpipe<half,float,CFG_ROW_MAJOR>(output[batch*stride*2+row*Panel],y,f);PipeBarrier<PIPE_ALL>();
        }
        // For Panel=96, lane3 owns no rows but joins the same rendezvous.
        SyncCube();
    }
    template<bool Second=false>
    __aicore__ inline void NewtonPanel(GlobalTensor<half> packed,GlobalTensor<float> result,GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        GlobalTensor<half> prefix;prefix.SetGlobalBuffer((__gm__ half*)defect.GetPhyAddr());
        ReplicatedNeumannPrefix(packed,prefix,stride,steps);
        if(GetBlockIdx()%4)return;
        const uint32_t batch=GetBlockIdx()/4;
        auto g=l1buf.Get<half>(),low=g[Panel*Panel],x=g[2*Panel*Panel],eye=g[3*Panel*Panel],two=g[4*Panel*Panel],r=g[5*Panel*Panel],minus=g[6*Panel*Panel];
        auto y=cbuf.Get<float>(),tmp=y[Panel*Panel];
        Nd2NzParams p{};p.ndNum=1;p.nValue=Panel;p.dValue=Panel;p.srcDValue=Panel;p.dstNzC0Stride=Panel;p.dstNzNStride=1;
        auto src=packed[batch*stride*2];
        DataCopy(g,src,p);DataCopy(low,src[4*Panel*Panel],p);DataCopy(x,prefix[batch*stride*2],p);DataCopy(two,src[3*Panel*Panel],p);
        if constexpr(Second)DataCopy(minus,src[6*Panel*Panel],p);PipeBarrier<PIPE_ALL>();
        ProductPanel(eye,two,tmp);ProductPanel<true>(g,x,tmp);StorePanel(r,tmp);
        ProductPanel(x,r,y);StorePanel(x,y);
        ProductPanel(eye,eye,tmp);ProductPanel<true>(g,x,tmp);ProductPanel<true>(low,x,tmp);StorePanel(r,tmp);
        ProductPanel(x,eye,y);ProductPanel<true>(x,r,y);
        if constexpr(Second) {
            StorePanel(x,y);ProductPanel<true>(x,minus,y);StorePanel(two,y);
            ProductPanel(eye,eye,tmp);ProductPanel<true>(g,x,tmp);ProductPanel<true>(low,x,tmp);ProductPanel<true>(g,two,tmp);ProductPanel<true>(low,two,tmp);StorePanel(r,tmp);
            ProductPanel(x,eye,y);ProductPanel<true>(two,eye,y);ProductPanel<true>(x,r,y);ProductPanel<true>(two,r,y);
        }
        FixpipeParamsV220 f{};f.mSize=Panel;f.nSize=Panel;f.srcStride=Panel;f.dstStride=128;
        Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
    }

    template<uint32_t Active=256>
    __aicore__ inline void GramPanel(GlobalTensor<float> ct,GlobalTensor<float> g,uint32_t batches,uint32_t stride,uint32_t panel) {
        if constexpr(Panel==96) {

            for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
                uint32_t batch=task/4,row=task%4/2*48,col=task%2*48;
                Tile<48,48,Active,false,true,false>(ct[batch*stride+(panel+row)*256],ct[batch*stride+(panel+col)*256],g[batch*stride+row*128+col],Active,256,256,128);
            }

        } else {

            static_assert(Active==256);
            for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
                uint32_t batch=task/4,row=task%4<2?0:64,col=task%2?64:0;
                auto A=ct[batch*stride+(panel+row)*256],B=ct[batch*stride+(panel+col)*256],C=g[batch*stride+row*128+col];
                if(row==0) {if(col==0)Tile<64,64,256,false,true,false>(A,B,C,256,256,256,128);else Tile<64,48,256,false,true,false>(A,B,C,256,256,256,128);}
                else {if(col==0)Tile<48,64,256,false,true,false>(A,B,C,256,256,256,128);else Tile<48,48,256,false,true,false>(A,B,C,256,256,256,128);}
            }
        }
    }
    __aicore__ inline void FactorPanel(GlobalTensor<float> gi,GlobalTensor<float> ct,GlobalTensor<float> d,uint32_t batches,uint32_t stride,uint32_t panel) {
        if constexpr(Panel==96) {

            for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
                uint32_t batch=task/4,row=task%4/2*48,col=task%2*128;
                Tile<48,128,96,false,false,false>(gi[batch*stride+row*128],ct[batch*stride+panel*256+col],d[batch*stride+(panel+row)*256+col],96,128,256,256);
            }

        } else {

            for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
                uint32_t batch=task/4,row=task%4<2?0:64,col=task%2*128;
                auto A=gi[batch*stride+row*128],B=ct[batch*stride+panel*256+col],C=d[batch*stride+(panel+row)*256+col];
                if(row==0)Tile<64,128,112,false,false,false>(A,B,C,112,128,256,256);
                else Tile<48,128,112,false,false,false>(A,B,C,112,128,256,256);
            }
        }
    }
    template<uint32_t Active=256>
    __aicore__ inline void UpdatePanel(GlobalTensor<float> h,GlobalTensor<float> ct,uint32_t batches,uint32_t stride,uint32_t panel) {
        if constexpr(Panel==96) {

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

        } else {

            static_assert(Active==256);
            for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
                uint32_t batch=task/4,col=task%2*128;
                if(panel==0) {
                    if(task%4<2)Tile<80,128,112,true,false,true>(h[batch*stride+112],ct[batch*stride+col],ct[batch*stride+112*256+col],112,256,256,256);
                    else Tile<64,128,112,true,false,true>(h[batch*stride+192],ct[batch*stride+col],ct[batch*stride+192*256+col],112,256,256,256);
                } else {
                    uint32_t row=224+task%4/2*16;
                    Tile<16,128,112,true,false,true>(h[batch*stride+panel*256+row],ct[batch*stride+panel*256+col],ct[batch*stride+row*256+col],112,256,256,256);
                }
            }
        }
    }
    __aicore__ __attribute__((noinline)) void ComposeFactorTileImpl(GlobalTensor<float> left,GlobalTensor<float> right,
        GlobalTensor<float> dst,uint32_t BM,uint32_t K,uint32_t BN) {
        auto la=l1buf.Get<float>(),lb=la[16384],a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        for(uint32_t at=0;at<K;at+=128) {
            const uint32_t chunk=K-at<128?K-at:128;
            PipeBarrier<PIPE_ALL>();uint8_t pad[4]={};SetFmatrix(1,chunk,pad,FmatrixMode::FMATRIX_LEFT);
            Nd2NzParams p{};p.ndNum=1;p.nValue=BM;p.dValue=chunk;p.srcDValue=256;p.dstNzC0Stride=BM;p.dstNzNStride=1;
            DataCopy(la,left[at],p);
            p.nValue=chunk;p.dValue=BN;p.dstNzC0Stride=chunk;DataCopy(lb,right[at*256],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams q{};q.repeatTimes=chunk/8;q.srcStride=BM/16;
            for(uint32_t row=0;row<BM/16;++row)LoadData(a[row*chunk*16],la[row*128],q);
            LoadData3DParamsV2Pro z{};z.channelSize=BN;z.extConfig=((uint64_t)chunk<<16)|BN;LoadData<float>(b,lb,z);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=BM;m.n=BN;m.k=chunk;m.cmatrixInitVal=at==0;Mmad(c,a,b,m);
        }
        SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN;f.srcStride=BM;f.dstStride=256;
        SetAtomicAdd<float>();Fixpipe<float,float,CFG_ROW_MAJOR>(dst,c,f);PipeBarrier<PIPE_ALL>();SetAtomicNone();
    }
    template<uint32_t BM,uint32_t K,uint32_t BN=128>
    __aicore__ inline void ComposeFactorTile(GlobalTensor<float> left,GlobalTensor<float> right,GlobalTensor<float> dst) {
        constexpr uint32_t MaxK=K<128?K:128;
        static_assert(BM*MaxK*sizeof(float)<=65536 && BN*MaxK*sizeof(float)<=65536);
        static_assert(BM*BN*sizeof(float)<=131072 && K%16==0);
        ComposeFactorTileImpl(left,right,dst,BM,K,BN);
    }
    template<uint32_t BM>
    __aicore__ inline void PrepareFactorTile(GlobalTensor<float> H,GlobalTensor<float> factor,uint32_t row,uint32_t col) {
        ComposeFactorTile<BM,Panel>(H[row*256+Panel],factor[Panel*256+col],factor[row*256+col]);
        if(col==0)ComposeFactorTile<BM,Panel,256-2*Panel>(H[row*256+Panel],H[Panel*256+2*Panel],H[row*256+2*Panel]);
    }
    __aicore__ inline void PrepareFactor(GlobalTensor<float> H,GlobalTensor<float> factor,uint32_t batches,uint32_t stride) {
        // D0 += H01*D1 and H02 += H01*H12 do not depend on the tail factor.
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            const uint32_t batch=task/4,col=task%2*128,row=(task%4/2)*(Panel==96?48:64);
            auto h=H[batch*stride],d=factor[batch*stride];
            if constexpr(Panel==96)PrepareFactorTile<48>(h,d,row,col);
            else {if(row==0)PrepareFactorTile<64>(h,d,row,col);else PrepareFactorTile<48>(h,d,row,col);}
        }
    }
    __aicore__ inline void CompleteFactor(GlobalTensor<float> H,GlobalTensor<float> factor,uint32_t batches,uint32_t stride) {
        // The two row blocks now depend only on the completed tail, not each other.
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            const uint32_t batch=task/4,col=task%2*128,localRow=(task%4/2)*(Panel==96?48:64);
            for(uint32_t block=0;block<2;++block) {
                const uint32_t row=block*Panel+localRow;
                auto left=H[batch*stride+row*256+2*Panel],right=factor[batch*stride+2*Panel*256+col],dst=factor[batch*stride+row*256+col];
                if constexpr(Panel==96)ComposeFactorTile<48,256-2*Panel>(left,right,dst);
                else {if(localRow==0)ComposeFactorTile<64,256-2*Panel>(left,right,dst);else ComposeFactorTile<48,256-2*Panel>(left,right,dst);}
            }
        }
    }
    __aicore__ inline void CrossGramFuture256(GlobalTensor<float> ct,GlobalTensor<float> cross,uint32_t batches,uint32_t stride,uint32_t panel) {
        const uint32_t id=GetBlockIdx();if(id%4==0)return;
        const uint32_t worker=(id/4)*3+id%4-1,workers=GetBlockNum()-batches;
        if constexpr(Panel==96) {

            if(panel==0) {
                for(uint32_t task=worker;task<batches*6;task+=workers) {
                    uint32_t batch=task/6,future=96+task%6/3*80,row=task%3*32;
                    Tile<80,32,128,false,true,false>(ct[batch*stride+future*256],ct[batch*stride+row*256],cross[batch*stride+future*128+row],256,256,256,128);
                }
            } else {
                for(uint32_t task=worker;task<batches*3;task+=workers) {
                    uint32_t batch=task/3,row=task%3*32;
                    Tile<64,32,256,false,true,false>(ct[batch*stride+192*256],ct[batch*stride+(panel+row)*256],cross[batch*stride+192*128+row],256,256,256,128);
                }
            }

        } else {

            if(panel==0) {
                for(uint32_t task=worker;task<batches*8;task+=workers) {
                    uint32_t batch=task/8,part=task%8/4,row=task%4*32,future=112+part*80;
                    auto A=ct[batch*stride+future*256],B=ct[batch*stride+row*256],C=cross[batch*stride+future*128+row];
                    if(part==0) {if(row<96)Tile<80,32,128,false,true,false>(A,B,C,256,256,256,128);else Tile<80,16,128,false,true,false>(A,B,C,256,256,256,128);}
                    else {if(row<96)Tile<64,32,256,false,true,false>(A,B,C,256,256,256,128);else Tile<64,16,256,false,true,false>(A,B,C,256,256,256,128);}
                }
            } else {
                for(uint32_t task=worker;task<batches*4;task+=workers) {
                    uint32_t batch=task/4,row=task%4*32;
                    auto A=ct[batch*stride+224*256],B=ct[batch*stride+(panel+row)*256],C=cross[batch*stride+224*128+row];
                    if(row<96)Tile<32,32,256,false,true,false>(A,B,C,256,256,256,128);else Tile<32,16,256,false,true,false>(A,B,C,256,256,256,128);
                }
            }
        }
    }
    template<uint32_t N,uint32_t Rows=48>
    __aicore__ inline void HWithResidentInverse(GlobalTensor<float> cross,GlobalTensor<float> h) {
        // FactorPanel leaves this core's inverse row tile in L0A.
        auto lb=l1buf.Get<float>()[16384],a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        PipeBarrier<PIPE_ALL>();
        Nd2NzParams p{};p.ndNum=1;p.nValue=N;p.dValue=Panel;p.srcDValue=128;p.dstNzC0Stride=N;p.dstNzNStride=1;
        DataCopy(lb,cross,p);
        SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData2DParams q{};q.repeatTimes=Panel*N/128;q.srcStride=1;LoadData(b,lb,q);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=Rows;m.n=N;m.k=Panel;m.cmatrixInitVal=true;Mmad(c,a,b,m);
        SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=Rows;f.nSize=N;f.srcStride=Rows;f.dstStride=256;
        Fixpipe<float,float,CFG_ROW_MAJOR>(h,c,f);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void HFromCross256(GlobalTensor<float> cross,GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        if constexpr(Panel==96) {

            // Identical task ownership to FactorPanel; no other MMA may intervene.
            for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
                uint32_t batch=task/4,row=task%4/2*48;
                if(panel==0) {
                    uint32_t col=96+task%2*80;
                    HWithResidentInverse<80>(cross[batch*stride+col*128],h[batch*stride+row*256+col]);
                } else {
                    uint32_t col=192+task%2*32;
                    HWithResidentInverse<32>(cross[batch*stride+col*128],h[batch*stride+(panel+row)*256+col]);
                }
            }

        } else {

            for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
                uint32_t batch=task/4,row=task%4<2?0:64,col=panel==0?(112+task%2*80):(224+task%2*16);
                auto A=cross[batch*stride+col*128],C=h[batch*stride+(panel+row)*256+col];
                if(panel==0) {
                    if(row==0) {if(task%2==0)HWithResidentInverse<80,64>(A,C);else HWithResidentInverse<64,64>(A,C);}
                    else {if(task%2==0)HWithResidentInverse<80,48>(A,C);else HWithResidentInverse<64,48>(A,C);}
                } else {if(row==0)HWithResidentInverse<16,64>(A,C);else HWithResidentInverse<16,48>(A,C);}
            }
        }
    }
    __aicore__ inline void ApplyPackedTail(GlobalTensor<float> mem,uint32_t offset,uint32_t batches) {
        GlobalTensor<int32_t> ready;ready.SetGlobalBuffer((__gm__ int32_t*)mem[REDUCE+96].GetPhyAddr());
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            const uint32_t batch=task/4,col=task%4*64;
            auto decision=ready[batch*STRIDE];
            DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decision);
            if(decision.GetValue(0)==0)continue;
            // Each task publishes a disjoint set of final factor columns.
            Tile<32,64,32,false,false,false>(mem[batch*STRIDE+GI],mem[batch*STRIDE+CT+offset*256+col],mem[batch*STRIDE+D+offset*256+col],32,32,256,256);
        }
    }

};
template<uint32_t Panel> class Vector {
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
    LocalTensor<float> w;
    uint32_t lane,teams;float inputScale=1.0f;
public:
    uint32_t batch;
    __aicore__ inline void Init(uint32_t batches) {
        uint32_t id=GetBlockIdx();batch=id%batches;lane=id/batches;teams=(2*GetBlockNum()+batches-1-batch)/batches;
        pipe.InitBuffer(buf,160*1024);w=buf.Get<float>();
        if(lane<Panel/16) {
            auto identity=w[36864];Duplicate(identity,0.0f,16*Panel);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<16;++j)identity.SetValue(j*Panel+lane*16+j,1.0f);
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
    __aicore__ inline void Normalize(GlobalTensor<float> normalized,GlobalTensor<float> current,GlobalTensor<float> src,GlobalTensor<float> rhs,uint32_t k,uint32_t n) {
        uint32_t count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams) {
            Duplicate(w[count],0.0f,4096);PipeBarrier<PIPE_ALL>();
            const uint32_t row=tile*16,realRows=row<n?(n-row<16?n-row:16):0;
            if(realRows)DataCopyPad(w[count],src[row*n],DataCopyExtParams{(uint16_t)realRows,n*4,0,(256-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
            count+=4096;
        }
        PipeBarrier<PIPE_ALL>();
        // Every lane independently derives one scale from identical A/B data.
        Duplicate(w[8192],0.0f,2304);PipeBarrier<PIPE_ALL>();
        DataCopyPad(w[8192],src,DataCopyExtParams{1,n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        DataCopyPad(w[8448],rhs,DataCopyExtParams{1,n*k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
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
        count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams) {
            for(uint32_t r=0;r<16;++r)if(tile*16+r>=n)w.SetValue(count+r*256+tile*16+r,1.0f);
            PipeBarrier<PIPE_ALL>();
            DataCopy(normalized[tile*4096],w[count],4096);DataCopy(current[tile*4096],w[count],4096);count+=4096;
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Finish(GlobalTensor<float> dst,GlobalTensor<float> x,GlobalTensor<float> dx,uint32_t k,uint32_t n,GM_ADDR rr,bool hasResidual) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*16+c)*4);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<n;row+=teams*16) {
            DataCopy(w,x[row*16],256);DataCopy(w[4096],dx[row*16],256);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],256);PipeBarrier<PIPE_ALL>();
            Gather(w[8192],w,ix,0,16*k);PipeBarrier<PIPE_ALL>();DataCopyPad(dst[row*k],w[8192],DataCopyExtParams{1,(n-row<16?n-row:16)*k*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
        if(lane==0 && hasResidual) {
            GlobalTensor<float> gr;gr.SetGlobalBuffer((__gm__ float*)rr);
            Duplicate(w,0.0f,8);PipeBarrier<PIPE_ALL>();
            DataCopyPad(gr[(uint64_t)batch*k],w,DataCopyExtParams{1,k*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
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
            for(uint32_t d=0;d<kOperandDigits;++d) {
                Adds(q,v,12582912.0f,L);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,L);PipeBarrier<PIPE_V>();
                Cast(hd,q,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();Cast(bd,hd,RoundMode::CAST_TRUNC,L);PipeBarrier<PIPE_V>();
                Sub(v,v,q,L);PipeBarrier<PIPE_V>();Muls(v,v,128.0f,L);PipeBarrier<PIPE_ALL>();
                DataCopyPad(digits[start*kResidualWidth+d*256],bd,DataCopyExtParams{8,256,0,kResidualWidth-256,0});PipeBarrier<PIPE_ALL>();
            }
        }
    }
    __aicore__ inline void ZeroUnusedFactorRows(GlobalTensor<float> dst) {
        if(lane<(256-2*Panel)/16) {Duplicate(w,0.0f,4096);PipeBarrier<PIPE_ALL>();DataCopy(dst[(2*Panel/16+lane)*4096],w,4096);PipeBarrier<PIPE_ALL>();}
    }
    template<uint32_t W> __aicore__ inline void ReprojectCoefficients(GlobalTensor<float> h,GlobalTensor<float> coefficient,uint32_t cut) {
        for(uint32_t row=lane*16;row<cut;row+=teams*16) {
            DataCopy(w,coefficient[row*W],16*W);
            DataCopyPad(w[4096],h[row*256+cut],DataCopyExtParams{16,W*4,(256-W)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
            Add(w[4096],w[4096],w,16*W);PipeBarrier<PIPE_ALL>();
            DataCopy(coefficient[row*W],w,512);DataCopyPad(h[row*256+cut],w[4096],DataCopyExtParams{16,W*4,0,(256-W)*4,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void TailQRDirect(LocalTensor<float> panel,uint32_t real) {
        auto triangular=w[16384],product=w[20480],partial=w[36864],sum=w[37376],coeff=w[37440],weights=w[37504];
        auto ix=w[38016].ReinterpretCast<int32_t>();
        Duplicate(triangular,0.0f,4096);Duplicate(partial,0.0f,512);PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<real;++j) {
            Mul(product,panel[j*256],panel[j*256],256);PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,product,64,4,1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,sum,4,1,1,1,8);PipeBarrier<PIPE_V>();Sqrt(sum,sum,1);PipeBarrier<PIPE_ALL>();
            float norm=sum.GetValue(0);triangular.SetValue(j*64+63,norm);PipeBarrier<PIPE_ALL>();
            Muls(panel[j*256],panel[j*256],1.0f/norm,256);PipeBarrier<PIPE_V>();
            uint32_t rows=real-j-1;
            if(rows)for(uint32_t pass=0;pass<(real<=4?1u:2u);++pass) {
                for(uint32_t q=0;q<4;++q)Mul(product[q*64],panel[(j+1)*256+q*64],panel[j*256+q*64],(uint64_t)64,(uint8_t)rows,BinaryRepeatParams{1,1,1,32,32,0});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(partial,product,64,rows*4,2,1,8);PipeBarrier<PIPE_V>();
                WholeReduceSum(coeff,partial,8,rows,1,1,1);PipeBarrier<PIPE_V>();
                Add(triangular[j*64],triangular[j*64],coeff,rows);PipeBarrier<PIPE_V>();
                Muls(coeff,coeff,-1.0f,rows);PipeBarrier<PIPE_V>();Brcb(weights,coeff,(uint8_t)((rows+7)/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(panel[(j+1)*256+q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)rows,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_ALL>();
            }
        }
        for(int32_t j=(int32_t)real-1;j>=0;--j) {
            float norm=triangular.GetValue(j*64+63);
            Muls(panel[j*256],panel[j*256],1.0f/norm,256);PipeBarrier<PIPE_V>();
            if(j) {
                for(int32_t r=0;r<j;++r)ix.SetValue(r,(r*64+j-r-1)*4);PipeBarrier<PIPE_ALL>();
                Gather(coeff,triangular,ix.ReinterpretCast<uint32_t>(),0,(uint32_t)j);PipeBarrier<PIPE_V>();
                Muls(coeff,coeff,-1.0f,(uint32_t)j);PipeBarrier<PIPE_V>();Brcb(weights,coeff,(uint8_t)((j+7)/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(panel[q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)j,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_V>();Muls(panel,panel,-1.0f,real*256);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) bool TailFour(LocalTensor<float> panel,uint32_t real) {
        // These temporaries avoid the persistent outer 16/24-row tail factors
        // and the direct QR scratch used by at most four remaining rows.
        auto gram=w[24576],aug=w[24640],pivot=w[24704],col=w[24720],weights=w[24736];
        auto ix=w[24800].ReinterpretCast<uint32_t>();auto factor=w[24832],first=w[25088];
        auto product=w[28672],partial=w[36864],coeff=w[37440];
        // Each 8-row slab holds one independent prefix-row product group.
        // Pad unused rows before the combined reductions for shorter tails.
        if(real<8)Duplicate(product,0.0f,8192);
        Duplicate(partial,0.0f,512);PipeBarrier<PIPE_V>();
        for(uint32_t j=0;j<4;++j)
            for(uint32_t q=0;q<4;++q)Mul(product[j*2048+q*64],panel[q*64],panel[j*256+q*64],(uint64_t)64,(uint8_t)real,BinaryRepeatParams{1,1,1,32,32,0});
        PipeBarrier<PIPE_V>();WholeReduceSum(partial,product,64,128,2,1,8);PipeBarrier<PIPE_V>();
        WholeReduceSum(gram,partial,8,32,1,1,1);PipeBarrier<PIPE_V>();
        Duplicate(aug,0.0f,64);Duplicate(factor,0.0f,32);Duplicate(first,0.0f,1024);PipeBarrier<PIPE_V>();
        Muls(aug,gram,1.0f,(uint64_t)4,4,UnaryRepeatParams{1,1,2,1});PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<4;++r){aug.SetValue(r*16+8+r,1.0f);ix.SetValue(r,r*16*4);}PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<4;++j) {
            const float inverse=1.0f/aug.GetValue(j*16+j);
            Muls(pivot,aug[j*16],-inverse,16);Gather(col,aug,ix,j*4,4);PipeBarrier<PIPE_V>();
            Brcb(weights,col,1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(aug,pivot,weights,(uint64_t)16,4,BinaryRepeatParams{1,1,0,2,0,1});PipeBarrier<PIPE_V>();
            Muls(aug[j*16],pivot,-1.0f,16);PipeBarrier<PIPE_ALL>();
        }
        for(uint32_t j=0;j<4;++j) {
            Gather(col,aug,ix,(8+j)*4,4);PipeBarrier<PIPE_V>();Muls(col,col,-1.0f,4);PipeBarrier<PIPE_V>();
            Brcb(weights,col,1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(factor,gram[j*8],weights,(uint64_t)8,4,BinaryRepeatParams{1,1,0,1,0,1});PipeBarrier<PIPE_V>();
        }
        DataCopy(product,factor,32);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<4;++r)product.SetValue(r*8+r,1.0f+product.GetValue(r*8+r));PipeBarrier<PIPE_ALL>();
        Abs(product,product,32);PipeBarrier<PIPE_V>();WholeReduceSum(partial,product,4,4,1,1,1);PipeBarrier<PIPE_V>();
        WholeReduceMax(coeff,partial,4,1,1,1,8);WholeReduceSum(coeff[8],partial,4,1,1,1,8);PipeBarrier<PIPE_ALL>();
        const float defect=coeff.GetValue(0),total=coeff.GetValue(8);
        if(!(defect<=0.015625f && total>=0.0f && total<=0.25f))return false;
        const uint32_t remaining=real-4;
        for(uint32_t j=0;j<4;++j) {
            Brcb(weights,factor[j*8],1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t q=0;q<4;++q)MulAddDst(panel[1024+q*64],panel[j*256+q*64],weights[32],(uint64_t)64,(uint8_t)remaining,BinaryRepeatParams{1,1,0,32,0,1});
            PipeBarrier<PIPE_V>();Gather(col,aug,ix,(8+j)*4,4);PipeBarrier<PIPE_V>();Muls(col,col,-1.0f,4);PipeBarrier<PIPE_V>();
            Brcb(weights,col,1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t q=0;q<4;++q)MulAddDst(first[q*64],panel[j*256+q*64],weights,(uint64_t)64,4,BinaryRepeatParams{1,1,0,32,0,1});
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();TailQRDirect(panel[1024],remaining);
        for(uint32_t r=0;r<4;++r)ix.SetValue(r,r*8*4);PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<remaining;++j) {
            Gather(col,factor,ix,(4+j)*4,4);PipeBarrier<PIPE_V>();Brcb(weights,col,1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t q=0;q<4;++q)MulAddDst(first[q*64],panel[(4+j)*256+q*64],weights,(uint64_t)64,4,BinaryRepeatParams{1,1,0,32,0,1});
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();DataCopy(panel,first,1024);PipeBarrier<PIPE_ALL>();return true;
    }
    __aicore__ inline void TailQRResident(LocalTensor<float> panel,uint32_t real) {
        if(real>4 && real<=8 && TailFour(panel,real))return;
        TailQRDirect(panel,real);
    }
    __aicore__ inline void TailQR(GlobalTensor<float> dst,GlobalTensor<float> src,uint32_t real) {
        if(lane)return;DataCopy(w,src,real*256);PipeBarrier<PIPE_ALL>();
        TailQRResident(w,real);DataCopy(dst,w,real*256);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void PublishTailReady(GlobalTensor<int32_t> dst,int32_t value) {
        auto flag=w[9216].ReinterpretCast<int32_t>();Duplicate(flag,value,8);
        PipeBarrier<PIPE_ALL>();DataCopy(dst,flag,8);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Prefix=16,bool CubeApply=false>
    __aicore__ __attribute__((noinline)) void LocalTail32(GlobalTensor<float> dst,GlobalTensor<float> src,GlobalTensor<float> gramSource,uint32_t real,GM_ADDR scratch=nullptr) {
        static_assert(Prefix==16 || Prefix==24);
        static_assert(!CubeApply || (Panel==112 && Prefix==24));
        if(lane)return;
        GlobalTensor<float> transform;GlobalTensor<int32_t> ready;
        if constexpr(CubeApply) {
            // scratch is the current batch's workspace, supplied by the caller.
            transform.SetGlobalBuffer((__gm__ float*)scratch+GI);
            ready.SetGlobalBuffer((__gm__ int32_t*)scratch+REDUCE+96);
        }
        constexpr uint32_t B=Prefix,S=2*B;
        auto panel=w,first=w[8192],gram=w[B==16?12288:20480],aug=w[B==16?12800:22528],factor=w[B==16?13312:14336];
        auto weights=w[B==16?13824:15104],pivot=w[B==16?13952:15360],col=w[B==16?14016:15424];
        auto index=w[B==16?14080:15456].ReinterpretCast<uint32_t>();
        auto factorIndex=w[B==16?14080:15520].ReinterpretCast<uint32_t>();
        auto product=w[20480],partial=w[36864],coeff=w[37440];
        Duplicate(panel,0.0f,8192);Duplicate(partial,0.0f,512);PipeBarrier<PIPE_ALL>();
        DataCopy(panel,src,real*256);DataCopy(gram,gramSource,B*32);PipeBarrier<PIPE_ALL>();
        Duplicate(aug,0.0f,B*S);Duplicate(factor,0.0f,B*32);Duplicate(first,0.0f,CubeApply?1024:B*256);PipeBarrier<PIPE_ALL>();
        DataCopy(aug,gram,DataCopyParams{B,B/8,(32-B)/8,B/8});PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<B;++r) {
            aug.SetValue(r*S+B+r,1.0f);index.SetValue(r,r*S*4);
            if constexpr(B!=16 && !CubeApply)factorIndex.SetValue(r,r*32*4);
        }
        PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            const float inverse=1.0f/aug.GetValue(j*S+j);
            Muls(pivot,aug[j*S],-inverse,S);Gather(col,aug,index,j*4,B);PipeBarrier<PIPE_V>();
            Brcb(weights,col,B/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(aug,pivot,weights,(uint64_t)S,(uint8_t)B,BinaryRepeatParams{1,1,0,S/8,0,1});PipeBarrier<PIPE_V>();
            Muls(aug[j*S],pivot,-1.0f,S);PipeBarrier<PIPE_ALL>();
        }
        for(uint32_t j=0;j<B;++j) {
            Gather(col,aug,index,(B+j)*4,B);PipeBarrier<PIPE_V>();Muls(col,col,-1.0f,B);PipeBarrier<PIPE_V>();
            Brcb(weights,col,B/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(factor,gram[j*32],weights,(uint64_t)32,(uint8_t)B,BinaryRepeatParams{1,1,0,4,0,1});PipeBarrier<PIPE_V>();
        }
        // Gram is dead here; the 24-row variant reuses that space for the defect.
        DataCopy(product,factor,DataCopyParams{B,B/8,(32-B)/8,0});PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<B;++r)product.SetValue(r*B+r,1.0f+product.GetValue(r*B+r));PipeBarrier<PIPE_ALL>();
        Abs(product,product,B*B);PipeBarrier<PIPE_V>();WholeReduceSum(partial,product,B,B,1,1,B/8);PipeBarrier<PIPE_V>();
        WholeReduceMax(coeff,partial,B,1,1,1,8);WholeReduceSum(coeff[8],partial,B,1,1,1,8);PipeBarrier<PIPE_ALL>();
        const float defect=coeff.GetValue(0),total=coeff.GetValue(8);
        if(!(defect<=0.015625f && total>=0.0f && total<=0.25f)) {
            TailQRResident(panel,real);DataCopy(dst,panel,real*256);
            if constexpr(CubeApply)PublishTailReady(ready,0);else PipeBarrier<PIPE_ALL>();
            return;
        }
        if constexpr(CubeApply) {
            // T = [-Ginv, H; 0, I]. Its bottom rows preserve the completed QR.
            Muls(aug,aug,-1.0f,B*S);PipeBarrier<PIPE_V>();
            DataCopy(first,aug[B],DataCopyParams{B,B/8,B/8,(32-B)/8});
            DataCopy(first[B],factor[B],DataCopyParams{B,(32-B)/8,B/8,B/8});PipeBarrier<PIPE_ALL>();
            for(uint32_t r=B;r<real;++r)first.SetValue(r*32+r,1.0f);
            PipeBarrier<PIPE_ALL>();
        }
        const uint32_t remaining=real-B;
        for(uint32_t j=0;j<B;++j) {
            Brcb(weights,factor[j*32+B],(remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t q=0;q<4;++q)MulAddDst(panel[B*256+q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)remaining,BinaryRepeatParams{1,1,0,32,0,1});
            PipeBarrier<PIPE_V>();
            if constexpr(!CubeApply) {
                Gather(col,aug,index,(B+j)*4,B);PipeBarrier<PIPE_V>();Muls(col,col,-1.0f,B);PipeBarrier<PIPE_V>();
                Brcb(weights,col,B/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(first[q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)B,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_V>();
            }
        }
        PipeBarrier<PIPE_ALL>();TailQRResident(panel[B*256],remaining);
        if constexpr(CubeApply) {
            DataCopy(transform,first,1024);DataCopy(src[B*256],panel[B*256],remaining*256);
            PublishTailReady(ready,1);
        } else {
            for(uint32_t j=0;j<remaining;++j) {
                Gather(col,factor,factorIndex,(B+j)*4,B);PipeBarrier<PIPE_V>();Brcb(weights,col,B/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(first[q*64],panel[(B+j)*256+q*64],weights,(uint64_t)64,(uint8_t)B,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_V>();
            }
            PipeBarrier<PIPE_ALL>();DataCopy(dst,first,B*256);DataCopy(dst[B*256],panel[B*256],remaining*256);PipeBarrier<PIPE_ALL>();
        }
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


    __aicore__ inline void PadGramPanel(GlobalTensor<float> gram) {
        uint32_t row=lane*16;
        if(row<Panel) {
            Duplicate(w,0.0f,16*(128-Panel));PipeBarrier<PIPE_ALL>();
            DataCopyPad(gram[row*128+Panel],w,DataCopyExtParams{16,(128-Panel)*4,0,Panel*4,0});
        } else if(row<128) {
            Duplicate(w,0.0f,2048);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<16;++j)w.SetValue(j*128+row+j,1.0f);
            PipeBarrier<PIPE_ALL>();DataCopy(gram[row*128],w,2048);
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void NewtonPreparePanel(GlobalTensor<half> packed,GlobalTensor<float> gram) {
        if(lane>=Panel/16)return;
        auto value=w,ab=w[16384],sum=w[32768];DataCopy(value,gram,8192);PipeBarrier<PIPE_ALL>();Abs(ab,value,8192);Duplicate(sum,0.0f,128);PipeBarrier<PIPE_V>();
        Add(ab,ab,ab[64],(uint64_t)64,(uint8_t)64,BinaryRepeatParams{1,1,1,16,16,16});PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,ab,64,64,1,1,16);PipeBarrier<PIPE_V>();WholeReduceMax(sum[256],sum,64,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceMax(sum[512],sum[256],2,1,1,1,8);PipeBarrier<PIPE_ALL>();
        float alpha=1.8f/sum.GetValue(512);auto temp=w[4096],eye=w[12288];auto h=w[16384].ReinterpretCast<half>();
        uint32_t row=lane*16;
        DataCopyPad(value,gram[row*128],DataCopyExtParams{16,Panel*4,(128-Panel)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();Cast(h,value,RoundMode::CAST_RINT,(1*16*Panel));PipeBarrier<PIPE_V>();
        Cast(temp,h,RoundMode::CAST_NONE,(1*16*Panel));PipeBarrier<PIPE_V>();Sub(temp,value,temp,(1*16*Panel));PipeBarrier<PIPE_V>();Cast(h[(4*16*Panel)],temp,RoundMode::CAST_RINT,(1*16*Panel));PipeBarrier<PIPE_V>();
        Muls(eye,w[36864],1.0f,(1*16*Panel));PipeBarrier<PIPE_V>();Cast(h[(2*16*Panel)],eye,RoundMode::CAST_RINT,(1*16*Panel));PipeBarrier<PIPE_V>();
        Muls(temp,value,-alpha,(1*16*Panel));PipeBarrier<PIPE_V>();Add(temp,temp,eye,(1*16*Panel));PipeBarrier<PIPE_V>();Cast(h[(5*16*Panel)],temp,RoundMode::CAST_RINT,(1*16*Panel));PipeBarrier<PIPE_V>();
        Muls(value,eye,-alpha,(1*16*Panel));PipeBarrier<PIPE_V>();Cast(h[(1*16*Panel)],value,RoundMode::CAST_RINT,(1*16*Panel));PipeBarrier<PIPE_V>();
        Muls(value,eye,2.0f,(1*16*Panel));PipeBarrier<PIPE_V>();Cast(h[(3*16*Panel)],value,RoundMode::CAST_RINT,(1*16*Panel));PipeBarrier<PIPE_V>();
        Muls(value,eye,-1.0f,(1*16*Panel));PipeBarrier<PIPE_V>();Cast(h[(6*16*Panel)],value,RoundMode::CAST_RINT,(1*16*Panel));PipeBarrier<PIPE_ALL>();
        DataCopy(packed[row*Panel],h,DataCopyParams{7,Panel,0,Panel*Panel/16-Panel});PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ZeroDecision(GlobalTensor<int32_t> decision) {
        if(batch || lane)return;auto flag=w.ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NewtonGuardPanel(GlobalTensor<float> dst,GlobalTensor<int32_t> decision) {
        if constexpr(Panel==96) {

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

        } else {

            if(lane>=Panel/16)return;auto ab=w[4096],sum=w[8192];uint32_t row=lane*16;
            Duplicate(w,0.0f,2048);PipeBarrier<PIPE_ALL>();
            DataCopyPad(w,dst[row*128],DataCopyExtParams{16,Panel*4,(128-Panel)*4,(128-Panel)/8,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();Abs(ab,w,2048);PipeBarrier<PIPE_V>();
            Add(ab,ab,ab[64],(uint64_t)64,(uint8_t)16,BinaryRepeatParams{1,1,1,16,16,16});PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,ab,64,16,1,1,16);PipeBarrier<PIPE_V>();WholeReduceMax(sum[128],sum,16,1,1,1,8);WholeReduceSum(sum[192],sum,16,1,1,1,8);PipeBarrier<PIPE_ALL>();
            float bound=sum.GetValue(128),total=sum.GetValue(192);
            if(!(bound<=0.0625f && total>=0.0f && total<=2.0f)) {
                auto flag=sum[128].ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);PipeBarrier<PIPE_ALL>();
                SetAtomicAdd<int32_t>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();SetAtomicNone();
            }
        }
    }
    template<bool Rhs> __aicore__ inline float SplitOperandDigits(LocalTensor<float> v,LocalTensor<int8_t> digits) {
        auto q=w[256],ab=w[512],mx=w[768];auto hd=w[1024].ReinterpretCast<half>();
        Abs(ab,v,256);PipeBarrier<PIPE_V>();WholeReduceMax(mx,ab,64,4,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);
        int32_t exponent=(int32_t)((mb>>23)&255)-127-5;
        if((mb&0x7fffffff)==0){exponent=0;Duplicate(v,0.0f,256);PipeBarrier<PIPE_V>();}
        else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;Muls(v,v,scale.value,256);PipeBarrier<PIPE_V>();}
        for(uint32_t d=0;d<kOperandDigits;++d) {
            Adds(q,v,12582912.0f,256);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,256);PipeBarrier<PIPE_V>();
            Cast(hd,q,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();Cast(digits[(Rhs?kOperandDigits-1-d:d)*256],hd,RoundMode::CAST_TRUNC,256);PipeBarrier<PIPE_V>();
            Sub(v,v,q,256);PipeBarrier<PIPE_V>();Muls(v,v,128.0f,256);PipeBarrier<PIPE_V>();
        }
        return (float)exponent;
    }
    __aicore__ __attribute__((noinline)) void SplitXPacked4(GlobalTensor<int8_t> digits,GlobalTensor<float> exponents,GlobalTensor<float> solution,uint32_t k) {
        auto input=w[10000],e=w[8000];auto d=w[1536].ReinterpretCast<int8_t>();auto ix=w[19000].ReinterpretCast<int32_t>();auto packed=w[20000].ReinterpretCast<int8_t>();
        CreateVecIndex(ix,(int32_t)0,256);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)64,256);PipeBarrier<PIPE_ALL>();
        if(lane<k){DataCopy(input,solution,4096);PipeBarrier<PIPE_ALL>();}
        for(uint32_t c=lane;c<8;c+=teams) {
            if(c<k)Gather(w,input[c],ix.ReinterpretCast<uint32_t>(),0,256);else Duplicate(w,0.0f,256);PipeBarrier<PIPE_ALL>();
            float exponent=0;if(c<k)exponent=SplitOperandDigits<true>(w,d);else Duplicate(d.ReinterpretCast<int32_t>(),(int32_t)0,kResidualWidth/4);PipeBarrier<PIPE_ALL>();
            Duplicate(packed.ReinterpretCast<int32_t>(),(int32_t)0,6*kResidualWidth/4);PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<kOperandDigits;++weight)DataCopy(packed[weight*kResidualWidth],d[(kOperandDigits-1-weight)*256],(weight+1)*256);
            // Degree five retains A1*X4 + ... + A4*X1. A0*X5 and A5*X0 are absent.
            DataCopy(packed[5*kResidualWidth+256],d,(kOperandDigits-1)*256);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(digits[c*kResidualWidth],packed,DataCopyExtParams{6,kResidualWidth,0,7*kResidualWidth,0});PipeBarrier<PIPE_ALL>();
            if(c<4) {
                e.SetValue(0,exponent);e.SetValue(8,0.0f);PipeBarrier<PIPE_ALL>();
                DataCopyPad(exponents[c],e,DataCopyExtParams{1,4,0,0,0});DataCopyPad(exponents[c+4],e[8],DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
        }
    }
    __aicore__ __attribute__((noinline)) void Assemble6Packed4(GlobalTensor<float> initial,GlobalTensor<int32_t> products,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> rhs) {
        constexpr uint32_t L=256;
        auto ae=w,xe=w[256],gs=w[512],high=w[1024],low=w[1536],term=w[2048],tmp=w[2560],u=w[3072],v=w[3584],factor=w[4096],ea=w[5120],ex=w[5632];
        auto packed=w[7000];auto allProducts=w[9000].ReinterpretCast<int32_t>();
        DataCopy(ae,expA,256);DataCopy(xe,expX,16);PipeBarrier<PIPE_ALL>();
        for(uint32_t tile=lane;tile<8;tile+=teams) {
            const uint32_t row=tile*32;
            Brcb(ea,ae[row],4,BrcbRepeatParams{1,8});
            Muls(ex,xe,1.0f,(uint64_t)8,32,UnaryRepeatParams{1,1,1,0});
            PipeBarrier<PIPE_ALL>();
            DataCopy(high,rhs[row*16],DataCopyParams{32,1,1,0});
            DataCopy(allProducts,products[row*48],1536);PipeBarrier<PIPE_ALL>();
            auto q0=w[14000].ReinterpretCast<int32_t>(),q1=w[14256].ReinterpretCast<int32_t>(),q2=w[14512].ReinterpretCast<int32_t>(),q3=w[14768].ReinterpretCast<int32_t>(),q4=w[15024].ReinterpretCast<int32_t>(),q5=w[15280].ReinterpretCast<int32_t>();
            // Each coefficient is one aligned eight-lane block per product row.
            DataCopy(q0,allProducts,DataCopyParams{32,1,5,0});
            DataCopy(q1,allProducts[8],DataCopyParams{32,1,5,0});
            DataCopy(q2,allProducts[16],DataCopyParams{32,1,5,0});
            DataCopy(q3,allProducts[24],DataCopyParams{32,1,5,0});
            DataCopy(q4,allProducts[32],DataCopyParams{32,1,5,0});
            DataCopy(q5,allProducts[40],DataCopyParams{32,1,5,0});PipeBarrier<PIPE_V>();
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
            // The canonical middle limb is smaller than a nonzero leading
            // limb; a zero leading limb also makes this FastTwoSum exact.
            // Form AX as two floats before its potentially cancelling sum with B.
            Mul(term,gs,factor,L);PipeBarrier<PIPE_V>();
            Muls(factor,factor,0.00006103515625f,L);PipeBarrier<PIPE_V>();
            Mul(u,q2.ReinterpretCast<float>(),factor,L);PipeBarrier<PIPE_V>();
            Add(tmp,term,u,L);PipeBarrier<PIPE_V>();Sub(v,tmp,term,L);PipeBarrier<PIPE_V>();Sub(low,u,v,L);PipeBarrier<PIPE_V>();
            Muls(factor,factor,0.00006103515625f,L);PipeBarrier<PIPE_V>();
            Mul(v,q4.ReinterpretCast<float>(),factor,L);PipeBarrier<PIPE_V>();Add(low,low,v,L);PipeBarrier<PIPE_V>();
            Add(term,high,tmp,L);PipeBarrier<PIPE_V>();Sub(u,term,high,L);PipeBarrier<PIPE_V>();Sub(v,term,u,L);PipeBarrier<PIPE_V>();Sub(v,high,v,L);PipeBarrier<PIPE_V>();Sub(u,tmp,u,L);PipeBarrier<PIPE_V>();Add(u,u,v,L);PipeBarrier<PIPE_V>();
            Add(low,low,u,L);PipeBarrier<PIPE_V>();Add(high,term,low,L);PipeBarrier<PIPE_V>();Duplicate(packed,0.0f,512);PipeBarrier<PIPE_V>();
            DataCopy(packed,high,DataCopyParams{32,1,0,1});PipeBarrier<PIPE_ALL>();DataCopy(initial[row*16],packed,512);PipeBarrier<PIPE_ALL>();
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
template<bool Cube,uint32_t Panel> __aicore__ inline void Tail(std::conditional_t<Cube,NativeMM<Panel>,Vector<Panel>> &op,
    GlobalTensor<float> mem,GlobalTensor<float> A,GlobalTensor<int8_t> ad,GM_ADDR workspace,
    uint32_t offset,uint32_t real,uint32_t n,uint32_t k,uint32_t batches) {
    const bool local=real>16 && real<=32;
    if(local) {
        if constexpr(Cube) {
            if constexpr(Panel==112) {
                if(real>24)op.template Run<32,32,256,false,true,false>(mem[CT+offset*256],mem[CT+offset*256],mem[GI+1024],batches,STRIDE,STRIDE,STRIDE,256,256,32);
                else op.template Run<16,32,256,false,true,false>(mem[CT+offset*256],mem[CT+offset*256],mem[GI+1024],batches,STRIDE,STRIDE,STRIDE,256,256,32);
            } else op.template Run<16,32,256,false,true,false>(mem[CT+offset*256],mem[CT+offset*256],mem[GI+1024],batches,STRIDE,STRIDE,STRIDE,256,256,32);
        }
        SyncAll<false>();
    }
    if constexpr(Cube)op.PrepareFactor(mem[H],mem[D],batches,STRIDE);
    if constexpr(!Cube) {
        if(local) {
            if constexpr(Panel==112) {
                if(real>24)op.template LocalTail32<24,true>(mem[D+offset*256],mem[CT+offset*256],mem[GI+1024],real,(GM_ADDR)mem.GetPhyAddr());
                else op.template LocalTail32<16>(mem[D+offset*256],mem[CT+offset*256],mem[GI+1024],real);
            } else op.template LocalTail32<16>(mem[D+offset*256],mem[CT+offset*256],mem[GI+1024],real);
        }
        else op.TailQR(mem[D+offset*256],mem[CT+offset*256],real);
        if(k<=4)for(uint32_t panel=96;panel<160;panel+=32)op.SplitAChunk(ad,mem[EA],A,panel,n);
    }
    if constexpr(Panel==112)if(local && real>24) {
        SyncAll<false>();
        if constexpr(Cube)op.ApplyPackedTail(mem,offset,batches);
    }

}
template<bool Cube,uint32_t Panel=96> __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgLstsqTilingData &t,GM_ADDR rr) {
    const uint32_t batches=t.batches,k=t.k,n=t.n;
    GlobalTensor<float> mem,A,B,X;
    mem.SetGlobalBuffer((__gm__ float*)workspace);A.SetGlobalBuffer((__gm__ float*)aa);B.SetGlobalBuffer((__gm__ float*)bb);X.SetGlobalBuffer((__gm__ float*)xx);
    static_assert(Panel==96 || Panel==112);
    std::conditional_t<Cube,NativeMM<Panel>,Vector<Panel>> op;
    uint32_t batch=0;
    if constexpr(Cube)op.Init();else {op.Init(batches);batch=op.batch;mem=mem[(uint64_t)batch*STRIDE];A=A[(uint64_t)batch*n*n];B=B[(uint64_t)batch*n*k];X=X[(uint64_t)batch*n*k];}
    GlobalTensor<int8_t> ad,xd;GlobalTensor<int32_t> products;
    ad.SetGlobalBuffer((__gm__ int8_t*)mem[AD].GetPhyAddr());xd.SetGlobalBuffer((__gm__ int8_t*)mem[XD].GetPhyAddr());products.SetGlobalBuffer((__gm__ int32_t*)mem[PRODUCTS].GetPhyAddr());
    GlobalTensor<half> halfMem;halfMem.SetGlobalBuffer((__gm__ half*)workspace+((uint64_t)batch*STRIDE+NH)*2);
    if constexpr(!Cube){op.Normalize(mem[AN],mem[CT],A,B,k,n);op.PackCachedB(mem[BP],mem[DX],k);op.ZeroUnusedFactorRows(mem[D]);}
    SyncAll<false>();
    for(uint32_t p=0;p<2*Panel;p+=Panel) {
        if constexpr(Cube)op.template GramPanel<256>(mem[CT],mem[G],batches,STRIDE,p);
        else op.PadGramPanel(mem[G]);
        SyncAll<false>();
        GlobalTensor<int32_t> decisions;decisions.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+128);
        if constexpr(!Cube){op.ZeroDecision(decisions);op.NewtonPreparePanel(halfMem,mem[G]);}
        SyncAll<false>();
        if constexpr(Cube) {
            if(p==0)op.template NewtonPanel<false>(halfMem,mem[GI],mem[NT],batches,STRIDE,Panel==96?8:9);
            else op.template NewtonPanel<true>(halfMem,mem[GI],mem[NT],batches,STRIDE,Panel==112?12:(n<=208?14:10));
            op.CrossGramFuture256(mem[CT],mem[CROSS],batches,STRIDE,p);
        } else {
            if(p==0)op.SplitAChunk(ad,mem[EA],A,0,n);
            else {op.SplitAChunk(ad,mem[EA],A,32,n);op.SplitAChunk(ad,mem[EA],A,64,n);}
        }
        SyncAll<false>();
        if constexpr(!Cube)op.NewtonGuardPanel(mem[NT],decisions);
        if constexpr(Cube){op.FactorPanel(mem[GI],mem[CT],mem[D],batches,STRIDE,p);op.HFromCross256(mem[CROSS],mem[H],batches,STRIDE,p);}
        SyncAll<false>();DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decisions);
        if(decisions.GetValue(0)!=0) {
            Inverse<128,Cube>(op,mem[GI],mem[G],batches);
            if constexpr(!Cube)op.template NegateBlock<128>(mem[GI]);SyncAll<false>();
            if constexpr(Cube){op.FactorPanel(mem[GI],mem[CT],mem[D],batches,STRIDE,p);op.HFromCross256(mem[CROSS],mem[H],batches,STRIDE,p);}SyncAll<false>();
        }
        if constexpr(Cube)op.template UpdatePanel<256>(mem[H],mem[CT],batches,STRIDE,p);
        if constexpr(Cube)SyncCube();
    }
    // Host assigns at most 32 trailing rows to either active panel width.
    constexpr uint32_t p=2*Panel;
    const uint32_t real=n-p;
    if constexpr(Panel==112) {
        // D's final 32 rows are zero; a 256-row projection keeps all tiles regular.
        if constexpr(Cube)op.template Run<256,32,256,false,true>(mem[D],mem[CT+p*256],mem[CROSS],batches,STRIDE,STRIDE,STRIDE);
        SyncAll<false>();
        if constexpr(!Cube)op.template ReprojectCoefficients<32>(mem[H],mem[CROSS],p);
        SyncAll<false>();
        if constexpr(Cube)op.template Run<32,256,256,true,false,true>(mem[CROSS],mem[CT],mem[CT+p*256],batches,STRIDE,STRIDE,STRIDE);
    } else {
        if constexpr(Cube)op.template Run<192,32,256,false,true>(mem[D],mem[CT+p*256],mem[CROSS],batches,STRIDE,STRIDE,STRIDE);
        SyncAll<false>();
        if constexpr(!Cube)op.template ReprojectCoefficients<32>(mem[H],mem[CROSS],p);
        SyncAll<false>();
        if constexpr(Cube)op.template Run<32,256,192,true,false,true>(mem[CROSS],mem[CT],mem[CT+p*256],batches,STRIDE,STRIDE,STRIDE);
    }
    SyncAll<false>();
    Tail<Cube,Panel>(op,mem,A,ad,workspace,p,real,n,k,batches);
    // Both local QR and Cube application publish directly to the final factor.
    SyncAll<false>();
    if constexpr(Cube) {
        // Tail's existing rendezvous publishes both the prefix work and D2.
        op.CompleteFactor(mem[H],mem[D],batches,STRIDE);SyncCube();
        op.template Run<256,16,256,true>(mem[D],mem[DX],mem[XP],batches,STRIDE,STRIDE,STRIDE);
    }
    SyncAll<false>();
    if constexpr(!Cube)op.SplitXPacked4(xd,mem[EX],mem[XP],k);
    SyncAll<false>();
    if constexpr(Cube)op.Int8Packed4(ad,xd,products,batches,STRIDE);
    SyncAll<false>();
    if constexpr(!Cube)op.Assemble6Packed4(mem[DX],products,mem[EA],mem[EX],mem[BP]);
    SyncAll<false>();
    if constexpr(Cube){op.template Run<256,16,256,true>(mem[D],mem[DX],mem[R],batches,STRIDE,STRIDE,STRIDE);SyncCube();}
    SyncAll<false>();
    if constexpr(!Cube)op.Finish(X,mem[XP],mem[R],k,n,rr,rr && t.residualElements>=(uint64_t)batches*k);
}
}

#ifndef LSTSQ_DIRECT_BENCH
 template <typename DT_A,uint32_t KERNEL_PATH>
__global__ __aicore__ __attribute__((aligned(4096))) void linalg_lstsq(GM_ADDR A,GM_ADDR B,GM_ADDR solution,GM_ADDR residuals,GM_ADDR workspace,GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    InitSocState();
    REGISTER_TILING_DEFAULT(LinalgLstsqTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgLstsqTilingData,t,tiling);
    if constexpr(KERNEL_PATH==0) {
        if ASCEND_IS_AIC {NarrowOrthogonal::Run<true>(A,B,solution,GetUserWorkspace(workspace),t,residuals);}
        if ASCEND_IS_AIV {NarrowOrthogonal::Run<false>(A,B,solution,GetUserWorkspace(workspace),t,residuals);}
    }
    else if constexpr(KERNEL_PATH==8) {
        if ASCEND_IS_AIC {NarrowOrthogonal::Run<true,112>(A,B,solution,GetUserWorkspace(workspace),t,residuals);}
        if ASCEND_IS_AIV {NarrowOrthogonal::Run<false,112>(A,B,solution,GetUserWorkspace(workspace),t,residuals);}
    }
    else if constexpr(KERNEL_PATH==5) {
        if ASCEND_IS_AIC {Orthogonal::Run<true>(A,B,solution,GetUserWorkspace(workspace),t,residuals);}
        if ASCEND_IS_AIV {Orthogonal::Run<false>(A,B,solution,GetUserWorkspace(workspace),t,residuals);}
    }
    else if constexpr(KERNEL_PATH==1) {ResidentElimination<16> op;op.Run(A,B,solution,residuals,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==2) {ResidentElimination<64> op;op.Run(A,B,solution,residuals,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==6) {ResidentElimination<64,false,false,false,true> op;op.Run(A,B,solution,residuals,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==3) {RefinedSquare<128,32> op;op.Run(A,B,solution,residuals,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==4) {RefinedSquare<256,32> op;op.Run(A,B,solution,residuals,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==7) {Lstsq op;op.Run(A,B,solution,residuals,GetUserWorkspace(workspace),t);}
}
#endif
