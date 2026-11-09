#pragma once
#include "kernel_operator.h"
#include "fire_candidate_decode_fusion_tiling.h"
#include "fire_static_metadata.h"
using namespace AscendC;
template<uint32_t Offset,uint32_t Bytes> struct StaticUBBuffer {
    template<class U> __aicore__ inline LocalTensor<U> Get() {return LocalTensor<U>(TPosition::VECCALC,Offset,Bytes/sizeof(U));}
};
template<class T,uint32_t K> struct FireDensePitchPolicy {
    static constexpr uint32_t Value=sizeof(T)==4 && K>1024?
        ((K+127)/128)*128+24:K+(sizeof(T)==4?24:(K>1024?16:8));
};
template<class T> __aicore__ inline void CopyValidInput(LocalTensor<T> dst,GlobalTensor<T> src,uint32_t offset,uint32_t count) {
    uint32_t bytes=count*sizeof(T);uint8_t pad=static_cast<uint8_t>(((32-bytes%32)%32)/sizeof(T));
    DataCopyPad(dst,src[offset],DataCopyExtParams{1,bytes,0,0,0},DataCopyPadExtParams<T>{pad!=0,0,pad,static_cast<T>(0)});
}
template<class T> __aicore__ inline void CopyDenseInput(LocalTensor<T> dst,
    GlobalTensor<T> src,uint32_t offset,uint32_t count) {
    uint32_t bytes=count*sizeof(T);
    uint64_t address=reinterpret_cast<uint64_t>(src.GetPhyAddr())+uint64_t(offset)*sizeof(T);
    if(((address|bytes)&31u)==0) {
        DataCopy(dst,src[offset],DataCopyParams{1,static_cast<uint16_t>(bytes/32),0,0});
    } else {
        CopyValidInput(dst,src,offset,count);
    }
}
template<class T> __aicore__ inline void CopyDenseOutput(GlobalTensor<T> dst,
    LocalTensor<T> src,uint32_t offset,uint32_t count) {
    uint32_t bytes=count*sizeof(T);
    uint64_t address=reinterpret_cast<uint64_t>(dst.GetPhyAddr())+uint64_t(offset)*sizeof(T);
    if(((address|bytes)&31u)==0) {
        DataCopy(dst[offset],src,DataCopyParams{1,static_cast<uint16_t>(bytes/32),0,0});
    } else {
        DataCopyPad(dst[offset],src,DataCopyExtParams{1,bytes,0,0,0});
    }
}
namespace FireSoft53 {
constexpr uint64_t INV=0x40471547652b82feULL;
constexpr uint64_t COEFF[3]={0x3ebc6af84b912394ULL,0x3f2ebfce50fac4f3ULL,0x3f962e42ff0c52d6ULL};
constexpr uint64_t TAB[32]={0x3ff0000000000000ULL,0x3ff059b0d3158574ULL,0x3ff0b5586cf9890fULL,0x3ff11301d0125b51ULL,0x3ff172b83c7d517bULL,0x3ff1d4873168b9aaULL,0x3ff2387a6e756238ULL,0x3ff29e9df51fdee1ULL,0x3ff306fe0a31b715ULL,0x3ff371a7373aa9cbULL,0x3ff3dea64c123422ULL,0x3ff44e086061892dULL,0x3ff4bfdad5362a27ULL,0x3ff5342b569d4f82ULL,0x3ff5ab07dd485429ULL,0x3ff6247eb03a5585ULL,0x3ff6a09e667f3bcdULL,0x3ff71f75e8ec5f74ULL,0x3ff7a11473eb0187ULL,0x3ff82589994cce13ULL,0x3ff8ace5422aa0dbULL,0x3ff93737b0cdc5e5ULL,0x3ff9c49182a3f090ULL,0x3ffa5503b23e255dULL,0x3ffae89f995ad3adULL,0x3ffb7f76f2fb5e47ULL,0x3ffc199bdd85529cULL,0x3ffcb720dcef9069ULL,0x3ffd5818dcfba487ULL,0x3ffdfc97337b9b5fULL,0x3ffea4afa2a490daULL,0x3fff50765b6e4540ULL};
__aicore__ inline int topbits(uint32_t v){
 if(!v)return 0;int n=1;
 if(v>>16){v>>=16;n+=16;}if(v>>8){v>>=8;n+=8;}
 if(v>>4){v>>=4;n+=4;}if(v>>2){v>>=2;n+=2;}if(v>>1)++n;return n;
}
struct Big {
 uint32_t w[16]{};
 __aicore__ inline int used()const{int n=16;while(n&&w[n-1]==0)--n;return n;}
 __aicore__ inline int length()const{int n=used();return n?32*(n-1)+topbits(w[n-1]):0;}
 __aicore__ inline bool bit(int b)const{return b>=0&&b<512&&((w[b/32]>>(b%32))&1);}
 __aicore__ inline bool below(int b)const{
 if(b<=0)return false;int q=b/32,s=b%32;
 for(int i=0;i<q&&i<16;++i)if(w[i])return true;
 return q<16&&s&&(w[q]&((uint32_t(1)<<s)-1));
}
 static __aicore__ inline Big of(uint64_t a){Big r;r.w[0]=uint32_t(a);r.w[1]=uint32_t(a>>32);return r;}
 __aicore__ inline Big shift(int b)const{Big r;int n=used(),q=b/32,s=b%32;for(int i=0;i<n;++i){r.w[i+q]|=w[i]<<s;if(s && i+q+1<16)r.w[i+q+1]|=w[i]>>(32-s);}return r;}
 __aicore__ inline uint64_t take(int b)const{
 if(b<0||b>=512)return 0;int q=b/32,s=b%32;
 uint64_t a=w[q],c=q+1<16?w[q+1]:0,d=q+2<16?w[q+2]:0;
 return s?((a>>s)|(c<<(32-s))|(d<<(64-s))):(a|(c<<32));
}
};
__aicore__ inline int compare(const Big&a,const Big&b){for(int i=15;i>=0;--i)if(a.w[i]!=b.w[i])return a.w[i]>b.w[i]?1:-1;return 0;}
__aicore__ inline Big plus(const Big&a,const Big&b){Big r;uint64_t carry=0;for(int i=0;i<16;++i){uint64_t t=uint64_t(a.w[i])+b.w[i]+carry;r.w[i]=uint32_t(t);carry=t>>32;}return r;}
__aicore__ inline Big minus(const Big&a,const Big&b){Big r;uint64_t borrow=0;for(int i=0;i<16;++i){uint64_t t=uint64_t(b.w[i])+borrow;r.w[i]=uint32_t(uint64_t(a.w[i])-t);borrow=uint64_t(a.w[i])<t;}return r;}
__aicore__ inline Big product(uint64_t a,uint64_t b){Big r;uint32_t aw[2]={uint32_t(a),uint32_t(a>>32)},bw[2]={uint32_t(b),uint32_t(b>>32)};for(int i=0;i<2;++i){uint64_t carry=0;for(int j=0;j<2;++j){uint64_t t=uint64_t(aw[i])*bw[j]+r.w[i+j]+carry;r.w[i+j]=uint32_t(t);carry=t>>32;}r.w[i+2]=uint32_t(carry);}return r;}
struct D{uint64_t m=0;int e=0;bool neg=false;}; // magnitude m*2^e; nonzero normal binary64 m has 53 bits
__aicore__ inline D round53(Big m,int e,bool neg){int n=m.length();if(!n)return {};int cut=n-53;uint64_t sig;if(cut>0){sig=m.take(cut);if(m.bit(cut-1)&&(m.below(cut-1)||(sig&1)))++sig;e+=cut;if(sig==(uint64_t(1)<<53)){sig>>=1;++e;}}else{sig=m.take(0)<<(-cut);e+=cut;}return {sig,e,neg};}
__aicore__ inline D unpack(uint64_t u){int ef=int((u>>52)&2047);uint64_t m=u&0xfffffffffffffULL;if(!ef){return {};}return {(uint64_t(1)<<52)|m,ef-1023-52,bool(u>>63)};}
__aicore__ inline uint64_t pack(D d){return d.m?(uint64_t(d.neg)<<63)|(uint64_t(d.e+52+1023)<<52)|(d.m&0xfffffffffffffULL):0;}
__aicore__ inline D from32(uint32_t u){uint32_t f=u&0x7fffff;int ef=int((u>>23)&255);if(!ef&&!f)return {};uint64_t m=ef?f|0x800000:f;int e=ef?ef-127-23:-149;return round53(Big::of(m),e,bool(u>>31));}
__aicore__ inline D integer(int k){return round53(Big::of(uint64_t(k<0?-int64_t(k):k)),0,k<0);}
__aicore__ inline D combine(Big a,int ae,bool as,Big b,int be,bool bs){if(!a.used())return round53(b,be,bs);if(!b.used())return round53(a,ae,as);int e=(ae<be?ae:be);a=a.shift(ae-e);b=b.shift(be-e);if(as==bs)return round53(plus(a,b),e,as);int c=compare(a,b);return c>=0?round53(minus(a,b),e,as):round53(minus(b,a),e,bs);}
__aicore__ inline D add53(D a,D b){return combine(Big::of(a.m),a.e,a.neg,Big::of(b.m),b.e,b.neg);}
__aicore__ inline D mul53(D a,D b){return round53(product(a.m,b.m),a.e+b.e,a.neg!=b.neg);}
__aicore__ inline D fma53(D a,D b,D c){return combine(product(a.m,b.m),a.e+b.e,a.neg!=b.neg,Big::of(c.m),c.e,c.neg);}
__aicore__ inline int away(D z){if(!z.m)return 0;int s=-z.e;if(s<1||s>63){if(s>63)return 0;return 0;}uint64_t k=z.m>>s;uint64_t mask=(uint64_t(1)<<s)-1;if((z.m&mask)>=(uint64_t(1)<<(s-1)))++k;return z.neg?-int(k):int(k);}
__aicore__ inline uint32_t narrow32(D d){if(!d.m)return 0;int e=d.e+52; uint64_t q=d.m>>29,rem=d.m&((uint64_t(1)<<29)-1);if(rem>(uint64_t(1)<<28)||(rem==(uint64_t(1)<<28)&&(q&1)))++q;if(q==(uint64_t(1)<<24)){q>>=1;++e;}return(uint32_t(d.neg)<<31)|(uint32_t(e+127)<<23)|(uint32_t(q)&0x7fffff);}
__aicore__ inline uint32_t evaluate(uint32_t u){
 D x=from32(u),z=mul53(x,unpack(INV));int k=away(z);D r=add53(z,integer(-k));
 D p0=fma53(unpack(COEFF[0]),r,unpack(COEFF[1]));D p1=fma53(unpack(COEFF[2]),r,integer(1));
 D rr=mul53(r,r),y=fma53(p0,rr,p1);int q=k/32,j=k%32;if(j<0){j+=32;--q;}
 D scale=unpack(TAB[j]);scale.e+=q;return narrow32(mul53(y,scale));
}
}

// RN32 reciprocal for a positive normal denominator in the bounded center repair.
__aicore__ inline uint32_t CenterReciprocalBits(uint32_t denominator) {
    uint32_t exponent = (denominator >> 23) & 255;
    uint32_t mantissa = (denominator & 0x7fffff) | 0x800000;
    if (mantissa == 0x800000) return (254 - exponent) << 23;
    uint32_t quotient = 0, remainder = 0x800000;
    for (uint32_t bit = 0; bit < 24; ++bit) {
        remainder <<= 1;
        quotient <<= 1;
        if (remainder >= mantissa) { remainder -= mantissa; ++quotient; }
    }
    if (2 * remainder > mantissa || (2 * remainder == mantissa && (quotient & 1))) ++quotient;
    uint32_t outExponent = 253 - exponent;
    if (quotient == 0x1000000) { quotient >>= 1; ++outExponent; }
    return (outExponent << 23) | (quotient & 0x7fffff);
}

template<class T, uint32_t K>
__aicore__ inline void RepairSensitiveCenters(LocalTensor<float>& values,
    LocalTensor<float>& priors, LocalTensor<uint32_t>& offsets, GlobalTensor<T>& raw,
    LocalTensor<float>& scratch, uint32_t pitch, uint32_t count, uint32_t rawOffset) {
    ResetMask(); SetMaskNorm();
    if constexpr (K <= 64) {
        WholeReduceMax(scratch, priors[2 * pitch], count, 1, 1, 1, 8,
            ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        ReduceMax(scratch, priors[2 * pitch], scratch[8], count, false);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID1); WaitFlag<HardEvent::V_S>(EVENT_ID1);
    if (scratch.GetValue(0) <= (sizeof(T) == 2 ? 4096.f : 32.f)) return;
    for (uint32_t channel = 0; channel < 2; ++channel) {
        ResetMask(); SetMaskNorm();
        Muls(scratch, values[channel * pitch], 2.f, count); PipeBarrier<PIPE_V>();
        Adds(scratch, scratch, -.5f, count); PipeBarrier<PIPE_V>();
        Add(scratch, scratch, priors[channel * pitch], count); PipeBarrier<PIPE_V>();
        Abs(scratch, scratch, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1); WaitFlag<HardEvent::V_S>(EVENT_ID1);
        for (uint32_t n = 0; n < count; ++n) {
            if (scratch.GetValue(n) >= 0x1p-4f ||
                priors.GetValue(2 * pitch + n) <= (sizeof(T) == 2 ? 4096.f : 32.f)) continue;
            float original = static_cast<float>(raw.GetValue(uint64_t(rawOffset) + n * 6 + channel));
            if (original > -20.f && original < 0.f) {
                union { float f; uint32_t u; } x;
                x.f = -original;
                uint32_t expBits = FireSoft53::evaluate(x.u);
                uint32_t denominator = FireSoft53::narrow32(FireSoft53::add53(
                    FireSoft53::from32(expBits), FireSoft53::integer(1)));
                values.template ReinterpretCast<uint32_t>().SetValue(channel * pitch + n,
                    CenterReciprocalBits(denominator));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1); WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
}

template<uint32_t Channels,bool RestoreNormal=true>
__aicore__ inline void GatherChannelPlanes(const LocalTensor<float>& destination,
    const LocalTensor<float>& source, const LocalTensor<uint32_t>& offsets,
    uint32_t pitch, uint32_t count) {
    // One counter mask covers valid rows in every independent channel.
    SetMaskCount();
    SetVectorMask<float,MaskMode::COUNTER>(count);
    uint32_t base = static_cast<uint32_t>(reinterpret_cast<uint64_t>(source.GetPhyAddr()));
    __ubuf__ uint32_t* indices = (__ubuf__ uint32_t*)offsets.GetPhyAddr();
#pragma unroll
    for (uint32_t channel=0;channel<Channels;++channel) {
        vgather((__ubuf__ uint32_t*)destination[channel*pitch].GetPhyAddr(),
            indices,base+channel*4,8,1);
    }
    PipeBarrier<PIPE_V>();
    if constexpr(RestoreNormal) {ResetMask(); SetMaskNorm();}
}

template<class T,uint32_t K,uint32_t valuesOffset,uint32_t priorOffset,uint32_t scratchOffset>
__aicore__ __attribute__((noinline)) void RepairDenseCold(GM_ADDR rawAddress,
    uint32_t count,uint32_t rawOffset);

template<class T,uint32_t K,bool StaticGuard=false>
__aicore__ inline bool SmallNeedsRepair(LocalTensor<float>& priors,
    LocalTensor<float>& scratch,uint32_t pitch,uint32_t count) {
    ResetMask();SetMaskNorm();
    if constexpr(StaticGuard) {
        static_assert(K==128,"two bounded reduction segments required");
        // The guard only asks whether any valid stride exceeds its bound.
        // Explicit events keep this bounded reduction independent of TPipe.
        const uint32_t first=count<64?count:64;
        WholeReduceMax(scratch,priors[2*pitch],first,1,1,1,8,
            ReduceOrder::ORDER_ONLY_VALUE);
        if(count>64) {
            WholeReduceMax(scratch[8],priors[2*pitch+64],count-64,1,1,1,8,
                ReduceOrder::ORDER_ONLY_VALUE);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const float bound=sizeof(T)==2?4096.f:32.f;
        const bool repair=!(scratch.GetValue(0)<=bound) ||
            (count>64 && !(scratch.GetValue(8)<=bound));
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        return repair;
    } else if constexpr(K<=64) {
        WholeReduceMax(scratch,priors[2*pitch],count,1,1,1,8,
            ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        ReduceMax(scratch,priors[2*pitch],scratch[8],count,false);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
    return !(scratch.GetValue(0)<=(sizeof(T)==2?4096.f:32.f));
}

template<class T, uint32_t K> class PadSmallDecode {
    static constexpr uint32_t P=K+8,Q=K+16;
    StaticUBBuffer<0,8*P*4> priorBuf;
    StaticUBBuffer<8*P*4,8*P*4> planeBuf;
    StaticUBBuffer<16*P*4,16*K*4> rawBuf;
    StaticUBBuffer<16*P*4+16*K*4,16*K*2> halfBuf;
    StaticUBBuffer<16*P*4+24*K*4,8*K*4> priorPackedBuf;
    StaticUBBuffer<16*P*4+32*K*4,8*K*4> outBuf;
    StaticUBBuffer<16*P*4+40*K*4,64*4> onesBuf;
    StaticUBBuffer<16*P*4+40*K*4+64*4,K> maskBuf;
    StaticUBBuffer<16*P*4+40*K*4+64*4+K,K*2*4> indexBuf;
    GlobalTensor<T> raw,out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
    uint64_t rawSrc[16],rawDst[16],priorSrc[16],priorDst[16],outSrc[16],outDst[16];
public:
#if FIRE_COMPILED_SMALL_SINGLE
    __aicore__ inline static void PrefetchCompleteSingleInputs(GM_ADDR r,GM_ADDR p) {
        static_assert(sizeof(T)==4 && K==32,"complete Float compact input");
        GlobalTensor<T> input;input.SetGlobalBuffer((__gm__ T*)r);
        GlobalTensor<float> priors;priors.SetGlobalBuffer((__gm__ float*)p);
        LocalTensor<T> inputLocal(TPosition::VECCALC,16*P*4,16*K);
        LocalTensor<float> priorLocal(TPosition::VECCALC,16*P*4+24*K*4,8*K);
        CopyDenseInput(priorLocal,priors,0,K*5);
        CopyDenseInput(inputLocal,input,0,K*6);
    }
#endif
    __aicore__ inline static void PrefetchIndexMetadata(GM_ADDR addressMetadata) {
        static_assert(K<=128,"index metadata is a compact-capacity resource");
        GlobalTensor<uint32_t> words;words.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(fire_metadata::Table<0,K>::Address()));
        LocalTensor<uint32_t> destination(TPosition::VECCALC,
            16*P*4+40*K*4+64*4+K,2*K);
        DataCopy(destination,words,2*K);
    }
    template<bool IndexPrefetched=false,bool EntryInitialized=false>
    __aicore__ inline void Init(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t) {
        d=t;
#if __NPU_ARCH__ == 2201
        if constexpr(!EntryInitialized) {InitSocState();ResetMask();SetMaskNorm();}
#else
        InitSocState();ResetMask();SetMaskNorm();
#endif
        raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);out.SetGlobalBuffer((__gm__ T*)o);
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto pr=priorBuf.template Get<float>();
        constexpr uint32_t row=sizeof(T)==2?16:8;
        for(uint32_t c=0;c<8;++c) {
            if constexpr(sizeof(T)==2) {
                auto hs=halfBuf.template Get<T>();auto hd=rawBuf.template Get<T>();
                rawSrc[c]=reinterpret_cast<uint64_t>(hs[c*16].GetPhyAddr());rawSrc[c+8]=reinterpret_cast<uint64_t>(hs[(c+8)*16].GetPhyAddr());
                rawDst[c]=reinterpret_cast<uint64_t>(hd[c*Q].GetPhyAddr());rawDst[c+8]=reinterpret_cast<uint64_t>(hd[(c+8)*Q].GetPhyAddr());
            } else {
                rawSrc[c]=reinterpret_cast<uint64_t>(s[c*row].GetPhyAddr());rawSrc[c+8]=reinterpret_cast<uint64_t>(s[(c+8)*row].GetPhyAddr());
                rawDst[2*c]=reinterpret_cast<uint64_t>(v[c*P].GetPhyAddr());rawDst[2*c+1]=reinterpret_cast<uint64_t>(v[c*P+8].GetPhyAddr());
            }
            priorSrc[c]=reinterpret_cast<uint64_t>(pp[c*8].GetPhyAddr());priorSrc[c+8]=reinterpret_cast<uint64_t>(pp[(c+8)*8].GetPhyAddr());
            priorDst[2*c]=reinterpret_cast<uint64_t>(pr[c*P].GetPhyAddr());priorDst[2*c+1]=reinterpret_cast<uint64_t>(pr[c*P+8].GetPhyAddr());
            uint32_t channel=c<5?c:0;
            outSrc[c]=reinterpret_cast<uint64_t>(v[channel*P].GetPhyAddr());outSrc[c+8]=reinterpret_cast<uint64_t>(v[channel*P+8].GetPhyAddr());
            outDst[2*c]=reinterpret_cast<uint64_t>(s[c*8].GetPhyAddr());outDst[2*c+1]=reinterpret_cast<uint64_t>(s[(c+8)*8].GetPhyAddr());
        }
        if constexpr(!IndexPrefetched) {
        auto idx=indexBuf.template Get<int32_t>();
        for(uint32_t j=0;j<8;++j) {idx.SetValue(j,static_cast<int32_t>(j));}
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for(uint32_t span=8;span<K;span*=2) {Adds(idx[span],idx,static_cast<int32_t>(span),span);PipeBarrier<PIPE_V>();}
        Muls(idx[K],idx,static_cast<int32_t>(20),K);PipeBarrier<PIPE_V>();
        Muls(idx,idx,static_cast<int32_t>(24),K);PipeBarrier<PIPE_V>();
        } else static_assert(K<=128,"prefetched index capacity");
        Duplicate(planeBuf.template Get<float>(),0.f,8*P);PipeBarrier<PIPE_V>();
        Duplicate(onesBuf.template Get<float>(),1.f,64);PipeBarrier<PIPE_V>();
        Duplicate(maskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),K/4);PipeBarrier<PIPE_V>();
    }
    template<bool StaticGuard=false,bool CompactGeometry=false>
    __aicore__ inline void ProcessSingle() {
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto p=priorBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto o=outBuf.template Get<float>();
#if FIRE_COMPILED_SMALL_SINGLE
        const uint32_t b=0,n=0,count=K;
#else
        const uint32_t b=GetBlockIdx(),n=0,count=d.n<K?d.n:K;
#endif
        uint64_t reserved=0;
#if FIRE_COMPILED_SMALL_SINGLE
        if constexpr (!(sizeof(T)==4 && K==32)) {
#endif
            CopyDenseInput(pp,prior,n*5,count*5);
            if constexpr(sizeof(T)==2) {CopyDenseInput(halfBuf.template Get<T>(),raw,(b*d.n+n)*6,count*6);}
            else {CopyDenseInput(rawBuf.template Get<T>(),raw,(b*d.n+n)*6,count*6);}
#if FIRE_COMPILED_SMALL_SINGLE
        }
#endif
            {
                // Only scalar initialization writes these immutable tables.
                // This path has no other VA-register consumer or producer.
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr(sizeof(T)==2) {
                auto bits=halfBuf.template Get<int16_t>();
                Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
                Cast(s,halfBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();
            } else {Muls(s,s,-1.f,count*6);PipeBarrier<PIPE_V>();}
            Exp(s,s,count*6);PipeBarrier<PIPE_V>();Adds(s,s,1.f,count*6);PipeBarrier<PIPE_V>();
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
            Div<float,false>(s,onesBuf.template Get<float>(),s,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
            auto idx=indexBuf.template Get<uint32_t>();
            GatherChannelPlanes<5>(p,pp,idx[K],P,count);
            bool needsRepair;
            if constexpr(K<=64) {
                ResetMask();SetMaskNorm();
                WholeReduceMax(o,p[2*P],count,1,1,1,8,
                    ReduceOrder::ORDER_ONLY_VALUE);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID1);
                GatherChannelPlanes<6>(v,s,idx,P,count);
                WaitFlag<HardEvent::V_S>(EVENT_ID1);
                needsRepair=!(*((__ubuf__ float*)o.GetPhyAddr())<=(sizeof(T)==2?4096.f:32.f));
            } else {
                GatherChannelPlanes<6>(v,s,idx,P,count);
                needsRepair=SmallNeedsRepair<T,K,StaticGuard>(p,s,P,count);
            }
            if(__builtin_expect(needsRepair,false)) {
                // Only primitive arguments cross the cold-call boundary.
                // The existing unscaled repair reconstructs these same views.
                RepairDenseCold<T,K,8*P*4,0,16*P*4>(
                    (GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            for(uint32_t off=0;off<count;off+=64) {
                uint64_t m=count-off<64?count-off:64;
                auto x=v[off];auto q=p[off];
                uint64_t rowMask = m == 64 ? ~uint64_t(0) : ((uint64_t(1) << m) - 1);
                SetVectorMask<float,MaskMode::NORMAL>(0,rowMask);
                if constexpr(K<=64 || CompactGeometry) {
                    auto* xp=(__ubuf__ float*)x.GetPhyAddr();
                    auto* qp=(__ubuf__ float*)q.GetPhyAddr();
                    vmuls(xp,xp,2.f,4,1,1,P/8,P/8);pipe_barrier(PIPE_V);
                    vadds(xp,xp,-.5f,2,1,1,P/8,P/8);
                    vmul(xp+2*P,xp+2*P,xp+2*P,2,1,1,1,P/8,P/8,P/8);
                    pipe_barrier(PIPE_V);
                    vadd(xp,xp,qp,2,1,1,1,P/8,P/8,P/8);
                    vmul(xp+2*P,xp+2*P,qp+3*P,2,1,1,1,P/8,P/8,P/8);
                    pipe_barrier(PIPE_V);
                    vmul(xp,xp,qp+2*P,4,1,1,1,P/8,P/8,0);
                    vmul(xp+4*P,xp+4*P,xp+5*P,1,1,1,1,0,0,0);
                    pipe_barrier(PIPE_V);
                } else {
                Muls<float,false>(x,x,2.f,m,4,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
                Adds<float,false>(x,x,-.5f,m,2,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
                Add<float,false>(x,x,q,m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[2*P],x[2*P],x[2*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[2*P],x[2*P],q[3*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x,x,q[2*P],m,4,{1,1,1,P/8,P/8,0});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[4*P],x[4*P],x[5*P],m,1,{1,1,1,0,0,0});PipeBarrier<PIPE_V>();
                }
            }
            ResetMask(); SetMaskNorm();
            uint8_t repeats=static_cast<uint8_t>((count+15)/16);
            scatter_vnchwconv_b32(VA0,VA2,repeats,repeats==1?0:16,repeats==1?0:2);
            PipeBarrier<PIPE_V>();
            GatherMask(o,s,maskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved);PipeBarrier<PIPE_V>();
            if constexpr(sizeof(T)==2) {Cast(halfBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();}
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if constexpr(sizeof(T)==2) CopyDenseOutput(out,halfBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            else CopyDenseOutput(out,outBuf.template Get<T>(),(b*d.n+n)*5,count*5);
        PipeBarrier<PIPE_ALL>();
    }
    template<bool IndexPrefetched=false,bool OwnerKnown=false,bool EntryInitialized=false>
    __aicore__ inline void InitWithInputPrefetch(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t,int64_t owner=0) {
        d=t;
#if __NPU_ARCH__ == 2201
        if constexpr(!EntryInitialized) {InitSocState();ResetMask();SetMaskNorm();}
#else
        InitSocState();ResetMask();SetMaskNorm();
#endif
        raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);out.SetGlobalBuffer((__gm__ T*)o);
        const uint32_t count=t.n<K?t.n:K;
        static_assert(!OwnerKnown || K==64,"known-owner input resource required");
        const int64_t inputOwner=OwnerKnown?owner:GetBlockIdx();
        CopyDenseInput(priorPackedBuf.template Get<float>(),prior,0,count*5);
        if constexpr(sizeof(T)==2) CopyDenseInput(halfBuf.template Get<T>(),raw,inputOwner*t.n*6,count*6);
        else CopyDenseInput(rawBuf.template Get<T>(),raw,inputOwner*t.n*6,count*6);
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto pr=priorBuf.template Get<float>();
        constexpr uint32_t row=sizeof(T)==2?16:8;
        for(uint32_t c=0;c<8;++c) {
            if constexpr(sizeof(T)==2) {
                auto hs=halfBuf.template Get<T>();auto hd=rawBuf.template Get<T>();
                rawSrc[c]=reinterpret_cast<uint64_t>(hs[c*16].GetPhyAddr());rawSrc[c+8]=reinterpret_cast<uint64_t>(hs[(c+8)*16].GetPhyAddr());
                rawDst[c]=reinterpret_cast<uint64_t>(hd[c*Q].GetPhyAddr());rawDst[c+8]=reinterpret_cast<uint64_t>(hd[(c+8)*Q].GetPhyAddr());
            } else {
                rawSrc[c]=reinterpret_cast<uint64_t>(s[c*row].GetPhyAddr());rawSrc[c+8]=reinterpret_cast<uint64_t>(s[(c+8)*row].GetPhyAddr());
                rawDst[2*c]=reinterpret_cast<uint64_t>(v[c*P].GetPhyAddr());rawDst[2*c+1]=reinterpret_cast<uint64_t>(v[c*P+8].GetPhyAddr());
            }
            priorSrc[c]=reinterpret_cast<uint64_t>(pp[c*8].GetPhyAddr());priorSrc[c+8]=reinterpret_cast<uint64_t>(pp[(c+8)*8].GetPhyAddr());
            priorDst[2*c]=reinterpret_cast<uint64_t>(pr[c*P].GetPhyAddr());priorDst[2*c+1]=reinterpret_cast<uint64_t>(pr[c*P+8].GetPhyAddr());
            uint32_t channel=c<5?c:0;
            outSrc[c]=reinterpret_cast<uint64_t>(v[channel*P].GetPhyAddr());outSrc[c+8]=reinterpret_cast<uint64_t>(v[channel*P+8].GetPhyAddr());
            outDst[2*c]=reinterpret_cast<uint64_t>(s[c*8].GetPhyAddr());outDst[2*c+1]=reinterpret_cast<uint64_t>(s[(c+8)*8].GetPhyAddr());
        }
        if constexpr(!IndexPrefetched) {
        auto idx=indexBuf.template Get<int32_t>();
        for(uint32_t j=0;j<8;++j) {idx.SetValue(j,static_cast<int32_t>(j));}
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for(uint32_t span=8;span<K;span*=2) {Adds(idx[span],idx,static_cast<int32_t>(span),span);PipeBarrier<PIPE_V>();}
        Muls(idx[K],idx,static_cast<int32_t>(20),K);PipeBarrier<PIPE_V>();
        Muls(idx,idx,static_cast<int32_t>(24),K);PipeBarrier<PIPE_V>();
        } else static_assert(K<=128,"prefetched index capacity");
        Duplicate(planeBuf.template Get<float>(),0.f,8*P);PipeBarrier<PIPE_V>();
        Duplicate(onesBuf.template Get<float>(),1.f,64);PipeBarrier<PIPE_V>();
        Duplicate(maskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),K/4);PipeBarrier<PIPE_V>();
    }
    template<bool PriorMaskForward=false,bool OwnerKnown=false>
    __aicore__ inline void ProcessSingleAfterInputPrefetch(int64_t owner=0) {
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto p=priorBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto o=outBuf.template Get<float>();
        static_assert(!OwnerKnown || K==64,"known-owner output resource required");
        const uint32_t b=static_cast<uint32_t>(OwnerKnown?owner:GetBlockIdx()),n=0,count=d.n<K?d.n:K;
        uint64_t reserved=0;
            {
                // Only scalar initialization writes these immutable tables.
                // This path has no other VA-register consumer or producer.
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr(sizeof(T)==2) {
                auto bits=halfBuf.template Get<int16_t>();
                Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
                Cast(s,halfBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();
            } else {Muls(s,s,-1.f,count*6);PipeBarrier<PIPE_V>();}
            Exp(s,s,count*6);PipeBarrier<PIPE_V>();Adds(s,s,1.f,count*6);PipeBarrier<PIPE_V>();
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
            Div<float,false>(s,onesBuf.template Get<float>(),s,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
            auto idx=indexBuf.template Get<uint32_t>();
            GatherChannelPlanes<5,!(PriorMaskForward && K==64)>(p,pp,idx[K],P,count);
            bool needsRepair;
            if constexpr(K<=64) {
                ResetMask();SetMaskNorm();
                WholeReduceMax(o,p[2*P],count,1,1,1,8,
                    ReduceOrder::ORDER_ONLY_VALUE);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID1);
                GatherChannelPlanes<6>(v,s,idx,P,count);
                WaitFlag<HardEvent::V_S>(EVENT_ID1);
                needsRepair=!(*((__ubuf__ float*)o.GetPhyAddr())<=(sizeof(T)==2?4096.f:32.f));
            } else {
                GatherChannelPlanes<6>(v,s,idx,P,count);
                needsRepair=SmallNeedsRepair<T,K>(p,s,P,count);
            }
            if(__builtin_expect(needsRepair,false)) {
                // Only primitive arguments cross the cold-call boundary.
                // The existing unscaled repair reconstructs these same views.
                RepairDenseCold<T,K,8*P*4,0,16*P*4>(
                    (GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            for(uint32_t off=0;off<count;off+=64) {
                uint64_t m=count-off<64?count-off:64;
                auto x=v[off];auto q=p[off];
                uint64_t rowMask = m == 64 ? ~uint64_t(0) : ((uint64_t(1) << m) - 1);
                SetVectorMask<float,MaskMode::NORMAL>(0,rowMask);
                if constexpr(K<=64) {
                    auto* xp=(__ubuf__ float*)x.GetPhyAddr();
                    auto* qp=(__ubuf__ float*)q.GetPhyAddr();
                    vmuls(xp,xp,2.f,4,1,1,P/8,P/8);pipe_barrier(PIPE_V);
                    vadds(xp,xp,-.5f,2,1,1,P/8,P/8);
                    vmul(xp+2*P,xp+2*P,xp+2*P,2,1,1,1,P/8,P/8,P/8);
                    pipe_barrier(PIPE_V);
                    vadd(xp,xp,qp,2,1,1,1,P/8,P/8,P/8);
                    vmul(xp+2*P,xp+2*P,qp+3*P,2,1,1,1,P/8,P/8,P/8);
                    pipe_barrier(PIPE_V);
                    vmul(xp,xp,qp+2*P,4,1,1,1,P/8,P/8,0);
                    vmul(xp+4*P,xp+4*P,xp+5*P,1,1,1,1,0,0,0);
                    pipe_barrier(PIPE_V);
                } else {
                Muls<float,false>(x,x,2.f,m,4,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
                Adds<float,false>(x,x,-.5f,m,2,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
                Add<float,false>(x,x,q,m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[2*P],x[2*P],x[2*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[2*P],x[2*P],q[3*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x,x,q[2*P],m,4,{1,1,1,P/8,P/8,0});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[4*P],x[4*P],x[5*P],m,1,{1,1,1,0,0,0});PipeBarrier<PIPE_V>();
                }
            }
            ResetMask(); SetMaskNorm();
            uint8_t repeats=static_cast<uint8_t>((count+15)/16);
            scatter_vnchwconv_b32(VA0,VA2,repeats,repeats==1?0:16,repeats==1?0:2);
            PipeBarrier<PIPE_V>();
            GatherMask(o,s,maskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved);PipeBarrier<PIPE_V>();
            if constexpr(sizeof(T)==2) {Cast(halfBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();}
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if constexpr(sizeof(T)==2) CopyDenseOutput(out,halfBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            else CopyDenseOutput(out,outBuf.template Get<T>(),(b*d.n+n)*5,count*5);
        PipeBarrier<PIPE_ALL>();
    }
    template<bool IndexPrefetched=false,bool EntryInitialized=false>
    __aicore__ inline void InitWithHalfRawReady(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t,int64_t owner) {
        d=t;
#if __NPU_ARCH__ == 2201
        if constexpr(!EntryInitialized) {InitSocState();ResetMask();SetMaskNorm();}
#else
        InitSocState();ResetMask();SetMaskNorm();
#endif
        raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);out.SetGlobalBuffer((__gm__ T*)o);
        const uint32_t count=t.n<K?t.n:K;
        static_assert(sizeof(T)==2 && K==32,"compact Half resource only");
        CopyDenseInput(halfBuf.template Get<T>(),raw,owner*t.n*6,count*6);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        CopyDenseInput(priorPackedBuf.template Get<float>(),prior,0,count*5);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto pr=priorBuf.template Get<float>();
        constexpr uint32_t row=sizeof(T)==2?16:8;
        for(uint32_t c=0;c<8;++c) {
            if constexpr(sizeof(T)==2) {
                auto hs=halfBuf.template Get<T>();auto hd=rawBuf.template Get<T>();
                rawSrc[c]=reinterpret_cast<uint64_t>(hs[c*16].GetPhyAddr());rawSrc[c+8]=reinterpret_cast<uint64_t>(hs[(c+8)*16].GetPhyAddr());
                rawDst[c]=reinterpret_cast<uint64_t>(hd[c*Q].GetPhyAddr());rawDst[c+8]=reinterpret_cast<uint64_t>(hd[(c+8)*Q].GetPhyAddr());
            } else {
                rawSrc[c]=reinterpret_cast<uint64_t>(s[c*row].GetPhyAddr());rawSrc[c+8]=reinterpret_cast<uint64_t>(s[(c+8)*row].GetPhyAddr());
                rawDst[2*c]=reinterpret_cast<uint64_t>(v[c*P].GetPhyAddr());rawDst[2*c+1]=reinterpret_cast<uint64_t>(v[c*P+8].GetPhyAddr());
            }
            priorSrc[c]=reinterpret_cast<uint64_t>(pp[c*8].GetPhyAddr());priorSrc[c+8]=reinterpret_cast<uint64_t>(pp[(c+8)*8].GetPhyAddr());
            priorDst[2*c]=reinterpret_cast<uint64_t>(pr[c*P].GetPhyAddr());priorDst[2*c+1]=reinterpret_cast<uint64_t>(pr[c*P+8].GetPhyAddr());
            uint32_t channel=c<5?c:0;
            outSrc[c]=reinterpret_cast<uint64_t>(v[channel*P].GetPhyAddr());outSrc[c+8]=reinterpret_cast<uint64_t>(v[channel*P+8].GetPhyAddr());
            outDst[2*c]=reinterpret_cast<uint64_t>(s[c*8].GetPhyAddr());outDst[2*c+1]=reinterpret_cast<uint64_t>(s[(c+8)*8].GetPhyAddr());
        }
        if constexpr(!IndexPrefetched) {
        auto idx=indexBuf.template Get<int32_t>();
        for(uint32_t j=0;j<8;++j) {idx.SetValue(j,static_cast<int32_t>(j));}
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for(uint32_t span=8;span<K;span*=2) {Adds(idx[span],idx,static_cast<int32_t>(span),span);PipeBarrier<PIPE_V>();}
        Muls(idx[K],idx,static_cast<int32_t>(20),K);PipeBarrier<PIPE_V>();
        Muls(idx,idx,static_cast<int32_t>(24),K);PipeBarrier<PIPE_V>();
        } else static_assert(K<=128,"prefetched index capacity");
        Duplicate(planeBuf.template Get<float>(),0.f,8*P);PipeBarrier<PIPE_V>();
        Duplicate(onesBuf.template Get<float>(),1.f,64);PipeBarrier<PIPE_V>();
        Duplicate(maskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),K/4);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ProcessSingleWithHalfRawReady(int64_t owner) {
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto p=priorBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto o=outBuf.template Get<float>();
        const uint32_t b=static_cast<uint32_t>(owner),n=0,count=d.n<K?d.n:K;
        uint64_t reserved=0;
            {
                // Only scalar initialization writes these immutable tables.
                // This path has no other VA-register consumer or producer.
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr(sizeof(T)==2) {
                auto bits=halfBuf.template Get<int16_t>();
                Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
                Cast(s,halfBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();
            } else {Muls(s,s,-1.f,count*6);PipeBarrier<PIPE_V>();}
            Exp(s,s,count*6);PipeBarrier<PIPE_V>();Adds(s,s,1.f,count*6);PipeBarrier<PIPE_V>();
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
            Div<float,false>(s,onesBuf.template Get<float>(),s,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
            auto idx=indexBuf.template Get<uint32_t>();
            // Div has produced all raw planes. These gathers only write V;
            // the still-pending prior DMA writes the disjoint packed slot.
            GatherChannelPlanes<6>(v,s,idx,P,count);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            GatherChannelPlanes<5>(p,pp,idx[K],P,count);
            bool needsRepair;
            if constexpr(K<=64) {
                ResetMask();SetMaskNorm();
                WholeReduceMax(o,p[2*P],count,1,1,1,8,
                    ReduceOrder::ORDER_ONLY_VALUE);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID1);
                WaitFlag<HardEvent::V_S>(EVENT_ID1);
                needsRepair=!(*((__ubuf__ float*)o.GetPhyAddr())<=(sizeof(T)==2?4096.f:32.f));
            } else {
                GatherChannelPlanes<6>(v,s,idx,P,count);
                needsRepair=SmallNeedsRepair<T,K>(p,s,P,count);
            }
            if(__builtin_expect(needsRepair,false)) {
                // Only primitive arguments cross the cold-call boundary.
                // The existing unscaled repair reconstructs these same views.
                RepairDenseCold<T,K,8*P*4,0,16*P*4>(
                    (GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            for(uint32_t off=0;off<count;off+=64) {
                uint64_t m=count-off<64?count-off:64;
                auto x=v[off];auto q=p[off];
                uint64_t rowMask = m == 64 ? ~uint64_t(0) : ((uint64_t(1) << m) - 1);
                SetVectorMask<float,MaskMode::NORMAL>(0,rowMask);
                if constexpr(K<=64) {
                    auto* xp=(__ubuf__ float*)x.GetPhyAddr();
                    auto* qp=(__ubuf__ float*)q.GetPhyAddr();
                    vmuls(xp,xp,2.f,4,1,1,P/8,P/8);pipe_barrier(PIPE_V);
                    vadds(xp,xp,-.5f,2,1,1,P/8,P/8);
                    vmul(xp+2*P,xp+2*P,xp+2*P,2,1,1,1,P/8,P/8,P/8);
                    pipe_barrier(PIPE_V);
                    vadd(xp,xp,qp,2,1,1,1,P/8,P/8,P/8);
                    vmul(xp+2*P,xp+2*P,qp+3*P,2,1,1,1,P/8,P/8,P/8);
                    pipe_barrier(PIPE_V);
                    vmul(xp,xp,qp+2*P,4,1,1,1,P/8,P/8,0);
                    vmul(xp+4*P,xp+4*P,xp+5*P,1,1,1,1,0,0,0);
                    pipe_barrier(PIPE_V);
                } else {
                Muls<float,false>(x,x,2.f,m,4,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
                Adds<float,false>(x,x,-.5f,m,2,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
                Add<float,false>(x,x,q,m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[2*P],x[2*P],x[2*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[2*P],x[2*P],q[3*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x,x,q[2*P],m,4,{1,1,1,P/8,P/8,0});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[4*P],x[4*P],x[5*P],m,1,{1,1,1,0,0,0});PipeBarrier<PIPE_V>();
                }
            }
            ResetMask(); SetMaskNorm();
            uint8_t repeats=static_cast<uint8_t>((count+15)/16);
            scatter_vnchwconv_b32(VA0,VA2,repeats,repeats==1?0:16,repeats==1?0:2);
            PipeBarrier<PIPE_V>();
            GatherMask(o,s,maskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved);PipeBarrier<PIPE_V>();
            if constexpr(sizeof(T)==2) {Cast(halfBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();}
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if constexpr(sizeof(T)==2) CopyDenseOutput(out,halfBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            else CopyDenseOutput(out,outBuf.template Get<T>(),(b*d.n+n)*5,count*5);
        PipeBarrier<PIPE_ALL>();
    }
    template<bool StaticGuard=false,bool CompactGeometry=false>
    __aicore__ inline void Process() {
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto p=priorBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto o=outBuf.template Get<float>();
        uint32_t tiles=(d.n+K-1)/K,total=tiles*d.batch,begin=total*GetBlockIdx()/d.blocks,end=total*(GetBlockIdx()+1)/d.blocks;
        uint64_t reserved=0;
        for(uint32_t tile=begin;tile<end;++tile) {
            uint32_t b=tile%d.batch,n=tile/d.batch*K,count=d.n-n<K?d.n-n:K;
            CopyDenseInput(pp,prior,n*5,count*5);
            if constexpr(sizeof(T)==2) {CopyDenseInput(halfBuf.template Get<T>(),raw,(b*d.n+n)*6,count*6);}
            else {CopyDenseInput(rawBuf.template Get<T>(),raw,(b*d.n+n)*6,count*6);}
            if(tile==begin) {
                // Only scalar initialization writes these immutable tables.
                // This path has no other VA-register consumer or producer.
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr(sizeof(T)==2) {Cast(s,halfBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();}
            Muls(s,s,-1.f,count*6);PipeBarrier<PIPE_V>();
            Exp(s,s,count*6);PipeBarrier<PIPE_V>();Adds(s,s,1.f,count*6);PipeBarrier<PIPE_V>();
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
            Div<float,false>(s,onesBuf.template Get<float>(),s,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
            auto idx=indexBuf.template Get<uint32_t>();
            GatherChannelPlanes<5>(p,pp,idx[K],P,count);
            GatherChannelPlanes<6>(v,s,idx,P,count);
            if(__builtin_expect(SmallNeedsRepair<T,K,StaticGuard>(p,s,P,count),false)) {
                // Only primitive arguments cross the cold-call boundary.
                // The existing unscaled repair reconstructs these same views.
                RepairDenseCold<T,K,8*P*4,0,16*P*4>(
                    (GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            for(uint32_t off=0;off<count;off+=64) {
                uint64_t m=count-off<64?count-off:64;
                auto x=v[off];auto q=p[off];
                uint64_t rowMask = m == 64 ? ~uint64_t(0) : ((uint64_t(1) << m) - 1);
                SetVectorMask<float,MaskMode::NORMAL>(0,rowMask);
                if constexpr(CompactGeometry) {
                    auto* xp=(__ubuf__ float*)x.GetPhyAddr();
                    auto* qp=(__ubuf__ float*)q.GetPhyAddr();
                    vmuls(xp,xp,2.f,4,1,1,P/8,P/8);pipe_barrier(PIPE_V);
                    vadds(xp,xp,-.5f,2,1,1,P/8,P/8);
                    vmul(xp+2*P,xp+2*P,xp+2*P,2,1,1,1,P/8,P/8,P/8);
                    pipe_barrier(PIPE_V);
                    vadd(xp,xp,qp,2,1,1,1,P/8,P/8,P/8);
                    vmul(xp+2*P,xp+2*P,qp+3*P,2,1,1,1,P/8,P/8,P/8);
                    pipe_barrier(PIPE_V);
                    vmul(xp,xp,qp+2*P,4,1,1,1,P/8,P/8,0);
                    vmul(xp+4*P,xp+4*P,xp+5*P,1,1,1,1,0,0,0);
                    pipe_barrier(PIPE_V);
                } else {
                Muls<float,false>(x,x,2.f,m,4,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
                Adds<float,false>(x,x,-.5f,m,2,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
                Add<float,false>(x,x,q,m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[2*P],x[2*P],x[2*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[2*P],x[2*P],q[3*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
                Mul<float,false>(x,x,q[2*P],m,4,{1,1,1,P/8,P/8,0});PipeBarrier<PIPE_V>();
                Mul<float,false>(x[4*P],x[4*P],x[5*P],m,1,{1,1,1,0,0,0});PipeBarrier<PIPE_V>();
                }
            }
            ResetMask(); SetMaskNorm();
            uint8_t repeats=static_cast<uint8_t>((count+15)/16);
            scatter_vnchwconv_b32(VA0,VA2,repeats,repeats==1?0:16,repeats==1?0:2);
            PipeBarrier<PIPE_V>();
            GatherMask(o,s,maskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved);PipeBarrier<PIPE_V>();
            if constexpr(sizeof(T)==2) {Cast(halfBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();}
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if constexpr(sizeof(T)==2) CopyDenseOutput(out,halfBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            else CopyDenseOutput(out,outBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            if(tile+1<end) {SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);}
            }
        PipeBarrier<PIPE_ALL>();
    }
};
template<class T,uint32_t K,bool EntryInitialized=false>
__aicore__ __attribute__((noinline)) void RunSmallManyIndexed(GM_ADDR r,GM_ADDR p,
    GM_ADDR o,uint32_t batch,uint32_t n,uint32_t blocks,uint32_t tile) {
    static_assert(K==128,"indexed many-tile resource required");
    FireCandidateDecodeFusionTilingData t{batch,n,blocks,tile};
    PadSmallDecode<T,K> op;op.template Init<true,EntryInitialized>(r,p,o,t);op.template Process<true,true>();
}
template<class T,uint32_t K>
__aicore__ __attribute__((noinline)) void RunSmallMany(GM_ADDR r,GM_ADDR p,
    GM_ADDR o,uint32_t batch,uint32_t n,uint32_t blocks,uint32_t tile) {
    FireCandidateDecodeFusionTilingData t{batch,n,blocks,tile};
    if constexpr(K==128) {
        PadSmallDecode<T,K> op;op.Init(r,p,o,t);op.template Process<true>();
    } else if constexpr(K<=64) {
        PadSmallDecode<T,K> op;op.Init(r,p,o,t);op.Process();
    } else {
        TPipe pipe;PadSmallDecode<T,K> op;op.Init(r,p,o,t);op.Process();
    }
}
template<class T, uint32_t K> class KernelFireCandidateDecodeFusion {
    static constexpr uint32_t P=K+8;
    static constexpr uint32_t P0=0,S0=P0+P*5*4,V0=S0+K*8*4,O0=V0+K*8*4;
    static constexpr uint32_t I0=O0+K*5*4,G0=I0+K*2*4,M0=G0+64*4;
    StaticUBBuffer<P0,P*5*4> priorBuf;
    StaticUBBuffer<S0,K*8*4> sigBuf;
    StaticUBBuffer<V0,K*8*4> planeBuf;
    StaticUBBuffer<O0,K*5*4> outBuf;
    StaticUBBuffer<I0,K*2*4> indexBuf;
    StaticUBBuffer<G0,64*4> genBuf;
    StaticUBBuffer<M0,K> maskBuf;
    GlobalTensor<T> raw, out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
public:
    __aicore__ inline void Init(GM_ADDR r, GM_ADDR p, GM_ADDR o, FireCandidateDecodeFusionTilingData t) {
        d=t; raw.SetGlobalBuffer((__gm__ T*)r); prior.SetGlobalBuffer((__gm__ float*)p); out.SetGlobalBuffer((__gm__ T*)o);
        InitSocState();ResetMask();SetMaskNorm();
        auto idx=indexBuf.template Get<int32_t>();
        for(uint32_t j=0;j<8;++j) idx.SetValue(j,static_cast<int32_t>(j));
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for(uint32_t span=8;span<K;span*=2) {
            uint32_t length=K-span<span?K-span:span;
            Adds(idx[span],idx,static_cast<int32_t>(span),length);PipeBarrier<PIPE_V>();
        }
        Muls(idx[K],idx,static_cast<int32_t>(20),K);PipeBarrier<PIPE_V>();
        Muls(idx,idx,static_cast<int32_t>(24),K);PipeBarrier<PIPE_V>();
        Duplicate(genBuf.template Get<float>(),1.0f,64); PipeBarrier<PIPE_V>();
        Duplicate(maskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),K/4); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    __aicore__ inline void Process() {
        auto p=priorBuf.template Get<float>();auto o=outBuf.template Get<float>();auto idx=indexBuf.template Get<uint32_t>();
        auto buf0=sigBuf.template Get<float>();auto buf1=planeBuf.template Get<float>();
        uint32_t tiles=(d.n+K-1)/K,total=tiles*d.batch;
        uint32_t begin=total*GetBlockIdx()/d.blocks,end=total*(GetBlockIdx()+1)/d.blocks;
        uint64_t srcList[2][16],dstList[2][16];
        for(uint32_t parity=0;parity<(end-begin>1?2u:1u);++parity) {
            auto src=parity==0?buf1:buf0;auto dst=parity==0?buf0:buf1;
            for(uint32_t c=0;c<8;++c) {
                uint32_t channel=c<5?c:0;
                srcList[parity][c]=reinterpret_cast<uint64_t>(src[channel*P].GetPhyAddr());
                srcList[parity][c+8]=reinterpret_cast<uint64_t>(src[channel*P+8].GetPhyAddr());
                dstList[parity][2*c]=reinterpret_cast<uint64_t>(dst[c*8].GetPhyAddr());
                dstList[parity][2*c+1]=reinterpret_cast<uint64_t>(dst[(c+8)*8].GetPhyAddr());
            }
        }
        Duplicate(buf0,0.f,K*8);Duplicate(buf1,0.f,K*8);Duplicate(p,0.f,P*5);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        uint64_t reserved=0;
        uint32_t cachedN=d.n;
        for(uint32_t tile=begin;tile<end;++tile) {
            uint32_t parity=(tile-begin)&1;
            auto s=parity==0?buf0:buf1;auto v=parity==0?buf1:buf0;auto packed=s;
            auto r=s.template ReinterpretCast<T>();if constexpr(sizeof(T)==2) r=r[8*K];
            uint32_t b=tile%d.batch, n=(tile/d.batch)*K;
            uint32_t count=(d.n-n<K?d.n-n:K);
            if(n!=cachedN) {
                CopyValidInput(o,prior,n*5,count*5);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                GatherChannelPlanes<5>(p,o,idx[K],P,count);
                cachedN=n;
            }
            {
                if(tile==begin) CopyValidInput(r,raw,(b*d.n+n)*6,count*6);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr(sizeof(T)==2) { Cast(s,r,RoundMode::CAST_NONE,count*6); PipeBarrier<PIPE_V>(); } 
                Muls(s,s,-1.0f,count*6); PipeBarrier<PIPE_V>(); Exp(s,s,count*6); PipeBarrier<PIPE_V>(); Adds(s,s,1.0f,count*6); PipeBarrier<PIPE_V>(); SetMaskCount(); SetVectorMask<float,MaskMode::COUNTER>(count*6); Div<float,false>(s,genBuf.template Get<float>(),s,MASK_PLACEHOLDER,1,{1,1,1,8,0,8}); ResetMask(); SetMaskNorm(); PipeBarrier<PIPE_V>();
                GatherChannelPlanes<6>(v,s,idx,P,count);
                RepairSensitiveCenters<T,K>(v,p,idx,raw,s,P,count,(b*d.n+n)*6);
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count);
                for(uint32_t c=0;c<4;++c) {Muls<float,false>(v[c*P],v[c*P],2.f,MASK_PLACEHOLDER,1,{1,1,8,8});}PipeBarrier<PIPE_V>();
                for(uint32_t c=0;c<2;++c) {Adds<float,false>(v[c*P],v[c*P],-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});}PipeBarrier<PIPE_V>();
                for(uint32_t c=0;c<2;++c) {Add<float,false>(v[c*P],v[c*P],p[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});}PipeBarrier<PIPE_V>();
                for(uint32_t c=2;c<4;++c) {Mul<float,false>(v[c*P],v[c*P],v[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});}PipeBarrier<PIPE_V>();
                for(uint32_t c=2;c<4;++c) {Mul<float,false>(v[c*P],v[c*P],p[(c+1)*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});}PipeBarrier<PIPE_V>();
                for(uint32_t c=0;c<4;++c) {Mul<float,false>(v[c*P],v[c*P],p[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});}PipeBarrier<PIPE_V>();
                Mul<float,false>(v[4*P],v[4*P],v[5*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                ResetMask();SetMaskNorm();
                SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                ResetMask();
                uint8_t repeats=static_cast<uint8_t>((count+15)/16);
                TransDataTo5HD<float>(dstList[parity],srcList[parity],TransDataTo5HDParams(false,false,repeats,repeats==1?0:16,repeats==1?0:2)); PipeBarrier<PIPE_V>();
                if(tile+1<end) {
                    uint32_t nextB=(tile+1)%d.batch,nextN=((tile+1)/d.batch)*K;
                    uint32_t nextCount=d.n-nextN<K?d.n-nextN:K;
                    auto nextRaw=v.template ReinterpretCast<T>();if constexpr(sizeof(T)==2) nextRaw=nextRaw[8*K];
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    CopyValidInput(nextRaw,raw,(nextB*d.n+nextN)*6,nextCount*6);
                }
                GatherMask(o,packed,maskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved); PipeBarrier<PIPE_V>();
                if constexpr(sizeof(T)==2) Cast(outBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5); PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(out[(b*d.n+n)*5],outBuf.template Get<T>(),DataCopyExtParams{1,count*5*uint32_t(sizeof(T)),0,0,0});
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
};
// Split packed six-channel activations with sequential mask compaction.
template<bool ExplicitScalarRead=false>
__aicore__ inline float DirectStrideMax(const LocalTensor<float>& source,
    LocalTensor<float>& scratch,uint32_t count) {
    ResetMask();SetMaskNorm();
    uint32_t full=count/64,tail=count%64;
    if(full) {
        WholeReduceMax(scratch,source,64,full,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        if(full>1) {
            WholeReduceMax(scratch,scratch,full,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
        }
    }
    if(tail) {
        WholeReduceMax(scratch[32],source[full*64],tail,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
    float maximum;
#if __NPU_ARCH__ == 2201 && !defined(ASCENDC_CPU_DEBUG) && !defined(__CCE_KT_TEST__)
    if constexpr(ExplicitScalarRead) {
        // V_S1 completes the reduction; S_V1 releases its Scalar readers.
        auto* reduced=reinterpret_cast<const volatile __ubuf__ float*>(scratch.GetPhyAddr());
        maximum=full?reduced[0]:0.f;
        if(tail){float last=reduced[32];if(last>maximum)maximum=last;}
    } else
#endif
    {
        maximum=full?scratch.GetValue(0):0.f;
        if(tail){float last=scratch.GetValue(32);if(last>maximum)maximum=last;}
    }
    SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
    return maximum;
}
// The one-tile guard owns a separate, immutable reduction result until END.
__aicore__ inline void BeginOneStrideMax(const LocalTensor<float>& source,
    LocalTensor<float>& scratch,uint32_t count) {
    ResetMask();SetMaskNorm();
    uint32_t full=count/64,tail=count%64;
    if(full) {
        WholeReduceMax(scratch,source,64,full,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        if(full>1) {
            WholeReduceMax(scratch,scratch,full,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
        }
    }
    if(tail) {
        WholeReduceMax(scratch[32],source[full*64],tail,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_S>(EVENT_ID1);
}
__aicore__ inline float FinishOneStrideMax(LocalTensor<float>& scratch,uint32_t count) {
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    float maximum=count>=64?scratch.GetValue(0):0.f;
    if(count%64){float last=scratch.GetValue(32);if(last>maximum)maximum=last;}
    return maximum;
}
template<class T,uint32_t K,uint32_t valuesOffset,uint32_t priorOffset,uint32_t scratchOffset>
__aicore__ __attribute__((noinline)) void RepairDenseCold(GM_ADDR rawAddress,
    uint32_t count,uint32_t rawOffset) {
    constexpr uint32_t pitch=K+8;
    LocalTensor<float> values(TPosition::VECCALC,valuesOffset,K*7);
    LocalTensor<float> priors(TPosition::VECCALC,priorOffset,pitch*5);
    LocalTensor<float> scratch(TPosition::VECCALC,scratchOffset,K*8);
    GlobalTensor<T> raw;raw.SetGlobalBuffer((__gm__ T*)rawAddress);
    for (uint32_t channel = 0; channel < 2; ++channel) {
        ResetMask(); SetMaskNorm();
        Muls(scratch, values[channel * pitch], 2.f, count); PipeBarrier<PIPE_V>();
        Adds(scratch, scratch, -.5f, count); PipeBarrier<PIPE_V>();
        Add(scratch, scratch, priors[channel * pitch], count); PipeBarrier<PIPE_V>();
        Abs(scratch, scratch, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1); WaitFlag<HardEvent::V_S>(EVENT_ID1);
        for (uint32_t n = 0; n < count; ++n) {
            if (scratch.GetValue(n) >= 0x1p-4f ||
                priors.GetValue(2 * pitch + n) <= (sizeof(T) == 2 ? 4096.f : 32.f)) continue;
            float original = static_cast<float>(raw.GetValue(uint64_t(rawOffset) + n * 6 + channel));
            if (original > -20.f && original < 0.f) {
                union { float f; uint32_t u; } x;
                x.f = -original;
                uint32_t expBits = FireSoft53::evaluate(x.u);
                uint32_t denominator = FireSoft53::narrow32(FireSoft53::add53(
                    FireSoft53::from32(expBits), FireSoft53::integer(1)));
                values.template ReinterpretCast<uint32_t>().SetValue(channel * pitch + n,
                    CenterReciprocalBits(denominator));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1); WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
}


template<class T,uint32_t K,uint32_t valuesOffset,uint32_t priorOffset,uint32_t scratchOffset>
__aicore__ __attribute__((noinline)) void RepairScaledDenseCold(GM_ADDR rawAddress,
    uint32_t count,uint32_t rawOffset) {
    constexpr uint32_t pitch=FireDensePitchPolicy<T,K>::Value;
    LocalTensor<float> values(TPosition::VECCALC,valuesOffset,K*7);
    LocalTensor<float> priors(TPosition::VECCALC,priorOffset,pitch*5);
    LocalTensor<float> scratch(TPosition::VECCALC,scratchOffset,K*8);
    GlobalTensor<T> raw;raw.SetGlobalBuffer((__gm__ T*)rawAddress);
    for (uint32_t channel = 0; channel < 2; ++channel) {
        ResetMask(); SetMaskNorm();
        Adds(scratch, values[channel * pitch], 0.f, count); PipeBarrier<PIPE_V>();
        Adds(scratch, scratch, -.5f, count); PipeBarrier<PIPE_V>();
        Add(scratch, scratch, priors[channel * pitch], count); PipeBarrier<PIPE_V>();
        Abs(scratch, scratch, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1); WaitFlag<HardEvent::V_S>(EVENT_ID1);
        for (uint32_t n = 0; n < count; ++n) {
            if (scratch.GetValue(n) >= 0x1p-4f ||
                priors.GetValue(2 * pitch + n) <= (sizeof(T) == 2 ? 4096.f : 32.f)) continue;
            float original = static_cast<float>(raw.GetValue(uint64_t(rawOffset) + n * 6 + channel));
            if (original > -20.f && original < 0.f) {
                union { float f; uint32_t u; } x;
                x.f = -original;
                uint32_t expBits = FireSoft53::evaluate(x.u);
                uint32_t denominator = FireSoft53::narrow32(FireSoft53::add53(
                    FireSoft53::from32(expBits), FireSoft53::integer(1)));
                values.template ReinterpretCast<uint32_t>().SetValue(channel * pitch + n,
                    CenterReciprocalBits(denominator)+0x800000u);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1); WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
}


template<class T,uint32_t K,bool Early=true> class MaskDecode {
    static constexpr uint32_t P=FireDensePitchPolicy<T,K>::Value;
    static_assert(6*P<=7*K,"dense planes overlap raw prefetch suffix");
    static constexpr bool FullRawPrefetch=sizeof(T)==4 && K>1024;
    static constexpr uint32_t OPlanes=FullRawPrefetch?6:5;
    //512-byte arena extents preserve the original source bank offsets. The
    // V arena also contains the padded16-row final transpose for K%16!=0.
    static constexpr uint32_t SSpan=FullRawPrefetch?(K*8*4+511)/512*512:K*8*4;
    static constexpr uint32_t OSpan=FullRawPrefetch?(K*OPlanes*4+511)/512*512:K*OPlanes*4;
    static constexpr uint32_t ISpan=FullRawPrefetch?(K*2*4+511)/512*512:K*2*4;
    static constexpr uint32_t P0=0,S0=P*5*4,V0=S0+SSpan+128,O0=V0+SSpan+128;
    static constexpr uint32_t I0=O0+OSpan+128,G0=I0+ISpan,M0=G0+128*4;
    // This kernel uses only the64-element numerator at G0. No K-element
    // numerator tensor exists; bound the buffers that are actually accessed.
    static_assert(M0+K<=196352,"large raw staging exceeds usable UB");
    static constexpr uint32_t MASK_WORDS=((3*K+255)/256)*8;
    static constexpr uint32_t PATTERN_WORDS=MASK_WORDS+16;
    static constexpr uint32_t SEED_WORDS=24;
    static constexpr uint32_t PRIOR_MASK_WORDS=((5*(K+7)+255)/256)*8;
    static constexpr uint32_t PRIOR_PATTERN_WORDS=PRIOR_MASK_WORDS+32;
    static constexpr uint32_t PRIOR_SEED_WORDS=40;
    static constexpr uint32_t PRIOR_STREAMS=PATTERN_WORDS+PRIOR_PATTERN_WORDS;
    static_assert(PRIOR_STREAMS+5*PRIOR_MASK_WORDS<=2*K,
        "independent prior compaction masks exceed index storage");
    static_assert(K*4+3*MASK_WORDS*4<=K*8,"mask metadata exceeds index storage");
    StaticUBBuffer<P0,P*5*4> priorBuf;
    StaticUBBuffer<S0,K*7*4> sigBuf;
    StaticUBBuffer<S0+K*7*4,K*4> rawSuffixBuf;
    StaticUBBuffer<V0,SSpan> workBuf;
    StaticUBBuffer<O0,OSpan> outBuf;
    StaticUBBuffer<I0,ISpan> indexBuf;
    StaticUBBuffer<G0,64*4> onesBuf;
    StaticUBBuffer<M0,K> outputMaskBuf;
    GlobalTensor<T> raw,out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
    uint32_t beginTile,endTile;
    uint64_t outSrc[16],outDst[16];

    __aicore__ inline LocalTensor<T> RawInput() {
        auto result=sigBuf.template Get<T>();
        if constexpr(sizeof(T)==2)result=result[8*K];
        return result;
    }
    // Only the large FP16 path moves bitwise channel separation before
    // widening. Sigmoid/decode/repair still operate entirely in FP32.
    static constexpr bool HalfPrepack=sizeof(T)==2 && K>1024;
    static constexpr uint32_t HQ=(P+15)/16*16;
    __aicore__ inline void UnpackHalfInput(LocalTensor<T> packed,uint32_t count) {
        auto even=sigBuf.template Get<T>();
        auto odd=even[3*K];
        LocalTensor<T> separated(TPosition::VECCALC,S0+16*K,8*K);
        static_assert(16*K+2*(5*HQ+K)<=32*K,"half plane staging exceeds S and suffix");
        SetMaskCount();SetVectorMask<T,MaskMode::COUNTER>(6*count);
        GatherMask(even,packed,uint8_t(1),GatherMaskParams{1,1,8,0});
        GatherMask(odd,packed,uint8_t(2),GatherMaskParams{1,1,8,0});
        PipeBarrier<PIPE_V>();
        SetVectorMask<T,MaskMode::COUNTER>(3*count);
        auto patterns=indexBuf.template Get<T>();
#pragma unroll
        for(uint32_t phase=0;phase<3;++phase) {
            auto mask=patterns[phase==0?0:(phase==1?32:16)];
            GatherMask(separated[(2*phase)*HQ],even,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
            GatherMask(separated[(2*phase+1)*HQ],odd,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
        auto values=sigBuf.template Get<float>();
        static_assert(!HalfPrepack || HQ==P,"half plane pitch must match widening");
        // Staging starts above the complete widening destination. The same
        // per-element cast now traverses the aligned planes in one command.
        Cast(values,separated,RoundMode::CAST_NONE,5*P+count);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StageNextRaw(uint32_t row,uint32_t count) {
        uint32_t elements=count*6;
        if constexpr(sizeof(T)==2) {
            CopyDenseInput(outBuf.template Get<T>(),raw,row*6,elements);
        } else if constexpr(FullRawPrefetch) {
            CopyDenseInput(outBuf.template Get<T>(),raw,row*6,elements);
        } else {
            uint32_t first=elements<5*K?elements:5*K;
            CopyDenseInput(outBuf.template Get<T>(),raw,row*6,first);
            if(elements>first)
                CopyDenseInput(rawSuffixBuf.template Get<T>(),raw,row*6+first,elements-first);
        }
    }
    __aicore__ inline void PrepareNextInput(uint32_t count) {
        auto next=sigBuf.template Get<float>();
        if constexpr(sizeof(T)==2) {
            if constexpr(HalfPrepack) {
                UnpackHalfInput(outBuf.template Get<T>(),count);
            } else {
                Cast(next,outBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);
                PipeBarrier<PIPE_V>();
            }
        } else if constexpr(FullRawPrefetch) {
            Adds(next.template ReinterpretCast<int32_t>(),outBuf.template Get<int32_t>(),int32_t(0),count*6);
            PipeBarrier<PIPE_V>();
        } else {
            uint32_t elements=count*6,first=elements<5*K?elements:5*K;
            auto target=next.template ReinterpretCast<int32_t>();
            Adds(target,outBuf.template Get<int32_t>(),int32_t(0),first);
            PipeBarrier<PIPE_V>();
            if(elements>first) {
                Adds(target[first],rawSuffixBuf.template Get<int32_t>(),int32_t(0),elements-first);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void SetPriorHeads(uint64_t keep) {
        constexpr uint32_t bits[5]={0x42108421U,0x10842108U,
            0x84210842U,0x21084210U,0x08421084U};
        auto masks=indexBuf.template Get<uint64_t>()[PRIOR_STREAMS/2];
#pragma unroll
        for(uint32_t c=0;c<5;++c) {
            uint32_t wordOffset=c==0?0:8*(5-c);
            uint64_t head=uint64_t(bits[wordOffset%5])|
                (uint64_t(bits[(wordOffset+1)%5])<<32);
            masks.SetValue(c*PRIOR_MASK_WORDS/2,head&keep);
        }
    }
    __aicore__ inline void UnpackContiguousPrior(LocalTensor<float>& destination,
        LocalTensor<float>& packed,uint32_t count) {
        // Each MaskDecode tile begins at a complete candidate row. Packed
        // prior DMA has no left prefix, so large masks remain immutable.
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(5*count);
        auto patterns=indexBuf.template Get<float>();
#pragma unroll
        for(uint32_t c=0;c<5;++c) {
            const uint32_t offset=K>1024?
                PATTERN_WORDS+(c==0?0:8*(5-c)):PRIOR_STREAMS+c*PRIOR_MASK_WORDS;
            GatherMask(destination[c*P],packed,patterns[offset],uint8_t(0),
                GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
    }
    __aicore__ inline void SplitPlanes(LocalTensor<float>& values,LocalTensor<float>& even,
        LocalTensor<float>& odd,uint32_t count) {
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*count);
        GatherMask(even,values,uint8_t(1),GatherMaskParams{1,1,8,0});
        GatherMask(odd,values,uint8_t(2),GatherMaskParams{1,1,8,0});
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::COUNTER>(3*count);
        auto patterns=indexBuf.template Get<float>();
#pragma unroll
        for(uint32_t phase=0;phase<(FullRawPrefetch?2:3);++phase) {
            auto mask=patterns[phase==0?0:(phase==1?16:8)];
            GatherMask(values[(2*phase)*P],even,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
            GatherMask(values[(2*phase+1)*P],odd,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
        }
        if constexpr(FullRawPrefetch) {
            PipeBarrier<PIPE_V>();
            // Phase2 pairs the two original score lanes. The same FP32
            // multiply commutes with their bitwise permutation.
            Mul<float,false>(even,even,odd,MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            PipeBarrier<PIPE_V>();
            GatherMask(values[4*P],even,patterns[8],uint8_t(0),GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
    }
public:
    __aicore__ inline void Init(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t) {
        d=t;InitSocState();ResetMask();SetMaskNorm();
        raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);out.SetGlobalBuffer((__gm__ T*)o);
        auto values=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();auto temp=outBuf.template Get<float>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        uint32_t tiles=(d.n+K-1)/K,total=tiles*d.batch;
        beginTile=total*GetBlockIdx()/d.blocks;endTile=total*(GetBlockIdx()+1)/d.blocks;
        uint32_t firstB=beginTile%d.batch,firstN=(beginTile/d.batch)*K;
        uint32_t firstCount=d.n-firstN<K?d.n-firstN:K;
        CopyDenseInput(temp,prior,firstN*5,firstCount*5);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        CopyDenseInput(RawInput(),raw,(firstB*d.n+firstN)*6,firstCount*6);
#pragma unroll
        for(uint32_t c=0;c<8;++c) {
            uint32_t channel=c<5?c:0;
            outSrc[c]=reinterpret_cast<uint64_t>(values[channel*P].GetPhyAddr());
            outSrc[c+8]=reinterpret_cast<uint64_t>(values[channel*P+8].GetPhyAddr());
            outDst[2*c]=reinterpret_cast<uint64_t>(v[c*8].GetPhyAddr());
            outDst[2*c+1]=reinterpret_cast<uint64_t>(v[(c+8)*8].GetPhyAddr());
        }
        // The dense path has one fixed transpose layout. No other operation
        // in this path writes VA0..VA3, including the numerical repair.
        set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
        set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);

        constexpr uint32_t bits[3]={0x49249249U,0x92492492U,0x24924924U};
        auto masks=indexBuf.template Get<uint64_t>();
#pragma unroll
        for(uint32_t j=0;j<SEED_WORDS/2;++j) {
            uint64_t value=uint64_t(bits[(2*j)%3])|
                (uint64_t(bits[(2*j+1)%3])<<32);
            masks.SetValue(j,value);
        }
        constexpr uint32_t priorBits[5]={0x42108421U,0x10842108U,
            0x84210842U,0x21084210U,0x08421084U};
        auto priorMasks=indexBuf.template Get<uint64_t>()[PATTERN_WORDS/2];
#pragma unroll
        for(uint32_t j=0;j<PRIOR_SEED_WORDS/2;++j) {
            uint64_t value=uint64_t(priorBits[(2*j)%5])|
                (uint64_t(priorBits[(2*j+1)%5])<<32);
            priorMasks.SetValue(j,value);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);

        auto maskWords=indexBuf.template Get<int32_t>();
        for(uint32_t span=SEED_WORDS;span<PATTERN_WORDS;span*=2) {
            uint32_t length=PATTERN_WORDS-span<span?PATTERN_WORDS-span:span;
            Adds(maskWords[span],maskWords,int32_t(0),length);
            PipeBarrier<PIPE_V>();
        }
        auto priorWords=indexBuf.template Get<int32_t>()[PATTERN_WORDS];
        for(uint32_t span=PRIOR_SEED_WORDS;span<PRIOR_PATTERN_WORDS;span*=2) {
            uint32_t length=PRIOR_PATTERN_WORDS-span<span?PRIOR_PATTERN_WORDS-span:span;
            Adds(priorWords[span],priorWords,int32_t(0),length);
            PipeBarrier<PIPE_V>();
        }
        if constexpr(K<=1024) {
            auto priorStreams=indexBuf.template Get<int32_t>()[PRIOR_STREAMS];
#pragma unroll
            for(uint32_t c=0;c<5;++c) {
                uint32_t wordOffset=c==0?0:8*(5-c);
                Adds(priorStreams[c*PRIOR_MASK_WORDS],priorWords[wordOffset],
                    int32_t(0),PRIOR_MASK_WORDS);
            }
            PipeBarrier<PIPE_V>();
        }
        Duplicate(onesBuf.template Get<float>(),2.f,64);
        Duplicate(outputMaskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),K/4);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        auto s=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorBuf.template Get<float>();auto o=outBuf.template Get<float>();
        auto idx=indexBuf.template Get<uint32_t>();auto r=RawInput();
        uint32_t tiles=(d.n+K-1)/K,cachedN=d.n;uint64_t reserved=0;
        bool needsRepair=true;
        uint32_t b=beginTile%d.batch,n=(beginTile/d.batch)*K;
        for(uint32_t tile=beginTile;tile<endTile;++tile) {
            uint32_t count=d.n-n<K?d.n-n:K;
            const bool reusePrior=n==cachedN;
            if(!reusePrior) {
                if(tile!=beginTile) {
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    CopyDenseInput(o,prior,n*5,count*5);
                }
                if(tile==beginTile)WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                else {SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);}
                UnpackContiguousPrior(p,o,count);
                // This predicate depends only on the cached priors and dtype.
                // The existing exact repair remains available for every tile.
                needsRepair=DirectStrideMax(p[2*P],v,count)>(sizeof(T)==2?4096.f:32.f);
                cachedN=n;
            }
            if(!Early || tile==beginTile) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr(sizeof(T)==2) {
                    if constexpr(HalfPrepack)UnpackHalfInput(r,count);
                    else {Cast(s,r,RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();}
                }
            }
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(HalfPrepack?5*P+count:count*6);
            Muls<float,false>(v,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Exp<float,false>(s,v,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(v,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Div<float,false>(s,onesBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
            PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
            // The next sigmoid uses S only; O remains owned by the previous
            // output DMA until its first new consumer, the even compaction.
            if(tile!=beginTile && reusePrior)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if constexpr(!HalfPrepack)SplitPlanes(s,o,v,count);
            if(__builtin_expect(needsRepair,false)) {
                RepairScaledDenseCold<T,K,S0,P0,V0>((GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            uint32_t nextRow=0,nextCount=0;
            if(tile+1<endTile) {
                uint32_t nextB=b+1,nextN=n;
                if(nextB==d.batch){nextB=0;nextN+=K;}
                nextRow=nextB*d.n+nextN;
                nextCount=d.n-nextN<K?d.n-nextN:K;
            }
            if(Early && nextCount) {
                // O was consumed by SplitPlanes. Suffix is a distinct buffer
                // beyond the active S planes; current decode never touches it.
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                StageNextRaw(nextRow,nextCount);
            }
            SetMaskCount();
            if(count*2>=P) {
                // The two center planes and two size planes are disjoint.
                // Each consumer follows a barrier after both producers.
                SetVectorMask<float,MaskMode::COUNTER>(P+count);
                Adds<float,false>(v,s,-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});
                Mul<float,false>(v[2*P],s[2*P],s[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                PipeBarrier<PIPE_V>();
                Add<float,false>(s,v,p,MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                Mul<float,false>(s[2*P],v[2*P],p[3*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                PipeBarrier<PIPE_V>();
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(count);
                for(uint32_t c=0;c<2;++c)Adds<float,false>(v[c*P],s[c*P],-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});
                for(uint32_t c=2;c<4;++c)Mul<float,false>(v[c*P],s[c*P],s[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                PipeBarrier<PIPE_V>();
                for(uint32_t c=0;c<2;++c)Add<float,false>(s[c*P],v[c*P],p[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                for(uint32_t c=2;c<4;++c)Mul<float,false>(s[c*P],v[c*P],p[(c+1)*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                PipeBarrier<PIPE_V>();
            }
            SetVectorMask<float,MaskMode::COUNTER>(count);
            for(uint32_t c=0;c<4;++c)Mul<float,false>(s[c*P],s[c*P],p[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            PipeBarrier<PIPE_V>();
            if constexpr(FullRawPrefetch) {
                Muls<float,false>(s[4*P],s[4*P],.25f,MASK_PLACEHOLDER,1,{1,1,8,8});
            } else {
                Mul<float,false>(v[4*P],s[4*P],s[5*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                PipeBarrier<PIPE_V>();
                Muls<float,false>(s[4*P],v[4*P],.25f,MASK_PLACEHOLDER,1,{1,1,8,8});
            }
            PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
            uint8_t repeats=static_cast<uint8_t>((count+15)/16);
            scatter_vnchwconv_b32(VA0,VA2,repeats,repeats==1?0:16,repeats==1?0:2);
            PipeBarrier<PIPE_V>();
            if(nextCount) {
                if constexpr(Early) {
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    PrepareNextInput(nextCount);
                } else {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    CopyDenseInput(r,raw,nextRow*6,nextCount*6);
                }
            }
            GatherMask(o,v,outputMaskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved);
            PipeBarrier<PIPE_V>();
            if constexpr(sizeof(T)==2) {Cast(outBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();}
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyDenseOutput(out,outBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            if(tile+1<endTile) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            if(++b==d.batch){b=0;n+=K;}
        }
        PipeBarrier<PIPE_ALL>();
    }
};

template<class T,uint32_t K,bool Early=true> class FlatMaskDecode {
    static constexpr uint32_t P=K+(sizeof(T)==4?24:8);
    static_assert(6*P<=7*K,"dense planes overlap raw prefetch suffix");
    static constexpr uint32_t P0=0,S0=P0+P*5*4,V0=S0+K*8*4+128,O0=V0+K*8*4+128;
    static constexpr uint32_t I0=O0+K*5*4+160,G0=I0+K*2*4,M0=G0+128*4;
    static constexpr uint32_t N0=S0+256+((M0+K-S0-256+511)/512)*512;
    static_assert(N0+K*4<=196352,"division numerator exceeds usable UB");
    static constexpr uint32_t MASK_WORDS=((3*K+255)/256)*8;
    static constexpr uint32_t CMP0=I0+K*4+3*MASK_WORDS*4;
    static_assert(CMP0+K/8<=I0+K*2*4,"wrap mask exceeds metadata storage");
    StaticUBBuffer<CMP0,K/8> wrapMaskBuf;
    static constexpr uint32_t PATTERN_WORDS=((MASK_WORDS+16+23)/24)*24;
    static constexpr uint32_t SEED_WORDS=24;
    static constexpr uint32_t PRIOR_MASK_WORDS=((5*(K+7)+255)/256)*8;
    static constexpr uint32_t PRIOR_PATTERN_WORDS=((PRIOR_MASK_WORDS+32+39)/40)*40;
    static constexpr uint32_t PRIOR_SEED_WORDS=40;
    static constexpr uint32_t PRIOR_STREAMS=PATTERN_WORDS+PRIOR_PATTERN_WORDS;
    static_assert(PRIOR_STREAMS+5*PRIOR_MASK_WORDS<=2*K,
        "independent prior compaction masks exceed index storage");
    static_assert(K*4+3*MASK_WORDS*4<=K*8,"mask metadata exceeds index storage");
    static_assert(PRIOR_STREAMS+5*PRIOR_MASK_WORDS<=(CMP0-I0)/4,
        "periodic patterns overlap wrap scratch");
    StaticUBBuffer<P0,P*5*4> priorBuf;
    StaticUBBuffer<S0,K*7*4> sigBuf;
    StaticUBBuffer<S0+K*7*4,K*4> rawSuffixBuf;
    StaticUBBuffer<V0,K*8*4> workBuf;
    StaticUBBuffer<O0,K*5*4+160> outBuf;
    StaticUBBuffer<I0,K*2*4> indexBuf;
    StaticUBBuffer<G0,64*4> onesBuf;
    StaticUBBuffer<M0,K> outputMaskBuf;
    GlobalTensor<T> raw,out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
    uint32_t beginRow,endRow;
    uint64_t outSrc[16],outDst[16];

    __aicore__ inline LocalTensor<T> RawInput() {
        auto result=sigBuf.template Get<T>();
        if constexpr(sizeof(T)==2)result=result[8*K];
        return result;
    }
    __aicore__ inline void StageNextRaw(uint32_t row,uint32_t count) {
        uint32_t elements=count*6;
        if constexpr(sizeof(T)==2) {
            CopyDenseInput(outBuf.template Get<T>(),raw,row*6,elements);
        } else {
            uint32_t first=elements<5*K?elements:5*K;
            CopyDenseInput(outBuf.template Get<T>(),raw,row*6,first);
            if(elements>first)
                CopyDenseInput(rawSuffixBuf.template Get<T>(),raw,row*6+first,elements-first);
        }
    }
    __aicore__ inline void PrepareNextInput(uint32_t count) {
        auto next=sigBuf.template Get<float>();
        if constexpr(sizeof(T)==2) {
            auto bits=outBuf.template Get<int16_t>();
            Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
            Cast(next,outBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);
            PipeBarrier<PIPE_V>();
        } else {
            uint32_t elements=count*6,first=elements<5*K?elements:5*K;
            auto target=next.template ReinterpretCast<int32_t>();
            Adds(target,outBuf.template Get<int32_t>(),int32_t(0),first);
            PipeBarrier<PIPE_V>();
            if(elements>first) {
                Adds(target[first],rawSuffixBuf.template Get<int32_t>(),int32_t(0),elements-first);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void SetPriorHeads(uint64_t keep) {
        constexpr uint32_t bits[5]={0x42108421U,0x10842108U,
            0x84210842U,0x21084210U,0x08421084U};
        auto masks=indexBuf.template Get<uint64_t>()[PRIOR_STREAMS/2];
#pragma unroll
        for(uint32_t c=0;c<5;++c) {
            uint32_t wordOffset=c==0?0:8*(5-c);
            uint64_t head=uint64_t(bits[wordOffset%5])|
                (uint64_t(bits[(wordOffset+1)%5])<<32);
            masks.SetValue(c*PRIOR_MASK_WORDS/2,head&keep);
        }
    }
    template<bool Restore=true,bool PreserveCounter=false>
    __aicore__ inline void UnpackContiguousPrior(LocalTensor<float>& destination,
        LocalTensor<float>& packed,uint32_t count,uint32_t prefix=0) {
        if(prefix) {
            // Each channel owns its mask head, so clearing one head cannot
            // remove valid middle rows from another channel's mask stream.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            SetPriorHeads(~uint64_t(0)<<(5*prefix));
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
        static_assert(!(Restore && PreserveCounter),"counter preservation needs immutable one-tile heads");
        if constexpr(!PreserveCounter) SetMaskCount();
        SetVectorMask<float,MaskMode::COUNTER>(5*(count+prefix));
        auto patterns=indexBuf.template Get<float>()[PRIOR_STREAMS];
#pragma unroll
        for(uint32_t c=0;c<5;++c) {
            GatherMask(destination[c*P],packed,patterns[c*PRIOR_MASK_WORDS],uint8_t(0),
                GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();
        if constexpr(!PreserveCounter) {ResetMask();SetMaskNorm();}
        if constexpr(Restore) {
            if(prefix) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                SetPriorHeads(~uint64_t(0));
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
        }
    }
    __aicore__ inline void SplitPlanes(LocalTensor<float>& values,LocalTensor<float>& even,
        LocalTensor<float>& odd,uint32_t count) {
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*count);
        GatherMask(even,values,uint8_t(1),GatherMaskParams{1,1,8,0});
        GatherMask(odd,values,uint8_t(2),GatherMaskParams{1,1,8,0});
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::COUNTER>(3*count);
        auto patterns=indexBuf.template Get<float>();
#pragma unroll
        for(uint32_t phase=0;phase<2;++phase) {
            auto mask=patterns[phase==0?0:(phase==1?16:8)];
            GatherMask(values[(2*phase)*P],even,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
            GatherMask(values[(2*phase+1)*P],odd,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();
        // Pair objectness and class confidence in the even/odd streams.
        // Only phase two of this product is consumed by the output.
        Mul<float,false>(even,even,odd,MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
        PipeBarrier<PIPE_V>();
        GatherMask(values[4*P],even,patterns[8],uint8_t(0),GatherMaskParams{1,1,8,0});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline uint32_t PriorPrefix(uint32_t n,uint32_t count) {
        uint32_t first=d.n-n;
        return first<count?((first+7)/8*8-first):0;
    }
    __aicore__ inline void StagePrior(uint32_t n,uint32_t count) {
        auto packed=outBuf.template Get<float>();
        uint32_t first=d.n-n;
        if(first>=count)CopyDenseInput(packed,prior,n*5,count*5);
        else {
            uint32_t prefix=PriorPrefix(n,count),length=first+prefix;
            CopyDenseInput(packed,prior,(n-prefix)*5,length*5);
            CopyDenseInput(packed[length*5],prior,0,(count-first)*5);
        }
    }
    template<bool Restore=true,bool PreserveCounter=false>
    __aicore__ inline void UnpackPrior(LocalTensor<float>& p,LocalTensor<float>& packed,
        LocalTensor<uint32_t>& indices,uint32_t n,uint32_t count) {
        uint32_t prefix=PriorPrefix(n,count);
        UnpackContiguousPrior<Restore,PreserveCounter>(p,packed,count,prefix);
    }
public:
    template<bool One=false>
    __aicore__ inline void Init(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t) {
        d=t;InitSocState();ResetMask();SetMaskNorm();
        raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);out.SetGlobalBuffer((__gm__ T*)o);
        auto values=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();auto temp=outBuf.template Get<float>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        constexpr uint32_t rowAlign=sizeof(T)==2?16:8;
        const uint32_t owner=GetBlockIdx();
        const uint32_t firstUnit=owner*d.unitsPerWorker+
            (owner<d.extraUnits?owner:d.extraUnits);
        const uint32_t ownerUnits=d.unitsPerWorker+(owner<d.extraUnits);
        beginRow=firstUnit*rowAlign;
        endRow=beginRow+ownerUnits*rowAlign;
        if(endRow>d.totalRows)endRow=d.totalRows;
        uint32_t firstRow=beginRow,firstN=firstRow%d.n;
        uint32_t firstCount=endRow-firstRow<K?endRow-firstRow:K;
        StagePrior(firstN,firstCount);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        CopyDenseInput(RawInput(),raw,firstRow*6,firstCount*6);
#pragma unroll
        for(uint32_t c=0;c<8;++c) {
            uint32_t channel=c<5?c:0;
            outSrc[c]=reinterpret_cast<uint64_t>(values[channel*P].GetPhyAddr());
            outSrc[c+8]=reinterpret_cast<uint64_t>(values[channel*P+8].GetPhyAddr());
            outDst[2*c]=reinterpret_cast<uint64_t>(v[c*8].GetPhyAddr());
            outDst[2*c+1]=reinterpret_cast<uint64_t>(v[(c+8)*8].GetPhyAddr());
        }
        // The dense path has one fixed transpose layout. No other operation
        // in this path writes VA0..VA3, including the numerical repair.
        set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
        set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);

        constexpr uint32_t bits[3]={0x49249249U,0x92492492U,0x24924924U};
        constexpr uint64_t rawLanes[3]={0x249249ULL,0x492492ULL,0x924924ULL};
        constexpr uint32_t priorBits[5]={0x42108421U,0x10842108U,
            0x84210842U,0x21084210U,0x08421084U};
        constexpr uint64_t priorLanes[5]={0x0842108421ULL,0x1084210842ULL,
            0x2108421084ULL,0x4210842108ULL,0x8421084210ULL};
        auto rawSeeds=indexBuf.template Get<int32_t>();
        auto priorSeeds=indexBuf.template Get<int32_t>()[PATTERN_WORDS];
        // A repeat advances by one full period group: 24 or 40 words.
        // The normal masks select only those first 24/40 lanes.
        SetMaskNorm();
#pragma unroll
        for(uint32_t c=0;c<3;++c) {
            SetVectorMask<int32_t,MaskMode::NORMAL>(0,rawLanes[c]);
            Duplicate<int32_t,false>(rawSeeds,static_cast<int32_t>(bits[c]),
                MASK_PLACEHOLDER,PATTERN_WORDS/24,1,3);
        }
#pragma unroll
        for(uint32_t c=0;c<5;++c) {
            SetVectorMask<int32_t,MaskMode::NORMAL>(0,priorLanes[c]);
            Duplicate<int32_t,false>(priorSeeds,static_cast<int32_t>(priorBits[c]),
                MASK_PLACEHOLDER,PRIOR_PATTERN_WORDS/40,1,5);
        }
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
        SetMaskCount();
        SetVectorMask<int32_t,MaskMode::COUNTER>(PRIOR_MASK_WORDS);
        auto priorWords=indexBuf.template Get<int32_t>()[PATTERN_WORDS];
        auto priorStreams=indexBuf.template Get<int32_t>()[PRIOR_STREAMS];
#pragma unroll
        for(uint32_t c=0;c<5;++c) {
            uint32_t wordOffset=c==0?0:8*(5-c);
            Adds<int32_t,false>(priorStreams[c*PRIOR_MASK_WORDS],priorWords[wordOffset],
                int32_t(0),MASK_PLACEHOLDER,1,{1,1,8,8});
        }
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::COUNTER>(64);
        Duplicate<float,false>(onesBuf.template Get<float>(),2.f,MASK_PLACEHOLDER,1,1,8);
        SetVectorMask<int32_t,MaskMode::COUNTER>(K/4);
        Duplicate<int32_t,false>(outputMaskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),MASK_PLACEHOLDER,1,1,8);
        PipeBarrier<PIPE_V>();
        if constexpr(!One) {ResetMask();SetMaskNorm();}
    }
    __aicore__ inline void ProcessOne() {
        auto s=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorBuf.template Get<float>();auto o=outBuf.template Get<float>();
        auto idx=indexBuf.template Get<uint32_t>();auto r=RawInput();
        uint32_t row=beginRow,count=endRow-beginRow;
        if(!count) {ResetMask();SetMaskNorm();return;}
        uint32_t n=row%d.n;uint64_t reserved=0;
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // This worker has no subsequent prior compaction.
        // Init and this compaction share counter mode. The following
        // stride guard establishes normal mode before its first reduction.
        UnpackPrior<false,true>(p,o,idx,n,count);
        static_assert(G0+128*4<=M0,"one-tile guard exceeds constant gap");
        LocalTensor<float> guard(TPosition::VECCALC,G0+64*4,64);
        BeginOneStrideMax(p[2*P],guard,count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr(sizeof(T)==2) {
            auto bits=r.template ReinterpretCast<int16_t>();
            Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
            Cast(s,r,RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();}
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
        if constexpr(sizeof(T)==4) {Muls<float,false>(v,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();}
        Exp<float,false>(s,sizeof(T)==2?s:v,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(v,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(s,onesBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
        PipeBarrier<PIPE_V>();
        SplitPlanes(s,o,v,count);
        bool needsRepair=FinishOneStrideMax(guard,count)>(sizeof(T)==2?4096.f:32.f);
        if(__builtin_expect(needsRepair,false)) {
            RepairScaledDenseCold<T,K,S0,P0,V0>((GM_ADDR)raw.GetPhyAddr(),count,row*6);
            set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
            set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
        }
        SetMaskCount();
        if(count*2>=P) {
            // The two center planes and two size planes are disjoint.
            // Each consumer follows a barrier after both producers.
            SetVectorMask<float,MaskMode::COUNTER>(P+count);
            Adds<float,false>(v,s,-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});
            Mul<float,false>(v[2*P],s[2*P],s[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            PipeBarrier<PIPE_V>();
            Add<float,false>(s,v,p,MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            Mul<float,false>(s[2*P],v[2*P],p[3*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            PipeBarrier<PIPE_V>();
        } else {
            SetVectorMask<float,MaskMode::COUNTER>(count);
            for(uint32_t c=0;c<2;++c)Adds<float,false>(v[c*P],s[c*P],-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});
            for(uint32_t c=2;c<4;++c)Mul<float,false>(v[c*P],s[c*P],s[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<2;++c)Add<float,false>(s[c*P],v[c*P],p[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            for(uint32_t c=2;c<4;++c)Mul<float,false>(s[c*P],v[c*P],p[(c+1)*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float,MaskMode::COUNTER>(count);
        for(uint32_t c=0;c<4;++c)Mul<float,false>(s[c*P],s[c*P],p[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
        PipeBarrier<PIPE_V>();
        Muls<float,false>(s[4*P],s[4*P],.25f,MASK_PLACEHOLDER,1,{1,1,8,8});
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
        uint8_t repeats=static_cast<uint8_t>((count+15)/16);
        scatter_vnchwconv_b32(VA0,VA2,repeats,repeats==1?0:16,repeats==1?0:2);
        PipeBarrier<PIPE_V>();
        GatherMask(o,v,outputMaskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved);
        PipeBarrier<PIPE_V>();
        if constexpr(sizeof(T)==2) {Cast(outBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();}
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        CopyDenseOutput(out,outBuf.template Get<T>(),row*5,count*5);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process() {
        auto s=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorBuf.template Get<float>();auto o=outBuf.template Get<float>();
        auto idx=indexBuf.template Get<uint32_t>();auto r=RawInput();
        uint32_t cachedN=d.n,cachedCount=0;uint64_t reserved=0;
        bool needsRepair=true;
        for(uint32_t row=beginRow;row<endRow;row+=K) {
            uint32_t n=row%d.n;
            uint32_t count=endRow-row<K?endRow-row:K;
            const bool reusePrior=n==cachedN && count==cachedCount;
            if(!reusePrior) {
                if(row!=beginRow) {
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    StagePrior(n,count);
                }
                if(row==beginRow)WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                else {SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);}
                UnpackPrior(p,o,idx,n,count);
                // This predicate depends only on the cached priors and dtype.
                // The existing exact repair remains available for every tile.
                needsRepair=DirectStrideMax(p[2*P],v,count)>(sizeof(T)==2?4096.f:32.f);
                cachedN=n;cachedCount=count;
            }
            if(!Early || row==beginRow) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr(sizeof(T)==2) {
            auto bits=r.template ReinterpretCast<int16_t>();
            Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
            Cast(s,r,RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();}
            }
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
            if constexpr(sizeof(T)==4) {Muls<float,false>(v,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();}
            Exp<float,false>(s,sizeof(T)==2?s:v,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(v,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Div<float,false>(s,onesBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
            PipeBarrier<PIPE_V>();
            // The next sigmoid uses S only; O remains owned by the previous
            // output DMA until its first new consumer, the even compaction.
            if(row!=beginRow && reusePrior)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SplitPlanes(s,o,v,count);
            if(__builtin_expect(needsRepair,false)) {
                RepairScaledDenseCold<T,K,S0,P0,V0>((GM_ADDR)raw.GetPhyAddr(),count,row*6);
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            uint32_t nextRow=row+K,nextCount=0;
            if(nextRow<endRow)nextCount=endRow-nextRow<K?endRow-nextRow:K;
            if(Early && nextCount) {
                // O was consumed by SplitPlanes. Suffix is a distinct buffer
                // beyond the active S planes; current decode never touches it.
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                StageNextRaw(nextRow,nextCount);
            }
            SetMaskCount();
            if(count*2>=P) {
                // The two center planes and two size planes are disjoint.
                // Each consumer follows a barrier after both producers.
                SetVectorMask<float,MaskMode::COUNTER>(P+count);
                Adds<float,false>(v,s,-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});
                Mul<float,false>(v[2*P],s[2*P],s[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                PipeBarrier<PIPE_V>();
                Add<float,false>(s,v,p,MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                Mul<float,false>(s[2*P],v[2*P],p[3*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                PipeBarrier<PIPE_V>();
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(count);
                for(uint32_t c=0;c<2;++c)Adds<float,false>(v[c*P],s[c*P],-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});
                for(uint32_t c=2;c<4;++c)Mul<float,false>(v[c*P],s[c*P],s[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                PipeBarrier<PIPE_V>();
                for(uint32_t c=0;c<2;++c)Add<float,false>(s[c*P],v[c*P],p[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                for(uint32_t c=2;c<4;++c)Mul<float,false>(s[c*P],v[c*P],p[(c+1)*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
                PipeBarrier<PIPE_V>();
            }
            SetVectorMask<float,MaskMode::COUNTER>(count);
            for(uint32_t c=0;c<4;++c)Mul<float,false>(s[c*P],s[c*P],p[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            PipeBarrier<PIPE_V>();
            Muls<float,false>(s[4*P],s[4*P],.25f,MASK_PLACEHOLDER,1,{1,1,8,8});
            PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
            uint8_t repeats=static_cast<uint8_t>((count+15)/16);
            scatter_vnchwconv_b32(VA0,VA2,repeats,repeats==1?0:16,repeats==1?0:2);
            PipeBarrier<PIPE_V>();
            if(nextCount) {
                if constexpr(Early) {
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    PrepareNextInput(nextCount);
                } else {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    CopyDenseInput(r,raw,nextRow*6,nextCount*6);
                }
            }
            GatherMask(o,v,outputMaskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved);
            PipeBarrier<PIPE_V>();
            if constexpr(sizeof(T)==2) {Cast(outBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();}
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyDenseOutput(out,outBuf.template Get<T>(),row*5,count*5);
            if(row+K<endRow) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
};


#include "blocked_transpose.h"

template<class T,uint32_t K,bool Early>
__aicore__ __attribute__((noinline)) void RunDenseShortN(GM_ADDR r,GM_ADDR p,
    GM_ADDR o,uint32_t batch,uint32_t n,uint32_t blocks,uint32_t tile) {
    FireCandidateDecodeFusionTilingData t{batch,n,blocks,tile};
    MaskDecode<T,K,Early> op;op.Init(r,p,o,t);op.Process();
}

template<class T,uint32_t K,bool Early>
__aicore__ __attribute__((noinline)) void RunDenseManyTiles(GM_ADDR r,GM_ADDR p,
    GM_ADDR o,uint32_t batch,uint32_t n,uint32_t blocks,uint32_t tile,
    uint32_t unitsPerWorker,uint32_t extraUnits,uint32_t totalRows) {
    FireCandidateDecodeFusionTilingData t{batch,n,blocks,tile,
        unitsPerWorker,extraUnits,0,totalRows};
    FlatMaskDecode<T,K,Early> op;op.Init(r,p,o,t);op.Process();
}

template<class T,uint32_t K,bool Early>
__aicore__ inline void RunDenseDecode(GM_ADDR r,GM_ADDR p,GM_ADDR o,
    FireCandidateDecodeFusionTilingData t) {
    if constexpr(K<=1024) {
        if(__builtin_expect(t.denseOne!=0,true)) {
            FlatMaskDecode<T,K,Early> op;op.template Init<true>(r,p,o,t);op.ProcessOne();
        } else if(t.n>=K) {
            RunDenseManyTiles<T,K,Early>(r,p,o,t.batch,t.n,t.blocks,t.tile,
                t.unitsPerWorker,t.extraUnits,t.totalRows);
        } else {
            RunDenseShortN<T,K,Early>(r,p,o,t.batch,t.n,t.blocks,t.tile);
        }
    } else {BlockedTransposeDecode<T,K> op;op.Init(r,p,o,t);op.Process();}
}

