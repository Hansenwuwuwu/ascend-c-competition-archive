#pragma once
#include <type_traits>

// This is the input type of the compiled binary, independent of each TPL
// instantiation. Keep the complete Half image on its original schedule.
#if defined(DTYPE_RAW_PRED) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
static constexpr bool FireInputFloatCompletionJoins = sizeof(DTYPE_RAW_PRED) == 4;
#else
static constexpr bool FireInputFloatCompletionJoins = false;
#endif

template<class T,uint32_t K,uint32_t P,uint32_t G,uint32_t Q,uint32_t S0,uint32_t V0>
// Preserve the next cold resource's measured instruction placement after
// removing manual padding from the preceding shared fallback.
__aicore__ __attribute__((noinline, aligned(
    FireInputFloatCompletionJoins && sizeof(T) == 4 && K == 1664 ? 128 : 4)))
void RepairBlockedCenters(GM_ADDR rawAddress,
    uint32_t count,uint32_t rawOffset) {
    LocalTensor<float> values(TPosition::VECCALC,S0,6*P);
    LocalTensor<float> priors(TPosition::VECCALC,0,5*P);
    LocalTensor<float> scratch(TPosition::VECCALC,V0,6*P);
    GlobalTensor<T> raw;raw.SetGlobalBuffer((__gm__ T*)rawAddress);
    for(uint32_t channel=0;channel<2;++channel) {
        ResetMask();SetMaskNorm();
        Adds(scratch,values[channel*P],-.5f,P);PipeBarrier<PIPE_V>();
        Add(scratch,scratch,priors[channel*P],P);PipeBarrier<PIPE_V>();
        Abs(scratch,scratch,P);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        for(uint32_t j=0;j<G*Q;++j) {
            const uint32_t row=(j%Q)*G+j/Q;
            if(row>=count || scratch.GetValue(j)>=0x1p-4f ||
                priors.GetValue(2*P+j)<=(sizeof(T)==2?4096.f:32.f))continue;
            const float original=static_cast<float>(raw.GetValue(uint64_t(rawOffset)+row*6+channel));
            if(original>-20.f && original<0.f) {
                union {float f;uint32_t u;} x;x.f=-original;
                const uint32_t expBits=FireSoft53::evaluate(x.u);
                const uint32_t denominator=FireSoft53::narrow32(FireSoft53::add53(
                    FireSoft53::from32(expBits),FireSoft53::integer(1)));
                values.template ReinterpretCast<uint32_t>().SetValue(channel*P+j,
                    CenterReciprocalBits(denominator)+0x800000u);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
}

// A matrix row contains G adjacent candidates: its raw/prior/output widths
// are whole32-byte blocks. Transpose each block column into the same phase
// permutation in every channel; arithmetic never mixes different candidates.
template<class T,uint32_t K,bool BatchOne=false,bool UniformFull=false,bool DualOutput=false,bool UniformTiles=false> class BlockedTransposeDecode {
    static constexpr bool LargeMetadata=!BatchOne && K>1024;
    static constexpr bool NativeHalfPlanes=sizeof(T)==2 && BatchOne;
    static constexpr bool FloatLayout=sizeof(T)==4 || BatchOne;
    static constexpr bool HalfEight=sizeof(T)==2 && !BatchOne && K>1024;
    static constexpr bool WidePhaseLayout=FloatLayout || HalfEight;
    using ShuffleT=std::conditional_t<FloatLayout,float,T>;
    using OutputShuffleT=std::conditional_t<WidePhaseLayout,float,T>;
    static constexpr uint32_t RAWBLOCKS=HalfEight?3:6;
    static constexpr uint32_t RAW_SOURCE_STRIDE=HalfEight?48:96;
    static constexpr uint32_t G=WidePhaseLayout?8:16;
    static constexpr uint32_t Q=(K+16*G-1)/(16*G)*16;
    static constexpr uint32_t P=G*Q+(NativeHalfPlanes?16:(FloatLayout?24:16));
    static constexpr uint32_t S0=5*P*4+128+(HalfEight?320:0),V0=S0+6*P*4+128;
    static constexpr uint32_t O0=V0+6*P*4+128;
    static constexpr uint32_t RAW_WORDS=(WidePhaseLayout?6:5)*P;
    static constexpr uint32_t F0=O0+RAW_WORDS*4+128,C0=F0+5*P*4+128;
    static_assert(C0+64*4<=196352,"blocked transpose exceeds usable UB");
    static constexpr uint32_t A0=C0+64*4;
    static constexpr bool DirectScoreDivide=sizeof(T)==4 && !BatchOne && K>1024;
#if __NPU_ARCH__ == 2201
    static constexpr bool ScoreGeometryCounterCarry=(DirectScoreDivide && !DualOutput) || HalfEight;
#else
    static constexpr bool ScoreGeometryCounterCarry=false;
#endif
    static_assert(!DualOutput || DirectScoreDivide,"dual output requires large Float resources");
    StaticUBBuffer<A0,(DirectScoreDivide?64*4:0)> scoreNumeratorBuf;
    static_assert(!DirectScoreDivide || A0+64*4<=196352,"score numerator exceeds UB");
    static constexpr uint32_t LargeContexts=HalfEight?18:21;
    static constexpr uint32_t MetadataBase=A0+(DirectScoreDivide?256:0);
    static_assert((!BatchOne && !LargeMetadata) || MetadataBase+(LargeMetadata?LargeContexts*128:16*128)<=196352,"VA metadata exceeds UB");
    StaticUBBuffer<MetadataBase,(LargeMetadata?LargeContexts*128:(BatchOne?16*128:0))> addressBuf;
    static constexpr uint32_t PRBLOCKS=5;
    StaticUBBuffer<0,5*P*4> priorsBuf;
    StaticUBBuffer<S0,6*P*4> valuesBuf;
    StaticUBBuffer<V0,6*P*4> workBuf;
    StaticUBBuffer<O0,RAW_WORDS*4> packedBuf;
    StaticUBBuffer<F0,5*P*4> priorPackedBuf;
    StaticUBBuffer<C0,64*4> numeratorBuf;
    GlobalTensor<T> raw,out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
    uint32_t beginTile,endTile;
    uint32_t oneN=0,oneCount=0,oneBatch=0;
    bool stablePriorOwner;
    uint64_t rawSrc[6][16],rawDst[6][16];
    uint64_t priorSrc[PRBLOCKS][16],priorDst[PRBLOCKS][16];
    uint64_t outputSrc[5][16],outputDst[5][16];

    template<bool Second=false>
    __aicore__ inline void LoadAddressContext(uint32_t index) {
        auto words=addressBuf.template Get<uint64_t>()[index*((LargeMetadata || BatchOne)?16:8)];
        auto source=(__ubuf__ uint64_t*)words.GetPhyAddr();
        if constexpr(LargeMetadata || BatchOne) {
            if constexpr(Second) {
                vld_va_reg(VA4,source,L128);vld_va_reg(VA5,source+4,L128);
                vld_va_reg(VA6,source+8,L128);vld_va_reg(VA7,source+12,L128);
            } else {
                vld_va_reg(VA0,source,L128);vld_va_reg(VA1,source+4,L128);
                vld_va_reg(VA2,source+8,L128);vld_va_reg(VA3,source+12,L128);
            }
        } else {
            auto destination=(__ubuf__ uint64_t*)words[4].GetPhyAddr();
            if constexpr(Second) {
                vld_va_reg(VA4,source,L128);vld_va_reg(VA5,source,H128);
                vld_va_reg(VA6,destination,L128);vld_va_reg(VA7,destination,H128);
            } else {
                vld_va_reg(VA0,source,L128);vld_va_reg(VA1,source,H128);
                vld_va_reg(VA2,destination,L128);vld_va_reg(VA3,destination,H128);
            }
        }
    }
    __aicore__ inline void StagePrior(uint32_t n,uint32_t count) {
        auto packed=priorPackedBuf.template Get<float>();
        const uint32_t padded=(count*5+7)/8*8;
        const uint32_t consumed=(count+16*G-1)/(16*G)*16*G*5;
        if constexpr(!BatchOne) {
            if(consumed>padded) {Duplicate(packed[padded],0.f,consumed-padded);PipeBarrier<PIPE_V>();}
        }
        // The previous stride guard uses the released packed-prior slot.
        // Its Scalar read completion is fenced inside DirectStrideMax.
        if constexpr(!BatchOne) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }
        CopyDenseInput(packed,prior,n*5,count*5);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }
    __aicore__ inline void LoadRaw(uint32_t row,uint32_t count) {
        if constexpr(UniformTiles && !DualOutput) {
            auto source=raw[row*6];
            CopyDenseInput(packedBuf.template Get<T>(),source,0,count*6);
        } else if constexpr(sizeof(T)==2 && BatchOne)
            CopyDenseInput(workBuf.template Get<T>(),raw,row*6,count*6);
        else CopyDenseInput(packedBuf.template Get<T>(),raw,row*6,count*6);
    }
    __aicore__ inline void RetainHalfContexts() {
        if constexpr(!FloatLayout && !LargeMetadata) {
            set_va_reg_sb(VA4,rawSrc[0]);set_va_reg_sb(VA5,rawSrc[0]+8);
            set_va_reg_sb(VA6,outputSrc[0]);set_va_reg_sb(VA7,outputSrc[0]+8);
        }
    }
    __aicore__ inline void EvaluatePackedRaw(uint32_t count) {
        auto input=packedBuf.template Get<float>();
        auto scratch=workBuf.template Get<float>();
        if constexpr(sizeof(T)==2) {
            Cast(input,workBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);
            PipeBarrier<PIPE_V>();
        }
        // Pointwise FP32 arithmetic commutes with the exact row permutation.
        // Only the valid contiguous stream participates in the sigmoid.
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
        Muls<float,false>(scratch,input,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Exp<float,false>(input,scratch,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(scratch,input,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(input,numeratorBuf.template Get<float>(),scratch,
            MASK_PLACEHOLDER,1,{1,1,1,8,0,8});PipeBarrier<PIPE_V>();
        ResetMask();SetMaskNorm();
    }
    __aicore__ inline void EvaluateFullHalfPlanes() {
        auto values=valuesBuf.template Get<float>();
        auto scratch=workBuf.template Get<float>();
        auto bits=packedBuf.template Get<int16_t>();
        SetMaskCount();SetVectorMask<int16_t,MaskMode::COUNTER>(6*P);
        Adds<int16_t,false>(bits,bits,static_cast<int16_t>(-32768),MASK_PLACEHOLDER,1,{1,1,8,8});
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
        Cast(scratch,packedBuf.template Get<T>(),RoundMode::CAST_NONE,6*P);
        PipeBarrier<PIPE_V>();
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
        Exp<float,false>(values,scratch,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(scratch,values,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(values,numeratorBuf.template Get<float>(),scratch,
            MASK_PLACEHOLDER,1,{1,1,1,8,0,8});PipeBarrier<PIPE_V>();
        ResetMask();SetMaskNorm();
    }
    template<bool FinalFence=true>
    __aicore__ inline void UnpackRaw(uint8_t repeats) {
        static_assert(FinalFence || (LargeMetadata && FloatLayout && !HalfEight && !BatchOne),
            "unpack completion requires the Hot large-Float event/mask joins");
        if constexpr(NativeHalfPlanes) {
            LoadAddressContext<false>(0);LoadAddressContext<true>(1);
            scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:1,repeats==1?0:48);
            scatter_vnchwconv_b16(VA6,VA4,repeats,repeats==1?0:1,repeats==1?0:48);
            LoadAddressContext<false>(2);
            scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:1,repeats==1?0:48);
            PipeBarrier<PIPE_V>();
            return;
        }
        if constexpr(HalfEight) {
            LoadAddressContext<false>(0);LoadAddressContext<true>(1);
            scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:1,repeats==1?0:48);
            scatter_vnchwconv_b16(VA6,VA4,repeats,repeats==1?0:1,repeats==1?0:48);
            LoadAddressContext<false>(2);
            scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:1,repeats==1?0:48);
            PipeBarrier<PIPE_V>();
            return;
        }
        if constexpr(BatchOne || LargeMetadata) {
            // Both metadata reads precede the first transpose. The two
            // matrices use disjoint VA registers and UB destinations.
#pragma unroll
            for(uint32_t block=0;block<6;block+=2) {
                LoadAddressContext<false>(block);
                LoadAddressContext<true>(block+1);
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:96);
                scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:2,repeats==1?0:96);
            }
            if constexpr(FinalFence)PipeBarrier<PIPE_V>();
            return;
        }
        if constexpr(FloatLayout && !BatchOne) {
            // Install two disjoint VA contexts before either transpose.
            // Reusing the other bank does not overwrite a queued context.
#pragma unroll
            for(uint32_t block=0;block<6;block+=2) {
                set_va_reg_sb(VA0,rawSrc[block]);set_va_reg_sb(VA1,rawSrc[block]+8);
                set_va_reg_sb(VA2,rawDst[block]);set_va_reg_sb(VA3,rawDst[block]+8);
                set_va_reg_sb(VA4,rawSrc[block+1]);set_va_reg_sb(VA5,rawSrc[block+1]+8);
                set_va_reg_sb(VA6,rawDst[block+1]);set_va_reg_sb(VA7,rawDst[block+1]+8);
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:96);
                scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:2,repeats==1?0:96);
            }
            PipeBarrier<PIPE_V>();
            return;
        }
#pragma unroll
        for(uint32_t block=0;block<RAWBLOCKS;++block) {
            if constexpr(BatchOne)LoadAddressContext(block);
            else {
            if constexpr(!FloatLayout) {
                if(block!=0) {set_va_reg_sb(VA0,rawSrc[block]);set_va_reg_sb(VA1,rawSrc[block]+8);}
            } else {set_va_reg_sb(VA0,rawSrc[block]);set_va_reg_sb(VA1,rawSrc[block]+8);}
            set_va_reg_sb(VA2,rawDst[block]);set_va_reg_sb(VA3,rawDst[block]+8);
            }
            if constexpr(!FloatLayout) {
                if(block==0)scatter_vnchwconv_b16(VA2,VA4,repeats,repeats==1?0:1,repeats==1?0:RAW_SOURCE_STRIDE);
                else scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:1,repeats==1?0:RAW_SOURCE_STRIDE);
            } else
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:96);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void UnpackPriors(uint8_t repeats) {
        if constexpr(BatchOne || LargeMetadata) {
            constexpr uint32_t context=HalfEight?3:6;
#pragma unroll
            for(uint32_t block=0;block+1<PRBLOCKS;block+=2) {
                LoadAddressContext<false>(context+block);
                LoadAddressContext<true>(context+1+block);
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:10*G);
                scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:2,repeats==1?0:10*G);
            }
            if constexpr(PRBLOCKS%2) {
                LoadAddressContext<false>(context+PRBLOCKS-1);
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:10*G);
            }
            PipeBarrier<PIPE_V>();
            return;
        }
#pragma unroll
        for(uint32_t block=0;block<PRBLOCKS;++block) {
            if constexpr(BatchOne)LoadAddressContext(6+block);
            else {
            set_va_reg_sb(VA0,priorSrc[block]);set_va_reg_sb(VA1,priorSrc[block]+8);
            set_va_reg_sb(VA2,priorDst[block]);set_va_reg_sb(VA3,priorDst[block]+8);
            }
            // The two VA destination halves select the low/high eight
            // phases. Half owners use twice as many eight-row repeats.
            const uint8_t priorRepeats=WidePhaseLayout?repeats:2*repeats;
            scatter_vnchwconv_b32(VA2,VA0,priorRepeats,
                priorRepeats==1?0:(WidePhaseLayout?2:1),
                priorRepeats==1?0:(WidePhaseLayout?10*G:80));
        }
        PipeBarrier<PIPE_V>();
    }
    template<uint32_t Context>
    __aicore__ inline __attribute__((always_inline)) void PackOutputWithContext(uint8_t repeats) {
        static_assert(Context+5<=LargeContexts,"output address contexts exceed metadata");
#pragma unroll
        for(uint32_t block=0;block+1<5;block+=2) {
            LoadAddressContext<false>(Context+block);
            LoadAddressContext<true>(Context+1+block);
            scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
            scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:80,repeats==1?0:2);
        }
        LoadAddressContext<false>(Context+4);
        scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PackOutput(uint8_t repeats) {
        if constexpr(HalfEight) {
            // Keep both legal owner layouts; materialize their addresses locally.
            if(stablePriorOwner)PackOutputWithContext<8>(repeats);
            else PackOutputWithContext<13>(repeats);
            return;
        }
        if constexpr(LargeMetadata && sizeof(T)==4) {
            // Both original owner layouts use locally materialized addresses.
            if(stablePriorOwner)PackOutputWithContext<11>(repeats);
            else PackOutputWithContext<16>(repeats);
            return;
        }
        if constexpr(BatchOne || LargeMetadata) {
            const uint32_t context=HalfEight?(stablePriorOwner?8:13):(LargeMetadata && !stablePriorOwner?16:11);
#pragma unroll
            for(uint32_t block=0;block+1<5;block+=2) {
                LoadAddressContext<false>(context+block);
                LoadAddressContext<true>(context+1+block);
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
                scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:80,repeats==1?0:2);
            }
            LoadAddressContext<false>(context+4);
            scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
            PipeBarrier<PIPE_V>();
            return;
        }
        if constexpr(FloatLayout && !BatchOne) {
#pragma unroll
            for(uint32_t block=0;block+1<5;block+=2) {
                set_va_reg_sb(VA0,outputSrc[block]);set_va_reg_sb(VA1,outputSrc[block]+8);
                set_va_reg_sb(VA2,outputDst[block]);set_va_reg_sb(VA3,outputDst[block]+8);
                set_va_reg_sb(VA4,outputSrc[block+1]);set_va_reg_sb(VA5,outputSrc[block+1]+8);
                set_va_reg_sb(VA6,outputDst[block+1]);set_va_reg_sb(VA7,outputDst[block+1]+8);
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
                scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:80,repeats==1?0:2);
            }
            set_va_reg_sb(VA0,outputSrc[4]);set_va_reg_sb(VA1,outputSrc[4]+8);
            set_va_reg_sb(VA2,outputDst[4]);set_va_reg_sb(VA3,outputDst[4]+8);
            scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
            PipeBarrier<PIPE_V>();
            return;
        }
#pragma unroll
        for(uint32_t block=0;block<5;++block) {
            if constexpr(BatchOne)LoadAddressContext(11+block);
            else {
            if constexpr(!FloatLayout) {
                if(block!=0) {set_va_reg_sb(VA0,outputSrc[block]);set_va_reg_sb(VA1,outputSrc[block]+8);}
            } else {set_va_reg_sb(VA0,outputSrc[block]);set_va_reg_sb(VA1,outputSrc[block]+8);}
            set_va_reg_sb(VA2,outputDst[block]);set_va_reg_sb(VA3,outputDst[block]+8);
            }
            if constexpr(HalfEight) {
                if(block==0)scatter_vnchwconv_b32(VA2,VA6,repeats,repeats==1?0:80,repeats==1?0:2);
                else scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
            } else if constexpr(!FloatLayout) {
                if(block==0)scatter_vnchwconv_b16(VA2,VA6,repeats,repeats==1?0:80,repeats==1?0:1);
                else scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:1);
            } else
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
        }
        PipeBarrier<PIPE_V>();
    }
public:
    __aicore__ inline static void PrefetchAddressMetadata(GM_ADDR addressMetadata) {
        GlobalTensor<uint64_t> metadata;
        constexpr uint32_t kind=LargeMetadata?(sizeof(T)==2?4U:3U):((sizeof(T)==2 && BatchOne)?2U:1U);
        metadata.SetGlobalBuffer(fire_metadata::Table<kind,K>::Address());
        LocalTensor<uint64_t> destination(TPosition::VECCALC,MetadataBase,LargeMetadata?LargeContexts*16:(BatchOne?256:128));
        DataCopy(destination,metadata,LargeMetadata?LargeContexts*16:(BatchOne?256:128));
    }
    template<bool SkipFullPadding=false>
    __aicore__ inline void Init(GM_ADDR r,GM_ADDR p,GM_ADDR o,
        FireCandidateDecodeFusionTilingData t,GM_ADDR addressMetadata=nullptr,uint32_t batchShift=0,uint32_t uniformTileCount=0,uint32_t uniformOwner=0) {
        static_assert(!SkipFullPadding || (BatchOne && UniformFull && K>128 && K<=1024),
            "padding elision requires a complete count-guard owner");
        d=t;
        // The C220 Vector entry already initializes the same SoC state.
        // Address metadata prefetch and scalar packet work do not consume it.
#if __NPU_ARCH__ == 2201
        if constexpr(!(K>1024 && !BatchOne)) {
            if constexpr(!BatchOne)InitSocState();
            ResetMask();SetMaskNorm();
        }
#else
        if constexpr(!BatchOne)InitSocState();
        ResetMask();SetMaskNorm();
#endif
        raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);
        out.SetGlobalBuffer((__gm__ T*)o);
        const uint32_t total=((d.n+K-1)/K)*d.batch;
        if constexpr(BatchOne) {
            uint32_t part;
            if(!(d.batch&(d.batch-1))) {
                part=GetBlockIdx()>>batchShift;oneBatch=GetBlockIdx()&(d.batch-1);
            } else {
                part=GetBlockIdx()/d.batch;oneBatch=GetBlockIdx()%d.batch;
            }
            // Ownership units follow the GM tensor dtype, independently
            // of the internal8-row FP32 shuffle layout.
            constexpr uint32_t OwnerUnit=sizeof(T)==2?16:8;
            if constexpr(UniformFull) {
                oneN=part*(G*Q);oneCount=G*Q;
            } else {
                oneN=(part*d.unitsPerWorker+(part<d.extraUnits?part:d.extraUnits))*OwnerUnit;
                const uint32_t length=(d.unitsPerWorker+(part<d.extraUnits))*OwnerUnit;
                oneCount=d.n-oneN<length?d.n-oneN:length;
            }
            beginTile=0;endTile=1;stablePriorOwner=true;
        } else {
            if constexpr(UniformTiles) {
                static_assert(DirectScoreDivide,"uniform tile partition requires large Float");
                beginTile=uniformOwner*uniformTileCount;
                endTile=beginTile+uniformTileCount;
            } else {
                beginTile=total*GetBlockIdx()/d.blocks;endTile=total*(GetBlockIdx()+1)/d.blocks;
            }
            stablePriorOwner=beginTile/d.batch==(endTile-1)/d.batch;
        }
        if constexpr(!BatchOne) {
            const uint32_t firstN=beginTile/d.batch*K;
            const uint32_t firstB=beginTile%d.batch;
            const uint32_t firstCount=d.n-firstN<K?d.n-firstN:K;
            constexpr uint32_t RawAlign=32/sizeof(T);
            const uint32_t padded=(firstCount*6+RawAlign-1)/RawAlign*RawAlign;
            const uint32_t consumed=(firstCount+16*G-1)/(16*G)*16*G*6;
            if(consumed>padded) {
                Duplicate(packedBuf.template Get<T>()[padded],static_cast<T>(0.f),consumed-padded);
                PipeBarrier<PIPE_V>();
            }
            // StagePrior fences both packed input tails before MTE2.
            // Later geometry clears end at O0 and cannot touch either DMA.
            StagePrior(firstN,firstCount);
            LoadRaw(firstB*d.n+firstN,firstCount);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        if constexpr(BatchOne) {
            if(UniformFull || BatchOne || oneCount==G*Q) {
                // Half packed tails are initialized separately from DMA prefixes.
                // Packed prior/raw input destinations are disjoint, so their
                // DMA can overlap that initialization without a V_MTE2 wait.
                StagePrior(oneN,UniformFull?G*Q:oneCount);
                LoadRaw(oneBatch*d.n+oneN,UniformFull?G*Q:oneCount);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
        }
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto pbuf=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();
        if constexpr(BatchOne) {
            if(UniformFull || oneCount==G*Q) {
                // Every packed raw/prior element read by the transposes is
                // written by MTE2 (and the valid half cast) before use.
                // The transposes write G*Q rows in every plane. Initialize
                // the remaining padding in the representation read next.
                if constexpr(!SkipFullPadding) {
                constexpr uint32_t padding=NativeHalfPlanes?16:24;
                SetVectorMask<float,MaskMode::NORMAL>(0,(uint64_t(1)<<padding)-1);
                Duplicate<float,false>(pbuf[G*Q],0.f,MASK_PLACEHOLDER,5,1,P/8);
                if constexpr(NativeHalfPlanes) {
                    SetVectorMask<uint16_t,MaskMode::NORMAL>(0,(uint64_t(1)<<padding)-1);
                    Duplicate<uint16_t,false>(packedBuf.template Get<uint16_t>()[G*Q],
                        0,MASK_PLACEHOLDER,6,1,P/16);
                } else Duplicate<float,false>(s[G*Q],0.f,MASK_PLACEHOLDER,6,1,P/8);
                }
                SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
                Duplicate<float,false>(numeratorBuf.template Get<float>(),2.f,MASK_PLACEHOLDER,1,1,8);
                ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
            } else {
            // One counter-mode clear includes all plane padding, packed tails
            // and scratch. Address metadata starts after the numerator slot.
            LocalTensor<float> zeroRegion(TPosition::VECCALC,0,C0/4);
            SetMaskCount();
            if constexpr(NativeHalfPlanes && !UniformFull) {
                // MTE2 writes the complete padded valid prefixes. Vector
                // writes only their disjoint suffixes and the output planes.
                const uint32_t rawPadded=(oneCount*6+15)/16*8;
                const uint32_t priorPadded=(oneCount*5+7)/8*8;
                SetVectorMask<float,MaskMode::COUNTER>(3*P-rawPadded);
                vector_dup((__ubuf__ float*)v.GetPhyAddr()+rawPadded,0.f,1,1,1,8,0);
                SetVectorMask<float,MaskMode::COUNTER>(5*P-priorPadded);
                vector_dup((__ubuf__ float*)packed.GetPhyAddr()+priorPadded,0.f,1,1,1,8,0);
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                vector_dup((__ubuf__ float*)pbuf.GetPhyAddr(),0.f,1,1,1,8,0);
                SetVectorMask<float,MaskMode::COUNTER>(3*P);
                vector_dup((__ubuf__ float*)packedBuf.template Get<float>().GetPhyAddr(),0.f,1,1,1,8,0);
            } else if constexpr(sizeof(T)==4 && BatchOne && !UniformFull) {
                // DMA owns the padded prefixes; these suffixes are disjoint.
                const uint32_t rawPadded=(oneCount*6+7)/8*8;
                const uint32_t priorPadded=(oneCount*5+7)/8*8;
                SetVectorMask<float,MaskMode::COUNTER>(6*P-rawPadded);
                vector_dup((__ubuf__ float*)packedBuf.template Get<float>().GetPhyAddr()+rawPadded,0.f,1,1,1,8,0);
                SetVectorMask<float,MaskMode::COUNTER>(5*P-priorPadded);
                vector_dup((__ubuf__ float*)packed.GetPhyAddr()+priorPadded,0.f,1,1,1,8,0);
                // P and S keep their full defined state. V is first written
                // by packed sigmoid or each original geometry/cold stage.
                SetVectorMask<float,MaskMode::COUNTER>(V0/4);
                vector_dup((__ubuf__ float*)zeroRegion.GetPhyAddr(),0.f,1,1,1,8,0);
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(C0/4);
                Duplicate<float,false>(zeroRegion,0.f,MASK_PLACEHOLDER,1,1,8);
            }
            SetVectorMask<float,MaskMode::COUNTER>(64);
            Duplicate<float,false>(numeratorBuf.template Get<float>(),2.f,MASK_PLACEHOLDER,1,1,8);
            ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
            }
        } else {
            // Raw/prior packed inputs are produced by MTE2 and explicit
            // tail fills. Only the disjoint prior/value/work planes need zero.
            LocalTensor<float> zeroRegion(TPosition::VECCALC,0,O0/4);
            SetMaskCount();
            if constexpr(HalfEight) {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Duplicate<float,false>(pbuf,0.f,MASK_PLACEHOLDER,1,1,8);
                SetVectorMask<uint16_t,MaskMode::COUNTER>(6*P);
                Duplicate<uint16_t,false>(workBuf.template Get<uint16_t>(),0,
                    MASK_PLACEHOLDER,1,1,8);
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(O0/4);
                Duplicate<float,false>(zeroRegion,0.f,MASK_PLACEHOLDER,1,1,8);
            }
            SetVectorMask<float,MaskMode::COUNTER>(64);
            Duplicate<float,false>(numeratorBuf.template Get<float>(),2.f,MASK_PLACEHOLDER,1,1,8);
            if constexpr(DirectScoreDivide) {
                Duplicate<float,false>(scoreNumeratorBuf.template Get<float>(),1.f,MASK_PLACEHOLDER,1,1,8);
            }
            ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
        }
        if constexpr(!BatchOne && !LargeMetadata) {
        auto rawPacked=packedBuf.template Get<ShuffleT>();
        auto rawPlanes=FloatLayout?valuesBuf.template Get<ShuffleT>():workBuf.template Get<ShuffleT>();
        auto outputPlanes=WidePhaseLayout?valuesBuf.template Get<OutputShuffleT>():
            workBuf.template Get<OutputShuffleT>();
        auto outputPacked=stablePriorOwner?priorPackedBuf.template Get<OutputShuffleT>():
            (WidePhaseLayout?workBuf.template Get<OutputShuffleT>():valuesBuf.template Get<OutputShuffleT>());
#pragma unroll
        for(uint32_t block=0;block<RAWBLOCKS;++block) {
#pragma unroll
            for(uint32_t i=0;i<16;++i) {
                constexpr uint32_t rawWidth=32/sizeof(ShuffleT);
                rawSrc[block][i]=reinterpret_cast<uint64_t>(rawPacked[i*6*G+block*rawWidth].GetPhyAddr());
                if constexpr(!FloatLayout) {
                    const uint32_t j=block*16+i;
                    rawDst[block][i]=reinterpret_cast<uint64_t>(rawPlanes[(j%6)*P+(j/6)*Q].GetPhyAddr());
                } else if(i<8) {
                    const uint32_t j=block*8+i,offset=(j%6)*P+(j/6)*Q;
                    rawDst[block][2*i]=reinterpret_cast<uint64_t>(rawPlanes[offset].GetPhyAddr());
                    rawDst[block][2*i+1]=reinterpret_cast<uint64_t>(rawPlanes[offset+8].GetPhyAddr());
                }
            }
        }
#pragma unroll
        for(uint32_t block=0;block<PRBLOCKS;++block) {
#pragma unroll
            for(uint32_t i=0;i<16;++i) {
                const uint32_t row=WidePhaseLayout?i:(i<8?2*i:2*(i-8)+1);
                priorSrc[block][i]=reinterpret_cast<uint64_t>(packed[row*40+block*8].GetPhyAddr());
                if(i<8) {
                    const uint32_t j=block*8+i,offset=(j%5)*P+(j/5)*Q;
                    priorDst[block][2*i]=reinterpret_cast<uint64_t>(pbuf[offset].GetPhyAddr());
                    priorDst[block][2*i+1]=reinterpret_cast<uint64_t>(pbuf[offset+(WidePhaseLayout?8:8*Q)].GetPhyAddr());
                }
            }
        }
#pragma unroll
        for(uint32_t block=0;block<5;++block) {
#pragma unroll
            for(uint32_t i=0;i<16;++i) {
                if constexpr(!WidePhaseLayout) {
                    const uint32_t j=block*16+i;
                    outputSrc[block][i]=reinterpret_cast<uint64_t>(outputPlanes[(j%5)*P+(j/5)*Q].GetPhyAddr());
                    outputDst[block][i]=reinterpret_cast<uint64_t>(outputPacked[i*5*G+block*16].GetPhyAddr());
                } else if(i<8) {
                    const uint32_t j=block*8+i,offset=(j%5)*P+(j/5)*Q;
                    outputSrc[block][i]=reinterpret_cast<uint64_t>(outputPlanes[offset].GetPhyAddr());
                    outputSrc[block][i+8]=reinterpret_cast<uint64_t>(outputPlanes[offset+8].GetPhyAddr());
                    outputDst[block][2*i]=reinterpret_cast<uint64_t>(outputPacked[i*5*G+block*8].GetPhyAddr());
                    outputDst[block][2*i+1]=reinterpret_cast<uint64_t>(outputPacked[(i+8)*5*G+block*8].GetPhyAddr());
                }
            }
        }
        }
        RetainHalfContexts();

    }
    __aicore__ inline __attribute__((always_inline)) bool JointRangeBitmapAllLE() {
#if __NPU_ARCH__ == 2201 && !defined(ASCENDC_CPU_DEBUG) && !defined(__CCE_KT_TEST__)
        static_assert(sizeof(T)==4 && LargeMetadata,"large Float bitmap resource");
        constexpr uint32_t Bits=3*P,Compared=(Bits+63)/64*64;
        // This guard is the first use of a stable owner. The original
        // zero fill still owns the aligned suffix (gap and initial S).
        constexpr uint32_t FullHalves=Compared/16;
        static_assert((2*P+Compared)*4<=O0,"comparison remains in initialized P/S/V");
        static_assert(FullHalves*2<=1024,"converted predicate overlaps reduction");
        static_assert(1024+128<=4*P*4,"predicate scratch overlaps bitmap");
        LocalTensor<float> source(TPosition::VECCALC,2*P*4,Compared);
        auto bitmap=priorPackedBuf.template Get<uint8_t>()[4*P*4];
        auto failed=bitmap.template ReinterpretCast<uint16_t>();
        auto converted=priorPackedBuf.template Get<half>();
        LocalTensor<half> reduced(TPosition::VECCALC,F0+1024,64);
        CompareScalar(bitmap,source,32.f,CMPMODE::LE,Compared);
        PipeBarrier<PIPE_V>();
        Not(failed,failed,Compared/16);PipeBarrier<PIPE_V>();
        // Every nonzero signed16 integer remains finite and nonzero in Half.
        // Half is only an exact zero/nonzero carrier for this predicate.
        Cast(converted,failed.template ReinterpretCast<int16_t>(),RoundMode::CAST_NONE,FullHalves);
        PipeBarrier<PIPE_V>();Abs(converted,converted,FullHalves);PipeBarrier<PIPE_V>();
        ResetMask();SetMaskNorm();
        constexpr uint32_t Full=FullHalves/128,Tail=FullHalves%128;
        if constexpr(Full) {
            WholeReduceMax(reduced,converted,128,Full,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            if constexpr(Full>1) {
                WholeReduceMax(reduced,reduced,Full,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
                PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(Tail) {
            WholeReduceMax(reduced[32],converted[Full*128],Tail,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const auto* result=reinterpret_cast<const volatile __ubuf__ uint32_t*>(reduced.GetPhyAddr());
        bool all=true;
        if constexpr(Full)all=(result[0]&0x7fffu)==0;
        if constexpr(Tail)all=all&&((result[16]&0x7fffu)==0);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        return all;
#else
        return false;
#endif
    }
    template<bool Hot>
    __aicore__ inline void ProcessKnownOwner(uint32_t cachedN,bool needsRepair,bool shiftGrid) {
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();
        auto outputPacked=stablePriorOwner?priorPackedBuf.template Get<T>():
            (sizeof(T)==2?valuesBuf.template Get<T>():workBuf.template Get<T>());
        uint32_t b=BatchOne?oneBatch:beginTile%d.batch;
        uint32_t n=BatchOne?oneN:beginTile/d.batch*K;
        for(uint32_t tile=beginTile;tile<endTile;++tile) {
            const uint32_t count=BatchOne?(UniformFull?G*Q:oneCount):(d.n-n<K?d.n-n:K);
            const uint8_t repeats=static_cast<uint8_t>((count+16*G-1)/(16*G));
            if constexpr(!Hot) {
            if(n!=cachedN) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                UnpackPriors(repeats);
                // Packed priors have expired; the independent slot now owns
                // the guard without touching either previous output buffer.
                if constexpr(BatchOne)BeginOneStrideMax(p[2*P],packed,G*Q);
                else if constexpr(sizeof(T)==4) {
                    shiftGrid=false;
                    if(stablePriorOwner && endTile-beginTile>1) {
                        // One sufficient range test also proves the original
                        // stride-only center guard. ScratchV expires here.
                        bool bitmapAll=false;
                        if constexpr(LargeMetadata)bitmapAll=JointRangeBitmapAllLE();
                        if(__builtin_expect(bitmapAll,true))shiftGrid=true;
                        else {
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmax(vp,pp+2*P,pp+3*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        vmax(vp,vp,pp+4*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        ResetMask();SetMaskNorm();
                        shiftGrid=DirectStrideMax<LargeMetadata>(v,packed,P)<=32.f;
                        }
                        // A failed joint range test keeps the original
                        // stride decision and untouched numerical priors.
                        needsRepair=shiftGrid?false:DirectStrideMax<LargeMetadata>(p[2*P],packed,P)>32.f;
                    } else needsRepair=DirectStrideMax<LargeMetadata>(p[2*P],packed,P)>32.f;
                    if(shiftGrid) {
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                        vadds(pp,pp,-.5f,1,1,1,8,8);pipe_barrier(PIPE_V);
                        SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmul(pp+3*P,pp+3*P,pp+2*P,1,1,1,1,8,8,8);
                        vmul(pp+4*P,pp+4*P,pp+2*P,1,1,1,1,8,8,8);
                        pipe_barrier(PIPE_V);ResetMask();SetMaskNorm();
                    }
                } else needsRepair=DirectStrideMax<HalfEight>(p[2*P],packed,P)>4096.f;
                cachedN=n;
            }
            }
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // Input transpose is disjoint from the previous output: FP32
            // writes S while MTE3 reads V, FP16 writes V while MTE3 reads S.
            if constexpr(BatchOne && !NativeHalfPlanes)EvaluatePackedRaw(count);
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
            // A live raw successor waits on V_MTE2 before reusing O; the
            // following mask transition joins S before its first math read.
            UnpackRaw<!(Hot && !BatchOne && !HalfEight && FireInputFloatCompletionJoins)>(repeats);
#else
            UnpackRaw(repeats);
#endif
            if constexpr(NativeHalfPlanes)EvaluateFullHalfPlanes();
            if(tile!=beginTile && !stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            uint32_t nextB,nextN,nextCount;
            if constexpr(HalfEight) {
                // Raw transpose has released O. Prior suffix fills precede
                // sign/Cast, which establish the6P counter for Exp.
                nextB=b+1;nextN=n;nextCount=0;
                if(nextB==d.batch){nextB=0;nextN+=K;}
                if(tile+1<endTile) {
                    nextCount=d.n-nextN<K?d.n-nextN:K;
                    if constexpr(!Hot) if(nextN!=n)StagePrior(nextN,nextCount);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    LoadRaw(nextB*d.n+nextN,nextCount);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
            }
            if constexpr(sizeof(T)==2 && !BatchOne) {
                // Adding the high bit modulo2^16 toggles only binary16 sign.
                // It preserves every finite significand, including subnormals.
                SetMaskCount();SetVectorMask<int16_t,MaskMode::COUNTER>(6*P);
                Adds<int16_t,false>(valuesBuf.template Get<int16_t>(),
                    workBuf.template Get<int16_t>(),static_cast<int16_t>(-32768),
                    MASK_PLACEHOLDER,1,{1,1,8,8});
                PipeBarrier<PIPE_V>();
                if constexpr(HalfEight) {
                    // Preserve the sign operation's counter mask for widening.
                    Cast<float,T,false>(v,valuesBuf.template Get<T>(),RoundMode::CAST_NONE,
                        MASK_PLACEHOLDER,1,{1,1,8,4});
                } else {
                    ResetMask();SetMaskNorm();
                    Cast(v,valuesBuf.template Get<T>(),RoundMode::CAST_NONE,6*P);
                }
                PipeBarrier<PIPE_V>();
            }
            if constexpr(!HalfEight) {
                nextB=b+1;nextN=n;nextCount=0;
                if(nextB==d.batch){nextB=0;nextN+=K;}
                if(tile+1<endTile) {
                    nextCount=d.n-nextN<K?d.n-nextN:K;
                    if constexpr(!Hot) if(nextN!=n)StagePrior(nextN,nextCount);
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
                    // Every live raw successor keeps an explicit completion
                    // join. The terminal tile joins at its first math mask.
                    if constexpr(Hot && !BatchOne && !HalfEight && FireInputFloatCompletionJoins)PipeBarrier<PIPE_V>();
#endif
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    LoadRaw(nextB*d.n+nextN,nextCount);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
            }
            if constexpr(DualOutput) {
                if constexpr(Hot) {
                    auto sigmoidWork=s;
            if constexpr(!BatchOne) {
            if constexpr(!HalfEight) {
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
            }
            if constexpr(sizeof(T)==4) {
                Muls<float,false>(sigmoidWork,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            }
            Exp<float,false>(s,sigmoidWork,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(sigmoidWork,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});
            if constexpr(!FireInputFloatCompletionJoins)PipeBarrier<PIPE_V>();
            // The following mask transition retains the VEC completion join
            // before coordinate Div in the guarded generated entry.
            // Coordinates retain the original2/denominator operations.
            // For score, form the denominator product and divide only once.
            if constexpr(DirectScoreDivide && DualOutput) {
                SetVectorMask<float,MaskMode::COUNTER>(4*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
            }
            SetVectorMask<float,MaskMode::COUNTER>(P);
            auto* den=(__ubuf__ float*)sigmoidWork.GetPhyAddr();
            vmul(den+4*P,den+4*P,den+5*P,1,1,1,1,8,8,8);
            pipe_barrier(PIPE_V);
            if constexpr(DirectScoreDivide) {
                if constexpr(!DualOutput) {
                    SetVectorMask<float,MaskMode::COUNTER>(4*P);
                    Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                    PipeBarrier<PIPE_V>();
                }
                // Geometry consumes the completed coordinate planes. Score
                // uses disjoint planes and joins at its first geometry fence.
                if constexpr(!DualOutput)SetVectorMask<float,MaskMode::COUNTER>(P);
                Div<float,false>(s[4*P],scoreNumeratorBuf.template Get<float>(),sigmoidWork[4*P],
                    MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            }
            }
                } else {
                    auto sigmoidWork=v;
            if constexpr(!BatchOne) {
            if constexpr(!HalfEight) {
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
            }
            if constexpr(sizeof(T)==4) {
                Muls<float,false>(sigmoidWork,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            }
            Exp<float,false>(s,sigmoidWork,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(sigmoidWork,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            // Coordinates retain the original2/denominator operations.
            // For score, form the denominator product and divide only once.
            if constexpr(DirectScoreDivide && DualOutput) {
                SetVectorMask<float,MaskMode::COUNTER>(4*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
            }
            SetVectorMask<float,MaskMode::COUNTER>(P);
            auto* den=(__ubuf__ float*)sigmoidWork.GetPhyAddr();
            vmul(den+4*P,den+4*P,den+5*P,1,1,1,1,8,8,8);
            pipe_barrier(PIPE_V);
            if constexpr(DirectScoreDivide) {
                if constexpr(!DualOutput) {
                    SetVectorMask<float,MaskMode::COUNTER>(4*P);
                    Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                    PipeBarrier<PIPE_V>();
                }
                // Geometry consumes the completed coordinate planes. Score
                // uses disjoint planes and joins at its first geometry fence.
                if constexpr(!DualOutput)SetVectorMask<float,MaskMode::COUNTER>(P);
                Div<float,false>(s[4*P],scoreNumeratorBuf.template Get<float>(),sigmoidWork[4*P],
                    MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            }
            }
                }
            } else {
            if constexpr(!BatchOne) {
            if constexpr(!HalfEight) {
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
            }
            if constexpr(sizeof(T)==4) {
                Muls<float,false>(v,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            }
            Exp<float,false>(s,v,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(v,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
            // The following counter-mask transition must retain its lowered
            // VEC completion fence before the first coordinate division.
            if constexpr(!(UniformTiles && DirectScoreDivide && !DualOutput))
                PipeBarrier<PIPE_V>();
#else
            PipeBarrier<PIPE_V>();
#endif
            // Coordinates retain the original2/denominator operations.
            // For score, form the denominator product and divide only once.
            if constexpr(UniformTiles && DirectScoreDivide && !DualOutput) {
                SetVectorMask<float,MaskMode::COUNTER>(4*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
            }
            SetVectorMask<float,MaskMode::COUNTER>(P);
            auto* den=(__ubuf__ float*)v.GetPhyAddr();
            vmul(den+4*P,den+4*P,den+5*P,1,1,1,1,8,8,8);
            pipe_barrier(PIPE_V);
            if constexpr(DirectScoreDivide) {
                if constexpr(!(UniformTiles && !DualOutput)) {
                    SetVectorMask<float,MaskMode::COUNTER>(4*P);
                    Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                    PipeBarrier<PIPE_V>();
                }
                // Geometry consumes the completed coordinate planes. Score
                // uses disjoint planes and joins at its first geometry fence.
                if constexpr(!(UniformTiles && !DualOutput))SetVectorMask<float,MaskMode::COUNTER>(P);
                Div<float,false>(s[4*P],scoreNumeratorBuf.template Get<float>(),v[4*P],
                    MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            }
            }
            }
            if constexpr(BatchOne) {
                needsRepair=FinishOneStrideMax(packed,G*Q)>(sizeof(T)==2?4096.f:32.f);
                SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            }
            if constexpr(!Hot) if(__builtin_expect(needsRepair,false)) {
                if constexpr(ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
                RepairBlockedCenters<T,K,P,G,Q,S0,V0>((GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                // No assumption about VA preservation across a cold call.
                RetainHalfContexts();
                if constexpr(ScoreGeometryCounterCarry)SetMaskCount();
            }
            if constexpr(DualOutput) {
                if constexpr(Hot) {
            {
                // The four independent stages use exactly the same FP32
                // primitives/strides as CANN8.5 Add/Mul/Scalar wrappers.
                // Keep the required stage fences explicit.
                auto* sp=(__ubuf__ float*)s.GetPhyAddr();
                auto* vp=(__ubuf__ float*)s.GetPhyAddr();
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                if constexpr(!ScoreGeometryCounterCarry)SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                auto* centerSource=vp;
                if constexpr(sizeof(T)==4 && !BatchOne) {
                    if(shiftGrid)centerSource=sp;
                    else vadds(vp,sp,-.5f,1,1,1,8,8);
                } else vadds(vp,sp,-.5f,1,1,1,8,8);
                vmul(vp+2*P,sp+2*P,sp+2*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                vadd(sp,centerSource,pp,1,1,1,1,8,8,8);
                vmul(sp+2*P,vp+2*P,pp+3*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                SetVectorMask<float,MaskMode::COUNTER>(P);
#pragma unroll
                for(uint32_t c=0;c<4;++c) {
                    if constexpr(sizeof(T)==4 && !BatchOne) {
                        if(c<2 || !shiftGrid)vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                    } else vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                }
                if constexpr(BatchOne)vmul(vp+4*P,sp+4*P,sp+5*P,1,1,1,1,8,8,8);
                // HalfEight coordinates and score write disjoint planes;
                // their original final barrier joins all five outputs.
                if constexpr(!HalfEight)pipe_barrier(PIPE_V);
                if constexpr(!DirectScoreDivide) {
                    if constexpr(BatchOne)vmuls(sp+4*P,vp+4*P,.25f,1,1,1,8,8);
                    else vmuls(sp+4*P,sp+4*P,.5f,1,1,1,8,8);
                    pipe_barrier(PIPE_V);
                }
                ResetMask();SetMaskNorm();
            }
                } else {
            {
                // The four independent stages use exactly the same FP32
                // primitives/strides as CANN8.5 Add/Mul/Scalar wrappers.
                // Keep the required stage fences explicit.
                auto* sp=(__ubuf__ float*)s.GetPhyAddr();
                auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                if constexpr(!ScoreGeometryCounterCarry)SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                auto* centerSource=vp;
                if constexpr(sizeof(T)==4 && !BatchOne) {
                    if(shiftGrid)centerSource=sp;
                    else vadds(vp,sp,-.5f,1,1,1,8,8);
                } else vadds(vp,sp,-.5f,1,1,1,8,8);
                vmul(vp+2*P,sp+2*P,sp+2*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                vadd(sp,centerSource,pp,1,1,1,1,8,8,8);
                vmul(sp+2*P,vp+2*P,pp+3*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                SetVectorMask<float,MaskMode::COUNTER>(P);
#pragma unroll
                for(uint32_t c=0;c<4;++c) {
                    if constexpr(sizeof(T)==4 && !BatchOne) {
                        if(c<2 || !shiftGrid)vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                    } else vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                }
                if constexpr(BatchOne)vmul(vp+4*P,sp+4*P,sp+5*P,1,1,1,1,8,8,8);
                // HalfEight coordinates and score write disjoint planes;
                // their original final barrier joins all five outputs.
                if constexpr(!HalfEight)pipe_barrier(PIPE_V);
                if constexpr(!DirectScoreDivide) {
                    if constexpr(BatchOne)vmuls(sp+4*P,vp+4*P,.25f,1,1,1,8,8);
                    else vmuls(sp+4*P,sp+4*P,.5f,1,1,1,8,8);
                    pipe_barrier(PIPE_V);
                }
                ResetMask();SetMaskNorm();
            }
                }
            } else {
            {
                // The four independent stages use exactly the same FP32
                // primitives/strides as CANN8.5 Add/Mul/Scalar wrappers.
                // Keep the required stage fences explicit.
                auto* sp=(__ubuf__ float*)s.GetPhyAddr();
                auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                if constexpr(!ScoreGeometryCounterCarry)SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                auto* centerSource=vp;
                if constexpr(sizeof(T)==4 && !BatchOne) {
                    if(shiftGrid)centerSource=sp;
                    else vadds(vp,sp,-.5f,1,1,1,8,8);
                } else vadds(vp,sp,-.5f,1,1,1,8,8);
                vmul(vp+2*P,sp+2*P,sp+2*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                vadd(sp,centerSource,pp,1,1,1,1,8,8,8);
                vmul(sp+2*P,vp+2*P,pp+3*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                SetVectorMask<float,MaskMode::COUNTER>(P);
#pragma unroll
                for(uint32_t c=0;c<4;++c) {
                    if constexpr(sizeof(T)==4 && !BatchOne) {
                        if(c<2 || !shiftGrid)vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                    } else vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                }
                if constexpr(BatchOne)vmul(vp+4*P,sp+4*P,sp+5*P,1,1,1,1,8,8,8);
                // HalfEight coordinates and score write disjoint planes;
                // their original final barrier joins all five outputs.
                if constexpr(!HalfEight)pipe_barrier(PIPE_V);
                if constexpr(!DirectScoreDivide) {
                    if constexpr(BatchOne)vmuls(sp+4*P,vp+4*P,.25f,1,1,1,8,8);
                    else vmuls(sp+4*P,sp+4*P,.5f,1,1,1,8,8);
                    pipe_barrier(PIPE_V);
                }
                ResetMask();SetMaskNorm();
            }
            }
            if constexpr(sizeof(T)==2 && !BatchOne && !HalfEight) {
                Cast(workBuf.template Get<T>(),s,RoundMode::CAST_RINT,5*P);PipeBarrier<PIPE_V>();
            }
            if constexpr(DualOutput) {
                if constexpr(Hot) {
                    const uint32_t ordinal=tile-beginTile;
                    const bool second=(ordinal&1U)!=0;
                    // Completion is requested only for an actual future reuse.
                    if(ordinal>=2U) {
                        if(second)WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                        else WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    }
                    // Constant context calls keep VA addresses out of the loop.
                    if(second)PackOutputWithContext<16>(repeats);
                    else PackOutputWithContext<11>(repeats);
                    if(second) {
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                        CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
                        if(tile+2U<endTile)SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                    } else {
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                        CopyDenseOutput(out,priorPackedBuf.template Get<T>(),(b*d.n+n)*5,count*5);
                        if(tile+2U<endTile)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    }
                } else {
            // With a recycled prior slot, all Cast and arithmetic are
            // disjoint from MTE3. Wait only at the actual output overwrite.
            if(tile!=beginTile && stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            PackOutput(repeats);
            if constexpr(HalfEight) {
                // O[0,3P) float words belongs to next raw DMA. The disjoint
                // O[3P,6P) slot belongs to the current/previous Half output.
                if(tile!=beginTile)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                auto floatOutput=stablePriorOwner?priorPackedBuf.template Get<float>():
                    workBuf.template Get<float>();
                auto halfOutput=packedBuf.template Get<T>()[6*P];
                Cast(halfOutput,floatOutput,RoundMode::CAST_RINT,count*5);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,halfOutput,(b*d.n+n)*5,count*5);
            } else if constexpr(sizeof(T)==2 && BatchOne) {
                Cast(workBuf.template Get<T>(),priorPackedBuf.template Get<float>(),
                    RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,outputPacked,(b*d.n+n)*5,count*5);
            }
            if(nextCount)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                }
            } else {
            // With a recycled prior slot, all Cast and arithmetic are
            // disjoint from MTE3. Wait only at the actual output overwrite.
            if(tile!=beginTile && stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            PackOutput(repeats);
            if constexpr(HalfEight) {
                // O[0,3P) float words belongs to next raw DMA. The disjoint
                // O[3P,6P) slot belongs to the current/previous Half output.
                if(tile!=beginTile)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                auto floatOutput=stablePriorOwner?priorPackedBuf.template Get<float>():
                    workBuf.template Get<float>();
                auto halfOutput=packedBuf.template Get<T>()[6*P];
                Cast(halfOutput,floatOutput,RoundMode::CAST_RINT,count*5);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,halfOutput,(b*d.n+n)*5,count*5);
            } else if constexpr(sizeof(T)==2 && BatchOne) {
                Cast(workBuf.template Get<T>(),priorPackedBuf.template Get<float>(),
                    RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,outputPacked,(b*d.n+n)*5,count*5);
            }
            if(nextCount)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            b=nextB;n=nextN;
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void ProcessOwnerSelected() {
        uint32_t cachedN=d.n;
        bool needsRepair=true,shiftGrid=false;
        if(stablePriorOwner) {
            auto p=priorsBuf.template Get<float>();
            auto v=workBuf.template Get<float>();
            auto packed=priorPackedBuf.template Get<float>();
            const uint32_t n=beginTile/d.batch*K;
            const uint32_t count=d.n-n<K?d.n-n:K;
            const uint8_t repeats=static_cast<uint8_t>((count+16*G-1)/(16*G));

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                UnpackPriors(repeats);
                // Packed priors have expired; the independent slot now owns
                // the guard without touching either previous output buffer.
                if constexpr(BatchOne)BeginOneStrideMax(p[2*P],packed,G*Q);
                else if constexpr(sizeof(T)==4) {
                    shiftGrid=false;
                    if(stablePriorOwner && endTile-beginTile>1) {
                        // One sufficient range test also proves the original
                        // stride-only center guard. ScratchV expires here.
                        bool bitmapAll=false;
                        if constexpr(LargeMetadata)bitmapAll=JointRangeBitmapAllLE();
                        if(__builtin_expect(bitmapAll,true))shiftGrid=true;
                        else {
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmax(vp,pp+2*P,pp+3*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        vmax(vp,vp,pp+4*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        ResetMask();SetMaskNorm();
                        shiftGrid=DirectStrideMax<LargeMetadata>(v,packed,P)<=32.f;
                        }
                        // A failed joint range test keeps the original
                        // stride decision and untouched numerical priors.
                        needsRepair=shiftGrid?false:DirectStrideMax<LargeMetadata>(p[2*P],packed,P)>32.f;
                    } else needsRepair=DirectStrideMax<LargeMetadata>(p[2*P],packed,P)>32.f;
                    if(shiftGrid) {
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                        vadds(pp,pp,-.5f,1,1,1,8,8);pipe_barrier(PIPE_V);
                        SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmul(pp+3*P,pp+3*P,pp+2*P,1,1,1,1,8,8,8);
                        vmul(pp+4*P,pp+4*P,pp+2*P,1,1,1,1,8,8,8);
                        pipe_barrier(PIPE_V);ResetMask();SetMaskNorm();
                    }
                } else needsRepair=DirectStrideMax<HalfEight>(p[2*P],packed,P)>4096.f;
                cachedN=n;
            
            if(!needsRepair) {
                ProcessKnownOwner<true>(cachedN,false,shiftGrid);
                return;
            }
        }
        ProcessKnownOwner<false>(cachedN,needsRepair,shiftGrid);
    }
    __aicore__ inline void Process() {
        if constexpr(DualOutput) {
            ProcessOwnerSelected();
            return;
        }
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();
        auto outputPacked=stablePriorOwner?priorPackedBuf.template Get<T>():
            (sizeof(T)==2?valuesBuf.template Get<T>():workBuf.template Get<T>());
        uint32_t b=BatchOne?oneBatch:beginTile%d.batch;
        uint32_t n=BatchOne?oneN:beginTile/d.batch*K,cachedN=d.n;
        bool needsRepair=true,shiftGrid=false;
        bool useDualOutput=false;
        for(uint32_t tile=beginTile;tile<endTile;++tile) {
            const uint32_t count=BatchOne?(UniformFull?G*Q:oneCount):(d.n-n<K?d.n-n:K);
            const uint8_t repeats=static_cast<uint8_t>((count+16*G-1)/(16*G));
            if(n!=cachedN) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                UnpackPriors(repeats);
                // Packed priors have expired; the independent slot now owns
                // the guard without touching either previous output buffer.
                if constexpr(BatchOne)BeginOneStrideMax(p[2*P],packed,G*Q);
                else if constexpr(sizeof(T)==4) {
                    shiftGrid=false;
                    if(stablePriorOwner && endTile-beginTile>1) {
                        // One sufficient range test also proves the original
                        // stride-only center guard. ScratchV expires here.
                        bool bitmapAll=false;
                        if constexpr(LargeMetadata)bitmapAll=JointRangeBitmapAllLE();
                        if(__builtin_expect(bitmapAll,true))shiftGrid=true;
                        else {
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmax(vp,pp+2*P,pp+3*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        vmax(vp,vp,pp+4*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        ResetMask();SetMaskNorm();
                        shiftGrid=DirectStrideMax<LargeMetadata>(v,packed,P)<=32.f;
                        }
                        // A failed joint range test keeps the original
                        // stride decision and untouched numerical priors.
                        needsRepair=shiftGrid?false:DirectStrideMax<LargeMetadata>(p[2*P],packed,P)>32.f;
                    } else needsRepair=DirectStrideMax<LargeMetadata>(p[2*P],packed,P)>32.f;
                    if(shiftGrid) {
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                        vadds(pp,pp,-.5f,1,1,1,8,8);pipe_barrier(PIPE_V);
                        SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmul(pp+3*P,pp+3*P,pp+2*P,1,1,1,1,8,8,8);
                        vmul(pp+4*P,pp+4*P,pp+2*P,1,1,1,1,8,8,8);
                        pipe_barrier(PIPE_V);ResetMask();SetMaskNorm();
                    }
                } else needsRepair=DirectStrideMax<HalfEight>(p[2*P],packed,P)>4096.f;
                cachedN=n;
            }
            if constexpr(DualOutput)useDualOutput=stablePriorOwner && !needsRepair;
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // Input transpose is disjoint from the previous output: FP32
            // writes S while MTE3 reads V, FP16 writes V while MTE3 reads S.
            if constexpr(BatchOne && !NativeHalfPlanes)EvaluatePackedRaw(count);
            UnpackRaw(repeats);
            if constexpr(NativeHalfPlanes)EvaluateFullHalfPlanes();
            if(tile!=beginTile && !stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            uint32_t nextB,nextN,nextCount;
            if constexpr(HalfEight) {
                // Raw transpose has released O. Prior suffix fills precede
                // sign/Cast, which establish the6P counter for Exp.
                nextB=b+1;nextN=n;nextCount=0;
                if(nextB==d.batch){nextB=0;nextN+=K;}
                if(tile+1<endTile) {
                    nextCount=d.n-nextN<K?d.n-nextN:K;
                    if(nextN!=n)StagePrior(nextN,nextCount);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    LoadRaw(nextB*d.n+nextN,nextCount);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
            }
            if constexpr(sizeof(T)==2 && !BatchOne) {
                // Adding the high bit modulo2^16 toggles only binary16 sign.
                // It preserves every finite significand, including subnormals.
                SetMaskCount();SetVectorMask<int16_t,MaskMode::COUNTER>(6*P);
                Adds<int16_t,false>(valuesBuf.template Get<int16_t>(),
                    workBuf.template Get<int16_t>(),static_cast<int16_t>(-32768),
                    MASK_PLACEHOLDER,1,{1,1,8,8});
                PipeBarrier<PIPE_V>();
                if constexpr(HalfEight) {
                    // Preserve the sign operation's counter mask for widening.
                    Cast<float,T,false>(v,valuesBuf.template Get<T>(),RoundMode::CAST_NONE,
                        MASK_PLACEHOLDER,1,{1,1,8,4});
                } else {
                    ResetMask();SetMaskNorm();
                    Cast(v,valuesBuf.template Get<T>(),RoundMode::CAST_NONE,6*P);
                }
                PipeBarrier<PIPE_V>();
            }
            if constexpr(!HalfEight) {
                nextB=b+1;nextN=n;nextCount=0;
                if(nextB==d.batch){nextB=0;nextN+=K;}
                if(tile+1<endTile) {
                    nextCount=d.n-nextN<K?d.n-nextN:K;
                    if(nextN!=n)StagePrior(nextN,nextCount);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    LoadRaw(nextB*d.n+nextN,nextCount);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
            }
            if constexpr(DualOutput) {
                if(useDualOutput) {
                    auto sigmoidWork=s;
            if constexpr(!BatchOne) {
            if constexpr(!HalfEight) {
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
            }
            if constexpr(sizeof(T)==4) {
                Muls<float,false>(sigmoidWork,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            }
            Exp<float,false>(s,sigmoidWork,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(sigmoidWork,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            // Coordinates retain the original2/denominator operations.
            // For score, form the denominator product and divide only once.
            SetVectorMask<float,MaskMode::COUNTER>(P);
            auto* den=(__ubuf__ float*)sigmoidWork.GetPhyAddr();
            vmul(den+4*P,den+4*P,den+5*P,1,1,1,1,8,8,8);
            pipe_barrier(PIPE_V);
            if constexpr(DirectScoreDivide) {
                SetVectorMask<float,MaskMode::COUNTER>(4*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                // Geometry consumes the completed coordinate planes. Score
                // uses disjoint planes and joins at its first geometry fence.
                SetVectorMask<float,MaskMode::COUNTER>(P);
                Div<float,false>(s[4*P],scoreNumeratorBuf.template Get<float>(),sigmoidWork[4*P],
                    MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            }
            }
                } else {
                    auto sigmoidWork=v;
            if constexpr(!BatchOne) {
            if constexpr(!HalfEight) {
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
            }
            if constexpr(sizeof(T)==4) {
                Muls<float,false>(sigmoidWork,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            }
            Exp<float,false>(s,sigmoidWork,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(sigmoidWork,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            // Coordinates retain the original2/denominator operations.
            // For score, form the denominator product and divide only once.
            SetVectorMask<float,MaskMode::COUNTER>(P);
            auto* den=(__ubuf__ float*)sigmoidWork.GetPhyAddr();
            vmul(den+4*P,den+4*P,den+5*P,1,1,1,1,8,8,8);
            pipe_barrier(PIPE_V);
            if constexpr(DirectScoreDivide) {
                SetVectorMask<float,MaskMode::COUNTER>(4*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                // Geometry consumes the completed coordinate planes. Score
                // uses disjoint planes and joins at its first geometry fence.
                SetVectorMask<float,MaskMode::COUNTER>(P);
                Div<float,false>(s[4*P],scoreNumeratorBuf.template Get<float>(),sigmoidWork[4*P],
                    MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),sigmoidWork,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            }
            }
                }
            } else {
            if constexpr(!BatchOne) {
            if constexpr(!HalfEight) {
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
            }
            if constexpr(sizeof(T)==4) {
                Muls<float,false>(v,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            }
            Exp<float,false>(s,v,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(v,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
            // The following counter-mask transition must retain its lowered
            // VEC completion fence before the first coordinate division.
            if constexpr(!(UniformTiles && DirectScoreDivide && !DualOutput))
                PipeBarrier<PIPE_V>();
#else
            PipeBarrier<PIPE_V>();
#endif
            // Coordinates retain the original2/denominator operations.
            // For score, form the denominator product and divide only once.
            if constexpr(UniformTiles && DirectScoreDivide && !DualOutput) {
                SetVectorMask<float,MaskMode::COUNTER>(4*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
            }
            SetVectorMask<float,MaskMode::COUNTER>(P);
            auto* den=(__ubuf__ float*)v.GetPhyAddr();
            vmul(den+4*P,den+4*P,den+5*P,1,1,1,1,8,8,8);
            pipe_barrier(PIPE_V);
            if constexpr(DirectScoreDivide) {
                if constexpr(!(UniformTiles && !DualOutput)) {
                    SetVectorMask<float,MaskMode::COUNTER>(4*P);
                    Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                    PipeBarrier<PIPE_V>();
                }
                // Geometry consumes the completed coordinate planes. Score
                // uses disjoint planes and joins at its first geometry fence.
                if constexpr(!(UniformTiles && !DualOutput))SetVectorMask<float,MaskMode::COUNTER>(P);
                Div<float,false>(s[4*P],scoreNumeratorBuf.template Get<float>(),v[4*P],
                    MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            }
            }
            }
            if constexpr(BatchOne) {
                needsRepair=FinishOneStrideMax(packed,G*Q)>(sizeof(T)==2?4096.f:32.f);
                SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            }
            if(__builtin_expect(needsRepair,false)) {
                if constexpr(ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
                RepairBlockedCenters<T,K,P,G,Q,S0,V0>((GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                // No assumption about VA preservation across a cold call.
                RetainHalfContexts();
                if constexpr(ScoreGeometryCounterCarry)SetMaskCount();
            }
            if constexpr(DualOutput) {
                if(useDualOutput) {
            {
                // The four independent stages use exactly the same FP32
                // primitives/strides as CANN8.5 Add/Mul/Scalar wrappers.
                // Keep the required stage fences explicit.
                auto* sp=(__ubuf__ float*)s.GetPhyAddr();
                auto* vp=(__ubuf__ float*)s.GetPhyAddr();
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                if constexpr(!ScoreGeometryCounterCarry)SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                auto* centerSource=vp;
                if constexpr(sizeof(T)==4 && !BatchOne) {
                    if(shiftGrid)centerSource=sp;
                    else vadds(vp,sp,-.5f,1,1,1,8,8);
                } else vadds(vp,sp,-.5f,1,1,1,8,8);
                vmul(vp+2*P,sp+2*P,sp+2*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                vadd(sp,centerSource,pp,1,1,1,1,8,8,8);
                vmul(sp+2*P,vp+2*P,pp+3*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                SetVectorMask<float,MaskMode::COUNTER>(P);
#pragma unroll
                for(uint32_t c=0;c<4;++c) {
                    if constexpr(sizeof(T)==4 && !BatchOne) {
                        if(c<2 || !shiftGrid)vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                    } else vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                }
                if constexpr(BatchOne)vmul(vp+4*P,sp+4*P,sp+5*P,1,1,1,1,8,8,8);
                // HalfEight coordinates and score write disjoint planes;
                // their original final barrier joins all five outputs.
                if constexpr(!HalfEight)pipe_barrier(PIPE_V);
                if constexpr(!DirectScoreDivide) {
                    if constexpr(BatchOne)vmuls(sp+4*P,vp+4*P,.25f,1,1,1,8,8);
                    else vmuls(sp+4*P,sp+4*P,.5f,1,1,1,8,8);
                    pipe_barrier(PIPE_V);
                }
                ResetMask();SetMaskNorm();
            }
                } else {
            {
                // The four independent stages use exactly the same FP32
                // primitives/strides as CANN8.5 Add/Mul/Scalar wrappers.
                // Keep the required stage fences explicit.
                auto* sp=(__ubuf__ float*)s.GetPhyAddr();
                auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                if constexpr(!ScoreGeometryCounterCarry)SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                auto* centerSource=vp;
                if constexpr(sizeof(T)==4 && !BatchOne) {
                    if(shiftGrid)centerSource=sp;
                    else vadds(vp,sp,-.5f,1,1,1,8,8);
                } else vadds(vp,sp,-.5f,1,1,1,8,8);
                vmul(vp+2*P,sp+2*P,sp+2*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                vadd(sp,centerSource,pp,1,1,1,1,8,8,8);
                vmul(sp+2*P,vp+2*P,pp+3*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                SetVectorMask<float,MaskMode::COUNTER>(P);
#pragma unroll
                for(uint32_t c=0;c<4;++c) {
                    if constexpr(sizeof(T)==4 && !BatchOne) {
                        if(c<2 || !shiftGrid)vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                    } else vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                }
                if constexpr(BatchOne)vmul(vp+4*P,sp+4*P,sp+5*P,1,1,1,1,8,8,8);
                // HalfEight coordinates and score write disjoint planes;
                // their original final barrier joins all five outputs.
                if constexpr(!HalfEight)pipe_barrier(PIPE_V);
                if constexpr(!DirectScoreDivide) {
                    if constexpr(BatchOne)vmuls(sp+4*P,vp+4*P,.25f,1,1,1,8,8);
                    else vmuls(sp+4*P,sp+4*P,.5f,1,1,1,8,8);
                    pipe_barrier(PIPE_V);
                }
                ResetMask();SetMaskNorm();
            }
                }
            } else {
            {
                // The four independent stages use exactly the same FP32
                // primitives/strides as CANN8.5 Add/Mul/Scalar wrappers.
                // Keep the required stage fences explicit.
                auto* sp=(__ubuf__ float*)s.GetPhyAddr();
                auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                if constexpr(!ScoreGeometryCounterCarry)SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                auto* centerSource=vp;
                if constexpr(sizeof(T)==4 && !BatchOne) {
                    if(shiftGrid)centerSource=sp;
                    else vadds(vp,sp,-.5f,1,1,1,8,8);
                } else vadds(vp,sp,-.5f,1,1,1,8,8);
                vmul(vp+2*P,sp+2*P,sp+2*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                vadd(sp,centerSource,pp,1,1,1,1,8,8,8);
                vmul(sp+2*P,vp+2*P,pp+3*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                SetVectorMask<float,MaskMode::COUNTER>(P);
#pragma unroll
                for(uint32_t c=0;c<4;++c) {
                    if constexpr(sizeof(T)==4 && !BatchOne) {
                        if(c<2 || !shiftGrid)vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                    } else vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                }
                if constexpr(BatchOne)vmul(vp+4*P,sp+4*P,sp+5*P,1,1,1,1,8,8,8);
                // HalfEight coordinates and score write disjoint planes;
                // their original final barrier joins all five outputs.
                if constexpr(!HalfEight)pipe_barrier(PIPE_V);
                if constexpr(!DirectScoreDivide) {
                    if constexpr(BatchOne)vmuls(sp+4*P,vp+4*P,.25f,1,1,1,8,8);
                    else vmuls(sp+4*P,sp+4*P,.5f,1,1,1,8,8);
                    pipe_barrier(PIPE_V);
                }
                ResetMask();SetMaskNorm();
            }
            }
            if constexpr(sizeof(T)==2 && !BatchOne && !HalfEight) {
                Cast(workBuf.template Get<T>(),s,RoundMode::CAST_RINT,5*P);PipeBarrier<PIPE_V>();
            }
            if constexpr(DualOutput) {
                if(useDualOutput) {
                    const uint32_t ordinal=tile-beginTile;
                    const bool second=(ordinal&1U)!=0;
                    // Completion is requested only for an actual future reuse.
                    if(ordinal>=2U) {
                        if(second)WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                        else WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    }
                    // Constant context calls keep VA addresses out of the loop.
                    if(second)PackOutputWithContext<16>(repeats);
                    else PackOutputWithContext<11>(repeats);
                    if(second) {
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                        CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
                        if(tile+2U<endTile)SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                    } else {
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                        CopyDenseOutput(out,priorPackedBuf.template Get<T>(),(b*d.n+n)*5,count*5);
                        if(tile+2U<endTile)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    }
                } else {
            // With a recycled prior slot, all Cast and arithmetic are
            // disjoint from MTE3. Wait only at the actual output overwrite.
            if(tile!=beginTile && stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            PackOutput(repeats);
            if constexpr(HalfEight) {
                // O[0,3P) float words belongs to next raw DMA. The disjoint
                // O[3P,6P) slot belongs to the current/previous Half output.
                if(tile!=beginTile)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                auto floatOutput=stablePriorOwner?priorPackedBuf.template Get<float>():
                    workBuf.template Get<float>();
                auto halfOutput=packedBuf.template Get<T>()[6*P];
                Cast(halfOutput,floatOutput,RoundMode::CAST_RINT,count*5);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,halfOutput,(b*d.n+n)*5,count*5);
            } else if constexpr(sizeof(T)==2 && BatchOne) {
                Cast(workBuf.template Get<T>(),priorPackedBuf.template Get<float>(),
                    RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,outputPacked,(b*d.n+n)*5,count*5);
            }
            if(nextCount)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                }
            } else {
            // With a recycled prior slot, all Cast and arithmetic are
            // disjoint from MTE3. Wait only at the actual output overwrite.
            if(tile!=beginTile && stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            PackOutput(repeats);
            if constexpr(HalfEight) {
                // O[0,3P) float words belongs to next raw DMA. The disjoint
                // O[3P,6P) slot belongs to the current/previous Half output.
                if(tile!=beginTile)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                auto floatOutput=stablePriorOwner?priorPackedBuf.template Get<float>():
                    workBuf.template Get<float>();
                auto halfOutput=packedBuf.template Get<T>()[6*P];
                Cast(halfOutput,floatOutput,RoundMode::CAST_RINT,count*5);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,halfOutput,(b*d.n+n)*5,count*5);
            } else if constexpr(sizeof(T)==2 && BatchOne) {
                Cast(workBuf.template Get<T>(),priorPackedBuf.template Get<float>(),
                    RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,outputPacked,(b*d.n+n)*5,count*5);
            }
            if(nextCount)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            b=nextB;n=nextN;
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void ProcessFullGuardCount() {
        static_assert(BatchOne && UniformFull,"count guard requires complete single-tile owners");
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();
        auto outputPacked=stablePriorOwner?priorPackedBuf.template Get<T>():
            (sizeof(T)==2?valuesBuf.template Get<T>():workBuf.template Get<T>());
        uint32_t b=BatchOne?oneBatch:beginTile%d.batch;
        uint32_t n=BatchOne?oneN:beginTile/d.batch*K,cachedN=d.n;
        bool needsRepair=true,shiftGrid=false;
        uint64_t retained=0;
        for(uint32_t tile=beginTile;tile<endTile;++tile) {
            const uint32_t count=BatchOne?(UniformFull?G*Q:oneCount):(d.n-n<K?d.n-n:K);
            const uint8_t repeats=static_cast<uint8_t>((count+16*G-1)/(16*G));
            if(n!=cachedN) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                UnpackPriors(repeats);
                // Packed priors have expired; the independent slot now owns
                // the guard without touching either previous output buffer.
                if constexpr(BatchOne) {
                    auto mask=priorPackedBuf.template Get<uint8_t>()[2*P*sizeof(float)];
                    CompareScalar(mask,p[2*P],sizeof(T)==2?4096.f:32.f,CMPMODE::LE,G*Q);
                    PipeBarrier<PIPE_V>();
                    GatherMask(packed,p[2*P],mask.template ReinterpretCast<uint32_t>(),
                        true,G*Q,{1,1,8,1},retained);
                    PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
                }
                else if constexpr(sizeof(T)==4) {
                    shiftGrid=false;
                    if(stablePriorOwner && endTile-beginTile>1) {
                        // One sufficient range test also proves the original
                        // stride-only center guard. ScratchV expires here.
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmax(vp,pp+2*P,pp+3*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        vmax(vp,vp,pp+4*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        ResetMask();SetMaskNorm();
                        shiftGrid=DirectStrideMax(v,packed,P)<=32.f;
                        // A failed joint range test keeps the original
                        // stride decision and untouched numerical priors.
                        needsRepair=shiftGrid?false:DirectStrideMax(p[2*P],packed,P)>32.f;
                    } else needsRepair=DirectStrideMax(p[2*P],packed,P)>32.f;
                    if(shiftGrid) {
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                        vadds(pp,pp,-.5f,1,1,1,8,8);pipe_barrier(PIPE_V);
                        SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmul(pp+3*P,pp+3*P,pp+2*P,1,1,1,1,8,8,8);
                        vmul(pp+4*P,pp+4*P,pp+2*P,1,1,1,1,8,8,8);
                        pipe_barrier(PIPE_V);ResetMask();SetMaskNorm();
                    }
                } else needsRepair=DirectStrideMax(p[2*P],packed,P)>4096.f;
                cachedN=n;
            }
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // Input transpose is disjoint from the previous output: FP32
            // writes S while MTE3 reads V, FP16 writes V while MTE3 reads S.
            if constexpr(BatchOne && !NativeHalfPlanes)EvaluatePackedRaw(count);
            UnpackRaw(repeats);
            if constexpr(NativeHalfPlanes)EvaluateFullHalfPlanes();
            if(tile!=beginTile && !stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if constexpr(sizeof(T)==2 && !BatchOne) {
                // Adding the high bit modulo2^16 toggles only binary16 sign.
                // It preserves every finite significand, including subnormals.
                SetMaskCount();SetVectorMask<int16_t,MaskMode::COUNTER>(6*P);
                Adds<int16_t,false>(valuesBuf.template Get<int16_t>(),
                    workBuf.template Get<int16_t>(),static_cast<int16_t>(-32768),
                    MASK_PLACEHOLDER,1,{1,1,8,8});
                PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
                Cast(v,valuesBuf.template Get<T>(),RoundMode::CAST_NONE,6*P);PipeBarrier<PIPE_V>();
            }
            uint32_t nextB=b+1,nextN=n,nextCount=0;
            if(nextB==d.batch){nextB=0;nextN+=K;}
            if(tile+1<endTile) {
                nextCount=d.n-nextN<K?d.n-nextN:K;
                if(nextN!=n)StagePrior(nextN,nextCount);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                LoadRaw(nextB*d.n+nextN,nextCount);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            if constexpr(!BatchOne) {
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
            if constexpr(sizeof(T)==4) {
                Muls<float,false>(v,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            }
            Exp<float,false>(s,v,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(v,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            // Coordinates retain the original2/denominator operations.
            // For score, form the denominator product and divide only once.
            SetVectorMask<float,MaskMode::COUNTER>(P);
            auto* den=(__ubuf__ float*)v.GetPhyAddr();
            vmul(den+4*P,den+4*P,den+5*P,1,1,1,1,8,8,8);
            pipe_barrier(PIPE_V);
            if constexpr(DirectScoreDivide) {
                SetVectorMask<float,MaskMode::COUNTER>(4*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                // Geometry consumes the completed coordinate planes. Score
                // uses disjoint planes and joins at its first geometry fence.
                SetVectorMask<float,MaskMode::COUNTER>(P);
                Div<float,false>(s[4*P],scoreNumeratorBuf.template Get<float>(),v[4*P],
                    MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                ResetMask();SetMaskNorm();
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
            }
            }
            if constexpr(BatchOne) {
                needsRepair=false;
                if(__builtin_expect(retained!=G*Q,false))
                    needsRepair=DirectStrideMax(p[2*P],packed,G*Q)>(sizeof(T)==2?4096.f:32.f);
            }
            if(__builtin_expect(needsRepair,false)) {
                RepairBlockedCenters<T,K,P,G,Q,S0,V0>((GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                // No assumption about VA preservation across a cold call.
                RetainHalfContexts();
            }
            {
                // The four independent stages use exactly the same FP32
                // primitives/strides as CANN8.5 Add/Mul/Scalar wrappers.
                // Keep the required stage fences explicit.
                auto* sp=(__ubuf__ float*)s.GetPhyAddr();
                auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                auto* centerSource=vp;
                if constexpr(sizeof(T)==4 && !BatchOne) {
                    if(shiftGrid)centerSource=sp;
                    else vadds(vp,sp,-.5f,1,1,1,8,8);
                } else vadds(vp,sp,-.5f,1,1,1,8,8);
                vmul(vp+2*P,sp+2*P,sp+2*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                vadd(sp,centerSource,pp,1,1,1,1,8,8,8);
                vmul(sp+2*P,vp+2*P,pp+3*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                SetVectorMask<float,MaskMode::COUNTER>(P);
#pragma unroll
                for(uint32_t c=0;c<4;++c) {
                    if constexpr(sizeof(T)==4 && !BatchOne) {
                        if(c<2 || !shiftGrid)vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                    } else vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                }
                if constexpr(BatchOne)vmul(vp+4*P,sp+4*P,sp+5*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                if constexpr(!DirectScoreDivide) {
                    if constexpr(BatchOne)vmuls(sp+4*P,vp+4*P,.25f,1,1,1,8,8);
                    else vmuls(sp+4*P,sp+4*P,.5f,1,1,1,8,8);
                    pipe_barrier(PIPE_V);
                }
                ResetMask();SetMaskNorm();
            }
            if constexpr(sizeof(T)==2 && !BatchOne && !HalfEight) {
                Cast(workBuf.template Get<T>(),s,RoundMode::CAST_RINT,5*P);PipeBarrier<PIPE_V>();
            }
            // With a recycled prior slot, all Cast and arithmetic are
            // disjoint from MTE3. Wait only at the actual output overwrite.
            if(tile!=beginTile && stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            PackOutput(repeats);
            if constexpr(HalfEight) {
                // O[0,3P) float words belongs to next raw DMA. The disjoint
                // O[3P,6P) slot belongs to the current/previous Half output.
                if(tile!=beginTile)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                auto floatOutput=stablePriorOwner?priorPackedBuf.template Get<float>():
                    workBuf.template Get<float>();
                auto halfOutput=packedBuf.template Get<T>()[6*P];
                Cast(halfOutput,floatOutput,RoundMode::CAST_RINT,count*5);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,halfOutput,(b*d.n+n)*5,count*5);
            } else if constexpr(sizeof(T)==2 && BatchOne) {
                Cast(workBuf.template Get<T>(),priorPackedBuf.template Get<float>(),
                    RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,outputPacked,(b*d.n+n)*5,count*5);
            }
            if(nextCount)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            b=nextB;n=nextN;
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void ProcessPartialGuardCount() {
        static_assert(BatchOne && !UniformFull,"partial owner required");
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();
        auto outputPacked=stablePriorOwner?priorPackedBuf.template Get<T>():
            (sizeof(T)==2?valuesBuf.template Get<T>():workBuf.template Get<T>());
        uint32_t b=BatchOne?oneBatch:beginTile%d.batch;
        uint32_t n=BatchOne?oneN:beginTile/d.batch*K,cachedN=d.n;
        bool needsRepair=true,shiftGrid=false;
        uint64_t retained=0;
        for(uint32_t tile=beginTile;tile<endTile;++tile) {
            const uint32_t count=BatchOne?(UniformFull?G*Q:oneCount):(d.n-n<K?d.n-n:K);
            const uint8_t repeats=static_cast<uint8_t>((count+16*G-1)/(16*G));
            if(n!=cachedN) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                UnpackPriors(repeats);
                // Packed priors have expired; the independent slot now owns
                // the guard without touching either previous output buffer.
                if constexpr(BatchOne) {
                    auto mask=priorPackedBuf.template Get<uint8_t>()[2*P*sizeof(float)];
                    CompareScalar(mask,p[2*P],sizeof(T)==2?4096.f:32.f,CMPMODE::LE,G*Q);
                    PipeBarrier<PIPE_V>();
                    GatherMask(packed,p[2*P],mask.template ReinterpretCast<uint32_t>(),
                        true,G*Q,{1,1,8,1},retained);
                    PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
                }
                else if constexpr(sizeof(T)==4) {
                    shiftGrid=false;
                    if(stablePriorOwner && endTile-beginTile>1) {
                        // One sufficient range test also proves the original
                        // stride-only center guard. ScratchV expires here.
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmax(vp,pp+2*P,pp+3*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        vmax(vp,vp,pp+4*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                        ResetMask();SetMaskNorm();
                        shiftGrid=DirectStrideMax(v,packed,P)<=32.f;
                        // A failed joint range test keeps the original
                        // stride decision and untouched numerical priors.
                        needsRepair=shiftGrid?false:DirectStrideMax(p[2*P],packed,P)>32.f;
                    } else needsRepair=DirectStrideMax(p[2*P],packed,P)>32.f;
                    if(shiftGrid) {
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                        vadds(pp,pp,-.5f,1,1,1,8,8);pipe_barrier(PIPE_V);
                        SetVectorMask<float,MaskMode::COUNTER>(P);
                        vmul(pp+3*P,pp+3*P,pp+2*P,1,1,1,1,8,8,8);
                        vmul(pp+4*P,pp+4*P,pp+2*P,1,1,1,1,8,8,8);
                        pipe_barrier(PIPE_V);ResetMask();SetMaskNorm();
                    }
                } else needsRepair=DirectStrideMax(p[2*P],packed,P)>4096.f;
                cachedN=n;
            }
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // Input transpose is disjoint from the previous output: FP32
            // writes S while MTE3 reads V, FP16 writes V while MTE3 reads S.
            if constexpr(BatchOne && !NativeHalfPlanes)EvaluatePackedRaw(count);
            UnpackRaw(repeats);
            if constexpr(NativeHalfPlanes)EvaluateFullHalfPlanes();
            if(tile!=beginTile && !stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if constexpr(sizeof(T)==2 && !BatchOne) {
                // Adding the high bit modulo2^16 toggles only binary16 sign.
                // It preserves every finite significand, including subnormals.
                SetMaskCount();SetVectorMask<int16_t,MaskMode::COUNTER>(6*P);
                Adds<int16_t,false>(valuesBuf.template Get<int16_t>(),
                    workBuf.template Get<int16_t>(),static_cast<int16_t>(-32768),
                    MASK_PLACEHOLDER,1,{1,1,8,8});
                PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
                Cast(v,valuesBuf.template Get<T>(),RoundMode::CAST_NONE,6*P);PipeBarrier<PIPE_V>();
            }
            uint32_t nextB=b+1,nextN=n,nextCount=0;
            if(nextB==d.batch){nextB=0;nextN+=K;}
            if(tile+1<endTile) {
                nextCount=d.n-nextN<K?d.n-nextN:K;
                if(nextN!=n)StagePrior(nextN,nextCount);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                LoadRaw(nextB*d.n+nextN,nextCount);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            if constexpr(!BatchOne) {
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
            if constexpr(sizeof(T)==4) {
                Muls<float,false>(v,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            }
            Exp<float,false>(s,v,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(v,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            // Coordinates retain the original2/denominator operations.
            // For score, form the denominator product and divide only once.
            SetVectorMask<float,MaskMode::COUNTER>(P);
            auto* den=(__ubuf__ float*)v.GetPhyAddr();
            vmul(den+4*P,den+4*P,den+5*P,1,1,1,1,8,8,8);
            pipe_barrier(PIPE_V);
            if constexpr(DirectScoreDivide) {
                SetVectorMask<float,MaskMode::COUNTER>(4*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
                // Geometry consumes the completed coordinate planes. Score
                // uses disjoint planes and joins at its first geometry fence.
                SetVectorMask<float,MaskMode::COUNTER>(P);
                Div<float,false>(s[4*P],scoreNumeratorBuf.template Get<float>(),v[4*P],
                    MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                ResetMask();SetMaskNorm();
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
            }
            }
            if constexpr(BatchOne) {
                needsRepair=false;
                if(__builtin_expect(retained!=G*Q,false))
                    needsRepair=DirectStrideMax(p[2*P],packed,G*Q)>(sizeof(T)==2?4096.f:32.f);
            }
            if(__builtin_expect(needsRepair,false)) {
                RepairBlockedCenters<T,K,P,G,Q,S0,V0>((GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                // No assumption about VA preservation across a cold call.
                RetainHalfContexts();
            }
            {
                // The four independent stages use exactly the same FP32
                // primitives/strides as CANN8.5 Add/Mul/Scalar wrappers.
                // Keep the required stage fences explicit.
                auto* sp=(__ubuf__ float*)s.GetPhyAddr();
                auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                auto* centerSource=vp;
                if constexpr(sizeof(T)==4 && !BatchOne) {
                    if(shiftGrid)centerSource=sp;
                    else vadds(vp,sp,-.5f,1,1,1,8,8);
                } else vadds(vp,sp,-.5f,1,1,1,8,8);
                vmul(vp+2*P,sp+2*P,sp+2*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                vadd(sp,centerSource,pp,1,1,1,1,8,8,8);
                vmul(sp+2*P,vp+2*P,pp+3*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                SetVectorMask<float,MaskMode::COUNTER>(P);
#pragma unroll
                for(uint32_t c=0;c<4;++c) {
                    if constexpr(sizeof(T)==4 && !BatchOne) {
                        if(c<2 || !shiftGrid)vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                    } else vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                }
                if constexpr(BatchOne)vmul(vp+4*P,sp+4*P,sp+5*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);
                if constexpr(!DirectScoreDivide) {
                    if constexpr(BatchOne)vmuls(sp+4*P,vp+4*P,.25f,1,1,1,8,8);
                    else vmuls(sp+4*P,sp+4*P,.5f,1,1,1,8,8);
                    pipe_barrier(PIPE_V);
                }
                ResetMask();SetMaskNorm();
            }
            if constexpr(sizeof(T)==2 && !BatchOne && !HalfEight) {
                Cast(workBuf.template Get<T>(),s,RoundMode::CAST_RINT,5*P);PipeBarrier<PIPE_V>();
            }
            // With a recycled prior slot, all Cast and arithmetic are
            // disjoint from MTE3. Wait only at the actual output overwrite.
            if(tile!=beginTile && stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            PackOutput(repeats);
            if constexpr(HalfEight) {
                // O[0,3P) float words belongs to next raw DMA. The disjoint
                // O[3P,6P) slot belongs to the current/previous Half output.
                if(tile!=beginTile)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                auto floatOutput=stablePriorOwner?priorPackedBuf.template Get<float>():
                    workBuf.template Get<float>();
                auto halfOutput=packedBuf.template Get<T>()[6*P];
                Cast(halfOutput,floatOutput,RoundMode::CAST_RINT,count*5);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,halfOutput,(b*d.n+n)*5,count*5);
            } else if constexpr(sizeof(T)==2 && BatchOne) {
                Cast(workBuf.template Get<T>(),priorPackedBuf.template Get<float>(),
                    RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyDenseOutput(out,outputPacked,(b*d.n+n)*5,count*5);
            }
            if(nextCount)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            b=nextB;n=nextN;
        }
        PipeBarrier<PIPE_ALL>();
    }
};
