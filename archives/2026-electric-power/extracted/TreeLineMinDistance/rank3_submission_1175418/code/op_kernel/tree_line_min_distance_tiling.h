#if defined(TREE_LINE_MIN_DISTANCE_DEVICE_BODY)
#ifndef TREE_LINE_MIN_DISTANCE_DEVICE_CLASS_DEFINED
#define TREE_LINE_MIN_DISTANCE_DEVICE_CLASS_DEFINED
template<class T, uint32_t ROWS=64, uint32_t CAPACITY=1024, uint32_t COEFFICIENTS=CAPACITY,bool COMPACT_TAIL=false,bool SINGLE_TAIL=false>
// Power-of-two segment capacity bounds the shared coefficient cache.
class KernelTreeBatch {
    static constexpr bool DEFER_DIRECTION_LOW=sizeof(T)==4 && CAPACITY==256
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool NATIVE_HALF_PREFIX_PRIMARY=sizeof(T)==2 && CAPACITY==1024 && COEFFICIENTS==576
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#else
        && false
#endif
        ;
protected:
    static constexpr bool NATIVE_FLOAT_PREFIX_SPARSE=sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==576
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        ;
#else
        && false;
#endif
private:
    static constexpr bool SAVE_PREFIX_CHUNK_PREDICATES=NATIVE_HALF_PREFIX_PRIMARY || NATIVE_FLOAT_PREFIX_SPARSE;
    static constexpr bool PREFIX_NORMALIZATION=sizeof(T)==4 && ((CAPACITY==1024 && COEFFICIENTS==576) || CAPACITY==256);
    static constexpr bool OVERLAP_POINT_INPUT=NATIVE_HALF_PREFIX_PRIMARY || (CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL);
    static constexpr bool DEFER_CHUNK_CERTIFICATE=CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL;
    static constexpr bool NORM_BEFORE_DOT=DEFER_CHUNK_CERTIFICATE && sizeof(T)==4;
    static constexpr bool CLEAN_PRIMARY64=DEFER_CHUNK_CERTIFICATE
#ifdef DTYPE_TREE_POINTS
        && (sizeof(T)==4 || sizeof(T)==sizeof(DTYPE_TREE_POINTS))
#else
        && sizeof(T)==4
#endif
        ;
    static constexpr bool GUARDED_ERROR=CLEAN_PRIMARY64;
    static constexpr bool FULL_FIXED_MASK=NATIVE_HALF_PREFIX_PRIMARY || (DEFER_CHUNK_CERTIFICATE && (sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        || sizeof(DTYPE_TREE_POINTS)==2
#endif
        ));
public:
    __aicore__ inline void Init(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,const TreeLineMinDistanceTilingData& t) {
        t_=t;m_=CAPACITY;cp_=m_+32;wp_=ROWS*64+128;pp_=(m_>ROWS*8?m_:ROWS*8)+32;count_=COEFFICIENTS;
        pg_.SetGlobalBuffer((__gm__ T*)p);sg_.SetGlobalBuffer((__gm__ T*)s);
        dg_.SetGlobalBuffer((__gm__ T*)d);ig_.SetGlobalBuffer((__gm__ int32_t*)i);
        uint32_t address=0;
        columns_=Allocate<float>(address,14*cp_);xyz_=Allocate<float>(address,3*pp_);
        work_=Allocate<float>(address,6*wp_);
        sf_=work_[6*m_];
        if constexpr(sizeof(T)==4)rawS_=sf_.template ReinterpretCast<T>();
        else rawS_=work_.template ReinterpretCast<T>();
        so_=work_[13*m_].template ReinterpretCast<int32_t>();ones_=work_[12*m_];
        segmentIndices_=Allocate<float>(address,64);
        pf_=Allocate<float>(address,192);
        if constexpr(sizeof(T)==4)rawP_=pf_.template ReinterpretCast<T>();
        else rawP_=Allocate<T>(address,192);
        chunkLower_=Allocate<float>(address,1024);
        compactP_=Allocate<float>(address,192);refined_=Allocate<float>(address,13*64);
        record_=Allocate<float>(address,128);record2_=Allocate<float>(address,16);
        lo_=Allocate<float>(address,64);second_=Allocate<float>(address,64);hi_=Allocate<float>(address,64);
        idx_=Allocate<int32_t>(address,64);cert_=Allocate<float>(address,64);certHigh_=Allocate<float>(address,64);oneHot_=Allocate<int32_t>(address,64);
        tileLo_=Allocate<float>(address,64);tileSecond_=Allocate<float>(address,64);tileHi_=Allocate<float>(address,64);
        tileIdx_=Allocate<int32_t>(address,64);offset_=Allocate<int32_t>(address,64);
        rowBase_=Allocate<int32_t>(address,64);pointOffset_=Allocate<int32_t>(address,64);
        outputOffset_=Allocate<int32_t>(address,64);outT_=Allocate<T>(address,64);
        mask_=Allocate<uint8_t>(address,ROWS*8);valid_=Allocate<uint8_t>(address,ROWS*8);smallMask_=Allocate<uint8_t>(address,32);
        if constexpr(CAPACITY==256 && sizeof(T)==4)so_=Allocate<int32_t>(address,COEFFICIENTS);
        if constexpr(!DEFER_DIRECTION_LOW) {
            Duplicate(rawP_,T(0),192);Duplicate(compactP_,0.0F,192);
        }
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            // First-tile input buffers are independent of the index/state arena.
            Duplicate(rawS_,T(0),COEFFICIENTS*6);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            const uint32_t firstPosition=t_.batches*t_.points*GetBlockIdx()/t_.blocks;
            const uint32_t firstEnd=t_.batches*t_.points*(GetBlockIdx()+1)/t_.blocks;
            const uint32_t firstBatch=firstPosition/t_.points,firstPoint=firstPosition%t_.points;
            const uint32_t firstRows=Smallest(Smallest(firstEnd-firstPosition,t_.points-firstPoint),ROWS);
            DataCopyPad(rawS_,sg_[firstBatch*t_.segments*6],DataCopyExtParams{1,t_.segments*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            DataCopyPad(rawP_,pg_[firstPosition*3],DataCopyExtParams{1,firstRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            // Warm fixed-arena callees while the first input records are in flight.
            AscendC::PreLoadImpl(reinterpret_cast<void*>(&KernelTreeBatch::LoadBoundedResidentSegments),int64_t(1));
            AscendC::PreLoadImpl(reinterpret_cast<void*>(&KernelTreeBatch::PrimaryCompensated16),int64_t(1));
        }
        MakeIndices(offset_,0,64);
        if constexpr(CAPACITY==1024) {
        ShiftLeft(oneHot_.template ReinterpretCast<uint32_t>(),offset_.template ReinterpretCast<uint32_t>(),uint32_t(23),32);PipeBarrier<PIPE_V>();
        Adds(oneHot_,oneHot_,int32_t(0x3f800000),32);PipeBarrier<PIPE_V>();
        Cast(oneHot_,oneHot_.template ReinterpretCast<float>(),RoundMode::CAST_RINT,31);PipeBarrier<PIPE_V>();
        SetMaskNorm();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x80000000ULL));
        Duplicate<int32_t,false>(oneHot_,int32_t(-2147483647-1),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        if constexpr(DEFER_CHUNK_CERTIFICATE) {
            // Three coordinate padding gaps and dead refinement storage hold
            // four groups of four complete 64-row comparison predicates.
            #pragma unroll
            for(uint32_t group=0;group<4;++group) {
                const uint32_t base=group<3?uint32_t(xyz_[group*pp_+ROWS*8].GetPhyAddr()):uint32_t(refined_.GetPhyAddr());
                Duplicate(oneHot_[32+group*8],int32_t(base),8);
            }
        } else Copy(oneHot_[32],oneHot_,32,1,CopyRepeatParams(1,1,4,4));
        PipeBarrier<PIPE_V>();
        }
        if constexpr(CLEAN_PRIMARY64 && (sizeof(T)==4 || sizeof(T)==2) && ROWS==64) {
            // Unused column padding preserves two aligned bitset rotations.
            auto rotate4=C(3)[CAPACITY+16].template ReinterpretCast<int32_t>();
            auto rotate2=C(3)[CAPACITY+24].template ReinterpretCast<int32_t>();
            Duplicate(tileLo_.template ReinterpretCast<int32_t>(),int32_t(7),8);
            Adds(rotate4,offset_,int32_t(4),8);
            Adds(rotate2,offset_,int32_t(2),8);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
            const BinaryRepeatParams bitsSame{1,1,1,8,8,8};
            And<uint16_t,false>(rotate4.template ReinterpretCast<uint16_t>(),
                rotate4.template ReinterpretCast<uint16_t>(),tileLo_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,bitsSame);
            And<uint16_t,false>(rotate2.template ReinterpretCast<uint16_t>(),
                rotate2.template ReinterpretCast<uint16_t>(),tileLo_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,bitsSame);
            PipeBarrier<PIPE_V>();
            Muls(rotate4,rotate4,int32_t(4),8);Muls(rotate2,rotate2,int32_t(4),8);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        Muls(rowBase_,offset_,int32_t(256),64);Muls(pointOffset_,offset_,int32_t(12),64);
        Muls(outputOffset_,offset_,int32_t(8),64);Cast(segmentIndices_,offset_,RoundMode::CAST_NONE,64);
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            // Persistent segment gather indices survive scratch reuse and batch changes.
            Muls(so_,offset_,int32_t(24),64);PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t chunk=64;chunk<COEFFICIENTS;chunk+=64)
                Adds(so_[chunk],so_,int32_t(chunk*24),64);
        }
        Duplicate(tileLo_,__builtin_inff(),64);
        if constexpr(!PREFIX_NORMALIZATION)Duplicate(tileSecond_,__builtin_inff(),64);
        Duplicate(tileHi_,__builtin_inff(),64);
        Duplicate(tileIdx_,int32_t(0),64);
        if constexpr(NATIVE_HALF_PREFIX_PRIMARY || NATIVE_FLOAT_PREFIX_SPARSE || (CLEAN_PRIMARY64 && sizeof(T)==2))Duplicate(C(3)[CAPACITY+8],-2.0F,8);
        if constexpr(NATIVE_HALF_PREFIX_PRIMARY || NATIVE_FLOAT_PREFIX_SPARSE)Duplicate(xyz_[pp_-16],2.1250002384185791015625F,8);
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
            Duplicate(C(3)[CAPACITY+8],-2.0F,8);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL)
                Duplicate(C(4)[CAPACITY+8],-2.0F,8);
#endif
            Duplicate(xyz_[pp_-16],2.125F,8);
        }
        PipeBarrier<PIPE_ALL>();
    }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ inline bool SpatialSortPointCoordinatesSafe(uint32_t begin,uint32_t end) {
        if(end-begin<ROWS*8)return false;
        auto raw=work_,absolute=work_[4224];
        for(uint32_t coordinate=begin*3;coordinate<end*3;coordinate+=4096) {
            const uint32_t available=Smallest(end*3-coordinate,4096);
            const uint32_t aligned=(available+63)/64*64;
            if(aligned!=available)Duplicate(raw,0.0F,aligned);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            DataCopyPad(raw,pg_[coordinate],DataCopyExtParams{1,available*4,0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            Abs(absolute,raw,aligned);PipeBarrier<PIPE_V>();
            WholeReduceMax(record_,absolute,64,aligned/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMax(tileLo_,record_,aligned/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            CompareScalar(smallMask_,tileLo_,1099511627776.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if((smallMask_.GetValue(0)&1U)==0) {PipeBarrier<PIPE_ALL>();return false;}
        }
        PipeBarrier<PIPE_ALL>();return true;
    }
#endif
    // Commit one exact fallback result while preserving all other vector rows.
    __aicore__ __attribute__((always_inline)) inline void CommitRefinedCandidate(uint32_t row) {
        Brcb(W(0),record_,1,BrcbRepeatParams(1,8));
        Brcb(W(1),tileIdx_.template ReinterpretCast<float>(),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(W(2),W(0),64,1,CopyRepeatParams(1,0,8,0));
        Copy(W(3),W(1),64,1,CopyRepeatParams(1,0,8,0));
        CompareScalar(smallMask_,segmentIndices_,float(int32_t(row)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetCmpMask(smallMask_);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,W(2),tileSecond_,1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),W(3),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
    }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ __attribute__((always_inline)) inline bool RefineFullFloatFallbackBody(uint32_t row,uint32_t chunks) {
        static_assert(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full resource required");
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
                    Compensated<true>();
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
                    const bool unique=CertifyWideScore(candidates);
                    columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
                    CommitRefinedCandidate(row);
                    return unique;
    }
    // Shared fixed UB layout; inlined into each existing static entry.
    __aicore__ __attribute__((always_inline)) static inline void InitFullFloatFrame(
        KernelTreeBatch& frame,uint32_t segmentCount) {
        frame.t_={};frame.t_.segments=segmentCount;
        frame.m_=1024U;frame.cp_=1056U;frame.pp_=1056U;frame.wp_=1056U;frame.count_=1024U;frame.liveRows_=64U;
        frame.directionLowReady_=false;
        frame.columns_=LocalTensor<float>(TPosition::VECCALC,0U,14784U);
        frame.xyz_=LocalTensor<float>(TPosition::VECCALC,59136U,3168U);
        frame.work_=LocalTensor<float>(TPosition::VECCALC,71808U,25344U);
        frame.segmentIndices_=LocalTensor<float>(TPosition::VECCALC,173184U,64U);
        frame.pf_=LocalTensor<float>(TPosition::VECCALC,173440U,192U);
        frame.chunkLower_=LocalTensor<float>(TPosition::VECCALC,174208U,1024U);
        frame.compactP_=LocalTensor<float>(TPosition::VECCALC,178304U,192U);
        frame.refined_=LocalTensor<float>(TPosition::VECCALC,179072U,832U);
        frame.record_=LocalTensor<float>(TPosition::VECCALC,182400U,128U);
        frame.record2_=LocalTensor<float>(TPosition::VECCALC,182912U,16U);
        frame.lo_=LocalTensor<float>(TPosition::VECCALC,182976U,64U);
        frame.second_=LocalTensor<float>(TPosition::VECCALC,183232U,64U);
        frame.hi_=LocalTensor<float>(TPosition::VECCALC,183488U,64U);
        frame.idx_=LocalTensor<int32_t>(TPosition::VECCALC,183744U,64U);
        frame.cert_=LocalTensor<float>(TPosition::VECCALC,184000U,64U);
        frame.certHigh_=LocalTensor<float>(TPosition::VECCALC,184256U,64U);
        frame.oneHot_=LocalTensor<int32_t>(TPosition::VECCALC,184512U,64U);
        frame.tileLo_=LocalTensor<float>(TPosition::VECCALC,184768U,64U);
        frame.tileSecond_=LocalTensor<float>(TPosition::VECCALC,185024U,64U);
        frame.tileHi_=LocalTensor<float>(TPosition::VECCALC,185280U,64U);
        frame.tileIdx_=LocalTensor<int32_t>(TPosition::VECCALC,185536U,64U);
        frame.offset_=LocalTensor<int32_t>(TPosition::VECCALC,185792U,64U);
        frame.rowBase_=LocalTensor<int32_t>(TPosition::VECCALC,186048U,64U);
        frame.pointOffset_=LocalTensor<int32_t>(TPosition::VECCALC,186304U,64U);
        frame.outputOffset_=LocalTensor<int32_t>(TPosition::VECCALC,186560U,64U);
        frame.outT_=LocalTensor<T>(TPosition::VECCALC,186816U,64U);
        frame.mask_=LocalTensor<uint8_t>(TPosition::VECCALC,187072U,512U);
        frame.valid_=LocalTensor<uint8_t>(TPosition::VECCALC,187584U,512U);
        frame.smallMask_=LocalTensor<uint8_t>(TPosition::VECCALC,188096U,32U);
        frame.sf_=frame.work_[6144];
        frame.rawS_=frame.sf_.template ReinterpretCast<T>();frame.rawP_=frame.pf_.template ReinterpretCast<T>();
        frame.so_=frame.work_[13312].template ReinterpretCast<int32_t>();frame.ones_=frame.work_[12288];
    }
    __aicore__ __attribute__((noinline)) static bool RefineFullFloatFallback(uint32_t row,uint32_t chunks,uint32_t segmentCount) {
        static_assert(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full resource required");
        KernelTreeBatch frame;
        InitFullFloatFrame(frame,segmentCount);
        return frame.RefineFullFloatFallbackBody(row,chunks);
    }
#endif
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ __attribute__((noinline)) static void LoadFullFloatSegmentsStatic(GM_ADDR segments,uint32_t batch,uint32_t segmentCount) {
        static_assert(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full resource required");
        static_assert(!DEFER_DIRECTION_LOW && !NATIVE_HALF_PREFIX_PRIMARY,"No mutable metadata is produced by this Float loader");
        KernelTreeBatch frame;
        InitFullFloatFrame(frame,segmentCount);
        frame.sg_.SetGlobalBuffer((__gm__ T*)segments);
        frame.LoadSegments(batch,false);
    }
#endif
    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineBoundedFullRow(
        uint32_t row,uint32_t chunks,uint32_t logicalSegments) {
        static_assert(DEFER_DIRECTION_LOW,"Bounded native Float cache required");
        KernelTreeBatch op;
        op.m_=CAPACITY;op.cp_=CAPACITY+32;op.wp_=ROWS*64+128;
        op.pp_=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;op.count_=COEFFICIENTS;
        op.t_.segments=logicalSegments;
        uint32_t address=0;
        op.columns_=op.template Allocate<float>(address,14*op.cp_);op.xyz_=op.template Allocate<float>(address,3*op.pp_);
        op.work_=op.template Allocate<float>(address,6*op.wp_);
        op.sf_=op.work_[6*op.m_];
        if constexpr(sizeof(T)==4)op.rawS_=op.sf_.template ReinterpretCast<T>();
        else op.rawS_=op.work_.template ReinterpretCast<T>();
        op.so_=op.work_[13*op.m_].template ReinterpretCast<int32_t>();op.ones_=op.work_[12*op.m_];
        op.segmentIndices_=op.template Allocate<float>(address,64);
        op.pf_=op.template Allocate<float>(address,192);
        if constexpr(sizeof(T)==4)op.rawP_=op.pf_.template ReinterpretCast<T>();
        else op.rawP_=op.template Allocate<T>(address,192);
        op.chunkLower_=op.template Allocate<float>(address,1024);
        op.compactP_=op.template Allocate<float>(address,192);op.refined_=op.template Allocate<float>(address,13*64);
        op.record_=op.template Allocate<float>(address,128);op.record2_=op.template Allocate<float>(address,16);
        op.lo_=op.template Allocate<float>(address,64);op.second_=op.template Allocate<float>(address,64);op.hi_=op.template Allocate<float>(address,64);
        op.idx_=op.template Allocate<int32_t>(address,64);op.cert_=op.template Allocate<float>(address,64);op.certHigh_=op.template Allocate<float>(address,64);op.oneHot_=op.template Allocate<int32_t>(address,64);
        op.tileLo_=op.template Allocate<float>(address,64);op.tileSecond_=op.template Allocate<float>(address,64);op.tileHi_=op.template Allocate<float>(address,64);
        op.tileIdx_=op.template Allocate<int32_t>(address,64);op.offset_=op.template Allocate<int32_t>(address,64);
        op.rowBase_=op.template Allocate<int32_t>(address,64);op.pointOffset_=op.template Allocate<int32_t>(address,64);
        op.outputOffset_=op.template Allocate<int32_t>(address,64);op.outT_=op.template Allocate<T>(address,64);
        op.mask_=op.template Allocate<uint8_t>(address,ROWS*8);op.valid_=op.template Allocate<uint8_t>(address,ROWS*8);op.smallMask_=op.template Allocate<uint8_t>(address,32);
        if constexpr(CAPACITY==256 && sizeof(T)==4)op.so_=op.template Allocate<int32_t>(address,COEFFICIENTS);
        op.wp_=CAPACITY+32;
        return op.RefineBoundedFullRowBody(row,chunks,logicalSegments);
    }
    __aicore__ __attribute__((always_inline)) inline bool RefineBoundedFullRowBody(
        uint32_t row,uint32_t chunks,uint32_t logicalSegments) {
        static_assert(DEFER_DIRECTION_LOW,"Bounded native Float cache required");
                    uint32_t first=0,second=0,hits=0;
                    for(uint32_t chunk=0;chunk<(logicalSegments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
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
                    CompareScalar(valid_,W(7),float(int32_t(logicalSegments)),CMPMODE::LT,candidates);PipeBarrier<PIPE_V>();
                    if constexpr(CAPACITY==256 && sizeof(T)==4) {
                        if(sparse) {
                            if(hits==1)SparseCompensated<64>();
                            else SparseCompensated<128>();
                        } else Compensated<true>();
                    } else Compensated<true>();
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
                    const bool unique=CertifyWideScore(candidates);
                    columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
                    CommitRefinedCandidate(row);
                    return unique;
    }
    __aicore__ __attribute__((noinline,aligned(128))) void RestartBoundedAfterExact() {
        static_assert(DEFER_DIRECTION_LOW || NATIVE_FLOAT_PREFIX_SPARSE,"Resident native Float cache required");
        const TreeLineMinDistanceTilingData t=t_;
        Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
             (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
    }
    __aicore__ __attribute__((always_inline)) inline void RestartBoundedInlineAfterExact() {
        static_assert(DEFER_DIRECTION_LOW || NATIVE_FLOAT_PREFIX_SPARSE,"Resident native Float cache required");
        const TreeLineMinDistanceTilingData t=t_;
        Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
             (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
    }
    template<bool OUTLINE_FULL_FLOAT_FALLBACK=false>
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_1
    __aicore__ inline void Process(uint32_t rangeBegin=0,uint32_t rangeEnd=0,GM_ADDR spatialWorkspace=nullptr,bool spatialPointCoordinatesSafe=false) {
#else // TREE_HALF_COORD_CERT_1
    __aicore__ inline void Process(uint32_t rangeBegin=0,uint32_t rangeEnd=0,GM_ADDR spatialWorkspace=nullptr) {
#endif // TREE_HALF_COORD_CERT_1
        const uint32_t total=t_.batches*t_.points;
        const uint32_t begin=rangeEnd?rangeBegin:total*GetBlockIdx()/t_.blocks,end=rangeEnd?rangeEnd:total*(GetBlockIdx()+1)/t_.blocks;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        bool spatialCoefficientDirty=false,spatialPruned=false;
        #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        uint32_t spatialDirtyCount=COEFFICIENTS;
#endif
        GlobalTensor<float> spatialCoefficients;
        if(spatialWorkspace)spatialCoefficients.SetGlobalBuffer((__gm__ float*)spatialWorkspace+6*4096);
#endif
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        bool corePointSafe=false,sharedCoefficientCoordinatesSafe=false;
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL)
            // Sorting has a wider range domain than the numerical certificate.
            // Validate each sorted full tile before using the shared bound.
            corePointSafe=spatialWorkspace?false:CertifyCorePointCoordinates(begin,end);
#endif
        uint32_t cached=0xffffffffU;
        bool prefetchedPoint=false,outputInFlight=false;
        for(uint32_t position=begin;position<end;) {
            const uint32_t batch=position/t_.points,start=position%t_.points;
            const uint32_t validRows=Smallest(Smallest(end-position,t_.points-start),ROWS);
            uint32_t logicalSegments=t_.segments;
            liveRows_=validRows;
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
            bool exactPointSafe=corePointSafe;
#endif // TREE_STREAM_EXACT
            count_=COEFFICIENTS;
            bool initialInputs=false;
            if constexpr(CAPACITY==256 && sizeof(T)==4)initialInputs=cached==0xffffffffU && position==begin;
            if(batch!=cached) {
                if constexpr(OVERLAP_POINT_INPUT) {
                    if(outputInFlight){WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);outputInFlight=false;}
                }
                wp_=m_+32;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
                if constexpr(OUTLINE_FULL_FLOAT_FALLBACK && CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                    LoadFullFloatSegmentsStatic((GM_ADDR)sg_.GetPhyAddr(),batch,t_.segments);
                } else
#endif
                {
                    if constexpr(DEFER_DIRECTION_LOW && ROWS==64) {
                        LoadBoundedResidentSegments((GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),t_.segments,initialInputs);
                        directionLowReady_=false;
                    } else if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)
                        LoadResidentSegments(batch,initialInputs);
                    else LoadSegments(batch,initialInputs);
                }
                wp_=ROWS*64+128;cached=batch;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
                if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                    sharedCoefficientCoordinatesSafe=CoarseCoordinatesSafe<true>();
                    tightCoefficientSafe_=Smallest(end-position,t_.points-start)>=ROWS*8 &&
                        sharedCoefficientCoordinatesSafe && CoarseDirectionSafe();
                } else
#endif
                if constexpr(GUARDED_ERROR)tightCoefficientSafe_=Smallest(end-position,t_.points-start)>=ROWS*8 &&
                    CoarseCoordinatesSafe<true>() && CoarseDirectionSafe();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_EXACT_TAIL
                if constexpr(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && COMPACT_TAIL)
                    sharedCoefficientCoordinatesSafe=CoarseCoordinatesSafe<true>();
#endif // TREE_EXACT_TAIL
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
                if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                    if(sharedCoefficientCoordinatesSafe) {
                        // Unused anchor-column padding holds one reduced absolute maximum per axis.
                        #pragma unroll
                        for(uint32_t axis=0;axis<3;++axis) {
                            Abs(W(0),C(axis),COEFFICIENTS);PipeBarrier<PIPE_V>();
                            WholeReduceMax(record_,W(0),64,COEFFICIENTS/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                            WholeReduceMax(C(axis)[CAPACITY+16],record_,COEFFICIENTS/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                        }
                    }
                }
#endif
            }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                if(spatialWorkspace) {
                    if(position==begin) {
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                        DataCopy(spatialCoefficients,columns_,14*cp_);
                        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                    } else if(spatialCoefficientDirty) {
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                        DataCopy(columns_,spatialCoefficients,14*cp_);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                    }
                    spatialPruned=false;
                }
            }
#endif
            if constexpr(OVERLAP_POINT_INPUT) {
                if(!prefetchedPoint) {
                    DataCopyPad(rawP_,pg_[position*3],DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
                }
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);prefetchedPoint=false;
            } else {
                PipeBarrier<PIPE_ALL>();
                if(!initialInputs) {
                    DataCopyPad(rawP_,pg_[(batch*t_.points+start)*3],DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
                    PipeBarrier<PIPE_ALL>();
                }
            }
            if constexpr(sizeof(T)==2)Cast(pf_,rawP_,RoundMode::CAST_NONE,192);
            PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            if constexpr(OUTLINE_FULL_FLOAT_FALLBACK && CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                uint32_t spatialQueryRows=liveRows_;
                auto queryOffsets=pointOffset_;
                if(spatialWorkspace && liveRows_<64U) {
                    // Repeated genuine queries complete the UB tile without a GM overread.
                    Mins(offset_,pointOffset_,int32_t((liveRows_-1U)*12U),64);
                    PipeBarrier<PIPE_V>();
                    queryOffsets=offset_;
                    spatialQueryRows=64U;
                }
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)
                    Gather(compactP_[axis*64],pf_,queryOffsets.template ReinterpretCast<uint32_t>(),axis*4,spatialQueryRows);
            } else
#endif
            {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Gather(compactP_[axis*64],pf_,pointOffset_.template ReinterpretCast<uint32_t>(),axis*4,liveRows_);
            }
            }
            PipeBarrier<PIPE_V>();
            if constexpr(OVERLAP_POINT_INPUT) {
                // The packed input is dead after all coordinate gathers.
                // Refill its storage while current coordinates remain in compactP/xyz.
                const uint32_t nextPosition=position+validRows;
                if(nextPosition<end) {
                    const uint32_t nextPoint=nextPosition%t_.points;
                    const uint32_t nextRows=Smallest(Smallest(end-nextPosition,t_.points-nextPoint),ROWS);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
                    DataCopyPad(rawP_,pg_[nextPosition*3],DataCopyExtParams{1,nextRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);prefetchedPoint=true;
                }
            }
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                if constexpr(OUTLINE_FULL_FLOAT_FALLBACK && CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                    Brcb(P(axis),compactP_[axis*64],spatialWorkspace?8U:(liveRows_+7)/8,BrcbRepeatParams(1,8));
                } else
#endif
                Brcb(P(axis),compactP_[axis*64],(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            }
            if constexpr(OVERLAP_POINT_INPUT) {
                // Search is the next writer of the prior output index buffer.
                if(outputInFlight){WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);outputInFlight=false;}
            }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                if(spatialWorkspace && spatialCoefficientDirty) {
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                    spatialCoefficientDirty=false;
                }
            }
#endif
            if constexpr(GUARDED_ERROR) {
                if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2
#ifdef DTYPE_TREE_POINTS
                    && sizeof(DTYPE_TREE_POINTS)==2
#endif
                ) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
                    if constexpr(ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL)
                        count_=SearchGuardedHalfTypeCertifiedRows(liveRows_,t_.segments,tightCoefficientSafe_);
                    else
#endif
                        SearchGuardedHalfCachedRows();
                }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
                else if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_3
                    bool spatialPointSafe=false;
                    if(spatialWorkspace && sharedCoefficientCoordinatesSafe && (liveRows_==64 || (OUTLINE_FULL_FLOAT_FALLBACK && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)))) {
                        if(spatialPointCoordinatesSafe) {
                            // The launch's producer certificate already proves every input coordinate.
                            // PruneSpatialStaticState overwrites its scratch and SearchSpatialStaticState
                            // produces the actual row predicates before their first consumer.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
                            Abs(W(1),compactP_,192);PipeBarrier<PIPE_V>();
                            Duplicate(record_.template ReinterpretCast<uint16_t>(),uint16_t(65535),12);
                            CompareScalar(mask_,W(1),0.0F,CMPMODE::EQ,192);PipeBarrier<PIPE_V>();
                            Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),12);PipeBarrier<PIPE_V>();
                            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
#endif
                            spatialPointSafe=true;
                        } else spatialPointSafe=CoarseCoordinatesSafe<false>();
                    }
                    if(spatialPointSafe) {
#else // TREE_HALF_COORD_CERT_3
                    if(spatialWorkspace && sharedCoefficientCoordinatesSafe &&
                       (liveRows_==64 || OUTLINE_FULL_FLOAT_FALLBACK) && CoarseCoordinatesSafe<false>()) {
#endif // TREE_HALF_COORD_CERT_3
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
                        exactPointSafe=true;
#endif // TREE_STREAM_EXACT
                        const uint32_t limit=PruneSpatialStaticState(spatialWorkspace,t_.segments,spatialPruned,logicalSegments);
                        spatialCoefficientDirty=spatialPruned;
                        #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                        spatialDirtyCount=spatialPruned?limit:COEFFICIENTS;
#endif
                        // The helper consumes the map event before its first
                        // overlapping W5 write (or before an empty resume returns).
                        count_=SearchSpatialStaticState(0,limit,false,tightCoefficientSafe_);
                    } else {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
                        exactPointSafe=
#endif // TREE_STREAM_EXACT
                        SearchGuardedCoreCertifiedRows(corePointSafe,sharedCoefficientCoordinatesSafe);
                    }
                }
#endif
                else SearchGuardedError();
            }
            else if constexpr(NATIVE_HALF_PREFIX_PRIMARY && ROWS==64)count_=SearchHalfPrefixOutlined(liveRows_,t_.segments);
            else Search();
            if constexpr(CLEAN_PRIMARY64 && (sizeof(T)==4 || sizeof(T)==2) && ROWS==64) {
                And(smallMask_.template ReinterpretCast<uint16_t>(),second_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),4);
                PipeBarrier<PIPE_V>();
            } else {
                Compare(smallMask_,second_,hi_,CMPMODE::GT,64);
                PipeBarrier<PIPE_V>();
                And(smallMask_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t certified=smallMask_.template ReinterpretCast<uint64_t>().GetValue(0);
            const uint64_t active=(~uint64_t(0))>>(64-validRows);
            uint64_t exactRows=active&~certified,uniqueRows=0;
            if constexpr(SAVE_PREFIX_CHUNK_PREDICATES || CLEAN_PRIMARY64) {
                if((certified&active)!=active) {
                    // Search is finished. Preserve all four predicate groups before
                    // primary gathers or fallback point broadcasts reuse their storage.
                    #pragma unroll
                    for(uint32_t group=0;group<4;++group) {
                        auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                        Copy(work_[14*1056+group*512],predicates,64,8,CopyRepeatParams(1,1,8,8));
                    }
                    PipeBarrier<PIPE_V>();
                }
            }
            if constexpr(DEFER_DIRECTION_LOW) {
                // The genuine score certificate identifies pending refinement.
                // Prefetch its bounded helper while primary gathers/computation run.
                if(exactRows)AscendC::PreLoadImpl(reinterpret_cast<void*>(&KernelTreeBatch::RefineFewStatic),int64_t(2));
            }
            uint32_t refinementRows=64;
            if constexpr(CAPACITY==256 && sizeof(T)==4)refinementRows=validRows<=16?16:validRows<=32?32:64;
            Muls(offset_,idx_,int32_t(4),64);PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t col=0;col<13;++col) {
                if constexpr((CLEAN_PRIMARY64 && sizeof(T)==2) || NATIVE_HALF_PREFIX_PRIMARY) {
                    if(col<6 || (col>9 && !exactDirectionLow_))
                        Gather(refined_[col*64],C(col),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
                } else {
                    if(col<6 || (col>9 && !DEFER_DIRECTION_LOW))Gather(refined_[col*64],C(col),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
                }
            }
            Gather(refined_[9*64],C(13),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
            PipeBarrier<PIPE_V>();
            if constexpr(CAPACITY==256 && sizeof(T)==4) {
                CompareScalar(valid_,segmentIndices_,float(int32_t(validRows)),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
                if(validRows<=16)PrimaryCompensated16();
                else if(validRows<=32)PrimaryCompensated32();
                else PrimaryCompensatedFull();
                Copy(tileSecond_,work_,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                count_=m_;wp_=m_+32;
            } else if constexpr(DEFER_CHUNK_CERTIFICATE && (sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
                || sizeof(DTYPE_TREE_POINTS)==2
#endif
            )) {
                if constexpr(!CLEAN_PRIMARY64) {
                    Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),128);PipeBarrier<PIPE_V>();
                }
                if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
                    if(exactDirectionLow_)PrimaryCompensatedExactDirection();
                    else PrimaryCompensatedFull();
                } else PrimaryCompensatedFull();
                Copy(tileSecond_,work_,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                count_=m_;wp_=m_+32;
            } else if constexpr(NATIVE_HALF_PREFIX_PRIMARY) {
                Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),128);PipeBarrier<PIPE_V>();
                if(exactDirectionLow_)PrimaryCompensatedExactDirection();
                else PrimaryCompensatedFull();
                Copy(tileSecond_,work_,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                count_=m_;wp_=m_+32;
            } else {
            auto originalColumns=columns_,originalPoints=xyz_;
            const uint32_t originalCp=cp_,originalPp=pp_;
            columns_=refined_;cp_=64;xyz_=compactP_;pp_=64;count_=refinementRows;wp_=96;
            if constexpr(!NATIVE_FLOAT_PREFIX_SPARSE) {
                Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),128);PipeBarrier<PIPE_V>();
            }
            Compensated();
            Copy(tileSecond_,W(0),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            columns_=originalColumns;cp_=originalCp;xyz_=originalPoints;pp_=originalPp;count_=m_;wp_=m_+32;
            }
            if(__builtin_expect((certified & active) != active,0)) {
                bool cachedChunks=false;
                if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
                    cachedChunks=PredicatePopulation(active&~certified)>=8;
                    if(cachedChunks)BuildOverlapChunks(t_.segments);
                }
                uint64_t acceptedSparseRows=0;
                if constexpr(SAVE_PREFIX_CHUNK_PREDICATES || (CLEAN_PRIMARY64 && sizeof(T)==4))
                    acceptedSparseRows=RefineSparseRows(active&~certified,validRows,logicalSegments,uniqueRows);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                    if(spatialPruned && spatialDirtyCount<COEFFICIENTS && ((certified|acceptedSparseRows)&active)!=active) {
                        // Selected sparse refiners read only searched packets. Any remaining
                        // fallback may scan the whole cache, so finish its original duplicate tail.
                        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                        for(uint32_t col=0;col<14;++col)
                            Brcb(work_[col*64U],C(col)[spatialDirtyCount-8U],1,BrcbRepeatParams(1,8));
                        PipeBarrier<PIPE_V>();
                        for(uint32_t col=0;col<14;++col)
                            Copy(C(col)[spatialDirtyCount],work_[col*64U+56U],64,(COEFFICIENTS-spatialDirtyCount)/64U,CopyRepeatParams(1,0,8,0));
                        PipeBarrier<PIPE_V>();
                        spatialDirtyCount=COEFFICIENTS;
                    }
                }
#endif
                constexpr bool sparseRowDispatch=DEFER_DIRECTION_LOW || NATIVE_HALF_PREFIX_PRIMARY || (CLEAN_PRIMARY64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && (sizeof(T)==4 || sizeof(T)==2)
#ifdef DTYPE_TREE_POINTS
                    && (sizeof(T)==4 || sizeof(DTYPE_TREE_POINTS)==sizeof(T))
#endif
                    );
                if constexpr(sparseRowDispatch) {
                    for(uint64_t pendingRows=active&~(certified|acceptedSparseRows);pendingRows!=0;pendingRows&=pendingRows-1) {
                        uint32_t row;
                        if constexpr(sizeof(T)==4) row=uint32_t(GetSFFValue<1>(pendingRows));
                        else row=uint32_t(__builtin_ctzll(pendingRows));
                    // Only chunks whose conservative lower bound overlaps the incumbent can win.
                    uint32_t chunks;
                    if(cachedChunks)chunks=SavedOverlapChunks().GetValue(row);
                    else {
                    Muls(offset_,outputOffset_,int32_t(32),16);PipeBarrier<PIPE_V>();
                    Adds(offset_,offset_,int32_t(row*4),16);
                    Duplicate(W(0),__builtin_inff(),64);
                    Brcb(W(1),hi_[(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                    Copy(W(2),W(1)[(row%8)*8],64,1,CopyRepeatParams(1,0,8,0));
                    Gather(W(0),chunkLower_,offset_.template ReinterpretCast<uint32_t>(),0U,(logicalSegments+63)/64);PipeBarrier<PIPE_V>();
                    Compare(smallMask_,W(0),W(2),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    chunks=smallMask_.template ReinterpretCast<uint32_t>().GetValue(0)&((1U<<((logicalSegments+63)/64))-1);
                    }
                    if constexpr(CAPACITY==256 && sizeof(T)==4) {
                        if(RefineFewStatic(row,chunks,uniqueRows))continue;
                    }
                    if constexpr(DEFER_DIRECTION_LOW)MaterializeDirectionLow();
                    if constexpr(CLEAN_PRIMARY64) {
                        if(chunks!=0 && (chunks&(chunks-1))!=0) {
                            if(RefineOverlappingFew(row,chunks,uniqueRows))continue;
                        } else if(RefineWinningFew(row,chunks,uniqueRows))continue;
                    } else if constexpr(CAPACITY==1024 && COEFFICIENTS==1024 && (sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
                        || sizeof(DTYPE_TREE_POINTS)==2
#endif
                    )) {
                        if(RefineWinningFew(row,chunks,uniqueRows))continue;
                    }
                    if constexpr(DEFER_DIRECTION_LOW) {
                        if(RefineBoundedFullRow(row,chunks,logicalSegments))uniqueRows|=uint64_t(1)<<row;
                        count_=m_;wp_=m_+32;
                    } else
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
                    if constexpr(OUTLINE_FULL_FLOAT_FALLBACK && CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                        if(RefineFullFloatFallback(row,chunks,logicalSegments))uniqueRows|=uint64_t(1)<<row;
                    } else
#endif
                    {
                    uint32_t first=0,second=0,hits=0;
                    for(uint32_t chunk=0;chunk<(logicalSegments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
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
                    CompareScalar(valid_,W(7),float(int32_t(logicalSegments)),CMPMODE::LT,candidates);PipeBarrier<PIPE_V>();
                    if constexpr(CAPACITY==256 && sizeof(T)==4) {
                        if(sparse) {
                            if(hits==1)SparseCompensated<64>();
                            else SparseCompensated<128>();
                        } else Compensated<true>();
                    } else Compensated<true>();
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
                    if(CertifyWideScore(candidates))uniqueRows|=uint64_t(1)<<row;
                    columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
                    CommitRefinedCandidate(row);
                    }
                    }
                } else {
                for(uint32_t row=0;row<validRows;++row) if(!((certified|acceptedSparseRows)&(uint64_t(1)<<row))) {
                    // Only chunks whose conservative lower bound overlaps the incumbent can win.
                    uint32_t chunks;
                    if(cachedChunks)chunks=SavedOverlapChunks().GetValue(row);
                    else {
                    Muls(offset_,outputOffset_,int32_t(32),16);PipeBarrier<PIPE_V>();
                    Adds(offset_,offset_,int32_t(row*4),16);
                    Duplicate(W(0),__builtin_inff(),64);
                    Brcb(W(1),hi_[(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                    Copy(W(2),W(1)[(row%8)*8],64,1,CopyRepeatParams(1,0,8,0));
                    Gather(W(0),chunkLower_,offset_.template ReinterpretCast<uint32_t>(),0U,(logicalSegments+63)/64);PipeBarrier<PIPE_V>();
                    Compare(smallMask_,W(0),W(2),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    chunks=smallMask_.template ReinterpretCast<uint32_t>().GetValue(0)&((1U<<((logicalSegments+63)/64))-1);
                    }
                    if constexpr(CAPACITY==256 && sizeof(T)==4) {
                        if(RefineFewStatic(row,chunks,uniqueRows))continue;
                    }
                    if constexpr(DEFER_DIRECTION_LOW)MaterializeDirectionLow();
                    if constexpr(CLEAN_PRIMARY64) {
                        if(chunks!=0 && (chunks&(chunks-1))!=0) {
                            if(RefineOverlappingFew(row,chunks,uniqueRows))continue;
                        } else if(RefineWinningFew(row,chunks,uniqueRows))continue;
                    } else if constexpr(CAPACITY==1024 && COEFFICIENTS==1024 && (sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
                        || sizeof(DTYPE_TREE_POINTS)==2
#endif
                    )) {
                        if(RefineWinningFew(row,chunks,uniqueRows))continue;
                    }
                    uint32_t first=0,second=0,hits=0;
                    for(uint32_t chunk=0;chunk<(logicalSegments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
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
                    CompareScalar(valid_,W(7),float(int32_t(logicalSegments)),CMPMODE::LT,candidates);PipeBarrier<PIPE_V>();
                    if constexpr(CAPACITY==256 && sizeof(T)==4) {
                        if(sparse) {
                            if(hits==1)SparseCompensated<64>();
                            else SparseCompensated<128>();
                        } else Compensated<true>();
                    } else Compensated<true>();
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
                    if(CertifyWideScore(candidates))uniqueRows|=uint64_t(1)<<row;
                    columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
                    CommitRefinedCandidate(row);
                }
                }
            }
            Sqrt(tileSecond_,tileSecond_,64);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL) {
                if(spatialPruned) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                    const LocalTensor<int32_t> originalIndices(TPosition::VECCALC,192512U,1024U);
#else
                    auto originalIndices=work_.template ReinterpretCast<int32_t>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                    GlobalTensor<int32_t> originalMap;originalMap.SetGlobalBuffer((__gm__ int32_t*)spatialWorkspace+6*4096+14*1056);
                    DataCopy(originalIndices,originalMap,COEFFICIENTS);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
#endif
                    Muls(tileIdx_,idx_,int32_t(4),64);PipeBarrier<PIPE_V>();
                    Gather(idx_,originalIndices,tileIdx_.template ReinterpretCast<uint32_t>(),0U,liveRows_);PipeBarrier<PIPE_V>();
                }
            }
#endif
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
            if constexpr(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024) {
                if(!exactPointSafe || !sharedCoefficientCoordinatesSafe)
                    RefineWideFloatRows((GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),t_.segments,
                        uint32_t(work_.GetPhyAddr()),uint32_t(compactP_.GetPhyAddr()),
                        uint32_t(idx_.GetPhyAddr()),uint32_t(tileSecond_.GetPhyAddr()),liveRows_,sharedCoefficientCoordinatesSafe);
            }
#endif // TREE_STREAM_EXACT
            auto selectedOutput=outT_;
            if constexpr(sizeof(T)==2)Cast(selectedOutput,tileSecond_,RoundMode::CAST_RINT,64);
            else selectedOutput=tileSecond_.template ReinterpretCast<T>();
            if constexpr(OVERLAP_POINT_INPUT) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
            } else PipeBarrier<PIPE_ALL>();
            DataCopyPad(dg_[batch*t_.points+start],selectedOutput,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
            DataCopyPad(ig_[batch*t_.points+start],idx_,DataCopyExtParams{1,validRows*4,0,0,0});
            if constexpr(OVERLAP_POINT_INPUT) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID3);outputInFlight=true;
            } else PipeBarrier<PIPE_ALL>();
#if defined(DTYPE_TREE_POINTS)
            exactRows&=~uniqueRows;
            if(exactRows) {
                if constexpr(OVERLAP_POINT_INPUT) {
                    if(prefetchedPoint){WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);prefetchedPoint=false;}
                    if(outputInFlight){WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);outputInFlight=false;}
                }
                PipeBarrier<PIPE_ALL>();
                for(uint32_t row=0;row<validRows;++row)if(exactRows&(uint64_t(1)<<row)) {
                    const uint32_t pos=position+row;
                    TreeGeneralExactFallback<T>((GM_ADDR)pg_[pos*3].GetPhyAddr(),
                        (GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),
                        (GM_ADDR)dg_[pos].GetPhyAddr(),(GM_ADDR)ig_[pos].GetPhyAddr(),t_.segments);
                }
                if(position+validRows<end) {
                    if constexpr(DEFER_DIRECTION_LOW)RestartBoundedInlineAfterExact();
                    else {
                        const TreeLineMinDistanceTilingData t=t_;
                        Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
                             (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
                    }
                    cached=0xffffffffU;
                    spatialCoefficientDirty=false;
                }
            }
#endif
            wp_=ROWS*64+128;
            position+=validRows;
        }
        if constexpr(OVERLAP_POINT_INPUT) {
            if(outputInFlight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
            PipeBarrier<PIPE_ALL>();
        }
    }
protected:
    __aicore__ inline uint32_t Smallest(uint32_t a,uint32_t b){return a<b?a:b;}
    __aicore__ inline LocalTensor<float> C(uint32_t k){
        const uint32_t offset=(cp_==m_+32 && k>=6 && k<=8)?6*cp_+(k-6)*m_:k*cp_;
        return columns_[offset];
    }
    __aicore__ inline LocalTensor<float> W(uint32_t k){
        constexpr uint32_t bankOffset[13]={64,0,0,64,96,32,96,0,0,0,0,0,0};
        constexpr bool fullFloatCache=sizeof(T)==4 && CAPACITY==1024 && COEFFICIENTS==1024;
        constexpr bool fullHalfCache=sizeof(T)==2 && CAPACITY==1024 && COEFFICIENTS==1024
#ifdef DTYPE_TREE_POINTS
            && sizeof(DTYPE_TREE_POINTS)==2
#endif
            ;
        // Native full caches use the same APx phase candidate within existing padding.
        const uint32_t phase=(fullFloatCache || fullHalfCache) && k==0?32U:bankOffset[k];
        return work_[k*wp_+(wp_==ROWS*64+128?phase:0)];
    }
    __aicore__ inline LocalTensor<float> P(uint32_t k){return xyz_[k*pp_];}
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SearchC(uint32_t k) {
        if constexpr(FIXED_ARENA ? (CAPACITY==256 && sizeof(T)==4) : NATIVE_FLOAT_PREFIX_SPARSE) {
            constexpr uint32_t pitch=CAPACITY+32;
            const uint32_t offset=k>=6 && k<=8?6*pitch+(k-6)*CAPACITY:k*pitch;
            return LocalTensor<float>(TPosition::VECCALC,offset*4,k>=6 && k<=8?CAPACITY:pitch);
        } else return C(k);
    }
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SearchW(uint32_t k) {
        if constexpr(FIXED_ARENA ? (CAPACITY==256 && sizeof(T)==4) : NATIVE_FLOAT_PREFIX_SPARSE) {
            constexpr uint32_t pitch=ROWS*64+128;
            constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
            constexpr uint32_t banks[13]={64,0,0,64,96,32,96,0,0,0,0,0,0};
            const uint32_t offset=k*pitch+banks[k];
            return LocalTensor<float>(TPosition::VECCALC,(base+offset)*4,6*pitch-offset);
        } else return W(k);
    }
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SearchP(uint32_t k) {
        if constexpr(FIXED_ARENA ? (CAPACITY==256 && sizeof(T)==4) : NATIVE_FLOAT_PREFIX_SPARSE) {
            constexpr uint32_t pitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t base=14*(CAPACITY+32);
            return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,pitch);
        } else return P(k);
    }

    // Copy handles so grouped initialization never changes the caller's tensor view.
    __aicore__ __attribute__((always_inline)) static inline void InitializeGroupedSearchState(
        LocalTensor<float> initialDistances,const LocalTensor<int32_t>& indices,uint32_t indexRepeats) {
        auto initialIndices=indices.template ReinterpretCast<float>();
        initialDistances.SetSize(3*64);initialIndices.SetSize(3*64);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<float,false>(initialDistances,__builtin_inff(),uint64_t(64),3,1,8);
        Duplicate<float,false>(initialIndices,0.0F,uint64_t(64),indexRepeats,1,8);
    }
    template<bool FIXED_ARENA,bool DIRECT_PREDICATES=false>
    __aicore__ __attribute__((always_inline)) inline void MergeSearchMinimum(uint32_t j) {
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        if constexpr(FIXED_ARENA && !DIRECT_PREDICATES) {
            Copy(chunkLower_[512+j*2],SearchW<true>(0),64,1,CopyRepeatParams(1,1,8,8));
            Copy(chunkLower_[512+j*2+64],SearchW<true>(1),64,1,CopyRepeatParams(1,1,8,8));
        }

            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,false>(tileLo_,lo_,CMPMODE::LT,uint64_t(64),same);
            Max<float,false>(record_,tileLo_,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,tileLo_,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,tileHi_,hi_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,(DIRECT_PREDICATES?chunkLower_[512+j*2]:SearchW<FIXED_ARENA>(0)),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,(DIRECT_PREDICATES?chunkLower_[512+j*2+64]:SearchW<FIXED_ARENA>(1)),certHigh_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
    }
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline void CompactSearchTail(uint32_t j,uint32_t tail) {
        // Tail segments use complete point vectors instead of padded segment rows.
        #pragma unroll
        for(uint32_t col=0;col<10;++col)if(col<3||col>=6)
            Brcb(refined_[col*64],SearchC<FIXED_ARENA>(col)[j],1,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const BinaryRepeatParams same{1,1,1,8,8,8},coefficient{1,1,0,8,8,1},pointCoefficient{1,1,0,8,0,1};
        const UnaryRepeatParams unary{1,1,8,8};
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Sub<float,false>(SearchW<FIXED_ARENA>(axis),compactP_[axis*64],refined_[axis*64],uint64_t(64),tail,pointCoefficient);
        PipeBarrier<PIPE_V>();
        Mul<float,false>(SearchW<FIXED_ARENA>(3),SearchW<FIXED_ARENA>(0),refined_[6*64],uint64_t(64),tail,coefficient);
        Mul<float,false>(SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(0),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(3),SearchW<FIXED_ARENA>(1),refined_[7*64],uint64_t(64),tail,coefficient);
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(1),SearchW<FIXED_ARENA>(1),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(3),SearchW<FIXED_ARENA>(2),refined_[8*64],uint64_t(64),tail,coefficient);
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(2),SearchW<FIXED_ARENA>(2),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        Maxs<float,false>(SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(3),0.0F,uint64_t(64),tail,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(0),refined_[9*64],uint64_t(64),tail,coefficient);PipeBarrier<PIPE_V>();
        Muls<float,false>(SearchW<FIXED_ARENA>(2),SearchW<FIXED_ARENA>(3),-2.0F,uint64_t(64),tail,unary);
        Muls<float,false>(SearchW<FIXED_ARENA>(5),SearchW<FIXED_ARENA>(4),1.52587890625e-5F,uint64_t(64),tail,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(SearchW<FIXED_ARENA>(1),SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(2),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(1),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        Sub<float,false>(SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(5),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        if constexpr(SINGLE_TAIL) {
            Copy(tileLo_,SearchW<FIXED_ARENA>(0),64,1,CopyRepeatParams(1,1,8,8));
            Muls(tileHi_,SearchW<FIXED_ARENA>(5),2.125F,64);Duplicate(tileIdx_,int32_t(0),64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);
            Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(SearchW<FIXED_ARENA>(0).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Duplicate(SearchW<FIXED_ARENA>(1),0.0F,64);PipeBarrier<PIPE_V>();
        } else {
            Duplicate(tileLo_,__builtin_inff(),64);Duplicate(tileHi_,__builtin_inff(),64);
            Duplicate(tileIdx_,int32_t(0),64);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<tail;++r) {
                Compare<float>(SearchW<FIXED_ARENA>(0)[r*64],tileLo_,CMPMODE::LT,uint64_t(64),same);
                Duplicate(offset_,int32_t(r),64);PipeBarrier<PIPE_V>();
                Min(tileLo_,SearchW<FIXED_ARENA>(0)[r*64],tileLo_,64);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileHi_,SearchW<FIXED_ARENA>(5)[r*64],tileHi_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileIdx_.template ReinterpretCast<float>(),offset_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
            Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);
            Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(SearchW<FIXED_ARENA>(2),0.0F,64);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<tail;++r) {
                Compare(mask_,SearchW<FIXED_ARENA>(0)[r*64],tileHi_,CMPMODE::LE,64);
                Duplicate(SearchW<FIXED_ARENA>(3).template ReinterpretCast<int32_t>(),int32_t(1U<<r),64);PipeBarrier<PIPE_V>();
                Select(SearchW<FIXED_ARENA>(3),mask_,SearchW<FIXED_ARENA>(3),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                Or(SearchW<FIXED_ARENA>(2).template ReinterpretCast<uint16_t>(),SearchW<FIXED_ARENA>(2).template ReinterpretCast<uint16_t>(),SearchW<FIXED_ARENA>(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            }
            Copy(SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(2),64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(SearchW<FIXED_ARENA>(1),0.0F,64);PipeBarrier<PIPE_V>();
        }
    }
    template<bool DIRECT_PREDICATES,uint32_t FIXED_SEARCH_ROWS=0>
    __aicore__ __attribute__((always_inline)) inline void SearchStaticPolicy() {
        const uint32_t liveRows_=FIXED_SEARCH_ROWS?FIXED_SEARCH_ROWS:this->liveRows_;
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        constexpr bool explicitStageBarrier = !DIRECT_PREDICATES || SINGLE_TAIL;
#else
        constexpr bool explicitStageBarrier = true;
#endif
        if constexpr(sizeof(T)==4 && (CAPACITY==1024 || CAPACITY==256)) {
            InitializeGroupedSearchState(lo_,idx_,3);
        } else {
            Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);
        }
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(liveRows_*64);
        // The one-tail 192-coefficient resource has two complete packets and one tail.
        const uint32_t segmentCount=(CAPACITY==256 && COEFFICIENTS==192 && SINGLE_TAIL)?129U:t_.segments;
        #pragma unroll
        for(uint32_t j=0;j<segmentCount;j+=64) {
            if constexpr(COMPACT_TAIL) {
                if(segmentCount-j<=8) {
                    CompactSearchTail<true>(j,segmentCount-j);MergeSearchMinimum<true>(j);continue;
                }
            }
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(SearchW<true>(axis),SearchP<true>(axis),SearchC<true>(axis)[j],uint64_t(64),liveRows_,pointSegment);
            if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
            Mul<float,false>(SearchW<true>(3),SearchW<true>(0),SearchC<true>(6)[j],uint64_t(64),liveRows_,segment);
            Mul<float,false>(SearchW<true>(4),SearchW<true>(0),SearchW<true>(0),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(SearchW<true>(3),SearchW<true>(1),SearchC<true>(7)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(SearchW<true>(4),SearchW<true>(1),SearchW<true>(1),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(SearchW<true>(3),SearchW<true>(2),SearchC<true>(8)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(SearchW<true>(4),SearchW<true>(2),SearchW<true>(2),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
            Maxs<float,false>(SearchW<true>(0),SearchW<true>(3),0.0F,uint64_t(64),liveRows_,unary);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
            Min<float,false>(SearchW<true>(0),SearchW<true>(0),SearchC<true>(9)[j],uint64_t(64),liveRows_,segment);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
            Muls<float,false>(SearchW<true>(2),SearchW<true>(3),-2.0F,uint64_t(64),liveRows_,unary);
            Muls<float,false>(SearchW<true>(5),SearchW<true>(4),1.52587890625e-5F,uint64_t(64),liveRows_,unary);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
            Add<float,false>(SearchW<true>(1),SearchW<true>(0),SearchW<true>(2),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(SearchW<true>(4),SearchW<true>(0),SearchW<true>(1),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
            Sub<float,false>(SearchW<true>(0),SearchW<true>(4),SearchW<true>(5),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
            if(j+64>segmentCount) {
                CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(segmentCount-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
                Select(SearchW<true>(0),valid_,SearchW<true>(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
            }
            WholeReduceMin(record_,SearchW<true>(0),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
            Gather(tileHi_,SearchW<true>(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);PipeBarrier<PIPE_V>();
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
            Brcb(SearchW<true>(1),tileHi_,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,uint8_t,false>(mask_,SearchW<true>(0),SearchW<true>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
            if constexpr(DIRECT_PREDICATES) {
                GatherMask(chunkLower_[512+j*2],mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                GatherMask(chunkLower_[512+j*2+64],mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            } else {
                GatherMask(SearchW<true>(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                GatherMask(SearchW<true>(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            }
            if constexpr(COMPACT_TAIL) {
                if constexpr(DIRECT_PREDICATES) MergeSearchMinimum<true,true>(j);
                else MergeSearchMinimum<true>(j);
            } else {
            if constexpr(!DIRECT_PREDICATES) {
                Copy(chunkLower_[512+j*2],SearchW<true>(0),64,1,CopyRepeatParams(1,1,8,8));
                Copy(chunkLower_[512+j*2+64],SearchW<true>(1),64,1,CopyRepeatParams(1,1,8,8));
            }

            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,false>(tileLo_,lo_,CMPMODE::LT,uint64_t(64),same);
            Max<float,false>(record_,tileLo_,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,tileLo_,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,tileHi_,hi_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,DIRECT_PREDICATES?chunkLower_[512+j*2]:SearchW<true>(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,DIRECT_PREDICATES?chunkLower_[512+j*2+64]:SearchW<true>(1),certHigh_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
        }
        // Refinement gathers a full register, including inactive point lanes.
        if(liveRows_<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<liveRows_)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        if constexpr(CAPACITY<1024) {
            // A unique bit across both words requires a nonzero one-bit union
            // and no bit occupied in both words.
            Duplicate(SearchW<true>(4).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Or(SearchW<true>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
            And(SearchW<true>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Sub(SearchW<true>(0).template ReinterpretCast<int32_t>(),SearchW<true>(2).template ReinterpretCast<int32_t>(),SearchW<true>(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            And(SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(2).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Or(SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(3).template ReinterpretCast<uint16_t>(),128);
            Cast(SearchW<true>(1),SearchW<true>(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Cast(SearchW<true>(0),SearchW<true>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,SearchW<true>(0),0.0F,CMPMODE::EQ,64);
            CompareScalar(smallMask_,SearchW<true>(1),0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        } else {
        Duplicate(SearchW<true>(4).template ReinterpretCast<int32_t>(),int32_t(31),64);PipeBarrier<PIPE_V>();
        And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<true>(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
        Gather(SearchW<true>(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        Or(SearchW<true>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
        And(SearchW<true>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Sub(SearchW<true>(0).template ReinterpretCast<int32_t>(),SearchW<true>(2).template ReinterpretCast<int32_t>(),SearchW<true>(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
        Or(SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Cast(SearchW<true>(0),SearchW<true>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,SearchW<true>(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ __attribute__((always_inline)) inline void SavePrefixTailPredicate(uint32_t start,uint32_t tail) {
        const uint32_t group=start/256,slot=(start/64)%4;
        auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
        auto words=predicates[slot*128].template ReinterpretCast<int32_t>();
        Duplicate(words,int32_t(0),128);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
        Duplicate<int32_t,false>(words,int32_t((uint32_t(1)<<tail)-1),uint64_t(64),2,1,8);
        PipeBarrier<PIPE_V>();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
    }

    // Float and Half prefix searches share operations; arena offsets are compile-time policies.
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrefixC(uint32_t k) {
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
            constexpr uint32_t pitch=CAPACITY+32;
            const uint32_t offset=k>=6 && k<=8?6*pitch+(k-6)*CAPACITY:k*pitch;
            return LocalTensor<float>(TPosition::VECCALC,offset*4,k>=6 && k<=8?CAPACITY:pitch);
            }

    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrefixW(uint32_t k) {
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
            constexpr uint32_t pitch=ROWS*64+128;
            constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
            constexpr uint32_t banks[13]={64,0,0,64,96,32,96,0,0,0,0,0,0};
            const uint32_t offset=k*pitch+banks[k];
            return LocalTensor<float>(TPosition::VECCALC,(base+offset)*4,6*pitch-offset);
            }

    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrefixP(uint32_t k) {
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
            constexpr uint32_t pitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t base=14*(CAPACITY+32);
            return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,pitch);
            }

    template<bool HALF_RESOURCE,bool MASK_READY=false>
    __aicore__ __attribute__((always_inline)) static inline void MergePrefixMinimum(uint32_t j) {
        constexpr uint32_t stateShift=HALF_RESOURCE?384U:0U;
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
        LocalTensor<float> certHigh_(TPosition::VECCALC,184256U+stateShift,64U);
        LocalTensor<float> cert_(TPosition::VECCALC,184000U+stateShift,64U);
        LocalTensor<float> hi_(TPosition::VECCALC,183488U+stateShift,64U);
        LocalTensor<int32_t> idx_(TPosition::VECCALC,183744U+stateShift,64U);
        LocalTensor<float> lo_(TPosition::VECCALC,182976U+stateShift,64U);
        LocalTensor<float> record_(TPosition::VECCALC,182400U+stateShift,128U);
        LocalTensor<float> second_(TPosition::VECCALC,183232U+stateShift,64U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U+stateShift,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185536U+stateShift,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,184768U+stateShift,64U);

        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
            if constexpr(!MASK_READY) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));}
            Compare<float,false>(tileLo_,lo_,CMPMODE::LT,uint64_t(64),same);
            Max<float,false>(record_,tileLo_,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,tileLo_,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,tileHi_,hi_,1,same);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
            if constexpr(!HALF_RESOURCE) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,PrefixW<HALF_RESOURCE>(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,PrefixW<HALF_RESOURCE>(1),certHigh_,1,same);
            }
#else
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,PrefixW<HALF_RESOURCE>(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,PrefixW<HALF_RESOURCE>(1),certHigh_,1,same);
#endif
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
    }

    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline void CompactPrefixTail(uint32_t j,uint32_t tail) {
        constexpr uint32_t stateShift=HALF_RESOURCE?384U:0U;
        constexpr uint32_t predicateShift=HALF_RESOURCE?256U:0U;
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
        LocalTensor<float> chunkLower_(TPosition::VECCALC,174208U+stateShift,1024U);
        LocalTensor<float> compactP_(TPosition::VECCALC,178304U+stateShift,192U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187072U+predicateShift,512U);
        LocalTensor<int32_t> offset_(TPosition::VECCALC,185792U+stateShift,64U);
        LocalTensor<float> refined_(TPosition::VECCALC,179072U+stateShift,832U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U+stateShift,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185536U+stateShift,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,184768U+stateShift,64U);

        // Tail segments use complete point vectors instead of padded segment rows.
        #pragma unroll
        for(uint32_t col=0;col<10;++col)if(col<3||col>=6)
            Brcb(refined_[col*64],PrefixC<HALF_RESOURCE>(col)[j],1,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const BinaryRepeatParams same{1,1,1,8,8,8},coefficient{1,1,0,8,8,1},pointCoefficient{1,1,0,8,0,1};
        const UnaryRepeatParams unary{1,1,8,8};
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Sub<float,false>(PrefixW<HALF_RESOURCE>(axis),compactP_[axis*64],refined_[axis*64],uint64_t(64),tail,pointCoefficient);
        PipeBarrier<PIPE_V>();
        Mul<float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(0),refined_[6*64],uint64_t(64),tail,coefficient);
        Mul<float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(1),refined_[7*64],uint64_t(64),tail,coefficient);
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(1),PrefixW<HALF_RESOURCE>(1),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(2),refined_[8*64],uint64_t(64),tail,coefficient);
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        Maxs<float,false>(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(3),0.0F,uint64_t(64),tail,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0),refined_[9*64],uint64_t(64),tail,coefficient);PipeBarrier<PIPE_V>();
        Muls<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),-2.0F,uint64_t(64),tail,unary);
        Muls<float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(4),1.52587890625e-5F,uint64_t(64),tail,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(PrefixW<HALF_RESOURCE>(1),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(2),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(1),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        Sub<float,false>(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(5),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        if constexpr(SINGLE_TAIL) {
            Copy(tileLo_,PrefixW<HALF_RESOURCE>(0),64,1,CopyRepeatParams(1,1,8,8));
            Muls(tileHi_,PrefixW<HALF_RESOURCE>(5),2.125F,64);Duplicate(tileIdx_,int32_t(0),64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);
            Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Duplicate(PrefixW<HALF_RESOURCE>(1),0.0F,64);PipeBarrier<PIPE_V>();
        } else {
            Duplicate(tileLo_,__builtin_inff(),64);Duplicate(tileHi_,__builtin_inff(),64);
            Duplicate(tileIdx_,int32_t(0),64);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<tail;++r) {
                Compare<float>(PrefixW<HALF_RESOURCE>(0)[r*64],tileLo_,CMPMODE::LT,uint64_t(64),same);
                Duplicate(offset_,int32_t(r),64);PipeBarrier<PIPE_V>();
                Min(tileLo_,PrefixW<HALF_RESOURCE>(0)[r*64],tileLo_,64);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileHi_,PrefixW<HALF_RESOURCE>(5)[r*64],tileHi_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileIdx_.template ReinterpretCast<float>(),offset_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
            Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);
            Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(PrefixW<HALF_RESOURCE>(2),0.0F,64);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<tail;++r) {
                Compare(mask_,PrefixW<HALF_RESOURCE>(0)[r*64],tileHi_,CMPMODE::LE,64);
                Duplicate(PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<int32_t>(),int32_t(1U<<r),64);PipeBarrier<PIPE_V>();
                Select(PrefixW<HALF_RESOURCE>(3),mask_,PrefixW<HALF_RESOURCE>(3),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                Or(PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            }
            Copy(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(2),64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(PrefixW<HALF_RESOURCE>(1),0.0F,64);PipeBarrier<PIPE_V>();
        }
    }

    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline void SavePrefixPredicate(uint32_t start,uint32_t tail) {
        constexpr uint32_t stateShift=HALF_RESOURCE?384U:0U;
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
        LocalTensor<float> refined_(TPosition::VECCALC,179072U+stateShift,832U);
        LocalTensor<float> xyz_(TPosition::VECCALC,59136U,3168U);
        constexpr uint32_t pp_=1056;

        const uint32_t group=start/256,slot=(start/64)%4;
        auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
        auto words=predicates[slot*128].template ReinterpretCast<int32_t>();
        Duplicate(words,int32_t(0),128);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
        Duplicate<int32_t,false>(words,int32_t((uint32_t(1)<<tail)-1),uint64_t(64),2,1,8);
        PipeBarrier<PIPE_V>();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
    }

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
    template<bool HALF_RESOURCE,bool FULL_ROWS=false>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixState(uint32_t inputRows,uint32_t segmentCount) {
        const uint32_t liveRows_=FULL_ROWS?64U:inputRows;
        constexpr uint32_t stateShift=HALF_RESOURCE?384U:0U;
        constexpr uint32_t predicateShift=HALF_RESOURCE?256U:0U;
        // Fold the conservative prefix interval into the norm before the projection FMA.
        // Keep W4 immutable so only the winning norm needs an error-scale operation.
        constexpr float intervalFactor=1.621246337890625e-5F;
        constexpr uint32_t lowerPlane=5U;
        constexpr uint32_t errorPlane=4U;
        static_assert(CAPACITY==1024 && COEFFICIENTS==576 && ROWS==64, "Bounded prefix resource required");
        static_assert(!CLEAN_PRIMARY64 && !NORM_BEFORE_DOT && !DEFER_CHUNK_CERTIFICATE, "Full-resource paths are separate");
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
        "Native prefix search requires its matching UB resource");
        LocalTensor<float> certHigh_(TPosition::VECCALC,184256U+stateShift,64U);
        LocalTensor<float> cert_(TPosition::VECCALC,184000U+stateShift,64U);
        LocalTensor<float> chunkLower_(TPosition::VECCALC,174208U+stateShift,1024U);
        LocalTensor<float> hi_(TPosition::VECCALC,183488U+stateShift,64U);
        LocalTensor<int32_t> idx_(TPosition::VECCALC,183744U+stateShift,64U);
        LocalTensor<float> lo_(TPosition::VECCALC,182976U+stateShift,64U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187072U+predicateShift,512U);
        LocalTensor<int32_t> offset_(TPosition::VECCALC,185792U+stateShift,64U);
        LocalTensor<int32_t> oneHot_(TPosition::VECCALC,184512U+stateShift,64U);
        LocalTensor<float> record_(TPosition::VECCALC,182400U+stateShift,128U);
        LocalTensor<float> refined_(TPosition::VECCALC,179072U+stateShift,832U);
        LocalTensor<int32_t> rowBase_(TPosition::VECCALC,186048U+stateShift,64U);
        LocalTensor<float> second_(TPosition::VECCALC,183232U+stateShift,64U);
        LocalTensor<float> segmentIndices_(TPosition::VECCALC,173184U,64U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U+stateShift,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185536U+stateShift,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,184768U+stateShift,64U);
        LocalTensor<uint8_t> valid_(TPosition::VECCALC,187584U+predicateShift,512U);
        LocalTensor<float> xyz_(TPosition::VECCALC,59136U,3168U);
        constexpr uint32_t pp_=1056;
        InitializeGroupedSearchState(lo_,idx_,3);

        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        const uint32_t count_=(liveRows_*64);

        for(uint32_t j=0;j<segmentCount;j+=64) {
            if constexpr(COMPACT_TAIL) {
                if(segmentCount-j<=8) {
                    SavePrefixPredicate<HALF_RESOURCE>(j,segmentCount-j);
                    CompactPrefixTail<HALF_RESOURCE>(j,segmentCount-j);MergePrefixMinimum<HALF_RESOURCE>(j);continue;
                }
            }
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(PrefixW<HALF_RESOURCE>(axis),PrefixP<HALF_RESOURCE>(axis),PrefixC<HALF_RESOURCE>(axis)[j],uint64_t(64),liveRows_,pointSegment);
            if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
            Mul<float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(0),PrefixC<HALF_RESOURCE>(6)[j],uint64_t(64),liveRows_,segment);
            Mul<float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0),uint64_t(64),liveRows_,same);

            if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(1),PrefixC<HALF_RESOURCE>(7)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(1),PrefixW<HALF_RESOURCE>(1),uint64_t(64),liveRows_,same);

            if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(2),PrefixC<HALF_RESOURCE>(8)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),uint64_t(64),liveRows_,same);

            if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
            // APz is dead; its bank phase differs from the dot accumulator.
            constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
            const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
            Muls<float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(4),1.0F-intervalFactor,uint64_t(64),liveRows_,unary);
            Maxs<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),0.0F,uint64_t(64),liveRows_,unary);if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
            Min<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),PrefixC<HALF_RESOURCE>(9)[j],uint64_t(64),liveRows_,segment);if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
            FusedMulAdd<float,false>(PrefixW<HALF_RESOURCE>(3),minusTwo,PrefixW<HALF_RESOURCE>(2),uint64_t(64),liveRows_,BinaryRepeatParams{1,0,1,8,0,8});if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),uint64_t(64),liveRows_,same);if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
            if(j+64>segmentCount) {
                CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(segmentCount-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
                Select(PrefixW<HALF_RESOURCE>(lowerPlane),valid_,PrefixW<HALF_RESOURCE>(lowerPlane),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
            }
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                WholeReduceMin<float,false>(record_,PrefixW<HALF_RESOURCE>(lowerPlane),uint64_t(64),liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            } else WholeReduceMin(record_,PrefixW<HALF_RESOURCE>(lowerPlane),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
            } else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            if(liveRows_==64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Gather(tileHi_,PrefixW<HALF_RESOURCE>(errorPlane),offset_.template ReinterpretCast<uint32_t>(),0U,64);
            } else {
                Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
                Gather(tileHi_,PrefixW<HALF_RESOURCE>(errorPlane),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
            }

            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Muls<float,false>(tileHi_,tileHi_,intervalFactor,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            auto chunkMinimum=tileLo_;
            auto chunkUpper=tileHi_;
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t upperConstantBase=14*(CAPACITY+32)+pointPitch-16;
            const LocalTensor<float> upperScale(TPosition::VECCALC,upperConstantBase*4,8);
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
            PipeBarrier<PIPE_V>();
            Brcb(PrefixW<HALF_RESOURCE>(1),chunkUpper,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            const uint32_t group=j/256,slot=(j/64)%4;
            auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
            if constexpr(HALF_RESOURCE) {
                if(liveRows_==64) {
                    Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),PrefixW<HALF_RESOURCE>(lowerPlane),PrefixW<HALF_RESOURCE>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));
                } else {
                    Compare<float,uint8_t,false>(mask_,PrefixW<HALF_RESOURCE>(lowerPlane),PrefixW<HALF_RESOURCE>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                }
            } else
#endif
            {
                Compare<float,uint8_t,false>(mask_,PrefixW<HALF_RESOURCE>(lowerPlane),PrefixW<HALF_RESOURCE>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
                PipeBarrier<PIPE_V>();
                GatherMask(PrefixW<HALF_RESOURCE>(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                GatherMask(PrefixW<HALF_RESOURCE>(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
            }

            PipeBarrier<PIPE_V>();
            if constexpr(COMPACT_TAIL) {MergePrefixMinimum<HALF_RESOURCE>(j);} else {

                Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
                Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
                Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
                Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
                if constexpr(!HALF_RESOURCE) {
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,PrefixW<HALF_RESOURCE>(0),cert_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,PrefixW<HALF_RESOURCE>(1),certHigh_,1,same);
                }
#else
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,PrefixW<HALF_RESOURCE>(0),cert_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,PrefixW<HALF_RESOURCE>(1),certHigh_,1,same);
#endif

                Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
        }
        // Refinement gathers a full register, including inactive point lanes.
        if(liveRows_<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<liveRows_)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(HALF_RESOURCE) {
            // Layout is [group][four chunks][64 rows][two uint32 words].
            // A compact tail stores a conservative superset of its predicates;
            // extra ambiguous rows retain the complete numerical fallback.
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            ShiftRight(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),idx_,int32_t(8),64);
            ShiftRight(PrefixW<HALF_RESOURCE>(1).template ReinterpretCast<int32_t>(),idx_,int32_t(6),64);PipeBarrier<PIPE_V>();
            Muls(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),int32_t(pp_*4-4*512),64);
            Muls(offset_,PrefixW<HALF_RESOURCE>(1).template ReinterpretCast<int32_t>(),int32_t(512),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),64);
            ShiftRight(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),rowBase_,int32_t(5),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            Adds(offset_,offset_,int32_t(ROWS*8*4),64);PipeBarrier<PIPE_V>();
            Gather(cert_,xyz_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Gather(certHigh_,xyz_,offset_.template ReinterpretCast<uint32_t>(),4U,64);PipeBarrier<PIPE_V>();
        }
#endif
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        const UnaryRepeatParams certUnary{1,1,8,8};
        const BinaryRepeatParams certBinary{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<int32_t,false>(PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
        Gather(PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
        And<uint16_t,false>(PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Sub<int32_t,false>(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<int32_t>(),PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        Cast(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,PrefixW<HALF_RESOURCE>(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();

    }

#else
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixState(uint32_t liveRows_,uint32_t segmentCount) {
        constexpr uint32_t stateShift=HALF_RESOURCE?384U:0U;
        constexpr uint32_t predicateShift=HALF_RESOURCE?256U:0U;
        // Fold the conservative prefix interval into the norm before the projection FMA.
        // Keep W4 immutable so only the winning norm needs an error-scale operation.
        constexpr float intervalFactor=1.621246337890625e-5F;
        constexpr uint32_t lowerPlane=5U;
        constexpr uint32_t errorPlane=4U;
        static_assert(CAPACITY==1024 && COEFFICIENTS==576 && ROWS==64, "Bounded prefix resource required");
        static_assert(!CLEAN_PRIMARY64 && !NORM_BEFORE_DOT && !DEFER_CHUNK_CERTIFICATE, "Full-resource paths are separate");
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
        "Native prefix search requires its matching UB resource");
        LocalTensor<float> certHigh_(TPosition::VECCALC,184256U+stateShift,64U);
        LocalTensor<float> cert_(TPosition::VECCALC,184000U+stateShift,64U);
        LocalTensor<float> chunkLower_(TPosition::VECCALC,174208U+stateShift,1024U);
        LocalTensor<float> hi_(TPosition::VECCALC,183488U+stateShift,64U);
        LocalTensor<int32_t> idx_(TPosition::VECCALC,183744U+stateShift,64U);
        LocalTensor<float> lo_(TPosition::VECCALC,182976U+stateShift,64U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187072U+predicateShift,512U);
        LocalTensor<int32_t> offset_(TPosition::VECCALC,185792U+stateShift,64U);
        LocalTensor<int32_t> oneHot_(TPosition::VECCALC,184512U+stateShift,64U);
        LocalTensor<float> record_(TPosition::VECCALC,182400U+stateShift,128U);
        LocalTensor<float> refined_(TPosition::VECCALC,179072U+stateShift,832U);
        LocalTensor<int32_t> rowBase_(TPosition::VECCALC,186048U+stateShift,64U);
        LocalTensor<float> second_(TPosition::VECCALC,183232U+stateShift,64U);
        LocalTensor<float> segmentIndices_(TPosition::VECCALC,173184U,64U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U+stateShift,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185536U+stateShift,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,184768U+stateShift,64U);
        LocalTensor<uint8_t> valid_(TPosition::VECCALC,187584U+predicateShift,512U);
        LocalTensor<float> xyz_(TPosition::VECCALC,59136U,3168U);
        constexpr uint32_t pp_=1056;
        InitializeGroupedSearchState(lo_,idx_,3);

        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        const uint32_t count_=(liveRows_*64);
        const uint64_t rowMask[2]={liveRows_==64?~uint64_t(0):((uint64_t(1)<<liveRows_)-1),uint64_t(0)};

        for(uint32_t j=0;j<segmentCount;j+=64) {
            if constexpr(COMPACT_TAIL) {
                if(segmentCount-j<=8) {
                    SavePrefixPredicate<HALF_RESOURCE>(j,segmentCount-j);
                    CompactPrefixTail<HALF_RESOURCE>(j,segmentCount-j);MergePrefixMinimum<HALF_RESOURCE>(j);continue;
                }
            }
            // Initialization and the previous merge leave the 64-lane normal mask.
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(PrefixW<HALF_RESOURCE>(axis),PrefixP<HALF_RESOURCE>(axis),PrefixC<HALF_RESOURCE>(axis)[j],uint64_t(64),liveRows_,pointSegment);
            if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
            Mul<float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(0),PrefixC<HALF_RESOURCE>(6)[j],uint64_t(64),liveRows_,segment);
            Mul<float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0),uint64_t(64),liveRows_,same);

            if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(1),PrefixC<HALF_RESOURCE>(7)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(1),PrefixW<HALF_RESOURCE>(1),uint64_t(64),liveRows_,same);

            if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(2),PrefixC<HALF_RESOURCE>(8)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),uint64_t(64),liveRows_,same);

            if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
            // APz is dead; its bank phase differs from the dot accumulator.
            constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
            const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
            Muls<float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(4),1.0F-intervalFactor,uint64_t(64),liveRows_,unary);
            Maxs<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),0.0F,uint64_t(64),liveRows_,unary);if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
            Min<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),PrefixC<HALF_RESOURCE>(9)[j],uint64_t(64),liveRows_,segment);if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
            FusedMulAdd<float,false>(PrefixW<HALF_RESOURCE>(3),minusTwo,PrefixW<HALF_RESOURCE>(2),uint64_t(64),liveRows_,BinaryRepeatParams{1,0,1,8,0,8});if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),uint64_t(64),liveRows_,same);if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
            if(j+64>segmentCount) {
                CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(segmentCount-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
                Select(PrefixW<HALF_RESOURCE>(lowerPlane),valid_,PrefixW<HALF_RESOURCE>(lowerPlane),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
                // The count-based tail select restores a full mask; narrow it here.
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
                WholeReduceMin<float,false>(record_,PrefixW<HALF_RESOURCE>(lowerPlane),uint64_t(64),liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            } else WholeReduceMin(record_,PrefixW<HALF_RESOURCE>(lowerPlane),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
                // WholeReduceMin and the fixed-pattern GatherMask preserve the mask.
                Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
            } else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            // Only live rows have valid reduction indices; preserve their bit mask
            // throughout offset construction without switching to counter mode.
            SetVectorMask<int32_t,MaskMode::NORMAL>(rowMask[1],rowMask[0]);
            Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Gather(tileHi_,PrefixW<HALF_RESOURCE>(errorPlane),offset_.template ReinterpretCast<uint32_t>(),0U,rowMask,uint8_t(1),uint16_t(8));

            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Muls<float,false>(tileHi_,tileHi_,intervalFactor,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            auto chunkMinimum=tileLo_;
            auto chunkUpper=tileHi_;
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t upperConstantBase=14*(CAPACITY+32)+pointPitch-16;
            const LocalTensor<float> upperScale(TPosition::VECCALC,upperConstantBase*4,8);
            FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
            PipeBarrier<PIPE_V>();
            Brcb(PrefixW<HALF_RESOURCE>(1),chunkUpper,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,uint8_t,false>(mask_,PrefixW<HALF_RESOURCE>(lowerPlane),PrefixW<HALF_RESOURCE>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
            const uint32_t group=j/256,slot=(j/64)%4;
            auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)
            Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
            else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
            PipeBarrier<PIPE_V>();
            GatherMask(PrefixW<HALF_RESOURCE>(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(PrefixW<HALF_RESOURCE>(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));

            PipeBarrier<PIPE_V>();
            if constexpr(COMPACT_TAIL) {MergePrefixMinimum<HALF_RESOURCE,true>(j);} else {

                Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
                Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
                Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
                Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,PrefixW<HALF_RESOURCE>(0),cert_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,PrefixW<HALF_RESOURCE>(1),certHigh_,1,same);

                Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
        }
        // Refinement gathers a full register, including inactive point lanes.
        if(liveRows_<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<liveRows_)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();

        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        const UnaryRepeatParams certUnary{1,1,8,8};
        const BinaryRepeatParams certBinary{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<int32_t,false>(PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
        Gather(PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
        And<uint16_t,false>(PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Sub<int32_t,false>(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<int32_t>(),PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        Cast(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,PrefixW<HALF_RESOURCE>(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();

    }

#endif

    __aicore__ __attribute__((noinline,aligned(128))) static uint32_t SearchPrefixOutlined(uint32_t liveRows_,uint32_t segmentCount) {
        AscendC::ICachePreLoad(int64_t(2));
        SearchPrefixState<false>(liveRows_,segmentCount);
        return liveRows_*64;
    }

    __aicore__ __attribute__((noinline)) static uint32_t SearchHalfPrefixOutlined(uint32_t liveRows,uint32_t segmentCount) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if(liveRows==64U)SearchPrefixState<true,true>(64U,segmentCount);
        else SearchPrefixState<true>(liveRows,segmentCount);
#else
        SearchPrefixState<true>(liveRows,segmentCount);
#endif
        return liveRows*64;
    }

    __aicore__ inline void Search() {
        constexpr bool fixedSearchMask=FULL_FIXED_MASK || NATIVE_FLOAT_PREFIX_SPARSE;
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            if constexpr(COEFFICIENTS==192 && SINGLE_TAIL) {
                if(liveRows_==16U)SearchStaticPolicy<true,16>();
                else SearchStaticPolicy<true>();
            } else SearchStaticPolicy<true>();
        } else {
        if constexpr((sizeof(T)==4 && (CAPACITY==1024 || CAPACITY==256)) || fixedSearchMask) {
            InitializeGroupedSearchState(lo_,idx_,DEFER_CHUNK_CERTIFICATE?1:3);
        } else {
            Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            if constexpr(!DEFER_CHUNK_CERTIFICATE){Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);}
        }
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)Duplicate(refined_[520],-2.0F,8);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(liveRows_*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        for(uint32_t j=0;j<t_.segments;j+=64) {
            if constexpr(COMPACT_TAIL) {
                if(t_.segments-j<=8) {
                    if constexpr(SAVE_PREFIX_CHUNK_PREDICATES)SavePrefixTailPredicate(j,t_.segments-j);
                    CompactSearchTail<false>(j,t_.segments-j);MergeSearchMinimum<false>(j);continue;
                }
            }
            if constexpr(!CLEAN_PRIMARY64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(SearchW<false>(axis),SearchP<false>(axis),SearchC<false>(axis)[j],uint64_t(64),liveRows_,pointSegment);
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            Mul<float,false>(SearchW<false>(4),SearchW<false>(0),SearchW<false>(0),uint64_t(64),liveRows_,same);
            Mul<float,false>(SearchW<false>(3),SearchW<false>(0),SearchC<false>(6)[j],uint64_t(64),liveRows_,segment);
            } else {
            Mul<float,false>(SearchW<false>(3),SearchW<false>(0),SearchC<false>(6)[j],uint64_t(64),liveRows_,segment);
            Mul<float,false>(SearchW<false>(4),SearchW<false>(0),SearchW<false>(0),uint64_t(64),liveRows_,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(1),SearchW<false>(1),uint64_t(64),liveRows_,same);
            MulAddDst<float,float,false>(SearchW<false>(3),SearchW<false>(1),SearchC<false>(7)[j],uint64_t(64),liveRows_,segment);
            } else {
            MulAddDst<float,float,false>(SearchW<false>(3),SearchW<false>(1),SearchC<false>(7)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(1),SearchW<false>(1),uint64_t(64),liveRows_,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(2),SearchW<false>(2),uint64_t(64),liveRows_,same);
            MulAddDst<float,float,false>(SearchW<false>(3),SearchW<false>(2),SearchC<false>(8)[j],uint64_t(64),liveRows_,segment);
            } else {
            MulAddDst<float,float,false>(SearchW<false>(3),SearchW<false>(2),SearchC<false>(8)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(2),SearchW<false>(2),uint64_t(64),liveRows_,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(CLEAN_PRIMARY64) {
                // APz is dead here; keep the clamp in a bank phase distinct from dot.
                if constexpr(sizeof(T)==2)
                    Muls<float,false>(SearchW<false>(5),SearchW<false>(4),1.52587890625e-5F,uint64_t(64),liveRows_,unary);
                Maxs<float,false>(SearchW<false>(2),SearchW<false>(3),0.0F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
                Min<float,false>(SearchW<false>(2),SearchW<false>(2),SearchC<false>(9)[j],uint64_t(64),liveRows_,segment);PipeBarrier<PIPE_V>();
                const BinaryRepeatParams fused{1,0,1,8,0,8};
                if constexpr(sizeof(T)==4)
                    FusedMulAdd<float,false>(SearchW<false>(3),refined_[520],SearchW<false>(2),uint64_t(64),liveRows_,fused);
                else {
                    constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                    const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                    FusedMulAdd<float,false>(SearchW<false>(3),minusTwo,SearchW<false>(2),uint64_t(64),liveRows_,fused);
                }
                if constexpr(sizeof(T)==4)
                    Muls<float,false>(SearchW<false>(5),SearchW<false>(4),1.52587890625e-5F,uint64_t(64),liveRows_,unary);
                PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(2),SearchW<false>(3),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
            } else {
            Maxs<float,false>(SearchW<false>(0),SearchW<false>(3),0.0F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(SearchW<false>(0),SearchW<false>(0),SearchC<false>(9)[j],uint64_t(64),liveRows_,segment);PipeBarrier<PIPE_V>();
            Muls<float,false>(SearchW<false>(2),SearchW<false>(3),-2.0F,uint64_t(64),liveRows_,unary);
            Muls<float,false>(SearchW<false>(5),SearchW<false>(4),1.52587890625e-5F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
            Add<float,false>(SearchW<false>(1),SearchW<false>(0),SearchW<false>(2),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(0),SearchW<false>(1),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
            }
            Sub<float,false>(SearchW<false>(0),SearchW<false>(4),SearchW<false>(5),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
            if(j+64>t_.segments) {
                CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(t_.segments-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
                Select(SearchW<false>(0),valid_,SearchW<false>(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
            }
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                WholeReduceMin<float,false>(record_,SearchW<false>(0),uint64_t(64),liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            } else WholeReduceMin(record_,SearchW<false>(0),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
            } else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            if constexpr(fixedSearchMask) {
                if(liveRows_==64) {
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                    Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,SearchW<false>(5),offset_.template ReinterpretCast<uint32_t>(),0U,64);
                } else {
                    Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
                    Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,SearchW<false>(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
                }
            } else {
                Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
                Gather(tileHi_,SearchW<false>(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
            }
            PipeBarrier<PIPE_V>();
            auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
            auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Axpy<float,float,false>(tileLo_,tileHi_,2.125F,uint64_t(64),1,unary);
                PipeBarrier<PIPE_V>();
            } else if constexpr(CLEAN_PRIMARY64) {
                constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
                constexpr uint32_t constantBase=14*(CAPACITY+32)+pointPitch-16;
                const LocalTensor<float> upperScale(TPosition::VECCALC,constantBase*4,8);
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
                PipeBarrier<PIPE_V>();
            } else if constexpr(NATIVE_HALF_PREFIX_PRIMARY) {
                constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
                constexpr uint32_t constantBase=14*(CAPACITY+32)+pointPitch-16;
                const LocalTensor<float> upperScale(TPosition::VECCALC,constantBase*4,8);
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
                PipeBarrier<PIPE_V>();
            } else if constexpr(fixedSearchMask) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<float,false>(tileHi_,tileHi_,2.125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<float,false>(tileHi_,tileLo_,tileHi_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            } else {
                Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
                Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
            }
            Brcb(SearchW<false>(1),chunkUpper,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,uint8_t,false>(mask_,SearchW<false>(0),SearchW<false>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
            if constexpr(SAVE_PREFIX_CHUNK_PREDICATES) {
                const uint32_t group=j/256,slot=(j/64)%4;
                auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
                PipeBarrier<PIPE_V>();
            }
            if constexpr(DEFER_CHUNK_CERTIFICATE) {
                const uint32_t group=j/256,slot=(j/64)%4;
                auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                if constexpr(fixedSearchMask)
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
            } else {
                GatherMask(SearchW<false>(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                GatherMask(SearchW<false>(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
            }
            PipeBarrier<PIPE_V>();
            if constexpr(COMPACT_TAIL) {MergeSearchMinimum<false>(j);} else {
            if constexpr(!(fixedSearchMask)) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
            Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
            if constexpr(!DEFER_CHUNK_CERTIFICATE) {
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,SearchW<false>(0),cert_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,SearchW<false>(1),certHigh_,1,same);
            }
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
        }
        // Refinement gathers a full register, including inactive point lanes.
        if(liveRows_<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<liveRows_)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();
        if constexpr(DEFER_CHUNK_CERTIFICATE) {
            // Resolve only the winning chunk's two predicate words per point.
            // Indices are byte offsets into the existing usable UB arena.
            const LocalTensor<float> arena(TPosition::VECCALC,0,188416/4);
            if constexpr(fixedSearchMask) {
                const UnaryRepeatParams resolveUnary{1,1,8,8};
                const BinaryRepeatParams resolveBinary{1,1,1,8,8,8};
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
                ShiftRight<uint32_t,false>(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Muls<int32_t,false>(offset_,offset_,int32_t(32),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(SearchW<false>(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Duplicate<int32_t,false>(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(192),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
                SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
                And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<false>(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,resolveBinary);PipeBarrier<PIPE_V>();
                SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,offset_,int32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,SearchW<false>(2).template ReinterpretCast<int32_t>(),uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,outputOffset_,uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Adds<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            } else {
            ShiftRight(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),64);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(32),64);PipeBarrier<PIPE_V>();
            Gather(SearchW<false>(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Duplicate(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(192),64);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<false>(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(8),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,SearchW<false>(2).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,outputOffset_,64);PipeBarrier<PIPE_V>();
            Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Adds(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
            Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            }
        }
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        if constexpr(CAPACITY<1024) {
            // A unique bit across both words requires a nonzero one-bit union
            // and no bit occupied in both words.
            Duplicate(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Or(SearchW<false>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
            And(SearchW<false>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Sub(SearchW<false>(0).template ReinterpretCast<int32_t>(),SearchW<false>(2).template ReinterpretCast<int32_t>(),SearchW<false>(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            And(SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(2).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Or(SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(3).template ReinterpretCast<uint16_t>(),128);
            Cast(SearchW<false>(1),SearchW<false>(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Cast(SearchW<false>(0),SearchW<false>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,SearchW<false>(0),0.0F,CMPMODE::EQ,64);
            CompareScalar(smallMask_,SearchW<false>(1),0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        } else {
        if constexpr(fixedSearchMask) {
            const UnaryRepeatParams certUnary{1,1,8,8};
            const BinaryRepeatParams certBinary{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Duplicate<int32_t,false>(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<false>(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
            Gather(SearchW<false>(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(SearchW<false>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
            And<uint16_t,false>(SearchW<false>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Sub<int32_t,false>(SearchW<false>(0).template ReinterpretCast<int32_t>(),SearchW<false>(2).template ReinterpretCast<int32_t>(),SearchW<false>(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else {
        Duplicate(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(31),64);PipeBarrier<PIPE_V>();
        And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<false>(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
        Gather(SearchW<false>(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        Or(SearchW<false>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
        And(SearchW<false>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Sub(SearchW<false>(0).template ReinterpretCast<int32_t>(),SearchW<false>(2).template ReinterpretCast<int32_t>(),SearchW<false>(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
        Or(SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        }
        Cast(SearchW<false>(0),SearchW<false>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,SearchW<false>(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        }
        }
    }

    template<uint32_t extent>
    __aicore__ __attribute__((always_inline)) inline bool ReduceCoordinateValidity() {
        Duplicate(W(0),0.0F,extent);PipeBarrier<PIPE_V>();
        Select(W(0),valid_,W(0),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,extent);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,W(0),64,extent/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,record_,extent/64,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        CompareScalar(smallMask_,record_,0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return (smallMask_.GetValue(0)&1U)!=0;
    }
    // Coordinate storage changes the admissible range, not certificate reduction.
    template<bool COEFFICIENT_INPUT>
    __aicore__ __attribute__((always_inline)) inline bool CoarseCoordinatesSafe() {
        constexpr uint32_t extent=COEFFICIENT_INPUT?COEFFICIENTS:192;
        constexpr uint32_t words=extent/16;
        auto temporaryMask=record_.template ReinterpretCast<uint8_t>();
        Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),words);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<(COEFFICIENT_INPUT?6U:1U);++axis) {
            auto x=COEFFICIENT_INPUT?C(axis):compactP_;
            Abs(W(1),x,extent);PipeBarrier<PIPE_V>();
            if constexpr(sizeof(T)==2) {
            // Finite Half coordinates have spacing at least 2^-24, including subnormals.
            CompareScalar(mask_,W(1),65504.0F,CMPMODE::LE,extent);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
            } else {
            CompareScalar(mask_,W(1),1099511627776.0F,CMPMODE::LE,extent);
            CompareScalar(temporaryMask,W(1),9.5367431640625e-7F,CMPMODE::GE,extent);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);
            CompareScalar(mask_,W(1),0.0F,CMPMODE::EQ,extent);PipeBarrier<PIPE_V>();
            Or(temporaryMask.template ReinterpretCast<uint16_t>(),temporaryMask.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),temporaryMask.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(COEFFICIENT_INPUT) {
            CompareScalar(mask_,C(13),8.673617379884035e-19F,CMPMODE::GE,extent);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
        }
        if constexpr(!COEFFICIENT_INPUT) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            auto packed=valid_.template ReinterpretCast<uint64_t>();
            return (packed.GetValue(0)&packed.GetValue(1)&packed.GetValue(2))==~uint64_t(0);
        } else {
        return ReduceCoordinateValidity<extent>();
        }
    }
    __aicore__ __attribute__((always_inline)) inline bool CoarseDirectionSafe() {
        constexpr uint32_t extent=COEFFICIENTS,words=extent/16;
        Mul(W(0),C(6),C(6),extent);PipeBarrier<PIPE_V>();
        MulAddDst(W(0),C(7),C(7),extent);PipeBarrier<PIPE_V>();
        MulAddDst(W(0),C(8),C(8),extent);PipeBarrier<PIPE_V>();
        MakeIndices(W(2).template ReinterpretCast<int32_t>(),0,extent);PipeBarrier<PIPE_V>();
        // C220 ordered comparison uses exactly represented FP32 coefficient indices.
        Cast(W(2),W(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,extent);PipeBarrier<PIPE_V>();
        CompareScalar(mask_,W(2),float(int32_t(t_.segments)),CMPMODE::LT,extent);PipeBarrier<PIPE_V>();
        // Padded coefficients do not represent legal candidate segments.
        Select(W(0),mask_,W(0),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,extent);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,W(0),0.999999523162841796875F,CMPMODE::GE,extent);
        CompareScalar(mask_,W(0),1.000000476837158203125F,CMPMODE::LE,extent);PipeBarrier<PIPE_V>();
        And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);
        Mul(W(1),C(9),C(9),extent);PipeBarrier<PIPE_V>();
        Sub(W(1),W(1),C(13),extent);PipeBarrier<PIPE_V>();
        Abs(W(1),W(1),extent);
        Muls(W(2),C(13),2.384185791015625e-7F,extent);PipeBarrier<PIPE_V>();
        Compare(mask_,W(1),W(2),CMPMODE::LE,extent);PipeBarrier<PIPE_V>();
        And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
        return ReduceCoordinateValidity<extent>();
    }
    // Shared certificate resolution, fully inlined into each original search.
    __aicore__ __attribute__((always_inline)) inline void FinalizeGuardedCertificate(uint32_t activeRows) {
        // Refinement gathers a full register, including inactive point lanes.
        if(activeRows<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<activeRows)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();
        if constexpr(DEFER_CHUNK_CERTIFICATE) {
            // Resolve only the winning chunk's two predicate words per point.
            // Indices are byte offsets into the existing usable UB arena.
            const LocalTensor<float> arena(TPosition::VECCALC,0,188416/4);
            if constexpr(FULL_FIXED_MASK) {
                const UnaryRepeatParams resolveUnary{1,1,8,8};
                const BinaryRepeatParams resolveBinary{1,1,1,8,8,8};
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
                ShiftRight<uint32_t,false>(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Muls<int32_t,false>(offset_,offset_,int32_t(32),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(W(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Duplicate<int32_t,false>(W(4).template ReinterpretCast<int32_t>(),int32_t(192),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
                SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
                And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),W(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,resolveBinary);PipeBarrier<PIPE_V>();
                SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,offset_,int32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,W(2).template ReinterpretCast<int32_t>(),uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,outputOffset_,uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Adds<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            } else {
            ShiftRight(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),64);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(32),64);PipeBarrier<PIPE_V>();
            Gather(W(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Duplicate(W(4).template ReinterpretCast<int32_t>(),int32_t(192),64);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),W(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(8),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,W(2).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,outputOffset_,64);PipeBarrier<PIPE_V>();
            Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Adds(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
            Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            }
        }
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        if constexpr(CAPACITY<1024) {
            // A unique bit across both words requires a nonzero one-bit union
            // and no bit occupied in both words.
            Duplicate(W(4).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Or(W(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
            And(W(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Sub(W(0).template ReinterpretCast<int32_t>(),W(2).template ReinterpretCast<int32_t>(),W(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            And(W(0).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(2).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Or(W(0).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(3).template ReinterpretCast<uint16_t>(),128);
            Cast(W(1),W(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Cast(W(0),W(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(0),0.0F,CMPMODE::EQ,64);
            CompareScalar(smallMask_,W(1),0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        } else {
        if constexpr(FULL_FIXED_MASK) {
            const UnaryRepeatParams certUnary{1,1,8,8};
            const BinaryRepeatParams certBinary{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Duplicate<int32_t,false>(W(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),W(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
            Gather(W(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(W(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
            And<uint16_t,false>(W(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Sub<int32_t,false>(W(0).template ReinterpretCast<int32_t>(),W(2).template ReinterpretCast<int32_t>(),W(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(W(0).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else {
        Duplicate(W(4).template ReinterpretCast<int32_t>(),int32_t(31),64);PipeBarrier<PIPE_V>();
        And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),W(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
        Gather(W(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        Or(W(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
        And(W(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Sub(W(0).template ReinterpretCast<int32_t>(),W(2).template ReinterpretCast<int32_t>(),W(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
        Or(W(0).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        }
        Cast(W(0),W(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,W(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        }
    }
    // Keep the cross-chunk certificate packed until the caller consumes it.
    // Both callers share this exact mask and barrier sequence.
    __aicore__ __attribute__((always_inline)) inline void BuildCrossChunkCertificate(
        uint32_t chunks,const BinaryRepeatParams& same,const UnaryRepeatParams& unary) {
        if(chunks<16) {
            Duplicate(chunkLower_[chunks*64],__builtin_inff(),(16-chunks)*64);
            PipeBarrier<PIPE_V>();
        }
        // Multi-repeat Compare packs sixteen64-bit results into128 bytes.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Compare<float,uint8_t,false>(mask_,chunkLower_,hi_,CMPMODE::LE,
            uint64_t(64),16,BinaryRepeatParams{1,1,1,8,8,0});
        PipeBarrier<PIPE_V>();
        auto sourceBits=mask_.template ReinterpretCast<uint16_t>();
        auto unions=W(0).template ReinterpretCast<uint16_t>();
        auto collisions=W(1).template ReinterpretCast<uint16_t>();
        auto nextUnions=W(2).template ReinterpretCast<uint16_t>();
        auto nextCollisions=W(3).template ReinterpretCast<uint16_t>();
        auto mergedCollisions=W(4).template ReinterpretCast<uint16_t>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(32));
        Or<uint16_t,false>(unions,sourceBits,sourceBits[32],uint64_t(128),1,same);
        And<uint16_t,false>(collisions,sourceBits,sourceBits[32],uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
        Or<uint16_t,false>(nextUnions,unions,unions[16],uint64_t(128),1,same);
        And<uint16_t,false>(nextCollisions,unions,unions[16],uint64_t(128),1,same);
        Or<uint16_t,false>(mergedCollisions,collisions,collisions[16],uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(nextCollisions,nextCollisions,mergedCollisions,uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        auto oldUnions=unions;unions=nextUnions;nextUnions=oldUnions;
        auto oldCollisions=collisions;collisions=nextCollisions;nextCollisions=oldCollisions;
        #pragma unroll
        for(uint32_t stage=0;stage<2;++stage) {
            auto rotation=C(3)[CAPACITY+16+stage*8].template ReinterpretCast<uint32_t>();
            Gather(nextUnions.template ReinterpretCast<float>(),unions.template ReinterpretCast<float>(),rotation,0U,8);
            Gather(nextCollisions.template ReinterpretCast<float>(),collisions.template ReinterpretCast<float>(),rotation,0U,8);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
            And<uint16_t,false>(mergedCollisions,unions,nextUnions,uint64_t(128),1,same);
            Or<uint16_t,false>(nextUnions,unions,nextUnions,uint64_t(128),1,same);
            Or<uint16_t,false>(nextCollisions,collisions,nextCollisions,uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            Or<uint16_t,false>(nextCollisions,nextCollisions,mergedCollisions,uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            auto oldU=unions;unions=nextUnions;nextUnions=oldU;
            auto oldC=collisions;collisions=nextCollisions;nextCollisions=oldC;
        }
        Not<uint16_t,false>(second_.template ReinterpretCast<uint16_t>(),collisions,uint64_t(128),1,unary);
        PipeBarrier<PIPE_V>();
        And<uint16_t,false>(second_.template ReinterpretCast<uint16_t>(),
            second_.template ReinterpretCast<uint16_t>(),unions,uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        // Keep the cross-chunk certificate as a packed bitset until Process.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    }
    __aicore__ inline void SearchGuardedHalfCachedRows() {
        const uint32_t guardedRows=liveRows_;
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            SearchStaticPolicy<false>();
        } else {
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==2 && ROWS==64;
        const float coarseErrorFactor=(tightCoefficientSafe_ && CoarseCoordinatesSafe<false>())?
            7.62939453125e-6F:1.52587890625e-5F;
        if constexpr((sizeof(T)==4 && (CAPACITY==1024 || CAPACITY==256)) || FULL_FIXED_MASK) {
            InitializeGroupedSearchState(lo_,idx_,DEFER_CHUNK_CERTIFICATE?1:3);
        } else {
            Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            if constexpr(!DEFER_CHUNK_CERTIFICATE){Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);}
        }
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)Duplicate(refined_[520],-2.0F,8);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(guardedRows*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        for(uint32_t j=0;j<t_.segments;j+=64) {
            if constexpr(COMPACT_TAIL) {
                if(t_.segments-j<=8) {
                    CompactSearchTail<false>(j,t_.segments-j);MergeSearchMinimum<false>(j);continue;
                }
            }
            if constexpr(!CLEAN_PRIMARY64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),P(axis),C(axis)[j],uint64_t(64),guardedRows,pointSegment);
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),guardedRows,same);
            Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),guardedRows,segment);
            } else {
            Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),guardedRows,segment);
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),guardedRows,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),guardedRows,same);
            MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),guardedRows,segment);
            } else {
            MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),guardedRows,segment);
            MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),guardedRows,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),guardedRows,same);
            MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),guardedRows,segment);
            } else {
            MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),guardedRows,segment);
            MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),guardedRows,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(CLEAN_PRIMARY64) {
                // APz is dead here; keep the clamp in a bank phase distinct from dot.
                if constexpr(sizeof(T)==2)
                    Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);
                Maxs<float,false>(W(2),W(3),0.0F,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
                Min<float,false>(W(2),W(2),C(9)[j],uint64_t(64),guardedRows,segment);PipeBarrier<PIPE_V>();
                const BinaryRepeatParams fused{1,0,1,8,0,8};
                if constexpr(sizeof(T)==4)
                    FusedMulAdd<float,false>(W(3),refined_[520],W(2),uint64_t(64),guardedRows,fused);
                else {
                    constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                    const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                    FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),guardedRows,fused);
                }
                if constexpr(sizeof(T)==4)
                    Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);
                PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(W(4),W(2),W(3),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
            } else {
            Maxs<float,false>(W(0),W(3),0.0F,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(W(0),W(0),C(9)[j],uint64_t(64),guardedRows,segment);PipeBarrier<PIPE_V>();
            Muls<float,false>(W(2),W(3),-2.0F,uint64_t(64),guardedRows,unary);
            Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
            Add<float,false>(W(1),W(0),W(2),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(4),W(0),W(1),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
            }
            Sub<float,false>(W(0),W(4),W(5),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
            if(j+64>t_.segments) {
                CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(t_.segments-j)),CMPMODE::LT,uint64_t(64),guardedRows,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
                Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(guardedRows*64));PipeBarrier<PIPE_V>();
            }
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
                if(j+64>t_.segments) {
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                }
                WholeReduceMin<float,false>(record_,W(0),uint64_t(64),guardedRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            } else WholeReduceMin(record_,W(0),64,guardedRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)
                Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
            else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            if constexpr(FULL_FIXED_MASK) {
                if(guardedRows==64) {
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                    Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,64);
                } else {
                    Muls(offset_,tileIdx_,int32_t(4),guardedRows);PipeBarrier<PIPE_V>();
                    Add(offset_,offset_,rowBase_,guardedRows);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,guardedRows);
                }
            } else {
                Muls(offset_,tileIdx_,int32_t(4),guardedRows);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,guardedRows);PipeBarrier<PIPE_V>();
                Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,guardedRows);
            }
            PipeBarrier<PIPE_V>();
            auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
            auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Axpy<float,float,false>(tileLo_,tileHi_,2.125F,uint64_t(64),1,unary);
                PipeBarrier<PIPE_V>();
            } else if constexpr(CLEAN_PRIMARY64) {
                constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
                constexpr uint32_t constantBase=14*(CAPACITY+32)+pointPitch-16;
                const LocalTensor<float> upperScale(TPosition::VECCALC,constantBase*4,8);
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
                PipeBarrier<PIPE_V>();
            } else if constexpr(FULL_FIXED_MASK) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<float,false>(tileHi_,tileHi_,2.125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<float,false>(tileHi_,tileLo_,tileHi_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            } else {
                Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
                Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
            }
            Brcb(W(1),chunkUpper,(guardedRows+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            if constexpr(CLEAN_PRIMARY64) {
                const uint32_t group=j/256,slot=(j/64)%4;
                auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                if(guardedRows==64) {
                    Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),W(0),W(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));
                } else {
                    Compare<float,uint8_t,false>(mask_,W(0),W(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                }
            } else {
                Compare<float,uint8_t,false>(mask_,W(0),W(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                if constexpr(DEFER_CHUNK_CERTIFICATE) {
                    const uint32_t group=j/256,slot=(j/64)%4;
                    auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                    if constexpr(FULL_FIXED_MASK)
                        Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                    else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
                } else {
                    GatherMask(W(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                    GatherMask(W(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
                }
            }
            PipeBarrier<PIPE_V>();
            if constexpr(COMPACT_TAIL) {MergeSearchMinimum<false>(j);} else {
            if constexpr(!(FULL_FIXED_MASK)) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
            if constexpr(!DEFER_SECOND)
                Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            if constexpr(!DEFER_SECOND)
                Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
            if constexpr(!DEFER_CHUNK_CERTIFICATE) {
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,W(0),cert_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,W(1),certHigh_,1,same);
            }
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(DEFER_SECOND) {
            const uint32_t chunks=(t_.segments+63)/64;
            BuildCrossChunkCertificate(chunks,same,unary);
        }
        FinalizeGuardedCertificate(guardedRows);
        }
    }
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> HalfFullC(uint32_t k) {
        constexpr uint32_t pitch=1056U;
        const uint32_t offset=k>=6U && k<=8U?6U*pitch+(k-6U)*1024U:k*pitch;
        return LocalTensor<float>(TPosition::VECCALC,offset*4U,14U*pitch-offset);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> HalfFullW(uint32_t k) {
        constexpr uint32_t banks[13]={32,0,0,64,96,32,96,0,0,0,0,0,0};
        const uint32_t offset=k*4224U+banks[k];
        return LocalTensor<float>(TPosition::VECCALC,71808U+offset*4U,6U*4224U-offset);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> HalfFullP(uint32_t k) {
        return LocalTensor<float>(TPosition::VECCALC,59136U+k*4224U,3U*1056U-k*1056U);
    }
    __aicore__ __attribute__((noinline)) static uint32_t SearchGuardedHalfTypeCertifiedRows(uint32_t liveRows_,uint32_t segmentCount,bool tightCoefficientSafe_) {
        static_assert(sizeof(T)==2 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Full native Half resource required");
        constexpr uint32_t pp_=1056U;
        const uint32_t guardedRows=liveRows_;
        LocalTensor<float> certHigh_(TPosition::VECCALC,184640U,64U);
        LocalTensor<float> cert_(TPosition::VECCALC,184384U,64U);
        LocalTensor<float> chunkLower_(TPosition::VECCALC,174592U,1024U);
        LocalTensor<float> hi_(TPosition::VECCALC,183872U,64U);
        LocalTensor<int32_t> idx_(TPosition::VECCALC,184128U,64U);
        LocalTensor<float> lo_(TPosition::VECCALC,183360U,64U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187328U,512U);
        LocalTensor<int32_t> offset_(TPosition::VECCALC,186176U,64U);
        LocalTensor<int32_t> oneHot_(TPosition::VECCALC,184896U,64U);
        LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,186944U,64U);
        LocalTensor<float> record_(TPosition::VECCALC,182784U,128U);
        LocalTensor<float> refined_(TPosition::VECCALC,179456U,832U);
        LocalTensor<int32_t> rowBase_(TPosition::VECCALC,186432U,64U);
        LocalTensor<float> second_(TPosition::VECCALC,183616U,64U);
        LocalTensor<float> segmentIndices_(TPosition::VECCALC,173184U,64U);
        LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,188352U,32U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185664U,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185920U,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,185152U,64U);
        LocalTensor<uint8_t> valid_(TPosition::VECCALC,187840U,512U);
        LocalTensor<float> xyz_(TPosition::VECCALC,59136U,3168U);
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==2 && ROWS==64;
        const float coarseErrorFactor=tightCoefficientSafe_?
            7.62939453125e-6F:1.52587890625e-5F;
        const float intervalFactor=coarseErrorFactor*1.0625F;
        const float lowerNormFactor=1.0F-intervalFactor;
        const float upperNormFactor=intervalFactor*2.125F*1.000003814697265625F;
        if constexpr((sizeof(T)==4 && (CAPACITY==1024 || CAPACITY==256)) || FULL_FIXED_MASK) {
            InitializeGroupedSearchState(lo_,idx_,DEFER_CHUNK_CERTIFICATE?1:3);
        } else {
            Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            if constexpr(!DEFER_CHUNK_CERTIFICATE){Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);}
        }
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)Duplicate(refined_[520],-2.0F,8);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        const uint32_t count_=(guardedRows*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        for(uint32_t j=0;j<segmentCount;j+=64) {
            if constexpr(!CLEAN_PRIMARY64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(HalfFullW(axis),HalfFullP(axis),HalfFullC(axis)[j],uint64_t(64),guardedRows,pointSegment);
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            Mul<float,false>(HalfFullW(4),HalfFullW(0),HalfFullW(0),uint64_t(64),guardedRows,same);
            Mul<float,false>(HalfFullW(3),HalfFullW(0),HalfFullC(6)[j],uint64_t(64),guardedRows,segment);
            } else {
            Mul<float,false>(HalfFullW(3),HalfFullW(0),HalfFullC(6)[j],uint64_t(64),guardedRows,segment);
            Mul<float,false>(HalfFullW(4),HalfFullW(0),HalfFullW(0),uint64_t(64),guardedRows,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(1),HalfFullW(1),uint64_t(64),guardedRows,same);
            MulAddDst<float,float,false>(HalfFullW(3),HalfFullW(1),HalfFullC(7)[j],uint64_t(64),guardedRows,segment);
            } else {
            MulAddDst<float,float,false>(HalfFullW(3),HalfFullW(1),HalfFullC(7)[j],uint64_t(64),guardedRows,segment);
            MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(1),HalfFullW(1),uint64_t(64),guardedRows,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(2),HalfFullW(2),uint64_t(64),guardedRows,same);
            MulAddDst<float,float,false>(HalfFullW(3),HalfFullW(2),HalfFullC(8)[j],uint64_t(64),guardedRows,segment);
            } else {
            MulAddDst<float,float,false>(HalfFullW(3),HalfFullW(2),HalfFullC(8)[j],uint64_t(64),guardedRows,segment);
            MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(2),HalfFullW(2),uint64_t(64),guardedRows,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(CLEAN_PRIMARY64) {
                // APz is dead here; keep the clamp in a bank phase distinct from dot.
                if constexpr(sizeof(T)==2)
                    Muls<float,false>(HalfFullW(5),HalfFullW(4),lowerNormFactor,uint64_t(64),guardedRows,unary);
                Maxs<float,false>(HalfFullW(2),HalfFullW(3),0.0F,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
                Min<float,false>(HalfFullW(2),HalfFullW(2),HalfFullC(9)[j],uint64_t(64),guardedRows,segment);PipeBarrier<PIPE_V>();
                const BinaryRepeatParams fused{1,0,1,8,0,8};
                if constexpr(sizeof(T)==4)
                    FusedMulAdd<float,false>(HalfFullW(3),refined_[520],HalfFullW(2),uint64_t(64),guardedRows,fused);
                else {
                    constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                    const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                    FusedMulAdd<float,false>(HalfFullW(3),minusTwo,HalfFullW(2),uint64_t(64),guardedRows,fused);
                }
                if constexpr(sizeof(T)==4)
                    Muls<float,false>(HalfFullW(5),HalfFullW(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);
                PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(HalfFullW(5),HalfFullW(2),HalfFullW(3),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
            } else {
            Maxs<float,false>(HalfFullW(0),HalfFullW(3),0.0F,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(HalfFullW(0),HalfFullW(0),HalfFullC(9)[j],uint64_t(64),guardedRows,segment);PipeBarrier<PIPE_V>();
            Muls<float,false>(HalfFullW(2),HalfFullW(3),-2.0F,uint64_t(64),guardedRows,unary);
            Muls<float,false>(HalfFullW(5),HalfFullW(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
            Add<float,false>(HalfFullW(1),HalfFullW(0),HalfFullW(2),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(0),HalfFullW(1),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
            }
            // W5 already holds an outward-biased lower score; normW4 stays live.
            if(j+64>segmentCount) {
                CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(segmentCount-j)),CMPMODE::LT,uint64_t(64),guardedRows,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
                Select(HalfFullW(5),valid_,HalfFullW(5),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(guardedRows*64));PipeBarrier<PIPE_V>();
            }
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
                if(j+64>segmentCount) {
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                }
                WholeReduceMin<float,false>(record_,HalfFullW(5),uint64_t(64),guardedRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            } else WholeReduceMin(record_,HalfFullW(5),64,guardedRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)
                Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
            else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            if constexpr(FULL_FIXED_MASK) {
                if(guardedRows==64) {
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                    Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,HalfFullW(4),offset_.template ReinterpretCast<uint32_t>(),0U,64);
                } else {
                    Muls(offset_,tileIdx_,int32_t(4),guardedRows);PipeBarrier<PIPE_V>();
                    Add(offset_,offset_,rowBase_,guardedRows);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,HalfFullW(4),offset_.template ReinterpretCast<uint32_t>(),0U,guardedRows);
                }
            } else {
                Muls(offset_,tileIdx_,int32_t(4),guardedRows);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,guardedRows);PipeBarrier<PIPE_V>();
                Gather(tileHi_,HalfFullW(4),offset_.template ReinterpretCast<uint32_t>(),0U,guardedRows);
            }
            PipeBarrier<PIPE_V>();
            auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
            auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Axpy<float,float,false>(tileLo_,tileHi_,upperNormFactor,uint64_t(64),1,unary);
                PipeBarrier<PIPE_V>();
            } else if constexpr(CLEAN_PRIMARY64) {
                constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
                constexpr uint32_t constantBase=14*(CAPACITY+32)+pointPitch-16;
                const LocalTensor<float> upperScale(TPosition::VECCALC,constantBase*4,8);
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
                PipeBarrier<PIPE_V>();
            } else if constexpr(FULL_FIXED_MASK) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<float,false>(tileHi_,tileHi_,2.125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<float,false>(tileHi_,tileLo_,tileHi_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            } else {
                Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
                Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
            }
            Brcb(HalfFullW(1),chunkUpper,(guardedRows+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            if constexpr(CLEAN_PRIMARY64) {
                const uint32_t group=j/256,slot=(j/64)%4;
                auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                if(guardedRows==64) {
                    Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),HalfFullW(5),HalfFullW(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));
                } else {
                    Compare<float,uint8_t,false>(mask_,HalfFullW(5),HalfFullW(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                }
            } else {
                Compare<float,uint8_t,false>(mask_,HalfFullW(5),HalfFullW(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                if constexpr(DEFER_CHUNK_CERTIFICATE) {
                    const uint32_t group=j/256,slot=(j/64)%4;
                    auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                    if constexpr(FULL_FIXED_MASK)
                        Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                    else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
                } else {
                    GatherMask(HalfFullW(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                    GatherMask(HalfFullW(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
                }
            }
            PipeBarrier<PIPE_V>();
            {
            if constexpr(!(FULL_FIXED_MASK)) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
            if constexpr(!DEFER_SECOND)
                Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            if constexpr(!DEFER_SECOND)
                Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
            if constexpr(!DEFER_CHUNK_CERTIFICATE) {
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,HalfFullW(0),cert_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,HalfFullW(1),certHigh_,1,same);
            }
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(DEFER_SECOND) {
            const uint32_t chunks=(segmentCount+63)/64;
            if(chunks<16) {
                Duplicate(chunkLower_[chunks*64],__builtin_inff(),(16-chunks)*64);
                PipeBarrier<PIPE_V>();
            }
            // Multi-repeat Compare packs sixteen64-bit results into128 bytes.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,uint8_t,false>(mask_,chunkLower_,hi_,CMPMODE::LE,
                uint64_t(64),16,BinaryRepeatParams{1,1,1,8,8,0});
            PipeBarrier<PIPE_V>();
            auto sourceBits=mask_.template ReinterpretCast<uint16_t>();
            auto unions=HalfFullW(0).template ReinterpretCast<uint16_t>();
            auto collisions=HalfFullW(1).template ReinterpretCast<uint16_t>();
            auto nextUnions=HalfFullW(2).template ReinterpretCast<uint16_t>();
            auto nextCollisions=HalfFullW(3).template ReinterpretCast<uint16_t>();
            auto mergedCollisions=HalfFullW(4).template ReinterpretCast<uint16_t>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(32));
            Or<uint16_t,false>(unions,sourceBits,sourceBits[32],uint64_t(128),1,same);
            And<uint16_t,false>(collisions,sourceBits,sourceBits[32],uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
            Or<uint16_t,false>(nextUnions,unions,unions[16],uint64_t(128),1,same);
            And<uint16_t,false>(nextCollisions,unions,unions[16],uint64_t(128),1,same);
            Or<uint16_t,false>(mergedCollisions,collisions,collisions[16],uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            Or<uint16_t,false>(nextCollisions,nextCollisions,mergedCollisions,uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            // Four independent packed groups remain after the first two levels.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const auto packedUnions=nextUnions.template ReinterpretCast<uint64_t>();
            const auto packedCollisions=nextCollisions.template ReinterpretCast<uint64_t>();
            const uint64_t u0=packedUnions.GetValue(0),u1=packedUnions.GetValue(1);
            const uint64_t u2=packedUnions.GetValue(2),u3=packedUnions.GetValue(3);
            const uint64_t c0=packedCollisions.GetValue(0),c1=packedCollisions.GetValue(1);
            const uint64_t c2=packedCollisions.GetValue(2),c3=packedCollisions.GetValue(3);
            const uint64_t u01=u0|u1,u23=u2|u3;
            const uint64_t c01=c0|c1|(u0&u1),c23=c2|c3|(u2&u3);
            const uint64_t unique=(u01|u23)&~(c01|c23|(u01&u23));
            second_.template ReinterpretCast<uint64_t>().SetValue(0,unique);
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            // Keep the cross-chunk certificate as a packed bitset until Process.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        // Refinement gathers a full register, including inactive point lanes.
        if(guardedRows<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<guardedRows)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();
        if constexpr(DEFER_CHUNK_CERTIFICATE) {
            // Resolve only the winning chunk's two predicate words per point.
            // Indices are byte offsets into the existing usable UB arena.
            const LocalTensor<float> arena(TPosition::VECCALC,0,188416/4);
            if constexpr(FULL_FIXED_MASK) {
                const UnaryRepeatParams resolveUnary{1,1,8,8};
                const BinaryRepeatParams resolveBinary{1,1,1,8,8,8};
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
                ShiftRight<uint32_t,false>(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Muls<int32_t,false>(offset_,offset_,int32_t(32),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(HalfFullW(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Duplicate<int32_t,false>(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(192),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
                SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
                And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),HalfFullW(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,resolveBinary);PipeBarrier<PIPE_V>();
                SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,offset_,int32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,HalfFullW(2).template ReinterpretCast<int32_t>(),uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,outputOffset_,uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Adds<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            } else {
            ShiftRight(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),64);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(32),64);PipeBarrier<PIPE_V>();
            Gather(HalfFullW(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Duplicate(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(192),64);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),HalfFullW(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(8),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,HalfFullW(2).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,outputOffset_,64);PipeBarrier<PIPE_V>();
            Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Adds(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
            Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            }
        }
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        if constexpr(CAPACITY<1024) {
            // A unique bit across both words requires a nonzero one-bit union
            // and no bit occupied in both words.
            Duplicate(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Or(HalfFullW(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
            And(HalfFullW(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Sub(HalfFullW(0).template ReinterpretCast<int32_t>(),HalfFullW(2).template ReinterpretCast<int32_t>(),HalfFullW(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            And(HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(2).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Or(HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(3).template ReinterpretCast<uint16_t>(),128);
            Cast(HalfFullW(1),HalfFullW(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Cast(HalfFullW(0),HalfFullW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,HalfFullW(0),0.0F,CMPMODE::EQ,64);
            CompareScalar(smallMask_,HalfFullW(1),0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        } else {
        if constexpr(FULL_FIXED_MASK) {
            const UnaryRepeatParams certUnary{1,1,8,8};
            const BinaryRepeatParams certBinary{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Duplicate<int32_t,false>(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),HalfFullW(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
            Gather(HalfFullW(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(HalfFullW(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
            And<uint16_t,false>(HalfFullW(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Sub<int32_t,false>(HalfFullW(0).template ReinterpretCast<int32_t>(),HalfFullW(2).template ReinterpretCast<int32_t>(),HalfFullW(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else {
        Duplicate(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(31),64);PipeBarrier<PIPE_V>();
        And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),HalfFullW(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
        Gather(HalfFullW(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        Or(HalfFullW(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
        And(HalfFullW(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Sub(HalfFullW(0).template ReinterpretCast<int32_t>(),HalfFullW(2).template ReinterpretCast<int32_t>(),HalfFullW(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
        Or(HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        }
        Cast(HalfFullW(0),HalfFullW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,HalfFullW(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        }
        return count_;
    }
#endif
    __aicore__ inline void SearchGuardedError() {
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            SearchStaticPolicy<false>();
        } else {
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64;
        const float coarseErrorFactor=(tightCoefficientSafe_ && CoarseCoordinatesSafe<false>())?
            7.62939453125e-6F:1.52587890625e-5F;
        if constexpr((sizeof(T)==4 && (CAPACITY==1024 || CAPACITY==256)) || FULL_FIXED_MASK) {
            InitializeGroupedSearchState(lo_,idx_,DEFER_CHUNK_CERTIFICATE?1:3);
        } else {
            Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            if constexpr(!DEFER_CHUNK_CERTIFICATE){Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);}
        }
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(liveRows_*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        for(uint32_t j=0;j<t_.segments;j+=64) {
            if constexpr(COMPACT_TAIL) {
                if(t_.segments-j<=8) {
                    CompactSearchTail<false>(j,t_.segments-j);MergeSearchMinimum<false>(j);continue;
                }
            }
            if constexpr(!CLEAN_PRIMARY64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),P(axis),C(axis)[j],uint64_t(64),liveRows_,pointSegment);
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),liveRows_,same);
            Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),liveRows_,segment);
            } else {
            Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),liveRows_,segment);
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),liveRows_,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),liveRows_,same);
            MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),liveRows_,segment);
            } else {
            MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),liveRows_,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),liveRows_,same);
            MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),liveRows_,segment);
            } else {
            MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),liveRows_,segment);
            MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),liveRows_,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(CLEAN_PRIMARY64) {
                // APz is dead here; keep the clamp in a bank phase distinct from dot.
                if constexpr(sizeof(T)==2)
                    Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),liveRows_,unary);
                Maxs<float,false>(W(2),W(3),0.0F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
                Min<float,false>(W(2),W(2),C(9)[j],uint64_t(64),liveRows_,segment);PipeBarrier<PIPE_V>();
                const BinaryRepeatParams fused{1,0,1,8,0,8};
                {
                    constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                    const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                    FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),liveRows_,fused);
                }
                if constexpr(sizeof(T)==4)
                    Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),liveRows_,unary);
                PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(W(4),W(2),W(3),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
            } else {
            Maxs<float,false>(W(0),W(3),0.0F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(W(0),W(0),C(9)[j],uint64_t(64),liveRows_,segment);PipeBarrier<PIPE_V>();
            Muls<float,false>(W(2),W(3),-2.0F,uint64_t(64),liveRows_,unary);
            Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
            Add<float,false>(W(1),W(0),W(2),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(4),W(0),W(1),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
            }
            Sub<float,false>(W(0),W(4),W(5),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
            if(j+64>t_.segments) {
                CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(t_.segments-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
                Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
            }
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
                if(j+64>t_.segments) {
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                }
                WholeReduceMin<float,false>(record_,W(0),uint64_t(64),liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            } else WholeReduceMin(record_,W(0),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)
                Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
            else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            if constexpr(FULL_FIXED_MASK) {
                if(liveRows_==64) {
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                    Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,64);
                } else {
                    Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
                    Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
                }
            } else {
                Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
                Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
            }
            PipeBarrier<PIPE_V>();
            auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
            auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Axpy<float,float,false>(tileLo_,tileHi_,2.125F,uint64_t(64),1,unary);
                PipeBarrier<PIPE_V>();
            } else if constexpr(CLEAN_PRIMARY64) {
                constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
                constexpr uint32_t constantBase=14*(CAPACITY+32)+pointPitch-16;
                const LocalTensor<float> upperScale(TPosition::VECCALC,constantBase*4,8);
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
                PipeBarrier<PIPE_V>();
            } else if constexpr(FULL_FIXED_MASK) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<float,false>(tileHi_,tileHi_,2.125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<float,false>(tileHi_,tileLo_,tileHi_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            } else {
                Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
                Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
            }
            Brcb(W(1),chunkUpper,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            if constexpr(CLEAN_PRIMARY64) {
                const uint32_t group=j/256,slot=(j/64)%4;
                auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                if(liveRows_==64) {
                    Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),W(0),W(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));
                } else {
                    Compare<float,uint8_t,false>(mask_,W(0),W(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                }
            } else {
                Compare<float,uint8_t,false>(mask_,W(0),W(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                if constexpr(DEFER_CHUNK_CERTIFICATE) {
                    const uint32_t group=j/256,slot=(j/64)%4;
                    auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                    if constexpr(FULL_FIXED_MASK)
                        Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                    else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
                } else {
                    GatherMask(W(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                    GatherMask(W(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
                }
            }
            PipeBarrier<PIPE_V>();
            if constexpr(COMPACT_TAIL) {MergeSearchMinimum<false>(j);} else {
            if constexpr(!(FULL_FIXED_MASK)) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
            if constexpr(!DEFER_SECOND)
                Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            if constexpr(!DEFER_SECOND)
                Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
            if constexpr(!DEFER_CHUNK_CERTIFICATE) {
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,W(0),cert_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,W(1),certHigh_,1,same);
            }
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(DEFER_SECOND) {
            const uint32_t chunks=(t_.segments+63)/64;
            BuildCrossChunkCertificate(chunks,same,unary);
        }
        FinalizeGuardedCertificate(liveRows_);
        }
    }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ __attribute__((always_inline)) inline bool CertifyCorePointCoordinates(uint32_t begin,uint32_t end) {
        // This is a data-range certificate, not a shape or test identity check.
        if(end-begin<ROWS*8)return false;
        auto raw=work_,absolute=work_[4224],nonzero=work_[8448],minimumRows=work_[12672];
        for(uint32_t coordinate=begin*3;coordinate<end*3;coordinate+=4096) {
            const uint32_t available=Smallest(end*3-coordinate,4096);
            const uint32_t aligned=(available+63)/64*64;
            if(aligned!=available)Duplicate(raw,0.0F,aligned);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            DataCopyPad(raw,pg_[coordinate],DataCopyExtParams{1,available*4,0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            Abs(absolute,raw,aligned);PipeBarrier<PIPE_V>();
            CompareScalar(mask_,absolute,0.0F,CMPMODE::NE,aligned);PipeBarrier<PIPE_V>();
            Select(nonzero,mask_,absolute,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,aligned);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(record_,absolute,64,aligned/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            WholeReduceMin(minimumRows,nonzero,64,aligned/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(tileLo_,record_,aligned/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            WholeReduceMin(tileHi_,minimumRows,aligned/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            CompareScalar(smallMask_,tileLo_,1099511627776.0F,CMPMODE::LE,64);
            CompareScalar(valid_,tileHi_,9.5367431640625e-7F,CMPMODE::GE,64);
            PipeBarrier<PIPE_V>();
            And(smallMask_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),4);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if((smallMask_.GetValue(0)&1U)==0) {PipeBarrier<PIPE_ALL>();return false;}
        }
        PipeBarrier<PIPE_ALL>();
        return true;
    }
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_RETURN
    __aicore__ __attribute__((always_inline)) inline bool SearchGuardedCoreCertifiedRows(bool corePointSafe,bool coefficientCoordinatesSafe) {
#else // TREE_STREAM_RETURN
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedCoreCertifiedRows(bool corePointSafe,bool coefficientCoordinatesSafe) {
#endif // TREE_STREAM_RETURN
        // Cache endpoint range checks per batch; use the original point certificate per tile when needed.
        const bool pointSafe=corePointSafe ||
            (coefficientCoordinatesSafe && liveRows_==64 && CoarseCoordinatesSafe<false>());
        if(coefficientCoordinatesSafe && pointSafe && liveRows_==64) {
            if(t_.segments==COEFFICIENTS)SearchGuardedCoreCertifiedRowsAlignedImpl(pointSafe);
            else SearchGuardedCoreCertifiedRowsImpl<true>(pointSafe);
        } else SearchGuardedCoreCertifiedRowsImpl<false>(corePointSafe);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
        return pointSafe;
#endif // TREE_STREAM_EXACT
    }
    template<bool SHARED_BOUND,bool COMPLETE_COEFFICIENTS>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBody(bool corePointSafe) {
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            SearchStaticPolicy<false>();
        } else {
        const uint32_t searchRows=SHARED_BOUND?64U:liveRows_;
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64;
        const float coarseErrorFactor=(tightCoefficientSafe_ &&
            (corePointSafe || CoarseCoordinatesSafe<false>()))?
            7.62939453125e-6F:1.52587890625e-5F;
        constexpr bool sharedBound=SHARED_BOUND;
        auto pointBound=refined_[12*64];
        const BinaryRepeatParams sharedSame{1,1,1,8,8,8};
        const UnaryRepeatParams sharedUnary{1,1,8,8};
        if constexpr(SHARED_BOUND) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            // RN inflation bounds every rounded AP norm by the positive coordinate envelope.
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Abs<float,false>(tileSecond_,compactP_[axis*64],uint64_t(64),1,sharedUnary);
                Brcb(W(0),C(axis)[CAPACITY+16],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Add<float,false>(tileSecond_,tileSecond_,W(0),uint64_t(64),1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
                if(axis==0)Mul<float,false>(pointBound,tileSecond_,tileSecond_,uint64_t(64),1,sharedSame);
                else MulAddDst<float,float,false>(pointBound,tileSecond_,tileSecond_,uint64_t(64),1,sharedSame);
                PipeBarrier<PIPE_V>();
            }
            Muls<float,false>(pointBound,pointBound,coarseErrorFactor*1.000003814697265625F,uint64_t(64),1,sharedUnary);PipeBarrier<PIPE_V>();
        }
        auto lowerScores=sharedBound?W(4):W(0);
        if constexpr((sizeof(T)==4 && (CAPACITY==1024 || CAPACITY==256)) || FULL_FIXED_MASK) {
            InitializeGroupedSearchState(lo_,idx_,DEFER_CHUNK_CERTIFICATE?1:3);
        } else {
            Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            if constexpr(!DEFER_CHUNK_CERTIFICATE){Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);}
        }
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(searchRows*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        for(uint32_t j=0;j<(COMPLETE_COEFFICIENTS?COEFFICIENTS:t_.segments);j+=64) {
            if constexpr(COMPACT_TAIL) {
                if((COMPLETE_COEFFICIENTS?COEFFICIENTS:t_.segments)-j<=8) {
                    CompactSearchTail<false>(j,(COMPLETE_COEFFICIENTS?COEFFICIENTS:t_.segments)-j);MergeSearchMinimum<false>(j);continue;
                }
            }
            if constexpr(!CLEAN_PRIMARY64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            if constexpr(COMPLETE_COEFFICIENTS) {
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && CLEAN_PRIMARY64) {
                auto dotZ=W(2);auto projectedClamp=W(3);
                Sub<float,false>(W(0),P(0),C(0)[j],uint64_t(64),searchRows,pointSegment);
                Sub<float,false>(W(1),P(1),C(1)[j],uint64_t(64),searchRows,pointSegment);
                Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
                Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
                Sub<float,false>(dotZ,P(2),C(2)[j],uint64_t(64),searchRows,pointSegment);
                MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
                MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
                MulAddDst<float,float,false>(W(4),dotZ,dotZ,uint64_t(64),searchRows,same);
                FusedMulAdd<float,false>(dotZ,C(8)[j],W(3),uint64_t(64),searchRows,BinaryRepeatParams{1,1,1,8,0,8});
                Maxs<float,false>(projectedClamp,dotZ,0.0F,uint64_t(64),searchRows,unary);
                Min<float,false>(projectedClamp,projectedClamp,C(9)[j],uint64_t(64),searchRows,segment);
                constexpr uint32_t constantBase=4*(CAPACITY+32)+CAPACITY+8;
                const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                FusedMulAdd<float,false>(dotZ,minusTwo,projectedClamp,uint64_t(64),searchRows,BinaryRepeatParams{1,0,1,8,0,8});
                MulAddDst<float,float,false>(W(4),projectedClamp,dotZ,uint64_t(64),searchRows,same);
            } else {
#endif
            Sub<float,false>(W(0),P(0),C(0)[j],uint64_t(64),searchRows,pointSegment);
            Sub<float,false>(W(1),P(1),C(1)[j],uint64_t(64),searchRows,pointSegment);
            
            if constexpr(NORM_BEFORE_DOT) {
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
            Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
            } else {
            Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
            }
            Sub<float,false>(W(2),P(2),C(2)[j],uint64_t(64),searchRows,pointSegment);
            
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
            MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
            } else {
            MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
            MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
            }
            
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),searchRows,same);
            MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),searchRows,segment);
            } else {
            MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),searchRows,segment);
            MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),searchRows,same);
            }
            
            if constexpr(CLEAN_PRIMARY64) {
                // APz is dead here; keep the clamp in a bank phase distinct from dot.
                if constexpr(sizeof(T)==2)
                    Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);
                Maxs<float,false>(W(2),W(3),0.0F,uint64_t(64),searchRows,unary);
                Min<float,false>(W(2),W(2),C(9)[j],uint64_t(64),searchRows,segment);
                const BinaryRepeatParams fused{1,0,1,8,0,8};
                if constexpr(sizeof(T)==4) {
                    constexpr uint32_t constantBase=4*(CAPACITY+32)+CAPACITY+8;
                    const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                    FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),searchRows,fused);
                } else {
                    constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                    const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                    FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),searchRows,fused);
                }
                if constexpr(sizeof(T)==4) {
                    if constexpr(!SHARED_BOUND)Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);
                }
                
                MulAddDst<float,float,false>(W(4),W(2),W(3),uint64_t(64),searchRows,same);
            } else {
            Maxs<float,false>(W(0),W(3),0.0F,uint64_t(64),searchRows,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(W(0),W(0),C(9)[j],uint64_t(64),searchRows,segment);PipeBarrier<PIPE_V>();
            Muls<float,false>(W(2),W(3),-2.0F,uint64_t(64),searchRows,unary);
            Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);PipeBarrier<PIPE_V>();
            Add<float,false>(W(1),W(0),W(2),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(4),W(0),W(1),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();
            }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            }
#endif
            {}

            } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),P(axis),C(axis)[j],uint64_t(64),searchRows,pointSegment);
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
            Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
            } else {
            Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
            MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
            } else {
            MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
            MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(NORM_BEFORE_DOT) {
            MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),searchRows,same);
            MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),searchRows,segment);
            } else {
            MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),searchRows,segment);
            MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),searchRows,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(CLEAN_PRIMARY64) {
                // APz is dead here; keep the clamp in a bank phase distinct from dot.
                if constexpr(sizeof(T)==2)
                    Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);
                Maxs<float,false>(W(2),W(3),0.0F,uint64_t(64),searchRows,unary);PipeBarrier<PIPE_V>();
                Min<float,false>(W(2),W(2),C(9)[j],uint64_t(64),searchRows,segment);PipeBarrier<PIPE_V>();
                const BinaryRepeatParams fused{1,0,1,8,0,8};
                {
                    constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                    const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                    FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),searchRows,fused);
                }
                if constexpr(sizeof(T)==4) {
                    if constexpr(!SHARED_BOUND)Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);
                }
                PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(W(4),W(2),W(3),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();
            } else {
            Maxs<float,false>(W(0),W(3),0.0F,uint64_t(64),searchRows,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(W(0),W(0),C(9)[j],uint64_t(64),searchRows,segment);PipeBarrier<PIPE_V>();
            Muls<float,false>(W(2),W(3),-2.0F,uint64_t(64),searchRows,unary);
            Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);PipeBarrier<PIPE_V>();
            Add<float,false>(W(1),W(0),W(2),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(4),W(0),W(1),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();
            }
            if constexpr(!SHARED_BOUND) {Sub<float,false>(W(0),W(4),W(5),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();}
            if(j+64>t_.segments) {
                CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(t_.segments-j)),CMPMODE::LT,uint64_t(64),searchRows,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
                Select(lowerScores,valid_,lowerScores,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(searchRows*64));PipeBarrier<PIPE_V>();
            }
            }
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
                if constexpr(!COMPLETE_COEFFICIENTS) {
                if(j+64>t_.segments) {
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                }
                }
                WholeReduceMin<float,false>(record_,lowerScores,uint64_t(64),searchRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            } else WholeReduceMin(record_,lowerScores,64,searchRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            if constexpr(!COMPLETE_COEFFICIENTS && !SHARED_BOUND) {
                if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)
                    Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
                else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            }
            if constexpr(SHARED_BOUND) {
                Copy<float,false>(tileHi_,pointBound,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
                Sub<float,false>(tileLo_,tileLo_,pointBound,uint64_t(64),1,sharedSame);PipeBarrier<PIPE_V>();
                Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
            } else if constexpr(FULL_FIXED_MASK) {
                if(searchRows==64) {
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                    Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,64);
                } else {
                    Muls(offset_,tileIdx_,int32_t(4),searchRows);PipeBarrier<PIPE_V>();
                    Add(offset_,offset_,rowBase_,searchRows);PipeBarrier<PIPE_V>();
                    Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,searchRows);
                }
            } else {
                Muls(offset_,tileIdx_,int32_t(4),searchRows);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,searchRows);PipeBarrier<PIPE_V>();
                Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,searchRows);
            }
            PipeBarrier<PIPE_V>();
            auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
            auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Axpy<float,float,false>(tileLo_,tileHi_,2.125F,uint64_t(64),1,unary);
                PipeBarrier<PIPE_V>();
            } else if constexpr(CLEAN_PRIMARY64) {
                constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
                constexpr uint32_t constantBase=14*(CAPACITY+32)+pointPitch-16;
                const LocalTensor<float> upperScale(TPosition::VECCALC,constantBase*4,8);
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
                PipeBarrier<PIPE_V>();
            } else if constexpr(FULL_FIXED_MASK) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<float,false>(tileHi_,tileHi_,2.125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<float,false>(tileHi_,tileLo_,tileHi_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            } else {
                Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
                Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
            }
            if constexpr(SHARED_BOUND) {
                Add<float,false>(tileSecond_,chunkUpper,pointBound,uint64_t(64),1,sharedSame);PipeBarrier<PIPE_V>();
                Brcb(W(1),tileSecond_,(searchRows+7)/8,BrcbRepeatParams(1,8));
            } else Brcb(W(1),chunkUpper,(searchRows+7)/8,BrcbRepeatParams(1,8));
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            if constexpr(CLEAN_PRIMARY64) {
                const uint32_t group=j/256,slot=(j/64)%4;
                auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                if(searchRows==64) {
                    Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),lowerScores,W(1),CMPMODE::LE,uint64_t(64),searchRows,BinaryRepeatParams(1,1,0,8,8,1));
                } else {
                    Compare<float,uint8_t,false>(mask_,lowerScores,W(1),CMPMODE::LE,uint64_t(64),searchRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                }
            } else {
                Compare<float,uint8_t,false>(mask_,lowerScores,W(1),CMPMODE::LE,uint64_t(64),searchRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                if constexpr(DEFER_CHUNK_CERTIFICATE) {
                    const uint32_t group=j/256,slot=(j/64)%4;
                    auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                    if constexpr(FULL_FIXED_MASK)
                        Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                    else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
                } else {
                    GatherMask(W(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                    GatherMask(W(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
                }
            }
            PipeBarrier<PIPE_V>();
            if constexpr(COMPACT_TAIL) {MergeSearchMinimum<false>(j);} else {
            if constexpr(!(FULL_FIXED_MASK)) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
            if constexpr(!DEFER_SECOND)
                Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            if constexpr(!DEFER_SECOND)
                Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
            if constexpr(!DEFER_CHUNK_CERTIFICATE) {
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,W(0),cert_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,W(1),certHigh_,1,same);
            }
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(DEFER_SECOND) {
            const uint32_t chunks=((COMPLETE_COEFFICIENTS?COEFFICIENTS:t_.segments)+63)/64;
            if(chunks<16) {
                Duplicate(chunkLower_[chunks*64],__builtin_inff(),(16-chunks)*64);
                PipeBarrier<PIPE_V>();
            }
            // Multi-repeat Compare packs sixteen64-bit results into128 bytes.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,uint8_t,false>(mask_,chunkLower_,hi_,CMPMODE::LE,
                uint64_t(64),16,BinaryRepeatParams{1,1,1,8,8,0});
            PipeBarrier<PIPE_V>();
            auto sourceBits=mask_.template ReinterpretCast<uint16_t>();
            auto unions=W(0).template ReinterpretCast<uint16_t>();
            auto collisions=W(1).template ReinterpretCast<uint16_t>();
            auto nextUnions=W(2).template ReinterpretCast<uint16_t>();
            auto nextCollisions=W(3).template ReinterpretCast<uint16_t>();
            auto mergedCollisions=W(4).template ReinterpretCast<uint16_t>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(32));
            Or<uint16_t,false>(unions,sourceBits,sourceBits[32],uint64_t(128),1,same);
            And<uint16_t,false>(collisions,sourceBits,sourceBits[32],uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
            Or<uint16_t,false>(nextUnions,unions,unions[16],uint64_t(128),1,same);
            And<uint16_t,false>(nextCollisions,unions,unions[16],uint64_t(128),1,same);
            Or<uint16_t,false>(mergedCollisions,collisions,collisions[16],uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            Or<uint16_t,false>(nextCollisions,nextCollisions,mergedCollisions,uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            auto oldUnions=unions;unions=nextUnions;nextUnions=oldUnions;
            auto oldCollisions=collisions;collisions=nextCollisions;nextCollisions=oldCollisions;
            #pragma unroll
            for(uint32_t stage=0;stage<2;++stage) {
                auto rotation=C(3)[CAPACITY+16+stage*8].template ReinterpretCast<uint32_t>();
                Gather(nextUnions.template ReinterpretCast<float>(),unions.template ReinterpretCast<float>(),rotation,0U,8);
                Gather(nextCollisions.template ReinterpretCast<float>(),collisions.template ReinterpretCast<float>(),rotation,0U,8);
                PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
                And<uint16_t,false>(mergedCollisions,unions,nextUnions,uint64_t(128),1,same);
                Or<uint16_t,false>(nextUnions,unions,nextUnions,uint64_t(128),1,same);
                Or<uint16_t,false>(nextCollisions,collisions,nextCollisions,uint64_t(128),1,same);
                PipeBarrier<PIPE_V>();
                Or<uint16_t,false>(nextCollisions,nextCollisions,mergedCollisions,uint64_t(128),1,same);
                PipeBarrier<PIPE_V>();
                auto oldU=unions;unions=nextUnions;nextUnions=oldU;
                auto oldC=collisions;collisions=nextCollisions;nextCollisions=oldC;
            }
            Not<uint16_t,false>(second_.template ReinterpretCast<uint16_t>(),collisions,uint64_t(128),1,unary);
            PipeBarrier<PIPE_V>();
            And<uint16_t,false>(second_.template ReinterpretCast<uint16_t>(),
                second_.template ReinterpretCast<uint16_t>(),unions,uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            // Keep the cross-chunk certificate as a packed bitset until Process.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        FinalizeGuardedCertificate(searchRows);
        }
    }

    template<bool SHARED_BOUND>
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ __attribute__((noinline)) void SearchGuardedCoreCertifiedRowsImpl(bool corePointSafe) {
#else
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedCoreCertifiedRowsImpl(bool corePointSafe) {
#endif
        SearchCertifiedRowsBody<SHARED_BOUND,false>(corePointSafe);
    }
    // The complete coefficient tile has no segment padding; all original row and range certificates still apply.
    __aicore__ __attribute__((noinline)) void SearchGuardedCoreCertifiedRowsAlignedImpl(bool corePointSafe) {
        SearchCertifiedRowsBody<true,true>(corePointSafe);
    }

    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> SpatialStateC(uint32_t k) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full spatial resource required");
        constexpr uint32_t pitch=1056;
        const uint32_t offset=k>=6 && k<=8?6*pitch+(k-6)*1024:k*pitch;
        return LocalTensor<float>(TPosition::VECCALC,offset*4,14*pitch-offset);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> SpatialStateW(uint32_t k) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full spatial resource required");
        constexpr uint32_t pitch=4224,base=17952;
        constexpr uint32_t phases[13]={32,0,0,64,96,32,96,0,0,0,0,0,0};
        const uint32_t offset=k*pitch+phases[k];
        return LocalTensor<float>(TPosition::VECCALC,(base+offset)*4,6*pitch-offset);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> SpatialStateP(uint32_t k) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full spatial resource required");
        constexpr uint32_t pitch=1056,base=14784;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,3*pitch-k*pitch);
    }

    __aicore__ __attribute__((noinline)) static uint32_t SearchSpatialStaticState(uint32_t first,uint32_t limit,bool resume,bool tightCoefficientSafe_) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Full native Float spatial state required");
        static_assert(CLEAN_PRIMARY64 && NORM_BEFORE_DOT && DEFER_CHUNK_CERTIFICATE && FULL_FIXED_MASK, "Full spatial search uses fixed-mask deferred certificates");
        LocalTensor<float> certHigh_(TPosition::VECCALC,184256U,64U);
        LocalTensor<float> cert_(TPosition::VECCALC,184000U,64U);
        LocalTensor<float> chunkLower_(TPosition::VECCALC,174208U,1024U);
        LocalTensor<float> hi_(TPosition::VECCALC,183488U,64U);
        LocalTensor<int32_t> idx_(TPosition::VECCALC,183744U,64U);
        LocalTensor<float> lo_(TPosition::VECCALC,182976U,64U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187072U,512U);
        LocalTensor<int32_t> offset_(TPosition::VECCALC,185792U,64U);
        LocalTensor<int32_t> oneHot_(TPosition::VECCALC,184512U,64U);
        LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,186560U,64U);
        LocalTensor<float> record_(TPosition::VECCALC,182400U,128U);
        LocalTensor<float> refined_(TPosition::VECCALC,179072U,832U);
        LocalTensor<int32_t> rowBase_(TPosition::VECCALC,186048U,64U);
        LocalTensor<float> second_(TPosition::VECCALC,183232U,64U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185536U,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,184768U,64U);
        LocalTensor<uint8_t> valid_(TPosition::VECCALC,187584U,512U);
        LocalTensor<float> xyz_(TPosition::VECCALC,59136U,3168U);
        constexpr uint32_t pp_=1056;

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        AscendC::ICachePreLoad(int64_t(2));
#endif
        // Only successful compaction starts below the full coefficient extent.
        // Complete its map transfer before the first overlapping W5 write.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        bool pendingSpatialMap=false;
#else
        bool pendingSpatialMap=!resume && first==0U && limit<COEFFICIENTS;
#endif
        constexpr uint32_t searchRows=64U;
        const float coarseErrorFactor=tightCoefficientSafe_?7.62939453125e-6F:1.52587890625e-5F;
        const float intervalFactor=coarseErrorFactor*1.0625F;
        const float lowerNormFactor=1.0F-intervalFactor;
        const float upperNormFactor=intervalFactor*2.125F*1.000003814697265625F;

        auto lowerScores=SpatialStateW(5);
        if(!resume) {
            InitializeGroupedSearchState(lo_,idx_,1);

            PipeBarrier<PIPE_V>();
        }
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        const uint32_t count_=(searchRows*64);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        for(uint32_t j=first;j<limit;j+=64) {

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            auto dotZ=SpatialStateW(2);auto projectedClamp=SpatialStateW(3);
            Sub<float,false>(SpatialStateW(0),SpatialStateP(0),SpatialStateC(0)[j],uint64_t(64),searchRows,pointSegment);
            Sub<float,false>(SpatialStateW(1),SpatialStateP(1),SpatialStateC(1)[j],uint64_t(64),searchRows,pointSegment);
            Mul<float,false>(SpatialStateW(4),SpatialStateW(0),SpatialStateW(0),uint64_t(64),searchRows,same);
            Mul<float,false>(SpatialStateW(3),SpatialStateW(0),SpatialStateC(6)[j],uint64_t(64),searchRows,segment);
            Sub<float,false>(dotZ,SpatialStateP(2),SpatialStateC(2)[j],uint64_t(64),searchRows,pointSegment);
            MulAddDst<float,float,false>(SpatialStateW(4),SpatialStateW(1),SpatialStateW(1),uint64_t(64),searchRows,same);
            MulAddDst<float,float,false>(SpatialStateW(3),SpatialStateW(1),SpatialStateC(7)[j],uint64_t(64),searchRows,segment);
            MulAddDst<float,float,false>(SpatialStateW(4),dotZ,dotZ,uint64_t(64),searchRows,same);
            FusedMulAdd<float,false>(dotZ,SpatialStateC(8)[j],SpatialStateW(3),uint64_t(64),searchRows,BinaryRepeatParams{1,1,1,8,0,8});
            // Complete the pending map before reusing its W5 source.
            if(pendingSpatialMap) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                pendingSpatialMap=false;
            }
            // Preserve the AP norm in W4; bias its independent W5 copy.
            Muls<float,false>(SpatialStateW(5),SpatialStateW(4),lowerNormFactor,uint64_t(64),searchRows,unary);
            Maxs<float,false>(projectedClamp,dotZ,0.0F,uint64_t(64),searchRows,unary);
            Min<float,false>(projectedClamp,projectedClamp,SpatialStateC(9)[j],uint64_t(64),searchRows,segment);
            constexpr uint32_t constantBase=4*(CAPACITY+32)+CAPACITY+8;
            const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
            FusedMulAdd<float,false>(dotZ,minusTwo,projectedClamp,uint64_t(64),searchRows,BinaryRepeatParams{1,0,1,8,0,8});
            MulAddDst<float,float,false>(SpatialStateW(5),projectedClamp,dotZ,uint64_t(64),searchRows,same);
#else
            Sub<float,false>(SpatialStateW(0),SpatialStateP(0),SpatialStateC(0)[j],uint64_t(64),searchRows,pointSegment);
            Sub<float,false>(SpatialStateW(1),SpatialStateP(1),SpatialStateC(1)[j],uint64_t(64),searchRows,pointSegment);

            Mul<float,false>(SpatialStateW(4),SpatialStateW(0),SpatialStateW(0),uint64_t(64),searchRows,same);
            Mul<float,false>(SpatialStateW(3),SpatialStateW(0),SpatialStateC(6)[j],uint64_t(64),searchRows,segment);

            Sub<float,false>(SpatialStateW(2),SpatialStateP(2),SpatialStateC(2)[j],uint64_t(64),searchRows,pointSegment);

            MulAddDst<float,float,false>(SpatialStateW(4),SpatialStateW(1),SpatialStateW(1),uint64_t(64),searchRows,same);
            MulAddDst<float,float,false>(SpatialStateW(3),SpatialStateW(1),SpatialStateC(7)[j],uint64_t(64),searchRows,segment);

            MulAddDst<float,float,false>(SpatialStateW(4),SpatialStateW(2),SpatialStateW(2),uint64_t(64),searchRows,same);
            MulAddDst<float,float,false>(SpatialStateW(3),SpatialStateW(2),SpatialStateC(8)[j],uint64_t(64),searchRows,segment);

            // APz is dead here; keep the clamp in a bank phase distinct from dot.

            Maxs<float,false>(SpatialStateW(2),SpatialStateW(3),0.0F,uint64_t(64),searchRows,unary);
            Min<float,false>(SpatialStateW(2),SpatialStateW(2),SpatialStateC(9)[j],uint64_t(64),searchRows,segment);
            const BinaryRepeatParams fused{1,0,1,8,0,8};
            constexpr uint32_t constantBase=4*(CAPACITY+32)+CAPACITY+8;
            const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
            FusedMulAdd<float,false>(SpatialStateW(3),minusTwo,SpatialStateW(2),uint64_t(64),searchRows,fused);

            Muls<float,false>(SpatialStateW(5),SpatialStateW(4),coarseErrorFactor,uint64_t(64),searchRows,unary);

            MulAddDst<float,float,false>(SpatialStateW(4),SpatialStateW(2),SpatialStateW(3),uint64_t(64),searchRows,same);

#endif
            // W5 already contains the outward-biased lower score.
            PipeBarrier<PIPE_V>();

            WholeReduceMin<float,false>(record_,lowerScores,uint64_t(64),searchRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);

            PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));

            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Gather(tileHi_,SpatialStateW(4),offset_.template ReinterpretCast<uint32_t>(),0U,64);

            PipeBarrier<PIPE_V>();
            auto chunkMinimum=chunkLower_[j];
            auto chunkUpper=tileLo_;
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            // The unchanged chunkLower copy retains the minimum for merging.
            Axpy<float,float,false>(tileLo_,tileHi_,upperNormFactor,uint64_t(64),1,unary);
            PipeBarrier<PIPE_V>();

            Brcb(SpatialStateW(1),chunkUpper,(searchRows+7)/8,BrcbRepeatParams(1,8));
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            const uint32_t group=j/256,slot=(j/64)%4;
            auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
            Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),lowerScores,SpatialStateW(1),CMPMODE::LE,uint64_t(64),searchRows,BinaryRepeatParams(1,1,0,8,8,1));

            PipeBarrier<PIPE_V>();

            Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);

            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();

            Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);

            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();

        }
        // A complete single-pass search also finalizes every chunk certificate.
        const uint32_t chunks=(limit+63)/64;
        if(chunks<16) {
            Duplicate(chunkLower_[chunks*64],__builtin_inff(),(16-chunks)*64);
            PipeBarrier<PIPE_V>();
        }
        // Multi-repeat Compare packs sixteen64-bit results into128 bytes.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Compare<float,uint8_t,false>(mask_,chunkLower_,hi_,CMPMODE::LE,
        uint64_t(64),16,BinaryRepeatParams{1,1,1,8,8,0});
        PipeBarrier<PIPE_V>();
        auto sourceBits=mask_.template ReinterpretCast<uint16_t>();
        auto unions=SpatialStateW(0).template ReinterpretCast<uint16_t>();
        auto collisions=SpatialStateW(1).template ReinterpretCast<uint16_t>();
        auto nextUnions=SpatialStateW(2).template ReinterpretCast<uint16_t>();
        auto nextCollisions=SpatialStateW(3).template ReinterpretCast<uint16_t>();
        auto mergedCollisions=SpatialStateW(4).template ReinterpretCast<uint16_t>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(32));
        Or<uint16_t,false>(unions,sourceBits,sourceBits[32],uint64_t(128),1,same);
        And<uint16_t,false>(collisions,sourceBits,sourceBits[32],uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
        Or<uint16_t,false>(nextUnions,unions,unions[16],uint64_t(128),1,same);
        And<uint16_t,false>(nextCollisions,unions,unions[16],uint64_t(128),1,same);
        Or<uint16_t,false>(mergedCollisions,collisions,collisions[16],uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(nextCollisions,nextCollisions,mergedCollisions,uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        // Fold the remaining four spatial union and collision groups.
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const auto packedUnions=nextUnions.template ReinterpretCast<uint64_t>();
        const auto packedCollisions=nextCollisions.template ReinterpretCast<uint64_t>();
        const uint64_t u0=packedUnions.GetValue(0),u1=packedUnions.GetValue(1);
        const uint64_t u2=packedUnions.GetValue(2),u3=packedUnions.GetValue(3);
        const uint64_t c0=packedCollisions.GetValue(0),c1=packedCollisions.GetValue(1);
        const uint64_t c2=packedCollisions.GetValue(2),c3=packedCollisions.GetValue(3);
        const uint64_t u01=u0|u1,u23=u2|u3;
        const uint64_t c01=c0|c1|(u0&u1),c23=c2|c3|(u2&u3);
        const uint64_t unique=(u01|u23)&~(c01|c23|(u01&u23));
        second_.template ReinterpretCast<uint64_t>().SetValue(0,unique);
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        // Keep the cross-chunk certificate as a packed bitset until Process.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        // Refinement gathers a full register, including inactive point lanes.

        PipeBarrier<PIPE_V>();
        // Resolve only the winning chunk's two predicate words per point.
        // Indices are byte offsets into the existing usable UB arena.
        const LocalTensor<float> arena(TPosition::VECCALC,0,188416/4);
        const UnaryRepeatParams resolveUnary{1,1,8,8};
        const BinaryRepeatParams resolveBinary{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
        ShiftRight<uint32_t,false>(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
        Muls<int32_t,false>(offset_,offset_,int32_t(32),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
        Gather(SpatialStateW(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<int32_t,false>(SpatialStateW(4).template ReinterpretCast<int32_t>(),int32_t(192),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SpatialStateW(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,resolveBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Muls<int32_t,false>(offset_,offset_,int32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(offset_,offset_,SpatialStateW(2).template ReinterpretCast<int32_t>(),uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(offset_,offset_,outputOffset_,uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
        Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Adds<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
        Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();

        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        const UnaryRepeatParams certUnary{1,1,8,8};
        const BinaryRepeatParams certBinary{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<int32_t,false>(SpatialStateW(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SpatialStateW(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
        Gather(SpatialStateW(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(SpatialStateW(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
        And<uint16_t,false>(SpatialStateW(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Sub<int32_t,false>(SpatialStateW(0).template ReinterpretCast<int32_t>(),SpatialStateW(2).template ReinterpretCast<int32_t>(),SpatialStateW(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(SpatialStateW(0).template ReinterpretCast<uint16_t>(),SpatialStateW(0).template ReinterpretCast<uint16_t>(),SpatialStateW(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        Cast(SpatialStateW(0),SpatialStateW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,SpatialStateW(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();

        // A compacted seed-only resume performs no W5 write, but its map
        // transfer must still finish before caller scratch or EVENT_ID2 reuse.
        if(pendingSpatialMap)WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        return count_;
    }

    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> SpatialPruneW(uint32_t k) {
        constexpr uint32_t base=17952,pitch=1056;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,6*4224-k*pitch);
    }
    __aicore__ __attribute__((always_inline)) static inline void SpatialPruneIndices(const LocalTensor<int32_t>& values,int32_t first,uint32_t count) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Full native Float pruning resource required");
        const LocalTensor<int32_t> rowBase(TPosition::VECCALC,186048U,64U);
        const uint32_t seed=count<64U?count:64U;
        ShiftRight(values,rowBase,int32_t(8),seed);PipeBarrier<PIPE_V>();
        if(first!=0) {Adds(values,values,first,seed);PipeBarrier<PIPE_V>();}
        for(uint32_t filled=64U;filled<count;filled*=2U) {
            const uint32_t n=count-filled<filled?count-filled:filled;
            Adds(values[filled],values,static_cast<int32_t>(filled),n);PipeBarrier<PIPE_V>();
        }
    }
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    __aicore__ __attribute__((noinline,minsize,aligned(128))) static uint32_t PruneSpatialStaticState(GM_ADDR workspace,uint32_t segments,bool& pruned,uint32_t& logicalSegments) {
#else
    __aicore__ __attribute__((noinline,aligned(128))) static uint32_t PruneSpatialStaticState(GM_ADDR workspace,uint32_t segments,bool& pruned,uint32_t& logicalSegments) {
#endif
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Full native Float spatial pruning state required");
        LocalTensor<float> work_(TPosition::VECCALC,71808U,25344U);
        LocalTensor<float> compactP_(TPosition::VECCALC,178304U,192U);
        LocalTensor<float> record_(TPosition::VECCALC,182400U,128U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U,64U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187072U,512U);
        LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,188096U,32U);

        GlobalTensor<int32_t> spatialIndices;spatialIndices.SetGlobalBuffer((__gm__ int32_t*)workspace+6*4096+14*1056);
        // A data-selected genuine witness packet supplies the initial upper bound.
        // Search initializes and finalizes once after stable compaction.
        // C220 CompareScalar count form requires complete 64-float repeats.
        // Both temporary input arenas have 64 words; duplicate the last center
        // index in unused lanes so the first packed byte covers all six centers.
        auto center=work_[10112],threshold=work_[10160];
        auto broadcast=work_[11008],centerValues=work_[11072];
        auto centerOffsets=work_[11136].template ReinterpretCast<int32_t>();
        WholeReduceMin(record_,compactP_,32,6,8,1,4,ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax(record_[64],compactP_,32,6,8,1,4,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Add(center,record_,record_[64],48);PipeBarrier<PIPE_V>();
        Muls(center,center,0.5F,48);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // Use a valid geometric center without a scalar safety branch.
        // Recompute all radii and bounds from these same snapped coordinates.
        Duplicate(center[48],0.0F,16);PipeBarrier<PIPE_V>();
        Abs(SpatialPruneW(0),center,64);PipeBarrier<PIPE_V>();
        CompareScalar(mask_,SpatialPruneW(0),9.5367431640625e-7F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
        Select(center,mask_,center,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
#else
        SpatialPruneIndices(centerOffsets,0,64);PipeBarrier<PIPE_V>();
        Mins(centerOffsets,centerOffsets,int32_t(5),64);PipeBarrier<PIPE_V>();
        Muls(centerOffsets,centerOffsets,int32_t(32),64);PipeBarrier<PIPE_V>();
        Gather(centerValues,center,centerOffsets.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
        Abs(centerValues,centerValues,64);PipeBarrier<PIPE_V>();
        CompareScalar(mask_,centerValues,9.5367431640625e-7F,CMPMODE::LT,64);
        CompareScalar(smallMask_,centerValues,0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if((mask_.GetValue(0)&smallMask_.GetValue(0))!=0) {return COEFFICIENTS;}
#endif
        Duplicate(SpatialPruneW(4),0.0F,64);Duplicate(SpatialPruneW(3),0.0F,64);PipeBarrier<PIPE_V>();
        // Axis-major pairs describe disjoint halves of the same real query tile.
        // Broadcast slots are temporary until the shared witness gather begins.
        Brcb(broadcast,center,6,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
        Sub<float,false>(SpatialPruneW(0),compactP_,broadcast,uint64_t(64),6,BinaryRepeatParams{1,1,0,8,4,8});
        Abs<float,false>(SpatialPruneW(1),compactP_,uint64_t(64),6,UnaryRepeatParams{1,1,8,4});PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
        Abs<float,false>(broadcast,broadcast,uint64_t(64),6,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
        Add<float,false>(SpatialPruneW(1),SpatialPruneW(1),broadcast,uint64_t(64),6,BinaryRepeatParams{1,1,0,8,8,8});PipeBarrier<PIPE_V>();
        for(uint32_t axis=0;axis<3;++axis) {
            MulAddDst<float,float,false>(SpatialPruneW(4),SpatialPruneW(0)[axis*128],SpatialPruneW(0)[axis*128],uint64_t(64),2,BinaryRepeatParams{1,1,1,4,8,8});
            MulAddDst<float,float,false>(SpatialPruneW(3),SpatialPruneW(1)[axis*128],SpatialPruneW(1)[axis*128],uint64_t(64),2,BinaryRepeatParams{1,1,1,4,8,8});PipeBarrier<PIPE_V>();
        }
        // Radius covers the exact input coordinates around the rounded center.
        Muls(SpatialPruneW(3),SpatialPruneW(3),1.5258847270160913e-5F,64);PipeBarrier<PIPE_V>();
        Add(SpatialPruneW(4),SpatialPruneW(4),SpatialPruneW(3),64);PipeBarrier<PIPE_V>();
        Maxs(SpatialPruneW(4),SpatialPruneW(4),0.0F,64);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(4),SpatialPruneW(4),1.0000152587890625F,64);PipeBarrier<PIPE_V>();
        // This 64-word slot is dead until coefficient compaction begins.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        auto queryRadius=work_[18432];
#else
        auto queryRadius=work_[10176];
#endif
        Sqrt(queryRadius,SpatialPruneW(4),64);PipeBarrier<PIPE_V>();
        auto savedCenterScores=work_[12288],otherCenterScores=work_[13312];
        // The radius phase has finished reading the temporary broadcasts.
        // Both center coordinates and error envelopes are shared across all segments.
        auto centerBroadcast=work_[11008],centerError=work_[15360];
        auto centerMagnitude=work_[15424],axisMaximum=work_[15552];
        auto axisEnvelope=work_[15616],centerErrorBroadcast=work_[15744];
        Brcb(centerBroadcast,center,6,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
        Abs<float,false>(centerMagnitude,centerBroadcast,uint64_t(64),6,UnaryRepeatParams{1,1,1,8});
        Duplicate(centerError,0.0F,16);PipeBarrier<PIPE_V>();
        for(uint32_t axis=0;axis<3;++axis) {
            Brcb(axisMaximum,SpatialStateC(axis)[CAPACITY+16],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
            Add<float,false>(axisEnvelope,axisMaximum,centerMagnitude[axis*16],uint64_t(64),2,BinaryRepeatParams{1,0,1,1,0,1});PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(centerError,axisEnvelope,axisEnvelope,uint64_t(64),2,BinaryRepeatParams{1,1,1,1,1,1});PipeBarrier<PIPE_V>();
        }
        Muls(centerError,centerError,1.5258847270160913e-5F,16);PipeBarrier<PIPE_V>();
        Brcb(centerErrorBroadcast,centerError,2,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // Each plane stores two independent 1024-lane centers plus 32 guard words.
        // Odd half-bank plane padding matches the retained scratch bank phase.
        // The old center scalars are dead; the live query radius uses a separate slot.
        const BinaryRepeatParams centerSame{1,1,1,8,8,8};
        const UnaryRepeatParams centerUnary{1,1,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Sub<float,false>(work_[0],centerBroadcast[0],SpatialStateC(0),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[1024],centerBroadcast[64],SpatialStateC(0),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[2080],centerBroadcast[128],SpatialStateC(1),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[3104],centerBroadcast[192],SpatialStateC(1),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[4160],centerBroadcast[256],SpatialStateC(2),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[5184],centerBroadcast[320],SpatialStateC(2),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        PipeBarrier<PIPE_V>();
        Mul<float,false>(work_[8320],work_[0],work_[0],uint64_t(64),COEFFICIENTS/64,centerSame);
        Mul<float,false>(work_[6240],work_[0],SpatialStateC(6),uint64_t(64),COEFFICIENTS/64,centerSame);
        Mul<float,false>(work_[9344],work_[1024],work_[1024],uint64_t(64),COEFFICIENTS/64,centerSame);
        Mul<float,false>(work_[7264],work_[1024],SpatialStateC(6),uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(work_[8320],work_[2080],work_[2080],uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[6240],work_[2080],SpatialStateC(7),uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[9344],work_[3104],work_[3104],uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[7264],work_[3104],SpatialStateC(7),uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(work_[8320],work_[4160],work_[4160],uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[9344],work_[5184],work_[5184],uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        FusedMulAdd<float,false>(work_[4160],SpatialStateC(8),work_[6240],uint64_t(64),COEFFICIENTS/64,centerSame);
        FusedMulAdd<float,false>(work_[5184],SpatialStateC(8),work_[7264],uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        Maxs<float,false>(work_[6240],work_[4160],0.0F,uint64_t(64),COEFFICIENTS/64,centerUnary);
        Maxs<float,false>(work_[7264],work_[5184],0.0F,uint64_t(64),COEFFICIENTS/64,centerUnary);
        PipeBarrier<PIPE_V>();
        Min<float,false>(work_[6240],work_[6240],SpatialStateC(9),uint64_t(64),COEFFICIENTS/64,centerSame);
        Min<float,false>(work_[7264],work_[7264],SpatialStateC(9),uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        FusedMulAdd<float,false>(work_[4160],SpatialStateC(4)[CAPACITY+8],work_[6240],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        FusedMulAdd<float,false>(work_[5184],SpatialStateC(4)[CAPACITY+8],work_[7264],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(work_[8320],work_[6240],work_[4160],uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[9344],work_[7264],work_[5184],uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        Sub<float,false>(work_[8320],work_[8320],centerErrorBroadcast[0],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});
        Sub<float,false>(work_[9344],work_[9344],centerErrorBroadcast[64],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});
        PipeBarrier<PIPE_V>();
        Copy<float,false>(work_[12288],work_[8320],uint64_t(64),COEFFICIENTS/64,CopyRepeatParams(1,1,8,8));
        Copy<float,false>(work_[13312],work_[9344],uint64_t(64),COEFFICIENTS/64,CopyRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
#else
        // Both centers retain the original conservative score and arithmetic order.
        for(uint32_t sphere=0;sphere<2;++sphere) {
            // Center lower scores select genuine witnesses and filter candidates.
            // Center-to-segment scores use the retained13-instruction dot/norm order.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            for(uint32_t axis=0;axis<3;++axis)
                Sub<float,false>(SpatialPruneW(axis),centerBroadcast[(2*axis+sphere)*64],SpatialStateC(axis),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
            PipeBarrier<PIPE_V>();
            Mul(SpatialPruneW(4),SpatialPruneW(0),SpatialPruneW(0),COEFFICIENTS);Mul(SpatialPruneW(3),SpatialPruneW(0),SpatialStateC(6),COEFFICIENTS);PipeBarrier<PIPE_V>();
            MulAddDst(SpatialPruneW(4),SpatialPruneW(1),SpatialPruneW(1),COEFFICIENTS);MulAddDst(SpatialPruneW(3),SpatialPruneW(1),SpatialStateC(7),COEFFICIENTS);PipeBarrier<PIPE_V>();
            MulAddDst(SpatialPruneW(4),SpatialPruneW(2),SpatialPruneW(2),COEFFICIENTS);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            FusedMulAdd<float,false>(SpatialPruneW(2),SpatialStateC(8),SpatialPruneW(3),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Maxs(SpatialPruneW(3),SpatialPruneW(2),0.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();Min(SpatialPruneW(3),SpatialPruneW(3),SpatialStateC(9),COEFFICIENTS);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            FusedMulAdd<float,false>(SpatialPruneW(2),SpatialStateC(4)[CAPACITY+8],SpatialPruneW(3),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            MulAddDst(SpatialPruneW(4),SpatialPruneW(3),SpatialPruneW(2),COEFFICIENTS);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Sub<float,false>(SpatialPruneW(4),SpatialPruneW(4),centerErrorBroadcast[sphere*64],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            Copy(work_[12288+sphere*1024],SpatialPruneW(4),64,COEFFICIENTS/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        }
#endif
        // Genuine witnesses are selected once for both groups.
        Min(SpatialPruneW(4),savedCenterScores,otherCenterScores,COEFFICIENTS);PipeBarrier<PIPE_V>();
        auto predicate=work_[10048].template ReinterpretCast<uint8_t>();
        // Distance to any real segment is 1-Lipschitz in the query point.
        // Padded coefficient records cannot seed the genuine witness packet.
        auto realCenterScores=SpatialPruneW(4);
        if(segments<COEFFICIENTS) {
            SpatialPruneIndices(SpatialPruneW(0).template ReinterpretCast<int32_t>(),0,COEFFICIENTS);PipeBarrier<PIPE_V>();
            Cast(SpatialPruneW(1),SpatialPruneW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,COEFFICIENTS);PipeBarrier<PIPE_V>();
            CompareScalar(predicate,SpatialPruneW(1),float(int32_t(segments)),CMPMODE::LT,COEFFICIENTS);PipeBarrier<PIPE_V>();
            Select(SpatialPruneW(1),predicate,SpatialPruneW(4),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,COEFFICIENTS);PipeBarrier<PIPE_V>();
            realCenterScores=SpatialPruneW(1);
        }
        // One value/index reduction selects the16 genuine witnesses.
        WholeReduceMin(record_,realCenterScores,64,COEFFICIENTS/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
        // Reuse the original16 genuine value/index pairs at record_[0..31].
        // The witness pairs remain live until their indices are gathered.
        auto witnessPairs=record_;
        SpatialPruneIndices(SpatialPruneW(0).template ReinterpretCast<int32_t>(),0,16);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(0).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),int32_t(8),16);PipeBarrier<PIPE_V>();
        Adds(SpatialPruneW(0).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),int32_t(4),16);PipeBarrier<PIPE_V>();
        Gather(SpatialPruneW(1).template ReinterpretCast<int32_t>(),witnessPairs.template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<uint32_t>(),0U,16);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(0).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),int32_t(8),16);PipeBarrier<PIPE_V>();
        Adds(SpatialPruneW(0).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),int32_t(-32),16);PipeBarrier<PIPE_V>();
        Add(SpatialPruneW(1).template ReinterpretCast<int32_t>(),SpatialPruneW(1).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),16);PipeBarrier<PIPE_V>();
        Mins(SpatialPruneW(1).template ReinterpretCast<int32_t>(),SpatialPruneW(1).template ReinterpretCast<int32_t>(),int32_t(segments-1U),16);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(1).template ReinterpretCast<int32_t>(),SpatialPruneW(1).template ReinterpretCast<int32_t>(),int32_t(4),16);PipeBarrier<PIPE_V>();
        auto rawWitness=work_[16384],witness=work_[11264];
        // Seven genuine coefficient planes; each scalar is expanded to8 lanes.
        auto witnessOffsets=work_[16512].template ReinterpretCast<int32_t>();
        for(uint32_t col=0;col<7;++col) {
            const uint32_t plane=col<3?col:col+3;
            const int32_t planeBytes=int32_t(uint32_t(SpatialStateC(plane).GetPhyAddr())-uint32_t(SpatialStateC(0).GetPhyAddr()));
            Adds(witnessOffsets[col*16],SpatialPruneW(1).template ReinterpretCast<int32_t>(),planeBytes,16);
        }
        PipeBarrier<PIPE_V>();
        Gather(rawWitness,SpatialStateC(0),witnessOffsets.template ReinterpretCast<uint32_t>(),0U,112);
        PipeBarrier<PIPE_V>();
        for(uint32_t col=0;col<7;++col)
            Brcb(witness[col*128],rawWitness[col*16],2,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        auto witnessError=work_[14336];
        constexpr uint64_t witnessLanes=64;
        // Matrix rows are16 witnesses; columns are all64 real queries.
        const BinaryRepeatParams witnessSame{1,1,1,8,8,8};
        const BinaryRepeatParams witnessSegment{1,1,0,8,8,1};
        const BinaryRepeatParams witnessPoint{1,1,0,8,0,1};
        const UnaryRepeatParams witnessUnary{1,1,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(witnessLanes);
        Sub<float,false>(SpatialPruneW(0),compactP_[0*64],witness[0*128],witnessLanes,16,witnessPoint);
        Sub<float,false>(SpatialPruneW(1),compactP_[1*64],witness[1*128],witnessLanes,16,witnessPoint);
        Mul<float,false>(SpatialPruneW(4),SpatialPruneW(0),SpatialPruneW(0),witnessLanes,16,witnessSame);
        Mul<float,false>(SpatialPruneW(3),SpatialPruneW(0),witness[3*128],witnessLanes,16,witnessSegment);
        Sub<float,false>(SpatialPruneW(2),compactP_[2*64],witness[2*128],witnessLanes,16,witnessPoint);
        MulAddDst<float,float,false>(SpatialPruneW(4),SpatialPruneW(1),SpatialPruneW(1),witnessLanes,16,witnessSame);
        MulAddDst<float,float,false>(SpatialPruneW(3),SpatialPruneW(1),witness[4*128],witnessLanes,16,witnessSegment);
        MulAddDst<float,float,false>(SpatialPruneW(4),SpatialPruneW(2),SpatialPruneW(2),witnessLanes,16,witnessSame);
        FusedMulAdd<float,false>(SpatialPruneW(2),witness[5*128],SpatialPruneW(3),witnessLanes,16,BinaryRepeatParams{1,0,1,8,1,8});
        Maxs<float,false>(SpatialPruneW(3),SpatialPruneW(2),0.0F,witnessLanes,16,witnessUnary);
        Min<float,false>(SpatialPruneW(3),SpatialPruneW(3),witness[6*128],witnessLanes,16,witnessSegment);
        Muls<float,false>(witnessError,SpatialPruneW(4),1.52587890625e-5F,witnessLanes,16,witnessUnary);
        FusedMulAdd<float,false>(SpatialPruneW(2),SpatialStateC(4)[CAPACITY+8],SpatialPruneW(3),witnessLanes,16,BinaryRepeatParams{1,0,1,8,0,8});
        MulAddDst<float,float,false>(SpatialPruneW(4),SpatialPruneW(3),SpatialPruneW(2),witnessLanes,16,witnessSame);PipeBarrier<PIPE_V>();
        Maxs<float,false>(SpatialPruneW(4),SpatialPruneW(4),0.0F,witnessLanes,16,witnessUnary);PipeBarrier<PIPE_V>();
        Muls<float,false>(witnessError,witnessError,2.25F,witnessLanes,16,witnessUnary);PipeBarrier<PIPE_V>();
        Add<float,false>(SpatialPruneW(4),SpatialPruneW(4),witnessError,witnessLanes,16,witnessSame);PipeBarrier<PIPE_V>();
        // Fold witness rows while retaining the64 query coordinates.
        Min<float,false>(SpatialPruneW(4),SpatialPruneW(4),SpatialPruneW(4)[512],witnessLanes,8,witnessSame);PipeBarrier<PIPE_V>();
        Min<float,false>(SpatialPruneW(4),SpatialPruneW(4),SpatialPruneW(4)[256],witnessLanes,4,witnessSame);PipeBarrier<PIPE_V>();
        Min<float,false>(SpatialPruneW(4),SpatialPruneW(4),SpatialPruneW(4)[128],witnessLanes,2,witnessSame);PipeBarrier<PIPE_V>();
        Min<float,false>(tileHi_,SpatialPruneW(4),SpatialPruneW(4)[64],witnessLanes,1,witnessSame);PipeBarrier<PIPE_V>();
        Maxs(tileHi_,tileHi_,0.0F,64);PipeBarrier<PIPE_V>();
        Sqrt(tileHi_,tileHi_,64);PipeBarrier<PIPE_V>();
        Add(tileHi_,tileHi_,queryRadius,64);PipeBarrier<PIPE_V>();
        Muls(tileHi_,tileHi_,1.0000152587890625F,64);PipeBarrier<PIPE_V>();
        Mul(tileHi_,tileHi_,tileHi_,64);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,tileHi_,32,2,8,1,4,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Copy(threshold,record_,16,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
        Brcb(broadcast,threshold,2,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        auto otherPredicate=work_[10080].template ReinterpretCast<uint8_t>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Compare<float,uint8_t,false>(predicate,savedCenterScores,broadcast,CMPMODE::LE,uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});
        Compare<float,uint8_t,false>(otherPredicate,otherCenterScores,broadcast[64],CMPMODE::LE,uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
        Or(predicate.template ReinterpretCast<uint16_t>(),predicate.template ReinterpretCast<uint16_t>(),otherPredicate.template ReinterpretCast<uint16_t>(),COEFFICIENTS/16);PipeBarrier<PIPE_V>();
        // No ordinal coefficient is forced into the surviving set.
        SpatialPruneIndices(SpatialPruneW(0).template ReinterpretCast<int32_t>(),0,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Cast(SpatialPruneW(1),SpatialPruneW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,COEFFICIENTS);PipeBarrier<PIPE_V>();
        auto genuinePredicate=work_[10080].template ReinterpretCast<uint8_t>();
        CompareScalar(genuinePredicate,SpatialPruneW(1),float(int32_t(segments)),CMPMODE::LT,COEFFICIENTS);PipeBarrier<PIPE_V>();
        And(predicate.template ReinterpretCast<uint16_t>(),predicate.template ReinterpretCast<uint16_t>(),genuinePredicate.template ReinterpretCast<uint16_t>(),COEFFICIENTS/16);PipeBarrier<PIPE_V>();
        auto packed=work_[8000];auto offsets=work_[9024].template ReinterpretCast<int32_t>();
        uint64_t kept=0;
        GatherMask(packed,SpatialPruneW(1),predicate.template ReinterpretCast<uint32_t>(),true,COEFFICIENTS,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        if(kept==0 || kept>=960) {return COEFFICIENTS;}
        logicalSegments=uint32_t(kept);
        // Every padded coefficient duplicates the final genuine retained segment.
        // Stable packing preserves the original argmin tie order.
        Mins(SpatialPruneW(1),SpatialPruneW(1),float(int32_t(kept-1)),COEFFICIENTS);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(1),SpatialPruneW(1),4.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Cast(offsets,SpatialPruneW(1),RoundMode::CAST_RINT,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Gather(SpatialPruneW(0),packed,offsets.template ReinterpretCast<uint32_t>(),0U,COEFFICIENTS);PipeBarrier<PIPE_V>();
        auto originalIndices=packed.template ReinterpretCast<int32_t>();
        Cast(originalIndices,SpatialPruneW(0),RoundMode::CAST_RINT,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Muls(offsets,originalIndices,int32_t(4),COEFFICIENTS);PipeBarrier<PIPE_V>();
        // Preserve the inverse map before reusing the dead packed-ID arena.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // The inverse map survives search/refinement above their UB arenas.
        const LocalTensor<int32_t> pendingIndices(TPosition::VECCALC,192512U,1024U);
        Copy(pendingIndices,originalIndices,64,(uint32_t(kept)+63U)/64U,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
#else
        auto pendingIndices=work_[5*(64*64+128)].template ReinterpretCast<int32_t>();
        Copy(pendingIndices,originalIndices,64,COEFFICIENTS/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
#endif
        // Every live search chunk is gathered once. The remaining payload
        // duplicates the final retained coefficient exactly, as before.
        const uint32_t compacted=(uint32_t(kept)+63U)/64U*64U;
        for(uint32_t col=0;col<14;++col) {
            const uint32_t stage=col<8?col*1024U:10240U+(col-8U)*1024U;
            Gather(work_[stage],SpatialStateC(col),offsets.template ReinterpretCast<uint32_t>(),0U,compacted);
        }
        PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        for(uint32_t col=0;col<14;++col) {
            const uint32_t stage=col<8?col*1024U:10240U+(col-8U)*1024U;
            Copy(SpatialStateC(col),work_[stage],64,compacted/64U,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
#else
        for(uint32_t col=0;col<14;++col) {
            const uint32_t stage=col<8?col*1024U:10240U+(col-8U)*1024U;
            Brcb(work_[16384U+col*64U],work_[stage+compacted-8U],1,BrcbRepeatParams(1,8));
            Copy(SpatialStateC(col),work_[stage],64,compacted/64U,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t col=0;col<14;++col)
            Copy(SpatialStateC(col)[compacted],work_[16384U+col*64U+56U],64,(COEFFICIENTS-compacted)/64U,CopyRepeatParams(1,0,8,0));
        PipeBarrier<PIPE_V>();
#endif
#if !defined(DTYPE_TREE_POINTS) || !TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(spatialIndices,pendingIndices,COEFFICIENTS);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
#endif
        pruned=true;
        return (uint32_t(kept)+63)/64*64;
    }

#endif
    __aicore__ inline void MakeIndices(const LocalTensor<int32_t>& values, int32_t first, uint32_t count) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE || (CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL)) {
            // The following masked additions consume the complete seed packet.
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(8));
            Duplicate<int32_t,false>(values,first,uint64_t(64),1,1,8);
            PipeBarrier<PIPE_V>();
        } else
#endif
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(sizeof(T)==4) {
            // The promoted arena may contain sorted pairs. Initialize all eight
            // seeds and finish the write before the masked read-modify-writes.
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(8));
            Duplicate<int32_t,false>(values,first,uint64_t(64),1,1,8);
            PipeBarrier<PIPE_V>();
        } else
#endif
        Duplicate(values, first, 8);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        if constexpr(CAPACITY==256 && sizeof(T)==4)PipeBarrier<PIPE_V>();
#endif
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
    __aicore__ __attribute__((always_inline)) inline void MaterializeDirectionLow() {
        if constexpr(DEFER_DIRECTION_LOW) {
            if(directionLowReady_)return;
            // Coarse and selected scratch is dead here; cache once per batch.
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub(FewWork(axis),C(axis+3),C(axis),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub(FewWork(axis+3),C(axis+3),FewWork(axis),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Add(FewWork(axis+6),FewWork(axis),FewWork(axis+3),COEFFICIENTS);
                Sub(C(axis+10),FewWork(axis+3),C(axis),COEFFICIENTS);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub(FewWork(axis+6),C(axis+3),FewWork(axis+6),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Add(C(axis+10),C(axis+10),FewWork(axis+6),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
            directionLowReady_=true;
        }
    }

    __aicore__ __attribute__((noinline,aligned(128))) static void LoadPrefixResidentSegments(
        GM_ADDR source,uint32_t segmentCount) {
        static_assert(NATIVE_FLOAT_PREFIX_SPARSE,"Fixed native Float prefix arena required");
        AscendC::ICachePreLoad(int64_t(2));
        constexpr uint32_t pitch=CAPACITY+32;
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*pitch+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        // Reconstruct only the handles read by the unchanged coefficient loader.
        // The local object never escapes this function, so its fields fold to constants.
        KernelTreeBatch op;
        op.m_=CAPACITY;op.cp_=pitch;op.wp_=pitch;op.pp_=pointPitch;op.count_=COEFFICIENTS;
        op.t_.segments=segmentCount;
        op.sg_.SetGlobalBuffer((__gm__ T*)source);
        op.columns_=LocalTensor<float>(TPosition::VECCALC,0,14*pitch);
        op.work_=LocalTensor<float>(TPosition::VECCALC,workBase*4,6*(ROWS*64+128));
        op.sf_=op.work_[6*CAPACITY];op.rawS_=op.sf_.template ReinterpretCast<T>();
        op.so_=op.work_[13*CAPACITY].template ReinterpretCast<int32_t>();op.ones_=op.work_[12*CAPACITY];
        op.rowBase_=LocalTensor<int32_t>(TPosition::VECCALC,(stateBase+128+16+12*64)*4,64);
        op.mask_=LocalTensor<uint8_t>(TPosition::VECCALC,(stateBase+128+16+16*64)*4,ROWS*8);
        op.LoadSegments(0,false);
    }

    __aicore__ __attribute__((noinline,aligned(128))) static void LoadBoundedResidentSegments(
        GM_ADDR source,uint32_t segmentCount,bool initialInputs) {
        static_assert(DEFER_DIRECTION_LOW && ROWS==64,"Fixed native Float bounded arena required");
        constexpr uint32_t pitch=CAPACITY+32;
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*pitch+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        KernelTreeBatch op;
        op.m_=CAPACITY;op.cp_=pitch;op.wp_=pitch;op.pp_=pointPitch;op.count_=COEFFICIENTS;
        op.t_.segments=segmentCount;
        op.sg_.SetGlobalBuffer((__gm__ T*)source);
        op.columns_=LocalTensor<float>(TPosition::VECCALC,0,14*pitch);
        op.work_=LocalTensor<float>(TPosition::VECCALC,workBase*4,6*(ROWS*64+128));
        op.sf_=op.work_[6*CAPACITY];op.rawS_=op.sf_.template ReinterpretCast<T>();
        op.ones_=op.work_[12*CAPACITY];
        op.so_=LocalTensor<int32_t>(TPosition::VECCALC,(stateBase+128+16+16*64+ROWS*4+8)*4,COEFFICIENTS);
        op.rowBase_=LocalTensor<int32_t>(TPosition::VECCALC,(stateBase+128+16+12*64)*4,64);
        op.mask_=LocalTensor<uint8_t>(TPosition::VECCALC,(stateBase+128+16+16*64)*4,ROWS*8);
        op.LoadSegments(0,initialInputs);
    }

    __aicore__ __attribute__((noinline,aligned(128))) void LoadResidentSegments(uint32_t batch,bool initialInputs) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)AscendC::ICachePreLoad(int64_t(2));
        static_assert(DEFER_DIRECTION_LOW || NATIVE_FLOAT_PREFIX_SPARSE,"Resident native Float cache required");
        LoadSegments(batch,initialInputs);
    }
    __aicore__ inline void LoadSegments(uint32_t batch,bool initialInputs=false) {
        if constexpr(DEFER_DIRECTION_LOW)directionLowReady_=false;
        if constexpr(!(CAPACITY==256 && sizeof(T)==4)) {
            ShiftRight(so_,rowBase_,int32_t(8),64);PipeBarrier<PIPE_V>();
            for(uint32_t filled=64;filled<COEFFICIENTS;filled*=2) {
                const uint32_t n=COEFFICIENTS-filled<filled?COEFFICIENTS-filled:filled;
                Adds(so_[filled],so_,int32_t(filled),n);PipeBarrier<PIPE_V>();
            }
            Muls(so_,so_,int32_t(24),COEFFICIENTS);
        }
        if(!initialInputs) {
            Duplicate(rawS_, T(0), COEFFICIENTS * 6);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(rawS_, sg_[batch * t_.segments * 6],
                        DataCopyExtParams{1, t_.segments * 6 * uint32_t(sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<T>{false, 0, 0, T(0)});
            PipeBarrier<PIPE_ALL>();
        } else PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == 2) Cast(sf_, rawS_, RoundMode::CAST_NONE, COEFFICIENTS * 6);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 6; ++axis)
            Gather(C(axis), sf_, so_.template ReinterpretCast<uint32_t>(), axis * 4, count_);
        PipeBarrier<PIPE_V>();
        if constexpr (true) {
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
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis)
                Select(W(axis), keep, C(axis), C(axis+3), SELMODE::VSEL_TENSOR_TENSOR_MODE, count_);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis)
                Select(C(axis+3), keep, C(axis+3), C(axis), SELMODE::VSEL_TENSOR_TENSOR_MODE, count_);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Adds(C(axis), W(axis), 0.0F, count_);
            PipeBarrier<PIPE_V>();
        }
        if constexpr(DEFER_DIRECTION_LOW) {
            constexpr uint64_t lanes=64;
            constexpr uint8_t repeats=COEFFICIENTS/64;
            const BinaryRepeatParams same{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(lanes);
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub<float,false>(C(axis+6),C(axis+3),C(axis),lanes,repeats,same);
            PipeBarrier<PIPE_V>();
            Mul<float,false>(C(9),C(6),C(6),lanes,repeats,same);
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(C(9),C(7),C(7),lanes,repeats,same);
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(C(9),C(8),C(8),lanes,repeats,same);
            PipeBarrier<PIPE_V>();
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub(C(axis+6), C(axis+3), C(axis), count_);
        PipeBarrier<PIPE_V>();
        if constexpr(!DEFER_DIRECTION_LOW) {
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
        }
        Mul(C(9), C(6), C(6), count_);
        PipeBarrier<PIPE_V>();
        MulAddDst(C(9), C(7), C(7), count_);
        PipeBarrier<PIPE_V>();
        MulAddDst(C(9), C(8), C(8), count_);
        PipeBarrier<PIPE_V>();
        }
        CompareScalar(mask_, C(9), 0.0F, CMPMODE::GT, count_);
        PipeBarrier<PIPE_V>();
        Select(C(9), mask_, C(9), 1.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, count_);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE || DEFER_DIRECTION_LOW)PipeBarrier<PIPE_V>();
#endif
        NormalizeDirections();
        if constexpr((CLEAN_PRIMARY64 && sizeof(T)==2) || NATIVE_HALF_PREFIX_PRIMARY) {
            Abs(W(0),C(10),COEFFICIENTS);
            Abs(W(1),C(11),COEFFICIENTS);
            Abs(W(2),C(12),COEFFICIENTS);PipeBarrier<PIPE_V>();
            Max(W(0),W(0),W(1),COEFFICIENTS);PipeBarrier<PIPE_V>();
            Max(W(0),W(0),W(2),COEFFICIENTS);PipeBarrier<PIPE_V>();
            WholeReduceMax(record_,W(0),64,COEFFICIENTS/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMax(tileSecond_,record_,COEFFICIENTS/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            exactDirectionLow_=tileSecond_.template ReinterpretCast<uint32_t>().GetValue(0)==0;
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void NormalizeDirections() {
        if constexpr(PREFIX_NORMALIZATION) {
            Adds(C(13),C(9),0.0F,COEFFICIENTS);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE || DEFER_DIRECTION_LOW)PipeBarrier<PIPE_V>();
#endif
            Sqrt(C(9),C(9),COEFFICIENTS);
            Duplicate(ones_,1.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();
            Div(W(0),ones_,C(9),COEFFICIENTS);PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Mul(C(axis+6),C(axis+6),W(0),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
        } else {
        Adds(W(0),C(9),0.0F,COEFFICIENTS);
        Sqrt(C(9),C(9),COEFFICIENTS);
        Duplicate(ones_,1.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Div(C(13),ones_,C(9),COEFFICIENTS);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Mul(C(axis+6),C(axis+6),C(13),COEFFICIENTS);
        PipeBarrier<PIPE_V>();
        Adds(C(13),W(0),0.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrimaryW(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        return LocalTensor<float>(TPosition::VECCALC,(workBase+k*96)*4,96);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrimaryC(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t refinedBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192;
        return LocalTensor<float>(TPosition::VECCALC,(refinedBase+k*64)*4,64);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrimaryP(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t pointBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024;
        return LocalTensor<float>(TPosition::VECCALC,(pointBase+k*64)*4,64);
    }
    template<uint32_t WIDTH=64,bool OMIT_DIRECTION_LOW=false>
    __aicore__ __attribute__((always_inline)) static inline void PrimaryCompensatedBody() {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<uint8_t> mask_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4)*4,ROWS*8);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        constexpr bool DIRECT_LENGTH=WIDTH==64 && (CLEAN_PRIMARY64 || NATIVE_HALF_PREFIX_PRIMARY);
        constexpr uint8_t repeats=1;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        if constexpr(DIRECT_LENGTH) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub<float,false>(PrimaryW(axis+9),PrimaryC(axis+3),PrimaryC(axis),uint64_t(64),repeats,binary);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub(PrimaryW(axis+9),PrimaryC(axis+3),PrimaryC(axis),WIDTH);
            PipeBarrier<PIPE_V>();
            Adds(PrimaryW(12),PrimaryC(9),0.0F,WIDTH);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(WIDTH));
        }
        if constexpr(DEFER_DIRECTION_LOW) {
            // Independent exact subtractions share producer/consumer waits.
            // W13..18 is dead coarse scratch, disjoint from points and columns.
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Sub<float,false>(PrimaryW(axis),PrimaryP(axis),PrimaryC(axis),uint64_t(64),repeats,binary);
                Sub<float,false>(PrimaryW(axis+16),PrimaryC(axis+3),PrimaryW(axis+9),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Sub<float,false>(PrimaryW(axis+3),PrimaryP(axis),PrimaryW(axis),uint64_t(64),repeats,binary);
                Add<float,false>(PrimaryW(axis+13),PrimaryW(axis+9),PrimaryW(axis+16),uint64_t(64),repeats,binary);
                Sub<float,false>(PrimaryC(axis+10),PrimaryW(axis+16),PrimaryC(axis),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Add<float,false>(PrimaryW(axis+6),PrimaryW(axis),PrimaryW(axis+3),uint64_t(64),repeats,binary);
                Sub<float,false>(PrimaryW(axis+3),PrimaryW(axis+3),PrimaryC(axis),uint64_t(64),repeats,binary);
                Sub<float,false>(PrimaryW(axis+13),PrimaryC(axis+3),PrimaryW(axis+13),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Sub<float,false>(PrimaryW(axis+6),PrimaryP(axis),PrimaryW(axis+6),uint64_t(64),repeats,binary);
                Add<float,false>(PrimaryC(axis+10),PrimaryC(axis+10),PrimaryW(axis+13),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Add<float,false>(PrimaryW(axis+3),PrimaryW(axis+3),PrimaryW(axis+6),uint64_t(64),repeats,binary);
            PipeBarrier<PIPE_V>();
        } else {
            // Error-free subtraction preserves short distances at large coordinates.
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(PrimaryW(axis), PrimaryP(axis), PrimaryC(axis), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(PrimaryW(axis+3), PrimaryP(axis), PrimaryW(axis), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) {
                Add<float, false>(PrimaryW(axis+6), PrimaryW(axis), PrimaryW(axis+3), uint64_t(64), repeats, binary);
                Sub<float, false>(PrimaryW(axis+3), PrimaryW(axis+3), PrimaryC(axis), uint64_t(64), repeats, binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(PrimaryW(axis+6), PrimaryP(axis), PrimaryW(axis+6), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(PrimaryW(axis+3), PrimaryW(axis+3), PrimaryW(axis+6), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
        }
        Mul<float, false>(PrimaryW(8), PrimaryW(0), PrimaryW(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(PrimaryW(8), PrimaryW(1), PrimaryW(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(PrimaryW(8), PrimaryW(2), PrimaryW(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(DIRECT_LENGTH)
            Div<float, false>(PrimaryW(8), PrimaryW(8), PrimaryC(9), uint64_t(64), repeats, binary);
        else
            Div<float, false>(PrimaryW(8), PrimaryW(8), PrimaryW(12), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(PrimaryW(8), PrimaryW(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(PrimaryW(8), PrimaryW(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        if constexpr(WIDTH<64)SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        CompareScalar<float, uint8_t, false>(mask_, PrimaryW(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        if constexpr(WIDTH<64)SetVectorMask<float,MaskMode::NORMAL>(uint64_t(WIDTH));
        Muls<float, false>(PrimaryW(8), PrimaryW(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(PrimaryW(axis), PrimaryW(8), PrimaryW(axis+9), uint64_t(64), repeats, binary);
            if constexpr(!OMIT_DIRECTION_LOW)
                MulAddDst<float, float, false>(PrimaryW(axis+3), PrimaryW(8), PrimaryC(axis+10), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(PrimaryW(axis), PrimaryW(axis), PrimaryW(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(PrimaryW(axis+3), PrimaryP(axis), PrimaryC(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        SetCmpMask(mask_);
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(PrimaryW(axis),PrimaryW(axis),PrimaryW(axis+3),1,binary);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(PrimaryW(0), PrimaryW(0), PrimaryW(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(PrimaryW(0), PrimaryW(1), PrimaryW(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(PrimaryW(0), PrimaryW(2), PrimaryW(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(WIDTH<64)SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        if constexpr(!DIRECT_LENGTH) {
            Select<float, uint8_t, false>(PrimaryW(0), valid_, PrimaryW(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ __attribute__((noinline,aligned(128))) static void PrimaryCompensated16() {
        PrimaryCompensatedBody<16>();
    }
    __aicore__ __attribute__((noinline,aligned(128))) static void PrimaryCompensated32() {
        if constexpr(DEFER_DIRECTION_LOW)AscendC::ICachePreLoad(int64_t(1));
        PrimaryCompensatedBody<32>();
    }
    __aicore__ __attribute__((noinline,aligned(128))) static void PrimaryCompensatedFull() {
        if constexpr(DEFER_DIRECTION_LOW)AscendC::ICachePreLoad(int64_t(1));
        PrimaryCompensatedBody<64>();
    }
    __aicore__ __attribute__((noinline,aligned(128))) static void PrimaryCompensatedExactDirection() {
        PrimaryCompensatedBody<64,true>();
    }
    template<uint32_t WIDTH=64>
    __aicore__ __attribute__((always_inline)) static inline void PrimaryCompensated() {
        PrimaryCompensatedBody<WIDTH>();
    }

    __aicore__ __attribute__((noinline,aligned(128))) static void PrefixPrimaryCompensatedAxes() {
        static_assert(NATIVE_FLOAT_PREFIX_SPARSE && ROWS==64,"Native prefix primary resource required");
        AscendC::ICachePreLoad(int64_t(1));
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        const LocalTensor<uint8_t> mask(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4)*4,ROWS*8);
        // This arena's primary has cp=pp=64, wp=96 and one64-lane packet.
        // Twelve-block work pitch keeps each axis'32-word gap untouched.
        const BinaryRepeatParams workCoefficient{1,1,1,12,8,8};
        const BinaryRepeatParams workPointWork{1,1,1,12,8,12};
        const BinaryRepeatParams workWorkCoefficient{1,1,1,12,12,8};
        const BinaryRepeatParams workAxes{1,1,1,12,12,12};
        const BinaryRepeatParams projectAxes{1,1,1,12,0,12};
        const BinaryRepeatParams projectLowAxes{1,1,1,12,0,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Sub<float,false>(PrimaryW(9),PrimaryC(3),PrimaryC(0),uint64_t(64),3,workCoefficient);
        PipeBarrier<PIPE_V>();
        Adds(PrimaryW(12),PrimaryC(9),0.0F,64);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Sub<float,false>(PrimaryW(0),PrimaryP(0),PrimaryC(0),uint64_t(64),3,workCoefficient);
        PipeBarrier<PIPE_V>();
        Sub<float,false>(PrimaryW(3),PrimaryP(0),PrimaryW(0),uint64_t(64),3,workPointWork);
        PipeBarrier<PIPE_V>();
        Add<float,false>(PrimaryW(6),PrimaryW(0),PrimaryW(3),uint64_t(64),3,workAxes);
        Sub<float,false>(PrimaryW(3),PrimaryW(3),PrimaryC(0),uint64_t(64),3,workWorkCoefficient);
        PipeBarrier<PIPE_V>();
        Sub<float,false>(PrimaryW(6),PrimaryP(0),PrimaryW(6),uint64_t(64),3,workPointWork);
        PipeBarrier<PIPE_V>();
        Add<float,false>(PrimaryW(3),PrimaryW(3),PrimaryW(6),uint64_t(64),3,workAxes);
        PipeBarrier<PIPE_V>();
        Mul<float,false>(PrimaryW(8),PrimaryW(0),PrimaryW(9),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(8),PrimaryW(1),PrimaryW(10),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(8),PrimaryW(2),PrimaryW(11),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Div<float,false>(PrimaryW(8),PrimaryW(8),PrimaryW(12),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Maxs<float,false>(PrimaryW(8),PrimaryW(8),0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        Mins<float,false>(PrimaryW(8),PrimaryW(8),1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(mask,PrimaryW(8),1.0F,CMPMODE::LT,uint64_t(64),1,unary);
        Muls<float,false>(PrimaryW(8),PrimaryW(8),-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(0),PrimaryW(8),PrimaryW(9),uint64_t(64),3,projectAxes);
        MulAddDst<float,float,false>(PrimaryW(3),PrimaryW(8),PrimaryC(10),uint64_t(64),3,projectLowAxes);
        PipeBarrier<PIPE_V>();
        Add<float,false>(PrimaryW(0),PrimaryW(0),PrimaryW(3),uint64_t(64),3,workAxes);PipeBarrier<PIPE_V>();
        Sub<float,false>(PrimaryW(3),PrimaryP(0),PrimaryC(3),uint64_t(64),3,workCoefficient);PipeBarrier<PIPE_V>();
        SetCmpMask(mask);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(PrimaryW(0),PrimaryW(0),PrimaryW(3),3,workAxes);PipeBarrier<PIPE_V>();
        Mul<float,false>(PrimaryW(0),PrimaryW(0),PrimaryW(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(0),PrimaryW(1),PrimaryW(1),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(0),PrimaryW(2),PrimaryW(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
    }

    template<bool CORRECT_ZERO=false>
    __aicore__ __attribute__((always_inline)) inline void Compensated() {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE && !CORRECT_ZERO) {
            PrefixPrimaryCompensatedAxes();return;
        }
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub(W(axis+9),C(axis+3),C(axis),count_);
        PipeBarrier<PIPE_V>();
        if constexpr(CORRECT_ZERO) {
            Adds(W(12),C(13),0.0F,count_);PipeBarrier<PIPE_V>();
        } else {
            Adds(W(12),C(9),0.0F,count_);PipeBarrier<PIPE_V>();
        }
        const uint8_t repeats = count_/64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(64));
        // Error-free subtraction preserves short distances at large coordinates.
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis), P(axis), C(axis), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+3), P(axis), W(axis), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            Add<float, false>(W(axis+6), W(axis), W(axis+3), uint64_t(64), repeats, binary);
            Sub<float, false>(W(axis+3), W(axis+3), C(axis), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+6), P(axis), W(axis+6), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(W(axis+3), W(axis+3), W(axis+6), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(W(8), W(0), W(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(8), W(1), W(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(8), W(2), W(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Div<float, false>(W(8), W(8), W(12), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(W(8), W(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(W(8), W(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        CompareScalar<float, uint8_t, false>(mask_, W(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(W(8), W(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(W(axis), W(8), W(axis+9), uint64_t(64), repeats, binary);
            MulAddDst<float, float, false>(W(axis+3), W(8), C(axis+10), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(W(axis), W(axis), W(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+3), P(axis), C(axis+3), uint64_t(64), repeats, binary);
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
        Mul<float, false>(W(0), W(0), W(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(0), W(1), W(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(0), W(2), W(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t, false>(W(0), valid_, W(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(CORRECT_ZERO) {
        WholeReduceMax(W(7),W(12),64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,W(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(record_,record_,1.8189894035458565e-12F,1);
        WholeReduceMin(W(7),W(0),64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(record_[64],W(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Compare(smallMask_,record_[64],record_,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(smallMask_.GetValue(0)&1U) {
            CorrectAxialZeroScores(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),pp_,
                uint32_t(columns_.GetPhyAddr()),cp_,uint32_t(W(1).GetPhyAddr()),wp_,uint32_t(mask_.GetPhyAddr()),
                (count_+255)/256*32,count_);
            CorrectCollinearZeroScores<sizeof(T)==2>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),pp_,
                uint32_t(columns_.GetPhyAddr()),cp_,uint32_t(work_[20*1056].GetPhyAddr()),uint32_t(record_.GetPhyAddr()),
                uint32_t(mask_.GetPhyAddr()),count_);
            Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);PipeBarrier<PIPE_V>();
        }
        }
    }

    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FewWork(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*(CAPACITY+32))*4,6*(ROWS*64+128)-k*(CAPACITY+32));
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FewCoefficients(uint32_t k) {
        constexpr uint32_t pitch=CAPACITY+32;
        const uint32_t offset=k>=6 && k<=8?6*pitch+(k-6)*CAPACITY:k*pitch;
        return LocalTensor<float>(TPosition::VECCALC,offset*4,k>=6 && k<=8?CAPACITY:pitch);
    }
    __aicore__ __attribute__((noinline,aligned(128))) inline bool CertifyWideScore(uint32_t candidates) {
        auto finite=W(8).template ReinterpretCast<uint8_t>();
        auto other=finite[128],possible=finite[256];
        CompareScalar(finite,W(0),1152921504606846976.0F,CMPMODE::LE,candidates);
        CompareScalar(other,C(13),1152921504606846976.0F,CMPMODE::LE,candidates);PipeBarrier<PIPE_V>();
        And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        CompareScalar(other,W(0),0.0F,CMPMODE::GE,candidates);PipeBarrier<PIPE_V>();
        And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        CompareScalar(other,C(13),0.0F,CMPMODE::GE,candidates);PipeBarrier<PIPE_V>();
        And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        Not(possible.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        And(possible.template ReinterpretCast<uint16_t>(),possible.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for(uint32_t k=0;k<candidates/64;++k)
            if(possible.template ReinterpretCast<uint64_t>().GetValue(k)!=0)return false;
        const uint32_t group=uint32_t(record_.template ReinterpretCast<int32_t>().GetValue(1));
        const uint32_t lane=uint32_t(offset_.GetValue(0));
        if(group>=candidates/64 || lane>=64)return false;
        Select(W(1),valid_,W(0),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,candidates);
        Select(W(2),valid_,C(13),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,candidates);PipeBarrier<PIPE_V>();
        Mul(W(3),W(1),W(2),candidates);PipeBarrier<PIPE_V>();
        Sqrt(W(3),W(3),candidates);PipeBarrier<PIPE_V>();
        Muls(W(3),W(3),4.76837158203125e-7F,candidates);
        Muls(W(4),W(1),9.5367431640625e-7F,candidates);
        Muls(W(5),W(2),9.094947017729282e-13F,candidates);PipeBarrier<PIPE_V>();
        Add(W(3),W(3),W(4),candidates);PipeBarrier<PIPE_V>();
        Add(W(3),W(3),W(5),candidates);PipeBarrier<PIPE_V>();
        Muls(W(3),W(3),1.0001220703125F,candidates);PipeBarrier<PIPE_V>();
        Adds(W(3),W(3),1.0e-30F,candidates);PipeBarrier<PIPE_V>();
        offset_.SetValue(0,int32_t((group*64+lane)*4));
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Gather(record_[8],W(3),offset_.template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
        Add(record_[16],record_,record_[8],1);PipeBarrier<PIPE_V>();
        Brcb(W(4),record_[16],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Brcb(W(5),W(4),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(W(4),W(5),64,candidates/64,CopyRepeatParams(1,0,8,0));
        Sub(W(3),W(1),W(3),candidates);PipeBarrier<PIPE_V>();
        Compare(possible,W(3),W(4),CMPMODE::LE,candidates);PipeBarrier<PIPE_V>();
        And(possible.template ReinterpretCast<uint16_t>(),possible.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        uint32_t count=0;
        for(uint32_t k=0;k<candidates/64;++k)count+=uint32_t(__builtin_popcountll(possible.template ReinterpretCast<uint64_t>().GetValue(k)));
        return count==1;
    }
    __aicore__ __attribute__((always_inline)) static inline uint32_t CertifyRefinedBody(
        const LocalTensor<float>& minima,uint64_t validLanes) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)AscendC::ICachePreLoad(int64_t(1));
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<uint8_t> valid(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        auto finite=FewWork(8).template ReinterpretCast<uint8_t>();
        auto normFinite=finite[32];auto possible=finite[64];
        Select(FewWork(1),valid,FixedFewW(0),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        Select(FewWork(2),valid,FixedFewC(13),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        CompareScalar(finite,FewWork(1),1152921504606846976.0F,CMPMODE::LE,64);
        CompareScalar(normFinite,FewWork(2),1152921504606846976.0F,CMPMODE::LE,64);
        PipeBarrier<PIPE_V>();
        And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),normFinite.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        if constexpr(DEFER_DIRECTION_LOW) {
            Muls(FewWork(5),FewWork(2),1.8189894035458565e-12F,64);PipeBarrier<PIPE_V>();
            Compare(normFinite,FewWork(1),FewWork(5),CMPMODE::GT,64);PipeBarrier<PIPE_V>();
            And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),normFinite.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            if(validLanes==1) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                return uint32_t(finite.template ReinterpretCast<uint64_t>().GetValue(0)&1U);
            }
            // sqrt(s*n) <= (s+n)/2. Both constants exceed the old
            // coefficients after this substitution, including float rounding slack.
            Muls(FewWork(3),FewWork(1),1.25e-6F,64);
            Muls(FewWork(5),FewWork(2),2.5e-7F,64);PipeBarrier<PIPE_V>();
            Add(FewWork(3),FewWork(3),FewWork(5),64);PipeBarrier<PIPE_V>();
        } else {
        Mul(FewWork(3),FewWork(1),FewWork(2),64);PipeBarrier<PIPE_V>();
        Sqrt(FewWork(3),FewWork(3),64);PipeBarrier<PIPE_V>();
        Muls(FewWork(3),FewWork(3),4.76837158203125e-7F,64);
        Muls(FewWork(4),FewWork(1),9.5367431640625e-7F,64);
        Muls(FewWork(5),FewWork(2),9.094947017729282e-13F,64);PipeBarrier<PIPE_V>();
        Add(FewWork(3),FewWork(3),FewWork(4),64);PipeBarrier<PIPE_V>();
        Add(FewWork(3),FewWork(3),FewWork(5),64);PipeBarrier<PIPE_V>();
        }
        WholeReduceMax(FewWork(5),FewWork(3),64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Brcb(FewWork(6),FewWork(5),1,BrcbRepeatParams(1,8));
        Brcb(FewWork(4),minima,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        if constexpr(CAPACITY==256) {
            Add<float,true>(FewWork(4),FewWork(4),FewWork(6),uint64_t(64),uint8_t(1),BinaryRepeatParams(1,1,0,8,8,0));
        } else {
            Brcb(FewWork(7),FewWork(6),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Add(FewWork(4),FewWork(4),FewWork(7),64);
        }
        Sub(FewWork(3),FewWork(1),FewWork(3),64);PipeBarrier<PIPE_V>();
        Compare(possible,FewWork(3),FewWork(4),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        And(possible.template ReinterpretCast<uint16_t>(),possible.template ReinterpretCast<uint16_t>(),valid.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t finiteBits=finite.template ReinterpretCast<uint64_t>().GetValue(0);
        const uint64_t possibleBits=possible.template ReinterpretCast<uint64_t>().GetValue(0);
        if constexpr(CAPACITY==256) {
            if((validLanes>>8)==0) {
                const uint64_t selected=possibleBits&validLanes;
                return (finiteBits&validLanes)==validLanes && selected && !(selected&(selected-1)) ? 1U : 0U;
            }
        }
        uint32_t result=0;
        for(uint32_t group=0;group<8;++group) {
            const uint32_t active=uint32_t((validLanes>>(8*group))&255U);
            const uint32_t finiteGroup=uint32_t((finiteBits>>(8*group))&255U);
            const uint32_t candidates=uint32_t((possibleBits>>(8*group))&255U);
            if(active && (finiteGroup&active)==active && candidates && !(candidates&(candidates-1U)))
                result|=uint32_t(1)<<group;
        }
        return result;
    }
    __aicore__ __attribute__((noinline,aligned(128))) static uint32_t CertifyRefined(
        const LocalTensor<float>& minima,uint64_t validLanes) {
        return CertifyRefinedBody(minima,validLanes);
    }
    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineFewStatic(uint32_t row,uint32_t chunks,uint64_t& uniqueRows) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        const LocalTensor<float> chunkLower_(TPosition::VECCALC,(workBase+6*(ROWS*64+128)+64+192)*4,1024);
        const LocalTensor<float> compactP_(TPosition::VECCALC,(stateBase-13*64-192)*4,192);
        const LocalTensor<float> segmentIndices_(TPosition::VECCALC,(workBase+6*(ROWS*64+128))*4,64);
        const LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,(stateBase+128+16+14*64)*4,64);
        const LocalTensor<int32_t> offset_(TPosition::VECCALC,(stateBase+128+16+11*64)*4,64);
        const LocalTensor<float> record_(TPosition::VECCALC,stateBase*4,128);
        const LocalTensor<float> record2_(TPosition::VECCALC,(stateBase+128)*4,16);
        const LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,(stateBase+128+16+10*64)*4,64);
        const LocalTensor<int32_t> idx_(TPosition::VECCALC,(stateBase+128+16+3*64)*4,64);
        const LocalTensor<float> tileSecond_(TPosition::VECCALC,(stateBase+128+16+8*64)*4,64);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+16*64+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,(stateBase+128+16+16*64+ROWS*4)*4,32);
        // Each cached set contains its chunk's true minimizer; chunk bounds
        // exclude only chunks that cannot improve the incumbent.
        Muls(offset_,outputOffset_,int32_t(32),8);PipeBarrier<PIPE_V>();
        Adds(offset_,offset_,int32_t(2048+row*4),8);PipeBarrier<PIPE_V>();
        Gather(valid_.template ReinterpretCast<uint32_t>(),chunkLower_.template ReinterpretCast<uint32_t>(),
            offset_.template ReinterpretCast<uint32_t>(),0U,8);
        Duplicate(FewWork(6),0.0F,8);PipeBarrier<PIPE_V>();
        uint32_t wordBits=chunks&15U;
        wordBits=(wordBits|(wordBits<<2))&0x33U;
        wordBits=(wordBits|(wordBits<<1))&0x55U;
        wordBits|=wordBits<<1;
        SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t((~wordBits)&255U));
        Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(0),uint64_t(64),1,1,8);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        #pragma unroll
        for(uint32_t chunk=0;chunk<COEFFICIENTS/64;++chunk)
            Adds(FewWork(7)[chunk*64],segmentIndices_,float(int32_t(chunk*64)),64);
        PipeBarrier<PIPE_V>();
        uint64_t kept=0;
        GatherMask(FewWork(6),FewWork(7),valid_.template ReinterpretCast<uint32_t>(),true,COEFFICIENTS,
            GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        if(kept==0 || kept>8)return false;
        Copy(record2_,FewWork(6),8,1,CopyRepeatParams(1,1,1,1));
        Cast(offset_,FewWork(6),RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),8);PipeBarrier<PIPE_V>();
        auto selectedColumns=FewWork(0)[16*1056];
        #pragma unroll
        for(uint32_t col=0;col<14;++col)if(col<6||col==13||(col>9&&!DEFER_DIRECTION_LOW))
            Gather(selectedColumns[col*128],FewCoefficients(col),offset_.template ReinterpretCast<uint32_t>(),0U,8);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Brcb(FewWork(0),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Copy(FixedFewP(axis),FewWork(0)[(row%8)*8],8,1,CopyRepeatParams(1,0,1,0));PipeBarrier<PIPE_V>();
        }
        Duplicate(valid_.template ReinterpretCast<uint32_t>(),uint32_t(0),8);PipeBarrier<PIPE_V>();
        Duplicate(valid_.template ReinterpretCast<uint32_t>(),uint32_t((uint32_t(1)<<uint32_t(kept))-1),1);PipeBarrier<PIPE_V>();
        bool accepted;
        if constexpr(sizeof(T)==4 && CAPACITY==256
#ifdef DTYPE_TREE_POINTS
            && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) accepted=FixedFewInlineBody<true>();
        else accepted=FixedFewCompensated();
        if(accepted) {
            // Interval certification below resolves ambiguous candidates exactly.
            WholeReduceMin(record_,FixedFewW(0),8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
            Gather(offset_,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(4),1);PipeBarrier<PIPE_V>();
            Gather(tileIdx_.template ReinterpretCast<float>(),record2_,offset_.template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
            Cast(tileIdx_,tileIdx_.template ReinterpretCast<float>(),RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
        }
        if(!accepted)return false;
        bool unique;
        if constexpr(DEFER_DIRECTION_LOW)unique=(CertifyRefinedBody(record_,(uint64_t(1)<<kept)-1)&1U)!=0;
        else unique=kept==1 || (CertifyRefined(record_,(uint64_t(1)<<kept)-1)&1U);
#if defined(DTYPE_TREE_POINTS)
        if(!unique)unique=ResolveExactSingle(uint32_t(kept),record2_,record_,tileIdx_,offset_,outputOffset_);
#endif
        if(unique)uniqueRows|=uint64_t(1)<<row;
        Brcb(FewWork(0),record_,1,BrcbRepeatParams(1,8));
        Brcb(FewWork(1),tileIdx_.template ReinterpretCast<float>(),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(FewWork(2),FewWork(0),64,1,CopyRepeatParams(1,0,8,0));
        Copy(FewWork(3),FewWork(1),64,1,CopyRepeatParams(1,0,8,0));
        CompareScalar(smallMask_,segmentIndices_,float(int32_t(row)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetCmpMask(smallMask_);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,FewWork(2),tileSecond_,1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),FewWork(3),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        return true;
    }

    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SparseW(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*160)*4,160);
    }
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SparseC(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch+16*1056;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*128)*4,128);
    }
    __aicore__ __attribute__((always_inline)) static inline uint32_t PredicatePopulation(uint64_t bits) {
        return uint32_t(__builtin_popcountll(bits));
    }
    __aicore__ __attribute__((always_inline)) static inline uint64_t SavedChunkPredicate(uint32_t row,uint32_t chunk) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        // Post-search snapshot lies above generic W0..12 and below selected columns.
        // Primary/few/zero-correction helpers and later fallback rows do not write it.
        constexpr uint32_t snapshot=workBase+14*1056;
        const uint32_t group=chunk/4,slot=chunk%4;
        const LocalTensor<uint64_t> saved(TPosition::VECCALC,(snapshot+group*512+slot*128)*4+row*8,1);
        return saved.GetValue(0);
    }
    template<bool OVERLAPPING_CHUNKS>
    __aicore__ __attribute__((always_inline)) static inline bool RefineFewCandidates(uint32_t row,uint32_t chunks,uint64_t& uniqueRows) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> compactP_(TPosition::VECCALC,(stateBase-13*64-192)*4,192);
        const LocalTensor<float> segmentIndices_(TPosition::VECCALC,(workBase+6*(ROWS*64+128))*4,64);
        const LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,(stateBase+128+16+14*64)*4,64);
        const LocalTensor<int32_t> offset_(TPosition::VECCALC,(stateBase+128+16+11*64)*4,64);
        const LocalTensor<float> record_(TPosition::VECCALC,stateBase*4,128);
        const LocalTensor<float> record2_(TPosition::VECCALC,(stateBase+128)*4,16);
        const LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,(stateBase+128+16+10*64)*4,64);
        const LocalTensor<int32_t> idx_(TPosition::VECCALC,(stateBase+128+16+3*64)*4,64);
        const LocalTensor<uint32_t> cert(TPosition::VECCALC,(stateBase+128+16+4*64)*4,64);
        const LocalTensor<uint32_t> certHigh(TPosition::VECCALC,(stateBase+128+16+5*64)*4,64);
        const LocalTensor<float> tileSecond_(TPosition::VECCALC,(stateBase+128+16+8*64)*4,64);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*4)*4,32);
        uint64_t kept;
        if constexpr(OVERLAPPING_CHUNKS) {
        // Each saved chunk predicate includes every possible chunk minimizer.
        // Chunk bounds exclude only chunks incapable of improving the winner.
        if(chunks==0 || (chunks & (chunks-1))==0)return false;
        const uint32_t chosenChunk=uint32_t(idx_.GetValue(row))/64;
        if(!(chunks&(uint32_t(1)<<chosenChunk)))return false;
        uint32_t candidates=0;
        for(uint32_t chunk=0;chunk<COEFFICIENTS/64;++chunk)if(chunks&(uint32_t(1)<<chunk)) {
            candidates+=PredicatePopulation(SavedChunkPredicate(row,chunk));
            if(candidates>8)return false;
        }
        if(candidates==0)return false;
        Duplicate(FewWork(6),0.0F,8);
        Duplicate(valid_.template ReinterpretCast<uint32_t>(),uint32_t(0),32);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        for(uint32_t chunk=0;chunk<COEFFICIENTS/64;++chunk)if(chunks&(uint32_t(1)<<chunk)) {
            const uint64_t bits=SavedChunkPredicate(row,chunk);
            valid_.template ReinterpretCast<uint32_t>().SetValue(chunk*2,uint32_t(bits));
            valid_.template ReinterpretCast<uint32_t>().SetValue(chunk*2+1,uint32_t(bits>>32));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        #pragma unroll
        for(uint32_t chunk=0;chunk<COEFFICIENTS/64;++chunk)
            Adds(FewWork(7)[chunk*64],segmentIndices_,float(int32_t(chunk*64)),64);
        PipeBarrier<PIPE_V>();
        kept=0;
        GatherMask(FewWork(6),FewWork(7),valid_.template ReinterpretCast<uint32_t>(),true,COEFFICIENTS,
            GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        if(kept==0 || kept>8)return false;
        } else {
        // The saved winning predicate contains every possible minimizer in
        // that chunk. A single overlapping chunk permits the same precise
        // refinement on only its compacted candidates.
        if(chunks==0 || (chunks & (chunks-1))!=0)return false;
        const uint32_t chosenChunk=uint32_t(idx_.GetValue(row))/64;
        if(chunks!=(uint32_t(1)<<chosenChunk))return false;
        const uint32_t low=cert.GetValue(row),high=certHigh.GetValue(row);
        Duplicate(valid_.template ReinterpretCast<uint32_t>(),low,2);
        Duplicate(FewWork(6),0.0F,8);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(2));
        Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),high,uint64_t(64),1,1,8);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();
        kept=0;
        GatherMask(FewWork(6),segmentIndices_,valid_.template ReinterpretCast<uint32_t>(),true,64,
            GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        if(kept==0 || kept>8)return false;
        Adds(FewWork(6),FewWork(6),float(int32_t(chosenChunk*64)),8);PipeBarrier<PIPE_V>();
        }
        Copy(record2_,FewWork(6),8,1,CopyRepeatParams(1,1,1,1));
        Cast(offset_,FewWork(6),RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),8);PipeBarrier<PIPE_V>();
        auto selectedColumns=FewWork(0)[16*1056];
        #pragma unroll
        for(uint32_t col=0;col<14;++col)if(col<6||col>9)
            Gather(selectedColumns[col*128],FewCoefficients(col),offset_.template ReinterpretCast<uint32_t>(),0U,8);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Brcb(FewWork(axis),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Copy(FixedFewP(axis),FewWork(axis)[(row%8)*8],8,1,CopyRepeatParams(1,0,1,0));
        PipeBarrier<PIPE_V>();
        CompareScalar(valid_,segmentIndices_,float(int32_t(kept)),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
        const bool accepted=FixedFewCompensated();
        if(accepted) {
            WholeReduceMin(record_,FixedFewW(0),8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
            Gather(offset_,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(4),1);PipeBarrier<PIPE_V>();
            Gather(tileIdx_.template ReinterpretCast<float>(),record2_,offset_.template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
            Cast(tileIdx_,tileIdx_.template ReinterpretCast<float>(),RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
        }
        if(!accepted)return false;
        bool unique=kept==1 || (CertifyRefined(record_,(uint64_t(1)<<kept)-1)&1U);
#if defined(DTYPE_TREE_POINTS)
        if(!unique)unique=ResolveExactSingle(uint32_t(kept),record2_,record_,tileIdx_,offset_,outputOffset_);
#endif
        if(unique)uniqueRows|=uint64_t(1)<<row;
        Brcb(FewWork(0),record_,1,BrcbRepeatParams(1,8));
        Brcb(FewWork(1),tileIdx_.template ReinterpretCast<float>(),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(FewWork(2),FewWork(0),64,1,CopyRepeatParams(1,0,8,0));
        Copy(FewWork(3),FewWork(1),64,1,CopyRepeatParams(1,0,8,0));
        CompareScalar(smallMask_,segmentIndices_,float(int32_t(row)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetCmpMask(smallMask_);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,FewWork(2),tileSecond_,1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),FewWork(3),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        return true;
    }
    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineOverlappingFew(uint32_t row,uint32_t chunks,uint64_t& uniqueRows) {
        return RefineFewCandidates<true>(row,chunks,uniqueRows);
    }

    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineWinningFew(uint32_t row,uint32_t chunks,uint64_t& uniqueRows) {
        return RefineFewCandidates<false>(row,chunks,uniqueRows);
    }

    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FixedFewW(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*160)*4,160);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FixedFewC(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch+16*1056;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*128)*4,128);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FixedFewP(uint32_t k) {
        constexpr uint32_t pitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32);
        return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,pitch);
    }

    __aicore__ __attribute__((noinline,aligned(128))) static bool BatchFewCompensated() {
        constexpr uint32_t COUNT=64;
        constexpr bool REJECT_ZERO=true;
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> record(TPosition::VECCALC,stateBase*4,128);
        const LocalTensor<uint8_t> mask(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4)*4,ROWS*8);
        const LocalTensor<uint8_t> valid(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*4)*4,32);
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub(FixedFewW(axis+9),FixedFewC(axis+3),FixedFewC(axis),COUNT);
        PipeBarrier<PIPE_V>();
        Adds(FixedFewW(12),FixedFewC(13),0.0F,COUNT);PipeBarrier<PIPE_V>();
        const uint8_t repeats = (COUNT+63)/64;
        constexpr uint32_t lanes=COUNT<64?COUNT:64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        auto axesW0=FixedFewW(0);axesW0.SetSize(3*160);
        auto axesW3=FixedFewW(3);axesW3.SetSize(3*160);
        auto axesW6=FixedFewW(6);axesW6.SetSize(3*160);
        auto axesW9=FixedFewW(9);axesW9.SetSize(3*160);
        auto axesC0=FixedFewC(0);axesC0.SetSize(3*128);
        auto axesC3=FixedFewC(3);axesC3.SetSize(3*128);
        auto axesC10=FixedFewC(10);axesC10.SetSize(3*128);
        auto axesP0=FixedFewP(0);axesP0.SetSize(3*pointPitch);
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(lanes));
        // Error-free subtraction preserves short distances at large coordinates.
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Sub<float,false>(axesW0,axesP0,axesC0,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,pointPitch/8,16));
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis), FixedFewP(axis), FixedFewC(axis), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Sub<float,false>(axesW3,axesP0,axesW0,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,pointPitch/8,20));
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+3), FixedFewP(axis), FixedFewW(axis), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Add<float,false>(axesW6,axesW0,axesW3,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,20,20));
            Sub<float,false>(axesW3,axesW3,axesC0,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,20,16));
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            Add<float, false>(FixedFewW(axis+6), FixedFewW(axis), FixedFewW(axis+3), uint64_t(64), repeats, binary);
            Sub<float, false>(FixedFewW(axis+3), FixedFewW(axis+3), FixedFewC(axis), uint64_t(64), repeats, binary);
        }
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Sub<float,false>(axesW6,axesP0,axesW6,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,pointPitch/8,20));
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+6), FixedFewP(axis), FixedFewW(axis+6), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Add<float,false>(axesW3,axesW3,axesW6,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,20,20));
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(FixedFewW(axis+3), FixedFewW(axis+3), FixedFewW(axis+6), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        Mul<float, false>(FixedFewW(8), FixedFewW(0), FixedFewW(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(8), FixedFewW(1), FixedFewW(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(8), FixedFewW(2), FixedFewW(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Div<float, false>(FixedFewW(8), FixedFewW(8), FixedFewW(12), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(FixedFewW(8), FixedFewW(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(FixedFewW(8), FixedFewW(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        CompareScalar<float, uint8_t, false>(mask, FixedFewW(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(FixedFewW(8), FixedFewW(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            MulAddDst<float,float,false>(axesW0,FixedFewW(8),axesW9,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,0,20));
            MulAddDst<float,float,false>(axesW3,FixedFewW(8),axesC10,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,0,16));
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(FixedFewW(axis), FixedFewW(8), FixedFewW(axis+9), uint64_t(64), repeats, binary);
            MulAddDst<float, float, false>(FixedFewW(axis+3), FixedFewW(8), FixedFewC(axis+10), uint64_t(64), repeats, binary);
        }
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Add<float,false>(axesW0,axesW0,axesW3,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,20,20));
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(FixedFewW(axis), FixedFewW(axis), FixedFewW(axis+3), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Sub<float,false>(axesW3,axesP0,axesC3,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,pointPitch/8,16));
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+3), FixedFewP(axis), FixedFewC(axis+3), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(COUNT<=64) {
            SetCmpMask(mask);
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(FixedFewW(axis),FixedFewW(axis),FixedFewW(axis+3),1,binary);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,uint8_t,false>(FixedFewW(axis),mask,FixedFewW(axis),FixedFewW(axis+3),SELMODE::VSEL_TENSOR_TENSOR_MODE,uint64_t(64),repeats,binary);
        }
        PipeBarrier<PIPE_V>();
        Mul<float, false>(FixedFewW(0), FixedFewW(0), FixedFewW(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(0), FixedFewW(1), FixedFewW(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(0), FixedFewW(2), FixedFewW(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t, false>(FixedFewW(0), valid, FixedFewW(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(true) {
        if constexpr(NATIVE_HALF_PREFIX_PRIMARY
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
                     || (sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024)
#endif
                     ) {
            // COUNT is64, hence repeats is1. A second one-element reduction
            // preserves the first result and needs no separate instruction.
            WholeReduceMax(record,FixedFewW(12),lanes,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            Muls(record,record,1.8189894035458565e-12F,1);
            WholeReduceMin(record[64],FixedFewW(0),lanes,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        } else {
        WholeReduceMax(FixedFewW(7),FixedFewW(12),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record,FixedFewW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(record,record,1.8189894035458565e-12F,1);
        WholeReduceMin(FixedFewW(7),FixedFewW(0),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(record[64],FixedFewW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        }
        Compare(smallMask,record[64],record,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(smallMask.GetValue(0)&1U) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            return false;
        }
        }
        if constexpr(COUNT<64) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));}
        return true;
    }

    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> BatchIds() {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch+13*1056;
        return LocalTensor<float>(TPosition::VECCALC,base*4,64);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<uint32_t> BatchPointIds() {
        return LocalTensor<uint32_t>(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr())+64*4,8);
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<uint32_t> BatchPointMap() {
        return LocalTensor<uint32_t>(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr())+96*4,64);
    }
    // Dead candidate-refinement scratch. No search coefficients, points, row state,
    // candidate IDs or output records overlap these 65 vector planes.
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> EndpointF(uint32_t k) {
        return FewWork(0)[20*1056+k*64];
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<int32_t> EndpointI(uint32_t k) {
        return EndpointF(k).template ReinterpretCast<int32_t>();
    }
    // Normalize one signed base-2^15 digit; carry is an arithmetic right shift.
    __aicore__ __attribute__((always_inline)) static inline void EndpointSplit(uint32_t digit) {
        ShiftRight(EndpointI(30),EndpointI(31),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(EndpointI(32),EndpointI(30),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(EndpointI(digit),EndpointI(31),EndpointI(32),64);PipeBarrier<PIPE_V>();
    }
    // Every finite binary16 coordinate times 2^24 is an exact signed integer.
    // A high part and a signed low part avoid any out-of-range int32 cast.
    __aicore__ __attribute__((always_inline)) static inline void EndpointEncode(LocalTensor<float> x,uint32_t dst) {
        Muls(EndpointF(34),x,0.0078125F,64);PipeBarrier<PIPE_V>();
        Cast(EndpointI(33),EndpointF(34),RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
        Cast(EndpointF(34),EndpointI(33),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(EndpointF(34),EndpointF(34),128.0F,64);PipeBarrier<PIPE_V>();
        Sub(EndpointF(35),x,EndpointF(34),64);PipeBarrier<PIPE_V>();
        Muls(EndpointF(35),EndpointF(35),16777216.0F,64);PipeBarrier<PIPE_V>();
        Cast(EndpointI(31),EndpointF(35),RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
        ShiftRight(EndpointI(36),EndpointI(31),int32_t(30),64);PipeBarrier<PIPE_V>();
        EndpointSplit(dst);
        Copy(EndpointI(31),EndpointI(30),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        EndpointSplit(dst+1);
        Muls(EndpointI(dst+2),EndpointI(33),int32_t(2),64);PipeBarrier<PIPE_V>();
        Add(EndpointI(dst+2),EndpointI(dst+2),EndpointI(36),64);PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((noinline,aligned(128))) static bool CorrectHalfLatticePacket(
        uint64_t validLanes,uint64_t near,LocalTensor<int32_t> outputOffset,uint32_t maskAddress) {
        uint64_t count=near-((near>>1)&uint64_t(0x5555555555555555ULL));
        count=(count&uint64_t(0x3333333333333333ULL))+((count>>2)&uint64_t(0x3333333333333333ULL));
        count=(count+(count>>4))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
        if(!(count&uint64_t(0x0e0e0e0e0e0e0e0eULL)))return false;
        uint64_t packets=0;
        for(uint32_t group=0;group<8;++group) {
            const uint32_t bits=uint32_t(near>>(group*8))&255U;
            if(bits && (bits&(bits-1U)))packets|=uint64_t(255)<<(group*8);
        }
        const LocalTensor<uint8_t> masks(TPosition::VECCALC,maskAddress,256);
        const uint64_t active=validLanes&packets;
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        masks.template ReinterpretCast<uint64_t>().SetValue(0,active);
        masks[128].template ReinterpretCast<uint64_t>().SetValue(0,active);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Sub(EndpointF(axis),FixedFewP(axis),FixedFewC(axis),64);
            Sub(EndpointF(axis+3),FixedFewC(axis+3),FixedFewC(axis),64);
        }
        PipeBarrier<PIPE_V>();
        Abs(EndpointF(6),EndpointF(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t k=1;k<6;++k) {
            Abs(EndpointF(7),EndpointF(k),64);PipeBarrier<PIPE_V>();
            Max(EndpointF(6),EndpointF(6),EndpointF(7),64);PipeBarrier<PIPE_V>();
        }
        Select(EndpointF(6),masks,EndpointF(6),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        WholeReduceMax(EndpointF(8),EndpointF(6),64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const uint32_t maximumBits=EndpointI(8).template ReinterpretCast<uint32_t>().GetValue(0);
        if(maximumBits==0 || maximumBits>=0x7f800000U)return false;
        const int32_t exponent=int32_t((maximumBits>>23)&255U)-127;
        if(exponent< -24 || exponent>17)return false;
        union FloatPower {uint32_t bits;float value;};
        FloatPower factor,unitSquared;factor.bits=uint32_t(131-exponent)<<23;
        unitSquared.bits=uint32_t(127+2*(exponent-4))<<23;
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        // Validate the same nine raw coordinate planes. Predicates keep
        // independent aligned destinations; only element-wise arithmetic is batched.
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        auto scaled=EndpointF(9),selected=EndpointF(18),rounded=EndpointF(27);
        auto integers=EndpointI(0);
        auto points=FixedFewP(0),endpoints=FixedFewC(0);
        scaled.SetSize(9*64);selected.SetSize(9*64);rounded.SetSize(9*64);
        integers.SetSize(9*64);points.SetSize(3*pointPitch);endpoints.SetSize(6*128);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Muls<float,false>(scaled,points,factor.value,uint64_t(64),uint8_t(3),UnaryRepeatParams{1,1,8,pointPitch/8});
        Muls<float,false>(scaled[3*64],endpoints,factor.value,uint64_t(64),uint8_t(6),UnaryRepeatParams{1,1,8,16});
        PipeBarrier<PIPE_V>();
        // The later round-trip comparison rejects every value outside this range.
        // Limit only the cast input; accepted integer coordinates are unchanged.
        Mins(selected,scaled,1048576.0F,9*64);PipeBarrier<PIPE_V>();
        Maxs(selected,selected,-1048576.0F,9*64);PipeBarrier<PIPE_V>();
        Cast(integers,selected,RoundMode::CAST_TRUNC,9*64);PipeBarrier<PIPE_V>();
        Cast(rounded,integers,RoundMode::CAST_NONE,9*64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<9;++axis) {
            Compare(masks[64],rounded[axis*64],scaled[axis*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(masks[128].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Sub(EndpointI(axis),EndpointI(axis),EndpointI(axis+3),64);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Sub(EndpointI(axis+3),EndpointI(axis+6),EndpointI(axis+3),64);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<6;++axis) {
            Cast(EndpointF(9),EndpointI(axis),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(masks[32],EndpointF(9),-32.0F,CMPMODE::GE,64);
            CompareScalar(masks[64],EndpointF(9),32.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            And(masks[32].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(masks[128].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const uint64_t integral=masks[128].template ReinterpretCast<uint64_t>().GetValue(0);
        uint64_t admitted=0;
        for(uint32_t group=0;group<8;++group) {
            const uint64_t packet=uint64_t(255)<<(group*8);
            if((active&packet) && !(active&packet&~integral))admitted|=active&packet;
        }
        if(!admitted)return false;
        masks.template ReinterpretCast<uint64_t>().SetValue(0,admitted);
        masks[128].template ReinterpretCast<uint64_t>().SetValue(0,admitted);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        #pragma unroll
        for(uint32_t axis=0;axis<6;++axis)
            Select(EndpointF(axis),masks,EndpointF(axis),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        PipeBarrier<PIPE_V>();
        Mul(EndpointI(9),EndpointI(0),EndpointI(3),64);
        Mul(EndpointI(10),EndpointI(3),EndpointI(3),64);
        Mul(EndpointI(11),EndpointI(0),EndpointI(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=1;axis<3;++axis) {
            Mul(EndpointI(6),EndpointI(axis),EndpointI(axis+3),64);
            Mul(EndpointI(7),EndpointI(axis+3),EndpointI(axis+3),64);
            Mul(EndpointI(8),EndpointI(axis),EndpointI(axis),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(9),EndpointI(9),EndpointI(6),64);
            Add(EndpointI(10),EndpointI(10),EndpointI(7),64);
            Add(EndpointI(11),EndpointI(11),EndpointI(8),64);PipeBarrier<PIPE_V>();
        }
        Mul(EndpointI(12),EndpointI(11),EndpointI(10),64);
        Mul(EndpointI(13),EndpointI(9),EndpointI(9),64);PipeBarrier<PIPE_V>();
        Sub(EndpointI(12),EndpointI(12),EndpointI(13),64);PipeBarrier<PIPE_V>();
        Copy(EndpointI(13),EndpointI(10),64,1,CopyRepeatParams(1,1,8,8));
        Duplicate(EndpointI(16),int32_t(1),64);PipeBarrier<PIPE_V>();
        Cast(EndpointF(14),EndpointI(9),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(masks[32],EndpointF(14),0.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(12),masks[32],EndpointF(11),EndpointF(12),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        Select(EndpointF(13),masks[32],EndpointF(16),EndpointF(13),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Sub(EndpointI(15),EndpointI(axis),EndpointI(axis+3),64);PipeBarrier<PIPE_V>();
            Mul(EndpointI(6),EndpointI(15),EndpointI(15),64);PipeBarrier<PIPE_V>();
            if(axis==0)Copy(EndpointI(14),EndpointI(6),64,1,CopyRepeatParams(1,1,8,8));
            else Add(EndpointI(14),EndpointI(14),EndpointI(6),64);
            PipeBarrier<PIPE_V>();
        }
        Cast(EndpointF(15),EndpointI(9),RoundMode::CAST_NONE,64);
        Cast(EndpointF(6),EndpointI(10),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Compare(masks[32],EndpointF(15),EndpointF(6),CMPMODE::GE,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(12),masks[32],EndpointF(14),EndpointF(12),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        Select(EndpointF(13),masks[32],EndpointF(16),EndpointF(13),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(12),masks,EndpointF(12),EndpointF(16),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        Duplicate(EndpointI(17),int32_t(0),64);PipeBarrier<PIPE_V>();
        Select(EndpointF(13),masks,EndpointF(13),EndpointF(17),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        ShiftRight(EndpointI(18),EndpointI(12),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(EndpointI(19),EndpointI(18),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(EndpointI(19),EndpointI(12),EndpointI(19),64);PipeBarrier<PIPE_V>();
        // Compare N/D exactly. Base-2^15 cross-products fit two small int32 limbs.
        for(uint32_t lane=0;lane<8;++lane) {
            Muls(EndpointI(20),outputOffset,int32_t(4),8);PipeBarrier<PIPE_V>();
            Adds(EndpointI(20),EndpointI(20),int32_t(lane*4),8);PipeBarrier<PIPE_V>();
            Gather(EndpointF(21),EndpointF(18),EndpointI(20).template ReinterpretCast<uint32_t>(),0U,8);
            Gather(EndpointF(22),EndpointF(19),EndpointI(20).template ReinterpretCast<uint32_t>(),0U,8);
            Gather(EndpointF(23),EndpointF(13),EndpointI(20).template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Brcb(EndpointF(24),EndpointF(21),1,BrcbRepeatParams(1,8));
            Brcb(EndpointF(25),EndpointF(22),1,BrcbRepeatParams(1,8));
            Brcb(EndpointF(26),EndpointF(23),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Mul(EndpointI(27),EndpointI(19),EndpointI(26),64);
            Mul(EndpointI(28),EndpointI(25),EndpointI(13),64);PipeBarrier<PIPE_V>();
            ShiftRight(EndpointI(29),EndpointI(27),int32_t(15),64);
            ShiftRight(EndpointI(30),EndpointI(28),int32_t(15),64);
            Mul(EndpointI(31),EndpointI(18),EndpointI(26),64);
            Mul(EndpointI(32),EndpointI(24),EndpointI(13),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(29),64);
            Add(EndpointI(32),EndpointI(32),EndpointI(30),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(29),EndpointI(29),int32_t(32768),64);
            Muls(EndpointI(30),EndpointI(30),int32_t(32768),64);PipeBarrier<PIPE_V>();
            Sub(EndpointI(27),EndpointI(27),EndpointI(29),64);
            Sub(EndpointI(28),EndpointI(28),EndpointI(30),64);PipeBarrier<PIPE_V>();
            Cast(EndpointF(6),EndpointI(31),RoundMode::CAST_NONE,64);
            Cast(EndpointF(7),EndpointI(32),RoundMode::CAST_NONE,64);
            Cast(EndpointF(8),EndpointI(27),RoundMode::CAST_NONE,64);
            Cast(EndpointF(9),EndpointI(28),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Compare(masks[32],EndpointF(6),EndpointF(7),CMPMODE::LT,64);
            Compare(masks[64],EndpointF(6),EndpointF(7),CMPMODE::EQ,64);
            Compare(masks[96],EndpointF(8),EndpointF(9),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            And(masks[64].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),masks[96].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(masks[32].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(masks[128].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        Cast(EndpointF(33),EndpointI(12),RoundMode::CAST_NONE,64);
        Cast(EndpointF(34),EndpointI(13),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Div(EndpointF(35),EndpointF(33),EndpointF(34),64);PipeBarrier<PIPE_V>();
        Muls(EndpointF(35),EndpointF(35),unitSquared.value,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(35),masks[128],EndpointF(35),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        Select(FixedFewW(0),masks,EndpointF(35),FixedFewW(0),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        return true;
    }

    __aicore__ __attribute__((always_inline)) static inline bool CorrectHalfEndpointPacketTies(
        uint64_t validLanes,uint64_t equal,LocalTensor<int32_t> outputOffset,uint32_t maskAddress) {
        const LocalTensor<uint8_t> masks(TPosition::VECCALC,maskAddress,256);
        uint64_t counts=equal-((equal>>1)&uint64_t(0x5555555555555555ULL));
        counts=(counts&uint64_t(0x3333333333333333ULL))+((counts>>2)&uint64_t(0x3333333333333333ULL));
        counts=(counts+(counts>>4))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
        if(!(counts&uint64_t(0x0e0e0e0e0e0e0e0eULL)))return false;
        uint64_t ambiguous=0;
        #pragma unroll 1
        for(uint32_t group=0;group<8;++group) {
            const uint32_t bits=uint32_t(equal>>(group*8))&255U;
            if(bits && (bits&(bits-1U)))ambiguous|=uint64_t(255)<<(group*8);
        }
        if(!ambiguous)return false;
        // A conservative projection margin admits only strict endpoint regions.
        // If any nearby candidate has an uncertain/interior projection, abstain
        // for that complete packet and preserve its original computation.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Sub(EndpointF(axis),FixedFewP(axis),FixedFewC(axis),64);
            Sub(EndpointF(axis+3),FixedFewC(axis+3),FixedFewC(axis),64);
        }
        PipeBarrier<PIPE_V>();
        Mul(EndpointF(9),EndpointF(0),EndpointF(3),64);
        Mul(EndpointF(10),EndpointF(3),EndpointF(3),64);
        Mul(EndpointF(11),EndpointF(0),EndpointF(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=1;axis<3;++axis) {
            Mul(EndpointF(12),EndpointF(axis),EndpointF(axis+3),64);
            Mul(EndpointF(13),EndpointF(axis+3),EndpointF(axis+3),64);
            Mul(EndpointF(15),EndpointF(axis),EndpointF(axis),64);PipeBarrier<PIPE_V>();
            Add(EndpointF(9),EndpointF(9),EndpointF(12),64);
            Add(EndpointF(10),EndpointF(10),EndpointF(13),64);
            Add(EndpointF(11),EndpointF(11),EndpointF(15),64);PipeBarrier<PIPE_V>();
        }
        Add(EndpointF(12),EndpointF(10),EndpointF(11),64);
        Sub(EndpointF(13),EndpointF(9),EndpointF(10),64);PipeBarrier<PIPE_V>();
        Muls(EndpointF(12),EndpointF(12),0.0009765625F,64);PipeBarrier<PIPE_V>();
        Compare(masks[64],EndpointF(13),EndpointF(12),CMPMODE::GT,64);
        Muls(EndpointF(13),EndpointF(12),-1.0F,64);PipeBarrier<PIPE_V>();
        Compare(masks[32],EndpointF(9),EndpointF(13),CMPMODE::LT,64);
        CompareScalar(masks[96],EndpointF(10),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        Or(masks[96].template ReinterpretCast<uint16_t>(),masks[96].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Or(masks[96].template ReinterpretCast<uint16_t>(),masks[96].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),4);
        Muls(EndpointF(15),EndpointF(14),0.0009765625F,64);PipeBarrier<PIPE_V>();
        Add(EndpointF(15),EndpointF(15),EndpointF(14),64);PipeBarrier<PIPE_V>();
        Compare(masks,FixedFewW(0),EndpointF(15),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const uint64_t near=masks.template ReinterpretCast<uint64_t>().GetValue(0)&validLanes&ambiguous;
        const uint64_t endpoint=masks[96].template ReinterpretCast<uint64_t>().GetValue(0);
        uint64_t admitted=0;
        #pragma unroll 1
        for(uint32_t group=0;group<8;++group) {
            const uint64_t packet=uint64_t(255)<<(group*8);
            if((near&packet) && !(near&packet&~endpoint))admitted|=near&packet;
        }
        if(!admitted)return false;
        masks.template ReinterpretCast<uint64_t>().SetValue(0,admitted);
        masks[128].template ReinterpretCast<uint64_t>().SetValue(0,admitted);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Select(EndpointF(6+axis),masks[64],FixedFewC(axis+3),FixedFewC(axis),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        #pragma unroll
        for(uint32_t k=0;k<6;++k)Duplicate(EndpointI(k),int32_t(0),64);
        PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t axis=0;axis<3;++axis) {
            EndpointEncode(FixedFewP(axis),9);EndpointEncode(EndpointF(6+axis),12);
            Sub(EndpointI(31),EndpointI(9),EndpointI(12),64);PipeBarrier<PIPE_V>();EndpointSplit(15);
            Sub(EndpointI(31),EndpointI(10),EndpointI(13),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(16);
            Sub(EndpointI(17),EndpointI(11),EndpointI(14),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(17),EndpointI(17),EndpointI(30),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(31),EndpointI(15),int32_t(-1),64);PipeBarrier<PIPE_V>();EndpointSplit(18);
            Muls(EndpointI(31),EndpointI(16),int32_t(-1),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(19);
            Muls(EndpointI(20),EndpointI(17),int32_t(-1),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(20),EndpointI(20),EndpointI(30),64);
            Cast(EndpointF(38),EndpointI(17),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(masks[192],EndpointF(38),0.0F,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t k=0;k<3;++k)
                Select(EndpointF(21+k),masks[192],EndpointF(18+k),EndpointF(15+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
            PipeBarrier<PIPE_V>();
            Mul(EndpointI(31),EndpointI(21),EndpointI(21),64);PipeBarrier<PIPE_V>();EndpointSplit(24);
            Mul(EndpointI(31),EndpointI(21),EndpointI(22),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(31),EndpointI(31),int32_t(2),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(25);
            Mul(EndpointI(31),EndpointI(22),EndpointI(22),64);
            Mul(EndpointI(37),EndpointI(21),EndpointI(23),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(37),EndpointI(37),int32_t(2),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(37),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(26);
            Mul(EndpointI(31),EndpointI(22),EndpointI(23),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(31),EndpointI(31),int32_t(2),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(27);
            Mul(EndpointI(31),EndpointI(23),EndpointI(23),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(28);
            Copy(EndpointI(29),EndpointI(30),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            Duplicate(EndpointI(30),int32_t(0),64);PipeBarrier<PIPE_V>();
            #pragma unroll 1
            for(uint32_t k=0;k<6;++k) {
                Add(EndpointI(31),EndpointI(k),EndpointI(24+k),64);PipeBarrier<PIPE_V>();
                Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(k);
            }
        }
        // Lexicographic comparison of exact squared-norm digits; each digit
        // converts exactly to float and each eight-lane packet is independent.
        auto record=EndpointF(41);record.SetSize(128);
        #pragma unroll 1
        for(int32_t k=5;k>=0;--k) {
            Cast(EndpointF(39),EndpointI(uint32_t(k)),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Select(EndpointF(39),masks[128],EndpointF(39),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            WholeReduceMin(record,EndpointF(39),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
            Gather(EndpointF(43),record,outputOffset.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Brcb(EndpointF(44),EndpointF(43),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Compare(masks[160],EndpointF(39),EndpointF(44),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(masks[128].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),masks[160].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        Cast(EndpointF(45),EndpointI(5),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(int32_t k=4;k>=0;--k) {
            Muls(EndpointF(45),EndpointF(45),32768.0F,64);
            Cast(EndpointF(46),EndpointI(uint32_t(k)),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Add(EndpointF(45),EndpointF(45),EndpointF(46),64);PipeBarrier<PIPE_V>();
        }
        Muls(EndpointF(45),EndpointF(45),3.5527136788005009e-15F,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(45),masks[128],EndpointF(45),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        Select(FixedFewW(0),masks,EndpointF(45),FixedFewW(0),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        return true;
    }

#if defined(DTYPE_TREE_POINTS)
    __aicore__ __attribute__((noinline,aligned(128))) static uint32_t ResolveExactPacketBody(
        uint64_t validLanes,uint32_t pending) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        const LocalTensor<float> laneIds(TPosition::VECCALC,(workBase+6*(ROWS*64+128))*4,64);
        const LocalTensor<float> packetState(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr()),1024);
        const LocalTensor<float> backup(TPosition::VECCALC,(workBase+20*1056)*4,1728);
        uint64_t representable=~uint64_t(0);
        // Both encodings are selected only after testing every active coordinate.
        const auto certificate=backup[4*64].template ReinterpretCast<uint8_t>();
        const auto shortCertificate=backup[5*64].template ReinterpretCast<uint8_t>();
        Duplicate(certificate.template ReinterpretCast<uint16_t>(),uint16_t(65535),4);
        Duplicate(shortCertificate.template ReinterpretCast<uint16_t>(),uint16_t(65535),4);
        Duplicate(backup[7*64].template ReinterpretCast<uint32_t>(),uint32_t(0x7fffffffU),64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t coordinate=0;coordinate<9;++coordinate) {
            const auto x=coordinate<3?FixedFewP(coordinate):FixedFewC(coordinate-3);
            And(backup[6*64].template ReinterpretCast<uint16_t>(),x.template ReinterpretCast<uint16_t>(),
                backup[7*64].template ReinterpretCast<uint16_t>(),128);
            Abs(backup[3*64],x,64);PipeBarrier<PIPE_V>();
            CompareScalar(certificate[64],backup[3*64],128.0F,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            CompareScalar(certificate[96],backup[6*64].template ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);
            CompareScalar(shortCertificate[64],backup[3*64],9.5367431640625e-7F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
            Or(shortCertificate[64].template ReinterpretCast<uint16_t>(),shortCertificate[64].template ReinterpretCast<uint16_t>(),
                certificate[96].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Select(backup[0],certificate[64],x,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            Muls(backup[0],backup[0],1048576.0F,64);PipeBarrier<PIPE_V>();
            Cast(backup[64].template ReinterpretCast<int32_t>(),backup[0],RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
            Cast(backup[2*64],backup[64].template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Compare(shortCertificate[32],backup[0],backup[2*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate.template ReinterpretCast<uint16_t>(),certificate[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
#if TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            CompareScalar(certificate[128],backup[3*64],5.960464477539063e-8F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
            Or(certificate[128].template ReinterpretCast<uint16_t>(),certificate[128].template ReinterpretCast<uint16_t>(),
                certificate[96].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Muls(backup[0],backup[0],16.0F,64);PipeBarrier<PIPE_V>();
            Cast(backup[64].template ReinterpretCast<int32_t>(),backup[0],RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
            Cast(backup[2*64],backup[64].template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Compare(certificate[32],backup[0],backup[2*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(certificate.template ReinterpretCast<uint16_t>(),certificate.template ReinterpretCast<uint16_t>(),certificate[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(certificate.template ReinterpretCast<uint16_t>(),certificate.template ReinterpretCast<uint16_t>(),certificate[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(certificate.template ReinterpretCast<uint16_t>(),certificate.template ReinterpretCast<uint16_t>(),certificate[128].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
#endif
        }
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
#if TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        representable=certificate.template ReinterpretCast<uint64_t>().GetValue(0);
#endif
        const uint64_t shortBits=shortCertificate.template ReinterpretCast<uint64_t>().GetValue(0);
        uint32_t resolved=0;
        uint64_t exactLanes=0,groupMask=0;
        for(uint32_t group=0;group<8;++group) {
            const uint64_t active=validLanes&(uint64_t(255)<<(8*group));
            if((pending&(1U<<group)) && active && (active&representable)==active) {
                resolved|=1U<<group;
                exactLanes|=active;
                groupMask|=uint64_t(255)<<(8*group);
            }
        }
        if(!resolved)return 0;
        Copy(backup,packetState,64,16,CopyRepeatParams(1,1,8,8));
        Copy(backup[1024],FixedFewW(0),64,1,CopyRepeatParams(1,1,8,8));
        #pragma unroll 1
        for(uint32_t coordinate=0;coordinate<9;++coordinate) {
            const auto x=coordinate<3?FixedFewP(coordinate):FixedFewC(coordinate-3);
            Copy(backup[1088+coordinate*64],x,64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        const auto activeMask=backup[1664].template ReinterpretCast<uint8_t>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        activeMask.template ReinterpretCast<uint64_t>().SetValue(0,exactLanes);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        #pragma unroll 1
        for(uint32_t coordinate=0;coordinate<9;++coordinate) {
            const auto x=coordinate<3?FixedFewP(coordinate):FixedFewC(coordinate-3);
            Select(x,activeMask,x,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        }
        PipeBarrier<PIPE_V>();
        const LocalTensor<float> arena(TPosition::VECCALC,uint32_t(FewWork(0).GetPhyAddr()),224*64);
        if((exactLanes&shortBits)==exactLanes) {
            TreePacketIntegerRational64<> exact(uint32_t(arena.GetPhyAddr()));exact.Init();
            exact.template MakeScores<true>(0,uint32_t(FixedFewP(0).GetPhyAddr()),pointPitch,
                uint32_t(FixedFewC(0).GetPhyAddr()),uint32_t(laneIds.GetPhyAddr()),CAPACITY,
                uint32_t(backup.GetPhyAddr()),exactLanes);
            exact.template Tournament<3>();
        } else {
            TreePacketRational64<> exact(uint32_t(arena.GetPhyAddr()));exact.Init();
            exact.template MakeScores<true>(0,uint32_t(FixedFewP(0).GetPhyAddr()),pointPitch,
                uint32_t(FixedFewC(0).GetPhyAddr()),uint32_t(laneIds.GetPhyAddr()),CAPACITY,
                uint32_t(backup.GetPhyAddr()),exactLanes);
            exact.template Tournament<3>();
        }
        const LocalTensor<uint8_t> predicates(TPosition::VECCALC,uint32_t(arena[220*64].GetPhyAddr()),512);
        Compare(predicates,backup,arena[96*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        Select(arena[216*64],predicates,backup[1024],__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        predicates[32].template ReinterpretCast<uint64_t>().SetValue(0,groupMask);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        Select(FixedFewW(0),predicates[32],arena[216*64],backup[1024],SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        Copy(packetState,backup,64,16,CopyRepeatParams(1,1,8,8));
        #pragma unroll 1
        for(uint32_t coordinate=0;coordinate<9;++coordinate) {
            const auto x=coordinate<3?FixedFewP(coordinate):FixedFewC(coordinate-3);
            Copy(x,backup[1088+coordinate*64],64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        return resolved;
    }

    __aicore__ __attribute__((always_inline)) static inline uint32_t ResolveExactPacket(
        uint64_t validLanes,uint32_t safeGroups) {
        uint32_t pending=0;
        for(uint32_t group=0;group<8;++group)
            if(((validLanes>>(group*8))&255U) && !(safeGroups&(1U<<group)))pending|=1U<<group;
        if(!pending)return 0;

        return ResolveExactPacketBody(validLanes,pending);
    }

    __aicore__ __attribute__((noinline,aligned(128))) static bool ResolveExactSingle(
        uint32_t kept,LocalTensor<float> candidates,LocalTensor<float> record,
        LocalTensor<int32_t> index,LocalTensor<int32_t> offset,LocalTensor<int32_t> outputOffset) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        const LocalTensor<float> saved(TPosition::VECCALC,(workBase+20*1056+1696)*4,8);
        Copy(saved,BatchIds(),8,1,CopyRepeatParams(1,1,1,1));
        Copy(BatchIds(),candidates,8,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
        const bool resolved=(ResolveExactPacket((uint64_t(1)<<kept)-1,0)&1U)!=0;
        Copy(BatchIds(),saved,8,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
        if(!resolved)return false;
        WholeReduceMin(record,FixedFewW(0),8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
        Gather(offset,record.template ReinterpretCast<int32_t>(),outputOffset.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
        Muls(offset,offset,int32_t(4),1);PipeBarrier<PIPE_V>();
        Gather(index.template ReinterpretCast<float>(),candidates,offset.template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
        Cast(index,index.template ReinterpretCast<float>(),RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
        return true;
    }
#endif
    __aicore__ __attribute__((noinline,aligned(128))) static bool CorrectPendingCollinearGroups(
        uint64_t validLanes,uint32_t safeGroups,uint32_t groups) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        const LocalTensor<uint8_t> valid(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        bool touched=false;
        for(uint32_t group=0;group<groups;++group)if(!(safeGroups&(1U<<group))) {
            const uint32_t bits=uint32_t(validLanes>>(group*8))&255U;
            if(bits && (bits&(bits-1)) && MayHaveCollinearInteriorTies8NormalMask(
                uint32_t(FixedFewW(0)[group*8].GetPhyAddr()),
                uint32_t(FixedFewC(0)[group*8].GetPhyAddr()),128,
                uint32_t(FewWork(0)[20*1056].GetPhyAddr()),uint32_t(valid.GetPhyAddr())-ROWS*8)) {
                CorrectCollinearInteriorTies8(uint32_t(FixedFewW(0)[group*8].GetPhyAddr()),
                    uint32_t(FixedFewP(0)[group*8].GetPhyAddr()),pointPitch,
                    uint32_t(FixedFewC(0)[group*8].GetPhyAddr()),128,
                    uint32_t(FewWork(0)[20*1056].GetPhyAddr()),uint32_t(valid.GetPhyAddr())-ROWS*8);
                touched=true;
            }
        }
        return touched;
    }
    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineCandidateBatch(uint64_t validLanes,uint64_t rows,uint64_t& uniqueRows) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)AscendC::ICachePreLoad(int64_t(1));
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> compactP_(TPosition::VECCALC,(stateBase-13*64-192)*4,192);
        const LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,(stateBase+128+16+14*64)*4,64);
        const LocalTensor<float> record_(TPosition::VECCALC,stateBase*4,128);
        const LocalTensor<int32_t> idx_(TPosition::VECCALC,(stateBase+128+16+3*64)*4,64);
        const LocalTensor<float> tileSecond_(TPosition::VECCALC,(stateBase+128+16+8*64)*4,64);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*4)*4,32);
        const LocalTensor<float> distances(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr())+160*4,8);
        const LocalTensor<int32_t> indices(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr())+192*4,8);
        Cast(FewWork(7).template ReinterpretCast<int32_t>(),BatchIds(),RoundMode::CAST_RINT,64);PipeBarrier<PIPE_V>();
        Muls(FewWork(7).template ReinterpretCast<int32_t>(),FewWork(7).template ReinterpretCast<int32_t>(),int32_t(4),64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t col=0;col<14;++col)if(col<6||col>9)
            Gather(FixedFewC(col),FewCoefficients(col),FewWork(7).template ReinterpretCast<uint32_t>(),0U,64);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Gather(FewWork(axis),compactP_[axis*64],BatchPointIds(),0U,8);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Brcb(FixedFewP(axis),FewWork(axis),1,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        valid_.template ReinterpretCast<uint64_t>().SetValue(0,validLanes);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        if(!BatchFewCompensated())return false;
        // Eight independent8-lane minima preserve each point's increasing global indices.
        WholeReduceMin(record_,FixedFewW(0),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS)
        if constexpr(sizeof(T)==4) {
            Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            uint32_t safeGroups=CertifyRefined(distances,validLanes);
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
                // A strictly separated interval cannot contain an exact tie.
                // Only unresolved groups enter the supporting-line certificate.
                const uint32_t groups=PredicatePopulation(rows);
                if(safeGroups!=((1U<<groups)-1U) &&
                   CorrectPendingCollinearGroups(validLanes,safeGroups,groups)) {
                    WholeReduceMin(record_,FixedFewW(0),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
                    safeGroups|=CertifyRefined(distances,validLanes);
                }
            }
            const uint32_t exactGroups=ResolveExactPacket(validLanes,safeGroups);
            if(exactGroups) {
                safeGroups|=exactGroups;
                WholeReduceMin(record_,FixedFewW(0),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            }
            for(uint32_t group=0;group<8;++group)if(safeGroups&(1U<<group)) {
                const uint32_t row=BatchPointIds().GetValue(group)/4;
                uniqueRows|=(uint64_t(1)<<row)&rows;
            }
        }
#endif
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(NATIVE_HALF_PREFIX_PRIMARY) {
            const auto localOffset=EndpointI(61);
            const auto packetOffset=EndpointI(62);
            const auto mappedDistance=EndpointF(63);
            const auto mappedIndex=EndpointI(64);
            uint32_t safeGroups=0,pendingGroups=0;
            for(uint32_t group=0;group<8;++group) {
                const uint32_t active=uint32_t(validLanes>>(group*8))&255U;
                if(active && !(active&(active-1U)))safeGroups|=1U<<group;
                else if(active)pendingGroups|=1U<<group;
            }
            if(pendingGroups) {
                Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
                safeGroups|=CertifyRefined(distances,validLanes);
                const uint32_t exactGroups=ResolveExactPacket(validLanes,safeGroups);
                if(exactGroups) {
                    safeGroups|=exactGroups;
                    WholeReduceMin(record_,FixedFewW(0),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                }
            }
            for(uint32_t group=0;group<8;++group)if(safeGroups&(1U<<group)) {
                const uint32_t row=BatchPointIds().GetValue(group)/4;
                uniqueRows|=(uint64_t(1)<<row)&rows;
            }
            Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);
            Gather(localOffset,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,8);PipeBarrier<PIPE_V>();
            Muls(localOffset,localOffset,int32_t(4),8);
            Muls(packetOffset,outputOffset_,int32_t(4),8);PipeBarrier<PIPE_V>();
            Add(localOffset,localOffset,packetOffset,8);PipeBarrier<PIPE_V>();
            Gather(indices.template ReinterpretCast<float>(),BatchIds(),localOffset.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Cast(indices,indices.template ReinterpretCast<float>(),RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
            Gather(mappedDistance,distances,BatchPointMap(),0U,64);
            Gather(mappedIndex,indices,BatchPointMap(),0U,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            smallMask_.template ReinterpretCast<uint64_t>().SetValue(0,rows);
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));SetCmpMask(smallMask_);
            const BinaryRepeatParams same{1,1,1,8,8,8};
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,mappedDistance,tileSecond_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),mappedIndex.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            return true;
        }
#endif
        Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);
        Gather(FewWork(0).template ReinterpretCast<int32_t>(),record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,8);PipeBarrier<PIPE_V>();
        Muls(FewWork(0).template ReinterpretCast<int32_t>(),FewWork(0).template ReinterpretCast<int32_t>(),int32_t(4),8);
        Muls(FewWork(1).template ReinterpretCast<int32_t>(),outputOffset_,int32_t(4),8);PipeBarrier<PIPE_V>();
        Add(FewWork(0).template ReinterpretCast<int32_t>(),FewWork(0).template ReinterpretCast<int32_t>(),FewWork(1).template ReinterpretCast<int32_t>(),8);PipeBarrier<PIPE_V>();
        Gather(indices.template ReinterpretCast<float>(),BatchIds(),FewWork(0).template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
        Cast(indices,indices.template ReinterpretCast<float>(),RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
        Gather(FewWork(0),distances,BatchPointMap(),0U,64);
        Gather(FewWork(1).template ReinterpretCast<int32_t>(),indices,BatchPointMap(),0U,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        smallMask_.template ReinterpretCast<uint64_t>().SetValue(0,rows);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));SetCmpMask(smallMask_);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,FewWork(0),tileSecond_,1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),FewWork(1),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        return true;
    }
    __aicore__ __attribute__((always_inline)) static inline void ClearCandidateBatch() {
        Duplicate(BatchIds(),0.0F,64);
        Duplicate(BatchPointIds(),uint32_t(0),8);
        Duplicate(BatchPointMap(),uint32_t(0),64);PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<int32_t> SavedOverlapChunks() {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch+13*1056+256;
        return LocalTensor<int32_t>(TPosition::VECCALC,base*4,64);
    }
    __aicore__ __attribute__((noinline,aligned(128))) static void BuildOverlapChunks(uint32_t segments) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> chunkLower(TPosition::VECCALC,(workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0))*4,1024);
        const LocalTensor<float> hi(TPosition::VECCALC,(stateBase+128+16+2*64)*4,64);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<int32_t,false>(SavedOverlapChunks(),int32_t(0),uint64_t(64),1,1,8);
        Duplicate<float,false>(FewWork(1),0.0F,uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        for(uint32_t chunk=0;chunk<(segments+63)/64;++chunk) {
            Compare<float,false>(chunkLower[chunk*64],hi,CMPMODE::LE,uint64_t(64),same);
            Duplicate<int32_t,false>(FewWork(0).template ReinterpretCast<int32_t>(),int32_t(uint32_t(1)<<chunk),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(FewWork(0),FewWork(0),FewWork(1),1,same);PipeBarrier<PIPE_V>();
            // Distinct power-of-two bits sum exactly to their union; no floating arithmetic.
            Add<int32_t,false>(SavedOverlapChunks(),SavedOverlapChunks(),FewWork(0).template ReinterpretCast<int32_t>(),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    __aicore__ __attribute__((noinline,aligned(128))) static uint64_t RefineSparseRows(uint64_t rows,uint32_t validRows,uint32_t segments,uint64_t& uniqueRows) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)AscendC::ICachePreLoad(int64_t(2));
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> chunkLower_(TPosition::VECCALC,(workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0))*4,1024);
        const LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,(stateBase+128+16+14*64)*4,64);
        const LocalTensor<int32_t> offset_(TPosition::VECCALC,(stateBase+128+16+11*64)*4,64);
        const LocalTensor<int32_t> idx_(TPosition::VECCALC,(stateBase+128+16+3*64)*4,64);
        const LocalTensor<float> hi_(TPosition::VECCALC,(stateBase+128+16+2*64)*4,64);
        const LocalTensor<float> segmentIndices_(TPosition::VECCALC,(workBase+6*(ROWS*64+128))*4,64);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*4)*4,32);
        uint32_t count=0;
        uint64_t validLanes=0,groupRows=0,accepted=0;
        ClearCandidateBatch();
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE || NATIVE_HALF_PREFIX_PRIMARY) {
            // Keep uncertain geometric-zero rows on their exact fallback.
            // A zero row must not reject seven independent nonzero queries.
            const LocalTensor<float> primaryScores(TPosition::VECCALC,(stateBase+128+16+8*64)*4,64);
            const LocalTensor<float> primaryNorm(TPosition::VECCALC,(stateBase-4*64)*4,64);
            Muls(FewWork(0),primaryNorm,1.8189894035458565e-12F,64);PipeBarrier<PIPE_V>();
            Compare(smallMask_,primaryScores,FewWork(0),CMPMODE::GT,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            rows&=smallMask_.template ReinterpretCast<uint64_t>().GetValue(0);
        }
#endif
        const uint64_t active=(~uint64_t(0))>>(64-validRows);
        for(uint64_t pendingRows=rows&active;pendingRows!=0;pendingRows&=pendingRows-1) {
            const uint32_t row=uint32_t(GetSFFValue<1>(pendingRows));
            Muls(offset_,outputOffset_,int32_t(32),16);PipeBarrier<PIPE_V>();
            Adds(offset_,offset_,int32_t(row*4),16);
            Duplicate(FewWork(0),__builtin_inff(),64);
            Brcb(FewWork(1),hi_[(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Copy(FewWork(2),FewWork(1)[(row%8)*8],64,1,CopyRepeatParams(1,0,8,0));
            Gather(FewWork(0),chunkLower_,offset_.template ReinterpretCast<uint32_t>(),0U,(segments+63)/64);PipeBarrier<PIPE_V>();
            Compare(smallMask_,FewWork(0),FewWork(2),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t chunks=smallMask_.template ReinterpretCast<uint32_t>().GetValue(0)&((1U<<((segments+63)/64))-1);
            const uint32_t chosenChunk=uint32_t(idx_.GetValue(row))/64;
            if(chunks==0 || !(chunks&(uint32_t(1)<<chosenChunk)))continue;
            uint32_t candidates=0;
            for(uint32_t pendingChunks=chunks;pendingChunks!=0;pendingChunks&=pendingChunks-1) {
                const uint32_t chunk=uint32_t(GetSFFValue<1>(uint64_t(pendingChunks)));
                uint64_t bits=SavedChunkPredicate(row,chunk);
                const uint32_t remaining=segments-chunk*64;
                if(remaining<64)bits&=(uint64_t(1)<<remaining)-1;
                candidates+=PredicatePopulation(bits);
            }
            if(candidates==0 || candidates>8)continue;
            Duplicate(FewWork(6),0.0F,8);
            Duplicate(valid_.template ReinterpretCast<uint32_t>(),uint32_t(0),32);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            uint32_t packed=0;
            for(uint32_t pendingChunks=chunks;pendingChunks!=0;pendingChunks&=pendingChunks-1) {
                const uint32_t chunk=uint32_t(GetSFFValue<1>(uint64_t(pendingChunks)));
                uint64_t bits=SavedChunkPredicate(row,chunk);
                const uint32_t remaining=segments-chunk*64;
                if(remaining<64)bits&=(uint64_t(1)<<remaining)-1;
                valid_.template ReinterpretCast<uint32_t>().SetValue(packed*2,uint32_t(bits));
                valid_.template ReinterpretCast<uint32_t>().SetValue(packed*2+1,uint32_t(bits>>32));
                ++packed;
            }
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            packed=0;
            for(uint32_t pendingChunks=chunks;pendingChunks!=0;pendingChunks&=pendingChunks-1) {
                const uint32_t chunk=uint32_t(GetSFFValue<1>(uint64_t(pendingChunks)));
                Adds(FewWork(7)[packed*64],segmentIndices_,float(int32_t(chunk*64)),64);
                ++packed;
            }
            PipeBarrier<PIPE_V>();
            uint64_t kept=0;
            GatherMask(FewWork(6),FewWork(7),valid_.template ReinterpretCast<uint32_t>(),true,packed*64,
                GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            if(kept==0 || kept>8)continue;
            Copy(BatchIds()[count*8],FewWork(6),8,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            BatchPointIds().SetValue(count,row*4);
            BatchPointMap().SetValue(row,count*4);
            validLanes|=((uint64_t(1)<<kept)-1)<<(count*8);
            groupRows|=uint64_t(1)<<row;
            ++count;
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            if(count==8) {
                if(RefineCandidateBatch(validLanes,groupRows,uniqueRows))accepted|=groupRows;
                ClearCandidateBatch();count=0;validLanes=0;groupRows=0;
            }
        }
        if(count!=0 && RefineCandidateBatch(validLanes,groupRows,uniqueRows))accepted|=groupRows;
        return accepted;
    }
    template<bool MERGED_CERTIFICATE=false>
    __aicore__ __attribute__((always_inline)) static inline bool FixedFewInlineBody() {
        constexpr uint32_t COUNT=8;
        constexpr bool REJECT_ZERO=true;
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> record(TPosition::VECCALC,stateBase*4,128);
        const LocalTensor<uint8_t> mask(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4)*4,ROWS*8);
        const LocalTensor<uint8_t> valid(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*4)*4,32);
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub(FixedFewW(axis+9),FixedFewC(axis+3),FixedFewC(axis),COUNT);
        PipeBarrier<PIPE_V>();
        constexpr bool directNorm=sizeof(T)==4 && CAPACITY==256
#ifdef DTYPE_TREE_POINTS
            && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
        const LocalTensor<float> norm=directNorm?FixedFewC(13):FixedFewW(12);
        if constexpr(!directNorm) {
            Adds(FixedFewW(12),FixedFewC(13),0.0F,COUNT);PipeBarrier<PIPE_V>();
        }
        const uint8_t repeats = (COUNT+63)/64;
        constexpr uint32_t lanes=COUNT<64?COUNT:64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(lanes));
        if constexpr(DEFER_DIRECTION_LOW) {
            // Independent exact subtractions share producer/consumer waits.
            // W13..18 is dead coarse scratch, disjoint from points and columns.
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Sub<float,false>(FixedFewW(axis),FixedFewP(axis),FixedFewC(axis),uint64_t(64),repeats,binary);
                Sub<float,false>(FixedFewW(axis+16),FixedFewC(axis+3),FixedFewW(axis+9),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Sub<float,false>(FixedFewW(axis+3),FixedFewP(axis),FixedFewW(axis),uint64_t(64),repeats,binary);
                Add<float,false>(FixedFewW(axis+13),FixedFewW(axis+9),FixedFewW(axis+16),uint64_t(64),repeats,binary);
                Sub<float,false>(FixedFewC(axis+10),FixedFewW(axis+16),FixedFewC(axis),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Add<float,false>(FixedFewW(axis+6),FixedFewW(axis),FixedFewW(axis+3),uint64_t(64),repeats,binary);
                Sub<float,false>(FixedFewW(axis+3),FixedFewW(axis+3),FixedFewC(axis),uint64_t(64),repeats,binary);
                Sub<float,false>(FixedFewW(axis+13),FixedFewC(axis+3),FixedFewW(axis+13),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Sub<float,false>(FixedFewW(axis+6),FixedFewP(axis),FixedFewW(axis+6),uint64_t(64),repeats,binary);
                Add<float,false>(FixedFewC(axis+10),FixedFewC(axis+10),FixedFewW(axis+13),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Add<float,false>(FixedFewW(axis+3),FixedFewW(axis+3),FixedFewW(axis+6),uint64_t(64),repeats,binary);
            PipeBarrier<PIPE_V>();
        } else {
            // Error-free subtraction preserves short distances at large coordinates.
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis), FixedFewP(axis), FixedFewC(axis), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+3), FixedFewP(axis), FixedFewW(axis), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) {
                Add<float, false>(FixedFewW(axis+6), FixedFewW(axis), FixedFewW(axis+3), uint64_t(64), repeats, binary);
                Sub<float, false>(FixedFewW(axis+3), FixedFewW(axis+3), FixedFewC(axis), uint64_t(64), repeats, binary);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+6), FixedFewP(axis), FixedFewW(axis+6), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(FixedFewW(axis+3), FixedFewW(axis+3), FixedFewW(axis+6), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
        }
        Mul<float, false>(FixedFewW(8), FixedFewW(0), FixedFewW(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(8), FixedFewW(1), FixedFewW(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(8), FixedFewW(2), FixedFewW(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Div<float, false>(FixedFewW(8), FixedFewW(8), norm, uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(FixedFewW(8), FixedFewW(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(FixedFewW(8), FixedFewW(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        CompareScalar<float, uint8_t, false>(mask, FixedFewW(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(FixedFewW(8), FixedFewW(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(FixedFewW(axis), FixedFewW(8), FixedFewW(axis+9), uint64_t(64), repeats, binary);
            MulAddDst<float, float, false>(FixedFewW(axis+3), FixedFewW(8), FixedFewC(axis+10), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(FixedFewW(axis), FixedFewW(axis), FixedFewW(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+3), FixedFewP(axis), FixedFewC(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(COUNT<=64) {
            SetCmpMask(mask);
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(FixedFewW(axis),FixedFewW(axis),FixedFewW(axis+3),1,binary);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,uint8_t,false>(FixedFewW(axis),mask,FixedFewW(axis),FixedFewW(axis+3),SELMODE::VSEL_TENSOR_TENSOR_MODE,uint64_t(64),repeats,binary);
        }
        PipeBarrier<PIPE_V>();
        Mul<float, false>(FixedFewW(0), FixedFewW(0), FixedFewW(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(0), FixedFewW(1), FixedFewW(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(0), FixedFewW(2), FixedFewW(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t, false>(FixedFewW(0), valid, FixedFewW(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(!MERGED_CERTIFICATE) {
        WholeReduceMax(FixedFewW(7),norm,lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record,FixedFewW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(record,record,1.8189894035458565e-12F,1);
        WholeReduceMin(FixedFewW(7),FixedFewW(0),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(record[64],FixedFewW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Compare(smallMask,record[64],record,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(smallMask.GetValue(0)&1U) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            return false;
        }
        }
        if constexpr(COUNT<64) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));}
        return true;
    }

    __aicore__ __attribute__((noinline,aligned(128))) static bool FixedFewCompensated() {
        return FixedFewInlineBody();
    }

    template<uint32_t COUNT,bool REJECT_ZERO=false>
    __aicore__ __attribute__((always_inline)) inline bool SparseCompensated() {
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub(SparseW(axis+9),SparseC(axis+3),SparseC(axis),COUNT);
        PipeBarrier<PIPE_V>();
        Adds(SparseW(12),SparseC(13),0.0F,COUNT);PipeBarrier<PIPE_V>();
        const uint8_t repeats = (COUNT+63)/64;
        constexpr uint32_t lanes=COUNT<64?COUNT:64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(lanes));
        // Error-free subtraction preserves short distances at large coordinates.
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(SparseW(axis), SearchP<true>(axis), SparseC(axis), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(SparseW(axis+3), SearchP<true>(axis), SparseW(axis), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            Add<float, false>(SparseW(axis+6), SparseW(axis), SparseW(axis+3), uint64_t(64), repeats, binary);
            Sub<float, false>(SparseW(axis+3), SparseW(axis+3), SparseC(axis), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(SparseW(axis+6), SearchP<true>(axis), SparseW(axis+6), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(SparseW(axis+3), SparseW(axis+3), SparseW(axis+6), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(SparseW(8), SparseW(0), SparseW(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(SparseW(8), SparseW(1), SparseW(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(SparseW(8), SparseW(2), SparseW(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Div<float, false>(SparseW(8), SparseW(8), SparseW(12), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(SparseW(8), SparseW(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(SparseW(8), SparseW(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        CompareScalar<float, uint8_t, false>(mask_, SparseW(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(SparseW(8), SparseW(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(SparseW(axis), SparseW(8), SparseW(axis+9), uint64_t(64), repeats, binary);
            MulAddDst<float, float, false>(SparseW(axis+3), SparseW(8), SparseC(axis+10), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(SparseW(axis), SparseW(axis), SparseW(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(SparseW(axis+3), SearchP<true>(axis), SparseC(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(COUNT<=64) {
            SetCmpMask(mask_);
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(SparseW(axis),SparseW(axis),SparseW(axis+3),1,binary);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,uint8_t,false>(SparseW(axis),mask_,SparseW(axis),SparseW(axis+3),SELMODE::VSEL_TENSOR_TENSOR_MODE,uint64_t(64),repeats,binary);
        }
        PipeBarrier<PIPE_V>();
        Mul<float, false>(SparseW(0), SparseW(0), SparseW(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(SparseW(0), SparseW(1), SparseW(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(SparseW(0), SparseW(2), SparseW(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t, false>(SparseW(0), valid_, SparseW(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(true) {
        WholeReduceMax(SparseW(7),SparseW(12),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,SparseW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(record_,record_,1.8189894035458565e-12F,1);
        WholeReduceMin(SparseW(7),SparseW(0),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(record_[64],SparseW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Compare(smallMask_,record_[64],record_,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(smallMask_.GetValue(0)&1U) {
            if constexpr(REJECT_ZERO) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                return false;
            } else {
            CorrectAxialZeroScores(uint32_t(SparseW(0).GetPhyAddr()),uint32_t(14*(CAPACITY+32)*4),uint32_t((CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32),
                uint32_t((14*(CAPACITY+32)+3*((CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32)+16*1056)*4),uint32_t(128),uint32_t(SparseW(1).GetPhyAddr()),uint32_t(160),uint32_t(mask_.GetPhyAddr()),
                (COUNT+255)/256*32,COUNT);
            CorrectCollinearZeroScores<sizeof(T)==2>(uint32_t(SparseW(0).GetPhyAddr()),uint32_t(14*(CAPACITY+32)*4),uint32_t((CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32),
                uint32_t((14*(CAPACITY+32)+3*((CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32)+16*1056)*4),uint32_t(128),uint32_t(work_[20*1056].GetPhyAddr()),uint32_t(record_.GetPhyAddr()),
                uint32_t(mask_.GetPhyAddr()),COUNT);
            Select(SparseW(0),valid_,SparseW(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,COUNT);PipeBarrier<PIPE_V>();
            }
        }
        }
        if constexpr(COUNT<64) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));}
        return true;
    }

    bool exactDirectionLow_=false;
    bool tightCoefficientSafe_=false;
    bool directionLowReady_;
    TreeLineMinDistanceTilingData t_;uint32_t m_,cp_,wp_,pp_,count_,liveRows_;
    GlobalTensor<T> pg_,sg_,dg_;GlobalTensor<int32_t> ig_;
    LocalTensor<T> rawS_,rawP_,outT_;
    LocalTensor<float> columns_,xyz_,work_,sf_,pf_,ones_,segmentIndices_,compactP_,refined_,chunkLower_;
    LocalTensor<float> record_,record2_,lo_,second_,hi_,tileLo_,tileSecond_,tileHi_,cert_,certHigh_;
    LocalTensor<int32_t> so_,idx_,tileIdx_,offset_,rowBase_,pointOffset_,outputOffset_,oneHot_;
    LocalTensor<uint8_t> mask_,valid_,smallMask_;
};

#endif
#else
#ifndef TREE_LINE_MIN_DISTANCE_TILING_DATA_DEFINED
#define TREE_LINE_MIN_DISTANCE_TILING_DATA_DEFINED

#include <cstdint>

struct TreeLineMinDistanceTilingData {
    uint32_t batches;
    uint32_t points;
    uint32_t segments;
    uint32_t paddedSegments;
    uint32_t pointTile;
    uint32_t blocks;
    uint32_t schedule;
    uint32_t segmentLog;
};

#endif
#endif
