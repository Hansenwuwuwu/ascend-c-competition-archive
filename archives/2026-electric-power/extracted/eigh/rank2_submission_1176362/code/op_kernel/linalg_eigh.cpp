#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "linalg_eigh_tiling.h"
#include "tiling_key_linalg_eigh.h"

// extended_float.h
// A two-float expansion keeps tridiagonal rotations above float32 precision.
struct ExtendedFloat {
    float h,m,l;
    __aicore__ inline ExtendedFloat(float x=0):h(x),m(0),l(0){}
    __aicore__ inline ExtendedFloat(float a,float b,float c):h(a),m(b),l(c){}
    __aicore__ static inline void Sum(float a,float b,float &s,float &e){s=a+b;float q=s-a;e=(a-(s-q))+(b-q);}
    __aicore__ static inline void Product(float a,float b,float &p,float &e){
        p=a*b;float t=4097.0f*a,ah=t-(t-a),al=a-ah;t=4097.0f*b;float bh=t-(t-b),bl=b-bh;e=((ah*bh-p)+ah*bl+al*bh)+al*bl;
    }
    __aicore__ static inline ExtendedFloat Norm(float a,float b,float c){float s,e;Sum(a,b+c,s,e);return ExtendedFloat(s,e,0);}
    __aicore__ static inline ExtendedFloat QuickNorm(float a,float b){float s=a+b,e=b-(s-a);return ExtendedFloat(s,e,0);}
    __aicore__ inline ExtendedFloat Scale(float b)const{return QuickNorm(h*b,m*b);}
    __aicore__ inline ExtendedFloat operator-()const{return ExtendedFloat(-h,-m,-l);}
    __aicore__ inline ExtendedFloat operator+(const ExtendedFloat &b)const{float s,e;Sum(h,b.h,s,e);return QuickNorm(s,e+m+b.m);}
    __aicore__ inline ExtendedFloat operator-(const ExtendedFloat &b)const{return *this+(-b);}
    __aicore__ inline ExtendedFloat operator*(const ExtendedFloat &b)const{float p,e;Product(h,b.h,p,e);return QuickNorm(p,e+(h*b.m+m*b.h));}
    __aicore__ inline ExtendedFloat operator/(const ExtendedFloat &b)const{float q=h/b.h,p,e;Product(b.h,q,p,e);ExtendedFloat product=QuickNorm(p,e+b.m*q);float residual=(h-product.h)+(m-product.m);return QuickNorm(q,residual/b.h);}
    __aicore__ inline ExtendedFloat AddFull(const ExtendedFloat &b)const{return *this+b;}
    __aicore__ inline ExtendedFloat MulFull(const ExtendedFloat &b)const{return *this*b;}
    __aicore__ inline float Float()const{return h+m;}
    __aicore__ inline bool Negative()const{return h<0 || (h==0 && m<0);}
    __aicore__ inline bool Zero()const{return h==0 && m==0;}
};


// linalg_eigh_vector.h
using namespace AscendC;

// Vector solvers own an explicitly aligned, nonoverlapping UB arena. Cube
// Matmul retains its separate TPipe for library-owned local buffers.
struct ResidentUbBuffer {
    uint32_t address,bytes;
    template<typename T> __aicore__ inline __sync_alias__ LocalTensor<T> Get(){
        return LocalTensor<T>(TPosition::VECCALC,get_imm(0)+address,bytes/sizeof(T));
    }
};
struct ResidentUbArena {
    uint32_t next;
    __aicore__ inline ResidentUbArena():next(0){InitSocState();}
    __aicore__ inline void InitBuffer(ResidentUbBuffer &buffer,uint32_t bytes){
        buffer.address=next;buffer.bytes=(bytes+31)/32*32;next+=buffer.bytes;
    }
};
template<bool Direct> struct EighMemoryTypes {
    using Pipe=TPipe;using Buffer=TBuf<TPosition::VECCALC>;
};
template<> struct EighMemoryTypes<true> {
    using Pipe=ResidentUbArena;using Buffer=ResidentUbBuffer;
};

// Distributed symmetric Householder reduction, Sturm bisection,
// inverse iteration, and reflector backtransform.
template <typename DT_A,uint32_t FixedStride=0,uint32_t CubeMode=0,uint32_t FixedOrder=0,bool PrivateSmall=false,bool DirectUb=false>
class KernelLinalgEigh {
    typename EighMemoryTypes<DirectUb>::Pipe pipe;
    typename EighMemoryTypes<DirectUb>::Buffer storage,wideStorage,smallStorage,historyStorage;
    LocalTensor<float> privateHistory;
    uint32_t currentBatch;
    LocalTensor<float> mat,prod,v,w,tmp,tmp2,red,param,d,e,origD,origE,beta,piv,fac,pack,wideBuf,pcrBand,pcrLeft;
    GlobalTensor<float> input,values,vectors,ws,hist,gw,gparam;
    uint32_t dynamicOrder,dynamicStride,batch,upper,team,groups,rows,laneStride,lane,group,base,count;
    uint32_t privateTailColumnCount;
    uint32_t privateCapacity,storageCapacity;
    bool packedMatrix,reservedLayout,directCoefficients;
    __aicore__ inline uint32_t Order()const{
        if constexpr(FixedOrder!=0)return FixedOrder;
        return dynamicOrder;
    }
    __aicore__ inline uint32_t Stride()const{
        if constexpr(FixedStride!=0)return FixedStride;
        return dynamicStride;
    }
    // Both the dynamic resident entry and fixed 16-wide resident tile
    // own small matrices. Cube entries remain outside this family.
    static constexpr bool ResidentEntry=(FixedStride==0 || FixedOrder!=0) && CubeMode==0;
    __aicore__ inline bool ResidentFamily()const{
        if constexpr(!ResidentEntry)return false;
        return Order()<=25;
    }
    float spectralBound,spectralCenter;
    bool active,smallFloatQL,smallQLSolved,smallQLNeedsCheck,smallQLReadyOutput;
    bool outputBackedHistory;
    float smallRowBound;
    // Cyclic ownership balances the work for localized eigenvectors.
    __aicore__ inline uint32_t EigenIndex(uint32_t r){return lane+r*team;}
    __aicore__ inline void Barrier(){PipeBarrier<PIPE_ALL>();}
    __aicore__ inline float Abs(float a){return a<0?-a:a;}
    __aicore__ inline float Root(float a){
        if(a<=0)return 0;
        union Bits{float f;uint32_t u;}b;b.f=a;
        if(a>=1e-30f && a<=1e30f){
            b.u=0x5f375a86u-(b.u>>1);float reciprocal=b.f;
            for(uint32_t i=0;i<2;++i)reciprocal=reciprocal*(1.5f-((0.5f*a)*reciprocal)*reciprocal);
            b.f=a*reciprocal;b.u+=128u;float x=b.f;
            // Two reciprocal-root Newton steps bound the initial error below
            // 2e-5; one square-root Newton step reaches float32 rounding.
            x=0.5f*(x+a/x);return x;
        }
        b.u=(b.u>>1)+0x1fc00000u;float x=b.f;
        x=0.5f*(x+a/x);x=0.5f*(x+a/x);x=0.5f*(x+a/x);x=0.5f*(x+a/x);return x;
    }
    template<HardEvent Event> __aicore__ inline event_t ImmediateEvent(){
        // Each flag is waited immediately, so no live event shares this ID.
        if constexpr(DirectUb)return EVENT_ID0;
        else return static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    }
    __aicore__ inline void ReduceSumWithArena(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float> scratch,uint32_t length){
        // Resident strides fit one SIMD repeat. WholeReduceSum has no implicit
        // TPipe event allocation; the caller synchronizes before scalar reads.
        if constexpr(DirectUb){
            if(length<=64)WholeReduceSum(dst,src,length,1,1,1,8);
            else{
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(length));
                vcadd((__ubuf__ float*)dst.GetPhyAddr(),(__ubuf__ float*)src.GetPhyAddr(),1,1,1,8,1);
                VectorToScalar();float value=GetReduceRepeatSumSpr<float>();
                SetMaskNorm();ResetMask();dst.SetValue(0,value);
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            }
        }
        else ReduceSum(dst,src,scratch,length);
    }
    __aicore__ inline void CreateIndicesWithArena(LocalTensor<int32_t> dst,int32_t first,uint32_t length){
        if constexpr(DirectUb){
            const uint32_t prefix=length<32?length:8;
            for(uint32_t i=0;i<prefix;++i)dst.SetValue(i,first+int32_t(i));
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            // Independent eight-element blocks reuse the initialized prefix.
            for(uint32_t i=prefix;i<length;i+=8){
                uint32_t live=length-i<8?length-i:8;
                Adds(dst[i],dst,int32_t(i),live);
            }
            PipeBarrier<PIPE_V>();
        }else CreateVecIndex(dst,first,length);
    }
    __aicore__ inline void VectorToScalar(){
        event_t id=ImmediateEvent<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(id);WaitFlag<HardEvent::V_S>(id);
    }
    __aicore__ inline float Dot(LocalTensor<float>a,LocalTensor<float>b){
        Mul(tmp2,a,b,Stride());PipeBarrier<PIPE_V>();
#if __NPU_ARCH__ == 2201
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(Stride()));
        vcadd((__ubuf__ float*)red.GetPhyAddr(),(__ubuf__ float*)tmp2.GetPhyAddr(),1,1,1,8,1);
        VectorToScalar();float result=GetReduceRepeatSumSpr<float>();
        SetMaskNorm();ResetMask();return result;
#else
        ReduceSumWithArena(red,tmp2,tmp,Stride());VectorToScalar();return red.GetValue(0);
#endif
    }
    __aicore__ inline float DotTail(LocalTensor<float>a,LocalTensor<float>b,uint32_t begin){
#if __NPU_ARCH__ == 2201
        uint32_t first=begin/64*64,length=Stride()-first;
        Mul(tmp2[first],a[first],b[first],length);PipeBarrier<PIPE_V>();
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(length));
        vcadd((__ubuf__ float*)red.GetPhyAddr(),(__ubuf__ float*)tmp2[first].GetPhyAddr(),1,1,1,8,1);
        VectorToScalar();float result=GetReduceRepeatSumSpr<float>();
        SetMaskNorm();ResetMask();return result;
#else
        return Dot(a,b);
#endif
    }
    // Eight rows by eight columns per 256-byte SIMD tile. This is a
    // physical UB layout only; GM tensors and all inverse work stay compact.
    __aicore__ inline void PackMatrix(){
        packedMatrix=!ResidentFamily() && Stride()%64==0 &&
            (Stride()/64>=(count+7)/8+2 || count*8>4*Stride());
        if(!packedMatrix)return;
        uint32_t padded=(count+7)/8*8;
        Duplicate(prod,0.0f,padded*Stride());PipeBarrier<PIPE_V>();SetVectorMask<float>(8);
        for(uint32_t r=0;r<count;++r)
            Copy<float,false>(prod[(r/8)*8*Stride()+(r%8)*8],mat[r*Stride()],8,uint8_t(Stride()/8),CopyRepeatParams{1,1,8,1});
        Barrier();DataCopy(mat,prod,padded*Stride());Barrier();
    }
    __aicore__ inline void ReadPackedRow(LocalTensor<float> dst,uint32_t r){
        if(packedMatrix){
            Copy(dst,mat[(r/8)*8*Stride()+(r%8)*8],8,uint8_t(Stride()/8),CopyRepeatParams{1,1,1,8});
        }else DataCopy(dst,mat[r*Stride()],Stride());
    }
    __aicore__ inline void UnpackMatrix(){
        if(!packedMatrix)return;
        for(uint32_t r=0;r<count;++r)ReadPackedRow(prod[r*Stride()],r);
        Barrier();DataCopy(mat,prod,count*Stride());Barrier();packedMatrix=false;
    }
    template<uint32_t Width>
    __aicore__ inline void AdjacentLevel(LocalTensor<float> dst,LocalTensor<float> src,uint32_t chunks){
        // Independent adjacent sums share a SIMD repeat. A block holds the
        // same eight row results; changing the instruction mask preserves
        // each scalar pair and the complete three-level reduction tree.
        SetVectorMask<float>(8*Width);
        for(uint32_t first=0;first<chunks;first+=255){
            uint8_t repeats=uint8_t(chunks-first<255?chunks-first:255);
            Add<float,false>(dst[first*Width*8],src[first*Width*16],src[first*Width*16+8],8*Width,repeats,BinaryRepeatParams{1,2,2,uint8_t(Width),uint8_t(2*Width),uint8_t(2*Width)});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DotPackedRows(LocalTensor<float> vector,uint32_t begin){
        uint32_t blocks=(count+7)/8,first=begin/64*64,width=Stride()-first;
        Duplicate(w,0.0f,blocks*8);PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
        for(uint32_t rb=0;rb<blocks;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            Mul<float,false>(prod[off],mat[off],vector[first],64,uint8_t((Stride()-first)/8),BinaryRepeatParams{1,1,0,8,8,1});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t rb=0;rb<blocks;++rb)
            BlockReduceSum<float,false>(pcrLeft[rb*width],prod[rb*8*Stride()+first*8],int32_t((Stride()-first)/8),64,1,1,8);
        PipeBarrier<PIPE_V>();
        // VCGADD followed by this adjacent binary tree has the exact VCADD
        // 64-element order. Keep the original left fold between 64-column
        // chunks, including its initial zero.
        uint32_t chunks=blocks*(width/64);
        AdjacentLevel<4>(prod,pcrLeft,chunks);
        AdjacentLevel<2>(pcrLeft,prod,chunks);
        AdjacentLevel<1>(prod,pcrLeft,chunks);
        SetVectorMask<float>(8);
        for(uint32_t j=0;j<width/64;++j){
            Add<float,false>(w,w,prod[j*8],8,uint8_t(blocks),BinaryRepeatParams{1,1,1,1,1,uint8_t(width/64)});
            PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(64);
    }
    __aicore__ inline void UpdatePacked(uint32_t k){
        uint32_t padded=(count+7)/8*8,first=(k+1)/8*8;
        if(directCoefficients){
            Brcb(pcrLeft,w[base],uint8_t(padded/8),{1,8});Brcb(pcrBand,v[base],uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
        }else{
        Gather(tmp2,w,piv.template ReinterpretCast<uint32_t>(),0,padded);PipeBarrier<PIPE_V>();
        Brcb(pcrLeft,tmp2,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
        Gather(tmp2,v,piv.template ReinterpretCast<uint32_t>(),0,padded);PipeBarrier<PIPE_V>();
        Brcb(pcrBand,tmp2,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(64);
        uint32_t firstBlock=base>k+1?0:(k+1-base)/8;
        uint8_t repeats=uint8_t((Stride()-first)/8);
        for(uint32_t rb=firstBlock;rb<padded/8;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            Mul<float,false>(prod[off],v[first],pcrLeft[rb*64],64,repeats,BinaryRepeatParams{1,0,1,8,1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t rb=firstBlock;rb<padded/8;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            MulAddDst<float,float,false>(prod[off],w[first],pcrBand[rb*64],64,repeats,BinaryRepeatParams{1,0,1,8,1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t rb=firstBlock;rb<padded/8;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            Sub<float,false>(mat[off],mat[off],prod[off],64,repeats,BinaryRepeatParams{1,1,1,8,8,8});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DotRows(LocalTensor<float> vector,uint32_t begin=0){
        if(packedMatrix){DotPackedRows(vector,begin);return;}
        if(Stride()%64==0 && count<=64){
            Duplicate(w,0.0f,64);Duplicate(tmp2,0.0f,64);PipeBarrier<PIPE_V>();
            const uint32_t firstColumn=begin/64*64,fullEnd=Order()/64*64;
            SetVectorMask<float>(64);
            for(uint32_t col=firstColumn;col<fullEnd;col+=64){
                Mul<float,false>(prod[col],mat[col],vector[col],64,uint8_t(count),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});PipeBarrier<PIPE_V>();
                WholeReduceSum<float,false>(tmp2,prod[col],64,uint8_t(count),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
                Add<float,false>(w,w,tmp2,64,1,BinaryRepeatParams{1,1,1,8,8,8});
            }
            const uint32_t tailColumns=Order()%64;
            if(tailColumns && firstColumn<=fullEnd){
                SetVectorMask<float>(tailColumns);
                Mul<float,false>(prod[fullEnd],mat[fullEnd],vector[fullEnd],tailColumns,uint8_t(count),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});PipeBarrier<PIPE_V>();
                WholeReduceSum<float,false>(tmp2,prod[fullEnd],tailColumns,uint8_t(count),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
                SetVectorMask<float>(64);
                Add<float,false>(w,w,tmp2,64,1,BinaryRepeatParams{1,1,1,8,8,8});
            }
            PipeBarrier<PIPE_V>();return;
        }
        uint32_t len=(count+7)/8*8;
        Duplicate(w,0.0f,len);Duplicate(tmp2,0.0f,len);PipeBarrier<PIPE_V>();
        for(uint32_t col=begin/64*64;col<Stride();col+=64){
            uint64_t mask=Stride()-col<64?Stride()-col:64;
            Mul(prod[col],mat[col],vector[col],mask,uint8_t(count),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tmp2,prod[col],mask,uint8_t(count),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
            Add(w,w,tmp2,len);
        }
        VectorToScalar();
    }
    __aicore__ inline ExtendedFloat ReadExtended(uint32_t i){i*=2;return ExtendedFloat(wideBuf.GetValue(i),wideBuf.GetValue(i+1),0);}
    __aicore__ inline void WriteExtended(uint32_t i,ExtendedFloat x){i*=2;wideBuf.SetValue(i,x.h);wideBuf.SetValue(i+1,x.m);}
    __aicore__ inline ExtendedFloat RootExtended(ExtendedFloat x){
        if(x.Zero() || x.Negative())return ExtendedFloat(0);
        ExtendedFloat y(Root(x.h));for(uint32_t i=0;i<1;++i){ExtendedFloat r=x-y*y;y=y+r/(y.Scale(2));}return y;
    }
    __aicore__ inline ExtendedFloat Reflector(uint32_t j){return ExtendedFloat(v.GetValue(j),tmp2.GetValue(j),0);}
    __aicore__ inline ExtendedFloat ProductRow(uint32_t j){return ExtendedFloat(w.GetValue(j),red.GetValue(j),0);}
    __aicore__ inline void ReduceExtended(){
        if(active)for(uint32_t r=0;r<count;++r)for(uint32_t j=0;j<Order();++j)WriteExtended(r*Order()+j,ExtendedFloat(mat.GetValue(r*Stride()+j)));
        for(uint32_t k=0;k<Order();++k){
            if(active && k>=base && k<base+count){
                uint32_t row=k-base;ExtendedFloat diag=ReadExtended(row*Order()+k),off(0),tau(0);
                Duplicate(v,0.0f,Stride());Duplicate(tmp2,0.0f,Stride());Barrier();
                if(k+1<Order()){
                    ExtendedFloat a=ReadExtended(row*Order()+k+1),norm(0);off=a;
                    for(uint32_t j=k+2;j<Order();++j){ExtendedFloat x=ReadExtended(row*Order()+j);norm=norm+x*x;}
                    v.SetValue(k+1,1);
                    if(!norm.Zero()){
                        off=RootExtended(a*a+norm);if(!a.Negative())off=-off;
                        tau=(off-a)/off;ExtendedFloat scale=ExtendedFloat(1)/(a-off);
                        for(uint32_t j=k+2;j<Order();++j){ExtendedFloat x=ReadExtended(row*Order()+j)*scale;v.SetValue(j,x.h);tmp2.SetValue(j,x.m);}
                    }
                    Barrier();DataCopy(hist[k*Stride()],v,Stride());DataCopy(hist[(Order()+k)*Stride()],tmp2,Stride());Barrier();
                }
                param.SetValue(0,diag.h);param.SetValue(1,diag.m);param.SetValue(2,off.h);param.SetValue(3,off.m);param.SetValue(4,tau.h);param.SetValue(5,tau.m);
                Barrier();DataCopy(gparam,param,8);Barrier();
            }
            SyncAll();
            if(active){
                DataCopy(param,gparam,8);if(k+1<Order()){DataCopy(v,hist[k*Stride()],Stride());DataCopy(tmp2,hist[(Order()+k)*Stride()],Stride());}Barrier();
                d.SetValue(k,param.GetValue(0));e.SetValue(k,param.GetValue(2));beta.SetValue(k,param.GetValue(4));
                ExtendedFloat tau(param.GetValue(4),param.GetValue(5),0);
                if(k+1<Order() && !tau.Zero()){
                    for(uint32_t r=0;r<count;++r){ExtendedFloat x(0);
                        if(base+r>k){for(uint32_t j=k+1;j<Order();++j)x=x+ReadExtended(r*Order()+j)*Reflector(j);x=x*tau;}
                        w.SetValue(r,x.h);red.SetValue(r,x.m);
                    }
                    Barrier();DataCopyPad(gw[base],w,DataCopyExtParams{1,count*4,0,0,0});DataCopyPad(gw[Stride()+base],red,DataCopyExtParams{1,count*4,0,0,0});Barrier();
                }
            }
            SyncAll();
            if(active && k+1<Order()){
                ExtendedFloat tau(param.GetValue(4),param.GetValue(5),0);
                if(!tau.Zero()){
                    DataCopy(w,gw,Stride());DataCopy(red,gw[Stride()],Stride());Barrier();
                    ExtendedFloat dot(0);for(uint32_t j=k+1;j<Order();++j)dot=dot+Reflector(j)*ProductRow(j);
                    ExtendedFloat correction=ExtendedFloat(-0.5f)*tau*dot;
                    for(uint32_t j=k+1;j<Order();++j){ExtendedFloat x=ProductRow(j)+correction*Reflector(j);w.SetValue(j,x.h);red.SetValue(j,x.m);}
                    for(uint32_t r=0;r<count;++r)if(base+r>k){
                        ExtendedFloat wi=ProductRow(base+r),vi=Reflector(base+r);
                        for(uint32_t j=k+1;j<Order();++j){ExtendedFloat x=ReadExtended(r*Order()+j)-(Reflector(j)*wi+ProductRow(j)*vi);WriteExtended(r*Order()+j,x);}
                    }
                }
            }
        }
        if(active){Barrier();DataCopy(origD,d,Stride());DataCopy(origE,e,Stride());Barrier();}
    }
    // Every active AIV owns a complete small matrix. Vector lanes are rows;
    // the column fold keeps the original expansion operation order.
    __aicore__ inline void SmallCopy(LocalTensor<float> a,LocalTensor<float> b){
        Copy<float,false>(a,b,Stride(),1,CopyRepeatParams{1,1,8,8});
    }
    __aicore__ inline void SmallAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c){
        Add<float,false>(a,b,c,Stride(),1,BinaryRepeatParams{1,1,1,8,8,8});
    }
    __aicore__ inline void SmallSub(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c){
        Sub<float,false>(a,b,c,Stride(),1,BinaryRepeatParams{1,1,1,8,8,8});
    }
    __aicore__ inline void SmallMuls(LocalTensor<float> a,LocalTensor<float> b,float x){
        Muls<float,false>(a,b,x,Stride(),1,UnaryRepeatParams{1,1,8,8});
    }
    template<bool Left>
    __aicore__ inline void SmallScale(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,ExtendedFloat b){
        auto scratch=smallStorage.template Get<float>()[6*Order()*Stride()];
        auto hi=scratch[4*Stride()],lo=scratch[5*Stride()];
        auto t=scratch[6*Stride()],u=scratch[7*Stride()];
        float z=4097.0f*b.h,bh=z-(z-b.h),bl=b.h-bh;
        SmallMuls(dh,ah,b.h);SmallMuls(t,ah,4097.0f);PipeBarrier<PIPE_V>();
        SmallSub(hi,t,ah);PipeBarrier<PIPE_V>();SmallSub(hi,t,hi);PipeBarrier<PIPE_V>();
        SmallSub(lo,ah,hi);SmallMuls(dm,hi,bh);PipeBarrier<PIPE_V>();
        SmallSub(dm,dm,dh);PipeBarrier<PIPE_V>();
        if constexpr(Left){
            SmallMuls(t,lo,bh);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
            SmallMuls(t,hi,bl);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        }else{
            SmallMuls(t,hi,bl);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
            SmallMuls(t,lo,bh);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        }
        SmallMuls(t,lo,bl);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        if constexpr(Left){SmallMuls(t,am,b.h);SmallMuls(u,ah,b.m);}
        else{SmallMuls(t,ah,b.m);SmallMuls(u,am,b.h);}
        PipeBarrier<PIPE_V>();SmallAdd(t,t,u);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        SmallAdd(t,dh,dm);PipeBarrier<PIPE_V>();SmallSub(u,t,dh);PipeBarrier<PIPE_V>();
        SmallSub(dm,dm,u);SmallCopy(dh,t);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void SmallMul(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c){
        Mul<float,false>(a,b,c,Stride(),1,BinaryRepeatParams{1,1,1,8,8,8});
    }
    __aicore__ inline void SmallProduct(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        auto scratch=smallStorage.template Get<float>()[6*Order()*Stride()];
        auto hi=scratch[4*Stride()],lo=scratch[5*Stride()];
        auto bhi=scratch[6*Stride()],blo=scratch[7*Stride()],t=scratch[8*Stride()];
        SetVectorMask<float>(Stride());
        SmallMul(dh,ah,bh);SmallMuls(hi,ah,4097.0f);SmallMuls(bhi,bh,4097.0f);PipeBarrier<PIPE_V>();
        SmallSub(lo,hi,ah);SmallSub(blo,bhi,bh);PipeBarrier<PIPE_V>();
        SmallSub(hi,hi,lo);SmallSub(bhi,bhi,blo);PipeBarrier<PIPE_V>();
        SmallSub(lo,ah,hi);SmallSub(blo,bh,bhi);PipeBarrier<PIPE_V>();
        SmallMul(dm,hi,bhi);PipeBarrier<PIPE_V>();SmallSub(dm,dm,dh);PipeBarrier<PIPE_V>();
        SmallMul(t,hi,blo);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        SmallMul(t,lo,bhi);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        SmallMul(t,lo,blo);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        SmallMul(t,ah,bm);SmallMul(hi,am,bh);PipeBarrier<PIPE_V>();SmallAdd(t,t,hi);PipeBarrier<PIPE_V>();
        SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();SmallAdd(t,dh,dm);PipeBarrier<PIPE_V>();
        SmallSub(hi,t,dh);PipeBarrier<PIPE_V>();SmallSub(dm,dm,hi);SmallCopy(dh,t);PipeBarrier<PIPE_V>();
    }
    template<bool Minus>
    __aicore__ inline void SmallSum(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        auto scratch=smallStorage.template Get<float>()[6*Order()*Stride()];
        auto t=scratch[4*Stride()],q=scratch[5*Stride()],a=scratch[6*Stride()];
        auto b=scratch[7*Stride()],ee=scratch[8*Stride()];
        if constexpr(Minus)SmallSub(t,ah,bh);else SmallAdd(t,ah,bh);
        PipeBarrier<PIPE_V>();SmallSub(q,t,ah);PipeBarrier<PIPE_V>();
        SmallSub(a,t,q);PipeBarrier<PIPE_V>();SmallSub(a,ah,a);
        if constexpr(Minus)SmallMuls(b,bh,-1.0f);else SmallCopy(b,bh);
        PipeBarrier<PIPE_V>();SmallSub(b,b,q);PipeBarrier<PIPE_V>();SmallAdd(ee,a,b);PipeBarrier<PIPE_V>();
        SmallAdd(ee,ee,am);PipeBarrier<PIPE_V>();
        if constexpr(Minus)SmallSub(ee,ee,bm);else SmallAdd(ee,ee,bm);
        PipeBarrier<PIPE_V>();SmallAdd(dh,t,ee);PipeBarrier<PIPE_V>();
        SmallSub(q,dh,t);PipeBarrier<PIPE_V>();SmallSub(dm,ee,q);PipeBarrier<PIPE_V>();
    }
    // Whole-matrix products amortize the expansion arithmetic over SIMD
    // repeats. Columns remain contiguous; row folds below remain ordered.
    __aicore__ inline void DenseExpand(LocalTensor<float> dst,LocalTensor<float> src){
        auto t=smallStorage.template Get<float>()[6*Order()*Stride()];
        Brcb(t,src,uint8_t((Order()+7)/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(8);
        for(uint32_t row=0;row<Stride();row+=8)
            Copy<float,false>(dst[row],t,8,uint8_t(Order()),CopyRepeatParams{1,1,uint16_t(Stride()/8),1});
        PipeBarrier<PIPE_V>();SetVectorMask<float>(Stride());
    }
    __aicore__ inline void DenseAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,uint8_t cs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Add<float,false>(a,b,c,Stride(),uint8_t(Order()),BinaryRepeatParams{1,1,1,pitch,pitch,cs==255?pitch:cs});
    }
    __aicore__ inline void DenseSub(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,uint8_t cs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Sub<float,false>(a,b,c,Stride(),uint8_t(Order()),BinaryRepeatParams{1,1,1,pitch,pitch,cs==255?pitch:cs});
    }
    __aicore__ inline void DenseMul(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,uint8_t cs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Mul<float,false>(a,b,c,Stride(),uint8_t(Order()),BinaryRepeatParams{1,1,1,pitch,pitch,cs==255?pitch:cs});
    }
    __aicore__ inline void DenseMuls(LocalTensor<float> a,LocalTensor<float> b,float x,uint8_t bs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Muls<float,false>(a,b,x,Stride(),uint8_t(Order()),UnaryRepeatParams{1,1,pitch,bs==255?pitch:bs});
    }
    __aicore__ inline void DenseCopy(LocalTensor<float> a,LocalTensor<float> b){
        uint16_t pitch=uint16_t(Stride()/8);
        Copy<float,false>(a,b,Stride(),uint8_t(Order()),CopyRepeatParams{1,1,pitch,pitch});
    }
    template<bool RowB>
    __aicore__ inline void DenseProduct(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        uint32_t dense=Order()*Stride();auto scratch=smallStorage.template Get<float>()[6*dense];
        auto hi=scratch,lo=scratch[dense],bhi=scratch[2*dense],blo=scratch[3*dense],t=scratch[4*dense];
        constexpr uint8_t bs=RowB?0:255;
        DenseMul(dh,ah,bh,bs);DenseMuls(hi,ah,4097.0f);DenseMuls(bhi,bh,4097.0f,bs);PipeBarrier<PIPE_V>();
        DenseSub(lo,hi,ah);DenseSub(blo,bhi,bh,bs);PipeBarrier<PIPE_V>();
        DenseSub(hi,hi,lo);DenseSub(bhi,bhi,blo);PipeBarrier<PIPE_V>();
        DenseSub(lo,ah,hi);
        // bh's repeated view becomes a full contiguous split, so subtraction
        // uses an explicit row repeat when the original b was a row vector.
        DenseMuls(blo,bh,1.0f,bs);PipeBarrier<PIPE_V>();DenseSub(blo,blo,bhi);PipeBarrier<PIPE_V>();
        DenseMul(dm,hi,bhi);PipeBarrier<PIPE_V>();DenseSub(dm,dm,dh);PipeBarrier<PIPE_V>();
        DenseMul(t,hi,blo);PipeBarrier<PIPE_V>();DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        DenseMul(t,lo,bhi);PipeBarrier<PIPE_V>();DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        DenseMul(t,lo,blo);PipeBarrier<PIPE_V>();DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        DenseMul(t,ah,bm,bs);DenseMul(hi,am,bh,bs);PipeBarrier<PIPE_V>();DenseAdd(t,t,hi);PipeBarrier<PIPE_V>();
        DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();DenseAdd(t,dh,dm);PipeBarrier<PIPE_V>();
        DenseSub(hi,t,dh);PipeBarrier<PIPE_V>();DenseSub(dm,dm,hi);DenseCopy(dh,t);PipeBarrier<PIPE_V>();
    }
    template<bool Minus>
    __aicore__ inline void DenseSum(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        uint32_t dense=Order()*Stride();auto scratch=smallStorage.template Get<float>()[6*dense];
        auto t=scratch,q=scratch[dense],a=scratch[2*dense],b=scratch[3*dense],ee=scratch[4*dense];
        if constexpr(Minus)DenseSub(t,ah,bh);else DenseAdd(t,ah,bh);
        PipeBarrier<PIPE_V>();DenseSub(q,t,ah);PipeBarrier<PIPE_V>();DenseSub(a,t,q);PipeBarrier<PIPE_V>();
        DenseSub(a,ah,a);if constexpr(Minus)DenseMuls(b,bh,-1.0f);else DenseCopy(b,bh);
        PipeBarrier<PIPE_V>();DenseSub(b,b,q);PipeBarrier<PIPE_V>();DenseAdd(ee,a,b);PipeBarrier<PIPE_V>();
        DenseAdd(ee,ee,am);PipeBarrier<PIPE_V>();if constexpr(Minus)DenseSub(ee,ee,bm);else DenseAdd(ee,ee,bm);
        PipeBarrier<PIPE_V>();DenseAdd(dh,t,ee);PipeBarrier<PIPE_V>();DenseSub(q,dh,t);PipeBarrier<PIPE_V>();DenseSub(dm,ee,q);PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadResidentTriangle(LocalTensor<float> matrix,LocalTensor<float> raw){
        auto indices=piv.template ReinterpretCast<int32_t>();
        CreateIndicesWithArena(indices,int32_t(0),Stride());PipeBarrier<PIPE_V>();
        Cast(param,indices,RoundMode::CAST_NONE,Stride());PipeBarrier<PIPE_V>();
        if(upper){Muls(param,param,-1.0f,Stride());PipeBarrier<PIPE_V>();Adds(param,param,float(int32_t(Order()-1)),Stride());PipeBarrier<PIPE_V>();}
        for(uint32_t i=0;i<Order();++i){
            Duplicate(v,float(int32_t(upper?Order()-1-i:i)),Stride());PipeBarrier<PIPE_V>();
            Min(w,v,param,Stride());Max(tmp,v,param,Stride());PipeBarrier<PIPE_V>();
            Muls(tmp2,upper?w:tmp,float(int32_t(Order())),Stride());PipeBarrier<PIPE_V>();
            Add(tmp2,tmp2,upper?tmp:w,Stride());PipeBarrier<PIPE_V>();
            Muls(tmp2,tmp2,4.0f,Stride());PipeBarrier<PIPE_V>();
            Cast(indices,tmp2,RoundMode::CAST_RINT,Stride());PipeBarrier<PIPE_V>();
            // Only Order() valid indices are read. The caller zeroed row padding.
            Gather(matrix[i*Stride()],raw,indices.template ReinterpretCast<uint32_t>(),0,Order());PipeBarrier<PIPE_V>();
        }
        Barrier();
    }

    __aicore__ __attribute__((noinline)) void ReduceReplicatedSmall(uint32_t batchIndex,bool synchronize=true){
        if(active){
            auto mh=smallStorage.template Get<float>(),ml=mh[Order()*Stride()],scratch=mh[2*Order()*Stride()];
            // The unused matrix region is large enough for the compact raw
            // input, and is dead before any vector scratch is reused.
            DataCopyPad(scratch,input[uint64_t(batchIndex)*Order()*Order()],DataCopyExtParams{1,Order()*Order()*4,0,0,0},
                DataCopyPadExtParams<float>{false,0,0,0});Barrier();
            Duplicate(mh,0.0f,2*Order()*Stride());Barrier();
            LoadResidentTriangle(mh,scratch);
            uint32_t dense=Order()*Stride();auto ph=mh[2*dense],pm=mh[3*dense],qh=mh[4*dense],qm=mh[5*dense];
            auto vh=mh[11*dense],vm=mh[12*dense],wh=mh[13*dense],wm=mh[14*dense];
            auto sh=mh[6*dense],sm=sh[Stride()];
            for(uint32_t k=0;k<Order();++k){
                Barrier();ExtendedFloat diag(mh.GetValue(k*Stride()+k),ml.GetValue(k*Stride()+k),0);
                ExtendedFloat off(0),tau(0);
                Duplicate(v,0.0f,Stride());Duplicate(tmp2,0.0f,Stride());Barrier();
                if(k+1<Order()){
                    ExtendedFloat a(mh.GetValue((k+1)*Stride()+k),ml.GetValue((k+1)*Stride()+k),0),norm(0);off=a;
                    for(uint32_t j=k+2;j<Order();++j){
                        ExtendedFloat x(mh.GetValue(j*Stride()+k),ml.GetValue(j*Stride()+k),0);norm=norm+x*x;
                    }
                    v.SetValue(k+1,1);
                    if(!norm.Zero()){
                        off=RootExtended(a*a+norm);if(!a.Negative())off=-off;
                        tau=(off-a)/off;ExtendedFloat scale=ExtendedFloat(1)/(a-off);
                        for(uint32_t j=k+2;j<Order();++j){
                            ExtendedFloat x=ExtendedFloat(mh.GetValue(j*Stride()+k),ml.GetValue(j*Stride()+k),0)*scale;
                            v.SetValue(j,x.h);tmp2.SetValue(j,x.m);
                        }
                    }
                    Barrier();
                    if constexpr(PrivateSmall){DataCopy(privateHistory[k*Stride()],v,Stride());DataCopy(privateHistory[(Order()+k)*Stride()],tmp2,Stride());Barrier();}
                    else if(lane+1==team){DataCopy(hist[k*Stride()],v,Stride());DataCopy(hist[(Order()+k)*Stride()],tmp2,Stride());Barrier();}
                }
                d.SetValue(k,diag.h);e.SetValue(k,off.h);beta.SetValue(k,tau.h);
                if(k+1<Order() && !tau.Zero()){
                    Duplicate(w,0.0f,Stride());Duplicate(red,0.0f,Stride());PipeBarrier<PIPE_V>();
                    DenseExpand(vh,v);DenseExpand(vm,tmp2);DenseProduct<false>(ph,pm,mh,ml,vh,vm);
                    for(uint32_t j=k+1;j<Order();++j)SmallSum<false>(w,red,w,red,ph[j*Stride()],pm[j*Stride()]);
                    SmallScale<false>(sh,sm,w,red,tau);SmallCopy(w,sh);SmallCopy(red,sm);Barrier();
                    for(uint32_t j=0;j<=k;++j){w.SetValue(j,0);red.SetValue(j,0);}
                    ExtendedFloat dot(0);for(uint32_t j=k+1;j<Order();++j)dot=dot+Reflector(j)*ProductRow(j);
                    ExtendedFloat correction=ExtendedFloat(-0.5f)*tau*dot;
                    for(uint32_t j=k+1;j<Order();++j){ExtendedFloat x=ProductRow(j)+correction*Reflector(j);w.SetValue(j,x.h);red.SetValue(j,x.m);}
                    Barrier();DenseExpand(wh,w);DenseExpand(wm,red);
                    DenseProduct<true>(ph,pm,vh,vm,w,red);DenseProduct<true>(qh,qm,wh,wm,v,tmp2);
                    DenseSum<false>(ph,pm,ph,pm,qh,qm);DenseSum<true>(mh,ml,mh,ml,ph,pm);
                    Barrier();ResetMask();
                }
            }
            DataCopy(origD,d,Stride());DataCopy(origE,e,Stride());Barrier();
        }
        // The full calculation is invocation-local. Publish the completed
        // reflectors before any AIV begins the original backtransform.
        if constexpr(!PrivateSmall){if(synchronize)SyncAll();}
    }

    __aicore__ inline float SmallNativeRoot(float value){
        if(value<=0.0f)return 0.0f;
        // param/red are dead between scalar operations in the private
        // FP32 reduction and QL recurrence, including rotation replay.
        param.SetValue(0,value);
        event_t ready=ImmediateEvent<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(ready);WaitFlag<HardEvent::S_V>(ready);
        Sqrt(red,param,1);VectorToScalar();return red.GetValue(0);
    }

    // Complete invocation-local FP32 Householder reduction for matrices
    // whose full small matrix fits the already reserved private UB arena.
    __aicore__ __attribute__((noinline)) void ReduceReplicatedFloatSmall(uint32_t b){
        if(active){
            uint32_t dense=Order()*Stride(),padded=(Order()+7)/8*8;
            auto a=smallStorage.template Get<float>(),product=a[dense],raw=a[2*dense];
            DataCopyPad(raw,input[uint64_t(b)*Order()*Order()],DataCopyExtParams{1,Order()*Order()*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});Barrier();
            Duplicate(a,0.0f,dense);Barrier();
            LoadResidentTriangle(a,raw);
            Barrier();
            // Bound the small FP32 path's absolute backward error and avoid
            // squared-norm underflow/overflow. More demanding magnitudes
            // retain the existing complete two-float reduction.
            AscendC::Abs(product,a,dense);Duplicate(w,0.0f,Stride());PipeBarrier<PIPE_V>();
            WholeReduceSum(w,product,Stride(),uint8_t(Order()),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
            if constexpr(DirectUb)WholeReduceMax(red,w,Stride(),1,1,1,8);else ReduceMax(red,w,tmp,Stride(),false);VectorToScalar();float rowBound=red.GetValue(0);
            smallRowBound=rowBound;
            smallQLNeedsCheck=rowBound>64.0f && rowBound<=281474976710656.0f && rows<=2;
            smallFloatQL=(rowBound>16.0f && rowBound<=64.0f) || smallQLNeedsCheck;
            if((rowBound>64.0f && !smallQLNeedsCheck) || (rowBound>0.0f && rowBound<1e-10f)){
                ReduceReplicatedSmall(b);return;
            }
            auto cw=wideBuf,cv=wideBuf[padded*8];
            for(uint32_t k=0;k<Order();++k){
                d.SetValue(k,a.GetValue(k*Stride()+k));e.SetValue(k,0.0f);beta.SetValue(k,0.0f);
                if(k+2<Order()){
                    DataCopy(v,a[k*Stride()],Stride());Barrier();Duplicate(v,0.0f,k+1);PipeBarrier<PIPE_V>();
                    float squared=Dot(v,v),norm=smallFloatQL?SmallNativeRoot(squared):Root(squared),x0=v.GetValue(k+1);
                    e.SetValue(k,x0>=0?-norm:norm);
                    if(norm>0){Muls(v,v,1.0f/norm,Stride());VectorToScalar();v.SetValue(k+1,x0/norm+(x0>=0?1.0f:-1.0f));Barrier();}
                    if constexpr(PrivateSmall)DataCopy(privateHistory[k*Stride()],v,Stride());
                    else if(lane+1==team)DataCopy(hist[k*Stride()],v,Stride());
                    if(norm>0){
                        // v=x/||x||+sign(x0)e1, hence v^Tv=2|v1|.
                        // The fixed tile avoids a second SIMD reduction and
                        // its vector-to-scalar dependency for each reflector.
                        float tau;
                        tau=1.0f/Abs(v.GetValue(k+1));
                        beta.SetValue(k,tau);
                        Mul(product,a,v,Stride(),uint8_t(Order()),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});PipeBarrier<PIPE_V>();
                        Duplicate(w,0.0f,Stride());PipeBarrier<PIPE_V>();
                        WholeReduceSum(w,product,Stride(),uint8_t(Order()),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
                        Duplicate(w,0.0f,k+1);PipeBarrier<PIPE_V>();Muls(w,w,tau,Stride());PipeBarrier<PIPE_V>();
                        float correction=-0.5f*tau*Dot(v,w);Axpy(w,v,correction,Stride());PipeBarrier<PIPE_V>();
                        Brcb(cw,w,uint8_t(padded/8),{1,8});Brcb(cv,v,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
                        Mul(product,v,cw,Stride(),uint8_t(Order()),BinaryRepeatParams{1,1,0,uint8_t(Stride()/8),0,1});PipeBarrier<PIPE_V>();
                        MulAddDst(product,w,cv,Stride(),uint8_t(Order()),BinaryRepeatParams{1,1,0,uint8_t(Stride()/8),0,1});PipeBarrier<PIPE_V>();
                        Sub(a,a,product,dense);PipeBarrier<PIPE_V>();
                    }
                }else if(k+1<Order())e.SetValue(k,a.GetValue(k*Stride()+k+1));
                Barrier();
            }
            DataCopy(origD,d,Stride());DataCopy(origE,e,Stride());Barrier();
            if(smallFloatQL){
                count=lane<Order()?(Order()-1-lane)/team+1:0;
                smallQLSolved=SolvePrivateSmallQL();
                if(smallQLSolved && smallQLNeedsCheck){
                    for(uint32_t i=0;i<Order();++i){
                        float value=d.GetValue(i);
                        float floor=1e-12f*float(int32_t(Order()*Order()))*smallRowBound+1.1920928955078125e-7f*Abs(value);
                        if(floor>0.25f*(1e-4f+1e-4f*Abs(value)))smallQLSolved=false;
                    }
                }
                if(!smallQLSolved)ReduceReplicatedSmall(b,false);
                if(smallQLSolved && smallQLNeedsCheck && lane==0)PublishSmallMode(true);
            }
        }
        if constexpr(!PrivateSmall)SyncAll();
    }

    __aicore__ inline float TailDot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> product,LocalTensor<float> result,uint32_t length){
        Mul(product,a,b,length);PipeBarrier<PIPE_V>();
#if __NPU_ARCH__ == 2201
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(length));
        vcadd((__ubuf__ float*)result.GetPhyAddr(),(__ubuf__ float*)product.GetPhyAddr(),1,1,1,8,1);
        VectorToScalar();float value=GetReduceRepeatSumSpr<float>();SetMaskNorm();ResetMask();return value;
#else
        ReduceSumWithArena(result,product,product[64],length);VectorToScalar();return result.GetValue(0);
#endif
    }
    __aicore__ __attribute__((noinline)) void PrivateTail(uint32_t first){
        uint32_t size=Order()-first,width=Stride()-first,pitch=width>64?128:64;
        uint32_t batchRows=4*Stride()+1344<=privateCapacity?64:32;
        LocalTensor<float> td=wideBuf,te=wideBuf[Stride()],tb=wideBuf[2*Stride()];
        LocalTensor<float> fullV=wideBuf[3*Stride()],tv=fullV[first];
        LocalTensor<float> tw=wideBuf[4*Stride()],pivot=wideBuf[4*Stride()+128];
        LocalTensor<float> result=wideBuf[4*Stride()+256];
        LocalTensor<float> cw=wideBuf[4*Stride()+320],cv=cw[batchRows*8];
        LocalTensor<float> product=storage.template Get<float>()[width*pitch];
        // Preserve all completed metadata and the separately updated pivot
        // before recycling the resident matrix and inverse work storage.
        DataCopy(td,d,Stride());DataCopy(te,e,Stride());DataCopy(tb,beta,Stride());
        DataCopy(pivot,pack[first],width);Duplicate(fullV,0.0f,Stride());Barrier();
        LocalTensor<float> tail=storage.template Get<float>();
        Duplicate(tail,0.0f,width*pitch);Barrier();
        auto stage=hist[CubeMode?(Order()+5)*Stride():first*Stride()];
        if(width==pitch)DataCopy(tail,stage,size*pitch);
        else DataCopyPad(tail,stage,DataCopyExtParams{uint16_t(size),width*4,0,(pitch-width)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
        Barrier();
        for(uint32_t k=0;k<size;++k){
            uint32_t global=first+k,localFirst=(k+1)/64*64;
            td.SetValue(global,pivot.GetValue(k));te.SetValue(global,0);tb.SetValue(global,0);
            float tau=0;
            if(k+2<size){
                DataCopy(tv,pivot,width);Barrier();Duplicate(tv,0.0f,k+1);Barrier();
                float norm=Root(TailDot(tv[localFirst],tv[localFirst],product,result,width-localFirst));float x0=tv.GetValue(k+1);
                te.SetValue(global,x0>=0?-norm:norm);
                if(norm>0){Muls(tv,tv,1.0f/norm,width);Barrier();tv.SetValue(k+1,x0/norm+(x0>=0?1.0f:-1.0f));Barrier();}
                if constexpr(CubeMode==0){
                    if(Order()<=64)DataCopy(privateHistory[global*Stride()],fullV,Stride());
                    else DataCopy(hist[global*Stride()],fullV,Stride());
                }else DataCopy(hist[global*Stride()],fullV,Stride());
                // v=x/||x||+sign(x0)e1 gives v^T v=2*abs(v1).
                if(norm>0)tau=1.0f/Abs(tv.GetValue(k+1));
                tb.SetValue(global,tau);
            }else if(k+1<size)te.SetValue(global,pivot.GetValue(k+1));
            if(k+1<size){DataCopy(pivot,tail[(k+1)*pitch],width);Barrier();}
            if(k+2<size && tau!=0){
                Duplicate(tw,0.0f,128);PipeBarrier<PIPE_V>();
                for(uint32_t rb=(k+1)/8*8;rb<size;rb+=batchRows){
                    uint8_t live=uint8_t(size-rb<batchRows?size-rb:batchRows);
                    for(uint32_t col=localFirst;col<width;col+=64){
                        uint64_t mask=width-col<64?width-col:64;
                        Mul(product,tail[rb*pitch+col],tv[col],mask,live,BinaryRepeatParams{1,1,1,8,uint8_t(pitch/8),0});PipeBarrier<PIPE_V>();
                        WholeReduceSum(result,product,mask,live,1,1,8);PipeBarrier<PIPE_V>();
                        Add(tw[rb],tw[rb],result,uint32_t(live));PipeBarrier<PIPE_V>();
                    }
                }
                Duplicate(tw,0.0f,k+1);PipeBarrier<PIPE_V>();Muls(tw,tw,tau,width);Barrier();
                float correction=-0.5f*tau*TailDot(tv[localFirst],tw[localFirst],product,result,width-localFirst);
                Axpy(tw,tv,correction,width);Barrier();
                Muls(cw,tv,tw.GetValue(k+1),width);PipeBarrier<PIPE_V>();
                Axpy(cw,tw,tv.GetValue(k+1),width);PipeBarrier<PIPE_V>();
                Sub(pivot,pivot,cw,width);PipeBarrier<PIPE_V>();
                for(uint32_t rb=(k+1)/8*8;rb<size;rb+=batchRows){
                    uint8_t live=uint8_t(size-rb<batchRows?size-rb:batchRows);
                    Brcb(cw,tw[rb],uint8_t((live+7)/8),{1,8});Brcb(cv,tv[rb],uint8_t((live+7)/8),{1,8});PipeBarrier<PIPE_V>();
                    for(uint32_t col=(k+1)/8*8;col<width;col+=64){
                        uint64_t mask=width-col<64?width-col:64;
                        Mul(product,tv[col],cw,mask,live,BinaryRepeatParams{1,1,0,8,0,1});PipeBarrier<PIPE_V>();
                        MulAddDst(product,tw[col],cv,mask,live,BinaryRepeatParams{1,1,0,8,0,1});PipeBarrier<PIPE_V>();
                        Sub(tail[rb*pitch+col],tail[rb*pitch+col],product,mask,live,BinaryRepeatParams{1,1,1,uint8_t(pitch/8),uint8_t(pitch/8),8});PipeBarrier<PIPE_V>();
                    }
                }
            }
            Barrier();
        }
        if constexpr(CubeMode==0){
            // No consumer reads the reflectors until Reduce's final SyncAll.
            // Publish the completed local history with one contiguous DMA.
            if(Order()<=64){DataCopy(hist,privateHistory,(Order()-2)*Stride());Barrier();}
        }
        // These destinations may overlap the now-dead private matrix.
        DataCopy(d,td,Stride());DataCopy(e,te,Stride());DataCopy(beta,tb,Stride());Barrier();
    }
    __aicore__ inline void GatherTail(uint32_t first){
        uint32_t width=Stride()-first;
        if(active){
            auto stage=hist[CubeMode?(Order()+5)*Stride():first*Stride()];
            for(uint32_t r=0;r<count;++r)if(base+r>=first){
                ReadPackedRow(tmp,r);Barrier();
                DataCopy(stage[(base+r-first)*width],tmp[first],width);Barrier();
            }
        }
        SyncAll();
        if(active && lane+1==team)PrivateTail(first);
        packedMatrix=false;
    }
    __aicore__ __attribute__((noinline)) void Reduce(){
        // Replicate the next pivot row. Two workspace rows prevent a fast core
        // from overwriting a row before another core has consumed it.
        auto pivotRows=gw[Stride()];
        auto rowOffsets=piv.template ReinterpretCast<uint32_t>();
        if(active){Duplicate(d,0.0f,Stride());Duplicate(e,0.0f,Stride());Duplicate(beta,0.0f,Stride());Barrier();for(uint32_t r=0;r<(count+7)/8*8;++r)rowOffsets.SetValue(r,(base+r<Order()?base+r:Order()-1)*4);Barrier();}
        if(active && lane==0){
            Duplicate(w,0.0f,Stride());Barrier();DataCopy(gw,w,Stride());DataCopy(gw[2*Stride()],w,Stride());
            if(packedMatrix){ReadPackedRow(tmp,0);Barrier();DataCopy(gparam,tmp,Stride());}
            else DataCopy(gparam,mat,Stride());Barrier();
        }
        SyncAll();
        if(active){DataCopy(pack,gparam,Stride());Barrier();}
        // A three-block tail uses 32-float alignment while avoiding the
        // serial cost of a full128-row private matrix.
        uint32_t privateFirst=privateTailColumnCount==96?Stride()-96:(Order()-1)/64*64;
        for(uint32_t k=0;k<Order();++k){
            if(k==privateFirst){GatherTail(privateFirst);break;}
            // A closed row is no longer used by any following reflector.
            // Its w slot has been explicitly zeroed on the last live step.
            bool working=active && k<base+count;
            if(working){
                d.SetValue(k,pack.GetValue(k));e.SetValue(k,0);beta.SetValue(k,0);
                if(k+2<Order()){
                    DataCopy(v,pack,Stride());Barrier();Duplicate(v,0.0f,k+1);PipeBarrier<PIPE_V>();
                    float norm=Root(DotTail(v,v,k+1));float x0=v.GetValue(k+1),alpha=x0>=0?-norm:norm;
                    e.SetValue(k,alpha);
                    if(norm>0){Muls(v,v,1.0f/norm,Stride());VectorToScalar();v.SetValue(k+1,x0/norm+(x0>=0?1.0f:-1.0f));Barrier();}
                    if(lane+1==team){Barrier();DataCopy(hist[k*Stride()],v,Stride());}
                    if(norm>0)beta.SetValue(k,1.0f/Abs(v.GetValue(k+1)));
                }else if(k+1<Order())e.SetValue(k,pack.GetValue(k+1));
                // Publish the unmodified next row alongside the matrix-vector product.
                if(k+1<Order() && k+1>=base && k+1<base+count){
                    if(packedMatrix){ReadPackedRow(tmp,k+1-base);Barrier();DataCopy(pivotRows[(k%2)*2*Stride()],tmp,Stride());}
                    else DataCopy(pivotRows[(k%2)*2*Stride()],mat[(k+1-base)*Stride()],Stride());
                }
                if(k+2<Order()){
                    if(beta.GetValue(k)!=0){
                        DotRows(v,k+1);Muls(w,w,beta.GetValue(k),(count+7)/8*8);Barrier();
                        if(base<=k){uint32_t z=k+1-base;z=z<count?z:count;Duplicate(w,0.0f,z);Barrier();}
                    }else{Duplicate(w,0.0f,Stride());Barrier();}
                    DataCopyPad(gw[(k%2)*2*Stride()+base],w,DataCopyExtParams{1,count*4,0,0,0});
                }
            }
            // SyncAll begins with PIPE_ALL before publishing its arrival.
            SyncAll();
            if(working){
                // One transfer fetches the matrix-vector result and next pivot row.
                if(k+1<Order())DataCopy(w,gw[(k%2)*2*Stride()],2*Stride());
                Barrier();
                if(k+2<Order() && beta.GetValue(k)!=0){
                    float correction=-0.5f*beta.GetValue(k)*DotTail(v,w,k+1);
                    Axpy(w,v,correction,Stride());VectorToScalar();
                    if(packedMatrix || (Stride()%64==0 && count*8<=4*Stride())){
                        SetVectorMask<float>(64);uint32_t first=(k+1)/64*64;
                        // Apply exactly the same arithmetic to the replicated row.
                        Muls<float,false>(tmp[first],v[first],w.GetValue(k+1),64,uint8_t((Stride()-first)/64),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                        Axpy<float,float,false>(tmp[first],w[first],v.GetValue(k+1),64,uint8_t((Stride()-first)/64),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                        Sub<float,false>(pack[first],pack[first],tmp[first],64,uint8_t((Stride()-first)/64),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                        if(packedMatrix)UpdatePacked(k);
                        else{
                        uint32_t firstRow=base>k?0:k+1-base;firstRow=firstRow<count?firstRow:count;
                        if(firstRow<count){
                            // Source blocks stay 32-byte aligned. Offset the
                            // expanded coefficients by the owner's row prefix.
                            const uint32_t rowPrefix=base%8,alignedBase=base-rowPrefix;
                            const uint32_t coefficientBlocks=(rowPrefix+count+7)/8;
                            Brcb(pack[Stride()],w[alignedBase],uint8_t(coefficientBlocks),{1,8});
                            Brcb(pack[5*Stride()],v[alignedBase],uint8_t(coefficientBlocks),{1,8});PipeBarrier<PIPE_V>();
                            SetVectorMask<float>(64);
                            BinaryRepeatParams br{1,1,0,uint8_t(Stride()/8),0,1};
                            for(uint32_t col=first;col<Order();col+=64){
                                const uint32_t validColumns=Order()-col<64?Order()-col:64;
                                SetVectorMask<float>(validColumns);
                                uint32_t off=firstRow*Stride()+col;
                                Mul<float,false>(prod[off],v[col],pack[Stride()+8*(rowPrefix+firstRow)],validColumns,uint8_t(count-firstRow),br);
                            }
                            PipeBarrier<PIPE_V>();
                            for(uint32_t col=first;col<Order();col+=64){
                                const uint32_t validColumns=Order()-col<64?Order()-col:64;
                                SetVectorMask<float>(validColumns);
                                uint32_t off=firstRow*Stride()+col;
                                MulAddDst<float,float,false>(prod[off],w[col],pack[5*Stride()+8*(rowPrefix+firstRow)],validColumns,uint8_t(count-firstRow),br);
                            }
                            PipeBarrier<PIPE_V>();
                            for(uint32_t col=first;col<Order();col+=64){
                                const uint32_t validColumns=Order()-col<64?Order()-col:64;
                                SetVectorMask<float>(validColumns);
                                uint32_t off=firstRow*Stride()+col;
                                Sub<float,false>(mat[off],mat[off],prod[off],validColumns,uint8_t(count-firstRow),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),uint8_t(Stride()/8)});
                            }
                            PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
                        }
                        }
                    }else{
                        Muls(tmp,v,w.GetValue(k+1),Stride());PipeBarrier<PIPE_V>();Axpy(tmp,w,v.GetValue(k+1),Stride());PipeBarrier<PIPE_V>();Sub(pack,pack,tmp,Stride());PipeBarrier<PIPE_V>();
                        for(uint32_t r=0;r<count;++r)if(base+r>k){auto a=mat[r*Stride()];Muls(tmp,v,w.GetValue(base+r),Stride());PipeBarrier<PIPE_V>();Axpy(tmp,w,v.GetValue(base+r),Stride());PipeBarrier<PIPE_V>();Sub(a,a,tmp,Stride());PipeBarrier<PIPE_V>();}
                    }
                    Barrier();
                }
            }
        }
        // The last row owner remains live throughout the reduction. The
        // last two high reflector rows are unused by the backtransform, so
        // they hold the completed diagonal/off-diagonal for the other cores.
        if(active && lane+1==team){
            DataCopy(hist[(Order()-2)*Stride()],d,Stride());DataCopy(hist[(Order()-1)*Stride()],e,Stride());DataCopy(gparam,beta,Stride());Barrier();
        }
        SyncAll();
        if(active){
            DataCopy(origD,hist[(Order()-2)*Stride()],Stride());DataCopy(origE,hist[(Order()-1)*Stride()],Stride());DataCopy(beta,gparam,Stride());Barrier();
            DataCopy(d,origD,Stride());DataCopy(e,origE,Stride());Barrier();
        }
    }
    // Independent Sturm recurrences occupy vector lanes; the recurrence along
    // the tridiagonal remains sequential, while different eigenvalues run together.
    // Evaluate a complete binary decision tree in otherwise idle SIMD lanes.
    // Every midpoint uses the original binary arithmetic and every rank stops
    // at the same tolerance; several sequential bisections finish per sweep.
    __aicore__ inline float OriginalCommonGrid(float lower,float upperBound,float tolerance){
        uint32_t limit=4*Stride()<64*team?4*Stride():64*team;
        uint32_t branches=1;while(2*branches<=limit)branches*=2;
        uint32_t queries=lane<branches-1?(branches-2-lane)/team+1:0;
        uint32_t lanes=queries>64?64:(queries+7)/8*8,repeats=(queries+63)/64;
        auto mid=pack,p=pack[Stride()],negative=pack[2*Stride()],flags=pack[4*Stride()],scratch=pack[5*Stride()];
        Duplicate(piv,lower,64);Duplicate(fac,upperBound,64);Duplicate(mid,0.0f,((queries+63)/64)*64);Barrier();
        Mul(tmp,origE,origE,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,tmp,uint8_t(Stride()/8),{1,8});
        Brcb(mat,origD,uint8_t(Stride()/8),{1,8});Barrier();
        for(uint32_t r=0;r<queries;++r){
            uint32_t target=lane+r*team+1,left=0,right=branches;float l=lower,h=upperBound;
            while(true){uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(j==target){mid.SetValue(r,value);break;}if(target<j){right=j;h=value;}else{left=j;l=value;}}
        }
        Barrier();SetVectorMask<float>(lanes);
        Sub<float,false>(p,mat,mid,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(p,p,-1e-30f,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight<uint32_t,false>(negative.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        for(uint32_t i=1;i<Order();++i){
            Div<float,false>(scratch,wideBuf[(i-1)*8],p,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            Sub<float,false>(p,mat[i*8],mid,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            Sub<float,false>(p,p,scratch,lanes,uint8_t(repeats),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(p,p,-1e-30f,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            ShiftRight<uint32_t,false>(flags.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<int32_t,false>(negative.template ReinterpretCast<int32_t>(),negative.template ReinterpretCast<int32_t>(),flags.template ReinterpretCast<int32_t>(),lanes,uint8_t(repeats),BinaryRepeatParams{1,1,1,8,8,8});
        }
        Barrier();Brcb(wideBuf,negative,uint8_t((queries+7)/8),{1,8});Barrier();
        DataCopyPad(gw[lane],wideBuf,DataCopyExtParams{uint16_t(queries),4,0,(team-1)*4,0});Barrier();
        SyncAll();DataCopy(prod,gw,branches);Barrier();
        for(uint32_t r=0;r<count;++r){
            uint32_t left=0,right=branches;float l=lower,h=upperBound;
            while(right-left>1 && h-l>tolerance){
                uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(value==l || value==h)break;
                if(prod.template ReinterpretCast<uint32_t>().GetValue(j-1)<=EigenIndex(r)){left=j;l=value;}else{right=j;h=value;}
            }
            piv.SetValue(r,l);fac.SetValue(r,h);
        }
        return 1.0f/float(int32_t(branches));
    }
    __aicore__ inline void BisectVector(float lower,float upperBound,float tolerance,bool shared=false,float widthScale=1.0f){
        const uint32_t len=64,lanes=laneStride;
        // Coefficients and shifts come from this invocation's tridiagonal.
        // Materialize all diagonal-minus-shift values with vector repeats.
        Mul(tmp,origE,origE,Stride());PipeBarrier<PIPE_V>();
        Brcb(wideBuf,tmp,uint8_t(Stride()/8),{1,8});
        for(uint32_t col=0;col<lanes;col+=8)
            Brcb(mat[col],origD,uint8_t(Stride()/8),{uint16_t(lanes/8),uint16_t(lanes)});
        PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        auto lo=piv,hi=fac,mid=pack,p=pack[Stride()],negative=pack[2*Stride()];
        auto rank=pack[3*Stride()],one=pack[4*Stride()],scratch=pack[5*Stride()];
        auto mask=pack[6*Stride()].template ReinterpretCast<uint8_t>();
        auto activeMask=pack[6*Stride()+8].template ReinterpretCast<uint8_t>();
        auto oldLo=pack[7*Stride()],oldHi=pack[8*Stride()];
        if(!shared){Duplicate<float,false>(lo,lower,len,1,1,8);Duplicate<float,false>(hi,upperBound,len,1,1,8);}Duplicate<float,false>(one,0.0f,len,1,1,8);Barrier();
        for(uint32_t r=0;r<len;++r)rank.SetValue(r,float(int32_t(EigenIndex(r))));Barrier();
        float width=(upperBound-lower)*widthScale;
        for(uint32_t it=0;it<32 && width>0.5f*tolerance;++it,width*=0.5f){
            Adds<float,false>(oldLo,lo,0.0f,len,1,UnaryRepeatParams{1,1,8,8});Adds<float,false>(oldHi,hi,0.0f,len,1,UnaryRepeatParams{1,1,8,8});
            Sub<float,false>(scratch,hi,lo,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Compares<float,uint8_t,false>(activeMask,scratch,tolerance,CMPMODE::GT,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SetVectorMask<float>(lanes);
            for(uint32_t row=0;row<Order();row+=255){
                uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
                Sub<float,false>(prod[row*lanes],mat[row*lanes],mid,lanes,repeats,BinaryRepeatParams{1,1,1,uint8_t(lanes/8),uint8_t(lanes/8),0});
            }
            PipeBarrier<PIPE_V>();Adds<float,false>(p,prod,0.0f,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            // The perturbation makes zero pivots negative before counting and division.
            Adds<float,false>(p,p,-1e-30f,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            ShiftRight<uint32_t,false>(negative.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            for(uint32_t i=1;i<Order();++i){
                Div<float,false>(scratch,wideBuf[(i-1)*8],p,lanes,1,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
                Sub<float,false>(p,prod[i*lanes],scratch,lanes,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Adds<float,false>(p,p,-1e-30f,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                ShiftRight<uint32_t,false>(one.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                // Updating the count is independent of the next recurrence division.
                Add<int32_t,false>(negative.template ReinterpretCast<int32_t>(),negative.template ReinterpretCast<int32_t>(),one.template ReinterpretCast<int32_t>(),lanes,1,BinaryRepeatParams{1,1,1,8,8,8});
            }
            PipeBarrier<PIPE_V>();Cast<float,int32_t,false>(one,negative.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            Compare<float,uint8_t,false>(mask,one,rank,CMPMODE::LE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,mask,mid,lo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,mask,hi,mid,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,activeMask,lo,oldLo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,activeMask,hi,oldHi,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
        Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});Barrier();
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),mid.GetValue(r));
    }
    // Congruence elimination preserves the inertia of a symmetric
    // tridiagonal. Odd pivots are independent, so each level runs in SIMD.
    // Retain every eliminated pivot in mat and count their signs at the end.
    __aicore__ inline void SchurInertia(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){uint8_t rep=uint8_t(Order()-row<255?Order()-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,rep,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});}
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){uint8_t rep=uint8_t(Order()-row<255?Order()-row:255);
            Adds<float,false>(pcrBand[row*len],wideBuf[row*8],0.0f,len,rep,UnaryRepeatParams{1,0,uint8_t(len/8),1});}
        PipeBarrier<PIPE_V>();
        for(uint32_t step=1;step<Order();step*=2){
            uint32_t odd=(Order()-1-step)/(2*step)+1,right=(Order()-1)/(2*step),dist=2*step*len;
            auto od=mat[step*len],oe=pcrBand[step*len];
            if(dist/8<=255){for(uint32_t first=0;first<odd;first+=255){uint8_t rep=uint8_t(odd-first<255?odd-first:255);
                Adds<float,false>(od[first*dist],od[first*dist],-1e-30f,len,rep,UnaryRepeatParams{1,1,uint8_t(dist/8),uint8_t(dist/8)});}}
            else for(uint32_t i=0;i<odd;++i)Adds<float,false>(od[i*dist],od[i*dist],-1e-30f,len,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            PcrOp<1>(pcrLeft,len,one,0,od,dist,len,odd);
            PcrOp<0>(wideBuf,len,pcrBand,dist,pcrLeft,len,len,odd);
            PcrOp<0>(wideBuf,len,wideBuf,len,pcrBand,dist,len,odd);
            PcrOp<2>(mat,dist,mat,dist,wideBuf,len,len,odd);
            if(right){
                PcrOp<0>(wideBuf,len,oe,dist,pcrLeft,len,len,right);
                PcrOp<0>(wideBuf,len,wideBuf,len,oe,dist,len,right);
                PcrOp<2>(mat[dist],dist,mat[dist],dist,wideBuf,len,len,right);
            }
            PcrOp<0>(wideBuf,len,pcrBand,dist,pcrLeft,len,len,odd);
            PcrOp<0>(wideBuf,len,wideBuf,len,oe,dist,len,odd);
            Muls(wideBuf,wideBuf,-1.0f,odd*len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            PcrOp<0>(pcrBand,dist,wideBuf,len,one,0,len,odd);
        }
        Adds<float,false>(mat,mat,-1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight(mat.template ReinterpretCast<uint32_t>(),mat.template ReinterpretCast<uint32_t>(),uint32_t(31),Order()*len);PipeBarrier<PIPE_V>();
        for(uint32_t activeRows=Order();activeRows>1;activeRows=(activeRows+1)/2){
            uint32_t half=activeRows/2,next=(activeRows+1)/2;
            Add(mat.template ReinterpretCast<int32_t>(),mat.template ReinterpretCast<int32_t>(),mat[next*len].template ReinterpretCast<int32_t>(),half*len);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
    }
    __aicore__ inline void BisectInertia(float lower,float upperBound,float tolerance,bool shared){
        uint32_t len=(count+7)/8*8;
        auto lo=piv,hi=fac,mid=pack,one=pack[Stride()],rank=pack[2*Stride()],counts=pack[3*Stride()],width=pack[4*Stride()];
        auto oldLo=pack[7*Stride()],oldHi=pack[8*Stride()];
        auto mask=pack[5*Stride()].template ReinterpretCast<uint8_t>(),activeMask=pack[6*Stride()].template ReinterpretCast<uint8_t>();
        if(!shared){Duplicate(lo,lower,len);Duplicate(hi,upperBound,len);}
        Duplicate(one,1.0f,len);Barrier();
        for(uint32_t i=0;i<len;++i)rank.SetValue(i,float(int32_t(EigenIndex(i))));Barrier();
        for(uint32_t it=0;it<32;++it){
            float widest=0;for(uint32_t i=0;i<count;++i){float span=hi.GetValue(i)-lo.GetValue(i);widest=widest>span?widest:span;}
            if(widest<=tolerance)break;
            SetVectorMask<float>(len);
            Adds<float,false>(oldLo,lo,0.0f,len,1,UnaryRepeatParams{1,1,8,8});Adds<float,false>(oldHi,hi,0.0f,len,1,UnaryRepeatParams{1,1,8,8});
            Sub<float,false>(width,hi,lo,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Compares<float,uint8_t,false>(activeMask,width,tolerance,CMPMODE::GT,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SchurInertia(mid,one,len);
            Cast<float,int32_t,false>(counts,mat.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Compare<float,uint8_t,false>(mask,counts,rank,CMPMODE::LE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,mask,mid,lo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,mask,hi,mid,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,activeMask,lo,oldLo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,activeMask,hi,oldHi,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});Barrier();
        }
        SetVectorMask<float>(len);Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});Barrier();
        for(uint32_t i=0;i<count;++i)d.SetValue(EigenIndex(i),mid.GetValue(i));
    }
    __aicore__ inline void ScaleOriginalEdges(LocalTensor<float> dst,LocalTensor<float> src,uint32_t sourceStride,uint32_t firstEdge,uint32_t len,uint32_t number){
        for(uint32_t first=0;first<number;first+=255){
            uint8_t repeats=uint8_t(number-first<255?number-first:255);
            Mul<float,false>(dst[first*len],wideBuf[(firstEdge+2*first)*8],src[first*sourceStride],len,repeats,
                BinaryRepeatParams{1,0,1,uint8_t(len/8),2,uint8_t(sourceStride/8)});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void SchurInertiaInplace(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
        // Match the original initialization, including the sign of zero.
        Adds(wideBuf,wideBuf,0.0f,8*Stride());
        Duplicate(tmp2.template ReinterpretCast<int32_t>(),int32_t(0),len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t step=1;step<Order();step*=2){
            uint32_t odd=(Order()-1-step)/(2*step)+1,right=(Order()-1)/(2*step),dist=2*step*len;
            auto od=mat[step*len];bool batchable=dist/8<=255;
            for(uint32_t first=0;first<odd;){
                uint8_t repeats=batchable?uint8_t(odd-first<255?odd-first:255):1;
                Adds<float,false>(od[first*dist],od[first*dist],-1e-30f,len,repeats,
                    UnaryRepeatParams{1,1,uint8_t(batchable?dist/8:8),uint8_t(batchable?dist/8:8)});
                first+=repeats;
            }
            PipeBarrier<PIPE_V>();
            for(uint32_t first=0;first<odd;){
                uint8_t repeats=batchable?uint8_t(odd-first<255?odd-first:255):1;
                ShiftRight<uint32_t,false>(prod[first*len].template ReinterpretCast<uint32_t>(),od[first*dist].template ReinterpretCast<uint32_t>(),31u,len,repeats,
                    UnaryRepeatParams{1,1,uint8_t(len/8),uint8_t(batchable?dist/8:8)});
                first+=repeats;
            }
            PipeBarrier<PIPE_V>();
            for(uint32_t number=odd;number>1;number=(number+1)/2){
                uint32_t half=number/2,next=(number+1)/2;
                Add(prod.template ReinterpretCast<int32_t>(),prod.template ReinterpretCast<int32_t>(),prod[next*len].template ReinterpretCast<int32_t>(),half*len);PipeBarrier<PIPE_V>();
            }
            Add(tmp2.template ReinterpretCast<int32_t>(),tmp2.template ReinterpretCast<int32_t>(),prod.template ReinterpretCast<int32_t>(),len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            // Once its sign is counted, an odd diagonal stores its reciprocal.
            // Once both neighbor updates finish, the same slot stores the new
            // off-diagonal. Old edges occupy disjoint midpoint slots.
            PcrOp<1>(od,dist,one,0,od,dist,len,odd);
            if(step==1){
                ScaleOriginalEdges(prod,od,dist,0,len,odd);
                ScaleOriginalEdges(prod,prod,len,0,len,odd);
            }else{
                auto left=mat[(step/2)*len];
                PcrOp<0>(prod,len,left,dist,od,dist,len,odd);
                PcrOp<0>(prod,len,prod,len,left,dist,len,odd);
            }
            PcrOp<2>(mat,dist,mat,dist,prod,len,len,odd);
            if(right){
                if(step==1){
                    ScaleOriginalEdges(prod,od,dist,1,len,right);
                    ScaleOriginalEdges(prod,prod,len,1,len,right);
                }else{
                    auto edge=mat[(step+step/2)*len];
                    PcrOp<0>(prod,len,edge,dist,od,dist,len,right);
                    PcrOp<0>(prod,len,prod,len,edge,dist,len,right);
                }
                PcrOp<2>(mat[dist],dist,mat[dist],dist,prod,len,len,right);
                if(step==1){
                    ScaleOriginalEdges(prod,od,dist,0,len,right);
                    ScaleOriginalEdges(prod,prod,len,1,len,right);
                }else{
                    PcrOp<0>(prod,len,mat[(step/2)*len],dist,od,dist,len,right);
                    PcrOp<0>(prod,len,prod,len,mat[(step+step/2)*len],dist,len,right);
                }
                Muls(prod,prod,-1.0f,right*len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
                PcrOp<0>(od,dist,prod,len,one,0,len,right);
            }
        }
        Adds<float,false>(mat,mat,-1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight<uint32_t,false>(prod.template ReinterpretCast<uint32_t>(),mat.template ReinterpretCast<uint32_t>(),31u,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Add(mat.template ReinterpretCast<int32_t>(),tmp2.template ReinterpretCast<int32_t>(),prod.template ReinterpretCast<int32_t>(),len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
    }

    __aicore__ __attribute__((noinline)) float CommonGrid(float lower,float upperBound,float tolerance){
        if(ResidentFamily() || laneStride==64)return OriginalCommonGrid(lower,upperBound,tolerance);
        uint32_t limit=4*Stride()<64*team?4*Stride():64*team;
        uint32_t branches=1;while(2*branches<=limit)branches*=2;
        uint32_t len=0;
        while(branches>=2){
            uint32_t maxQueries=(branches-2)/team+1;
            len=(maxQueries+7)/8*8;
            uint32_t needed=(Order()+(Order()+1)/2)*len+4*Stride()+3*len+64;
            if(len<=64 && needed<=storageCapacity)break;
            branches/=2;
        }
        if(branches<2)return OriginalCommonGrid(lower,upperBound,tolerance);
        uint32_t queries=lane<branches-1?(branches-2-lane)/team+1:0;
        auto oldMat=mat,oldProd=prod,oldTmp2=tmp2;
        auto oldD=d,oldOrigD=origD,oldOrigE=origE,oldBeta=beta;
        DataCopy(wideBuf,origD,Stride());DataCopy(wideBuf[Stride()],origE,Stride());
        DataCopy(wideBuf[2*Stride()],beta,Stride());DataCopy(wideBuf[3*Stride()],d,Stride());Barrier();
        auto memory=storage.template Get<float>();uint32_t prefix=(Order()+(Order()+1)/2)*len;
        mat=memory;prod=memory[Order()*len];origD=memory[prefix];origE=memory[prefix+Stride()];
        beta=memory[prefix+2*Stride()];d=memory[prefix+3*Stride()];
        auto mid=memory[prefix+4*Stride()],one=mid[len];tmp2=one[len];
        DataCopy(origD,wideBuf,Stride());DataCopy(origE,wideBuf[Stride()],Stride());
        DataCopy(beta,wideBuf[2*Stride()],Stride());DataCopy(d,wideBuf[3*Stride()],Stride());
        Duplicate(mid,0.0f,len);Duplicate(one,1.0f,len);Barrier();
        for(uint32_t r=0;r<queries;++r){
            uint32_t target=lane+r*team+1,left=0,right=branches;float l=lower,h=upperBound;
            while(true){uint32_t j=(left+right)/2;float value=0.5f*(l+h);
                if(j==target){mid.SetValue(r,value);break;}
                if(target<j){right=j;h=value;}else{left=j;l=value;}
            }
        }
        Barrier();SchurInertiaInplace(mid,one,len);Barrier();
        if(queries){
            Brcb(wideBuf,mat,uint8_t((queries+7)/8),{1,8});Barrier();
            DataCopyPad(gw[lane],wideBuf,DataCopyExtParams{uint16_t(queries),4,0,(team-1)*4,0});Barrier();
        }
        // Save all metadata before rebinding views that may overlap the
        // temporary query matrix. Counts have already been published to GM.
        DataCopy(wideBuf,origD,Stride());DataCopy(wideBuf[Stride()],origE,Stride());
        DataCopy(wideBuf[2*Stride()],beta,Stride());DataCopy(wideBuf[3*Stride()],d,Stride());Barrier();
        mat=oldMat;prod=oldProd;tmp2=oldTmp2;d=oldD;origD=oldOrigD;origE=oldOrigE;beta=oldBeta;
        DataCopy(origD,wideBuf,Stride());DataCopy(origE,wideBuf[Stride()],Stride());
        DataCopy(beta,wideBuf[2*Stride()],Stride());DataCopy(d,wideBuf[3*Stride()],Stride());Barrier();
        SyncAll();DataCopy(prod,gw,branches);Barrier();
        for(uint32_t r=0;r<count;++r){
            uint32_t left=0,right=branches;float l=lower,h=upperBound;
            while(right-left>1 && h-l>tolerance){
                uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(value==l || value==h)break;
                if(prod.template ReinterpretCast<uint32_t>().GetValue(j-1)<=EigenIndex(r)){left=j;l=value;}
                else{right=j;h=value;}
            }
            piv.SetValue(r,l);fac.SetValue(r,h);
        }
        return 1.0f/float(int32_t(branches));
    }
    // Solve independent shift queries in one SIMD repeat. Reduction has
    // finished, so the small-matrix arena is now transient query storage.
    // Every interval is selected from current-input Sturm counts; the
    // original scalar convergence tolerance and complete spectrum remain.
    __aicore__ inline void BisectResidentGrid(float lower,float upperBound,float tolerance){
        constexpr uint32_t queryLanes=32;
        auto memory=smallStorage.template Get<float>();
        auto shifts=memory,pivot=memory[queryLanes],negative=memory[2*queryLanes];
        auto quotient=memory[3*queryLanes],flags=memory[4*queryLanes];
        auto diagonal=memory[5*queryLanes],squaredEdge=diagonal[8*Stride()];
        Mul(tmp,origE,origE,Stride());PipeBarrier<PIPE_V>();
        Brcb(diagonal,origD,uint8_t(Stride()/8),{1,8});
        Brcb(squaredEdge,tmp,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
        for(uint32_t r=0;r<count;++r){
            float lo=lower,hi=upperBound;
            for(uint32_t iteration=0;iteration<16 && hi-lo>tolerance;++iteration){
                CreateIndicesWithArena(flags.template ReinterpretCast<int32_t>(),1,queryLanes);PipeBarrier<PIPE_V>();
                Cast(shifts,flags.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,queryLanes);PipeBarrier<PIPE_V>();
                Muls(shifts,shifts,(hi-lo)*0.03125f,queryLanes);PipeBarrier<PIPE_V>();
                Adds(shifts,shifts,lo,queryLanes);VectorToScalar();
                shifts.SetValue(queryLanes-1,hi);
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
                SetVectorMask<float>(queryLanes);
                Sub<float,false>(pivot,diagonal,shifts,queryLanes,1,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
                Adds<float,false>(pivot,pivot,-1e-30f,queryLanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                ShiftRight<uint32_t,false>(negative.template ReinterpretCast<uint32_t>(),pivot.template ReinterpretCast<uint32_t>(),31u,queryLanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                for(uint32_t i=1;i<Order();++i){
                    Div<float,false>(quotient,squaredEdge[(i-1)*8],pivot,queryLanes,1,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
                    Sub<float,false>(pivot,diagonal[i*8],shifts,queryLanes,1,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
                    Sub<float,false>(pivot,pivot,quotient,queryLanes,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                    Adds<float,false>(pivot,pivot,-1e-30f,queryLanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    ShiftRight<uint32_t,false>(flags.template ReinterpretCast<uint32_t>(),pivot.template ReinterpretCast<uint32_t>(),31u,queryLanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    Add<int32_t,false>(negative.template ReinterpretCast<int32_t>(),negative.template ReinterpretCast<int32_t>(),flags.template ReinterpretCast<int32_t>(),queryLanes,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                }
                VectorToScalar();
                uint32_t first=0,last=queryLanes,index=EigenIndex(r);
                while(first<last){
                    const uint32_t middle=(first+last)/2;
                    if(negative.template ReinterpretCast<uint32_t>().GetValue(middle)<=index)first=middle+1;
                    else last=middle;
                }
                const float nextLo=first?shifts.GetValue(first-1):lo;
                const float nextHi=first<queryLanes?shifts.GetValue(first):hi;
                if(nextLo==lo && nextHi==hi)break;
                lo=nextLo;hi=nextHi;
            }
            d.SetValue(EigenIndex(r),0.5f*(lo+hi));
        }
        ResetMask();Barrier();
    }
    __aicore__ inline void Bisect(){
        float lower=origD.GetValue(0),upperBound=lower;
        for(uint32_t i=0;i<Order();++i){
            float radius=(i>0?Abs(origE.GetValue(i-1)):0)+(i+1<Order()?Abs(origE.GetValue(i)):0);
            float a=origD.GetValue(i)-radius,b=origD.GetValue(i)+radius;
            lower=lower<a?lower:a;upperBound=upperBound>b?upperBound:b;
        }
        float bound=Abs(lower)>Abs(upperBound)?Abs(lower):Abs(upperBound);
        spectralBound=bound;
        spectralCenter=0.5f*lower+0.5f*upperBound;
        float tolerance=1e-6f*bound+1e-30f;
        lower-=tolerance;upperBound+=tolerance;
        if constexpr(ResidentEntry){if(Order()>=3 && ResidentFamily()){BisectResidentGrid(lower,upperBound,tolerance);return;}}
        bool shared=Stride()>=64 && batch%groups==0;
        float widthScale=shared?CommonGrid(lower,upperBound,tolerance):1.0f;
        if(count>=2 && !ResidentFamily()){if(laneStride==64 && Stride()>=64)BisectVector(lower,upperBound,tolerance,shared,widthScale);else BisectInertia(lower,upperBound,tolerance,shared);return;}
        for(uint32_t r=0;r<count;++r){
            uint32_t index=EigenIndex(r);float lo=shared?piv.GetValue(r):lower,hi=shared?fac.GetValue(r):upperBound;
            for(uint32_t it=0;it<64 && hi-lo>tolerance;++it){
                float mid=0.5f*(lo+hi);if(mid==lo || mid==hi)break;
                float p=origD.GetValue(0)-mid;uint32_t negative=p<=0;
                for(uint32_t i=1;i<Order();++i){
                    if(Abs(p)<1e-30f)p=-1e-30f;
                    float off=origE.GetValue(i-1);p=origD.GetValue(i)-mid-off*off/p;
                    negative+=p<=0;
                }
                if(negative<=index)lo=mid;else hi=mid;
            }
            d.SetValue(index,0.5f*(lo+hi));
        }
    }
    __aicore__ inline float SafePivot(float x){return Abs(x)<1e-20f?(x<0?-1e-20f:1e-20f):x;}
    __aicore__ inline uint32_t BackPanelWidth(){
        return !ResidentFamily() && Stride()>=64 && Stride()%64==0 && laneStride>=16 && CubeMode!=0?(CubeMode==1?16:64):0;
    }
    __aicore__ inline uint64_t WorkspacePerMatrix(){
        if(outputBackedHistory)return 5*uint64_t(Stride());
        uint64_t size=uint64_t(ResidentFamily()?2*Order()+3:(CubeMode?2*Order()+5:Order()+5))*Stride();
        if(BackPanelWidth()){
            size+=uint64_t(Order())*Stride()+uint64_t((team+1)/2*2)*laneStride*(Stride()+(CubeMode==1?16:64));
            if constexpr(CubeMode==2)size+=uint64_t((Order()-2+63)/64)*6*64*64;
        }
        return size;
    }
    __aicore__ inline void CubeBackHandshake(bool hasRows){
        auto q=hist[uint64_t(3*Order()+5)*Stride()+uint64_t(lane)*laneStride*Stride()];
        auto y=hist[uint64_t(3*Order()+5)*Stride()+uint64_t((team+1)/2*2)*laneStride*Stride()+uint64_t(lane)*laneStride*(CubeMode==1?16:64)];
        Barrier();
        Duplicate(prod,0.0f,laneStride*(CubeMode==1?16:64));Barrier();DataCopy(y,prod,laneStride*(CubeMode==1?16:64));
        if(hasRows){
            if(count<laneStride)Duplicate(mat[count*Stride()],0.0f,(laneStride-count)*Stride());
            Barrier();DataCopy(q,mat,laneStride*Stride());
        }else{
            Barrier();Duplicate(prod,0.0f,laneStride*Stride());Barrier();DataCopy(q,prod,laneStride*Stride());
        }
        // Complete Q/Y payload transfers before the Cube starts the
        // complete backtransform from its invocation-local GM copy.
        Barrier();CrossCoreSetFlag<2,PIPE_MTE3>(10);
        CrossCoreWaitFlag(9);
        if(hasRows){
            DataCopy(mat,q,laneStride*Stride());Barrier();
        }
    }
    __aicore__ __attribute__((noinline)) void PrepareBackPanels(){
        if constexpr(CubeMode==1){
        uint32_t tile=BackPanelWidth();if(!tile)return;
        auto transformed=hist[(2*Order()+5)*Stride()];
        for(uint32_t low=lane*tile;low<Order()-2;low+=team*tile){
            uint32_t number=Order()-2-low;number=number<tile?number:tile;
            uint32_t height=(number+7)/8*8,dotTile=Stride()/64;
            auto gramMatrix=pack,partial=pack[Stride()>256?Stride():256];
            DataCopy(mat,hist[low*Stride()],number*Stride());
            Duplicate(gramMatrix,0.0f,number*height);Barrier();SetVectorMask<float>(64);
            for(uint32_t block=0;block<number;block+=dotTile){
                uint32_t cols=number-block<dotTile?number-block:dotTile,total=cols*height;
                Duplicate(prod,0.0f,total*64);PipeBarrier<PIPE_V>();
                for(uint32_t col=(low+1)/64*64;col<Stride();col+=64){
                    for(uint32_t j=0;j<cols;++j)
                        Mul<float,false>(prod[j*height*64],mat[col],mat[(block+j)*Stride()+col],64,uint8_t(number),BinaryRepeatParams{1,1,1,8,uint8_t(Stride()/8),0});
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum<float,false>(partial,prod,64,uint8_t(total),1,1,8);PipeBarrier<PIPE_V>();
                    Add(gramMatrix[block*height],gramMatrix[block*height],partial,total);PipeBarrier<PIPE_V>();
                }
            }
            VectorToScalar();
            for(int j=int(number)-1;j>=0;--j){
                auto z=prod[uint32_t(j)*Stride()];
                DataCopy(z,mat[uint32_t(j)*Stride()],Stride());Barrier();
                for(int i=int(number)-1;i>j;--i){
                    float gram=gramMatrix.GetValue(uint32_t(j)*height+uint32_t(i));
                    Axpy(z,prod[uint32_t(i)*Stride()],-gram,Stride());PipeBarrier<PIPE_V>();
                }
                Muls(z,z,beta.GetValue(low+uint32_t(j)),Stride());Barrier();
            }
            Muls(prod,prod,-1.0f,number*Stride());PipeBarrier<PIPE_V>();
            uint32_t stored=number;
            if(low+number==Order()-2){Duplicate(prod[number*Stride()],0.0f,2*Stride());stored+=2;}
            Barrier();DataCopy(transformed[low*Stride()],prod,stored*Stride());Barrier();
        }

        }else{
        uint32_t tile=64,panels=(Order()-2+63)/64,parts=team/panels;
        uint32_t panel=lane/parts,part=lane%parts;if(panel>=panels)return;
        uint32_t low=panel*64,number=Order()-2-low;number=number<64?number:64;
        auto gamma=hist[uint64_t(3*Order()+5)*Stride()+uint64_t((team+1)/2*2)*laneStride*(Stride()+64)+uint64_t(panel)*6*64*64];
        // The Gram MMA writes its actual number-by-number rectangle only.
        // Copy that rectangle with explicit zero padding into a zeroed tile.
        uint32_t gramPitch=(number+7)/8*8;
        Duplicate(mat,0.0f,64*64);Barrier();
        DataCopyPad(mat,gamma,DataCopyExtParams{uint16_t(number),number*4,(64-number)*4,(64-gramPitch)/8,0},DataCopyPadExtParams<float>{true,0,uint8_t(gramPitch-number),0});Barrier();
        for(uint32_t row=part;row<64;row+=parts){
            // Only the selected strict upper triangle contributes to R.
            // Padding rows are explicitly zero and the identity is complete.
            if(row<number){Mul(prod,mat[row*64],beta[low],64);PipeBarrier<PIPE_V>();Duplicate(prod,0.0f,row+1);}
            else Duplicate(prod,0.0f,64);
            PipeBarrier<PIPE_V>();
            float scale=row<number?beta.GetValue(low+row):0.0f;
            Muls(w,prod,scale,64);Duplicate(tmp,0.0f,64);VectorToScalar();
            w.SetValue(row,-scale);tmp.SetValue(row,1.0f);Barrier();
            DataCopy(gamma[3*64*64+row*64],prod,64);
            DataCopy(gamma[64*64+row*64],w,64);
            DataCopy(gamma[5*64*64+row*64],tmp,64);Barrier();
        }
        if(lane==0){Duplicate(prod,0.0f,2*Stride());Barrier();DataCopy(hist[(2*Order()+5)*Stride()+(Order()-2)*Stride()],prod,2*Stride());Barrier();}
        }

    }
    __aicore__ inline void ReadReflectors(LocalTensor<float> destination,uint32_t offset,uint32_t length){
        if constexpr(PrivateSmall)DataCopy(destination,privateHistory[offset],length);
        else DataCopy(destination,hist[offset],length);
    }
    __aicore__ inline void Backtransform(){
        if(BackPanelWidth()){CubeBackHandshake(true);return;}
        if(Order()<3)return;PackMatrix();
        // Retired solve storage holds two reflector tiles. Each tile comes
        // from this invocation's reduction; every reflector is applied in
        // exactly the original descending order and precision.
        uint32_t tile=laneStride/2;tile=tile<16?tile:16;
        uint32_t first=(Order()-3)/tile*tile,slot=0;
        ReadReflectors(pcrBand,first*Stride(),(Order()-2-first)*Stride());Barrier();
        for(int block=int(first);block>=0;block-=int(tile)){
            if(block>=int(tile))ReadReflectors(pcrBand[(1-slot)*tile*Stride()],(block-tile)*Stride(),tile*Stride());
            uint32_t last=uint32_t(block)+tile-1;last=last<Order()-2?last:Order()-3;
            for(int k=int(last);k>=block;--k){
                auto current=pcrBand[(slot*tile+uint32_t(k-block))*Stride()];
            float b=beta.GetValue(k);
            if(b!=0){
                DotRows(current,k+1);
                if(packedMatrix){
                    uint32_t padded=(count+7)/8*8,first=(k+1)/8*8;
                    Brcb(pcrLeft,w,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
                    Muls(pcrLeft,pcrLeft,-b,padded*8);PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
                    for(uint32_t rb=0;rb<padded/8;++rb)
                        MulAddDst<float,float,false>(mat[rb*8*Stride()+first*8],current[first],pcrLeft[rb*64],64,uint8_t((Stride()-first)/8),BinaryRepeatParams{1,0,1,8,1,0});
                }else if(Stride()%64==0){
                    Brcb(pack,w,uint8_t((count+7)/8),{1,8});PipeBarrier<PIPE_V>();Muls(pack,pack,-b,8*((count+7)/8*8));PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
                    for(uint32_t col=(k+1)/64*64;col<Stride();col+=64)
                        MulAddDst<float,float,false>(mat[col],current[col],pack,64,uint8_t(count),BinaryRepeatParams{1,1,0,uint8_t(Stride()/8),0,1});
                }else for(uint32_t r=0;r<count;++r){auto q=mat[r*Stride()];Axpy(q,current,-b*w.GetValue(r),Stride());}
            }
            Barrier();
            }
            slot=1-slot;
        }
        UnpackMatrix();
    }
    __aicore__ inline void FactorVector(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        auto a=pack[5*Stride()];
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();Mul(v,origE,origE,Stride());PipeBarrier<PIPE_V>();Muls(v,v,-1.0f,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        Adds<float,false>(a,mat,1e-20f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(mat,one,a,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        for(uint32_t i=1;i<Order();++i){
            MulAddDst<float,float,false>(mat[i*len],mat[(i-1)*len],wideBuf[(i-1)*8],len,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            Adds<float,false>(a,mat[i*len],1e-20f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(mat[i*len],one,a,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
        // Reuse factors within this solve. The next factorization overwrites
        // this table; every kernel invocation computes it from its own input.
        Muls(v,origE,-1.0f,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Mul<float,false>(pcrBand[row*len],mat[row*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();
    }
    // Pair disjoint contiguous halves: no strided gather and no overlapping
    // repeat addresses. An odd middle row remains in the next active prefix.
    __aicore__ inline void SumEigenRows(uint32_t len){
        for(uint32_t activeRows=Order();activeRows>1;activeRows=(activeRows+1)/2){
            uint32_t half=activeRows/2,next=(activeRows+1)/2;
            Add(mat,mat,mat[next*len],half*len);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
    }
    __aicore__ inline void RayleighVector(LocalTensor<float> shifts,uint32_t len){
        auto norm=pack[3*Stride()];
        Adds(v,origD,-spectralCenter,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Mul<float,false>(mat[row*len],prod[row*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=1;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            MulAddDst<float,float,false>(mat[row*len],prod[(row-1)*len],wideBuf[(row-1)*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t row=0;row+1<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-1-row<255?Order()-1-row:255);
            MulAddDst<float,float,false>(mat[row*len],prod[(row+1)*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();Mul(mat,mat,prod,Order()*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
        Adds<float,false>(shifts,mat,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Mul(mat,prod,prod,Order()*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
        Adds<float,false>(norm,mat,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(shifts,shifts,norm,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(shifts,shifts,spectralCenter,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    }
    template<uint32_t Op>
    __aicore__ inline void PcrOp(LocalTensor<float> dst,uint32_t ds,LocalTensor<float> a,uint32_t as,LocalTensor<float> b,uint32_t bs,uint32_t len,uint32_t number){
        bool batchable=ds/8<=255 && as/8<=255 && bs/8<=255;
        for(uint32_t first=0;first<number;){
            uint8_t repeats=batchable?uint8_t(number-first<255?number-first:255):1;
            BinaryRepeatParams br{1,1,1,uint8_t(batchable?ds/8:8),uint8_t(batchable?as/8:8),uint8_t(batchable?bs/8:8)};
            if constexpr(Op==0)Mul<float,false>(dst[first*ds],a[first*as],b[first*bs],len,repeats,br);
            if constexpr(Op==1)Div<float,false>(dst[first*ds],a[first*as],b[first*bs],len,repeats,br);
            if constexpr(Op==2)Sub<float,false>(dst[first*ds],a[first*as],b[first*bs],len,repeats,br);
            first+=repeats;
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PcrSolve(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t rep=uint8_t(Order()-row<255?Order()-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,rep,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t rep=uint8_t(Order()-row<255?Order()-row:255);
            Adds<float,false>(pcrBand[row*len],wideBuf[row*8],0.0f,len,rep,UnaryRepeatParams{1,0,uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();
        uint32_t step=1;
        for(;step<Order();step*=2){
            uint32_t odd=(Order()-1-step)/(2*step)+1,keep=(Order()-1)/(2*step)+1,dist=2*step*len;
            auto od=mat[step*len],oq=prod[step*len],oe=pcrBand[step*len];
            // Normalize eliminated rows and retain the left factors temporarily.
            if(dist/8<=255){for(uint32_t first=0;first<odd;first+=255){uint8_t rep=uint8_t(odd-first<255?odd-first:255);Adds<float,false>(od[first*dist],od[first*dist],1e-20f,len,rep,UnaryRepeatParams{1,1,uint8_t(dist/8),uint8_t(dist/8)});}}
            else for(uint32_t j=0;j<odd;++j)Adds<float,false>(od[j*dist],od[j*dist],1e-20f,len,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            PcrOp<1>(pcrLeft,len,one,0,od,dist,len,odd);
            PcrOp<0>(oe,dist,oe,dist,pcrLeft,len,len,odd);
            PcrOp<0>(oq,dist,oq,dist,pcrLeft,len,len,odd);
            PcrOp<0>(pcrLeft,len,pcrBand,dist,pcrLeft,len,len,odd);
            // Right-neighbor corrections use the original left off-diagonal.
            PcrOp<0>(wideBuf,len,pcrBand,dist,pcrLeft,len,len,odd);
            PcrOp<2>(mat,dist,mat,dist,wideBuf,len,len,odd);
            PcrOp<0>(wideBuf,len,pcrBand,dist,oq,dist,len,odd);
            PcrOp<2>(prod,dist,prod,dist,wideBuf,len,len,odd);
            // The original right off-diagonal is D_odd times its normalized value.
            PcrOp<0>(wideBuf,len,od,dist,oe,dist,len,keep-1);
            PcrOp<0>(wideBuf,len,wideBuf,len,oe,dist,len,keep-1);
            PcrOp<2>(mat[dist],dist,mat[dist],dist,wideBuf,len,len,keep-1);
            PcrOp<0>(wideBuf,len,od,dist,oe,dist,len,keep-1);
            PcrOp<0>(wideBuf,len,wideBuf,len,oq,dist,len,keep-1);
            PcrOp<2>(prod[dist],dist,prod[dist],dist,wideBuf,len,len,keep-1);
            // Connect the surviving rows and save both interpolation factors.
            PcrOp<0>(wideBuf,len,pcrBand,dist,oe,dist,len,odd);
            Muls(wideBuf,wideBuf,-1.0f,odd*len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            PcrOp<0>(pcrBand,dist,wideBuf,len,one,0,len,odd);
            PcrOp<0>(od,dist,pcrLeft,len,one,0,len,odd);
        }
        auto scratch=pack[5*Stride()];
        Adds<float,false>(scratch,mat,1e-20f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(prod,prod,scratch,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        for(step/=2;step;step/=2){
            uint32_t odd=(Order()-1-step)/(2*step)+1,right=(Order()-1)/(2*step),dist=2*step*len;
            auto od=mat[step*len],oq=prod[step*len],oe=pcrBand[step*len];
            PcrOp<0>(wideBuf,len,od,dist,prod,dist,len,odd);
            PcrOp<2>(oq,dist,oq,dist,wideBuf,len,len,odd);
            if(right){PcrOp<0>(wideBuf,len,oe,dist,prod[dist],dist,len,right);PcrOp<2>(oq,dist,oq,dist,wideBuf,len,len,right);}
        }
    }
    __aicore__ inline void MaximumEigenRows(LocalTensor<float> mx,uint32_t len){
        AscendC::Abs(mat,prod,Order()*len);PipeBarrier<PIPE_V>();
        for(uint32_t number=Order();number>1;number=(number+1)/2){uint32_t half=number/2,next=(number+1)/2;Max(mat,mat,mat[next*len],half*len);PipeBarrier<PIPE_V>();}
        SetVectorMask<float>(len);Adds<float,false>(mx,mat,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    }
    // Evaluate the actual tridiagonal Rayleigh quotient for scalar solves.
    __aicore__ inline float ReferenceRayleigh(LocalTensor<float> q){
        auto left=pcrLeft[Stride()].template ReinterpretCast<uint32_t>(),right=pcrLeft[2*Stride()].template ReinterpretCast<uint32_t>();
        Mul(w,q,origD,Stride());PipeBarrier<PIPE_V>();
        Duplicate(tmp2,0.0f,Stride());Duplicate(tmp,0.0f,Stride());PipeBarrier<PIPE_V>();
        Gather(tmp2,q,left,0,Order());Gather(tmp,origE,left,0,Order());PipeBarrier<PIPE_V>();
        Mul(tmp2,tmp2,tmp,Stride());PipeBarrier<PIPE_V>();Duplicate(tmp2,0.0f,1);PipeBarrier<PIPE_V>();
        Add(w,w,tmp2,Stride());PipeBarrier<PIPE_V>();
        Duplicate(tmp2,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(tmp2,q,right,0,Order());PipeBarrier<PIPE_V>();
        Mul(tmp2,tmp2,origE,Stride());PipeBarrier<PIPE_V>();Add(w,w,tmp2,Order()-1);PipeBarrier<PIPE_V>();
        return Dot(q,w);
    }
    __aicore__ inline void InverseVector(){
        uint32_t len=laneStride;SetVectorMask<float>(len);
        auto shifts=pack,one=pack[Stride()],mx=pack[2*Stride()],norm=pack[3*Stride()];
        auto inv=pack[4*Stride()],a=pack[5*Stride()],b=pack[6*Stride()];
        Duplicate<float,false>(shifts,0.0f,len,1,1,8);Duplicate<float,false>(one,1.0f,len,1,1,8);Barrier();
        for(uint32_t r=0;r<len;++r)shifts.SetValue(r,d.GetValue(EigenIndex(r)<Order()?EigenIndex(r):Order()-1));
        // Generate the same invocation-local hash start vectors with integer
        // SIMD operations. No input-independent tensor is supplied by the host.
        auto rowHash=v.template ReinterpretCast<int32_t>(),colHash=pack[7*Stride()].template ReinterpretCast<int32_t>();
        CreateIndicesWithArena(rowHash,int32_t(1),Stride());PipeBarrier<PIPE_V>();Muls(rowHash,rowHash,int32_t(747796405),Stride());Barrier();
        for(uint32_t r=0;r<len;++r)colHash.SetValue(r,int32_t((EigenIndex(r)+1)*2891336453u));Barrier();
        Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Add<int32_t,false>(prod[row*len].template ReinterpretCast<int32_t>(),wideBuf[row*8].template ReinterpretCast<int32_t>(),colHash,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<Order()*len;first+=4*Stride()){
            uint32_t k=Order()*len-first<4*Stride()?Order()*len-first:4*Stride();
            auto h=prod[first].template ReinterpretCast<uint32_t>(),shifted=mat[first].template ReinterpretCast<uint32_t>();
            auto h16=h.template ReinterpretCast<uint16_t>(),s16=shifted.template ReinterpretCast<uint16_t>(),t16=wideBuf.template ReinterpretCast<uint16_t>();
            ShiftRight(shifted,h,uint32_t(16),k);PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*k);PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*k);PipeBarrier<PIPE_V>();Not(t16,t16,2*k);PipeBarrier<PIPE_V>();And(h16,h16,t16,2*k);PipeBarrier<PIPE_V>();
            Muls(h.template ReinterpretCast<int32_t>(),h.template ReinterpretCast<int32_t>(),int32_t(2246822519u),k);PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(13),k);PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*k);PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*k);PipeBarrier<PIPE_V>();Not(t16,t16,2*k);PipeBarrier<PIPE_V>();And(h16,h16,t16,2*k);PipeBarrier<PIPE_V>();
            ShiftLeft(h,h,uint32_t(16),k);PipeBarrier<PIPE_V>();ShiftRight(h,h,uint32_t(16),k);PipeBarrier<PIPE_V>();
            Cast(prod[first],h.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,k);PipeBarrier<PIPE_V>();
            Muls(prod[first],prod[first],1.0f/65536.0f,k);PipeBarrier<PIPE_V>();Adds(prod[first],prod[first],-0.5f,k);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
        Barrier();
        for(uint32_t it=0;it<3;++it){
            if(it==1)FactorVector(shifts,one,len);
            if(it==2){RayleighVector(shifts,len);FactorVector(shifts,one,len);}
            if(it<1){PcrSolve(shifts,one,len);}else{
            for(uint32_t i=1;i<Order();++i){
                MulAddDst<float,float,false>(prod[i*laneStride],pcrBand[(i-1)*laneStride],prod[(i-1)*laneStride],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            Duplicate<float,false>(mx,0.0f,len,1,1,8);PipeBarrier<PIPE_V>();
            for(int i=int(Order())-1;i>=0;--i){
                auto q=prod[i*laneStride];
                if(i+1<int(Order())){MulAddDst<float,float,false>(q,prod[(i+1)*laneStride],wideBuf[i*8],len,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();}
                Mul<float,false>(q,q,mat[i*laneStride],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            }
            MaximumEigenRows(mx,len);
            Maxs<float,false>(mx,mx,1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(inv,one,mx,len,1,BinaryRepeatParams{1,1,1,8,8,8});Duplicate<float,false>(norm,0.0f,len,1,1,8);PipeBarrier<PIPE_V>();
            if(it<2){
                for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
                PipeBarrier<PIPE_V>();continue;
            }
            for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();Mul(mat,prod,prod,Order()*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
            Sqrt<float,false>(norm,mat,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(inv,one,norm,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();
        }
        RayleighVector(shifts,len);Barrier();
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),shifts.GetValue(r));
        auto offsets=piv.template ReinterpretCast<uint32_t>();
        CreateIndicesWithArena(offsets.template ReinterpretCast<int32_t>(),int32_t(0),Stride());PipeBarrier<PIPE_V>();
        Muls(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),int32_t(laneStride*4),Stride());Barrier();
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];
            Duplicate(q,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(q,prod,offsets,r*4,Order());
        }
        Barrier();

        Backtransform();
    }
    __aicore__ inline float ScalarEigenvector(LocalTensor<float> q,uint32_t index,float ev){
        Duplicate(q,0.0f,Stride());Barrier();
        if(!ResidentFamily()){
            auto left=pcrLeft[Stride()].template ReinterpretCast<int32_t>(),right=pcrLeft[2*Stride()].template ReinterpretCast<int32_t>();
            CreateIndicesWithArena(left,int32_t(-1),Stride());CreateIndicesWithArena(right,int32_t(1),Stride());PipeBarrier<PIPE_V>();
            Muls(left,left,int32_t(4),Stride());Muls(right,right,int32_t(4),Stride());PipeBarrier<PIPE_V>();
            Maxs(left,left,int32_t(0),Stride());Mins(right,right,int32_t((Order()-1)*4),Stride());Barrier();
        }
        float reciprocal=1.0f/SafePivot(origD.GetValue(0)-ev);piv.SetValue(0,reciprocal);
        for(uint32_t i=1;i<Order();++i){
            float f=origE.GetValue(i-1)*reciprocal;fac.SetValue(i-1,f);
            reciprocal=1.0f/SafePivot(origD.GetValue(i)-ev-origE.GetValue(i-1)*f);piv.SetValue(i,reciprocal);
        }
        if(!ResidentFamily()){
            // The same invocation-local start, generated with integer SIMD.
            auto h=q.template ReinterpretCast<uint32_t>(),shifted=w.template ReinterpretCast<uint32_t>();
            auto h16=h.template ReinterpretCast<uint16_t>(),s16=shifted.template ReinterpretCast<uint16_t>(),t16=red.template ReinterpretCast<uint16_t>();
            CreateIndicesWithArena(q.template ReinterpretCast<int32_t>(),int32_t(1),Stride());PipeBarrier<PIPE_V>();
            Muls(q.template ReinterpretCast<int32_t>(),q.template ReinterpretCast<int32_t>(),int32_t(747796405),Stride());PipeBarrier<PIPE_V>();
            Adds(q.template ReinterpretCast<int32_t>(),q.template ReinterpretCast<int32_t>(),int32_t((index+1)*2891336453u),Stride());PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(16),Stride());PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Not(t16,t16,2*Stride());PipeBarrier<PIPE_V>();And(h16,h16,t16,2*Stride());PipeBarrier<PIPE_V>();
            Muls(q.template ReinterpretCast<int32_t>(),q.template ReinterpretCast<int32_t>(),int32_t(2246822519u),Stride());PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(13),Stride());PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Not(t16,t16,2*Stride());PipeBarrier<PIPE_V>();And(h16,h16,t16,2*Stride());PipeBarrier<PIPE_V>();
            ShiftLeft(h,h,uint32_t(16),Stride());PipeBarrier<PIPE_V>();ShiftRight(h,h,uint32_t(16),Stride());PipeBarrier<PIPE_V>();
            Cast(q,q.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,Stride());PipeBarrier<PIPE_V>();
            Muls(q,q,1.0f/65536.0f,Stride());PipeBarrier<PIPE_V>();Adds(q,q,-0.5f,Stride());Barrier();
            for(uint32_t i=Order();i<Stride();++i)q.SetValue(i,0);Barrier();
        }else{
        for(uint32_t i=0;i<Order();++i){uint32_t h=(i+1)*747796405u+(index+1)*2891336453u;h^=h>>16;h*=2246822519u;h^=h>>13;q.SetValue(i,float(int32_t(h&65535u))/65536.0f-0.5f);}        }

        for(uint32_t it=0;it<3;++it){
            if(it==2){
                if(!ResidentFamily()){ev=ReferenceRayleigh(q);}else{
                for(uint32_t i=0;i<Order();++i){
                    float a=origD.GetValue(i)*q.GetValue(i);
                    if(i>0)a+=origE.GetValue(i-1)*q.GetValue(i-1);
                    if(i+1<Order())a+=origE.GetValue(i)*q.GetValue(i+1);
                    w.SetValue(i,a);
                }
                for(uint32_t i=Order();i<Stride();++i)w.SetValue(i,0);
                Barrier();ev=Dot(q,w);
                }
                reciprocal=1.0f/SafePivot(origD.GetValue(0)-ev);piv.SetValue(0,reciprocal);
                for(uint32_t i=1;i<Order();++i){
                    float f=origE.GetValue(i-1)*reciprocal;fac.SetValue(i-1,f);
                    reciprocal=1.0f/SafePivot(origD.GetValue(i)-ev-origE.GetValue(i-1)*f);piv.SetValue(i,reciprocal);
                }
            }
            float previous=q.GetValue(0);for(uint32_t i=1;i<Order();++i){previous=q.GetValue(i)-fac.GetValue(i-1)*previous;q.SetValue(i,previous);}
            float a=q.GetValue(Order()-1)*piv.GetValue(Order()-1);q.SetValue(Order()-1,a);float mx=Abs(a);
            for(int i=int(Order())-2;i>=0;--i){a=(q.GetValue(i)-origE.GetValue(i)*a)*piv.GetValue(i);q.SetValue(i,a);mx=mx>Abs(a)?mx:Abs(a);}
            Barrier();Muls(q,q,1.0f/mx,Stride());Barrier();float norm=Root(Dot(q,q));Muls(q,q,1.0f/norm,Stride());Barrier();
        }
        // Refine each bisection shift with the converged tridiagonal eigenvector.
        if(!ResidentFamily())return ReferenceRayleigh(q);
        for(uint32_t i=0;i<Order();++i){
            float a=origD.GetValue(i)*q.GetValue(i);
            if(i>0)a+=origE.GetValue(i-1)*q.GetValue(i-1);
            if(i+1<Order())a+=origE.GetValue(i)*q.GetValue(i+1);
            w.SetValue(i,a);
        }
        for(uint32_t i=Order();i<Stride();++i)w.SetValue(i,0);
        Barrier();return Dot(q,w);
    }
    __aicore__ inline ExtendedFloat QExtended(LocalTensor<float> q,uint32_t i){return ExtendedFloat(q.GetValue(i),prod.GetValue(i),0);}
    __aicore__ inline void SetQExtended(LocalTensor<float> q,uint32_t i,ExtendedFloat x){q.SetValue(i,x.h);prod.SetValue(i,x.m);}
    __aicore__ inline ExtendedFloat AbsExtended(ExtendedFloat x){return x.Negative()?-x:x;}
    __aicore__ inline ExtendedFloat SafeExtended(ExtendedFloat x){
        float magnitude=Abs(x.h);if(magnitude<1e-25f && Abs(x.m)<1e-25f)return ExtendedFloat(x.Negative()?-1e-25f:1e-25f);return x;
    }
    __aicore__ inline void PivotFactorExtended(ExtendedFloat shift){
        auto swaps=piv.template ReinterpretCast<uint32_t>();
        for(uint32_t i=0;i<Order();++i){
            WriteExtended(i,ExtendedFloat(origD.GetValue(i))-shift);
            WriteExtended(Order()+i,ExtendedFloat(i+1<Order()?origE.GetValue(i):0));
            WriteExtended(2*Order()+i,ExtendedFloat(i+1<Order()?origE.GetValue(i):0));
            WriteExtended(3*Order()+i,ExtendedFloat(0));swaps.SetValue(i,0);
        }
        for(uint32_t i=0;i+1<Order();++i){
            ExtendedFloat diagonal=ReadExtended(i),lower=ReadExtended(Order()+i);
            if(AbsExtended(diagonal).h>=AbsExtended(lower).h){
                ExtendedFloat multiplier=lower/SafeExtended(diagonal);WriteExtended(Order()+i,multiplier);
                WriteExtended(i+1,ReadExtended(i+1)-multiplier*ReadExtended(2*Order()+i));
            }else{
                ExtendedFloat multiplier=diagonal/SafeExtended(lower),next=ReadExtended(i+1);
                WriteExtended(i,lower);WriteExtended(i+1,ReadExtended(2*Order()+i)-multiplier*next);
                WriteExtended(Order()+i,multiplier);WriteExtended(2*Order()+i,next);swaps.SetValue(i,1);
                if(i+2<Order()){ExtendedFloat upper=ReadExtended(2*Order()+i+1);WriteExtended(3*Order()+i,upper);WriteExtended(2*Order()+i+1,-multiplier*upper);}
            }
        }
    }
    __aicore__ inline ExtendedFloat RayleighExtended(LocalTensor<float> q){
        ExtendedFloat result(0),norm(0);
        for(uint32_t i=0;i<Order();++i){
            ExtendedFloat x=QExtended(q,i);norm=norm+x*x;
            result=result+ExtendedFloat(origD.GetValue(i))*x*x;
            if(i+1<Order())result=result+(ExtendedFloat(origE.GetValue(i))*x*QExtended(q,i+1)).Scale(2.0f);
        }
        return result/norm;
    }
    __aicore__ __attribute__((noinline)) float PivotedExtendedEigenvector(LocalTensor<float> q,uint32_t index,float ev){
        for(uint32_t i=0;i<Order();++i){
            uint32_t hash=(i+1)*747796405u+(index+1)*2891336453u;hash^=hash>>16;hash*=2246822519u;hash^=hash>>13;
            SetQExtended(q,i,ExtendedFloat(float(int32_t(hash&65535u))/65536.0f-0.5f));
        }
        ExtendedFloat shift(ev);PivotFactorExtended(shift);
        auto swaps=piv.template ReinterpretCast<uint32_t>();
        for(uint32_t it=0;it<5;++it){
            if(it>=3){shift=RayleighExtended(q);PivotFactorExtended(shift);}
            for(uint32_t i=0;i+1<Order();++i){
                ExtendedFloat a=QExtended(q,i),b=QExtended(q,i+1),f=ReadExtended(Order()+i);
                if(swaps.GetValue(i)){SetQExtended(q,i,b);SetQExtended(q,i+1,a-f*b);}
                else SetQExtended(q,i+1,b-f*a);
            }
            for(int i=int(Order())-1;i>=0;--i){
                ExtendedFloat value=QExtended(q,uint32_t(i));
                if(i+1<int(Order()))value=value-ReadExtended(2*Order()+uint32_t(i))*QExtended(q,uint32_t(i)+1);
                if(i+2<int(Order()))value=value-ReadExtended(3*Order()+uint32_t(i))*QExtended(q,uint32_t(i)+2);
                SetQExtended(q,uint32_t(i),value/SafeExtended(ReadExtended(uint32_t(i))));
            }
            float maximum=0;for(uint32_t i=0;i<Order();++i){float x=Abs(QExtended(q,i).h);maximum=maximum>x?maximum:x;}
            ExtendedFloat scale(maximum>0?maximum:1),norm(0);
            for(uint32_t i=0;i<Order();++i){ExtendedFloat x=QExtended(q,i)/scale;SetQExtended(q,i,x);norm=norm+x*x;}
            ExtendedFloat length=RootExtended(norm);
            for(uint32_t i=0;i<Order();++i)SetQExtended(q,i,QExtended(q,i)/length);
        }
        ExtendedFloat refined=RayleighExtended(q);
        for(uint32_t i=Order();i<Stride();++i)q.SetValue(i,0);
        Barrier();return refined.Float();
    }


    __aicore__ inline void PublishSmallMode(bool selected){
        if constexpr(PrivateSmall)return;
        Duplicate(param,0.0f,8);Barrier();param.SetValue(0,selected?1.0f:0.0f);
        event_t ready=ImmediateEvent<HardEvent::S_MTE3>();
        SetFlag<HardEvent::S_MTE3>(ready);WaitFlag<HardEvent::S_MTE3>(ready);
        DataCopy(gparam,param,8);Barrier();
    }
    __aicore__ inline bool AnySmallMode(uint32_t step){
        if constexpr(PrivateSmall){
            // One matrix group obtains the same reduction and QL decision on
            // every launched lane, including padded lanes. Multi-group input
            // can select different numerical paths and keeps full barriers.
            if(groups==1 && rows<=2)return smallQLNeedsCheck && smallQLSolved;
            return true;
        }
        bool any=false;
        for(uint32_t g=0;g<groups;++g){
            uint32_t matrix=step*groups+g;
            if(matrix>=batch)continue;
            uint64_t offset=uint64_t(matrix)*WorkspacePerMatrix()+uint64_t(2*Order()+2)*Stride();
            DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE>(ws[offset]);Barrier();
            any=any || ws.GetValue(offset)!=0.0f;
        }
        return any;
    }

    __aicore__ inline void TransposeSmallProducts(LocalTensor<float> high,LocalTensor<float> low,
        LocalTensor<float> transposedHigh,LocalTensor<float> transposedLow,uint32_t pitch){
        uint64_t sources[16],destinations[16];
        TransDataTo5HDParams parameters{false,false,1,0,0};
        for(uint32_t which=0;which<2;++which){
            auto input=which?low:high,output=which?transposedLow:transposedHigh;
            for(uint32_t row=0;row<pitch;row+=16)for(uint32_t column=0;column<Stride();column+=8){
                for(uint32_t i=0;i<16;++i)
                    sources[i]=uint64_t((row+i<Order()?input[(row+i)*Stride()+column]:v[column]).GetPhyAddr());
                for(uint32_t j=0;j<8;++j){
                    destinations[2*j]=uint64_t(output[(column+j)*pitch+row].GetPhyAddr());
                    destinations[2*j+1]=uint64_t(output[(column+j)*pitch+row+8].GetPhyAddr());
                }
                TransDataTo5HD<float>(destinations,sources,parameters);PipeBarrier<PIPE_V>();
            }
        }
        Barrier();
    }
    // Check the complete current-input solution before accepting the wider
    // FP32 magnitude family. All references here are this invocation's A/Q.
    __aicore__ __attribute__((noinline)) void CheckSmallQL(){
        uint32_t pitch=Stride(),dense=Order()*pitch;
        auto arena=smallStorage.template Get<float>();
        auto original=arena,product=arena[dense],allQ=arena[2*dense];
        auto ph=arena[3*dense],pm=arena[4*dense],zero=arena[5*dense];
        Duplicate(original,0.0f,dense);Barrier();
        // The compact input copy is still live in allQ until reconstruction.
        for(uint32_t i=0;i<Order();++i)for(uint32_t j=0;j<Order();++j){
            uint32_t pi=upper?Order()-1-i:i,pj=upper?Order()-1-j:j;
            uint32_t row=pi>pj?pi:pj,col=pi>pj?pj:pi;
            if(upper){uint32_t t=row;row=col;col=t;}
            original.SetValue(i*pitch+j,allQ.GetValue(row*Order()+col));
        }
        Barrier();Mul(product,original,original,dense);Duplicate(w,0.0f,pitch);PipeBarrier<PIPE_V>();
        WholeReduceSum(w,product,pitch,uint8_t(Order()),1,1,uint8_t(pitch/8));PipeBarrier<PIPE_V>();
        ReduceSumWithArena(red,w,tmp,pitch);VectorToScalar();float frobenius2=red.GetValue(0);
        if constexpr(PrivateSmall){
            DataCopyPad(allQ,vectors[uint64_t(currentBatch)*Order()*Order()],
                DataCopyExtParams{uint16_t(Order()),Order()*4,0,0,0},
                DataCopyPadExtParams<float>{true,0,uint8_t(Stride()-Order()),0});
        }else DataCopy(allQ,hist,dense);
        Duplicate(zero,0.0f,dense);Duplicate(v,0.0f,pitch);Barrier();
        const float frobenius=Root(frobenius2);
        const float perColumn=frobenius/Root(float(int32_t(Order())));
        float safeGap=5e-5f*perColumn;
        float observedGap=3.4e38f;
        bool completeSpectrum=true;
        for(uint32_t i=0;i<Order();++i){
            float value=d.GetValue(i);
            if(!(value==value) || Abs(value)>2.0f*frobenius)completeSpectrum=false;
            if(i){float gap=value-d.GetValue(i-1);if(gap<observedGap)observedGap=gap;}
        }
        // All columns must also pass the original Gram/residual checks.
        // Their joint bounds and the polar factor give an ordered spectral
        // perturbation below0.001*||A||F. Subtract0.01*||A||F and halve
        // the remaining observed gap, retaining substantial rounding margin.
        const float candidateGap=0.5f*(observedGap-0.01f*frobenius);
        const bool useObservedGap=completeSpectrum && candidateGap>safeGap;
        if(useObservedGap)safeGap=candidateGap;
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*pitch];
            Mul(product,allQ,q,pitch,uint8_t(Order()),BinaryRepeatParams{1,1,1,uint8_t(pitch/8),uint8_t(pitch/8),0});
            Duplicate(w,0.0f,pitch);PipeBarrier<PIPE_V>();
            WholeReduceSum(w,product,pitch,uint8_t(Order()),1,1,uint8_t(pitch/8));VectorToScalar();
            float gram=0.0f;
            for(uint32_t j=0;j<Order();++j){float value=w.GetValue(j)-(j==EigenIndex(r)?1.0f:0.0f);gram+=value*value;}
            SetVectorMask<float>(pitch);DenseProduct<true>(ph,pm,original,zero,q,v);Barrier();
            ExtendedFloat rayleigh(0.0f),length(0.0f);
            float eigenvalue=0.0f,residual2=0.0f;
            if(Order()>=8){
                uint32_t transposePitch=(Order()+15)/16*16;
                auto high=arena[11*dense],low=high[pitch*transposePitch];
                TransposeSmallProducts(ph,pm,high,low,transposePitch);
                Duplicate(w,0.0f,pitch);Duplicate(red,0.0f,pitch);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float>(pitch);
                // Identical ascending column order, with independent rows
                // in SIMD lanes. Retain every two-float sum component.
                for(uint32_t j=0;j<Order();++j)
                    SmallSum<false>(w,red,w,red,high[j*transposePitch],low[j*transposePitch]);
                SmallProduct(ph,pm,q,v,w,red);Barrier();
                for(uint32_t i=0;i<Order();++i)rayleigh=rayleigh+ExtendedFloat(ph.GetValue(i),pm.GetValue(i),0.0f);
                SmallProduct(ph,pm,q,v,q,v);Barrier();
                for(uint32_t i=0;i<Order();++i)length=length+ExtendedFloat(ph.GetValue(i),pm.GetValue(i),0.0f);
                eigenvalue=(rayleigh/length).Float();
                SmallScale<false>(ph,pm,q,v,ExtendedFloat(eigenvalue));
                SmallSum<true>(ph,pm,w,red,ph,pm);Barrier();
                Add(product,ph,pm,pitch);PipeBarrier<PIPE_V>();Mul(product,product,product,pitch);PipeBarrier<PIPE_V>();
                ReduceSumWithArena(red,product,tmp,pitch);VectorToScalar();residual2=red.GetValue(0);
            }else{
                for(uint32_t i=0;i<Order();++i){
                    ExtendedFloat aq(0.0f);
                    for(uint32_t j=0;j<Order();++j)aq=aq+ExtendedFloat(ph.GetValue(i*pitch+j),pm.GetValue(i*pitch+j),0.0f);
                    w.SetValue(i,aq.h);red.SetValue(i,aq.m);
                    ExtendedFloat x(q.GetValue(i));rayleigh=rayleigh+x*aq;length=length+x*x;
                }
                eigenvalue=(rayleigh/length).Float();
                for(uint32_t i=0;i<Order();++i){
                    ExtendedFloat residual=ExtendedFloat(w.GetValue(i),red.GetValue(i),0.0f)-ExtendedFloat(eigenvalue)*ExtendedFloat(q.GetValue(i));
                    float value=residual.Float();residual2+=value*value;
                }
            }
            const float tolerance=1e-4f+1e-4f*Abs(eigenvalue);
            const float arithmetic=1e-12f*float(int32_t(Order()*Order()))*smallRowBound+1.1920928955078125e-7f*Abs(eigenvalue);
            bool bad=!(eigenvalue==eigenvalue) || !(residual2==residual2) || !(gram==gram) || !(safeGap>0.0f);
            bad=bad || Abs(eigenvalue)>3.4e38f || gram>9e-10f/float(int32_t(Order()));
            bad=bad || residual2>6.25e-10f/float(int32_t(Order()))*frobenius2;
            bad=bad || residual2>safeGap*safeGap*0.0625f;
            bad=bad || (useObservedGap && Abs(eigenvalue-d.GetValue(EigenIndex(r)))>1e-4f*perColumn);
            bad=bad || 2.0f*residual2/safeGap+arithmetic>0.25f*tolerance;
            pack.SetValue(8*r,bad?1.0f:0.0f);
            pack.SetValue(8*(count+r),PrivateSmall && bad?3.402823466e38f:eigenvalue);
        }
        event_t ready=ImmediateEvent<HardEvent::S_MTE3>();
        SetFlag<HardEvent::S_MTE3>(ready);WaitFlag<HardEvent::S_MTE3>(ready);
        if constexpr(PrivateSmall){
            // Each current eigenvalue slot temporarily carries its refined
            // value or a rejected-certificate marker. Final output overwrites it.
            DataCopyPad(values[uint64_t(currentBatch)*Order()+lane],pack[count*8],
                DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});
        }else{
            DataCopyPad(gw[lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});
            DataCopyPad(gw[pitch+lane],pack[count*8],DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});
        }Barrier();
    }
    __aicore__ inline bool AcceptSmallQL(){
        if constexpr(PrivateSmall){
            DataCopyPad(d,values[uint64_t(currentBatch)*Order()],DataCopyExtParams{1,Order()*4,0,0,0},
                DataCopyPadExtParams<float>{true,0,uint8_t(Stride()-Order()),0});
            Duplicate(tmp,0.0f,Stride());
        }else{DataCopy(tmp,gw,Stride());DataCopy(d,gw[Stride()],Stride());}Barrier();
        bool accepted=true;
        for(uint32_t i=0;i<Order();++i){
            float value=d.GetValue(i);
            if(tmp.GetValue(i)!=0.0f || !(value==value) || Abs(value)>3.4e38f)accepted=false;
            if(i && value<d.GetValue(i-1))accepted=false;
        }
        return accepted;
    }

    __aicore__ inline float SmallHypot(float x,float y){
        x=Abs(x);y=Abs(y);float scale=x>y?x:y;
        if(scale==0.0f)return 0.0f;
        x/=scale;y/=scale;return scale*SmallNativeRoot(x*x+y*y);
    }
    // Implicit shifted QL of the complete current tridiagonal. Store the
    // accumulated orthogonal eigenvectors as contiguous transposed rows.
    // Algorithm reference: LAPACK SSTEQR, https://netlib.org/lapack/single/ssteqr.f
    __aicore__ __attribute__((noinline)) bool SolvePrivateSmallQL(){
        const uint32_t pitch=Stride();
        auto z=smallStorage.template Get<float>()[3*Order()*pitch];
        const bool selective=count<=2;
        const uint32_t rotationCapacity=3*Order()*pitch;
        uint32_t rotationCount=0;
        auto diagonal=pack,off=pack[pitch];
        DataCopy(diagonal,origD,pitch);DataCopy(off,origE,pitch);
        if(!selective)Duplicate(z,0.0f,Order()*pitch);Barrier();
        off.SetValue(Order()-1,0.0f);
        if(!selective)for(uint32_t j=0;j<Order();++j)z.SetValue(j*pitch+j,1.0f);
        Barrier();uint32_t iterations=0;
        for(uint32_t first=0;first<Order();++first){
            while(true){
                uint32_t last=first;
                for(;last+1<Order();++last){
                    const float threshold=5.960464477539063e-8f*(Abs(diagonal.GetValue(last))+Abs(diagonal.GetValue(last+1)))+1e-30f;
                    if(Abs(off.GetValue(last))<=threshold)break;
                }
                if(last==first)break;
                if(++iterations>30*Order())return false;
                float shift=(diagonal.GetValue(first+1)-diagonal.GetValue(first))/(2.0f*off.GetValue(first));
                float radius=SmallHypot(shift,1.0f);
                float denominator=shift+(shift>=0.0f?radius:-radius);
                if(denominator==0.0f)return false;
                shift=diagonal.GetValue(last)-diagonal.GetValue(first)+off.GetValue(first)/denominator;
                float sine=1.0f,cosine=1.0f,correction=0.0f;
                bool restart=false;
                for(int j=int(last)-1;j>=int(first);--j){
                    uint32_t i=uint32_t(j);
                    float lower=sine*off.GetValue(i),upper=cosine*off.GetValue(i);
                    radius=SmallHypot(lower,shift);off.SetValue(i+1,radius);
                    if(radius==0.0f){
                        diagonal.SetValue(i+1,diagonal.GetValue(i+1)-correction);
                        off.SetValue(last,0.0f);restart=true;break;
                    }
                    sine=lower/radius;cosine=shift/radius;
                    shift=diagonal.GetValue(i+1)-correction;
                    radius=(diagonal.GetValue(i)-shift)*sine+2.0f*cosine*upper;
                    correction=sine*radius;diagonal.SetValue(i+1,shift+correction);
                    shift=cosine*radius-upper;
                    // Capacity failure is independent of lane ownership,
                    // keeping fallback decisions uniform inside each matrix.
                    if(rotationCount>=rotationCapacity)return false;
                    if(selective){
                        z.SetValue(4*rotationCount,cosine);
                        z.SetValue(4*rotationCount+1,sine);
                        z.template ReinterpretCast<uint32_t>().SetValue(4*rotationCount+2,i);
                    }else{
                        auto left=z[i*pitch],right=z[(i+1)*pitch];
                        Muls(tmp,left,sine,pitch);PipeBarrier<PIPE_V>();
                        Axpy(tmp,right,cosine,pitch);PipeBarrier<PIPE_V>();
                        Muls(left,left,cosine,pitch);PipeBarrier<PIPE_V>();
                        Axpy(left,right,-sine,pitch);PipeBarrier<PIPE_V>();
                        DataCopy(right,tmp,pitch);Barrier();
                    }
                    ++rotationCount;
                }
                if(restart)continue;
                diagonal.SetValue(first,diagonal.GetValue(first)-correction);
                off.SetValue(first,shift);off.SetValue(last,0.0f);
            }
        }
        auto order=piv.template ReinterpretCast<uint32_t>();
        for(uint32_t j=0;j<Order();++j){
            float value=diagonal.GetValue(j);
            if(!(value==value) || Abs(value)>3.4e38f)return false;
            order.SetValue(j,j);
        }
        for(uint32_t j=1;j<Order();++j){
            float value=diagonal.GetValue(j);uint32_t index=order.GetValue(j),position=j;
            while(position && diagonal.GetValue(position-1)>value){
                diagonal.SetValue(position,diagonal.GetValue(position-1));
                order.SetValue(position,order.GetValue(position-1));--position;
            }
            diagonal.SetValue(position,value);order.SetValue(position,index);
        }
        for(uint32_t j=0;j<Order();++j)d.SetValue(j,diagonal.GetValue(j));
        Barrier();
        for(uint32_t r=0;r<count;++r){
            uint32_t column=order.GetValue(EigenIndex(r));
            if(selective){
                auto q=mat[r*pitch];Duplicate(q,0.0f,pitch);Barrier();q.SetValue(column,1.0f);
                for(uint32_t k=rotationCount;k>0;){
                    --k;uint32_t i=z.template ReinterpretCast<uint32_t>().GetValue(4*k+2);
                    float cosine=z.GetValue(4*k),sine=z.GetValue(4*k+1);
                    float left=q.GetValue(i),right=q.GetValue(i+1);
                    q.SetValue(i,cosine*left+sine*right);
                    q.SetValue(i+1,cosine*right-sine*left);
                }
            }else DataCopy(mat[r*pitch],z[column*pitch],pitch);
        }
        Barrier();return true;
    }

    __aicore__ inline void Inverse(){
        if(!ResidentFamily() && count>=2){InverseVector();return;}
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];float ev=d.GetValue(EigenIndex(r));
            if(!ResidentFamily() && (Order()&(Order()-1))!=0)ev=PivotedExtendedEigenvector(q,EigenIndex(r),ev);
            else ev=ScalarEigenvector(q,EigenIndex(r),ev);d.SetValue(EigenIndex(r),ev);
            Barrier(); // Eigenvector column signs are free under the contract.
        }
        Backtransform();
    }

    __aicore__ inline void CanonicalizeColumns(){
        // Reduce in original row order so equal maxima choose the first row.
        if(Order()<=2){
            for(uint32_t r=0;r<count;++r){
                auto q=mat[r*Stride()];float magnitude=-1.0f,pivot=0.0f;
                for(uint32_t row=0;row<Order();++row){float value=q.GetValue(upper?Order()-1-row:row);float a=Abs(value);if(a>magnitude){magnitude=a;pivot=value;}}
                if(pivot<0.0f){Muls(q,q,-1.0f,Stride());Barrier();}
            }
            return;
        }
        auto offsets=piv.template ReinterpretCast<int32_t>();
        if(upper){
            CreateIndicesWithArena(offsets,int32_t(0),Stride());PipeBarrier<PIPE_V>();
            Muls(offsets,offsets,int32_t(-4),Stride());PipeBarrier<PIPE_V>();
            Adds(offsets,offsets,int32_t((Order()-1)*4),Stride());Barrier();
        }
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];
            if(upper){Gather(tmp2,q,offsets.template ReinterpretCast<uint32_t>(),0,Order());PipeBarrier<PIPE_V>();AscendC::Abs(tmp2,tmp2,Order());}
            else AscendC::Abs(tmp2,q,Order());
            PipeBarrier<PIPE_V>();
            uint32_t full=Order()/64,tail=Order()%64;
            if(full)WholeReduceMax(red,tmp2,64,int32_t(full),1,1,8);
            if(tail)WholeReduceMax(param,tmp2[full*64],int32_t(tail),1,1,1,8);
            VectorToScalar();
            float magnitude=-1.0f;uint32_t index=0;
            for(uint32_t block=0;block<full;++block){
                float value=red.GetValue(2*block);
                if(value>magnitude){magnitude=value;index=64*block+red.template ReinterpretCast<uint32_t>().GetValue(2*block+1);}
            }
            if(tail && param.GetValue(0)>magnitude)index=64*full+param.template ReinterpretCast<uint32_t>().GetValue(1);
            float pivot=q.GetValue(upper?Order()-1-index:index);
            if(pivot<0.0f){Muls(q,q,-1.0f,Stride());Barrier();}
        }
    }
    __aicore__ inline void Output(uint32_t b){
        if(count){for(uint32_t r=0;r<count;++r)pack.SetValue(8*r,d.GetValue(EigenIndex(r)));Barrier();DataCopyPad(values[uint64_t(b)*Order()+lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});Barrier();}
        auto offsets=piv.template ReinterpretCast<uint32_t>();
        // For Order()>2, CanonicalizeColumns already prepared the original-row offsets.
        if(upper && Order()<=2){for(uint32_t i=0;i<Order();++i)offsets.SetValue(i,(Order()-1-i)*4);Barrier();}
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];
            if(upper){Duplicate(tmp,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(tmp,q,offsets,0,Order());PipeBarrier<PIPE_V>();Brcb(pack,tmp,uint8_t((Order()+7)/8),{1,8});}
            else Brcb(pack,q,uint8_t((Order()+7)/8),{1,8});Barrier();
            DataCopyPad(vectors[uint64_t(b)*Order()*Order()+EigenIndex(r)],pack,DataCopyExtParams{uint16_t(Order()),4,0,(Order()-1)*4,0});Barrier();
        }
    }
    __aicore__ inline void OutputStaged(uint32_t b){
        // All vector lanes finish reading the reflector history before that
        // dead storage becomes the transposed-output staging matrix.
        if constexpr(CubeMode==0)SyncAll();
        if(active){
            if(count){for(uint32_t r=0;r<count;++r)pack.SetValue(8*r,d.GetValue(EigenIndex(r)));Barrier();DataCopyPad(values[uint64_t(b)*Order()+lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});}
            auto stage=hist[CubeMode?(Order()+5)*Stride():0];Barrier();
            for(uint32_t r=0;r<count;++r)DataCopy(stage[EigenIndex(r)*Stride()],mat[r*Stride()],Stride());
        }
        SyncAll();
        if(active){
            auto stage=hist[CubeMode?(Order()+5)*Stride():0];
            uint32_t outputRows=base<Order()?(Order()-base<rows?Order()-base:rows):0;
            if(outputRows){
                uint32_t pitch=(outputRows+7)/8*8,first=upper?Order()-base-outputRows:base;
                Duplicate(prod,0.0f,Stride()*pitch);Barrier();
                DataCopyPad(prod,stage[first],DataCopyExtParams{uint16_t(Order()),outputRows*4,(Stride()-outputRows)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                Barrier();
                if(Stride()%16==0){
                    uint64_t source[16],destination[16];
                    TransDataTo5HDParams params{false,false,uint8_t(Stride()/16),2,uint16_t(pitch*2)};
                    for(uint32_t col=0;col<pitch;col+=8){
                        for(uint32_t i=0;i<16;++i)source[i]=uint64_t(prod[i*pitch+col].GetPhyAddr());
                        for(uint32_t i=0;i<8;++i){
                            uint32_t logical=col+i,row=logical<outputRows?(upper?outputRows-1-logical:logical):logical;
                            destination[2*i]=uint64_t(mat[row*Stride()].GetPhyAddr());
                            destination[2*i+1]=uint64_t(mat[row*Stride()+8].GetPhyAddr());
                        }
                        TransDataTo5HD<float>(destination,source,params);PipeBarrier<PIPE_V>();
                    }
                }else{
                    auto offsets=piv.template ReinterpretCast<int32_t>();
                    CreateIndicesWithArena(offsets,int32_t(0),Stride());PipeBarrier<PIPE_V>();Muls(offsets,offsets,int32_t(pitch*4),Stride());Barrier();
                    for(uint32_t r=0;r<outputRows;++r){
                        auto row=mat[r*Stride()];Duplicate(row,0.0f,Stride());PipeBarrier<PIPE_V>();
                        Gather(row,prod,offsets.template ReinterpretCast<uint32_t>(),(upper?outputRows-1-r:r)*4,Order());
                    }
                }
                Barrier();
                // All readers must finish their transpose loads before any
                // final row store replaces output-backed history/Q staging.
                if(outputBackedHistory)SyncAll();
                if(Order()==Stride())DataCopy(vectors[uint64_t(b)*Order()*Order()+base*Order()],mat,outputRows*Stride());
                else DataCopyPad(vectors[uint64_t(b)*Order()*Order()+base*Order()],mat,DataCopyExtParams{uint16_t(outputRows),Order()*4,(Stride()-(Order()+7)/8*8)/8,0,0});
                Barrier();
            }
        }
    }
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR ev,GM_ADDR eq,GM_ADDR work,LinalgEighTilingData t){
        if constexpr(DirectUb){pipe.next=0;InitSocState();}
        if constexpr(ResidentEntry){if(t.n>=3 && t.n<=25)pipe.InitBuffer(smallStorage,15*t.n*t.stride*4);}
        if constexpr(PrivateSmall){pipe.InitBuffer(historyStorage,2*t.n*t.stride*4);privateHistory=historyStorage.template Get<float>();}
        else if constexpr(CubeMode==0){
            if(t.n>25 && t.n<=64){pipe.InitBuffer(historyStorage,t.n*t.stride*4);privateHistory=historyStorage.template Get<float>();}
        }
        outputBackedHistory=(t.vectorFlags&EIGH_DISTRIBUTED_OUTPUT_HISTORY)!=0;
        privateTailColumnCount=(t.vectorFlags&EIGH_DISTRIBUTED_TAIL_96)?96:64;
        packedMatrix=false;reservedLayout=false;dynamicOrder=t.n;dynamicStride=t.stride;batch=t.batch;upper=t.upper;team=t.team;groups=t.groups;rows=t.rows;laneStride=(rows+7)/8*8;
        uint32_t physicalTeam=GetTaskRation()==2?(team+1)/2*2:team;lane=GetBlockIdx()%physicalTeam;group=GetBlockIdx()/physicalTeam;base=lane*rows;count=base<Order()?((Order()-base)<rows?(Order()-base):rows):0;
        input.SetGlobalBuffer((__gm__ float*)a);values.SetGlobalBuffer((__gm__ float*)ev);vectors.SetGlobalBuffer((__gm__ float*)eq);ws.SetGlobalBuffer((__gm__ float*)work);
        uint32_t wideN=ResidentFamily()?Order():25;uint32_t wideSize=3*wideN*wideN+21*wideN+64;wideSize=wideSize>8*Stride()?wideSize:8*Stride();wideSize=wideSize>((Stride()+1)/2)*laneStride?wideSize:((Stride()+1)/2)*laneStride;privateCapacity=wideSize+((!ResidentFamily() && Stride()%64==0)?64:0);pipe.InitBuffer(wideStorage,(wideSize+((!ResidentFamily() && Stride()%64==0)?128:0))*4);wideBuf=wideStorage.template Get<float>()[((!ResidentFamily() && Stride()%64==0)?64:0)];
        uint32_t dense=rows*Stride()>Stride()*laneStride?rows*Stride():Stride()*laneStride;
        uint32_t color=(!ResidentFamily() && Stride()%64==0)?1:0,alternate=Stride()%128==64?1:0;
        uint32_t storageSize=3*dense+((dense+1)/2+7)/8*8+22*Stride()+64+(128+128*alternate)*color;storageSize=!ResidentFamily() && storageSize<8192?8192:storageSize;// PrivateTail preserves metadata in wideStorage before recycling the
        // matrix arena. Reserve the complete matrix plus product tile.
        if(privateTailColumnCount==96 && storageSize<2*96*128)storageSize=2*96*128;
        storageCapacity=storageSize;pipe.InitBuffer(storage,storageSize*4);auto b=storage.template Get<float>();mat=b;prod=b[dense+64*color];pcrBand=b[2*dense+64*color];pcrLeft=b[3*dense+64*color];uint32_t off=3*dense+((dense+1)/2+7)/8*8+64*color;
        // Keep matrix/products, reflector data and partial sums at distinct
        // repeated UB offsets. The matrix and vector arithmetic is unchanged.
        v=b[off];tmp=b[off+Stride()+64*alternate*color];tmp2=b[off+2*Stride()+128*alternate*color];red=b[off+3*Stride()+(64+64*alternate)*color];param=b[off+4*Stride()+(64+64*alternate)*color];
        d=b[off+5*Stride()+(64+64*alternate)*color];e=b[off+6*Stride()+(64+64*alternate)*color];origD=b[off+7*Stride()+(64+64*alternate)*color];origE=b[off+8*Stride()+(64+64*alternate)*color];beta=b[off+9*Stride()+(64+64*alternate)*color];
        piv=b[off+10*Stride()+(64+64*alternate)*color];fac=b[off+11*Stride()+(64+64*alternate)*color];w=b[off+12*Stride()+(64+128*alternate)*color];pack=b[off+13*Stride()+(64+128*alternate)*color];
    }
    __aicore__ inline void ReduceByShape(uint32_t b){
        if constexpr(ResidentEntry){
            if(Order()>=3 && ResidentFamily()){
                if constexpr(PrivateSmall){
                    if(groups==1 && rows<=2){
                        bool owned=active;active=true;
                        ReduceReplicatedFloatSmall(currentBatch);
                        active=owned;return;
                    }
                }
                ReduceReplicatedFloatSmall(b);return;
            }
            if(ResidentFamily()){ReduceExtended();return;}
        }
        if(active)PackMatrix();Reduce();if(active)UnpackMatrix();
    }
    __aicore__ __attribute__((noinline)) void Process(){
        uint32_t rounds=(batch+groups-1)/groups;
        uint32_t compareLength=(Order()+63)/64*64;
        for(uint32_t step=0;step<rounds;++step){
            smallFloatQL=false;smallQLSolved=false;smallQLNeedsCheck=false;smallQLReadyOutput=false;smallRowBound=0.0f;uint32_t b=group<groups?step*groups+group:batch;active=b<batch && lane<team;count=base<Order()?((Order()-base)<rows?(Order()-base):rows):0;directCoefficients=(base%8==0 && count%8==0 && count>0);
            currentBatch=b;
            if constexpr(PrivateSmall){if(groups==1 && rows<=2)currentBatch=step;}
            if(active){
                if constexpr(!PrivateSmall){
                    if(outputBackedHistory){
                        // This call writes every reflector before consuming it.
                        hist=vectors[uint64_t(b)*Order()*Order()];
                        gw=ws[uint64_t(b)*WorkspacePerMatrix()];gparam=gw[4*Stride()];
                    }else{
                        hist=ws[uint64_t(b)*WorkspacePerMatrix()];
                        gw=hist[(ResidentFamily()?2*Order():Order())*Stride()];
                        gparam=gw[(ResidentFamily()?2:4)*Stride()];
                    }
                }
                if(Order()>=3 && ResidentFamily() && rows<=2 && lane==0)PublishSmallMode(false);
                Duplicate(d,0.0f,Stride());Duplicate(e,0.0f,Stride());Duplicate(beta,0.0f,Stride());Barrier();
                if(!ResidentFamily()){
                    auto columnOffsets=piv.template ReinterpretCast<uint32_t>(),rowOffsets=tmp2.template ReinterpretCast<uint32_t>();
                    CreateIndicesWithArena(rowOffsets.template ReinterpretCast<int32_t>(),int32_t(0),Stride());PipeBarrier<PIPE_V>();
                    // Compares(calCount) requires a whole 64-float repeat, including the tail.
                    CreateIndicesWithArena(wideBuf.template ReinterpretCast<int32_t>(),int32_t(0),compareLength);PipeBarrier<PIPE_V>();
                    Cast(wideBuf[compareLength],wideBuf.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,compareLength);PipeBarrier<PIPE_V>();
                    if(upper){Muls(rowOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(-1),Stride());PipeBarrier<PIPE_V>();Adds(rowOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(Order()-1),Stride());PipeBarrier<PIPE_V>();}
                    Muls(rowOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(4),Stride());PipeBarrier<PIPE_V>();
                    Muls(columnOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(Order()%64==0 && count?(count+7)/8*8:8),Stride());Barrier();
                }
                if(Order()<3 || !ResidentFamily()){
                // Fetch a block of rows and its transposed counterpart using
                // two DMA commands when the owned columns are block aligned.
                if(!ResidentFamily() && Stride()%16==0 && count){
                    uint32_t first=upper?Order()-base-count:base;
                    const uint32_t columnPitch=(count+7)/8*8;
                    if(Order()==Stride()){
                        DataCopy(prod,input[uint64_t(b)*Order()*Order()+uint64_t(first)*Order()],count*Stride());
                    }else{
                        // The raw row copy and transposed column copy have
                        // different physical pitches. Initialize every padded
                        // UB lane before the bulk copies and 16-row transpose.
                        Duplicate(mat,0.0f,Stride()*laneStride);
                        Duplicate(prod,0.0f,Stride()*laneStride);
                        Duplicate(pcrBand,0.0f,Stride()*laneStride);Barrier();
                        DataCopyPad(prod,input[uint64_t(b)*Order()*Order()+uint64_t(first)*Order()],
                            DataCopyExtParams{uint16_t(count),Order()*4,0,(Stride()-(Order()+7)/8*8)/8,0},
                            DataCopyPadExtParams<float>{false,0,0,0});
                    }
                    DataCopyPad(pcrBand,input[uint64_t(b)*Order()*Order()+first],DataCopyExtParams{uint16_t(Order()),count*4,(Order()-count)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                    Barrier();
                    auto columnOffsets=piv.template ReinterpretCast<uint32_t>(),rowOffsets=tmp2.template ReinterpretCast<uint32_t>();

                    uint64_t source[16],destination[16];
                    TransDataTo5HDParams params{false,false,uint8_t((Order()+15)/16),2,uint16_t(columnPitch*2)};
                    for(uint32_t col=0;col<count;col+=8){
                        for(uint32_t i=0;i<16;++i)source[i]=uint64_t(pcrBand[i*columnPitch+col].GetPhyAddr());
                        for(uint32_t i=0;i<8;++i){
                            uint32_t logical=col+i;
                            uint32_t row=logical<count?(upper?count-1-logical:logical):logical;
                            destination[2*i]=uint64_t(mat[row*Stride()].GetPhyAddr());
                            destination[2*i+1]=uint64_t(mat[row*Stride()+8].GetPhyAddr());
                        }
                        TransDataTo5HD<float>(destination,source,params);PipeBarrier<PIPE_V>();
                    }
                    for(uint32_t r=0;r<count;++r){
                        auto a=mat[r*Stride()];uint32_t i=base+r,physical=upper?count-1-r:r;
                        if(upper)Gather(tmp,a,rowOffsets,0,Order());
                        Gather(w,prod[physical*Stride()],rowOffsets,0,Order());PipeBarrier<PIPE_V>();
                        auto selected=beta.template ReinterpretCast<uint8_t>();Compares(selected,wideBuf[compareLength],float(int32_t(i)),CMPMODE::LE,compareLength);PipeBarrier<PIPE_V>();
                        if(upper)Select(a,selected,w,tmp,SELMODE::VSEL_TENSOR_TENSOR_MODE,Order());
                        else Select(a,selected,w,a,SELMODE::VSEL_TENSOR_TENSOR_MODE,Order());
                        Barrier();
                    }
                }else{
                for(uint32_t r=0;r<count;++r){auto a=mat[r*Stride()];Duplicate(a,0.0f,Stride());Barrier();uint32_t i=base+r;
                    if(!ResidentFamily()){
                        uint32_t x=upper?Order()-1-i:i;
                        DataCopyPad(tmp,input[uint64_t(b)*Order()*Order()+x*Order()],DataCopyExtParams{1,Order()*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                        DataCopyPad(pack,input[uint64_t(b)*Order()*Order()+x],DataCopyExtParams{uint16_t(Order()),4,(Order()-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});Barrier();
                        {
                            Gather(a,pack,piv.template ReinterpretCast<uint32_t>(),0,Order());Gather(w,tmp,tmp2.template ReinterpretCast<uint32_t>(),0,Order());PipeBarrier<PIPE_V>();
                            auto selected=beta.template ReinterpretCast<uint8_t>();Compares(selected,wideBuf[compareLength],float(int32_t(i)),CMPMODE::LE,compareLength);PipeBarrier<PIPE_V>();
                            Select(a,selected,w,a,SELMODE::VSEL_TENSOR_TENSOR_MODE,Order());Barrier();
                        }
                    }else for(uint32_t j=0;j<Order();++j){uint32_t x=upper?Order()-1-i:i,y=upper?Order()-1-j:j;if((upper && x>y)||(!upper && x<y)){uint32_t z=x;x=y;y=z;}a.SetValue(j,input.GetValue(uint64_t(b)*Order()*Order()+x*Order()+y));}
                }
                }
                }
                Barrier();
            }
            ReduceByShape(b);if(!ResidentFamily() && BackPanelWidth()){if constexpr(CubeMode==2){if(b<batch){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);}SyncAll();}if(active)PrepareBackPanels();SyncAll();if constexpr(CubeMode==2){if(b<batch)CrossCoreSetFlag<2,PIPE_MTE3>(8);}}count=lane<Order()?(Order()-1-lane)/team+1:0;
            if(Order()>=3 && ResidentFamily() && rows<=2 && AnySmallMode(step)){
                if(active && smallQLNeedsCheck && smallQLSolved)Backtransform();
                if constexpr(!PrivateSmall)SyncAll();
                if(active && smallQLNeedsCheck && smallQLSolved){
                    for(uint32_t r=0;r<count;++r){
                        if constexpr(PrivateSmall)DataCopyPad(vectors[uint64_t(b)*Order()*Order()+EigenIndex(r)*Order()],mat[r*Stride()],DataCopyExtParams{1,Order()*4,0,0,0});
                        else DataCopy(hist[EigenIndex(r)*Stride()],mat[r*Stride()],Stride());
                    }
                    Barrier();
                }
                SyncAll();
                if(active && smallQLNeedsCheck && smallQLSolved)CheckSmallQL();
                SyncAll();
                if(active && smallQLNeedsCheck && smallQLSolved){
                    if(AcceptSmallQL())smallQLReadyOutput=true;
                    else{smallQLSolved=false;ReduceReplicatedSmall(b,false);}
                }
                SyncAll();
            }
            if(active){
                if constexpr(ResidentEntry){
                    if(smallQLSolved){if(!smallQLReadyOutput)Backtransform();}
                    else{Bisect();Barrier();DataCopy(param,d,Stride());Barrier();Inverse();}
                }else{Bisect();Barrier();DataCopy(param,d,Stride());Barrier();Inverse();}
                // Keep the same deterministic column orientation as the
                // validated release, including original-row tie ordering.
                CanonicalizeColumns();
            }else if(b<batch && lane>=team){if(Stride()>=64 && batch%groups==0)SyncAll();if(BackPanelWidth())CubeBackHandshake(false);}else if(b>=batch){if(Stride()>=64 && batch%groups==0)SyncAll();}if(!ResidentFamily())OutputStaged(b);else if(active)Output(b);if constexpr(!PrivateSmall)SyncAll();if(BackPanelWidth() && b<batch)CrossCoreSetFlag<2,PIPE_MTE3>(6);
        }
    }
};



// Matrices fitting one SIMD row use the repository's independent-owner
// solver. Each owner performs the complete current-input reduction, solves
// its eigenvalue interval, and participates in residual/Gram certification.
namespace CompleteResidentEigh {
template <typename DT_A,uint32_t FixedStride=0,uint32_t CubeMode=0,uint32_t FixedOrder=0,bool PrivateSmall=false,bool DirectUb=false,bool IndependentMedium=false,bool IsolatedSmall=false>
class Solver {
    typename EighMemoryTypes<DirectUb>::Pipe pipe;
    typename EighMemoryTypes<DirectUb>::Buffer storage,wideStorage,smallStorage,historyStorage,refinementStorage;
    LocalTensor<float> privateHistory;
    uint32_t currentBatch;
    LocalTensor<float> mat,prod,v,w,tmp,tmp2,red,param,d,e,origD,origE,beta,piv,fac,pack,wideBuf,pcrBand,pcrLeft;
    GlobalTensor<float> input,values,vectors,ws,hist,gw,gparam,gramScratch;
    uint32_t dynamicOrder,dynamicStride,batch,upper,team,groups,rows,laneStride,lane,group,base,count;
    uint32_t privateCapacity,storageCapacity;
    bool packedMatrix,mediumVectorReflectors,directCoefficients;
    __aicore__ inline uint32_t Order()const{
        if constexpr(FixedOrder!=0)return FixedOrder;
        return dynamicOrder;
    }
    __aicore__ inline uint32_t Stride()const{
        if constexpr(FixedStride!=0)return FixedStride;
        return dynamicStride;
    }
    // Both the dynamic resident entry and fixed 16-wide resident tile
    // own small matrices. Cube entries remain outside this family.
    static constexpr bool ResidentEntry=(FixedStride<=32 || FixedOrder!=0) && CubeMode==0;
    static constexpr bool StaticResidentStorage=PrivateSmall && FixedStride>=8 && FixedStride<=32 && CubeMode==0;
    __aicore__ inline bool ResidentFamily()const{
        // Static private tiles are only instantiated by the validated
        // resident dispatch, whose matrix order is in [3,25].
        if constexpr(StaticResidentStorage)return true;
        if constexpr(!ResidentEntry)return false;
        return Order()<=25;
    }
    float spectralBound,spectralCenter,matrixScale,commonDiagonalShift;
    bool active,smallFloatQL,smallQLSolved,smallQLNeedsCheck,smallQLReadyOutput;
    bool mediumVectorOriginalQuotient;
    bool mediumBoundaryConsensus,mediumBoundaryConsensusReady;
    uint32_t mediumWaveFirstMatrix,mediumWaveMatrixCount;
    bool mediumOutputScratch,mediumTwoInverseIterations;
    bool smallTwistedFactorization,mediumCertifiedFloatRayleigh,mediumPrivateSolve,mediumSharedReduction,mediumPrefersCooperative;
    bool useExtendedEigenvectors;
    uint64_t extendedEigenvectorMask;
    // Sturm queries retire before ownership-boundary classification begins.
    // Reuse this scalar slot without enlarging unrelated solver instances.
    uint64_t boundaryClusterMask;
    uint64_t possibleCloseMask;
    bool hasClusterGrid;
    float smallRowBound;
    float tridiagonalFrobeniusSquared,ownedResidualSquared;
    // Cyclic ownership balances the work for localized eigenvectors.
    __aicore__ inline uint32_t EigenIndex(uint32_t r){if constexpr(IndependentMedium)return base+r;return lane+r*team;}
    __aicore__ inline uint32_t OwnedRowCount()const{
        if constexpr(IndependentMedium){
            if(lane>=team)return 0;
            return Order()*(lane+1)/team-Order()*lane/team;
        }
        return base<Order()?(Order()-base<rows?Order()-base:rows):0;
    }
    __aicore__ inline void Barrier(){PipeBarrier<PIPE_ALL>();}
    __aicore__ inline float Abs(float a){return a<0?-a:a;}
    __aicore__ inline float Root(float a){
        if(a<=0)return 0;
        union Bits{float f;uint32_t u;}b;b.f=a;
        if(a>=1e-30f && a<=1e30f){
            b.u=0x5f375a86u-(b.u>>1);float reciprocal=b.f;
            for(uint32_t i=0;i<2;++i)reciprocal=reciprocal*(1.5f-((0.5f*a)*reciprocal)*reciprocal);
            b.f=a*reciprocal;b.u+=128u;float x=b.f;
            // The twice-corrected reciprocal and upward seed bias leave
            // less than 2e-5 relative seed error in this normal range.
            // One Newton correction reduces that below float32 roundoff.
            x=0.5f*(x+a/x);return x;
        }
        b.u=(b.u>>1)+0x1fc00000u;float x=b.f;
        x=0.5f*(x+a/x);x=0.5f*(x+a/x);x=0.5f*(x+a/x);x=0.5f*(x+a/x);return x;
    }
    template<HardEvent Event> __aicore__ inline event_t ImmediateEvent(){
        // Each flag is waited immediately, so no live event shares this ID.
        if constexpr(DirectUb)return EVENT_ID0;
        else return static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    }
    __aicore__ inline void ReduceSumWithArena(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float> scratch,uint32_t length){
        // Resident strides fit one SIMD repeat. WholeReduceSum has no implicit
        // TPipe event allocation; the caller synchronizes before scalar reads.
        if constexpr(DirectUb){
            if(length<=64)WholeReduceSum(dst,src,length,1,1,1,8);
            else{
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(length));
                vcadd((__ubuf__ float*)dst.GetPhyAddr(),(__ubuf__ float*)src.GetPhyAddr(),1,1,1,8,1);
                VectorToScalar();float value=GetReduceRepeatSumSpr<float>();
                SetMaskNorm();ResetMask();dst.SetValue(0,value);
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            }
        }
        else ReduceSum(dst,src,scratch,length);
    }
    __aicore__ inline void CreateIndicesWithArena(LocalTensor<int32_t> dst,int32_t first,uint32_t length){
        if constexpr(DirectUb){
            const uint32_t prefix=length<32?length:8;
            for(uint32_t i=0;i<prefix;++i)dst.SetValue(i,first+int32_t(i));
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            // Grow the initialized prefix by doubling; every read is initialized.
            for(uint32_t initialized=prefix;initialized<length;){
                uint32_t live=length-initialized<initialized?length-initialized:initialized;
                Adds(dst[initialized],dst,int32_t(initialized),live);PipeBarrier<PIPE_V>();
                initialized+=live;
            }
            PipeBarrier<PIPE_V>();
        }else CreateVecIndex(dst,first,length);
    }
    __aicore__ inline void VectorToScalar(){
        event_t id=ImmediateEvent<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(id);WaitFlag<HardEvent::V_S>(id);
    }
    __aicore__ inline void ScalarToVector(){
        event_t id=ImmediateEvent<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(id);WaitFlag<HardEvent::S_V>(id);
    }
    __aicore__ inline float Dot(LocalTensor<float>a,LocalTensor<float>b){
        Mul(tmp2,a,b,Stride());PipeBarrier<PIPE_V>();
#if __NPU_ARCH__ == 2201
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(Stride()));
        vcadd((__ubuf__ float*)red.GetPhyAddr(),(__ubuf__ float*)tmp2.GetPhyAddr(),1,1,1,8,1);
        VectorToScalar();float result=GetReduceRepeatSumSpr<float>();
        SetMaskNorm();ResetMask();return result;
#else
        ReduceSumWithArena(red,tmp2,tmp,Stride());VectorToScalar();return red.GetValue(0);
#endif
    }
    __aicore__ inline float DotTail(LocalTensor<float>a,LocalTensor<float>b,uint32_t begin){
#if __NPU_ARCH__ == 2201
        uint32_t first=begin/64*64,length=Stride()-first;
        Mul(tmp2[first],a[first],b[first],length);PipeBarrier<PIPE_V>();
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(length));
        vcadd((__ubuf__ float*)red.GetPhyAddr(),(__ubuf__ float*)tmp2[first].GetPhyAddr(),1,1,1,8,1);
        VectorToScalar();float result=GetReduceRepeatSumSpr<float>();
        SetMaskNorm();ResetMask();return result;
#else
        return Dot(a,b);
#endif
    }
    // Eight rows by eight columns per 256-byte SIMD tile. This is a
    // physical UB layout only; GM tensors and all inverse work stay compact.
    __aicore__ inline void PackMatrix(){
        packedMatrix=!ResidentFamily() && Stride()%64==0 &&
            (Stride()/64>=(count+7)/8+2 || count*8>4*Stride());
        if(!packedMatrix)return;
        uint32_t padded=(count+7)/8*8;
        Duplicate(prod,0.0f,padded*Stride());PipeBarrier<PIPE_V>();SetVectorMask<float>(8);
        for(uint32_t r=0;r<count;++r)
            Copy<float,false>(prod[(r/8)*8*Stride()+(r%8)*8],mat[r*Stride()],8,uint8_t(Stride()/8),CopyRepeatParams{1,1,8,1});
        Barrier();DataCopy(mat,prod,padded*Stride());Barrier();
    }
    __aicore__ inline void ReadPackedRow(LocalTensor<float> dst,uint32_t r){
        if(packedMatrix){
            Copy(dst,mat[(r/8)*8*Stride()+(r%8)*8],8,uint8_t(Stride()/8),CopyRepeatParams{1,1,1,8});
        }else DataCopy(dst,mat[r*Stride()],Stride());
    }
    __aicore__ inline void UnpackMatrix(){
        if(!packedMatrix)return;
        for(uint32_t r=0;r<count;++r)ReadPackedRow(prod[r*Stride()],r);
        Barrier();DataCopy(mat,prod,count*Stride());Barrier();packedMatrix=false;
    }
    template<uint32_t Width>
    __aicore__ inline void AdjacentLevel(LocalTensor<float> dst,LocalTensor<float> src,uint32_t chunks){
        // Independent adjacent sums share a SIMD repeat. A block holds the
        // same eight row results; changing the instruction mask preserves
        // each scalar pair and the complete three-level reduction tree.
        SetVectorMask<float>(8*Width);
        for(uint32_t first=0;first<chunks;first+=255){
            uint8_t repeats=uint8_t(chunks-first<255?chunks-first:255);
            Add<float,false>(dst[first*Width*8],src[first*Width*16],src[first*Width*16+8],8*Width,repeats,BinaryRepeatParams{1,2,2,uint8_t(Width),uint8_t(2*Width),uint8_t(2*Width)});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DotPackedRows(LocalTensor<float> vector,uint32_t begin){
        uint32_t blocks=(count+7)/8,first=begin/64*64,width=Stride()-first;
        Duplicate(w,0.0f,blocks*8);PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
        for(uint32_t rb=0;rb<blocks;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            Mul<float,false>(prod[off],mat[off],vector[first],64,uint8_t((Stride()-first)/8),BinaryRepeatParams{1,1,0,8,8,1});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t rb=0;rb<blocks;++rb)
            BlockReduceSum<float,false>(pcrLeft[rb*width],prod[rb*8*Stride()+first*8],int32_t((Stride()-first)/8),64,1,1,8);
        PipeBarrier<PIPE_V>();
        // VCGADD followed by this adjacent binary tree has the exact VCADD
        // 64-element order. Keep the original left fold between 64-column
        // chunks, including its initial zero.
        uint32_t chunks=blocks*(width/64);
        AdjacentLevel<4>(prod,pcrLeft,chunks);
        AdjacentLevel<2>(pcrLeft,prod,chunks);
        AdjacentLevel<1>(prod,pcrLeft,chunks);
        SetVectorMask<float>(8);
        for(uint32_t j=0;j<width/64;++j){
            Add<float,false>(w,w,prod[j*8],8,uint8_t(blocks),BinaryRepeatParams{1,1,1,1,1,uint8_t(width/64)});
            PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(64);
    }
    __aicore__ inline void UpdatePacked(uint32_t k){
        uint32_t padded=(count+7)/8*8,first=(k+1)/8*8;
        if(directCoefficients){
            Brcb(pcrLeft,w[base],uint8_t(padded/8),{1,8});Brcb(pcrBand,v[base],uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
        }else{
        Gather(tmp2,w,piv.template ReinterpretCast<uint32_t>(),0,padded);PipeBarrier<PIPE_V>();
        Brcb(pcrLeft,tmp2,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
        Gather(tmp2,v,piv.template ReinterpretCast<uint32_t>(),0,padded);PipeBarrier<PIPE_V>();
        Brcb(pcrBand,tmp2,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(64);
        uint32_t firstBlock=base>k+1?0:(k+1-base)/8;
        uint8_t repeats=uint8_t((Stride()-first)/8);
        for(uint32_t rb=firstBlock;rb<padded/8;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            Mul<float,false>(prod[off],v[first],pcrLeft[rb*64],64,repeats,BinaryRepeatParams{1,0,1,8,1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t rb=firstBlock;rb<padded/8;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            MulAddDst<float,float,false>(prod[off],w[first],pcrBand[rb*64],64,repeats,BinaryRepeatParams{1,0,1,8,1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t rb=firstBlock;rb<padded/8;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            Sub<float,false>(mat[off],mat[off],prod[off],64,repeats,BinaryRepeatParams{1,1,1,8,8,8});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DotRows(LocalTensor<float> vector,uint32_t begin=0){
        if(packedMatrix){DotPackedRows(vector,begin);return;}
        if constexpr(IndependentMedium){
            if(mediumVectorReflectors && Stride()==64){
                // One physical SIMD repeat covers every coordinate. The
                // ordinary block accumulator would only add this result to0.
                SetMaskNorm();SetVectorMask<float>(64);
                if(count%8){Duplicate(w,0.0f,(count+7)/8*8);PipeBarrier<PIPE_V>();}
                Mul<float,false>(prod,mat,vector,64,uint8_t(count),BinaryRepeatParams{1,1,1,8,8,0});
                PipeBarrier<PIPE_V>();
                WholeReduceSum<float,false>(w,prod,64,uint8_t(count),1,1,8);
                PipeBarrier<PIPE_V>();return;
            }
        }

        if(Stride()<64){
            // All owned rows share one reflector. Keep their inner products
            // in vector storage so the backtransform needs no scalar readback.
            Mul(prod,mat,vector,Stride(),uint8_t(count),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});
            Duplicate(w,0.0f,Stride());PipeBarrier<PIPE_V>();
            WholeReduceSum(w,prod,Stride(),uint8_t(count),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();return;
        }
        // Complete SIMD blocks share one mask across products, row reductions,
        // and the ordered sum of partials. Padding lanes remain zero.
        if(Stride()%64==0 && count<=64){
            Duplicate(w,0.0f,64);Duplicate(tmp2,0.0f,64);PipeBarrier<PIPE_V>();
            SetVectorMask<float>(64);
            for(uint32_t col=begin/64*64;col<Stride();col+=64){
                Mul<float,false>(prod[col],mat[col],vector[col],64,uint8_t(count),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});
                PipeBarrier<PIPE_V>();
                WholeReduceSum<float,false>(tmp2,prod[col],64,uint8_t(count),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
                Add<float,false>(w,w,tmp2,64,1,BinaryRepeatParams{1,1,1,8,8,8});
            }
            PipeBarrier<PIPE_V>();return;
        }
        uint32_t len=(count+7)/8*8;
        Duplicate(w,0.0f,len);Duplicate(tmp2,0.0f,len);PipeBarrier<PIPE_V>();
        for(uint32_t col=begin/64*64;col<Stride();col+=64){
            uint64_t mask=Stride()-col<64?Stride()-col:64;
            Mul(prod[col],mat[col],vector[col],mask,uint8_t(count),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tmp2,prod[col],mask,uint8_t(count),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
            Add(w,w,tmp2,len);
        }
        VectorToScalar();
    }
    __aicore__ inline ExtendedFloat ReadExtended(uint32_t i){i*=2;return ExtendedFloat(wideBuf.GetValue(i),wideBuf.GetValue(i+1),0);}
    __aicore__ inline void WriteExtended(uint32_t i,ExtendedFloat x){i*=2;wideBuf.SetValue(i,x.h);wideBuf.SetValue(i+1,x.m);}
    __aicore__ inline ExtendedFloat RootExtended(ExtendedFloat x){
        if(x.Zero() || x.Negative())return ExtendedFloat(0);
        ExtendedFloat y(Root(x.h));for(uint32_t i=0;i<1;++i){ExtendedFloat r=x-y*y;y=y+r/(y.Scale(2));}return y;
    }
    __aicore__ inline ExtendedFloat Reflector(uint32_t j){return ExtendedFloat(v.GetValue(j),tmp2.GetValue(j),0);}
    __aicore__ inline ExtendedFloat ProductRow(uint32_t j){return ExtendedFloat(w.GetValue(j),red.GetValue(j),0);}
    __aicore__ inline void ReduceExtended(){
        if(active)for(uint32_t r=0;r<count;++r)for(uint32_t j=0;j<Order();++j)WriteExtended(r*Order()+j,ExtendedFloat(mat.GetValue(r*Stride()+j)));
        for(uint32_t k=0;k<Order();++k){
            if(active && k>=base && k<base+count){
                uint32_t row=k-base;ExtendedFloat diag=ReadExtended(row*Order()+k),off(0),tau(0);
                Duplicate(v,0.0f,Stride());Duplicate(tmp2,0.0f,Stride());Barrier();
                if(k+1<Order()){
                    ExtendedFloat a=ReadExtended(row*Order()+k+1),norm(0);off=a;
                    for(uint32_t j=k+2;j<Order();++j){ExtendedFloat x=ReadExtended(row*Order()+j);norm=norm+x*x;}
                    v.SetValue(k+1,1);
                    if(!norm.Zero()){
                        off=RootExtended(a*a+norm);if(!a.Negative())off=-off;
                        tau=(off-a)/off;ExtendedFloat scale=ExtendedFloat(1)/(a-off);
                        for(uint32_t j=k+2;j<Order();++j){ExtendedFloat x=ReadExtended(row*Order()+j)*scale;v.SetValue(j,x.h);tmp2.SetValue(j,x.m);}
                    }
                    Barrier();DataCopy(hist[k*Stride()],v,Stride());DataCopy(hist[(Order()+k)*Stride()],tmp2,Stride());Barrier();
                }
                param.SetValue(0,diag.h);param.SetValue(1,diag.m);param.SetValue(2,off.h);param.SetValue(3,off.m);param.SetValue(4,tau.h);param.SetValue(5,tau.m);
                Barrier();DataCopy(gparam,param,8);Barrier();
            }
            SyncAll();
            if(active){
                DataCopy(param,gparam,8);if(k+1<Order()){DataCopy(v,hist[k*Stride()],Stride());DataCopy(tmp2,hist[(Order()+k)*Stride()],Stride());}Barrier();
                d.SetValue(k,param.GetValue(0));e.SetValue(k,param.GetValue(2));beta.SetValue(k,param.GetValue(4));
                ExtendedFloat tau(param.GetValue(4),param.GetValue(5),0);
                if(k+1<Order() && !tau.Zero()){
                    for(uint32_t r=0;r<count;++r){ExtendedFloat x(0);
                        if(base+r>k){for(uint32_t j=k+1;j<Order();++j)x=x+ReadExtended(r*Order()+j)*Reflector(j);x=x*tau;}
                        w.SetValue(r,x.h);red.SetValue(r,x.m);
                    }
                    Barrier();DataCopyPad(gw[base],w,DataCopyExtParams{1,count*4,0,0,0});DataCopyPad(gw[Stride()+base],red,DataCopyExtParams{1,count*4,0,0,0});Barrier();
                }
            }
            SyncAll();
            if(active && k+1<Order()){
                ExtendedFloat tau(param.GetValue(4),param.GetValue(5),0);
                if(!tau.Zero()){
                    DataCopy(w,gw,Stride());DataCopy(red,gw[Stride()],Stride());Barrier();
                    ExtendedFloat dot(0);for(uint32_t j=k+1;j<Order();++j)dot=dot+Reflector(j)*ProductRow(j);
                    ExtendedFloat correction=ExtendedFloat(-0.5f)*tau*dot;
                    for(uint32_t j=k+1;j<Order();++j){ExtendedFloat x=ProductRow(j)+correction*Reflector(j);w.SetValue(j,x.h);red.SetValue(j,x.m);}
                    for(uint32_t r=0;r<count;++r)if(base+r>k){
                        ExtendedFloat wi=ProductRow(base+r),vi=Reflector(base+r);
                        for(uint32_t j=k+1;j<Order();++j){ExtendedFloat x=ReadExtended(r*Order()+j)-(Reflector(j)*wi+ProductRow(j)*vi);WriteExtended(r*Order()+j,x);}
                    }
                }
            }
        }
        if(active){Barrier();DataCopy(origD,d,Stride());DataCopy(origE,e,Stride());Barrier();}
    }
    // Every active AIV owns a complete small matrix. Vector lanes are rows;
    // the column fold keeps the original expansion operation order.
    __aicore__ inline void SmallCopy(LocalTensor<float> a,LocalTensor<float> b){
        Copy<float,false>(a,b,Stride(),1,CopyRepeatParams{1,1,8,8});
    }
    __aicore__ inline void SmallAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c){
        Add<float,false>(a,b,c,Stride(),1,BinaryRepeatParams{1,1,1,8,8,8});
    }
    __aicore__ inline void SmallSub(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c){
        Sub<float,false>(a,b,c,Stride(),1,BinaryRepeatParams{1,1,1,8,8,8});
    }
    __aicore__ inline void SmallMuls(LocalTensor<float> a,LocalTensor<float> b,float x){
        Muls<float,false>(a,b,x,Stride(),1,UnaryRepeatParams{1,1,8,8});
    }
    template<bool Left>
    __aicore__ inline void SmallScale(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,ExtendedFloat b){
        auto scratch=smallStorage.template Get<float>()[6*Order()*Stride()];
        auto hi=scratch[4*Stride()],lo=scratch[5*Stride()];
        auto t=scratch[6*Stride()],u=scratch[7*Stride()];
        float z=4097.0f*b.h,bh=z-(z-b.h),bl=b.h-bh;
        SmallMuls(dh,ah,b.h);SmallMuls(t,ah,4097.0f);PipeBarrier<PIPE_V>();
        SmallSub(hi,t,ah);PipeBarrier<PIPE_V>();SmallSub(hi,t,hi);PipeBarrier<PIPE_V>();
        SmallSub(lo,ah,hi);SmallMuls(dm,hi,bh);PipeBarrier<PIPE_V>();
        SmallSub(dm,dm,dh);PipeBarrier<PIPE_V>();
        if constexpr(Left){
            SmallMuls(t,lo,bh);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
            SmallMuls(t,hi,bl);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        }else{
            SmallMuls(t,hi,bl);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
            SmallMuls(t,lo,bh);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        }
        SmallMuls(t,lo,bl);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        if constexpr(Left){SmallMuls(t,am,b.h);SmallMuls(u,ah,b.m);}
        else{SmallMuls(t,ah,b.m);SmallMuls(u,am,b.h);}
        PipeBarrier<PIPE_V>();SmallAdd(t,t,u);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        SmallAdd(t,dh,dm);PipeBarrier<PIPE_V>();SmallSub(u,t,dh);PipeBarrier<PIPE_V>();
        SmallSub(dm,dm,u);SmallCopy(dh,t);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void SmallMul(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c){
        Mul<float,false>(a,b,c,Stride(),1,BinaryRepeatParams{1,1,1,8,8,8});
    }
    __aicore__ inline void SmallProduct(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        auto scratch=smallStorage.template Get<float>()[6*Order()*Stride()];
        auto hi=scratch[4*Stride()],lo=scratch[5*Stride()];
        auto bhi=scratch[6*Stride()],blo=scratch[7*Stride()],t=scratch[8*Stride()];
        SetVectorMask<float>(Stride());
        SmallMul(dh,ah,bh);SmallMuls(hi,ah,4097.0f);SmallMuls(bhi,bh,4097.0f);PipeBarrier<PIPE_V>();
        SmallSub(lo,hi,ah);SmallSub(blo,bhi,bh);PipeBarrier<PIPE_V>();
        SmallSub(hi,hi,lo);SmallSub(bhi,bhi,blo);PipeBarrier<PIPE_V>();
        SmallSub(lo,ah,hi);SmallSub(blo,bh,bhi);PipeBarrier<PIPE_V>();
        SmallMul(dm,hi,bhi);PipeBarrier<PIPE_V>();SmallSub(dm,dm,dh);PipeBarrier<PIPE_V>();
        SmallMul(t,hi,blo);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        SmallMul(t,lo,bhi);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        SmallMul(t,lo,blo);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        SmallMul(t,ah,bm);SmallMul(hi,am,bh);PipeBarrier<PIPE_V>();SmallAdd(t,t,hi);PipeBarrier<PIPE_V>();
        SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();SmallAdd(t,dh,dm);PipeBarrier<PIPE_V>();
        SmallSub(hi,t,dh);PipeBarrier<PIPE_V>();SmallSub(dm,dm,hi);SmallCopy(dh,t);PipeBarrier<PIPE_V>();
    }
    template<bool Minus>
    __aicore__ inline void SmallSum(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        auto scratch=smallStorage.template Get<float>()[6*Order()*Stride()];
        auto t=scratch[4*Stride()],q=scratch[5*Stride()],a=scratch[6*Stride()];
        auto b=scratch[7*Stride()],ee=scratch[8*Stride()];
        if constexpr(Minus)SmallSub(t,ah,bh);else SmallAdd(t,ah,bh);
        PipeBarrier<PIPE_V>();SmallSub(q,t,ah);PipeBarrier<PIPE_V>();
        SmallSub(a,t,q);PipeBarrier<PIPE_V>();SmallSub(a,ah,a);
        if constexpr(Minus)SmallMuls(b,bh,-1.0f);else SmallCopy(b,bh);
        PipeBarrier<PIPE_V>();SmallSub(b,b,q);PipeBarrier<PIPE_V>();SmallAdd(ee,a,b);PipeBarrier<PIPE_V>();
        SmallAdd(ee,ee,am);PipeBarrier<PIPE_V>();
        if constexpr(Minus)SmallSub(ee,ee,bm);else SmallAdd(ee,ee,bm);
        PipeBarrier<PIPE_V>();SmallAdd(dh,t,ee);PipeBarrier<PIPE_V>();
        SmallSub(q,dh,t);PipeBarrier<PIPE_V>();SmallSub(dm,ee,q);PipeBarrier<PIPE_V>();
    }
    // Whole-matrix products amortize the expansion arithmetic over SIMD
    // repeats. Columns remain contiguous; row folds below remain ordered.
    __aicore__ inline void DenseExpand(LocalTensor<float> dst,LocalTensor<float> src){
        auto t=smallStorage.template Get<float>()[6*Order()*Stride()];
        Brcb(t,src,uint8_t((Order()+7)/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(8);
        for(uint32_t row=0;row<Stride();row+=8)
            Copy<float,false>(dst[row],t,8,uint8_t(Order()),CopyRepeatParams{1,1,uint16_t(Stride()/8),1});
        PipeBarrier<PIPE_V>();SetVectorMask<float>(Stride());
    }
    __aicore__ inline void DenseAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,uint8_t cs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Add<float,false>(a,b,c,Stride(),uint8_t(Order()),BinaryRepeatParams{1,1,1,pitch,pitch,cs==255?pitch:cs});
    }
    __aicore__ inline void DenseSub(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,uint8_t cs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Sub<float,false>(a,b,c,Stride(),uint8_t(Order()),BinaryRepeatParams{1,1,1,pitch,pitch,cs==255?pitch:cs});
    }
    __aicore__ inline void DenseMul(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,uint8_t cs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Mul<float,false>(a,b,c,Stride(),uint8_t(Order()),BinaryRepeatParams{1,1,1,pitch,pitch,cs==255?pitch:cs});
    }
    __aicore__ inline void DenseMuls(LocalTensor<float> a,LocalTensor<float> b,float x,uint8_t bs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Muls<float,false>(a,b,x,Stride(),uint8_t(Order()),UnaryRepeatParams{1,1,pitch,bs==255?pitch:bs});
    }
    __aicore__ inline void DenseCopy(LocalTensor<float> a,LocalTensor<float> b){
        uint16_t pitch=uint16_t(Stride()/8);
        Copy<float,false>(a,b,Stride(),uint8_t(Order()),CopyRepeatParams{1,1,pitch,pitch});
    }
    template<bool RowB>
    __aicore__ inline void DenseProduct(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        uint32_t dense=Order()*Stride();auto scratch=smallStorage.template Get<float>()[6*dense];
        auto hi=scratch,lo=scratch[dense],bhi=scratch[2*dense],blo=scratch[3*dense],t=scratch[4*dense];
        constexpr uint8_t bs=RowB?0:255;
        DenseMul(dh,ah,bh,bs);DenseMuls(hi,ah,4097.0f);DenseMuls(bhi,bh,4097.0f,bs);PipeBarrier<PIPE_V>();
        DenseSub(lo,hi,ah);DenseSub(blo,bhi,bh,bs);PipeBarrier<PIPE_V>();
        DenseSub(hi,hi,lo);DenseSub(bhi,bhi,blo);PipeBarrier<PIPE_V>();
        DenseSub(lo,ah,hi);
        // bh's repeated view becomes a full contiguous split, so subtraction
        // uses an explicit row repeat when the original b was a row vector.
        DenseMuls(blo,bh,1.0f,bs);PipeBarrier<PIPE_V>();DenseSub(blo,blo,bhi);PipeBarrier<PIPE_V>();
        DenseMul(dm,hi,bhi);PipeBarrier<PIPE_V>();DenseSub(dm,dm,dh);PipeBarrier<PIPE_V>();
        DenseMul(t,hi,blo);PipeBarrier<PIPE_V>();DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        DenseMul(t,lo,bhi);PipeBarrier<PIPE_V>();DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        DenseMul(t,lo,blo);PipeBarrier<PIPE_V>();DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        DenseMul(t,ah,bm,bs);DenseMul(hi,am,bh,bs);PipeBarrier<PIPE_V>();DenseAdd(t,t,hi);PipeBarrier<PIPE_V>();
        DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();DenseAdd(t,dh,dm);PipeBarrier<PIPE_V>();
        DenseSub(hi,t,dh);PipeBarrier<PIPE_V>();DenseSub(dm,dm,hi);DenseCopy(dh,t);PipeBarrier<PIPE_V>();
    }
    template<bool Minus>
    __aicore__ inline void DenseSum(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        uint32_t dense=Order()*Stride();auto scratch=smallStorage.template Get<float>()[6*dense];
        auto t=scratch,q=scratch[dense],a=scratch[2*dense],b=scratch[3*dense],ee=scratch[4*dense];
        if constexpr(Minus)DenseSub(t,ah,bh);else DenseAdd(t,ah,bh);
        PipeBarrier<PIPE_V>();DenseSub(q,t,ah);PipeBarrier<PIPE_V>();DenseSub(a,t,q);PipeBarrier<PIPE_V>();
        DenseSub(a,ah,a);if constexpr(Minus)DenseMuls(b,bh,-1.0f);else DenseCopy(b,bh);
        PipeBarrier<PIPE_V>();DenseSub(b,b,q);PipeBarrier<PIPE_V>();DenseAdd(ee,a,b);PipeBarrier<PIPE_V>();
        DenseAdd(ee,ee,am);PipeBarrier<PIPE_V>();if constexpr(Minus)DenseSub(ee,ee,bm);else DenseAdd(ee,ee,bm);
        PipeBarrier<PIPE_V>();DenseAdd(dh,t,ee);PipeBarrier<PIPE_V>();DenseSub(q,dh,t);PipeBarrier<PIPE_V>();DenseSub(dm,ee,q);PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadResidentTriangle(LocalTensor<float> matrix,LocalTensor<float> raw){
        if constexpr(StaticResidentStorage){
            const uint32_t elements=Order()*Stride();
            auto indices=smallStorage.template Get<float>()[elements].template ReinterpretCast<int32_t>();
            auto row=matrix.template ReinterpretCast<int32_t>();
            auto column=privateHistory[Order()*Stride()].template ReinterpretCast<int32_t>();
            auto rowIds=piv.template ReinterpretCast<int32_t>();
            auto columnIds=fac.template ReinterpretCast<int32_t>();
            auto broadcast=wideBuf.template ReinterpretCast<int32_t>();
            CreateIndicesWithArena(rowIds,int32_t(0),Stride());
            DataCopy(columnIds,rowIds,Stride());PipeBarrier<PIPE_V>();
            Mins(columnIds,columnIds,int32_t(Order()-1),Stride());PipeBarrier<PIPE_V>();
            if(upper){
                Muls(rowIds,rowIds,int32_t(-1),Stride());Muls(columnIds,columnIds,int32_t(-1),Stride());PipeBarrier<PIPE_V>();
                Adds(rowIds,rowIds,int32_t(Order()-1),Stride());Adds(columnIds,columnIds,int32_t(Order()-1),Stride());PipeBarrier<PIPE_V>();
            }
            Brcb(broadcast,rowIds,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<int32_t>(8);
            for(uint32_t firstColumn=0;firstColumn<Stride();firstColumn+=8)
                Copy<int32_t,false>(row[firstColumn],broadcast,8,uint8_t(Order()),CopyRepeatParams{1,1,uint16_t(Stride()/8),1});
            PipeBarrier<PIPE_V>();SetVectorMask<int32_t>(Stride());
            Copy<int32_t,false>(column,columnIds,Stride(),uint8_t(Order()),CopyRepeatParams{1,1,uint16_t(Stride()/8),0});
            PipeBarrier<PIPE_V>();ResetMask();
            Min(indices,row,column,elements);Max(row,row,column,elements);PipeBarrier<PIPE_V>();
            if(upper){Muls(indices,indices,int32_t(Order()),elements);PipeBarrier<PIPE_V>();Add(indices,indices,row,elements);}
            else{Muls(row,row,int32_t(Order()),elements);PipeBarrier<PIPE_V>();Add(indices,row,indices,elements);}
            PipeBarrier<PIPE_V>();Muls(indices,indices,int32_t(4),elements);PipeBarrier<PIPE_V>();
            Gather(matrix,raw,indices.template ReinterpretCast<uint32_t>(),0,elements);PipeBarrier<PIPE_V>();
            if(Order()<Stride()){
                const uint64_t paddingMask=((uint64_t(1)<<Stride())-1)&~((uint64_t(1)<<Order())-1);
                SetMaskNorm();SetVectorMask<float>(uint64_t(0),paddingMask);
                Duplicate<float,false>(matrix,0.0f,64,uint8_t(Order()),1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();ResetMask();
            }
            Barrier();return;
        }

        if constexpr(IndependentMedium){
            const uint32_t elements=Order()*Stride();
            auto indices=smallStorage.template Get<float>()[elements].template ReinterpretCast<int32_t>();
            auto row=matrix.template ReinterpretCast<int32_t>(),column=privateHistory[Order()*Stride()].template ReinterpretCast<int32_t>();
            CreateIndicesWithArena(indices,int32_t(0),elements);PipeBarrier<PIPE_V>();
            ShiftRight(row.template ReinterpretCast<uint32_t>(),indices.template ReinterpretCast<uint32_t>(),uint32_t(6),elements);
            ShiftLeft(column.template ReinterpretCast<uint32_t>(),indices.template ReinterpretCast<uint32_t>(),uint32_t(26),elements);PipeBarrier<PIPE_V>();
            ShiftRight(column.template ReinterpretCast<uint32_t>(),column.template ReinterpretCast<uint32_t>(),uint32_t(26),elements);PipeBarrier<PIPE_V>();
            Mins(column,column,int32_t(Order()-1),elements);PipeBarrier<PIPE_V>();
            if(upper){Muls(row,row,int32_t(-1),elements);Muls(column,column,int32_t(-1),elements);PipeBarrier<PIPE_V>();Adds(row,row,int32_t(Order()-1),elements);Adds(column,column,int32_t(Order()-1),elements);PipeBarrier<PIPE_V>();}
            Min(indices,row,column,elements);Max(row,row,column,elements);PipeBarrier<PIPE_V>();
            if(upper){Muls(indices,indices,int32_t(Order()),elements);PipeBarrier<PIPE_V>();Add(indices,indices,row,elements);}
            else{Muls(row,row,int32_t(Order()),elements);PipeBarrier<PIPE_V>();Add(indices,row,indices,elements);}
            PipeBarrier<PIPE_V>();Muls(indices,indices,int32_t(4),elements);PipeBarrier<PIPE_V>();
            Gather(matrix,raw,indices.template ReinterpretCast<uint32_t>(),0,elements);PipeBarrier<PIPE_V>();
            if(Order()<64){
                SetMaskNorm();SetVectorMask<float>(uint64_t(0),~uint64_t(0)<<Order());
                Duplicate<float,false>(matrix,0.0f,64,uint8_t(Order()),1,8);PipeBarrier<PIPE_V>();ResetMask();
            }
            Barrier();return;
        }

        auto indices=piv.template ReinterpretCast<int32_t>();
        CreateIndicesWithArena(indices,int32_t(0),Stride());PipeBarrier<PIPE_V>();
        Cast(param,indices,RoundMode::CAST_NONE,Stride());PipeBarrier<PIPE_V>();
        if(upper){Muls(param,param,-1.0f,Stride());PipeBarrier<PIPE_V>();Adds(param,param,float(int32_t(Order()-1)),Stride());PipeBarrier<PIPE_V>();}
        for(uint32_t i=0;i<Order();++i){
            Duplicate(v,float(int32_t(upper?Order()-1-i:i)),Stride());PipeBarrier<PIPE_V>();
            Min(w,v,param,Stride());Max(tmp,v,param,Stride());PipeBarrier<PIPE_V>();
            Muls(tmp2,upper?w:tmp,float(int32_t(Order())),Stride());PipeBarrier<PIPE_V>();
            Add(tmp2,tmp2,upper?tmp:w,Stride());PipeBarrier<PIPE_V>();
            Muls(tmp2,tmp2,4.0f,Stride());PipeBarrier<PIPE_V>();
            Cast(indices,tmp2,RoundMode::CAST_RINT,Stride());PipeBarrier<PIPE_V>();
            // Only Order() valid indices are read. The caller zeroed row padding.
            Gather(matrix[i*Stride()],raw,indices.template ReinterpretCast<uint32_t>(),0,Order());PipeBarrier<PIPE_V>();
        }
        Barrier();
    }

    __aicore__ __attribute__((noinline)) void ReduceReplicatedSmall(uint32_t batchIndex,bool synchronize=true){
        if(active){
            auto mh=smallStorage.template Get<float>(),ml=mh[Order()*Stride()],scratch=mh[2*Order()*Stride()];
            // The unused matrix region is large enough for the compact raw
            // input, and is dead before any vector scratch is reused.
            DataCopyPad(scratch,input[uint64_t(batchIndex)*Order()*Order()],DataCopyExtParams{1,Order()*Order()*4,0,0,0},
                DataCopyPadExtParams<float>{false,0,0,0});Barrier();
            Duplicate(mh,0.0f,2*Order()*Stride());Barrier();
            LoadResidentTriangle(mh,scratch);
            // Power-of-two scaling preserves input significands and keeps
            // products in the compensated reduction away from overflow/FTZ.
            if constexpr(IsolatedSmall){
                if(matrixScale!=1.0f){Muls(mh,mh,1.0f/matrixScale,Order()*Stride());Barrier();}
            }
            uint32_t dense=Order()*Stride();auto ph=mh[2*dense],pm=mh[3*dense],qh=mh[4*dense],qm=mh[5*dense];
            auto vh=mh[11*dense],vm=mh[12*dense],wh=mh[13*dense],wm=mh[14*dense];
            auto sh=mh[6*dense],sm=sh[Stride()];
            for(uint32_t k=0;k<Order();++k){
                Barrier();ExtendedFloat diag(mh.GetValue(k*Stride()+k),ml.GetValue(k*Stride()+k),0);
                ExtendedFloat off(0),tau(0);
                Duplicate(v,0.0f,Stride());Duplicate(tmp2,0.0f,Stride());Barrier();
                if(k+1<Order()){
                    ExtendedFloat a(mh.GetValue((k+1)*Stride()+k),ml.GetValue((k+1)*Stride()+k),0),norm(0);off=a;
                    for(uint32_t j=k+2;j<Order();++j){
                        ExtendedFloat x(mh.GetValue(j*Stride()+k),ml.GetValue(j*Stride()+k),0);norm=norm+x*x;
                    }
                    v.SetValue(k+1,1);
                    if(!norm.Zero()){
                        off=RootExtended(a*a+norm);if(!a.Negative())off=-off;
                        tau=(off-a)/off;ExtendedFloat scale=ExtendedFloat(1)/(a-off);
                        for(uint32_t j=k+2;j<Order();++j){
                            ExtendedFloat x=ExtendedFloat(mh.GetValue(j*Stride()+k),ml.GetValue(j*Stride()+k),0)*scale;
                            v.SetValue(j,x.h);tmp2.SetValue(j,x.m);
                        }
                    }
                    Barrier();
                    if constexpr(PrivateSmall){DataCopy(privateHistory[k*Stride()],v,Stride());DataCopy(privateHistory[(Order()+k)*Stride()],tmp2,Stride());Barrier();}
                    else if(lane+1==team){DataCopy(hist[k*Stride()],v,Stride());DataCopy(hist[(Order()+k)*Stride()],tmp2,Stride());Barrier();}
                }
                d.SetValue(k,diag.h);e.SetValue(k,off.h);beta.SetValue(k,tau.h);
                if(k+1<Order() && !tau.Zero()){
                    Duplicate(w,0.0f,Stride());Duplicate(red,0.0f,Stride());PipeBarrier<PIPE_V>();
                    DenseExpand(vh,v);DenseExpand(vm,tmp2);DenseProduct<false>(ph,pm,mh,ml,vh,vm);
                    for(uint32_t j=k+1;j<Order();++j)SmallSum<false>(w,red,w,red,ph[j*Stride()],pm[j*Stride()]);
                    SmallScale<false>(sh,sm,w,red,tau);SmallCopy(w,sh);SmallCopy(red,sm);Barrier();
                    for(uint32_t j=0;j<=k;++j){w.SetValue(j,0);red.SetValue(j,0);}
                    ExtendedFloat dot(0);for(uint32_t j=k+1;j<Order();++j)dot=dot+Reflector(j)*ProductRow(j);
                    ExtendedFloat correction=ExtendedFloat(-0.5f)*tau*dot;
                    for(uint32_t j=k+1;j<Order();++j){ExtendedFloat x=ProductRow(j)+correction*Reflector(j);w.SetValue(j,x.h);red.SetValue(j,x.m);}
                    Barrier();DenseExpand(wh,w);DenseExpand(wm,red);
                    DenseProduct<true>(ph,pm,vh,vm,w,red);DenseProduct<true>(qh,qm,wh,wm,v,tmp2);
                    DenseSum<false>(ph,pm,ph,pm,qh,qm);DenseSum<true>(mh,ml,mh,ml,ph,pm);
                    Barrier();ResetMask();
                }
            }
            DataCopy(origD,d,Stride());DataCopy(origE,e,Stride());Barrier();
        }
        // The full calculation is invocation-local. Publish the completed
        // reflectors before any AIV begins the original backtransform.
        if constexpr(!PrivateSmall){if(synchronize)SyncAll();}
    }

    __aicore__ inline float ResidentNorm(LocalTensor<float> vector){
        // Reduction and square root stay on the vector pipeline; only the
        // completed norm crosses to scalar reflector construction.
        Mul(tmp,vector,vector,Stride());PipeBarrier<PIPE_V>();
        WholeReduceSum(red,tmp,Stride(),1,1,1,8);PipeBarrier<PIPE_V>();
        Sqrt(red,red,1);VectorToScalar();return red.GetValue(0);
    }

    __aicore__ inline float SmallNativeRoot(float value){
        if(value<=0.0f)return 0.0f;
        // param/red are dead between scalar operations in the private
        // FP32 reduction and QL recurrence, including rotation replay.
        param.SetValue(0,value);
        event_t ready=ImmediateEvent<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(ready);WaitFlag<HardEvent::S_V>(ready);
        Sqrt(red,param,1);VectorToScalar();return red.GetValue(0);
    }

    // Complete invocation-local FP32 Householder reduction for matrices
    // whose full small matrix fits the already reserved private UB arena.
    __aicore__ __attribute__((noinline)) void ReduceReplicatedFloatSmall(uint32_t b){
        if(active){
            uint32_t dense=Order()*Stride(),padded=(Order()+7)/8*8;
            auto a=smallStorage.template Get<float>(),product=a[dense],raw=a[2*dense];
            DataCopyPad(raw,input[uint64_t(b)*Order()*Order()],DataCopyExtParams{1,Order()*Order()*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});Barrier();
            // Both dense gather layouts overwrite every coordinate and
            // explicitly clear row padding. The scalar loader needs its zeros.
            if constexpr(!StaticResidentStorage && !IndependentMedium){Duplicate(a,0.0f,dense);Barrier();}
            LoadResidentTriangle(a,raw);
            Barrier();
            // Bound the small FP32 path's absolute backward error and avoid
            // squared-norm underflow/overflow. More demanding magnitudes
            // retain the existing complete two-float reduction.
            AscendC::Abs(product,a,dense);Duplicate(w,0.0f,Stride());PipeBarrier<PIPE_V>();
            WholeReduceSum(w,product,Stride(),uint8_t(Order()),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
            if constexpr(DirectUb)WholeReduceMax(red,w,Stride(),1,1,1,8);else ReduceMax(red,w,tmp,Stride(),false);VectorToScalar();float rowBound=red.GetValue(0);
            smallRowBound=rowBound;
            if constexpr(IndependentMedium){
                matrixScale=1.0f;
                if(rowBound>64.0f || (rowBound>0.0f && rowBound<1e-10f)){
                    // Binary scaling preserves every input significand; only
                    // the same magnitude guard selects when scaling is needed.
                    union ScaleBits {float value;uint32_t bits;};ScaleBits scale;
                    scale.value=rowBound;scale.bits&=0x7f800000U;
                    if(scale.bits==0)scale.bits=0x00800000U;
                    matrixScale=scale.value;
                }
                if(matrixScale!=1.0f){Muls(a,a,1.0f/matrixScale,dense);Barrier();}
                smallQLNeedsCheck=false;
                smallFloatQL=false;
            }else if constexpr(IsolatedSmall){
                // Private copies need no cross-core acceptance protocol. Wide
                // magnitudes use the complete two-float reduction directly.
                smallQLNeedsCheck=false;
                // Retain the measured small-norm inverse path; extending full QL
                // below this norm threshold was numerically valid but slower.
                smallFloatQL=rowBound>16.0f && rowBound<=64.0f;
                if(rowBound>64.0f || (rowBound>0.0f && rowBound<1e-10f)){
                    union ScaleBits {float value;uint32_t bits;};
                    ScaleBits scale;scale.value=rowBound;scale.bits&=0x7f800000U;
                    if(scale.bits==0)scale.bits=0x00800000U;
                    matrixScale=scale.value;smallRowBound=rowBound/matrixScale;
                    useExtendedEigenvectors=true;
                    ReduceReplicatedSmall(b);return;
                }
            }else{
                smallQLNeedsCheck=rowBound>64.0f && rowBound<=281474976710656.0f && rows<=2;
                smallFloatQL=(rowBound>16.0f && rowBound<=64.0f) || smallQLNeedsCheck;
                if((rowBound>64.0f && !smallQLNeedsCheck) || (rowBound>0.0f && rowBound<1e-10f)){
                    ReduceReplicatedSmall(b);return;
                }
            }

            auto cw=wideBuf,cv=wideBuf[padded*8];
            if constexpr(IndependentMedium){Duplicate(e,0.0f,Stride());Duplicate(beta,0.0f,Stride());PipeBarrier<PIPE_V>();}
            for(uint32_t k=0;k<Order();++k){
                if constexpr(IndependentMedium){
                    // The preceding matrix update already retires V->V.
                    // Record the diagonal without crossing to scalar.
                    SetMaskNorm();SetVectorMask<float>(uint64_t(0),uint64_t(1)<<k);
                    Muls<float,false>(d,a[k*Stride()],1.0f,64,1,UnaryRepeatParams{1,1,8,8});
                    SetVectorMask<float>(64);
                }else{d.SetValue(k,a.GetValue(k*Stride()+k));e.SetValue(k,0.0f);beta.SetValue(k,0.0f);}
                if(k+2<Order()){
                    if constexpr(PrivateSmall){
                        Copy<float>(v,a[k*Stride()],Stride(),1,CopyRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    }else{DataCopy(v,a[k*Stride()],Stride());Barrier();}
                    Duplicate(v,0.0f,k+1);PipeBarrier<PIPE_V>();
                    float norm=ResidentNorm(v),x0=v.GetValue(k+1);
                    e.SetValue(k,x0>=0?-norm:norm);
                    const float firstReflectorCoordinate=norm>0?x0/norm+(x0>=0?1.0f:-1.0f):0.0f;
                    if(norm>0){
                        if constexpr(IndependentMedium){
                            v.SetValue(k+1,firstReflectorCoordinate);ScalarToVector();
                            SetMaskNorm();SetVectorMask<float>(uint64_t(0),~(uint64_t(1)<<(k+1)));
                            Muls<float,false>(v,v,1.0f/norm,64,1,UnaryRepeatParams{1,1,8,8});
                            PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
                        }else{
                            Muls(v,v,1.0f/norm,Stride());VectorToScalar();v.SetValue(k+1,firstReflectorCoordinate);
                            if constexpr(PrivateSmall)ScalarToVector();else Barrier();
                        }
                    }
                    if constexpr(PrivateSmall)Copy<float>(privateHistory[k*Stride()],v,Stride(),1,CopyRepeatParams{1,1,8,8});
                    else if(lane+1==team)DataCopy(hist[k*Stride()],v,Stride());
                    if(norm>0){
                        // v=x/||x||+sign(x0)e1, hence v^Tv=2|v1|.
                        // The normalized reflector avoids a second SIMD reduction and
                        // its vector-to-scalar dependency for each reflector.
                        float tau;
                        if constexpr(PrivateSmall || FixedOrder==16 || IndependentMedium)tau=1.0f/Abs(firstReflectorCoordinate);
                        else tau=2.0f/Dot(v,v);
                        beta.SetValue(k,tau);
                        const uint32_t firstActiveRow=(IndependentMedium || StaticResidentStorage)?(k+1)/8*8:0,activeRows=Order()-firstActiveRow;
                        Mul(product[firstActiveRow*Stride()],a[firstActiveRow*Stride()],v,Stride(),uint8_t(activeRows),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});
                        if constexpr(!StaticResidentStorage)PipeBarrier<PIPE_V>();
                        Duplicate(w,0.0f,Stride());PipeBarrier<PIPE_V>();
                        WholeReduceSum(w[firstActiveRow],product[firstActiveRow*Stride()],Stride(),uint8_t(activeRows),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
                        Duplicate(w,0.0f,k+1);PipeBarrier<PIPE_V>();Muls(w,w,tau,Stride());PipeBarrier<PIPE_V>();
                        // The rank-two correction is shared by every lane. Broadcast
                        // its vector reduction without a scalar dependency.
                        Mul(tmp2,v,w,Stride());Duplicate(red,0.0f,8);PipeBarrier<PIPE_V>();
                        WholeReduceSum(red,tmp2,Stride(),1,1,1,8);PipeBarrier<PIPE_V>();
                        Muls(red,red,-0.5f*tau,8);PipeBarrier<PIPE_V>();
                        Brcb(cv,red,1,{1,8});PipeBarrier<PIPE_V>();
                        MulAddDst(w,v,cv,Stride(),1,BinaryRepeatParams{1,1,0,0,0,0});PipeBarrier<PIPE_V>();
                        Brcb(cw,w,uint8_t(padded/8),{1,8});Brcb(cv,v,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
                        // Closed rows have zero reflector coefficients and
                        // cannot contribute to subsequent Householder steps.
                        // The dot-product destination w needs an aligned
                        // first row. The matrix update has a full aligned row
                        // pitch and can start at the exact active row instead.
                        const uint32_t firstUpdatedRow=(IndependentMedium || StaticResidentStorage)?k+1:firstActiveRow;
                        const uint32_t updatedRows=Order()-firstUpdatedRow;
                        Mul(product[firstUpdatedRow*Stride()],v,cw[8*firstUpdatedRow],Stride(),uint8_t(updatedRows),BinaryRepeatParams{1,1,0,uint8_t(Stride()/8),0,1});PipeBarrier<PIPE_V>();
                        MulAddDst(product[firstUpdatedRow*Stride()],w,cv[8*firstUpdatedRow],Stride(),uint8_t(updatedRows),BinaryRepeatParams{1,1,0,uint8_t(Stride()/8),0,1});PipeBarrier<PIPE_V>();
                        Sub(a[firstUpdatedRow*Stride()],a[firstUpdatedRow*Stride()],product[firstUpdatedRow*Stride()],updatedRows*Stride());
                        if constexpr(!StaticResidentStorage)PipeBarrier<PIPE_V>();
                    }
                }else if(k+1<Order()){
                    if constexpr(IndependentMedium)VectorToScalar();
                    e.SetValue(k,a.GetValue(k*Stride()+k+1));
                }
                // Medium diagonals remain vector records. The next norm
                // supplies its own V->S; the final scalar edge read above
                // retains its explicit dependency.
                if constexpr(!IndependentMedium){if constexpr(PrivateSmall)VectorToScalar();else Barrier();}
            }
            if constexpr(PrivateSmall){
                ScalarToVector();
                if constexpr(IndependentMedium)PipeBarrier<PIPE_V>();
                Copy<float>(origD,d,Stride(),1,CopyRepeatParams{1,1,8,8});
                Copy<float>(origE,e,Stride(),1,CopyRepeatParams{1,1,8,8});VectorToScalar();
            }else{DataCopy(origD,d,Stride());DataCopy(origE,e,Stride());Barrier();}
            if(smallFloatQL){
                count=IndependentMedium?OwnedRowCount():(lane<Order()?(Order()-1-lane)/team+1:0);
                smallQLSolved=SolvePrivateSmallQL();
                if(smallQLSolved && smallQLNeedsCheck){
                    for(uint32_t i=0;i<Order();++i){
                        float value=d.GetValue(i);
                        float floor=1e-12f*float(int32_t(Order()*Order()))*smallRowBound+1.1920928955078125e-7f*Abs(value);
                        if(floor>0.25f*(1e-4f+1e-4f*Abs(value)))smallQLSolved=false;
                    }
                }
                if(!smallQLSolved)ReduceReplicatedSmall(b,false);
                if(smallQLSolved && smallQLNeedsCheck && lane==0)PublishSmallMode(true);
            }
        }
        if constexpr(!PrivateSmall)SyncAll();
    }

    __aicore__ inline float TailDot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> product,LocalTensor<float> result,uint32_t length){
        Mul(product,a,b,length);PipeBarrier<PIPE_V>();
#if __NPU_ARCH__ == 2201
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(length));
        vcadd((__ubuf__ float*)result.GetPhyAddr(),(__ubuf__ float*)product.GetPhyAddr(),1,1,1,8,1);
        VectorToScalar();float value=GetReduceRepeatSumSpr<float>();SetMaskNorm();ResetMask();return value;
#else
        ReduceSumWithArena(result,product,product[64],length);VectorToScalar();return result.GetValue(0);
#endif
    }
    __aicore__ __attribute__((noinline)) void PrivateTail(uint32_t first){
        uint32_t size=Order()-first,width=Stride()-first,pitch=size>64?128:64;
        uint32_t batchRows=4*Stride()+1344<=privateCapacity?64:32;
        LocalTensor<float> td=wideBuf,te=wideBuf[Stride()],tb=wideBuf[2*Stride()];
        LocalTensor<float> fullV=wideBuf[3*Stride()],tv=fullV[first];
        LocalTensor<float> tw=wideBuf[4*Stride()],pivot=wideBuf[4*Stride()+128];
        LocalTensor<float> result=wideBuf[4*Stride()+256];
        LocalTensor<float> cw=wideBuf[4*Stride()+320],cv=cw[batchRows*8];
        LocalTensor<float> product=storage.template Get<float>()[width*pitch];
        // Preserve all completed metadata and the separately updated pivot
        // before recycling the resident matrix and inverse work storage.
        DataCopy(td,d,Stride());DataCopy(te,e,Stride());DataCopy(tb,beta,Stride());
        DataCopy(pivot,pack[first],width);Duplicate(fullV,0.0f,Stride());Barrier();
        LocalTensor<float> tail=storage.template Get<float>();
        Duplicate(tail,0.0f,width*pitch);Barrier();
        auto stage=hist[(CubeMode==1 || CubeMode==2)?(Order()+5)*Stride():first*Stride()];
        if(width==pitch)DataCopy(tail,stage,size*pitch);
        else DataCopyPad(tail,stage,DataCopyExtParams{uint16_t(size),width*4,0,(pitch-width)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
        Barrier();
        for(uint32_t k=0;k<size;++k){
            uint32_t global=first+k,localFirst=(k+1)/64*64;
            td.SetValue(global,pivot.GetValue(k));te.SetValue(global,0);tb.SetValue(global,0);
            float tau=0;
            if(k+2<size){
                DataCopy(tv,pivot,width);Barrier();Duplicate(tv,0.0f,k+1);Barrier();
                float norm=Root(TailDot(tv[localFirst],tv[localFirst],product,result,width-localFirst));float x0=tv.GetValue(k+1);
                te.SetValue(global,x0>=0?-norm:norm);
                if(norm>0){Muls(tv,tv,1.0f/norm,width);Barrier();tv.SetValue(k+1,x0/norm+(x0>=0?1.0f:-1.0f));Barrier();}
                DataCopy(hist[global*Stride()],fullV,Stride());
                if(norm>0)tau=1.0f/Abs(tv.GetValue(k+1));
                tb.SetValue(global,tau);
            }else if(k+1<size)te.SetValue(global,pivot.GetValue(k+1));
            if(k+1<size){DataCopy(pivot,tail[(k+1)*pitch],width);Barrier();}
            if(k+2<size && tau!=0){
                Duplicate(tw,0.0f,128);PipeBarrier<PIPE_V>();
                for(uint32_t rb=(k+1)/8*8;rb<size;rb+=batchRows){
                    uint8_t live=uint8_t(size-rb<batchRows?size-rb:batchRows);
                    for(uint32_t col=localFirst;col<width;col+=64){
                        uint64_t mask=width-col<64?width-col:64;
                        Mul(product,tail[rb*pitch+col],tv[col],mask,live,BinaryRepeatParams{1,1,1,8,uint8_t(pitch/8),0});PipeBarrier<PIPE_V>();
                        WholeReduceSum(result,product,mask,live,1,1,8);PipeBarrier<PIPE_V>();
                        Add(tw[rb],tw[rb],result,uint32_t(live));PipeBarrier<PIPE_V>();
                    }
                }
                Duplicate(tw,0.0f,k+1);PipeBarrier<PIPE_V>();Muls(tw,tw,tau,width);Barrier();
                float correction=-0.5f*tau*TailDot(tv[localFirst],tw[localFirst],product,result,width-localFirst);
                Axpy(tw,tv,correction,width);Barrier();
                Muls(cw,tv,tw.GetValue(k+1),width);PipeBarrier<PIPE_V>();
                Axpy(cw,tw,tv.GetValue(k+1),width);PipeBarrier<PIPE_V>();
                Sub(pivot,pivot,cw,width);PipeBarrier<PIPE_V>();
                for(uint32_t rb=(k+1)/8*8;rb<size;rb+=batchRows){
                    uint8_t live=uint8_t(size-rb<batchRows?size-rb:batchRows);
                    Brcb(cw,tw[rb],uint8_t((live+7)/8),{1,8});Brcb(cv,tv[rb],uint8_t((live+7)/8),{1,8});PipeBarrier<PIPE_V>();
                    for(uint32_t col=(k+1)/8*8;col<width;col+=64){
                        uint64_t mask=width-col<64?width-col:64;
                        Mul(product,tv[col],cw,mask,live,BinaryRepeatParams{1,1,0,8,0,1});PipeBarrier<PIPE_V>();
                        MulAddDst(product,tw[col],cv,mask,live,BinaryRepeatParams{1,1,0,8,0,1});PipeBarrier<PIPE_V>();
                        Sub(tail[rb*pitch+col],tail[rb*pitch+col],product,mask,live,BinaryRepeatParams{1,1,1,uint8_t(pitch/8),uint8_t(pitch/8),8});PipeBarrier<PIPE_V>();
                    }
                }
            }
            Barrier();
        }
        // These destinations may overlap the now-dead private matrix.
        DataCopy(d,td,Stride());DataCopy(e,te,Stride());DataCopy(beta,tb,Stride());Barrier();
    }
    __aicore__ inline void GatherTail(uint32_t first){
        uint32_t width=Stride()-first;
        if(active){
            auto stage=hist[(CubeMode==1 || CubeMode==2)?(Order()+5)*Stride():first*Stride()];
            for(uint32_t r=0;r<count;++r)if(base+r>=first){
                ReadPackedRow(tmp,r);Barrier();
                DataCopy(stage[(base+r-first)*width],tmp[first],width);Barrier();
            }
        }
        SyncAll();
        if(active && lane+1==team)PrivateTail(first);
        packedMatrix=false;
    }
    __aicore__ __attribute__((noinline)) void Reduce(){
        // Replicate the next pivot row. Two workspace rows prevent a fast core
        // from overwriting a row before another core has consumed it.
        auto pivotRows=gw[Stride()];
        auto rowOffsets=piv.template ReinterpretCast<uint32_t>();
        if(active){Duplicate(d,0.0f,Stride());Duplicate(e,0.0f,Stride());Duplicate(beta,0.0f,Stride());Barrier();for(uint32_t r=0;r<(count+7)/8*8;++r)rowOffsets.SetValue(r,(base+r<Order()?base+r:Order()-1)*4);Barrier();}
        if(active && lane==0){
            Duplicate(w,0.0f,Stride());Barrier();DataCopy(gw,w,Stride());DataCopy(gw[2*Stride()],w,Stride());
            if(packedMatrix){ReadPackedRow(tmp,0);Barrier();DataCopy(gparam,tmp,Stride());}
            else DataCopy(gparam,mat,Stride());Barrier();
        }
        SyncAll();
        if(active){DataCopy(pack,gparam,Stride());Barrier();}
        uint32_t privateFirst=(Order()-1)/64*64;
        for(uint32_t k=0;k<Order();++k){
            if(k==privateFirst){GatherTail(privateFirst);break;}
            // A closed row is no longer used by any following reflector.
            // Its w slot has been explicitly zeroed on the last live step.
            bool working=active && k<base+count;
            if(working){
                d.SetValue(k,pack.GetValue(k));e.SetValue(k,0);beta.SetValue(k,0);
                if(k+2<Order()){
                    DataCopy(v,pack,Stride());Barrier();Duplicate(v,0.0f,k+1);PipeBarrier<PIPE_V>();
                    float norm=Root(DotTail(v,v,k+1));float x0=v.GetValue(k+1),alpha=x0>=0?-norm:norm;
                    e.SetValue(k,alpha);
                    if(norm>0){Muls(v,v,1.0f/norm,Stride());VectorToScalar();v.SetValue(k+1,x0/norm+(x0>=0?1.0f:-1.0f));Barrier();}
                    if(lane+1==team){Barrier();DataCopy(hist[k*Stride()],v,Stride());}
                    if(norm>0)beta.SetValue(k,1.0f/Abs(v.GetValue(k+1)));
                }else if(k+1<Order())e.SetValue(k,pack.GetValue(k+1));
                // Publish the unmodified next row alongside the matrix-vector product.
                if(k+1<Order() && k+1>=base && k+1<base+count){
                    if(packedMatrix){ReadPackedRow(tmp,k+1-base);Barrier();DataCopy(pivotRows[(k%2)*2*Stride()],tmp,Stride());}
                    else DataCopy(pivotRows[(k%2)*2*Stride()],mat[(k+1-base)*Stride()],Stride());
                }
                if(k+2<Order()){
                    if(beta.GetValue(k)!=0){
                        DotRows(v,k+1);Muls(w,w,beta.GetValue(k),(count+7)/8*8);Barrier();
                        if(base<=k){uint32_t z=k+1-base;z=z<count?z:count;Duplicate(w,0.0f,z);Barrier();}
                    }else{Duplicate(w,0.0f,Stride());Barrier();}
                    DataCopyPad(gw[(k%2)*2*Stride()+base],w,DataCopyExtParams{1,count*4,0,0,0});
                }
            }
            // SyncAll begins with PIPE_ALL before publishing its arrival.
            SyncAll();
            if(working){
                // One transfer fetches the matrix-vector result and next pivot row.
                if(k+1<Order())DataCopy(w,gw[(k%2)*2*Stride()],2*Stride());
                Barrier();
                if(k+2<Order() && beta.GetValue(k)!=0){
                    float correction=-0.5f*beta.GetValue(k)*DotTail(v,w,k+1);
                    Axpy(w,v,correction,Stride());VectorToScalar();
                    if(packedMatrix || (Stride()%64==0 && count*8<=4*Stride())){
                        SetVectorMask<float>(64);uint32_t first=(k+1)/64*64;
                        // Apply exactly the same arithmetic to the replicated row.
                        Muls<float,false>(tmp[first],v[first],w.GetValue(k+1),64,uint8_t((Stride()-first)/64),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                        Axpy<float,float,false>(tmp[first],w[first],v.GetValue(k+1),64,uint8_t((Stride()-first)/64),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                        Sub<float,false>(pack[first],pack[first],tmp[first],64,uint8_t((Stride()-first)/64),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                        if(packedMatrix)UpdatePacked(k);
                        else{
                        uint32_t firstRow=base>k?0:k+1-base;firstRow=firstRow<count?firstRow:count;
                        if(firstRow<count){
                            if(directCoefficients){
                                Brcb(pack[Stride()],w[base],uint8_t(count/8),{1,8});Brcb(pack[5*Stride()],v[base],uint8_t(count/8),{1,8});PipeBarrier<PIPE_V>();
                            }else{
                            Gather(tmp2,w,rowOffsets,0,count);PipeBarrier<PIPE_V>();Brcb(pack[Stride()],tmp2,uint8_t((count+7)/8),{1,8});PipeBarrier<PIPE_V>();
                            Gather(tmp2,v,rowOffsets,0,count);PipeBarrier<PIPE_V>();Brcb(pack[5*Stride()],tmp2,uint8_t((count+7)/8),{1,8});PipeBarrier<PIPE_V>();
                            }
                            SetVectorMask<float>(64);
                            BinaryRepeatParams br{1,1,0,uint8_t(Stride()/8),0,1};
                            for(uint32_t col=first;col<Stride();col+=64){
                                uint32_t off=firstRow*Stride()+col;
                                Mul<float,false>(prod[off],v[col],pack[Stride()+8*firstRow],64,uint8_t(count-firstRow),br);
                            }
                            PipeBarrier<PIPE_V>();
                            for(uint32_t col=first;col<Stride();col+=64){
                                uint32_t off=firstRow*Stride()+col;
                                MulAddDst<float,float,false>(prod[off],w[col],pack[5*Stride()+8*firstRow],64,uint8_t(count-firstRow),br);
                            }
                            PipeBarrier<PIPE_V>();
                            for(uint32_t col=first;col<Stride();col+=64){
                                uint32_t off=firstRow*Stride()+col;
                                Sub<float,false>(mat[off],mat[off],prod[off],64,uint8_t(count-firstRow),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),uint8_t(Stride()/8)});
                            }
                            PipeBarrier<PIPE_V>();
                        }
                        }
                    }else{
                        Muls(tmp,v,w.GetValue(k+1),Stride());PipeBarrier<PIPE_V>();Axpy(tmp,w,v.GetValue(k+1),Stride());PipeBarrier<PIPE_V>();Sub(pack,pack,tmp,Stride());PipeBarrier<PIPE_V>();
                        for(uint32_t r=0;r<count;++r)if(base+r>k){auto a=mat[r*Stride()];Muls(tmp,v,w.GetValue(base+r),Stride());PipeBarrier<PIPE_V>();Axpy(tmp,w,v.GetValue(base+r),Stride());PipeBarrier<PIPE_V>();Sub(a,a,tmp,Stride());PipeBarrier<PIPE_V>();}
                    }
                    Barrier();
                }
            }
        }
        // The last row owner remains live throughout the reduction. The
        // last two high reflector rows are unused by the backtransform, so
        // they hold the completed diagonal/off-diagonal for the other cores.
        if(active && lane+1==team){
            DataCopy(hist[(Order()-2)*Stride()],d,Stride());DataCopy(hist[(Order()-1)*Stride()],e,Stride());DataCopy(gparam,beta,Stride());Barrier();
        }
        SyncAll();
        if(active){
            DataCopy(origD,hist[(Order()-2)*Stride()],Stride());DataCopy(origE,hist[(Order()-1)*Stride()],Stride());DataCopy(beta,gparam,Stride());Barrier();
            DataCopy(d,origD,Stride());DataCopy(e,origE,Stride());Barrier();
        }
    }
    // Independent Sturm recurrences occupy vector lanes; the recurrence along
    // the tridiagonal remains sequential, while different eigenvalues run together.
    // Evaluate a complete binary decision tree in otherwise idle SIMD lanes.
    // Every midpoint uses the original binary arithmetic and every rank stops
    // at the same tolerance; several sequential bisections finish per sweep.
    __aicore__ inline float BuildSharedSturmGrid(float lower,float upperBound,float tolerance){
        hasClusterGrid=true;
        uint32_t limit=4*Stride()<128*team?4*Stride():128*team;
        uint32_t branches=1;while(2*branches<=limit)branches*=2;
        uint32_t queries=lane<branches-1?(branches-2-lane)/team+1:0;
        uint32_t lanes=queries>64?64:(queries+7)/8*8,repeats=(queries+63)/64;
        auto mid=pack,p=pack[Stride()],negative=pack[2*Stride()],flags=pack[4*Stride()],scratch=pack[5*Stride()];
        Duplicate(piv,lower,64);Duplicate(fac,upperBound,64);Duplicate(mid,0.0f,((queries+63)/64)*64);Barrier();
        Mul(tmp,origE,origE,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,tmp,uint8_t(Stride()/8),{1,8});
        Brcb(mat,origD,uint8_t(Stride()/8),{1,8});Barrier();
        for(uint32_t r=0;r<queries;++r){
            uint32_t target=lane+r*team+1,left=0,right=branches;float l=lower,h=upperBound;
            while(true){uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(j==target){mid.SetValue(r,value);break;}if(target<j){right=j;h=value;}else{left=j;l=value;}}
        }
        Barrier();SetVectorMask<float>(lanes);
        Sub<float,false>(p,mat,mid,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(p,p,-1e-30f,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight<uint32_t,false>(negative.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        for(uint32_t i=1;i<Order();++i){
            Div<float,false>(scratch,wideBuf[(i-1)*8],p,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            Sub<float,false>(p,mat[i*8],mid,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            Sub<float,false>(p,p,scratch,lanes,uint8_t(repeats),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(p,p,-1e-30f,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            ShiftRight<uint32_t,false>(flags.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<int32_t,false>(negative.template ReinterpretCast<int32_t>(),negative.template ReinterpretCast<int32_t>(),flags.template ReinterpretCast<int32_t>(),lanes,uint8_t(repeats),BinaryRepeatParams{1,1,1,8,8,8});
        }
        Barrier();Brcb(wideBuf,negative,uint8_t((queries+7)/8),{1,8});Barrier();
        DataCopyPad(gw[lane],wideBuf,DataCopyExtParams{uint16_t(queries),4,0,(team-1)*4,0});Barrier();
        SyncAll();DataCopy(prod,gw,branches);Barrier();
        for(uint32_t r=0;r<count;++r){
            uint32_t left=0,right=branches;float l=lower,h=upperBound;
            while(right-left>1 && h-l>tolerance){
                uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(value==l || value==h)break;
                if(prod.template ReinterpretCast<uint32_t>().GetValue(j-1)<=EigenIndex(r)){left=j;l=value;}else{right=j;h=value;}
            }
            const uint32_t countLeft=left?prod.template ReinterpretCast<uint32_t>().GetValue(left-1):0;
            const uint32_t countRight=right<branches?prod.template ReinterpretCast<uint32_t>().GetValue(right-1):Order();
            const uint32_t beforeLeft=left>1?prod.template ReinterpretCast<uint32_t>().GetValue(left-2):0;
            const uint32_t afterRight=right+1<branches?prod.template ReinterpretCast<uint32_t>().GetValue(right):Order();
            if(countRight-beforeLeft>1 || afterRight-countLeft>1)possibleCloseMask|=uint64_t(1)<<r;
            piv.SetValue(r,l);fac.SetValue(r,h);
        }
        return 1.0f/float(int32_t(branches));
    }
    __aicore__ inline void BisectVector(float lower,float upperBound,float tolerance,bool shared=false,float widthScale=1.0f){
        const uint32_t len=64,lanes=laneStride;
        // Coefficients and shifts come from this invocation's tridiagonal.
        // Materialize all diagonal-minus-shift values with vector repeats.
        Mul(tmp,origE,origE,Stride());PipeBarrier<PIPE_V>();
        Brcb(wideBuf,tmp,uint8_t(Stride()/8),{1,8});
        for(uint32_t col=0;col<lanes;col+=8)
            Brcb(mat[col],origD,uint8_t(Stride()/8),{uint16_t(lanes/8),uint16_t(lanes)});
        PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        auto lo=piv,hi=fac,mid=pack,p=pack[Stride()],negative=pack[2*Stride()];
        auto rank=pack[3*Stride()],one=pack[4*Stride()],scratch=pack[5*Stride()];
        auto mask=pack[6*Stride()].template ReinterpretCast<uint8_t>();
        auto activeMask=pack[6*Stride()+8].template ReinterpretCast<uint8_t>();
        auto oldLo=pack[7*Stride()],oldHi=pack[8*Stride()];
        if(!shared){Duplicate<float,false>(lo,lower,len,1,1,8);Duplicate<float,false>(hi,upperBound,len,1,1,8);}Duplicate<float,false>(one,0.0f,len,1,1,8);Barrier();
        for(uint32_t r=0;r<len;++r)rank.SetValue(r,float(int32_t(EigenIndex(r))));Barrier();
        float width=(upperBound-lower)*widthScale;
        for(uint32_t it=0;it<32 && width>0.5f*tolerance;++it,width*=0.5f){
            Adds<float,false>(oldLo,lo,0.0f,len,1,UnaryRepeatParams{1,1,8,8});Adds<float,false>(oldHi,hi,0.0f,len,1,UnaryRepeatParams{1,1,8,8});
            Sub<float,false>(scratch,hi,lo,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Compares<float,uint8_t,false>(activeMask,scratch,tolerance,CMPMODE::GT,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SetVectorMask<float>(lanes);
            for(uint32_t row=0;row<Order();row+=255){
                uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
                Sub<float,false>(prod[row*lanes],mat[row*lanes],mid,lanes,repeats,BinaryRepeatParams{1,1,1,uint8_t(lanes/8),uint8_t(lanes/8),0});
            }
            PipeBarrier<PIPE_V>();Adds<float,false>(p,prod,0.0f,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            // The perturbation makes zero pivots negative before counting and division.
            Adds<float,false>(p,p,-1e-30f,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            ShiftRight<uint32_t,false>(negative.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            for(uint32_t i=1;i<Order();++i){
                Div<float,false>(scratch,wideBuf[(i-1)*8],p,lanes,1,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
                Sub<float,false>(p,prod[i*lanes],scratch,lanes,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Adds<float,false>(p,p,-1e-30f,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                ShiftRight<uint32_t,false>(one.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                // Updating the count is independent of the next recurrence division.
                Add<int32_t,false>(negative.template ReinterpretCast<int32_t>(),negative.template ReinterpretCast<int32_t>(),one.template ReinterpretCast<int32_t>(),lanes,1,BinaryRepeatParams{1,1,1,8,8,8});
            }
            PipeBarrier<PIPE_V>();Cast<float,int32_t,false>(one,negative.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            Compare<float,uint8_t,false>(mask,one,rank,CMPMODE::LE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,mask,mid,lo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,mask,hi,mid,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,activeMask,lo,oldLo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,activeMask,hi,oldHi,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
        Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});Barrier();
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),mid.GetValue(r));
    }
    __aicore__ inline void CountSturmQueries(LocalTensor<float> queries,LocalTensor<float> pivot,
        LocalTensor<float> quotient,LocalTensor<int32_t> counts,LocalTensor<uint32_t> signs,
        LocalTensor<float> diagonalBroadcast,uint8_t repeats){
        // Snapshot the current Sturm phase before the boundary phase begins.
        const uint64_t sturmZeroSquaredEdgeMask=IndependentMedium?boundaryClusterMask:0;
        SetMaskNorm();SetVectorMask<float>(64);
        Sub<float,false>(pivot,diagonalBroadcast,queries,64,repeats,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
        if constexpr(!IndependentMedium){Adds<float,false>(pivot,pivot,-1e-30f,64,repeats,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();}
        if constexpr(IndependentMedium)Duplicate(counts,int32_t(0),64*uint32_t(repeats));
        else ShiftRight<uint32_t,false>(counts.template ReinterpretCast<uint32_t>(),pivot.template ReinterpretCast<uint32_t>(),uint32_t(31),64,repeats,UnaryRepeatParams{1,1,8,8});
        PipeBarrier<PIPE_V>();
        // Materialize diagonal-minus-query for a short row block. Reusing
        // it avoids one dependent vector instruction per Sturm recurrence row.
        const uint32_t span=64*uint32_t(repeats);
        LocalTensor<float> shifted;
        uint32_t blockRows;
        if constexpr(IndependentMedium){shifted=smallStorage.template Get<float>();blockRows=2*Order()*Stride()/span;blockRows=blockRows<16?blockRows:16;}
        else{
            const uint32_t offset=1792+8*Stride(),dense=Stride()*laneStride;
            const uint32_t available=3*dense+((dense+1)/2+7)/8*8+64-offset;
            shifted=mat[offset];blockRows=available/span;blockRows=blockRows<32?blockRows:32;
        }
        for(uint32_t first=1;first<Order();first+=blockRows){
            const uint32_t live=Order()-first<blockRows?Order()-first:blockRows;
            SetVectorMask<float>(64);
            for(uint32_t part=0;part<uint32_t(repeats);++part)
                Sub<float,false>(shifted[part*64],diagonalBroadcast[first*8],queries[part*64],64,uint8_t(live),BinaryRepeatParams{1,0,1,uint8_t(span/8),1,0});
            PipeBarrier<PIPE_V>();
            for(uint32_t local=0;local<live;++local){
                const uint32_t row=first+local;
                if constexpr(IndependentMedium){
                    // Division and sign extraction both read the old pivot.
                    // Accumulate its sign while forming the next pivot; the
                    // next recurrence does not consume the integer count.
                    // At a zero coupling this is a new independent block;
                    // form its diagonal pivot directly rather than 0/0.
                    if(sturmZeroSquaredEdgeMask&(uint64_t(1)<<(row-1)))
                        Duplicate<float,false>(quotient,0.0f,64,repeats,1,8);
                    else Div<float,false>(quotient,wideBuf[(row-1)*8],pivot,64,repeats,BinaryRepeatParams{1,0,1,8,0,8});
                    ShiftRight<uint32_t,false>(signs,pivot.template ReinterpretCast<uint32_t>(),uint32_t(31),64,repeats,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                    Sub<float,false>(pivot,shifted[local*span],quotient,64,repeats,BinaryRepeatParams{1,1,1,8,8,8});
                    Add<int32_t,false>(counts,counts,signs.template ReinterpretCast<int32_t>(),64,repeats,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                }else{
                Div<float,false>(quotient,wideBuf[(row-1)*8],pivot,64,repeats,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
                Sub<float,false>(pivot,shifted[local*span],quotient,64,repeats,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Adds<float,false>(pivot,pivot,-1e-30f,64,repeats,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                ShiftRight<uint32_t,false>(signs,pivot.template ReinterpretCast<uint32_t>(),uint32_t(31),64,repeats,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                // The next quotient does not read the count accumulator.
                Add<int32_t,false>(counts,counts,signs.template ReinterpretCast<int32_t>(),64,repeats,BinaryRepeatParams{1,1,1,8,8,8});
                }
            }
        }
        if constexpr(IndependentMedium){
            // The last pivot has no following division to pair with.
            ShiftRight<uint32_t,false>(signs,pivot.template ReinterpretCast<uint32_t>(),uint32_t(31),64,repeats,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<int32_t,false>(counts,counts,signs.template ReinterpretCast<int32_t>(),64,repeats,BinaryRepeatParams{1,1,1,8,8,8});
        }
        PipeBarrier<PIPE_V>();
    }
    // Eight ordered Sturm queries per eigenvalue produce nine subintervals.
    // The private-medium query pool covers the complete 64-root SIMD domain.
    __aicore__ inline void BisectMediumQueries(float lower,float upperBound,float tolerance,bool shared,float widthScale){
        if(!IndependentMedium && laneStride>32){BisectVector(lower,upperBound,tolerance,shared,widthScale);return;}
        uint64_t &sturmZeroSquaredEdgeMask=boundaryClusterMask;
        constexpr uint32_t maximumRootCount=64;
        // Keep eight points when they already fit one physical SIMD repeat.
        const uint32_t pointsPerRoot=laneStride>8?4:8;
        const float queryStep=pointsPerRoot==4?0.2f:(1.0f/9.0f);
        constexpr uint32_t maximumQueryCount=8*maximumRootCount;
        constexpr uint32_t vectorOffset=4*maximumQueryCount;
        constexpr uint32_t candidateOffset=vectorOffset+4*maximumRootCount+16;
        constexpr uint32_t coefficientOffset=(candidateOffset+maximumQueryCount+63)/64*64;
        const uint32_t slots=laneStride;const uint8_t queryRepeats=uint8_t((pointsPerRoot*slots+63)/64);
        // The whole-spectrum fallback needs 512 query lanes. Keep their
        // pivots, counts, masks and coefficient broadcasts disjoint.
        auto queries=mat,pivot=mat[maximumQueryCount],quotient=mat[2*maximumQueryCount];
        auto counts=mat[3*maximumQueryCount].template ReinterpretCast<int32_t>();
        auto width=mat[vectorOffset],rank=mat[vectorOffset+maximumRootCount];
        auto oldLo=mat[vectorOffset+2*maximumRootCount],oldHi=mat[vectorOffset+3*maximumRootCount];
        auto mask=mat[vectorOffset+4*maximumRootCount].template ReinterpretCast<uint8_t>();
        auto activeMask=mat[vectorOffset+4*maximumRootCount+8].template ReinterpretCast<uint8_t>();
        auto candidate=mat[candidateOffset];
        auto signs=mat[candidateOffset].template ReinterpretCast<uint32_t>();
        auto diagonalBroadcast=mat[coefficientOffset],lo=piv,hi=fac;
        Mul(v,origE,origE,Stride());PipeBarrier<PIPE_V>();
        Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});Brcb(diagonalBroadcast,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
        if constexpr(IndependentMedium){
            // Observe the actual vector-squared coefficients, including any
            // target underflow, once per current tridiagonal problem.
            VectorToScalar();sturmZeroSquaredEdgeMask=0;
            for(uint32_t edge=0;edge+1<Order();++edge)
                if(v.GetValue(edge)==0.0f)sturmZeroSquaredEdgeMask|=uint64_t(1)<<edge;
            // This grid only seeds brackets; fine Sturm searches retain
            // their original tolerance and all root/cluster checks.
            constexpr uint32_t coarseQueryCount=512;
            // A private coarse grid is shared by every eigenvalue owned by
            // this AIV; no inter-core workspace or communication is needed.
            CreateIndicesWithArena(queries.template ReinterpretCast<int32_t>(),int32_t(1),coarseQueryCount);
            Cast(queries,queries.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,coarseQueryCount);PipeBarrier<PIPE_V>();
            Muls(queries,queries,(upperBound-lower)/float(int32_t(coarseQueryCount)+1),coarseQueryCount);PipeBarrier<PIPE_V>();Adds(queries,queries,lower,coarseQueryCount);PipeBarrier<PIPE_V>();
            hasClusterGrid=true;
            CountSturmQueries(queries,pivot,quotient,counts,signs,diagonalBroadcast,uint8_t(coarseQueryCount/64));VectorToScalar();
            for(uint32_t r=0;r<slots;++r){
                uint32_t first=0,last=coarseQueryCount,target=EigenIndex(r);
                while(first<last){uint32_t mid=(first+last)/2;if(counts.GetValue(mid)<=int32_t(target))first=mid+1;else last=mid;}
                if(r<count){
                    uint32_t left=first?uint32_t(counts.GetValue(first-1)):0;
                    uint32_t right=first<coarseQueryCount?uint32_t(counts.GetValue(first)):Order();
                    uint32_t before=first>1?uint32_t(counts.GetValue(first-2)):0;
                    uint32_t after=first+1<coarseQueryCount?uint32_t(counts.GetValue(first+1)):Order();
                    if(right-before>1 || after-left>1)possibleCloseMask|=uint64_t(1)<<r;
                }
                lo.SetValue(r,first?queries.GetValue(first-1):lower);hi.SetValue(r,first<coarseQueryCount?queries.GetValue(first):upperBound);
            }
        }else if(!shared){Duplicate(lo,lower,slots);Duplicate(hi,upperBound,slots);}
        for(uint32_t r=0;r<slots;++r){
            rank.SetValue(r,float(int32_t(EigenIndex(r))));
            if(r>=count){lo.SetValue(r,lower);hi.SetValue(r,upperBound);}
        }
        Duplicate(queries,0.0f,maximumQueryCount);
        // Eight root slots and eight query points form one SIMD repeat.
        // These current-call offsets transpose only that 8x8 query tile.
        auto transposeOffsets=candidate[256].template ReinterpretCast<uint32_t>();
        if(slots==8){
            for(uint32_t root=0;root<8;++root)
                for(uint32_t point=0;point<8;++point)
                    transposeOffsets.SetValue(root*8+point,((point&(pointsPerRoot-1))*8+root)*sizeof(float));
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            SetVectorMask<float>(8);
            Copy<float,false>(rank[8],rank,8,7,CopyRepeatParams{1,1,1,0});PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);Barrier();
        bool invalid=false;
        for(uint32_t iteration=0;iteration<32;++iteration){
            float widest=0.0f;
            for(uint32_t r=0;r<count;++r){float span=hi.GetValue(r)-lo.GetValue(r);if(span<0)invalid=true;if(span>widest)widest=span;}
            if(invalid || widest<=tolerance)break;
            SetVectorMask<float>(slots);
            Adds<float,false>(oldLo,lo,0.0f,slots,1,UnaryRepeatParams{1,1,8,8});Adds<float,false>(oldHi,hi,0.0f,slots,1,UnaryRepeatParams{1,1,8,8});
            Sub<float,false>(width,hi,lo,slots,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Compares<float,uint8_t,false>(activeMask,width,tolerance,CMPMODE::GT,slots,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            for(uint32_t point=0;point<pointsPerRoot;++point)Adds<float,false>(queries[point*slots],lo,0.0f,slots,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            for(uint32_t point=0;point<pointsPerRoot;++point)Axpy<float,float,false>(queries[point*slots],width,float(int32_t(point+1))*queryStep,slots,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            CountSturmQueries(queries,pivot,quotient,counts,signs,diagonalBroadcast,queryRepeats);
            Cast<float,int32_t,false>(counts.template ReinterpretCast<float>(),counts,RoundMode::CAST_NONE,64,queryRepeats,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SetVectorMask<float>(slots);
            if(slots==8){
                auto lowerCandidates=candidate,upperCandidates=candidate[64];
                auto rootLower=candidate[128],rootUpper=candidate[192];
                auto previousLower=candidate[384],previousUpper=candidate[448];
                // Bounds for rejected query points come from this iteration,
                // preserving the old ordered loop's exact interval semantics.
                Copy<float,false>(previousLower,oldLo,8,8,CopyRepeatParams{1,1,1,0});
                Copy<float,false>(previousUpper,oldHi,8,8,CopyRepeatParams{1,1,1,0});PipeBarrier<PIPE_V>();
                SetVectorMask<float>(64);
                Compare<float,uint8_t,false>(mask,counts.template ReinterpretCast<float>(),rank,CMPMODE::LE,64,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Select<float,uint8_t,false>(lowerCandidates,mask,queries,previousLower,SELMODE::VSEL_TENSOR_TENSOR_MODE,64,1,BinaryRepeatParams{1,1,1,8,8,8});
                Select<float,uint8_t,false>(upperCandidates,mask,previousUpper,queries,SELMODE::VSEL_TENSOR_TENSOR_MODE,64,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Gather(rootLower,lowerCandidates,transposeOffsets,0,64);
                Gather(rootUpper,upperCandidates,transposeOffsets,0,64);PipeBarrier<PIPE_V>();
                WholeReduceMax(lo,rootLower,8,8,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
                WholeReduceMin(hi,rootUpper,8,8,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                SetVectorMask<float>(slots);
            }else{
            for(uint32_t point=0;point<pointsPerRoot;++point){
                auto query=queries[point*slots],number=counts.template ReinterpretCast<float>()[point*slots];
                Compare<float,uint8_t,false>(mask,number,rank,CMPMODE::LE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Select<float,uint8_t,false>(lo,mask,query,lo,SELMODE::VSEL_TENSOR_TENSOR_MODE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});
                Min<float,false>(candidate,hi,query,slots,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Select<float,uint8_t,false>(hi,mask,hi,candidate,SELMODE::VSEL_TENSOR_TENSOR_MODE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            }
            Select<float,uint8_t,false>(lo,activeMask,lo,oldLo,SELMODE::VSEL_TENSOR_TENSOR_MODE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,activeMask,hi,oldHi,SELMODE::VSEL_TENSOR_TENSOR_MODE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});VectorToScalar();
        }
        ResetMask();
        if(invalid){BisectVector(lower,upperBound,tolerance,false,1.0f);return;}
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),0.5f*(lo.GetValue(r)+hi.GetValue(r)));
        Barrier();
    }

    // Eight ordered Sturm queries per eigenvalue produce nine subintervals.
    // The query pool uses at most four SIMD repeats, including padding slots.
    __aicore__ inline void BisectMultisection(float lower,float upperBound,float tolerance,bool shared,float widthScale){
        if constexpr(IndependentMedium){BisectMediumQueries(lower,upperBound,tolerance,shared,widthScale);return;}
        if(laneStride>32){BisectVector(lower,upperBound,tolerance,shared,widthScale);return;}
        const uint32_t slots=laneStride;const uint8_t queryRepeats=uint8_t((8*slots+63)/64);
        auto queries=mat,pivot=mat[256],quotient=mat[512];
        auto counts=mat[768].template ReinterpretCast<int32_t>();
        auto width=mat[1024],rank=mat[1088],oldLo=mat[1152],oldHi=mat[1216];
        auto mask=mat[1280].template ReinterpretCast<uint8_t>(),activeMask=mat[1288].template ReinterpretCast<uint8_t>();
        auto candidate=mat[1296];
        auto signs=mat[1296].template ReinterpretCast<uint32_t>();
        auto diagonalBroadcast=mat[1792],lo=piv,hi=fac;
        Mul(v,origE,origE,Stride());PipeBarrier<PIPE_V>();
        Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});Brcb(diagonalBroadcast,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
        if(!shared){Duplicate(lo,lower,slots);Duplicate(hi,upperBound,slots);}
        for(uint32_t r=0;r<slots;++r){
            rank.SetValue(r,float(int32_t(EigenIndex(r))));
            if(r>=count){lo.SetValue(r,lower);hi.SetValue(r,upperBound);}
        }
        Duplicate(queries,0.0f,256);
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);Barrier();
        bool invalid=false;
        for(uint32_t iteration=0;iteration<32;++iteration){
            float widest=0.0f;
            for(uint32_t r=0;r<count;++r){float span=hi.GetValue(r)-lo.GetValue(r);if(span<0)invalid=true;if(span>widest)widest=span;}
            if(invalid || widest<=tolerance)break;
            SetVectorMask<float>(slots);
            Adds<float,false>(oldLo,lo,0.0f,slots,1,UnaryRepeatParams{1,1,8,8});Adds<float,false>(oldHi,hi,0.0f,slots,1,UnaryRepeatParams{1,1,8,8});
            Sub<float,false>(width,hi,lo,slots,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Compares<float,uint8_t,false>(activeMask,width,tolerance,CMPMODE::GT,slots,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            for(uint32_t point=0;point<8;++point)Adds<float,false>(queries[point*slots],lo,0.0f,slots,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            for(uint32_t point=0;point<8;++point)Axpy<float,float,false>(queries[point*slots],width,float(int32_t(point+1))*(1.0f/9.0f),slots,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            CountSturmQueries(queries,pivot,quotient,counts,signs,diagonalBroadcast,queryRepeats);
            Cast<float,int32_t,false>(counts.template ReinterpretCast<float>(),counts,RoundMode::CAST_NONE,64,queryRepeats,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SetVectorMask<float>(slots);
            for(uint32_t point=0;point<8;++point){
                auto query=queries[point*slots],number=counts.template ReinterpretCast<float>()[point*slots];
                Compare<float,uint8_t,false>(mask,number,rank,CMPMODE::LE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Select<float,uint8_t,false>(lo,mask,query,lo,SELMODE::VSEL_TENSOR_TENSOR_MODE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});
                Min<float,false>(candidate,hi,query,slots,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Select<float,uint8_t,false>(hi,mask,hi,candidate,SELMODE::VSEL_TENSOR_TENSOR_MODE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            Select<float,uint8_t,false>(lo,activeMask,lo,oldLo,SELMODE::VSEL_TENSOR_TENSOR_MODE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,activeMask,hi,oldHi,SELMODE::VSEL_TENSOR_TENSOR_MODE,slots,1,BinaryRepeatParams{1,1,1,8,8,8});VectorToScalar();
        }
        ResetMask();
        if(invalid){BisectVector(lower,upperBound,tolerance,false,1.0f);return;}
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),0.5f*(lo.GetValue(r)+hi.GetValue(r)));
        Barrier();
    }

    // One full SIMD repeat evaluates 64 distinct Sturm queries. This
    // replaces six dependent scalar bisections with one vector recurrence.
    // Nearby eigenvalues amplify the backward error of float LU factors.
    // Two current-input Sturm queries select only columns needing two-float LU.
    __aicore__ inline void DetectNearbyRoots(){
        // This entry checks every column through its complete Gram matrix.
        if constexpr(CubeMode==3){extendedEigenvectorMask=0;return;}
        extendedEigenvectorMask=0;
        if(ResidentFamily()){
            if(Order()>=9 && smallRowBound>=1e-10f && smallRowBound<=16.0f)ConfirmResidentRootClusters();
            return;
        }
        if(hasClusterGrid && !possibleCloseMask){
            if constexpr(IndependentMedium)boundaryClusterMask=0;
            return;
        }
        const uint32_t slots=laneStride;
        const uint8_t repeats=uint8_t((2*slots+63)/64);
        auto shifts=mat,pivot=mat[128],quotient=mat[256];
        auto counts=mat[384].template ReinterpretCast<int32_t>();
        auto signs=mat[784].template ReinterpretCast<uint32_t>();
        auto diagonalBroadcast=mat[1024];
        Duplicate(shifts,0.0f,128);VectorToScalar();
        const float radius=0.001f*spectralBound;
        for(uint32_t r=0;r<count;++r){float value=d.GetValue(EigenIndex(r));shifts.SetValue(r,value-radius);shifts.SetValue(r+slots,value+radius);}
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Mul(v,origE,origE,Stride());PipeBarrier<PIPE_V>();
        Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});Brcb(diagonalBroadcast,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
        CountSturmQueries(shifts,pivot,quotient,counts,signs,diagonalBroadcast,repeats);VectorToScalar();
        // All zero-edge queries are complete. Begin a fresh boundary mask.
        if constexpr(IndependentMedium)boundaryClusterMask=0;
        for(uint32_t r=0;r<count;++r)if((!hasClusterGrid || (possibleCloseMask&(uint64_t(1)<<r))) && counts.GetValue(r+slots)-counts.GetValue(r)>1)extendedEigenvectorMask|=uint64_t(1)<<r;
        if constexpr(IndependentMedium){
            for(uint32_t owned=0;owned<count;++owned)if(extendedEigenvectorMask&(uint64_t(1)<<owned)){
                const uint32_t below=uint32_t(counts.GetValue(owned));
                const uint32_t above=uint32_t(counts.GetValue(owned+slots));
                if(below<base || above>base+count)boundaryClusterMask|=uint64_t(1)<<owned;
            }
        }
        ResetMask();Barrier();
    }
    __aicore__ inline void CountResidentSturmQueries(){
        constexpr uint32_t queryCount=128;
        auto arena=smallStorage.template Get<float>();
        auto queries=arena[queryCount],savedPivot=arena[2*queryCount];
        auto reciprocal=arena[3*queryCount],shifted=arena[8*queryCount];
        auto counts=arena[5*queryCount].template ReinterpretCast<uint32_t>();
        auto signs=arena[6*queryCount].template ReinterpretCast<uint32_t>();
        auto ones=arena[7*queryCount],diagonalBroadcast=wideBuf;
        const uint32_t availableRows=(15*Order()*Stride()-8*queryCount)/queryCount;
        const uint32_t rowsPerBlock=availableRows<Order()?availableRows:Order();
        SetMaskNorm();SetVectorMask<float>(64);
                auto pivot=savedPivot;
                for(uint32_t first=0;first<Order();first+=rowsPerBlock){
                    if(first){Adds<float,false>(savedPivot,pivot,0.0f,64,2,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();pivot=savedPivot;}
                    const uint32_t live=Order()-first<rowsPerBlock?Order()-first:rowsPerBlock;
                    // Recycle the retired matrix area for d_i-shift. The
                    // recurrence then needs only five vector instructions.
                    for(uint32_t part=0;part<2;++part)
                        Sub<float,false>(shifted[part*64],diagonalBroadcast[first*8],queries[part*64],64,uint8_t(live),BinaryRepeatParams{1,0,1,16,1,0});
                    PipeBarrier<PIPE_V>();
                    if(!first){pivot=shifted;Duplicate(counts,uint32_t(0),queryCount);PipeBarrier<PIPE_V>();}
                    for(uint32_t local=first?0:1;local<live;++local){
                        const uint32_t row=first+local;
                        // Keep the unbiased pivot for the sign count while
                        // dividing its tiny-negative-biased copy independently.
                        Adds<float,false>(reciprocal,pivot,-1e-30f,64,2,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                        Div<float,false>(reciprocal,ones,reciprocal,64,2,BinaryRepeatParams{1,1,1,8,8,8});
                        ShiftRight<uint32_t,false>(signs,pivot.template ReinterpretCast<uint32_t>(),uint32_t(31),64,2,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                        const float edge=origE.GetValue(row-1);auto next=shifted[local*queryCount];
                        Axpy<float,float,false>(next,reciprocal,-edge*edge,64,2,UnaryRepeatParams{1,1,8,8});
                        Add<int32_t,false>(counts.template ReinterpretCast<int32_t>(),counts.template ReinterpretCast<int32_t>(),signs.template ReinterpretCast<int32_t>(),64,2,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                        pivot=next;
                    }
                }
        // The last pivot has no following division to overlap with.
        ShiftRight<uint32_t,false>(signs,pivot.template ReinterpretCast<uint32_t>(),uint32_t(31),64,2,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Add<int32_t,false>(counts.template ReinterpretCast<int32_t>(),counts.template ReinterpretCast<int32_t>(),signs.template ReinterpretCast<int32_t>(),64,2,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ConfirmResidentRootClusters(){
        if(!useExtendedEigenvectors)return;
        constexpr uint32_t queryCount=128;
        auto arena=smallStorage.template Get<float>();
        auto queries=arena[queryCount],ones=arena[7*queryCount];
        auto counts=arena[5*queryCount].template ReinterpretCast<uint32_t>();
        Duplicate(queries,0.0f,queryCount);Duplicate(ones,1.0f,queryCount);VectorToScalar();
        const float radius=0.001f*spectralBound;
        for(uint32_t owned=0;owned<count;++owned){
            const float eigenvalue=d.GetValue(EigenIndex(owned));
            queries.SetValue(owned,eigenvalue-radius);
            queries.SetValue(owned+count,eigenvalue+radius);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
        CountResidentSturmQueries();VectorToScalar();
        // Coarse cells only nominate possible clusters. Confirm them at the
        // computed eigenvalues before selecting the more costly two-float LU.
        useExtendedEigenvectors=false;
        for(uint32_t owned=0;owned<count;++owned)
            if(counts.GetValue(owned+count)-counts.GetValue(owned)>1)useExtendedEigenvectors=true;
        ResetMask();Barrier();
    }

    __aicore__ inline void SmallGridBisection(float lower,float upperBound,float tolerance){
        constexpr uint32_t queryCount=128;
        auto arena=smallStorage.template Get<float>();
        auto positions=arena,queries=arena[queryCount],savedPivot=arena[2*queryCount];
        auto reciprocal=arena[3*queryCount],shifted=arena[8*queryCount];
        auto counts=arena[5*queryCount].template ReinterpretCast<uint32_t>();
        auto signs=arena[6*queryCount].template ReinterpretCast<uint32_t>();
        auto ones=arena[7*queryCount],diagonalBroadcast=wideBuf;
        const uint32_t availableRows=(15*Order()*Stride()-8*queryCount)/queryCount;
        const uint32_t rowsPerBlock=availableRows<Order()?availableRows:Order();
        CreateIndicesWithArena(positions.template ReinterpretCast<int32_t>(),int32_t(1),queryCount);
        Cast(positions,positions.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,queryCount);
        Duplicate(ones,1.0f,queryCount);Brcb(diagonalBroadcast,origD,uint8_t(Stride()/8),{1,8});Barrier();SetMaskNorm();SetVectorMask<float>(64);
        for(uint32_t owned=0;owned<count;++owned){
            float lo=lower,hi=upperBound;const uint32_t rank=EigenIndex(owned);
            for(uint32_t pass=0;pass<12 && hi-lo>tolerance;++pass){
                Muls<float,false>(queries,positions,(hi-lo)/float(int32_t(queryCount+1)),64,2,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                Adds<float,false>(queries,queries,lo,64,2,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                CountResidentSturmQueries();
                VectorToScalar();float nextLo=lo,nextHi=hi;
                uint32_t first=0,last=queryCount;
                while(first<last){uint32_t midpoint=(first+last)/2;if(counts.GetValue(midpoint)<=rank)first=midpoint+1;else last=midpoint;}
                if(pass==0){
                    uint32_t lowRank=first>1?counts.GetValue(first-2):0;
                    uint32_t highRank=first+1<queryCount?counts.GetValue(first+1):Order();
                    useExtendedEigenvectors=useExtendedEigenvectors || highRank-lowRank>1;
                }
                if(first)nextLo=queries.GetValue(first-1);
                if(first<queryCount)nextHi=queries.GetValue(first);
                if(nextLo==lo && nextHi==hi)break;
                lo=nextLo;hi=nextHi;
                // An empty neighboring coarse cell on both sides gives a
                // separation of at least one first-pass grid spacing. After
                // two 129-way passes, the midpoint error is at most 1/258
                // of that separation. The two twisted Rayleigh corrections
                // then compute the eigenpair; clustered roots retain the
                // original full bisection tolerance and compensated fallback.
                if(pass==1 && smallTwistedFactorization && !useExtendedEigenvectors)break;
            }
            d.SetValue(rank,0.5f*(lo+hi));
        }
        ResetMask();Barrier();
    }


    __aicore__ inline float ReferenceCommonGrid(float lower,float upperBound,float tolerance){
        uint32_t limit=4*Stride()<64*team?4*Stride():64*team;
        uint32_t branches=1;while(2*branches<=limit)branches*=2;
        uint32_t queries=lane<branches-1?(branches-2-lane)/team+1:0;
        uint32_t lanes=queries>64?64:(queries+7)/8*8,repeats=(queries+63)/64;
        auto mid=pack,p=pack[Stride()],negative=pack[2*Stride()],flags=pack[4*Stride()],scratch=pack[5*Stride()];
        Duplicate(piv,lower,64);Duplicate(fac,upperBound,64);Duplicate(mid,0.0f,((queries+63)/64)*64);Barrier();
        Mul(tmp,origE,origE,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,tmp,uint8_t(Stride()/8),{1,8});
        Brcb(mat,origD,uint8_t(Stride()/8),{1,8});Barrier();
        for(uint32_t r=0;r<queries;++r){
            uint32_t target=lane+r*team+1,left=0,right=branches;float l=lower,h=upperBound;
            while(true){uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(j==target){mid.SetValue(r,value);break;}if(target<j){right=j;h=value;}else{left=j;l=value;}}
        }
        Barrier();SetVectorMask<float>(lanes);
        Sub<float,false>(p,mat,mid,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(p,p,-1e-30f,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight<uint32_t,false>(negative.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        for(uint32_t i=1;i<Order();++i){
            Div<float,false>(scratch,wideBuf[(i-1)*8],p,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            Sub<float,false>(p,mat[i*8],mid,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            Sub<float,false>(p,p,scratch,lanes,uint8_t(repeats),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(p,p,-1e-30f,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            ShiftRight<uint32_t,false>(flags.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<int32_t,false>(negative.template ReinterpretCast<int32_t>(),negative.template ReinterpretCast<int32_t>(),flags.template ReinterpretCast<int32_t>(),lanes,uint8_t(repeats),BinaryRepeatParams{1,1,1,8,8,8});
        }
        Barrier();Brcb(wideBuf,negative,uint8_t((queries+7)/8),{1,8});Barrier();
        DataCopyPad(gw[lane],wideBuf,DataCopyExtParams{uint16_t(queries),4,0,(team-1)*4,0});Barrier();
        SyncAll();DataCopy(prod,gw,branches);Barrier();
        for(uint32_t r=0;r<count;++r){
            uint32_t left=0,right=branches;float l=lower,h=upperBound;
            while(right-left>1 && h-l>tolerance){
                uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(value==l || value==h)break;
                if(prod.template ReinterpretCast<uint32_t>().GetValue(j-1)<=EigenIndex(r)){left=j;l=value;}else{right=j;h=value;}
            }
            piv.SetValue(r,l);fac.SetValue(r,h);
        }
        return 1.0f/float(int32_t(branches));
    }
    __aicore__ inline void SchurInertia(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){uint8_t rep=uint8_t(Order()-row<255?Order()-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,rep,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});}
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){uint8_t rep=uint8_t(Order()-row<255?Order()-row:255);
            Adds<float,false>(pcrBand[row*len],wideBuf[row*8],0.0f,len,rep,UnaryRepeatParams{1,0,uint8_t(len/8),1});}
        PipeBarrier<PIPE_V>();
        for(uint32_t step=1;step<Order();step*=2){
            uint32_t odd=(Order()-1-step)/(2*step)+1,right=(Order()-1)/(2*step),dist=2*step*len;
            auto od=mat[step*len],oe=pcrBand[step*len];
            if(dist/8<=255){for(uint32_t first=0;first<odd;first+=255){uint8_t rep=uint8_t(odd-first<255?odd-first:255);
                Adds<float,false>(od[first*dist],od[first*dist],-1e-30f,len,rep,UnaryRepeatParams{1,1,uint8_t(dist/8),uint8_t(dist/8)});}}
            else for(uint32_t i=0;i<odd;++i)Adds<float,false>(od[i*dist],od[i*dist],-1e-30f,len,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            PcrOp<1>(pcrLeft,len,one,0,od,dist,len,odd);
            PcrOp<0>(wideBuf,len,pcrBand,dist,pcrLeft,len,len,odd);
            PcrOp<0>(wideBuf,len,wideBuf,len,pcrBand,dist,len,odd);
            PcrOp<2>(mat,dist,mat,dist,wideBuf,len,len,odd);
            if(right){
                PcrOp<0>(wideBuf,len,oe,dist,pcrLeft,len,len,right);
                PcrOp<0>(wideBuf,len,wideBuf,len,oe,dist,len,right);
                PcrOp<2>(mat[dist],dist,mat[dist],dist,wideBuf,len,len,right);
            }
            PcrOp<0>(wideBuf,len,pcrBand,dist,pcrLeft,len,len,odd);
            PcrOp<0>(wideBuf,len,wideBuf,len,oe,dist,len,odd);
            Muls(wideBuf,wideBuf,-1.0f,odd*len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            PcrOp<0>(pcrBand,dist,wideBuf,len,one,0,len,odd);
        }
        Adds<float,false>(mat,mat,-1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight(mat.template ReinterpretCast<uint32_t>(),mat.template ReinterpretCast<uint32_t>(),uint32_t(31),Order()*len);PipeBarrier<PIPE_V>();
        for(uint32_t activeRows=Order();activeRows>1;activeRows=(activeRows+1)/2){
            uint32_t half=activeRows/2,next=(activeRows+1)/2;
            Add(mat.template ReinterpretCast<int32_t>(),mat.template ReinterpretCast<int32_t>(),mat[next*len].template ReinterpretCast<int32_t>(),half*len);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
    }
    __aicore__ inline void BisectInertia(float lower,float upperBound,float tolerance,bool shared){
        uint32_t len=(count+7)/8*8;
        auto lo=piv,hi=fac,mid=pack,one=pack[Stride()],rank=pack[2*Stride()],counts=pack[3*Stride()],width=pack[4*Stride()];
        auto oldLo=pack[7*Stride()],oldHi=pack[8*Stride()];
        auto mask=pack[5*Stride()].template ReinterpretCast<uint8_t>(),activeMask=pack[6*Stride()].template ReinterpretCast<uint8_t>();
        if(!shared){Duplicate(lo,lower,len);Duplicate(hi,upperBound,len);}
        Duplicate(one,1.0f,len);Barrier();
        for(uint32_t i=0;i<len;++i)rank.SetValue(i,float(int32_t(EigenIndex(i))));Barrier();
        for(uint32_t it=0;it<32;++it){
            float widest=0;for(uint32_t i=0;i<count;++i){float span=hi.GetValue(i)-lo.GetValue(i);widest=widest>span?widest:span;}
            if(widest<=tolerance)break;
            SetVectorMask<float>(len);
            Adds<float,false>(oldLo,lo,0.0f,len,1,UnaryRepeatParams{1,1,8,8});Adds<float,false>(oldHi,hi,0.0f,len,1,UnaryRepeatParams{1,1,8,8});
            Sub<float,false>(width,hi,lo,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Compares<float,uint8_t,false>(activeMask,width,tolerance,CMPMODE::GT,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SchurInertia(mid,one,len);
            Cast<float,int32_t,false>(counts,mat.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Compare<float,uint8_t,false>(mask,counts,rank,CMPMODE::LE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,mask,mid,lo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,mask,hi,mid,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,activeMask,lo,oldLo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,activeMask,hi,oldHi,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});Barrier();
        }
        SetVectorMask<float>(len);Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});Barrier();
        for(uint32_t i=0;i<count;++i)d.SetValue(EigenIndex(i),mid.GetValue(i));
    }
    __aicore__ inline void ScaleOriginalEdges(LocalTensor<float> dst,LocalTensor<float> src,uint32_t sourceStride,uint32_t firstEdge,uint32_t len,uint32_t number){
        for(uint32_t first=0;first<number;first+=255){
            uint8_t repeats=uint8_t(number-first<255?number-first:255);
            Mul<float,false>(dst[first*len],wideBuf[(firstEdge+2*first)*8],src[first*sourceStride],len,repeats,
                BinaryRepeatParams{1,0,1,uint8_t(len/8),2,uint8_t(sourceStride/8)});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void SchurInertiaInplace(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
        // Match the original initialization, including the sign of zero.
        Adds(wideBuf,wideBuf,0.0f,8*Stride());
        Duplicate(tmp2.template ReinterpretCast<int32_t>(),int32_t(0),len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t step=1;step<Order();step*=2){
            uint32_t odd=(Order()-1-step)/(2*step)+1,right=(Order()-1)/(2*step),dist=2*step*len;
            auto od=mat[step*len];bool batchable=dist/8<=255;
            for(uint32_t first=0;first<odd;){
                uint8_t repeats=batchable?uint8_t(odd-first<255?odd-first:255):1;
                Adds<float,false>(od[first*dist],od[first*dist],-1e-30f,len,repeats,
                    UnaryRepeatParams{1,1,uint8_t(batchable?dist/8:8),uint8_t(batchable?dist/8:8)});
                first+=repeats;
            }
            PipeBarrier<PIPE_V>();
            for(uint32_t first=0;first<odd;){
                uint8_t repeats=batchable?uint8_t(odd-first<255?odd-first:255):1;
                ShiftRight<uint32_t,false>(prod[first*len].template ReinterpretCast<uint32_t>(),od[first*dist].template ReinterpretCast<uint32_t>(),31u,len,repeats,
                    UnaryRepeatParams{1,1,uint8_t(len/8),uint8_t(batchable?dist/8:8)});
                first+=repeats;
            }
            PipeBarrier<PIPE_V>();
            for(uint32_t number=odd;number>1;number=(number+1)/2){
                uint32_t half=number/2,next=(number+1)/2;
                Add(prod.template ReinterpretCast<int32_t>(),prod.template ReinterpretCast<int32_t>(),prod[next*len].template ReinterpretCast<int32_t>(),half*len);PipeBarrier<PIPE_V>();
            }
            Add(tmp2.template ReinterpretCast<int32_t>(),tmp2.template ReinterpretCast<int32_t>(),prod.template ReinterpretCast<int32_t>(),len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            // Once its sign is counted, an odd diagonal stores its reciprocal.
            // Once both neighbor updates finish, the same slot stores the new
            // off-diagonal. Old edges occupy disjoint midpoint slots.
            PcrOp<1>(od,dist,one,0,od,dist,len,odd);
            if(step==1){
                ScaleOriginalEdges(prod,od,dist,0,len,odd);
                ScaleOriginalEdges(prod,prod,len,0,len,odd);
            }else{
                auto left=mat[(step/2)*len];
                PcrOp<0>(prod,len,left,dist,od,dist,len,odd);
                PcrOp<0>(prod,len,prod,len,left,dist,len,odd);
            }
            PcrOp<2>(mat,dist,mat,dist,prod,len,len,odd);
            if(right){
                if(step==1){
                    ScaleOriginalEdges(prod,od,dist,1,len,right);
                    ScaleOriginalEdges(prod,prod,len,1,len,right);
                }else{
                    auto edge=mat[(step+step/2)*len];
                    PcrOp<0>(prod,len,edge,dist,od,dist,len,right);
                    PcrOp<0>(prod,len,prod,len,edge,dist,len,right);
                }
                PcrOp<2>(mat[dist],dist,mat[dist],dist,prod,len,len,right);
                if(step==1){
                    ScaleOriginalEdges(prod,od,dist,0,len,right);
                    ScaleOriginalEdges(prod,prod,len,1,len,right);
                }else{
                    PcrOp<0>(prod,len,mat[(step/2)*len],dist,od,dist,len,right);
                    PcrOp<0>(prod,len,prod,len,mat[(step+step/2)*len],dist,len,right);
                }
                Muls(prod,prod,-1.0f,right*len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
                PcrOp<0>(od,dist,prod,len,one,0,len,right);
            }
        }
        Adds<float,false>(mat,mat,-1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight<uint32_t,false>(prod.template ReinterpretCast<uint32_t>(),mat.template ReinterpretCast<uint32_t>(),31u,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Add(mat.template ReinterpretCast<int32_t>(),tmp2.template ReinterpretCast<int32_t>(),prod.template ReinterpretCast<int32_t>(),len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
    }

    __aicore__ __attribute__((noinline)) float CommonGrid(float lower,float upperBound,float tolerance){
        if(ResidentFamily() || laneStride==64)return ReferenceCommonGrid(lower,upperBound,tolerance);
        uint32_t limit=4*Stride()<64*team?4*Stride():64*team;
        uint32_t branches=1;while(2*branches<=limit)branches*=2;
        uint32_t len=0;
        while(branches>=2){
            uint32_t maxQueries=(branches-2)/team+1;
            len=(maxQueries+7)/8*8;
            uint32_t needed=(Order()+(Order()+1)/2)*len+4*Stride()+3*len+64;
            if(len<=64 && needed<=storageCapacity)break;
            branches/=2;
        }
        if(branches<2)return ReferenceCommonGrid(lower,upperBound,tolerance);
        uint32_t queries=lane<branches-1?(branches-2-lane)/team+1:0;
        auto oldMat=mat,oldProd=prod,oldTmp2=tmp2;
        auto oldD=d,oldOrigD=origD,oldOrigE=origE,oldBeta=beta;
        DataCopy(wideBuf,origD,Stride());DataCopy(wideBuf[Stride()],origE,Stride());
        DataCopy(wideBuf[2*Stride()],beta,Stride());DataCopy(wideBuf[3*Stride()],d,Stride());Barrier();
        auto memory=storage.template Get<float>();uint32_t prefix=(Order()+(Order()+1)/2)*len;
        mat=memory;prod=memory[Order()*len];origD=memory[prefix];origE=memory[prefix+Stride()];
        beta=memory[prefix+2*Stride()];d=memory[prefix+3*Stride()];
        auto mid=memory[prefix+4*Stride()],one=mid[len];tmp2=one[len];
        DataCopy(origD,wideBuf,Stride());DataCopy(origE,wideBuf[Stride()],Stride());
        DataCopy(beta,wideBuf[2*Stride()],Stride());DataCopy(d,wideBuf[3*Stride()],Stride());
        Duplicate(mid,0.0f,len);Duplicate(one,1.0f,len);Barrier();
        for(uint32_t r=0;r<queries;++r){
            uint32_t target=lane+r*team+1,left=0,right=branches;float l=lower,h=upperBound;
            while(true){uint32_t j=(left+right)/2;float value=0.5f*(l+h);
                if(j==target){mid.SetValue(r,value);break;}
                if(target<j){right=j;h=value;}else{left=j;l=value;}
            }
        }
        Barrier();SchurInertiaInplace(mid,one,len);Barrier();
        if(queries){
            Brcb(wideBuf,mat,uint8_t((queries+7)/8),{1,8});Barrier();
            DataCopyPad(gw[lane],wideBuf,DataCopyExtParams{uint16_t(queries),4,0,(team-1)*4,0});Barrier();
        }
        // Save all metadata before rebinding views that may overlap the
        // temporary query matrix. Counts have already been published to GM.
        DataCopy(wideBuf,origD,Stride());DataCopy(wideBuf[Stride()],origE,Stride());
        DataCopy(wideBuf[2*Stride()],beta,Stride());DataCopy(wideBuf[3*Stride()],d,Stride());Barrier();
        mat=oldMat;prod=oldProd;tmp2=oldTmp2;d=oldD;origD=oldOrigD;origE=oldOrigE;beta=oldBeta;
        DataCopy(origD,wideBuf,Stride());DataCopy(origE,wideBuf[Stride()],Stride());
        DataCopy(beta,wideBuf[2*Stride()],Stride());DataCopy(d,wideBuf[3*Stride()],Stride());Barrier();
        SyncAll();DataCopy(prod,gw,branches);Barrier();
        for(uint32_t r=0;r<count;++r){
            uint32_t left=0,right=branches;float l=lower,h=upperBound;
            while(right-left>1 && h-l>tolerance){
                uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(value==l || value==h)break;
                if(prod.template ReinterpretCast<uint32_t>().GetValue(j-1)<=EigenIndex(r)){left=j;l=value;}
                else{right=j;h=value;}
            }
            piv.SetValue(r,l);fac.SetValue(r,h);
        }
        return 1.0f/float(int32_t(branches));
    }
    __aicore__ inline void Bisect(){
        float lower=origD.GetValue(0),upperBound=lower;
        for(uint32_t i=0;i<Order();++i){
            float radius=(i>0?Abs(origE.GetValue(i-1)):0)+(i+1<Order()?Abs(origE.GetValue(i)):0);
            float a=origD.GetValue(i)-radius,b=origD.GetValue(i)+radius;
            lower=lower<a?lower:a;upperBound=upperBound>b?upperBound:b;
        }
        // A common diagonal shift preserves every eigenvector. Use it
        // only when the Gershgorin enclosure stays well away from zero;
        // adding the shift back then cannot cancel a small output eigenvalue.
        // Centered root tolerances and LU pivots depend on the spectral span
        // instead of a numerically irrelevant common offset.
        const float enclosureCenter=0.5f*lower+0.5f*upperBound;
        const float centerMagnitude=Abs(enclosureCenter);
        const bool separatedFromZero=(enclosureCenter>0.0f && lower>0.125f*centerMagnitude) ||
            (enclosureCenter<0.0f && upperBound<-0.125f*centerMagnitude);
        if(separatedFromZero && centerMagnitude<=3.4e38f){
            commonDiagonalShift=enclosureCenter;
            Adds(origD,origD,-enclosureCenter,Stride());Barrier();
            lower-=enclosureCenter;upperBound-=enclosureCenter;
        }
        float bound=Abs(lower)>Abs(upperBound)?Abs(lower):Abs(upperBound);
        spectralBound=bound;
        spectralCenter=0.5f*lower+0.5f*upperBound;
        float tolerance=1e-6f*bound+1e-30f;
        lower-=tolerance;upperBound+=tolerance;
        if(ResidentFamily() && Order()>=9 && smallRowBound>=1e-10f && smallRowBound<=16.0f){SmallGridBisection(lower,upperBound,tolerance);return;}
        if constexpr(IndependentMedium){BisectMultisection(lower,upperBound,tolerance,false,1.0f);return;}
        bool shared=Stride()>=64 && batch%groups==0;
        if(!shared && count>=2 && !ResidentFamily()){
            // A partial batch wave has no shared spectral-grid enclosure.
            // Use ordered Sturm recurrences on this wider interval to avoid
            // cancellation-sensitive parallel congruence counts.
            BisectMultisection(lower,upperBound,tolerance,false,1.0f);return;
        }
        float widthScale=shared?ReferenceCommonGrid(lower,upperBound,tolerance):1.0f;
        if(count>=2 && !ResidentFamily()){if(laneStride==64 && Stride()>=64)BisectVector(lower,upperBound,tolerance,shared,widthScale);else BisectInertia(lower,upperBound,tolerance,shared);return;}
        for(uint32_t r=0;r<count;++r){
            uint32_t index=EigenIndex(r);float lo=shared?piv.GetValue(r):lower,hi=shared?fac.GetValue(r):upperBound;
            for(uint32_t it=0;it<64 && hi-lo>tolerance;++it){
                float mid=0.5f*(lo+hi);if(mid==lo || mid==hi)break;
                float p=origD.GetValue(0)-mid;uint32_t negative=p<=0;
                for(uint32_t i=1;i<Order();++i){
                    if(Abs(p)<1e-30f)p=-1e-30f;
                    float off=origE.GetValue(i-1);p=origD.GetValue(i)-mid-off*off/p;
                    negative+=p<=0;
                }
                if(negative<=index)lo=mid;else hi=mid;
            }
            d.SetValue(index,0.5f*(lo+hi));
        }
    }
    __aicore__ inline float SafePivot(float x){return Abs(x)<1e-20f?(x<0?-1e-20f:1e-20f):x;}
    __aicore__ inline uint32_t BackPanelWidth(){
        return !ResidentFamily() && Stride()>=64 && Stride()%64==0 && laneStride>=16 && (CubeMode==1 || CubeMode==2)?(CubeMode==1?16:64):0;
    }
    __aicore__ inline uint64_t WorkspacePerMatrix(){
        uint64_t size=uint64_t(ResidentFamily()?2*Order()+3:((CubeMode==1 || CubeMode==2)?2*Order()+5:Order()+5))*Stride();
        if(BackPanelWidth()){
            size+=uint64_t(Order())*Stride()+uint64_t((team+1)/2*2)*laneStride*(Stride()+(CubeMode==1?16:64));
            if constexpr(CubeMode==2)size+=uint64_t((Order()-2+63)/64)*6*64*64;
        }
        return size;
    }
    __aicore__ inline void CubeBackHandshake(bool hasRows){
        auto q=hist[uint64_t(3*Order()+5)*Stride()+uint64_t(lane)*laneStride*Stride()];
        auto y=hist[uint64_t(3*Order()+5)*Stride()+uint64_t((team+1)/2*2)*laneStride*Stride()+uint64_t(lane)*laneStride*(CubeMode==1?16:64)];
        Barrier();
        Duplicate(prod,0.0f,laneStride*(CubeMode==1?16:64));Barrier();DataCopy(y,prod,laneStride*(CubeMode==1?16:64));
        if(hasRows){
            if(count<laneStride)Duplicate(mat[count*Stride()],0.0f,(laneStride-count)*Stride());
            Barrier();DataCopy(q,mat,laneStride*Stride());
        }else{
            Barrier();Duplicate(prod,0.0f,laneStride*Stride());Barrier();DataCopy(q,prod,laneStride*Stride());
        }
        // Complete Q/Y payload transfers before the Cube starts the
        // complete backtransform from its invocation-local GM copy.
        Barrier();CrossCoreSetFlag<2,PIPE_MTE3>(10);
        CrossCoreWaitFlag(9);
        if(hasRows){
            DataCopy(mat,q,laneStride*Stride());Barrier();
        }
    }
    __aicore__ __attribute__((noinline)) void PrepareBackPanels(){
        if constexpr(CubeMode==1){
        uint32_t tile=BackPanelWidth();if(!tile)return;
        auto transformed=hist[(2*Order()+5)*Stride()];
        for(uint32_t low=lane*tile;low<Order()-2;low+=team*tile){
            uint32_t number=Order()-2-low;number=number<tile?number:tile;
            uint32_t height=(number+7)/8*8,dotTile=Stride()/64;
            auto gramMatrix=pack,partial=pack[Stride()>256?Stride():256];
            DataCopy(mat,hist[low*Stride()],number*Stride());
            Duplicate(gramMatrix,0.0f,number*height);Barrier();SetVectorMask<float>(64);
            for(uint32_t block=0;block<number;block+=dotTile){
                uint32_t cols=number-block<dotTile?number-block:dotTile,total=cols*height;
                Duplicate(prod,0.0f,total*64);PipeBarrier<PIPE_V>();
                for(uint32_t col=(low+1)/64*64;col<Stride();col+=64){
                    for(uint32_t j=0;j<cols;++j)
                        Mul<float,false>(prod[j*height*64],mat[col],mat[(block+j)*Stride()+col],64,uint8_t(number),BinaryRepeatParams{1,1,1,8,uint8_t(Stride()/8),0});
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum<float,false>(partial,prod,64,uint8_t(total),1,1,8);PipeBarrier<PIPE_V>();
                    Add(gramMatrix[block*height],gramMatrix[block*height],partial,total);PipeBarrier<PIPE_V>();
                }
            }
            VectorToScalar();
            for(int j=int(number)-1;j>=0;--j){
                auto z=prod[uint32_t(j)*Stride()];
                DataCopy(z,mat[uint32_t(j)*Stride()],Stride());Barrier();
                for(int i=int(number)-1;i>j;--i){
                    float gram=gramMatrix.GetValue(uint32_t(j)*height+uint32_t(i));
                    Axpy(z,prod[uint32_t(i)*Stride()],-gram,Stride());PipeBarrier<PIPE_V>();
                }
                Muls(z,z,beta.GetValue(low+uint32_t(j)),Stride());Barrier();
            }
            Muls(prod,prod,-1.0f,number*Stride());PipeBarrier<PIPE_V>();
            uint32_t stored=number;
            if(low+number==Order()-2){Duplicate(prod[number*Stride()],0.0f,2*Stride());stored+=2;}
            Barrier();DataCopy(transformed[low*Stride()],prod,stored*Stride());Barrier();
        }

        }else{
        uint32_t tile=64,panels=(Order()-2+63)/64,parts=team/panels;
        uint32_t panel=lane/parts,part=lane%parts;if(panel>=panels)return;
        uint32_t low=panel*64,number=Order()-2-low;number=number<64?number:64;
        auto gamma=hist[uint64_t(3*Order()+5)*Stride()+uint64_t((team+1)/2*2)*laneStride*(Stride()+64)+uint64_t(panel)*6*64*64];
        // The Gram MMA writes its actual number-by-number rectangle only.
        // Copy that rectangle with explicit zero padding into a zeroed tile.
        uint32_t gramPitch=(number+7)/8*8;
        Duplicate(mat,0.0f,64*64);Barrier();
        DataCopyPad(mat,gamma,DataCopyExtParams{uint16_t(number),number*4,(64-number)*4,(64-gramPitch)/8,0},DataCopyPadExtParams<float>{true,0,uint8_t(gramPitch-number),0});Barrier();
        for(uint32_t row=part;row<64;row+=parts){
            // Only the selected strict upper triangle contributes to R.
            // Padding rows are explicitly zero and the identity is complete.
            if(row<number){Mul(prod,mat[row*64],beta[low],64);PipeBarrier<PIPE_V>();Duplicate(prod,0.0f,row+1);}
            else Duplicate(prod,0.0f,64);
            PipeBarrier<PIPE_V>();
            float scale=row<number?beta.GetValue(low+row):0.0f;
            Muls(w,prod,scale,64);Duplicate(tmp,0.0f,64);VectorToScalar();
            w.SetValue(row,-scale);tmp.SetValue(row,1.0f);Barrier();
            DataCopy(gamma[3*64*64+row*64],prod,64);
            DataCopy(gamma[64*64+row*64],w,64);
            DataCopy(gamma[5*64*64+row*64],tmp,64);Barrier();
        }
        if(lane==0){Duplicate(prod,0.0f,2*Stride());Barrier();DataCopy(hist[(2*Order()+5)*Stride()+(Order()-2)*Stride()],prod,2*Stride());Barrier();}
        }

    }
    __aicore__ inline void ReadReflectors(LocalTensor<float> destination,uint32_t offset,uint32_t length){
        if constexpr(IndependentMedium && PrivateSmall){
            if(mediumVectorReflectors && Stride()==64){
                // Current-call history is private UB, so keep this transfer
                // on the vector pipeline used by its immediate consumers.
                SetMaskNorm();SetVectorMask<float>(64);
                Copy<float,false>(destination,privateHistory[offset],64,uint8_t(length/64),CopyRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();return;
            }
        }
        if constexpr(PrivateSmall)DataCopy(destination,privateHistory[offset],length);
        else DataCopy(destination,hist[offset],length);
    }
    __aicore__ inline void Backtransform(){
        if(BackPanelWidth()){CubeBackHandshake(true);return;}
        if(Order()<3)return;PackMatrix();
        // Retired solve storage holds two reflector tiles. Each tile comes
        // from this invocation's reduction; every reflector is applied in
        // exactly the original descending order and precision.
        uint32_t tile=laneStride/2;tile=tile<16?tile:16;
        if constexpr(IndependentMedium && PrivateSmall){
            // This arena reserves a complete padded matrix. Two sixteen-row
            // reflector tiles fit without increasing any buffer allocation.
            if(mediumVectorReflectors && Stride()==64 && 32<=(Order()+7)/8*8)tile=16;
        }

        uint32_t first=(Order()-3)/tile*tile,slot=0;
        ReadReflectors(pcrBand,first*Stride(),(Order()-2-first)*Stride());Barrier();
        for(int block=int(first);block>=0;block-=int(tile)){
            if(block>=int(tile))ReadReflectors(pcrBand[(1-slot)*tile*Stride()],(block-tile)*Stride(),tile*Stride());
            uint32_t last=uint32_t(block)+tile-1;last=last<Order()-2?last:Order()-3;
            for(int k=int(last);k>=block;--k){
                auto current=pcrBand[(slot*tile+uint32_t(k-block))*Stride()];
            float b=beta.GetValue(k);
            if(b!=0){
                DotRows(current,k+1);
                if(packedMatrix){
                    uint32_t padded=(count+7)/8*8,first=(k+1)/8*8;
                    Brcb(pcrLeft,w,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
                    Muls(pcrLeft,pcrLeft,-b,padded*8);PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
                    for(uint32_t rb=0;rb<padded/8;++rb)
                        MulAddDst<float,float,false>(mat[rb*8*Stride()+first*8],current[first],pcrLeft[rb*64],64,uint8_t((Stride()-first)/8),BinaryRepeatParams{1,0,1,8,1,0});
                }else if(Stride()%64==0){
                    Brcb(pack,w,uint8_t((count+7)/8),{1,8});PipeBarrier<PIPE_V>();Muls(pack,pack,-b,8*((count+7)/8*8));PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
                    for(uint32_t col=(k+1)/64*64;col<Stride();col+=64)
                        MulAddDst<float,float,false>(mat[col],current[col],pack,64,uint8_t(count),BinaryRepeatParams{1,1,0,uint8_t(Stride()/8),0,1});
                }else{
                    const uint32_t paddedRows=(count+7)/8*8;
                    Brcb(pack,w,uint8_t(paddedRows/8),{1,8});PipeBarrier<PIPE_V>();Muls(pack,pack,-b,8*paddedRows);PipeBarrier<PIPE_V>();
                    MulAddDst(mat,current,pack,Stride(),uint8_t(count),BinaryRepeatParams{1,1,0,uint8_t(Stride()/8),0,1});PipeBarrier<PIPE_V>();
                }
            }
            if constexpr(IndependentMedium && PrivateSmall){
                // Beta is immutable, reflector copies and every Q update
                // are now vector operations. No DMA or scalar UB writer
                // crosses this resident-history iteration.
                if(mediumVectorReflectors && Stride()==64)PipeBarrier<PIPE_V>();
                else if(Stride()<64)PipeBarrier<PIPE_V>();else Barrier();
            }else{if(Stride()<64)PipeBarrier<PIPE_V>();else Barrier();}
            }
            slot=1-slot;
        }
        Barrier(); // Output may initialize scalar indices in the retired vector scratch.
        UnpackMatrix();
    }
    __aicore__ inline void FactorVector(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        auto a=pack[5*Stride()];
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();Mul(v,origE,origE,Stride());PipeBarrier<PIPE_V>();Muls(v,v,-1.0f,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        Adds<float,false>(a,mat,1e-20f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(mat,one,a,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        for(uint32_t i=1;i<Order();++i){
            MulAddDst<float,float,false>(mat[i*len],mat[(i-1)*len],wideBuf[(i-1)*8],len,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            Adds<float,false>(a,mat[i*len],1e-20f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(mat[i*len],one,a,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
        // Reuse factors within this solve. The next factorization overwrites
        // this table; every kernel invocation computes it from its own input.
        Muls(v,origE,-1.0f,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Mul<float,false>(pcrBand[row*len],mat[row*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();
    }
    // Pair disjoint contiguous halves: no strided gather and no overlapping
    // repeat addresses. An odd middle row remains in the next active prefix.
    __aicore__ inline void SumEigenRows(uint32_t len){
        for(uint32_t activeRows=Order();activeRows>1;activeRows=(activeRows+1)/2){
            uint32_t half=activeRows/2,next=(activeRows+1)/2;
            Add(mat,mat,mat[next*len],half*len);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
    }
    __aicore__ inline void RayleighVector(LocalTensor<float> shifts,uint32_t len){
        auto norm=pack[3*Stride()];
        Adds(v,origD,-spectralCenter,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Mul<float,false>(mat[row*len],prod[row*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=1;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            MulAddDst<float,float,false>(mat[row*len],prod[(row-1)*len],wideBuf[(row-1)*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t row=0;row+1<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-1-row<255?Order()-1-row:255);
            MulAddDst<float,float,false>(mat[row*len],prod[(row+1)*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();Mul(mat,mat,prod,Order()*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
        Adds<float,false>(shifts,mat,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Mul(mat,prod,prod,Order()*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
        Adds<float,false>(norm,mat,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(shifts,shifts,norm,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(shifts,shifts,spectralCenter,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    }
    template<uint32_t Op>
    __aicore__ inline void PcrOp(LocalTensor<float> dst,uint32_t ds,LocalTensor<float> a,uint32_t as,LocalTensor<float> b,uint32_t bs,uint32_t len,uint32_t number){
        bool batchable=ds/8<=255 && as/8<=255 && bs/8<=255;
        for(uint32_t first=0;first<number;){
            uint8_t repeats=batchable?uint8_t(number-first<255?number-first:255):1;
            BinaryRepeatParams br{1,1,1,uint8_t(batchable?ds/8:8),uint8_t(batchable?as/8:8),uint8_t(batchable?bs/8:8)};
            if constexpr(Op==0)Mul<float,false>(dst[first*ds],a[first*as],b[first*bs],len,repeats,br);
            if constexpr(Op==1)Div<float,false>(dst[first*ds],a[first*as],b[first*bs],len,repeats,br);
            if constexpr(Op==2)Sub<float,false>(dst[first*ds],a[first*as],b[first*bs],len,repeats,br);
            first+=repeats;
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PcrSolve(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t rep=uint8_t(Order()-row<255?Order()-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,rep,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t rep=uint8_t(Order()-row<255?Order()-row:255);
            Adds<float,false>(pcrBand[row*len],wideBuf[row*8],0.0f,len,rep,UnaryRepeatParams{1,0,uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();
        uint32_t step=1;
        for(;step<Order();step*=2){
            uint32_t odd=(Order()-1-step)/(2*step)+1,keep=(Order()-1)/(2*step)+1,dist=2*step*len;
            auto od=mat[step*len],oq=prod[step*len],oe=pcrBand[step*len];
            // Normalize eliminated rows and retain the left factors temporarily.
            if(dist/8<=255){for(uint32_t first=0;first<odd;first+=255){uint8_t rep=uint8_t(odd-first<255?odd-first:255);Adds<float,false>(od[first*dist],od[first*dist],1e-20f,len,rep,UnaryRepeatParams{1,1,uint8_t(dist/8),uint8_t(dist/8)});}}
            else for(uint32_t j=0;j<odd;++j)Adds<float,false>(od[j*dist],od[j*dist],1e-20f,len,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            PcrOp<1>(pcrLeft,len,one,0,od,dist,len,odd);
            PcrOp<0>(oe,dist,oe,dist,pcrLeft,len,len,odd);
            PcrOp<0>(oq,dist,oq,dist,pcrLeft,len,len,odd);
            PcrOp<0>(pcrLeft,len,pcrBand,dist,pcrLeft,len,len,odd);
            // Right-neighbor corrections use the original left off-diagonal.
            PcrOp<0>(wideBuf,len,pcrBand,dist,pcrLeft,len,len,odd);
            PcrOp<2>(mat,dist,mat,dist,wideBuf,len,len,odd);
            PcrOp<0>(wideBuf,len,pcrBand,dist,oq,dist,len,odd);
            PcrOp<2>(prod,dist,prod,dist,wideBuf,len,len,odd);
            // The original right off-diagonal is D_odd times its normalized value.
            PcrOp<0>(wideBuf,len,od,dist,oe,dist,len,keep-1);
            PcrOp<0>(wideBuf,len,wideBuf,len,oe,dist,len,keep-1);
            PcrOp<2>(mat[dist],dist,mat[dist],dist,wideBuf,len,len,keep-1);
            PcrOp<0>(wideBuf,len,od,dist,oe,dist,len,keep-1);
            PcrOp<0>(wideBuf,len,wideBuf,len,oq,dist,len,keep-1);
            PcrOp<2>(prod[dist],dist,prod[dist],dist,wideBuf,len,len,keep-1);
            // Connect the surviving rows and save both interpolation factors.
            PcrOp<0>(wideBuf,len,pcrBand,dist,oe,dist,len,odd);
            Muls(wideBuf,wideBuf,-1.0f,odd*len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            PcrOp<0>(pcrBand,dist,wideBuf,len,one,0,len,odd);
            PcrOp<0>(od,dist,pcrLeft,len,one,0,len,odd);
        }
        auto scratch=pack[5*Stride()];
        Adds<float,false>(scratch,mat,1e-20f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(prod,prod,scratch,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        for(step/=2;step;step/=2){
            uint32_t odd=(Order()-1-step)/(2*step)+1,right=(Order()-1)/(2*step),dist=2*step*len;
            auto od=mat[step*len],oq=prod[step*len],oe=pcrBand[step*len];
            PcrOp<0>(wideBuf,len,od,dist,prod,dist,len,odd);
            PcrOp<2>(oq,dist,oq,dist,wideBuf,len,len,odd);
            if(right){PcrOp<0>(wideBuf,len,oe,dist,prod[dist],dist,len,right);PcrOp<2>(oq,dist,oq,dist,wideBuf,len,len,right);}
        }
    }
    __aicore__ inline void MaximumEigenRows(LocalTensor<float> mx,uint32_t len){
        AscendC::Abs(mat,prod,Order()*len);PipeBarrier<PIPE_V>();
        for(uint32_t number=Order();number>1;number=(number+1)/2){uint32_t half=number/2,next=(number+1)/2;Max(mat,mat,mat[next*len],half*len);PipeBarrier<PIPE_V>();}
        SetVectorMask<float>(len);Adds<float,false>(mx,mat,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void MaximumEigenRowsPreservingFactors(LocalTensor<float> maximum,uint32_t columnCount){
        // Fixed-shift inverse iteration can reuse D/L factors within this
        // invocation. The ordinary maximum reduction overwrites their mat
        // arena; use retired PCR half-matrix scratch instead.
        const uint32_t halfRowCount=Order()/2;
        const uint32_t halfElementCount=halfRowCount*columnCount;
        const uint32_t scratchElementCount=privateCapacity/8*8;
        AscendC::Abs(pcrLeft,prod,halfElementCount);PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<halfElementCount;first+=scratchElementCount){
            const uint32_t live=halfElementCount-first<scratchElementCount?
                halfElementCount-first:scratchElementCount;
            AscendC::Abs(wideBuf,prod[halfElementCount+first],live);PipeBarrier<PIPE_V>();
            Max(pcrLeft[first],pcrLeft[first],wideBuf,live);PipeBarrier<PIPE_V>();
        }
        for(uint32_t rowCount=halfRowCount;rowCount>1;rowCount=(rowCount+1)/2){
            const uint32_t half=rowCount/2,next=(rowCount+1)/2;
            Max(pcrLeft,pcrLeft,pcrLeft[next*columnCount],half*columnCount);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(columnCount);
        Adds<float,false>(maximum,pcrLeft,0.0f,columnCount,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        if(Order()%2){
            AscendC::Abs(wideBuf,prod[(Order()-1)*columnCount],columnCount);PipeBarrier<PIPE_V>();
            Max(maximum,maximum,wideBuf,columnCount);PipeBarrier<PIPE_V>();
        }
        // wideBuf was temporary reduction scratch. Restore the complete
        // unchanged upper coefficient broadcasts for the next local solve.
        Muls(v,origE,-1.0f,Stride());PipeBarrier<PIPE_V>();
        Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
        SetVectorMask<float>(columnCount);
    }
    // Evaluate the actual tridiagonal Rayleigh quotient for scalar solves.
    __aicore__ inline float ReferenceRayleigh(LocalTensor<float> q){
        auto left=pcrLeft[Stride()].template ReinterpretCast<uint32_t>(),right=pcrLeft[2*Stride()].template ReinterpretCast<uint32_t>();
        Mul(w,q,origD,Stride());PipeBarrier<PIPE_V>();
        Duplicate(tmp2,0.0f,Stride());Duplicate(tmp,0.0f,Stride());PipeBarrier<PIPE_V>();
        Gather(tmp2,q,left,0,Order());Gather(tmp,origE,left,0,Order());PipeBarrier<PIPE_V>();
        Mul(tmp2,tmp2,tmp,Stride());PipeBarrier<PIPE_V>();Duplicate(tmp2,0.0f,1);PipeBarrier<PIPE_V>();
        Add(w,w,tmp2,Stride());PipeBarrier<PIPE_V>();
        Duplicate(tmp2,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(tmp2,q,right,0,Order());PipeBarrier<PIPE_V>();
        Mul(tmp2,tmp2,origE,Stride());PipeBarrier<PIPE_V>();Add(w,w,tmp2,Order()-1);PipeBarrier<PIPE_V>();
        return Dot(q,w);
    }
    __aicore__ inline void FlagUnstableInverseColumns(LocalTensor<float> shifts,uint32_t len){

        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Mul<float,false>(mat[row*len],prod[row*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});}
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=1;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            MulAddDst<float,float,false>(mat[row*len],prod[(row-1)*len],wideBuf[(row-1)*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});}
        PipeBarrier<PIPE_V>();
        for(uint32_t row=0;row+1<Order();row+=255){uint8_t repeats=uint8_t(Order()-1-row<255?Order()-1-row:255);
            MulAddDst<float,float,false>(mat[row*len],prod[(row+1)*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});}
        PipeBarrier<PIPE_V>();
        for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Mul<float,false>(pcrBand[row*len],prod[row*len],shifts,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
        PipeBarrier<PIPE_V>();Sub(mat,mat,pcrBand,Order()*len);PipeBarrier<PIPE_V>();Mul(mat,mat,mat,Order()*len);PipeBarrier<PIPE_V>();
        SumEigenRows(len);VectorToScalar();
        // A small residual also protects eigenvectors far from close pairs
        // against unstable float LU pivots. Flagged columns are recomputed.
        // A per-column share of 20% of the reconstruction tolerance gives
        // ||TQ-Q*Lambda||_F <= 2e-5*||T||_F. The Gram certificate and
        // reflector rounding retain the remaining error budget.
        Mul(v,origD,origD,Stride());Mul(tmp2,origE,origE,Stride());PipeBarrier<PIPE_V>();
        Axpy(v,tmp2,2.0f,Stride());PipeBarrier<PIPE_V>();
        ReduceSumWithArena(red,v,tmp,Stride());VectorToScalar();
        tridiagonalFrobeniusSquared=red.GetValue(0);
        const float residualSquaredBudget=4e-10f*tridiagonalFrobeniusSquared/float(int32_t(Order()));
        ownedResidualSquared=0.0f;
        for(uint32_t owned=0;owned<count;++owned){
            const float squaredResidual=mat.GetValue(owned);
            ownedResidualSquared+=squaredResidual;
            if(!(squaredResidual>=0.0f) || squaredResidual>residualSquaredBudget)
                extendedEigenvectorMask|=uint64_t(1)<<owned;
        }
        Barrier();
    }
    __aicore__ inline void AwaitCubeGram(){
        // Every launched AIV, including empty padding groups, notifies its
        // paired AIC. AIC barriers join the independently owned matrices.
        CrossCoreSetFlag<2,PIPE_MTE3>(8);
        CrossCoreWaitFlag(9);
    }
    __aicore__ inline void RefillVectorEigenvectors(){
        uint32_t len=laneStride;SetVectorMask<float>(len);
        auto shifts=pack,one=pack[Stride()],mx=pack[2*Stride()],norm=pack[3*Stride()];
        auto inv=pack[4*Stride()],a=pack[5*Stride()],b=pack[6*Stride()];
        Duplicate<float,false>(shifts,0.0f,len,1,1,8);Duplicate<float,false>(one,1.0f,len,1,1,8);Barrier();
        for(uint32_t r=0;r<len;++r)shifts.SetValue(r,param.GetValue(EigenIndex(r)<Order()?EigenIndex(r):Order()-1));
        // Generate the same invocation-local hash start vectors with integer
        // SIMD operations. No input-independent tensor is supplied by the host.
        auto rowHash=v.template ReinterpretCast<int32_t>(),colHash=pack[7*Stride()].template ReinterpretCast<int32_t>();
        CreateIndicesWithArena(rowHash,int32_t(1),Stride());PipeBarrier<PIPE_V>();Muls(rowHash,rowHash,int32_t(747796405),Stride());Barrier();
        for(uint32_t r=0;r<len;++r)colHash.SetValue(r,int32_t((EigenIndex(r)+1)*2891336453u));Barrier();
        Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Add<int32_t,false>(prod[row*len].template ReinterpretCast<int32_t>(),wideBuf[row*8].template ReinterpretCast<int32_t>(),colHash,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<Order()*len;first+=4*Stride()){
            uint32_t k=Order()*len-first<4*Stride()?Order()*len-first:4*Stride();
            auto h=prod[first].template ReinterpretCast<uint32_t>(),shifted=mat[first].template ReinterpretCast<uint32_t>();
            auto h16=h.template ReinterpretCast<uint16_t>(),s16=shifted.template ReinterpretCast<uint16_t>(),t16=wideBuf.template ReinterpretCast<uint16_t>();
            ShiftRight(shifted,h,uint32_t(16),k);PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*k);PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*k);PipeBarrier<PIPE_V>();Not(t16,t16,2*k);PipeBarrier<PIPE_V>();And(h16,h16,t16,2*k);PipeBarrier<PIPE_V>();
            Muls(h.template ReinterpretCast<int32_t>(),h.template ReinterpretCast<int32_t>(),int32_t(2246822519u),k);PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(13),k);PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*k);PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*k);PipeBarrier<PIPE_V>();Not(t16,t16,2*k);PipeBarrier<PIPE_V>();And(h16,h16,t16,2*k);PipeBarrier<PIPE_V>();
            ShiftLeft(h,h,uint32_t(16),k);PipeBarrier<PIPE_V>();ShiftRight(h,h,uint32_t(16),k);PipeBarrier<PIPE_V>();
            Cast(prod[first],h.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,k);PipeBarrier<PIPE_V>();
            Muls(prod[first],prod[first],1.0f/65536.0f,k);PipeBarrier<PIPE_V>();Adds(prod[first],prod[first],-0.5f,k);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
        Barrier();
        for(uint32_t it=0;it<5;++it){
            // All five solves use this invocation's unchanged indexed shifts.
            // Form D/L once; preserve them across intermediate normalization.
            if(it==1)FactorVector(shifts,one,len);
            if(it<1){PcrSolve(shifts,one,len);}else{
            for(uint32_t i=1;i<Order();++i){
                MulAddDst<float,float,false>(prod[i*laneStride],pcrBand[(i-1)*laneStride],prod[(i-1)*laneStride],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            Duplicate<float,false>(mx,0.0f,len,1,1,8);PipeBarrier<PIPE_V>();
            for(int i=int(Order())-1;i>=0;--i){
                auto q=prod[i*laneStride];
                if(i+1<int(Order())){MulAddDst<float,float,false>(q,prod[(i+1)*laneStride],wideBuf[i*8],len,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();}
                Mul<float,false>(q,q,mat[i*laneStride],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            }
            if(it>=1 && it<4)MaximumEigenRowsPreservingFactors(mx,len);
            else MaximumEigenRows(mx,len);
            Maxs<float,false>(mx,mx,1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(inv,one,mx,len,1,BinaryRepeatParams{1,1,1,8,8,8});Duplicate<float,false>(norm,0.0f,len,1,1,8);PipeBarrier<PIPE_V>();
            if(it<4){
                for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
                PipeBarrier<PIPE_V>();continue;
            }
            for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();Mul(mat,prod,prod,Order()*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
            Sqrt<float,false>(norm,mat,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(inv,one,norm,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();
        }
        RayleighVector(shifts,len);Barrier();
        extendedEigenvectorMask=0;
        FlagUnstableInverseColumns(shifts,len);
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),shifts.GetValue(r));
        auto offsets=piv.template ReinterpretCast<uint32_t>();
        CreateIndicesWithArena(offsets.template ReinterpretCast<int32_t>(),int32_t(0),Stride());PipeBarrier<PIPE_V>();
        Muls(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),int32_t(laneStride*4),Stride());Barrier();
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];
            Duplicate(q,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(q,prod,offsets,r*4,Order());
        }
        Barrier();


    }
    __aicore__ inline float OwnedGramError(){
        const uint64_t matrixElements=uint64_t(Order())*Stride();
        const uint64_t matrixOffset=uint64_t(currentBatch)*matrixElements;
        const uint64_t allMatrixElements=uint64_t(batch)*matrixElements;
        const uint32_t vectorBlocks=Stride()/64;
        // Transfer all cyclically owned Gram rows into a zeroed dense tile.
        // Padding beyond the MMA tail is never read from global memory.
        Duplicate(prod,0.0f,count*Stride());Barrier();
        DataCopyPad(prod,gramScratch[allMatrixElements+matrixOffset+uint64_t(lane)*Stride()],
            DataCopyExtParams{uint16_t(count),Order()*4,(team*Stride()-Order())*4,
                             (Stride()-(Order()+7)/8*8)/8,0},
            DataCopyPadExtParams<float>{true,0,uint8_t((8-Order()%8)%8),0});Barrier();
        for(uint32_t owned=0;owned<count;++owned){
            const uint32_t diagonal=owned*Stride()+EigenIndex(owned);
            prod.SetValue(diagonal,prod.GetValue(diagonal)-1.0f);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Mul(prod,prod,prod,count*Stride());PipeBarrier<PIPE_V>();
        const uint32_t rowsPerBatch=vectorBlocks<=7?32:24;
        for(uint32_t firstRow=0;firstRow<count;firstRow+=rowsPerBatch){
            const uint32_t liveRows=count-firstRow<rowsPerBatch?count-firstRow:rowsPerBatch;
            Duplicate(wideBuf,0.0f,liveRows*vectorBlocks*8);PipeBarrier<PIPE_V>();
            WholeReduceSum(wideBuf,prod[firstRow*Stride()],64,uint8_t(liveRows*vectorBlocks),8,1,8);PipeBarrier<PIPE_V>();
            WholeReduceSum(w[firstRow],wideBuf,8*vectorBlocks,uint8_t(liveRows),1,1,uint8_t(vectorBlocks));PipeBarrier<PIPE_V>();
        }
        ReduceSumWithArena(red,w,tmp,count);VectorToScalar();
        const float localSquaredGramError=red.GetValue(0);
        return localSquaredGramError;
    }
    __aicore__ inline void StoreGramMetrics(float localSquaredGramError){
        red.SetValue(0,localSquaredGramError);fac.SetValue(0,ownedResidualSquared);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(gw[lane],red,DataCopyExtParams{1,4,0,0,0});
        DataCopyPad(gw[Stride()+lane],fac,DataCopyExtParams{1,4,0,0,0});Barrier();
    }
    __aicore__ inline bool MatrixGramRepairable(float &squaredGramError){
        Duplicate(v,0.0f,Stride());Duplicate(tmp2,0.0f,Stride());Barrier();
        DataCopyPad(v,gw,DataCopyExtParams{1,team*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t((8-team%8)%8),0});
        DataCopyPad(tmp2,gw[Stride()],DataCopyExtParams{1,team*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t((8-team%8)%8),0});Barrier();
        ReduceSumWithArena(red,v,tmp,team);PipeBarrier<PIPE_V>();
        ReduceSumWithArena(fac,tmp2,tmp,team);VectorToScalar();
        squaredGramError=red.GetValue(0);const float squaredResidual=fac.GetValue(0);
        // The complete current Rayleigh spectrum is published before the
        // Gram barrier. Full rank from the Gram check, the complete residual
        // bound and strict ordering certify a sorted spectral decomposition.
        // Acceptance does not depend on the initial bisection error.
        Duplicate(pack,0.0f,Stride());Duplicate(v,0.0f,Stride());Barrier();
        DataCopyPad(pack,gw[3*Stride()],DataCopyExtParams{1,Order()*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t((8-Order()%8)%8),0});
        DataCopyPad(v,gw[3*Stride()+1],DataCopyExtParams{1,(Order()-1)*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t((9-Order()%8)%8),0});Barrier();
        Sub(tmp2,v,pack,Stride());PipeBarrier<PIPE_V>();
        // Direct UB entries have no TPipe for high-level ReduceMin scratch
        // events. Reduce aligned SIMD chunks with the basic instruction.
        float minimumRayleighGap=3.4e38f;
        for(uint32_t first=0;first+1<Order();first+=64){
            const uint32_t live=Order()-1-first<64?Order()-1-first:64;
            WholeReduceMin(fac,tmp2[first],live,1,1,1,8);VectorToScalar();
            const float chunkMinimum=fac.GetValue(0);
            if(!(chunkMinimum>0.0f))minimumRayleighGap=0.0f;
            else if(chunkMinimum<minimumRayleighGap)minimumRayleighGap=chunkMinimum;
        }
        // For E=Q^TQ-I, one polar step leaves
        // ||E_new||F <= .75*delta^2+.25*delta^3. Delta=.0075 gives
        // <4.23e-5. The wider Gram range uses half the residual norm
        // budget; symmetry bounds the corrected residual by (2+1.5*delta)R.
        // Their reconstruction bound is <6.25e-5 before reflector rounding.
        const float squaredResidualFraction=squaredGramError<=2.5e-5f?4e-10f:1e-10f;
        const bool repairable=squaredGramError>=0.0f && squaredGramError<=5.625e-5f &&
            squaredResidual>=0.0f && squaredResidual<=squaredResidualFraction*tridiagonalFrobeniusSquared &&
            minimumRayleighGap>0.0f;
        return repairable;
    }
    __aicore__ inline void ApplyCubePolarCorrection(){
        const uint64_t elements=uint64_t(Order())*Stride();
        for(uint32_t owned=0;owned<count;++owned){
            auto q=mat[owned*Stride()];
            Barrier(); // Finish V reads before reusing prod for MTE2.
            DataCopy(prod,gramScratch[2*uint64_t(batch)*elements+uint64_t(currentBatch)*elements+uint64_t(EigenIndex(owned))*Stride()],Stride());Barrier();
            Muls(q,q,1.5f,Stride());PipeBarrier<PIPE_V>();Axpy(q,prod,-0.5f,Stride());PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void PublishRayleighSpectrum(){
        // The fourth metric row is retired reduction scratch. Every current
        // Rayleigh value has one producer; the existing barrier joins them.
        Barrier();
        for(uint32_t owned=0;owned<count;++owned)
            pack.SetValue(8*owned,d.GetValue(EigenIndex(owned)));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(gw[3*Stride()+lane],pack,
            DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});
    }
    __aicore__ inline void CertifyCubeMatrixAndRepair(){
        const uint64_t matrixOffset=uint64_t(currentBatch)*Order()*Stride();
        PublishRayleighSpectrum();
        for(uint32_t owned=0;owned<count;++owned)
            DataCopy(gramScratch[matrixOffset+uint64_t(EigenIndex(owned))*Stride()],mat[owned*Stride()],Stride());
        Barrier();SyncAll();AwaitCubeGram();
        StoreGramMetrics(OwnedGramError());SyncAll();
        float squaredGramError=0.0f;
        const bool needsRefill=!MatrixGramRepairable(squaredGramError);
        const bool needsPolar=!needsRefill && squaredGramError>3.6e-9f;
        if(needsRefill){
            // Five fixed-shift iterations retain the bisection root index;
            // the following complete Gram/residual check certifies the result.
            RefillVectorEigenvectors();
            PublishRayleighSpectrum();
            for(uint32_t owned=0;owned<count;++owned)
                DataCopy(gramScratch[matrixOffset+uint64_t(EigenIndex(owned))*Stride()],mat[owned*Stride()],Stride());
        }
        if(lane==0){
            // Current-call commands: 0 keeps a certified Q, 1 refills its
            // Gram and correction, 2 forms only the accepted Q correction.
            red.SetValue(0,needsRefill?1.0f:(needsPolar?2.0f:0.0f));
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(gw[2*Stride()],red,DataCopyExtParams{1,4,0,0,0});
        }
        Barrier();SyncAll();
        CrossCoreSetFlag<2,PIPE_MTE3>(10);CrossCoreWaitFlag(9);
        if(needsRefill)StoreGramMetrics(OwnedGramError());
        SyncAll();
        if(needsRefill){
            if(MatrixGramRepairable(squaredGramError)){
                if(squaredGramError>3.6e-9f)ApplyCubePolarCorrection();
            }else{
                for(uint32_t owned=0;owned<count;++owned){
                    const uint32_t index=EigenIndex(owned);
                    d.SetValue(index,PivotedExtendedEigenvector(mat[owned*Stride()],index,param.GetValue(index)));
                }
            }
        }else if(needsPolar)ApplyCubePolarCorrection();
        Barrier();SyncAll();
    }

    __aicore__ inline void CertifyMatrixAndRepair(){
        if constexpr(CubeMode==3){CertifyCubeMatrixAndRepair();return;}
        const uint64_t matrixOffset=uint64_t(currentBatch)*Order()*Stride();
        Barrier();
        for(uint32_t owned=0;owned<count;++owned)
            DataCopy(gramScratch[matrixOffset+uint64_t(EigenIndex(owned))*Stride()],mat[owned*Stride()],Stride());
        Barrier();SyncAll();
        constexpr uint32_t maximumRowsPerChunk=16,maximumVectorBlocks=8;
        constexpr uint32_t accumulatorOffset=maximumRowsPerChunk*maximumVectorBlocks*8;
        const uint32_t halfRowCapacity=laneStride/2;
        const uint32_t chunkCapacity=halfRowCapacity<maximumRowsPerChunk?halfRowCapacity:maximumRowsPerChunk;
        const uint32_t vectorBlocks=Stride()/64;
        auto accumulated=wideBuf[accumulatorOffset];
        Duplicate(accumulated,0.0f,count*maximumRowsPerChunk);
        Duplicate(pcrBand,0.0f,count*Stride());PipeBarrier<PIPE_V>();
        for(uint32_t firstColumn=0;firstColumn<Order();firstColumn+=chunkCapacity){
            const uint32_t liveColumns=Order()-firstColumn<chunkCapacity?Order()-firstColumn:chunkCapacity;
            Barrier();
            DataCopy(prod,gramScratch[matrixOffset+uint64_t(firstColumn)*Stride()],liveColumns*Stride());Barrier();
            for(uint32_t owned=0;owned<count;++owned){
                for(uint32_t column=0;column<Stride();column+=64)
                    Mul(pcrLeft[column],prod[column],mat[owned*Stride()+column],64,uint8_t(liveColumns),
                        BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});
                Duplicate(wideBuf,0.0f,liveColumns*vectorBlocks*8);
                Duplicate(w,0.0f,(liveColumns+7)/8*8);PipeBarrier<PIPE_V>();
                WholeReduceSum(wideBuf,pcrLeft,64,uint8_t(liveColumns*vectorBlocks),8,1,8);PipeBarrier<PIPE_V>();
                WholeReduceSum(w,wideBuf,8*vectorBlocks,uint8_t(liveColumns),1,1,uint8_t(vectorBlocks));PipeBarrier<PIPE_V>();
                const uint32_t index=EigenIndex(owned);
                if(index>=firstColumn && index<firstColumn+liveColumns){
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const uint32_t diagonal=index-firstColumn;
                    w.SetValue(diagonal,w.GetValue(diagonal)-1.0f);
                    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
                }
                // Save the squared dot products before recycling product and
                // broadcast scratch for the polar correction.
                Mul(v,w,w,liveColumns);PipeBarrier<PIPE_V>();
                Add(accumulated[owned*maximumRowsPerChunk],accumulated[owned*maximumRowsPerChunk],v,liveColumns);PipeBarrier<PIPE_V>();
                // Preserve the complete Gram correction, C_i=sum_j E_ij*q_j.
                // pcrLeft is retired product scratch; pcrBand holds C_i.
                Brcb(wideBuf,w,uint8_t((liveColumns+7)/8),{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t column=0;column<Stride();column+=64)
                    Mul(pcrLeft[column],prod[column],wideBuf,64,uint8_t(liveColumns),
                        BinaryRepeatParams{1,1,0,uint8_t(Stride()/8),uint8_t(Stride()/8),1});
                PipeBarrier<PIPE_V>();
                for(uint32_t activeRows=liveColumns;activeRows>1;activeRows=(activeRows+1)/2){
                    const uint32_t half=activeRows/2,next=(activeRows+1)/2;
                    Add(pcrLeft,pcrLeft,pcrLeft[next*Stride()],half*Stride());PipeBarrier<PIPE_V>();
                }
                Add(pcrBand[owned*Stride()],pcrBand[owned*Stride()],pcrLeft,Stride());PipeBarrier<PIPE_V>();

            }
        }
        WholeReduceSum(w,accumulated,maximumRowsPerChunk,uint8_t(count),1,1,2);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for(uint32_t owned=0;owned<count;++owned)pack.SetValue(8*owned,w.GetValue(owned));
        red.SetValue(0,ownedResidualSquared);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(gw[lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});
        DataCopyPad(gw[Stride()+lane],red,DataCopyExtParams{1,4,0,0,0});Barrier();SyncAll();
        // DMA padding covers one 32-byte block. Initialize the remaining
        // SIMD row separately rather than requesting more than seven floats.
        Duplicate(w,0.0f,Stride());Barrier();
        DataCopyPad(w,gw,DataCopyExtParams{1,Order()*4,0,0,0},
                    DataCopyPadExtParams<float>{true,0,uint8_t((8-Order()%8)%8),0});
        DataCopyPad(tmp2,gw[Stride()],DataCopyExtParams{1,team*4,0,0,0},
                    DataCopyPadExtParams<float>{true,0,uint8_t((8-team%8)%8),0});Barrier();
        ReduceSumWithArena(red,w,tmp,Stride());
        PipeBarrier<PIPE_V>();
        ReduceSumWithArena(fac,tmp2,tmp,team);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float squaredGramError=red.GetValue(0);
        const float totalSquaredResidual=fac.GetValue(0);
        const bool failedResidual=!(totalSquaredResidual>=0.0f &&
            totalSquaredResidual<=4e-10f*tridiagonalFrobeniusSquared);
        const bool repairable=!failedResidual && squaredGramError>=0.0f && squaredGramError<=2.5e-5f;
        if(repairable){
            if(squaredGramError>3.6e-9f){
                // One Newton-Schulz polar step: Q'=Q*(I-E/2), E=Q^TQ-I.
                // ||E||_F<=0.005 bounds the new exact Gram error below 1.9e-5.
                Axpy(mat,pcrBand,-0.5f,count*Stride());PipeBarrier<PIPE_V>();
            }
        }else{
            for(uint32_t owned=0;owned<count;++owned){
                const uint32_t index=EigenIndex(owned);
                d.SetValue(index,PivotedExtendedEigenvector(mat[owned*Stride()],index,param.GetValue(index)));
            }
        }
        Barrier();SyncAll();
    }

    __aicore__ inline void OrthogonalizeMediumClusters(uint64_t clusterMask,uint64_t residualMask){
        uint64_t refinementMask=clusterMask|residualMask;
        if constexpr(IndependentMedium)refinementMask=boundaryClusterMask|residualMask;
        // Roots crossing an owner boundary and failed residuals retain the
        // complete two-float solve. Interior clusters can be orthogonalized
        // locally without publishing a matrix or initializing GM scratch.
        for(uint32_t owned=0;owned<count;++owned)if(refinementMask&(uint64_t(1)<<owned)){
            const uint32_t index=EigenIndex(owned);
            d.SetValue(index,PivotedExtendedEigenvector(mat[owned*Stride()],index,param.GetValue(index)));
        }
        if constexpr(IndependentMedium){
            const uint64_t localMask=clusterMask&~refinementMask;
            for(uint32_t owned=0;owned<count;++owned)if(localMask&(uint64_t(1)<<owned)){
                auto vector=mat[owned*Stride()];
                for(uint32_t pass=0;pass<2;++pass){
                    for(uint32_t previous=0;previous<owned;++previous){
                        auto basis=mat[previous*Stride()];
                        const float projection=Dot(vector,basis);
                        Axpy(vector,basis,-projection,Stride());PipeBarrier<PIPE_V>();
                    }
                    const float length=ResidentNorm(vector);
                    if(!(length>1e-6f)){
                        const uint32_t index=EigenIndex(owned);
                        d.SetValue(index,PivotedExtendedEigenvector(vector,index,param.GetValue(index)));
                    }else{Muls(vector,vector,1.0f/length,Stride());PipeBarrier<PIPE_V>();}
                }
            }
        }
    }

    __aicore__ inline float CompleteMediumGramSquared(){
        float squaredError=0.0f;
        for(uint32_t column=0;column<count;++column){
            Mul(prod,mat,mat[column*Stride()],Stride(),uint8_t(count),
                BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});
            Duplicate(w,0.0f,Stride());PipeBarrier<PIPE_V>();
            WholeReduceSum(w,prod,Stride(),uint8_t(count),1,1,uint8_t(Stride()/8));VectorToScalar();
            w.SetValue(column,w.GetValue(column)-1.0f);
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Mul(v,w,w,Stride());PipeBarrier<PIPE_V>();
            ReduceSumWithArena(red,v,tmp,Stride());VectorToScalar();
            squaredError+=red.GetValue(0);
        }
        return squaredError;
    }
    __aicore__ inline bool TryCertifiedMediumClusterQr(uint64_t clusterMask,uint64_t residualMask){
        if(residualMask || count!=Order() || !(ownedResidualSquared>=0.0f) ||
            ownedResidualSquared>4e-10f*tridiagonalFrobeniusSquared)return false;
        const uint32_t dense=Order()*Stride();
        auto coefficients=smallStorage.template Get<float>();
        auto originalRows=coefficients[dense],originalValues=coefficients[2*dense];
        DataCopy(originalRows,mat,dense);DataCopy(originalValues,d,Stride());
        Duplicate(coefficients,0.0f,dense);Barrier();
        for(uint32_t row=0;row<Order();++row)coefficients.SetValue(row*Stride()+row,1.0f);
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        bool valid=true;
        const float neighbourhood=0.002f*spectralBound;
        for(uint32_t row=0;row<count && valid;++row)if(clusterMask&(uint64_t(1)<<row)){
            auto vector=mat[row*Stride()],transform=coefficients[row*Stride()];
            for(uint32_t pass=0;pass<2 && valid;++pass){
                for(uint32_t previous=0;previous<row;++previous){
                    if(Abs(d.GetValue(row)-d.GetValue(previous))>neighbourhood)continue;
                    const float projection=Dot(vector,mat[previous*Stride()]);
                    Axpy(vector,mat[previous*Stride()],-projection,Stride());
                    Axpy(transform,coefficients[previous*Stride()],-projection,Stride());PipeBarrier<PIPE_V>();
                }
                const float length=ResidentNorm(vector);
                if(!(length>1e-6f)){valid=false;break;}
                Muls(vector,vector,1.0f/length,Stride());
                Muls(transform,transform,1.0f/length,Stride());PipeBarrier<PIPE_V>();
            }
        }
        float squaredTransformDeviation=0.0f,squaredCommutator=0.0f;
        if(valid){
            // Preserve an explicit per-invocation transform S: Qnew=Qold*S^T.
            // Its commutator with the current eigenvalues bounds the residual
            // after the nearby-column correction, without assuming far dots vanish.
            auto left=pcrLeft[Stride()].template ReinterpretCast<int32_t>();
            auto right=pcrLeft[2*Stride()].template ReinterpretCast<int32_t>();
            CreateIndicesWithArena(left,int32_t(-1),Stride());
            CreateIndicesWithArena(right,int32_t(1),Stride());PipeBarrier<PIPE_V>();
            Muls(left,left,int32_t(4),Stride());Muls(right,right,int32_t(4),Stride());PipeBarrier<PIPE_V>();
            Maxs(left,left,int32_t(0),Stride());
            Mins(right,right,int32_t((Order()-1)*4),Stride());Barrier();
            for(uint32_t row=0;row<count;++row)
                d.SetValue(row,ReferenceRayleigh(mat[row*Stride()]));
            for(uint32_t row=1;row<count;++row)
                if(!(d.GetValue(row)>d.GetValue(row-1)))valid=false;
            for(uint32_t row=0;row<count && valid;++row){
                auto transform=coefficients[row*Stride()];
                Adds(v,originalValues,-d.GetValue(row),Stride());PipeBarrier<PIPE_V>();
                Mul(v,v,transform,Stride());PipeBarrier<PIPE_V>();Mul(v,v,v,Stride());PipeBarrier<PIPE_V>();
                ReduceSumWithArena(red,v,tmp,Stride());VectorToScalar();squaredCommutator+=red.GetValue(0);
                const float diagonal=transform.GetValue(row);
                transform.SetValue(row,diagonal-1.0f);
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Mul(v,transform,transform,Stride());PipeBarrier<PIPE_V>();
                ReduceSumWithArena(red,v,tmp,Stride());VectorToScalar();
                squaredTransformDeviation+=red.GetValue(0);transform.SetValue(row,diagonal);
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
        }
        // ||S-I||F<=1/8 gives ||S||2<=9/8 and ||S^-1||2<=8/7.
        // Together with the complete Gram and commutator checks, the exact
        // reconstruction budget is below 5.17e-5 before reduction rounding.
        bool accepted=valid && squaredTransformDeviation>=0.0f && squaredTransformDeviation<=0.015625f &&
            squaredCommutator>=0.0f && squaredCommutator<=6.4e-11f*tridiagonalFrobeniusSquared;
        if(accepted){
            const float squaredGramError=CompleteMediumGramSquared();
            accepted=squaredGramError>=0.0f && squaredGramError<=4e-10f;
        }
        if(!accepted){
            DataCopy(mat,originalRows,dense);DataCopy(d,originalValues,Stride());Barrier();
        }
        return accepted;
    }

    __aicore__ inline void RebuildWholeMediumEigensystem(){
        const uint32_t ownedBase=base,ownedCount=count,ownedLaneStride=laneStride;
        base=0;count=Order();laneStride=(Order()+7)/8*8;
        hasClusterGrid=false;possibleCloseMask=0;boundaryClusterMask=0;extendedEigenvectorMask=0;
        // The current tridiagonal and reflector history remain live. A full
        // local solve gives every owner the same ordered orthogonal basis.
        Bisect();Barrier();DetectNearbyRoots();
        DataCopy(param,d,Stride());Barrier();
        const uint64_t completeClusterMask=SolveInverseColumns();
        if(!TryCertifiedMediumClusterQr(completeClusterMask,extendedEigenvectorMask))
            OrthogonalizeMediumClusters(completeClusterMask,extendedEigenvectorMask);
        Backtransform();
        // Keep each core's original, disjoint output interval. Retire the
        // unused rows without changing the arena base for the next matrix.
        if(ownedBase){
            DataCopy(prod,mat[ownedBase*Stride()],ownedCount*Stride());Barrier();
            DataCopy(mat,prod,ownedCount*Stride());Barrier();
        }
        base=ownedBase;count=ownedCount;laneStride=ownedLaneStride;
    }
    __aicore__ inline void RefineOwnedClusters(uint64_t clusterMask,uint64_t residualMask){
        if constexpr(IndependentMedium)OrthogonalizeMediumClusters(clusterMask,residualMask);
        else CertifyMatrixAndRepair();
    }

    __aicore__ inline uint64_t SolveInverseColumns(){
        uint32_t len=laneStride;SetVectorMask<float>(len);
        auto shifts=pack,one=pack[Stride()],mx=pack[2*Stride()],norm=pack[3*Stride()];
        auto inv=pack[4*Stride()],a=pack[5*Stride()],b=pack[6*Stride()];
        Duplicate<float,false>(shifts,0.0f,len,1,1,8);Duplicate<float,false>(one,1.0f,len,1,1,8);Barrier();
        for(uint32_t r=0;r<len;++r)shifts.SetValue(r,d.GetValue(EigenIndex(r)<Order()?EigenIndex(r):Order()-1));
        // Generate the same invocation-local hash start vectors with integer
        // SIMD operations. No input-independent tensor is supplied by the host.
        auto rowHash=v.template ReinterpretCast<int32_t>(),colHash=pack[7*Stride()].template ReinterpretCast<int32_t>();
        CreateIndicesWithArena(rowHash,int32_t(1),Stride());PipeBarrier<PIPE_V>();Muls(rowHash,rowHash,int32_t(747796405),Stride());Barrier();
        for(uint32_t r=0;r<len;++r)colHash.SetValue(r,int32_t((EigenIndex(r)+1)*2891336453u));Barrier();
        Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<Order();row+=255){
            uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);
            Add<int32_t,false>(prod[row*len].template ReinterpretCast<int32_t>(),wideBuf[row*8].template ReinterpretCast<int32_t>(),colHash,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<Order()*len;first+=4*Stride()){
            uint32_t k=Order()*len-first<4*Stride()?Order()*len-first:4*Stride();
            auto h=prod[first].template ReinterpretCast<uint32_t>(),shifted=mat[first].template ReinterpretCast<uint32_t>();
            auto h16=h.template ReinterpretCast<uint16_t>(),s16=shifted.template ReinterpretCast<uint16_t>(),t16=wideBuf.template ReinterpretCast<uint16_t>();
            ShiftRight(shifted,h,uint32_t(16),k);PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*k);PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*k);PipeBarrier<PIPE_V>();Not(t16,t16,2*k);PipeBarrier<PIPE_V>();And(h16,h16,t16,2*k);PipeBarrier<PIPE_V>();
            Muls(h.template ReinterpretCast<int32_t>(),h.template ReinterpretCast<int32_t>(),int32_t(2246822519u),k);PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(13),k);PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*k);PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*k);PipeBarrier<PIPE_V>();Not(t16,t16,2*k);PipeBarrier<PIPE_V>();And(h16,h16,t16,2*k);PipeBarrier<PIPE_V>();
            ShiftLeft(h,h,uint32_t(16),k);PipeBarrier<PIPE_V>();ShiftRight(h,h,uint32_t(16),k);PipeBarrier<PIPE_V>();
            Cast(prod[first],h.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,k);PipeBarrier<PIPE_V>();
            Muls(prod[first],prod[first],1.0f/65536.0f,k);PipeBarrier<PIPE_V>();Adds(prod[first],prod[first],-0.5f,k);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
        Barrier();
        // A shorter attempt retains the original residual/Gram checks and
        // high-precision repair. Whole-spectrum rebuilds keep three solves.
        const uint32_t initialIterationCount=CubeMode==3?2:
            ((IndependentMedium && mediumTwoInverseIterations && count<Order())?2:3);
        for(uint32_t it=0;it<initialIterationCount;++it){
            if(it==1)FactorVector(shifts,one,len);
            if(it==2){
                if constexpr(!IndependentMedium){
                    RayleighVector(shifts,len);FactorVector(shifts,one,len);
                }
                // Medium shifts did not change. Their factors survived the
                // preceding maximum reduction and remain current-call data.
            }
            if(it<1){PcrSolve(shifts,one,len);}else{
            for(uint32_t i=1;i<Order();++i){
                MulAddDst<float,float,false>(prod[i*laneStride],pcrBand[(i-1)*laneStride],prod[(i-1)*laneStride],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            Duplicate<float,false>(mx,0.0f,len,1,1,8);PipeBarrier<PIPE_V>();
            for(int i=int(Order())-1;i>=0;--i){
                auto q=prod[i*laneStride];
                if(i+1<int(Order())){MulAddDst<float,float,false>(q,prod[(i+1)*laneStride],wideBuf[i*8],len,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();}
                Mul<float,false>(q,q,mat[i*laneStride],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            }
            if constexpr(IndependentMedium){
                if(it==1 && it+1<initialIterationCount)MaximumEigenRowsPreservingFactors(mx,len);
                else MaximumEigenRows(mx,len);
            }else MaximumEigenRows(mx,len);
            Maxs<float,false>(mx,mx,1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(inv,one,mx,len,1,BinaryRepeatParams{1,1,1,8,8,8});Duplicate<float,false>(norm,0.0f,len,1,1,8);PipeBarrier<PIPE_V>();
            if(it+1<initialIterationCount){
                for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
                PipeBarrier<PIPE_V>();continue;
            }
            for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();Mul(mat,prod,prod,Order()*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
            Sqrt<float,false>(norm,mat,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(inv,one,norm,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            for(uint32_t row=0;row<Order();row+=255){uint8_t repeats=uint8_t(Order()-row<255?Order()-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();
        }
        RayleighVector(shifts,len);Barrier();
        const uint64_t clusterMask=extendedEigenvectorMask;extendedEigenvectorMask=0;
        FlagUnstableInverseColumns(shifts,len);
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),shifts.GetValue(r));
        auto offsets=piv.template ReinterpretCast<uint32_t>();
        CreateIndicesWithArena(offsets.template ReinterpretCast<int32_t>(),int32_t(0),Stride());PipeBarrier<PIPE_V>();
        Muls(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),int32_t(laneStride*4),Stride());Barrier();
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];
            Duplicate(q,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(q,prod,offsets,r*4,Order());
        }
        Barrier();

        return clusterMask;
    }
    __aicore__ inline uint64_t MediumWorkspaceFloatOffset()const{
        const uint64_t matrixFloatCount=mediumOutputScratch?0:uint64_t(Order())*Stride();
        const uint32_t metricFloatCount=team*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT;
        const uint64_t historyPrefix=mediumSharedReduction?uint64_t(batch)*((Order()+1)*Stride()+16):0;
        return historyPrefix+uint64_t(currentBatch)*(matrixFloatCount+metricFloatCount);
    }
    __aicore__ inline uint64_t MediumMetricFloatOffset()const{
        return MediumWorkspaceFloatOffset()+(mediumOutputScratch?0:uint64_t(Order())*Stride());
    }
    __aicore__ inline void ReadMediumQRows(LocalTensor<float> destination,uint32_t firstRow,uint32_t rowCount){
        if(mediumOutputScratch){
            const uint32_t paddedColumns=(Order()+7)/8*8;
            DataCopyPad(destination,vectors[uint64_t(currentBatch)*Order()*Order()+uint64_t(firstRow)*Order()],
                DataCopyExtParams{uint16_t(rowCount),Order()*4,0,(Stride()-paddedColumns)/8,0},
                DataCopyPadExtParams<float>{true,0,uint8_t((8-Order()%8)%8),0});
        }else DataCopy(destination,ws[MediumWorkspaceFloatOffset()+uint64_t(firstRow)*Stride()],rowCount*Stride());
    }
    __aicore__ inline void PublishMediumCandidate(){
        // Independent owners have contiguous root indices and contiguous UB
        // rows. One matrix transfer publishes the same complete candidate.
        const uint64_t matrixOffset=MediumWorkspaceFloatOffset();
        if(mediumOutputScratch){
            const uint64_t outputOffset=uint64_t(currentBatch)*Order()*Order()+uint64_t(base)*Order();
            if(Order()==Stride())DataCopy(vectors[outputOffset],mat,count*Stride());
            else{
                const uint32_t paddedColumns=(Order()+7)/8*8;
                DataCopyPad(vectors[outputOffset],mat,
                    DataCopyExtParams{uint16_t(count),Order()*4,(Stride()-paddedColumns)/8,0,0});
            }
        }else DataCopy(ws[matrixOffset+uint64_t(base)*Stride()],mat,count*Stride());
        for(uint32_t owned=0;owned<count;++owned)
            red.SetValue(owned,d.GetValue(EigenIndex(owned)));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(values[uint64_t(currentBatch)*Order()+base],red,
            DataCopyExtParams{1,count*4,0,0,0});Barrier();
    }
    __aicore__ inline bool MediumRayleighSpectrumOrdered(){
        Duplicate(w,0.0f,Stride());Duplicate(tmp2,0.0f,Stride());Barrier();
        DataCopyPad(w,values[uint64_t(currentBatch)*Order()],
            DataCopyExtParams{1,Order()*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t((8-Order()%8)%8),0});
        DataCopyPad(tmp2,values[uint64_t(currentBatch)*Order()+1],
            DataCopyExtParams{1,(Order()-1)*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t((9-Order()%8)%8),0});Barrier();
        Sub(w,tmp2,w,Order()-1);PipeBarrier<PIPE_V>();
        WholeReduceMin(red,w,Order()-1,1,1,1,8);VectorToScalar();
        return red.GetValue(0)>0.0f;
    }
    __aicore__ inline float ComputeMediumGramError(){
        constexpr uint32_t pairTile=8;
        auto products=smallStorage.template Get<float>();
        auto gramCoefficients=wideBuf[512],squaredCoefficients=wideBuf[576];
        auto squaredRowErrors=wideBuf[640];
        Duplicate(squaredRowErrors,0.0f,Stride());
        Duplicate(pcrBand,0.0f,count*Stride());PipeBarrier<PIPE_V>();
        // The full fallback arena already reserves these padded UB rows.
        // This current-call snapshot survives through Gram correction only.
        const uint32_t paddedSnapshotRows=(Order()+7)/8*8;
        Duplicate(prod,0.0f,paddedSnapshotRows*Stride());Barrier();
        ReadMediumQRows(prod,0,Order());Barrier();
        for(uint32_t first=0;first<Order();first+=pairTile){
            const uint32_t liveRows=Order()-first<pairTile?Order()-first:pairTile;
            auto publishedRows=prod[first*Stride()];
            for(uint32_t firstOwned=0;firstOwned<count;firstOwned+=pairTile){
                const uint32_t liveOwned=count-firstOwned<pairTile?count-firstOwned:pairTile;
                for(uint32_t owned=0;owned<liveOwned;++owned)
                    Mul(products[owned*pairTile*Stride()],publishedRows,mat[(firstOwned+owned)*Stride()],
                        64,uint8_t(pairTile),BinaryRepeatParams{1,1,1,8,8,0});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(gramCoefficients,products,64,uint8_t(liveOwned*pairTile),1,1,8);PipeBarrier<PIPE_V>();
                uint64_t diagonalMask=0;
                for(uint32_t owned=0;owned<liveOwned;++owned){
                    const uint32_t root=EigenIndex(firstOwned+owned);
                    if(root>=first && root<first+liveRows)
                        diagonalMask|=uint64_t(1)<<(owned*pairTile+root-first);
                }
                if(diagonalMask){
                    // Coordinates are known from ownership; no Gram value
                    // crosses V/S merely to subtract the identity matrix.
                    SetMaskNorm();SetVectorMask<float>(0,diagonalMask);
                    Adds<float,false>(gramCoefficients,gramCoefficients,-1.0f,64,1,UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();ResetMask();
                }
                Mul(squaredCoefficients,gramCoefficients,gramCoefficients,liveOwned*pairTile);PipeBarrier<PIPE_V>();
                WholeReduceSum(w,squaredCoefficients,pairTile,uint8_t(liveOwned),1,1,1);PipeBarrier<PIPE_V>();
                Add(squaredRowErrors[firstOwned],squaredRowErrors[firstOwned],w,liveOwned);
                // Retain every owned Gram row. The correction is formed
                // only after the complete group certificate accepts it.
                SetVectorMask<float>(pairTile);
                Copy<float,false>(pcrBand[firstOwned*Stride()+first],gramCoefficients,pairTile,uint8_t(liveOwned),
                    CopyRepeatParams{1,1,uint8_t(Stride()/8),1});
                PipeBarrier<PIPE_V>();ResetMask();
            }
        }
        ReduceSumWithArena(red,squaredRowErrors,tmp,count);VectorToScalar();return red.GetValue(0);
    }
    __aicore__ inline void ApplyMediumGramCorrection(){
        // prod still holds the immutable snapshot formed by the Gram pass.
        constexpr uint32_t pairTile=8;
        auto products=smallStorage.template Get<float>();
        auto correction=refinementStorage.template Get<float>();
        auto gramCoefficients=wideBuf[512];
        Duplicate(correction,0.0f,count*Stride());PipeBarrier<PIPE_V>();
        const uint64_t matrixOffset=MediumWorkspaceFloatOffset();
        for(uint32_t first=0;first<Order();first+=pairTile){
            const uint32_t liveRows=Order()-first<pairTile?Order()-first:pairTile;
            auto publishedRows=prod[first*Stride()];
            for(uint32_t firstOwned=0;firstOwned<count;firstOwned+=pairTile){
                const uint32_t liveOwned=count-firstOwned<pairTile?count-firstOwned:pairTile;
                SetVectorMask<float>(pairTile);
                Copy<float,false>(gramCoefficients,pcrBand[firstOwned*Stride()+first],pairTile,uint8_t(liveOwned),
                    CopyRepeatParams{1,1,1,uint8_t(Stride()/8)});
                PipeBarrier<PIPE_V>();ResetMask();
                Brcb(wideBuf,gramCoefficients,uint8_t(liveOwned),{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t owned=0;owned<liveOwned;++owned)
                    Mul(products[owned*pairTile*Stride()],publishedRows,wideBuf[owned*pairTile*8],
                        64,uint8_t(pairTile),BinaryRepeatParams{1,1,0,8,8,1});
                PipeBarrier<PIPE_V>();
                for(uint32_t half=pairTile/2;half;half/=2){
                    for(uint32_t row=0;row<half;++row)
                        Add(products[row*Stride()],products[row*Stride()],products[(row+half)*Stride()],
                            64,uint8_t(liveOwned),BinaryRepeatParams{1,1,1,64,64,64});
                    PipeBarrier<PIPE_V>();
                }
                Add(correction[firstOwned*Stride()],correction[firstOwned*Stride()],products,
                    64,uint8_t(liveOwned),BinaryRepeatParams{1,1,1,8,8,64});PipeBarrier<PIPE_V>();
            }
        }
        Axpy(mat,correction,-0.5f,count*Stride());PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool ReadMediumCertificateSummary(float &squaredGramError){
        const uint64_t metricsOffset=MediumMetricFloatOffset();
        DataCopy(wideBuf,ws[metricsOffset],team*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT);Barrier();
        auto certificateOffsets=piv.template ReinterpretCast<int32_t>();
        CreateIndicesWithArena(certificateOffsets,int32_t(0),Stride());PipeBarrier<PIPE_V>();
        Muls(certificateOffsets,certificateOffsets,int32_t(EIGH_CERTIFICATE_SLOT_FLOAT_COUNT*sizeof(float)),Stride());PipeBarrier<PIPE_V>();
        Gather(tmp2,wideBuf,certificateOffsets.template ReinterpretCast<uint32_t>(),0,team);PipeBarrier<PIPE_V>();
        WholeReduceMin(red,tmp2,team,1,1,1,8);
        WholeReduceSum(fac,tmp2,team,1,1,1,8);VectorToScalar();
        squaredGramError=fac.GetValue(0);
        return red.GetValue(0)>=0.0f && squaredGramError>=0.0f;
    }
    __aicore__ inline bool CertifyMediumOutputAndRepair(){
        PublishMediumCandidate();SyncAll();
        const bool ordered=MediumRayleighSpectrumOrdered();
        const float squaredGramError=ComputeMediumGramError();
        const float ownerFraction=float(int32_t(count))/float(int32_t(Order()));
        const bool accepted=ordered && squaredGramError>=0.0f &&
            squaredGramError<=5.625e-5f*ownerFraction && ownedResidualSquared>=0.0f &&
            ownedResidualSquared<=1e-10f*tridiagonalFrobeniusSquared*ownerFraction;
        // Boundary readers never access these error slots, so this
        // publication needs no additional boundary-consumer rendezvous.
        Duplicate(red,0.0f,EIGH_CERTIFICATE_SLOT_FLOAT_COUNT);VectorToScalar();
        red.SetValue(0,accepted?squaredGramError:-1.0f);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(ws[MediumMetricFloatOffset()+uint64_t(lane)*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT],red,
            EIGH_CERTIFICATE_SLOT_FLOAT_COUNT);Barrier();SyncAll();
        float completeSquaredGramError=0.0f;
        if(!ReadMediumCertificateSummary(completeSquaredGramError)){
            // A rejected complete certificate retains the established full
            // eigensystem fallback, including its original-input refinement.
            RebuildWholeMediumEigensystem();return true;
        }else if(completeSquaredGramError>3.6e-9f){
            // Every Gram entry has been checked. Form the correction
            // from the same snapshot only when this branch consumes it.
            ApplyMediumGramCorrection();
        }
        // Q stays immutable until the uniform pre-output rendezvous. That
        // barrier retires all readers before an output-backed snapshot is
        // replaced with final eigenvectors.
        Barrier();return false;
    }
    __aicore__ inline void AwaitInactiveMediumCertificate(){
        SelectMediumBoundaryCertificate();
        if(mediumPrivateSolve)return;
        SyncAll();SyncAll();
    }
    __aicore__ inline void SolvePrivateInverseVector(){
        if constexpr(IndependentMedium){
            // DetectNearbyRoots already computed this current-input decision.
            // The complete path computes every required column itself.
            if(boundaryClusterMask && count<Order()){
                RebuildWholeMediumEigensystem();
                return;
            }
        }
        const uint64_t clusterMask=SolveInverseColumns();
        RefineOwnedClusters(clusterMask,extendedEigenvectorMask);
        Backtransform();
    }
    __aicore__ inline void ZeroMediumTail(LocalTensor<float> buffer,uint32_t firstZeroElement){
        // Medium resident rows are one 64-float repeat. Keep the instruction
        // address aligned and select an unaligned logical tail with a mask.
        constexpr uint32_t floatVectorElementCount=64;
        if(firstZeroElement>=floatVectorElementCount)return;
        const uint64_t tailMask=~uint64_t(0)<<firstZeroElement;
        SetMaskNorm();SetVectorMask<float>(0,tailMask);
        Duplicate<float,false>(buffer,0.0f,64,1,1,8);
        PipeBarrier<PIPE_V>();ResetMask();
    }
    __aicore__ inline bool PreferCooperativeMediumSolve(){
        // The bisection matrix scratch is retired here. These eight rows
        // fit the complete resident arena for every medium matrix order.
        auto diagonal=mat,edge=mat[Stride()],diagonalSquared=mat[2*Stride()];
        auto edgeSquared=mat[3*Stride()],fourthTerms=mat[4*Stride()];
        auto shifted=mat[5*Stride()],term=mat[6*Stride()];
        // Scalar prefix writes must not race with an earlier vector reader.
        // Distinct retired rows keep both offset arrays immutable until the
        // final vector-to-scalar rendezvous; PIPE_V alone cannot retire S/V WAR.
        auto previousIndices=mat[7*Stride()].template ReinterpretCast<int32_t>();
        auto nextIndices=mat[8*Stride()].template ReinterpretCast<int32_t>();
        AscendC::Abs(diagonalSquared,origD,Stride());
        AscendC::Abs(edgeSquared,origE,Stride());PipeBarrier<PIPE_V>();
        ZeroMediumTail(edgeSquared,Order()-1);
        Max(diagonalSquared,diagonalSquared,edgeSquared,Stride());PipeBarrier<PIPE_V>();
        WholeReduceMax(red,diagonalSquared,Order(),1,1,1,8);
        WholeReduceSum(fac,origD,Order(),1,1,1,8);VectorToScalar();
        const float coefficientScale=red.GetValue(0);
        if(!(coefficientScale>1e-30f) || !(coefficientScale<1e30f))return false;
        const float inverseScale=1.0f/coefficientScale;
        const float meanDiagonal=fac.GetValue(0)*inverseScale/float(int32_t(Order()));
        Muls(diagonal,origD,inverseScale,Stride());
        Muls(edge,origE,inverseScale,Stride());PipeBarrier<PIPE_V>();
        Adds(diagonal,diagonal,-meanDiagonal,Stride());PipeBarrier<PIPE_V>();
        ZeroMediumTail(diagonal,Order());
        ZeroMediumTail(edge,Order()-1);
        Mul(diagonalSquared,diagonal,diagonal,Stride());
        Mul(edgeSquared,edge,edge,Stride());PipeBarrier<PIPE_V>();
        Muls(term,edgeSquared,2.0f,Stride());PipeBarrier<PIPE_V>();
        Add(term,term,diagonalSquared,Stride());PipeBarrier<PIPE_V>();
        WholeReduceSum(red,term,Order(),1,1,1,8);
        CreateIndicesWithArena(previousIndices,int32_t(-1),Stride());PipeBarrier<PIPE_V>();
        Maxs(previousIndices,previousIndices,int32_t(0),Stride());PipeBarrier<PIPE_V>();
        Muls(previousIndices,previousIndices,int32_t(sizeof(float)),Stride());
        Duplicate(shifted,0.0f,Stride());PipeBarrier<PIPE_V>();
        Gather(shifted,edgeSquared,previousIndices.template ReinterpretCast<uint32_t>(),0,Order());PipeBarrier<PIPE_V>();
        Duplicate(shifted,0.0f,1);PipeBarrier<PIPE_V>();
        Add(fourthTerms,diagonalSquared,edgeSquared,Stride());PipeBarrier<PIPE_V>();
        Add(fourthTerms,fourthTerms,shifted,Stride());PipeBarrier<PIPE_V>();
        Mul(fourthTerms,fourthTerms,fourthTerms,Stride());PipeBarrier<PIPE_V>();
        CreateIndicesWithArena(nextIndices,int32_t(1),Stride());PipeBarrier<PIPE_V>();
        Mins(nextIndices,nextIndices,int32_t(Order()-1),Stride());PipeBarrier<PIPE_V>();
        Muls(nextIndices,nextIndices,int32_t(sizeof(float)),Stride());PipeBarrier<PIPE_V>();
        Gather(shifted,diagonal,nextIndices.template ReinterpretCast<uint32_t>(),0,Order());PipeBarrier<PIPE_V>();
        Add(term,diagonal,shifted,Stride());PipeBarrier<PIPE_V>();
        Mul(term,term,term,Stride());PipeBarrier<PIPE_V>();
        Mul(term,term,edgeSquared,Stride());PipeBarrier<PIPE_V>();
        Axpy(fourthTerms,term,2.0f,Stride());PipeBarrier<PIPE_V>();
        Gather(shifted,edgeSquared,nextIndices.template ReinterpretCast<uint32_t>(),0,Order());PipeBarrier<PIPE_V>();
        Mul(term,shifted,edgeSquared,Stride());PipeBarrier<PIPE_V>();
        Axpy(fourthTerms,term,2.0f,Stride());PipeBarrier<PIPE_V>();
        WholeReduceSum(fac,fourthTerms,Order(),1,1,1,8);VectorToScalar();
        const float squaredFrobenius=red.GetValue(0),fourthMoment=fac.GetValue(0);
        // Spectral concentration is sum(lambda^4)/sum(lambda^2)^2 for
        // the centered tridiagonal. Replicated owners make the same choice.
        return squaredFrobenius>0.0f &&
            fourthMoment>0.125f*squaredFrobenius*squaredFrobenius;
    }
    __aicore__ inline uint64_t MediumBoundaryRecordsFloatOffset()const{
        const uint64_t matrixFloatCount=mediumOutputScratch?0:uint64_t(Order())*Stride();
        const uint64_t metricFloatCount=uint64_t(team)*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT;
        const uint64_t historyFloatCount=mediumSharedReduction?uint64_t(batch)*((Order()+1)*Stride()+16):0;
        return historyFloatCount+uint64_t(batch)*(matrixFloatCount+metricFloatCount);
    }
    __aicore__ inline void SelectMediumBoundaryCertificate(){
        if(!mediumBoundaryConsensus || mediumBoundaryConsensusReady)return;
        if(!mediumPrivateSolve){
            // The existing spectral decision is already uniform across all
            // launched blocks. A complete certificate is valid for every
            // matrix, including mixed groups, without another rendezvous.
            mediumPrefersCooperative=true;
            mediumBoundaryConsensusReady=true;ScalarToVector();return;
        }
        const uint64_t recordsOffset=MediumBoundaryRecordsFloatOffset();
        if(active){
            // Every real owner publishes a complete record for this input.
            // Classification never shares storage with a later certificate.
            Duplicate(red,0.0f,EIGH_CERTIFICATE_SLOT_FLOAT_COUNT);VectorToScalar();
            red.SetValue(0,(boundaryClusterMask || mediumPrefersCooperative)?-1.0f:1.0f);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(ws[recordsOffset+(uint64_t(currentBatch)*team+lane)*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT],red,EIGH_CERTIFICATE_SLOT_FLOAT_COUNT);
            Barrier();
        }
        // Padding participants join here too. This rendezvous publishes all
        // current-wave classifications before any block chooses its protocol.
        SyncAll();
        const uint32_t recordCount=mediumWaveMatrixCount*team;
        DataCopy(wideBuf,ws[recordsOffset+uint64_t(mediumWaveFirstMatrix)*team*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT],recordCount*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT);Barrier();
        bool anyCooperative=false,ownMatrixCooperative=false;
        if(recordCount<=64 && team<=64){
            // Each complete record contains one exact +1/-1 flag at its
            // first coordinate. A one-element repeat extracts these flags
            // without reading padding or changing their classification.
            constexpr uint8_t recordRepeatStride=EIGH_CERTIFICATE_SLOT_FLOAT_COUNT/8;
            WholeReduceMin(tmp2,wideBuf,1,uint8_t(recordCount),1,1,recordRepeatStride,ReduceOrder::ORDER_ONLY_VALUE);
            if(group<mediumWaveMatrixCount)
                WholeReduceMin(fac,wideBuf[group*team*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT],1,uint8_t(team),1,1,recordRepeatStride,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(red,tmp2,recordCount,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            if(group<mediumWaveMatrixCount)
                WholeReduceMin(w,fac,team,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            VectorToScalar();
            anyCooperative=red.GetValue(0)<0.0f;
            if(group<mediumWaveMatrixCount)ownMatrixCooperative=w.GetValue(0)<0.0f;
        }else{
        for(uint32_t matrix=0;matrix<mediumWaveMatrixCount;++matrix){
            bool matrixCooperative=false;
            for(uint32_t owner=0;owner<team;++owner)
                matrixCooperative=matrixCooperative || wideBuf.GetValue((matrix*team+owner)*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT)<0.0f;
            anyCooperative=anyCooperative || matrixCooperative;
            if(matrix==group)ownMatrixCooperative=matrixCooperative;
        }
        }
        // All blocks derive one launch-wide rendezvous decision; owners of a
        // crossing cluster use the existing complete Gram/residual certificate.
        mediumPrivateSolve=!anyCooperative;
        mediumPrefersCooperative=ownMatrixCooperative;
        mediumBoundaryConsensusReady=true;
        // Retire scalar classification reads before inverse iteration reuses
        // their wide-buffer storage with vector instructions.
        ScalarToVector();
    }
    __aicore__ inline void InverseVector(){
        if constexpr(IndependentMedium){
            SelectMediumBoundaryCertificate();
            // All owners use the established complete private solve, including
            // full-spectrum reconstruction when a cluster crosses ownership.
            // No GM candidate is published, so there are no shared readers.
            if(mediumPrivateSolve){SolvePrivateInverseVector();return;}
            if(mediumSharedReduction?mediumPrefersCooperative:PreferCooperativeMediumSolve()){
                SolveInverseColumns();
                if(!CertifyMediumOutputAndRepair())Backtransform();
            }else{
                SolvePrivateInverseVector();
                // Other matrix groups can choose the shared route. All
                // launch participants must execute its two rendezvous.
                SyncAll();SyncAll();
            }
        }else SolvePrivateInverseVector();
    }
    __aicore__ inline float ResidentRayleigh(LocalTensor<float> q){
        // The final inverse solve has retired its LU pivots and factors.
        // For symmetric T, q^T*T*q=sum(d_i*q_i^2)+2*sum(e_i*q_i*q_(i+1)).
        // e_(n-1) and row padding were explicitly zeroed by reduction.
        auto neighborByteOffsets=piv.template ReinterpretCast<int32_t>();
        auto diagonalTerms=fac,edgeProducts=tmp2;
        CreateIndicesWithArena(neighborByteOffsets,int32_t(1),Stride());PipeBarrier<PIPE_V>();
        Mins(neighborByteOffsets,neighborByteOffsets,int32_t(Order()-1),Stride());PipeBarrier<PIPE_V>();
        Muls(neighborByteOffsets,neighborByteOffsets,int32_t(sizeof(float)),Stride());
        Duplicate(w,0.0f,Stride());PipeBarrier<PIPE_V>();
        Gather(w,q,neighborByteOffsets.template ReinterpretCast<uint32_t>(),0,Order());
        Mul(edgeProducts,q,origE,Stride());
        Mul(diagonalTerms,q,origD,Stride());PipeBarrier<PIPE_V>();
        Mul(edgeProducts,edgeProducts,w,Stride());
        Mul(diagonalTerms,diagonalTerms,q,Stride());PipeBarrier<PIPE_V>();
        Axpy(diagonalTerms,edgeProducts,2.0f,Stride());PipeBarrier<PIPE_V>();
        WholeReduceSum(red,diagonalTerms,Stride(),1,1,1,8);VectorToScalar();
        return red.GetValue(0);
    }
    __aicore__ inline float ScalarEigenvector(LocalTensor<float> q,uint32_t index,float ev){
        Duplicate(q,0.0f,Stride());Barrier();
        if(!ResidentFamily()){
            auto left=pcrLeft[Stride()].template ReinterpretCast<int32_t>(),right=pcrLeft[2*Stride()].template ReinterpretCast<int32_t>();
            CreateIndicesWithArena(left,int32_t(-1),Stride());CreateIndicesWithArena(right,int32_t(1),Stride());PipeBarrier<PIPE_V>();
            Muls(left,left,int32_t(4),Stride());Muls(right,right,int32_t(4),Stride());PipeBarrier<PIPE_V>();
            Maxs(left,left,int32_t(0),Stride());Mins(right,right,int32_t((Order()-1)*4),Stride());Barrier();
        }
        float reciprocal=1.0f/SafePivot(origD.GetValue(0)-ev);piv.SetValue(0,reciprocal);
        for(uint32_t i=1;i<Order();++i){
            float f=origE.GetValue(i-1)*reciprocal;fac.SetValue(i-1,f);
            reciprocal=1.0f/SafePivot(origD.GetValue(i)-ev-origE.GetValue(i-1)*f);piv.SetValue(i,reciprocal);
        }
        if(!ResidentFamily()){
            // The same invocation-local start, generated with integer SIMD.
            auto h=q.template ReinterpretCast<uint32_t>(),shifted=w.template ReinterpretCast<uint32_t>();
            auto h16=h.template ReinterpretCast<uint16_t>(),s16=shifted.template ReinterpretCast<uint16_t>(),t16=red.template ReinterpretCast<uint16_t>();
            CreateIndicesWithArena(q.template ReinterpretCast<int32_t>(),int32_t(1),Stride());PipeBarrier<PIPE_V>();
            Muls(q.template ReinterpretCast<int32_t>(),q.template ReinterpretCast<int32_t>(),int32_t(747796405),Stride());PipeBarrier<PIPE_V>();
            Adds(q.template ReinterpretCast<int32_t>(),q.template ReinterpretCast<int32_t>(),int32_t((index+1)*2891336453u),Stride());PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(16),Stride());PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Not(t16,t16,2*Stride());PipeBarrier<PIPE_V>();And(h16,h16,t16,2*Stride());PipeBarrier<PIPE_V>();
            Muls(q.template ReinterpretCast<int32_t>(),q.template ReinterpretCast<int32_t>(),int32_t(2246822519u),Stride());PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(13),Stride());PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Not(t16,t16,2*Stride());PipeBarrier<PIPE_V>();And(h16,h16,t16,2*Stride());PipeBarrier<PIPE_V>();
            ShiftLeft(h,h,uint32_t(16),Stride());PipeBarrier<PIPE_V>();ShiftRight(h,h,uint32_t(16),Stride());PipeBarrier<PIPE_V>();
            Cast(q,q.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,Stride());PipeBarrier<PIPE_V>();
            Muls(q,q,1.0f/65536.0f,Stride());PipeBarrier<PIPE_V>();Adds(q,q,-0.5f,Stride());Barrier();
            for(uint32_t i=Order();i<Stride();++i)q.SetValue(i,0);Barrier();
        }else{
        for(uint32_t i=0;i<Order();++i){uint32_t h=(i+1)*747796405u+(index+1)*2891336453u;h^=h>>16;h*=2246822519u;h^=h>>13;q.SetValue(i,float(int32_t(h&65535u))/65536.0f-0.5f);}        }

        for(uint32_t it=0;it<3;++it){
            // A fixed bisection shift preserves the requested eigenpair even
            // when the first inverse iterate has a small target projection.

            float previous=q.GetValue(0);for(uint32_t i=1;i<Order();++i){previous=q.GetValue(i)-fac.GetValue(i-1)*previous;q.SetValue(i,previous);}
            float a=q.GetValue(Order()-1)*piv.GetValue(Order()-1);q.SetValue(Order()-1,a);float mx=Abs(a);
            for(int i=int(Order())-2;i>=0;--i){a=(q.GetValue(i)-origE.GetValue(i)*a)*piv.GetValue(i);q.SetValue(i,a);mx=mx>Abs(a)?mx:Abs(a);}
            // Inverse iteration is homogeneous. Bound the intermediate magnitude;
            // only the completed eigenvector requires unit Euclidean norm.
            Barrier();Muls(q,q,1.0f/mx,Stride());Barrier();
            if(it+1==3){float norm=ResidentFamily()?ResidentNorm(q):Root(Dot(q,q));Muls(q,q,1.0f/norm,Stride());Barrier();}
        }
        // Evaluate the complete current tridiagonal quotient in either layout.
        if(!ResidentFamily())return ReferenceRayleigh(q);
        return ResidentRayleigh(q);
    }
    __aicore__ inline ExtendedFloat QExtended(LocalTensor<float> q,uint32_t i){return ExtendedFloat(q.GetValue(i),prod.GetValue(i),0);}
    __aicore__ inline void SetQExtended(LocalTensor<float> q,uint32_t i,ExtendedFloat x){q.SetValue(i,x.h);prod.SetValue(i,x.m);}
    __aicore__ inline ExtendedFloat AbsExtended(ExtendedFloat x){return x.Negative()?-x:x;}
    __aicore__ inline ExtendedFloat SafeExtended(ExtendedFloat x){
        float magnitude=Abs(x.h);if(magnitude<1e-25f && Abs(x.m)<1e-25f)return ExtendedFloat(x.Negative()?-1e-25f:1e-25f);return x;
    }
    __aicore__ inline void PivotFactorExtended(ExtendedFloat shift){
        auto swaps=piv.template ReinterpretCast<uint32_t>();
        for(uint32_t i=0;i<Order();++i){
            WriteExtended(i,ExtendedFloat(origD.GetValue(i))-shift);
            WriteExtended(Order()+i,ExtendedFloat(i+1<Order()?origE.GetValue(i):0));
            WriteExtended(2*Order()+i,ExtendedFloat(i+1<Order()?origE.GetValue(i):0));
            WriteExtended(3*Order()+i,ExtendedFloat(0));swaps.SetValue(i,0);
        }
        for(uint32_t i=0;i+1<Order();++i){
            ExtendedFloat diagonal=ReadExtended(i),lower=ReadExtended(Order()+i);
            if(AbsExtended(diagonal).h>=AbsExtended(lower).h){
                ExtendedFloat multiplier=lower/SafeExtended(diagonal);WriteExtended(Order()+i,multiplier);
                WriteExtended(i+1,ReadExtended(i+1)-multiplier*ReadExtended(2*Order()+i));
            }else{
                ExtendedFloat multiplier=diagonal/SafeExtended(lower),next=ReadExtended(i+1);
                WriteExtended(i,lower);WriteExtended(i+1,ReadExtended(2*Order()+i)-multiplier*next);
                WriteExtended(Order()+i,multiplier);WriteExtended(2*Order()+i,next);swaps.SetValue(i,1);
                if(i+2<Order()){ExtendedFloat upper=ReadExtended(2*Order()+i+1);WriteExtended(3*Order()+i,upper);WriteExtended(2*Order()+i+1,-multiplier*upper);}
            }
        }
    }
    __aicore__ inline ExtendedFloat RayleighExtended(LocalTensor<float> q){
        ExtendedFloat result(0),norm(0);
        for(uint32_t i=0;i<Order();++i){
            ExtendedFloat x=QExtended(q,i);norm=norm+x*x;
            result=result+ExtendedFloat(origD.GetValue(i))*x*x;
            if(i+1<Order())result=result+(ExtendedFloat(origE.GetValue(i))*x*QExtended(q,i+1)).Scale(2.0f);
        }
        return result/norm;
    }
    __aicore__ __attribute__((noinline)) float PivotedExtendedEigenvector(LocalTensor<float> q,uint32_t index,float ev){
        for(uint32_t i=0;i<Order();++i){
            uint32_t hash=(i+1)*747796405u+(index+1)*2891336453u;hash^=hash>>16;hash*=2246822519u;hash^=hash>>13;
            SetQExtended(q,i,ExtendedFloat(float(int32_t(hash&65535u))/65536.0f-0.5f));
        }
        ExtendedFloat shift(ev);PivotFactorExtended(shift);
        auto swaps=piv.template ReinterpretCast<uint32_t>();
        for(uint32_t it=0;it<5;++it){
            if(it>=3){shift=RayleighExtended(q);PivotFactorExtended(shift);}
            for(uint32_t i=0;i+1<Order();++i){
                ExtendedFloat a=QExtended(q,i),b=QExtended(q,i+1),f=ReadExtended(Order()+i);
                if(swaps.GetValue(i)){SetQExtended(q,i,b);SetQExtended(q,i+1,a-f*b);}
                else SetQExtended(q,i+1,b-f*a);
            }
            for(int i=int(Order())-1;i>=0;--i){
                ExtendedFloat value=QExtended(q,uint32_t(i));
                if(i+1<int(Order()))value=value-ReadExtended(2*Order()+uint32_t(i))*QExtended(q,uint32_t(i)+1);
                if(i+2<int(Order()))value=value-ReadExtended(3*Order()+uint32_t(i))*QExtended(q,uint32_t(i)+2);
                SetQExtended(q,uint32_t(i),value/SafeExtended(ReadExtended(uint32_t(i))));
            }
            float maximum=0;for(uint32_t i=0;i<Order();++i){float x=Abs(QExtended(q,i).h);maximum=maximum>x?maximum:x;}
            ExtendedFloat scale(maximum>0?maximum:1),norm(0);
            for(uint32_t i=0;i<Order();++i){ExtendedFloat x=QExtended(q,i)/scale;SetQExtended(q,i,x);norm=norm+x*x;}
            ExtendedFloat length=RootExtended(norm);
            for(uint32_t i=0;i<Order();++i)SetQExtended(q,i,QExtended(q,i)/length);
        }
        ExtendedFloat refined=RayleighExtended(q);
        for(uint32_t i=Order();i<Stride();++i)q.SetValue(i,0);
        Barrier();return refined.Float();
    }


    __aicore__ inline void PublishSmallMode(bool selected){
        if constexpr(PrivateSmall)return;
        Duplicate(param,0.0f,8);Barrier();param.SetValue(0,selected?1.0f:0.0f);
        event_t ready=ImmediateEvent<HardEvent::S_MTE3>();
        SetFlag<HardEvent::S_MTE3>(ready);WaitFlag<HardEvent::S_MTE3>(ready);
        DataCopy(gparam,param,8);Barrier();
    }
    __aicore__ inline bool AnySmallMode(uint32_t step){
        if constexpr(PrivateSmall){
            // One matrix group obtains the same reduction and QL decision on
            // every launched lane, including padded lanes. Multi-group input
            // can select different numerical paths and keeps full barriers.
            if(groups==1 && rows<=2)return smallQLNeedsCheck && smallQLSolved;
            return true;
        }
        bool any=false;
        for(uint32_t g=0;g<groups;++g){
            uint32_t matrix=step*groups+g;
            if(matrix>=batch)continue;
            uint64_t offset=uint64_t(matrix)*WorkspacePerMatrix()+uint64_t(2*Order()+2)*Stride();
            DataCacheCleanAndInvalid<float,CacheLine::SINGLE_CACHE_LINE>(ws[offset]);Barrier();
            DataCopyPad(param,ws[offset],DataCopyExtParams{1,4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});Barrier();
            any=any || param.GetValue(0)!=0.0f;
        }
        return any;
    }

    __aicore__ inline void TransposeSmallProducts(LocalTensor<float> high,LocalTensor<float> low,
        LocalTensor<float> transposedHigh,LocalTensor<float> transposedLow,uint32_t pitch){
        uint64_t sources[16],destinations[16];
        TransDataTo5HDParams parameters{false,false,1,0,0};
        for(uint32_t which=0;which<2;++which){
            auto input=which?low:high,output=which?transposedLow:transposedHigh;
            for(uint32_t row=0;row<pitch;row+=16)for(uint32_t column=0;column<Stride();column+=8){
                for(uint32_t i=0;i<16;++i)
                    sources[i]=uint64_t((row+i<Order()?input[(row+i)*Stride()+column]:v[column]).GetPhyAddr());
                for(uint32_t j=0;j<8;++j){
                    destinations[2*j]=uint64_t(output[(column+j)*pitch+row].GetPhyAddr());
                    destinations[2*j+1]=uint64_t(output[(column+j)*pitch+row+8].GetPhyAddr());
                }
                TransDataTo5HD<float>(destinations,sources,parameters);PipeBarrier<PIPE_V>();
            }
        }
        Barrier();
    }
    // Check the complete current-input solution before accepting the wider
    // FP32 magnitude family. All references here are this invocation's A/Q.
    __aicore__ __attribute__((noinline)) void CheckSmallQL(){
        uint32_t pitch=Stride(),dense=Order()*pitch;
        auto arena=smallStorage.template Get<float>();
        auto original=arena,product=arena[dense],allQ=arena[2*dense];
        auto ph=arena[3*dense],pm=arena[4*dense],zero=arena[5*dense];
        Duplicate(original,0.0f,dense);Barrier();
        // The compact input copy is still live in allQ until reconstruction.
        for(uint32_t i=0;i<Order();++i)for(uint32_t j=0;j<Order();++j){
            uint32_t pi=upper?Order()-1-i:i,pj=upper?Order()-1-j:j;
            uint32_t row=pi>pj?pi:pj,col=pi>pj?pj:pi;
            if(upper){uint32_t t=row;row=col;col=t;}
            original.SetValue(i*pitch+j,allQ.GetValue(row*Order()+col));
        }
        Barrier();Mul(product,original,original,dense);Duplicate(w,0.0f,pitch);PipeBarrier<PIPE_V>();
        WholeReduceSum(w,product,pitch,uint8_t(Order()),1,1,uint8_t(pitch/8));PipeBarrier<PIPE_V>();
        ReduceSumWithArena(red,w,tmp,pitch);VectorToScalar();float frobenius2=red.GetValue(0);
        if constexpr(PrivateSmall){
            DataCopyPad(allQ,vectors[uint64_t(currentBatch)*Order()*Order()],
                DataCopyExtParams{uint16_t(Order()),Order()*4,0,0,0},
                DataCopyPadExtParams<float>{true,0,uint8_t(Stride()-Order()),0});
        }else DataCopy(allQ,hist,dense);
        Duplicate(zero,0.0f,dense);Duplicate(v,0.0f,pitch);Barrier();
        const float frobenius=Root(frobenius2);
        const float perColumn=frobenius/Root(float(int32_t(Order())));
        float safeGap=5e-5f*perColumn;
        float observedGap=3.4e38f;
        bool completeSpectrum=true;
        for(uint32_t i=0;i<Order();++i){
            float value=d.GetValue(i);
            if(!(value==value) || Abs(value)>2.0f*frobenius)completeSpectrum=false;
            if(i){float gap=value-d.GetValue(i-1);if(gap<observedGap)observedGap=gap;}
        }
        // All columns must also pass the original Gram/residual checks.
        // Their joint bounds and the polar factor give an ordered spectral
        // perturbation below0.001*||A||F. Subtract0.01*||A||F and halve
        // the remaining observed gap, retaining substantial rounding margin.
        const float candidateGap=0.5f*(observedGap-0.01f*frobenius);
        const bool useObservedGap=completeSpectrum && candidateGap>safeGap;
        if(useObservedGap)safeGap=candidateGap;
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*pitch];
            Mul(product,allQ,q,pitch,uint8_t(Order()),BinaryRepeatParams{1,1,1,uint8_t(pitch/8),uint8_t(pitch/8),0});
            Duplicate(w,0.0f,pitch);PipeBarrier<PIPE_V>();
            WholeReduceSum(w,product,pitch,uint8_t(Order()),1,1,uint8_t(pitch/8));VectorToScalar();
            float gram=0.0f;
            for(uint32_t j=0;j<Order();++j){float value=w.GetValue(j)-(j==EigenIndex(r)?1.0f:0.0f);gram+=value*value;}
            SetVectorMask<float>(pitch);DenseProduct<true>(ph,pm,original,zero,q,v);Barrier();
            ExtendedFloat rayleigh(0.0f),length(0.0f);
            float eigenvalue=0.0f,residual2=0.0f;
            if(Order()>=8){
                uint32_t transposePitch=(Order()+15)/16*16;
                auto high=arena[11*dense],low=high[pitch*transposePitch];
                TransposeSmallProducts(ph,pm,high,low,transposePitch);
                Duplicate(w,0.0f,pitch);Duplicate(red,0.0f,pitch);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float>(pitch);
                // Identical ascending column order, with independent rows
                // in SIMD lanes. Retain every two-float sum component.
                for(uint32_t j=0;j<Order();++j)
                    SmallSum<false>(w,red,w,red,high[j*transposePitch],low[j*transposePitch]);
                SmallProduct(ph,pm,q,v,w,red);Barrier();
                for(uint32_t i=0;i<Order();++i)rayleigh=rayleigh+ExtendedFloat(ph.GetValue(i),pm.GetValue(i),0.0f);
                SmallProduct(ph,pm,q,v,q,v);Barrier();
                for(uint32_t i=0;i<Order();++i)length=length+ExtendedFloat(ph.GetValue(i),pm.GetValue(i),0.0f);
                eigenvalue=(rayleigh/length).Float();
                SmallScale<false>(ph,pm,q,v,ExtendedFloat(eigenvalue));
                SmallSum<true>(ph,pm,w,red,ph,pm);Barrier();
                Add(product,ph,pm,pitch);PipeBarrier<PIPE_V>();Mul(product,product,product,pitch);PipeBarrier<PIPE_V>();
                ReduceSumWithArena(red,product,tmp,pitch);VectorToScalar();residual2=red.GetValue(0);
            }else{
                for(uint32_t i=0;i<Order();++i){
                    ExtendedFloat aq(0.0f);
                    for(uint32_t j=0;j<Order();++j)aq=aq+ExtendedFloat(ph.GetValue(i*pitch+j),pm.GetValue(i*pitch+j),0.0f);
                    w.SetValue(i,aq.h);red.SetValue(i,aq.m);
                    ExtendedFloat x(q.GetValue(i));rayleigh=rayleigh+x*aq;length=length+x*x;
                }
                eigenvalue=(rayleigh/length).Float();
                for(uint32_t i=0;i<Order();++i){
                    ExtendedFloat residual=ExtendedFloat(w.GetValue(i),red.GetValue(i),0.0f)-ExtendedFloat(eigenvalue)*ExtendedFloat(q.GetValue(i));
                    float value=residual.Float();residual2+=value*value;
                }
            }
            const float tolerance=1e-4f+1e-4f*Abs(eigenvalue);
            const float arithmetic=1e-12f*float(int32_t(Order()*Order()))*smallRowBound+1.1920928955078125e-7f*Abs(eigenvalue);
            bool bad=!(eigenvalue==eigenvalue) || !(residual2==residual2) || !(gram==gram) || !(safeGap>0.0f);
            bad=bad || Abs(eigenvalue)>3.4e38f || gram>9e-10f/float(int32_t(Order()));
            bad=bad || residual2>6.25e-10f/float(int32_t(Order()))*frobenius2;
            bad=bad || residual2>safeGap*safeGap*0.0625f;
            bad=bad || (useObservedGap && Abs(eigenvalue-d.GetValue(EigenIndex(r)))>1e-4f*perColumn);
            bad=bad || 2.0f*residual2/safeGap+arithmetic>0.25f*tolerance;
            pack.SetValue(8*r,bad?1.0f:0.0f);
            pack.SetValue(8*(count+r),PrivateSmall && bad?3.402823466e38f:eigenvalue);
        }
        event_t ready=ImmediateEvent<HardEvent::S_MTE3>();
        SetFlag<HardEvent::S_MTE3>(ready);WaitFlag<HardEvent::S_MTE3>(ready);
        if constexpr(PrivateSmall){
            // Each current eigenvalue slot temporarily carries its refined
            // value or a rejected-certificate marker. Final output overwrites it.
            DataCopyPad(values[uint64_t(currentBatch)*Order()+lane],pack[count*8],
                DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});
        }else{
            DataCopyPad(gw[lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});
            DataCopyPad(gw[pitch+lane],pack[count*8],DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});
        }Barrier();
    }
    __aicore__ inline bool AcceptSmallQL(){
        if constexpr(PrivateSmall){
            DataCopyPad(d,values[uint64_t(currentBatch)*Order()],DataCopyExtParams{1,Order()*4,0,0,0},
                DataCopyPadExtParams<float>{true,0,uint8_t(Stride()-Order()),0});
            Duplicate(tmp,0.0f,Stride());
        }else{DataCopy(tmp,gw,Stride());DataCopy(d,gw[Stride()],Stride());}Barrier();
        bool accepted=true;
        for(uint32_t i=0;i<Order();++i){
            float value=d.GetValue(i);
            if(tmp.GetValue(i)!=0.0f || !(value==value) || Abs(value)>3.4e38f)accepted=false;
            if(i && value<d.GetValue(i-1))accepted=false;
        }
        return accepted;
    }

    __aicore__ inline float SmallHypot(float x,float y){
        x=Abs(x);y=Abs(y);float scale=x>y?x:y;
        if(scale==0.0f)return 0.0f;
        x/=scale;y/=scale;
        // max(|x|,|y|)=1 bounds the squared norm to [1,2]. The existing
        // scalar Newton square root is accurate in this normal interval and
        // avoids a scalar/vector handshake for each tridiagonal QL rotation.
        return scale*Root(x*x+y*y);
    }
    // Implicit shifted QL of the complete current tridiagonal. Store the
    // accumulated orthogonal eigenvectors as contiguous transposed rows.
    // Algorithm reference: LAPACK SSTEQR, https://netlib.org/lapack/single/ssteqr.f
    __aicore__ __attribute__((noinline)) bool SolvePrivateSmallQL(){
        const uint32_t pitch=Stride();
        auto z=smallStorage.template Get<float>()[3*Order()*pitch];
        const bool selective=count<=2;
        const uint32_t rotationCapacity=3*Order()*pitch;
        uint32_t rotationCount=0;
        auto diagonal=pack,off=pack[pitch];
        DataCopy(diagonal,origD,pitch);DataCopy(off,origE,pitch);
        if(!selective)Duplicate(z,0.0f,Order()*pitch);Barrier();
        off.SetValue(Order()-1,0.0f);
        if(!selective)for(uint32_t j=0;j<Order();++j)z.SetValue(j*pitch+j,1.0f);
        Barrier();uint32_t iterations=0;
        for(uint32_t first=0;first<Order();++first){
            while(true){
                uint32_t last=first;
                for(;last+1<Order();++last){
                    const float threshold=5.960464477539063e-8f*(Abs(diagonal.GetValue(last))+Abs(diagonal.GetValue(last+1)))+1e-30f;
                    if(Abs(off.GetValue(last))<=threshold)break;
                }
                if(last==first)break;
                if(++iterations>30*Order())return false;
                float shift=(diagonal.GetValue(first+1)-diagonal.GetValue(first))/(2.0f*off.GetValue(first));
                float radius=SmallHypot(shift,1.0f);
                float denominator=shift+(shift>=0.0f?radius:-radius);
                if(denominator==0.0f)return false;
                shift=diagonal.GetValue(last)-diagonal.GetValue(first)+off.GetValue(first)/denominator;
                float sine=1.0f,cosine=1.0f,correction=0.0f;
                bool restart=false;
                for(int j=int(last)-1;j>=int(first);--j){
                    uint32_t i=uint32_t(j);
                    float lower=sine*off.GetValue(i),upper=cosine*off.GetValue(i);
                    radius=SmallHypot(lower,shift);off.SetValue(i+1,radius);
                    if(radius==0.0f){
                        diagonal.SetValue(i+1,diagonal.GetValue(i+1)-correction);
                        off.SetValue(last,0.0f);restart=true;break;
                    }
                    sine=lower/radius;cosine=shift/radius;
                    shift=diagonal.GetValue(i+1)-correction;
                    radius=(diagonal.GetValue(i)-shift)*sine+2.0f*cosine*upper;
                    correction=sine*radius;diagonal.SetValue(i+1,shift+correction);
                    shift=cosine*radius-upper;
                    // Capacity failure is independent of lane ownership,
                    // keeping fallback decisions uniform inside each matrix.
                    if(rotationCount>=rotationCapacity)return false;
                    if(selective){
                        z.SetValue(4*rotationCount,cosine);
                        z.SetValue(4*rotationCount+1,sine);
                        z.template ReinterpretCast<uint32_t>().SetValue(4*rotationCount+2,i);
                    }else{
                        auto left=z[i*pitch],right=z[(i+1)*pitch];
                        Muls(tmp,left,sine,pitch);PipeBarrier<PIPE_V>();
                        Axpy(tmp,right,cosine,pitch);PipeBarrier<PIPE_V>();
                        Muls(left,left,cosine,pitch);PipeBarrier<PIPE_V>();
                        Axpy(left,right,-sine,pitch);PipeBarrier<PIPE_V>();
                        DataCopy(right,tmp,pitch);Barrier();
                    }
                    ++rotationCount;
                }
                if(restart)continue;
                diagonal.SetValue(first,diagonal.GetValue(first)-correction);
                off.SetValue(first,shift);off.SetValue(last,0.0f);
            }
        }
        auto order=piv.template ReinterpretCast<uint32_t>();
        for(uint32_t j=0;j<Order();++j){
            float value=diagonal.GetValue(j);
            if(!(value==value) || Abs(value)>3.4e38f)return false;
            order.SetValue(j,j);
        }
        for(uint32_t j=1;j<Order();++j){
            float value=diagonal.GetValue(j);uint32_t index=order.GetValue(j),position=j;
            while(position && diagonal.GetValue(position-1)>value){
                diagonal.SetValue(position,diagonal.GetValue(position-1));
                order.SetValue(position,order.GetValue(position-1));--position;
            }
            diagonal.SetValue(position,value);order.SetValue(position,index);
        }
        for(uint32_t j=0;j<Order();++j)d.SetValue(j,diagonal.GetValue(j));
        Barrier();
        for(uint32_t r=0;r<count;++r){
            uint32_t column=order.GetValue(EigenIndex(r));
            if(selective){
                auto q=mat[r*pitch];Duplicate(q,0.0f,pitch);Barrier();q.SetValue(column,1.0f);
                for(uint32_t k=rotationCount;k>0;){
                    --k;uint32_t i=z.template ReinterpretCast<uint32_t>().GetValue(4*k+2);
                    float cosine=z.GetValue(4*k),sine=z.GetValue(4*k+1);
                    float left=q.GetValue(i),right=q.GetValue(i+1);
                    q.SetValue(i,cosine*left+sine*right);
                    q.SetValue(i+1,cosine*right-sine*left);
                }
            }else DataCopy(mat[r*pitch],z[column*pitch],pitch);
        }
        Barrier();return true;
    }

    __aicore__ inline float TwistedResidentEigenvector(LocalTensor<float> q,uint32_t index,float eigenvalue){
        const float originalEigenvalue=eigenvalue;
        // A left LDL and right UDU factorization meet at the most singular
        // twist. This avoids transporting an inverse iterate across a tiny
        // off-diagonal or a nearly singular unpivoted elimination pivot.
        for(uint32_t refinement=0;refinement<2;++refinement){
            piv.SetValue(0,SafePivot(origD.GetValue(0)-eigenvalue));
            for(uint32_t row=1;row<Order();++row){
                const float edge=origE.GetValue(row-1);
                piv.SetValue(row,SafePivot(origD.GetValue(row)-eigenvalue-edge*(edge/piv.GetValue(row-1))));
            }
            fac.SetValue(Order()-1,SafePivot(origD.GetValue(Order()-1)-eigenvalue));
            for(int row=int(Order())-2;row>=0;--row){
                const float edge=origE.GetValue(row);
                fac.SetValue(row,SafePivot(origD.GetValue(row)-eigenvalue-edge*(edge/fac.GetValue(row+1))));
            }
            uint32_t twist=0;float minimumMagnitude=3.4e38f;
            for(uint32_t row=0;row<Order();++row){
                const float gamma=piv.GetValue(row)+fac.GetValue(row)-(origD.GetValue(row)-eigenvalue);
                if(Abs(gamma)<minimumMagnitude){minimumMagnitude=Abs(gamma);twist=row;}
            }
            Duplicate(q,0.0f,Stride());VectorToScalar();q.SetValue(twist,1.0f);
            for(int row=int(twist)-1;row>=0;--row)
                q.SetValue(row,-(origE.GetValue(row)/piv.GetValue(row))*q.GetValue(row+1));
            for(uint32_t row=twist+1;row<Order();++row)
                q.SetValue(row,-(origE.GetValue(row-1)/fac.GetValue(row))*q.GetValue(row-1));
            Barrier();const float norm=ResidentNorm(q);
            if(!(norm>0.0f && norm<1e30f))
                return PivotedExtendedEigenvector(q,index,originalEigenvalue);
            Muls(q,q,1.0f/norm,Stride());Barrier();
            eigenvalue=ResidentRayleigh(q);
        }
        return eigenvalue;
    }

    __aicore__ inline void Inverse(){
        if(!ResidentFamily() && count>=2){InverseVector();return;}
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];float ev=d.GetValue(EigenIndex(r));
            if((ResidentFamily() && useExtendedEigenvectors) || (!ResidentFamily() && (Order()&(Order()-1))!=0))ev=PivotedExtendedEigenvector(q,EigenIndex(r),ev);
            else{
                // The recurrence uses the directly addressed resident UB arena.
                if(smallTwistedFactorization && ResidentFamily())ev=TwistedResidentEigenvector(q,EigenIndex(r),ev);
                else ev=ScalarEigenvector(q,EigenIndex(r),ev);
            }
            d.SetValue(EigenIndex(r),ev);
            Barrier(); // Eigenvector column signs are free under the contract.
        }
        if(!ResidentFamily() && !IndependentMedium)CertifyMatrixAndRepair();
        Backtransform();
    }

    __aicore__ inline void NormalizeColumnSigns(){
        // Reduce in original row order so equal maxima choose the first row.
        if(Order()<=2){
            for(uint32_t r=0;r<count;++r){
                auto q=mat[r*Stride()];float magnitude=-1.0f,pivot=0.0f;
                for(uint32_t row=0;row<Order();++row){float value=q.GetValue(upper?Order()-1-row:row);float a=Abs(value);if(a>magnitude){magnitude=a;pivot=value;}}
                if(pivot<0.0f){Muls(q,q,-1.0f,Stride());Barrier();}
            }
            return;
        }
        auto offsets=piv.template ReinterpretCast<int32_t>();
        if(upper){
            CreateIndicesWithArena(offsets,int32_t(0),Stride());PipeBarrier<PIPE_V>();
            Muls(offsets,offsets,int32_t(-4),Stride());PipeBarrier<PIPE_V>();
            Adds(offsets,offsets,int32_t((Order()-1)*4),Stride());Barrier();
        }
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];
            if(upper){Gather(tmp2,q,offsets.template ReinterpretCast<uint32_t>(),0,Order());PipeBarrier<PIPE_V>();AscendC::Abs(tmp2,tmp2,Order());}
            else AscendC::Abs(tmp2,q,Order());
            PipeBarrier<PIPE_V>();
            uint32_t full=Order()/64,tail=Order()%64;
            if(full)WholeReduceMax(red,tmp2,64,int32_t(full),1,1,8);
            if(tail)WholeReduceMax(param,tmp2[full*64],int32_t(tail),1,1,1,8);
            VectorToScalar();
            float magnitude=-1.0f;uint32_t index=0;
            for(uint32_t block=0;block<full;++block){
                float value=red.GetValue(2*block);
                if(value>magnitude){magnitude=value;index=64*block+red.template ReinterpretCast<uint32_t>().GetValue(2*block+1);}
            }
            if(tail && param.GetValue(0)>magnitude)index=64*full+param.template ReinterpretCast<uint32_t>().GetValue(1);
            float pivot=q.GetValue(upper?Order()-1-index:index);
            if(pivot<0.0f){Muls(q,q,-1.0f,Stride());Barrier();}
        }
    }
    __aicore__ __attribute__((noinline)) void RefineScaledResidentEigenvalues(){
        // Rescaling a near-zero tridiagonal root magnifies its absolute error.
        // Re-evaluate the completed eigenvectors against the selected original
        // input using compensated products, after reflector storage is retired.
        if(matrixScale<=1.0f)return;
        const uint32_t denseElementCount=Order()*Stride();
        auto original=smallStorage.template Get<float>();
        auto raw=original[2*denseElementCount];
        DataCopyPad(raw,input[uint64_t(currentBatch)*Order()*Order()],
            DataCopyExtParams{1,uint32_t(Order()*Order()*sizeof(float)),0,0,0},
            DataCopyPadExtParams<float>{false,0,0,0});Barrier();
        Duplicate(original,0.0f,denseElementCount);Barrier();
        LoadResidentTriangle(original,raw);Barrier();
        for(uint32_t owned=0;owned<count;++owned){
            auto vector=mat[owned*Stride()];
            ExtendedFloat energy(0.0f),squaredLength(0.0f);
            for(uint32_t row=0;row<Order();++row){
                ExtendedFloat product(0.0f);
                for(uint32_t column=0;column<Order();++column)
                    product=product+ExtendedFloat(original.GetValue(row*Stride()+column))*
                        ExtendedFloat(vector.GetValue(column));
                const ExtendedFloat coordinate(vector.GetValue(row));
                energy=energy+coordinate*product;
                squaredLength=squaredLength+coordinate*coordinate;
            }
            d.SetValue(EigenIndex(owned),(energy/squaredLength).Float());
        }
        Barrier();
    }
    __aicore__ inline void Output(uint32_t b){
        if(count){for(uint32_t r=0;r<count;++r)pack.SetValue(8*r,d.GetValue(EigenIndex(r)));Barrier();DataCopyPad(values[uint64_t(b)*Order()+lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});Barrier();}
        auto offsets=piv.template ReinterpretCast<uint32_t>();
        // Prepare the selected row order before the final column scatter.
        if(upper){CreateIndicesWithArena(offsets.template ReinterpretCast<int32_t>(),int32_t(0),Stride());PipeBarrier<PIPE_V>();Muls(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),int32_t(-4),Stride());PipeBarrier<PIPE_V>();Adds(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),int32_t((Order()-1)*4),Stride());Barrier();}
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];
            if(upper){Duplicate(tmp,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(tmp,q,offsets,0,Order());PipeBarrier<PIPE_V>();Brcb(pack,tmp,uint8_t((Order()+7)/8),{1,8});}
            else Brcb(pack,q,uint8_t((Order()+7)/8),{1,8});Barrier();
            DataCopyPad(vectors[uint64_t(b)*Order()*Order()+EigenIndex(r)],pack,DataCopyExtParams{uint16_t(Order()),4,0,(Order()-1)*4,0});Barrier();
        }
    }
    __aicore__ inline void MultiplyOriginalRows(LocalTensor<float> matrix,LocalTensor<float> vector,
        LocalTensor<float> high,LocalTensor<float> low,LocalTensor<float> splitHigh,
        LocalTensor<float> splitLow,LocalTensor<float> temporary){
        auto vectorHigh=wideBuf,vectorLow=wideBuf[Stride()];
        DenseMul(high,matrix,vector,0);
        DenseMuls(splitHigh,matrix,4097.0f);Muls(vectorHigh,vector,4097.0f,Stride());PipeBarrier<PIPE_V>();
        DenseSub(splitLow,splitHigh,matrix);Sub(vectorLow,vectorHigh,vector,Stride());PipeBarrier<PIPE_V>();
        DenseSub(splitHigh,splitHigh,splitLow);Sub(vectorHigh,vectorHigh,vectorLow,Stride());PipeBarrier<PIPE_V>();
        DenseSub(splitLow,matrix,splitHigh);Sub(vectorLow,vector,vectorHigh,Stride());PipeBarrier<PIPE_V>();
        DenseMul(low,splitHigh,vectorHigh,0);PipeBarrier<PIPE_V>();DenseSub(low,low,high);PipeBarrier<PIPE_V>();
        DenseMul(temporary,splitHigh,vectorLow,0);PipeBarrier<PIPE_V>();DenseAdd(low,low,temporary);PipeBarrier<PIPE_V>();
        DenseMul(temporary,splitLow,vectorHigh,0);PipeBarrier<PIPE_V>();DenseAdd(low,low,temporary);PipeBarrier<PIPE_V>();
        DenseMul(temporary,splitLow,vectorLow,0);PipeBarrier<PIPE_V>();DenseAdd(low,low,temporary);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FoldOriginalRows(LocalTensor<float> high,LocalTensor<float> low,
        LocalTensor<float> sum,LocalTensor<float> error,LocalTensor<float> temporary){
        const uint8_t rowStride=uint8_t(Stride()/8);
        for(uint32_t half=Stride()/2;half>=8;half/=2){
            const BinaryRepeatParams operands{1,1,1,rowStride,rowStride,rowStride};
            Add(sum,high,high[half],half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Sub(error,sum,high,half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Sub(temporary,sum,error,half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Sub(temporary,high,temporary,half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Sub(error,high[half],error,half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Add(error,error,temporary,half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Add(error,error,low,half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Add(error,error,low[half],half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Add(high,sum,error,half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Sub(temporary,high,sum,half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
            Sub(low,error,temporary,half,uint8_t(Order()),operands);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void ReduceFloatRows64(LocalTensor<float> result,
        LocalTensor<float> terms,uint32_t rowCount){
        // Three explicit pairwise levels leave eight terms per row. The
        // final hardware reduction has at most seven additions per term,
        // independent of its internal ordering; all addresses stay aligned.
        for(uint32_t half=32;half>=8;half/=2){
            Add(terms,terms,terms[half],half,uint8_t(rowCount),BinaryRepeatParams{1,1,1,8,8,8});
            PipeBarrier<PIPE_V>();
        }
        WholeReduceSum(result,terms,8,uint8_t(rowCount),1,1,8);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float>(64);
    }

    __aicore__ __attribute__((noinline)) bool TryCertifiedFloatOriginalRayleigh(
        LocalTensor<float> original,LocalTensor<float> vector,LocalTensor<float> products,
        LocalTensor<float> absoluteProducts,float previousEigenvalue,float &eigenvalue){
        // This certificate uses the problem's 1e-4 spectral-gap guarantee.
        // Outside a safe FP32 range the established compensated calculation
        // remains responsible for the complete original-input quotient.
        if(!(smallRowBound>=1e-6f && smallRowBound<=1e6f))return false;
        constexpr float unitRoundoff=5.960464477539063e-8f;
        // Explicit 64->8 folding needs three additions; the last eight-term
        // sum needs at most seven. Twelve roundoffs cover the product too.
        constexpr float gamma=12.0f*unitRoundoff;
        // S=|q|^T |A| |q| <= ||A||_infinity*||q||^2 for symmetric A.
        // If this inexpensive pessimistic quotient-rounding envelope already
        // exceeds twice the allowed error, prefer the compensated quotient
        // directly. Otherwise the full residual certificate still decides;
        // rejecting this attempt never changes the required calculation.
        const float requestedTolerance=1e-4f+1e-4f*Abs(previousEigenvalue);
        if(gamma*smallRowBound>2.0f*requestedTolerance)return false;
        const uint32_t dense=Order()*Stride();
        auto image=origD,absoluteRowSums=origE;
        SetMaskNorm();SetVectorMask<float>(64);
        Duplicate(image,0.0f,Stride());Duplicate(absoluteRowSums,0.0f,Stride());PipeBarrier<PIPE_V>();
        DenseMul(products,original,vector,0);PipeBarrier<PIPE_V>();
        AscendC::Abs(absoluteProducts,products,dense);PipeBarrier<PIPE_V>();
        ReduceFloatRows64(image,products,Order());
        ReduceFloatRows64(absoluteRowSums,absoluteProducts,Order());
        Mul(tmp,image,vector,Stride());Mul(tmp2,vector,vector,Stride());PipeBarrier<PIPE_V>();
        ReduceFloatRows64(red,tmp,1);
        ReduceFloatRows64(fac,tmp2,1);VectorToScalar();
        const float numerator=red.GetValue(0),squaredLength=fac.GetValue(0);
        SetMaskNorm();SetVectorMask<float>(64);
        if(!(squaredLength>0.5f && squaredLength<2.0f) || !(Abs(numerator)<1e10f))return false;
        const float estimate=numerator/squaredLength;

        AscendC::Abs(v,vector,Stride());PipeBarrier<PIPE_V>();
        Mul(tmp,absoluteRowSums,v,Stride());PipeBarrier<PIPE_V>();
        ReduceFloatRows64(red,tmp,1);VectorToScalar();
        const float absoluteBilinearSum=red.GetValue(0);
        SetMaskNorm();SetVectorMask<float>(64);
        Muls(w,vector,estimate,Stride());PipeBarrier<PIPE_V>();
        Sub(w,image,w,Stride());PipeBarrier<PIPE_V>();
        Mul(tmp,w,w,Stride());Mul(tmp2,absoluteRowSums,absoluteRowSums,Stride());PipeBarrier<PIPE_V>();
        ReduceFloatRows64(red,tmp,1);
        ReduceFloatRows64(fac,tmp2,1);VectorToScalar();
        const float residualSquared=red.GetValue(0),absoluteRowsSquared=fac.GetValue(0);
        SetMaskNorm();SetVectorMask<float>(64);
        if(!(absoluteBilinearSum>=0.0f && absoluteBilinearSum<1e10f &&
             residualSquared>=0.0f && absoluteRowsSquared>=0.0f))return false;
        const float residualNorm=Root(residualSquared/squaredLength);
        const float absoluteRowsNorm=Root(absoluteRowsSquared/squaredLength);
        const float quotientRoundoff=1.01f*(gamma*(absoluteBilinearSum/squaredLength+
            2.0f*Abs(estimate)+2.0f*residualNorm)+2.0f*unitRoundoff*Abs(estimate));
        const float residualUpper=1.01f*(residualNorm+gamma*absoluteRowsNorm+
            4.0f*unitRoundoff*Abs(estimate));
        // ||A||_infinity/sqrt(n) <= ||A||_2. Deflate the measured positive
        // row sum to account for its own FP32 summation error.
        const float gapLower=0.999e-4f*smallRowBound/Root(float(int32_t(Order())));
        // The exact Rayleigh quotient minimizes the residual over scalar
        // shifts, so its residual is no larger than this estimate's bound.
        const float exactRayleighResidual=residualUpper;
        if(!(gapLower>0.0f && residualUpper+quotientRoundoff<0.25f*gapLower &&
             Abs(estimate-previousEigenvalue)<0.25f*gapLower))return false;
        // The isolated Rayleigh root has error <= 2*r^2/gap. Add the
        // quotient's floating-point envelope and retain half the tolerance.
        const float errorUpper=quotientRoundoff+
            2.0f*exactRayleighResidual*exactRayleighResidual/gapLower;
        const float tolerance=1e-4f+1e-4f*Abs(estimate);
        if(!(errorUpper<=0.5f*tolerance))return false;
        eigenvalue=estimate;return true;
    }

    __aicore__ inline void PrepareOriginalQuotientOffsets(){
        static_assert(FixedStride==64,"Medium quotient rows use one 64-float repeat");
        const uint32_t terms=8*Stride();
        auto offsets=pcrBand.template ReinterpretCast<int32_t>();
        auto rows=pcrBand[512].template ReinterpretCast<int32_t>();
        auto columns=pcrBand[1024].template ReinterpretCast<int32_t>();
        CreateIndicesWithArena(offsets,int32_t(0),terms);
        ShiftRight(columns.template ReinterpretCast<uint32_t>(),offsets.template ReinterpretCast<uint32_t>(),6u,terms);
        ShiftLeft(rows.template ReinterpretCast<uint32_t>(),offsets.template ReinterpretCast<uint32_t>(),26u,terms);PipeBarrier<PIPE_V>();
        ShiftRight(rows.template ReinterpretCast<uint32_t>(),rows.template ReinterpretCast<uint32_t>(),26u,terms);PipeBarrier<PIPE_V>();
        // Padding rows gather a valid coordinate and subsequently multiply
        // a zero vector coordinate; no source outside the matrix is read.
        Mins(rows,rows,int32_t(Order()-1),terms);PipeBarrier<PIPE_V>();
        Muls(rows,rows,int32_t(Stride()*sizeof(float)),terms);
        Muls(columns,columns,int32_t(sizeof(float)),terms);PipeBarrier<PIPE_V>();
        Add(offsets,rows,columns,terms);PipeBarrier<PIPE_V>();
        // Dense product intrinsics deliberately reuse a full-repeat mask.
        SetMaskNorm();SetVectorMask<float>(64);
    }
    __aicore__ inline void AddCompensatedRowVectors(LocalTensor<float> high,LocalTensor<float> low,
        LocalTensor<float> otherHigh,LocalTensor<float> otherLow,uint32_t lanes,uint8_t repeats){
        auto sum=pcrBand[512],error=pcrBand[768],temporary=pcrBand[1024];
        const BinaryRepeatParams rows{1,1,1,8,8,8};
        Add(sum,high,otherHigh,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Sub(error,sum,high,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Sub(temporary,sum,error,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Sub(temporary,high,temporary,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Sub(error,otherHigh,error,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Add(error,error,temporary,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Add(error,error,low,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Add(error,error,otherLow,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Add(high,sum,error,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Sub(temporary,high,sum,lanes,repeats,rows);PipeBarrier<PIPE_V>();
        Sub(low,error,temporary,lanes,repeats,rows);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool TryVectorOriginalQuotient(LocalTensor<float> vector,
        LocalTensor<float> foldedHigh,LocalTensor<float> foldedLow,float &eigenvalue){
        // Backtransformation has retired prod and the LU/Gram arena. Their
        // invocation-local storage holds eight terms per coordinate and the
        // compensated dot products; the original folded rows stay intact.
        SetMaskNorm();SetVectorMask<float>(64);
        auto offsets=pcrBand.template ReinterpretCast<uint32_t>();
        auto rowHigh=prod,rowLow=prod[8*Stride()];
        Gather(rowHigh,foldedHigh,offsets,0,8*Stride());
        Gather(rowLow,foldedLow,offsets,0,8*Stride());PipeBarrier<PIPE_V>();
        for(uint32_t half=4;half;half/=2)
            AddCompensatedRowVectors(rowHigh,rowLow,rowHigh[half*Stride()],rowLow[half*Stride()],64,uint8_t(half));
        auto vectorHigh=wideBuf,vectorLow=wideBuf[Stride()];
        auto imageHigh=wideBuf[2*Stride()],imageLow=wideBuf[3*Stride()],temporary=wideBuf[4*Stride()];
        Muls(imageHigh,rowHigh,4097.0f,Stride());PipeBarrier<PIPE_V>();
        Sub(imageLow,imageHigh,rowHigh,Stride());PipeBarrier<PIPE_V>();
        Sub(imageHigh,imageHigh,imageLow,Stride());PipeBarrier<PIPE_V>();
        Sub(imageLow,rowHigh,imageHigh,Stride());PipeBarrier<PIPE_V>();
        auto energyHigh=prod[Stride()],lengthHigh=prod[2*Stride()];
        auto energyLow=prod[3*Stride()],lengthLow=prod[4*Stride()];
        Mul(energyHigh,rowHigh,vector,Stride());PipeBarrier<PIPE_V>();
        Mul(energyLow,imageHigh,vectorHigh,Stride());PipeBarrier<PIPE_V>();
        Sub(energyLow,energyLow,energyHigh,Stride());PipeBarrier<PIPE_V>();
        Mul(temporary,imageHigh,vectorLow,Stride());PipeBarrier<PIPE_V>();
        Add(energyLow,energyLow,temporary,Stride());PipeBarrier<PIPE_V>();
        Mul(temporary,imageLow,vectorHigh,Stride());PipeBarrier<PIPE_V>();
        Add(energyLow,energyLow,temporary,Stride());PipeBarrier<PIPE_V>();
        Mul(temporary,imageLow,vectorLow,Stride());PipeBarrier<PIPE_V>();
        Add(energyLow,energyLow,temporary,Stride());PipeBarrier<PIPE_V>();
        Mul(temporary,rowLow,vector,Stride());PipeBarrier<PIPE_V>();
        Add(energyLow,energyLow,temporary,Stride());PipeBarrier<PIPE_V>();
        Mul(lengthHigh,vector,vector,Stride());PipeBarrier<PIPE_V>();
        Mul(lengthLow,vectorHigh,vectorHigh,Stride());PipeBarrier<PIPE_V>();
        Sub(lengthLow,lengthLow,lengthHigh,Stride());PipeBarrier<PIPE_V>();
        Mul(temporary,vectorHigh,vectorLow,Stride());PipeBarrier<PIPE_V>();
        Muls(temporary,temporary,2.0f,Stride());PipeBarrier<PIPE_V>();
        Add(lengthLow,lengthLow,temporary,Stride());PipeBarrier<PIPE_V>();
        Mul(temporary,vectorLow,vectorLow,Stride());PipeBarrier<PIPE_V>();
        Add(lengthLow,lengthLow,temporary,Stride());PipeBarrier<PIPE_V>();
        // Energy and squared length are adjacent 64-element rows.
        for(uint32_t half=32;half>=8;half/=2)
            AddCompensatedRowVectors(energyHigh,energyLow,energyHigh[half],energyLow[half],half,2);
        VectorToScalar();ExtendedFloat energy(0.0f),length(0.0f);
        for(uint32_t term=0;term<8;++term){
            energy=energy+ExtendedFloat(energyHigh.GetValue(term),energyLow.GetValue(term),0.0f);
            length=length+ExtendedFloat(lengthHigh.GetValue(term),lengthLow.GetValue(term),0.0f);
        }
        SetMaskNorm();ResetMask();
        if(!(length.h>0.5f && length.h<2.0f))return false;
        const float estimate=(energy/length).Float();
        if(!(estimate==estimate && Abs(estimate)<3.4e38f))return false;
        eigenvalue=estimate;return true;
    }
    __aicore__ inline void RefineIndependentEigenvalues(){
        if(matrixScale==1.0f)return;
        bool required=false;
        for(uint32_t owned=0;owned<count;++owned)
            if(Abs(d.GetValue(EigenIndex(owned)))<=0.001f*smallRowBound)required=true;
        if(!required)return;
        const uint32_t dense=Order()*Stride();
        auto arena=smallStorage.template Get<float>();
        auto original=arena,high=arena[dense],low=arena[2*dense];
        // A whole-spectrum QR fallback reuses this arena for coefficients
        // and saved eigenvalues. Reload the original input after that scratch
        // lifetime ends; its initial raw copy is no longer guaranteed live.
        DataCopyPad(low,input[uint64_t(currentBatch)*Order()*Order()],
            DataCopyExtParams{1,uint32_t(Order()*Order()*sizeof(float)),0,0,0},
            DataCopyPadExtParams<float>{false,0,0,0});Barrier();
        Duplicate(original,0.0f,dense);Barrier();LoadResidentTriangle(original,low);Barrier();
        auto splitHigh=privateHistory,splitLow=privateHistory[dense];
        auto temporary=refinementStorage.template Get<float>();
        SetMaskNorm();SetVectorMask<float>(64);
        bool quotientOffsetsReady=false;
        for(uint32_t owned=0;owned<count;++owned){
            const uint32_t index=EigenIndex(owned);
            if(Abs(d.GetValue(index))>0.001f*smallRowBound)continue;
            auto vector=mat[owned*Stride()];
            const float floatQuotientEnvelope=12.0f*5.960464477539063e-8f*smallRowBound;
            const float requestedTolerance=1e-4f+1e-4f*Abs(d.GetValue(index));
            if(mediumCertifiedFloatRayleigh && floatQuotientEnvelope<=2.0f*requestedTolerance){
                float estimate=0.0f;
                if(TryCertifiedFloatOriginalRayleigh(original,vector,high,low,d.GetValue(index),estimate)){
                    d.SetValue(index,estimate);continue;
                }
            }
            if(mediumVectorOriginalQuotient && !quotientOffsetsReady){PrepareOriginalQuotientOffsets();quotientOffsetsReady=true;}
            MultiplyOriginalRows(original,vector,high,low,splitHigh,splitLow,temporary);
            FoldOriginalRows(high,low,splitHigh,splitLow,temporary);VectorToScalar();
            if(mediumVectorOriginalQuotient){
                float estimate=0.0f;
                if(TryVectorOriginalQuotient(vector,high,low,estimate)){d.SetValue(index,estimate);continue;}
            }
            ExtendedFloat rayleigh(0.0f),length(0.0f);
            for(uint32_t row=0;row<Order();++row){
                ExtendedFloat product(0.0f);
#pragma unroll
                for(uint32_t column=0;column<8;++column)
                    product=product+ExtendedFloat(high.GetValue(row*Stride()+column),low.GetValue(row*Stride()+column),0.0f);
                const ExtendedFloat value(vector.GetValue(row));
                rayleigh=rayleigh+value*product;length=length+value*value;
            }
            d.SetValue(index,(rayleigh/length).Float());
        }
        Barrier();
    }
    __aicore__ inline void OutputIndependent(uint32_t b){
        RefineIndependentEigenvalues();
        uint32_t pitch=8,logPitch=3;while(pitch<count){pitch*=2;++logPitch;}
        const uint32_t elements=Order()*pitch,dense=Order()*Stride();
        auto arena=smallStorage.template Get<float>();
        auto indices=arena.template ReinterpretCast<int32_t>();
        auto row=arena[dense].template ReinterpretCast<uint32_t>();
        auto column=arena[2*dense].template ReinterpretCast<uint32_t>();
        CreateIndicesWithArena(indices,int32_t(0),elements);PipeBarrier<PIPE_V>();
        ShiftRight(row,indices.template ReinterpretCast<uint32_t>(),logPitch,elements);
        ShiftLeft(column,indices.template ReinterpretCast<uint32_t>(),uint32_t(32)-logPitch,elements);PipeBarrier<PIPE_V>();
        ShiftRight(column,column,uint32_t(32)-logPitch,elements);PipeBarrier<PIPE_V>();
        Mins(column.template ReinterpretCast<int32_t>(),column.template ReinterpretCast<int32_t>(),int32_t(count-1),elements);PipeBarrier<PIPE_V>();
        if(upper){Muls(row.template ReinterpretCast<int32_t>(),row.template ReinterpretCast<int32_t>(),int32_t(-1),elements);PipeBarrier<PIPE_V>();Adds(row.template ReinterpretCast<int32_t>(),row.template ReinterpretCast<int32_t>(),int32_t(Order()-1),elements);PipeBarrier<PIPE_V>();}
        Muls(indices,column.template ReinterpretCast<int32_t>(),int32_t(Stride()),elements);PipeBarrier<PIPE_V>();
        Add(indices,indices,row.template ReinterpretCast<int32_t>(),elements);PipeBarrier<PIPE_V>();Muls(indices,indices,int32_t(4),elements);PipeBarrier<PIPE_V>();
        Gather(prod,mat,indices.template ReinterpretCast<uint32_t>(),0,elements);Barrier();
        for(uint32_t r=0;r<count;++r)pack.SetValue(r,d.GetValue(EigenIndex(r)));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(values[uint64_t(b)*Order()+base],pack,DataCopyExtParams{1,count*4,0,0,0});
        DataCopyPad(vectors[uint64_t(b)*Order()*Order()+base],prod,DataCopyExtParams{uint16_t(Order()),count*4,(pitch-(count+7)/8*8)/8,(Order()-count)*4,0});Barrier();
    }

    __aicore__ inline void OutputStaged(uint32_t b){
        // All vector lanes finish reading the reflector history before that
        // dead storage becomes the transposed-output staging matrix.
        if constexpr(CubeMode==0 || CubeMode==3)SyncAll();
        if(active){
            if(count){for(uint32_t r=0;r<count;++r)pack.SetValue(8*r,d.GetValue(EigenIndex(r)));Barrier();DataCopyPad(values[uint64_t(b)*Order()+lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});}
            auto stage=hist[(CubeMode==1 || CubeMode==2)?(Order()+5)*Stride():0];Barrier();
            for(uint32_t r=0;r<count;++r)DataCopy(stage[EigenIndex(r)*Stride()],mat[r*Stride()],Stride());
        }
        SyncAll();
        if(active){
            auto stage=hist[(CubeMode==1 || CubeMode==2)?(Order()+5)*Stride():0];
            uint32_t outputRows=base<Order()?(Order()-base<rows?Order()-base:rows):0;
            if(outputRows){
                uint32_t pitch=(outputRows+7)/8*8,first=upper?Order()-base-outputRows:base;
                Duplicate(prod,0.0f,Stride()*pitch);Barrier();
                DataCopyPad(prod,stage[first],DataCopyExtParams{uint16_t(Order()),outputRows*4,(Stride()-outputRows)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                Barrier();
                if(Stride()%16==0){
                    uint64_t source[16],destination[16];
                    TransDataTo5HDParams params{false,false,uint8_t(Stride()/16),2,uint16_t(pitch*2)};
                    for(uint32_t col=0;col<pitch;col+=8){
                        for(uint32_t i=0;i<16;++i)source[i]=uint64_t(prod[i*pitch+col].GetPhyAddr());
                        for(uint32_t i=0;i<8;++i){
                            uint32_t logical=col+i,row=logical<outputRows?(upper?outputRows-1-logical:logical):logical;
                            destination[2*i]=uint64_t(mat[row*Stride()].GetPhyAddr());
                            destination[2*i+1]=uint64_t(mat[row*Stride()+8].GetPhyAddr());
                        }
                        TransDataTo5HD<float>(destination,source,params);PipeBarrier<PIPE_V>();
                    }
                }else{
                    auto offsets=piv.template ReinterpretCast<int32_t>();
                    CreateIndicesWithArena(offsets,int32_t(0),Stride());PipeBarrier<PIPE_V>();Muls(offsets,offsets,int32_t(pitch*4),Stride());Barrier();
                    for(uint32_t r=0;r<outputRows;++r){
                        auto row=mat[r*Stride()];Duplicate(row,0.0f,Stride());PipeBarrier<PIPE_V>();
                        Gather(row,prod,offsets.template ReinterpretCast<uint32_t>(),(upper?outputRows-1-r:r)*4,Order());
                    }
                }
                Barrier();
                if(Order()==Stride())DataCopy(vectors[uint64_t(b)*Order()*Order()+base*Order()],mat,outputRows*Stride());
                else DataCopyPad(vectors[uint64_t(b)*Order()*Order()+base*Order()],mat,DataCopyExtParams{uint16_t(outputRows),Order()*4,(Stride()-(Order()+7)/8*8)/8,0,0});
                Barrier();
            }
        }
    }
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR ev,GM_ADDR eq,GM_ADDR work,LinalgEighTilingData t){
        if constexpr(DirectUb){pipe.next=0;} // The direct arena constructor initializes SoC state once.
        if constexpr(StaticResidentStorage)pipe.InitBuffer(smallStorage,15*FixedStride*FixedStride*4);
        else if constexpr(IndependentMedium)pipe.InitBuffer(smallStorage,3*t.n*t.stride*4);
        else if constexpr(ResidentEntry){if(t.n>=3 && t.n<=25)pipe.InitBuffer(smallStorage,15*(StaticResidentStorage?FixedStride*FixedStride:t.n*t.stride)*4);}
        if constexpr(PrivateSmall){pipe.InitBuffer(historyStorage,2*(StaticResidentStorage?FixedStride*FixedStride:t.n*t.stride)*4);privateHistory=historyStorage.template Get<float>();}
        mediumOutputScratch=(t.vectorFlags&EIGH_MEDIUM_OUTPUT_SCRATCH)!=0;
        mediumPrivateSolve=(t.vectorFlags&EIGH_MEDIUM_PRIVATE_SOLVE)!=0;
        mediumSharedReduction=(t.vectorFlags&EIGH_MEDIUM_SHARED_REDUCTION)!=0;
        if constexpr(IndependentMedium)mediumVectorReflectors=(t.vectorFlags&(EIGH_MEDIUM_UB_REFLECTOR_PIPELINE|EIGH_MEDIUM_NARROW_ROW_PIPELINE))!=0;
        else mediumVectorReflectors=false;
        mediumPrefersCooperative=false;
        smallTwistedFactorization=(t.vectorFlags&EIGH_SMALL_TWISTED_FACTORIZATION)!=0;
        mediumCertifiedFloatRayleigh=(t.vectorFlags&EIGH_MEDIUM_CERTIFIED_FLOAT_RAYLEIGH)!=0;
        mediumTwoInverseIterations=(t.vectorFlags&EIGH_MEDIUM_TWO_INVERSE_ITERATIONS)!=0;
        mediumBoundaryConsensus=(t.vectorFlags&EIGH_MEDIUM_BOUNDARY_CONSENSUS)!=0;
        mediumVectorOriginalQuotient=(t.vectorFlags&EIGH_MEDIUM_VECTOR_ORIGINAL_QUOTIENT)!=0;
        mediumBoundaryConsensusReady=false;mediumWaveFirstMatrix=0;mediumWaveMatrixCount=0;
        packedMatrix=false;dynamicOrder=t.n;dynamicStride=t.stride;batch=t.batch;upper=t.upper;team=t.team;groups=t.groups;rows=t.rows;laneStride=(rows+7)/8*8;
        uint32_t physicalTeam=(IsolatedSmall || CubeMode==3)?team:(GetTaskRation()==2?(team+1)/2*2:team);lane=GetBlockIdx()%physicalTeam;group=GetBlockIdx()/physicalTeam;base=IndependentMedium?Order()*lane/team:lane*rows;count=OwnedRowCount();
        input.SetGlobalBuffer((__gm__ float*)a);values.SetGlobalBuffer((__gm__ float*)ev);vectors.SetGlobalBuffer((__gm__ float*)eq);if constexpr(!StaticResidentStorage)ws.SetGlobalBuffer((__gm__ float*)work);
        if constexpr(!StaticResidentStorage && !IndependentMedium){
            if(!ResidentFamily())gramScratch.SetGlobalBuffer((__gm__ float*)work+uint64_t(batch)*WorkspacePerMatrix());
        }
        uint32_t wideN=StaticResidentStorage?FixedStride:(ResidentFamily()?Order():25);uint32_t wideSize=3*wideN*wideN+21*wideN+64;wideSize=wideSize>8*Stride()?wideSize:8*Stride();if constexpr(!StaticResidentStorage)wideSize=wideSize>((Stride()+1)/2)*laneStride?wideSize:((Stride()+1)/2)*laneStride;privateCapacity=wideSize+((!ResidentFamily() && Stride()%64==0)?64:0);pipe.InitBuffer(wideStorage,(wideSize+((!ResidentFamily() && Stride()%64==0)?128:0))*4);wideBuf=wideStorage.template Get<float>()[((!ResidentFamily() && Stride()%64==0)?64:0)];
        uint32_t dense=StaticResidentStorage?FixedStride*FixedStride:(rows*Stride()>Stride()*laneStride?rows*Stride():Stride()*laneStride);
        // Packed backtransformation visits the complete eight-row tile.
        if constexpr(IndependentMedium)dense=(Order()+7)/8*8*Stride();
        uint32_t color=(!ResidentFamily() && Stride()%64==0)?1:0,alternate=Stride()%128==64?1:0;
        uint32_t storageSize=3*dense+((dense+1)/2+7)/8*8+22*Stride()+64+(128+128*alternate)*color;storageSize=!ResidentFamily() && storageSize<8192?8192:storageSize;storageCapacity=storageSize;pipe.InitBuffer(storage,storageSize*4);auto b=storage.template Get<float>();mat=b;prod=b[dense+64*color];pcrBand=b[2*dense+64*color];pcrLeft=b[3*dense+64*color];uint32_t off=3*dense+((dense+1)/2+7)/8*8+64*color;
        // Keep matrix/products, reflector data and partial sums at distinct
        // repeated UB offsets. The matrix and vector arithmetic is unchanged.
        v=b[off];tmp=b[off+Stride()+64*alternate*color];tmp2=b[off+2*Stride()+128*alternate*color];red=b[off+3*Stride()+(64+64*alternate)*color];param=b[off+4*Stride()+(64+64*alternate)*color];
        d=b[off+5*Stride()+(64+64*alternate)*color];e=b[off+6*Stride()+(64+64*alternate)*color];origD=b[off+7*Stride()+(64+64*alternate)*color];origE=b[off+8*Stride()+(64+64*alternate)*color];beta=b[off+9*Stride()+(64+64*alternate)*color];
        piv=b[off+10*Stride()+(64+64*alternate)*color];fac=b[off+11*Stride()+(64+64*alternate)*color];w=b[off+12*Stride()+(64+128*alternate)*color];pack=b[off+13*Stride()+(64+128*alternate)*color];
        if constexpr(IndependentMedium)pipe.InitBuffer(refinementStorage,t.n*t.stride*sizeof(float));
    }
    __aicore__ inline void ReduceSharedMedium(uint32_t b,uint32_t firstMatrix){
        mediumWaveFirstMatrix=firstMatrix;
        mediumWaveMatrixCount=batch-firstMatrix<groups?batch-firstMatrix:groups;
        // One complete reduction per current matrix, with immutable history
        // and coefficients in a dedicated prefix. Certificate writes use a
        // disjoint suffix and cannot race with a slower snapshot reader.
        const uint32_t perMatrixElements=(Order()+1)*Stride()+16;
        const uint64_t matrixOffset=uint64_t(b)*perMatrixElements;
        const uint32_t historyElements=(Order()-2)*Stride();
        const uint32_t metadataOffset=(Order()+1)*Stride();
        if(active && lane==0){
            ReduceReplicatedFloatSmall(b);
            mediumPrefersCooperative=PreferCooperativeMediumSolve();
            DataCopy(ws[matrixOffset],privateHistory,historyElements);
            DataCopy(ws[matrixOffset+historyElements],origD,Stride());
            DataCopy(ws[matrixOffset+historyElements+Stride()],origE,Stride());
            DataCopy(ws[matrixOffset+historyElements+2*Stride()],beta,Stride());
            Barrier();Duplicate(pack,0.0f,8);VectorToScalar();
            pack.SetValue(0,smallRowBound);pack.SetValue(1,matrixScale);
            pack.SetValue(2,mediumPrefersCooperative?1.0f:0.0f);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(ws[matrixOffset+metadataOffset],pack,8);Barrier();
        }
        // Every launched participant joins once, including padded blocks.
        // This also publishes each live matrix's current cooperative choice.
        SyncAll();
        if(active && lane!=0){
            DataCopy(privateHistory,ws[matrixOffset],historyElements);
            DataCopy(origD,ws[matrixOffset+historyElements],Stride());
            DataCopy(origE,ws[matrixOffset+historyElements+Stride()],Stride());
            DataCopy(beta,ws[matrixOffset+historyElements+2*Stride()],Stride());
            DataCopy(pack,ws[matrixOffset+metadataOffset],8);Barrier();
            smallRowBound=pack.GetValue(0);matrixScale=pack.GetValue(1);
            mediumPrefersCooperative=pack.GetValue(2)>0.0f;
        }
        // A single strided DMA reads only flags written before this barrier.
        // All blocks derive the same launch-wide synchronization decision.
        const uint32_t liveGroups=batch-firstMatrix<groups?batch-firstMatrix:groups;
        DataCopyPad(wideBuf,ws[uint64_t(firstMatrix)*perMatrixElements+metadataOffset+2],
            DataCopyExtParams{uint16_t(liveGroups),4,(perMatrixElements-1)*4,0,0},
            DataCopyPadExtParams<float>{false,0,0,0});Barrier();
        bool anyCooperative=false;
        for(uint32_t g=0;g<liveGroups;++g)
            anyCooperative=anyCooperative || wideBuf.GetValue(8*g)>0.0f;
        // Normal spectra retain the barrier-free private solve. Concentrated
        // spectra retain the complete cooperative Gram/residual certificate.
        mediumPrivateSolve=!anyCooperative;
    }
    __aicore__ inline void ReduceByShape(uint32_t b,uint32_t firstMatrix){
        if constexpr(IndependentMedium){
            if(mediumSharedReduction)ReduceSharedMedium(b,firstMatrix);else ReduceReplicatedFloatSmall(b);return;
        }
        if constexpr(ResidentEntry){
            if(Order()>=3 && ResidentFamily()){
                if constexpr(PrivateSmall){
                    if(groups==1 && rows<=2){
                        bool owned=active;active=true;
                        ReduceReplicatedFloatSmall(currentBatch);
                        active=owned;return;
                    }
                }
                ReduceReplicatedFloatSmall(b);return;
            }
            if(ResidentFamily()){ReduceExtended();return;}
        }
        if(active)PackMatrix();Reduce();if(active)UnpackMatrix();
    }
    __aicore__ __attribute__((noinline)) void Process(){
        uint32_t rounds=(batch+groups-1)/groups;
        uint32_t compareLength=(Order()+63)/64*64;
        for(uint32_t step=0;step<rounds;++step){
            tridiagonalFrobeniusSquared=0.0f;ownedResidualSquared=0.0f;commonDiagonalShift=0.0f;matrixScale=1.0f;mediumBoundaryConsensusReady=false;
            boundaryClusterMask=0;
            possibleCloseMask=0;hasClusterGrid=false;extendedEigenvectorMask=0;useExtendedEigenvectors=false;smallFloatQL=false;smallQLSolved=false;smallQLNeedsCheck=false;smallQLReadyOutput=false;smallRowBound=0.0f;uint32_t b=group<groups?step*groups+group:batch;active=b<batch && lane<team;count=OwnedRowCount();directCoefficients=(base%8==0 && count%8==0 && count>0);
            currentBatch=b;
            if constexpr(PrivateSmall){if(groups==1 && rows<=2)currentBatch=step;}
            if(active){
                if constexpr(!PrivateSmall){hist=ws[uint64_t(b)*WorkspacePerMatrix()];gw=hist[(ResidentFamily()?2*Order():Order())*Stride()];gparam=gw[(ResidentFamily()?2:4)*Stride()];}
                if(Order()>=3 && ResidentFamily() && rows<=2 && lane==0)PublishSmallMode(false);
                Duplicate(d,0.0f,Stride());Duplicate(e,0.0f,Stride());Duplicate(beta,0.0f,Stride());Barrier();
                if(!IndependentMedium && !ResidentFamily()){
                    auto columnOffsets=piv.template ReinterpretCast<uint32_t>(),rowOffsets=tmp2.template ReinterpretCast<uint32_t>();
                    CreateIndicesWithArena(rowOffsets.template ReinterpretCast<int32_t>(),int32_t(0),Stride());PipeBarrier<PIPE_V>();
                    // Compares(calCount) requires a whole 64-float repeat, including the tail.
                    CreateIndicesWithArena(wideBuf.template ReinterpretCast<int32_t>(),int32_t(0),compareLength);PipeBarrier<PIPE_V>();
                    Cast(wideBuf[compareLength],wideBuf.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,compareLength);PipeBarrier<PIPE_V>();
                    if(upper){Muls(rowOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(-1),Stride());PipeBarrier<PIPE_V>();Adds(rowOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(Order()-1),Stride());PipeBarrier<PIPE_V>();}
                    Muls(rowOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(4),Stride());PipeBarrier<PIPE_V>();
                    Muls(columnOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(Order()%64==0 && count && count%8==0?count:8),Stride());Barrier();
                }
                if(!IndependentMedium && (Order()<3 || !ResidentFamily())){
                // Fetch a block of rows and its transposed counterpart using
                // two DMA commands when the owned columns are block aligned.
                if(!ResidentFamily() && Order()%64==0 && count && count%8==0){
                    uint32_t first=upper?Order()-base-count:base;
                    DataCopy(prod,input[uint64_t(b)*Order()*Order()+uint64_t(first)*Order()],count*Stride());
                    DataCopyPad(pcrBand,input[uint64_t(b)*Order()*Order()+first],DataCopyExtParams{uint16_t(Order()),count*4,(Order()-count)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                    Barrier();
                    auto columnOffsets=piv.template ReinterpretCast<uint32_t>(),rowOffsets=tmp2.template ReinterpretCast<uint32_t>();

                    uint64_t source[16],destination[16];
                    TransDataTo5HDParams params{false,false,uint8_t(Order()/16),2,uint16_t(count*2)};
                    for(uint32_t col=0;col<count;col+=8){
                        for(uint32_t i=0;i<16;++i)source[i]=uint64_t(pcrBand[i*count+col].GetPhyAddr());
                        for(uint32_t i=0;i<8;++i){
                            uint32_t row=upper?count-1-col-i:col+i;
                            destination[2*i]=uint64_t(mat[row*Stride()].GetPhyAddr());
                            destination[2*i+1]=uint64_t(mat[row*Stride()+8].GetPhyAddr());
                        }
                        TransDataTo5HD<float>(destination,source,params);PipeBarrier<PIPE_V>();
                    }
                    for(uint32_t r=0;r<count;++r){
                        auto a=mat[r*Stride()];uint32_t i=base+r,physical=upper?count-1-r:r;
                        if(upper)Gather(tmp,a,rowOffsets,0,Order());
                        Gather(w,prod[physical*Stride()],rowOffsets,0,Order());PipeBarrier<PIPE_V>();
                        auto selected=beta.template ReinterpretCast<uint8_t>();Compares(selected,wideBuf[compareLength],float(int32_t(i)),CMPMODE::LE,compareLength);PipeBarrier<PIPE_V>();
                        if(upper)Select(a,selected,w,tmp,SELMODE::VSEL_TENSOR_TENSOR_MODE,Order());
                        else Select(a,selected,w,a,SELMODE::VSEL_TENSOR_TENSOR_MODE,Order());
                        Barrier();
                    }
                }else{
                for(uint32_t r=0;r<count;++r){auto a=mat[r*Stride()];Duplicate(a,0.0f,Stride());Barrier();uint32_t i=base+r;
                    if(!ResidentFamily()){
                        uint32_t x=upper?Order()-1-i:i;
                        DataCopyPad(tmp,input[uint64_t(b)*Order()*Order()+x*Order()],DataCopyExtParams{1,Order()*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                        DataCopyPad(pack,input[uint64_t(b)*Order()*Order()+x],DataCopyExtParams{uint16_t(Order()),4,(Order()-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});Barrier();
                        {
                            Gather(a,pack,piv.template ReinterpretCast<uint32_t>(),0,Order());Gather(w,tmp,tmp2.template ReinterpretCast<uint32_t>(),0,Order());PipeBarrier<PIPE_V>();
                            auto selected=beta.template ReinterpretCast<uint8_t>();Compares(selected,wideBuf[compareLength],float(int32_t(i)),CMPMODE::LE,compareLength);PipeBarrier<PIPE_V>();
                            Select(a,selected,w,a,SELMODE::VSEL_TENSOR_TENSOR_MODE,Order());Barrier();
                        }
                    }else{
                        // The analytic-size family needs at most four input
                        // elements. Fetch them once rather than issuing scalar GM loads.
                        DataCopyPad(pack,input[uint64_t(b)*Order()*Order()],DataCopyExtParams{1,Order()*Order()*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});Barrier();
                        for(uint32_t j=0;j<Order();++j){uint32_t x=upper?Order()-1-i:i,y=upper?Order()-1-j:j;if((upper && x>y)||(!upper && x<y)){uint32_t z=x;x=y;y=z;}a.SetValue(j,pack.GetValue(x*Order()+y));}
                    }
                }
                }
                }
                Barrier();
            }
            ReduceByShape(b,step*groups);if(!ResidentFamily() && BackPanelWidth()){if constexpr(CubeMode==2){if(b<batch){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);}SyncAll();}if(active)PrepareBackPanels();SyncAll();if constexpr(CubeMode==2){if(b<batch)CrossCoreSetFlag<2,PIPE_MTE3>(8);}}count=IndependentMedium?OwnedRowCount():(lane<Order()?(Order()-1-lane)/team+1:0);
            if(!IsolatedSmall && Order()>=3 && ResidentFamily() && rows<=2 && AnySmallMode(step)){
                if(active && smallQLNeedsCheck && smallQLSolved)Backtransform();
                if constexpr(!PrivateSmall)SyncAll();
                if(active && smallQLNeedsCheck && smallQLSolved){
                    for(uint32_t r=0;r<count;++r){
                        if constexpr(PrivateSmall)DataCopyPad(vectors[uint64_t(b)*Order()*Order()+EigenIndex(r)*Order()],mat[r*Stride()],DataCopyExtParams{1,Order()*4,0,0,0});
                        else DataCopy(hist[EigenIndex(r)*Stride()],mat[r*Stride()],Stride());
                    }
                    Barrier();
                }
                SyncAll();
                if(active && smallQLNeedsCheck && smallQLSolved)CheckSmallQL();
                SyncAll();
                if(active && smallQLNeedsCheck && smallQLSolved){
                    if(AcceptSmallQL())smallQLReadyOutput=true;
                    else{smallQLSolved=false;ReduceReplicatedSmall(b,false);}
                }
                SyncAll();
            }
            if(active){
                if constexpr(ResidentEntry){
                    if(smallQLSolved){if(!smallQLReadyOutput)Backtransform();}
                    else{Bisect();Barrier();DetectNearbyRoots();if(!ResidentFamily()){DataCopy(param,d,Stride());Barrier();}Inverse();}
                }else{Bisect();Barrier();DetectNearbyRoots();if(!ResidentFamily()){DataCopy(param,d,Stride());Barrier();}Inverse();}
                if(commonDiagonalShift!=0.0f){Adds(d,d,commonDiagonalShift,Stride());Barrier();}
                if constexpr(IndependentMedium || IsolatedSmall){if(matrixScale!=1.0f){Muls(d,d,matrixScale,Stride());Barrier();}}
                if constexpr(IsolatedSmall)RefineScaledResidentEigenvalues();
                // Use the same deterministic column orientation as the protected
                // release: its largest-magnitude original coordinate is positive.
                NormalizeColumnSigns();
            }else if(b<batch && lane>=team){if constexpr(IndependentMedium)AwaitInactiveMediumCertificate();if(!IndependentMedium && Stride()>=64 && batch%groups==0)SyncAll();if(!ResidentFamily() && !IndependentMedium){SyncAll();if constexpr(CubeMode==3)AwaitCubeGram();SyncAll();SyncAll();if constexpr(CubeMode==3){CrossCoreSetFlag<2,PIPE_MTE3>(10);CrossCoreWaitFlag(9);SyncAll();SyncAll();}}if(BackPanelWidth())CubeBackHandshake(false);}else if(b>=batch){if constexpr(IndependentMedium)AwaitInactiveMediumCertificate();if(!IndependentMedium && Stride()>=64 && batch%groups==0)SyncAll();if(!ResidentFamily() && !IndependentMedium){SyncAll();if constexpr(CubeMode==3)AwaitCubeGram();SyncAll();SyncAll();if constexpr(CubeMode==3){CrossCoreSetFlag<2,PIPE_MTE3>(10);CrossCoreWaitFlag(9);SyncAll();SyncAll();}}}if constexpr(IndependentMedium){
                // Every launched participant joins, including inactive padding.
                // Final stores must not race with any Gram-correction reader.
                if(mediumOutputScratch && !mediumPrivateSolve)SyncAll();
                if(active)OutputIndependent(b);
            }else{if(!ResidentFamily())OutputStaged(b);else if(active)Output(b);}if constexpr(!PrivateSmall)SyncAll();if(BackPanelWidth() && b<batch)CrossCoreSetFlag<2,PIPE_MTE3>(6);
        }
    }
};




} // namespace CompleteResidentEigh

// linalg_eigh_cube.h
using namespace AscendC;

using CubeBackA=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using CubeBackBT=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;
using CubeBackB=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using CubeBackC=MatmulType<TPosition::GM,CubeFormat::ND,float>;
__aicore__ constexpr MatmulConfig BackConfig(uint32_t kind){
    auto c=GetNormalConfig(false,false,false,BatchMode::BATCH_LESS_THAN_L1,true,IterateOrder::ORDER_M,ScheduleType::INNER_PRODUCT,false,true);
    c.singleCoreM=kind==0?64:128;c.singleCoreN=kind==2?512:64;c.singleCoreK=kind==2?64:512;
    c.basicM=32;c.basicN=kind==2?64:32;c.basicK=64;c.enableSetBias=false;
    return c;
}
constexpr auto BACK_GRAM_CFG=BackConfig(0),BACK_PROJ_CFG=BackConfig(1),BACK_UPDATE_CFG=BackConfig(2);
constexpr auto BACK_GRAM_TILING=GetMatmulApiTiling<CubeBackA,CubeBackBT,CubeBackC,CubeBackC>(BACK_GRAM_CFG,131072);
constexpr auto BACK_PROJ_TILING=GetMatmulApiTiling<CubeBackA,CubeBackBT,CubeBackC,CubeBackC>(BACK_PROJ_CFG,131072);
constexpr auto BACK_UPDATE_TILING=GetMatmulApiTiling<CubeBackA,CubeBackB,CubeBackC,CubeBackC>(BACK_UPDATE_CFG,131072);
__aicore__ constexpr MatmulConfig NeumannConfig(){
    auto c=GetNormalConfig(false,false,false,BatchMode::BATCH_LESS_THAN_L1,true,IterateOrder::ORDER_M,ScheduleType::INNER_PRODUCT,false,true);
    c.singleCoreM=64;c.singleCoreN=64;c.singleCoreK=64;
    c.basicM=64;c.basicN=64;c.basicK=64;c.enableSetBias=false;return c;
}
constexpr auto NEUMANN_CFG=NeumannConfig();
constexpr auto NEUMANN_TILING=GetMatmulApiTiling<CubeBackA,CubeBackB,CubeBackC,CubeBackC>(NEUMANN_CFG,65536);
static_assert(NEUMANN_TILING.baseM==64 && NEUMANN_TILING.baseN==64 && NEUMANN_TILING.baseK==64);
static_assert(NEUMANN_TILING.singleCoreM==64 && NEUMANN_TILING.singleCoreN==64 && NEUMANN_TILING.singleCoreK==64);
__aicore__ inline void CubeBackProcess(GM_ADDR workspace,LinalgEighTilingData data){
    uint32_t padded=(data.rows+7)/8*8,physicalTeam=(data.team+1)/2*2;
    if(!(data.n>25 && data.stride>=64 && data.stride%64==0 && padded>=16))return;
    TPipe cubePipe;
    MatmulImpl<CubeBackA,CubeBackBT,CubeBackC,CubeBackC,BACK_GRAM_TILING> gram;
    MatmulImpl<CubeBackA,CubeBackBT,CubeBackC,CubeBackC,BACK_PROJ_TILING> projection;
    MatmulImpl<CubeBackA,CubeBackB,CubeBackC,CubeBackC,BACK_UPDATE_TILING> update;
    MatmulImpl<CubeBackA,CubeBackB,CubeBackC,CubeBackC,NEUMANN_TILING> polynomial;
    gram.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    projection.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    update.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    polynomial.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    gram.SetHF32(false);projection.SetHF32(false);update.SetHF32(false);polynomial.SetHF32(false);
    uint32_t pairCount=physicalTeam/2,group=GetBlockIdx()/pairCount,pair=GetBlockIdx()%pairCount;
    uint32_t height=2*padded,first=(data.n-3)/64*64,panels=(data.n-2+63)/64;
    uint64_t perMatrix=uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*(data.stride+64)+uint64_t(panels)*6*64*64;
    GlobalTensor<float> work;work.SetGlobalBuffer((__gm__ float*)workspace);
    for(uint32_t step=0;step<(data.batch+data.groups-1)/data.groups;++step){
        uint32_t b=group<data.groups?step*data.groups+group:data.batch;
        if(b<data.batch){
            auto hist=work[uint64_t(b)*perMatrix];
            auto gamma=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*(data.stride+64)];
            CrossCoreWaitFlag(8);
            for(uint32_t panel=pair;panel<panels;panel+=pairCount){
                uint32_t low=panel*64,number=data.n-2-low;number=number<64?number:64;
                gram.SetOrgShape(64,64,data.stride,data.stride,64);gram.SetTail(number,number,data.stride);
                gram.SetTensorA(hist[low*data.stride]);gram.SetTensorB(hist[low*data.stride],true);
                gram.IterateAll(gamma[panel*6*64*64]);gram.End();PipeBarrier<PIPE_ALL>();
            }
            CrossCoreSetFlag<2,PIPE_FIX>(9);
            CrossCoreWaitFlag(8);
            for(uint32_t panel=pair;panel<panels;panel+=pairCount){
                auto tensors=gamma[panel*6*64*64];
                auto pc=tensors[64*64],pn=tensors[2*64*64];
                auto rc=tensors[3*64*64],rn=tensors[4*64*64],identity=tensors[5*64*64];
                polynomial.SetOrgShape(64,64,64,64,64);polynomial.SetTail(64,64,64);
                // Five exact finite-product factors, no convergence shortcut.
                for(uint32_t level=0;level<5;++level){
                    polynomial.SetTensorA(rc);polynomial.SetTensorB(rc,false);
                    polynomial.IterateAll(rn);polynomial.End();PipeBarrier<PIPE_ALL>();
                    // All old P elements are in the complete input tile
                    // before Fixpipe atomically adds this tile's product.
                    polynomial.SetTensorA(pc);polynomial.SetTensorB(rn,false);
                    polynomial.IterateAll(pc,1);polynomial.End();PipeBarrier<PIPE_ALL>();
                    auto rt=rc;rc=rn;rn=rt;
                }
                uint32_t low=panel*64,number=data.n-2-low;number=number<64?number:64;
                update.SetOrgShape(64,data.stride,64,64,data.stride);update.SetTail(number,data.stride,number);
                update.SetTensorA(pc);update.SetTensorB(hist[low*data.stride],false);
                update.IterateAll(hist[uint64_t(2*data.n+5)*data.stride+low*data.stride]);update.End();PipeBarrier<PIPE_ALL>();
            }
        }
        // Native AIC-only barrier: every launched AIC, including an empty
        // final group, finishes panel construction before any Q reads Z.
        CrossCoreSetFlag<0,PIPE_FIX>(7);CrossCoreWaitFlag(7);
        if(b>=data.batch)continue;
        auto hist=work[uint64_t(b)*perMatrix],z=hist[uint64_t(2*data.n+5)*data.stride];
        auto q=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(pair)*height*data.stride];
        auto y=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*data.stride+uint64_t(pair)*height*64];
        CrossCoreWaitFlag(10);
        for(int low=int(first);low>=0;low-=64){
            uint32_t number=data.n-2-uint32_t(low);number=number<64?number:64;
            projection.SetOrgShape(height,64,data.stride,data.stride,64);projection.SetTail(height,number,data.stride);
            projection.SetTensorA(q);projection.SetTensorB(z[uint32_t(low)*data.stride],true);
            projection.IterateAll(y);projection.End();PipeBarrier<PIPE_ALL>();
            update.SetOrgShape(height,data.stride,64,64,data.stride);update.SetTail(height,data.stride,number);
            update.SetTensorA(y);update.SetTensorB(hist[uint32_t(low)*data.stride],false);
            update.IterateAll(q,1);update.End();PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<2,PIPE_FIX>(9);
        CrossCoreWaitFlag(6);
    }
}
__aicore__ constexpr MatmulConfig BackConfig16(bool projection){
    auto c=GetNormalConfig(false,false,false,BatchMode::BATCH_LESS_THAN_L1,true,IterateOrder::ORDER_M,ScheduleType::INNER_PRODUCT,false,true);
    c.singleCoreM=128;c.singleCoreN=projection?16:512;c.singleCoreK=projection?512:16;
    c.basicM=32;c.basicN=projection?16:64;c.basicK=projection?64:16;c.enableSetBias=false;
    return c;
}
constexpr auto BACK16_PROJ_CFG=BackConfig16(true),BACK16_UPDATE_CFG=BackConfig16(false);
constexpr auto BACK16_PROJ_TILING=GetMatmulApiTiling<CubeBackA,CubeBackBT,CubeBackC,CubeBackC>(BACK16_PROJ_CFG);
constexpr auto BACK16_UPDATE_TILING=GetMatmulApiTiling<CubeBackA,CubeBackB,CubeBackC,CubeBackC>(BACK16_UPDATE_CFG);
__aicore__ inline void CubeBackProcess16(GM_ADDR workspace,LinalgEighTilingData data){
    uint32_t padded=(data.rows+7)/8*8,physicalTeam=(data.team+1)/2*2;
    if(!(data.n>25 && data.stride>=64 && data.stride%64==0 && padded>=16))return;
    TPipe cubePipe;
    MatmulImpl<CubeBackA,CubeBackBT,CubeBackC,CubeBackC,BACK16_PROJ_TILING> projection;
    MatmulImpl<CubeBackA,CubeBackB,CubeBackC,CubeBackC,BACK16_UPDATE_TILING> update;
    projection.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    update.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    projection.SetHF32(false);update.SetHF32(false);
    uint32_t pairCount=physicalTeam/2,group=GetBlockIdx()/pairCount,pair=GetBlockIdx()%pairCount;
    uint32_t height=2*padded,first=(data.n-3)/16*16;
    uint64_t perMatrix=uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*(data.stride+16);
    GlobalTensor<float> work;work.SetGlobalBuffer((__gm__ float*)workspace);
    for(uint32_t step=0;step<(data.batch+data.groups-1)/data.groups;++step){
        uint32_t b=group<data.groups?step*data.groups+group:data.batch;if(b>=data.batch)continue;
        auto hist=work[uint64_t(b)*perMatrix],z=hist[uint64_t(2*data.n+5)*data.stride];
        auto q=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(pair)*height*data.stride];
        auto y=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*data.stride+uint64_t(pair)*height*16];
        CrossCoreWaitFlag(10);
        for(int low=int(first);low>=0;low-=16){
            uint32_t number=data.n-2-uint32_t(low);number=number<16?number:16;
            projection.SetOrgShape(height,16,data.stride,data.stride,16);projection.SetTail(height,number,data.stride);
            projection.SetTensorA(q);projection.SetTensorB(z[uint32_t(low)*data.stride],true);
            projection.IterateAll(y);projection.End();PipeBarrier<PIPE_ALL>();
            update.SetOrgShape(height,data.stride,16,16,data.stride);update.SetTail(height,data.stride,number);
            update.SetTensorA(y);update.SetTensorB(hist[uint32_t(low)*data.stride],false);
            update.IterateAll(q,1);update.End();PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<2,PIPE_FIX>(9);
        CrossCoreWaitFlag(6);
    }
}

// linalg_eigh.cpp
using namespace AscendC;

// The generated mixed launch clears KFC message headers and UB availability
// entries. Direct MatmulImpl does not use the KFC UB maps or client queues.
#if __NPU_ARCH__ == 2201
static_assert(__builtin_offsetof(AscendC::SysWorkspaceDesc,ubMap)<=LINALG_EIGH_CUBE_PROTOCOL_BYTES,
    "Cube launch protocol exceeds the reserved prefix");
#endif


template<typename InputType,uint32_t Pitch>
__aicore__ inline void RunPrivateEigh(GM_ADDR input,GM_ADDR values,GM_ADDR vectors,LinalgEighTilingData data){
    CompleteResidentEigh::Solver<InputType,Pitch,0,0,true,true,false,true> solver;
    solver.Init(input,values,vectors,nullptr,data);solver.Process();
}

// Orders one and two have complete analytic eigensystems. Each AIV owns whole
// matrices, so this path needs neither cross-core barriers nor a workspace.
__aicore__ inline float TinyPositiveRoot(float value){
    if(value<=0.0f)return 0.0f;
    union Bits {float value;uint32_t bits;};
    Bits estimate;estimate.value=value;
    // Halve the IEEE-754 exponent to seed Newton iteration; 0x1fc00000
    // restores half of the float32 exponent bias after the bit shift.
    estimate.bits=(estimate.bits>>1)+0x1fc00000U;
    float root=estimate.value;
    for(uint32_t iteration=0;iteration<6;++iteration)root=0.5f*(root+value/root);
    return root;
}

__aicore__ inline void RunAnalyticTinyEigh(GM_ADDR inputAddress,GM_ADDR valueAddress,
    GM_ADDR vectorAddress,LinalgEighTilingData data){
    TPipe pipe;TBuf<QuePosition::VECCALC> storage;pipe.InitBuffer(storage,96);
    auto inputLocal=storage.Get<float>();auto valuesLocal=inputLocal[8],vectorsLocal=inputLocal[16];
    GlobalTensor<float> input,values,vectors;
    input.SetGlobalBuffer((__gm__ float*)inputAddress);
    values.SetGlobalBuffer((__gm__ float*)valueAddress);
    vectors.SetGlobalBuffer((__gm__ float*)vectorAddress);
    for(uint32_t matrix=GetBlockIdx();matrix<data.batch;matrix+=GetBlockNum()){
        DataCopyPad(inputLocal,input[uint64_t(matrix)*data.n*data.n],
            DataCopyExtParams{1,data.n*data.n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        if(data.n==1){valuesLocal.SetValue(0,inputLocal.GetValue(0));vectorsLocal.SetValue(0,1.0f);}
        else{
            const float a=inputLocal.GetValue(0),c=inputLocal.GetValue(3);
            const float b=inputLocal.GetValue(data.upper?1:2);
            const float aa=a<0.0f?-a:a,bb=b<0.0f?-b:b,cc=c<0.0f?-c:c;
            const float maximum=aa>bb?(aa>cc?aa:cc):(bb>cc?bb:cc);
            union Bits {float value;uint32_t bits;};Bits magnitude;magnitude.value=maximum;
            magnitude.bits&=0x7f800000U;
            // A normal power-of-two scale preserves every input significand.
            // Clamp the scale for subnormal input instead of dividing by zero.
            if(magnitude.bits==0)magnitude.bits=0x00800000U;
            const float scale=magnitude.value,inverseScale=1.0f/scale;
            const ExtendedFloat diagonalA(a*inverseScale),diagonalC(c*inverseScale),offDiagonal(b*inverseScale);
            const ExtendedFloat center=(diagonalA+diagonalC).Scale(0.5f);
            const ExtendedFloat difference=(diagonalA-diagonalC).Scale(0.5f);
            const ExtendedFloat squared=difference*difference+offDiagonal*offDiagonal;
            ExtendedFloat radius(TinyPositiveRoot(squared.Float()));
            radius=radius+(squared-radius*radius)/radius.Scale(2.0f);
            const ExtendedFloat determinant=diagonalA*diagonalC-offDiagonal*offDiagonal;
            ExtendedFloat lower,upper;
            // Compute the root nearest zero through the determinant. Direct
            // center-radius subtraction loses it for almost singular matrices.
            if(center.Float()>=0.0f){upper=center+radius;lower=determinant/upper;}
            else{lower=center-radius;upper=determinant/lower;}
            float first=1.0f,second=1.0f;
            if(a<=c)second=-(offDiagonal/(diagonalC-lower)).Float();
            else first=-(offDiagonal/(diagonalA-lower)).Float();
            const float length=TinyPositiveRoot(first*first+second*second);
            first/=length;second/=length;
            valuesLocal.SetValue(0,lower.Scale(scale).Float());valuesLocal.SetValue(1,upper.Scale(scale).Float());
            vectorsLocal.SetValue(0,first);vectorsLocal.SetValue(1,-second);
            vectorsLocal.SetValue(2,second);vectorsLocal.SetValue(3,first);
            for(uint32_t column=0;column<2;++column){
                float top=vectorsLocal.GetValue(column),bottom=vectorsLocal.GetValue(2+column);
                float topMagnitude=top<0.0f?-top:top,bottomMagnitude=bottom<0.0f?-bottom:bottom;
                float pivot=topMagnitude>=bottomMagnitude?top:bottom;
                if(pivot<0.0f){vectorsLocal.SetValue(column,-top);vectorsLocal.SetValue(2+column,-bottom);}
            }
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(values[uint64_t(matrix)*data.n],valuesLocal,DataCopyExtParams{1,data.n*4,0,0,0});
        DataCopyPad(vectors[uint64_t(matrix)*data.n*data.n],vectorsLocal,DataCopyExtParams{1,data.n*data.n*4,0,0,0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
}


template <typename DT_A,uint32_t USE_CUBE>
__global__ __aicore__ void linalg_eigh(GM_ADDR A,GM_ADDR eigenvalues,GM_ADDR eigenvectors,GM_ADDR workspace,GM_ADDR tiling){
    REGISTER_TILING_DEFAULT(LinalgEighTilingData);GET_TILING_DATA_WITH_STRUCT(LinalgEighTilingData,data,tiling);
    if constexpr(USE_CUBE==0){
        if ASCEND_IS_AIV {
            if(data.n<=2){RunAnalyticTinyEigh(A,eigenvalues,eigenvectors,data);return;}
            if(data.vectorAlgorithm==static_cast<uint32_t>(EighVectorAlgorithm::ResidentSmall)){
                if(GetBlockIdx()>=data.groups*data.team)return;
                // Only physical row-pitch buckets remain compile-time values.
                // Complete and partial matrices share the same runtime order.
                switch(data.stride){
                    case 8:RunPrivateEigh<DT_A,8>(A,eigenvalues,eigenvectors,data);break;
                    case 16:RunPrivateEigh<DT_A,16>(A,eigenvalues,eigenvectors,data);break;
                    case 24:RunPrivateEigh<DT_A,24>(A,eigenvalues,eigenvectors,data);break;
                    default:RunPrivateEigh<DT_A,32>(A,eigenvalues,eigenvectors,data);break;
                }
                return;
            }
            if(data.vectorAlgorithm==static_cast<uint32_t>(EighVectorAlgorithm::ResidentMedium)){
                CompleteResidentEigh::Solver<DT_A,64,0,0,true,true,true> solver;
                solver.Init(A,eigenvalues,eigenvectors,GetSysWorkSpacePtr(),data);solver.Process();return;
            }
        }
    }else{
        if ASCEND_IS_AIC {
            if constexpr(USE_CUBE==1)CubeBackProcess16(GetSysWorkSpacePtr()+LINALG_EIGH_CUBE_PROTOCOL_BYTES,data);
            else CubeBackProcess(GetSysWorkSpacePtr()+LINALG_EIGH_CUBE_PROTOCOL_BYTES,data);
            return;
        }
    }
    if ASCEND_IS_AIV {
        if constexpr(USE_CUBE==0){
            // These pitches determine vector repeat strides and UB layout.
            // Every other pitch uses the same dynamic distributed solver.
            switch(data.stride){
                case 64:{KernelLinalgEigh<DT_A,64,0,0,false,true> op;op.Init(A,eigenvalues,eigenvectors,GetSysWorkSpacePtr(),data);op.Process();break;}
                case 128:{KernelLinalgEigh<DT_A,128,0,0,false,true> op;op.Init(A,eigenvalues,eigenvectors,GetSysWorkSpacePtr(),data);op.Process();break;}
                case 256:{KernelLinalgEigh<DT_A,256,0,0,false,true> op;op.Init(A,eigenvalues,eigenvectors,GetSysWorkSpacePtr(),data);op.Process();break;}
                case 512:{KernelLinalgEigh<DT_A,512,0,0,false,true> op;op.Init(A,eigenvalues,eigenvectors,GetSysWorkSpacePtr(),data);op.Process();break;}
                default:{KernelLinalgEigh<DT_A,0,0,0,false,true> op;op.Init(A,eigenvalues,eigenvectors,GetSysWorkSpacePtr(),data);op.Process();break;}
            }
        }else{
            // Cube variants need different core protocols, while their row
            // pitch and matrix order remain runtime values within each entry.
            KernelLinalgEigh<DT_A,0,USE_CUBE,0,false,true> op;
            op.Init(A,eigenvalues,eigenvectors,GetSysWorkSpacePtr()+LINALG_EIGH_CUBE_PROTOCOL_BYTES,data);op.Process();
        }
    }
}
