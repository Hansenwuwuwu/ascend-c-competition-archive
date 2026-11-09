#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"
using namespace AscendC;

// Resident column-major elimination. The row permutation is logical, not copied.
class KernelColumnDet {
public:
 __aicore__ inline ~KernelColumnDet(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void MakeStrides(LocalTensor<int32_t> target,uint32_t count,uint32_t step){
  for(uint32_t i=0;i<8;++i)target.SetValue(i,int32_t(i*step));
  Fence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<count;filled*=2){
   uint32_t len=count-filled<filled?count-filled:filled;
   Adds(target[filled],target,int32_t(filled*step),len);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;pitch=n<=8?8:n<=16?16:n<=32?32:n<=64?72:(n+7)/8*8;cols=(n+7)/8*8;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  uint32_t ioCount=n<=64?cols*pitch:8*pitch,offset=0;
  mat=LocalTensor<float>(TPosition::VECCALC,offset,cols*pitch);offset+=cols*pitch*4;
  loaded=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  transposed=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  absolute=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  reduced=LocalTensor<float>(TPosition::VECCALC,offset,8);offset+=32;
  pivot=LocalTensor<float>(TPosition::VECCALC,offset,cols);offset+=cols*4;
  spread=LocalTensor<float>(TPosition::VECCALC,offset,cols*8);offset+=cols*32;
  bases=LocalTensor<int32_t>(TPosition::VECCALC,offset,cols);offset+=cols*4;
  result=LocalTensor<float>(TPosition::VECCALC,offset,8);
  MakeStrides(bases,cols,pitch*4);
  auto offsets=indices.ReinterpretCast<int32_t>();
  uint32_t size=n<=64?cols*pitch:n*8,rows=n<=64?pitch:8,stride=n<=64?n:pitch;
  MakeStrides(offsets,rows,stride*4);
  for(uint32_t filled=rows;filled<size;filled*=2){
   uint32_t count=size-filled<filled?size-filled:filled;
   Adds(offsets[filled],offsets,int32_t(filled/rows*4),count);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Load(uint64_t offset){
  if(n<=64){
   Fence<HardEvent::S_V>();Duplicate(loaded,0.0f,cols*pitch);Fence<HardEvent::V_MTE2>();
   DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
   Fence<HardEvent::MTE2_V>();Gather(mat,loaded,indices,0,cols*pitch);PipeBarrier<PIPE_V>();return;
  }
  Fence<HardEvent::S_V>();Duplicate(mat,0.0f,cols*pitch);PipeBarrier<PIPE_V>();
  for(uint32_t base=0;base<n;base+=8){
   uint32_t rows=n-base<8?n-base:8;
   Duplicate(loaded,0.0f,8*pitch);Fence<HardEvent::V_MTE2>();
   DataCopyExtParams cp{static_cast<uint16_t>(rows),n*4,0,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
   DataCopyPad(loaded,input[offset+uint64_t(base)*n],cp,pad);Fence<HardEvent::MTE2_V>();
   Gather(transposed,loaded,indices,0,n*8);PipeBarrier<PIPE_V>();
   DataCopyParams dest{static_cast<uint16_t>(n),1,0,static_cast<uint16_t>((pitch-8)/8)};
   DataCopy(mat[base],transposed,dest);PipeBarrier<PIPE_V>();
  }
 }
 union Bits {float f;uint32_t u;};
 __aicore__ inline void Multiply(float value,float& product,int32_t& exponent){
  Bits b;b.f=value;int32_t e=int32_t((b.u>>23)&255)-127;
  if(e==-127){b.f=value*16777216.0f;e=int32_t((b.u>>23)&255)-127-24;}
  b.u=(b.u&0x807fffffu)|0x3f800000u;product*=b.f;exponent+=e;
  if(product>=2.0f||product<=-2.0f){product*=.5f;++exponent;}
 }
 __aicore__ inline float Finish(float product,int32_t exponent){
  Bits b;if(exponent>127){b.u=product<0?0xff800000u:0x7f800000u;return b.f;}
  if(exponent>=-126){b.u=uint32_t(exponent+127)<<23;return product*b.f;}
  if(exponent < -150)return 0.0f;
  b.u=uint32_t(exponent+151)<<23;return (product*b.f)*5.960464477539063e-8f;
 }
 __aicore__ inline uint32_t Parity(uint64_t x){
  x^=x>>32;x^=x>>16;x^=x>>8;x^=x>>4;return (0x6996u>>(uint32_t(x)&15))&1;
 }
 __aicore__ inline void Process(){
  if(GetBlockIdx()>=cores)return;
  for(uint64_t batch=GetBlockIdx();batch<batches;batch+=cores){
   Load(batch*n*n);
   uint64_t active[2]={n>=64?~uint64_t(0):(uint64_t(1)<<n)-1,n<=64?0:(n==128?~uint64_t(0):(uint64_t(1)<<(n-64))-1)};
   float product=1.0f;int32_t exponent=0;bool singular=false;
   for(uint32_t p=0;p<n;++p){
    factor=mat[p*pitch];
    Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
    uint32_t best=0;float largest=0.0f;
    for(uint32_t block=0;block<(n+63)/64;++block){
     if(!active[block])continue;
     uint64_t mask[2]={active[block],0};
     WholeReduceMax(reduced,absolute[block*64],mask,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);Fence<HardEvent::V_S>();
     float value=reduced.GetValue(0);
     if(value>largest){largest=value;best=block*64+reduced.ReinterpretCast<uint32_t>().GetValue(1);}
    }
    if(largest==0.0f){singular=true;break;}
    float diagonal=factor.GetValue(best);
    uint32_t remaining=n-p-1;
    if(remaining){
    Gather(pivot,mat[(p+1)*pitch],bases.ReinterpretCast<uint32_t>(),best*4,remaining);
    factor.SetValue(best,0.0f);Fence<HardEvent::S_V>();
    Bits normalization;normalization.f=diagonal;
    float divisor=diagonal;
    if((normalization.u&0x7f800000u)==0u){
     Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
     divisor*=16777216.0f;
    }
    Muls(factor,factor,-1.0f/divisor,pitch);PipeBarrier<PIPE_V>();
    Brcb(spread,pivot,(remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
    for(uint32_t r=0;r<n;r+=64){
     uint32_t count=n-r<64?n-r:64;
     MulAddDst(mat[(p+1)*pitch+r],factor[r],spread,count,remaining,
       BinaryRepeatParams{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
    }
    }
    uint32_t block=best/64,bit=best%64;
    uint32_t odd=Parity(active[block]&((uint64_t(1)<<bit)-1));
    if(block)odd^=Parity(active[0]);
    if(odd)product=-product;
    active[block]&=~(uint64_t(1)<<bit);
    Multiply(diagonal,product,exponent);
    PipeBarrier<PIPE_V>();
   }
   Fence<HardEvent::V_S>();result.SetValue(0,singular?0.0f:Finish(product,exponent));Fence<HardEvent::S_MTE3>();
   DataCopyExtParams cp{1,4,0,0,0};DataCopyPad(output[batch],result,cp);Fence<HardEvent::MTE3_S>();
  }
 }
private:
 LocalTensor<float> mat,loaded,transposed,factor,absolute,reduced,pivot,spread,result;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 uint32_t n,pitch,cols,cores;uint64_t batches;
};

class KernelMediumDet {
public:
 __aicore__ inline ~KernelMediumDet(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void MakeStrides(LocalTensor<int32_t> target,uint32_t count,uint32_t step){
  for(uint32_t i=0;i<8;++i)target.SetValue(i,int32_t(i*step));
  Fence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<count;filled*=2){
   uint32_t len=count-filled<filled?count-filled:filled;
   Adds(target[filled],target,int32_t(filled*step),len);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;cols=(n+7)/8*8;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  uint32_t ioCount=n<=64?cols*pitch:8*pitch,offset=0;
  mat=LocalTensor<float>(TPosition::VECCALC,offset,cols*pitch);offset+=cols*pitch*4;
  loaded=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  transposed=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  absolute=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  reduced=LocalTensor<float>(TPosition::VECCALC,offset,8);offset+=32;
  pivot=LocalTensor<float>(TPosition::VECCALC,offset,cols);offset+=cols*4;
  spread=LocalTensor<float>(TPosition::VECCALC,offset,cols*8);offset+=cols*32;
  bases=LocalTensor<int32_t>(TPosition::VECCALC,offset,128);offset+=512;
  paired=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  pairSpread=LocalTensor<float>(TPosition::VECCALC,offset,1024);offset+=4096;
  result=LocalTensor<float>(TPosition::VECCALC,offset,8);
  for(uint32_t i=0;i<8;++i)bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*pitch*4)|(uint64_t(i*pitch*4+4)<<32));
  Fence<HardEvent::S_V>();
  for(uint32_t filled=16;filled<128;filled*=2){Adds(bases[filled],bases,int32_t(filled/2*pitch*4),filled);PipeBarrier<PIPE_V>();}

 }
 __aicore__ inline void Load(uint64_t offset){
  if((n&7)==0){
   // Live row words are contiguous in GM; inactive row padding is never consumed.
   Fence<HardEvent::V_MTE2>();
   DataCopy(loaded,input[offset],n*n);
   Fence<HardEvent::MTE2_V>();
   DataCopy(mat,loaded,DataCopyParams{uint16_t(n),uint16_t(n/8),0,uint16_t((pitch-n)/8)});
   PipeBarrier<PIPE_V>();
   return;
  }
  // The large contiguous transfer runs while the shape-only address map is built.
  Fence<HardEvent::V_MTE2>();
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},
    DataCopyPadExtParams<float>{true,0,uint8_t((8-n*n%8)%8),0.0f});
  Duplicate(loaded[(n*n+7)/8*8],0.0f,64);
  if(!flatIndicesReady){
   MakeStrides(indices.ReinterpretCast<int32_t>(),n*64,4);
   auto temporary=transposed.ReinterpretCast<int32_t>();
   ShiftRight(temporary,indices.ReinterpretCast<int32_t>(),8,n*64);PipeBarrier<PIPE_V>();
   Muls(temporary,temporary,(int32_t(n)-64)*4,n*64);PipeBarrier<PIPE_V>();
   Add(indices.ReinterpretCast<int32_t>(),indices.ReinterpretCast<int32_t>(),temporary,n*64);PipeBarrier<PIPE_V>();
   flatIndicesReady=true;
  }
  Fence<HardEvent::MTE2_V>();
  Gather(transposed,loaded,indices,0,n*64);PipeBarrier<PIPE_V>();
  DataCopy(mat,transposed,DataCopyParams{uint16_t(n),8,0,1});PipeBarrier<PIPE_V>();
 }
 union Bits {float f;uint32_t u;};
 __aicore__ inline void Multiply(float value,float& product,int32_t& exponent){product*=value;}
 __aicore__ inline float Finish(float product,int32_t exponent){return product;}
 __aicore__ inline uint32_t Parity(uint64_t x){
  x^=x>>32;x^=x>>16;x^=x>>8;x^=x>>4;return (0x6996u>>(uint32_t(x)&15))&1;
 }
 __aicore__ inline void Process(){
  if(GetBlockIdx()>=cores)return;
  for(uint64_t batch=GetBlockIdx();batch<batches;batch+=cores){
   Load(batch*n*n);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;bool singular=false;
   for(uint32_t p=0;p<n;++p){
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if((p&1)==0){
     Fence<HardEvent::V_S>();diagonal=factor.GetValue(p);
     if(remaining){
      below=factor.GetValue(p+1);
      cross=mat[(p+1)*pitch].GetValue(p);
      future=mat[(p+1)*pitch].GetValue(p+1);
     }
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((p&1)==0)Gather(paired,mat[(p+1)*pitch],bases.ReinterpretCast<uint32_t>(),p*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(p&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     if((p&1)==0){Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();}
     MulAddDst<float,float,false>(mat[(p+1)*pitch],factor,pairSpread[(p&1)?24:0],uint64_t(64),remaining,
       BinaryRepeatParams{1,1,0,9,0,2});
     if((p&1)==0){
      float normalizedBelow=below*inverse;
      cachedDiagonal=normalizedBelow*cross+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      Axpy(pairSpread[8],pairSpread,normalizedBelow,uint64_t(64),(remaining+7)/8,UnaryRepeatParams{2,2,16,16});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
   }
   Fence<HardEvent::V_S>();result.SetValue(0,singular?0.0f:product);Fence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0});Fence<HardEvent::MTE3_S>();
  }
 }
private:
 LocalTensor<float> mat,loaded,transposed,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 bool flatIndicesReady=false;
 static constexpr uint32_t pitch=72;uint32_t n,cols,cores;uint64_t batches;
};

// Distributed resident columns; one packet publishes four consecutive pivots.
class KernelCooperativeDet {
public:
 template<HardEvent E> __aicore__ inline void Fence(){event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);}
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,GM_ADDR W,const LinalgDetTilingData& t){
  n=t.n;pitch=(n+7)/8*8;cores=t.cores;batches=t.batches;workers=n<=256&&t.resident==4?4:8;
  lanes=workers+(workers==4?1:0);sink=workers==4?workers:0;lane=GetBlockIdx()%lanes;group=GetBlockIdx()/lanes;groups=cores/lanes;
  columns=(n+workers*8-1)/(workers*8)*8;begin=lane*columns;end=begin+columns<n?begin+columns:n;
  count=begin<n?end-begin:0;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  work.SetGlobalBuffer((__gm__ float*)W+uint64_t(group)*16*(pitch+8));
  uint32_t offset=0;
  mat=LocalTensor<float>(TPosition::VECCALC,offset,columns*pitch);offset+=columns*pitch*4;
  loaded=LocalTensor<float>(TPosition::VECCALC,offset,8*pitch);offset+=8*pitch*4;
  transposed=LocalTensor<float>(TPosition::VECCALC,offset,8*columns);offset+=8*columns*4;
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,8*columns);offset+=8*columns*4;
  packet=LocalTensor<float>(TPosition::VECCALC,offset,8*(pitch+8));offset+=8*(pitch+8)*4;
  absolute=LocalTensor<float>(TPosition::VECCALC,offset,pitch);offset+=pitch*4;
  active=LocalTensor<float>(TPosition::VECCALC,offset,pitch);offset+=pitch*4;
  reduced=LocalTensor<float>(TPosition::VECCALC,offset,64);offset+=256;
  pivot=LocalTensor<float>(TPosition::VECCALC,offset,columns);offset+=columns*4;
  spread=LocalTensor<float>(TPosition::VECCALC,offset,columns*8);offset+=columns*32;
  bases=LocalTensor<int32_t>(TPosition::VECCALC,offset,columns);offset+=columns*4;
  result=LocalTensor<float>(TPosition::VECCALC,offset,8);
  for(uint32_t i=0;i<8;++i)bases.SetValue(i,int32_t(i*pitch*4));
  Fence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<columns;filled*=2){
   uint32_t len=columns-filled<filled?columns-filled:filled;
   Adds(bases[filled],bases,int32_t(filled*pitch*4),len);PipeBarrier<PIPE_V>();
  }
  if(!count)return;
  auto off=indices.ReinterpretCast<int32_t>();
  for(uint32_t i=0;i<8;++i)off.SetValue(i,int32_t((begin+i*pitch)*4));
  Fence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<count*8;filled*=2){
   uint32_t len=count*8-filled<filled?count*8-filled:filled;
   Adds(off[filled],off,int32_t(filled/8*4),len);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Load(uint64_t batch){
  Fence<HardEvent::S_V>();Duplicate(mat,0.0f,columns*pitch);PipeBarrier<PIPE_V>();
  for(uint32_t r=0;r<n;r+=8){uint32_t rows=n-r<8?n-r:8;
   Duplicate(loaded,0.0f,8*pitch);Fence<HardEvent::V_MTE2>();
   DataCopyPad(loaded,input[(batch*n+r)*n],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
   Fence<HardEvent::MTE2_V>();Gather(transposed,loaded,indices,0,count*8);PipeBarrier<PIPE_V>();
   DataCopy(mat[r],transposed,DataCopyParams{static_cast<uint16_t>(count),1,0,static_cast<uint16_t>((pitch-8)/8)});PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void NormalizeOwner(LocalTensor<float> factor,LocalTensor<float> tile,uint32_t cols,uint32_t row,float diagonal){
  // Pivot() fenced all prior tile updates. Each column coefficient read is disjoint from preceding Axpy writes.
  SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,pitch);
  Bits normalization;normalization.f=diagonal;
  float divisor=diagonal;
  if((normalization.u&0x7f800000u)==0u){
   Muls<float,false>(factor,factor,16777216.0f,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
   PipeBarrier<PIPE_V>();divisor*=16777216.0f;
  }
  Muls<float,false>(factor,factor,-1.0f/divisor,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
  PipeBarrier<PIPE_V>();
  for(uint32_t col=0;col<cols;++col){
   float coefficient=tile[col*pitch].GetValue(row);
   Axpy<float,float,false>(tile[col*pitch],factor,coefficient,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
  }
  ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Update(LocalTensor<float> dst,uint32_t cols,LocalTensor<float> factor,uint32_t row){
  if(!cols)return;
  Gather(pivot,dst,bases.ReinterpretCast<uint32_t>(),row*4,cols);PipeBarrier<PIPE_V>();
  Brcb(spread,pivot,(cols+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
  uint32_t full=n/64*64;
  SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
  for(uint32_t r=0;r<full;r+=64){
   MulAddDst<float,float,false>(dst[r],factor[r],spread,uint64_t(64),cols,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
  }
  if(full<n)MulAddDst(dst[full],factor[full],spread,n-full,cols,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
  PipeBarrier<PIPE_V>();
 }
 __aicore__ inline uint32_t Pivot(LocalTensor<float> factor,float& largest){
  Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
  WholeReduceMax(reduced,absolute,64,pitch/64,4,1,8,ReduceOrder::ORDER_VALUE_INDEX);
  if(pitch%64)WholeReduceMax(reduced[(pitch/64)*8],absolute[(pitch/64)*64],pitch%64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
  PipeBarrier<PIPE_V>();uint64_t mask[2]={0,0};for(uint32_t b=0;b<(n+63)/64;++b)mask[0]|=uint64_t(1)<<(b*8);
  WholeReduceMax(absolute,reduced,mask,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);Fence<HardEvent::V_S>();
  largest=absolute.GetValue(0);uint32_t slot=absolute.ReinterpretCast<uint32_t>().GetValue(1);
  return slot*8+reduced.ReinterpretCast<uint32_t>().GetValue(slot+1);
 }
 union Bits {float f;uint32_t u;};
 __aicore__ inline void Multiply(float value,float& product,int32_t& exponent){
  Bits b;b.f=value;int32_t e=int32_t((b.u>>23)&255)-127;
  if(e==-127){b.f=value*16777216.0f;e=int32_t((b.u>>23)&255)-127-24;}
  b.u=(b.u&0x807fffffu)|0x3f800000u;product*=b.f;exponent+=e;if(product>=2.0f||product<=-2.0f){product*=.5f;++exponent;}
 }
 __aicore__ inline float Finish(float product,int32_t exponent){
  Bits b;if(exponent>127){b.u=product<0?0xff800000u:0x7f800000u;return b.f;}
  if(exponent>=-126){b.u=uint32_t(exponent+127)<<23;return product*b.f;}if(exponent < -150)return 0;
  b.u=uint32_t(exponent+151)<<23;return product*b.f*5.960464477539063e-8f;
 }
 __aicore__ inline uint32_t Parity(uint64_t x){x^=x>>32;x^=x>>16;x^=x>>8;x^=x>>4;return(0x6996u>>(uint32_t(x)&15))&1;}
 __aicore__ inline void Process(){
  uint64_t waves=(batches+groups-1)/groups;
  for(uint64_t wave=0;wave<waves;++wave){
   uint64_t batch=wave*groups+group;bool valid=batch<batches,singular=false;
   float product=1;int32_t exponent=0;uint64_t remainingRows[8]={};
   for(uint32_t b=0;b<(n+63)/64;++b){uint32_t count=n-b*64;remainingRows[b]=count>=64?~uint64_t(0):(uint64_t(1)<<count)-1;}
   if(valid){if(count)Load(batch);Duplicate(active,0.0f,pitch);PipeBarrier<PIPE_V>();Duplicate(active,1.0f,n);PipeBarrier<PIPE_V>();}
   for(uint32_t panel=0;panel<n;panel+=8){
    uint32_t stop=panel+8<n?panel+8:n,steps=stop-panel,slot=((panel/8)%2)*8*(pitch+8);
    bool owner=panel>=begin&&panel<end;
    if(valid&&!singular&&owner){
     Fence<HardEvent::S_V>();Duplicate(packet,0.0f,8*(pitch+8));PipeBarrier<PIPE_V>();
     for(uint32_t p=panel;p<stop;++p){
      auto factor=packet[(p-panel)*(pitch+8)];
      Fence<HardEvent::S_V>();Mul(factor,mat[(p-begin)*pitch],active,pitch);PipeBarrier<PIPE_V>();
      float largest;uint32_t row=Pivot(factor,largest);if(largest==0.0f)break;
      float diagonal=factor.GetValue(row);factor.SetValue(pitch,diagonal);factor.ReinterpretCast<uint32_t>().SetValue(pitch+1,row);factor.SetValue(pitch+2,1.0f);
      active.SetValue(row,0.0f);factor.SetValue(row,0.0f);Fence<HardEvent::S_V>();
      NormalizeOwner(factor,mat[(p+1-begin)*pitch],stop-p-1,row,diagonal);
     }
     Fence<HardEvent::V_MTE3>();DataCopy(work[slot],packet,steps*(pitch+8));
    }
    SyncAll();
    if(valid&&!singular){
     if(!owner){DataCopy(packet,work[slot],steps*(pitch+8));Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();}
     Fence<HardEvent::V_S>();
     for(uint32_t p=panel;p<stop;++p){
      auto factor=packet[(p-panel)*(pitch+8)];
      if(factor.GetValue(pitch+2)==0.0f){singular=true;break;}
      uint32_t row=factor.ReinterpretCast<uint32_t>().GetValue(pitch+1),block=row/64,bit=row%64;
      if(lane==sink){
      uint32_t odd=Parity(remainingRows[block]&((uint64_t(1)<<bit)-1));for(uint32_t b=0;b<block;++b)odd^=Parity(remainingRows[b]);
      if(odd)product=-product;remainingRows[block]&=~(uint64_t(1)<<bit);Multiply(factor.GetValue(pitch),product,exponent);
      }
      active.SetValue(row,0.0f);
      uint32_t first=stop>begin?stop:begin;if(first<end)Update(mat[(first-begin)*pitch],end-first,factor,row);
     }
    }
   }
   if(valid&&lane==sink){result.SetValue(0,singular?0:Finish(product,exponent));Fence<HardEvent::S_MTE3>();
    DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0});Fence<HardEvent::MTE3_S>();}
   SyncAll();
  }
 }
private:
 TPipe pipe;
 LocalTensor<float> mat,loaded,transposed,packet,absolute,active,reduced,pivot,spread,result;LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;
 GlobalTensor<float> input,output,work;uint32_t n,pitch,cores,workers,lanes,sink,lane,group,groups,columns,begin,end,count;uint64_t batches;
};

class KernelTinyDet {
public:
 __aicore__ inline ~KernelTinyDet(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  mat=LocalTensor<float>(TPosition::VECCALC,0,256);
  loaded=LocalTensor<float>(TPosition::VECCALC,1024,256);
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,2048,256);
  pivot=LocalTensor<float>(TPosition::VECCALC,3072,16);
  spread=LocalTensor<float>(TPosition::VECCALC,3136,128);
  bases=LocalTensor<int32_t>(TPosition::VECCALC,4352,32);
  paired=LocalTensor<float>(TPosition::VECCALC,4480,32);
  pairSpread=LocalTensor<float>(TPosition::VECCALC,4608,256);
  absolute=LocalTensor<float>(TPosition::VECCALC,3712,128);
  reduced=LocalTensor<float>(TPosition::VECCALC,4224,8);
  result=LocalTensor<float>(TPosition::VECCALC,4256,8);
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64+4)<<32));
  Fence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){offsets.SetValue(i,int32_t(i*n*4));bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64+4)<<32));}
  Fence<HardEvent::S_V>();
  Adds(offsets[8],offsets,int32_t(8*n*4),8);Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  #pragma unroll
  for(uint32_t filled=16;filled<256;filled*=2){
   Adds(offsets[filled],offsets,int32_t(filled/16*4),filled);PipeBarrier<PIPE_V>();
  }
  }
 }
 __aicore__ inline void Load(uint64_t offset){
  if(n==pitch){
   Fence<HardEvent::V_MTE2>();DataCopy(mat,input[offset],n*n);Fence<HardEvent::MTE2_V>();return;
  }
  Fence<HardEvent::S_V>();Duplicate(loaded,0.0f,cols*pitch);Fence<HardEvent::V_MTE2>();
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
  Fence<HardEvent::MTE2_V>();Gather(mat,loaded,indices,0,cols*pitch);PipeBarrier<PIPE_V>();
 }
 union Bits {float f;uint32_t u;};
 __aicore__ inline void Multiply(float value,float& product,int32_t& exponent){product*=value;}
 __aicore__ inline float Finish(float product,int32_t exponent){return product;}
 __aicore__ inline uint32_t Parity(uint64_t x){
  x^=x>>8;x^=x>>4;return (0x6996u>>(uint32_t(x)&15))&1;
 }
 __aicore__ inline void Process(){
  if(GetBlockIdx()>=cores)return;
  uint32_t owners=batches<cores?uint32_t(batches):cores;
  if(GetBlockIdx()>=owners)return;
  for(uint64_t batch=GetBlockIdx()%owners;batch<batches;batch+=owners){
   Load(batch*n*n);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;bool singular=false;
   #pragma unroll
   for(uint32_t p=0;p<16;++p){
    if(p>=n)break;
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if((p&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(p/2);
      uint64_t second=mat[(p+1)*pitch].ReinterpretCast<uint64_t>().GetValue(p/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(p);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((p&1)==0)Gather(paired,mat[(p+1)*pitch],bases.ReinterpretCast<uint32_t>(),p*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(p&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((p&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(p+1)*pitch],factor,pairSpread[(p&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((p&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
   }
   Fence<HardEvent::V_S>();result.SetValue(0,singular?0.0f:product);Fence<HardEvent::S_MTE3>();
   if(GetBlockIdx()<owners)DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0});Fence<HardEvent::MTE3_S>();
  }
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t pitch=16,cols=16;uint32_t n,cores;uint64_t batches;
};


class KernelTinyFullwidthDet {
public:
 __aicore__ inline ~KernelTinyFullwidthDet(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  mat=LocalTensor<float>(TPosition::VECCALC,0,256);
  loaded=LocalTensor<float>(TPosition::VECCALC,1024,256);
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,2048,256);
  pivot=LocalTensor<float>(TPosition::VECCALC,3072,16);
  spread=LocalTensor<float>(TPosition::VECCALC,3136,128);
  bases=LocalTensor<int32_t>(TPosition::VECCALC,5760,64);
  paired=LocalTensor<float>(TPosition::VECCALC,6016,64);
  pairSpread=LocalTensor<float>(TPosition::VECCALC,6272,512);
  factorWide=LocalTensor<float>(TPosition::VECCALC,8320,64);
  absolute=LocalTensor<float>(TPosition::VECCALC,3712,128);
  reduced=LocalTensor<float>(TPosition::VECCALC,4224,8);
  result=LocalTensor<float>(TPosition::VECCALC,4256,8);
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64)<<32));
  Fence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){offsets.SetValue(i,int32_t(i*n*4));bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64)<<32));}
  Fence<HardEvent::S_V>();
  Adds(offsets[8],offsets,int32_t(8*n*4),8);Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  #pragma unroll
  for(uint32_t filled=16;filled<256;filled*=2){
   Adds(offsets[filled],offsets,int32_t(filled/16*4),filled);PipeBarrier<PIPE_V>();
  }
  }
  Adds(bases[32],bases,int32_t(4),32);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Load(uint64_t offset){
  if(n==pitch){
   Fence<HardEvent::V_MTE2>();DataCopy(mat,input[offset],n*n);Fence<HardEvent::MTE2_V>();return;
  }
  Fence<HardEvent::S_V>();Duplicate(loaded,0.0f,cols*pitch);Fence<HardEvent::V_MTE2>();
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
  Fence<HardEvent::MTE2_V>();Gather(mat,loaded,indices,0,cols*pitch);PipeBarrier<PIPE_V>();
 }
 union Bits {float f;uint32_t u;};
 __aicore__ inline void Multiply(float value,float& product,int32_t& exponent){product*=value;}
 __aicore__ inline float Finish(float product,int32_t exponent){return product;}
 __aicore__ inline uint32_t Parity(uint64_t x){
  x^=x>>8;x^=x>>4;return (0x6996u>>(uint32_t(x)&15))&1;
 }
 __aicore__ inline void Process(){
  if(GetBlockIdx()>=cores)return;
  uint32_t owners=batches<cores?uint32_t(batches):cores;
  if(GetBlockIdx()>=owners)return;
  for(uint64_t batch=GetBlockIdx()%owners;batch<batches;batch+=owners){
   Load(batch*n*n);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;bool singular=false;
   #pragma unroll
   for(uint32_t p=0;p<16;++p){
    if(p>=n)break;
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if((p&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(p/2);
      uint64_t second=mat[(p+1)*pitch].ReinterpretCast<uint64_t>().GetValue(p/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(p);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((p&1)==0)Gather(paired,mat,bases.ReinterpretCast<uint32_t>(),p*4,64);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(p&1)?cachedInverse:-1.0f/divisor;
     Muls(factorWide,factor,inverse,uint64_t(16),4,UnaryRepeatParams{1,1,2,0});PipeBarrier<PIPE_V>();
     if((p&1)==0){
      Brcb(pairSpread,paired,8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     // Retired rows may change; only live diagonal and trailing rows are consumed.
     MulAddDst(mat,factorWide,pairSpread[(p&1)?256:0],uint64_t(64),4,
       BinaryRepeatParams{1,1,1,8,0,8});
     if((p&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[256],pairSpread,float(normalizedBelow),256);
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
   }
   Fence<HardEvent::V_S>();result.SetValue(0,singular?0.0f:product);Fence<HardEvent::S_MTE3>();
   if(GetBlockIdx()<owners)DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0});Fence<HardEvent::MTE3_S>();
  }
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread,factorWide;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t pitch=16,cols=16;uint32_t n,cores;uint64_t batches;
};


class KernelTinyExplicit16 {
public:
 __aicore__ inline ~KernelTinyExplicit16(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  mat=LocalTensor<float>(TPosition::VECCALC,0,256);
  loaded=LocalTensor<float>(TPosition::VECCALC,1024,256);
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,2048,256);
  pivot=LocalTensor<float>(TPosition::VECCALC,3072,16);
  spread=LocalTensor<float>(TPosition::VECCALC,3136,128);
  bases=LocalTensor<int32_t>(TPosition::VECCALC,4352,32);
  paired=LocalTensor<float>(TPosition::VECCALC,4480,32);
  pairSpread=LocalTensor<float>(TPosition::VECCALC,4608,256);
  absolute=LocalTensor<float>(TPosition::VECCALC,3712,128);
  reduced=LocalTensor<float>(TPosition::VECCALC,4224,8);
  result=LocalTensor<float>(TPosition::VECCALC,4256,8);
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64+4)<<32));
  Fence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){offsets.SetValue(i,int32_t(i*n*4));bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64+4)<<32));}
  Fence<HardEvent::S_V>();
  Adds(offsets[8],offsets,int32_t(8*n*4),8);Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  #pragma unroll
  for(uint32_t filled=16;filled<256;filled*=2){
   Adds(offsets[filled],offsets,int32_t(filled/16*4),filled);PipeBarrier<PIPE_V>();
  }
  }
 }
 __aicore__ inline void Load(uint64_t offset){
  if(n==pitch){
   Fence<HardEvent::V_MTE2>();DataCopy(mat,input[offset],n*n);Fence<HardEvent::MTE2_V>();return;
  }
  Fence<HardEvent::S_V>();Duplicate(loaded,0.0f,cols*pitch);Fence<HardEvent::V_MTE2>();
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
  Fence<HardEvent::MTE2_V>();Gather(mat,loaded,indices,0,cols*pitch);PipeBarrier<PIPE_V>();
 }
 union Bits {float f;uint32_t u;};
 __aicore__ inline void Multiply(float value,float& product,int32_t& exponent){product*=value;}
 __aicore__ inline float Finish(float product,int32_t exponent){return product;}
 __aicore__ inline uint32_t Parity(uint64_t x){
  x^=x>>8;x^=x>>4;return (0x6996u>>(uint32_t(x)&15))&1;
 }
 __aicore__ inline void Process(){
  if(GetBlockIdx()>=cores)return;
  uint32_t owners=batches<cores?uint32_t(batches):cores;
  if(GetBlockIdx()>=owners)return;
  for(uint64_t batch=GetBlockIdx()%owners;batch<batches;batch+=owners){
   Load(batch*n*n);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;bool singular=false;
   // Explicit code generation, not a requested compiler loop unroll.
   do {
    { // pivot 0


    factor=mat[0*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-0-1;
    if((0&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(0/2);
      uint64_t second=mat[(0+1)*pitch].ReinterpretCast<uint64_t>().GetValue(0/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(0);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((0&1)==0)Gather(paired,mat[(0+1)*pitch],bases.ReinterpretCast<uint32_t>(),0*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(0&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((0&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(0+1)*pitch],factor,pairSpread[(0&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((0&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 1


    factor=mat[1*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-1-1;
    if((1&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(1/2);
      uint64_t second=mat[(1+1)*pitch].ReinterpretCast<uint64_t>().GetValue(1/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(1);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((1&1)==0)Gather(paired,mat[(1+1)*pitch],bases.ReinterpretCast<uint32_t>(),1*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(1&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((1&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(1+1)*pitch],factor,pairSpread[(1&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((1&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 2


    factor=mat[2*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-2-1;
    if((2&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(2/2);
      uint64_t second=mat[(2+1)*pitch].ReinterpretCast<uint64_t>().GetValue(2/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(2);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((2&1)==0)Gather(paired,mat[(2+1)*pitch],bases.ReinterpretCast<uint32_t>(),2*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(2&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((2&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(2+1)*pitch],factor,pairSpread[(2&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((2&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 3


    factor=mat[3*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-3-1;
    if((3&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(3/2);
      uint64_t second=mat[(3+1)*pitch].ReinterpretCast<uint64_t>().GetValue(3/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(3);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((3&1)==0)Gather(paired,mat[(3+1)*pitch],bases.ReinterpretCast<uint32_t>(),3*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(3&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((3&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(3+1)*pitch],factor,pairSpread[(3&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((3&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 4


    factor=mat[4*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-4-1;
    if((4&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(4/2);
      uint64_t second=mat[(4+1)*pitch].ReinterpretCast<uint64_t>().GetValue(4/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(4);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((4&1)==0)Gather(paired,mat[(4+1)*pitch],bases.ReinterpretCast<uint32_t>(),4*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(4&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((4&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(4+1)*pitch],factor,pairSpread[(4&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((4&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 5


    factor=mat[5*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-5-1;
    if((5&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(5/2);
      uint64_t second=mat[(5+1)*pitch].ReinterpretCast<uint64_t>().GetValue(5/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(5);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((5&1)==0)Gather(paired,mat[(5+1)*pitch],bases.ReinterpretCast<uint32_t>(),5*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(5&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((5&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(5+1)*pitch],factor,pairSpread[(5&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((5&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 6


    factor=mat[6*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-6-1;
    if((6&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(6/2);
      uint64_t second=mat[(6+1)*pitch].ReinterpretCast<uint64_t>().GetValue(6/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(6);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((6&1)==0)Gather(paired,mat[(6+1)*pitch],bases.ReinterpretCast<uint32_t>(),6*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(6&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((6&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(6+1)*pitch],factor,pairSpread[(6&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((6&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 7


    factor=mat[7*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-7-1;
    if((7&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(7/2);
      uint64_t second=mat[(7+1)*pitch].ReinterpretCast<uint64_t>().GetValue(7/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(7);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((7&1)==0)Gather(paired,mat[(7+1)*pitch],bases.ReinterpretCast<uint32_t>(),7*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(7&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((7&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(7+1)*pitch],factor,pairSpread[(7&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((7&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 8


    factor=mat[8*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-8-1;
    if((8&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(8/2);
      uint64_t second=mat[(8+1)*pitch].ReinterpretCast<uint64_t>().GetValue(8/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(8);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((8&1)==0)Gather(paired,mat[(8+1)*pitch],bases.ReinterpretCast<uint32_t>(),8*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(8&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((8&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(8+1)*pitch],factor,pairSpread[(8&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((8&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 9


    factor=mat[9*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-9-1;
    if((9&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(9/2);
      uint64_t second=mat[(9+1)*pitch].ReinterpretCast<uint64_t>().GetValue(9/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(9);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((9&1)==0)Gather(paired,mat[(9+1)*pitch],bases.ReinterpretCast<uint32_t>(),9*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(9&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((9&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(9+1)*pitch],factor,pairSpread[(9&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((9&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 10


    factor=mat[10*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-10-1;
    if((10&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(10/2);
      uint64_t second=mat[(10+1)*pitch].ReinterpretCast<uint64_t>().GetValue(10/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(10);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((10&1)==0)Gather(paired,mat[(10+1)*pitch],bases.ReinterpretCast<uint32_t>(),10*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(10&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((10&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(10+1)*pitch],factor,pairSpread[(10&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((10&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 11


    factor=mat[11*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-11-1;
    if((11&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(11/2);
      uint64_t second=mat[(11+1)*pitch].ReinterpretCast<uint64_t>().GetValue(11/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(11);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((11&1)==0)Gather(paired,mat[(11+1)*pitch],bases.ReinterpretCast<uint32_t>(),11*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(11&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((11&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(11+1)*pitch],factor,pairSpread[(11&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((11&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 12


    factor=mat[12*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-12-1;
    if((12&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(12/2);
      uint64_t second=mat[(12+1)*pitch].ReinterpretCast<uint64_t>().GetValue(12/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(12);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((12&1)==0)Gather(paired,mat[(12+1)*pitch],bases.ReinterpretCast<uint32_t>(),12*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(12&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((12&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(12+1)*pitch],factor,pairSpread[(12&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((12&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 13


    factor=mat[13*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-13-1;
    if((13&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(13/2);
      uint64_t second=mat[(13+1)*pitch].ReinterpretCast<uint64_t>().GetValue(13/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(13);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((13&1)==0)Gather(paired,mat[(13+1)*pitch],bases.ReinterpretCast<uint32_t>(),13*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(13&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((13&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(13+1)*pitch],factor,pairSpread[(13&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((13&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 14


    factor=mat[14*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-14-1;
    if((14&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(14/2);
      uint64_t second=mat[(14+1)*pitch].ReinterpretCast<uint64_t>().GetValue(14/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(14);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((14&1)==0)Gather(paired,mat[(14+1)*pitch],bases.ReinterpretCast<uint32_t>(),14*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(14&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((14&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(14+1)*pitch],factor,pairSpread[(14&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((14&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 15


    factor=mat[15*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-15-1;
    if((15&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(15/2);
      uint64_t second=mat[(15+1)*pitch].ReinterpretCast<uint64_t>().GetValue(15/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(15);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
     if((15&1)==0)Gather(paired,mat[(15+1)*pitch],bases.ReinterpretCast<uint32_t>(),15*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(15&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((15&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(15+1)*pitch],factor,pairSpread[(15&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((15&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
   } while(false);
   Fence<HardEvent::V_S>();result.SetValue(0,singular?0.0f:product);Fence<HardEvent::S_MTE3>();
   if(GetBlockIdx()<owners)DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0});Fence<HardEvent::MTE3_S>();
  }
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t n=16,pitch=16,cols=16;uint32_t cores;uint64_t batches;
};

class KernelLinalgDet {
public:
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,GM_ADDR W,const LinalgDetTilingData& t){a=A;y=Y;w=W;data=t;}
 __aicore__ inline void Process(){
  if(data.n==16){KernelTinyExplicit16 op;op.Init(a,y,data);op.Process();}
 else if(data.n<=16){if(data.resident){KernelTinyFullwidthDet op;op.Init(a,y,data);op.Process();}else{KernelTinyDet op;op.Init(a,y,data);op.Process();}}
  else if(data.n>=33&&data.n<=64){KernelMediumDet op;op.Init(a,y,data);op.Process();}
  else if(data.n<=128){KernelColumnDet op;op.Init(a,y,data);op.Process();}
  else {KernelCooperativeDet op;op.Init(a,y,w,data);op.Process();}
 }
private: GM_ADDR a;GM_ADDR y;GM_ADDR w;LinalgDetTilingData data;
};

#define DET_FAST_VECTOR_FINISH 1
// Experimental complete FP32 blocked path. A singular diagonal block produces NaN;
// there is no precision admission or fallback in this speed/error experiment.
// TPipe's three L0 TBuf allocations seed M_MTE1 events 0,1,2.
// Manual reuse acknowledgements must not consume those initial tokens.
template<HardEvent E> __aicore__ inline void FastFence(){
 constexpr event_t id=E==HardEvent::M_MTE1?EVENT_ID4:EVENT_ID0;
 SetFlag<E>(id);WaitFlag<E>(id);
}
__aicore__ inline uint32_t FastParity(uint64_t x){x^=x>>32;x^=x>>16;x^=x>>8;x^=x>>4;return(0x6996u>>(uint32_t(x)&15))&1;}
__aicore__ inline void FastLoadLocal(LocalTensor<float>l1,LocalTensor<float>l0,uint32_t rows,uint32_t cols){
 LoadData3DParamsV2<float>p;p.l1H=1;p.l1W=rows;p.channelSize=cols;p.kExtension=cols;p.mExtension=rows;
 p.strideW=1;p.strideH=1;p.filterW=1;p.filterH=1;p.dilationFilterW=1;p.dilationFilterH=1;p.enTranspose=false;LoadData(l0,l1,p);
}
// Block LU in A = L * U form, with block-unit diagonal U. Reconstruct only
// the current cross from original A and all previously published block factors.
struct DetLeftJob { uint32_t batch,row,col,m,n; };
__aicore__ inline uint32_t DetLeftColumnWidth(uint32_t size,uint32_t start,uint32_t slots,uint32_t cores){
 uint32_t remain=size-start;if(remain<=64)return 64;
 uint32_t rows=(remain+127)/128,oldJobs=(rows+(remain-32+63)/64)*slots,newJobs=(rows+(remain-32+127)/128)*slots;
 return (newJobs+cores-1)/cores<(oldJobs+cores-1)/cores?128:64;
}
__aicore__ inline uint32_t DetLeftJobs(uint32_t size,uint32_t start,uint32_t slots,uint32_t cores){
 if(size-start<=64)return 1;
 uint32_t width=DetLeftColumnWidth(size,start,slots,cores);
 return (size-start+127)/128+(size-start-32+width-1)/width;
}
__aicore__ inline DetLeftJob DetLeftTile(uint32_t size,uint32_t start,uint32_t job,uint32_t slots,uint32_t cores){
 uint32_t count=DetLeftJobs(size,start,slots,cores),index=job%count;
 DetLeftJob t;t.batch=job/count;
 if(size-start<=64){t.row=start;t.col=start;t.m=(size-start+31)/32*32;t.n=t.m;return t;}
 uint32_t rowJobs=(size-start+127)/128,width=DetLeftColumnWidth(size,start,slots,cores);
 if(index<rowJobs){
  t.row=start;t.col=start+index*128;t.m=32;
  uint32_t live=size-t.col;if(live>128)live=128;t.n=(live+31)/32*32;
 }else{
  t.row=start+32+(index-rowJobs)*width;t.col=start;t.n=32;
  uint32_t live=size-t.row;if(live>width)live=width;t.m=(live+31)/32*32;
 }
 return t;
}


__aicore__ inline uint32_t DetOwnerRhsChunk(uint32_t n,uint32_t start){
 uint32_t chunk=(n-start-32+63)/64*8;
 return n<=320&&chunk<32?chunk:32;
}

 template<uint32_t ColumnBegin,uint32_t ActiveWidth>
 __aicore__ inline void DetTailPairEliminate(LocalTensor<float> mat,LocalTensor<float> paired,LocalTensor<float> pairSpread,LocalTensor<int32_t> bases,uint32_t n,uint32_t first,uint32_t last,float& product){
  constexpr uint32_t pitch=72;
  float cachedDiagonal=0.0f,cachedInverse=0.0f;
   for(uint32_t pairBase=first;pairBase<last;pairBase+=2){
    {const uint32_t p=pairBase;
    auto factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(0==0){
     // The event covers prior matrix producers; Gather and scalar reads only read them.
     event_t priorMatrixReady=static_cast<event_t>(0);
     SetFlag<HardEvent::V_S>(priorMatrixReady);
     if(remaining){
      // Matrix plus the initialized input copy are owned by this core.
      // Extra gathered words never enter a live coefficient or matrix update.
      // Gather only the two live pivot coefficients for each trailing column.
      auto* dst=reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr());
      auto* src=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(p+1)*pitch;
      auto* offsets=reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr());
      uint64_t firstMask[2]={remaining>=32?~uint64_t(0):(uint64_t(1)<<(2*remaining))-1,0};
      AscendC::GatherImpl<float>(dst,src,offsets,4608,p*4,firstMask,uint8_t(1),uint16_t(8));
      if(remaining>32){
       uint64_t secondMask[2]={(uint64_t(1)<<(2*remaining-64))-1,0};
       AscendC::GatherImpl<float>(dst+64,src,offsets+64,4608,p*4,secondMask,uint8_t(1),uint16_t(8));
      }
     }
     WaitFlag<HardEvent::V_S>(priorMatrixReady);
     diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[p];
     if(remaining){
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[p+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }
    }else diagonal=cachedDiagonal;
    // No per-pivot zero guard in this bound performance path.
    if(remaining){
     if(0==0){PipeBarrier<PIPE_V>();Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});}
     float divisor=diagonal;
     float inverse=0?cachedInverse:-1.0f/divisor;
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       inverse,uint64_t(ActiveWidth),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     MulAddDst<float,float,false>(mat[(p+1)*pitch+ColumnBegin],factor[ColumnBegin],pairSpread[0?24:0],uint64_t(ActiveWidth),remaining,
       BinaryRepeatParams{1,1,0,9,0,2});
     if(0==0){
      float normalizedBelow=below*inverse;
      cachedDiagonal=normalizedBelow*cross+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // VMLA reads only even coefficient blocks. This writes disjoint odd blocks.
      // The existing loop-end V barrier covers both before the next dependent update.
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,normalizedBelow,uint8_t((remaining+7)/8),
        uint16_t(2),uint16_t(2),uint16_t(16),uint16_t(16));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
    }
    {const uint32_t p=pairBase+1;
    auto factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(1==0){
     // The event covers prior matrix producers; Gather and scalar reads only read them.
     event_t priorMatrixReady=static_cast<event_t>(0);
     SetFlag<HardEvent::V_S>(priorMatrixReady);
     if(remaining){
      // Matrix plus the initialized input copy are owned by this core.
      // Extra gathered words never enter a live coefficient or matrix update.
      // Gather only the two live pivot coefficients for each trailing column.
      auto* dst=reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr());
      auto* src=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(p+1)*pitch;
      auto* offsets=reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr());
      uint64_t firstMask[2]={remaining>=32?~uint64_t(0):(uint64_t(1)<<(2*remaining))-1,0};
      AscendC::GatherImpl<float>(dst,src,offsets,4608,p*4,firstMask,uint8_t(1),uint16_t(8));
      if(remaining>32){
       uint64_t secondMask[2]={(uint64_t(1)<<(2*remaining-64))-1,0};
       AscendC::GatherImpl<float>(dst+64,src,offsets+64,4608,p*4,secondMask,uint8_t(1),uint16_t(8));
      }
     }
     WaitFlag<HardEvent::V_S>(priorMatrixReady);
     diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[p];
     if(remaining){
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[p+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }
    }else diagonal=cachedDiagonal;
    // No per-pivot zero guard in this bound performance path.
    if(remaining){
     if(1==0){PipeBarrier<PIPE_V>();Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});}
     float divisor=diagonal;
     float inverse=1?cachedInverse:-1.0f/divisor;
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       inverse,uint64_t(ActiveWidth),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     MulAddDst<float,float,false>(mat[(p+1)*pitch+ColumnBegin],factor[ColumnBegin],pairSpread[1?24:0],uint64_t(ActiveWidth),remaining,
       BinaryRepeatParams{1,1,0,9,0,2});
     if(1==0){
      float normalizedBelow=below*inverse;
      cachedDiagonal=normalizedBelow*cross+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // VMLA reads only even coefficient blocks. This writes disjoint odd blocks.
      // The existing loop-end V barrier covers both before the next dependent update.
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,normalizedBelow,uint8_t((remaining+7)/8),
        uint16_t(2),uint16_t(2),uint16_t(16),uint16_t(16));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
    }
   }
 }

class FastBlockVector {
 TPipe pipe;GlobalTensor<float>a,y,w;LocalTensor<float>mat,loaded,tmpProduct,pivot,spread,q,absolute,reduced,result;
 LocalTensor<int32_t>indices,indexTemp,bases,outputIndices,loadIndices;
#ifdef DET_FAST_VECTOR_FINISH
 LocalTensor<float>inverseDiag,inverseOff,pairDiagonal,pairOff;LocalTensor<int32_t>partnerDelta,pairIndices;LocalTensor<int32_t>permutationDelta;
#endif
 uint32_t n,pitch,slots,batches,lane,slot,within,cores,leftOffset,vOffset,tempOffset,currentBatch;float product;int32_t exponent;bool bad;
 __aicore__ inline float PairAbs(float x){return x<0?-x:x;}
 __aicore__ inline float PairMax(float x,float y){return x>y?x:y;}
 union Bits{float f;uint32_t u;};
 __aicore__ inline bool Finite(float x){Bits b;b.f=x;return(b.u&0x7f800000u)!=0x7f800000u;}
 __aicore__ inline void Multiply(float x){product*=x;}
 __aicore__ inline float Finish(){Bits b;if(bad){b.u=0x7fc00000u;return b.f;}return product;}
public:
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,GM_ADDR W,uint32_t size,uint32_t count,uint32_t owners){
  n=size;pitch=(n+127)/128*128;slots=count<owners?count:owners;batches=count;lane=GetBlockIdx();slot=lane/8;within=lane%8;cores=owners*4;
  leftOffset=slots*pitch*pitch;vOffset=leftOffset+slots*pitch*32;tempOffset=vOffset+slots*32*pitch;
  a.SetGlobalBuffer((__gm__ float*)A);y.SetGlobalBuffer((__gm__ float*)Y);w.SetGlobalBuffer((__gm__ float*)W);SetMaskNorm();SetAtomicNone();
  mat=LocalTensor<float>(TPosition::VECCALC,65536,2560);loaded=LocalTensor<float>(TPosition::VECCALC,8192,4096);
  tmpProduct=LocalTensor<float>(TPosition::VECCALC,24576,4096);indices=LocalTensor<int32_t>(TPosition::VECCALC,40960,1024);
  indexTemp=LocalTensor<int32_t>(TPosition::VECCALC,45056,1024);pivot=LocalTensor<float>(TPosition::VECCALC,49152,64);
  spread=LocalTensor<float>(TPosition::VECCALC,49408,512);bases=LocalTensor<int32_t>(TPosition::VECCALC,51456,64);
  absolute=LocalTensor<float>(TPosition::VECCALC,51840,32);reduced=LocalTensor<float>(TPosition::VECCALC,51968,8);result=LocalTensor<float>(TPosition::VECCALC,52000,8);
  inverseDiag=LocalTensor<float>(TPosition::VECCALC,52032,32);permutationDelta=LocalTensor<int32_t>(TPosition::VECCALC,52160,32);
  outputIndices=LocalTensor<int32_t>(TPosition::VECCALC,56320,1024);pairIndices=LocalTensor<int32_t>(TPosition::VECCALC,81920,1024);
  loadIndices=LocalTensor<int32_t>(TPosition::VECCALC,86016,1024);
  CreateVecIndex(indices,int32_t(0),1024);PipeBarrier<PIPE_V>();
  ShiftRight(indexTemp,indices,int32_t(5),1024);PipeBarrier<PIPE_V>();
  ShiftLeft(loaded.ReinterpretCast<int32_t>(),indexTemp,int32_t(5),1024);PipeBarrier<PIPE_V>();
  Sub(indices,indices,loaded.ReinterpretCast<int32_t>(),1024);PipeBarrier<PIPE_V>();
  Muls(outputIndices,indexTemp,int32_t(40),1024);PipeBarrier<PIPE_V>();
  Add(outputIndices,outputIndices,indices,1024);PipeBarrier<PIPE_V>();
  ShiftLeft(outputIndices,outputIndices,int32_t(2),1024);
  ShiftLeft(indexTemp,indices,int32_t(2),1024);
  ShiftRight(loadIndices,indices,int32_t(1),1024);PipeBarrier<PIPE_V>();
  ShiftLeft(loadIndices,loadIndices,int32_t(2),1024);PipeBarrier<PIPE_V>();
  CreateVecIndex(bases,int32_t(0),64);PipeBarrier<PIPE_V>();Muls(bases,bases,int32_t(160),64);PipeBarrier<PIPE_V>();
 }

 __aicore__ inline void Initialize(uint32_t wave){
  product=1;exponent=0;bad=false;currentBatch=wave*slots+slot;
 }
 __aicore__ inline void PrefetchOwnedRaw(uint32_t start){
  uint32_t rhs=n-start-32,chunk=DetOwnerRhsChunk(n,start),first=(within/2)*(2*chunk);
  if(slot>=slots||first>=rhs)return;
  uint32_t live=rhs-first<2*chunk?rhs-first:2*chunk,width=(live+15)/16*16;
  uint32_t row=start+(lane%2)*16,col=start+32+first;
  auto rawBase=LocalTensor<float>(TPosition::VECCALC,98304,1024);
  // Previous phase SyncAll covers the previous rawBase reader. This phase has
  // no vector writer; only live words are consumed or published.
  if(currentBatch<batches){
   DataCopyPad(rawBase,a[uint64_t(currentBatch)*n*n+row*n+col],
    DataCopyExtParams{16,live*4,(n-live)*4,(width-(live+7)/8*8)/8,0},
    DataCopyPadExtParams<float>{false,0,0,0});
  }else{Duplicate(rawBase,0.0f,16*width);PipeBarrier<PIPE_V>();}
 }
 __aicore__ inline void SolveChunk(uint32_t start,uint32_t stop,uint32_t first,uint32_t live,bool privateInput=false){
  uint32_t width=stop-start,rhs=n-stop,sbase=slot*pitch*pitch;bool singular=false;
  auto source=(start==0||privateInput)?a[uint64_t(currentBatch)*n*n]:w[sbase];uint32_t inputPitch=(start==0||privateInput)?n:pitch;
  // DMA overwrites all 32 live rows in every live panel/RHS column.
  // Keep the empty-wave behavior; padding and unused RHS columns have no live consumer.
  if(currentBatch>=batches){Duplicate(mat,0.0f,2560);if(privateInput)Duplicate(loaded,0.0f,2560);}
  FastFence<HardEvent::V_MTE2>();
  if(currentBatch<batches){
   DataCopyPad(mat,source[start*inputPitch+start],
     DataCopyExtParams{32,128,(inputPitch-32)*4,1,0},DataCopyPadExtParams<float>{false,0,0,0});
   if(live)DataCopyPad(mat[width*40],source[(stop+first)*inputPitch+start],
     DataCopyExtParams{uint16_t(live),128,(inputPitch-32)*4,1,0},DataCopyPadExtParams<float>{false,0,0,0});
  }
  if(privateInput){
   CrossCoreWaitFlag(10);
   uint32_t address=tempOffset+(lane/2)*8192;
   DataCopyPad(loaded,w[address],DataCopyExtParams{32,128,0,1,0},DataCopyPadExtParams<float>{false,0,0,0});
   DataCopyPad(loaded[width*40],w[address+1024+(lane%2)*DetOwnerRhsChunk(n,start)*32],
     DataCopyExtParams{uint16_t(live),128,0,1,0},DataCopyPadExtParams<float>{false,0,0,0});
  }
  FastFence<HardEvent::MTE2_V>();
  if(privateInput){Sub(mat,mat,loaded,2560);PipeBarrier<PIPE_V>();PrefetchOwnedRaw(start);}
  uint64_t active[2]={(uint64_t(1)<<width)-1,0};
  for(uint32_t col=0;col<width;col+=2){
   FastFence<HardEvent::V_S>();
   uint64_t bit0=active[0]&(~active[0]+1),rest=active[0]^bit0,bit1=rest&(~rest+1);
   uint32_t row0=63-static_cast<uint32_t>(CountLeadingZero(bit0));
   uint32_t row1=63-static_cast<uint32_t>(CountLeadingZero(bit1));
   auto col0=mat[col*40],col1=mat[(col+1)*40];
   float a00=reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row0],a10=reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row1],a01=reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row0],a11=reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row1];
   float determinant=a00*a11-a01*a10;
   float maxEntry=PairMax(PairMax(PairAbs(a00),PairAbs(a10)),PairMax(PairAbs(a01),PairAbs(a11)));
   if(!(PairAbs(determinant)>=0.25f*maxEntry)||!Finite(determinant)){
    Abs(absolute,col0,32);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,absolute,active,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);FastFence<HardEvent::V_S>();
    row0=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];a00=reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row0];a01=reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row0];
    Muls(loaded,col1,a00,32);Muls(loaded[32],col0,a01,32);PipeBarrier<PIPE_V>();
    Sub(absolute,loaded,loaded[32],32);PipeBarrier<PIPE_V>();Abs(absolute,absolute,32);PipeBarrier<PIPE_V>();
    uint64_t secondMask[2]={active[0]&~(uint64_t(1)<<row0),0};
    WholeReduceMax(reduced,absolute,secondMask,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);FastFence<HardEvent::V_S>();
    row1=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];a10=reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row1];a11=reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row1];
    determinant=a00*a11-a01*a10;
    maxEntry=PairMax(PairMax(PairAbs(a00),PairAbs(a10)),PairMax(PairAbs(a01),PairAbs(a11)));
   }
   if(determinant==0||!Finite(determinant)){singular=true;break;}
   Bits scaleWord,inverseScaleWord;scaleWord.f=maxEntry;scaleWord.u&=0x7f800000u;
   inverseScaleWord.u=0x7f000000u-scaleWord.u;
   float rowScale=scaleWord.f,inverseScale=inverseScaleWord.f;
   float inverse=1.0f/determinant;
   float weight00=a11*inverse,weight11=a00*inverse,weight01=-a01*inverse,weight10=-a10*inverse;
   uint32_t remaining=width+live-col-2;
   if(remaining){
    auto q0=loaded,q1=loaded[32],pivot1=loaded[64],spread1=loaded[128];
    // remaining <= 62: one normal-mask gather; disjoint coefficient outputs.
    uint64_t gatherMask[2]={(uint64_t(1)<<remaining)-1,0};
    auto* trailingWords=reinterpret_cast<__ubuf__ float*>(mat[(col+2)*40].GetPhyAddr());
    auto* baseWords=reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr());
    AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(pivot.GetPhyAddr()),trailingWords,
      baseWords,2560,row0*4,gatherMask,uint8_t(1),uint16_t(8));
    AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(pivot1.GetPhyAddr()),trailingWords,
      baseWords,2560,row1*4,gatherMask,uint8_t(1),uint16_t(8));
    reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row0]=a00-rowScale;reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row1]=a11-rowScale;FastFence<HardEvent::S_V>();
    SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xffffffffu));
    vmuls(reinterpret_cast<__ubuf__ float*>(q0.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr()),-weight00,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));vmuls(reinterpret_cast<__ubuf__ float*>(q1.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr()),-weight11,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));PipeBarrier<PIPE_V>();
    AscendC::BrcbImpl<float>(reinterpret_cast<__ubuf__ float*>(spread.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(pivot.GetPhyAddr()),uint8_t((remaining+7)/8),BrcbRepeatParams{1,8});
    AscendC::BrcbImpl<float>(reinterpret_cast<__ubuf__ float*>(spread1.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(pivot1.GetPhyAddr()),uint8_t((remaining+7)/8),BrcbRepeatParams{1,8});
     SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xffffffffu));
    vaxpy(reinterpret_cast<__ubuf__ float*>(q0.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr()),-weight10,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));vaxpy(reinterpret_cast<__ubuf__ float*>(q1.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr()),-weight01,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));PipeBarrier<PIPE_V>();
    SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xffffffffu));
    MulAddDst<float,float,false>(mat[(col+2)*40],q0,spread,uint64_t(32),remaining,BinaryRepeatParams{1,1,0,5,0,1});PipeBarrier<PIPE_V>();
    MulAddDst<float,float,false>(mat[(col+2)*40],q1,spread1,uint64_t(32),remaining,BinaryRepeatParams{1,1,0,5,0,1});
   }
   reinterpret_cast<__ubuf__ float*>(inverseDiag.GetPhyAddr())[col/2]=inverseScale;
   reinterpret_cast<__ubuf__ int32_t*>(permutationDelta.GetPhyAddr())[col]=int32_t(row0-col)*4;reinterpret_cast<__ubuf__ int32_t*>(permutationDelta.GetPhyAddr())[col+1]=int32_t(row1-col-1)*4;
   if(first==0&&FastParity(active[0]&((uint64_t(1)<<row0)-1)))product=-product;
   active[0]&=~(uint64_t(1)<<row0);
   if(first==0&&FastParity(active[0]&((uint64_t(1)<<row1)-1)))product=-product;
   active[0]&=~(uint64_t(1)<<row1);
   if(first==0)Multiply(determinant);
   PipeBarrier<PIPE_V>();
  }
  if(first==0&&singular)bad=true;
  if(rhs){
   const uint32_t packed=live*32;
   if(singular){Duplicate(tmpProduct,0.0f,packed);PipeBarrier<PIPE_V>();}
   else{
    FastFence<HardEvent::S_V>();
    // Both row permutation and row scale repeat for each RHS column.
    // pivot is dead after panel elimination; keep the repeated index source
    // separate from pairIndices, whose first repeat is overwritten by Add.
    auto rowDelta=pivot.ReinterpretCast<int32_t>();
    Gather(rowDelta,permutationDelta,indexTemp.ReinterpretCast<uint32_t>(),0,32);PipeBarrier<PIPE_V>();
    Add(pairIndices,rowDelta,outputIndices,uint64_t(32),uint8_t(live),
      BinaryRepeatParams{1,1,1,4,0,4});PipeBarrier<PIPE_V>();
    Gather(tmpProduct,mat[width*40],pairIndices.ReinterpretCast<uint32_t>(),0,packed);PipeBarrier<PIPE_V>();
    Gather(loaded,inverseDiag,loadIndices.ReinterpretCast<uint32_t>(),0,32);PipeBarrier<PIPE_V>();
    Mul(tmpProduct,tmpProduct,loaded,uint64_t(32),uint8_t(live),
      BinaryRepeatParams{1,1,1,4,4,0});PipeBarrier<PIPE_V>();
   }
   FastFence<HardEvent::V_MTE3>();DataCopyPad(w[sbase+(stop+first)*pitch+start],tmpProduct,DataCopyExtParams{uint16_t(live),128,0,(pitch-32)*4,0});FastFence<HardEvent::MTE3_V>();
  }
 }

 __aicore__ inline void Prepare(uint32_t start,uint32_t stop){
  if(slot>=slots)return;
  uint32_t rhs=n-stop,chunk=DetOwnerRhsChunk(n,start);
  for(uint32_t first=within*chunk;first<rhs;first+=8*chunk)
   SolveChunk(start,stop,first,rhs-first<chunk?rhs-first:chunk);
  if(start)return;
  for(uint32_t first=within*chunk;first<rhs;first+=8*chunk){
   uint32_t live=rhs-first<chunk?rhs-first:chunk;
   Duplicate(loaded,0.0f,1024);FastFence<HardEvent::V_MTE2>();
   if(currentBatch<batches){
    DataCopyPad(loaded,a[uint64_t(currentBatch)*n*n+stop+first],
      DataCopyExtParams{32,live*4,(n-live)*4,(32-(live+7)/8*8)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
    FastFence<HardEvent::MTE2_V>();
   }
   FastFence<HardEvent::V_MTE3>();
   DataCopyPad(w[slot*pitch*pitch+stop+first],loaded,
     DataCopyExtParams{32,live*4,(32-(live+7)/8*8)/8,(pitch-live)*4,0});
   FastFence<HardEvent::MTE3_V>();
  }
 }

 __aicore__ inline void PrepareOwned(uint32_t start){
  uint32_t rhs=n-start-32,chunk=DetOwnerRhsChunk(n,start),first=(within/2)*(2*chunk);
  if(slot>=slots)return;
  CrossCoreSetFlag<2,PIPE_MTE3>(11);
  if(first>=rhs)return;
  uint32_t mine=within*chunk;
  if(mine<rhs)SolveChunk(start,start+32,mine,rhs-mine<chunk?rhs-mine:chunk,true);
  else {CrossCoreWaitFlag(10);PrefetchOwnedRaw(start);}
 }
 __aicore__ inline void ConsumeOwnedRaw(uint32_t start){
  uint32_t rhs=n-start-32,chunk=DetOwnerRhsChunk(n,start),first=(within/2)*(2*chunk);
  if(slot>=slots||first>=rhs)return;
  uint32_t live=rhs-first<2*chunk?rhs-first:2*chunk,width=(live+15)/16*16;
  uint32_t core=lane/2,sub=lane%2,row=start+sub*16,col=start+32+first;
  auto rawBase=LocalTensor<float>(TPosition::VECCALC,98304,1024);
  // Independent of SolveChunk packing and its MTE3 reader.
  auto rawProduct=LocalTensor<float>(TPosition::VECCALC,102400,1024);
  CrossCoreWaitFlag(12);
  DataCopy(rawProduct,w[tempOffset+core*8192+4096+sub*16*width],16*width);
  FastFence<HardEvent::MTE2_V>();
  Sub(loaded,rawBase,rawProduct,16*width);PipeBarrier<PIPE_V>();FastFence<HardEvent::V_MTE3>();
  DataCopyPad(w[slot*pitch*pitch+row*pitch+col],loaded,
    DataCopyExtParams{16,live*4,(width-(live+7)/8*8)/8,(pitch-live)*4,0});
  FastFence<HardEvent::MTE3_V>();
 }

 __aicore__ inline void Rebuild(uint32_t start){
  uint32_t core=lane/2,sub=lane%2,jobs=slots*DetLeftJobs(n,start,slots,cores);
  if(core<jobs)CrossCoreSetFlag<2,PIPE_MTE3>(11);
  for(uint32_t job=core;job<jobs;job+=cores){
   DetLeftJob tile=DetLeftTile(n,start,job,slots,cores);
   uint32_t row=tile.col+sub*64,sourceBatch=currentBatch-slot+tile.batch;
   uint32_t rows=(sub*64<tile.n&&row<n)?(n-row<64?n-row:64):0;
   if(rows>tile.n-sub*64)rows=tile.n-sub*64;
   uint32_t cols=n-tile.row<tile.m?n-tile.row:tile.m;
   uint32_t chunkRows=tile.n>sub*64?tile.n-sub*64:0;if(chunkRows>64)chunkRows=64;
   if(rows){
    Duplicate(loaded,0.0f,chunkRows*tile.m);FastFence<HardEvent::V_MTE2>();
    if(sourceBatch<batches){
     DataCopyPad(loaded,a[uint64_t(sourceBatch)*n*n+row*n+tile.row],
       DataCopyExtParams{uint16_t(rows),cols*4,(n-cols)*4,(tile.m-(cols+7)/8*8)/8,0},
       DataCopyPadExtParams<float>{false,0,0,0});
     FastFence<HardEvent::MTE2_V>();
    }
   }
   FastFence<HardEvent::V_MTE3>();
   if(job!=core)CrossCoreSetFlag<2,PIPE_MTE3>(11);
   CrossCoreWaitFlag(10);
   if(rows){
    FastFence<HardEvent::V_MTE2>();
    DataCopy(tmpProduct,w[tempOffset+core*8192+sub*64*tile.m],chunkRows*tile.m);
    FastFence<HardEvent::MTE2_V>();
    Sub(loaded,loaded,tmpProduct,chunkRows*tile.m);PipeBarrier<PIPE_V>();
    FastFence<HardEvent::V_MTE3>();
    DataCopyPad(w[tile.batch*pitch*pitch+row*pitch+tile.row],loaded,
      DataCopyExtParams{uint16_t(rows),cols*4,(tile.m-(cols+7)/8*8)/8,(pitch-cols)*4,0});
    FastFence<HardEvent::MTE3_V>();
   }
  }
 }
 __aicore__ inline void FinishTail(uint32_t start){
  if(within||slot>=slots)return;
  uint32_t width=n-start;
  auto matrix=LocalTensor<float>(TPosition::VECCALC,0,4608);
  auto absoluteTail=LocalTensor<float>(TPosition::VECCALC,18432,64);
  auto pivotTail=LocalTensor<float>(TPosition::VECCALC,18688,64);
  auto spreadTail=LocalTensor<float>(TPosition::VECCALC,18944,512);
  auto basesTail=LocalTensor<int32_t>(TPosition::VECCALC,20992,64);
  FastFence<HardEvent::S_V>();Duplicate(matrix,0.0f,4608);FastFence<HardEvent::V_MTE2>();
  DataCopyPad(matrix,w[slot*pitch*pitch+start*pitch+start],
   DataCopyExtParams{uint16_t(width),width*4,(pitch-width)*4,(72-(width+7)/8*8)/8,0},
   DataCopyPadExtParams<float>{true,0,uint8_t((width+7)/8*8-width),0.0f});
  FastFence<HardEvent::MTE2_V>();
  if((width&1)==0){
   auto paired=LocalTensor<float>(TPosition::VECCALC,24576,128);
   auto pairSpread=LocalTensor<float>(TPosition::VECCALC,25088,1024);
   auto pairBases=LocalTensor<int32_t>(TPosition::VECCALC,29184,128);
   for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(pairBases.GetPhyAddr())[i]=uint64_t(i*72*4)|(uint64_t(i*72*4+4)<<32);
   FastFence<HardEvent::S_V>();
   for(uint32_t filled=16;filled<128;filled*=2){Adds(pairBases[filled],pairBases,int32_t(filled/2*72*4),filled);PipeBarrier<PIPE_V>();}
   DetTailPairEliminate<0,64>(matrix,paired,pairSpread,pairBases,width,0,width<32?width:32,product);
   if(width>32)DetTailPairEliminate<32,32>(matrix,paired,pairSpread,pairBases,width,32,width,product);
   return;
  }
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ int32_t*>(basesTail.GetPhyAddr())[i]=int32_t(i*72*4);
  FastFence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<64;filled*=2){
   Adds(basesTail[filled],basesTail,int32_t(filled*72*4),filled);PipeBarrier<PIPE_V>();
  }
  uint64_t activeTail[2]={width==64?~uint64_t(0):(uint64_t(1)<<width)-1,0};
  float cachedDiagonal=0.0f,cachedInverse=0.0f;uint32_t cachedRow=0;
  for(uint32_t col=0;col<width;++col){
   auto factor=matrix[col*72];
   uint32_t best;float diagonal;bool predicted=(col&1)!=0;
   if(predicted){best=cachedRow;diagonal=cachedDiagonal;}
   else{
    FastFence<HardEvent::V_S>();
    uint64_t first=activeTail[0]&(~activeTail[0]+1);
    best=63-static_cast<uint32_t>(CountLeadingZero(first));diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[best];
   }
   if(!(diagonal>=1.0f||diagonal<=-1.0f)||!Finite(diagonal)){
    predicted=false;
    Abs(absoluteTail,factor,64);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,absoluteTail,activeTail,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
    FastFence<HardEvent::V_S>();
    float largest=reinterpret_cast<__ubuf__ float*>(reduced.GetPhyAddr())[0];
    if(largest==0){product=0;return;}if(!Finite(largest)){bad=true;return;}
    best=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[best];
   }
   uint32_t remaining=width-col-1;
   if(remaining){
    float below=0.0f,cross=0.0f,future=0.0f;
    if((col&1)==0){
     uint64_t nextActive=activeTail[0]&~(uint64_t(1)<<best);
     uint64_t first=nextActive&(~nextActive+1);
     cachedRow=63-static_cast<uint32_t>(CountLeadingZero(first));
     below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[cachedRow];
     cross=reinterpret_cast<__ubuf__ float*>(matrix[(col+1)*72].GetPhyAddr())[best];
     future=reinterpret_cast<__ubuf__ float*>(matrix[(col+1)*72].GetPhyAddr())[cachedRow];
    }
    // remaining <= 63; the gathered trailing columns and factor are disjoint.
    uint64_t liveMask[2]={(uint64_t(1)<<remaining)-1,0};
    AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(pivotTail.GetPhyAddr()),
      reinterpret_cast<__ubuf__ float*>(matrix[(col+1)*72].GetPhyAddr()),
      reinterpret_cast<__ubuf__ uint32_t*>(basesTail.GetPhyAddr()),4608,best*4,
      liveMask,uint8_t(1),uint16_t(8));
    Bits bits;bits.f=diagonal;float divisor=diagonal;
    if((bits.u&0x7f800000u)==0){
     Muls(factor,factor,16777216.0f,64);PipeBarrier<PIPE_V>();
     divisor*=16777216.0f;below*=16777216.0f;
    }
    float inverse=predicted?cachedInverse:-1.0f/divisor;
    SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
    auto* factorWords=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr());
    vmuls(factorWords,factorWords,inverse,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));PipeBarrier<PIPE_V>();
    Brcb(spreadTail,pivotTail,(remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
    MulAddDst<float,float,false>(matrix[(col+1)*72],factor,spreadTail,uint64_t(64),remaining,
      BinaryRepeatParams{1,1,0,9,0,1});
    if((col&1)==0){
     float normalizedBelow=below*inverse;
     cachedDiagonal=normalizedBelow*cross+future;
     Bits nextBits;nextBits.f=cachedDiagonal;float nextDivisor=cachedDiagonal;
     if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
     cachedInverse=-1.0f/nextDivisor;
    }
   }
   if(FastParity(activeTail[0]&((uint64_t(1)<<best)-1)))product=-product;
   Multiply(diagonal);activeTail[0]&=~(uint64_t(1)<<best);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Store(uint32_t wave){
  uint32_t batch=wave*slots+slot;if(within||batch>=batches)return;FastFence<HardEvent::V_S>();reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())[0]=Finish();FastFence<HardEvent::S_MTE3>();
  DataCopyPad(y[batch],result,DataCopyExtParams{1,4,0,0,0});FastFence<HardEvent::MTE3_V>();
 }
};
class DetLeftCube {
 TPipe pipe;TBuf<TPosition::A1>ab;TBuf<TPosition::B1>bb;
 TBuf<TPosition::A2>ac;TBuf<TPosition::B2>bc;TBuf<TPosition::CO1>cc;
public:
 __aicore__ inline void Init(){
  pipe.InitBuffer(ab,65536);pipe.InitBuffer(bb,65536);
  pipe.InitBuffer(ac,65536);pipe.InitBuffer(bc,65536);pipe.InitBuffer(cc,65536);
  SetHF32Mode(HF32Mode::DISABLE);
 }
 __aicore__ inline void OwnedProduct(GlobalTensor<float> w,uint32_t base,uint32_t pitch,
     uint32_t start,uint32_t first,uint32_t width,uint32_t k,uint32_t count,bool raw,bool init){
  auto a1=ab.Get<float>(),b1=bb.Get<float>(),a2=ac.Get<float>(),b2=bc.Get<float>();
  uint32_t rows=raw?32:32+width,cols=raw?width:32;
  if(raw){
   DataCopy(a1,w[base+start*pitch+k],Nd2NzParams{1,32,uint16_t(count),0,uint16_t(pitch),32,1,0});
   DataCopy(b1,w[base+k*pitch+start+32+first],Nd2NzParams{1,uint16_t(count),uint16_t(width),0,uint16_t(pitch),uint16_t(count),1,0});
  }else{
   // FP32 NZ uses eight elements per C0. The second copy starts at row32
   // within each C0 plane; both copies share the full stacked row stride.
   DataCopy(a1,w[base+start*pitch+k],Nd2NzParams{1,32,uint16_t(count),0,uint16_t(pitch),uint16_t(rows),1,0});
   DataCopy(a1[32*8],w[base+(start+32+first)*pitch+k],Nd2NzParams{1,uint16_t(width),uint16_t(count),0,uint16_t(pitch),uint16_t(rows),1,0});
   DataCopy(b1,w[base+k*pitch+start],Nd2NzParams{1,uint16_t(count),32,0,uint16_t(pitch),uint16_t(count),1,0});
  }
  FastFence<HardEvent::MTE2_MTE1>();
  FastLoadLocal(a1,a2,rows,count);FastLoadLocal(b1,b2,count,cols);FastFence<HardEvent::MTE1_M>();
  Mmad(cc.Get<float>()[raw?4096:0],a2,b2,MmadParams{uint16_t(rows),uint16_t(cols),uint16_t(count),0,false,init});
  FastFence<HardEvent::M_MTE1>();FastFence<HardEvent::MTE1_MTE2>();
 }
 __aicore__ inline void RebuildOwned(GM_ADDR W,uint32_t size,uint32_t slots,uint32_t start){
  uint32_t core=GetBlockIdx(),batch=core/4,rhs=size-start-32,chunk=DetOwnerRhsChunk(size,start),first=(core%4)*(2*chunk);
  if(batch>=slots)return;
  if(first>=rhs){CrossCoreWaitFlag(11);return;}
  uint32_t live=rhs-first<2*chunk?rhs-first:2*chunk,width=(live+15)/16*16;
  uint32_t pitch=(size+127)/128*128,base=batch*pitch*pitch,temp=slots*(pitch*pitch+64*pitch);
  uint32_t history=start-32;GlobalTensor<float>w;w.SetGlobalBuffer((__gm__ float*)W);
  // Idle stages consumed11 too: old history is already published for every owner.
  for(uint32_t k=0;k<history;k+=128){
   uint32_t count=history-k;if(count>128)count=128;
   OwnedProduct(w,base,pitch,start,first,width,k,count,false,k==0);
   OwnedProduct(w,base,pitch,start,first,width,k,count,true,k==0);
  }
  CrossCoreWaitFlag(11);
  OwnedProduct(w,base,pitch,start,first,width,history,32,false,history==0);
  FastFence<HardEvent::M_FIX>();
  Fixpipe(w[temp+core*8192],cc.Get<float>(),FixpipeParamsV220{32,uint16_t(32+width),uint16_t(32+width),32,false,QuantMode_t::NoQuant,0,1,0,0,0});
  CrossCoreSetFlag<2,PIPE_FIX>(10);
  // Separate scratch retains both products until the following publication
  // barrier. Raw reconstruction overlaps the sibling AIV factorization.
  OwnedProduct(w,base,pitch,start,first,width,history,32,true,history==0);
  FastFence<HardEvent::M_FIX>();
  Fixpipe(w[temp+core*8192+4096],cc.Get<float>()[4096],FixpipeParamsV220{uint16_t(width),32,32,uint16_t(width),false,QuantMode_t::NoQuant,0,1,0,0,0});
  PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(12);
 }

 __aicore__ inline void Rebuild(GM_ADDR W,uint32_t size,uint32_t slots,uint32_t cores,uint32_t start){
  uint32_t pitch=(size+127)/128*128,core=GetBlockIdx();
  uint32_t jobs=slots*DetLeftJobs(size,start,slots,cores),temp=slots*(pitch*pitch+64*pitch);
  if(core>=jobs)return;
  GlobalTensor<float>w;w.SetGlobalBuffer((__gm__ float*)W);
  auto a1=ab.Get<float>(),b1=bb.Get<float>(),a2=ac.Get<float>(),b2=bc.Get<float>(),c=cc.Get<float>();
  uint32_t history=start-32;
  // Every real owner group consumed the preceding11, including idle groups.
  // That global publication proves old tail history ready for every matrix.
  bool ownerTail=size<=320&&size-start<=64;
  bool deferredHistory=history&&!ownerTail&&core>=slots*DetLeftJobs(size,start-32,slots,cores);
  if(deferredHistory)CrossCoreWaitFlag(11);
  // k<history was published before the preceding reconstruction. Its factors
  // are immutable while AIV consumes that cross and solves the newest block.
  if(history){
   for(uint32_t job=core;job<jobs;job+=cores){
    DetLeftJob tile=DetLeftTile(size,start,job,slots,cores);uint32_t base=tile.batch*pitch*pitch;
    auto partial=c[((job-core)/cores)*4096];
    for(uint32_t k=0;k<history;k+=128){
     uint32_t count=history-k;if(count>128)count=128;
     DataCopy(a1,w[base+tile.col*pitch+k],Nd2NzParams{1,uint16_t(tile.n),uint16_t(count),0,uint16_t(pitch),uint16_t(tile.n),1,0});
     DataCopy(b1,w[base+k*pitch+tile.row],Nd2NzParams{1,uint16_t(count),uint16_t(tile.m),0,uint16_t(pitch),uint16_t(count),1,0});
     FastFence<HardEvent::MTE2_MTE1>();
     FastLoadLocal(a1,a2,tile.n,count);FastLoadLocal(b1,b2,count,tile.m);
     FastFence<HardEvent::MTE1_M>();
     Mmad(partial,a2,b2,MmadParams{uint16_t(tile.n),uint16_t(tile.m),uint16_t(count),0,false,k==0});
     FastFence<HardEvent::M_MTE1>();FastFence<HardEvent::MTE1_MTE2>();
    }
   }
  }
  if(!deferredHistory)CrossCoreWaitFlag(11);
  for(uint32_t job=core;job<jobs;job+=cores){
   DetLeftJob tile=DetLeftTile(size,start,job,slots,cores);uint32_t base=tile.batch*pitch*pitch;
   auto partial=c[((job-core)/cores)*4096];
   DataCopy(a1,w[base+tile.col*pitch+history],Nd2NzParams{1,uint16_t(tile.n),32,0,uint16_t(pitch),uint16_t(tile.n),1,0});
   DataCopy(b1,w[base+history*pitch+tile.row],Nd2NzParams{1,32,uint16_t(tile.m),0,uint16_t(pitch),32,1,0});
   FastFence<HardEvent::MTE2_MTE1>();
   FastLoadLocal(a1,a2,tile.n,32);FastLoadLocal(b1,b2,32,tile.m);
   FastFence<HardEvent::MTE1_M>();
   Mmad(partial,a2,b2,MmadParams{uint16_t(tile.n),uint16_t(tile.m),32,0,false,history==0});
   FastFence<HardEvent::M_FIX>();
   if(job!=core)CrossCoreWaitFlag(11);
   Fixpipe(w[temp+core*8192],partial,FixpipeParamsV220{uint16_t(tile.m),uint16_t(tile.n),uint16_t(tile.n),uint16_t(tile.m),false,QuantMode_t::NoQuant,0,1,0,0,0});
   PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(10);
  }
 }

};

__aicore__ inline void FastBlockProcess(GM_ADDR a,GM_ADDR y,GM_ADDR w,uint32_t n,uint32_t batches,uint32_t slots){
 InitSocState();uint32_t activeSlots=batches<slots?batches:slots;
 if ASCEND_IS_AIV {
  FastBlockVector vector;vector.Init(a,y,w,n,batches,slots);
  for(uint32_t wave=0;wave<(batches+activeSlots-1)/activeSlots;++wave){
   vector.Initialize(wave);
   for(uint32_t start=0;start<n;start+=32){
    if(start&&n<=320&&n-start>64){
     vector.PrepareOwned(start);vector.ConsumeOwnedRaw(start);SyncAll();continue;
    }
    if(start){vector.Rebuild(start);SyncAll();}
    if(n-start<=64){vector.FinishTail(start);break;}
    vector.Prepare(start,start+32);SyncAll();
   }
   vector.Store(wave);
   // Later waves reuse shared factors/scratch; after the last output no consumer remains.
   // Store completes its own DMA and each TPipe destructor drains its local engines.
   if(wave+1<(batches+activeSlots-1)/activeSlots)SyncAll();
  }
 }
 if ASCEND_IS_AIC {
  DetLeftCube cube;cube.Init();
  for(uint32_t wave=0;wave<(batches+activeSlots-1)/activeSlots;++wave)
   for(uint32_t start=32;start<n;start+=32){
    if(n<=320&&n-start>64)cube.RebuildOwned(w,n,activeSlots,start);else cube.Rebuild(w,n,activeSlots,slots*4,start);if(n-start<=64)break;
   }
 }
}

class KernelTinyBoundPrefetchPadded16 {
public:
 __aicore__ inline ~KernelTinyBoundPrefetchPadded16(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  mat=LocalTensor<float>(TPosition::VECCALC,0,256);
  loaded=LocalTensor<float>(TPosition::VECCALC,1024,256);
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,2048,256);
  pivot=LocalTensor<float>(TPosition::VECCALC,3072,16);
  spread=LocalTensor<float>(TPosition::VECCALC,3136,128);
  bases=LocalTensor<int32_t>(TPosition::VECCALC,5760,64);
  paired=LocalTensor<float>(TPosition::VECCALC,6016,64);
  pairSpread=LocalTensor<float>(TPosition::VECCALC,6272,512);
  factorWide=LocalTensor<float>(TPosition::VECCALC,8320,64);
  absolute=LocalTensor<float>(TPosition::VECCALC,3712,128);
  reduced=LocalTensor<float>(TPosition::VECCALC,4224,8);
  result=LocalTensor<float>(TPosition::VECCALC,4256,8);
  // DMA overwrites every word contributing to a live matrix entry.
  DataCopyPad(loaded,input,DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64)<<32);
  Fence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr())[i]=int32_t(i*n*4);reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64)<<32);}
  Fence<HardEvent::S_V>();
  Adds(offsets[8],offsets,int32_t(8*n*4),8);Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  #pragma unroll
  for(uint32_t filled=16;filled<256;filled*=2){
   Adds(offsets[filled],offsets,int32_t(filled/16*4),filled);PipeBarrier<PIPE_V>();
  }
  }
  Adds(bases[32],bases,int32_t(4),32);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Load(uint64_t offset){
  // This class is reachable only for one matrix on owner0; Init issued its DMA.
  Fence<HardEvent::MTE2_V>();
  Gather(mat,loaded,indices,0,cols*pitch);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Process(){
  // Bound entry admits only owner0. Keep the pivot loop rolled.
  {
   constexpr uint64_t batch=0;
   Load(0);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;
   // Keep one compact pair body and precompute the odd inverse once.
   #pragma clang loop unroll(disable)
   for(uint32_t pairBase=0;pairBase+1<n;pairBase+=2){
    {const uint32_t p=pairBase;
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(0==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=factor.GetValue(p);
      below=factor.GetValue(p+1);
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }else diagonal=factor.GetValue(p);
    }else diagonal=cachedDiagonal;
    if(remaining){
     if(0==0)Gather(paired,mat,bases.ReinterpretCast<uint32_t>(),p*4,64);
     float divisor=diagonal;
     float inverse=0?cachedInverse:-1.0f/divisor;
     // Gather only reads matrix words;factorWide is separate from both matrix and paired coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factorWide.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(4),UnaryRepeatParams{1,1,2,0}); // Consumer synchronization must retain the required V dependency.
     if(0==0){
      Brcb(pairSpread,paired,8,BrcbRepeatParams{1,8}); // Consumer synchronization must retain the required V dependency.
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     // Retired rows may change; only live diagonal and trailing rows are consumed.
     MulAddDst(mat,factorWide,pairSpread[0?256:0],uint64_t(64),4,
       BinaryRepeatParams{1,1,1,8,0,8});
     if(0==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Preserve SDK count-mask preparation, bypass only the annotated tensor interface.
      // The first256 coefficient words are read-only;only the second256 are written.
      AscendC::AxpyImpl<float,float>(reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr())+256,
        reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr()),float(normalizedBelow),int32_t(256));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    {const uint32_t p=pairBase+1;
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(1==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=factor.GetValue(p);
      below=factor.GetValue(p+1);
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }else diagonal=factor.GetValue(p);
    }else diagonal=cachedDiagonal;
    if(remaining){
     if(1==0)Gather(paired,mat,bases.ReinterpretCast<uint32_t>(),p*4,64);
     float divisor=diagonal;
     float inverse=1?cachedInverse:-1.0f/divisor;
     // Gather only reads matrix words;factorWide is separate from both matrix and paired coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factorWide.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(4),UnaryRepeatParams{1,1,2,0}); // Consumer synchronization must retain the required V dependency.
     if(1==0){
      Brcb(pairSpread,paired,8,BrcbRepeatParams{1,8}); // Consumer synchronization must retain the required V dependency.
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     // Retired rows may change; only live diagonal and trailing rows are consumed.
     MulAddDst(mat,factorWide,pairSpread[1?256:0],uint64_t(64),4,
       BinaryRepeatParams{1,1,1,8,0,8});
     if(1==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Preserve SDK count-mask preparation, bypass only the annotated tensor interface.
      // The first256 coefficient words are read-only;only the second256 are written.
      AscendC::AxpyImpl<float,float>(reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr())+256,
        reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr()),float(normalizedBelow),int32_t(256));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
   }
   if(n&1){
    Fence<HardEvent::V_S>();
    // The preceding explicit V_S covers the final matrix value.
    product*=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(n-1)*pitch+n-1];
   }
   *reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())=product;Fence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0}); // Final PIPE_ALL covers output completion.
  }
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread,factorWide;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t pitch=16,cols=16;uint32_t n,cores;uint64_t batches;
};

class KernelTinyBoundTerminal15 {
public:
 __aicore__ inline ~KernelTinyBoundTerminal15(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  mat=LocalTensor<float>(TPosition::VECCALC,0,256);
  loaded=LocalTensor<float>(TPosition::VECCALC,1024,256);
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,2048,256);
  pivot=LocalTensor<float>(TPosition::VECCALC,3072,16);
  spread=LocalTensor<float>(TPosition::VECCALC,3136,128);
  bases=LocalTensor<int32_t>(TPosition::VECCALC,5760,64);
  paired=LocalTensor<float>(TPosition::VECCALC,6016,64);
  pairSpread=LocalTensor<float>(TPosition::VECCALC,6272,512);
  factorWide=LocalTensor<float>(TPosition::VECCALC,8320,64);
  absolute=LocalTensor<float>(TPosition::VECCALC,3712,128);
  reduced=LocalTensor<float>(TPosition::VECCALC,4224,8);
  result=LocalTensor<float>(TPosition::VECCALC,4256,8);
  // DMA overwrites every word contributing to a live matrix entry.
  DataCopyPad(loaded,input,DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64)<<32);
  Fence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr())[i]=int32_t(i*n*4);reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64)<<32);}
  Fence<HardEvent::S_V>();
  Adds(offsets[8],offsets,int32_t(8*n*4),8);Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  #pragma unroll
  for(uint32_t filled=16;filled<256;filled*=2){
   Adds(offsets[filled],offsets,int32_t(filled/16*4),filled);PipeBarrier<PIPE_V>();
  }
  }
  Adds(bases[32],bases,int32_t(4),32);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Load(uint64_t offset){
  // This class is reachable only for one matrix on owner0; Init issued its DMA.
  Fence<HardEvent::MTE2_V>();
  Gather(mat,loaded,indices,0,cols*pitch);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Process(){
  // Bound entry admits only owner0. Keep the pivot loop rolled.
  {
   constexpr uint64_t batch=0;
   Load(0);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;
   // Keep one compact pair body and precompute the odd inverse once.
   #pragma clang loop unroll(disable)
   for(uint32_t pairBase=0;pairBase+1<(n==15?12:n);pairBase+=2){
    {const uint32_t p=pairBase;
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(0==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=factor.GetValue(p);
      below=factor.GetValue(p+1);
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }else diagonal=factor.GetValue(p);
    }else diagonal=cachedDiagonal;
    if(remaining){
     if(0==0)Gather(paired,mat,bases.ReinterpretCast<uint32_t>(),p*4,64);
     float divisor=diagonal;
     float inverse=0?cachedInverse:-1.0f/divisor;
     // Gather only reads matrix words;factorWide is separate from both matrix and paired coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factorWide.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(4),UnaryRepeatParams{1,1,2,0}); // Consumer synchronization must retain the required V dependency.
     if(0==0){
      Brcb(pairSpread,paired,8,BrcbRepeatParams{1,8}); // Consumer synchronization must retain the required V dependency.
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     // Retired rows may change; only live diagonal and trailing rows are consumed.
     MulAddDst(mat,factorWide,pairSpread[0?256:0],uint64_t(64),4,
       BinaryRepeatParams{1,1,1,8,0,8});
     if(0==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Preserve SDK count-mask preparation, bypass only the annotated tensor interface.
      // The first256 coefficient words are read-only;only the second256 are written.
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
      vaxpy(reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr())+256,
        reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr()),float(normalizedBelow),uint8_t(4),
        uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    {const uint32_t p=pairBase+1;
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(1==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=factor.GetValue(p);
      below=factor.GetValue(p+1);
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }else diagonal=factor.GetValue(p);
    }else diagonal=cachedDiagonal;
    if(remaining){
     if(1==0)Gather(paired,mat,bases.ReinterpretCast<uint32_t>(),p*4,64);
     float divisor=diagonal;
     float inverse=1?cachedInverse:-1.0f/divisor;
     // Gather only reads matrix words;factorWide is separate from both matrix and paired coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factorWide.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(4),UnaryRepeatParams{1,1,2,0}); // Consumer synchronization must retain the required V dependency.
     if(1==0){
      Brcb(pairSpread,paired,8,BrcbRepeatParams{1,8}); // Consumer synchronization must retain the required V dependency.
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     // Retired rows may change; only live diagonal and trailing rows are consumed.
     MulAddDst(mat,factorWide,pairSpread[1?256:0],uint64_t(64),4,
       BinaryRepeatParams{1,1,1,8,0,8});
     if(1==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Preserve SDK count-mask preparation, bypass only the annotated tensor interface.
      // The first256 coefficient words are read-only;only the second256 are written.
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
      vaxpy(reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr())+256,
        reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr()),float(normalizedBelow),uint8_t(4),
        uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
   }
   if(n==15){
    Fence<HardEvent::V_S>();
    auto* words=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr());
    const float a00=words[12*pitch+12],a01=words[12*pitch+13],a02=words[12*pitch+14];
    const float a10=words[13*pitch+12],a11=words[13*pitch+13],a12=words[13*pitch+14];
    const float a20=words[14*pitch+12],a21=words[14*pitch+13],a22=words[14*pitch+14];
    product*=a00*(a11*a22-a12*a21)-a01*(a10*a22-a12*a20)+a02*(a10*a21-a11*a20);
   }else if(n&1){
    Fence<HardEvent::V_S>();
    // The preceding explicit V_S covers the final matrix value.
    product*=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(n-1)*pitch+n-1];
   }
   *reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())=product;Fence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0}); // Final PIPE_ALL covers output completion.
  }
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread,factorWide;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t pitch=16,cols=16;uint32_t n,cores;uint64_t batches;
};

class KernelTinyBoundPrefetch16 {
public:
 __aicore__ inline ~KernelTinyBoundPrefetch16(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  mat=LocalTensor<float>(TPosition::VECCALC,0,256);
  loaded=LocalTensor<float>(TPosition::VECCALC,1024,256);
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,2048,256);
  pivot=LocalTensor<float>(TPosition::VECCALC,3072,16);
  spread=LocalTensor<float>(TPosition::VECCALC,3136,128);
  bases=LocalTensor<int32_t>(TPosition::VECCALC,4352,32);
  paired=LocalTensor<float>(TPosition::VECCALC,4480,32);
  pairSpread=LocalTensor<float>(TPosition::VECCALC,4608,256);
  absolute=LocalTensor<float>(TPosition::VECCALC,3712,128);
  reduced=LocalTensor<float>(TPosition::VECCALC,4224,8);
  result=LocalTensor<float>(TPosition::VECCALC,4256,8);
  DataCopy(mat,input,256);
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64+4)<<32);
  Fence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr())[i]=int32_t(i*n*4);reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64+4)<<32);}
  Fence<HardEvent::S_V>();
  Adds(offsets[8],offsets,int32_t(8*n*4),8);Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  #pragma unroll
  for(uint32_t filled=16;filled<256;filled*=2){
   Adds(offsets[filled],offsets,int32_t(filled/16*4),filled);PipeBarrier<PIPE_V>();
  }
  }
 }
 __aicore__ inline void Load(uint64_t offset){
  // This class is reachable only for one matrix on owner0; Init issued its DMA.
  Fence<HardEvent::MTE2_V>();
 }
 __aicore__ inline void Process(){
  // TASK_MODE2/3 bind one matrix; the kernel entry admits only logical block0.
  {
   constexpr uint64_t batch=0;
   Load(0);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;
   // Explicit code generation, not a requested compiler loop unroll.
   do {
    { // pivot 0


    factor=mat[0*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-0-1;
    if((0&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[0];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[0+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(0+1)*pitch+0];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(0+1)*pitch+0+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[0];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((0&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(0+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       0*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(0&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     if((0&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(0+1)*pitch],factor,pairSpread[(0&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((0&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 1


    factor=mat[1*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-1-1;
    if((1&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[1];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[1+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(1+1)*pitch+1];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(1+1)*pitch+1+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[1];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((1&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(1+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       1*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(1&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8}); // Consumer auto-sync must retain the required V dependency.
     if((1&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(1+1)*pitch],factor,pairSpread[(1&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((1&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 2


    factor=mat[2*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-2-1;
    if((2&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[2];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[2+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(2+1)*pitch+2];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(2+1)*pitch+2+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[2];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((2&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(2+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       2*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(2&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     if((2&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(2+1)*pitch],factor,pairSpread[(2&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((2&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 3


    factor=mat[3*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-3-1;
    if((3&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[3];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[3+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(3+1)*pitch+3];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(3+1)*pitch+3+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[3];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((3&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(3+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       3*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(3&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8}); // Consumer auto-sync must retain the required V dependency.
     if((3&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(3+1)*pitch],factor,pairSpread[(3&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((3&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 4


    factor=mat[4*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-4-1;
    if((4&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[4];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[4+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(4+1)*pitch+4];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(4+1)*pitch+4+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[4];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((4&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(4+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       4*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(4&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     if((4&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(4+1)*pitch],factor,pairSpread[(4&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((4&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 5


    factor=mat[5*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-5-1;
    if((5&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[5];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[5+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(5+1)*pitch+5];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(5+1)*pitch+5+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[5];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((5&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(5+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       5*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(5&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8}); // Consumer auto-sync must retain the required V dependency.
     if((5&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(5+1)*pitch],factor,pairSpread[(5&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((5&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 6


    factor=mat[6*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-6-1;
    if((6&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[6];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[6+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(6+1)*pitch+6];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(6+1)*pitch+6+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[6];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((6&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(6+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       6*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(6&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     if((6&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(6+1)*pitch],factor,pairSpread[(6&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((6&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 7


    factor=mat[7*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-7-1;
    if((7&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[7];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[7+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(7+1)*pitch+7];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(7+1)*pitch+7+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[7];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((7&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(7+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       7*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(7&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8}); // Consumer auto-sync must retain the required V dependency.
     if((7&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(7+1)*pitch],factor,pairSpread[(7&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((7&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 8


    factor=mat[8*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-8-1;
    if((8&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[8];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[8+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(8+1)*pitch+8];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(8+1)*pitch+8+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[8];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((8&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(8+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       8*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(8&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     if((8&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(8+1)*pitch],factor,pairSpread[(8&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((8&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 9


    factor=mat[9*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-9-1;
    if((9&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[9];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[9+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(9+1)*pitch+9];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(9+1)*pitch+9+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[9];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((9&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(9+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       9*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(9&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8}); // Consumer auto-sync must retain the required V dependency.
     if((9&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(9+1)*pitch],factor,pairSpread[(9&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((9&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 10


    factor=mat[10*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-10-1;
    if((10&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[10];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[10+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(10+1)*pitch+10];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(10+1)*pitch+10+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[10];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((10&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(10+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       10*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(10&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     if((10&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(10+1)*pitch],factor,pairSpread[(10&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((10&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // pivot 11


    factor=mat[11*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-11-1;
    if((11&1)==0){
     Fence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[11];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[11+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(11+1)*pitch+11];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(11+1)*pitch+11+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[11];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((11&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(11+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       11*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(11&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8}); // Consumer auto-sync must retain the required V dependency.
     if((11&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(11+1)*pitch],factor,pairSpread[(11&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((11&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
       }
    { // Determinant of the live final 4x4 Schur complement.
     Fence<HardEvent::V_S>();
     auto* words=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr());
     const float a00=words[12*pitch+12];
     const float a01=words[12*pitch+13];
     const float a02=words[12*pitch+14];
     const float a03=words[12*pitch+15];
     const float a10=words[13*pitch+12];
     const float a11=words[13*pitch+13];
     const float a12=words[13*pitch+14];
     const float a13=words[13*pitch+15];
     const float a20=words[14*pitch+12];
     const float a21=words[14*pitch+13];
     const float a22=words[14*pitch+14];
     const float a23=words[14*pitch+15];
     const float a30=words[15*pitch+12];
     const float a31=words[15*pitch+13];
     const float a32=words[15*pitch+14];
     const float a33=words[15*pitch+15];
     const float s0=a00*a11-a01*a10;
     const float c0=a20*a31-a21*a30;
     const float s1=a00*a12-a02*a10;
     const float c1=a20*a32-a22*a30;
     const float s2=a00*a13-a03*a10;
     const float c2=a20*a33-a23*a30;
     const float s3=a01*a12-a02*a11;
     const float c3=a21*a32-a22*a31;
     const float s4=a01*a13-a03*a11;
     const float c4=a21*a33-a23*a31;
     const float s5=a02*a13-a03*a12;
     const float c5=a22*a33-a23*a32;
     const float determinant=s0*c5-s1*c4+s2*c3+s3*c2-s4*c1+s5*c0;
     product*=determinant;
    }
   } while(false);
   *reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())=product;Fence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0}); // Final PIPE_ALL covers output completion.
  }
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t n=16,pitch=16,cols=16;uint32_t cores;uint64_t batches;
};

class KernelMediumBoundCompact56 {
public:
 __aicore__ inline ~KernelMediumBoundCompact56(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void MakeStrides(LocalTensor<int32_t> target,uint32_t count,uint32_t step){
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ int32_t*>(target.GetPhyAddr())[i]=int32_t(i*step);
  Fence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<count;filled*=2){
   uint32_t len=count-filled<filled?count-filled:filled;
   Adds(target[filled],target,int32_t(filled*step),len);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;cols=(n+7)/8*8;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  uint32_t ioCount=cols*72,offset=0;
  mat=LocalTensor<float>(TPosition::VECCALC,offset,cols*72);offset+=cols*72*4;
  loaded=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  transposed=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  absolute=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  reduced=LocalTensor<float>(TPosition::VECCALC,offset,8);offset+=32;
  pivot=LocalTensor<float>(TPosition::VECCALC,offset,cols);offset+=cols*4;
  spread=LocalTensor<float>(TPosition::VECCALC,offset,cols*8);offset+=cols*32;
  bases=LocalTensor<int32_t>(TPosition::VECCALC,offset,128);offset+=512;
  paired=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  pairSpread=LocalTensor<float>(TPosition::VECCALC,offset,1024);offset+=4096;
  result=LocalTensor<float>(TPosition::VECCALC,offset,8);
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=
    uint64_t(2*i*pitch*4)|(uint64_t((2*i+1)*pitch*4)<<32);
  Fence<HardEvent::S_V>();
  for(uint32_t filled=16;filled<64;filled*=2){
   Adds(bases[filled],bases,int32_t(filled*pitch*4),filled);PipeBarrier<PIPE_V>();
  }
  Adds(bases[64],bases,int32_t(4),64);PipeBarrier<PIPE_V>();

 }
 __aicore__ inline void Load(uint64_t offset){
  // First input DMA has no prior writer or reader of loaded.
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},
    DataCopyPadExtParams<float>{true,0,uint8_t((8-n*n%8)%8),0.0f});

  if(!flatIndicesReady){
   // Only the live matrix cells contribute to the determinant.
   // Gather overwrites every live cell; padded lanes never feed live coefficients.

   // Build one 56-word row, then replicate the row map with GM stride n.
   auto offsets=indices.ReinterpretCast<int32_t>();
   MakeStrides(offsets,pitch,4);
   for(uint32_t rows=1;rows<n;rows*=2){
    uint32_t live=n-rows<rows?n-rows:rows;
    Adds(offsets[rows*pitch],offsets,int32_t(rows*n*4),live*pitch);
    PipeBarrier<PIPE_V>();
   }
   flatIndicesReady=true;
  }
  Fence<HardEvent::MTE2_V>();
  Gather(mat,loaded,indices,0,n*pitch);PipeBarrier<PIPE_V>();
 }
 template<uint32_t ColumnBegin,uint32_t ActiveWidth>
 __aicore__ inline void Eliminate(uint32_t first,uint32_t last,float& product){
  auto* words=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr());
  auto* coefficients=reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr());
  auto* expanded=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
  for(uint32_t p=first;p<last;p+=2){
   const uint32_t remaining=n-p-1;
   SetFlag<HardEvent::V_S>(static_cast<event_t>(0));
   // Fixed phase mask; dead extra rows stay within owned matrix/input UB and never enter live coefficients.
   AscendC::GatherImpl<float>(coefficients,words+(p+1)*pitch,
     reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),n*pitch,
     p*4,uint64_t(ColumnBegin==0?64:32),uint8_t(2),uint16_t(8));
   WaitFlag<HardEvent::V_S>(static_cast<event_t>(0));
   const float diagonal=words[p*pitch+p];
   const float below=words[p*pitch+p+1];
   const float cross=words[(p+1)*pitch+p];
   const float future=words[(p+1)*pitch+p+1];
   const float inverse=-1.0f/diagonal;
   // Gather must complete before Brcb. Factor scaling has independent storage.
   PipeBarrier<PIPE_V>();
   AscendC::BrcbImpl<float>(expanded,coefficients,uint8_t((remaining+7)/8),BrcbRepeatParams{1,8});
   AscendC::MulsImpl<float>(words+p*pitch+ColumnBegin,words+p*pitch+ColumnBegin,
     inverse,uint64_t(ActiveWidth),uint8_t(1),UnaryRepeatParams{1,1,8,8});
   PipeBarrier<PIPE_V>();
   AscendC::MulAddDstImpl<float,float>(words+(p+1)*pitch+ColumnBegin,
     words+p*pitch+ColumnBegin,expanded,uint64_t(ActiveWidth),uint8_t(remaining),
     BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
   const float normalizedBelow=below*inverse;
   const float nextDiagonal=normalizedBelow*cross+future;
   const float nextInverse=-1.0f/nextDiagonal;
   // Compact odd coefficient writes are disjoint from the first matrix update.
   SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
   vaxpy(coefficients+64,coefficients,float(normalizedBelow),uint8_t(1),
     uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
   PipeBarrier<PIPE_V>();
   if(remaining>1){
    // Odd broadcast and factor scaling can share their producer join.
    AscendC::BrcbImpl<float>(expanded+512,coefficients+64,uint8_t((remaining+7)/8),BrcbRepeatParams{1,8});
    AscendC::MulsImpl<float>(words+(p+1)*pitch+ColumnBegin,words+(p+1)*pitch+ColumnBegin,
      nextInverse,uint64_t(ActiveWidth),uint8_t(1),UnaryRepeatParams{1,1,8,8});
    PipeBarrier<PIPE_V>();
    AscendC::MulAddDstImpl<float,float>(words+(p+2)*pitch+ColumnBegin,
      words+(p+1)*pitch+ColumnBegin,expanded+520,uint64_t(ActiveWidth),uint8_t(remaining-1),
      BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
   }
   product*=diagonal;
   product*=nextDiagonal;
   PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Process(uint64_t batch){
  // The bound entry checked this owner; batches<=cores guarantees one matrix.
  {
   Load(batch*n*n);
   float product=1.0f;
   Eliminate<0,56>(0,24,product);
   Eliminate<24,32>(24,n-4,product);
   { // Direct determinant of the remaining 4x4 Schur complement.
    Fence<HardEvent::V_S>();
    const uint32_t tail=n-4;
    auto* words=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr());
    const float a00=words[(tail+0)*pitch+tail+0];
    const float a01=words[(tail+0)*pitch+tail+1];
    const float a02=words[(tail+0)*pitch+tail+2];
    const float a03=words[(tail+0)*pitch+tail+3];
    const float a10=words[(tail+1)*pitch+tail+0];
    const float a11=words[(tail+1)*pitch+tail+1];
    const float a12=words[(tail+1)*pitch+tail+2];
    const float a13=words[(tail+1)*pitch+tail+3];
    const float a20=words[(tail+2)*pitch+tail+0];
    const float a21=words[(tail+2)*pitch+tail+1];
    const float a22=words[(tail+2)*pitch+tail+2];
    const float a23=words[(tail+2)*pitch+tail+3];
    const float a30=words[(tail+3)*pitch+tail+0];
    const float a31=words[(tail+3)*pitch+tail+1];
    const float a32=words[(tail+3)*pitch+tail+2];
    const float a33=words[(tail+3)*pitch+tail+3];
    const float s0=a00*a11-a01*a10;
    const float c0=a20*a31-a21*a30;
    const float s1=a00*a12-a02*a10;
    const float c1=a20*a32-a22*a30;
    const float s2=a00*a13-a03*a10;
    const float c2=a20*a33-a23*a30;
    const float s3=a01*a12-a02*a11;
    const float c3=a21*a32-a22*a31;
    const float s4=a01*a13-a03*a11;
    const float c4=a21*a33-a23*a31;
    const float s5=a02*a13-a03*a12;
    const float c5=a22*a33-a23*a32;
    product*=s0*c5-s1*c4+s2*c3+s3*c2-s4*c1+s5*c0;
   }
   *reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())=product;Fence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0}); // Final PIPE_ALL covers output completion.
  }
 }
private:
 LocalTensor<float> mat,loaded,transposed,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 bool flatIndicesReady=false;
 static constexpr uint32_t pitch=56;uint32_t n,cols,cores;uint64_t batches;
};

class KernelMediumBoundStaticSuffix64 {
public:
 __aicore__ inline ~KernelMediumBoundStaticSuffix64(){PipeBarrier<PIPE_ALL>();}
 template<HardEvent E> __aicore__ inline void Fence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
 __aicore__ inline void MakeStrides(LocalTensor<int32_t> target,uint32_t count,uint32_t step){
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ int32_t*>(target.GetPhyAddr())[i]=int32_t(i*step);
  Fence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<count;filled*=2){
   uint32_t len=count-filled<filled?count-filled:filled;
   Adds(target[filled],target,int32_t(filled*step),len);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;cols=(n+7)/8*8;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  uint32_t ioCount=n<=64?cols*pitch:8*pitch,offset=0;
  mat=LocalTensor<float>(TPosition::VECCALC,offset,cols*pitch);offset+=cols*pitch*4;
  loaded=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  transposed=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  absolute=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  reduced=LocalTensor<float>(TPosition::VECCALC,offset,8);offset+=32;
  pivot=LocalTensor<float>(TPosition::VECCALC,offset,cols);offset+=cols*4;
  spread=LocalTensor<float>(TPosition::VECCALC,offset,cols*8);offset+=cols*32;
  bases=LocalTensor<int32_t>(TPosition::VECCALC,offset,128);offset+=512;
  paired=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  pairSpread=LocalTensor<float>(TPosition::VECCALC,offset,1024);offset+=4096;
  result=LocalTensor<float>(TPosition::VECCALC,offset,8);
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*pitch*4)|(uint64_t(i*pitch*4+4)<<32);
  Fence<HardEvent::S_V>();
  for(uint32_t filled=16;filled<128;filled*=2){Adds(bases[filled],bases,int32_t(filled/2*pitch*4),filled);PipeBarrier<PIPE_V>();}

 }
 __aicore__ inline void Load(uint64_t offset){
  if((n&7)==0){
   // Live row words are contiguous in GM; inactive row padding is never consumed.
   // First input DMA has no prior writer or reader of loaded.
   DataCopy(loaded,input[offset],n*n);
   Fence<HardEvent::MTE2_V>();
   DataCopy(mat,loaded,DataCopyParams{uint16_t(n),uint16_t(n/8),0,uint16_t((pitch-n)/8)});
   PipeBarrier<PIPE_V>();
   return;
  }
  // The large contiguous transfer runs while the shape-only address map is built.
  // First input DMA has no prior writer or reader of loaded.
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},
    DataCopyPadExtParams<float>{true,0,uint8_t((8-n*n%8)%8),0.0f});
  Duplicate(loaded[(n*n+7)/8*8],0.0f,64);
  if(!flatIndicesReady){
   MakeStrides(indices.ReinterpretCast<int32_t>(),n*64,4);
   auto temporary=transposed.ReinterpretCast<int32_t>();
   ShiftRight(temporary,indices.ReinterpretCast<int32_t>(),8,n*64);PipeBarrier<PIPE_V>();
   Muls(temporary,temporary,(int32_t(n)-64)*4,n*64);PipeBarrier<PIPE_V>();
   Add(indices.ReinterpretCast<int32_t>(),indices.ReinterpretCast<int32_t>(),temporary,n*64);PipeBarrier<PIPE_V>();
   flatIndicesReady=true;
  }
  Fence<HardEvent::MTE2_V>();
  Gather(transposed,loaded,indices,0,n*64);PipeBarrier<PIPE_V>();
  DataCopy(mat,transposed,DataCopyParams{uint16_t(n),8,0,1});PipeBarrier<PIPE_V>();
 }
 template<uint32_t ColumnBegin,uint32_t ActiveWidth>
 __aicore__ inline void Eliminate(uint32_t first,uint32_t last,float& product){
  float cachedDiagonal=0.0f,cachedInverse=0.0f;
   for(uint32_t pairBase=first;pairBase<last;pairBase+=2){
    {const uint32_t p=pairBase;
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(0==0){
     // The event covers prior matrix producers; Gather and scalar reads only read them.
     event_t priorMatrixReady=static_cast<event_t>(0);
     SetFlag<HardEvent::V_S>(priorMatrixReady);
     if(remaining){
      // Matrix plus the initialized input copy are owned by this core.
      // Extra gathered words never enter a live coefficient or matrix update.
      auto readable=LocalTensor<float>(TPosition::VECCALC,0,64*72+64*64);
      Gather(paired,readable[(p+1)*pitch],bases.ReinterpretCast<uint32_t>(),p*4,
        uint64_t(64),ColumnBegin==0?2:1,8);
     }
     WaitFlag<HardEvent::V_S>(priorMatrixReady);
     diagonal=factor.GetValue(p);
     if(remaining){
      below=factor.GetValue(p+1);
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }
    }else diagonal=cachedDiagonal;
    // No per-pivot zero guard in this bound performance path.
    if(remaining){
     if(0==0){PipeBarrier<PIPE_V>();Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});}
     float divisor=diagonal;
     float inverse=0?cachedInverse:-1.0f/divisor;
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       inverse,uint64_t(ActiveWidth),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     MulAddDst<float,float,false>(mat[(p+1)*pitch+ColumnBegin],factor[ColumnBegin],pairSpread[0?24:0],uint64_t(ActiveWidth),remaining,
       BinaryRepeatParams{1,1,0,9,0,2});
     if(0==0){
      float normalizedBelow=below*inverse;
      cachedDiagonal=normalizedBelow*cross+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // VMLA reads only even coefficient blocks. This writes disjoint odd blocks.
      // The existing loop-end V barrier covers both before the next dependent update.
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,normalizedBelow,uint8_t((remaining+7)/8),
        uint16_t(2),uint16_t(2),uint16_t(16),uint16_t(16));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
    }
    {const uint32_t p=pairBase+1;
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(1==0){
     // The event covers prior matrix producers; Gather and scalar reads only read them.
     event_t priorMatrixReady=static_cast<event_t>(0);
     SetFlag<HardEvent::V_S>(priorMatrixReady);
     if(remaining){
      // Matrix plus the initialized input copy are owned by this core.
      // Extra gathered words never enter a live coefficient or matrix update.
      auto readable=LocalTensor<float>(TPosition::VECCALC,0,64*72+64*64);
      Gather(paired,readable[(p+1)*pitch],bases.ReinterpretCast<uint32_t>(),p*4,
        uint64_t(64),ColumnBegin==0?2:1,8);
     }
     WaitFlag<HardEvent::V_S>(priorMatrixReady);
     diagonal=factor.GetValue(p);
     if(remaining){
      below=factor.GetValue(p+1);
      cross=mat[(p+1)*pitch].GetValue(p);
      future=mat[(p+1)*pitch].GetValue(p+1);
     }
    }else diagonal=cachedDiagonal;
    // No per-pivot zero guard in this bound performance path.
    if(remaining){
     if(1==0){PipeBarrier<PIPE_V>();Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});}
     float divisor=diagonal;
     float inverse=1?cachedInverse:-1.0f/divisor;
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       inverse,uint64_t(ActiveWidth),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     MulAddDst<float,float,false>(mat[(p+1)*pitch+ColumnBegin],factor[ColumnBegin],pairSpread[1?24:0],uint64_t(ActiveWidth),remaining,
       BinaryRepeatParams{1,1,0,9,0,2});
     if(1==0){
      float normalizedBelow=below*inverse;
      cachedDiagonal=normalizedBelow*cross+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // VMLA reads only even coefficient blocks. This writes disjoint odd blocks.
      // The existing loop-end V barrier covers both before the next dependent update.
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,normalizedBelow,uint8_t((remaining+7)/8),
        uint16_t(2),uint16_t(2),uint16_t(16),uint16_t(16));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
    }
   }
 }
 __aicore__ inline void Process(uint64_t batch){
  // The bound entry admits each owner once, batches<=cores.
  {
   Load(batch*n*n);
   float product=1.0f;
   Eliminate<0,64>(0,32,product);
   Eliminate<32,32>(32,n,product);
   *reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())=product;Fence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0}); // Final PIPE_ALL covers output completion.
  }
 }
private:
 LocalTensor<float> mat,loaded,transposed,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 bool flatIndicesReady=false;
 static constexpr uint32_t pitch=72;uint32_t n,cols,cores;uint64_t batches;
};

template<typename DT_A,uint32_t TASK_MODE>
__global__ __aicore__ void linalg_det(GM_ADDR A,GM_ADDR det,GM_ADDR workspace,GM_ADDR tiling){
 constexpr bool tiny=TASK_MODE>=2&&TASK_MODE<=17;
 constexpr bool fullVector=(TASK_MODE>=32&&TASK_MODE<=39)||(TASK_MODE>=160&&TASK_MODE<=183);
 constexpr bool compact=TASK_MODE>=48&&TASK_MODE<=79;
 constexpr bool alignedLarge=TASK_MODE>=96&&TASK_MODE<=159;
 LinalgDetTilingData t;
 if constexpr(tiny){
  t.n=TASK_MODE-1;t.cores=8;t.batches=1;t.resident=t.n<16?1:0;
 }else if constexpr(fullVector){
  if constexpr(TASK_MODE>=160){t.n=58+2*((TASK_MODE-160)/8);t.batches=(TASK_MODE-160)%8+1;}
  else {t.n=64;t.batches=TASK_MODE-32+1;}
  t.cores=8;t.resident=0;
 }else if constexpr(compact){
  t.n=50+2*((TASK_MODE-48)/8);t.cores=8;t.batches=(TASK_MODE-48)%8+1;t.resident=0;
 }else if constexpr(alignedLarge){
  t.n=(17+(TASK_MODE-96)/4)*8;t.cores=16;t.batches=(TASK_MODE-96)%4+1;t.resident=0;
 }else{
  REGISTER_TILING_DEFAULT(LinalgDetTilingData);GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData,data,tiling);t=data;
 }
 uint64_t owner=0;
 if constexpr(fullVector||compact){owner=uint64_t(GetBlockIdx());if(owner>=t.batches)return;}
 else if(t.n<=64&&uint64_t(GetBlockIdx())>=t.batches)return;
 if constexpr(TASK_MODE==1||alignedLarge){FastBlockProcess(A,det,GetUserWorkspace(workspace),t.n,t.batches,t.cores/4);}
 else if constexpr(tiny){
  if constexpr(TASK_MODE==17){KernelTinyBoundPrefetch16 op;op.Init(A,det,t);op.Process();}
  else if constexpr(TASK_MODE==16){KernelTinyBoundTerminal15 op;op.Init(A,det,t);op.Process();}
  else if constexpr(TASK_MODE>=10){KernelTinyBoundPrefetchPadded16 op;op.Init(A,det,t);op.Process();}
  else {KernelTinyFullwidthDet op;op.Init(A,det,t);op.Process();}
 }else if constexpr(fullVector){KernelMediumBoundStaticSuffix64 op;op.Init(A,det,t);op.Process(owner);}
 else if constexpr(compact){KernelMediumBoundCompact56 op;op.Init(A,det,t);op.Process(owner);}
 else if(t.n==16){KernelTinyExplicit16 op;op.Init(A,det,t);op.Process();}
 else if(t.n<=16){if(t.resident){KernelTinyFullwidthDet op;op.Init(A,det,t);op.Process();}else{KernelTinyDet op;op.Init(A,det,t);op.Process();}}
 else if(t.n>=33&&t.n<=64){KernelMediumDet op;op.Init(A,det,t);op.Process();}
 else {KernelColumnDet op;op.Init(A,det,t);op.Process();}
}

// The bound singleton kernels use only UB and explicit input/output pointers.
// Keep system workspace registration for all other modes and debug facilities.
// CANN 9 appends its generated wrapper after this source and defines
// TEMPLATE_PARAMS before invoking SetSysWorkspaceForce in that wrapper.
namespace AscendC {
template<typename DT_A, uint32_t TASK_MODE>
__aicore__ inline void DetSetSysWorkspaceForMode(GM_ADDR workspace) {
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
 SetSysWorkspaceForce(workspace);
#else
 if constexpr(!((TASK_MODE>=2&&TASK_MODE<=17)||(TASK_MODE>=32&&TASK_MODE<=39)||(TASK_MODE>=48&&TASK_MODE<=79)||(TASK_MODE>=160&&TASK_MODE<=183))) SetSysWorkspaceForce(workspace);
#endif
}
}
#define SetSysWorkspaceForce(workspace) DetSetSysWorkspaceForMode<TEMPLATE_PARAMS>(workspace)

// The same singleton modes perform no cross-core synchronization. These
// adapters affect only the generated calls after the user source is included.
template<typename DT_A, uint32_t TASK_MODE>
__aicore__ inline void DetSetFftsForMode(uint64_t address) {
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
 set_ffts_base_addr(address);
#else
 if constexpr(!((TASK_MODE>=2&&TASK_MODE<=17)||(TASK_MODE>=32&&TASK_MODE<=39)||(TASK_MODE>=48&&TASK_MODE<=79)||(TASK_MODE>=160&&TASK_MODE<=183))) set_ffts_base_addr(address);
#endif
}
template<typename DT_A, uint32_t TASK_MODE>
__aicore__ inline void DetGeneratedPreloadForMode(uint64_t mode) {
 (void)mode; // The audited CANN9 generated call always passes1.
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
 icache_preload(1);
#else
 if constexpr(!((TASK_MODE>=2&&TASK_MODE<=17)||(TASK_MODE>=32&&TASK_MODE<=39)||(TASK_MODE>=48&&TASK_MODE<=79)||(TASK_MODE>=160&&TASK_MODE<=183))) icache_preload(1);
#endif
}
#define set_ffts_base_addr(address) DetSetFftsForMode<TEMPLATE_PARAMS>(address)
#define icache_preload(mode) DetGeneratedPreloadForMode<TEMPLATE_PARAMS>(mode)
