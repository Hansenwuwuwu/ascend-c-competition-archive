#include "kernel_operator.h"
#include <cmath>
#include "lib/matmul_intf.h"
#include "linalg_pinv_tiling.h"
#include "tiling_key_linalg_pinv.h"
using namespace AscendC;
using FP32Cube = Matmul<MatmulType<TPosition::GM,CubeFormat::ND,float>,MatmulType<TPosition::GM,CubeFormat::ND,float>,MatmulType<TPosition::GM,CubeFormat::ND,float>>;
template<HardEvent H> __aicore__ inline void LocalSync(){SetFlag<H>(EVENT_ID0);WaitFlag<H>(EVENT_ID0);}
__aicore__ inline void FastTranspose8x256(LocalTensor<float> dst,LocalTensor<float> src,LocalTensor<float> scratch){
LocalTensor<float> destinations[16],sources[16];
#pragma unroll
for(uint32_t i=0;i<8;i++){destinations[2*i]=dst[i*8];destinations[2*i+1]=scratch[i*8];sources[i]=src[i*256];sources[i+8]=src[i*256];}
TransDataTo5HD(destinations,sources,{false,false,32,8,1});
}
__aicore__ inline void FastTranspose16x256(LocalTensor<float> dst,LocalTensor<float> src){
LocalTensor<float> destinations[16],sources[16];
#pragma unroll
for(uint32_t i=0;i<8;i++){destinations[2*i]=dst[i*16];destinations[2*i+1]=dst[i*16+8];sources[i]=src[i*256];sources[i+8]=src[(i+8)*256];}
TransDataTo5HD(destinations,sources,{false,false,32,16,1});
}
template<class T> class Pinv {
TPipe pipe;TBuf<QuePosition::VECCALC> buf;
GlobalTensor<T> input,output;GlobalTensor<float> work,vg;
LocalTensor<float> mem,wc,vc,u,v,h,hv,a,b,tmp,tmp2,red,d,e,norms,out;
LocalTensor<T> raw;LocalTensor<uint32_t> offsets,bfOffsets;
LinalgPinvTilingData t;uint32_t batch,base;
__aicore__ inline float Fabs(float x){return x<0?-x:x;}
__aicore__ inline float Root(float x){red.SetValue(0,x>0?x:0);PipeBarrier<PIPE_ALL>();Sqrt(red,red,1);PipeBarrier<PIPE_ALL>();return red.GetValue(0);}
__aicore__ inline float Hyp(float x,float y){x=Fabs(x);y=Fabs(y);float z=x>y?x:y;if(z==0)return 0;x/=z;y/=z;return z*Root(x*x+y*y);}
__aicore__ inline float Signed(float x,float y){return y<0?-Fabs(x):Fabs(x);}
__aicore__ inline void Fence(){PipeBarrier<PIPE_ALL>();}
__aicore__ inline float Dot(LocalTensor<float> x,LocalTensor<float> y,uint32_t len){
Mul(tmp,x,y,len);PipeBarrier<PIPE_V>();ReduceSum(red,tmp,tmp2,len);Fence();return red.GetValue(0);
}
__aicore__ inline float MaxAbs(LocalTensor<float> x,uint32_t len){
Abs(tmp,x,len);PipeBarrier<PIPE_V>();uint32_t full=len/64,tail=len%64;
if(full)WholeReduceMax(red,tmp,64,full,1,1,8);
if(tail)WholeReduceMax(red[full*2],tmp[full*64],tail,1,1,1,8);Fence();
float largest=0;for(uint32_t i=0;i<full+(tail?1:0);i++){float z=red.GetValue(i*2);if(z>largest)largest=z;}return largest;
}
__aicore__ inline void ReadInput(LocalTensor<float> dst,uint32_t col){
Duplicate(dst,0.0f,t.qp);Fence();
if(t.m<t.n){
if constexpr(sizeof(T)==4)DataCopyPad(dst,input[base+col*t.q],{1,t.q*4,0,0,0},{true,0,(uint8_t)(t.qp-t.q),0});
else{DataCopyPad(raw,input[base+col*t.q],{1,t.q*2,0,0,0},{false,0,0,0});Fence();Cast(dst,raw,RoundMode::CAST_NONE,t.q);}
}else{
if constexpr(sizeof(T)==4){DataCopyPad(raw,input[base+col],{(uint16_t)t.q,4,(t.p-1)*4,0,0},{false,0,0,0});Fence();Gather(dst,raw.template ReinterpretCast<float>(),offsets,0,t.q);}
else for(uint32_t start=0;start<t.q;start+=32){
uint32_t count=t.q-start<32?t.q-start:32;auto converted=mem[36864];
DataCopyPad(raw,input[base+col+start*t.p],{(uint16_t)count,2,(t.p-1)*2,0,0},{false,0,0,0});Fence();
Cast(converted,raw,RoundMode::CAST_NONE,count*16);Fence();Gather(dst[start],converted,bfOffsets,0,count);Fence();
}
}Fence();
}
__aicore__ inline void ReadCol(bool right,uint32_t col,LocalTensor<float> dst){
uint32_t len=right?t.pp:t.qp;Fence();
if(t.cached)Adds(dst,(right?vc:wc)[col*len],0.0f,len);
else DataCopy(dst,(right?vg:work)[col*len],len);
Fence();
}
__aicore__ inline void WriteCol(bool right,uint32_t col,LocalTensor<float> src){
uint32_t len=right?t.pp:t.qp;Fence();
if(t.cached)Adds((right?vc:wc)[col*len],src,0.0f,len);
else DataCopy((right?vg:work)[col*len],src,len);
Fence();
}
__aicore__ inline void Rotate(bool right,uint32_t j,uint32_t k,float c,float s){
uint32_t len=right?t.pp:t.qp;ReadCol(right,j,a);ReadCol(right,k,b);
Muls(out,a,c,len);Muls(tmp2,b,s,len);PipeBarrier<PIPE_V>();Sub(out,out,tmp2,len);
Muls(b,b,c,len);Muls(tmp2,a,s,len);PipeBarrier<PIPE_V>();Add(b,b,tmp2,len);Fence();
WriteCol(right,j,out);WriteCol(right,k,b);
}
__aicore__ inline void WriteOutput(LocalTensor<float> src,uint32_t at,uint32_t len){
Fence();if constexpr(sizeof(T)==4)DataCopyPad(output[at],src,{1,len*4,0,0,0});
else{Cast(raw,src,RoundMode::CAST_RINT,len);Fence();DataCopyPad(output[at],raw,{1,len*2,0,0,0});}Fence();
}
template<uint32_t N=0> __aicore__ inline bool FastInverse(){
if(t.m!=t.n || t.p>64)return false;
uint32_t n=N?N:t.p,np=N?(N+7)/8*8:t.pp,span=np*2,width=span+8;auto mat=mem;auto u=mem[38408],v=mem[38928],h=mem[39448],tmp=mem[41496],red=mem[42528];
Duplicate(mat,0.0f,n*width);Fence();
if constexpr(sizeof(T)==4)DataCopyPad(mat,input[base],{(uint16_t)n,n*4,0,(width-np)/8,0},{true,0,(uint8_t)(np-n),0});
else for(uint32_t i=0;i<n;i++){DataCopyPad(raw,input[base+i*n],{1,n*2,0,0,0},{false,0,0,0});Fence();Cast(mat[i*width],raw,RoundMode::CAST_NONE,n);}Fence();
Mul(mem[16384],mat,mat,n*width);PipeBarrier<PIPE_V>();ReduceSum(red,mem[16384],mem[24576],n*width);LocalSync<HardEvent::V_S>();float anorm=red.GetValue(0);
if(anorm==0)return false;
for(uint32_t i=0;i<n;i++){mat.SetValue(i*width+np+i,1.0f);offsets.SetValue(i,i*width*4);}Fence();
for(uint32_t j=0;j<n;j++){
Gather(u,mat,offsets,j*4,n);PipeBarrier<PIPE_V>();Abs(tmp,u,n);PipeBarrier<PIPE_V>();
uint64_t mask[2]={((~0ULL)<<j)&(n==64?~0ULL:((1ULL<<n)-1)),0};WholeReduceMax(red,tmp,mask,1,1,1,8);LocalSync<HardEvent::V_S>();
float mx=red.GetValue(0);uint32_t pi=red.template ReinterpretCast<uint32_t>().GetValue(1);
if(mx*mx<anorm*1e-10f)return false;
float inv=1.0f/u.GetValue(pi);Muls(v,mat[pi*width],-inv,span);Brcb(h,u,(n+7)/8,{1,8});PipeBarrier<PIPE_V>();
for(uint32_t c=j/8*8;c<span;c+=64){uint32_t count=span-c<64?span-c:64;MulAddDst(mat[c],v[c],h,count,n,{1,1,0,(uint8_t)(width/8),0,1});}
PipeBarrier<PIPE_V>();if(pi!=j){Adds(mat[pi*width],mat[j*width],0.0f,width);PipeBarrier<PIPE_V>();}
Muls(mat[j*width],v,-1.0f,span);PipeBarrier<PIPE_V>();
}
Fence(); // Augmented Frobenius norm is a conservative inverse-norm bound.
Mul(mem[16384],mat,mat,n*width);PipeBarrier<PIPE_V>();ReduceSum(red,mem[16384],mem[24576],n*width);LocalSync<HardEvent::V_S>();float ix=red.GetValue(0);
float threshold=t.atol*t.atol>anorm*t.rtol*t.rtol?t.atol*t.atol:anorm*t.rtol*t.rtol;
if(!(threshold*ix<0.25f))return false;
if constexpr(sizeof(T)==4){LocalSync<HardEvent::V_MTE3>();DataCopyPad(output[base],mat[np],{(uint16_t)n,n*4,(width-np)/8,0,0});LocalSync<HardEvent::MTE3_MTE2>();}
else for(uint32_t i=0;i<n;i++)WriteOutput(mat[i*width+np],base+i*n,n);
return true;
}
__aicore__ inline void Bidiagonal(){
Duplicate(d,0.0f,t.pp);Duplicate(e,0.0f,t.pp);Fence();
for(uint32_t k=0;k<t.p;k++){
ReadCol(false,k,u);if(k)Duplicate(u,0.0f,k);Fence();float scale=MaxAbs(u,t.qp);
if(scale>0){
Muls(u,u,1.0f/scale,t.qp);Fence();float ss=Dot(u,u,t.qp);float f=u.GetValue(k),g=-Signed(Root(ss),f),beta=f*g-ss;
u.SetValue(k,f-g);d.SetValue(k,g*scale);Fence();
for(uint32_t j=k+1;j<t.p;j++){ReadCol(false,j,a);float factor=Dot(u,a,t.qp)/beta;Axpy(a,u,factor,t.qp);Fence();WriteCol(false,j,a);}
}
if(k+1==t.p)continue;
Duplicate(v,0.0f,t.pp);Fence();
for(uint32_t j=k+1;j<t.p;j++){ReadCol(false,j,a);v.SetValue(j,a.GetValue(k));}Fence();scale=MaxAbs(v,t.pp);
if(scale>0){
Muls(v,v,1.0f/scale,t.pp);Fence();float ss=Dot(v,v,t.pp),f=v.GetValue(k+1),g=-Signed(Root(ss),f),beta=f*g-ss;
v.SetValue(k+1,f-g);e.SetValue(k+1,g*scale);Duplicate(h,0.0f,t.qp);Duplicate(hv,0.0f,t.pp);Fence();
for(uint32_t j=k+1;j<t.p;j++){
float factor=v.GetValue(j);ReadCol(false,j,a);ReadCol(true,j,b);Axpy(h,a,factor,t.qp);Axpy(hv,b,factor,t.pp);Fence();
}
if(k+1)Duplicate(h,0.0f,k+1);Fence();Muls(h,h,1.0f/beta,t.qp);Muls(hv,hv,1.0f/beta,t.pp);Fence();
for(uint32_t j=k+1;j<t.p;j++){
float factor=v.GetValue(j);ReadCol(false,j,a);ReadCol(true,j,b);Axpy(a,h,factor,t.qp);Axpy(b,hv,factor,t.pp);Fence();WriteCol(false,j,a);WriteCol(true,j,b);
}
}
}
}
__aicore__ inline void DiagonalQR(){
float anorm=0;for(uint32_t i=0;i<t.p;i++){float z=Fabs(d.GetValue(i))+Fabs(e.GetValue(i));if(z>anorm)anorm=z;}float tol=anorm*1.1920928955078125e-7f;
for(int k=int(t.p)-1;k>=0;k--){
for(uint32_t it=0;it<80;it++){
bool flag=true;int l=k;
for(;l>=0;l--){if(Fabs(e.GetValue(l))<=tol){flag=false;break;}if(l==0 || Fabs(d.GetValue(l-1))<=tol)break;}
if(flag){float c=0,s=1;for(int i=l;i<=k;i++){float f=s*e.GetValue(i);e.SetValue(i,c*e.GetValue(i));if(Fabs(f)<=tol)break;float g=d.GetValue(i),z=Hyp(f,g);d.SetValue(i,z);c=g/z;s=-f/z;}}
float z=d.GetValue(k);
if(l==k){if(z<0){d.SetValue(k,-z);ReadCol(true,k,a);Muls(a,a,-1.0f,t.pp);Fence();WriteCol(true,k,a);}break;}
float x=d.GetValue(l),y=d.GetValue(k-1),g=e.GetValue(k-1),hh=e.GetValue(k);
float f=((y-z)*(y+z)+(g-hh)*(g+hh))/(2*hh*y);g=Hyp(f,1);
f=((x-z)*(x+z)+hh*(y/(f+Signed(g,f))-hh))/x;float c=1,s=1;
for(int j=l;j<k;j++){
int i=j+1;g=e.GetValue(i);y=d.GetValue(i);hh=s*g;g=c*g;z=Hyp(f,hh);e.SetValue(j,z);
c=z?f/z:1;s=z?hh/z:0;f=x*c+g*s;g=g*c-x*s;hh=y*s;y*=c;
Rotate(true,j,i,c,-s);z=Hyp(f,hh);d.SetValue(j,z);if(z){c=f/z;s=hh/z;}
f=c*g+s*y;x=c*y-s*g;
}e.SetValue(l,0.0f);e.SetValue(k,f);d.SetValue(k,x);
}
}
}
__aicore__ inline void RebuildAndPolish(){
for(uint32_t j=0;j<t.p;j++){
ReadCol(true,j,v);Duplicate(h,0.0f,t.qp);Fence();
for(uint32_t k=0;k<t.p;k++){ReadInput(a,k);Axpy(h,a,v.GetValue(k),t.qp);Fence();}
WriteCol(false,j,h);
}
for(uint32_t j=0;j<t.p;j++)for(uint32_t k=j+1;k<t.p;k++){
ReadCol(false,j,u);ReadCol(false,k,v);float aa=Dot(u,u,t.qp),bb=Dot(v,v,t.qp),ab=Dot(u,v,t.qp);
if(aa<=0 || bb<=0 || Fabs(ab)<=2e-7f*Root(aa*bb))continue;
float tau=(bb-aa)/(2*ab),tt=Signed(1.0f,tau)/(Fabs(tau)+Hyp(1.0f,tau)),c=1.0f/Root(1+tt*tt),s=tt*c;
Rotate(false,j,k,c,s);Rotate(true,j,k,c,s);
}
}
__aicore__ inline void Assemble(){
float largest=0;for(uint32_t j=0;j<t.p;j++){float z=Fabs(d.GetValue(j));if(z>largest)largest=z;}
float threshold=t.atol>largest*t.rtol?t.atol:largest*t.rtol;
for(uint32_t j=0;j<t.p;j++){ReadCol(false,j,a);float sq=Dot(a,a,t.qp);norms.SetValue(j,(Fabs(d.GetValue(j))>threshold && sq>0)?1.0f/sq:0.0f);}Fence();
if(t.m>=t.n){
for(uint32_t j=0;j<t.p;j++){
Duplicate(out,0.0f,t.qp);Fence();
for(uint32_t k=0;k<t.p;k++){float factor=norms.GetValue(k);if(factor==0)continue;ReadCol(true,k,v);factor*=v.GetValue(j);ReadCol(false,k,a);Axpy(out,a,factor,t.qp);Fence();}
WriteOutput(out,base+j*t.q,t.q);
}
}else{
for(uint32_t j=0;j<t.q;j++){
Duplicate(out,0.0f,t.pp);Fence();
for(uint32_t k=0;k<t.p;k++){float factor=norms.GetValue(k);if(factor==0)continue;ReadCol(false,k,a);factor*=a.GetValue(j);ReadCol(true,k,v);Axpy(out,v,factor,t.pp);Fence();}
WriteOutput(out,base+j*t.p,t.p);
}
}
}
public:
__aicore__ inline void Init(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData tt){
t=tt;input.SetGlobalBuffer((__gm__ T*)ai);output.SetGlobalBuffer((__gm__ T*)ao);work.SetGlobalBuffer((__gm__ float*)ws);
pipe.InitBuffer(buf,194560);mem=buf.Get<float>();wc=mem;vc=mem[t.p*t.qp];
u=mem[38400];v=mem[38912];h=mem[39424];hv=mem[39936];a=mem[40448];b=mem[40960];tmp=mem[41472];tmp2=mem[41984];red=mem[42496];d=mem[43008];e=mem[43520];norms=mem[44032];out=mem[44544];raw=mem.template ReinterpretCast<T>()[32768*4/sizeof(T)];offsets=mem.template ReinterpretCast<uint32_t>()[45568];bfOffsets=mem.template ReinterpretCast<uint32_t>()[46080];
}
__aicore__ inline void Run(bool tryFast=true,uint32_t start=~0U){
auto allwork=work;
for(batch=start==~0U?GetBlockIdx():start;batch<t.batch;batch+=GetBlockNum()){
base=batch*t.m*t.n;work=allwork[batch*t.p*(t.qp+t.pp)];vg=work[t.p*t.qp];
bool fast=false;if(tryFast && t.m==t.n)fast=FastInverse<>();if(fast)continue;
for(uint32_t i=0;i<t.q;i++)offsets.SetValue(i,i*32);for(uint32_t i=0;i<32;i++)bfOffsets.SetValue(i,i*64);Fence();
for(uint32_t j=0;j<t.p;j++){ReadInput(u,j);WriteCol(false,j,u);Duplicate(v,0.0f,t.pp);Fence();v.SetValue(j,1.0f);Fence();WriteCol(true,j,v);}
Bidiagonal();DiagonalQR();RebuildAndPolish();Assemble();
}
}
};
template<class T,uint32_t N> class TinyPinv {
GlobalTensor<T> input,output;LocalTensor<float> mem;LocalTensor<uint32_t> offsets;LocalTensor<T> raw;
LinalgPinvTilingData t;uint32_t base;
__aicore__ inline void Fence(){PipeBarrier<PIPE_ALL>();}
__aicore__ inline void Sum(LocalTensor<float> dst,LocalTensor<float> src,uint32_t len){
auto first=mem[30000],second=mem[30256];uint32_t full=len/64,tail=len%64;
if(full)WholeReduceSum(first,src,64,full,1,1,8);if(tail)WholeReduceSum(first[full],src[full*64],tail,1,1,1,8);PipeBarrier<PIPE_V>();
uint32_t count=full+(tail?1:0);if(count<=64)WholeReduceSum(dst,first,count,1,1,1,8);
else{full=count/64;tail=count%64;WholeReduceSum(second,first,64,full,1,1,8);if(tail)WholeReduceSum(second[full],first[full*64],tail,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(dst,second,full+(tail?1:0),1,1,1,8);}PipeBarrier<PIPE_V>();
}
__aicore__ inline void WriteOutput(LocalTensor<float> src,uint32_t at,uint32_t len){Fence();if constexpr(sizeof(T)==4)DataCopyPad(output[at],src,{1,len*4,0,0,0});else{Cast(raw,src,RoundMode::CAST_RINT,len);Fence();DataCopyPad(output[at],raw,{1,len*2,0,0,0});}Fence();}
__aicore__ inline bool FastInverse(){
uint32_t n=N,np=(N+7)/8*8,span=np*2,width=span+8;auto mat=mem;auto u=mem[38408],v=mem[38928],h=mem[39448],tmp=mem[41496],red=mem[42528];
Duplicate(mat,0.0f,n*width);Fence();
if constexpr(sizeof(T)==4)DataCopyPad(mat,input[base],{(uint16_t)n,n*4,0,(width-np)/8,0},{true,0,(uint8_t)(np-n),0});
else for(uint32_t i=0;i<n;i++){DataCopyPad(raw,input[base+i*n],{1,n*2,0,0,0},{false,0,0,0});Fence();Cast(mat[i*width],raw,RoundMode::CAST_NONE,n);}Fence();
Mul(mem[16384],mat,mat,np,n,{1,1,1,(uint8_t)(np/8),(uint8_t)(width/8),(uint8_t)(width/8)});PipeBarrier<PIPE_V>();Sum(red,mem[16384],n*np);LocalSync<HardEvent::V_S>();float anorm=red.GetValue(0);
if(!(anorm>0))return false;
for(uint32_t i=0;i<n;i++){mat.SetValue(i*width+np+i,1.0f);offsets.SetValue(i,i*width*4);}Fence();
for(uint32_t j=0;j<n;j++){
Gather(u,mat,offsets,j*4,n);PipeBarrier<PIPE_V>();Abs(tmp,u,n);PipeBarrier<PIPE_V>();
uint64_t mask[2]={((~0ULL)<<j)&(n==64?~0ULL:((1ULL<<n)-1)),0};WholeReduceMax(red,tmp,mask,1,1,1,8);LocalSync<HardEvent::V_S>();
float mx=red.GetValue(0);uint32_t pi=red.template ReinterpretCast<uint32_t>().GetValue(1);
if(mx*mx<anorm*1e-10f)return false;
float inv=1.0f/u.GetValue(pi);Muls(v,mat[pi*width],-inv,span);Brcb(h,u,(n+7)/8,{1,8});PipeBarrier<PIPE_V>();
for(uint32_t c=j/8*8;c<span;c+=64){uint32_t count=span-c<64?span-c:64;MulAddDst(mat[c],v[c],h,count,n,{1,1,0,(uint8_t)(width/8),0,1});}
PipeBarrier<PIPE_V>();if(pi!=j){Adds(mat[pi*width],mat[j*width],0.0f,width);PipeBarrier<PIPE_V>();}
Muls(mat[j*width],v,-1.0f,span);PipeBarrier<PIPE_V>();
}
Fence(); // Augmented Frobenius norm is a conservative inverse-norm bound.
Mul(mem[16384],mat[np],mat[np],np,n,{1,1,1,(uint8_t)(np/8),(uint8_t)(width/8),(uint8_t)(width/8)});PipeBarrier<PIPE_V>();Sum(red,mem[16384],n*np);LocalSync<HardEvent::V_S>();float ix=red.GetValue(0);
float threshold=t.atol*t.atol>anorm*t.rtol*t.rtol?t.atol*t.atol:anorm*t.rtol*t.rtol;
if(!(threshold*ix<25.0f))return false;
if constexpr(sizeof(T)==4){LocalSync<HardEvent::V_MTE3>();DataCopyPad(output[base],mat[np],{(uint16_t)n,n*4,(width-np)/8,0,0});LocalSync<HardEvent::MTE3_MTE2>();}
else for(uint32_t i=0;i<n;i++)WriteOutput(mat[i*width+np],base+i*n,n);
return true;
}
public:
__aicore__ inline void Init(GM_ADDR ai,GM_ADDR ao,LinalgPinvTilingData tt){t=tt;input.SetGlobalBuffer((__gm__ T*)ai);output.SetGlobalBuffer((__gm__ T*)ao);InitSocState();mem=LocalTensor<float>(TPosition::VECCALC,0,48640);offsets=mem.template ReinterpretCast<uint32_t>()[45568];raw=mem.template ReinterpretCast<T>()[32768*4/sizeof(T)];}
__aicore__ inline bool Run(){for(uint32_t batch=GetBlockIdx();batch<t.batch;batch+=GetBlockNum()){base=batch*N*N;if(!FastInverse())return false;}return true;}
};
template<class T,uint32_t N,uint32_t G> class ColumnTinyPinv {
GlobalTensor<T> input,output;GlobalTensor<float> state;LocalTensor<float> mem;LocalTensor<uint32_t> offsets,permutation;
LinalgPinvTilingData t;uint32_t part,base;float anorm;
static constexpr uint32_t n=N,np=(N+7)/8*8,cols=(N+G-1)/G,span=n+cols,ld=(N>16 && G>1)?136:np+(N==50?16:8);
__aicore__ inline void Fence(){PipeBarrier<PIPE_ALL>();}
__aicore__ inline void Sum(LocalTensor<float> dst,LocalTensor<float> src,uint32_t len){
auto first=mem[30000],second=mem[30256];uint32_t full=len/64,tail=len%64;
if(full)WholeReduceSum(first,src,64,full,1,1,8);if(tail)WholeReduceSum(first[full],src[full*64],tail,1,1,1,8);PipeBarrier<PIPE_V>();
uint32_t count=full+(tail?1:0);if(count<=64)WholeReduceSum(dst,first,count,1,1,1,8);
else{full=count/64;tail=count%64;WholeReduceSum(second,first,64,full,1,1,8);if(tail)WholeReduceSum(second[full],first[full*64],tail,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(dst,second,full+(tail?1:0),1,1,1,8);}PipeBarrier<PIPE_V>();
}
__aicore__ inline float Compute(){
auto mat=mem,piv=mem[38928],h=mem[39448],values=mem[41024],tmp=mem[41496],red=mem[42528];
if constexpr(N==15){auto load=mem[16384];Duplicate(load,0.0f,232);LocalSync<HardEvent::V_MTE2>();DataCopyPad(load,input[base],{1,n*n*4,0,0,0},{false,0,0,0});}
else DataCopyPad(mat,input[base],{(uint16_t)n,n*4,0,(ld-np)/8,0},{true,0,(uint8_t)(np-n),0});
Duplicate(mat[n*ld],0.0f,cols*ld);
for(uint32_t c=0;c<8;c++)offsets.SetValue(c,c*ld*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<span;step*=2){Adds(offsets[step].template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),int32_t(step*ld*4),step);PipeBarrier<PIPE_V>();}
for(uint32_t r=n;r<np;r++)permutation.SetValue(r,n*ld*4);
auto pattern=mem.template ReinterpretCast<uint32_t>()[45760];uint64_t bits=(~0ULL)>>(64-N);
pattern.SetValue(0,uint32_t(bits));pattern.SetValue(1,uint32_t(bits>>32));
if constexpr(N==15){
auto load=mem[16384],packed=mem[24096];auto loadIndices=mem[42048].template ReinterpretCast<uint32_t>();
for(uint32_t c=0;c<16;c++)loadIndices.SetValue(c,c*4);LocalSync<HardEvent::S_V>();
for(uint32_t step=16;step<n*np;step*=2){Adds(loadIndices[step].template ReinterpretCast<int32_t>(),loadIndices.template ReinterpretCast<int32_t>(),int32_t(step/16*n*4),step);PipeBarrier<PIPE_V>();}
LocalSync<HardEvent::MTE2_V>();Gather(packed,load,loadIndices,0,n*np);PipeBarrier<PIPE_V>();
Adds(mat,packed,0.0f,np,n,{1,1,(uint8_t)(ld/8),(uint8_t)(np/8)});PipeBarrier<PIPE_V>();
uint64_t padding[2]={128,0};Duplicate(mat[8],0.0f,padding,n,1,ld/8);PipeBarrier<PIPE_V>();
}else LocalSync<HardEvent::MTE2_V>();
Mul(mem[16384],mat,mat,np,n,{1,1,1,(uint8_t)(np/8),(uint8_t)(ld/8),(uint8_t)(ld/8)});PipeBarrier<PIPE_V>();
Sum(red,mem[16384],n*np);LocalSync<HardEvent::V_S>();anorm=red.GetValue(0);if(!(anorm>0))return -1;
float factor=1.0f;
if constexpr(N<=16){if(anorm<0.000000059604644775390625f || anorm>16777216.0f)return -1;}
else{
union ScaleBits{uint32_t bits;float value;};ScaleBits scale;
int exponent=int((red.template ReinterpretCast<uint32_t>().GetValue(0)>>23)&255)-127;
int halfExponent=exponent>=0?(exponent+1)/2:exponent/2;scale.bits=uint32_t(127-halfExponent)<<23;factor=scale.value;
Muls(mat,mat,factor,np,n,{1,1,(uint8_t)(ld/8),(uint8_t)(ld/8)});
}
for(uint32_t c=0;c<cols;c++)if(part*cols+c<n)mat.SetValue((n+c)*ld+part*cols+c,factor);
LocalSync<HardEvent::S_V>();PipeBarrier<PIPE_V>();
uint64_t active=bits;float minPivot2=anorm*factor*factor*1e-10f;
for(uint32_t j=0;j<n;j++){
auto column=mat[j*ld];Abs(tmp,column,n);PipeBarrier<PIPE_V>();
uint64_t mask[2]={active,0};WholeReduceMax(red,tmp,mask,1,1,1,8);LocalSync<HardEvent::V_S>();
float mx,indexBits;GetReduceMaxMinCount(mx,indexBits);union IndexBits{float value;uint32_t bits;};IndexBits chosen;chosen.value=indexBits;uint32_t pi=chosen.bits;
if(!(mx*mx>minPivot2))return -1;
float pval=column.GetValue(pi);
uint32_t remaining=span-j-1;
Gather(values,mat[(j+1)*ld],offsets,pi*4,remaining);
uint64_t pivotMask[2]={1ULL<<(pi%8),0};
Adds(column[pi/8*8],column[pi/8*8],-1.0f,pivotMask,1,{1,1,8,8});
float inversePivot=1.0f/pval;active&=~(1ULL<<pi);permutation.SetValue(j,(n*ld+pi)*4);
PipeBarrier<PIPE_V>();
Muls(piv,column,-inversePivot,np);
if constexpr(N>16 && G>1)h=mem[39424+((j+9)&15)*8];
Brcb(h,values,(remaining+7)/8,{1,8});PipeBarrier<PIPE_V>();
MulAddDst(mat[(j+1)*ld],piv,h,np,remaining,{1,1,0,(uint8_t)(ld/8),0,1});PipeBarrier<PIPE_V>();
}
Mul(mem[16384],mat[n*ld],mat[n*ld],np,cols,{1,1,1,(uint8_t)(np/8),(uint8_t)(ld/8),(uint8_t)(ld/8)});PipeBarrier<PIPE_V>();
Sum(red,mem[16384],np*cols);LocalSync<HardEvent::V_S>();return red.GetValue(0);
}
__aicore__ inline void PrepareOutput(){
uint32_t first=part*cols,ac=first<n?(n-first<cols?n-first:cols):0;if(!ac)return;
auto h=mem[39448],packed=mem[(N>16 && G>1)?12000:10000],compact=mem[24096];auto indices=mem.template ReinterpretCast<int32_t>()[20000];
LocalSync<HardEvent::S_V>();Brcb(h,offsets.template ReinterpretCast<float>(),(ac+7)/8,{1,8});PipeBarrier<PIPE_V>();
Add(indices,permutation.template ReinterpretCast<int32_t>(),h.template ReinterpretCast<int32_t>(),np,ac,{1,1,0,(uint8_t)(np/8),0,1});PipeBarrier<PIPE_V>();
Gather(packed,mem,indices.template ReinterpretCast<uint32_t>(),0,ac*np);
if constexpr(N%8!=0){PipeBarrier<PIPE_V>();uint64_t count=0;GatherMask(compact,packed,mem.template ReinterpretCast<uint32_t>()[45760],false,0,{1,(uint16_t)ac,(uint16_t)(np/8),0},count);}
}
__aicore__ inline void WriteOutput(){
uint32_t first=part*cols,ac=first<n?(n-first<cols?n-first:cols):0;if(!ac)return;
auto src=N%8==0?mem[(N>16 && G>1)?12000:10000]:mem[24096];
LocalSync<HardEvent::V_MTE3>();DataCopyPad(output[base+first*n],src,{1,ac*n*4,0,0,0});
LocalSync<HardEvent::MTE3_MTE2>();
}
public:
__aicore__ inline void Init(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData tt){
t=tt;part=GetBlockIdx()%G;input.SetGlobalBuffer((__gm__ T*)ai);output.SetGlobalBuffer((__gm__ T*)ao);if constexpr(G>1)state.SetGlobalBuffer((__gm__ float*)ws);
InitSocState();mem=LocalTensor<float>(TPosition::VECCALC,0,48640);offsets=mem.template ReinterpretCast<uint32_t>()[45568];permutation=mem.template ReinterpretCast<uint32_t>()[45824];
}
__aicore__ inline void Init(GM_ADDR ai,GM_ADDR ao,LinalgPinvTilingData tt){Init(ai,ao,nullptr,tt);}
__aicore__ inline bool Run(){
if constexpr(G==1){
for(uint32_t batch=GetBlockIdx();batch<t.batch;batch+=GetBlockNum()){
base=batch*n*n;float norm=Compute();if(!(norm>=0))return false;
PrepareOutput();
float threshold=t.atol*t.atol>anorm*t.rtol*t.rtol?t.atol*t.atol:anorm*t.rtol*t.rtol;
if(!(threshold*norm<25.0f)){Fence();return false;}WriteOutput();
}return true;
}else{
base=GetBlockIdx()/G*n*n;float norm=Compute();auto red=mem[42528],status=mem[42600];red.SetValue(0,norm);LocalSync<HardEvent::S_MTE3>();
DataCopy(state[GetBlockIdx()*8],red,8);LocalSync<HardEvent::MTE3_S>();SyncAll();
DataCopy(status,state[GetBlockIdx()/G*G*8],G*8);
PrepareOutput();
LocalSync<HardEvent::MTE2_S>();float total=0;
for(uint32_t k=0;k<G;k++){float x=status.GetValue(k*8);if(!(x>=0)){Fence();return false;}total+=x;}
float threshold=t.atol*t.atol>anorm*t.rtol*t.rtol?t.atol*t.atol:anorm*t.rtol*t.rtol;
if(!(threshold*total<25.0f)){Fence();return false;}WriteOutput();return true;
}
}
};
template<class T,uint32_t N,uint32_t G>
__aicore__ __attribute__((noinline,aligned(128))) bool AlignedTinyInverse(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData t){
ColumnTinyPinv<T,N,G> fast;fast.Init(ai,ao,ws,t);return fast.Run();
}
template<class T,uint32_t N,uint32_t G>
__aicore__ __attribute__((noinline)) bool SmallInverse(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData t){
ColumnTinyPinv<T,N,G> fast;fast.Init(ai,ao,ws,t);return fast.Run();
}
template<class T,uint32_t N=0> class ParallelInverse {
TPipe pipe;TBuf<QuePosition::VECCALC> buf;LinalgPinvTilingData t;
GlobalTensor<T> input,output;GlobalTensor<float> exchange,iall,state;
LocalTensor<float> mat,rm,piv,tmp,panel,red,broad,invs;LocalTensor<uint32_t> idx,perm,ofs;LocalTensor<T> raw;
uint32_t part,groupid,first,ld,cols,panelCols;bool outputReady;bool cubePrecision=true;FP32Cube* cube=nullptr;
__aicore__ inline void Fence(){PipeBarrier<PIPE_ALL>();}
__aicore__ inline void Sync(){Fence();if(t.group*t.groups>1)SyncAll();Fence();}
__aicore__ inline void Store(GlobalTensor<float> dst,float a,float b,float c){red.SetValue(0,a);red.SetValue(1,b);red.SetValue(2,c);LocalSync<HardEvent::S_MTE3>();DataCopy(dst,red,8);LocalSync<HardEvent::MTE3_S>();}
__aicore__ inline void Load(GlobalTensor<float> src){DataCopy(red,src,8);LocalSync<HardEvent::MTE2_S>();}
__aicore__ inline void Sum(LocalTensor<float> dst,LocalTensor<float> src){
uint32_t full=t.pp/64,tail=t.pp%64;auto sums=mat[41536];
if(full)WholeReduceSum(sums,src,64,full,1,1,8);if(tail)WholeReduceSum(sums[full],src[full*64],tail,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(dst,sums,full+(tail?1:0),1,1,1,8);PipeBarrier<PIPE_V>();
}
__aicore__ inline float Dot(LocalTensor<float> x){Mul(tmp,x,x,t.pp);PipeBarrier<PIPE_V>();Sum(red,tmp);LocalSync<HardEvent::V_S>();return red.GetValue(0);}
__aicore__ inline uint32_t Pivot(){
if constexpr(N==256){WholeReduceMax<float,false>(mat[40000],tmp,64,4,1,1,8);LocalSync<HardEvent::V_S>();float maximum,indexBits;GetReduceMaxMinCount(maximum,indexBits);union IndexBits{float value;uint32_t bits;};IndexBits selected;selected.value=indexBits;return selected.bits;}
SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,t.p);WholeReduceMax<float,false>(mat[40000],tmp,64,1,1,1,8);LocalSync<HardEvent::V_S>();float maximum,indexBits;GetReduceMaxMinCount(maximum,indexBits);SetMaskNorm();ResetMask();union IndexBits{float value;uint32_t bits;};IndexBits selected;selected.value=indexBits;return selected.bits;
}
__aicore__ inline void Update(uint32_t c0,uint32_t c1,uint32_t row){
if(c0<c1){Gather(tmp,mat[c0*ld],ofs,row*4,c1-c0);PipeBarrier<PIPE_V>();
if constexpr(N==256){auto opposite=mat[38912];Brcb(broad,tmp,(c1-c0+7)/8,{1,8});Brcb(opposite,tmp,(c1-c0+7)/8,{1,8});PipeBarrier<PIPE_V>();MulAddDst(mat[c0*ld],piv,broad,64,c1-c0,{1,1,0,33,0,1});MulAddDst(mat[c0*ld+64],piv[64],opposite,64,c1-c0,{1,1,0,33,0,1});MulAddDst(mat[c0*ld+128],piv[128],broad,64,c1-c0,{1,1,0,33,0,1});MulAddDst(mat[c0*ld+192],piv[192],opposite,64,c1-c0,{1,1,0,33,0,1});}else{Brcb(broad,tmp,(c1-c0+7)/8,{1,8});PipeBarrier<PIPE_V>();for(uint32_t r=0;r<t.pp;r+=64){uint32_t count=t.pp-r<64?t.pp-r:64;MulAddDst(mat[c0*ld+r],piv[r],broad,count,c1-c0,{1,1,0,(uint8_t)(ld/8),0,1});}}
}PipeBarrier<PIPE_V>();
}
__aicore__ inline float NormRoot(float z){red.SetValue(0,z>0?z:0);LocalSync<HardEvent::S_V>();Sqrt(red,red,1);LocalSync<HardEvent::V_S>();return red.GetValue(0);}
__aicore__ inline void PowerRow(uint32_t batch,uint32_t row,bool transpose,bool inverse,LocalTensor<float> dst){
Duplicate(dst,0.0f,t.pp);Fence();
if(inverse){
if(transpose){DataCopy(dst,iall[batch*t.pp*t.pp+row*t.pp],t.pp);Fence();}
else{DataCopyPad(rm,iall[batch*t.pp*t.pp+row],{(uint16_t)t.p,4,(t.pp-1)*4,0,0},{false,0,0,0});Fence();Gather(dst,rm,idx,0,t.p);Fence();}
}else{
uint32_t base=batch*t.p*t.p;
if constexpr(sizeof(T)==4){
if(!transpose)DataCopyPad(dst,input[base+row*t.p],{1,t.p*4,0,0,0},{true,0,(uint8_t)(t.pp-t.p),0});
else{DataCopyPad(rm,input[base+row],{(uint16_t)t.p,4,(t.p-1)*4,0,0},{false,0,0,0});Fence();Gather(dst,rm,idx,0,t.p);}Fence();
}else{
auto packed=rm.template ReinterpretCast<T>();auto cv=mat[26000];
if(!transpose){DataCopyPad(packed,input[base+row*t.p],{1,t.p*2,0,0,0},{false,0,0,0});Fence();Cast(dst,packed,RoundMode::CAST_NONE,t.p);}
else{DataCopyPad(packed,input[base+row],{(uint16_t)t.p,2,(t.p-1)*2,0,0},{false,0,0,0});Fence();Cast(cv,packed,RoundMode::CAST_NONE,t.p*16);Fence();Gather(dst,cv,perm,0,t.p);}Fence();
}
}
}
__aicore__ __attribute__((noinline)) float PowerNorm(uint32_t batch,bool inverse){
if constexpr(N>0){t.p=N;t.pp=(N+7)/8*8;}
auto x=mat[38000],y=mat[38512],vals=mat[39024];uint32_t rows=(t.p+t.group*8-1)/(t.group*8)*8,firstRow=part*rows,ac=firstRow<t.p?(t.p-firstRow<rows?t.p-firstRow:rows):0;
auto cache=mat,cacheT=mat[rows*t.pp];
Duplicate(x,0.0f,t.pp);Fence();for(uint32_t j=0;j<t.p;j++){x.SetValue(j,(float)((int32_t)((j*17+13)%31)-15));idx.SetValue(j,j*32);perm.SetValue(j,j*64);}LocalSync<HardEvent::S_V>();
for(uint32_t r=0;r<ac;r++){PowerRow(batch,firstRow+r,false,inverse,cache[r*t.pp]);PowerRow(batch,firstRow+r,true,inverse,cacheT[r*t.pp]);}
float nx=NormRoot(Dot(x));Muls(x,x,nx>0?1.0f/nx:0.0f,t.pp);Fence();float estimate=0;
for(uint32_t it=0;it<4;it++){
for(uint32_t r=0;r<ac;r++){Mul(tmp,cache[r*t.pp],x,t.pp);PipeBarrier<PIPE_V>();Sum(red,tmp);LocalSync<HardEvent::V_S>();vals.SetValue(r,red.GetValue(0));}
if(ac){LocalSync<HardEvent::S_MTE3>();DataCopyPad(exchange[firstRow],vals,{1,ac*4,0,0,0});LocalSync<HardEvent::MTE3_S>();}
Sync();DataCopy(y,exchange,t.pp);Fence();if(t.pp>t.p)Duplicate(y[t.p],0.0f,t.pp-t.p);Fence();estimate=Dot(y);float ny=NormRoot(estimate);Muls(y,y,ny>0?1.0f/ny:0.0f,t.pp);Fence();
for(uint32_t r=0;r<ac;r++){Mul(tmp,cacheT[r*t.pp],y,t.pp);PipeBarrier<PIPE_V>();Sum(red,tmp);LocalSync<HardEvent::V_S>();vals.SetValue(r,red.GetValue(0));}
if(ac){LocalSync<HardEvent::S_MTE3>();DataCopyPad(exchange[t.pp+firstRow],vals,{1,ac*4,0,0,0});LocalSync<HardEvent::MTE3_S>();}
Sync();DataCopy(x,exchange[t.pp],t.pp);Fence();if(t.pp>t.p)Duplicate(x[t.p],0.0f,t.pp-t.p);Fence();nx=NormRoot(Dot(x));Muls(x,x,nx>0?1.0f/nx:0.0f,t.pp);Fence();
}return estimate;
}
__aicore__ inline void CubeRefine(uint32_t batch){
constexpr uint32_t P=256;
auto residual=state[(t.group*t.groups+1)*8+batch*P*P];auto savedA=state[(t.group*t.groups+1)*8+t.batch*P*P+batch*P*P];auto corr=state[(t.group*t.groups+1)*8+2*t.batch*P*P+batch*P*P];auto inv=iall[batch*P*P];
uint32_t aic=part>>1,row0=P,rows=0;
if((part&1)==0){
if(t.group>=16){if(aic<8){row0=aic*32;rows=32;}}
else if(t.group>=12){if(aic<4){row0=aic*48;rows=48;}else if(aic<6){row0=192+(aic-4)*32;rows=32;}}
else if(t.group>=10){if(aic<3){row0=aic*64;rows=64;}else if(aic<5){row0=192+(aic-3)*32;rows=32;}}
else{if(aic<4){row0=aic*64;rows=64;}}
}
if(rows>0){
cube->SetTensorA(inv[row0*P]);cube->SetTensorB(savedA);cube->SetTail(rows,P,P);cube->IterateAll(residual[row0*P]);cube->End();
DataCopy(mat,residual[row0*P],rows*P);LocalSync<HardEvent::MTE2_V>();Muls(mat,mat,-1.0f,rows*P);PipeBarrier<PIPE_V>();
for(uint32_t r=0;r<rows;r++){uint32_t c=row0+r;uint64_t mask[2]={1ULL<<(c%8),0};Adds(mat[r*P+c/8*8],mat[r*P+c/8*8],1.0f,mask,1,{1,1,8,8});}
LocalSync<HardEvent::V_MTE3>();DataCopy(residual[row0*P],mat,rows*P);Fence();
DataCopy(mat[16384],inv[row0*P],rows*P);
cube->SetTensorA(residual[row0*P]);cube->SetTensorB(inv);cube->SetTail(rows,P,P);cube->IterateAll(corr[row0*P]);cube->End();
DataCopy(mat,corr[row0*P],rows*P);LocalSync<HardEvent::MTE2_V>();Add(mat,mat,mat[16384],rows*P);PipeBarrier<PIPE_V>();
for(uint32_t r=0;r<rows;r+=16){
FastTranspose16x256(rm,mat[r*P]);LocalSync<HardEvent::V_MTE3>();DataCopyPad(output[batch*P*P+row0+r],rm,{P,64,0,(P-16)*4,0});
if(r+16<rows)LocalSync<HardEvent::MTE3_V>();else Fence();
}
}
if(batch+t.groups<t.batch)Sync();
outputReady=true;
}
__aicore__ inline void CubeCorrection(uint32_t batch){
constexpr uint32_t P=256;auto residual=state[(t.group*t.groups+1)*8+batch*P*P],corrected=state[(t.group*t.groups+1)*8+t.batch*P*P+batch*P*P],inv=iall[batch*P*P];uint32_t row0=part*32;
if(row0<P){cube->SetTensorA(residual[row0*P]);cube->SetTensorB(inv);cube->SetTail(32,P,P);cube->IterateAll(corrected[row0*P]);cube->End();}Sync();
for(uint32_t c=part*16;c<P;c+=t.group*16){DataCopy(mat,corrected[c*P],16*P);DataCopy(mat[8192],inv[c*P],16*P);LocalSync<HardEvent::MTE2_V>();Add(mat,mat,mat[8192],16*P);PipeBarrier<PIPE_V>();FastTranspose16x256(rm,mat);LocalSync<HardEvent::V_MTE3>();DataCopyPad(output[batch*P*P+c],rm,{P,64,0,(P-16)*4,0});Fence();}Sync();outputReady=true;
}
__aicore__ inline void RefineInverse(uint32_t batch){
if constexpr(N==256 && sizeof(T)==4){if(cube && cubePrecision){CubeRefine(batch);return;}}
auto residual=state[(t.group*t.groups+1)*8+batch*t.pp*t.pp];auto corrected=state[(t.group*t.groups+1)*8+t.batch*t.pp*t.pp+batch*t.pp*t.pp];auto inv=iall[batch*t.pp*t.pp];
if constexpr(!(sizeof(T)==4 && N>0)){for(uint32_t j=0;j<t.p;j++){idx.SetValue(j,j*32);perm.SetValue(j,j*64);}LocalSync<HardEvent::S_V>();}
uint32_t wp=64;while(wp<t.pp)wp*=2;
auto row=mat[38000],out=mat[38512];uint32_t chunk=16384/wp;if(chunk>64)chunk=64;
if constexpr(sizeof(T)==4 && N>0){
auto savedA=corrected;auto acc=mat,acc1=mat[2048],acc2=mat[4096],acc3=mat[6144],coeff=mat[8192],cache=mat[12288],br=mat[30000];
auto outputIndices=mat[40000].template ReinterpretCast<uint32_t>();auto columnIndices=br.template ReinterpretCast<uint32_t>();uint32_t count=t.pp*8;
for(uint32_t c=0;c<8;c++)outputIndices.SetValue(c,c*t.pp*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<count;step*=2){Adds(outputIndices[step].template ReinterpretCast<int32_t>(),outputIndices.template ReinterpretCast<int32_t>(),int32_t(step/8*4),step);PipeBarrier<PIPE_V>();}
for(uint32_t c0=part*8;c0<t.p;c0+=t.group*8){
uint32_t ac=t.p-c0<8?t.p-c0:8;
DataCopy(acc,inv[c0*t.pp],ac*t.pp);LocalSync<HardEvent::MTE2_V>();
Gather(coeff,acc,outputIndices,0,t.pp*8);PipeBarrier<PIPE_V>();Duplicate(acc,0.0f,8192);PipeBarrier<PIPE_V>();
for(uint32_t cc=0;cc<t.p;cc+=chunk){uint32_t kc=t.p-cc<chunk?t.p-cc:chunk;
DataCopy(cache,savedA[cc*t.pp],kc*t.pp);Brcb(br,coeff[cc*8],kc,{1,8});LocalSync<HardEvent::MTE2_V>();PipeBarrier<PIPE_V>();
for(uint32_t j=0;j<kc;j+=4){
for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc[r],cache[j*t.pp+r],br[j*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(j+1<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc1[r],cache[(j+1)*t.pp+r],br[(j+1)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(j+2<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc2[r],cache[(j+2)*t.pp+r],br[(j+2)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(j+3<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc3[r],cache[(j+3)*t.pp+r],br[(j+3)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
PipeBarrier<PIPE_V>();
}LocalSync<HardEvent::V_MTE2>();
}
Add(acc,acc,acc1,ac*t.pp);Add(acc2,acc2,acc3,ac*t.pp);PipeBarrier<PIPE_V>();Add(acc,acc,acc2,ac*t.pp);PipeBarrier<PIPE_V>();Muls(acc,acc,-1.0f,ac*t.pp);LocalSync<HardEvent::V_S>();
for(uint32_t c=0;c<ac;c++)acc.SetValue(c*t.pp+c0+c,acc.GetValue(c*t.pp+c0+c)+1.0f);
if constexpr(N==256){if(cube){LocalSync<HardEvent::S_MTE3>();DataCopy(residual[c0*t.pp],acc,t.pp*8);Fence();}else{LocalSync<HardEvent::S_V>();Gather(rm,acc,outputIndices,0,t.pp*8);LocalSync<HardEvent::V_MTE3>();DataCopy(residual[c0*t.pp],rm,t.pp*8);Fence();}}
else{LocalSync<HardEvent::S_V>();Gather(rm,acc,outputIndices,0,t.pp*8);LocalSync<HardEvent::V_MTE3>();DataCopy(residual[c0*t.pp],rm,t.pp*8);Fence();}
}Sync();
if constexpr(N==256){if(cube){CubeCorrection(batch);return;}}
}else{
if constexpr(N==199){
auto row=mat[38000],out=mat[38512],products=mat[24576],sums=mat[32768],values=mat[33152];uint32_t chunk=16384/wp;if(chunk>64)chunk=64;
uint32_t segments=wp/64,batchDots=wp<=256?32:16;
Duplicate(sums,0.0f,batchDots*8);Duplicate(out,0.0f,chunk);
auto aCache=mat[16384];uint32_t owned=(t.p+t.group-1-part)/t.group;bool cacheA=owned*wp<=8192;
if(cacheA){
if constexpr(sizeof(T)==4){
if(wp>t.pp){Duplicate(aCache,0.0f,owned*wp);LocalSync<HardEvent::V_MTE2>();}
if(owned)DataCopyPad(aCache,input[batch*t.p*t.p+part*t.p],{(uint16_t)owned,t.p*4,(t.group-1)*t.p*4,(wp-t.pp)/8,0},{true,0,(uint8_t)(t.pp-t.p),0});
Fence();
}else{Duplicate(aCache,0.0f,owned*wp);Fence();for(uint32_t k=0;k<owned;k++)PowerRow(batch,part+k*t.group,false,false,aCache[k*wp]);}
}
for(uint32_t cc=0;cc<t.p;cc+=chunk){
uint32_t kc=t.p-cc<chunk?t.p-cc:chunk;
if(kc<chunk){Duplicate(sums,0.0f,batchDots*8);Duplicate(out,0.0f,chunk);}
if(wp==t.pp){LocalSync<HardEvent::V_MTE2>();DataCopy(mat,inv[cc*t.pp],kc*t.pp);Fence();}
else{Duplicate(mat,0.0f,kc*wp);LocalSync<HardEvent::V_MTE2>();DataCopyPad(mat,inv[cc*t.pp],{(uint16_t)kc,t.pp*4,0,(wp-t.pp)/8,0},{false,0,0,0});Fence();}
for(uint32_t r=part;r<t.p;r+=t.group){
if(cacheA)row=aCache[(r-part)/t.group*wp];else {Duplicate(row,0.0f,wp);Fence();PowerRow(batch,r,false,false,row);}
for(uint32_t c=0;c<kc;c+=batchDots){uint32_t ac=kc-c<batchDots?kc-c:batchDots;
for(uint32_t q=0;q<wp;q+=64)Mul(products[q],mat[c*wp+q],row[q],64,ac,{1,1,1,(uint8_t)(wp/8),(uint8_t)(wp/8),0});
PipeBarrier<PIPE_V>();WholeReduceSum(sums,products,64,ac*segments,8/segments,1,8);PipeBarrier<PIPE_V>();BlockReduceSum(values,sums,batchDots/8,64,1,1,8);PipeBarrier<PIPE_V>();Muls(out[c],values,-1.0f,ac);PipeBarrier<PIPE_V>();
}
if(r>=cc && r<cc+kc){LocalSync<HardEvent::V_S>();out.SetValue(r-cc,out.GetValue(r-cc)+1.0f);LocalSync<HardEvent::S_MTE3>();}else LocalSync<HardEvent::V_MTE3>();
DataCopyPad(residual[cc*t.pp+r*8],out,{(uint16_t)((kc+7)/8),32,0,(t.pp-1)*32,0});LocalSync<HardEvent::MTE3_V>();
}
}Sync();
}else{
auto row=mat[41472],out=mat[44096],products=mat[24576],sums=mat[40960];uint32_t chunk=16384/wp;if(chunk>64)chunk=64;
uint32_t segments=wp/64,batchDots=wp<=256?32:16;
Duplicate(sums,0.0f,512);
auto aCache=mat[16384];uint32_t owned=(t.p+t.group-1-part)/t.group;bool cacheA=owned*wp<=8192;
if(cacheA){
if constexpr(sizeof(T)==4){
if(wp>t.pp){Duplicate(aCache,0.0f,owned*wp);LocalSync<HardEvent::V_MTE2>();}
if(owned)DataCopyPad(aCache,input[batch*t.p*t.p+part*t.p],{(uint16_t)owned,t.p*4,(t.group-1)*t.p*4,(wp-t.pp)/8,0},{true,0,(uint8_t)(t.pp-t.p),0});
Fence();
}else{Duplicate(aCache,0.0f,owned*wp);Fence();for(uint32_t k=0;k<owned;k++)PowerRow(batch,part+k*t.group,false,false,aCache[k*wp]);}
}
for(uint32_t cc=0;cc<t.p;cc+=chunk){
uint32_t kc=t.p-cc<chunk?t.p-cc:chunk;
if(kc<chunk)Duplicate(sums,0.0f,512);
if(wp==t.pp){LocalSync<HardEvent::V_MTE2>();DataCopy(mat,inv[cc*t.pp],kc*t.pp);Fence();}
else{Duplicate(mat,0.0f,kc*wp);LocalSync<HardEvent::V_MTE2>();DataCopyPad(mat,inv[cc*t.pp],{(uint16_t)kc,t.pp*4,0,(wp-t.pp)/8,0},{false,0,0,0});Fence();}
for(uint32_t r=part;r<t.p;r+=t.group){
if(cacheA)row=aCache[(r-part)/t.group*wp];else {Duplicate(row,0.0f,wp);Fence();PowerRow(batch,r,false,false,row);}
for(uint32_t c=0;c<kc;c+=batchDots){uint32_t ac=kc-c<batchDots?kc-c:batchDots;
for(uint32_t q=0;q<wp;q+=64)Mul(products[c*wp+q],mat[c*wp+q],row[q],64,ac,{1,1,1,(uint8_t)(wp/8),(uint8_t)(wp/8),0});
}PipeBarrier<PIPE_V>();
for(uint32_t c=0;c<kc;c+=batchDots){uint32_t ac=kc-c<batchDots?kc-c:batchDots;
WholeReduceSum(sums[c*8],products[c*wp],64,ac*segments,8/segments,1,8);
}PipeBarrier<PIPE_V>();
uint32_t blocks=(kc+7)/8;
BlockReduceSum(out,sums,blocks,64,1,1,8);PipeBarrier<PIPE_V>();
Muls(out,out,-1.0f,blocks*8);
if(r>=cc && r<cc+kc){LocalSync<HardEvent::V_S>();out.SetValue(r-cc,out.GetValue(r-cc)+1.0f);LocalSync<HardEvent::S_MTE3>();}else LocalSync<HardEvent::V_MTE3>();
DataCopyPad(residual[cc*t.pp+r*8],out,{(uint16_t)blocks,32,0,(t.pp-1)*32,0});LocalSync<HardEvent::MTE3_V>();
}
}Sync();
}
}
auto acc=mat,cf=mat[8192],cache=mat[12288],br=mat[30000],vals=mat[34560];auto ix=mat.template ReinterpretCast<uint32_t>()[35584];
auto outputIndices=mat.template ReinterpretCast<uint32_t>()[40000];
if constexpr(sizeof(T)==4 && N==0){
auto columnIndices=br.template ReinterpretCast<uint32_t>();uint32_t count=t.pp*8;
for(uint32_t c=0;c<8;c++)outputIndices.SetValue(c,c*t.pp*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<count;step*=2){Adds(outputIndices[step].template ReinterpretCast<int32_t>(),outputIndices.template ReinterpretCast<int32_t>(),int32_t(step/8*4),step);PipeBarrier<PIPE_V>();}
}
for(uint32_t c0=part*8;c0<t.p;c0+=t.group*8){
uint32_t ac=t.p-c0<8?t.p-c0:8;DataCopy(acc,inv[c0*t.pp],ac*t.pp);
DataCopy(cf,residual[c0*t.pp],8*t.pp);
auto acc1=mat[2048],acc2=mat[4096],acc3=mat[6144];
if(t.pp<=256)Duplicate(acc1,0.0f,6144);
else Duplicate(acc2,0.0f,ac*t.pp);
Fence();if(t.pp>t.p){Duplicate(cf[t.p*8],0.0f,(t.pp-t.p)*8);PipeBarrier<PIPE_V>();}
for(uint32_t cc=0;cc<t.p;cc+=chunk){uint32_t kc=t.p-cc<chunk?t.p-cc:chunk;
DataCopy(cache,inv[cc*t.pp],kc*t.pp);
Brcb(br,cf[cc*8],kc,{1,8});LocalSync<HardEvent::MTE2_V>();PipeBarrier<PIPE_V>();
if(t.pp<=256){
for(uint32_t k=0;k<kc;k+=4){
for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc[r],cache[k*t.pp+r],br[k*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(k+1<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc1[r],cache[(k+1)*t.pp+r],br[(k+1)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(k+2<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc2[r],cache[(k+2)*t.pp+r],br[(k+2)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(k+3<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc3[r],cache[(k+3)*t.pp+r],br[(k+3)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
PipeBarrier<PIPE_V>();
}
}else{
for(uint32_t k=0;k<kc;k+=2){
for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc[r],cache[k*t.pp+r],br[k*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(k+1<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc2[r],cache[(k+1)*t.pp+r],br[(k+1)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
PipeBarrier<PIPE_V>();
}
}
LocalSync<HardEvent::V_MTE2>();
}
if(t.pp<=256){Add(acc,acc,acc1,ac*t.pp);Add(acc2,acc2,acc3,ac*t.pp);PipeBarrier<PIPE_V>();}
Add(acc,acc,acc2,ac*t.pp);PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==4){
Gather(rm,acc,outputIndices,0,t.pp*8);LocalSync<HardEvent::V_MTE3>();
DataCopyPad(output[batch*t.p*t.p+c0],rm,{(uint16_t)t.p,ac*4,0,(t.p-ac)*4,0});Fence();
}
else{DataCopy(corrected[c0*t.pp],acc,ac*t.pp);Fence();}
}Sync();if constexpr(sizeof(T)==4){outputReady=true;return;}
for(uint32_t c=part;c<t.p;c+=t.group){DataCopy(out,corrected[c*t.pp],t.pp);Fence();DataCopy(inv[c*t.pp],out,t.pp);Fence();}Sync();
}
public:
__aicore__ inline TPipe* Pipe(){return &pipe;}
__aicore__ inline void SetCube(FP32Cube* mm){cube=mm;}
__aicore__ inline void RefineOnly(){for(uint32_t batch=groupid;batch<t.batch;batch+=t.groups)CubeRefine(batch);}
__aicore__ inline void Init(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData tt){
outputReady=false;t=tt;if constexpr(N>0){t.m=N;t.n=N;t.p=N;t.q=N;t.pp=(N+7)/8*8;t.qp=t.pp;}ld=t.pp+8;part=GetBlockIdx()%t.group;groupid=GetBlockIdx()/t.group;cols=t.cols;panelCols=t.cols;first=part*cols;
if constexpr(N>0){if(t.batch>1 && t.group>=3){
uint32_t leftCores=t.group*2/3,rightCores=t.group-leftCores;
uint32_t leftCols=(t.pp+leftCores*8-1)/(leftCores*8)*8,rightCols=(t.pp+rightCores*8-1)/(rightCores*8)*8;
if(leftCols*(t.pp+8)<=19000 && rightCols*(t.pp+8)<=19000 && leftCols<=t.cols){
panelCols=leftCols;
if(part<leftCores){cols=leftCols;first=part*cols;if(first>=t.pp)first=2*t.pp;}
else{cols=rightCols;first=t.pp+(part-leftCores)*cols;}
}
}}
input.SetGlobalBuffer((__gm__ T*)ai);output.SetGlobalBuffer((__gm__ T*)ao);
exchange.SetGlobalBuffer((__gm__ float*)ws+groupid*2*t.cols*(t.pp+8));
iall.SetGlobalBuffer((__gm__ float*)ws+t.groups*2*t.cols*(t.pp+8));state=iall[t.batch*t.pp*t.pp];
pipe.InitBuffer(buf,194560);mat=buf.Get<float>();rm=mat[20000];panel=rm;piv=mat[40000];tmp=mat[41024];idx=mat.template ReinterpretCast<uint32_t>()[42048];perm=mat.template ReinterpretCast<uint32_t>()[43072];invs=mat[43584];broad=mat[44096];raw=mat.template ReinterpretCast<T>()[45056*4/sizeof(T)];ofs=raw.template ReinterpretCast<uint32_t>();red=mat[47872];
}
__aicore__ inline bool Run(){
bool failed=false;
for(uint32_t batch=groupid;batch<t.batch;batch+=t.groups){
outputReady=false;cubePrecision=true;uint32_t base=batch*t.p*t.p,ac=first<t.p?(t.p-first<cols?t.p-first:cols):0;
Duplicate(rm,0.0f,cols*t.pp);Fence();
if constexpr(sizeof(T)==4){if(ac)DataCopyPad(rm,input[base+first],{(uint16_t)t.p,ac*4,(t.p-ac)*4,(cols-(ac+7)/8*8)/8,0},{false,0,0,0});}
else for(uint32_t i=0;i<t.p;i++)if(ac){DataCopyPad(raw,input[base+i*t.p+first],{1,ac*2,0,0,0},{false,0,0,0});Fence();Cast(rm[i*cols],raw,RoundMode::CAST_NONE,ac);Fence();}
for(uint32_t i=0;i<8;i++)idx.SetValue(i,i*cols*4);
for(uint32_t c=0;c<8;c++)ofs.SetValue(c,c*ld*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<t.pp;step*=2){Adds(idx[step].template ReinterpretCast<int32_t>(),idx.template ReinterpretCast<int32_t>(),int32_t(step*cols*4),step);PipeBarrier<PIPE_V>();}
for(uint32_t step=8;step<cols;step*=2){Adds(ofs[step].template ReinterpretCast<int32_t>(),ofs.template ReinterpretCast<int32_t>(),int32_t(step*ld*4),step);PipeBarrier<PIPE_V>();}
Fence();
for(uint32_t c=0;c<ac;c++)Gather(mat[c*ld],rm,idx,c*4,t.pp);
if constexpr(sizeof(T)==4 && N>0){
if(ac){auto savedA=state[(t.group*t.groups+1)*8+t.batch*t.pp*t.pp+batch*t.pp*t.pp];
LocalSync<HardEvent::V_MTE3>();DataCopyPad(savedA[first*t.pp],mat,{(uint16_t)ac,t.pp*4,1,0,0});LocalSync<HardEvent::MTE3_V>();}
}
Duplicate(tmp,0.0f,t.pp);PipeBarrier<PIPE_V>();
for(uint32_t c=0;c<ac;c++){MulAddDst(tmp,mat[c*ld],mat[c*ld],t.pp);PipeBarrier<PIPE_V>();}
Sum(red,tmp);
for(uint32_t c=0;c<cols;c++)if(first+c>=t.pp && first+c<2*t.pp)Duplicate(mat[c*ld],0.0f,t.pp);
Duplicate(idx,uint32_t(0x7fffffff),t.pp);
LocalSync<HardEvent::V_S>();float asq=red.GetValue(0);
for(uint32_t c=0;c<cols;c++)if(first+c>=t.pp && first+c<t.pp+t.p)mat.SetValue(c*ld+first+c-t.pp,1.0f);
LocalSync<HardEvent::S_V>();
bool bad=false;
uint32_t c1=first<2*t.pp?(2*t.pp-first<cols?2*t.pp-first:cols):0;
for(uint32_t pc=0;pc<t.p;pc+=panelCols){
uint32_t slot=((pc/panelCols)&1)*panelCols*(t.pp+8),stop=pc+panelCols<t.p?pc+panelCols:t.p;
if(first==pc){
for(uint32_t j=pc;j<stop;j++){
auto col=mat[(j-first)*ld];And(tmp.template ReinterpretCast<uint16_t>(),col.template ReinterpretCast<uint16_t>(),idx.template ReinterpretCast<uint16_t>(),t.pp*2);PipeBarrier<PIPE_V>();uint32_t row=Pivot();float value=col.GetValue(row);
if(!(value>1e-20f || value<-1e-20f)){bad=true;value=1.0f;}
float inv=1.0f/value;piv=panel[(j-pc)*(t.pp+8)];
col.SetValue(row,0.0f);piv.SetValue(t.pp,(float)(int32_t)row);piv.SetValue(t.pp+1,inv);
idx.SetValue(row,uint32_t(0));
LocalSync<HardEvent::S_V>();
Muls(piv,col,-inv,t.pp);Update(j-first+1,cols,row);
}
if(first+cols>t.pp && first<t.pp+t.p){Gather(tmp,panel,ofs,t.pp*4,stop-pc);PipeBarrier<PIPE_V>();Cast(perm[pc].template ReinterpretCast<int32_t>(),tmp,RoundMode::CAST_RINT,stop-pc);Gather(invs[pc],panel,ofs,(t.pp+1)*4,stop-pc);PipeBarrier<PIPE_V>();}
LocalSync<HardEvent::V_MTE3>();DataCopy(exchange[slot],panel,(stop-pc)*(t.pp+8));Fence();
}Sync();
if(first>pc && c1>0){LocalSync<HardEvent::V_MTE2>();DataCopy(panel,exchange[slot],(stop-pc)*(t.pp+8));Fence();
if(first+cols>t.pp && first<t.pp+t.p){
Gather(tmp,panel,ofs,t.pp*4,stop-pc);PipeBarrier<PIPE_V>();Cast(perm[pc].template ReinterpretCast<int32_t>(),tmp,RoundMode::CAST_RINT,stop-pc);
Gather(invs[pc],panel,ofs,(t.pp+1)*4,stop-pc);PipeBarrier<PIPE_V>();
}
for(uint32_t j=pc;j<stop;j++){piv=panel[(j-pc)*(t.pp+8)];uint32_t row=(uint32_t)(int32_t)piv.GetValue(t.pp);if(first<t.pp)idx.SetValue(row,uint32_t(0));Update(0,c1,row);}
}
piv=mat[40000];
}
auto invall=iall[batch*t.pp*t.pp];float isq=0;
if(first+cols>t.pp && first<t.pp+t.p){
LocalSync<HardEvent::S_V>();ShiftLeft(idx,perm,uint32_t(2),(int32_t)t.p);
Duplicate(broad,0.0f,t.pp);
if constexpr(sizeof(T)==4){
uint32_t begin=first<t.pp?t.pp-first:0;
uint32_t end=first+cols>t.pp+t.p?t.pp+t.p-first:cols;
uint32_t count=end-begin;
if(t.pp>t.p){Duplicate(rm,0.0f,count*t.pp);for(uint32_t r=t.p;r<t.pp;r++)invs.SetValue(r,0.0f);LocalSync<HardEvent::S_V>();}
PipeBarrier<PIPE_V>();
for(uint32_t c=0;c<count;c++)Gather(rm[c*t.pp],mat[(begin+c)*ld],idx,0,t.p);
PipeBarrier<PIPE_V>();
for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;
Mul(rm[r],rm[r],invs[r],mask,count,{1,1,1,(uint8_t)(t.pp/8),(uint8_t)(t.pp/8),0});
}
LocalSync<HardEvent::V_MTE3>();DataCopy(invall[(first+begin-t.pp)*t.pp],rm,count*t.pp);
for(uint32_t c=0;c<count;c++){MulAddDst(broad,rm[c*t.pp],rm[c*t.pp],t.pp);PipeBarrier<PIPE_V>();}
LocalSync<HardEvent::MTE3_S>();
}else{
Duplicate(piv,0.0f,t.pp);PipeBarrier<PIPE_V>();
for(uint32_t c=0;c<cols;c++)if(first+c>=t.pp && first+c<t.pp+t.p){
Gather(piv,mat[c*ld],idx,0,t.p);PipeBarrier<PIPE_V>();Mul(piv,piv,invs,t.p);PipeBarrier<PIPE_V>();
MulAddDst(broad,piv,piv,t.pp);LocalSync<HardEvent::V_MTE3>();DataCopy(invall[(first+c-t.pp)*t.pp],piv,t.pp);LocalSync<HardEvent::MTE3_V>();
}
}
Sum(red,broad);LocalSync<HardEvent::V_S>();isq=red.GetValue(0);
}
Store(state[(part*t.groups+groupid)*8],asq,isq,bad?1.0f:0.0f);Sync();
DataCopy(red,state,t.group*t.groups*8);LocalSync<HardEvent::MTE2_V>();
for(uint32_t s=32;s>0;s>>=1){if(t.group>s){uint32_t rem=t.group-s<s?t.group-s:s;Add(red,red,red[s*t.groups*8],rem*t.groups*8);PipeBarrier<PIPE_V>();}}
LocalSync<HardEvent::V_S>();
float needs=0,unbounded=0;bool accept=false,myOk=false,sensitive=false;
for(uint32_t g=0;g<t.groups;g++){
uint32_t idx8=g*8;float aa=red.GetValue(idx8),ii=red.GetValue(idx8+1),bb=red.GetValue(idx8+2);
if constexpr(N==256){if(!(ii*1.4210854715202004e-14f<6.4e-9f))sensitive=true;}
float cut2=t.atol*t.atol>aa*t.rtol*t.rtol?t.atol*t.atol:aa*t.rtol*t.rtol;
bool gAccept=bb==0 && cut2*ii<0.25f,gBounded=bb==0 && cut2*ii<25.0f;
if(!gAccept)needs+=1.0f;
if(!gBounded)unbounded+=1.0f;
if(g==groupid){accept=gAccept||gBounded;myOk=(bb==0);}
}
if(sensitive){accept=false;needs=0;}
if(needs>0){
if(unbounded>0){
float ap=PowerNorm(batch,false),ip=PowerNorm(batch,true);
float spectralCut=t.atol*t.atol>ap*t.rtol*t.rtol*4.0f?t.atol*t.atol:ap*t.rtol*t.rtol*4.0f;
if(myOk && spectralCut*ip*4.0f<25.0f)accept=true;
}
RefineInverse(batch);
}
if(!accept)failed=true;
if(t.batch<=t.groups && unbounded==0 && !sensitive && outputReady)return true;
if(!failed && !outputReady){
if constexpr(sizeof(T)==4){
auto outputIndices=mat[8000].template ReinterpretCast<uint32_t>();
auto columnIndices=mat[12096].template ReinterpretCast<uint32_t>();uint32_t count=t.pp*8;
for(uint32_t c=0;c<8;c++)outputIndices.SetValue(c,c*t.pp*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<count;step*=2){Adds(outputIndices[step].template ReinterpretCast<int32_t>(),outputIndices.template ReinterpretCast<int32_t>(),int32_t(step/8*4),step);PipeBarrier<PIPE_V>();}
for(uint32_t c=part*8;c<t.p;c+=t.group*8){
uint32_t ac=t.p-c<8?t.p-c:8;DataCopy(mat,invall[c*t.pp],ac*t.pp);LocalSync<HardEvent::MTE2_V>();
Gather(rm,mat,outputIndices,0,t.pp*8);LocalSync<HardEvent::V_MTE3>();
DataCopyPad(output[base+c],rm,{(uint16_t)t.p,ac*4,0,(t.p-ac)*4,0});Fence();
}
}else{
for(uint32_t c=part;c<t.p;c+=t.group){DataCopy(tmp,invall[c*t.pp],t.pp);Fence();
Cast(raw,tmp,RoundMode::CAST_RINT,t.p);Fence();auto bits=raw.template ReinterpretCast<uint16_t>(),packed=rm.template ReinterpretCast<uint16_t>();
for(uint32_t i=0;i<t.p;i++)packed.SetValue(i*16,bits.GetValue(i));LocalSync<HardEvent::S_MTE3>();
DataCopyPad(output[base+c],rm.template ReinterpretCast<T>(),{(uint16_t)t.p,2,0,(t.p-1)*2,0});Fence();
}
}
}Sync();
}
Store(state[GetBlockIdx()*8],failed?1.0f:0.0f,0,0);Sync();DataCopy(red,state,t.group*t.groups*8);LocalSync<HardEvent::MTE2_S>();float any=0;for(uint32_t k=0;k<t.group*t.groups;k++){any+=red.GetValue(k*8);}Sync();return any==0;
}
};
template<class T,uint32_t N=0,bool FIXP=false> class Independent199Inverse {
LinalgPinvTilingData t;
GlobalTensor<T> input,output;GlobalTensor<float> exchange,iall,state;
LocalTensor<float> mat,rm,piv,tmp,panel,red,broad,invs;LocalTensor<uint32_t> idx,perm,ofs;LocalTensor<T> raw;
uint32_t part,groupid,first,ld,cols,panelCols;bool outputReady;
__aicore__ inline void Fence(){PipeBarrier<PIPE_ALL>();}
__aicore__ inline void Sync(){Fence();if(t.group*t.groups>1)SyncAll();Fence();}
__aicore__ inline void Store(GlobalTensor<float> dst,float a,float b,float c){red.SetValue(0,a);red.SetValue(1,b);red.SetValue(2,c);LocalSync<HardEvent::S_MTE3>();DataCopy(dst,red,8);LocalSync<HardEvent::MTE3_S>();}
__aicore__ inline void Load(GlobalTensor<float> src){DataCopy(red,src,8);LocalSync<HardEvent::MTE2_S>();}
__aicore__ inline void Sum(LocalTensor<float> dst,LocalTensor<float> src){
uint32_t full=t.pp/64,tail=t.pp%64;auto sums=mat[41536];
if(full)WholeReduceSum(sums,src,64,full,1,1,8);if(tail)WholeReduceSum(sums[full],src[full*64],tail,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(dst,sums,full+(tail?1:0),1,1,1,8);PipeBarrier<PIPE_V>();
}
__aicore__ inline float Dot(LocalTensor<float> x){Mul(tmp,x,x,t.pp);PipeBarrier<PIPE_V>();Sum(red,tmp);LocalSync<HardEvent::V_S>();return red.GetValue(0);}
__aicore__ inline uint32_t Pivot(){
if constexpr(N==256){WholeReduceMax<float,false>(mat[40000],tmp,64,4,1,1,8);LocalSync<HardEvent::V_S>();float maximum,indexBits;GetReduceMaxMinCount(maximum,indexBits);union IndexBits{float value;uint32_t bits;};IndexBits selected;selected.value=indexBits;return selected.bits;}
SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,t.p);WholeReduceMax<float,false>(mat[40000],tmp,64,1,1,1,8);LocalSync<HardEvent::V_S>();float maximum,indexBits;GetReduceMaxMinCount(maximum,indexBits);SetMaskNorm();ResetMask();union IndexBits{float value;uint32_t bits;};IndexBits selected;selected.value=indexBits;return selected.bits;
}
__aicore__ inline void Update(uint32_t c0,uint32_t c1,uint32_t row){
if(c0<c1){Gather(tmp,mat[c0*ld],ofs,row*4,c1-c0);PipeBarrier<PIPE_V>();
if(c0==0 && c1<=56 && panelCols<=56){auto low=mat[13376],high=mat[33408];Brcb(low,tmp,(c1+7)/8,{10,80});Brcb(high,tmp,(c1+7)/8,{10,80});PipeBarrier<PIPE_V>();MulAddDst(mat,piv,low,64,c1,{1,1,0,26,0,10});MulAddDst(mat[64],piv[64],high,64,c1,{1,1,0,26,0,10});MulAddDst(mat[128],piv[128],low,64,c1,{1,1,0,26,0,10});MulAddDst(mat[192],piv[192],high,8,c1,{1,1,0,26,0,10});}
else{Brcb(broad,tmp,(c1-c0+7)/8,{1,8});PipeBarrier<PIPE_V>();for(uint32_t r=0;r<t.pp;r+=64){uint32_t count=t.pp-r<64?t.pp-r:64;MulAddDst(mat[c0*ld+r],piv[r],broad,count,c1-c0,{1,1,0,(uint8_t)(ld/8),0,1});}}
}PipeBarrier<PIPE_V>();
}
__aicore__ inline float NormRoot(float z){red.SetValue(0,z>0?z:0);LocalSync<HardEvent::S_V>();Sqrt(red,red,1);LocalSync<HardEvent::V_S>();return red.GetValue(0);}
__aicore__ inline void PowerRow(uint32_t batch,uint32_t row,bool transpose,bool inverse,LocalTensor<float> dst){
Duplicate(dst,0.0f,t.pp);Fence();
if(inverse){
if(transpose){DataCopy(dst,iall[batch*t.pp*t.pp+row*t.pp],t.pp);Fence();}
else{DataCopyPad(rm,iall[batch*t.pp*t.pp+row],{(uint16_t)t.p,4,(t.pp-1)*4,0,0},{false,0,0,0});Fence();Gather(dst,rm,idx,0,t.p);Fence();}
}else{
uint32_t base=batch*t.p*t.p;
if constexpr(sizeof(T)==4){
if(!transpose)DataCopyPad(dst,input[base+row*t.p],{1,t.p*4,0,0,0},{true,0,(uint8_t)(t.pp-t.p),0});
else{DataCopyPad(rm,input[base+row],{(uint16_t)t.p,4,(t.p-1)*4,0,0},{false,0,0,0});Fence();Gather(dst,rm,idx,0,t.p);}Fence();
}else{
auto packed=rm.template ReinterpretCast<T>();auto cv=mat[26000];
if(!transpose){DataCopyPad(packed,input[base+row*t.p],{1,t.p*2,0,0,0},{false,0,0,0});Fence();Cast(dst,packed,RoundMode::CAST_NONE,t.p);}
else{DataCopyPad(packed,input[base+row],{(uint16_t)t.p,2,(t.p-1)*2,0,0},{false,0,0,0});Fence();Cast(cv,packed,RoundMode::CAST_NONE,t.p*16);Fence();Gather(dst,cv,perm,0,t.p);}Fence();
}
}
}
__aicore__ __attribute__((noinline)) float PowerNorm(uint32_t batch,bool inverse){
if constexpr(FIXP && N>0){t.p=N;t.pp=(N+7)/8*8;}
auto x=mat[38000],y=mat[38512],vals=mat[39024];uint32_t rows=(t.p+t.group*8-1)/(t.group*8)*8,firstRow=part*rows,ac=firstRow<t.p?(t.p-firstRow<rows?t.p-firstRow:rows):0;
auto cache=mat,cacheT=mat[rows*t.pp];
Duplicate(x,0.0f,t.pp);Fence();for(uint32_t j=0;j<t.p;j++){x.SetValue(j,(float)((int32_t)((j*17+13)%31)-15));idx.SetValue(j,j*32);perm.SetValue(j,j*64);}LocalSync<HardEvent::S_V>();
for(uint32_t r=0;r<ac;r++){PowerRow(batch,firstRow+r,false,inverse,cache[r*t.pp]);PowerRow(batch,firstRow+r,true,inverse,cacheT[r*t.pp]);}
float nx=NormRoot(Dot(x));Muls(x,x,nx>0?1.0f/nx:0.0f,t.pp);Fence();float estimate=0;
for(uint32_t it=0;it<4;it++){
for(uint32_t r=0;r<ac;r++){Mul(tmp,cache[r*t.pp],x,t.pp);PipeBarrier<PIPE_V>();Sum(red,tmp);LocalSync<HardEvent::V_S>();vals.SetValue(r,red.GetValue(0));}
if(ac){LocalSync<HardEvent::S_MTE3>();DataCopyPad(exchange[firstRow],vals,{1,ac*4,0,0,0});LocalSync<HardEvent::MTE3_S>();}
Sync();DataCopy(y,exchange,t.pp);Fence();if(t.pp>t.p)Duplicate(y[t.p],0.0f,t.pp-t.p);Fence();estimate=Dot(y);float ny=NormRoot(estimate);Muls(y,y,ny>0?1.0f/ny:0.0f,t.pp);Fence();
for(uint32_t r=0;r<ac;r++){Mul(tmp,cacheT[r*t.pp],y,t.pp);PipeBarrier<PIPE_V>();Sum(red,tmp);LocalSync<HardEvent::V_S>();vals.SetValue(r,red.GetValue(0));}
if(ac){LocalSync<HardEvent::S_MTE3>();DataCopyPad(exchange[t.pp+firstRow],vals,{1,ac*4,0,0,0});LocalSync<HardEvent::MTE3_S>();}
Sync();DataCopy(x,exchange[t.pp],t.pp);Fence();if(t.pp>t.p)Duplicate(x[t.p],0.0f,t.pp-t.p);Fence();nx=NormRoot(Dot(x));Muls(x,x,nx>0?1.0f/nx:0.0f,t.pp);Fence();
}return estimate;
}
__aicore__ inline void RefineInverse(uint32_t batch){
auto residual=state[(t.group*t.groups+1)*8+batch*t.pp*t.pp];auto corrected=state[(t.group*t.groups+1)*8+t.batch*t.pp*t.pp+batch*t.pp*t.pp];auto inv=iall[batch*t.pp*t.pp];
if constexpr(!(sizeof(T)==4 && N>0)){for(uint32_t j=0;j<t.p;j++){idx.SetValue(j,j*32);perm.SetValue(j,j*64);}LocalSync<HardEvent::S_V>();}
uint32_t wp=64;while(wp<t.pp)wp*=2;
auto row=mat[38000],out=mat[38512];uint32_t chunk=16384/wp;if(chunk>64)chunk=64;
if constexpr(sizeof(T)==4 && N>0){
auto savedA=corrected;auto acc=mat,acc1=mat[2048],acc2=mat[4096],acc3=mat[6144],coeff=mat[8192],cache=mat[N==256?12352:12288],br=mat[30000];
auto outputIndices=mat[40000].template ReinterpretCast<uint32_t>();auto columnIndices=br.template ReinterpretCast<uint32_t>();uint32_t count=t.pp*8;
for(uint32_t c=0;c<8;c++)outputIndices.SetValue(c,c*t.pp*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<count;step*=2){Adds(outputIndices[step].template ReinterpretCast<int32_t>(),outputIndices.template ReinterpretCast<int32_t>(),int32_t(step/8*4),step);PipeBarrier<PIPE_V>();}
for(uint32_t c0=part*8;c0<t.p;c0+=t.group*8){
uint32_t ac=t.p-c0<8?t.p-c0:8;
DataCopy(acc,inv[c0*t.pp],ac*t.pp);LocalSync<HardEvent::MTE2_V>();
Gather(coeff,acc,outputIndices,0,t.pp*8);PipeBarrier<PIPE_V>();Duplicate(acc,0.0f,8192);PipeBarrier<PIPE_V>();
for(uint32_t cc=0;cc<t.p;cc+=chunk){uint32_t kc=t.p-cc<chunk?t.p-cc:chunk;
DataCopy(cache,savedA[cc*t.pp],kc*t.pp);Brcb(br,coeff[cc*8],kc,{1,8});LocalSync<HardEvent::MTE2_V>();PipeBarrier<PIPE_V>();
for(uint32_t j=0;j<kc;j+=4){
for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc[r],cache[j*t.pp+r],br[j*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(j+1<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc1[r],cache[(j+1)*t.pp+r],br[(j+1)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(j+2<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc2[r],cache[(j+2)*t.pp+r],br[(j+2)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(j+3<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc3[r],cache[(j+3)*t.pp+r],br[(j+3)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
PipeBarrier<PIPE_V>();
}LocalSync<HardEvent::V_MTE2>();
}
Add(acc,acc,acc1,ac*t.pp);Add(acc2,acc2,acc3,ac*t.pp);PipeBarrier<PIPE_V>();Add(acc,acc,acc2,ac*t.pp);PipeBarrier<PIPE_V>();Muls(acc,acc,-1.0f,ac*t.pp);LocalSync<HardEvent::V_S>();
for(uint32_t c=0;c<ac;c++)acc.SetValue(c*t.pp+c0+c,acc.GetValue(c*t.pp+c0+c)+1.0f);
LocalSync<HardEvent::S_V>();Gather(rm,acc,outputIndices,0,t.pp*8);LocalSync<HardEvent::V_MTE3>();DataCopy(residual[c0*t.pp],rm,t.pp*8);Fence();
}Sync();
}
auto acc=mat,cf=mat[8192],cache=mat[N==256?12352:12288],br=mat[30000],vals=mat[34560];auto ix=mat.template ReinterpretCast<uint32_t>()[35584];
auto outputIndices=mat.template ReinterpretCast<uint32_t>()[40000];
if constexpr(sizeof(T)==4 && N==0){
auto columnIndices=br.template ReinterpretCast<uint32_t>();uint32_t count=t.pp*8;
for(uint32_t c=0;c<8;c++)outputIndices.SetValue(c,c*t.pp*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<count;step*=2){Adds(outputIndices[step].template ReinterpretCast<int32_t>(),outputIndices.template ReinterpretCast<int32_t>(),int32_t(step/8*4),step);PipeBarrier<PIPE_V>();}
}
for(uint32_t c0=part*8;c0<t.p;c0+=t.group*8){
uint32_t ac=t.p-c0<8?t.p-c0:8;DataCopy(acc,inv[c0*t.pp],ac*t.pp);
DataCopy(cf,residual[c0*t.pp],8*t.pp);
auto acc1=mat[2048],acc2=mat[4096],acc3=mat[6144];
if(t.pp<=256)Duplicate(acc1,0.0f,6144);
else Duplicate(acc2,0.0f,ac*t.pp);
Fence();if(t.pp>t.p){Duplicate(cf[t.p*8],0.0f,(t.pp-t.p)*8);PipeBarrier<PIPE_V>();}
for(uint32_t cc=0;cc<t.p;cc+=chunk){uint32_t kc=t.p-cc<chunk?t.p-cc:chunk;
DataCopy(cache,inv[cc*t.pp],kc*t.pp);
Brcb(br,cf[cc*8],kc,{1,8});LocalSync<HardEvent::MTE2_V>();PipeBarrier<PIPE_V>();
if(t.pp<=256){
for(uint32_t k=0;k<kc;k+=4){
for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc[r],cache[k*t.pp+r],br[k*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(k+1<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc1[r],cache[(k+1)*t.pp+r],br[(k+1)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(k+2<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc2[r],cache[(k+2)*t.pp+r],br[(k+2)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(k+3<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc3[r],cache[(k+3)*t.pp+r],br[(k+3)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
PipeBarrier<PIPE_V>();
}
}else{
for(uint32_t k=0;k<kc;k+=2){
for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc[r],cache[k*t.pp+r],br[k*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
if(k+1<kc)for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;MulAddDst(acc2[r],cache[(k+1)*t.pp+r],br[(k+1)*64],mask,ac,{1,1,0,(uint8_t)(t.pp/8),0,1});}
PipeBarrier<PIPE_V>();
}
}
LocalSync<HardEvent::V_MTE2>();
}
if(t.pp<=256){Add(acc,acc,acc1,ac*t.pp);Add(acc2,acc2,acc3,ac*t.pp);PipeBarrier<PIPE_V>();}
Add(acc,acc,acc2,ac*t.pp);PipeBarrier<PIPE_V>();
if constexpr(sizeof(T)==4){
Gather(rm,acc,outputIndices,0,t.pp*8);LocalSync<HardEvent::V_MTE3>();
DataCopyPad(output[batch*t.p*t.p+c0],rm,{(uint16_t)t.p,ac*4,0,(t.p-ac)*4,0});Fence();
}
else{DataCopy(corrected[c0*t.pp],acc,ac*t.pp);Fence();}
}Sync();if constexpr(sizeof(T)==4){outputReady=true;return;}
for(uint32_t c=part;c<t.p;c+=t.group){DataCopy(out,corrected[c*t.pp],t.pp);Fence();DataCopy(inv[c*t.pp],out,t.pp);Fence();}Sync();
}
public:
__aicore__ inline void Init(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData tt){
outputReady=false;t=tt;if constexpr(FIXP && N>0){t.m=N;t.n=N;t.p=N;t.q=N;t.pp=(N+7)/8*8;t.qp=t.pp;}ld=t.pp+8;part=GetBlockIdx()%t.group;groupid=GetBlockIdx()/t.group;cols=t.cols;panelCols=t.cols;first=part*cols;
if constexpr(N==199 && sizeof(T)==4){
uint32_t singleCols=(t.pp+t.group*8-1)/(t.group*8)*8;if(singleCols<16)singleCols=16;
if(singleCols<=t.cols && singleCols*ld<=19000){panelCols=singleCols;cols=singleCols;first=part*cols;}
}
input.SetGlobalBuffer((__gm__ T*)ai);output.SetGlobalBuffer((__gm__ T*)ao);
exchange.SetGlobalBuffer((__gm__ float*)ws+groupid*2*t.cols*(t.pp+8));
iall.SetGlobalBuffer((__gm__ float*)ws+t.groups*2*t.cols*(t.pp+8));state=iall[t.batch*t.pp*t.pp];
InitSocState();mat=LocalTensor<float>(TPosition::VECCALC,0,48640);rm=mat[20000];panel=rm;piv=mat[40000];tmp=mat[41024];idx=mat.template ReinterpretCast<uint32_t>()[42048];perm=mat.template ReinterpretCast<uint32_t>()[43072];invs=mat[43584];broad=mat[44096];raw=mat.template ReinterpretCast<T>()[45056*4/sizeof(T)];ofs=raw.template ReinterpretCast<uint32_t>();red=mat[47872];
}
__aicore__ inline bool Run(){
bool failed=false;
for(uint32_t batch=groupid;batch<t.batch;batch+=t.groups){
outputReady=false;uint32_t base=batch*t.p*t.p,ac=first<t.p?(t.p-first<cols?t.p-first:cols):0;
Duplicate(rm,0.0f,cols*t.pp);Fence();
if constexpr(sizeof(T)==4){if(ac)DataCopyPad(rm,input[base+first],{(uint16_t)t.p,ac*4,(t.p-ac)*4,(cols-(ac+7)/8*8)/8,0},{false,0,0,0});}
else for(uint32_t i=0;i<t.p;i++)if(ac){DataCopyPad(raw,input[base+i*t.p+first],{1,ac*2,0,0,0},{false,0,0,0});Fence();Cast(rm[i*cols],raw,RoundMode::CAST_NONE,ac);Fence();}
for(uint32_t i=0;i<8;i++)idx.SetValue(i,i*cols*4);
for(uint32_t c=0;c<8;c++)ofs.SetValue(c,c*ld*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<t.pp;step*=2){Adds(idx[step].template ReinterpretCast<int32_t>(),idx.template ReinterpretCast<int32_t>(),int32_t(step*cols*4),step);PipeBarrier<PIPE_V>();}
for(uint32_t step=8;step<cols;step*=2){Adds(ofs[step].template ReinterpretCast<int32_t>(),ofs.template ReinterpretCast<int32_t>(),int32_t(step*ld*4),step);PipeBarrier<PIPE_V>();}
Fence();
for(uint32_t c=0;c<ac;c++)Gather(mat[c*ld],rm,idx,c*4,t.pp);
Duplicate(panel,0.0f,1024);PipeBarrier<PIPE_V>();
for(uint32_t c=0;c<ac;c+=4){
MulAddDst(panel,mat[c*ld],mat[c*ld],t.pp);
if(c+1<ac)MulAddDst(panel[256],mat[(c+1)*ld],mat[(c+1)*ld],t.pp);
if(c+2<ac)MulAddDst(panel[512],mat[(c+2)*ld],mat[(c+2)*ld],t.pp);
if(c+3<ac)MulAddDst(panel[768],mat[(c+3)*ld],mat[(c+3)*ld],t.pp);
PipeBarrier<PIPE_V>();
}
Add(panel,panel,panel[256],512);PipeBarrier<PIPE_V>();
Add(tmp,panel,panel[512],t.pp);PipeBarrier<PIPE_V>();
Sum(red,tmp);
Duplicate(idx,uint32_t(0x7fffffff),t.pp);
LocalSync<HardEvent::V_S>();float asq=red.GetValue(0);
LocalSync<HardEvent::S_V>();
bool bad=false;
for(uint32_t pc=0;pc<t.p;pc+=panelCols){
uint32_t slot=((pc/panelCols)&1)*panelCols*(t.pp+8),stop=pc+panelCols<t.p?pc+panelCols:t.p;
if(first==pc){
for(uint32_t j=pc;j<stop;j++){
auto col=mat[(j-first)*ld];And(tmp.template ReinterpretCast<uint16_t>(),col.template ReinterpretCast<uint16_t>(),idx.template ReinterpretCast<uint16_t>(),t.pp*2);PipeBarrier<PIPE_V>();uint32_t row=Pivot();
uint32_t c0=j-first+1;
auto lowC=mat[13376+(c0&7)*80],highC=mat[33408+(c0&7)*80];
if(c0<ac){Gather(tmp,mat[c0*ld],ofs,row*4,ac-c0);PipeBarrier<PIPE_V>();Brcb(lowC,tmp,(ac-c0+7)/8,{10,80});Brcb(highC,tmp,(ac-c0+7)/8,{10,80});}
float value=col.GetValue(row);
if(!(value>1e-20f || value<-1e-20f)){bad=true;value=1.0f;}
float inv=1.0f/value;piv=panel[(j-pc)*(t.pp+8)];
col.SetValue(row,0.0f);piv.SetValue(t.pp,(float)(int32_t)row);piv.SetValue(t.pp+1,inv);
idx.SetValue(row,uint32_t(0));perm.SetValue(j,row);invs.SetValue(j,inv);
LocalSync<HardEvent::S_V>();
Muls(piv,col,-inv,t.pp);PipeBarrier<PIPE_V>();
if(c0<ac){
MulAddDst(mat[c0*ld],piv,lowC,64,ac-c0,{1,1,0,26,0,10});
MulAddDst(mat[c0*ld+64],piv[64],highC,64,ac-c0,{1,1,0,26,0,10});
MulAddDst(mat[c0*ld+128],piv[128],lowC,64,ac-c0,{1,1,0,26,0,10});
MulAddDst(mat[c0*ld+192],piv[192],highC,8,ac-c0,{1,1,0,26,0,10});
PipeBarrier<PIPE_V>();
}
}
LocalSync<HardEvent::V_MTE3>();DataCopy(exchange[slot],panel,(stop-pc)*(t.pp+8));Fence();
}Sync();
if(first==pc){
uint32_t cnt=stop-pc;
Adds(mat,panel,0.0f,cnt*ld);LocalSync<HardEvent::V_S>();
for(uint32_t m=0;m<cnt;m++){uint32_t r=perm.GetValue(pc+m);mat.SetValue(m*ld+r,1.0f);}
LocalSync<HardEvent::S_V>();
for(uint32_t m=1;m<cnt;m++){piv=panel[m*ld];Update(0,m,perm.GetValue(pc+m));}
}
if(first!=pc && ac>0){LocalSync<HardEvent::V_MTE2>();DataCopy(panel,exchange[slot],(stop-pc)*(t.pp+8));Fence();
auto low0=mat[13376],high0=mat[33408],low1=mat[15424],high1=mat[35456];
piv=panel;uint32_t row=(uint32_t)(int32_t)piv.GetValue(t.pp);perm.SetValue(pc,row);invs.SetValue(pc,piv.GetValue(t.pp+1));if(first>pc)idx.SetValue(row,uint32_t(0));
Gather(tmp,mat,ofs,row*4,ac);PipeBarrier<PIPE_V>();
Brcb(low0,tmp,(ac+7)/8,{10,80});Brcb(high0,tmp,(ac+7)/8,{10,80});PipeBarrier<PIPE_V>();
for(uint32_t j=pc;j<stop;j++){
piv=panel[(j-pc)*(t.pp+8)];
auto lCur=((j-pc)&1)?low1:low0,hCur=((j-pc)&1)?high1:high0;
auto lNxt=((j-pc)&1)?low0:low1,hNxt=((j-pc)&1)?high0:high1;
if(j+1<stop){
auto nextPiv=panel[(j+1-pc)*(t.pp+8)];
uint32_t nextRow=(uint32_t)(int32_t)nextPiv.GetValue(t.pp);perm.SetValue(j+1,nextRow);invs.SetValue(j+1,nextPiv.GetValue(t.pp+1));if(first>pc)idx.SetValue(nextRow,uint32_t(0));
uint32_t q0=nextRow&192,q1=(q0+64)&192,q2=(q0+128)&192,q3=(q0+192)&192;
MulAddDst(mat[q0],piv[q0],(q0&64)?hCur:lCur,q0==192?8:64,ac,{1,1,0,26,0,10});
PipeBarrier<PIPE_V>();
Gather(tmp,mat,ofs,nextRow*4,ac);
MulAddDst(mat[q1],piv[q1],(q1&64)?hCur:lCur,q1==192?8:64,ac,{1,1,0,26,0,10});
PipeBarrier<PIPE_V>();
Brcb(lNxt,tmp,(ac+7)/8,{10,80});Brcb(hNxt,tmp,(ac+7)/8,{10,80});
MulAddDst(mat[q2],piv[q2],(q2&64)?hCur:lCur,q2==192?8:64,ac,{1,1,0,26,0,10});
MulAddDst(mat[q3],piv[q3],(q3&64)?hCur:lCur,q3==192?8:64,ac,{1,1,0,26,0,10});
PipeBarrier<PIPE_V>();
}else{
MulAddDst(mat,piv,lCur,64,ac,{1,1,0,26,0,10});
MulAddDst(mat[64],piv[64],hCur,64,ac,{1,1,0,26,0,10});
MulAddDst(mat[128],piv[128],lCur,64,ac,{1,1,0,26,0,10});
MulAddDst(mat[192],piv[192],hCur,8,ac,{1,1,0,26,0,10});
PipeBarrier<PIPE_V>();
}
}
}
piv=mat[40000];
}
auto invall=iall[batch*t.pp*t.pp];float isq=0;
if(ac>0){
LocalSync<HardEvent::S_V>();ShiftLeft(idx,perm,uint32_t(2),(int32_t)t.p);
Duplicate(rm,0.0f,ac*t.pp);for(uint32_t r=t.p;r<t.pp;r++)invs.SetValue(r,0.0f);LocalSync<HardEvent::S_V>();PipeBarrier<PIPE_V>();
for(uint32_t c=0;c<ac;c++)Gather(rm[c*t.pp],mat[c*ld],idx,0,t.p);
PipeBarrier<PIPE_V>();
for(uint32_t r=0;r<t.pp;r+=64){uint32_t mask=t.pp-r<64?t.pp-r:64;Mul(rm[r],rm[r],invs[r],mask,ac,{1,1,1,(uint8_t)(t.pp/8),(uint8_t)(t.pp/8),0});}
LocalSync<HardEvent::V_MTE3>();
for(uint32_t c=0;c<ac;c++)DataCopy(invall[perm.GetValue(first+c)*t.pp],rm[c*t.pp],t.pp);
Duplicate(mat,0.0f,1024);PipeBarrier<PIPE_V>();
for(uint32_t c=0;c<ac;c+=4){
MulAddDst(mat,rm[c*t.pp],rm[c*t.pp],t.pp);
if(c+1<ac)MulAddDst(mat[256],rm[(c+1)*t.pp],rm[(c+1)*t.pp],t.pp);
if(c+2<ac)MulAddDst(mat[512],rm[(c+2)*t.pp],rm[(c+2)*t.pp],t.pp);
if(c+3<ac)MulAddDst(mat[768],rm[(c+3)*t.pp],rm[(c+3)*t.pp],t.pp);
PipeBarrier<PIPE_V>();
}
Add(mat,mat,mat[256],512);PipeBarrier<PIPE_V>();
Add(broad,mat,mat[512],t.pp);PipeBarrier<PIPE_V>();
LocalSync<HardEvent::MTE3_S>();
Sum(red,broad);LocalSync<HardEvent::V_S>();isq=red.GetValue(0);
}
Store(state[(part*t.groups+groupid)*8],asq,isq,bad?1.0f:0.0f);Sync();
DataCopy(red,state,t.group*t.groups*8);LocalSync<HardEvent::MTE2_V>();
for(uint32_t s=32;s>0;s>>=1){if(t.group>s){uint32_t rem=t.group-s<s?t.group-s:s;Add(red,red,red[s*t.groups*8],rem*t.groups*8);PipeBarrier<PIPE_V>();}}
LocalSync<HardEvent::V_S>();
float needs=0,unbounded=0;bool accept=false,myOk=false,sensitive=false;
for(uint32_t g=0;g<t.groups;g++){
uint32_t idx8=g*8;float aa=red.GetValue(idx8),ii=red.GetValue(idx8+1),bb=red.GetValue(idx8+2);
if(!(ii*1.4210854715202004e-14f<6.4e-9f))sensitive=true;
float cut2=t.atol*t.atol>aa*t.rtol*t.rtol?t.atol*t.atol:aa*t.rtol*t.rtol;
bool gAccept=bb==0 && cut2*ii<0.25f,gBounded=bb==0 && cut2*ii<25.0f;
if(!gAccept)needs+=1.0f;
if(!gBounded)unbounded+=1.0f;
if(g==groupid){accept=gAccept||gBounded;myOk=(bb==0);}
}
if(sensitive){accept=false;needs=0;}
if(needs>0){
if(unbounded>0){
float ap=PowerNorm(batch,false),ip=PowerNorm(batch,true);
float spectralCut=t.atol*t.atol>ap*t.rtol*t.rtol*4.0f?t.atol*t.atol:ap*t.rtol*t.rtol*4.0f;
if(myOk && spectralCut*ip*4.0f<25.0f)accept=true;
}
}
if(!accept)failed=true;
if(!failed && !outputReady){
if constexpr(sizeof(T)==4){
auto loaded=mat,packed=mat[40000];auto rowIndices=mat[42048].template ReinterpretCast<uint32_t>();
if constexpr(FIXP && N==199){
DataCopy(loaded,invall,40000);
for(uint32_t c=0;c<8;c++)rowIndices.SetValue(c,c*200*4);LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<200;step*=2){Adds(rowIndices[step].template ReinterpretCast<int32_t>(),rowIndices.template ReinterpretCast<int32_t>(),int32_t(step*200*4),step);PipeBarrier<PIPE_V>();}
LocalSync<HardEvent::MTE2_V>();
for(uint32_t r=part*8;r<199;r+=t.group*8){uint32_t ac=199-r<8?199-r:8;for(uint32_t k=0;k<ac;k++)Gather(packed[k*200],loaded,rowIndices,(r+k)*4,199);LocalSync<HardEvent::V_MTE3>();DataCopyPad(output[base+r*199],packed,{(uint16_t)ac,199*4,0,0,0});LocalSync<HardEvent::MTE3_V>();}
}else{
DataCopy(loaded,invall,t.p*t.pp);
for(uint32_t c=0;c<8;c++)rowIndices.SetValue(c,c*t.pp*4);LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<t.pp;step*=2){Adds(rowIndices[step].template ReinterpretCast<int32_t>(),rowIndices.template ReinterpretCast<int32_t>(),int32_t(step*t.pp*4),step);PipeBarrier<PIPE_V>();}
LocalSync<HardEvent::MTE2_V>();
for(uint32_t r=part*8;r<t.p;r+=t.group*8){uint32_t ac=t.p-r<8?t.p-r:8;for(uint32_t k=0;k<ac;k++)Gather(packed[k*t.pp],loaded,rowIndices,(r+k)*4,t.p);LocalSync<HardEvent::V_MTE3>();DataCopyPad(output[base+r*t.p],packed,{(uint16_t)ac,t.p*4,0,0,0});LocalSync<HardEvent::MTE3_V>();}
}
}else{
for(uint32_t c=part;c<t.p;c+=t.group){DataCopy(tmp,invall[c*t.pp],t.pp);Fence();
Cast(raw,tmp,RoundMode::CAST_RINT,t.p);Fence();auto bits=raw.template ReinterpretCast<uint16_t>(),packed=rm.template ReinterpretCast<uint16_t>();
for(uint32_t i=0;i<t.p;i++)packed.SetValue(i*16,bits.GetValue(i));LocalSync<HardEvent::S_MTE3>();
DataCopyPad(output[base+c],rm.template ReinterpretCast<T>(),{(uint16_t)t.p,2,0,(t.p-1)*2,0});Fence();
}
}
if(t.batch<=t.groups && unbounded==0 && !sensitive)return true;
}Sync();
}
Store(state[GetBlockIdx()*8],failed?1.0f:0.0f,0,0);Sync();DataCopy(red,state,t.group*t.groups*8);LocalSync<HardEvent::MTE2_S>();float any=0;for(uint32_t k=0;k<t.group*t.groups;k++){any+=red.GetValue(k*8);}Sync();return any==0;
}
};
static __aicore__ __attribute__((noinline)) bool FactorOwnerFixed256(uint32_t pc,uint32_t count){
if ASCEND_IS_AIV {
auto mem=LocalTensor<float>(TPosition::VECCALC,0,48640);
__ubuf__ float* m=(__ubuf__ float*)0;
__ubuf__ float* panel=m+20000;__ubuf__ float* tmp=m+41024;
__ubuf__ uint32_t* used=(__ubuf__ uint32_t*)(m+42048);
__ubuf__ uint32_t* perm=(__ubuf__ uint32_t*)(m+43072);
__ubuf__ float* invs=m+43584;
bool bad=false;
for(uint32_t j=0;j<count;j++){
__ubuf__ float* col=m+j*264;
set_vector_mask((uint64_t)-1,(uint64_t)-1);
vand((__ubuf__ uint16_t*)tmp,(__ubuf__ uint16_t*)col,(__ubuf__ uint16_t*)used,4,1,1,1,8,8,8);PipeBarrier<PIPE_V>();
vcmax(m+40000,tmp,4,1,1,8,Order_t::VALUE_INDEX);LocalSync<HardEvent::V_S>();
float maximum,indexBits;GetReduceMaxMinCount(maximum,indexBits);union Bits{float value;uint32_t bits;};Bits selected;selected.value=indexBits;uint32_t row=selected.bits;
uint32_t c0=j+1;
if(c0<count){
set_vector_mask((uint64_t)0,((uint64_t)1<<(count-c0))-1);vgather((__ubuf__ uint32_t*)tmp,(__ubuf__ uint32_t*)(m+45056),(c0*264+row)*4,8,1);PipeBarrier<PIPE_V>();
set_vector_mask((uint64_t)-1,(uint64_t)-1);
vbrcb((__ubuf__ uint32_t*)(m+44096),(__ubuf__ uint32_t*)tmp,1,8,(count-c0+7)/8);
vbrcb((__ubuf__ uint32_t*)(m+38912),(__ubuf__ uint32_t*)tmp,1,8,(count-c0+7)/8);
}
float value=col[row];if(!(value>1e-20f || value<-1e-20f)){bad=true;value=1.0f;}
float inv=1.0f/value;__ubuf__ float* piv=panel+j*264;
col[row]=0.0f;piv[256]=float(int32_t(row));piv[257]=inv;used[row]=0;perm[pc+j]=row;invs[pc+j]=inv;
LocalSync<HardEvent::S_V>();set_vector_mask((uint64_t)0,(uint64_t)-1);
vmuls(piv,col,-inv,4,1,1,8,8);PipeBarrier<PIPE_V>();
if(c0<count){
vmla(m+c0*264,piv,m+44096,count-c0,1,1,0,33,0,1);
vmla(m+c0*264+64,piv+64,m+38912,count-c0,1,1,0,33,0,1);
vmla(m+c0*264+128,piv+128,m+44096,count-c0,1,1,0,33,0,1);
vmla(m+c0*264+192,piv+192,m+38912,count-c0,1,1,0,33,0,1);PipeBarrier<PIPE_V>();
}
}
set_vector_mask((uint64_t)-1,(uint64_t)-1);return bad;
}
return false;
}

static __aicore__ __attribute__((noinline)) void ReceiverFixed256(uint32_t pc,uint32_t count,uint32_t first,uint32_t ac){
if ASCEND_IS_AIV {
auto mem=LocalTensor<float>(TPosition::VECCALC,0,48640);
__ubuf__ float* m=(__ubuf__ float*)0;__ubuf__ float* panel=m+20000;
__ubuf__ uint32_t* used=(__ubuf__ uint32_t*)(m+42048);__ubuf__ uint32_t* perm=(__ubuf__ uint32_t*)(m+43072);__ubuf__ float* invs=m+43584;
uint32_t row=uint32_t(int32_t(panel[256]));perm[pc]=row;invs[pc]=panel[257];if(first>pc)used[row]=0;
set_vector_mask((uint64_t)0,((uint64_t)1<<ac)-1);vgather((__ubuf__ uint32_t*)(m+41024),(__ubuf__ uint32_t*)(m+45056),row*4,8,1);PipeBarrier<PIPE_V>();
set_vector_mask((uint64_t)-1,(uint64_t)-1);
vbrcb((__ubuf__ uint32_t*)(m+44096),(__ubuf__ uint32_t*)(m+41024),1,8,(ac+7)/8);
vbrcb((__ubuf__ uint32_t*)(m+38912),(__ubuf__ uint32_t*)(m+41024),1,8,(ac+7)/8);PipeBarrier<PIPE_V>();
for(uint32_t j=0;j<count;j++){
__ubuf__ float* piv=panel+j*264;
__ubuf__ float* bc=m+((j&1)?38400:44096);__ubuf__ float* oc=m+((j&1)?39424:38912);
__ubuf__ float* bn=m+((j&1)?44096:38400);__ubuf__ float* on=m+((j&1)?38912:39424);
set_vector_mask((uint64_t)0,(uint64_t)-1);
if(j+1<count){
uint32_t nextRow=uint32_t(int32_t(piv[264+256]));perm[pc+j+1]=nextRow;invs[pc+j+1]=piv[264+257];if(first>pc)used[nextRow]=0;
uint32_t q0=nextRow&192,q1=(q0+64)&192,q2=(q0+128)&192,q3=(q0+192)&192;
vmla(m+q0,piv+q0,(q0&64)?oc:bc,ac,1,1,0,33,0,1);PipeBarrier<PIPE_V>();
set_vector_mask((uint64_t)0,((uint64_t)1<<ac)-1);vgather((__ubuf__ uint32_t*)(m+41024),(__ubuf__ uint32_t*)(m+45056),nextRow*4,8,1);
set_vector_mask((uint64_t)0,(uint64_t)-1);
vmla(m+q1,piv+q1,(q1&64)?oc:bc,ac,1,1,0,33,0,1);PipeBarrier<PIPE_V>();
set_vector_mask((uint64_t)-1,(uint64_t)-1);
vbrcb((__ubuf__ uint32_t*)bn,(__ubuf__ uint32_t*)(m+41024),1,8,(ac+7)/8);
vbrcb((__ubuf__ uint32_t*)on,(__ubuf__ uint32_t*)(m+41024),1,8,(ac+7)/8);
set_vector_mask((uint64_t)0,(uint64_t)-1);
vmla(m+q2,piv+q2,(q2&64)?oc:bc,ac,1,1,0,33,0,1);
vmla(m+q3,piv+q3,(q3&64)?oc:bc,ac,1,1,0,33,0,1);PipeBarrier<PIPE_V>();
}else{
vmla(m,piv,bc,ac,1,1,0,33,0,1);vmla(m+64,piv+64,oc,ac,1,1,0,33,0,1);
vmla(m+128,piv+128,bc,ac,1,1,0,33,0,1);vmla(m+192,piv+192,oc,ac,1,1,0,33,0,1);PipeBarrier<PIPE_V>();
}
}
set_vector_mask((uint64_t)-1,(uint64_t)-1);
}
}

template<class T,uint32_t N=0> class DeferredInverse256 {
LinalgPinvTilingData t;
GlobalTensor<T> input,output;GlobalTensor<float> exchange,iall,state;
LocalTensor<float> mat,rm,piv,tmp,panel,red,broad,invs;LocalTensor<uint32_t> idx,perm,ofs;LocalTensor<T> raw;
uint32_t part,groupid,first,ld,cols,panelCols;bool outputReady;bool needsRefinement=false;
__aicore__ inline void Fence(){PipeBarrier<PIPE_ALL>();}
__aicore__ inline void Sync(){Fence();if(t.group*t.groups>1)SyncAll();Fence();}
__aicore__ inline void Store(GlobalTensor<float> dst,float a,float b,float c){red.SetValue(0,a);red.SetValue(1,b);red.SetValue(2,c);LocalSync<HardEvent::S_MTE3>();DataCopy(dst,red,8);LocalSync<HardEvent::MTE3_S>();}
__aicore__ inline void Load(GlobalTensor<float> src){DataCopy(red,src,8);LocalSync<HardEvent::MTE2_S>();}
__aicore__ inline void Sum(LocalTensor<float> dst,LocalTensor<float> src){
uint32_t full=t.pp/64,tail=t.pp%64;auto sums=mat[41536];
if(full)WholeReduceSum(sums,src,64,full,1,1,8);if(tail)WholeReduceSum(sums[full],src[full*64],tail,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(dst,sums,full+(tail?1:0),1,1,1,8);PipeBarrier<PIPE_V>();
}
__aicore__ inline float Dot(LocalTensor<float> x){Mul(tmp,x,x,t.pp);PipeBarrier<PIPE_V>();Sum(red,tmp);LocalSync<HardEvent::V_S>();return red.GetValue(0);}
__aicore__ inline uint32_t Pivot(){
if constexpr(N==256){WholeReduceMax<float,false>(mat[40000],tmp,64,4,1,1,8);LocalSync<HardEvent::V_S>();float maximum,indexBits;GetReduceMaxMinCount(maximum,indexBits);union IndexBits{float value;uint32_t bits;};IndexBits selected;selected.value=indexBits;return selected.bits;}
SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,t.p);WholeReduceMax<float,false>(mat[40000],tmp,64,1,1,1,8);LocalSync<HardEvent::V_S>();float maximum,indexBits;GetReduceMaxMinCount(maximum,indexBits);SetMaskNorm();ResetMask();union IndexBits{float value;uint32_t bits;};IndexBits selected;selected.value=indexBits;return selected.bits;
}
__aicore__ inline void Update(uint32_t c0,uint32_t c1,uint32_t row){
if(c0<c1){Gather(tmp,mat[c0*ld],ofs,row*4,c1-c0);PipeBarrier<PIPE_V>();
if constexpr(N==256){auto opposite=mat[38912];Brcb(broad,tmp,(c1-c0+7)/8,{1,8});Brcb(opposite,tmp,(c1-c0+7)/8,{1,8});PipeBarrier<PIPE_V>();MulAddDst(mat[c0*ld],piv,broad,64,c1-c0,{1,1,0,33,0,1});MulAddDst(mat[c0*ld+64],piv[64],opposite,64,c1-c0,{1,1,0,33,0,1});MulAddDst(mat[c0*ld+128],piv[128],broad,64,c1-c0,{1,1,0,33,0,1});MulAddDst(mat[c0*ld+192],piv[192],opposite,64,c1-c0,{1,1,0,33,0,1});}else{Brcb(broad,tmp,(c1-c0+7)/8,{1,8});PipeBarrier<PIPE_V>();for(uint32_t r=0;r<t.pp;r+=64){uint32_t count=t.pp-r<64?t.pp-r:64;MulAddDst(mat[c0*ld+r],piv[r],broad,count,c1-c0,{1,1,0,(uint8_t)(ld/8),0,1});}}
}PipeBarrier<PIPE_V>();
}
__aicore__ inline float NormRoot(float z){red.SetValue(0,z>0?z:0);LocalSync<HardEvent::S_V>();Sqrt(red,red,1);LocalSync<HardEvent::V_S>();return red.GetValue(0);}
__aicore__ inline void PowerRow(uint32_t batch,uint32_t row,bool transpose,bool inverse,LocalTensor<float> dst){
Duplicate(dst,0.0f,t.pp);Fence();
if(inverse){
if(transpose){DataCopy(dst,iall[batch*t.pp*t.pp+row*t.pp],t.pp);Fence();}
else{DataCopyPad(rm,iall[batch*t.pp*t.pp+row],{(uint16_t)t.p,4,(t.pp-1)*4,0,0},{false,0,0,0});Fence();Gather(dst,rm,idx,0,t.p);Fence();}
}else{
uint32_t base=batch*t.p*t.p;
if constexpr(sizeof(T)==4){
if(!transpose)DataCopyPad(dst,input[base+row*t.p],{1,t.p*4,0,0,0},{true,0,(uint8_t)(t.pp-t.p),0});
else{DataCopyPad(rm,input[base+row],{(uint16_t)t.p,4,(t.p-1)*4,0,0},{false,0,0,0});Fence();Gather(dst,rm,idx,0,t.p);}Fence();
}else{
auto packed=rm.template ReinterpretCast<T>();auto cv=mat[26000];
if(!transpose){DataCopyPad(packed,input[base+row*t.p],{1,t.p*2,0,0,0},{false,0,0,0});Fence();Cast(dst,packed,RoundMode::CAST_NONE,t.p);}
else{DataCopyPad(packed,input[base+row],{(uint16_t)t.p,2,(t.p-1)*2,0,0},{false,0,0,0});Fence();Cast(cv,packed,RoundMode::CAST_NONE,t.p*16);Fence();Gather(dst,cv,perm,0,t.p);}Fence();
}
}
}
__aicore__ __attribute__((noinline)) float PowerNorm(uint32_t batch,bool inverse){
if constexpr(N>0){t.p=N;t.pp=(N+7)/8*8;}
auto x=mat[38000],y=mat[38512],vals=mat[39024];uint32_t rows=(t.p+t.group*8-1)/(t.group*8)*8,firstRow=part*rows,ac=firstRow<t.p?(t.p-firstRow<rows?t.p-firstRow:rows):0;
auto cache=mat,cacheT=mat[rows*t.pp];
Duplicate(x,0.0f,t.pp);Fence();for(uint32_t j=0;j<t.p;j++){x.SetValue(j,(float)((int32_t)((j*17+13)%31)-15));idx.SetValue(j,j*32);perm.SetValue(j,j*64);}LocalSync<HardEvent::S_V>();
for(uint32_t r=0;r<ac;r++){PowerRow(batch,firstRow+r,false,inverse,cache[r*t.pp]);PowerRow(batch,firstRow+r,true,inverse,cacheT[r*t.pp]);}
float nx=NormRoot(Dot(x));Muls(x,x,nx>0?1.0f/nx:0.0f,t.pp);Fence();float estimate=0;
for(uint32_t it=0;it<4;it++){
for(uint32_t r=0;r<ac;r++){Mul(tmp,cache[r*t.pp],x,t.pp);PipeBarrier<PIPE_V>();Sum(red,tmp);LocalSync<HardEvent::V_S>();vals.SetValue(r,red.GetValue(0));}
if(ac){LocalSync<HardEvent::S_MTE3>();DataCopyPad(exchange[firstRow],vals,{1,ac*4,0,0,0});LocalSync<HardEvent::MTE3_S>();}
Sync();DataCopy(y,exchange,t.pp);Fence();if(t.pp>t.p)Duplicate(y[t.p],0.0f,t.pp-t.p);Fence();estimate=Dot(y);float ny=NormRoot(estimate);Muls(y,y,ny>0?1.0f/ny:0.0f,t.pp);Fence();
for(uint32_t r=0;r<ac;r++){Mul(tmp,cacheT[r*t.pp],y,t.pp);PipeBarrier<PIPE_V>();Sum(red,tmp);LocalSync<HardEvent::V_S>();vals.SetValue(r,red.GetValue(0));}
if(ac){LocalSync<HardEvent::S_MTE3>();DataCopyPad(exchange[t.pp+firstRow],vals,{1,ac*4,0,0,0});LocalSync<HardEvent::MTE3_S>();}
Sync();DataCopy(x,exchange[t.pp],t.pp);Fence();if(t.pp>t.p)Duplicate(x[t.p],0.0f,t.pp-t.p);Fence();nx=NormRoot(Dot(x));Muls(x,x,nx>0?1.0f/nx:0.0f,t.pp);Fence();
}return estimate;
}
__aicore__ inline void RefineInverse(uint32_t batch){needsRefinement=true;outputReady=true;}
public:

__aicore__ inline bool NeedsRefine(){return needsRefinement;}
__aicore__ inline void Init(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData tt){
outputReady=false;t=tt;if constexpr(N>0){t.m=N;t.n=N;t.p=N;t.q=N;t.pp=(N+7)/8*8;t.qp=t.pp;}ld=t.pp+8;part=GetBlockIdx()%t.group;groupid=GetBlockIdx()/t.group;cols=t.cols;panelCols=t.cols;first=part*cols;
if constexpr(N==256 && sizeof(T)==4){
uint32_t singleCols=(t.pp+t.group*8-1)/(t.group*8)*8;
if(singleCols<=t.cols && singleCols*ld<=19000){panelCols=singleCols;cols=singleCols;first=part*cols;}
}else if constexpr(N>0){if(t.batch>1 && t.group>=3){
uint32_t leftCores=t.group*2/3,rightCores=t.group-leftCores;
uint32_t leftCols=(t.pp+leftCores*8-1)/(leftCores*8)*8,rightCols=(t.pp+rightCores*8-1)/(rightCores*8)*8;
if(leftCols*(t.pp+8)<=19000 && rightCols*(t.pp+8)<=19000 && leftCols<=t.cols){
panelCols=leftCols;
if(part<leftCores){cols=leftCols;first=part*cols;if(first>=t.pp)first=2*t.pp;}
else{cols=rightCols;first=t.pp+(part-leftCores)*cols;}
}
}}
input.SetGlobalBuffer((__gm__ T*)ai);output.SetGlobalBuffer((__gm__ T*)ao);
exchange.SetGlobalBuffer((__gm__ float*)ws+groupid*2*t.cols*(t.pp+8));
iall.SetGlobalBuffer((__gm__ float*)ws+t.groups*2*t.cols*(t.pp+8));state=iall[t.batch*t.pp*t.pp];
InitSocState();mat=LocalTensor<float>(TPosition::VECCALC,0,48640);rm=mat[20000];panel=rm;piv=mat[40000];tmp=mat[41024];idx=mat.template ReinterpretCast<uint32_t>()[42048];perm=mat.template ReinterpretCast<uint32_t>()[43072];invs=mat[43584];broad=mat[44096];raw=mat.template ReinterpretCast<T>()[45056*4/sizeof(T)];ofs=raw.template ReinterpretCast<uint32_t>();red=mat[47872];
}
__aicore__ inline bool Run(){
bool failed=false;
for(uint32_t batch=groupid;batch<t.batch;batch+=t.groups){
outputReady=false;uint32_t base=batch*t.p*t.p,ac=first<t.p?(t.p-first<cols?t.p-first:cols):0;
Duplicate(rm,0.0f,cols*t.pp);Fence();
if constexpr(sizeof(T)==4){if(ac)DataCopyPad(rm,input[base+first],{(uint16_t)t.p,ac*4,(t.p-ac)*4,(cols-(ac+7)/8*8)/8,0},{false,0,0,0});}
else for(uint32_t i=0;i<t.p;i++)if(ac){DataCopyPad(raw,input[base+i*t.p+first],{1,ac*2,0,0,0},{false,0,0,0});Fence();Cast(rm[i*cols],raw,RoundMode::CAST_NONE,ac);Fence();}
for(uint32_t i=0;i<8;i++)idx.SetValue(i,i*cols*4);
for(uint32_t c=0;c<8;c++)ofs.SetValue(c,c*ld*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<t.pp;step*=2){Adds(idx[step].template ReinterpretCast<int32_t>(),idx.template ReinterpretCast<int32_t>(),int32_t(step*cols*4),step);PipeBarrier<PIPE_V>();}
for(uint32_t step=8;step<cols;step*=2){Adds(ofs[step].template ReinterpretCast<int32_t>(),ofs.template ReinterpretCast<int32_t>(),int32_t(step*ld*4),step);PipeBarrier<PIPE_V>();}
Fence();
for(uint32_t c=0;c<ac;c++)Gather(mat[c*ld],rm,idx,c*4,t.pp);
if constexpr(sizeof(T)==4 && N>0){
if(ac){auto savedA=state[(t.group*t.groups+1)*8+t.batch*t.pp*t.pp+batch*t.pp*t.pp];
LocalSync<HardEvent::V_MTE3>();DataCopyPad(savedA[first*t.pp],mat,{(uint16_t)ac,t.pp*4,1,0,0});}
}
Duplicate(panel,0.0f,1024);PipeBarrier<PIPE_V>();
for(uint32_t c=0;c<ac;c+=4){
MulAddDst(panel,mat[c*ld],mat[c*ld],t.pp);
if(c+1<ac)MulAddDst(panel[256],mat[(c+1)*ld],mat[(c+1)*ld],t.pp);
if(c+2<ac)MulAddDst(panel[512],mat[(c+2)*ld],mat[(c+2)*ld],t.pp);
if(c+3<ac)MulAddDst(panel[768],mat[(c+3)*ld],mat[(c+3)*ld],t.pp);
PipeBarrier<PIPE_V>();
}
Add(panel,panel,panel[256],512);PipeBarrier<PIPE_V>();
Add(tmp,panel,panel[512],256);PipeBarrier<PIPE_V>();
Sum(red,tmp);
Duplicate(idx,uint32_t(0x7fffffff),t.pp);
LocalSync<HardEvent::V_S>();float asq=red.GetValue(0);
if constexpr(sizeof(T)==4 && N>0){if(ac)LocalSync<HardEvent::MTE3_V>();}
LocalSync<HardEvent::S_V>();
bool bad=false;
for(uint32_t pc=0;pc<t.p;pc+=panelCols){
uint32_t slot=((pc/panelCols)&1)*panelCols*(t.pp+8),stop=pc+panelCols<t.p?pc+panelCols:t.p;
if(first==pc){
if constexpr(N==256){if(FactorOwnerFixed256(pc,stop-pc))bad=true;}else{
auto opposite=mat[38912];
for(uint32_t j=pc;j<stop;j++){
auto col=mat[(j-first)*ld];And(tmp.template ReinterpretCast<uint16_t>(),col.template ReinterpretCast<uint16_t>(),idx.template ReinterpretCast<uint16_t>(),t.pp*2);PipeBarrier<PIPE_V>();uint32_t row=Pivot();
uint32_t c0=j-first+1;
if(c0<ac){Gather(tmp,mat[c0*ld],ofs,row*4,ac-c0);PipeBarrier<PIPE_V>();Brcb(broad,tmp,(ac-c0+7)/8,{1,8});Brcb(opposite,tmp,(ac-c0+7)/8,{1,8});}
float value=col.GetValue(row);
if(!(value>1e-20f || value<-1e-20f)){bad=true;value=1.0f;}
float inv=1.0f/value;piv=panel[(j-pc)*(t.pp+8)];
col.SetValue(row,0.0f);piv.SetValue(t.pp,(float)(int32_t)row);piv.SetValue(t.pp+1,inv);
idx.SetValue(row,uint32_t(0));perm.SetValue(j,row);invs.SetValue(j,inv);
LocalSync<HardEvent::S_V>();
Muls(piv,col,-inv,t.pp);PipeBarrier<PIPE_V>();
if(c0<ac){
MulAddDst(mat[c0*ld],piv,broad,64,ac-c0,{1,1,0,33,0,1});
MulAddDst(mat[c0*ld+64],piv[64],opposite,64,ac-c0,{1,1,0,33,0,1});
MulAddDst(mat[c0*ld+128],piv[128],broad,64,ac-c0,{1,1,0,33,0,1});
MulAddDst(mat[c0*ld+192],piv[192],opposite,64,ac-c0,{1,1,0,33,0,1});
PipeBarrier<PIPE_V>();
}
}
}
LocalSync<HardEvent::V_MTE3>();DataCopy(exchange[slot],panel,(stop-pc)*(t.pp+8));Fence();
}Sync();
if(first==pc){
uint32_t cnt=stop-pc;
Adds(mat,panel,0.0f,cnt*ld);LocalSync<HardEvent::V_S>();
for(uint32_t m=0;m<cnt;m++){uint32_t r=perm.GetValue(pc+m);mat.SetValue(m*ld+r,1.0f);}
LocalSync<HardEvent::S_V>();
for(uint32_t m=1;m<cnt;m++){piv=panel[m*ld];Update(0,m,perm.GetValue(pc+m));}
}
if(first!=pc && ac>0){LocalSync<HardEvent::V_MTE2>();DataCopy(panel,exchange[slot],(stop-pc)*(t.pp+8));Fence();
if constexpr(N==256){ReceiverFixed256(pc,stop-pc,first,ac);}else{
auto br0=broad,op0=mat[38912],br1=mat[38400],op1=mat[39424];
piv=panel;uint32_t row=(uint32_t)(int32_t)piv.GetValue(t.pp);perm.SetValue(pc,row);invs.SetValue(pc,piv.GetValue(t.pp+1));if(first>pc)idx.SetValue(row,uint32_t(0));
Gather(tmp,mat,ofs,row*4,ac);PipeBarrier<PIPE_V>();
Brcb(br0,tmp,(ac+7)/8,{1,8});Brcb(op0,tmp,(ac+7)/8,{1,8});PipeBarrier<PIPE_V>();
for(uint32_t j=pc;j<stop;j++){
piv=panel[(j-pc)*(t.pp+8)];
auto bCur=((j-pc)&1)?br1:br0,oCur=((j-pc)&1)?op1:op0;
auto bNxt=((j-pc)&1)?br0:br1,oNxt=((j-pc)&1)?op0:op1;
if(j+1<stop){
auto nextPiv=panel[(j+1-pc)*(t.pp+8)];
uint32_t nextRow=(uint32_t)(int32_t)nextPiv.GetValue(t.pp);
perm.SetValue(j+1,nextRow);invs.SetValue(j+1,nextPiv.GetValue(t.pp+1));if(first>pc)idx.SetValue(nextRow,uint32_t(0));
uint32_t q0=nextRow&192,q1=(q0+64)&192,q2=(q0+128)&192,q3=(q0+192)&192;
MulAddDst(mat[q0],piv[q0],(q0&64)?oCur:bCur,64,ac,{1,1,0,33,0,1});
PipeBarrier<PIPE_V>();
Gather(tmp,mat,ofs,nextRow*4,ac);
MulAddDst(mat[q1],piv[q1],(q1&64)?oCur:bCur,64,ac,{1,1,0,33,0,1});
PipeBarrier<PIPE_V>();
Brcb(bNxt,tmp,(ac+7)/8,{1,8});Brcb(oNxt,tmp,(ac+7)/8,{1,8});
MulAddDst(mat[q2],piv[q2],(q2&64)?oCur:bCur,64,ac,{1,1,0,33,0,1});
MulAddDst(mat[q3],piv[q3],(q3&64)?oCur:bCur,64,ac,{1,1,0,33,0,1});
PipeBarrier<PIPE_V>();
}else{
MulAddDst(mat,piv,bCur,64,ac,{1,1,0,33,0,1});
MulAddDst(mat[64],piv[64],oCur,64,ac,{1,1,0,33,0,1});
MulAddDst(mat[128],piv[128],bCur,64,ac,{1,1,0,33,0,1});
MulAddDst(mat[192],piv[192],oCur,64,ac,{1,1,0,33,0,1});
PipeBarrier<PIPE_V>();
}
}
}
}
piv=mat[40000];
}
auto invall=iall[batch*t.pp*t.pp];float isq=0;
if(ac>0){
LocalSync<HardEvent::S_V>();ShiftLeft(idx,perm,uint32_t(2),(int32_t)t.p);
PipeBarrier<PIPE_V>();
for(uint32_t c=0;c<ac;c++)Gather(rm[c*t.pp],mat[c*ld],idx,0,t.p);
PipeBarrier<PIPE_V>();
for(uint32_t r=0;r<t.pp;r+=64)Mul(rm[r],rm[r],invs[r],64,ac,{1,1,1,(uint8_t)(t.pp/8),(uint8_t)(t.pp/8),0});
LocalSync<HardEvent::V_MTE3>();
for(uint32_t c=0;c<ac;c++)DataCopy(invall[perm.GetValue(first+c)*t.pp],rm[c*t.pp],t.pp);
Mul(mat,rm,rm,ac*t.pp);PipeBarrier<PIPE_V>();
WholeReduceSum(tmp,mat,64,ac*4,1,1,8);PipeBarrier<PIPE_V>();
if(ac*4>64){Add(tmp,tmp,tmp[64],ac*4-64);PipeBarrier<PIPE_V>();}
WholeReduceSum(red,tmp,ac*4>64?64:ac*4,1,1,1,8);
LocalSync<HardEvent::MTE3_S>();LocalSync<HardEvent::V_S>();isq=red.GetValue(0);
}
Store(state[(part*t.groups+groupid)*8],asq,isq,bad?1.0f:0.0f);Sync();
DataCopy(red,state,t.group*t.groups*8);LocalSync<HardEvent::MTE2_V>();
for(uint32_t s=32;s>0;s>>=1){if(t.group>s){uint32_t rem=t.group-s<s?t.group-s:s;Add(red,red,red[s*t.groups*8],rem*t.groups*8);PipeBarrier<PIPE_V>();}}
LocalSync<HardEvent::V_S>();
float needs=0,unbounded=0;bool accept=false,myOk=false,sensitive=false;
for(uint32_t g=0;g<t.groups;g++){
uint32_t idx8=g*8;float aa=red.GetValue(idx8),ii=red.GetValue(idx8+1),bb=red.GetValue(idx8+2);
if constexpr(N==256){if(!(ii*1.4210854715202004e-14f<6.4e-9f))sensitive=true;}
float cut2=t.atol*t.atol>aa*t.rtol*t.rtol?t.atol*t.atol:aa*t.rtol*t.rtol;
bool gAccept=bb==0 && cut2*ii<0.25f,gBounded=bb==0 && cut2*ii<25.0f;
if(!gAccept)needs+=1.0f;
if(!gBounded)unbounded+=1.0f;
if(g==groupid){accept=gAccept||gBounded;myOk=(bb==0);}
}
if(sensitive){accept=false;needs=0;}
if(needs>0){
if(unbounded>0){
float ap=PowerNorm(batch,false),ip=PowerNorm(batch,true);
float spectralCut=t.atol*t.atol>ap*t.rtol*t.rtol*4.0f?t.atol*t.atol:ap*t.rtol*t.rtol*4.0f;
if(myOk && spectralCut*ip*4.0f<25.0f)accept=true;
}
RefineInverse(batch);
}
if(!accept)failed=true;
if(t.batch<=t.groups && unbounded==0 && !sensitive && outputReady)return true;
if(!failed && !outputReady){
if constexpr(sizeof(T)==4){
auto outputIndices=mat[8000].template ReinterpretCast<uint32_t>();
auto columnIndices=mat[12096].template ReinterpretCast<uint32_t>();uint32_t count=t.pp*8;
for(uint32_t c=0;c<8;c++)outputIndices.SetValue(c,c*t.pp*4);
LocalSync<HardEvent::S_V>();
for(uint32_t step=8;step<count;step*=2){Adds(outputIndices[step].template ReinterpretCast<int32_t>(),outputIndices.template ReinterpretCast<int32_t>(),int32_t(step/8*4),step);PipeBarrier<PIPE_V>();}
for(uint32_t c=part*8;c<t.p;c+=t.group*8){
uint32_t ac=t.p-c<8?t.p-c:8;DataCopy(mat,invall[c*t.pp],ac*t.pp);LocalSync<HardEvent::MTE2_V>();
Gather(rm,mat,outputIndices,0,t.pp*8);LocalSync<HardEvent::V_MTE3>();
DataCopyPad(output[base+c],rm,{(uint16_t)t.p,ac*4,0,(t.p-ac)*4,0});Fence();
}
}else{
for(uint32_t c=part;c<t.p;c+=t.group){DataCopy(tmp,invall[c*t.pp],t.pp);Fence();
Cast(raw,tmp,RoundMode::CAST_RINT,t.p);Fence();auto bits=raw.template ReinterpretCast<uint16_t>(),packed=rm.template ReinterpretCast<uint16_t>();
for(uint32_t i=0;i<t.p;i++)packed.SetValue(i*16,bits.GetValue(i));LocalSync<HardEvent::S_MTE3>();
DataCopyPad(output[base+c],rm.template ReinterpretCast<T>(),{(uint16_t)t.p,2,0,(t.p-1)*2,0});Fence();
}
}
}Sync();
}
Store(state[GetBlockIdx()*8],failed?1.0f:0.0f,0,0);Sync();DataCopy(red,state,t.group*t.groups*8);LocalSync<HardEvent::MTE2_S>();float any=0;for(uint32_t k=0;k<t.group*t.groups;k++){any+=red.GetValue(k*8);}Sync();return any==0;
}
};

template<class T,uint32_t N>
__aicore__ __attribute__((noinline,aligned(128))) bool SpecializedInverse(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData t){
if constexpr(sizeof(T)==4 && N==199){Independent199Inverse<T,N,true> fast;fast.Init(ai,ao,ws,t);return fast.Run();}
else{ParallelInverse<T,N> fast;fast.Init(ai,ao,ws,t);return fast.Run();}
}
template<class T>
__aicore__ __attribute__((noinline,aligned(128))) bool WideInPlaceInverse(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData t){
Independent199Inverse<T,199,false> fast;fast.Init(ai,ao,ws,t);return fast.Run();
}
template<class T> class ParallelJacobi {
TPipe pipe;TBuf<QuePosition::VECCALC> buf;LinalgPinvTilingData t;
GlobalTensor<T> input,output;GlobalTensor<float> all,work,vg,ns,state,yall,y;
LocalTensor<float> mem,u,v,a,b,tmp,tmp2,red,vj,vk,cj,ck,out,converted;
LocalTensor<T> raw;LocalTensor<uint32_t> offsets,bfOffsets;
uint32_t part,groupid,batch,base;bool inputCached;
__aicore__ inline void Fence(){PipeBarrier<PIPE_ALL>();}
__aicore__ inline void Sync(){Fence();if(t.group*t.groups>1)SyncAll();Fence();}
__aicore__ inline float AbsScalar(float z){return z<0?-z:z;}
__aicore__ inline float Root(float z){red.SetValue(0,z>0?z:0);LocalSync<HardEvent::S_V>();Sqrt(red,red,1);LocalSync<HardEvent::V_S>();return red.GetValue(0);}
__aicore__ inline float Dot(LocalTensor<float> x,LocalTensor<float> y,uint32_t count){Mul(tmp,x,y,count);PipeBarrier<PIPE_V>();ReduceSum(red,tmp,tmp2,count);LocalSync<HardEvent::V_S>();return red.GetValue(0);}
__aicore__ inline void Store(GlobalTensor<float> dst,float value){red.SetValue(0,value);LocalSync<HardEvent::S_MTE3>();DataCopyPad(dst,red,{1,4,0,0,0});LocalSync<HardEvent::MTE3_S>();}
__aicore__ inline float Load(GlobalTensor<float> src){DataCopy(red,src,8);LocalSync<HardEvent::MTE2_S>();return red.GetValue(0);}
__aicore__ inline void ReadInput(uint32_t col,LocalTensor<float> dst){
if(inputCached){DataCopy(dst,all[batch*t.p*(t.qp+t.pp)+col*t.qp],t.qp);Fence();return;}
Duplicate(dst,0.0f,t.qp);Fence();
if(t.m<t.n){
if constexpr(sizeof(T)==4)DataCopyPad(dst,input[base+col*t.q],{1,t.q*4,0,0,0},{true,0,(uint8_t)(t.qp-t.q),0});
else{DataCopyPad(raw,input[base+col*t.q],{1,t.q*2,0,0,0},{false,0,0,0});Fence();Cast(dst,raw,RoundMode::CAST_NONE,t.q);}
}else{
if constexpr(sizeof(T)==4){DataCopyPad(raw,input[base+col],{(uint16_t)t.q,4,(t.p-1)*4,0,0},{false,0,0,0});Fence();Gather(dst,raw.template ReinterpretCast<float>(),offsets,0,t.q);}
else for(uint32_t start=0;start<t.q;start+=32){uint32_t count=t.q-start<32?t.q-start:32;DataCopyPad(raw,input[base+col+start*t.p],{(uint16_t)count,2,(t.p-1)*2,0,0},{false,0,0,0});Fence();Cast(converted,raw,RoundMode::CAST_NONE,count*16);Fence();Gather(dst[start],converted,bfOffsets,0,count);Fence();}
}Fence();
}
__aicore__ inline void Read(GlobalTensor<float> src,LocalTensor<float> dst,uint32_t len){DataCopy(dst,src,len);Fence();}
__aicore__ inline void Write(GlobalTensor<float> dst,LocalTensor<float> src,uint32_t len){Fence();DataCopy(dst,src,len);Fence();}
__aicore__ inline bool Pair(uint32_t j,uint32_t k,float tiny,bool precise=false){
Read(work[j*t.qp],u,t.qp);Read(work[k*t.qp],v,t.qp);
float aa=Dot(u,u,t.qp),bb=Dot(v,v,t.qp),ab=precise?PreciseDot(u,v,t.qp):Dot(u,v,t.qp);
if(aa<=0 || bb<=0 || aa+bb<tiny)return false;
if(AbsScalar(ab)<=2e-7f*Root(aa)*Root(bb))return false;
float tau=(bb-aa)/(2*ab),tt;
if(AbsScalar(tau)>1e8f)tt=0.5f/tau;
else tt=(tau<0?-1.0f:1.0f)/(AbsScalar(tau)+Root(1+tau*tau));
float c=1.0f/Root(1+tt*tt),s=tt*c;
Muls(a,u,c,t.qp);Muls(tmp,v,s,t.qp);PipeBarrier<PIPE_V>();Sub(a,a,tmp,t.qp);
Muls(b,v,c,t.qp);Muls(tmp,u,s,t.qp);PipeBarrier<PIPE_V>();Add(b,b,tmp,t.qp);Fence();
float na=Dot(a,a,t.qp),nb=Dot(b,b,t.qp);
Write(work[j*t.qp],a,t.qp);Write(work[k*t.qp],b,t.qp);
Read(vg[j*t.pp],vj,t.pp);Read(vg[k*t.pp],vk,t.pp);
Muls(cj,vj,c,t.pp);Muls(tmp,vk,s,t.pp);PipeBarrier<PIPE_V>();Sub(cj,cj,tmp,t.pp);
Muls(ck,vk,c,t.pp);Muls(tmp,vj,s,t.pp);PipeBarrier<PIPE_V>();Add(ck,ck,tmp,t.pp);Fence();
Write(vg[j*t.pp],cj,t.pp);Write(vg[k*t.pp],ck,t.pp);Store(ns[j*8],na);Store(ns[k*8],nb);
return true;
}
__aicore__ inline float PreciseDot(LocalTensor<float> av,LocalTensor<float> xv,uint32_t len){
auto sums=mem[18944],zz=mem[19968],errors=mem[19456],low=mem[17920];
auto ah=mem[15360],al=mem[15872],bh=mem[16384],bl=mem[16896],pp=mem[18432];
uint32_t count=1;while(count<len)count*=2;
Duplicate(tmp,0.0f,count);Duplicate(low,0.0f,count);PipeBarrier<PIPE_V>();Mul(tmp,av,xv,len);PipeBarrier<PIPE_V>();
Muls(ah,av,4097.0f,len);Muls(bh,xv,4097.0f,len);PipeBarrier<PIPE_V>();
Sub(al,ah,av,len);Sub(bl,bh,xv,len);PipeBarrier<PIPE_V>();Sub(ah,ah,al,len);Sub(bh,bh,bl,len);PipeBarrier<PIPE_V>();
Sub(al,av,ah,len);Sub(bl,xv,bh,len);PipeBarrier<PIPE_V>();Mul(low,ah,bh,len);PipeBarrier<PIPE_V>();Sub(low,low,tmp,len);PipeBarrier<PIPE_V>();
Mul(pp,ah,bl,len);PipeBarrier<PIPE_V>();Add(low,low,pp,len);PipeBarrier<PIPE_V>();Mul(pp,al,bh,len);PipeBarrier<PIPE_V>();Add(low,low,pp,len);PipeBarrier<PIPE_V>();
Mul(pp,al,bl,len);PipeBarrier<PIPE_V>();Add(low,low,pp,len);PipeBarrier<PIPE_V>();
for(uint32_t h=count/2;h>=8;h/=2){
Add(sums,tmp,tmp[h],h);PipeBarrier<PIPE_V>();Sub(zz,sums,tmp,h);PipeBarrier<PIPE_V>();Sub(errors,tmp[h],zz,h);PipeBarrier<PIPE_V>();
Sub(zz,sums,zz,h);PipeBarrier<PIPE_V>();Sub(zz,tmp,zz,h);PipeBarrier<PIPE_V>();Add(errors,errors,zz,h);PipeBarrier<PIPE_V>();
Add(low,low,low[h],h);PipeBarrier<PIPE_V>();Add(low,low,errors,h);Adds(tmp,sums,0.0f,h);PipeBarrier<PIPE_V>();
}
LocalSync<HardEvent::V_S>();volatile float sum=0.0f,err=0.0f;
for(uint32_t i=0;i<8;i++){volatile float a=tmp.GetValue(i),ns=sum+a,z=ns-sum,q=ns-z,e0=sum-q,e1=a-z;err=err+e0;err=err+e1;err=err+low.GetValue(i);sum=ns;}
return sum+err;
}
__aicore__ inline void Rebuild(){
auto compensation=mem[7168];
for(uint32_t col=part;col<t.p;col+=t.group){
Read(vg[col*t.pp],vj,t.pp);Duplicate(out,0.0f,t.qp);Duplicate(compensation,0.0f,t.qp);Fence();
for(uint32_t k=0;k<t.p;k++){
ReadInput(k,u);Muls(a,u,vj.GetValue(k),t.qp);PipeBarrier<PIPE_V>();Sub(a,a,compensation,t.qp);PipeBarrier<PIPE_V>();
Add(b,out,a,t.qp);PipeBarrier<PIPE_V>();Sub(compensation,b,out,t.qp);PipeBarrier<PIPE_V>();Sub(compensation,compensation,a,t.qp);Adds(out,b,0.0f,t.qp);PipeBarrier<PIPE_V>();
}
float norm=Dot(out,out,t.qp);Write(work[col*t.qp],out,t.qp);Store(ns[col*8],norm);
}
Sync();uint32_t circle=(t.p+1)/2*2-1,pairs=(t.p+1)/2;
float mx=Load(state[t.group*t.groups*8+groupid*8]),tiny=0.01f*(t.atol*t.atol>mx*t.rtol*t.rtol?t.atol*t.atol:mx*t.rtol*t.rtol);
for(uint32_t round=0;round<circle;round++){
for(uint32_t pair=part;pair<pairs;pair+=t.group){uint32_t j=(round+pair)%circle,k=pair==0?circle:(round+circle-pair)%circle;if(j<t.p && k<t.p)Pair(j,k,tiny,true);}
Sync();
}
if(part==0){float largest=0;for(uint32_t col=0;col<t.p;col++){float z=Load(ns[col*8]);if(z>largest)largest=z;}Store(state[t.group*t.groups*8+groupid*8],largest);}
Sync();
}
__aicore__ inline void OutputRow(uint32_t row,float threshold2){
Duplicate(out,0.0f,t.qp);Fence();
for(uint32_t k=0;k<t.p;k++){
float norm=Load(ns[k*8]);if(!(norm>threshold2))continue;
Read(work[k*t.qp],u,t.qp);Read(vg[k*t.pp],v,t.pp);
float factor=v.GetValue(row)/norm;Axpy(out,u,factor,t.qp);Fence();
}
Write(y[row*t.qp],out,t.qp);
}
__aicore__ inline void Refine(){
auto rr=mem[6656],correction=mem[7168];
for(uint32_t row=part;row<t.p;row+=t.group){
Read(y[row*t.qp],u,t.qp);Duplicate(rr,0.0f,t.pp);Fence();
for(uint32_t col=0;col<t.p;col++){ReadInput(col,v);float z=(col==row?1.0f:0.0f)-PreciseDot(u,v,t.qp);rr.SetValue(col,z);}Fence();
Write(vg[row*t.pp],rr,t.pp);
}
Sync();
for(uint32_t row=part;row<t.p;row+=t.group){
Read(vg[row*t.pp],rr,t.pp);Duplicate(out,0.0f,t.qp);Duplicate(correction,0.0f,t.qp);Fence();
for(uint32_t col=0;col<t.p;col++){
Read(y[col*t.qp],u,t.qp);Muls(a,u,rr.GetValue(col),t.qp);PipeBarrier<PIPE_V>();Sub(a,a,correction,t.qp);PipeBarrier<PIPE_V>();
Add(b,out,a,t.qp);PipeBarrier<PIPE_V>();Sub(correction,b,out,t.qp);PipeBarrier<PIPE_V>();Sub(correction,correction,a,t.qp);Adds(out,b,0.0f,t.qp);PipeBarrier<PIPE_V>();
}
Read(y[row*t.qp],u,t.qp);Add(out,out,u,t.qp);Fence();Write(work[row*t.qp],out,t.qp);
}
Sync();
uint32_t rows=t.m>=t.n?t.p:t.q;
for(uint32_t row=part;row<rows;row+=t.group){
uint32_t count=t.m>=t.n?t.q:t.p;
if(t.m>=t.n)Read(work[row*t.qp],out,t.qp);
else{auto packed=raw.template ReinterpretCast<float>();DataCopyPad(packed,work[row],{(uint16_t)t.p,4,(t.qp-1)*4,0,0},{false,0,0,0});Fence();Gather(out,packed,offsets,0,t.p);Fence();}
uint32_t at=base+row*count;
if constexpr(sizeof(T)==4)DataCopyPad(output[at],out,{1,count*4,0,0,0});
else{Cast(raw,out,RoundMode::CAST_RINT,count);Fence();DataCopyPad(output[at],raw,{1,count*2,0,0,0});}Fence();
}
}
public:
__aicore__ inline void Init(GM_ADDR ai,GM_ADDR ao,GM_ADDR ws,LinalgPinvTilingData tt){
t=tt;inputCached=false;part=GetBlockIdx()%t.group;groupid=GetBlockIdx()/t.group;
input.SetGlobalBuffer((__gm__ T*)ai);output.SetGlobalBuffer((__gm__ T*)ao);all.SetGlobalBuffer((__gm__ float*)ws);
yall=all[t.batch*t.p*(t.qp+t.pp)];ns=yall[t.batch*t.p*t.qp];state=ns[t.batch*t.p*8];
pipe.InitBuffer(buf,194560);mem=buf.Get<float>();u=mem;v=mem[512];a=mem[1024];b=mem[1536];tmp=mem[2048];tmp2=mem[2560];red=mem[3072];vj=mem[3584];vk=mem[4096];cj=mem[4608];ck=mem[5120];out=mem[5632];raw=mem.template ReinterpretCast<T>()[8192*4/sizeof(T)];converted=mem[12288];offsets=mem.template ReinterpretCast<uint32_t>()[14336];bfOffsets=mem.template ReinterpretCast<uint32_t>()[14848];
for(uint32_t i=0;i<t.q;i++)offsets.SetValue(i,i*32);for(uint32_t i=0;i<32;i++)bfOffsets.SetValue(i,i*64);Fence();
}
__aicore__ inline void FormGram(GM_ADDR gram){
GlobalTensor<float> gg;gg.SetGlobalBuffer((__gm__ float*)gram);auto cache=mem[24576];uint32_t chunk=16384/t.qp;if(chunk>64)chunk=64;chunk=chunk/8*8;
for(batch=groupid;batch<t.batch;batch+=t.groups){
base=batch*t.m*t.n;work=all[batch*t.p*(t.qp+t.pp)];inputCached=false;
for(uint32_t c=part;c<t.p;c+=t.group){ReadInput(c,u);Write(work[c*t.qp],u,t.qp);}Sync();inputCached=true;
for(uint32_t cc=0;cc<t.p;cc+=chunk){uint32_t kc=t.p-cc<chunk?t.p-cc:chunk;DataCopy(cache,work[cc*t.qp],kc*t.qp);Fence();
for(uint32_t row=part;row<t.p;row+=t.group){ReadInput(row,u);for(uint32_t c=0;c<kc;c++)out.SetValue(c,Dot(u,cache[c*t.qp],t.qp));LocalSync<HardEvent::S_MTE3>();DataCopyPad(gg[(batch*t.p+row)*t.p+cc],out,{1,kc*4,0,0,0});Fence();}
}Sync();
}
}
__aicore__ inline void GramOutput(GM_ADDR inverse){
GlobalTensor<float> ig;ig.SetGlobalBuffer((__gm__ float*)inverse);inputCached=true;
for(batch=groupid;batch<t.batch;batch+=t.groups){
base=batch*t.m*t.n;work=all[batch*t.p*(t.qp+t.pp)];vg=work[t.p*t.qp];y=yall[batch*t.p*t.qp];
for(uint32_t row=part;row<t.p;row+=t.group){
DataCopyPad(vj,ig[(batch*t.p+row)*t.p],{1,t.p*4,0,0,0},{true,0,(uint8_t)(t.pp-t.p),0});Fence();Duplicate(out,0.0f,t.qp);Fence();
for(uint32_t col=0;col<t.p;col++){ReadInput(col,u);Axpy(out,u,vj.GetValue(col),t.qp);Fence();}
Write(y[row*t.qp],out,t.qp);
}Sync();
Refine();Sync();
}
}
__aicore__ inline void Run(){
auto normsAll=ns;uint32_t blocks=t.group*t.groups;
for(batch=groupid;batch<t.batch;batch+=t.groups){
base=batch*t.m*t.n;work=all[batch*t.p*(t.qp+t.pp)];vg=work[t.p*t.qp];ns=normsAll[batch*t.p*8];y=yall[batch*t.p*t.qp];
for(uint32_t col=part;col<t.p;col+=t.group){ReadInput(col,u);float norm=Dot(u,u,t.qp);Write(work[col*t.qp],u,t.qp);Duplicate(v,0.0f,t.pp);Fence();v.SetValue(col,1.0f);Fence();Write(vg[col*t.pp],v,t.pp);Store(ns[col*8],norm);}
Sync();
if(part==0){float mx=0;for(uint32_t col=0;col<t.p;col++){float z=Load(ns[col*8]);if(z>mx)mx=z;}Store(state[blocks*8+groupid*8],mx);}
Sync();uint32_t circle=(t.p+1)/2*2-1,pairs=(t.p+1)/2;
for(uint32_t sweep=0;sweep<20;sweep++){
float mx=Load(state[blocks*8+groupid*8]),rt=t.rtol*t.rtol;
float tiny=0.01f*(t.atol*t.atol>mx*rt?t.atol*t.atol:mx*rt);bool changed=false;
for(uint32_t round=0;round<circle;round++){
for(uint32_t pair=part;pair<pairs;pair+=t.group){uint32_t j=(round+pair)%circle,k=pair==0?circle:(round+circle-pair)%circle;if(j<t.p && k<t.p && Pair(j,k,tiny))changed=true;}
Sync();
}
Store(state[GetBlockIdx()*8],changed?1.0f:0.0f);Sync();
if(part==0){float largest=0;for(uint32_t col=0;col<t.p;col++){float z=Load(ns[col*8]);if(z>largest)largest=z;}Store(state[blocks*8+groupid*8],largest);}
if(GetBlockIdx()==0){float any=0;for(uint32_t i=0;i<blocks;i++)any+=Load(state[i*8]);Store(state[(blocks+t.groups)*8],any);}
Sync();if(Load(state[(blocks+t.groups)*8])==0)break;
}
Rebuild();float mx=Load(state[blocks*8+groupid*8]);float threshold2=t.atol*t.atol>mx*t.rtol*t.rtol?t.atol*t.atol:mx*t.rtol*t.rtol;
for(uint32_t row=part;row<t.p;row+=t.group)OutputRow(row,threshold2);
Sync();Refine();Sync();
}
}
};
// One row or column has a single singular value: ||A||_2.
template<class T> __aicore__ inline void VectorPinv(GM_ADDR ai,GM_ADDR ao,LinalgPinvTilingData t){
InitSocState();
GlobalTensor<T> input,output;input.SetGlobalBuffer((__gm__ T*)ai);output.SetGlobalBuffer((__gm__ T*)ao);
LocalTensor<float> mem(TPosition::VECCALC,0,2048);auto a=mem;auto sq=mem[512];auto red=mem[1024];auto scratch=mem[1088];auto raw=mem[1536].template ReinterpretCast<T>();
for(uint32_t batch=GetBlockIdx();batch<t.batch;batch+=GetBlockNum()){
uint32_t base=batch*t.q;
if(t.q==1){
DataCopyPad(raw,input[base],{1,(uint32_t)sizeof(T),0,0,0},{false,0,0,0});
float x;
if constexpr(sizeof(T)==4){LocalSync<HardEvent::MTE2_S>();x=raw.GetValue(0);}
else{LocalSync<HardEvent::MTE2_V>();Cast(a,raw,RoundMode::CAST_NONE,1);LocalSync<HardEvent::V_S>();x=a.GetValue(0);}
float ax=x<0?-x:x;
float threshold=t.atol>ax*t.rtol?t.atol:ax*t.rtol;
float y=ax>threshold?1.0f/x:0.0f;
if constexpr(sizeof(T)==4){raw.SetValue(0,y);LocalSync<HardEvent::S_MTE3>();}
else{a.SetValue(0,y);LocalSync<HardEvent::S_V>();Cast(raw,a,RoundMode::CAST_RINT,1);LocalSync<HardEvent::V_MTE3>();}
DataCopyPad(output[base],raw,{1,(uint32_t)sizeof(T),0,0,0});LocalSync<HardEvent::MTE3_MTE2>();
continue;
}
if constexpr(sizeof(T)==4){DataCopyPad(a,input[base],{1,t.q*4,0,0,0},{false,0,0,0});LocalSync<HardEvent::MTE2_V>();}
else{DataCopyPad(raw,input[base],{1,t.q*2,0,0,0},{false,0,0,0});LocalSync<HardEvent::MTE2_V>();Cast(a,raw,RoundMode::CAST_NONE,t.q);PipeBarrier<PIPE_V>();}
Mul(sq,a,a,t.q);PipeBarrier<PIPE_V>();
uint32_t full=t.q/64,tail=t.q%64;
if(full)WholeReduceSum(scratch,sq,64,full,1,1,8);
if(tail)WholeReduceSum(scratch[full],sq[full*64],tail,1,1,1,8);
PipeBarrier<PIPE_V>();WholeReduceSum(red,scratch,full+(tail?1:0),1,1,1,8);LocalSync<HardEvent::V_S>();
float norm=red.GetValue(0),scale=(norm>t.atol*t.atol && t.rtol<1.0f)?1.0f/norm:0.0f;
Muls(a,a,scale,t.q);
if constexpr(sizeof(T)==4){LocalSync<HardEvent::V_MTE3>();DataCopyPad(output[base],a,{1,t.q*4,0,0,0});}
else{PipeBarrier<PIPE_V>();Cast(raw,a,RoundMode::CAST_RINT,t.q);LocalSync<HardEvent::V_MTE3>();DataCopyPad(output[base],raw,{1,t.q*2,0,0,0});}
LocalSync<HardEvent::MTE3_MTE2>();
}
}
template<class T,bool SINGLE>
__aicore__ __attribute__((noinline)) void CachedFallback(GM_ADDR A,GM_ADDR out,GM_ADDR user,LinalgPinvTilingData t){
Pinv<T> op;op.Init(A,out,user,t);
if constexpr(SINGLE)op.Run(false,0);else op.Run();
}
template<class T> __aicore__ __attribute__((noinline,aligned(128))) bool PrepareDeferred256(GM_ADDR A,GM_ADDR out,GM_ADDR ws,LinalgPinvTilingData t,bool& need){DeferredInverse256<float,256> fast;fast.Init(A,out,ws,t);bool okay=fast.Run();need=fast.NeedsRefine();return okay;}
template<class T>
__aicore__ inline void CubeInverse256(GM_ADDR A,GM_ADDR out,GM_ADDR ws,LinalgPinvTilingData t,GM_ADDR systemWs,AscendC::tiling::TCubeTiling cubeTiling,bool& okay){
if ASCEND_IS_AIC {SetHF32Mode(false);SetHF32TransMode(false);}
bool need=false;okay=true;if ASCEND_IS_AIV {okay=PrepareDeferred256<T>(A,out,ws,t,need);}
ParallelInverse<float,256> fast;FP32Cube mm;
REGIST_MATMUL_OBJ(fast.Pipe(),systemWs,mm,&cubeTiling);
if ASCEND_IS_AIV {if(okay && need){fast.Init(A,out,ws,t);fast.SetCube(&mm);fast.RefineOnly();}}
}
template<class T>
__aicore__ inline void CubePinv256(GM_ADDR A,GM_ADDR out,GM_ADDR ws,LinalgPinvTilingData t,GM_ADDR systemWs,AscendC::tiling::TCubeTiling cubeTiling){
bool okay=true;CubeInverse256<T>(A,out,ws,t,systemWs,cubeTiling,okay);
if ASCEND_IS_AIV {if(!okay){ParallelJacobi<float> fallback;fallback.Init(A,out,ws,t);fallback.Run();}}
}
template<typename DT_A,bool LARGE,uint32_t WIDTH>
__global__ __aicore__ void linalg_pinv(GM_ADDR A,GM_ADDR A_pinv,GM_ADDR workspace,GM_ADDR tiling){
REGISTER_TILING_DEFAULT(LinalgPinvCubeTilingData);
REGISTER_TILING_FOR_TILINGKEY("LARGE != 1 || WIDTH != 2",LinalgPinvTilingData);
GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData,t,tiling);
if constexpr(LARGE && WIDTH==3){auto user=GetUserWorkspace(workspace);bool ok=t.p==199?SpecializedInverse<DT_A,199>(A,A_pinv,user,t):WideInPlaceInverse<DT_A>(A,A_pinv,user,t);if(!ok){ParallelJacobi<DT_A> fallback;fallback.Init(A,A_pinv,user,t);fallback.Run();}return;}
if constexpr(LARGE && WIDTH==2){GET_TILING_DATA_WITH_STRUCT(LinalgPinvCubeTilingData,ct,tiling);CubePinv256<DT_A>(A,A_pinv,GetUserWorkspace(workspace),t,GetSysWorkSpacePtr(),ct.cube);return;}
if constexpr(LARGE){
auto user=GetUserWorkspace(workspace);bool ok=false;
if(t.m==t.n){
if constexpr(sizeof(DT_A)==4){
if(t.p==199)ok=SpecializedInverse<DT_A,199>(A,A_pinv,user,t);
else if(t.p==256)ok=SpecializedInverse<DT_A,256>(A,A_pinv,user,t);
else ok=SpecializedInverse<DT_A,0>(A,A_pinv,user,t);
}else ok=SpecializedInverse<DT_A,0>(A,A_pinv,user,t);
}
else{
uint64_t oldSize=(uint64_t)t.batch*t.p*(2*t.qp+t.pp+8)*4+(uint64_t)(t.group*t.groups+t.groups+1)*32;
uint64_t fastSize=(uint64_t)t.groups*2*t.cols*(t.pp+8)*4+(uint64_t)t.batch*t.pp*t.pp*4+(uint64_t)(t.group*t.groups+1)*32;
uint64_t offset=((oldSize>fastSize?oldSize:fastSize)+255)/256*256;
GM_ADDR gram=user+offset;GM_ADDR inv=gram+(uint64_t)t.batch*t.p*t.p*4;GM_ADDR scratch=inv+(uint64_t)t.batch*t.p*t.p*4;
{ParallelJacobi<DT_A> pre;pre.Init(A,A_pinv,user,t);pre.FormGram(gram);}
auto gt=t;gt.m=t.p;gt.n=t.p;gt.q=t.p;gt.qp=t.pp;gt.atol=t.atol*t.atol;gt.rtol=t.rtol*t.rtol;if(gt.rtol<1e-6f)gt.rtol=1e-6f;
{ParallelInverse<float> fast;fast.Init(gram,inv,scratch,gt);ok=fast.Run();}
if(ok){ParallelJacobi<DT_A> post;post.Init(A,A_pinv,user,t);post.GramOutput(inv);}
}
if(!ok){ParallelJacobi<DT_A> op;op.Init(A,A_pinv,user,t);op.Run();}
}
else{
if constexpr(WIDTH<=1){if(t.p==1){VectorPinv<DT_A>(A,A_pinv,t);return;}}
if constexpr(WIDTH>64){
constexpr uint32_t order=WIDTH-64;
if(GetBlockIdx()>=t.batch*4){SyncAll();return;}
auto user=GetUserWorkspace(workspace);
if(!SmallInverse<DT_A,order,4>(A,A_pinv,user,t) && GetBlockIdx()%4==0){auto one=t;one.batch=1;uint64_t offset=uint64_t(GetBlockIdx()/4)*order*order*sizeof(DT_A);CachedFallback<DT_A,true>(A+offset,A_pinv+offset,user,one);}
return;
}else if constexpr(WIDTH>0 && sizeof(DT_A)==4){if(GetBlockIdx()>=t.batch)return;
if constexpr(WIDTH<=16){if(AlignedTinyInverse<DT_A,WIDTH,1>(A,A_pinv,nullptr,t))return;}
else{if(SmallInverse<DT_A,WIDTH,1>(A,A_pinv,nullptr,t))return;}
}else if constexpr(WIDTH>0){TinyPinv<DT_A,WIDTH> fast;fast.Init(A,A_pinv,t);if(fast.Run())return;}
CachedFallback<DT_A,false>(A,A_pinv,GetUserWorkspace(workspace),t);
}
}
