#if defined(EIGH_SOLVER_REPEAT)
// Packaging aliases: scoped to this repeated class body only.
#ifndef Z0
#define Z0 uint32_t
#else
#error "packaging alias collision: Z0"
#endif
#ifndef Z1
#define Z1 float
#else
#error "packaging alias collision: Z1"
#endif
#ifndef Z2
#define Z2 c_first
#else
#error "packaging alias collision: Z2"
#endif
#ifndef Z3
#define Z3 PipeBarrier
#else
#error "packaging alias collision: Z3"
#endif
#ifndef Z4
#define Z4 LocalTensor
#else
#error "packaging alias collision: Z4"
#endif
#ifndef Z5
#define Z5 static_cast
#else
#error "packaging alias collision: Z5"
#endif
#ifndef Z6
#define Z6 BinaryRepeatParams
#else
#error "packaging alias collision: Z6"
#endif
#ifndef Z7
#define Z7 ScalarAbs
#else
#error "packaging alias collision: Z7"
#endif
#ifndef Z8
#define Z8 SetValue
#else
#error "packaging alias collision: Z8"
#endif
#ifndef Z9
#define Z9 stride
#else
#error "packaging alias collision: Z9"
#endif
#ifndef ZA
#define ZA uint8_t
#else
#error "packaging alias collision: ZA"
#endif
#ifndef ZB
#define ZB row_count
#else
#error "packaging alias collision: ZB"
#endif
#ifndef ZC
#define ZC PIPE_V
#else
#error "packaging alias collision: ZC"
#endif
#ifndef ZD
#define ZD act_rows
#else
#error "packaging alias collision: ZD"
#endif
#ifndef ZE
#define ZE sc_rep_stride
#else
#error "packaging alias collision: ZE"
#endif
#ifndef ZF
#define ZF GetValue
#else
#error "packaging alias collision: ZF"
#endif
#ifndef ZG
#define ZG int32_t
#else
#error "packaging alias collision: ZG"
#endif
#ifndef ZH
#define ZH ub_mat
#else
#error "packaging alias collision: ZH"
#endif
#ifndef ZI
#define ZI ScalarRsqrt
#else
#error "packaging alias collision: ZI"
#endif
#ifndef ZJ
#define ZJ row_start
#else
#error "packaging alias collision: ZJ"
#endif
#if EIGH_K96_PASS
template<typename DT_A,Z0 SOLVE_MODE=0u,bool FAST200=false>
class KernelLinalgEighK96{
#else
template<typename DT_A,Z0 SOLVE_MODE=0u,bool FAST200=false>
class KernelLinalgEigh{
#endif
public:
#if EIGH_K96_PASS
__aicore__ inline KernelLinalgEighK96(){}
#else
__aicore__ inline KernelLinalgEigh(){}
#endif

__aicore__ inline void Init(
GM_ADDR A,
GM_ADDR eigenvalues,
GM_ADDR eigenvectors,
GM_ADDR workspace,
const LinalgEighTilingData&t,
#if EIGH_K96_PASS
TPipe*pipe,Z4<Z1>sharedBase){
#else
TPipe*pipe){
#endif
n=t.n;
Z9=t.Z9;
uplo=t.uplo;
batch_size=t.batch_size;
num_groups=t.num_groups;
group_lanes=t.group_lanes;
rows_per_lane=t.rows_per_lane;
vec_rows_per_lane=t.vec_rows_per_lane;
solve_mode=t.solve_mode;

Z0 blk_idx=GetBlockIdx();
if(num_groups==1u){
lane_id=blk_idx;
group_id=0u;
}else{
lane_id=blk_idx%group_lanes;
group_id=blk_idx/group_lanes;
}

if constexpr(SOLVE_MODE==1u){
vec_row_start=lane_id*vec_rows_per_lane;
vec_row_count=(vec_row_start<n)
?(((n-vec_row_start)<vec_rows_per_lane)?(n-vec_row_start):vec_rows_per_lane)
:0u;
}else if(n==256u&&group_lanes==10u){
if(lane_id==0u){
vec_row_start=0u;
vec_row_count=24u;
}else if(lane_id<9u){
vec_row_start=24u+(lane_id-1u)*26u;
vec_row_count=26u;
}else if(lane_id==9u){
vec_row_start=232u;
vec_row_count=24u;
}else{
vec_row_start=n;
vec_row_count=0u;
}
}else{
Z0 base_vecs=n/group_lanes;
Z0 extra_vecs=n%group_lanes;
Z0 extra_start=(group_lanes-extra_vecs)>>1u;
Z0 extra_end=extra_start+extra_vecs;
if(lane_id<group_lanes){
Z0 extra_before=(lane_id<=extra_start)?0u:((lane_id<extra_end)?(lane_id-extra_start):extra_vecs);
Z0 has_extra=(lane_id>=extra_start&&lane_id<extra_end)?1u:0u;
vec_row_start=lane_id*base_vecs+extra_before;
vec_row_count=(vec_row_start<n)?(base_vecs+has_extra):0u;
}else{
vec_row_start=n;
vec_row_count=0u;
}
}

if(solve_mode==0u){

Z0 num_blocks=(n+7u)/8u;
Z0 base_blocks=num_blocks/group_lanes;
Z0 extra_blocks=num_blocks%group_lanes;
Z0 start_block=lane_id*base_blocks+((lane_id<extra_blocks)?lane_id:extra_blocks);
my_blocks=base_blocks+((lane_id<extra_blocks)?1u:0u);
coop_row_start=start_block*8u;
Z0 coop_end=coop_row_start+my_blocks*8u;
if(coop_end>n){
coop_end=n;
}
coop_row_count=(coop_row_start<n)?(coop_end-coop_row_start):0u;
}else{
my_blocks=0u;
coop_row_start=0u;
coop_row_count=n;
}

ZJ=vec_row_start;
ZB=vec_row_count;

gm_A.SetGlobalBuffer(reinterpret_cast<__gm__ Z1*>(A));
gm_eigvals.SetGlobalBuffer(reinterpret_cast<__gm__ Z1*>(eigenvalues));
gm_eigvecs.SetGlobalBuffer(reinterpret_cast<__gm__ Z1*>(eigenvectors));
#if EIGH_K96_PASS
gm_ckpt.SetGlobalBuffer(reinterpret_cast<__gm__ Z1*>(workspace),4u*65536u);
#else
#endif

if constexpr(SOLVE_MODE==1u){
pipe->InitBuffer(ub_buf,(2u*rows_per_lane*Z9+18u*Z9)*sizeof(Z1));
Z4<Z1>base=ub_buf.Get<Z1>();
ZH=base;
ub_vecs=base[rows_per_lane*Z9];
Z4<Z1>scratch_base=base[2u*rows_per_lane*Z9];
ub_row=scratch_base;
ub_gather=scratch_base[2u*Z9];
return;
}

if constexpr(SOLVE_MODE==2u){
Z0 plane_rows=3u*rows_per_lane;
pipe->InitBuffer(ub_buf,(plane_rows*Z9+29u*Z9+192u)*sizeof(Z1));
Z4<Z1>base=ub_buf.Get<Z1>();
ZH=base;
ub_vecs=base[rows_per_lane*Z9];
ub_refl=base[2u*rows_per_lane*Z9];

Z4<Z1>scratch_base=base[plane_rows*Z9];
ub_row=scratch_base;
ub_v=scratch_base[Z9];
ub_w=scratch_base[2u*Z9];
ub_cur_row=scratch_base[3u*Z9];
ub_prod=scratch_base[5u*Z9];
ub_tmp=scratch_base[6u*Z9];
ub_red=scratch_base[7u*Z9];
ub_diag=scratch_base[8u*Z9];
ub_off=scratch_base[9u*Z9];
ub_tau=scratch_base[10u*Z9];
ub_eig=scratch_base[11u*Z9];
ub_comm=scratch_base[12u*Z9];
ub_gather=scratch_base[13u*Z9];

Z4<Z1>mask_base=scratch_base[29u*Z9];
ub_mask_ge=mask_base;
ub_mask_gt=mask_base[64];
ub_mask_eq=mask_base[128];
for(Z0 rem=0;rem<8u;++rem){
for(Z0 c=0;c<8u;++c){
ub_mask_ge.Z8(rem*8u+c,(c>=rem)?1.0f:0.0f);
ub_mask_gt.Z8(rem*8u+c,(c>rem)?1.0f:0.0f);
ub_mask_eq.Z8(rem*8u+c,(c==rem)?1.0f:0.0f);
}
}
return;
}

if constexpr(SOLVE_MODE==3u){
if(n<=200u){

Z0 refl_rows=(49152u-1280u)/Z9-n-rows_per_lane-8u;
vecs_offset=(n+4u+refl_rows)*Z9;
Z0 total_floats=(n==200u)?49152u:((vecs_offset+(rows_per_lane+4u)*Z9)+1280u);
pipe->InitBuffer(ub_buf,total_floats*sizeof(Z1));

Z4<Z1>base=ub_buf.Get<Z1>();
ZH=base;
ub_w=base[n*Z9];
ub_diag=base[(n+1u)*Z9];
ub_off=base[(n+2u)*Z9];
ub_tau=base[(n+3u)*Z9];
ub_refl=base[(n+4u)*Z9];
ub_vecs=base[vecs_offset];

Z4<Z1>s1d=base[(n==200u)?47072u:(vecs_offset+rows_per_lane*Z9)];
ub_prod=s1d;
ub_tmp=s1d[Z9];
ub_gather=s1d[2u*Z9];  
ub_row=ub_gather[1024];      
ub_v=ub_row[Z9];         
ub_eig=ub_v;
ub_red=ub_v[Z9];         

Z4<Z1>mask_base=ub_red[64];  
ub_mask_ge=mask_base;
ub_mask_gt=mask_base[64];
ub_mask_eq=mask_base[128];
}else{

num_zones=(n+31u)>>5u;
Z0 cur_z_off=0u;
for(Z0 z=0;z<num_zones;++z){
zone_offset[z]=cur_z_off;
cur_z_off+=32u*(Z9-(z<<5u));
}
zone_offset[num_zones]=cur_z_off;

vecs_offset=cur_z_off+14u*Z9;
#if EIGH_K96_PASS
// The stage is finished and its MTE3 writes completed before this reuse.
Z4<Z1>base=sharedBase;
#else
Z0 total_floats=49152u;
pipe->InitBuffer(ub_buf,total_floats*sizeof(Z1));

Z4<Z1>base=ub_buf.Get<Z1>();
#endif
ZH=base;
ub_w=base[cur_z_off];
ub_diag=base[cur_z_off+Z9];
ub_off=base[cur_z_off+2u*Z9];
ub_tau=base[cur_z_off+3u*Z9];
ub_refl=base[cur_z_off+4u*Z9];  
ub_vecs=base[vecs_offset];              

Z4<Z1>s1d=base[vecs_offset+rows_per_lane*Z9];
ub_prod=s1d;
ub_tmp=s1d[Z9];
ub_gather=s1d[2u*Z9];  
ub_row=ub_gather[512];       
ub_v=ub_gather[1024];
ub_eig=ub_v;
ub_red=ub_v[Z9];         

#if EIGH_K96_PASS
// n256 K96 continuation skips LoadAndSymmetrizeLarge entirely: Stage96
// already reconstructed the selected triangle. Its otherwise-needed gt
// mask is dead for this entry, so eq owns the former gt 64-float region.
#else
#endif
Z4<Z1>mask_base=base[48960];
#if EIGH_K96_PASS
ub_mask_gt=mask_base; // unused on the specialized n256 K96 continuation
ub_mask_eq=mask_base;
#else
ub_mask_gt=mask_base;
#endif
ub_mask_ge=mask_base[64];
#if EIGH_K96_PASS
#else
ub_mask_eq=mask_base[128];
#endif
}
for(Z0 rem=0;rem<8u;++rem){
for(Z0 c=0;c<8u;++c){
ub_mask_ge.Z8(rem*8u+c,(c>=rem)?1.0f:0.0f);
#if EIGH_K96_PASS
if(n<=200u) ub_mask_gt.Z8(rem*8u+c,(c>rem)?1.0f:0.0f);
#else
ub_mask_gt.Z8(rem*8u+c,(c>rem)?1.0f:0.0f);
#endif
ub_mask_eq.Z8(rem*8u+c,(c==rem)?1.0f:0.0f);
}
}
return;
}

gm_ws.SetGlobalBuffer(reinterpret_cast<__gm__ Z1*>(workspace));
uint64_t group_ws_span=2u*Z5<uint64_t>(n)*Z9+7u*Z9+8u*group_lanes+64u;
ws_refl_offset=Z5<uint64_t>(group_id)*group_ws_span;
ws_w_offset=ws_refl_offset+Z5<uint64_t>(n)*Z9;
ws_scale_offset=ws_w_offset+6u*Z9;
ws_tail_offset=ws_scale_offset+8u*group_lanes+Z9+64u;

Z0 plane_rows=2u*rows_per_lane;
pipe->InitBuffer(ub_buf,(plane_rows*Z9+29u*Z9+192u)*sizeof(Z1));

Z4<Z1>base=ub_buf.Get<Z1>();
ZH=base;
ub_vecs=base[rows_per_lane*Z9];
ub_refl=base[2u*rows_per_lane*Z9];

Z4<Z1>scratch_base=base[plane_rows*Z9];
ub_row=scratch_base;
ub_v=scratch_base[Z9];
ub_w=scratch_base[2u*Z9];
ub_cur_row=scratch_base[3u*Z9];
ub_prod=scratch_base[5u*Z9];
ub_tmp=scratch_base[6u*Z9];
ub_red=scratch_base[7u*Z9];
ub_diag=scratch_base[8u*Z9];
ub_off=scratch_base[9u*Z9];
ub_tau=scratch_base[10u*Z9];
ub_gather=scratch_base[11u*Z9];
ub_eig=scratch_base[27u*Z9];
ub_comm=scratch_base[28u*Z9];

Z4<Z1>mask_base=scratch_base[29u*Z9];
ub_mask_ge=mask_base;
ub_mask_gt=mask_base[64];
ub_mask_eq=mask_base[128];

for(Z0 rem=0;rem<8u;++rem){
for(Z0 c=0;c<8u;++c){
ub_mask_ge.Z8(rem*8u+c,(c>=rem)?1.0f:0.0f);
ub_mask_gt.Z8(rem*8u+c,(c>rem)?1.0f:0.0f);
ub_mask_eq.Z8(rem*8u+c,(c==rem)?1.0f:0.0f);
}
}
}

__aicore__ inline void Process(){
if constexpr(SOLVE_MODE==1u){
Z1 work_small[1792];
ZA swap_small[32];
for(uint64_t b=group_id;b<batch_size;b+=num_groups){
SolveSmallBatch(b,work_small,swap_small);
}
return;
}else if constexpr(SOLVE_MODE==2u){
ProcessMediumM2();
return;
}else if constexpr(SOLVE_MODE==3u){
ProcessLargeM2();
return;
}else{
Z1 work[10*512];
ZA swap_mask[512];
for(uint64_t wave=0u;wave*num_groups<batch_size;++wave){
uint64_t b=wave*num_groups+group_id;
is_active=(b<batch_size);
ZJ=coop_row_start;
ZB=coop_row_count;
LoadAndSymmetrizeCoop(b);
TridiagonalizeCoop();
if(is_active){
ZJ=vec_row_start;
ZB=vec_row_count;
ComputeEigenvalues(work);
if(n==121&&lane_id==0){}ComputeInverseVectors(work,swap_mask);
BacktransformCoop(work);
WriteOutputs(b);
}
if((wave+1u)*num_groups<batch_size){
Z3<PIPE_ALL>();
SyncAll<true>();
}
}
}
}

private:
__aicore__ __attribute__((noinline))void ProcessMediumM2(){
Z1 work[4096];
ZA swap_mask[512];
for(uint64_t b=group_id;b<batch_size;b+=num_groups){
ZJ=0u;
ZB=n;
ZA*tau_mask=swap_mask+256u;
LoadAndSymmetrizeMedium(b);
TridiagonalizeMedium(work,tau_mask);
ZJ=vec_row_start;
ZB=vec_row_count;
ComputeEigenvalues(work);
ComputeInverseVectors(work,swap_mask);
BacktransformMedium(tau_mask);
WriteOutputs(b);
}
}

#if EIGH_K96_PASS

// CPU-only experiment. Workspace holds four 65536-float checkpoint records.
// Header: compact tail 160^2, 96 full reflectors, d/e/tau, then scale.
__aicore__ inline void LoadK96Checkpoint(uint64_t b,Z1*work,ZA*tau){
 const uint64_t base=b*65536ull;
 for(Z0 off=0;off<25600u;off+=4096u){
  const Z0 count=(25600u-off<4096u)?25600u-off:4096u;
  DataCopy(ZH[15360u+off],gm_ckpt[base+off],count);
 }
 DataCopy(ZH,gm_ckpt[base+25600u+24u*256u],2048u);
 DataCopyPad(ZH[2048u],gm_ckpt[base+25600u+32u*256u+32u],
             DataCopyExtParams{32u,896u,128u,0u,0u},
             DataCopyPadExtParams<Z1>{false,0u,0u,0.0f});
 DataCopyPad(ZH[9216u],gm_ckpt[base+25600u+64u*256u+64u],
             DataCopyExtParams{32u,768u,256u,0u,0u},
             DataCopyPadExtParams<Z1>{false,0u,0u,0.0f});
 DataCopy(ub_gather,gm_ckpt[base+50176u],296u);
 fence<HardEvent::MTE2_S>();
 for(Z0 k=0;k<96u;++k){
  work[k]=ub_gather.ZF(k); work[256u+k]=ub_gather.ZF(96u+k);
  tau[k]=Z5<ZA>(Z5<ZG>(ub_gather.ZF(192u+k)));
 }
 matrix_scale=ub_gather.ZF(288u);
}

#else
#endif
__aicore__ __attribute__((noinline))void ProcessLargeM2(){
Z1 work[4096];
ZA swap_mask[512];
for(uint64_t b=group_id;b<batch_size;b+=num_groups){
ZJ=0u;
ZB=n;
ZA*tau_mask=swap_mask+256u;
#if EIGH_K96_PASS
LoadK96Checkpoint(b,work,tau_mask);
// Init writes the continuation masks on S; the skipped parent's input phase
// used to provide this ordering before any vector instruction consumes them.
fence<HardEvent::S_V>();
#else
LoadAndSymmetrizeLarge(b);
#endif
TridiagonalizeLarge(b,work,tau_mask);
ZJ=vec_row_start;
ZB=vec_row_count;
SolveTridiagSturmLarge(work);
ComputeInverseVectorsLarge(work,swap_mask);
BacktransformLarge(b,tau_mask);
WriteOutputsLarge(b);
}
}

template<HardEvent event>
__aicore__ inline void fence()const{
const ZG id=Z5<ZG>(GetTPipePtr()->FetchEventID(event));
SetFlag<event>(id);
WaitFlag<event>(id);
}

__aicore__ inline Z1 Z7(Z1 x)const{
#if EIGH_K96_PASS
return __builtin_fabsf(x);
#else
if constexpr(SOLVE_MODE==3u){return __builtin_fabsf(x);}else{return(x<0.0f)?-x:x;}
#endif
}


__aicore__ inline Z1 ZI(Z1 x)const{
if(x<=0.0f){
return 0.0f;
}
union{
Z1 f;
Z0 u;
}bits;
bits.f=x;
bits.u=0x5f375a86u-(bits.u>>1);
Z1 y=bits.f;
Z1 xhalf=0.5f*x;
y=y*(1.5f-xhalf*y*y);
y=y*(1.5f-xhalf*y*y);
y=y*(1.5f-xhalf*y*y);
return y;
}


__aicore__ inline Z1 ScalarRsqrtFast(Z1 x)const{
if(x<=0.0f){
return 0.0f;
}
union{
Z1 f;
Z0 u;
}bits;
bits.f=x;
bits.u=0x5f375a86u-(bits.u>>1);
Z1 y=bits.f;
Z1 xhalf=0.5f*x;
y=y*(1.5f-xhalf*y*y);
y=y*(1.5f-xhalf*y*y);
return y;
}


__aicore__ inline Z1 ScalarSqrt(Z1 x)const{
return x*ZI(x);
}

__aicore__ inline void ComputePow2Scale(Z1 max_val,Z1&scale_out,Z1&inv_scale_out)const{
union{
Z1 f;
Z0 u;
}sbits;
sbits.f=max_val;
Z0 exp_bits=sbits.u&0x7f800000u;
if(exp_bits==0u||exp_bits==0x7f800000u){
scale_out=1.0f;
inv_scale_out=1.0f;
}else{
sbits.u=exp_bits;
scale_out=sbits.f;
sbits.u=0x7f000000u-exp_bits;
inv_scale_out=sbits.f;
}
}

__aicore__ inline Z1 ClampPivot(Z1 x)const{
return(Z7(x)<1e-15f)?((x<0.0f)?-1e-15f:1e-15f):x;
}


__aicore__ inline void FactorizeShiftedTridiag1Way(
const Z1*dd,
const Z1*ee,
Z1 lambda,
Z1*piv,
Z1*mult,
Z1*u1,
ZA*swap_out)const{
Z1 cur_p=dd[0]-lambda;Z1 cur_u=ee[0];
for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i];Z1 nd=dd[i+1u]-lambda;
Z1 ne=ee[i+1u];bool sw=(Z7(cur_p)<Z7(m_i));
Z1 den=sw?m_i:ClampPivot(cur_p);Z1 inv_p=1.0f/den;
Z1 num=sw?cur_p:m_i;Z1 f=num*inv_p;
piv[i]=inv_p;mult[i]=f;
u1[i]=sw?nd:cur_u;swap_out[i]=sw?1u:0u;
cur_p=(sw?cur_u:nd)-f*(sw?nd:cur_u);cur_u=sw?(-f*ne):ne;
}
piv[n-1u]=1.0f/ClampPivot(cur_p);}


__aicore__ inline Z1 SolveShiftedTridiag1Way(
Z1*qv,
Z1 in_scale,
const Z1*ee,
const Z1*piv,
const Z1*mult,
const Z1*u1,
const ZA*swap_in)const{
Z1 cq=qv[0]*in_scale;for(Z0 i=0;i+1u<n;++i){
ZA sw=swap_in[i];Z1 nq=qv[i+1u]*in_scale;
Z1 qi=(sw!=0u)?nq:cq;Z1 base_q=(sw!=0u)?cq:nq;
qv[i]=qi;cq=base_q-mult[i]*qi;
}
Z1 q_n1=cq*piv[n-1u];qv[n-1u]=q_n1;
Z1 nsq=q_n1*q_n1;Z1 q_n2=0.0f;
if(n>1u){
Z0 nm2=n-2u;q_n2=q_n1;
q_n1=(qv[nm2]-u1[nm2]*q_n2)*piv[nm2];qv[nm2]=q_n1;
nsq+=q_n1*q_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);Z1 u2_val=(swap_in[ui]!=0u)?ee[ui+1u]:0.0f;
Z1 q_cur=(qv[ui]-u1[ui]*q_n1-u2_val*q_n2)*piv[ui];qv[ui]=q_cur;
q_n2=q_n1;q_n1=q_cur;
nsq+=q_cur*q_cur;}
return nsq;}


__aicore__ inline void EvalSturmSmall4(
const Z1*d,
const Z1*e2,
const Z1*b_mid,
Z0*b_neg,
Z1*b_pval)const{
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 d0=d[0];Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;
Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;
Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;Z0 s0=(p0<0.0f)?1u:0u,c0=s0;
Z0 s1=(p1<0.0f)?1u:0u,c1=s1;Z0 s2=(p2<0.0f)?1u:0u,c2=s2;
Z0 s3=(p3<0.0f)?1u:0u,c3=s3;Z0 i=1u;
if((n&1u)==0u){
Z1 d_1=d[1],e2_0=e2[0];i=2u;
pp0=p0;pp1=p1;pp2=p2;pp3=p3;p0=(d_1-m0)*pp0-e2_0;
p1=(d_1-m1)*pp1-e2_0;p2=(d_1-m2)*pp2-e2_0;
p3=(d_1-m3)*pp3-e2_0;Z0 sn0=(p0<=0.0f)?1u:0u;c0+=(sn0^s0);s0=sn0;
Z0 sn1=(p1<=0.0f)?1u:0u;c1+=(sn1^s1);s1=sn1;Z0 sn2=(p2<=0.0f)?1u:0u;c2+=(sn2^s2);s2=sn2;
Z0 sn3=(p3<=0.0f)?1u:0u;c3+=(sn3^s3);s3=sn3;}
for(;i+3u<n;i+=4u){
{
Z1 d_a=d[i],e2_a=e2[i-1u],d_b=d[i+1u],e2_b=e2[i];Z1 ta0=d_a-m0,ta1=d_a-m1,ta2=d_a-m2,ta3=d_a-m3;
Z1 tb0=d_b-m0,tb1=d_b-m1,tb2=d_b-m2,tb3=d_b-m3;pp0=ta0*p0-e2_a*pp0;
pp1=ta1*p1-e2_a*pp1;pp2=ta2*p2-e2_a*pp2;
pp3=ta3*p3-e2_a*pp3;p0=tb0*pp0-e2_b*p0;
p1=tb1*pp1-e2_b*p1;p2=tb2*pp2-e2_b*p2;
p3=tb3*pp3-e2_b*p3;Z0 sa0=(pp0<=0.0f)?1u:0u;c0+=(sa0^s0);
Z0 sa1=(pp1<=0.0f)?1u:0u;c1+=(sa1^s1);Z0 sa2=(pp2<=0.0f)?1u:0u;c2+=(sa2^s2);
Z0 sa3=(pp3<=0.0f)?1u:0u;c3+=(sa3^s3);Z0 sb0=(p0<=0.0f)?1u:0u;c0+=(sb0^sa0);s0=sb0;
Z0 sb1=(p1<=0.0f)?1u:0u;c1+=(sb1^sa1);s1=sb1;Z0 sb2=(p2<=0.0f)?1u:0u;c2+=(sb2^sa2);s2=sb2;
Z0 sb3=(p3<=0.0f)?1u:0u;c3+=(sb3^sa3);s3=sb3;}
{
Z1 d_a=d[i+2u],e2_a=e2[i+1u],d_b=d[i+3u],e2_b=e2[i+2u];
Z1 ta0=d_a-m0,ta1=d_a-m1,ta2=d_a-m2,ta3=d_a-m3;Z1 tb0=d_b-m0,tb1=d_b-m1,tb2=d_b-m2,tb3=d_b-m3;
pp0=ta0*p0-e2_a*pp0;pp1=ta1*p1-e2_a*pp1;
pp2=ta2*p2-e2_a*pp2;pp3=ta3*p3-e2_a*pp3;
p0=tb0*pp0-e2_b*p0;p1=tb1*pp1-e2_b*p1;
p2=tb2*pp2-e2_b*p2;p3=tb3*pp3-e2_b*p3;
Z0 sa0=(pp0<=0.0f)?1u:0u;c0+=(sa0^s0);Z0 sa1=(pp1<=0.0f)?1u:0u;c1+=(sa1^s1);
Z0 sa2=(pp2<=0.0f)?1u:0u;c2+=(sa2^s2);Z0 sa3=(pp3<=0.0f)?1u:0u;c3+=(sa3^s3);
Z0 sb0=(p0<=0.0f)?1u:0u;c0+=(sb0^sa0);s0=sb0;Z0 sb1=(p1<=0.0f)?1u:0u;c1+=(sb1^sa1);s1=sb1;
Z0 sb2=(p2<=0.0f)?1u:0u;c2+=(sb2^sa2);s2=sb2;Z0 sb3=(p3<=0.0f)?1u:0u;c3+=(sb3^sa3);s3=sb3;
}
}
for(;i<n;i+=2u){
Z1 d_a=d[i],e2_a=e2[i-1u],d_b=d[i+1u],e2_b=e2[i];Z1 ta0=d_a-m0,ta1=d_a-m1,ta2=d_a-m2,ta3=d_a-m3;
Z1 tb0=d_b-m0,tb1=d_b-m1,tb2=d_b-m2,tb3=d_b-m3;pp0=ta0*p0-e2_a*pp0;
pp1=ta1*p1-e2_a*pp1;pp2=ta2*p2-e2_a*pp2;
pp3=ta3*p3-e2_a*pp3;p0=tb0*pp0-e2_b*p0;
p1=tb1*pp1-e2_b*p1;p2=tb2*pp2-e2_b*p2;
p3=tb3*pp3-e2_b*p3;Z0 sa0=(pp0<=0.0f)?1u:0u;c0+=(sa0^s0);
Z0 sa1=(pp1<=0.0f)?1u:0u;c1+=(sa1^s1);Z0 sa2=(pp2<=0.0f)?1u:0u;c2+=(sa2^s2);
Z0 sa3=(pp3<=0.0f)?1u:0u;c3+=(sa3^s3);Z0 sb0=(p0<=0.0f)?1u:0u;c0+=(sb0^sa0);s0=sb0;
Z0 sb1=(p1<=0.0f)?1u:0u;c1+=(sb1^sa1);s1=sb1;Z0 sb2=(p2<=0.0f)?1u:0u;c2+=(sb2^sa2);s2=sb2;
Z0 sb3=(p3<=0.0f)?1u:0u;c3+=(sb3^sa3);s3=sb3;}
b_neg[0]=c0;b_neg[1]=c1;b_neg[2]=c2;b_neg[3]=c3;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);}

__aicore__ inline void SolveSmallBatch(uint64_t b,Z1*work,ZA*swap_mask){
ZJ=vec_row_start;ZB=vec_row_count;
if(ZB==0u){
return;}
Z0 total_elems=n*n;ZA rpad=Z5<ZA>(((total_elems+7u)&~7u)-total_elems);
DataCopyPad(
ZH,
gm_A[b*total_elems],
DataCopyExtParams{1,total_elems*4u,0,0,0},
DataCopyPadExtParams<Z1>{true,0,rpad,0.0f});fence<HardEvent::MTE2_S>();

Z0 a_cap=(n<=16u)?512u:1024u;Z1*a_loc=work;                 
Z1*dd=work+a_cap;            
Z1*ee=dd+32;                 
Z1*tt=ee+32;                 
Z1*vv=tt+32;                 
Z1*ww=vv+32;                 
Z1*orig_d=ww+32;             
Z1*orig_e=orig_d+32;         
Z1*ee_sq=orig_e+32;          
Z1*aos=work+a_cap+256u;    

if(uplo!=0u){
for(Z0 i=0;i<n;++i){
Z0 ub_row_off=i*n;Z1*a_row=a_loc+i*32u;
for(Z0 j=i;j<n;++j){
a_row[j]=ZH.ZF(ub_row_off+j);}
}
}else{
for(Z0 j=0;j<n;++j){
Z0 ub_row_off=j*n;for(Z0 i=0;i<=j;++i){
a_loc[i*32u+j]=ZH.ZF(ub_row_off+i);}
}
}
Z1 p_a0=a_loc[0],p_a1=a_loc[1],p_a33=a_loc[33];Z1 probe_sq=p_a0*p_a0+p_a1*p_a1+p_a33*p_a33;
bool pre_scaled=false;if(!(probe_sq>=0.0625f&&probe_sq<=16.0f)){
Z1 max_val=0.0f;for(Z0 i=0;i<n;++i){
const Z1*a_row=a_loc+i*32u;for(Z0 j=i;j<n;++j){
Z1 av=Z7(a_row[j]);if(av>max_val)max_val=av;
}
}
if(max_val>0.0f&&(max_val<0.25f||max_val>4.0f)){
Z1 inv_scale=1.0f;ComputePow2Scale(max_val,matrix_scale,inv_scale);
for(Z0 i=0;i<n;++i){
Z1*a_row=a_loc+i*32u;for(Z0 j=i;j<n;++j){
a_row[j]*=inv_scale;}
}
pre_scaled=true;}else{
matrix_scale=1.0f;}
}else{
matrix_scale=1.0f;}

Z1 next_sq_sum=-1.0f;for(Z0 k=0;k<n;++k){
Z1*a_k=a_loc+k*32u;dd[k]=a_k[k];
ee[k]=0.0f;tt[k]=0.0f;
if(k+2u>=n){
if(k+1u<n){
ee[k]=a_k[k+1u];}
continue;}
Z1 sq_sum=next_sq_sum;if(sq_sum<0.0f){
sq_sum=0.0f;for(Z0 j=k+1u;j<n;++j){
Z1 v_j=a_k[j];sq_sum+=v_j*v_j;
}
}
Z1 norm=ScalarSqrt(sq_sum);if(norm==0.0f){
continue;}
Z1 x0=a_k[k+1u];Z1 beta=(x0>=0.0f)?-norm:norm;
ee[k]=beta;tt[k]=1.0f;
Z1 inv_s=ZI(sq_sum-beta*x0);Z1 v0=(x0-beta)*inv_s;
a_k[k+1u]=v0;

Z1 vw_dot=0.0f;{
const Z1*a_row0=a_loc+(k+1u)*32u;Z1 acc0=a_row0[k+1u]*v0;
for(Z0 j=k+2u;j<n;++j){
Z1 vj=a_k[j]*inv_s;a_k[j]=vj;
Z1 a_ij=a_row0[j];acc0+=a_ij*vj;
ww[j]=a_ij*v0;}
ww[k+1u]=acc0;vw_dot=v0*acc0;
}
Z0 i_mv=k+2u;for(;i_mv+1u<n;i_mv+=2u){
const Z1*a_r0=a_loc+i_mv*32u;const Z1*a_r1=a_r0+32u;
Z1 vi0=a_k[i_mv],vi1=a_k[i_mv+1u];Z1 a01=a_r0[i_mv+1u];
Z1 acc0=ww[i_mv]+a_r0[i_mv]*vi0+a01*vi1;Z1 acc1=ww[i_mv+1u]+a01*vi0+a_r1[i_mv+1u]*vi1;
for(Z0 j=i_mv+2u;j<n;++j){
Z1 vj=a_k[j];Z1 v0j=a_r0[j],v1j=a_r1[j];
acc0+=v0j*vj;acc1+=v1j*vj;
ww[j]+=v0j*vi0+v1j*vi1;}
ww[i_mv]=acc0;ww[i_mv+1u]=acc1;
vw_dot+=vi0*acc0+vi1*acc1;}
for(;i_mv<n;++i_mv){
const Z1*a_row=a_loc+i_mv*32u;Z1 vi=a_k[i_mv];
Z1 acc=ww[i_mv]+a_row[i_mv]*vi;for(Z0 j=i_mv+1u;j<n;++j){
Z1 a_ij=a_row[j];acc+=a_ij*a_k[j];
ww[j]+=a_ij*vi;}
ww[i_mv]=acc;vw_dot+=vi*acc;
}
Z1 alpha=-0.5f*vw_dot;{
Z1 v1=v0;Z1 w1=ww[k+1u]+alpha*v1;
ww[k+1u]=w1;Z1*a_row1=a_loc+(k+1u)*32u;
a_row1[k+1u]-=2.0f*v1*w1;Z1 nsq=0.0f;
for(Z0 j=k+2u;j<n;++j){
Z1 vj=a_k[j];Z1 wj=ww[j]+alpha*vj;
ww[j]=wj;Z1 val=a_row1[j]-(v1*wj+w1*vj);
a_row1[j]=val;nsq+=val*val;
}
next_sq_sum=nsq;}
Z0 i_r2=k+2u;for(;i_r2+1u<n;i_r2+=2u){
Z1*a_r0=a_loc+i_r2*32u;Z1*a_r1=a_r0+32u;
Z1 vi0=a_k[i_r2],wi0=ww[i_r2];Z1 vi1=a_k[i_r2+1u],wi1=ww[i_r2+1u];
a_r0[i_r2]-=2.0f*vi0*wi0;a_r0[i_r2+1u]-=(vi0*wi1+wi0*vi1);
a_r1[i_r2+1u]-=2.0f*vi1*wi1;for(Z0 j=i_r2+2u;j<n;++j){
Z1 vj=a_k[j],wj=ww[j];a_r0[j]-=(vi0*wj+wi0*vj);
a_r1[j]-=(vi1*wj+wi1*vj);}
}
for(;i_r2<n;++i_r2){
Z1*a_row=a_loc+i_r2*32u;Z1 vi=a_k[i_r2];
Z1 wi=ww[i_r2];for(Z0 j=i_r2;j<n;++j){
a_row[j]-=(vi*ww[j]+wi*a_k[j]);}
}
}

if(!pre_scaled){
Z1 max_val=0.0f;for(Z0 i=0;i<n;++i){
Z1 ad=Z7(dd[i]);if(ad>max_val)max_val=ad;
if(i+1u<n){
Z1 ae=Z7(ee[i]);if(ae>max_val)max_val=ae;
}
}
if(max_val>0.0f&&(max_val<0.125f||max_val>16.0f)){
Z1 inv_scale=1.0f;ComputePow2Scale(max_val,matrix_scale,inv_scale);
for(Z0 i=0;i<n;++i){
dd[i]*=inv_scale;ee[i]*=inv_scale;
}
}
}
bool is_diag=true;for(Z0 i=0;i<n;++i){
orig_d[i]=dd[i];orig_e[i]=ee[i];
ee_sq[i]=ee[i]*ee[i];if(i+1u<n&&ee[i]!=0.0f){
is_diag=false;}
}

if(!is_diag&&n<=18u&&ZB<=2u){
Z1 g_min=0.0f,g_max=0.0f;for(Z0 i=0;i<n;++i){
Z1 rad=((i>0u)?Z7(orig_e[i-1u]):0.0f)+((i+1u<n)?Z7(orig_e[i]):0.0f);
Z1 lo_i=orig_d[i]-rad,hi_i=orig_d[i]+rad;if(i==0u||lo_i<g_min)g_min=lo_i;
if(i==0u||hi_i>g_max)g_max=hi_i;}
Z1 pad=1e-5f*(Z7(g_min)+Z7(g_max))+1e-6f;g_min-=pad;g_max+=pad;

Z0 t0=ZJ+1u;Z0 t1=(ZB>1u)?(ZJ+2u):t0;
Z1 lo0=g_min,hi0=g_max,plo0=-1.0f,phi0=-1.0f;Z1 lo1=g_min,hi1=g_max,plo1=-1.0f,phi1=-1.0f;
Z0 lc0=0u,hc0=n,lc1=0u,hc1=n;

Z1 lo_p=g_min,hi_p_min=g_max,plo_p=-1.0f,phi_p_min=-1.0f,lo_n_min=g_max;Z0 lc_p=0u,hc_p_min=n;
bool check_prev=(ZJ>0u);bool prev_separated=!check_prev;
Z1 b_mid[4],b_pval[4];Z0 b_neg[4];
const Z0 min_sturm=(n==16u)?4u:5u;const Z0 max_sturm=(n==16u)?5u:6u;
for(Z0 iter=0;iter<8u;++iter){
if(iter>=min_sturm&&(hc0-lc0<=1u)){
if(iter>=max_sturm&&(lo0-hi_p_min>=0.0015f||prev_separated))break;if(check_prev&&(lo0-hi_p_min>=0.0015f)){
prev_separated=true;}
if(iter==min_sturm&&(lo0-hi_p_min<0.0015f)&&check_prev&&(hi_p_min-lo_p>0.0015f)){
Z1 s0=(hi0-lo0)*0.25f;b_mid[0]=lo0-0.0015f;
b_mid[1]=lo0+s0;b_mid[2]=lo0+2.0f*s0;
b_mid[3]=lo0+3.0f*s0;EvalSturmSmall4(orig_d,ee_sq,b_mid,b_neg,b_pval);
if(b_neg[0]>=ZJ){
prev_separated=true;hi_p_min=b_mid[0];phi_p_min=b_pval[0];hc_p_min=b_neg[0];
}else{
lo_p=b_mid[0];plo_p=b_pval[0];lc_p=b_neg[0];}
for(Z0 k=1u;k<4u;++k){
Z0 c=b_neg[k];if(c<t0){lo0=b_mid[k];plo0=b_pval[k];lc0=c;}
else{hi0=b_mid[k];phi0=b_pval[k];hc0=c;break;}
}
if(prev_separated)break;continue;
}
if(iter>=max_sturm)break;}
Z1 s0=(hi0-lo0)*0.2f;b_mid[0]=lo0+s0;b_mid[1]=lo0+2.0f*s0;b_mid[2]=lo0+3.0f*s0;b_mid[3]=lo0+4.0f*s0;
EvalSturmSmall4(orig_d,ee_sq,b_mid,b_neg,b_pval);
for(Z0 k=0;k<4u;++k){if(b_neg[k]>t0&&b_mid[k]<lo_n_min)lo_n_min=b_mid[k];}
if(iter==0u&&ZB>1u){
for(Z0 k=0;k<4u;++k){
Z0 c=b_neg[k];if(c<t1){lo1=b_mid[k];plo1=b_pval[k];lc1=c;}
else if(hi1==g_max){hi1=b_mid[k];phi1=b_pval[k];hc1=c;}
}
}
if(check_prev){
for(Z0 k=0;k<4u;++k){
Z0 c=b_neg[k];if(c<ZJ){
if(b_mid[k]>lo_p){lo_p=b_mid[k];plo_p=b_pval[k];lc_p=c;}
}else if(b_mid[k]<hi_p_min){
hi_p_min=b_mid[k];phi_p_min=b_pval[k];hc_p_min=c;}
}
}
for(Z0 k=0;k<4u;++k){
Z0 c=b_neg[k];if(c<t0){lo0=b_mid[k];plo0=b_pval[k];lc0=c;}
else{hi0=b_mid[k];phi0=b_pval[k];hc0=c;break;}
}
}
Z1 eig0=0.5f*(lo0+hi0);if(hc0-lc0==1u&&plo0>=0.0f&&phi0>=0.0f&&(plo0+phi0)>1e-30f){
eig0=lo0+(plo0/(plo0+phi0))*(hi0-lo0);}
dd[ZJ]=eig0;
orig_d[31]=((prev_separated||(eig0-hi_p_min)>=0.015f)&&(ZJ+1u>=n||(lo_n_min-eig0)>=0.015f))?1.0f:0.0f;
if(check_prev){
if(!prev_separated&&(eig0-hi_p_min)<0.0015f){
Z1 hi_p=hi_p_min,phi_p=phi_p_min;Z0 hc_p=hc_p_min;
for(Z0 iter=0u;iter<5u;++iter){
if(iter>=4u&&(hc_p-lc_p<=1u))break;Z1 sp=(hi_p-lo_p)*0.2f;
b_mid[0]=lo_p+sp;b_mid[1]=lo_p+2.0f*sp;b_mid[2]=lo_p+3.0f*sp;b_mid[3]=lo_p+4.0f*sp;
EvalSturmSmall4(orig_d,ee_sq,b_mid,b_neg,b_pval);for(Z0 k=0;k<4u;++k){
Z0 c=b_neg[k];if(c<ZJ){lo_p=b_mid[k];plo_p=b_pval[k];lc_p=c;}
else{hi_p=b_mid[k];phi_p=b_pval[k];hc_p=c;break;}
}
}
Z1 eig_p=0.5f*(lo_p+hi_p);if(hc_p-lc_p==1u&&plo_p>=0.0f&&phi_p>=0.0f&&(plo_p+phi_p)>1e-30f){
eig_p=lo_p+(plo_p/(plo_p+phi_p))*(hi_p-lo_p);}
dd[ZJ-1u]=eig_p;}else{
dd[ZJ-1u]=-1.0e30f;}
}
if(ZB>1u){
for(Z0 iter=0;iter<9u;++iter){
if(iter>=5u&&(hc1-lc1<=1u))break;Z1 s1=(hi1-lo1)*0.2f;
b_mid[0]=lo1+s1;b_mid[1]=lo1+2.0f*s1;b_mid[2]=lo1+3.0f*s1;b_mid[3]=lo1+4.0f*s1;
EvalSturmSmall4(orig_d,ee_sq,b_mid,b_neg,b_pval);for(Z0 k=0;k<4u;++k){
Z0 c=b_neg[k];if(c<t1){lo1=b_mid[k];plo1=b_pval[k];lc1=c;}
else{hi1=b_mid[k];phi1=b_pval[k];hc1=c;break;}
}
}
Z1 eig1=0.5f*(lo1+hi1);if(hc1-lc1==1u&&plo1>=0.0f&&phi1>=0.0f&&(plo1+phi1)>1e-30f){
eig1=lo1+(plo1/(plo1+phi1))*(hi1-lo1);}
dd[ZJ+1u]=eig1;}
}else{

const Z1 eps_sq=1.4210855e-14f;Z0 total_iters=0u;
for(Z0 l=0;l<n;){
Z0 m=l;for(;m+1u<n;++m){
if(ee_sq[m]<=eps_sq*Z7(dd[m]*dd[m+1u])+1e-38f){
break;}
}
if(m+1u<n){
ee_sq[m]=0.0f;}
if(m==l){
++l;continue;
}
if(m==l+1u){
Z1 mid=0.5f*(dd[l]+dd[l+1u]);Z1 hdiff=0.5f*(dd[l]-dd[l+1u]);
Z1 rad=ScalarSqrt(hdiff*hdiff+ee_sq[l]);dd[l]=mid-rad;
dd[l+1u]=mid+rad;ee_sq[l]=0.0f;
l+=2u;continue;
}
if(++total_iters>30u*n){
break;}
Z1 delta=0.5f*(dd[l+1u]-dd[l]);Z1 rad=ScalarSqrt(delta*delta+ee_sq[l]);
Z1 denom=delta+((delta>=0.0f)?rad:-rad);
Z1 sigma=(denom!=0.0f)?(dd[l]-ee_sq[l]/denom):(dd[l]-ScalarSqrt(ee_sq[l]));

Z1 c=1.0f;Z1 s=0.0f;
Z1 gamma=dd[m]-sigma;Z1 pp=gamma*gamma;
for(ZG ii=Z5<ZG>(m)-1;ii>=Z5<ZG>(l);--ii){
Z0 i=Z5<Z0>(ii);Z1 bb=ee_sq[i];
Z1 rr=pp+bb;if(i+1u!=m){
ee_sq[i+1u]=s*rr;}
Z1 old_c=c;Z1 old_gamma=gamma;
Z1 alpha=dd[i];Z1 inv_rr=1.0f/rr;
c=pp*inv_rr;s=bb*inv_rr;
gamma=c*(alpha-sigma)-s*old_gamma;dd[i+1u]=old_gamma+(alpha-gamma);
pp=(c!=0.0f)?((gamma*gamma)/c):(old_c*bb);}
ee_sq[l]=s*pp;dd[l]=sigma+gamma;
}

for(Z0 i=1u;i<n;++i){
Z1 val=dd[i];Z0 j=i;
while(j>0u&&dd[j-1u]>val){
dd[j]=dd[j-1u];--j;
}
dd[j]=val;}
}

if(ZB>1u&&!is_diag){
for(Z0 i=0;i+1u<n;++i){
Z1 inv_e=(orig_e[i]!=0.0f)?(1.0f/orig_e[i]):0.0f;ww[i]=inv_e;
ee_sq[i]=orig_e[i+1u]*inv_e;}
}
Z0 tile_stride=(ZB+7u)&~7u;for(Z0 r0=0;r0<ZB;r0+=4u){
Z0 bcnt=((ZB-r0)<4u)?(ZB-r0):4u;if(is_diag){
for(Z0 b=0;b<bcnt;++b){
Z0 r=r0+b;Z0 col_idx=ZJ+r;
ub_row.Z8(r,dd[col_idx]*matrix_scale);for(Z0 i=0;i<n;++i){
Z0 rank=0u;for(Z0 k=0;k<n;++k){
if(orig_d[k]<orig_d[i]||(orig_d[k]==orig_d[i]&&k<i)){
++rank;}
}
Z1 v_i=(rank==col_idx)?1.0f:0.0f;if(ZB==n)ub_vecs.Z8(i*n+col_idx,v_i);
else ub_gather.Z8(i*tile_stride+r,v_i);}
}
continue;}

Z0 c0=ZJ+r0;bool need_orth_prev=(bcnt==1u&&c0>0u&&Z7(dd[c0]-dd[c0-1u])<0.0015f);
if(need_orth_prev){
for(Z0 i=0;i+1u<n;++i){
ee_sq[i]=(orig_e[i]!=0.0f)?(orig_e[i+1u]/orig_e[i]):0.0f;}
}
if(bcnt==1u&&!need_orth_prev){
Z1 lam0=dd[c0];Z0 sm0=(c0*13u)%23u;
Z1 cq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 cur_p0=orig_d[0]-lam0,cur_u0=orig_e[0];for(Z0 i=0;i+1u<n;++i){
Z1 m_i=orig_e[i],abs_m=Z7(m_i),nd0=orig_d[i+1u]-lam0,e_next=orig_e[i+1u];
bool sw0=(Z7(cur_p0)<abs_m);swap_mask[i]=sw0?1u:0u;
Z1 inv0=1.0f/(sw0?m_i:ClampPivot(cur_p0));Z1 f0=(sw0?cur_p0:m_i)*inv0;
Z1 u1_0=sw0?nd0:cur_u0;Z1*s=aos+i*4u;
s[0]=inv0;s[1]=f0;
s[2]=u1_0*inv0;s[3]=sw0?(e_next*inv0):0.0f;
cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_u0=sw0?(-f0*e_next):e_next;
sm0+=7u;if(sm0>=23u)sm0-=23u;Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 qi0=sw0?nq0:cq0;Z1 b0=sw0?cq0:nq0;
vv[i]=qi0*inv0;cq0=b0-f0*qi0;
}
Z1 inv_last=1.0f/ClampPivot(cur_p0);Z1 in0=1.0f,s0_small=0.0f;
for(Z0 iter=0;iter<3u;++iter){
if(iter>0u){
cq0=vv[0]*in0;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];const Z1*s=aos+i*4u;
Z1 nq0=vv[i+1u]*in0;Z1 qi0=(sm!=0u)?nq0:cq0;
Z1 b0=(sm!=0u)?cq0:nq0;vv[i]=qi0*s[0];
cq0=b0-s[1]*qi0;}
}
Z1 q0_n1=cq0*inv_last;vv[n-1u]=q0_n1;
Z1 nsq0=q0_n1*q0_n1;Z1 q0_n2=0.0f;
if(n>1u){
const Z1*s_nm2=aos+(n-2u)*4u;q0_n2=q0_n1;
q0_n1=vv[n-2u]-s_nm2[2]*q0_n2;vv[n-2u]=q0_n1;
nsq0+=q0_n1*q0_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<ZG>(i);const Z1*s=aos+ui*4u;
Z1 c0_v=(vv[ui]-s[3]*q0_n2)-s[2]*q0_n1;vv[ui]=c0_v;
q0_n2=q0_n1;q0_n1=c0_v;nsq0+=c0_v*c0_v;
}
if(iter==0u)s0_small=nsq0;
if(iter==0u||(iter==1u&&((orig_d[31]==0.0f&&(n!=16u||nsq0<1.0e8f||s0_small*nsq0<1.0e15f))||((n==16u)?(s0_small<1.0e5f&&s0_small*nsq0<1.0e10f):(s0_small<5.0e6f&&s0_small*nsq0<1.0e13f))))){
in0=ScalarRsqrtFast(nsq0);}else{
in0=ZI(nsq0);Z1 rq0=orig_d[n-1u]*vv[n-1u]*vv[n-1u];
for(Z0 i=0;i+1u<n;++i){
Z1 vi=vv[i];rq0+=orig_d[i]*vi*vi+(2.0f*orig_e[i])*vi*vv[i+1u];
}
ub_row.Z8(r0,(rq0/nsq0)*matrix_scale);break;
}
}
ZG k_bt=Z5<ZG>(n)-3;for(;k_bt>=1;k_bt-=2){
Z0 uk=Z5<Z0>(k_bt);if(tt[uk]!=0.0f&&tt[uk-1u]!=0.0f){
const Z1*hk1=a_loc+uk*32u;const Z1*hk0=a_loc+(uk-1u)*32u;
Z1 d1=0.0f,d0=0.0f,ov=0.0f;for(Z0 i=uk+1u;i<n;++i){
Z1 v_i=vv[i],h1=hk1[i],h0=hk0[i];d1+=h1*v_i;
d0+=h0*v_i;ov+=h0*h1;
}
Z1 h0_uk=hk0[uk];d0=(d0+h0_uk*vv[uk])-d1*ov;
vv[uk]-=d0*h0_uk;for(Z0 i=uk+1u;i<n;++i){
vv[i]-=d1*hk1[i]+d0*hk0[i];}
}else{
for(ZG kk=k_bt;kk>=k_bt-1;--kk){
Z0 ukk=Z5<Z0>(kk);if(tt[ukk]!=0.0f){
const Z1*hk=a_loc+ukk*32u;Z1 d0=0.0f;
for(Z0 i=ukk+1u;i<n;++i)d0+=hk[i]*vv[i];for(Z0 i=ukk+1u;i<n;++i)vv[i]-=d0*hk[i];
}
}
}
}
for(;k_bt>=0;--k_bt){
Z0 uk=Z5<Z0>(k_bt);if(tt[uk]!=0.0f){
const Z1*hk=a_loc+uk*32u;Z1 d0=0.0f;
for(Z0 i=uk+1u;i<n;++i)d0+=hk[i]*vv[i];for(Z0 i=uk+1u;i<n;++i)vv[i]-=d0*hk[i];
}
}
Z0 mi0=0u;Z1 ma0=Z7(vv[0]);
for(Z0 i=1u;i<n;++i){
Z1 a0=Z7(vv[i]);if(a0>ma0){ma0=a0;mi0=i;}
}
Z1 sc0=(vv[mi0]<0.0f)?-in0:in0;if(ZB==n){
for(Z0 i=0;i<n;++i)ub_vecs.Z8(i*n+c0,vv[i]*sc0);}else{
for(Z0 i=0;i<n;++i)ub_gather.Z8(i*tile_stride+r0,vv[i]*sc0);}
continue;}
Z0 c0_eff=need_orth_prev?(c0-1u):c0;Z0 c1=need_orth_prev?c0:((bcnt>1u)?(c0+1u):c0);
if(bcnt<=2u){
Z0 c0=c0_eff;Z1 lam0=dd[c0],lam1=dd[c1];
Z0 sm0=(c0*13u)%23u;Z0 sm1=sm0+13u;if(sm1>=23u)sm1-=23u;
Z1 cq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 cq1=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm1));
Z1 cur_p0=orig_d[0]-lam0,cur_u0=orig_e[0];Z1 cur_p1=orig_d[0]-lam1,cur_u1=orig_e[0];
for(Z0 i=0;i+1u<n;++i){
Z1 m_i=orig_e[i],abs_m=Z7(m_i),d_next=orig_d[i+1u],e_next=orig_e[i+1u];
Z1 nd0=d_next-lam0,nd1=d_next-lam1;bool sw0=(Z7(cur_p0)<abs_m),sw1=(Z7(cur_p1)<abs_m);
swap_mask[i]=(sw0?1u:0u)|(sw1?2u:0u);Z1 inv_m=ww[i];
Z1 inv0=sw0?inv_m:(1.0f/ClampPivot(cur_p0));Z1 inv1=sw1?inv_m:(1.0f/ClampPivot(cur_p1));
Z1 f0=(sw0?cur_p0:m_i)*inv0,f1=(sw1?cur_p1:m_i)*inv1;Z1*s=aos+i*8u;
s[0]=inv0;s[1]=inv1;s[2]=f0;s[3]=f1;
Z1 u1_0=sw0?nd0:cur_u0,u1_1=sw1?nd1:cur_u1;s[4]=u1_0*inv0;s[5]=u1_1*inv1;
cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
sm0+=7u;if(sm0>=23u)sm0-=23u;sm1+=7u;if(sm1>=23u)sm1-=23u;
Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 nq1=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm1));
Z1 qi0=sw0?nq0:cq0,qi1=sw1?nq1:cq1;Z1 b0=sw0?cq0:nq0,b1=sw1?cq1:nq1;
s[6]=qi0*inv0;s[7]=qi1*inv1;cq0=b0-f0*qi0;cq1=b1-f1*qi1;
}
Z1*s_last=aos+(n-1u)*8u;s_last[0]=1.0f/ClampPivot(cur_p0);
s_last[1]=1.0f/ClampPivot(cur_p1);const Z0 max_iter_s=3u;
Z1 in0=1.0f,in1=1.0f;for(Z0 iter=0;iter<max_iter_s;++iter){
if(iter>0u){
cq0=aos[6]*in0;cq1=aos[7]*in1;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*8u;
Z1 nq0=s[14]*in0,nq1=s[15]*in1;Z1 qi0=(sm&1u)?nq0:cq0,qi1=(sm&2u)?nq1:cq1;
Z1 b0=(sm&1u)?cq0:nq0,b1=(sm&2u)?cq1:nq1;s[6]=qi0*s[0];s[7]=qi1*s[1];
cq0=b0-s[2]*qi0;cq1=b1-s[3]*qi1;}
}
Z1 q0_n1=cq0*s_last[0],q1_n1=cq1*s_last[1];s_last[6]=q0_n1;s_last[7]=q1_n1;
Z1 nsq0=q0_n1*q0_n1,nsq1=q1_n1*q1_n1;Z1 q0_n2=0.0f,q1_n2=0.0f;
if(n>1u){
Z1*s_nm2=aos+(n-2u)*8u;q0_n2=q0_n1;q1_n2=q1_n1;
q0_n1=s_nm2[6]-s_nm2[4]*q0_n2;q1_n1=s_nm2[7]-s_nm2[5]*q1_n2;
s_nm2[6]=q0_n1;s_nm2[7]=q1_n1;nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;
}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_sq[ui];Z1*s=aos+ui*8u;
Z1 c0_v=(s[6]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[4]*q0_n1;
Z1 c1_v=(s[7]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[5]*q1_n1;s[6]=c0_v;s[7]=c1_v;
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;}
if(iter+1u<max_iter_s){
in0=ScalarRsqrtFast(nsq0);in1=ScalarRsqrtFast(nsq1);
}else{
in0=ZI(nsq0);in1=ZI(nsq1);
if(bcnt==2u&&c0>0u&&Z7(lam0-dd[c0-1u])<0.008f){
Z1 lam_p=dd[c0-1u];Z0 smp=((c0-1u)*13u)%23u;
Z1 cqp=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(smp));
Z1 cur_pp=orig_d[0]-lam_p,cur_up=orig_e[0];ZA sw_p[32];
Z1*sp_aos=aos+256u;for(Z0 i=0;i+1u<n;++i){
Z1 m_i=orig_e[i],abs_m=Z7(m_i),ndp=orig_d[i+1u]-lam_p,e_next=orig_e[i+1u];
bool swp=(Z7(cur_pp)<abs_m);sw_p[i]=swp?1u:0u;
Z1 invp=swp?(1.0f/m_i):(1.0f/ClampPivot(cur_pp));Z1 fp=(swp?cur_pp:m_i)*invp;
Z1 u1_p=swp?ndp:cur_up;Z1*sp=sp_aos+i*4u;
sp[0]=invp;sp[1]=fp;sp[2]=u1_p*invp;sp[3]=swp?(e_next*invp):0.0f;cur_pp=(swp?cur_up:ndp)-fp*u1_p;
cur_up=swp?(-fp*e_next):e_next;smp+=7u;if(smp>=23u)smp-=23u;
Z1 nqp=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(smp));Z1 qip=swp?nqp:cqp,bp=swp?cqp:nqp;
vv[i]=qip*invp;cqp=bp-fp*qip;
}
Z1 inv_lp=1.0f/ClampPivot(cur_pp),inp=1.0f;for(Z0 it_p=0;it_p<3u;++it_p){
if(it_p>0u){
cqp=vv[0]*inp;for(Z0 i=0;i+1u<n;++i){
const Z1*sp=sp_aos+i*4u;Z1 nqp=vv[i+1u]*inp;
Z1 qip=(sw_p[i]!=0u)?nqp:cqp,bp=(sw_p[i]!=0u)?cqp:nqp;vv[i]=qip*sp[0];
cqp=bp-sp[1]*qip;}
}
Z1 qp_n1=cqp*inv_lp;vv[n-1u]=qp_n1;
Z1 nsqp=qp_n1*qp_n1,qp_n2=0.0f;if(n>1u){
const Z1*sp_nm2=sp_aos+(n-2u)*4u;qp_n2=qp_n1;
qp_n1=vv[n-2u]-sp_nm2[2]*qp_n2;vv[n-2u]=qp_n1;
nsqp+=qp_n1*qp_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);const Z1*sp=sp_aos+ui*4u;
Z1 cp_v=(vv[ui]-((sw_p[ui]!=0u)?ee_sq[ui]:0.0f)*qp_n2)-sp[2]*qp_n1;vv[ui]=cp_v;
qp_n2=qp_n1;qp_n1=cp_v;nsqp+=cp_v*cp_v;
}
inp=(it_p+1u<3u)?ScalarRsqrtFast(nsqp):ZI(nsqp);}
Z1 dot0=0.0f;for(Z0 i=0;i<n;++i)dot0+=(aos[i*8u+6u]*in0)*(vv[i]*inp);
nsq0=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+6u]*in0-dot0*(vv[i]*inp);aos[i*8u+6u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
if((bcnt>1u||need_orth_prev)&&Z7(lam1-lam0)<0.008f){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+7u]*in1)*(aos[i*8u+6u]*in0);
nsq1=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+7u]*in1-dot*(aos[i*8u+6u]*in0);aos[i*8u+7u]=v;
nsq1+=v*v;}
in1=ZI(nsq1);}
Z1 rq0=0.0f,rq1=0.0f;Z1 cur0=aos[6]*in0,cur1=aos[7]*in1;
aos[6]=cur0;aos[7]=cur1;for(Z0 i=0;i+1u<n;++i){
Z1*sn=aos+(i+1u)*8u;Z1 nxt0=sn[6]*in0,nxt1=sn[7]*in1;
sn[6]=nxt0;sn[7]=nxt1;Z1 d_i=orig_d[i],two_e=2.0f*orig_e[i];
rq0+=d_i*cur0*cur0+two_e*cur0*nxt0;rq1+=d_i*cur1*cur1+two_e*cur1*nxt1;
cur0=nxt0;cur1=nxt1;}
Z1 d_last=orig_d[n-1u];rq0+=d_last*cur0*cur0;
rq1+=d_last*cur1*cur1;ub_row.Z8(r0,(need_orth_prev?rq1:rq0)*matrix_scale);
if(bcnt>1u)ub_row.Z8(r0+1u,rq1*matrix_scale);}
}
for(ZG k=Z5<ZG>(n)-3;k>=0;--k){
Z0 uk=Z5<Z0>(k);if(tt[uk]!=0.0f){
const Z1*hk=a_loc+uk*32u;Z1 d0=0.0f,d1=0.0f;
for(Z0 i=uk+1u;i<n;++i){
Z1 v_i=hk[i];const Z1*s=aos+i*8u;
d0+=v_i*s[6];d1+=v_i*s[7];}
for(Z0 i=uk+1u;i<n;++i){
Z1 v_i=hk[i];Z1*s=aos+i*8u;
s[6]-=d0*v_i;s[7]-=d1*v_i;}
}
}
if(bcnt==2u){
Z1 nsq0=0.0f,nsq1=0.0f;Z0 mi0=0u,mi1=0u;
Z1 ma0=Z7(aos[6]),ma1=Z7(aos[7]);for(Z0 i=0;i<n;++i){
const Z1*s=aos+i*8u;Z1 v0=s[6],v1=s[7];
nsq0+=v0*v0;nsq1+=v1*v1;Z1 a0=Z7(v0),a1=Z7(v1);
if(a0>ma0){ma0=a0;mi0=i;}
if(a1>ma1){ma1=a1;mi1=i;}
}
Z1 sc0=((aos[mi0*8u+6u]<0.0f)?-1.0f:1.0f)*ZI(nsq0);
Z1 sc1=((aos[mi1*8u+7u]<0.0f)?-1.0f:1.0f)*ZI(nsq1);if(ZB==n){
for(Z0 i=0;i<n;++i){
const Z1*s=aos+i*8u;Z0 dst_off=i*n+c0;
ub_vecs.Z8(dst_off+0u,s[6]*sc0);ub_vecs.Z8(dst_off+1u,s[7]*sc1);
}
}else{
for(Z0 i=0;i<n;++i){
const Z1*s=aos+i*8u;Z0 dst_off=i*tile_stride+r0;
ub_gather.Z8(dst_off+0u,s[6]*sc0);ub_gather.Z8(dst_off+1u,s[7]*sc1);
}
}
}else{
Z0 v_col=need_orth_prev?7u:6u;Z1 nsq0=0.0f;
Z0 mi0=0u;Z1 ma0=Z7(aos[v_col]);
for(Z0 i=0;i<n;++i){
const Z1*s=aos+i*8u;Z1 v0=s[v_col];
nsq0+=v0*v0;Z1 a0=Z7(v0);
if(a0>ma0){ma0=a0;mi0=i;}
}
Z1 sc0=((aos[mi0*8u+v_col]<0.0f)?-1.0f:1.0f)*ZI(nsq0);if(ZB==n){
for(Z0 i=0;i<n;++i)ub_vecs.Z8(i*n+c1,aos[i*8u+v_col]*sc0);}else{
for(Z0 i=0;i<n;++i)ub_gather.Z8(i*tile_stride+r0,aos[i*8u+v_col]*sc0);}
}
continue;}
Z0 c2=(bcnt>2u)?(c0+2u):c1;Z0 c3=(bcnt>3u)?(c0+3u):c2;
Z1 lam0=dd[c0],lam1=dd[c1],lam2=dd[c2],lam3=dd[c3];


Z0 sm0=(c0*13u)%23u;Z0 sm1=sm0+13u;if(sm1>=23u)sm1-=23u;
Z0 sm2=sm1+13u;if(sm2>=23u)sm2-=23u;Z0 sm3=sm2+13u;if(sm3>=23u)sm3-=23u;
Z1 cq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 cq1=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm1));
Z1 cq2=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm2));
Z1 cq3=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm3));

Z1 cur_p0=orig_d[0]-lam0,cur_u0=orig_e[0];Z1 cur_p1=orig_d[0]-lam1,cur_u1=orig_e[0];
Z1 cur_p2=orig_d[0]-lam2,cur_u2=orig_e[0];Z1 cur_p3=orig_d[0]-lam3,cur_u3=orig_e[0];

for(Z0 i=0;i+1u<n;++i){
Z1 m_i=orig_e[i];Z1 abs_m=Z7(m_i);
Z1 d_next=orig_d[i+1u];Z1 e_next=orig_e[i+1u];
Z1 nd0=d_next-lam0,nd1=d_next-lam1,nd2=d_next-lam2,nd3=d_next-lam3;

bool sw0=(Z7(cur_p0)<abs_m);bool sw1=(Z7(cur_p1)<abs_m);
bool sw2=(Z7(cur_p2)<abs_m);bool sw3=(Z7(cur_p3)<abs_m);

Z1 inv_m=ww[i];Z1 inv0=sw0?inv_m:(1.0f/ClampPivot(cur_p0));
Z1 inv1=sw1?inv_m:(1.0f/ClampPivot(cur_p1));Z1 inv2=sw2?inv_m:(1.0f/ClampPivot(cur_p2));
Z1 inv3=sw3?inv_m:(1.0f/ClampPivot(cur_p3));

Z1 f0=(sw0?cur_p0:m_i)*inv0;Z1 f1=(sw1?cur_p1:m_i)*inv1;
Z1 f2=(sw2?cur_p2:m_i)*inv2;Z1 f3=(sw3?cur_p3:m_i)*inv3;

Z1*s=aos+i*16u;s[0]=inv0;s[1]=inv1;s[2]=inv2;s[3]=inv3;
s[4]=f0;s[5]=f1;s[6]=f2;s[7]=f3;Z1 u1_0=sw0?nd0:cur_u0;
Z1 u1_1=sw1?nd1:cur_u1;Z1 u1_2=sw2?nd2:cur_u2;
Z1 u1_3=sw3?nd3:cur_u3;s[8]=u1_0*inv0;
s[9]=u1_1*inv1;s[10]=u1_2*inv2;
s[11]=u1_3*inv3;

cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_p2=(sw2?cur_u2:nd2)-f2*u1_2;cur_p3=(sw3?cur_u3:nd3)-f3*u1_3;

cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
cur_u2=sw2?(-f2*e_next):e_next;cur_u3=sw3?(-f3*e_next):e_next;

swap_mask[i]=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u)|(sw2?4u:0u)|(sw3?8u:0u));

sm0+=17u;if(sm0>=23u)sm0-=23u;sm1+=17u;if(sm1>=23u)sm1-=23u;
sm2+=17u;if(sm2>=23u)sm2-=23u;sm3+=17u;if(sm3>=23u)sm3-=23u;
Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 nq1=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm1));
Z1 nq2=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm2));
Z1 nq3=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm3));

Z1 qi0=sw0?nq0:cq0;Z1 qi1=sw1?nq1:cq1;
Z1 qi2=sw2?nq2:cq2;Z1 qi3=sw3?nq3:cq3;
Z1 b0=sw0?cq0:nq0;Z1 b1=sw1?cq1:nq1;
Z1 b2=sw2?cq2:nq2;Z1 b3=sw3?cq3:nq3;
s[12]=qi0*inv0;s[13]=qi1*inv1;s[14]=qi2*inv2;s[15]=qi3*inv3;cq0=b0-f0*qi0;
cq1=b1-f1*qi1;cq2=b2-f2*qi2;
cq3=b3-f3*qi3;}
Z1*s_last=aos+(n-1u)*16u;s_last[0]=1.0f/ClampPivot(cur_p0);
s_last[1]=1.0f/ClampPivot(cur_p1);s_last[2]=1.0f/ClampPivot(cur_p2);
s_last[3]=1.0f/ClampPivot(cur_p3);

Z1 in0=1.0f,in1=1.0f,in2=1.0f,in3=1.0f;for(Z0 iter=0;iter<2u;++iter){
if(iter>0u){
cq0=aos[12]*in0;cq1=aos[13]*in1;cq2=aos[14]*in2;cq3=aos[15]*in3;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*16u;
Z1 nq0=s[28]*in0,nq1=s[29]*in1,nq2=s[30]*in2,nq3=s[31]*in3;Z1 qi0=(sm&1u)?nq0:cq0;
Z1 qi1=(sm&2u)?nq1:cq1;Z1 qi2=(sm&4u)?nq2:cq2;
Z1 qi3=(sm&8u)?nq3:cq3;Z1 b0=(sm&1u)?cq0:nq0;
Z1 b1=(sm&2u)?cq1:nq1;Z1 b2=(sm&4u)?cq2:nq2;
Z1 b3=(sm&8u)?cq3:nq3;s[12]=qi0*s[0];s[13]=qi1*s[1];s[14]=qi2*s[2];s[15]=qi3*s[3];
cq0=b0-s[4]*qi0;cq1=b1-s[5]*qi1;
cq2=b2-s[6]*qi2;cq3=b3-s[7]*qi3;
}
}
Z1 q0_n1=cq0*s_last[0],q1_n1=cq1*s_last[1],q2_n1=cq2*s_last[2],q3_n1=cq3*s_last[3];
s_last[12]=q0_n1;s_last[13]=q1_n1;s_last[14]=q2_n1;s_last[15]=q3_n1;
Z1 nsq0=q0_n1*q0_n1,nsq1=q1_n1*q1_n1,nsq2=q2_n1*q2_n1,nsq3=q3_n1*q3_n1;
Z1 q0_n2=0.0f,q1_n2=0.0f,q2_n2=0.0f,q3_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*16u;q0_n2=q0_n1;q1_n2=q1_n1;q2_n2=q2_n1;q3_n2=q3_n1;
q0_n1=s_nm2[12]-s_nm2[8]*q0_n2;q1_n1=s_nm2[13]-s_nm2[9]*q1_n2;
q2_n1=s_nm2[14]-s_nm2[10]*q2_n2;q3_n1=s_nm2[15]-s_nm2[11]*q3_n2;
s_nm2[12]=q0_n1;s_nm2[13]=q1_n1;s_nm2[14]=q2_n1;s_nm2[15]=q3_n1;
nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;nsq2+=q2_n1*q2_n1;nsq3+=q3_n1*q3_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_sq[ui];Z1*s=aos+ui*16u;
Z1 c0_v=(s[12]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[8]*q0_n1;
Z1 c1_v=(s[13]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[9]*q1_n1;
Z1 c2_v=(s[14]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[10]*q2_n1;
Z1 c3_v=(s[15]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[11]*q3_n1;s[12]=c0_v;s[13]=c1_v;s[14]=c2_v;s[15]=c3_v;
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;nsq2+=c2_v*c2_v;nsq3+=c3_v*c3_v;}
if(iter<1u){
in0=ScalarRsqrtFast(nsq0);in1=ScalarRsqrtFast(nsq1);
in2=ScalarRsqrtFast(nsq2);in3=ScalarRsqrtFast(nsq3);
}else{
in0=ZI(nsq0);in1=ZI(nsq1);
in2=ZI(nsq2);in3=ZI(nsq3);
}
}

Z1 rq0=0.0f,rq1=0.0f,rq2=0.0f,rq3=0.0f;
Z1 cur0=aos[12]*in0,cur1=aos[13]*in1,cur2=aos[14]*in2,cur3=aos[15]*in3;
aos[12]=cur0;aos[13]=cur1;aos[14]=cur2;aos[15]=cur3;for(Z0 i=0;i+1u<n;++i){
Z1*sn=aos+(i+1u)*16u;Z1 nxt0=sn[12]*in0,nxt1=sn[13]*in1,nxt2=sn[14]*in2,nxt3=sn[15]*in3;
sn[12]=nxt0;sn[13]=nxt1;sn[14]=nxt2;sn[15]=nxt3;Z1 d_i=orig_d[i];
Z1 two_e=2.0f*orig_e[i];rq0+=d_i*cur0*cur0+two_e*cur0*nxt0;
rq1+=d_i*cur1*cur1+two_e*cur1*nxt1;rq2+=d_i*cur2*cur2+two_e*cur2*nxt2;
rq3+=d_i*cur3*cur3+two_e*cur3*nxt3;cur0=nxt0;cur1=nxt1;cur2=nxt2;cur3=nxt3;
}
Z1 dl=orig_d[n-1u];rq0+=dl*cur0*cur0;
rq1+=dl*cur1*cur1;rq2+=dl*cur2*cur2;
rq3+=dl*cur3*cur3;ub_row.Z8(r0,rq0*matrix_scale);
if(bcnt>1u)ub_row.Z8(r0+1u,rq1*matrix_scale);if(bcnt>2u)ub_row.Z8(r0+2u,rq2*matrix_scale);
if(bcnt>3u)ub_row.Z8(r0+3u,rq3*matrix_scale);

for(ZG k=Z5<ZG>(n)-3;k>=0;--k){
Z0 uk=Z5<Z0>(k);if(tt[uk]!=0.0f){
const Z1*hk=a_loc+uk*32u;Z1 d0=0.0f,d1=0.0f,d2=0.0f,d3=0.0f;
for(Z0 i=uk+1u;i<n;++i){
Z1 v_i=hk[i];const Z1*s=aos+i*16u;
d0+=v_i*s[12];d1+=v_i*s[13];
d2+=v_i*s[14];d3+=v_i*s[15];
}
for(Z0 i=uk+1u;i<n;++i){
Z1 v_i=hk[i];Z1*s=aos+i*16u;
s[12]-=d0*v_i;s[13]-=d1*v_i;
s[14]-=d2*v_i;s[15]-=d3*v_i;
}
}
}

if(bcnt>=3u){
Z0 mi0=0u,mi1=0u,mi2=0u,mi3=0u;
Z1 ma0=Z7(aos[12]),ma1=Z7(aos[13]),ma2=Z7(aos[14]),ma3=Z7(aos[15]);
for(Z0 i=1u;i<n;++i){
const Z1*s=aos+i*16u;
Z1 a0=Z7(s[12]),a1=Z7(s[13]),a2=Z7(s[14]),a3=Z7(s[15]);
if(a0>ma0){ma0=a0;mi0=i;}
if(a1>ma1){ma1=a1;mi1=i;}
if(a2>ma2){ma2=a2;mi2=i;}
if(a3>ma3){ma3=a3;mi3=i;}
}
Z1 sc0=(aos[mi0*16u+12u]<0.0f)?-1.0f:1.0f;Z1 sc1=(aos[mi1*16u+13u]<0.0f)?-1.0f:1.0f;
Z1 sc2=(aos[mi2*16u+14u]<0.0f)?-1.0f:1.0f;Z1 sc3=(aos[mi3*16u+15u]<0.0f)?-1.0f:1.0f;
if(ZB==n){
if(bcnt==4u){
for(Z0 i=0;i<n;++i){
const Z1*s=aos+i*16u;Z0 dst_off=i*n+c0;
ub_vecs.Z8(dst_off+0u,s[12]*sc0);ub_vecs.Z8(dst_off+1u,s[13]*sc1);
ub_vecs.Z8(dst_off+2u,s[14]*sc2);ub_vecs.Z8(dst_off+3u,s[15]*sc3);
}
}else{
for(Z0 i=0;i<n;++i){
const Z1*s=aos+i*16u;Z0 dst_off=i*n+c0;
ub_vecs.Z8(dst_off+0u,s[12]*sc0);ub_vecs.Z8(dst_off+1u,s[13]*sc1);
ub_vecs.Z8(dst_off+2u,s[14]*sc2);}
}
}else{
for(Z0 i=0;i<n;++i){
const Z1*s=aos+i*16u;Z0 dst_off=i*tile_stride+r0;
ub_gather.Z8(dst_off+0u,s[12]*sc0);ub_gather.Z8(dst_off+1u,s[13]*sc1);
ub_gather.Z8(dst_off+2u,s[14]*sc2);if(bcnt>3u)ub_gather.Z8(dst_off+3u,s[15]*sc3);
}
}
}else{
for(Z0 b=0;b<bcnt;++b){
Z0 r=r0+b;Z0 col_idx=ZJ+r;
Z0 max_idx=0u;Z1 max_abs=Z7(aos[12u+b]);
for(Z0 i=1u;i<n;++i){
Z1 av=Z7(aos[i*16u+12u+b]);if(av>max_abs){
max_abs=av;max_idx=i;
}
}
Z1 sign_scale=(aos[max_idx*16u+12u+b]<0.0f)?-1.0f:1.0f;if(ZB==n){
for(Z0 i=0;i<n;++i){
ub_vecs.Z8(i*n+col_idx,aos[i*16u+12u+b]*sign_scale);}
}else{
for(Z0 i=0;i<n;++i){
ub_gather.Z8(i*tile_stride+r,aos[i*16u+12u+b]*sign_scale);}
}
}
}
}

fence<HardEvent::S_MTE3>();if(ZB==n){
DataCopyPad(
gm_eigvecs[b*total_elems],
ub_vecs,
DataCopyExtParams{1,total_elems*4u,0,0,0});DataCopyPad(
gm_eigvals[b*n+ZJ],
ub_row,
DataCopyExtParams{1,ZB*4u,0,0,0});}else{
DataCopyPad(
gm_eigvecs[b*total_elems+ZJ],
ub_gather,
DataCopyExtParams{Z5<uint16_t>(n),ZB*4u,0,(n-ZB)*4u,0});DataCopyPad(
gm_eigvals[b*n+ZJ],
ub_row,
DataCopyExtParams{1,ZB*4u,0,0,0});}
fence<HardEvent::MTE3_S>();}

__aicore__ inline Z1 VecDotSub(Z4<Z1>x,Z4<Z1>y,Z0 Z2){
Z0 len=n-Z2;Z3<ZC>();
Mul(ub_prod[Z2],x[Z2],y[Z2],len);Z3<ZC>();
if(len<=64u){
WholeReduceSum(ub_red,ub_prod[Z2],len,1,1,1,8);}else{
ReduceSum(ub_red,ub_prod[Z2],ub_tmp,len);}
Z3<PIPE_ALL>();return ub_red.ZF(0);
}

__aicore__ inline void VecAxpySub(Z4<Z1>dst,Z4<Z1>x,Z1 alpha,Z0 Z2){
Z0 len=n-Z2;Muls(ub_prod[Z2],x[Z2],alpha,len);
Z3<ZC>();Add(dst[Z2],dst[Z2],ub_prod[Z2],len);
Z3<ZC>();}


__aicore__ inline void FastReduceSum256(
Z4<Z1>dst,
Z4<Z1>src,
Z4<Z1>tmp,
Z0 len){
if(len<=64u){
WholeReduceSum(dst,src,len,1,1,1,8);}else if((len&7u)==0u){
WholeReduceSum(tmp,src,8u,Z5<ZA>(len>>3u),1,1,1);Z3<ZC>();
WholeReduceSum(dst,tmp,len>>3u,1,1,1,8);}else{
ReduceSum(dst,src,tmp,len);}
}

__aicore__ inline void ReduceRows64ByBlock2Stage(
Z4<Z1>dst,
Z4<Z1>sc,
Z0 rows){
BlockReduceSum(sc,sc,Z5<ZG>(rows),64,1,1,8);Z3<ZC>();
Z0 full8=rows>>3u;Z0 rem8=rows&7u;
if(full8>0u){
BlockReduceSum(dst,sc,Z5<ZG>(full8),64,1,1,8);}
if(rem8>0u){
BlockReduceSum(dst[full8<<3u],sc[full8<<6u],1,Z5<ZG>(rem8<<3u),1,1,8);}
}

__aicore__ inline void VecDotAxpyMaskBrcbSub(
Z4<Z1>w,
Z4<Z1>v,
Z0 Z2,
Z0 rem,
Z0 zero_base,
Z4<Z1>brcb_v,
Z0 brcb_Start,
ZA brcb_rep){
Z0 len=n-Z2;Mul(ub_prod[Z2],ub_prod[Z2],w[Z2],len);
if(brcb_rep>0u){
Brcb(brcb_v,v[brcb_Start],brcb_rep,{1,8});}
if(rem==0u&&Z2>zero_base){
Duplicate(v[Z2-8u],0.0f,8u);Duplicate(w[Z2-8u],0.0f,8u);
}
Z3<ZC>();if(rem!=0u){
Mul(w[Z2],w[Z2],ub_mask_ge[rem*8u],8u);}
if(len<=64u){
WholeReduceSum(ub_red,ub_prod[Z2],len,1,1,1,8);}else{
ReduceSum(ub_red,ub_prod[Z2],ub_tmp,len);}
Z3<ZC>();Z4<Z1>brcb_dst=(Z9<64u)?ub_cur_row:ub_tmp;
Brcb(brcb_dst,ub_red,1u,{1,8});Z3<ZC>();
Z0 full_s=len>>6u;Z0 tail_s=len&63u;
if(full_s>0u){
MulAddDst(w[Z2],v[Z2],brcb_dst,64u,Z5<ZA>(full_s),Z6{1,1,0,8,8,0});
}
if(tail_s>0u){
MulAddDst(w[Z2+(full_s<<6u)],v[Z2+(full_s<<6u)],brcb_dst,tail_s,1u,Z6{1,1,0,8,8,0});
}
Z3<ZC>();}

__aicore__ inline void VecDotAxpyNegWMaskBrcbSub(
Z4<Z1>w,
Z4<Z1>v,
Z0 Z2,
Z0 rem,
Z0 zero_base,
Z4<Z1>brcb_v,
Z0 brcb_Start,
ZA brcb_rep){
Z0 len=n-Z2;Z4<Z1>brcb_dst=(Z9<64u)?ub_cur_row:ub_tmp;
Mul(ub_prod[Z2],ub_prod[Z2],w[Z2],len);if(brcb_rep>0u){
Brcb(brcb_v,v[brcb_Start],brcb_rep,{1,8});}
Z3<ZC>();if(rem!=0u){
Mul(w[Z2],w[Z2],ub_mask_ge[rem*8u],8u);}
if(len<=64u){
WholeReduceSum(ub_red,ub_prod[Z2],len,1,1,1,8);}else{
ReduceSum(ub_red,ub_prod[Z2],ub_tmp,len);}
Z3<ZC>();Muls(w[Z2],w[Z2],-1.0f,len);
Brcb(brcb_dst,ub_red,1u,{1,8});Z3<ZC>();
Z0 full_s=len>>6u;Z0 tail_s=len&63u;
if(full_s>0u){
MulAddDst(w[Z2],v[Z2],brcb_dst,64u,Z5<ZA>(full_s),Z6{1,1,0,8,8,0});
}
if(tail_s>0u){
MulAddDst(w[Z2+(full_s<<6u)],v[Z2+(full_s<<6u)],brcb_dst,tail_s,1u,Z6{1,1,0,8,8,0});
}
Z3<ZC>();}

__aicore__ inline void BacktransformPairUnitRowsSub(
Z4<Z1>matrix_sub,
Z4<Z1>v_k,
Z4<Z1>v_km1,
Z4<Z1>scratch0,
Z4<Z1>scratch1,
Z0 ZD,
Z0 k){
#if EIGH_K96_PASS
#else
if constexpr(FAST200){
Z0 Z2=k&~7u;Z0 len=200u-Z2;
Z0 dual_rows=2u*ZD;Z0 aligned_dual=(dual_rows+7u)&~7u;
Z0 raw_sc=(len+7u)&~7u;Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;
constexpr ZA rep_stride=25u;ZA ZE=Z5<ZA>(sc_stride/8u);
ZA brcb_dual_rep=Z5<ZA>(aligned_dual/8u);Z4<Z1>brcb0=ub_gather;
Z4<Z1>brcb1=ub_gather[ZD*8u];Z0 full_s=len>>6u;
Z0 tail_s=len&63u;for(Z0 c=Z2;c<200u;c+=64u){
Z0 width=(200u-c<64u)?(200u-c):64u;Z0 sc=c-Z2;
Mul(scratch0[sc],matrix_sub[c],v_k[c],width,ZD,Z6{1,1,1,ZE,rep_stride,0});
Mul(scratch1[sc],matrix_sub[c],v_km1[c],width,ZD,Z6{1,1,1,ZE,rep_stride,0});
}
Mul(ub_prod[Z2],v_k[Z2],v_km1[Z2],len);Z3<ZC>();
if(len>64u){
for(Z0 c=Z2;c<200u;c+=64u){
Z0 width=(200u-c<64u)?(200u-c):64u;Z4<Z1>sc=scratch0[c-Z2];
BlockReduceSum(sc,sc,Z5<ZG>(dual_rows),Z5<ZG>(width),ZE,1,ZE);
}
BlockReduceSum(ub_tmp,ub_prod[Z2],Z5<ZG>(full_s),64,1,1,8);if(tail_s>0u){
WholeReduceSum(ub_tmp[full_s*8u],ub_prod[Z2+(full_s<<6u)],Z5<ZG>(tail_s),1,1,1,8);}
Z3<ZC>();WholeReduceSum(ub_row,scratch0,ZE,dual_rows,1,8,ZE);
WholeReduceSum(ub_red,ub_tmp,Z5<ZG>(full_s*8u+((tail_s>0u)?1u:0u)),1,1,1,8);
Muls(ub_prod[Z2],v_k[Z2],-1.0f,len);Z3<ZC>();
Brcb(ub_tmp,ub_red,1u,{1,8});Brcb(brcb0,ub_row,brcb_dual_rep,{1,8});
Muls(scratch1,v_km1[Z2],-1.0f,len);Z3<ZC>();
if(full_s>0u){
MulAddDst(ub_prod[Z2],v_km1[Z2],ub_tmp,64u,Z5<ZA>(full_s),Z6{1,1,0,8,8,0});
}
if(tail_s>0u){
MulAddDst(ub_prod[Z2+(full_s<<6u)],v_km1[Z2+(full_s<<6u)],ub_tmp,tail_s,1u,Z6{1,1,0,8,8,0});
}
for(Z0 c=Z2;c<200u;c+=64u){
Z0 width=(200u-c<64u)?(200u-c):64u;
MulAddDst(matrix_sub[c],scratch1[c-Z2],brcb1,width,ZD,Z6{1,1,0,rep_stride,0,1});
}
Z3<ZC>();for(Z0 c=Z2;c<200u;c+=64u){
Z0 width=(200u-c<64u)?(200u-c):64u;
MulAddDst(matrix_sub[c],ub_prod[c],brcb0,width,ZD,Z6{1,1,0,rep_stride,0,1});}
Z3<ZC>();return;
}else{
WholeReduceSum(ub_row,scratch0,len,dual_rows,1,1,ZE);
WholeReduceSum(ub_red,ub_prod[Z2],len,1,1,1,8);Z3<ZC>();
Brcb(ub_tmp,ub_red,1u,{1,8});Brcb(brcb0,ub_row,brcb_dual_rep,{1,8});
Muls(scratch1,v_km1[Z2],-1.0f,len);Muls(ub_prod[Z2],v_k[Z2],-1.0f,len);
Z3<ZC>();
MulAddDst(ub_prod[Z2],v_km1[Z2],ub_tmp,len,1u,Z6{1,1,0,8,8,0});
MulAddDst(matrix_sub[Z2],scratch1,brcb1,len,ZD,Z6{1,1,0,rep_stride,0,1});
Z3<ZC>();
MulAddDst(matrix_sub[Z2],ub_prod[Z2],brcb0,len,ZD,Z6{1,1,0,rep_stride,0,1});
Z3<ZC>();return;
}
}
#endif
if constexpr(SOLVE_MODE==3u){
Z0 Z2=k&~7u;Z0 len=n-Z2;
Z0 dual_rows=2u*ZD;Z0 aligned_dual=(dual_rows+7u)&~7u;
Z0 raw_sc=(len+7u)&~7u;Z0 sc_stride=(raw_sc>=64u)?raw_sc:((Z9<64u)?Z9:64u);
ZA rep_stride=Z5<ZA>(Z9/8u);
ZA ZE=Z5<ZA>(sc_stride/8u);
ZA brcb_dual_rep=Z5<ZA>(aligned_dual/8u);Z4<Z1>brcb0=ub_gather;
Z4<Z1>brcb1=ub_gather[ZD*8u];Z0 full_s=len>>6u;
Z0 tail_s=len&63u;

SetMaskNorm();Z0 last_mask=0u;for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
Mul<Z1,false>(scratch0[sc],matrix_sub[c],v_k[c],MASK_PLACEHOLDER,ZD,Z6{1,1,1,ZE,rep_stride,0});
Mul<Z1,false>(scratch1[sc],matrix_sub[c],v_km1[c],MASK_PLACEHOLDER,ZD,Z6{1,1,1,ZE,rep_stride,0});
}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);Mul(ub_prod[Z2],v_k[Z2],v_km1[Z2],len);
Z3<ZC>();

if(len>64u){
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z4<Z1>sc=scratch0[c-Z2];
BlockReduceSum(sc,sc,Z5<ZG>(dual_rows),
Z5<ZG>(width),ZE,1,ZE);}
if((Z9&7u)==0u){
BlockReduceSum(ub_tmp,ub_prod[Z2],Z5<ZG>(full_s),64,1,1,8);if(tail_s>0u){
WholeReduceSum(ub_tmp[full_s*8u],ub_prod[Z2+(full_s<<6u)],Z5<ZG>(tail_s),1,1,1,8);}
Z3<ZC>();WholeReduceSum(ub_row,scratch0,ZE,dual_rows,1,8,ZE);
WholeReduceSum(ub_red,ub_tmp,Z5<ZG>(full_s*8u+((tail_s>0u)?1u:0u)),1,1,1,8);
Muls(ub_prod[Z2],v_k[Z2],-1.0f,len);Z3<ZC>();
Brcb(ub_tmp,ub_red,1u,{1,8});Brcb(brcb0,ub_row,brcb_dual_rep,{1,8});
Muls(scratch1,v_km1[Z2],-1.0f,len);Z3<ZC>();
if(full_s>0u){
MulAddDst(ub_prod[Z2],v_km1[Z2],ub_tmp,64u,Z5<ZA>(full_s),Z6{1,1,0,8,8,0});
}
if(tail_s>0u){
MulAddDst(ub_prod[Z2+(full_s<<6u)],v_km1[Z2+(full_s<<6u)],ub_tmp,tail_s,1u,Z6{1,1,0,8,8,0});
}
SetMaskNorm();Z0 last_bt_mask=0u;for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;
if(width!=last_bt_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_bt_mask=width;}
MulAddDst<Z1,Z1,false>(matrix_sub[c],scratch1[c-Z2],brcb1,MASK_PLACEHOLDER,ZD,Z6{1,1,0,rep_stride,0,1});
}
Z3<ZC>();for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;
if(width!=last_bt_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_bt_mask=width;}
MulAddDst<Z1,Z1,false>(matrix_sub[c],ub_prod[c],brcb0,MASK_PLACEHOLDER,ZD,Z6{1,1,0,rep_stride,0,1});
}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);Z3<ZC>();
return;}
ReduceSum(ub_red,ub_prod[Z2],ub_tmp,len);Z3<ZC>();
WholeReduceSum(ub_row,scratch0,ZE,dual_rows,1,8,ZE);Brcb(ub_tmp,ub_red,1u,{1,8});
Z3<ZC>();Brcb(brcb0,ub_row,brcb_dual_rep,{1,8});
if(full_s>0u){
Mul(ub_prod[Z2],v_km1[Z2],ub_tmp,64u,Z5<ZA>(full_s),Z6{1,1,0,8,8,0});
}
if(tail_s>0u){
Mul(ub_prod[Z2+(full_s<<6u)],v_km1[Z2+(full_s<<6u)],ub_tmp,tail_s,1u,Z6{1,1,0,8,8,0});
}
Z3<ZC>();Sub(ub_prod[Z2],v_k[Z2],ub_prod[Z2],len);
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(scratch1[sc],v_km1[c],brcb1,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch1[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
Mul(scratch0[sc],ub_prod[c],brcb0,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch0[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
Z3<ZC>();return;
}else{
WholeReduceSum(ub_row,scratch0,len,dual_rows,1,1,ZE);
WholeReduceSum(ub_red,ub_prod[Z2],len,1,1,1,8);Z3<ZC>();
Brcb(ub_tmp,ub_red,1u,{1,8});Brcb(brcb0,ub_row,brcb_dual_rep,{1,8});
if((Z9&7u)==0u){
Muls(scratch1,v_km1[Z2],-1.0f,len);Muls(ub_prod[Z2],v_k[Z2],-1.0f,len);
Z3<ZC>();
MulAddDst(ub_prod[Z2],v_km1[Z2],ub_tmp,len,1u,Z6{1,1,0,8,8,0});
MulAddDst(matrix_sub[Z2],scratch1,brcb1,len,ZD,Z6{1,1,0,rep_stride,0,1});
Z3<ZC>();
MulAddDst(matrix_sub[Z2],ub_prod[Z2],brcb0,len,ZD,Z6{1,1,0,rep_stride,0,1});
Z3<ZC>();return;
}
Z3<ZC>();Mul(ub_prod[Z2],v_km1[Z2],ub_tmp,len,1u,Z6{1,1,0,8,8,0});
Mul(scratch1[0],v_km1[Z2],brcb1,len,ZD,Z6{1,1,0,ZE,0,1});
Z3<ZC>();Sub(ub_prod[Z2],v_k[Z2],ub_prod[Z2],len);
if(Z2==0u&&n==Z9&&sc_stride==Z9){
Sub(matrix_sub,matrix_sub,scratch1,ZD*Z9);}else{
Sub(matrix_sub[Z2],matrix_sub[Z2],scratch1[0],len,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
Z3<ZC>();
Mul(scratch0[0],ub_prod[Z2],brcb0,len,ZD,Z6{1,1,0,ZE,0,1});
Z3<ZC>();if(Z2==0u&&n==Z9&&sc_stride==Z9){
Sub(matrix_sub,matrix_sub,scratch0,ZD*Z9);}else{
Sub(matrix_sub[Z2],matrix_sub[Z2],scratch0[0],len,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
Z3<ZC>();return;
}

for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(scratch1[sc],v_km1[c],brcb1,width,ZD,Z6{1,1,0,ZE,0,1});}
Sub(ub_prod[Z2],v_k[Z2],ub_prod[Z2],len);Z3<ZC>();
if(Z2==0u&&n==Z9&&sc_stride==Z9){
Sub(matrix_sub,matrix_sub,scratch1,ZD*Z9);for(Z0 c=0u;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;
Mul(scratch0[c],ub_prod[c],brcb0,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();Sub(matrix_sub,matrix_sub,scratch0,ZD*Z9);
Z3<ZC>();return;
}
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch1[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
Mul(scratch0[sc],ub_prod[c],brcb0,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch0[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
Z3<ZC>();}else{
Z0 Z2=k&~7u;Z0 len=n-Z2;
Z0 dual_rows=2u*ZD;Z0 aligned_dual=(dual_rows+7u)&~7u;
Z0 raw_sc=(len+7u)&~7u;Z0 sc_stride=(raw_sc>=64u)?raw_sc:((Z9<64u)?Z9:64u);
ZA rep_stride=Z5<ZA>(Z9/8u);
ZA ZE=Z5<ZA>(sc_stride/8u);
ZA brcb_dual_rep=Z5<ZA>(aligned_dual/8u);Z4<Z1>brcb0=ub_gather;
Z4<Z1>brcb1=ub_gather[ZD*8u];Z0 full_s=len>>6u;
Z0 tail_s=len&63u;

for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(scratch0[sc],matrix_sub[c],v_k[c],width,ZD,Z6{1,1,1,ZE,rep_stride,0});
Mul(scratch1[sc],matrix_sub[c],v_km1[c],width,ZD,Z6{1,1,1,ZE,rep_stride,0});
}
Mul(ub_prod[Z2],v_k[Z2],v_km1[Z2],len);Z3<ZC>();

Z0 strip_idx=0u;for(Z0 c=Z2;c<n;c+=64u,++strip_idx){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
if(strip_idx==0u){
WholeReduceSum(ub_row,scratch0[sc],width,dual_rows,1,1,ZE);}else{
WholeReduceSum(ub_gather[(strip_idx-1u)*aligned_dual],scratch0[sc],width,dual_rows,1,1,ZE);}
}
if(len<=64u){
WholeReduceSum(ub_red,ub_prod[Z2],len,1,1,1,8);}else{
ReduceSum(ub_red,ub_prod[Z2],ub_tmp,len);}
Z3<ZC>();

Brcb(ub_tmp,ub_red,1u,{1,8});if(strip_idx==4u){
Add(ub_row,ub_row,ub_gather,dual_rows);
Add(ub_gather[aligned_dual],ub_gather[aligned_dual],ub_gather[2u*aligned_dual],dual_rows);
Z3<ZC>();Add(ub_row,ub_row,ub_gather[aligned_dual],dual_rows);
if(full_s>0u){
Mul(ub_prod[Z2],v_km1[Z2],ub_tmp,64u,Z5<ZA>(full_s),Z6{1,1,0,8,8,0});
}
if(tail_s>0u){
Mul(ub_prod[Z2+(full_s<<6u)],v_km1[Z2+(full_s<<6u)],ub_tmp,tail_s,1u,Z6{1,1,0,8,8,0});
}
Z3<ZC>();}else if(strip_idx==3u){
Add(ub_row,ub_row,ub_gather,dual_rows);Z3<ZC>();
Add(ub_row,ub_row,ub_gather[aligned_dual],dual_rows);if(full_s>0u){
Mul(ub_prod[Z2],v_km1[Z2],ub_tmp,64u,Z5<ZA>(full_s),Z6{1,1,0,8,8,0});
}
if(tail_s>0u){
Mul(ub_prod[Z2+(full_s<<6u)],v_km1[Z2+(full_s<<6u)],ub_tmp,tail_s,1u,Z6{1,1,0,8,8,0});
}
Z3<ZC>();}else if(strip_idx==2u){
Add(ub_row,ub_row,ub_gather,dual_rows);Z3<ZC>();
}else{
Brcb(brcb0,ub_row,brcb_dual_rep,{1,8});if constexpr(SOLVE_MODE==2u){
if((Z9&15u)==0u){
Muls(scratch1,v_km1[Z2],-1.0f,len);Muls(ub_prod[Z2],v_k[Z2],-1.0f,len);
Z3<ZC>();
MulAddDst(ub_prod[Z2],v_km1[Z2],ub_tmp,len,1u,Z6{1,1,0,8,8,0});
MulAddDst(matrix_sub[Z2],scratch1,brcb1,len,ZD,Z6{1,1,0,rep_stride,0,1});
Z3<ZC>();
MulAddDst(matrix_sub[Z2],ub_prod[Z2],brcb0,len,ZD,Z6{1,1,0,rep_stride,0,1});
Z3<ZC>();return;
}
}
Z3<ZC>();Mul(ub_prod[Z2],v_km1[Z2],ub_tmp,len,1u,Z6{1,1,0,8,8,0});
Mul(scratch1[0],v_km1[Z2],brcb1,len,ZD,Z6{1,1,0,ZE,0,1});
Z3<ZC>();Sub(ub_prod[Z2],v_k[Z2],ub_prod[Z2],len);
if(Z2==0u&&n==Z9&&sc_stride==Z9){
Sub(matrix_sub,matrix_sub,scratch1,ZD*Z9);}else{
Sub(matrix_sub[Z2],matrix_sub[Z2],scratch1[0],len,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
Z3<ZC>();
Mul(scratch0[0],ub_prod[Z2],brcb0,len,ZD,Z6{1,1,0,ZE,0,1});
Z3<ZC>();if(Z2==0u&&n==Z9&&sc_stride==Z9){
Sub(matrix_sub,matrix_sub,scratch0,ZD*Z9);}else{
Sub(matrix_sub[Z2],matrix_sub[Z2],scratch0[0],len,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
Z3<ZC>();return;
}

if(strip_idx>=2u){
Brcb(brcb0,ub_row,brcb_dual_rep,{1,8});if(strip_idx==2u){
if(full_s>0u){
Mul(ub_prod[Z2],v_km1[Z2],ub_tmp,64u,Z5<ZA>(full_s),Z6{1,1,0,8,8,0});
}
if(tail_s>0u){
Mul(ub_prod[Z2+(full_s<<6u)],v_km1[Z2+(full_s<<6u)],ub_tmp,tail_s,1u,Z6{1,1,0,8,8,0});
}
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(scratch1[sc],v_km1[c],brcb1,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();Sub(ub_prod[Z2],v_k[Z2],ub_prod[Z2],len);
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch1[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
Mul(scratch0[sc],ub_prod[c],brcb0,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch0[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
Z3<ZC>();return;
}else{
Sub(ub_prod[Z2],v_k[Z2],ub_prod[Z2],len);Z3<ZC>();
}
}

if(strip_idx>=3u){
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(scratch1[sc],v_km1[c],brcb1,width,ZD,Z6{1,1,0,ZE,0,1});
Mul(scratch0[sc],ub_prod[c],brcb0,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();Add(scratch0,scratch0,scratch1,ZD*sc_stride);
Z3<ZC>();if(Z2==0u&&n==Z9&&sc_stride==Z9){
Sub(matrix_sub,matrix_sub,scratch0,ZD*Z9);}else{
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch0[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
}
Z3<ZC>();return;
}

for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(scratch1[sc],v_km1[c],brcb1,width,ZD,Z6{1,1,0,ZE,0,1});}
Sub(ub_prod[Z2],v_k[Z2],ub_prod[Z2],len);Z3<ZC>();
if(Z2==0u&&n==Z9&&sc_stride==Z9){
Sub(matrix_sub,matrix_sub,scratch1,ZD*Z9);for(Z0 c=0u;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;
Mul(scratch0[c],ub_prod[c],brcb0,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();Sub(matrix_sub,matrix_sub,scratch0,ZD*Z9);
Z3<ZC>();return;
}
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch1[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
Mul(scratch0[sc],ub_prod[c],brcb0,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch0[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
Z3<ZC>();}
}

__aicore__ inline void VecDotAxpySub(Z4<Z1>w,Z4<Z1>v,Z1 scale,Z0 Z2){
Z0 len=n-Z2;Z3<ZC>();
Mul(ub_prod[Z2],v[Z2],w[Z2],len);Z3<ZC>();
if(len<=64u){
WholeReduceSum(ub_red,ub_prod[Z2],len,1,1,1,8);}else{
ReduceSum(ub_red,ub_prod[Z2],ub_tmp,len);}
Z3<ZC>();Muls(ub_red,ub_red,scale,8u);
Z3<ZC>();Z4<Z1>brcb_dst=(Z9<64u)?ub_gather:ub_tmp;
Brcb(brcb_dst,ub_red,1u,{1,8});Z3<ZC>();
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Mul(ub_prod[c],v[c],brcb_dst,width,1u,Z6{1,1,0,8,8,0});
}
Z3<ZC>();Add(w[Z2],w[Z2],ub_prod[Z2],len);
Z3<ZC>();}

__aicore__ inline void MatVecRowsSub(
Z4<Z1>matrix_sub,
Z4<Z1>x,
Z4<Z1>sums_sub,
Z4<Z1>scratch,
Z0 ZD,
Z0 Z2,
Z0 row_stride=0u,
Z0 col_off=0u){
if(ZD==0u||Z2>=n){
return;}
Z0 r_stride=(row_stride!=0u)?row_stride:Z9;Z0 raw_sc=((n-Z2)+7u)&~7u;
Z0 sc_stride=(raw_sc>=64u)?raw_sc:((r_stride<64u)?r_stride:64u);
Z0 aligned_cnt=(ZD+7u)&~7u;ZA rep_stride=Z5<ZA>(r_stride/8u);
ZA ZE=Z5<ZA>(sc_stride/8u);for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 mc=c-col_off;
Z0 sc=c-Z2;Mul(scratch[sc],
matrix_sub[mc],
x[c],
width,
ZD,
Z6{1,1,1,ZE,rep_stride,0});}
Z3<ZC>();Z0 strip_idx=0u;
for(Z0 c=Z2;c<n;c+=64u,++strip_idx){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
if(strip_idx==0u){
WholeReduceSum(sums_sub,scratch[sc],width,ZD,1,1,ZE);}else{
WholeReduceSum(ub_gather[(strip_idx-1u)*aligned_cnt],scratch[sc],width,ZD,1,1,ZE);}
}
Z3<ZC>();if(strip_idx==4u){
Add(sums_sub,sums_sub,ub_gather,ZD);
Add(ub_gather[aligned_cnt],ub_gather[aligned_cnt],ub_gather[2u*aligned_cnt],ZD);
Z3<ZC>();Add(sums_sub,sums_sub,ub_gather[aligned_cnt],ZD);
Z3<ZC>();}else{
for(Z0 s=1u;s<strip_idx;++s){
Add(sums_sub,sums_sub,ub_gather[(s-1u)*aligned_cnt],ZD);Z3<ZC>();
}
}
}


__aicore__ inline void BacktransformStepRowsSub(
Z4<Z1>matrix_sub,
Z4<Z1>v_k,
Z1 neg_tau_k,
Z4<Z1>scratch,
Z0 ZD,
Z0 Z2){
if(ZD==0u||Z2>=n){
return;}
Z0 aligned_cnt=(ZD+7u)&~7u;ZA rep_stride=Z5<ZA>(Z9/8u);
if(n<=64u){
Z0 width=n-Z2;Mul(scratch,
matrix_sub[Z2],
v_k[Z2],
width,
ZD,
Z6{1,1,1,rep_stride,rep_stride,0});Muls(ub_tmp,v_k[Z2],neg_tau_k,width);
Z3<ZC>();WholeReduceSum(ub_row,scratch,width,ZD,1,1,rep_stride);
Z3<ZC>();Brcb(ub_gather,ub_row,Z5<ZA>(aligned_cnt/8u),{1,8});
Z3<ZC>();if((Z9&15u)==0u){
MulAddDst(matrix_sub[Z2],
ub_tmp,
ub_gather,
width,
ZD,
Z6{1,1,0,rep_stride,0,1});Z3<ZC>();
}else{
Mul(scratch,
ub_tmp,
ub_gather,
width,
ZD,
Z6{1,1,0,rep_stride,0,1});Z3<ZC>();
Add(matrix_sub[Z2],
matrix_sub[Z2],
scratch,
width,
ZD,
Z6{1,1,1,rep_stride,rep_stride,rep_stride});Z3<ZC>();
}
return;}
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Mul(scratch[c],
matrix_sub[c],
v_k[c],
width,
ZD,
Z6{1,1,1,rep_stride,rep_stride,0});}
Muls(ub_tmp,v_k[Z2],neg_tau_k,n-Z2);Z3<ZC>();

Z0 strip_idx=0u;for(Z0 c=Z2;c<n;c+=64u,++strip_idx){
Z0 width=(n-c<64u)?(n-c):64u;if(strip_idx==0u){
WholeReduceSum(ub_row,scratch[c],width,ZD,1,1,rep_stride);}else{
WholeReduceSum(ub_gather[(strip_idx-1u)*aligned_cnt],scratch[c],width,ZD,1,1,rep_stride);}
}
Z3<ZC>();

if(strip_idx==4u){
Add(ub_row,ub_row,ub_gather,ZD);
Add(ub_gather[aligned_cnt],ub_gather[aligned_cnt],ub_gather[2u*aligned_cnt],ZD);
Z3<ZC>();Add(ub_row,ub_row,ub_gather[aligned_cnt],ZD);
Z3<ZC>();}else{
for(Z0 s=1u;s<strip_idx;++s){
Add(ub_row,ub_row,ub_gather[(s-1u)*aligned_cnt],ZD);Z3<ZC>();
}
}

Brcb(ub_gather,ub_row,Z5<ZA>(aligned_cnt/8u),{1,8});Z3<ZC>();

if((Z9&7u)==0u){
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;MulAddDst(matrix_sub[c],
ub_tmp[c-Z2],
ub_gather,
width,
ZD,
Z6{1,1,0,rep_stride,0,1});}
Z3<ZC>();}else{
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Mul(scratch[c],
ub_tmp[c-Z2],
ub_gather,
width,
ZD,
Z6{1,1,0,rep_stride,0,1});}
Z3<ZC>();

for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Add(matrix_sub[c],
matrix_sub[c],
scratch[c],
width,
ZD,
Z6{1,1,1,rep_stride,rep_stride,rep_stride});}
Z3<ZC>();}
}


__aicore__ inline void BacktransformPairRowsSub(
Z4<Z1>matrix_sub,
Z4<Z1>v_k,
Z4<Z1>v_km1,
Z1 tau_k,
Z1 tau_km1,
Z4<Z1>scratch0,
Z4<Z1>scratch1,
Z0 ZD,
Z0 k){
Z0 Z2=k&~7u;Z0 aligned_cnt=(ZD+7u)&~7u;
Z0 raw_sc=((n-Z2)+7u)&~7u;Z0 sc_stride=(raw_sc>=64u)?raw_sc:((Z9<64u)?Z9:64u);
ZA rep_stride=Z5<ZA>(Z9/8u);
ZA ZE=Z5<ZA>(sc_stride/8u);
ZA brcb_repeats=Z5<ZA>(aligned_cnt/8u);Z4<Z1>sums0=ub_row;
Z4<Z1>sums1=ub_row[aligned_cnt];

for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(scratch0[sc],matrix_sub[c],v_k[c],width,ZD,Z6{1,1,1,ZE,rep_stride,0});
Mul(scratch1[sc],matrix_sub[c],v_km1[c],width,ZD,Z6{1,1,1,ZE,rep_stride,0});
}
Mul(ub_prod[Z2],v_k[Z2],v_km1[Z2],n-Z2);Z3<ZC>();

Z0 strip_idx=0u;for(Z0 c=Z2;c<n;c+=64u,++strip_idx){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
if(strip_idx==0u){
WholeReduceSum(sums0,scratch0[sc],width,ZD,1,1,ZE);
WholeReduceSum(sums1,scratch1[sc],width,ZD,1,1,ZE);}else{
WholeReduceSum(ub_gather[(strip_idx-1u)*aligned_cnt],scratch0[sc],width,ZD,1,1,ZE);
WholeReduceSum(ub_gather[(strip_idx+2u)*aligned_cnt],scratch1[sc],width,ZD,1,1,ZE);}
}
ReduceSum(ub_red,ub_prod[Z2],ub_tmp,n-Z2);Z3<ZC>();

Muls(ub_red,ub_red,-tau_km1*tau_k,8u);if(strip_idx==4u){
Add(sums0,sums0,ub_gather,ZD);
Add(ub_gather[aligned_cnt],ub_gather[aligned_cnt],ub_gather[2u*aligned_cnt],ZD);
Add(sums1,sums1,ub_gather[3u*aligned_cnt],ZD);
Add(ub_gather[4u*aligned_cnt],ub_gather[4u*aligned_cnt],ub_gather[5u*aligned_cnt],ZD);
Z3<ZC>();Add(sums0,sums0,ub_gather[aligned_cnt],ZD);
Add(sums1,sums1,ub_gather[4u*aligned_cnt],ZD);Z3<ZC>();
}else{
for(Z0 s=1u;s<strip_idx;++s){
Add(sums0,sums0,ub_gather[(s-1u)*aligned_cnt],ZD);
Add(sums1,sums1,ub_gather[(s+2u)*aligned_cnt],ZD);Z3<ZC>();
}
if(strip_idx<=1u){
Z3<ZC>();}
}

Muls(ub_tmp,sums0,tau_k,ZD);Muls(ub_tmp[aligned_cnt],sums1,tau_km1,ZD);
Brcb(ub_tmp[2u*aligned_cnt],ub_red,1u,{1,8});Z3<ZC>();

Z4<Z1>brcb0=ub_gather;Z4<Z1>brcb1=ub_gather[aligned_cnt*8u];
Brcb(brcb0,ub_tmp,brcb_repeats,{1,8});
Mul(ub_prod,sums0,ub_tmp[2u*aligned_cnt],ZD,1u,Z6{1,1,0,8,8,0});
Z3<ZC>();

Add(ub_tmp[aligned_cnt],ub_tmp[aligned_cnt],ub_prod,ZD);for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(scratch0[sc],v_k[c],brcb0,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();

Brcb(brcb1,ub_tmp[aligned_cnt],brcb_repeats,{1,8});if(Z2==0u&&n==Z9&&sc_stride==Z9){
Sub(matrix_sub,matrix_sub,scratch0,ZD*Z9);for(Z0 c=0u;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;
Mul(scratch0[c],v_km1[c],brcb1,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();Sub(matrix_sub,matrix_sub,scratch0,ZD*Z9);
Z3<ZC>();return;
}
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch0[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
Mul(scratch0[sc],v_km1[c],brcb1,width,ZD,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();

for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(matrix_sub[c],matrix_sub[c],scratch0[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
Z3<ZC>();}


__aicore__ inline void Rank2UpdateRowsSub(
Z4<Z1>matrix_sub,
Z4<Z1>v_full,
Z4<Z1>w_full,
Z4<Z1>v_sub,
Z4<Z1>w_sub,
Z4<Z1>scratch,
Z0 ZD,
Z0 Z2,
Z0 row_stride=0u,
Z0 col_off=0u){
if(ZD==0u||Z2>=n){
return;}
Z0 r_stride=(row_stride!=0u)?row_stride:Z9;Z0 raw_sc=((n-Z2)+7u)&~7u;
Z0 sc_stride=(raw_sc>=64u)?raw_sc:((r_stride<64u)?r_stride:64u);
Z0 aligned_rows=(ZD+7u)&~7u;ZA rep_stride=Z5<ZA>(r_stride/8u);
ZA ZE=Z5<ZA>(sc_stride/8u);
ZA brcb_repeats=Z5<ZA>(aligned_rows/8u);

Z4<Z1>brcb_v=ub_gather;Z4<Z1>brcb_w=ub_gather[aligned_rows*8u];
Brcb(brcb_v,v_sub,brcb_repeats,{1,8});Brcb(brcb_w,w_sub,brcb_repeats,{1,8});
Z3<ZC>();

for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(scratch[sc],
w_full[c],
brcb_v,
width,
ZD,
Z6{1,1,0,ZE,0,1});}
Z3<ZC>();for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 mc=c-col_off;
Z0 sc=c-Z2;Sub(matrix_sub[mc],
matrix_sub[mc],
scratch[sc],
width,
ZD,
Z6{1,1,1,rep_stride,rep_stride,ZE});Mul(scratch[sc],
v_full[c],
brcb_w,
width,
ZD,
Z6{1,1,0,ZE,0,1});}
Z3<ZC>();for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 mc=c-col_off;
Z0 sc=c-Z2;Sub(matrix_sub[mc],
matrix_sub[mc],
scratch[sc],
width,
ZD,
Z6{1,1,1,rep_stride,rep_stride,ZE});}
Z3<ZC>();}

__aicore__ inline Z4<Z1>GetStaircaseReflector(Z0 k)const{
if(n==256u){
if(k>=96u){
if(k<112u)return ZH[15360u+(k-96u)*160u-96u];if(k<128u)return ZH[17920u+(k-112u)*144u-112u];
if(k<144u)return ZH[20224u+(k-128u)*128u-128u];if(k<160u)return ZH[22272u+(k-144u)*112u-144u];
if(k<192u)return ZH[24064u+(k-160u)*96u-160u];if(k<224u)return ZH[27136u+(k-192u)*64u-192u];
return ZH[29184u+(k-224u)*32u-224u];}
Z0 z=k>>5u;Z0 z_row=z<<5u;
Z0 stride_z=256u-z_row;Z0 z_off=(k>=24u)?(zone_offset[z]-6144u):29184u;
return ZH[z_off+(k-z_row)*stride_z-z_row];}
Z0 z=k>>5u;Z0 z_row=z<<5u;
Z0 stride_z=Z9-z_row;return ZH[zone_offset[z]+(k-z_row)*stride_z-z_row];
}

__aicore__ inline Z4<Z1>GetReflector200(Z0 k)const{
const Z0 z200_off[11]={0u,1600u,4672u,7488u,10048u,12352u,14400u,16192u,16960u,19072u,21120u};
const Z0 z200_row[11]={0u,8u,24u,40u,56u,72u,88u,104u,112u,136u,168u};
Z0 z=(k<8u)?0u:((k<104u)?(1u+((k-8u)>>4u)):((k<112u)?7u:((k<136u)?8u:((k<168u)?9u:10u))));
Z0 z_row=z200_row[z];return ZH[z200_off[z]+(k-z_row)*(200u-z_row)-z_row];
}


__aicore__ inline void LoadAndSymmetrizeMedium(uint64_t b){
uint64_t batch_base=b*n*n;DataCopyPad(
ZH,
gm_A[batch_base],
DataCopyExtParams{Z5<uint16_t>(n),n*4u,0,0,0},
DataCopyPadExtParams<Z1>{true,0,Z5<ZA>(Z9-n),0.0f});
Z4<Z0>col_idx_buf=ub_diag.ReinterpretCast<Z0>();for(Z0 j=0;j<Z9;++j){
col_idx_buf.Z8(j,j*Z9*4u);}
Z3<PIPE_ALL>();

if(uplo!=0u){
Z0 n_end8=n&~7u;Sub(ub_tmp,ub_mask_ge,ub_mask_ge,8u,8u,Z6{1,0,1,1,0,1});
for(Z0 i=0u;i<n;++i){
Z0 c_split=i&~7u;if(c_split>0u){
Gather(ZH[i*Z9],ZH,col_idx_buf,i*4u,c_split);}
if(c_split<n_end8){
Gather(ub_refl[i*8u],ZH[c_split*Z9],col_idx_buf,i*4u,8u);}
}
Z3<ZC>();Z0 num_g=n_end8>>3u;
ZA m_rep=Z5<ZA>(Z9/8u);for(Z0 g=0;g<num_g;++g){
Z0 c_s=g<<3u;Z4<Z1>m_blk=ZH[c_s*Z9+c_s];
Mul(ub_refl[g*64u],ub_refl[g*64u],ub_tmp,64u);
Mul(m_blk,m_blk,ub_mask_ge,8u,8u,Z6{1,1,1,m_rep,m_rep,1});}
Z3<ZC>();for(Z0 g=0;g<num_g;++g){
Z0 c_s=g<<3u;Z4<Z1>m_blk=ZH[c_s*Z9+c_s];
Add(m_blk,m_blk,ub_refl[g*64u],8u,8u,Z6{1,1,1,m_rep,m_rep,1});}
if(n_end8<n){
Z3<PIPE_ALL>();for(Z0 i=n_end8;i<n;++i){
Z0 row_base=i*Z9;for(Z0 j=n_end8;j<i;++j){
ZH.Z8(row_base+j,ZH.ZF(j*Z9+i));}
}
}
}else{
Z0 n_end8=n&~7u;Sub(ub_tmp,ub_mask_ge,ub_mask_gt,8u,8u,Z6{1,0,1,1,0,1});
for(Z0 i=0u;i<n;++i){
Z0 c_split=i&~7u;Z0 c_next8=c_split+8u;
if(c_split<n_end8){
Gather(ub_refl[i*8u],ZH[c_split*Z9],col_idx_buf,i*4u,8u);}
if(c_next8<n_end8){
Gather(ZH[i*Z9+c_next8],ZH[c_next8*Z9],col_idx_buf,i*4u,n_end8-c_next8);}
}
Z3<ZC>();Z0 num_g=n_end8>>3u;
ZA m_rep=Z5<ZA>(Z9/8u);for(Z0 g=0;g<num_g;++g){
Z0 c_s=g<<3u;Z4<Z1>m_blk=ZH[c_s*Z9+c_s];
Mul(ub_refl[g*64u],ub_refl[g*64u],ub_mask_gt,64u);
Mul(m_blk,m_blk,ub_tmp,8u,8u,Z6{1,1,1,m_rep,m_rep,1});}
Z3<ZC>();for(Z0 g=0;g<num_g;++g){
Z0 c_s=g<<3u;Z4<Z1>m_blk=ZH[c_s*Z9+c_s];
Add(m_blk,m_blk,ub_refl[g*64u],8u,8u,Z6{1,1,1,m_rep,m_rep,1});}
if(n_end8<n){
Z3<PIPE_ALL>();for(Z0 i=0u;i<n;++i){
Z0 row_base=i*Z9;Z0 j_start=(i>=n_end8)?(i+1u):n_end8;
for(Z0 j=j_start;j<n;++j){
ZH.Z8(row_base+j,ZH.ZF(j*Z9+i));}
}
}
}
if((n&7u)!=0u){
fence<HardEvent::S_V>();}
Z3<ZC>();Abs(ub_vecs,ZH,n*Z9);
Z3<ZC>();ReduceMax(ub_red,ub_vecs,ub_gather,n*Z9);
Duplicate(ub_v,0.0f,Z9);Duplicate(ub_w,0.0f,Z9);
if(n<Z9){
Duplicate(ub_prod,0.0f,Z9);}
fence<HardEvent::V_S>();Z1 max_val=ub_red.ZF(0);

Z1 inv_scale=1.0f;ComputePow2Scale(max_val,matrix_scale,inv_scale);
Muls(ZH,ZH,inv_scale,n*Z9);Z3<ZC>();
}


__aicore__ inline void LoadAndSymmetrizeLarge(uint64_t b){
uint64_t batch_base=b*n*n;if(n<=200u){
DataCopyPad(
ZH,
gm_A[batch_base],
DataCopyExtParams{Z5<uint16_t>(n),n*4u,0,0,0},
DataCopyPadExtParams<Z1>{true,0,Z5<ZA>(Z9-n),0.0f});
Z4<Z0>col_idx_buf=ub_diag.ReinterpretCast<Z0>();for(Z0 j=0;j<Z9;++j){
col_idx_buf.Z8(j,j*Z9*4u);}
Z3<PIPE_ALL>();

if(uplo!=0u){
Z0 n_end8=n&~7u;Sub(ub_tmp,ub_mask_ge,ub_mask_ge,8u,8u,Z6{1,0,1,1,0,1});
for(Z0 i=0u;i<n;++i){
Z0 c_split=i&~7u;if(c_split>0u){
Gather(ZH[i*Z9],ZH,col_idx_buf,i*4u,c_split);}
if(c_split<n_end8){
Gather(ub_refl[i*8u],ZH[c_split*Z9],col_idx_buf,i*4u,8u);}
}
Z3<ZC>();Z0 num_g=n_end8>>3u;
ZA m_rep=Z5<ZA>(Z9/8u);for(Z0 g=0;g<num_g;++g){
Z0 c_s=g<<3u;Z4<Z1>m_blk=ZH[c_s*Z9+c_s];
Mul(ub_refl[g*64u],ub_refl[g*64u],ub_tmp,64u);
Mul(m_blk,m_blk,ub_mask_ge,8u,8u,Z6{1,1,1,m_rep,m_rep,1});}
Z3<ZC>();for(Z0 g=0;g<num_g;++g){
Z0 c_s=g<<3u;Z4<Z1>m_blk=ZH[c_s*Z9+c_s];
Add(m_blk,m_blk,ub_refl[g*64u],8u,8u,Z6{1,1,1,m_rep,m_rep,1});}
if(n_end8<n){
Z3<PIPE_ALL>();for(Z0 i=n_end8;i<n;++i){
Z0 row_base=i*Z9;for(Z0 j=n_end8;j<i;++j){
ZH.Z8(row_base+j,ZH.ZF(j*Z9+i));}
}
}
}else{
Z0 n_end8=n&~7u;Sub(ub_tmp,ub_mask_ge,ub_mask_gt,8u,8u,Z6{1,0,1,1,0,1});
for(Z0 i=0u;i<n;++i){
Z0 c_split=i&~7u;Z0 c_next8=c_split+8u;
if(c_split<n_end8){
Gather(ub_refl[i*8u],ZH[c_split*Z9],col_idx_buf,i*4u,8u);}
if(c_next8<n_end8){
Gather(ZH[i*Z9+c_next8],ZH[c_next8*Z9],col_idx_buf,i*4u,n_end8-c_next8);}
}
Z3<ZC>();Z0 num_g=n_end8>>3u;
ZA m_rep=Z5<ZA>(Z9/8u);for(Z0 g=0;g<num_g;++g){
Z0 c_s=g<<3u;Z4<Z1>m_blk=ZH[c_s*Z9+c_s];
Mul(ub_refl[g*64u],ub_refl[g*64u],ub_mask_gt,64u);
Mul(m_blk,m_blk,ub_tmp,8u,8u,Z6{1,1,1,m_rep,m_rep,1});}
Z3<ZC>();for(Z0 g=0;g<num_g;++g){
Z0 c_s=g<<3u;Z4<Z1>m_blk=ZH[c_s*Z9+c_s];
Add(m_blk,m_blk,ub_refl[g*64u],8u,8u,Z6{1,1,1,m_rep,m_rep,1});}
if(n_end8<n){
Z3<PIPE_ALL>();for(Z0 i=0u;i<n;++i){
Z0 row_base=i*Z9;Z0 j_start=(i>=n_end8)?(i+1u):n_end8;
for(Z0 j=j_start;j<n;++j){
ZH.Z8(row_base+j,ZH.ZF(j*Z9+i));}
}
}
}
if((n&7u)!=0u){
fence<HardEvent::S_V>();}
Z3<ZC>();Z0 total_m=n*Z9;
Z0 chunk_len=2880u;Z4<Z1>sc_max=ub_w;
Abs(sc_max,ZH,chunk_len);if(total_m>2880u){
Z0 c1=(total_m-2880u<2880u)?(total_m-2880u):2880u;Abs(sc_max[2880],ZH[2880],c1);
}
Z3<ZC>();Z0 buf_sel=0u;
Z0 prev_len=(total_m>2880u)?((total_m-2880u<2880u)?(total_m-2880u):2880u):0u;
for(Z0 off=5760u;off<total_m;off+=2880u,buf_sel^=1u){
Z0 c_len=(total_m-off<2880u)?(total_m-off):2880u;
Max(sc_max,sc_max,sc_max[(buf_sel==0u)?2880u:5760u],prev_len);
Abs(sc_max[(buf_sel==0u)?5760u:2880u],ZH[off],c_len);Z3<ZC>();
prev_len=c_len;}
if(prev_len>0u){
Max(sc_max,sc_max,sc_max[(buf_sel==0u)?2880u:5760u],prev_len);Z3<ZC>();
}
ReduceMax(ub_red,sc_max,sc_max[2880],chunk_len);Duplicate(ub_v,0.0f,Z9);
Duplicate((n==200u)?ub_row:ub_w,0.0f,Z9);if(n<Z9){
Duplicate(ub_prod,0.0f,Z9);}
fence<HardEvent::V_S>();Z1 max_val=ub_red.ZF(0);

Z1 inv_scale=1.0f;ComputePow2Scale(max_val,matrix_scale,inv_scale);
Muls(ZH,ZH,inv_scale,n*Z9);Z3<ZC>();
return;}

Z0 last_z=num_zones-1u;Duplicate(ZH[zone_offset[last_z]],0.0f,32u*(Z9-(last_z<<5u)));
Z4<Z0>col_idx_buf=ub_off.ReinterpretCast<Z0>();if(uplo==0u){
for(Z0 j=0;j<(n&~7u);++j){
col_idx_buf.Z8(j,j*32u*4u);}
Sub(ub_tmp,ub_mask_ge,ub_mask_gt,8u,8u,Z6{1,0,1,1,0,1});Z3<PIPE_ALL>();
for(Z0 z=0;z<num_zones;++z){
Z0 z_row=z<<5u;Z0 z_cnt=(z_row+32u<n)?32u:(n-z_row);
Z0 col_stride=(z_cnt+7u)&~7u;ZA col_pad=Z5<ZA>(col_stride-z_cnt);
Z0 n_z=n-z_row;Z0 stride_z=Z9-z_row;
Z0 n_z_end8=n_z&~7u;DataCopyPad(
ub_refl,
gm_A[batch_base+Z5<uint64_t>(z_row)*n+z_row],
DataCopyExtParams{Z5<uint16_t>(n_z),z_cnt*4u,(n-z_cnt)*4u,0,0},
DataCopyPadExtParams<Z1>{col_pad!=0u,0,col_pad,0.0f});if(col_stride!=32u){
for(Z0 j=0;j<n_z_end8;++j){
col_idx_buf.Z8(j,j*col_stride*4u);}
Z3<PIPE_ALL>();}else{
fence<HardEvent::MTE2_V>();}

for(Z0 ir=0;ir<z_cnt;++ir){
Z4<Z1>row_z=ZH[zone_offset[z]+ir*stride_z];Z0 c_split=ir&~7u;
if(c_split>0u){
Adds(row_z,ub_refl[ir*col_stride],0.0f,c_split);}
if(c_split<n_z_end8){
Gather(row_z[c_split],ub_refl[c_split*col_stride],col_idx_buf,ir*4u,n_z_end8-c_split);}
}
Z3<ZC>();Z0 full_g=(z_cnt&~7u)>>3u;
ZA z_rep=Z5<ZA>(stride_z/8u);ZA c_rep=Z5<ZA>(col_stride/8u);
for(Z0 g=0;g<full_g;++g){
Z0 c_s=g<<3u;if(c_s<n_z_end8){
Z4<Z1>rz_blk=ZH[zone_offset[z]+c_s*stride_z+c_s];
Mul(rz_blk,rz_blk,ub_mask_gt,8u,8u,Z6{1,1,1,z_rep,z_rep,1});
Mul(ub_w[g*64u],ub_refl[c_s*col_stride+c_s],ub_tmp,8u,8u,Z6{1,1,1,1,c_rep,1});}
}
for(Z0 ir=full_g<<3u;ir<z_cnt;++ir){
Z0 c_split=ir&~7u;if(c_split<n_z_end8){
Z4<Z1>rz_row=ZH[zone_offset[z]+ir*stride_z+c_split];
Mul(rz_row,rz_row,ub_mask_gt[(ir&7u)*8u],8u);
Mul(ub_w[ir*8u],ub_refl[ir*col_stride+c_split],ub_tmp[(ir&7u)*8u],8u);}
}
if constexpr(!FAST200){if(z+1u<num_zones&&n_z==stride_z&&n_z_end8==n_z){fence<HardEvent::V_MTE2>();}}
Z3<ZC>();for(Z0 g=0;g<full_g;++g){
Z0 c_s=g<<3u;if(c_s<n_z_end8){
Z4<Z1>rz_blk=ZH[zone_offset[z]+c_s*stride_z+c_s];
Add(rz_blk,rz_blk,ub_w[g*64u],8u,8u,Z6{1,1,1,z_rep,z_rep,1});}
}
for(Z0 ir=full_g<<3u;ir<z_cnt;++ir){
Z0 c_split=ir&~7u;if(c_split<n_z_end8){
Z4<Z1>rz_row=ZH[zone_offset[z]+ir*stride_z+c_split];Add(rz_row,rz_row,ub_w[ir*8u],8u);
}
}
if(n_z!=stride_z||n_z_end8!=n_z){
Z3<PIPE_ALL>();for(Z0 ir=0;ir<z_cnt;++ir){
Z4<Z1>row_z=ZH[zone_offset[z]+ir*stride_z];Z0 c_split=ir&~7u;
if(c_split>=n_z_end8){
for(Z0 jc=c_split;jc<=ir&&jc<n_z;++jc){
row_z.Z8(jc,ub_refl.ZF(ir*col_stride+jc));}
for(Z0 jc=ir+1u;jc<n_z;++jc){
row_z.Z8(jc,ub_refl.ZF(jc*col_stride+ir));}
}else{
for(Z0 jc=n_z_end8;jc<n_z;++jc){
row_z.Z8(jc,ub_refl.ZF(jc*col_stride+ir));}
}
for(Z0 jc=n_z;jc<stride_z;++jc){
row_z.Z8(jc,0.0f);}
}
Z3<PIPE_ALL>();}else if constexpr(!FAST200){Z3<ZC>();}else{Z3<PIPE_ALL>();}}
}else{
Sub(ub_tmp,ub_mask_ge,ub_mask_ge,8u,8u,Z6{1,0,1,1,0,1});for(Z0 z=0;z<num_zones;++z){
Z0 z_row=z<<5u;Z0 z_cnt=(z_row+32u<n)?32u:(n-z_row);
Z0 n_z=n-z_row;Z0 stride_z=Z9-z_row;
DataCopyPad(
ZH[zone_offset[z]],
gm_A[batch_base+Z5<uint64_t>(z_row)*n+z_row],
DataCopyExtParams{Z5<uint16_t>(z_cnt),n_z*4u,z_row*4u,0,0},
DataCopyPadExtParams<Z1>{true,0,Z5<ZA>(stride_z-n_z),0.0f});for(Z0 j=0;j<32u;++j){
col_idx_buf.Z8(z*32u+j,j*stride_z*4u);}
}
if constexpr(!FAST200){fence<HardEvent::S_V>();fence<HardEvent::MTE2_V>();}else{Z3<PIPE_ALL>();}
for(Z0 z=0;z<num_zones;++z){
Z0 z_row=z<<5u;Z0 z_cnt=(z_row+32u<n)?32u:(n-z_row);
Z0 stride_z=Z9-z_row;Z0 z_end8=z_cnt&~7u;
Z4<Z0>c_idx_z=col_idx_buf[z*32u];for(Z0 ir=0u;ir<z_cnt;++ir){
Z4<Z1>row_z=ZH[zone_offset[z]+ir*stride_z];Z0 c_split=ir&~7u;
if(c_split>0u){
Gather(row_z,ZH[zone_offset[z]],c_idx_z,ir*4u,c_split);}
if(c_split<z_end8){
Gather(ub_refl[z*256u+ir*8u],ZH[zone_offset[z]+c_split*stride_z],c_idx_z,ir*4u,8u);}
}
}
Z3<ZC>();for(Z0 z=0;z<num_zones;++z){
Z0 z_row=z<<5u;Z0 z_cnt=(z_row+32u<n)?32u:(n-z_row);
Z0 stride_z=Z9-z_row;Z0 full_g=(z_cnt&~7u)>>3u;
ZA z_rep=Z5<ZA>(stride_z/8u);for(Z0 g=0;g<full_g;++g){
Z0 c_s=g<<3u;Z4<Z1>rz_blk=ZH[zone_offset[z]+c_s*stride_z+c_s];
Mul(ub_refl[z*256u+g*64u],ub_refl[z*256u+g*64u],ub_tmp,64u);
Mul(rz_blk,rz_blk,ub_mask_ge,8u,8u,Z6{1,1,1,z_rep,z_rep,1});}
}
Z3<ZC>();for(Z0 z=0;z<num_zones;++z){
Z0 z_row=z<<5u;Z0 z_cnt=(z_row+32u<n)?32u:(n-z_row);
Z0 stride_z=Z9-z_row;Z0 full_g=(z_cnt&~7u)>>3u;
ZA z_rep=Z5<ZA>(stride_z/8u);for(Z0 g=0;g<full_g;++g){
Z0 c_s=g<<3u;Z4<Z1>rz_blk=ZH[zone_offset[z]+c_s*stride_z+c_s];
Add(rz_blk,rz_blk,ub_refl[z*256u+g*64u],8u,8u,Z6{1,1,1,z_rep,z_rep,1});}
Z0 z_end8=z_cnt&~7u;if(z_end8<z_cnt){
Z3<PIPE_ALL>();for(Z0 ir=z_end8;ir<z_cnt;++ir){
Z4<Z1>row_z=ZH[zone_offset[z]+ir*stride_z];for(Z0 jc=z_end8;jc<ir;++jc){
row_z.Z8(jc,ZH.ZF(zone_offset[z]+jc*stride_z+ir));}
}
}
}
Z3<ZC>();}

Z0 zone_total=zone_offset[num_zones];Z0 chunk_init=(zone_total<3840u)?zone_total:3840u;
Z4<Z1>sc_max=ub_diag;Abs(sc_max,ZH,chunk_init);
if(zone_total>3840u){
Z0 c1=(zone_total-3840u<3840u)?(zone_total-3840u):3840u;Abs(sc_max[3840],ZH[3840],c1);
}
Z3<ZC>();Z0 buf_sel=0u;
Z0 prev_len=(zone_total>3840u)?((zone_total-3840u<3840u)?(zone_total-3840u):3840u):0u;
for(Z0 off=7680u;off<zone_total;off+=3840u,buf_sel^=1u){
Z0 c_len=(zone_total-off<3840u)?(zone_total-off):3840u;
Max(sc_max,sc_max,sc_max[(buf_sel==0u)?3840u:7680u],prev_len);
Abs(sc_max[(buf_sel==0u)?7680u:3840u],ZH[off],c_len);Z3<ZC>();
prev_len=c_len;}
if(prev_len>0u){
Max(sc_max,sc_max,sc_max[(buf_sel==0u)?3840u:7680u],prev_len);Z3<ZC>();
}
ReduceMax(ub_red,sc_max,sc_max[3840],chunk_init);Duplicate(ub_v,0.0f,Z9);
Duplicate(ub_w,0.0f,Z9);if(n<Z9){
Duplicate(ub_prod,0.0f,Z9);}
fence<HardEvent::V_S>();Z1 max_val=ub_red.ZF(0);

Z1 inv_scale=1.0f;ComputePow2Scale(max_val,matrix_scale,inv_scale);
for(Z0 off=0u;off<zone_total;off+=8192u){
Z0 c_len=(zone_total-off<8192u)?(zone_total-off):8192u;Muls(ZH[off],ZH[off],inv_scale,c_len);
}
Z3<ZC>();}

__aicore__ inline ZG IssueLookaheadNormRed(
Z4<Z1>row_next,
Z4<Z1>sc_prod,
Z4<Z1>sc_tmp,
Z0 c_next,
Z0 c_off_next,
Z0 rem_next,
Z0 len_next){
Mul(sc_prod,row_next[c_off_next],row_next[c_off_next],len_next);Z3<ZC>();
if(rem_next!=0u){
Mul(sc_prod,sc_prod,ub_mask_ge[rem_next*8u],8u);Z3<ZC>();
}
if(len_next<=64u){WholeReduceSum(ub_red,sc_prod,len_next,1,1,1,8);}else{ReduceSum(ub_red,sc_prod,sc_tmp,len_next);}
const ZG ev=Z5<ZG>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(ev);return ev;
}





template<Z0 N,Z0 PHASE=0u>
__aicore__ inline void Rank2UpdateFlatMulAdd(
Z4<Z1>mat_t,Z4<Z1>negative_w,
Z4<Z1>v,Z4<Z1>brcb_v,
Z4<Z1>brcb_negative_w,Z0 row_base,Z0 Z2,Z0 rows){
const Z0 stride_t=N-row_base;const ZA t_rep=Z5<ZA>(stride_t/8u);
SetMaskNorm();Z0 last_mask=0u;if constexpr(PHASE!=2u){for(Z0 c=Z2;c<N;c+=64u){
Z0 width=(N-c<64u)?(N-c):64u;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
MulAddDst<Z1,Z1,false>(mat_t[c-row_base],negative_w[c],brcb_v,MASK_PLACEHOLDER,rows,
Z6{1,1,0,t_rep,0,1});}}
if constexpr(PHASE==0u){Z3<ZC>();}if constexpr(PHASE!=1u){for(Z0 c=Z2;c<N;c+=64u){
Z0 width=(N-c<64u)?(N-c):64u;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
MulAddDst<Z1,Z1,false>(mat_t[c-row_base],v[c],brcb_negative_w,MASK_PLACEHOLDER,rows,
Z6{1,1,0,t_rep,0,1});}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);}}

__aicore__ inline void TridiagonalizeMedium(Z1*work,ZA*tau_mask){
Z1*work_diag=work;Z1*work_off=work+256;
if(n<=120u){
ZA rep_st=Z5<ZA>(Z9/8u);bool lookahead_ready=false;
ZG lookahead_ev=0;for(Z0 k=0;k<n;++k){
Z4<Z1>row_k=ZH[k*Z9];if(k+2u>=n){
if(lookahead_ready){
WaitFlag<HardEvent::V_S>(lookahead_ev);lookahead_ready=false;
}
fence<HardEvent::V_S>();work_diag[k]=row_k.ZF(k);
work_off[k]=(k+1u<n)?row_k.ZF(k+1u):0.0f;tau_mask[k]=0u;
continue;}
Z0 Z2=(k+1u)&~7u;Z0 rem=(k+1u)&7u;
Z0 len_k=n-Z2;if(lookahead_ready){
if(rem!=0u){
Mul(ub_v[Z2],row_k[Z2],ub_mask_ge[rem*8u],8u);Z3<ZC>();
}
WaitFlag<HardEvent::V_S>(lookahead_ev);lookahead_ready=false;
}else{
if(rem==0u){
Mul(ub_prod[Z2],row_k[Z2],row_k[Z2],len_k);Z3<ZC>();
}else{
Mul(ub_v[Z2],row_k[Z2],ub_mask_ge[rem*8u],8u);if(len_k>8u){
Mul(ub_prod[Z2+8u],row_k[Z2+8u],row_k[Z2+8u],len_k-8u);}
Z3<ZC>();Mul(ub_prod[Z2],ub_v[Z2],ub_v[Z2],(len_k<8u)?len_k:8u);
Z3<ZC>();}
if(len_k<=64u){
WholeReduceSum(ub_red,ub_prod[Z2],len_k,1,1,1,8);}else{
ReduceSum(ub_red,ub_prod[Z2],ub_tmp,len_k);}
fence<HardEvent::V_S>();}
Z1 sq_sum=ub_red.ZF(0);Z1 norm=ScalarSqrt(sq_sum);
work_diag[k]=row_k.ZF(k);work_off[k]=0.0f;
ZA tau_k=0u;if(norm!=0.0f){
Z1 x0=row_k.ZF(k+1u)+0.0f;Z1 beta=(x0>=0.0f)?-norm:norm;
work_off[k]=beta;tau_k=1u;
Z1 inv_s=ZI(sq_sum-beta*x0);if(rem==0u){
Muls(ub_v[Z2],row_k[Z2],inv_s,Z9-Z2);}else{
Muls(ub_v[Z2],ub_v[Z2],inv_s,(len_k<8u)?len_k:8u);
if(Z9>Z2+8u)Muls(ub_v[Z2+8u],row_k[Z2+8u],inv_s,Z9-Z2-8u);}
Muls(ub_tmp,ub_mask_eq[rem*8u],-beta*inv_s,8u);Z3<ZC>();
Add(ub_v[Z2],ub_v[Z2],ub_tmp,8u);Z3<ZC>();

Z0 r0=Z2;Z0 ZD=n-r0;
if(Z9<=64u){
Mul(ub_vecs,ZH[r0*Z9+Z2],ub_v[Z2],len_k,ZD,Z6{1,1,1,rep_st,rep_st,0});
Muls(ub_prod[Z2],ub_v[Z2],((Z9&15u)==0u)?0.5f:-0.5f,len_k);if(rem==0u&&Z2>=8u){
Duplicate(ub_refl[k*Z9+Z2-8u],0.0f,8u);}
Adds(ub_refl[k*Z9+Z2],ub_v[Z2],0.0f,Z9-Z2);Z3<ZC>();
WholeReduceSum(ub_w[r0],ub_vecs,len_k,ZD,1,1,rep_st);Z3<ZC>();

Z0 aligned_rows=(ZD+7u)&~7u;ZA brcb_repeats=Z5<ZA>(aligned_rows/8u);
Z4<Z1>brcb_v=ub_gather;Z4<Z1>brcb_w=ub_gather[aligned_rows*8u];
Z0 act_p3=n-(k+1u);Z4<Z1>mat_p3=ZH[(k+1u)*Z9];
Z4<Z1>bv_p3=brcb_v[rem*8u];Z4<Z1>bw_p3=brcb_w[rem*8u];
if((Z9&15u)==0u){
VecDotAxpyNegWMaskBrcbSub(ub_w,ub_v,Z2,rem,0u,brcb_v,r0,brcb_repeats);
Brcb(brcb_w,ub_w[r0],brcb_repeats,{1,8});if(act_p3>8u&&k+2u<n){
MulAddDst(mat_p3[Z2],ub_w[Z2],bv_p3,len_k,8u,Z6{1,1,0,rep_st,0,1});
Z3<ZC>();
MulAddDst(mat_p3[Z2],ub_v[Z2],bw_p3,len_k,8u,Z6{1,1,0,rep_st,0,1});
Z3<ZC>();Z0 c_n=(k+2u)&~7u;
lookahead_ev=IssueLookaheadNormRed(mat_p3,ub_prod[c_n],ub_tmp,c_n,c_n,(k+2u)&7u,n-c_n);
lookahead_ready=true;Z0 rem_p3=act_p3-8u;
Z4<Z1>mat_p3_1=mat_p3[8u*Z9];
MulAddDst(mat_p3_1[Z2],ub_w[Z2],bv_p3[64],len_k,rem_p3,Z6{1,1,0,rep_st,0,1});
Z3<ZC>();
MulAddDst(mat_p3_1[Z2],ub_v[Z2],bw_p3[64],len_k,rem_p3,Z6{1,1,0,rep_st,0,1});
Z3<ZC>();}else{
MulAddDst(mat_p3[Z2],ub_w[Z2],bv_p3,len_k,act_p3,Z6{1,1,0,rep_st,0,1});
Z3<ZC>();
MulAddDst(mat_p3[Z2],ub_v[Z2],bw_p3,len_k,act_p3,Z6{1,1,0,rep_st,0,1});
Z3<ZC>();}
}else{
VecDotAxpyMaskBrcbSub(ub_w,ub_v,Z2,rem,0u,brcb_v,r0,brcb_repeats);
Brcb(brcb_w,ub_w[r0],brcb_repeats,{1,8});if(act_p3>8u&&k+2u<n){
Mul(ub_vecs,ub_w,bv_p3,Z9,8u,Z6{1,1,0,rep_st,0,1});
Mul(ub_vecs[8u*Z9],ub_v,bw_p3,Z9,8u,Z6{1,1,0,rep_st,0,1});Z3<ZC>();
Sub(mat_p3,mat_p3,ub_vecs,8u*Z9);Z3<ZC>();
Sub(mat_p3,mat_p3,ub_vecs[8u*Z9],8u*Z9);Z3<ZC>();
Z0 c_n=(k+2u)&~7u;
lookahead_ev=IssueLookaheadNormRed(mat_p3,ub_prod[c_n],ub_tmp,c_n,c_n,(k+2u)&7u,n-c_n);
lookahead_ready=true;Z0 rem_p3=act_p3-8u;
Z4<Z1>mat_p3_1=mat_p3[8u*Z9];
Mul(ub_vecs,ub_w,bv_p3[64],Z9,rem_p3,Z6{1,1,0,rep_st,0,1});Z3<ZC>();
Sub(mat_p3_1,mat_p3_1,ub_vecs,rem_p3*Z9);
Mul(ub_vecs,ub_v,bw_p3[64],Z9,rem_p3,Z6{1,1,0,rep_st,0,1});Z3<ZC>();
Sub(mat_p3_1,mat_p3_1,ub_vecs,rem_p3*Z9);Z3<ZC>();
}else{
Mul(ub_vecs,ub_w,bv_p3,Z9,act_p3,Z6{1,1,0,rep_st,0,1});Z3<ZC>();
Sub(mat_p3,mat_p3,ub_vecs,act_p3*Z9);
Mul(ub_vecs,ub_v,bw_p3,Z9,act_p3,Z6{1,1,0,rep_st,0,1});Z3<ZC>();
Sub(mat_p3,mat_p3,ub_vecs,act_p3*Z9);Z3<ZC>();
}
}
}else{
Muls(ub_prod[Z2],ub_v[Z2],-0.5f,len_k);if(rem==0u&&Z2>=8u){
Duplicate(ub_refl[k*Z9+Z2-8u],0.0f,8u);}
Adds(ub_refl[k*Z9+Z2],ub_v[Z2],0.0f,Z9-Z2);
MatVecRowsSub(ZH[r0*Z9],ub_v,ub_w[r0],ub_vecs,ZD,Z2);
VecDotAxpyMaskBrcbSub(ub_w,ub_v,Z2,rem,Z9,ub_gather,r0,0u);
Rank2UpdateRowsSub(ZH[r0*Z9],ub_v,ub_w,ub_v[r0],ub_w[r0],ub_vecs,ZD,Z2);}
}
tau_mask[k]=tau_k;}
return;}
}


__aicore__ inline void ExpandTail176AtK79(){
Z4<Z1>flat176=ZH[12288u];Z4<Z1>bounce=ZH[43264u];
for(Z0 c=0u;c<176u;c+=64u){
Z0 width=(176u-c<64u)?(176u-c):64u;
Adds(bounce[c],ZH[12304u+c],0.0f,width,16u,UnaryRepeatParams{1,1,22,24});}
Z3<ZC>();Adds(flat176,bounce,0.0f,2816u);
Z3<ZC>();for(ZG zi=7;zi>=3;--zi){
Z0 z=Z5<Z0>(zi);Z0 z_row=z<<5u;
Z0 stride_z=256u-z_row;Adds(bounce,ZH[zone_offset[z]-6144u],0.0f,32u*stride_z);
Z3<ZC>();Z0 dst_off=(z_row-80u)*176u+(z_row-80u);
for(Z0 c=0u;c<stride_z;c+=64u){
Z0 width=(stride_z-c<64u)?(stride_z-c):64u;Adds(flat176[dst_off+c],bounce[c],0.0f,width,32u,
UnaryRepeatParams{1,1,22,Z5<ZA>(stride_z>>3u)});}
Z3<ZC>();}
Z4<Z0>goff=ZH[48384u].ReinterpretCast<Z0>();
for(Z0 c=0u;c<144u;++c)goff.Z8(c,c*704u);Z3<PIPE_ALL>();
for(Z0 r=16u;r<176u;++r){
Z0 sym_cols=(((r+80u)>>5u)<<5u)-80u;Gather(flat176[r*176u],flat176,goff,r*4u,sym_cols);
}
Z3<PIPE_ALL>();}

__aicore__ inline void RepackTail160AndR80AtK95(){
Z4<Z1>flat176=ZH[12288u];Z4<Z1>flat160=ZH[15360u];
Z4<Z1>bounce=ZH[43264u];for(Z0 r0=0u;r0<160u;r0+=32u){
Z4<Z1>src=flat176[(r0+16u)*176u+16u];
Adds(bounce,src,0.0f,64u,32u,UnaryRepeatParams{1,1,20,22});
Adds(bounce[64u],src[64u],0.0f,64u,32u,UnaryRepeatParams{1,1,20,22});
Adds(bounce[128u],src[128u],0.0f,32u,32u,UnaryRepeatParams{1,1,20,22});Z3<ZC>();
Adds(flat160[r0*160u],bounce,0.0f,5120u);Z3<ZC>();
}
Adds(bounce,flat176,0.0f,2816u);Z3<ZC>();
for(Z0 r=0u;r<16u;++r)Duplicate(ZH[12288u+r*192u],0.0f,16u);for(Z0 c=0u;c<176u;c+=64u){
Z0 width=(176u-c<64u)?(176u-c):64u;
Adds(ZH[12304u+c],bounce[c],0.0f,width,16u,UnaryRepeatParams{1,1,24,22});}
Z3<ZC>();}

__aicore__ __attribute__((noinline)) void TridiagonalizeLarge(uint64_t b,Z1*work,ZA*tau_mask){
Z1*work_diag=work;Z1*work_off=work+256;
if(n<=200u){
if(n==200u){


const Z0 z200_off[11]={0u,1600u,4672u,7488u,10048u,12352u,14400u,16192u,16960u,19072u,21120u};
const Z0 z200_row[11]={0u,8u,24u,40u,56u,72u,88u,104u,112u,136u,168u};Z4<Z1>w_200=ub_row;
Duplicate(w_200,0.0f,200u);Z3<ZC>();
bool lookahead_ready=false;ZG lookahead_ev=0;

for(Z0 k=0;k<200u;++k){
Z0 z_k=(k<8u)?0u:((k<104u)?(1u+((k-8u)>>4u)):((k<112u)?7u:((k<136u)?8u:((k<168u)?9u:10u))));
Z0 z_k_row=z200_row[z_k];Z0 stride_zk=200u-z_k_row;
Z4<Z1>row_k=ZH[z200_off[z_k]+(k-z_k_row)*stride_zk];if(k+2u>=200u){
if(lookahead_ready){
WaitFlag<HardEvent::V_S>(lookahead_ev);lookahead_ready=false;
}
fence<HardEvent::V_S>();work_diag[k]=row_k.ZF(k-z_k_row);
work_off[k]=(k+1u<200u)?row_k.ZF(k+1u-z_k_row):0.0f;tau_mask[k]=0u;
continue;}
Z0 Z2=(k+1u)&~7u;Z0 c_off_k=Z2-z_k_row;
Z0 rem=(k+1u)&7u;Z0 len_k=200u-Z2;
if(lookahead_ready){
if(rem!=0u){
Mul(ub_v[Z2],row_k[c_off_k],ub_mask_ge[rem*8u],8u);
Z3<ZC>();}
WaitFlag<HardEvent::V_S>(lookahead_ev);lookahead_ready=false;
}else{
if(rem==0u){
Adds(ub_v[Z2],row_k[c_off_k],0.0f,len_k);Mul(ub_prod[Z2],row_k[c_off_k],row_k[c_off_k],len_k);
Z3<ZC>();}else{
Mul(ub_v[Z2],row_k[c_off_k],ub_mask_ge[rem*8u],8u);if(200u>Z2+8u){
Adds(ub_v[Z2+8u],row_k[c_off_k+8u],0.0f,200u-Z2-8u);
Mul(ub_prod[Z2+8u],row_k[c_off_k+8u],row_k[c_off_k+8u],len_k-8u);}
Z3<ZC>();Mul(ub_prod[Z2],ub_v[Z2],ub_v[Z2],8u);
Z3<ZC>();}
if(len_k<=64u){WholeReduceSum(ub_red,ub_prod[Z2],len_k,1,1,1,8);}else{ReduceSum(ub_red,ub_prod[Z2],ub_tmp,len_k);}
fence<HardEvent::V_S>();}
Z1 sq_sum=ub_red.ZF(0);Z1 norm=ScalarSqrt(sq_sum);
work_diag[k]=row_k.ZF(k-z_k_row);work_off[k]=0.0f;

ZA tau_k=0u;bool v_norm_done=false;
if(norm!=0.0f){
Z1 x0=row_k.ZF(k+1u-z_k_row)+0.0f;Z1 beta=(x0>=0.0f)?-norm:norm;
work_off[k]=beta;tau_k=1u;
Z1 inv_s=ZI(sq_sum-beta*x0);if(rem==0u){
Muls(ub_v[Z2],row_k[c_off_k],inv_s,200u-Z2);}else{
Muls(ub_v[Z2],ub_v[Z2],inv_s,8u);if(200u>Z2+8u){
Muls(ub_v[Z2+8u],row_k[c_off_k+8u],inv_s,200u-Z2-8u);}}
Muls(ub_red[16],ub_mask_eq[rem*8u],-beta*inv_s,8u);}

if((k<104u&&((k&15u)==7u))||k==111u||k==135u||k==167u){
Z0 next_z=z_k+1u;Z0 new_stride=200u-Z2;
Z0 col_shift=stride_zk-new_stride;Z0 z_base=z200_off[next_z];
ZA dst_rep=Z5<ZA>(new_stride/8u);ZA src_rep=Z5<ZA>(stride_zk/8u);
if(new_stride<=128u){
Z4<Z1>bounce=ZH[(new_stride==128u)?30784u:34000u];
Z4<Z1>src_b=ZH[z_base+col_shift];for(Z0 c=0;c<new_stride;c+=64u){
Z0 w=(new_stride-c<64u)?(new_stride-c):64u;
Adds(bounce[c],src_b[c],0.0f,w,Z5<ZA>(new_stride),UnaryRepeatParams{1,1,dst_rep,src_rep});}
Z3<ZC>();if(norm!=0.0f){
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);v_norm_done=true;
}
Adds(ZH[z_base],bounce,0.0f,new_stride*new_stride);Z3<ZC>();
}else{
Z4<Z1>bounce=ZH[z_base+new_stride*stride_zk];
Z0 b_step=(new_stride==192u)?32u:((new_stride==176u)?44u:((new_stride==160u)?56u:72u));
for(Z0 r0=0;r0<new_stride;r0+=b_step){
Z0 b_rows=(new_stride-r0<b_step)?(new_stride-r0):b_step;
Z4<Z1>src_b=ZH[z_base+r0*stride_zk+col_shift];for(Z0 c=0;c<new_stride;c+=64u){
Z0 w=(new_stride-c<64u)?(new_stride-c):64u;
Adds(bounce[c],src_b[c],0.0f,w,Z5<ZA>(b_rows),UnaryRepeatParams{1,1,dst_rep,src_rep});}
Z3<ZC>();if(r0==0u&&norm!=0.0f){
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);v_norm_done=true;
}
Adds(ZH[z_base+r0*new_stride],bounce,0.0f,b_rows*new_stride);Z3<ZC>();
}
}
}

if(norm!=0.0f){
if(!v_norm_done){
Z3<ZC>();Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);
Z3<ZC>();}

Z0 z_act=(Z2<8u)?0u:((Z2<104u)?(1u+((Z2-8u)>>4u)):((Z2<112u)?7u:((Z2<136u)?8u:((Z2<168u)?9u:10u))));
Z0 row_base=z200_row[z_act];Z0 stride_t=200u-row_base;
ZA t_rep=Z5<ZA>(stride_t/8u);Z4<Z1>ub_tail=ZH[z200_off[z_act]];
Z4<Z1>tail_sc=ub_tail[stride_t*stride_t];

Z0 act_rows_p1=200u-Z2;ZA sc_rep_p1=(z_act==10u)?4u:8u;
Z0 p1_tile=(z_act==0u)?104u:((z_act==1u)?128u:176u);
for(Z0 r_off=0;r_off<act_rows_p1;r_off+=p1_tile){
Z0 cur_rows=(act_rows_p1-r_off<p1_tile)?(act_rows_p1-r_off):p1_tile;
Z4<Z1>mat_t=ub_tail[(Z2+r_off-row_base)*stride_t+(Z2-row_base)];
Z0 width0=(act_rows_p1<64u)?act_rows_p1:64u;
Mul(tail_sc,mat_t,ub_v[Z2],width0,cur_rows,Z6{1,1,1,sc_rep_p1,t_rep,0});
for(Z0 c=Z2+64u;c<200u;c+=64u){
Z0 width=(200u-c<64u)?(200u-c):64u;Z3<ZC>();
MulAddDst(tail_sc,mat_t[c-Z2],ub_v[c],width,cur_rows,Z6{1,1,1,sc_rep_p1,t_rep,0});}
Z3<ZC>();if(width0==64u){
ReduceRows64ByBlock2Stage(w_200[Z2+r_off],tail_sc,cur_rows);}else{
WholeReduceSum(w_200[Z2+r_off],tail_sc,width0,cur_rows,1,1,sc_rep_p1);}
if(r_off+p1_tile>=act_rows_p1){
Muls(ub_prod[Z2],ub_v[Z2],0.5f,act_rows_p1);}
Z3<ZC>();}

Z4<Z1>brcb_v=ZH[43800];Z4<Z1>brcb_w=ZH[45400];
ZA init_b_rep=Z5<ZA>((act_rows_p1+7u)>>3u);

VecDotAxpyNegWMaskBrcbSub(w_200,ub_v,Z2,rem,row_base,brcb_v,Z2,init_b_rep);
Brcb(brcb_w,w_200[Z2],init_b_rep,{1,8});

Z0 total_p3_rows=200u-(k+1u);Z0 max_r0=(z_act<=9u)?1u:32u;
Z0 first_rows=(total_p3_rows<max_r0)?total_p3_rows:max_r0;{
Z0 r0=k+1u;Z4<Z1>mat_t=ub_tail[(r0-row_base)*stride_t];
Z4<Z1>bv_t=brcb_v[rem*8u];Z4<Z1>bw_t=brcb_w[rem*8u];
if(FAST200&&z_act<=9u){Rank2UpdateFlatMulAdd<200u,1u>(mat_t,w_200,ub_v,bv_t,bw_t,row_base,Z2,total_p3_rows);Z3<ZC>();Rank2UpdateFlatMulAdd<200u,2u>(mat_t,w_200,ub_v,bv_t,bw_t,row_base,Z2,first_rows);}else{Rank2UpdateFlatMulAdd<200u>(mat_t,w_200,ub_v,bv_t,bw_t,row_base,Z2,first_rows);}
if(first_rows>=total_p3_rows){
if(rem==0u&&c_off_k>=8u){
Duplicate(row_k[c_off_k-8u],0.0f,8u);}
Adds(row_k[c_off_k],ub_v[Z2],0.0f,200u-Z2);}
Z3<ZC>();}
if(first_rows<total_p3_rows){
if(k+3u<200u){
Z0 c_n=(k+2u)&~7u;
lookahead_ev=IssueLookaheadNormRed(ub_tail[(k+1u-row_base)*stride_t],tail_sc,tail_sc[256],c_n,c_n-row_base,(k+2u)&7u,200u-c_n);
lookahead_ready=true;}
Z0 r1=k+1u+first_rows;Z0 rem_rows=200u-r1;
Z4<Z1>mat_t1=ub_tail[(r1-row_base)*stride_t];Z4<Z1>bv_t1=brcb_v[(r1-Z2)*8u];
Z4<Z1>bw_t1=brcb_w[(r1-Z2)*8u];
if constexpr(FAST200){Rank2UpdateFlatMulAdd<200u,2u>(mat_t1,w_200,ub_v,bv_t1,bw_t1,row_base,Z2,rem_rows);}else{Rank2UpdateFlatMulAdd<200u>(mat_t1,w_200,ub_v,bv_t1,bw_t1,row_base,Z2,rem_rows);}
if(rem==0u&&c_off_k>=8u){
Duplicate(row_k[c_off_k-8u],0.0f,8u);}
Adds(row_k[c_off_k],ub_v[Z2],0.0f,200u-Z2);Z3<ZC>();
}
}
tau_mask[k]=tau_k;}
return;}

Z0 refl_rows=(49152u-1280u)/Z9-n-rows_per_lane-8u;
Z0 scratch_cap=(refl_rows+rows_per_lane)*Z9;ZA rep_stride=Z5<ZA>(Z9/8u);
for(Z0 k=0;k<n;++k){
Z4<Z1>row_k=ZH[k*Z9];if(k+2u>=n){
Z3<PIPE_ALL>();work_diag[k]=row_k.ZF(k);
work_off[k]=(k+1u<n)?row_k.ZF(k+1u):0.0f;tau_mask[k]=0u;
continue;}
Z0 Z2=(k+1u)&~7u;Z0 rem=(k+1u)&7u;
Mul(ub_v[Z2],row_k[Z2],ub_mask_ge[rem*8u],8u);if(Z9>Z2+8u){
Adds(ub_v[Z2+8u],row_k[Z2+8u],0.0f,Z9-Z2-8u);}

Z1 sq_sum=VecDotSub(ub_v,ub_v,Z2);Z1 norm=ScalarSqrt(sq_sum);
work_diag[k]=row_k.ZF(k);work_off[k]=0.0f;
ZA tau_k=0u;if(norm!=0.0f){
Z1 x0=ub_v.ZF(k+1u);Z1 beta=(x0>=0.0f)?-norm:norm;
work_off[k]=beta;tau_k=1u;
Z1 inv_s=ZI(sq_sum-beta*x0);Muls(ub_v[Z2],ub_v[Z2],inv_s,n-Z2);
Muls(ub_red[16],ub_mask_eq[rem*8u],-beta*inv_s,8u);Z3<ZC>();
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);Z3<ZC>();
Muls(ub_prod[Z2],ub_v[Z2],-0.5f,n-Z2);

Z0 r0=Z2;Z0 ZD=n-r0;
Z0 raw_sc=((n-Z2)+7u)&~7u;Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;
ZA ZE=Z5<ZA>(sc_stride/8u);Z0 tile_rows=(scratch_cap/sc_stride)&~7u;
if(tile_rows>64u){
tile_rows=64u;}

for(Z0 r_off=0;r_off<ZD;r_off+=tile_rows){
Z0 cur_rows=(ZD-r_off<tile_rows)?(ZD-r_off):tile_rows;
Z4<Z1>mat_tile=ZH[(r0+r_off)*Z9];for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;
Mul(ub_refl[c-Z2],mat_tile[c],ub_v[c],width,cur_rows,Z6{1,1,1,ZE,rep_stride,0});
}
Z3<ZC>();Z0 s_idx=0u;
for(Z0 c=Z2;c<n;c+=64u,++s_idx){
Z0 width=(n-c<64u)?(n-c):64u;if(s_idx==0u){
WholeReduceSum(ub_w[r0+r_off],ub_refl[c-Z2],width,cur_rows,1,1,ZE);}else{
WholeReduceSum(ub_gather[(s_idx-1u)*Z9+r_off],ub_refl[c-Z2],width,cur_rows,1,1,ZE);}
}
Z3<ZC>();}
Z0 n_strips=(n-Z2+63u)>>6u;if(n_strips==4u){
Add(ub_w[r0],ub_w[r0],ub_gather,ZD);
Add(ub_gather[Z9],ub_gather[Z9],ub_gather[2u*Z9],ZD);Z3<ZC>();
Add(ub_w[r0],ub_w[r0],ub_gather[Z9],ZD);Z3<ZC>();
}else if(n_strips==3u){
Add(ub_w[r0],ub_w[r0],ub_gather,ZD);Z3<ZC>();
Add(ub_w[r0],ub_w[r0],ub_gather[Z9],ZD);Z3<ZC>();
}else if(n_strips==2u){
Add(ub_w[r0],ub_w[r0],ub_gather,ZD);Z3<ZC>();
}

Z4<Z1>brcb_v_base=ub_gather;Z4<Z1>brcb_w_base=ub_gather[512];
Z0 init_b=(ZD<64u)?ZD:64u;ZA init_rep=Z5<ZA>(((init_b+7u)&~7u)/8u);
Z0 brcb_Start=0u;Z0 brcb_End=init_b;

VecDotAxpyMaskBrcbSub(ub_w,ub_v,Z2,rem,Z9,brcb_v_base,r0,init_rep);

for(Z0 r_off=0;r_off<ZD;r_off+=tile_rows){
Z0 cur_rows=(ZD-r_off<tile_rows)?(ZD-r_off):tile_rows;if(r_off==0u){
Brcb(brcb_w_base,ub_w[r0],init_rep,{1,8});}else if(r_off+cur_rows>brcb_End){
brcb_Start=r_off;Z0 b_rows=(ZD-r_off<64u)?(ZD-r_off):64u;
Z0 aligned_b=(b_rows+7u)&~7u;ZA b_rep=Z5<ZA>(aligned_b/8u);
Brcb(brcb_v_base,ub_v[r0+r_off],b_rep,{1,8});Brcb(brcb_w_base,ub_w[r0+r_off],b_rep,{1,8});
Z3<ZC>();brcb_End=r_off+b_rows;
}
Z4<Z1>brcb_v=brcb_v_base[(r_off-brcb_Start)*8u];
Z4<Z1>brcb_w=brcb_w_base[(r_off-brcb_Start)*8u];
Z4<Z1>mat_tile=ZH[(r0+r_off)*Z9];

for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Mul(ub_refl[sc],ub_w[c],brcb_v,width,cur_rows,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(mat_tile[c],mat_tile[c],ub_refl[sc],width,cur_rows,Z6{1,1,1,rep_stride,rep_stride,ZE});
Mul(ub_refl[sc],ub_v[c],brcb_w,width,cur_rows,Z6{1,1,0,ZE,0,1});}
Z3<ZC>();if(Z2==0u&&n==Z9){
Sub(mat_tile,mat_tile,ub_refl,cur_rows*Z9);}else{
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-Z2;
Sub(mat_tile[c],mat_tile[c],ub_refl[sc],width,cur_rows,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
}
Z3<ZC>();}
if(rem==0u&&Z2>=8u){
Duplicate(row_k[Z2-8u],0.0f,8u);}
Adds(row_k[Z2],ub_v[Z2],0.0f,Z9-Z2);Z3<ZC>();
}
tau_mask[k]=tau_k;}
return;}
#if EIGH_K96_PASS


#else
if constexpr(!FAST200){
#endif
uint64_t priv_gm_256=b*65536ull+vec_row_start;Z4<Z1>orig_ub_w=ub_w;
Z4<Z1>ub_tridiag_sc=ub_diag;
Z0 p1_scratch_cap=(15u+rows_per_lane)*Z9+((n==256u)?768u:0u);
Z0 p3_scratch_cap=(13u+rows_per_lane)*Z9;Z4<Z1>ub_vacc=ub_tridiag_sc;
Z0 z_shift=0u;bool lookahead_ready=false;
ZG lookahead_ev=0;

#if EIGH_K96_PASS
for(Z0 k=96u;k<n;++k){
#else
for(Z0 k=0;k<n;++k){
#endif
Z0 z_k=k>>5u;
Z0 z_k_row=(n==256u&&k>=96u)?((k<112u)?96u:((k<128u)?112u:((k<144u)?128u:((k<160u)?144u:((k<192u)?160u:((k<224u)?192u:224u)))))):(z_k<<5u);
Z0 stride_zk=Z9-z_k_row;
Z0 row_k_off=(n==256u&&k>=96u)?(((k<112u)?15360u:((k<128u)?17920u:((k<144u)?20224u:((k<160u)?22272u:((k<192u)?24064u:((k<224u)?27136u:29184u))))))+(k-z_k_row)*stride_zk):(zone_offset[z_k]+(k-z_k_row)*stride_zk-z_shift);
if(n==256u&&k>=80u&&k<=95u){
z_k_row=80u;stride_zk=176u;row_k_off=12288u+(k-80u)*176u;}
Z4<Z1>row_k=ZH[row_k_off];if(k+2u>=n){
if(lookahead_ready){
WaitFlag<HardEvent::V_S>(lookahead_ev);lookahead_ready=false;
}
fence<HardEvent::V_S>();work_diag[k]=row_k.ZF(k-z_k_row);
if(k+1u<n){
work_off[k]=row_k.ZF(k+1u-z_k_row);}else{
work_off[k]=0.0f;}
tau_mask[k]=0u;continue;
}
Z0 Z2=(k+1u)&~7u;Z0 c_off_k=Z2-z_k_row;
Z0 rem=(k+1u)&7u;Z0 len_k=n-Z2;
if(lookahead_ready){
if(rem!=0u){
Mul(ub_v[Z2],row_k[c_off_k],ub_mask_ge[rem*8u],8u);if(n!=256u&&Z9>Z2+8u){
Adds(ub_v[Z2+8u],row_k[c_off_k+8u],0.0f,Z9-Z2-8u);}
Z3<ZC>();}else if(n!=256u){
Adds(ub_v[Z2],row_k[c_off_k],0.0f,Z9-Z2);Z3<ZC>();
}
WaitFlag<HardEvent::V_S>(lookahead_ev);lookahead_ready=false;
}else{
if(rem==0u){
if(n!=256u)Adds(ub_v[Z2],row_k[c_off_k],0.0f,Z9-Z2);
Mul(ub_prod[Z2],row_k[c_off_k],row_k[c_off_k],len_k);Z3<ZC>();
}else{
Mul(ub_v[Z2],row_k[c_off_k],ub_mask_ge[rem*8u],8u);if(n!=256u&&Z9>Z2+8u){
Adds(ub_v[Z2+8u],row_k[c_off_k+8u],0.0f,Z9-Z2-8u);}
if(len_k>8u){
Mul(ub_prod[Z2+8u],row_k[c_off_k+8u],row_k[c_off_k+8u],len_k-8u);}
Z3<ZC>();Mul(ub_prod[Z2],ub_v[Z2],ub_v[Z2],(len_k<8u)?len_k:8u);
Z3<ZC>();}
if(len_k<=64u){WholeReduceSum(ub_red,ub_prod[Z2],len_k,1,1,1,8);}else{ReduceSum(ub_red,ub_prod[Z2],ub_tmp,len_k);}
fence<HardEvent::V_S>();}
Z1 sq_sum=ub_red.ZF(0);Z1 norm=ScalarSqrt(sq_sum);
work_diag[k]=row_k.ZF(k-z_k_row);work_off[k]=0.0f;
ZA tau_k=0u;

if(n==256u&&k>=79u){
bool v_norm_done=false;if(norm!=0.0f&&k!=79u&&k!=95u){
Z1 x0=row_k.ZF(k+1u-z_k_row)+0.0f;Z1 beta=(x0>=0.0f)?-norm:norm;
work_off[k]=beta;tau_k=1u;
Z1 inv_s=ZI(sq_sum-beta*x0);
if(rem==0u){Muls(ub_v[Z2],row_k[c_off_k],inv_s,256u-Z2);}else{Muls(ub_v[Z2],ub_v[Z2],inv_s,8u);if(256u>Z2+8u)Muls(ub_v[Z2+8u],row_k[c_off_k+8u],inv_s,256u-Z2-8u);}
Muls(ub_red[16],ub_mask_eq[rem*8u],-beta*inv_s,8u);}
if(k==79u){
if(norm!=0.0f){
Z1 x0=row_k.ZF(k+1u-z_k_row)+0.0f;Z1 beta=(x0>=0.0f)?-norm:norm;
work_off[k]=beta;tau_k=1u;Z1 inv_s=ZI(sq_sum-beta*x0);
Muls(ub_v[Z2],row_k[c_off_k],inv_s,176u);Muls(ub_red[16],ub_mask_eq[0],-beta*inv_s,8u);
Z3<ZC>();Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);
Z3<ZC>();v_norm_done=true;
Duplicate(row_k[c_off_k-8u],0.0f,8u);Adds(row_k[c_off_k],ub_v[Z2],0.0f,176u);
Z3<ZC>();}
ExpandTail176AtK79();}else if(k==95u){
if(norm!=0.0f){
Z1 x0=row_k.ZF(k+1u-z_k_row)+0.0f;Z1 beta=(x0>=0.0f)?-norm:norm;
work_off[k]=beta;tau_k=1u;
Z1 inv_s=ZI(sq_sum-beta*x0);Muls(ub_v[Z2],row_k[c_off_k],inv_s,160u);
Muls(ub_red[16],ub_mask_eq[0],-beta*inv_s,8u);Z3<ZC>();
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);Z3<ZC>();
v_norm_done=true;Duplicate(row_k[c_off_k-8u],0.0f,8u);
Adds(row_k[c_off_k],ub_v[Z2],0.0f,160u);Z3<ZC>();
}
RepackTail160AndR80AtK95();}else if(k==111u){
Z4<Z1>bounce=ZH[40960];for(Z0 r0=0;r0<144u;r0+=48u){
Z4<Z1>src_b=ZH[17920u+r0*160u+16u];
Adds(bounce,src_b,0.0f,64u,48u,UnaryRepeatParams{1,1,18,20});
Adds(bounce[64],src_b[64],0.0f,64u,48u,UnaryRepeatParams{1,1,18,20});
Adds(bounce[128],src_b[128],0.0f,16u,48u,UnaryRepeatParams{1,1,18,20});Z3<ZC>();
if(r0==0u&&norm!=0.0f){
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);v_norm_done=true;
}
Adds(ZH[17920u+r0*144u],bounce,0.0f,6912u);Z3<ZC>();
}
}else if(k==127u){
Z4<Z1>bounce=ZH[38656];for(Z0 r0=0;r0<128u;r0+=64u){
Z4<Z1>src_b=ZH[20224u+r0*144u+16u];
Adds(bounce,src_b,0.0f,64u,64u,UnaryRepeatParams{1,1,16,18});
Adds(bounce[64],src_b[64],0.0f,64u,64u,UnaryRepeatParams{1,1,16,18});Z3<ZC>();
if(r0==0u&&norm!=0.0f){
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);v_norm_done=true;
}
Adds(ZH[20224u+r0*128u],bounce,0.0f,8192u);Z3<ZC>();
}
}else if(k==143u){
Z4<Z1>bounce=ZH[36608];for(Z0 r0=0;r0<112u;r0+=56u){
Z4<Z1>src_b=ZH[22272u+r0*128u+16u];
Adds(bounce,src_b,0.0f,64u,56u,UnaryRepeatParams{1,1,14,16});
Adds(bounce[64],src_b[64],0.0f,48u,56u,UnaryRepeatParams{1,1,14,16});Z3<ZC>();
if(r0==0u&&norm!=0.0f){
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);v_norm_done=true;
}
Adds(ZH[22272u+r0*112u],bounce,0.0f,6272u);Z3<ZC>();
}
}else if(k==159u){
Z4<Z1>bounce=ZH[34816];Z4<Z1>src_b=ZH[24064u+16u];
Adds(bounce,src_b,0.0f,64u,96u,UnaryRepeatParams{1,1,12,14});
Adds(bounce[64],src_b[64],0.0f,32u,96u,UnaryRepeatParams{1,1,12,14});Z3<ZC>();
if(norm!=0.0f){
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);v_norm_done=true;
}
Adds(ZH[24064],bounce,0.0f,4608u);Adds(ZH[24064u+4608u],bounce[4608],0.0f,4608u);
Z3<ZC>();}else if(k==191u){
Adds(ZH[34816],ZH[24064u+32u*96u+32u],0.0f,64u,64u,UnaryRepeatParams{1,1,8,12});
Z3<ZC>();if(norm!=0.0f){
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);v_norm_done=true;
}
Adds(ZH[27136],ZH[34816],0.0f,4096u);Z3<ZC>();
}else if(k==223u){
Adds(ZH[34816],ZH[27136u+32u*64u+32u],0.0f,32u,32u,UnaryRepeatParams{1,1,4,8});
Z3<ZC>();if(norm!=0.0f){
Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);v_norm_done=true;
}
Adds(ZH[29184],ZH[34816],0.0f,1024u);Z3<ZC>();
}

if(norm!=0.0f){
if(!v_norm_done){
Z3<ZC>();Add(ub_v[Z2],ub_v[Z2],ub_red[16],8u);
Z3<ZC>();}

Z0 row_base=(Z2<96u)?80u:((Z2<112u)?96u:((Z2<128u)?112u:((Z2<144u)?128u:((Z2<160u)?144u:((Z2<192u)?160u:((Z2<224u)?192u:224u))))));
Z0 stride_t=256u-row_base;ZA t_rep=Z5<ZA>(stride_t/8u);
Z0 tail_off=(Z2<96u)?12288u:((Z2<112u)?15360u:((Z2<128u)?17920u:((Z2<144u)?20224u:((Z2<160u)?22272u:((Z2<192u)?24064u:((Z2<224u)?27136u:29184u))))));
Z4<Z1>ub_tail=ZH[tail_off];Z4<Z1>tail_sc=ub_tail[stride_t*stride_t];
Z4<Z1>w_tail=ub_gather[768];

Z0 cur_cols=256u-Z2;ZA sc_rep_p1=(row_base==224u)?4u:8u;

if(Z2<144u){
Z0 p1_tile=(Z2<96u)?80u:((Z2<112u)?96u:((Z2<128u)?144u:128u));
for(Z0 r0=Z2;r0<256u;r0+=p1_tile){
Z0 act_r=(256u-r0<p1_tile)?(256u-r0):p1_tile;
Z4<Z1>mat_t=ub_tail[(r0-row_base)*stride_t+(Z2-row_base)];
Mul(tail_sc,mat_t,ub_v[Z2],64u,act_r,Z6{1,1,1,8,t_rep,0});Z3<ZC>();
MulAddDst(tail_sc,mat_t[64],ub_v[Z2+64u],(cur_cols>128u)?64u:(cur_cols-64u),act_r,Z6{1,1,1,8,t_rep,0});
if(cur_cols>128u){
Z3<ZC>();
MulAddDst(tail_sc,mat_t[128],ub_v[Z2+128u],cur_cols-128u,act_r,Z6{1,1,1,8,t_rep,0});}
Z3<ZC>();ReduceRows64ByBlock2Stage(w_tail[r0],tail_sc,act_r);
if(r0+p1_tile>=256u){
Muls(ub_prod[Z2],ub_v[Z2],0.5f,cur_cols);}
Z3<ZC>();}

Z4<Z1>brcb_v=(Z2<96u)?tail_sc[512u]:ZH[44400];
Z4<Z1>brcb_w=(Z2<96u)?tail_sc[1920u]:ZH[45680];
ZA b_rep=Z5<ZA>((cur_cols+7u)>>3u);

VecDotAxpyNegWMaskBrcbSub(w_tail,ub_v,Z2,rem,row_base,brcb_v,Z2,b_rep);
Brcb(brcb_w,w_tail[Z2],b_rep,{1,8});

Z0 total_p3_rows=256u-(k+1u);Z0 max_r0=1u;
Z0 first_rows=(total_p3_rows<max_r0)?total_p3_rows:max_r0;{
Z0 r0=k+1u;Z4<Z1>mat_t=ub_tail[(r0-row_base)*stride_t];
Z4<Z1>bv_t=brcb_v[rem*8u];Z4<Z1>bw_t=brcb_w[rem*8u];
Rank2UpdateFlatMulAdd<256u,1u>(mat_t,w_tail,ub_v,bv_t,bw_t,row_base,Z2,total_p3_rows);Z3<ZC>();
Rank2UpdateFlatMulAdd<256u,2u>(mat_t,w_tail,ub_v,bv_t,bw_t,row_base,Z2,first_rows);Z3<ZC>();
}
if(first_rows<total_p3_rows){
if(k+3u<256u){
Z0 c_n=(k+2u)&~7u;
lookahead_ev=IssueLookaheadNormRed(ub_tail[(k+1u-row_base)*stride_t],tail_sc,tail_sc[256],c_n,c_n-row_base,(k+2u)&7u,256u-c_n);
lookahead_ready=true;}
Z0 r1=k+1u+first_rows;Z0 rem_rows=256u-r1;
Z4<Z1>mat_t1=ub_tail[(r1-row_base)*stride_t];Z4<Z1>bv_t1=brcb_v[(r1-Z2)*8u];
Z4<Z1>bw_t1=brcb_w[(r1-Z2)*8u];
Rank2UpdateFlatMulAdd<256u,2u>(mat_t1,w_tail,ub_v,bv_t1,bw_t1,row_base,Z2,rem_rows);if(k!=79u&&k!=95u){
if(rem==0u&&c_off_k>=8u){
Duplicate(row_k[c_off_k-8u],0.0f,8u);}
Adds(row_k[c_off_k],ub_v[Z2],0.0f,cur_cols);}
Z3<ZC>();}
}else{
Z0 width0_p1=(cur_cols<64u)?cur_cols:64u;
Z4<Z1>mat_p1=ub_tail[(Z2-row_base)*(stride_t+1u)];
Mul(tail_sc,mat_p1,ub_v[Z2],width0_p1,cur_cols,Z6{1,1,1,sc_rep_p1,t_rep,0});
if(cur_cols>64u){
Z3<ZC>();
MulAddDst(tail_sc,mat_p1[64],ub_v[Z2+64u],cur_cols-64u,cur_cols,Z6{1,1,1,sc_rep_p1,t_rep,0});
}
Z3<ZC>();
if(width0_p1==64u){ReduceRows64ByBlock2Stage(w_tail[Z2],tail_sc,cur_cols);}else{WholeReduceSum(w_tail[Z2],tail_sc,width0_p1,cur_cols,1,1,sc_rep_p1);}
Muls(ub_prod[Z2],ub_v[Z2],0.5f,cur_cols);Z3<ZC>();

Z4<Z1>brcb_v=ZH[44400];Z4<Z1>brcb_w=ZH[45680];
ZA b_rep=Z5<ZA>(cur_cols/8u);

VecDotAxpyNegWMaskBrcbSub(w_tail,ub_v,Z2,rem,row_base,brcb_v,Z2,b_rep);

Z0 r0=k+1u;Z0 total_p3_rows=256u-r0;
Z0 max_r0=(Z2<224u)?1u:32u;Z0 first_rows=(total_p3_rows<max_r0)?total_p3_rows:max_r0;
Brcb(brcb_w,w_tail[Z2],b_rep,{1,8});Z4<Z1>mat_t=ub_tail[(r0-row_base)*stride_t];
Z4<Z1>bv_t=brcb_v[rem*8u];Z4<Z1>bw_t=brcb_w[rem*8u];
if(first_rows<total_p3_rows){Rank2UpdateFlatMulAdd<256u,1u>(mat_t,w_tail,ub_v,bv_t,bw_t,row_base,Z2,total_p3_rows);Z3<ZC>();Rank2UpdateFlatMulAdd<256u,2u>(mat_t,w_tail,ub_v,bv_t,bw_t,row_base,Z2,first_rows);}else{Rank2UpdateFlatMulAdd<256u>(mat_t,w_tail,ub_v,bv_t,bw_t,row_base,Z2,first_rows);}
if(first_rows>=total_p3_rows){
if(rem==0u&&c_off_k>=8u){
Duplicate(row_k[c_off_k-8u],0.0f,8u);}
Adds(row_k[c_off_k],ub_v[Z2],0.0f,cur_cols);}
Z3<ZC>();if(first_rows<total_p3_rows){
if(k+3u<256u){
Z0 c_n=(k+2u)&~7u;
lookahead_ev=IssueLookaheadNormRed(mat_t,tail_sc,tail_sc[256],c_n,c_n-row_base,(k+2u)&7u,256u-c_n);
lookahead_ready=true;}
Z0 r1=r0+first_rows;Z0 rem_rows=256u-r1;
Z4<Z1>mat_t1=ub_tail[(r1-row_base)*stride_t];Z4<Z1>bv_t1=brcb_v[(r1-Z2)*8u];
Z4<Z1>bw_t1=brcb_w[(r1-Z2)*8u];
Rank2UpdateFlatMulAdd<256u,2u>(mat_t1,w_tail,ub_v,bv_t1,bw_t1,row_base,Z2,rem_rows);
if(rem==0u&&c_off_k>=8u){
Duplicate(row_k[c_off_k-8u],0.0f,8u);}
Adds(row_k[c_off_k],ub_v[Z2],0.0f,cur_cols);Z3<ZC>();
}
}
}
tau_mask[k]=tau_k;continue;
}
if(norm!=0.0f){
Z1 x0=row_k.ZF(k+1u-z_k_row)+0.0f;Z1 beta=(x0>=0.0f)?-norm:norm;
work_off[k]=beta;tau_k=1u;
Z1 inv_s=ZI(sq_sum-beta*x0);
if(n==256u){if(rem==0u){Muls(ub_v[Z2],row_k[c_off_k],inv_s,256u-Z2);}else{Muls(ub_v[Z2],ub_v[Z2],inv_s,8u);if(256u>Z2+8u)Muls(ub_v[Z2+8u],row_k[c_off_k+8u],inv_s,256u-Z2-8u);}}else{Muls(ub_v[Z2],ub_v[Z2],inv_s,n-Z2);}
Muls(ub_tmp,ub_mask_eq[rem*8u],-beta*inv_s,8u);Z3<ZC>();
Add(ub_v[Z2],ub_v[Z2],ub_tmp,8u);Z3<ZC>();
if(n==256u&&(k==15u||k==23u)){
Duplicate(row_k[c_off_k-8u],0.0f,8u);Adds(row_k[c_off_k],ub_v[Z2],0.0f,256u-Z2);
}
}

if(n==256u&&k==15u){
Z3<PIPE_ALL>();DataCopyPad(
gm_eigvecs[priv_gm_256],
ZH,
DataCopyExtParams{256u,64u,0,960u,0});Z3<PIPE_ALL>();
for(Z0 s_off=0;s_off<32768u;s_off+=4096u){
Adds(ZH[s_off],ZH[s_off+4096u],0.0f,4096u);Z3<ZC>();
}
Duplicate(ZH[32768],0.0f,256u);Z3<ZC>();
z_shift=4096u;ub_w=ZH[32768];
ub_tridiag_sc=ZH[33024];ub_vacc=ub_tridiag_sc;
p1_scratch_cap=15360u;p3_scratch_cap=12032u;
}else if(n==256u&&k==23u){
Z3<PIPE_ALL>();DataCopyPad(
gm_eigvecs[priv_gm_256+16u],
ZH,
DataCopyExtParams{256u,32u,0,992u,0});Z3<PIPE_ALL>();
for(Z0 s_off=0;s_off<30720u;s_off+=2048u){
Adds(ZH[s_off],ZH[s_off+2048u],0.0f,2048u);Z3<ZC>();
}
Duplicate(ZH[30720],0.0f,256u);Z3<ZC>();
z_shift=6144u;ub_w=ZH[30720];
ub_tridiag_sc=ZH[30976];ub_vacc=ub_tridiag_sc;
p1_scratch_cap=17408u;p3_scratch_cap=14080u;
}

if(norm!=0.0f){
Z0 z_start=Z2>>5u;

Z0 p23_z_start=(k+1u)>>5u;Z0 first_z_next=(p23_z_start+1u)<<5u;
bool has_p2=(first_z_next<n);Z0 v_p2_rows=has_p2?(((num_zones-1u)<<5u)-Z2):0u;
Z0 aligned_p2_rows=(v_p2_rows+7u)&~7u;Z0 brcb_p2_floats=aligned_p2_rows*8u;
Z0 p2_total_cap=((19u+rows_per_lane)*Z9+z_shift)-brcb_p2_floats;
Z4<Z1>brcb_v_p2=ub_tridiag_sc[p2_total_cap];
Z0 raw_vacc_st=has_p2?(Z9-first_z_next):64u;
Z0 vacc_stride=(n==Z9)?raw_vacc_st:((raw_vacc_st>=64u)?raw_vacc_st:64u);
ZA vacc_rep_stride=Z5<ZA>(vacc_stride/8u);
Z0 base_vr=(p23_z_start>=1u&&n==256u)?32u:16u;
Z0 vacc_rows=(v_p2_rows<base_vr)?aligned_p2_rows:base_vr;Z0 vacc_floats=vacc_rows*vacc_stride;
Z0 first_act_rows=has_p2?(first_z_next-(k+1u)):0u;bool p2_brcb_issued=false;

if(n==256u){

Z0 zb0=z_start;while(zb0<8u){
Z0 zb1=(k<15u&&zb0==0u)?5u:8u;Z0 b_r0=(Z2>(zb0<<5u))?Z2:(zb0<<5u);
Z0 max_strips=(256u-b_r0+63u)>>6u;Z0 last_p1_mask=0u;
SetMaskNorm();for(Z0 s_idx=0u;s_idx<max_strips;++s_idx){
Z0 c_off_s=s_idx<<6u;for(Z0 z=zb0;z<zb1;++z){
Z0 z_row=z<<5u;Z0 r_start=(Z2>z_row)?Z2:z_row;
Z0 c=r_start+c_off_s;if(c>=256u)continue;
Z0 ZD=(z_row+32u)-r_start;Z0 width=(256u-c<64u)?(256u-c):64u;
Z0 stride_z=256u-z_row;ZA rep_stride=Z5<ZA>(stride_z>>3u);
ZA sc_rep=(z==7u)?4u:8u;
Z4<Z1>mat_z=ZH[zone_offset[z]+(r_start-z_row)*stride_z-z_shift];
Z4<Z1>sc_z=ub_tridiag_sc[(r_start-b_r0)*64u];
if(width!=last_p1_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_p1_mask=width;}
if(s_idx==0u){
Mul<Z1,false>(sc_z,mat_z[c-z_row],ub_v[c],MASK_PLACEHOLDER,ZD,Z6{1,1,1,sc_rep,rep_stride,0});
}else{
MulAddDst<Z1,Z1,false>(sc_z,mat_z[c-z_row],ub_v[c],MASK_PLACEHOLDER,ZD,Z6{1,1,1,sc_rep,rep_stride,0});
}
}
Z3<ZC>();}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);if(zb1<=7u){
Z0 rows64=(zb1<<5u)-b_r0;ReduceRows64ByBlock2Stage(ub_w[b_r0],ub_tridiag_sc,rows64);
}else{
Z0 rows64=224u-b_r0;if(rows64>0u){
ReduceRows64ByBlock2Stage(ub_w[b_r0],ub_tridiag_sc,rows64);}
WholeReduceSum(ub_w[224],ub_tridiag_sc[rows64*64u],32u,32u,1,1,4);if(has_p2){
Brcb(brcb_v_p2,ub_v[Z2],Z5<ZA>(aligned_p2_rows/8u),{1,8});p2_brcb_issued=true;
}
}
Z3<ZC>();zb0=zb1;
}
}else{

Z4<Z1>s1_buf=ub_gather;Z4<Z1>s2_buf=ub_gather[Z9];
Z4<Z1>s3_buf=ub_gather[2u*Z9];for(Z0 zb0=z_start;zb0<num_zones;){
Z0 zb1=zb0;Z0 used_sc=0u;
while(zb1<num_zones){
Z0 z_row=zb1<<5u;Z0 r_start=(Z2>z_row)?Z2:z_row;
Z0 r_end=(z_row+32u<n)?(z_row+32u):n;Z0 ZD=r_end-r_start;
Z0 raw_sc=((n-r_start)+7u)&~7u;
Z0 sc_stride=(n==Z9&&r_start==z_row)?(Z9-z_row):((raw_sc>=64u)?raw_sc:64u);
if(zb1>zb0&&used_sc+ZD*sc_stride>p1_scratch_cap){
break;}
used_sc+=ZD*sc_stride;++zb1;
}
Z0 sc_off=0u;for(Z0 z=zb0;z<zb1;++z){
Z0 z_row=z<<5u;Z0 stride_z=Z9-z_row;
Z0 r_start=(Z2>z_row)?Z2:z_row;Z0 r_end=(z_row+32u<n)?(z_row+32u):n;
Z0 ZD=r_end-r_start;Z0 raw_sc=((n-r_start)+7u)&~7u;
Z0 sc_stride=(n==Z9&&r_start==z_row)?stride_z:((raw_sc>=64u)?raw_sc:64u);
ZA rep_stride=Z5<ZA>(stride_z/8u);
ZA ZE=Z5<ZA>(sc_stride/8u);
Z4<Z1>mat_z=ZH[zone_offset[z]+(r_start-z_row)*stride_z-z_shift];
Z4<Z1>sc_z=ub_tridiag_sc[sc_off];for(Z0 c=r_start;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;
Mul(sc_z[c-r_start],mat_z[c-z_row],ub_v[c],width,ZD,Z6{1,1,1,ZE,rep_stride,0});
}
sc_off+=ZD*sc_stride;}
Z3<ZC>();sc_off=0u;
for(Z0 z=zb0;z<zb1;++z){
Z0 z_row=z<<5u;Z0 r_start=(Z2>z_row)?Z2:z_row;
Z0 r_end=(z_row+32u<n)?(z_row+32u):n;Z0 ZD=r_end-r_start;
Z0 raw_sc=((n-r_start)+7u)&~7u;
Z0 sc_stride=(n==Z9&&r_start==z_row)?(Z9-z_row):((raw_sc>=64u)?raw_sc:64u);
ZA ZE=Z5<ZA>(sc_stride/8u);Z4<Z1>sc_z=ub_tridiag_sc[sc_off];
Z0 strip_idx=0u;for(Z0 c=r_start;c<n;c+=64u,++strip_idx){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-r_start;
if(strip_idx==0u){
WholeReduceSum(ub_w[r_start],sc_z[sc],width,ZD,1,1,ZE);}else if(strip_idx==1u){
WholeReduceSum(s1_buf[r_start],sc_z[sc],width,ZD,1,1,ZE);}else if(strip_idx==2u){
WholeReduceSum(s2_buf[r_start],sc_z[sc],width,ZD,1,1,ZE);}else{
WholeReduceSum(s3_buf[r_start],sc_z[sc],width,ZD,1,1,ZE);}
}
sc_off+=ZD*sc_stride;}
Z3<ZC>();zb0=zb1;
}
if(n>Z2+64u){
Z0 s1_end=(n>64u)?(((n-64u+31u)&~31u)<n?((n-64u+31u)&~31u):n):Z2;if(s1_end>Z2){
Z0 s1_len=s1_end-Z2;if(n>Z2+192u){
Z0 s2_end=((n-128u+31u)&~31u)<n?((n-128u+31u)&~31u):n;
Z0 s3_end=((n-192u+31u)&~31u)<n?((n-192u+31u)&~31u):n;
Add(ub_w[Z2],ub_w[Z2],s1_buf[Z2],s1_len);
Add(s2_buf[Z2],s2_buf[Z2],s3_buf[Z2],s3_end-Z2);Z3<ZC>();
Add(ub_w[Z2],ub_w[Z2],s2_buf[Z2],s2_end-Z2);}else if(n>Z2+128u){
Z0 s2_end=((n-128u+31u)&~31u)<n?((n-128u+31u)&~31u):n;
Add(ub_w[Z2],ub_w[Z2],s1_buf[Z2],s1_len);Z3<ZC>();
Add(ub_w[Z2],ub_w[Z2],s2_buf[Z2],s2_end-Z2);}else{
Add(ub_w[Z2],ub_w[Z2],s1_buf[Z2],s1_len);}
if(has_p2){
Brcb(brcb_v_p2,ub_v[Z2],Z5<ZA>(aligned_p2_rows/8u),{1,8});
if(first_act_rows+1u<vacc_rows){
Duplicate(ub_vacc[(first_act_rows+1u)*vacc_stride],0.0f,(vacc_rows-first_act_rows-1u)*vacc_stride);}
p2_brcb_issued=true;}
Z3<ZC>();}
}
}


if(has_p2&&n==256u&&k<15u){
Z0 v0_stride=224u;ZA v0_rep=28u;
Z4<Z1>ub_vacc0=ub_tridiag_sc;Z4<Z1>ub_vacc1=ub_tridiag_sc[7168];
if(!p2_brcb_issued){
Brcb(brcb_v_p2,ub_v[Z2],Z5<ZA>(aligned_p2_rows/8u),{1,8});Z3<ZC>();
}
if(first_act_rows+1u<32u){
Duplicate(ub_vacc0[(first_act_rows+1u)*224u],0.0f,(32u-first_act_rows-1u)*224u);}
{
Z4<Z1>mat_z0=ZH[(k+1u)*256u];Z4<Z1>brcb_z0=brcb_v_p2[((k+1u)-Z2)*8u];
for(Z0 c=32u;c<256u;c+=64u){
Z0 width=(256u-c<64u)?(256u-c):64u;
Mul(ub_vacc0[c-32u],mat_z0[c],brcb_z0,width,first_act_rows,Z6{1,1,0,v0_rep,32,1});}
if(first_act_rows<32u){
Adds(ub_vacc0[first_act_rows*224u],ub_w[32],0.0f,224u);}
Z4<Z1>mat_z6=ZH[zone_offset[6]];Z4<Z1>brcb_z6=brcb_v_p2[(192u-Z2)*8u];
Mul(ub_vacc1,mat_z6[32],brcb_z6,32u,32u,Z6{1,1,0,4,8,1});Z3<ZC>();
}
for(Z0 z=1u;z<6u;++z){
Z0 z_row=z<<5u;Z0 z_next=z_row+32u;
Z0 stride_z=256u-z_row;ZA rep_z=Z5<ZA>(stride_z>>3u);
Z4<Z1>mat_z=ZH[zone_offset[z]];Z4<Z1>brcb_z=brcb_v_p2[(z_row-Z2)*8u];
for(Z0 c=z_next;c<256u;c+=64u){
Z0 width=(256u-c<64u)?(256u-c):64u;
MulAddDst(ub_vacc0[c-32u],mat_z[c-z_row],brcb_z,width,32u,Z6{1,1,0,v0_rep,rep_z,1});}
if(z==1u){
Add(ub_vacc1,ub_vacc1,ub_vacc1[16u*32u],16u*32u);}else if(z==2u){
Add(ub_vacc1,ub_vacc1,ub_vacc1[8u*32u],8u*32u);}else if(z==3u){
Add(ub_vacc1,ub_vacc1,ub_vacc1[4u*32u],4u*32u);}else if(z==4u){
Add(ub_vacc1,ub_vacc1,ub_vacc1[2u*32u],2u*32u);}else if(z==5u){
Add(ub_vacc1,ub_vacc1,ub_vacc1[32u],32u);}
Z3<ZC>();}
SetMaskNorm();SetVectorMask<Z1,MaskMode::NORMAL>(64u);for(Z0 half=16u;half>=2u;half>>=1u){
Add<Z1,false>(ub_vacc0,ub_vacc0,ub_vacc0[half*224u],MASK_PLACEHOLDER,
Z5<ZA>((half*224u)>>6u),Z6{1,1,1,8,8,8});Z3<ZC>();
}
Add(ub_w[32],ub_vacc0,ub_vacc0[224u],192u);Add(ub_vacc0[192],ub_vacc0[192],ub_vacc0[224u+192u],32u);
Muls(ub_prod[Z2],ub_v[Z2],0.5f,256u-Z2);Z3<ZC>();
Add(ub_w[224],ub_vacc0[192],ub_vacc1,32u);Z3<ZC>();
}else if(has_p2&&n==256u&&k>=15u){
Z0 v0_stride=vacc_stride;Z0 v1_stride=vacc_stride-32u;
ZA v0_rep=Z5<ZA>(v0_stride>>3u);ZA v1_rep=Z5<ZA>(v1_stride>>3u);
Z4<Z1>ub_vacc0=ub_tridiag_sc;Z4<Z1>ub_vacc1=ub_tridiag_sc[32u*v0_stride];
if(!p2_brcb_issued){
Brcb(brcb_v_p2,ub_v[Z2],Z5<ZA>(aligned_p2_rows/8u),{1,8});Z3<ZC>();
}
if(first_act_rows+1u<32u){
Duplicate(ub_vacc0[(first_act_rows+1u)*v0_stride],0.0f,(32u-first_act_rows-1u)*v0_stride);}
{
Z0 z0=p23_z_start;Z0 z0_row=z0<<5u;
Z0 stride_z0=256u-z0_row;ZA rep_z0=Z5<ZA>(stride_z0>>3u);
Z4<Z1>mat_z0=ZH[zone_offset[z0]+((k+1u)-z0_row)*stride_z0-z_shift];
Z4<Z1>brcb_z0=brcb_v_p2[((k+1u)-Z2)*8u];{SetMaskNorm();Z0 last_mask=0u;
for(Z0 c=first_z_next;c<256u;c+=64u){
Z0 width=(256u-c<64u)?(256u-c):64u;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
Mul<Z1,false>(ub_vacc0[c-first_z_next],mat_z0[c-z0_row],brcb_z0,MASK_PLACEHOLDER,first_act_rows,Z6{1,1,0,v0_rep,rep_z0,1});
}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);}

if(first_act_rows<32u){
Adds(ub_vacc0[first_act_rows*v0_stride],ub_w[first_z_next],0.0f,v0_stride);}
Z0 z1=p23_z_start+1u;Z0 z1_row=z1<<5u;
Z0 z1_next=z1_row+32u;Z0 stride_z1=256u-z1_row;
ZA rep_z1=Z5<ZA>(stride_z1>>3u);
Z4<Z1>mat_z1=ZH[zone_offset[z1]-z_shift];
Z4<Z1>brcb_z1=brcb_v_p2[(z1_row-Z2)*8u];{SetMaskNorm();Z0 last_mask=0u;
for(Z0 c=z1_next;c<256u;c+=64u){
Z0 width=(256u-c<64u)?(256u-c):64u;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
Mul<Z1,false>(ub_vacc1[c-z1_next],mat_z1[c-z1_row],brcb_z1,MASK_PLACEHOLDER,32u,Z6{1,1,0,v1_rep,rep_z1,1});
}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);}

Z3<ZC>();}
bool v1_folded16=false;for(Z0 zp=p23_z_start+2u;zp<7u;zp+=2u){
Z0 ze_row=zp<<5u;Z0 ze_next=ze_row+32u;
Z0 stride_ze=256u-ze_row;ZA rep_ze=Z5<ZA>(stride_ze>>3u);
Z4<Z1>mat_ze=ZH[zone_offset[zp]-z_shift];
Z4<Z1>brcb_ze=brcb_v_p2[(ze_row-Z2)*8u];{SetMaskNorm();Z0 last_mask=0u;
for(Z0 c=ze_next;c<256u;c+=64u){
Z0 width=(256u-c<64u)?(256u-c):64u;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
MulAddDst<Z1,Z1,false>(ub_vacc0[c-first_z_next],mat_ze[c-ze_row],brcb_ze,MASK_PLACEHOLDER,32u,Z6{1,1,0,v0_rep,rep_ze,1});
}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);}

if(zp+1u<7u){
Z0 zo=zp+1u;Z0 zo_row=zo<<5u;
Z0 zo_next=zo_row+32u;Z0 stride_zo=256u-zo_row;
ZA rep_zo=Z5<ZA>(stride_zo>>3u);
Z4<Z1>mat_zo=ZH[zone_offset[zo]-z_shift];
Z4<Z1>brcb_zo=brcb_v_p2[(zo_row-Z2)*8u];{SetMaskNorm();Z0 last_mask=0u;
for(Z0 c=zo_next;c<256u;c+=64u){
Z0 width=(256u-c<64u)?(256u-c):64u;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
MulAddDst<Z1,Z1,false>(ub_vacc1[c-(first_z_next+32u)],mat_zo[c-zo_row],brcb_zo,MASK_PLACEHOLDER,32u,Z6{1,1,0,v1_rep,rep_zo,1});
}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);}

}else{
Add(ub_vacc1,ub_vacc1,ub_vacc1[16u*v1_stride],16u*v1_stride);v1_folded16=true;
}
Z3<ZC>();}
SetMaskNorm();SetVectorMask<Z1,MaskMode::NORMAL>(64u);
Add<Z1,false>(ub_vacc0,ub_vacc0,ub_vacc0[16u*v0_stride],MASK_PLACEHOLDER,Z5<ZA>((16u*v0_stride)>>6u),Z6{1,1,1,8,8,8});
if(!v1_folded16){
Add<Z1,false>(ub_vacc1,ub_vacc1,ub_vacc1[16u*v1_stride],MASK_PLACEHOLDER,Z5<ZA>((16u*v1_stride)>>6u),Z6{1,1,1,8,8,8});
}else{
Add<Z1,false>(ub_vacc1,ub_vacc1,ub_vacc1[8u*v1_stride],MASK_PLACEHOLDER,Z5<ZA>((8u*v1_stride)>>6u),Z6{1,1,1,8,8,8});
}
Z3<ZC>();
Add<Z1,false>(ub_vacc0,ub_vacc0,ub_vacc0[8u*v0_stride],MASK_PLACEHOLDER,Z5<ZA>((8u*v0_stride)>>6u),Z6{1,1,1,8,8,8});
if(!v1_folded16){
Add<Z1,false>(ub_vacc1,ub_vacc1,ub_vacc1[8u*v1_stride],MASK_PLACEHOLDER,Z5<ZA>((8u*v1_stride)>>6u),Z6{1,1,1,8,8,8});
}else{
Add<Z1,false>(ub_vacc1,ub_vacc1,ub_vacc1[4u*v1_stride],MASK_PLACEHOLDER,Z5<ZA>((4u*v1_stride)>>6u),Z6{1,1,1,8,8,8});
}
Z3<ZC>();
Add<Z1,false>(ub_vacc0,ub_vacc0,ub_vacc0[4u*v0_stride],MASK_PLACEHOLDER,Z5<ZA>((4u*v0_stride)>>6u),Z6{1,1,1,8,8,8});
if(!v1_folded16){
Add<Z1,false>(ub_vacc1,ub_vacc1,ub_vacc1[4u*v1_stride],MASK_PLACEHOLDER,Z5<ZA>((4u*v1_stride)>>6u),Z6{1,1,1,8,8,8});
}else{
Add<Z1,false>(ub_vacc1,ub_vacc1,ub_vacc1[2u*v1_stride],MASK_PLACEHOLDER,Z5<ZA>((2u*v1_stride)>>6u),Z6{1,1,1,8,8,8});
}
Z3<ZC>();
Add<Z1,false>(ub_vacc0,ub_vacc0,ub_vacc0[2u*v0_stride],MASK_PLACEHOLDER,Z5<ZA>((2u*v0_stride)>>6u),Z6{1,1,1,8,8,8});
if(!v1_folded16){
Add<Z1,false>(ub_vacc1,ub_vacc1,ub_vacc1[2u*v1_stride],MASK_PLACEHOLDER,Z5<ZA>((2u*v1_stride)>>6u),Z6{1,1,1,8,8,8});
}else{
Add(ub_vacc1,ub_vacc1,ub_vacc1[v1_stride],v1_stride);
SetMaskNorm();SetVectorMask<Z1,MaskMode::NORMAL>(64u);}
Z3<ZC>();if(first_act_rows<32u){
Add(ub_w[first_z_next],ub_vacc0,ub_vacc0[v0_stride],32u);
Add(ub_vacc0[32],ub_vacc0[32],ub_vacc0[v0_stride+32u],v1_stride);if(!v1_folded16){
Add(ub_vacc1,ub_vacc1,ub_vacc1[v1_stride],v1_stride);}
Muls(ub_prod[Z2],ub_v[Z2],0.5f,256u-Z2);Z3<ZC>();
Add(ub_w[first_z_next+32u],ub_vacc0[32],ub_vacc1,v1_stride);Z3<ZC>();
}else{
Add(ub_vacc0,ub_vacc0,ub_vacc0[v0_stride],v0_stride);if(!v1_folded16){
Add(ub_vacc1,ub_vacc1,ub_vacc1[v1_stride],v1_stride);}
Z3<ZC>();Add(ub_w[first_z_next],ub_w[first_z_next],ub_vacc0,32u);
Add(ub_vacc0[32],ub_vacc0[32],ub_vacc1,v1_stride);Muls(ub_prod[Z2],ub_v[Z2],0.5f,256u-Z2);
Z3<ZC>();Add(ub_w[first_z_next+32u],ub_w[first_z_next+32u],ub_vacc0[32],v1_stride);
Z3<ZC>();}
}else if(has_p2){
if(n==256u&&first_act_rows+1u<vacc_rows){
Duplicate(ub_vacc[(first_act_rows+1u)*vacc_stride],0.0f,(vacc_rows-first_act_rows-1u)*vacc_stride);
Z3<ZC>();}
Z4<Z1>v_scratch=ub_tridiag_sc[vacc_floats];Z0 v_scratch_cap=p2_total_cap-vacc_floats;
if(!p2_brcb_issued){
Brcb(brcb_v_p2,ub_v[Z2],Z5<ZA>(aligned_p2_rows/8u),{1,8});
if(first_act_rows+1u<vacc_rows){
Duplicate(ub_vacc[(first_act_rows+1u)*vacc_stride],0.0f,(vacc_rows-first_act_rows-1u)*vacc_stride);}
Z3<ZC>();}

for(Z0 zb0=p23_z_start;zb0+1u<num_zones;){
Z0 zb1=zb0;Z0 used_sc=0u;
Z0 cur_cap=(zb0==p23_z_start)?p2_total_cap:v_scratch_cap;while(zb1+1u<num_zones){
Z0 z_row=zb1<<5u;Z0 z_next=z_row+32u;
Z0 r_start=(k+1u>z_row)?(k+1u):z_row;Z0 ZD=z_next-r_start;
Z0 raw_vsc=((n-z_next)+7u)&~7u;
Z0 sc_v=(zb1==p23_z_start)?vacc_stride:((n==Z9)?raw_vsc:((raw_vsc>=64u)?raw_vsc:64u));
Z0 need_sc=(zb1==p23_z_start&&ZD<vacc_rows)?vacc_floats:(ZD*sc_v);
if(zb1>zb0&&used_sc+need_sc>cur_cap){
break;}
used_sc+=need_sc;++zb1;
}

Z0 sc_off=0u;bool any_gt16=false;
for(Z0 z=zb0;z<zb1;++z){
Z0 z_row=z<<5u;Z0 z_next=z_row+32u;
Z0 stride_z=Z9-z_row;Z0 r_start=(k+1u>z_row)?(k+1u):z_row;
Z0 ZD=z_next-r_start;Z0 raw_vsc=((n-z_next)+7u)&~7u;
Z0 sc_v=(z==p23_z_start)?vacc_stride:((n==Z9)?raw_vsc:((raw_vsc>=64u)?raw_vsc:64u));
ZA rep_stride=Z5<ZA>(stride_z/8u);ZA ZE=Z5<ZA>(sc_v/8u);
Z4<Z1>mat_z=ZH[zone_offset[z]+(r_start-z_row)*stride_z-z_shift];
Z4<Z1>sc_z=(z==p23_z_start)?ub_vacc:v_scratch[sc_off];
Z4<Z1>brcb_z=brcb_v_p2[(r_start-Z2)*8u];for(Z0 c=z_next;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;
Mul(sc_z[c-z_next],mat_z[c-z_row],brcb_z,width,ZD,Z6{1,1,0,ZE,rep_stride,1});
}
if(ZD>16u)any_gt16=true;if(z==p23_z_start){
if(first_act_rows<vacc_rows){
Adds(ub_vacc[first_act_rows*vacc_stride],ub_w[first_z_next],0.0f,n-first_z_next);}
sc_off=(ZD>vacc_rows)?((ZD-vacc_rows)*vacc_stride):0u;}else{
sc_off+=ZD*sc_v;}
}
Z3<ZC>();

if(vacc_rows<=16u&&any_gt16){
sc_off=0u;for(Z0 z=zb0;z<zb1;++z){
Z0 z_row=z<<5u;Z0 z_next=z_row+32u;
Z0 r_start=(k+1u>z_row)?(k+1u):z_row;Z0 ZD=z_next-r_start;
Z0 raw_vsc=((n-z_next)+7u)&~7u;
Z0 sc_v=(z==p23_z_start)?vacc_stride:((n==Z9)?raw_vsc:((raw_vsc>=64u)?raw_vsc:64u));
Z4<Z1>sc_z=(z==p23_z_start)?ub_vacc:v_scratch[sc_off];if(ZD>16u){
Z0 fold_cnt=ZD-16u;if(sc_v==n-z_next){
Add(sc_z,sc_z,sc_z[16u*sc_v],fold_cnt*sc_v);}else{
ZA ZE=Z5<ZA>(sc_v/8u);for(Z0 c=z_next;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-z_next;
Add(sc_z[sc],sc_z[sc],sc_z[16u*sc_v+sc],width,fold_cnt,Z6{1,1,1,ZE,ZE,ZE});
}
}
}
if(z==p23_z_start){
if(n==Z9&&first_act_rows>=vacc_rows&&first_act_rows<32u){
Add(ub_vacc[15u*vacc_stride],ub_vacc[15u*vacc_stride],ub_w[first_z_next],n-first_z_next);}
sc_off=(ZD>vacc_rows)?((ZD-vacc_rows)*vacc_stride):0u;}else{
sc_off+=ZD*sc_v;}
}
Z3<ZC>();}

Z0 add_z0=(zb0==p23_z_start)?(zb0+1u):zb0;if(add_z0<zb1){
sc_off=(zb0==p23_z_start&&first_act_rows>vacc_rows)?((first_act_rows-vacc_rows)*vacc_stride):0u;
for(Z0 z=add_z0;z<zb1;++z){
Z0 z_row=z<<5u;Z0 z_next=z_row+32u;
Z0 r_start=(k+1u>z_row)?(k+1u):z_row;Z0 ZD=z_next-r_start;
Z0 rem_r=(ZD>vacc_rows)?vacc_rows:ZD;Z0 raw_vsc=((n-z_next)+7u)&~7u;
Z0 sc_v=(n==Z9)?raw_vsc:((raw_vsc>=64u)?raw_vsc:64u);
ZA ZE=Z5<ZA>(sc_v/8u);Z4<Z1>sc_z=v_scratch[sc_off];
for(Z0 c=z_next;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 sc=c-z_next;
Add(ub_vacc[c-first_z_next],ub_vacc[c-first_z_next],sc_z[sc],width,rem_r,Z6{1,1,1,vacc_rep_stride,vacc_rep_stride,ZE});
}
sc_off+=ZD*sc_v;}
Z3<ZC>();}
zb0=zb1;}

if(n==Z9){
if(vacc_rows>16u){
Add(ub_vacc,ub_vacc,ub_vacc[16u*vacc_stride],(vacc_rows-16u)*vacc_stride);Z3<ZC>();
}
if(vacc_rows>8u){
Z0 r16=(vacc_rows>16u)?16u:vacc_rows;
Add(ub_vacc,ub_vacc,ub_vacc[8u*vacc_stride],(r16-8u)*vacc_stride);Z3<ZC>();
}
Add(ub_vacc,ub_vacc,ub_vacc[4u*vacc_stride],4u*vacc_stride);Z3<ZC>();
Add(ub_vacc,ub_vacc,ub_vacc[2u*vacc_stride],2u*vacc_stride);Z3<ZC>();
if(first_act_rows<32u){
Add(ub_w[first_z_next],ub_vacc,ub_vacc[vacc_stride],n-first_z_next);
Muls(ub_prod[Z2],ub_v[Z2],(n==256u)?0.5f:-0.5f,n-Z2);Z3<ZC>();
}else{
Add(ub_vacc,ub_vacc,ub_vacc[vacc_stride],n-first_z_next);Z3<ZC>();
Add(ub_w[first_z_next],ub_w[first_z_next],ub_vacc,n-first_z_next);
Muls(ub_prod[Z2],ub_v[Z2],(n==256u)?0.5f:-0.5f,n-Z2);Z3<ZC>();
}
}else{
for(Z0 half=16u;half>0u;half>>=1u){
if(vacc_rows<=half)continue;Z0 cur_r=(vacc_rows<2u*half)?(vacc_rows-half):half;
for(Z0 c=first_z_next;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Add(ub_vacc[c-first_z_next],
ub_vacc[c-first_z_next],
ub_vacc[half*vacc_stride+(c-first_z_next)],
width,
cur_r,
Z6{1,1,1,vacc_rep_stride,vacc_rep_stride,vacc_rep_stride});}
Z3<ZC>();}
if(first_act_rows>=vacc_rows){
Add(ub_w[first_z_next],ub_w[first_z_next],ub_vacc,n-first_z_next);}else{
Adds(ub_w[first_z_next],ub_vacc,0.0f,n-first_z_next);}
Muls(ub_prod[Z2],ub_v[Z2],(n==256u)?0.5f:-0.5f,n-Z2);Z3<ZC>();
}
}else{
Muls(ub_prod[Z2],ub_v[Z2],(n==256u)?0.5f:-0.5f,n-Z2);Z3<ZC>();
}


if(n==256u){
Z4<Z1>brcb_v=ub_tridiag_sc[512];Z4<Z1>brcb_w=ub_tridiag_sc[2560];
Z0 init_b_rows=256u-Z2;ZA init_b_rep=Z5<ZA>(init_b_rows>>3u);
VecDotAxpyNegWMaskBrcbSub(ub_w,ub_v,Z2,rem,0u,brcb_v,Z2,init_b_rep);
Brcb(brcb_w,ub_w[Z2],init_b_rep,{1,8});


{
SetMaskNorm();Z0 last_mask=0u;for(Z0 z=p23_z_start;z<num_zones;++z){
Z0 z_row=z<<5u;Z0 r_start=(z==p23_z_start)?(k+1u):z_row;Z0 ZD=z_row+32u-r_start;
Z0 stride_z=256u-z_row;ZA z_rep=Z5<ZA>(stride_z>>3u);
Z4<Z1>mat_z=ZH[zone_offset[z]+(r_start-z_row)*stride_z-z_shift];
Z4<Z1>bv_z=brcb_v[(r_start-Z2)*8u];Z0 c0=(Z2>z_row)?Z2:z_row;
for(Z0 c=c0;c<256u;c+=64u){
Z0 width=(256u-c<64u)?(256u-c):64u;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
MulAddDst<Z1,Z1,false>(mat_z[c-z_row],ub_w[c],bv_z,MASK_PLACEHOLDER,ZD,Z6{1,1,0,z_rep,0,1});
}}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);}
Z3<ZC>();
{
Z0 z=p23_z_start;Z0 z_row=z<<5u;
Z0 r_start=k+1u;Z0 ZD=(z_row+32u)-r_start;
Z0 stride_z=256u-z_row;ZA z_rep=Z5<ZA>(stride_z>>3u);
Z0 mat_off=zone_offset[z]+(r_start-z_row)*stride_z-z_shift;Z4<Z1>mat_z=ZH[mat_off];
Z4<Z1>bw_z=brcb_w[(r_start-Z2)*8u];
SetMaskNorm();Z0 last_mask=0u;for(Z0 c=Z2;c<256u;c+=64u){
Z0 width=(256u-c<64u)?(256u-c):64u;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
MulAddDst<Z1,Z1,false>(mat_z[c-z_row],ub_v[c],bw_z,MASK_PLACEHOLDER,ZD,Z6{1,1,0,z_rep,0,1});
}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);Z3<ZC>();
if(p23_z_start+1u<num_zones&&k+3u<256u){
Z0 c_n=(k+2u)&~7u;
lookahead_ev=IssueLookaheadNormRed(mat_z,ub_tridiag_sc,ub_tridiag_sc[256],c_n,c_n-z_row,(k+2u)&7u,256u-c_n);
lookahead_ready=true;}
}


if(p23_z_start+1u<num_zones){
SetMaskNorm();Z0 last_mask=0u;for(Z0 z=p23_z_start+1u;z<num_zones;++z){
Z0 z_row=z<<5u;Z0 stride_z=256u-z_row;
ZA z_rep=Z5<ZA>(stride_z>>3u);Z4<Z1>mat_z=ZH[zone_offset[z]-z_shift];
Z4<Z1>bw_z=brcb_w[(z_row-Z2)*8u];for(Z0 c=z_row;c<256u;c+=64u){
Z0 width=(256u-c<64u)?(256u-c):64u;
if(width!=last_mask){SetVectorMask<Z1,MaskMode::NORMAL>(width);last_mask=width;}
MulAddDst<Z1,Z1,false>(mat_z[c-z_row],ub_v[c],bw_z,MASK_PLACEHOLDER,32u,Z6{1,1,0,z_rep,0,1});
}
}
SetVectorMask<Z1,MaskMode::NORMAL>(64u);if(k!=15u&&k!=23u){
if(rem==0u&&c_off_k>=8u){
Duplicate(row_k[c_off_k-8u],0.0f,8u);}
Adds(row_k[c_off_k],ub_v[Z2],0.0f,256u-Z2);}
Z3<ZC>();}
tau_mask[k]=tau_k;continue;
}
Z0 max_p3_brcb=64u;Z0 cur_p3_cap=p3_scratch_cap;
Z4<Z1>brcb_v=ub_gather[256];Z4<Z1>brcb_w=ub_prod;
Z0 brcb_Start=Z2;Z0 init_b_rows=(n-Z2<max_p3_brcb)?(n-Z2):max_p3_brcb;
ZA init_b_rep=Z5<ZA>(((init_b_rows+7u)&~7u)/8u);
Z0 brcb_End=brcb_Start+init_b_rows;

VecDotAxpyMaskBrcbSub(ub_w,ub_v,Z2,rem,Z9,brcb_v,brcb_Start,init_b_rep);

for(Z0 zb0=p23_z_start;zb0<num_zones;){
Z0 zb1=zb0;Z0 used_sc=0u;
Z0 used_rows=0u;while(zb1<num_zones){
Z0 z_row=zb1<<5u;Z0 r_start=(k+1u>z_row)?(k+1u):z_row;
Z0 r_end=(z_row+32u<n)?(z_row+32u):n;Z0 ZD=r_end-r_start;
Z0 c_start=(Z2>z_row)?Z2:z_row;Z0 raw_sc=((n-c_start)+7u)&~7u;
Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;
if(zb1>zb0&&(used_sc+ZD*sc_stride>cur_p3_cap||(r_end-((zb0==p23_z_start)?Z2:(zb0<<5u)))>max_p3_brcb)){
break;}
used_sc+=ZD*sc_stride;used_rows+=ZD;
++zb1;}

Z0 batch_r0=(k+1u>(zb0<<5u))?(k+1u):(zb0<<5u);if(zb0==p23_z_start){
Brcb(brcb_w,ub_w[brcb_Start],init_b_rep,{1,8});}else if(batch_r0+used_rows>brcb_End){
brcb_Start=batch_r0&~7u;Z0 rem_total_rows=n-brcb_Start;
Z0 b_rows=(rem_total_rows<max_p3_brcb)?rem_total_rows:max_p3_brcb;
Z0 aligned_b_rows=(b_rows+7u)&~7u;ZA brcb_rep=Z5<ZA>(aligned_b_rows/8u);
Brcb(brcb_v,ub_v[brcb_Start],brcb_rep,{1,8});Brcb(brcb_w,ub_w[brcb_Start],brcb_rep,{1,8});
Z3<ZC>();brcb_End=brcb_Start+b_rows;
}

Z0 sc_off=0u;for(Z0 z=zb0;z<zb1;++z){
Z0 z_row=z<<5u;Z0 r_start=(k+1u>z_row)?(k+1u):z_row;
Z0 r_end=(z_row+32u<n)?(z_row+32u):n;Z0 ZD=r_end-r_start;
Z0 c_start=(Z2>z_row)?Z2:z_row;Z0 raw_sc=((n-c_start)+7u)&~7u;
Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;ZA ZE=Z5<ZA>(sc_stride/8u);
Z4<Z1>sc_z=ub_tridiag_sc[sc_off];Z4<Z1>bv_z=brcb_v[(r_start-brcb_Start)*8u];
for(Z0 c=c_start;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;
Mul(sc_z[c-c_start],ub_w[c],bv_z,width,ZD,Z6{1,1,0,ZE,0,1});}
sc_off+=ZD*sc_stride;}
Z3<ZC>();

sc_off=0u;for(Z0 z=zb0;z<zb1;++z){
Z0 z_row=z<<5u;Z0 stride_z=Z9-z_row;
Z0 r_start=(k+1u>z_row)?(k+1u):z_row;Z0 r_end=(z_row+32u<n)?(z_row+32u):n;
Z0 ZD=r_end-r_start;Z0 c_start=(Z2>z_row)?Z2:z_row;
Z0 raw_sc=((n-c_start)+7u)&~7u;Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;
ZA rep_stride=Z5<ZA>(stride_z/8u);
ZA ZE=Z5<ZA>(sc_stride/8u);
Z4<Z1>mat_z=ZH[zone_offset[z]+(r_start-z_row)*stride_z];
Z4<Z1>sc_z=ub_tridiag_sc[sc_off];Z4<Z1>bw_z=brcb_w[(r_start-brcb_Start)*8u];
for(Z0 c=c_start;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 mc=c-z_row;
Z0 sc=c-c_start;
Sub(mat_z[mc],mat_z[mc],sc_z[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
Mul(sc_z[sc],ub_v[c],bw_z,width,ZD,Z6{1,1,0,ZE,0,1});}
sc_off+=ZD*sc_stride;}
Z3<ZC>();

sc_off=0u;for(Z0 z=zb0;z<zb1;++z){
Z0 z_row=z<<5u;Z0 stride_z=Z9-z_row;
Z0 r_start=(k+1u>z_row)?(k+1u):z_row;Z0 r_end=(z_row+32u<n)?(z_row+32u):n;
Z0 ZD=r_end-r_start;Z0 c_start=(Z2>z_row)?Z2:z_row;
Z0 raw_sc=((n-c_start)+7u)&~7u;Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;
ZA rep_stride=Z5<ZA>(stride_z/8u);
ZA ZE=Z5<ZA>(sc_stride/8u);
Z4<Z1>mat_z=ZH[zone_offset[z]+(r_start-z_row)*stride_z];
Z4<Z1>sc_z=ub_tridiag_sc[sc_off];for(Z0 c=c_start;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Z0 mc=c-z_row;
Z0 sc=c-c_start;
Sub(mat_z[mc],mat_z[mc],sc_z[sc],width,ZD,Z6{1,1,1,rep_stride,rep_stride,ZE});
}
sc_off+=ZD*sc_stride;}
Z3<ZC>();zb0=zb1;
}

if(n!=Z9){
if(rem==0u&&c_off_k>=8u){
Duplicate(row_k[c_off_k-8u],0.0f,8u);}
Adds(row_k[c_off_k],ub_v[Z2],0.0f,Z9-Z2);Z3<ZC>();
}
}
tau_mask[k]=tau_k;}
ub_w=orig_ub_w;}
#if EIGH_K96_PASS
#else
}
#endif


__aicore__ inline void BacktransformMedium(const ZA*tau_mask){
if(ZB==0u||n<3u){
return;}
Z3<ZC>();if(n<=120u){
bool can_pair=(2u*ZB<=rows_per_lane)&&(ZB<=64u);Z4<Z1>scratch0=ZH;
ZG k=Z5<ZG>(n)-3;for(;can_pair&&k>=1;k-=2){
Z0 uk=Z5<Z0>(k);Z0 c_first_pair=uk&~7u;
Z0 raw_sc=((n-c_first_pair)+7u)&~7u;
Z0 sc_stride=(raw_sc>=64u)?raw_sc:((Z9<64u)?Z9:64u);ZA tau_k=tau_mask[uk];
ZA tau_km1=tau_mask[uk-1u];if(tau_k!=0u&&tau_km1!=0u){
BacktransformPairUnitRowsSub(
ub_vecs,
ub_refl[uk*Z9],
ub_refl[(uk-1u)*Z9],
scratch0,
scratch0[ZB*sc_stride],
ZB,
uk);}else{
if(tau_k!=0u){
Z0 Z2=(uk+1u)&~7u;
BacktransformStepRowsSub(ub_vecs,ub_refl[uk*Z9],-1.0f,scratch0,ZB,Z2);}
if(tau_km1!=0u){
Z0 ukm1=uk-1u;Z0 Z2=(ukm1+1u)&~7u;
BacktransformStepRowsSub(ub_vecs,ub_refl[ukm1*Z9],-1.0f,scratch0,ZB,Z2);}
}
}
for(;k>=0;--k){
Z0 uk=Z5<Z0>(k);if(tau_mask[uk]==0u){
continue;}
Z0 Z2=(uk+1u)&~7u;
BacktransformStepRowsSub(ub_vecs,ub_refl[uk*Z9],-1.0f,scratch0,ZB,Z2);}
}
Z3<PIPE_ALL>();}

__aicore__ inline void BacktransformLarge(uint64_t b,const ZA*tau_mask){
if(ZB==0u||n<3u){
return;}
Z3<ZC>();if(n<=200u){
ZG k=Z5<ZG>(n)-3;for(;k>=1;k-=2){
Z0 uk=Z5<Z0>(k);Z0 scratch_off=(n==200u)?22144u:((uk+1u)*Z9);
Z4<Z1>scratch_base=ZH[scratch_off];
Z4<Z1>v_k=(n==200u)?GetReflector200(uk):ZH[uk*Z9];
Z4<Z1>v_km1=(n==200u)?GetReflector200(uk-1u):ZH[(uk-1u)*Z9];
Z0 c_first_pair=uk&~7u;Z0 raw_sc=((n-c_first_pair)+7u)&~7u;
Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;ZA tau_k=tau_mask[uk];
ZA tau_km1=tau_mask[uk-1u];if(tau_k!=0u&&tau_km1!=0u){
BacktransformPairUnitRowsSub(
ub_vecs,
v_k,
v_km1,
scratch_base,
scratch_base[ZB*sc_stride],
ZB,
uk);}else{
if(tau_k!=0u){
Z0 Z2=(uk+1u)&~7u;BacktransformStepRowsSub(ub_vecs,v_k,-1.0f,scratch_base,ZB,Z2);
}
if(tau_km1!=0u){
Z0 ukm1=uk-1u;Z0 Z2=(ukm1+1u)&~7u;
BacktransformStepRowsSub(ub_vecs,v_km1,-1.0f,scratch_base,ZB,Z2);}
}
}
for(;k>=0;--k){
Z0 uk=Z5<Z0>(k);if(tau_mask[uk]==0u){
continue;}
Z0 Z2=(uk+1u)&~7u;Z4<Z1>v_k=(n==200u)?GetReflector200(uk):ZH[uk*Z9];
Z0 scratch_off=(n==200u)?22144u:((uk+1u)*Z9);
BacktransformStepRowsSub(ub_vecs,v_k,-1.0f,ZH[scratch_off],ZB,Z2);}
#if EIGH_K96_PASS
}else if(n==256u){
#else
}else if constexpr(!FAST200){
if(n==256u){
#endif
uint64_t priv_gm_256=b*65536ull+vec_row_start;ZG bt_mte2_ev=0;
ZG k=253;for(;k>=1;k-=2){
Z0 uk=Z5<Z0>(k);if(uk==95u){
const ZG v_mte2_ev=Z5<ZG>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
SetFlag<HardEvent::V_MTE2>(v_mte2_ev);WaitFlag<HardEvent::V_MTE2>(v_mte2_ev);
#if EIGH_K96_PASS
// Same uk95 issue point; compact shared R0..23 replaces the old
// per-lane gm_eigvecs format. uk23 still waits on the MTE2_V event.
DataCopy(ZH[29184],gm_ckpt[b*65536ull+25600u],6144u);
#else
DataCopyPad(
ZH[29184],
gm_eigvecs[priv_gm_256],
DataCopyExtParams{256u,64u,960u,0,0},
DataCopyPadExtParams<Z1>{false,0,0,0.0f});DataCopyPad(
ZH[33280],
gm_eigvecs[priv_gm_256+16u],
DataCopyExtParams{256u,32u,992u,0,0},
DataCopyPadExtParams<Z1>{false,0,0,0.0f});
#endif
bt_mte2_ev=Z5<ZG>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
SetFlag<HardEvent::MTE2_V>(bt_mte2_ev);}
if(uk==23u){
WaitFlag<HardEvent::MTE2_V>(bt_mte2_ev);}
Z0 scratch_off=(uk>=224u)?30208u:((uk>=192u)?29184u:((uk>=160u)?27136u:((uk>=144u)?24064u:((uk>=128u)?22272u:((uk>=112u)?20224u:((uk>=96u)?17920u:((uk>=24u)?15360u:6144u)))))));
Z4<Z1>scratch_base=ZH[scratch_off];Z0 c_first_pair=uk&~7u;
Z0 raw_sc=((256u-c_first_pair)+7u)&~7u;Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;
ZA tau_k=tau_mask[uk];ZA tau_km1=tau_mask[uk-1u];
Z4<Z1>v_k=GetStaircaseReflector(uk);Z4<Z1>v_km1=GetStaircaseReflector(uk-1u);
if(tau_k!=0u&&tau_km1!=0u){
BacktransformPairUnitRowsSub(
ub_vecs,
v_k,
v_km1,
scratch_base,
scratch_base[ZB*sc_stride],
ZB,
uk);}else{
if(tau_k!=0u){
Z0 Z2=(uk+1u)&~7u;BacktransformStepRowsSub(ub_vecs,v_k,-1.0f,scratch_base,ZB,Z2);
}
if(tau_km1!=0u){
Z0 ukm1=uk-1u;Z0 Z2=(ukm1+1u)&~7u;
BacktransformStepRowsSub(ub_vecs,v_km1,-1.0f,scratch_base,ZB,Z2);}
}
}
for(;k>=0;--k){
Z0 uk=Z5<Z0>(k);if(tau_mask[uk]==0u){
continue;}
Z0 Z2=(uk+1u)&~7u;Z4<Z1>v_k=GetStaircaseReflector(uk);
BacktransformStepRowsSub(ub_vecs,v_k,-1.0f,ZH[6144],ZB,Z2);}
}else{
ZG k=Z5<ZG>(n)-3;for(;k>=1;k-=2){
Z0 uk=Z5<Z0>(k);Z0 z_k=uk>>5u;
Z0 z_k_row=z_k<<5u;Z0 scratch_off=zone_offset[z_k]+(uk+1u-z_k_row)*(Z9-z_k_row);
Z0 avail_scratch=vecs_offset-scratch_off;Z4<Z1>scratch_base=ZH[scratch_off];

Z0 c_first_pair=uk&~7u;Z0 raw_sc=((n-c_first_pair)+7u)&~7u;
Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;

ZA tau_k=tau_mask[uk];ZA tau_km1=tau_mask[uk-1u];
Z4<Z1>v_k=GetStaircaseReflector(uk);Z4<Z1>v_km1=GetStaircaseReflector(uk-1u);

Z0 max_pair_rows=avail_scratch/(2u*sc_stride);
Z0 bt_tile_rows=(max_pair_rows>=ZB)?ZB:(max_pair_rows&~7u);

if(tau_k!=0u&&tau_km1!=0u&&bt_tile_rows>=8u){
for(Z0 r_off=0;r_off<ZB;r_off+=bt_tile_rows){
Z0 cur_rows=(ZB-r_off<bt_tile_rows)?(ZB-r_off):bt_tile_rows;
BacktransformPairUnitRowsSub(
ub_vecs[r_off*Z9],
v_k,
v_km1,
scratch_base,
scratch_base[cur_rows*sc_stride],
cur_rows,
uk);}
}else{
Z0 max_single_rows=avail_scratch/Z9;
Z0 single_tile_rows=(max_single_rows>=ZB)?ZB:(max_single_rows&~7u);if(tau_k!=0u){
Z0 Z2=(uk+1u)&~7u;for(Z0 r_off=0;r_off<ZB;r_off+=single_tile_rows){
Z0 cur_rows=(ZB-r_off<single_tile_rows)?(ZB-r_off):single_tile_rows;
BacktransformStepRowsSub(ub_vecs[r_off*Z9],v_k,-1.0f,scratch_base,cur_rows,Z2);}
}
if(tau_km1!=0u){
Z0 ukm1=uk-1u;Z0 Z2=(ukm1+1u)&~7u;
for(Z0 r_off=0;r_off<ZB;r_off+=single_tile_rows){
Z0 cur_rows=(ZB-r_off<single_tile_rows)?(ZB-r_off):single_tile_rows;
BacktransformStepRowsSub(ub_vecs[r_off*Z9],v_km1,-1.0f,scratch_base,cur_rows,Z2);}
}
}
}
for(;k>=0;--k){
Z0 uk=Z5<Z0>(k);if(tau_mask[uk]==0u){
continue;}
Z0 z_k=uk>>5u;Z0 z_k_row=z_k<<5u;
Z0 scratch_off=zone_offset[z_k]+(uk+1u-z_k_row)*(Z9-z_k_row);
Z0 avail_scratch=vecs_offset-scratch_off;Z0 max_single_rows=avail_scratch/Z9;
Z0 single_tile_rows=(max_single_rows>=ZB)?ZB:(max_single_rows&~7u);
Z0 Z2=(uk+1u)&~7u;Z4<Z1>v_k=GetStaircaseReflector(uk);
for(Z0 r_off=0;r_off<ZB;r_off+=single_tile_rows){
Z0 cur_rows=(ZB-r_off<single_tile_rows)?(ZB-r_off):single_tile_rows;
BacktransformStepRowsSub(ub_vecs[r_off*Z9],v_k,-1.0f,ZH[scratch_off],cur_rows,Z2);
#if EIGH_K96_PASS
#else
}
#endif
}
}
}
}

__aicore__ inline void LoadAndSymmetrizeCoop(uint64_t b){
Z1 local_max=0.0f;uint64_t batch_base=b*n*n;
if(is_active&&ZB>0u){
Z0 col_stride=(ZB+7u)&~7u;ZA col_pad=Z5<ZA>(col_stride-ZB);
DataCopyPad(
ZH,
gm_A[batch_base+Z5<uint64_t>(ZJ)*n],
DataCopyExtParams{Z5<uint16_t>(ZB),n*4u,0,0,0},
DataCopyPadExtParams<Z1>{true,0,Z5<ZA>(Z9-n),0.0f});DataCopyPad(
ub_vecs,
gm_A[batch_base+ZJ],
DataCopyExtParams{Z5<uint16_t>(n),ZB*4u,(n-ZB)*4u,0,0},
DataCopyPadExtParams<Z1>{col_pad!=0u,0,col_pad,0.0f});Z3<PIPE_ALL>();

Z4<Z0>col_idx_buf=ub_diag.ReinterpretCast<Z0>();for(Z0 j=0;j<Z9;++j){
col_idx_buf.Z8(j,j*col_stride*4u);}
Z3<PIPE_ALL>();for(Z0 r=0;r<ZB;++r){
Z0 i=ZJ+r;Z4<Z1>row_i=ZH[r*Z9];
if(uplo!=0u){
Z0 c_end8=i&~7u;if(c_end8>0u){
Gather(row_i,ub_vecs,col_idx_buf,r*4u,c_end8);}
for(Z0 j=c_end8;j<i;++j){
row_i.Z8(j,ub_vecs.ZF(j*col_stride+r));}
}else{
Z0 n_end8=n&~7u;Z0 c_next8=(i+8u)&~7u;
if(c_next8<n_end8){
Gather(row_i[c_next8],ub_vecs[c_next8*col_stride],col_idx_buf,r*4u,n_end8-c_next8);}
Z0 j_head_lim=(c_next8<n)?c_next8:n;for(Z0 j=i+1u;j<j_head_lim;++j){
row_i.Z8(j,ub_vecs.ZF(j*col_stride+r));}
Z0 j_tail_start=(n_end8>j_head_lim)?n_end8:j_head_lim;for(Z0 j=j_tail_start;j<n;++j){
row_i.Z8(j,ub_vecs.ZF(j*col_stride+r));}
}
}
Z3<PIPE_ALL>();Z3<PIPE_ALL>();
Abs(ub_vecs,ZH,ZB*Z9);Z3<ZC>();
ReduceMax(ub_red,ub_vecs,ub_gather,ZB*Z9);Z3<PIPE_ALL>();
local_max=ub_red.ZF(0);

if(lane_id==0u){
DataCopy(gm_ws[ws_scale_offset+8u*group_lanes],ZH,Z9);}
}
ub_comm.Z8(0,local_max);Z3<PIPE_ALL>();
DataCopy(gm_ws[ws_scale_offset+lane_id*8u],ub_comm,8u);Z3<PIPE_ALL>();
SyncAll<true>();

if(is_active){
DataCopy(ub_gather,gm_ws[ws_scale_offset],8u*group_lanes);
DataCopy(ub_cur_row,gm_ws[ws_scale_offset+8u*group_lanes],Z9);Z3<PIPE_ALL>();
Z1 max_val=0.0f;for(Z0 l=0;l<group_lanes;++l){
Z1 s_l=ub_gather.ZF(l*8u);if(s_l>max_val){
max_val=s_l;}
}
Z1 inv_scale=1.0f;ComputePow2Scale(max_val,matrix_scale,inv_scale);
if(ZB>0u){
Muls(ZH,ZH,inv_scale,ZB*Z9);}
Muls(ub_cur_row,ub_cur_row,inv_scale,Z9);Duplicate(ub_v,0.0f,2u*Z9);
Duplicate(ub_diag,0.0f,3u*Z9);Z3<PIPE_ALL>();
}
}


__aicore__ inline void TridiagonalizeCoop(){
if(n==1u){
if(is_active){
ub_diag.Z8(0,ub_cur_row.ZF(0));}
return;}
if(n==2u){
if(is_active){
ub_diag.Z8(0,ub_cur_row.ZF(0));ub_off.Z8(0,ub_cur_row.ZF(1));
if(1u>=ZJ&&1u<ZJ+ZB){
ub_diag.Z8(1,ZH.ZF((1u-ZJ)*Z9+1u));}
}
return;}

Z0 plane_cap=rows_per_lane*Z9;
Z0 tail_dim=(plane_cap>=144u*144u)?144u:((plane_cap>=136u*136u)?136u:((plane_cap>=128u*128u)?128u:((plane_cap>=96u*96u)?96u:0u)));
Z0 k_switch=(n>tail_dim&&tail_dim>=64u)?((n-tail_dim)&~7u):(n-2u);Z0 tail_rows=n-k_switch;
Z0 tail_stride=(tail_rows+7u)&~7u;if(k_switch<8u||k_switch+3u>=n||tail_rows*tail_stride>plane_cap){
k_switch=n-2u;}
bool expanded_tail=false;Z0 tail_scratch_cap=plane_cap;
if(n==256u){
const Z0 arena_cap=2u*plane_cap;for(Z0 target=192u;target>=96u;target-=8u){
const Z0 candidate_k=(n-target)&~7u;const Z0 candidate_rows=n-candidate_k;
const Z0 candidate_stride=(candidate_rows+7u)&~7u;
const Z0 matrix_need=candidate_rows*candidate_stride;if(candidate_k>=8u&&candidate_k+3u<n&&
matrix_need+24u*candidate_stride<=arena_cap){
k_switch=candidate_k;tail_rows=candidate_rows;
tail_stride=candidate_stride;tail_scratch_cap=arena_cap-matrix_need;
expanded_tail=true;break;
}
}
}
Z4<Z1>tail_scratch=expanded_tail?ZH[tail_rows*tail_stride]:ub_vecs;

Z0 k_coop_end=(k_switch<n-2u)?(k_switch+1u):(n-2u);for(Z0 k=0;k<k_coop_end;++k){
uint64_t w_bank=ws_w_offset+Z5<uint64_t>(k&1u)*3u*Z9;Z0 Z2=(k+1u)&~7u;
Z0 rem=(k+1u)&7u;Z1 tau_k=0.0f;
Z0 r0=0u;Z0 ZD=0u;

if(is_active){
Mul(ub_v[Z2],ub_cur_row[Z2],ub_mask_ge[rem*8u],8u);if(Z9>Z2+8u){
Adds(ub_v[Z2+8u],ub_cur_row[Z2+8u],0.0f,Z9-Z2-8u);}

Z1 norm=ScalarSqrt(VecDotSub(ub_v,ub_v,Z2));ub_diag.Z8(k,ub_cur_row.ZF(k));
if(norm!=0.0f){
Z1 x0=ub_v.ZF(k+1u);ub_v.Z8(k+1u,0.0f);
Z1 beta=(x0>=0.0f)?-norm:norm;ub_off.Z8(k,beta);
tau_k=(beta-x0)/beta;Muls(ub_v[Z2],ub_v[Z2],1.0f/(x0-beta),n-Z2);
Z3<ZC>();Add(ub_v[Z2],ub_v[Z2],ub_mask_eq[rem*8u],8u);
}
ub_tau.Z8(k,tau_k);

if(ZJ+ZB<=k+1u){
r0=ZB;}else if(ZJ<k+1u){
r0=(k+1u-ZJ)&~7u;}
ZD=(r0<ZB)?(ZB-r0):0u;

if(tau_k!=0.0f&&ZD>0u){
Z3<ZC>();MatVecRowsSub(ZH[r0*Z9],ub_v,ub_row[r0],ub_vecs,ZD,Z2);
Muls(ub_row[r0],ub_row[r0],tau_k,ZD);if(ZJ+r0<=k){
Z3<ZC>();Mul(ub_row[r0],ub_row[r0],ub_mask_ge[rem*8u],8u);
}
}
Z3<PIPE_ALL>();

if((k%group_lanes)==lane_id&&tau_k!=0.0f){
DataCopy(gm_ws[ws_refl_offset+Z5<uint64_t>(k)*Z9+Z2],ub_v[Z2],Z9-Z2);}
if(tau_k!=0.0f&&ZD>0u){
Z0 aligned_act=(ZD+7u)&~7u;DataCopy(gm_ws[w_bank+ZJ+r0],ub_row[r0],aligned_act);
}
if(k+1u>=ZJ&&k+1u<ZJ+ZB){
DataCopy(gm_ws[w_bank+Z9],ZH[(k+1u-ZJ)*Z9],Z9);}
if(k==k_switch&&ZJ+ZB>k_switch){
Z0 r_ts=(ZJ>=k_switch)?0u:(k_switch-ZJ);Z0 g_ts=ZJ+r_ts;
Z0 t_cnt=ZB-r_ts;DataCopyPad(
gm_ws[ws_tail_offset+Z5<uint64_t>(g_ts-k_switch)*tail_stride],
ZH[r_ts*Z9+k_switch],
DataCopyExtParams{Z5<uint16_t>(t_cnt),tail_stride*4u,(Z9-tail_stride)/8u,0,0});}
if(k+3u==n&&(n-1u)>=ZJ&&(n-1u)<ZJ+ZB){
ub_comm.Z8(0,ZH.ZF((n-1u-ZJ)*Z9+(n-1u)));Z3<PIPE_ALL>();
DataCopy(gm_ws[w_bank+2u*Z9],ub_comm,8u);}
}

Z3<PIPE_ALL>();SyncAll<true>();

if(is_active){
Z0 load_len=(2u*Z9-Z2)+((k+3u==n)?8u:0u);
DataCopy(ub_w[Z2],gm_ws[w_bank+Z2],load_len);if(k==k_switch){
DataCopy(ZH,gm_ws[ws_tail_offset],tail_rows*tail_stride);}
Z3<PIPE_ALL>();

if(tau_k!=0.0f){
VecDotAxpySub(ub_w,ub_v,-0.5f*tau_k,Z2);

if(k==k_switch){
Z0 tile_rows=tail_rows;if(expanded_tail){
const Z0 raw_sc=((n-Z2)+7u)&~7u;const Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;
tile_rows=(tail_scratch_cap/sc_stride)&~7u;if(tile_rows>48u)tile_rows=48u;
}
for(Z0 r_off=0u;r_off<tail_rows;r_off+=tile_rows){
const Z0 cur_rows=(tail_rows-r_off<tile_rows)?(tail_rows-r_off):tile_rows;Rank2UpdateRowsSub(
ZH[r_off*tail_stride],
ub_v,
ub_w,
ub_v[k_switch+r_off],
ub_w[k_switch+r_off],
tail_scratch,
cur_rows,
Z2,
tail_stride,
k_switch);}
}else{
Z0 len=n-Z2;Mul(ub_red,ub_w[Z2],ub_mask_eq[rem*8u],8u);
Sub(ub_cur_row[Z2],ub_cur_row[Z2],ub_w[Z2],len);Z3<ZC>();
WholeReduceSum(ub_red,ub_red,8u,1,1,1,1);Z3<ZC>();
Brcb(ub_tmp,ub_red,1u,{1,8});Z3<ZC>();
for(Z0 c=Z2;c<n;c+=64u){
Z0 width=(n-c<64u)?(n-c):64u;Mul(ub_prod[c],ub_v[c],ub_tmp,width,1u,Z6{1,1,0,8,8,0});
}
Z3<ZC>();Sub(ub_cur_row[Z2],ub_cur_row[Z2],ub_prod[Z2],len);

if(ZD>0u){
Rank2UpdateRowsSub(
ZH[r0*Z9],
ub_v,
ub_w,
ub_v[ZJ+r0],
ub_w[ZJ+r0],
ub_vecs,
ZD,
Z2);}else{
Z3<ZC>();}
}
}
if(k!=k_switch&&k+3u==n){
Z3<PIPE_ALL>();ub_diag.Z8(n-2u,ub_cur_row.ZF(n-2u));
ub_off.Z8(n-2u,ub_cur_row.ZF(n-1u));ub_diag.Z8(n-1u,ub_w.ZF(2u*Z9));
}
}
}

if(k_switch<n-2u){
if(is_active){
for(Z0 k=k_switch+1u;k+2u<n;++k){
Z0 tk=k-k_switch;Z0 Z2=(k+1u)&~7u;
Z0 tc_first=Z2-k_switch;Z0 rem=(k+1u)&7u;
Z4<Z1>row_tk=ZH[tk*tail_stride];

Mul(ub_v[Z2],row_tk[tc_first],ub_mask_ge[rem*8u],8u);if(tail_stride>tc_first+8u){
Adds(ub_v[Z2+8u],row_tk[tc_first+8u],0.0f,tail_stride-tc_first-8u);}

Z1 norm=ScalarSqrt(VecDotSub(ub_v,ub_v,Z2));ub_diag.Z8(k,row_tk.ZF(tk));
Z1 tau_k=0.0f;if(norm!=0.0f){
Z1 x0=ub_v.ZF(k+1u);ub_v.Z8(k+1u,0.0f);
Z1 beta=(x0>=0.0f)?-norm:norm;ub_off.Z8(k,beta);
tau_k=(beta-x0)/beta;Muls(ub_v[Z2],ub_v[Z2],1.0f/(x0-beta),n-Z2);
Z3<ZC>();Add(ub_v[Z2],ub_v[Z2],ub_mask_eq[rem*8u],8u);
Z3<ZC>();if((k%group_lanes)==lane_id){
DataCopy(gm_ws[ws_refl_offset+Z5<uint64_t>(k)*Z9+Z2],ub_v[Z2],Z9-Z2);}

Z0 ZD=n-Z2;Z0 tile_rows=ZD;
if(expanded_tail){
const Z0 raw_sc=((n-Z2)+7u)&~7u;const Z0 sc_stride=(raw_sc>=64u)?raw_sc:64u;
tile_rows=(tail_scratch_cap/sc_stride)&~7u;if(tile_rows>48u)tile_rows=48u;
}
for(Z0 r_off=0u;r_off<ZD;r_off+=tile_rows){
const Z0 cur_rows=(ZD-r_off<tile_rows)?(ZD-r_off):tile_rows;MatVecRowsSub(
ZH[(tc_first+r_off)*tail_stride],
ub_v,
ub_w[Z2+r_off],
tail_scratch,
cur_rows,
Z2,
tail_stride,
k_switch);}
Muls(ub_w[Z2],ub_w[Z2],tau_k,ZD);VecDotAxpySub(ub_w,ub_v,-0.5f*tau_k,Z2);
for(Z0 r_off=0u;r_off<ZD;r_off+=tile_rows){
const Z0 cur_rows=(ZD-r_off<tile_rows)?(ZD-r_off):tile_rows;Rank2UpdateRowsSub(
ZH[(tc_first+r_off)*tail_stride],
ub_v,
ub_w,
ub_v[Z2+r_off],
ub_w[Z2+r_off],
tail_scratch,
cur_rows,
Z2,
tail_stride,
k_switch);}
}
ub_tau.Z8(k,tau_k);}
Z3<PIPE_ALL>();Z0 tk_nm2=(n-2u)-k_switch;
Z0 tk_nm1=(n-1u)-k_switch;ub_diag.Z8(n-2u,ZH.ZF(tk_nm2*tail_stride+tk_nm2));
ub_off.Z8(n-2u,ZH.ZF(tk_nm2*tail_stride+tk_nm1));
ub_diag.Z8(n-1u,ZH.ZF(tk_nm1*tail_stride+tk_nm1));}
Z3<PIPE_ALL>();SyncAll<true>();
}else if(is_active){
Z3<PIPE_ALL>();ub_diag.Z8(n-2u,ub_cur_row.ZF(n-2u));
ub_off.Z8(n-2u,ub_cur_row.ZF(n-1u));Z1 last_diag=ub_w.ZF(2u*Z9);
if(ub_tau.ZF(n-3u)!=0.0f){
last_diag-=2.0f*ub_v.ZF(n-1u)*ub_w.ZF(n-1u);}
ub_diag.Z8(n-1u,last_diag);}
Z3<PIPE_ALL>();}

__aicore__ inline bool SolveTridiagPWK(Z1*work){
Z1*dd=(SOLVE_MODE==2u)?(work+1024):work;Z1*ee=(SOLVE_MODE==2u)?(work+1536):(work+512);
for(Z0 i=0;i<n;++i){
dd[i]=(SOLVE_MODE==2u)?work[i]:ub_diag.ZF(i);
Z1 e_val=(SOLVE_MODE==2u)?work[256u+i]:ub_off.ZF(i);ee[i]=e_val*e_val;
}
Z0 total_iters=0u;const Z1 eps_sq=1.4210855e-14f;
for(Z0 l=0;l<n;){
Z0 m=l;for(;m+1u<n;++m){
if(ee[m]<=eps_sq*Z7(dd[m]*dd[m+1u])+1e-38f){
break;}
}
if(m+1u<n){
ee[m]=0.0f;}
if(m==l){
++l;continue;
}
if(m==l+1u){
Z1 mid=0.5f*(dd[l]+dd[l+1u]);Z1 hdiff=0.5f*(dd[l]-dd[l+1u]);
Z1 rad=ScalarSqrt(hdiff*hdiff+ee[l]);dd[l]=mid-rad;
dd[l+1u]=mid+rad;ee[l]=0.0f;
l+=2u;continue;
}
if(++total_iters>30u*n){
return false;}
Z1 delta=0.5f*(dd[l+1u]-dd[l]);Z1 rad=ScalarSqrt(delta*delta+ee[l]);
Z1 denom=delta+((delta>=0.0f)?rad:-rad);
Z1 sigma=(denom!=0.0f)?(dd[l]-ee[l]/denom):(dd[l]-ScalarSqrt(ee[l]));

Z1 c=1.0f;Z1 s=0.0f;
Z1 gamma=dd[m]-sigma;Z1 pp=gamma*gamma;
for(ZG ii=Z5<ZG>(m)-1;ii>=Z5<ZG>(l);--ii){
Z0 i=Z5<Z0>(ii);Z1 bb=ee[i];
Z1 rr=pp+bb;if(!(rr>0.0f)||rr>3e38f){
return false;}
if(i+1u!=m){
ee[i+1u]=s*rr;}
Z1 old_c=c;Z1 old_gamma=gamma;
Z1 alpha=dd[i];if(pp>1e-28f&&rr<1e28f){
Z1 inv_rp=1.0f/(rr*pp);Z1 inv_rr=inv_rp*pp;
Z1 inv_c=(inv_rp*rr)*rr;c=pp*inv_rr;
s=bb*inv_rr;gamma=c*(alpha-sigma)-s*old_gamma;
dd[i+1u]=old_gamma+(alpha-gamma);pp=(gamma*gamma)*inv_c;
}else{
Z1 inv_rr=1.0f/rr;c=pp*inv_rr;
s=bb*inv_rr;gamma=c*(alpha-sigma)-s*old_gamma;
dd[i+1u]=old_gamma+(alpha-gamma);pp=(c!=0.0f)?((gamma*gamma)/c):(old_c*bb);
}
}
ee[l]=s*pp;dd[l]=sigma+gamma;
}
for(Z0 i=1u;i<n;++i){
Z1 val=dd[i];Z0 j=i;
while(j>0u&&dd[j-1u]>val){
dd[j]=dd[j-1u];--j;
}
dd[j]=val;}
for(Z0 i=0;i<n;++i){
if(!(dd[i]>=-3e38f&&dd[i]<=3e38f)){
return false;}
}
for(Z0 r=0;r<ZB;++r){
ub_eig.Z8(ZJ+r,dd[ZJ+r]);}
return true;}


__aicore__ inline void EvalSturm4(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_neg,
Z1*b_pval,
ZG*b_pexp)const{
#if EIGH_K96_PASS
if constexpr(SOLVE_MODE==3u){EvalSturm4Fast(d0,de_pairs,b_mid,b_neg,b_pval,b_pexp);return;}
#else
if constexpr(FAST200||SOLVE_MODE==3u){EvalSturm4Fast(d0,de_pairs,b_mid,b_neg,b_pval,b_pexp);return;}
#endif
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;
Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;
Z0 s0=(p0<0.0f)?1u:0u,c0=(256u<<16)|s0;Z0 s1=(p1<0.0f)?1u:0u,c1=(256u<<16)|s1;
Z0 s2=(p2<0.0f)?1u:0u,c2=(256u<<16)|s2;Z0 s3=(p3<0.0f)?1u:0u,c3=(256u<<16)|s3;
Z0 i0=1u;if((n&1u)==0u){
Z1 d_1=de_pairs[0],e2_1=de_pairs[1];Z1 t10=d_1-m0,t11=d_1-m1,t12=d_1-m2,t13=d_1-m3;
Z1 pn0=t10*p0-e2_1*pp0;Z1 pn1=t11*p1-e2_1*pp1;
Z1 pn2=t12*p2-e2_1*pp2;Z1 pn3=t13*p3-e2_1*pp3;
pp0=p0;p0=pn0;pp1=p1;p1=pn1;
pp2=p2;p2=pn2;pp3=p3;p3=pn3;
Z0 sn0=(pn0<=0.0f)?1u:0u;c0+=(sn0^s0);s0=sn0;Z0 sn1=(pn1<=0.0f)?1u:0u;c1+=(sn1^s1);s1=sn1;
Z0 sn2=(pn2<=0.0f)?1u:0u;c2+=(sn2^s2);s2=sn2;Z0 sn3=(pn3<=0.0f)?1u:0u;c3+=(sn3^s3);s3=sn3;
i0=2u;}
if constexpr(SOLVE_MODE==3u){
const Z1*de=de_pairs+(i0-1u)*2u;for(;i0+12u<=n;i0+=12u){
for(Z0 k=0;k<3u;++k,de+=8u){
{
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;Z0 sa0=(pp0<=0.0f)?1u:0u;c0+=(sa0^s0);
Z0 sa1=(pp1<=0.0f)?1u:0u;c1+=(sa1^s1);Z0 sa2=(pp2<=0.0f)?1u:0u;c2+=(sa2^s2);
Z0 sa3=(pp3<=0.0f)?1u:0u;c3+=(sa3^s3);Z0 sb0=(p0<=0.0f)?1u:0u;c0+=(sb0^sa0);s0=sb0;
Z0 sb1=(p1<=0.0f)?1u:0u;c1+=(sb1^sa1);s1=sb1;Z0 sb2=(p2<=0.0f)?1u:0u;c2+=(sb2^sa2);s2=sb2;
Z0 sb3=(p3<=0.0f)?1u:0u;c3+=(sb3^sa3);s3=sb3;}
{
Z1 d_a=de[4],e2_a=de[5],d_b=de[6],e2_b=de[7];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;Z0 sa0=(pp0<=0.0f)?1u:0u;c0+=(sa0^s0);
Z0 sa1=(pp1<=0.0f)?1u:0u;c1+=(sa1^s1);Z0 sa2=(pp2<=0.0f)?1u:0u;c2+=(sa2^s2);
Z0 sa3=(pp3<=0.0f)?1u:0u;c3+=(sa3^s3);Z0 sb0=(p0<=0.0f)?1u:0u;c0+=(sb0^sa0);s0=sb0;
Z0 sb1=(p1<=0.0f)?1u:0u;c1+=(sb1^sa1);s1=sb1;Z0 sb2=(p2<=0.0f)?1u:0u;c2+=(sb2^sa2);s2=sb2;
Z0 sb3=(p3<=0.0f)?1u:0u;c3+=(sb3^sa3);s3=sb3;}
}
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if((ap0>1e12f)|(ap0<1e-12f)|(ap1>1e12f)|(ap1<1e-12f)|(ap2>1e12f)|(ap2<1e-12f)|(ap3>1e12f)|(ap3<1e-12f)){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;c0+=(1u<<16);}else if(ap0<1e-12f){p0=(p0==0.0f)?((s0!=0u)?-1e-15f:1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;c0-=(1u<<16);}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;c1+=(1u<<16);}else if(ap1<1e-12f){p1=(p1==0.0f)?((s1!=0u)?-1e-15f:1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;c1-=(1u<<16);}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;c2+=(1u<<16);}else if(ap2<1e-12f){p2=(p2==0.0f)?((s2!=0u)?-1e-15f:1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;c2-=(1u<<16);}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;c3+=(1u<<16);}else if(ap3<1e-12f){p3=(p3==0.0f)?((s3!=0u)?-1e-15f:1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;c3-=(1u<<16);}
}
}
for(;i0+1u<n;i0+=2u,de+=4u){
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;Z0 sa0=(pp0<=0.0f)?1u:0u;c0+=(sa0^s0);
Z0 sa1=(pp1<=0.0f)?1u:0u;c1+=(sa1^s1);Z0 sa2=(pp2<=0.0f)?1u:0u;c2+=(sa2^s2);
Z0 sa3=(pp3<=0.0f)?1u:0u;c3+=(sa3^s3);Z0 sb0=(p0<=0.0f)?1u:0u;c0+=(sb0^sa0);s0=sb0;
Z0 sb1=(p1<=0.0f)?1u:0u;c1+=(sb1^sa1);s1=sb1;Z0 sb2=(p2<=0.0f)?1u:0u;c2+=(sb2^sa2);s2=sb2;
Z0 sb3=(p3<=0.0f)?1u:0u;c3+=(sb3^sa3);s3=sb3;}
}else{
for(;i0<n;i0+=12u){
Z0 i_lim=(i0+12u<n)?(i0+12u):n;const Z1*de=de_pairs+(i0-1u)*2u;
Z0 i=i0;for(;i+3u<i_lim;i+=4u,de+=8u){
{
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];Z1 ta0=d_a-m0,ta1=d_a-m1,ta2=d_a-m2,ta3=d_a-m3;
Z1 tb0=d_b-m0,tb1=d_b-m1,tb2=d_b-m2,tb3=d_b-m3;pp0=ta0*p0-e2_a*pp0;
pp1=ta1*p1-e2_a*pp1;pp2=ta2*p2-e2_a*pp2;
pp3=ta3*p3-e2_a*pp3;p0=tb0*pp0-e2_b*p0;
p1=tb1*pp1-e2_b*p1;p2=tb2*pp2-e2_b*p2;
p3=tb3*pp3-e2_b*p3;Z0 sa0=(pp0<=0.0f)?1u:0u;c0+=(sa0^s0);
Z0 sa1=(pp1<=0.0f)?1u:0u;c1+=(sa1^s1);Z0 sa2=(pp2<=0.0f)?1u:0u;c2+=(sa2^s2);
Z0 sa3=(pp3<=0.0f)?1u:0u;c3+=(sa3^s3);Z0 sb0=(p0<=0.0f)?1u:0u;c0+=(sb0^sa0);s0=sb0;
Z0 sb1=(p1<=0.0f)?1u:0u;c1+=(sb1^sa1);s1=sb1;Z0 sb2=(p2<=0.0f)?1u:0u;c2+=(sb2^sa2);s2=sb2;
Z0 sb3=(p3<=0.0f)?1u:0u;c3+=(sb3^sa3);s3=sb3;}
{
Z1 d_a=de[4],e2_a=de[5],d_b=de[6],e2_b=de[7];Z1 ta0=d_a-m0,ta1=d_a-m1,ta2=d_a-m2,ta3=d_a-m3;
Z1 tb0=d_b-m0,tb1=d_b-m1,tb2=d_b-m2,tb3=d_b-m3;pp0=ta0*p0-e2_a*pp0;
pp1=ta1*p1-e2_a*pp1;pp2=ta2*p2-e2_a*pp2;
pp3=ta3*p3-e2_a*pp3;p0=tb0*pp0-e2_b*p0;
p1=tb1*pp1-e2_b*p1;p2=tb2*pp2-e2_b*p2;
p3=tb3*pp3-e2_b*p3;Z0 sa0=(pp0<=0.0f)?1u:0u;c0+=(sa0^s0);
Z0 sa1=(pp1<=0.0f)?1u:0u;c1+=(sa1^s1);Z0 sa2=(pp2<=0.0f)?1u:0u;c2+=(sa2^s2);
Z0 sa3=(pp3<=0.0f)?1u:0u;c3+=(sa3^s3);Z0 sb0=(p0<=0.0f)?1u:0u;c0+=(sb0^sa0);s0=sb0;
Z0 sb1=(p1<=0.0f)?1u:0u;c1+=(sb1^sa1);s1=sb1;Z0 sb2=(p2<=0.0f)?1u:0u;c2+=(sb2^sa2);s2=sb2;
Z0 sb3=(p3<=0.0f)?1u:0u;c3+=(sb3^sa3);s3=sb3;}
}
for(;i+1u<i_lim;i+=2u,de+=4u){
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];Z1 ta0=d_a-m0,ta1=d_a-m1,ta2=d_a-m2,ta3=d_a-m3;
Z1 tb0=d_b-m0,tb1=d_b-m1,tb2=d_b-m2,tb3=d_b-m3;pp0=ta0*p0-e2_a*pp0;
pp1=ta1*p1-e2_a*pp1;pp2=ta2*p2-e2_a*pp2;
pp3=ta3*p3-e2_a*pp3;p0=tb0*pp0-e2_b*p0;
p1=tb1*pp1-e2_b*p1;p2=tb2*pp2-e2_b*p2;
p3=tb3*pp3-e2_b*p3;Z0 sa0=(pp0<=0.0f)?1u:0u;c0+=(sa0^s0);
Z0 sa1=(pp1<=0.0f)?1u:0u;c1+=(sa1^s1);Z0 sa2=(pp2<=0.0f)?1u:0u;c2+=(sa2^s2);
Z0 sa3=(pp3<=0.0f)?1u:0u;c3+=(sa3^s3);Z0 sb0=(p0<=0.0f)?1u:0u;c0+=(sb0^sa0);s0=sb0;
Z0 sb1=(p1<=0.0f)?1u:0u;c1+=(sb1^sa1);s1=sb1;Z0 sb2=(p2<=0.0f)?1u:0u;c2+=(sb2^sa2);s2=sb2;
Z0 sb3=(p3<=0.0f)?1u:0u;c3+=(sb3^sa3);s3=sb3;}
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if((ap0>1e12f)|(ap0<1e-12f)|(ap1>1e12f)|(ap1<1e-12f)|(ap2>1e12f)|(ap2<1e-12f)|(ap3>1e12f)|(ap3<1e-12f)){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;c0+=(1u<<16);}else if(ap0<1e-12f){p0=(p0==0.0f)?((s0!=0u)?-1e-15f:1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;c0-=(1u<<16);}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;c1+=(1u<<16);}else if(ap1<1e-12f){p1=(p1==0.0f)?((s1!=0u)?-1e-15f:1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;c1-=(1u<<16);}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;c2+=(1u<<16);}else if(ap2<1e-12f){p2=(p2==0.0f)?((s2!=0u)?-1e-15f:1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;c2-=(1u<<16);}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;c3+=(1u<<16);}else if(ap3<1e-12f){p3=(p3==0.0f)?((s3!=0u)?-1e-15f:1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;c3-=(1u<<16);}
}
}
}
b_neg[0]=c0&0xFFFFu;b_neg[1]=c1&0xFFFFu;b_neg[2]=c2&0xFFFFu;b_neg[3]=c3&0xFFFFu;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);
b_pexp[0]=Z5<ZG>(c0>>16)-256;b_pexp[1]=Z5<ZG>(c1>>16)-256;
b_pexp[2]=Z5<ZG>(c2>>16)-256;b_pexp[3]=Z5<ZG>(c3>>16)-256;}

__aicore__ inline void EvalPoly4(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_sign_neg,
Z1*b_pval,
ZG*b_pexp)const{
#if EIGH_K96_PASS
if constexpr(SOLVE_MODE==3u){EvalPoly4Fast(d0,de_pairs,b_mid,b_sign_neg,b_pval,b_pexp);return;}
#else
if constexpr(FAST200||SOLVE_MODE==3u){EvalPoly4Fast(d0,de_pairs,b_mid,b_sign_neg,b_pval,b_pexp);return;}
#endif
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;
Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;
ZG exp0=0,exp1=0,exp2=0,exp3=0;Z0 i0=1u;
if((n&1u)==0u){
Z1 d_1=de_pairs[0],e2_1=de_pairs[1];Z1 pn0=(d_1-m0)*p0-e2_1*pp0;pp0=p0;p0=pn0;
Z1 pn1=(d_1-m1)*p1-e2_1*pp1;pp1=p1;p1=pn1;Z1 pn2=(d_1-m2)*p2-e2_1*pp2;pp2=p2;p2=pn2;
Z1 pn3=(d_1-m3)*p3-e2_1*pp3;pp3=p3;p3=pn3;i0=2u;
}
if constexpr(SOLVE_MODE==3u){
const Z1*de=de_pairs+(i0-1u)*2u;for(;i0+12u<=n;i0+=12u){
for(Z0 k=0;k<3u;++k,de+=8u){
{
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;}
{
Z1 d_a=de[4],e2_a=de[5],d_b=de[6],e2_b=de[7];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;}
}
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if((ap0>1e12f)|(ap0<1e-12f)|(ap1>1e12f)|(ap1<1e-12f)|(ap2>1e12f)|(ap2<1e-12f)|(ap3>1e12f)|(ap3<1e-12f)){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;++exp0;}else if(ap0<1e-12f){p0=(p0==0.0f)?((pp0<=0.0f)?1e-15f:-1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;--exp0;}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;++exp1;}else if(ap1<1e-12f){p1=(p1==0.0f)?((pp1<=0.0f)?1e-15f:-1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;--exp1;}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;++exp2;}else if(ap2<1e-12f){p2=(p2==0.0f)?((pp2<=0.0f)?1e-15f:-1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;--exp2;}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;++exp3;}else if(ap3<1e-12f){p3=(p3==0.0f)?((pp3<=0.0f)?1e-15f:-1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;--exp3;}
}
}
for(;i0+1u<n;i0+=2u,de+=4u){
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;}
}else{
for(;i0<n;i0+=12u){
Z0 i_lim=(i0+12u<n)?(i0+12u):n;const Z1*de=de_pairs+(i0-1u)*2u;
Z0 i=i0;for(;i+3u<i_lim;i+=4u,de+=8u){
{
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;}
{
Z1 d_a=de[4],e2_a=de[5],d_b=de[6],e2_b=de[7];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;}
}
for(;i+1u<i_lim;i+=2u,de+=4u){
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;}
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if((ap0>1e12f)|(ap0<1e-12f)|(ap1>1e12f)|(ap1<1e-12f)|(ap2>1e12f)|(ap2<1e-12f)|(ap3>1e12f)|(ap3<1e-12f)){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;++exp0;}else if(ap0<1e-12f){p0=(p0==0.0f)?((pp0<=0.0f)?1e-15f:-1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;--exp0;}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;++exp1;}else if(ap1<1e-12f){p1=(p1==0.0f)?((pp1<=0.0f)?1e-15f:-1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;--exp1;}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;++exp2;}else if(ap2<1e-12f){p2=(p2==0.0f)?((pp2<=0.0f)?1e-15f:-1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;--exp2;}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;++exp3;}else if(ap3<1e-12f){p3=(p3==0.0f)?((pp3<=0.0f)?1e-15f:-1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;--exp3;}
}
}
}
b_sign_neg[0]=(p0<=0.0f)?1u:0u;b_sign_neg[1]=(p1<=0.0f)?1u:0u;
b_sign_neg[2]=(p2<=0.0f)?1u:0u;b_sign_neg[3]=(p3<=0.0f)?1u:0u;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);
b_pexp[0]=exp0;b_pexp[1]=exp1;b_pexp[2]=exp2;b_pexp[3]=exp3;}

__aicore__ inline void EvalPoly4SameScaleAs8(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_sign_neg,
Z1*b_pval,
ZG*b_pexp)const{
#if EIGH_K96_PASS
if constexpr(SOLVE_MODE==3u){EvalPoly4SameScaleAs8Fast(d0,de_pairs,b_mid,b_sign_neg,b_pval,b_pexp);return;}
#else
if constexpr(FAST200||SOLVE_MODE==3u){EvalPoly4SameScaleAs8Fast(d0,de_pairs,b_mid,b_sign_neg,b_pval,b_pexp);return;}
#endif
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;
Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;
ZG exp0=0,exp1=0,exp2=0,exp3=0;Z0 i0=1u;
if((n&1u)==0u){
Z1 d_1=de_pairs[0],e2_1=de_pairs[1];Z1 pn0=(d_1-m0)*p0-e2_1*pp0;pp0=p0;p0=pn0;
Z1 pn1=(d_1-m1)*p1-e2_1*pp1;pp1=p1;p1=pn1;Z1 pn2=(d_1-m2)*p2-e2_1*pp2;pp2=p2;p2=pn2;
Z1 pn3=(d_1-m3)*p3-e2_1*pp3;pp3=p3;p3=pn3;i0=2u;
}

for(;i0<n;i0+=12u){
Z0 i_lim=(i0+12u<n)?(i0+12u):n;const Z1*de=de_pairs+(i0-1u)*2u;
Z0 i=i0;for(;i+3u<i_lim;i+=4u,de+=8u){
{
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;}
{
Z1 d_a=de[4],e2_a=de[5],d_b=de[6],e2_b=de[7];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;}
}
for(;i+1u<i_lim;i+=2u,de+=4u){
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;}
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if((ap0>1e12f)|(ap0<1e-12f)|(ap1>1e12f)|(ap1<1e-12f)|(ap2>1e12f)|(ap2<1e-12f)|(ap3>1e12f)|(ap3<1e-12f)){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;++exp0;}else if(ap0<1e-12f){p0=(p0==0.0f)?((pp0<=0.0f)?1e-15f:-1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;--exp0;}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;++exp1;}else if(ap1<1e-12f){p1=(p1==0.0f)?((pp1<=0.0f)?1e-15f:-1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;--exp1;}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;++exp2;}else if(ap2<1e-12f){p2=(p2==0.0f)?((pp2<=0.0f)?1e-15f:-1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;--exp2;}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;++exp3;}else if(ap3<1e-12f){p3=(p3==0.0f)?((pp3<=0.0f)?1e-15f:-1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;--exp3;}
}
}

b_sign_neg[0]=(p0<=0.0f)?1u:0u;b_sign_neg[1]=(p1<=0.0f)?1u:0u;
b_sign_neg[2]=(p2<=0.0f)?1u:0u;b_sign_neg[3]=(p3<=0.0f)?1u:0u;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);
b_pexp[0]=exp0;b_pexp[1]=exp1;b_pexp[2]=exp2;b_pexp[3]=exp3;}

__aicore__ inline void EvalPoly8(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_sign_neg,
Z1*b_pval,
ZG*b_pexp)const{
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];
Z1 m4=b_mid[4],m5=b_mid[5],m6=b_mid[6],m7=b_mid[7];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 pp4=1.0f,pp5=1.0f,pp6=1.0f,pp7=1.0f;Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;
Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;
Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;Z1 p4=d0-m4;if(Z7(p4)<1e-20f)p4=-1e-20f;
Z1 p5=d0-m5;if(Z7(p5)<1e-20f)p5=-1e-20f;Z1 p6=d0-m6;if(Z7(p6)<1e-20f)p6=-1e-20f;
Z1 p7=d0-m7;if(Z7(p7)<1e-20f)p7=-1e-20f;
ZG exp0=0,exp1=0,exp2=0,exp3=0,exp4=0,exp5=0,exp6=0,exp7=0;Z0 i0=1u;
if((n&1u)==0u){
Z1 d_1=de_pairs[0],e2_1=de_pairs[1];Z1 pn0=(d_1-m0)*p0-e2_1*pp0;pp0=p0;p0=pn0;
Z1 pn1=(d_1-m1)*p1-e2_1*pp1;pp1=p1;p1=pn1;Z1 pn2=(d_1-m2)*p2-e2_1*pp2;pp2=p2;p2=pn2;
Z1 pn3=(d_1-m3)*p3-e2_1*pp3;pp3=p3;p3=pn3;Z1 pn4=(d_1-m4)*p4-e2_1*pp4;pp4=p4;p4=pn4;
Z1 pn5=(d_1-m5)*p5-e2_1*pp5;pp5=p5;p5=pn5;Z1 pn6=(d_1-m6)*p6-e2_1*pp6;pp6=p6;p6=pn6;
Z1 pn7=(d_1-m7)*p7-e2_1*pp7;pp7=p7;p7=pn7;i0=2u;
}
for(;i0<n;i0+=12u){
Z0 i_lim=(i0+12u<n)?(i0+12u):n;const Z1*de=de_pairs+(i0-1u)*2u;
Z0 i=i0;for(;i+1u<i_lim;i+=2u,de+=4u){
Z1 d_a=de[0],e2_a=de[1],d_b=de[2],e2_b=de[3];pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-e2_b*p0;
pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-e2_b*p1;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-e2_b*p2;
pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-e2_b*p3;pp4=(d_a-m4)*p4-e2_a*pp4;p4=(d_b-m4)*pp4-e2_b*p4;
pp5=(d_a-m5)*p5-e2_a*pp5;p5=(d_b-m5)*pp5-e2_b*p5;pp6=(d_a-m6)*p6-e2_a*pp6;p6=(d_b-m6)*pp6-e2_b*p6;
pp7=(d_a-m7)*p7-e2_a*pp7;p7=(d_b-m7)*pp7-e2_b*p7;}
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
Z1 ap4=Z7(p4),ap5=Z7(p5),ap6=Z7(p6),ap7=Z7(p7);
if((ap0>1e12f)|(ap0<1e-12f)|(ap1>1e12f)|(ap1<1e-12f)|(ap2>1e12f)|(ap2<1e-12f)|(ap3>1e12f)|(ap3<1e-12f)|
(ap4>1e12f)|(ap4<1e-12f)|(ap5>1e12f)|(ap5<1e-12f)|(ap6>1e12f)|(ap6<1e-12f)|(ap7>1e12f)|(ap7<1e-12f)){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;++exp0;}else if(ap0<1e-12f){p0=(p0==0.0f)?((pp0<=0.0f)?1e-15f:-1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;--exp0;}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;++exp1;}else if(ap1<1e-12f){p1=(p1==0.0f)?((pp1<=0.0f)?1e-15f:-1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;--exp1;}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;++exp2;}else if(ap2<1e-12f){p2=(p2==0.0f)?((pp2<=0.0f)?1e-15f:-1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;--exp2;}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;++exp3;}else if(ap3<1e-12f){p3=(p3==0.0f)?((pp3<=0.0f)?1e-15f:-1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;--exp3;}
if(ap4>1e12f){p4*=5.4210109e-20f;pp4*=5.4210109e-20f;++exp4;}else if(ap4<1e-12f){p4=(p4==0.0f)?((pp4<=0.0f)?1e-15f:-1e-15f):(p4*1.8446744e19f);pp4*=1.8446744e19f;--exp4;}
if(ap5>1e12f){p5*=5.4210109e-20f;pp5*=5.4210109e-20f;++exp5;}else if(ap5<1e-12f){p5=(p5==0.0f)?((pp5<=0.0f)?1e-15f:-1e-15f):(p5*1.8446744e19f);pp5*=1.8446744e19f;--exp5;}
if(ap6>1e12f){p6*=5.4210109e-20f;pp6*=5.4210109e-20f;++exp6;}else if(ap6<1e-12f){p6=(p6==0.0f)?((pp6<=0.0f)?1e-15f:-1e-15f):(p6*1.8446744e19f);pp6*=1.8446744e19f;--exp6;}
if(ap7>1e12f){p7*=5.4210109e-20f;pp7*=5.4210109e-20f;++exp7;}else if(ap7<1e-12f){p7=(p7==0.0f)?((pp7<=0.0f)?1e-15f:-1e-15f):(p7*1.8446744e19f);pp7*=1.8446744e19f;--exp7;}
}
}
b_sign_neg[0]=(p0<=0.0f)?1u:0u;b_sign_neg[1]=(p1<=0.0f)?1u:0u;
b_sign_neg[2]=(p2<=0.0f)?1u:0u;b_sign_neg[3]=(p3<=0.0f)?1u:0u;
b_sign_neg[4]=(p4<=0.0f)?1u:0u;b_sign_neg[5]=(p5<=0.0f)?1u:0u;
b_sign_neg[6]=(p6<=0.0f)?1u:0u;b_sign_neg[7]=(p7<=0.0f)?1u:0u;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);
b_pval[4]=Z7(p4);b_pval[5]=Z7(p5);b_pval[6]=Z7(p6);b_pval[7]=Z7(p7);
b_pexp[0]=exp0;b_pexp[1]=exp1;b_pexp[2]=exp2;b_pexp[3]=exp3;
b_pexp[4]=exp4;b_pexp[5]=exp5;b_pexp[6]=exp6;b_pexp[7]=exp7;}

__aicore__ inline void EvalSturm8(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_neg,
Z1*b_pval,
ZG*b_pexp,
Z0 active_lanes=8u)const{
EvalSturm4(d0,de_pairs,b_mid,b_neg,b_pval,b_pexp);if(active_lanes>4u){
EvalSturm4(d0,de_pairs,b_mid+4u,b_neg+4u,b_pval+4u,b_pexp+4u);}
}

__aicore__ inline bool ComputeSecantWeight(Z1 plo,ZG elo,Z1 phi,ZG ehi,Z1&t_out)const{
if(elo<=-10000||ehi<=-10000)return false;ZG ediff=elo-ehi;
if(ediff<-1||ediff>1)return false;Z1 fl=plo;
Z1 fh=phi;if(ediff==1)fh*=5.4210109e-20f;
else if(ediff==-1)fl*=5.4210109e-20f;Z1 denom=fl+fh;
if(!(denom>1e-35f))return false;t_out=fl/denom;
return true;}


__aicore__ inline void SolveTridiagSturm(Z1*work){
Z0 w_cap=(n<=256u)?256u:512u;Z1*dd=work;
Z1*ee=work+w_cap;Z1*de_pairs=work+2u*w_cap;
Z1*q_lo=work+4u*w_cap;Z1*q_hi=q_lo+64;
Z1*iso_lo=q_hi+64;Z1*iso_hi=iso_lo+64;

Z0*u_base=reinterpret_cast<Z0*>(iso_hi+64);Z0*q_lc=u_base;
Z0*q_hc=q_lc+64;Z0*q_depth=q_hc+64;
Z0*act_q=q_depth+64;Z0*act_depth=act_q+64;


Z1*q_plo=reinterpret_cast<Z1*>(act_depth+64);Z1*q_phi=q_plo+64;
Z1*iso_plo=q_phi+64;Z1*iso_phi=iso_plo+64;
ZG*q_elo=reinterpret_cast<ZG*>(iso_phi+64);ZG*q_ehi=q_elo+64;
ZG*iso_elo=q_ehi+64;ZG*iso_ehi=iso_elo+64;
int8_t*iso_stag=reinterpret_cast<int8_t*>(iso_ehi+64);

for(Z0 r=0;r<ZB;++r){
iso_plo[r]=0.0f;iso_phi[r]=0.0f;iso_elo[r]=-30000;iso_ehi[r]=-30000;
iso_stag[r]=0;}

Z1 gersh_min=0.0f;Z1 gersh_max=0.0f;
Z1 max_elem=0.0f;bool is_diag=true;
Z1 prev_abs_e=0.0f;for(Z0 i=0;i<n;++i){
Z1 d_i,e_i;if constexpr(SOLVE_MODE!=2u){
d_i=ub_diag.ZF(i);e_i=ub_off.ZF(i);
dd[i]=d_i;ee[i]=e_i;
}else{
d_i=dd[i];e_i=ee[i];
}
if(i>0u){
de_pairs[(i-1u)*2u+0u]=d_i;}
Z1 abs_e=0.0f;if(i+1u<n){
de_pairs[i*2u+1u]=e_i*e_i;if(e_i!=0.0f){
is_diag=false;}
abs_e=Z7(e_i);}
Z1 rad=prev_abs_e+abs_e;prev_abs_e=abs_e;
Z1 lo_i=d_i-rad;Z1 hi_i=d_i+rad;
if(i==0u||lo_i<gersh_min)gersh_min=lo_i;if(i==0u||hi_i>gersh_max)gersh_max=hi_i;
Z1 abs_d=Z7(d_i);if(abs_d>max_elem)max_elem=abs_d;
if(abs_e>max_elem)max_elem=abs_e;}
if(is_diag){
Z1*sd=work+2u*w_cap;for(Z0 i=0;i<n;++i){
Z1 val=dd[i];Z0 j=i;
while(j>0u&&sd[j-1u]>val){
sd[j]=sd[j-1u];--j;
}
sd[j]=val;}
for(Z0 r=0;r<ZB;++r){
ub_eig.Z8(ZJ+r,sd[ZJ+r]);}
work[1023]=-1.0f;return;
}
work[1023]=max_elem;if(ZB==0u){
return;}
Z1 pad=1e-5f*(Z7(gersh_min)+Z7(gersh_max))+1e-6f;gersh_min-=pad;
gersh_max+=pad;

Z0 row_end=ZJ+ZB;Z0 ext_start=(ZJ>0u)?(ZJ-1u):0u;
Z0 ext_end=(row_end<n)?(row_end+1u):n;
Z1 left_nb_lo=gersh_min,left_nb_hi=gersh_min,left_nb_plo=0.0f,left_nb_phi=0.0f;
ZG left_nb_elo=-30000,left_nb_ehi=-30000;
Z1 right_nb_lo=gersh_max,right_nb_hi=gersh_max,right_nb_plo=0.0f,right_nb_phi=0.0f;
ZG right_nb_elo=-30000,right_nb_ehi=-30000;Z1 tol_cluster=1.0e-6f*max_elem;
Z1 tol_single=1.0e-5f*max_elem;

Z0 q_head=0u;Z0 q_tail=1u;
Z0 q_count=1u;q_lo[0]=gersh_min;
q_hi[0]=gersh_max;q_plo[0]=0.0f;q_phi[0]=0.0f;
q_elo[0]=-30000;q_ehi[0]=-30000;q_lc[0]=0u;
q_hc[0]=n;q_depth[0]=0u;

Z1 b_lo[4],b_hi[4],b_mid[8];Z1 b_plo[4],b_phi[4],b_pval[8];
ZG b_elo[4],b_ehi[4],b_pexp[8];Z0 b_lc[4],b_hc[4],b_depth[4],b_kp[8],b_pos[8],b_neg[8];


while(q_count>0u){
Z0 bcnt=0u;while(q_count>0u&&bcnt<4u){
Z1 lo=q_lo[q_head],hi=q_hi[q_head];Z1 plo=q_plo[q_head],phi=q_phi[q_head];
ZG elo=q_elo[q_head],ehi=q_ehi[q_head];Z0 lc=q_lc[q_head],hc=q_hc[q_head],depth=q_depth[q_head];
q_head=(q_head+1u)&63u;--q_count;

if(hc-lc==1u){
if(lc>=ZJ&&lc<row_end){
Z0 r=lc-ZJ;iso_lo[r]=lo;iso_hi[r]=hi;
iso_plo[r]=plo;iso_phi[r]=phi;iso_elo[r]=elo;iso_ehi[r]=ehi;
}else if(ZJ>0u&&lc==ZJ-1u){
left_nb_lo=lo;left_nb_hi=hi;left_nb_plo=plo;left_nb_phi=phi;left_nb_elo=elo;left_nb_ehi=ehi;
}else if(row_end<n&&lc==row_end){
right_nb_lo=lo;right_nb_hi=hi;right_nb_plo=plo;right_nb_phi=phi;right_nb_elo=elo;right_nb_ehi=ehi;}
continue;}
Z1 mid=0.5f*lo+0.5f*hi;if(depth>=32u||mid==lo||mid==hi||(hi-lo)<=tol_cluster+1e-30f){
Z0 s_idx=(lc>ZJ)?lc:ZJ;Z0 e_idx=(hc<row_end)?hc:row_end;
for(Z0 j=s_idx;j<e_idx;++j){
iso_lo[j-ZJ]=mid;iso_hi[j-ZJ]=mid;
}
if(ZJ>0u&&lc<=ZJ-1u&&hc>ZJ-1u){left_nb_lo=mid;left_nb_hi=mid;}
if(row_end<n&&lc<=row_end&&hc>row_end){right_nb_lo=mid;right_nb_hi=mid;}
continue;}
b_lo[bcnt]=lo;b_hi[bcnt]=hi;b_plo[bcnt]=plo;b_phi[bcnt]=phi;
b_elo[bcnt]=elo;b_ehi[bcnt]=ehi;b_lc[bcnt]=lc;b_hc[bcnt]=hc;
b_depth[bcnt]=depth;++bcnt;
}
if(bcnt==0u)break;

Z0 best_b3=0u;if(bcnt==3u){
Z0 c0=((b_hc[0]<ext_end)?b_hc[0]:ext_end)-((b_lc[0]>ext_start)?b_lc[0]:ext_start);
Z0 c1=((b_hc[1]<ext_end)?b_hc[1]:ext_end)-((b_lc[1]>ext_start)?b_lc[1]:ext_start);
Z0 c2=((b_hc[2]<ext_end)?b_hc[2]:ext_end)-((b_lc[2]>ext_start)?b_lc[2]:ext_start);
best_b3=(c1>c0)?((c2>c1)?2u:1u):((c2>c0)?2u:0u);}
Z0 lane_pos=0u;for(Z0 b=0;b<bcnt;++b){
Z0 kp=(bcnt==1u)?4u:((bcnt==2u)?2u:((bcnt==3u&&b==best_b3)?2u:1u));b_kp[b]=kp;
b_pos[b]=lane_pos;Z1 lo=b_lo[b],hi=b_hi[b];
Z0 lc=b_lc[b],hc=b_hc[b];Z1 inv_kp1=(kp==4u)?0.2f:((kp==2u)?(1.0f/3.0f):0.5f);
Z1 step=(b_hi[b]-lo)*inv_kp1;for(Z0 k=0;k<kp;++k){
b_mid[lane_pos+k]=lo+step*Z5<Z1>(Z5<ZG>(k+1u));}
lane_pos+=kp;}

EvalSturm4(dd[0],de_pairs,b_mid,b_neg,b_pval,b_pexp);

for(Z0 b=0;b<bcnt;++b){
Z0 lc=b_lc[b],hc=b_hc[b],kp=b_kp[b],pos=b_pos[b];Z0 next_depth=b_depth[b]+((kp>=3u)?2u:1u);
Z1 prev_v=b_lo[b],prev_p=b_plo[b];ZG prev_e=b_elo[b];
Z0 prev_c=lc;for(Z0 k=0;k<=kp;++k){
Z1 cur_v=(k<kp)?b_mid[pos+k]:b_hi[b];Z1 cur_p=(k<kp)?b_pval[pos+k]:b_phi[b];
ZG cur_e=(k<kp)?b_pexp[pos+k]:b_ehi[b];Z0 cur_c=(k<kp)?b_neg[pos+k]:hc;
if(cur_c<prev_c)cur_c=prev_c;if(cur_c>hc)cur_c=hc;
if(prev_c<cur_c&&cur_c>ext_start&&prev_c<ext_end){
if(cur_c-prev_c==1u){
if(prev_c>=ZJ&&prev_c<row_end){
Z0 r=prev_c-ZJ;iso_lo[r]=prev_v;iso_hi[r]=cur_v;
iso_plo[r]=prev_p;iso_phi[r]=cur_p;iso_elo[r]=prev_e;iso_ehi[r]=cur_e;
}else if(ZJ>0u&&prev_c==ZJ-1u){
left_nb_lo=prev_v;left_nb_hi=cur_v;left_nb_plo=prev_p;left_nb_phi=cur_p;left_nb_elo=prev_e;left_nb_ehi=cur_e;
}else if(row_end<n&&prev_c==row_end){
right_nb_lo=prev_v;right_nb_hi=cur_v;right_nb_plo=prev_p;right_nb_phi=cur_p;right_nb_elo=prev_e;right_nb_ehi=cur_e;
}
}else{
q_lo[q_tail]=prev_v;q_hi[q_tail]=cur_v;q_plo[q_tail]=prev_p;q_phi[q_tail]=cur_p;
q_elo[q_tail]=prev_e;q_ehi[q_tail]=cur_e;q_lc[q_tail]=prev_c;q_hc[q_tail]=cur_c;
q_depth[q_tail]=next_depth;q_tail=(q_tail+1u)&63u;
++q_count;}
}
prev_v=cur_v;prev_p=cur_p;prev_e=cur_e;prev_c=cur_c;}
}
}

if(ZJ>0u){
Z1 eig_l=left_nb_hi;Z1 t_nb=0.5f;
if(ComputeSecantWeight(left_nb_plo,left_nb_elo,left_nb_phi,left_nb_ehi,t_nb)){
eig_l=left_nb_lo+t_nb*(left_nb_hi-left_nb_lo);left_nb_hi=0.5f*(eig_l+left_nb_hi);
}
ub_eig.Z8(ZJ-1u,eig_l);}
if(row_end<n){
Z1 eig_r=right_nb_lo;Z1 t_nb=0.5f;
if(ComputeSecantWeight(right_nb_plo,right_nb_elo,right_nb_phi,right_nb_ehi,t_nb)){
eig_r=right_nb_lo+t_nb*(right_nb_hi-right_nb_lo);right_nb_lo=0.5f*(right_nb_lo+eig_r);
}
ub_eig.Z8(row_end,eig_r);}


Z1 iso_clo[64],iso_chi[64];for(Z0 r=0;r<ZB;++r){
Z1 lo=iso_lo[r],hi=iso_hi[r];Z1 t_s=0.5f;
if(ComputeSecantWeight(iso_plo[r],iso_elo[r],iso_phi[r],iso_ehi[r],t_s)){
Z1 est=lo+t_s*(hi-lo);iso_clo[r]=0.5f*(lo+est);
iso_chi[r]=0.5f*(est+hi);}else{
iso_clo[r]=lo;iso_chi[r]=hi;
}
}

Z0 a_head=0u,a_tail=0u,a_cnt=0u;Z1 max_gap_tol=4.5e-4f*max_elem;
Z1 min_sep_gap=0.0005f*max_elem;Z1 near_zero_thr=0.015f*max_elem;

for(Z0 r=0;r<ZB;++r){
act_depth[r]=0u;Z1 lo=iso_lo[r],hi=iso_hi[r];
Z1 mid=0.5f*lo+0.5f*hi;Z1 cur_tol=tol_single;
Z1 l_gap=(r>0u)?(iso_clo[r]-iso_chi[r-1u]):((ZJ>0u)?(iso_clo[r]-left_nb_hi):1.0e30f);
Z1 r_gap=((r+1u)<ZB)?(iso_clo[r+1u]-iso_chi[r]):((row_end<n)?(right_nb_lo-iso_chi[r]):1.0e30f);
Z1 gap=(l_gap<r_gap)?l_gap:r_gap;
if(gap>=min_sep_gap&&(Z7(mid)>=near_zero_thr||gap>=0.0015f*max_elem)){
Z1 ptol=0.030f*gap;if(ptol>max_gap_tol)ptol=max_gap_tol;
if(ptol>cur_tol)cur_tol=ptol;}
if(mid==lo||mid==hi||(hi-lo)<=cur_tol+1e-30f){
Z1 t_sec=0.5f;if(ComputeSecantWeight(iso_plo[r],iso_elo[r],iso_phi[r],iso_ehi[r],t_sec)){
mid=lo+t_sec*(hi-lo);}
ub_eig.Z8(ZJ+r,mid);}else{
act_q[a_tail]=r;a_tail=(a_tail+1u)&63u;
++a_cnt;}
}

Z0 r_batch[8];while(a_cnt>0u){
bool use_8way=(a_cnt>=5u);Z0 bcnt=use_8way?((a_cnt<8u)?a_cnt:8u):((a_cnt<4u)?a_cnt:4u);
Z0 lane_pos=0u;for(Z0 b=0;b<bcnt;++b){
Z0 r=act_q[a_head];a_head=(a_head+1u)&63u;
--a_cnt;r_batch[b]=r;
Z0 kp=use_8way
?((bcnt==5u&&b<3u)?2u:((bcnt==6u&&b<2u)?2u:((bcnt==7u&&b==0u)?2u:1u)))
:((bcnt==1u)?4u:((bcnt==2u)?2u:((bcnt==3u&&b==0u)?2u:1u)));b_kp[b]=kp;
b_pos[b]=lane_pos;Z1 lo=iso_lo[r],hi=iso_hi[r];
Z1 width=hi-lo;Z1 t_sec=0.5f;
bool has_sec=ComputeSecantWeight(iso_plo[r],iso_elo[r],iso_phi[r],iso_ehi[r],t_sec);if(kp==1u&&has_sec){
int8_t stg=iso_stag[r];
Z1 lg=(r>0u)?(iso_clo[r]-iso_chi[r-1u]):((ZJ>0u)?(iso_clo[r]-left_nb_hi):1.0e30f);
Z1 rg=((r+1u)<ZB)?(iso_clo[r+1u]-iso_chi[r]):((row_end<n)?(right_nb_lo-iso_chi[r]):1.0e30f);
Z1 gp=(lg<rg)?lg:rg;if(width>0.55f*gp||stg>=3||stg<=-3){
t_sec=0.5f;iso_stag[r]=0;
}else if(stg==2){
t_sec=(t_sec<0.5f)?(0.45f*t_sec):(t_sec-0.20f);if(t_sec<0.015f)t_sec=0.015f;
}else if(stg==-2){
t_sec=(t_sec>0.5f)?(1.0f-0.45f*(1.0f-t_sec)):(t_sec+0.20f);if(t_sec>0.985f)t_sec=0.985f;
}else{
if(stg==1){
t_sec=(t_sec>0.92f)?(1.0f-2.2f*(1.0f-t_sec)-0.012f):((t_sec<0.25f)?(0.65f*t_sec):(t_sec-0.045f));
}else if(stg==-1){
t_sec=(t_sec<0.08f)?(2.2f*t_sec+0.012f):((t_sec>0.75f)?(1.0f-0.65f*(1.0f-t_sec)):(t_sec+0.045f));}
if((ZJ+r==0u||ZJ+r+1u==n)&&width>8.0f*max_gap_tol){
t_sec=0.5f;}else{
constexpr Z1 clamp_lo=0.015f;if(t_sec<clamp_lo)t_sec=clamp_lo;
if(t_sec>1.0f-clamp_lo)t_sec=1.0f-clamp_lo;}
}
b_mid[lane_pos]=lo+t_sec*width;}else if(kp==2u&&has_sec){
Z1 tc=t_sec;if(tc<0.12f)tc=0.12f;
if(tc>0.88f)tc=0.88f;b_mid[lane_pos+0u]=lo+(tc-0.08f)*width;
b_mid[lane_pos+1u]=lo+(tc+0.08f)*width;}else{
Z1 inv_kp1=(kp==4u)?0.2f:((kp==2u)?(1.0f/3.0f):0.5f);Z1 step=width*inv_kp1;
for(Z0 k=0;k<kp;++k){
b_mid[lane_pos+k]=lo+step*Z5<Z1>(Z5<ZG>(k+1u));}
}
lane_pos+=kp;}

if(use_8way){
EvalPoly4(dd[0],de_pairs,b_mid,b_neg,b_pval,b_pexp);
EvalPoly4(dd[0],de_pairs,b_mid+4u,b_neg+4u,b_pval+4u,b_pexp+4u);}else{
EvalPoly4(dd[0],de_pairs,b_mid,b_neg,b_pval,b_pexp);}

for(Z0 b=0;b<bcnt;++b){
Z0 r=r_batch[b];Z0 kp=b_kp[b];
Z0 pos=b_pos[b];Z1 lo=iso_lo[r],hi=iso_hi[r];
Z1 plo=iso_plo[r],phi=iso_phi[r];ZG elo=iso_elo[r],ehi=iso_ehi[r];
Z0 target=ZJ+r+1u;Z1 new_lo=b_mid[pos+kp-1u];
Z1 new_plo=b_pval[pos+kp-1u];ZG new_elo=b_pexp[pos+kp-1u];
Z1 new_hi=hi;Z1 new_phi=phi;
ZG new_ehi=ehi;int8_t next_stag=(iso_stag[r]<0)?(iso_stag[r]-1):-1;
for(Z0 k=0;k<kp;++k){
if(b_neg[pos+k]==(target&1u)){
if(k==0u){
new_lo=lo;new_plo=plo;
new_elo=elo;next_stag=(iso_stag[r]>0)?(iso_stag[r]+1):1;
}else{
new_lo=b_mid[pos+k-1u];new_plo=b_pval[pos+k-1u];
new_elo=b_pexp[pos+k-1u];next_stag=0;
}
new_hi=b_mid[pos+k];new_phi=b_pval[pos+k];
new_ehi=b_pexp[pos+k];break;
}
}
iso_stag[r]=next_stag;iso_lo[r]=new_lo;iso_hi[r]=new_hi;
iso_plo[r]=new_plo;iso_phi[r]=new_phi;iso_elo[r]=new_elo;iso_ehi[r]=new_ehi;
act_depth[r]+=(kp>=7u)?3u:((kp>=3u)?2u:1u);Z1 next_mid=0.5f*new_lo+0.5f*new_hi;
Z1 t_sec=0.5f;bool has_sec=ComputeSecantWeight(new_plo,new_elo,new_phi,new_ehi,t_sec);
Z1 width=new_hi-new_lo;if(has_sec){
Z1 est=new_lo+t_sec*width;iso_clo[r]=0.5f*(new_lo+est);
iso_chi[r]=0.5f*(est+new_hi);}else{
iso_clo[r]=new_lo;iso_chi[r]=new_hi;
}
Z1 cur_tol=tol_single;
Z1 l_gap=(r>0u)?(iso_clo[r]-iso_chi[r-1u]):((ZJ>0u)?(iso_clo[r]-left_nb_hi):1.0e30f);
Z1 r_gap=((r+1u)<ZB)?(iso_clo[r+1u]-iso_chi[r]):((row_end<n)?(right_nb_lo-iso_chi[r]):1.0e30f);
Z1 gap=(l_gap<r_gap)?l_gap:r_gap;
if(gap>=min_sep_gap&&(Z7(next_mid)>=near_zero_thr||gap>=0.0015f*max_elem)){
Z1 ptol=0.030f*gap;if(ptol>max_gap_tol)ptol=max_gap_tol;
if(ptol>cur_tol)cur_tol=ptol;}
Z1 eff_width=has_sec?(2.2f*t_sec*(1.0f-t_sec)*width):width;
if(act_depth[r]>=32u||next_mid==new_lo||next_mid==new_hi||(next_stag>=-1&&next_stag<=1&&width<=((Z7(next_mid)>=near_zero_thr)?8.0f:4.0f)*cur_tol&&eff_width<=cur_tol+1e-30f)||width<=cur_tol+1e-30f){
if(has_sec){
next_mid=new_lo+t_sec*width;}
ub_eig.Z8(ZJ+r,next_mid);}else{
act_q[a_tail]=r;a_tail=(a_tail+1u)&63u;
++a_cnt;}
}
}
}
__aicore__ inline void ComputeEigenvalues(Z1*work){
if(group_lanes>=4u&&n>=40u&&ZB<=64u){
SolveTridiagSturm(work);return;
}
if(SolveTridiagPWK(work)){
return;}
SolveTridiagSturm(work);}


__aicore__ inline void ComputeInverseVectors(Z1*work,ZA*swap_mask){
Z1*dd=work;Z1*ee=work+((n<=256u)?256u:512u);
bool is_diag=true;Z1 max_elem=0.0f;

if constexpr(SOLVE_MODE==2u){
Z1 saved_max=work[1023];is_diag=(saved_max<0.0f);
max_elem=is_diag?0.0f:saved_max;}else{
bool already_loaded=(group_lanes>=4u&&n>=40u&&ZB<=64u);for(Z0 i=0;i<n;++i){
Z1 d_i=already_loaded?dd[i]:ub_diag.ZF(i);Z1 e_i=already_loaded?ee[i]:ub_off.ZF(i);
if(!already_loaded){
dd[i]=d_i;ee[i]=e_i;
}
if(Z7(d_i)>max_elem)max_elem=Z7(d_i);if(Z7(e_i)>max_elem)max_elem=Z7(e_i);
if(i+1u<n&&e_i!=0.0f){
is_diag=false;}
}
}
Z1 wide_gap_thr=0.036f*max_elem;

if(ZB>0u&&(is_diag||n<Z9)){
Duplicate(ub_vecs,0.0f,ZB*Z9);Z3<PIPE_ALL>();
}

if(is_diag){
if constexpr(SOLVE_MODE!=2u){
for(Z0 i=0;i<n;++i){
dd[i]=ub_diag.ZF(i);}
}
for(Z0 r=0;r<ZB;++r){
Z0 col_idx=ZJ+r;Z4<Z1>vec_r=ub_vecs[r*Z9];
for(Z0 i=0;i<n;++i){
Z0 rank=0u;Z1 di=dd[i];
for(Z0 k=0;k<n;++k){
if(dd[k]<di||(dd[k]==di&&k<i)){
++rank;}
}
if(rank==col_idx){
vec_r.Z8(i,1.0f);break;
}
}
}
Z3<PIPE_ALL>();return;
}

Z1 v_scale[64];for(Z0 ri=0;ri<ZB;++ri)v_scale[ri]=0.0f;
Z0 r=0u;if(n<=256u){
Z1*ee_rat=work+512;Z1*inv_ee=work+768;
Z0 i_ee=0u;for(;i_ee+4u<n;i_ee+=4u){
Z1 e0=ee[i_ee],e1=ee[i_ee+1u],e2=ee[i_ee+2u],e3=ee[i_ee+3u],e4=ee[i_ee+4u];
if(Z7(e0)>=1e-7f&&Z7(e1)>=1e-7f&&Z7(e2)>=1e-7f&&Z7(e3)>=1e-7f){
Z1 p01=e0*e1,p23=e2*e3;Z1 inv_all=1.0f/(p01*p23);
Z1 inv01=inv_all*p23,inv23=inv_all*p01;Z1 iv0=inv01*e1,iv1=inv01*e0,iv2=inv23*e3,iv3=inv23*e2;
inv_ee[i_ee]=iv0;inv_ee[i_ee+1u]=iv1;inv_ee[i_ee+2u]=iv2;inv_ee[i_ee+3u]=iv3;
ee_rat[i_ee]=e1*iv0;ee_rat[i_ee+1u]=e2*iv1;ee_rat[i_ee+2u]=e3*iv2;ee_rat[i_ee+3u]=e4*iv3;}else{
for(Z0 k=0;k<4u;++k){
Z1 ek=ee[i_ee+k];Z1 ivk=(ek!=0.0f)?(1.0f/ek):0.0f;
inv_ee[i_ee+k]=ivk;ee_rat[i_ee+k]=ee[i_ee+k+1u]*ivk;
}
}
}
for(;i_ee+1u<n;++i_ee){
Z1 ei=ee[i_ee];Z1 inv_e=(ei!=0.0f)?(1.0f/ei):0.0f;
inv_ee[i_ee]=inv_e;ee_rat[i_ee]=ee[i_ee+1u]*inv_e;
}

Z1*aos=work+1024;for(;r+2u<ZB;r+=4u){
Z0 bcnt=((ZB-r)<4u)?(ZB-r):4u;Z0 c0=ZJ+r;
Z0 c1=(bcnt>1u)?(c0+1u):c0;Z0 c2=(bcnt>2u)?(c0+2u):c1;
Z0 c3=(bcnt>3u)?(c0+3u):c2;Z1 lam0=ub_eig.ZF(c0);
Z1 lam1=ub_eig.ZF(c1);Z1 lam2=ub_eig.ZF(c2);
Z1 lam3=ub_eig.ZF(c3);


Z0 sm0=(c0*13u)%23u;Z0 sm1=sm0+13u;if(sm1>=23u)sm1-=23u;
Z0 sm2=sm1+13u;if(sm2>=23u)sm2-=23u;Z0 sm3=sm2+13u;if(sm3>=23u)sm3-=23u;
Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 nq1=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm1));
Z1 nq2=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm2));
Z1 nq3=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm3));
Z1 cq0=nq0,cq1=nq1,cq2=nq2,cq3=nq3;

Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];Z1 cur_p1=dd[0]-lam1,cur_u1=ee[0];
Z1 cur_p2=dd[0]-lam2,cur_u2=ee[0];Z1 cur_p3=dd[0]-lam3,cur_u3=ee[0];

for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i];Z1 abs_m=Z7(m_i);
Z1 d_next=dd[i+1u];Z1 e_next=ee[i+1u];
Z1 nd0=d_next-lam0;Z1 nd1=d_next-lam1;
Z1 nd2=d_next-lam2;Z1 nd3=d_next-lam3;

bool sw0=(Z7(cur_p0)<abs_m);bool sw1=(Z7(cur_p1)<abs_m);
bool sw2=(Z7(cur_p2)<abs_m);bool sw3=(Z7(cur_p3)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u)|(sw2?4u:0u)|(sw3?8u:0u));swap_mask[i]=sm;

Z1 inv_m=inv_ee[i];Z1 inv0,inv1,inv2,inv3;
if(sm==15u){
inv0=inv_m;inv1=inv_m;inv2=inv_m;inv3=inv_m;}else if(abs_m>=1e-7f){
Z1 p0=sw0?1.0f:cur_p0;Z1 p1=sw1?1.0f:cur_p1;
Z1 p2=sw2?1.0f:cur_p2;Z1 p3=sw3?1.0f:cur_p3;
Z1 p01=p0*p1;Z1 p23=p2*p3;
Z1 inv_all=1.0f/(p01*p23);Z1 inv01=inv_all*p23;
Z1 inv23=inv_all*p01;inv0=sw0?inv_m:(inv01*p1);
inv1=sw1?inv_m:(inv01*p0);inv2=sw2?inv_m:(inv23*p3);
inv3=sw3?inv_m:(inv23*p2);}else{
inv0=sw0?inv_m:(1.0f/ClampPivot(cur_p0));inv1=sw1?inv_m:(1.0f/ClampPivot(cur_p1));
inv2=sw2?inv_m:(1.0f/ClampPivot(cur_p2));inv3=sw3?inv_m:(1.0f/ClampPivot(cur_p3));
}

Z1 f0=(sw0?cur_p0:m_i)*inv0;Z1 f1=(sw1?cur_p1:m_i)*inv1;
Z1 f2=(sw2?cur_p2:m_i)*inv2;Z1 f3=(sw3?cur_p3:m_i)*inv3;

constexpr Z0 kA=(SOLVE_MODE==2u)?16u:12u;constexpr Z0 kU=(SOLVE_MODE==2u)?8u:4u;
constexpr Z0 kY=(SOLVE_MODE==2u)?12u:8u;Z1*s=aos+i*kA;
if constexpr(SOLVE_MODE==2u){
s[0]=inv0;s[1]=inv1;s[2]=inv2;s[3]=inv3;s[4]=f0;s[5]=f1;s[6]=f2;s[7]=f3;
}else{
s[0]=sw0?f0:inv0;s[1]=sw1?f1:inv1;
s[2]=sw2?f2:inv2;s[3]=sw3?f3:inv3;
}
Z1 u1_0=sw0?nd0:cur_u0;Z1 u1_1=sw1?nd1:cur_u1;
Z1 u1_2=sw2?nd2:cur_u2;Z1 u1_3=sw3?nd3:cur_u3;
s[kU+0u]=u1_0*inv0;s[kU+1u]=u1_1*inv1;
s[kU+2u]=u1_2*inv2;s[kU+3u]=u1_3*inv3;

cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_p2=(sw2?cur_u2:nd2)-f2*u1_2;cur_p3=(sw3?cur_u3:nd3)-f3*u1_3;

cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
cur_u2=sw2?(-f2*e_next):e_next;cur_u3=sw3?(-f3*e_next):e_next;

nq0+=(nq0>=1.1875f)?-0.1875f:0.53125f;nq1+=(nq1>=1.1875f)?-0.1875f:0.53125f;
nq2+=(nq2>=1.1875f)?-0.1875f:0.53125f;nq3+=(nq3>=1.1875f)?-0.1875f:0.53125f;

Z1 qi0=sw0?nq0:cq0;Z1 qi1=sw1?nq1:cq1;
Z1 qi2=sw2?nq2:cq2;Z1 qi3=sw3?nq3:cq3;
Z1 b0=sw0?cq0:nq0;Z1 b1=sw1?cq1:nq1;
Z1 b2=sw2?cq2:nq2;Z1 b3=sw3?cq3:nq3;
s[kY+0u]=qi0*inv0;s[kY+1u]=qi1*inv1;s[kY+2u]=qi2*inv2;s[kY+3u]=qi3*inv3;cq0=b0-f0*qi0;
cq1=b1-f1*qi1;cq2=b2-f2*qi2;
cq3=b3-f3*qi3;}
constexpr Z0 kA=(SOLVE_MODE==2u)?16u:12u;constexpr Z0 kU=(SOLVE_MODE==2u)?8u:4u;
constexpr Z0 kY=(SOLVE_MODE==2u)?12u:8u;Z1*s_last=aos+(n-1u)*kA;
s_last[0]=1.0f/ClampPivot(cur_p0);s_last[1]=1.0f/ClampPivot(cur_p1);
s_last[2]=1.0f/ClampPivot(cur_p2);s_last[3]=1.0f/ClampPivot(cur_p3);


Z1 gl0=(c0>0u)?(lam0-ub_eig.ZF(c0-1u)):max_elem;
Z1 gr3=(c0+bcnt<n)?(ub_eig.ZF(c0+bcnt)-lam3):max_elem;Z1 g01=(bcnt>1u)?(lam1-lam0):max_elem;
Z1 g12=(bcnt>2u)?(lam2-lam1):max_elem;Z1 g23=(bcnt>3u)?(lam3-lam2):gr3;
Z1 g0=(gl0<g01)?gl0:g01;Z1 g1=(g01<g12)?g01:g12;
Z1 g2=(g12<g23)?g12:g23;Z1 g3=(bcnt>3u)?((g23<gr3)?g23:gr3):max_elem;
if(g0<0.0f)g0=0.0f;if(g1<0.0f)g1=0.0f;
if(g2<0.0f)g2=0.0f;if(g3<0.0f)g3=0.0f;
Z1 g0_sq=g0*g0,g1_sq=g1*g1,g2_sq=g2*g2,g3_sq=g3*g3;
Z1 min_g=((g0<g1)?g0:g1)<((g2<g3)?g2:g3)?((g0<g1)?g0:g1):((g2<g3)?g2:g3);
bool can_early_stop=(min_g>=0.00025f*max_elem);Z1 s0_0=0.0f,s0_1=0.0f,s0_2=0.0f,s0_3=0.0f;
Z0 max_iter=3u;

Z1 in0=1.0f,in1=1.0f,in2=1.0f,in3=1.0f;for(Z0 iter=0;iter<max_iter;++iter){
if(iter>0u){
cq0=aos[kY+0u]*in0;cq1=aos[kY+1u]*in1;cq2=aos[kY+2u]*in2;cq3=aos[kY+3u]*in3;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*kA;
Z1 nq0=s[kA+kY+0u]*in0,nq1=s[kA+kY+1u]*in1,nq2=s[kA+kY+2u]*in2,nq3=s[kA+kY+3u]*in3;
Z1 qi0=(sm&1u)?nq0:cq0;Z1 qi1=(sm&2u)?nq1:cq1;
Z1 qi2=(sm&4u)?nq2:cq2;Z1 qi3=(sm&8u)?nq3:cq3;
Z1 b0=(sm&1u)?cq0:nq0;Z1 b1=(sm&2u)?cq1:nq1;
Z1 b2=(sm&4u)?cq2:nq2;Z1 b3=(sm&8u)?cq3:nq3;
if constexpr(SOLVE_MODE==2u){
s[12]=qi0*s[0];s[13]=qi1*s[1];
s[14]=qi2*s[2];s[15]=qi3*s[3];
cq0=b0-s[4]*qi0;cq1=b1-s[5]*qi1;
cq2=b2-s[6]*qi2;cq3=b3-s[7]*qi3;
}else{
Z1 m_i=ee[i],inv_m=inv_ee[i];Z1 sv0=s[0],sv1=s[1],sv2=s[2],sv3=s[3];
s[8]=qi0*((sm&1u)?inv_m:sv0);s[9]=qi1*((sm&2u)?inv_m:sv1);
s[10]=qi2*((sm&4u)?inv_m:sv2);s[11]=qi3*((sm&8u)?inv_m:sv3);
cq0=b0-((sm&1u)?sv0:(m_i*sv0))*qi0;cq1=b1-((sm&2u)?sv1:(m_i*sv1))*qi1;
cq2=b2-((sm&4u)?sv2:(m_i*sv2))*qi2;cq3=b3-((sm&8u)?sv3:(m_i*sv3))*qi3;
}
}
}
Z1 q0_n1=cq0*s_last[0];Z1 q1_n1=cq1*s_last[1];
Z1 q2_n1=cq2*s_last[2];Z1 q3_n1=cq3*s_last[3];
Z1 nsq0=q0_n1*q0_n1;Z1 nsq1=q1_n1*q1_n1;
Z1 nsq2=q2_n1*q2_n1;Z1 nsq3=q3_n1*q3_n1;

bool no_gs=(min_g>=0.0001f*max_elem)&&(gr3>=0.0001f*max_elem);if(iter+1u>=max_iter&&no_gs){
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+1u)*Z9];
Z4<Z1>vr2=ub_vecs[(r+2u)*Z9];Z4<Z1>vr3=ub_vecs[(r+((bcnt>3u)?3u:2u))*Z9];
vr0.Z8(n-1u,q0_n1);vr1.Z8(n-1u,q1_n1);
vr3.Z8(n-1u,q3_n1);vr2.Z8(n-1u,q2_n1);
Z1 d_last=dd[n-1u];Z1 rq0=d_last*nsq0,rq1=d_last*nsq1,rq2=d_last*nsq2,rq3=d_last*nsq3;
Z1 q0_n2=0.0f,q1_n2=0.0f,q2_n2=0.0f,q3_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*kA;q0_n2=q0_n1;q1_n2=q1_n1;q2_n2=q2_n1;q3_n2=q3_n1;
q0_n1=s_nm2[kY+0u]-s_nm2[kU+0u]*q0_n2;q1_n1=s_nm2[kY+1u]-s_nm2[kU+1u]*q1_n2;
q2_n1=s_nm2[kY+2u]-s_nm2[kU+2u]*q2_n2;q3_n1=s_nm2[kY+3u]-s_nm2[kU+3u]*q3_n2;
vr0.Z8(n-2u,q0_n1);vr1.Z8(n-2u,q1_n1);
vr3.Z8(n-2u,q3_n1);vr2.Z8(n-2u,q2_n1);
nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;
nsq2+=q2_n1*q2_n1;nsq3+=q3_n1*q3_n1;
Z1 d_nm2=dd[n-2u],two_e_nm2=2.0f*ee[n-2u];rq0+=q0_n1*(d_nm2*q0_n1+two_e_nm2*q0_n2);
rq1+=q1_n1*(d_nm2*q1_n1+two_e_nm2*q1_n2);rq2+=q2_n1*(d_nm2*q2_n1+two_e_nm2*q2_n2);
rq3+=q3_n1*(d_nm2*q3_n1+two_e_nm2*q3_n2);}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*kA;
Z1 c0_v=(s[kY+0u]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[kU+0u]*q0_n1;
Z1 c1_v=(s[kY+1u]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[kU+1u]*q1_n1;
Z1 c2_v=(s[kY+2u]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[kU+2u]*q2_n1;
Z1 c3_v=(s[kY+3u]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[kU+3u]*q3_n1;vr0.Z8(ui,c0_v);
vr1.Z8(ui,c1_v);vr3.Z8(ui,c3_v);
vr2.Z8(ui,c2_v);Z1 d_i=dd[ui],two_e=2.0f*ee[ui];
rq0+=c0_v*(d_i*c0_v+two_e*q0_n1);rq1+=c1_v*(d_i*c1_v+two_e*q1_n1);
rq2+=c2_v*(d_i*c2_v+two_e*q2_n1);rq3+=c3_v*(d_i*c3_v+two_e*q3_n1);
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;
nsq2+=c2_v*c2_v;nsq3+=c3_v*c3_v;
}
ub_eig.Z8(c0,rq0/nsq0);ub_eig.Z8(c1,rq1/nsq1);
ub_eig.Z8(c2,rq2/nsq2);if(bcnt>3u)ub_eig.Z8(c3,rq3/nsq3);
v_scale[r+0u]=ZI(nsq0);v_scale[r+1u]=ZI(nsq1);
v_scale[r+2u]=ZI(nsq2);if(bcnt>3u)v_scale[r+3u]=ZI(nsq3);
break;}

s_last[kY+0u]=q0_n1;s_last[kY+1u]=q1_n1;
s_last[kY+2u]=q2_n1;s_last[kY+3u]=q3_n1;

Z1 q0_n2=0.0f,q1_n2=0.0f,q2_n2=0.0f,q3_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*kA;q0_n2=q0_n1;q1_n2=q1_n1;q2_n2=q2_n1;q3_n2=q3_n1;
q0_n1=s_nm2[kY+0u]-s_nm2[kU+0u]*q0_n2;q1_n1=s_nm2[kY+1u]-s_nm2[kU+1u]*q1_n2;
q2_n1=s_nm2[kY+2u]-s_nm2[kU+2u]*q2_n2;q3_n1=s_nm2[kY+3u]-s_nm2[kU+3u]*q3_n2;
s_nm2[kY+0u]=q0_n1;s_nm2[kY+1u]=q1_n1;s_nm2[kY+2u]=q2_n1;s_nm2[kY+3u]=q3_n1;nsq0+=q0_n1*q0_n1;
nsq1+=q1_n1*q1_n1;nsq2+=q2_n1*q2_n1;
nsq3+=q3_n1*q3_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*kA;
Z1 c0_v=(s[kY+0u]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[kU+0u]*q0_n1;
Z1 c1_v=(s[kY+1u]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[kU+1u]*q1_n1;
Z1 c2_v=(s[kY+2u]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[kU+2u]*q2_n1;
Z1 c3_v=(s[kY+3u]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[kU+3u]*q3_n1;

s[kY+0u]=c0_v;s[kY+1u]=c1_v;s[kY+2u]=c2_v;s[kY+3u]=c3_v;

q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;

nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;
nsq2+=c2_v*c2_v;nsq3+=c3_v*c3_v;
}
if(iter==0u){
s0_0=nsq0*g0_sq;s0_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
s0_2=(bcnt>2u)?(nsq2*g2_sq):1.0e30f;s0_3=(bcnt>3u)?(nsq3*g3_sq):1.0e30f;
Z1 ms0_01=(s0_0<s0_1)?s0_0:s0_1;Z1 ms0_23=(s0_2<s0_3)?s0_2:s0_3;
if(no_gs&&((ms0_01<ms0_23)?ms0_01:ms0_23)>=250.0f){
max_iter=2u;}
}
bool is_final_iter=(iter+1u>=max_iter);if(iter==1u&&can_early_stop){
Z1 s1_0=nsq0*g0_sq;Z1 s1_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
Z1 s1_2=(bcnt>2u)?(nsq2*g2_sq):1.0e30f;Z1 s1_3=(bcnt>3u)?(nsq3*g3_sq):1.0e30f;
Z1 ms01=(s1_0<s1_1)?s1_0:s1_1;Z1 ms23=(s1_2<s1_3)?s1_2:s1_3;
Z1 min_s1=(ms01<ms23)?ms01:ms23;Z1 p0=s0_0*s1_0,p1=s0_1*s1_1,p2=s0_2*s1_2,p3=s0_3*s1_3;
Z1 mp01=(p0<p1)?p0:p1;Z1 mp23=(p2<p3)?p2:p3;
Z1 min_p=(mp01<mp23)?mp01:mp23;if((min_s1>=4.0e4f&&min_p>=6.0e8f)||(min_s1>=2.5e6f&&min_p>=5.0e7f)){
is_final_iter=true;}
}
if(!is_final_iter){
in0=ScalarRsqrtFast(nsq0);in1=ScalarRsqrtFast(nsq1);
in2=ScalarRsqrtFast(nsq2);in3=ScalarRsqrtFast(nsq3);
}else{
in0=ZI(nsq0);in1=ZI(nsq1);
in2=ZI(nsq2);in3=ZI(nsq3);
if(r>0u&&Z7(lam0-ub_eig.ZF(c0-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(aos[i*kA+kY+0u]*in0)*prev_v.ZF(i);nsq0=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=aos[i*kA+kY+0u]*in0-dot*prev_v.ZF(i);aos[i*kA+kY+0u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
if(bcnt>1u&&Z7(lam1-lam0)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*kA+kY+1u]*in1)*(aos[i*kA+kY+0u]*in0);
nsq1=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*kA+kY+1u]*in1-dot*(aos[i*kA+kY+0u]*in0);aos[i*kA+kY+1u]=v;
nsq1+=v*v;}
in1=ZI(nsq1);}
if(bcnt>2u&&Z7(lam2-lam1)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*kA+kY+2u]*in2)*(aos[i*kA+kY+1u]*in1);
nsq2=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*kA+kY+2u]*in2-dot*(aos[i*kA+kY+1u]*in1);aos[i*kA+kY+2u]=v;
nsq2+=v*v;}
in2=ZI(nsq2);}
if(bcnt>3u&&Z7(lam3-lam2)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*kA+kY+3u]*in3)*(aos[i*kA+kY+2u]*in2);
nsq3=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*kA+kY+3u]*in3-dot*(aos[i*kA+kY+2u]*in2);aos[i*kA+kY+3u]=v;
nsq3+=v*v;}
in3=ZI(nsq3);}
Z1 rq0=0.0f,rq1=0.0f,rq2=0.0f,rq3=0.0f;Z1 cur0=aos[kY+0u]*in0;
Z1 cur1=aos[kY+1u]*in1;Z1 cur2=aos[kY+2u]*in2;
Z1 cur3=aos[kY+3u]*in3;Z4<Z1>vr0=ub_vecs[r*Z9];
Z4<Z1>vr1=ub_vecs[(r+1u)*Z9];Z4<Z1>vr2=ub_vecs[(r+2u)*Z9];
Z4<Z1>vr3=ub_vecs[(r+((bcnt>3u)?3u:2u))*Z9];vr0.Z8(0,cur0);
vr1.Z8(0,cur1);vr3.Z8(0,cur3);
vr2.Z8(0,cur2);for(Z0 i=0;i+1u<n;++i){
const Z1*sn=aos+(i+1u)*kA;Z1 nxt0=sn[kY+0u]*in0;
Z1 nxt1=sn[kY+1u]*in1;Z1 nxt2=sn[kY+2u]*in2;
Z1 nxt3=sn[kY+3u]*in3;vr0.Z8(i+1u,nxt0);
vr1.Z8(i+1u,nxt1);vr3.Z8(i+1u,nxt3);
vr2.Z8(i+1u,nxt2);Z1 d_i=dd[i];
Z1 two_e=2.0f*ee[i];rq0+=cur0*(d_i*cur0+two_e*nxt0);
rq1+=cur1*(d_i*cur1+two_e*nxt1);rq2+=cur2*(d_i*cur2+two_e*nxt2);
rq3+=cur3*(d_i*cur3+two_e*nxt3);cur0=nxt0;cur1=nxt1;cur2=nxt2;cur3=nxt3;
}
Z1 d_last=dd[n-1u];rq0+=d_last*cur0*cur0;
rq1+=d_last*cur1*cur1;rq2+=d_last*cur2*cur2;
rq3+=d_last*cur3*cur3;ub_eig.Z8(c0,rq0);
ub_eig.Z8(c1,rq1);ub_eig.Z8(c2,rq2);
if(bcnt>3u)ub_eig.Z8(c3,rq3);break;
}
}
}

for(;r<ZB;r+=2u){
Z0 bcnt=((ZB-r)<2u)?(ZB-r):2u;Z0 c0=ZJ+r;
if(bcnt==1u){
Z1 lam0=ub_eig.ZF(c0);Z0 sm0=(c0*13u)%23u;
Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));Z1 cq0=nq0;
Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i],abs_m=Z7(m_i),nd0=dd[i+1u]-lam0,e_next=ee[i+1u];bool sw0=(Z7(cur_p0)<abs_m);
swap_mask[i]=sw0?1u:0u;Z1 inv0=sw0?inv_ee[i]:(1.0f/ClampPivot(cur_p0));
Z1 f0=(sw0?cur_p0:m_i)*inv0;Z1 u1_0=sw0?nd0:cur_u0;
Z1*s=aos+i*4u;s[0]=inv0;
s[1]=f0;s[2]=u1_0*inv0;
cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_u0=sw0?(-f0*e_next):e_next;
nq0+=(nq0>=1.1875f)?-0.1875f:0.53125f;Z1 qi0=sw0?nq0:cq0;
Z1 b0=sw0?cq0:nq0;s[3]=qi0*inv0;
cq0=b0-f0*qi0;}
Z1 inv_last=1.0f/ClampPivot(cur_p0);Z1 gl0=(c0>0u)?(lam0-ub_eig.ZF(c0-1u)):max_elem;
Z1 gr0=(c0+1u<n)?(ub_eig.ZF(c0+1u)-lam0):max_elem;Z1 g0=(gl0<gr0)?gl0:gr0;
if(g0<0.0f)g0=0.0f;Z1 g0_sq=g0*g0;
bool can_early_stop=(g0>=0.00025f*max_elem);bool no_gs1=(g0>=0.0001f*max_elem)&&(gr0>=0.0001f*max_elem);
Z1 s0_0=0.0f;Z0 max_iter=3u;
Z1 in0=1.0f;Z4<Z1>vr0=ub_vecs[r*Z9];
for(Z0 iter=0;iter<max_iter;++iter){
if(iter>0u){
cq0=aos[3]*in0;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*4u;
Z1 nq0=s[7]*in0;Z1 qi0=(sm!=0u)?nq0:cq0;
Z1 b0=(sm!=0u)?cq0:nq0;s[3]=qi0*s[0];
cq0=b0-s[1]*qi0;}
}
Z1 q0_n1=cq0*inv_last;Z1 nsq0=q0_n1*q0_n1;
if(iter+1u>=max_iter&&no_gs1){
vr0.Z8(n-1u,q0_n1);Z1 d_last=dd[n-1u];
Z1 rq0=d_last*nsq0;Z1 q0_n2=0.0f;
if(n>1u){
Z1*s_nm2=aos+(n-2u)*4u;q0_n2=q0_n1;
q0_n1=s_nm2[3]-s_nm2[2]*q0_n2;vr0.Z8(n-2u,q0_n1);
nsq0+=q0_n1*q0_n1;rq0+=q0_n1*(dd[n-2u]*q0_n1+(2.0f*ee[n-2u])*q0_n2);
}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1*s=aos+ui*4u;Z1 c0_v=(s[3]-((sm!=0u)?ee_rat[ui]:0.0f)*q0_n2)-s[2]*q0_n1;
vr0.Z8(ui,c0_v);rq0+=c0_v*(dd[ui]*c0_v+(2.0f*ee[ui])*q0_n1);
q0_n2=q0_n1;q0_n1=c0_v;nsq0+=c0_v*c0_v;
}
ub_eig.Z8(c0,rq0/nsq0);v_scale[r]=ZI(nsq0);
break;}
aos[(n-1u)*4u+3u]=q0_n1;Z1 q0_n2=0.0f;
if(n>1u){
Z1*s_nm2=aos+(n-2u)*4u;q0_n2=q0_n1;
q0_n1=s_nm2[3]-s_nm2[2]*q0_n2;s_nm2[3]=q0_n1;
nsq0+=q0_n1*q0_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1*s=aos+ui*4u;Z1 c0_v=(s[3]-((sm!=0u)?ee_rat[ui]:0.0f)*q0_n2)-s[2]*q0_n1;
s[3]=c0_v;q0_n2=q0_n1;q0_n1=c0_v;
nsq0+=c0_v*c0_v;}
if(iter==0u){
s0_0=nsq0*g0_sq;if(no_gs1&&s0_0>=250.0f){
max_iter=2u;}
}
bool is_final_iter=(iter+1u>=max_iter);if(iter==1u&&can_early_stop){
Z1 s1_0=nsq0*g0_sq;Z1 p0=s0_0*s1_0;
if((s1_0>=4.0e4f&&p0>=6.0e8f)||(s1_0>=2.5e6f&&p0>=5.0e7f)){
is_final_iter=true;}
}
if(!is_final_iter){
in0=ScalarRsqrtFast(nsq0);}else{
in0=ZI(nsq0);if(r>0u&&Z7(lam0-ub_eig.ZF(c0-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(aos[i*4u+3u]*in0)*prev_v.ZF(i);nsq0=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=aos[i*4u+3u]*in0-dot*prev_v.ZF(i);aos[i*4u+3u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
Z1 rq0=0.0f;Z1 cur0=aos[3]*in0;
vr0.Z8(0,cur0);for(Z0 i=0;i+1u<n;++i){
Z1 nxt0=aos[(i+1u)*4u+3u]*in0;vr0.Z8(i+1u,nxt0);
rq0+=dd[i]*cur0*cur0+(2.0f*ee[i])*cur0*nxt0;cur0=nxt0;
}
rq0+=dd[n-1u]*cur0*cur0;ub_eig.Z8(c0,rq0);
break;}
}
continue;}
Z0 c1=(bcnt>1u)?(c0+1u):c0;Z1 lam0=ub_eig.ZF(c0);
Z1 lam1=ub_eig.ZF(c1);Z0 sm0=(c0*13u)%23u;
Z0 sm1=sm0+13u;if(sm1>=23u)sm1-=23u;
Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 nq1=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm1));Z1 cq0=nq0,cq1=nq1;
Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];Z1 cur_p1=dd[0]-lam1,cur_u1=ee[0];
for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i],abs_m=Z7(m_i),d_next=dd[i+1u],e_next=ee[i+1u];Z1 nd0=d_next-lam0,nd1=d_next-lam1;
bool sw0=(Z7(cur_p0)<abs_m),sw1=(Z7(cur_p1)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u));swap_mask[i]=sm;
Z1 inv_m=inv_ee[i];Z1 inv0,inv1;
if(sm==3u){
inv0=inv_m;inv1=inv_m;}else if(abs_m>=1e-7f){
Z1 p0=sw0?1.0f:cur_p0;Z1 p1=sw1?1.0f:cur_p1;
Z1 inv01=1.0f/(p0*p1);inv0=sw0?inv_m:(inv01*p1);
inv1=sw1?inv_m:(inv01*p0);}else{
inv0=sw0?inv_m:(1.0f/ClampPivot(cur_p0));inv1=sw1?inv_m:(1.0f/ClampPivot(cur_p1));
}
Z1 f0=(sw0?cur_p0:m_i)*inv0,f1=(sw1?cur_p1:m_i)*inv1;Z1*s=aos+i*8u;
s[0]=inv0;s[1]=inv1;s[2]=f0;s[3]=f1;
Z1 u1_0=sw0?nd0:cur_u0,u1_1=sw1?nd1:cur_u1;s[4]=u1_0*inv0;s[5]=u1_1*inv1;
cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
nq0+=(nq0>=1.1875f)?-0.1875f:0.53125f;nq1+=(nq1>=1.1875f)?-0.1875f:0.53125f;
Z1 qi0=sw0?nq0:cq0,qi1=sw1?nq1:cq1;Z1 b0=sw0?cq0:nq0,b1=sw1?cq1:nq1;
s[6]=qi0*inv0;s[7]=qi1*inv1;cq0=b0-f0*qi0;cq1=b1-f1*qi1;
}
Z1*s_last=aos+(n-1u)*8u;s_last[0]=1.0f/ClampPivot(cur_p0);
s_last[1]=1.0f/ClampPivot(cur_p1);Z1 gl0=(c0>0u)?(lam0-ub_eig.ZF(c0-1u)):max_elem;
Z1 g01=(bcnt>1u)?(lam1-lam0):max_elem;Z1 gr1=(c0+bcnt<n)?(ub_eig.ZF(c0+bcnt)-lam1):max_elem;
Z1 g0=(gl0<g01)?gl0:g01;Z1 g1=(bcnt>1u)?((g01<gr1)?g01:gr1):g0;
if(g0<0.0f)g0=0.0f;if(g1<0.0f)g1=0.0f;
Z1 g0_sq=g0*g0,g1_sq=g1*g1;bool can_early_stop=(((g0<g1)?g0:g1)>=0.00025f*max_elem);
Z1 s0_0=0.0f,s0_1=0.0f;Z0 max_iter=3u;
Z1 in0=1.0f,in1=1.0f;for(Z0 iter=0;iter<max_iter;++iter){
if(iter>0u){
cq0=aos[6]*in0;cq1=aos[7]*in1;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*8u;
Z1 nq0=s[14]*in0,nq1=s[15]*in1;Z1 qi0=(sm&1u)?nq0:cq0,qi1=(sm&2u)?nq1:cq1;
Z1 b0=(sm&1u)?cq0:nq0,b1=(sm&2u)?cq1:nq1;s[6]=qi0*s[0];s[7]=qi1*s[1];
cq0=b0-s[2]*qi0;cq1=b1-s[3]*qi1;}
}
Z1 q0_n1=cq0*s_last[0],q1_n1=cq1*s_last[1];Z1 nsq0=q0_n1*q0_n1,nsq1=q1_n1*q1_n1;
bool no_gs2=(((g0<g1)?g0:g1)>=0.0001f*max_elem)&&(gr1>=0.0001f*max_elem);if(iter+1u>=max_iter&&no_gs2){
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+((bcnt>1u)?1u:0u))*Z9];
vr1.Z8(n-1u,q1_n1);vr0.Z8(n-1u,q0_n1);
Z1 d_last=dd[n-1u];Z1 rq0=d_last*nsq0,rq1=d_last*nsq1;
Z1 q0_n2=0.0f,q1_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*8u;q0_n2=q0_n1;q1_n2=q1_n1;
q0_n1=s_nm2[6]-s_nm2[4]*q0_n2;q1_n1=s_nm2[7]-s_nm2[5]*q1_n2;
vr1.Z8(n-2u,q1_n1);vr0.Z8(n-2u,q0_n1);
nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;Z1 d_nm2=dd[n-2u],two_e_nm2=2.0f*ee[n-2u];
rq0+=q0_n1*(d_nm2*q0_n1+two_e_nm2*q0_n2);rq1+=q1_n1*(d_nm2*q1_n1+two_e_nm2*q1_n2);
}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*8u;
Z1 c0_v=(s[6]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[4]*q0_n1;
Z1 c1_v=(s[7]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[5]*q1_n1;vr1.Z8(ui,c1_v);
vr0.Z8(ui,c0_v);Z1 d_i=dd[ui],two_e=2.0f*ee[ui];
rq0+=c0_v*(d_i*c0_v+two_e*q0_n1);rq1+=c1_v*(d_i*c1_v+two_e*q1_n1);
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;}
ub_eig.Z8(c0,rq0/nsq0);if(bcnt>1u)ub_eig.Z8(c1,rq1/nsq1);
v_scale[r+0u]=ZI(nsq0);if(bcnt>1u)v_scale[r+1u]=ZI(nsq1);
break;}
s_last[6]=q0_n1;s_last[7]=q1_n1;Z1 q0_n2=0.0f,q1_n2=0.0f;
if(n>1u){
Z1*s_nm2=aos+(n-2u)*8u;q0_n2=q0_n1;q1_n2=q1_n1;
q0_n1=s_nm2[6]-s_nm2[4]*q0_n2;q1_n1=s_nm2[7]-s_nm2[5]*q1_n2;
s_nm2[6]=q0_n1;s_nm2[7]=q1_n1;nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;
}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*8u;
Z1 c0_v=(s[6]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[4]*q0_n1;
Z1 c1_v=(s[7]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[5]*q1_n1;s[6]=c0_v;s[7]=c1_v;
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;}
if(iter==0u){
s0_0=nsq0*g0_sq;s0_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
if(no_gs2&&((s0_0<s0_1)?s0_0:s0_1)>=250.0f){
max_iter=2u;}
}
bool is_final_iter=(iter+1u>=max_iter);if(iter==1u&&can_early_stop){
Z1 s1_0=nsq0*g0_sq;Z1 s1_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
Z1 min_s1=(s1_0<s1_1)?s1_0:s1_1;Z1 p0=s0_0*s1_0;
Z1 p1=(bcnt>1u)?(s0_1*s1_1):1.0e30f;Z1 min_p=(p0<p1)?p0:p1;
if((min_s1>=4.0e4f&&min_p>=6.0e8f)||(min_s1>=2.5e6f&&min_p>=5.0e7f)){
is_final_iter=true;}
}
if(!is_final_iter){
in0=ScalarRsqrtFast(nsq0);in1=ScalarRsqrtFast(nsq1);
}else{
in0=ZI(nsq0);in1=ZI(nsq1);
if(r>0u&&Z7(lam0-ub_eig.ZF(c0-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+6u]*in0)*prev_v.ZF(i);nsq0=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+6u]*in0-dot*prev_v.ZF(i);aos[i*8u+6u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
if(bcnt>1u&&Z7(lam1-lam0)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+7u]*in1)*(aos[i*8u+6u]*in0);
nsq1=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+7u]*in1-dot*(aos[i*8u+6u]*in0);aos[i*8u+7u]=v;
nsq1+=v*v;}
in1=ZI(nsq1);}
Z1 rq0=0.0f,rq1=0.0f;Z1 cur0=aos[6]*in0,cur1=aos[7]*in1;
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+((bcnt>1u)?1u:0u))*Z9];
vr1.Z8(0,cur1);vr0.Z8(0,cur0);
for(Z0 i=0;i+1u<n;++i){
const Z1*sn=aos+(i+1u)*8u;Z1 nxt0=sn[6]*in0,nxt1=sn[7]*in1;
vr1.Z8(i+1u,nxt1);vr0.Z8(i+1u,nxt0);
Z1 d_i=dd[i],two_e=2.0f*ee[i];rq0+=d_i*cur0*cur0+two_e*cur0*nxt0;
rq1+=d_i*cur1*cur1+two_e*cur1*nxt1;cur0=nxt0;cur1=nxt1;
}
Z1 d_last=dd[n-1u];rq0+=d_last*cur0*cur0;
rq1+=d_last*cur1*cur1;ub_eig.Z8(c0,rq0);
if(bcnt>1u)ub_eig.Z8(c1,rq1);break;
}
}
}
}


Z1*pv=work+1024;Z1*mult=work+1536;
Z1*u1=work+2048;Z1*qv=work+2560;
for(;r<ZB;++r){
Z0 col_idx=ZJ+r;Z4<Z1>vec_r=ub_vecs[r*Z9];
Z1 lambda=ub_eig.ZF(col_idx);

for(Z0 i=0;i<n;++i){
qv[i]=1.0f+0.03125f*Z5<Z1>(Z5<ZG>((i*17u+col_idx*13u)%23u));}
FactorizeShiftedTridiag1Way(dd,ee,lambda,pv,mult,u1,swap_mask);
bool left_ok=(col_idx==0u)||((lambda-ub_eig.ZF(col_idx-1u))>=wide_gap_thr);
bool right_ok=(col_idx+1u>=n)||((ub_eig.ZF(col_idx+1u)-lambda)>=wide_gap_thr);
Z0 max_iter=(left_ok&&right_ok)?2u:3u;Z1 in_scale=1.0f;
for(Z0 iter=0;iter<max_iter;++iter){
Z1 norm_sq=SolveShiftedTridiag1Way(qv,in_scale,ee,pv,mult,u1,swap_mask);if(iter+1u<max_iter){
in_scale=ScalarRsqrtFast(norm_sq);}else{
Z1 inv_norm=ZI(norm_sq);
if(r>0u&&Z7(lambda-ub_eig.ZF(col_idx-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(qv[i]*inv_norm)*prev_v.ZF(i);norm_sq=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=qv[i]*inv_norm-dot*prev_v.ZF(i);qv[i]=v;
norm_sq+=v*v;}
inv_norm=ZI(norm_sq);}
Z1 rq=0.0f;Z1 cur=qv[0]*inv_norm;
for(Z0 i=0;i+1u<n;++i){
Z1 nxt=qv[i+1u]*inv_norm;rq+=dd[i]*cur*cur+2.0f*ee[i]*cur*nxt;
vec_r.Z8(i,cur);cur=nxt;
}
rq+=dd[n-1u]*cur*cur;vec_r.Z8(n-1u,cur);
ub_eig.Z8(col_idx,rq);}
}
}
fence<HardEvent::S_V>();for(Z0 ri=0;ri<ZB;++ri){
if(v_scale[ri]!=0.0f){
Muls(ub_vecs[ri*Z9],ub_vecs[ri*Z9],v_scale[ri],n);}
}
}


__aicore__ inline void BacktransformCoop(Z1*work){
if(ZB==0u||n<3u){
return;}
for(Z0 k=0;k+2u<n;++k){
work[k]=ub_tau.ZF(k);}
bool can_pair=(2u*ZB+16u<=rows_per_lane)&&(ZB<=64u);
Z0 scratch_rows=can_pair?(2u*ZB):ZB;
ZG avail_tile_rows=(rows_per_lane>scratch_rows)?Z5<ZG>(rows_per_lane-scratch_rows):0;
ZG tile_step=(avail_tile_rows>=64)?64:((avail_tile_rows>=48)?48:((avail_tile_rows>=28)?28:((avail_tile_rows>=16)?16:12)));
Z4<Z1>ub_refl_tile=(avail_tile_rows>=16)?ZH[scratch_rows*Z9]:ub_gather[4u*Z9];
Z4<Z1>scratch0=ZH;Z4<Z1>scratch1=ZH[ZB*Z9];

for(ZG k_end=Z5<ZG>(n)-3;k_end>=0;k_end-=tile_step){
ZG k_start=(k_end>=tile_step-1)?(k_end-(tile_step-1)):0;
Z0 blk_cnt=Z5<Z0>(k_end-k_start+1);Z3<PIPE_ALL>();
DataCopy(
ub_refl_tile,
gm_ws[ws_refl_offset+Z5<uint64_t>(k_start)*Z9],
blk_cnt*Z9);Z3<PIPE_ALL>();

ZG kk=k_end;for(;can_pair&&kk-1>=k_start;kk-=2){
Z0 k=Z5<Z0>(kk);Z1 tau_k=work[k];
Z1 tau_km1=work[k-1u];if(tau_k!=0.0f&&tau_km1!=0.0f){
BacktransformPairRowsSub(
ub_vecs,
ub_refl_tile[(k-Z5<Z0>(k_start))*Z9],
ub_refl_tile[(k-1u-Z5<Z0>(k_start))*Z9],
tau_k,
tau_km1,
scratch0,
scratch1,
ZB,
k);}else{
if(tau_k!=0.0f){
Z0 Z2=(k+1u)&~7u;Z4<Z1>v_k=ub_refl_tile[(k-Z5<Z0>(k_start))*Z9];
BacktransformStepRowsSub(ub_vecs,v_k,-tau_k,scratch0,ZB,Z2);}
if(tau_km1!=0.0f){
Z0 km1=k-1u;Z0 Z2=(km1+1u)&~7u;
Z4<Z1>v_km1=ub_refl_tile[(km1-Z5<Z0>(k_start))*Z9];
BacktransformStepRowsSub(ub_vecs,v_km1,-tau_k*0.0f-tau_km1,scratch0,ZB,Z2);}
}
}
for(;kk>=k_start;--kk){
Z0 k=Z5<Z0>(kk);Z1 tau_k=work[k];
if(tau_k==0.0f){
continue;}
Z0 Z2=(k+1u)&~7u;Z4<Z1>v_k=ub_refl_tile[(k-Z5<Z0>(k_start))*Z9];
BacktransformStepRowsSub(ub_vecs,v_k,-tau_k,scratch0,ZB,Z2);}
}
Z3<PIPE_ALL>();}


__aicore__ inline void WriteOutputs(uint64_t b){
if(ZB==0u){
return;}
Z0 aligned_cnt=(ZB+7u)&~7u;Z0 tile_stride=aligned_cnt;
ZA rep_stride=Z5<ZA>(Z9/8u);Mul(ZH,ub_vecs,ub_vecs,ZB*Z9);
Z3<ZC>();bool fast_strip_max=(n>=40u&&n<=256u&&aligned_cnt<=64u);
Z4<Z1>strip_max_base=(SOLVE_MODE==2u)?ub_gather:ub_gather[3u*aligned_cnt];
Z0 max_strip_cnt=0u;Z0 strip_step=(SOLVE_MODE==2u&&n<=64u)?8u:((n<=64u)?16u:32u);
if constexpr(SOLVE_MODE==2u){
if(fast_strip_max){
for(Z0 c=0;c<n;c+=strip_step,++max_strip_cnt){
Z0 width=(n-c<strip_step)?(n-c):strip_step;
WholeReduceMax(strip_max_base[max_strip_cnt*aligned_cnt],ZH[c],width,ZB,1,1,rep_stride,ReduceOrder::ORDER_ONLY_VALUE);
}
}
}else{
if(n<=64u){
WholeReduceSum(ub_tmp,ZH,n,ZB,1,1,rep_stride);if(fast_strip_max){
for(Z0 c=0;c<n;c+=16u,++max_strip_cnt){
Z0 width=(n-c<16u)?(n-c):16u;
WholeReduceMax(strip_max_base[max_strip_cnt*aligned_cnt],ZH[c],width,ZB,1,1,rep_stride,ReduceOrder::ORDER_ONLY_VALUE);
}
}
Z3<ZC>();}else{
Z0 strip_idx=0u;for(Z0 c=0;c<n;c+=64u,++strip_idx){
Z0 width=(n-c<64u)?(n-c):64u;if(strip_idx==0u){
WholeReduceSum(ub_tmp,ZH[c],width,ZB,1,1,rep_stride);}else{
WholeReduceSum(ub_gather[(strip_idx-1u)*aligned_cnt],ZH[c],width,ZB,1,1,rep_stride);}
}
if(fast_strip_max){
for(Z0 c=0;c<n;c+=32u,++max_strip_cnt){
Z0 width=(n-c<32u)?(n-c):32u;
WholeReduceMax(strip_max_base[max_strip_cnt*aligned_cnt],ZH[c],width,ZB,1,1,rep_stride,ReduceOrder::ORDER_ONLY_VALUE);
}
}
Z3<ZC>();if(strip_idx==4u){
Add(ub_tmp,ub_tmp,ub_gather,ZB);
Add(ub_gather[aligned_cnt],ub_gather[aligned_cnt],ub_gather[2u*aligned_cnt],ZB);
Z3<ZC>();Add(ub_tmp,ub_tmp,ub_gather[aligned_cnt],ZB);
}else{
for(Z0 s=1u;s<strip_idx;++s){
Add(ub_tmp,ub_tmp,ub_gather[(s-1u)*aligned_cnt],ZB);if(s+1u<strip_idx){
Z3<ZC>();}
}
}
}
}
Z4<Z0>col_idx_buf=ub_diag.ReinterpretCast<Z0>();for(Z0 c=0;c<tile_stride;++c){
col_idx_buf.Z8(c,c*Z9*4u);}
for(Z0 r=0;r<ZB;++r){
ub_row.Z8(r,ub_eig.ZF(ZJ+r)*matrix_scale);}
Z3<PIPE_ALL>();DataCopyPad(
gm_eigvals[b*n+ZJ],
ub_row,
DataCopyExtParams{1,ZB*4u,0,0,0});

for(Z0 r=0;r<ZB;++r){
Z4<Z1>sq_r=ZH[r*Z9];Z4<Z1>vec_r=ub_vecs[r*Z9];
Z0 scan_start=0u;Z0 scan_end=n;
if(fast_strip_max){
Z0 best_s=0u;Z1 best_m=strip_max_base.ZF(r);
for(Z0 s=1u;s<max_strip_cnt;++s){
Z1 m_s=strip_max_base.ZF(s*aligned_cnt+r);if(m_s>best_m){
best_m=m_s;best_s=s;
}
}
scan_start=best_s*strip_step;scan_end=(scan_start+strip_step<n)?(scan_start+strip_step):n;
}
Z0 max_idx=scan_start;Z1 max_sq=sq_r.ZF(scan_start);
Z0 i=scan_start+1u;for(;i+3u<scan_end;i+=4u){
Z1 v0=sq_r.ZF(i+0u);Z1 v1=sq_r.ZF(i+1u);
Z1 v2=sq_r.ZF(i+2u);Z1 v3=sq_r.ZF(i+3u);
if(v0>max_sq){max_sq=v0;max_idx=i+0u;}
if(v1>max_sq){max_sq=v1;max_idx=i+1u;}
if(v2>max_sq){max_sq=v2;max_idx=i+2u;}
if(v3>max_sq){max_sq=v3;max_idx=i+3u;}
}
for(;i<scan_end;++i){
Z1 v_sq=sq_r.ZF(i);if(v_sq>max_sq){
max_sq=v_sq;max_idx=i;
}
}
if constexpr(SOLVE_MODE==2u){
if(vec_r.ZF(max_idx)<0.0f){
Muls(vec_r,vec_r,-1.0f,n);}
}else{
Z1 sign_scale=((vec_r.ZF(max_idx)<0.0f)?-1.0f:1.0f)*ZI(ub_tmp.ZF(r));
Muls(vec_r,vec_r,sign_scale,n);}
}
Z3<ZC>();for(Z0 i=0;i<n;++i){
Gather(ZH[i*tile_stride],ub_vecs,col_idx_buf,i*4u,ZB);}
Z3<PIPE_ALL>();if(ZB==n&&tile_stride==n){
DataCopyPad(
gm_eigvecs[b*n*n],
ZH,
DataCopyExtParams{1,n*n*4u,0,0,0});}else if((n&63u)!=0u){
Z0 rot=(lane_id*8u)%n;Z0 tail_rows=n-rot;
DataCopyPad(
gm_eigvecs[b*n*n+rot*n+ZJ],
ZH[rot*tile_stride],
DataCopyExtParams{Z5<uint16_t>(tail_rows),ZB*4u,0,(n-ZB)*4u,0});if(rot>0u){
DataCopyPad(
gm_eigvecs[b*n*n+ZJ],
ZH,
DataCopyExtParams{Z5<uint16_t>(rot),ZB*4u,0,(n-ZB)*4u,0});}
}else{
DataCopyPad(
gm_eigvecs[b*n*n+ZJ],
ZH,
DataCopyExtParams{Z5<uint16_t>(n),ZB*4u,0,(n-ZB)*4u,0});}
Z3<PIPE_ALL>();}


__aicore__ __attribute__((noinline)) void SolveTridiagSturmLarge(Z1*work){
Z0 w_cap=(n<=256u)?256u:512u;Z1*dd=work;
Z1*ee=work+w_cap;Z1*de_pairs=work+2u*w_cap;
Z1*q_lo=work+4u*w_cap;Z1*q_hi=q_lo+64;
Z1*iso_lo=q_hi+64;Z1*iso_hi=iso_lo+64;

Z0*u_base=reinterpret_cast<Z0*>(iso_hi+64);Z0*q_lc=u_base;
Z0*q_hc=q_lc+64;Z0*q_depth=q_hc+64;
Z0*act_q=q_depth+64;Z0*act_depth=act_q+64;


Z1*q_plo=reinterpret_cast<Z1*>(act_depth+64);Z1*q_phi=q_plo+64;
Z1*iso_plo=q_phi+64;Z1*iso_phi=iso_plo+64;
ZG*q_elo=reinterpret_cast<ZG*>(iso_phi+64);ZG*q_ehi=q_elo+64;
ZG*iso_elo=q_ehi+64;ZG*iso_ehi=iso_elo+64;
int8_t*iso_stag=reinterpret_cast<int8_t*>(iso_ehi+64);

for(Z0 r=0;r<ZB;++r){
iso_plo[r]=0.0f;iso_phi[r]=0.0f;iso_elo[r]=-30000;iso_ehi[r]=-30000;
iso_stag[r]=0;}

Z1 d0_v=dd[0],e0_v=ee[0],abs_e0=Z7(e0_v);Z1 gersh_min=d0_v-abs_e0,gersh_max=d0_v+abs_e0;
Z1 max_elem=__builtin_fmaxf(Z7(d0_v),abs_e0);bool is_diag=(e0_v==0.0f);
de_pairs[1]=e0_v*e0_v;Z1 prev_abs_e=abs_e0;
for(Z0 i=1u;i+1u<n;++i){
Z1 d_i=dd[i],e_i=ee[i],abs_e=Z7(e_i);de_pairs[i*2u-2u]=d_i;
de_pairs[i*2u+1u]=e_i*e_i;if(e_i!=0.0f)is_diag=false;
Z1 rad=prev_abs_e+abs_e;prev_abs_e=abs_e;
gersh_min=__builtin_fminf(gersh_min,d_i-rad);gersh_max=__builtin_fmaxf(gersh_max,d_i+rad);
max_elem=__builtin_fmaxf(max_elem,__builtin_fmaxf(Z7(d_i),abs_e));}
{
Z1 d_last=dd[n-1u];de_pairs[(n-2u)*2u]=d_last;
gersh_min=__builtin_fminf(gersh_min,d_last-prev_abs_e);
gersh_max=__builtin_fmaxf(gersh_max,d_last+prev_abs_e);
max_elem=__builtin_fmaxf(max_elem,Z7(d_last));}
if(is_diag){
Z1*sd=work+2u*w_cap;for(Z0 i=0;i<n;++i){
Z1 val=dd[i];Z0 j=i;
while(j>0u&&sd[j-1u]>val){
sd[j]=sd[j-1u];--j;
}
sd[j]=val;}
for(Z0 r=0;r<ZB;++r){
ub_eig.Z8(ZJ+r,sd[ZJ+r]);}
work[1023]=-1.0f;return;
}
work[1023]=max_elem;if(ZB==0u){
return;}
Z1 pad=1e-5f*(Z7(gersh_min)+Z7(gersh_max))+1e-6f;gersh_min-=pad;
gersh_max+=pad;

Z0 row_end=ZJ+ZB;Z0 ext_start=(ZJ>0u)?(ZJ-1u):0u;
Z0 ext_end=(row_end<n)?(row_end+1u):n;
Z1 left_nb_lo=gersh_min,left_nb_hi=gersh_min,left_nb_plo=0.0f,left_nb_phi=0.0f;
ZG left_nb_elo=-30000,left_nb_ehi=-30000;
Z1 right_nb_lo=gersh_max,right_nb_hi=gersh_max,right_nb_plo=0.0f,right_nb_phi=0.0f;
ZG right_nb_elo=-30000,right_nb_ehi=-30000;Z1 tol_cluster=1.0e-6f*max_elem;
Z1 tol_single=(FAST200?1.5e-6f:((n>200u)?8.0e-7f:1.0e-5f))*max_elem;

Z0 q_head=0u;Z0 q_tail=1u;
Z0 q_count=1u;q_lo[0]=gersh_min;
q_hi[0]=gersh_max;q_plo[0]=0.0f;q_phi[0]=0.0f;
q_elo[0]=-30000;q_ehi[0]=-30000;q_lc[0]=0u;
q_hc[0]=n;q_depth[0]=0u;

Z1 b_lo[4],b_hi[4],b_mid[4];Z1 b_plo[4],b_phi[4],b_pval[4];
ZG b_elo[4],b_ehi[4],b_pexp[4];Z0 b_lc[4],b_hc[4],b_depth[4],b_kp[4],b_pos[4],b_neg[4];


while(q_count>0u){
Z0 bcnt=0u;while(q_count>0u&&bcnt<4u){
Z1 lo=q_lo[q_head],hi=q_hi[q_head];Z1 plo=q_plo[q_head],phi=q_phi[q_head];
ZG elo=q_elo[q_head],ehi=q_ehi[q_head];Z0 lc=q_lc[q_head],hc=q_hc[q_head],depth=q_depth[q_head];
q_head=(q_head+1u)&63u;--q_count;

if(hc-lc==1u){
if(lc>=ZJ&&lc<row_end){
Z0 r=lc-ZJ;iso_lo[r]=lo;iso_hi[r]=hi;
iso_plo[r]=plo;iso_phi[r]=phi;iso_elo[r]=elo;iso_ehi[r]=ehi;
}else if(ZJ>0u&&lc==ZJ-1u){
left_nb_lo=lo;left_nb_hi=hi;left_nb_plo=plo;left_nb_phi=phi;left_nb_elo=elo;left_nb_ehi=ehi;
}else if(row_end<n&&lc==row_end){
right_nb_lo=lo;right_nb_hi=hi;right_nb_plo=plo;right_nb_phi=phi;right_nb_elo=elo;right_nb_ehi=ehi;}
continue;}
Z1 mid=0.5f*lo+0.5f*hi;if(depth>=32u||mid==lo||mid==hi||(hi-lo)<=tol_cluster+1e-30f){
Z0 s_idx=(lc>ZJ)?lc:ZJ;Z0 e_idx=(hc<row_end)?hc:row_end;
for(Z0 j=s_idx;j<e_idx;++j){
iso_lo[j-ZJ]=mid;iso_hi[j-ZJ]=mid;
}
if(ZJ>0u&&lc<=ZJ-1u&&hc>ZJ-1u){left_nb_lo=mid;left_nb_hi=mid;}
if(row_end<n&&lc<=row_end&&hc>row_end){right_nb_lo=mid;right_nb_hi=mid;}
continue;}
b_lo[bcnt]=lo;b_hi[bcnt]=hi;b_plo[bcnt]=plo;b_phi[bcnt]=phi;
b_elo[bcnt]=elo;b_ehi[bcnt]=ehi;b_lc[bcnt]=lc;b_hc[bcnt]=hc;
b_depth[bcnt]=depth;++bcnt;
}
if(bcnt==0u)break;

Z0 lane_pos=0u;for(Z0 b=0;b<bcnt;++b){
Z0 kp=(bcnt==1u)?4u:((bcnt==2u)?2u:((bcnt==3u&&b==0u)?2u:1u));b_kp[b]=kp;
b_pos[b]=lane_pos;Z1 lo=b_lo[b],hi=b_hi[b],w_b=hi-lo;
Z0 lc_b=b_lc[b],hc_b=b_hc[b],sp_b=hc_b-lc_b;Z0 wl=(lc_b>ext_start)?lc_b:ext_start;
Z0 wh=(hc_b<ext_end)?hc_b:ext_end;if((FAST200||n==256u)&&sp_b>=6u&&(wh-wl)<sp_b){
Z1 inv_sp=1.0f/Z5<Z1>(Z5<ZG>(sp_b));
Z1 f0=Z5<Z1>(Z5<ZG>(wl-lc_b))*inv_sp;
Z1 f1=Z5<Z1>(Z5<ZG>(wh-lc_b))*inv_sp;Z1 a=0.6f*f0;
Z1 b_end=(1.0f-0.6f)+0.6f*f1;lo=lo+a*w_b;
w_b=(b_end-a)*w_b;}
Z1 inv_kp1=(kp==4u)?0.2f:((kp==2u)?(1.0f/3.0f):0.5f);Z1 step=w_b*inv_kp1;
b_mid[lane_pos]=lo+step;if(kp>=2u){
b_mid[lane_pos+1u]=lo+2.0f*step;if(kp==4u){
b_mid[lane_pos+2u]=lo+3.0f*step;b_mid[lane_pos+3u]=lo+4.0f*step;
}
}
lane_pos+=kp;}

#if !EIGH_K96_PASS
if constexpr(SOLVE_MODE==3u&&!FAST200){
if(n==256u)EvalSturm4Fast256(dd[0],de_pairs,b_mid,b_neg,b_pval,b_pexp);
else EvalSturm4(dd[0],de_pairs,b_mid,b_neg,b_pval,b_pexp);}else
#endif
{EvalSturm4(dd[0],de_pairs,b_mid,b_neg,b_pval,b_pexp);}

for(Z0 b=0;b<bcnt;++b){
Z0 lc=b_lc[b],hc=b_hc[b],kp=b_kp[b],pos=b_pos[b];Z0 next_depth=b_depth[b]+((kp>=3u)?2u:1u);
Z1 prev_v=b_lo[b],prev_p=b_plo[b];ZG prev_e=b_elo[b];
Z0 prev_c=lc;for(Z0 k=0;k<=kp;++k){
Z1 cur_v=(k<kp)?b_mid[pos+k]:b_hi[b];Z1 cur_p=(k<kp)?b_pval[pos+k]:b_phi[b];
ZG cur_e=(k<kp)?b_pexp[pos+k]:b_ehi[b];Z0 cur_c=(k<kp)?b_neg[pos+k]:hc;
if(cur_c<prev_c)cur_c=prev_c;if(cur_c>hc)cur_c=hc;
if(prev_c<cur_c&&cur_c>ext_start&&prev_c<ext_end){
if(cur_c-prev_c==1u){
if(prev_c>=ZJ&&prev_c<row_end){
Z0 r=prev_c-ZJ;iso_lo[r]=prev_v;iso_hi[r]=cur_v;
iso_plo[r]=prev_p;iso_phi[r]=cur_p;iso_elo[r]=prev_e;iso_ehi[r]=cur_e;
}else if(ZJ>0u&&prev_c==ZJ-1u){
left_nb_lo=prev_v;left_nb_hi=cur_v;left_nb_plo=prev_p;left_nb_phi=cur_p;left_nb_elo=prev_e;left_nb_ehi=cur_e;
}else if(row_end<n&&prev_c==row_end){
right_nb_lo=prev_v;right_nb_hi=cur_v;right_nb_plo=prev_p;right_nb_phi=cur_p;right_nb_elo=prev_e;right_nb_ehi=cur_e;
}
}else{
q_lo[q_tail]=prev_v;q_hi[q_tail]=cur_v;q_plo[q_tail]=prev_p;q_phi[q_tail]=cur_p;
q_elo[q_tail]=prev_e;q_ehi[q_tail]=cur_e;q_lc[q_tail]=prev_c;q_hc[q_tail]=cur_c;
q_depth[q_tail]=next_depth;q_tail=(q_tail+1u)&63u;
++q_count;}
}
prev_v=cur_v;prev_p=cur_p;prev_e=cur_e;prev_c=cur_c;}
}
}

if(ZJ>0u){
Z1 eig_l=left_nb_hi;Z1 t_nb=0.5f;
if(ComputeSecantWeight(left_nb_plo,left_nb_elo,left_nb_phi,left_nb_ehi,t_nb)){
eig_l=left_nb_lo+t_nb*(left_nb_hi-left_nb_lo);left_nb_hi=0.5f*(eig_l+left_nb_hi);
}
ub_eig.Z8(ZJ-1u,eig_l);}
if(row_end<n){
Z1 eig_r=right_nb_lo;Z1 t_nb=0.5f;
if(ComputeSecantWeight(right_nb_plo,right_nb_elo,right_nb_phi,right_nb_ehi,t_nb)){
eig_r=right_nb_lo+t_nb*(right_nb_hi-right_nb_lo);right_nb_lo=0.5f*(right_nb_lo+eig_r);
}
ub_eig.Z8(row_end,eig_r);}


Z1*iso_clo=q_lo,*iso_chi=q_hi;Z1*iso_tsec=reinterpret_cast<Z1*>(q_hc);
for(Z0 r=0;r<ZB;++r){
Z1 lo=iso_lo[r],hi=iso_hi[r];Z1 t_s=0.5f;
if(ComputeSecantWeight(iso_plo[r],iso_elo[r],iso_phi[r],iso_ehi[r],t_s)){
iso_tsec[r]=t_s;Z1 est=lo+t_s*(hi-lo);
iso_clo[r]=0.5f*(lo+est);iso_chi[r]=0.5f*(est+hi);
}else{
iso_tsec[r]=-1.0f;iso_clo[r]=lo;
iso_chi[r]=hi;}
}

Z1*hist_x=q_phi;Z1*hist_p=q_plo;ZG*hist_e=q_elo;
if(FAST200||n==256u)for(Z0 r=0;r<ZB;++r)hist_e[r]=-30000;Z0 a_head=0u,a_tail=0u,a_cnt=0u;
Z1 max_gap_tol=1.6e-4f*max_elem;Z1 min_sep_gap=0.0005f*max_elem;
Z1 near_zero_thr=0.015f*max_elem;

for(Z0 r=0;r<ZB;++r){
act_depth[r]=0u;Z1 lo=iso_lo[r],hi=iso_hi[r];
Z1 mid=0.5f*lo+0.5f*hi;Z1 cur_tol=tol_single;
Z1 l_gap=(r>0u)?(iso_clo[r]-iso_chi[r-1u]):((ZJ>0u)?(iso_clo[r]-left_nb_hi):1.0e30f);
Z1 r_gap=((r+1u)<ZB)?(iso_clo[r+1u]-iso_chi[r]):((row_end<n)?(right_nb_lo-iso_chi[r]):1.0e30f);
Z1 gap=(l_gap<r_gap)?l_gap:r_gap;
if(gap>=min_sep_gap&&(Z7(mid)>=near_zero_thr||gap>=0.0015f*max_elem)){
Z1 ptol=(FAST200?0.0060f:0.0068f)*gap;if(ptol>max_gap_tol)ptol=max_gap_tol;
if(ptol>cur_tol)cur_tol=ptol;}
Z1 t_sec=iso_tsec[r];bool has_sec=(t_sec>=0.0f);
Z1 width=hi-lo;Z1 eff_width=has_sec?(1.85f*t_sec*(1.0f-t_sec)*width):width;
if(mid==lo||mid==hi||((FAST200||n==256u)&&width<=7.0f*cur_tol&&eff_width<=cur_tol+1e-30f)||width<=cur_tol+1e-30f){
if(has_sec)mid=lo+t_sec*width;ub_eig.Z8(ZJ+r,mid);
}else{
act_q[a_tail]=r;a_tail=(a_tail+1u)&63u;
++a_cnt;}
}

Z1 b8_mid[8],b8_pval[8];ZG b8_pexp[8];
Z0 b8_kp[8],b8_pos[8],b8_neg[8],r_batch[8];while(a_cnt>0u){
bool use_poly8=(n!=256u&&a_cnt>4u);Z0 max_b=use_poly8?8u:4u;
Z0 bcnt=(a_cnt<max_b)?a_cnt:max_b;Z0 lane_pos=0u;
for(Z0 b=0;b<bcnt;++b){
Z0 r=act_q[a_head];a_head=(a_head+1u)&63u;
--a_cnt;r_batch[b]=r;
Z0 kp=use_poly8?((b<8u-bcnt)?2u:1u):((bcnt==1u)?4u:((bcnt==2u)?2u:((bcnt==3u&&b==0u)?2u:1u)));
b8_kp[b]=kp;b8_pos[b]=lane_pos;
Z1 lo=iso_lo[r],hi=iso_hi[r];Z1 width=hi-lo;
Z1 t_sec=iso_tsec[r];bool has_sec=(t_sec>=0.0f);
if(kp==1u&&has_sec){
int8_t stg=iso_stag[r];
Z1 lg=(r>0u)?(iso_clo[r]-iso_chi[r-1u]):((ZJ>0u)?(iso_clo[r]-left_nb_hi):1.0e30f);
Z1 rg=((r+1u)<ZB)?(iso_clo[r+1u]-iso_chi[r]):((row_end<n)?(right_nb_lo-iso_chi[r]):1.0e30f);
Z1 gp=(lg<rg)?lg:rg;if(width>0.55f*gp||stg>=3||stg<=-3){
t_sec=0.5f;iso_stag[r]=0;
}else{
bool iqi_ok=false;Z1 iqi_t=t_sec;
if((FAST200||n==256u)&&hist_e[r]!=-30000&&(hist_x[r]<lo||hist_x[r]>hi)){
ZG el=iso_elo[r],eh=iso_ehi[r],ez=hist_e[r];ZG em=(el>eh)?el:eh;if(ez>em)em=ez;
if(em-el<=1&&em-eh<=1&&em-ez<=1){
Z1 fl=iso_plo[r],fh=iso_phi[r],fz=hist_p[r];if(el<em)fl*=5.4210108624275222e-20f;
if(eh<em)fh*=5.4210108624275222e-20f;if(ez<em)fz*=5.4210108624275222e-20f;
Z1 fm=(fl>fh)?fl:fh;if(fz>fm)fm=fz;if(fm>0.0f){
fl/=fm;fh/=fm;fz/=fm;if(hist_x[r]<lo)fz=-fz;
Z1 tz=(hist_x[r]-lo)/width;Z1 d0=fz-fh,d1=fz+fl;
if(Z7(d0)>1.0e-6f&&Z7(d1)>1.0e-6f&&Z7(tz)<8.0f){
iqi_t=(fl/(fl+fh))*(fz/d0)-tz*(fl/d1)*(fh/d0);iqi_ok=iqi_t>=0.015f&&iqi_t<=0.985f;
}
}
}
}
if(iqi_ok){
t_sec=iqi_t;}else if(stg==2){
t_sec=(t_sec<0.5f)?(0.45f*t_sec):(t_sec-0.20f);if(t_sec<0.015f)t_sec=0.015f;
}else if(stg==-2){
t_sec=(t_sec>0.5f)?(1.0f-0.45f*(1.0f-t_sec)):(t_sec+0.20f);if(t_sec>0.985f)t_sec=0.985f;
}else{
if(stg==1){
t_sec=(t_sec>0.92f)?(1.0f-2.2f*(1.0f-t_sec)-0.012f):((t_sec<0.25f)?(0.65f*t_sec):(t_sec-0.045f));
}else if(stg==-1){
t_sec=(t_sec<0.08f)?(2.2f*t_sec+0.012f):((t_sec>0.75f)?(1.0f-0.65f*(1.0f-t_sec)):(t_sec+0.045f));}
if((ZJ+r==0u||ZJ+r+1u==n)&&width>6.0f*max_gap_tol){
t_sec=0.5f;}else{
constexpr Z1 clamp_lo=0.015f;if(t_sec<clamp_lo)t_sec=clamp_lo;
if(t_sec>1.0f-clamp_lo)t_sec=1.0f-clamp_lo;}
}
}
b8_mid[lane_pos]=lo+t_sec*width;}else if(kp==2u&&has_sec){
Z1 tc=t_sec;if(tc<0.12f)tc=0.12f;
if(tc>0.88f)tc=0.88f;b8_mid[lane_pos+0u]=lo+(tc-0.08f)*width;
b8_mid[lane_pos+1u]=lo+(tc+0.08f)*width;
}else if((FAST200||n==256u)&&kp==4u&&has_sec&&iso_stag[r]>=-1&&iso_stag[r]<=1){
Z1 tc=t_sec;if(tc<0.19f)tc=0.19f;
if(tc>0.81f)tc=0.81f;b8_mid[lane_pos+0u]=lo+(tc-0.15f)*width;
b8_mid[lane_pos+1u]=lo+(tc-0.03f)*width;b8_mid[lane_pos+2u]=lo+(tc+0.03f)*width;
b8_mid[lane_pos+3u]=lo+(tc+0.15f)*width;}else{
Z1 inv_kp1=(kp==4u)?0.2f:((kp==2u)?(1.0f/3.0f):0.5f);Z1 step=width*inv_kp1;
b8_mid[lane_pos]=lo+step;if(kp>=2u){
b8_mid[lane_pos+1u]=lo+2.0f*step;if(kp==4u){
b8_mid[lane_pos+2u]=lo+3.0f*step;b8_mid[lane_pos+3u]=lo+4.0f*step;
}
}
}
lane_pos+=kp;}


#if !EIGH_K96_PASS
if constexpr(SOLVE_MODE==3u&&!FAST200){
if(n==256u){
EvalPoly4Sequential256(dd[0],de_pairs,b8_mid,b8_neg,b8_pval,b8_pexp);
if(use_poly8)EvalPoly4Sequential256(dd[0],de_pairs,b8_mid+4u,b8_neg+4u,b8_pval+4u,b8_pexp+4u);}else{
if(use_poly8){
EvalPoly4SameScaleAs8(dd[0],de_pairs,b8_mid,b8_neg,b8_pval,b8_pexp);
EvalPoly4SameScaleAs8(dd[0],de_pairs,b8_mid+4u,b8_neg+4u,b8_pval+4u,b8_pexp+4u);}else{
EvalPoly4(dd[0],de_pairs,b8_mid,b8_neg,b8_pval,b8_pexp);}
}
}else
#endif
{
if(use_poly8){
EvalPoly4SameScaleAs8(dd[0],de_pairs,b8_mid,b8_neg,b8_pval,b8_pexp);
EvalPoly4SameScaleAs8(dd[0],de_pairs,b8_mid+4u,b8_neg+4u,b8_pval+4u,b8_pexp+4u);}else{
EvalPoly4(dd[0],de_pairs,b8_mid,b8_neg,b8_pval,b8_pexp);}
}


for(Z0 b=0;b<bcnt;++b){
Z0 r=r_batch[b];Z0 kp=b8_kp[b];
Z0 pos=b8_pos[b];Z1 lo=iso_lo[r],hi=iso_hi[r];
Z1 plo=iso_plo[r],phi=iso_phi[r];ZG elo=iso_elo[r],ehi=iso_ehi[r];
Z0 target=ZJ+r+1u;Z1 new_lo=b8_mid[pos+kp-1u];
Z1 new_plo=b8_pval[pos+kp-1u];ZG new_elo=b8_pexp[pos+kp-1u];
Z1 new_hi=hi;Z1 new_phi=phi;
ZG new_ehi=ehi;int8_t next_stag=(iso_stag[r]<0)?(iso_stag[r]-1):-1;
Z0 hit_k=kp;for(Z0 k=0;k<kp;++k){
if(b8_neg[pos+k]==(target&1u)){
hit_k=k;if(k==0u){
new_lo=lo;new_plo=plo;
new_elo=elo;next_stag=(iso_stag[r]>0)?(iso_stag[r]+1):1;
}else{
new_lo=b8_mid[pos+k-1u];new_plo=b8_pval[pos+k-1u];
new_elo=b8_pexp[pos+k-1u];next_stag=0;
}
new_hi=b8_mid[pos+k];new_phi=b8_pval[pos+k];
new_ehi=b8_pexp[pos+k];break;
}
}
if(FAST200||n==256u){
if(hit_k>=2u){Z0 hk=hit_k-2u;hist_x[r]=b8_mid[pos+hk];hist_p[r]=b8_pval[pos+hk];hist_e[r]=b8_pexp[pos+hk];}
else if(hit_k+1u<kp){Z0 hk=hit_k+1u;hist_x[r]=b8_mid[pos+hk];hist_p[r]=b8_pval[pos+hk];hist_e[r]=b8_pexp[pos+hk];}
else if(new_lo>lo){hist_x[r]=lo;hist_p[r]=plo;hist_e[r]=elo;}
else if(new_hi<hi){hist_x[r]=hi;hist_p[r]=phi;hist_e[r]=ehi;}
}
iso_stag[r]=next_stag;iso_lo[r]=new_lo;iso_hi[r]=new_hi;
iso_plo[r]=new_plo;iso_phi[r]=new_phi;iso_elo[r]=new_elo;iso_ehi[r]=new_ehi;
act_depth[r]+=(kp>=7u)?3u:((kp>=3u)?2u:1u);Z1 next_mid=0.5f*new_lo+0.5f*new_hi;
Z1 t_sec=0.5f;bool has_sec=ComputeSecantWeight(new_plo,new_elo,new_phi,new_ehi,t_sec);
Z1 width=new_hi-new_lo;if(has_sec){
iso_tsec[r]=t_sec;Z1 est=new_lo+t_sec*width;
iso_clo[r]=0.5f*(new_lo+est);iso_chi[r]=0.5f*(est+new_hi);
}else{
iso_tsec[r]=-1.0f;iso_clo[r]=new_lo;
iso_chi[r]=new_hi;}
Z1 cur_tol=tol_single;
Z1 l_gap=(r>0u)?(iso_clo[r]-iso_chi[r-1u]):((ZJ>0u)?(iso_clo[r]-left_nb_hi):1.0e30f);
Z1 r_gap=((r+1u)<ZB)?(iso_clo[r+1u]-iso_chi[r]):((row_end<n)?(right_nb_lo-iso_chi[r]):1.0e30f);
Z1 gap=(l_gap<r_gap)?l_gap:r_gap;
if(gap>=min_sep_gap&&(Z7(next_mid)>=near_zero_thr||gap>=0.0015f*max_elem)){
Z1 ptol=(FAST200?0.0060f:0.0068f)*gap;if(ptol>max_gap_tol)ptol=max_gap_tol;
if(ptol>cur_tol)cur_tol=ptol;}
Z1 eff_width=has_sec?((((FAST200||n==256u)&&hist_e[r]!=-30000)?1.35f:1.85f)*t_sec*(1.0f-t_sec)*width):width;
if(act_depth[r]>=32u||next_mid==new_lo||next_mid==new_hi||(((FAST200||n==256u)?(next_stag>=-2&&next_stag<=2):(next_stag>=-1&&next_stag<=1))&&width<=7.0f*cur_tol&&eff_width<=cur_tol+1e-30f)||width<=cur_tol+1e-30f){
if(has_sec){
if((FAST200||n==256u)&&hist_e[r]!=-30000&&(hist_x[r]<new_lo||hist_x[r]>new_hi)){
ZG el=new_elo,eh=new_ehi,ez=hist_e[r];ZG em=(el>eh)?el:eh;if(ez>em)em=ez;
if(em-el<=1&&em-eh<=1&&em-ez<=1){
Z1 fl=new_plo,fh=new_phi,fz=hist_p[r];if(el<em)fl*=5.4210108624275222e-20f;
if(eh<em)fh*=5.4210108624275222e-20f;if(ez<em)fz*=5.4210108624275222e-20f;
Z1 fm=(fl>fh)?fl:fh;if(fz>fm)fm=fz;if(fm>0.0f){
fl/=fm;fh/=fm;fz/=fm;if(hist_x[r]<new_lo)fz=-fz;
Z1 tz=(hist_x[r]-new_lo)/width;Z1 d0=fz-fh,d1=fz+fl;
if(Z7(d0)>1.0e-6f&&Z7(d1)>1.0e-6f&&Z7(tz)<8.0f){
Z1 it=(fl/(fl+fh))*(fz/d0)-tz*(fl/d1)*(fh/d0);if(it>=0.005f&&it<=0.995f)t_sec=it;
}
}
}
}
next_mid=new_lo+t_sec*width;}
ub_eig.Z8(ZJ+r,next_mid);}else{
act_q[a_tail]=r;a_tail=(a_tail+1u)&63u;
++a_cnt;}
}
}
}
__aicore__ inline void ComputeInverseVectorsLarge(Z1*work,ZA*swap_mask){
#if EIGH_K96_PASS
#else
if constexpr(FAST200){ComputeInverseVectorsLarge200Fast(work,swap_mask);return;}
#endif
if(n<=200u){
ComputeInverseVectorsLarge200(work,swap_mask);}else{
ComputeInverseVectorsLarge256(work,swap_mask);}
}

__aicore__ __attribute__((noinline))void ComputeInverseVectorsLarge200(Z1*work,ZA*swap_mask){
Z1*dd=work;Z1*ee=work+((n<=256u)?256u:512u);
bool is_diag=true;Z1 max_elem=0.0f;

Z1 saved_max=work[1023];is_diag=(saved_max<0.0f);
max_elem=is_diag?0.0f:saved_max;

if(ZB>0u&&(is_diag||n<Z9)){
Duplicate(ub_vecs,0.0f,ZB*Z9);Z3<PIPE_ALL>();
}

if(is_diag){
if constexpr(SOLVE_MODE!=2u&&SOLVE_MODE!=3u){
for(Z0 i=0;i<n;++i){
dd[i]=ub_diag.ZF(i);}
}
for(Z0 r=0;r<ZB;++r){
Z0 col_idx=ZJ+r;Z4<Z1>vec_r=ub_vecs[r*Z9];
for(Z0 i=0;i<n;++i){
Z0 rank=0u;Z1 di=dd[i];
for(Z0 k=0;k<n;++k){
if(dd[k]<di||(dd[k]==di&&k<i)){
++rank;}
}
if(rank==col_idx){
vec_r.Z8(i,1.0f);break;
}
}
}
Z3<PIPE_ALL>();return;
}

Z1 v_scale[32];for(Z0 ri=0;ri<ZB;++ri)v_scale[ri]=0.0f;
Z0 r=0u;if(n<=256u){
Z1*ee_rat=work+480;Z1*inv_ee=work+688;
Z0 i_ee=0u;for(;i_ee+4u<n;i_ee+=4u){
Z1 e0=ee[i_ee],e1=ee[i_ee+1u],e2=ee[i_ee+2u],e3=ee[i_ee+3u],e4=ee[i_ee+4u];
if(Z7(e0)>=1e-7f&&Z7(e1)>=1e-7f&&Z7(e2)>=1e-7f&&Z7(e3)>=1e-7f){
Z1 p01=e0*e1,p23=e2*e3;Z1 inv_all=1.0f/(p01*p23);
Z1 inv01=inv_all*p23,inv23=inv_all*p01;Z1 iv0=inv01*e1,iv1=inv01*e0,iv2=inv23*e3,iv3=inv23*e2;
inv_ee[i_ee]=iv0;inv_ee[i_ee+1u]=iv1;inv_ee[i_ee+2u]=iv2;inv_ee[i_ee+3u]=iv3;
ee_rat[i_ee]=e1*iv0;ee_rat[i_ee+1u]=e2*iv1;ee_rat[i_ee+2u]=e3*iv2;ee_rat[i_ee+3u]=e4*iv3;}else{
for(Z0 k=0;k<4u;++k){
Z1 ek=ee[i_ee+k];Z1 ivk=(ek!=0.0f)?(1.0f/ek):0.0f;
inv_ee[i_ee+k]=ivk;ee_rat[i_ee+k]=ee[i_ee+k+1u]*ivk;
}
}
}
for(;i_ee+1u<n;++i_ee){
Z1 ei=ee[i_ee];Z1 inv_e=(ei!=0.0f)?(1.0f/ei):0.0f;
inv_ee[i_ee]=inv_e;ee_rat[i_ee]=ee[i_ee+1u]*inv_e;
}


Z1*aos=work+896;for(;r+2u<ZB;r+=4u){
Z0 bcnt=(ZB-r<4u)?(ZB-r):4u;Z0 c0=ZJ+r;
Z0 c1=(bcnt>1u)?(c0+1u):c0;Z0 c2=(bcnt>2u)?(c0+2u):c1;
Z0 c3=(bcnt>3u)?(c0+3u):c2;Z1 lam0=ub_eig.ZF(c0);
Z1 lam1=ub_eig.ZF(c1);Z1 lam2=ub_eig.ZF(c2);
Z1 lam3=ub_eig.ZF(c3);


Z0 sm0=(c0*13u)%23u;Z0 sm1=sm0+13u;if(sm1>=23u)sm1-=23u;
Z0 sm2=sm1+13u;if(sm2>=23u)sm2-=23u;Z0 sm3=sm2+13u;if(sm3>=23u)sm3-=23u;
Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 nq1=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm1));
Z1 nq2=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm2));
Z1 nq3=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm3));
Z1 cq0=nq0,cq1=nq1,cq2=nq2,cq3=nq3;

Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];Z1 cur_p1=dd[0]-lam1,cur_u1=ee[0];
Z1 cur_p2=dd[0]-lam2,cur_u2=ee[0];Z1 cur_p3=dd[0]-lam3,cur_u3=ee[0];

for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i];Z1 abs_m=Z7(m_i);
Z1 d_next=dd[i+1u];Z1 e_next=ee[i+1u];
Z1 nd0=d_next-lam0;Z1 nd1=d_next-lam1;
Z1 nd2=d_next-lam2;Z1 nd3=d_next-lam3;

bool sw0=(Z7(cur_p0)<abs_m);bool sw1=(Z7(cur_p1)<abs_m);
bool sw2=(Z7(cur_p2)<abs_m);bool sw3=(Z7(cur_p3)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u)|(sw2?4u:0u)|(sw3?8u:0u));swap_mask[i]=sm;

Z1 inv_m=inv_ee[i];Z1 inv0,inv1,inv2,inv3;
if(sm==15u){
inv0=inv_m;inv1=inv_m;inv2=inv_m;inv3=inv_m;}else if(abs_m>=1e-7f){
Z1 p0=sw0?1.0f:cur_p0;Z1 p1=sw1?1.0f:cur_p1;
Z1 p2=sw2?1.0f:cur_p2;Z1 p3=sw3?1.0f:cur_p3;
Z1 p01=p0*p1;Z1 p23=p2*p3;
Z1 inv_all=1.0f/(p01*p23);Z1 inv01=inv_all*p23;
Z1 inv23=inv_all*p01;inv0=sw0?inv_m:(inv01*p1);
inv1=sw1?inv_m:(inv01*p0);inv2=sw2?inv_m:(inv23*p3);
inv3=sw3?inv_m:(inv23*p2);}else{
inv0=sw0?inv_m:(1.0f/ClampPivot(cur_p0));inv1=sw1?inv_m:(1.0f/ClampPivot(cur_p1));
inv2=sw2?inv_m:(1.0f/ClampPivot(cur_p2));inv3=sw3?inv_m:(1.0f/ClampPivot(cur_p3));
}

Z1 f0=(sw0?cur_p0:m_i)*inv0;Z1 f1=(sw1?cur_p1:m_i)*inv1;
Z1 f2=(sw2?cur_p2:m_i)*inv2;Z1 f3=(sw3?cur_p3:m_i)*inv3;

Z1*s=aos+i*16u;s[0]=inv0;s[1]=inv1;s[2]=inv2;s[3]=inv3;
s[4]=f0;s[5]=f1;s[6]=f2;s[7]=f3;Z1 u1_0=sw0?nd0:cur_u0;
Z1 u1_1=sw1?nd1:cur_u1;Z1 u1_2=sw2?nd2:cur_u2;
Z1 u1_3=sw3?nd3:cur_u3;s[8]=u1_0*inv0;
s[9]=u1_1*inv1;s[10]=u1_2*inv2;
s[11]=u1_3*inv3;

cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_p2=(sw2?cur_u2:nd2)-f2*u1_2;cur_p3=(sw3?cur_u3:nd3)-f3*u1_3;

cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
cur_u2=sw2?(-f2*e_next):e_next;cur_u3=sw3?(-f3*e_next):e_next;

nq0+=(nq0>=1.1875f)?-0.1875f:0.53125f;nq1+=(nq1>=1.1875f)?-0.1875f:0.53125f;
nq2+=(nq2>=1.1875f)?-0.1875f:0.53125f;nq3+=(nq3>=1.1875f)?-0.1875f:0.53125f;

Z1 qi0=sw0?nq0:cq0;Z1 qi1=sw1?nq1:cq1;
Z1 qi2=sw2?nq2:cq2;Z1 qi3=sw3?nq3:cq3;
Z1 b0=sw0?cq0:nq0;Z1 b1=sw1?cq1:nq1;
Z1 b2=sw2?cq2:nq2;Z1 b3=sw3?cq3:nq3;
s[12]=qi0*inv0;s[13]=qi1*inv1;s[14]=qi2*inv2;s[15]=qi3*inv3;cq0=b0-f0*qi0;
cq1=b1-f1*qi1;cq2=b2-f2*qi2;
cq3=b3-f3*qi3;}
Z1*s_last=aos+(n-1u)*16u;Z1 lden0=ClampPivot(cur_p0);
Z1 lden1=ClampPivot(cur_p1);Z1 lden2=ClampPivot(cur_p2);
Z1 lden3=ClampPivot(cur_p3);
if(Z7(lden0)>=1e-7f&&Z7(lden1)>=1e-7f&&Z7(lden2)>=1e-7f&&Z7(lden3)>=1e-7f){
Z1 lp01=lden0*lden1,lp23=lden2*lden3;Z1 linv_all=1.0f/(lp01*lp23);
Z1 linv01=linv_all*lp23,linv23=linv_all*lp01;s_last[0]=linv01*lden1;
s_last[1]=linv01*lden0;s_last[2]=linv23*lden3;
s_last[3]=linv23*lden2;}else{
s_last[0]=1.0f/lden0;s_last[1]=1.0f/lden1;
s_last[2]=1.0f/lden2;s_last[3]=1.0f/lden3;
}


Z1 gl0=(c0>0u)?(lam0-ub_eig.ZF(c0-1u)):max_elem;
Z1 gr3=(c0+bcnt<n)?(ub_eig.ZF(c0+bcnt)-lam3):max_elem;Z1 g01=(bcnt>1u)?(lam1-lam0):max_elem;
Z1 g12=(bcnt>2u)?(lam2-lam1):max_elem;Z1 g23=(bcnt>3u)?(lam3-lam2):gr3;
Z1 g0=(gl0<g01)?gl0:g01;Z1 g1=(g01<g12)?g01:g12;
Z1 g2=(g12<g23)?g12:g23;Z1 g3=(bcnt>3u)?((g23<gr3)?g23:gr3):max_elem;
if(g0<0.0f)g0=0.0f;if(g1<0.0f)g1=0.0f;
if(g2<0.0f)g2=0.0f;if(g3<0.0f)g3=0.0f;
Z1 g0_sq=g0*g0,g1_sq=g1*g1,g2_sq=g2*g2,g3_sq=g3*g3;
Z1 min_g=((g0<g1)?g0:g1)<((g2<g3)?g2:g3)?((g0<g1)?g0:g1):((g2<g3)?g2:g3);
bool can_early_stop=(min_g>=0.00008f*max_elem);Z1 s0_0=0.0f,s0_1=0.0f,s0_2=0.0f,s0_3=0.0f;
Z0 max_iter=3u;

Z1 in0=1.0f,in1=1.0f,in2=1.0f,in3=1.0f;for(Z0 iter=0;iter<max_iter;++iter){
if(iter>0u){
cq0=aos[12]*in0;cq1=aos[13]*in1;cq2=aos[14]*in2;cq3=aos[15]*in3;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*16u;
Z1 nq0=s[28]*in0,nq1=s[29]*in1,nq2=s[30]*in2,nq3=s[31]*in3;Z1 qi0=(sm&1u)?nq0:cq0;
Z1 qi1=(sm&2u)?nq1:cq1;Z1 qi2=(sm&4u)?nq2:cq2;
Z1 qi3=(sm&8u)?nq3:cq3;Z1 b0=(sm&1u)?cq0:nq0;
Z1 b1=(sm&2u)?cq1:nq1;Z1 b2=(sm&4u)?cq2:nq2;
Z1 b3=(sm&8u)?cq3:nq3;s[12]=qi0*s[0];
s[13]=qi1*s[1];s[14]=qi2*s[2];
s[15]=qi3*s[3];cq0=b0-s[4]*qi0;
cq1=b1-s[5]*qi1;cq2=b2-s[6]*qi2;
cq3=b3-s[7]*qi3;}
}
Z1 q0_n1=cq0*s_last[0];Z1 q1_n1=cq1*s_last[1];
Z1 q2_n1=cq2*s_last[2];Z1 q3_n1=cq3*s_last[3];
Z1 nsq0=q0_n1*q0_n1;Z1 nsq1=q1_n1*q1_n1;
Z1 nsq2=q2_n1*q2_n1;Z1 nsq3=q3_n1*q3_n1;

bool no_gs=(min_g>=0.0001f*max_elem)&&(gr3>=0.0001f*max_elem);if(iter+1u>=max_iter&&no_gs){
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+1u)*Z9];
Z4<Z1>vr2=ub_vecs[(r+2u)*Z9];Z4<Z1>vr3=ub_vecs[(r+((bcnt>3u)?3u:2u))*Z9];
vr0.Z8(n-1u,q0_n1);vr1.Z8(n-1u,q1_n1);
vr3.Z8(n-1u,q3_n1);vr2.Z8(n-1u,q2_n1);
Z1 d_last=dd[n-1u];Z1 rq0=d_last*nsq0,rq1=d_last*nsq1,rq2=d_last*nsq2,rq3=d_last*nsq3;
Z1 q0_n2=0.0f,q1_n2=0.0f,q2_n2=0.0f,q3_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*16u;q0_n2=q0_n1;q1_n2=q1_n1;q2_n2=q2_n1;q3_n2=q3_n1;
q0_n1=s_nm2[12]-s_nm2[8]*q0_n2;q1_n1=s_nm2[13]-s_nm2[9]*q1_n2;
q2_n1=s_nm2[14]-s_nm2[10]*q2_n2;q3_n1=s_nm2[15]-s_nm2[11]*q3_n2;
vr0.Z8(n-2u,q0_n1);vr1.Z8(n-2u,q1_n1);
vr3.Z8(n-2u,q3_n1);vr2.Z8(n-2u,q2_n1);
nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;
nsq2+=q2_n1*q2_n1;nsq3+=q3_n1*q3_n1;
Z1 d_nm2=dd[n-2u],two_e_nm2=2.0f*ee[n-2u];rq0+=q0_n1*(d_nm2*q0_n1+two_e_nm2*q0_n2);
rq1+=q1_n1*(d_nm2*q1_n1+two_e_nm2*q1_n2);rq2+=q2_n1*(d_nm2*q2_n1+two_e_nm2*q2_n2);
rq3+=q3_n1*(d_nm2*q3_n1+two_e_nm2*q3_n2);}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*16u;
Z1 c0_v=(s[12]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[8]*q0_n1;
Z1 c1_v=(s[13]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[9]*q1_n1;
Z1 c2_v=(s[14]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[10]*q2_n1;
Z1 c3_v=(s[15]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[11]*q3_n1;vr0.Z8(ui,c0_v);
vr1.Z8(ui,c1_v);vr3.Z8(ui,c3_v);
vr2.Z8(ui,c2_v);Z1 d_i=dd[ui],two_e=2.0f*ee[ui];
rq0+=c0_v*(d_i*c0_v+two_e*q0_n1);rq1+=c1_v*(d_i*c1_v+two_e*q1_n1);
rq2+=c2_v*(d_i*c2_v+two_e*q2_n1);rq3+=c3_v*(d_i*c3_v+two_e*q3_n1);
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;
nsq2+=c2_v*c2_v;nsq3+=c3_v*c3_v;
}
ub_eig.Z8(c0,rq0/nsq0);ub_eig.Z8(c1,rq1/nsq1);
ub_eig.Z8(c2,rq2/nsq2);if(bcnt>3u)ub_eig.Z8(c3,rq3/nsq3);
v_scale[r+0u]=ZI(nsq0);v_scale[r+1u]=ZI(nsq1);
v_scale[r+2u]=ZI(nsq2);if(bcnt>3u)v_scale[r+3u]=ZI(nsq3);
break;}

s_last[12]=q0_n1;s_last[13]=q1_n1;
s_last[14]=q2_n1;s_last[15]=q3_n1;

Z1 q0_n2=0.0f,q1_n2=0.0f,q2_n2=0.0f,q3_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*16u;q0_n2=q0_n1;q1_n2=q1_n1;q2_n2=q2_n1;q3_n2=q3_n1;
q0_n1=s_nm2[12]-s_nm2[8]*q0_n2;q1_n1=s_nm2[13]-s_nm2[9]*q1_n2;
q2_n1=s_nm2[14]-s_nm2[10]*q2_n2;q3_n1=s_nm2[15]-s_nm2[11]*q3_n2;
s_nm2[12]=q0_n1;s_nm2[13]=q1_n1;s_nm2[14]=q2_n1;s_nm2[15]=q3_n1;nsq0+=q0_n1*q0_n1;
nsq1+=q1_n1*q1_n1;nsq2+=q2_n1*q2_n1;
nsq3+=q3_n1*q3_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*16u;
Z1 c0_v=(s[12]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[8]*q0_n1;
Z1 c1_v=(s[13]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[9]*q1_n1;
Z1 c2_v=(s[14]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[10]*q2_n1;
Z1 c3_v=(s[15]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[11]*q3_n1;

s[12]=c0_v;s[13]=c1_v;s[14]=c2_v;s[15]=c3_v;

q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;

nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;
nsq2+=c2_v*c2_v;nsq3+=c3_v*c3_v;
}
if(iter==0u){
s0_0=nsq0*g0_sq;s0_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
s0_2=(bcnt>2u)?(nsq2*g2_sq):1.0e30f;s0_3=(bcnt>3u)?(nsq3*g3_sq):1.0e30f;
Z1 ms0_01=(s0_0<s0_1)?s0_0:s0_1;Z1 ms0_23=(s0_2<s0_3)?s0_2:s0_3;
if(no_gs&&((ms0_01<ms0_23)?ms0_01:ms0_23)>=250.0f){
max_iter=2u;}
}
bool is_final_iter=(iter+1u>=max_iter);if(iter==1u&&can_early_stop){
Z1 s1_0=nsq0*g0_sq;Z1 s1_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
Z1 s1_2=(bcnt>2u)?(nsq2*g2_sq):1.0e30f;Z1 s1_3=(bcnt>3u)?(nsq3*g3_sq):1.0e30f;
Z1 ms01=(s1_0<s1_1)?s1_0:s1_1;Z1 ms23=(s1_2<s1_3)?s1_2:s1_3;
Z1 min_s1=(ms01<ms23)?ms01:ms23;Z1 p0=s0_0*s1_0;
Z1 p1=(bcnt>1u)?(s0_1*s1_1):1.0e30f;Z1 p2=(bcnt>2u)?(s0_2*s1_2):1.0e30f;
Z1 p3=(bcnt>3u)?(s0_3*s1_3):1.0e30f;Z1 mp01=(p0<p1)?p0:p1;
Z1 mp23=(p2<p3)?p2:p3;Z1 min_p=(mp01<mp23)?mp01:mp23;
if((min_s1>=4.0e4f&&min_p>=1.0e9f)||(min_s1>=8.0e5f&&min_p>=3.5e8f)){
is_final_iter=true;}
}
if(!is_final_iter){
in0=ScalarRsqrtFast(nsq0);in1=ScalarRsqrtFast(nsq1);
in2=ScalarRsqrtFast(nsq2);in3=ScalarRsqrtFast(nsq3);
}else{
in0=ZI(nsq0);in1=ZI(nsq1);
in2=ZI(nsq2);in3=ZI(nsq3);
if(r>0u&&Z7(lam0-ub_eig.ZF(c0-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(aos[i*16u+12u]*in0)*prev_v.ZF(i);nsq0=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=aos[i*16u+12u]*in0-dot*prev_v.ZF(i);aos[i*16u+12u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
if(bcnt>1u&&Z7(lam1-lam0)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*16u+13u]*in1)*(aos[i*16u+12u]*in0);
nsq1=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*16u+13u]*in1-dot*(aos[i*16u+12u]*in0);aos[i*16u+13u]=v;
nsq1+=v*v;}
in1=ZI(nsq1);}
if(bcnt>2u&&Z7(lam2-lam1)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*16u+14u]*in2)*(aos[i*16u+13u]*in1);
nsq2=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*16u+14u]*in2-dot*(aos[i*16u+13u]*in1);aos[i*16u+14u]=v;
nsq2+=v*v;}
in2=ZI(nsq2);}
if(bcnt>3u&&Z7(lam3-lam2)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*16u+15u]*in3)*(aos[i*16u+14u]*in2);
nsq3=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*16u+15u]*in3-dot*(aos[i*16u+14u]*in2);aos[i*16u+15u]=v;
nsq3+=v*v;}
in3=ZI(nsq3);}
Z1 rq0=0.0f,rq1=0.0f,rq2=0.0f,rq3=0.0f;Z1 cur0=aos[12]*in0;
Z1 cur1=aos[13]*in1;Z1 cur2=aos[14]*in2;
Z1 cur3=aos[15]*in3;Z4<Z1>vr0=ub_vecs[r*Z9];
Z4<Z1>vr1=ub_vecs[(r+1u)*Z9];Z4<Z1>vr2=ub_vecs[(r+2u)*Z9];
Z4<Z1>vr3=ub_vecs[(r+((bcnt>3u)?3u:2u))*Z9];vr0.Z8(0,cur0);
vr1.Z8(0,cur1);vr3.Z8(0,cur3);
vr2.Z8(0,cur2);for(Z0 i=0;i+1u<n;++i){
const Z1*sn=aos+(i+1u)*16u;Z1 nxt0=sn[12]*in0;
Z1 nxt1=sn[13]*in1;Z1 nxt2=sn[14]*in2;
Z1 nxt3=sn[15]*in3;vr0.Z8(i+1u,nxt0);
vr1.Z8(i+1u,nxt1);vr3.Z8(i+1u,nxt3);
vr2.Z8(i+1u,nxt2);Z1 d_i=dd[i];
Z1 two_e=2.0f*ee[i];rq0+=cur0*(d_i*cur0+two_e*nxt0);
rq1+=cur1*(d_i*cur1+two_e*nxt1);rq2+=cur2*(d_i*cur2+two_e*nxt2);
rq3+=cur3*(d_i*cur3+two_e*nxt3);cur0=nxt0;cur1=nxt1;cur2=nxt2;cur3=nxt3;
}
Z1 d_last=dd[n-1u];rq0+=d_last*cur0*cur0;
rq1+=d_last*cur1*cur1;rq2+=d_last*cur2*cur2;
rq3+=d_last*cur3*cur3;ub_eig.Z8(c0,rq0);
ub_eig.Z8(c1,rq1);ub_eig.Z8(c2,rq2);
if(bcnt>3u)ub_eig.Z8(c3,rq3);break;
}
}
}

for(;r<ZB;r+=2u){
Z0 bcnt=((ZB-r)<2u)?(ZB-r):2u;Z0 c0=ZJ+r;
if(bcnt==1u){
Z1 lam0=ub_eig.ZF(c0);Z0 sm0=(c0*13u)%23u;
Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));Z1 cq0=nq0;
Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i],abs_m=Z7(m_i),nd0=dd[i+1u]-lam0,e_next=ee[i+1u];bool sw0=(Z7(cur_p0)<abs_m);
swap_mask[i]=sw0?1u:0u;Z1 inv0=sw0?inv_ee[i]:(1.0f/ClampPivot(cur_p0));
Z1 f0=(sw0?cur_p0:m_i)*inv0;Z1 u1_0=sw0?nd0:cur_u0;
Z1*s=aos+i*4u;s[0]=inv0;
s[1]=f0;s[2]=u1_0*inv0;
cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_u0=sw0?(-f0*e_next):e_next;
nq0+=(nq0>=1.1875f)?-0.1875f:0.53125f;Z1 qi0=sw0?nq0:cq0;
Z1 b0=sw0?cq0:nq0;s[3]=qi0*inv0;
cq0=b0-f0*qi0;}
Z1 inv_last=1.0f/ClampPivot(cur_p0);Z1 gl0=(c0>0u)?(lam0-ub_eig.ZF(c0-1u)):max_elem;
Z1 gr0=(c0+1u<n)?(ub_eig.ZF(c0+1u)-lam0):max_elem;Z1 g0=(gl0<gr0)?gl0:gr0;
if(g0<0.0f)g0=0.0f;Z1 g0_sq=g0*g0;
bool can_early_stop=(g0>=0.00025f*max_elem);bool no_gs1=(g0>=0.0001f*max_elem)&&(gr0>=0.0001f*max_elem);
Z1 s0_0=0.0f;Z0 max_iter=3u;
Z1 in0=1.0f;Z4<Z1>vr0=ub_vecs[r*Z9];
for(Z0 iter=0;iter<max_iter;++iter){
if(iter>0u){
cq0=aos[3]*in0;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*4u;
Z1 nq0=s[7]*in0;Z1 qi0=(sm!=0u)?nq0:cq0;
Z1 b0=(sm!=0u)?cq0:nq0;s[3]=qi0*s[0];
cq0=b0-s[1]*qi0;}
}
Z1 q0_n1=cq0*inv_last;Z1 nsq0=q0_n1*q0_n1;
if(iter+1u>=max_iter&&no_gs1){
vr0.Z8(n-1u,q0_n1);Z1 d_last=dd[n-1u];
Z1 rq0=d_last*nsq0;Z1 q0_n2=0.0f;
if(n>1u){
Z1*s_nm2=aos+(n-2u)*4u;q0_n2=q0_n1;
q0_n1=s_nm2[3]-s_nm2[2]*q0_n2;vr0.Z8(n-2u,q0_n1);
nsq0+=q0_n1*q0_n1;rq0+=q0_n1*(dd[n-2u]*q0_n1+(2.0f*ee[n-2u])*q0_n2);
}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1*s=aos+ui*4u;Z1 c0_v=(s[3]-((sm!=0u)?ee_rat[ui]:0.0f)*q0_n2)-s[2]*q0_n1;
vr0.Z8(ui,c0_v);rq0+=c0_v*(dd[ui]*c0_v+(2.0f*ee[ui])*q0_n1);
q0_n2=q0_n1;q0_n1=c0_v;nsq0+=c0_v*c0_v;
}
ub_eig.Z8(c0,rq0/nsq0);v_scale[r]=ZI(nsq0);
break;}
aos[(n-1u)*4u+3u]=q0_n1;Z1 q0_n2=0.0f;
if(n>1u){
Z1*s_nm2=aos+(n-2u)*4u;q0_n2=q0_n1;
q0_n1=s_nm2[3]-s_nm2[2]*q0_n2;s_nm2[3]=q0_n1;
nsq0+=q0_n1*q0_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1*s=aos+ui*4u;Z1 c0_v=(s[3]-((sm!=0u)?ee_rat[ui]:0.0f)*q0_n2)-s[2]*q0_n1;
s[3]=c0_v;q0_n2=q0_n1;q0_n1=c0_v;
nsq0+=c0_v*c0_v;}
if(iter==0u){
s0_0=nsq0*g0_sq;if(no_gs1&&s0_0>=250.0f){
max_iter=2u;}
}
bool is_final_iter=(iter+1u>=max_iter);if(iter==1u&&can_early_stop){
Z1 s1_0=nsq0*g0_sq;Z1 p0=s0_0*s1_0;
if((s1_0>=4.0e4f&&p0>=6.0e8f)||(s1_0>=2.5e6f&&p0>=5.0e7f)){
is_final_iter=true;}
}
if(!is_final_iter){
in0=ScalarRsqrtFast(nsq0);}else{
in0=ZI(nsq0);if(r>0u&&Z7(lam0-ub_eig.ZF(c0-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(aos[i*4u+3u]*in0)*prev_v.ZF(i);nsq0=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=aos[i*4u+3u]*in0-dot*prev_v.ZF(i);aos[i*4u+3u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
Z1 rq0=0.0f;Z1 cur0=aos[3]*in0;
vr0.Z8(0,cur0);for(Z0 i=0;i+1u<n;++i){
Z1 nxt0=aos[(i+1u)*4u+3u]*in0;vr0.Z8(i+1u,nxt0);
rq0+=dd[i]*cur0*cur0+(2.0f*ee[i])*cur0*nxt0;cur0=nxt0;
}
rq0+=dd[n-1u]*cur0*cur0;ub_eig.Z8(c0,rq0);
break;}
}
continue;}
Z0 c1=(bcnt>1u)?(c0+1u):c0;Z1 lam0=ub_eig.ZF(c0);
Z1 lam1=ub_eig.ZF(c1);Z0 sm0=(c0*13u)%23u;
Z0 sm1=sm0+13u;if(sm1>=23u)sm1-=23u;
Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 nq1=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm1));Z1 cq0=nq0,cq1=nq1;
Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];Z1 cur_p1=dd[0]-lam1,cur_u1=ee[0];
for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i],abs_m=Z7(m_i),d_next=dd[i+1u],e_next=ee[i+1u];Z1 nd0=d_next-lam0,nd1=d_next-lam1;
bool sw0=(Z7(cur_p0)<abs_m),sw1=(Z7(cur_p1)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u));swap_mask[i]=sm;
Z1 inv_m=inv_ee[i];Z1 inv0,inv1;
if(sm==3u){
inv0=inv_m;inv1=inv_m;}else if(abs_m>=1e-7f){
Z1 p0=sw0?1.0f:cur_p0;Z1 p1=sw1?1.0f:cur_p1;
Z1 inv01=1.0f/(p0*p1);inv0=sw0?inv_m:(inv01*p1);
inv1=sw1?inv_m:(inv01*p0);}else{
inv0=sw0?inv_m:(1.0f/ClampPivot(cur_p0));inv1=sw1?inv_m:(1.0f/ClampPivot(cur_p1));
}
Z1 f0=(sw0?cur_p0:m_i)*inv0,f1=(sw1?cur_p1:m_i)*inv1;Z1*s=aos+i*8u;
s[0]=inv0;s[1]=inv1;s[2]=f0;s[3]=f1;
Z1 u1_0=sw0?nd0:cur_u0,u1_1=sw1?nd1:cur_u1;s[4]=u1_0*inv0;s[5]=u1_1*inv1;
cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
nq0+=(nq0>=1.1875f)?-0.1875f:0.53125f;nq1+=(nq1>=1.1875f)?-0.1875f:0.53125f;
Z1 qi0=sw0?nq0:cq0,qi1=sw1?nq1:cq1;Z1 b0=sw0?cq0:nq0,b1=sw1?cq1:nq1;
s[6]=qi0*inv0;s[7]=qi1*inv1;cq0=b0-f0*qi0;cq1=b1-f1*qi1;
}
Z1*s_last=aos+(n-1u)*8u;s_last[0]=1.0f/ClampPivot(cur_p0);
s_last[1]=1.0f/ClampPivot(cur_p1);Z1 gl0=(c0>0u)?(lam0-ub_eig.ZF(c0-1u)):max_elem;
Z1 g01=(bcnt>1u)?(lam1-lam0):max_elem;Z1 gr1=(c0+bcnt<n)?(ub_eig.ZF(c0+bcnt)-lam1):max_elem;
Z1 g0=(gl0<g01)?gl0:g01;Z1 g1=(bcnt>1u)?((g01<gr1)?g01:gr1):g0;
if(g0<0.0f)g0=0.0f;if(g1<0.0f)g1=0.0f;
Z1 g0_sq=g0*g0,g1_sq=g1*g1;bool can_early_stop=(((g0<g1)?g0:g1)>=0.00008f*max_elem);
Z1 s0_0=0.0f,s0_1=0.0f;Z0 max_iter=3u;
Z1 in0=1.0f,in1=1.0f;for(Z0 iter=0;iter<max_iter;++iter){
if(iter>0u){
cq0=aos[6]*in0;cq1=aos[7]*in1;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*8u;
Z1 nq0=s[14]*in0,nq1=s[15]*in1;Z1 qi0=(sm&1u)?nq0:cq0,qi1=(sm&2u)?nq1:cq1;
Z1 b0=(sm&1u)?cq0:nq0,b1=(sm&2u)?cq1:nq1;s[6]=qi0*s[0];s[7]=qi1*s[1];
cq0=b0-s[2]*qi0;cq1=b1-s[3]*qi1;}
}
Z1 q0_n1=cq0*s_last[0],q1_n1=cq1*s_last[1];Z1 nsq0=q0_n1*q0_n1,nsq1=q1_n1*q1_n1;
bool no_gs2=(((g0<g1)?g0:g1)>=0.0001f*max_elem)&&(gr1>=0.0001f*max_elem);if(iter+1u>=max_iter&&no_gs2){
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+((bcnt>1u)?1u:0u))*Z9];
vr1.Z8(n-1u,q1_n1);vr0.Z8(n-1u,q0_n1);
Z1 d_last=dd[n-1u];Z1 rq0=d_last*nsq0,rq1=d_last*nsq1;
Z1 q0_n2=0.0f,q1_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*8u;q0_n2=q0_n1;q1_n2=q1_n1;
q0_n1=s_nm2[6]-s_nm2[4]*q0_n2;q1_n1=s_nm2[7]-s_nm2[5]*q1_n2;
vr1.Z8(n-2u,q1_n1);vr0.Z8(n-2u,q0_n1);
nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;Z1 d_nm2=dd[n-2u],two_e_nm2=2.0f*ee[n-2u];
rq0+=q0_n1*(d_nm2*q0_n1+two_e_nm2*q0_n2);rq1+=q1_n1*(d_nm2*q1_n1+two_e_nm2*q1_n2);
}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*8u;
Z1 c0_v=(s[6]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[4]*q0_n1;
Z1 c1_v=(s[7]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[5]*q1_n1;vr1.Z8(ui,c1_v);
vr0.Z8(ui,c0_v);Z1 d_i=dd[ui],two_e=2.0f*ee[ui];
rq0+=c0_v*(d_i*c0_v+two_e*q0_n1);rq1+=c1_v*(d_i*c1_v+two_e*q1_n1);
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;}
ub_eig.Z8(c0,rq0/nsq0);if(bcnt>1u)ub_eig.Z8(c1,rq1/nsq1);
v_scale[r+0u]=ZI(nsq0);if(bcnt>1u)v_scale[r+1u]=ZI(nsq1);
break;}
s_last[6]=q0_n1;s_last[7]=q1_n1;Z1 q0_n2=0.0f,q1_n2=0.0f;
if(n>1u){
Z1*s_nm2=aos+(n-2u)*8u;q0_n2=q0_n1;q1_n2=q1_n1;
q0_n1=s_nm2[6]-s_nm2[4]*q0_n2;q1_n1=s_nm2[7]-s_nm2[5]*q1_n2;
s_nm2[6]=q0_n1;s_nm2[7]=q1_n1;nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;
}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*8u;
Z1 c0_v=(s[6]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[4]*q0_n1;
Z1 c1_v=(s[7]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[5]*q1_n1;s[6]=c0_v;s[7]=c1_v;
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;}
if(iter==0u){
s0_0=nsq0*g0_sq;s0_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
if(no_gs2&&((s0_0<s0_1)?s0_0:s0_1)>=250.0f){
max_iter=2u;}
}
bool is_final_iter=(iter+1u>=max_iter);if(iter==1u&&can_early_stop){
Z1 s1_0=nsq0*g0_sq;Z1 s1_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
Z1 min_s1=(s1_0<s1_1)?s1_0:s1_1;Z1 p0=s0_0*s1_0;
Z1 p1=(bcnt>1u)?(s0_1*s1_1):1.0e30f;Z1 min_p=(p0<p1)?p0:p1;
if((min_s1>=4.0e4f&&min_p>=1.0e9f)||(min_s1>=8.0e5f&&min_p>=3.5e8f)){
is_final_iter=true;}
}
if(!is_final_iter){
in0=ScalarRsqrtFast(nsq0);in1=ScalarRsqrtFast(nsq1);
}else{
in0=ZI(nsq0);in1=ZI(nsq1);
if(r>0u&&Z7(lam0-ub_eig.ZF(c0-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+6u]*in0)*prev_v.ZF(i);nsq0=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+6u]*in0-dot*prev_v.ZF(i);aos[i*8u+6u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
if(bcnt>1u&&Z7(lam1-lam0)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+7u]*in1)*(aos[i*8u+6u]*in0);
nsq1=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+7u]*in1-dot*(aos[i*8u+6u]*in0);aos[i*8u+7u]=v;
nsq1+=v*v;}
in1=ZI(nsq1);}
Z1 rq0=0.0f,rq1=0.0f;Z1 cur0=aos[6]*in0,cur1=aos[7]*in1;
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+((bcnt>1u)?1u:0u))*Z9];
vr1.Z8(0,cur1);vr0.Z8(0,cur0);
for(Z0 i=0;i+1u<n;++i){
const Z1*sn=aos+(i+1u)*8u;Z1 nxt0=sn[6]*in0,nxt1=sn[7]*in1;
vr1.Z8(i+1u,nxt1);vr0.Z8(i+1u,nxt0);
Z1 d_i=dd[i],two_e=2.0f*ee[i];rq0+=d_i*cur0*cur0+two_e*cur0*nxt0;
rq1+=d_i*cur1*cur1+two_e*cur1*nxt1;cur0=nxt0;cur1=nxt1;
}
Z1 d_last=dd[n-1u];rq0+=d_last*cur0*cur0;
rq1+=d_last*cur1*cur1;ub_eig.Z8(c0,rq0);
if(bcnt>1u)ub_eig.Z8(c1,rq1);break;
}
}
}
}

fence<HardEvent::S_V>();for(Z0 ri=0;ri<ZB;++ri){
if(v_scale[ri]!=0.0f){
Muls(ub_vecs[ri*Z9],ub_vecs[ri*Z9],v_scale[ri],n);}
}
}


#define EIGH_HAS_SOLVE_MODE_4 1
#define EIGH_SCALE_CHK() (__builtin_expect((__builtin_fmaxf(__builtin_fmaxf(ap0,ap1),__builtin_fmaxf(ap2,ap3))>1e12f)|(__builtin_fminf(__builtin_fminf(ap0,ap1),__builtin_fminf(ap2,ap3))<1e-12f),0))
#define EIGH_STURM4_STEP2(DA, E2A, DB, E2B) \
{ \
Z1 d_a=(DA),e2_a=(E2A),d_b=(DB),e2_b=(E2B); \
{Z1 eb=e2_b*p0;pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-eb;Z0 sa=(pp0<=0.0f)?1u:0u;c0+=(sa^s0);Z0 sb=(p0<=0.0f)?1u:0u;c0+=(sb^sa);s0=sb;} \
{Z1 eb=e2_b*p1;pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-eb;Z0 sa=(pp1<=0.0f)?1u:0u;c1+=(sa^s1);Z0 sb=(p1<=0.0f)?1u:0u;c1+=(sb^sa);s1=sb;} \
{Z1 eb=e2_b*p2;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-eb;Z0 sa=(pp2<=0.0f)?1u:0u;c2+=(sa^s2);Z0 sb=(p2<=0.0f)?1u:0u;c2+=(sb^sa);s2=sb;} \
{Z1 eb=e2_b*p3;pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-eb;Z0 sa=(pp3<=0.0f)?1u:0u;c3+=(sa^s3);Z0 sb=(p3<=0.0f)?1u:0u;c3+=(sb^sa);s3=sb;} \
}

// Preserve each lane's operation order, but end its step-local lifetimes before
// starting the next lane. Native register scheduling still requires ELF review.
#define EIGH_POLY4_SEQ_STEP2(DA, E2A, DB, E2B) \
{ \
Z1 d_a=(DA),e2_a=(E2A),d_b=(DB),e2_b=(E2B); \
{Z1 eb=e2_b*p0;pp0=(d_a-m0)*p0-e2_a*pp0;p0=(d_b-m0)*pp0-eb;} \
{Z1 eb=e2_b*p1;pp1=(d_a-m1)*p1-e2_a*pp1;p1=(d_b-m1)*pp1-eb;} \
{Z1 eb=e2_b*p2;pp2=(d_a-m2)*p2-e2_a*pp2;p2=(d_b-m2)*pp2-eb;} \
{Z1 eb=e2_b*p3;pp3=(d_a-m3)*p3-e2_a*pp3;p3=(d_b-m3)*pp3-eb;} \
}
#define EIGH_POLY4_STEP2(DA, E2A, DB, E2B) \
{ \
Z1 d_a=(DA),e2_a=(E2A),d_b=(DB),e2_b=(E2B); \
Z1 ta0=d_a-m0,ta1=d_a-m1,ta2=d_a-m2,ta3=d_a-m3; \
Z1 tb0=d_b-m0,tb1=d_b-m1,tb2=d_b-m2,tb3=d_b-m3; \
Z1 eb0=e2_b*p0,eb1=e2_b*p1,eb2=e2_b*p2,eb3=e2_b*p3; \
pp0=ta0*p0-e2_a*pp0;pp1=ta1*p1-e2_a*pp1;pp2=ta2*p2-e2_a*pp2;pp3=ta3*p3-e2_a*pp3; \
p0=tb0*pp0-eb0;p1=tb1*pp1-eb1;p2=tb2*pp2-eb2;p3=tb3*pp3-eb3; \
}

__aicore__ __attribute__((noinline)) void EvalSturm4Fast(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_neg,
Z1*b_pval,
ZG*b_pexp)const{
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;
Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;
Z0 s0=(p0<0.0f)?1u:0u,c0=(256u<<16)|s0;Z0 s1=(p1<0.0f)?1u:0u,c1=(256u<<16)|s1;
Z0 s2=(p2<0.0f)?1u:0u,c2=(256u<<16)|s2;Z0 s3=(p3<0.0f)?1u:0u,c3=(256u<<16)|s3;
Z0 i0=1u;if((n&1u)==0u){
Z1 d_1=de_pairs[0],e2_1=de_pairs[1];Z1 t10=d_1-m0,t11=d_1-m1,t12=d_1-m2,t13=d_1-m3;
Z1 pn0=t10*p0-e2_1*pp0;Z1 pn1=t11*p1-e2_1*pp1;
Z1 pn2=t12*p2-e2_1*pp2;Z1 pn3=t13*p3-e2_1*pp3;
pp0=p0;p0=pn0;pp1=p1;p1=pn1;
pp2=p2;p2=pn2;pp3=p3;p3=pn3;
Z0 sn0=(pn0<=0.0f)?1u:0u;c0+=(sn0^s0);s0=sn0;Z0 sn1=(pn1<=0.0f)?1u:0u;c1+=(sn1^s1);s1=sn1;
Z0 sn2=(pn2<=0.0f)?1u:0u;c2+=(sn2^s2);s2=sn2;Z0 sn3=(pn3<=0.0f)?1u:0u;c3+=(sn3^s3);s3=sn3;
i0=2u;}
#if EIGH_K96_PASS
if constexpr(SOLVE_MODE==3u){
#else
if constexpr(FAST200||SOLVE_MODE==3u){
#endif
const Z1*de=de_pairs+(i0-1u)*2u;for(;i0+12u<=n;i0+=12u,de+=24u){
EIGH_STURM4_STEP2(de[0],de[1],de[2],de[3]);EIGH_STURM4_STEP2(de[4],de[5],de[6],de[7]);
EIGH_STURM4_STEP2(de[8],de[9],de[10],de[11]);EIGH_STURM4_STEP2(de[12],de[13],de[14],de[15]);
EIGH_STURM4_STEP2(de[16],de[17],de[18],de[19]);EIGH_STURM4_STEP2(de[20],de[21],de[22],de[23]);
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if(EIGH_SCALE_CHK()){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;c0+=(1u<<16);}else if(ap0<1e-12f){p0=(p0==0.0f)?((s0!=0u)?-1e-15f:1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;c0-=(1u<<16);}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;c1+=(1u<<16);}else if(ap1<1e-12f){p1=(p1==0.0f)?((s1!=0u)?-1e-15f:1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;c1-=(1u<<16);}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;c2+=(1u<<16);}else if(ap2<1e-12f){p2=(p2==0.0f)?((s2!=0u)?-1e-15f:1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;c2-=(1u<<16);}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;c3+=(1u<<16);}else if(ap3<1e-12f){p3=(p3==0.0f)?((s3!=0u)?-1e-15f:1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;c3-=(1u<<16);}
}
}
for(;i0+1u<n;i0+=2u,de+=4u){
EIGH_STURM4_STEP2(de[0],de[1],de[2],de[3]);}
}else{
const Z1*de=de_pairs+(i0-1u)*2u;for(;i0+12u<=n;i0+=12u,de+=24u){
EIGH_STURM4_STEP2(de[0],de[1],de[2],de[3]);EIGH_STURM4_STEP2(de[4],de[5],de[6],de[7]);
EIGH_STURM4_STEP2(de[8],de[9],de[10],de[11]);EIGH_STURM4_STEP2(de[12],de[13],de[14],de[15]);
EIGH_STURM4_STEP2(de[16],de[17],de[18],de[19]);EIGH_STURM4_STEP2(de[20],de[21],de[22],de[23]);
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if(EIGH_SCALE_CHK()){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;c0+=(1u<<16);}else if(ap0<1e-12f){p0=(p0==0.0f)?((s0!=0u)?-1e-15f:1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;c0-=(1u<<16);}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;c1+=(1u<<16);}else if(ap1<1e-12f){p1=(p1==0.0f)?((s1!=0u)?-1e-15f:1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;c1-=(1u<<16);}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;c2+=(1u<<16);}else if(ap2<1e-12f){p2=(p2==0.0f)?((s2!=0u)?-1e-15f:1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;c2-=(1u<<16);}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;c3+=(1u<<16);}else if(ap3<1e-12f){p3=(p3==0.0f)?((s3!=0u)?-1e-15f:1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;c3-=(1u<<16);}
}
}
if(i0<n){
for(;i0+1u<n;i0+=2u,de+=4u){
EIGH_STURM4_STEP2(de[0],de[1],de[2],de[3]);}
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if(EIGH_SCALE_CHK()){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;c0+=(1u<<16);}else if(ap0<1e-12f){p0=(p0==0.0f)?((s0!=0u)?-1e-15f:1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;c0-=(1u<<16);}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;c1+=(1u<<16);}else if(ap1<1e-12f){p1=(p1==0.0f)?((s1!=0u)?-1e-15f:1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;c1-=(1u<<16);}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;c2+=(1u<<16);}else if(ap2<1e-12f){p2=(p2==0.0f)?((s2!=0u)?-1e-15f:1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;c2-=(1u<<16);}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;c3+=(1u<<16);}else if(ap3<1e-12f){p3=(p3==0.0f)?((s3!=0u)?-1e-15f:1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;c3-=(1u<<16);}
}
}
}
b_neg[0]=c0&0xFFFFu;b_neg[1]=c1&0xFFFFu;b_neg[2]=c2&0xFFFFu;b_neg[3]=c3&0xFFFFu;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);
b_pexp[0]=Z5<ZG>(c0>>16)-256;b_pexp[1]=Z5<ZG>(c1>>16)-256;
b_pexp[2]=Z5<ZG>(c2>>16)-256;b_pexp[3]=Z5<ZG>(c3>>16)-256;}

__aicore__ __attribute__((noinline)) void EvalPoly4Fast(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_sign_neg,
Z1*b_pval,
ZG*b_pexp)const{
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;
Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;
ZG exp0=0,exp1=0,exp2=0,exp3=0;Z0 i0=1u;
if((n&1u)==0u){
Z1 d_1=de_pairs[0],e2_1=de_pairs[1];Z1 pn0=(d_1-m0)*p0-e2_1*pp0;pp0=p0;p0=pn0;
Z1 pn1=(d_1-m1)*p1-e2_1*pp1;pp1=p1;p1=pn1;Z1 pn2=(d_1-m2)*p2-e2_1*pp2;pp2=p2;p2=pn2;
Z1 pn3=(d_1-m3)*p3-e2_1*pp3;pp3=p3;p3=pn3;i0=2u;
}
#if EIGH_K96_PASS
if constexpr(SOLVE_MODE==3u){
#else
if constexpr(FAST200||SOLVE_MODE==3u){
#endif
const Z1*de=de_pairs+(i0-1u)*2u;for(;i0+12u<=n;i0+=12u,de+=24u){
EIGH_POLY4_SEQ_STEP2(de[0],de[1],de[2],de[3]);EIGH_POLY4_SEQ_STEP2(de[4],de[5],de[6],de[7]);
EIGH_POLY4_SEQ_STEP2(de[8],de[9],de[10],de[11]);EIGH_POLY4_SEQ_STEP2(de[12],de[13],de[14],de[15]);
EIGH_POLY4_SEQ_STEP2(de[16],de[17],de[18],de[19]);EIGH_POLY4_SEQ_STEP2(de[20],de[21],de[22],de[23]);
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if(EIGH_SCALE_CHK()){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;++exp0;}else if(ap0<1e-12f){p0=(p0==0.0f)?((pp0<=0.0f)?1e-15f:-1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;--exp0;}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;++exp1;}else if(ap1<1e-12f){p1=(p1==0.0f)?((pp1<=0.0f)?1e-15f:-1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;--exp1;}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;++exp2;}else if(ap2<1e-12f){p2=(p2==0.0f)?((pp2<=0.0f)?1e-15f:-1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;--exp2;}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;++exp3;}else if(ap3<1e-12f){p3=(p3==0.0f)?((pp3<=0.0f)?1e-15f:-1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;--exp3;}
}
}
for(;i0+1u<n;i0+=2u,de+=4u){
EIGH_POLY4_SEQ_STEP2(de[0],de[1],de[2],de[3]);}
}else{
const Z1*de=de_pairs+(i0-1u)*2u;for(;i0+12u<=n;i0+=12u,de+=24u){
EIGH_POLY4_SEQ_STEP2(de[0],de[1],de[2],de[3]);EIGH_POLY4_SEQ_STEP2(de[4],de[5],de[6],de[7]);
EIGH_POLY4_SEQ_STEP2(de[8],de[9],de[10],de[11]);EIGH_POLY4_SEQ_STEP2(de[12],de[13],de[14],de[15]);
EIGH_POLY4_SEQ_STEP2(de[16],de[17],de[18],de[19]);EIGH_POLY4_SEQ_STEP2(de[20],de[21],de[22],de[23]);
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if(EIGH_SCALE_CHK()){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;++exp0;}else if(ap0<1e-12f){p0=(p0==0.0f)?((pp0<=0.0f)?1e-15f:-1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;--exp0;}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;++exp1;}else if(ap1<1e-12f){p1=(p1==0.0f)?((pp1<=0.0f)?1e-15f:-1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;--exp1;}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;++exp2;}else if(ap2<1e-12f){p2=(p2==0.0f)?((pp2<=0.0f)?1e-15f:-1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;--exp2;}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;++exp3;}else if(ap3<1e-12f){p3=(p3==0.0f)?((pp3<=0.0f)?1e-15f:-1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;--exp3;}
}
}
if(i0<n){
for(;i0+1u<n;i0+=2u,de+=4u){
EIGH_POLY4_SEQ_STEP2(de[0],de[1],de[2],de[3]);}
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if(EIGH_SCALE_CHK()){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;++exp0;}else if(ap0<1e-12f){p0=(p0==0.0f)?((pp0<=0.0f)?1e-15f:-1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;--exp0;}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;++exp1;}else if(ap1<1e-12f){p1=(p1==0.0f)?((pp1<=0.0f)?1e-15f:-1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;--exp1;}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;++exp2;}else if(ap2<1e-12f){p2=(p2==0.0f)?((pp2<=0.0f)?1e-15f:-1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;--exp2;}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;++exp3;}else if(ap3<1e-12f){p3=(p3==0.0f)?((pp3<=0.0f)?1e-15f:-1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;--exp3;}
}
}
}
b_sign_neg[0]=(p0<=0.0f)?1u:0u;b_sign_neg[1]=(p1<=0.0f)?1u:0u;
b_sign_neg[2]=(p2<=0.0f)?1u:0u;b_sign_neg[3]=(p3<=0.0f)?1u:0u;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);
b_pexp[0]=exp0;b_pexp[1]=exp1;b_pexp[2]=exp2;b_pexp[3]=exp3;}

__aicore__ __attribute__((noinline)) void EvalPoly4SameScaleAs8Fast(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_sign_neg,
Z1*b_pval,
ZG*b_pexp)const{
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;
Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;
ZG exp0=0,exp1=0,exp2=0,exp3=0;Z0 i0=1u;
if((n&1u)==0u){
Z1 d_1=de_pairs[0],e2_1=de_pairs[1];Z1 pn0=(d_1-m0)*p0-e2_1*pp0;pp0=p0;p0=pn0;
Z1 pn1=(d_1-m1)*p1-e2_1*pp1;pp1=p1;p1=pn1;Z1 pn2=(d_1-m2)*p2-e2_1*pp2;pp2=p2;p2=pn2;
Z1 pn3=(d_1-m3)*p3-e2_1*pp3;pp3=p3;p3=pn3;i0=2u;
}
const Z1*de=de_pairs+(i0-1u)*2u;for(;i0+12u<=n;i0+=12u,de+=24u){
EIGH_POLY4_SEQ_STEP2(de[0],de[1],de[2],de[3]);EIGH_POLY4_SEQ_STEP2(de[4],de[5],de[6],de[7]);
EIGH_POLY4_SEQ_STEP2(de[8],de[9],de[10],de[11]);EIGH_POLY4_SEQ_STEP2(de[12],de[13],de[14],de[15]);
EIGH_POLY4_SEQ_STEP2(de[16],de[17],de[18],de[19]);EIGH_POLY4_SEQ_STEP2(de[20],de[21],de[22],de[23]);
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
if(EIGH_SCALE_CHK()){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;++exp0;}else if(ap0<1e-12f){p0=(p0==0.0f)?((pp0<=0.0f)?1e-15f:-1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;--exp0;}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;++exp1;}else if(ap1<1e-12f){p1=(p1==0.0f)?((pp1<=0.0f)?1e-15f:-1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;--exp1;}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;++exp2;}else if(ap2<1e-12f){p2=(p2==0.0f)?((pp2<=0.0f)?1e-15f:-1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;--exp2;}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;++exp3;}else if(ap3<1e-12f){p3=(p3==0.0f)?((pp3<=0.0f)?1e-15f:-1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;--exp3;}
}
}
for(;i0+1u<n;i0+=2u,de+=4u){
EIGH_POLY4_SEQ_STEP2(de[0],de[1],de[2],de[3]);}
b_sign_neg[0]=(p0<=0.0f)?1u:0u;b_sign_neg[1]=(p1<=0.0f)?1u:0u;
b_sign_neg[2]=(p2<=0.0f)?1u:0u;b_sign_neg[3]=(p3<=0.0f)?1u:0u;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);
b_pexp[0]=exp0;b_pexp[1]=exp1;b_pexp[2]=exp2;b_pexp[3]=exp3;}



__aicore__ __attribute__((noinline)) void EvalSturm4Fast256(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_neg,
Z1*b_pval,
ZG*b_pexp)const{
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;
Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;
Z0 s0=(p0<0.0f)?1u:0u,c0=(256u<<16)|s0;Z0 s1=(p1<0.0f)?1u:0u,c1=(256u<<16)|s1;
Z0 s2=(p2<0.0f)?1u:0u,c2=(256u<<16)|s2;Z0 s3=(p3<0.0f)?1u:0u,c3=(256u<<16)|s3;
{
Z1 d_1=de_pairs[0],e2_1=de_pairs[1];
Z1 pn0=(d_1-m0)*p0-e2_1*pp0;pp0=p0;p0=pn0;Z0 sn0=(pn0<=0.0f)?1u:0u;c0+=(sn0^s0);s0=sn0;
Z1 pn1=(d_1-m1)*p1-e2_1*pp1;pp1=p1;p1=pn1;Z0 sn1=(pn1<=0.0f)?1u:0u;c1+=(sn1^s1);s1=sn1;
Z1 pn2=(d_1-m2)*p2-e2_1*pp2;pp2=p2;p2=pn2;Z0 sn2=(pn2<=0.0f)?1u:0u;c2+=(sn2^s2);s2=sn2;
Z1 pn3=(d_1-m3)*p3-e2_1*pp3;pp3=p3;p3=pn3;Z0 sn3=(pn3<=0.0f)?1u:0u;c3+=(sn3^s3);s3=sn3;}
const Z1*de=de_pairs+2u;for(Z0 i0=2u;i0<254u;i0+=12u,de+=24u){
EIGH_STURM4_STEP2(de[0],de[1],de[2],de[3]);EIGH_STURM4_STEP2(de[4],de[5],de[6],de[7]);
EIGH_STURM4_STEP2(de[8],de[9],de[10],de[11]);EIGH_STURM4_STEP2(de[12],de[13],de[14],de[15]);
EIGH_STURM4_STEP2(de[16],de[17],de[18],de[19]);EIGH_STURM4_STEP2(de[20],de[21],de[22],de[23]);
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
Z1 mx=__builtin_fmaxf(__builtin_fmaxf(ap0,ap1),__builtin_fmaxf(ap2,ap3)),mn=__builtin_fminf(__builtin_fminf(ap0,ap1),__builtin_fminf(ap2,ap3));if(__builtin_expect((mx>1e12f)|(mn<1e-12f),0)){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;c0+=(1u<<16);}else if(ap0<1e-12f){p0=(p0==0.0f)?((s0!=0u)?-1e-15f:1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;c0-=(1u<<16);}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;c1+=(1u<<16);}else if(ap1<1e-12f){p1=(p1==0.0f)?((s1!=0u)?-1e-15f:1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;c1-=(1u<<16);}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;c2+=(1u<<16);}else if(ap2<1e-12f){p2=(p2==0.0f)?((s2!=0u)?-1e-15f:1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;c2-=(1u<<16);}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;c3+=(1u<<16);}else if(ap3<1e-12f){p3=(p3==0.0f)?((s3!=0u)?-1e-15f:1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;c3-=(1u<<16);}
}
}
EIGH_STURM4_STEP2(de[0],de[1],de[2],de[3]);
b_neg[0]=c0&0xFFFFu;b_neg[1]=c1&0xFFFFu;b_neg[2]=c2&0xFFFFu;b_neg[3]=c3&0xFFFFu;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);
b_pexp[0]=Z5<ZG>(c0>>16)-256;b_pexp[1]=Z5<ZG>(c1>>16)-256;
b_pexp[2]=Z5<ZG>(c2>>16)-256;b_pexp[3]=Z5<ZG>(c3>>16)-256;}

__aicore__ __attribute__((noinline)) void EvalPoly4Sequential256(
Z1 d0,
const Z1*de_pairs,
const Z1*b_mid,
Z0*b_sign_neg,
Z1*b_pval,
ZG*b_pexp)const{
Z1 m0=b_mid[0],m1=b_mid[1],m2=b_mid[2],m3=b_mid[3];Z1 pp0=1.0f,pp1=1.0f,pp2=1.0f,pp3=1.0f;
Z1 p0=d0-m0;if(Z7(p0)<1e-20f)p0=-1e-20f;Z1 p1=d0-m1;if(Z7(p1)<1e-20f)p1=-1e-20f;
Z1 p2=d0-m2;if(Z7(p2)<1e-20f)p2=-1e-20f;Z1 p3=d0-m3;if(Z7(p3)<1e-20f)p3=-1e-20f;
ZG exp0=0,exp1=0,exp2=0,exp3=0;{
Z1 d_1=de_pairs[0],e2_1=de_pairs[1];Z1 pn0=(d_1-m0)*p0-e2_1*pp0;pp0=p0;p0=pn0;
Z1 pn1=(d_1-m1)*p1-e2_1*pp1;pp1=p1;p1=pn1;Z1 pn2=(d_1-m2)*p2-e2_1*pp2;pp2=p2;p2=pn2;
Z1 pn3=(d_1-m3)*p3-e2_1*pp3;pp3=p3;p3=pn3;}
const Z1*de=de_pairs+2u;for(Z0 i0=2u;i0<254u;i0+=12u,de+=24u){
EIGH_POLY4_SEQ_STEP2(de[0],de[1],de[2],de[3]);EIGH_POLY4_SEQ_STEP2(de[4],de[5],de[6],de[7]);
EIGH_POLY4_SEQ_STEP2(de[8],de[9],de[10],de[11]);EIGH_POLY4_SEQ_STEP2(de[12],de[13],de[14],de[15]);
EIGH_POLY4_SEQ_STEP2(de[16],de[17],de[18],de[19]);EIGH_POLY4_SEQ_STEP2(de[20],de[21],de[22],de[23]);
Z1 ap0=Z7(p0),ap1=Z7(p1),ap2=Z7(p2),ap3=Z7(p3);
Z1 mx=__builtin_fmaxf(__builtin_fmaxf(ap0,ap1),__builtin_fmaxf(ap2,ap3)),mn=__builtin_fminf(__builtin_fminf(ap0,ap1),__builtin_fminf(ap2,ap3));if(__builtin_expect((mx>1e12f)|(mn<1e-12f),0)){
if(ap0>1e12f){p0*=5.4210109e-20f;pp0*=5.4210109e-20f;++exp0;}else if(ap0<1e-12f){p0=(p0==0.0f)?((pp0<=0.0f)?1e-15f:-1e-15f):(p0*1.8446744e19f);pp0*=1.8446744e19f;--exp0;}
if(ap1>1e12f){p1*=5.4210109e-20f;pp1*=5.4210109e-20f;++exp1;}else if(ap1<1e-12f){p1=(p1==0.0f)?((pp1<=0.0f)?1e-15f:-1e-15f):(p1*1.8446744e19f);pp1*=1.8446744e19f;--exp1;}
if(ap2>1e12f){p2*=5.4210109e-20f;pp2*=5.4210109e-20f;++exp2;}else if(ap2<1e-12f){p2=(p2==0.0f)?((pp2<=0.0f)?1e-15f:-1e-15f):(p2*1.8446744e19f);pp2*=1.8446744e19f;--exp2;}
if(ap3>1e12f){p3*=5.4210109e-20f;pp3*=5.4210109e-20f;++exp3;}else if(ap3<1e-12f){p3=(p3==0.0f)?((pp3<=0.0f)?1e-15f:-1e-15f):(p3*1.8446744e19f);pp3*=1.8446744e19f;--exp3;}
}
}
EIGH_POLY4_SEQ_STEP2(de[0],de[1],de[2],de[3]);
b_sign_neg[0]=(p0<=0.0f)?1u:0u;b_sign_neg[1]=(p1<=0.0f)?1u:0u;
b_sign_neg[2]=(p2<=0.0f)?1u:0u;b_sign_neg[3]=(p3<=0.0f)?1u:0u;
b_pval[0]=Z7(p0);b_pval[1]=Z7(p1);b_pval[2]=Z7(p2);b_pval[3]=Z7(p3);
b_pexp[0]=exp0;b_pexp[1]=exp1;b_pexp[2]=exp2;b_pexp[3]=exp3;}

#undef EIGH_POLY4_SEQ_STEP2

#undef EIGH_STURM4_STEP2
#undef EIGH_POLY4_STEP2
#undef EIGH_SCALE_CHK

#if EIGH_K96_PASS
#else
__aicore__ inline void StartEvaluateFinalPacket200(Z0 r,Z1*v_scale){
Z4<Z1>d_copy=ZH[22144u];Z4<Z1>e_copy=ZH[22344u];
Z4<Z0>shift_idx=ZH[22544u].ReinterpretCast<Z0>();
Z4<Z1>sq=ZH[22744u];Z4<Z1>off=ZH[23544u];
Z4<Z1>next=ZH[24344u];Z4<Z1>terms=ZH[25144u];
Z4<Z1>red=ZH[25944u];fence<HardEvent::S_V>();
Mul(sq,ub_vecs[r*200u],ub_vecs[r*200u],800u);for(Z0 j=0u;j<4u;++j){
Gather(next[j*200u],ub_vecs[(r+j)*200u],shift_idx,0u,200u);}
Z3<ZC>();Mul(off,next,ub_vecs[r*200u],800u);
for(Z0 j=0u;j<4u;++j){
Mul(terms[j*200u],sq[j*200u],d_copy,200u);}
Z3<ZC>();for(Z0 c=0u;c<200u;c+=64u){
Z0 w=(200u-c<64u)?(200u-c):64u;
MulAddDst(terms[c],off[c],e_copy[c],w,4u,Z6{1,1,1,25,25,0});}
Z3<ZC>();for(Z0 s=0u;s<3u;++s){
WholeReduceSum(red[s*8u],sq[s*64u],64u,4u,1u,1u,25u);
WholeReduceSum(red[32u+s*8u],terms[s*64u],64u,4u,1u,1u,25u);}
WholeReduceSum(red[24u],sq[192u],8u,4u,1u,1u,25u);WholeReduceSum(red[56u],terms[192u],8u,4u,1u,1u,25u);
Z3<ZC>();Add(red,red,red[8u],4u);Add(red[16u],red[16u],red[24u],4u);
Add(red[32u],red[32u],red[40u],4u);Add(red[48u],red[48u],red[56u],4u);Z3<ZC>();
Add(red,red,red[16u],4u);Add(red[32u],red[32u],red[48u],4u);if(r>=4u){
for(Z0 ri=r-4u;ri<r;++ri){
if(v_scale[ri]!=0.0f){
Muls(ub_vecs[ri*200u],ub_vecs[ri*200u],v_scale[ri],200u);v_scale[ri]=0.0f;
}
}
}
}

__aicore__ inline void FinishEvaluateFinalPacket200(Z0 r,Z0 bcnt,Z0 c0,Z0 c1,Z0 c2,Z0 c3,Z1*v_scale){
Z4<Z1>red=ZH[25944u];fence<HardEvent::V_S>();
Z0 cols[4]={c0,c1,c2,c3};for(Z0 j=0u;j<bcnt;++j){
Z1 norm=red.ZF(j);ub_eig.Z8(cols[j],red.ZF(32u+j)/norm);
v_scale[r+j]=ZI(norm);}
}

__aicore__ __attribute__((noinline))void ComputeInverseVectorsLarge200Fast(Z1*work,ZA*swap_mask){
Z1*dd=work;Z1*ee=work+((n<=256u)?256u:512u);
bool is_diag=true;Z1 max_elem=0.0f;

Z1 saved_max=work[1023];is_diag=(saved_max<0.0f);
max_elem=is_diag?0.0f:saved_max;

if(ZB>0u&&(is_diag||n<Z9)){
Duplicate(ub_vecs,0.0f,ZB*Z9);Z3<PIPE_ALL>();
}

if(is_diag){
if constexpr(SOLVE_MODE!=2u&&SOLVE_MODE!=3u){
for(Z0 i=0;i<n;++i){
dd[i]=ub_diag.ZF(i);}
}
for(Z0 r=0;r<ZB;++r){
Z0 col_idx=ZJ+r;Z4<Z1>vec_r=ub_vecs[r*Z9];
for(Z0 i=0;i<n;++i){
Z0 rank=0u;Z1 di=dd[i];
for(Z0 k=0;k<n;++k){
if(dd[k]<di||(dd[k]==di&&k<i)){
++rank;}
}
if(rank==col_idx){
vec_r.Z8(i,1.0f);break;
}
}
}
Z3<PIPE_ALL>();return;
}

Z4<Z1>d_copy_init=ZH[22144u];Z4<Z1>e_copy_init=ZH[22344u];
Z4<Z0>shift_idx_init=ZH[22544u].ReinterpretCast<Z0>();
for(Z0 i=0u;i<199u;++i){
d_copy_init.Z8(i,dd[i]);e_copy_init.Z8(i,2.0f*ee[i]);
shift_idx_init.Z8(i,(i+1u)*4u);}
d_copy_init.Z8(199u,dd[199]);e_copy_init.Z8(199u,0.0f);
shift_idx_init.Z8(199u,0u);bool pend_post=false;
Z0 pend_r=0u,pend_bcnt=0u,pend_c0=0u,pend_c1=0u,pend_c2=0u,pend_c3=0u;Z1 v_scale[32];
for(Z0 ri=0;ri<ZB;++ri)v_scale[ri]=0.0f;Z0 r=0u;
if(n<=256u){
Z1*ee_rat=work+480;Z1*inv_ee=work+688;
bool all_e_nz=true;Z0 i_ee=0u;for(;i_ee+4u<n;i_ee+=4u){
Z1 e0=ee[i_ee],e1=ee[i_ee+1u],e2=ee[i_ee+2u],e3=ee[i_ee+3u],e4=ee[i_ee+4u];
if(Z7(e0)>=1e-7f&&Z7(e1)>=1e-7f&&Z7(e2)>=1e-7f&&Z7(e3)>=1e-7f){
Z1 p01=e0*e1,p23=e2*e3;Z1 inv_all=1.0f/(p01*p23);
Z1 inv01=inv_all*p23,inv23=inv_all*p01;Z1 iv0=inv01*e1,iv1=inv01*e0,iv2=inv23*e3,iv3=inv23*e2;
inv_ee[i_ee]=iv0;inv_ee[i_ee+1u]=iv1;inv_ee[i_ee+2u]=iv2;inv_ee[i_ee+3u]=iv3;
ee_rat[i_ee]=e1*iv0;ee_rat[i_ee+1u]=e2*iv1;ee_rat[i_ee+2u]=e3*iv2;ee_rat[i_ee+3u]=e4*iv3;}else{
all_e_nz=false;for(Z0 k=0;k<4u;++k){
Z1 ek=ee[i_ee+k];Z1 ivk=(ek!=0.0f)?(1.0f/ek):0.0f;
inv_ee[i_ee+k]=ivk;ee_rat[i_ee+k]=ee[i_ee+k+1u]*ivk;
}
}
}
for(;i_ee+1u<n;++i_ee){
Z1 ei=ee[i_ee];if(Z7(ei)<1e-7f)all_e_nz=false;
Z1 inv_e=(ei!=0.0f)?(1.0f/ei):0.0f;
inv_ee[i_ee]=inv_e;ee_rat[i_ee]=ee[i_ee+1u]*inv_e;
}


Z1*aos=work+896;for(;r+2u<ZB;r+=4u){
Z0 bcnt=(ZB-r<4u)?(ZB-r):4u;Z0 c0=ZJ+r;
Z0 c1=(bcnt>1u)?(c0+1u):c0;Z0 c2=(bcnt>2u)?(c0+2u):c1;
Z0 c3=(bcnt>3u)?(c0+3u):c2;Z1 lam0=ub_eig.ZF(c0);
Z1 lam1=ub_eig.ZF(c1);Z1 lam2=ub_eig.ZF(c2);
Z1 lam3=ub_eig.ZF(c3);
Z1 gl0=(c0>0u)?(lam0-ub_eig.ZF(c0-1u)):max_elem;
Z1 gr3=(c0+bcnt<n)?(ub_eig.ZF(c0+bcnt)-lam3):max_elem;Z1 g01=(bcnt>1u)?(lam1-lam0):max_elem;
Z1 g12=(bcnt>2u)?(lam2-lam1):max_elem;Z1 g23=(bcnt>3u)?(lam3-lam2):gr3;
Z1 g0=(gl0<g01)?gl0:g01;Z1 g1=(g01<g12)?g01:g12;
Z1 g2=(g12<g23)?g12:g23;Z1 g3=(bcnt>3u)?((g23<gr3)?g23:gr3):max_elem;
if(g0<0.0f)g0=0.0f;if(g1<0.0f)g1=0.0f;
if(g2<0.0f)g2=0.0f;if(g3<0.0f)g3=0.0f;
Z1 min_g=((g0<g1)?g0:g1)<((g2<g3)?g2:g3)?((g0<g1)?g0:g1):((g2<g3)?g2:g3);
bool no_gs=(min_g>=0.0001f*max_elem)&&(gr3>=0.0001f*max_elem);
if(n==200u&&bcnt>=3u&&min_g>=0.00025f*max_elem&&gr3>=0.00025f*max_elem&&all_e_nz){
Z1 cq0=1.375f,cq1=1.375f,cq2=1.375f,cq3=1.375f;
Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];Z1 cur_p1=dd[0]-lam1,cur_u1=ee[0];
Z1 cur_p2=dd[0]-lam2,cur_u2=ee[0];Z1 cur_p3=dd[0]-lam3,cur_u3=ee[0];
Z1*s=aos;for(Z0 i=0u;i<199u;++i,s+=8u){
Z1 m_i=ee[i],abs_m=Z7(m_i),d_next=dd[i+1u],e_next=ee[i+1u];
Z1 nd0=d_next-lam0,nd1=d_next-lam1,nd2=d_next-lam2,nd3=d_next-lam3;
bool sw0=(Z7(cur_p0)<abs_m),sw1=(Z7(cur_p1)<abs_m),sw2=(Z7(cur_p2)<abs_m),sw3=(Z7(cur_p3)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u)|(sw2?4u:0u)|(sw3?8u:0u));swap_mask[i]=sm;
if(sm==15u){
Z1 inv_m=inv_ee[i],pim=1.375f*inv_m,nim=-pim;
Z1 f0=cur_p0*inv_m,f1=cur_p1*inv_m,f2=cur_p2*inv_m,f3=cur_p3*inv_m;
s[0]=nd0*inv_m;s[1]=nd1*inv_m;s[2]=nd2*inv_m;s[3]=nd3*inv_m;
cur_p0=cur_u0-f0*nd0;cur_p1=cur_u1-f1*nd1;cur_p2=cur_u2-f2*nd2;cur_p3=cur_u3-f3*nd3;
cur_u0=-f0*e_next;cur_u1=-f1*e_next;cur_u2=-f2*e_next;cur_u3=-f3*e_next;
bool p0=(f0*cq0>0.0f),p1=(f1*cq1>0.0f),p2=(f2*cq2>0.0f),p3=(f3*cq3>0.0f);
s[4]=p0?nim:pim;s[5]=p1?nim:pim;s[6]=p2?nim:pim;s[7]=p3?nim:pim;
cq0=cq0-f0*(p0?-1.375f:1.375f);cq1=cq1-f1*(p1?-1.375f:1.375f);cq2=cq2-f2*(p2?-1.375f:1.375f);cq3=cq3-f3*(p3?-1.375f:1.375f);
}else if(sm==0u){
Z1 p01=cur_p0*cur_p1,p23=cur_p2*cur_p3;Z1 inv_all=1.0f/(p01*p23);
Z1 inv01=inv_all*p23,inv23=inv_all*p01;
Z1 inv0=inv01*cur_p1,inv1=inv01*cur_p0,inv2=inv23*cur_p3,inv3=inv23*cur_p2;
Z1 f0=m_i*inv0,f1=m_i*inv1,f2=m_i*inv2,f3=m_i*inv3;
s[0]=cur_u0*inv0;s[1]=cur_u1*inv1;s[2]=cur_u2*inv2;s[3]=cur_u3*inv3;
cur_p0=nd0-f0*cur_u0;cur_p1=nd1-f1*cur_u1;cur_p2=nd2-f2*cur_u2;cur_p3=nd3-f3*cur_u3;
cur_u0=e_next;cur_u1=e_next;cur_u2=e_next;cur_u3=e_next;
Z1 sn0=(f0*cq0>0.0f)?-1.375f:1.375f,sn1=(f1*cq1>0.0f)?-1.375f:1.375f,sn2=(f2*cq2>0.0f)?-1.375f:1.375f,sn3=(f3*cq3>0.0f)?-1.375f:1.375f;
s[4]=cq0*inv0;s[5]=cq1*inv1;s[6]=cq2*inv2;s[7]=cq3*inv3;
cq0=sn0-f0*cq0;cq1=sn1-f1*cq1;cq2=sn2-f2*cq2;cq3=sn3-f3*cq3;}else{
Z1 inv_m=inv_ee[i];Z1 p0=sw0?1.0f:cur_p0,p1=sw1?1.0f:cur_p1,p2=sw2?1.0f:cur_p2,p3=sw3?1.0f:cur_p3;
Z1 p01=p0*p1,p23=p2*p3;Z1 inv_all=1.0f/(p01*p23);
Z1 inv01=inv_all*p23,inv23=inv_all*p01;
Z1 inv0=sw0?inv_m:(inv01*p1),inv1=sw1?inv_m:(inv01*p0),inv2=sw2?inv_m:(inv23*p3),inv3=sw3?inv_m:(inv23*p2);
Z1 f0=(sw0?cur_p0:m_i)*inv0,f1=(sw1?cur_p1:m_i)*inv1,f2=(sw2?cur_p2:m_i)*inv2,f3=(sw3?cur_p3:m_i)*inv3;
Z1 u1_0=sw0?nd0:cur_u0,u1_1=sw1?nd1:cur_u1,u1_2=sw2?nd2:cur_u2,u1_3=sw3?nd3:cur_u3;
s[0]=u1_0*inv0;s[1]=u1_1*inv1;s[2]=u1_2*inv2;s[3]=u1_3*inv3;
cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;cur_p2=(sw2?cur_u2:nd2)-f2*u1_2;cur_p3=(sw3?cur_u3:nd3)-f3*u1_3;
cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;cur_u2=sw2?(-f2*e_next):e_next;cur_u3=sw3?(-f3*e_next):e_next;
Z1 sn0=(f0*cq0>0.0f)?-1.375f:1.375f,sn1=(f1*cq1>0.0f)?-1.375f:1.375f,sn2=(f2*cq2>0.0f)?-1.375f:1.375f,sn3=(f3*cq3>0.0f)?-1.375f:1.375f;
Z1 qi0=sw0?sn0:cq0,qi1=sw1?sn1:cq1,qi2=sw2?sn2:cq2,qi3=sw3?sn3:cq3;
Z1 b0=sw0?cq0:sn0,b1=sw1?cq1:sn1,b2=sw2?cq2:sn2,b3=sw3?cq3:sn3;
s[4]=qi0*inv0;s[5]=qi1*inv1;s[6]=qi2*inv2;s[7]=qi3*inv3;
cq0=b0-f0*qi0;cq1=b1-f1*qi1;cq2=b2-f2*qi2;cq3=b3-f3*qi3;}
}
Z1 lden0=ClampPivot(cur_p0),lden1=ClampPivot(cur_p1),lden2=ClampPivot(cur_p2),lden3=ClampPivot(cur_p3);
Z1 sl0,sl1,sl2,sl3;
if(Z7(lden0)>=1e-7f&&Z7(lden1)>=1e-7f&&Z7(lden2)>=1e-7f&&Z7(lden3)>=1e-7f){
Z1 lp01=lden0*lden1,lp23=lden2*lden3;Z1 linv_all=1.0f/(lp01*lp23);
Z1 linv01=linv_all*lp23,linv23=linv_all*lp01;
sl0=linv01*lden1;sl1=linv01*lden0;sl2=linv23*lden3;sl3=linv23*lden2;}else{
sl0=1.0f/lden0;sl1=1.0f/lden1;sl2=1.0f/lden2;sl3=1.0f/lden3;}
if(pend_post){
FinishEvaluateFinalPacket200(pend_r,pend_bcnt,pend_c0,pend_c1,pend_c2,pend_c3,v_scale);pend_post=false;
}
Z4<Z1>vr0=ub_vecs[r*200u];Z4<Z1>vr1=ub_vecs[(r+1u)*200u];
Z4<Z1>vr2=ub_vecs[(r+2u)*200u];Z4<Z1>vr3=ub_vecs[(r+3u)*200u];
Z1 q0_n1=cq0*sl0,q1_n1=cq1*sl1,q2_n1=cq2*sl2,q3_n1=cq3*sl3;
vr0.Z8(199u,q0_n1);vr1.Z8(199u,q1_n1);vr3.Z8(199u,q3_n1);vr2.Z8(199u,q2_n1);
const Z1*s_nm2=aos+198u*8u;Z1 q0_n2=q0_n1,q1_n2=q1_n1,q2_n2=q2_n1,q3_n2=q3_n1;
q0_n1=s_nm2[4]-s_nm2[0]*q0_n2;q1_n1=s_nm2[5]-s_nm2[1]*q1_n2;
q2_n1=s_nm2[6]-s_nm2[2]*q2_n2;q3_n1=s_nm2[7]-s_nm2[3]*q3_n2;
vr0.Z8(198u,q0_n1);vr1.Z8(198u,q1_n1);vr3.Z8(198u,q3_n1);vr2.Z8(198u,q2_n1);
const Z1*sb=aos+197u*8u;for(ZG i=197;i>=0;--i,sb-=8u){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
const Z1*s=sb;Z1 u2_s=ee_rat[ui];
Z1 c0_v=(s[4]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[0]*q0_n1;
Z1 c1_v=(s[5]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[1]*q1_n1;
Z1 c2_v=(s[6]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[2]*q2_n1;
Z1 c3_v=(s[7]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[3]*q3_n1;
vr0.Z8(ui,c0_v);vr1.Z8(ui,c1_v);vr3.Z8(ui,c3_v);vr2.Z8(ui,c2_v);
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;}
StartEvaluateFinalPacket200(r,v_scale);pend_post=true;
pend_r=r;pend_bcnt=bcnt;pend_c0=c0;pend_c1=c1;pend_c2=c2;pend_c3=c3;continue;
}


Z0 sm0=(c0*13u)%23u;Z0 sm1=sm0+13u;if(sm1>=23u)sm1-=23u;
Z0 sm2=sm1+13u;if(sm2>=23u)sm2-=23u;Z0 sm3=sm2+13u;if(sm3>=23u)sm3-=23u;
Z1 nq0=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm0));
Z1 nq1=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm1));
Z1 nq2=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm2));
Z1 nq3=1.0f+0.03125f*Z5<Z1>(Z5<ZG>(sm3));
Z1 cq0=nq0,cq1=nq1,cq2=nq2,cq3=nq3;

Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];Z1 cur_p1=dd[0]-lam1,cur_u1=ee[0];
Z1 cur_p2=dd[0]-lam2,cur_u2=ee[0];Z1 cur_p3=dd[0]-lam3,cur_u3=ee[0];

for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i];Z1 abs_m=Z7(m_i);
Z1 d_next=dd[i+1u];Z1 e_next=ee[i+1u];
Z1 nd0=d_next-lam0;Z1 nd1=d_next-lam1;
Z1 nd2=d_next-lam2;Z1 nd3=d_next-lam3;

bool sw0=(Z7(cur_p0)<abs_m);bool sw1=(Z7(cur_p1)<abs_m);
bool sw2=(Z7(cur_p2)<abs_m);bool sw3=(Z7(cur_p3)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u)|(sw2?4u:0u)|(sw3?8u:0u));swap_mask[i]=sm;

Z1 inv_m=inv_ee[i];Z1 inv0,inv1,inv2,inv3;
if(sm==15u){
inv0=inv_m;inv1=inv_m;inv2=inv_m;inv3=inv_m;}else if(abs_m>=1e-7f){
Z1 p0=sw0?1.0f:cur_p0;Z1 p1=sw1?1.0f:cur_p1;
Z1 p2=sw2?1.0f:cur_p2;Z1 p3=sw3?1.0f:cur_p3;
Z1 p01=p0*p1;Z1 p23=p2*p3;
Z1 inv_all=1.0f/(p01*p23);Z1 inv01=inv_all*p23;
Z1 inv23=inv_all*p01;inv0=sw0?inv_m:(inv01*p1);
inv1=sw1?inv_m:(inv01*p0);inv2=sw2?inv_m:(inv23*p3);
inv3=sw3?inv_m:(inv23*p2);}else{
inv0=sw0?inv_m:(1.0f/ClampPivot(cur_p0));inv1=sw1?inv_m:(1.0f/ClampPivot(cur_p1));
inv2=sw2?inv_m:(1.0f/ClampPivot(cur_p2));inv3=sw3?inv_m:(1.0f/ClampPivot(cur_p3));
}

Z1 f0=(sw0?cur_p0:m_i)*inv0;Z1 f1=(sw1?cur_p1:m_i)*inv1;
Z1 f2=(sw2?cur_p2:m_i)*inv2;Z1 f3=(sw3?cur_p3:m_i)*inv3;

Z1*s=aos+i*16u;s[0]=inv0;s[1]=inv1;s[2]=inv2;s[3]=inv3;
s[4]=f0;s[5]=f1;s[6]=f2;s[7]=f3;Z1 u1_0=sw0?nd0:cur_u0;
Z1 u1_1=sw1?nd1:cur_u1;Z1 u1_2=sw2?nd2:cur_u2;
Z1 u1_3=sw3?nd3:cur_u3;s[8]=u1_0*inv0;
s[9]=u1_1*inv1;s[10]=u1_2*inv2;
s[11]=u1_3*inv3;

cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_p2=(sw2?cur_u2:nd2)-f2*u1_2;cur_p3=(sw3?cur_u3:nd3)-f3*u1_3;

cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
cur_u2=sw2?(-f2*e_next):e_next;cur_u3=sw3?(-f3*e_next):e_next;

nq0+=(nq0>=1.1875f)?-0.1875f:0.53125f;nq1+=(nq1>=1.1875f)?-0.1875f:0.53125f;
nq2+=(nq2>=1.1875f)?-0.1875f:0.53125f;nq3+=(nq3>=1.1875f)?-0.1875f:0.53125f;

Z1 qi0=sw0?nq0:cq0;Z1 qi1=sw1?nq1:cq1;
Z1 qi2=sw2?nq2:cq2;Z1 qi3=sw3?nq3:cq3;
Z1 b0=sw0?cq0:nq0;Z1 b1=sw1?cq1:nq1;
Z1 b2=sw2?cq2:nq2;Z1 b3=sw3?cq3:nq3;
s[12]=qi0*inv0;s[13]=qi1*inv1;s[14]=qi2*inv2;s[15]=qi3*inv3;cq0=b0-f0*qi0;
cq1=b1-f1*qi1;cq2=b2-f2*qi2;
cq3=b3-f3*qi3;}
Z1*s_last=aos+(n-1u)*16u;Z1 lden0=ClampPivot(cur_p0);
Z1 lden1=ClampPivot(cur_p1);Z1 lden2=ClampPivot(cur_p2);
Z1 lden3=ClampPivot(cur_p3);
if(Z7(lden0)>=1e-7f&&Z7(lden1)>=1e-7f&&Z7(lden2)>=1e-7f&&Z7(lden3)>=1e-7f){
Z1 lp01=lden0*lden1,lp23=lden2*lden3;Z1 linv_all=1.0f/(lp01*lp23);
Z1 linv01=linv_all*lp23,linv23=linv_all*lp01;s_last[0]=linv01*lden1;
s_last[1]=linv01*lden0;s_last[2]=linv23*lden3;
s_last[3]=linv23*lden2;}else{
s_last[0]=1.0f/lden0;s_last[1]=1.0f/lden1;
s_last[2]=1.0f/lden2;s_last[3]=1.0f/lden3;
}


if(pend_post){
FinishEvaluateFinalPacket200(pend_r,pend_bcnt,pend_c0,pend_c1,pend_c2,pend_c3,v_scale);pend_post=false;
}
Z1 g0_sq=g0*g0,g1_sq=g1*g1,g2_sq=g2*g2,g3_sq=g3*g3;
bool can_early_stop=(min_g>=0.00008f*max_elem);Z1 s0_0=0.0f,s0_1=0.0f,s0_2=0.0f,s0_3=0.0f;
Z0 max_iter=3u;

Z1 in0=1.0f,in1=1.0f,in2=1.0f,in3=1.0f;for(Z0 iter=0;iter<max_iter;++iter){
if(iter>0u){
cq0=aos[12]*in0;cq1=aos[13]*in1;cq2=aos[14]*in2;cq3=aos[15]*in3;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*16u;
Z1 nq0=s[28]*in0,nq1=s[29]*in1,nq2=s[30]*in2,nq3=s[31]*in3;Z1 qi0=(sm&1u)?nq0:cq0;
Z1 qi1=(sm&2u)?nq1:cq1;Z1 qi2=(sm&4u)?nq2:cq2;
Z1 qi3=(sm&8u)?nq3:cq3;Z1 b0=(sm&1u)?cq0:nq0;
Z1 b1=(sm&2u)?cq1:nq1;Z1 b2=(sm&4u)?cq2:nq2;
Z1 b3=(sm&8u)?cq3:nq3;s[12]=qi0*s[0];
s[13]=qi1*s[1];s[14]=qi2*s[2];
s[15]=qi3*s[3];cq0=b0-s[4]*qi0;
cq1=b1-s[5]*qi1;cq2=b2-s[6]*qi2;
cq3=b3-s[7]*qi3;}
}
Z1 q0_n1=cq0*s_last[0];Z1 q1_n1=cq1*s_last[1];
Z1 q2_n1=cq2*s_last[2];Z1 q3_n1=cq3*s_last[3];
Z1 nsq0=q0_n1*q0_n1;Z1 nsq1=q1_n1*q1_n1;
Z1 nsq2=q2_n1*q2_n1;Z1 nsq3=q3_n1*q3_n1;

if(n==200u&&iter+1u>=max_iter&&no_gs&&bcnt>=3u){
Z4<Z1>vr0=ub_vecs[r*200u];Z4<Z1>vr1=ub_vecs[(r+1u)*200u];
Z4<Z1>vr2=ub_vecs[(r+2u)*200u];Z4<Z1>vr3=ub_vecs[(r+3u)*200u];
vr0.Z8(199u,q0_n1);vr1.Z8(199u,q1_n1);
vr3.Z8(199u,q3_n1);vr2.Z8(199u,q2_n1);
Z1*s_nm2=aos+198u*16u;Z1 q0_n2=q0_n1,q1_n2=q1_n1,q2_n2=q2_n1,q3_n2=q3_n1;
q0_n1=s_nm2[12]-s_nm2[8]*q0_n2;q1_n1=s_nm2[13]-s_nm2[9]*q1_n2;
q2_n1=s_nm2[14]-s_nm2[10]*q2_n2;q3_n1=s_nm2[15]-s_nm2[11]*q3_n2;
vr0.Z8(198u,q0_n1);vr1.Z8(198u,q1_n1);
vr3.Z8(198u,q3_n1);vr2.Z8(198u,q2_n1);
for(ZG i=197;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*16u;
Z1 c0_v=(s[12]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[8]*q0_n1;
Z1 c1_v=(s[13]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[9]*q1_n1;
Z1 c2_v=(s[14]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[10]*q2_n1;
Z1 c3_v=(s[15]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[11]*q3_n1;vr0.Z8(ui,c0_v);
vr1.Z8(ui,c1_v);vr3.Z8(ui,c3_v);
vr2.Z8(ui,c2_v);q0_n2=q0_n1;q0_n1=c0_v;
q1_n2=q1_n1;q1_n1=c1_v;q2_n2=q2_n1;q2_n1=c2_v;
q3_n2=q3_n1;q3_n1=c3_v;}
StartEvaluateFinalPacket200(r,v_scale);pend_post=true;
pend_r=r;pend_bcnt=bcnt;pend_c0=c0;pend_c1=c1;pend_c2=c2;pend_c3=c3;break;
}
if(iter+1u>=max_iter&&no_gs){
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+1u)*Z9];
Z4<Z1>vr2=ub_vecs[(r+2u)*Z9];Z4<Z1>vr3=ub_vecs[(r+((bcnt>3u)?3u:2u))*Z9];
vr0.Z8(n-1u,q0_n1);vr1.Z8(n-1u,q1_n1);
vr3.Z8(n-1u,q3_n1);vr2.Z8(n-1u,q2_n1);
Z1 d_last=dd[n-1u];Z1 rq0=d_last*nsq0,rq1=d_last*nsq1,rq2=d_last*nsq2,rq3=d_last*nsq3;
Z1 q0_n2=0.0f,q1_n2=0.0f,q2_n2=0.0f,q3_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*16u;q0_n2=q0_n1;q1_n2=q1_n1;q2_n2=q2_n1;q3_n2=q3_n1;
q0_n1=s_nm2[12]-s_nm2[8]*q0_n2;q1_n1=s_nm2[13]-s_nm2[9]*q1_n2;
q2_n1=s_nm2[14]-s_nm2[10]*q2_n2;q3_n1=s_nm2[15]-s_nm2[11]*q3_n2;
vr0.Z8(n-2u,q0_n1);vr1.Z8(n-2u,q1_n1);
vr3.Z8(n-2u,q3_n1);vr2.Z8(n-2u,q2_n1);
nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;
nsq2+=q2_n1*q2_n1;nsq3+=q3_n1*q3_n1;
Z1 d_nm2=dd[n-2u],two_e_nm2=2.0f*ee[n-2u];rq0+=q0_n1*(d_nm2*q0_n1+two_e_nm2*q0_n2);
rq1+=q1_n1*(d_nm2*q1_n1+two_e_nm2*q1_n2);rq2+=q2_n1*(d_nm2*q2_n1+two_e_nm2*q2_n2);
rq3+=q3_n1*(d_nm2*q3_n1+two_e_nm2*q3_n2);}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*16u;
Z1 c0_v=(s[12]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[8]*q0_n1;
Z1 c1_v=(s[13]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[9]*q1_n1;
Z1 c2_v=(s[14]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[10]*q2_n1;
Z1 c3_v=(s[15]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[11]*q3_n1;vr0.Z8(ui,c0_v);
vr1.Z8(ui,c1_v);vr3.Z8(ui,c3_v);
vr2.Z8(ui,c2_v);Z1 d_i=dd[ui],two_e=2.0f*ee[ui];
rq0+=c0_v*(d_i*c0_v+two_e*q0_n1);rq1+=c1_v*(d_i*c1_v+two_e*q1_n1);
rq2+=c2_v*(d_i*c2_v+two_e*q2_n1);rq3+=c3_v*(d_i*c3_v+two_e*q3_n1);
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;
nsq2+=c2_v*c2_v;nsq3+=c3_v*c3_v;
}
ub_eig.Z8(c0,rq0/nsq0);ub_eig.Z8(c1,rq1/nsq1);
ub_eig.Z8(c2,rq2/nsq2);if(bcnt>3u)ub_eig.Z8(c3,rq3/nsq3);
v_scale[r+0u]=ZI(nsq0);v_scale[r+1u]=ZI(nsq1);
v_scale[r+2u]=ZI(nsq2);if(bcnt>3u)v_scale[r+3u]=ZI(nsq3);
break;}

s_last[12]=q0_n1;s_last[13]=q1_n1;
s_last[14]=q2_n1;s_last[15]=q3_n1;

Z1 q0_n2=0.0f,q1_n2=0.0f,q2_n2=0.0f,q3_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*16u;q0_n2=q0_n1;q1_n2=q1_n1;q2_n2=q2_n1;q3_n2=q3_n1;
q0_n1=s_nm2[12]-s_nm2[8]*q0_n2;q1_n1=s_nm2[13]-s_nm2[9]*q1_n2;
q2_n1=s_nm2[14]-s_nm2[10]*q2_n2;q3_n1=s_nm2[15]-s_nm2[11]*q3_n2;
s_nm2[12]=q0_n1;s_nm2[13]=q1_n1;s_nm2[14]=q2_n1;s_nm2[15]=q3_n1;nsq0+=q0_n1*q0_n1;
nsq1+=q1_n1*q1_n1;nsq2+=q2_n1*q2_n1;
nsq3+=q3_n1*q3_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*16u;
Z1 c0_v=(s[12]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[8]*q0_n1;
Z1 c1_v=(s[13]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[9]*q1_n1;
Z1 c2_v=(s[14]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[10]*q2_n1;
Z1 c3_v=(s[15]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[11]*q3_n1;

s[12]=c0_v;s[13]=c1_v;s[14]=c2_v;s[15]=c3_v;

q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;

nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;
nsq2+=c2_v*c2_v;nsq3+=c3_v*c3_v;
}
if(iter==0u){
s0_0=nsq0*g0_sq;s0_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
s0_2=(bcnt>2u)?(nsq2*g2_sq):1.0e30f;s0_3=(bcnt>3u)?(nsq3*g3_sq):1.0e30f;
Z1 ms0_01=(s0_0<s0_1)?s0_0:s0_1;Z1 ms0_23=(s0_2<s0_3)?s0_2:s0_3;
if(no_gs&&((ms0_01<ms0_23)?ms0_01:ms0_23)>=250.0f){
max_iter=2u;}
}
bool is_final_iter=(iter+1u>=max_iter);if(iter==1u&&can_early_stop){
Z1 s1_0=nsq0*g0_sq;Z1 s1_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
Z1 s1_2=(bcnt>2u)?(nsq2*g2_sq):1.0e30f;Z1 s1_3=(bcnt>3u)?(nsq3*g3_sq):1.0e30f;
Z1 ms01=(s1_0<s1_1)?s1_0:s1_1;Z1 ms23=(s1_2<s1_3)?s1_2:s1_3;
Z1 min_s1=(ms01<ms23)?ms01:ms23;Z1 p0=s0_0*s1_0;
Z1 p1=(bcnt>1u)?(s0_1*s1_1):1.0e30f;Z1 p2=(bcnt>2u)?(s0_2*s1_2):1.0e30f;
Z1 p3=(bcnt>3u)?(s0_3*s1_3):1.0e30f;Z1 mp01=(p0<p1)?p0:p1;
Z1 mp23=(p2<p3)?p2:p3;Z1 min_p=(mp01<mp23)?mp01:mp23;
if((min_s1>=4.0e4f&&min_p>=1.0e9f)||(min_s1>=8.0e5f&&min_p>=3.5e8f)){
is_final_iter=true;}
}
if(!is_final_iter){
in0=ScalarRsqrtFast(nsq0);in1=ScalarRsqrtFast(nsq1);
in2=ScalarRsqrtFast(nsq2);in3=ScalarRsqrtFast(nsq3);
}else{
in0=ZI(nsq0);in1=ZI(nsq1);
in2=ZI(nsq2);in3=ZI(nsq3);
if(r>0u&&Z7(lam0-ub_eig.ZF(c0-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(aos[i*16u+12u]*in0)*prev_v.ZF(i);nsq0=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=aos[i*16u+12u]*in0-dot*prev_v.ZF(i);aos[i*16u+12u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
if(bcnt>1u&&Z7(lam1-lam0)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*16u+13u]*in1)*(aos[i*16u+12u]*in0);
nsq1=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*16u+13u]*in1-dot*(aos[i*16u+12u]*in0);aos[i*16u+13u]=v;
nsq1+=v*v;}
in1=ZI(nsq1);}
if(bcnt>2u&&Z7(lam2-lam1)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*16u+14u]*in2)*(aos[i*16u+13u]*in1);
nsq2=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*16u+14u]*in2-dot*(aos[i*16u+13u]*in1);aos[i*16u+14u]=v;
nsq2+=v*v;}
in2=ZI(nsq2);}
if(bcnt>3u&&Z7(lam3-lam2)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*16u+15u]*in3)*(aos[i*16u+14u]*in2);
nsq3=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*16u+15u]*in3-dot*(aos[i*16u+14u]*in2);aos[i*16u+15u]=v;
nsq3+=v*v;}
in3=ZI(nsq3);}
Z1 rq0=0.0f,rq1=0.0f,rq2=0.0f,rq3=0.0f;Z1 cur0=aos[12]*in0;
Z1 cur1=aos[13]*in1;Z1 cur2=aos[14]*in2;
Z1 cur3=aos[15]*in3;Z4<Z1>vr0=ub_vecs[r*Z9];
Z4<Z1>vr1=ub_vecs[(r+1u)*Z9];Z4<Z1>vr2=ub_vecs[(r+2u)*Z9];
Z4<Z1>vr3=ub_vecs[(r+((bcnt>3u)?3u:2u))*Z9];vr0.Z8(0,cur0);
vr1.Z8(0,cur1);vr3.Z8(0,cur3);
vr2.Z8(0,cur2);for(Z0 i=0;i+1u<n;++i){
const Z1*sn=aos+(i+1u)*16u;Z1 nxt0=sn[12]*in0;
Z1 nxt1=sn[13]*in1;Z1 nxt2=sn[14]*in2;
Z1 nxt3=sn[15]*in3;vr0.Z8(i+1u,nxt0);
vr1.Z8(i+1u,nxt1);vr3.Z8(i+1u,nxt3);
vr2.Z8(i+1u,nxt2);Z1 d_i=dd[i];
Z1 two_e=2.0f*ee[i];rq0+=cur0*(d_i*cur0+two_e*nxt0);
rq1+=cur1*(d_i*cur1+two_e*nxt1);rq2+=cur2*(d_i*cur2+two_e*nxt2);
rq3+=cur3*(d_i*cur3+two_e*nxt3);cur0=nxt0;cur1=nxt1;cur2=nxt2;cur3=nxt3;
}
Z1 d_last=dd[n-1u];rq0+=d_last*cur0*cur0;
rq1+=d_last*cur1*cur1;rq2+=d_last*cur2*cur2;
rq3+=d_last*cur3*cur3;ub_eig.Z8(c0,rq0);
ub_eig.Z8(c1,rq1);ub_eig.Z8(c2,rq2);
if(bcnt>3u)ub_eig.Z8(c3,rq3);break;
}
}
}

}
if(pend_post){
FinishEvaluateFinalPacket200(pend_r,pend_bcnt,pend_c0,pend_c1,pend_c2,pend_c3,v_scale);pend_post=false;
}
fence<HardEvent::S_V>();for(Z0 ri=0;ri<ZB;++ri){
if(v_scale[ri]!=0.0f){
Muls(ub_vecs[ri*Z9],ub_vecs[ri*Z9],v_scale[ri],n);}
}
}
#endif
__aicore__ inline void StartEvaluateFinalPacket256(Z0 r,Z1*v_scale,Z0 bcnt=4u){
Z4<Z1>d_copy=ZH[30208u];Z4<Z1>e_copy=ZH[30464u];
Z4<Z0>shift_idx=ZH[30720u].ReinterpretCast<Z0>();
Z4<Z1>sq=ZH[30976u];Z4<Z1>off=ZH[32000u];
Z4<Z1>next=ZH[33024u];Z4<Z1>terms=ZH[34048u];
Z4<Z1>red=ZH[35072u];Z0 tot=bcnt*256u;
ZA rep=Z5<ZA>(bcnt);fence<HardEvent::S_V>();
Mul(sq,ub_vecs[r*Z9],ub_vecs[r*Z9],tot);for(Z0 j=0u;j<bcnt;++j){
Gather(next[j*256u],ub_vecs[(r+j)*Z9],shift_idx,0u,256u);}
Z3<ZC>();Mul(off,next,ub_vecs[r*Z9],tot);
for(Z0 j=0u;j<bcnt;++j){
Mul(terms[j*256u],sq[j*256u],d_copy,256u);}
Z3<ZC>();for(Z0 c=0u;c<256u;c+=64u){
MulAddDst(terms[c],off[c],e_copy[c],64u,rep,Z6{1,1,1,32,32,0});}
Z3<ZC>();for(Z0 s=0u;s<4u;++s){
WholeReduceSum(red[s*8u],sq[s*64u],64u,rep,1u,1u,32u);
WholeReduceSum(red[32u+s*8u],terms[s*64u],64u,rep,1u,1u,32u);}
Z3<ZC>();Add(red,red,red[8u],bcnt);Add(red[16u],red[16u],red[24u],bcnt);
Add(red[32u],red[32u],red[40u],bcnt);Add(red[48u],red[48u],red[56u],bcnt);Z3<ZC>();
Add(red,red,red[16u],bcnt);Add(red[32u],red[32u],red[48u],bcnt);if(r>=4u){
for(Z0 ri=r-4u;ri<r;++ri){
if(v_scale[ri]!=0.0f){
Muls(ub_vecs[ri*256u],ub_vecs[ri*256u],v_scale[ri],256u);v_scale[ri]=0.0f;
}
}
}
}

__aicore__ inline void FinishEvaluateFinalPacket256(Z0 r,Z0 c0,Z0 c1,Z0 c2,Z0 c3,Z1*v_scale,Z0 bcnt=4u){
Z4<Z1>red=ZH[35072u];fence<HardEvent::V_S>();
Z0 cols[4]={c0,c1,c2,c3};for(Z0 j=0u;j<bcnt;++j){
Z1 norm=red.ZF(j);ub_eig.Z8(cols[j],red.ZF(32u+j)/norm);
v_scale[r+j]=ZI(norm);}
}

__aicore__ __attribute__((noinline))void ComputeInverseVectorsLarge256(Z1*work,ZA*swap_mask){
Z1*dd=work;Z1*ee=work+((n<=256u)?256u:512u);
bool is_diag=true;Z1 max_elem=0.0f;

Z1 saved_max=work[1023];is_diag=(saved_max<0.0f);
max_elem=is_diag?0.0f:saved_max;

if(ZB>0u&&(is_diag||n<Z9)){
Duplicate(ub_vecs,0.0f,ZB*Z9);Z3<PIPE_ALL>();
}

if(is_diag){
if constexpr(SOLVE_MODE!=2u&&SOLVE_MODE!=3u){
for(Z0 i=0;i<n;++i){
dd[i]=ub_diag.ZF(i);}
}
for(Z0 r=0;r<ZB;++r){
Z0 col_idx=ZJ+r;Z4<Z1>vec_r=ub_vecs[r*Z9];
for(Z0 i=0;i<n;++i){
Z0 rank=0u;Z1 di=dd[i];
for(Z0 k=0;k<n;++k){
if(dd[k]<di||(dd[k]==di&&k<i)){
++rank;}
}
if(rank==col_idx){
vec_r.Z8(i,1.0f);break;
}
}
}
Z3<PIPE_ALL>();return;
}

Z4<Z1>d_copy_init=ZH[30208u];Z4<Z1>e_copy_init=ZH[30464u];
Z4<Z0>shift_idx_init=ZH[30720u].ReinterpretCast<Z0>();
for(Z0 i=0u;i<255u;++i){
d_copy_init.Z8(i,dd[i]);e_copy_init.Z8(i,2.0f*ee[i]);
shift_idx_init.Z8(i,(i+1u)*4u);}
d_copy_init.Z8(255u,dd[255]);e_copy_init.Z8(255u,0.0f);
shift_idx_init.Z8(255u,0u);bool pend_post=false;
Z0 pend_r=0u,pend_c0=0u,pend_c1=0u,pend_c2=0u,pend_c3=0u;Z1 v_scale[32];
for(Z0 ri=0;ri<ZB;++ri)v_scale[ri]=0.0f;Z0 r=0u;
if(n<=256u){
Z1*ee_rat=work+512;Z1*inv_ee=work+768;
bool all_e_nz=true;Z0 i_ee=0u;
for(;i_ee+4u<n;i_ee+=4u){
Z1 e0=ee[i_ee],e1=ee[i_ee+1u],e2=ee[i_ee+2u],e3=ee[i_ee+3u],e4=ee[i_ee+4u];
if(Z7(e0)>=1e-7f&&Z7(e1)>=1e-7f&&Z7(e2)>=1e-7f&&Z7(e3)>=1e-7f){
Z1 p01=e0*e1,p23=e2*e3;Z1 inv_all=1.0f/(p01*p23);
Z1 inv01=inv_all*p23,inv23=inv_all*p01;Z1 iv0=inv01*e1,iv1=inv01*e0,iv2=inv23*e3,iv3=inv23*e2;
inv_ee[i_ee]=iv0;inv_ee[i_ee+1u]=iv1;inv_ee[i_ee+2u]=iv2;inv_ee[i_ee+3u]=iv3;
ee_rat[i_ee]=e1*iv0;ee_rat[i_ee+1u]=e2*iv1;ee_rat[i_ee+2u]=e3*iv2;ee_rat[i_ee+3u]=e4*iv3;}else{
all_e_nz=false;for(Z0 k=0;k<4u;++k){
Z1 ek=ee[i_ee+k];Z1 ivk=(ek!=0.0f)?(1.0f/ek):0.0f;
inv_ee[i_ee+k]=ivk;ee_rat[i_ee+k]=ee[i_ee+k+1u]*ivk;
}
}
}
for(;i_ee+1u<n;++i_ee){
Z1 ei=ee[i_ee];if(Z7(ei)<1e-7f)all_e_nz=false;
Z1 inv_e=(ei!=0.0f)?(1.0f/ei):0.0f;inv_ee[i_ee]=inv_e;
ee_rat[i_ee]=ee[i_ee+1u]*inv_e;}


Z1*aos=work+1024;for(;r+2u<ZB;r+=4u){
Z0 bcnt=(ZB-r<4u)?(ZB-r):4u;Z0 c0=ZJ+r;
Z0 c1=(bcnt>1u)?(c0+1u):c0;Z0 c2=(bcnt>2u)?(c0+2u):c1;
Z0 c3=(bcnt>3u)?(c0+3u):c2;Z1 lam0=ub_eig.ZF(c0);
Z1 lam1=ub_eig.ZF(c1);Z1 lam2=ub_eig.ZF(c2);
Z1 lam3=ub_eig.ZF(c3);


Z1 gl0=(c0>0u)?(lam0-ub_eig.ZF(c0-1u)):max_elem;
Z1 gr3=(c0+bcnt<n)?(ub_eig.ZF(c0+bcnt)-lam3):max_elem;Z1 g01=(bcnt>1u)?(lam1-lam0):max_elem;
Z1 g12=(bcnt>2u)?(lam2-lam1):max_elem;Z1 g23=(bcnt>3u)?(lam3-lam2):gr3;
Z1 g0=(gl0<g01)?gl0:g01;Z1 g1=(g01<g12)?g01:g12;
Z1 g2=(g12<g23)?g12:g23;Z1 g3=(bcnt>3u)?((g23<gr3)?g23:gr3):max_elem;
if(g0<0.0f)g0=0.0f;if(g1<0.0f)g1=0.0f;
if(g2<0.0f)g2=0.0f;if(g3<0.0f)g3=0.0f;
Z1 min_g=((g0<g1)?g0:g1)<((g2<g3)?g2:g3)?((g0<g1)?g0:g1):((g2<g3)?g2:g3);
const bool no_gs=(min_g>=0.0001f*max_elem)&&(gr3>=0.0001f*max_elem);
const bool fast_1pass4=(n==256u&&bcnt==4u&&no_gs&&all_e_nz);

Z1 cq0=1.375f,cq1=1.375f,cq2=1.375f,cq3=1.375f;

Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];Z1 cur_p1=dd[0]-lam1,cur_u1=ee[0];
Z1 cur_p2=dd[0]-lam2,cur_u2=ee[0];Z1 cur_p3=dd[0]-lam3,cur_u3=ee[0];

if(fast_1pass4){
Z1*s=aos;for(Z0 i=0u;i<255u;++i,s+=8u){
Z1 m_i=ee[i],abs_m=Z7(m_i),d_next=dd[i+1u],e_next=ee[i+1u];
Z1 nd0=d_next-lam0,nd1=d_next-lam1,nd2=d_next-lam2,nd3=d_next-lam3;
bool sw0=(Z7(cur_p0)<abs_m),sw1=(Z7(cur_p1)<abs_m),sw2=(Z7(cur_p2)<abs_m),sw3=(Z7(cur_p3)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u)|(sw2?4u:0u)|(sw3?8u:0u));swap_mask[i]=sm;
if(sm==15u){
Z1 inv_m=inv_ee[i],pim=1.375f*inv_m,nim=-pim;
Z1 f0=cur_p0*inv_m,f1=cur_p1*inv_m,f2=cur_p2*inv_m,f3=cur_p3*inv_m;
s[0]=nd0*inv_m;s[1]=nd1*inv_m;s[2]=nd2*inv_m;s[3]=nd3*inv_m;
cur_p0=cur_u0-f0*nd0;cur_p1=cur_u1-f1*nd1;cur_p2=cur_u2-f2*nd2;cur_p3=cur_u3-f3*nd3;
cur_u0=-f0*e_next;cur_u1=-f1*e_next;cur_u2=-f2*e_next;cur_u3=-f3*e_next;
bool p0=(f0*cq0>0.0f),p1=(f1*cq1>0.0f),p2=(f2*cq2>0.0f),p3=(f3*cq3>0.0f);
s[4]=p0?nim:pim;s[5]=p1?nim:pim;s[6]=p2?nim:pim;s[7]=p3?nim:pim;
cq0=cq0-f0*(p0?-1.375f:1.375f);cq1=cq1-f1*(p1?-1.375f:1.375f);cq2=cq2-f2*(p2?-1.375f:1.375f);cq3=cq3-f3*(p3?-1.375f:1.375f);
}else if(sm==0u){
Z1 p01=cur_p0*cur_p1,p23=cur_p2*cur_p3;Z1 inv_all=1.0f/(p01*p23);
Z1 inv01=inv_all*p23,inv23=inv_all*p01;
Z1 inv0=inv01*cur_p1,inv1=inv01*cur_p0,inv2=inv23*cur_p3,inv3=inv23*cur_p2;
Z1 f0=m_i*inv0,f1=m_i*inv1,f2=m_i*inv2,f3=m_i*inv3;
s[0]=cur_u0*inv0;s[1]=cur_u1*inv1;s[2]=cur_u2*inv2;s[3]=cur_u3*inv3;
cur_p0=nd0-f0*cur_u0;cur_p1=nd1-f1*cur_u1;cur_p2=nd2-f2*cur_u2;cur_p3=nd3-f3*cur_u3;
cur_u0=e_next;cur_u1=e_next;cur_u2=e_next;cur_u3=e_next;
Z1 sn0=(f0*cq0>0.0f)?-1.375f:1.375f,sn1=(f1*cq1>0.0f)?-1.375f:1.375f,sn2=(f2*cq2>0.0f)?-1.375f:1.375f,sn3=(f3*cq3>0.0f)?-1.375f:1.375f;
s[4]=cq0*inv0;s[5]=cq1*inv1;s[6]=cq2*inv2;s[7]=cq3*inv3;
cq0=sn0-f0*cq0;cq1=sn1-f1*cq1;cq2=sn2-f2*cq2;cq3=sn3-f3*cq3;}else{
Z1 inv_m=inv_ee[i];Z1 p0=sw0?1.0f:cur_p0,p1=sw1?1.0f:cur_p1,p2=sw2?1.0f:cur_p2,p3=sw3?1.0f:cur_p3;
Z1 p01=p0*p1,p23=p2*p3;Z1 inv_all=1.0f/(p01*p23);
Z1 inv01=inv_all*p23,inv23=inv_all*p01;
Z1 inv0=sw0?inv_m:(inv01*p1),inv1=sw1?inv_m:(inv01*p0),inv2=sw2?inv_m:(inv23*p3),inv3=sw3?inv_m:(inv23*p2);
Z1 f0=(sw0?cur_p0:m_i)*inv0,f1=(sw1?cur_p1:m_i)*inv1,f2=(sw2?cur_p2:m_i)*inv2,f3=(sw3?cur_p3:m_i)*inv3;
Z1 u1_0=sw0?nd0:cur_u0,u1_1=sw1?nd1:cur_u1,u1_2=sw2?nd2:cur_u2,u1_3=sw3?nd3:cur_u3;
s[0]=u1_0*inv0;s[1]=u1_1*inv1;s[2]=u1_2*inv2;s[3]=u1_3*inv3;
cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;cur_p2=(sw2?cur_u2:nd2)-f2*u1_2;cur_p3=(sw3?cur_u3:nd3)-f3*u1_3;
cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;cur_u2=sw2?(-f2*e_next):e_next;cur_u3=sw3?(-f3*e_next):e_next;
Z1 sn0=(f0*cq0>0.0f)?-1.375f:1.375f,sn1=(f1*cq1>0.0f)?-1.375f:1.375f,sn2=(f2*cq2>0.0f)?-1.375f:1.375f,sn3=(f3*cq3>0.0f)?-1.375f:1.375f;
Z1 qi0=sw0?sn0:cq0,qi1=sw1?sn1:cq1,qi2=sw2?sn2:cq2,qi3=sw3?sn3:cq3;
Z1 b0=sw0?cq0:sn0,b1=sw1?cq1:sn1,b2=sw2?cq2:sn2,b3=sw3?cq3:sn3;
s[4]=qi0*inv0;s[5]=qi1*inv1;s[6]=qi2*inv2;s[7]=qi3*inv3;
cq0=b0-f0*qi0;cq1=b1-f1*qi1;cq2=b2-f2*qi2;cq3=b3-f3*qi3;}
}
Z1 lden0=ClampPivot(cur_p0),lden1=ClampPivot(cur_p1),lden2=ClampPivot(cur_p2),lden3=ClampPivot(cur_p3);
Z1 sl0,sl1,sl2,sl3;
if(Z7(lden0)>=1e-7f&&Z7(lden1)>=1e-7f&&Z7(lden2)>=1e-7f&&Z7(lden3)>=1e-7f){
Z1 lp01=lden0*lden1,lp23=lden2*lden3;Z1 linv_all=1.0f/(lp01*lp23);
Z1 linv01=linv_all*lp23,linv23=linv_all*lp01;
sl0=linv01*lden1;sl1=linv01*lden0;sl2=linv23*lden3;sl3=linv23*lden2;}else{
sl0=1.0f/lden0;sl1=1.0f/lden1;sl2=1.0f/lden2;sl3=1.0f/lden3;}
if(pend_post){
FinishEvaluateFinalPacket256(pend_r,pend_c0,pend_c1,pend_c2,pend_c3,v_scale);pend_post=false;
}
Z4<Z1>vr0_f=ub_vecs[r*256u];Z4<Z1>vr1_f=ub_vecs[(r+1u)*256u];
Z4<Z1>vr2_f=ub_vecs[(r+2u)*256u];Z4<Z1>vr3_f=ub_vecs[(r+3u)*256u];
Z1 q0_n1=cq0*sl0,q1_n1=cq1*sl1,q2_n1=cq2*sl2,q3_n1=cq3*sl3;
vr0_f.Z8(255u,q0_n1);vr1_f.Z8(255u,q1_n1);vr2_f.Z8(255u,q2_n1);vr3_f.Z8(255u,q3_n1);
const Z1*s_nm2=aos+254u*8u;Z1 q0_n2=q0_n1,q1_n2=q1_n1,q2_n2=q2_n1,q3_n2=q3_n1;
q0_n1=s_nm2[4]-s_nm2[0]*q0_n2;q1_n1=s_nm2[5]-s_nm2[1]*q1_n2;
q2_n1=s_nm2[6]-s_nm2[2]*q2_n2;q3_n1=s_nm2[7]-s_nm2[3]*q3_n2;
vr0_f.Z8(254u,q0_n1);vr1_f.Z8(254u,q1_n1);vr2_f.Z8(254u,q2_n1);vr3_f.Z8(254u,q3_n1);
const Z1*sb=aos+253u*8u;for(ZG i=253;i>=0;--i,sb-=8u){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
const Z1*s=sb;Z1 u2_s=ee_rat[ui];
Z1 c0_v=(s[4]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[0]*q0_n1;
Z1 c1_v=(s[5]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[1]*q1_n1;
Z1 c2_v=(s[6]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[2]*q2_n1;
Z1 c3_v=(s[7]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[3]*q3_n1;
vr0_f.Z8(ui,c0_v);vr1_f.Z8(ui,c1_v);vr2_f.Z8(ui,c2_v);vr3_f.Z8(ui,c3_v);
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;
}
StartEvaluateFinalPacket256(r,v_scale);pend_post=true;
pend_r=r;pend_c0=c0;pend_c1=c1;pend_c2=c2;pend_c3=c3;continue;
}

for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i];Z1 abs_m=Z7(m_i);
Z1 d_next=dd[i+1u];Z1 e_next=ee[i+1u];
Z1 nd0=d_next-lam0;Z1 nd1=d_next-lam1;
Z1 nd2=d_next-lam2;Z1 nd3=d_next-lam3;

bool sw0=(Z7(cur_p0)<abs_m);bool sw1=(Z7(cur_p1)<abs_m);
bool sw2=(Z7(cur_p2)<abs_m);bool sw3=(Z7(cur_p3)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u)|(sw2?4u:0u)|(sw3?8u:0u));swap_mask[i]=sm;

Z1 inv_m=inv_ee[i];Z1 inv0,inv1,inv2,inv3;
if(sm==15u){
inv0=inv_m;inv1=inv_m;inv2=inv_m;inv3=inv_m;}else if(all_e_nz||abs_m>=1e-7f){
Z1 p0=sw0?1.0f:cur_p0;Z1 p1=sw1?1.0f:cur_p1;
Z1 p2=sw2?1.0f:cur_p2;Z1 p3=sw3?1.0f:cur_p3;
Z1 p01=p0*p1;Z1 p23=p2*p3;
Z1 inv_all=1.0f/(p01*p23);Z1 inv01=inv_all*p23;
Z1 inv23=inv_all*p01;inv0=sw0?inv_m:(inv01*p1);
inv1=sw1?inv_m:(inv01*p0);inv2=sw2?inv_m:(inv23*p3);
inv3=sw3?inv_m:(inv23*p2);}else{
inv0=sw0?inv_m:(1.0f/ClampPivot(cur_p0));inv1=sw1?inv_m:(1.0f/ClampPivot(cur_p1));
inv2=sw2?inv_m:(1.0f/ClampPivot(cur_p2));inv3=sw3?inv_m:(1.0f/ClampPivot(cur_p3));
}

Z1 f0=(sw0?cur_p0:m_i)*inv0;Z1 f1=(sw1?cur_p1:m_i)*inv1;
Z1 f2=(sw2?cur_p2:m_i)*inv2;Z1 f3=(sw3?cur_p3:m_i)*inv3;

Z1*s=aos+i*8u;Z1*sf=aos+2048u+i*4u;
sf[0]=sw0?f0:inv0;sf[1]=sw1?f1:inv1;
sf[2]=sw2?f2:inv2;sf[3]=sw3?f3:inv3;
Z1 u1_0=sw0?nd0:cur_u0;Z1 u1_1=sw1?nd1:cur_u1;
Z1 u1_2=sw2?nd2:cur_u2;Z1 u1_3=sw3?nd3:cur_u3;
s[0]=u1_0*inv0;s[1]=u1_1*inv1;
s[2]=u1_2*inv2;s[3]=u1_3*inv3;

cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_p2=(sw2?cur_u2:nd2)-f2*u1_2;cur_p3=(sw3?cur_u3:nd3)-f3*u1_3;

cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
cur_u2=sw2?(-f2*e_next):e_next;cur_u3=sw3?(-f3*e_next):e_next;

Z1 sn0=(f0*cq0>0.0f)?-1.375f:1.375f;Z1 sn1=(f1*cq1>0.0f)?-1.375f:1.375f;
Z1 sn2=(f2*cq2>0.0f)?-1.375f:1.375f;Z1 sn3=(f3*cq3>0.0f)?-1.375f:1.375f;
Z1 qi0=sw0?sn0:cq0;Z1 qi1=sw1?sn1:cq1;
Z1 qi2=sw2?sn2:cq2;Z1 qi3=sw3?sn3:cq3;
Z1 b0=sw0?cq0:sn0;Z1 b1=sw1?cq1:sn1;
Z1 b2=sw2?cq2:sn2;Z1 b3=sw3?cq3:sn3;
s[4]=qi0*inv0;s[5]=qi1*inv1;s[6]=qi2*inv2;s[7]=qi3*inv3;cq0=b0-f0*qi0;
cq1=b1-f1*qi1;cq2=b2-f2*qi2;
cq3=b3-f3*qi3;}
Z1*s_last=aos+(n-1u)*8u;Z1 lden0=ClampPivot(cur_p0);
Z1 lden1=ClampPivot(cur_p1);Z1 lden2=ClampPivot(cur_p2);
Z1 lden3=ClampPivot(cur_p3);
if(Z7(lden0)>=1e-7f&&Z7(lden1)>=1e-7f&&Z7(lden2)>=1e-7f&&Z7(lden3)>=1e-7f){
Z1 lp01=lden0*lden1,lp23=lden2*lden3;Z1 linv_all=1.0f/(lp01*lp23);
Z1 linv01=linv_all*lp23,linv23=linv_all*lp01;s_last[0]=linv01*lden1;
s_last[1]=linv01*lden0;s_last[2]=linv23*lden3;
s_last[3]=linv23*lden2;}else{
s_last[0]=1.0f/lden0;s_last[1]=1.0f/lden1;
s_last[2]=1.0f/lden2;s_last[3]=1.0f/lden3;
}

if(pend_post){
FinishEvaluateFinalPacket256(pend_r,pend_c0,pend_c1,pend_c2,pend_c3,v_scale);pend_post=false;
}
Z1 g0_sq=g0*g0,g1_sq=g1*g1,g2_sq=g2*g2,g3_sq=g3*g3;bool can_early_stop=(min_g>=0.00008f*max_elem);
Z1 s0_0=0.0f,s0_1=0.0f,s0_2=0.0f,s0_3=0.0f;Z0 max_iter=3u;

Z1 in0=1.0f,in1=1.0f,in2=1.0f,in3=1.0f;for(Z0 iter=0;iter<max_iter;++iter){
if(iter>0u){
cq0=aos[4]*in0;cq1=aos[5]*in1;
cq2=aos[6]*in2;cq3=aos[7]*in3;
for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1 m_i=ee[i],inv_m=inv_ee[i];
Z1*s=aos+i*8u;Z1*sf=aos+2048u+i*4u;
Z1 nq0=s[12]*in0,nq1=s[13]*in1,nq2=s[14]*in2,nq3=s[15]*in3;
Z1 qi0=(sm&1u)?nq0:cq0,qi1=(sm&2u)?nq1:cq1,qi2=(sm&4u)?nq2:cq2,qi3=(sm&8u)?nq3:cq3;
Z1 b0=(sm&1u)?cq0:nq0,b1=(sm&2u)?cq1:nq1,b2=(sm&4u)?cq2:nq2,b3=(sm&8u)?cq3:nq3;
Z1 sv0=sf[0],sv1=sf[1],sv2=sf[2],sv3=sf[3];s[4]=qi0*((sm&1u)?inv_m:sv0);
s[5]=qi1*((sm&2u)?inv_m:sv1);s[6]=qi2*((sm&4u)?inv_m:sv2);
s[7]=qi3*((sm&8u)?inv_m:sv3);cq0=b0-((sm&1u)?sv0:(m_i*sv0))*qi0;
cq1=b1-((sm&2u)?sv1:(m_i*sv1))*qi1;cq2=b2-((sm&4u)?sv2:(m_i*sv2))*qi2;
cq3=b3-((sm&8u)?sv3:(m_i*sv3))*qi3;}
}
Z1 q0_n1=cq0*s_last[0];Z1 q1_n1=cq1*s_last[1];
Z1 q2_n1=cq2*s_last[2];Z1 q3_n1=cq3*s_last[3];
Z1 nsq0=q0_n1*q0_n1;Z1 nsq1=q1_n1*q1_n1;
Z1 nsq2=q2_n1*q2_n1;Z1 nsq3=q3_n1*q3_n1;
if(iter+1u>=max_iter&&no_gs){
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+1u)*Z9];
Z4<Z1>vr2=ub_vecs[(r+2u)*Z9];Z4<Z1>vr3=ub_vecs[(r+((bcnt>3u)?3u:2u))*Z9];
vr0.Z8(n-1u,q0_n1);vr1.Z8(n-1u,q1_n1);
vr3.Z8(n-1u,q3_n1);vr2.Z8(n-1u,q2_n1);
Z1 d_last=dd[n-1u];Z1 rq0=d_last*nsq0,rq1=d_last*nsq1,rq2=d_last*nsq2,rq3=d_last*nsq3;
Z1 q0_n2=0.0f,q1_n2=0.0f,q2_n2=0.0f,q3_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*8u;q0_n2=q0_n1;q1_n2=q1_n1;q2_n2=q2_n1;q3_n2=q3_n1;
q0_n1=s_nm2[4]-s_nm2[0]*q0_n2;q1_n1=s_nm2[5]-s_nm2[1]*q1_n2;
q2_n1=s_nm2[6]-s_nm2[2]*q2_n2;q3_n1=s_nm2[7]-s_nm2[3]*q3_n2;
vr0.Z8(n-2u,q0_n1);vr1.Z8(n-2u,q1_n1);
vr3.Z8(n-2u,q3_n1);vr2.Z8(n-2u,q2_n1);
nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;
nsq2+=q2_n1*q2_n1;nsq3+=q3_n1*q3_n1;
Z1 d_nm2=dd[n-2u],two_e_nm2=2.0f*ee[n-2u];rq0+=q0_n1*(d_nm2*q0_n1+two_e_nm2*q0_n2);
rq1+=q1_n1*(d_nm2*q1_n1+two_e_nm2*q1_n2);rq2+=q2_n1*(d_nm2*q2_n1+two_e_nm2*q2_n2);
rq3+=q3_n1*(d_nm2*q3_n1+two_e_nm2*q3_n2);}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*8u;
Z1 c0_v=(s[4]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[0]*q0_n1;
Z1 c1_v=(s[5]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[1]*q1_n1;
Z1 c2_v=(s[6]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[2]*q2_n1;
Z1 c3_v=(s[7]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[3]*q3_n1;vr0.Z8(ui,c0_v);
vr1.Z8(ui,c1_v);vr3.Z8(ui,c3_v);
vr2.Z8(ui,c2_v);Z1 d_i=dd[ui],two_e=2.0f*ee[ui];
rq0+=c0_v*(d_i*c0_v+two_e*q0_n1);rq1+=c1_v*(d_i*c1_v+two_e*q1_n1);
rq2+=c2_v*(d_i*c2_v+two_e*q2_n1);rq3+=c3_v*(d_i*c3_v+two_e*q3_n1);
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;
nsq2+=c2_v*c2_v;nsq3+=c3_v*c3_v;
}
ub_eig.Z8(c0,rq0/nsq0);ub_eig.Z8(c1,rq1/nsq1);
ub_eig.Z8(c2,rq2/nsq2);if(bcnt>3u)ub_eig.Z8(c3,rq3/nsq3);
v_scale[r+0u]=ZI(nsq0);v_scale[r+1u]=ZI(nsq1);
v_scale[r+2u]=ZI(nsq2);if(bcnt>3u)v_scale[r+3u]=ZI(nsq3);
break;}

s_last[4]=q0_n1;s_last[5]=q1_n1;s_last[6]=q2_n1;s_last[7]=q3_n1;
Z1 q0_n2=0.0f,q1_n2=0.0f,q2_n2=0.0f,q3_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*8u;q0_n2=q0_n1;q1_n2=q1_n1;q2_n2=q2_n1;q3_n2=q3_n1;
q0_n1=s_nm2[4]-s_nm2[0]*q0_n2;q1_n1=s_nm2[5]-s_nm2[1]*q1_n2;
q2_n1=s_nm2[6]-s_nm2[2]*q2_n2;q3_n1=s_nm2[7]-s_nm2[3]*q3_n2;
s_nm2[4]=q0_n1;s_nm2[5]=q1_n1;s_nm2[6]=q2_n1;s_nm2[7]=q3_n1;
nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;nsq2+=q2_n1*q2_n1;nsq3+=q3_n1*q3_n1;}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*8u;
Z1 c0_v=(s[4]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[0]*q0_n1;
Z1 c1_v=(s[5]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[1]*q1_n1;
Z1 c2_v=(s[6]-((sm&4u)?u2_s:0.0f)*q2_n2)-s[2]*q2_n1;
Z1 c3_v=(s[7]-((sm&8u)?u2_s:0.0f)*q3_n2)-s[3]*q3_n1;s[4]=c0_v;s[5]=c1_v;s[6]=c2_v;s[7]=c3_v;
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
q2_n2=q2_n1;q2_n1=c2_v;q3_n2=q3_n1;q3_n1=c3_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;nsq2+=c2_v*c2_v;nsq3+=c3_v*c3_v;}
if(iter==0u){
s0_0=nsq0*g0_sq;s0_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
s0_2=(bcnt>2u)?(nsq2*g2_sq):1.0e30f;s0_3=(bcnt>3u)?(nsq3*g3_sq):1.0e30f;
Z1 ms0_01=(s0_0<s0_1)?s0_0:s0_1;Z1 ms0_23=(s0_2<s0_3)?s0_2:s0_3;
if(no_gs&&((ms0_01<ms0_23)?ms0_01:ms0_23)>=250.0f){
max_iter=2u;}
}
bool is_final_iter=(iter+1u>=max_iter);if(iter==1u&&can_early_stop){
Z1 s1_0=nsq0*g0_sq;Z1 s1_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
Z1 s1_2=(bcnt>2u)?(nsq2*g2_sq):1.0e30f;Z1 s1_3=(bcnt>3u)?(nsq3*g3_sq):1.0e30f;
Z1 ms01=(s1_0<s1_1)?s1_0:s1_1;Z1 ms23=(s1_2<s1_3)?s1_2:s1_3;
Z1 min_s1=(ms01<ms23)?ms01:ms23;Z1 p0=s0_0*s1_0;
Z1 p1=(bcnt>1u)?(s0_1*s1_1):1.0e30f;Z1 p2=(bcnt>2u)?(s0_2*s1_2):1.0e30f;
Z1 p3=(bcnt>3u)?(s0_3*s1_3):1.0e30f;Z1 mp01=(p0<p1)?p0:p1;
Z1 mp23=(p2<p3)?p2:p3;Z1 min_p=(mp01<mp23)?mp01:mp23;
if((min_s1>=4.0e4f&&min_p>=1.0e9f)||(min_s1>=8.0e5f&&min_p>=3.5e8f)){
is_final_iter=true;}
}
if(!is_final_iter){
in0=ScalarRsqrtFast(nsq0);in1=ScalarRsqrtFast(nsq1);
in2=ScalarRsqrtFast(nsq2);in3=ScalarRsqrtFast(nsq3);
}else{
in0=ZI(nsq0);in1=ZI(nsq1);
in2=ZI(nsq2);in3=ZI(nsq3);
if(r>0u&&Z7(lam0-ub_eig.ZF(c0-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+4u]*in0)*prev_v.ZF(i);nsq0=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+4u]*in0-dot*prev_v.ZF(i);aos[i*8u+4u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
if(bcnt>1u&&Z7(lam1-lam0)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+5u]*in1)*(aos[i*8u+4u]*in0);
nsq1=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+5u]*in1-dot*(aos[i*8u+4u]*in0);aos[i*8u+5u]=v;
nsq1+=v*v;}
in1=ZI(nsq1);}
if(bcnt>2u&&Z7(lam2-lam1)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+6u]*in2)*(aos[i*8u+5u]*in1);
nsq2=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+6u]*in2-dot*(aos[i*8u+5u]*in1);aos[i*8u+6u]=v;
nsq2+=v*v;}
in2=ZI(nsq2);}
if(bcnt>3u&&Z7(lam3-lam2)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+7u]*in3)*(aos[i*8u+6u]*in2);
nsq3=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+7u]*in3-dot*(aos[i*8u+6u]*in2);aos[i*8u+7u]=v;
nsq3+=v*v;}
in3=ZI(nsq3);}
Z1 rq0=0.0f,rq1=0.0f,rq2=0.0f,rq3=0.0f;Z1 cur0=aos[4]*in0;
Z1 cur1=aos[5]*in1;Z1 cur2=aos[6]*in2;
Z1 cur3=aos[7]*in3;Z4<Z1>vr0=ub_vecs[r*Z9];
Z4<Z1>vr1=ub_vecs[(r+1u)*Z9];Z4<Z1>vr2=ub_vecs[(r+2u)*Z9];
Z4<Z1>vr3=ub_vecs[(r+((bcnt>3u)?3u:2u))*Z9];vr0.Z8(0,cur0);
vr1.Z8(0,cur1);vr3.Z8(0,cur3);
vr2.Z8(0,cur2);for(Z0 i=0;i+1u<n;++i){
const Z1*sn=aos+(i+1u)*8u;Z1 nxt0=sn[4]*in0;
Z1 nxt1=sn[5]*in1;Z1 nxt2=sn[6]*in2;
Z1 nxt3=sn[7]*in3;vr0.Z8(i+1u,nxt0);
vr1.Z8(i+1u,nxt1);vr3.Z8(i+1u,nxt3);
vr2.Z8(i+1u,nxt2);Z1 d_i=dd[i];
Z1 two_e=2.0f*ee[i];rq0+=cur0*(d_i*cur0+two_e*nxt0);
rq1+=cur1*(d_i*cur1+two_e*nxt1);rq2+=cur2*(d_i*cur2+two_e*nxt2);
rq3+=cur3*(d_i*cur3+two_e*nxt3);cur0=nxt0;cur1=nxt1;cur2=nxt2;cur3=nxt3;
}
Z1 d_last=dd[n-1u];rq0+=d_last*cur0*cur0;
rq1+=d_last*cur1*cur1;rq2+=d_last*cur2*cur2;
rq3+=d_last*cur3*cur3;ub_eig.Z8(c0,rq0);
ub_eig.Z8(c1,rq1);ub_eig.Z8(c2,rq2);
if(bcnt>3u)ub_eig.Z8(c3,rq3);break;
}
}
}

for(;r<ZB;r+=2u){
Z0 bcnt=((ZB-r)<2u)?(ZB-r):2u;Z0 c0=ZJ+r;

Z0 c1=(bcnt>1u)?(c0+1u):c0;Z1 lam0=ub_eig.ZF(c0);
Z1 lam1=ub_eig.ZF(c1);Z1 gl0=(c0>0u)?(lam0-ub_eig.ZF(c0-1u)):max_elem;
Z1 g01=(bcnt>1u)?(lam1-lam0):max_elem;Z1 gr1=(c0+bcnt<n)?(ub_eig.ZF(c0+bcnt)-lam1):max_elem;
Z1 g0=(gl0<g01)?gl0:g01;Z1 g1=(bcnt>1u)?((g01<gr1)?g01:gr1):g0;
if(g0<0.0f)g0=0.0f;if(g1<0.0f)g1=0.0f;
const bool no_gs2=(((g0<g1)?g0:g1)>=0.0001f*max_elem)&&(gr1>=0.0001f*max_elem);
const bool fast_1pass2=(n==256u&&bcnt==2u&&no_gs2&&all_e_nz);Z1 cq0=1.375f,cq1=1.375f;
Z1 cur_p0=dd[0]-lam0,cur_u0=ee[0];Z1 cur_p1=dd[0]-lam1,cur_u1=ee[0];
if(fast_1pass2){
Z1*s=aos;for(Z0 i=0u;i<255u;++i,s+=4u){
Z1 m_i=ee[i],abs_m=Z7(m_i),d_next=dd[i+1u],e_next=ee[i+1u];Z1 nd0=d_next-lam0,nd1=d_next-lam1;
bool sw0=(Z7(cur_p0)<abs_m),sw1=(Z7(cur_p1)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u));swap_mask[i]=sm;
if(sm==3u){
Z1 inv_m=inv_ee[i],pim=1.375f*inv_m,nim=-pim;Z1 f0=cur_p0*inv_m,f1=cur_p1*inv_m;
s[0]=nd0*inv_m;s[1]=nd1*inv_m;cur_p0=cur_u0-f0*nd0;cur_p1=cur_u1-f1*nd1;
cur_u0=-f0*e_next;cur_u1=-f1*e_next;bool p0=(f0*cq0>0.0f),p1=(f1*cq1>0.0f);
s[2]=p0?nim:pim;s[3]=p1?nim:pim;cq0=cq0-f0*(p0?-1.375f:1.375f);cq1=cq1-f1*(p1?-1.375f:1.375f);
}else if(sm==0u){
Z1 inv01=1.0f/(cur_p0*cur_p1);Z1 inv0=inv01*cur_p1,inv1=inv01*cur_p0;
Z1 f0=m_i*inv0,f1=m_i*inv1;s[0]=cur_u0*inv0;s[1]=cur_u1*inv1;
cur_p0=nd0-f0*cur_u0;cur_p1=nd1-f1*cur_u1;cur_u0=e_next;cur_u1=e_next;
Z1 sn0=(f0*cq0>0.0f)?-1.375f:1.375f,sn1=(f1*cq1>0.0f)?-1.375f:1.375f;s[2]=cq0*inv0;s[3]=cq1*inv1;
cq0=sn0-f0*cq0;cq1=sn1-f1*cq1;}else{
Z1 inv_m=inv_ee[i];Z1 p0=sw0?1.0f:cur_p0,p1=sw1?1.0f:cur_p1;
Z1 inv01=1.0f/(p0*p1);Z1 inv0=sw0?inv_m:(inv01*p1),inv1=sw1?inv_m:(inv01*p0);
Z1 f0=(sw0?cur_p0:m_i)*inv0,f1=(sw1?cur_p1:m_i)*inv1;Z1 u1_0=sw0?nd0:cur_u0,u1_1=sw1?nd1:cur_u1;
s[0]=u1_0*inv0;s[1]=u1_1*inv1;cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
Z1 sn0=(f0*cq0>0.0f)?-1.375f:1.375f,sn1=(f1*cq1>0.0f)?-1.375f:1.375f;
Z1 qi0=sw0?sn0:cq0,qi1=sw1?sn1:cq1;Z1 b0=sw0?cq0:sn0,b1=sw1?cq1:sn1;
s[2]=qi0*inv0;s[3]=qi1*inv1;cq0=b0-f0*qi0;cq1=b1-f1*qi1;
}
}
Z1 sl0=1.0f/ClampPivot(cur_p0),sl1=1.0f/ClampPivot(cur_p1);if(pend_post){
FinishEvaluateFinalPacket256(pend_r,pend_c0,pend_c1,pend_c2,pend_c3,v_scale);pend_post=false;
}
Z4<Z1>vr0_2=ub_vecs[r*256u];Z4<Z1>vr1_2=ub_vecs[(r+1u)*256u];
Z1 q0_n1=cq0*sl0,q1_n1=cq1*sl1;vr0_2.Z8(255u,q0_n1);vr1_2.Z8(255u,q1_n1);
const Z1*s_nm2=aos+254u*4u;Z1 q0_n2=q0_n1,q1_n2=q1_n1;
q0_n1=s_nm2[2]-s_nm2[0]*q0_n2;q1_n1=s_nm2[3]-s_nm2[1]*q1_n2;
vr0_2.Z8(254u,q0_n1);vr1_2.Z8(254u,q1_n1);const Z1*sb=aos+253u*4u;
for(ZG i=253;i>=0;--i,sb-=4u){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
const Z1*s=sb;Z1 u2_s=ee_rat[ui];
Z1 c0_v=(s[2]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[0]*q0_n1;
Z1 c1_v=(s[3]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[1]*q1_n1;
vr0_2.Z8(ui,c0_v);vr1_2.Z8(ui,c1_v);q0_n2=q0_n1;q0_n1=c0_v;
q1_n2=q1_n1;q1_n1=c1_v;}
StartEvaluateFinalPacket256(r,v_scale,2u);FinishEvaluateFinalPacket256(r,c0,c1,c1,c1,v_scale,2u);
continue;}
for(Z0 i=0;i+1u<n;++i){
Z1 m_i=ee[i],abs_m=Z7(m_i),d_next=dd[i+1u],e_next=ee[i+1u];Z1 nd0=d_next-lam0,nd1=d_next-lam1;
bool sw0=(Z7(cur_p0)<abs_m),sw1=(Z7(cur_p1)<abs_m);
ZA sm=Z5<ZA>((sw0?1u:0u)|(sw1?2u:0u));swap_mask[i]=sm;
Z1 inv_m=inv_ee[i];Z1 inv0,inv1;
if(sm==3u){
inv0=inv_m;inv1=inv_m;}else if(all_e_nz||abs_m>=1e-7f){
Z1 p0=sw0?1.0f:cur_p0;Z1 p1=sw1?1.0f:cur_p1;
Z1 inv01=1.0f/(p0*p1);inv0=sw0?inv_m:(inv01*p1);
inv1=sw1?inv_m:(inv01*p0);}else{
inv0=sw0?inv_m:(1.0f/ClampPivot(cur_p0));inv1=sw1?inv_m:(1.0f/ClampPivot(cur_p1));
}
Z1 f0=(sw0?cur_p0:m_i)*inv0,f1=(sw1?cur_p1:m_i)*inv1;Z1*s=aos+i*8u;
s[0]=inv0;s[1]=inv1;s[2]=f0;s[3]=f1;
Z1 u1_0=sw0?nd0:cur_u0,u1_1=sw1?nd1:cur_u1;s[4]=u1_0*inv0;s[5]=u1_1*inv1;
cur_p0=(sw0?cur_u0:nd0)-f0*u1_0;cur_p1=(sw1?cur_u1:nd1)-f1*u1_1;
cur_u0=sw0?(-f0*e_next):e_next;cur_u1=sw1?(-f1*e_next):e_next;
Z1 sn0=(f0*cq0>0.0f)?-1.375f:1.375f,sn1=(f1*cq1>0.0f)?-1.375f:1.375f;
Z1 qi0=sw0?sn0:cq0,qi1=sw1?sn1:cq1;Z1 b0=sw0?cq0:sn0,b1=sw1?cq1:sn1;
s[6]=qi0*inv0;s[7]=qi1*inv1;cq0=b0-f0*qi0;cq1=b1-f1*qi1;
}
Z1*s_last=aos+(n-1u)*8u;s_last[0]=1.0f/ClampPivot(cur_p0);
s_last[1]=1.0f/ClampPivot(cur_p1);if(pend_post){
FinishEvaluateFinalPacket256(pend_r,pend_c0,pend_c1,pend_c2,pend_c3,v_scale);pend_post=false;
}
Z1 g0_sq=g0*g0,g1_sq=g1*g1;bool can_early_stop=(((g0<g1)?g0:g1)>=0.00008f*max_elem);
Z1 s0_0=0.0f,s0_1=0.0f;Z0 max_iter=3u;
Z1 in0=1.0f,in1=1.0f;for(Z0 iter=0;iter<max_iter;++iter){
if(iter>0u){
cq0=aos[6]*in0;cq1=aos[7]*in1;for(Z0 i=0;i+1u<n;++i){
ZA sm=swap_mask[i];Z1*s=aos+i*8u;
Z1 nq0=s[14]*in0,nq1=s[15]*in1;Z1 qi0=(sm&1u)?nq0:cq0,qi1=(sm&2u)?nq1:cq1;
Z1 b0=(sm&1u)?cq0:nq0,b1=(sm&2u)?cq1:nq1;s[6]=qi0*s[0];s[7]=qi1*s[1];
cq0=b0-s[2]*qi0;cq1=b1-s[3]*qi1;}
}
Z1 q0_n1=cq0*s_last[0],q1_n1=cq1*s_last[1];Z1 nsq0=q0_n1*q0_n1,nsq1=q1_n1*q1_n1;
bool no_gs2=(((g0<g1)?g0:g1)>=0.0001f*max_elem)&&(gr1>=0.0001f*max_elem);if(iter+1u>=max_iter&&no_gs2){
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+((bcnt>1u)?1u:0u))*Z9];
vr1.Z8(n-1u,q1_n1);vr0.Z8(n-1u,q0_n1);
Z1 d_last=dd[n-1u];Z1 rq0=d_last*nsq0,rq1=d_last*nsq1;
Z1 q0_n2=0.0f,q1_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*8u;q0_n2=q0_n1;q1_n2=q1_n1;
q0_n1=s_nm2[6]-s_nm2[4]*q0_n2;q1_n1=s_nm2[7]-s_nm2[5]*q1_n2;
vr1.Z8(n-2u,q1_n1);vr0.Z8(n-2u,q0_n1);
nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;Z1 d_nm2=dd[n-2u],two_e_nm2=2.0f*ee[n-2u];
rq0+=q0_n1*(d_nm2*q0_n1+two_e_nm2*q0_n2);rq1+=q1_n1*(d_nm2*q1_n1+two_e_nm2*q1_n2);
}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*8u;
Z1 c0_v=(s[6]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[4]*q0_n1;
Z1 c1_v=(s[7]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[5]*q1_n1;vr1.Z8(ui,c1_v);
vr0.Z8(ui,c0_v);Z1 d_i=dd[ui],two_e=2.0f*ee[ui];
rq0+=c0_v*(d_i*c0_v+two_e*q0_n1);rq1+=c1_v*(d_i*c1_v+two_e*q1_n1);
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;}
ub_eig.Z8(c0,rq0/nsq0);if(bcnt>1u)ub_eig.Z8(c1,rq1/nsq1);
v_scale[r+0u]=ZI(nsq0);if(bcnt>1u)v_scale[r+1u]=ZI(nsq1);
break;}
s_last[6]=q0_n1;s_last[7]=q1_n1;Z4<Z1>vr0_0=ub_vecs[r*Z9];
Z4<Z1>vr1_0=ub_vecs[(r+((bcnt>1u)?1u:0u))*Z9];const bool try_fast_rem2=(iter==0u&&no_gs2);
Z1 rq0_0=0.0f,rq1_0=0.0f;if(try_fast_rem2){
vr1_0.Z8(n-1u,q1_n1);vr0_0.Z8(n-1u,q0_n1);
Z1 d_last=dd[n-1u];rq0_0=d_last*nsq0;rq1_0=d_last*nsq1;
}
Z1 q0_n2=0.0f,q1_n2=0.0f;if(n>1u){
Z1*s_nm2=aos+(n-2u)*8u;q0_n2=q0_n1;q1_n2=q1_n1;
q0_n1=s_nm2[6]-s_nm2[4]*q0_n2;q1_n1=s_nm2[7]-s_nm2[5]*q1_n2;
s_nm2[6]=q0_n1;s_nm2[7]=q1_n1;nsq0+=q0_n1*q0_n1;nsq1+=q1_n1*q1_n1;
if(try_fast_rem2){
vr1_0.Z8(n-2u,q1_n1);vr0_0.Z8(n-2u,q0_n1);
Z1 d_nm2=dd[n-2u],two_e_nm2=2.0f*ee[n-2u];rq0_0+=q0_n1*(d_nm2*q0_n1+two_e_nm2*q0_n2);
rq1_0+=q1_n1*(d_nm2*q1_n1+two_e_nm2*q1_n2);}
}
for(ZG i=Z5<ZG>(n)-3;i>=0;--i){
Z0 ui=Z5<Z0>(i);ZA sm=swap_mask[ui];
Z1 u2_s=ee_rat[ui];Z1*s=aos+ui*8u;
Z1 c0_v=(s[6]-((sm&1u)?u2_s:0.0f)*q0_n2)-s[4]*q0_n1;
Z1 c1_v=(s[7]-((sm&2u)?u2_s:0.0f)*q1_n2)-s[5]*q1_n1;s[6]=c0_v;s[7]=c1_v;
if(try_fast_rem2){
vr1_0.Z8(ui,c1_v);vr0_0.Z8(ui,c0_v);
Z1 d_i=dd[ui],two_e=2.0f*ee[ui];rq0_0+=c0_v*(d_i*c0_v+two_e*q0_n1);
rq1_0+=c1_v*(d_i*c1_v+two_e*q1_n1);}
q0_n2=q0_n1;q0_n1=c0_v;q1_n2=q1_n1;q1_n1=c1_v;
nsq0+=c0_v*c0_v;nsq1+=c1_v*c1_v;}
if(iter==0u){
s0_0=nsq0*g0_sq;s0_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
Z1 min_s0_2=(s0_0<s0_1)?s0_0:s0_1;if(try_fast_rem2&&min_s0_2>=2.0e7f){
ub_eig.Z8(c0,rq0_0/nsq0);if(bcnt>1u)ub_eig.Z8(c1,rq1_0/nsq1);
v_scale[r+0u]=ZI(nsq0);if(bcnt>1u)v_scale[r+1u]=ZI(nsq1);
break;}
if(no_gs2&&min_s0_2>=250.0f){
max_iter=2u;}
}
bool is_final_iter=(iter+1u>=max_iter);if(iter==1u&&can_early_stop){
Z1 s1_0=nsq0*g0_sq;Z1 s1_1=(bcnt>1u)?(nsq1*g1_sq):1.0e30f;
Z1 min_s1=(s1_0<s1_1)?s1_0:s1_1;Z1 p0=s0_0*s1_0;
Z1 p1=(bcnt>1u)?(s0_1*s1_1):1.0e30f;Z1 min_p=(p0<p1)?p0:p1;
if((min_s1>=4.0e4f&&min_p>=1.0e9f)||(min_s1>=8.0e5f&&min_p>=3.5e8f)){
is_final_iter=true;}
}
if(!is_final_iter){
in0=ScalarRsqrtFast(nsq0);in1=ScalarRsqrtFast(nsq1);
}else{
in0=ZI(nsq0);in1=ZI(nsq1);
if(r>0u&&Z7(lam0-ub_eig.ZF(c0-1u))<0.0001f*max_elem){
Z4<Z1>prev_v=ub_vecs[(r-1u)*Z9];Z1 dot=0.0f;
for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+6u]*in0)*prev_v.ZF(i);nsq0=0.0f;
for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+6u]*in0-dot*prev_v.ZF(i);aos[i*8u+6u]=v;
nsq0+=v*v;}
in0=ZI(nsq0);}
if(bcnt>1u&&Z7(lam1-lam0)<0.0001f*max_elem){
Z1 dot=0.0f;for(Z0 i=0;i<n;++i)dot+=(aos[i*8u+7u]*in1)*(aos[i*8u+6u]*in0);
nsq1=0.0f;for(Z0 i=0;i<n;++i){
Z1 v=aos[i*8u+7u]*in1-dot*(aos[i*8u+6u]*in0);aos[i*8u+7u]=v;
nsq1+=v*v;}
in1=ZI(nsq1);}
Z1 rq0=0.0f,rq1=0.0f;Z1 cur0=aos[6]*in0,cur1=aos[7]*in1;
Z4<Z1>vr0=ub_vecs[r*Z9];Z4<Z1>vr1=ub_vecs[(r+((bcnt>1u)?1u:0u))*Z9];
vr1.Z8(0,cur1);vr0.Z8(0,cur0);
for(Z0 i=0;i+1u<n;++i){
const Z1*sn=aos+(i+1u)*8u;Z1 nxt0=sn[6]*in0,nxt1=sn[7]*in1;
vr1.Z8(i+1u,nxt1);vr0.Z8(i+1u,nxt0);
Z1 d_i=dd[i],two_e=2.0f*ee[i];rq0+=d_i*cur0*cur0+two_e*cur0*nxt0;
rq1+=d_i*cur1*cur1+two_e*cur1*nxt1;cur0=nxt0;cur1=nxt1;
}
Z1 d_last=dd[n-1u];rq0+=d_last*cur0*cur0;
rq1+=d_last*cur1*cur1;ub_eig.Z8(c0,rq0);
if(bcnt>1u)ub_eig.Z8(c1,rq1);break;
}
}
}
}

if(pend_post){
FinishEvaluateFinalPacket256(pend_r,pend_c0,pend_c1,pend_c2,pend_c3,v_scale);pend_post=false;
}
fence<HardEvent::S_V>();for(Z0 ri=0;ri<ZB;++ri){
if(v_scale[ri]!=0.0f){
Muls(ub_vecs[ri*Z9],ub_vecs[ri*Z9],v_scale[ri],n);}
}
}



__aicore__ inline void WriteOutputsLarge(uint64_t b){
if(ZB==0u){
return;}
Z0 aligned_cnt=(ZB+7u)&~7u;Z0 tile_stride=aligned_cnt;
ZA rep_stride=Z5<ZA>(Z9/8u);Mul(ZH,ub_vecs,ub_vecs,ZB*Z9);
Z3<ZC>();bool fast_strip_max=(n>=40u&&n<=256u&&aligned_cnt<=64u);
Z4<Z1>strip_max_base=ub_gather;Z0 max_strip_cnt=0u;
Z0 strip_step=16u;if(fast_strip_max){
for(Z0 c=0;c<n;c+=strip_step,++max_strip_cnt){
Z0 width=(n-c<strip_step)?(n-c):strip_step;
WholeReduceMax(strip_max_base[max_strip_cnt*aligned_cnt],ZH[c],width,ZB,1,1,rep_stride,ReduceOrder::ORDER_ONLY_VALUE);
}
}
Z4<Z0>col_idx_buf=ub_diag.ReinterpretCast<Z0>();for(Z0 c=0;c<tile_stride;++c){
col_idx_buf.Z8(c,c*Z9*4u);}
for(Z0 r=0;r<ZB;++r){
ub_row.Z8(r,ub_eig.ZF(ZJ+r)*matrix_scale);}
Z3<PIPE_ALL>();

for(Z0 r=0;r<ZB;++r){
Z4<Z1>sq_r=ZH[r*Z9];Z4<Z1>vec_r=ub_vecs[r*Z9];
Z0 scan_start=0u;Z0 scan_end=n;
if(fast_strip_max){
Z0 best_s=0u;Z1 best_m=strip_max_base.ZF(r);
for(Z0 s=1u;s<max_strip_cnt;++s){
Z1 m_s=strip_max_base.ZF(s*aligned_cnt+r);if(m_s>best_m){
best_m=m_s;best_s=s;
}
}
scan_start=best_s*strip_step;scan_end=(scan_start+strip_step<n)?(scan_start+strip_step):n;
}
Z0 max_idx=scan_start;Z1 max_sq=sq_r.ZF(scan_start);
Z0 i=scan_start+1u;for(;i+3u<scan_end;i+=4u){
Z1 v0=sq_r.ZF(i+0u);Z1 v1=sq_r.ZF(i+1u);
Z1 v2=sq_r.ZF(i+2u);Z1 v3=sq_r.ZF(i+3u);
if(v0>max_sq){max_sq=v0;max_idx=i+0u;}
if(v1>max_sq){max_sq=v1;max_idx=i+1u;}
if(v2>max_sq){max_sq=v2;max_idx=i+2u;}
if(v3>max_sq){max_sq=v3;max_idx=i+3u;}
}
for(;i<scan_end;++i){
Z1 v_sq=sq_r.ZF(i);if(v_sq>max_sq){
max_sq=v_sq;max_idx=i;
}
}
if(vec_r.ZF(max_idx)<0.0f){
Muls(vec_r,vec_r,-1.0f,n);}
}
Z3<ZC>();
for(Z0 i=0;i<n;++i){
Gather(ZH[i*tile_stride],ub_vecs,col_idx_buf,i*4u,ZB);
}
Z3<PIPE_ALL>();
DataCopyPad(
gm_eigvals[b*n+ZJ],
ub_row,
DataCopyExtParams{1,ZB*4u,0,0,0});
DataCopyPad(
gm_eigvecs[b*n*n+ZJ],
ZH,
DataCopyExtParams{Z5<uint16_t>(n),ZB*4u,0,(n-ZB)*4u,0});
}

Z0 n;
Z0 Z9;
Z0 uplo;
Z0 group_lanes;
Z0 num_groups;
Z0 rows_per_lane;
Z0 vec_rows_per_lane;
Z0 lane_id;
Z0 group_id;
Z0 my_blocks;
Z0 coop_row_start;
Z0 coop_row_count;
Z0 vec_row_start;
Z0 vec_row_count;
Z0 ZJ;
Z0 ZB;
Z0 solve_mode;
Z0 num_zones;
Z0 vecs_offset;
Z0 zone_offset[9];
uint64_t batch_size;
uint64_t ws_refl_offset;
uint64_t ws_w_offset;
uint64_t ws_scale_offset;
uint64_t ws_tail_offset;
bool is_active;
Z1 matrix_scale;

TBuf<TPosition::VECCALC>ub_buf;
GlobalTensor<Z1>gm_A;
GlobalTensor<Z1>gm_eigvals;
GlobalTensor<Z1>gm_eigvecs;
GlobalTensor<Z1>gm_ws;
#if EIGH_K96_PASS
GlobalTensor<Z1>gm_ckpt;
#else
#endif

Z4<Z1>ZH;
Z4<Z1>ub_vecs;
Z4<Z1>ub_refl;
Z4<Z1>ub_row;
Z4<Z1>ub_v;
Z4<Z1>ub_w;
Z4<Z1>ub_cur_row;
Z4<Z1>ub_prod;
Z4<Z1>ub_tmp;
Z4<Z1>ub_red;
Z4<Z1>ub_diag;
Z4<Z1>ub_off;
Z4<Z1>ub_tau;
Z4<Z1>ub_eig;
Z4<Z1>ub_comm;
Z4<Z1>ub_gather;
Z4<Z1>ub_mask_ge;
Z4<Z1>ub_mask_gt;
Z4<Z1>ub_mask_eq;
#if EIGH_K96_PASS
};
#else
};
#endif

// Do not expose packaging aliases to headers or kernel entry.
#undef Z0
#undef Z1
#undef Z2
#undef Z3
#undef Z4
#undef Z5
#undef Z6
#undef Z7
#undef Z8
#undef Z9
#undef ZA
#undef ZB
#undef ZC
#undef ZD
#undef ZE
#undef ZF
#undef ZG
#undef ZH
#undef ZI
#undef ZJ

#else

#include "kernel_operator.h"
#include "linalg_eigh_tiling.h"
#include "tiling_key_linalg_eigh.h"

using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

// PRIVATE: pure-vector 40-AIV K96 stage sharing the shared 49088-float UB.
// Variant v2: 8 active AIVs/group (32 active AIVs total, 32 rows/lane instead of 64 rows/lane)
// with pre-SyncAll PIPE_V/MTE3 overlap and fused next/mat rank-2 updates.
constexpr uint32_t STEPS=96u;
constexpr bool HANDOFF=true;
constexpr bool DIAG=false;
constexpr bool CHECKPOINT_ONLY=false;
constexpr bool INIT_SHELL=false;
template<HardEvent E> __aicore__ inline void stage_fence() {
 const auto id=static_cast<int32_t>(GetTPipePtr()->FetchEventID(E));
 SetFlag<E>(id); WaitFlag<E>(id);
}
__aicore__ inline uint32_t owner(uint32_t r) { return (r>>3u)&7u; }
__aicore__ inline uint32_t localrow(uint32_t r) { return (r>>6u)*8u+(r&7u); }
__aicore__ inline float rsqrt_parent(float x) {
 if(x<=0.0f) return 0.0f;
 union { float f; uint32_t u; } b; b.f=x; b.u=0x5f375a86u-(b.u>>1u);
 float y=b.f,h=0.5f*x;
 y=y*(1.5f-h*y*y); y=y*(1.5f-h*y*y); y=y*(1.5f-h*y*y); return y;
}
__aicore__ inline void stage_scale(float mx,float&scale,float&inv) {
 union { float f; uint32_t u; } b; b.f=mx;
 uint32_t exp=b.u&0x7f800000u;
 if(exp==0u||exp==0x7f800000u){ scale=1.0f; inv=1.0f; }
 else { b.u=exp; scale=b.f; b.u=0x7f000000u-exp; inv=b.f; }
}
class Stage96 {
public:
 __aicore__ inline void Init(GM_ADDR a,GM_ADDR w,GM_ADDR q,GM_ADDR ws,
                             const LinalgEighTilingData&t,LocalTensor<float>ub) {
  n=t.n; batch=static_cast<uint32_t>(t.batch_size); uplo=t.uplo;
  cid=GetBlockIdx(); group=cid/10u; lane=cid%10u; active=lane<8u;
  in.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
  outw.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
  outq.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(q));
  gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws),268288u);
  mat=ub; v=ub[8192u]; qpart=ub[8448u]; qfull=ub[8704u]; next=ub[8960u];
  sc=ub[9216u]; prod=ub[13312u]; tmp=ub[13568u]; red=ub[13824u];
  bv=ub[14080u]; bw=ub[14592u]; result=ub[15104u];
  mask_ge=ub[15616u]; mask_eq=ub[15680u];
  indices=ub[15744u].ReinterpretCast<uint32_t>();
  vown=ub[15808u]; wown=ub[15872u]; hand=ub[15936u];
  bv_next=ub[19840u]; bw_next=ub[19904u];
  for(uint32_t rem=0;rem<8u;++rem) for(uint32_t i=0;i<8u;++i) {
   mask_ge.SetValue(rem*8u+i,(i>=rem)?1.0f:0.0f);
   mask_eq.SetValue(rem*8u+i,(i==rem)?1.0f:0.0f);
  }
  if(active) {
   for(uint32_t i=0;i<32u;++i) {
    const uint32_t gr=(i>>3u)*64u+lane*8u+(i&7u);
    indices.SetValue(i,gr*4u);
   }
   auto ix=hand.ReinterpretCast<uint32_t>();
   for(uint32_t i=0;i<256u;++i) ix.SetValue(i,i*32u);
  }
  stage_fence<HardEvent::S_V>();
 }
 __aicore__ inline void Run() {
  const uint32_t base=group*65536u;
  if(active) {
   Duplicate(result,0.0f,512u);
   auto ix=hand.ReinterpretCast<uint32_t>();
   for(uint32_t r0=lane*8u;r0<256u;r0+=64u) {
    DataCopy(mat[localrow(r0)*256u],in[base+r0*256u],2048u);
    const uint32_t start=(uplo==0u)?r0+8u:0u;
    const uint32_t cnt=(uplo==0u)?256u-(r0+8u):r0;
    if(cnt>0u) DataCopy(sc,in[base+start*256u+r0],
                        DataCopyParams{static_cast<uint16_t>(cnt),1u,31u,0u});
    stage_fence<HardEvent::MTE2_S>();
    for(uint32_t i=0;i<8u;++i) for(uint32_t j=i+1u;j<8u;++j) {
     const uint32_t ri=localrow(r0+i)*256u+r0+j;
     const uint32_t rj=localrow(r0+j)*256u+r0+i;
     if(uplo==0u) mat.SetValue(ri,mat.GetValue(rj));
     else mat.SetValue(rj,mat.GetValue(ri));
    }
    stage_fence<HardEvent::S_V>();
    if(cnt>0u) for(uint32_t j=0;j<8u;++j)
     Gather(mat[localrow(r0+j)*256u+start],sc,ix,j*4u,cnt);
    stage_fence<HardEvent::V_MTE2>();
   }
   float local_max=0.0f;
   for(uint32_t off=0;off<8192u;off+=3840u) {
    const uint32_t count=(8192u-off<3840u)?8192u-off:3840u;
    Abs(sc,mat[off],count); PipeBarrier<PIPE_V>();
    ReduceMax(red,sc,hand,count); stage_fence<HardEvent::V_S>();
    const float m=red.GetValue(0); if(m>local_max) local_max=m;
   }
   Duplicate(tmp,0.0f,8u); stage_fence<HardEvent::V_S>();
   tmp.SetValue(0,local_max);
   stage_fence<HardEvent::S_MTE3>(); stage_fence<HardEvent::V_MTE3>();
   DataCopy(gm[4096u+group*512u+lane*8u],tmp,8u);
   if(lane==0u) DataCopy(gm[4096u+group*512u+256u],mat,256u);
   stage_fence<HardEvent::MTE3_S>();
  }
  SyncAll<true>();
  if(active) {
   DataCopy(tmp,gm[4096u+group*512u],64u);
   DataCopy(next,gm[4096u+group*512u+256u],256u);
   stage_fence<HardEvent::MTE2_S>();
   float mx=0.0f;
   for(uint32_t i=0;i<8u;++i) { float m=tmp.GetValue(i*8u); if(m>mx)mx=m; }
   float inv=1.0f; stage_scale(mx,scale,inv);
   Muls(mat,mat,inv,8192u);
   Muls(next,next,inv,256u); PipeBarrier<PIPE_V>();
  }
  if constexpr(INIT_SHELL) {
   if(active) {
    stage_fence<HardEvent::V_MTE3>();
    for(uint32_t r=lane*8u;r<256u;r+=64u)
     DataCopy(outq[base+r*256u],mat[localrow(r)*256u],2048u);
    if(lane==0u) DataCopy(outw[group*256u],result,256u);
    stage_fence<HardEvent::MTE3_S>();
   }
   SyncAll<true>(); return;
  }
  bool la_ready=false;
  for(uint32_t k=0;k<STEPS;++k) {
   const uint32_t c0=(k+1u)&~7u, rem=(k+1u)&7u, len=n-c0;
   uint32_t blocks=0u, first_local=0u, rows=0u;
   if(active) {
   Mul(v[c0],next[c0],mask_ge[rem*8u],8u);
   if(len>8u) Adds(v[c0+8u],next[c0+8u],0.0f,len-8u);
   PipeBarrier<PIPE_V>();
   if(!la_ready) {
    Mul(prod[c0],v[c0],v[c0],len); PipeBarrier<PIPE_V>();
    if(len<=64u) WholeReduceSum(red,prod[c0],len,1,1,1,8);
    else ReduceSum(red,prod[c0],tmp,len);
   }
   la_ready=false;
   stage_fence<HardEvent::V_S>();
   const float sq=red.GetValue(0), norm=sq*rsqrt_parent(sq), diag=next.GetValue(k);
   float beta=0.0f; uint32_t tau=0u;
   if(norm!=0.0f) {
    const float x0=next.GetValue(k+1u)+0.0f;
    beta=(x0>=0.0f)?-norm:norm;
    const float inv=rsqrt_parent(sq-beta*x0);
    Muls(v[c0],v[c0],inv,len); Muls(red[16],mask_eq[rem*8u],-beta*inv,8u);
    PipeBarrier<PIPE_V>(); Add(v[c0],v[c0],red[16],8u); tau=1u;
   } else Duplicate(v[c0],0.0f,len);
   result.SetValue(k,diag); result.SetValue(96u+k,beta);
   result.SetValue(192u+k,static_cast<float>(static_cast<int32_t>(tau))); PipeBarrier<PIPE_V>();

   blocks=((n/8u)+7u-lane)/8u;
   uint32_t first_block=(k+1u)>>3u;
   first_block+=(lane+8u-(first_block&7u))&7u;
   first_local=(first_block>>3u)*8u;
   if(first_local>blocks*8u) first_local=blocks*8u;
   rows=blocks*8u-first_local;
   if(first_local>0u) Duplicate(qpart,0.0f,first_local);
   if(rows>0u) {
    const uint32_t first=(len<64u)?len:64u;
    LocalTensor<float> m=mat[first_local*n+c0];
    Mul(sc,m,v[c0],first,static_cast<uint8_t>(rows),
        BinaryRepeatParams{1,1,1,8,static_cast<uint8_t>(n/8u),0});
    for(uint32_t c=c0+64u;c<n;c+=64u) {
     PipeBarrier<PIPE_V>(); const uint32_t width=(n-c<64u)?n-c:64u;
     MulAddDst(sc,m[c-c0],v[c],width,static_cast<uint8_t>(rows),
               BinaryRepeatParams{1,1,1,8,static_cast<uint8_t>(n/8u),0});
    }
    PipeBarrier<PIPE_V>();
    WholeReduceSum(qpart[first_local],sc,first,static_cast<uint8_t>(rows),1,1,8);
   }
   const uint32_t bank=group*4u*256u+(k&1u)*2u*256u;
   const uint32_t v_lane=(owner(k+1u)+1u)&7u;
   if constexpr(HANDOFF) {
    if(lane==v_lane && c0>0u) Duplicate(v,0.0f,c0);
   }
   if(rows>0u) Gather(vown,v,indices[first_local],0u,rows);
   Brcb(bv_next,v[c0],1u,{1,8});
   PipeBarrier<PIPE_V>();
   stage_fence<HardEvent::V_MTE3>();
   DataCopy(gm[bank+lane*8u],qpart,DataCopyParams{static_cast<uint16_t>(blocks),1,0,7});
   if(owner(k+1u)==lane) DataCopy(gm[bank+256u],mat[localrow(k+1u)*n],n);
   if constexpr(HANDOFF) {
    if(lane==v_lane) DataCopy(gm[6144u+group*65536u+25600u+k*n],v,n);
   }
   Muls(prod[c0],v[c0],-0.5f,len);
   if(rows>0u) Brcb(bv,vown,static_cast<uint8_t>(rows/8u),{1,8});
   PipeBarrier<PIPE_V>();
   stage_fence<HardEvent::MTE3_S>();
   }
   SyncAll<true>();
   if(active) {
   const uint32_t bank=group*4u*256u+(k&1u)*2u*256u;
   DataCopy(qfull,gm[bank],n); DataCopy(next,gm[bank+256u],n);
   stage_fence<HardEvent::MTE2_S>();
   if(rem!=0u) { Mul(qfull[c0],qfull[c0],mask_ge[rem*8u],8u); PipeBarrier<PIPE_V>(); }

   Mul(prod[c0],prod[c0],qfull[c0],len); PipeBarrier<PIPE_V>();
   if(len<=64u) WholeReduceSum(red,prod[c0],len,1,1,1,8);
   else ReduceSum(red,prod[c0],tmp,len);
   PipeBarrier<PIPE_V>(); Muls(qfull[c0],qfull[c0],-1.0f,len);
   Brcb(tmp,red,1u,{1,8}); PipeBarrier<PIPE_V>();
   const uint32_t full=len>>6u, tail=len&63u;
   if(full>0u) Mul(prod[c0],v[c0],tmp,64u,static_cast<uint8_t>(full),BinaryRepeatParams{1,1,0,8,8,0});
   if(tail>0u) Mul(prod[c0+full*64u],v[c0+full*64u],tmp,tail,1u,BinaryRepeatParams{1,1,0,8,8,0});
   PipeBarrier<PIPE_V>(); Sub(qfull[c0],qfull[c0],prod[c0],len); PipeBarrier<PIPE_V>();

   Brcb(bw_next,qfull[c0],1u,{1,8});
   if(rows>0u) Gather(wown,qfull,indices[first_local],0u,rows);
   PipeBarrier<PIPE_V>();
   if(rows>0u) Brcb(bw,wown,static_cast<uint8_t>(rows/8u),{1,8});
   for(uint32_t c=c0;c<n;c+=64u) {
    const uint32_t width=(n-c<64u)?n-c:64u;
    MulAddDst(next[c],qfull[c],bv_next[rem*8u],width,1u,
              BinaryRepeatParams{1,1,0,static_cast<uint8_t>(n/8u),0,1});
   }
   PipeBarrier<PIPE_V>();
   for(uint32_t c=c0;c<n;c+=64u) {
    const uint32_t width=(n-c<64u)?n-c:64u;
    MulAddDst(next[c],v[c],bw_next[rem*8u],width,1u,
              BinaryRepeatParams{1,1,0,static_cast<uint8_t>(n/8u),0,1});
    if(rows>0u) {
     MulAddDst(mat[first_local*n+c],qfull[c],bv,width,static_cast<uint8_t>(rows),
               BinaryRepeatParams{1,1,0,static_cast<uint8_t>(n/8u),0,1});
    }
   }
   PipeBarrier<PIPE_V>();
   if(rows>0u) {
    for(uint32_t c=c0;c<n;c+=64u) {
     const uint32_t width=(n-c<64u)?n-c:64u;
     MulAddDst(mat[first_local*n+c],v[c],bw,width,static_cast<uint8_t>(rows),
               BinaryRepeatParams{1,1,0,static_cast<uint8_t>(n/8u),0,1});
    }
   }
   if(k+1u<STEPS) {
    const uint32_t c0_n=(k+2u)&~7u, rem_n=(k+2u)&7u, len_n=n-c0_n;
    if(rem_n==0u) {
     Mul(prod[c0_n],next[c0_n],next[c0_n],len_n);
    } else {
     Mul(prod[c0_n],next[c0_n],mask_ge[rem_n*8u],8u);
     if(len_n>8u) Mul(prod[c0_n+8u],next[c0_n+8u],next[c0_n+8u],len_n-8u);
    }
    PipeBarrier<PIPE_V>();
    if(rem_n!=0u) {
     Mul(prod[c0_n],prod[c0_n],prod[c0_n],(len_n<8u)?len_n:8u);
     PipeBarrier<PIPE_V>();
    }
    if(len_n<=64u) WholeReduceSum(red,prod[c0_n],len_n,1,1,1,8);
    else ReduceSum(red,prod[c0_n],tmp,len_n);
    la_ready=true;
   } else if(rows>0u) {
    PipeBarrier<PIPE_V>();
   }
   if constexpr(DIAG) {
    if(owner(k+1u)==lane) {
     LocalTensor<float> owned=mat[localrow(k+1u)*n+c0];
     Sub(sc,next[c0],owned,len); PipeBarrier<PIPE_V>();
     Abs(sc,sc,len); PipeBarrier<PIPE_V>();
     ReduceMax(red,sc,tmp,len); stage_fence<HardEvent::V_S>();
     result.SetValue(288u+k,red.GetValue(0)); result.SetValue(384u+k,1.0f);
    }
   }
   }
  }

  if(active) {
   stage_fence<HardEvent::V_MTE3>(); stage_fence<HardEvent::S_MTE3>();
   for(uint32_t r=lane*8u;r<256u;r+=64u) if(r>=96u)
    DataCopy(gm[6144u+group*65536u+(r-96u)*160u],mat[localrow(r)*256u+96u],
             DataCopyParams{8u,20u,12u,0u});
   if(lane==0u) {
    DataCopy(gm[6144u+group*65536u+50176u],result,288u);
    Duplicate(tmp,0.0f,8u); stage_fence<HardEvent::V_S>();
    tmp.SetValue(0,scale); stage_fence<HardEvent::S_MTE3>();
    DataCopy(gm[6144u+group*65536u+50464u],tmp,8u);
   }
   if constexpr(DIAG) {
    if(lane<4u) {
     DataCopy(gm[6144u+group*65536u+50472u+lane*96u],result[288u],96u);
     DataCopy(gm[6144u+group*65536u+50856u+lane*96u],result[384u],96u);
    }
   }
   stage_fence<HardEvent::MTE3_S>();
  }
  SyncAll<true>();
 }
 __aicore__ inline void EmitCheckpoint(LocalTensor<float>ub) {
  if(lane!=0u) return;
  const uint32_t base=group*65536u;
  const uint32_t end=DIAG?51240u:50472u;
  for(uint32_t off=0;off<end;) {
   const uint32_t count=(end-off>4096u)?4096u:end-off;
   DataCopy(ub,gm[6144u+base+off],count); stage_fence<HardEvent::MTE2_S>();
   DataCopy(outq[base+off],ub,count); stage_fence<HardEvent::MTE3_S>();
   off+=count;
  }
  DataCopy(ub,gm[6144u+base+50176u],256u); stage_fence<HardEvent::MTE2_S>();
  DataCopy(outw[group*256u],ub,256u); stage_fence<HardEvent::MTE3_S>();
  Duplicate(ub,0.0f,4096u); stage_fence<HardEvent::V_MTE3>();
  for(uint32_t off=end;off<65536u;) {
   const uint32_t count=(65536u-off>4096u)?4096u:65536u-off;
   DataCopy(outq[base+off],ub,count); off+=count;
  }
  stage_fence<HardEvent::MTE3_S>();
 }
private:
 GlobalTensor<float> in,outw,outq,gm;
 LocalTensor<float> mat,v,qpart,qfull,next,sc,prod,tmp,red,bv,bw,result,mask_ge,mask_eq,vown,wown,hand,bv_next,bw_next;
 LocalTensor<uint32_t> indices;
 uint32_t n,batch,uplo,cid,group,lane; bool active; float scale;
};



#define EIGH_SOLVER_REPEAT 1
#define EIGH_K96_PASS 1
#include "linalg_eigh.cpp"
#undef EIGH_K96_PASS
#define EIGH_K96_PASS 0
#include "linalg_eigh.cpp"
#undef EIGH_K96_PASS
#undef EIGH_SOLVER_REPEAT

template<typename DT_A,uint32_t SOLVE_MODE=0u>
__global__ __aicore__ void linalg_eigh(
GM_ADDR A,
GM_ADDR eigenvalues,
GM_ADDR eigenvectors,
GM_ADDR workspace,
GM_ADDR tiling){
REGISTER_TILING_DEFAULT(LinalgEighTilingData);
GET_TILING_DATA_WITH_STRUCT(LinalgEighTilingData,tiling_data,tiling);
TPipe pipe;
if constexpr(SOLVE_MODE==5u){
 TBuf<TPosition::VECCALC> shared;
 pipe.InitBuffer(shared,49088u*sizeof(float));
 LocalTensor<float> ub=shared.Get<float>();
 __gm__ float* ws=reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace))+tiling_data.wsOffsetFloats;
 Stage96 stage; stage.Init(A,eigenvalues,eigenvectors,reinterpret_cast<GM_ADDR>(ws),tiling_data,ub);
 stage.Run();
 LinalgEighTilingData parent=tiling_data; parent.solve_mode=3u;
 KernelLinalgEighK96<DT_A,3u> op;
 op.Init(A,eigenvalues,eigenvectors,reinterpret_cast<GM_ADDR>(ws+6144u),parent,&pipe,ub);
 op.Process();
}else{
 KernelLinalgEigh<DT_A,(SOLVE_MODE==4u?3u:SOLVE_MODE),(SOLVE_MODE==4u)>op;
 if constexpr(SOLVE_MODE==0u){
  op.Init(A,eigenvalues,eigenvectors,GetUserWorkspace(workspace),tiling_data,&pipe);
 }else{
  op.Init(A,eigenvalues,eigenvectors,workspace,tiling_data,&pipe);
 }
 op.Process();
}
}

#endif
