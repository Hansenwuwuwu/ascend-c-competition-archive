#pragma once
#include "pinv_dsl_common.h"
#pragma once
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void ba_atomic_on(D,A,B){AscendC::SetAtomicAdd<float>();}
template<class D,class A,class B> __aicore__ inline void ba_atomic_off(D,A,B){AscendC::DisableDmaAtomic();}
}

namespace tl::ascend {
__aicore__ inline void BlockInplacePanel( GM_ADDR D_handle,  GM_ADDR P_handle, int32_t blocks, int32_t start_row,  GM_ADDR shape_hint, int64_t n, int64_t batch, uint64_t fftsAddr) {
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> D;
  D.SetGlobalBuffer((__gm__ float*)D_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<uint8_t> ShapeHint;
  ShapeHint.SetGlobalBuffer((__gm__ uint8_t*)shape_hint);

  AscendC::LocalTensor<float> ascend_l0a(AscendC::TPosition::A2, 0, 16384);
  AscendC::LocalTensor<float> ascend_l0b(AscendC::TPosition::B2, 0, 16384);



  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  AscendC::LocalTensor<float> a(AscendC::TPosition::A1, 0, 4096);
  AscendC::LocalTensor<float> b(AscendC::TPosition::A1, 16384, 8192);
  AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, 0, 8192);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t iteration = 0; iteration < ((((batch * (n / 128)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < (batch * (n / 128))) {
        tl::ascend::br_load_a_shape(a[0], D[(((bid + (iteration * blocks)) / (n / 128)) * 4096)], a[0], 64, 64, 64);
        tl::ascend::copy_gm_to_l1<float, 64, 128>(b[0], P[((((bid + (iteration * blocks)) % (n / 128)) * 128) + (((((bid + (iteration * blocks)) / (n / 128)) * n) + start_row) * n))], n, ((64 <= ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row)) ? 64 : ((0 < ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row)) ? ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row) : 0)), ((1 <= ((n / 128) - ((bid + (iteration * blocks)) % (n / 128)))) ? 128 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 128)) * 128))) ? (n - (((bid + (iteration * blocks)) % (n / 128)) * 128)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::gemm_v0<float, float, 64, 128, 64, false, false>(a[0], b[0], c[0], ascend_l0a, ascend_l0b, (bool)1);
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 0>(P[((((bid + (iteration * blocks)) % (n / 128)) * 128) + (((((bid + (iteration * blocks)) / (n / 128)) * n) + start_row) * n))], c[0], n, ((64 <= ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row)) ? 64 : ((0 < ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row)) ? ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row) : 0)), ((1 <= ((n / 128) - ((bid + (iteration * blocks)) % (n / 128)))) ? 128 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 128)) * 128))) ? (n - (((bid + (iteration * blocks)) % (n / 128)) * 128)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
      }
    }
  }

}

}
namespace tl::ascend {
__aicore__ inline void BlockAtomicRows( GM_ADDR F_handle,  GM_ADDR P_handle, int32_t blocks, int32_t start_row,  GM_ADDR shape_hint, int64_t n, int64_t batch, uint64_t fftsAddr) {
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> F;
  F.SetGlobalBuffer((__gm__ float*)F_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<uint8_t> ShapeHint;
  ShapeHint.SetGlobalBuffer((__gm__ uint8_t*)shape_hint);

  AscendC::LocalTensor<float> ascend_l0a(AscendC::TPosition::A2, 0, 16384);
  AscendC::LocalTensor<float> ascend_l0b(AscendC::TPosition::B2, 0, 16384);



  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  AscendC::LocalTensor<float> a(AscendC::TPosition::A1, 0, 4096);
  AscendC::LocalTensor<float> b(AscendC::TPosition::A1, 16384, 16384);
  AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, 0, 16384);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t iteration = 0; iteration < (((((batch * (n / 64)) * (n / 256)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < ((batch * (n / 64)) * (n / 256))) {
        if (((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) != start_row) {
          tl::ascend::br_load_a_shape(a[0], F[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], a[0], 64, 64, 64);
          tl::ascend::copy_gm_to_l1<float, 64, 256>(b[0], P[((((bid + (iteration * blocks)) % (n / 256)) * 256) + (((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) + start_row) * n))], n, ((64 <= ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - start_row)) ? 64 : ((0 < ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - start_row)) ? ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - start_row) : 0)), ((1 <= ((n / 256) - ((bid + (iteration * blocks)) % (n / 256)))) ? 256 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 256)) * 256))) ? (n - (((bid + (iteration * blocks)) % (n / 256)) * 256)) : 0)));
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::gemm_v0<float, float, 64, 256, 64, false, false>(a[0], b[0], c[0], ascend_l0a, ascend_l0b, (bool)1);
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::ba_atomic_on(P[0], P[0], P[0]);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 256, 0>(P[((((bid + (iteration * blocks)) % (n / 256)) * 256) + ((((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) + (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n)) * n))], c[0], n, ((1 <= (((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) / 64) - (((bid + (iteration * blocks)) / (n / 256)) % (n / 64)))) ? 64 : ((0 < ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - ((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64))) ? ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - ((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64)) : 0)), ((1 <= ((n / 256) - ((bid + (iteration * blocks)) % (n / 256)))) ? 256 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 256)) * 256))) ? (n - (((bid + (iteration * blocks)) % (n / 256)) * 256)) : 0)));
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::ba_atomic_off(P[0], P[0], P[0]);
        }
      }
    }
  }

}

}
#define br_inplace_panel(d,a,b,oa,op,bl,ba,n,st) BlockInplacePanel((GM_ADDR)(d.GetPhyAddr()+oa),(GM_ADDR)(d.GetPhyAddr()+op),bl,st,(GM_ADDR)d.GetPhyAddr(),n,ba,fftsAddr)
#define br_atomic_rows(d,a,b,oa,op,bl,ba,n,st) BlockAtomicRows((GM_ADDR)(d.GetPhyAddr()+oa),(GM_ADDR)(d.GetPhyAddr()+op),bl,st,(GM_ADDR)d.GetPhyAddr(),n,ba,fftsAddr)
namespace tl::ascend {
template<class D,class S,class X> __aicore__ inline void bc_store(D d,S s,X,int m,int n,int stride){AscendC::FixpipeParamsV220 p;p.nSize=n;p.mSize=m;p.srcStride=m;p.dstStride=stride;p.ndNum=1;p.srcNdStride=0;p.dstNdStride=0;AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(d,s,p);}
}

namespace tl::ascend {
__aicore__ inline void BlockChainRows( GM_ADDR F_handle,  GM_ADDR D_handle,  GM_ADDR P_handle,  GM_ADDR Tmp_handle,  GM_ADDR Y_handle, int32_t blocks, int32_t start_row,  GM_ADDR shape_hint, int64_t n, int64_t batch, uint64_t fftsAddr) {
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> F;
  F.SetGlobalBuffer((__gm__ float*)F_handle);
  AscendC::GlobalTensor<float> D;
  D.SetGlobalBuffer((__gm__ float*)D_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<float> Tmp;
  Tmp.SetGlobalBuffer((__gm__ float*)Tmp_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<uint8_t> ShapeHint;
  ShapeHint.SetGlobalBuffer((__gm__ uint8_t*)shape_hint);

  AscendC::LocalTensor<float> ascend_l0a(AscendC::TPosition::A2, 0, 16384);
  AscendC::LocalTensor<float> ascend_l0b(AscendC::TPosition::B2, 0, 16384);



  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  AscendC::LocalTensor<float> f(AscendC::TPosition::A1, 0, 4096);
  AscendC::LocalTensor<float> inv(AscendC::TPosition::A1, 16384, 4096);
  AscendC::LocalTensor<float> c1(AscendC::TPosition::CO1, 0, 4096);
  AscendC::LocalTensor<float> a(AscendC::TPosition::A1, 32768, 4096);
  AscendC::LocalTensor<float> p(AscendC::TPosition::A1, 49152, 16384);
  AscendC::LocalTensor<float> c2(AscendC::TPosition::CO1, 16384, 16384);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t iteration = 0; iteration < (((((batch * (n / 64)) * (n / 256)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < ((batch * (n / 64)) * (n / 256))) {
        if (((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) != start_row) {
          tl::ascend::br_load_a_shape(f[0], F[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], f[0], 64, 64, 64);
          tl::ascend::copy_gm_to_l1<float, 64, 64>(inv[0], D[(((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * 4096)], 64, ((1 <= (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) ? 64 : (((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) - batch) < 0) ? ((batch * 64) - (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * 64)) : 0)), 64);
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::gemm_v0<float, float, 64, 64, 64, false, false>(f[0], inv[0], c1[0], ascend_l0a, ascend_l0b, (bool)1);
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::bc_store(Tmp[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], c1[0], Tmp[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], 64, 64, 64);
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::br_load_a_shape(a[0], Tmp[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], a[0], 64, 64, 64);
        } else {
          tl::ascend::br_load_a_shape(a[0], D[(((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * 4096)], a[0], 64, 64, 64);
        }
        tl::ascend::copy_gm_to_l1<float, 64, 256>(p[0], P[((((bid + (iteration * blocks)) % (n / 256)) * 256) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], n, ((1 <= (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) ? 64 : (((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) - batch) < 0) ? ((batch * 64) - (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * 64)) : 0)), ((1 <= ((n / 256) - ((bid + (iteration * blocks)) % (n / 256)))) ? 256 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 256)) * 256))) ? (n - (((bid + (iteration * blocks)) % (n / 256)) * 256)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::gemm_v0<float, float, 64, 256, 64, false, false>(a[0], p[0], c2[0], ascend_l0a, ascend_l0b, (bool)1);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) != start_row) {
          tl::ascend::ba_atomic_on(Y[0], Y[0], Y[0]);
        }
        tl::ascend::bc_store(Y[((((bid + (iteration * blocks)) % (n / 256)) * 256) + ((((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) + (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n)) * n))], c2[0], Y[((((bid + (iteration * blocks)) % (n / 256)) * 256) + ((((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) + (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n)) * n))], 64, 256, n);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) != start_row) {
          tl::ascend::ba_atomic_off(Y[0], Y[0], Y[0]);
        }
      }
    }
  }

}

}
#define br_chain(d,a,b,fo,di,pi,tm,sw,bl,ba,n,st) BlockChainRows((GM_ADDR)(d.GetPhyAddr()+fo),(GM_ADDR)(d.GetPhyAddr()+di),(GM_ADDR)(d.GetPhyAddr()+pi),(GM_ADDR)(d.GetPhyAddr()+tm),(GM_ADDR)(d.GetPhyAddr()+sw),bl,st,(GM_ADDR)d.GetPhyAddr(),n,ba,fftsAddr)

#ifndef PINV_BLOCK_LOOKAHEAD_H
#define PINV_BLOCK_LOOKAHEAD_H
#include "kernel_operator.h"
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void br_late(D,A,B){AscendC::CrossCoreSetFlag<2,PIPE_FIX>(12);}
}
#endif

namespace tl::ascend {
__aicore__ inline void BlockLookaheadRows( GM_ADDR F_handle,  GM_ADDR D_handle,  GM_ADDR P_handle,  GM_ADDR Tmp_handle,  GM_ADDR Y_handle, int32_t blocks, int32_t start_row,  GM_ADDR shape_hint, int64_t n, int64_t batch, uint64_t fftsAddr) {
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> F;
  F.SetGlobalBuffer((__gm__ float*)F_handle);
  AscendC::GlobalTensor<float> D;
  D.SetGlobalBuffer((__gm__ float*)D_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<float> Tmp;
  Tmp.SetGlobalBuffer((__gm__ float*)Tmp_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<uint8_t> ShapeHint;
  ShapeHint.SetGlobalBuffer((__gm__ uint8_t*)shape_hint);

  AscendC::LocalTensor<float> ascend_l0a(AscendC::TPosition::A2, 0, 16384);
  AscendC::LocalTensor<float> ascend_l0b(AscendC::TPosition::B2, 0, 16384);



  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  AscendC::LocalTensor<float> f(AscendC::TPosition::A1, 0, 4096);
  AscendC::LocalTensor<float> inv(AscendC::TPosition::A1, 16384, 4096);
  AscendC::LocalTensor<float> c1(AscendC::TPosition::CO1, 0, 4096);
  AscendC::LocalTensor<float> a(AscendC::TPosition::A1, 32768, 4096);
  AscendC::LocalTensor<float> lead(AscendC::TPosition::A1, 49152, 4096);
  AscendC::LocalTensor<float> c2(AscendC::TPosition::CO1, 16384, 4096);
  AscendC::LocalTensor<float> tail(AscendC::TPosition::A1, 65536, 12288);
  AscendC::LocalTensor<float> c3(AscendC::TPosition::CO1, 32768, 12288);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    if (bid < (batch * (n / 64))) {
      if (((bid % (n / 64)) * 64) != start_row) {
        tl::ascend::br_load_a_shape(f[0], F[(((bid % (n / 64)) * 4096) + (((bid / (n / 64)) * n) * 64))], f[0], 64, 64, 64);
        tl::ascend::copy_gm_to_l1<float, 64, 64>(inv[0], D[((bid / (n / 64)) * 4096)], 64, ((1 <= (batch - (bid / (n / 64)))) ? 64 : ((((bid / (n / 64)) - batch) < 0) ? ((batch * 64) - ((bid / (n / 64)) * 64)) : 0)), 64);
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::gemm_v0<float, float, 64, 64, 64, false, false>(f[0], inv[0], c1[0], ascend_l0a, ascend_l0b, (bool)1);
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::bc_store(Tmp[(((bid % (n / 64)) * 4096) + (((bid / (n / 64)) * n) * 64))], c1[0], Tmp[(((bid % (n / 64)) * 4096) + (((bid / (n / 64)) * n) * 64))], 64, 64, 64);
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::br_load_a_shape(a[0], Tmp[(((bid % (n / 64)) * 4096) + (((bid / (n / 64)) * n) * 64))], a[0], 64, 64, 64);
      } else {
        tl::ascend::br_load_a_shape(a[0], D[((bid / (n / 64)) * 4096)], a[0], 64, 64, 64);
      }
      tl::ascend::copy_gm_to_l1<float, 64, 64>(lead[0], P[(((bid / (n / 64)) * n) * 64)], n, ((1 <= (batch - (bid / (n / 64)))) ? 64 : ((((bid / (n / 64)) - batch) < 0) ? ((batch * 64) - ((bid / (n / 64)) * 64)) : 0)), ((64 <= n) ? 64 : ((0 < n) ? n : 0)));
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::gemm_v0<float, float, 64, 64, 64, false, false>(a[0], lead[0], c2[0], ascend_l0a, ascend_l0b, (bool)1);
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((bid % (n / 64)) * 64) != start_row) {
        tl::ascend::ba_atomic_on(Y[0], Y[0], Y[0]);
      }
      tl::ascend::bc_store(Y[((((bid % (n / 64)) * 64) + ((bid / (n / 64)) * n)) * n)], c2[0], Y[((((bid % (n / 64)) * 64) + ((bid / (n / 64)) * n)) * n)], 64, 64, n);
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((bid % (n / 64)) * 64) != start_row) {
        tl::ascend::ba_atomic_off(Y[0], Y[0], Y[0]);
      }
    }
    tl::ascend::pc_done(Y[0], Y[0], Y[0]);
    if (bid < (batch * (n / 64))) {
      tl::ascend::copy_gm_to_l1<float, 64, 192>(tail[0], P[((((bid / (n / 64)) * n) * 64) + 64)], n, ((1 <= (batch - (bid / (n / 64)))) ? 64 : ((((bid / (n / 64)) - batch) < 0) ? ((batch * 64) - ((bid / (n / 64)) * 64)) : 0)), ((256 <= n) ? 192 : ((64 < n) ? (n - 64) : 0)));
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::gemm_v0<float, float, 64, 192, 64, false, false>(a[0], tail[0], c3[0], ascend_l0a, ascend_l0b, (bool)1);
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((bid % (n / 64)) * 64) != start_row) {
        tl::ascend::ba_atomic_on(Y[0], Y[0], Y[0]);
      }
      tl::ascend::bc_store(Y[(((((bid % (n / 64)) * 64) + ((bid / (n / 64)) * n)) * n) + 64)], c3[0], Y[(((((bid % (n / 64)) * 64) + ((bid / (n / 64)) * n)) * n) + 64)], 64, 192, n);
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((bid % (n / 64)) * 64) != start_row) {
        tl::ascend::ba_atomic_off(Y[0], Y[0], Y[0]);
      }
    }
  }

}
}
#define br_lookahead_chain(d,a,b,fo,di,pi,tm,sw,bl,ba,n,st) BlockLookaheadRows((GM_ADDR)(d.GetPhyAddr()+fo),(GM_ADDR)(d.GetPhyAddr()+di),(GM_ADDR)(d.GetPhyAddr()+pi),(GM_ADDR)(d.GetPhyAddr()+tm),(GM_ADDR)(d.GetPhyAddr()+sw),bl,st,(GM_ADDR)d.GetPhyAddr(),n,ba,fftsAddr)
namespace tl::ascend {
__aicore__ inline void BlockRaw( GM_ADDR A_handle,  GM_ADDR Y_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, uint64_t fftsAddr) {

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int lookahead = 0;
  int row_tile = 0;
  int source = 0;
  int swapped = 0;
  int cube_swap = 0;
  float magnitude = 0.000000e+00f;
  int begin = 0;
  int count = 0;
  float magnitude_1 = 0.000000e+00f;
  float power = 0.000000e+00f;
  int quick_index = 0;
  int quick_active = 0;
  int admitted = 0;
  float cost = 0.000000e+00f;
  int chosen = 0;
  float best = 0.000000e+00f;
  int original = 0;
  auto vid = AscendC::GetSubBlockIdx();
  lookahead = 0;
  if ((batch * n) == (blocks * 64)) {
    lookahead = 1;
  }
  row_tile = 8;
  if ((blocks * 64) <= (batch * n)) {
    row_tile = 32;
  }
  source = 0;
  swapped = (((batch * n) * n) * 2);
  if ASCEND_IS_AIC {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    cube_swap = (((batch * n) * n) * 2);
    for (int32_t stage = 0; stage < (n / 64); ++stage) {
      cube_swap = (((batch * n) * n) * 2);
      if (0 < (stage % 2)) {
        cube_swap = 0;
      }
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 8);
      if (lookahead != 0) {
        tl::ascend::br_lookahead_chain(W[0], W[0], W[0], (((batch * 4096) + ((batch * n) * 384)) + (((batch * n) * n) * 4)), (((batch * n) * 128) + (((batch * n) * n) * 4)), (((batch * 4096) + ((batch * n) * 128)) + (((batch * n) * n) * 4)), (((batch * 4096) + ((batch * n) * 256)) + (((batch * n) * n) * 4)), cube_swap, blocks, batch, n, (stage * 64));
      } else {
        tl::ascend::br_chain(W[0], W[0], W[0], (((batch * 4096) + ((batch * n) * 384)) + (((batch * n) * n) * 4)), (((batch * n) * 128) + (((batch * n) * n) * 4)), (((batch * 4096) + ((batch * n) * 128)) + (((batch * n) * n) * 4)), (((batch * 4096) + ((batch * n) * 256)) + (((batch * n) * n) * 4)), cube_swap, blocks, batch, n, (stage * 64));
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      if (lookahead != 0) {
        tl::ascend::br_late(W[0], W[0], W[0]);
      } else {
        tl::ascend::pc_done(W[0], W[0], W[0]);
      }
    }
  }
  if ASCEND_IS_AIV {
    AscendC::TPipe &pipe=*::GetTPipePtr();

    AscendC::LocalTensor<float> pool(AscendC::TPosition::VECCALC, 0, 24576);
    tl::ascend::fr_bind(A[0], Y[0], W[0]);
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    if (bid < (batch * ((blocks * 2) / batch))) {
      magnitude = 0.000000e+00f;
      for (int32_t chunk = 0; chunk < ((((((blocks * 2) / batch) * 4096) + (n * n)) - 1) / (((blocks * 2) / batch) * 4096)); ++chunk) {
        begin = (((bid % ((blocks * 2) / batch)) * 4096) + ((chunk * ((blocks * 2) / batch)) * 4096));
        if (begin < (n * n)) {
          count = ((n * n) - begin);
          if (4096 < count) {
            count = 4096;
          }
          tl::ascend::pv_load_rows(pool[0], A[((((bid / ((blocks * 2) / batch)) * n) * n) + begin)], pool[0], 1, count, 0, 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          tl::ascend::pv_abs(pool[0], pool[0], pool[0], count);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_reduce_max(pool[16384], pool[0], pool[4096], count);
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          magnitude = max(magnitude, pool.GetValue(16384));
        }
      }
      pool.SetValue(0, magnitude);
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(S[((batch * 32) + (bid * 8))], pool[0], S[((batch * 32) + (bid * 8))], 1, 8, 0, 0);
      tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    if (bid < batch) {
      tl::ascend::pv_load_rows(pool[0], S[((batch * 32) + ((bid * ((blocks * 2) / batch)) * 8))], pool[0], 1, (((blocks * 2) / batch) * 8), 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      magnitude_1 = 0.000000e+00f;
      for (int32_t lane = 0; lane < ((blocks * 2) / batch); ++lane) {
        magnitude_1 = max(magnitude_1, pool.GetValue((((int64_t)lane) * (int64_t)8)));
      }
      power = 1.000000e+00f;
      if ((0.000000e+00f < magnitude_1) && (magnitude_1 < 1.000000e+30f)) {
        while (1) {
          if (!((power < magnitude_1))) { break; }
          power = (power * 2.000000e+00f);
        }
        while (1) {
          if (!((magnitude_1 <= (power * 5.000000e-01f)))) { break; }
          power = (power * 5.000000e-01f);
        }
      }
      pool.SetValue(0, power);
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(S[(bid * 32)], pool[0], S[(bid * 32)], 1, 8, 0, 0);
      tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    tl::ascend::pv_load_rows(pool[20480], S[0], pool[20480], 1, (batch * 32), 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t iteration = 0; iteration < ((((blocks * 2) + ((batch * n) / row_tile)) - 1) / (blocks * 2)); ++iteration) {
      int32_t row = ((((iteration * blocks) * 2) + bid) * row_tile);
      if (row < (batch * n)) {
        float power_1 = pool.GetValue((((((int64_t)row) / ((int64_t)n)) * (int64_t)32) + (int64_t)20480));
        tl::ascend::pv_load_rows(pool[0], A[(row * n)], pool[0], row_tile, n, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_1), (row_tile * n));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(W[(source + (row * n))], pool[0], W[(source + (row * n))], row_tile, n, 0, 0);
        tl::ascend::pv_store_rows(W[((((batch * n) * n) * 11) + (row * n))], pool[0], W[((((batch * n) * n) * 11) + (row * n))], row_tile, n, 0, 0);
        tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
      }
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    for (int32_t stage_1 = 0; stage_1 < (n / 64); ++stage_1) {
      source = 0;
      swapped = (((batch * n) * n) * 2);
      if (0 < (stage_1 % 2)) {
        source = (((batch * n) * n) * 2);
        swapped = 0;
      }
      quick_index = bid;
      quick_active = ((int32_t)(bid < (batch * ((n / 64) - stage_1))));
      if (lookahead != 0) {
        quick_index = (((((bid / 2) / (n / 64)) * ((n / 64) - stage_1)) + ((bid / 2) % (n / 64))) - stage_1);
        quick_active = 1;
        if ((0 < (bid % 2)) || (((bid / 2) % (n / 64)) < stage_1)) {
          quick_active = 0;
        }
      }
      if (quick_active != 0) {
        int32_t job = (quick_index / ((n / 64) - stage_1));
        int32_t position = (((quick_index % ((n / 64) - stage_1)) * 64) + (stage_1 * 64));
        tl::ascend::br_quick_strided(W[0], W[0], W[0], ((source + ((job * n) * n)) + (position * n)), (((quick_index * 4096) + ((batch * n) * 64)) + (((batch * n) * n) * 4)), ((((batch * 4096) + ((batch * n) * 449)) + (quick_index * 32)) + (((batch * n) * n) * 6)), blocks, n);
        admitted = ((int32_t)W.GetValue(((((((int64_t)batch) * (int64_t)4096) + ((((int64_t)batch) * ((int64_t)n)) * (int64_t)449)) + (((int64_t)quick_index) * (int64_t)32)) + (((((int64_t)batch) * ((int64_t)n)) * ((int64_t)n)) * (int64_t)6))));
        cost = 1.000000e+30f;
        if (admitted != 0) {
          cost = W.GetValue((((((((int64_t)batch) * (int64_t)4096) + ((((int64_t)batch) * ((int64_t)n)) * (int64_t)449)) + (((int64_t)quick_index) * (int64_t)32)) + (((((int64_t)batch) * ((int64_t)n)) * ((int64_t)n)) * (int64_t)6)) + (int64_t)1));
        }
        pool.SetValue(0, cost);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        tl::ascend::pv_store_rows(W[((((batch * 4096) + ((batch * n) * 448)) + (quick_index * 8)) + (((batch * n) * n) * 6))], pool[0], W[((((batch * 4096) + ((batch * n) * 448)) + (quick_index * 8)) + (((batch * n) * n) * 6))], 1, 8, 0, 0);
        tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
      }
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      if (lookahead != 0) {
        if (0 < stage_1) {
          tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 12);
        }
      }
      tl::ascend::pv_load_rows(pool[18432], W[(((batch * 4096) + ((batch * n) * 448)) + (((batch * n) * n) * 6))], pool[18432], 1, ((batch * ((n / 64) - stage_1)) * 8), 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      for (int32_t iteration_1 = 0; iteration_1 < ((((blocks * 2) + ((batch * n) / row_tile)) - 1) / (blocks * 2)); ++iteration_1) {
        int32_t row_1 = ((((iteration_1 * blocks) * 2) + bid) * row_tile);
        if (row_1 < (batch * n)) {
          chosen = 0;
          best = 1.000000e+30f;
          for (int32_t candidate = 0; candidate < ((n / 64) - stage_1); ++candidate) {
            float cost_1 = pool.GetValue((((((((int64_t)row_1) / ((int64_t)n)) * ((((int64_t)n) / (int64_t)64) - ((int64_t)stage_1))) * (int64_t)8) + (((int64_t)candidate) * (int64_t)8)) + (int64_t)18432));
            if ((0.000000e+00f <= cost_1) && (cost_1 < best)) {
              chosen = candidate;
              best = cost_1;
            }
          }
          int32_t position_1 = ((chosen * 64) + (stage_1 * 64));
          if ((row_1 % n) == (stage_1 * 64)) {
            pool.SetValue(0, best);
            pool.SetValue(1, ((float)position_1));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(S[((((stage_1 * batch) * 32) + ((row_1 / n) * 32)) + 512)], pool[0], S[((((stage_1 * batch) * 32) + ((row_1 / n) * 32)) + 512)], 1, 8, 0, 0);
            tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          }
          original = (row_1 % n);
          if (((stage_1 * 64) <= (row_1 % n)) && ((row_1 % n) < ((stage_1 * 64) + 64))) {
            original = ((position_1 + (row_1 % n)) - (stage_1 * 64));
          } else {
            if ((position_1 <= (row_1 % n)) && ((row_1 % n) < (position_1 + 64))) {
              original = (((stage_1 * 64) + (row_1 % n)) - position_1);
            }
          }
          tl::ascend::pv_load_rows(pool[16384], W[((source + (((row_1 / n) * n) * n)) + (original * n))], pool[16384], row_tile, 64, (n - 64), 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          if (((stage_1 * 64) <= (row_1 % n)) && ((row_1 % n) < ((stage_1 * 64) + 64))) {
            tl::ascend::pv_fill(pool[16384], pool[16384], pool[16384], 0.000000e+00f, (row_tile * 64));
          } else {
            tl::ascend::pv_scale(pool[16384], pool[16384], pool[16384], -1.000000e+00f, (row_tile * 64));
          }
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(W[((((batch * 4096) + ((batch * n) * 384)) + (row_1 * 64)) + (((batch * n) * n) * 4))], pool[16384], W[((((batch * 4096) + ((batch * n) * 384)) + (row_1 * 64)) + (((batch * n) * n) * 4))], row_tile, 64, 0, 0);
          tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (row_tile * n));
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
          tl::ascend::pv_load_rows(pool[0], W[(((source + (((row_1 / n) * n) * n)) + (original * n)) + 64)], pool[0], row_tile, (n - 64), 64, 64);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
          if (((stage_1 * 64) <= (row_1 % n)) && ((row_1 % n) < ((stage_1 * 64) + 64))) {
            for (int32_t lane_1 = 0; lane_1 < row_tile; ++lane_1) {
              pool.SetValue((((((((int64_t)n) * (((int64_t)lane_1) + (int64_t)1)) + (((int64_t)row_1) % ((int64_t)n))) + ((int64_t)lane_1)) - (((int64_t)stage_1) * (int64_t)64)) - (int64_t)64), 1.000000e+00f);
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
          tl::ascend::pv_store_rows(W[(swapped + (row_1 * n))], pool[0], W[(swapped + (row_1 * n))], row_tile, n, 0, 0);
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          if (((stage_1 * 64) <= (row_1 % n)) && ((row_1 % n) < ((stage_1 * 64) + 64))) {
            tl::ascend::pv_store_rows(W[(((((batch * 4096) + ((batch * n) * 128)) + (((row_1 / n) * n) * 64)) + (((batch * n) * n) * 4)) + (((row_1 % n) - (stage_1 * 64)) * n))], pool[0], W[(((((batch * 4096) + ((batch * n) * 128)) + (((row_1 / n) * n) * 64)) + (((batch * n) * n) * 4)) + (((row_1 % n) - (stage_1 * 64)) * n))], row_tile, n, 0, 0);
            tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
            tl::ascend::pv_load_rows(pool[16384], W[((((((((row_1 / n) * ((n / 64) - stage_1)) * 4096) + (chosen * 4096)) + ((row_1 % n) * 64)) + ((batch * n) * 64)) + (((batch * n) * n) * 4)) - (stage_1 * 4096))], pool[16384], row_tile, 64, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            tl::ascend::pv_store_rows(W[((((((row_1 / n) * 4096) + ((batch * n) * 128)) + ((row_1 % n) * 64)) + (((batch * n) * n) * 4)) - (stage_1 * 4096))], pool[16384], W[((((((row_1 / n) * 4096) + ((batch * n) * 128)) + ((row_1 % n) * 64)) + (((batch * n) * n) * 4)) - (stage_1 * 4096))], row_tile, 64, 0, 0);
            tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          }
        }
      }
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
      if (lookahead == 0) {
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      }
    }
    if (lookahead != 0) {
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 12);
    }
    source = 0;
    if (63 < (n % 128)) {
      source = (((batch * n) * n) * 2);
    }
    tl::ascend::pv_load_rows(pool[22016], S[512], pool[22016], 1, 512, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t iteration_2 = 0; iteration_2 < ((((blocks * 2) + ((batch * n) / row_tile)) - 1) / (blocks * 2)); ++iteration_2) {
      int32_t row_2 = ((((iteration_2 * blocks) * 2) + bid) * row_tile);
      if (row_2 < (batch * n)) {
        for (int32_t col = 0; col < (n / 64); ++col) {
          pool.SetValue((col + 21200), ((float)col));
        }
        for (int32_t stage_2 = 0; stage_2 < (n / 64); ++stage_2) {
          int32_t chosen_1 = (((int32_t)pool.GetValue(((((((int64_t)stage_2) * ((int64_t)batch)) * (int64_t)32) + ((((int64_t)row_2) / ((int64_t)n)) * (int64_t)32)) + (int64_t)22017))) / 64);
          float old = pool.GetValue((stage_2 + 21200));
          pool.SetValue((stage_2 + 21200), pool.GetValue((((int64_t)chosen_1) + (int64_t)21200)));
          pool.SetValue((((int64_t)chosen_1) + (int64_t)21200), old);
        }
        for (int32_t col_1 = 0; col_1 < (n / 64); ++col_1) {
          int32_t original_1 = ((int32_t)pool.GetValue((col_1 + 21200)));
          tl::ascend::pv_load_rows(pool[0], W[(((col_1 * 64) + source) + (row_2 * n))], pool[0], row_tile, 64, (n - 64), 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
          tl::ascend::pv_store_rows(W[(((original_1 * 64) + (((batch * n) * n) * 8)) + (row_2 * n))], pool[0], W[(((original_1 * 64) + (((batch * n) * n) * 8)) + (row_2 * n))], row_tile, 64, 0, (n - 64));
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
        }
      }
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);

  }
  AscendC::PipeBarrier<PIPE_ALL>();
}
}
