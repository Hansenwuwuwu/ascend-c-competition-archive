#pragma once
#include "pinv_dsl_common.h"
namespace tl::ascend {
__aicore__ static __attribute__((noinline)) void BlockQuick( GM_ADDR A_handle,  GM_ADDR Y_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t job, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
  AscendC::TPipe &pipe=*::GetTPipePtr();

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);


  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  AscendC::LocalTensor<int> accepted(AscendC::TPosition::VECCALC, 0, 8);
  int count = 0;
  int dma_pad = 0;
  float condition = 0.000000e+00f;
  int valid = 0;
  AscendC::LocalTensor<float> pool(AscendC::TPosition::VECCALC, 32, 23296);
  float power = 0.000000e+00f;
  AscendC::LocalTensor<uint32_t> indices(AscendC::TPosition::VECCALC, 93216, 64);
  int previous = 0;
  float recovered = 0.000000e+00f;
  AscendC::LocalTensor<uint32_t> order(AscendC::TPosition::VECCALC, 93472, 128);
  int rejected = 0;
  float best = 0.000000e+00f;
  int pivot = 0;
  int consumed = 0;
  int base = 0;
  int active = 0;
  float best_1 = 0.000000e+00f;
  int pivot_1 = 0;
  int consumed_1 = 0;
  int base_1 = 0;
  int active_1 = 0;
  int word = 0;
  float metric = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    accepted.SetValue(0, 0);
    count = 5120;
    dma_pad = 64;
    condition = 1.000000e+30f;
    valid = 1;
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[0], A[(job * 4096)], pool[0], 64, 64, max((m - 64), 0), (80 - dma_pad), (bool)0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    power = 1.000000e+00f;
    if (valid != 0) {
      tl::ascend::pv_mul(pool[8704], pool[0], pool[0], count);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(pool[23168], pool[8704], pool[22656], count);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      float total = pool.GetValue(23168);
      tl::ascend::lv_index_iota(indices[0], indices[0], indices[0], 64);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::lv_index_scale(indices[0], indices[0], indices[0], 320, 64);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_fill(pool[23200], pool[23200], pool[23200], 0.000000e+00f, 64);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_mask_begin(pool[0], pool[0], pool[0], 64);
      previous = 23265;
      recovered = 0.000000e+00f;
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      for (int32_t pivot_tile = 0; pivot_tile < 64; ++pivot_tile) {
        tl::ascend::pv_gather(pool[23200], pool[0], indices[0], (pivot_tile * 4), 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_abs(pool[22592], pool[23200], pool[22592], 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mask_reduce(pool[23168], pool[22592], pool[23168]);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best, pivot);
        if (best <= 0.000000e+00f) {
          valid = 0;
          break;
        }
        tl::ascend::pv_mask_remove(pool[0], pool[0], pool[0], pivot);
        order.SetValue(pivot, ((uint32_t)(pivot_tile * 4)));
        order.SetValue((pivot_tile + 64), ((uint32_t)(pivot * 320)));
        consumed = 0;
        base = 0;
        float divisor = pool.GetValue((((int64_t)pivot) + (int64_t)23200));
        pool.SetValue(((((int64_t)pivot) * (int64_t)80) + ((int64_t)pivot_tile)), (divisor + 1.000000e+00f));
        pool.SetValue(((((int64_t)previous) + ((int64_t)pivot_tile)) - (int64_t)1), recovered);
        recovered = (-1.000000e+00f / divisor);
        previous = (pivot * 80);
        pool.SetValue((((int64_t)pivot) + (int64_t)23200), 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        active = 64;
        if (consumed != 0) {
          active = (active - 8);
        }
        tl::ascend::pv_scale(pool[((pivot * 80) + base)], pool[((pivot * 80) + base)], pool[((pivot * 80) + base)], (-1.000000e+00f / divisor), active);
        tl::ascend::pv_brcb_rows(pool[17472], pool[23200], pool[17472], 64, 80);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mad_strided(pool[base], pool[((pivot * 80) + base)], pool[17472], 64, 64, 1, 1, 0, 10, 0, 10);
        AscendC::PipeBarrier<PIPE_V>();
      }
      if (valid != 0) {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        pool.SetValue((((int64_t)previous) + (int64_t)63), recovered);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        tl::ascend::pv_fill(pool[8704], pool[8704], pool[8704], 0.000000e+00f, count);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_square_padded_rows(pool[8704], pool[0], pool[8704], 64, 64, 80);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_reduce_sum(pool[23168], pool[8704], pool[22656], count);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        float inverse_total = pool.GetValue(23168);
        condition = (total * inverse_total);
        pool.SetValue(23168, total);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        tl::ascend::pv_sqrt(pool[23168], pool[23168], pool[23168], 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        rejected = ((int32_t)(inverse_total <= 0.000000e+00f));
        uint32_t inverse_bits = (*(uint32_t *)(&(inverse_total)));
        if ((inverse_bits & (uint32_t)2139095040) == (uint32_t)2139095040) {
          rejected = 1;
        }
        if (rejected != 0) {
          valid = 0;
        } else {
          tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power), count);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          tl::ascend::bl_index_brcb(pool[17472], order[64], pool[17472], 64);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::bl_index_matrix(pool[12800], pool[17472], order[0], 64, 64);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::bl_gather(pool[8704], pool[0], pool[12800], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(Y[(job * 4096)], pool[8704], Y[(job * 4096)], 64, 64, (64 - dma_pad), 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
      }
    }
    accepted.SetValue(0, valid);
    accepted.SetValue(1, (*(int32_t *)(&(condition))));
    S.SetValue((((int64_t)job) * (int64_t)32), ((float)accepted.GetValue(0)));
    word = accepted.GetValue(1);
    metric = (*(float *)(&(word)));
    S.SetValue(((((int64_t)job) * (int64_t)32) + (int64_t)1), metric);
  }

}


}

#include "kernel_operator.h"
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void br_cube_sync(D,A,B){AscendC::CrossCoreSetFlag<0,PIPE_FIX>(6);AscendC::CrossCoreWaitFlag<0,PIPE_MTE2>(6);}
}

namespace tl::ascend {
__aicore__ inline void BlockRowGemm( GM_ADDR A_handle,  GM_ADDR P_handle,  GM_ADDR C_handle, int32_t blocks,  GM_ADDR shape_hint, int64_t inner, int64_t cols, int64_t batch, int64_t rows, uint64_t fftsAddr) {
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<float> C;
  C.SetGlobalBuffer((__gm__ float*)C_handle);
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
    for (int32_t iteration = 0; iteration < (((((batch * (rows / 64)) * (cols / 256)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < ((batch * (rows / 64)) * (cols / 256))) {
        tl::ascend::br_load_a_shape(a[0], A[((((((bid + (iteration * blocks)) / (cols / 256)) % (rows / 64)) * 64) + (((bid + (iteration * blocks)) / ((rows / 64) * (cols / 256))) * rows)) * inner)], a[0], inner, inner, 64);
        tl::ascend::copy_gm_to_l1<float, 64, 256>(b[0], P[((((bid + (iteration * blocks)) % (cols / 256)) * 256) + ((((bid + (iteration * blocks)) / ((rows / 64) * (cols / 256))) * inner) * cols))], cols, ((64 <= (inner * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 256)))))) ? 64 : ((0 < (inner * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 256)))))) ? (inner * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 256))))) : 0)), ((1 <= ((cols / 256) - ((bid + (iteration * blocks)) % (cols / 256)))) ? 256 : ((0 < (cols - (((bid + (iteration * blocks)) % (cols / 256)) * 256))) ? (cols - (((bid + (iteration * blocks)) % (cols / 256)) * 256)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::gemm_v0<float, float, 64, 256, 64, false, false>(a[0], b[0], c[0], ascend_l0a, ascend_l0b, (bool)1);
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 256, 0>(C[((((bid + (iteration * blocks)) % (cols / 256)) * 256) + ((((((bid + (iteration * blocks)) / (cols / 256)) % (rows / 64)) * 64) + (((bid + (iteration * blocks)) / ((rows / 64) * (cols / 256))) * rows)) * cols))], c[0], cols, ((1 <= (((rows * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 256))))) / 64) - (((bid + (iteration * blocks)) / (cols / 256)) % (rows / 64)))) ? 64 : ((0 < ((rows * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 256))))) - ((((bid + (iteration * blocks)) / (cols / 256)) % (rows / 64)) * 64))) ? ((rows * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 256))))) - ((((bid + (iteration * blocks)) / (cols / 256)) % (rows / 64)) * 64)) : 0)), ((1 <= ((cols / 256) - ((bid + (iteration * blocks)) % (cols / 256)))) ? 256 : ((0 < (cols - (((bid + (iteration * blocks)) % (cols / 256)) * 256))) ? (cols - (((bid + (iteration * blocks)) % (cols / 256)) * 256)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
      }
    }
  }

}
}
#define br_row_gemm(d,a,b,oa,ob,oc,bl,ba,m,k,n) BlockRowGemm((GM_ADDR)(d.GetPhyAddr()+oa),(GM_ADDR)(d.GetPhyAddr()+ob),(GM_ADDR)(d.GetPhyAddr()+oc),bl,(GM_ADDR)d.GetPhyAddr(),k,n,ba,m,fftsAddr)
