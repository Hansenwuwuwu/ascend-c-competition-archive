#pragma once
#include "pinv_dsl_common.h"
namespace tl::ascend {
__aicore__ inline void cp_zero_pad(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int rows){uint64_t mask[2]={0x8000800080008000ULL,0};AscendC::Duplicate(d,0.0f,mask,uint8_t((rows+3)/4),1,8);}
}

namespace tl::ascend {
__aicore__ inline void cp_scale(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int n){AscendC::Muls(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),int32_t(4),n);}
}

namespace tl::ascend {
__aicore__ inline void cp_sub(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int n){AscendC::Sub(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),n);}
}

namespace tl::ascend {
__aicore__ inline void cp_shift(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int n){AscendC::ShiftRight(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),int32_t(4),n);}
}

namespace tl::ascend {
__aicore__ inline void cp_iota(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int n){AscendC::CreateVecIndex(d.ReinterpretCast<int32_t>(),int32_t(0),n);}
}

__aicore__ static __attribute__((noinline)) void PvStageCompact( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
  if (static_cast<int32_t>(AscendC::GetBlockIdx()) >= batch) return;
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);


  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int job = 0;
  int offset = 0;
  int dimension = 0;
  int accepted = 0;
  int valid = 0;
  AscendC::LocalTensor<float> pool(AscendC::TPosition::VECCALC, 0, 1440);
  float power = 0.000000e+00f;
  uint32_t power_word = (uint32_t)0;
  float total = 0.000000e+00f;
  AscendC::LocalTensor<uint32_t> indices(AscendC::TPosition::VECCALC, 5760, 64);
  float inverse_total = 0.000000e+00f;
  int valid_1 = 0;
  float power_1 = 0.000000e+00f;
  uint32_t power_word_1 = (uint32_t)0;
  float total_1 = 0.000000e+00f;
  float inverse_total_1 = 0.000000e+00f;
  uint64_t permutation = (uint64_t)0;
  int valid_2 = 0;
  float power_2 = 0.000000e+00f;
  uint32_t power_word_2 = (uint32_t)0;
  float total_2 = 0.000000e+00f;
  float inverse_total_2 = 0.000000e+00f;
  int decision = 0;
  uint32_t index_word = (uint32_t)0;
  float best = 0.000000e+00f;
  int pivot = 0;
  uint32_t shift = (uint32_t)0;
  float best_1 = 0.000000e+00f;
  int pivot_1 = 0;
  uint32_t shift_1 = (uint32_t)0;
  uint32_t shift_2 = (uint32_t)0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_module_bind(A[0], X[0], W[0]);
    job = bid;
    while (1) {
      if (!((job < batch))) { break; }
      offset = ((job * m) * n);
      dimension = n;
      accepted = 0;
      if ((m == n) && (n <= 32)) {
        if ((15 <= n) && (n <= 16)) {
          if (dimension == 16) {
            valid = 1;
            tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, 640);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], 16, 16, 0, 24, (bool)0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tl::ascend::pv_abs(pool[640], pool[0], pool[640], 640);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::sr_max(pool[1336], pool[640], pool[1336], 64, 10);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::sr_max(pool[1368], pool[1336], pool[1368], 10, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            float magnitude = pool.GetValue(1368);
            power = 1.000000e+00f;
            if ((magnitude < 1.000000e-01f) || (1.701412e+38f <= magnitude)) {
              valid = 0;
            }
            if (valid != 0) {
              power_word = (((*(uint32_t *)(&(magnitude))) + (uint32_t)8388607) & (uint32_t)2139095040);
              power = (*(float *)(&(power_word)));
              tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power), 640);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_mul(pool[640], pool[0], pool[0], 640);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], 640);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              tl::ascend::ns_acc(pool[0], pool[0], pool[0], total);
              for (int32_t i = 0; i < 16; ++i) {
                pool.SetValue(((i * 41) + 16), 1.000000e+00f);
                indices.SetValue(i, ((uint32_t)(i * 160)));
              }
              tl::ascend::pv_fill(pool[1416], pool[1416], pool[1416], 0.000000e+00f, 16);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 0, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor = pool.GetValue(1416);
              if (max(divisor, (divisor * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / divisor), 32);
              pool.SetValue(1416, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[0], pool[640], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 4, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_1 = pool.GetValue(1417);
              if (max(divisor_1, (divisor_1 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[40], pool[40], pool[40], (-1.000000e+00f / divisor_1), 32);
              pool.SetValue(1417, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[40], pool[640], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 8, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_2 = pool.GetValue(1418);
              if (max(divisor_2, (divisor_2 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[80], pool[80], pool[80], (-1.000000e+00f / divisor_2), 32);
              pool.SetValue(1418, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[80], pool[640], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 12, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_3 = pool.GetValue(1419);
              if (max(divisor_3, (divisor_3 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[120], pool[120], pool[120], (-1.000000e+00f / divisor_3), 32);
              pool.SetValue(1419, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[120], pool[640], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 16, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_4 = pool.GetValue(1420);
              if (max(divisor_4, (divisor_4 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[160], pool[160], pool[160], (-1.000000e+00f / divisor_4), 32);
              pool.SetValue(1420, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[160], pool[640], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 20, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_5 = pool.GetValue(1421);
              if (max(divisor_5, (divisor_5 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[200], pool[200], pool[200], (-1.000000e+00f / divisor_5), 32);
              pool.SetValue(1421, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[200], pool[640], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 24, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_6 = pool.GetValue(1422);
              if (max(divisor_6, (divisor_6 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[240], pool[240], pool[240], (-1.000000e+00f / divisor_6), 32);
              pool.SetValue(1422, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[240], pool[640], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 28, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_7 = pool.GetValue(1423);
              if (max(divisor_7, (divisor_7 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[288], pool[288], pool[288], (-1.000000e+00f / divisor_7), 24);
              pool.SetValue(1423, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[288], pool[640], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 32, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_8 = pool.GetValue(1424);
              if (max(divisor_8, (divisor_8 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[328], pool[328], pool[328], (-1.000000e+00f / divisor_8), 24);
              pool.SetValue(1424, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[328], pool[640], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 36, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_9 = pool.GetValue(1425);
              if (max(divisor_9, (divisor_9 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[368], pool[368], pool[368], (-1.000000e+00f / divisor_9), 24);
              pool.SetValue(1425, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[368], pool[640], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 40, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_10 = pool.GetValue(1426);
              if (max(divisor_10, (divisor_10 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[408], pool[408], pool[408], (-1.000000e+00f / divisor_10), 24);
              pool.SetValue(1426, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[408], pool[640], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 44, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_11 = pool.GetValue(1427);
              if (max(divisor_11, (divisor_11 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[448], pool[448], pool[448], (-1.000000e+00f / divisor_11), 24);
              pool.SetValue(1427, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[448], pool[640], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 48, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_12 = pool.GetValue(1428);
              if (max(divisor_12, (divisor_12 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[488], pool[488], pool[488], (-1.000000e+00f / divisor_12), 24);
              pool.SetValue(1428, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[488], pool[640], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 52, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_13 = pool.GetValue(1429);
              if (max(divisor_13, (divisor_13 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[528], pool[528], pool[528], (-1.000000e+00f / divisor_13), 24);
              pool.SetValue(1429, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[528], pool[640], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 56, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_14 = pool.GetValue(1430);
              if (max(divisor_14, (divisor_14 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[568], pool[568], pool[568], (-1.000000e+00f / divisor_14), 24);
              pool.SetValue(1430, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[568], pool[640], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 60, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_15 = pool.GetValue(1431);
              if (max(divisor_15, (divisor_15 * -1.000000e+00f)) <= 3.906250e-03f) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[616], pool[616], pool[616], (-1.000000e+00f / divisor_15), 16);
              pool.SetValue(1431, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[16], pool[616], pool[640], 16, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              if (valid != 0) {
                tl::ascend::pv_fill(pool[640], pool[640], pool[640], 0.000000e+00f, 640);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_square_padded_rows(pool[640], pool[16], pool[640], 16, 16, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], 640);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::ns_acc(pool[0], pool[0], pool[0], inverse_total);
                float absolute = (atol / power);
                float cutoff = max(((total * rtol) * rtol), (absolute * absolute));
                if ((inverse_total <= 0.000000e+00f) || (8.100000e-01f <= (inverse_total * cutoff))) {
                  valid = 0;
                }
                if (valid != 0) {
                  tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power), 640);
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                  tl::ascend::pv_store_rows(X[offset], pool[16], X[offset], 16, 16, 24, 0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                }
              }
            }
            accepted = valid;
          } else {
            if (0 < dimension) {
              int32_t n_1 = dimension;
              valid_1 = 1;
              tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (n_1 * 40));
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              if (n_1 == 15) {
                tl::ascend::pv_load_rows(pool[640], A[offset], pool[640], 1, (n_1 * n_1), 0, 0);
              } else {
                tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], n_1, n_1, 0, 24, (0 < (n_1 % 8)));
              }
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              if (n_1 == 15) {
                indices.SetValue(0, (uint32_t)0);
                indices.SetValue(16, (uint32_t)0);
                indices.SetValue(1, (uint32_t)4);
                indices.SetValue(17, (uint32_t)60);
                indices.SetValue(2, (uint32_t)8);
                indices.SetValue(18, (uint32_t)120);
                indices.SetValue(3, (uint32_t)12);
                indices.SetValue(19, (uint32_t)180);
                indices.SetValue(4, (uint32_t)16);
                indices.SetValue(20, (uint32_t)240);
                indices.SetValue(5, (uint32_t)20);
                indices.SetValue(21, (uint32_t)300);
                indices.SetValue(6, (uint32_t)24);
                indices.SetValue(22, (uint32_t)360);
                indices.SetValue(7, (uint32_t)28);
                indices.SetValue(23, (uint32_t)420);
                indices.SetValue(8, (uint32_t)32);
                indices.SetValue(24, (uint32_t)480);
                indices.SetValue(9, (uint32_t)36);
                indices.SetValue(25, (uint32_t)540);
                indices.SetValue(10, (uint32_t)40);
                indices.SetValue(26, (uint32_t)600);
                indices.SetValue(11, (uint32_t)44);
                indices.SetValue(27, (uint32_t)660);
                indices.SetValue(12, (uint32_t)48);
                indices.SetValue(28, (uint32_t)720);
                indices.SetValue(13, (uint32_t)52);
                indices.SetValue(29, (uint32_t)780);
                indices.SetValue(14, (uint32_t)56);
                indices.SetValue(30, (uint32_t)840);
                indices.SetValue(15, (uint32_t)60);
                indices.SetValue(31, (uint32_t)900);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::bl_index_brcb(pool[896], indices[16], pool[896], 16);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::bl_index_matrix(pool[1152], pool[896], indices[0], 16, 16);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_gather_bits(pool[896], pool[640], pool[1152], 0, (n_1 * 16));
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::cp_zero_pad(pool[896], pool[896], pool[896], n_1);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_copy_ub_rows(pool[0], pool[896], pool[0], n_1, 16, 0, 24);
                AscendC::PipeBarrier<PIPE_V>();
              }
              tl::ascend::pv_abs(pool[640], pool[0], pool[640], (n_1 * 40));
              AscendC::PipeBarrier<PIPE_V>();
              if (1 < n_1) {
                tl::ascend::sr_max(pool[1336], pool[640], pool[1336], 64, ((n_1 * 40) / 64));
              }
              if (7 < ((n_1 * 40) % 64)) {
                tl::ascend::sr_max(pool[(((n_1 * 40) / 64) + 1336)], pool[((((n_1 * 40) / 64) * 64) + 640)], pool[(((n_1 * 40) / 64) + 1336)], ((n_1 * 40) % 64), 1);
              }
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::sr_max(pool[1368], pool[1336], pool[1368], (((n_1 * 5) + 7) / 8), 1);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float magnitude_1 = pool.GetValue(1368);
              power_1 = 1.000000e+00f;
              if ((magnitude_1 < 1.000000e-01f) || (1.701412e+38f <= magnitude_1)) {
                valid_1 = 0;
              }
              if (valid_1 != 0) {
                power_word_1 = (((*(uint32_t *)(&(magnitude_1))) + (uint32_t)8388607) & (uint32_t)2139095040);
                power_1 = (*(float *)(&(power_word_1)));
                tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_1), (n_1 * 40));
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mul(pool[640], pool[0], pool[0], (n_1 * 40));
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], (n_1 * 40));
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::ns_acc(pool[0], pool[0], pool[0], total_1);
                for (int32_t i_1 = 0; i_1 < n_1; ++i_1) {
                  pool.SetValue(((((int64_t)i_1) * (int64_t)41) + (int64_t)16), 1.000000e+00f);
                  indices.SetValue(i_1, ((uint32_t)(i_1 * 160)));
                }
                tl::ascend::pv_fill(pool[1416], pool[1416], pool[1416], 0.000000e+00f, 16);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                for (int32_t column_tile = 0; column_tile < n_1; ++column_tile) {
                  tl::ascend::pv_gather(pool[1416], pool[0], indices[0], (column_tile * 4), n_1);
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  float divisor_16 = pool.GetValue((((int64_t)column_tile) + (int64_t)1416));
                  if (max(divisor_16, (divisor_16 * -1.000000e+00f)) <= 3.906250e-03f) {
                    valid_1 = 0;
                    break;
                  }
                  tl::ascend::pv_scale(pool[((column_tile * 40) + (((column_tile + 1) / 8) * 8))], pool[((column_tile * 40) + (((column_tile + 1) / 8) * 8))], pool[((column_tile * 40) + (((column_tile + 1) / 8) * 8))], (-1.000000e+00f / divisor_16), (32 - (((column_tile + 1) / 8) * 8)));
                  pool.SetValue((((int64_t)column_tile) + (int64_t)1416), 0.000000e+00f);
                  AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                  tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_gj_update(pool[(((column_tile + 1) / 8) * 8)], pool[((column_tile * 40) + (((column_tile + 1) / 8) * 8))], pool[640], (32 - (((column_tile + 1) / 8) * 8)), n_1, 40);
                  AscendC::PipeBarrier<PIPE_V>();
                }
                if (valid_1 != 0) {
                  tl::ascend::pv_fill(pool[640], pool[640], pool[640], 0.000000e+00f, (n_1 * 40));
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_square_padded_rows(pool[640], pool[16], pool[640], 16, n_1, 40);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], (n_1 * 40));
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  tl::ascend::ns_acc(pool[0], pool[0], pool[0], inverse_total_1);
                  float absolute_1 = (atol / power_1);
                  float cutoff_1 = max(((total_1 * rtol) * rtol), (absolute_1 * absolute_1));
                  if ((inverse_total_1 <= 0.000000e+00f) || (8.100000e-01f <= (inverse_total_1 * cutoff_1))) {
                    valid_1 = 0;
                  }
                  if (valid_1 != 0) {
                    tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power_1), (n_1 * 40));
                    AscendC::PipeBarrier<PIPE_V>();
                    if (n_1 == 15) {
                      tl::ascend::pv_pack15(pool[640], pool[16], pool[640], n_1, 40);
                      AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                    if (n_1 == 15) {
                      tl::ascend::pv_store_rows(X[offset], pool[640], X[offset], 1, (n_1 * n_1), 0, 0);
                    } else {
                      tl::ascend::pv_store_rows(X[offset], pool[16], X[offset], n_1, n_1, 24, 0);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  }
                }
              }
              accepted = valid_1;
            }
          }
          if (accepted == 0) {
            int32_t count = (dimension * 40);
            permutation = (uint64_t)0;
            valid_2 = 1;
            tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], dimension, dimension, 0, 24, (0 < (dimension % 8)));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tl::ascend::pv_abs(pool[640], pool[0], pool[640], count);
            AscendC::PipeBarrier<PIPE_V>();
            if (63 < count) {
              tl::ascend::sr_max(pool[1336], pool[640], pool[1336], 64, (count / 64));
            }
            if (0 < (count % 64)) {
              tl::ascend::sr_max(pool[((count / 64) + 1336)], pool[(((count / 64) * 64) + 640)], pool[((count / 64) + 1336)], (count % 64), 1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::sr_max(pool[1368], pool[1336], pool[1368], ((count + 63) / 64), 1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            float scale = pool.GetValue(1368);
            power_2 = 1.000000e+00f;
            if ((scale < 1.000000e-01f) || (1.701412e+38f <= scale)) {
              valid_2 = 0;
            }
            if (valid_2 != 0) {
              power_word_2 = (((*(uint32_t *)(&(scale))) + (uint32_t)8388607) & (uint32_t)2139095040);
              power_2 = (*(float *)(&(power_word_2)));
              tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_2), count);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_mul(pool[640], pool[0], pool[0], count);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], count);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              tl::ascend::ns_acc(pool[0], pool[0], pool[0], total_2);
              for (int32_t i_2 = 0; i_2 < dimension; ++i_2) {
                pool.SetValue(((((int64_t)i_2) * (int64_t)41) + (int64_t)16), 1.000000e+00f);
                index_word = ((uint32_t)(i_2 * 160));
                pool.SetValue((((int64_t)i_2) + (int64_t)1400), (*(float *)(&(index_word))));
              }
              tl::ascend::pv_mask_begin(pool[0], pool[0], pool[0], dimension);
              tl::ascend::pv_fill(pool[1416], pool[1416], pool[1416], 0.000000e+00f, 16);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              for (int32_t pivot_tile = 0; pivot_tile < dimension; ++pivot_tile) {
                tl::ascend::pv_gather_bits(pool[1416], pool[0], pool[1400], (pivot_tile * 4), dimension);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_abs(pool[1320], pool[1416], pool[1320], 16);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mask_reduce(pool[1368], pool[1320], pool[1368]);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best, pivot);
                if (best <= 0.000000e+00f) {
                  valid_2 = 0;
                  break;
                }
                tl::ascend::pv_mask_remove(pool[0], pool[0], pool[0], pivot);
                shift = ((uint32_t)(pivot * 4));
                permutation = (permutation | (((uint64_t)pivot_tile) << ((uint64_t)shift)));
                float factor = (-1.000000e+00f / pool.GetValue(((((int64_t)pivot) * (int64_t)40) + ((int64_t)pivot_tile))));
                tl::ascend::pv_scale(pool[(pivot * 40)], pool[(pivot * 40)], pool[(pivot * 40)], factor, 32);
                AscendC::PipeBarrier<PIPE_V>();
                pool.SetValue((((int64_t)pivot) + (int64_t)1416), 0.000000e+00f);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_gj_update(pool[0], pool[(pivot * 40)], pool[640], 32, dimension, 40);
                AscendC::PipeBarrier<PIPE_V>();
              }
              if (valid_2 != 0) {
                tl::ascend::pv_fill(pool[640], pool[640], pool[640], 0.000000e+00f, count);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_square_padded_rows(pool[640], pool[16], pool[640], 16, dimension, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], count);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::ns_acc(pool[0], pool[0], pool[0], inverse_total_2);
                pool.SetValue(1368, total_2);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::pv_sqrt(pool[1368], pool[1368], pool[1368], 1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float cutoff_2 = max((pool.GetValue(1368) * rtol), (atol / power_2));
                if ((inverse_total_2 <= 0.000000e+00f) || (8.100000e-01f <= ((inverse_total_2 * cutoff_2) * cutoff_2))) {
                  valid_2 = 0;
                }
                if (valid_2 != 0) {
                  tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power_2), count);
                  AscendC::PipeBarrier<PIPE_V>();
                  for (int32_t row = 0; row < dimension; ++row) {
                    shift_2 = ((uint32_t)(row * 4));
                    int32_t target = ((int32_t)((permutation >> ((uint64_t)shift_2)) & (uint64_t)15));
                    tl::ascend::pv_copy(pool[((target * 40) + 640)], pool[(row * 40)], pool[((target * 40) + 640)], 40);
                  }
                  AscendC::PipeBarrier<PIPE_V>();
                  if (dimension == 15) {
                    tl::ascend::pv_pack15(pool[0], pool[656], pool[0], dimension, 40);
                    AscendC::PipeBarrier<PIPE_V>();
                  }
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                  if (dimension == 15) {
                    tl::ascend::pv_store_rows(X[offset], pool[0], X[offset], 1, (dimension * dimension), 0, 0);
                  } else {
                    tl::ascend::pv_store_rows(X[offset], pool[656], X[offset], dimension, dimension, 24, 0);
                  }
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                }
              }
            }
            accepted = valid_2;
          }
        }
        if (accepted == 0) {
          decision = 0;
          tl::ascend::pv_module_resident_accurate(pool[0], pool[0], pool[0], decision, blocks, job, m, n, batch, atol, rtol);
          if (decision == 0) {
            tl::ascend::pv_module_resident_rectangular(pool[0], pool[0], pool[0], decision, blocks, job, m, n, batch, atol, rtol);
          }
        }
      } else {
        tl::ascend::pv_module_rectangular(S[0], S[0], S[0], blocks, job, m, n, batch, atol, rtol);
      }
      job = (job + blocks);
    }
  }

}

__aicore__ static __attribute__((noinline)) void PvStageCompactAligned( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
  if (static_cast<int32_t>(AscendC::GetBlockIdx()) >= batch) return;
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);


  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int job = 0;
  int offset = 0;
  int dimension = 0;
  int accepted = 0;
  int valid = 0;
  AscendC::LocalTensor<float> pool(AscendC::TPosition::VECCALC, 0, 1440);
  float power = 0.000000e+00f;
  uint32_t power_word = (uint32_t)0;
  float pivot_floor = 0.000000e+00f;
  float total = 0.000000e+00f;
  AscendC::LocalTensor<uint32_t> indices(AscendC::TPosition::VECCALC, 5760, 64);
  float inverse_total = 0.000000e+00f;
  int valid_1 = 0;
  float power_1 = 0.000000e+00f;
  uint32_t power_word_1 = (uint32_t)0;
  float total_1 = 0.000000e+00f;
  float inverse_total_1 = 0.000000e+00f;
  uint64_t permutation = (uint64_t)0;
  int valid_2 = 0;
  float power_2 = 0.000000e+00f;
  uint32_t power_word_2 = (uint32_t)0;
  float total_2 = 0.000000e+00f;
  float inverse_total_2 = 0.000000e+00f;
  int decision = 0;
  uint32_t index_word = (uint32_t)0;
  float best = 0.000000e+00f;
  int pivot = 0;
  uint32_t shift = (uint32_t)0;
  float best_1 = 0.000000e+00f;
  int pivot_1 = 0;
  uint32_t shift_1 = (uint32_t)0;
  uint32_t shift_2 = (uint32_t)0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_module_bind(A[0], X[0], W[0]);
    job = bid;
    while (1) {
      if (!((job < batch))) { break; }
      offset = ((job * m) * n);
      dimension = n;
      accepted = 0;
      if ((m == n) && (n <= 32)) {
        if ((15 <= n) && (n <= 16)) {
          if (dimension == 16) {
            valid = 1;
            tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, 640);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], 16, 16, 0, 24, (bool)0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tl::ascend::pv_abs(pool[640], pool[0], pool[640], 640);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::sr_max(pool[1336], pool[640], pool[1336], 64, 10);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::sr_max(pool[1368], pool[1336], pool[1368], 10, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            float magnitude = pool.GetValue(1368);
            power = 1.000000e+00f;
            if ((magnitude < 1.000000e-01f) || (6.553600e+04f <= magnitude)) {
              valid = 0;
            }
            if (valid != 0) {
              power_word = (((*(uint32_t *)(&(magnitude))) + (uint32_t)8388607) & (uint32_t)2139095040);
              pivot_floor = 3.906250e-03f;
              pivot_floor = ((*(float *)(&(power_word))) * 3.906250e-03f);
              total = (((2.560000e+02f * magnitude) * magnitude) * 1.000010e+00f);
              for (int32_t i = 0; i < 16; ++i) {
                pool.SetValue(((i * 41) + 16), -1.000000e+00f);
                indices.SetValue(i, ((uint32_t)(i * 160)));
              }
              tl::ascend::pv_fill(pool[1416], pool[1416], pool[1416], 0.000000e+00f, 16);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 0, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor = pool.GetValue(1416);
              if (max(divisor, (divisor * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / divisor), 32);
              pool.SetValue(1416, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[0], pool[672], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 4, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_1 = pool.GetValue(1417);
              if (max(divisor_1, (divisor_1 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[40], pool[40], pool[40], (-1.000000e+00f / divisor_1), 32);
              pool.SetValue(1417, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[40], pool[672], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 8, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_2 = pool.GetValue(1418);
              if (max(divisor_2, (divisor_2 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[80], pool[80], pool[80], (-1.000000e+00f / divisor_2), 32);
              pool.SetValue(1418, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[80], pool[672], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 12, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_3 = pool.GetValue(1419);
              if (max(divisor_3, (divisor_3 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[120], pool[120], pool[120], (-1.000000e+00f / divisor_3), 32);
              pool.SetValue(1419, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[120], pool[672], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 16, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_4 = pool.GetValue(1420);
              if (max(divisor_4, (divisor_4 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[160], pool[160], pool[160], (-1.000000e+00f / divisor_4), 32);
              pool.SetValue(1420, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[160], pool[672], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 20, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_5 = pool.GetValue(1421);
              if (max(divisor_5, (divisor_5 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[200], pool[200], pool[200], (-1.000000e+00f / divisor_5), 32);
              pool.SetValue(1421, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[200], pool[672], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 24, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_6 = pool.GetValue(1422);
              if (max(divisor_6, (divisor_6 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[240], pool[240], pool[240], (-1.000000e+00f / divisor_6), 32);
              pool.SetValue(1422, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[0], pool[240], pool[672], 32, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 28, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_7 = pool.GetValue(1423);
              if (max(divisor_7, (divisor_7 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[288], pool[288], pool[288], (-1.000000e+00f / divisor_7), 24);
              pool.SetValue(1423, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[288], pool[672], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 32, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_8 = pool.GetValue(1424);
              if (max(divisor_8, (divisor_8 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[328], pool[328], pool[328], (-1.000000e+00f / divisor_8), 24);
              pool.SetValue(1424, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[328], pool[672], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 36, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_9 = pool.GetValue(1425);
              if (max(divisor_9, (divisor_9 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[368], pool[368], pool[368], (-1.000000e+00f / divisor_9), 24);
              pool.SetValue(1425, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[368], pool[672], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 40, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_10 = pool.GetValue(1426);
              if (max(divisor_10, (divisor_10 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[408], pool[408], pool[408], (-1.000000e+00f / divisor_10), 24);
              pool.SetValue(1426, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[408], pool[672], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 44, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_11 = pool.GetValue(1427);
              if (max(divisor_11, (divisor_11 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[448], pool[448], pool[448], (-1.000000e+00f / divisor_11), 24);
              pool.SetValue(1427, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[448], pool[672], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 48, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_12 = pool.GetValue(1428);
              if (max(divisor_12, (divisor_12 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[488], pool[488], pool[488], (-1.000000e+00f / divisor_12), 24);
              pool.SetValue(1428, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[488], pool[672], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 52, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_13 = pool.GetValue(1429);
              if (max(divisor_13, (divisor_13 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[528], pool[528], pool[528], (-1.000000e+00f / divisor_13), 24);
              pool.SetValue(1429, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[528], pool[672], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 56, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_14 = pool.GetValue(1430);
              if (max(divisor_14, (divisor_14 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[568], pool[568], pool[568], (-1.000000e+00f / divisor_14), 24);
              pool.SetValue(1430, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[8], pool[568], pool[672], 24, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gather(pool[1416], pool[0], indices[0], 60, 16);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float divisor_15 = pool.GetValue(1431);
              if (max(divisor_15, (divisor_15 * -1.000000e+00f)) <= pivot_floor) {
                valid = 0;
                break;
              }
              tl::ascend::pv_scale(pool[616], pool[616], pool[616], (-1.000000e+00f / divisor_15), 16);
              pool.SetValue(1431, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              tl::ascend::pv_brcb_rows(pool[672], pool[1416], pool[672], 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_gj_update(pool[16], pool[616], pool[672], 16, 16, 40);
              AscendC::PipeBarrier<PIPE_V>();
              if (valid != 0) {
                tl::ascend::lv_square_packed_rows(pool[640], pool[16], pool[640], 16, 16, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], 256);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::ns_acc(pool[0], pool[0], pool[0], inverse_total);
                float absolute = (atol / power);
                float cutoff = max(((total * rtol) * rtol), (absolute * absolute));
                if ((inverse_total <= 0.000000e+00f) || (8.100000e-01f <= (inverse_total * cutoff))) {
                  valid = 0;
                }
                if (valid != 0) {
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                  tl::ascend::pv_store_rows(X[offset], pool[16], X[offset], 16, 16, 24, 0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                }
              }
            }
            accepted = valid;
          } else {
            if (0 < dimension) {
              int32_t n_1 = dimension;
              valid_1 = 1;
              tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (n_1 * 40));
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              if (n_1 == 15) {
                tl::ascend::pv_load_rows(pool[640], A[offset], pool[640], 1, (n_1 * n_1), 0, 0);
              } else {
                tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], n_1, n_1, 0, 24, (0 < (n_1 % 8)));
              }
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              if (n_1 == 15) {
                indices.SetValue(0, (uint32_t)0);
                indices.SetValue(16, (uint32_t)0);
                indices.SetValue(1, (uint32_t)4);
                indices.SetValue(17, (uint32_t)60);
                indices.SetValue(2, (uint32_t)8);
                indices.SetValue(18, (uint32_t)120);
                indices.SetValue(3, (uint32_t)12);
                indices.SetValue(19, (uint32_t)180);
                indices.SetValue(4, (uint32_t)16);
                indices.SetValue(20, (uint32_t)240);
                indices.SetValue(5, (uint32_t)20);
                indices.SetValue(21, (uint32_t)300);
                indices.SetValue(6, (uint32_t)24);
                indices.SetValue(22, (uint32_t)360);
                indices.SetValue(7, (uint32_t)28);
                indices.SetValue(23, (uint32_t)420);
                indices.SetValue(8, (uint32_t)32);
                indices.SetValue(24, (uint32_t)480);
                indices.SetValue(9, (uint32_t)36);
                indices.SetValue(25, (uint32_t)540);
                indices.SetValue(10, (uint32_t)40);
                indices.SetValue(26, (uint32_t)600);
                indices.SetValue(11, (uint32_t)44);
                indices.SetValue(27, (uint32_t)660);
                indices.SetValue(12, (uint32_t)48);
                indices.SetValue(28, (uint32_t)720);
                indices.SetValue(13, (uint32_t)52);
                indices.SetValue(29, (uint32_t)780);
                indices.SetValue(14, (uint32_t)56);
                indices.SetValue(30, (uint32_t)840);
                indices.SetValue(15, (uint32_t)60);
                indices.SetValue(31, (uint32_t)900);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::bl_index_brcb(pool[896], indices[16], pool[896], 16);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::bl_index_matrix(pool[1152], pool[896], indices[0], 16, 16);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_gather_bits(pool[896], pool[640], pool[1152], 0, (n_1 * 16));
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::cp_zero_pad(pool[896], pool[896], pool[896], n_1);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_copy_ub_rows(pool[0], pool[896], pool[0], n_1, 16, 0, 24);
                AscendC::PipeBarrier<PIPE_V>();
              }
              tl::ascend::pv_abs(pool[640], pool[0], pool[640], (n_1 * 40));
              AscendC::PipeBarrier<PIPE_V>();
              if (1 < n_1) {
                tl::ascend::sr_max(pool[1336], pool[640], pool[1336], 64, ((n_1 * 40) / 64));
              }
              if (7 < ((n_1 * 40) % 64)) {
                tl::ascend::sr_max(pool[(((n_1 * 40) / 64) + 1336)], pool[((((n_1 * 40) / 64) * 64) + 640)], pool[(((n_1 * 40) / 64) + 1336)], ((n_1 * 40) % 64), 1);
              }
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::sr_max(pool[1368], pool[1336], pool[1368], (((n_1 * 5) + 7) / 8), 1);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float magnitude_1 = pool.GetValue(1368);
              power_1 = 1.000000e+00f;
              if ((magnitude_1 < 1.000000e-01f) || (1.701412e+38f <= magnitude_1)) {
                valid_1 = 0;
              }
              if (valid_1 != 0) {
                power_word_1 = (((*(uint32_t *)(&(magnitude_1))) + (uint32_t)8388607) & (uint32_t)2139095040);
                power_1 = (*(float *)(&(power_word_1)));
                tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_1), (n_1 * 40));
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mul(pool[640], pool[0], pool[0], (n_1 * 40));
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], (n_1 * 40));
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::ns_acc(pool[0], pool[0], pool[0], total_1);
                for (int32_t i_1 = 0; i_1 < n_1; ++i_1) {
                  pool.SetValue(((((int64_t)i_1) * (int64_t)41) + (int64_t)16), 1.000000e+00f);
                  indices.SetValue(i_1, ((uint32_t)(i_1 * 160)));
                }
                tl::ascend::pv_fill(pool[1416], pool[1416], pool[1416], 0.000000e+00f, 16);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                for (int32_t column_tile = 0; column_tile < n_1; ++column_tile) {
                  tl::ascend::pv_gather(pool[1416], pool[0], indices[0], (column_tile * 4), n_1);
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  float divisor_16 = pool.GetValue((((int64_t)column_tile) + (int64_t)1416));
                  if (max(divisor_16, (divisor_16 * -1.000000e+00f)) <= 3.906250e-03f) {
                    valid_1 = 0;
                    break;
                  }
                  tl::ascend::pv_scale(pool[((column_tile * 40) + (((column_tile + 1) / 8) * 8))], pool[((column_tile * 40) + (((column_tile + 1) / 8) * 8))], pool[((column_tile * 40) + (((column_tile + 1) / 8) * 8))], (-1.000000e+00f / divisor_16), (32 - (((column_tile + 1) / 8) * 8)));
                  pool.SetValue((((int64_t)column_tile) + (int64_t)1416), 0.000000e+00f);
                  AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                  tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_gj_update(pool[(((column_tile + 1) / 8) * 8)], pool[((column_tile * 40) + (((column_tile + 1) / 8) * 8))], pool[640], (32 - (((column_tile + 1) / 8) * 8)), n_1, 40);
                  AscendC::PipeBarrier<PIPE_V>();
                }
                if (valid_1 != 0) {
                  tl::ascend::pv_fill(pool[640], pool[640], pool[640], 0.000000e+00f, (n_1 * 40));
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_square_padded_rows(pool[640], pool[16], pool[640], 16, n_1, 40);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], (n_1 * 40));
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  tl::ascend::ns_acc(pool[0], pool[0], pool[0], inverse_total_1);
                  float absolute_1 = (atol / power_1);
                  float cutoff_1 = max(((total_1 * rtol) * rtol), (absolute_1 * absolute_1));
                  if ((inverse_total_1 <= 0.000000e+00f) || (8.100000e-01f <= (inverse_total_1 * cutoff_1))) {
                    valid_1 = 0;
                  }
                  if (valid_1 != 0) {
                    tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power_1), (n_1 * 40));
                    AscendC::PipeBarrier<PIPE_V>();
                    if (n_1 == 15) {
                      tl::ascend::pv_pack15(pool[640], pool[16], pool[640], n_1, 40);
                      AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                    if (n_1 == 15) {
                      tl::ascend::pv_store_rows(X[offset], pool[640], X[offset], 1, (n_1 * n_1), 0, 0);
                    } else {
                      tl::ascend::pv_store_rows(X[offset], pool[16], X[offset], n_1, n_1, 24, 0);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  }
                }
              }
              accepted = valid_1;
            }
          }
          if (accepted == 0) {
            int32_t count = (dimension * 40);
            permutation = (uint64_t)0;
            valid_2 = 1;
            tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], dimension, dimension, 0, 24, (0 < (dimension % 8)));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tl::ascend::pv_abs(pool[640], pool[0], pool[640], count);
            AscendC::PipeBarrier<PIPE_V>();
            if (63 < count) {
              tl::ascend::sr_max(pool[1336], pool[640], pool[1336], 64, (count / 64));
            }
            if (0 < (count % 64)) {
              tl::ascend::sr_max(pool[((count / 64) + 1336)], pool[(((count / 64) * 64) + 640)], pool[((count / 64) + 1336)], (count % 64), 1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::sr_max(pool[1368], pool[1336], pool[1368], ((count + 63) / 64), 1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            float scale = pool.GetValue(1368);
            power_2 = 1.000000e+00f;
            if ((scale < 1.000000e-01f) || (1.701412e+38f <= scale)) {
              valid_2 = 0;
            }
            if (valid_2 != 0) {
              power_word_2 = (((*(uint32_t *)(&(scale))) + (uint32_t)8388607) & (uint32_t)2139095040);
              power_2 = (*(float *)(&(power_word_2)));
              tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_2), count);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_mul(pool[640], pool[0], pool[0], count);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], count);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              tl::ascend::ns_acc(pool[0], pool[0], pool[0], total_2);
              for (int32_t i_2 = 0; i_2 < dimension; ++i_2) {
                pool.SetValue(((((int64_t)i_2) * (int64_t)41) + (int64_t)16), 1.000000e+00f);
                index_word = ((uint32_t)(i_2 * 160));
                pool.SetValue((((int64_t)i_2) + (int64_t)1400), (*(float *)(&(index_word))));
              }
              tl::ascend::pv_mask_begin(pool[0], pool[0], pool[0], dimension);
              tl::ascend::pv_fill(pool[1416], pool[1416], pool[1416], 0.000000e+00f, 16);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              for (int32_t pivot_tile = 0; pivot_tile < dimension; ++pivot_tile) {
                tl::ascend::pv_gather_bits(pool[1416], pool[0], pool[1400], (pivot_tile * 4), dimension);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_abs(pool[1320], pool[1416], pool[1320], 16);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mask_reduce(pool[1368], pool[1320], pool[1368]);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best, pivot);
                if (best <= 0.000000e+00f) {
                  valid_2 = 0;
                  break;
                }
                tl::ascend::pv_mask_remove(pool[0], pool[0], pool[0], pivot);
                shift = ((uint32_t)(pivot * 4));
                permutation = (permutation | (((uint64_t)pivot_tile) << ((uint64_t)shift)));
                float factor = (-1.000000e+00f / pool.GetValue(((((int64_t)pivot) * (int64_t)40) + ((int64_t)pivot_tile))));
                tl::ascend::pv_scale(pool[(pivot * 40)], pool[(pivot * 40)], pool[(pivot * 40)], factor, 32);
                AscendC::PipeBarrier<PIPE_V>();
                pool.SetValue((((int64_t)pivot) + (int64_t)1416), 0.000000e+00f);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_gj_update(pool[0], pool[(pivot * 40)], pool[640], 32, dimension, 40);
                AscendC::PipeBarrier<PIPE_V>();
              }
              if (valid_2 != 0) {
                tl::ascend::pv_fill(pool[640], pool[640], pool[640], 0.000000e+00f, count);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_square_padded_rows(pool[640], pool[16], pool[640], 16, dimension, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::ns_sum(pool[1368], pool[640], pool[1368], count);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::ns_acc(pool[0], pool[0], pool[0], inverse_total_2);
                pool.SetValue(1368, total_2);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::pv_sqrt(pool[1368], pool[1368], pool[1368], 1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float cutoff_2 = max((pool.GetValue(1368) * rtol), (atol / power_2));
                if ((inverse_total_2 <= 0.000000e+00f) || (8.100000e-01f <= ((inverse_total_2 * cutoff_2) * cutoff_2))) {
                  valid_2 = 0;
                }
                if (valid_2 != 0) {
                  tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power_2), count);
                  AscendC::PipeBarrier<PIPE_V>();
                  for (int32_t row = 0; row < dimension; ++row) {
                    shift_2 = ((uint32_t)(row * 4));
                    int32_t target = ((int32_t)((permutation >> ((uint64_t)shift_2)) & (uint64_t)15));
                    tl::ascend::pv_copy(pool[((target * 40) + 640)], pool[(row * 40)], pool[((target * 40) + 640)], 40);
                  }
                  AscendC::PipeBarrier<PIPE_V>();
                  if (dimension == 15) {
                    tl::ascend::pv_pack15(pool[0], pool[656], pool[0], dimension, 40);
                    AscendC::PipeBarrier<PIPE_V>();
                  }
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                  if (dimension == 15) {
                    tl::ascend::pv_store_rows(X[offset], pool[0], X[offset], 1, (dimension * dimension), 0, 0);
                  } else {
                    tl::ascend::pv_store_rows(X[offset], pool[656], X[offset], dimension, dimension, 24, 0);
                  }
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                }
              }
            }
            accepted = valid_2;
          }
        }
        if (accepted == 0) {
          decision = 0;
          tl::ascend::pv_module_resident_accurate(pool[0], pool[0], pool[0], decision, blocks, job, m, n, batch, atol, rtol);
          if (decision == 0) {
            tl::ascend::pv_module_resident_rectangular(pool[0], pool[0], pool[0], decision, blocks, job, m, n, batch, atol, rtol);
          }
        }
      } else {
        tl::ascend::pv_module_rectangular(S[0], S[0], S[0], blocks, job, m, n, batch, atol, rtol);
      }
      job = (job + blocks);
    }
  }

}
