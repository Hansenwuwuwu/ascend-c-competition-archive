#pragma once
#include "pinv_dsl_common.h"
namespace tl::ascend {
__aicore__ static __attribute__((noinline)) void PvDslPeerTileRows( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, int32_t n, int32_t stride, int32_t all_batch, int32_t job, int32_t core, int32_t cores, int32_t window, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int k = 0;
  int wide = 0;
  int owned = 0;
  int factor_rows = 0;
  int packet_stride = 0;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  int valid = 0;
  float power = 0.000000e+00f;
  int window_end = 0;
  float inverse = 0.000000e+00f;
  float peak = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    k = 200;
    wide = (((k + 63) / 64) * 64);
    owned = ((208 - core) / 9);
    factor_rows = (((owned + 7) / 8) * 8);
    packet_stride = (k + 16);
    int32_t packets = (owned * 456);
    int32_t output = ((packet_stride * 16) + packets);
    int32_t factors = (max((owned * k), 512) + output);
    int32_t meta = ((max(factor_rows, 16) * 8) + factors);
    int32_t vector = ((meta + wide) + 64);
    int32_t product = (vector + wide);
    int32_t accumulator = (product + wide);
    int32_t scratch = (accumulator + wide);
    int32_t red = (scratch + wide);
    int32_t packet_indices = ((red + factor_rows) + 32);
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (owned * 456));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[0], A[((((int64_t)job) * (int64_t)40000) + (((int64_t)core) * (int64_t)200))], pool[0], owned, 200, 1600, (456 - k));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::pv_fill(pool[(meta + 64)], pool[(meta + 64)], pool[(meta + 64)], 1.000000e+00f, k);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    for (int32_t i = 0; i < factor_rows; ++i) {
      int32_t condval;
      if ((i < owned)) {
        condval = (i * 1824);
      } else {
        condval = 0;
      }
      int32_t index = condval;
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], ((red + i) + 32), index);
    }
    for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (packet_indices + i_1), ((i_1 * packet_stride) * 4));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    valid = 1;
    power = 1.000000e+00f;
    window_end = 200;
    if (0 < window) {
      window_end = min(200, window);
    }
    tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
    for (int32_t panel = 0; panel < 13; ++panel) {
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_load_rows(pool[meta], W[(((((int64_t)all_batch) * (int64_t)182400) + (((int64_t)job) * (int64_t)8192)) + (((int64_t)panel) * (int64_t)64))], pool[meta], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      if (pool.GetValue((((int64_t)meta) + (int64_t)34)) <= 5.000000e-01f) {
        valid = 0;
      }
      power = pool.GetValue((((int64_t)meta) + (int64_t)32));
      int32_t condval_1;
      if ((0 < window)) {
        condval_1 = ((panel * packet_stride) * 16);
      } else {
        condval_1 = 0;
      }
      int32_t packet_begin = condval_1;
      if (valid != 0) {
        if (panel == 0) {
          for (int32_t chunk = 0; chunk < (((owned * 19) + 679) / 680); ++chunk) {
            tl::ascend::pv_scale(pool[(chunk * 16320)], pool[(chunk * 16320)], pool[(chunk * 16320)], (1.000000e+00f / power), min(((owned * 456) - (chunk * 16320)), 16320));
            AscendC::PipeBarrier<PIPE_V>();
          }
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        for (int32_t column = 0; column < (200 - max((panel * 16), 184)); ++column) {
          int32_t selected = ((int32_t)pool.GetValue((((int64_t)meta) + ((int64_t)column))));
          pool.SetValue(((((int64_t)meta) + ((int64_t)selected)) + (int64_t)64), (((float)(((panel * 16) + column) + 1)) * -1.000000e+00f));
          if ((selected % 9) == core) {
            pool.SetValue((((((((int64_t)selected) / (int64_t)9) * (int64_t)456) + (((int64_t)panel) * (int64_t)16)) + ((int64_t)k)) + ((int64_t)column)), 1.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(W[(((((int64_t)job) * (int64_t)91200) + ((int64_t)packet_begin)) + ((int64_t)(column * packet_stride)))], pool[(((selected / 9) * 456) + (panel * 16))], W[(((((int64_t)job) * (int64_t)91200) + ((int64_t)packet_begin)) + ((int64_t)(column * packet_stride)))], 1, packet_stride, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          }
        }
      }
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
      if (valid != 0) {
        tl::ascend::pv_load_rows(pool[packets], W[((((int64_t)job) * (int64_t)91200) + ((int64_t)packet_begin))], pool[packets], 1, ((200 - max((panel * 16), 184)) * packet_stride), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        for (int32_t column_1 = 0; column_1 < (200 - max((panel * 16), 184)); ++column_1) {
          tl::ascend::pv_gather_offsets(pool[product], pool[packets], pool[packet_indices], (column_1 * 4), (200 - max((panel * 16), 184)));
          AscendC::PipeBarrier<PIPE_V>();
          float reciprocal = pool.GetValue(((((int64_t)meta) + ((int64_t)column_1)) + (int64_t)16));
          tl::ascend::pv_scale(pool[(packets + (column_1 * packet_stride))], pool[(packets + (column_1 * packet_stride))], pool[(packets + (column_1 * packet_stride))], (reciprocal * -1.000000e+00f), packet_stride);
          AscendC::PipeBarrier<PIPE_V>();
          if ((max((panel * 16), 184) + column_1) < 199) {
            tl::ascend::pv_brcb_rows(pool[factors], pool[product], pool[factors], 16, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t chunk_1 = 0; chunk_1 < ((packet_stride + 63) / 64); ++chunk_1) {
              tl::ascend::pv_mad_strided(pool[(((chunk_1 * 64) + packets) + ((column_1 + 1) * packet_stride))], pool[(((chunk_1 * 64) + packets) + (column_1 * packet_stride))], pool[(((column_1 * 8) + factors) + 8)], min((packet_stride - (chunk_1 * 64)), 64), ((199 - max((panel * 16), 184)) - column_1), 1, 1, 0, (packet_stride / 8), 0, 1);
            }
            AscendC::PipeBarrier<PIPE_V>();
          }
        }
        for (int32_t column_2 = 0; column_2 < (200 - max((panel * 16), 184)); ++column_2) {
          int32_t selected_1 = ((int32_t)pool.GetValue((((int64_t)meta) + ((int64_t)column_2))));
          tl::ascend::pv_gather_offsets(pool[vector], pool[0], pool[(red + 32)], ((panel * 64) + (column_2 * 4)), factor_rows);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_brcb_rows(pool[factors], pool[vector], pool[factors], factor_rows, 8);
          AscendC::PipeBarrier<PIPE_V>();
          bool consumed = (((column_2 % 8) == 7) || (((panel * 16) + column_2) == 199));
          int32_t condval_2;
          if (consumed) {
            condval_2 = 8;
          } else {
            condval_2 = 0;
          }
          int32_t condval_3;
          if (consumed) {
            condval_3 = 0;
          } else {
            condval_3 = 8;
          }
          int32_t live_limit = (((((panel * 16) + ((column_2 / 8) * 8)) + condval_2) + k) + condval_3);
          for (int32_t row_group = 0; row_group < ((owned + 7) / 8); ++row_group) {
            int32_t lane_rows = min((owned - (row_group * 8)), 8);
            int32_t condval_4;
            if (consumed) {
              condval_4 = 8;
            } else {
              condval_4 = 0;
            }
            int32_t condval_5;
            if (consumed) {
              condval_5 = 8;
            } else {
              condval_5 = 0;
            }
            int32_t condval_6;
            if (consumed) {
              condval_6 = 8;
            } else {
              condval_6 = 0;
            }
            tl::ascend::pv_mad_strided(pool[((((row_group * 3648) + (panel * 16)) + ((column_2 / 8) * 8)) + condval_4)], pool[((row_group * 64) + factors)], pool[(((((column_2 / 8) * 8) + packets) + (column_2 * packet_stride)) + condval_5)], (lane_rows * 8), ((((live_limit - condval_6) / 8) - (column_2 / 8)) - (panel * 2)), 57, 1, 0, 1, 0, 1);
          }
          AscendC::PipeBarrier<PIPE_V>();
          if ((selected_1 % 9) == core) {
            tl::ascend::pv_copy(pool[(((selected_1 / 9) * 456) + (panel * 16))], pool[(packets + (column_2 * packet_stride))], pool[(((selected_1 / 9) * 456) + (panel * 16))], packet_stride);
            AscendC::PipeBarrier<PIPE_V>();
          }
        }
      }
      if (((0 < window) && ((min(0, ((panel * 16) - 184)) + 200) == window_end)) && (window_end < 200)) {
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 3);
        if (valid != 0) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          int32_t next_columns = min(window, (200 - window_end));
          tl::ascend::pv_store_rows(W[(((((int64_t)job) * (int64_t)91200) + ((int64_t)(core * k))) + ((int64_t)window_end))], pool[window_end], W[(((((int64_t)job) * (int64_t)91200) + ((int64_t)(core * k))) + ((int64_t)window_end))], owned, next_columns, (456 - next_columns), ((k * 9) - next_columns));
        }
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
        window_end = min(200, (window_end + window));
      }
    }
    inverse = 0.000000e+00f;
    peak = 0.000000e+00f;
    if (valid != 0) {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      tl::ascend::pv_scale(pool[(packet_indices + 16)], pool[(meta + 64)], pool[(packet_indices + 16)], -4.000000e+00f, 200);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_add_scalar(pool[(packet_indices + 16)], pool[(packet_indices + 16)], pool[(packet_indices + 16)], -4.000000e+00f, 200);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_cast_indices(pool[(packet_indices + 16)], pool[(packet_indices + 16)], pool[(packet_indices + 16)], 200);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pf_shift(pool[(packet_indices + 16)], pool[(packet_indices + 16)], pool[(packet_indices + 16)], (k * 4), 200);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_brcb_rows(pool[factors], pool[(red + 32)], pool[factors], factor_rows, 8);
      AscendC::PipeBarrier<PIPE_V>();
      for (int32_t chunk_2 = 0; chunk_2 < ((k + 63) / 64); ++chunk_2) {
        tl::ascend::pf_indices(pool[((chunk_2 * 64) + 33000)], pool[factors], pool[(((chunk_2 * 64) + packet_indices) + 16)], min((k - (chunk_2 * 64)), 64), owned, k);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_gather_offsets(pool[output], pool[0], pool[33000], 0, (owned * k));
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_mul(pool[0], pool[output], pool[output], (owned * k));
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(pool[red], pool[0], pool[(owned * k)], (owned * k));
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      inverse = (pool.GetValue(((int64_t)red)) * 1.000010e+00f);
      tl::ascend::pv_scale(pool[output], pool[output], pool[output], (-1.000000e+00f / power), (owned * k));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      for (int32_t i_2 = 0; i_2 < owned; ++i_2) {
        int32_t target = (((int32_t)(pool.GetValue(((((((int64_t)i_2) * (int64_t)9) + ((int64_t)meta)) + ((int64_t)core)) + (int64_t)64)) * -1.000000e+00f)) - 1);
        tl::ascend::pv_store_rows(X[((((int64_t)job) * (int64_t)40000) + (((int64_t)target) * (int64_t)200))], pool[(output + (i_2 * k))], X[((((int64_t)job) * (int64_t)40000) + (((int64_t)target) * (int64_t)200))], 1, 200, 0, 0);
      }
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    }
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, inverse);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((((int64_t)all_batch) * (int64_t)182400) + (((int64_t)job) * (int64_t)8192)) + (((int64_t)core) * (int64_t)8)) + (int64_t)2048)], pool[0], W[((((((int64_t)all_batch) * (int64_t)182400) + (((int64_t)job) * (int64_t)8192)) + (((int64_t)core) * (int64_t)8)) + (int64_t)2048)], 1, 2, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}


}
