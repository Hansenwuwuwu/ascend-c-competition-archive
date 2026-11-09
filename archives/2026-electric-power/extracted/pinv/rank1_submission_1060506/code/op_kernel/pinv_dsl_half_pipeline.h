#pragma once
#ifndef PINV_EMBEDDED_PINV_DSL_HALF_ROWS_H
#define PINV_EMBEDDED_PINV_DSL_HALF_ROWS_H
#include "pinv_dsl_common.h"
#ifndef PINV_EMBEDDED_PINV_HALF_INSTRUCTIONS_H
#define PINV_EMBEDDED_PINV_HALF_INSTRUCTIONS_H
#include "kernel_operator.h"
namespace tl::ascend {
using namespace AscendC;
__aicore__ inline void hp_copy(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int n){DataCopy(d,a,n);}
__aicore__ inline void hp_scale(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,float v,int n){Muls(d,a,static_cast<half>(v),n);}
__aicore__ inline void hp_gather(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<float>i,uint32_t base,int n){Gather(d,a,i.ReinterpretCast<uint32_t>(),base,n);}
__aicore__ inline void hp_brcb(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int n){Brcb(d,a,uint8_t(n/8),BrcbRepeatParams{1,8});}
__aicore__ inline void hp_mad(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>b,int width,int rows,int pitch){MulAddDst(d,a,b,uint64_t(width),uint8_t(rows),BinaryRepeatParams{1,1,0,uint8_t(pitch/16),0,1});}
__aicore__ inline void hp_compact(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int rows,int width,int sourceGap){DataCopy(d,a,DataCopyParams{uint16_t(rows),uint16_t(width/16),uint16_t(sourceGap/16),0});}
__aicore__ inline void hp_float(LocalTensor<float>d,LocalTensor<half>a,LocalTensor<float>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
__aicore__ inline void hp_half(LocalTensor<half>d,LocalTensor<float>a,LocalTensor<half>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
__aicore__ inline void hp_rows(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int rows,int width,int sourceGap,int destinationGap){DataCopy(d,a,DataCopyParams{uint16_t(rows),uint16_t(width/16),uint16_t(sourceGap/16),uint16_t(destinationGap/16)});}
__aicore__ inline void hp_store_header(GlobalTensor<float>d,LocalTensor<half>a,GlobalTensor<float>,int rows,int pitch){GlobalTensor<half> target;target.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(const_cast<__gm__ float*>(d.GetPhyAddr())));DataCopy(target,a,DataCopyParams{uint16_t(rows),1,uint16_t(pitch/16-1),0});}
__aicore__ inline void hp_load_header(LocalTensor<half>d,GlobalTensor<float>a,LocalTensor<half>,int rows,int pitch){GlobalTensor<half> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(const_cast<__gm__ float*>(a.GetPhyAddr())));DataCopy(d,source,DataCopyParams{uint16_t(rows),1,0,uint16_t(pitch/16-1)});}
__aicore__ inline void hp_abs(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int n){Abs(d,a,n);}
__aicore__ inline void hp_whole_max(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int mask,int repeats){WholeReduceMax(d,a,uint64_t(mask),uint8_t(repeats),1,1,8);}
__aicore__ inline void hp_reduce_scalars(LocalTensor<half>,LocalTensor<half>,LocalTensor<half>,float &best,int32_t &position){half value,bits;GetReduceRepeatMaxMinSpr<half>(value,bits);best=static_cast<float>(value);union{half value;uint16_t word;} index;index.value=bits;position=int32_t(index.word);}
__aicore__ inline void hp_index(LocalTensor<half>d,LocalTensor<half>,LocalTensor<half>,int offset,int32_t &position){position=int32_t(d.ReinterpretCast<uint16_t>().GetValue(offset));}
}
namespace tl::ascend {
__aicore__ inline void pv_add_scalar(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,float v,int n){AscendC::Adds(d,a,v,n);}
__aicore__ inline void pv_cast_indices(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int n){AscendC::Cast(d.ReinterpretCast<int32_t>(),a,AscendC::RoundMode::CAST_RINT,n);}
__aicore__ inline void pv_mask_width(AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int width){AscendC::SetMaskNorm();AscendC::SetVectorMask<float>(width);}
__aicore__ inline void pv_mad_reuse(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int rows,int pitch){AscendC::MulAddDst<float,float,false>(d,a,b,uint64_t(64),uint8_t(rows),AscendC::BinaryRepeatParams{1,1,0,uint8_t(pitch),0,1});}
}
#endif

namespace tl::ascend {
__aicore__ inline void PvDslHalfRows( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, uint32_t n, uint32_t stride, uint32_t all_batch, uint32_t job, uint32_t core, uint32_t cores, uint32_t window, uint64_t fftsAddr) {
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
  uint32_t k = (uint32_t)0;
  uint32_t wide = (uint32_t)0;
  uint32_t owned = (uint32_t)0;
  uint32_t factor_rows = (uint32_t)0;
  uint32_t packet_stride = (uint32_t)0;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  int valid = 0;
  float power = 0.000000e+00f;
  uint32_t window_end = (uint32_t)0;
  float inverse = 0.000000e+00f;
  float peak = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    k = ((((n + (uint32_t)8) - (uint32_t)1) / (uint32_t)8) * (uint32_t)8);
    wide = ((((k + (uint32_t)64) - (uint32_t)1) / (uint32_t)64) * (uint32_t)64);
    owned = ((((n + cores) - (uint32_t)1) - core) / cores);
    factor_rows = ((((owned + (uint32_t)8) - (uint32_t)1) / (uint32_t)8) * (uint32_t)8);
    packet_stride = (k + (uint32_t)16);
    uint32_t packets = (owned * stride);
    uint32_t output = (packets + ((uint32_t)16 * packet_stride));
    uint32_t factors = (output + max((owned * k), (uint32_t)512));
    uint32_t meta = (factors + ((uint32_t)8 * max(factor_rows, (uint32_t)16)));
    uint32_t norms = (meta + (uint32_t)64);
    uint32_t vector = (norms + wide);
    uint32_t product = (vector + wide);
    uint32_t accumulator = (product + wide);
    uint32_t scratch = (accumulator + wide);
    uint32_t red = (scratch + wide);
    uint32_t row_indices = (red + (uint32_t)32);
    uint32_t packet_indices = (row_indices + factor_rows);
    uint32_t mapping = (packet_indices + (uint32_t)16);
    uint64_t work = ((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)stride));
    uint64_t metadata = ((((((uint64_t)all_batch) * (uint64_t)2) * ((uint64_t)n)) * ((uint64_t)stride)) + (((uint64_t)job) * (uint64_t)8192));
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (owned * stride));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[0], A[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(core * n)))], pool[0], owned, n, ((cores - (uint32_t)1) * n), (stride - k));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::pv_fill(pool[norms], pool[norms], pool[norms], 1.000000e+00f, k);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    for (uint32_t i = (uint32_t)0; i < factor_rows; ++i) {
      uint32_t condval;
      if ((i < owned)) {
        condval = ((i * stride) * (uint32_t)4);
      } else {
        condval = (uint32_t)0;
      }
      uint32_t index = condval;
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (row_indices + i), index);
    }
    for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (packet_indices + ((uint32_t)i_1)), ((((uint32_t)i_1) * packet_stride) * (uint32_t)4));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    valid = 1;
    power = 1.000000e+00f;
    window_end = n;
    if ((uint32_t)0 < window) {
      window_end = min(n, window);
    }
    for (uint32_t panel = (uint32_t)0; panel < (((n + (uint32_t)16) - (uint32_t)1) / (uint32_t)16); ++panel) {
      uint32_t first = (panel * (uint32_t)16);
      uint32_t rows = min((n - first), (uint32_t)16);
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_load_rows(pool[meta], W[(metadata + ((uint64_t)(panel * (uint32_t)64)))], pool[meta], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      if (pool.GetValue((meta + (uint32_t)34)) <= 5.000000e-01f) {
        valid = 0;
      }
      power = pool.GetValue((meta + (uint32_t)32));
      uint32_t condval_1;
      if (((uint32_t)0 < window)) {
        condval_1 = (first * packet_stride);
      } else {
        condval_1 = (uint32_t)0;
      }
      uint32_t packet_begin = condval_1;
      if (valid != 0) {
        if (first == (uint32_t)0) {
          for (uint32_t chunk = (uint32_t)0; chunk < ((((owned * stride) + (uint32_t)16320) - (uint32_t)1) / (uint32_t)16320); ++chunk) {
            uint32_t begin = (chunk * (uint32_t)16320);
            tl::ascend::pv_scale(pool[begin], pool[begin], pool[begin], (1.000000e+00f / power), min(((owned * stride) - begin), (uint32_t)16320));
          }
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        for (uint32_t column = (uint32_t)0; column < rows; ++column) {
          int32_t selected = ((int32_t)pool.GetValue((meta + column)));
          pool.SetValue((norms + ((uint32_t)selected)), (((float)((int32_t)((first + column) + (uint32_t)1))) * -1.000000e+00f));
          if ((((uint32_t)selected) % cores) == core) {
            pool.SetValue((((((((uint32_t)selected) / cores) * stride) + k) + first) + column), 1.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(W[((work + ((uint64_t)packet_begin)) + ((uint64_t)(column * packet_stride)))], pool[(((((uint32_t)selected) / cores) * stride) + first)], W[((work + ((uint64_t)packet_begin)) + ((uint64_t)(column * packet_stride)))], 1, packet_stride, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          }
        }
      }
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
      if (valid != 0) {
        tl::ascend::pv_load_rows(pool[packets], W[(work + ((uint64_t)packet_begin))], pool[packets], 1, (rows * packet_stride), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        for (uint32_t column_1 = (uint32_t)0; column_1 < rows; ++column_1) {
          tl::ascend::pv_gather_offsets(pool[product], pool[packets], pool[packet_indices], (column_1 * (uint32_t)4), rows);
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          float reciprocal = (-1.000000e+00f / pool.GetValue(((packets + (column_1 * packet_stride)) + column_1)));
          tl::ascend::pv_scale(pool[(packets + (column_1 * packet_stride))], pool[(packets + (column_1 * packet_stride))], pool[(packets + (column_1 * packet_stride))], reciprocal, packet_stride);
          if ((column_1 + (uint32_t)1) < rows) {
            tl::ascend::pv_brcb_rows(pool[factors], pool[product], pool[factors], 16, 8);
            uint32_t full_packet = (packet_stride / (uint32_t)64);
            if ((uint32_t)0 < full_packet) {
              tl::ascend::pv_mask_width(pool[0], pool[0], pool[0], 64);
              for (uint32_t chunk_1 = (uint32_t)0; chunk_1 < full_packet; ++chunk_1) {
                uint32_t col = (chunk_1 * (uint32_t)64);
                tl::ascend::pv_mad_reuse(pool[((packets + ((column_1 + (uint32_t)1) * packet_stride)) + col)], pool[((packets + (column_1 * packet_stride)) + col)], pool[(factors + ((column_1 + (uint32_t)1) * (uint32_t)8))], ((rows - column_1) - (uint32_t)1), (packet_stride / (uint32_t)8));
              }
            }
            uint32_t tail_packet = (packet_stride - (full_packet * (uint32_t)64));
            if ((uint32_t)0 < tail_packet) {
              uint32_t col_1 = (full_packet * (uint32_t)64);
              tl::ascend::pv_mad_strided(pool[((packets + ((column_1 + (uint32_t)1) * packet_stride)) + col_1)], pool[((packets + (column_1 * packet_stride)) + col_1)], pool[(factors + ((column_1 + (uint32_t)1) * (uint32_t)8))], tail_packet, ((rows - column_1) - (uint32_t)1), 1, 1, 0, (packet_stride / (uint32_t)8), 0, 1);
            }
          }
        }
        for (uint32_t column_2 = (uint32_t)0; column_2 < rows; ++column_2) {
          int32_t selected_1 = ((int32_t)pool.GetValue((meta + column_2)));
          tl::ascend::pv_gather_offsets(pool[vector], pool[0], pool[row_indices], ((first + column_2) * (uint32_t)4), factor_rows);
          tl::ascend::pv_brcb_rows(pool[factors], pool[vector], pool[factors], factor_rows, 8);
          uint32_t pivot_column = (first + column_2);
          bool consumed = (((pivot_column % (uint32_t)8) == (uint32_t)7) || ((pivot_column + (uint32_t)1) == n));
          int32_t condval_2;
          if (consumed) {
            condval_2 = 8;
          } else {
            condval_2 = 0;
          }
          uint32_t live_begin = (((pivot_column / (uint32_t)8) * (uint32_t)8) + ((uint32_t)condval_2));
          int32_t condval_3;
          if (consumed) {
            condval_3 = 0;
          } else {
            condval_3 = 8;
          }
          uint32_t live_limit = ((live_begin + k) + ((uint32_t)condval_3));
          uint32_t full = ((live_limit - live_begin) / (uint32_t)64);
          if ((uint32_t)0 < full) {
            tl::ascend::pv_mask_width(pool[0], pool[0], pool[0], 64);
            for (uint32_t chunk_2 = (uint32_t)0; chunk_2 < full; ++chunk_2) {
              uint32_t col_2 = (live_begin + (chunk_2 * (uint32_t)64));
              tl::ascend::pv_mad_reuse(pool[col_2], pool[(((packets + (column_2 * packet_stride)) + col_2) - first)], pool[factors], owned, (stride / (uint32_t)8));
            }
          }
          uint32_t tail = ((live_limit - live_begin) - (full * (uint32_t)64));
          if ((uint32_t)0 < tail) {
            uint32_t col_3 = (live_begin + (full * (uint32_t)64));
            tl::ascend::pv_mad_strided(pool[col_3], pool[(((packets + (column_2 * packet_stride)) + col_3) - first)], pool[factors], tail, owned, 1, 1, 0, (stride / (uint32_t)8), 0, 1);
          }
          if ((((uint32_t)selected_1) % cores) == core) {
            tl::ascend::pv_copy(pool[(((((uint32_t)selected_1) / cores) * stride) + first)], pool[(packets + (column_2 * packet_stride))], pool[(((((uint32_t)selected_1) / cores) * stride) + first)], packet_stride);
          }
        }
      }
      if ((((uint32_t)0 < window) && ((first + rows) == window_end)) && (window_end < n)) {
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 3);
        if (valid != 0) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          uint32_t next_columns = min(window, (n - window_end));
          tl::ascend::pv_store_rows(W[((work + ((uint64_t)(core * k))) + ((uint64_t)window_end))], pool[window_end], W[((work + ((uint64_t)(core * k))) + ((uint64_t)window_end))], owned, next_columns, (stride - next_columns), ((cores * k) - next_columns));
        }
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
        window_end = min(n, (window_end + window));
      }
    }
    inverse = 0.000000e+00f;
    peak = 0.000000e+00f;
    if (valid != 0) {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      tl::ascend::pv_scale(pool[mapping], pool[norms], pool[mapping], -4.000000e+00f, n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_add_scalar(pool[mapping], pool[mapping], pool[mapping], -4.000000e+00f, n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_cast_indices(pool[mapping], pool[mapping], pool[mapping], n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_fill(pool[accumulator], pool[accumulator], pool[accumulator], 0.000000e+00f, wide);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t i_2 = (uint32_t)0; i_2 < owned; ++i_2) {
        tl::ascend::pv_gather_offsets(pool[(output + (i_2 * k))], pool[((i_2 * stride) + k)], pool[mapping], 0, n);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mul(pool[product], pool[(output + (i_2 * k))], pool[(output + (i_2 * k))], n);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_add(pool[accumulator], pool[accumulator], pool[product], n);
        AscendC::PipeBarrier<PIPE_V>();
      }
      tl::ascend::pv_reduce_sum(pool[red], pool[accumulator], pool[scratch], k);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      inverse = pool.GetValue(red);
      for (uint32_t i_3 = (uint32_t)0; i_3 < owned; ++i_3) {
        tl::ascend::pv_scale(pool[(output + (i_3 * k))], pool[(output + (i_3 * k))], pool[(output + (i_3 * k))], (-1.000000e+00f / power), k);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t target = (((int32_t)(pool.GetValue(((norms + core) + (i_3 * cores))) * -1.000000e+00f)) - 1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(X[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(((uint32_t)target) * n)))], pool[(output + (i_3 * k))], X[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(((uint32_t)target) * n)))], 1, n, 0, 0);
      }
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    }
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, inverse);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], pool[0], W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], 1, 2, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}

}
#endif

#ifndef PINV_EMBEDDED_PINV_DSL_HALF_SELECTOR_H
#define PINV_EMBEDDED_PINV_DSL_HALF_SELECTOR_H
#include "pinv_dsl_common.h"
#ifndef PINV_EMBEDDED_PINV_HALF_INSTRUCTIONS_H
#define PINV_EMBEDDED_PINV_HALF_INSTRUCTIONS_H
#include "kernel_operator.h"
namespace tl::ascend {
using namespace AscendC;
__aicore__ inline void hp_copy(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int n){DataCopy(d,a,n);}
__aicore__ inline void hp_scale(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,float v,int n){Muls(d,a,static_cast<half>(v),n);}
__aicore__ inline void hp_gather(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<float>i,uint32_t base,int n){Gather(d,a,i.ReinterpretCast<uint32_t>(),base,n);}
__aicore__ inline void hp_brcb(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int n){Brcb(d,a,uint8_t(n/8),BrcbRepeatParams{1,8});}
__aicore__ inline void hp_mad(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>b,int width,int rows,int pitch){MulAddDst(d,a,b,uint64_t(width),uint8_t(rows),BinaryRepeatParams{1,1,0,uint8_t(pitch/16),0,1});}
__aicore__ inline void hp_compact(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int rows,int width,int sourceGap){DataCopy(d,a,DataCopyParams{uint16_t(rows),uint16_t(width/16),uint16_t(sourceGap/16),0});}
__aicore__ inline void hp_float(LocalTensor<float>d,LocalTensor<half>a,LocalTensor<float>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
__aicore__ inline void hp_half(LocalTensor<half>d,LocalTensor<float>a,LocalTensor<half>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
__aicore__ inline void hp_rows(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int rows,int width,int sourceGap,int destinationGap){DataCopy(d,a,DataCopyParams{uint16_t(rows),uint16_t(width/16),uint16_t(sourceGap/16),uint16_t(destinationGap/16)});}
__aicore__ inline void hp_store_header(GlobalTensor<float>d,LocalTensor<half>a,GlobalTensor<float>,int rows,int pitch){GlobalTensor<half> target;target.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(const_cast<__gm__ float*>(d.GetPhyAddr())));DataCopy(target,a,DataCopyParams{uint16_t(rows),1,uint16_t(pitch/16-1),0});}
__aicore__ inline void hp_load_header(LocalTensor<half>d,GlobalTensor<float>a,LocalTensor<half>,int rows,int pitch){GlobalTensor<half> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(const_cast<__gm__ float*>(a.GetPhyAddr())));DataCopy(d,source,DataCopyParams{uint16_t(rows),1,0,uint16_t(pitch/16-1)});}
__aicore__ inline void hp_abs(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int n){Abs(d,a,n);}
__aicore__ inline void hp_whole_max(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int mask,int repeats){WholeReduceMax(d,a,uint64_t(mask),uint8_t(repeats),1,1,8);}
__aicore__ inline void hp_reduce_scalars(LocalTensor<half>,LocalTensor<half>,LocalTensor<half>,float &best,int32_t &position){half value,bits;GetReduceRepeatMaxMinSpr<half>(value,bits);best=static_cast<float>(value);union{half value;uint16_t word;} index;index.value=bits;position=int32_t(index.word);}
__aicore__ inline void hp_index(LocalTensor<half>d,LocalTensor<half>,LocalTensor<half>,int offset,int32_t &position){position=int32_t(d.ReinterpretCast<uint16_t>().GetValue(offset));}
}
namespace tl::ascend {
__aicore__ inline void pv_add_scalar(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,float v,int n){AscendC::Adds(d,a,v,n);}
__aicore__ inline void pv_cast_indices(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,int n){AscendC::Cast(d.ReinterpretCast<int32_t>(),a,AscendC::RoundMode::CAST_RINT,n);}
__aicore__ inline void pv_mask_width(AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int width){AscendC::SetMaskNorm();AscendC::SetVectorMask<float>(width);}
__aicore__ inline void pv_mad_reuse(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int rows,int pitch){AscendC::MulAddDst<float,float,false>(d,a,b,uint64_t(64),uint8_t(rows),AscendC::BinaryRepeatParams{1,1,0,uint8_t(pitch),0,1});}
}
#endif

namespace tl::ascend {
__aicore__ inline void PvDslHalfSelector( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, int32_t n, int32_t stride, int32_t all_batch, int32_t job, int32_t core, int32_t workers, int32_t window, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 188416);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int role = 0;
  int k = 0;
  int wide = 0;
  int split = 0;
  int own_width = 0;
  int pitch = 0;
  int capacity = 0;
  int actual_width = 0;
  int prefix = 0;
  int source_column = 0;
  auto state = ascend_ub.GetWithOffset<float>(32768, 0);
  float local_scale = 0.000000e+00f;
  float local_total = 0.000000e+00f;
  float local_power = 0.000000e+00f;
  int valid = 0;
  float power = 0.000000e+00f;
  auto matrix = ascend_ub.GetWithOffset<half>(24576, 131072);
  int size = 0;
  int header_issued = 0;
  int pending_ready = 0;
  int owner = 0;
  int drop = 0;
  int old_prefix = 0;
  int new_width = 0;
  int next_owner = 0;
  int new_prefix = 0;
  int next_pitch = 0;
  auto half_work = ascend_ub.GetWithOffset<half>(4096, 180224);
  float best = 0.000000e+00f;
  int chunk = 0;
  int local_index = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    role = (core - workers);
    k = (((n + 7) / 8) * 8);
    wide = (((k + 63) / 64) * 64);
    split = (((n + 31) / 32) * 16);
    own_width = split;
    if (role != 0) {
      own_width = ((((n + 15) - split) / 16) * 16);
    }
    pitch = own_width;
    if (role != 0) {
      pitch = (pitch + 16);
    }
    if (pitch < 16) {
      pitch = 16;
    }
    capacity = (n * pitch);
    int32_t scratch = capacity;
    int32_t vector = (((k * 8) + scratch) + 4096);
    int32_t product = (vector + wide);
    int32_t temporary = (product + wide);
    int32_t permutation = ((temporary + k) + 64);
    int32_t records = (permutation + k);
    actual_width = split;
    prefix = 0;
    if (role != 0) {
      actual_width = (n - split);
      prefix = 16;
    }
    source_column = 0;
    if (role != 0) {
      source_column = split;
    }
    tl::ascend::pv_fill(state[0], state[0], state[0], 0.000000e+00f, capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(state[prefix], A[(((job * n) * n) + source_column)], state[prefix], n, actual_width, (n - actual_width), (pitch - (((actual_width + 7) / 8) * 8)));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    local_scale = 0.000000e+00f;
    local_total = 0.000000e+00f;
    local_power = 1.000000e+00f;
    for (int32_t chunk_1 = 0; chunk_1 < ((capacity + 4095) / 4096); ++chunk_1) {
      int32_t count = min((capacity - (chunk_1 * 4096)), 4096);
      tl::ascend::pv_abs(state[scratch], state[(chunk_1 * 4096)], state[scratch], count);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(state[(temporary + 32)], state[scratch], state[product], count);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      local_scale = max(local_scale, state.GetValue((((int64_t)temporary) + (int64_t)32)));
    }
    if ((0.000000e+00f < local_scale) && (local_scale < 1.701412e+38f)) {
      while (1) {
        if (!((local_power < local_scale))) { break; }
        local_power = (local_power * 2.000000e+00f);
      }
      while (1) {
        if (!((local_scale <= (local_power * 5.000000e-01f)))) { break; }
        local_power = (local_power * 5.000000e-01f);
      }
    }
    for (int32_t chunk_2 = 0; chunk_2 < ((capacity + 16319) / 16320); ++chunk_2) {
      tl::ascend::pv_scale(state[(chunk_2 * 16320)], state[(chunk_2 * 16320)], state[(chunk_2 * 16320)], (1.000000e+00f / local_power), min((capacity - (chunk_2 * 16320)), 16320));
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t chunk_3 = 0; chunk_3 < ((capacity + 4095) / 4096); ++chunk_3) {
      int32_t count_1 = min((capacity - (chunk_3 * 4096)), 4096);
      tl::ascend::pv_mul(state[scratch], state[(chunk_3 * 4096)], state[(chunk_3 * 4096)], count_1);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(state[(temporary + 32)], state[scratch], state[product], count_1);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      local_total = (local_total + state.GetValue((((int64_t)temporary) + (int64_t)32)));
    }
    state.SetValue(((int64_t)scratch), local_scale);
    state.SetValue((((int64_t)scratch) + (int64_t)1), local_total);
    state.SetValue((((int64_t)scratch) + (int64_t)2), local_power);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((job * 8192) + (role * 8)) + (((all_batch * n) * stride) * 2)) + 3072)], state[scratch], W[((((job * 8192) + (role * 8)) + (((all_batch * n) * stride) * 2)) + 3072)], 1, 8, 0, 0);
    tl::ascend::pv_cross_set1(state[0], state[0], state[0], 4);
    tl::ascend::pv_cross_wait(state[0], state[0], state[0], 4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    tl::ascend::pv_load_rows(state[scratch], W[(((job * 8192) + (((all_batch * n) * stride) * 2)) + 3072)], state[scratch], 1, 16, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    float magnitude = max(state.GetValue(((int64_t)scratch)), state.GetValue((((int64_t)scratch) + (int64_t)8)));
    float sum0 = state.GetValue((((int64_t)scratch) + (int64_t)1));
    float sum1 = state.GetValue((((int64_t)scratch) + (int64_t)9));
    float power0 = state.GetValue((((int64_t)scratch) + (int64_t)2));
    float power1 = state.GetValue((((int64_t)scratch) + (int64_t)10));
    tl::ascend::pv_cross_set1(state[0], state[0], state[0], 5);
    tl::ascend::pv_cross_wait(state[0], state[0], state[0], 5);
    valid = 1;
    if ((magnitude < 1.000000e-01f) || (1.701412e+38f <= magnitude)) {
      valid = 0;
    }
    power = 1.000000e+00f;
    if (valid != 0) {
      while (1) {
        if (!((power < magnitude))) { break; }
        power = (power * 2.000000e+00f);
      }
      while (1) {
        if (!((magnitude <= (power * 5.000000e-01f)))) { break; }
        power = (power * 5.000000e-01f);
      }
    }
    float ratio0 = (power0 / power);
    float ratio1 = (power1 / power);
    float total = ((((sum0 * ratio0) * ratio0) + ((sum1 * ratio1) * ratio1)) * 1.000010e+00f);
    for (int32_t chunk_4 = 0; chunk_4 < ((capacity + 16319) / 16320); ++chunk_4) {
      tl::ascend::pv_scale(state[(chunk_4 * 16320)], state[(chunk_4 * 16320)], state[(chunk_4 * 16320)], (local_power / power), min((capacity - (chunk_4 * 16320)), 16320));
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t chunk_5 = 0; chunk_5 < ((capacity + 16319) / 16320); ++chunk_5) {
      tl::ascend::hp_half(matrix[(chunk_5 * 16320)], state[(chunk_5 * 16320)], matrix[(chunk_5 * 16320)], min((capacity - (chunk_5 * 16320)), 16320));
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t row = 0; row < k; ++row) {
      state.SetValue((((int64_t)permutation) + ((int64_t)row)), ((float)row));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    size = n;
    header_issued = 0;
    pending_ready = 0;
    for (int32_t panel = 0; panel < ((n + 15) / 16); ++panel) {
      if (valid != 0) {
        if (header_issued != 0) {
          if (pending_ready != 0) {
            tl::ascend::pv_cross_wait(state[0], state[0], state[0], 7);
          }
          tl::ascend::pv_cross_wait(state[0], state[0], state[0], 9);
          header_issued = 0;
        }
        owner = 0;
        if (split <= (panel * 16)) {
          owner = 1;
        }
        if (role == owner) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::hp_store_header(W[(((job * 8192) + (((all_batch * n) * stride) * 2)) + 3072)], matrix[0], W[(((job * 8192) + (((all_batch * n) * stride) * 2)) + 3072)], size, pitch);
          tl::ascend::pv_cross_set1(state[0], state[0], state[0], 7);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          pending_ready = 1;
        } else {
          tl::ascend::pv_cross_set1(state[0], state[0], state[0], 7);
          tl::ascend::pv_cross_wait(state[0], state[0], state[0], 7);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
          tl::ascend::hp_load_header(matrix[0], W[(((job * 8192) + (((all_batch * n) * stride) * 2)) + 3072)], matrix[0], size, pitch);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          pending_ready = 0;
        }
        tl::ascend::pv_cross_set1(state[0], state[0], state[0], 9);
        header_issued = 1;
        tl::ascend::pv_iota(state[(temporary + 64)], state[(temporary + 64)], state[(temporary + 64)], size);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_scale_indices(state[(temporary + 64)], state[(temporary + 64)], state[(temporary + 64)], (pitch * 2), size);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t block = min(size, 16);
        for (int32_t pivot = 0; pivot < block; ++pivot) {
          int32_t remaining = (size - pivot);
          tl::ascend::hp_gather(half_work[0], matrix[0], state[(temporary + 64)], ((pivot * (pitch + 1)) * 2), remaining);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::hp_abs(half_work[3840], half_work[0], half_work[3840], remaining);
          AscendC::PipeBarrier<PIPE_V>();
          if (remaining <= 128) {
            tl::ascend::hp_whole_max(half_work[3728], half_work[3840], half_work[3728], remaining, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            tl::ascend::hp_reduce_scalars(half_work[0], half_work[0], half_work[0], best, chunk);
          } else {
            tl::ascend::hp_whole_max(half_work[3728], half_work[3840], half_work[3728], 128, 1);
            tl::ascend::hp_whole_max(half_work[3730], half_work[3968], half_work[3730], (remaining - 128), 1);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::hp_float(state[product], half_work[3728], state[product], 8);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_even_max(state[temporary], state[product], state[temporary], 2);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            tl::ascend::pv_reduce_scalars(state[0], state[0], state[0], best, chunk);
            tl::ascend::hp_index(half_work[3728], half_work[3728], half_work[3728], (chunk + 1), local_index);
            chunk = (((chunk / 2) * 128) + local_index);
          }
          int32_t selected = (pivot + chunk);
          if (best <= 1.000000e-06f) {
            valid = 0;
            break;
          }
          float divisor = ((float)half_work.GetValue((((int64_t)selected) - ((int64_t)pivot))));
          if (selected != pivot) {
            float old = state.GetValue((((int64_t)permutation) + ((int64_t)pivot)));
            state.SetValue((((int64_t)permutation) + ((int64_t)pivot)), state.GetValue((((int64_t)permutation) + ((int64_t)selected))));
            state.SetValue((((int64_t)permutation) + ((int64_t)selected)), old);
          }
          state.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)), state.GetValue((((int64_t)permutation) + ((int64_t)pivot))));
          state.SetValue(((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)) + (int64_t)16), (1.000000e+00f / divisor));
          if (1 < remaining) {
            half_work.SetValue((((int64_t)selected) - ((int64_t)pivot)), half(0.000000e+00f));
            tl::ascend::hp_copy(half_work[3584], matrix[(selected * pitch)], half_work[3584], pitch);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::hp_scale(half_work[3584], half_work[3584], half_work[3584], (-1.000000e+00f / divisor), pitch);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            tl::ascend::hp_brcb(half_work[256], half_work[0], half_work[256], (((remaining + 7) / 8) * 8));
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t chunk_6 = 0; chunk_6 < (((pitch + 127) - ((pivot / 16) * 16)) / 128); ++chunk_6) {
              for (int32_t row_chunk = 0; row_chunk < ((remaining + 254) / 255); ++row_chunk) {
                tl::ascend::hp_mad(matrix[(((chunk_6 * 128) + ((pivot / 16) * 16)) + (((row_chunk * 255) + pivot) * pitch))], half_work[(((chunk_6 * 128) + ((pivot / 16) * 16)) + 3584)], half_work[((row_chunk * 4080) + 256)], min(((pitch - ((pivot / 16) * 16)) - (chunk_6 * 128)), 128), min((remaining - (row_chunk * 255)), 255), pitch);
              }
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (selected != pivot) {
              tl::ascend::hp_copy(matrix[(selected * pitch)], matrix[(pivot * pitch)], matrix[(selected * pitch)], pitch);
              AscendC::PipeBarrier<PIPE_V>();
            }
          }
        }
        int32_t remaining_1 = (size - block);
        drop = 0;
        old_prefix = 16;
        if (role == owner) {
          drop = block;
          old_prefix = 0;
        }
        new_width = (own_width - drop);
        next_owner = 0;
        if (split <= ((panel * 16) + block)) {
          next_owner = 1;
        }
        new_prefix = 16;
        if (role == next_owner) {
          new_prefix = 0;
        }
        next_pitch = max(16, (new_prefix + new_width));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        if ((0 < remaining_1) && (0 < new_width)) {
          for (int32_t row_chunk_1 = 0; row_chunk_1 < ((remaining_1 + 15) / 16); ++row_chunk_1) {
            tl::ascend::hp_rows(matrix[(((row_chunk_1 * next_pitch) * 16) + new_prefix)], matrix[(((((row_chunk_1 * 16) + block) * pitch) + old_prefix) + drop)], matrix[(((row_chunk_1 * next_pitch) * 16) + new_prefix)], min((remaining_1 - (row_chunk_1 * 16)), 16), new_width, (pitch - new_width), (next_pitch - new_width));
            AscendC::PipeBarrier<PIPE_V>();
          }
        }
        if (0 < remaining_1) {
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          tl::ascend::pv_copy(state[scratch], state[(permutation + block)], state[scratch], (((remaining_1 + 7) / 8) * 8));
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_copy(state[permutation], state[scratch], state[permutation], (((remaining_1 + 7) / 8) * 8));
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        size = remaining_1;
        own_width = new_width;
        pitch = next_pitch;
      }
      if (0 < panel) {
        tl::ascend::pv_cross_wait(state[0], state[0], state[0], 0);
        tl::ascend::pv_cross_wait(state[0], state[0], state[0], 1);
      }
      state.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)32), power);
      state.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)33), total);
      state.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)34), ((float)valid));
      if (role == 1) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        tl::ascend::pv_store_rows(W[(((job * 8192) + (panel * 64)) + (((all_batch * n) * stride) * 2))], state[((panel * 64) + records)], W[(((job * 8192) + (panel * 64)) + (((all_batch * n) * stride) * 2))], 1, 64, 0, 0);
      }
      tl::ascend::pv_cross_set0(state[0], state[0], state[0], 0);
      tl::ascend::pv_cross_set0(state[0], state[0], state[0], 1);
    }
    if (header_issued != 0) {
      if (pending_ready != 0) {
        tl::ascend::pv_cross_wait(state[0], state[0], state[0], 7);
      }
      tl::ascend::pv_cross_wait(state[0], state[0], state[0], 9);
    }
    tl::ascend::pv_cross_wait(state[0], state[0], state[0], 0);
    tl::ascend::pv_cross_wait(state[0], state[0], state[0], 1);
    tl::ascend::pv_all_barrier(state[0], state[0], state[0]);
    state.SetValue(0, ((float)valid));
    state.SetValue(1, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], state[0], W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], 1, 2, 0, 0);
    tl::ascend::pv_all_barrier(state[0], state[0], state[0]);
  }
  pipe.Destroy();
}

}
#endif

#undef pv_pipeline_rows
#define pv_pipeline_rows(d,a,b,...) PvDslHalfRows((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
#undef pv_pipeline_peer
#define pv_pipeline_peer(d,a,b,...) PvDslHalfSelector((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
__aicore__ inline void PvStageHalfPipeline( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, int32_t stride, int32_t window, float atol, float rtol, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 4096);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int valid = 0;
  auto pool = ascend_ub.GetWithOffset<float>(1024, 0);
  float inverse_total = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_pipeline_bind(A[0], X[0], W[0]);
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    if (((blocks * 2) / batch) <= ((bid % ((blocks * 2) / batch)) + 2)) {
      tl::ascend::pv_pipeline_peer(W[0], W[0], W[0], blocks, n, stride, batch, (bid / ((blocks * 2) / batch)), (bid % ((blocks * 2) / batch)), (((blocks * 2) / batch) - 2), window);
    } else {
      tl::ascend::pv_pipeline_rows(W[0], W[0], W[0], blocks, n, stride, batch, (bid / ((blocks * 2) / batch)), (bid % ((blocks * 2) / batch)), (((blocks * 2) / batch) - 2), window);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    valid = 1;
    for (int32_t b = 0; b < batch; ++b) {
      tl::ascend::pv_load_rows(pool[0], W[(((b * 8192) + (((batch * n) * stride) * 2)) + 2048)], pool[0], 1, (((blocks * 2) / batch) * 8), 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      inverse_total = 0.000000e+00f;
      for (int32_t c = 0; c < ((blocks * 2) / batch); ++c) {
        if (pool.GetValue((((int64_t)c) * (int64_t)8)) < 5.000000e-01f) {
          valid = 0;
        }
        inverse_total = (inverse_total + pool.GetValue(((((int64_t)c) * (int64_t)8) + (int64_t)1)));
      }
      tl::ascend::pv_load_rows(pool[384], W[((b * 8192) + (((batch * n) * stride) * 2))], pool[384], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      float power = pool.GetValue(416);
      float total = pool.GetValue(417);
      pool.SetValue(640, total);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      tl::ascend::pv_sqrt(pool[640], pool[640], pool[640], 1);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      float cutoff = max((pool.GetValue(640) * rtol), (atol / power));
      if (((inverse_total <= 0.000000e+00f) || (8.100000e+01f <= ((inverse_total * cutoff) * cutoff))) || (3.600000e+09f <= (total * inverse_total))) {
        valid = 0;
      }
    }
    if (bid == 0) {
      S.SetValue(0, ((float)valid));
      S.SetValue(31, ((float)valid));
    }
    if (valid == 0) {
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      if (bid < batch) {
        tl::ascend::pv_module_bind(A[0], X[0], W[0]);
        tl::ascend::pv_module_rectangular(S[0], S[0], S[0], blocks, bid, n, n, batch, atol, rtol);
      }
    }
    tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
  }
  pipe.Destroy();
}

#undef pv_pipeline_rows
#define pv_pipeline_rows(d,a,b,...) PvDslPipelineRows((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
#undef pv_pipeline_peer
#define pv_pipeline_peer(d,a,b,...) PvDslPipelinePeer((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
