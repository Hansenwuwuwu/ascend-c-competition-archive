#pragma once
#include "pinv_dsl_common.h"
#include "pinv_dsl_peer_rows.h"
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void pp_mailbox_cache(D d,A,B){AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(d);}
}

namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void pp_mailbox_load(D d,A,B,float &value){value=*reinterpret_cast<const volatile __gm__ float*>(d.GetPhyAddr());}
}

namespace tl::ascend {




__aicore__ static __attribute__((noinline)) void PvDslPeerTileProducer( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, int32_t n, int32_t stride, int32_t all_batch, int32_t job, int32_t core, int32_t cores, int32_t window, uint64_t fftsAddr) {
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
  int role = 0;
  int k = 0;
  int wide = 0;
  int half = 0;
  int own_width = 0;
  int pitch = 0;
  int capacity = 0;
  int actual_width = 0;
  int prefix = 0;
  int source_column = 0;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  float local_scale = 0.000000e+00f;
  float local_total = 0.000000e+00f;
  float local_power = 0.000000e+00f;
  float magnitude = 0.000000e+00f;
  int valid = 0;
  float power = 0.000000e+00f;
  float total = 0.000000e+00f;
  int size = 0;
  int owner = 0;
  int matrix = 0;
  int active_width = 0;
  int polls = 0;
  float observed = 0.000000e+00f;
  int drop = 0;
  int old_prefix = 0;
  int new_width = 0;
  int next_owner = 0;
  int new_prefix = 0;
  int next_pitch = 0;
  int polls_1 = 0;
  float observed_1 = 0.000000e+00f;
  float best = 0.000000e+00f;
  int chunk = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    role = (core - 9);
    k = 200;
    wide = (((k + 63) / 64) * 64);
    half = 48;
    own_width = 64;
    if ((role == 1) || (role == 2)) {
      own_width = 48;
    }
    if (role == 3) {
      own_width = 40;
    }
    pitch = own_width;
    if (role != 0) {
      pitch = (pitch + 16);
    }
    if (pitch < 16) {
      pitch = 16;
    }
    pitch = 72;
    capacity = (pitch * 200);
    int32_t scratch = capacity;
    int32_t vector = (((k * 8) + scratch) + 16320);
    int32_t product = (vector + wide);
    int32_t temporary = (product + wide);
    int32_t permutation = ((temporary + k) + 64);
    int32_t records = (permutation + k);
    W.SetValue(((((((int64_t)all_batch) * (int64_t)182400) + (((int64_t)job) * (int64_t)8192)) + (((int64_t)role) * (int64_t)32)) + (int64_t)6656), 0.000000e+00f);
    tl::ascend::pp_mailbox_cache(W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6656)]);
    W.SetValue(((((((int64_t)all_batch) * (int64_t)182400) + (((int64_t)job) * (int64_t)8192)) + (((int64_t)role) * (int64_t)32)) + (int64_t)6848), 0.000000e+00f);
    tl::ascend::pp_mailbox_cache(W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6848)]);
    actual_width = own_width;
    prefix = 0;
    if (role != 0) {
      prefix = 16;
    }
    source_column = 0;
    if (role == 1) {
      source_column = 64;
    }
    if (role == 2) {
      source_column = 112;
    }
    if (role == 3) {
      source_column = 160;
    }
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, capacity);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[prefix], A[((job * 40000) + source_column)], pool[prefix], 200, actual_width, (200 - actual_width), (pitch - own_width));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    local_scale = 0.000000e+00f;
    local_total = 0.000000e+00f;
    local_power = 1.000000e+00f;
    for (int32_t chunk_1 = 0; chunk_1 < ((capacity + 16319) / 16320); ++chunk_1) {
      int32_t count = min((capacity - (chunk_1 * 16320)), 16320);
      tl::ascend::pv_abs(pool[scratch], pool[(chunk_1 * 16320)], pool[scratch], count);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[(temporary + 32)], pool[scratch], pool[product], count);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      local_scale = max(local_scale, pool.GetValue((((int64_t)temporary) + (int64_t)32)));
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
    float local_ratio = (local_scale / local_power);
    local_total = (((((float)(own_width * 200)) * local_ratio) * local_ratio) * 1.000010e+00f);
    pool.SetValue(((int64_t)scratch), local_scale);
    pool.SetValue((((int64_t)scratch) + (int64_t)1), local_total);
    pool.SetValue((((int64_t)scratch) + (int64_t)2), local_power);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((all_batch * 182400) + (job * 8192)) + (role * 8)) + 6400)], pool[scratch], W[((((all_batch * 182400) + (job * 8192)) + (role * 8)) + 6400)], 1, 8, 0, 0);
    tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    tl::ascend::pv_load_rows(pool[scratch], W[(((all_batch * 182400) + (job * 8192)) + 6400)], pool[scratch], 1, 32, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    magnitude = 0.000000e+00f;
    for (int32_t part = 0; part < 4; ++part) {
      magnitude = max(magnitude, pool.GetValue(((((int64_t)part) * (int64_t)8) + ((int64_t)scratch))));
    }
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
    total = 0.000000e+00f;
    for (int32_t part_1 = 0; part_1 < 4; ++part_1) {
      float ratio = (pool.GetValue((((((int64_t)part_1) * (int64_t)8) + ((int64_t)scratch)) + (int64_t)2)) / power);
      total = (total + ((pool.GetValue((((((int64_t)part_1) * (int64_t)8) + ((int64_t)scratch)) + (int64_t)1)) * ratio) * ratio));
    }
    total = (total * 1.000010e+00f);
    for (int32_t chunk_2 = 0; chunk_2 < ((capacity + 16319) / 16320); ++chunk_2) {
      tl::ascend::pv_scale(pool[(chunk_2 * 16320)], pool[(chunk_2 * 16320)], pool[(chunk_2 * 16320)], (1.000000e+00f / power), min((capacity - (chunk_2 * 16320)), 16320));
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t row = 0; row < k; ++row) {
      pool.SetValue((((int64_t)permutation) + ((int64_t)row)), ((float)row));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    size = 200;
    for (int32_t panel = 0; panel < 13; ++panel) {
      if (valid != 0) {
        owner = 0;
        owner = 0;
        if (4 <= panel) {
          owner = 1;
        }
        if (7 <= panel) {
          owner = 2;
        }
        if (10 <= panel) {
          owner = 3;
        }
        matrix = ((panel * pitch) * 16);
        if (role == owner) {
          matrix = ((((panel * 16) + matrix) + prefix) - source_column);
        }
        active_width = own_width;
        if (role != owner) {
          active_width = (active_width + 16);
        }
        if (active_width < 16) {
          active_width = 16;
        }
        if (0 < panel) {
          for (int32_t part_2 = 0; part_2 < 4; ++part_2) {
            tl::ascend::pp_mailbox_cache(W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)]);
            polls_1 = 0;
            tl::ascend::pp_mailbox_load(W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], observed_1);
            while (1) {
              if (!(((observed_1 < ((float)panel)) && (polls_1 < 10000)))) { break; }
              tl::ascend::pp_mailbox_cache(W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)]);
              tl::ascend::pp_mailbox_load(W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (part_2 * 32)) + 6848)], observed_1);
              polls_1 = (polls_1 + 1);
            }
            if (10000 <= polls_1) {
              valid = 0;
            }
          }
        }
        if (role == owner) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(W[(((all_batch * 182400) + (job * 8192)) + 3072)], pool[matrix], W[(((all_batch * 182400) + (job * 8192)) + 3072)], size, 16, (pitch - 16), 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
          W.SetValue(((((((int64_t)all_batch) * (int64_t)182400) + (((int64_t)job) * (int64_t)8192)) + (((int64_t)role) * (int64_t)32)) + (int64_t)6656), ((float)(panel + 1)));
          tl::ascend::pp_mailbox_cache(W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6656)]);
        }
        if (role != owner) {
          tl::ascend::pp_mailbox_cache(W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)]);
          polls = 0;
          tl::ascend::pp_mailbox_load(W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], observed);
          while (1) {
            if (!(((observed < ((float)(panel + 1))) && (polls < 10000)))) { break; }
            tl::ascend::pp_mailbox_cache(W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)]);
            tl::ascend::pp_mailbox_load(W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], W[((((all_batch * 182400) + (job * 8192)) + (owner * 32)) + 6656)], observed);
            polls = (polls + 1);
          }
          if (10000 <= polls) {
            valid = 0;
          }
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
          tl::ascend::pv_load_rows(pool[matrix], W[(((all_batch * 182400) + (job * 8192)) + 3072)], pool[matrix], size, 16, 0, (pitch - 16));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        } else {
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
        W.SetValue(((((((int64_t)all_batch) * (int64_t)182400) + (((int64_t)job) * (int64_t)8192)) + (((int64_t)role) * (int64_t)32)) + (int64_t)6848), ((float)(panel + 1)));
        tl::ascend::pp_mailbox_cache(W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6848)], W[((((all_batch * 182400) + (job * 8192)) + (role * 32)) + 6848)]);
        tl::ascend::pv_iota(pool[(temporary + 64)], pool[(temporary + 64)], pool[(temporary + 64)], size);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_scale_indices(pool[(temporary + 64)], pool[(temporary + 64)], pool[(temporary + 64)], (pitch * 4), size);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t block = min(size, 16);
        for (int32_t pivot = 0; pivot < block; ++pivot) {
          int32_t remaining = (size - pivot);
          tl::ascend::pv_gather_offsets(pool[vector], pool[matrix], pool[(temporary + 64)], ((pivot * (pitch + 1)) * 4), remaining);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_abs(pool[product], pool[vector], pool[product], min(remaining, 32));
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_whole_max(pool[(temporary + 32)], pool[product], pool[(temporary + 32)], min(remaining, 32), 1);
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best, chunk);
          int32_t selected = (pivot + chunk);
          if (best <= 1.000000e-06f) {
            valid = 0;
            break;
          }
          float divisor = pool.GetValue(((((int64_t)vector) + ((int64_t)selected)) - ((int64_t)pivot)));
          if (selected != pivot) {
            float old = pool.GetValue((((int64_t)permutation) + ((int64_t)pivot)));
            pool.SetValue((((int64_t)permutation) + ((int64_t)pivot)), pool.GetValue((((int64_t)permutation) + ((int64_t)selected))));
            pool.SetValue((((int64_t)permutation) + ((int64_t)selected)), old);
          }
          pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)), pool.GetValue((((int64_t)permutation) + ((int64_t)pivot))));
          pool.SetValue(((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)) + (int64_t)16), (1.000000e+00f / divisor));
          if (1 < remaining) {
            pool.SetValue(((((int64_t)vector) + ((int64_t)selected)) - ((int64_t)pivot)), 0.000000e+00f);
            tl::ascend::pv_copy(pool[scratch], pool[(matrix + (selected * pitch))], pool[scratch], active_width);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_scale(pool[scratch], pool[scratch], pool[scratch], (-1.000000e+00f / divisor), active_width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            tl::ascend::pv_brcb_rows(pool[(scratch + 16320)], pool[vector], pool[(scratch + 16320)], (((remaining + 7) / 8) * 8), 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t chunk_3 = 0; chunk_3 < (((active_width + 63) - ((pivot / 8) * 8)) / 64); ++chunk_3) {
              for (int32_t row_chunk = 0; row_chunk < ((remaining + 254) / 255); ++row_chunk) {
                tl::ascend::pv_mad_strided(pool[((((chunk_3 * 64) + ((pivot / 8) * 8)) + matrix) + (((row_chunk * 255) + pivot) * pitch))], pool[(((chunk_3 * 64) + ((pivot / 8) * 8)) + scratch)], pool[(((row_chunk * 2040) + scratch) + 16320)], min(((active_width - ((pivot / 8) * 8)) - (chunk_3 * 64)), 64), min((remaining - (row_chunk * 255)), 255), 1, 1, 0, (pitch / 8), 0, 1);
              }
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (selected != pivot) {
              tl::ascend::pv_copy(pool[(matrix + (selected * pitch))], pool[(matrix + (pivot * pitch))], pool[(matrix + (selected * pitch))], active_width);
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
        next_owner = 0;
        if (4 <= ((block / 16) + panel)) {
          next_owner = 1;
        }
        if (7 <= ((block / 16) + panel)) {
          next_owner = 2;
        }
        if (10 <= ((block / 16) + panel)) {
          next_owner = 3;
        }
        new_prefix = 16;
        if (role == next_owner) {
          new_prefix = 0;
        }
        next_pitch = max(16, (new_prefix + new_width));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        if (0 < remaining_1) {
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          tl::ascend::pv_copy(pool[scratch], pool[(permutation + block)], pool[scratch], (((remaining_1 + 7) / 8) * 8));
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_copy(pool[permutation], pool[scratch], pool[permutation], (((remaining_1 + 7) / 8) * 8));
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        size = remaining_1;
        own_width = new_width;
      }
      if (0 < panel) {
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
      }
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)32), power);
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)33), total);
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)34), ((float)valid));
      if (role == 3) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        tl::ascend::pv_store_rows(W[(((all_batch * 182400) + (job * 8192)) + (panel * 64))], pool[((panel * 64) + records)], W[(((all_batch * 182400) + (job * 8192)) + (panel * 64))], 1, 64, 0, 0);
      }
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
    }
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((all_batch * 182400) + (job * 8192)) + (core * 8)) + 2048)], pool[0], W[((((all_batch * 182400) + (job * 8192)) + (core * 8)) + 2048)], 1, 2, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}




}
__aicore__ inline void PvStagePeerPipeline( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, int32_t stride, int32_t window, float atol, float rtol, uint64_t fftsAddr) {
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

__aicore__ inline void PvStagePeerTile( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, int32_t stride, int32_t window, float atol, float rtol, uint64_t fftsAddr) {
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
  int job = 0;
  int core = 0;
  int worker_index = 0;
  int valid = 0;
  auto pool = ascend_ub.GetWithOffset<float>(1024, 0);
  float inverse_total = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_pipeline_bind(A[0], X[0], W[0]);
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    if ((bid < (batch * 8)) && ((bid % 2) == 0)) {
      job = (bid / 8);
      core = (((bid % 8) / 2) + 9);
    } else {
      worker_index = (bid - (batch * 4));
      if (bid < (batch * 8)) {
        worker_index = (bid / 2);
      }
      job = (worker_index / 9);
      core = (worker_index % 9);
    }
    if (job < batch) {
      if (9 <= core) {
        tl::ascend::PvDslPeerTileProducer(A_handle,X_handle,W_handle,blocks, n, stride, batch, job, core, 9, window,fftsAddr);
      } else {
        tl::ascend::PvDslPeerTileRows(A_handle,X_handle,W_handle,blocks, n, stride, batch, job, core, 9, window,fftsAddr);
      }
    } else {
      tl::ascend::pv_cross_set0(W[0], W[0], W[0], 4);
      tl::ascend::pv_cross_wait(W[0], W[0], W[0], 4);
      for (int32_t dummy_panel = 0; dummy_panel < 13; ++dummy_panel) {
        tl::ascend::pv_cross_set0(W[0], W[0], W[0], 0);
        tl::ascend::pv_cross_wait(W[0], W[0], W[0], 0);
        tl::ascend::pv_cross_set0(W[0], W[0], W[0], 1);
        tl::ascend::pv_cross_wait(W[0], W[0], W[0], 1);
      }
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    valid = 1;
    if (bid < batch) {
      tl::ascend::pv_load_rows(pool[0], W[(((bid * 8192) + (((batch * n) * stride) * 2)) + 2048)], pool[0], 1, 104, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      inverse_total = 0.000000e+00f;
      for (int32_t c = 0; c < 13; ++c) {
        if (pool.GetValue((c * 8)) < 5.000000e-01f) {
          valid = 0;
        }
        inverse_total = (inverse_total + pool.GetValue(((c * 8) + 1)));
      }
      tl::ascend::pv_load_rows(pool[384], W[((bid * 8192) + (((batch * n) * stride) * 2))], pool[384], 1, 64, 0, 0);
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
      pool.SetValue(0, ((float)valid));
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(W[(((bid * 8) + (((batch * n) * stride) * 2)) + 7040)], pool[0], W[(((bid * 8) + (((batch * n) * stride) * 2)) + 7040)], 1, 8, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    tl::ascend::pv_load_rows(pool[0], W[((((batch * n) * stride) * 2) + 7040)], pool[0], 1, (batch * 8), 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    valid = 1;
    for (int32_t b = 0; b < batch; ++b) {
      if (pool.GetValue((((int64_t)b) * (int64_t)8)) < 5.000000e-01f) {
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
