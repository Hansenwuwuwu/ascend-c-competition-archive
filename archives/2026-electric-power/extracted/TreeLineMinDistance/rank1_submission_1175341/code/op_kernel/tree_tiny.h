#pragma once
// Inlined local header: include/elec/portable.h
#ifndef CANNJUDGE_INLINE_2529B83BF31798449902BD03
#define CANNJUDGE_INLINE_2529B83BF31798449902BD03

// Inlined local header: include/elec/config.h
#ifndef CANNJUDGE_INLINE_690E6E24C5AB526862289F78
#define CANNJUDGE_INLINE_690E6E24C5AB526862289F78

#include <cstdint>

namespace elec {
enum DType : uint32_t { F32 = 0, F16 = 1, BF16 = 2, I32 = 3, I64 = 4 };
enum Op : uint32_t { SAFETY = 0, FIRE, TREE, SWIGLU, SPLINE, DET, SOLVE, LSTSQ, EIGH, PINV };
enum Flags : uint32_t { ROOT = 1, BIAS = 2, MEAN = 4, UPPER = 8, RTOL_SET = 16 };
// Fixed-width, naturally aligned ABI shared by Python, host and device.
// Shapes are flattened to a batch count; all arrays are contiguous ND.
struct Config {
    uint64_t batch = 1, m = 1, n = 1, k = 1, edges = 0, dim = 2;
    uint64_t rhs_batch_stride = 0;
    uint32_t dtype = F32, flags = 0, degree = 1;
    uint32_t kernel_size[3] = {2, 2, 2};
    uint32_t open_spline[3] = {1, 1, 1};
    uint32_t max_sweeps = 100;
    float waist_ratio = 1.0f, threshold = 0.0f;
    float atol = 0.0f, rtol = 0.0f, tolerance = 1.0e-7f;
};
enum Status : int32_t { SUCCESS = 0, SINGULAR = 1, NO_CONVERGENCE = 2, INVALID = 3, NUMERIC_ERROR = 4 };
}

#endif // CANNJUDGE_INLINE_690E6E24C5AB526862289F78

#ifdef ELEC_CPU
#include <cmath>
#define ELEC_FN inline
#else
#include "kernel_operator.h"
#define ELEC_FN __aicore__ inline
#endif

namespace elec {
ELEC_FN uint32_t float_bits(float x) { union { float f; uint32_t u; } v; v.f=x; return v.u; }
ELEC_FN float bits_float(uint32_t x) { union { float f; uint32_t u; } v; v.u=x; return v.f; }
ELEC_FN bool is_finite(float x) { return (float_bits(x) & 0x7f800000u) != 0x7f800000u; }
ELEC_FN float half_to_float(uint16_t h) {
    uint32_t s = uint32_t(h & 0x8000u) << 16;
    int32_t e = (h >> 10) & 31;
    uint32_t m = h & 1023;
    if (e == 0) {
        if (m == 0) return bits_float(s);
        e = 1;
        while (!(m & 1024)) { m <<= 1; --e; }
        m &= 1023;
    } else if (e == 31) return bits_float(s | 0x7f800000u | (m << 13));
    return bits_float(s | (uint32_t(e + 112) << 23) | (m << 13));
}
// IEEE round-to-nearest, ties-to-even, including subnormals and overflow.
ELEC_FN uint16_t float_to_half(float x) {
    uint32_t u = float_bits(x), sign = (u >> 16) & 0x8000u;
    uint32_t m = u & 0x7fffffu, e0 = (u >> 23) & 255;
    if (e0 == 255) return uint16_t(sign | 0x7c00u | (m ? 0x0200u : 0));
    int32_t e = int32_t(e0) - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return uint16_t(sign);
        m |= 0x800000u;
        uint32_t shift = uint32_t(14 - e), q = m >> shift;
        uint32_t remainder = m & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        q += remainder > halfway || (remainder == halfway && (q & 1));
        return uint16_t(sign | q);
    }
    m += 0xfffu + ((m >> 13) & 1);
    if (m & 0x800000u) { m = 0; ++e; }
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    return uint16_t(sign | (uint32_t(e) << 10) | (m >> 13));
}
ELEC_FN float bf16_to_float(uint16_t h) { return bits_float(uint32_t(h) << 16); }
ELEC_FN uint16_t float_to_bf16(float x) {
    uint32_t u = float_bits(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}

// Scalar GM accessor. Parallel kernels partition writes at cacheline boundaries.
class Tensor {
public:
    ELEC_FN Tensor() = default;
#ifdef ELEC_CPU
    ELEC_FN Tensor(void* p, uint32_t type = F32) : ptr_(p), type_(type) {}
#else
    ELEC_FN Tensor(GM_ADDR p, uint32_t type = F32) : ptr_(p), type_(type) {
        if (p) {
            f_.SetGlobalBuffer((__gm__ float*)p);
            h_.SetGlobalBuffer((__gm__ uint16_t*)p);
            i32_.SetGlobalBuffer((__gm__ int32_t*)p);
            i64_.SetGlobalBuffer((__gm__ int64_t*)p);
        }
    }
#endif
    ELEC_FN bool valid() const { return ptr_ != nullptr; }
    ELEC_FN uint32_t dtype() const { return type_; }
#ifndef ELEC_CPU
    // Only FP32 workspace uses this view; storage offsets are in elements.
    ELEC_FN AscendC::GlobalTensor<float> float_global() const { return f_[offset_]; }
#endif
    ELEC_FN Tensor slice(uint64_t offset) const {
        Tensor t = *this; t.offset_ += offset; return t;
    }
    ELEC_FN float get(uint64_t i) const {
        i += offset_;
#ifdef ELEC_CPU
        if (type_ == F32) return ((float*)ptr_)[i];
        uint16_t x = ((uint16_t*)ptr_)[i];
#else
        if (type_ == F32) return f_.GetValue(i);
        uint16_t x = h_.GetValue(i);
#endif
        return type_ == F16 ? half_to_float(x) : bf16_to_float(x);
    }
    ELEC_FN void set(uint64_t i, float x) {
        i += offset_;
#ifdef ELEC_CPU
        if (type_ == F32) ((float*)ptr_)[i] = x;
        else ((uint16_t*)ptr_)[i] = type_ == F16 ? float_to_half(x) : float_to_bf16(x);
#else
        if (type_ == F32) f_.SetValue(i, x);
        else h_.SetValue(i, type_ == F16 ? float_to_half(x) : float_to_bf16(x));
#endif
    }
    ELEC_FN int64_t get_i64(uint64_t i) const {
#ifdef ELEC_CPU
        return ((int64_t*)ptr_)[i + offset_];
#else
        return i64_.GetValue(i + offset_);
#endif
    }
    ELEC_FN int32_t get_i32(uint64_t i) const {
#ifdef ELEC_CPU
        return ((int32_t*)ptr_)[i + offset_];
#else
        return i32_.GetValue(i + offset_);
#endif
    }
    ELEC_FN void set_i32(uint64_t i, int32_t x) {
#ifdef ELEC_CPU
        ((int32_t*)ptr_)[i + offset_] = x;
#else
        i32_.SetValue(i + offset_, x);
#endif
    }
    ELEC_FN void flush() {
#ifndef ELEC_CPU
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                        AscendC::DcciDst::CACHELINE_OUT>(f_);
#endif
    }
private:
#ifdef ELEC_CPU
    void* ptr_ = nullptr;
#else
    GM_ADDR ptr_ = nullptr;
    AscendC::GlobalTensor<float> f_;
    AscendC::GlobalTensor<uint16_t> h_;
    AscendC::GlobalTensor<int32_t> i32_;
    AscendC::GlobalTensor<int64_t> i64_;
#endif
    uint32_t type_ = F32;
    uint64_t offset_ = 0;
};

// Transcendentals run on the device Vector unit, never on the host.
// CPU mode substitutes libm only for algorithm testing, NOT an NPU fallback.
class Math {
public:
    ELEC_FN Math() {
#ifndef ELEC_CPU
        pipe_.InitBuffer(buffer_, 32);
        data_ = buffer_.Get<float>();
        for (uint32_t i = 0; i < 8; ++i) data_.SetValue(i, 0.0f);
#endif
    }
    ELEC_FN float sqrt(float x) {
#ifdef ELEC_CPU
        return std::sqrt(x);
#else
        data_.SetValue(0, x);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::Sqrt(data_, data_, 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return data_.GetValue(0);
#endif
    }
    ELEC_FN float exp(float x) {
#ifdef ELEC_CPU
        return std::exp(x);
#else
        data_.SetValue(0, x);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::Exp(data_, data_, 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return data_.GetValue(0);
#endif
    }
private:
#ifndef ELEC_CPU
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    AscendC::LocalTensor<float> data_;
#endif
};
}

#endif // CANNJUDGE_INLINE_2529B83BF31798449902BD03


namespace elec {

template <typename U>
__aicore__ inline AscendC::LocalTensor<U> local_tensor_at(
    uint32_t byte_offset, uint32_t /*byte_size*/, uint32_t count) {
    return AscendC::LocalTensor<U>(AscendC::TPosition::VECCALC, byte_offset, count);
}

union Half4 {
    uint64_t u64;
    half h[4];
};

union Half2 {
    uint32_t u32;
    half h[2];
};

union Float2 {
    uint64_t u64;
    float f[2];
};

__aicore__ inline float point_seg_dist2_sc(
    float px, float py, float pz,
    float ax, float ay, float az,
    float ux, float uy, float uz,
    float length) {
    const float dx = px - ax;
    const float dy = py - ay;
    const float dz = pz - az;
    const float dot = dx * ux + dy * uy + dz * uz;
    if (dot <= 0.0f) {
        return dx * dx + dy * dy + dz * dz;
    }
    if (dot >= length) {
        const float ex = dx - ux;
        const float ey = dy - uy;
        const float ez = dz - uz;
        return ex * ex + ey * ey + ez * ez;
    }
    const float t = dot / length;
    const float ex = dx - t * ux;
    const float ey = dy - t * uy;
    const float ez = dz - t * uz;
    return ex * ex + ey * ey + ez * ez;
}

template <typename T, uint32_t segment_count = 0>
__aicore__ inline void tree_tiny(
    const Config& c,
    GM_ADDR points_addr,
    GM_ADDR segments_addr,
    GM_ADDR min_distance_addr,
    GM_ADDR nearest_segment_idx_addr) {
    const __gm__ T* pts = reinterpret_cast<const __gm__ T*>(points_addr);
    const __gm__ T* segs = reinterpret_cast<const __gm__ T*>(segments_addr);
    AscendC::GlobalTensor<T> output;
    AscendC::GlobalTensor<int32_t> output_index;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(min_distance_addr));
    output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nearest_segment_idx_addr));

    AscendC::LocalTensor<float> distance = local_tensor_at<float>(0, 128, 32);
    AscendC::LocalTensor<int32_t> index = local_tensor_at<int32_t>(128, 128, 32);

    const uint64_t pre_p = reinterpret_cast<const volatile __gm__ uint64_t*>(points_addr)[0];
    const uint64_t pre_s = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[0];
    (void)pre_p;
    (void)pre_s;

    const uint32_t batch = static_cast<uint32_t>(c.batch);
    const uint32_t m_per_batch = static_cast<uint32_t>(c.m);
    const uint32_t count = batch * m_per_batch;
    const uint32_t segments_per_batch = segment_count ? segment_count : static_cast<uint32_t>(c.n);
    uint32_t i = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t seg_base = b * segments_per_batch * 6;
        for (uint32_t m = 0; m < m_per_batch; ++m, ++i) {
            const uint32_t p = i * 3;
            const float px = static_cast<float>(pts[p]);
            const float py = static_cast<float>(pts[p + 1]);
            const float pz = static_cast<float>(pts[p + 2]);
            float best = bits_float(0x7f800000u);
            int32_t best_index = 0;
            #pragma clang loop unroll(disable)
            for (uint32_t j = 0; j < segments_per_batch; ++j) {
                const uint32_t s = seg_base + j * 6;
                const float ax = static_cast<float>(segs[s]);
                const float ay = static_cast<float>(segs[s + 1]);
                const float az = static_cast<float>(segs[s + 2]);
                const float ux = static_cast<float>(segs[s + 3]) - ax;
                const float uy = static_cast<float>(segs[s + 4]) - ay;
                const float uz = static_cast<float>(segs[s + 5]) - az;
                const float len = ux * ux + uy * uy + uz * uz;
                const float value = point_seg_dist2_sc(px, py, pz, ax, ay, az, ux, uy, uz, len);
                if (value < best) {
                    best = value;
                    best_index = static_cast<int32_t>(j);
                }
            }
            distance.SetValue(i, best);
            index.SetValue(i, best_index);
        }
    }

    AscendC::DataCopyExtParams store_distance{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyExtParams store_index{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    AscendC::Sqrt(distance, distance, 32);
    AscendC::LocalTensor<T> distance_half = local_tensor_at<T>(256, 128, 64);
    if constexpr (sizeof(T) == 2) {
        AscendC::Cast(distance_half, distance, AscendC::RoundMode::CAST_RINT, 32);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::DataCopyPad(output_index, index, store_index);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    if constexpr (sizeof(T) == 2) {
        AscendC::DataCopyPad(output, distance_half, store_distance);
    } else {
        AscendC::DataCopyPad(output, distance, store_distance);
    }
}

template <typename T, uint32_t P, uint32_t M>
__aicore__ inline void tree_tiny_fixed(
    GM_ADDR points_addr,
    GM_ADDR segments_addr,
    GM_ADDR min_distance_addr,
    GM_ADDR nearest_segment_idx_addr) {
    if (AscendC::GetBlockIdx() != 0) return;
    const uint64_t pre_p0 = reinterpret_cast<const volatile __gm__ uint64_t*>(points_addr)[0];
    const uint64_t pre_s0 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[0];
    (void)pre_p0;
    (void)pre_s0;
    if constexpr (P * 3 * sizeof(T) > 32) {
        const uint64_t pre_p1 = reinterpret_cast<const volatile __gm__ uint64_t*>(points_addr)[4];
        (void)pre_p1;
    }
    if constexpr (M * 6 * sizeof(T) > 32) {
        const uint64_t pre_s1 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[4];
        (void)pre_s1;
    }
    if constexpr (M * 6 * sizeof(T) > 64) {
        const uint64_t pre_s2 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[8];
        const uint64_t pre_s3 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[12];
        const uint64_t pre_s4 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[16];
        (void)pre_s2;
        (void)pre_s3;
        (void)pre_s4;
    }

    const __gm__ T* pts = reinterpret_cast<const __gm__ T*>(points_addr);
    const __gm__ T* segs = reinterpret_cast<const __gm__ T*>(segments_addr);
    AscendC::GlobalTensor<T> output;
    AscendC::GlobalTensor<int32_t> output_index;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(min_distance_addr));
    output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nearest_segment_idx_addr));

    AscendC::LocalTensor<float> distance = local_tensor_at<float>(0, 128, 32);
    AscendC::LocalTensor<int32_t> index = local_tensor_at<int32_t>(128, 128, 32);
    AscendC::LocalTensor<T> distance_half = local_tensor_at<T>(256, 128, 64);
    AscendC::DataCopyExtParams store_distance{1, static_cast<uint32_t>(P * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyExtParams store_index{1, static_cast<uint32_t>(P * sizeof(int32_t)), 0, 0, 0};

    int32_t best_idx0 = 0, best_idx1 = 0, best_idx2 = 0;

    if constexpr (P == 1 && M == 1) {
        float px, py, pz, ax0, ay0, az0, ux0, uy0, uz0;
        if constexpr (sizeof(T) == 4) {
            const __gm__ uint64_t* p64 = reinterpret_cast<const __gm__ uint64_t*>(points_addr);
            const __gm__ uint64_t* s64 = reinterpret_cast<const __gm__ uint64_t*>(segments_addr);
            Float2 s01{s64[0]}, s23{s64[1]}, s45{s64[2]}, p01{p64[0]};
            pz = pts[2];
            ax0 = s01.f[0]; ay0 = s01.f[1]; az0 = s23.f[0];
            ux0 = s23.f[1] - ax0; uy0 = s45.f[0] - ay0; uz0 = s45.f[1] - az0;
            px = p01.f[0]; py = p01.f[1];
        } else {
            ax0 = static_cast<float>(segs[0]); ay0 = static_cast<float>(segs[1]); az0 = static_cast<float>(segs[2]);
            ux0 = static_cast<float>(segs[3]) - ax0; uy0 = static_cast<float>(segs[4]) - ay0; uz0 = static_cast<float>(segs[5]) - az0;
            px = static_cast<float>(pts[0]); py = static_cast<float>(pts[1]); pz = static_cast<float>(pts[2]);
        }
        const float len0 = ux0 * ux0 + uy0 * uy0 + uz0 * uz0;
        distance.SetValue(0, point_seg_dist2_sc(px, py, pz, ax0, ay0, az0, ux0, uy0, uz0, len0));
    } else if constexpr (P == 1 && M == 2) {
        float px, py, pz;
        float ax0, ay0, az0, ux0, uy0, uz0;
        float ax1, ay1, az1, ux1, uy1, uz1;
        if constexpr (sizeof(T) == 2) {
            const __gm__ uint32_t* p32 = reinterpret_cast<const __gm__ uint32_t*>(points_addr);
            const __gm__ uint64_t* s64 = reinterpret_cast<const __gm__ uint64_t*>(segments_addr);
            Half2 p01{p32[0]};
            const half p2_h = pts[2];
            Half4 s0{s64[0]}, s1{s64[1]}, s2{s64[2]};
            px = static_cast<float>(p01.h[0]);
            py = static_cast<float>(p01.h[1]);
            pz = static_cast<float>(p2_h);
            ax0 = static_cast<float>(s0.h[0]);
            ay0 = static_cast<float>(s0.h[1]);
            az0 = static_cast<float>(s0.h[2]);
            ux0 = static_cast<float>(s0.h[3]) - ax0;
            uy0 = static_cast<float>(s1.h[0]) - ay0;
            uz0 = static_cast<float>(s1.h[1]) - az0;
            ax1 = static_cast<float>(s1.h[2]);
            ay1 = static_cast<float>(s1.h[3]);
            az1 = static_cast<float>(s2.h[0]);
            ux1 = static_cast<float>(s2.h[1]) - ax1;
            uy1 = static_cast<float>(s2.h[2]) - ay1;
            uz1 = static_cast<float>(s2.h[3]) - az1;
        } else {
            const __gm__ uint64_t* p64 = reinterpret_cast<const __gm__ uint64_t*>(points_addr);
            const __gm__ uint64_t* s64 = reinterpret_cast<const __gm__ uint64_t*>(segments_addr);
            Float2 s01{s64[0]}, s23{s64[1]}, s45{s64[2]}, s67{s64[3]}, s89{s64[4]}, s1011{s64[5]}, p01{p64[0]};
            pz = pts[2];
            ax0 = s01.f[0]; ay0 = s01.f[1]; az0 = s23.f[0];
            ux0 = s23.f[1] - ax0; uy0 = s45.f[0] - ay0; uz0 = s45.f[1] - az0;
            ax1 = s67.f[0]; ay1 = s67.f[1]; az1 = s89.f[0];
            ux1 = s89.f[1] - ax1; uy1 = s1011.f[0] - ay1; uz1 = s1011.f[1] - az1;
            px = p01.f[0]; py = p01.f[1];
        }
        const float len0 = ux0 * ux0 + uy0 * uy0 + uz0 * uz0;
        const float d0 = point_seg_dist2_sc(px, py, pz, ax0, ay0, az0, ux0, uy0, uz0, len0);
        const float len1 = ux1 * ux1 + uy1 * uy1 + uz1 * uz1;
        const float d1 = point_seg_dist2_sc(px, py, pz, ax1, ay1, az1, ux1, uy1, uz1, len1);
        distance.SetValue(0, d1 < d0 ? d1 : d0);
        best_idx0 = d1 < d0 ? 1 : 0;
    } else if constexpr (P == 1 && M == 6) {
        float px, py, pz;
        if constexpr (sizeof(T) == 4) {
            const __gm__ uint64_t* p64 = reinterpret_cast<const __gm__ uint64_t*>(points_addr);
            const __gm__ uint64_t* s64 = reinterpret_cast<const __gm__ uint64_t*>(segments_addr);
            Float2 p01{p64[0]};
            px = p01.f[0]; py = p01.f[1]; pz = pts[2];
            float best = bits_float(0x7f800000u);
            #pragma clang loop unroll(disable)
            for (uint32_t j = 0; j < 6; ++j) {
                const uint32_t s = j * 3;
                Float2 s01{s64[s]}, s23{s64[s + 1]}, s45{s64[s + 2]};
                const float ax = s01.f[0], ay = s01.f[1], az = s23.f[0];
                const float ux = s23.f[1] - ax, uy = s45.f[0] - ay, uz = s45.f[1] - az;
                const float len = ux * ux + uy * uy + uz * uz;
                const float v = point_seg_dist2_sc(px, py, pz, ax, ay, az, ux, uy, uz, len);
                if (v < best) {
                    best = v;
                    best_idx0 = static_cast<int32_t>(j);
                }
            }
            distance.SetValue(0, best);
        } else {
            px = static_cast<float>(pts[0]); py = static_cast<float>(pts[1]); pz = static_cast<float>(pts[2]);
            float best = bits_float(0x7f800000u);
            #pragma clang loop unroll(disable)
            for (uint32_t j = 0; j < 6; ++j) {
                const uint32_t s = j * 6;
                const float ax = static_cast<float>(segs[s]);
                const float ay = static_cast<float>(segs[s + 1]);
                const float az = static_cast<float>(segs[s + 2]);
                const float ux = static_cast<float>(segs[s + 3]) - ax;
                const float uy = static_cast<float>(segs[s + 4]) - ay;
                const float uz = static_cast<float>(segs[s + 5]) - az;
                const float len = ux * ux + uy * uy + uz * uz;
                const float v = point_seg_dist2_sc(px, py, pz, ax, ay, az, ux, uy, uz, len);
                if (v < best) {
                    best = v;
                    best_idx0 = static_cast<int32_t>(j);
                }
            }
            distance.SetValue(0, best);
        }
    } else if constexpr (P == 3 && M == 1) {
        float ax0, ay0, az0, ux0, uy0, uz0;
        if constexpr (sizeof(T) == 4) {
            const __gm__ uint64_t* s64 = reinterpret_cast<const __gm__ uint64_t*>(segments_addr);
            Float2 s01{s64[0]}, s23{s64[1]}, s45{s64[2]};
            ax0 = s01.f[0]; ay0 = s01.f[1]; az0 = s23.f[0];
            ux0 = s23.f[1] - ax0; uy0 = s45.f[0] - ay0; uz0 = s45.f[1] - az0;
        } else {
            ax0 = static_cast<float>(segs[0]); ay0 = static_cast<float>(segs[1]); az0 = static_cast<float>(segs[2]);
            ux0 = static_cast<float>(segs[3]) - ax0; uy0 = static_cast<float>(segs[4]) - ay0; uz0 = static_cast<float>(segs[5]) - az0;
        }
        const float len0 = ux0 * ux0 + uy0 * uy0 + uz0 * uz0;
        #pragma clang loop unroll(disable)
        for (uint32_t i = 0; i < 3; ++i) {
            const uint32_t p = i * 3;
            const float px = static_cast<float>(pts[p]);
            const float py = static_cast<float>(pts[p + 1]);
            const float pz = static_cast<float>(pts[p + 2]);
            distance.SetValue(i, point_seg_dist2_sc(px, py, pz, ax0, ay0, az0, ux0, uy0, uz0, len0));
        }
    } else if constexpr (P == 3 && M == 2) {
        float ax0, ay0, az0, ux0, uy0, uz0, ax1, ay1, az1, ux1, uy1, uz1;
        if constexpr (sizeof(T) == 4) {
            const __gm__ uint64_t* s64 = reinterpret_cast<const __gm__ uint64_t*>(segments_addr);
            Float2 s01{s64[0]}, s23{s64[1]}, s45{s64[2]}, s67{s64[3]}, s89{s64[4]}, s1011{s64[5]};
            ax0 = s01.f[0]; ay0 = s01.f[1]; az0 = s23.f[0];
            ux0 = s23.f[1] - ax0; uy0 = s45.f[0] - ay0; uz0 = s45.f[1] - az0;
            ax1 = s67.f[0]; ay1 = s67.f[1]; az1 = s89.f[0];
            ux1 = s89.f[1] - ax1; uy1 = s1011.f[0] - ay1; uz1 = s1011.f[1] - az1;
        } else {
            ax0 = static_cast<float>(segs[0]); ay0 = static_cast<float>(segs[1]); az0 = static_cast<float>(segs[2]);
            ux0 = static_cast<float>(segs[3]) - ax0; uy0 = static_cast<float>(segs[4]) - ay0; uz0 = static_cast<float>(segs[5]) - az0;
            ax1 = static_cast<float>(segs[6]); ay1 = static_cast<float>(segs[7]); az1 = static_cast<float>(segs[8]);
            ux1 = static_cast<float>(segs[9]) - ax1; uy1 = static_cast<float>(segs[10]) - ay1; uz1 = static_cast<float>(segs[11]) - az1;
        }
        const float len0 = ux0 * ux0 + uy0 * uy0 + uz0 * uz0;
        const float len1 = ux1 * ux1 + uy1 * uy1 + uz1 * uz1;
        #pragma clang loop unroll(disable)
        for (uint32_t i = 0; i < 3; ++i) {
            const uint32_t p = i * 3;
            const float px = static_cast<float>(pts[p]);
            const float py = static_cast<float>(pts[p + 1]);
            const float pz = static_cast<float>(pts[p + 2]);
            const float d0 = point_seg_dist2_sc(px, py, pz, ax0, ay0, az0, ux0, uy0, uz0, len0);
            const float d1 = point_seg_dist2_sc(px, py, pz, ax1, ay1, az1, ux1, uy1, uz1, len1);
            distance.SetValue(i, d1 < d0 ? d1 : d0);
            if (i == 0) best_idx0 = d1 < d0 ? 1 : 0;
            else if (i == 1) best_idx1 = d1 < d0 ? 1 : 0;
            else best_idx2 = d1 < d0 ? 1 : 0;
        }
    }

    // Dispatch PIPE_V Sqrt (+ Cast) immediately before writing index to UB!
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    AscendC::Sqrt(distance, distance, 8);
    if constexpr (sizeof(T) == 2) {
        AscendC::Cast(distance_half, distance, AscendC::RoundMode::CAST_RINT, 8);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);

    // While PIPE_V computes Sqrt (+ Cast), PIPE_S writes index to UB and dispatches S_MTE3!
    index.SetValue(0, best_idx0);
    if constexpr (P == 3) {
        index.SetValue(1, best_idx1);
        index.SetValue(2, best_idx2);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::DataCopyPad(output_index, index, store_index);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    if constexpr (sizeof(T) == 2) {
        AscendC::DataCopyPad(output, distance_half, store_distance);
    } else {
        AscendC::DataCopyPad(output, distance, store_distance);
    }
}

}  // namespace elec
