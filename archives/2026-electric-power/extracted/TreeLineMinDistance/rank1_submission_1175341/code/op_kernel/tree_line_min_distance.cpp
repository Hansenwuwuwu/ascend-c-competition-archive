// Inlined local header: include/elec/template_device.h
#ifndef CANNJUDGE_INLINE_1A6F6DCE3D66EF704D217609
#define CANNJUDGE_INLINE_1A6F6DCE3D66EF704D217609

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

// Inlined local header: include/elec/template_tiling.h
#ifndef CANNJUDGE_INLINE_728A063E3555C4F12FC632AC
#define CANNJUDGE_INLINE_728A063E3555C4F12FC632AC

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


namespace elec {
// Plain fixed-width ABI copied by GET_TILING_DATA_WITH_STRUCT on the device.
struct TemplateTilingData {
    Config config;
    uint64_t scratch_bytes = 0;
    uint32_t input_types[6] = {};
    uint32_t output_types[2] = {};
};
}

#endif // CANNJUDGE_INLINE_728A063E3555C4F12FC632AC

namespace elec {
ELEC_FN void bind_template_tensors(const TemplateTilingData& t, GM_ADDR* inputs, GM_ADDR* outputs,
                                    Tensor* in, Tensor* out) {
    for(uint32_t i=0;i<6;++i) in[i]=Tensor(inputs[i],t.input_types[i]);
    for(uint32_t i=0;i<2;++i) out[i]=Tensor(outputs[i],t.output_types[i]);
}
ELEC_FN void check_template_status(int status) {
    // Release builds compile ASCENDC_ASSERT away; Trap must propagate failure to ACL.
    if (status != SUCCESS) AscendC::Trap();
}
}

#endif // CANNJUDGE_INLINE_1A6F6DCE3D66EF704D217609

#include "tree_pairs.h"
#include "tree_edges.h"
#include "tree_tiny.h"
#include "tree_short.h"
#include "tree_short_plan.h"
// Inlined local header: compute.h
#ifndef CANNJUDGE_INLINE_665407782BBB5DC6083065B0
#define CANNJUDGE_INLINE_665407782BBB5DC6083065B0

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
ELEC_FN int compute_tree_range(const Config& c, Tensor* in, Tensor* out,
                               uint64_t begin, uint64_t end, Math& math) {
    if (!in[0].valid() || !in[1].valid() || !out[0].valid() || !out[1].valid() ||
        c.batch < 1 || c.batch > 8 || c.m < 1 || c.m > 16384 || c.n < 1 || c.n > 1024) return INVALID;
    // Streaming min/argmin: O(B*N*M) arithmetic, O(1) extra storage.
    for (uint64_t point = begin; point < end; ++point) {
            const uint64_t b = point / c.m;
            const uint64_t i = point % c.m;
            const uint64_t p = (b*c.m+i)*3;
            const float px = in[0].get(p), py = in[0].get(p+1), pz = in[0].get(p+2);
            float best = bits_float(0x7f800000u);
            int32_t best_index = 0;
            for (uint64_t j = 0; j < c.n; ++j) {
                const uint64_t s = (b*c.n+j)*6;
                const float ax = in[1].get(s), ay = in[1].get(s+1), az = in[1].get(s+2);
                const float ux = in[1].get(s+3)-ax, uy = in[1].get(s+4)-ay, uz = in[1].get(s+5)-az;
                const float dx = px-ax, dy = py-ay, dz = pz-az;
                const float length2 = (ux*ux+uy*uy)+uz*uz;
                float t = 0.0f;
                if (length2 > 0.0f) {
                    t = ((dx*ux+dy*uy)+dz*uz)/length2;
                    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
                }
                const float ex = px-(ax+t*ux), ey = py-(ay+t*uy), ez = pz-(az+t*uz);
                const float d2 = (ex*ex+ey*ey)+ez*ez;
                // Ascending traversal and strict < preserve smallest-index ties.
                if (d2 < best) { best = d2; best_index = int32_t(j); }
            }
            out[0].set(b*c.m+i, math.sqrt(best));
            out[1].set_i32(b*c.m+i, best_index);
    }
    return SUCCESS;
}
ELEC_FN int compute_tree(const Config& c, Tensor* in, Tensor* out, Tensor, Math& math) {
    return compute_tree_range(c, in, out, 0, c.batch*c.m, math);
}
}

#endif // CANNJUDGE_INLINE_665407782BBB5DC6083065B0

#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

template <typename DT_TREE_POINTS,uint32_t PATH>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments, GM_ADDR min_distance, GM_ADDR nearest_segment_idx, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    if constexpr(PATH==14) {
        if(AscendC::GetBlockIdx()!=0)return;
        elec::Config c{1,1,1024,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::tree_one_point<DT_TREE_POINTS>(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==13) {
        if(AscendC::GetBlockIdx()>=3)return;
        elec::Config c{1,19,7,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::TreeShort<DT_TREE_POINTS,8,8> implementation;
        implementation.run(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==16) {
        if(AscendC::GetBlockIdx()>=9)return;
        elec::Config c{1,33,31,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::TreeShort<DT_TREE_POINTS,32,4> implementation;
        implementation.run(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==17) {
        if(AscendC::GetBlockIdx()>=8)return;
        elec::Config c{8,3,19,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::TreeShort<DT_TREE_POINTS,32,4> implementation;
        implementation.run(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==18) {
        if(AscendC::GetBlockIdx()>=16)return;
        elec::Config c{4,127,129,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::TreePairs<DT_TREE_POINTS,32> implementation;
        implementation.run(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==19) {
        if(AscendC::GetBlockIdx()>=8)return;
        elec::Config c{2,256,64,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::TreePairs<DT_TREE_POINTS,64> implementation;
        implementation.run(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==20) {
        if(AscendC::GetBlockIdx()>=8)return;
        elec::Config c{1,16384,1,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::tree_one_segment<DT_TREE_POINTS>(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==21) {
        elec::Config c{2,1003,513,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::TreePairs<DT_TREE_POINTS,64> implementation;
        implementation.run(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==22) {
        elec::Config c{2,4097,513,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::TreePairs<DT_TREE_POINTS,64> implementation;
        implementation.run(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==15) {
        elec::Config c{sizeof(DT_TREE_POINTS)==4?8u:4u,sizeof(DT_TREE_POINTS)==4?16384u:8192u,1024u,sizeof(DT_TREE_POINTS)==4?elec::F32:elec::F16};
        elec::TreePairs<DT_TREE_POINTS,64> implementation;
        implementation.run_1024(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH>=8&&PATH<=12) {
        elec::tree_tiny_fixed<DT_TREE_POINTS,PATH>=11?3:1,(PATH==8||PATH==11)?1:(PATH==10?6:2)>(
            tree_points,line_segments,min_distance,nearest_segment_idx);
        return;
    }
    if constexpr(PATH==7||PATH==0) {
        if(AscendC::GetBlockIdx()!=0)return;
    }
    elec::Config c;
    const auto* data=reinterpret_cast<__gm__ TreeLineMinDistanceTilingData*>(tiling);
    c.batch=data->config.batch;c.m=data->config.m;c.n=data->config.n;
    if constexpr(PATH==0) c.dtype=data->config.dtype;
    if constexpr(PATH==23) {
        elec::TreePairs<DT_TREE_POINTS,64> implementation;
        implementation.run_1024(c,tree_points,line_segments,min_distance,nearest_segment_idx);
    } else if constexpr(PATH==5||PATH==6) {
        constexpr uint32_t width=PATH==5?8:32;
        constexpr uint32_t tile=elec::tree_short_point_tile(sizeof(DT_TREE_POINTS),width);
        elec::TreeShort<DT_TREE_POINTS,width,tile> implementation;
        implementation.run(c,tree_points,line_segments,min_distance,nearest_segment_idx);
    } else if constexpr(PATH==4) {
        elec::tree_one_point<DT_TREE_POINTS>(c,tree_points,line_segments,min_distance,nearest_segment_idx);
    } else if constexpr(PATH==7) {
        if(c.n==1) elec::tree_tiny<DT_TREE_POINTS,1>(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        else if(c.n==2) elec::tree_tiny<DT_TREE_POINTS,2>(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        else if(c.n==6) elec::tree_tiny<DT_TREE_POINTS,6>(c,tree_points,line_segments,min_distance,nearest_segment_idx);
        else elec::tree_tiny<DT_TREE_POINTS>(c,tree_points,line_segments,min_distance,nearest_segment_idx);
    } else if constexpr(PATH>=2) {
        elec::TreePairs<DT_TREE_POINTS,PATH==3?64:32> implementation;
        implementation.run(c,tree_points,line_segments,min_distance,nearest_segment_idx);
    } else if constexpr(PATH==0) {
        if(AscendC::GetBlockIdx()!=0)return;
        elec::Tensor inputs[2]={elec::Tensor(tree_points,c.dtype),elec::Tensor(line_segments,c.dtype)};
        elec::Tensor outputs[2]={elec::Tensor(min_distance,c.dtype),elec::Tensor(nearest_segment_idx,elec::I32)};
        elec::Math math;
        elec::check_template_status(elec::compute_tree_range(c,inputs,outputs,0,c.batch*c.m,math));
        outputs[0].flush();outputs[1].flush();
    } else {
        elec::tree_one_segment<DT_TREE_POINTS>(c,tree_points,line_segments,min_distance,nearest_segment_idx);
    }
}
