#pragma once
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

#include "lstsq_options.h"

namespace elec::lstsq_fast {

template <AscendC::HardEvent Event>
ELEC_FN void fence() {
    AscendC::SetFlag<Event>(0);
    AscendC::WaitFlag<Event>(0);
}

// Fast resident Row-Major Augmented Gauss-Jordan solver for square systems
// with 1 <= m == n <= 64 and 1 <= k <= 8.
template <typename T, uint32_t B_CONST = 0u, uint32_t N_CONST = 0u, uint32_t K_CONST = 0u, bool ZERO_R = false>
ELEC_FN int solve_square_small(
    const __gm__ Config* c_ptr, GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals) {
    const uint32_t block_idx = AscendC::GetBlockIdx();
    const uint64_t batches = (B_CONST != 0u) ? static_cast<uint64_t>(B_CONST) : c_ptr->batch;
    if constexpr (B_CONST == 0u) {
        if (block_idx >= batches) return SUCCESS;
    }

    const uint32_t n = (N_CONST != 0u) ? N_CONST : static_cast<uint32_t>(c_ptr->n);
    const uint32_t k = (K_CONST != 0u) ? K_CONST : static_cast<uint32_t>(c_ptr->k);
    const uint64_t rhs_batch_stride = (B_CONST != 0u)
        ? static_cast<uint64_t>(N_CONST * K_CONST)
        : c_ptr->rhs_batch_stride;
    const uint64_t flags = (B_CONST != 0u) ? (ZERO_R ? LSTSQ_SQUARE_RESIDUAL : 0ULL) : c_ptr->flags;
    const uint32_t pad_a = (n + 7u) & ~7u;
    const uint32_t rhs_col = (N_CONST == 15u && K_CONST == 1u) ? 15u : ((N_CONST == 50u && K_CONST == 2u) ? 50u : pad_a);
    const uint32_t row_valid_pad = (N_CONST == 15u || N_CONST == 50u) ? pad_a : (pad_a + 8u);
    // Always use an odd number of 32B blocks for UB row stride (24 for n<=16, 56 for n=50, 72 for n=64)
    // to eliminate UB bank conflicts in Gather and MulAddDst:
    const uint32_t pad = ((row_valid_pad >> 3) | 1u) << 3;
    const uint32_t pad_b = pad - pad_a;
    const uint32_t block_num = (B_CONST != 0u) ? B_CONST : AscendC::GetBlockNum();

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    // Single UB buffer allocation to minimize kernel entry overhead on tiny cases:
    // [0 .. 4608): M (64 * 72 floats)
    // [4608 .. 5632): coefficients (128 * 8 floats)
    // [5632 .. 6656): vec temporaries (8 * 128 floats)
    // [6656 .. 7168): x_out (512 floats)
    // [7168 .. 7744): offsets (576 int32s)
    // [7744 .. 9792): raw (64 * 64 T, only used for BF16)
    constexpr uint32_t total_floats = (sizeof(T) == 4) ? 7744u : 9792u;
    pipe.InitBuffer(work_buf, total_floats * sizeof(float));

    auto base = work_buf.Get<float>();
    auto M = base[0];
    auto coefficients = base[4672];
    auto vec = base[5760];
    auto col_vec0 = vec[0];
    auto abs_col = vec[96];
    auto reduced = vec[288];
    auto col_vec1 = vec[544];
    auto zero_buf = vec[640];
    auto x_out = base[6560];
    auto offsets = base[7168].template ReinterpretCast<int32_t>();
    auto col_offsets = offsets[0];
    auto rhs_pack_offsets = offsets[64];
    auto raw = base[7744].template ReinterpretCast<T>();

    AscendC::GlobalTensor<T> input_a, input_b, output_x, output_r;
    input_a.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A));
    input_b.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(B));
    output_x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(solution));
    const AscendC::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};

    // Issue MTE2 DMA for the first batch BEFORE PIPE_V offset table construction
    // so GM->UB transfer overlaps 100% with the 2-wave PIPE_V setup:
    if constexpr (sizeof(T) == 4) {
        const uint64_t b_offset0 = static_cast<uint64_t>(block_idx) * rhs_batch_stride;
        if constexpr (B_CONST != 0u) {
            if constexpr ((N_CONST & 7u) == 0u) {
                const AscendC::DataCopyParams a_copy{
                    static_cast<uint16_t>(N_CONST),
                    static_cast<uint16_t>(N_CONST >> 3),
                    0,
                    static_cast<uint16_t>(pad_b >> 3)};
                AscendC::DataCopy(M, input_a[static_cast<uint64_t>(block_idx) * N_CONST * N_CONST], a_copy);
            } else {
                const AscendC::DataCopyExtParams a_load{
                    static_cast<uint16_t>(N_CONST),
                    static_cast<uint32_t>(N_CONST * sizeof(float)),
                    0,
                    static_cast<uint32_t>(pad_b >> 3),
                    0};
                AscendC::DataCopyPad(M, input_a[static_cast<uint64_t>(block_idx) * N_CONST * N_CONST], a_load, no_pad);
            }
        } else if ((n & 7u) == 0u) {
            const AscendC::DataCopyParams a_copy{
                static_cast<uint16_t>(n),
                static_cast<uint16_t>(n >> 3),
                0,
                static_cast<uint16_t>(pad_b >> 3)};
            AscendC::DataCopy(M, input_a[static_cast<uint64_t>(block_idx) * n * n], a_copy);
        } else {
            const AscendC::DataCopyExtParams a_load{
                static_cast<uint16_t>(n),
                static_cast<uint32_t>(n * sizeof(float)),
                0,
                static_cast<uint32_t>(pad_b >> 3),
                0};
            AscendC::DataCopyPad(M, input_a[static_cast<uint64_t>(block_idx) * n * n], a_load, no_pad);
        }
        if constexpr (B_CONST != 0u) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        }
        if constexpr (B_CONST != 0u) {
            if constexpr (K_CONST == 1u) {
                if constexpr ((N_CONST & 7u) == 0u) {
                    AscendC::DataCopy(col_vec1, input_b[b_offset0], N_CONST);
                } else {
                    const AscendC::DataCopyExtParams b_load1{1, static_cast<uint32_t>(N_CONST * sizeof(float)), 0, 0, 0};
                    AscendC::DataCopyPad(col_vec1, input_b[b_offset0], b_load1, no_pad);
                }
            } else {
                if constexpr ((N_CONST & 7u) == 0u) {
                    AscendC::DataCopy(x_out, input_b[b_offset0], K_CONST * N_CONST);
                } else {
                    const AscendC::DataCopyExtParams b_load2{1, static_cast<uint32_t>(K_CONST * N_CONST * sizeof(float)), 0, 0, 0};
                    AscendC::DataCopyPad(x_out, input_b[b_offset0], b_load2, no_pad);
                }
            }
        } else if (k == 1u) {
            if ((n & 7u) == 0u) {
                AscendC::DataCopy(col_vec1, input_b[b_offset0], n);
            } else {
                const AscendC::DataCopyExtParams b_load1{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(col_vec1, input_b[b_offset0], b_load1, no_pad);
            }
        } else if (k == 2u) {
            if ((n & 7u) == 0u) {
                AscendC::DataCopy(x_out, input_b[b_offset0], 2u * n);
            } else {
                const AscendC::DataCopyExtParams b_load2{1, static_cast<uint32_t>(2u * n * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(x_out, input_b[b_offset0], b_load2, no_pad);
            }
        } else {
            const AscendC::DataCopyExtParams b_load{
                static_cast<uint16_t>(n),
                static_cast<uint32_t>(k * sizeof(float)),
                0,
                static_cast<uint32_t>((pad - 8u) >> 3),
                0};
            AscendC::DataCopyPad(M[pad_a], input_b[b_offset0], b_load, no_pad);
        }
        if constexpr (B_CONST != 0u) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        }
    }

    const bool need_zero_r = (B_CONST != 0u)
        ? (ZERO_R && block_idx == 0u)
        : ((flags & LSTSQ_SQUARE_RESIDUAL) && residuals != nullptr && block_idx == 0u);
    if (need_zero_r) {
        output_r.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(residuals));
        AscendC::Duplicate(zero_buf, 0.0f, (B_CONST != 0u) ? 8u : 64u);
    }

    AscendC::CreateVecIndex(col_offsets, int32_t(0), pad_a);
    AscendC::Duplicate(abs_col, -1.0f, (B_CONST != 0u) ? pad_a : 64u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(col_offsets, col_offsets, static_cast<int32_t>(pad * sizeof(float)), pad_a);
    if constexpr (B_CONST != 0u && ZERO_R) {
        if (block_idx == 0u) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();

    auto col_offsets_u32 = col_offsets.template ReinterpretCast<uint32_t>();
    auto rhs_pack_offsets_u32 = rhs_pack_offsets.template ReinterpretCast<uint32_t>();
    volatile __ubuf__ const uint32_t* col_vec_u32_ptr =
        reinterpret_cast<volatile __ubuf__ const uint32_t*>(col_vec0.GetPhyAddr());
    __ubuf__ int32_t* rhs_pack_offsets_ptr = reinterpret_cast<__ubuf__ int32_t*>(rhs_pack_offsets.GetPhyAddr());
    volatile __ubuf__ const uint64_t* reduced_u64_ptr =
        reinterpret_cast<volatile __ubuf__ const uint64_t*>(reduced.GetPhyAddr());

    const uint64_t n_mask = (n == 64u) ? ~0ULL : ((1ULL << n) - 1ULL);
    const AscendC::BinaryRepeatParams repeat{1, 1, 0, static_cast<uint8_t>(pad >> 3), 0, 1};
    const AscendC::BinaryRepeatParams repeat_rhs8{
        static_cast<uint8_t>(pad >> 3), 0, 1,
        static_cast<uint8_t>((pad >> 3) * 8u), 0, 8};

    if constexpr (B_CONST != 0u && sizeof(T) == 4) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Gather(col_vec0, M[0], col_offsets_u32, uint32_t(0), n);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        if constexpr (K_CONST == 2u) {
            AscendC::Brcb(coefficients, x_out, static_cast<uint8_t>((2u * n + 7u) >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (N_CONST == 50u) {
                uint64_t mask_lane2[2] = {0x0404040404040404ULL, 0ULL};
                uint64_t mask_lane3[2] = {0x0808080808080808ULL, 0ULL};
                AscendC::Muls(M[48], coefficients[0], -1.0f, mask_lane2, 7u,
                              {static_cast<uint16_t>(pad >> 3), 2, static_cast<uint8_t>((pad >> 3) * 8u), 16});
                AscendC::Muls(M[48], coefficients[8], -1.0f, mask_lane3, 7u,
                              {static_cast<uint16_t>(pad >> 3), 2, static_cast<uint8_t>((pad >> 3) * 8u), 16});
            } else {
                AscendC::Muls(M[pad_a], coefficients[0], -1.0f, 64u, 8u,
                              {static_cast<uint16_t>(pad >> 3), 2, static_cast<uint8_t>((pad >> 3) * 8u), 16});
                uint64_t mask_lane1[2] = {0xfefefefefefefefeULL, 0ULL};
                AscendC::Muls(M[pad_a], coefficients[8], -1.0f, mask_lane1, 8u,
                              {static_cast<uint16_t>(pad >> 3), 2, static_cast<uint8_t>((pad >> 3) * 8u), 16});
            }
            if constexpr (ZERO_R) {
                if (block_idx == 0u) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    if constexpr (((B_CONST * K_CONST) & 7u) == 0u) {
                        AscendC::DataCopy(output_r, zero_buf.template ReinterpretCast<T>(), B_CONST * K_CONST);
                    } else {
                        const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(B_CONST * K_CONST * sizeof(T)), 0, 0, 0};
                        AscendC::DataCopyPad(output_r, zero_buf.template ReinterpretCast<T>(), r_copy);
                    }
                }
            }
        } else {
            AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t avail_mask = n_mask;
        for (uint32_t j = 0; j + 1u < N_CONST; ++j) {
            const uint32_t c0 = (j + 1u) & ~7u;
            const uint32_t rem = row_valid_pad - c0;
            uint64_t mask[2] = {avail_mask, 0ULL};
            AscendC::Abs(abs_col, col_vec0, mask, 1u, {1, 1, 8, 8});
            if constexpr (K_CONST == 1u) {
                if (j == 0u) {
                    AscendC::Brcb(coefficients, col_vec1, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                }
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WholeReduceMax<float>(
                reduced, abs_col, static_cast<int32_t>(n), 1, 1, 1, 8,
                AscendC::ReduceOrder::ORDER_VALUE_INDEX);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            if constexpr (K_CONST == 1u) {
                if (j == 0u) {
                    if constexpr (N_CONST == 15u) {
                        uint64_t mask_lane7[2] = {0x8080808080808080ULL, 0ULL};
                        AscendC::Muls(M[8], coefficients, -1.0f, mask_lane7, 2u,
                                      {static_cast<uint16_t>(pad >> 3), 1, static_cast<uint8_t>((pad >> 3) * 8u), 8});
                    } else {
                        AscendC::Muls(M[pad_a], coefficients, -1.0f, 64u, 2u,
                                      {static_cast<uint16_t>(pad >> 3), 1, static_cast<uint8_t>((pad >> 3) * 8u), 8});
                    }
                    AscendC::Duplicate(abs_col, -1.0f, pad_a);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Brcb(coefficients, col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    AscendC::Brcb(coefficients, col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate(abs_col, -1.0f, pad_a);
                }
                if constexpr (ZERO_R) {
                    if (j == 0u && block_idx == 0u) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                        if constexpr (((B_CONST * K_CONST) & 7u) == 0u) {
                            AscendC::DataCopy(output_r, zero_buf.template ReinterpretCast<T>(), B_CONST * K_CONST);
                        } else {
                            const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(B_CONST * K_CONST * sizeof(T)), 0, 0, 0};
                            AscendC::DataCopyPad(output_r, zero_buf.template ReinterpretCast<T>(), r_copy);
                        }
                    }
                }
            } else {
                AscendC::Brcb(coefficients, col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate(abs_col, -1.0f, pad_a);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

            const uint64_t red_val = reduced_u64_ptr[0];
            const uint32_t max_bits = static_cast<uint32_t>(red_val);
            const uint32_t pivot = static_cast<uint32_t>(red_val >> 32);
            const float neg_inv_abs = -1.0f / bits_float(max_bits);
            AscendC::Duplicate(coefficients[pivot * 8u], 0.0f, 8u);
            avail_mask &= ~(1ULL << pivot);
            const uint32_t sign_bit = col_vec_u32_ptr[pivot] & 0x80000000u;
            const int32_t row_byte_off = static_cast<int32_t>((pivot * pad + (rhs_col & 7u)) * sizeof(float));
            if constexpr (K_CONST == 1u) {
                rhs_pack_offsets_ptr[j] = row_byte_off;
            } else {
                rhs_pack_offsets_ptr[2u * j + 0u] = row_byte_off;
                rhs_pack_offsets_ptr[2u * j + 1u] = row_byte_off + static_cast<int32_t>(sizeof(float));
            }
            const float neg_inv_diag = bits_float(float_bits(neg_inv_abs) ^ sign_bit);

            AscendC::Muls(M[pivot * pad + c0], M[pivot * pad + c0], neg_inv_diag, rem);
            AscendC::PipeBarrier<PIPE_V>();

            if constexpr (N_CONST == 64u) {
                if (rem <= 64u) {
                    AscendC::MulAddDst(M[c0], M[pivot * pad + c0], coefficients,
                                       rem, static_cast<uint8_t>(n), repeat);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Gather(col_vec0, M[j + 1u], col_offsets_u32, uint32_t(0), n);
                } else {
                    AscendC::MulAddDst(M[0], M[pivot * pad], coefficients,
                                       64u, static_cast<uint8_t>(n), repeat);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Gather(col_vec0, M[j + 1u], col_offsets_u32, uint32_t(0), n);
                    AscendC::MulAddDst(M[64], M[pivot * pad + 64], coefficients,
                                       64u, 8u, repeat_rhs8);
                }
            } else {
                if constexpr (N_CONST <= 50u) {
                    if (rem == 8u) {
                        AscendC::MulAddDst(M[c0], M[pivot * pad + c0], coefficients,
                                           64u, static_cast<uint8_t>(pad_a >> 3), repeat_rhs8);
                    } else {
                        AscendC::MulAddDst(M[c0], M[pivot * pad + c0], coefficients,
                                           rem, static_cast<uint8_t>(n), repeat);
                    }
                } else {
                    AscendC::MulAddDst(M[c0], M[pivot * pad + c0], coefficients,
                                       rem, static_cast<uint8_t>(n), repeat);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(col_vec0, M[j + 1u], col_offsets_u32, uint32_t(0), n);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            constexpr uint32_t j = N_CONST - 1u;
            constexpr uint32_t c0 = rhs_col & ~7u;
            constexpr uint32_t rem = row_valid_pad - c0;
            const uint32_t pivot = static_cast<uint32_t>(AscendC::ScalarGetSFFValue<1>(avail_mask));
            AscendC::Brcb(coefficients, col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(coefficients[pivot * 8u], 0.0f, 8u);
            const int32_t row_byte_off = static_cast<int32_t>((pivot * pad + (rhs_col & 7u)) * sizeof(float));
            if constexpr (K_CONST == 1u) {
                rhs_pack_offsets_ptr[j] = row_byte_off;
            } else {
                rhs_pack_offsets_ptr[2u * j + 0u] = row_byte_off;
                rhs_pack_offsets_ptr[2u * j + 1u] = row_byte_off + static_cast<int32_t>(sizeof(float));
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            const float neg_inv_diag = -1.0f / bits_float(col_vec_u32_ptr[pivot]);
            AscendC::Muls(M[pivot * pad + c0], M[pivot * pad + c0], neg_inv_diag, rem);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(M[c0], M[pivot * pad + c0], coefficients,
                               64u, static_cast<uint8_t>(pad_a >> 3), repeat_rhs8);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Gather(x_out, M[rhs_col & ~7u], rhs_pack_offsets_u32, uint32_t(0), N_CONST * K_CONST);
        fence<AscendC::HardEvent::V_MTE3>();
        if constexpr (((N_CONST * K_CONST) & 7u) == 0u) {
            AscendC::DataCopy(output_x[static_cast<uint64_t>(block_idx) * N_CONST * K_CONST], x_out, N_CONST * K_CONST);
        } else {
            const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(N_CONST * K_CONST * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(output_x[static_cast<uint64_t>(block_idx) * N_CONST * K_CONST], x_out, x_store);
        }
        return SUCCESS;
    } else {

    for (uint64_t batch = block_idx; batch < batches; batch += block_num) {
        const uint64_t b_offset = batch * rhs_batch_stride;
        if constexpr (sizeof(T) == 4) {
            if (batch != block_idx) {
                if ((n & 7u) == 0u) {
                    const AscendC::DataCopyParams a_copy{
                        static_cast<uint16_t>(n),
                        static_cast<uint16_t>(n >> 3),
                        0,
                        static_cast<uint16_t>(pad_b >> 3)};
                    AscendC::DataCopy(M, input_a[batch * n * n], a_copy);
                } else {
                    const AscendC::DataCopyExtParams a_load{
                        static_cast<uint16_t>(n),
                        static_cast<uint32_t>(n * sizeof(float)),
                        0,
                        static_cast<uint32_t>(pad_b >> 3),
                        0};
                    AscendC::DataCopyPad(M, input_a[batch * n * n], a_load, no_pad);
                }
                if (k == 1u) {
                    if ((n & 7u) == 0u) {
                        AscendC::DataCopy(col_vec1, input_b[b_offset], n);
                    } else {
                        const AscendC::DataCopyExtParams b_load1{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
                        AscendC::DataCopyPad(col_vec1, input_b[b_offset], b_load1, no_pad);
                    }
                } else if (k == 2u) {
                    if ((n & 7u) == 0u) {
                        AscendC::DataCopy(x_out, input_b[b_offset], 2u * n);
                    } else {
                        const AscendC::DataCopyExtParams b_load2{1, static_cast<uint32_t>(2u * n * sizeof(float)), 0, 0, 0};
                        AscendC::DataCopyPad(x_out, input_b[b_offset], b_load2, no_pad);
                    }
                } else {
                    const AscendC::DataCopyExtParams b_load{
                        static_cast<uint16_t>(n),
                        static_cast<uint32_t>(k * sizeof(float)),
                        0,
                        static_cast<uint32_t>((pad - 8u) >> 3),
                        0};
                    AscendC::DataCopyPad(M[pad_a], input_b[b_offset], b_load, no_pad);
                }
            }
            if constexpr (B_CONST != 0u && N_CONST != 64u) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::Gather(col_vec0, M[0], col_offsets_u32, uint32_t(0), n);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
                if constexpr (K_CONST == 2u) {
                    AscendC::Brcb(coefficients, x_out, static_cast<uint8_t>((2u * n + 7u) >> 3), {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    if constexpr (N_CONST == 50u) {
                        uint64_t mask_lane2[2] = {0x04ULL, 0ULL};
                        uint64_t mask_lane3[2] = {0x08ULL, 0ULL};
                        AscendC::Muls(M[48], coefficients[0], -1.0f, mask_lane2, static_cast<uint8_t>(n),
                                      {1, 1, static_cast<uint8_t>(pad >> 3), 2});
                        AscendC::Muls(M[48], coefficients[8], -1.0f, mask_lane3, static_cast<uint8_t>(n),
                                      {1, 1, static_cast<uint8_t>(pad >> 3), 2});
                    } else {
                        AscendC::Muls(M[pad_a], coefficients[0], -1.0f, 8u, static_cast<uint8_t>(n),
                                      {1, 1, static_cast<uint8_t>(pad >> 3), 2});
                        uint64_t mask_lane1[2] = {0xfeULL, 0ULL};
                        AscendC::Muls(M[pad_a], coefficients[8], -1.0f, mask_lane1, static_cast<uint8_t>(n),
                                      {1, 1, static_cast<uint8_t>(pad >> 3), 2});
                    }
                } else {
                    AscendC::PipeBarrier<PIPE_V>();
                }
            } else {
                fence<AscendC::HardEvent::MTE2_V>();
                AscendC::Gather(col_vec0, M[0], col_offsets_u32, uint32_t(0), n);
                if (k == 1u) {
                    AscendC::Brcb(coefficients, col_vec1, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(M[pad_a], coefficients, -1.0f, 8u, static_cast<uint8_t>(n),
                                  {1, 1, static_cast<uint8_t>(pad >> 3), 1});
                } else if (k == 2u) {
                    AscendC::Brcb(coefficients, x_out, static_cast<uint8_t>((2u * n + 7u) >> 3), {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(M[pad_a], coefficients[0], -1.0f, 8u, static_cast<uint8_t>(n),
                                  {1, 1, static_cast<uint8_t>(pad >> 3), 2});
                    uint64_t mask_lane1[2] = {0xfeULL, 0ULL};
                    AscendC::Muls(M[pad_a], coefficients[8], -1.0f, mask_lane1, static_cast<uint8_t>(n),
                                  {1, 1, static_cast<uint8_t>(pad >> 3), 2});
                } else {
                    AscendC::Muls(M[pad_a], M[pad_a], -1.0f, 8u, static_cast<uint8_t>(n),
                                  {1, 1, static_cast<uint8_t>(pad >> 3), static_cast<uint8_t>(pad >> 3)});
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        } else {
            const uint32_t raw_pad = (n + 15u) & ~15u;
            const AscendC::DataCopyExtParams a_load{
                static_cast<uint16_t>(n),
                static_cast<uint32_t>(n * sizeof(T)),
                0, 0, 0};
            AscendC::DataCopyPad(raw, input_a[batch * n * n], a_load, no_pad);
            fence<AscendC::HardEvent::MTE2_V>();
            for (uint32_t r = 0; r < n; ++r) {
                AscendC::Cast(M[r * pad], raw[r * raw_pad], AscendC::RoundMode::CAST_NONE, pad_a);
            }
            fence<AscendC::HardEvent::V_MTE2>();
            const uint32_t b_elems = n * k;
            const AscendC::DataCopyExtParams b_load{1, static_cast<uint32_t>(b_elems * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(raw, input_b[b_offset], b_load, no_pad);
            fence<AscendC::HardEvent::MTE2_V>();
            AscendC::Cast(x_out, raw, AscendC::RoundMode::CAST_NONE, (b_elems + 7u) & ~7u);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(x_out, x_out, -1.0f, (b_elems + 7u) & ~7u);
            fence<AscendC::HardEvent::V_S>();
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t t = 0; t < k; ++t) {
                    M.SetValue(i * pad + pad_a + t, x_out.GetValue(i * k + t));
                }
            }
            fence<AscendC::HardEvent::S_V>();
            AscendC::Gather(col_vec0, M[0], col_offsets_u32, uint32_t(0), n);
            AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t avail_mask = n_mask;
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t pivot = 0u;
            float neg_inv_diag = 0.0f;
            const uint32_t c0 = (j + 1u < n) ? ((j + 1u) & ~7u) : (rhs_col & ~7u);
            const uint32_t rem = row_valid_pad - c0;
            if (B_CONST == 0u || j + 1u < n) {
                uint64_t mask[2] = {avail_mask, 0ULL};
                AscendC::Abs(abs_col, col_vec0, mask, 1u, {1, 1, 8, 8});
                if constexpr (B_CONST != 0u && K_CONST == 1u) {
                    if (j == 0u) {
                        AscendC::Brcb(coefficients, col_vec1, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::WholeReduceMax<float>(
                    reduced, abs_col, static_cast<int32_t>(n), 1, 1, 1, 8,
                    AscendC::ReduceOrder::ORDER_VALUE_INDEX);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                if constexpr (B_CONST != 0u && K_CONST == 1u) {
                    if (j == 0u) {
                        if constexpr (N_CONST == 15u) {
                            uint64_t mask_lane7[2] = {0x80ULL, 0ULL};
                            AscendC::Muls(M[8], coefficients, -1.0f, mask_lane7, static_cast<uint8_t>(n),
                                          {1, 1, static_cast<uint8_t>(pad >> 3), 1});
                        } else {
                            AscendC::Muls(M[pad_a], coefficients, -1.0f, 8u, static_cast<uint8_t>(n),
                                          {1, 1, static_cast<uint8_t>(pad >> 3), 1});
                        }
                        AscendC::Duplicate(abs_col, -1.0f, pad_a);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Brcb(coefficients, col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        AscendC::Brcb(coefficients, col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Duplicate(abs_col, -1.0f, pad_a);
                    }
                } else {
                    AscendC::Brcb(coefficients, col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate(abs_col, -1.0f, pad_a);
                }
                if constexpr (B_CONST != 0u && ZERO_R) {
                    if (j == 0u && block_idx == 0u) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                        if constexpr (((B_CONST * K_CONST) & 7u) == 0u) {
                            AscendC::DataCopy(output_r, zero_buf.template ReinterpretCast<T>(), B_CONST * K_CONST);
                        } else {
                            const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(B_CONST * K_CONST * sizeof(T)), 0, 0, 0};
                            AscendC::DataCopyPad(output_r, zero_buf.template ReinterpretCast<T>(), r_copy);
                        }
                    }
                }
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

                const uint64_t red_val = reduced_u64_ptr[0];
                const uint32_t max_bits = static_cast<uint32_t>(red_val);
                pivot = static_cast<uint32_t>(red_val >> 32);
                if constexpr (B_CONST == 0u) {
                    if (max_bits == 0u) return SINGULAR;
                    if (max_bits >= 0x7f800000u || pivot >= n) return NUMERIC_ERROR;
                }
                const float neg_inv_abs = -1.0f / bits_float(max_bits);
                AscendC::Duplicate(coefficients[pivot * 8u], 0.0f, 8u);
                avail_mask &= ~(1ULL << pivot);
                const uint32_t sign_bit = col_vec_u32_ptr[pivot] & 0x80000000u;
                neg_inv_diag = bits_float(float_bits(neg_inv_abs) ^ sign_bit);
            } else {
                pivot = static_cast<uint32_t>(AscendC::ScalarGetSFFValue<1>(avail_mask));
                AscendC::Brcb(coefficients, col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate(coefficients[pivot * 8u], 0.0f, 8u);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                neg_inv_diag = -1.0f / bits_float(col_vec_u32_ptr[pivot]);
            }

            AscendC::Muls(M[pivot * pad + c0], M[pivot * pad + c0], neg_inv_diag, rem);
            AscendC::PipeBarrier<PIPE_V>();

            const int32_t row_byte_off = static_cast<int32_t>((pivot * pad + (rhs_col & 7u)) * sizeof(float));
            if (k == 1u) {
                rhs_pack_offsets_ptr[j] = row_byte_off;
            } else if (k == 2u) {
                rhs_pack_offsets_ptr[2u * j + 0u] = row_byte_off;
                rhs_pack_offsets_ptr[2u * j + 1u] = row_byte_off + static_cast<int32_t>(sizeof(float));
            } else {
                for (uint32_t t = 0; t < k; ++t) {
                    rhs_pack_offsets_ptr[j * k + t] = row_byte_off + static_cast<int32_t>(t * sizeof(float));
                }
            }

            if (rem <= 64u) {
                AscendC::MulAddDst(M[c0], M[pivot * pad + c0], coefficients,
                                   rem, static_cast<uint8_t>(n), repeat);
                AscendC::PipeBarrier<PIPE_V>();
                if (j + 1u < n) {
                    AscendC::Gather(col_vec0, M[j + 1u], col_offsets_u32, uint32_t(0), n);
                    if (B_CONST != 0u && j + 2u == n) {
                        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
            } else {
                AscendC::MulAddDst(M[c0], M[pivot * pad + c0], coefficients,
                                   64u, static_cast<uint8_t>(n), repeat);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(col_vec0, M[j + 1u], col_offsets_u32, uint32_t(0), n);
                AscendC::MulAddDst(M[c0 + 64u], M[pivot * pad + c0 + 64u], coefficients,
                                   rem - 64u, static_cast<uint8_t>(n), repeat);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        fence<AscendC::HardEvent::S_V>();
        AscendC::Gather(x_out, M[rhs_col & ~7u], rhs_pack_offsets_u32, uint32_t(0), n * k);
        if constexpr (sizeof(T) == 4) {
            fence<AscendC::HardEvent::V_MTE3>();
            if (((n * k) & 7u) == 0u) {
                AscendC::DataCopy(output_x[batch * n * k], x_out, n * k);
            } else {
                const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(n * k * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(output_x[batch * n * k], x_out, x_store);
            }
        } else {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(raw, x_out, AscendC::RoundMode::CAST_RINT, (n * k + 7u) & ~7u);
            fence<AscendC::HardEvent::V_MTE3>();
            const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(n * k * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(output_x[batch * n * k], raw, x_store);
        }
        if (batch + block_num < batches) {
            fence<AscendC::HardEvent::MTE3_V>();
        } else if constexpr (B_CONST == 0u) {
            if (need_zero_r) {
                const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(batches * k * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(output_r, zero_buf.template ReinterpretCast<T>(), r_copy);
            }
        }
    }
    return SUCCESS;
    }
}

template <typename T>
ELEC_FN void load_panel_a(
    uint32_t n, uint64_t batch, uint32_t col_start, uint32_t width, uint32_t width_pad, uint32_t m_stride,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::LocalTensor<float>& M,
    const AscendC::LocalTensor<float>& lu_stage) {
    const AscendC::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};
    if constexpr (sizeof(T) == 4) {
        if ((n & 7u) == 0u && (width & 7u) == 0u) {
            const AscendC::DataCopyParams a_copy{
                static_cast<uint16_t>(n),
                static_cast<uint16_t>(width >> 3),
                static_cast<uint16_t>((n - width) >> 3),
                static_cast<uint16_t>((m_stride - width) >> 3)};
            AscendC::DataCopy(M, input_a[batch * n * n + col_start], a_copy);
        } else {
            const AscendC::DataCopyExtParams a_load{
                static_cast<uint16_t>(n),
                static_cast<uint32_t>(width * sizeof(float)),
                static_cast<uint32_t>((n - width) * sizeof(float)),
                static_cast<uint32_t>((m_stride - width_pad) >> 3),
                0};
            AscendC::DataCopyPad(M, input_a[batch * n * n + col_start], a_load, no_pad);
        }
        fence<AscendC::HardEvent::MTE2_V>();
    } else {
        auto raw_panel = lu_stage.template ReinterpretCast<T>();
        const uint32_t raw_w_pad = (width + 15u) & ~15u;
        for (uint32_t r0 = 0; r0 < n; r0 += 8u) {
            const uint32_t rows = n - r0 < 8u ? n - r0 : 8u;
            const AscendC::DataCopyExtParams a_load{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(width * sizeof(T)),
                static_cast<uint32_t>((n - width) * sizeof(T)),
                0, 0};
            AscendC::DataCopyPad(raw_panel, input_a[(batch * n + r0) * n + col_start], a_load, no_pad);
            fence<AscendC::HardEvent::MTE2_V>();
            for (uint32_t r = 0; r < rows; ++r) {
                AscendC::Cast(M[(r0 + r) * m_stride], raw_panel[r * raw_w_pad], AscendC::RoundMode::CAST_NONE, width_pad);
            }
            fence<AscendC::HardEvent::V_MTE2>();
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T>
ELEC_FN void load_rhs_b(
    uint32_t n, uint32_t pad_n, uint32_t k, uint64_t b_offset,
    const AscendC::GlobalTensor<T>& input_b,
    const AscendC::LocalTensor<float>& y,
    const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<float>& tmp_cast) {
    const AscendC::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};
    if constexpr (sizeof(T) == 4) {
        AscendC::Duplicate(y, 0.0f, (pad_n + 8u) * 8u);
        fence<AscendC::HardEvent::V_MTE2>();
        const AscendC::DataCopyExtParams b_load{
            static_cast<uint16_t>(n),
            static_cast<uint32_t>(k * sizeof(float)),
            0, 0, 0};
        AscendC::DataCopyPad(y, input_b[b_offset], b_load, no_pad);
        fence<AscendC::HardEvent::MTE2_V>();
    } else {
        AscendC::Duplicate(y, 0.0f, (pad_n + 8u) * 8u);
        fence<AscendC::HardEvent::V_MTE2>();
        auto raw_b = lu_stage.template ReinterpretCast<T>();
        const AscendC::DataCopyExtParams b_load{1, static_cast<uint32_t>(n * k * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(raw_b, input_b[b_offset], b_load, no_pad);
        fence<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(tmp_cast, raw_b, AscendC::RoundMode::CAST_NONE, (n * k + 7u) & ~7u);
        fence<AscendC::HardEvent::V_S>();
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t t = 0; t < k; ++t) {
                y.SetValue(i * 8u + t, tmp_cast.GetValue(i * k + t));
            }
        }
        fence<AscendC::HardEvent::S_V>();
    }
}

// 1-Barrier unit-upper-triangular back-substitution panel:
// Since lu_buf[s * pad_n + 0..uj-1] stores -U[0..uj-1, uj] * inv_diag[uj],
// solving U * x = y factors as U_tilde * (D * x) = y, so no per-step y[uj] *= inv_diag[uj]
// is needed inside the loop; Brcb of step s-1 overlaps with MulAddDst of step s in 1 barrier!
ELEC_FN void back_sub_panel(
    uint32_t col_start, uint32_t width, uint32_t pad_n,
    const AscendC::LocalTensor<float>& lu_buf,
    const AscendC::LocalTensor<float>& y,
    const AscendC::LocalTensor<float>& buf0,
    const AscendC::LocalTensor<float>& buf1,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    const int32_t first_s = static_cast<int32_t>(width) - 1;
    const int32_t last_s = (col_start == 0u) ? 1 : 0;
    if (first_s < last_s) return;
    const uint32_t first_uj = col_start + static_cast<uint32_t>(first_s);
    const uint32_t first_j_pad = (first_uj + 7u) & ~7u;
    AscendC::Brcb(buf0, lu_buf[static_cast<uint32_t>(first_s) * pad_n],
                  static_cast<uint8_t>(first_j_pad >> 3), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t s = first_s; s >= last_s; --s) {
        const uint32_t uj = col_start + static_cast<uint32_t>(s);
        const bool use_buf1 = ((first_s - s) & 1) != 0;
        if (s > last_s) {
            const uint32_t next_uj = uj - 1u;
            const uint32_t next_j_pad = (next_uj + 7u) & ~7u;
            if (use_buf1) {
                AscendC::Brcb(buf0, lu_buf[static_cast<uint32_t>(s - 1) * pad_n],
                              static_cast<uint8_t>(next_j_pad >> 3), {1, 8});
            } else {
                AscendC::Brcb(buf1, lu_buf[static_cast<uint32_t>(s - 1) * pad_n],
                              static_cast<uint8_t>(next_j_pad >> 3), {1, 8});
            }
        }
        if (use_buf1) {
            if (uj >= 8u) {
                AscendC::MulAddDst(y[0], y[uj * 8u], buf1[0],
                                   64u, static_cast<uint8_t>(uj >> 3), y_repeat8);
            }
            if ((uj & 7u) != 0u) {
                const uint32_t tail_row = uj & ~7u;
                AscendC::MulAddDst(y[tail_row * 8u], y[uj * 8u], buf1[tail_row * 8u],
                                   static_cast<uint64_t>((uj & 7u) * 8u), 1u, y_repeat8);
            }
        } else {
            if (uj >= 8u) {
                AscendC::MulAddDst(y[0], y[uj * 8u], buf0[0],
                                   64u, static_cast<uint8_t>(uj >> 3), y_repeat8);
            }
            if ((uj & 7u) != 0u) {
                const uint32_t tail_row = uj & ~7u;
                AscendC::MulAddDst(y[tail_row * 8u], y[uj * 8u], buf0[tail_row * 8u],
                                   static_cast<uint64_t>((uj & 7u) * 8u), 1u, y_repeat8);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

ELEC_FN void back_sub_panel_p8(
    uint32_t col_start, uint32_t width, uint32_t pad_n,
    const AscendC::LocalTensor<float>& lu_buf,
    const AscendC::LocalTensor<float>& y_pad,
    const AscendC::LocalTensor<float>& y,
    const AscendC::LocalTensor<float>& buf0_pad,
    const AscendC::LocalTensor<float>& buf0,
    const AscendC::LocalTensor<float>& buf1_pad,
    const AscendC::LocalTensor<float>& buf1,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    const int32_t first_s = static_cast<int32_t>(width) - 1;
    const int32_t last_s = (col_start == 0u) ? 1 : 0;
    if (first_s < last_s) return;
    const uint32_t first_uj = col_start + static_cast<uint32_t>(first_s);
    const uint32_t first_j_pad = (first_uj + 7u) & ~7u;
    AscendC::Brcb(buf0, lu_buf[static_cast<uint32_t>(first_s) * pad_n],
                  static_cast<uint8_t>(first_j_pad >> 3), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t s = first_s; s >= last_s; --s) {
        const uint32_t uj = col_start + static_cast<uint32_t>(s);
        const bool use_buf1 = ((first_s - s) & 1) != 0;
        if (s > last_s) {
            const uint32_t next_uj = uj - 1u;
            const uint32_t next_j_pad = (next_uj + 7u) & ~7u;
            if (use_buf1) {
                AscendC::Brcb(buf0, lu_buf[static_cast<uint32_t>(s - 1) * pad_n],
                              static_cast<uint8_t>(next_j_pad >> 3), {1, 8});
            } else {
                AscendC::Brcb(buf1, lu_buf[static_cast<uint32_t>(s - 1) * pad_n],
                              static_cast<uint8_t>(next_j_pad >> 3), {1, 8});
            }
        }
        const uint32_t start8 = (((uj - 1u) & 7u) + 1u) * 8u;
        const uint8_t reps = static_cast<uint8_t>((uj + 7u) >> 3);
        if (use_buf1) {
            AscendC::MulAddDst(y_pad[start8], y[uj * 8u], buf1_pad[start8],
                               64u, reps, y_repeat8);
        } else {
            AscendC::MulAddDst(y_pad[start8], y[uj * 8u], buf0_pad[start8],
                               64u, reps, y_repeat8);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// Ping-pong Brcb forward substitution L * y = P * R panel:
ELEC_FN void fwd_sub_panel(
    uint32_t col_start, uint32_t width, uint32_t n, uint32_t pad_n,
    const uint16_t* piv,
    const AscendC::LocalTensor<float>& lu_buf,
    const AscendC::LocalTensor<float>& y,
    const AscendC::LocalTensor<float>& row_tmp,
    const AscendC::LocalTensor<float>& buf0,
    const AscendC::LocalTensor<float>& buf1,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    if (width == 0u) return;
    const uint32_t first_j0 = col_start & ~7u;
    if (col_start + 1u < n) {
        AscendC::Brcb(buf0[first_j0 * 8u], lu_buf[first_j0],
                      static_cast<uint8_t>((pad_n - first_j0) >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t s = 0; s < width; ++s) {
        const uint32_t j = col_start + s;
        const uint32_t pivot = piv[j];
        const bool use_buf1 = (s & 1u) != 0u;
        if (pivot != j) {
            AscendC::Adds(row_tmp[0], y[pivot * 8u], 0.0f, 8u);
            AscendC::Adds(y[pivot * 8u], y[j * 8u], 0.0f, 8u);
        }
        if (s + 1u < width && j + 2u < n) {
            const uint32_t next_j0 = (j + 1u) & ~7u;
            if (use_buf1) {
                AscendC::Brcb(buf0[next_j0 * 8u], lu_buf[(s + 1u) * pad_n + next_j0],
                              static_cast<uint8_t>((pad_n - next_j0) >> 3), {1, 8});
            } else {
                AscendC::Brcb(buf1[next_j0 * 8u], lu_buf[(s + 1u) * pad_n + next_j0],
                              static_cast<uint8_t>((pad_n - next_j0) >> 3), {1, 8});
            }
        }
        if (pivot != j) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(y[j * 8u], row_tmp[0], 0.0f, 8u);
        }
        if (j + 1u < n) {
            const uint32_t sub_rows = n - 1u - j;
            auto src_y = (pivot != j) ? row_tmp[0] : y[j * 8u];
            if (use_buf1) {
                AscendC::MulAddDst(y[(j + 1u) * 8u], src_y, buf1[(j + 1u) * 8u],
                                   64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), y_repeat8);
            } else {
                AscendC::MulAddDst(y[(j + 1u) * 8u], src_y, buf0[(j + 1u) * 8u],
                                   64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), y_repeat8);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// Single step of 4-barrier compensated residual accumulation without LocalTensor copies:
ELEC_FN void residual_step_k4(
    uint32_t step, uint32_t width, uint32_t cur_k, uint32_t pad_n, uint32_t len,
    __ubuf__ const float* x0_col_ptr,
    const AscendC::LocalTensor<float>& M,
    const AscendC::LocalTensor<uint32_t>& col_offsets_u32,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16,
    const AscendC::LocalTensor<float>& col_vec,
    const AscendC::LocalTensor<float>& src_hi,
    const AscendC::LocalTensor<float>& dst_hi,
    const AscendC::LocalTensor<float>& s_lo_chunk,
    const AscendC::LocalTensor<float>& p_hi_4,
    const AscendC::LocalTensor<float>& p_err_cur,
    const AscendC::LocalTensor<float>& p_err_prev,
    const AscendC::LocalTensor<float>& a_hi_cur,
    const AscendC::LocalTensor<float>& a_hi_next,
    const AscendC::LocalTensor<float>& a_lo_cur,
    const AscendC::LocalTensor<float>& a_lo_next,
    const AscendC::LocalTensor<float>& v_tmp_4,
    const AscendC::LocalTensor<float>& u_tmp_4,
    const AscendC::LocalTensor<float>& d1_tmp_4) {
    float x_val[4], x_lo_val[4];
    if (step > 0u) {
        AscendC::Sub(u_tmp_4, dst_hi, u_tmp_4, len);
        AscendC::Add(d1_tmp_4, d1_tmp_4, p_err_prev, len);
    }
    for (uint32_t t = 0; t < cur_k; ++t) {
        const float x = -x0_col_ptr[t];
        const float x_hi = bits_float(float_bits(x) & 0xfffff000u);
        const float x_lo = x - x_hi;
        x_val[t] = x;
        x_lo_val[t] = x_lo;
        AscendC::Muls(p_hi_4[t * pad_n], col_vec, x, pad_n);
        AscendC::Muls(p_err_cur[t * pad_n], a_hi_cur, x_hi, pad_n);
    }
    AscendC::PipeBarrier<PIPE_V>();

    if (step > 0u) {
        AscendC::Add(s_lo_chunk, s_lo_chunk, d1_tmp_4, len);
    }
    AscendC::Sub(p_err_cur, p_err_cur, p_hi_4, len);
    AscendC::Add(dst_hi, src_hi, p_hi_4, len);
    if (step + 1u < width) {
        AscendC::Gather(col_vec, M[step + 1u], col_offsets_u32, uint32_t(0), pad_n);
    }
    AscendC::PipeBarrier<PIPE_V>();

    if (step > 0u) {
        AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
    }
    for (uint32_t t = 0; t < cur_k; ++t) {
        AscendC::Axpy(p_err_cur[t * pad_n], a_hi_cur, x_lo_val[t], pad_n);
    }
    AscendC::Sub(v_tmp_4, dst_hi, src_hi, len);
    if (step + 1u < width) {
        AscendC::And(a_hi_next.template ReinterpretCast<uint16_t>(),
                     col_vec.template ReinterpretCast<uint16_t>(),
                     split_mask_u16, pad_n * 2u);
    }
    AscendC::PipeBarrier<PIPE_V>();

    for (uint32_t t = 0; t < cur_k; ++t) {
        AscendC::Axpy(p_err_cur[t * pad_n], a_lo_cur, x_val[t], pad_n);
    }
    AscendC::Sub(u_tmp_4, dst_hi, v_tmp_4, len);
    AscendC::Sub(d1_tmp_4, p_hi_4, v_tmp_4, len);
    if (step + 1u < width) {
        AscendC::Sub(a_lo_next, col_vec, a_hi_next, pad_n);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// Single step of 3-barrier compensated residual accumulation (3 PipeBarriers per column):
ELEC_FN void residual_step_k4_3wave(
    uint32_t step, uint32_t width, uint32_t cur_k, uint32_t pad_n, uint32_t len,
    __ubuf__ const float* x0_col_ptr,
    const AscendC::LocalTensor<float>& M,
    const AscendC::LocalTensor<uint32_t>& col_offsets_u32,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16,
    const AscendC::LocalTensor<float>& col_vec_cur,
    const AscendC::LocalTensor<float>& col_vec_next,
    const AscendC::LocalTensor<float>& src_hi,
    const AscendC::LocalTensor<float>& dst_hi,
    const AscendC::LocalTensor<float>& s_lo_chunk,
    const AscendC::LocalTensor<float>& s_lo_acc2,
    const AscendC::LocalTensor<float>& p_hi_cur,
    const AscendC::LocalTensor<float>& p_hi_prev,
    const AscendC::LocalTensor<float>& p_err_4,
    const AscendC::LocalTensor<float>& a_hi_cur,
    const AscendC::LocalTensor<float>& a_hi_next,
    const AscendC::LocalTensor<float>& a_lo_cur,
    const AscendC::LocalTensor<float>& a_lo_next,
    const AscendC::LocalTensor<float>& v_tmp_4,
    const AscendC::LocalTensor<float>& u_tmp_4) {
    float x_val[4];
    if (step > 0u) {
        AscendC::Sub(u_tmp_4, src_hi, v_tmp_4, len);
        AscendC::Sub(p_hi_prev, p_hi_prev, v_tmp_4, len);
    }
    for (uint32_t t = 0; t < cur_k; ++t) {
        const float x = -x0_col_ptr[t];
        const float x_hi = bits_float(float_bits(x) & 0xfffff000u);
        const float x_lo = x - x_hi;
        x_val[t] = x;
        AscendC::Muls(p_hi_cur[t * pad_n], col_vec_cur, x, pad_n);
        AscendC::Muls(p_err_4[t * pad_n], a_hi_cur, x_hi, pad_n);
        AscendC::Axpy(s_lo_chunk[t * pad_n], a_hi_cur, x_lo, pad_n);
    }
    if (step + 1u < width) {
        AscendC::Gather(col_vec_next, M[step + 1u], col_offsets_u32, uint32_t(0), pad_n);
    }
    AscendC::PipeBarrier<PIPE_V>();

    if (step > 0u) {
        AscendC::Sub(u_tmp_4, dst_hi, u_tmp_4, len);
        AscendC::Add(s_lo_acc2, s_lo_acc2, p_hi_prev, len);
    }
    AscendC::Sub(p_err_4, p_err_4, p_hi_cur, len);
    for (uint32_t t = 0; t < cur_k; ++t) {
        AscendC::Axpy(s_lo_chunk[t * pad_n], a_lo_cur, x_val[t], pad_n);
    }
    AscendC::Add(dst_hi, src_hi, p_hi_cur, len);
    if (step + 1u < width) {
        AscendC::And(a_hi_next.template ReinterpretCast<uint16_t>(),
                     col_vec_next.template ReinterpretCast<uint16_t>(),
                     split_mask_u16, pad_n * 2u);
    }
    AscendC::PipeBarrier<PIPE_V>();

    if (step > 0u) {
        AscendC::Add(s_lo_acc2, s_lo_acc2, u_tmp_4, len);
    }
    AscendC::Add(s_lo_chunk, s_lo_chunk, p_err_4, len);
    AscendC::Sub(v_tmp_4, dst_hi, src_hi, len);
    if (step + 1u < width) {
        AscendC::Sub(a_lo_next, col_vec_next, a_hi_next, pad_n);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

ELEC_FN void apply_chunk_p4(
    uint32_t c_idx, uint32_t sub_id, uint32_t last_sub_id, uint32_t n, uint32_t pad_n, uint32_t my_w, uint32_t m_stride_p4,
    uint16_t* piv, volatile __ubuf__ int32_t* piv_stage_ptr,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<int32_t>& piv_gm,
    const AscendC::LocalTensor<float>& M,
    const AscendC::LocalTensor<float>& y,
    const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<int32_t>& piv_stage,
    const AscendC::LocalTensor<float>& coefficients,
    const AscendC::LocalTensor<float>& pivot_row,
    const AscendC::LocalTensor<float>& row_tmp,
    const AscendC::LocalTensor<float>& scratch,
    const AscendC::BinaryRepeatParams& m_p4_repeat,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    fence<AscendC::HardEvent::V_MTE2>();
    fence<AscendC::HardEvent::S_MTE2>();
    AscendC::DataCopy(lu_stage, lu_gm[c_idx * 8u * pad_n], 8u * pad_n);
    AscendC::DataCopy(piv_stage, piv_gm[c_idx * 8u], 8u);
    fence<AscendC::HardEvent::MTE2_V>();
    fence<AscendC::HardEvent::MTE2_S>();
    const uint32_t j0 = c_idx * 8u;
    const uint8_t brcb_blocks = static_cast<uint8_t>((pad_n - j0) >> 3);
    for (uint32_t s = 0; s < 8u; ++s) {
        const uint32_t j = j0 + s;
        uint32_t pivot = static_cast<uint32_t>(piv_stage_ptr[s]);
        if (pivot >= n) pivot = j;
        if (sub_id == last_sub_id) {
            piv[j] = static_cast<uint16_t>(pivot);
        }
        if (pivot != j) {
            AscendC::Adds(pivot_row, M[pivot * m_stride_p4], 0.0f, my_w);
            AscendC::Adds(row_tmp, M[j * m_stride_p4], 0.0f, my_w);
            if (sub_id == last_sub_id) {
                AscendC::Adds(scratch[0], y[pivot * 8u], 0.0f, 8u);
                AscendC::Adds(scratch[8], y[j * 8u], 0.0f, 8u);
            }
        }
        AscendC::Brcb(coefficients[j0 * 8u], lu_stage[s * pad_n + j0], brcb_blocks, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        if (pivot != j) {
            AscendC::Adds(M[pivot * m_stride_p4], row_tmp, 0.0f, my_w);
            AscendC::Adds(M[j * m_stride_p4], pivot_row, 0.0f, my_w);
            if (sub_id == last_sub_id) {
                AscendC::Adds(y[j * 8u], scratch[0], 0.0f, 8u);
                AscendC::Adds(y[pivot * 8u], scratch[8], 0.0f, 8u);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (j + 1u < n) {
            const uint32_t sub_rows = n - 1u - j;
            auto src_m = (pivot != j) ? pivot_row : M[j * m_stride_p4];
            AscendC::MulAddDst(M[(j + 1u) * m_stride_p4], src_m, coefficients[(j + 1u) * 8u],
                               my_w, static_cast<uint8_t>(sub_rows), m_p4_repeat);
            if (sub_id == last_sub_id) {
                AscendC::MulAddDst(y[(j + 1u) * 8u], y[j * 8u], coefficients[(j + 1u) * 8u],
                                   64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), y_repeat8);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

ELEC_FN void factor_chunk_p4(
    uint32_t c_idx, uint32_t sub_id, uint32_t last_sub_id, uint32_t n, uint32_t pad_n, uint32_t my_col_start, uint32_t my_w, uint32_t m_stride_p4,
    uint16_t* piv,
    volatile __ubuf__ float* col_vec_ptr, volatile __ubuf__ float* abs_col_ptr, volatile __ubuf__ float* inv_diag_ptr,
    volatile __ubuf__ int32_t* piv_stage_ptr, volatile __ubuf__ float* inv_diag_stage_ptr,
    volatile __ubuf__ const uint64_t* reduced_u64_ptr,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<int32_t>& piv_gm,
    const AscendC::GlobalTensor<float>& inv_diag_gm,
    const AscendC::LocalTensor<float>& M,
    const AscendC::LocalTensor<float>& y,
    const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<int32_t>& piv_stage,
    const AscendC::LocalTensor<float>& inv_diag_stage,
    const AscendC::LocalTensor<float>& coefficients,
    const AscendC::LocalTensor<float>& col_vec,
    const AscendC::LocalTensor<float>& abs_col,
    const AscendC::LocalTensor<float>& pivot_row,
    const AscendC::LocalTensor<float>& row_tmp,
    const AscendC::LocalTensor<float>& scratch,
    const AscendC::LocalTensor<float>& reduced,
    const AscendC::LocalTensor<uint32_t>& col_offsets_u32,
    const AscendC::BinaryRepeatParams& m_p4_repeat,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    const uint32_t j0 = c_idx * 8u;
    const uint32_t rem_a = pad_n - j0;
    const uint32_t chunk0 = rem_a < 64u ? rem_a : 64u;
    const uint64_t c_mask = (chunk0 == 64u) ? ~0ULL : ((1ULL << chunk0) - 1ULL);
    for (uint32_t s = 0; s < 8u; ++s) {
        const uint32_t j = j0 + s;
        const uint32_t local_col = j - my_col_start;
        AscendC::Gather(col_vec, M[local_col], col_offsets_u32, uint32_t(0), n);
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t mask[2] = {c_mask & (~0ULL << s), 0ULL};
        AscendC::Abs(abs_col[j0], col_vec[j0], mask, 1u, {1, 1, 8, 8});
        if (rem_a > 64u) {
            AscendC::Abs(abs_col[j0 + 64u], col_vec[j0 + 64u], rem_a - 64u);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax(reduced, abs_col[j0], scratch, n - j0, true);
        fence<AscendC::HardEvent::V_S>();

        const uint64_t red_val = reduced_u64_ptr[0];
        uint32_t pivot = j0 + static_cast<uint32_t>(red_val >> 32);
        if (pivot >= n) pivot = j;

        auto lu_col_slot = lu_stage[s * pad_n];
        const bool has_rem_cols = (local_col + 1u < my_w);
        const uint32_t c0 = (local_col + 1u) & ~7u;
        const uint32_t rem = has_rem_cols ? (my_w - c0) : 0u;

        // Issue row copies on PIPE_V immediately while PIPE_S computes 1.0f / diag:
        if (has_rem_cols) {
            AscendC::Adds(pivot_row, M[pivot * m_stride_p4 + c0], 0.0f, rem);
        }
        if (pivot != j) {
            if (has_rem_cols) {
                AscendC::Adds(row_tmp, M[j * m_stride_p4 + c0], 0.0f, rem);
            }
            if (sub_id == last_sub_id) {
                AscendC::Adds(scratch[0], y[pivot * 8u], 0.0f, 8u);
                AscendC::Adds(scratch[8], y[j * 8u], 0.0f, 8u);
            }
        }

        const float diag = col_vec_ptr[pivot];
        const float inv_diag = (diag != 0.0f) ? (1.0f / diag) : 0.0f;
        if (pivot != j) {
            col_vec_ptr[pivot] = col_vec_ptr[j];
        }
        col_vec_ptr[j] = 0.0f;
        abs_col_ptr[j] = -1.0f;
        piv_stage_ptr[s] = static_cast<int32_t>(pivot);
        inv_diag_stage_ptr[s] = inv_diag;
        if (sub_id == last_sub_id) {
            piv[j] = static_cast<uint16_t>(pivot);
            inv_diag_ptr[j] = inv_diag;
        }
        fence<AscendC::HardEvent::S_V>();

        AscendC::Muls(lu_col_slot, col_vec, -inv_diag, pad_n);
        AscendC::PipeBarrier<PIPE_V>();

        if (has_rem_cols || (sub_id == last_sub_id && j + 1u < n)) {
            AscendC::Brcb(coefficients[j0 * 8u], lu_col_slot[j0],
                          static_cast<uint8_t>((pad_n - j0) >> 3), {1, 8});
        }
        if (pivot != j) {
            if (has_rem_cols) {
                AscendC::Adds(M[pivot * m_stride_p4 + c0], row_tmp, 0.0f, rem);
                AscendC::Adds(M[j * m_stride_p4 + c0], pivot_row, 0.0f, rem);
            }
            if (sub_id == last_sub_id) {
                AscendC::Adds(y[j * 8u], scratch[0], 0.0f, 8u);
                AscendC::Adds(y[pivot * 8u], scratch[8], 0.0f, 8u);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        if (j + 1u < n) {
            const uint32_t sub_rows = n - 1u - j;
            if (has_rem_cols) {
                AscendC::MulAddDst(M[(j + 1u) * m_stride_p4 + c0], pivot_row,
                                   coefficients[(j + 1u) * 8u],
                                   rem, static_cast<uint8_t>(sub_rows), m_p4_repeat);
            }
            if (sub_id == last_sub_id) {
                AscendC::MulAddDst(y[(j + 1u) * 8u], y[j * 8u], coefficients[(j + 1u) * 8u],
                                   64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), y_repeat8);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    fence<AscendC::HardEvent::V_MTE3>();
    fence<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopy(lu_gm[c_idx * 8u * pad_n], lu_stage, 8u * pad_n);
    if (sub_id != last_sub_id) {
        AscendC::DataCopy(piv_gm[c_idx * 8u], piv_stage, 8u);
        AscendC::DataCopy(inv_diag_gm[c_idx * 8u], inv_diag_stage, 8u);
    }
    fence<AscendC::HardEvent::MTE3_V>();
    fence<AscendC::HardEvent::MTE3_S>();
}

ELEC_FN void two_sum_combine_buf(
    uint32_t offset, uint32_t chunk_len,
    const AscendC::LocalTensor<float>& other_hi,
    const AscendC::LocalTensor<float>& other_lo,
    const AscendC::LocalTensor<float>& s_hi,
    const AscendC::LocalTensor<float>& s_lo,
    const AscendC::LocalTensor<float>& t_sum_4,
    const AscendC::LocalTensor<float>& d_tmp,
    const AscendC::LocalTensor<float>& u_tmp) {
    AscendC::Add(s_lo[offset], s_lo[offset], other_lo, chunk_len);
    AscendC::Add(t_sum_4, s_hi[offset], other_hi, chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(d_tmp, t_sum_4, s_hi[offset], chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(u_tmp, t_sum_4, d_tmp, chunk_len);
    AscendC::Sub(other_hi, other_hi, d_tmp, chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(u_tmp, s_hi[offset], u_tmp, chunk_len);
    AscendC::Add(s_lo[offset], s_lo[offset], other_hi, chunk_len);
    AscendC::Adds(s_hi[offset], t_sum_4, 0.0f, chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(s_lo[offset], s_lo[offset], u_tmp, chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
}

template <bool SMALL_CONST = false>
ELEC_FN void two_sum_combine_from_gm(
    uint32_t src_sub, uint32_t total_len,
    const AscendC::GlobalTensor<float>& r_gm,
    const AscendC::LocalTensor<float>& s_hi,
    const AscendC::LocalTensor<float>& s_lo,
    const AscendC::LocalTensor<float>& t_sum_4,
    const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<float>& y) {
    if (SMALL_CONST || total_len <= 1024u) {
        auto other_hi = lu_stage[0];
        auto other_lo = lu_stage[total_len];
        auto d_tmp = y[0];
        auto u_tmp = y[1056];
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(other_hi, r_gm[src_sub * 2048u], 2u * total_len);
        fence<AscendC::HardEvent::MTE2_V>();
        two_sum_combine_buf(0u, total_len, other_hi, other_lo, s_hi, s_lo, t_sum_4, d_tmp, u_tmp);
        return;
    }
    if constexpr (!SMALL_CONST) {
        for (uint32_t offset = 0; offset < total_len; offset += 1024u) {
            const uint32_t chunk_len = (total_len - offset < 1024u) ? (total_len - offset) : 1024u;
            auto other_hi = lu_stage[0];
            auto other_lo = lu_stage[1056];
            auto d_tmp = y[0];
            auto u_tmp = y[1056];
            fence<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(other_hi, r_gm[src_sub * 4096u + offset], chunk_len);
            AscendC::DataCopy(other_lo, r_gm[src_sub * 4096u + 2048u + offset], chunk_len);
            fence<AscendC::HardEvent::MTE2_V>();
            two_sum_combine_buf(offset, chunk_len, other_hi, other_lo, s_hi, s_lo, t_sum_4, d_tmp, u_tmp);
        }
    }
}

template <bool SMALL_CONST = false>
ELEC_FN void two_sum_combine_pair_from_gm(
    uint32_t src_sub0, uint32_t src_sub1, uint32_t total_len,
    const AscendC::GlobalTensor<float>& r_gm,
    const AscendC::LocalTensor<float>& s_hi,
    const AscendC::LocalTensor<float>& s_lo,
    const AscendC::LocalTensor<float>& t_sum_4,
    const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<float>& alt_stage,
    const AscendC::LocalTensor<float>& y) {
    if (SMALL_CONST || total_len <= 1024u) {
        auto other_hi0 = lu_stage[0];
        auto other_lo0 = lu_stage[total_len];
        auto other_hi1 = alt_stage[0];
        auto other_lo1 = alt_stage[total_len];
        auto t2 = alt_stage[2048];
        auto d2 = alt_stage[3072];
        auto u2 = alt_stage[4096];
        auto d_tmp = y[0];
        auto u_tmp = y[1056];
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(other_hi0, r_gm[src_sub0 * 2048u], 2u * total_len);
        fence<AscendC::HardEvent::MTE2_V>();
        AscendC::DataCopy(other_hi1, r_gm[src_sub1 * 2048u], 2u * total_len);
        AscendC::Add(s_lo, s_lo, other_lo0, total_len);
        AscendC::Add(t_sum_4, s_hi, other_hi0, total_len);
        AscendC::PipeBarrier<PIPE_V>();
        fence<AscendC::HardEvent::MTE2_V>();
        AscendC::Sub(d_tmp, t_sum_4, s_hi, total_len);
        AscendC::Add(t2, t_sum_4, other_hi1, total_len);
        AscendC::Add(s_lo, s_lo, other_lo1, total_len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(u_tmp, t_sum_4, d_tmp, total_len);
        AscendC::Sub(other_hi0, other_hi0, d_tmp, total_len);
        AscendC::Sub(d2, t2, t_sum_4, total_len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(u_tmp, s_hi, u_tmp, total_len);
        AscendC::Add(s_lo, s_lo, other_hi0, total_len);
        AscendC::Sub(u2, t2, d2, total_len);
        AscendC::Sub(other_hi1, other_hi1, d2, total_len);
        AscendC::Adds(s_hi, t2, 0.0f, total_len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(s_lo, s_lo, u_tmp, total_len);
        AscendC::Sub(u2, t_sum_4, u2, total_len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(u2, u2, other_hi1, total_len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(s_lo, s_lo, u2, total_len);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }
    if constexpr (!SMALL_CONST) {
        for (uint32_t offset = 0; offset < total_len; offset += 1024u) {
            const uint32_t chunk_len = (total_len - offset < 1024u) ? (total_len - offset) : 1024u;
            auto other_hi0 = lu_stage[0];
            auto other_lo0 = lu_stage[1056];
            auto other_hi1 = alt_stage[0];
            auto other_lo1 = alt_stage[1056];
            auto d_tmp = y[0];
            auto u_tmp = y[1056];
            fence<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(other_hi0, r_gm[src_sub0 * 4096u + offset], chunk_len);
            AscendC::DataCopy(other_lo0, r_gm[src_sub0 * 4096u + 2048u + offset], chunk_len);
            fence<AscendC::HardEvent::MTE2_V>();
            AscendC::DataCopy(other_hi1, r_gm[src_sub1 * 4096u + offset], chunk_len);
            AscendC::DataCopy(other_lo1, r_gm[src_sub1 * 4096u + 2048u + offset], chunk_len);
            two_sum_combine_buf(offset, chunk_len, other_hi0, other_lo0, s_hi, s_lo, t_sum_4, d_tmp, u_tmp);
            fence<AscendC::HardEvent::MTE2_V>();
            two_sum_combine_buf(offset, chunk_len, other_hi1, other_lo1, s_hi, s_lo, t_sum_4, d_tmp, u_tmp);
        }
    }
}

ELEC_FN void apply_chunk_p8(
    uint32_t col_start, uint32_t chunk_w, uint32_t sub_id, uint32_t n, uint32_t pad_n, uint32_t row_w,
    uint16_t* piv, volatile __ubuf__ int32_t* piv_stage_ptr,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<int32_t>& piv_gm,
    const AscendC::LocalTensor<float>& M,
    const AscendC::LocalTensor<float>& lu_chunk,
    const AscendC::LocalTensor<int32_t>& piv_stage,
    const AscendC::LocalTensor<float>& buf0,
    const AscendC::LocalTensor<float>& buf1,
    const AscendC::LocalTensor<float>& pivot_row,
    const AscendC::LocalTensor<float>& row_tmp,
    const AscendC::BinaryRepeatParams& m_p8_repeat) {
    fence<AscendC::HardEvent::V_MTE2>();
    fence<AscendC::HardEvent::S_MTE2>();
    AscendC::DataCopy(lu_chunk, lu_gm[col_start * pad_n], chunk_w * pad_n);
    AscendC::DataCopy(piv_stage, piv_gm[col_start], chunk_w);
    fence<AscendC::HardEvent::MTE2_V>();
    fence<AscendC::HardEvent::MTE2_S>();
    if (col_start + 1u < n) {
        AscendC::Brcb(buf0[col_start * 8u], lu_chunk[col_start],
                      static_cast<uint8_t>((pad_n - col_start) >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t s = 0; s < chunk_w; ++s) {
        const uint32_t j = col_start + s;
        uint32_t pivot = static_cast<uint32_t>(piv_stage_ptr[s]);
        if (pivot >= n) pivot = j;
        if (sub_id == 7u) {
            piv[j] = static_cast<uint16_t>(pivot);
        }
        const bool use_buf1 = (s & 1u) != 0u;
        if (pivot != j) {
            AscendC::Adds(pivot_row[0], M[pivot * 40u], 0.0f, row_w);
            AscendC::Adds(M[pivot * 40u], M[j * 40u], 0.0f, row_w);
        }
        if (s + 1u < chunk_w && j + 2u < n) {
            const uint32_t next_j0 = (j + 1u) & ~7u;
            if (use_buf1) {
                AscendC::Brcb(buf0[next_j0 * 8u], lu_chunk[(s + 1u) * pad_n + next_j0],
                              static_cast<uint8_t>((pad_n - next_j0) >> 3), {1, 8});
            } else {
                AscendC::Brcb(buf1[next_j0 * 8u], lu_chunk[(s + 1u) * pad_n + next_j0],
                              static_cast<uint8_t>((pad_n - next_j0) >> 3), {1, 8});
            }
        }
        if (pivot != j) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(M[j * 40u], pivot_row[0], 0.0f, row_w);
        }
        if (j + 1u < n) {
            const uint32_t sub_rows = n - 1u - j;
            auto src_m = (pivot != j) ? pivot_row[0] : M[j * 40u];
            if (use_buf1) {
                AscendC::MulAddDst(M[(j + 1u) * 40u], src_m, buf1[(j + 1u) * 8u],
                                   row_w, static_cast<uint8_t>(sub_rows), m_p8_repeat);
            } else {
                AscendC::MulAddDst(M[(j + 1u) * 40u], src_m, buf0[(j + 1u) * 8u],
                                   row_w, static_cast<uint8_t>(sub_rows), m_p8_repeat);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

ELEC_FN void factor_chunk_p8(
    uint32_t col_start, uint32_t chunk_w, uint32_t sub_id, uint32_t n, uint32_t pad_n, uint32_t my_col_start, uint32_t row_w,
    uint16_t* piv,
    volatile __ubuf__ float* col_vec_ptr,
    volatile __ubuf__ float* abs_col_ptr, volatile __ubuf__ float* inv_diag_ptr,
    volatile __ubuf__ int32_t* piv_stage_ptr, volatile __ubuf__ float* inv_diag_stage_ptr,
    volatile __ubuf__ const uint64_t* reduced_u64_ptr,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<int32_t>& piv_gm,
    const AscendC::GlobalTensor<float>& inv_diag_gm,
    const AscendC::LocalTensor<float>& M,
    const AscendC::LocalTensor<float>& lu_chunk,
    const AscendC::LocalTensor<int32_t>& piv_stage,
    const AscendC::LocalTensor<float>& inv_diag_stage,
    const AscendC::LocalTensor<float>& coefficients,
    const AscendC::LocalTensor<float>& col_vec,
    const AscendC::LocalTensor<float>& abs_col,
    const AscendC::LocalTensor<float>& pivot_row,
    const AscendC::LocalTensor<float>& row_tmp,
    const AscendC::LocalTensor<float>& scratch,
    const AscendC::LocalTensor<float>& reduced,
    const AscendC::LocalTensor<uint32_t>& col_offsets_u32,
    const AscendC::BinaryRepeatParams& m_p8_repeat) {
    (void)abs_col_ptr;
    (void)row_tmp;
    const uint32_t local_col0 = col_start - my_col_start;

    for (uint32_t s = 0; s < chunk_w; ++s) {
        if (sub_id != 7u && s + 1u == chunk_w && chunk_w > 1u) {
            fence<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(lu_gm[col_start * pad_n], lu_chunk[0], (chunk_w - 1u) * pad_n);
        }
        const uint32_t j = col_start + s;
        const uint32_t j0 = j & ~7u;
        const uint32_t s8 = j & 7u;
        const uint32_t rem_a = n - j0;
        const uint32_t chunk0 = rem_a < 64u ? rem_a : 64u;
        const uint64_t c_mask = (chunk0 == 64u) ? ~0ULL : ((1ULL << chunk0) - 1ULL);
        const uint32_t local_col = local_col0 + s;

        auto lu_col_slot = lu_chunk[s * pad_n];
        const bool has_rem_cols = (local_col + 1u < row_w);
        const bool is_last_s = (s + 1u == chunk_w);
        const uint32_t c0 = (local_col + 1u) & ~7u;
        const uint32_t rem = has_rem_cols ? (row_w - c0) : 0u;

        AscendC::Gather(col_vec[j0], M[j0 * 40u + local_col], col_offsets_u32, uint32_t(0), n - j0);
        uint32_t pivot = j;
        int32_t best_bits = 0;
        if (j + 1u < n) {
            AscendC::PipeBarrier<PIPE_V>();
            uint64_t mask[2] = {c_mask & (~0ULL << s8), 0ULL};
            AscendC::Abs(abs_col[j0], col_vec[j0], mask, 1u, {1, 1, 8, 8});
            if (rem_a > 64u) {
                AscendC::Abs(abs_col[j0 + 64u], col_vec[j0 + 64u], rem_a - 64u);
            }
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t reps = (n - j0 + 63u) >> 6;
            AscendC::WholeReduceMax<float>(
                reduced, abs_col[j0], 64, static_cast<int32_t>(reps), 1, 1, 8,
                AscendC::ReduceOrder::ORDER_VALUE_INDEX);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            if (j0 > 0u) {
                AscendC::Gather(col_vec[0], M[local_col], col_offsets_u32, uint32_t(0), j0);
            }
            if (has_rem_cols) {
                AscendC::Brcb(coefficients[j0 * 8u], col_vec[j0],
                              static_cast<uint8_t>((pad_n - j0) >> 3), {1, 8});
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(abs_col[j0], -1.0f, 8u);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

            const uint64_t best_rv = reduced_u64_ptr[0];
            best_bits = static_cast<int32_t>(best_rv);
            pivot = j0 + static_cast<uint32_t>(best_rv >> 32);
            for (uint32_t r = 1u; r < reps; ++r) {
                const uint64_t rv = reduced_u64_ptr[r];
                const int32_t bits = static_cast<int32_t>(rv);
                if (bits > best_bits) {
                    best_bits = bits;
                    pivot = j0 + (r << 6) + static_cast<uint32_t>(rv >> 32);
                }
            }
            if (pivot >= n) pivot = j;
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            if (j0 > 0u) {
                AscendC::Gather(col_vec[0], M[local_col], col_offsets_u32, uint32_t(0), j0);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }

        // Start 1.0f / |diag| and UB scalar loads immediately:
        float inv_diag = 0.0f;
        if (j + 1u < n) {
            const float diag = col_vec_ptr[pivot];
            const float col_j = col_vec_ptr[j];
            const float inv_abs = (best_bits > 0) ? (1.0f / bits_float(static_cast<uint32_t>(best_bits))) : 0.0f;
            if (pivot != j) {
                col_vec_ptr[pivot] = col_j;
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            }
            const uint32_t sign_bit = (best_bits > 0) ? (float_bits(diag) & 0x80000000u) : 0u;
            inv_diag = bits_float(float_bits(inv_abs) | sign_bit);
        } else {
            const float diag = col_vec_ptr[j];
            inv_diag = (diag != 0.0f) ? (1.0f / diag) : 0.0f;
        }

        if (has_rem_cols && j + 1u < n) {
            AscendC::Muls(scratch, M[pivot * 40u + c0], -inv_diag, rem);
            if (pivot != j) {
                AscendC::Adds(pivot_row, M[pivot * 40u + c0], 0.0f, rem);
                AscendC::Adds(M[pivot * 40u + c0], M[j * 40u + c0], 0.0f, rem);
                AscendC::Adds(coefficients[pivot * 8u], coefficients[j * 8u], 0.0f, 8u);
            }
            if (is_last_s) {
                if (pivot != j) {
                    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                }
                AscendC::Muls(lu_col_slot, col_vec, -inv_diag, pad_n);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (pivot != j) {
                AscendC::Adds(M[j * 40u + c0], pivot_row, 0.0f, rem);
            }
            if (is_last_s) {
                if (sub_id != 7u) {
                    fence<AscendC::HardEvent::V_MTE3>();
                    AscendC::DataCopy(lu_gm[(col_start + chunk_w - 1u) * pad_n], lu_col_slot, pad_n);
                }
            } else {
                if (pivot != j) {
                    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                }
                AscendC::Muls(lu_col_slot, col_vec, -inv_diag, pad_n);
            }
            const uint32_t sub_rows = n - 1u - j;
            if (rem == 8u) {
                const AscendC::BinaryRepeatParams m_p8_rep8{5, 0, 1, 40, 0, 8};
                AscendC::MulAddDst(M[(j + 1u) * 40u + c0], scratch,
                                   coefficients[(j + 1u) * 8u],
                                   64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), m_p8_rep8);
            } else {
                AscendC::MulAddDst(M[(j + 1u) * 40u + c0], scratch,
                                   coefficients[(j + 1u) * 8u],
                                   rem, static_cast<uint8_t>(sub_rows), m_p8_repeat);
            }
            piv_stage_ptr[s] = static_cast<int32_t>(pivot);
            inv_diag_stage_ptr[local_col] = inv_diag;
            if (sub_id == 7u) {
                piv[j] = static_cast<uint16_t>(pivot);
                inv_diag_ptr[j] = inv_diag;
            } else if (is_last_s) {
                fence<AscendC::HardEvent::S_MTE3>();
                AscendC::DataCopy(piv_gm[col_start], piv_stage, (chunk_w + 7u) & ~7u);
                if (!has_rem_cols) {
                    AscendC::DataCopy(inv_diag_gm[my_col_start], inv_diag_stage, row_w);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            if (j + 1u < n && pivot != j) {
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            }
            AscendC::Muls(lu_col_slot, col_vec, -inv_diag, pad_n);
            piv_stage_ptr[s] = static_cast<int32_t>(pivot);
            inv_diag_stage_ptr[local_col] = inv_diag;
            if (sub_id == 7u) {
                piv[j] = static_cast<uint16_t>(pivot);
                inv_diag_ptr[j] = inv_diag;
            } else if (is_last_s) {
                fence<AscendC::HardEvent::S_MTE3>();
                AscendC::DataCopy(piv_gm[col_start], piv_stage, (chunk_w + 7u) & ~7u);
                if (!has_rem_cols) {
                    AscendC::DataCopy(inv_diag_gm[my_col_start], inv_diag_stage, row_w);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (sub_id != 7u && is_last_s) {
                fence<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopy(lu_gm[(col_start + chunk_w - 1u) * pad_n], lu_col_slot, pad_n);
            }
        }
    }
    if (sub_id != 7u) {
        fence<AscendC::HardEvent::MTE3_V>();
        fence<AscendC::HardEvent::MTE3_S>();
    }
}

// 8-Core-Per-Batch Wavefront LU + Resident A_orig + Parallel Tree Compensated Residual Solver
// for square systems with n == 200 or n == 256 and 1 <= k <= 8.
template <typename T, uint32_t B_CONST = 0u, uint32_t N_CONST = 0u, uint32_t K_CONST = 0u>
__aicore__ __attribute__((noinline)) int solve_square_large_p8(
    const __gm__ Config* c_ptr, GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals,
    GM_ADDR ws_base, uint64_t per_core) {
    const uint32_t n = (N_CONST != 0u) ? N_CONST : static_cast<uint32_t>(c_ptr->n);
    const uint32_t k = (K_CONST != 0u) ? K_CONST : static_cast<uint32_t>(c_ptr->k);
    const uint32_t pad_n = (n + 7u) & ~7u;
    const uint32_t block_idx = AscendC::GetBlockIdx();
    const uint32_t batch = block_idx >> 3;
    const uint32_t sub_id = block_idx & 7u;
    if constexpr (B_CONST == 0u) {
        if (batch >= c_ptr->batch) return SUCCESS;
    }

    // 8-panel partitioning:
    // - n == 200: Panel 0 has width 32, Panels 1..7 have width w_p = 24 (32 + 7 * 24 = 200)
    // - n == 256: Panels 0..7 all have width w_p = 32 (8 * 32 = 256)
    const uint32_t w_p = (n == 256u) ? 32u : 24u;
    const uint32_t my_col_start = (sub_id == 0u) ? 0u : (32u + (sub_id - 1u) * w_p);
    const uint32_t my_w = (sub_id == 0u) ? 0u + 32u : w_p;
    const uint32_t row_w = (sub_id < 7u) ? my_w : (my_w + 8u); // Core 7 augments y into M[r * 40 + my_w .. my_w + 7]!
    constexpr uint32_t m_stride_p8 = 40u; // 5 blocks (odd block stride, conflict-free)

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    // Single 48,896-float (191.0 KiB) UB buffer:
    // [0 .. 10240): M in Phase 1 (256 * 40 = 10,240 floats); lu_buf0 [0 .. 8192) after Phase 1!
    // [10240 .. 16384): lu_stage_p1 in Phase 1 (24 * 256 = 6,144 floats); lu_buf1 [8192 .. 16384) after Phase 1!
    // [16384 .. 24576): lu_panel7 (32 * 256 = 8,192 floats; permanently holds Panel 7 on Core 7!)
    // [24576 .. 34816): A_orig (256 * 40 = 10,240 floats; permanently holds original A panel!)
    // [34816 .. 36928): x0 (2112 floats)
    // [36928 .. 38976): s_hi / x_out (2048 floats)
    // [38976 .. 40000): t_sum_4 (1024 floats)
    // [40000 .. 42112): lu_stage (2112 floats)
    // [42112 .. 44224): y (2112 floats)
    // [44224 .. 46336): coefficients / s_lo (2112 floats)
    // [46336 .. 48896): vec (10 * 256 = 2560 floats)
    pipe.InitBuffer(work_buf, 48896u * sizeof(float));

    auto base = work_buf.Get<float>();
    auto M = base[0];
    auto lu_buf0 = base[0];
    auto lu_buf1 = base[8192];
    auto lu_stage_p1 = base[10560];
    auto lu_panel7 = base[16384];
    auto A_orig = base[24576];
    auto x0 = base[34816];
    auto s_hi = base[36928];
    auto x_out = base[36928];
    auto t_sum_4 = base[38976];
    auto lu_stage_pad = base[39936];
    auto lu_stage = base[40000];
    auto y_pad = base[42048];
    auto y = base[42112];
    auto coeff_pad = base[44160];
    auto coefficients = base[44224];
    auto s_lo = base[44224];
    auto vec = base[46336];

    auto col_vec = vec[0 * 256];
    auto lu_col = vec[1 * 256];
    auto piv_stage = vec[1 * 256].template ReinterpretCast<int32_t>();
    auto inv_diag_stage = vec[1 * 256 + 64];
    auto abs_col = vec[2 * 256];      // 320 floats: [512 .. 832) so WholeReduceMax at j0 > 0 never reads past -1.0f!
    auto pivot_row = vec[832];        // 256 floats: [832 .. 1088)
    auto row_tmp = vec[1088];         // 256 floats: [1088 .. 1344)
    auto inv_diag_vec = vec[1344];    // 256 floats: [1344 .. 1600)
    auto scratch = vec[1600];         // 256 floats: [1600 .. 1856)
    auto reduced = vec[1856];         // 64 floats:  [1856 .. 1920)
    auto unit_masks = vec[1920];      // 64 floats:  [1920 .. 1984)
    auto col_offsets = vec[8 * 256].template ReinterpretCast<int32_t>();
    auto y_col_offsets = vec[9 * 256].template ReinterpretCast<int32_t>();

    auto p_hi_4 = lu_stage[0];
    auto p_err_4 = lu_stage[1056];
    auto v_tmp_4 = y[0];
    auto u_tmp_4 = y[1056];

    AscendC::GlobalTensor<T> input_a, input_b, output_x, output_r;
    input_a.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A));
    input_b.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(B));
    output_x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(solution));
    if (residuals != nullptr) {
        output_r.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(residuals));
    }

    AscendC::GlobalTensor<float> lu_gm, inv_diag_gm, x0_gm, r_gm;
    AscendC::GlobalTensor<int32_t> piv_gm;
    GM_ADDR batch_ws = ws_base + 16384u + static_cast<uint64_t>(batch) * (7u * per_core);
    __gm__ float* batch_ws_f32 = reinterpret_cast<__gm__ float*>(batch_ws);
    lu_gm.SetGlobalBuffer(batch_ws_f32);                              // 65,536 floats (256 KiB)
    piv_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(batch_ws_f32 + 65536u)); // 256 int32s
    inv_diag_gm.SetGlobalBuffer(batch_ws_f32 + 65536u + 256u);        // 256 floats
    x0_gm.SetGlobalBuffer(batch_ws_f32 + 66048u);                     // 2048 floats
    r_gm.SetGlobalBuffer(batch_ws_f32 + 68096u);                      // 7 * 4096 floats

    const AscendC::DataCopyParams a_copy{
        static_cast<uint16_t>(n),
        static_cast<uint16_t>(my_w >> 3),
        static_cast<uint16_t>((n - my_w) >> 3),
        static_cast<uint16_t>((m_stride_p8 - my_w) >> 3)};
    // Overlap MTE2 panel load directly into M with PIPE_V setup so Core 0 starts c_idx = 0 immediately:
    if constexpr (sizeof(T) == 4) {
        AscendC::DataCopy(M, input_a[batch * n * n + my_col_start], a_copy);
    }

    AscendC::CreateVecIndex(y_col_offsets, int32_t(0), pad_n);
    AscendC::Duplicate(abs_col, -1.0f, 320u);
    AscendC::Duplicate(unit_masks, 0.0f, 64u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(col_offsets, y_col_offsets, static_cast<int32_t>(m_stride_p8 * sizeof(float)), pad_n);
    AscendC::Muls(y_col_offsets, y_col_offsets, static_cast<int32_t>(8u * sizeof(float)), pad_n);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (K_CONST == 0u) {
        if (sub_id == 7u && k > 5u) {
            fence<AscendC::HardEvent::V_S>();
            for (uint32_t t = 0; t < 8u; ++t) {
                unit_masks.SetValue(t * 8u + t, 1.0f);
            }
            fence<AscendC::HardEvent::S_V>();
        }
    }

    auto col_offsets_u32 = col_offsets.template ReinterpretCast<uint32_t>();
    auto y_col_offsets_u32 = y_col_offsets.template ReinterpretCast<uint32_t>();
    volatile __ubuf__ float* col_vec_ptr = reinterpret_cast<volatile __ubuf__ float*>(col_vec.GetPhyAddr());
    volatile __ubuf__ float* abs_col_ptr = reinterpret_cast<volatile __ubuf__ float*>(abs_col.GetPhyAddr());
    volatile __ubuf__ float* inv_diag_ptr = reinterpret_cast<volatile __ubuf__ float*>(inv_diag_vec.GetPhyAddr());
    volatile __ubuf__ int32_t* piv_stage_ptr = reinterpret_cast<volatile __ubuf__ int32_t*>(piv_stage.GetPhyAddr());
    volatile __ubuf__ float* inv_diag_stage_ptr = reinterpret_cast<volatile __ubuf__ float*>(inv_diag_stage.GetPhyAddr());
    __ubuf__ const float* x0_ptr = reinterpret_cast<__ubuf__ const float*>(x0.GetPhyAddr());
    volatile __ubuf__ const uint64_t* reduced_u64_ptr = reinterpret_cast<volatile __ubuf__ const uint64_t*>(reduced.GetPhyAddr());

    const AscendC::BinaryRepeatParams m_p8_repeat{1, 1, 0, static_cast<uint8_t>(m_stride_p8 >> 3), 0, 1};
    const AscendC::BinaryRepeatParams y_repeat8{1, 0, 1, 8, 0, 8};
    const AscendC::BinaryRepeatParams lane_rep{1, 1, 0, 8, 8, 0};
    const AscendC::UnaryRepeatParams y_to_m_rep{1, 1, 5, 1};
    const AscendC::UnaryRepeatParams m_to_y_rep{1, 1, 1, 5};
    uint16_t piv[256];

    if constexpr (sizeof(T) == 4) {
        fence<AscendC::HardEvent::MTE2_V>();
    } else {
        load_panel_a<T>(n, batch, my_col_start, my_w, my_w, m_stride_p8, input_a, M, lu_stage);
    }

    const uint64_t rhs_batch_stride = (B_CONST != 0u)
        ? static_cast<uint64_t>(N_CONST * K_CONST)
        : c_ptr->rhs_batch_stride;
    const uint64_t b_offset = static_cast<uint64_t>(batch) * rhs_batch_stride;

    // Phase 1: 8-Core Wavefront Panel LU with 8-column chunks
    // - n == 200: 4 chunks on Panel 0 + 3 chunks on Panels 1..6 (22 syncs, leaving only 24 cols for Panel 7)
    // - n == 256: 4 chunks of 8 per panel for Panels 0..6 (28 syncs)
    const uint32_t c7 = (32u + 6u * w_p) >> 3;
    for (uint32_t c_idx = 0; c_idx <= c7; ++c_idx) {
        if (c_idx > 0u) {
            const uint32_t prev = c_idx - 1u;
            const uint32_t prev_owner = (n == 256u) ? (prev >> 2) : ((prev < 4u) ? 0u : (1u + (prev - 4u) / 3u));
            if (sub_id > prev_owner) {
                apply_chunk_p8(
                    prev * 8u, 8u, sub_id, n, pad_n, row_w, piv, &piv_stage_ptr[32],
                    lu_gm, piv_gm, M, lu_panel7, piv_stage[32], coefficients, lu_stage, pivot_row, row_tmp,
                    m_p8_repeat);
            }
        }
        if (c_idx < c7) {
            const uint32_t owner = (n == 256u) ? (c_idx >> 2) : ((c_idx < 4u) ? 0u : (1u + (c_idx - 4u) / 3u));
            if (sub_id == owner) {
                factor_chunk_p8(
                    c_idx * 8u, 8u, sub_id, n, pad_n, my_col_start, row_w, piv,
                    col_vec_ptr, abs_col_ptr, inv_diag_ptr,
                    piv_stage_ptr, inv_diag_stage_ptr, reduced_u64_ptr,
                    lu_gm, piv_gm, inv_diag_gm, M, lu_stage_p1, piv_stage, inv_diag_stage,
                    coefficients, col_vec, abs_col, pivot_row, row_tmp, scratch, reduced, col_offsets_u32,
                    m_p8_repeat);
            } else if (c_idx == 0u) {
                // Hide A_orig backup (on Cores 1..7) and RHS load (on Core 7) inside Core 0's c_idx = 0!
                if (sizeof(T) != 4 || n == 256u || sub_id >= 4u) {
                    AscendC::Adds(A_orig, M, 0.0f, n * m_stride_p8);
                }
                if (sub_id == 7u) {
                    load_rhs_b<T>(n, pad_n, k, b_offset, input_b, y, lu_stage, coefficients);
                    for (uint32_t t = 0; t < k; ++t) {
                        AscendC::Gather(s_hi[t * pad_n], y[t], y_col_offsets_u32, uint32_t(0), n);
                    }
                    AscendC::Adds(M[my_w], y[0], 0.0f, 8u, 128u, y_to_m_rep);
                    AscendC::Adds(M[128u * 40u + my_w], y[128u * 8u], 0.0f, 8u, static_cast<uint8_t>(pad_n - 128u), y_to_m_rep);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SyncAll<true>();
        }
    }

    const bool rebal_200 = (sizeof(T) == 4 && n == 200u);
    const uint32_t p4_col_start = (rebal_200 && sub_id < 4u) ? (sub_id * 26u) : my_col_start;
    const uint32_t p4_w = (rebal_200 && sub_id < 4u) ? 26u : my_w;

    if constexpr (sizeof(T) == 4) {
        if (rebal_200 && sub_id < 4u) {
            fence<AscendC::HardEvent::V_MTE2>();
            const AscendC::DataCopyExtParams a_p4_load{
                static_cast<uint16_t>(n),
                static_cast<uint32_t>(p4_w * sizeof(float)),
                static_cast<uint32_t>((n - p4_w) * sizeof(float)),
                static_cast<uint32_t>((m_stride_p8 - ((p4_w + 7u) & ~7u)) >> 3),
                0};
            const AscendC::DataCopyPadExtParams<float> no_pad{false, 0, 0, 0};
            AscendC::DataCopyPad(A_orig, input_a[batch * n * n + p4_col_start], a_p4_load, no_pad);
            fence<AscendC::HardEvent::MTE2_V>();
        } else if (sub_id == 0u) {
            AscendC::DataCopy(A_orig, input_a[batch * n * n], a_copy);
            fence<AscendC::HardEvent::MTE2_V>();
        }
    } else if (sub_id == 0u) {
        load_panel_a<T>(n, batch, 0u, my_w, my_w, m_stride_p8, input_a, A_orig, lu_stage);
    }
    if (sub_id == 0u && batch == 0u && residuals != nullptr && (c_ptr->flags & LSTSQ_SQUARE_RESIDUAL)) {
        fence<AscendC::HardEvent::V_MTE3>();
        const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(((B_CONST != 0u) ? (B_CONST * K_CONST) : (c_ptr->batch * k)) * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(output_r, unit_masks.template ReinterpretCast<T>(), r_copy);
    }

    if (sub_id == 7u) {
        factor_chunk_p8(
            my_col_start, my_w, 7u, n, pad_n, my_col_start, row_w, piv,
            col_vec_ptr, abs_col_ptr, inv_diag_ptr,
            piv_stage_ptr, inv_diag_stage_ptr, reduced_u64_ptr,
            lu_gm, piv_gm, inv_diag_gm, M, lu_panel7, piv_stage, inv_diag_stage,
            coefficients, col_vec, abs_col, pivot_row, row_tmp, scratch, reduced, col_offsets_u32,
            m_p8_repeat);

        AscendC::Adds(y[0], M[my_w], 0.0f, 8u, 128u, m_to_y_rep);
        AscendC::Adds(y[128u * 8u], M[128u * 40u + my_w], 0.0f, 8u, static_cast<uint8_t>(pad_n - 128u), m_to_y_rep);
        AscendC::Duplicate(lu_stage_pad, 0.0f, 64u);
        AscendC::Duplicate(coeff_pad, 0.0f, 64u);
        fence<AscendC::HardEvent::S_V>();
        AscendC::PipeBarrier<PIPE_V>();

        // Phase 3: Double-buffered back-substitution U * X0 = y across panels 7 .. 0
        // Panel 7 is already in lu_panel7; prefetch Panel 6 & inv_diag_vec while solving Panel 7!
        fence<AscendC::HardEvent::V_MTE2>();
        fence<AscendC::HardEvent::S_MTE2>();
        AscendC::DataCopy(lu_buf0, lu_gm[(32u + 5u * w_p) * pad_n], w_p * pad_n);
        AscendC::DataCopy(inv_diag_vec, inv_diag_gm, my_col_start);
        back_sub_panel_p8(my_col_start, my_w, pad_n, lu_panel7, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);
        for (int32_t p = 6; p >= 0; --p) {
            const uint32_t p_col_start = (p == 0) ? 0u : (32u + static_cast<uint32_t>(p - 1) * w_p);
            const uint32_t p_w = (p == 0) ? 32u : w_p;
            auto cur_buf = ((p & 1) == 0) ? lu_buf0 : lu_buf1;
            auto next_buf = ((p & 1) == 0) ? lu_buf1 : lu_buf0;
            fence<AscendC::HardEvent::MTE2_V>();
            if (p >= 1) {
                if (p <= 5) {
                    fence<AscendC::HardEvent::V_MTE2>();
                }
                const uint32_t prev_col_start = (p - 1 == 0) ? 0u : (32u + static_cast<uint32_t>(p - 2) * w_p);
                const uint32_t prev_w = (p - 1 == 0) ? 32u : w_p;
                AscendC::DataCopy(next_buf, lu_gm[prev_col_start * pad_n], prev_w * pad_n);
            }
            back_sub_panel_p8(p_col_start, p_w, pad_n, cur_buf, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);
        }
        // At end of Phase 3: Panel 0 is in lu_buf0, Panel 1 is in lu_buf1, Panel 7 is in lu_panel7.
        AscendC::Brcb(lu_stage, inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(x0, y, lu_stage, pad_n * 8u);
        fence<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(x0_gm, x0, pad_n * 8u);
    }

    auto split_mask_i32 = lu_col.template ReinterpretCast<int32_t>();
    auto split_mask_u16 = split_mask_i32.template ReinterpretCast<uint16_t>();
    if constexpr (N_CONST != 256u) {
        AscendC::Duplicate(split_mask_i32, static_cast<int32_t>(0xfffff000u), pad_n);
        if (sub_id < 7u) {
            AscendC::Duplicate(s_hi, 0.0f, 2048u);
        } else {
            AscendC::Duplicate(s_hi[1024], 0.0f, 1024u);
        }
        AscendC::Duplicate(s_lo, 0.0f, k * pad_n);
        AscendC::Gather(col_vec, A_orig[0], col_offsets_u32, uint32_t(0), pad_n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(abs_col.template ReinterpretCast<uint16_t>(),
                     col_vec.template ReinterpretCast<uint16_t>(),
                     split_mask_u16, pad_n * 2u);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(pivot_row, col_vec, abs_col, pad_n);
        AscendC::PipeBarrier<PIPE_V>();
        if (sub_id == 7u) {
            fence<AscendC::HardEvent::V_S>();
            fence<AscendC::HardEvent::MTE3_V>();
        }
        AscendC::SyncAll<true>();

        if (sub_id < 7u) {
            fence<AscendC::HardEvent::V_MTE2>();
            fence<AscendC::HardEvent::S_MTE2>();
            AscendC::DataCopy(x0[p4_col_start * 8u], x0_gm[p4_col_start * 8u], p4_w * 8u);
            fence<AscendC::HardEvent::MTE2_S>();
        }
    } else {
        if (sub_id < 7u || n != 256u) {
            AscendC::Duplicate(split_mask_i32, static_cast<int32_t>(0xfffff000u), pad_n);
            if (sub_id < 7u) {
                AscendC::Duplicate(s_hi, 0.0f, 2048u);
            } else {
                AscendC::Duplicate(s_hi[1024], 0.0f, 1024u);
            }
            AscendC::Duplicate(s_lo, 0.0f, k * pad_n);
            AscendC::Gather(col_vec, A_orig[0], col_offsets_u32, uint32_t(0), pad_n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(abs_col.template ReinterpretCast<uint16_t>(),
                         col_vec.template ReinterpretCast<uint16_t>(),
                         split_mask_u16, pad_n * 2u);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(pivot_row, col_vec, abs_col, pad_n);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (sub_id == 7u) {
            fence<AscendC::HardEvent::V_S>();
            fence<AscendC::HardEvent::MTE3_V>();
        }
        AscendC::SyncAll<true>();

        if (sub_id < 7u) {
            fence<AscendC::HardEvent::V_MTE2>();
            fence<AscendC::HardEvent::S_MTE2>();
            AscendC::DataCopy(x0[p4_col_start * 8u], x0_gm[p4_col_start * 8u], p4_w * 8u);
            fence<AscendC::HardEvent::MTE2_S>();
        } else if (n == 256u) {
            AscendC::Duplicate(split_mask_i32, static_cast<int32_t>(0xfffff000u), pad_n);
            AscendC::Duplicate(s_hi[1024], 0.0f, 1024u);
            AscendC::Duplicate(s_lo, 0.0f, k * pad_n);
            AscendC::Gather(col_vec, A_orig[0], col_offsets_u32, uint32_t(0), pad_n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(abs_col.template ReinterpretCast<uint16_t>(),
                         col_vec.template ReinterpretCast<uint16_t>(),
                         split_mask_u16, pad_n * 2u);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(pivot_row, col_vec, abs_col, pad_n);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Phase 4a: All 8 cores accumulate compensated residual over their p4_w <= 32 panel in A_orig
    auto col_vec1 = y_col_offsets.template ReinterpretCast<float>();
    for (uint32_t rhs_base = 0; rhs_base < k; rhs_base += 4u) {
        const uint32_t cur_k = (k - rhs_base < 4u) ? (k - rhs_base) : 4u;
        const uint32_t len = cur_k * pad_n;
        auto s_hi_chunk = s_hi[rhs_base * pad_n];
        auto s_lo_chunk = s_lo[rhs_base * pad_n];
        auto s_lo_acc2 = s_hi[1024];
        auto p_hi_alt_4 = s_lo[1056];

        if constexpr (K_CONST == 0u || K_CONST > 4u) {
            if (rhs_base > 0u) {
                AscendC::Gather(col_vec, A_orig[0], col_offsets_u32, uint32_t(0), pad_n);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::And(abs_col.template ReinterpretCast<uint16_t>(),
                             col_vec.template ReinterpretCast<uint16_t>(),
                             split_mask_u16, pad_n * 2u);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(pivot_row, col_vec, abs_col, pad_n);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        if ((K_CONST != 0u && K_CONST <= 4u) || k <= 4u) {
            for (uint32_t step = 0; step < p4_w; step += 2u) {
                const uint32_t col0 = p4_col_start + step;
                residual_step_k4_3wave(
                    step + 0u, p4_w, cur_k, pad_n, len, &x0_ptr[(col0 + 0u) * 8u + rhs_base],
                    A_orig, col_offsets_u32, split_mask_u16, col_vec, col_vec1,
                    s_hi_chunk, t_sum_4, s_lo_chunk, s_lo_acc2, p_hi_4, p_hi_alt_4, p_err_4,
                    abs_col, row_tmp, pivot_row, scratch, v_tmp_4, u_tmp_4);
                residual_step_k4_3wave(
                    step + 1u, p4_w, cur_k, pad_n, len, &x0_ptr[(col0 + 1u) * 8u + rhs_base],
                    A_orig, col_offsets_u32, split_mask_u16, col_vec1, col_vec,
                    t_sum_4, s_hi_chunk, s_lo_chunk, s_lo_acc2, p_hi_alt_4, p_hi_4, p_err_4,
                    row_tmp, abs_col, scratch, pivot_row, v_tmp_4, u_tmp_4);
            }
            AscendC::Sub(u_tmp_4, s_hi_chunk, v_tmp_4, len);
            AscendC::Sub(p_hi_alt_4, p_hi_alt_4, v_tmp_4, len);
            AscendC::Add(s_lo_chunk, s_lo_chunk, s_lo_acc2, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(u_tmp_4, t_sum_4, u_tmp_4, len);
            AscendC::Add(s_lo_chunk, s_lo_chunk, p_hi_alt_4, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(s_hi[len], s_lo_chunk, u_tmp_4, len);
            AscendC::PipeBarrier<PIPE_V>();
        } else if constexpr (K_CONST == 0u || K_CONST > 4u) {
            for (uint32_t step = 0; step < p4_w; ++step) {
                const uint32_t col = p4_col_start + step;
                float x_val[4], x_lo_val[4];
                if (step > 0u) {
                    AscendC::Add(s_lo_chunk, s_lo_chunk, v_tmp_4, len);
                }
                for (uint32_t t = 0; t < cur_k; ++t) {
                    const float x = -x0_ptr[col * 8u + rhs_base + t];
                    const float x_hi = bits_float(float_bits(x) & 0xfffff000u);
                    const float x_lo = x - x_hi;
                    x_val[t] = x;
                    x_lo_val[t] = x_lo;
                    AscendC::Muls(p_hi_4[t * pad_n], col_vec, x, pad_n);
                    AscendC::Muls(p_err_4[t * pad_n], abs_col, x_hi, pad_n);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (step > 0u) {
                    AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
                }
                AscendC::Sub(p_err_4, p_err_4, p_hi_4, len);
                if ((step & 1u) != 0u) {
                    AscendC::Add(s_hi_chunk, t_sum_4, p_hi_4, len);
                } else {
                    AscendC::Add(t_sum_4, s_hi_chunk, p_hi_4, len);
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t t = 0; t < cur_k; ++t) {
                    AscendC::Axpy(p_err_4[t * pad_n], abs_col, x_lo_val[t], pad_n);
                }
                if ((step & 1u) != 0u) {
                    AscendC::Sub(v_tmp_4, s_hi_chunk, t_sum_4, len);
                } else {
                    AscendC::Sub(v_tmp_4, t_sum_4, s_hi_chunk, len);
                }
                if (step + 1u < p4_w) {
                    AscendC::Gather(col_vec, A_orig[step + 1u], col_offsets_u32, uint32_t(0), pad_n);
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t t = 0; t < cur_k; ++t) {
                    AscendC::Axpy(p_err_4[t * pad_n], pivot_row, x_val[t], pad_n);
                }
                if ((step & 1u) != 0u) {
                    AscendC::Sub(u_tmp_4, s_hi_chunk, v_tmp_4, len);
                } else {
                    AscendC::Sub(u_tmp_4, t_sum_4, v_tmp_4, len);
                }
                AscendC::Sub(p_hi_4, p_hi_4, v_tmp_4, len);
                if (step + 1u < p4_w) {
                    AscendC::And(abs_col.template ReinterpretCast<uint16_t>(),
                                 col_vec.template ReinterpretCast<uint16_t>(),
                                 split_mask_u16, pad_n * 2u);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(v_tmp_4, p_err_4, p_hi_4, len);
                if ((step & 1u) != 0u) {
                    AscendC::Sub(u_tmp_4, t_sum_4, u_tmp_4, len);
                } else {
                    AscendC::Sub(u_tmp_4, s_hi_chunk, u_tmp_4, len);
                }
                if (step + 1u < p4_w) {
                    AscendC::Sub(pivot_row, col_vec, abs_col, pad_n);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Add(s_lo_chunk, s_lo_chunk, v_tmp_4, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // 2-Stage Parallel Tree TwoSum Reduction across the 8 cores:
    constexpr bool SMALL_TOTAL_CONST = (K_CONST != 0u && K_CONST <= 4u);
    const uint32_t total_len = k * pad_n;
    auto s_lo_red = (SMALL_TOTAL_CONST || total_len <= 1024u) ? s_hi[total_len] : s_lo;
    if (sub_id < 4u || sub_id == 6u) {
        fence<AscendC::HardEvent::V_MTE3>();
        if (SMALL_TOTAL_CONST || total_len <= 1024u) {
            AscendC::DataCopy(r_gm[sub_id * 2048u], s_hi, 2u * total_len);
        } else {
            AscendC::DataCopy(r_gm[sub_id * 4096u], s_hi, total_len);
            AscendC::DataCopy(r_gm[sub_id * 4096u + 2048u], s_lo, total_len);
        }
        fence<AscendC::HardEvent::MTE3_V>();
    }
    AscendC::SyncAll<true>();

    if (sub_id == 4u) {
        two_sum_combine_pair_from_gm<SMALL_TOTAL_CONST>(0u, 1u, total_len, r_gm, s_hi, s_lo_red, t_sum_4, lu_stage, A_orig, y);
        fence<AscendC::HardEvent::V_MTE3>();
        if (SMALL_TOTAL_CONST || total_len <= 1024u) {
            AscendC::DataCopy(r_gm[4u * 2048u], s_hi, 2u * total_len);
        } else {
            AscendC::DataCopy(r_gm[4u * 4096u], s_hi, total_len);
            AscendC::DataCopy(r_gm[4u * 4096u + 2048u], s_lo, total_len);
        }
        fence<AscendC::HardEvent::MTE3_V>();
    } else if (sub_id == 5u) {
        two_sum_combine_pair_from_gm<SMALL_TOTAL_CONST>(2u, 3u, total_len, r_gm, s_hi, s_lo_red, t_sum_4, lu_stage, A_orig, y);
        fence<AscendC::HardEvent::V_MTE3>();
        if (SMALL_TOTAL_CONST || total_len <= 1024u) {
            AscendC::DataCopy(r_gm[5u * 2048u], s_hi, 2u * total_len);
        } else {
            AscendC::DataCopy(r_gm[5u * 4096u], s_hi, total_len);
            AscendC::DataCopy(r_gm[5u * 4096u + 2048u], s_lo, total_len);
        }
        fence<AscendC::HardEvent::MTE3_V>();
    } else if (sub_id == 7u) {
        two_sum_combine_from_gm<SMALL_TOTAL_CONST>(6u, total_len, r_gm, s_hi, s_lo_red, t_sum_4, lu_stage, y);
        AscendC::Duplicate(lu_stage_pad, 0.0f, 64u);
        AscendC::Duplicate(coeff_pad, 0.0f, 64u);
    }
    AscendC::SyncAll<true>();

    if (sub_id == 7u) {
        two_sum_combine_pair_from_gm<SMALL_TOTAL_CONST>(4u, 5u, total_len, r_gm, s_hi, s_lo_red, t_sum_4, lu_stage, A_orig, y);

        AscendC::Add(s_hi, s_hi, s_lo_red, total_len);
        AscendC::Duplicate(y, 0.0f, pad_n * 8u);
        AscendC::PipeBarrier<PIPE_V>();
        if ((K_CONST != 0u && K_CONST <= 5u) || k <= 5u) {
            AscendC::Brcb(A_orig, s_hi, static_cast<uint8_t>(total_len >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t t = 0; t < k; ++t) {
                uint64_t lane_mask[2] = {0x0101010101010101ULL << t, 0ULL};
                AscendC::Adds(y, A_orig[t * pad_n * 8u], 0.0f, lane_mask, static_cast<uint8_t>(pad_n >> 3), {1, 1, 8, 8});
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else if constexpr (K_CONST == 0u || K_CONST > 5u) {
            for (uint32_t t = 0; t < k; ++t) {
                AscendC::Brcb(coefficients, s_hi[t * pad_n], static_cast<uint8_t>(pad_n >> 3), {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(y, coefficients, unit_masks[t * 8u], 64u, static_cast<uint8_t>(pad_n >> 3), lane_rep);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        // Phase 4b: Triple-buffered (lu_buf0, lu_buf1, lu_buf2 = A_orig) forward substitution L * y = P * R across panels 0 .. 7
        // At start: Panel 0 is in lu_buf0, Panel 1 is in lu_buf1, Panel 7 is in lu_panel7, A_orig is free!
        auto lu_buf2 = A_orig;
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_buf2, lu_gm[(32u + w_p) * pad_n], w_p * pad_n);
        fwd_sub_panel(0u, 32u, n, pad_n, piv, lu_buf0, y, row_tmp, coefficients, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_buf0, lu_gm[(32u + 2u * w_p) * pad_n], w_p * pad_n);
        fwd_sub_panel(32u, w_p, n, pad_n, piv, lu_buf1, y, row_tmp, coefficients, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::MTE2_V>();
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_buf1, lu_gm[(32u + 3u * w_p) * pad_n], w_p * pad_n);
        fwd_sub_panel(32u + w_p, w_p, n, pad_n, piv, lu_buf2, y, row_tmp, coefficients, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::MTE2_V>();
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_buf2, lu_gm[(32u + 4u * w_p) * pad_n], w_p * pad_n);
        fwd_sub_panel(32u + 2u * w_p, w_p, n, pad_n, piv, lu_buf0, y, row_tmp, coefficients, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::MTE2_V>();
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_buf0, lu_gm[(32u + 5u * w_p) * pad_n], w_p * pad_n);
        fwd_sub_panel(32u + 3u * w_p, w_p, n, pad_n, piv, lu_buf1, y, row_tmp, coefficients, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::MTE2_V>();
        fwd_sub_panel(32u + 4u * w_p, w_p, n, pad_n, piv, lu_buf2, y, row_tmp, coefficients, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::MTE2_V>();
        fwd_sub_panel(32u + 5u * w_p, w_p, n, pad_n, piv, lu_buf0, y, row_tmp, coefficients, lu_stage, y_repeat8);

        fwd_sub_panel(my_col_start, my_w, n, pad_n, piv, lu_panel7, y, row_tmp, coefficients, lu_stage, y_repeat8);

        // Phase 4c: 4-buffer back-substitution U * delta_X = y across panels 7 .. 0
        // At start: Panel 7 is in lu_panel7, Panel 6 is in lu_buf0, Panel 5 is in lu_buf2, Panel 4 is STILL in lu_buf1!
        back_sub_panel_p8(my_col_start, my_w, pad_n, lu_panel7, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_panel7, lu_gm[(32u + 2u * w_p) * pad_n], w_p * pad_n);
        back_sub_panel_p8(32u + 5u * w_p, w_p, pad_n, lu_buf0, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_buf0, lu_gm[(32u + w_p) * pad_n], w_p * pad_n);
        back_sub_panel_p8(32u + 4u * w_p, w_p, pad_n, lu_buf2, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_buf2, lu_gm[32u * pad_n], w_p * pad_n);
        back_sub_panel_p8(32u + 3u * w_p, w_p, pad_n, lu_buf1, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::MTE2_V>();
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_buf1, lu_gm[0], 32u * pad_n);
        back_sub_panel_p8(32u + 2u * w_p, w_p, pad_n, lu_panel7, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);

        back_sub_panel_p8(32u + w_p, w_p, pad_n, lu_buf0, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);
        back_sub_panel_p8(32u, w_p, pad_n, lu_buf2, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);

        fence<AscendC::HardEvent::MTE2_V>();
        back_sub_panel_p8(0u, 32u, pad_n, lu_buf1, y_pad, y, coeff_pad, coefficients, lu_stage_pad, lu_stage, y_repeat8);
        AscendC::Brcb(lu_stage, inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(y, y, lu_stage, pad_n * 8u);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(x0, x0, y, pad_n * 8u);
        AscendC::PipeBarrier<PIPE_V>();
    }

    if (sub_id == 7u) {
        if constexpr (sizeof(T) == 4) {
            fence<AscendC::HardEvent::V_MTE3>();
            const AscendC::DataCopyExtParams x_store{
                static_cast<uint16_t>(n),
                static_cast<uint32_t>(k * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyPad(output_x[batch * n * k], x0, x_store);
        } else {
            auto pack_offsets = lu_buf1.template ReinterpretCast<int32_t>();
            fence<AscendC::HardEvent::V_S>();
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t t = 0; t < k; ++t) {
                    pack_offsets.SetValue(i * k + t, static_cast<int32_t>((i * 8u + t) * sizeof(float)));
                }
            }
            fence<AscendC::HardEvent::S_V>();
            AscendC::Gather(x_out, x0, pack_offsets.template ReinterpretCast<uint32_t>(), uint32_t(0), n * k);
            AscendC::PipeBarrier<PIPE_V>();
            auto raw_x = lu_stage.template ReinterpretCast<T>();
            AscendC::Cast(raw_x, x_out, AscendC::RoundMode::CAST_RINT, (n * k + 7u) & ~7u);
            fence<AscendC::HardEvent::V_MTE3>();
            const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(n * k * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(output_x[batch * n * k], raw_x, x_store);
        }
    }
    return SUCCESS;
}

// 4-Core-Per-Batch Wavefront LU + Parallel Compensated Residual Solver
// for square systems with n == 200 or n == 256 and 1 <= k <= 8.
template <typename T>
__aicore__ __attribute__((noinline)) int solve_square_large_p4(
    const Config& c, GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals,
    GM_ADDR ws_base, uint64_t per_core) {
    const uint32_t n = static_cast<uint32_t>(c.n);
    const uint32_t k = static_cast<uint32_t>(c.k);
    const uint32_t pad_n = (n + 7u) & ~7u;
    const uint32_t block_idx = AscendC::GetBlockIdx();
    const uint32_t batch = block_idx >> 2;
    const uint32_t sub_id = block_idx & 3u;
    if (batch >= c.batch) return SUCCESS;

    // 4-panel partitioning: w_p = 56 for n = 200 (56, 56, 56, 32), w_p = 64 for n = 256 (64, 64, 64, 64)
    const uint32_t w_p = (((n + 3u) >> 2) + 7u) & ~7u;
    const uint32_t my_col_start = sub_id * w_p;
    const uint32_t my_w = (sub_id < 3u) ? w_p : (n - 3u * w_p);
    constexpr uint32_t m_stride_p4 = 72u; // 9 blocks (odd block stride, conflict-free)

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    pipe.InitBuffer(work_buf, 48896u * sizeof(float));

    auto base = work_buf.Get<float>();
    auto M = base[0];
    auto lu_buf = base[18432];
    auto x0 = base[34816];
    auto s_hi = base[36928];
    auto x_out = base[36928];
    auto t_sum_4 = base[38976];
    auto lu_stage_pad = base[39936];
    auto lu_stage = base[40000];
    auto y_pad = base[42048];
    auto y = base[42112];
    auto coeff_pad = base[44160];
    auto coefficients = base[44224];
    auto s_lo = base[44224];
    auto vec = base[46336];

    auto col_vec = vec[0 * 256];
    auto lu_col = vec[1 * 256];
    auto piv_stage = vec[1 * 256].template ReinterpretCast<int32_t>();
    auto inv_diag_stage = vec[1 * 256 + 64];
    auto abs_col = vec[2 * 256];
    auto pivot_row = vec[3 * 256];
    auto row_tmp = vec[4 * 256];
    auto inv_diag_vec = vec[5 * 256];
    auto scratch = vec[6 * 256];
    auto reduced = vec[7 * 256];
    auto unit_masks = vec[7 * 256 + 64];
    auto col_offsets = vec[8 * 256].template ReinterpretCast<int32_t>();
    auto y_col_offsets = vec[9 * 256].template ReinterpretCast<int32_t>();

    auto p_hi_4 = lu_stage[0];
    auto p_err_4 = lu_stage[1024];
    auto v_tmp_4 = y[0];
    auto u_tmp_4 = y[1024];

    AscendC::GlobalTensor<T> input_a, input_b, output_x, output_r;
    input_a.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A));
    input_b.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(B));
    output_x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(solution));
    if (residuals != nullptr) {
        output_r.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(residuals));
    }

    AscendC::GlobalTensor<float> lu_gm, inv_diag_gm, x0_gm, r_gm;
    AscendC::GlobalTensor<int32_t> piv_gm, global_flags_gm;
    global_flags_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ws_base));
    GM_ADDR batch_ws = ws_base + 16384u + static_cast<uint64_t>(batch) * (3u * per_core);
    __gm__ float* batch_ws_f32 = reinterpret_cast<__gm__ float*>(batch_ws);
    lu_gm.SetGlobalBuffer(batch_ws_f32);                              // 65,536 floats (256 KiB)
    piv_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(batch_ws_f32 + 65536u)); // 256 int32s
    inv_diag_gm.SetGlobalBuffer(batch_ws_f32 + 65536u + 256u);        // 256 floats
    x0_gm.SetGlobalBuffer(batch_ws_f32 + 66048u);                     // 2048 floats
    r_gm.SetGlobalBuffer(batch_ws_f32 + 68096u);                      // 3 * 4096 floats

    AscendC::CreateVecIndex(y_col_offsets, int32_t(0), pad_n);
    AscendC::Duplicate(abs_col, -1.0f, 256u);
    AscendC::Duplicate(unit_masks, 0.0f, 64u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(col_offsets, y_col_offsets, static_cast<int32_t>(m_stride_p4 * sizeof(float)), pad_n);
    AscendC::Muls(y_col_offsets, y_col_offsets, static_cast<int32_t>(8u * sizeof(float)), pad_n);
    AscendC::PipeBarrier<PIPE_V>();
    if (sub_id == 3u) {
        fence<AscendC::HardEvent::V_S>();
        for (uint32_t t = 0; t < 8u; ++t) {
            unit_masks.SetValue(t * 8u + t, 1.0f);
        }
        fence<AscendC::HardEvent::S_V>();
    }

    auto col_offsets_u32 = col_offsets.template ReinterpretCast<uint32_t>();
    auto y_col_offsets_u32 = y_col_offsets.template ReinterpretCast<uint32_t>();
    volatile __ubuf__ float* col_vec_ptr = reinterpret_cast<volatile __ubuf__ float*>(col_vec.GetPhyAddr());
    volatile __ubuf__ float* abs_col_ptr = reinterpret_cast<volatile __ubuf__ float*>(abs_col.GetPhyAddr());
    volatile __ubuf__ float* inv_diag_ptr = reinterpret_cast<volatile __ubuf__ float*>(inv_diag_vec.GetPhyAddr());
    volatile __ubuf__ int32_t* piv_stage_ptr = reinterpret_cast<volatile __ubuf__ int32_t*>(piv_stage.GetPhyAddr());
    volatile __ubuf__ float* inv_diag_stage_ptr = reinterpret_cast<volatile __ubuf__ float*>(inv_diag_stage.GetPhyAddr());
    __ubuf__ const float* x0_ptr = reinterpret_cast<__ubuf__ const float*>(x0.GetPhyAddr());
    volatile __ubuf__ const uint64_t* reduced_u64_ptr = reinterpret_cast<volatile __ubuf__ const uint64_t*>(reduced.GetPhyAddr());

    const AscendC::BinaryRepeatParams m_p4_repeat{1, 1, 0, static_cast<uint8_t>(m_stride_p4 >> 3), 0, 1};
    const AscendC::BinaryRepeatParams y_repeat8{1, 0, 1, 8, 0, 8};
    const AscendC::BinaryRepeatParams lane_rep{1, 1, 0, 8, 8, 0};
    uint16_t piv[256];

    const uint64_t b_offset = static_cast<uint64_t>(batch) * c.rhs_batch_stride;
    if (sub_id == 3u) {
        load_rhs_b<T>(n, pad_n, k, b_offset, input_b, y, lu_stage, coefficients);
        for (uint32_t t = 0; t < k; ++t) {
            AscendC::Gather(s_hi[t * pad_n], y[t], y_col_offsets_u32, uint32_t(0), n);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    load_panel_a<T>(n, batch, my_col_start, my_w, my_w, m_stride_p4, input_a, M, lu_stage);

    // Phase 1: Wavefront LU across 4 panels of 8-column chunks
    const uint32_t num_chunks = n >> 3;
    const uint32_t c1 = w_p >> 3;
    const uint32_t c2 = c1 * 2u;
    const uint32_t c3 = c1 * 3u;

    for (uint32_t c_idx = 0; c_idx < num_chunks; ++c_idx) {
        const uint32_t owner = (c_idx < c1) ? 0u : ((c_idx < c2) ? 1u : ((c_idx < c3) ? 2u : 3u));
        if (sub_id == owner) {
            auto dst_lu = (sub_id == 3u) ? lu_buf[(c_idx - c3) * 8u * pad_n] : lu_stage;
            factor_chunk_p4(
                c_idx, sub_id, 3u, n, pad_n, my_col_start, my_w, m_stride_p4, piv,
                col_vec_ptr, abs_col_ptr, inv_diag_ptr, piv_stage_ptr, inv_diag_stage_ptr, reduced_u64_ptr,
                lu_gm, piv_gm, inv_diag_gm, M, y, dst_lu, piv_stage, inv_diag_stage,
                coefficients, col_vec, abs_col, pivot_row, row_tmp, scratch, reduced, col_offsets_u32,
                m_p4_repeat, y_repeat8);
        }
        if (owner < 3u) {
            AscendC::SyncAll<true>();
            if (sub_id > owner) {
                apply_chunk_p4(
                    c_idx, sub_id, 3u, n, pad_n, my_w, m_stride_p4, piv, piv_stage_ptr,
                    lu_gm, piv_gm, M, y, lu_stage, piv_stage, coefficients, pivot_row, row_tmp, scratch,
                    m_p4_repeat, y_repeat8);
            }
        }
    }

    if (sub_id < 3u) {
        fence<AscendC::HardEvent::V_MTE2>();
        load_panel_a<T>(n, batch, my_col_start, my_w, my_w, m_stride_p4, input_a, M, lu_stage);
    } else {
        fence<AscendC::HardEvent::V_MTE2>();
        fence<AscendC::HardEvent::S_MTE2>();
        AscendC::DataCopy(inv_diag_vec, inv_diag_gm, 3u * w_p);
        load_panel_a<T>(n, batch, my_col_start, my_w, my_w, m_stride_p4, input_a, M, lu_stage);
        fence<AscendC::HardEvent::MTE2_V>();
        fence<AscendC::HardEvent::S_V>();

        // Phase 3: Back-substitution U * X0 = y across panels 3, 2, 1, 0
        // Note: lu_buf ALREADY holds Panel 3 (p == 3) from Phase 1!
        for (int32_t p = 3; p >= 0; --p) {
            const uint32_t p_col_start = static_cast<uint32_t>(p) * w_p;
            const uint32_t p_width = (p < 3) ? w_p : (n - 3u * w_p);
            if (p < 3) {
                fence<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopy(lu_buf, lu_gm[p_col_start * pad_n], p_width * pad_n);
                fence<AscendC::HardEvent::MTE2_V>();
            }
            back_sub_panel(p_col_start, p_width, pad_n, lu_buf, y, coefficients, lu_stage, y_repeat8);
        }
        AscendC::Brcb(lu_stage, inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(x0, y, lu_stage, pad_n * 8u);
        fence<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(x0_gm, x0, pad_n * 8u);
        fence<AscendC::HardEvent::MTE3_V>();
    }
    AscendC::SyncAll<true>();

    // Phase 4: 4-Core Parallel Compensated Residual Refinement
    auto split_mask_i32 = lu_col.template ReinterpretCast<int32_t>();
    AscendC::Duplicate(split_mask_i32, static_cast<int32_t>(0xfffff000u), pad_n);
    AscendC::PipeBarrier<PIPE_V>();
    auto split_mask_u16 = split_mask_i32.template ReinterpretCast<uint16_t>();
    const uint32_t max_refine_iters = (n > 200u) ? 2u : 1u;
    int32_t my_batch_need_more = 1;

    for (uint32_t refine_iter = 0; refine_iter < max_refine_iters; ++refine_iter) {
        if (my_batch_need_more == 0) {
            AscendC::SyncAll<true>();
            continue;
        }
        if (sub_id < 3u) {
            fence<AscendC::HardEvent::V_MTE2>();
            fence<AscendC::HardEvent::S_MTE2>();
            AscendC::DataCopy(x0, x0_gm, pad_n * 8u);
            AscendC::Duplicate(s_hi, 0.0f, k * pad_n);
            AscendC::Duplicate(s_lo, 0.0f, k * pad_n);
            AscendC::PipeBarrier<PIPE_V>();
            fence<AscendC::HardEvent::MTE2_S>();
        } else {
            if (refine_iter > 0u) {
                fence<AscendC::HardEvent::V_MTE2>();
                load_rhs_b<T>(n, pad_n, k, b_offset, input_b, y, lu_stage, coefficients);
                for (uint32_t t = 0; t < k; ++t) {
                    AscendC::Gather(s_hi[t * pad_n], y[t], y_col_offsets_u32, uint32_t(0), n);
                }
            }
            AscendC::Duplicate(s_lo, 0.0f, k * pad_n);
            AscendC::PipeBarrier<PIPE_V>();
            fence<AscendC::HardEvent::V_S>();
        }

        // Phase 4a: All 4 cores accumulate compensated residual over their my_w <= 64 panel in M
        for (uint32_t rhs_base = 0; rhs_base < k; rhs_base += 4u) {
            const uint32_t cur_k = (k - rhs_base < 4u) ? (k - rhs_base) : 4u;
            const uint32_t len = cur_k * pad_n;
            auto s_hi_chunk = s_hi[rhs_base * pad_n];
            auto s_lo_chunk = s_lo[rhs_base * pad_n];
            auto d1_tmp_4 = s_hi[1024];
            auto p_err_alt_4 = s_lo[1024];

            AscendC::Gather(col_vec, M[0], col_offsets_u32, uint32_t(0), pad_n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::And(abs_col.template ReinterpretCast<uint16_t>(),
                         col_vec.template ReinterpretCast<uint16_t>(),
                         split_mask_u16, pad_n * 2u);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(pivot_row, col_vec, abs_col, pad_n);
            AscendC::PipeBarrier<PIPE_V>();

            if (k <= 4u) {
                for (uint32_t step = 0; step < my_w; step += 2u) {
                    const uint32_t col0 = my_col_start + step;
                    residual_step_k4(
                        step, my_w, cur_k, pad_n, len, &x0_ptr[col0 * 8u + rhs_base],
                        M, col_offsets_u32, split_mask_u16, col_vec,
                        s_hi_chunk, t_sum_4, s_lo_chunk, p_hi_4, p_err_4, p_err_alt_4,
                        abs_col, row_tmp, pivot_row, scratch, v_tmp_4, u_tmp_4, d1_tmp_4);
                    residual_step_k4(
                        step + 1u, my_w, cur_k, pad_n, len, &x0_ptr[(col0 + 1u) * 8u + rhs_base],
                        M, col_offsets_u32, split_mask_u16, col_vec,
                        t_sum_4, s_hi_chunk, s_lo_chunk, p_hi_4, p_err_alt_4, p_err_4,
                        row_tmp, abs_col, scratch, pivot_row, v_tmp_4, u_tmp_4, d1_tmp_4);
                }
                // Since my_w (56, 32, 64) is always even, last step is odd:
                AscendC::Sub(u_tmp_4, t_sum_4, u_tmp_4, len);
                AscendC::Add(d1_tmp_4, d1_tmp_4, p_err_alt_4, len);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(s_lo_chunk, s_lo_chunk, d1_tmp_4, len);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                for (uint32_t step = 0; step < my_w; ++step) {
                    const uint32_t col = my_col_start + step;
                    float x_val[4], x_lo_val[4];
                    if (step > 0u) {
                        AscendC::Add(s_lo_chunk, s_lo_chunk, v_tmp_4, len);
                    }
                    for (uint32_t t = 0; t < cur_k; ++t) {
                        const float x = -x0_ptr[col * 8u + rhs_base + t];
                        const float x_hi = bits_float(float_bits(x) & 0xfffff000u);
                        const float x_lo = x - x_hi;
                        x_val[t] = x;
                        x_lo_val[t] = x_lo;
                        AscendC::Muls(p_hi_4[t * pad_n], col_vec, x, pad_n);
                        AscendC::Muls(p_err_4[t * pad_n], abs_col, x_hi, pad_n);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    if (step > 0u) {
                        AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
                    }
                    AscendC::Sub(p_err_4, p_err_4, p_hi_4, len);
                    if ((step & 1u) != 0u) {
                        AscendC::Add(s_hi_chunk, t_sum_4, p_hi_4, len);
                    } else {
                        AscendC::Add(t_sum_4, s_hi_chunk, p_hi_4, len);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t t = 0; t < cur_k; ++t) {
                        AscendC::Axpy(p_err_4[t * pad_n], abs_col, x_lo_val[t], pad_n);
                    }
                    if ((step & 1u) != 0u) {
                        AscendC::Sub(v_tmp_4, s_hi_chunk, t_sum_4, len);
                    } else {
                        AscendC::Sub(v_tmp_4, t_sum_4, s_hi_chunk, len);
                    }
                    if (step + 1u < my_w) {
                        AscendC::Gather(col_vec, M[step + 1u], col_offsets_u32, uint32_t(0), pad_n);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t t = 0; t < cur_k; ++t) {
                        AscendC::Axpy(p_err_4[t * pad_n], pivot_row, x_val[t], pad_n);
                    }
                    if ((step & 1u) != 0u) {
                        AscendC::Sub(u_tmp_4, s_hi_chunk, v_tmp_4, len);
                    } else {
                        AscendC::Sub(u_tmp_4, t_sum_4, v_tmp_4, len);
                    }
                    AscendC::Sub(p_hi_4, p_hi_4, v_tmp_4, len);
                    if (step + 1u < my_w) {
                        AscendC::And(abs_col.template ReinterpretCast<uint16_t>(),
                                     col_vec.template ReinterpretCast<uint16_t>(),
                                     split_mask_u16, pad_n * 2u);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(v_tmp_4, p_err_4, p_hi_4, len);
                    if ((step & 1u) != 0u) {
                        AscendC::Sub(u_tmp_4, t_sum_4, u_tmp_4, len);
                    } else {
                        AscendC::Sub(u_tmp_4, s_hi_chunk, u_tmp_4, len);
                    }
                    if (step + 1u < my_w) {
                        AscendC::Sub(pivot_row, col_vec, abs_col, pad_n);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Add(s_lo_chunk, s_lo_chunk, v_tmp_4, len);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        if (sub_id < 3u) {
            fence<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(r_gm[sub_id * 4096u], s_hi, k * pad_n);
            AscendC::DataCopy(r_gm[sub_id * 4096u + 2048u], s_lo, k * pad_n);
            fence<AscendC::HardEvent::MTE3_V>();
        }
        AscendC::SyncAll<true>();

        if (sub_id == 3u) {
            // Combine partial (s_hi, s_lo) from Cores 0, 1, 2 via vectorized Knuth TwoSum
            // without touching lu_buf (which still holds Panel 0 of lu_gm!):
            const uint32_t total_len = k * pad_n;
            for (uint32_t sub = 0; sub < 3u; ++sub) {
                for (uint32_t offset = 0; offset < total_len; offset += 1024u) {
                    const uint32_t chunk_len = (total_len - offset < 1024u) ? (total_len - offset) : 1024u;
                    auto other_hi = lu_stage[0];
                    auto other_lo = lu_stage[1024];
                    auto d_tmp = y[0];
                    auto u_tmp = y[1024];
                    fence<AscendC::HardEvent::V_MTE2>();
                    AscendC::DataCopy(other_hi, r_gm[sub * 4096u + offset], chunk_len);
                    AscendC::DataCopy(other_lo, r_gm[sub * 4096u + 2048u + offset], chunk_len);
                    fence<AscendC::HardEvent::MTE2_V>();
                    AscendC::Add(s_lo[offset], s_lo[offset], other_lo, chunk_len);
                    AscendC::Add(t_sum_4, s_hi[offset], other_hi, chunk_len);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(d_tmp, t_sum_4, s_hi[offset], chunk_len);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(u_tmp, t_sum_4, d_tmp, chunk_len);
                    AscendC::Sub(other_hi, other_hi, d_tmp, chunk_len);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(u_tmp, s_hi[offset], u_tmp, chunk_len);
                    AscendC::Add(s_lo[offset], s_lo[offset], other_hi, chunk_len);
                    AscendC::Adds(s_hi[offset], t_sum_4, 0.0f, chunk_len);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(s_lo[offset], s_lo[offset], u_tmp, chunk_len);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            AscendC::Add(s_hi, s_hi, s_lo, total_len);
            AscendC::Duplicate(y, 0.0f, (pad_n + 8u) * 8u);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t t = 0; t < k; ++t) {
                AscendC::Brcb(coefficients, s_hi[t * pad_n], static_cast<uint8_t>(pad_n >> 3), {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(y, coefficients, unit_masks[t * 8u], 64u, static_cast<uint8_t>(pad_n >> 3), lane_rep);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Duplicate(lu_stage[pad_n * 8u], 0.0f, 64u);
            AscendC::Duplicate(coefficients[pad_n * 8u], 0.0f, 64u);
            AscendC::PipeBarrier<PIPE_V>();

            // Phase 4b: Forward substitution L * y = P * R across panels 0, 1, 2, 3
            // Note: lu_buf ALREADY holds Panel 0 (p == 0)!
            for (uint32_t p = 0; p < 4u; ++p) {
                const uint32_t p_col_start = p * w_p;
                const uint32_t p_width = (p < 3u) ? w_p : (n - 3u * w_p);
                if (p > 0u) {
                    fence<AscendC::HardEvent::V_MTE2>();
                    AscendC::DataCopy(lu_buf, lu_gm[p_col_start * pad_n], p_width * pad_n);
                    fence<AscendC::HardEvent::MTE2_V>();
                }
                fwd_sub_panel(p_col_start, p_width, n, pad_n, piv, lu_buf, y, row_tmp,
                              coefficients, lu_stage, y_repeat8);
            }

            // Phase 4c: Back-substitution U * delta_X = y across panels 3, 2, 1, 0
            // Note: lu_buf ALREADY holds Panel 3 (p == 3)!
            for (int32_t p = 3; p >= 0; --p) {
                const uint32_t p_col_start = static_cast<uint32_t>(p) * w_p;
                const uint32_t p_width = (p < 3) ? w_p : (n - 3u * w_p);
                if (p < 3) {
                    fence<AscendC::HardEvent::V_MTE2>();
                    AscendC::DataCopy(lu_buf, lu_gm[p_col_start * pad_n], p_width * pad_n);
                    fence<AscendC::HardEvent::MTE2_V>();
                }
                back_sub_panel(p_col_start, p_width, pad_n, lu_buf, y, coefficients, lu_stage, y_repeat8);
            }
            AscendC::Brcb(lu_stage, inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(y, y, lu_stage, pad_n * 8u);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(x0, x0, y, pad_n * 8u);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (refine_iter + 1u < max_refine_iters) {
            if (sub_id == 3u) {
                AscendC::Abs(lu_stage, y, pad_n * 8u);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceMax(reduced, lu_stage, scratch, pad_n * 8u, false);
                fence<AscendC::HardEvent::V_S>();
                const float max_delta = reduced.GetValue(0);
                const int32_t need_more_flag = (max_delta > 3.5e-3f) ? 1 : 0;
                piv_stage_ptr[0] = need_more_flag;
                fence<AscendC::HardEvent::S_MTE3>();
                AscendC::DataCopy(global_flags_gm[batch * 8u], piv_stage, 8u);
                fence<AscendC::HardEvent::MTE3_S>();
                if (need_more_flag != 0) {
                    fence<AscendC::HardEvent::V_MTE3>();
                    AscendC::DataCopy(x0_gm, x0, pad_n * 8u);
                    fence<AscendC::HardEvent::MTE3_V>();
                }
            }
            AscendC::SyncAll<true>();
            fence<AscendC::HardEvent::S_MTE2>();
            AscendC::DataCopy(piv_stage, global_flags_gm, 32u);
            fence<AscendC::HardEvent::MTE2_S>();
            int32_t any_need_more = 0;
            for (uint32_t b = 0; b < c.batch; ++b) {
                any_need_more |= piv_stage_ptr[b * 8u];
            }
            my_batch_need_more = piv_stage_ptr[batch * 8u];
            if (any_need_more == 0) {
                break;
            }
        }
    }

    if (sub_id == 3u) {
        if constexpr (sizeof(T) == 4) {
            fence<AscendC::HardEvent::V_MTE3>();
            const AscendC::DataCopyExtParams x_store{
                static_cast<uint16_t>(n),
                static_cast<uint32_t>(k * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyPad(output_x[batch * n * k], x0, x_store);
        } else {
            auto pack_offsets = lu_buf.template ReinterpretCast<int32_t>();
            fence<AscendC::HardEvent::V_S>();
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t t = 0; t < k; ++t) {
                    pack_offsets.SetValue(i * k + t, static_cast<int32_t>((i * 8u + t) * sizeof(float)));
                }
            }
            fence<AscendC::HardEvent::S_V>();
            AscendC::Gather(x_out, x0, pack_offsets.template ReinterpretCast<uint32_t>(), uint32_t(0), n * k);
            AscendC::PipeBarrier<PIPE_V>();
            auto raw_x = lu_stage.template ReinterpretCast<T>();
            AscendC::Cast(raw_x, x_out, AscendC::RoundMode::CAST_RINT, (n * k + 7u) & ~7u);
            fence<AscendC::HardEvent::V_MTE3>();
            const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(n * k * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(output_x[batch * n * k], raw_x, x_store);
        }
        if (sub_id == 3u && (c.flags & LSTSQ_SQUARE_RESIDUAL) && residuals != nullptr && batch == 0u) {
            const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(c.batch * k * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(output_r, coefficients[pad_n * 8u].template ReinterpretCast<T>(), r_copy);
        }
    }
    return SUCCESS;
}

// Fast Conflict-Free Row-Major Vector LU + Vectorized Compensated Refinement
// for square systems with 64 < m == n <= 256 and 1 <= k <= 8 (1-core-per-batch fallback).
template <typename T>
__aicore__ __attribute__((noinline)) int solve_square_large(
    const Config& c, GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals, GM_ADDR user_workspace) {
    const uint32_t n = static_cast<uint32_t>(c.n);
    const uint32_t k = static_cast<uint32_t>(c.k);
    const uint32_t pad_n = (n + 7u) & ~7u;

    const uint32_t w0_lu = (n <= 200u) ? n : 128u;
    const uint32_t w0_lu_pad = (w0_lu + 7u) & ~7u;
    const uint32_t w1_lu = (n > w0_lu) ? (n - w0_lu) : 0u;
    const uint32_t w1_lu_pad = (w1_lu + 7u) & ~7u;
    const uint32_t m_stride_lu = ((w0_lu_pad >> 3) | 1u) << 3;

    const uint32_t w0 = (n < 128u) ? n : 128u;
    const uint32_t w0_pad = (w0 + 7u) & ~7u;
    const uint32_t w1 = (n > 128u) ? (n - 128u) : 0u;
    const uint32_t w1_pad = (w1 + 7u) & ~7u;
    constexpr uint32_t m_stride_ref = 136u;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    pipe.InitBuffer(work_buf, 48896u * sizeof(float));

    auto base = work_buf.Get<float>();
    auto M = base[0];
    auto x0 = base[34816];
    auto s_hi = base[36928];
    auto x_out = base[36928];
    auto t_sum_4 = base[38976];
    auto lu_stage_pad = base[39936];
    auto lu_stage = base[40000];
    auto y_pad = base[42048];
    auto y = base[42112];
    auto coeff_pad = base[44160];
    auto coefficients = base[44224];
    auto s_lo = base[44224];
    auto vec = base[46336];

    auto col_vec = vec[0 * 256];
    auto lu_col = vec[1 * 256];
    auto abs_col = vec[2 * 256];
    auto pivot_row = vec[3 * 256];
    auto row_tmp = vec[4 * 256];
    auto inv_diag_vec = vec[5 * 256];
    auto scratch = vec[6 * 256];
    auto reduced = vec[7 * 256];
    auto unit_masks = vec[7 * 256 + 64];
    auto col_offsets = vec[8 * 256].template ReinterpretCast<int32_t>();
    auto y_col_offsets = vec[9 * 256].template ReinterpretCast<int32_t>();

    auto p_hi_4 = lu_stage[0];
    auto p_err_4 = lu_stage[1024];
    auto v_tmp_4 = y[0];
    auto u_tmp_4 = y[1024];

    AscendC::GlobalTensor<T> input_a, input_b, output_x, output_r;
    AscendC::GlobalTensor<float> lu_gm;
    input_a.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A));
    input_b.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(B));
    output_x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(solution));
    if (residuals != nullptr) {
        output_r.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(residuals));
    }
    lu_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(user_workspace));

    AscendC::CreateVecIndex(y_col_offsets, int32_t(0), pad_n);
    AscendC::Duplicate(lu_stage[pad_n * 8u], 0.0f, 64u);
    AscendC::Duplicate(coefficients[pad_n * 8u], 0.0f, 64u);
    AscendC::Duplicate(unit_masks, 0.0f, 64u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(col_offsets, y_col_offsets, static_cast<int32_t>(m_stride_lu * sizeof(float)), pad_n);
    AscendC::Muls(y_col_offsets, y_col_offsets, static_cast<int32_t>(8u * sizeof(float)), pad_n);
    AscendC::PipeBarrier<PIPE_V>();
    fence<AscendC::HardEvent::V_S>();
    for (uint32_t t = 0; t < 8u; ++t) {
        unit_masks.SetValue(t * 8u + t, 1.0f);
    }
    fence<AscendC::HardEvent::S_V>();

    if ((c.flags & LSTSQ_SQUARE_RESIDUAL) && residuals != nullptr && AscendC::GetBlockIdx() == 0) {
        AscendC::Duplicate(lu_stage, 0.0f, 256u);
        fence<AscendC::HardEvent::V_MTE3>();
        const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(c.batch * k * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(output_r, lu_stage.template ReinterpretCast<T>(), r_copy);
        fence<AscendC::HardEvent::MTE3_V>();
    }

    auto col_offsets_u32 = col_offsets.template ReinterpretCast<uint32_t>();
    auto y_col_offsets_u32 = y_col_offsets.template ReinterpretCast<uint32_t>();
    __ubuf__ float* col_vec_ptr = reinterpret_cast<__ubuf__ float*>(col_vec.GetPhyAddr());
    __ubuf__ float* abs_col_ptr = reinterpret_cast<__ubuf__ float*>(abs_col.GetPhyAddr());
    __ubuf__ float* inv_diag_ptr = reinterpret_cast<__ubuf__ float*>(inv_diag_vec.GetPhyAddr());
    __ubuf__ const float* x0_ptr = reinterpret_cast<__ubuf__ const float*>(x0.GetPhyAddr());
    __ubuf__ const uint64_t* reduced_u64_ptr = reinterpret_cast<__ubuf__ const uint64_t*>(reduced.GetPhyAddr());

    const AscendC::BinaryRepeatParams m_lu_repeat{1, 1, 0, static_cast<uint8_t>(m_stride_lu >> 3), 0, 1};
    const AscendC::BinaryRepeatParams y_repeat8{1, 0, 1, 8, 0, 8};
    const AscendC::BinaryRepeatParams lane_rep{1, 1, 0, 8, 8, 0};
    uint16_t piv[256];

    for (uint64_t batch = AscendC::GetBlockIdx(); batch < c.batch; batch += AscendC::GetBlockNum()) {
        if (batch != AscendC::GetBlockIdx() && m_stride_lu != m_stride_ref) {
            AscendC::CreateVecIndex(col_offsets, int32_t(0), pad_n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(col_offsets, col_offsets, static_cast<int32_t>(m_stride_lu * sizeof(float)), pad_n);
            AscendC::PipeBarrier<PIPE_V>();
        }

        const uint64_t b_offset = batch * c.rhs_batch_stride;
        load_rhs_b<T>(n, pad_n, k, b_offset, input_b, y, lu_stage, coefficients);
        AscendC::Duplicate(abs_col, -1.0f, pad_n);
        AscendC::Duplicate(coefficients[pad_n * 8u], 0.0f, 64u);
        AscendC::PipeBarrier<PIPE_V>();

        // Phase 1: Factor Panel 0
        load_panel_a<T>(n, batch, 0u, w0_lu, w0_lu_pad, m_stride_lu, input_a, M, lu_stage);
        for (uint32_t j = 0; j < w0_lu; ++j) {
            AscendC::Gather(col_vec, M[j], col_offsets_u32, uint32_t(0), n);
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t j0 = j & ~7u;
            const uint32_t rem_a = pad_n - j0;
            const uint32_t chunk0 = rem_a < 64u ? rem_a : 64u;
            const uint64_t c_mask = (chunk0 == 64u) ? ~0ULL : ((1ULL << chunk0) - 1ULL);
            uint64_t mask[2] = {c_mask & (~0ULL << (j - j0)), 0ULL};
            AscendC::Abs(abs_col[j0], col_vec[j0], mask, 1u, {1, 1, 8, 8});
            if (rem_a > 64u) {
                AscendC::Abs(abs_col[j0 + 64u], col_vec[j0 + 64u], rem_a - 64u);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax(reduced, abs_col[j0], scratch, n - j0, true);
            fence<AscendC::HardEvent::V_S>();

            const uint64_t red_val = reduced_u64_ptr[0];
            const uint32_t max_bits = static_cast<uint32_t>(red_val);
            const uint32_t pivot = j0 + static_cast<uint32_t>(red_val >> 32);
            if (max_bits == 0u) return SINGULAR;
            if (max_bits >= 0x7f800000u || pivot >= n) return NUMERIC_ERROR;
            piv[j] = static_cast<uint16_t>(pivot);

            const uint32_t slot = j & 7u;
            auto lu_col_slot = lu_stage[slot * pad_n];
            const bool has_rem_cols = (j + 1u < w0_lu);
            const uint32_t c0 = (j + 1u) & ~7u;
            const uint32_t rem = has_rem_cols ? (w0_lu_pad - c0) : 0u;

            if (has_rem_cols) {
                AscendC::Adds(pivot_row, M[pivot * m_stride_lu + c0], 0.0f, rem);
            }
            if (pivot != j) {
                if (has_rem_cols) {
                    AscendC::Adds(row_tmp, M[j * m_stride_lu + c0], 0.0f, rem);
                }
                AscendC::Adds(scratch[0], y[pivot * 8u], 0.0f, 8u);
                AscendC::Adds(scratch[8], y[j * 8u], 0.0f, 8u);
            }

            const float diag = col_vec_ptr[pivot];
            const float inv_diag = 1.0f / diag;
            if (pivot != j) {
                col_vec_ptr[pivot] = col_vec_ptr[j];
            }
            col_vec_ptr[j] = 0.0f;
            abs_col_ptr[j] = -1.0f;
            inv_diag_ptr[j] = inv_diag;
            fence<AscendC::HardEvent::S_V>();

            AscendC::Muls(lu_col_slot, col_vec, -inv_diag, pad_n);
            AscendC::PipeBarrier<PIPE_V>();

            if (j + 1u < n) {
                AscendC::Brcb(coefficients[j0 * 8u], lu_col_slot[j0], static_cast<uint8_t>((pad_n - j0) >> 3), {1, 8});
            }
            if (pivot != j) {
                if (has_rem_cols) {
                    AscendC::Adds(M[pivot * m_stride_lu + c0], row_tmp, 0.0f, rem);
                    AscendC::Adds(M[j * m_stride_lu + c0], pivot_row, 0.0f, rem);
                }
                AscendC::Adds(y[j * 8u], scratch[0], 0.0f, 8u);
                AscendC::Adds(y[pivot * 8u], scratch[8], 0.0f, 8u);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (j + 1u < n) {
                const uint32_t sub_rows = n - 1u - j;
                for (uint32_t offset = 0; offset < rem; offset += 64u) {
                    const uint64_t lanes = rem - offset < 64u ? rem - offset : 64u;
                    AscendC::MulAddDst(M[(j + 1u) * m_stride_lu + c0 + offset],
                                       pivot_row[offset],
                                       coefficients[(j + 1u) * 8u],
                                       lanes, static_cast<uint8_t>(sub_rows), m_lu_repeat);
                }
                AscendC::MulAddDst(y[(j + 1u) * 8u], y[j * 8u], coefficients[(j + 1u) * 8u],
                                   64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), y_repeat8);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if (slot == 7u || j + 1u == w0_lu) {
                fence<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopy(lu_gm[(j - slot) * pad_n], lu_stage, (slot + 1u) * pad_n);
                fence<AscendC::HardEvent::MTE3_V>();
            }
        }
        fence<AscendC::HardEvent::MTE3_MTE2>();

        // Phase 2: Factor Panel 1 if w1_lu > 0
        if (w1_lu > 0u) {
            load_panel_a<T>(n, batch, 128u, w1_lu, w1_lu_pad, m_stride_lu, input_a, M, lu_stage);
            for (uint32_t chunk = 0; chunk < 128u; chunk += 8u) {
                fence<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopy(lu_stage, lu_gm[chunk * pad_n], 8u * pad_n);
                fence<AscendC::HardEvent::MTE2_V>();
                for (uint32_t s = 0; s < 8u; ++s) {
                    const uint32_t j = chunk + s;
                    const uint32_t j0 = j & ~7u;
                    const uint32_t pivot = piv[j];
                    AscendC::Adds(pivot_row, M[pivot * m_stride_lu], 0.0f, w1_lu_pad);
                    if (pivot != j) {
                        AscendC::Adds(row_tmp, M[j * m_stride_lu], 0.0f, w1_lu_pad);
                    }
                    AscendC::Brcb(coefficients[j0 * 8u], lu_stage[s * pad_n + j0],
                                  static_cast<uint8_t>((pad_n - j0) >> 3), {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    if (pivot != j) {
                        AscendC::Adds(M[pivot * m_stride_lu], row_tmp, 0.0f, w1_lu_pad);
                        AscendC::Adds(M[j * m_stride_lu], pivot_row, 0.0f, w1_lu_pad);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    const uint32_t sub_rows = n - 1u - j;
                    if (w1_lu_pad <= 64u) {
                        AscendC::MulAddDst(M[(j + 1u) * m_stride_lu],
                                           pivot_row[0],
                                           coefficients[(j + 1u) * 8u],
                                           w1_lu_pad, static_cast<uint8_t>(sub_rows), m_lu_repeat);
                    } else {
                        AscendC::MulAddDst(M[(j + 1u) * m_stride_lu],
                                           pivot_row[0],
                                           coefficients[(j + 1u) * 8u],
                                           64u, static_cast<uint8_t>(sub_rows), m_lu_repeat);
                        AscendC::MulAddDst(M[(j + 1u) * m_stride_lu + 64u],
                                           pivot_row[64],
                                           coefficients[(j + 1u) * 8u],
                                           w1_lu_pad - 64u, static_cast<uint8_t>(sub_rows), m_lu_repeat);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
            for (uint32_t step = 0; step < w1_lu; ++step) {
                const uint32_t j = 128u + step;
                AscendC::Gather(col_vec, M[step], col_offsets_u32, uint32_t(0), n);
                AscendC::PipeBarrier<PIPE_V>();
                const uint32_t j0 = j & ~7u;
                const uint32_t rem_a = pad_n - j0;
                const uint32_t chunk0 = rem_a < 64u ? rem_a : 64u;
                const uint64_t c_mask = (chunk0 == 64u) ? ~0ULL : ((1ULL << chunk0) - 1ULL);
                uint64_t mask[2] = {c_mask & (~0ULL << (j - j0)), 0ULL};
                AscendC::Abs(abs_col[j0], col_vec[j0], mask, 1u, {1, 1, 8, 8});
                if (rem_a > 64u) {
                    AscendC::Abs(abs_col[j0 + 64u], col_vec[j0 + 64u], rem_a - 64u);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceMax(reduced, abs_col[j0], scratch, n - j0, true);
                fence<AscendC::HardEvent::V_S>();

                const uint64_t red_val = reduced_u64_ptr[0];
                const uint32_t max_bits = static_cast<uint32_t>(red_val);
                const uint32_t pivot = j0 + static_cast<uint32_t>(red_val >> 32);
                if (max_bits == 0u) return SINGULAR;
                if (max_bits >= 0x7f800000u || pivot >= n) return NUMERIC_ERROR;
                piv[j] = static_cast<uint16_t>(pivot);

                const uint32_t slot = step & 7u;
                auto lu_col_slot = lu_stage[slot * pad_n];
                const bool has_rem_cols = (j + 1u < n);
                const uint32_t c0 = (step + 1u) & ~7u;
                const uint32_t rem = has_rem_cols ? (w1_lu_pad - c0) : 0u;

                if (has_rem_cols) {
                    AscendC::Adds(pivot_row, M[pivot * m_stride_lu + c0], 0.0f, rem);
                }
                if (pivot != j) {
                    if (has_rem_cols) {
                        AscendC::Adds(row_tmp, M[j * m_stride_lu + c0], 0.0f, rem);
                    }
                    AscendC::Adds(scratch[0], y[pivot * 8u], 0.0f, 8u);
                    AscendC::Adds(scratch[8], y[j * 8u], 0.0f, 8u);
                }

                const float diag = col_vec_ptr[pivot];
                const float inv_diag = 1.0f / diag;
                if (pivot != j) {
                    col_vec_ptr[pivot] = col_vec_ptr[j];
                }
                col_vec_ptr[j] = 0.0f;
                abs_col_ptr[j] = -1.0f;
                inv_diag_ptr[j] = inv_diag;
                fence<AscendC::HardEvent::S_V>();

                AscendC::Muls(lu_col_slot, col_vec, -inv_diag, pad_n);
                AscendC::PipeBarrier<PIPE_V>();

                if (has_rem_cols) {
                    AscendC::Brcb(coefficients[j0 * 8u], lu_col_slot[j0],
                                  static_cast<uint8_t>((pad_n - j0) >> 3), {1, 8});
                }
                if (pivot != j) {
                    if (has_rem_cols) {
                        AscendC::Adds(M[pivot * m_stride_lu + c0], row_tmp, 0.0f, rem);
                        AscendC::Adds(M[j * m_stride_lu + c0], pivot_row, 0.0f, rem);
                    }
                    AscendC::Adds(y[j * 8u], scratch[0], 0.0f, 8u);
                    AscendC::Adds(y[pivot * 8u], scratch[8], 0.0f, 8u);
                }
                AscendC::PipeBarrier<PIPE_V>();

                if (has_rem_cols) {
                    const uint32_t sub_rows = n - 1u - j;
                    if (rem <= 64u) {
                        AscendC::MulAddDst(M[(j + 1u) * m_stride_lu + c0],
                                           pivot_row[0],
                                           coefficients[(j + 1u) * 8u],
                                           rem, static_cast<uint8_t>(sub_rows), m_lu_repeat);
                    } else {
                        AscendC::MulAddDst(M[(j + 1u) * m_stride_lu + c0],
                                           pivot_row[0],
                                           coefficients[(j + 1u) * 8u],
                                           64u, static_cast<uint8_t>(sub_rows), m_lu_repeat);
                        AscendC::MulAddDst(M[(j + 1u) * m_stride_lu + c0 + 64u],
                                           pivot_row[64],
                                           coefficients[(j + 1u) * 8u],
                                           rem - 64u, static_cast<uint8_t>(sub_rows), m_lu_repeat);
                    }
                    AscendC::MulAddDst(y[(j + 1u) * 8u], y[j * 8u], coefficients[(j + 1u) * 8u],
                                       64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), y_repeat8);
                    AscendC::PipeBarrier<PIPE_V>();
                }

                if (slot == 7u || step + 1u == w1_lu) {
                    fence<AscendC::HardEvent::V_MTE3>();
                    AscendC::DataCopy(lu_gm[(j - slot) * pad_n], lu_stage, (slot + 1u) * pad_n);
                    fence<AscendC::HardEvent::MTE3_V>();
                }
            }
            fence<AscendC::HardEvent::MTE3_MTE2>();
        }

        // Phase 3: 1-Barrier Back substitution U * X0 = y
        if (w1 > 0u) {
            AscendC::DataCopy(M, lu_gm[128u * pad_n], w1 * pad_n);
            fence<AscendC::HardEvent::MTE2_V>();
            back_sub_panel(128u, w1, pad_n, M, y, coefficients, lu_stage, y_repeat8);
            fence<AscendC::HardEvent::V_MTE2>();
        }
        AscendC::DataCopy(M, lu_gm[0], w0 * pad_n);
        fence<AscendC::HardEvent::MTE2_V>();
        back_sub_panel(0u, w0, pad_n, M, y, coefficients, lu_stage, y_repeat8);
        AscendC::Brcb(lu_stage, inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(x0, y, lu_stage, pad_n * 8u);
        AscendC::PipeBarrier<PIPE_V>();

        // Phase 4: Pipelined 4-Barrier Vectorized Compensated Refinement
        if (m_stride_lu != m_stride_ref) {
            AscendC::CreateVecIndex(col_offsets, int32_t(0), pad_n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(col_offsets, col_offsets, static_cast<int32_t>(m_stride_ref * sizeof(float)), pad_n);
            AscendC::PipeBarrier<PIPE_V>();
        }
        auto split_mask_i32 = lu_col.template ReinterpretCast<int32_t>();
        AscendC::Duplicate(split_mask_i32, static_cast<int32_t>(0xfffff000u), pad_n);
        AscendC::PipeBarrier<PIPE_V>();
        auto split_mask_u16 = split_mask_i32.template ReinterpretCast<uint16_t>();
        const uint32_t num_panels = w1 > 0u ? 2u : 1u;
        const uint32_t max_refine_iters = (n > 200u) ? 2u : 1u;

        for (uint32_t refine_iter = 0; refine_iter < max_refine_iters; ++refine_iter) {
            fence<AscendC::HardEvent::V_MTE2>();
            load_rhs_b<T>(n, pad_n, k, b_offset, input_b, y, lu_stage, coefficients);
            for (uint32_t t = 0; t < k; ++t) {
                AscendC::Gather(s_hi[t * pad_n], y[t], y_col_offsets_u32, uint32_t(0), n);
            }
            AscendC::Duplicate(s_lo, 0.0f, k * pad_n);
            AscendC::PipeBarrier<PIPE_V>();
            fence<AscendC::HardEvent::V_S>();

            for (uint32_t panel = 0; panel < num_panels; ++panel) {
                const uint32_t col_start = panel * 128u;
                const uint32_t width = panel == 0u ? w0 : w1;
                const uint32_t width_pad = panel == 0u ? w0_pad : w1_pad;
                fence<AscendC::HardEvent::V_MTE2>();
                load_panel_a<T>(n, batch, col_start, width, width_pad, m_stride_ref, input_a, M, lu_stage);

                for (uint32_t rhs_base = 0; rhs_base < k; rhs_base += 4u) {
                    const uint32_t cur_k = (k - rhs_base < 4u) ? (k - rhs_base) : 4u;
                    const uint32_t len = cur_k * pad_n;
                    auto s_hi_chunk = s_hi[rhs_base * pad_n];
                    auto s_lo_chunk = s_lo[rhs_base * pad_n];
                    auto d1_tmp_4 = s_hi[1024];
                    auto p_err_alt_4 = s_lo[1024];

                    AscendC::Gather(col_vec, M[0], col_offsets_u32, uint32_t(0), pad_n);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::And(abs_col.template ReinterpretCast<uint16_t>(),
                                 col_vec.template ReinterpretCast<uint16_t>(),
                                 split_mask_u16, pad_n * 2u);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(pivot_row, col_vec, abs_col, pad_n);
                    AscendC::PipeBarrier<PIPE_V>();

                    if (k <= 4u) {
                        for (uint32_t step = 0; step < width; ++step) {
                            const uint32_t col = col_start + step;
                            if ((step & 1u) == 0u) {
                                residual_step_k4(
                                    step, width, cur_k, pad_n, len, &x0_ptr[col * 8u + rhs_base],
                                    M, col_offsets_u32, split_mask_u16, col_vec,
                                    s_hi_chunk, t_sum_4, s_lo_chunk, p_hi_4, p_err_4, p_err_alt_4,
                                    abs_col, row_tmp, pivot_row, scratch, v_tmp_4, u_tmp_4, d1_tmp_4);
                            } else {
                                residual_step_k4(
                                    step, width, cur_k, pad_n, len, &x0_ptr[col * 8u + rhs_base],
                                    M, col_offsets_u32, split_mask_u16, col_vec,
                                    t_sum_4, s_hi_chunk, s_lo_chunk, p_hi_4, p_err_alt_4, p_err_4,
                                    row_tmp, abs_col, scratch, pivot_row, v_tmp_4, u_tmp_4, d1_tmp_4);
                            }
                        }
                        if (((width - 1u) & 1u) != 0u) {
                            AscendC::Sub(u_tmp_4, t_sum_4, u_tmp_4, len);
                            AscendC::Add(d1_tmp_4, d1_tmp_4, p_err_alt_4, len);
                        } else {
                            AscendC::Sub(u_tmp_4, s_hi_chunk, u_tmp_4, len);
                            AscendC::Add(d1_tmp_4, d1_tmp_4, p_err_4, len);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(s_lo_chunk, s_lo_chunk, d1_tmp_4, len);
                        if ((width & 1u) != 0u) {
                            AscendC::Adds(s_hi_chunk, t_sum_4, 0.0f, len);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        for (uint32_t step = 0; step < width; ++step) {
                            const uint32_t col = col_start + step;
                            float x_val[4], x_lo_val[4];
                            if (step > 0u) {
                                AscendC::Add(s_lo_chunk, s_lo_chunk, v_tmp_4, len);
                            }
                            for (uint32_t t = 0; t < cur_k; ++t) {
                                const float x = -x0_ptr[col * 8u + rhs_base + t];
                                const float x_hi = bits_float(float_bits(x) & 0xfffff000u);
                                const float x_lo = x - x_hi;
                                x_val[t] = x;
                                x_lo_val[t] = x_lo;
                                AscendC::Muls(p_hi_4[t * pad_n], col_vec, x, pad_n);
                                AscendC::Muls(p_err_4[t * pad_n], abs_col, x_hi, pad_n);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            if (step > 0u) {
                                AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
                            }
                            AscendC::Sub(p_err_4, p_err_4, p_hi_4, len);
                            if ((step & 1u) != 0u) {
                                AscendC::Add(s_hi_chunk, t_sum_4, p_hi_4, len);
                            } else {
                                AscendC::Add(t_sum_4, s_hi_chunk, p_hi_4, len);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            for (uint32_t t = 0; t < cur_k; ++t) {
                                AscendC::Axpy(p_err_4[t * pad_n], abs_col, x_lo_val[t], pad_n);
                            }
                            if ((step & 1u) != 0u) {
                                AscendC::Sub(v_tmp_4, s_hi_chunk, t_sum_4, len);
                            } else {
                                AscendC::Sub(v_tmp_4, t_sum_4, s_hi_chunk, len);
                            }
                            if (step + 1u < width) {
                                AscendC::Gather(col_vec, M[step + 1u], col_offsets_u32, uint32_t(0), pad_n);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            for (uint32_t t = 0; t < cur_k; ++t) {
                                AscendC::Axpy(p_err_4[t * pad_n], pivot_row, x_val[t], pad_n);
                            }
                            if ((step & 1u) != 0u) {
                                AscendC::Sub(u_tmp_4, s_hi_chunk, v_tmp_4, len);
                            } else {
                                AscendC::Sub(u_tmp_4, t_sum_4, v_tmp_4, len);
                            }
                            AscendC::Sub(p_hi_4, p_hi_4, v_tmp_4, len);
                            if (step + 1u < width) {
                                AscendC::And(abs_col.template ReinterpretCast<uint16_t>(),
                                             col_vec.template ReinterpretCast<uint16_t>(),
                                             split_mask_u16, pad_n * 2u);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Add(v_tmp_4, p_err_4, p_hi_4, len);
                            if ((step & 1u) != 0u) {
                                AscendC::Sub(u_tmp_4, t_sum_4, u_tmp_4, len);
                            } else {
                                AscendC::Sub(u_tmp_4, s_hi_chunk, u_tmp_4, len);
                            }
                            if (step + 1u < width) {
                                AscendC::Sub(pivot_row, col_vec, abs_col, pad_n);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        AscendC::Add(s_lo_chunk, s_lo_chunk, v_tmp_4, len);
                        if ((width & 1u) != 0u) {
                            AscendC::Adds(s_hi_chunk, t_sum_4, 0.0f, len);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
            }

            AscendC::Add(s_hi, s_hi, s_lo, k * pad_n);
            AscendC::Duplicate(y, 0.0f, (pad_n + 8u) * 8u);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t t = 0; t < k; ++t) {
                AscendC::Brcb(coefficients, s_hi[t * pad_n], static_cast<uint8_t>(pad_n / 8u), {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(y, coefficients, unit_masks[t * 8u], 64u, static_cast<uint8_t>(pad_n / 8u), lane_rep);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Duplicate(lu_stage[pad_n * 8u], 0.0f, 64u);
            AscendC::Duplicate(coefficients[pad_n * 8u], 0.0f, 64u);
            AscendC::PipeBarrier<PIPE_V>();

            // Phase 4b + Phase 4c
            fence<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(M, lu_gm[0], w0 * pad_n);
            fence<AscendC::HardEvent::MTE2_V>();
            fwd_sub_panel(0u, w0, n, pad_n, piv, M, y, row_tmp, coefficients, lu_stage, y_repeat8);
            if (w1 > 0u) {
                fence<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopy(M, lu_gm[128u * pad_n], w1 * pad_n);
                fence<AscendC::HardEvent::MTE2_V>();
                fwd_sub_panel(128u, w1, n, pad_n, piv, M, y, row_tmp, coefficients, lu_stage, y_repeat8);
                back_sub_panel(128u, w1, pad_n, M, y, coefficients, lu_stage, y_repeat8);
                fence<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopy(M, lu_gm[0], w0 * pad_n);
                fence<AscendC::HardEvent::MTE2_V>();
            }
            back_sub_panel(0u, w0, pad_n, M, y, coefficients, lu_stage, y_repeat8);
            AscendC::Brcb(lu_stage, inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(y, y, lu_stage, pad_n * 8u);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(x0, x0, y, pad_n * 8u);
            AscendC::PipeBarrier<PIPE_V>();

            if (refine_iter + 1u < max_refine_iters) {
                AscendC::Abs(lu_stage, y, pad_n * 8u);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceMax(reduced, lu_stage, scratch, pad_n * 8u, false);
                fence<AscendC::HardEvent::V_S>();
                const float max_delta = reduced.GetValue(0);
                if (!(max_delta > 3.5e-3f)) {
                    break;
                }
            }
        }

        if constexpr (sizeof(T) == 4) {
            fence<AscendC::HardEvent::V_MTE3>();
            const AscendC::DataCopyExtParams x_store{
                static_cast<uint16_t>(n),
                static_cast<uint32_t>(k * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyPad(output_x[batch * n * k], x0, x_store);
            fence<AscendC::HardEvent::MTE3_V>();
        } else {
            auto pack_offsets = M.template ReinterpretCast<int32_t>();
            fence<AscendC::HardEvent::V_S>();
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t t = 0; t < k; ++t) {
                    pack_offsets.SetValue(i * k + t, static_cast<int32_t>((i * 8u + t) * sizeof(float)));
                }
            }
            fence<AscendC::HardEvent::S_V>();
            AscendC::Gather(x_out, x0, pack_offsets.template ReinterpretCast<uint32_t>(), uint32_t(0), n * k);
            AscendC::PipeBarrier<PIPE_V>();
            auto raw_x = lu_stage.template ReinterpretCast<T>();
            AscendC::Cast(raw_x, x_out, AscendC::RoundMode::CAST_RINT, (n * k + 7u) & ~7u);
            fence<AscendC::HardEvent::V_MTE3>();
            const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(n * k * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(output_x[batch * n * k], raw_x, x_store);
            fence<AscendC::HardEvent::MTE3_V>();
        }
    }
    return SUCCESS;
}

}  // namespace elec::lstsq_fast
