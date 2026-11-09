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
template<typename T>
ELEC_FN void tree_vector(const Config& c,GM_ADDR points_address,GM_ADDR segments_address,
                         GM_ADDR distance_address,GM_ADDR index_address) {
    constexpr uint32_t tile=512;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage,mask_buffer,out_buffer,index_buffer,raw_buffer,point_buffer,offset_buffer;
    pipe.InitBuffer(storage,14*tile*sizeof(float));pipe.InitBuffer(mask_buffer,32);
    pipe.InitBuffer(out_buffer,tile*sizeof(T));pipe.InitBuffer(index_buffer,tile*sizeof(int32_t));
    pipe.InitBuffer(raw_buffer,3*tile*sizeof(T));pipe.InitBuffer(point_buffer,sizeof(T)==2?3*tile*sizeof(float):32);
    pipe.InitBuffer(offset_buffer,tile*sizeof(int32_t));
    auto raw=raw_buffer.Get<T>();auto values=sizeof(T)==2?point_buffer.Get<float>():raw.template ReinterpretCast<float>();
    auto offsets=offset_buffer.Get<int32_t>();
    AscendC::CreateVecIndex(offsets,int32_t(0),tile);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(offsets,offsets,int32_t(12),tile);AscendC::PipeBarrier<PIPE_V>();
    auto all=storage.Get<float>();
    auto px=all[0*tile],py=all[1*tile],pz=all[2*tile];
    auto dx=all[3*tile],dy=all[4*tile],dz=all[5*tile];
    auto t=all[6*tile],tmp=all[7*tile],ex=all[8*tile],ey=all[9*tile],ez=all[10*tile];
    auto best=all[11*tile],indices=all[12*tile],candidate=all[13*tile];
    auto mask=mask_buffer.Get<uint8_t>();auto output=out_buffer.Get<T>();
    auto half_output=out_buffer.Get<uint16_t>();auto index=index_buffer.Get<int32_t>();
    Tensor segments(segments_address,c.dtype);
    AscendC::GlobalTensor<T> points;points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
    AscendC::GlobalTensor<T> distance;
    AscendC::GlobalTensor<int32_t> nearest;
    distance.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
    nearest.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));
    const uint64_t tiles_per_batch=(c.m+tile-1)/tile;
    for(uint64_t task=AscendC::GetBlockIdx();task<c.batch*tiles_per_batch;task+=AscendC::GetBlockNum()) {
        const uint64_t b=task/tiles_per_batch,first=(task%tiles_per_batch)*tile;
        const uint32_t count=static_cast<uint32_t>(c.m-first<tile?c.m-first:tile);
        const AscendC::DataCopyExtParams load{1,count*3*uint32_t(sizeof(T)),0,0,0};
        const AscendC::DataCopyPadExtParams<T> padding{false,0,0,0};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::DataCopyPad(raw,points[(b*c.m+first)*3],load,padding);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        if constexpr(sizeof(T)==2) {AscendC::Cast(values,raw,AscendC::RoundMode::CAST_NONE,count*3);AscendC::PipeBarrier<PIPE_V>();}
        AscendC::Gather(px,values,offsets.template ReinterpretCast<uint32_t>(),0,count);
        AscendC::Gather(py,values[1],offsets.template ReinterpretCast<uint32_t>(),0,count);
        AscendC::Gather(pz,values[2],offsets.template ReinterpretCast<uint32_t>(),0,count);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(best,bits_float(0x7f800000u),count);
        AscendC::Duplicate(indices,0.0f,count);AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t j=0;j<c.n;++j) {
            const uint64_t s=(b*c.n+j)*6;
            const float ax=segments.get(s),ay=segments.get(s+1),az=segments.get(s+2);
            const float ux=segments.get(s+3)-ax,uy=segments.get(s+4)-ay,uz=segments.get(s+5)-az;
            const float length2=(ux*ux+uy*uy)+uz*uz;
            if(length2>0.0f) {
                AscendC::Adds(dx,px,-ax,count);AscendC::Adds(dy,py,-ay,count);AscendC::Adds(dz,pz,-az,count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(t,dx,ux,count);AscendC::Muls(tmp,dy,uy,count);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(t,t,tmp,count);AscendC::Muls(tmp,dz,uz,count);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(t,t,tmp,count);AscendC::Duplicate(tmp,length2,count);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(t,t,tmp,count);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(t,t,0.0f,count);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(t,t,1.0f,count);AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::Duplicate(t,0.0f,count);AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Muls(ex,t,ux,count);AscendC::Muls(ey,t,uy,count);AscendC::Muls(ez,t,uz,count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ex,ex,ax,count);AscendC::Adds(ey,ey,ay,count);AscendC::Adds(ez,ez,az,count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(ex,px,ex,count);AscendC::Sub(ey,py,ey,count);AscendC::Sub(ez,pz,ez,count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ex,ex,ex,count);AscendC::Mul(ey,ey,ey,count);AscendC::Mul(ez,ez,ez,count);
            AscendC::PipeBarrier<PIPE_V>();AscendC::Add(tmp,ex,ey,count);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(tmp,tmp,ez,count);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(candidate,static_cast<float>(static_cast<int32_t>(j)),count);AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t offset=0;offset<count;offset+=64) {
                const uint32_t lanes=count-offset<64?count-offset:64;
                // 910B Compare's counter overload executes complete 64-FP32
                // repeats. Compare one full repeat and mask Select's tail.
                AscendC::Compare(mask,tmp[offset],best[offset],AscendC::CMPMODE::LT,uint32_t(64));AscendC::PipeBarrier<PIPE_V>();
                AscendC::Select(indices[offset],mask,candidate[offset],indices[offset],AscendC::SELMODE::VSEL_CMPMASK_SPR,lanes);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Min(best,best,tmp,count);AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Sqrt(best,best,count);
        AscendC::Cast(index,indices,AscendC::RoundMode::CAST_RINT,count);
        if constexpr(sizeof(T)==2) {AscendC::Cast(output,best,AscendC::RoundMode::CAST_RINT,count);}
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const AscendC::DataCopyExtParams copy_distance{1,count*uint32_t(sizeof(T)),0,0,0};
        const AscendC::DataCopyExtParams copy_index{1,count*uint32_t(sizeof(int32_t)),0,0,0};
        if constexpr(sizeof(T)==2)AscendC::DataCopyPad(distance[b*c.m+first],output,copy_distance);
        else AscendC::DataCopyPad(distance[b*c.m+first],best,copy_distance);
        AscendC::DataCopyPad(nearest[b*c.m+first],index,copy_index);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }
}
}
