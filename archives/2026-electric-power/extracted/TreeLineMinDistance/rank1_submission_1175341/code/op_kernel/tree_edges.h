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
template<AscendC::HardEvent Event>
ELEC_FN void tree_edge_fence() {
    AscendC::SetFlag<Event>(0);
    AscendC::WaitFlag<Event>(0);
}
template<int32_t stride>
ELEC_FN void tree_edge_offsets(const AscendC::LocalTensor<int32_t>& dst,uint32_t count) {
    const uint32_t init_n=count<32?count:32;
    for(uint32_t i=0;i<init_n;++i)dst.SetValue(i,static_cast<int32_t>(i)*stride);
    tree_edge_fence<AscendC::HardEvent::S_V>();
    for(uint32_t filled=32;filled<count;filled*=2) {
        const uint32_t size=count-filled<filled?count-filled:filled;
        AscendC::Adds(dst[filled],dst,static_cast<int32_t>(filled)*stride,size);
    }
}

// With one segment, vectorize across the point tile and skip WholeReduceMin.
template<typename T>
ELEC_FN void tree_one_segment(const Config& c,GM_ADDR points_address,GM_ADDR segments_address,
                              GM_ADDR distance_address,GM_ADDR index_address) {
    constexpr uint32_t tile=2048,stride=tile+64;
    constexpr uint32_t work_at=3*tile*sizeof(T),offset_at=work_at+9*stride*sizeof(float);
    constexpr uint32_t index_at=offset_at+tile*sizeof(int32_t);
    AscendC::LocalTensor<T> raw(AscendC::TPosition::VECCALC,0,3*tile);
    AscendC::LocalTensor<float> work(AscendC::TPosition::VECCALC,work_at,9*stride);
    auto values=sizeof(T)==2?work[7*stride]:raw.template ReinterpretCast<float>();
    auto dx=work,dy=work[stride],dz=work[2*stride],projection=work[3*stride],temp=work[4*stride];
    auto t1=work[5*stride],t2=work[6*stride];
    auto half_output=work[5*stride].template ReinterpretCast<T>();
    AscendC::LocalTensor<int32_t> offsets(AscendC::TPosition::VECCALC,offset_at,tile);
    AscendC::LocalTensor<int32_t> indices(AscendC::TPosition::VECCALC,index_at,tile);
    AscendC::GlobalTensor<T> points,segments,output;
    AscendC::GlobalTensor<int32_t> output_index;
    points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
    segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
    output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));
    const AscendC::DataCopyPadExtParams<T> padding{false,0,0,0};
    const uint64_t per_batch=(c.m+tile-1)/tile;
    bool offsets_ready=false;
    for(uint64_t task=AscendC::GetBlockIdx();task<c.batch*per_batch;task+=AscendC::GetBlockNum()) {
        const uint64_t batch=task/per_batch,first=task%per_batch*tile;
        const uint32_t count=static_cast<uint32_t>(c.m-first<tile?c.m-first:tile);
        if(offsets_ready) {
            tree_edge_fence<AscendC::HardEvent::V_MTE2>();
            tree_edge_fence<AscendC::HardEvent::MTE3_V>();
        }
        const AscendC::DataCopyExtParams load{1,count*3*uint32_t(sizeof(T)),0,0,0};
        const AscendC::DataCopyExtParams store{1,count*uint32_t(sizeof(T)),0,0,0},store_index{1,count*4,0,0,0};
        AscendC::DataCopyPad(raw,points[(batch*c.m+first)*3],load,padding);
        if(!offsets_ready) {
            tree_edge_offsets<12>(offsets,tile);
            AscendC::Duplicate(indices,int32_t(0),tile);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
        }
        const float ax=static_cast<float>(segments.GetValue(batch*6)),ay=static_cast<float>(segments.GetValue(batch*6+1)),az=static_cast<float>(segments.GetValue(batch*6+2));
        const float ux=static_cast<float>(segments.GetValue(batch*6+3))-ax,uy=static_cast<float>(segments.GetValue(batch*6+4))-ay,uz=static_cast<float>(segments.GetValue(batch*6+5))-az;
        const float length2=(ux*ux+uy*uy)+uz*uz;
        const float inv_length2=length2>0.0f?(1.0f/length2):0.0f;
        const float vx=ux*inv_length2,vy=uy*inv_length2,vz=uz*inv_length2;
        if(!offsets_ready) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
            offsets_ready=true;
        }
        tree_edge_fence<AscendC::HardEvent::MTE2_V>();
        // Overlap 64 KiB output_index write on PIPE_MTE3 with the 22 Vector instructions on PIPE_V
        AscendC::DataCopyPad(output_index[batch*c.m+first],indices,store_index);
        if constexpr(sizeof(T)==2)AscendC::Cast(values,raw,AscendC::RoundMode::CAST_NONE,count*3);
        AscendC::Gather(dx,values,offsets.template ReinterpretCast<uint32_t>(),0,count);
        AscendC::Gather(dy,values[1],offsets.template ReinterpretCast<uint32_t>(),0,count);
        AscendC::Gather(dz,values[2],offsets.template ReinterpretCast<uint32_t>(),0,count);
        AscendC::Adds(dx,dx,-ax,count);AscendC::Adds(dy,dy,-ay,count);AscendC::Adds(dz,dz,-az,count);
        if(length2>0.0f) {
            AscendC::Muls(projection,dx,vx,count);
            AscendC::Muls(temp,dy,vy,count);
            AscendC::Muls(t1,dz,vz,count);
            AscendC::Add(projection,projection,temp,count);
            AscendC::Add(projection,projection,t1,count);
            AscendC::Maxs(projection,projection,0.0f,count);
            AscendC::Mins(projection,projection,1.0f,count);
            AscendC::Muls(temp,projection,-ux,count);
            AscendC::Muls(t1,projection,-uy,count);
            AscendC::Muls(t2,projection,-uz,count);
            AscendC::Add(dx,dx,temp,count);
            AscendC::Add(dy,dy,t1,count);
            AscendC::Add(dz,dz,t2,count);
            AscendC::Mul(dx,dx,dx,count);AscendC::Mul(dy,dy,dy,count);AscendC::Mul(dz,dz,dz,count);
            AscendC::Add(dx,dx,dy,count);AscendC::Add(dx,dx,dz,count);
            AscendC::Sqrt(dx,dx,count);
        } else {
            AscendC::Mul(dx,dx,dx,count);AscendC::Mul(dy,dy,dy,count);AscendC::Mul(dz,dz,dz,count);
            AscendC::Add(dx,dx,dy,count);AscendC::Add(dx,dx,dz,count);
            AscendC::Sqrt(dx,dx,count);
        }
        if constexpr(sizeof(T)==2) {AscendC::Cast(half_output,dx,AscendC::RoundMode::CAST_RINT,count);}
        tree_edge_fence<AscendC::HardEvent::V_MTE3>();
        if constexpr(sizeof(T)==2)AscendC::DataCopyPad(output[batch*c.m+first],half_output,store);
        else AscendC::DataCopyPad(output[batch*c.m+first],dx,store);
    }
}

// With one point, vectorize across the entire segment set and reduce in two PIPE_V stages.
template<typename T>
ELEC_FN void tree_one_point(const Config& c,GM_ADDR points_address,GM_ADDR segments_address,
                            GM_ADDR distance_address,GM_ADDR index_address) {
    if(AscendC::GetBlockIdx()>=c.batch) return;
    constexpr uint32_t tile=1024;
    constexpr uint32_t soa_at=24576,work_at=soa_at+8448*sizeof(float);
    constexpr uint32_t offset_at=work_at+6272*sizeof(float),point_at=offset_at+tile*sizeof(int32_t);
    constexpr uint32_t result_at=point_at+128;
    AscendC::LocalTensor<T> raw(AscendC::TPosition::VECCALC,0,6*tile);
    AscendC::LocalTensor<float> work(AscendC::TPosition::VECCALC,work_at,6272);
    auto input=sizeof(T)==2?work:raw.template ReinterpretCast<float>();
    AscendC::LocalTensor<float> soa(AscendC::TPosition::VECCALC,soa_at,8448);
    auto ax=soa[0],ay=soa[1024],az=soa[2048],ux=soa[3136],uy=soa[4160],uz=soa[5184];
    auto projection=soa[6272],length=soa[7360];
    auto dx=work[0],dy=work[1024],dz=work[2048],temp=work[3136],temp2=work[4160],temp3=work[5184];
    AscendC::LocalTensor<int32_t> offsets(AscendC::TPosition::VECCALC,offset_at,tile);
    AscendC::LocalTensor<float> result(AscendC::TPosition::VECCALC,result_at,32);
    auto dist_out=result[24];
    auto half_output=result[16].template ReinterpretCast<T>();
    auto index_out=result[8].template ReinterpretCast<int32_t>();
    AscendC::GlobalTensor<T> points,segments,output;
    AscendC::GlobalTensor<int32_t> output_index;
    points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
    segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
    output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));
    const AscendC::DataCopyPadExtParams<T> padding{false,0,0,0};
    const uint32_t n=static_cast<uint32_t>(c.n);
    const uint32_t repeats=(n+63)/64;
    bool offsets_ready=false;
    for(uint64_t batch=AscendC::GetBlockIdx();batch<c.batch;batch+=AscendC::GetBlockNum()) {
        if(offsets_ready) {
            tree_edge_fence<AscendC::HardEvent::V_MTE2>();
            tree_edge_fence<AscendC::HardEvent::MTE3_V>();
            tree_edge_fence<AscendC::HardEvent::MTE3_S>();
        }
        const AscendC::DataCopyExtParams load{1,static_cast<uint32_t>(n*6*sizeof(T)),0,0,0};
        AscendC::DataCopyPad(raw,segments[batch*n*6],load,padding);
        if(!offsets_ready) {
            tree_edge_offsets<24>(offsets,n);
            offsets_ready=true;
        }
        if(n%64) {
            AscendC::Duplicate(dx[(n/64)*64],bits_float(0x7f800000u),64);
        }
        tree_edge_fence<AscendC::HardEvent::MTE2_V>();
        if constexpr(sizeof(T)==2)AscendC::Cast(input,raw,AscendC::RoundMode::CAST_NONE,n*6);
        const uint32_t soa_off[6]={0,1024,2048,3136,4160,5184};
        for(uint32_t axis=0;axis<6;++axis)AscendC::Gather(soa[soa_off[axis]],input[axis],offsets.template ReinterpretCast<uint32_t>(),0,n);
        if(n==1024) {
            AscendC::Sub(ux,ax,ux,3072);
            AscendC::Mul(temp,ux,ux,3072);
            AscendC::Add(length,temp,temp2,1024);
            AscendC::Add(length,length,temp3,1024);
            AscendC::Maxs(length,length,bits_float(0x00800000u),1024);
            const float px=static_cast<float>(points.GetValue(batch*3));
            const float py=static_cast<float>(points.GetValue(batch*3+1));
            const float pz=static_cast<float>(points.GetValue(batch*3+2));
            AscendC::Adds(dx,ax,-px,1024);AscendC::Adds(dy,ay,-py,1024);AscendC::Adds(dz,az,-pz,1024);
            AscendC::Mul(temp,dx,ux,3072);
            AscendC::Add(projection,temp,temp2,1024);
            AscendC::Add(projection,projection,temp3,1024);
            AscendC::Div(projection,projection,length,1024);
            AscendC::Maxs(projection,projection,0.0f,1024);
            AscendC::Mins(projection,projection,1.0f,1024);
            AscendC::Mul(temp,projection,ux,1024);
            AscendC::Mul(temp2,projection,uy,1024);
            AscendC::Mul(temp3,projection,uz,1024);
            AscendC::Sub(dx,dx,temp,3072);
            AscendC::Mul(dx,dx,dx,3072);
            AscendC::Add(dx,dx,dy,1024);
            AscendC::Add(dx,dx,dz,1024);
        } else {
            AscendC::Sub(ux,ax,ux,n);AscendC::Sub(uy,ay,uy,n);AscendC::Sub(uz,az,uz,n);
            AscendC::Mul(length,ux,ux,n);AscendC::Mul(temp,uy,uy,n);AscendC::Mul(temp2,uz,uz,n);
            AscendC::Add(length,length,temp,n);
            AscendC::Add(length,length,temp2,n);
            AscendC::Maxs(length,length,bits_float(0x00800000u),n);
            const float px=static_cast<float>(points.GetValue(batch*3));
            const float py=static_cast<float>(points.GetValue(batch*3+1));
            const float pz=static_cast<float>(points.GetValue(batch*3+2));
            AscendC::Adds(dx,ax,-px,n);AscendC::Adds(dy,ay,-py,n);AscendC::Adds(dz,az,-pz,n);
            AscendC::Mul(projection,dx,ux,n);AscendC::Mul(temp,dy,uy,n);AscendC::Mul(temp2,dz,uz,n);
            AscendC::Add(projection,projection,temp,n);
            AscendC::Add(projection,projection,temp2,n);
            AscendC::Div(projection,projection,length,n);
            AscendC::Maxs(projection,projection,0.0f,n);
            AscendC::Mins(projection,projection,1.0f,n);
            AscendC::Mul(ax,projection,ux,n);
            AscendC::Mul(ay,projection,uy,n);
            AscendC::Mul(az,projection,uz,n);
            AscendC::Sub(dx,dx,ax,n);
            AscendC::Sub(dy,dy,ay,n);
            AscendC::Sub(dz,dz,az,n);
            AscendC::Mul(dx,dx,dx,n);AscendC::Mul(dy,dy,dy,n);AscendC::Mul(dz,dz,dz,n);
            AscendC::Add(dx,dx,dy,n);
            AscendC::Add(dx,dx,dz,n);
        }
        const AscendC::DataCopyExtParams store{1,uint32_t(sizeof(T)),0,0,0},store_index{1,4,0,0,0};
        if(repeats==1) {
            AscendC::WholeReduceMin(result,dx,static_cast<int32_t>(n),1,1,1,8);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::Sqrt(dist_out,result,8);
            if constexpr(sizeof(T)==2) {AscendC::Cast(half_output,dist_out,AscendC::RoundMode::CAST_RINT,8);}
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            if constexpr(sizeof(T)==2)AscendC::DataCopyPad(output[batch],half_output,store);
            else AscendC::DataCopyPad(output[batch],dist_out,store);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            const int32_t winner=static_cast<int32_t>(result.template ReinterpretCast<uint32_t>().GetValue(1));
            index_out.SetValue(0,winner);
            tree_edge_fence<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(output_index[batch],index_out,store_index);
        } else {
            AscendC::WholeReduceMin(projection,dx,64,static_cast<int32_t>(repeats),1,1,8);
            const uint64_t value_mask[2]={((uint64_t(1)<<(2*repeats))-1)&uint64_t(0x55555555ULL),0};
            AscendC::WholeReduceMin(result,projection,value_mask,1,1,1,8);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::Sqrt(dist_out,result,8);
            if constexpr(sizeof(T)==2) {AscendC::Cast(half_output,dist_out,AscendC::RoundMode::CAST_RINT,8);}
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            if constexpr(sizeof(T)==2)AscendC::DataCopyPad(output[batch],half_output,store);
            else AscendC::DataCopyPad(output[batch],dist_out,store);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            const uint32_t slot=result.template ReinterpretCast<uint32_t>().GetValue(1);
            const int32_t winner=static_cast<int32_t>((slot/2)*64+projection.template ReinterpretCast<uint32_t>().GetValue(slot+1));
            index_out.SetValue(0,winner);
            tree_edge_fence<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(output_index[batch],index_out,store_index);
        }
    }
}
}
