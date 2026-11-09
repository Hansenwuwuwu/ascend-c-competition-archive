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
template<typename T,uint32_t width,uint32_t point_tile>
class TreeShort {
public:
    ELEC_FN void run(const Config& c,GM_ADDR points_address,GM_ADDR segments_address,
                     GM_ADDR distance_address,GM_ADDR index_address) {
        constexpr uint32_t pairs=width*point_tile,stage_size=6*width+3*point_tile;
        constexpr uint32_t work_stride=(pairs&64)?pairs:(pairs+64);
        static_assert(pairs>=8 && (pairs&(pairs-1))==0,"index expansion needs power-of-two pairs");
        constexpr uint32_t raw_bytes=(stage_size*sizeof(T)+31)/32*32;
        constexpr uint32_t stage_bytes=(stage_size*sizeof(float)+31)/32*32;
        constexpr uint32_t work_floats=1024+7*work_stride;
        constexpr uint32_t work_bytes=work_floats*sizeof(float);
        constexpr uint32_t offset_bytes=96*sizeof(int32_t);
        constexpr uint32_t stage_at=raw_bytes,work_at=stage_at+stage_bytes;
        constexpr uint32_t offset_at=work_at+work_bytes,result_at=offset_at+offset_bytes;
        AscendC::LocalTensor<T> raw(AscendC::TPosition::VECCALC,0,stage_size);
        AscendC::LocalTensor<float> stage(AscendC::TPosition::VECCALC,stage_at,stage_size);
        auto values=sizeof(T)==2?stage:raw.template ReinterpretCast<float>();
        AscendC::LocalTensor<float> work(AscendC::TPosition::VECCALC,work_at,work_floats);
        AscendC::LocalTensor<int32_t> offsets(AscendC::TPosition::VECCALC,offset_at,96);
        AscendC::LocalTensor<float> result(AscendC::TPosition::VECCALC,result_at,160);
        auto distance=result[64];
        auto indices=result[96].template ReinterpretCast<int32_t>();
        auto packed=result[128].template ReinterpretCast<T>();
        AscendC::GlobalTensor<T> points,segments,output;
        AscendC::GlobalTensor<int32_t> output_index;
        points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
        segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
        output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));
        const AscendC::DataCopyPadExtParams<T> padding{false,0,0,0};
        const uint64_t per_batch=(c.m+point_tile-1)/point_tile;
        if(AscendC::GetBlockIdx()>=c.batch*per_batch) return;
        bool offsets_ready=false;

        constexpr uint32_t s_stride=width==8?8:32;
        auto s_ax=work[0],s_ay=work[s_stride],s_az=work[2*s_stride],s_ux=work[3*s_stride],s_uy=work[4*s_stride],s_uz=work[5*s_stride];
        auto s_vx=work[384],s_vy=work[384+s_stride],s_vz=work[384+2*s_stride];
        auto bpx=work[576],bpy=work[640],bpz=work[704];
        auto px=work[768],py=work[776],pz=work[784];
        auto dx_p=work[1024],dy_p=work[1024+(width==8?64:work_stride)],dz_p=work[1024+2*(width==8?64:work_stride)];
        auto projection_p=work[1024+3*work_stride];
        auto temp_p=work[1024+4*work_stride],d2_p=work[1024+5*work_stride],extra_p=work[1024+6*work_stride];
        auto wx_p=work[1024+192],wy_p=work[1024+256],wz_p=work[1024+320];
        auto seg_off=offsets,pt_off=offsets[32];
        const uint32_t seg_n=static_cast<uint32_t>(c.n);

        for(uint64_t task=AscendC::GetBlockIdx();task<c.batch*per_batch;task+=AscendC::GetBlockNum()) {
            const uint64_t batch=task/per_batch,first=task%per_batch*point_tile;
            const uint32_t count=static_cast<uint32_t>(c.m-first<point_tile?c.m-first:point_tile),lanes=count*width;
            if(offsets_ready) {
                fence<AscendC::HardEvent::V_MTE2>();
                fence<AscendC::HardEvent::MTE3_V>();
            }
            const AscendC::DataCopyExtParams load_segments{1,uint32_t(c.n*6*sizeof(T)),0,0,0},load_points{1,count*3*uint32_t(sizeof(T)),0,0,0};
            AscendC::DataCopyPad(raw,segments[batch*c.n*6],load_segments,padding);
            AscendC::DataCopyPad(raw[6*width],points[(batch*c.m+first)*3],load_points,padding);
            if(!offsets_ready) {
                for(uint32_t i=0;i<8;++i) {
                    seg_off.SetValue(i,static_cast<int32_t>(i)*24);
                    pt_off.SetValue(i,static_cast<int32_t>(i)*12);
                }
                fence<AscendC::HardEvent::S_V>();
                if constexpr(width==32) {
                    AscendC::Adds(seg_off[8],seg_off,int32_t(8*24),8);
                    AscendC::Adds(seg_off[16],seg_off,int32_t(16*24),16);
                }
                offsets_ready=true;
            }
            fence<AscendC::HardEvent::MTE2_V>();
            if constexpr(sizeof(T)==2) {
                AscendC::Cast(values,raw,AscendC::RoundMode::CAST_NONE,6*width+count*3);
            }
            for(uint32_t axis=0;axis<6;++axis) {
                AscendC::Gather(work[axis*s_stride],values[axis],seg_off.template ReinterpretCast<uint32_t>(),0,seg_n);
            }
            AscendC::Gather(px,values[6*width],pt_off.template ReinterpretCast<uint32_t>(),0,count);
            AscendC::Gather(py,values[6*width+1],pt_off.template ReinterpretCast<uint32_t>(),0,count);
            AscendC::Gather(pz,values[6*width+2],pt_off.template ReinterpretCast<uint32_t>(),0,count);

            if constexpr(width==8) {
                AscendC::Sub(s_ux,s_ux,s_ax,24);
                AscendC::Brcb(bpx,px,3,{1,8});
                AscendC::Mul(s_vx,s_ux,s_ux,24);
                const AscendC::BinaryRepeatParams seg_3axis{1,1,0,8,8,1},seg_blk0{1,1,0,8,8,0},proj_3axis{1,1,0,8,0,1},contig{1,1,1,8,8,8};
                const AscendC::UnaryRepeatParams ucontig{1,1,8,8};
                AscendC::Sub<float,true>(dx_p,bpx,s_ax,uint64_t(64),3,seg_3axis);
                AscendC::Add(s_vx,s_vx,s_vy,seg_n);
                AscendC::Add(s_vx,s_vx,s_vz,seg_n);
                AscendC::Maxs(s_vx,s_vx,bits_float(0x00800000u),seg_n);
                AscendC::Mul<float,true>(wx_p,dx_p,s_ux,uint64_t(64),3,seg_3axis);
                AscendC::Add<float,false>(wx_p,wx_p,wy_p,uint64_t(64),1,contig);
                AscendC::Add<float,false>(projection_p,wx_p,wz_p,uint64_t(64),1,contig);
                AscendC::Div<float,false>(projection_p,projection_p,s_vx,uint64_t(64),1,seg_blk0);
                AscendC::Maxs<float,false>(projection_p,projection_p,0.0f,uint64_t(64),1,ucontig);
                AscendC::Mins<float,false>(projection_p,projection_p,1.0f,uint64_t(64),1,ucontig);
                AscendC::Mul<float,false>(wx_p,projection_p,s_ux,uint64_t(64),3,proj_3axis);
                AscendC::Sub<float,false>(dx_p,dx_p,wx_p,uint64_t(64),3,contig);
                AscendC::Mul<float,false>(dx_p,dx_p,dx_p,uint64_t(64),3,contig);
                AscendC::Add<float,false>(dx_p,dx_p,dy_p,uint64_t(64),1,contig);
                AscendC::Add<float,false>(dx_p,dx_p,dz_p,uint64_t(64),1,contig);
                AscendC::WholeReduceMin<float,true>(result,dx_p,static_cast<int32_t>(c.n),count,1,1,1);
            } else {
                const AscendC::BinaryRepeatParams point_segment{1,0,1,4,1,0},segment{1,1,1,4,4,0},contig4{1,1,1,4,4,4};
                const AscendC::UnaryRepeatParams ucontig4{1,1,4,4};
                const uint8_t r_cnt=static_cast<uint8_t>(count);
                AscendC::Sub<float,true>(s_ux,s_ux,s_ax,uint64_t(32),3,contig4);
                AscendC::Brcb(bpx,px,3,{1,8});
                AscendC::Mul<float,false>(s_vx,s_ux,s_ux,uint64_t(32),3,contig4);
                AscendC::Add<float,false>(s_vx,s_vx,s_vy,uint64_t(32),1,contig4);
                AscendC::Add<float,false>(s_vx,s_vx,s_vz,uint64_t(32),1,contig4);
                AscendC::Maxs<float,false>(s_vx,s_vx,bits_float(0x00800000u),uint64_t(32),1,ucontig4);
                AscendC::Sub<float,false>(dx_p,bpx,s_ax,uint64_t(32),r_cnt,point_segment);
                AscendC::Sub<float,false>(dy_p,bpy,s_ay,uint64_t(32),r_cnt,point_segment);
                AscendC::Sub<float,false>(dz_p,bpz,s_az,uint64_t(32),r_cnt,point_segment);
                AscendC::Mul<float,false>(projection_p,dx_p,s_ux,uint64_t(32),r_cnt,segment);
                AscendC::Mul<float,false>(temp_p,dy_p,s_uy,uint64_t(32),r_cnt,segment);
                AscendC::Mul<float,false>(d2_p,dz_p,s_uz,uint64_t(32),r_cnt,segment);
                AscendC::Add<float,false>(projection_p,projection_p,temp_p,uint64_t(32),r_cnt,contig4);
                AscendC::Add<float,false>(projection_p,projection_p,d2_p,uint64_t(32),r_cnt,contig4);
                AscendC::Div<float,false>(projection_p,projection_p,s_vx,uint64_t(32),r_cnt,segment);
                AscendC::Maxs<float,false>(projection_p,projection_p,0.0f,uint64_t(32),r_cnt,ucontig4);
                AscendC::Mins<float,false>(projection_p,projection_p,1.0f,uint64_t(32),r_cnt,ucontig4);
                AscendC::Mul<float,false>(temp_p,projection_p,s_ux,uint64_t(32),r_cnt,segment);
                AscendC::Mul<float,false>(d2_p,projection_p,s_uy,uint64_t(32),r_cnt,segment);
                AscendC::Mul<float,false>(extra_p,projection_p,s_uz,uint64_t(32),r_cnt,segment);
                AscendC::Sub<float,false>(dx_p,dx_p,temp_p,uint64_t(32),r_cnt,contig4);
                AscendC::Sub<float,false>(dy_p,dy_p,d2_p,uint64_t(32),r_cnt,contig4);
                AscendC::Sub<float,false>(dz_p,dz_p,extra_p,uint64_t(32),r_cnt,contig4);
                AscendC::Mul<float,false>(d2_p,dx_p,dx_p,uint64_t(32),r_cnt,contig4);
                AscendC::Mul<float,false>(temp_p,dy_p,dy_p,uint64_t(32),r_cnt,contig4);
                AscendC::Mul<float,false>(extra_p,dz_p,dz_p,uint64_t(32),r_cnt,contig4);
                AscendC::Add<float,false>(d2_p,d2_p,temp_p,uint64_t(32),r_cnt,contig4);
                AscendC::Add<float,false>(d2_p,d2_p,extra_p,uint64_t(32),r_cnt,contig4);
                AscendC::WholeReduceMin<float,true>(result,d2_p,static_cast<int32_t>(c.n),count,1,1,4);
            }
            const AscendC::DataCopyExtParams store{1,count*uint32_t(sizeof(T)),0,0,0},store_index{1,count*4,0,0,0};
            uint64_t rsvd_cnt=0;
            const AscendC::GatherMaskParams gm_red{1,1,8,0};
            AscendC::GatherMask(indices,result.template ReinterpretCast<int32_t>(),static_cast<uint8_t>(2),false,uint32_t(0),gm_red,rsvd_cnt);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::GatherMask(distance,result,static_cast<uint8_t>(1),false,uint32_t(0),gm_red,rsvd_cnt);
            AscendC::Sqrt(distance,distance,count);
            if constexpr(sizeof(T)==2) {
                AscendC::Cast(packed,distance,AscendC::RoundMode::CAST_RINT,count);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopyPad(output_index[batch*c.m+first],indices,store_index);
            fence<AscendC::HardEvent::V_MTE3>();
            if constexpr(sizeof(T)==2)AscendC::DataCopyPad(output[batch*c.m+first],packed,store);
            else AscendC::DataCopyPad(output[batch*c.m+first],distance,store);
        }
    }
private:
    template<AscendC::HardEvent Event>ELEC_FN void fence(){AscendC::SetFlag<Event>(0);AscendC::WaitFlag<Event>(0);}
};
}
