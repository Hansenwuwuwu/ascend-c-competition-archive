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
template<typename T,uint32_t points_per_tile=16>
class TreePairs {
public:
    ELEC_FN void run(const Config& c,GM_ADDR points_address,GM_ADDR segments_address,
                     GM_ADDR distance_address,GM_ADDR index_address) {
        constexpr uint32_t segment_width=64,point_stride=512;
        constexpr uint32_t seg_stride=1024;
        constexpr uint32_t seg_planes=9;
        constexpr uint32_t offset_count=1024+2*points_per_tile;
        // Tier 0 (0 .. 64 KB): ONLY segment & point inputs + small metadata!
        //   soa:      256 .. 37120 (36 KB, Odd 256 mod 512)
        //   raw/brc:  37376 .. 43520 (6 KB, Even 0 mod 512)
        //   point:    43520 .. 44288 (768 B)
        //   offsets:  44288 .. 48896 (4.5 KB)
        //   result:   49152 .. 51200 (2 KB)
        //   mask:     51200 .. 51456 (256 B)
        // Tier 1 (64 .. 128 KB): 4 working planes (dx, r0 Even; dy, dz Odd)
        //   dx:       65536 .. 81920 (16 KB, Even 0 mod 512, Tier 1)
        //   r0:       81920 .. 98304 (16 KB, Even 0 mod 512, Tier 1)
        //   dy:       98560 .. 114944 (16 KB, Odd 256 mod 512, Tier 1)
        //   dz:       114944 .. 131328 (16 KB, Odd 256 mod 512, Tier 1)
        // Tier 2 (128 .. 192 KB): 3 working planes (p0, p1 Even; q0 Odd)
        //   p0:       131584 .. 147968 (16 KB, Even 0 mod 512, Tier 2)
        //   p1:       147968 .. 164352 (16 KB, Even 0 mod 512, Tier 2)
        //   q0:       164608 .. 180992 (16 KB, Odd 256 mod 512, Tier 2)
        // Tier 0 (0 .. 64 KB): soa, raw/broadcast, point, offsets, reduced_all, result, mask
        // Tier 1 (64 .. 128 KB): 100% EVEN (0 mod 512), 4 x 16 KB (dx, dy, dz, p0)
        // Tier 2 (128 .. 192 KB): 100% ODD (256 mod 512), 3 x 16 KB (r0, p1, q0)
        constexpr uint32_t seg_at=256;
        constexpr uint32_t raw_at=37376;
        constexpr uint32_t point_at=43520;
        constexpr uint32_t offset_at=44288;
        constexpr uint32_t red_all_at=49152;
        constexpr uint32_t result_at=57344;
        constexpr uint32_t mask_at=59392;
        constexpr uint32_t pt_raw_at=60416;
        constexpr uint32_t pt_f32_at=61440;
        constexpr uint32_t dx_at=65536;
        constexpr uint32_t dy_at=81920;
        constexpr uint32_t dz_at=98304;
        constexpr uint32_t p0_at=114688;
        constexpr uint32_t r0_at=131328;
        constexpr uint32_t p1_at=147712;
        constexpr uint32_t q0_at=164096;

        AscendC::LocalTensor<T> raw(AscendC::TPosition::VECCALC,raw_at,3*point_stride*(sizeof(float)/sizeof(T)));
        AscendC::LocalTensor<T> point_raw(AscendC::TPosition::VECCALC,pt_raw_at,192);
        AscendC::LocalTensor<float> point_f32(AscendC::TPosition::VECCALC,pt_f32_at,192);
        AscendC::LocalTensor<float> soa(AscendC::TPosition::VECCALC,seg_at,seg_planes*seg_stride);
        AscendC::LocalTensor<float> r0(AscendC::TPosition::VECCALC,r0_at,4096);
        AscendC::LocalTensor<float> dx(AscendC::TPosition::VECCALC,dx_at,4096);
        AscendC::LocalTensor<float> dy(AscendC::TPosition::VECCALC,dy_at,4096);
        AscendC::LocalTensor<float> dz(AscendC::TPosition::VECCALC,dz_at,4096);
        AscendC::LocalTensor<float> p0(AscendC::TPosition::VECCALC,p0_at,4096);
        AscendC::LocalTensor<float> p1(AscendC::TPosition::VECCALC,p1_at,4096);
        AscendC::LocalTensor<float> q0(AscendC::TPosition::VECCALC,q0_at,4096);
        AscendC::LocalTensor<float> point(AscendC::TPosition::VECCALC,point_at,192);
        AscendC::LocalTensor<int32_t> offsets(AscendC::TPosition::VECCALC,offset_at,offset_count);
        AscendC::LocalTensor<float> result(AscendC::TPosition::VECCALC,result_at,512);
        AscendC::LocalTensor<uint8_t> mask(AscendC::TPosition::VECCALC,mask_at,512);
        AscendC::LocalTensor<float> reduced_all(AscendC::TPosition::VECCALC,red_all_at,2048);

        AscendC::LocalTensor<T> seg_raw(AscendC::TPosition::VECCALC,131072,8192*(sizeof(float)/sizeof(T)));
        AscendC::LocalTensor<float> segments(AscendC::TPosition::VECCALC,dx_at,8192);
        auto ax=soa[0*seg_stride],ay=soa[1*seg_stride],az=soa[2*seg_stride],ux=soa[3*seg_stride],uy=soa[4*seg_stride],uz=soa[5*seg_stride];
        auto vx=soa[6*seg_stride],vy=soa[7*seg_stride],vz=soa[8*seg_stride];
        auto reduced=result,candidate=result[128],best=result[256],distance=result[384];
        auto candidate_index=result[192].template ReinterpretCast<int32_t>(),best_index=result[320].template ReinterpretCast<int32_t>();
        auto packed=result[448].template ReinterpretCast<T>();

        AscendC::GlobalTensor<T> input_points,input_segments,output;
        AscendC::GlobalTensor<int32_t> output_index;
        input_points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
        input_segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
        output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));

        const uint64_t per_batch=(c.m+points_per_tile-1)/points_per_tile;
        const uint64_t block_idx=AscendC::GetBlockIdx(),block_num=AscendC::GetBlockNum();
        const bool batch_pinned=(c.m>=512 && c.batch>1 && block_num>=c.batch && (block_num%c.batch)==0);
        const uint64_t cores_per_batch=batch_pinned?(block_num/c.batch):1;
        const uint64_t pinned_batch=batch_pinned?(block_idx/cores_per_batch):0;
        const uint64_t cid=batch_pinned?(block_idx%cores_per_batch):0;
        const uint64_t wave_cap=cores_per_batch*points_per_tile;
        const uint64_t full_waves=batch_pinned?(c.m/wave_cap):0;
        const uint64_t rem_points=batch_pinned?(c.m-full_waves*wave_cap):0;
        const uint64_t tail_tile=batch_pinned?((rem_points+cores_per_batch-1)/cores_per_batch):points_per_tile;
        const uint64_t task_start=batch_pinned?0:block_idx;
        const uint64_t task_limit=batch_pinned?(full_waves+(rem_points>0?1:0)):(c.batch*per_batch);
        if(!batch_pinned&&block_idx>=task_limit)return;
        const uint64_t task_step=batch_pinned?1:block_num;
        uint64_t cached_batch=c.batch;
        bool offsets_ready=false;
        const AscendC::DataCopyPadExtParams<T> padding{false,0,0,0};
        const bool has_tail1=(c.n>64)&&((c.n&63)==1);
        const uint32_t n_pow2=has_tail1?static_cast<uint32_t>(c.n-1):static_cast<uint32_t>(c.n);
        const uint32_t n_aligned=(static_cast<uint32_t>(c.n)+63u)&~63u;
        const uint8_t n_rep=static_cast<uint8_t>(n_aligned>>6);
        auto seg_brc=reduced_all[1280];
        const AscendC::BinaryRepeatParams point_segment{1,0,1,8,1,0},segment{1,1,1,8,8,0},contig{1,1,1,8,8,8};
        const AscendC::UnaryRepeatParams ucontig{1,1,8,8};

        for(uint64_t task=task_start;task<task_limit;task+=task_step) {
            const uint64_t batch=batch_pinned?pinned_batch:(task/per_batch);
            uint64_t first=0;
            uint32_t count=0;
            if(batch_pinned) {
                if(task<full_waves) {
                    first=(task*cores_per_batch+cid)*points_per_tile;
                    count=points_per_tile;
                } else {
                    first=full_waves*wave_cap+cid*tail_tile;
                    if(first>=c.m) break;
                    count=static_cast<uint32_t>(c.m-first<tail_tile?c.m-first:tail_tile);
                }
            } else {
                first=(task%per_batch)*points_per_tile;
                count=static_cast<uint32_t>(c.m-first<points_per_tile?c.m-first:points_per_tile);
            }
            const AscendC::DataCopyExtParams load_points{1,count*3*uint32_t(sizeof(T)),0,0,0};
            if(batch!=cached_batch) {
                if(offsets_ready) {AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);fence<AscendC::HardEvent::V_MTE2>();}
                const AscendC::DataCopyExtParams load{1,static_cast<uint32_t>(c.n*6*sizeof(T)),0,0,0};
                AscendC::DataCopyPad(seg_raw,input_segments[batch*c.n*6],load,padding);
                AscendC::DataCopyPad(point_raw,input_points[(batch*c.m+first)*3],load_points,padding);
                if(!offsets_ready) {
                    for(uint32_t i=0;i<8;++i) {
                        offsets.SetValue(i,static_cast<int32_t>(i)*24);
                        offsets.SetValue(1024+i,static_cast<int32_t>(i)*12);
                    }
                    fence<AscendC::HardEvent::S_V>();
                    if(has_tail1) {
                        AscendC::Duplicate(offsets[n_pow2],static_cast<int32_t>(n_pow2)*24,64);
                    }
                    const uint32_t max_n=n_pow2>points_per_tile?n_pow2:points_per_tile;
                    for(uint32_t filled=8;filled<max_n;filled*=2) {
                        if(filled<n_pow2) {
                            AscendC::Adds(offsets[filled],offsets,static_cast<int32_t>(filled)*24,filled);
                        }
                        if(filled<points_per_tile) {
                            AscendC::Adds(offsets[1024+filled],offsets[1024],static_cast<int32_t>(filled)*12,filled);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    offsets_ready=true;
                }
                fence<AscendC::HardEvent::MTE2_V>();
                if constexpr(sizeof(T)==2)AscendC::Cast(segments,seg_raw,AscendC::RoundMode::CAST_NONE,c.n*6);
                for(uint32_t axis=0;axis<6;++axis) {
                    if constexpr(sizeof(T)==2)
                        AscendC::Gather(soa[axis*seg_stride],segments[axis],offsets.template ReinterpretCast<uint32_t>(),0,n_aligned);
                    else
                        AscendC::Gather(soa[axis*seg_stride],seg_raw.template ReinterpretCast<float>()[axis],offsets.template ReinterpretCast<uint32_t>(),0,n_aligned);
                }
                AscendC::Sub<float,true>(ux,ux,ax,uint64_t(64),n_rep,contig);
                AscendC::Sub<float,false>(uy,uy,ay,uint64_t(64),n_rep,contig);
                AscendC::Sub<float,false>(uz,uz,az,uint64_t(64),n_rep,contig);
                AscendC::Mul<float,false>(vx,ux,ux,uint64_t(64),n_rep,contig);
                AscendC::Mul<float,false>(r0,uy,uy,uint64_t(64),n_rep,contig);
                AscendC::Mul<float,false>(q0,uz,uz,uint64_t(64),n_rep,contig);
                AscendC::Add<float,false>(vx,vx,r0,uint64_t(64),n_rep,contig);
                AscendC::Add<float,false>(vx,vx,q0,uint64_t(64),n_rep,contig);
                AscendC::Maxs<float,false>(vx,vx,bits_float(0x00800000u),uint64_t(64),n_rep,ucontig);
                AscendC::Div<float,false>(vy,uy,vx,uint64_t(64),n_rep,contig);
                AscendC::Div<float,false>(vz,uz,vx,uint64_t(64),n_rep,contig);
                AscendC::Div<float,false>(vx,ux,vx,uint64_t(64),n_rep,contig);
                if(has_tail1) {
                    const uint32_t last_seg=static_cast<uint32_t>(c.n-1);
                    AscendC::Brcb(seg_brc,ax[last_seg],1,{1,8});
                    AscendC::Brcb(seg_brc[64],ay[last_seg],1,{1,8});
                    AscendC::Brcb(seg_brc[128],az[last_seg],1,{1,8});
                    AscendC::Brcb(seg_brc[192],ux[last_seg],1,{1,8});
                    AscendC::Brcb(seg_brc[256],uy[last_seg],1,{1,8});
                    AscendC::Brcb(seg_brc[320],uz[last_seg],1,{1,8});
                    AscendC::Brcb(seg_brc[384],vx[last_seg],1,{1,8});
                    AscendC::Brcb(seg_brc[448],vy[last_seg],1,{1,8});
                    AscendC::Brcb(seg_brc[512],vz[last_seg],1,{1,8});
                }
                cached_batch=batch;
            } else {
                if(task!=task_start) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::DataCopyPad(point_raw,input_points[(batch*c.m+first)*3],load_points,padding);
                fence<AscendC::HardEvent::MTE2_V>();
            }
            if constexpr(sizeof(T)==2) {
                AscendC::Cast(point_f32,point_raw,AscendC::RoundMode::CAST_NONE,count*3);
                AscendC::Gather(point,point_f32,offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
                AscendC::Gather(point[64],point_f32[1],offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
                AscendC::Gather(point[128],point_f32[2],offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
            } else {
                auto point_values=point_raw.template ReinterpretCast<float>();
                AscendC::Gather(point,point_values,offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
                AscendC::Gather(point[64],point_values[1],offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
                AscendC::Gather(point[128],point_values[2],offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
            }
            const bool has_next=(task+task_step<task_limit)&&(!batch_pinned||(task+1<full_waves)||(full_waves*wave_cap+cid*tail_tile<c.m));
            if(has_next) AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            auto broadcast=raw.template ReinterpretCast<float>();
            AscendC::Brcb(broadcast,point,24,{1,8});
            for(uint32_t start=0;start<c.n;start+=segment_width) {
                const uint32_t valid=static_cast<uint32_t>(c.n-start<segment_width?c.n-start:segment_width);
                if(valid==1&&start>0) break;
                if(start==0) {
                    AscendC::Sub<float,true>(dx,broadcast,ax[start],uint64_t(64),static_cast<uint8_t>(count),point_segment);
                } else {
                    AscendC::Sub<float,false>(dx,broadcast,ax[start],uint64_t(64),static_cast<uint8_t>(count),point_segment);
                }
                AscendC::Sub<float,false>(dy,broadcast[point_stride],ay[start],uint64_t(64),static_cast<uint8_t>(count),point_segment);
                AscendC::Sub<float,false>(dz,broadcast[2*point_stride],az[start],uint64_t(64),static_cast<uint8_t>(count),point_segment);
                AscendC::Mul<float,false>(p0,dx,vx[start],uint64_t(64),static_cast<uint8_t>(count),segment);
                AscendC::Mul<float,false>(r0,dy,vy[start],uint64_t(64),static_cast<uint8_t>(count),segment);
                AscendC::Mul<float,false>(p1,dz,vz[start],uint64_t(64),static_cast<uint8_t>(count),segment);
                AscendC::Add<float,false>(p0,p0,r0,uint64_t(64),static_cast<uint8_t>(count),contig);
                AscendC::Add<float,false>(p0,p0,p1,uint64_t(64),static_cast<uint8_t>(count),contig);
                AscendC::Maxs<float,false>(q0,p0,0.0f,uint64_t(64),static_cast<uint8_t>(count),ucontig);
                AscendC::Mins<float,false>(p0,q0,1.0f,uint64_t(64),static_cast<uint8_t>(count),ucontig);
                AscendC::Mul<float,false>(r0,p0,ux[start],uint64_t(64),static_cast<uint8_t>(count),segment);
                AscendC::Mul<float,false>(p1,p0,uy[start],uint64_t(64),static_cast<uint8_t>(count),segment);
                AscendC::Mul<float,false>(q0,p0,uz[start],uint64_t(64),static_cast<uint8_t>(count),segment);
                AscendC::Sub<float,false>(dx,dx,r0,uint64_t(64),static_cast<uint8_t>(count),contig);
                AscendC::Sub<float,false>(dy,dy,p1,uint64_t(64),static_cast<uint8_t>(count),contig);
                AscendC::Sub<float,false>(dz,dz,q0,uint64_t(64),static_cast<uint8_t>(count),contig);
                AscendC::Mul<float,false>(dx,dx,dx,uint64_t(64),static_cast<uint8_t>(count),contig);
                AscendC::Mul<float,false>(r0,dy,dy,uint64_t(64),static_cast<uint8_t>(count),contig);
                AscendC::Mul<float,false>(p1,dz,dz,uint64_t(64),static_cast<uint8_t>(count),contig);
                AscendC::Add<float,false>(dx,dx,r0,uint64_t(64),static_cast<uint8_t>(count),contig);
                AscendC::Add<float,false>(p0,dx,p1,uint64_t(64),static_cast<uint8_t>(count),contig);
                auto red_dst=reduced_all[(start>>6)*128];
                if(valid==64) {
                    AscendC::WholeReduceMin<float,false>(red_dst,p0,int32_t(64),static_cast<int32_t>(count),1,1,8);
                } else {
                    AscendC::WholeReduceMin<float,true>(red_dst,p0,static_cast<int32_t>(valid),static_cast<int32_t>(count),1,1,8);
                }
            }
            const uint32_t num_chunks=static_cast<uint32_t>((c.n+63)>>6);
            const uint32_t gm_chunks=has_tail1?(num_chunks-1):num_chunks;
            uint64_t rsvd_cnt=0;
            const AscendC::GatherMaskParams gm_params{1,static_cast<uint16_t>(gm_chunks*2),8,0};
            auto dy_i32=dy.template ReinterpretCast<int32_t>();
            if(task!=task_start) fence<AscendC::HardEvent::MTE3_V>();
            auto idx_dst=(num_chunks==1)?best_index:dy_i32;
            AscendC::GatherMask(idx_dst,reduced_all.template ReinterpretCast<int32_t>(),static_cast<uint8_t>(2),false,uint32_t(0),gm_params,rsvd_cnt);
            AscendC::GatherMask(dx,reduced_all,static_cast<uint8_t>(1),false,uint32_t(0),gm_params,rsvd_cnt);
            if(num_chunks>1) {
                for(uint32_t c_idx=1;c_idx<gm_chunks;++c_idx) {
                    if(c_idx==1) AscendC::Adds<int32_t,true>(dy_i32[c_idx*64],dy_i32[c_idx*64],static_cast<int32_t>(c_idx*64),uint64_t(64),1,ucontig);
                    else AscendC::Adds<int32_t,false>(dy_i32[c_idx*64],dy_i32[c_idx*64],static_cast<int32_t>(c_idx*64),uint64_t(64),1,ucontig);
                }
                if(has_tail1) {
                    const uint32_t start=static_cast<uint32_t>(c.n-1);
                    const uint32_t last_off=gm_chunks*64;
                    const AscendC::BinaryRepeatParams one_seg{1,1,0,8,8,0};
                    AscendC::Sub<float,true>(dz,point,seg_brc,uint64_t(64),1,one_seg);
                    AscendC::Sub<float,false>(dz[64],point[64],seg_brc[64],uint64_t(64),1,one_seg);
                    AscendC::Sub<float,false>(dz[128],point[128],seg_brc[128],uint64_t(64),1,one_seg);
                    AscendC::Mul<float,false>(p0,dz,seg_brc[384],uint64_t(64),1,one_seg);
                    AscendC::Mul<float,false>(r0,dz[64],seg_brc[448],uint64_t(64),1,one_seg);
                    AscendC::Mul<float,false>(p1,dz[128],seg_brc[512],uint64_t(64),1,one_seg);
                    AscendC::Add<float,false>(p0,p0,r0,uint64_t(64),1,contig);
                    AscendC::Add<float,false>(p0,p0,p1,uint64_t(64),1,contig);
                    AscendC::Maxs<float,false>(q0,p0,0.0f,uint64_t(64),1,ucontig);
                    AscendC::Mins<float,false>(p0,q0,1.0f,uint64_t(64),1,ucontig);
                    AscendC::Mul<float,false>(r0,p0,seg_brc[192],uint64_t(64),1,one_seg);
                    AscendC::Mul<float,false>(p1,p0,seg_brc[256],uint64_t(64),1,one_seg);
                    AscendC::Mul<float,false>(q0,p0,seg_brc[320],uint64_t(64),1,one_seg);
                    AscendC::Sub<float,false>(dz,dz,r0,uint64_t(64),1,contig);
                    AscendC::Sub<float,false>(dz[64],dz[64],p1,uint64_t(64),1,contig);
                    AscendC::Sub<float,false>(dz[128],dz[128],q0,uint64_t(64),1,contig);
                    AscendC::Mul<float,false>(dz,dz,dz,uint64_t(64),1,contig);
                    AscendC::Mul<float,false>(r0,dz[64],dz[64],uint64_t(64),1,contig);
                    AscendC::Mul<float,false>(p1,dz[128],dz[128],uint64_t(64),1,contig);
                    AscendC::Add<float,false>(dz,dz,r0,uint64_t(64),1,contig);
                    AscendC::Add<float,false>(dx[last_off],dz,p1,uint64_t(64),1,contig);
                    AscendC::Duplicate<int32_t,false>(dy_i32[last_off],static_cast<int32_t>(start),uint64_t(64),1,1,8);
                }
                auto best_idx_f32=best_index.template ReinterpretCast<float>();
                for(uint32_t c_idx=1;c_idx<num_chunks;++c_idx) {
                    AscendC::Compare<float,false>(dx[c_idx*64],dx,AscendC::CMPMODE::LT,uint64_t(64),contig);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SelectCal<float,AscendC::SELMODE::VSEL_CMPMASK_SPR>(reinterpret_cast<__ubuf__ float*>(dx.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(dx[c_idx*64].GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(dx.GetPhyAddr()),1,contig);
                    auto dst_idx=(c_idx+1==num_chunks)?best_idx_f32:dy;
                    AscendC::SelectCal<float,AscendC::SELMODE::VSEL_CMPMASK_SPR>(reinterpret_cast<__ubuf__ float*>(dst_idx.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(dy[c_idx*64].GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(dy.GetPhyAddr()),1,contig);
                }
            }
            const AscendC::DataCopyExtParams store{1,count*uint32_t(sizeof(T)),0,0,0},store_index{1,count*4,0,0,0};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::Sqrt(distance,dx,count);
            if constexpr(sizeof(T)==2) {
                AscendC::Cast(packed,distance,AscendC::RoundMode::CAST_RINT,count);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopyPad(output_index[batch*c.m+first],best_index,store_index);
            fence<AscendC::HardEvent::V_MTE3>();
            if constexpr(sizeof(T)==2) AscendC::DataCopyPad(output[batch*c.m+first],packed,store);
            else AscendC::DataCopyPad(output[batch*c.m+first],distance,store);
        }
    }

    ELEC_FN void run_1024(const Config& c,GM_ADDR points_address,GM_ADDR segments_address,
                          GM_ADDR distance_address,GM_ADDR index_address) {
        constexpr uint32_t point_stride=512;
        constexpr uint32_t seg_stride=1024;
        constexpr uint32_t seg_planes=9;
        constexpr uint32_t offset_count=1024+2*points_per_tile;
        // Tier 0 (0 .. 64 KB):
        //   soa (9*1024 f32 = 36 KB):     256 .. 37120 (Odd 256 mod 512)
        //   soa_f16 (9*1024 f16 = 18 KB): 37120 .. 55552 (Odd 256 mod 512)
        //   raw/brc (6 KB):               55808 .. 61952 (Even 0 mod 512)
        //   point (768 B):                61952 .. 62720
        //   point_f16 (384 B):            62720 .. 63104
        //   point_raw (768 B):            63232 .. 64000
        //   point_f32 (768 B):            64000 .. 64768
        //   mask (512 B):                 64768 .. 65280
        //   ramp_f16 (256 B):             65280 .. 65536 (Odd 256 mod 512)
        // Tier 1 (64 .. 128 KB): 100% Even (0 mod 512), 4 x 16 KB (dx, dy, dz, p0)
        // Tier 2 (128 .. 192 KB): 3 x 16 KB Odd (256 mod 512) (r0, p1, q0) + metadata
        constexpr uint32_t seg_at=256;
        constexpr uint32_t red_f16_at=seg_at+7*seg_stride*4;
        constexpr uint32_t soa_f16_at=37120;
        constexpr uint32_t raw_at=55808;
        constexpr uint32_t point_at=61952;
        constexpr uint32_t pt_f16_at=62720;
        constexpr uint32_t pt_raw_at=63232;
        constexpr uint32_t pt_f32_at=64000;
        constexpr uint32_t ramp_at=65280;
        constexpr uint32_t dx_at=65536;
        constexpr uint32_t dy_at=81920;
        constexpr uint32_t dz_at=98304;
        constexpr uint32_t p0_at=114688;
        constexpr uint32_t r0_at=131328;
        constexpr uint32_t p1_at=147712;
        constexpr uint32_t q0_at=164096;
        constexpr uint32_t offset_at=180480;
        constexpr uint32_t result_at=185088;
        constexpr uint32_t mask1k_at=187136;
        // Skip TMP_UB_OFFSET = 184 * 1024 = 188416
        constexpr uint32_t brc_c1_at=193024;
        constexpr uint32_t c1_norm_at=195072;
        constexpr uint32_t inf_at=195328;

        AscendC::LocalTensor<float> soa(AscendC::TPosition::VECCALC,seg_at,seg_planes*seg_stride);
        AscendC::LocalTensor<half> soa_f16(AscendC::TPosition::VECCALC,soa_f16_at,seg_planes*seg_stride);
        AscendC::LocalTensor<T> raw(AscendC::TPosition::VECCALC,raw_at,3*point_stride*(sizeof(float)/sizeof(T)));
        AscendC::LocalTensor<float> point(AscendC::TPosition::VECCALC,point_at,192);
        AscendC::LocalTensor<half> point_f16(AscendC::TPosition::VECCALC,pt_f16_at,192);
        AscendC::LocalTensor<T> point_raw(AscendC::TPosition::VECCALC,pt_raw_at,192);
        AscendC::LocalTensor<float> point_f32(AscendC::TPosition::VECCALC,pt_f32_at,192);
        AscendC::LocalTensor<uint8_t> mask(AscendC::TPosition::VECCALC,mask1k_at,1024);
        AscendC::LocalTensor<half> ramp_f16(AscendC::TPosition::VECCALC,ramp_at,128);
        AscendC::LocalTensor<int32_t> offsets(AscendC::TPosition::VECCALC,offset_at,offset_count);
        AscendC::LocalTensor<half> red_f16(AscendC::TPosition::VECCALC,red_f16_at,3072);
        AscendC::LocalTensor<float> result(AscendC::TPosition::VECCALC,result_at,512);
        AscendC::LocalTensor<half> c1_norm(AscendC::TPosition::VECCALC,c1_norm_at,64);
        AscendC::LocalTensor<half> brc_c1(AscendC::TPosition::VECCALC,brc_c1_at,1024);
        AscendC::LocalTensor<half> inf_f16(AscendC::TPosition::VECCALC,inf_at,128);

        AscendC::LocalTensor<float> dx(AscendC::TPosition::VECCALC,dx_at,4096);
        AscendC::LocalTensor<float> dy(AscendC::TPosition::VECCALC,dy_at,4096);
        AscendC::LocalTensor<float> dz(AscendC::TPosition::VECCALC,dz_at,4096);
        AscendC::LocalTensor<float> p0(AscendC::TPosition::VECCALC,p0_at,4096);
        AscendC::LocalTensor<float> r0(AscendC::TPosition::VECCALC,r0_at,4096);
        AscendC::LocalTensor<float> p1(AscendC::TPosition::VECCALC,p1_at,4096);
        AscendC::LocalTensor<float> q0(AscendC::TPosition::VECCALC,q0_at,4096);

        AscendC::LocalTensor<half> dx_f16(AscendC::TPosition::VECCALC,dx_at,8192);
        AscendC::LocalTensor<half> dy_f16(AscendC::TPosition::VECCALC,dy_at,8192);
        AscendC::LocalTensor<half> dz_f16(AscendC::TPosition::VECCALC,dz_at,8192);
        AscendC::LocalTensor<half> p0_f16(AscendC::TPosition::VECCALC,p0_at,8192);
        AscendC::LocalTensor<half> r0_f16(AscendC::TPosition::VECCALC,r0_at,8192);
        AscendC::LocalTensor<half> p1_f16(AscendC::TPosition::VECCALC,p1_at,8192);
        AscendC::LocalTensor<half> q0_f16(AscendC::TPosition::VECCALC,q0_at,8192);

        AscendC::LocalTensor<T> seg_raw(AscendC::TPosition::VECCALC,131072,6144);
        AscendC::LocalTensor<float> segments(AscendC::TPosition::VECCALC,dx_at,8192);
        auto ax=soa[0*seg_stride],ay=soa[1*seg_stride],az=soa[2*seg_stride],ux=soa[3*seg_stride],uy=soa[4*seg_stride],uz=soa[5*seg_stride],vx=soa[6*seg_stride],vy=soa[7*seg_stride],vz=soa[8*seg_stride];
        auto ax_f16=soa_f16[0*seg_stride],ay_f16=soa_f16[1*seg_stride],az_f16=soa_f16[2*seg_stride],ux_f16=soa_f16[3*seg_stride],uy_f16=soa_f16[4*seg_stride],uz_f16=soa_f16[5*seg_stride],vx_f16=soa_f16[6*seg_stride],vy_f16=soa_f16[7*seg_stride],vz_f16=soa_f16[8*seg_stride];
        auto best_index=result[320].template ReinterpretCast<int32_t>();
        auto distance=result[384];
        auto packed=result[448].template ReinterpretCast<T>();

        AscendC::GlobalTensor<T> input_points,input_segments,output;
        AscendC::GlobalTensor<int32_t> output_index;
        input_points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
        input_segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
        output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));

        const uint64_t block_idx=AscendC::GetBlockIdx(),block_num=AscendC::GetBlockNum();
        const uint64_t cores_per_batch=block_num/c.batch;
        const uint64_t batch=block_idx/cores_per_batch;
        const uint64_t cid=block_idx%cores_per_batch;
        const uint64_t wave_cap=cores_per_batch*points_per_tile;
        const uint64_t full_waves=c.m/wave_cap;
        const uint64_t rem_points=c.m-full_waves*wave_cap;
        const uint64_t tail_tile=(rem_points+cores_per_batch-1)/cores_per_batch;
        const uint64_t task_limit=full_waves+(rem_points>0?1:0);
        const AscendC::DataCopyPadExtParams<T> padding{false,0,0,0};

        // Precompute segment planes once per core (pinned to batch) and prefetch task 0 points
        const AscendC::DataCopyExtParams load{1,static_cast<uint32_t>(1024*6*sizeof(T)),0,0,0};
        AscendC::DataCopyPad(seg_raw,input_segments[batch*1024*6],load,padding);
        const uint64_t init_first=(0<full_waves)?(cid*points_per_tile):(cid*tail_tile);
        const uint32_t init_count=(0<full_waves)?points_per_tile:static_cast<uint32_t>(c.m-init_first<tail_tile?c.m-init_first:tail_tile);
        if(init_first<c.m) {
            const AscendC::DataCopyExtParams init_load_pts{1,init_count*3*uint32_t(sizeof(T)),0,0,0};
            AscendC::DataCopyPad(point_raw,input_points[(batch*c.m+init_first)*3],init_load_pts,padding);
        }
        auto ramp_i16=ramp_f16.template ReinterpretCast<int16_t>();
        for(uint32_t i=0;i<8;++i) {
            offsets.SetValue(i,static_cast<int32_t>(i)*24);
            offsets.SetValue(1024+i,static_cast<int32_t>(i)*12);
        }
        for(uint32_t i=0;i<16;++i) {
            ramp_i16.SetValue(i,static_cast<int16_t>(0x3800+i));
        }
        auto mask_addr_ub=result.template ReinterpretCast<uint32_t>();
        const uint32_t mask_sel_addr=static_cast<uint32_t>(reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(mask.GetPhyAddr())));
        fence<AscendC::HardEvent::S_V>();
        AscendC::Duplicate(mask_addr_ub,mask_sel_addr,8);
        AscendC::Duplicate(inf_f16.template ReinterpretCast<uint16_t>(),uint16_t(0x7bffu),128);
        for(uint32_t filled=8;filled<1024;filled*=2) {
            AscendC::Adds(offsets[filled],offsets,static_cast<int32_t>(filled)*24,filled);
            if(filled<points_per_tile) {
                AscendC::Adds(offsets[1024+filled],offsets[1024],static_cast<int32_t>(filled)*12,filled);
            }
            if(filled>=16&&filled<128) {
                AscendC::Adds(ramp_i16[filled],ramp_i16,static_cast<int16_t>(filled),filled);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        fence<AscendC::HardEvent::MTE2_V>();
        if constexpr(sizeof(T)==2) AscendC::Cast(segments,seg_raw,AscendC::RoundMode::CAST_NONE,1024*6);
        for(uint32_t axis=0;axis<6;++axis) {
            if constexpr(sizeof(T)==2)
                AscendC::Gather(soa[axis*seg_stride],segments[axis],offsets.template ReinterpretCast<uint32_t>(),0,1024);
            else
                AscendC::Gather(soa[axis*seg_stride],seg_raw.template ReinterpretCast<float>()[axis],offsets.template ReinterpretCast<uint32_t>(),0,1024);
        }
        AscendC::Sub(ux,ux,ax,3072);
        AscendC::Mul(vx,ux,ux,3072);
        AscendC::Add(vx,vx,vy,1024);
        AscendC::Add(vx,vx,vz,1024);
        AscendC::Maxs(vx,vx,bits_float(0x00800000u),1024);
        AscendC::Duplicate(red_f16.template ReinterpretCast<int16_t>(),int16_t(0),3072);
        AscendC::Duplicate(point,0.0f,192);
        AscendC::Duplicate(c1_norm.template ReinterpretCast<int16_t>(),int16_t(0x3800),64);

        // Compute dynamic power-of-2 coordinate scale so Stage 1 FP16 never overflows 65504
        AscendC::Abs(dx,ax,3072);
        AscendC::Abs(dy,ux,3072);
        AscendC::Max(dx,dx,dy,3072);
        AscendC::Max(dx,dx,dx[2048],1024);
        AscendC::Max(dx,dx,dx[1024],1024);
        AscendC::WholeReduceMax<float>(r0,dx,64,16,1,1,8);
        AscendC::WholeReduceMax<float>(r0,r0,32,1,1,1,8);
        AscendC::ResetMask();
        fence<AscendC::HardEvent::V_S>();
        float max_coord=r0.GetValue(0);
        if(!(max_coord>1e-6f)) max_coord=1.0f;
        uint32_t exp_bits=float_bits(max_coord)&0x7f800000u;
        if(exp_bits<0x1f800000u) exp_bits=0x1f800000u;
        if(exp_bits>0x5f800000u) exp_bits=0x5f800000u;
        const float coord_scale=bits_float((254u-(exp_bits>>23))<<23);
        fence<AscendC::HardEvent::S_V>();

        AscendC::Muls(dx,ax,coord_scale,3072);
        AscendC::Cast(ax_f16,dx,AscendC::RoundMode::CAST_RINT,3072);
        AscendC::Muls(dx,ux,coord_scale,3072);
        AscendC::Cast(ux_f16,dx,AscendC::RoundMode::CAST_RINT,3072);
        AscendC::Mul(r0,dx,dx,1024);
        AscendC::Mul(p1,dx[1024],dx[1024],1024);
        AscendC::Add(r0,r0,p1,1024);
        AscendC::Mul(p1,dx[2048],dx[2048],1024);
        AscendC::Add(r0,r0,p1,1024);
        AscendC::Maxs(r0,r0,bits_float(0x00800000u),1024);
        AscendC::Div(dx,dx,r0,1024);
        AscendC::Div(dx[1024],dx[1024],r0,1024);
        AscendC::Div(dx[2048],dx[2048],r0,1024);
        AscendC::Maxs(dx,dx,-65504.0f,3072);
        AscendC::Mins(dx,dx,65504.0f,3072);
        AscendC::Cast(vx_f16,dx,AscendC::RoundMode::CAST_RINT,3072);

        const AscendC::BinaryRepeatParams point_segment{1,0,1,8,1,0},segment{1,1,1,8,8,0},contig{1,1,1,8,8,8};
        const AscendC::BinaryRepeatParams pt_rows{1,1,1,8,0,8};
        const AscendC::UnaryRepeatParams ucontig{1,1,8,8},ustep16{1,1,16,16};
        const AscendC::GatherMaskParams gm2{1,2,8,0},gm16{1,16,8,0};
        auto brc_f16=raw.template ReinterpretCast<half>();
        auto p0_i16=p0_f16.template ReinterpretCast<int16_t>();
        auto r0_i16=r0_f16.template ReinterpretCast<int16_t>();
        auto p1_i32=p1.template ReinterpretCast<int32_t>();
        auto cand_off=p1.template ReinterpretCast<uint32_t>();
        auto best_idx_f32=best_index.template ReinterpretCast<float>();

        for(uint64_t task=0;task<task_limit;++task) {
            uint64_t first=0;
            uint32_t count=0;
            if(task<full_waves) {
                first=(task*cores_per_batch+cid)*points_per_tile;
                count=points_per_tile;
            } else {
                first=full_waves*wave_cap+cid*tail_tile;
                if(first>=c.m) break;
                count=static_cast<uint32_t>(c.m-first<tail_tile?c.m-first:tail_tile);
            }
            if(task!=0) AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            if constexpr(sizeof(T)==2) {
                AscendC::Cast(point_f32,point_raw,AscendC::RoundMode::CAST_NONE,count*3);
                AscendC::Gather(point,point_f32,offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
                AscendC::Gather(point[64],point_f32[1],offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
                AscendC::Gather(point[128],point_f32[2],offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
            } else {
                auto point_values=point_raw.template ReinterpretCast<float>();
                AscendC::Gather(point,point_values,offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
                AscendC::Gather(point[64],point_values[1],offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
                AscendC::Gather(point[128],point_values[2],offsets[1024].template ReinterpretCast<uint32_t>(),0,count);
            }
            const bool has_next=(task+1<task_limit)&&((task+1<full_waves)||(full_waves*wave_cap+cid*tail_tile<c.m));
            if(has_next) {
                const uint64_t next_task=task+1;
                const uint64_t next_first=(next_task<full_waves)?((next_task*cores_per_batch+cid)*points_per_tile):(full_waves*wave_cap+cid*tail_tile);
                const uint32_t next_count=(next_task<full_waves)?points_per_tile:static_cast<uint32_t>(c.m-next_first<tail_tile?c.m-next_first:tail_tile);
                fence<AscendC::HardEvent::V_MTE2>();
                const AscendC::DataCopyExtParams next_load_pts{1,next_count*3*uint32_t(sizeof(T)),0,0,0};
                AscendC::DataCopyPad(point_raw,input_points[(batch*c.m+next_first)*3],next_load_pts,padding);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            }

            AscendC::Muls(point_f32,point,coord_scale,192);
            AscendC::Cast(point_f16,point_f32,AscendC::RoundMode::CAST_RINT,192);
            AscendC::Brcb(brc_f16,point_f16,24,{1,8});

            // Stage 1: 8-chunk FP16 screening (Top-2 per 128-segment chunk)
            uint64_t rsvd_cnt=0;
            AscendC::AscendCUtils::SetMask<half>(128);
            for(uint32_t c_idx=0;c_idx<8;++c_idx) {
                const uint32_t start=c_idx*128;
                AscendC::Sub<half,false>(dx_f16,brc_f16,ax_f16[start],uint64_t(128),static_cast<uint8_t>(count),point_segment);
                AscendC::Sub<half,false>(dy_f16,brc_f16[1024],ay_f16[start],uint64_t(128),static_cast<uint8_t>(count),point_segment);
                AscendC::Sub<half,false>(dz_f16,brc_f16[2048],az_f16[start],uint64_t(128),static_cast<uint8_t>(count),point_segment);
                AscendC::Mul<half,false>(p0_f16,dx_f16,vx_f16[start],uint64_t(128),static_cast<uint8_t>(count),segment);
                AscendC::Mul<half,false>(r0_f16,dy_f16,vy_f16[start],uint64_t(128),static_cast<uint8_t>(count),segment);
                AscendC::Mul<half,false>(p1_f16,dz_f16,vz_f16[start],uint64_t(128),static_cast<uint8_t>(count),segment);
                AscendC::Add<half,false>(p0_f16,p0_f16,r0_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::Add<half,false>(p0_f16,p0_f16,p1_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::Maxs<half,false>(p0_f16,p0_f16,static_cast<half>(0.0f),uint64_t(128),static_cast<uint8_t>(count),ucontig);
                AscendC::Mins<half,false>(p0_f16,p0_f16,static_cast<half>(1.0f),uint64_t(128),static_cast<uint8_t>(count),ucontig);
                AscendC::Mul<half,false>(r0_f16,p0_f16,ux_f16[start],uint64_t(128),static_cast<uint8_t>(count),segment);
                AscendC::Mul<half,false>(p1_f16,p0_f16,uy_f16[start],uint64_t(128),static_cast<uint8_t>(count),segment);
                AscendC::Mul<half,false>(q0_f16,p0_f16,uz_f16[start],uint64_t(128),static_cast<uint8_t>(count),segment);
                AscendC::Sub<half,false>(dx_f16,dx_f16,r0_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::Sub<half,false>(dy_f16,dy_f16,p1_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::Sub<half,false>(dz_f16,dz_f16,q0_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::Mul<half,false>(p0_f16,dx_f16,dx_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::Mul<half,false>(r0_f16,dy_f16,dy_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::Mul<half,false>(p1_f16,dz_f16,dz_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::Add<half,false>(p0_f16,p0_f16,r0_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::Add<half,false>(p0_f16,p0_f16,p1_f16,uint64_t(128),static_cast<uint8_t>(count),contig);
                AscendC::WholeReduceMin<half,false>(red_f16[(c_idx*2)*128],p0_f16,int32_t(128),static_cast<int32_t>(count),1,1,8);
                AscendC::GatherMask(c1_norm.template ReinterpretCast<uint16_t>(),red_f16[(c_idx*2)*128].template ReinterpretCast<uint16_t>(),static_cast<uint8_t>(2),false,uint32_t(0),gm2,rsvd_cnt);
                AscendC::Adds<int16_t,true>(c1_norm.template ReinterpretCast<int16_t>(),c1_norm.template ReinterpretCast<int16_t>(),static_cast<int16_t>(0x3800),uint64_t(128),1,ucontig);
                AscendC::Brcb(brc_c1,c1_norm,8,{1,8});
                AscendC::Compare<half,uint8_t,false>(mask,brc_c1,ramp_f16,AscendC::CMPMODE::NE,uint64_t(128),static_cast<uint8_t>(count),point_segment);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::VselIntrinsicsImplPre(reinterpret_cast<__ubuf__ uint32_t*>(mask_addr_ub.GetPhyAddr()),AscendC::SELMODE::VSEL_CMPMASK_SPR);
                AscendC::SelectCal<half,AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE>(reinterpret_cast<__ubuf__ half*>(p0_f16.GetPhyAddr()),reinterpret_cast<__ubuf__ half*>(p0_f16.GetPhyAddr()),reinterpret_cast<__ubuf__ half*>(inf_f16.GetPhyAddr()),static_cast<int32_t>(count),segment);
                AscendC::WholeReduceMin<half,false>(red_f16[(c_idx*2+1)*128],p0_f16,int32_t(128),static_cast<int32_t>(count),1,1,8);
            }

            // Stage 2: Sort 2 candidate indices per chunk ascending, then evaluate 16 candidates in exact FP32
            AscendC::GatherMask(p0_f16.template ReinterpretCast<uint16_t>(),red_f16.template ReinterpretCast<uint16_t>(),static_cast<uint8_t>(2),false,uint32_t(0),gm16,rsvd_cnt);
            AscendC::AscendCUtils::SetMask<int16_t>(64);
            AscendC::Min<int16_t,false>(r0_i16,p0_i16,p0_i16[64],uint64_t(64),8,contig);
            AscendC::Max<int16_t,false>(r0_i16[64],p0_i16,p0_i16[64],uint64_t(64),8,contig);
            AscendC::AscendCUtils::SetMask<int16_t>(128);
            AscendC::Adds<int16_t,false>(r0_i16,r0_i16,static_cast<int16_t>(0x3800),uint64_t(128),8,ucontig);
            AscendC::Adds<int16_t,false>(r0_i16[128],r0_i16[128],static_cast<int16_t>(128),uint64_t(128),4,ustep16);
            AscendC::Adds<int16_t,false>(r0_i16[256],r0_i16[256],static_cast<int16_t>(256),uint64_t(128),2,ucontig);
            AscendC::Adds<int16_t,false>(r0_i16[768],r0_i16[768],static_cast<int16_t>(256),uint64_t(128),2,ucontig);
            AscendC::Adds<int16_t,false>(r0_i16[512],r0_i16[512],static_cast<int16_t>(512),uint64_t(128),4,ucontig);
            AscendC::Cast(q0,r0_f16,AscendC::RoundMode::CAST_NONE,1024);
            AscendC::Adds(q0,q0,-0.5f,1024);
            AscendC::Muls(q0,q0,8192.0f,1024);
            AscendC::Cast(p1_i32,q0,AscendC::RoundMode::CAST_RINT,1024);

            AscendC::Gather(dx,ax,cand_off,0,1024);
            AscendC::Gather(dy,ay,cand_off,0,1024);
            AscendC::Gather(dz,az,cand_off,0,1024);
            AscendC::Sub<float,true>(dx,point,dx,uint64_t(64),16,pt_rows);
            AscendC::Sub<float,false>(dy,point[64],dy,uint64_t(64),16,pt_rows);
            AscendC::Sub<float,false>(dz,point[128],dz,uint64_t(64),16,pt_rows);
            AscendC::Gather(r0,ux,cand_off,0,1024);
            AscendC::Gather(q0,uy,cand_off,0,1024);
            AscendC::Gather(r0[1024],uz,cand_off,0,1024);
            AscendC::Gather(q0[1024],vx,cand_off,0,1024);
            AscendC::Mul<float,true>(p0,dx,r0,uint64_t(64),16,contig);
            AscendC::Mul<float,false>(r0[2048],dy,q0,uint64_t(64),16,contig);
            AscendC::Mul<float,false>(q0[2048],dz,r0[1024],uint64_t(64),16,contig);
            AscendC::Add<float,false>(p0,p0,r0[2048],uint64_t(64),16,contig);
            AscendC::Add<float,false>(p0,p0,q0[2048],uint64_t(64),16,contig);
            AscendC::Div<float,false>(p0,p0,q0[1024],uint64_t(64),16,contig);
            AscendC::Maxs<float,false>(p0,p0,0.0f,uint64_t(64),16,ucontig);
            AscendC::Mins<float,false>(p0,p0,1.0f,uint64_t(64),16,ucontig);
            AscendC::Mul<float,false>(r0,p0,r0,uint64_t(64),16,contig);
            AscendC::Mul<float,false>(q0,p0,q0,uint64_t(64),16,contig);
            AscendC::Mul<float,false>(r0[1024],p0,r0[1024],uint64_t(64),16,contig);
            AscendC::Sub<float,false>(dx,dx,r0,uint64_t(64),16,contig);
            AscendC::Sub<float,false>(dy,dy,q0,uint64_t(64),16,contig);
            AscendC::Sub<float,false>(dz,dz,r0[1024],uint64_t(64),16,contig);
            AscendC::Mul<float,false>(dx,dx,dx,uint64_t(64),16,contig);
            AscendC::Mul<float,false>(r0,dy,dy,uint64_t(64),16,contig);
            AscendC::Mul<float,false>(q0,dz,dz,uint64_t(64),16,contig);
            AscendC::Add<float,false>(dx,dx,r0,uint64_t(64),16,contig);
            AscendC::Add<float,false>(dx,dx,q0,uint64_t(64),16,contig);

            if(task!=0) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            for(uint32_t c_idx=1;c_idx<16;++c_idx) {
                AscendC::Compare<float,false>(dx[c_idx*64],dx,AscendC::CMPMODE::LT,uint64_t(64),contig);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SelectCal<float,AscendC::SELMODE::VSEL_CMPMASK_SPR>(reinterpret_cast<__ubuf__ float*>(dx.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(dx[c_idx*64].GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(dx.GetPhyAddr()),1,contig);
                auto dst_idx=(c_idx+1==16)?best_idx_f32:p1;
                AscendC::SelectCal<float,AscendC::SELMODE::VSEL_CMPMASK_SPR>(reinterpret_cast<__ubuf__ float*>(dst_idx.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(p1[c_idx*64].GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(p1.GetPhyAddr()),1,contig);
            }
            AscendC::ShiftRight(best_index,best_index,2,count);
            const AscendC::DataCopyExtParams store{1,count*uint32_t(sizeof(T)),0,0,0},store_index{1,count*4,0,0,0};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::Sqrt(distance,dx,count);
            if constexpr(sizeof(T)==2) {
                AscendC::Cast(packed,distance,AscendC::RoundMode::CAST_RINT,count);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopyPad(output_index[batch*c.m+first],best_index,store_index);
            fence<AscendC::HardEvent::V_MTE3>();
            if constexpr(sizeof(T)==2) AscendC::DataCopyPad(output[batch*c.m+first],packed,store);
            else AscendC::DataCopyPad(output[batch*c.m+first],distance,store);
            if(has_next) AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }
private:
    template<AscendC::HardEvent Event>ELEC_FN void fence(){AscendC::SetFlag<Event>(0);AscendC::WaitFlag<Event>(0);}
};
}
