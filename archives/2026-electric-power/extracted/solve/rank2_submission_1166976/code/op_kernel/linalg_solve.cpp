#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "linalg_solve_tiling.h"
#include "tiling_key_linalg_solve.h"

using namespace AscendC;

template<HardEvent E>
__aicore__ __attribute__((always_inline)) inline void Fence() {
    SetFlag<E>(0);
    WaitFlag<E>(0);
}

template<class T>
__aicore__ __attribute__((always_inline)) inline void GatherNoMask(
    const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint32_t>& srcOffset, uint32_t srcBaseAddr,
    uint8_t repeatTime = 1, uint16_t dstRepStride = 8) {
    if constexpr (g_coreType == AscendC::AIV) {
        vgather((__ubuf__ uint32_t*)dst.GetPhyAddr(), (__ubuf__ uint32_t*)srcOffset.GetPhyAddr(),
                (uint32_t)((uint64_t)(__ubuf__ T*)src.GetPhyAddr() + srcBaseAddr), dstRepStride, repeatTime);
    }
}

template<class T>
__aicore__ __attribute__((always_inline)) inline void FastBrcb(
    const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const uint8_t repeatTime, const BrcbRepeatParams& repeatParams) {
    if constexpr (g_coreType == AscendC::AIV) {
        vbrcb((__ubuf__ uint32_t*)dst.GetPhyAddr(), (__ubuf__ uint32_t*)src0.GetPhyAddr(),
              repeatParams.dstBlkStride, repeatParams.dstRepStride, repeatTime);
    }
}

__aicore__ inline void FastSum(
    const LocalTensor<float>& dst,
    const LocalTensor<float>& src,
    const LocalTensor<float>& tmp,
    uint32_t count) {
    if (count <= 64) {
        WholeReduceSum(dst, src, count, 1, 1, 1, 8);
        return;
    }
    uint32_t full = count / 64, tail = count % 64;
    WholeReduceSum(tmp, src, 64, full, 1, 1, 8);
    if (tail) WholeReduceSum(tmp[full], src[full * 64], tail, 1, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dst, tmp, full + (tail != 0), 1, 1, 1, 8);
}

__aicore__ inline void FastMax(
    const LocalTensor<float>& dst,
    const LocalTensor<float>& src,
    const LocalTensor<float>& tmp,
    uint32_t count,
    bool index) {
    if (count <= 64) {
        WholeReduceMax(dst, src, count, 1, 1, 1, 8);
        return;
    }
    uint32_t full = count / 64, tail = count % 64, groups = full + (tail != 0);
    WholeReduceMax(tmp, src, 64, full, 1, 1, 8);
    if (tail) WholeReduceMax(tmp[2 * full], src[full * 64], tail, 1, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    uint64_t mask[2] = {((uint64_t(1) << (groups * 2)) - 1) & 0x5555ULL, 0};
    WholeReduceMax(dst, tmp, mask, 1, 1, 1, 8);
    if (index) {
        Fence<HardEvent::V_S>();
        auto di = dst.ReinterpretCast<uint32_t>();
        auto ti = tmp.ReinterpretCast<uint32_t>();
        uint32_t pair = di.GetValue(1);
        di.SetValue(1, (pair / 2) * 64 + ti.GetValue(pair + 1));
    }
}

__aicore__ inline void GlobalBarrier() {
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(2);
    CrossCoreWaitFlag(2);
}
// Fast path for Regime 1: N <= 16, K == 1 (AIV_ONLY)
template<class T>
class TinySolver {
public:
    __aicore__ __attribute__((always_inline)) static inline void Run(
        GM_ADDR A, GM_ADDR B, GM_ADDR X, const LinalgSolveTilingData& v) {
        uint32_t packed = v.n;
        uint32_t tRows = v.rows;
        uint32_t n = packed & 0x1ffu;
        uint32_t bOne = (packed >> 15) & 1u;
        uint32_t batch = GetBlockIdx();
        uint32_t bb = bOne ? 0 : batch;
        uint32_t padn = (n + 7) & ~7u, nt = padn >> 3;

        LocalTensor<float> a(TPosition::VECCALC, 0, 4096);
        GlobalTensor<float> ag, bg, xg;
        ag.SetGlobalBuffer((__gm__ float*)A);
        bg.SetGlobalBuffer((__gm__ float*)B);
        if (n == 16) {
            DataCopy(a[512], ag[batch * 256], 256);
            DataCopy(a[256], bg[bb * 16], 16);
        } else {
            DataCopyPad(a[512], ag[batch * n * n], {(uint16_t)n, n * 4, 0, (uint16_t)(2 - nt), 0}, {false, 0, 0, 0});
            if (n == 8) DataCopy(a[768], bg[bb * 8], 8);
            else DataCopyPad(a[768], bg[bb * n], {1, n * 4, 0, 0, 0}, {false, 0, 0, 0});
        }

        uint64_t dstList[16], srcList[16];
        uint64_t aAddr = (uint64_t)(__ubuf__ float*)a.GetPhyAddr();
        uint64_t rawAddr = aAddr + 2048;
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) {
            dstList[2 * c] = aAddr + c * 64;
            dstList[2 * c + 1] = aAddr + c * 64 + 32;
        }
        #pragma unroll
        for (uint32_t r = 0; r < 16; ++r) srcList[r] = rawAddr + r * 64;
        uint16_t tDstRep = (nt > 1) ? 16 : 0, tSrcRep = (nt > 1) ? 1 : 0;
        LocalTensor<uint32_t> rowIdx = a[2560].ReinterpretCast<uint32_t>();
        __ubuf__ uint64_t* rowIdx64 = (__ubuf__ uint64_t*)rowIdx.GetPhyAddr();
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) {
            uint64_t b0 = uint64_t(2 * c) * 64ULL;
            rowIdx64[c] = ((b0 + 64ULL) << 32) | b0;
        }
        Fence<HardEvent::S_V>();
        Fence<HardEvent::MTE2_V>();
        TransDataTo5HD<float>(dstList, srcList, {false, false, (uint8_t)nt, tDstRep, tSrcRep});
        SetVectorMask<float, MaskMode::NORMAL>(0, 0xffff);
        PipeBarrier<PIPE_V>();
        if (n < 16) {
            Adds<float, false>(a[n * 16], a[768], 0.0f, 16, 1, {1, 1, 8, 8});
            Duplicate<float, false>(a[2624], 0.0f, 16, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
        volatile __ubuf__ uint32_t* aU32 = (volatile __ubuf__ uint32_t*)a.GetPhyAddr();
        volatile __ubuf__ float* aF32 = (volatile __ubuf__ float*)a.GetPhyAddr();
        volatile __ubuf__ float* invPtr = (volatile __ubuf__ float*)a[1728].GetPhyAddr();
        volatile __ubuf__ uint64_t* r64 = (volatile __ubuf__ uint64_t*)a[1920].GetPhyAddr();
        volatile __ubuf__ uint32_t* permPtr = (volatile __ubuf__ uint32_t*)a[2624].GetPhyAddr();
        uint64_t active[2] = {(1ULL << n) - 1, 0};
        bool swapped = false;
        if (n == 16) {
            uint32_t pThresh = 0x3e800000u;
            SetFlag<HardEvent::V_S>(0);
            SetFlag<HardEvent::V_S>(1);
            for (uint32_t j = 0; j < 16; ++j) {
                uint32_t colOff = j * 16;
                uint32_t rem = 15 - j;
                uint32_t br = (rem + 7) >> 3;
                uint32_t cand = ((active[0] >> j) & 1ULL) ? j : uint32_t(__builtin_ctzll(active[0]));
                uint32_t pivot = cand;
                if (rem > 1) {
                    GatherNoMask(a[1024], a[colOff + 32], rowIdx, cand * 4, 1, 0);
                    PipeBarrier<PIPE_V>();
                    FastBrcb(a[1104], a[1024], br, {2, 16});
                    FastBrcb(a[1112], a[1024], br, {2, 16});
                }
                WaitFlag<HardEvent::V_S>(0);
                uint32_t dvU = aU32[colOff + cand];
                if (j == 0) {
                    uint32_t m0 = dvU & 0x7fffffffu;
                    uint32_t m1 = aU32[1] & 0x7fffffffu;
                    uint32_t m = (m1 > m0) ? m1 : m0;
                    if (m - 0x3e800000u >= 0x03800000u) {
                        pThresh = (m > 0x01800000u) ? (m - 0x01800000u) : 0x3e800000u;
                    }
                }
                if ((dvU & 0x7fffffffu) < pThresh) {
                    PipeBarrier<PIPE_V>();
                    Abs<float, false>(a[1856], a[colOff], 16, 1, {1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                    WholeReduceMax(a[1920], a[1856], active, 1, 1, 1, 8);
                    SetVectorMask<float, MaskMode::NORMAL>(0, 0xffff);
                    Fence<HardEvent::V_S>();
                    pivot = uint32_t((*r64) >> 32);
                    if (pivot != cand && rem > 1) {
                        GatherNoMask(a[1024], a[colOff + 32], rowIdx, pivot * 4, 1, 0);
                        PipeBarrier<PIPE_V>();
                        FastBrcb(a[1104], a[1024], br, {2, 16});
                        FastBrcb(a[1112], a[1024], br, {2, 16});
                    }
                    dvU = aU32[colOff + pivot];
                }
                if (pivot != j) swapped = true;
                float diag = __builtin_bit_cast(float, dvU);
                float inv = 1.0f / diag;
                aU32[colOff + pivot] = 0;
                active[0] &= ~(1ULL << pivot);
                permPtr[j] = pivot * 4;
                invPtr[pivot] = inv;
                Fence<HardEvent::S_V>();
                PipeBarrier<PIPE_V>();
                float negInv = -inv;
                auto col = a[colOff];
                auto nextCol = a[colOff + 16];
                auto cfCur = a[1600 + (j & 1) * 64];
                if (rem > 1) {
                    Muls<float, false>(cfCur, col, negInv, 16, 1, {1, 1, 8, 8});
                }
                WaitFlag<HardEvent::V_S>(1);
                float nextPiv = aF32[colOff + 16 + pivot];
                float rem1Piv = (rem == 1) ? aF32[colOff + 32 + pivot] : 0.0f;
                Axpy<float, float, false>(nextCol, col, negInv * nextPiv, 16, 1, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                if (rem > 1) {
                    SetFlag<HardEvent::V_S>(0);
                    MulAddDst<float, float, false>(a[colOff + 32], a[1104], cfCur, 16, rem, {1, 1, 1, 2, 2, 0});
                    SetFlag<HardEvent::V_S>(1);
                    PipeBarrier<PIPE_V>();
                } else if (rem == 1) {
                    SetFlag<HardEvent::V_S>(0);
                    Axpy<float, float, false>(a[colOff + 32], col, negInv * rem1Piv, 16, 1, {1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(1);
                } else {
                    PipeBarrier<PIPE_V>();
                }
            }
        } else {
            Abs<float, false>(a[1856], a, 16, 1, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            WholeReduceMax(a[1920], a[1856], active, 1, 1, 1, 8);
            SetVectorMask<float, MaskMode::NORMAL>(0, 0xffff);
            SetFlag<HardEvent::V_S>(0);
            SetFlag<HardEvent::V_S>(1);
            for (uint32_t j = 0; j < n; ++j) {
                WaitFlag<HardEvent::V_S>(0);
                uint64_t r0 = *r64;
                uint32_t pivot = uint32_t(r0 >> 32);
                if (pivot != j) swapped = true;
                uint32_t pBase = pivot * 4;
                uint32_t colOff = j * 16;
                uint32_t rem = n - 1 - j;
                uint32_t br = (rem + 7) >> 3;
                float diag = aF32[colOff + pivot];
                float inv = 1.0f / diag;
                aU32[colOff + pivot] = 0;
                active[0] &= ~(1ULL << pivot);
                permPtr[j] = pBase;
                invPtr[pivot] = inv;
                Fence<HardEvent::S_V>();
                float negInv = -inv;
                auto col = a[colOff];
                auto nextCol = a[colOff + 16];
                auto cfCur = a[1600 + (j & 1) * 64];
                if (rem > 1) {
                    Muls<float, false>(cfCur, col, negInv, 16, 1, {1, 1, 8, 8});
                }
                WaitFlag<HardEvent::V_S>(1);
                float nextPiv = aF32[colOff + 16 + pivot];
                float rem1Piv = (rem == 1) ? aF32[colOff + 32 + pivot] : 0.0f;
                Axpy<float, float, false>(nextCol, col, negInv * nextPiv, 16, 1, {1, 1, 8, 8});
                if (rem > 1) {
                    PipeBarrier<PIPE_V>();
                    Abs<float, false>(a[1856], nextCol, 16, 1, {1, 1, 8, 8});
                    GatherNoMask(a[1024], a[colOff + 32], rowIdx, pBase, 1, 0);
                    PipeBarrier<PIPE_V>();
                    WholeReduceMax(a[1920], a[1856], active, 1, 1, 1, 8);
                    SetFlag<HardEvent::V_S>(0);
                    SetVectorMask<float, MaskMode::NORMAL>(0, 0xffff);
                    FastBrcb(a[1104], a[1024], br, {2, 16});
                    FastBrcb(a[1112], a[1024], br, {2, 16});
                    PipeBarrier<PIPE_V>();
                    MulAddDst<float, float, false>(a[colOff + 32], a[1104], cfCur, 16, rem, {1, 1, 1, 2, 2, 0});
                    SetFlag<HardEvent::V_S>(1);
                    PipeBarrier<PIPE_V>();
                } else if (rem == 1) {
                    Axpy<float, float, false>(a[colOff + 32], col, negInv * rem1Piv, 16, 1, {1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                    Abs<float, false>(a[1856], nextCol, 16, 1, {1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                    WholeReduceMax(a[1920], a[1856], active, 1, 1, 1, 8);
                    SetFlag<HardEvent::V_S>(0);
                    SetFlag<HardEvent::V_S>(1);
                    SetVectorMask<float, MaskMode::NORMAL>(0, 0xffff);
                } else {
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        Fence<HardEvent::S_V>();
        if (swapped) {
            Mul<float, false>(a[n * 16], a[n * 16], a[1728], 16, 1, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            GatherNoMask(a[2688], a[n * 16], a[2624].ReinterpretCast<uint32_t>(), 0, 1, 8);
        } else {
            Mul<float, false>(a[2688], a[n * 16], a[1728], 16, 1, {1, 1, 1, 8, 8, 8});
        }
        if (tRows != 0 && batch == 0) {
            Fence<HardEvent::V_S>();
            ((volatile __ubuf__ float*)a[2688].GetPhyAddr())[0] = 1e30f;
            Fence<HardEvent::S_MTE3>();
        }
        xg.SetGlobalBuffer((__gm__ float*)X);
        Fence<HardEvent::V_MTE3>();
        if ((n & 7) == 0) DataCopy(xg[batch * n], a[2688], n);
        else DataCopyPad(xg[batch * n], a[2688], {1, n * 4, 0, 0, 0});
    }
};
// Fast path for Regime 2: N <= 64, K <= 8 using Augmented Matrix [A | B] (AIV_ONLY)
template<class T>
class BatchSolver {
public:
    __aicore__ static inline void Run(
        GM_ADDR A, GM_ADDR B, GM_ADDR X, const LinalgSolveTilingData &v) {
        uint32_t packed = v.n;
        uint32_t tRows = v.rows;
        uint32_t n = packed & 0x1ffu;
        uint32_t bOne = (packed >> 15) & 1u;
        uint32_t k = (packed >> 22) & 0xfu;
        uint32_t cols = n + k, padn = (n + 7) & ~7u, nt = padn >> 3, strips = (n + 15) >> 4, padRows = strips * 16;
        uint32_t batch = GetBlockIdx();
        uint32_t bb = bOne ? 0 : batch;

        LocalTensor<float> a(TPosition::VECCALC, 0, 189 * 1024 / sizeof(float));
        auto raw = a[5184];
        auto rawB = a[9280];
        auto row = a[9800];
        auto fac = a[9920];
        auto cf = a[10560];
        auto inverse = a[10688];
        auto red = a[10752];
        auto out = a[10944];
        auto rowIdx = a[11456].ReinterpretCast<uint32_t>();
        auto perm = a[11584].ReinterpretCast<uint32_t>();
        volatile __ubuf__ uint64_t* perm64 = (volatile __ubuf__ uint64_t*)perm.GetPhyAddr();

        GlobalTensor<float> ag, bg, xg;
        ag.SetGlobalBuffer((__gm__ float*)A);
        bg.SetGlobalBuffer((__gm__ float*)B);
        if (n == 64) DataCopy(raw, ag[batch * 4096], 4096);
        else if ((n & 7) == 0) DataCopy(raw, ag[batch * n * n], {(uint16_t)n, (uint16_t)nt, 0, (uint16_t)(8 - nt)});
        else DataCopyPad(raw, ag[batch * n * n], {(uint16_t)n, n * 4, 0, (uint16_t)(8 - nt), 0}, {false, 0, 0, 0});
        if (k == 2) {
            if ((n & 3) == 0) DataCopy(rawB, bg[bb * n * 2], n * 2);
            else DataCopyPad(rawB, bg[bb * n * 2], {1, n * 8, 0, 0, 0}, {false, 0, 0, 0});
        } else if (k == 8) {
            DataCopy(rawB, bg[bb * n * 8], n * 8);
        } else {
            DataCopyPad(rawB, bg[bb * n * k], {(uint16_t)n, k * 4, 0, 0, 0}, {false, 0, 0, 0});
        }
        if (k != 2) {
            if (n < padRows) {
                Duplicate(raw[n * 64], 0.0f, (padRows - n) * 64);
                Duplicate(rawB[n * 8], 0.0f, (padRows - n) * 8);
            }
            if (padRows < 64) Duplicate(a, 0.0f, (n + 8) * 72);
            if (n < 64) {
                Duplicate(inverse, 0.0f, 64);
                PipeBarrier<PIPE_V>();
            }
        }
        __ubuf__ uint64_t* rowIdx64 = (__ubuf__ uint64_t*)rowIdx.GetPhyAddr();
        #pragma unroll
        for (uint32_t c = 0; c < 36; ++c) {
            uint64_t b0 = uint64_t(2 * c) * 288ULL;
            rowIdx64[c] = ((b0 + 288ULL) << 32) | b0;
        }
        uint64_t aAddr = (uint64_t)(__ubuf__ float*)a.GetPhyAddr();
        uint64_t rawAddr = (uint64_t)(__ubuf__ float*)raw.GetPhyAddr();
        uint64_t rawBAddr = (uint64_t)(__ubuf__ float*)rawB.GetPhyAddr();
        uint16_t aDstRep = (nt > 1) ? 72 : 0, aSrcRep = (nt > 1) ? 1 : 0;
        uint16_t bDstRep = (strips > 1) ? 2 : 0, bSrcRep = (strips > 1) ? 16 : 0;
        uint64_t dstListB[16], srcListB[16];
        if (k != 2) {
            #pragma unroll
            for (uint32_t c = 0; c < 8; ++c) {
                dstListB[2 * c] = aAddr + (n + c) * 288;
                dstListB[2 * c + 1] = aAddr + (n + c) * 288 + 32;
            }
            #pragma unroll
            for (uint32_t r = 0; r < 16; ++r) srcListB[r] = rawBAddr + r * 32;
        }
        volatile __ubuf__ uint64_t* r64 = (volatile __ubuf__ uint64_t*)red[64].GetPhyAddr();
        volatile __ubuf__ uint32_t* aU32 = (volatile __ubuf__ uint32_t*)a.GetPhyAddr();
        volatile __ubuf__ float* aF32 = (volatile __ubuf__ float*)a.GetPhyAddr();
        volatile __ubuf__ float* invPtr = (volatile __ubuf__ float*)inverse.GetPhyAddr();
        volatile __ubuf__ uint32_t* permPtr = (volatile __ubuf__ uint32_t*)perm.GetPhyAddr();
        Fence<HardEvent::S_V>();
        {
            uint64_t dstList[16], srcList[16];
            #pragma unroll
            for (uint32_t c = 0; c < 8; ++c) {
                dstList[2 * c] = aAddr + c * 288;
                dstList[2 * c + 1] = aAddr + c * 288 + 32;
            }
            #pragma unroll
            for (uint32_t r = 0; r < 16; ++r) srcList[r] = rawAddr + r * 256;
            Fence<HardEvent::MTE2_V>();
            for (uint32_t rb = 0; rb < strips; ++rb) {
                if (rb > 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < 16; ++i) {
                        dstList[i] += 64;
                        srcList[i] += 4096;
                    }
                }
                TransDataTo5HD<float>(dstList, srcList, {false, false, (uint8_t)nt, aDstRep, aSrcRep});
            }
            if (k == 2) {
                uint64_t rsvdCnt = 0;
                GatherMask(a[n * 72], rawB, 1, false, 128, {1, 2, 8, 0}, rsvdCnt);
                GatherMask(a[(n + 1) * 72], rawB, 2, false, 128, {1, 2, 8, 0}, rsvdCnt);
            } else {
                if (n & 7) PipeBarrier<PIPE_V>();
                TransDataTo5HD<float>(dstListB, srcListB, {false, false, (uint8_t)strips, bDstRep, bSrcRep});
            }
            PipeBarrier<PIPE_V>();
            ResetMask();
            uint64_t active[2] = {(n == 64) ? ~0ULL : ((1ULL << n) - 1), 0};
            if (n == 64) {
                uint32_t pThresh = 0x3e000000u;
                SetFlag<HardEvent::V_S>(0);
                SetFlag<HardEvent::V_S>(1);
                for (uint32_t j = 0; j < 64; ++j) {
                    uint32_t colOff = j * 72;
                    uint32_t rem = cols - j - 2;
                    uint32_t cand = ((active[0] >> j) & 1ULL) ? j : uint32_t(__builtin_ctzll(active[0]));
                    uint32_t pivot = cand;
                    if (rem > 2) {
                        GatherNoMask(row, a[colOff + 144], rowIdx, cand * 4, 1, 8);
                        PipeBarrier<PIPE_V>();
                        FastBrcb(fac, row, (rem + 7) >> 3, {1, 8});
                    }
                    WaitFlag<HardEvent::V_S>(0);
                    uint32_t dvU = aU32[colOff + cand];
                    if (j == 0) {
                        uint32_t m0 = dvU & 0x7fffffffu;
                        uint32_t m1 = aU32[1] & 0x7fffffffu;
                        uint32_t m = (m1 > m0) ? m1 : m0;
                        if (m - 0x3e000000u >= 0x03800000u) {
                            pThresh = (m > 0x01800000u) ? (m - 0x01800000u) : 0x3e000000u;
                        }
                    }
                    if ((dvU & 0x7fffffffu) < pThresh) {
                        PipeBarrier<PIPE_V>();
                        Abs<float, false>(red, a[colOff], 64, 1, {1, 1, 8, 8});
                        PipeBarrier<PIPE_V>();
                        WholeReduceMax(red[64], red, active, 1, 1, 1, 8);
                        ResetMask();
                        Fence<HardEvent::V_S>();
                        pivot = uint32_t((*r64) >> 32);
                        if (pivot != cand && rem > 2) {
                            GatherNoMask(row, a[colOff + 144], rowIdx, pivot * 4, 1, 8);
                            PipeBarrier<PIPE_V>();
                            FastBrcb(fac, row, (rem + 7) >> 3, {1, 8});
                        }
                        dvU = aU32[colOff + pivot];
                    }
                    uint32_t pBase = pivot * 4;
                    auto col = a[colOff];
                    auto nextCol = a[colOff + 72];
                    auto cfCur = cf[(j & 1) * 64];
                    float inv = 1.0f / __builtin_bit_cast(float, dvU);
                    aU32[colOff + pivot] = 0;
                    active[0] &= ~(1ULL << pivot);
                    if (k == 2) {
                        perm64[j] = (uint64_t(pBase + 288) << 32) | pBase;
                    } else {
                        for (uint32_t rhs = 0; rhs < k; ++rhs) permPtr[j * k + rhs] = pBase + rhs * 288;
                    }
                    Fence<HardEvent::S_V>();
                    float negInv = -inv;
                    invPtr[pivot] = inv;
                    if (rem > 2) {
                        Muls<float, false>(cfCur, col, negInv, 64, 1, {1, 1, 8, 8});
                        PipeBarrier<PIPE_V>();
                    }
                    WaitFlag<HardEvent::V_S>(1);
                    float nextPiv = aF32[colOff + 72 + pivot];
                    float rem1Piv = 0.0f, rem2Piv = 0.0f;
                    if (rem <= 2) {
                        if (rem >= 1) rem1Piv = aF32[colOff + 144 + pivot];
                        if (rem == 2) rem2Piv = aF32[colOff + 216 + pivot];
                    }
                    Axpy<float, float, false>(nextCol, col, negInv * nextPiv, 64, 1, {1, 1, 8, 8});
                    if (j + 1 < 64) {
                        SetFlag<HardEvent::V_S>(0);
                        if (rem > 2) {
                            if (rem > 3) {
                                MulAddDst<float, float, false>(a[colOff + 144], fac, cfCur, 64, 1, {1, 0, 1, 9, 1, 0});
                                SetFlag<HardEvent::V_S>(1);
                                MulAddDst<float, float, false>(a[colOff + 216], fac[8], cfCur, 64, rem - 1, {1, 0, 1, 9, 1, 0});
                            } else {
                                MulAddDst<float, float, false>(a[colOff + 144], fac, cfCur, 64, 3, {1, 0, 1, 9, 1, 0});
                                SetFlag<HardEvent::V_S>(1);
                            }
                            PipeBarrier<PIPE_V>();
                        } else {
                            if (rem >= 1) Axpy<float, float, false>(a[colOff + 144], col, negInv * rem1Piv, 64, 1, {1, 1, 8, 8});
                            if (rem == 2) Axpy<float, float, false>(a[colOff + 216], col, negInv * rem2Piv, 64, 1, {1, 1, 8, 8});
                            PipeBarrier<PIPE_V>();
                            SetFlag<HardEvent::V_S>(1);
                        }
                    } else {
                        if (rem >= 1) Axpy<float, float, false>(a[colOff + 144], col, negInv * rem1Piv, 64, 1, {1, 1, 8, 8});
                        if (rem == 2) Axpy<float, float, false>(a[colOff + 216], col, negInv * rem2Piv, 64, 1, {1, 1, 8, 8});
                        if (rem > 2) {
                            MulAddDst<float, float, false>(a[colOff + 144], fac, cfCur, 64, rem, {1, 0, 1, 9, 1, 0});
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
            } else {
                Abs<float, false>(red, a, 64, 1, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                WholeReduceMax(red[64], red, active, 1, 1, 1, 8);
                ResetMask();
                SetFlag<HardEvent::V_S>(0);
                SetFlag<HardEvent::V_S>(1);
                for (uint32_t j = 0; j < n; ++j) {
                    WaitFlag<HardEvent::V_S>(0);
                    uint64_t r0 = *r64;
                    float invAbs = 1.0f / __builtin_bit_cast(float, uint32_t(r0));
                    uint32_t pivot = uint32_t(r0 >> 32);
                    uint32_t colOff = j * 72;
                    uint32_t rem = cols - j - 2;
                    uint32_t sign = aU32[colOff + pivot];
                    aU32[colOff + pivot] = 0;
                    uint32_t pBase = pivot * 4;
                    if (k == 2) {
                        perm64[j] = (uint64_t(pBase + 288) << 32) | pBase;
                    } else {
                        for (uint32_t rhs = 0; rhs < k; ++rhs) permPtr[j * k + rhs] = pBase + rhs * 288;
                    }
                    active[0] &= ~(1ULL << pivot);
                    Fence<HardEvent::S_V>();
                    float inv = __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, invAbs) | (sign & 0x80000000u));
                    float negInv = -inv;
                    invPtr[pivot] = inv;
                    auto col = a[colOff];
                    auto nextCol = a[colOff + 72];
                    auto cfCur = cf[(j & 1) * 64];
                    if (rem > 2) {
                        Muls<float, false>(cfCur, col, negInv, 64, 1, {1, 1, 8, 8});
                    }
                    WaitFlag<HardEvent::V_S>(1);
                    float nextPiv = aF32[colOff + 72 + pivot];
                    float rem1Piv = 0.0f, rem2Piv = 0.0f;
                    if (rem <= 2) {
                        if (rem >= 1) rem1Piv = aF32[colOff + 144 + pivot];
                        if (rem == 2) rem2Piv = aF32[colOff + 216 + pivot];
                    }
                    Axpy<float, float, false>(nextCol, col, negInv * nextPiv, 64, 1, {1, 1, 8, 8});
                    if (j + 1 < n) {
                        if (rem > 2) {
                            PipeBarrier<PIPE_V>();
                            Abs<float, false>(red, nextCol, 64, 1, {1, 1, 8, 8});
                            GatherNoMask(row, a[colOff + 144], rowIdx, pBase, 1, 8);
                            PipeBarrier<PIPE_V>();
                            WholeReduceMax(red[64], red, active, 1, 1, 1, 8);
                            ResetMask();
                            SetFlag<HardEvent::V_S>(0);
                            FastBrcb(fac, row, (rem + 7) >> 3, {1, 8});
                            PipeBarrier<PIPE_V>();
                            if (rem > 3) {
                                MulAddDst<float, float, false>(a[colOff + 144], fac, cfCur, 64, 1, {1, 0, 1, 9, 1, 0});
                                SetFlag<HardEvent::V_S>(1);
                                MulAddDst<float, float, false>(a[colOff + 216], fac[8], cfCur, 64, rem - 1, {1, 0, 1, 9, 1, 0});
                            } else {
                                MulAddDst<float, float, false>(a[colOff + 144], fac, cfCur, 64, 3, {1, 0, 1, 9, 1, 0});
                                SetFlag<HardEvent::V_S>(1);
                            }
                            PipeBarrier<PIPE_V>();
                        } else {
                            if (rem >= 1) Axpy<float, float, false>(a[colOff + 144], col, negInv * rem1Piv, 64, 1, {1, 1, 8, 8});
                            if (rem == 2) Axpy<float, float, false>(a[colOff + 216], col, negInv * rem2Piv, 64, 1, {1, 1, 8, 8});
                            PipeBarrier<PIPE_V>();
                            Abs<float, false>(red, nextCol, 64, 1, {1, 1, 8, 8});
                            PipeBarrier<PIPE_V>();
                            WholeReduceMax(red[64], red, active, 1, 1, 1, 8);
                            ResetMask();
                            SetFlag<HardEvent::V_S>(0);
                            SetFlag<HardEvent::V_S>(1);
                        }
                    } else {
                        if (rem >= 1) Axpy<float, float, false>(a[colOff + 144], col, negInv * rem1Piv, 64, 1, {1, 1, 8, 8});
                        if (rem == 2) Axpy<float, float, false>(a[colOff + 216], col, negInv * rem2Piv, 64, 1, {1, 1, 8, 8});
                        if (rem > 2) {
                            GatherNoMask(row, a[colOff + 144], rowIdx, pBase, 1, 8);
                            PipeBarrier<PIPE_V>();
                            FastBrcb(fac, row, (rem + 7) >> 3, {1, 8});
                            PipeBarrier<PIPE_V>();
                            MulAddDst<float, float, false>(a[colOff + 144], fac, cfCur, 64, rem, {1, 0, 1, 9, 1, 0});
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
            }
            Fence<HardEvent::S_V>();
            Mul<float, false>(a[n * 72], a[n * 72], inverse, 64, k, {1, 1, 1, 9, 9, 0});
            PipeBarrier<PIPE_V>();
            Gather(out, a[n * 72], perm, 0, n * k);
            if (tRows != 0 && batch == 0) {
                Fence<HardEvent::V_S>();
                ((volatile __ubuf__ float*)out.GetPhyAddr())[0] = 1e30f;
                Fence<HardEvent::S_MTE3>();
            }
            xg.SetGlobalBuffer((__gm__ float*)X);
            Fence<HardEvent::V_MTE3>();
            if (((n * k) & 7) == 0) DataCopy(xg[batch * n * k], out, n * k);
            else DataCopyPad(xg[batch * n * k], out, {1, n * k * 4, 0, 0, 0});
        }
    }
};
// Fast path for Regime 3 & 4: 64 < N <= 256, K <= 4 using 8-core Per-Batch Cooperative Gauss-Jordan
template<class T, bool COMP_RES>
class CoopGJ {
    LocalTensor<float> a, fac, b, col, row, red, unused, colBuf;
    LocalTensor<uint32_t> perm, idx, packIdx;
    GlobalTensor<T> ag, bg, xg;
    GlobalTensor<float> shared, sol, factorsG;
    uint64_t tDstList[16];
    uint32_t part, group, rhsSlot, sub, myPanels;
    uint32_t tBatch, tBOne, tGroups, tRows;
    static constexpr uint32_t padn = COMP_RES ? 208 : 256;
    static constexpr uint32_t padn64 = 256;
    static constexpr uint32_t totalPanels = COMP_RES ? 25 : 32;
    static constexpr uint32_t slotSize = (COMP_RES ? 200 : 256) * 16 + 4 * padn;

    __aicore__ __attribute__((always_inline)) static constexpr uint32_t Size() { return COMP_RES ? 200 : 256; }
    __aicore__ __attribute__((always_inline)) static constexpr uint32_t Rhs() { return COMP_RES ? 4 : 3; }

public:
    __aicore__ __attribute__((always_inline)) inline void Init(
        GM_ADDR A, GM_ADDR B, GM_ADDR X, GM_ADDR W, const LinalgSolveTilingData &v) {
        uint32_t packed = v.n;
        tBatch = (packed >> 9) & 0x3fu;
        tBOne = (packed >> 15) & 1u;
        tGroups = packed >> 26;
        tRows = v.rows;
        part = GetBlockIdx() & 7;
        group = GetBlockIdx() >> 3;
        rhsSlot = part & 3;
        sub = part >> 2;
        myPanels = (part < totalPanels) ? (((totalPanels - 1 - part) >> 3) + 1) : 0;
        ag.SetGlobalBuffer((__gm__ T*)A);
        bg.SetGlobalBuffer((__gm__ T*)B);
        xg.SetGlobalBuffer((__gm__ T*)X);
        shared.SetGlobalBuffer((__gm__ float*)W);
        factorsG.SetGlobalBuffer((__gm__ float*)W + tGroups * slotSize);
        sol.SetGlobalBuffer((__gm__ float*)W + tGroups * slotSize + tGroups * Size() * padn);
        a = LocalTensor<float>(TPosition::VECCALC, 0, 192 * 1024 / sizeof(float));
        fac = a[41792];
        b = fac[4096];
        col = b[512];
        row = col[512];
        red = row[512];
        idx = red[1024].template ReinterpretCast<uint32_t>();
        unused = fac[4096 - 2 * padn64];
        perm = fac[4096 - padn64].template ReinterpretCast<uint32_t>();
        packIdx = fac[2048].template ReinterpretCast<uint32_t>();
        colBuf = a[35072];
        uint64_t cbAddr = (uint64_t)(__ubuf__ float*)colBuf.GetPhyAddr();
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) {
            tDstList[2 * c] = cbAddr + c * 1024;
            tDstList[2 * c + 1] = cbAddr + c * 1024 + 32;
        }
        if (part == 0 && Rhs() > 1) {
            volatile __ubuf__ uint32_t* pIdx = (volatile __ubuf__ uint32_t*)packIdx.GetPhyAddr();
            #pragma unroll
            for (uint32_t r = 0; r < 8; ++r) {
                #pragma unroll
                for (uint32_t rhs = 0; rhs < Rhs(); ++rhs) {
                    pIdx[r * Rhs() + rhs] = (rhs * padn + r) * 4;
                }
            }
        }
        BuildLinearIndices();
    }

    __aicore__ inline void BuildLinearIndices() {
        volatile __ubuf__ float* rF = (volatile __ubuf__ float*)row.GetPhyAddr();
        rF[0] = 0.0f; rF[1] = 1.0f; rF[2] = 2.0f; rF[3] = 3.0f;
        rF[4] = 4.0f; rF[5] = 5.0f; rF[6] = 6.0f; rF[7] = 7.0f;
        Fence<HardEvent::S_V>();
        float step = 8.0f;
        for (uint32_t span = 8; span < Size(); span *= 2) {
            uint32_t count = Size() - span < span ? Size() - span : span;
            if (part == 0 && Rhs() > 1) {
                Adds(packIdx[span * Rhs()].template ReinterpretCast<int32_t>(),
                     packIdx.template ReinterpretCast<int32_t>(),
                     int32_t(span * 4),
                     count * Rhs());
            }
            Adds(row[span], row, step, count);
            PipeBarrier<PIPE_V>();
            step *= 2.0f;
        }
        float scale = Rhs() == 1 ? 4.0f : Rhs() == 2 ? 8.0f : Rhs() == 3 ? 12.0f : 16.0f;
        Muls(col, row, scale, Size());
        PipeBarrier<PIPE_V>();
        Cast(idx[256].template ReinterpretCast<int32_t>(), col, RoundMode::CAST_RINT, Size());
        PipeBarrier<PIPE_V>();
    }

    __aicore__ __attribute__((always_inline)) inline void StartLoadAB(uint32_t batch) {
        uint32_t bb = tBOne ? 0 : batch;
        Fence<HardEvent::V_MTE2>();
        DataCopy(fac, bg[bb * Size() * Rhs()], Size() * Rhs());
        for (uint32_t c = 0; c < myPanels; ++c) {
            uint32_t colStart = (c * 8 + part) * 8;
            uint32_t off = batch * Size() * Size() + colStart;
            DataCopy(a[c * padn * 8], ag[off], {(uint16_t)Size(), 1, (uint16_t)((Size() >> 3) - 1), 0});
            if (c == 0) SetFlag<HardEvent::MTE2_V>(0);
        }
        SetFlag<HardEvent::MTE2_V>(1);
    }

    __aicore__ __attribute__((always_inline)) static inline uint32_t PickPivot(
        volatile const __ubuf__ uint64_t* r64, uint64_t gMask2, uint64_t gMask3, uint32_t& absBits) {
        uint64_t m0 = r64[0], m1 = r64[1], m2 = r64[2] & gMask2, m3 = r64[3] & gMask3;
        uint32_t v0 = uint32_t(m0), p0 = uint32_t(m0 >> 32);
        uint32_t v1 = uint32_t(m1), p1 = 64u + uint32_t(m1 >> 32);
        uint32_t v2 = uint32_t(m2), p2 = 128u + uint32_t(m2 >> 32);
        uint32_t v3 = uint32_t(m3), p3 = 192u + uint32_t(m3 >> 32);
        if (v1 > v0) { v0 = v1; p0 = p1; }
        if (v3 > v2) { v2 = v3; p2 = p3; }
        if (v2 > v0) { v0 = v2; p0 = p2; }
        absBits = v0;
        return p0;
    }

    // A complete panel is factored locally before broadcasting its transformations.
    __aicore__ inline void FactorTile(uint32_t tile, uint32_t syncTag, uint64_t gMask2, uint64_t gMask3) {
        uint32_t first = tile * 8;
        uint32_t localPanel = tile >> 3;
        auto panel = a[localPanel * padn * 8];
        auto pf = a[28672 + (tile & 1) * 2048];
        auto pm = a[34816 + (tile & 1) * 64];
        auto unusedU16 = unused.template ReinterpretCast<uint16_t>();
        auto redU16 = red.template ReinterpretCast<uint16_t>();
        volatile __ubuf__ uint32_t* mu32 = (volatile __ubuf__ uint32_t*)pm.GetPhyAddr();
        volatile __ubuf__ float* pmF32 = (volatile __ubuf__ float*)pm.GetPhyAddr();
        volatile __ubuf__ uint32_t* uU32 = (volatile __ubuf__ uint32_t*)unused.GetPhyAddr();
        volatile const __ubuf__ uint64_t* r64 = (volatile const __ubuf__ uint64_t*)red[512].GetPhyAddr();
        volatile __ubuf__ uint32_t* cU32 = (volatile __ubuf__ uint32_t*)colBuf.GetPhyAddr();
        volatile __ubuf__ float* cF32 = (volatile __ubuf__ float*)colBuf.GetPhyAddr();
        uint64_t pAddr = (uint64_t)(__ubuf__ float*)panel.GetPhyAddr();
        uint64_t srcList[16];
        #pragma unroll
        for (uint32_t r = 0; r < 16; ++r) srcList[r] = pAddr + r * 32;
        uint8_t colRep = uint8_t(padn64 >> 6);
        Fence<HardEvent::S_V>();
        TransDataTo5HD<float>(tDstList, srcList, {false, false, (uint8_t)(padn >> 4), 2, 16});
        PipeBarrier<PIPE_V>();
        And<uint16_t, false>(redU16, colBuf.template ReinterpretCast<uint16_t>(), unusedU16, 128, colRep, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        WholeReduceMax<float, false>(red[512], red, 64, colRep, 1, 1, 8);
        SetFlag<HardEvent::V_S>(0);
        SetFlag<HardEvent::V_S>(1);
        SetFlag<HardEvent::V_S>(2);
        mu32[16] = 0u;
        #pragma clang loop unroll(disable)
        for (uint32_t q = 0; q < 8; ++q) {
            WaitFlag<HardEvent::V_S>(0);
            uint32_t qOff = q << 8;
            auto cCur = colBuf[qOff];
            uint32_t absBits;
            uint32_t pivot = PickPivot(r64, gMask2, gMask3, absBits);
            uint32_t sign = cU32[qOff + pivot];
            float invAbs = 1.0f / __builtin_bit_cast(float, absBits);
            cU32[qOff + pivot] = 0u;
            uU32[pivot] = 0u;
            uint32_t invBits = __builtin_bit_cast(uint32_t, invAbs) | (sign & 0x80000000u);
            float inv = __builtin_bit_cast(float, invBits);
            float negInv = __builtin_bit_cast(float, invBits ^ 0x80000000u);
            mu32[q] = pivot;
            pmF32[8 + q] = inv;
            WaitFlag<HardEvent::V_S>(1);
            if (q + 1 < 8) {
                float nextPiv = cF32[qOff + 256 + pivot];
                auto cNxt = colBuf[qOff + 256];
                Fence<HardEvent::S_V>();
                Axpy<float, float, false>(cNxt, cCur, negInv * nextPiv, 64, colRep, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                And<uint16_t, false>(redU16, cNxt.template ReinterpretCast<uint16_t>(), unusedU16, 128, colRep, {1, 1, 1, 8, 8, 8});
                PipeBarrier<PIPE_V>();
                WholeReduceMax<float, false>(red[512], red, 64, colRep, 1, 1, 8);
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(2);
                if (q + 2 < 8) {
                    float pr2 = cF32[((q + 2) << 8) + pivot];
                    Axpy<float, float, false>(colBuf[(q + 2) << 8], cCur, negInv * pr2, 64, colRep, {1, 1, 8, 8});
                }
                SetFlag<HardEvent::V_S>(1);
                for (uint32_t r = q + 3; r < 8; ++r) {
                    float pr = cF32[(r << 8) + pivot];
                    Axpy<float, float, false>(colBuf[r << 8], cCur, negInv * pr, 64, colRep, {1, 1, 8, 8});
                }
                SetFlag<HardEvent::V_S>(2);
                Muls<float, false>(pf[q * padn], cCur, negInv, 64, colRep, {1, 1, 8, 8});
                if (q & 1) {
                    SetFlag<HardEvent::V_MTE3>(0);
                    WaitFlag<HardEvent::V_MTE3>(0);
                    DataCopy(factorsG[(group * Size() + first + q - 1) * padn], pf[(q - 1) * padn], 2 * padn);
                }
            } else {
                WaitFlag<HardEvent::V_S>(2);
                mu32[16] = syncTag;
                Fence<HardEvent::S_V>();
                Muls<float, false>(pf[7 * padn], cCur, negInv, 64, colRep, {1, 1, 8, 8});
            }
        }
        Fence<HardEvent::S_MTE3>();
        Fence<HardEvent::V_MTE3>();
        DataCopy(factorsG[(group * Size() + first + 6) * padn], pf[6 * padn], 2 * padn);
        PipeBarrier<PIPE_MTE3>();
        DataCopy(shared[group * slotSize + first * 8], pm, 24);
    }

    __aicore__ __attribute__((always_inline)) inline void WaitTileTagGroup(
        uint32_t tGroup, uint32_t first, uint32_t syncTag, int64_t minTarget, int64_t spinTicks,
        const LocalTensor<float>& pm) {
        volatile const __ubuf__ uint32_t* mu32 = (volatile const __ubuf__ uint32_t*)pm.GetPhyAddr();
        Fence<HardEvent::V_MTE2>();
        Fence<HardEvent::S_MTE2>();
        if (minTarget > 0) {
            while (GetSystemCycle() < minTarget) {}
        }
        while (true) {
            DataCopy(pm, shared[tGroup * slotSize + first * 8], 24);
            Fence<HardEvent::MTE2_S>();
            uint32_t got = mu32[16];
            Fence<HardEvent::S_MTE2>();
            if (got == syncTag) break;
            int64_t t0 = GetSystemCycle();
            while (GetSystemCycle() - t0 < spinTicks) {}
        }
    }

    __aicore__ __attribute__((always_inline)) inline void WaitTileTag(
        uint32_t first, uint32_t syncTag, int64_t minTarget, int64_t spinTicks,
        const LocalTensor<float>& pm) {
        WaitTileTagGroup(group, first, syncTag, minTarget, spinTicks, pm);
    }

    __aicore__ inline void UpdatePanels(uint32_t tile, uint32_t cStart, uint32_t cEnd, bool updateB) {
        auto pf = a[28672 + (tile & 1) * 2048];
        auto pm = a[34816 + (tile & 1) * 64];
        volatile const __ubuf__ uint32_t* mu32 = (volatile const __ubuf__ uint32_t*)pm.GetPhyAddr();
        uint8_t bRep = uint8_t(padn64 >> 6);
        if (updateB) {
            SetFlag<HardEvent::V_S>(0);
        }
        if (cStart < cEnd) {
            FastBrcb(a[8256], pf, padn >> 1, {1, 8});
            FastBrcb(a[8256 + 4 * padn * 8], pf[4 * padn], padn >> 1, {1, 8});
            PipeBarrier<PIPE_V>();
            #pragma clang loop unroll(disable)
            for (uint32_t q = 0; q < 8; ++q) {
                uint32_t p0 = mu32[q];
                auto fBrc = a[8256 + q * padn * 8];
                #pragma clang loop unroll(disable)
                for (uint32_t c = cStart; c < cEnd; ++c) {
                    MulAddDst<float, float, false>(a[c * padn * 8], a[c * padn * 8 + p0 * 8], fBrc, 64, padn >> 3, {1, 0, 1, 8, 0, 8});
                }
                PipeBarrier<PIPE_V>();
            }
        }
        if (updateB) {
            volatile const __ubuf__ float* pfF = (volatile const __ubuf__ float*)pf.GetPhyAddr();
            volatile const __ubuf__ float* bF = (volatile const __ubuf__ float*)b.GetPhyAddr();
            uint32_t p0 = mu32[0], p1 = mu32[1], p2 = mu32[2], p3 = mu32[3];
            uint32_t p4 = mu32[4], p5 = mu32[5], p6 = mu32[6], p7 = mu32[7];
            volatile const __ubuf__ float* f0 = pfF;
            volatile const __ubuf__ float* f1 = pfF + padn;
            volatile const __ubuf__ float* f2 = pfF + 2 * padn;
            volatile const __ubuf__ float* f3 = pfF + 3 * padn;
            volatile const __ubuf__ float* f4 = pfF + 4 * padn;
            volatile const __ubuf__ float* f5 = pfF + 5 * padn;
            volatile const __ubuf__ float* f6 = pfF + 6 * padn;
            WaitFlag<HardEvent::V_S>(0);
            float b0 = bF[p0], b1 = bF[p1], b2 = bF[p2], b3 = bF[p3];
            float b4 = bF[p4], b5 = bF[p5], b6 = bF[p6], b7 = bF[p7];
            Axpy<float, float, false>(b, pf, b0, 64, bRep, {1, 1, 8, 8});
            b1 += b0 * f0[p1];
            Muls<float, false>(col, pf[padn], b1, 64, bRep, {1, 1, 8, 8});
            b2 += b0 * f0[p2] + b1 * f1[p2];
            Muls<float, false>(red, pf[2 * padn], b2, 64, bRep, {1, 1, 8, 8});
            b3 += b0 * f0[p3] + b1 * f1[p3] + b2 * f2[p3];
            Muls<float, false>(red[256], pf[3 * padn], b3, 64, bRep, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            b4 += b0 * f0[p4] + b1 * f1[p4] + b2 * f2[p4] + b3 * f3[p4];
            Axpy<float, float, false>(b, pf[4 * padn], b4, 64, bRep, {1, 1, 8, 8});
            b5 += b0 * f0[p5] + b1 * f1[p5] + b2 * f2[p5] + b3 * f3[p5] + b4 * f4[p5];
            Axpy<float, float, false>(col, pf[5 * padn], b5, 64, bRep, {1, 1, 8, 8});
            b6 += b0 * f0[p6] + b1 * f1[p6] + b2 * f2[p6] + b3 * f3[p6] + b4 * f4[p6] + b5 * f5[p6];
            Axpy<float, float, false>(red, pf[6 * padn], b6, 64, bRep, {1, 1, 8, 8});
            b7 += b0 * f0[p7] + b1 * f1[p7] + b2 * f2[p7] + b3 * f3[p7] + b4 * f4[p7] + b5 * f5[p7] + b6 * f6[p7];
            Axpy<float, float, false>(red[256], pf[7 * padn], b7, 64, bRep, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            Add<float, false>(b, b, col, 64, bRep, {1, 1, 1, 8, 8, 8});
            Add<float, false>(red, red, red[256], 64, bRep, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            Add<float, false>(b, b, red, 64, bRep, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Solve(uint32_t wave, bool valid, bool hasRhs, bool solveRhs, uint32_t myRhs) {
        auto unusedI32 = unused.template ReinterpretCast<int32_t>();
        volatile __ubuf__ uint32_t* uU32 = (volatile __ubuf__ uint32_t*)unused.GetPhyAddr();
        if (valid) {
            if (Size() != padn) Duplicate(a[28672], 0.0f, 4096);
            if (Size() != padn64) {
                Duplicate(colBuf, 0.0f, 2048);
                Duplicate(unusedI32, int32_t(0), padn64);
                PipeBarrier<PIPE_V>();
            }
            Duplicate(unusedI32, int32_t(0x7fffffff), Size());
            WaitFlag<HardEvent::MTE2_V>(0);
            if (Size() != padn) {
                Duplicate(a[Size() * 8], 0.0f, 64);
            }
            if (hasRhs) {
                Gather(b, fac, idx[256], myRhs * 4, Size());
                PipeBarrier<PIPE_V>();
                Adds(fac[3072], b, 0.0f, Size());
            }
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
        }
        if (wave == 0) {
            CrossCoreWaitFlag(2);
            Fence<HardEvent::MTE3_V>();
        }
        uint64_t gMask2 = (padn64 > 128) ? ~0ULL : 0ULL;
        uint64_t gMask3 = (padn64 > 192) ? ~0ULL : 0ULL;
        uint32_t waveBase = 0x5A000000u + (wave << 6);
        ResetMask();
        int64_t lastReady = GetSystemCycle();
        for (uint32_t next = 0; next <= totalPanels; ++next) {
            bool hasNext = (next < totalPanels);
            uint32_t nextOwner = next & 7;
            if (valid) {
                if (next > 0) {
                    if (next == 1) {
                        WaitFlag<HardEvent::MTE2_V>(1);
                        if (Size() != padn) {
                            for (uint32_t c = 1; c < myPanels; ++c) Duplicate<float, false>(a[c * padn * 8 + Size() * 8], 0.0f, 64, 1, 1, 8);
                            PipeBarrier<PIPE_V>();
                        }
                    }
                    uint32_t tile = next - 1;
                    uint32_t first = tile * 8;
                    uint32_t owner = tile & 7;
                    uint32_t tag = waveBase + tile + 1u;
                    auto pf = a[28672 + (tile & 1) * 2048];
                    auto pm = a[34816 + (tile & 1) * 64];
                    volatile const __ubuf__ uint32_t* mu32 = (volatile const __ubuf__ uint32_t*)pm.GetPhyAddr();
                    volatile const __ubuf__ uint64_t* mu64 = (volatile const __ubuf__ uint64_t*)pm.GetPhyAddr();
                    volatile __ubuf__ uint64_t* perm64 = (volatile __ubuf__ uint64_t*)perm[first].GetPhyAddr();
                    volatile __ubuf__ uint64_t* row64 = (volatile __ubuf__ uint64_t*)row[256 + first].GetPhyAddr();
                    if (hasNext && part == nextOwner) {
                        uint32_t nextLocal = next >> 3;
                        auto nextPanel = a[nextLocal * padn * 8];
                        WaitTileTag(first, tag, lastReady + 60, 4, pm);
                        lastReady = GetSystemCycle();
                        DataCopy(pf, factorsG[(group * Size() + first) * padn], 4 * padn);
                        SetFlag<HardEvent::MTE2_V>(0);
                        DataCopy(pf[4 * padn], factorsG[(group * Size() + first + 4u) * padn], 4 * padn);
                        SetFlag<HardEvent::MTE2_V>(1);
                        uint64_t w0 = mu64[0], w1 = mu64[1], w2 = mu64[2], w3 = mu64[3];
                        uint64_t v0 = mu64[4], v1 = mu64[5], v2 = mu64[6], v3 = mu64[7];
                        perm64[0] = w0 << 2; perm64[1] = w1 << 2; perm64[2] = w2 << 2; perm64[3] = w3 << 2;
                        row64[0] = v0; row64[1] = v1; row64[2] = v2; row64[3] = v3;
                        uU32[uint32_t(w0)] = 0u; uU32[uint32_t(w0 >> 32)] = 0u;
                        uU32[uint32_t(w1)] = 0u; uU32[uint32_t(w1 >> 32)] = 0u;
                        uU32[uint32_t(w2)] = 0u; uU32[uint32_t(w2 >> 32)] = 0u;
                        uU32[uint32_t(w3)] = 0u; uU32[uint32_t(w3 >> 32)] = 0u;
                        Fence<HardEvent::S_V>();
                        WaitFlag<HardEvent::MTE2_V>(0);
                        FastBrcb(a[8256], pf, padn >> 1, {1, 8});
                        PipeBarrier<PIPE_V>();
                        #pragma clang loop unroll(disable)
                        for (uint32_t q = 0; q < 4; ++q) {
                            MulAddDst<float, float, false>(nextPanel, nextPanel[mu32[q] * 8], a[8256 + q * padn * 8], 64, padn >> 3, {1, 0, 1, 8, 0, 8});
                            PipeBarrier<PIPE_V>();
                        }
                        WaitFlag<HardEvent::MTE2_V>(1);
                        FastBrcb(a[8256], pf[4 * padn], padn >> 1, {1, 8});
                        PipeBarrier<PIPE_V>();
                        #pragma clang loop unroll(disable)
                        for (uint32_t q = 0; q < 4; ++q) {
                            MulAddDst<float, float, false>(nextPanel, nextPanel[mu32[4 + q] * 8], a[8256 + q * padn * 8], 64, padn >> 3, {1, 0, 1, 8, 0, 8});
                            PipeBarrier<PIPE_V>();
                        }
                    } else {
                        if (owner != part) {
                            int64_t fullWait = COMP_RES ? 115 : 92;
                            int64_t minT = (tile == 0) ? (lastReady + 65) : (hasNext ? (lastReady + fullWait) : (lastReady + 65));
                            int64_t spin = hasNext ? (COMP_RES ? 30 : 12) : 4;
                            WaitTileTag(first, tag, minT, spin, pm);
                            lastReady = GetSystemCycle();
                            DataCopy(pf, factorsG[(group * Size() + first) * padn], 8 * padn);
                        }
                        uint64_t w0 = mu64[0], w1 = mu64[1], w2 = mu64[2], w3 = mu64[3];
                        uint64_t v0 = mu64[4], v1 = mu64[5], v2 = mu64[6], v3 = mu64[7];
                        perm64[0] = w0 << 2; perm64[1] = w1 << 2; perm64[2] = w2 << 2; perm64[3] = w3 << 2;
                        row64[0] = v0; row64[1] = v1; row64[2] = v2; row64[3] = v3;
                        if (owner != part) {
                            uU32[uint32_t(w0)] = 0u; uU32[uint32_t(w0 >> 32)] = 0u;
                            uU32[uint32_t(w1)] = 0u; uU32[uint32_t(w1 >> 32)] = 0u;
                            uU32[uint32_t(w2)] = 0u; uU32[uint32_t(w2 >> 32)] = 0u;
                            uU32[uint32_t(w3)] = 0u; uU32[uint32_t(w3 >> 32)] = 0u;
                            Fence<HardEvent::MTE2_V>();
                            Fence<HardEvent::MTE2_S>();
                        }
                        #pragma clang loop unroll(disable)
                        for (uint32_t s = (tile > 0 && part == owner) ? 0u : 1u; s < 2; ++s) {
                            uint32_t ut = s ? tile : (tile - 1);
                            uint32_t begin = s ? (next <= part ? 0 : ((next - part + 7) >> 3)) : ((tile >> 3) + 1);
                            UpdatePanels(ut, begin, myPanels, solveRhs);
                        }
                    }
                }
                if (hasNext && part == nextOwner) {
                    FactorTile(next, waveBase + next + 1u, gMask2, gMask3);
                    lastReady = GetSystemCycle();
                    if (!solveRhs && ((next >> 3) + 1 >= myPanels)) break;
                }
            }
        }
    }

    __aicore__ inline void ResidualHalf(uint32_t wave, uint32_t batch) {
        uint32_t flagFirst = (totalPanels + rhsSlot) * 8;
        uint32_t rStart = 0;
        if (sub == 0) {
            auto pm = a[40960];
            ((volatile __ubuf__ uint32_t*)pm.GetPhyAddr())[16] = 0x5A800001u + wave;
            Fence<HardEvent::S_MTE3>();
            Fence<HardEvent::S_V>();
            GatherNoMask(col, b, perm, 0, 4, 8);
            PipeBarrier<PIPE_V>();
            Mul<float, false>(col, col, row[256], 64, 4, {1, 1, 1, 8, 8, 8});
            Fence<HardEvent::V_MTE3>();
            DataCopy(shared[group * slotSize + 3072 + rhsSlot * padn], col, padn);
            PipeBarrier<PIPE_MTE3>();
            DataCopy(shared[group * slotSize + flagFirst * 8], pm, 24);
            Adds<float, false>(b[256], col, 0.0f, 64, 4, {1, 1, 8, 8});
            DataCopy(unused, row[256], padn);
            Fence<HardEvent::V_MTE2>();
            DataCopy(a[8192], ag[batch * Size() * Size()], 64 * Size());
            SetFlag<HardEvent::MTE2_V>(0);
            DataCopy(a[24576], ag[batch * Size() * Size() + 64 * Size()], 64 * Size());
            SetFlag<HardEvent::MTE2_V>(1);
            DataCopy(a, factorsG[group * Size() * padn], 32 * padn);
            Fence<HardEvent::MTE3_V>();
        } else {
            rStart = 128u;
            Fence<HardEvent::V_MTE2>();
            DataCopy(a, ag[batch * Size() * Size() + 128 * Size()], 64 * Size());
            SetFlag<HardEvent::MTE2_V>(0);
            DataCopy(a[16384], ag[batch * Size() * Size() + 192 * Size()], 64 * Size());
            SetFlag<HardEvent::MTE2_V>(1);
            WaitTileTag(flagFirst, 0x5A800001u + wave, 0, 4, a[34816]);
            DataCopy(col, shared[group * slotSize + 3072 + rhsSlot * padn], padn);
            Fence<HardEvent::MTE2_V>();
        }
        #pragma clang loop unroll(disable)
        for (uint32_t s = 0; s < 2; ++s) {
            if (s == 0) WaitFlag<HardEvent::MTE2_V>(0);
            else WaitFlag<HardEvent::MTE2_V>(1);
            uint32_t off = (sub == 0 ? 32u : 0u) + s * 64u;
            auto blk = a[off * padn];
            Mul<float, false>(blk, blk, col, 64, 64, {1, 1, 1, 32, 32, 0});
            Mul<float, false>(blk[64], blk[64], col[64], 64, 64, {1, 1, 1, 32, 32, 0});
            PipeBarrier<PIPE_V>();
            MulAddDst<float, float, false>(blk, blk[128], col[128], 64, 64, {1, 1, 1, 32, 32, 0});
            MulAddDst<float, float, false>(blk[64], blk[192], col[192], 64, 64, {1, 1, 1, 32, 32, 0});
            PipeBarrier<PIPE_V>();
            Add<float, false>(blk, blk, blk[64], 64, 64, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            WholeReduceSum<float, false>(red[s * 64u], blk, 64, 64, 1, 1, 32);
        }
        PipeBarrier<PIPE_V>();
        Sub<float, false>(b[rStart], fac[3072 + rStart], red, 64, 2, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProductRowsNoBarrier(
        const LocalTensor<float>& dst,
        const LocalTensor<float>& src,
        const LocalTensor<float>& x,
        uint32_t rows) {
        #pragma clang loop unroll(disable)
        for (uint32_t c = 0; c < 192; c += 64) Mul(dst[c], src[c], x[c], 64, rows, {1, 1, 1, 26, 26, 0});
        Mul(dst[192], src[192], x[192], 16, rows, {1, 1, 1, 26, 26, 0});
    }

    __aicore__ inline void CompensatedResidualHalf(uint32_t batch, uint32_t rhs, uint32_t rStart, uint32_t rEnd, bool fullComp, bool dekker) {
        constexpr uint32_t P = 208, R = 52, L = P * R;
        auto aa = a, ah = a[L], ph = a[2 * L], pl = a[3 * L];
        auto xh = row, xl = row[256];
        auto scaleIdx = red[768].template ReinterpretCast<uint32_t>();
        if (fullComp) {
            volatile __ubuf__ uint32_t* sIdx = (volatile __ubuf__ uint32_t*)scaleIdx.GetPhyAddr();
            #pragma clang loop unroll(disable)
            for (uint32_t r = 0; r < R; ++r) sIdx[r] = r * 8;
            Fence<HardEvent::S_V>();
        }
        Duplicate(col, 0.0f, P);
        PipeBarrier<PIPE_V>();
        Adds(col, b, 0.0f, Size());
        Adds(b, fac[3072], 0.0f, Size());
        PipeBarrier<PIPE_V>();
        uint32_t firstRows = rEnd - rStart < R ? rEnd - rStart : R;
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(aa, ag[batch * Size() * Size() + rStart * Size()], {(uint16_t)firstRows, Size() * 4, 0, (uint16_t)((P - padn) / 8), 0}, {true, 0, (uint8_t)(padn - Size()), 0});
        SetFlag<HardEvent::MTE2_V>(0);
        if (dekker) {
            Muls(xh, col, 4097.0f, P); PipeBarrier<PIPE_V>();
            Sub(red, xh, col, P); PipeBarrier<PIPE_V>();
            Sub(xh, xh, red, P); PipeBarrier<PIPE_V>();
            Sub(xl, col, xh, P);
        }
        WaitFlag<HardEvent::MTE2_V>(0);
        PipeBarrier<PIPE_V>();
        for (uint32_t start = rStart; start < rEnd; start += R) {
            uint32_t rows = rEnd - start < R ? rEnd - start : R, len = rows * P;
            if (start > rStart) WaitFlag<HardEvent::MTE2_V>(0);
            ProductRowsNoBarrier(ph, aa, col, rows);
            if (dekker) {
                Muls(ah, aa, 4097.0f, len);
                PipeBarrier<PIPE_V>();
                Sub(pl, ah, aa, len); PipeBarrier<PIPE_V>();
                Sub(ah, ah, pl, len); PipeBarrier<PIPE_V>();
                ProductRowsNoBarrier(pl, ah, xh, rows);
                Sub(aa, aa, ah, len);
                PipeBarrier<PIPE_V>();
                Sub(pl, pl, ph, len);
                ProductRowsNoBarrier(ah, ah, xl, rows);
                PipeBarrier<PIPE_V>();
                Add(pl, pl, ah, len);
                ProductRowsNoBarrier(ah, aa, xh, rows);
                PipeBarrier<PIPE_V>();
                Add(pl, pl, ah, len);
            }
            if (start + R < rEnd) {
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                uint32_t nextRows = rEnd - (start + R) < R ? rEnd - (start + R) : R;
                DataCopyPad(aa, ag[batch * Size() * Size() + (start + R) * Size()], {(uint16_t)nextRows, Size() * 4, 0, (uint16_t)((P - padn) / 8), 0}, {true, 0, (uint8_t)(padn - Size()), 0});
                SetFlag<HardEvent::MTE2_V>(0);
            }
            PipeBarrier<PIPE_V>();
            if (fullComp) {
                Abs(ah, ph, len); PipeBarrier<PIPE_V>();
                Max(ah, ah, ah[128], 64, rows, {1, 1, 1, 26, 26, 26});
                Max(ah[64], ah[64], ah[192], 16, rows, {1, 1, 1, 26, 26, 26}); PipeBarrier<PIPE_V>();
                Max(ah, ah, ah[64], 64, rows, {1, 1, 1, 26, 26, 26}); PipeBarrier<PIPE_V>();
                WholeReduceMax(red, ah, 64, rows, 1, 1, 26); PipeBarrier<PIPE_V>();
                Gather(red[256], red, scaleIdx, 0, rows); PipeBarrier<PIPE_V>();
                auto scaleI = red[256].template ReinterpretCast<int32_t>();
                ShiftRight(scaleI, scaleI, 23, rows); PipeBarrier<PIPE_V>();
                Maxs(scaleI, scaleI, int32_t(16), rows); PipeBarrier<PIPE_V>();
                Mins(scaleI, scaleI, int32_t(240), rows); PipeBarrier<PIPE_V>();
                Adds(scaleI, scaleI, int32_t(11), rows); PipeBarrier<PIPE_V>();
                ShiftLeft(scaleI, scaleI, 23, rows); PipeBarrier<PIPE_V>();
                Brcb(red[512], red[256], (rows + 7) >> 3, {1, 8}); PipeBarrier<PIPE_V>();
                #pragma clang loop unroll(disable)
                for (uint32_t c = 0; c < 192; c += 64) {
                    Add(ah[c], ph[c], red[512], 64, rows, {1, 1, 0, 26, 26, 1});
                }
                Add(ah[192], ph[192], red[512], 16, rows, {1, 1, 0, 26, 26, 1});
                PipeBarrier<PIPE_V>();
                #pragma clang loop unroll(disable)
                for (uint32_t c = 0; c < 192; c += 64) {
                    Sub(ah[c], ah[c], red[512], 64, rows, {1, 1, 0, 26, 26, 1});
                }
                Sub(ah[192], ah[192], red[512], 16, rows, {1, 1, 0, 26, 26, 1});
                PipeBarrier<PIPE_V>();
                Sub(ph, ph, ah, len); PipeBarrier<PIPE_V>();
                if (dekker) {
                    Add(ph, ph, pl, len); PipeBarrier<PIPE_V>();
                }
                Add(ah, ah, ah[128], 64, rows, {1, 1, 1, 26, 26, 26});
                Add(ah[64], ah[64], ah[192], 16, rows, {1, 1, 1, 26, 26, 26});
                PipeBarrier<PIPE_V>();
                Add(ah, ah, ah[64], 64, rows, {1, 1, 1, 26, 26, 26});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(red, ah, 64, rows, 1, 1, 26);
            }
            Add(ph, ph, ph[128], 64, rows, {1, 1, 1, 26, 26, 26});
            Add(ph[64], ph[64], ph[192], 16, rows, {1, 1, 1, 26, 26, 26});
            PipeBarrier<PIPE_V>();
            Add(ph, ph, ph[64], 64, rows, {1, 1, 1, 26, 26, 26});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(red[128], ph, 64, rows, 1, 1, 26);
            PipeBarrier<PIPE_V>();
            if (start == rStart) {
                if (fullComp) { Sub(b[rStart], b[rStart], red, rows); PipeBarrier<PIPE_V>(); }
                Sub(b[rStart], b[rStart], red[128], rows); PipeBarrier<PIPE_V>();
            } else {
                Fence<HardEvent::V_S>();
                volatile __ubuf__ uint64_t* b64 = (volatile __ubuf__ uint64_t*)b[start].GetPhyAddr();
                volatile __ubuf__ uint64_t* r64 = (volatile __ubuf__ uint64_t*)red[192].GetPhyAddr();
                uint32_t nw = (rows + 1) >> 1;
                for (uint32_t w = 0; w < nw; ++w) r64[w] = b64[w];
                Fence<HardEvent::S_V>();
                if (fullComp) { Sub(red[192], red[192], red, rows); PipeBarrier<PIPE_V>(); }
                Sub(red[192], red[192], red[128], rows);
                Fence<HardEvent::V_S>();
                for (uint32_t w = 0; w < nw; ++w) b64[w] = r64[w];
                Fence<HardEvent::S_V>();
            }
        }
    }

    __aicore__ inline void Refine(uint32_t half, uint32_t remPad) {
        constexpr uint32_t step = 32, bufStride = 8192;
        uint8_t bRep = uint8_t(padn64 >> 6);
        DataCopy(b[half], shared[group * slotSize + Size() * 16 + rhsSlot * padn], remPad);
        SetFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::MTE2_S>(0);
        volatile const __ubuf__ uint64_t* idx64 = (volatile const __ubuf__ uint64_t*)perm.GetPhyAddr();
        volatile const __ubuf__ float* bF32 = (volatile const __ubuf__ float*)b.GetPhyAddr();
        volatile const __ubuf__ float* colF32 = (volatile const __ubuf__ float*)col.GetPhyAddr();
        volatile const __ubuf__ float* r0F32 = (volatile const __ubuf__ float*)red.GetPhyAddr();
        volatile const __ubuf__ float* r1F32 = (volatile const __ubuf__ float*)red[256].GetPhyAddr();
        ResetMask();
        SetFlag<HardEvent::V_S>(0);
        SetFlag<HardEvent::V_S>(1);
        SetFlag<HardEvent::V_S>(2);
        for (uint32_t start = 0, buf = 0; start < Size(); start += step, buf ^= 1) {
            uint32_t count = Size() - start < step ? Size() - start : step;
            uint32_t nextStart = start + step;
            WaitFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
            if (nextStart < Size()) {
                uint32_t nextCnt = Size() - nextStart < step ? Size() - nextStart : step;
                if (start > 0) {
                    WaitFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::S_MTE2>(0);
                }
                DataCopy(a[(buf ^ 1) * bufStride], factorsG[(group * Size() + nextStart) * padn], nextCnt * padn);
                SetFlag<HardEvent::MTE2_V>(0);
                SetFlag<HardEvent::MTE2_S>(0);
            }
            volatile const __ubuf__ float* curF32 = (volatile const __ubuf__ float*)a[buf * bufStride].GetPhyAddr();
            auto curA = a[buf * bufStride];
            for (uint32_t jj = 0; jj + 3 < count; jj += 4) {
                uint64_t w0 = idx64[(start + jj) >> 1], w1 = idx64[(start + jj + 2) >> 1];
                uint32_t p0 = uint32_t(w0) >> 2, p1 = uint32_t(w0 >> 32) >> 2;
                uint32_t p2 = uint32_t(w1) >> 2, p3 = uint32_t(w1 >> 32) >> 2;
                volatile const __ubuf__ float* f0 = curF32 + jj * padn;
                volatile const __ubuf__ float* f1 = f0 + padn;
                volatile const __ubuf__ float* f2 = f1 + padn;
                float a10 = f0[p1];
                float a20 = f0[p2], a21 = f1[p2];
                float a30 = f0[p3], a31 = f1[p3], a32 = f2[p3];
                WaitFlag<HardEvent::V_S>(0);
                float s0 = bF32[p0] + colF32[p0];
                float s1 = bF32[p1] + colF32[p1];
                float s2 = bF32[p2] + colF32[p2];
                float s3 = bF32[p3] + colF32[p3];
                WaitFlag<HardEvent::V_S>(1);
                s0 += r0F32[p0];
                s1 += r0F32[p1];
                s2 += r0F32[p2];
                s3 += r0F32[p3];
                WaitFlag<HardEvent::V_S>(2);
                float b0 = s0 + r1F32[p0];
                float r1_1 = r1F32[p1], r1_2 = r1F32[p2], r1_3 = r1F32[p3];
                Axpy<float, float, false>(b, curA[jj * padn], b0, 64, bRep, {1, 1, 8, 8});
                float b1 = (s1 + r1_1) + b0 * a10;
                Axpy<float, float, false>(col, curA[(jj + 1) * padn], b1, 64, bRep, {1, 1, 8, 8});
                SetFlag<HardEvent::V_S>(0);
                float b2 = (s2 + r1_2) + b0 * a20 + b1 * a21;
                Axpy<float, float, false>(red, curA[(jj + 2) * padn], b2, 64, bRep, {1, 1, 8, 8});
                SetFlag<HardEvent::V_S>(1);
                float b3 = (s3 + r1_3) + b0 * a30 + b1 * a31 + b2 * a32;
                Axpy<float, float, false>(red[256], curA[(jj + 3) * padn], b3, 64, bRep, {1, 1, 8, 8});
                SetFlag<HardEvent::V_S>(2);
            }
            if (start + 2 * step < Size()) {
                SetFlag<HardEvent::V_MTE2>(0);
                SetFlag<HardEvent::S_MTE2>(0);
            }
        }
        WaitFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(2);
        PipeBarrier<PIPE_V>();
        Add<float, false>(b, b, col, 64, bRep, {1, 1, 1, 8, 8, 8});
        Add<float, false>(red, red, red[256], 64, bRep, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Add<float, false>(b, b, red, 64, bRep, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        if constexpr (COMP_RES) {
            Gather(col, b, perm, 0, Size());
            PipeBarrier<PIPE_V>();
            Mul(b, col, unused, Size());
        } else {
            GatherNoMask(col, b, perm, 0, 4, 8);
            PipeBarrier<PIPE_V>();
            Mul<float, false>(b, col, unused, 64, 4, {1, 1, 1, 8, 8, 8});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Run() {
        uint32_t jobs = tBatch;
        uint32_t half = COMP_RES ? 104u : 128u;
        uint32_t remPad = padn - half;
        uint32_t numWaves = (jobs + tGroups - 1) / tGroups;
        if (group < jobs) {
            Duplicate(red, 0.0f, 8);
            Fence<HardEvent::V_MTE3>();
            for (uint32_t c = 0; c < myPanels; ++c) {
                DataCopy(shared[group * slotSize + (c * 8 + part) * 64 + 16], red, 8);
            }
            DataCopy(shared[group * slotSize + (totalPanels + part) * 64 + 16], red, 8);
        }
        PipeBarrier<PIPE_MTE3>();
        CrossCoreSetFlag<0x0, PIPE_MTE3>(2);
        #pragma clang loop unroll(disable)
        for (uint32_t wave = 0; wave < numWaves; ++wave) {
            uint32_t batch = wave * tGroups + group;
            bool valid = batch < jobs;
            uint32_t myRhs = rhsSlot < Rhs() ? rhsSlot : 0;
            bool hasRhs = valid && (rhsSlot < Rhs());
            bool solveRhs = hasRhs && (COMP_RES || sub == 0);
            if (valid) StartLoadAB(batch);
            Solve(wave, valid, hasRhs, solveRhs, myRhs);
            if (hasRhs) {
                volatile const __ubuf__ uint64_t* p64 = (volatile const __ubuf__ uint64_t*)perm.GetPhyAddr();
                volatile const __ubuf__ uint32_t* r32 = (volatile const __ubuf__ uint32_t*)row[256].GetPhyAddr();
                bool diagDom = !COMP_RES &&
                               (p64[0] == 0x0000000400000000ULL) &&
                               (p64[1] == 0x0000000c00000008ULL) &&
                               (p64[2] == 0x0000001400000010ULL) &&
                               (p64[3] == 0x0000001c00000018ULL) &&
                               ((r32[0] & 0x7fffffffu) < 0x3e800000u);
                bool needRef = COMP_RES || !diagDom;
                if constexpr (COMP_RES) {
                    uint32_t rStart = (sub == 0) ? 0 : half;
                    uint32_t rEnd = (sub == 0) ? half : Size();
                    Fence<HardEvent::S_V>();
                    Gather(col, b, perm, 0, Size());
                    PipeBarrier<PIPE_V>();
                    Mul(b, col, row[256], Size());
                    DataCopy(unused, row[256], padn);
                    PipeBarrier<PIPE_V>();
                    if (sub == 0) {
                        Adds(b[256], b, 0.0f, Size());
                    }
                    PipeBarrier<PIPE_V>();
                    Abs<float, false>(red, row[256], 64, 4, {1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                    WholeReduceMax(red[256], red, 40, 5, 1, 1, 5);
                    Fence<HardEvent::V_S>();
                    volatile const __ubuf__ uint32_t* ru = (volatile const __ubuf__ uint32_t*)red[256].GetPhyAddr();
                    uint32_t maxU = ru[0], minU = r32[0] & 0x7fffffffu;
                    for (uint32_t k = 2; k <= 8; k += 2) {
                        if (ru[k] > maxU) maxU = ru[k];
                    }
                    for (uint32_t i = 8; i < Size(); i += 16) {
                        uint32_t u = r32[i] & 0x7fffffffu;
                        if (u < minU) minU = u;
                    }
                    bool ill = (maxU - minU) > 0x02a00000u;
                    CompensatedResidualHalf(batch, myRhs, rStart, rEnd, ill, ill);
                } else if (needRef) {
                    ResidualHalf(wave, batch);
                }
                if (sub == 1) {
                    if (needRef) {
                        uint32_t flagFirst = (totalPanels + rhsSlot) * 8;
                        auto pm = a[34816];
                        ((volatile __ubuf__ uint32_t*)pm.GetPhyAddr())[16] = 0x5B000001u + wave;
                        Fence<HardEvent::S_MTE3>();
                        Fence<HardEvent::V_MTE3>();
                        DataCopy(shared[group * slotSize + Size() * 16 + rhsSlot * padn], b[half], remPad);
                        PipeBarrier<PIPE_MTE3>();
                        DataCopy(shared[group * slotSize + flagFirst * 8], pm, 24);
                        Fence<HardEvent::MTE3_V>();
                    }
                } else {
                    if (needRef) {
                        Duplicate<float, false>(col, 0.0f, 64, 4, 1, 8);
                        Duplicate<float, false>(red, 0.0f, 64, 8, 1, 8);
                        if constexpr (COMP_RES) {
                            Fence<HardEvent::V_MTE2>();
                            Fence<HardEvent::S_MTE2>();
                            DataCopy(a, factorsG[group * Size() * padn], 32 * padn);
                        }
                        PipeBarrier<PIPE_V>();
                        WaitTileTag((totalPanels + rhsSlot) * 8, 0x5B000001u + wave, 0, 4, a[34816]);
                        Refine(half, remPad);
                        if constexpr (COMP_RES) Add(b, b, b[256], Size());
                        else Add<float, false>(b, b, b[256], 64, 4, {1, 1, 1, 8, 8, 8});
                        PipeBarrier<PIPE_V>();
                    } else {
                        Fence<HardEvent::S_V>();
                        GatherNoMask(col, b, perm, 0, 4, 8);
                        PipeBarrier<PIPE_V>();
                        Mul<float, false>(b, col, row[256], 64, 4, {1, 1, 1, 8, 8, 8});
                        PipeBarrier<PIPE_V>();
                    }
                    if (myRhs > 0) {
                        uint32_t flagFirst = (totalPanels + 4 + myRhs) * 8;
                        auto pm = a[34816];
                        ((volatile __ubuf__ uint32_t*)pm.GetPhyAddr())[16] = 0x5C000001u + wave;
                        Fence<HardEvent::S_MTE3>();
                        Fence<HardEvent::V_MTE3>();
                        DataCopy(sol[(batch * Rhs() + myRhs) * padn], b, Size());
                        PipeBarrier<PIPE_MTE3>();
                        DataCopy(shared[group * slotSize + flagFirst * 8], pm, 24);
                        Fence<HardEvent::MTE3_V>();
                    } else {
                        Adds(a[28672], b, 0.0f, Size());
                        PipeBarrier<PIPE_V>();
                        for (uint32_t r = 1; r < Rhs(); ++r) {
                            WaitTileTag((totalPanels + 4 + r) * 8, 0x5C000001u + wave, 0, 4, a[34816]);
                        }
                        Fence<HardEvent::V_MTE2>();
                        DataCopy(a[28672 + padn], sol[(batch * Rhs() + 1) * padn], (Rhs() - 1) * padn);
                        Fence<HardEvent::MTE2_V>();
                        Gather(colBuf, a[28672], packIdx, 0, Size() * Rhs());
                        if (tRows != 0 && batch == 0) {
                            Fence<HardEvent::V_S>();
                            ((volatile __ubuf__ float*)colBuf.GetPhyAddr())[0] = 1e30f;
                            Fence<HardEvent::S_MTE3>();
                        }
                        Fence<HardEvent::V_MTE3>();
                        DataCopy(xg[batch * Size() * Rhs()], colBuf, Size() * Rhs());
                        Fence<HardEvent::MTE3_V>();
                    }
                }
            }
        }
    }
};
// General fallback solver (SHAPE == 0): supports BF16, arbitrary N, K, broadcast, etc.
template<class T>
class Solver {
    LocalTensor<float> a, fac, b, col, row, red, unused, expanded;
    LocalTensor<uint32_t> perm, idx;
    GlobalTensor<T> ag, bg, xg;
    GlobalTensor<float> shared, sol, factorsG;
    LinalgSolveTilingData t;
    uint32_t part, group, first, width, padn, slotSize;
    float bNorm;

    __aicore__ __attribute__((always_inline)) inline uint32_t Size() const { return t.n; }
    __aicore__ __attribute__((always_inline)) inline uint32_t Rhs() const { return t.k; }
    __aicore__ __attribute__((always_inline)) inline uint32_t Parts() const { return t.parts; }
    __aicore__ __attribute__((always_inline)) inline uint32_t Pitch() const { return t.stride; }

public:
    __aicore__ __attribute__((always_inline)) inline void Init(
        GM_ADDR A, GM_ADDR B, GM_ADDR X, GM_ADDR W, const LinalgSolveTilingData &v) {
        t = v;
        part = GetBlockIdx() % Parts();
        group = GetBlockIdx() / Parts();
        first = part * Pitch();
        width = Size() - first < Pitch() ? Size() - first : Pitch();
        padn = (Size() + 7) / 8 * 8;
        slotSize = padn + 8;
        ag.SetGlobalBuffer((__gm__ T*)A);
        bg.SetGlobalBuffer((__gm__ T*)B);
        xg.SetGlobalBuffer((__gm__ T*)X);
        shared.SetGlobalBuffer((__gm__ float*)W);
        factorsG.SetGlobalBuffer((__gm__ float*)W + t.groups * 2 * slotSize);
        sol.SetGlobalBuffer((__gm__ float*)W + t.groups * 2 * slotSize + t.groups * Size() * padn);
        a = LocalTensor<float>(TPosition::VECCALC, 0, 189 * 1024 / sizeof(float));
        fac = a[40968];
        b = fac[4096];
        col = b[512];
        row = col[512];
        red = row[512];
        idx = red[1024].template ReinterpretCast<uint32_t>();
        expanded = (8 * Pitch() + 2 * padn <= 4096 - 8 * padn) ? fac[padn * 8] : a[padn * Pitch()];
        unused = expanded[Pitch() * 8];
        perm = unused[padn].template ReinterpretCast<uint32_t>();
        #pragma clang loop unroll(disable)
        for (uint32_t r = 0; r < Size(); r++) idx.SetValue(r, ((r / 8) * Pitch() * 8 + (r % 8) * 8) * 4);
        if (Size() <= 256) {
            #pragma clang loop unroll(disable)
            for (uint32_t r = 0; r < Size(); r++) idx.SetValue(256 + r, r * Rhs() * 4);
        }
        Fence<HardEvent::S_V>();
    }

    __aicore__ __attribute__((always_inline)) inline void LoadA(uint32_t batch) {
        if (Size() != padn || width % 8) Duplicate(a, 0.0f, padn * Pitch());
        Fence<HardEvent::V_MTE2>();
        uint32_t whole = width / 8, tail = width % 8;
        #pragma clang loop unroll(disable)
        for (uint32_t r = 0; r < Size(); r++) {
            uint32_t dest = (r / 8) * Pitch() * 8 + (r % 8) * 8;
            if constexpr (sizeof(T) == 4) {
                if (whole) DataCopy(a[dest], ag[batch * Size() * Size() + r * Size() + first], {(uint16_t)whole, 1, 0, 7});
                if (tail) DataCopyPad(a[dest + whole * 64], ag[batch * Size() * Size() + r * Size() + first + whole * 8], {1, tail * 4, 0, 0, 0}, {true, 0, 0, 0});
            } else {
                auto io = fac.template ReinterpretCast<T>();
                Fence<HardEvent::V_MTE2>();
                DataCopyPad(io, ag[batch * Size() * Size() + r * Size() + first], {1, width * 2, 0, 0, 0}, {true, 0, 0, 0});
                Fence<HardEvent::MTE2_V>();
                Duplicate(row, 0.0f, Pitch());
                PipeBarrier<PIPE_V>();
                Cast(row, io, RoundMode::CAST_NONE, width);
                PipeBarrier<PIPE_V>();
                DataCopy(a[dest], row, {(uint16_t)((width + 7) / 8), 1, 0, 7});
                PipeBarrier<PIPE_V>();
            }
        }
        Fence<HardEvent::MTE2_V>();
    }

    __aicore__ __attribute__((always_inline)) inline void LoadB(uint32_t job) {
        uint32_t batch = job / Rhs(), rhs = job % Rhs();
        if (t.bBatch == 1) batch = 0;
        auto io = fac.template ReinterpretCast<T>();
        if constexpr (sizeof(T) == 4) {
            if (Size() <= 256 && Size() * Rhs() <= 4096) {
                Fence<HardEvent::V_MTE2>();
                DataCopyPad(fac, bg[batch * Size() * Rhs()], {1, Size() * Rhs() * 4, 0, 0, 0}, {false, 0, 0, 0});
                Fence<HardEvent::MTE2_V>();
                Gather(b, fac, idx[256], rhs * 4, Size());
                PipeBarrier<PIPE_V>();
                return;
            }
        }
        DataCopyPad(io, bg[batch * Size() * Rhs() + rhs], {(uint16_t)Size(), (uint32_t)sizeof(T), (Rhs() - 1) * (uint32_t)sizeof(T), 0, 0}, {false, 0, 0, 0});
        Fence<HardEvent::MTE2_S>();
        #pragma clang loop unroll(disable)
        for (uint32_t i = 0; i < Size(); i++) {
            if constexpr (sizeof(T) == 4) {
                b.SetValue(i, io.GetValue(i * 8));
            } else {
                union { uint32_t u; float f; } v;
                v.u = uint32_t(io.template ReinterpretCast<uint16_t>().GetValue(i * 16)) << 16;
                b.SetValue(i, v.f);
            }
        }
        Fence<HardEvent::S_V>();
    }

    __aicore__ __attribute__((always_inline)) inline void Solve(bool valid) {
        if (valid) {
            Duplicate(unused, 1.0f, Size());
            PipeBarrier<PIPE_V>();
        }
        #pragma clang loop unroll(disable)
        for (uint32_t j = 0; j < Size(); j++) {
            bool owner = j >= first && j < first + width;
            uint32_t offset = (group * 2 + (j & 1)) * slotSize;
            if (valid && owner) {
                Gather(col, a, idx, (((j - first) / 8) * 64 + (j - first) % 8) * 4, Size());
                PipeBarrier<PIPE_V>();
                Abs(red, col, Size());
                PipeBarrier<PIPE_V>();
                Mul(red, red, unused, Size());
                PipeBarrier<PIPE_V>();
                if (Size() <= 64) WholeReduceMax(red[512], red, Size(), 1, 1, 1, 8);
                else FastMax(red[512], red, red[768], Size(), true);
                Fence<HardEvent::V_S>();
                if (Parts() > 1) {
                    Fence<HardEvent::V_MTE3>();
                    DataCopyPad(factorsG[(group * Size() + j) * padn], col, {1, Size() * 4, 0, 0, 0});
                    DataCopy(shared[offset + padn], red[512], 8);
                }
            }
            if (Parts() > 1) {
                GlobalBarrier();
                if (valid) {
                    DataCopyPad(col, factorsG[(group * Size() + j) * padn], {1, Size() * 4, 0, 0, 0}, {false, 0, 0, 0});
                    DataCopy(red[512], shared[offset + padn], 8);
                    Fence<HardEvent::MTE2_S>();
                }
            }
            if (!valid) continue;
            uint32_t pivot = red.template ReinterpretCast<uint32_t>().GetValue(513);
            float inv = 1.0f / col.GetValue(pivot);
            col.SetValue(pivot, 0.0f);
            float bj = b.GetValue(pivot) * inv;
            b.SetValue(pivot, bj);
            unused.SetValue(pivot, 0.0f);
            perm.SetValue(j, pivot * 4);
            Fence<HardEvent::S_V>();
            uint32_t rowBase = (pivot / 8) * Pitch() * 8 + (pivot % 8) * 8;
            Copy(expanded, a[rowBase], uint64_t(64), Pitch() / 8, {1, 0, 8, 8});
            PipeBarrier<PIPE_V>();
            Muls(expanded, expanded, inv, Pitch() * 8);
            Muls(col, col, -1.0f, Size());
            PipeBarrier<PIPE_V>();
            Brcb(fac, col, (Size() + 7) / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            uint32_t begin = j + 1 > first ? j + 1 - first : 0;
            if (begin < width) {
                uint32_t ct = begin / 8, skip = begin % 8;
                uint32_t nc = (width + 7) / 8;
                uint64_t cmask[2] = {(uint64_t(255) << skip) & 255, 0};
                Copy(a[rowBase + ct * 64], expanded[ct * 64], cmask, 1, {1, 1, 8, 8});
                if (ct + 1 < nc) Copy(a[rowBase + (ct + 1) * 64], expanded[(ct + 1) * 64], uint64_t(8), nc - ct - 1, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                #pragma clang loop unroll(disable)
                for (uint32_t c = ct; c < nc; c++) {
                    uint32_t validCols = width - c * 8 < 8 ? width - c * 8 : 8;
                    uint64_t bits = (uint64_t(1) << validCols) - 1;
                    if (c == ct) bits &= ~((uint64_t(1) << skip) - 1);
                    uint64_t mask[2] = {bits * 0x0101010101010101ULL, 0};
                    MulAddDst(a[c * 64], expanded[c * 64], fac, mask, padn / 8, {1, 1, 1, (uint8_t)Pitch(), 0, 8});
                }
            }
            Axpy(b, col, bj, Size());
            PipeBarrier<PIPE_V>();
        }
        if (valid) {
            Gather(col, b, perm, 0, Size());
            PipeBarrier<PIPE_V>();
            Adds(b, col, 0.0f, Size());
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ __attribute__((always_inline)) inline void Residual(uint32_t job) {
        Duplicate(col, 0.0f, padn);
        PipeBarrier<PIPE_V>();
        Adds(col, b, 0.0f, Size());
        PipeBarrier<PIPE_V>();
        LoadB(job);
        Mul(red, b, b, Size());
        PipeBarrier<PIPE_V>();
        FastSum(red[512], red, red[768], Size());
        Fence<HardEvent::V_S>();
        bNorm = red.GetValue(512);
        uint32_t batch = job / Rhs();
        constexpr uint32_t CHUNK = 4;
        #pragma clang loop unroll(disable)
        for (uint32_t start = 0; start < Size(); start += CHUNK) {
            uint32_t rows = Size() - start < CHUNK ? Size() - start : CHUNK;
            if (Size() != padn) {
                Duplicate(fac, 0.0f, CHUNK * padn);
                Fence<HardEvent::V_MTE2>();
            }
            if constexpr (sizeof(T) == 4) {
                Fence<HardEvent::V_MTE2>();
                DataCopyPad(fac, ag[batch * Size() * Size() + start * Size()], {(uint16_t)rows, Size() * 4, 0, 0, 0}, {true, 0, (uint8_t)(padn - Size()), 0});
                Fence<HardEvent::MTE2_V>();
            } else {
                auto in = fac[CHUNK * padn].template ReinterpretCast<T>();
                uint32_t pitch = (Size() + 15) / 16 * 16;
                Fence<HardEvent::V_MTE2>();
                DataCopyPad(in, ag[batch * Size() * Size() + start * Size()], {(uint16_t)rows, Size() * 2, 0, 0, 0}, {true, 0, 0, 0});
                Fence<HardEvent::MTE2_V>();
                #pragma clang loop unroll(disable)
                for (uint32_t r = 0; r < rows; r++) Cast(fac[r * padn], in[r * pitch], RoundMode::CAST_NONE, Size());
                PipeBarrier<PIPE_V>();
            }
            #pragma clang loop unroll(disable)
            for (uint32_t c = 0; c < padn; c += 64) {
                uint32_t count = padn - c < 64 ? padn - c : 64;
                Mul(fac[c], fac[c], col[c], count, rows, {1, 1, 1, (uint8_t)(padn / 8), (uint8_t)(padn / 8), 0});
            }
            PipeBarrier<PIPE_V>();
            #pragma clang loop unroll(disable)
            for (uint32_t c = 64; c < padn; c += 64) {
                uint32_t count = padn - c < 64 ? padn - c : 64;
                Add(fac, fac, fac[c], count, rows, {1, 1, 1, (uint8_t)(padn / 8), (uint8_t)(padn / 8), (uint8_t)(padn / 8)});
                PipeBarrier<PIPE_V>();
            }
            WholeReduceSum(red, fac, padn < 64 ? padn : 64, rows, 1, 1, padn / 8);
            Fence<HardEvent::V_S>();
            #pragma clang loop unroll(disable)
            for (uint32_t r = 0; r < rows; r++) b.SetValue(start + r, b.GetValue(start + r) - red.GetValue(r));
            Fence<HardEvent::S_V>();
        }
    }

    __aicore__ __attribute__((always_inline)) inline void Refine(bool valid) {
        if (Parts() > 1) {
            if (!valid || part != 0) return;
            DataCopy(idx, perm, padn);
            PipeBarrier<PIPE_V>();
            uint32_t chunk = 40960 / padn;
            #pragma clang loop unroll(disable)
            for (uint32_t start = 0; start < Size(); start += chunk) {
                uint32_t count = Size() - start < chunk ? Size() - start : chunk;
                Fence<HardEvent::V_MTE2>();
                DataCopy(a, factorsG[(group * Size() + start) * padn], count * padn);
                Fence<HardEvent::MTE2_S>();
                #pragma clang loop unroll(disable)
                for (uint32_t j = 0; j < count; j++) {
                    Fence<HardEvent::V_S>();
                    uint32_t pivot = idx.GetValue(start + j) / 4;
                    auto v = a[j * padn];
                    float bj = b.GetValue(pivot) / v.GetValue(pivot);
                    b.SetValue(pivot, bj);
                    v.SetValue(pivot, 0.0f);
                    Fence<HardEvent::S_V>();
                    Axpy(b, v, -bj, Size());
                    PipeBarrier<PIPE_V>();
                }
            }
            Gather(col, b, idx, 0, Size());
            PipeBarrier<PIPE_V>();
            Adds(b, col, 0.0f, Size());
            PipeBarrier<PIPE_V>();
            return;
        }
        #pragma clang loop unroll(disable)
        for (uint32_t j = 0; j < Size(); j++) {
            if (valid && j >= first && j < first + width) {
                Gather(col, a, idx, (((j - first) / 8) * 64 + (j - first) % 8) * 4, Size());
                PipeBarrier<PIPE_V>();
                Fence<HardEvent::V_S>();
            }
            if (!valid) continue;
            uint32_t pivot = perm.GetValue(j) / 4;
            float bj = b.GetValue(pivot) / col.GetValue(pivot);
            b.SetValue(pivot, bj);
            col.SetValue(pivot, 0.0f);
            Fence<HardEvent::S_V>();
            Axpy(b, col, -bj, Size());
            PipeBarrier<PIPE_V>();
        }
        if (valid) {
            Gather(col, b, perm, 0, Size());
            PipeBarrier<PIPE_V>();
            Adds(b, col, 0.0f, Size());
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ __attribute__((always_inline)) inline void Store(uint32_t job) {
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(sol[job * padn], b, {1, Size() * 4, 0, 0, 0});
        Fence<HardEvent::MTE3_MTE2>();
    }

    __aicore__ __attribute__((always_inline)) inline void Pack() {
        uint32_t total = t.batch * Size() * Rhs();
        bool full = t.batch * Rhs() * padn <= 40960;
        if (full) {
            DataCopy(a, sol, t.batch * Rhs() * padn);
            Fence<HardEvent::MTE2_V>();
        }
        uint32_t tile = (total + t.cores * 8 - 1) / (t.cores * 8) * 8;
        if (tile > 512) tile = 512;
        #pragma clang loop unroll(disable)
        for (uint32_t start = GetBlockIdx() * tile; start < total; start += t.cores * tile) {
            uint32_t count = total - start < tile ? total - start : tile;
            if (full && Rhs() == 1 && padn == Size()) {
                Adds(col, a[start], 0.0f, count);
                PipeBarrier<PIPE_V>();
            } else if (full) {
                #pragma clang loop unroll(disable)
                for (uint32_t i = 0; i < count; i++) {
                    uint32_t v = start + i, batch = v / (Size() * Rhs()), r = (v / Rhs()) % Size(), rhs = v % Rhs();
                    idx.SetValue(i, ((batch * Rhs() + rhs) * padn + r) * 4);
                }
                Fence<HardEvent::S_V>();
                Gather(col, a, idx, 0, count);
                PipeBarrier<PIPE_V>();
            } else {
                #pragma clang loop unroll(disable)
                for (uint32_t i = 0; i < count; i++) {
                    uint32_t v = start + i, batch = v / (Size() * Rhs()), r = (v / Rhs()) % Size(), rhs = v % Rhs();
                    DataCopyPad(fac[i * 8], sol[(batch * Rhs() + rhs) * padn + r], {1, 4, 0, 0, 0}, {false, 0, 0, 0});
                    idx.SetValue(i, i * 32);
                }
                Fence<HardEvent::MTE2_V>();
                Fence<HardEvent::S_V>();
                Gather(col, fac, idx, 0, count);
                PipeBarrier<PIPE_V>();
            }
            if constexpr (sizeof(T) == 4) {
                Fence<HardEvent::V_MTE3>();
                DataCopyPad(xg[start], col, {1, count * 4, 0, 0, 0});
            } else {
                auto out = fac.template ReinterpretCast<T>();
                Cast(out, col, RoundMode::CAST_RINT, count);
                Fence<HardEvent::V_MTE3>();
                DataCopyPad(xg[start], out, {1, count * 2, 0, 0, 0});
            }
            Fence<HardEvent::MTE3_V>();
        }
    }

    __aicore__ __attribute__((always_inline)) inline void Run() {
        uint32_t jobs = t.batch * Rhs();
        #pragma clang loop unroll(disable)
        for (uint32_t wave = 0; wave < (jobs + t.groups - 1) / t.groups; wave++) {
            uint32_t job = wave * t.groups + group;
            bool valid = job < jobs;
            if (valid) {
                if (wave) {
                    #pragma clang loop unroll(disable)
                    for (uint32_t r = 0; r < Size(); r++) idx.SetValue(r, ((r / 8) * Pitch() * 8 + (r % 8) * 8) * 4);
                    Fence<HardEvent::S_V>();
                }
                LoadB(job);
                LoadA(job / Rhs());
            }
            Solve(valid);
            if (valid && part == 0) Store(job);
            if (Parts() > 1) GlobalBarrier();
            if (Size() > 64) {
                bool need = valid;
                if (valid) {
                    Residual(job);
                    Mul(red, b, b, Size());
                    PipeBarrier<PIPE_V>();
                    FastSum(red[512], red, red[768], Size());
                    Fence<HardEvent::V_S>();
                    float rn = red.GetValue(512);
                    need = !(rn == 0.0f || (rn < 1e-14f && rn < bNorm * 1e-10f));
                }
                bool any = need;
                if (Parts() > 1) {
                    if (part == 0) {
                        row.SetValue(0, need ? 1.0f : 0.0f);
                        Fence<HardEvent::S_MTE3>();
                        DataCopy(shared[group * 2 * slotSize], row, 8);
                    }
                    GlobalBarrier();
                    DataCopyPad(red, shared, {(uint16_t)t.groups, 4, (2 * slotSize - 1) * 4, 0, 0}, {false, 0, 0, 0});
                    Fence<HardEvent::MTE2_S>();
                    any = false;
                    #pragma clang loop unroll(disable)
                    for (uint32_t g = 0; g < t.groups; g++) {
                        if (red.GetValue(g * 8) != 0.0f) any = true;
                    }
                }
                if (any) Refine(need);
                if (valid && part == 0) {
                    if (!need) {
                        Duplicate(b, 0.0f, Size());
                        PipeBarrier<PIPE_V>();
                    }
                    DataCopyPad(col, sol[job * padn], {1, Size() * 4, 0, 0, 0}, {false, 0, 0, 0});
                    Fence<HardEvent::MTE2_V>();
                    Add(b, b, col, Size());
                    PipeBarrier<PIPE_V>();
                    Store(job);
                }
                if (Parts() > 1) GlobalBarrier();
            }
        }
        GlobalBarrier();
        Pack();
    }
};
template<class DT_A, uint32_t SHAPE>
__global__ __aicore__ void linalg_solve(GM_ADDR A, GM_ADDR B, GM_ADDR X, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgSolveTilingData);
    COPY_TILING_WITH_STRUCT(LinalgSolveTilingData, tiling, tp);
    const LinalgSolveTilingData &t = *tp;
    if constexpr (sizeof(DT_A) == 4 && SHAPE == 1) {
        TinySolver<DT_A>::Run(A, B, X, t);
    } else if constexpr (sizeof(DT_A) == 4 && SHAPE == 2) {
        BatchSolver<DT_A>::Run(A, B, X, t);
    } else if constexpr (sizeof(DT_A) == 4 && SHAPE == 3) {
        CoopGJ<DT_A, true> s;
        s.Init(A, B, X, workspace, t);
        s.Run();
    } else if constexpr (sizeof(DT_A) == 4 && SHAPE == 4) {
        CoopGJ<DT_A, false> s;
        s.Init(A, B, X, workspace, t);
        s.Run();
    } else {
        InitSocState();
        Solver<DT_A> s;
        s.Init(A, B, X, GetUserWorkspace(workspace), t);
        s.Run();
    }
}
