#include "kernel_operator.h"
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"

using namespace AscendC;
#include "det_small.h"

template <typename T,bool ALLOW_FAST=true>
class KernelLinalgDet {
public:
    __aicore__ inline KernelLinalgDet(TPipe &external):pipe(external){}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR det, GM_ADDR workspace,
                               const LinalgDetConfig &tiling) {
        t = tiling;
        lane = GetBlockIdx() % t.coresPerMatrix;
        group = GetBlockIdx() / t.coresPerMatrix;
        first = lane * t.columnsPerCore;
        count = first < t.n ? Min(t.columnsPerCore, t.n - first) : 0;
        input.SetGlobalBuffer((__gm__ T *)a);
        output.SetGlobalBuffer((__gm__ float *)det);
        if(t.coresPerMatrix>1){
            shared.SetGlobalBuffer((__gm__ float *)workspace + group * t.sharedStride);
            allShared.SetGlobalBuffer((__gm__ float *)workspace);
        }
        pipe.InitBuffer(panelBuf, 2 * t.columnsPerCore * t.paddedN * sizeof(float));
        pipe.InitBuffer(vectorBuf, 2 * t.paddedN * sizeof(float));
        pipe.InitBuffer(tempBuf, 8 * t.paddedN * sizeof(float));
        pipe.InitBuffer(rawBuf, t.paddedN * 8 * sizeof(float));
        pipe.InitBuffer(indexBuf, t.paddedN * sizeof(uint32_t));
        pipe.InitBuffer(metaBuf, 32);
        pipe.InitBuffer(scaleBuf, ((t.columnsPerCore + 7) / 8 * 8) * sizeof(int32_t));
        if (t.n > 128 && !t.cubeMode && t.columnsPerCore == 16) {
            pipe.InitBuffer(lowerBuf, t.columnsPerCore * t.paddedN * sizeof(float));
            lower = lowerBuf.Get<float>();
            pipe.InitBuffer(correctionBuf, t.columnsPerCore * t.paddedN * sizeof(float));
            correction = correctionBuf.Get<float>();
        }
        panel = panelBuf.Get<float>();
        panelLo = panel[t.columnsPerCore * t.paddedN];
        v = vectorBuf.Get<float>();
        vLo = v[t.paddedN];
        s0 = tempBuf.Get<float>();
        s1 = s0[t.paddedN]; s2 = s1[t.paddedN];
        ph = s2[t.paddedN]; pl = ph[t.paddedN];
        sh = pl[t.paddedN]; sl = sh[t.paddedN];
        raw = rawBuf.Get<float>();
        indices = indexBuf.Get<uint32_t>();
        meta = metaBuf.Get<float>();
        scales = scaleBuf.Get<int32_t>();
        const bool directFast = ALLOW_FAST && t.coresPerMatrix == 1 && t.n >= 3 && t.n <= 128;
        if (!directFast) {
            for (uint32_t r = 0; r < t.paddedN; ++r) indices.SetValue(r, r * 32);
            Fence<HardEvent::S_V>();
            Duplicate(meta, 0.0f, 8);
            Fence<HardEvent::V_S>();
        }
    }

    __aicore__ inline void Process() {
        for (uint32_t round = 0; round < (t.batch + t.groups - 1) / t.groups; ++round) {
            const uint32_t batch = round * t.groups + group;
            const bool active = batch < t.batch;
            if (t.n > 128 && !t.cubeMode && t.columnsPerCore == 16 && BlockFast(batch, active)) continue;
            if (ALLOW_FAST && active && t.coresPerMatrix == 1 && t.n >= 3 && t.n <= 128 && Fast(batch)) continue;
            if (active) Load(batch);
            float product = 1.0f;
            int32_t exponent = 0;
            for (uint32_t k = 0; k < t.n; ++k) {
                if (active && k >= first && k < first + count) MakePivot(k);
                if (t.coresPerMatrix > 1) {
                    SyncAll<true>();
                    if (active) {
                        DataCopy(v, shared, 2 * t.paddedN);
                        DataCopy(meta, shared[2 * t.paddedN], 8);
                        Fence<HardEvent::MTE2_V>();
                        Fence<HardEvent::MTE2_S>();
                    }
                }
                if (active) {
                    const uint32_t pivotRow = meta.template ReinterpretCast<uint32_t>().GetValue(0);
                    float pivot = meta.GetValue(1);
                    exponent += static_cast<int32_t>(meta.GetValue(2));
                    if (pivotRow != k) product = -product;
                    // Keep the intermediate product near one: the final determinant
                    // may be small even when a partial product would overflow FP32.
                    if (pivot == 0.0f) {
                        product = 0.0f;
                    } else {
                        union { float f; uint32_t u; } bits;
                        bits.f = pivot;
                        exponent += static_cast<int32_t>((bits.u >> 23) & 255) - 127;
                        bits.u = (bits.u & 0x807fffffU) | (127U << 23);
                        product *= bits.f;
                        bits.f = product;
                        if (product != 0.0f) {
                            exponent += static_cast<int32_t>((bits.u >> 23) & 255) - 127;
                            bits.u = (bits.u & 0x807fffffU) | (127U << 23);
                            product = bits.f;
                        }
                    }
                    Update(k, pivotRow);
                }
                if (t.coresPerMatrix > 1) SyncAll<true>();
            }
            if (active && lane == 0) {
                meta.SetValue(0, Rescale(product, exponent));
                Fence<HardEvent::S_MTE3>();
                DataCopyExtParams cp{1, sizeof(float), 0, 0, 0};
                DataCopyPad(output[batch], meta, cp);
                Fence<HardEvent::MTE3_S>();
            }
        }
    }

private:
    template <HardEvent event>
    __aicore__ inline void Fence() {
        const auto id = pipe.FetchEventID(event);
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
    __aicore__ inline float PowerOfTwo(int32_t exponent) {
        union { uint32_t u; float f; } bits;
        bits.u = static_cast<uint32_t>(exponent + 127) << 23;
        return bits.f;
    }
    __aicore__ inline float Rescale(float value, int32_t exponent) {
        if (value == 0.0f) return value;
        while (exponent > 120) { value *= PowerOfTwo(120); exponent -= 120; }
        while (exponent < -120) { value *= PowerOfTwo(-120); exponent += 120; }
        return value * PowerOfTwo(exponent);
    }

    // A complete rank-one update is one strided vector multiply and one
    // contiguous subtraction. Small matrices therefore need no cross-core sync.
    __aicore__ inline bool Fast(uint32_t batch) {
        const uint32_t n = t.n;
        const uint32_t pitch = t.paddedN;
        const uint32_t len = n * pitch;
        auto index = indices.template ReinterpretCast<int32_t>();
        CreateVecIndex(index, 0, pitch); PipeBarrier<PIPE_V>();
        Muls(index, index, static_cast<int32_t>(pitch * 4), pitch); PipeBarrier<PIPE_V>();
        DataCopyExtParams inputCopy{static_cast<uint16_t>(n), n * 4, 0, 0, 0};
        DataCopyPadExtParams<float> inputPad{true, 0, static_cast<uint8_t>(pitch - n), 0};
        DataCopyPad(panel, input[static_cast<uint64_t>(batch) * n * n], inputCopy, inputPad);
        Fence<HardEvent::MTE2_V>();
        Abs(panelLo, panel, len); PipeBarrier<PIPE_V>();
        if (len > 8192) {
            ReduceMax(ph, panelLo, s0, len / 2, false); PipeBarrier<PIPE_V>();
            ReduceMax(pl, panelLo[len / 2], s0, len / 2, false);
            PipeBarrier<PIPE_V>();
            Max(meta, ph, pl, 1);
        } else {
            ReduceMax(meta, panelLo, s0, len, false);
        }
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();
        const float magnitude = meta.GetValue(0);
        bool safe = magnitude > 1.0e-20f && magnitude < 1.0e20f;
        float product = 1.0f;
        int32_t exponent = 0;
        for (uint32_t k = 0; safe && k < n; ++k) {
            const uint32_t remaining=n-k;
            Gather(s0,panel,indices,(k*pitch+k)*4,remaining);PipeBarrier<PIPE_V>();Fence<HardEvent::V_S>();
            const uint32_t row=k;
            const float pivot=s0.GetValue(0);
            const float absolute = pivot < 0 ? -pivot : pivot;
            // Tiny pivots amplify FP32 error. Recompute such matrices with
            // the compensated factorization rather than accepting that error.
            if (absolute < 0.0001f * magnitude) { safe = false; break; }
            if (k+1<n) {
                Muls(v,s0,-1.0f/pivot,remaining);PipeBarrier<PIPE_V>();
                Brcb(raw,v,(remaining+7)/8,{1,8});PipeBarrier<PIPE_V>();
                for (uint32_t offset = (k+1)/8*8; offset < pitch; offset += 64) {
                    const uint32_t mask = Min(64U, pitch - offset);
                    MulAddDst(panel[(k+1)*pitch+offset],panel[k*pitch+offset],raw[8],mask,n-k-1,
                        {1, 1, 0, static_cast<uint8_t>(pitch / 8), 0, 1});
                }
            }
            union { uint32_t u; float f; } value;
            value.f=pivot;exponent+=static_cast<int32_t>((value.u>>23)&255)-127;
            value.u=(value.u&0x807fffffU)|(127U<<23);product*=value.f;
            if(row!=k)product=-product;
            value.f=product;exponent+=static_cast<int32_t>((value.u>>23)&255)-127;
            value.u=(value.u&0x807fffffU)|(127U<<23);product=value.f;
            PipeBarrier<PIPE_V>();
        }
        if (safe) {
            PipeBarrier<PIPE_V>();
            meta.SetValue(0, Rescale(product, exponent));
            Fence<HardEvent::S_MTE3>();
            DataCopyExtParams cp{1, sizeof(float), 0, 0, 0};
            DataCopyPad(output[batch], meta, cp);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_S>();
        } else {
            CreateVecIndex(index, 0, pitch); PipeBarrier<PIPE_V>();
            ShiftLeft(indices, indices, uint32_t(5), pitch); PipeBarrier<PIPE_V>();
        }
        return safe;
    }

    __aicore__ inline void MultiplyDet(float pivot, float &product, int32_t &exponent) {
        if (pivot == 0.0f) { product = 0.0f; return; }
        union { uint32_t u; float f; } bits;
        bits.f = pivot;
        exponent += static_cast<int32_t>((bits.u >> 23) & 255) - 127;
        bits.u = (bits.u & 0x807fffffU) | (127U << 23);
        product *= bits.f;
        if (product != 0.0f) {
            bits.f = product;
            exponent += static_cast<int32_t>((bits.u >> 23) & 255) - 127;
            bits.u = (bits.u & 0x807fffffU) | (127U << 23);
            product = bits.f;
        }
    }

    // A 16-column panel is factored by its owning core. Its permutation and
    // lower factor are broadcast once, amortizing synchronization over 16 pivots.
    __aicore__ inline void PanelRankOne(uint32_t firstRow, int32_t storeColumn = -1) {
        const uint32_t start = firstRow / 8 * 8;
        const uint32_t rows = t.paddedN - start;
        if (!rows) return;
        auto hi = panel[start * 16];
        auto prod = panelLo[start * 16];
        auto corr = correction[start * 16];
        Brcb(prod, v[start], rows / 8, {2, 16});
        Brcb(prod[8], v[start], rows / 8, {2, 16});
        if (storeColumn >= 0) {
            PipeBarrier<PIPE_V>();
            uint64_t mask[2] = {uint64_t(1) << storeColumn, 0};
            for (uint32_t r = 0; r < rows; r += 128) {
                Adds(hi[r * 16], prod[r * 16], 0.0f, mask,
                     Min(128U, rows - r), {1, 1, 2, 2});
                Muls(corr[r * 16], corr[r * 16], 0.0f, mask,
                     Min(128U, rows - r), {1, 1, 2, 2});
            }
        }
        Adds(raw, sh, 0.0f, 16);
        Adds(raw[16], sh, 0.0f, 16);
        Adds(raw[32], sh, 0.0f, 16);
        Adds(raw[48], sh, 0.0f, 16);
        PipeBarrier<PIPE_V>();
        Mul(prod, prod, raw, 64, rows / 4, {1, 1, 1, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        // Retain the rounding residual of each update; repeated tiny updates
        // otherwise lose accuracy even on well-conditioned matrices.
        Sub(prod, prod, corr, rows * 16); PipeBarrier<PIPE_V>();
        Sub(corr, hi, prod, rows * 16); PipeBarrier<PIPE_V>();
        Sub(hi, corr, hi, rows * 16); PipeBarrier<PIPE_V>();
        Add(prod, hi, prod, rows * 16); PipeBarrier<PIPE_V>();
        Adds(hi, corr, 0.0f, rows * 16); PipeBarrier<PIPE_V>();
        Muls(corr, prod, -1.0f, rows * 16); PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SwapPanelRows(uint32_t a, uint32_t b) {
        if (a == b) return;
        Adds(sh, panel[a * 16], 0.0f, 16);
        Adds(sl, panel[b * 16], 0.0f, 16); PipeBarrier<PIPE_V>();
        Adds(panel[a * 16], sl, 0.0f, 16);
        Adds(panel[b * 16], sh, 0.0f, 16); PipeBarrier<PIPE_V>();
        Adds(sh, correction[a * 16], 0.0f, 16);
        Adds(sl, correction[b * 16], 0.0f, 16); PipeBarrier<PIPE_V>();
        Adds(correction[a * 16], sl, 0.0f, 16);
        Adds(correction[b * 16], sh, 0.0f, 16); PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool BlockFast(uint32_t batch, bool active) {
        const uint32_t rows = t.paddedN;
        const uint32_t panelSize = 16 * rows;
        const uint32_t metaOffset = panelSize;
        auto index = indices.template ReinterpretCast<int32_t>();
        CreateVecIndex(index, 0, rows); PipeBarrier<PIPE_V>();
        ShiftLeft(indices, indices, uint32_t(6), rows);
        float originalMagnitude = 1.0f;
        bool risky = false;
        if (active) {
            Duplicate(panel, 0.0f, panelSize);
            Duplicate(correction, 0.0f, panelSize);
            Fence<HardEvent::V_MTE2>();
            const uint32_t copyPitch = (count + 7) / 8 * 8;
            DataCopyExtParams cp{static_cast<uint16_t>(t.n), count * 4,
                (t.n - count) * 4, (16 - copyPitch) / 8, 0};
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(copyPitch - count), 0};
            DataCopyPad(panel, input[static_cast<uint64_t>(batch) * t.n * t.n + first], cp, pad);
            Fence<HardEvent::MTE2_V>();
            Abs(panelLo, panel, panelSize); PipeBarrier<PIPE_V>();
            ReduceMax(meta, panelLo, s0, panelSize, false);
            Fence<HardEvent::V_S>();
            originalMagnitude = meta.GetValue(0);
            risky = !(originalMagnitude > 1.0e-20f && originalMagnitude < 1.0e20f);
        }
        float product = 1.0f;
        int32_t exponent = 0;
        for (uint32_t block = 0; block < t.coresPerMatrix; ++block) {
            const uint32_t start = block * 16;
            const uint32_t width = Min(16U, t.n - start);
            if (active && lane == block) {
                float blockProduct = 1.0f;
                int32_t blockExponent = 0;
                for (uint32_t i = 0; i < width; ++i) {
                    const uint32_t k = start + i;
                    Gather(s0, panel, indices, i * 4, rows); PipeBarrier<PIPE_V>();
                    Abs(s1, s0, rows); PipeBarrier<PIPE_V>();
                    if (k) { Duplicate(s1, 0.0f, k); PipeBarrier<PIPE_V>(); }
                    ReduceMax(meta, s1, s2, rows, true);
                    Fence<HardEvent::V_S>();
                    const uint32_t row = meta.template ReinterpretCast<uint32_t>().GetValue(1);
                    const float pivot = s0.GetValue(row);
                    const float absolute = pivot < 0 ? -pivot : pivot;
                    if (absolute < 0.0625f * originalMagnitude) risky = true;
                    MultiplyDet(pivot, blockProduct, blockExponent);
                    if (row != k) blockProduct = -blockProduct;
                    scales.SetValue(i, static_cast<int32_t>(row));
                    if (row != k) {
                        s0.SetValue(row, s0.GetValue(k));
                        s0.SetValue(k, pivot);
                    }
                    SwapPanelRows(k, row);
                    Fence<HardEvent::S_V>();
                    Muls(v, s0, pivot == 0.0f ? 0.0f : 1.0f / pivot, rows);
                    PipeBarrier<PIPE_V>();
                    Adds(sh, panel[k * 16], 0.0f, 16); PipeBarrier<PIPE_V>();
                    Duplicate(sh, 0.0f, i + 1);
                    Duplicate(v, 0.0f, k + 1); PipeBarrier<PIPE_V>();
                    PanelRankOne(k + 1, static_cast<int32_t>(i));
                }
                Fence<HardEvent::V_S>();
                meta.SetValue(0, blockProduct);
                meta.template ReinterpretCast<int32_t>().SetValue(1, blockExponent);
                meta.template ReinterpretCast<int32_t>().SetValue(2, risky ? 1 : 0);
                Fence<HardEvent::V_MTE3>();
                Fence<HardEvent::S_MTE3>();
                DataCopy(shared, panel, panelSize);
                DataCopy(shared[metaOffset], meta, 8);
                DataCopy(shared[metaOffset + 8], scales.template ReinterpretCast<float>(), 16);
                Fence<HardEvent::MTE3_MTE2>();
            }
            SyncAll<true>();
            if (active) {
                DataCopy(meta, shared[metaOffset], 8);
                DataCopy(scales.template ReinterpretCast<float>(), shared[metaOffset + 8], 16);
                if (lane > block) DataCopy(lower, shared, panelSize);
                Fence<HardEvent::MTE2_S>();
                Fence<HardEvent::MTE2_V>();
                const float blockProduct = meta.GetValue(0);
                exponent += meta.template ReinterpretCast<int32_t>().GetValue(1);
                risky = risky || meta.template ReinterpretCast<int32_t>().GetValue(2) != 0;
                MultiplyDet(blockProduct, product, exponent);
                if (lane > block) {
                    for (uint32_t i = 0; i < width; ++i)
                        SwapPanelRows(start + i, static_cast<uint32_t>(scales.GetValue(i)));
                    for (uint32_t i = 0; i < width; ++i) {
                        const uint32_t k = start + i;
                        Gather(v, lower, indices, i * 4, rows); PipeBarrier<PIPE_V>();
                        Duplicate(v, 0.0f, k + 1);
                        Adds(sh, panel[k * 16], 0.0f, 16); PipeBarrier<PIPE_V>();
                        PanelRankOne(k + 1);
                    }
                }
            }
            SyncAll<true>();
        }
        if (lane == 0) {
            meta.template ReinterpretCast<int32_t>().SetValue(0, active && risky ? 1 : 0);
            Fence<HardEvent::S_MTE3>();
            DataCopy(shared[metaOffset], meta, 8);
            Fence<HardEvent::MTE3_MTE2>();
        }
        SyncAll<true>();
        bool anyRisk = false;
        for (uint32_t g = 0; g < t.groups; ++g) {
            DataCopy(meta, allShared[g * t.sharedStride + metaOffset], 8);
            Fence<HardEvent::MTE2_S>();
            anyRisk = anyRisk || meta.template ReinterpretCast<int32_t>().GetValue(0) != 0;
            Fence<HardEvent::S_MTE2>();
        }
        SyncAll<true>();
        if (!anyRisk && active && lane == 0) {
            meta.SetValue(0, Rescale(product, exponent));
            Fence<HardEvent::S_MTE3>();
            DataCopyExtParams cp{1, sizeof(float), 0, 0, 0};
            DataCopyPad(output[batch], meta, cp);
            Fence<HardEvent::MTE3_S>();
        }
        if (anyRisk) {
            CreateVecIndex(index, 0, rows); PipeBarrier<PIPE_V>();
            ShiftLeft(indices, indices, uint32_t(5), rows); PipeBarrier<PIPE_V>();
        }
        return !anyRisk;
    }

    // Two FP32 components retain the rounding errors in LU updates. This is
    // needed for determinants of poorly conditioned but nonsingular matrices.
    struct Pair { float hi, lo; };
    __aicore__ inline Pair Sum(Pair a, Pair b) {
        const volatile float h = a.hi + b.hi;
        const volatile float z = h - a.hi;
        const volatile float e = ((a.hi - (h - z)) + (b.hi - z)) + (a.lo + b.lo);
        const volatile float r = h + e;
        return {r, e - (r - h)};
    }
    __aicore__ inline Pair Product(Pair a, Pair b) {
        const volatile float ca = 4097.0f * a.hi, cb = 4097.0f * b.hi;
        const volatile float ah = ca - (ca - a.hi), al = a.hi - ah;
        const volatile float bh = cb - (cb - b.hi), bl = b.hi - bh;
        const volatile float p = a.hi * b.hi;
        const volatile float e = (((ah * bh - p) + ah * bl) + al * bh) + al * bl;
        return Sum({p, e}, {a.hi * b.lo + a.lo * b.hi, a.lo * b.lo});
    }
    __aicore__ inline Pair Divide(Pair a, Pair b) {
        const volatile float q = a.hi / b.hi;
        const Pair p = Product(b, {q, 0.0f});
        const Pair r = Sum(a, {-p.hi, -p.lo});
        return Sum({q, 0.0f}, {(r.hi + r.lo) / b.hi, 0.0f});
    }
    __aicore__ inline void VectorProduct(LocalTensor<float> ah, LocalTensor<float> al, Pair b) {
        const uint32_t len = t.paddedN;
        const volatile float cb = 4097.0f * b.hi;
        const volatile float bh = cb - (cb - b.hi), bl = b.hi - bh;
        Muls(ph, ah, b.hi, len);
        Muls(s0, ah, 4097.0f, len); PipeBarrier<PIPE_V>();
        Sub(s1, s0, ah, len); PipeBarrier<PIPE_V>();
        Sub(s0, s0, s1, len); PipeBarrier<PIPE_V>();
        Sub(s1, ah, s0, len);
        Muls(pl, s0, static_cast<float>(bh), len); PipeBarrier<PIPE_V>();
        Sub(pl, pl, ph, len);
        Muls(s2, s0, static_cast<float>(bl), len); PipeBarrier<PIPE_V>();
        Add(pl, pl, s2, len);
        Muls(s2, s1, static_cast<float>(bh), len); PipeBarrier<PIPE_V>();
        Add(pl, pl, s2, len);
        Muls(s2, s1, static_cast<float>(bl), len); PipeBarrier<PIPE_V>();
        Add(pl, pl, s2, len);
        Muls(s2, ah, b.lo, len); PipeBarrier<PIPE_V>();
        Add(pl, pl, s2, len);
        Muls(s2, al, b.hi, len); PipeBarrier<PIPE_V>();
        Add(pl, pl, s2, len); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void VectorSubtract(LocalTensor<float> ah, LocalTensor<float> al) {
        const uint32_t len = t.paddedN;
        Sub(sh, ah, ph, len); PipeBarrier<PIPE_V>();
        Sub(s0, sh, ah, len); PipeBarrier<PIPE_V>();
        Sub(s1, sh, s0, len); PipeBarrier<PIPE_V>();
        Sub(s1, ah, s1, len);
        Add(s0, ph, s0, len); PipeBarrier<PIPE_V>();
        Sub(sl, s1, s0, len); PipeBarrier<PIPE_V>();
        Add(sl, sl, al, len); PipeBarrier<PIPE_V>();
        Sub(sl, sl, pl, len); PipeBarrier<PIPE_V>();
        Add(s0, sh, sl, len); PipeBarrier<PIPE_V>();
        Sub(s1, s0, sh, len); PipeBarrier<PIPE_V>();
        Sub(al, sl, s1, len);
        Adds(ah, s0, 0.0f, len); PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Load(uint32_t batch) {
        const uint64_t base = static_cast<uint64_t>(batch) * t.n * t.n;
        Duplicate(panelLo, 0.0f, t.columnsPerCore * t.paddedN);
        Duplicate(raw, 0.0f, t.paddedN * 8);
        Fence<HardEvent::V_MTE2>();
        for (uint32_t c = 0; c < count; ++c) {
            // A strided DMA followed by gather transposes a column directly into UB.
            DataCopyExtParams cp{static_cast<uint16_t>(t.n), sizeof(float),
                                 (t.n - 1) * 4U, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(raw, input[base + first + c], cp, pad);
            Fence<HardEvent::MTE2_V>();
            auto col = panel[c * t.paddedN];
            Gather(col, raw, indices, 0, t.paddedN);
            Fence<HardEvent::V_S>();
            uint32_t magnitude = 0;
            const auto bits = col.template ReinterpretCast<uint32_t>();
            for (uint32_t r = 0; r < t.n; ++r) {
                const uint32_t absBits = bits.GetValue(r) & 0x7fffffffU;
                magnitude = absBits > magnitude ? absBits : magnitude;
            }
            int32_t scale = magnitude ? static_cast<int32_t>(magnitude >> 23) - 127 : 0;
            // Exact powers of two equilibrate columns without rounding their mantissas.
            scale = scale < -120 ? -120 : (scale > 120 ? 120 : scale);
            scales.SetValue(c, scale);
            Muls(col, col, PowerOfTwo(-scale), t.paddedN);
            Fence<HardEvent::V_MTE2>();
        }
        Fence<HardEvent::V_S>();
    }

    __aicore__ inline void MakePivot(uint32_t k) {
        auto col = panel[(k - first) * t.paddedN];
        auto colLo = panelLo[(k - first) * t.paddedN];
        uint32_t row = k;
        float maxAbs = 0.0f;
        for (uint32_t r = k; r < t.n; ++r) {
            const float x = col.GetValue(r);
            const float a = x < 0.0f ? -x : x;
            if (a > maxAbs) { maxAbs = a; row = r; }
        }
        const Pair pivot = {col.GetValue(row), colLo.GetValue(row)};
        if (row != k) {
            const float old = col.GetValue(k);
            col.SetValue(row, old);
            col.SetValue(k, pivot.hi);
            colLo.SetValue(row, colLo.GetValue(k));
            colLo.SetValue(k, pivot.lo);
        }
        Fence<HardEvent::S_V>();
        const Pair inv = pivot.hi == 0.0f ? Pair{0.0f, 0.0f} : Divide({1.0f, 0.0f}, pivot);
        VectorProduct(col, colLo, inv);
        Add(v, ph, pl, t.paddedN); PipeBarrier<PIPE_V>();
        Sub(s0, v, ph, t.paddedN); PipeBarrier<PIPE_V>();
        Sub(vLo, pl, s0, t.paddedN);
        Fence<HardEvent::V_S>();
        // The already eliminated rows are not part of the trailing update.
        for (uint32_t r = 0; r <= k; ++r) {
            v.SetValue(r, 0.0f); vLo.SetValue(r, 0.0f);
        }
        meta.template ReinterpretCast<uint32_t>().SetValue(0, row);
        meta.SetValue(1, pivot.hi);
        meta.SetValue(2, static_cast<float>(scales.GetValue(k - first)));
        Fence<HardEvent::S_V>();
        if (t.coresPerMatrix > 1) {
            Fence<HardEvent::V_MTE3>();
            Fence<HardEvent::S_MTE3>();
            DataCopy(shared, v, 2 * t.paddedN);
            DataCopy(shared[2 * t.paddedN], meta, 8);
            Fence<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void Update(uint32_t k, uint32_t pivotRow) {
        const uint32_t begin = k >= first ? k - first + 1 : 0;
        for (uint32_t c = begin; c < count; ++c) {
            auto col = panel[c * t.paddedN];
            auto colLo = panelLo[c * t.paddedN];
            const Pair factor = {col.GetValue(pivotRow), colLo.GetValue(pivotRow)};
            if (pivotRow != k) {
                col.SetValue(pivotRow, col.GetValue(k));
                col.SetValue(k, factor.hi);
                colLo.SetValue(pivotRow, colLo.GetValue(k));
                colLo.SetValue(k, factor.lo);
            }
            Fence<HardEvent::S_V>();
            VectorProduct(v, vLo, factor);
            VectorSubtract(col, colLo);
            Fence<HardEvent::V_S>();
        }
    }

    TPipe &pipe;
    TBuf<TPosition::VECCALC> panelBuf, vectorBuf, tempBuf, rawBuf, indexBuf, metaBuf, scaleBuf, lowerBuf, correctionBuf;
    LocalTensor<float> panel, panelLo, v, vLo, raw, meta;
    LocalTensor<float> s0, s1, s2, ph, pl, sh, sl, lower, correction;
    LocalTensor<uint32_t> indices;
    LocalTensor<int32_t> scales;
    GlobalTensor<float> input, output, shared, allShared;
    LinalgDetConfig t;
    uint32_t lane, group, first, count;
};

#include "det_cube_lu.h"
#include "det_coop64.h"
#include "det_coop64_direct_diag.h"
#include "det_cube_static.h"
#include "det_cube_wide.h"
template<typename DT_A>
__aicore__ __attribute__((noinline)) void DetRunFallback(GM_ADDR a,GM_ADDR out,GM_ADDR ws,const LinalgDetConfig &t){
    TPipe pipe;KernelLinalgDet<DT_A,false> op(pipe);op.Init(a,out,ws,t);op.Process();
}
__aicore__ inline LinalgDetConfig ReadDetConfig(GM_ADDR tiling){
    LinalgDetConfig t;
    auto dst=reinterpret_cast<uint32_t*>(&t);
    auto src=reinterpret_cast<__gm__ const uint32_t*>(tiling);
    #pragma unroll
    for(uint32_t i=0;i<sizeof(LinalgDetConfig)/sizeof(uint32_t);++i)dst[i]=src[i];
    return t;
}
template<typename DT_A>
__aicore__ __attribute__((noinline)) static void DetRunK64Fixed4Fallback(GM_ADDR a,GM_ADDR out,GM_ADDR ws,GM_ADDR tiling){
    auto f=ReadDetConfig(tiling);
    f.coresPerMatrix=8;f.columnsPerCore=32;f.sharedStride=34*256+128;
    DetRunFallback<DT_A>(a,out,ws+4*164096U*4,f);
}
template<typename DT_A>
__aicore__ __attribute__((noinline)) static void DetRunK64WaveFallback(GM_ADDR a,GM_ADDR out,GM_ADDR ws,GM_ADDR tiling){
    auto f=ReadDetConfig(tiling);
    // a/out already point at this wave; never reprocess the full input batch.
    f.batch=4;f.coresPerMatrix=8;f.columnsPerCore=32;f.sharedStride=34*256+128;
    DetRunFallback<DT_A>(a,out,ws+4*164096U*4,f);
}
__aicore__ __attribute__((noinline)) static bool DetRunFastCube(GM_ADDR a,GM_ADDR out,GM_ADDR ws,const LinalgDetConfig &t,uint32_t batch){
    TPipe fastPipe;CubeLUVector fast(fastPipe);fast.Init(a,out,ws,t);return fast.ProcessOne(batch);
}
__aicore__ __attribute__((noinline)) static bool DetRunSequentialCube(GM_ADDR a,GM_ADDR out,GM_ADDR ws,const LinalgDetConfig &t,uint32_t batch){
    TPipe fastPipe;CubeLUSequentialVector fast(fastPipe);fast.Init(a,out,ws,t);return fast.ProcessOne(batch);
}
__aicore__ __attribute__((noinline)) static bool DetRunStaticCube200(GM_ADDR a,GM_ADDR out,GM_ADDR ws,const LinalgDetConfig &t,uint32_t batch){
    DetStaticSequentialVector<200> op;op.Init(a,out,ws,t);return op.ProcessOne(batch);
}
__aicore__ __attribute__((noinline)) static bool DetRunStaticCooperativeCube200(GM_ADDR a,GM_ADDR out,GM_ADDR ws,const LinalgDetConfig &t,uint32_t batch){
    DetStaticSequentialVector<200,true> op;op.Init(a,out,ws,t);return op.ProcessOne(batch);
}
template <typename DT_A, bool USE_CUBE, uint32_t PADDED_N, bool ONE_ROUND,uint32_t STATIC_BATCH>
__global__ __aicore__ void linalg_det(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgDetTilingPayload);
    GM_ADDR ws=nullptr;
    if constexpr(!(PADDED_N>=67&&PADDED_N<=80))ws=GetUserWorkspace(workspace);
    if constexpr(USE_CUBE&&STATIC_BATCH==5){
        // Host admits this image only for four256 matrices and four groups.
        if ASCEND_IS_AIC {DetK64::RunCubeFixed4(ws);}
        if ASCEND_IS_AIV {
            const uint32_t batch=GetBlockIdx()/8;
            const bool safe=DetK64DirectDiag::RunVectorFixed4(A,det,ws,batch);
            if(!safe)DetRunK64Fixed4Fallback<DT_A>(A,det,ws,tiling);
        }
    }else if constexpr(USE_CUBE&&STATIC_BATCH==6){
        const auto t=ReadDetConfig(tiling);
        const uint32_t waves=t.batch/4;
        for(uint32_t wave=0;wave<waves;++wave){
            GM_ADDR waveA=A+uint64_t(wave)*4*256*256*4;
            GM_ADDR waveOut=det+uint64_t(wave)*4*4;
            if ASCEND_IS_AIC {DetK64::RunCubeFixed4(ws);}
            if ASCEND_IS_AIV {
                const uint32_t batch=GetBlockIdx()/8;
                const bool safe=DetK64DirectDiag::RunVectorFixed4(waveA,waveOut,ws,batch);
                if(!safe)DetRunK64WaveFallback<DT_A>(waveA,waveOut,ws,tiling);
            }
            // SDK mixed-core barrier drains AIV and AIC before workspace reuse.
            // The old single-wave entry above remains exactly unchanged.
            if(wave+1<waves)SyncAll<false>();
        }
    }else if constexpr(USE_CUBE&&STATIC_BATCH==3){
        if ASCEND_IS_AIC {GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData,t,tiling);DetK64::RunCube(t,ws);}
        if ASCEND_IS_AIV {
            const auto t=ReadDetConfig(tiling);
            for(uint32_t round=0;round<(t.batch+t.groups-1)/t.groups;++round){
                const uint32_t batch=round*t.groups+GetBlockIdx()/(2*t.coresPerMatrix);
                const bool safe=DetK64::RunVector(A,det,ws,t,batch);
                if(!safe){
                    auto f=t;f.batch=t.batch-round*t.groups;if(f.batch>t.groups)f.batch=t.groups;
                    f.coresPerMatrix=2*t.coresPerMatrix;f.columnsPerCore=32;f.sharedStride=34*t.paddedN+128;
                    DetRunFallback<DT_A>(A+uint64_t(round)*t.groups*t.n*t.n*4,det+round*t.groups*4,ws+t.groups*t.robustOffset*4,f);
                }
            }
        }
    }else if constexpr(USE_CUBE&&STATIC_BATCH==2){
        // Dedicated n200 image keeps unrelated n256 implementations out of
        // this entry while retaining the exact paired algorithm and fallback.
        if ASCEND_IS_AIC {GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData,t,tiling);RunLUCubePrefetch200(t,ws);}
        if ASCEND_IS_AIV {
            const auto t=ReadDetConfig(tiling);
            for(uint32_t round=0;round<(t.batch+t.groups-1)/t.groups;++round){
                const uint32_t batch=round*t.groups+GetBlockIdx()/(2*t.coresPerMatrix);
                bool safe;
                if((GetBlockIdx()&1)==0)safe=DetRunStaticCooperativeCube200(A,det,ws,t,batch);
                else {TPipe pipe;CubeLUVectorHelpers op(pipe);op.Init(A,det,ws,t);safe=op.ProcessHelper200(batch);}
                if(!safe){
                    auto f=t;f.batch=t.batch-round*t.groups;if(f.batch>t.groups)f.batch=t.groups;
                    f.coresPerMatrix=2*t.coresPerMatrix;f.columnsPerCore=32;f.sharedStride=34*t.paddedN+128;
                    DetRunFallback<DT_A>(A+uint64_t(round)*t.groups*t.n*t.n*4,det+round*t.groups*4,ws+t.groups*t.robustOffset*4,f);
                }
            }
        }
    }else if constexpr(USE_CUBE&&STATIC_BATCH==1){
        if ASCEND_IS_AIC {GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData,t,tiling);RunLUCubePrefetch256(t,ws);}
        if ASCEND_IS_AIV {
            const auto t=ReadDetConfig(tiling);
            for(uint32_t round=0;round<(t.batch+t.groups-1)/t.groups;++round){
                const uint32_t batch=round*t.groups+GetBlockIdx()/(2*t.coresPerMatrix);
                bool safe;
                {TPipe pipe;CubeLUVectorHelpers op(pipe);op.Init(A,det,ws,t);
                 if((GetBlockIdx()&1)==0)safe=op.ProcessOne(batch);else safe=op.ProcessHelper(batch);}
                if(!safe){
                    auto f=t;f.batch=t.batch-round*t.groups;if(f.batch>t.groups)f.batch=t.groups;
                    f.coresPerMatrix=2*t.coresPerMatrix;f.columnsPerCore=32;f.sharedStride=34*t.paddedN+128;
                    DetRunFallback<DT_A>(A+uint64_t(round)*t.groups*t.n*t.n*4,det+round*t.groups*4,ws+t.groups*t.robustOffset*4,f);
                }
            }
        }
    }else if constexpr (USE_CUBE) {
        if ASCEND_IS_AIC {
            GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, t, tiling);
            if(t.n>256)DetWide::RunLUCube(t,ws);else RunLUCube(t,ws);
        }
        if ASCEND_IS_AIV {
            const auto t=ReadDetConfig(tiling);
            for(uint32_t round=0;round<(t.batch+t.groups-1)/t.groups;++round) {
                const uint32_t batch=round*t.groups+GetBlockIdx()/t.coresPerMatrix;
                bool safe;
                if(t.n>256){TPipe fastPipe;DetWide::CubeLUVector fast(fastPipe);fast.Init(A,det,ws,t);safe=fast.ProcessOne(batch);}
                else{ if(t.n==200&&t.coresPerMatrix==4) safe=DetRunStaticCube200(A,det,ws,t,batch); else if(t.n-32<t.coresPerMatrix*48) safe=DetRunSequentialCube(A,det,ws,t,batch); else safe=DetRunFastCube(A,det,ws,t,batch); }
                if(!safe) {
                    auto fallback=t; fallback.batch=t.batch-round*t.groups; if(fallback.batch>t.groups)fallback.batch=t.groups;
                    fallback.columnsPerCore=t.n>256?32:64;fallback.sharedStride=(fallback.columnsPerCore+2)*t.paddedN+128;
                    DetRunFallback<DT_A>(A+uint64_t(round)*t.groups*t.n*t.n*4,det+round*t.groups*4,ws+t.groups*t.robustOffset*4,fallback);
                }
            }
        }
    } else {
        if constexpr(PADDED_N>0){
            auto fields=reinterpret_cast<__gm__ const uint32_t*>(tiling);
            constexpr uint32_t staticN=PADDED_N-64;
            constexpr bool exactSmall=staticN<=16;
            if(exactSmall ? GetBlockIdx()==0 : GetBlockIdx()<(STATIC_BATCH?STATIC_BATCH:fields[4])){
            constexpr uint32_t aligned=(staticN+7)/8*8;
            constexpr bool fixed=staticN%8==0;
            const uint32_t n=staticN;
            bool safe=true;
            GM_ADDR indices=tiling+__builtin_offsetof(LinalgDetTilingPayload,gatherIndices);
            if constexpr(ONE_ROUND){
                safe=SmallDet<aligned,fixed,staticN,true>(A,det,n,GetBlockIdx(),indices);
            }else{
                const uint32_t total=fields[1],groups=fields[4];
                for(uint32_t round=0;round<(total-1)/groups+1;++round){
                    const uint64_t batch=uint64_t(round)*groups+GetBlockIdx();
                    if(batch<total)safe=SmallDet<aligned,fixed,staticN>(A,det,n,uint32_t(batch),indices)&&safe;
                }
            }
            if(!safe){const auto t=ReadDetConfig(tiling);DetRunFallback<DT_A>(A,det,ws,t);}
            }
        }else{
            const auto t=ReadDetConfig(tiling);
            TPipe mainPipe;KernelLinalgDet<DT_A> op(mainPipe);op.Init(A,det,ws,t);op.Process();
        }
    }
}
