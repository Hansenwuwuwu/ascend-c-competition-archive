#pragma once
#include "candidate_index_ramp.h"
#include "candidate_native_five_pack.h"
#include "candidate_native_prior.h"
#include "candidate_denominator_geometry.h"
#include "candidate_native_half_input.h"
#include "candidate_transfer.h"

template <typename T, uint32_t Tile, bool CounterPrior = false, bool FastRamp = false, bool RowScratch = true, bool DoublingIndex = false, bool PatternIndex = false, bool StaticEvents = false, uint32_t BatchGroup = 2, bool NativeSix = false, bool NativePack = false, bool Preload = false, bool NativePrior = false, bool FuseConfidence = false, bool HalfNativeInput = false, bool AlignedTransfers = false, bool SkewStride = false, bool ReplicatedStride = false>
class PairedBatchCandidateDecode {
    static constexpr uint32_t HalfTile = Tile / BatchGroup;
    static constexpr uint32_t Pitch = Tile + (HalfNativeInput ? 16 : 8);
    static constexpr bool Reuse = sizeof(T) == 4;
    static constexpr uint32_t RawGapBytes = Reuse && NativePrior && NativePack && Tile >= 1408 ? 32 : 0;
    static_assert(RawGapBytes == 0 || BatchGroup == 2);
    static constexpr uint32_t RawBytes = Tile * 6 * sizeof(T) + RawGapBytes;
    static constexpr uint32_t FloatAddr = RawBytes;
    static constexpr uint32_t SharedAddr = FloatAddr + (Reuse ? 0 : Tile * 24);
    static constexpr uint32_t SharedBytes = Tile * 5 * sizeof(T);
    static constexpr uint32_t ColumnAddr = SharedAddr + SharedBytes;
    static constexpr uint32_t OneAddr = ColumnAddr + Pitch * 44;
    static constexpr uint32_t StrideAddr = OneAddr + (sizeof(T) == 4 ? 928 : 576);
    static constexpr bool CachedAddresses = (Reuse || HalfNativeInput) && NativePrior && NativePack && Tile >= 1408;
    static constexpr bool TransformPriors = false;
    static constexpr uint32_t AddressCache = StrideAddr + Pitch * 16;
    static_assert(!CachedAddresses || AddressCache + 704 <= 196608);
    static_assert(!ReplicatedStride || (NativePrior && NativePack && StrideAddr + Pitch * 16 <= 196608));
    static constexpr uint32_t RawIdxAddr = OneAddr + (FuseConfidence ? 512 : 256);
    static constexpr uint32_t RawIndexRows = NativeSix && Tile < 640 ? 640 : Tile;
    static constexpr uint32_t PriorIdxAddr = RawIdxAddr + RawIndexRows * 4;
    static constexpr uint32_t PriorRows = NativeSix ? Tile / 2 : (FastRamp ? ((HalfTile + 63) / 64) * 64 : HalfTile);
    static constexpr uint32_t PackIdxAddr = PriorIdxAddr + PriorRows * 4;
    static constexpr uint32_t RampCacheAddr = RowScratch ? ColumnAddr + Tile * 20 : 0;
    static_assert(!HalfNativeInput || (sizeof(T) == 2 && NativePack && NativePrior && Tile % 128 == 0));
public:
    __aicore__ inline void Init(GM_ADDR raw, GM_ADDR prior, GM_ADDR out,
                               const FireCandidateDecodeFusionTilingData &d, TPipe *pipe,
                               uint32_t spanBegin = 0, uint32_t spanEnd = 0)
    {
        pipe_ = pipe;
        batch_ = d.batch; n_ = d.candidates; cores_ = d.cores;
        spanBegin_ = spanBegin; spanEnd_ = spanEnd;
        raw_.SetGlobalBuffer((__gm__ T *)raw);
        prior_.SetGlobalBuffer((__gm__ float *)prior);
        out_.SetGlobalBuffer((__gm__ T *)out);
        if constexpr (StaticEvents) {
            rawReady_ = 0; priorReady_ = 1; rawFree_ = 0;
            outReady_ = 0; outFree_ = 0; poolFree_ = 0;
        } else {
        rawReady_ = pipe->AllocEventID<HardEvent::MTE2_V>();
        priorReady_ = pipe->AllocEventID<HardEvent::MTE2_V>();
        rawFree_ = pipe->AllocEventID<HardEvent::V_MTE2>();
        outReady_ = pipe->FetchEventID(HardEvent::V_MTE3);
        outFree_ = pipe->FetchEventID(HardEvent::MTE3_V);
        poolFree_ = pipe->FetchEventID(HardEvent::MTE3_MTE2);
        }
        if constexpr (Preload) {
            uint32_t begin = spanBegin_, end = spanEnd_;
            if constexpr (!(NativePrior && BatchGroup == 4)) {
            const uint32_t groups = (n_ + 15) / 16;
            begin = (groups * GetBlockIdx() / cores_) * 16;
            end = (groups * (GetBlockIdx() + 1) / cores_) * 16;
            if (end > n_) end = n_;
            }
            spanBegin_ = begin;
            spanEnd_ = end;
            if (begin < end) PrefetchTile(begin, end - begin < HalfTile ? end - begin : HalfTile);
        }
        auto ri = At<int32_t>(RawIdxAddr, RawIndexRows);
        auto pi = At<int32_t>(PriorIdxAddr, NativeSix ? Tile / 2 : HalfTile);
        if constexpr (NativeSix) {
            static_assert(BatchGroup == 2 || BatchGroup == 4);
            if constexpr (NativePack) {
                if constexpr (!NativePrior)
                BuildNativeEightCandidateIndices<Tile, BatchGroup, StaticEvents>(
                    At<int32_t>(ColumnAddr, Pitch * 11),
                    At<int32_t>(RawIdxAddr, RawIndexRows), pi, pipe);
            } else {
            BuildNativeSixCandidateIndices<Tile, Pitch, BatchGroup, StaticEvents>(
                At<int32_t>(ColumnAddr, Pitch * 11),
                At<int32_t>(RawIdxAddr, RawIndexRows), pi,
                At<int32_t>(PackIdxAddr, Tile * 5), pipe);
            }
        } else {
        if constexpr (PatternIndex) {
            auto sequence = At<int32_t>(ColumnAddr, Pitch * 11);
            constexpr uint32_t RampElements = Tile < 320 ? 320 : Tile;
            BuildDoublingCandidateIndex<RampElements, StaticEvents>(sequence, sequence[RampElements], pipe);
            DataCopy(ri, sequence, Tile);
            DataCopy(pi, sequence, HalfTile);
        } else if constexpr (DoublingIndex) {
            auto sequence = At<int32_t>(ColumnAddr, Pitch * 11);
            BuildDoublingCandidateIndex<Tile * 5, StaticEvents>(sequence, sequence[Tile * 5], pipe);
            DataCopy(ri, sequence, Tile);
            DataCopy(pi, sequence, HalfTile);
        } else if constexpr (FastRamp) {
            BuildRampCache();
            FillRamp(ri, Tile);
            FillRamp(At<int32_t>(PriorIdxAddr, PriorRows), PriorRows);
        } else {
        CreateVecIndex(ri, int32_t(0), Tile);
        CreateVecIndex(pi, int32_t(0), HalfTile);
        }
        Muls(ri, ri, int32_t(24), Tile);
        Muls(pi, pi, int32_t(20), HalfTile);
        auto seq = At<int32_t>(ColumnAddr, Pitch * 11);
        auto row = seq[Tile * 5];
        auto index = At<int32_t>(PackIdxAddr, Tile * 5);
        if constexpr (DoublingIndex || PatternIndex) {}
        else if constexpr (FastRamp) FillRamp(seq, Tile * 5);
        else CreateVecIndex(seq, int32_t(0), Tile * 5);
        constexpr uint32_t IndexSeed = PatternIndex ? 320 : Tile * 5;
        Muls(row, seq, int32_t(13108), IndexSeed);
        ShiftRight(row, row, int32_t(16), IndexSeed);
        Muls(seq, seq, int32_t(4 * Pitch), IndexSeed);
        Muls(row, row, int32_t(4 - 20 * Pitch), IndexSeed);
        Add(index, seq, row, IndexSeed);
        if constexpr (PatternIndex) {
            ExpandCandidatePatternIndex<Tile * 5>(index);
        }
        }
        Duplicate(At<float>(OneAddr, 64), Tile >= 1024 ? 2.0f : 1.0f, 64);
        if constexpr (FuseConfidence) Duplicate(At<float>(OneAddr + 256, 64), 1.0f, 64);
        if constexpr (CachedAddresses)
            BuildCandidateAddressCache<Tile, Pitch, ColumnAddr, Reuse ? SharedAddr : FloatAddr,
                AddressCache, HalfNativeInput, FloatAddr, RawGapBytes>();
        if constexpr (FastRamp && !RowScratch) {
            SetFlag<HardEvent::V_MTE2>(rawFree_);
            WaitFlag<HardEvent::V_MTE2>(rawFree_);
        }
    }
    __aicore__ inline void Process()
    {
        uint32_t begin = spanBegin_, end = spanEnd_;
        if constexpr (!Preload && !(NativePrior && BatchGroup == 4)) {
            const uint32_t groups = (n_ + 15) / 16;
            begin = (groups * GetBlockIdx() / cores_) * 16;
            end = (groups * (GetBlockIdx() + 1) / cores_) * 16;
            if (end > n_) end = n_;
        }
        for (uint32_t start = begin; start < end; start += HalfTile) {
            const uint32_t count = start + HalfTile <= end ? HalfTile : end - start;
            if (poolPending_) {
                WaitFlag<HardEvent::MTE3_MTE2>(poolFree_);
                poolPending_ = false;
            }
            auto prior = At<float>(SharedAddr, HalfTile * 5, TPosition::VECIN);
            if constexpr (Preload) {
                if (start != begin) PrefetchTile(start, count);
            } else {
            CopyCandidateInput<float, AlignedTransfers>(prior, prior_, start * 5,
                1, count * 5, count * 5, count * 5);
            SetFlag<HardEvent::MTE2_V>(priorReady_);
            }
            WaitFlag<HardEvent::MTE2_V>(priorReady_);
            if constexpr (TransformPriors)
                ClearCandidatePriorTail<HalfTile>(prior, count);
            auto c = At<float>(ColumnAddr, Pitch * 11);
            const uint32_t priorCount = NativeSix ? Tile / 2 : count;
            if constexpr (NativePrior) {
                static_assert((BatchGroup == 2 || BatchGroup == 4) && NativePack);
                GatherNativePriorCandidates<Tile, Pitch, SharedAddr, ColumnAddr, BatchGroup, HalfNativeInput>();
            } else {
            for (uint32_t j = 0; j < 5; ++j) {
                if constexpr (CounterPrior)
                    Gather(c[(j + 6) * Pitch], prior, At<uint32_t>(PriorIdxAddr, NativeSix ? Tile / 2 : HalfTile),
                        j * 4, priorCount);
                else
                    Gather(c[(j + 6) * Pitch], prior, At<uint32_t>(PriorIdxAddr, NativeSix ? Tile / 2 : HalfTile),
                        j * 4, uint64_t(64), uint8_t((priorCount + 63) / 64), 8);
            }
            }
            if constexpr (NativePrior) {
                if constexpr (BatchGroup == 4) {
                    if constexpr (HalfNativeInput) {
                        for (uint32_t field = 0; field < 5; ++field)
                            DataCopy(c[(field + 6) * Pitch + Tile / 16], c[(field + 6) * Pitch],
                                DataCopyParams{8, uint16_t(Tile / 128),
                                    uint16_t(Tile / 128), uint16_t(Tile / 128)});
                    } else {
                        DataCopy(c[6 * Pitch + Tile / 2], c[6 * Pitch],
                            DataCopyParams{5, uint16_t(Tile / 16),
                                uint16_t((Pitch - Tile / 2) / 8), uint16_t((Pitch - Tile / 2) / 8)});
                    }
                }
            } else if constexpr (NativeSix) {
                for (uint32_t j = 0; j < 5; ++j)
                    DataCopy(c[(j + 6) * Pitch + Tile / 2], c[(j + 6) * Pitch], Tile / 2);
            } else {
            #pragma unroll
            for (uint32_t q = 1; q < BatchGroup; ++q)
                for (uint32_t j = 0; j < 5; ++j)
                    DataCopy(c[(j + 6) * Pitch + q * HalfTile], c[(j + 6) * Pitch], HalfTile);
            }
            if constexpr (TransformPriors)
                transformedPriors_ = CanTransformCandidatePriors<Tile, Pitch, HalfTile>(prior, c,
                    At<float>(OneAddr + 512, 16), At<float>(ColumnAddr, Pitch * 6));
            if constexpr (ReplicatedStride) {
                #pragma unroll
                for (uint32_t field = 0; field < 4; ++field) {
                    if constexpr (TransformPriors) {
                        if (field < 2 || !transformedPriors_)
                            DataCopy(At<float>(StrideAddr + field * Pitch * 4, Tile), c[8 * Pitch], Tile);
                    } else {
                        DataCopy(At<float>(StrideAddr + field * Pitch * 4, Tile), c[8 * Pitch], Tile);
                    }
                }
                if constexpr (TransformPriors) {
                    if (transformedPriors_)
                        TransformCandidatePriors<Pitch>(c, At<float>(StrideAddr, Pitch * 2));
                }
            } else if constexpr (SkewStride) {
                DataCopy(At<float>(OneAddr + 544, Tile), c[8 * Pitch], Tile);
            }
            if constexpr (!Preload) LoadRaw(0, start, count);
            for (uint32_t b = 0; b < batch_; b += BatchGroup) {
                WaitFlag<HardEvent::MTE2_V>(rawReady_);
                auto raw = At<T>(0, Tile * 6 + RawGapBytes / sizeof(T), TPosition::VECIN);
                Compute(raw, prior, b, start, count);
            }
        }
        if (outPending_) WaitFlag<HardEvent::MTE3_V>(outFree_);
        if (poolPending_) WaitFlag<HardEvent::MTE3_MTE2>(poolFree_);
        if (rawUsed_) WaitFlag<HardEvent::V_MTE2>(rawFree_);
        if constexpr (!StaticEvents) {
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(rawReady_);
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(priorReady_);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(rawFree_);
        }
    }
private:
    __aicore__ inline void PrefetchTile(uint32_t start, uint32_t count)
    {
        CopyCandidateInput<float, AlignedTransfers>(
            At<float>(SharedAddr, HalfTile * 5, TPosition::VECIN), prior_, start * 5,
            1, count * 5, count * 5, count * 5);
        SetFlag<HardEvent::MTE2_V>(priorReady_);
        LoadRaw(0, start, count);
    }
    __aicore__ inline void BuildRampCache()
    {
        auto seed = At<int32_t>(OneAddr, 8);
        for (uint32_t i = 0; i < 8; ++i) seed.SetValue(i, int32_t(i));
        const auto ready = pipe_->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(ready);
        WaitFlag<HardEvent::S_V>(ready);
        auto lanes = At<int32_t>(RampCacheAddr, 64);
        Brcb(lanes, seed, uint8_t(1), BrcbRepeatParams{1, 8});
        Muls(lanes, lanes, int32_t(8), 64);
        Add(lanes, lanes, seed, uint64_t(64), uint8_t(1), {1, 1, 0, 8, 8, 0});
        auto steps = At<int32_t>(RampCacheAddr + 256, 192);
        DataCopy(steps, lanes, 64);
        Adds(steps[64], lanes, int32_t(64), uint64_t(64), uint8_t(1), {1, 1, 8, 8});
        Adds(steps[128], lanes, int32_t(128), uint64_t(64), uint8_t(1), {1, 1, 8, 8});
        Muls(steps, steps, int32_t(64), 192);
    }
    __aicore__ inline void FillRamp(LocalTensor<int32_t> target, uint32_t elements)
    {
        const uint32_t repeats = (elements + 63) / 64;
        auto spread = At<int32_t>(RampCacheAddr + 1024, ((repeats + 7) / 8) * 64);
        Brcb(spread, At<int32_t>(RampCacheAddr + 256, 192), uint8_t((repeats + 7) / 8),
            BrcbRepeatParams{1, 8});
        Brcb(target, spread, uint8_t(repeats), BrcbRepeatParams{1, 8});
        Add(target, target, At<int32_t>(RampCacheAddr, 64), uint64_t(64), uint8_t(repeats),
            {1, 1, 1, 8, 8, 0});
    }
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t addr, uint32_t count, TPosition pos = TPosition::VECCALC)
    {
        return LocalTensor<U>(pos, addr, count);
    }
    __aicore__ inline void LoadRaw(uint32_t b, uint32_t start, uint32_t count)
    {
        if (rawUsed_) WaitFlag<HardEvent::V_MTE2>(rawFree_);
        auto raw = At<T>(0, Tile * 6 + RawGapBytes / sizeof(T), TPosition::VECIN);
        CopyCandidateInput<T, AlignedTransfers>(raw, raw_, (b * n_ + start) * 6,
            uint16_t(BatchGroup), count * 6, HalfTile * 6 + RawGapBytes / sizeof(T), n_ * 6);
        SetFlag<HardEvent::MTE2_V>(rawReady_);
        rawUsed_ = true;
    }
    __aicore__ inline void ReleaseRawAndPrefetch(uint32_t b, uint32_t start, uint32_t count)
    {
        SetFlag<HardEvent::V_MTE2>(rawFree_);
        if (b + BatchGroup < batch_) LoadRaw(b + BatchGroup, start, count);
    }
    __aicore__ inline void Compute(LocalTensor<T> raw, LocalTensor<float> prior,
                                  uint32_t b, uint32_t start, uint32_t validCount)
    {
            constexpr uint32_t count = Tile;
            LocalTensor<float> sig;
            if constexpr (HalfNativeInput) {
                if constexpr (CachedAddresses)
                    TransposeCachedCandidates<Tile, AddressCache, false, true>();
                else GatherNativeHalfCandidates<Tile, Pitch, 0, FloatAddr>();
                ReleaseRawAndPrefetch(b, start, validCount);
                sig = At<float>(ColumnAddr, Pitch * 6);
                Cast(sig, At<half>(FloatAddr, Pitch * 6), RoundMode::CAST_NONE, Pitch * 6);
                ActivateNativeHalfColumns<Pitch * 6, FuseConfidence>(sig, At<float>(OneAddr, 64));
            } else {
            if constexpr (Reuse) {
                sig = raw;
            } else {
                sig = At<float>(FloatAddr, Tile * 6);
                if constexpr (sizeof(T) == 2)
                    Cast(sig, raw, RoundMode::CAST_NONE, count * 6);
                else
                    DataCopy(sig, raw, ((count * 6 + 7) / 8) * 8);
            }
            if constexpr (!Reuse) ReleaseRawAndPrefetch(b, start, validCount);
            if constexpr (RawGapBytes != 0) {
                Duplicate(At<float>(HalfTile * 24, 8), 0.0f, 8);
                SetMaskCount();
                SetVectorMask<float, MaskMode::COUNTER>(count * 6 + RawGapBytes / 4);
                Muls<float, false>(sig, sig, -1.0f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                Exp<float, false>(sig, sig, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                Adds<float, false>(sig, sig, 1.0f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                if constexpr (!FuseConfidence)
                    Div<float, false>(sig, At<float>(OneAddr, 64), sig,
                        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 0, 8});
                SetMaskNorm();
                ResetMask();
            } else {
            Muls(sig, sig, -1.0f, uint64_t(64), uint8_t((count * 6 + 63) / 64), {1, 1, 8, 8});
            Exp(sig, sig, uint64_t(64), uint8_t((count * 6 + 63) / 64), {1, 1, 8, 8});
            Adds(sig, sig, 1.0f, uint64_t(64), uint8_t((count * 6 + 63) / 64), {1, 1, 8, 8});
            if constexpr (!FuseConfidence) {
                Div(sig, At<float>(OneAddr, 64), sig, uint64_t(64),
                    uint8_t((count * 6 + 63) / 64), {1, 1, 1, 8, 0, 8});
            }
            }
            }
            auto c = At<float>(ColumnAddr, Pitch * 11);
            const auto ri = At<uint32_t>(RawIdxAddr, Tile);

            if constexpr (HalfNativeInput) {}
            else if constexpr (CachedAddresses)
                TransposeCachedCandidates<Tile, AddressCache, false>();
            else if constexpr (NativePack)
                GatherNativeEightCandidates<Tile, Pitch, Reuse ? 0 : FloatAddr, ColumnAddr>();
            else if constexpr (NativeSix)
                GatherNativeSixCandidates<Tile, Pitch, Reuse ? 0 : FloatAddr, ColumnAddr>();
            else {
            for (uint32_t j = 0; j < 6; ++j)
                Gather(c[j * Pitch], sig, ri, j * 4, uint64_t(64), uint8_t((count + 63) / 64), 8);
            }
            if constexpr (Reuse) ReleaseRawAndPrefetch(b, start, validCount);
            if constexpr (FuseConfidence) {
                if constexpr (TransformPriors) {
                    if (transformedPriors_)
                        DecodeCandidateDenominators<Tile, Pitch, false, true, true>(c,
                            At<float>(OneAddr, 64), At<float>(OneAddr + 256, 64),
                            At<float>(StrideAddr, Pitch * 4));
                    else
                        DecodeCandidateDenominators<Tile, Pitch, false, true>(c,
                            At<float>(OneAddr, 64), At<float>(OneAddr + 256, 64),
                            At<float>(StrideAddr, Pitch * 4));
                } else if constexpr (ReplicatedStride)
                    DecodeCandidateDenominators<Tile, Pitch, false, true>(c,
                        At<float>(OneAddr, 64), At<float>(OneAddr + 256, 64),
                        At<float>(StrideAddr, Pitch * 4));
                else if constexpr (SkewStride)
                    DecodeCandidateDenominators<Tile, Pitch>(c,
                        At<float>(OneAddr, 64), At<float>(OneAddr + 256, 64),
                        At<float>(OneAddr + 544, Tile));
                else DecodeCandidateDenominators<Tile, Pitch>(c,
                    At<float>(OneAddr, 64), At<float>(OneAddr + 256, 64));
            } else {
            if constexpr (Tile < 1024) {
                Muls(c, c, 2.0f, Pitch * 4);
            }
            Adds(c, c, -0.5f, Pitch * 2);
            Mul(c[2 * Pitch], c[2 * Pitch], c[2 * Pitch], Pitch * 2);
            Mul(c[4 * Pitch], c[4 * Pitch], c[5 * Pitch], count);
            Add(c, c, c[6 * Pitch], Pitch * 2);
            Mul(c[2 * Pitch], c[2 * Pitch], c[9 * Pitch], Pitch * 2);
            if constexpr (Tile >= 1024)
                Muls(c[4 * Pitch], c[4 * Pitch], 0.25f, count);
            for (uint32_t j = 0; j < 4; ++j)
                Mul(c[j * Pitch], c[j * Pitch], c[8 * Pitch], count);
            }
            if (outPending_) WaitFlag<HardEvent::MTE3_V>(outFree_);
            auto dst = At<T>(SharedAddr, Tile * 5, TPosition::VECOUT);
            LocalTensor<float> packed;
            if constexpr (Reuse) packed = dst;
            else packed = At<float>(FloatAddr, Tile * 6);
            if constexpr (CachedAddresses)
                TransposeCachedCandidates<Tile, AddressCache, true, HalfNativeInput>();
            else if constexpr (NativePack)
                PackNativeFiveCandidates<Tile, Pitch, ColumnAddr, Reuse ? SharedAddr : FloatAddr, HalfNativeInput>();
            else {
            Gather(packed, c, At<uint32_t>(PackIdxAddr, Tile * 5), 0, uint64_t(64),
                uint8_t((count * 5 + 63) / 64), 8);
            }
            if constexpr (sizeof(T) == 2)
                Cast(dst, packed, RoundMode::CAST_RINT, count * 5);
            else if constexpr (!Reuse)
                DataCopy(dst, packed, ((count * 5 + 7) / 8) * 8);
            SetFlag<HardEvent::V_MTE3>(outReady_);
            WaitFlag<HardEvent::V_MTE3>(outReady_);
            CopyCandidateOutput<T, AlignedTransfers>(out_, dst, (b * n_ + start) * 5,
                uint16_t(BatchGroup), validCount * 5, HalfTile * 5, n_ * 5);
            SetFlag<HardEvent::MTE3_V>(outFree_);
            outPending_ = true;
            if (b + BatchGroup >= batch_) {
                SetFlag<HardEvent::MTE3_MTE2>(poolFree_);
                poolPending_ = true;
            }
    }

    TPipe *pipe_;
    GlobalTensor<T> raw_, out_;
    GlobalTensor<float> prior_;
    uint32_t batch_, n_, cores_;
    uint32_t spanBegin_, spanEnd_;
    int32_t rawReady_, rawFree_, priorReady_, outReady_, outFree_, poolFree_;
    bool rawUsed_ = false, outPending_ = false, poolPending_ = false, transformedPriors_ = false;
};

