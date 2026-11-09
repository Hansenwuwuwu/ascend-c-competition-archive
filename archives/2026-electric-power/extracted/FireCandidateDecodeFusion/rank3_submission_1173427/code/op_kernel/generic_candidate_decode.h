#pragma once

template <typename T, uint32_t Tile, bool Balanced = false, bool CachePrior = false>
class KernelFireCandidateDecodeFusion {
    static constexpr uint32_t Pitch = Tile + 8;
    static constexpr bool Reuse = (Tile == 1024 || CachePrior) && sizeof(T) == 4;
public:
    __aicore__ inline void Init(GM_ADDR raw, GM_ADDR prior, GM_ADDR out,
                               const FireCandidateDecodeFusionTilingData &data)
    {
        batch_ = data.batch;
        n_ = data.candidates;
        cores_ = data.cores;
        segments_ = data.segments;
        raw_.SetGlobalBuffer((__gm__ T *)raw);
        prior_.SetGlobalBuffer((__gm__ float *)prior);
        out_.SetGlobalBuffer((__gm__ T *)out);
        pipe_.InitBuffer(inRaw_, (Tile == 1024 || CachePrior) ? 2 : 1, Tile * 6 * sizeof(T));
        pipe_.InitBuffer(inPrior_, (Tile == 1024 && !CachePrior) ? 2 : 1, Tile * 5 * sizeof(float));
        pipe_.InitBuffer(outQueue_, 1, Tile * 5 * sizeof(T));
        if constexpr (!Reuse) pipe_.InitBuffer(rawFloat_, Tile * 6 * sizeof(float));
        pipe_.InitBuffer(columns_, Pitch * 11 * sizeof(float));
        if constexpr (!Reuse) pipe_.InitBuffer(packed_, Tile * 5 * sizeof(float));
        pipe_.InitBuffer(ones_, 64 * sizeof(float));
        pipe_.InitBuffer(rawIndex_, Tile * sizeof(uint32_t));
        pipe_.InitBuffer(priorIndex_, Tile * sizeof(uint32_t));
        pipe_.InitBuffer(packIndex_, Tile * 5 * sizeof(uint32_t));
        auto r = rawIndex_.Get<int32_t>();
        auto p = priorIndex_.Get<int32_t>();
        LocalTensor<int32_t> seq, row;
        if constexpr (Reuse) {
            seq = columns_.Get<int32_t>();
            row = seq[Tile * 5];
        } else {
            seq = packed_.Get<int32_t>();
            row = rawFloat_.Get<int32_t>();
        }
        auto index = packIndex_.Get<int32_t>();
        if constexpr (Tile == 1024) {
            auto cache = Reuse ? row : row[Tile * 5];
            InitCandidateIndexRamp(cache, &pipe_);
            FillCandidateIndexRamp(r, Tile, cache);
            FillCandidateIndexRamp(p, Tile, cache);
            FillCandidateIndexRamp(seq, Tile * 5, cache);
            Muls(r, r, int32_t(24), Tile);
            Muls(p, p, int32_t(20), Tile);
        } else {
            CreateVecIndex(r, int32_t(0), Tile);
            CreateVecIndex(p, int32_t(0), Tile);
            PipeBarrier<PIPE_V>();
            Muls(r, r, int32_t(24), Tile);
            Muls(p, p, int32_t(20), Tile);
            CreateVecIndex(seq, int32_t(0), Tile * 5);
        }
        PipeBarrier<PIPE_V>();
        Muls(row, seq, int32_t(13108), Tile * 5);
        PipeBarrier<PIPE_V>();
        ShiftRight(row, row, int32_t(16), Tile * 5);
        Muls(seq, seq, int32_t(4 * Pitch), Tile * 5);
        PipeBarrier<PIPE_V>();
        Muls(row, row, int32_t(4 - 20 * Pitch), Tile * 5);
        PipeBarrier<PIPE_V>();
        Add(index, seq, row, Tile * 5);
        Duplicate(ones_.Get<float>(), 2.0f, 64);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        if constexpr (CachePrior) { ProcessCached(); return; }
        const uint32_t tiles = Balanced ? segments_ : (n_ + Tile - 1) / Tile;
        if constexpr (Tile == 1024) {
            if (GetBlockIdx() < batch_ * tiles) Load(GetBlockIdx(), tiles);
        }
        for (uint32_t task = GetBlockIdx(); task < batch_ * tiles; task += cores_) {
            const uint32_t b = task / tiles;
            const uint32_t start = Balanced ? n_ * (task % tiles) / tiles : (task % tiles) * Tile;
            const uint32_t count = Balanced ? n_ * ((task % tiles) + 1) / tiles - start : (start + Tile <= n_ ? Tile : n_ - start);
            if constexpr (Tile != 1024) Load(task, tiles);
            auto raw = inRaw_.DeQue<T>();
            auto prior = inPrior_.DeQue<float>();
            if constexpr (Tile == 1024) {
                if (task + cores_ < batch_ * tiles) Load(task + cores_, tiles);
            }
            Compute(raw, prior, b, start, count);
            inRaw_.FreeTensor(raw);
            inPrior_.FreeTensor(prior);
        }
    }
private:
    __aicore__ inline void Compute(LocalTensor<T> raw, LocalTensor<float> prior,
                                  uint32_t b, uint32_t start, uint32_t count)
    {
            LocalTensor<float> sig;
            if constexpr (Reuse) {
                sig = raw;
            } else {
                sig = rawFloat_.Get<float>();
                if constexpr (sizeof(T) == 2)
                    Cast(sig, raw, RoundMode::CAST_NONE, count * 6);
                else
                    DataCopy(sig, raw, ((count * 6 + 7) / 8) * 8);
            }
            PipeBarrier<PIPE_V>();
            Muls(sig, sig, -1.0f, uint64_t(64), uint8_t((count * 6 + 63) / 64), {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            Exp(sig, sig, uint64_t(64), uint8_t((count * 6 + 63) / 64), {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            Adds(sig, sig, 1.0f, uint64_t(64), uint8_t((count * 6 + 63) / 64), {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            Div(sig, ones_.Get<float>(), sig, uint64_t(64),
                uint8_t((count * 6 + 63) / 64), {1, 1, 1, 8, 0, 8});
            PipeBarrier<PIPE_V>();
            auto c = columns_.Get<float>();
            const auto ri = rawIndex_.Get<uint32_t>();
            const auto pi = priorIndex_.Get<uint32_t>();
            for (uint32_t j = 0; j < 6; ++j)
                Gather(c[j * Pitch], sig, ri, j * 4, uint64_t(64), uint8_t((count + 63) / 64), 8);
            if constexpr (!CachePrior) {
            for (uint32_t j = 0; j < 5; ++j)
                Gather(c[(j + 6) * Pitch], prior, pi, j * 4, uint64_t(64), uint8_t((count + 63) / 64), 8);
            }
            PipeBarrier<PIPE_V>();
            Muls(c[5 * Pitch], c[5 * Pitch], 0.25f, count);
            Adds(c, c, -0.5f, Pitch * 2);
            Mul(c[2 * Pitch], c[2 * Pitch], c[2 * Pitch], Pitch * 2);
            Mul(c[4 * Pitch], c[4 * Pitch], c[5 * Pitch], count);
            PipeBarrier<PIPE_V>();
            Add(c, c, c[6 * Pitch], Pitch * 2);
            Mul(c[2 * Pitch], c[2 * Pitch], c[9 * Pitch], Pitch * 2);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < 4; ++j)
                Mul(c[j * Pitch], c[j * Pitch], c[8 * Pitch], count);
            PipeBarrier<PIPE_V>();
            auto dst = outQueue_.AllocTensor<T>();
            LocalTensor<float> packed;
            if constexpr (Reuse) packed = dst;
            else packed = packed_.Get<float>();
            Gather(packed, c, packIndex_.Get<uint32_t>(), 0, uint64_t(64),
                uint8_t((count * 5 + 63) / 64), 8);
            PipeBarrier<PIPE_V>();
            if constexpr (sizeof(T) == 2)
                Cast(dst, packed, RoundMode::CAST_RINT, count * 5);
            else if constexpr (!Reuse)
                DataCopy(dst, packed, ((count * 5 + 7) / 8) * 8);
            outQueue_.EnQue(dst);
            dst = outQueue_.DeQue<T>();
            DataCopyPad(out_[(b * n_ + start) * 5], dst,
                        DataCopyExtParams{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0});
            outQueue_.FreeTensor(dst);
    }
    __aicore__ inline void LoadRaw(uint32_t b, uint32_t start, uint32_t count)
    {
        auto raw = inRaw_.AllocTensor<T>();
        DataCopyPad(raw, raw_[(b * n_ + start) * 6],
            DataCopyExtParams{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        inRaw_.EnQue(raw);
    }
    __aicore__ inline void ProcessCached()
    {
        const uint32_t groups = (n_ + 7) / 8;
        const uint32_t begin = (groups * GetBlockIdx() / cores_) * 8;
        uint32_t end = (groups * (GetBlockIdx() + 1) / cores_) * 8;
        if (end > n_) end = n_;
        for (uint32_t start = begin; start < end; start += Tile) {
            const uint32_t count = start + Tile <= end ? Tile : end - start;
            if (batch_ >= 2 && count < Tile && batch_ * count <= 32) {
                ProcessGroupedTail(start, count);
                break;
            }
            auto prior = inPrior_.AllocTensor<float>();
            DataCopyPad(prior, prior_[start * 5],
                DataCopyExtParams{1, count * 20, 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
            inPrior_.EnQue(prior);
            prior = inPrior_.DeQue<float>();
            auto c = columns_.Get<float>();
            for (uint32_t j = 0; j < 5; ++j)
                Gather(c[(j + 6) * Pitch], prior, priorIndex_.Get<uint32_t>(), j * 4,
                       uint64_t(64), uint8_t((count + 63) / 64), 8);
            PipeBarrier<PIPE_V>();
            inPrior_.FreeTensor(prior);
            LoadRaw(0, start, count);
            for (uint32_t b = 0; b < batch_; ++b) {
                auto raw = inRaw_.DeQue<T>();
                if (b + 1 < batch_) LoadRaw(b + 1, start, count);
                LocalTensor<float> unused;
                Compute(raw, unused, b, start, count);
                inRaw_.FreeTensor(raw);
            }
        }
    }

    __aicore__ inline void ProcessGroupedTail(uint32_t start, uint32_t count)
    {
        auto raw = inRaw_.AllocTensor<T>();
        auto prior = inPrior_.AllocTensor<float>();
        DataCopyPad(raw, raw_[start * 6],
            DataCopyExtParams{uint16_t(batch_), count * 6 * uint32_t(sizeof(T)),
                (n_ - count) * 6 * uint32_t(sizeof(T)), 0, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        DataCopyPad(prior, prior_[start * 5],
            DataCopyExtParams{1, count * 20, 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        inRaw_.EnQue(raw);
        inPrior_.EnQue(prior);
        raw = inRaw_.DeQue<T>();
        prior = inPrior_.DeQue<float>();
        auto dst = outQueue_.AllocTensor<T>();
        LocalTensor<float> sig, packed;
        if constexpr (Reuse) { sig = raw; packed = dst; }
        else { sig = rawFloat_.Get<float>(); packed = packed_.Get<float>(); }
        const uint32_t rawStride = ((count * 6 * sizeof(T) + 31) / 32) * 32 / sizeof(T);
        const uint32_t dstStride = ((count * 5 * sizeof(T) + 31) / 32) * 32 / sizeof(T);
        DecodeGroupedCandidateTail<T, Pitch, Tile>(raw, prior, sig,
            columns_.Get<float>(), ones_.Get<float>(), rawIndex_.Get<uint32_t>(),
            priorIndex_.Get<uint32_t>(), packIndex_.Get<uint32_t>(), packed, dst,
            batch_, count, rawStride, dstStride, &pipe_);
        outQueue_.EnQue(dst);
        dst = outQueue_.DeQue<T>();
        DataCopyPad(out_[start * 5], dst,
            DataCopyExtParams{uint16_t(batch_), count * 5 * uint32_t(sizeof(T)), 0,
                (n_ - count) * 5 * uint32_t(sizeof(T)), 0});
        outQueue_.FreeTensor(dst);
        inRaw_.FreeTensor(raw);
        inPrior_.FreeTensor(prior);
    }

    __aicore__ inline void Load(uint32_t task, uint32_t tiles)
    {
        const uint32_t b = task / tiles;
        const uint32_t start = Balanced ? n_ * (task % tiles) / tiles : (task % tiles) * Tile;
        const uint32_t count = Balanced ? n_ * ((task % tiles) + 1) / tiles - start :
            (start + Tile <= n_ ? Tile : n_ - start);
        auto raw = inRaw_.AllocTensor<T>();
        auto prior = inPrior_.AllocTensor<float>();
        DataCopyPad(raw, raw_[(b * n_ + start) * 6],
            DataCopyExtParams{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0},
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        DataCopyPad(prior, prior_[start * 5],
            DataCopyExtParams{1, count * 20, 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        inRaw_.EnQue(raw);
        inPrior_.EnQue(prior);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inRaw_, inPrior_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> rawFloat_, columns_, packed_, ones_;
    TBuf<TPosition::VECCALC> rawIndex_, priorIndex_, packIndex_;
    GlobalTensor<T> raw_, out_;
    GlobalTensor<float> prior_;
    uint32_t batch_, n_, cores_, segments_;
};
