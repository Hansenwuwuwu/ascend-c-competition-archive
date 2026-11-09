#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "spline_conv_tiling.h"
#include "tiling_key_spline_conv.h"
using namespace AscendC;

// CI/CO template arguments count input/output channels. Specializations are
// complete alignment/capacity families selected by metadata on the host.
// Every active route computes each output within this physical kernel call.

template <HardEvent E> __aicore__ inline void Fence() {
    SetFlag<E>(0);
    WaitFlag<E>(0);
}
// A direct output owner assigns one 32-byte-aligned UB slab and completes
// each DMA before returning. Buffer counts below are element counts.
template <class U>
__aicore__ inline LocalTensor<U> DirectUbAlloc(uint32_t &offsetBytes, uint32_t elementCount) {
    LocalTensor<U> tensor(TPosition::VECCALC, offsetBytes, elementCount);
    const uint32_t allocationBytes =
        (elementCount * sizeof(U) + kUbAlignmentBytes - 1) / kUbAlignmentBytes * kUbAlignmentBytes;
    offsetBytes += allocationBytes;
    return tensor;
}
// General degree/dimension implementation: each core owns disjoint output
// rows. Optional private edge lists are built and consumed by that same core.
template <class T, uint32_t MODE, uint32_t DIM, uint32_t DEG> class KernelSplineConv {
    LocalTensor<float> weights, acc, slot, xlocal, product, spread, rows;
    LocalTensor<int32_t> counts;
    LocalTensor<int64_t> graphControls;
    LocalTensor<T> raw;
    GlobalTensor<T> x, a, w, out;
    GlobalTensor<int64_t> edge;
    GlobalTensor<uint16_t> xb, ab;
    GlobalTensor<int64_t> ownedEdges;
    SplineConvTilingData t;
    __aicore__ inline void LoadWeight(uint32_t k, uint32_t dest) {
        Fence<HardEvent::V_MTE2>();
        if constexpr (sizeof(T) == 4) {
            DataCopyPad(weights[dest], w[uint64_t(k) * t.inputChannels * t.outputChannels],
                        {uint16_t(t.inputChannels), uint32_t(t.outputChannels * 4), 0, 0, 0},
                        {false, 0, 0, 0});
        } else {
            DataCopyPad(raw, w[uint64_t(k) * t.inputChannels * t.outputChannels],
                        {uint16_t(t.inputChannels), uint32_t(t.outputChannels * 2), 0, 0, 0},
                        {false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            Cast(weights[dest], raw, RoundMode::CAST_NONE, t.inputChannels * t.alignedOutputChannels);
        }
        Fence<HardEvent::MTE2_V>();
    }
    // Cached buffers already fit the Host UB budget, hence a contiguous
    // byte count fits DataCopyPad's21-bit blockLen. A2 blockCount has12bits:
    // 4096rows cannot be encoded as one multi-row transfer.
    __aicore__ inline void CopyCachedWeightRows(uint32_t knotCount) {
        const uint32_t rowCount = knotCount * t.inputChannels;
        if (t.alignedOutputChannels == t.outputChannels) {
            if constexpr (sizeof(T) == 4)
                DataCopyPad(weights, w, {1, uint32_t(rowCount * t.outputChannels * 4), 0, 0, 0},
                            {false, 0, 0, 0});
            else
                DataCopyPad(raw, w, {1, uint32_t(rowCount * t.outputChannels * 2), 0, 0, 0},
                            {false, 0, 0, 0});
            return;
        }
        constexpr uint32_t maxRowsPerTransfer = 4095;
        for (uint32_t begin = 0; begin < rowCount; begin += maxRowsPerTransfer) {
            const uint32_t count = rowCount - begin < maxRowsPerTransfer
                                       ? rowCount - begin : maxRowsPerTransfer;
            if constexpr (sizeof(T) == 4)
                DataCopyPad(weights[begin * t.alignedOutputChannels], w[uint64_t(begin) * t.outputChannels],
                            {uint16_t(count), uint32_t(t.outputChannels * 4), 0, 0, 0}, {false, 0, 0, 0});
            else
                DataCopyPad(raw[begin * t.alignedOutputChannels], w[uint64_t(begin) * t.outputChannels],
                            {uint16_t(count), uint32_t(t.outputChannels * 2), 0, 0, 0}, {false, 0, 0, 0});
        }
    }
    __aicore__ inline void LoadAll(uint32_t ktotal) {
        if constexpr (sizeof(T) == 4) {
            if (t.contractKnots)
                Fence<HardEvent::V_MTE2>();
            CopyCachedWeightRows(ktotal);
            Fence<HardEvent::MTE2_V>();
        } else {
            Fence<HardEvent::V_MTE2>();
            CopyCachedWeightRows(ktotal);
            Fence<HardEvent::MTE2_V>();
            Cast(weights, raw, RoundMode::CAST_NONE, ktotal * t.inputChannels * t.alignedOutputChannels);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline float Basis(float v, uint32_t j) {
        if constexpr (DEG == 1)
            return j == 0 ? 1 - v : v;
        if constexpr (DEG == 2)
            return j == 0 ? 0.5f * v * v - v + 0.5f : (j == 1 ? -v * v + v + 0.5f : 0.5f * v * v);
        float v2 = v * v, v3 = v2 * v;
        if (j == 0) {
            float z = 1 - v;
            return z * z * z * (1.0f / 6);
        }
        if (j == 1)
            return (3 * v3 - 6 * v2 + 4) * (1.0f / 6);
        if (j == 2)
            return (-3 * v3 + 3 * v2 + 3 * v + 1) * (1.0f / 6);
        return v3 * (1.0f / 6);
    }
    __aicore__ inline float Scalar(GlobalTensor<T> tensor, GlobalTensor<uint16_t> bits, uint64_t i) {
        if constexpr (sizeof(T) == 4)
            return tensor.GetValue(i);
        else {
            union {
                uint32_t u;
                float f;
            } v;
            v.u = uint32_t(bits.GetValue(i)) << 16;
            return v.f;
        }
    }
    __aicore__ inline void LoadX(uint32_t n) {
        Fence<HardEvent::V_MTE2>();
        if constexpr (sizeof(T) == 4) {
            DataCopyPad(xlocal, x[uint64_t(n) * t.inputChannels], {1, uint32_t(t.inputChannels * 4), 0, 0, 0},
                        {false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
        } else {
            DataCopyPad(raw, x[uint64_t(n) * t.inputChannels], {1, uint32_t(t.inputChannels * 2), 0, 0, 0},
                        {false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            Cast(xlocal, raw, RoundMode::CAST_NONE, t.inputChannels);
            PipeBarrier<PIPE_V>();
        }
    }
    // A full FP32 vector is 64 elements. Set both mask words explicitly;
    // only FP32 DMA or already-cached weights may preserve this mask across
    // supports. Streaming BF16 Cast changes it and uses the count fallback.
    // Both versions visit all supports in the original supportCoefficient/FMA order.
    template <bool reuseFullVectorMask>
    __aicore__ inline void AccumulateKnotSupport(const uint32_t *axisBaseIndices, const float *axisBasisValues,
                                                uint32_t matrixElementCount, uint8_t fullVectorRepeatCount) {
        if constexpr (reuseFullVectorMask) {
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(uint64_t(0), ~uint64_t(0));
        }
#pragma unroll 1
        for (uint32_t supportIndex = 0; supportIndex < ((DEG + 1) * (DEG + 1) * (DIM == 3 ? DEG + 1 : 1)); ++supportIndex) {
            uint32_t remainingSupportDigits = supportIndex, supportKnotIndex = 0, axisKnotStride = 1;
            float supportCoefficient = 1;
            for (uint32_t axis = 0; axis < DIM; ++axis) {
                uint32_t axisBasisIndex = remainingSupportDigits % (DEG + 1);
                remainingSupportDigits /= DEG + 1;
                supportKnotIndex += ((axisBaseIndices[axis] + axisBasisIndex) % t.knotSizes[axis]) * axisKnotStride;
                axisKnotStride *= t.knotSizes[axis];
                supportCoefficient *= axisBasisValues[axis * 4 + axisBasisIndex];
            }
            uint32_t weightOffsetElements = t.cacheWeights ? supportKnotIndex * matrixElementCount : 0;
            if (!t.cacheWeights)
                LoadWeight(supportKnotIndex, 0);
            if constexpr (reuseFullVectorMask)
                Axpy<float, float, false>(product, weights[weightOffsetElements], supportCoefficient,
                                          MASK_PLACEHOLDER, fullVectorRepeatCount, {1, 1, 8, 8});
            else
                Axpy(product, weights[weightOffsetElements], supportCoefficient, matrixElementCount);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Gemv(LocalTensor<float> matrix) {
        Brcb(spread, xlocal, (t.inputChannels + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < t.alignedOutputChannels; c += 64) {
            uint32_t count = t.alignedOutputChannels - c < 64 ? t.alignedOutputChannels - c : 64;
            Mul(product[c], matrix[c], spread, uint64_t(count), uint8_t(t.inputChannels),
                {1, 1, 0, uint8_t(t.alignedOutputChannels / 8), uint8_t(t.alignedOutputChannels / 8), 1});
        }
        PipeBarrier<PIPE_V>();
        uint32_t rows = t.inputChannels;
        while (rows > 1) {
            uint32_t half = (rows + 1) / 2;
            Add(product, product, product[half * t.alignedOutputChannels],
                (rows - half) * t.alignedOutputChannels);
            PipeBarrier<PIPE_V>();
            rows = half;
        }
        if constexpr (MODE == kGlobalAccumulationMode) {
            Adds(slot, product, 0.0f, t.outputChannels);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void WriteNode(uint32_t n, float count, LocalTensor<float> result) {
        if (t.applyMean && count > 1) {
            Muls(result, result, 1 / count, t.outputChannels);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (sizeof(T) == 4) {
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(out[uint64_t(n) * t.outputChannels], result,
                        {1, uint32_t(t.outputChannels * 4), 0, 0, 0});
        } else {
            Cast(raw, result, RoundMode::CAST_RINT, t.outputChannels);
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(out[uint64_t(n) * t.outputChannels], raw,
                        {1, uint32_t(t.outputChannels * 2), 0, 0, 0});
        }
        Fence<HardEvent::MTE3_V>();
    }
    __aicore__ inline void Message(uint32_t e, uint32_t ktotal, bool &ready) {
        uint32_t src = edge.GetValue(e), base[3];
        float b[12];
        if constexpr (sizeof(T) == 4) {
            Fence<HardEvent::V_MTE2>();
            if (t.cacheWeights && !ready) {
                CopyCachedWeightRows(ktotal);
                ready = true;
            }
            DataCopyPad(xlocal, x[uint64_t(src) * t.inputChannels],
                        {1, uint32_t(t.inputChannels * 4), 0, 0, 0}, {false, 0, 0, 0});
        } else {
            if (t.cacheWeights && !ready) {
                LoadAll(ktotal);
                ready = true;
            }
            LoadX(src);
        }
        for (uint32_t d = 0; d < DIM; ++d) {
            float v = Scalar(a, ab, e * DIM + d) * t.coordinateScales[d];
            base[d] = int32_t(v);
            v -= int32_t(base[d]);
            for (uint32_t j = 0; j <= DEG; ++j)
                b[d * 4 + j] = Basis(v, j);
        }
        if constexpr (sizeof(T) == 4) {
            Fence<HardEvent::MTE2_V>();
        }
        // This direct-owner helper uses slot as its support accumulator.
        // Multiply x by each real knot matrix before applying its basis value.
        // Interpolating W first can overflow at max_float even when x@W and
        // the required output are finite. The original global path uses this
        // same stable product-before-basis order.
        Duplicate(slot, 0.0f, t.outputChannels);
        PipeBarrier<PIPE_V>();
#pragma unroll 1
        for (uint32_t supportIndex = 0; supportIndex < ((DEG + 1) * (DEG + 1) * (DIM == 3 ? DEG + 1 : 1)); ++supportIndex) {
            uint32_t remainingSupportDigits = supportIndex, supportKnotIndex = 0, axisKnotStride = 1;
            float supportCoefficient = 1;
            for (uint32_t axis = 0; axis < DIM; ++axis) {
                uint32_t axisBasisIndex = remainingSupportDigits % (DEG + 1);
                remainingSupportDigits /= DEG + 1;
                supportKnotIndex += ((base[axis] + axisBasisIndex) % t.knotSizes[axis]) * axisKnotStride;
                axisKnotStride *= t.knotSizes[axis];
                supportCoefficient *= b[axis * 4 + axisBasisIndex];
            }
            uint32_t weightOffsetElements =
                t.cacheWeights ? supportKnotIndex * t.inputChannels * t.alignedOutputChannels : 0;
            if (!t.cacheWeights)
                LoadWeight(supportKnotIndex, 0);
            // MODE1 Gemv returns its reduced row in product and leaves slot
            // untouched. Cached weights are never overwritten, so ready may
            // retain them across later messages owned by this worker.
            Gemv(weights[weightOffsetElements]);
            Axpy(slot, product, supportCoefficient, t.outputChannels);
            PipeBarrier<PIPE_V>();
        }
        Adds(product, slot, 0.0f, t.outputChannels);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Direct(uint32_t ktotal) {
        bool ready = false;
        for (uint32_t n = GetBlockIdx(); n < t.nodeCount; n += t.workerCount) {
            uint32_t count = 0;
            for (uint32_t e = 0; e < t.edgeCount; ++e) {
                if (uint32_t(edge.GetValue(t.edgeCount + e)) != n)
                    continue;
                if (count == 1) {
                    Adds(acc, product, 0.0f, t.outputChannels);
                    PipeBarrier<PIPE_V>();
                }
                ++count;
                Message(e, ktotal, ready);
                if (count > 1) {
                    Add(acc, acc, product, t.outputChannels);
                    PipeBarrier<PIPE_V>();
                }
            }
            if (count == 0) {
                Duplicate(acc, 0.0f, t.outputChannels);
                PipeBarrier<PIPE_V>();
            }
            WriteNode(n, int32_t(count), count == 1 ? product : acc);
        }
    }
    __aicore__ inline uint32_t BuildOwnedEdgeList(uint32_t ownerBeginRow, uint32_t ownerEndRow) {
        constexpr uint32_t batchEntries = kOwnedDestinationBatchEntries;
        LocalTensor<int64_t> destinations = graphControls;
        LocalTensor<int64_t> stagedMatches = graphControls[batchEntries];
        auto destinationValues = (__ubuf__ int64_t *)destinations.GetPhyAddr();
        auto matchValues = (__ubuf__ int64_t *)stagedMatches.GetPhyAddr();
        uint32_t storedEntries = 0, bufferedEntries = 0;
        for (uint32_t edgeBegin = 0; edgeBegin < t.edgeCount; edgeBegin += batchEntries) {
            const uint32_t actualEntries = t.edgeCount - edgeBegin < batchEntries
                                               ? t.edgeCount - edgeBegin : batchEntries;
            Fence<HardEvent::S_MTE2>();
            DataCopyPad(destinations, edge[t.edgeCount + edgeBegin],
                        {1, uint32_t(actualEntries * sizeof(int64_t)), 0, 0, 0}, {false, 0, 0, 0});
            Fence<HardEvent::MTE2_S>();
            for (uint32_t position = 0; position < actualEntries; ++position) {
                const uint32_t destinationRow = uint32_t(destinationValues[position]);
                if (destinationRow < ownerBeginRow || destinationRow >= ownerEndRow)
                    continue;
                matchValues[bufferedEntries++] = int64_t((uint64_t(destinationRow) << 32) | (edgeBegin + position));
                if (bufferedEntries == batchEntries) {
                    Fence<HardEvent::S_MTE3>();
                    DataCopy(ownedEdges[storedEntries], stagedMatches, batchEntries);
                    Fence<HardEvent::MTE3_S>();
                    storedEntries += bufferedEntries;
                    bufferedEntries = 0;
                }
            }
        }
        if (bufferedEntries != 0) {
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(ownedEdges[storedEntries], stagedMatches,
                        {1, uint32_t(bufferedEntries * sizeof(int64_t)), 0, 0, 0});
            Fence<HardEvent::MTE3_S>();
            storedEntries += bufferedEntries;
        }
        Fence<HardEvent::MTE3_MTE2>();
        return storedEntries;
    }
    __aicore__ inline void AccumulateOwnedEdge(uint32_t edgeIndex, uint32_t destinationRow,
                                              uint32_t tileBeginRow, uint32_t knotCount, bool &weightsReady) {
        const uint32_t localRow = destinationRow - tileBeginRow;
        Message(edgeIndex, knotCount, weightsReady);
        Add(rows[localRow * t.alignedOutputChannels], rows[localRow * t.alignedOutputChannels],
            product, t.outputChannels);
        PipeBarrier<PIPE_V>();
        ((__ubuf__ int32_t *)counts.GetPhyAddr())[localRow] += 1;
    }
    __aicore__ inline void StoreOwnedRows(uint32_t tileBeginRow, uint32_t rowCount) {
        auto incomingCounts = (__ubuf__ int32_t *)counts.GetPhyAddr();
        if constexpr (sizeof(T) == 4) {
            if (t.applyMean) {
                for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
                    const uint32_t incomingCount = uint32_t(incomingCounts[localRow]);
                    if (incomingCount > 1)
                        Muls(rows[localRow * t.alignedOutputChannels], rows[localRow * t.alignedOutputChannels],
                             1.0f / incomingCount, t.outputChannels);
                }
            }
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(out[uint64_t(tileBeginRow) * t.outputChannels], rows,
                        {uint16_t(rowCount), uint32_t(t.outputChannels * sizeof(float)), 0, 0, 0});
            Fence<HardEvent::MTE3_V>();
        } else {
            for (uint32_t localRow = 0; localRow < rowCount; ++localRow)
                WriteNode(tileBeginRow + localRow, incomingCounts[localRow],
                          rows[localRow * t.alignedOutputChannels]);
        }
    }
    __aicore__ inline void RowDirect(uint32_t knotCount, GM_ADDR scratch) {
        const uint32_t rowsPerOwner = (t.nodeCount + t.workerCount - 1) / t.workerCount;
        const uint32_t ownerBeginRow = GetBlockIdx() * rowsPerOwner;
        const uint32_t ownerEndRow = ownerBeginRow + rowsPerOwner < t.nodeCount
                                          ? ownerBeginRow + rowsPerOwner : t.nodeCount;
        if (ownerBeginRow >= ownerEndRow)
            return;
        uint32_t ownedEdgeCount = 0;
        if (t.usePrivateEdgeBuckets) {
            const uint64_t entriesPerOwner =
                (uint64_t(t.edgeCount) + kOwnedEdgeWorkspaceAlignmentEntries - 1) /
                kOwnedEdgeWorkspaceAlignmentEntries * kOwnedEdgeWorkspaceAlignmentEntries;
            ownedEdges.SetGlobalBuffer((__gm__ int64_t *)scratch + GetBlockIdx() * entriesPerOwner);
            ownedEdgeCount = BuildOwnedEdgeList(ownerBeginRow, ownerEndRow);
        }
        bool weightsReady = false;
        for (uint32_t tileBeginRow = ownerBeginRow; tileBeginRow < ownerEndRow; tileBeginRow += t.rowCapacity) {
            const uint32_t rowCount = ownerEndRow - tileBeginRow < t.rowCapacity
                                           ? ownerEndRow - tileBeginRow : t.rowCapacity;
            const uint32_t tileEndRow = tileBeginRow + rowCount;
            Fence<HardEvent::S_V>();
            Duplicate(rows, 0.0f, rowCount * t.alignedOutputChannels);
            Duplicate(counts, int32_t(0), rowCount);
            Fence<HardEvent::V_S>();
            if (t.usePrivateEdgeBuckets) {
                constexpr uint32_t batchEntries = kOwnedControlBufferBytes / sizeof(int64_t);
                auto packedEntries = (__ubuf__ int64_t *)graphControls.GetPhyAddr();
                for (uint32_t edgeBegin = 0; edgeBegin < ownedEdgeCount; edgeBegin += batchEntries) {
                    const uint32_t actualEntries = ownedEdgeCount - edgeBegin < batchEntries
                                                       ? ownedEdgeCount - edgeBegin : batchEntries;
                    Fence<HardEvent::S_MTE2>();
                    DataCopyPad(graphControls, ownedEdges[edgeBegin],
                                {1, uint32_t(actualEntries * sizeof(int64_t)), 0, 0, 0}, {false, 0, 0, 0});
                    Fence<HardEvent::MTE2_S>();
                    for (uint32_t position = 0; position < actualEntries; ++position) {
                        const uint64_t entry = uint64_t(packedEntries[position]);
                        const uint32_t destinationRow = entry >> 32;
                        if (destinationRow >= tileBeginRow && destinationRow < tileEndRow)
                            AccumulateOwnedEdge(uint32_t(entry), destinationRow, tileBeginRow, knotCount, weightsReady);
                    }
                }
            } else {
                for (uint32_t edgeIndex = 0; edgeIndex < t.edgeCount; ++edgeIndex) {
                    const uint32_t destinationRow = uint32_t(edge.GetValue(t.edgeCount + edgeIndex));
                    if (destinationRow >= tileBeginRow && destinationRow < tileEndRow)
                        AccumulateOwnedEdge(edgeIndex, destinationRow, tileBeginRow, knotCount, weightsReady);
                }
            }
            StoreOwnedRows(tileBeginRow, rowCount);
        }
    }

    // Assigns each direct-owner buffer once. Counts are elements; the allocator
    // advances an aligned byte offset. Host UB checks include this same capacity.
    __aicore__ inline void AllocateDirectBuffers(uint32_t knotCount) {
        uint32_t offsetBytes = 0;
        xlocal = DirectUbAlloc<float>(offsetBytes, 128);
        weights = DirectUbAlloc<float>(offsetBytes, (t.cacheWeights ? knotCount : 1) * t.inputChannels *
                                                        t.alignedOutputChannels);
        raw = DirectUbAlloc<T>(offsetBytes, sizeof(T) == 4 ? 8
                                                           : (t.cacheWeights ? knotCount : 1) *
                                                                 t.inputChannels * t.alignedOutputChannels);
        product = DirectUbAlloc<float>(offsetBytes, (t.contractKnots ? knotCount : 1) * t.inputChannels *
                                                        t.alignedOutputChannels);
        spread = DirectUbAlloc<float>(offsetBytes, 1024);
        acc = DirectUbAlloc<float>(offsetBytes, t.scratchRowElements);
        slot = DirectUbAlloc<float>(offsetBytes, t.alignedOutputChannels);
        if (t.directRowIntervals) {
            rows = DirectUbAlloc<float>(offsetBytes, t.rowCapacity * t.alignedOutputChannels);
            uint32_t metadataCapacityRows = (t.rowCapacity + 7) / 8 * 8;
            counts = DirectUbAlloc<int32_t>(offsetBytes, metadataCapacityRows);
            graphControls = DirectUbAlloc<int64_t>(offsetBytes, kOwnedControlBufferBytes / sizeof(int64_t));
        }
    }

  public:
    __aicore__ inline void Run(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR scratch,
                               const SplineConvTilingData &data) {
        // Padded launch lanes own no rows or workspace. Return before any
        // graph transfer, UB allocation, or output initialization.
        if (GetBlockIdx() >= data.workerCount)
            return;
        t = data;
        xb.SetGlobalBuffer((__gm__ uint16_t *)xp);
        ab.SetGlobalBuffer((__gm__ uint16_t *)ap);
        x.SetGlobalBuffer((__gm__ T *)xp);
        a.SetGlobalBuffer((__gm__ T *)ap);
        w.SetGlobalBuffer((__gm__ T *)wp);
        out.SetGlobalBuffer((__gm__ T *)yp);
        edge.SetGlobalBuffer((__gm__ int64_t *)ep);
        uint32_t ktotal = 1;
        for (uint32_t d = 0; d < DIM; ++d)
            ktotal *= t.knotSizes[d];
        AllocateDirectBuffers(ktotal);
        if (t.directRowIntervals)
            RowDirect(ktotal, scratch);
        else
            Direct(ktotal);
    }

};

// Fast-family UB layout and shared interpolation/reduction primitives.
template <class U> __aicore__ inline LocalTensor<U> FastAlloc(uint32_t &offset, uint32_t count) {
    LocalTensor<U> tensor(TPosition::VECCALC, offset, count);
    uint32_t phase = ((offset & 511) + 256) & 511;
    offset = (offset + (count * sizeof(U) + 31) / 32 * 32 + 511) / 512 * 512 + phase;
    return tensor;
}
template <class U> __aicore__ inline LocalTensor<U> OperandAlloc(uint32_t &offset, uint32_t count) {
    offset += 32;
    return FastAlloc<U>(offset, count);
}
template <class U> __aicore__ inline LocalTensor<U> FastOutAlloc(uint32_t &offset, uint32_t count) {
    offset += 256;
    return FastAlloc<U>(offset, count);
}
template <class T>
__aicore__ inline float PseudoValue(GlobalTensor<T> input, GlobalTensor<uint16_t> bits, uint32_t index) {
    if constexpr (sizeof(T) == 4)
        return input.GetValue(index);
    else {
        union {
            uint32_t u;
            float f;
        } v;
        v.u = uint32_t(bits.GetValue(index)) << 16;
        return v.f;
    }
}


// Public pointer DMA API. Byte counts and 32-byte strides match DataCopy.
template<class T> __aicore__ inline void SplineCCopy(LocalTensor<T> d, GlobalTensor<T> s,uint32_t n) {
    asc_copy_gm2ub((__ubuf__ void *)d.GetPhyAddr(),(__gm__ void *)s.GetPhyAddr(),n*sizeof(T));
}
template<class T> __aicore__ inline void SplineCCopy(GlobalTensor<T> d, LocalTensor<T> s,uint32_t n) {
    asc_copy_ub2gm((__gm__ void *)d.GetPhyAddr(),(__ubuf__ void *)s.GetPhyAddr(),n*sizeof(T));
}
template<class T> __aicore__ inline void SplineCCopyPad(LocalTensor<T> d, GlobalTensor<T> s,
    DataCopyExtParams p, DataCopyPadExtParams<T> padding) {
    // All call sites in this experiment use false/zero explicit padding.
    asc_copy_gm2ub_align((__ubuf__ uint8_t *)d.GetPhyAddr(),(__gm__ uint8_t *)s.GetPhyAddr(),
                         p.blockCount,p.blockLen,0,0,p.srcStride,p.dstStride);
}
template<class T> __aicore__ inline void SplineCCopyPad(GlobalTensor<T> d, LocalTensor<T> s,DataCopyExtParams p) {
    asc_copy_ub2gm_align((__gm__ uint8_t *)d.GetPhyAddr(),(__ubuf__ uint8_t *)s.GetPhyAddr(),
                         p.blockCount,p.blockLen,0,0,p.srcStride,p.dstStride);
}

// Official pointer C API math; synchronization is explicit at every dependency.
__aicore__ inline __ubuf__ float *SplinePtr(LocalTensor<float> t) {
    return (__ubuf__ float *)t.GetPhyAddr();
}
__aicore__ inline void SplineCAdd(LocalTensor<float> d, LocalTensor<float> a,
                                 LocalTensor<float> b, uint32_t count) {
    asc_add(SplinePtr(d), SplinePtr(a), SplinePtr(b), count);
}
__aicore__ inline void SplineCAdd(LocalTensor<float> d, LocalTensor<float> a,
                                 LocalTensor<float> b, uint64_t mask, uint8_t repeat,
                                 BinaryRepeatParams p) {
    SetMaskNorm();
    asc_set_vector_mask(0, mask == 64 ? ~uint64_t(0) : (uint64_t(1) << mask) - 1);
    asc_add(SplinePtr(d), SplinePtr(a), SplinePtr(b), repeat,
            p.dstBlkStride,p.src0BlkStride,p.src1BlkStride,
            p.dstRepStride,p.src0RepStride,p.src1RepStride);
}
__aicore__ inline void SplineCMul(LocalTensor<float> d, LocalTensor<float> a,
                                 LocalTensor<float> b, uint32_t count) {
    asc_mul(SplinePtr(d), SplinePtr(a), SplinePtr(b), count);
}
__aicore__ inline void SplineCMul(LocalTensor<float> d, LocalTensor<float> a,
                                 LocalTensor<float> b, uint64_t mask, uint8_t repeat,
                                 BinaryRepeatParams p) {
    SetMaskNorm();
    asc_set_vector_mask(0, mask == 64 ? ~uint64_t(0) : (uint64_t(1) << mask) - 1);
    asc_mul(SplinePtr(d), SplinePtr(a), SplinePtr(b), repeat,
            p.dstBlkStride,p.src0BlkStride,p.src1BlkStride,
            p.dstRepStride,p.src0RepStride,p.src1RepStride);
}
__aicore__ inline void SplineCMuls(LocalTensor<float> d, LocalTensor<float> a,float v,uint32_t n) {
    asc_mul_scalar(SplinePtr(d),SplinePtr(a),v,n);
}
__aicore__ inline void SplineCAxpy(LocalTensor<float> d, LocalTensor<float> a,float v,uint32_t n) {
    asc_axpy(SplinePtr(d),SplinePtr(a),v,n);
}
__aicore__ inline void SplineCBrcb(LocalTensor<float> d, LocalTensor<float> a,uint8_t repeat,BrcbRepeatParams p) {
    asc_brcb((__ubuf__ uint32_t *)SplinePtr(d),(__ubuf__ uint32_t *)SplinePtr(a),p.dstBlkStride,p.dstRepStride,repeat);
}


template<class T> __aicore__ inline T SplineCGet(LocalTensor<T> s,uint32_t i) {
    return ((__ubuf__ T *)s.GetPhyAddr())[i];
}
template<class T> __aicore__ inline void SplineCSet(LocalTensor<T> tensor, uint32_t elementIndex, T value) {
    ((__ubuf__ T *)tensor.GetPhyAddr())[elementIndex] = value;
}
template<class T> __aicore__ inline void SplineCDuplicate(LocalTensor<T> d,T value,uint32_t n) {
    asc_duplicate((__ubuf__ T *)d.GetPhyAddr(),value,n);
}
__aicore__ inline void SplineCCast(LocalTensor<float> d,LocalTensor<bfloat16_t> s,RoundMode,uint32_t n) {
    asc_bfloat162float(SplinePtr(d),(__ubuf__ bfloat16_t *)s.GetPhyAddr(),n);
}
__aicore__ inline void SplineCCast(LocalTensor<bfloat16_t> d,LocalTensor<float> s,RoundMode,uint32_t n) {
    asc_float2bfloat16_rn((__ubuf__ bfloat16_t *)d.GetPhyAddr(),SplinePtr(s),n);
}

__aicore__ inline void SplineCAdds(LocalTensor<float> d, LocalTensor<float> s, float v, uint32_t n) {
    asc_add_scalar(SplinePtr(d), SplinePtr(s), v, n);
}
__aicore__ inline void SplineCAdds(LocalTensor<float> d, LocalTensor<float> s, float v,
                                  uint64_t mask, uint8_t repeat, UnaryRepeatParams p) {
    SetMaskNorm();
    asc_set_vector_mask(0, mask == 64 ? ~uint64_t(0) : (uint64_t(1) << mask) - 1);
    asc_add_scalar(SplinePtr(d), SplinePtr(s), v, repeat,
                   p.dstBlkStride, p.srcBlkStride, p.dstRepStride, p.srcRepStride);
}
template <class T>
__aicore__ inline void SplineCDuplicate(LocalTensor<T> d, T value, uint64_t mask,
                                       uint8_t repeat, uint16_t blockStride, uint16_t repeatStride) {
    const uint64_t lo = mask >= 64 ? ~uint64_t(0) : (uint64_t(1) << mask) - 1;
    const uint64_t hi = mask <= 64 ? 0 : mask == 128 ? ~uint64_t(0) : (uint64_t(1) << (mask - 64)) - 1;
    SetMaskNorm();
    asc_set_vector_mask(hi, lo);
    asc_duplicate((__ubuf__ T *)d.GetPhyAddr(), value, repeat, blockStride, repeatStride);
}
template <uint32_t ROWS, uint32_t CO> __aicore__ inline void FastReduce(LocalTensor<float> result) {
    if constexpr (ROWS > 1) {
        constexpr uint32_t half = (ROWS + 1) / 2, count = (ROWS - half) * CO;
        if constexpr (count <= 64)
            SplineCAdd(result, result, result[half * CO], uint64_t(count), uint8_t(1), {1, 1, 1, 8, 8, 8});
        else if constexpr (count % 64 == 0 && count / 64 <= 255)
            SplineCAdd(result, result, result[half * CO], uint64_t(64), uint8_t(count / 64), {1, 1, 1, 8, 8, 8});
        else
            SplineCAdd(result, result, result[half * CO], count);
        PipeBarrier<PIPE_V>();
        FastReduce<half, CO>(result);
    }
}
// Distribute the four exact basis coefficients into a complete CI-by-CO
// kernel, then evaluate one GEMV. The original weights stay immutable for
// repeated destinations; each message rebuilds all coefficients and elements.
// This reassociates FP32 operations and requires separate precision testing.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void ContractCanonicalBilinear(const float *coefficients,
                                               LocalTensor<float> feature, LocalTensor<float> weights,
                                               LocalTensor<float> result, LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO;
    static_assert(matrixElements % (2 * kFloatVectorElements) == 0,
                  "Both CI halves contain complete vectors.");
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    SplineCMuls(result, weights, coefficients[0], matrixElements);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(result, weights[support * matrixElements], coefficients[support], matrixElements);
        PipeBarrier<PIPE_V>();
    }
    // Lower-half products overwrite only their consumed kernel elements.
    // Upper-half elements remain intact until the paired FMA completes.
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
    asc_mul(SplinePtr(result), SplinePtr(result), SplinePtr(broadcast),
            uint8_t(matrixElements / (2 * kFloatVectorElements)), 1, 1, 1, 8, 8, 8);
    PipeBarrier<PIPE_V>();
    asc_fma(SplinePtr(result), SplinePtr(result[matrixElements / 2]),
            SplinePtr(broadcast[matrixElements / 2]),
            uint8_t(matrixElements / (2 * kFloatVectorElements)), 1, 1, 1, 8, 8, 8);
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 2, CO>(result);
}

// Advance one CI tree layer in each independent contiguous support matrix.
// All four stores are disjoint; one PIPE_V publishes the complete layer.
template <uint32_t RemainingRows, uint32_t CI, uint32_t CO>
__aicore__ inline void ReduceContiguousSupportLayer(LocalTensor<float> products) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t halfRows = RemainingRows / 2;
    constexpr uint32_t count = halfRows * CO;
    if constexpr (count <= 64 || (count % 64 == 0 && count / 64 <= 255)) {
        constexpr uint32_t maskElements = count <= 64 ? count : 64;
        SetMaskNorm();
        asc_set_vector_mask(0, maskElements == 64 ? ~uint64_t(0) : (uint64_t(1) << maskElements) - 1);
#pragma unroll
        for (uint32_t support = 0; support < 4; ++support)
            asc_add(SplinePtr(products[support * matrixElements]),
                    SplinePtr(products[support * matrixElements]),
                    SplinePtr(products[support * matrixElements + halfRows * CO]),
                    uint8_t(count <= 64 ? 1 : count / 64), 1, 1, 1, 8, 8, 8);
    } else {
#pragma unroll
        for (uint32_t support = 0; support < 4; ++support)
            SplineCAdd(products[support * matrixElements], products[support * matrixElements],
                        products[support * matrixElements + halfRows * CO], count);
    }
    PipeBarrier<PIPE_V>();
}

// Four full-vector products use contiguous support storage first. After two
// CI layers, the next layer fits in half of the existing feature broadcast
// buffer. That layer simultaneously changes the UB layout, with no extra
// allocation. The remaining tree and basis sum preserve the original order.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void EvaluateContiguousFourSupports(const float *coefficients,
                                                      LocalTensor<float> weights,
                                                      LocalTensor<float> products,
                                                      LocalTensor<float> broadcast) {
    static_assert(CI % 8 == 0, "The resource family has complete groups of eight CI rows.");
    constexpr uint32_t matrixElements = CI * CO;
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_mul(SplinePtr(products[support * matrixElements]), SplinePtr(weights[support * matrixElements]),
                SplinePtr(broadcast), uint8_t(matrixElements / 64), 1, 1, 1, 8, 8, 8);
    PipeBarrier<PIPE_V>();
    ReduceContiguousSupportLayer<CI, CI, CO>(products);
    ReduceContiguousSupportLayer<CI / 2, CI, CO>(products);
    // The feature broadcast is dead after all products have completed.
    // Each support now stores its own CO subrow in this reused buffer.
    SetMaskNorm();
    asc_set_vector_mask(0, (uint64_t(1) << CO) - 1);
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_add(SplinePtr(broadcast[support * CO]), SplinePtr(products[support * matrixElements]),
                SplinePtr(products[support * matrixElements + (CI / 8) * CO]),
                uint8_t(CI / 8), 1, 1, 1, uint8_t(4 * CO / 8), uint8_t(CO / 8), uint8_t(CO / 8));
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 8, 4 * CO>(broadcast);
    SplineCMuls(products, broadcast, coefficients[0], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(products, broadcast[support * CO], coefficients[support], CO);
        PipeBarrier<PIPE_V>();
    }
}

// Full-vector chunks keep each support's nearby CI rows contiguous. The
// first lower/upper CI pair is produced directly in [chunk,4supports,64]
// order, using half of the existing product scratch and no UB transpose.
template <uint32_t ChunkCount>
__aicore__ inline void ReducePackedSupportChunks(LocalTensor<float> products) {
    if constexpr (ChunkCount > 1) {
        constexpr uint32_t count = (ChunkCount / 2) * 4 * kFloatVectorElements;
        // All chunk rows contain256 FP32 values, so every layer retains the
        // full64-lane mask established by the product instructions.
        asc_add(SplinePtr(products), SplinePtr(products), SplinePtr(products[count]),
                uint8_t(count / kFloatVectorElements), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        ReducePackedSupportChunks<ChunkCount / 2>(products);
    }
}

template <uint32_t RemainingRows, uint32_t CO>
__aicore__ inline void ReduceRowsInsideSupportChunk(LocalTensor<float> products) {
    if constexpr (RemainingRows > 1) {
        constexpr uint32_t halfRows = RemainingRows / 2;
        constexpr uint32_t count = halfRows * CO;
        SetMaskNorm();
        asc_set_vector_mask(0, (uint64_t(1) << count) - 1);
        // Four independent64-float support slots remain in the first chunk.
        asc_add(SplinePtr(products), SplinePtr(products), SplinePtr(products[count]),
                uint8_t(4), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        ReduceRowsInsideSupportChunk<halfRows, CO>(products);
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void EvaluatePackedFirstPair(const float *coefficients,
                                               LocalTensor<float> weights,
                                               LocalTensor<float> products,
                                               LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t halfMatrixElements = matrixElements / 2;
    constexpr uint32_t rowsPerSupportChunk = kFloatVectorElements / CO;
    constexpr uint32_t chunkCount = halfMatrixElements / kFloatVectorElements;
    static_assert(chunkCount > 0 && (chunkCount & (chunkCount - 1)) == 0 && chunkCount <= 255,
                  "The binary CI tree needs a positive power-of-two repeat count.");
    static_assert(matrixElements % (2 * kFloatVectorElements) == 0,
                  "Each CI half must contain complete FP32 vectors.");
    static_assert(kFloatVectorElements % CO == 0,
                  "A support chunk contains complete CO rows.");
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_mul(SplinePtr(products[support * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements]), SplinePtr(broadcast),
                uint8_t(chunkCount), 1, 1, 1, 32, 8, 8);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        // Both CI halves contribute their real products to the same slots.
        // Device cancellation probes qualify multiply-add rounding on the target.
        asc_fma(SplinePtr(products[support * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements + halfMatrixElements]),
                SplinePtr(broadcast[halfMatrixElements]),
                uint8_t(chunkCount), 1, 1, 1, 32, 8, 8);
    PipeBarrier<PIPE_V>();
    ReducePackedSupportChunks<chunkCount>(products);
    ReduceRowsInsideSupportChunk<rowsPerSupportChunk, CO>(products);
    SplineCMuls(products, products, coefficients[0], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(products, products[support * kFloatVectorElements], coefficients[support], CO);
        PipeBarrier<PIPE_V>();
    }
}

// The complete-output owner can blend directly into its actual target row.
// Product scratch remains separate; no support is read after being overwritten.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void EvaluatePackedMessageToOutput(const float *coefficients,
                                               LocalTensor<float> weights,
                                               LocalTensor<float> products,
                                               LocalTensor<float> broadcast,
                                               LocalTensor<float> outputRow) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t halfMatrixElements = matrixElements / 2;
    constexpr uint32_t rowsPerSupportChunk = kFloatVectorElements / CO;
    constexpr uint32_t chunkCount = halfMatrixElements / kFloatVectorElements;
    static_assert(chunkCount > 0 && (chunkCount & (chunkCount - 1)) == 0 && chunkCount <= 255,
                  "The binary CI tree needs a positive power-of-two repeat count.");
    static_assert(matrixElements % (2 * kFloatVectorElements) == 0,
                  "Each CI half must contain complete FP32 vectors.");
    static_assert(kFloatVectorElements % CO == 0,
                  "A support chunk contains complete CO rows.");
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_mul(SplinePtr(products[support * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements]), SplinePtr(broadcast),
                uint8_t(chunkCount), 1, 1, 1, 32, 8, 8);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        // Both CI halves contribute their real products to the same slots.
        // Device cancellation probes qualify multiply-add rounding on the target.
        asc_fma(SplinePtr(products[support * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements + halfMatrixElements]),
                SplinePtr(broadcast[halfMatrixElements]),
                uint8_t(chunkCount), 1, 1, 1, 32, 8, 8);
    PipeBarrier<PIPE_V>();
    ReducePackedSupportChunks<chunkCount>(products);
    ReduceRowsInsideSupportChunk<rowsPerSupportChunk, CO>(products);
    // Host already emits separate CI/CO layout keys. Restrict this measured
    // instruction schedule to a CI half that occupies exactly one vector;
    // larger matrix layouts retain their original counted blend operations.
    if constexpr (halfMatrixElements == kFloatVectorElements) {
        // The final CI layer above explicitly establishes normal mode with
        // exactly CO active lanes in this invocation. The repeat overloads keep
        // that mask while preserving all four ordered, dependent operations.
        static_assert(rowsPerSupportChunk > 1 && CO <= 32,
                      "The final CI layer establishes the complete output-row mask");
        asc_mul_scalar(SplinePtr(outputRow), SplinePtr(products), coefficients[0],
                       uint8_t(1), 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();
    #pragma unroll
        for (uint32_t support = 1; support < 4; ++support) {
            asc_axpy(SplinePtr(outputRow), SplinePtr(products[support * kFloatVectorElements]),
                     coefficients[support], uint8_t(1), 1, 1, 8, 8);
            PipeBarrier<PIPE_V>();
        }
    } else {
        SplineCMuls(outputRow, products, coefficients[0], CO);
        PipeBarrier<PIPE_V>();
    #pragma unroll
        for (uint32_t support = 1; support < 4; ++support) {
            SplineCAxpy(outputRow, products[support * kFloatVectorElements], coefficients[support], CO);
            PipeBarrier<PIPE_V>();
        }
    }
}

// Keep each knot's complete channel reduction before its basis coefficient.
// Interleaving independent supports in the output-channel dimension lets one
// vector reduction advance all of them without interpolating large weights.
// The existing product scratch budget bounds the batch width.
template <uint32_t CI, uint32_t CO, uint32_t SupportCount, bool CanonicalGrid = false, bool AllowWeightContraction = true>
__aicore__ inline void EvaluateSupportMessages(const uint32_t *knotIndices, const float *coefficients,
                                               LocalTensor<float> feature, LocalTensor<float> weights,
                                               LocalTensor<float> result, LocalTensor<float> broadcast,
                                               LocalTensor<float> messageSum) {
    constexpr uint32_t matrixElements = CI * CO;
    if constexpr (AllowWeightContraction && CanonicalGrid && SupportCount == 4 && UsePackedBilinearWeights(CI, CO) &&
                  CO % 16 == 0 && CI >= 8 && (CI & (CI - 1)) == 0) {
        ContractCanonicalBilinear<CI, CO>(coefficients, feature, weights, result, broadcast);
        return;
    }
    // Canonical bilinear callers already reserve the full packed product
    // when its byte footprint fits the shared 32-KiB capacity contract.
    constexpr uint32_t supportBatchCount =
        CanonicalGrid && SupportCount == 4 && UsePackedBilinearWeights(CI, CO)
            ? 4 : SupportBatchWidth(matrixElements, SupportCount);
    constexpr uint32_t productRowElements = supportBatchCount * CO;
    if constexpr (supportBatchCount != SupportCount)
        SplineCDuplicate(messageSum, 0.0f, CO);
    if constexpr (CO <= 32) {
#pragma unroll
        for (uint32_t column = 0; column < CO; column += 8)
            SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    } else {
        SplineCBrcb(broadcast, feature, CI / 8, {1, 8});
    }
    PipeBarrier<PIPE_V>();
    if constexpr (CanonicalGrid && SupportCount == 4 && supportBatchCount == 4 && CO <= 32 &&
                  matrixElements >= 2 * kFloatVectorElements) {
        if constexpr (matrixElements % (2 * kFloatVectorElements) == 0 &&
                      kFloatVectorElements % CO == 0 && (CI & (CI - 1)) == 0)
            EvaluatePackedFirstPair<CI, CO>(coefficients, weights, result, broadcast);
        else
            EvaluateContiguousFourSupports<CI, CO>(coefficients, weights, result, broadcast);
        return;
    }
#pragma unroll
    for (uint32_t firstSupport = 0; firstSupport < SupportCount; firstSupport += supportBatchCount) {
        if constexpr (CanonicalGrid && matrixElements / 8 <= 255 && CO / 8 < supportBatchCount) {
            // One active eight-lane block per support gathers a common
            // output-column slice from consecutive real knot matrices.
            // All strides count 32-byte UB blocks, and the support masks
            // store disjoint CO subrows inside each input-channel row.
#pragma unroll
            for (uint32_t column = 0; column < CO; column += 8) {
                SplineCMul(result[column], weights[firstSupport * matrixElements + column], broadcast[column],
                            uint64_t(supportBatchCount * 8), uint8_t(CI),
                            {uint8_t(CO / 8), uint8_t(matrixElements / 8), 0,
                             uint8_t(productRowElements / 8), uint8_t(CO / 8), uint8_t(CO / 8)});
            }
        } else {
#pragma unroll
        for (uint32_t support = 0; support < supportBatchCount; ++support) {
            auto knotWeight = weights[knotIndices[firstSupport + support] * matrixElements];
            if constexpr (CO <= 32) {
                SplineCMul(result[support * CO], knotWeight, broadcast, uint64_t(CO), uint8_t(CI),
                            {1, 1, 1, uint8_t(productRowElements / 8), uint8_t(CO / 8), uint8_t(CO / 8)});
            } else {
#pragma unroll
                for (uint32_t column = 0; column < CO; column += 64)
                    SplineCMul(result[support * CO + column], knotWeight[column], broadcast,
                                uint64_t(64), uint8_t(CI),
                                {1, 1, 0, uint8_t(productRowElements / 8), uint8_t(CO / 8), 1});
            }
        }
        }
        PipeBarrier<PIPE_V>();
        FastReduce<CI, productRowElements>(result);
        if constexpr (supportBatchCount == SupportCount) {
            // The first reduced support row can own the final message. Its
            // neighboring support rows remain intact until each is consumed.
            SplineCMuls(result, result, coefficients[0], CO);
            PipeBarrier<PIPE_V>();
#pragma unroll
            for (uint32_t support = 1; support < supportBatchCount; ++support) {
                SplineCAxpy(result, result[support * CO], coefficients[support], CO);
                PipeBarrier<PIPE_V>();
            }
        } else {
#pragma unroll
            for (uint32_t support = 0; support < supportBatchCount; ++support) {
                SplineCAxpy(messageSum, result[support * CO], coefficients[firstSupport + support], CO);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    if constexpr (supportBatchCount != SupportCount) {
        SplineCAdds(result, messageSum, 0.0f, CO);
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t CI, uint32_t CO, bool AllowWeightContraction = true>
__aicore__ inline void BuildBilinearProducts(float u, float v, LocalTensor<float> feature, LocalTensor<float> weight,
                                           LocalTensor<float> result, LocalTensor<float> broadcast,
                                           LocalTensor<float> messageSum) {
    // On an open 2x2 grid, canonical bilinear weighting extends to both
    // endpoints. Boundary wrapping only permutes zero-coefficient terms.
    uint32_t knotIndices[4] = {0, 1, 2, 3};
    float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    EvaluateSupportMessages<CI, CO, 4, true, AllowWeightContraction>(knotIndices, coefficients, feature, weight, result, broadcast, messageSum);
}
// The final channel reduction can store directly into an already cleared
// FP32 output row. Earlier tree levels keep the exact original buffer/order.
// No later code reads the old final row in the intermediate matrix.
template <uint32_t remainingRowCount, uint32_t CO>
__aicore__ inline void ReduceToOutputRow(LocalTensor<float> channelProducts, LocalTensor<float> outputRow) {
    static_assert(remainingRowCount > 1, "The aligned input-channel family has at least 8 rows.");
    constexpr uint32_t leadingRowCount = (remainingRowCount + 1) / 2;
    constexpr uint32_t mergedElementCount = (remainingRowCount - leadingRowCount) * CO;
    if constexpr (remainingRowCount == 2) {
        SplineCAdd(outputRow, channelProducts, channelProducts[CO], uint64_t(CO), uint8_t(1), {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
    } else {
        if constexpr (mergedElementCount <= 64)
            SplineCAdd(channelProducts, channelProducts, channelProducts[leadingRowCount * CO],
                uint64_t(mergedElementCount), uint8_t(1), {1, 1, 1, 8, 8, 8});
        else if constexpr (mergedElementCount % 64 == 0 && mergedElementCount / 64 <= 255)
            SplineCAdd(channelProducts, channelProducts, channelProducts[leadingRowCount * CO],
                uint64_t(64), uint8_t(mergedElementCount / 64), {1, 1, 1, 8, 8, 8});
        else
            SplineCAdd(channelProducts, channelProducts, channelProducts[leadingRowCount * CO],
                mergedElementCount);
        PipeBarrier<PIPE_V>();
        ReduceToOutputRow<leadingRowCount, CO>(channelProducts, outputRow);
    }
}
template <uint32_t CI, uint32_t CO, bool AllowWeightContraction = true>
__aicore__ inline void FastBilinear(float u, float v, LocalTensor<float> feature, LocalTensor<float> weight,
                                  LocalTensor<float> result, LocalTensor<float> broadcast,
                                  LocalTensor<float> messageSum) {
    BuildBilinearProducts<CI, CO, AllowWeightContraction>(u, v, feature, weight, result, broadcast, messageSum);
}
template <uint32_t CI, uint32_t CO>
__aicore__ inline void BilinearToOutputRow(float u, float v, LocalTensor<float> feature,
                                          LocalTensor<float> weight, LocalTensor<float> result,
                                          LocalTensor<float> broadcast, LocalTensor<float> outputRow,
                                          LocalTensor<float> messageSum) {
    BuildBilinearProducts<CI, CO>(u, v, feature, weight, result, broadcast, messageSum);
    SplineCAdds(outputRow, result, 0.0f, CO);
    PipeBarrier<PIPE_V>();
}
template <uint32_t CI, uint32_t CO>
__aicore__ inline void IndexedBilinear(float u, float v, uint32_t k00, uint32_t k10, uint32_t k01,
                                      uint32_t k11, LocalTensor<float> feature, LocalTensor<float> weight,
                                      LocalTensor<float> result, LocalTensor<float> broadcast,
                                      LocalTensor<float> messageSum) {
    uint32_t knotIndices[4] = {k00, k10, k01, k11};
    float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    EvaluateSupportMessages<CI, CO, 4>(knotIndices, coefficients, feature, weight, result, broadcast, messageSum);
}
template <uint32_t CI, uint32_t CO>
__aicore__ inline void IndexedTrilinear(float u, float v, float z, uint32_t k00, uint32_t k10, uint32_t k01,
                                       uint32_t k11, uint32_t k00Upper, uint32_t k10Upper, uint32_t k01Upper,
                                       uint32_t k11Upper, LocalTensor<float> feature,
                                       LocalTensor<float> weight, LocalTensor<float> result,
                                       LocalTensor<float> broadcast, LocalTensor<float> messageSum) {
    uint32_t knotIndices[8] = {k00, k10, k01, k11, k00Upper, k10Upper, k01Upper, k11Upper};
    float coefficients[8] = {(1 - u) * (1 - v) * (1 - z), u * (1 - v) * (1 - z),
                             (1 - u) * v * (1 - z), u * v * (1 - z),
                             (1 - u) * (1 - v) * z, u * (1 - v) * z, (1 - u) * v * z, u * v * z};
    EvaluateSupportMessages<CI, CO, 8>(knotIndices, coefficients, feature, weight, result, broadcast, messageSum);
}
template <class T> __aicore__ inline float LocalPseudo(LocalTensor<T> coords, uint32_t i) {
    if constexpr (sizeof(T) == 4)
        return SplineCGet(coords, i);
    else {
        union {
            uint32_t u;
            float f;
        } v;
        v.u = uint32_t(SplineCGet(coords.template ReinterpretCast<uint16_t>(),i)) << 16;
        return v.f;
    }
}
template <class T, uint32_t CI>
__aicore__ inline void StageFeature(GlobalTensor<T> x, uint32_t source, LocalTensor<float> feature,
                                    LocalTensor<T> raw) {
    if constexpr (sizeof(T) == 4)
        SplineCCopy(feature, x[uint64_t(source) * CI], CI);
    else if constexpr (CI % 16 == 0)
        SplineCCopy(raw, x[uint64_t(source) * CI], CI);
    else
        SplineCCopyPad(raw, x[uint64_t(source) * CI], {1, uint32_t(CI * 2), 0, 0, 0}, {false, 0, 0, 0});
}
template <class T>
__aicore__ inline void CopyControls(uint32_t e, LocalTensor<int64_t> indices, LocalTensor<T> coords,
                                    GlobalTensor<int64_t> edge, GlobalTensor<T> attr) {
    if (e % 2 == 0)
        SplineCCopy(indices, edge, 2 * e);
    else
        SplineCCopyPad(indices, edge, {1, uint32_t(2 * e * 8), 0, 0, 0}, {false, 0, 0, 0});
    if ((2 * e * sizeof(T)) % 32 == 0)
        SplineCCopy(coords, attr, 2 * e);
    else
        SplineCCopyPad(coords, attr, {1, uint32_t(2 * e * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
}
template <class T>
__aicore__ inline void FastStore(GlobalTensor<T> output, uint64_t start, LocalTensor<float> values,
                                 LocalTensor<T> raw, uint32_t count) {
    if constexpr (sizeof(T) == 4) {
        Fence<HardEvent::V_MTE3>();
        SplineCCopy(output[start], values, count);
    } else {
        SplineCCast(raw, values, RoundMode::CAST_RINT, count);
        Fence<HardEvent::V_MTE3>();
        if (count % 16 == 0)
            SplineCCopy(output[start], raw, count);
        else
            SplineCCopyPad(output[start], raw, {1, uint32_t(count * 2), 0, 0, 0});
    }
    Fence<HardEvent::MTE3_V>();
}
// Aligned message and zero-complement owners store disjoint row intervals.
// Small runtime single-edge kernels stage real controls; larger existing
// single-edge families retain their original scalar-control path.
template <class T, uint32_t CO, uint32_t FixedNodeCount = 0, bool StagedControls = false>
__aicore__ inline void ZeroSingleEdgeComplement(GM_ADDR edgeAddress, GM_ADDR outputAddress, GM_ADDR tilingAddress) {
    if constexpr (FixedNodeCount == 1)
        return;
    GlobalTensor<uint64_t> metadataGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<T> outputGm;
    metadataGm.SetGlobalBuffer((__gm__ uint64_t *)tilingAddress);
    edgesGm.SetGlobalBuffer((__gm__ int64_t *)edgeAddress);
    outputGm.SetGlobalBuffer((__gm__ T *)outputAddress);
    if constexpr (StagedControls) {
        // Host selects this owner only for the complete small-output family:
        // N*CO<=1024. Real controls occupy disjoint 32-byte UB blocks.
        LocalTensor<int64_t> endpointControl(TPosition::VECCALC, 0, 2);
        LocalTensor<uint64_t> nodeControl(TPosition::VECCALC, 32, 1);
        LocalTensor<T> zeroTile(TPosition::VECCALC, 64, 1024);
        SplineCCopyPad(endpointControl, edgesGm, {1, 16, 0, 0, 0}, {false, 0, 0, 0});
        if constexpr (FixedNodeCount == 0)
            SplineCCopyPad(nodeControl, metadataGm, {1, 8, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        constexpr uint32_t zeroElements = FixedNodeCount ? FixedNodeCount * CO : 1024;
        if constexpr (sizeof(T) == 4)
            SplineCDuplicate(zeroTile, 0.0f, zeroElements);
        else
            SplineCDuplicate(zeroTile.template ReinterpretCast<uint16_t>(), uint16_t(0), zeroElements);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        const uint32_t rows = FixedNodeCount ? FixedNodeCount : SplineCGet(nodeControl, 0) & kNodeCountMask;
        const uint32_t target = uint32_t(SplineCGet(endpointControl, 1));
        Fence<HardEvent::V_MTE3>();
        if (target != 0)
            SplineCCopyPad(outputGm, zeroTile, {1, uint32_t(target * CO * sizeof(T)), 0, 0, 0});
        if (target + 1 < rows)
            SplineCCopyPad(outputGm[uint64_t(target + 1) * CO], zeroTile,
                {1, uint32_t((rows - target - 1) * CO * sizeof(T)), 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        return;
    }
    const uint32_t nodeCount = FixedNodeCount ? FixedNodeCount : metadataGm.GetValue(0) & kNodeCountMask;
    if (nodeCount <= 1)
        return;
    LocalTensor<T> zeroValues(TPosition::VECCALC, 0, kWholeOutputCapacityElements);
    if constexpr (sizeof(T) == 4)
        SplineCDuplicate(zeroValues, 0.0f, nodeCount * CO);
    else
        SplineCDuplicate(zeroValues.template ReinterpretCast<uint16_t>(), uint16_t(0), nodeCount * CO);
    // Endpoints are immutable scalar controls. A direct GM read avoids a
    // separate control DMA; this owner never consumes another core's data.
    const uint32_t targetRowIndex = uint32_t(edgesGm.GetValue(1));
    Fence<HardEvent::V_MTE3>();
    if (targetRowIndex != 0)
        SplineCCopyPad(outputGm, zeroValues, {1, uint32_t(targetRowIndex * CO * sizeof(T)), 0, 0, 0});
    const uint32_t nextRowIndex = targetRowIndex + 1;
    if (nextRowIndex < nodeCount)
        SplineCCopyPad(outputGm[uint64_t(nextRowIndex) * CO], zeroValues,
                      {1, uint32_t((nodeCount - nextRowIndex) * CO * sizeof(T)), 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
}

// Allocate complete 32-byte blocks without adding another bank-phase shift.
// The single-edge family has already reserved these buffers in its UB budget.
template <class U>
__aicore__ inline LocalTensor<U> SmallSingleAlloc(uint32_t &offsetBytes, uint32_t elementCount) {
    LocalTensor<U> tensor(TPosition::VECCALC, offsetBytes, elementCount);
    offsetBytes += (elementCount * sizeof(U) + kUbAlignmentBytes - 1) /
                   kUbAlignmentBytes * kUbAlignmentBytes;
    return tensor;
}

// Hints and reads stay within the actual source row. Each row is an
// integral number of 64-bit words in the admitted channel families.
template <class T>
__aicore__ inline void PrefetchActualFeatureRow(GM_ADDR inputAddress, uint32_t sourceRow,
                                               uint32_t inputChannels) {
    const uint32_t rowBytes = inputChannels * sizeof(T);
    const uint64_t startByte = uint64_t(sourceRow) * rowBytes;
    for (uint32_t byteOffset = 0; byteOffset < rowBytes; byteOffset += 64) {
        GlobalTensor<uint64_t> cacheLine;
        cacheLine.SetGlobalBuffer((__gm__ uint64_t*)((__gm__ uint8_t*)inputAddress + startByte + byteOffset));
        DataCachePreload(cacheLine, int64_t(0));
    }
}

template <class T>
__aicore__ inline void ReadActualFeatureWords(GM_ADDR inputAddress, uint32_t sourceRow,
                                              uint32_t inputChannels, LocalTensor<float> feature,
                                              LocalTensor<T> rawFeature) {
    const uint32_t rowWords = inputChannels * sizeof(T) / sizeof(uint64_t);
    volatile __gm__ const uint64_t* source =
        ((__gm__ const uint64_t*)inputAddress) + uint64_t(sourceRow) * rowWords;
    LocalTensor<uint64_t> words = feature.template ReinterpretCast<uint64_t>();
    if constexpr (sizeof(T) != 4)
        words = rawFeature.template ReinterpretCast<uint64_t>();
    volatile __ubuf__ uint64_t* destination = (volatile __ubuf__ uint64_t*)words.GetPhyAddr();
    for (uint32_t begin = 0; begin < rowWords; begin += 8) {
        const uint32_t remaining = rowWords - begin;
        const uint64_t word0 = source[begin];
        const uint64_t word1 = source[begin + 1];
        uint64_t word2 = 0, word3 = 0, word4 = 0, word5 = 0, word6 = 0, word7 = 0;
        if (remaining >= 4) {
            word2 = source[begin + 2];
            word3 = source[begin + 3];
        }
        if (remaining >= 8) {
            word4 = source[begin + 4];
            word5 = source[begin + 5];
            word6 = source[begin + 6];
            word7 = source[begin + 7];
        }
        destination[begin] = word0;
        destination[begin + 1] = word1;
        if (remaining >= 4) {
            destination[begin + 2] = word2;
            destination[begin + 3] = word3;
        }
        if (remaining >= 8) {
            destination[begin + 4] = word4;
            destination[begin + 5] = word5;
            destination[begin + 6] = word6;
            destination[begin + 7] = word7;
        }
    }
    Fence<HardEvent::S_V>();
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void BlendSmallEffectiveKernel(float u, float v, LocalTensor<float> weights,
                                                 LocalTensor<float> effectiveKernel) {
    constexpr uint32_t matrixElements = CI * CO;
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    SplineCMuls(effectiveKernel, weights, coefficients[0], matrixElements);
    PipeBarrier<PIPE_V>();
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(effectiveKernel, weights[support * matrixElements], coefficients[support], matrixElements);
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void FinishStreamedGemv(LocalTensor<float> feature, LocalTensor<float> result,
                                         LocalTensor<float> broadcast);

// Read every channel of the actual row on this invocation. The complete
// raw row fits in scalar registers; no cross-call state or UB feature exists.
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void FinishScalarRowGemv(GM_ADDR inputAddress, uint32_t sourceRow,
                                          LocalTensor<float> effectiveKernel) {
    constexpr uint32_t wordCount = CI * sizeof(T) / sizeof(uint64_t);
    volatile __gm__ const uint64_t* input =
        ((__gm__ const uint64_t*)inputAddress) + uint64_t(sourceRow) * wordCount;
    uint64_t words[wordCount];
#pragma unroll
    for (uint32_t word = 0; word < wordCount; ++word)
        words[word] = input[word];
    float features[CI];
#pragma unroll
    for (uint32_t channel = 0; channel < CI; ++channel) {
        union { uint32_t bits; float value; } converted;
        if constexpr (sizeof(T) == 4)
            converted.bits = uint32_t(words[channel / 2] >> (32 * (channel % 2)));
        else
            converted.bits = uint32_t((words[channel / 4] >> (16 * (channel % 4))) & 65535) << 16;
        features[channel] = converted.value;
    }
    // Independent lower-half rows can issue together. Upper rows remain
    // intact until their corresponding scalar multiply-add consumes them.
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCMuls(effectiveKernel[row * CO], effectiveKernel[row * CO], features[row], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCAxpy(effectiveKernel[row * CO], effectiveKernel[(row + CI / 2) * CO],
                    features[row + CI / 2], CO);
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 2, CO>(effectiveKernel);
}

// Keep each support's matrix-vector product separate until CI is reduced.
// The DMA layout is [CI, support, CO], so all four products share each
// scalar feature load and each reduction instruction. Basis blending last
// avoids overflowing a rounded convex weight sum before a small X scales it.
template <uint32_t CI>
__aicore__ inline void ReadFloatScalarRow(GM_ADDR inputAddress, uint32_t sourceRow,
                                         float* features) {
    volatile __gm__ const uint64_t* input =
        ((__gm__ const uint64_t*)inputAddress) + uint64_t(sourceRow) * (CI / 2);
#pragma unroll
    for (uint32_t word = 0; word < CI / 2; ++word) {
        const uint64_t bits = input[word];
        union { uint32_t bits; float value; } low, high;
        low.bits = uint32_t(bits);
        high.bits = uint32_t(bits >> 32);
        features[2 * word] = low.value;
        features[2 * word + 1] = high.value;
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void SmallSupportFirstGemv(const float* features, float u, float v,
                                            LocalTensor<float> products, LocalTensor<float> result) {
    constexpr uint32_t supportRowElements = 4 * CO;
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCMuls(products[row * supportRowElements], products[row * supportRowElements],
                    features[row], supportRowElements);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCAxpy(products[row * supportRowElements], products[(row + CI / 2) * supportRowElements],
                    features[row + CI / 2], supportRowElements);
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 2, supportRowElements>(products);
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    SplineCMuls(result, products, coefficients[0], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(result, products[support * CO], coefficients[support], CO);
        PipeBarrier<PIPE_V>();
    }
}

// One contiguous four-support weight copy and one actual feature-row copy
// feed a shared masked product. Four eight-lane blocks gather the support
// matrices into [CI,4,CO] products; immutable weights never need a transpose.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void SmallPackedSupportDmaGemv(float u, float v, LocalTensor<float> feature,
                                               LocalTensor<float> weights, LocalTensor<float> products,
                                               LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t supportRowElements = 4 * CO;
    static_assert(CI % 8 == 0 && CO % 8 == 0 && CO / 8 < 4,
                  "A product slice contains four distinct support blocks.");
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCMul(products[column], weights[column], broadcast[column], uint64_t(32), uint8_t(CI),
            {uint8_t(CO / 8), uint8_t(matrixElements / 8), 0,
             uint8_t(supportRowElements / 8), uint8_t(CO / 8), uint8_t(CO / 8)});
    PipeBarrier<PIPE_V>();
    FastReduce<CI, supportRowElements>(products);
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    SplineCMuls(products, products, coefficients[0], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(products, products[support * CO], coefficients[support], CO);
        PipeBarrier<PIPE_V>();
    }
}

// Small source rows use actual scalar values when they fit the port budget;
// wider families keep their row DMA. The function does not specialize N.
template <class T, uint32_t HintWordCount>
__aicore__ inline void HintBoundedInputSpan(GM_ADDR inputAddress) {
    static_assert(HintWordCount > 0 && HintWordCount <= 8, "Hint count exceeds its resource class.");
#pragma unroll
    for (uint32_t byteOffset = 0; byteOffset < HintWordCount * 64; byteOffset += 64) {
        GlobalTensor<uint64_t> hintWord;
        hintWord.SetGlobalBuffer((__gm__ uint64_t*)((__gm__ uint8_t*)inputAddress + byteOffset));
        DataCachePreload(hintWord, int64_t(0));
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void PrepareSmallSupportFirstGemv(const float* features,
                                                   LocalTensor<float> products) {
    constexpr uint32_t supportRowElements = 4 * CO;
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCMuls(products[row * supportRowElements], products[row * supportRowElements],
                    features[row], supportRowElements);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCAxpy(products[row * supportRowElements], products[(row + CI / 2) * supportRowElements],
                    features[row + CI / 2], supportRowElements);
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 2, supportRowElements>(products);
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void PrepareSmallPackedSupportDmaGemv(LocalTensor<float> feature,
                                                       LocalTensor<float> weights,
                                                       LocalTensor<float> products,
                                                       LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t supportRowElements = 4 * CO;
    static_assert(CI % 8 == 0 && CO % 8 == 0 && CO / 8 < 4,
                  "A product slice contains four distinct support blocks.");
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCMul(products[column], weights[column], broadcast[column], uint64_t(32), uint8_t(CI),
            {uint8_t(CO / 8), uint8_t(matrixElements / 8), 0,
             uint8_t(supportRowElements / 8), uint8_t(CO / 8), uint8_t(CO / 8)});
    PipeBarrier<PIPE_V>();
    FastReduce<CI, supportRowElements>(products);
}

template <uint32_t CO>
__aicore__ inline void FinishSmallSupportGemv(float u, float v,
                                             LocalTensor<float> products,
                                             LocalTensor<float> result) {
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    SplineCMuls(result, products, coefficients[0], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(result, products[support * CO], coefficients[support], CO);
        PipeBarrier<PIPE_V>();
    }
}

template <class T, uint32_t CI, uint32_t CO, uint32_t HintWordCount = 0>
__aicore__ inline void RuntimeSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp,
                                       GM_ADDR yp, GM_ADDR tiling) {
    constexpr bool alignedRows = (CO * sizeof(T)) % kUbAlignmentBytes == 0;
    constexpr bool overlapFeature = CI <= 24 && CO <= 32 && CI * CO <= 256;
    // This matrix resource family regressed under packed feature DMA in
    // stable ABBA. Keep its original paired scalar support arithmetic.
    constexpr bool scalarSupportFamily = sizeof(T) == 4 && overlapFeature && CO / 8 < 4 && (4 * CO) % kFloatVectorElements == 0;
    constexpr bool packedSmallFeature = sizeof(T) == 4 && overlapFeature && CO / 8 < 4 && !scalarSupportFamily;
    constexpr uint32_t hintMinRows = HintWordCount > 0
        ? ((HintWordCount - 1) * 64 / (CI * sizeof(T))) + 1 : 0;
    constexpr uint32_t hintMaxRows = HintWordCount > 0
        ? HintWordCount * 64 / (CI * sizeof(T)) : 0;
    constexpr bool exactHintRows = HintWordCount > 0 && hintMinRows == hintMaxRows;
    const uint32_t worker = GetBlockIdx();
    if (worker != 0) {
        if constexpr (alignedRows)
            if (worker == 1)
                ZeroSingleEdgeComplement<T, CO, 0, true>(ep, yp, tiling);
        return;
    }
    GlobalTensor<T> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> metadataGm;
    x.SetGlobalBuffer((__gm__ T*)xp);
    weightsGm.SetGlobalBuffer((__gm__ T*)wp);
    coordsGm.SetGlobalBuffer((__gm__ T*)ap);
    outputGm.SetGlobalBuffer((__gm__ T*)yp);
    edgesGm.SetGlobalBuffer((__gm__ int64_t*)ep);
    metadataGm.SetGlobalBuffer((__gm__ uint64_t*)tiling);
    // Preload current-call controls while independent weight DMA and buffer
    // setup proceed. Values are still read from the actual input tensors.
    GlobalTensor<uint64_t> edgeCacheLine, coordinateCacheLine, nodeCacheLine;
    GlobalTensor<uint16_t> coordinateBits;
    edgeCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ep);
    coordinateCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ap);
    nodeCacheLine.SetGlobalBuffer((__gm__ uint64_t*)tiling);
    coordinateBits.SetGlobalBuffer((__gm__ uint16_t*)ap);
    DataCachePreload(edgeCacheLine, int64_t(0));
    DataCachePreload(coordinateCacheLine, int64_t(0));
    if constexpr (!alignedRows && !exactHintRows)
        DataCachePreload(nodeCacheLine, int64_t(0));
    // NN counts in-bounds early hint words; actual N remains runtime metadata.
    if constexpr (sizeof(T) == 2 && !alignedRows && HintWordCount > 0 &&
                  CI <= 24 && CO <= 32 && CI * CO <= 256)
        HintBoundedInputSpan<T, HintWordCount>(xp);
    uint32_t offsetBytes = 2048;
    auto feature = SmallSingleAlloc<float>(offsetBytes, CI);
    auto weights = SmallSingleAlloc<float>(offsetBytes, 4 * CI * CO);
    offsetBytes += 128;
    auto result = SmallSingleAlloc<float>(offsetBytes, BilinearProductElementCount(CI, CO));
    offsetBytes += 160;
    auto broadcast = SmallSingleAlloc<float>(offsetBytes, CI * CO);
    auto messageSum = SmallSingleAlloc<float>(offsetBytes, CO);
    auto output = SmallSingleAlloc<float>(offsetBytes, 1024);
    LocalTensor<T> rawFeature(TPosition::VECCALC, 0, 1), rawWeights(TPosition::VECCALC, 0, 1),
                   rawOutput(TPosition::VECCALC, 0, 1);
    if constexpr (sizeof(T) == 4) {
        if constexpr (overlapFeature && !packedSmallFeature) {
            // Four aligned strided copies transpose support and CI in UB.
            // Each descriptor copies all actual CI rows of one support.
            const DataCopyParams rows {uint16_t(CI), uint16_t(CO / 8), 0, uint16_t(3 * CO / 8)};
#pragma unroll
            for (uint32_t support = 0; support < 4; ++support)
                DataCopy(weights[support * CO], weightsGm[support * CI * CO], rows);
        } else {
            SplineCCopy(weights, weightsGm, 4 * CI * CO);
        }
    } else {
        offsetBytes = 65536;
        rawFeature = FastAlloc<T>(offsetBytes, CI);
        rawWeights = FastAlloc<T>(offsetBytes, 4 * CI * CO);
        rawOutput = FastAlloc<T>(offsetBytes, 1024);
        SplineCCopy(rawWeights, weightsGm, 4 * CI * CO);
    }
    if constexpr (!alignedRows) {
        // NN covers the entire actual input span. Its upper row bound also
        // covers every logical output row, without specializing exact N.
        constexpr uint32_t zeroElements = HintWordCount > 0
            ? (HintWordCount * 64 / (CI * sizeof(T))) * CO : 1024;
        static_assert(zeroElements > 0 && zeroElements <= 1024, "Whole logical output must fit");
        SplineCDuplicate(output, 0.0f, zeroElements);
    }
    const uint32_t source = uint32_t(edgesGm.GetValue(0));
    uint32_t target = 0, nodeCount = 0;
    // Limit this scalar source-row path to24 features (at most96 bytes).
    // Longer rows keep DMA; non-power-of-two CI uses the ceil reduction tree.
    if constexpr (sizeof(T) == 4 && overlapFeature && !scalarSupportFamily) {
        // The actual source index determines the feature row. All four
        // support products are independent of the later coordinate reads.
        static_assert(alignedRows, "FP32 small outputs contain complete DMA blocks");
        if constexpr (packedSmallFeature) {
            SplineCCopy(feature, x[uint64_t(source) * CI], CI);
            Fence<HardEvent::MTE2_V>();
            PrepareSmallPackedSupportDmaGemv<CI, CO>(feature, weights, result, broadcast);
        } else {
            PrefetchActualFeatureRow<T>(xp, source, CI);
            float features[CI];
            ReadFloatScalarRow<CI>(xp, source, features);
            Fence<HardEvent::MTE2_V>();
            PrepareSmallSupportFirstGemv<CI, CO>(features, weights);
        }
        const float u = PseudoValue(coordsGm, coordinateBits, 0);
        const float v = PseudoValue(coordsGm, coordinateBits, 1);
        if constexpr (packedSmallFeature)
            FinishSmallSupportGemv<CO>(u, v, result, result);
        else
            FinishSmallSupportGemv<CO>(u, v, weights, result);
        target = uint32_t(edgesGm.GetValue(1));
    } else if constexpr (overlapFeature) {
        if constexpr (packedSmallFeature)
            SplineCCopy(feature, x[uint64_t(source) * CI], CI);
        else
            PrefetchActualFeatureRow<T>(xp, source, CI);
        if constexpr (CI > 16 && CI * sizeof(T) < 64) {
            // A48-byte BF16 row can cross a64-byte cache line. Hint its
            // actual final word as well; both addresses stay inside the row.
            GlobalTensor<uint64_t> lastFeatureWord;
            lastFeatureWord.SetGlobalBuffer((__gm__ uint64_t*)((__gm__ uint8_t*)xp +
                uint64_t(source) * CI * sizeof(T) + CI * sizeof(T) - sizeof(uint64_t)));
            DataCachePreload(lastFeatureWord, int64_t(0));
        }
        target = uint32_t(edgesGm.GetValue(1));
        const float u = PseudoValue(coordsGm, coordinateBits, 0);
        const float v = PseudoValue(coordsGm, coordinateBits, 1);
        // Unaligned BF16 output keeps its original sole whole-output owner.
        // It still needs the actual runtime N for exact-length final storage.
        if constexpr (!alignedRows) {
            if constexpr (exactHintRows)
                nodeCount = hintMaxRows;
            else
                nodeCount = metadataGm.GetValue(0) & kNodeCountMask;
        }
        if constexpr (sizeof(T) == 4) {
            if constexpr (packedSmallFeature) {
                Fence<HardEvent::MTE2_V>();
                SmallPackedSupportDmaGemv<CI, CO>(u, v, feature, weights, result, broadcast);
            } else {
                float features[CI];
                ReadFloatScalarRow<CI>(xp, source, features);
                Fence<HardEvent::MTE2_V>();
                SmallSupportFirstGemv<CI, CO>(features, u, v, weights, result);
            }
        } else {
            Fence<HardEvent::MTE2_V>();
            SplineCCast(weights, rawWeights, RoundMode::CAST_NONE, 4 * CI * CO);
            PipeBarrier<PIPE_V>();
            BlendSmallEffectiveKernel<CI, CO>(u, v, weights, result);
            FinishScalarRowGemv<T, CI, CO>(xp, source, result);
        }
    } else {
        if constexpr (CI <= 16) {
            GlobalTensor<uint16_t> featureBits;
            featureBits.SetGlobalBuffer((__gm__ uint16_t*)xp);
            // No data-derived lookup is retained between calls. Every channel
            // of the actual source row is read and converted on this invocation.
            for (uint32_t channel = 0; channel < CI; ++channel)
                SplineCSet(feature, channel, PseudoValue(x, featureBits, source * CI + channel));
            Fence<HardEvent::S_V>();
        } else {
            StageFeature<T, CI>(x, source, feature, rawFeature);
        }
        target = uint32_t(edgesGm.GetValue(1));
        const float u = PseudoValue(coordsGm, coordinateBits, 0);
        const float v = PseudoValue(coordsGm, coordinateBits, 1);
        if constexpr (!alignedRows) {
            if constexpr (exactHintRows)
                nodeCount = hintMaxRows;
            else
                nodeCount = metadataGm.GetValue(0) & kNodeCountMask;
        }
        Fence<HardEvent::MTE2_V>();
        if constexpr (sizeof(T) != 4) {
            SplineCCast(weights, rawWeights, RoundMode::CAST_NONE, 4 * CI * CO);
            if constexpr (CI > 16)
                SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
            PipeBarrier<PIPE_V>();
        }
        FastBilinear<CI, CO, (sizeof(T) != 4)>(u, v, feature, weights, result, broadcast, messageSum);
    }
    if constexpr (alignedRows) {
        FastStore(outputGm, uint64_t(target) * CO, result, rawOutput, CO);
    } else {
        // Unaligned BF16 rows share DMA blocks. One owner emits the complete
        // tensor so a zero-complement writer cannot overwrite a message row.
        SplineCAdds(output[target * CO], result, 0.0f, CO);
        PipeBarrier<PIPE_V>();
        FastStore(outputGm, 0, output, rawOutput, nodeCount * CO);
    }
    PipeBarrier<PIPE_ALL>();
}
// Host admits only the complete actual input in full32-byte DMA blocks. Every
// legal source row is initialized before its aligned vector input view.
template <uint32_t CI, uint32_t CO, uint32_t InputDmaBlocks>
__aicore__ inline void CachedFloatSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap,
                                              GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
    static_assert(InputDmaBlocks > 0 && InputDmaBlocks <= 16, "Reserved512byte actual-input slab");
    constexpr uint32_t prefixElements = InputDmaBlocks * kUbAlignmentBytes / sizeof(float);
    constexpr uint32_t matrixElements = CI * CO;
    static_assert(prefixElements % CI == 0 && CO % 16 == 0 && matrixElements <= 256,
                  "Complete actual rows and full support-vector resource family");
    constexpr uint32_t outputElements = (prefixElements / CI) * CO;
    static_assert(outputElements <= 1024, "Complete small-output Host capacity");
    if (GetBlockIdx() != 0)
        return;
    GlobalTensor<float> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> edgeCacheLine, coordinateCacheLine;
    GlobalTensor<uint16_t> coordinateBits;
    x.SetGlobalBuffer((__gm__ float*)xp);
    weightsGm.SetGlobalBuffer((__gm__ float*)wp);
    coordsGm.SetGlobalBuffer((__gm__ float*)ap);
    outputGm.SetGlobalBuffer((__gm__ float*)yp);
    edgesGm.SetGlobalBuffer((__gm__ int64_t*)ep);
    edgeCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ep);
    coordinateCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ap);
    coordinateBits.SetGlobalBuffer((__gm__ uint16_t*)ap);
    DataCachePreload(edgeCacheLine, int64_t(0));
    DataCachePreload(coordinateCacheLine, int64_t(0));
    LocalTensor<float> inputPrefix(TPosition::VECCALC, 0, prefixElements);
    SplineCCopy(inputPrefix, x, prefixElements);
    uint32_t offsetBytes = kSparseControlBytes;
    auto weights = SmallSingleAlloc<float>(offsetBytes, 4 * matrixElements);
    offsetBytes += 4 * kUbAlignmentBytes;
    auto products = SmallSingleAlloc<float>(offsetBytes, 4 * matrixElements);
    offsetBytes += 5 * kUbAlignmentBytes;
    auto broadcast = SmallSingleAlloc<float>(offsetBytes, matrixElements);
    auto messageSum = SmallSingleAlloc<float>(offsetBytes, CO);
    auto wholeOutput = SmallSingleAlloc<float>(offsetBytes, outputElements);
    LocalTensor<float> unusedRawOutput(TPosition::VECCALC, 0, 1);
    SplineCDuplicate(wholeOutput, 0.0f, outputElements);
    // One real contiguous four-support DMA replaces four transposed packets.
    // The complete cached input makes the real source view independent of
    // another source-row DMA or a scalar-facing input-ready token.
    SplineCCopy(weights, weightsGm, 4 * matrixElements);
    const uint32_t source = uint32_t(edgesGm.GetValue(0));
    const uint32_t target = uint32_t(edgesGm.GetValue(1));
    const float u = PseudoValue(coordsGm, coordinateBits, 0);
    const float v = PseudoValue(coordsGm, coordinateBits, 1);
    Fence<HardEvent::MTE2_V>();
    static_assert(CO <= 32 && CI * CO >= 128 && (CI & (CI - 1)) == 0,
                  "Existing complete-input cache family uses packed support reductions.");
    float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCBrcb(broadcast[column], inputPrefix[source * CI], CI / 8, {CO / 8, CO});
    PipeBarrier<PIPE_V>();
    EvaluatePackedMessageToOutput<CI, CO>(coefficients, weights, products, broadcast,
                                         wholeOutput[target * CO]);
    FastStore(outputGm, 0, wholeOutput, unusedRawOutput, outputElements);
    PipeBarrier<PIPE_ALL>();
}

// Host admits complete input DMA blocks in a partial64-byte cache line. Every
// legal source row is initialized before its scalar feature read.
template <uint32_t CI, uint32_t CO, uint32_t InputDmaBlocks>
__aicore__ inline void PartitionPartialCachedOutput(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap,
                                              GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
    constexpr uint32_t blockElements = kUbAlignmentBytes / sizeof(float);
    constexpr uint32_t partElements = kGmCacheLineBytes / sizeof(float);
    constexpr uint32_t messageWorkers = CO / partElements;
    // Match the existing Host contract: this path launches eight AIV workers.
    constexpr uint32_t zeroWorkers = 1;
    constexpr uint32_t inputElements = InputDmaBlocks * blockElements;
    constexpr uint32_t nodeCount = inputElements / CI;
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t supportPitchElements = matrixElements + blockElements;
    static_assert(InputDmaBlocks > 0 && InputDmaBlocks <= 16 && inputElements % CI == 0,
                  "Complete actual rows fit the reserved input slab.");
    static_assert(CO % 16 == 0 && CO <= 32 && CI % 8 == 0 && matrixElements <= 256,
                  "Existing full-cache aligned small-matrix resource family.");
    const uint32_t worker = GetBlockIdx();
    if (worker >= messageWorkers + zeroWorkers)
        return;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> edgeCacheLine;
    GlobalTensor<float> outputGm;
    edgesGm.SetGlobalBuffer((__gm__ int64_t*)ep);
    edgeCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ep);
    outputGm.SetGlobalBuffer((__gm__ float*)yp);
    DataCachePreload(edgeCacheLine, int64_t(0));
    if (worker >= messageWorkers) {
        const uint32_t target = uint32_t(edgesGm.GetValue(1));
        constexpr uint32_t rowsPerZeroWorker = (nodeCount + zeroWorkers - 1) / zeroWorkers;
        const uint32_t beginRow = (worker - messageWorkers) * rowsPerZeroWorker;
        const uint32_t limitRow = beginRow + rowsPerZeroWorker;
        const uint32_t endRow = limitRow < nodeCount ? limitRow : nodeCount;
        if (beginRow >= endRow)
            return;
        LocalTensor<float> zero(TPosition::VECCALC, 0, rowsPerZeroWorker * CO);
        SplineCDuplicate(zero, 0.0f, rowsPerZeroWorker * CO);
        Fence<HardEvent::V_MTE3>();
        if (target < beginRow || target >= endRow) {
            SplineCCopy(outputGm[beginRow * CO], zero, (endRow - beginRow) * CO);
        } else {
            if (beginRow < target)
                SplineCCopy(outputGm[beginRow * CO], zero, (target - beginRow) * CO);
            if (target + 1 < endRow)
                SplineCCopy(outputGm[(target + 1) * CO], zero, (endRow - target - 1) * CO);
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }
    GlobalTensor<float> x, weightsGm, coordsGm;
    GlobalTensor<uint64_t> coordinateCacheLine;
    x.SetGlobalBuffer((__gm__ float*)xp);
    weightsGm.SetGlobalBuffer((__gm__ float*)wp);
    coordsGm.SetGlobalBuffer((__gm__ float*)ap);
    coordinateCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ap);
    if constexpr (InputDmaBlocks % 2 == 0)
        DataCachePreload(coordinateCacheLine, int64_t(0));
    LocalTensor<float> input(TPosition::VECCALC, 0, inputElements);
    SplineCCopy(input, x, inputElements);
    LocalTensor<float> coordinates(TPosition::VECCALC, kCachedFloatInputCapacityBytes, 2);
    if constexpr (InputDmaBlocks % 2 != 0) {
        // Preserve the qualified partial-cache coordinate DMA and scalar
        // readiness event. It never aliases the at-most-512-byte input slab.
        SplineCCopyPad(coordinates, coordsGm, {1, 2 * sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    uint32_t offsetBytes = kSparseControlBytes;
    auto weights = SmallSingleAlloc<float>(offsetBytes, 4 * supportPitchElements);
    auto products = SmallSingleAlloc<float>(offsetBytes, (CI / 2) * 4 * partElements);
    offsetBytes += kUbAlignmentBytes;
    auto broadcast = SmallSingleAlloc<float>(offsetBytes, CI * blockElements);
    // The public non-padding DMA API counts burst length and gaps in 32B
    // blocks. Each actual support matrix is contiguous; its 32B UB gap is
    // never read. The odd support pitch avoids coincident operand banks.
    asc_copy_gm2ub((__ubuf__ void*)weights.GetPhyAddr(), (__gm__ void*)weightsGm.GetPhyAddr(),
                  uint16_t(4), uint16_t(matrixElements / blockElements), uint16_t(0), uint16_t(1));
    // Submit the complete input and weights before scalar endpoint reads.
    const uint32_t sourceRow = uint32_t(edgesGm.GetValue(0));
    const uint32_t target = uint32_t(edgesGm.GetValue(1));
    float u, v;
    if constexpr (InputDmaBlocks % 2 != 0) {
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        u = SplineCGet(coordinates, 0);
        v = SplineCGet(coordinates, 1);
    } else {
        u = coordsGm.GetValue(0);
        v = coordsGm.GetValue(1);
    }
    Fence<HardEvent::MTE2_V>();
    SplineCBrcb(broadcast, input[sourceRow * CI], CI / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    const uint32_t channelOffset = worker * partElements;
    // Each output owner writes complete 64B cache lines. Two masked
    // eight-column slices fill separate halves of every support subrow;
    // they reuse one narrow feature broadcast and all actual weight bursts.
    SetMaskNorm();
    asc_set_vector_mask(0, (uint64_t(1) << 32) - 1);
#pragma unroll
    for (uint32_t column = 0; column < partElements; column += blockElements)
        asc_mul(SplinePtr(products[column]), SplinePtr(weights[channelOffset + column]), SplinePtr(broadcast),
                uint8_t(CI / 2), uint8_t(partElements / blockElements),
                uint8_t(supportPitchElements / blockElements), 0,
                uint8_t(4 * partElements / blockElements), uint8_t(CO / blockElements), 1);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t column = 0; column < partElements; column += blockElements)
        asc_fma(SplinePtr(products[column]),
                SplinePtr(weights[(CI / 2) * CO + channelOffset + column]),
                SplinePtr(broadcast[(CI / 2) * blockElements]), uint8_t(CI / 2),
                uint8_t(partElements / blockElements), uint8_t(supportPitchElements / blockElements), 0,
                uint8_t(4 * partElements / blockElements), uint8_t(CO / blockElements), 1);
    PipeBarrier<PIPE_V>();
    // Keep the qualified lower/upper CI FMA, then the same binary CI tree.
    FastReduce<CI / 2, 4 * partElements>(products);
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    // FastReduce explicitly ends in normal mode in this invocation.
    // Narrow that mask once to this owner's complete 64-byte output slice;
    // the same ordered coefficient operations then reuse it unchanged.
    static_assert(CI >= 4 && partElements == 16,
                  "A nonempty CI reduction establishes normal mode for a full output slice");
    asc_set_vector_mask(0, (uint64_t(1) << partElements) - 1);
    asc_mul_scalar(SplinePtr(products), SplinePtr(products), coefficients[0],
                   uint8_t(1), 1, 1, 8, 8);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        asc_axpy(SplinePtr(products), SplinePtr(products[support * partElements]),
                 coefficients[support], uint8_t(1), 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();
    }
    Fence<HardEvent::V_MTE3>();
    SplineCCopy(outputGm[target * CO + channelOffset], products, partElements);
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t CI, uint32_t CO, uint32_t InputDmaBlocks>
__aicore__ inline void CachedCoordinateSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap,
                                              GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
    // The input extent selects only the coordinate transport; ownership and
    // exact support-first arithmetic are shared with the full-cache family.
    PartitionPartialCachedOutput<CI, CO, InputDmaBlocks>(xp, ep, ap, wp, yp, tiling);
}

// CI is restricted to powers of two by tiling. CO remains a small width family;
// the CI thresholds select reduction layers without creating CI instantiations.
template <uint32_t CO, uint32_t MaximumRows = 128>
__aicore__ inline void RuntimeFeatureReduce(uint32_t inputChannels, LocalTensor<float> values) {
    if constexpr (MaximumRows > 8) {
        if (inputChannels >= MaximumRows) {
            constexpr uint32_t count = MaximumRows / 2 * CO;
            if constexpr (count <= 64)
                SplineCAdd(values, values, values[count], uint64_t(count), uint8_t(1), {1, 1, 1, 8, 8, 8});
            else
                SplineCAdd(values, values, values[count], uint64_t(64), uint8_t(count / 64), {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
        }
        RuntimeFeatureReduce<CO, MaximumRows / 2>(inputChannels, values);
    } else
        FastReduce<8, CO>(values);
}
template <uint32_t CI, uint32_t CO, bool PrepareOnly = false, bool AllowWeightContraction = true>
__aicore__ inline void RuntimeBilinear(uint32_t inputChannels, float u, float v, LocalTensor<float> feature,
                                      LocalTensor<float> weights, LocalTensor<float> result,
                                      LocalTensor<float> broadcast, LocalTensor<float> messageSum) {
    // On an open 2x2 grid, canonical bilinear weighting extends to both
    // endpoints. Boundary wrapping only permutes zero-coefficient terms.
    uint32_t knotIndices[4] = {0, 1, 2, 3};
    float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    EvaluateSupportMessages<CI, CO, 4, true, AllowWeightContraction>(knotIndices, coefficients, feature, weights, result, broadcast, messageSum);
}
template <class T>
__aicore__ inline void StageRuntimeFeature(GlobalTensor<T> x, uint32_t source, uint32_t inputChannels,
                                           LocalTensor<float> feature, LocalTensor<T> rawFeature) {
    if constexpr (sizeof(T) == 4)
        SplineCCopy(feature, x[uint64_t(source) * inputChannels], inputChannels);
    else if (inputChannels % 16 == 0)
        SplineCCopy(rawFeature, x[uint64_t(source) * inputChannels], inputChannels);
    else
        SplineCCopyPad(rawFeature, x[uint64_t(source) * inputChannels],
                    {1, uint32_t(inputChannels * 2), 0, 0, 0}, {false, 0, 0, 0});
}
// Take the actual source row from the complete immutable input cache.
// FP32 uses a local view; BF16 keeps the original input dtype until the
// message helper casts just this row. Every valid source is in the cache.
template <class T>
__aicore__ inline void SelectCachedFeatureRow(LocalTensor<T> cachedInput, uint32_t source,
                                              uint32_t inputChannels, LocalTensor<float> &feature,
                                              LocalTensor<T> &rawFeature) {
    if constexpr (sizeof(T) == 4)
        feature = cachedInput[source * inputChannels];
    else
        rawFeature = cachedInput[source * inputChannels];
}

template <class T, uint32_t CI, uint32_t CO, bool FEATURE_CACHED = false>
__aicore__ inline void
FinishRuntimeMessage(uint32_t edge, uint32_t inputChannels, LocalTensor<T> coords, LocalTensor<float> feature,
                     LocalTensor<float> weights, LocalTensor<float> result, LocalTensor<float> broadcast,
                     LocalTensor<float> messageSum, LocalTensor<T> rawFeature, LocalTensor<T> rawWeight, bool &ready, bool firstRowPublished = false) {
    float u = LocalPseudo(coords, 2 * edge), v = LocalPseudo(coords, 2 * edge + 1);
    // The complete input/weight DMA is drained once for a cached sequence.
    // Later cached rows issue no MTE2 stores; vector instructions keep their
    // existing PIPE_V barriers and never modify the immutable cache.
    if (firstRowPublished && !ready)
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    else if (!FEATURE_CACHED || !ready)
        Fence<HardEvent::MTE2_V>();
    if constexpr (sizeof(T) != 4) {
        if (!ready)
            SplineCCast(weights, rawWeight, RoundMode::CAST_NONE, 4 * inputChannels * CO);
        SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, inputChannels);
        PipeBarrier<PIPE_V>();
    }
    ready = true;
    RuntimeBilinear<CI, CO, false, (sizeof(T) != 4)>(inputChannels, u, v, feature, weights, result, broadcast, messageSum);
}
// Each token publishes one complete support matrix. Event0 remains the
// first feature-row token; the four weight tokens are consumed exactly once.
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void StageSupportWeightPackets(GlobalTensor<T> source, LocalTensor<float> weights,
                                               LocalTensor<T> rawWeights) {
    constexpr uint32_t matrixElements = CI * CO;
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support) {
        if constexpr (sizeof(T) == 4)
            SplineCCopy(weights[support * matrixElements], source[support * matrixElements], matrixElements);
        else
            SplineCCopy(rawWeights[support * matrixElements], source[support * matrixElements], matrixElements);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1 + support);
    }
}
__aicore__ inline void DrainSupportWeightPackets() {
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1 + support);
}
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void BlendSupportWeightPackets(float u, float v, LocalTensor<float> weights,
                                               LocalTensor<T> rawWeights, LocalTensor<float> effectiveKernel) {
    constexpr uint32_t matrixElements = CI * CO;
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1 + support);
        if constexpr (sizeof(T) != 4) {
            SplineCCast(weights[support * matrixElements], rawWeights[support * matrixElements],
                        RoundMode::CAST_NONE, matrixElements);
            PipeBarrier<PIPE_V>();
        }
        if (support == 0)
            SplineCMuls(effectiveKernel, weights, coefficients[0], matrixElements);
        else
            SplineCAxpy(effectiveKernel, weights[support * matrixElements], coefficients[support], matrixElements);
        PipeBarrier<PIPE_V>();
    }
}
// Effective-kernel storage can be overwritten by the consumed lower half.
// Wide CO uses compact scalar broadcast blocks; small CO uses full rows.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void FinishStreamedGemv(LocalTensor<float> feature, LocalTensor<float> result,
                                         LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO;
    if constexpr (CO <= 32) {
#pragma unroll
        for (uint32_t column = 0; column < CO; column += 8)
            SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    } else {
        SplineCBrcb(broadcast, feature, CI / 8, {1, 8});
    }
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
    if constexpr (CO <= 32) {
        asc_mul(SplinePtr(result), SplinePtr(result), SplinePtr(broadcast),
                uint8_t(matrixElements / 128), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        asc_fma(SplinePtr(result), SplinePtr(result[matrixElements / 2]),
                SplinePtr(broadcast[matrixElements / 2]), uint8_t(matrixElements / 128),
                1, 1, 1, 8, 8, 8);
    } else {
#pragma unroll
        for (uint32_t column = 0; column < CO; column += 64)
            asc_mul(SplinePtr(result[column]), SplinePtr(result[column]), SplinePtr(broadcast),
                    uint8_t(CI / 2), 1, 1, 0, CO / 8, CO / 8, 1);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t column = 0; column < CO; column += 64)
            asc_fma(SplinePtr(result[column]), SplinePtr(result[matrixElements / 2 + column]),
                    SplinePtr(broadcast[(CI / 2) * 8]), uint8_t(CI / 2),
                    1, 1, 0, CO / 8, CO / 8, 1);
    }
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 2, CO>(result);
}
template <class T, uint32_t CO, bool Published>
__aicore__ inline void StoreSparseZeroComplement(GlobalTensor<T> output, LocalTensor<T> zero,
                                                 uint32_t begin, uint32_t end, uint64_t occupiedRows) {
    if constexpr (Published) {
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    } else {
        if constexpr (sizeof(T) == 4)
            SplineCDuplicate(zero, 0.0f, (end - begin) * CO);
        else
            SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0), (end - begin) * CO);
        Fence<HardEvent::V_MTE3>();
    }
    uint32_t cursor = end - begin;
    while (occupiedRows) {
        const uint32_t row = 63 - uint32_t(CountLeadingZero(occupiedRows));
        if (cursor > row + 1)
            SplineCCopyPad(output[uint64_t(begin + row + 1) * CO], zero,
                          {1, uint32_t((cursor - row - 1) * CO * sizeof(T)), 0, 0, 0});
        cursor = row;
        occupiedRows &= ~(uint64_t(1) << row);
    }
    if (cursor)
        SplineCCopyPad(output[uint64_t(begin) * CO], zero,
                      {1, uint32_t(cursor * CO * sizeof(T)), 0, 0, 0});
}
template <class U>
__aicore__ inline LocalTensor<U> SparseRuntimeAlloc(uint32_t &offsetBytes, uint32_t elementCount) {
    LocalTensor<U> buffer(TPosition::VECCALC, offsetBytes, elementCount);
    const uint32_t allocationBytes =
        (elementCount * sizeof(U) + kUbAlignmentBytes - 1) / kUbAlignmentBytes * kUbAlignmentBytes;
    offsetBytes += allocationBytes;
    return buffer;
}
// Two message slots share a channel-reduction instruction. Each slot keeps
// the original lower/upper CI pair and support-coefficient arithmetic.
template <uint32_t ChunkCount>
__aicore__ inline void ReducePairedMessageChunks(LocalTensor<float> products) {
    if constexpr (ChunkCount > 1) {
        constexpr uint32_t elements = (ChunkCount / 2) * 8 * kFloatVectorElements;
        asc_add(SplinePtr(products), SplinePtr(products), SplinePtr(products[elements]),
                uint8_t(elements / kFloatVectorElements), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        ReducePairedMessageChunks<ChunkCount / 2>(products);
    }
}
template <uint32_t RemainingRows, uint32_t CO>
__aicore__ inline void ReducePairedRowsInsideChunk(LocalTensor<float> products) {
    if constexpr (RemainingRows > 1) {
        constexpr uint32_t halfRows = RemainingRows / 2;
        constexpr uint32_t count = halfRows * CO;
        SetMaskNorm();
        asc_set_vector_mask(0, (uint64_t(1) << count) - 1);
        asc_add(SplinePtr(products), SplinePtr(products), SplinePtr(products[count]),
                uint8_t(8), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        ReducePairedRowsInsideChunk<halfRows, CO>(products);
    }
}
template <uint32_t CI, uint32_t CO>
__aicore__ inline void ProduceIncomingPairSlot(LocalTensor<float> feature,
                                               LocalTensor<float> weights,
                                               LocalTensor<float> products,
                                               LocalTensor<float> broadcast,
                                               uint32_t slot) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t halfMatrixElements = matrixElements / 2;
    constexpr uint32_t chunkCount = halfMatrixElements / kFloatVectorElements;
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_mul(SplinePtr(products[(slot * 4 + support) * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements]), SplinePtr(broadcast),
                uint8_t(chunkCount), 1, 1, 1, 64, 8, 8);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_fma(SplinePtr(products[(slot * 4 + support) * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements + halfMatrixElements]),
                SplinePtr(broadcast[halfMatrixElements]),
                uint8_t(chunkCount), 1, 1, 1, 64, 8, 8);
    PipeBarrier<PIPE_V>();
}
template <uint32_t CO>
__aicore__ inline void FinishIncomingPairSlot(float u, float v, LocalTensor<float> products) {
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    SplineCMuls(products, products, coefficients[0], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(products, products[support * kFloatVectorElements], coefficients[support], CO);
        PipeBarrier<PIPE_V>();
    }
}

template <class T, uint32_t CI, uint32_t CO, bool Coowned, bool CacheFeatures = false,
          uint32_t StaticEdgeCount = 0>
__aicore__ inline void RuntimeSparseGraph(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp,
                                          GM_ADDR tiling) {
    constexpr uint32_t inputChannels = CI, matrixElements = CI * CO;
    // At least16 FP32 vector repeats per matrix amortize four packet tokens.
    // The existing complete owner group and UB budget remain unchanged.
    constexpr bool streamWeightStages = sizeof(T) != 4 && Coowned && StaticEdgeCount == 8 && !CacheFeatures &&
        matrixElements >= 16 * kFloatVectorElements && matrixElements <= 2048 && CO * sizeof(T) >= 32;
    uint32_t worker = GetBlockIdx(), offsetBytes = 2048;
    LocalTensor<int64_t> indices(TPosition::VECCALC, 0, 128);
    LocalTensor<T> coords(TPosition::VECCALC, 1024, 128);
    LocalTensor<uint64_t> metadataLocal(TPosition::VECCALC, 1536, 4);
    // Repartition the existing nine32-byte gap blocks. Only the product
    // base moves; weight, broadcast and later slabs retain their addresses.
    // Relative operand phases are a measured layout hypothesis on this target.
    constexpr uint32_t productGapBytes = 4 * kUbAlignmentBytes;
    constexpr uint32_t broadcastGapBytes = 5 * kUbAlignmentBytes;
    static_assert(productGapBytes + broadcastGapBytes == 256 + 32,
                  "The original UB budget must remain unchanged.");
    auto feature = SparseRuntimeAlloc<float>(offsetBytes, inputChannels),
         weights = SparseRuntimeAlloc<float>(offsetBytes, 4 * matrixElements);
    offsetBytes += productGapBytes;
    auto result = SparseRuntimeAlloc<float>(offsetBytes, BilinearProductElementCount(CI, CO));
    offsetBytes += broadcastGapBytes;
    auto broadcast = SparseRuntimeAlloc<float>(offsetBytes, inputChannels * (CO <= 32 ? CO : 8)),
         accumulator = SparseRuntimeAlloc<float>(offsetBytes, CO),
         messageSum = SparseRuntimeAlloc<float>(offsetBytes, CO);
    LocalTensor<T> rawFeature(TPosition::VECCALC, 0, 1), rawWeight(TPosition::VECCALC, 0, 1),
        rawOutput(TPosition::VECCALC, 0, 1);
    if constexpr (sizeof(T) != 4) {
        offsetBytes = (offsetBytes + 65535) / 65536 * 65536;
        rawFeature = SparseRuntimeAlloc<T>(offsetBytes, inputChannels);
        rawWeight = SparseRuntimeAlloc<T>(offsetBytes, 4 * matrixElements);
        rawOutput = SparseRuntimeAlloc<T>(offsetBytes, CO);
    }
    auto zero = SparseRuntimeAlloc<T>(offsetBytes, 4096);
    GlobalTensor<T> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> metadata;
    x.SetGlobalBuffer((__gm__ T *)xp);
    weightsGm.SetGlobalBuffer((__gm__ T *)wp);
    coordsGm.SetGlobalBuffer((__gm__ T *)ap);
    outputGm.SetGlobalBuffer((__gm__ T *)yp);
    edgesGm.SetGlobalBuffer((__gm__ int64_t *)ep);
    metadata.SetGlobalBuffer((__gm__ uint64_t *)tiling);
    // Runtime owner counts arrive in metadata. A static complete owner group
    // already knows every active worker and control-array length before DMA.
    bool weightPrefetched = Coowned || worker < 8;
    uint64_t packed = 0;
    if constexpr (Coowned) {
        if constexpr (StaticEdgeCount != 0) {
            if (worker >= StaticEdgeCount)
                return;
        }
        if constexpr (StaticEdgeCount != 0) {
            DataCachePreload(metadata, int64_t(0));
        } else {
            SplineCCopy(metadataLocal, metadata, 4);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        }
        if constexpr (StaticEdgeCount != 0) {
            // A complete static owner group knows the control lengths before
            // N arrives. Queue endpoints, coordinates and weights now so the
            // metadata wait does not serialize the remaining input DMA.
            SplineCCopyPad(indices, edgesGm, {1, uint32_t(2 * StaticEdgeCount * 8), 0, 0, 0}, {false, 0, 0, 0});
            SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * StaticEdgeCount * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            if constexpr (streamWeightStages)
                StageSupportWeightPackets<T, CI, CO>(weightsGm, weights, rawWeight);
            else if constexpr (sizeof(T) == 4)
                SplineCCopy(weights, weightsGm, 4 * matrixElements);
            else
                SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
        }
        if constexpr (StaticEdgeCount != 0) {
            // Controls have compile-time lengths, so the packed metadata
            // fetch can overlap endpoint, coordinate and weight transfers.
            packed = metadata.GetValue(0);
        } else {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            packed = SplineCGet(metadataLocal, 0);
        }
    } else {
        DataCachePreload(metadata, int64_t(0));
        if (weightPrefetched) {
            if constexpr (sizeof(T) == 4)
                SplineCCopy(weights, weightsGm, 4 * matrixElements);
            else
                SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
        }
        packed = metadata.GetValue(0);
    }
    uint32_t edgeCount = 0;
    uint32_t nodeCount = packed & kNodeCountMask,
             tileRows = (packed >> kZeroTileRowsShift) & kZeroTileRowsMask;
    bool mean = (packed >> kMeanShift) & 1;
    constexpr bool cacheFeatures = CacheFeatures;
    LocalTensor<T> cachedInput(TPosition::VECCALC, 0, 1);
    constexpr bool coowned = Coowned;
    if constexpr (Coowned && StaticEdgeCount != 0) {
        const uint32_t zeroBeginRow = worker * tileRows;
        const uint32_t zeroEndRow = zeroBeginRow + tileRows < nodeCount ? zeroBeginRow + tileRows : nodeCount;
        if (zeroBeginRow < zeroEndRow) {
            if constexpr (sizeof(T) == 4)
                SplineCDuplicate(zero, 0.0f, (zeroEndRow - zeroBeginRow) * CO);
            else
                SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0),
                                 (zeroEndRow - zeroBeginRow) * CO);
            // Event 1 publishes only this zero buffer. Later interpolation
            // instructions can proceed while MTE3 consumes the early buffer.
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        }
    }
    if constexpr (Coowned) {
        if constexpr (StaticEdgeCount != 0) {
            edgeCount = StaticEdgeCount;
        } else {
            edgeCount = (packed >> kSparseEdgeShift) & kSparseEdgeMask;
            // Inactive workers drain metadata loading and perform no
            // graph input DMA or output store. Active owners complete the full graph.
            if (worker >= edgeCount) {
                PipeBarrier<PIPE_ALL>();
                return;
            }
            SplineCCopyPad(indices, edgesGm, {1, uint32_t(2 * edgeCount * 8), 0, 0, 0}, {false, 0, 0, 0});
            SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * edgeCount * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            if constexpr (sizeof(T) == 4)
                SplineCCopy(weights, weightsGm, 4 * matrixElements);
            else
                SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
        }
    }

    if constexpr (!Coowned) {
        edgeCount = (packed >> kSparseEdgeShift) & kSparseEdgeMask;
        SplineCCopyPad(indices, edgesGm, {1, uint32_t(2 * edgeCount * 8), 0, 0, 0}, {false, 0, 0, 0});
        if (worker < edgeCount)
            SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * edgeCount * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    if (worker < edgeCount && !weightPrefetched) {
        if constexpr (sizeof(T) == 4)
            SplineCCopy(weights, weightsGm, 4 * matrixElements);
        else
            SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
    }
    if (cacheFeatures && worker < edgeCount) {
        const uint32_t featureElements = nodeCount * inputChannels;
        cachedInput = SparseRuntimeAlloc<T>(offsetBytes, featureElements);
        // Queue the complete input before waiting for endpoints. Ownership
        // scanning can overlap this DMA; FinishRuntimeMessage drains MTE2
        // before reading the chosen row. Idle/zero owners never load X.
        SplineCCopy(cachedInput, x, featureElements);
    }
    if constexpr (!Coowned) {
        if (worker >= edgeCount) {
            if constexpr (sizeof(T) == 4)
                SplineCDuplicate(zero, 0.0f, tileRows * CO);
            else
                SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0), tileRows * CO);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t edge = 1; edge < edgeCount; ++edge) {
                int64_t key = SplineCGet(indices, edgeCount + edge);
                uint32_t position = edge;
                while (position && SplineCGet(indices, edgeCount + position - 1) > key) {
                    SplineCSet(indices, edgeCount + position, SplineCGet(indices, edgeCount + position - 1));
                    --position;
                }
                SplineCSet(indices, edgeCount + position, key);
            }
            Fence<HardEvent::V_MTE3>();
            uint32_t tileCount = (nodeCount + tileRows - 1) / tileRows,
                     zeroWorkerCount = GetBlockNum() - edgeCount;
            for (uint32_t tile = worker - edgeCount; tile < tileCount; tile += zeroWorkerCount) {
                uint32_t begin = tile * tileRows,
                         end = begin + tileRows < nodeCount ? begin + tileRows : nodeCount, cursor = begin;
                for (uint32_t edge = 0; edge < edgeCount; ++edge) {
                    uint32_t destination = SplineCGet(indices, edgeCount + edge);
                    if (destination < cursor)
                        continue;
                    if (destination >= end)
                        break;
                    if (destination > cursor)
                        SplineCCopyPad(outputGm[uint64_t(cursor) * CO], zero,
                                    {1, uint32_t((destination - cursor) * CO * sizeof(T)), 0, 0, 0});
                    cursor = destination + 1;
                }
                if (cursor < end)
                    SplineCCopyPad(outputGm[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((end - cursor) * CO * sizeof(T)), 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
            return;
        }
    }
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint32_t target = SplineCGet(indices, edgeCount + worker), incomingCount = 0;
    // Every message worker owns a real edge. Stage its actual source row as
    // soon as endpoints are ready; ownership scans do not consume this row.
    // Duplicate-target nonleaders drain this issued DMA before returning.
    if constexpr (!CacheFeatures) {
        StageRuntimeFeature(x, SplineCGet(indices, worker), inputChannels, feature, rawFeature);
        if constexpr (StaticEdgeCount != 0)
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    // Queue current-edge weight arithmetic before scalar ownership scanning.
    // Every active worker consumes its four packet tokens here. A nonleader
    // later drains the queued work; leaders reuse this current-call kernel.
    if constexpr (streamWeightStages)
        BlendSupportWeightPackets<T, CI, CO>(LocalPseudo(coords, 2 * worker),
            LocalPseudo(coords, 2 * worker + 1), weights, rawWeight, result);
    bool leader = true;
    uint32_t begin = worker * tileRows, end = begin + tileRows < nodeCount ? begin + tileRows : nodeCount;
    uint64_t occupiedRows = 0;
    if constexpr (Coowned) {
#pragma unroll
        for (uint32_t edge = 0; edge < 8; ++edge) {
            if (edge >= edgeCount)
                continue;
            uint32_t destination = SplineCGet(indices, edgeCount + edge);
            if (destination == target) {
                ++incomingCount;
                if (edge < worker)
                    leader = false;
            }
            if (destination >= begin && destination < end)
                occupiedRows |= uint64_t(1) << (destination - begin);
        }
    } else {
        for (uint32_t edge = 0; edge < edgeCount; ++edge) {
            uint32_t destination = SplineCGet(indices, edgeCount + edge);
            if (destination == target) {
                ++incomingCount;
                if (edge < worker)
                    leader = false;
            }
        }
    }
    // The complete static owner group can overlap its first real feature row
    // with zero-complement stores, including repeated-target accumulation.
    if (leader && cacheFeatures && (StaticEdgeCount != 0 || incomingCount == 1))
        SelectCachedFeatureRow(cachedInput, SplineCGet(indices, worker), inputChannels, feature, rawFeature);
    if constexpr (streamWeightStages)
        if (coowned && begin < end)
            StoreSparseZeroComplement<T, CO, true>(outputGm, zero, begin, end, occupiedRows);
    bool messagePrepared = false;
    if constexpr (streamWeightStages) {
        if (leader) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (sizeof(T) != 4) {
                SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, inputChannels);
                PipeBarrier<PIPE_V>();
            }
            FinishStreamedGemv<CI, CO>(feature, result, broadcast);
            messagePrepared = true;
        }
    } else if constexpr (Coowned && StaticEdgeCount != 0 && !CacheFeatures) {
        if (leader && incomingCount == 1) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (sizeof(T) != 4) {
                SplineCCast(weights, rawWeight, RoundMode::CAST_NONE, 4 * matrixElements);
                SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, inputChannels);
                PipeBarrier<PIPE_V>();
            }
            RuntimeBilinear<CI, CO, true, (sizeof(T) != 4)>(inputChannels, LocalPseudo(coords, 2 * worker),
                                         LocalPseudo(coords, 2 * worker + 1), feature, weights, result,
                                         broadcast, messageSum);
            messagePrepared = true;
        }
    }
    if constexpr (!streamWeightStages)
        if (coowned && begin < end)
            StoreSparseZeroComplement<T, CO, StaticEdgeCount != 0>(outputGm, zero, begin, end, occupiedRows);
    if (!leader) {
        if constexpr (StaticEdgeCount != 0 && !CacheFeatures)
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    if (messagePrepared && incomingCount == 1) {
        // The complete support GEMVs finished before independent zero DMA.
        FastStore(outputGm, uint64_t(target) * CO, result, rawOutput, CO);
        return;
    }
    const bool firstMessageReady = streamWeightStages && messagePrepared;
    bool ready = firstMessageReady;
    if (incomingCount == 1) {
        if (cacheFeatures)
            FinishRuntimeMessage<T, CI, CO, true>(worker, inputChannels, coords, feature, weights, result,
                                              broadcast, messageSum, rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
        else
            FinishRuntimeMessage<T, CI, CO>(worker, inputChannels, coords, feature, weights, result, broadcast, messageSum,
                                        rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
        FastStore(outputGm, uint64_t(target) * CO, result, rawOutput, CO);
        return;
    }
    if (firstMessageReady)
        SplineCAdds(accumulator, result, 0.0f, CO);
    else
        SplineCDuplicate(accumulator, 0.0f, CO);
    PipeBarrier<PIPE_V>();
    const uint32_t firstUnprocessedEdge = firstMessageReady ? worker + 1 : worker;
    // Dispatch once per target. The uncached loop retains its original
    // row-DMA schedule and pays no cache selection branch per message.
    if (cacheFeatures) {
        for (uint32_t edge = firstUnprocessedEdge; edge < edgeCount; ++edge) {
            if (SplineCGet(indices, edgeCount + edge) != target)
                continue;
            SelectCachedFeatureRow(cachedInput, SplineCGet(indices, edge), inputChannels, feature, rawFeature);
            FinishRuntimeMessage<T, CI, CO, true>(edge, inputChannels, coords, feature, weights, result,
                                              broadcast, messageSum, rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
            SplineCAdd(accumulator, accumulator, result, CO);
            PipeBarrier<PIPE_V>();
        }
    } else {
        for (uint32_t edge = firstUnprocessedEdge; edge < edgeCount; ++edge) {
            if (SplineCGet(indices, edgeCount + edge) != target)
                continue;
            // The first matching edge is the leader's own staged row. Only
            // later messages replace it, after the previous V consumer.
            if (ready) {
                Fence<HardEvent::V_MTE2>();
                StageRuntimeFeature(x, SplineCGet(indices, edge), inputChannels, feature, rawFeature);
            }
            FinishRuntimeMessage<T, CI, CO>(edge, inputChannels, coords, feature, weights, result, broadcast, messageSum,
                                        rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
            SplineCAdd(accumulator, accumulator, result, CO);
            PipeBarrier<PIPE_V>();
        }
    }
    if (mean) {
        SplineCMuls(accumulator, accumulator, 1.0f / incomingCount, CO);
        PipeBarrier<PIPE_V>();
    }
    FastStore(outputGm, uint64_t(target) * CO, accumulator, rawOutput, CO);
}
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void PairedIncomingGraph(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp,
                                          GM_ADDR tiling) {
    constexpr bool Coowned = true, CacheFeatures = false;
    constexpr uint32_t StaticEdgeCount = kPairedOwnerGroupWidth;
    constexpr uint32_t inputChannels = CI, matrixElements = CI * CO;
    static_assert(sizeof(T) == 4 && CO >= 16 && CO <= 32 &&
        kFloatVectorElements % CO == 0 && (CI & (CI - 1)) == 0 &&
        uint64_t(4) * matrixElements * sizeof(float) >= kPairedProductMinimumBytes &&
        uint64_t(4) * matrixElements * sizeof(float) <= kPairedProductMaximumBytes,
        "Host-selected complete paired-product resource family");
    // At least16 FP32 vector repeats per matrix amortize four packet tokens.
    // The existing complete owner group and UB budget remain unchanged.
    constexpr bool streamWeightStages = sizeof(T) != 4 && Coowned && StaticEdgeCount == 8 && !CacheFeatures &&
        matrixElements >= 16 * kFloatVectorElements && matrixElements <= 2048 && CO * sizeof(T) >= 32;
    uint32_t worker = GetBlockIdx(), offsetBytes = 2048;
    LocalTensor<int64_t> indices(TPosition::VECCALC, 0, 128);
    LocalTensor<T> coords(TPosition::VECCALC, 1024, 128);
    LocalTensor<uint64_t> metadataLocal(TPosition::VECCALC, 1536, 4);
    // Move broadcast back one32-byte block and place the same block after
    // it. Every other live buffer base and the total UB budget stay exact.
    // The relative operand phase was measured on the target device.
    constexpr uint32_t productGapBytes = 4 * kUbAlignmentBytes;
    constexpr uint32_t broadcastGapBytes = 4 * kUbAlignmentBytes;
    static_assert(productGapBytes + broadcastGapBytes + kUbAlignmentBytes == 256 + 32,
                  "The original UB budget must remain unchanged.");
    auto feature = SparseRuntimeAlloc<float>(offsetBytes, inputChannels),
         weights = SparseRuntimeAlloc<float>(offsetBytes, 4 * matrixElements);
    offsetBytes += productGapBytes;
    auto result = SparseRuntimeAlloc<float>(offsetBytes, BilinearProductElementCount(CI, CO));
    offsetBytes += broadcastGapBytes;
    auto broadcast = SparseRuntimeAlloc<float>(offsetBytes, inputChannels * (CO <= 32 ? CO : 8));
    // Move only the broadcast base back one32-byte block. Restore that block
    // here so accumulator,message scratch,zero slab and total UB stay exact.
    offsetBytes += kUbAlignmentBytes;
    auto accumulator = SparseRuntimeAlloc<float>(offsetBytes, CO),
         messageSum = SparseRuntimeAlloc<float>(offsetBytes, CO);
    LocalTensor<T> rawFeature(TPosition::VECCALC, 0, 1), rawWeight(TPosition::VECCALC, 0, 1),
        rawOutput(TPosition::VECCALC, 0, 1);
    if constexpr (sizeof(T) != 4) {
        offsetBytes = (offsetBytes + 65535) / 65536 * 65536;
        rawFeature = SparseRuntimeAlloc<T>(offsetBytes, inputChannels);
        rawWeight = SparseRuntimeAlloc<T>(offsetBytes, 4 * matrixElements);
        rawOutput = SparseRuntimeAlloc<T>(offsetBytes, CO);
    }
    auto zero = SparseRuntimeAlloc<T>(offsetBytes, 4096);
    GlobalTensor<T> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> metadata;
    x.SetGlobalBuffer((__gm__ T *)xp);
    weightsGm.SetGlobalBuffer((__gm__ T *)wp);
    coordsGm.SetGlobalBuffer((__gm__ T *)ap);
    outputGm.SetGlobalBuffer((__gm__ T *)yp);
    edgesGm.SetGlobalBuffer((__gm__ int64_t *)ep);
    metadata.SetGlobalBuffer((__gm__ uint64_t *)tiling);
    // Runtime owner counts arrive in metadata. A static complete owner group
    // already knows every active worker and control-array length before DMA.
    bool weightPrefetched = Coowned || worker < 8;
    uint64_t packed = 0;
    if constexpr (Coowned) {
        if constexpr (StaticEdgeCount != 0) {
            if (worker >= StaticEdgeCount)
                return;
        }
        if constexpr (StaticEdgeCount != 0) {
            DataCachePreload(metadata, int64_t(0));
        } else {
            SplineCCopy(metadataLocal, metadata, 4);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        }
        if constexpr (StaticEdgeCount != 0) {
            // A complete static owner group knows the control lengths before
            // N arrives. Queue endpoints, coordinates and weights now so the
            // metadata wait does not serialize the remaining input DMA.
            SplineCCopyPad(indices, edgesGm, {1, uint32_t(2 * StaticEdgeCount * 8), 0, 0, 0}, {false, 0, 0, 0});
            SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * StaticEdgeCount * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            if constexpr (streamWeightStages)
                StageSupportWeightPackets<T, CI, CO>(weightsGm, weights, rawWeight);
            else if constexpr (sizeof(T) == 4)
                SplineCCopy(weights, weightsGm, 4 * matrixElements);
            else
                SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
        }
        if constexpr (StaticEdgeCount != 0) {
            // Controls have compile-time lengths, so the packed metadata
            // fetch can overlap endpoint, coordinate and weight transfers.
            packed = metadata.GetValue(0);
        } else {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            packed = SplineCGet(metadataLocal, 0);
        }
    } else {
        DataCachePreload(metadata, int64_t(0));
        if (weightPrefetched) {
            if constexpr (sizeof(T) == 4)
                SplineCCopy(weights, weightsGm, 4 * matrixElements);
            else
                SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
        }
        packed = metadata.GetValue(0);
    }
    uint32_t edgeCount = 0;
    uint32_t nodeCount = packed & kNodeCountMask,
             tileRows = (packed >> kZeroTileRowsShift) & kZeroTileRowsMask;
    bool mean = (packed >> kMeanShift) & 1;
    constexpr bool cacheFeatures = CacheFeatures;
    LocalTensor<T> cachedInput(TPosition::VECCALC, 0, 1);
    constexpr bool coowned = Coowned;
    if constexpr (Coowned && StaticEdgeCount != 0) {
        const uint32_t zeroBeginRow = worker * tileRows;
        const uint32_t zeroEndRow = zeroBeginRow + tileRows < nodeCount ? zeroBeginRow + tileRows : nodeCount;
        if (zeroBeginRow < zeroEndRow) {
            if constexpr (sizeof(T) == 4)
                SplineCDuplicate(zero, 0.0f, (zeroEndRow - zeroBeginRow) * CO);
            else
                SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0),
                                 (zeroEndRow - zeroBeginRow) * CO);
            // Event 1 publishes only this zero buffer. Later interpolation
            // instructions can proceed while MTE3 consumes the early buffer.
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        }
    }
    if constexpr (Coowned) {
        if constexpr (StaticEdgeCount != 0) {
            edgeCount = StaticEdgeCount;
        } else {
            edgeCount = (packed >> kSparseEdgeShift) & kSparseEdgeMask;
            // Inactive workers drain metadata loading and perform no
            // graph input DMA or output store. Active owners complete the full graph.
            if (worker >= edgeCount) {
                PipeBarrier<PIPE_ALL>();
                return;
            }
            SplineCCopyPad(indices, edgesGm, {1, uint32_t(2 * edgeCount * 8), 0, 0, 0}, {false, 0, 0, 0});
            SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * edgeCount * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            if constexpr (sizeof(T) == 4)
                SplineCCopy(weights, weightsGm, 4 * matrixElements);
            else
                SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
        }
    }

    if constexpr (!Coowned) {
        edgeCount = (packed >> kSparseEdgeShift) & kSparseEdgeMask;
        SplineCCopyPad(indices, edgesGm, {1, uint32_t(2 * edgeCount * 8), 0, 0, 0}, {false, 0, 0, 0});
        if (worker < edgeCount)
            SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * edgeCount * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    if (worker < edgeCount && !weightPrefetched) {
        if constexpr (sizeof(T) == 4)
            SplineCCopy(weights, weightsGm, 4 * matrixElements);
        else
            SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
    }
    if (cacheFeatures && worker < edgeCount) {
        const uint32_t featureElements = nodeCount * inputChannels;
        cachedInput = SparseRuntimeAlloc<T>(offsetBytes, featureElements);
        // Queue the complete input before waiting for endpoints. Ownership
        // scanning can overlap this DMA; FinishRuntimeMessage drains MTE2
        // before reading the chosen row. Idle/zero owners never load X.
        SplineCCopy(cachedInput, x, featureElements);
    }
    if constexpr (!Coowned) {
        if (worker >= edgeCount) {
            if constexpr (sizeof(T) == 4)
                SplineCDuplicate(zero, 0.0f, tileRows * CO);
            else
                SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0), tileRows * CO);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t edge = 1; edge < edgeCount; ++edge) {
                int64_t key = SplineCGet(indices, edgeCount + edge);
                uint32_t position = edge;
                while (position && SplineCGet(indices, edgeCount + position - 1) > key) {
                    SplineCSet(indices, edgeCount + position, SplineCGet(indices, edgeCount + position - 1));
                    --position;
                }
                SplineCSet(indices, edgeCount + position, key);
            }
            Fence<HardEvent::V_MTE3>();
            uint32_t tileCount = (nodeCount + tileRows - 1) / tileRows,
                     zeroWorkerCount = GetBlockNum() - edgeCount;
            for (uint32_t tile = worker - edgeCount; tile < tileCount; tile += zeroWorkerCount) {
                uint32_t begin = tile * tileRows,
                         end = begin + tileRows < nodeCount ? begin + tileRows : nodeCount, cursor = begin;
                for (uint32_t edge = 0; edge < edgeCount; ++edge) {
                    uint32_t destination = SplineCGet(indices, edgeCount + edge);
                    if (destination < cursor)
                        continue;
                    if (destination >= end)
                        break;
                    if (destination > cursor)
                        SplineCCopyPad(outputGm[uint64_t(cursor) * CO], zero,
                                    {1, uint32_t((destination - cursor) * CO * sizeof(T)), 0, 0, 0});
                    cursor = destination + 1;
                }
                if (cursor < end)
                    SplineCCopyPad(outputGm[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((end - cursor) * CO * sizeof(T)), 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
            return;
        }
    }
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint32_t target = SplineCGet(indices, edgeCount + worker), incomingCount = 0;
    // Every message worker owns a real edge. Stage its actual source row as
    // soon as endpoints are ready; ownership scans do not consume this row.
    // Duplicate-target nonleaders drain this issued DMA before returning.
    if constexpr (!CacheFeatures) {
        StageRuntimeFeature(x, SplineCGet(indices, worker), inputChannels, feature, rawFeature);
        if constexpr (StaticEdgeCount != 0)
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    // Queue current-edge weight arithmetic before scalar ownership scanning.
    // Every active worker consumes its four packet tokens here. A nonleader
    // later drains the queued work; leaders reuse this current-call kernel.
    if constexpr (streamWeightStages)
        BlendSupportWeightPackets<T, CI, CO>(LocalPseudo(coords, 2 * worker),
            LocalPseudo(coords, 2 * worker + 1), weights, rawWeight, result);
    bool leader = true;
    uint32_t begin = worker * tileRows, end = begin + tileRows < nodeCount ? begin + tileRows : nodeCount;
    uint64_t occupiedRows = 0;
    if constexpr (Coowned) {
#pragma unroll
        for (uint32_t edge = 0; edge < 8; ++edge) {
            if (edge >= edgeCount)
                continue;
            uint32_t destination = SplineCGet(indices, edgeCount + edge);
            if (destination == target) {
                ++incomingCount;
                if (edge < worker)
                    leader = false;
            }
            if (destination >= begin && destination < end)
                occupiedRows |= uint64_t(1) << (destination - begin);
        }
    } else {
        for (uint32_t edge = 0; edge < edgeCount; ++edge) {
            uint32_t destination = SplineCGet(indices, edgeCount + edge);
            if (destination == target) {
                ++incomingCount;
                if (edge < worker)
                    leader = false;
            }
        }
    }
    // The complete static owner group can overlap its first real feature row
    // with zero-complement stores, including repeated-target accumulation.
    if (leader && cacheFeatures && (StaticEdgeCount != 0 || incomingCount == 1))
        SelectCachedFeatureRow(cachedInput, SplineCGet(indices, worker), inputChannels, feature, rawFeature);
    if constexpr (streamWeightStages)
        if (coowned && begin < end)
            StoreSparseZeroComplement<T, CO, true>(outputGm, zero, begin, end, occupiedRows);
    bool messagePrepared = false;
    if constexpr (streamWeightStages) {
        if (leader) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (sizeof(T) != 4) {
                SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, inputChannels);
                PipeBarrier<PIPE_V>();
            }
            FinishStreamedGemv<CI, CO>(feature, result, broadcast);
            messagePrepared = true;
        }
    } else if constexpr (Coowned && StaticEdgeCount != 0 && !CacheFeatures) {
        if (leader && incomingCount == 1) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (sizeof(T) != 4) {
                SplineCCast(weights, rawWeight, RoundMode::CAST_NONE, 4 * matrixElements);
                SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, inputChannels);
                PipeBarrier<PIPE_V>();
            }
            RuntimeBilinear<CI, CO, true, (sizeof(T) != 4)>(inputChannels, LocalPseudo(coords, 2 * worker),
                                         LocalPseudo(coords, 2 * worker + 1), feature, weights, result,
                                         broadcast, messageSum);
            messagePrepared = true;
        }
    }
    if constexpr (!streamWeightStages)
        if (coowned && begin < end)
            StoreSparseZeroComplement<T, CO, StaticEdgeCount != 0>(outputGm, zero, begin, end, occupiedRows);
    if (!leader) {
        if constexpr (StaticEdgeCount != 0 && !CacheFeatures)
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    if (messagePrepared && incomingCount == 1) {
        // The complete support GEMVs finished before independent zero DMA.
        FastStore(outputGm, uint64_t(target) * CO, result, rawOutput, CO);
        return;
    }
    // Host fixed the capacity/alignment family. Actual graph fan-in still
    // determines whether this current invocation has two messages to batch.
    if (incomingCount > 1) {
        constexpr uint32_t secondSlot = 4 * kFloatVectorElements;
        static_assert(CI <= secondSlot, "Second real row fits one untouched message slot");
        SplineCDuplicate(accumulator, 0.0f, CO);
        PipeBarrier<PIPE_V>();
        uint32_t currentEdge = worker;
        bool firstStagedRow = true;
        while (currentEdge < edgeCount) {
            if (!firstStagedRow) {
                Fence<HardEvent::V_MTE2>();
                StageRuntimeFeature(x, SplineCGet(indices, currentEdge), inputChannels, feature, rawFeature);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            uint32_t secondEdge = currentEdge + 1;
            while (secondEdge < edgeCount && SplineCGet(indices, edgeCount + secondEdge) != target)
                ++secondEdge;
            if (secondEdge < edgeCount) {
                // Slot1 is untouched by slot0 products. The first input's
                // vector work overlaps this complete second-row transfer.
                SplineCCopy(result[secondSlot], x[uint64_t(SplineCGet(indices, secondEdge)) * CI], CI);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                ProduceIncomingPairSlot<CI, CO>(feature, weights, result, broadcast, 0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                ProduceIncomingPairSlot<CI, CO>(result[secondSlot], weights, result, broadcast, 1);
                ReducePairedMessageChunks<matrixElements / (2 * kFloatVectorElements)>(result);
                ReducePairedRowsInsideChunk<kFloatVectorElements / CO, CO>(result);
                FinishIncomingPairSlot<CO>(LocalPseudo(coords, 2 * currentEdge),
                                           LocalPseudo(coords, 2 * currentEdge + 1), result);
                SplineCAdd(accumulator, accumulator, result, CO);
                PipeBarrier<PIPE_V>();
                FinishIncomingPairSlot<CO>(LocalPseudo(coords, 2 * secondEdge),
                                           LocalPseudo(coords, 2 * secondEdge + 1), result[secondSlot]);
                SplineCAdd(accumulator, accumulator, result[secondSlot], CO);
                PipeBarrier<PIPE_V>();
                currentEdge = secondEdge + 1;
            } else {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                RuntimeBilinear<CI, CO, false, false>(inputChannels,
                    LocalPseudo(coords, 2 * currentEdge), LocalPseudo(coords, 2 * currentEdge + 1),
                    feature, weights, result, broadcast, messageSum);
                SplineCAdd(accumulator, accumulator, result, CO);
                PipeBarrier<PIPE_V>();
                currentEdge = edgeCount;
            }
            firstStagedRow = false;
            while (currentEdge < edgeCount && SplineCGet(indices, edgeCount + currentEdge) != target)
                ++currentEdge;
        }
        if (mean) {
            SplineCMuls(accumulator, accumulator, 1.0f / incomingCount, CO);
            PipeBarrier<PIPE_V>();
        }
        FastStore(outputGm, uint64_t(target) * CO, accumulator, rawOutput, CO);
        return;
    }
    const bool firstMessageReady = streamWeightStages && messagePrepared;
    bool ready = firstMessageReady;
    if (incomingCount == 1) {
        if (cacheFeatures)
            FinishRuntimeMessage<T, CI, CO, true>(worker, inputChannels, coords, feature, weights, result,
                                              broadcast, messageSum, rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
        else
            FinishRuntimeMessage<T, CI, CO>(worker, inputChannels, coords, feature, weights, result, broadcast, messageSum,
                                        rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
        FastStore(outputGm, uint64_t(target) * CO, result, rawOutput, CO);
        return;
    }
    if (firstMessageReady)
        SplineCAdds(accumulator, result, 0.0f, CO);
    else
        SplineCDuplicate(accumulator, 0.0f, CO);
    PipeBarrier<PIPE_V>();
    const uint32_t firstUnprocessedEdge = firstMessageReady ? worker + 1 : worker;
    // Dispatch once per target. The uncached loop retains its original
    // row-DMA schedule and pays no cache selection branch per message.
    if (cacheFeatures) {
        for (uint32_t edge = firstUnprocessedEdge; edge < edgeCount; ++edge) {
            if (SplineCGet(indices, edgeCount + edge) != target)
                continue;
            SelectCachedFeatureRow(cachedInput, SplineCGet(indices, edge), inputChannels, feature, rawFeature);
            FinishRuntimeMessage<T, CI, CO, true>(edge, inputChannels, coords, feature, weights, result,
                                              broadcast, messageSum, rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
            SplineCAdd(accumulator, accumulator, result, CO);
            PipeBarrier<PIPE_V>();
        }
    } else {
        for (uint32_t edge = firstUnprocessedEdge; edge < edgeCount; ++edge) {
            if (SplineCGet(indices, edgeCount + edge) != target)
                continue;
            // The first matching edge is the leader's own staged row. Only
            // later messages replace it, after the previous V consumer.
            if (ready) {
                Fence<HardEvent::V_MTE2>();
                StageRuntimeFeature(x, SplineCGet(indices, edge), inputChannels, feature, rawFeature);
            }
            FinishRuntimeMessage<T, CI, CO>(edge, inputChannels, coords, feature, weights, result, broadcast, messageSum,
                                        rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
            SplineCAdd(accumulator, accumulator, result, CO);
            PipeBarrier<PIPE_V>();
        }
    }
    if (mean) {
        SplineCMuls(accumulator, accumulator, 1.0f / incomingCount, CO);
        PipeBarrier<PIPE_V>();
    }
    FastStore(outputGm, uint64_t(target) * CO, accumulator, rawOutput, CO);
}
// A complete small-edge bucket has one leader per connected destination and
// one separate zero-complement owner. Padding reserves whole 8-worker groups;
// padding workers return before DMA. N and mean remain runtime metadata.
template <class U> __aicore__ inline LocalTensor<U> StaticGraphAlloc(uint32_t &offset, uint32_t count) {
    LocalTensor<U> tensor(TPosition::VECCALC, offset, count);
    offset += (count * sizeof(U) + 31) / 32 * 32;
    return tensor;
}
// Separate the packed result from weights by 128 bytes and retain the
// existing total 288-byte padding before the broadcast buffer. Both offsets
// are included in the host's conservative UB budget. This avoids the aliases
// created by contiguous power-of-two matrix buffers.
template <class U, uint32_t paddingBytes>
__aicore__ inline LocalTensor<U> StaticGraphPaddedAlloc(uint32_t &offset, uint32_t count) {
    offset += paddingBytes;
    return StaticGraphAlloc<U>(offset, count);
}
template <class T, uint32_t CI, uint32_t CO, bool PREFETCH = false>
__aicore__ inline void StaticGraphMessage(uint32_t e, uint32_t source, GlobalTensor<T> x, GlobalTensor<T> gmW,
                                          GlobalTensor<T> pseudo, GlobalTensor<uint16_t> pseudoBits,
                                          LocalTensor<float> feature, LocalTensor<float> weight,
                                          LocalTensor<float> result, LocalTensor<float> broadcast,
                                          LocalTensor<float> messageSum, LocalTensor<T> rawFeature, LocalTensor<T> rawWeight, bool &ready) {
    constexpr uint32_t M = CI * CO;
    if (ready)
        Fence<HardEvent::V_MTE2>();
    if constexpr (sizeof(T) == 4) {
        if (!ready && !PREFETCH)
            SplineCCopy(weight, gmW, 4 * M);
        SplineCCopy(feature, x[uint64_t(source) * CI], CI);
    } else {
        if (!ready && !PREFETCH)
            SplineCCopy(rawWeight, gmW, 4 * M);
        if constexpr (CI % 16 == 0)
            SplineCCopy(rawFeature, x[uint64_t(source) * CI], CI);
        else
            SplineCCopyPad(rawFeature, x[uint64_t(source) * CI], {1, CI * 2, 0, 0, 0}, {false, 0, 0, 0});
    }
    float u = PseudoValue(pseudo, pseudoBits, 2 * e), v = PseudoValue(pseudo, pseudoBits, 2 * e + 1);
    Fence<HardEvent::MTE2_V>();
    if constexpr (sizeof(T) != 4) {
        if (!ready)
            SplineCCast(weight, rawWeight, RoundMode::CAST_NONE, 4 * M);
        SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
        PipeBarrier<PIPE_V>();
    }
    ready = true;
    FastBilinear<CI, CO, (sizeof(T) != 4)>(u, v, feature, weight, result, broadcast, messageSum);
}
// The caller reads the actual coordinates before consuming the input-ready
// token. Arithmetic and first weight conversion keep the original order.
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void StaticGraphStagedMessage(float u, float v,
                                              LocalTensor<float> feature, LocalTensor<float> weight,
                                              LocalTensor<float> result, LocalTensor<float> broadcast,
                                              LocalTensor<float> messageSum, LocalTensor<T> rawFeature, LocalTensor<T> rawWeight,
                                              bool &ready) {
    constexpr uint32_t matrixElements = CI * CO;
    if constexpr (sizeof(T) != 4) {
        if (!ready)
            SplineCCast(weight, rawWeight, RoundMode::CAST_NONE, 4 * matrixElements);
        SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
        PipeBarrier<PIPE_V>();
    }
    ready = true;
    FastBilinear<CI, CO, (sizeof(T) != 4)>(u, v, feature, weight, result, broadcast, messageSum);
}
template <class T, uint32_t CI, uint32_t CO, bool MEAN, uint32_t ES = 0, bool RUNTIME_MEAN = false>
__aicore__ inline void StaticEdgeGraph(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp,
                                       GM_ADDR tiling, uint32_t runtimeEdgeCount = 0) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr bool streamWeightStages = sizeof(T) != 4 && matrixElements >= 512 && matrixElements <= 2048 &&
        matrixElements % 128 == 0 && CO % 16 == 0;
    uint32_t e = ES ? ES : runtimeEdgeCount;
    if (GetBlockIdx() > e)
        return;
    GlobalTensor<T> x, w, a, y;
    GlobalTensor<int64_t> edge;
    GlobalTensor<uint16_t> bits;
    x.SetGlobalBuffer((__gm__ T *)xp);
    w.SetGlobalBuffer((__gm__ T *)wp);
    a.SetGlobalBuffer((__gm__ T *)ap);
    y.SetGlobalBuffer((__gm__ T *)yp);
    edge.SetGlobalBuffer((__gm__ int64_t *)ep);
    bits.SetGlobalBuffer((__gm__ uint16_t *)ap);
    uint32_t offset = 0, block = GetBlockIdx();
    auto indices = StaticGraphAlloc<int64_t>(offset, 2 * e);
    auto stagedCoordinates = StaticGraphAlloc<T>(offset, 2 * e);
    SplineCCopyPad(indices, edge, {1, uint32_t(2 * e * 8), 0, 0, 0}, {false, 0, 0, 0});
    if (block < e)
        SplineCCopyPad(stagedCoordinates, a, {1, uint32_t(2 * e * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    if (block < e) {
        auto feature = StaticGraphAlloc<float>(offset, CI),
             weight = StaticGraphAlloc<float>(offset, 4 * CI * CO),
             result = StaticGraphPaddedAlloc<float, 128>(offset, BilinearProductElementCount(CI, CO)),
             broadcast = StaticGraphPaddedAlloc<float, 160>(offset, CI * (CO <= 32 ? CO : 8)),
             output = StaticGraphAlloc<float>(offset, CO),
             messageSum = StaticGraphAlloc<float>(offset, CO);
        LocalTensor<T> rawFeature(TPosition::VECCALC, 0, 1), rawWeight(TPosition::VECCALC, 0, 1);
        if constexpr (sizeof(T) != 4) {
            rawFeature = StaticGraphAlloc<T>(offset, CI > CO ? CI : CO);
            rawWeight = StaticGraphAlloc<T>(offset, 4 * CI * CO);
        }
        if constexpr (streamWeightStages)
            StageSupportWeightPackets<T, CI, CO>(w, weight, rawWeight);
        else if constexpr (sizeof(T) == 4)
            SplineCCopy(weight, w, 4 * CI * CO);
        else
            SplineCCopy(rawWeight, w, 4 * CI * CO);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        const bool stageFirstFeature = e > 0 && e < 8;
        if (stageFirstFeature) {
            // The actual own-edge row is independent of target ownership.
            // Publish it before any later mean or alternate-slot DMA.
            StageFeature<T, CI>(x, SplineCGet(indices, block), feature, rawFeature);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        uint32_t target = SplineCGet(indices, e + block), count = 0;
        for (uint32_t j = 0; j < e; ++j) {
            if (SplineCGet(indices, e + j) == target) {
                if (j < block) {
                    if constexpr (streamWeightStages)
                        DrainSupportWeightPackets();
                    if (stageFirstFeature)
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    // Retain the original MTE2-to-scalar drain on nonleaders.
                    Fence<HardEvent::MTE2_S>();
                    return;
                }
                ++count;
            }
        }
        bool ready = false;
        if (count == 1) {
            if (stageFirstFeature) {
                float coordinateU = LocalPseudo(stagedCoordinates, 2 * block),
                      coordinateV = LocalPseudo(stagedCoordinates, 2 * block + 1);
                if constexpr (streamWeightStages) {
                    BlendSupportWeightPackets<T, CI, CO>(coordinateU, coordinateV, weight, rawWeight, result);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    if constexpr (sizeof(T) != 4) {
                        SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
                        PipeBarrier<PIPE_V>();
                    }
                    FinishStreamedGemv<CI, CO>(feature, result, broadcast);
                    ready = true;
                } else {
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    StaticGraphStagedMessage<T, CI, CO>(coordinateU, coordinateV, feature, weight, result,
                                                        broadcast, messageSum, rawFeature, rawWeight, ready);
                }
            } else {
                if constexpr (streamWeightStages)
                    DrainSupportWeightPackets();
                StaticGraphMessage<T, CI, CO, true>(block, SplineCGet(indices, block), x, w, a, bits, feature,
                                                    weight, result, broadcast, messageSum, rawFeature, rawWeight, ready);
            }
            FastStore(y, uint64_t(target) * CO, result, rawFeature, CO);
        } else {
            // Repeated destinations retain the established double-buffered
            // row loop. Drain each weight token once before its first cast.
            if constexpr (streamWeightStages)
                DrainSupportWeightPackets();
            LocalTensor<uint64_t> meanData(TPosition::VECCALC, 0, 1);
            if constexpr (RUNTIME_MEAN) {
                meanData = StaticGraphAlloc<uint64_t>(offset, 4);
                GlobalTensor<uint64_t> metadata;
                metadata.SetGlobalBuffer((__gm__ uint64_t *)tiling);
                SplineCCopyPad(meanData, metadata, {1, 8, 0, 0, 0}, {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            }
            // One edge has no future message to overlap. Keep the original
            // ES1 accumulator, including its empty invalid-endpoint fallback.
            if (e == 1) {
                // Preserve the original empty invalid-endpoint fallback while
                // consuming the early row's already published token.
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SplineCDuplicate(output, 0.0f, CO);
                PipeBarrier<PIPE_V>();
                for (uint32_t j = block; j < e; ++j)
                    if (SplineCGet(indices, e + j) == target) {
                        StaticGraphMessage<T, CI, CO, true>(j, SplineCGet(indices, j), x, w, a, bits, feature,
                                                            weight, result, broadcast, messageSum, rawFeature, rawWeight,
                                                            ready);
                        SplineCAdd(output, output, result, CO);
                        PipeBarrier<PIPE_V>();
                    }
            } else {
                // A future input row occupies the other staging slot. BF16 keeps
                // one converted FP32 feature and alternates only its raw input.
                LocalTensor<float> alternateFeature = feature;
                LocalTensor<T> alternateRawFeature = rawFeature;
                if constexpr (sizeof(T) == 4)
                    alternateFeature = StaticGraphAlloc<float>(offset, CI);
                else
                    alternateRawFeature = StaticGraphAlloc<T>(offset, CI);
                SplineCDuplicate(output, 0.0f, CO);
                PipeBarrier<PIPE_V>();
                uint32_t currentEdge = block;
                while (currentEdge < e && SplineCGet(indices, e + currentEdge) != target)
                    ++currentEdge;
                if (currentEdge < e) {
                    bool alternateInput = false;
                    if (!stageFirstFeature) {
                        StageFeature<T, CI>(x, SplineCGet(indices, currentEdge), feature, rawFeature);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    }
                    // Every active small-edge owner has already published its
                    // first row; the slot0 wait consumes that token.
                    while (currentEdge < e) {
                        uint32_t nextEdge = currentEdge + 1;
                        while (nextEdge < e && SplineCGet(indices, e + nextEdge) != target)
                            ++nextEdge;
                        if (nextEdge < e) {
                            if (alternateInput) {
                                StageFeature<T, CI>(x, SplineCGet(indices, nextEdge), feature, rawFeature);
                                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                            } else {
                                StageFeature<T, CI>(x, SplineCGet(indices, nextEdge), alternateFeature,
                                                    alternateRawFeature);
                                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                            }
                        }
                        float coordinateU = LocalPseudo(stagedCoordinates, 2 * currentEdge),
                              coordinateV = LocalPseudo(stagedCoordinates, 2 * currentEdge + 1);
                        if (alternateInput)
                            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                        else
                            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        auto currentFeature = alternateInput ? alternateFeature : feature;
                        auto currentRawFeature = alternateInput ? alternateRawFeature : rawFeature;
                        StaticGraphStagedMessage<T, CI, CO>(coordinateU, coordinateV, currentFeature, weight,
                                                            result, broadcast, messageSum, currentRawFeature, rawWeight,
                                                            ready);
                        SplineCAdd(output, output, result, CO);
                        PipeBarrier<PIPE_V>();
                        if (nextEdge == e)
                            break;
                        // Only a consumed slot can be refilled. This retains one
                        // V_MTE2 dependency per transition, with no unused final token.
                        Fence<HardEvent::V_MTE2>();
                        currentEdge = nextEdge;
                        alternateInput = !alternateInput;
                    }
                }
            }
            bool applyMean = MEAN;
            if constexpr (RUNTIME_MEAN) {
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
                applyMean = (SplineCGet(meanData, 0) >> kMeanShift) & 1;
            }
            if (applyMean) {
                // This branch follows count==1. In a legal two-edge graph,
                // the current edge always matches its own destination, so
                // count is exactly2 here and its reciprocal is exactly0.5.
                // Other complete edge-count families keep runtime division.
                const float meanScale = e == 2 ? 0.5f : 1.0f / count;
                SplineCMuls(output, output, meanScale, CO);
                PipeBarrier<PIPE_V>();
            }
            FastStore(y, uint64_t(target) * CO, output, rawFeature, CO);
        }
    } else {
        if (e <= 8) {
            GlobalTensor<uint64_t> metadata;
            metadata.SetGlobalBuffer((__gm__ uint64_t *)tiling);
            auto md = StaticGraphAlloc<uint64_t>(offset, 4);
            SplineCCopyPad(md, metadata, {1, 8, 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            uint32_t n = SplineCGet(md, 0) & kNodeCountMask;
            auto zero = StaticGraphAlloc<T>(offset, 4096);
            if constexpr (sizeof(T) == 4)
                SplineCDuplicate(zero, 0.0f, n * CO);
            else
                SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0), n * CO);
            if (e > 0 && e < 8) {
                // One bit represents one real destination row. The capacity
                // test is independent of endpoint values and avoids sorting
                // a padded list when the whole node domain fits uint64_t.
                if (n <= 64) {
                    uint64_t occupiedRows = 0;
                    for (uint32_t edgeIndex = 0; edgeIndex < e; ++edgeIndex) {
                        uint32_t destinationRow = uint32_t(SplineCGet(indices, e + edgeIndex));
                        if (destinationRow < n)
                            occupiedRows |= uint64_t(1) << destinationRow;
                    }
                    Fence<HardEvent::V_MTE3>();
                    // Enumerate occupied rows in descending order. Every
                    // emitted interval excludes message rows; duplicate
                    // destinations share one bit and require no extra store.
                    uint32_t intervalEndRow = n;
                    while (occupiedRows != 0) {
                        uint32_t destinationRow = 63 - uint32_t(CountLeadingZero(occupiedRows));
                        uint32_t intervalBeginRow = destinationRow + 1;
                        if (intervalBeginRow < intervalEndRow)
                            SplineCCopyPad(y[uint64_t(intervalBeginRow) * CO], zero,
                                        {1, uint32_t((intervalEndRow - intervalBeginRow) * CO * sizeof(T)), 0, 0, 0});
                        intervalEndRow = destinationRow;
                        occupiedRows &= ~(uint64_t(1) << destinationRow);
                    }
                    if (intervalEndRow != 0)
                        SplineCCopyPad(y, zero, {1, uint32_t(intervalEndRow * CO * sizeof(T)), 0, 0, 0});
                    Fence<HardEvent::MTE3_V>();
                    return;
                }
            }
            uint32_t z0 = e > 0 ? uint32_t(SplineCGet(indices, e + 0)) : n,
                     z1 = e > 1 ? uint32_t(SplineCGet(indices, e + 1)) : n,
                     z2 = e > 2 ? uint32_t(SplineCGet(indices, e + 2)) : n,
                     z3 = e > 3 ? uint32_t(SplineCGet(indices, e + 3)) : n,
                     z4 = e > 4 ? uint32_t(SplineCGet(indices, e + 4)) : n,
                     z5 = e > 5 ? uint32_t(SplineCGet(indices, e + 5)) : n,
                     z6 = e > 6 ? uint32_t(SplineCGet(indices, e + 6)) : n,
                     z7 = e > 7 ? uint32_t(SplineCGet(indices, e + 7)) : n;
            uint32_t lo, hi;
            if constexpr (ES > 0 && ES <= 2) {
                lo = z0 < z1 ? z0 : z1;
                hi = z0 > z1 ? z0 : z1;
                z0 = lo;
                z1 = hi;
            } else if constexpr (ES > 0 && ES <= 4) {
                lo = z0 < z1 ? z0 : z1;
                hi = z0 > z1 ? z0 : z1;
                z0 = lo;
                z1 = hi;
                lo = z2 < z3 ? z2 : z3;
                hi = z2 > z3 ? z2 : z3;
                z2 = lo;
                z3 = hi;
                lo = z0 < z2 ? z0 : z2;
                hi = z0 > z2 ? z0 : z2;
                z0 = lo;
                z2 = hi;
                lo = z1 < z3 ? z1 : z3;
                hi = z1 > z3 ? z1 : z3;
                z1 = lo;
                z3 = hi;
                lo = z1 < z2 ? z1 : z2;
                hi = z1 > z2 ? z1 : z2;
                z1 = lo;
                z2 = hi;
            } else {
                lo = z0 < z1 ? z0 : z1;
                hi = z0 > z1 ? z0 : z1;
                z0 = lo;
                z1 = hi;
                lo = z2 < z3 ? z2 : z3;
                hi = z2 > z3 ? z2 : z3;
                z2 = lo;
                z3 = hi;
                lo = z4 < z5 ? z4 : z5;
                hi = z4 > z5 ? z4 : z5;
                z4 = lo;
                z5 = hi;
                lo = z6 < z7 ? z6 : z7;
                hi = z6 > z7 ? z6 : z7;
                z6 = lo;
                z7 = hi;
                lo = z0 < z2 ? z0 : z2;
                hi = z0 > z2 ? z0 : z2;
                z0 = lo;
                z2 = hi;
                lo = z1 < z3 ? z1 : z3;
                hi = z1 > z3 ? z1 : z3;
                z1 = lo;
                z3 = hi;
                lo = z4 < z6 ? z4 : z6;
                hi = z4 > z6 ? z4 : z6;
                z4 = lo;
                z6 = hi;
                lo = z5 < z7 ? z5 : z7;
                hi = z5 > z7 ? z5 : z7;
                z5 = lo;
                z7 = hi;
                lo = z1 < z2 ? z1 : z2;
                hi = z1 > z2 ? z1 : z2;
                z1 = lo;
                z2 = hi;
                lo = z5 < z6 ? z5 : z6;
                hi = z5 > z6 ? z5 : z6;
                z5 = lo;
                z6 = hi;
                lo = z0 < z4 ? z0 : z4;
                hi = z0 > z4 ? z0 : z4;
                z0 = lo;
                z4 = hi;
                lo = z3 < z7 ? z3 : z7;
                hi = z3 > z7 ? z3 : z7;
                z3 = lo;
                z7 = hi;
                lo = z1 < z5 ? z1 : z5;
                hi = z1 > z5 ? z1 : z5;
                z1 = lo;
                z5 = hi;
                lo = z2 < z6 ? z2 : z6;
                hi = z2 > z6 ? z2 : z6;
                z2 = lo;
                z6 = hi;
                lo = z1 < z4 ? z1 : z4;
                hi = z1 > z4 ? z1 : z4;
                z1 = lo;
                z4 = hi;
                lo = z3 < z6 ? z3 : z6;
                hi = z3 > z6 ? z3 : z6;
                z3 = lo;
                z6 = hi;
                lo = z2 < z4 ? z2 : z4;
                hi = z2 > z4 ? z2 : z4;
                z2 = lo;
                z4 = hi;
                lo = z3 < z5 ? z3 : z5;
                hi = z3 > z5 ? z3 : z5;
                z3 = lo;
                z5 = hi;
                lo = z3 < z4 ? z3 : z4;
                hi = z3 > z4 ? z3 : z4;
                z3 = lo;
                z4 = hi;
            }
            Fence<HardEvent::V_MTE3>();
            uint32_t cursor = 0;
            if (z0 >= cursor && z0 < n) {
                if (z0 > cursor)
                    SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((z0 - cursor) * CO * sizeof(T)), 0, 0, 0});
                cursor = z0 + 1;
            }
            if (z1 >= cursor && z1 < n) {
                if (z1 > cursor)
                    SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((z1 - cursor) * CO * sizeof(T)), 0, 0, 0});
                cursor = z1 + 1;
            }
            if (z2 >= cursor && z2 < n) {
                if (z2 > cursor)
                    SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((z2 - cursor) * CO * sizeof(T)), 0, 0, 0});
                cursor = z2 + 1;
            }
            if (z3 >= cursor && z3 < n) {
                if (z3 > cursor)
                    SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((z3 - cursor) * CO * sizeof(T)), 0, 0, 0});
                cursor = z3 + 1;
            }
            if (z4 >= cursor && z4 < n) {
                if (z4 > cursor)
                    SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((z4 - cursor) * CO * sizeof(T)), 0, 0, 0});
                cursor = z4 + 1;
            }
            if (z5 >= cursor && z5 < n) {
                if (z5 > cursor)
                    SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((z5 - cursor) * CO * sizeof(T)), 0, 0, 0});
                cursor = z5 + 1;
            }
            if (z6 >= cursor && z6 < n) {
                if (z6 > cursor)
                    SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((z6 - cursor) * CO * sizeof(T)), 0, 0, 0});
                cursor = z6 + 1;
            }
            if (z7 >= cursor && z7 < n) {
                if (z7 > cursor)
                    SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((z7 - cursor) * CO * sizeof(T)), 0, 0, 0});
                cursor = z7 + 1;
            }
            if (cursor < n)
                SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                            {1, uint32_t((n - cursor) * CO * sizeof(T)), 0, 0, 0});
            Fence<HardEvent::MTE3_V>();
            return;
        }
        GlobalTensor<uint64_t> metadata;
        metadata.SetGlobalBuffer((__gm__ uint64_t *)tiling);
        uint32_t n = metadata.GetValue(0) & kNodeCountMask, begin = 0, rows = n, end = n;
        auto zero = StaticGraphAlloc<T>(offset, rows * CO);
        if constexpr (sizeof(T) == 4)
            SplineCDuplicate(zero, 0.0f, rows * CO);
        else
            SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0), rows * CO);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t j = 1; j < e; ++j) {
            int64_t key = SplineCGet(indices, e + j);
            uint32_t q = j;
            while (q > 0 && SplineCGet(indices, e + q - 1) > key) {
                SplineCSet(indices, e + q, SplineCGet(indices, e + q - 1));
                --q;
            }
            SplineCSet(indices, e + q, key);
        }
        Fence<HardEvent::V_MTE3>();
        uint32_t cursor = begin;
        for (uint32_t j = 0; j < e; ++j) {
            uint32_t dst = SplineCGet(indices, e + j);
            if (dst < cursor)
                continue;
            if (dst >= end)
                break;
            if (dst > cursor)
                SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                            {1, uint32_t((dst - cursor) * CO * sizeof(T)), 0, 0, 0});
            cursor = dst + 1;
        }
        if (cursor < end)
            SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                        {1, uint32_t((end - cursor) * CO * sizeof(T)), 0, 0, 0});
        Fence<HardEvent::MTE3_V>();
    }
}
// A complete 2x2x2 degree-one open grid uses eight fixed support matrices.
// Loading all three coordinates and all weights together removes the second
// coordinate DMA and runtime grid decoding. N remains metadata.
template <class T, uint32_t CI, uint32_t CO, bool BoundedInput = false>
__aicore__ inline void UnitCubeSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp,
                                          GM_ADDR tiling) {
    if (GetBlockIdx() != 0) {
        if constexpr ((CO * sizeof(T)) % kUbAlignmentBytes == 0) {
            if (GetBlockIdx() == 1)
                ZeroSingleEdgeComplement<T, CO>(ep, yp, tiling);
        }
        return;
    }
    constexpr bool partitionOutput = (CO * sizeof(T)) % kUbAlignmentBytes == 0;
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t prefixElements = kSingleOwnerFeaturePrefixElements;
    constexpr uint32_t prefixRows = prefixElements / CI;
    // Allocate controls, FP32 operands, then optional BF16 staging buffers.
    // CI is a power of two in [8,128], so prefixRows contains complete rows.
    uint32_t offsetBytes = kSparseControlBytes;
    auto feature = SparseRuntimeAlloc<float>(offsetBytes, prefixElements),
         weights = SparseRuntimeAlloc<float>(offsetBytes, 8 * matrixElements);
    offsetBytes += kResultPhasePaddingBytes;
    auto result = SparseRuntimeAlloc<float>(offsetBytes, matrixElements * SupportBatchWidth(matrixElements, 8));
    offsetBytes += kOperandPhasePaddingBytes;
    auto broadcast = SparseRuntimeAlloc<float>(offsetBytes, CI * (CO <= 32 ? CO : 8)),
         output = SparseRuntimeAlloc<float>(offsetBytes, kWholeOutputCapacityElements),
         messageSum = SparseRuntimeAlloc<float>(offsetBytes, CO);
    LocalTensor<T> rawFeature(TPosition::VECCALC, 0, 1), rawWeight(TPosition::VECCALC, 0, 1),
        rawOutput(TPosition::VECCALC, 0, 1);
    if constexpr (sizeof(T) != 4) {
        // BF16 input/output storage starts in a separate 64-KiB UB region.
        offsetBytes = (offsetBytes + kBf16StorageBoundaryBytes - 1) / kBf16StorageBoundaryBytes *
                      kBf16StorageBoundaryBytes;
        rawFeature = SparseRuntimeAlloc<T>(offsetBytes, prefixElements);
        rawWeight = SparseRuntimeAlloc<T>(offsetBytes, 8 * matrixElements);
        rawOutput = SparseRuntimeAlloc<T>(offsetBytes, kWholeOutputCapacityElements);
    }
    GlobalTensor<T> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<uint16_t> coordinateBits;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> metadataGm;
    x.SetGlobalBuffer((__gm__ T *)xp);
    weightsGm.SetGlobalBuffer((__gm__ T *)wp);
    coordsGm.SetGlobalBuffer((__gm__ T *)ap);
    coordinateBits.SetGlobalBuffer((__gm__ uint16_t *)ap);
    outputGm.SetGlobalBuffer((__gm__ T *)yp);
    edgesGm.SetGlobalBuffer((__gm__ int64_t *)ep);
    metadataGm.SetGlobalBuffer((__gm__ uint64_t *)tiling);
    // Bulk operand DMA overlaps independent immutable scalar control reads.
    if constexpr (sizeof(T) == 4)
        SplineCCopy(weights, weightsGm, 8 * matrixElements);
    else
        SplineCCopy(rawWeight, weightsGm, 8 * matrixElements);
    uint32_t scalarSource = 0;
    if constexpr (BoundedInput) {
        // The real source row is legal even when the fixed prefix does not fit.
        scalarSource = uint32_t(edgesGm.GetValue(0));
        StageFeature<T, CI>(x, scalarSource, feature, rawFeature);
    } else if constexpr (sizeof(T) == 4) {
        SplineCCopy(feature, x, prefixElements);
    } else {
        SplineCCopy(rawFeature, x, prefixElements);
    }
    // The fixed local zero capacity can initialize while the controls load.
    // Only nodeCount*CO elements are subsequently stored to the output tensor.
    if constexpr (!partitionOutput)
        SplineCDuplicate(output, 0.0f, uint64_t(64), uint8_t(64), 1, 8);
    // The packed word and endpoints use direct scalar GM reads.
    uint64_t packedMetadataWord = metadataGm.GetValue(0);
    uint32_t nodeCount = packedMetadataWord & kNodeCountMask,
             source = BoundedInput ? scalarSource : uint32_t(edgesGm.GetValue(0)),
             target = uint32_t(edgesGm.GetValue(1));
    // UnboundedInput has N*CI >= prefixElements. The actual endpoint is
    // arbitrary; sources beyond the prefix replace staging with their row.
    bool prefetched = !BoundedInput && source < prefixRows;
    if constexpr (!BoundedInput) {
        if (!prefetched) {
            // Complete the old prefix before replacing the same UB slot.
            Fence<HardEvent::MTE2_S>();
            StageFeature<T, CI>(x, source, feature, rawFeature);
        }
    }
    float u = PseudoValue(coordsGm, coordinateBits, 0), v = PseudoValue(coordsGm, coordinateBits, 1), z = PseudoValue(coordsGm, coordinateBits, 2);
    // The open 2x2x2 grid uses the original coordinates at all endpoints;
    // canonical support slots avoid dynamic permutations of zero terms.
    Fence<HardEvent::MTE2_V>();
    if constexpr (sizeof(T) != 4) {
        SplineCCast(weights, rawWeight, RoundMode::CAST_NONE, 8 * matrixElements);
        if (prefetched)
            SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, prefixElements);
        else
            SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
        PipeBarrier<PIPE_V>();
    }
    // Accumulate all eight support matrices in the original FP32 order.
    // A single incoming edge makes add and mean identical at its destination.
    IndexedTrilinear<CI, CO>(u, v, z, 0, 1, 2, 3, 4, 5, 6, 7,
                             feature[prefetched ? source * CI : 0], weights, result, broadcast, messageSum);
    if constexpr (CO <= 64)
        SplineCAdds(output[partitionOutput ? 0 : target * CO], result, 0.0f, uint64_t(CO), uint8_t(1), {1, 1, 8, 8});
    else
        SplineCAdds(output[partitionOutput ? 0 : target * CO], result, 0.0f, uint64_t(64), uint8_t(CO / 64), {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    FastStore(outputGm, partitionOutput ? target * CO : 0, output, rawOutput,
              partitionOutput ? CO : nodeCount * CO);
    PipeBarrier<PIPE_ALL>();
}
// A single edge has one output owner. Loading its complete bounded knot set
// avoids per-support DMA, and writing the whole output needs no cross-core sync.
// Graph size and knot sizes are metadata, never template dimensions.
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void GeneralSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp,
                                         GM_ADDR tiling) {
    if (GetBlockIdx() != 0) {
        if constexpr ((CO * sizeof(T)) % kUbAlignmentBytes == 0) {
            if (GetBlockIdx() == 1)
                ZeroSingleEdgeComplement<T, CO>(ep, yp, tiling);
        }
        return;
    }
    constexpr bool partitionOutput = (CO * sizeof(T)) % kUbAlignmentBytes == 0;
    constexpr uint32_t matrixElements = CI * CO;
    uint32_t offsetBytes = 2048;
    LocalTensor<int64_t> indices(TPosition::VECCALC, 0, 4);
    LocalTensor<T> coords(TPosition::VECCALC, 512, 32 / sizeof(T));
    LocalTensor<T> coordsZ(TPosition::VECCALC, 1536, 32 / sizeof(T));
    auto feature = SparseRuntimeAlloc<float>(offsetBytes, CI),
         weights = SparseRuntimeAlloc<float>(offsetBytes, kSingleOwnerWeightCapacityElements);
    offsetBytes += 256;
    auto result = SparseRuntimeAlloc<float>(offsetBytes, matrixElements * SupportBatchWidth(matrixElements, 8));
    offsetBytes += 32;
    auto broadcast = SparseRuntimeAlloc<float>(offsetBytes, CI * (CO <= 32 ? CO : 8)),
         output = SparseRuntimeAlloc<float>(offsetBytes, 4096),
         messageSum = SparseRuntimeAlloc<float>(offsetBytes, CO);
    LocalTensor<T> rawFeature(TPosition::VECCALC, 0, 1), rawWeight(TPosition::VECCALC, 0, 1),
        rawOutput(TPosition::VECCALC, 0, 1);
    if constexpr (sizeof(T) != 4) {
        offsetBytes = (offsetBytes + 65535) / 65536 * 65536;
        rawFeature = SparseRuntimeAlloc<T>(offsetBytes, CI);
        rawWeight = SparseRuntimeAlloc<T>(offsetBytes, kSingleOwnerWeightCapacityElements);
        rawOutput = SparseRuntimeAlloc<T>(offsetBytes, 4096);
    }
    GlobalTensor<T> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> metadataGm;
    x.SetGlobalBuffer((__gm__ T *)xp);
    weightsGm.SetGlobalBuffer((__gm__ T *)wp);
    coordsGm.SetGlobalBuffer((__gm__ T *)ap);
    outputGm.SetGlobalBuffer((__gm__ T *)yp);
    edgesGm.SetGlobalBuffer((__gm__ int64_t *)ep);
    metadataGm.SetGlobalBuffer((__gm__ uint64_t *)tiling);
    SplineCCopyPad(indices, edgesGm, {1, 16, 0, 0, 0}, {false, 0, 0, 0});
    SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    // Every supported degree-one knot grid contains at least four matrices.
    if constexpr (sizeof(T) == 4)
        SplineCCopy(weights, weightsGm, 4 * matrixElements);
    else
        SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
    // All independent control/operand/vector issues retain their order.
    // Read the same immutable packed word while those transfers can run;
    // EVENT_ID0 still publishes every endpoint/coordinate UB read below.
    uint64_t packedMetadataWord = metadataGm.GetValue(0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint64_t packed = packedMetadataWord;
    uint32_t nodeCount = packed & kNodeCountMask, knotSizeX = (packed >> kKnotSizeXShift) & kGridFieldMask,
             knotSizeY = (packed >> kKnotSizeYShift) & kGridFieldMask,
             scaleX = (packed >> kScaleXShift) & kGridFieldMask,
             scaleY = (packed >> kScaleYShift) & kGridFieldMask;
    bool threeDimensional = (packed >> kIs3dShift) & 1;
    uint32_t knotSizeZ = (packed >> kKnotSizeZShift) & kGridFieldMask,
             scaleZ = (packed >> kScaleZShift) & kGridFieldMask;
    uint32_t knotCount = knotSizeX * knotSizeY * (threeDimensional ? knotSizeZ : 1),
             source = SplineCGet(indices, 0), target = SplineCGet(indices, 1), outputElements = nodeCount * CO;
    if (threeDimensional) {
        SplineCCopyPad(coordsZ, coordsGm[2], {1, uint32_t(sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
    }
    if (knotCount > 4) {
        if constexpr (sizeof(T) == 4)
            SplineCCopy(weights[4 * matrixElements], weightsGm[4 * matrixElements],
                     (knotCount - 4) * matrixElements);
        else
            SplineCCopy(rawWeight[4 * matrixElements], weightsGm[4 * matrixElements],
                     (knotCount - 4) * matrixElements);
    }
    StageFeature<T, CI>(x, source, feature, rawFeature);
    if constexpr (!partitionOutput)
        SplineCDuplicate(output, 0.0f, outputElements);
    float scaledX = LocalPseudo(coords, 0) * scaleX, scaledY = LocalPseudo(coords, 1) * scaleY;
    int32_t baseX = int32_t(scaledX), baseY = int32_t(scaledY);
    float u = scaledX - baseX, v = scaledY - baseY;
    uint32_t x0 = baseX < knotSizeX ? baseX : baseX - knotSizeX,
             x1 = baseX + 1 < knotSizeX ? baseX + 1 : baseX + 1 - knotSizeX;
    uint32_t y0 = baseY < knotSizeY ? baseY : baseY - knotSizeY,
             y1 = baseY + 1 < knotSizeY ? baseY + 1 : baseY + 1 - knotSizeY;
    float z = 0.0f;
    uint32_t lowerPlane = 0, upperPlane = 0;
    if (threeDimensional) {
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        float scaledZ = LocalPseudo(coordsZ, 0) * scaleZ;
        int32_t baseZ = int32_t(scaledZ);
        z = scaledZ - baseZ;
        uint32_t z0 = baseZ < knotSizeZ ? baseZ : baseZ - knotSizeZ,
                 z1 = baseZ + 1 < knotSizeZ ? baseZ + 1 : baseZ + 1 - knotSizeZ;
        lowerPlane = z0 * knotSizeX * knotSizeY;
        upperPlane = z1 * knotSizeX * knotSizeY;
    }
    Fence<HardEvent::MTE2_V>();
    if constexpr (sizeof(T) != 4) {
        SplineCCast(weights, rawWeight, RoundMode::CAST_NONE, knotCount * matrixElements);
        SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
        PipeBarrier<PIPE_V>();
    }
    if (threeDimensional)
        IndexedTrilinear<CI, CO>(u, v, z, x0 + knotSizeX * y0 + lowerPlane, x1 + knotSizeX * y0 + lowerPlane,
                                 x0 + knotSizeX * y1 + lowerPlane, x1 + knotSizeX * y1 + lowerPlane,
                                 x0 + knotSizeX * y0 + upperPlane, x1 + knotSizeX * y0 + upperPlane,
                                 x0 + knotSizeX * y1 + upperPlane, x1 + knotSizeX * y1 + upperPlane, feature,
                                 weights, result, broadcast, messageSum);
    else
        IndexedBilinear<CI, CO>(u, v, x0 + knotSizeX * y0, x1 + knotSizeX * y0, x0 + knotSizeX * y1,
                                x1 + knotSizeX * y1, feature, weights, result, broadcast, messageSum);
    if constexpr (CO <= 64)
        SplineCAdds(output[partitionOutput ? 0 : target * CO], result, 0.0f, uint64_t(CO), uint8_t(1), {1, 1, 8, 8});
    else
        SplineCAdds(output[partitionOutput ? 0 : target * CO], result, 0.0f, uint64_t(64), uint8_t(CO / 64), {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    FastStore(outputGm, partitionOutput ? target * CO : 0, output, rawOutput,
              partitionOutput ? CO : outputElements);
    PipeBarrier<PIPE_ALL>();
}
template <uint32_t MODE, uint32_t DIM, uint32_t DEG, uint32_t CI, uint32_t CO, uint32_t ES, uint32_t NN, typename INPUT_T>
__global__ __aicore__ void spline_conv(GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr, GM_ADDR weight,
                                       GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    using DT_X = INPUT_T;
    REGISTER_TILING_DEFAULT(SplineConvTilingData);
    // Each launch establishes normal mode before any manual mask operation.
    SetMaskNorm();
    if constexpr (MODE == kBoundedUnitCubeSingleMode)
        UnitCubeSingleEdge<DT_X, CI, CO, true>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kUnitCubeSingleMode)
        UnitCubeSingleEdge<DT_X, CI, CO>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kGeneralSingleMode)
        GeneralSingleEdge<DT_X, CI, CO>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kCachedCoordinateSingleMode)
        CachedCoordinateSingleEdge<CI, CO, NN>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kCachedFloatSingleMode)
        CachedFloatSingleEdge<CI, CO, NN>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kPairedIncomingGraphMode)
        PairedIncomingGraph<DT_X, CI, CO>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kSmallBilinearSingleMode)
        RuntimeSingleEdge<DT_X, CI, CO, NN>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kSparseSeparateZeroMode)
        RuntimeSparseGraph<DT_X, CI, CO, false>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kSparseCoownedZeroMode)
        RuntimeSparseGraph<DT_X, CI, CO, true>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kCachedSparseSeparateZeroMode)
        RuntimeSparseGraph<DT_X, CI, CO, false, true>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kCachedSparseCoownedZeroMode)
        RuntimeSparseGraph<DT_X, CI, CO, true, true>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kStaticSparseMode) {
        if constexpr (ES == 8)
            RuntimeSparseGraph<DT_X, CI, CO, true, false, ES>(x, edge_index, edge_attr, weight, out, tiling);
        else if constexpr (ES != 0) {
            StaticEdgeGraph<DT_X, CI, CO, false, ES, true>(x, edge_index, edge_attr, weight, out, tiling);
        } else {
            GlobalTensor<uint64_t> metadata;
            metadata.SetGlobalBuffer((__gm__ uint64_t*)tiling);
            const uint32_t edgeCount = (metadata.GetValue(0) >> kSparseEdgeShift) & kSparseEdgeMask;
            StaticEdgeGraph<DT_X, CI, CO, false, 0, true>(x, edge_index, edge_attr, weight, out, tiling, edgeCount);
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(SplineConvTilingData, t, tiling);
        KernelSplineConv<DT_X, MODE, DIM, DEG> op;
        op.Run(x, edge_index, edge_attr, weight, out, workspace, t);
    }
}

// Basic vector operations need no library system workspace. The generated
// entry passes the raw user workspace through unchanged; only the owned-row
// fallback consumes it, using disjoint private edge-list regions per core.
// No MODE0 entry is compiled, so this covers every executable template key.
#if (!defined(ASCENDC_CPU_DEBUG) || (ASCENDC_CPU_DEBUG == 0)) && \
    (!defined(ASCENDC_DUMP) || (ASCENDC_DUMP == 0)) && \
    (!defined(ASCENDC_TIME_STAMP_ON) || (ASCENDC_TIME_STAMP_ON == 0))
namespace AscendC {
__aicore__ inline void SplineNoSystemWorkspace(GM_ADDR) {}
__aicore__ inline GM_ADDR SplineRawUserWorkspace(GM_ADDR workspace) { return workspace; }
} // namespace AscendC
#define SetSysWorkspaceForce SplineNoSystemWorkspace
#define GetUserWorkspace SplineRawUserWorkspace
#endif


// CANN inlines the user template into a separate AIV entry. Annotate that
// same entry's declaration so its instruction address is consistently aligned.
// The seven arguments and the entire generated computation remain unchanged.
// Other frontends, including Host metadata extraction, have no such entry macro.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201) && defined(auto_gen_spline_conv_kernel)
extern "C" __aicore__ __attribute__((cce_kernel, aligned(512)))
void auto_gen_spline_conv_kernel(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
#endif
