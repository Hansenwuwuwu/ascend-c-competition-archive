#include "kernel_operator.h"
#include "tiling_key_tree_line_min_distance.h"
#include "tree_line_min_distance_tiling.h"
using namespace AscendC;
template <HardEvent E> __aicore__ inline void Fence() {
  SetFlag<E>(0);
  WaitFlag<E>(0);
}
// Build device-local byte-offset sequences by doubling an aligned prefix.
__aicore__ inline void DeviceOffsets(LocalTensor<uint32_t> dst, uint32_t count, uint32_t stride) {
  Fence<HardEvent::V_S>();
  uint32_t seed = count < 8 ? count : 8;
  for (uint32_t i = 0; i < seed; ++i)
    dst.SetValue(i, i * stride);
  Fence<HardEvent::S_V>();
  for (uint32_t have = 8; have < count; have *= 2) {
    uint32_t take = count - have;
    if (take > have)
      take = have;
    Adds(dst[have].ReinterpretCast<int32_t>(), dst.ReinterpretCast<int32_t>(), int32_t(have * stride),
      take);
    PipeBarrier<PIPE_V>();
  }
}

// These two helpers initialize disjoint fresh scratch tables in one S->V
// handoff. They may not replace DeviceOffsets when a previous phase still
// reads or writes the same table. Expansion can overlap an unrelated input DMA.
__aicore__ inline void SeedDeviceOffsets(LocalTensor<uint32_t> destination, uint32_t elementCount,
                    uint32_t byteStride) {
  uint32_t seedElementCount = elementCount < 8 ? elementCount : 8;
  for (uint32_t element = 0; element < seedElementCount; ++element)
    destination.SetValue(element, element * byteStride);
}
__aicore__ inline void ExpandDeviceOffsetSeeds(LocalTensor<uint32_t> destination, uint32_t elementCount,
                       uint32_t byteStride) {
  for (uint32_t prefixElements = 8; prefixElements < elementCount; prefixElements *= 2) {
    uint32_t appendedElements = elementCount - prefixElements;
    if (appendedElements > prefixElements)
      appendedElements = prefixElements;
    Adds(destination[prefixElements].ReinterpretCast<int32_t>(), destination.ReinterpretCast<int32_t>(),
      int32_t(prefixElements * byteStride), appendedElements);
    PipeBarrier<PIPE_V>();
  }
}

// Device-side spatial ordering and conservative segment-box lower bounds.
template <typename T> class SpatialTreeLine {
  // The first 64 state elements hold the ends of these cells. Geometry uses
  // 64 point rows; a 32-point target balances cell bounds and padded chunks.
  static constexpr uint32_t maximumSpatialCellCount = 64;
  static constexpr uint32_t minimumSpatialCellPointCount = 32;
  static constexpr uint32_t coefficientPlaneCount = 7, coefficientLanesPerPlane = 64;
  static constexpr uint32_t coefficientOffsetElements = coefficientPlaneCount * coefficientLanesPerPlane;
  static constexpr uint32_t coefficientScratchElements = coefficientOffsetElements * 2;
  static constexpr uint32_t queryPairStateElements = 4 * 4160 + 2688;
  // Reuse the available UB budget for segment invariants.
  static constexpr uint32_t segmentStatePlaneCount = 22;
  LocalTensor<float> absoluteSegmentDirection[3];
  bool usePackedCoefficientGather;
  TPipe pipe;
  TBuf<TPosition::VECCALC> storage;
  GlobalTensor<T> pg, sg, og;
  GlobalTensor<float> coordG;
  GlobalTensor<int32_t> ig, orderG;
  LocalTensor<float> out, order, pool, raw, lo[3], hi[3], inv, recip, sorted, sortTmp, key, seq, red,
    group, perpendicularInverse[3];
  LocalTensor<float> x, y, z, dot, tmp, dist, c[8], best, broadcast;
  LocalTensor<int32_t> outIdx;
  LocalTensor<float> bestArg;
  uint32_t m, n, start, batch, mp, k, live, cellShift, cellCount, rankCount;
  LocalTensor<int32_t> cellEnd;
  float scaleBound, firstScore;
  bool keepPoints, outputPending;
  uint32_t pointPitch;
  LocalTensor<float> localPoints;
  LocalTensor<int32_t> localOrder;
  LocalTensor<uint32_t> pointOffsets, coefficientSourceByteOffsets, coefficientPlaneByteOffsets;
  __aicore__ inline uint32_t Spread(uint32_t v) {
    v = (v | (v << 16)) & 0x030000ffU;
    v = (v | (v << 8)) & 0x0300f00fU;
    v = (v | (v << 4)) & 0x030c30c3U;
    v = (v | (v << 2)) & 0x09249249U;
    return v;
  }
  __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<T> src, uint32_t count,
                LocalTensor<float> halfScratch) {
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(dst, src, {1, count * 4, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    } else {
      auto h = halfScratch.ReinterpretCast<half>();
      DataCopyPad(h, src, {1, count * 2, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(dst, h, RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
    }
  }
  __aicore__ inline void PointSort() {
    auto keys = pool, idxf = pool[k], sortedPoints = pool[k * 2], scratch = pool[k * 4],
      praw = pool[k * 6];
    Load(praw, pg[start * 3], n * 3, scratch);
    ReduceMin(keys, praw, scratch, n * 3, false);
    PipeBarrier<PIPE_V>();
    ReduceMax(keys[8], praw, scratch, n * 3, false);
    Fence<HardEvent::V_S>();
    float low = keys.GetValue(0), high = keys.GetValue(8),
       factor = high > low ? 1023.0f / (high - low) : 0.0f;
    Fence<HardEvent::S_V>();
    DeviceOffsets(idxf.ReinterpretCast<uint32_t>(), n, 12);
    auto code = keys.ReinterpretCast<uint32_t>(), bits = sortedPoints[k].ReinterpretCast<uint32_t>();
    auto shifted = scratch.ReinterpretCast<uint32_t>(),
      bitsMask = scratch[k].ReinterpretCast<uint32_t>();
    Duplicate(code, uint32_t(0), k);
    PipeBarrier<PIPE_V>();
    for (uint32_t axis = 0; axis < 3; ++axis) {
      Gather(sortedPoints, praw, idxf.ReinterpretCast<uint32_t>(), axis * 4, n);
      PipeBarrier<PIPE_V>();
      Adds(sortedPoints, sortedPoints, -low, n);
      PipeBarrier<PIPE_V>();
      Muls(sortedPoints, sortedPoints, factor, n);
      PipeBarrier<PIPE_V>();
      Maxs(sortedPoints, sortedPoints, 0.0f, n);
      PipeBarrier<PIPE_V>();
      Mins(sortedPoints, sortedPoints, 1023.0f, n);
      PipeBarrier<PIPE_V>();
      Cast(bits.ReinterpretCast<int32_t>(), sortedPoints, RoundMode::CAST_FLOOR, n);
      PipeBarrier<PIPE_V>();
      const uint32_t shifts[4] = {16, 8, 4, 2},
             masks[4] = {0x030000ffU, 0x0300f00fU, 0x030c30c3U, 0x09249249U};
      for (uint32_t phase = 0; phase < 4; ++phase) {
        ShiftLeft(shifted, bits, shifts[phase], n);
        Duplicate(bitsMask, masks[phase], n);
        PipeBarrier<PIPE_V>();
        Or(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
         shifted.ReinterpretCast<uint16_t>(), n * 2);
        PipeBarrier<PIPE_V>();
        And(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
          bitsMask.ReinterpretCast<uint16_t>(), n * 2);
        PipeBarrier<PIPE_V>();
      }
      if (axis) {
        ShiftLeft(bits, bits, axis, n);
        PipeBarrier<PIPE_V>();
      }
      Or(code.ReinterpretCast<uint16_t>(), code.ReinterpretCast<uint16_t>(),
       bits.ReinterpretCast<uint16_t>(), n * 2);
      PipeBarrier<PIPE_V>();
    }
    Adds(code.ReinterpretCast<int32_t>(), code.ReinterpretCast<int32_t>(), int32_t(0x00800000), n);
    PipeBarrier<PIPE_V>();
    uint32_t aligned = (n + 7) / 8 * 8;
    if (aligned < k)
      Duplicate(keys[aligned], -3.402823466e38f, k - aligned);
    Fence<HardEvent::V_S>();
    for (uint32_t i = n; i < aligned; ++i)
      keys.SetValue(i, -3.402823466e38f);
    Fence<HardEvent::S_V>();
    DeviceOffsets(idxf.ReinterpretCast<uint32_t>(), k, 1);
    Sort<float, true>(sortedPoints, keys, idxf.ReinterpretCast<uint32_t>(), scratch, k / 32);
    PipeBarrier<PIPE_V>();
    Extract(keys, order.ReinterpretCast<uint32_t>(), sortedPoints, k / 32);
    PipeBarrier<PIPE_V>();
    if (keepPoints) {
      DataCopy(localOrder, order.ReinterpretCast<int32_t>(), k);
      PipeBarrier<PIPE_V>();
    } else {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(orderG[start], order.ReinterpretCast<int32_t>(), {1, n * 4, 0, 0, 0});
      Fence<HardEvent::MTE3_V>();
    }
    Muls(idxf.ReinterpretCast<int32_t>(), order.ReinterpretCast<int32_t>(), 12, n);
    PipeBarrier<PIPE_V>();
    for (uint32_t a = 0; a < 3; ++a) {
      Gather(scratch, praw, idxf.ReinterpretCast<uint32_t>(), a * 4, n);
      if (keepPoints) {
        Adds(localPoints[a * pointPitch].ReinterpretCast<int32_t>(),
          scratch.ReinterpretCast<int32_t>(), int32_t(0), n);
        PipeBarrier<PIPE_V>();
      } else {
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(coordG[start * 3 + a * n], scratch, {1, n * 4, 0, 0, 0});
        Fence<HardEvent::MTE3_V>();
      }
    }
    Fence<HardEvent::V_S>();
    cellCount = 0;
    auto bitsKey = keys.ReinterpretCast<uint32_t>();
    auto endTmp = out.ReinterpretCast<int32_t>();
    for (uint32_t at = 0; at < n;) {
      uint32_t region = (bitsKey.GetValue(at) - 0x00800000U) >> cellShift, low = at + 1, high = n;
      while (low < high) {
        uint32_t mid = (low + high) / 2;
        if (((bitsKey.GetValue(mid) - 0x00800000U) >> cellShift) == region)
          low = mid + 1;
        else
          high = mid;
      }
      endTmp.SetValue(cellCount++, int32_t(low));
      at = low;
    }
    for (uint32_t i = cellCount; i < (cellCount + 7) / 8 * 8; ++i)
      endTmp.SetValue(i, int32_t(n));
    Fence<HardEvent::S_V>();
  }
  __aicore__ inline void Segments() {
    for (int a = 0; a < 3; ++a) {
      lo[a] = pool[a * mp];
      hi[a] = pool[(a + 3) * mp];
    }
    inv = pool[6 * mp];
    raw = pool[7 * mp];
    // A, D and the reciprocal occupy seven adjacent source planes.
    // The compacted index list and bound keys stay disjoint from them.
    recip = pool[13 * mp];
    sorted = pool[14 * mp];
    key = pool[15 * mp];
    seq = pool[16 * mp];
    perpendicularInverse[0] = inv;
    perpendicularInverse[1] = pool[17 * mp];
    perpendicularInverse[2] = pool[18 * mp];
    for (uint32_t axis = 0; axis < 3; ++axis)
      absoluteSegmentDirection[axis] = pool[(19 + axis) * mp];
    sortTmp = pool[segmentStatePlaneCount * mp];
    auto work = pool[segmentStatePlaneCount * mp + 64];
    x = work;
    y = work[4160];
    z = work[8320];
    dot = work[12480];
    // Rank uses a short temporary slice in the dot plane. Chunk reuses x
    // after its first product and dot after the projection has been consumed.
    tmp = dot[mp];
    dist = dot;
    for (int axis = 0; axis < 8; ++axis)
      c[axis] = work[16640 + axis * 64];
    red = work[17152];
    group = work[17408];
    best = work[17600];
    bestArg = work[17664];
    broadcast = work[17792];
    auto input = work, endpoint = work[9 * mp], square = work[10 * mp];
    Duplicate(input, 0.0f, 6 * mp);
    PipeBarrier<PIPE_V>();
    Load(input, sg[batch * m * 6], m * 6, work[6 * mp]);
    ReduceMin(red, input, square, m * 6, false);
    PipeBarrier<PIPE_V>();
    ReduceMax(red[8], input, square, m * 6, false);
    Fence<HardEvent::V_S>();
    float av = red.GetValue(0), bv = red.GetValue(8);
    scaleBound = av * av > bv * bv ? av * av : bv * bv;
    Fence<HardEvent::S_V>();
    // Indices remain integer bits through compaction, selection and workspace.
    // Float views below serve bit-preserving data-movement APIs only.
    DeviceOffsets(seq.ReinterpretCast<uint32_t>(), mp, 1);
    Muls(key.ReinterpretCast<int32_t>(), seq.ReinterpretCast<int32_t>(), int32_t(24), mp);
    auto offsets = key.ReinterpretCast<uint32_t>();
    for (uint32_t axis = 0; axis < 3; ++axis) {
      Gather(raw[axis * mp], input, offsets, axis * 4, mp);
      Gather(endpoint, input, offsets, (axis + 3) * 4, mp);
      Min(lo[axis], raw[axis * mp], endpoint, mp);
      Max(hi[axis], raw[axis * mp], endpoint, mp);
      Sub(raw[(axis + 3) * mp], endpoint, raw[axis * mp], mp);
      Mul(square, raw[(axis + 3) * mp], raw[(axis + 3) * mp], mp);
      if (axis == 0)
        Adds(inv, square, 0.0f, mp);
      else
        Add(inv, inv, square, mp);
    }
    auto mask = red.ReinterpretCast<uint8_t>();
    CompareScalar(mask, inv, 0.0f, CMPMODE::GT, mp);
    Select(inv, mask, inv, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp);
    Duplicate(recip, 1.0f, mp);
    Div(recip, recip, inv, mp);
    // The three cross-product axes are perpendicular to the segment.
    // A floor only weakens the distance bound and avoids division by zero
    // for a parallel or point segment; it never changes exact geometry.
    if (m >= kSatMinimumSegmentCount) {
      for (uint32_t axis = 0; axis < 3; ++axis)
        Abs(absoluteSegmentDirection[axis], raw[(axis + 3) * mp], mp);
      for (uint32_t axis = 0; axis < 3; ++axis) {
        uint32_t other = (axis + 1) % 3, last = (axis + 2) % 3;
        Mul(endpoint, raw[(other + 3) * mp], raw[(other + 3) * mp], mp);
        Mul(square, raw[(last + 3) * mp], raw[(last + 3) * mp], mp);
        Add(endpoint, endpoint, square, mp);
        Maxs(endpoint, endpoint, 1.0e-20f, mp);
        Duplicate(square, 1.0f, mp);
        Div(perpendicularInverse[axis], square, endpoint, mp);
      }
    }
    pointOffsets = work[17728].ReinterpretCast<uint32_t>();
    if (keepPoints)
      DeviceOffsets(pointOffsets, 64, 4);
    // A short candidate chunk overwrites only its valid coefficient lanes.
    // Define every inactive lane once so later full-width geometry and
    // scratch reuse never consume uninitialized coefficient memory.
    Duplicate(c[0], 0.0f, 7 * 64);
    if (usePackedCoefficientGather) {
      coefficientSourceByteOffsets = work[queryPairStateElements].ReinterpretCast<uint32_t>();
      coefficientPlaneByteOffsets =
        work[queryPairStateElements + coefficientOffsetElements].ReinterpretCast<uint32_t>();
      // Define inactive offsets too; the masked gather only consumes live
      // columns, and later chunks can safely retain their previous offsets.
      Duplicate(coefficientSourceByteOffsets, uint32_t(0), coefficientOffsetElements);
      for (uint32_t plane = 0; plane < coefficientPlaneCount; ++plane)
        Duplicate(coefficientPlaneByteOffsets[plane * coefficientLanesPerPlane],
             plane * mp * uint32_t(sizeof(float)), coefficientLanesPerPlane);
    }
  }
  __aicore__ inline void Rank(uint32_t at) {
    float mn[3], mx[3];
    Duplicate(group, 0.0f, 192);
    if (keepPoints) {
      for (uint32_t a = 0; a < 3; ++a)
        Gather(group[a * 64], localPoints[a * pointPitch], pointOffsets, at * 4, live);
    } else {
      Fence<HardEvent::V_MTE2>();
      for (uint32_t a = 0; a < 3; ++a)
        DataCopyPad(group[a * 64], coordG[start * 3 + a * n + at], {1, live * 4, 0, 0, 0},
              {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    }
    for (uint32_t a = 0; a < 3; ++a) {
      WholeReduceMin(red[a * 16], group[a * 64], int32_t(live), 1, 1, 1, 8);
      WholeReduceMax(red[a * 16 + 8], group[a * 64], int32_t(live), 1, 1, 1, 8);
    }
    Fence<HardEvent::V_S>();
    for (uint32_t a = 0; a < 3; ++a) {
      mn[a] = red.GetValue(a * 16);
      mx[a] = red.GetValue(a * 16 + 8);
    }
    Fence<HardEvent::S_V>();
    if (outputPending) {
      WaitFlag<HardEvent::MTE3_V>(0);
      outputPending = false;
    }
    Brcb(broadcast, group, uint8_t((live + 7) / 8), {1, 8});
    Brcb(broadcast[512], group[64], uint8_t((live + 7) / 8), {1, 8});
    Brcb(broadcast[1024], group[128], uint8_t((live + 7) / 8), {1, 8});
    for (uint32_t a = 0; a < 3; ++a) {
      Adds(x, lo[a], -mx[a], rankCount);
      Adds(y, hi[a], -mn[a], rankCount);
      Maxs(x, x, 0.0f, rankCount);
      Mins(y, y, 0.0f, rankCount);
      Add(x, x, y, rankCount);
      if (a == 0)
        Mul(key, x, x, rankCount);
      else
        MulAddDst(key, x, x, rankCount);
    }
    if (m >= kSatMinimumSegmentCount) {
      // Segment-box SAT axes: coordinate axes above, then D cross each
      // coordinate axis. Every plane gap divided by its normal length is a
      // lower bound on every point-to-segment distance in this point box.
      LocalTensor<float> delta[3] = {x, y, z};
      for (uint32_t axis = 0; axis < 3; ++axis)
        Adds(delta[axis], raw[axis * mp], -(mn[axis] + mx[axis]) * 0.5f, rankCount);
      auto support = x[mp], absoluteDirection = x[mp * 2];
      for (uint32_t axis = 0; axis < 3; ++axis) {
        uint32_t other = (axis + 1) % 3, last = (axis + 2) % 3;
        Mul(dot, delta[other], raw[(last + 3) * mp], rankCount);
        Mul(tmp, delta[last], raw[(other + 3) * mp], rankCount);
        Sub(dot, dot, tmp, rankCount);
        Abs(dot, dot, rankCount);
        Muls(support, absoluteSegmentDirection[last], (mx[other] - mn[other]) * 0.5f,
             rankCount);
        Axpy(support, absoluteSegmentDirection[other], (mx[last] - mn[last]) * 0.5f,
             rankCount);
        Sub(dot, dot, support, rankCount);
        Maxs(dot, dot, 0.0f, rankCount);
        Mul(dot, dot, dot, rankCount);
        Mul(dot, dot, perpendicularInverse[axis], rankCount);
        Max(key, key, dot, rankCount);
      }
    } else {
      // Project the point box onto a separating plane of each segment.
      auto norm = x[mp], support = x[mp * 2], aScratch = x[mp * 3];
      LocalTensor<float> delta[3] = {x, y, z};
      for (uint32_t axis = 0; axis < 3; ++axis) {
        Adds(delta[axis], raw[axis * mp], -(mn[axis] + mx[axis]) * 0.5f, rankCount);
        if (axis == 0)
          Mul(dot, delta[axis], raw[(axis + 3) * mp], rankCount);
        else
          MulAddDst(dot, delta[axis], raw[(axis + 3) * mp], rankCount);
      }
      Mul(dot, dot, recip, rankCount);
      Maxs(dot, dot, -1.0f, rankCount);
      Mins(dot, dot, 0.0f, rankCount);
      Muls(dot, dot, -1.0f, rankCount);
      // Only conservative candidate bounds use fused arithmetic; exact Q remains unchanged.
      for (uint32_t axis = 0; axis < 3; ++axis) {
        MulAddDst(delta[axis], raw[(axis + 3) * mp], dot, rankCount);
        if (axis == 0)
          Mul(norm, delta[axis], delta[axis], rankCount);
        else
          MulAddDst(norm, delta[axis], delta[axis], rankCount);
        Abs(tmp, delta[axis], rankCount);
        if (axis == 0)
          Muls(support, tmp, (mx[axis] - mn[axis]) * 0.5f, rankCount);
        else
          Axpy(support, tmp, (mx[axis] - mn[axis]) * 0.5f, rankCount);
      }
      Sub(dot, norm, support, rankCount);
      Maxs(dot, dot, 0.0f, rankCount);
      Mul(dot, dot, dot, rankCount);
      auto nonzero = red[128].ReinterpretCast<uint8_t>();
      CompareScalar(nonzero, norm, 0.0f, CMPMODE::GT, rankCount);
      Select(tmp, nonzero, norm, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, rankCount);
      Div(dot, dot, tmp, rankCount);
      Max(key, key, dot, rankCount);
    }
    Muls(key, key, -1.0f, rankCount);
    uint32_t aligned = (m + 7) / 8 * 8;
    if (aligned < mp)
      Duplicate(key[aligned], -3.402823466e38f, mp - aligned);
    Fence<HardEvent::V_S>();
    for (uint32_t i = m; i < aligned; ++i)
      key.SetValue(i, -3.402823466e38f);
    Fence<HardEvent::S_V>();
  }
  __aicore__ inline uint32_t Compress(float threshold, bool second) {
    auto maskA = red[128].ReinterpretCast<uint8_t>(), maskB = red[192].ReinterpretCast<uint8_t>();
    if (second) {
      CompareScalar(maskA, key, -threshold, CMPMODE::GE, mp);
      CompareScalar(maskB, key, firstScore, CMPMODE::LT, mp);
      And(maskA.ReinterpretCast<uint16_t>(), maskA.ReinterpretCast<uint16_t>(),
        maskB.ReinterpretCast<uint16_t>(), mp / 16);
    } else {
      CompareScalar(maskA, key, threshold, CMPMODE::EQ, mp);
    }
    uint64_t count = 0;
    GatherMask(sorted, seq, maskA.ReinterpretCast<uint32_t>(), true, mp, {1, 1, 8, 1}, count);
    return uint32_t(count);
  }
  __aicore__ inline void BroadcastMul(LocalTensor<float> dst, LocalTensor<float> a,
                    LocalTensor<float> b) {
    Mul(dst, a, b, uint64_t(64), live, {1, 1, 1, 8, 8, 0});
  }
  __aicore__ inline void UpdateClosestRows(bool initializeBest) {
    // Every live row is overwritten below. The first 64 scratch
    // lanes were all defined by geometry (live>=1); inactive row values
    // never enter the live-row bound reduction or the output DMA.
    // ORDER_VALUE_INDEX stores tightly packed (float, uint32) pairs.
    // Fixed patterns 1/2 select their even/odd elements without indexed
    // loads or a reserved-count read; both outputs contain live rows.
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(live * 2);
    GatherMask(x, red, uint8_t(1), {1, 1, 8, 1});
    GatherMask(y, red, uint8_t(2), {1, 1, 8, 1});
    SetMaskNorm();
    ResetMask();
    ShiftLeft(y.ReinterpretCast<uint32_t>(), y.ReinterpretCast<uint32_t>(), uint32_t(2), 64);
    Gather(z, c[7], y.ReinterpretCast<uint32_t>(), 0, live);
    if (initializeBest) {
      // The seed/fallback path always supplies a nonempty list. Its first chunk
      // defines both best arrays before refinement can read them.
      // Geometry defines all first 64 scratch lanes even for live=1;
      // inactive point rows never enter the live output or bound reduction.
      Adds(best.ReinterpretCast<int32_t>(), x.ReinterpretCast<int32_t>(), int32_t(0), 64);
      Adds(bestArg.ReinterpretCast<int32_t>(), z.ReinterpretCast<int32_t>(), int32_t(0), 64);
    } else {
      auto lt = red[128].ReinterpretCast<uint8_t>(), eq = red[160].ReinterpretCast<uint8_t>();
      Compare(lt, x, best, CMPMODE::LT, 64);
      Compare(eq, x, best, CMPMODE::EQ, 64);
      Min(y.ReinterpretCast<int32_t>(), z.ReinterpretCast<int32_t>(),
        bestArg.ReinterpretCast<int32_t>(), 64);
      Select(bestArg, lt, z, bestArg, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
      Min(best, best, x, 64);
      Select(bestArg, eq, y, bestArg, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
    }
    tmp = dot[mp];
  }
  __aicore__ inline void Chunk(uint32_t pos, uint32_t count, bool initializeBest = false) {
    tmp = x;
    // The compacted list stays read-only until both geometry passes finish.
    // Alias this chunk instead of copying and converting its 64 indices.
    c[7] = sorted[pos];
    Muls(tmp.ReinterpretCast<int32_t>(), c[7].ReinterpretCast<int32_t>(), 4, count);
    if (usePackedCoefficientGather) {
      // Each row adds one source plane's byte origin to the same indices.
      Add(coefficientSourceByteOffsets.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(),
        coefficientPlaneByteOffsets.ReinterpretCast<int32_t>(), uint64_t(count),
        uint8_t(coefficientPlaneCount), {1, 1, 1, 8, 0, 8});
      Gather(c[0], raw, coefficientSourceByteOffsets, 0, uint64_t(count),
         uint8_t(coefficientPlaneCount), uint16_t(coefficientLanesPerPlane / 8));
      ResetMask();
    } else {
      // Preserve on-chip points when the extra offset tables would
      // otherwise displace them. The original gathers need no scratch.
      auto offsets = tmp.ReinterpretCast<uint32_t>();
      for (uint32_t axis = 0; axis < 6; ++axis)
        Gather(c[axis], raw[axis * mp], offsets, 0, count);
      Gather(c[6], recip, offsets, 0, count);
    }

    Sub(x, c[0], broadcast, uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(y, c[1], broadcast[512], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(z, c[2], broadcast[1024], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    BroadcastMul(dot, x, c[3]);
    BroadcastMul(tmp, y, c[4]);
    uint32_t q = live * 64;
    Add(dot, dot, tmp, q);
    BroadcastMul(tmp, z, c[5]);
    Add(dot, dot, tmp, q);
    BroadcastMul(dot, dot, c[6]);
    Maxs(dot, dot, -1.0f, q);
    Mins(dot, dot, 0.0f, q);
    BroadcastMul(x, dot, c[4]);
    Sub(y, y, x, q);
    BroadcastMul(x, dot, c[5]);
    Sub(z, z, x, q);
    BroadcastMul(x, dot, c[3]);
    Sub(dot, c[0], broadcast, uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(x, dot, x, q);
    Mul(dist, x, x, q);
    if constexpr (IsSameType<T, half>::value) {
      // Inputs were converted to Float; accumulation retains Float range.
      MulAddDst(dist, y, y, q);
      MulAddDst(dist, z, z, q);
    } else {
      Mul(x, y, y, q);
      Add(dist, dist, x, q);
      Mul(x, z, z, q);
      Add(dist, dist, x, q);
    }
    WholeReduceMin(red, dist, int32_t(count), live, 1, 1, 8);
    UpdateClosestRows(initializeBest);
  }
  template <uint32_t segmentLanes>
  __aicore__ inline void PackedCandidateChunk(uint32_t pos, uint32_t count, bool initializeBest) {
    static_assert(segmentLanes == 8 || segmentLanes == 16 || segmentLanes == 32);
    // Replicate a short index row before the single coefficient gather.
    // Every padded index is a legal zero index; min masks exclude padded
    // candidates. Integer Copy preserves the byte offsets exactly.
    tmp = x;
    c[7] = sorted[pos];
    Duplicate(tmp.ReinterpretCast<int32_t>(), int32_t(0), 64);
    Muls(tmp.ReinterpretCast<int32_t>(), c[7].ReinterpretCast<int32_t>(), int32_t(4), count);
    Copy(tmp[segmentLanes].ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(),
      uint64_t(segmentLanes), uint8_t(64 / segmentLanes - 1), {1, 1, segmentLanes / 8, 0});
    Add(coefficientSourceByteOffsets.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(),
      coefficientPlaneByteOffsets.ReinterpretCast<int32_t>(), uint64_t(64),
      uint8_t(coefficientPlaneCount), {1, 1, 1, 8, 0, 8});
    Gather(c[0], raw, coefficientSourceByteOffsets, 0, coefficientOffsetElements);
    constexpr uint32_t pointRowsPerVector = 64 / segmentLanes;
    uint32_t vectorRepeats = (live + pointRowsPerVector - 1) / pointRowsPerVector;
    uint32_t activeLanes = vectorRepeats * 64;
    auto bx = broadcast, by = broadcast[512], bz = broadcast[1024];
    if constexpr (segmentLanes > 8) {
      // Geometry occupies at most 2048 floats in each 4160-float plane.
      // The upper half stages expanded point rows until all residuals finish.
      bx = x[2048];
      by = y[2048];
      bz = z[2048];
      uint32_t paddedPointRows = vectorRepeats * pointRowsPerVector;
      Copy(bx, broadcast, uint64_t(segmentLanes), uint8_t(paddedPointRows),
        {1, 0, segmentLanes / 8, 1});
      Copy(by, broadcast[512], uint64_t(segmentLanes), uint8_t(paddedPointRows),
        {1, 0, segmentLanes / 8, 1});
      Copy(bz, broadcast[1024], uint64_t(segmentLanes), uint8_t(paddedPointRows),
        {1, 0, segmentLanes / 8, 1});
    }
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams sharedCoefficient{1, 1, 1, 8, 8, 0};
    Sub(x, c[0], bx, uint64_t(64), vectorRepeats, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], by, uint64_t(64), vectorRepeats, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], bz, uint64_t(64), vectorRepeats, {1, 1, 1, 8, 0, 8});
    Mul(dot, x, c[3], uint64_t(64), vectorRepeats, sharedCoefficient);
    Mul(x, y, c[4], uint64_t(64), vectorRepeats, sharedCoefficient);
    Add(dot, dot, x, activeLanes);
    Mul(x, z, c[5], uint64_t(64), vectorRepeats, sharedCoefficient);
    Add(dot, dot, x, activeLanes);
    Mul(dot, dot, c[6], uint64_t(64), vectorRepeats, sharedCoefficient);
    Maxs(dot, dot, -1.0f, activeLanes);
    Mins(dot, dot, 0.0f, activeLanes);
    Mul(x, dot, c[4], uint64_t(64), vectorRepeats, sharedCoefficient);
    Sub(y, y, x, activeLanes);
    Mul(x, dot, c[5], uint64_t(64), vectorRepeats, sharedCoefficient);
    Sub(z, z, x, activeLanes);
    Mul(x, dot, c[3], uint64_t(64), vectorRepeats, sharedCoefficient);
    Sub(dot, c[0], bx, uint64_t(64), vectorRepeats, {1, 1, 1, 8, 0, 8});
    Sub(x, dot, x, activeLanes);
    Mul(dist, x, x, activeLanes);
    if constexpr (IsSameType<T, half>::value) {
      // Inputs were converted to Float; accumulation retains Float range.
      MulAddDst(dist, y, y, activeLanes);
      MulAddDst(dist, z, z, activeLanes);
    } else {
      Mul(x, y, y, activeLanes);
      Add(dist, dist, x, activeLanes);
      Mul(x, z, z, activeLanes);
      Add(dist, dist, x, activeLanes);
    }
    WholeReduceMin(red, dist, int32_t(count), live, 1, 1, segmentLanes / 8);
    UpdateClosestRows(initializeBest);
  }
  __aicore__ inline void ComputeCandidateChunk(uint32_t pos, uint32_t count,
                        bool initializeBest = false) {
    // The capacity fallback retains the original full-width geometry.
    if (!usePackedCoefficientGather || count > 32)
      Chunk(pos, count, initializeBest);
    else if (count <= 8)
      PackedCandidateChunk<8>(pos, count, initializeBest);
    else if (count <= 16)
      PackedCandidateChunk<16>(pos, count, initializeBest);
    else
      PackedCandidateChunk<32>(pos, count, initializeBest);
  }
  __aicore__ inline void Restore() {
    PipeBarrier<PIPE_ALL>();
    Fence<HardEvent::V_S>();
    Fence<HardEvent::MTE3_S>();
    auto keys = pool, rank = pool[k], sortOut = pool[k * 2], scratch = pool[k * 4];
    if (keepPoints) {
      Adds(keys.ReinterpretCast<int32_t>(), localOrder, int32_t(0), n);
      PipeBarrier<PIPE_V>();
    } else {
      Fence<HardEvent::V_MTE2>();
      DataCopyPad(keys.ReinterpretCast<int32_t>(), orderG[start], {1, n * 4, 0, 0, 0},
            {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    }
    Cast(keys, keys.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, n);
    Muls(keys, keys, -1.0f, n);
    uint32_t aligned = (n + 7) / 8 * 8;
    if (aligned < k)
      Duplicate(keys[aligned], -3.402823466e38f, k - aligned);
    Fence<HardEvent::V_S>();
    for (uint32_t i = n; i < aligned; ++i)
      keys.SetValue(i, -3.402823466e38f);
    Fence<HardEvent::S_V>();
    DeviceOffsets(rank.ReinterpretCast<uint32_t>(), k, 1);
    Sort<float, true>(sortOut, keys, rank.ReinterpretCast<uint32_t>(), scratch, k / 32);
    Extract(keys, rank.ReinterpretCast<uint32_t>(), sortOut, k / 32);
    Muls(rank.ReinterpretCast<int32_t>(), rank.ReinterpretCast<int32_t>(), 4, n);
    Fence<HardEvent::V_MTE2>();
    DataCopyPad(sortOut, coordG[start * 3], {1, n * 4, 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(sortOut[k], coordG[start * 3 + n], {1, n * 4, 0, 0, 0}, {false, 0, 0, 0});
    Fence<HardEvent::MTE2_V>();
    Gather(out, sortOut, rank.ReinterpretCast<uint32_t>(), 0, n);
    Gather(outIdx.ReinterpretCast<float>(), sortOut[k], rank.ReinterpretCast<uint32_t>(), 0, n);
    // Workspace stores raw int32 bits; Gather already produced the output type.
  }

 public:
  __aicore__ inline void Run(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx, GM_ADDR workspace,
               const TreeLineMinDistanceTilingData &t) {
    outputPending = false;
    coordG.SetGlobalBuffer((__gm__ float *)workspace);
    orderG.SetGlobalBuffer((__gm__ int32_t *)workspace + t.batchCount * t.pointsPerBatch * 3);
    pg.SetGlobalBuffer((__gm__ T *)p);
    sg.SetGlobalBuffer((__gm__ T *)s);
    og.SetGlobalBuffer((__gm__ T *)o);
    ig.SetGlobalBuffer((__gm__ int32_t *)idx);
    batch = GetBlockIdx() / t.workersPerBatch;
    uint32_t worker = GetBlockIdx() % t.workersPerBatch;
    uint32_t begin = t.pointsPerBatch * worker / t.workersPerBatch,
        end = t.pointsPerBatch * (worker + 1) / t.workersPerBatch;
    n = end - begin;
    start = batch * t.pointsPerBatch + begin;
    m = t.segmentCount;
    k = 32;
    while (k < n)
      k *= 2;
    mp = 32;
    while (mp < m)
      mp *= 2;
    rankCount = (m + 63) / 64 * 64;
    pipe.InitBuffer(storage, t.unifiedBufferBytes - 8192);
    auto mem = storage.Get<float>();
    uint32_t pitch = (n + 15) / 16 * 16;
    out = mem;
    outIdx = mem[pitch].ReinterpretCast<int32_t>();
    order = mem[pitch * 2];
    pool = mem[pitch * 2 + k];
    pointPitch = pitch;
    uint32_t stateSize = 3 * pitch + k, userSize = (t.unifiedBufferBytes - 8192) / 4;
    uint32_t initialSize = 2 * pitch + 7 * k + 3 * n,
        querySize = 64 + segmentStatePlaneCount * mp + 64 + queryPairStateElements;
    keepPoints = stateSize < userSize && initialSize <= userSize - stateSize &&
          querySize <= userSize - stateSize;
    usePackedCoefficientGather =
      querySize + coefficientScratchElements <= userSize &&
      (!keepPoints || querySize + coefficientScratchElements <= userSize - stateSize);
    if (keepPoints) {
      localPoints = mem[userSize - stateSize];
      localOrder = localPoints[3 * pitch].ReinterpretCast<int32_t>();
    }
    cellShift = 30;
    // Each additional Morton prefix bit bisects a cell. Growing by one bit
    // avoids an eightfold jump in bound/reduction setup work while targeting
    // at least thirty-two points per cell and at most sixty-four cells.
    for (uint32_t spatialCellCount = 1; spatialCellCount < maximumSpatialCellCount &&
                      spatialCellCount * 2 <= n / minimumSpatialCellPointCount;
      spatialCellCount *= 2)
      --cellShift;
    PointSort();
    PipeBarrier<PIPE_ALL>();
    Fence<HardEvent::V_S>();
    Fence<HardEvent::MTE3_S>();
    cellEnd = mem.ReinterpretCast<int32_t>();
    pool = mem[64];
    Segments();
    Fence<HardEvent::V_S>();
    uint32_t cell = 0;
    for (uint32_t at = 0; at < n;) {
      uint32_t limit = uint32_t(cellEnd.GetValue(cell));
      live = limit - at;
      if (live > 64)
        live = 64;
      Rank(at);
      firstScore = 0;
      uint32_t initial = Compress(firstScore, false);
      if (!initial) {
        ReduceMax(red, key, red[64], mp, false);
        Fence<HardEvent::V_S>();
        firstScore = red.GetValue(0);
        Fence<HardEvent::S_V>();
        initial = Compress(firstScore, false);
      }
      if (!initial) {
        // Exceptional bound arithmetic still evaluates every segment;
        // best state is always defined by an actual nonempty chunk.
        DeviceOffsets(sorted.ReinterpretCast<uint32_t>(), m, 1);
        initial = m;
      }
      for (uint32_t pos = 0; pos < initial; pos += 64) {
        uint32_t count = initial - pos;
        if (count > 64)
          count = 64;
        ComputeCandidateChunk(pos, count, pos == 0);
      }
      WholeReduceMax(red, best, int32_t(live), 1, 1, 1, 8);
      Fence<HardEvent::V_S>();
      float bound = red.GetValue(0);
      Fence<HardEvent::S_V>();
      uint32_t remaining = Compress(bound + 0.000032f * (scaleBound + bound + 1.0f), true);
      for (uint32_t pos = 0; pos < remaining; pos += 64) {
        uint32_t count = remaining - pos;
        if (count > 64)
          count = 64;
        ComputeCandidateChunk(pos, count);
      }
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(coordG[start * 3 + at], best, {1, live * 4, 0, 0, 0});
      DataCopyPad(coordG[start * 3 + n + at], bestArg, {1, live * 4, 0, 0, 0});
      SetFlag<HardEvent::MTE3_V>(0);
      outputPending = true;
      at += live;
      if (at == limit)
        ++cell;
    }
    if (outputPending) {
      WaitFlag<HardEvent::MTE3_V>(0);
      outputPending = false;
    }
    pool = mem[pitch * 2];
    Restore();
    PipeBarrier<PIPE_V>();
    Sqrt(out, out, n);
    if constexpr (IsSameType<T, float>::value) {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(og[start], out, {1, n * 4, 0, 0, 0});
    } else {
      auto h = out.ReinterpretCast<half>();
      Cast(h, out, RoundMode::CAST_RINT, n);
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(og[start], h, {1, n * 2, 0, 0, 0});
    }
    DataCopyPad(ig[start], outIdx, {1, n * 4, 0, 0, 0});
    Fence<HardEvent::MTE3_S>();
  }
};

// A single query point needs no point-tile loop or staged output DMA.
template <typename T, uint32_t ALGORITHM> class KernelTreeLineMinDistance {
  static constexpr uint32_t STATIC_WIDTH =
    ALGORITHM >= 24 ? 8U * (ALGORITHM - 23)
            : (ALGORITHM >= 13 && ALGORITHM <= 16 ? (128U << (ALGORITHM - 13)) : 0);
  uint32_t bufferBase = 0;
  LocalTensor<float> outLocal;
  uint32_t outStart;
  LocalTensor<int32_t> idxLocal;
  GlobalTensor<T> points, segments, output;
  GlobalTensor<int32_t> indicesOut;
  TreeLineMinDistanceTilingData t;
  uint32_t width, rows, pairs;
  LocalTensor<float> raw, ax, ay, az, dx, dy, dz, len, x, y, z, dot, tmp, dist, red;
  LocalTensor<uint32_t> gather;
  __aicore__ inline void Store(uint32_t i, float d, int32_t arg) {
    outLocal.SetValue(i - outStart, d);
    idxLocal.SetValue(i - outStart, arg);
  }
  __aicore__ inline void Flush(uint32_t count) {
    Fence<HardEvent::S_V>();
    Sqrt(outLocal, outLocal, count);
    if constexpr (IsSameType<T, float>::value) {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(output[outStart], outLocal, {1, count * 4, 0, 0, 0});
    } else {
      auto halfout = outLocal.ReinterpretCast<half>();
      Cast(halfout, outLocal, RoundMode::CAST_RINT, count);
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(output[outStart], halfout, {1, count * 2, 0, 0, 0});
    }
    DataCopyPad(indicesOut[outStart], idxLocal, {1, count * 4, 0, 0, 0});
    Fence<HardEvent::MTE3_S>();
  }
  __aicore__ inline void Scalar(uint32_t start, uint32_t end) {
    for (uint32_t i = start; i < end; ++i) {
      uint32_t batch = i / t.pointsPerBatch;
      float px = float(points.GetValue(i * 3)), py = float(points.GetValue(i * 3 + 1)),
         pz = float(points.GetValue(i * 3 + 2));
      float best = 3.402823466e38f;
      int32_t arg = 0;
      for (uint32_t j = 0; j < t.segmentCount; ++j) {
        uint32_t k = (batch * t.segmentCount + j) * 6;
        float a = float(segments.GetValue(k)), b = float(segments.GetValue(k + 1)),
           c = float(segments.GetValue(k + 2));
        float u = float(segments.GetValue(k + 3)) - a, v = float(segments.GetValue(k + 4)) - b,
           w = float(segments.GetValue(k + 5)) - c;
        float l = (u * u + v * v) + w * w, q = 0;
        if (l > 0)
          q = (((px - a) * u + (py - b) * v) + (pz - c) * w) / l;
        q = q < 0 ? 0 : (q > 1 ? 1 : q);
        float ex = px - (a + q * u), ey = py - (b + q * v), ez = pz - (c + q * w);
        float d = (ex * ex + ey * ey) + ez * ez;
        if (d < best) {
          best = d;
          arg = j;
        }
      }
      Store(i, best, arg);
    }
  }
  __aicore__ inline void LoadSegments(uint32_t batch) {
    Duplicate(raw, 0.0f, width * 6);
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(raw, segments[batch * t.segmentCount * 6], {1, t.segmentCount * 24, 0, 0, 0},
            {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    } else {
      auto halfraw = raw.ReinterpretCast<half>()[width * 12];
      DataCopyPad(halfraw, segments[batch * t.segmentCount * 6], {1, t.segmentCount * 12, 0, 0, 0},
            {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(raw, halfraw, RoundMode::CAST_NONE, t.segmentCount * 6);
    }
    Gather(ax, raw, gather, 0, width);
    Gather(ay, raw, gather, 4, width);
    Gather(az, raw, gather, 8, width);
    Gather(dx, raw, gather, 12, width);
    Gather(dy, raw, gather, 16, width);
    Gather(dz, raw, gather, 20, width);
    Sub(dx, dx, ax, width);
    Sub(dy, dy, ay, width);
    Sub(dz, dz, az, width);
    Mul(len, dx, dx, width);
    Mul(tmp, dy, dy, width);
    Mul(dot, dz, dz, width);
    Add(len, len, tmp, width);
    Add(len, len, dot, width);
    // Repair exactly zero nonnegative squared lengths without a vector select.
    Mins(tmp.ReinterpretCast<int32_t>(), len.ReinterpretCast<int32_t>(), int32_t(1), width);
    Muls(tmp.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(-0x3f800000), width);
    Adds(tmp.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(0x3f800000), width);
    Or(len.ReinterpretCast<uint16_t>(), len.ReinterpretCast<uint16_t>(),
     tmp.ReinterpretCast<uint16_t>(), width * 2);
    // A batch shares segment lengths across every point tile.
    Duplicate(tmp, 1.0f, width);
    Div(len, tmp, len, width);
  }
  __aicore__ inline void PointVector(uint32_t start, uint32_t end) {
    uint32_t w = (end - start + 63) / 64 * 64;
    if (w > 2048)
      w = 2048;
    LocalTensor<float> mem(TPosition::VECCALC, bufferBase,
               (t.unifiedBufferBytes - 8192 - bufferBase) / 4);
    raw = mem;
    x = mem[6 * w];
    y = mem[7 * w];
    z = mem[8 * w];
    ax = mem[9 * w];
    ay = mem[10 * w];
    az = mem[11 * w];
    dot = mem[12 * w];
    tmp = mem[13 * w];
    dist = mem[14 * w];
    auto best = mem[15 * w];
    auto arg = mem[16 * w];
    auto next = mem[17 * w];
    gather = mem[18 * w].template ReinterpretCast<uint32_t>();
    auto mask = mem[19 * w].template ReinterpretCast<uint8_t>();
    auto segmentLocal = mem[19 * w + 256];
    DeviceOffsets(gather, w, 12);
    uint32_t oldBatch = 0xffffffff;
    for (uint32_t i = start; i < end;) {
      uint32_t batch = i / t.pointsPerBatch, count = end - i;
      if (count > w)
        count = w;
      // All points in a batch share the same short segment table.
      if (t.segmentCount != 1 && batch != oldBatch) {
        Fence<HardEvent::V_MTE2>();
        if constexpr (IsSameType<T, float>::value) {
          DataCopyPad(segmentLocal, segments[batch * t.segmentCount * 6],
                {1, t.segmentCount * 24, 0, 0, 0}, {false, 0, 0, 0});
          Fence<HardEvent::MTE2_S>();
        } else {
          auto halfSegments = segmentLocal[64].ReinterpretCast<half>();
          DataCopyPad(halfSegments, segments[batch * t.segmentCount * 6],
                {1, t.segmentCount * 12, 0, 0, 0}, {false, 0, 0, 0});
          Fence<HardEvent::MTE2_V>();
          Cast(segmentLocal, halfSegments, RoundMode::CAST_NONE, t.segmentCount * 6);
          Fence<HardEvent::V_S>();
        }
        oldBatch = batch;
      }
      uint32_t remain = (batch + 1) * t.pointsPerBatch - i;
      if (count > remain)
        count = remain;
      uint32_t live = count;
      count = (count + 63) / 64 * 64;
      Duplicate(raw, 0.0f, w * 3);
      Fence<HardEvent::V_MTE2>();
      if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(raw, points[i * 3], {1, live * 12, 0, 0, 0}, {false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
      } else {
        auto hr = raw.ReinterpretCast<half>()[w * 6];
        DataCopyPad(hr, points[i * 3], {1, live * 6, 0, 0, 0}, {false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        Cast(raw, hr, RoundMode::CAST_NONE, live * 3);
      }
      Gather(x, raw, gather, 0, live);
      Gather(y, raw, gather, 4, live);
      Gather(z, raw, gather, 8, live);
      if (t.segmentCount != 1) {
        Duplicate(best, 3.402823466e38f, count);
        Duplicate(arg, 0.0f, count);
      }
      for (uint32_t j = 0; j < t.segmentCount; ++j) {
        float a, b, c, u, v, q;
        if (t.segmentCount == 1) {
          uint32_t k = batch * 6;
          a = float(segments.GetValue(k));
          b = float(segments.GetValue(k + 1));
          c = float(segments.GetValue(k + 2));
          u = float(segments.GetValue(k + 3)) - a;
          v = float(segments.GetValue(k + 4)) - b;
          q = float(segments.GetValue(k + 5)) - c;
        } else {
          uint32_t k = j * 6;
          a = segmentLocal.GetValue(k);
          b = segmentLocal.GetValue(k + 1);
          c = segmentLocal.GetValue(k + 2);
          u = segmentLocal.GetValue(k + 3) - a;
          v = segmentLocal.GetValue(k + 4) - b;
          q = segmentLocal.GetValue(k + 5) - c;
        }
        float l = (u * u + v * v) + q * q;
        Adds(ax, x, -a, count);
        Adds(ay, y, -b, count);
        Adds(az, z, -c, count);
        if (t.segmentCount == 1) {
          Muls(dot, ax, u, count);
          Axpy(dot, ay, v, count);
          Axpy(dot, az, q, count);
          // Segment invariants use the same vector arithmetic as the dot product.
          Duplicate(next, u, 64);
          Mul(next, next, next, 64);
          Duplicate(tmp, v, 64);
          Axpy(next, tmp, v, 64);
          Duplicate(tmp, q, 64);
          Axpy(next, tmp, q, 64);
          if (l == 0.0f)
            Duplicate(next, 1.0f, 64);
          Duplicate(tmp, 1.0f, 64);
          Div(tmp, tmp, next, 64);
          Duplicate(raw[64], 1.0f, 64);
          Muls(dist, tmp, -1.0f, 64);
          MulAddDst(raw[64], dist, next, 64);
          MulAddDst(tmp, tmp, raw[64], 64);
          Copy(raw, next, uint64_t(64), uint8_t(count / 64), {1, 1, 8, 0});
          Copy(next, tmp, uint64_t(64), uint8_t(count / 64), {1, 1, 8, 0});
          Compare(mask, dot, raw, CMPMODE::LT, count);
          Mul(dot, dot, next, count);
          Select(dot, mask, dot, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
          Maxs(dot, dot, 0.0f, count);
          Mins(dot, dot, 1.0f, count);
          Axpy(ax, dot, -u, count);
          Axpy(ay, dot, -v, count);
          Axpy(az, dot, -q, count);
          Mul(dist, ax, ax, count);
          MulAddDst(dist, ay, ay, count);
          MulAddDst(dist, az, az, count);
        } else {
          Muls(dot, ax, u, count);
          Muls(tmp, ay, v, count);
          Add(dot, dot, tmp, count);
          Muls(tmp, az, q, count);
          Add(dot, dot, tmp, count);
          Duplicate(next, u, count);
          Mul(next, next, next, count);
          Duplicate(tmp, v, count);
          Mul(tmp, tmp, tmp, count);
          Add(next, next, tmp, count);
          Duplicate(tmp, q, count);
          Mul(tmp, tmp, tmp, count);
          Add(next, next, tmp, count);
          if (l == 0.0f)
            Duplicate(next, 1.0f, count);
          Div(tmp, dot, next, count);
          Muls(dist, tmp, -1.0f, count);
          MulAddDst(dot, dist, next, count);
          Div(dot, dot, next, count);
          Add(dot, tmp, dot, count);
          Maxs(dot, dot, 0.0f, count);
          Mins(dot, dot, 1.0f, count);
          Muls(ax, dot, u, count);
          Muls(ay, dot, v, count);
          Muls(az, dot, q, count);
          Adds(ax, ax, a, count);
          Adds(ay, ay, b, count);
          Adds(az, az, c, count);
          Sub(ax, x, ax, count);
          Sub(ay, y, ay, count);
          Sub(az, z, az, count);
          Mul(ax, ax, ax, count);
          Mul(ay, ay, ay, count);
          Mul(az, az, az, count);
          Add(dist, ax, ay, count);
          Add(dist, dist, az, count);
        }
        if (t.segmentCount == 1)
          best = dist;
        else {
          Compare(mask, dist, best, CMPMODE::LT, count);
          Duplicate(next, float(int32_t(j)), count);
          Select(best, mask, dist, best, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
          Select(arg, mask, next, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        }
      }
      Sqrt(best, best, count);
      if (t.segmentCount == 1)
        Duplicate(next.template ReinterpretCast<int32_t>(), int32_t(0), count);
      else
        Cast(next.template ReinterpretCast<int32_t>(), arg, RoundMode::CAST_RINT, count);
      if constexpr (IsSameType<T, float>::value) {
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(output[i], best, {1, live * 4, 0, 0, 0});
      } else {
        auto halfout = best.ReinterpretCast<half>();
        Cast(halfout, best, RoundMode::CAST_RINT, count);
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(output[i], halfout, {1, live * 2, 0, 0, 0});
      }
      DataCopyPad(indicesOut[i], next.template ReinterpretCast<int32_t>(), {1, live * 4, 0, 0, 0});
      Fence<HardEvent::MTE3_V>();
      i += live;
    }
  }
  __aicore__ inline void Product(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b,
                 uint32_t live) {
    uint32_t ew = STATIC_WIDTH ? STATIC_WIDTH : width;
    if (ew <= 64 || live >= ew / 64) {
#pragma unroll
      for (uint32_t c = 0; c < ew; c += 64) {
        uint32_t mask = ew < 64 ? ew : 64;
        Mul(dst[c], a[c], b[c], uint64_t(mask), live,
          {1, 1, 1, uint8_t(ew / 8), uint8_t(ew / 8), 0});
      }
    } else {
      for (uint32_t r = 0; r < live; ++r)
        Mul(dst[r * ew], a[r * ew], b, ew);
    }
  }
  __aicore__ inline void Delta(LocalTensor<float> dst, LocalTensor<float> axis, uint32_t i, uint32_t live,
                uint32_t coordinate) {
    for (uint32_t r = 0; r < live; ++r) {
      float p = float(points.GetValue((i + r) * 3 + coordinate));
      Adds(dst[r * width], axis, -p, width);
    }
  }
  __aicore__ inline void Vector(uint32_t start, uint32_t end) {
    if constexpr (ALGORITHM >= 13)
      width = STATIC_WIDTH;
    else {
      width = (t.segmentCount + 7) / 8 * 8;
      if (width > 64) {
        width = 128;
        while (width < t.segmentCount)
          width *= 2;
      }
    }
    uint32_t outBytes = ((end - start + 15) / 16 * 16) * 8;
    uint32_t available = t.unifiedBufferBytes - outBytes - width * 32 - 14592;
    rows = available / (width * 20);
    if (rows > 64)
      rows = 64;
    if (rows < 1)
      rows = 1;
    if (rows > end - start)
      rows = end - start;
    if (rows > t.pointsPerBatch)
      rows = t.pointsPerBatch;
    if (width < 64)
      rows = (rows + 7) / 8 * 8;
    pairs = rows * width;
    uint32_t planePitch = pairs + 64, work = 5 * planePitch;
    if (work < 9 * width)
      work = 9 * width;
    LocalTensor<float> mem(TPosition::VECCALC, bufferBase,
               (t.unifiedBufferBytes - 8192 - bufferBase) / 4);
    ax = mem;
    ay = mem[width];
    az = mem[width * 2];
    dx = mem[width * 3];
    dy = mem[width * 4];
    dz = mem[width * 5];
    len = mem[width * 6];
    gather = mem[width * 7].template ReinterpretCast<uint32_t>();
    auto v = mem[width * 8];
    raw = v;
    x = v;
    y = v[planePitch];
    z = v[planePitch * 2];
    dot = v[planePitch * 3];
    tmp = v[planePitch * 4];
    dist = tmp;
    red = v[work];
    DeviceOffsets(gather, width, 24);
    if (rows >= 8)
      DeviceOffsets(red[704].ReinterpretCast<uint32_t>(), 64, 12);
    if (rows >= 16)
      DeviceOffsets(red[1024].ReinterpretCast<uint32_t>(), 64, 8);
    if (rows >= 16 && width > 64) {
      uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2;
      DeviceOffsets(red[1088].ReinterpretCast<uint32_t>(), 64, chunks * spacing * 4);
    }
    uint32_t oldBatch = 0xffffffff;
    for (uint32_t i = start; i < end;) {
      uint32_t batch = i / t.pointsPerBatch;
      if (batch != oldBatch) {
        LoadSegments(batch);
        oldBatch = batch;
      }
      uint32_t live = end - i;
      if (live > rows)
        live = rows;
      uint32_t batchEnd = (batch + 1) * t.pointsPerBatch;
      if (live > batchEnd - i)
        live = batchEnd - i;
      uint32_t count = live * width;
      if (live >= 8) {
        Fence<HardEvent::V_MTE2>();
        if constexpr (IsSameType<T, float>::value) {
          DataCopyPad(red[512], points[i * 3], {1, live * 12, 0, 0, 0}, {false, 0, 0, 0});
          Fence<HardEvent::MTE2_V>();
        } else {
          auto h = red[768].ReinterpretCast<half>();
          DataCopyPad(h, points[i * 3], {1, live * 6, 0, 0, 0}, {false, 0, 0, 0});
          Fence<HardEvent::MTE2_V>();
          Cast(red[512], h, RoundMode::CAST_NONE, live * 3);
        }
        Duplicate(red[800], 0.0f, 192);
        for (uint32_t axis = 0; axis < 3; ++axis)
          Gather(red[800 + axis * 64], red[512], red[704].ReinterpretCast<uint32_t>(), axis * 4,
             live);
        uint32_t block = (live + 7) / 8 * 64;
        if (width < 24) {
          LocalTensor<float> target[3] = {x, y, z}, source[3] = {ax, ay, az};
          for (uint32_t axis = 0; axis < 3; ++axis) {
            Brcb(tmp, red[800 + axis * 64], uint8_t((live + 7) / 8), {1, 8});
            PipeBarrier<PIPE_V>();
            Sub(target[axis], source[axis], tmp, uint64_t(width), live,
              {1, 1, 0, uint8_t(width / 8), 0, 1});
            PipeBarrier<PIPE_V>();
          }
        } else {
          for (uint32_t axis = 0; axis < 3; ++axis)
            Brcb(tmp[axis * block], red[800 + axis * 64], uint8_t((live + 7) / 8), {1, 8});
          for (uint32_t c = 0; c < width; c += 64) {
            uint32_t mask = width < 64 ? width : 64;
            Sub(x[c], ax[c], tmp, uint64_t(mask), live, {1, 1, 0, uint8_t(width / 8), 0, 1});
            Sub(y[c], ay[c], tmp[block], uint64_t(mask), live,
              {1, 1, 0, uint8_t(width / 8), 0, 1});
            Sub(z[c], az[c], tmp[block * 2], uint64_t(mask), live,
              {1, 1, 0, uint8_t(width / 8), 0, 1});
          }
        }
      } else {
        for (uint32_t r = 0; r < live; ++r) {
          float px = float(points.GetValue((i + r) * 3)),
             py = float(points.GetValue((i + r) * 3 + 1)),
             pz = float(points.GetValue((i + r) * 3 + 2));
          Adds(x[r * width], ax, -px, width);
          Adds(y[r * width], ay, -py, width);
          Adds(z[r * width], az, -pz, width);
        }
      }

      Product(dot, x, dx, live);
      Product(tmp, y, dy, live);
      Add(dot, dot, tmp, count);
      Product(tmp, z, dz, live);
      Add(dot, dot, tmp, count);
      Product(dot, dot, len, live);
      Maxs(dot, dot, -1.0f, count);
      Mins(dot, dot, 0.0f, count);
      Product(tmp, dot, dx, live);
      Sub(x, x, tmp, count);
      Product(tmp, dot, dy, live);
      Sub(y, y, tmp, count);
      Product(tmp, dot, dz, live);
      Sub(z, z, tmp, count);
      Mul(dist, x, x, count);
      MulAddDst(dist, y, y, count);
      MulAddDst(dist, z, z, count);
      if (width <= 64) {
        WholeReduceMin(red, dist, int32_t(t.segmentCount), live, 1, 1, width / 8);
      } else {
        uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2, pitch = chunks * spacing;
        auto partial = red[128];
        WholeReduceMin(partial, dist, 64, live * chunks, spacing / 2, 1, 8);
        uint32_t tail = t.segmentCount % 64;
        if (tail) {
          WholeReduceMin(partial[(t.segmentCount / 64) * spacing],
                 dist[(t.segmentCount / 64) * 64], int32_t(tail), live, pitch / 2, 1,
                 width / 8);
        }
        uint64_t mask[2] = {0, 0};
        for (uint32_t c = 0; c < (t.segmentCount + 63) / 64; ++c)
          mask[0] |= uint64_t(1) << (c * spacing);
        WholeReduceMin(red, partial, mask, live, 1, 1, pitch / 8);
      }
      uint32_t offset = i - outStart;
      if (rows >= 16 && (offset & 7) == 0) {
        Gather(outLocal[offset], red, red[1024].ReinterpretCast<uint32_t>(), 0, live);
        auto dst = idxLocal[offset];
        Gather(dst.ReinterpretCast<float>(), red, red[1024].ReinterpretCast<uint32_t>(), 4, live);
        if (width > 64) {
          uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2;
          auto addr = red[1216].ReinterpretCast<int32_t>();
          Muls(addr, dst, int32_t(4), live);
          Add(addr, addr, red[1088].ReinterpretCast<int32_t>(), live);
          Adds(addr, addr, int32_t(128 * 4 + 4), live);
          Gather(red[1152], red, addr.ReinterpretCast<uint32_t>(), 0, live);
          ShiftRight(dst, dst, int32_t(spacing == 4 ? 2 : 1), live);
          ShiftLeft(dst, dst, int32_t(6), live);
          Add(dst, dst, red[1152].ReinterpretCast<int32_t>(), live);
        }
      } else {
        Fence<HardEvent::V_S>();
        auto ir = red.template ReinterpretCast<int32_t>();
        for (uint32_t r = 0; r < live; ++r) {
          int32_t arg = ir.GetValue(r * 2 + 1);
          if (width > 64) {
            uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2,
                pitch = chunks * spacing;
            arg = (arg / spacing) * 64 + ir.GetValue(128 + r * pitch + arg + 1);
          }
          Store(i + r, red.GetValue(r * 2), arg);
        }
        Fence<HardEvent::S_V>();
      }
      i += live;
    }
  }

 public:
  __aicore__ inline void Init(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx,
                const TreeLineMinDistanceTilingData &tiling) {
    t = tiling;
    points.SetGlobalBuffer((__gm__ T *)p);
    segments.SetGlobalBuffer((__gm__ T *)s);
    output.SetGlobalBuffer((__gm__ T *)o);
    indicesOut.SetGlobalBuffer((__gm__ int32_t *)idx);
  }
  __aicore__ inline void Process() {
    uint32_t total = t.batchCount * t.pointsPerBatch, start = total * GetBlockIdx() / t.workerCount,
        end = total * (GetBlockIdx() + 1) / t.workerCount;
    if constexpr (ALGORITHM == 3) {
      PointVector(start, end);
      return;
    }
    uint32_t pitch = (end - start + 15) / 16 * 16;
    outStart = start;
    outLocal = LocalTensor<float>(TPosition::VECCALC, 0, pitch);
    idxLocal = LocalTensor<int32_t>(TPosition::VECCALC, pitch * 4, pitch);
    bufferBase = pitch * 8;
    if constexpr (ALGORITHM == 1)
      Scalar(start, end);
    else
      Vector(start, end);
    Flush(end - start);
  }
};
// An axis-parallel finite segment projects by clamping that coordinate.
// This also covers a point segment, without a divide or a rounded reconstructed endpoint.
__aicore__ inline float AxisIntervalResidual(float point, float begin, float end) {
  float low = begin < end ? begin : end, high = begin > end ? begin : end;
  return point < low ? point - low : (point > high ? point - high : 0.0f);
}
__aicore__ inline float PointSegmentSquaredDistance(float px, float py, float pz, float a, float b, float c,
                          float endX, float endY, float endZ) {
  float u = endX - a, v = endY - b, w = endZ - c;
  float ex = px - a, ey = py - b, ez = pz - c;
  if (v == 0.0f && w == 0.0f) {
    ex = AxisIntervalResidual(px, a, endX);
  } else if (u == 0.0f && w == 0.0f) {
    ey = AxisIntervalResidual(py, b, endY);
  } else if (u == 0.0f && v == 0.0f) {
    ez = AxisIntervalResidual(pz, c, endZ);
  } else {
    float squaredLength = (u * u + v * v) + w * w;
    float quotient = 0.0f;
    if (squaredLength > 0.0f)
      quotient = ((ex * u + ey * v) + ez * w) / squaredLength;
    quotient = quotient < 0.0f ? 0.0f : (quotient > 1.0f ? 1.0f : quotient);
    ex = px - (a + quotient * u);
    ey = py - (b + quotient * v);
    ez = pz - (c + quotient * w);
  }
  return (ex * ex + ey * ey) + ez * ez;
}
template <typename T>
__aicore__ inline void CachedSingleSegmentTiny(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx,
                       const TreeLineMinDistanceTilingData &t) {
  constexpr uint32_t maximumSlicePoints = 32, maximumScalarSegments = 32;
  constexpr uint32_t elementBytes = sizeof(T), indexBytes = sizeof(int32_t);
  constexpr uint32_t pointBufferBytes = maximumSlicePoints * 3 * sizeof(float);
  constexpr uint32_t segmentBufferBytes = maximumScalarSegments * 6 * sizeof(float);
  constexpr uint32_t distanceBufferBytes = maximumSlicePoints * sizeof(float);
  GlobalTensor<T> points, segments, output;
  GlobalTensor<int32_t> indicesOut;
  points.SetGlobalBuffer((__gm__ T *)p);
  segments.SetGlobalBuffer((__gm__ T *)s);
  output.SetGlobalBuffer((__gm__ T *)o);
  indicesOut.SetGlobalBuffer((__gm__ int32_t *)idx);
  uint32_t totalPoints = t.batchCount * t.pointsPerBatch;
  uint32_t begin = totalPoints * GetBlockIdx() / t.workerCount;
  uint32_t end = totalPoints * (GetBlockIdx() + 1) / t.workerCount;
  uint32_t slicePoints = end - begin;
  LocalTensor<T> pointValues(TPosition::VECCALC, 0, maximumSlicePoints * 3);
  LocalTensor<T> segmentValues(TPosition::VECCALC, pointBufferBytes, maximumScalarSegments * 6);
  LocalTensor<T> distanceValues(TPosition::VECCALC, pointBufferBytes + segmentBufferBytes,
                 maximumSlicePoints);
  LocalTensor<int32_t> indexValues(TPosition::VECCALC,
                  pointBufferBytes + segmentBufferBytes + distanceBufferBytes,
                  maximumSlicePoints);
  DataCopyPad(pointValues, points[begin * 3], {1, slicePoints * 3 * elementBytes, 0, 0, 0},
        {false, 0, 0, 0});
  uint32_t loadedBatch = uint32_t(-1);
  for (uint32_t point = begin; point < end; ++point) {
    uint32_t batch = point / t.pointsPerBatch, localPoint = point - begin;
    if (batch != loadedBatch) {
      // Only this batch reuses the table. The fence also protects a later
      // reload when a nonaligned worker slice crosses a batch boundary.
      Fence<HardEvent::S_MTE2>();
      DataCopyPad(segmentValues, segments[batch * t.segmentCount * 6],
            {1, t.segmentCount * 6 * elementBytes, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_S>();
      loadedBatch = batch;
    }
    float px = float(pointValues.GetValue(localPoint * 3));
    float py = float(pointValues.GetValue(localPoint * 3 + 1));
    float pz = float(pointValues.GetValue(localPoint * 3 + 2));
    float best = 3.402823466e38f;
    int32_t arg = 0;
    for (uint32_t segment = 0; segment < t.segmentCount; ++segment) {
      uint32_t coordinate = segment * 6;
      float a = float(segmentValues.GetValue(coordinate));
      float b = float(segmentValues.GetValue(coordinate + 1));
      float c = float(segmentValues.GetValue(coordinate + 2));
      float endX = float(segmentValues.GetValue(coordinate + 3));
      float endY = float(segmentValues.GetValue(coordinate + 4));
      float endZ = float(segmentValues.GetValue(coordinate + 5));
      float squaredDistance = PointSegmentSquaredDistance(px, py, pz, a, b, c, endX, endY, endZ);
      if (segment == 0 || squaredDistance < best) {
        best = squaredDistance;
        arg = int32_t(segment);
      }
    }
    distanceValues.SetValue(localPoint, T(sqrt(best)));
    indexValues.SetValue(localPoint, arg);
  }
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(output[begin], distanceValues, {1, slicePoints * elementBytes, 0, 0, 0});
  DataCopyPad(indicesOut[begin], indexValues, {1, slicePoints * indexBytes, 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}
template <typename T>
__aicore__ inline void ScalarDmaTiny(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx,
                  const TreeLineMinDistanceTilingData &t) {
  uint32_t i = GetBlockIdx(), batch = i / t.pointsPerBatch;
  GlobalTensor<T> points, segments, output;
  GlobalTensor<int32_t> indicesOut;
  points.SetGlobalBuffer((__gm__ T *)p);
  segments.SetGlobalBuffer((__gm__ T *)s);
  output.SetGlobalBuffer((__gm__ T *)o);
  indicesOut.SetGlobalBuffer((__gm__ int32_t *)idx);
  // Two independent DMA requests stage the whole bounded scalar workload.
  // Scalar reads then use UB rather than serial GM cache-line misses.
  constexpr uint32_t segmentBufferBytes = kMaximumScalarPairCount * 6 * sizeof(T);
  constexpr uint32_t outputBaseBytes = 32 + segmentBufferBytes;
  LocalTensor<T> pointValues(TPosition::VECCALC, 0, 32 / sizeof(T));
  LocalTensor<T> segmentValues(TPosition::VECCALC, 32, kMaximumScalarPairCount * 6);
  DataCopyPad(pointValues, points[i * 3], {1, uint32_t(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  DataCopyPad(segmentValues, segments[batch * t.segmentCount * 6],
        {1, t.segmentCount * uint32_t(6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  Fence<HardEvent::MTE2_S>();
  float px = float(pointValues.GetValue(0)), py = float(pointValues.GetValue(1)),
     pz = float(pointValues.GetValue(2));
  float best = 3.402823466e38f;
  int32_t arg = 0;
  for (uint32_t j = 0; j < t.segmentCount; ++j) {
    uint32_t k = j * 6;
    float a = float(segmentValues.GetValue(k)), b = float(segmentValues.GetValue(k + 1)),
       c = float(segmentValues.GetValue(k + 2));
    float endX = float(segmentValues.GetValue(k + 3)), endY = float(segmentValues.GetValue(k + 4)),
       endZ = float(segmentValues.GetValue(k + 5));
    float d = PointSegmentSquaredDistance(px, py, pz, a, b, c, endX, endY, endZ);
    if (j == 0 || d < best) {
      best = d;
      arg = int32_t(j);
    }
  }
  LocalTensor<T> dist(TPosition::VECCALC, outputBaseBytes, 32 / sizeof(T));
  LocalTensor<int32_t> ind(TPosition::VECCALC, outputBaseBytes + 32, 8);
  dist.SetValue(0, T(sqrt(best)));
  ind.SetValue(0, arg);
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(output[i], dist, {1, sizeof(T), 0, 0, 0});
  DataCopyPad(indicesOut[i], ind, {1, 4, 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}
template <typename T, uint32_t M>
__aicore__ inline void FixedPoint(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx) {
  if (GetBlockIdx() != 0)
    return;
  GlobalTensor<T> points, segments, output;
  GlobalTensor<int32_t> indicesOut;
  points.SetGlobalBuffer((__gm__ T *)p);
  segments.SetGlobalBuffer((__gm__ T *)s);
  output.SetGlobalBuffer((__gm__ T *)o);
  indicesOut.SetGlobalBuffer((__gm__ int32_t *)idx);
  constexpr uint32_t segmentBaseElements = 32 / sizeof(T);
  static_assert(segmentBaseElements + M * 6 <= 64);
  LocalTensor<T> coordinates(TPosition::VECCALC, 0, 64);
  DataCopyPad(coordinates, points, {1, 3 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  DataCopyPad(coordinates[segmentBaseElements], segments, {1, M * 6 * sizeof(T), 0, 0, 0},
        {false, 0, 0, 0});
  Fence<HardEvent::MTE2_S>();
  float px = float(coordinates.GetValue(0)), py = float(coordinates.GetValue(1)),
     pz = float(coordinates.GetValue(2));
  float best = 3.402823466e38f;
  int32_t arg = 0;
#pragma unroll
  for (uint32_t j = 0; j < M; ++j) {
    float a = float(coordinates.GetValue(segmentBaseElements + j * 6)),
       b = float(coordinates.GetValue(segmentBaseElements + j * 6 + 1)),
       c = float(coordinates.GetValue(segmentBaseElements + j * 6 + 2));
    float d;
    if constexpr (M == 1) {
      float u = float(coordinates.GetValue(segmentBaseElements + j * 6 + 3)) - a,
         v = float(coordinates.GetValue(segmentBaseElements + j * 6 + 4)) - b,
         w = float(coordinates.GetValue(segmentBaseElements + j * 6 + 5)) - c;
      float l = (u * u + v * v) + w * w, q = 0.0f;
      if (l > 0.0f)
        q = (((px - a) * u + (py - b) * v) + (pz - c) * w) / l;
      q = q < 0.0f ? 0.0f : (q > 1.0f ? 1.0f : q);
      float ex = px - (a + q * u), ey = py - (b + q * v), ez = pz - (c + q * w),
         squaredDistance = (ex * ex + ey * ey) + ez * ez;
      d = squaredDistance;
    } else {
      float endX = float(coordinates.GetValue(segmentBaseElements + j * 6 + 3)),
         endY = float(coordinates.GetValue(segmentBaseElements + j * 6 + 4)),
         endZ = float(coordinates.GetValue(segmentBaseElements + j * 6 + 5));
      if constexpr (M == 2 || (M >= 2 && IsSameType<T, float>::value)) {
        float u = endX - a, v = endY - b, w = endZ - c;
        float ex = px - a, ey = py - b, ez = pz - c;
        if (v == 0.0f && w == 0.0f) {
          ex = AxisIntervalResidual(px, a, endX);
        } else if (u == 0.0f && w == 0.0f) {
          ey = AxisIntervalResidual(py, b, endY);
        } else if (u == 0.0f && v == 0.0f) {
          ez = AxisIntervalResidual(pz, c, endZ);
        } else {
          float squaredLength = (u * u + v * v) + w * w;
          float projection = (ex * u + ey * v) + ez * w;
          // Clipping selects a real endpoint without division. Interior
          // projections retain the original residual evaluation order.
          if (squaredLength > 0.0f && projection > 0.0f) {
            if (projection >= squaredLength) {
              ex = px - endX;
              ey = py - endY;
              ez = pz - endZ;
            } else {
              float quotient = projection / squaredLength;
              ex = px - (a + quotient * u);
              ey = py - (b + quotient * v);
              ez = pz - (c + quotient * w);
            }
          }
        }
        d = (ex * ex + ey * ey) + ez * ez;
      } else {
        d = PointSegmentSquaredDistance(px, py, pz, a, b, c, endX, endY, endZ);
      }
    }
    if (j == 0 || d < best) {
      best = d;
      arg = int32_t(j);
    }
  }
  if constexpr (M <= 2 && IsSameType<T, float>::value) {
    // FixedPoint has exactly one producer. The bypass stores and DDR fence
    // are the same completion protocol as the existing wide single query.
    float distanceValue = float(sqrt(best));
    WriteGmByPassDCache(reinterpret_cast<__gm__ uint32_t *>(o),
                       __builtin_bit_cast(uint32_t, distanceValue));
    WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t *>(idx), arg);
    DataSyncBarrier<MemDsbT::DDR>();
    return;
  }
  coordinates.SetValue(0, T(sqrt(best)));
  LocalTensor<int32_t> indexLocal(TPosition::VECCALC, 256, 8);
  indexLocal.SetValue(0, arg);
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(output, coordinates, {1, sizeof(T), 0, 0, 0});
  DataCopyPad(indicesOut, indexLocal, {1, 4, 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}

// Keep the complete formal legacy implementation above. New layouts are independent.
namespace OptimizedLayout {

template <HardEvent E> __aicore__ inline void Fence() {
  SetFlag<E>(0);
  WaitFlag<E>(0);
}
// Build device-local byte-offset sequences by doubling an aligned prefix.
__aicore__ inline void DeviceOffsets(LocalTensor<uint32_t> dst, uint32_t count, uint32_t stride) {
  Fence<HardEvent::V_S>();
  uint32_t seed = count < 8 ? count : 8;
  for (uint32_t i = 0; i < seed; ++i)
    dst.SetValue(i, i * stride);
  Fence<HardEvent::S_V>();
  for (uint32_t have = 8; have < count; have *= 2) {
    uint32_t take = count - have;
    if (take > have)
      take = have;
    Adds(dst[have].ReinterpretCast<int32_t>(), dst.ReinterpretCast<int32_t>(), int32_t(have * stride),
      take);
    PipeBarrier<PIPE_V>();
  }
}

// These two helpers initialize disjoint fresh scratch tables in one S->V
// handoff. They may not replace DeviceOffsets when a previous phase still
// reads or writes the same table. Expansion can overlap an unrelated input DMA.
__aicore__ inline void SeedDeviceOffsets(LocalTensor<uint32_t> destination, uint32_t elementCount,
                    uint32_t byteStride) {
  uint32_t seedElementCount = elementCount < 8 ? elementCount : 8;
  for (uint32_t element = 0; element < seedElementCount; ++element)
    destination.SetValue(element, element * byteStride);
}
__aicore__ inline void ExpandDeviceOffsetSeeds(LocalTensor<uint32_t> destination, uint32_t elementCount,
                       uint32_t byteStride) {
  for (uint32_t prefixElements = 8; prefixElements < elementCount; prefixElements *= 2) {
    uint32_t appendedElements = elementCount - prefixElements;
    if (appendedElements > prefixElements)
      appendedElements = prefixElements;
    Adds(destination[prefixElements].ReinterpretCast<int32_t>(), destination.ReinterpretCast<int32_t>(),
      int32_t(prefixElements * byteStride), appendedElements);
    PipeBarrier<PIPE_V>();
  }
}

// Device-side spatial ordering and conservative segment-box lower bounds.

// A single query point needs no point-tile loop or staged output DMA.
// Only the existing single-point algorithms call this helper. One
// producer writes both current output elements and waits for GM completion.
template <typename T>
__aicore__ inline void StoreSinglePointResult(GM_ADDR distanceAddress, GM_ADDR indexAddress,
                      float squaredDistance, int32_t segmentIndex) {
  T distanceValue = T(sqrt(squaredDistance));
  if constexpr (IsSameType<T, float>::value) {
    uint32_t distanceBits = __builtin_bit_cast(uint32_t, distanceValue);
    WriteGmByPassDCache(reinterpret_cast<__gm__ uint32_t *>(distanceAddress), distanceBits);
  } else {
    uint16_t distanceBits = __builtin_bit_cast(uint16_t, distanceValue);
    WriteGmByPassDCache(reinterpret_cast<__gm__ uint16_t *>(distanceAddress), distanceBits);
  }
  WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t *>(indexAddress), segmentIndex);
  DataSyncBarrier<MemDsbT::DDR>();
}

template <typename T, uint32_t W>
__aicore__ inline void SinglePointTreeLine(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR oi, uint32_t m) {
  GlobalTensor<T> pg, sg;
  pg.SetGlobalBuffer((__gm__ T *)p);
  sg.SetGlobalBuffer((__gm__ T *)s);
  LocalTensor<float> mem(TPosition::VECCALC, 0, 32 * W + 512);
  auto raw = mem, ax = mem[7 * W], ay = mem[8 * W], az = mem[9 * W], dx = mem[10 * W], dy = mem[11 * W],
    dz = mem[12 * W], len = mem[13 * W], x = mem[14 * W], y = mem[15 * W], z = mem[16 * W],
    dot = mem[17 * W], tmp = mem[18 * W], red = mem[28 * W];
  auto off = mem[6 * W].template ReinterpretCast<uint32_t>();
  auto pointValues = mem[32 * W].template ReinterpretCast<T>();
  DataCopyPad(pointValues, pg, {1, uint32_t(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  SeedDeviceOffsets(off, W, 24);
  Duplicate(raw, 0.0f, 6 * W);
  Fence<HardEvent::V_MTE2>();
  if constexpr (IsSameType<T, float>::value) {
    DataCopyPad(raw, sg, {1, m * 24, 0, 0, 0}, {false, 0, 0, 0});
  } else {
    auto h = mem[24 * W].template ReinterpretCast<half>();
    DataCopyPad(h, sg, {1, m * 12, 0, 0, 0}, {false, 0, 0, 0});
  }
  Fence<HardEvent::S_V>();
  ExpandDeviceOffsetSeeds(off, W, 24);
  Fence<HardEvent::MTE2_S>();
  float px = float(pointValues.GetValue(0)), py = float(pointValues.GetValue(1)),
     pz = float(pointValues.GetValue(2));
  Fence<HardEvent::MTE2_V>();
  if constexpr (IsSameType<T, half>::value) {
    Cast(raw, mem[24 * W].template ReinterpretCast<half>(), RoundMode::CAST_NONE, m * 6);
    // The old offset generator's V->S fence also completed this cast.
    // Fresh-table expansion now runs before it, so retain the local RAW barrier.
    PipeBarrier<PIPE_V>();
  }
  Gather(ax, raw, off, 0, W);
  Gather(ay, raw, off, 4, W);
  Gather(az, raw, off, 8, W);
  Gather(dx, raw, off, 12, W);
  Gather(dy, raw, off, 16, W);
  Gather(dz, raw, off, 20, W);
  Sub(dx, dx, ax, W);
  Sub(dy, dy, ay, W);
  Sub(dz, dz, az, W);
  Mul(len, dx, dx, W);
  Mul(tmp, dy, dy, W);
  Add(len, len, tmp, W);
  Mul(tmp, dz, dz, W);
  Add(len, len, tmp, W);
  Mins(tmp.template ReinterpretCast<int32_t>(), len.template ReinterpretCast<int32_t>(), int32_t(1), W);
  Muls(tmp.template ReinterpretCast<int32_t>(), tmp.template ReinterpretCast<int32_t>(),
    int32_t(-0x3f800000), W);
  Adds(tmp.template ReinterpretCast<int32_t>(), tmp.template ReinterpretCast<int32_t>(),
    int32_t(0x3f800000), W);
  Or(len.template ReinterpretCast<uint16_t>(), len.template ReinterpretCast<uint16_t>(),
   tmp.template ReinterpretCast<uint16_t>(), W * 2);
  Adds(x, ax, -px, W);
  Adds(y, ay, -py, W);
  Adds(z, az, -pz, W);
  Mul(dot, x, dx, W);
  Mul(tmp, y, dy, W);
  Add(dot, dot, tmp, W);
  Mul(tmp, z, dz, W);
  Add(dot, dot, tmp, W);
  Div(dot, dot, len, W);
  Maxs(dot, dot, -1.0f, W);
  Mins(dot, dot, 0.0f, W);
  Mul(tmp, dot, dx, W);
  Sub(x, x, tmp, W);
  Mul(tmp, dot, dy, W);
  Sub(y, y, tmp, W);
  Mul(tmp, dot, dz, W);
  Sub(z, z, tmp, W);
  Mul(tmp, x, x, W);
  MulAddDst(tmp, y, y, W);
  MulAddDst(tmp, z, z, W);
  int32_t index;
  float value;
  if constexpr (W <= 64) {
    WholeReduceMin(red, tmp, int32_t(m), 1, 1, 1, 8);
    Fence<HardEvent::V_S>();
    value = red.GetValue(0);
    index = red.template ReinterpretCast<int32_t>().GetValue(1);
  } else {
    WholeReduceMin(red, tmp, int32_t(64), W / 64, 1, 1, 8);
    uint32_t tail = m % 64;
    if (tail)
      WholeReduceMin(red[(m / 64) * 2], tmp[(m / 64) * 64], int32_t(tail), 1, 1, 1, 8);
    uint64_t bits[2] = {0, 0};
    for (uint32_t c = 0; c < (m + 63) / 64; ++c)
      bits[0] |= uint64_t(1) << (c * 2);
    WholeReduceMin(red[128], red, bits, 1, 1, 1, 8);
    Fence<HardEvent::V_S>();
    value = red.GetValue(128);
    uint32_t at = red.template ReinterpretCast<uint32_t>().GetValue(129);
    index = int32_t((at / 2) * 64 + red.template ReinterpretCast<uint32_t>().GetValue(at + 1));
  }
  StoreSinglePointResult<T>(o, oi, value, index);
}

// Reuse each device-loaded segment across a tile of query points.
template <typename T, uint32_t StaticSegmentPitchElements = 0> class TiledTreeLine {
  static_assert(StaticSegmentPitchElements == 0 ||
         (StaticSegmentPitchElements >= 128 && StaticSegmentPitchElements <= 1024 &&
         StaticSegmentPitchElements % 64 == 0));
  static constexpr uint32_t residualScratchElements = 4096;
  GlobalTensor<T> pg, sg, og;
  GlobalTensor<int32_t> ig;
  LocalTensor<float> ax, ay, az, dx, dy, dz, inv, raw, x, y, z, dot, tmp, group, broadcast, c[8], red,
    best, arg, pointInput;
  // arg uses a float view only for bit-preserving Select; its values are int32.
  LocalTensor<uint32_t> off, pointOff;
  uint32_t mp, m, n, live;
  bool usesResidualGeometry;
  __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<T> src, uint32_t count,
                LocalTensor<float> scratch) {
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(dst, src, {1, count * 4, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    } else {
      auto h = scratch.ReinterpretCast<half>();
      DataCopyPad(h, src, {1, count * 2, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(dst, h, RoundMode::CAST_NONE, count);
    }
  }
  __aicore__ inline void PrefetchPointTile(uint32_t pointBegin, uint32_t pointCount) {
    DataCopyPad(pointInput.ReinterpretCast<T>(), pg[pointBegin * 3],
          {1, pointCount * uint32_t(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  }
  __aicore__ inline void StartInitialInputPrefetch(uint32_t batch, uint32_t pointBegin,
                          uint32_t pointCount) {
    Duplicate(raw, 0.0f, 6 * mp);
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value)
      DataCopyPad(raw, sg[batch * m * 6], {1, m * 24, 0, 0, 0}, {false, 0, 0, 0});
    else
      DataCopyPad(raw[6 * mp].ReinterpretCast<half>(), sg[batch * m * 6],
            {1, m * 12, 0, 0, 0}, {false, 0, 0, 0});
    PrefetchPointTile(pointBegin, pointCount);
  }
  __aicore__ inline void FinishInitialInputPrefetch() {
    Fence<HardEvent::MTE2_V>();
    if constexpr (IsSameType<T, half>::value)
      Cast(raw, raw[6 * mp].ReinterpretCast<half>(), RoundMode::CAST_NONE, m * 6);
  }
  __aicore__ inline void Segments(uint32_t batch, bool hasPrefetchedInputs = false) {
    if (!hasPrefetchedInputs) {
      Duplicate(raw, 0.0f, 6 * mp);
      Load(raw, sg[batch * m * 6], m * 6, raw[6 * mp]);
    }
    Gather(ax, raw, off, 0, mp);
    Gather(ay, raw, off, 4, mp);
    Gather(az, raw, off, 8, mp);
    Gather(dx, raw, off, 12, mp);
    Gather(dy, raw, off, 16, mp);
    Gather(dz, raw, off, 20, mp);
    Sub(dx, dx, ax, mp);
    Sub(dy, dy, ay, mp);
    Sub(dz, dz, az, mp);
    Mul(inv, dx, dx, mp);
    Mul(tmp, dy, dy, mp);
    Add(inv, inv, tmp, mp);
    Mul(tmp, dz, dz, mp);
    Add(inv, inv, tmp, mp);
    auto mask = red.ReinterpretCast<uint8_t>();
    CompareScalar(mask, inv, 0.0f, CMPMODE::GT, mp);
    Select(inv, mask, inv, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp);
    // Zero lengths have a safe denominator; their zero direction makes t irrelevant.
    Duplicate(tmp, 1.0f, mp);
    Div(inv, tmp, inv, mp);
  }
  __aicore__ inline void Mul64(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b) {
    Mul(dst, a, b, uint64_t(64), live, {1, 1, 1, 8, 8, 0});
  }
  __aicore__ inline void ResidualSquaredDistances(uint32_t repeatCount) {
    // A dense plane has 8192 data elements. Short point slices occupy
    // at most its lower half, leaving the upper half for products.
    auto scratch = x[residualScratchElements];
    uint32_t laneCount = repeatCount * 64;
    Mul(dot, x, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Mul(scratch, y, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, scratch, laneCount);
    Mul(scratch, z, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, scratch, laneCount);
    Mul(dot, dot, c[6], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Maxs(dot, dot, -1.0f, laneCount);
    Mins(dot, dot, 0.0f, laneCount);
    Mul(scratch, dot, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Sub(x, x, scratch, laneCount);
    Mul(scratch, dot, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Sub(y, y, scratch, laneCount);
    Mul(scratch, dot, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Sub(z, z, scratch, laneCount);
    Mul(x, x, x, laneCount);
    MulAddDst(x, y, y, laneCount);
    MulAddDst(x, z, z, laneCount);
  }
  __aicore__ inline void Chunk(uint32_t at, uint32_t count) {
    LocalTensor<float> source[7] = {ax, ay, az, dx, dy, dz, inv};
    for (uint32_t i = 0; i < 7; ++i)
      c[i] = source[i][at];

    Sub(x, c[0], broadcast, uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(y, c[1], broadcast[1024], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(z, c[2], broadcast[2048], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    if (usesResidualGeometry) {
      ResidualSquaredDistances(live);
    } else {
    Mul64(dot, x, c[3]);
    Mul64(x, y, c[4]);
    Add(dot, dot, x, live * 64);
    Mul64(x, z, c[5]);
    Add(dot, dot, x, live * 64);
    Mul64(dot, dot, c[6]);
    Maxs(dot, dot, -1.0f, live * 64);
    Mins(dot, dot, 0.0f, live * 64);
    Mul64(x, dot, c[3]);
    Mul64(y, dot, c[4]);
    Mul64(z, dot, c[5]);
    Sub(x, c[0], x, uint64_t(64), live, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], y, uint64_t(64), live, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], z, uint64_t(64), live, {1, 1, 1, 8, 0, 8});

    Sub(x, broadcast, x, uint64_t(64), live, {1, 0, 1, 8, 1, 8});
    Sub(y, broadcast[1024], y, uint64_t(64), live, {1, 0, 1, 8, 1, 8});
    Sub(z, broadcast[2048], z, uint64_t(64), live, {1, 0, 1, 8, 1, 8});
    Mul(x, x, x, live * 64);
    Mul(y, y, y, live * 64);
    Add(x, x, y, live * 64);
    Mul(y, z, z, live * 64);
    Add(x, x, y, live * 64);
    }
    WholeReduceMin(red, x, int32_t(count), live, 1, 1, 8);
    // ORDER_VALUE_INDEX already packs value/index pairs consecutively.
    // Fixed deinterleave masks avoid indexed loads and an offset table.
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(live * 2);
    GatherMask(x, red, uint8_t(1), {1, 1, 8, 1});
    GatherMask(y, red, uint8_t(2), {1, 1, 8, 1});
    SetMaskNorm();
    ResetMask();
    Adds(y.ReinterpretCast<int32_t>(), y.ReinterpretCast<int32_t>(), int32_t(at), live);
    // Keep comparison masks at a full vector width, including short point tails.
    auto mask = red[256].ReinterpretCast<uint8_t>();
    Compare(mask, x, best, CMPMODE::LT, 128);
    Min(best, best, x, 128);
    Select(arg, mask, y, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, 128);
  }
  template <uint32_t W> __aicore__ inline void TailPacked(uint32_t at, uint32_t count) {
    LocalTensor<float> source[7] = {ax, ay, az, dx, dy, dz, inv};
    for (uint32_t a = 0; a < 7; ++a) {
      c[a] = x[30720 + a * 64];
      Copy(c[a], source[a][at], uint64_t(W), uint8_t(64 / W), {1, 1, W / 8, 0});
    }
    uint32_t reps = (live + 64 / W - 1) / (64 / W), lanes = reps * 64;
    LocalTensor<float> bx = broadcast, by = broadcast[1024], bz = broadcast[2048];
    if constexpr (W > 8) {
      bx = x[4096];
      by = y[4096];
      bz = z[4096];
      uint32_t rows = reps * (64 / W);
      Copy(bx, broadcast, uint64_t(W), uint8_t(rows), {1, 0, W / 8, 1});
      Copy(by, broadcast[1024], uint64_t(W), uint8_t(rows), {1, 0, W / 8, 1});
      Copy(bz, broadcast[2048], uint64_t(W), uint8_t(rows), {1, 0, W / 8, 1});
    }
    Sub(x, c[0], bx, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], by, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], bz, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    if (usesResidualGeometry) {
      // Wider packed tails put bx in the same upper half. Its last
      // read must finish before that storage becomes product scratch.
      PipeBarrier<PIPE_V>();
      ResidualSquaredDistances(reps);
    } else {
    Mul(dot, x, c[3], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Mul(x, y, c[4], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, x, lanes);
    Mul(x, z, c[5], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, x, lanes);
    Mul(dot, dot, c[6], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Maxs(dot, dot, -1.0f, lanes);
    Mins(dot, dot, 0.0f, lanes);
    Mul(x, dot, c[3], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Mul(y, dot, c[4], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Mul(z, dot, c[5], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Sub(x, c[0], x, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], y, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], z, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(x, bx, x, lanes);
    Sub(y, by, y, lanes);
    Sub(z, bz, z, lanes);
    Mul(x, x, x, lanes);
    Mul(y, y, y, lanes);
    Add(x, x, y, lanes);
    Mul(y, z, z, lanes);
    Add(x, x, y, lanes);
    }
    WholeReduceMin(red, x, int32_t(count), live, 1, 1, W / 8);
    // ORDER_VALUE_INDEX already packs value/index pairs consecutively.
    // Fixed deinterleave masks avoid indexed loads and an offset table.
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(live * 2);
    GatherMask(x, red, uint8_t(1), {1, 1, 8, 1});
    GatherMask(y, red, uint8_t(2), {1, 1, 8, 1});
    SetMaskNorm();
    ResetMask();
    Adds(y.ReinterpretCast<int32_t>(), y.ReinterpretCast<int32_t>(), int32_t(at), live);
    auto mask = red[256].ReinterpretCast<uint8_t>();
    Compare(mask, x, best, CMPMODE::LT, 128);
    Min(best, best, x, 128);
    Select(arg, mask, y, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, 128);
  }
  __aicore__ inline void Tail(uint32_t at, uint32_t count) {
    uint32_t lanes = (live + 63) / 64 * 64;
    auto savedTmp = tmp;
    tmp = x[32512];
    Fence<HardEvent::V_S>();
    for (uint32_t j = at; j < at + count; ++j) {
      float a = ax.GetValue(j), b = ay.GetValue(j), c0 = az.GetValue(j), u = dx.GetValue(j),
         v = dy.GetValue(j), w = dz.GetValue(j), r = inv.GetValue(j);
      Adds(x, group, -a, lanes);
      Adds(y, group[128], -b, lanes);
      Adds(z, group[256], -c0, lanes);
      Muls(dot, x, u, lanes);
      Muls(tmp, y, v, lanes);
      Add(dot, dot, tmp, lanes);
      Muls(tmp, z, w, lanes);
      Add(dot, dot, tmp, lanes);
      Muls(dot, dot, r, lanes);
      Maxs(dot, dot, 0.0f, lanes);
      Mins(dot, dot, 1.0f, lanes);
      if (usesResidualGeometry) {
        // Match the residual formula and sum order of full chunks,
        // including ties between a full chunk and this scalar tail.
        Muls(tmp, dot, u, lanes);
        Sub(x, x, tmp, lanes);
        Muls(tmp, dot, v, lanes);
        Sub(y, y, tmp, lanes);
        Muls(tmp, dot, w, lanes);
        Sub(z, z, tmp, lanes);
        Mul(x, x, x, lanes);
        MulAddDst(x, y, y, lanes);
        MulAddDst(x, z, z, lanes);
      } else {
      Muls(x, dot, u, lanes);
      Muls(y, dot, v, lanes);
      Muls(z, dot, w, lanes);
      Adds(x, x, a, lanes);
      Adds(y, y, b, lanes);
      Adds(z, z, c0, lanes);
      Sub(x, group, x, lanes);
      Sub(y, group[128], y, lanes);
      Sub(z, group[256], z, lanes);
      Mul(x, x, x, lanes);
      Mul(tmp, y, y, lanes);
      Add(x, x, tmp, lanes);
      Mul(tmp, z, z, lanes);
      Add(x, x, tmp, lanes);
      }
      auto mask = red[256].ReinterpretCast<uint8_t>();
      Compare(mask, x, best, CMPMODE::LT, lanes);
      Duplicate(y.ReinterpretCast<int32_t>(), int32_t(j), lanes);
      Min(best, best, x, lanes);
      Select(arg, mask, y, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
    }
    tmp = savedTmp;
  }

 public:
  __aicore__ inline void Run(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR oi,
               const TreeLineMinDistanceTilingData &t) {
    pg.SetGlobalBuffer((__gm__ T *)p);
    sg.SetGlobalBuffer((__gm__ T *)s);
    og.SetGlobalBuffer((__gm__ T *)o);
    ig.SetGlobalBuffer((__gm__ int32_t *)oi);
    m = t.segmentCount;
    n = t.pointsPerBatch;
    // Template pitch changes addresses only; actual m still bounds all loads and reductions.
    mp = StaticSegmentPitchElements ? StaticSegmentPitchElements : (m + 63) / 64 * 64;
    LocalTensor<float> mem(TPosition::VECCALC, 0, (t.unifiedBufferBytes - 8192) / 4);
    ax = mem;
    ay = mem[mp];
    az = mem[2 * mp];
    dx = mem[3 * mp];
    dy = mem[4 * mp];
    dz = mem[5 * mp];
    inv = mem[6 * mp];
    off = mem[7 * mp].ReinterpretCast<uint32_t>();
    // Separate the four dense planes by 256 bytes to vary their UB bank origins.
    raw = mem[8 * mp];
    x = raw;
    y = x[8256];
    z = x[16512];
    dot = x[24768];
    tmp = dot;
    group = x[33024];
    red = x[33408];
    best = x[33920];
    arg = x[34048];
    pointOff = x[34176].ReinterpretCast<uint32_t>();
    broadcast = x[34432];
    // The staging tile lies beyond all four geometry planes and output
    // state. Host reserves these 384 FP32 elements in the UB budget.
    pointInput = x[37504];
    uint32_t total = t.batchCount * n, start = total * GetBlockIdx() / t.workerCount,
        end = total * (GetBlockIdx() + 1) / t.workerCount, oldBatch = 0xffffffff;
    usesResidualGeometry = uint64_t(end - start) * 64 <= residualScratchElements;
    uint32_t firstPointCount = end - start;
    if (firstPointCount > 128)
      firstPointCount = 128;
    uint32_t firstBatch = start / n;
    if (firstPointCount > (firstBatch + 1) * n - start)
      firstPointCount = (firstBatch + 1) * n - start;
    StartInitialInputPrefetch(firstBatch, start, firstPointCount);
    // These two scratch tables are disjoint and fresh. Share their
    // scalar-to-vector handoff and retain segment offsets across batches.
    SeedDeviceOffsets(pointOff, 128, 12);
    SeedDeviceOffsets(off, mp, 24);
    Fence<HardEvent::S_V>();
    ExpandDeviceOffsetSeeds(pointOff, 128, 12);
    ExpandDeviceOffsetSeeds(off, mp, 24);

    for (uint32_t at = start; at < end;) {
      uint32_t batch = at / n;
      if (batch != oldBatch) {
        if (at == start)
          FinishInitialInputPrefetch();
        Segments(batch, at == start);
        oldBatch = batch;
      }
      live = end - at;
      if (live > 128)
        live = 128;
      if (live > (batch + 1) * n - at)
        live = (batch + 1) * n - at;
      Duplicate(group, 0.0f, 384);
      if (at != start)
        Fence<HardEvent::MTE2_V>();
      auto pointCoordinates = pointInput;
      if constexpr (IsSameType<T, half>::value) {
        Cast(raw, pointInput.ReinterpretCast<half>(), RoundMode::CAST_NONE, live * 3);
        pointCoordinates = raw;
      }
      Gather(group, pointCoordinates, pointOff, 0, live);
      Gather(group[128], pointCoordinates, pointOff, 4, live);
      Gather(group[256], pointCoordinates, pointOff, 8, live);
      uint32_t nextPoint = at + live;
      if (nextPoint < end) {
        uint32_t nextPointCount = end - nextPoint;
        if (nextPointCount > 128)
          nextPointCount = 128;
        uint32_t nextBatchEnd = (nextPoint / n + 1) * n;
        if (nextPointCount > nextBatchEnd - nextPoint)
          nextPointCount = nextBatchEnd - nextPoint;
        Fence<HardEvent::V_MTE2>();
        PrefetchPointTile(nextPoint, nextPointCount);
      }
      Brcb(broadcast, group, uint8_t((live + 7) / 8), {1, 8});
      Brcb(broadcast[1024], group[128], uint8_t((live + 7) / 8), {1, 8});
      Brcb(broadcast[2048], group[256], uint8_t((live + 7) / 8), {1, 8});
      Duplicate(best, 3.402823466e38f, 128);
      Duplicate(arg.ReinterpretCast<int32_t>(), int32_t(0), 128);
      for (uint32_t j = 0; j < m; j += 64) {
        uint32_t count = m - j;
        if (count > 64)
          count = 64;
        if (count <= 3 && live >= 32)
          Tail(j, count);
        else if (count <= 32 && live >= 8) {
          // A short point slice packs a small segment tail into
          // full vectors, avoiding a 64-segment row for one tail.
          if (count <= 8)
            TailPacked<8>(j, count);
          else if (count <= 16)
            TailPacked<16>(j, count);
          else
            TailPacked<32>(j, count);
        } else
          Chunk(j, count);
      }
      Sqrt(best, best, live);
      if constexpr (IsSameType<T, float>::value) {
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(og[at], best, {1, live * 4, 0, 0, 0});
      } else {
        auto h = best.ReinterpretCast<half>();
        Cast(h, best, RoundMode::CAST_RINT, live);
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(og[at], h, {1, live * 2, 0, 0, 0});
      }
      DataCopyPad(ig[at], arg.ReinterpretCast<int32_t>(), {1, live * 4, 0, 0, 0});
      Fence<HardEvent::MTE3_V>();
      at += live;
    }
  }
};

// Separate entries retain existing paths while testing the qualified dense
// scratch component. All arithmetic, tails and per-call state are complete.
template <typename T, uint32_t StaticSegmentPitchElements = 0> class DenseScratchTiledTreeLine {
  static_assert(StaticSegmentPitchElements == 0 ||
         (StaticSegmentPitchElements >= 128 && StaticSegmentPitchElements <= 1024 &&
         StaticSegmentPitchElements % 64 == 0));
  static constexpr uint32_t residualScratchElements = 4096;
  GlobalTensor<T> pg, sg, og;
  GlobalTensor<int32_t> ig;
  LocalTensor<float> ax, ay, az, dx, dy, dz, inv, raw, x, y, z, dot, tmp, group, broadcast, c[8], red,
    best, arg, pointInput, residualProductScratch, residualY, residualSquared;
  // arg uses a float view only for bit-preserving Select; its values are int32.
  LocalTensor<uint32_t> off, pointOff;
  uint32_t mp, m, n, live;
  bool usesResidualGeometry;
  __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<T> src, uint32_t count,
                LocalTensor<float> scratch) {
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(dst, src, {1, count * 4, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    } else {
      auto h = scratch.ReinterpretCast<half>();
      DataCopyPad(h, src, {1, count * 2, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(dst, h, RoundMode::CAST_NONE, count);
    }
  }
  __aicore__ inline void PrefetchPointTile(uint32_t pointBegin, uint32_t pointCount) {
    DataCopyPad(pointInput.ReinterpretCast<T>(), pg[pointBegin * 3],
          {1, pointCount * uint32_t(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  }
  __aicore__ inline void StartInitialInputPrefetch(uint32_t batch, uint32_t pointBegin,
                          uint32_t pointCount) {
    Duplicate(raw, 0.0f, 6 * mp);
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value)
      DataCopyPad(raw, sg[batch * m * 6], {1, m * 24, 0, 0, 0}, {false, 0, 0, 0});
    else
      DataCopyPad(raw[6 * mp].ReinterpretCast<half>(), sg[batch * m * 6],
            {1, m * 12, 0, 0, 0}, {false, 0, 0, 0});
    PrefetchPointTile(pointBegin, pointCount);
  }
  __aicore__ inline void FinishInitialInputPrefetch() {
    Fence<HardEvent::MTE2_V>();
    if constexpr (IsSameType<T, half>::value)
      Cast(raw, raw[6 * mp].ReinterpretCast<half>(), RoundMode::CAST_NONE, m * 6);
  }
  __aicore__ inline void Segments(uint32_t batch, bool hasPrefetchedInputs = false) {
    if (!hasPrefetchedInputs) {
      Duplicate(raw, 0.0f, 6 * mp);
      Load(raw, sg[batch * m * 6], m * 6, raw[6 * mp]);
    }
    Gather(ax, raw, off, 0, mp);
    Gather(ay, raw, off, 4, mp);
    Gather(az, raw, off, 8, mp);
    Gather(dx, raw, off, 12, mp);
    Gather(dy, raw, off, 16, mp);
    Gather(dz, raw, off, 20, mp);
    Sub(dx, dx, ax, mp);
    Sub(dy, dy, ay, mp);
    Sub(dz, dz, az, mp);
    Mul(inv, dx, dx, mp);
    Mul(tmp, dy, dy, mp);
    Add(inv, inv, tmp, mp);
    Mul(tmp, dz, dz, mp);
    Add(inv, inv, tmp, mp);
    auto mask = red.ReinterpretCast<uint8_t>();
    CompareScalar(mask, inv, 0.0f, CMPMODE::GT, mp);
    Select(inv, mask, inv, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp);
    // Zero lengths have a safe denominator; their zero direction makes t irrelevant.
    Duplicate(tmp, 1.0f, mp);
    Div(inv, tmp, inv, mp);
  }
  __aicore__ inline void Mul64(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b) {
    Mul(dst, a, b, uint64_t(64), live, {1, 1, 1, 8, 8, 0});
  }
  __aicore__ inline void ResidualSquaredDistances(uint32_t repeatCount) {
    if constexpr (StaticSegmentPitchElements == 0 || StaticSegmentPitchElements >= 256) {
      // A dense plane has 8192 data elements. Short point slices occupy
      // at most its lower half, leaving the upper half for products.
      auto scratch = x[residualScratchElements];
      Mul(dot, x, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Mul(scratch, y, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Add(dot, dot, scratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(scratch, z, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Add(dot, dot, scratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(dot, dot, c[6], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Maxs(dot, dot, -1.0f, uint64_t(64), repeatCount, {1, 1, 8, 8});
      Mins(dot, dot, 0.0f, uint64_t(64), repeatCount, {1, 1, 8, 8});
      Mul(residualProductScratch, dot, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Sub(x, x, residualProductScratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(residualProductScratch, dot, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Sub(residualY, y, residualProductScratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(residualProductScratch, dot, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Sub(z, z, residualProductScratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(residualSquared, x, x, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      MulAddDst(residualSquared, residualY, residualY, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      MulAddDst(residualSquared, z, z, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
    } else {
      // A dense plane has 8192 data elements. Short point slices occupy
      // at most its lower half, leaving the upper half for products.
      auto scratch = x[residualScratchElements];
      Mul(dot, x, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Mul(scratch, y, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Add(dot, dot, scratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(scratch, z, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Add(dot, dot, scratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(dot, dot, c[6], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Maxs(dot, dot, -1.0f, uint64_t(64), repeatCount, {1, 1, 8, 8});
      Mins(dot, dot, 0.0f, uint64_t(64), repeatCount, {1, 1, 8, 8});
      Mul(scratch, dot, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Sub(x, x, scratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(scratch, dot, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Sub(y, y, scratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(scratch, dot, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Sub(z, z, scratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(x, x, x, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      MulAddDst(x, y, y, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      MulAddDst(x, z, z, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
    }
  }
  __aicore__ inline void Chunk(uint32_t at, uint32_t count) {
    LocalTensor<float> source[7] = {ax, ay, az, dx, dy, dz, inv};
    for (uint32_t i = 0; i < 7; ++i)
      c[i] = source[i][at];

    Sub(x, c[0], broadcast, uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(y, c[1], broadcast[1024], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(z, c[2], broadcast[2048], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    if (usesResidualGeometry) {
      ResidualSquaredDistances(live);
    } else {
    Mul64(dot, x, c[3]);
    Mul64(x, y, c[4]);
    Add(dot, dot, x, live * 64);
    Mul64(x, z, c[5]);
    Add(dot, dot, x, live * 64);
    Mul64(dot, dot, c[6]);
    Maxs(dot, dot, -1.0f, live * 64);
    Mins(dot, dot, 0.0f, live * 64);
    Mul64(x, dot, c[3]);
    Mul64(y, dot, c[4]);
    Mul64(z, dot, c[5]);
    Sub(x, c[0], x, uint64_t(64), live, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], y, uint64_t(64), live, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], z, uint64_t(64), live, {1, 1, 1, 8, 0, 8});

    Sub(x, broadcast, x, uint64_t(64), live, {1, 0, 1, 8, 1, 8});
    Sub(y, broadcast[1024], y, uint64_t(64), live, {1, 0, 1, 8, 1, 8});
    Sub(z, broadcast[2048], z, uint64_t(64), live, {1, 0, 1, 8, 1, 8});
    Mul(x, x, x, live * 64);
    Mul(y, y, y, live * 64);
    Add(x, x, y, live * 64);
    Mul(y, z, z, live * 64);
    Add(x, x, y, live * 64);
    }
    if constexpr (StaticSegmentPitchElements == 0 || StaticSegmentPitchElements >= 256)
      WholeReduceMin(red, residualSquared, int32_t(count), live, 1, 1, 8);
    else
      WholeReduceMin(red, x, int32_t(count), live, 1, 1, 8);
    // ORDER_VALUE_INDEX already packs value/index pairs consecutively.
    // Fixed deinterleave masks avoid indexed loads and an offset table.
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(live * 2);
    GatherMask(x, red, uint8_t(1), {1, 1, 8, 1});
    GatherMask(y, red, uint8_t(2), {1, 1, 8, 1});
    SetMaskNorm();
    ResetMask();
    Adds(y.ReinterpretCast<int32_t>(), y.ReinterpretCast<int32_t>(), int32_t(at), live);
    // Keep comparison masks at a full vector width, including short point tails.
    auto mask = red[256].ReinterpretCast<uint8_t>();
    Compare(mask, x, best, CMPMODE::LT, 128);
    Min(best, best, x, 128);
    Select(arg, mask, y, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, 128);
  }
  template <uint32_t W> __aicore__ inline void TailPacked(uint32_t at, uint32_t count) {
    LocalTensor<float> source[7] = {ax, ay, az, dx, dy, dz, inv};
    for (uint32_t a = 0; a < 7; ++a) {
      c[a] = x[30720 + a * 64];
      Copy(c[a], source[a][at], uint64_t(W), uint8_t(64 / W), {1, 1, W / 8, 0});
    }
    uint32_t reps = (live + 64 / W - 1) / (64 / W), lanes = reps * 64;
    LocalTensor<float> bx = broadcast, by = broadcast[1024], bz = broadcast[2048];
    if constexpr (W > 8) {
      bx = x[4096];
      by = y[4096];
      bz = z[4096];
      uint32_t rows = reps * (64 / W);
      Copy(bx, broadcast, uint64_t(W), uint8_t(rows), {1, 0, W / 8, 1});
      Copy(by, broadcast[1024], uint64_t(W), uint8_t(rows), {1, 0, W / 8, 1});
      Copy(bz, broadcast[2048], uint64_t(W), uint8_t(rows), {1, 0, W / 8, 1});
    }
    Sub(x, c[0], bx, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], by, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], bz, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    if (usesResidualGeometry) {
      // Wider packed tails put bx in the same upper half. Its last
      // read must finish before that storage becomes product scratch.
      PipeBarrier<PIPE_V>();
      ResidualSquaredDistances(reps);
    } else {
    Mul(dot, x, c[3], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Mul(x, y, c[4], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, x, lanes);
    Mul(x, z, c[5], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, x, lanes);
    Mul(dot, dot, c[6], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Maxs(dot, dot, -1.0f, lanes);
    Mins(dot, dot, 0.0f, lanes);
    Mul(x, dot, c[3], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Mul(y, dot, c[4], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Mul(z, dot, c[5], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Sub(x, c[0], x, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], y, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], z, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(x, bx, x, lanes);
    Sub(y, by, y, lanes);
    Sub(z, bz, z, lanes);
    Mul(x, x, x, lanes);
    Mul(y, y, y, lanes);
    Add(x, x, y, lanes);
    Mul(y, z, z, lanes);
    Add(x, x, y, lanes);
    }
    if constexpr (StaticSegmentPitchElements == 0 || StaticSegmentPitchElements >= 256)
      WholeReduceMin(red, residualSquared, int32_t(count), live, 1, 1, W / 8);
    else
      WholeReduceMin(red, x, int32_t(count), live, 1, 1, W / 8);
    // ORDER_VALUE_INDEX already packs value/index pairs consecutively.
    // Fixed deinterleave masks avoid indexed loads and an offset table.
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(live * 2);
    GatherMask(x, red, uint8_t(1), {1, 1, 8, 1});
    GatherMask(y, red, uint8_t(2), {1, 1, 8, 1});
    SetMaskNorm();
    ResetMask();
    Adds(y.ReinterpretCast<int32_t>(), y.ReinterpretCast<int32_t>(), int32_t(at), live);
    auto mask = red[256].ReinterpretCast<uint8_t>();
    Compare(mask, x, best, CMPMODE::LT, 128);
    Min(best, best, x, 128);
    Select(arg, mask, y, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, 128);
  }
  __aicore__ inline void Tail(uint32_t at, uint32_t count) {
    uint32_t lanes = (live + 63) / 64 * 64;
    auto savedTmp = tmp;
    tmp = x[32512];
    Fence<HardEvent::V_S>();
    for (uint32_t j = at; j < at + count; ++j) {
      float a = ax.GetValue(j), b = ay.GetValue(j), c0 = az.GetValue(j), u = dx.GetValue(j),
         v = dy.GetValue(j), w = dz.GetValue(j), r = inv.GetValue(j);
      Adds(x, group, -a, lanes);
      Adds(y, group[128], -b, lanes);
      Adds(z, group[256], -c0, lanes);
      Muls(dot, x, u, lanes);
      Muls(tmp, y, v, lanes);
      Add(dot, dot, tmp, lanes);
      Muls(tmp, z, w, lanes);
      Add(dot, dot, tmp, lanes);
      Muls(dot, dot, r, lanes);
      Maxs(dot, dot, 0.0f, lanes);
      Mins(dot, dot, 1.0f, lanes);
      if (usesResidualGeometry) {
        // Match the residual formula and sum order of full chunks,
        // including ties between a full chunk and this scalar tail.
        Muls(tmp, dot, u, lanes);
        Sub(x, x, tmp, lanes);
        Muls(tmp, dot, v, lanes);
        Sub(y, y, tmp, lanes);
        Muls(tmp, dot, w, lanes);
        Sub(z, z, tmp, lanes);
        Mul(x, x, x, lanes);
        MulAddDst(x, y, y, lanes);
        MulAddDst(x, z, z, lanes);
      } else {
      Muls(x, dot, u, lanes);
      Muls(y, dot, v, lanes);
      Muls(z, dot, w, lanes);
      Adds(x, x, a, lanes);
      Adds(y, y, b, lanes);
      Adds(z, z, c0, lanes);
      Sub(x, group, x, lanes);
      Sub(y, group[128], y, lanes);
      Sub(z, group[256], z, lanes);
      Mul(x, x, x, lanes);
      Mul(tmp, y, y, lanes);
      Add(x, x, tmp, lanes);
      Mul(tmp, z, z, lanes);
      Add(x, x, tmp, lanes);
      }
      auto mask = red[256].ReinterpretCast<uint8_t>();
      Compare(mask, x, best, CMPMODE::LT, lanes);
      Duplicate(y.ReinterpretCast<int32_t>(), int32_t(j), lanes);
      Min(best, best, x, lanes);
      Select(arg, mask, y, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
    }
    tmp = savedTmp;
  }

 public:
  __aicore__ inline void Run(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR oi,
               const TreeLineMinDistanceTilingData &t) {
    pg.SetGlobalBuffer((__gm__ T *)p);
    sg.SetGlobalBuffer((__gm__ T *)s);
    og.SetGlobalBuffer((__gm__ T *)o);
    ig.SetGlobalBuffer((__gm__ int32_t *)oi);
    m = t.segmentCount;
    n = t.pointsPerBatch;
    // Template pitch changes addresses only; actual m still bounds all loads and reductions.
    mp = StaticSegmentPitchElements ? StaticSegmentPitchElements : (m + 63) / 64 * 64;
    LocalTensor<float> mem(TPosition::VECCALC, 0, (t.unifiedBufferBytes - 8192) / 4);
    ax = mem;
    ay = mem[mp];
    az = mem[2 * mp];
    dx = mem[3 * mp];
    dy = mem[4 * mp];
    dz = mem[5 * mp];
    inv = mem[6 * mp];
    off = mem[7 * mp].ReinterpretCast<uint32_t>();
    // Separate the four dense planes by 256 bytes to vary their UB bank origins.
    raw = mem[8 * mp];
    x = raw;
    y = x[8256];
    z = x[16512];
    dot = x[24768];
    tmp = dot;
    group = x[33024];
    red = x[33408];
    best = x[33920];
    arg = x[34048];
    pointOff = x[34176].ReinterpretCast<uint32_t>();
    broadcast = x[34432];
    // The staging tile lies beyond all four geometry planes and output
    // state. Host reserves these 384 FP32 elements in the UB budget.
    pointInput = x[37504];
    uint32_t total = t.batchCount * n, start = total * GetBlockIdx() / t.workerCount,
        end = total * (GetBlockIdx() + 1) / t.workerCount, oldBatch = 0xffffffff;
    usesResidualGeometry = uint64_t(end - start) * 64 <= residualScratchElements;
    uint32_t firstPointCount = end - start;
    if (firstPointCount > 128)
      firstPointCount = 128;
    uint32_t firstBatch = start / n;
    if (firstPointCount > (firstBatch + 1) * n - start)
      firstPointCount = (firstBatch + 1) * n - start;
    StartInitialInputPrefetch(firstBatch, start, firstPointCount);
    // These two scratch tables are disjoint and fresh. Share their
    // scalar-to-vector handoff and retain segment offsets across batches.
    SeedDeviceOffsets(pointOff, 128, 12);
    SeedDeviceOffsets(off, mp, 24);
    Fence<HardEvent::S_V>();
    ExpandDeviceOffsetSeeds(pointOff, 128, 12);
    ExpandDeviceOffsetSeeds(off, mp, 24);

    for (uint32_t at = start; at < end;) {
      uint32_t batch = at / n;
      if (batch != oldBatch) {
        if (at == start)
          FinishInitialInputPrefetch();
        Segments(batch, at == start);
        oldBatch = batch;
      }
      live = end - at;
      if (live > 128)
        live = 128;
      if (live > (batch + 1) * n - at)
        live = (batch + 1) * n - at;
      if constexpr (StaticSegmentPitchElements == 0 || StaticSegmentPitchElements >= 256) {
        // Select once for this complete point tile; full and packed
        // chunks share the same physical squared-norm destination.
        residualProductScratch = x[residualScratchElements];
        residualY = y;
        residualSquared = x;
        if (usesResidualGeometry && live >= 32 && mp >= 256) {
          // Projection uses the lower dot half. Packed tails can
          // overwrite only dead anchors below direction staging.
          residualProductScratch = dot[residualScratchElements];
          residualY = x[residualScratchElements];
          residualSquared = y;
        }
      }
      Duplicate(group, 0.0f, 384);
      if (at != start)
        Fence<HardEvent::MTE2_V>();
      auto pointCoordinates = pointInput;
      if constexpr (IsSameType<T, half>::value) {
        Cast(raw, pointInput.ReinterpretCast<half>(), RoundMode::CAST_NONE, live * 3);
        pointCoordinates = raw;
      }
      Gather(group, pointCoordinates, pointOff, 0, live);
      Gather(group[128], pointCoordinates, pointOff, 4, live);
      Gather(group[256], pointCoordinates, pointOff, 8, live);
      uint32_t nextPoint = at + live;
      if (nextPoint < end) {
        uint32_t nextPointCount = end - nextPoint;
        if (nextPointCount > 128)
          nextPointCount = 128;
        uint32_t nextBatchEnd = (nextPoint / n + 1) * n;
        if (nextPointCount > nextBatchEnd - nextPoint)
          nextPointCount = nextBatchEnd - nextPoint;
        Fence<HardEvent::V_MTE2>();
        PrefetchPointTile(nextPoint, nextPointCount);
      }
      Brcb(broadcast, group, uint8_t((live + 7) / 8), {1, 8});
      Brcb(broadcast[1024], group[128], uint8_t((live + 7) / 8), {1, 8});
      Brcb(broadcast[2048], group[256], uint8_t((live + 7) / 8), {1, 8});
      Duplicate(best, 3.402823466e38f, 128);
      Duplicate(arg.ReinterpretCast<int32_t>(), int32_t(0), 128);
      for (uint32_t j = 0; j < m; j += 64) {
        uint32_t count = m - j;
        if (count > 64)
          count = 64;
        if (count <= 3 && live >= 32)
          Tail(j, count);
        else if (count <= 32 && live >= 8) {
          // A short point slice packs a small segment tail into
          // full vectors, avoiding a 64-segment row for one tail.
          if (count <= 8)
            TailPacked<8>(j, count);
          else if (count <= 16)
            TailPacked<16>(j, count);
          else
            TailPacked<32>(j, count);
        } else
          Chunk(j, count);
      }
      Sqrt(best, best, live);
      if constexpr (IsSameType<T, float>::value) {
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(og[at], best, {1, live * 4, 0, 0, 0});
      } else {
        auto h = best.ReinterpretCast<half>();
        Cast(h, best, RoundMode::CAST_RINT, live);
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(og[at], h, {1, live * 2, 0, 0, 0});
      }
      DataCopyPad(ig[at], arg.ReinterpretCast<int32_t>(), {1, live * 4, 0, 0, 0});
      Fence<HardEvent::MTE3_V>();
      at += live;
    }
  }
};

// Segment-major triples let three short queries share one SIMD repeat.
// Every real pair is computed. Fixed slot masks keep reductions independent.
__aicore__ inline void ComputeThreeInterleavedFloatQueries(
    GM_ADDR pointAddress, GM_ADDR segmentAddress, GM_ADDR distanceAddress,
    GM_ADDR indexAddress, uint32_t pointCount, uint32_t segmentCount) {
  constexpr uint64_t firstSlotMask = 0x9249249249249249ULL;
  LocalTensor<float> memory(TPosition::VECCALC, 0, 2048);
  auto raw = memory, ax = memory[192], ay = memory[256], az = memory[320],
       dx = memory[384], dy = memory[448], dz = memory[512], reciprocal = memory[576],
       x = memory[640], y = memory[704], z = memory[768], projection = memory[832],
       temporary = memory[896], reduction = memory[960], pointInput = memory[1024];
  auto pointOffsets = memory[1088].ReinterpretCast<uint32_t>();
  auto slotIndices = memory[1152].ReinterpretCast<uint32_t>();
  auto segmentOffsets = memory[1216].ReinterpretCast<uint32_t>();
  GlobalTensor<float> points, segments, distances;
  GlobalTensor<int32_t> indices;
  points.SetGlobalBuffer((__gm__ float *)pointAddress);
  segments.SetGlobalBuffer((__gm__ float *)segmentAddress);
  distances.SetGlobalBuffer((__gm__ float *)distanceAddress);
  indices.SetGlobalBuffer((__gm__ int32_t *)indexAddress);
  Duplicate(raw, 0.0f, 192);
  Duplicate(pointInput, 0.0f, 16);
  Fence<HardEvent::V_MTE2>();
  DataCopyPad(raw, segments, {1, segmentCount * 24, 0, 0, 0}, {false, 0, 0, 0});
  // Initialize all three point slots, including rows outside a short tail.
  uint8_t pointPaddingElements = uint8_t((8 - pointCount * 3 % 8) % 8);
  DataCopyPad(pointInput, points, {1, pointCount * 12, 0, 0, 0},
              {true, 0, pointPaddingElements, 0.0f});
  uint64_t slot0[2] = {firstSlotMask, 0};
  uint64_t slot1[2] = {firstSlotMask << 1, 0};
  uint64_t slot2[2] = {firstSlotMask << 2, 0};
  Duplicate(slotIndices, uint32_t(0), slot0, 1, 1, 8);
  Duplicate(slotIndices, uint32_t(1), slot1, 1, 1, 8);
  Duplicate(slotIndices, uint32_t(2), slot2, 1, 1, 8);
  Muls(pointOffsets.ReinterpretCast<int32_t>(), slotIndices.ReinterpretCast<int32_t>(),
       int32_t(12), 64);
  Muls(slotIndices.ReinterpretCast<int32_t>(), slotIndices.ReinterpretCast<int32_t>(),
       int32_t(8), 64);
  DeviceOffsets(segmentOffsets, 64, 8);
  // lane = 3*segment + slot, hence byteOffset = 8*(lane-slot).
  Sub(segmentOffsets.ReinterpretCast<int32_t>(), segmentOffsets.ReinterpretCast<int32_t>(),
      slotIndices.ReinterpretCast<int32_t>(), 64);
  Fence<HardEvent::MTE2_V>();
  Gather(ax, raw, segmentOffsets, 0, 64);
  Gather(ay, raw, segmentOffsets, 4, 64);
  Gather(az, raw, segmentOffsets, 8, 64);
  Gather(dx, raw, segmentOffsets, 12, 64);
  Gather(dy, raw, segmentOffsets, 16, 64);
  Gather(dz, raw, segmentOffsets, 20, 64);
  Sub(dx, dx, ax, 64);
  Sub(dy, dy, ay, 64);
  Sub(dz, dz, az, 64);
  Mul(reciprocal, dx, dx, 64);
  Mul(temporary, dy, dy, 64);
  Add(reciprocal, reciprocal, temporary, 64);
  Mul(temporary, dz, dz, 64);
  Add(reciprocal, reciprocal, temporary, 64);
  auto positiveLength = reduction.ReinterpretCast<uint8_t>();
  CompareScalar(positiveLength, reciprocal, 0.0f, CMPMODE::GT, 64);
  Select(reciprocal, positiveLength, reciprocal, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
  Duplicate(temporary, 1.0f, 64);
  Div(reciprocal, temporary, reciprocal, 64);
  Gather(x, pointInput, pointOffsets, 0, 64);
  Gather(y, pointInput, pointOffsets, 4, 64);
  Gather(z, pointInput, pointOffsets, 8, 64);
  Sub(x, ax, x, 64);
  Sub(y, ay, y, 64);
  Sub(z, az, z, 64);
  Mul(projection, x, dx, 64);
  MulAddDst(projection, y, dy, 64);
  MulAddDst(projection, z, dz, 64);
  Mul(projection, projection, reciprocal, 64);
  Maxs(projection, projection, -1.0f, 64);
  Mins(projection, projection, 0.0f, 64);
  Mul(temporary, projection, dx, 64);
  Sub(x, x, temporary, 64);
  Mul(temporary, projection, dy, 64);
  Sub(y, y, temporary, 64);
  Mul(temporary, projection, dz, 64);
  Sub(z, z, temporary, 64);
  Mul(temporary, x, x, 64);
  MulAddDst(temporary, y, y, 64);
  MulAddDst(temporary, z, z, 64);
  uint64_t validPairMask = (uint64_t(1) << (segmentCount * 3)) - 1;
  slot0[0] &= validPairMask;
  slot1[0] &= validPairMask;
  slot2[0] &= validPairMask;
  WholeReduceMin(reduction, temporary, slot0, 1, 1, 1, 8);
  WholeReduceMin(reduction[2], temporary, slot1, 1, 1, 1, 8);
  WholeReduceMin(reduction[4], temporary, slot2, 1, 1, 1, 8);
  SetMaskCount();
  SetVectorMask<float, MaskMode::COUNTER>(pointCount * 2);
  GatherMask(x, reduction, uint8_t(1), {1, 1, 8, 1});
  GatherMask(y, reduction, uint8_t(2), {1, 1, 8, 1});
  SetMaskNorm();
  ResetMask();
  // All selected indices are lanes0..63; floor(43*lane/128) is exactly lane/3.
  Muls(y.ReinterpretCast<int32_t>(), y.ReinterpretCast<int32_t>(), int32_t(43), pointCount);
  ShiftRight(y.ReinterpretCast<uint32_t>(), y.ReinterpretCast<uint32_t>(), uint32_t(7), pointCount);
  Sqrt(x, x, pointCount);
  Fence<HardEvent::V_MTE3>();
  DataCopyPad(distances, x, {1, pointCount * 4, 0, 0, 0});
  DataCopyPad(indices, y.ReinterpretCast<int32_t>(), {1, pointCount * 4, 0, 0, 0});
  Fence<HardEvent::MTE3_V>();
}

// Pack several point rows into a full 64-float SIMD iteration.
template <typename T, uint32_t segmentLaneCount, bool FusedDotProduct = false> class PackedRowsTreeLine {
  // A vector repeat contains 64 FP32 lanes. Each point row must fit uint8 repeat fields;
  // these capacities also keep the six dense planes within the available UB budget.
  static constexpr uint32_t pointRowCapacity =
                 segmentLaneCount == 2 ? 224 : (segmentLaneCount <= 16 ? 240 : 216),
               pairCapacity = pointRowCapacity * segmentLaneCount,
               planePitchElements = pairCapacity + 64,
               pointRowsPerVector = 64 / segmentLaneCount;
  // Sub-block rows retain compensated quotient reconstruction. Rows of
  // eight or more segments use the same reciprocal/residual arithmetic as
  // the original general vector path, with all intermediates in FP32.
  static constexpr bool usesCompensatedDivision = segmentLaneCount < 8;
  GlobalTensor<T> pg, sg, og;
  GlobalTensor<int32_t> ig;
  LocalTensor<float> mem, ax, ay, az, dx, dy, dz, inv, raw, x, y, z, dot, tmp, brc, group, seed, red,
    segmentSquaredLengths, pointInput;
  LocalTensor<uint32_t> segOff, pointOff, pointPackingMask;
  uint32_t segmentCount, pointsPerBatch, livePointCount, vectorRepeatCount, activeLaneCount;
  __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<T> src, uint32_t items,
                LocalTensor<float> scratch) {
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(dst, src, {1, items * 4, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    } else {
      auto h = scratch.ReinterpretCast<half>();
      DataCopyPad(h, src, {1, items * 2, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(dst, h, RoundMode::CAST_NONE, items);
    }
  }
  __aicore__ inline void Product(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b) {
    Mul(dst, a, b, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
  }
  __aicore__ inline void PointBroadcast(uint32_t axis) {
    if constexpr (segmentLaneCount < 8) {
      uint32_t paddedPointRows = vectorRepeatCount * pointRowsPerVector;
      Brcb(seed, group[axis * pointRowCapacity], uint8_t(paddedPointRows / 8), {1, 8});
      // Brcb creates eight copies; retain W of each. The packed row
      // count is known, so keep counter semantics without reading its SPR.
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(int32_t(paddedPointRows * 8));
      GatherMask(brc, seed, pointPackingMask.ReinterpretCast<float>(), uint8_t(0), {1, 1, 8, 1});
      SetMaskNorm();
      ResetMask();
    } else {
      Brcb(seed, group[axis * pointRowCapacity], uint8_t((livePointCount + 7) / 8), {1, 8});
      Copy(brc, seed, uint64_t(segmentLaneCount),
        uint8_t((livePointCount + pointRowsPerVector - 1) / pointRowsPerVector *
            pointRowsPerVector),
        {1, 0, segmentLaneCount / 8, 1});
    }
  }
  // A dedicated point staging tile fits after the live offset table. Once
  // all three coordinate gathers complete, the next DMA may replace it while
  // geometry and output DMA use disjoint planes.
  __aicore__ inline void PrefetchPointTile(uint32_t pointBegin, uint32_t pointCount) {
    DataCopyPad(pointInput.ReinterpretCast<T>(), pg[pointBegin * 3],
          {1, pointCount * uint32_t(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  }
  __aicore__ inline void StartInitialInputPrefetch(uint32_t batch, uint32_t pointBegin,
                          uint32_t pointCount) {
    Duplicate(raw, 0.0f, 256);
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(raw, sg[batch * segmentCount * 6],
            {1, segmentCount * 24, 0, 0, 0}, {false, 0, 0, 0});
    } else {
      DataCopyPad(brc.ReinterpretCast<half>(), sg[batch * segmentCount * 6],
            {1, segmentCount * 12, 0, 0, 0}, {false, 0, 0, 0});
    }
    PrefetchPointTile(pointBegin, pointCount);
  }
  __aicore__ inline void FinishInitialInputPrefetch() {
    Fence<HardEvent::MTE2_V>();
    if constexpr (IsSameType<T, half>::value)
      Cast(raw, brc.ReinterpretCast<half>(), RoundMode::CAST_NONE, segmentCount * 6);
  }
  __aicore__ inline void Segments(uint32_t batch, bool hasPrefetchedInputs = false) {
    if (!hasPrefetchedInputs) {
      Duplicate(raw, 0.0f, 256);
      Load(raw, sg[batch * segmentCount * 6], segmentCount * 6, brc);
    }
    LocalTensor<float> cols[6] = {ax, ay, az, dx, dy, dz};
    for (uint32_t a = 0; a < 6; ++a) {
      if constexpr (segmentLaneCount < 8) {
        Gather(cols[a], raw, segOff, a * 4, 64);
      } else {
        Gather(cols[a], raw, segOff, a * 4, segmentLaneCount);
        if constexpr (pointRowsPerVector > 1)
          Copy(cols[a][segmentLaneCount], cols[a], uint64_t(segmentLaneCount),
            uint8_t(pointRowsPerVector - 1), {1, 1, segmentLaneCount / 8, 0});
      }
    }
    Sub(dx, dx, ax, 64);
    Sub(dy, dy, ay, 64);
    Sub(dz, dz, az, 64);
    Mul(inv, dx, dx, 64);
    Mul(tmp, dy, dy, 64);
    Add(inv, inv, tmp, 64);
    Mul(tmp, dz, dz, 64);
    Add(inv, inv, tmp, 64);
    auto mask = red.ReinterpretCast<uint8_t>();
    CompareScalar(mask, inv, 0.0f, CMPMODE::GT, 64);
    Select(inv, mask, inv, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
    if constexpr (usesCompensatedDivision)
      Adds(segmentSquaredLengths, inv, 0.0f, 64);
    if constexpr (segmentLaneCount >= 8) {
      // Preserve the existing W>=8 setup; W<8 consumes L directly.
      Duplicate(tmp, 1.0f, 64);
      Div(inv, tmp, inv, 64);
    }
  }
  __aicore__ inline void Compute() {
    LocalTensor<float> delta[3] = {x, y, z}, starts[3] = {ax, ay, az}, directions[3] = {dx, dy, dz};
    for (uint32_t a = 0; a < 3; ++a) {
      PointBroadcast(a);
      Sub(delta[a], starts[a], brc, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 0, 8});
    }
    Product(dot, x, dx);
    if constexpr (FusedDotProduct) {
      MulAddDst(dot, y, dy, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
      MulAddDst(dot, z, dz, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
    } else {
      Product(tmp, y, dy);
      Add(dot, dot, tmp, activeLaneCount);
      Product(tmp, z, dz);
      Add(dot, dot, tmp, activeLaneCount);
    }
    if constexpr (usesCompensatedDivision) {
      Div(tmp, dot, segmentSquaredLengths, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
      Muls(x, tmp, -1.0f, activeLaneCount);
      MulAddDst(dot, x, segmentSquaredLengths, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
      Div(dot, dot, segmentSquaredLengths, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
      Add(dot, tmp, dot, activeLaneCount);
    } else {
      Product(dot, dot, inv);
    }
    Maxs(dot, dot, -1.0f, activeLaneCount);
    Mins(dot, dot, 0.0f, activeLaneCount);
    if constexpr (usesCompensatedDivision) {
      for (uint32_t a = 0; a < 3; ++a) {
        Product(delta[a], dot, directions[a]);
        Sub(delta[a], starts[a], delta[a], uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 0, 8});
        PointBroadcast(a);
        Sub(delta[a], brc, delta[a], activeLaneCount);
      }
      Mul(tmp, x, x, activeLaneCount);
      Mul(y, y, y, activeLaneCount);
      Add(tmp, tmp, y, activeLaneCount);
      Mul(z, z, z, activeLaneCount);
      Add(tmp, tmp, z, activeLaneCount);
    } else {
      for (uint32_t a = 0; a < 3; ++a) {
        Product(tmp, dot, directions[a]);
        Sub(delta[a], delta[a], tmp, activeLaneCount);
      }
      Mul(tmp, x, x, activeLaneCount);
      MulAddDst(tmp, y, y, activeLaneCount);
      MulAddDst(tmp, z, z, activeLaneCount);
    }
    auto reductionInput = tmp;
    if constexpr (usesCompensatedDivision) {
      // Preserve the sub-block row reduction layout. Other rows
      // consume the products directly and avoid one dense copy.
      Adds(x, tmp, 0.0f, activeLaneCount);
      reductionInput = x;
    }
    if constexpr (segmentLaneCount < 8) {
      // Reduce independent sub-block point rows without requiring a 16B/8B repeat stride.
      constexpr uint32_t pointRowsPerBlock = 8 / segmentLaneCount;
#pragma unroll
      for (uint32_t slot = 0; slot < pointRowsPerBlock; ++slot) {
        uint64_t laneMask[2] = {((uint64_t(1) << segmentCount) - 1) << (slot * segmentLaneCount),
                    0};
        WholeReduceMin(red[2 * slot], reductionInput, laneMask,
               uint8_t((livePointCount + pointRowsPerBlock - 1) / pointRowsPerBlock),
               pointRowsPerBlock, 1, 1);
      }
    } else {
      WholeReduceMin(red, reductionInput, int32_t(segmentCount), livePointCount, 1, 1,
             segmentLaneCount / 8);
    }
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(livePointCount * 2);
    GatherMask(x, red, uint8_t(1), {1, 1, 8, 1});
    GatherMask(y, red, uint8_t(2), {1, 1, 8, 1});
    SetMaskNorm();
    ResetMask();
    if constexpr (segmentLaneCount < 8) {
      constexpr uint32_t indexShiftBits = segmentLaneCount == 2 ? 31 : 30;
      ShiftLeft(y.ReinterpretCast<uint32_t>(), y.ReinterpretCast<uint32_t>(), indexShiftBits,
           livePointCount);
      ShiftRight(y.ReinterpretCast<uint32_t>(), y.ReinterpretCast<uint32_t>(), indexShiftBits,
           livePointCount);
    }
    Sqrt(x, x, livePointCount);
  }

 public:
  __aicore__ inline void Run(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR oi,
               const TreeLineMinDistanceTilingData &t) {
    if constexpr (segmentLaneCount == 32 && IsSameType<T, float>::value) {
      uint32_t totalPoints = t.batchCount * t.pointsPerBatch;
      uint32_t pointBegin = totalPoints * GetBlockIdx() / t.workerCount;
      uint32_t pointEnd = totalPoints * (GetBlockIdx() + 1) / t.workerCount;
      if (pointEnd > pointBegin && pointEnd - pointBegin <= 3 && t.segmentCount * 3 <= 64 &&
          pointBegin / t.pointsPerBatch == (pointEnd - 1) / t.pointsPerBatch) {
        uint32_t batchIndex = pointBegin / t.pointsPerBatch;
        ComputeThreeInterleavedFloatQueries(
          p + uint64_t(pointBegin) * 12, s + uint64_t(batchIndex) * t.segmentCount * 24,
          o + uint64_t(pointBegin) * 4, oi + uint64_t(pointBegin) * 4,
          pointEnd - pointBegin, t.segmentCount);
        return;
      }
    }
    pg.SetGlobalBuffer((__gm__ T *)p);
    sg.SetGlobalBuffer((__gm__ T *)s);
    og.SetGlobalBuffer((__gm__ T *)o);
    ig.SetGlobalBuffer((__gm__ int32_t *)oi);
    segmentCount = t.segmentCount;
    pointsPerBatch = t.pointsPerBatch;
    mem = LocalTensor<float>(TPosition::VECCALC, 0, (t.unifiedBufferBytes - 8192) / 4);
    ax = mem;
    ay = mem[64];
    az = mem[128];
    dx = mem[192];
    dy = mem[256];
    dz = mem[320];
    inv = mem[384];
    segOff = mem[448].ReinterpretCast<uint32_t>();
    raw = mem[512];
    segmentSquaredLengths = mem[832];
    x = mem[1024];
    y = x[planePitchElements];
    z = x[2 * planePitchElements];
    dot = x[3 * planePitchElements];
    tmp = x[4 * planePitchElements];
    brc = x[5 * planePitchElements];
    group = x[6 * planePitchElements];
    seed = group[3 * pointRowCapacity];
    red = seed[8 * pointRowCapacity];
    pointOff = red[512].ReinterpretCast<uint32_t>();
    pointInput = pointOff[segmentLaneCount < 8 ? 2 * pointRowCapacity + 64
                         : pointRowCapacity].ReinterpretCast<float>();
    uint32_t total = t.batchCount * pointsPerBatch, begin = total * GetBlockIdx() / t.workerCount,
        end = total * (GetBlockIdx() + 1) / t.workerCount, oldBatch = 0xffffffff;
    uint32_t firstPointCount = end - begin;
    if (firstPointCount > pointRowCapacity)
      firstPointCount = pointRowCapacity;
    uint32_t firstBatch = begin / pointsPerBatch;
    uint32_t firstBatchRemaining = (firstBatch + 1) * pointsPerBatch - begin;
    if (firstPointCount > firstBatchRemaining)
      firstPointCount = firstBatchRemaining;
    StartInitialInputPrefetch(firstBatch, begin, firstPointCount);
    if constexpr (segmentLaneCount < 8) {
      pointPackingMask = red[512 + 2 * pointRowCapacity].ReinterpretCast<uint32_t>();
      constexpr uint32_t packingByte = (1U << segmentLaneCount) - 1U;
      Duplicate(pointPackingMask, packingByte * uint32_t(0x01010101), 64);
      Fence<HardEvent::V_S>();
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i)
        segOff.SetValue(i, (i % segmentLaneCount) * 24);
      Fence<HardEvent::S_V>();
#pragma unroll
      for (uint32_t prefixElements = 8; prefixElements < 64; prefixElements *= 2) {
        Adds(segOff[prefixElements].ReinterpretCast<int32_t>(), segOff.ReinterpretCast<int32_t>(),
          int32_t(0), prefixElements);
        PipeBarrier<PIPE_V>();
      }
    }

    uint32_t offsetPointCount = end - begin;
    if (offsetPointCount > pointRowCapacity)
      offsetPointCount = pointRowCapacity;
    // Only live point rows index these tables; a batch boundary can shorten
    // a tile but cannot enlarge it beyond this worker's entire point slice.
    if constexpr (segmentLaneCount >= 8) {
      SeedDeviceOffsets(segOff, segmentLaneCount, 24);
      SeedDeviceOffsets(pointOff, offsetPointCount, 12);
      Fence<HardEvent::S_V>();
      ExpandDeviceOffsetSeeds(segOff, segmentLaneCount, 24);
      ExpandDeviceOffsetSeeds(pointOff, offsetPointCount, 12);
    } else {
      DeviceOffsets(pointOff, offsetPointCount, 12);
    }
    if constexpr (segmentLaneCount >= 16)
      Duplicate(group, 0.0f, 3 * pointRowCapacity);
    for (uint32_t at = begin; at < end;) {
      uint32_t batch = at / pointsPerBatch;
      if (batch != oldBatch) {
        if (at == begin)
          FinishInitialInputPrefetch();
        Segments(batch, at == begin);
        oldBatch = batch;
      }
      livePointCount = end - at;
      if (livePointCount > pointRowCapacity)
        livePointCount = pointRowCapacity;
      if (livePointCount > (batch + 1) * pointsPerBatch - at)
        livePointCount = (batch + 1) * pointsPerBatch - at;
      vectorRepeatCount = (livePointCount + pointRowsPerVector - 1) / pointRowsPerVector;
      activeLaneCount = vectorRepeatCount * 64;
      // Wide rows reduce independently. Define padding once and
      // overwrite every live coordinate on each tile; padded rows never
      // enter output reduction. Narrow packing retains its full reset.
      if constexpr (segmentLaneCount < 16)
        Duplicate(group, 0.0f, 3 * pointRowCapacity);
      if (at != begin)
        Fence<HardEvent::MTE2_V>();
      auto pointCoordinates = pointInput;
      if constexpr (IsSameType<T, half>::value) {
        Cast(x, pointInput.ReinterpretCast<half>(), RoundMode::CAST_NONE, livePointCount * 3);
        pointCoordinates = x;
      }
      Gather(group, pointCoordinates, pointOff, 0, livePointCount);
      Gather(group[pointRowCapacity], pointCoordinates, pointOff, 4, livePointCount);
      Gather(group[2 * pointRowCapacity], pointCoordinates, pointOff, 8, livePointCount);
      uint32_t nextPoint = at + livePointCount;
      if (nextPoint < end) {
        uint32_t nextPointCount = end - nextPoint;
        if (nextPointCount > pointRowCapacity)
          nextPointCount = pointRowCapacity;
        uint32_t nextBatchEnd = (nextPoint / pointsPerBatch + 1) * pointsPerBatch;
        if (nextPointCount > nextBatchEnd - nextPoint)
          nextPointCount = nextBatchEnd - nextPoint;
        Fence<HardEvent::V_MTE2>();
        PrefetchPointTile(nextPoint, nextPointCount);
      }
      Compute();
      if constexpr (IsSameType<T, float>::value) {
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(og[at], x, {1, livePointCount * 4, 0, 0, 0});
      } else {
        auto h = x.ReinterpretCast<half>();
        Cast(h, x, RoundMode::CAST_RINT, livePointCount);
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(og[at], h, {1, livePointCount * 2, 0, 0, 0});
      }
      DataCopyPad(ig[at], y.ReinterpretCast<int32_t>(), {1, livePointCount * 4, 0, 0, 0});
      Fence<HardEvent::MTE3_V>();
      at += livePointCount;
    }
  }
};
template <typename T, uint32_t ALGORITHM> class KernelTreeLineMinDistance {
  static constexpr uint32_t STATIC_WIDTH =
    ALGORITHM >= 24 ? 8U * (ALGORITHM - 23)
            : (ALGORITHM >= 13 && ALGORITHM <= 16 ? (128U << (ALGORITHM - 13)) : 0);
  uint32_t bufferBase = 0;
  LocalTensor<float> outLocal;
  uint32_t outStart;
  LocalTensor<int32_t> idxLocal;
  GlobalTensor<T> points, segments, output;
  GlobalTensor<int32_t> indicesOut;
  TreeLineMinDistanceTilingData t;
  uint32_t width, rows, pairs;
  LocalTensor<float> raw, ax, ay, az, dx, dy, dz, len, x, y, z, dot, tmp, dist, red;
  LocalTensor<uint32_t> gather;
  __aicore__ inline void SeedOffsetTable(LocalTensor<uint32_t> destination,
                       uint32_t elementCount, uint32_t byteStride) {
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      uint32_t seedElements = elementCount < 8 ? elementCount : 8;
      auto pairedSeeds = destination.ReinterpretCast<uint64_t>();
      // Offset tables are32-byte aligned. Little-endian paired stores keep
      // the existing byte offsets; all table contents are generated here.
      uint64_t packedPair = uint64_t(byteStride) << 32;
      uint64_t packedStep = (uint64_t(byteStride * 2) << 32) | uint64_t(byteStride * 2);
      for (uint32_t pair = 0; pair < seedElements / 2; ++pair) {
        pairedSeeds.SetValue(pair, packedPair);
        packedPair += packedStep;
      }
      if (seedElements & 1)
        destination.SetValue(seedElements - 1, (seedElements - 1) * byteStride);
    } else {
      SeedDeviceOffsets(destination, elementCount, byteStride);
    }
  }
  __aicore__ inline void Store(uint32_t i, float d, int32_t arg) {
    outLocal.SetValue(i - outStart, d);
    idxLocal.SetValue(i - outStart, arg);
  }
  __aicore__ inline void Flush(uint32_t count) {
    Fence<HardEvent::S_V>();
    Sqrt(outLocal, outLocal, count);
    if constexpr (IsSameType<T, float>::value) {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(output[outStart], outLocal, {1, count * 4, 0, 0, 0});
    } else {
      auto halfout = outLocal.ReinterpretCast<half>();
      Cast(halfout, outLocal, RoundMode::CAST_RINT, count);
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(output[outStart], halfout, {1, count * 2, 0, 0, 0});
    }
    DataCopyPad(indicesOut[outStart], idxLocal, {1, count * 4, 0, 0, 0});
    Fence<HardEvent::MTE3_S>();
  }
  __aicore__ inline void Scalar(uint32_t start, uint32_t end) {
    for (uint32_t i = start; i < end; ++i) {
      uint32_t batch = i / t.pointsPerBatch;
      float px = float(points.GetValue(i * 3)), py = float(points.GetValue(i * 3 + 1)),
         pz = float(points.GetValue(i * 3 + 2));
      float best = 3.402823466e38f;
      int32_t arg = 0;
      for (uint32_t j = 0; j < t.segmentCount; ++j) {
        uint32_t k = (batch * t.segmentCount + j) * 6;
        float a = float(segments.GetValue(k)), b = float(segments.GetValue(k + 1)),
           c = float(segments.GetValue(k + 2));
        float u = float(segments.GetValue(k + 3)) - a, v = float(segments.GetValue(k + 4)) - b,
           w = float(segments.GetValue(k + 5)) - c;
        float l = (u * u + v * v) + w * w, q = 0;
        if (l > 0)
          q = (((px - a) * u + (py - b) * v) + (pz - c) * w) / l;
        q = q < 0 ? 0 : (q > 1 ? 1 : q);
        float ex = px - (a + q * u), ey = py - (b + q * v), ez = pz - (c + q * w);
        float d = (ex * ex + ey * ey) + ez * ez;
        if (d < best) {
          best = d;
          arg = j;
        }
      }
      Store(i, best, arg);
    }
  }
  __aicore__ inline void PrefetchSegmentInputs(uint32_t batch) {
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      // A full segment tile is overwritten by this invocation's DMA/Cast
      // before Gather. Partial tiles still initialize every padded lane.
      if (t.segmentCount < width)
        Duplicate(raw, 0.0f, width * 6);
    } else {
      Duplicate(raw, 0.0f, width * 6);
    }
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(raw, segments[batch * t.segmentCount * 6], {1, t.segmentCount * 24, 0, 0, 0},
            {false, 0, 0, 0});
    } else {
      auto halfraw = raw.ReinterpretCast<half>()[width * 12];
      DataCopyPad(halfraw, segments[batch * t.segmentCount * 6], {1, t.segmentCount * 12, 0, 0, 0},
            {false, 0, 0, 0});
    }
  }
  __aicore__ inline void PrefetchPointInputs(uint32_t pointBegin, uint32_t pointCount) {
    if constexpr (IsSameType<T, float>::value)
      DataCopyPad(red[512], points[pointBegin * 3], {1, pointCount * 12, 0, 0, 0},
            {false, 0, 0, 0});
    else
      DataCopyPad(red[768].ReinterpretCast<half>(), points[pointBegin * 3],
            {1, pointCount * 6, 0, 0, 0}, {false, 0, 0, 0});
  }
  __aicore__ inline void LoadSegments(uint32_t batch, bool inputsAlreadyStaged = false) {
    if (!inputsAlreadyStaged)
      PrefetchSegmentInputs(batch);
    Fence<HardEvent::MTE2_V>();
    if constexpr (IsSameType<T, half>::value) {
      auto halfraw = raw.ReinterpretCast<half>()[width * 12];
      Cast(raw, halfraw, RoundMode::CAST_NONE, t.segmentCount * 6);
    }
    Gather(ax, raw, gather, 0, width);
    Gather(ay, raw, gather, 4, width);
    Gather(az, raw, gather, 8, width);
    Gather(dx, raw, gather, 12, width);
    Gather(dy, raw, gather, 16, width);
    Gather(dz, raw, gather, 20, width);
    Sub(dx, dx, ax, width);
    Sub(dy, dy, ay, width);
    Sub(dz, dz, az, width);
    Mul(len, dx, dx, width);
    Mul(tmp, dy, dy, width);
    Mul(dot, dz, dz, width);
    Add(len, len, tmp, width);
    Add(len, len, dot, width);
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      // Only an exactly zero squared length receives a safe denominator.
      // The mask uses reduction scratch, disjoint from staged input data.
      auto nonzeroLengthMask = red.ReinterpretCast<uint8_t>();
      CompareScalar(nonzeroLengthMask, len, 0.0f, CMPMODE::NE, width);
      Select(len, nonzeroLengthMask, len, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, width);
    } else {
    // Repair exactly zero nonnegative squared lengths without a vector select.
    Mins(tmp.ReinterpretCast<int32_t>(), len.ReinterpretCast<int32_t>(), int32_t(1), width);
    Muls(tmp.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(-0x3f800000), width);
    Adds(tmp.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(0x3f800000), width);
    Or(len.ReinterpretCast<uint16_t>(), len.ReinterpretCast<uint16_t>(),
     tmp.ReinterpretCast<uint16_t>(), width * 2);
    }
    // A batch shares segment lengths across every point tile.
    Duplicate(tmp, 1.0f, width);
    Div(len, tmp, len, width);
  }
  __aicore__ inline void PointVector(uint32_t start, uint32_t end) {
    uint32_t w = (end - start + 63) / 64 * 64;
    if (w > 2048)
      w = 2048;
    // A64-float gap separates the origins of simultaneously read planes.
    // Scalar invariant caching avoids the old full-width length broadcasts.
    bool usesPaddedPointPlanes = IsSameType<T, float>::value &&
      t.segmentCount == 1 && w >= 1024 && (w % 128) == 0;
    uint32_t planePitchElements = w + (usesPaddedPointPlanes ? 64 : 0);
    LocalTensor<float> mem(TPosition::VECCALC, bufferBase,
               (t.unifiedBufferBytes - 8192 - bufferBase) / 4);
    raw = mem;
    x = mem[6 * planePitchElements];
    y = mem[7 * planePitchElements];
    z = mem[8 * planePitchElements];
    ax = mem[9 * planePitchElements];
    ay = mem[10 * planePitchElements];
    az = mem[11 * planePitchElements];
    dot = mem[12 * planePitchElements];
    tmp = mem[13 * planePitchElements];
    dist = mem[14 * planePitchElements];
    auto best = mem[15 * planePitchElements];
    auto arg = mem[16 * planePitchElements];
    auto next = mem[17 * planePitchElements];
    gather = mem[18 * planePitchElements].template ReinterpretCast<uint32_t>();
    auto mask = mem[19 * planePitchElements].template ReinterpretCast<uint8_t>();
    auto segmentLocal = mem[19 * planePitchElements + 256];
    bool usesScalarM1Invariants = IsSameType<T, float>::value && t.segmentCount == 1 && w >= 512;
    float cachedA = 0.0f, cachedB = 0.0f, cachedC = 0.0f;
    float cachedU = 0.0f, cachedV = 0.0f, cachedW = 0.0f;
    float cachedSquaredLength = 1.0f, cachedInverseLength = 1.0f;
    if (usesPaddedPointPlanes) {
      // The single-segment route has no argmin state in these two planes.
      // Its norm accumulator reads residuals from different bank origins.
      tmp = mem[14 * planePitchElements];
      dist = mem[16 * planePitchElements];
    }
    uint32_t firstPointCount = end - start;
    if (firstPointCount > w)
      firstPointCount = w;
    uint32_t firstBatchRemaining = t.pointsPerBatch - start % t.pointsPerBatch;
    if (firstPointCount > firstBatchRemaining)
      firstPointCount = firstBatchRemaining;
    // The first point DMA is independent of offset generation and the
    // segment table. Every live RGB coordinate is written by current
    // DMA/Cast before Gather, so no redundant raw reset is needed.
    // All input transfers and arithmetic stay inside this timed kernel.
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value)
      DataCopyPad(raw, points[start * 3], {1, firstPointCount * 12, 0, 0, 0}, {false, 0, 0, 0});
    else
      DataCopyPad(raw.ReinterpretCast<half>()[w * 6], points[start * 3],
            {1, firstPointCount * 6, 0, 0, 0}, {false, 0, 0, 0});
    DeviceOffsets(gather, w, 12);
    uint32_t oldBatch = 0xffffffff;
    for (uint32_t i = start; i < end;) {
      uint32_t batch = i / t.pointsPerBatch, count = end - i;
      if (count > w)
        count = w;
      if (usesScalarM1Invariants && batch != oldBatch) {
        uint32_t k = batch * 6;
        cachedA = float(segments.GetValue(k));
        cachedB = float(segments.GetValue(k + 1));
        cachedC = float(segments.GetValue(k + 2));
        cachedU = float(segments.GetValue(k + 3)) - cachedA;
        cachedV = float(segments.GetValue(k + 4)) - cachedB;
        cachedW = float(segments.GetValue(k + 5)) - cachedC;
        float squaredLength = (cachedU * cachedU + cachedV * cachedV) + cachedW * cachedW;
        // Preserve the original FP32 vector FMA and reciprocal correction.
        // This prefix overlaps the first point DMA; scratch is disjoint from raw input.
        Duplicate(next, cachedU, 64);
        Mul(next, next, next, 64);
        Duplicate(tmp, cachedV, 64);
        Axpy(next, tmp, cachedV, 64);
        Duplicate(tmp, cachedW, 64);
        Axpy(next, tmp, cachedW, 64);
        if (squaredLength == 0.0f)
          Duplicate(next, 1.0f, 64);
        Duplicate(tmp, 1.0f, 64);
        Div(tmp, tmp, next, 64);
        Duplicate(segmentLocal, 1.0f, 64);
        Muls(dist, tmp, -1.0f, 64);
        MulAddDst(segmentLocal, dist, next, 64);
        MulAddDst(tmp, tmp, segmentLocal, 64);
        Fence<HardEvent::V_S>();
        cachedSquaredLength = next.GetValue(0);
        cachedInverseLength = tmp.GetValue(0);
        oldBatch = batch;
      }
      // All points in a batch share the same short segment table.
      if (t.segmentCount != 1 && batch != oldBatch) {
        Fence<HardEvent::V_MTE2>();
        if constexpr (IsSameType<T, float>::value) {
          DataCopyPad(segmentLocal, segments[batch * t.segmentCount * 6],
                {1, t.segmentCount * 24, 0, 0, 0}, {false, 0, 0, 0});
          Fence<HardEvent::MTE2_S>();
        } else {
          auto halfSegments = segmentLocal[64].ReinterpretCast<half>();
          DataCopyPad(halfSegments, segments[batch * t.segmentCount * 6],
                {1, t.segmentCount * 12, 0, 0, 0}, {false, 0, 0, 0});
          Fence<HardEvent::MTE2_V>();
          Cast(segmentLocal, halfSegments, RoundMode::CAST_NONE, t.segmentCount * 6);
          Fence<HardEvent::V_S>();
        }
        oldBatch = batch;
      }
      uint32_t remain = (batch + 1) * t.pointsPerBatch - i;
      if (count > remain)
        count = remain;
      uint32_t live = count;
      count = (count + 63) / 64 * 64;
      if (i != start) {
        Fence<HardEvent::V_MTE2>();
        if constexpr (IsSameType<T, float>::value)
          DataCopyPad(raw, points[i * 3], {1, live * 12, 0, 0, 0}, {false, 0, 0, 0});
        else
          DataCopyPad(raw.ReinterpretCast<half>()[w * 6], points[i * 3],
                {1, live * 6, 0, 0, 0}, {false, 0, 0, 0});
      }
      Fence<HardEvent::MTE2_V>();
      if constexpr (IsSameType<T, half>::value)
        Cast(raw, raw.ReinterpretCast<half>()[w * 6], RoundMode::CAST_NONE, live * 3);
      Gather(x, raw, gather, 0, live);
      Gather(y, raw, gather, 4, live);
      Gather(z, raw, gather, 8, live);
      if (t.segmentCount != 1) {
        Duplicate(best, 3.402823466e38f, count);
        Duplicate(arg, 0.0f, count);
      }
      for (uint32_t j = 0; j < t.segmentCount; ++j) {
        float a, b, c, u, v, q;
        if (usesScalarM1Invariants) {
          a = cachedA; b = cachedB; c = cachedC;
          u = cachedU; v = cachedV; q = cachedW;
        } else if (t.segmentCount == 1) {
          uint32_t k = batch * 6;
          a = float(segments.GetValue(k));
          b = float(segments.GetValue(k + 1));
          c = float(segments.GetValue(k + 2));
          u = float(segments.GetValue(k + 3)) - a;
          v = float(segments.GetValue(k + 4)) - b;
          q = float(segments.GetValue(k + 5)) - c;
        } else {
          uint32_t k = j * 6;
          a = segmentLocal.GetValue(k);
          b = segmentLocal.GetValue(k + 1);
          c = segmentLocal.GetValue(k + 2);
          u = segmentLocal.GetValue(k + 3) - a;
          v = segmentLocal.GetValue(k + 4) - b;
          q = segmentLocal.GetValue(k + 5) - c;
        }
        float l = (u * u + v * v) + q * q;
        Adds(ax, x, -a, count);
        Adds(ay, y, -b, count);
        Adds(az, z, -c, count);
        if (t.segmentCount == 1) {
          Muls(dot, ax, u, count);
          Axpy(dot, ay, v, count);
          Axpy(dot, az, q, count);
          if (usesScalarM1Invariants) {
            CompareScalar(mask, dot, cachedSquaredLength, CMPMODE::LT, count);
            Muls(dot, dot, cachedInverseLength, count);
          } else {
          // Segment invariants use the same vector arithmetic as the dot product.
          Duplicate(next, u, 64);
          Mul(next, next, next, 64);
          Duplicate(tmp, v, 64);
          Axpy(next, tmp, v, 64);
          Duplicate(tmp, q, 64);
          Axpy(next, tmp, q, 64);
          if (l == 0.0f)
            Duplicate(next, 1.0f, 64);
          Duplicate(tmp, 1.0f, 64);
          Div(tmp, tmp, next, 64);
          Duplicate(raw[64], 1.0f, 64);
          Muls(dist, tmp, -1.0f, 64);
          MulAddDst(raw[64], dist, next, 64);
          MulAddDst(tmp, tmp, raw[64], 64);
          Copy(raw, next, uint64_t(64), uint8_t(count / 64), {1, 1, 8, 0});
          Copy(next, tmp, uint64_t(64), uint8_t(count / 64), {1, 1, 8, 0});
          Compare(mask, dot, raw, CMPMODE::LT, count);
          Mul(dot, dot, next, count);
          }
          Select(dot, mask, dot, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
          Maxs(dot, dot, 0.0f, count);
          Mins(dot, dot, 1.0f, count);
          Axpy(ax, dot, -u, count);
          Axpy(ay, dot, -v, count);
          Axpy(az, dot, -q, count);
          Mul(dist, ax, ax, count);
          MulAddDst(dist, ay, ay, count);
          MulAddDst(dist, az, az, count);
        } else {
          Muls(dot, ax, u, count);
          Muls(tmp, ay, v, count);
          Add(dot, dot, tmp, count);
          Muls(tmp, az, q, count);
          Add(dot, dot, tmp, count);
          Duplicate(next, u, count);
          Mul(next, next, next, count);
          Duplicate(tmp, v, count);
          Mul(tmp, tmp, tmp, count);
          Add(next, next, tmp, count);
          Duplicate(tmp, q, count);
          Mul(tmp, tmp, tmp, count);
          Add(next, next, tmp, count);
          if (l == 0.0f)
            Duplicate(next, 1.0f, count);
          Div(tmp, dot, next, count);
          Muls(dist, tmp, -1.0f, count);
          MulAddDst(dot, dist, next, count);
          Div(dot, dot, next, count);
          Add(dot, tmp, dot, count);
          Maxs(dot, dot, 0.0f, count);
          Mins(dot, dot, 1.0f, count);
          Muls(ax, dot, u, count);
          Muls(ay, dot, v, count);
          Muls(az, dot, q, count);
          Adds(ax, ax, a, count);
          Adds(ay, ay, b, count);
          Adds(az, az, c, count);
          Sub(ax, x, ax, count);
          Sub(ay, y, ay, count);
          Sub(az, z, az, count);
          Mul(ax, ax, ax, count);
          Mul(ay, ay, ay, count);
          Mul(az, az, az, count);
          Add(dist, ax, ay, count);
          Add(dist, dist, az, count);
        }
        if (t.segmentCount == 1)
          best = dist;
        else {
          Compare(mask, dist, best, CMPMODE::LT, count);
          Duplicate(next, float(int32_t(j)), count);
          Select(best, mask, dist, best, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
          Select(arg, mask, next, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        }
      }
      Sqrt(best, best, count);
      if (t.segmentCount == 1)
        Duplicate(next.template ReinterpretCast<int32_t>(), int32_t(0), count);
      else
        Cast(next.template ReinterpretCast<int32_t>(), arg, RoundMode::CAST_RINT, count);
      if constexpr (IsSameType<T, float>::value) {
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(output[i], best, {1, live * 4, 0, 0, 0});
      } else {
        auto halfout = best.ReinterpretCast<half>();
        Cast(halfout, best, RoundMode::CAST_RINT, count);
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(output[i], halfout, {1, live * 2, 0, 0, 0});
      }
      DataCopyPad(indicesOut[i], next.template ReinterpretCast<int32_t>(), {1, live * 4, 0, 0, 0});
      Fence<HardEvent::MTE3_V>();
      i += live;
    }
  }
  __aicore__ inline void Product(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b,
                 uint32_t live) {
    uint32_t ew = STATIC_WIDTH ? STATIC_WIDTH : width;
    if (ew <= 64 || live >= ew / 64) {
#pragma unroll
      for (uint32_t c = 0; c < ew; c += 64) {
        uint32_t mask = ew < 64 ? ew : 64;
        Mul(dst[c], a[c], b[c], uint64_t(mask), live,
          {1, 1, 1, uint8_t(ew / 8), uint8_t(ew / 8), 0});
      }
    } else {
      for (uint32_t r = 0; r < live; ++r)
        Mul(dst[r * ew], a[r * ew], b, ew);
    }
  }
  __aicore__ inline void Delta(LocalTensor<float> dst, LocalTensor<float> axis, uint32_t i, uint32_t live,
                uint32_t coordinate) {
    for (uint32_t r = 0; r < live; ++r) {
      float p = float(points.GetValue((i + r) * 3 + coordinate));
      Adds(dst[r * width], axis, -p, width);
    }
  }
  __aicore__ inline void Vector(uint32_t start, uint32_t end) {
    if constexpr (ALGORITHM >= 13)
      width = STATIC_WIDTH;
    else {
      width = (t.segmentCount + 7) / 8 * 8;
      if (width > 64) {
        width = 128;
        while (width < t.segmentCount)
          width *= 2;
      }
    }
    uint32_t outBytes = ((end - start + 15) / 16 * 16) * 8;
    uint32_t available = t.unifiedBufferBytes - outBytes - width * 32 - 14592;
    rows = available / (width * 20);
    if (rows > 64)
      rows = 64;
    if (rows < 1)
      rows = 1;
    if (rows > end - start)
      rows = end - start;
    if (rows > t.pointsPerBatch)
      rows = t.pointsPerBatch;
    if (width < 64)
      rows = (rows + 7) / 8 * 8;
    pairs = rows * width;
    uint32_t planePitch = pairs + 64, work = 5 * planePitch;
    if (work < 9 * width)
      work = 9 * width;
    LocalTensor<float> mem(TPosition::VECCALC, bufferBase,
               (t.unifiedBufferBytes - 8192 - bufferBase) / 4);
    ax = mem;
    ay = mem[width];
    az = mem[width * 2];
    dx = mem[width * 3];
    dy = mem[width * 4];
    dz = mem[width * 5];
    len = mem[width * 6];
    gather = mem[width * 7].template ReinterpretCast<uint32_t>();
    auto v = mem[width * 8];
    raw = v;
    x = v;
    y = v[planePitch];
    z = v[planePitch * 2];
    dot = v[planePitch * 3];
    tmp = v[planePitch * 4];
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      // Projection is dead after all three residuals have been formed.
      // Its plane is distinct from each residual source during the norm.
      dist = dot;
    } else {
      dist = tmp;
    }
    red = v[work];
    // Rounded allocation rows can exceed the actual point slice. Only
    // initialize point-gather offsets if a tile can consume them.
    bool usesPointCoordinateGather = rows >= 8 && end - start >= 8 && t.pointsPerBatch >= 8;
    uint32_t firstBatch = start / t.pointsPerBatch;
    uint32_t firstPointCount = end - start;
    if (firstPointCount > rows)
      firstPointCount = rows;
    uint32_t firstBatchRemaining = (firstBatch + 1) * t.pointsPerBatch - start;
    if (firstPointCount > firstBatchRemaining)
      firstPointCount = firstBatchRemaining;
    PrefetchSegmentInputs(firstBatch);
    bool hasStagedPointTile = firstPointCount >= 8;
    if (hasStagedPointTile)
      PrefetchPointInputs(start, firstPointCount);
    // These fresh offset tables are disjoint from each other and from
    // both DMA destinations. Seed them with one scalar-to-vector handoff.
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      // All three address tables are linear in the physical lane. Seed
      // one sequence here, then derive the current point/output/segment
      // byte strides on device. No table survives the invocation.
      SeedOffsetTable(gather, width, 1);
      Fence<HardEvent::S_V>();
      ExpandDeviceOffsetSeeds(gather, width, 1);
      if (usesPointCoordinateGather)
        Muls(red[704].ReinterpretCast<int32_t>(), gather.ReinterpretCast<int32_t>(),
             int32_t(12), 64);
      if (rows >= 8)
        Muls(red[1024].ReinterpretCast<int32_t>(), gather.ReinterpretCast<int32_t>(),
             int32_t(8), 64);
      // Preserve both consumers of the lane sequence before replacing it.
      PipeBarrier<PIPE_V>();
      Muls(gather.ReinterpretCast<int32_t>(), gather.ReinterpretCast<int32_t>(),
           int32_t(24), width);
      PipeBarrier<PIPE_V>();
    } else {
      SeedOffsetTable(gather, width, 24);
      if (usesPointCoordinateGather)
        SeedOffsetTable(red[704].ReinterpretCast<uint32_t>(), 64, 12);
      if (rows >= 8)
        SeedOffsetTable(red[1024].ReinterpretCast<uint32_t>(), 64, 8);
      if (rows >= 8 && width > 64) {
        uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2;
        SeedOffsetTable(red[1088].ReinterpretCast<uint32_t>(), 64, chunks * spacing * 4);
      }
      Fence<HardEvent::S_V>();
      ExpandDeviceOffsetSeeds(gather, width, 24);
      if (usesPointCoordinateGather)
        ExpandDeviceOffsetSeeds(red[704].ReinterpretCast<uint32_t>(), 64, 12);
      if (rows >= 8)
        ExpandDeviceOffsetSeeds(red[1024].ReinterpretCast<uint32_t>(), 64, 8);
      if (rows >= 8 && width > 64) {
        uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2;
        ExpandDeviceOffsetSeeds(red[1088].ReinterpretCast<uint32_t>(), 64, chunks * spacing * 4);
      }
    }
    uint32_t oldBatch = 0xffffffff;
    for (uint32_t i = start; i < end;) {
      uint32_t batch = i / t.pointsPerBatch;
      if (batch != oldBatch) {
        LoadSegments(batch, i == start);
        oldBatch = batch;
      }
      uint32_t live = end - i;
      if (live > rows)
        live = rows;
      uint32_t batchEnd = (batch + 1) * t.pointsPerBatch;
      if (live > batchEnd - i)
        live = batchEnd - i;
      uint32_t count = live * width;
      if (live >= 8) {
        if (!(i == start && hasStagedPointTile)) {
          Fence<HardEvent::V_MTE2>();
          PrefetchPointInputs(i, live);
          Fence<HardEvent::MTE2_V>();
        }
        if constexpr (IsSameType<T, half>::value) {
          auto h = red[768].ReinterpretCast<half>();
          Cast(red[512], h, RoundMode::CAST_NONE, live * 3);
        }
        if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
          // An aligned tile overwrites every coordinate Brcb will read.
          // Retain complete initialization for the final partial group.
          if (live % 8 != 0)
            Duplicate(red[800], 0.0f, 192);
        } else {
          Duplicate(red[800], 0.0f, 192);
        }
        for (uint32_t axis = 0; axis < 3; ++axis)
          Gather(red[800 + axis * 64], red[512], red[704].ReinterpretCast<uint32_t>(), axis * 4,
             live);
        uint32_t block = (live + 7) / 8 * 64;
        if (width < 24) {
          LocalTensor<float> target[3] = {x, y, z}, source[3] = {ax, ay, az};
          for (uint32_t axis = 0; axis < 3; ++axis) {
            Brcb(tmp, red[800 + axis * 64], uint8_t((live + 7) / 8), {1, 8});
            PipeBarrier<PIPE_V>();
            Sub(target[axis], source[axis], tmp, uint64_t(width), live,
              {1, 1, 0, uint8_t(width / 8), 0, 1});
            PipeBarrier<PIPE_V>();
          }
        } else {
          for (uint32_t axis = 0; axis < 3; ++axis)
            Brcb(tmp[axis * block], red[800 + axis * 64], uint8_t((live + 7) / 8), {1, 8});
          for (uint32_t c = 0; c < width; c += 64) {
            uint32_t mask = width < 64 ? width : 64;
            Sub(x[c], ax[c], tmp, uint64_t(mask), live, {1, 1, 0, uint8_t(width / 8), 0, 1});
            Sub(y[c], ay[c], tmp[block], uint64_t(mask), live,
              {1, 1, 0, uint8_t(width / 8), 0, 1});
            Sub(z[c], az[c], tmp[block * 2], uint64_t(mask), live,
              {1, 1, 0, uint8_t(width / 8), 0, 1});
          }
        }
      } else {
        for (uint32_t r = 0; r < live; ++r) {
          float px = float(points.GetValue((i + r) * 3)),
             py = float(points.GetValue((i + r) * 3 + 1)),
             pz = float(points.GetValue((i + r) * 3 + 2));
          Adds(x[r * width], ax, -px, width);
          Adds(y[r * width], ay, -py, width);
          Adds(z[r * width], az, -pz, width);
        }
      }

      Product(dot, x, dx, live);
      Product(tmp, y, dy, live);
      Add(dot, dot, tmp, count);
      Product(tmp, z, dz, live);
      Add(dot, dot, tmp, count);
      Product(dot, dot, len, live);
      Maxs(dot, dot, -1.0f, count);
      Mins(dot, dot, 0.0f, count);
      Product(tmp, dot, dx, live);
      Sub(x, x, tmp, count);
      Product(tmp, dot, dy, live);
      Sub(y, y, tmp, count);
      Product(tmp, dot, dz, live);
      Sub(z, z, tmp, count);
      Mul(dist, x, x, count);
      MulAddDst(dist, y, y, count);
      MulAddDst(dist, z, z, count);
      if (width <= 64) {
        WholeReduceMin(red, dist, int32_t(t.segmentCount), live, 1, 1, width / 8);
      } else {
        uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2, pitch = chunks * spacing;
        auto partial = red[128];
        WholeReduceMin(partial, dist, 64, live * chunks, spacing / 2, 1, 8);
        uint32_t tail = t.segmentCount % 64;
        if (tail) {
          WholeReduceMin(partial[(t.segmentCount / 64) * spacing],
                 dist[(t.segmentCount / 64) * 64], int32_t(tail), live, pitch / 2, 1,
                 width / 8);
        }
        uint64_t mask[2] = {0, 0};
        for (uint32_t c = 0; c < (t.segmentCount + 63) / 64; ++c)
          mask[0] |= uint64_t(1) << (c * spacing);
        WholeReduceMin(red, partial, mask, live, 1, 1, pitch / 8);
      }
      uint32_t offset = i - outStart;
      if (rows >= 8 && (offset & 7) == 0) {
        Gather(outLocal[offset], red, red[1024].ReinterpretCast<uint32_t>(), 0, live);
        auto dst = idxLocal[offset];
        Gather(dst.ReinterpretCast<float>(), red, red[1024].ReinterpretCast<uint32_t>(), 4, live);
        if (width > 64) {
          uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2;
          auto addr = red[1216].ReinterpretCast<int32_t>();
          Muls(addr, dst, int32_t(4), live);
          Add(addr, addr, red[1088].ReinterpretCast<int32_t>(), live);
          Adds(addr, addr, int32_t(128 * 4 + 4), live);
          Gather(red[1152], red, addr.ReinterpretCast<uint32_t>(), 0, live);
          ShiftRight(dst, dst, int32_t(spacing == 4 ? 2 : 1), live);
          ShiftLeft(dst, dst, int32_t(6), live);
          Add(dst, dst, red[1152].ReinterpretCast<int32_t>(), live);
        }
      } else {
        Fence<HardEvent::V_S>();
        auto ir = red.template ReinterpretCast<int32_t>();
        for (uint32_t r = 0; r < live; ++r) {
          int32_t arg = ir.GetValue(r * 2 + 1);
          if (width > 64) {
            uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2,
                pitch = chunks * spacing;
            arg = (arg / spacing) * 64 + ir.GetValue(128 + r * pitch + arg + 1);
          }
          Store(i + r, red.GetValue(r * 2), arg);
        }
        Fence<HardEvent::S_V>();
      }
      i += live;
    }
  }

 public:
  __aicore__ inline void Init(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx,
                const TreeLineMinDistanceTilingData &tiling) {
    t = tiling;
    points.SetGlobalBuffer((__gm__ T *)p);
    segments.SetGlobalBuffer((__gm__ T *)s);
    output.SetGlobalBuffer((__gm__ T *)o);
    indicesOut.SetGlobalBuffer((__gm__ int32_t *)idx);
  }
  __aicore__ inline void Process() {
    uint32_t total = t.batchCount * t.pointsPerBatch, start = total * GetBlockIdx() / t.workerCount,
        end = total * (GetBlockIdx() + 1) / t.workerCount;
    if constexpr (ALGORITHM == 3) {
      PointVector(start, end);
      return;
    }
    uint32_t pitch = (end - start + 15) / 16 * 16;
    outStart = start;
    outLocal = LocalTensor<float>(TPosition::VECCALC, 0, pitch);
    idxLocal = LocalTensor<int32_t>(TPosition::VECCALC, pitch * 4, pitch);
    bufferBase = pitch * 8;
    if constexpr (ALGORITHM == 1)
      Scalar(start, end);
    else
      Vector(start, end);
    Flush(end - start);
  }
};
// An axis-parallel finite segment projects by clamping that coordinate.
// This also covers a point segment, without a divide or a rounded reconstructed endpoint.
__aicore__ inline float AxisIntervalResidual(float point, float begin, float end) {
  float low = begin < end ? begin : end, high = begin > end ? begin : end;
  return point < low ? point - low : (point > high ? point - high : 0.0f);
}
__aicore__ inline float PointSegmentSquaredDistance(float px, float py, float pz, float a, float b, float c,
                          float endX, float endY, float endZ) {
  float u = endX - a, v = endY - b, w = endZ - c;
  float ex = px - a, ey = py - b, ez = pz - c;
  if (v == 0.0f && w == 0.0f) {
    ex = AxisIntervalResidual(px, a, endX);
  } else if (u == 0.0f && w == 0.0f) {
    ey = AxisIntervalResidual(py, b, endY);
  } else if (u == 0.0f && v == 0.0f) {
    ez = AxisIntervalResidual(pz, c, endZ);
  } else {
    float squaredLength = (u * u + v * v) + w * w;
    float quotient = 0.0f;
    if (squaredLength > 0.0f)
      quotient = ((ex * u + ey * v) + ez * w) / squaredLength;
    quotient = quotient < 0.0f ? 0.0f : (quotient > 1.0f ? 1.0f : quotient);
    ex = px - (a + quotient * u);
    ey = py - (b + quotient * v);
    ez = pz - (c + quotient * w);
  }
  return (ex * ex + ey * ey) + ez * ez;
}
// Scalar workloads consume a small tiling descriptor. Read its current
// fields directly, leaving UB free for the input DMA; this changes metadata
// transport only and never moves input computation outside the kernel.
__aicore__ inline TreeLineMinDistanceTilingData LoadScalarTilingMetadata(GM_ADDR tilingAddress) {
  GlobalTensor<uint32_t> metadataWords;
  metadataWords.SetGlobalBuffer((__gm__ uint32_t *)tilingAddress);
  TreeLineMinDistanceTilingData metadata;
  metadata.batchCount = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, batchCount) / sizeof(uint32_t));
  metadata.pointsPerBatch = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, pointsPerBatch) / sizeof(uint32_t));
  metadata.segmentCount = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, segmentCount) / sizeof(uint32_t));
  metadata.workerCount = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, workerCount) / sizeof(uint32_t));
  metadata.algorithm = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, algorithm) / sizeof(uint32_t));
  metadata.workersPerBatch = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, workersPerBatch) / sizeof(uint32_t));
  metadata.unifiedBufferBytes = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, unifiedBufferBytes) / sizeof(uint32_t));
  return metadata;
}
// Read the metadata-only 51-bit descriptor used exclusively by layout templates.
__aicore__ inline TreeLineMinDistanceTilingData LoadLayoutTemplateMetadata(GM_ADDR address) {
  uint64_t descriptor;
  if ((reinterpret_cast<uint64_t>(address) & 7) == 0) {
    GlobalTensor<uint64_t> words;
    words.SetGlobalBuffer((__gm__ uint64_t *)address);
    descriptor = words.GetValue(0);
  } else {
    GlobalTensor<uint32_t> words;
    words.SetGlobalBuffer((__gm__ uint32_t *)address);
    descriptor = uint64_t(words.GetValue(0)) | (uint64_t(words.GetValue(1)) << 32);
  }
  TreeLineMinDistanceTilingData metadata;
  metadata.batchCount = uint32_t(descriptor & 7) + 1;
  metadata.pointsPerBatch = uint32_t((descriptor >> kLayoutDescriptorPointShiftBits) & 16383) + 1;
  metadata.segmentCount = uint32_t((descriptor >> kLayoutDescriptorSegmentShiftBits) & 1023) + 1;
  metadata.algorithm = 2;
  metadata.workerCount = uint32_t((descriptor >> kLayoutDescriptorWorkerShiftBits) & 63) + 1;
  metadata.workersPerBatch = 0; // Tiled ownership uses total points and workerCount.
  metadata.unifiedBufferBytes = uint32_t((descriptor >> kLayoutDescriptorBufferShiftBits) & 262143);
  return metadata;
}

// FP32 distance and int32 index have equal element widths. A two-block
// DMA can write both current results when the address gap fits dstStride.
// Pack in increasing GM address order; all other addresses use the original
// two requests. FP16 keeps its original unequal-width writeback.
__aicore__ inline bool TryStorePairedFloatResults(GM_ADDR distanceAddress, GM_ADDR indexAddress,
                        uint32_t outputUbByteOffset,
                        float squaredDistance, int32_t segmentIndex) {
  uint64_t distanceByteAddress = reinterpret_cast<uint64_t>(distanceAddress);
  uint64_t indexByteAddress = reinterpret_cast<uint64_t>(indexAddress);
  bool distanceFirst = distanceByteAddress < indexByteAddress;
  uint64_t gapBytes = distanceFirst ? indexByteAddress - distanceByteAddress
                  : distanceByteAddress - indexByteAddress;
  if (gapBytes < sizeof(uint32_t) || gapBytes - sizeof(uint32_t) > uint64_t(0xffffffffU))
    return false;
  float distanceValue = float(sqrt(squaredDistance));
  uint32_t distanceBits = __builtin_bit_cast(uint32_t, distanceValue);
  uint32_t indexBits = __builtin_bit_cast(uint32_t, segmentIndex);
  LocalTensor<uint32_t> packedResults(TPosition::VECCALC, outputUbByteOffset, 16);
  packedResults.SetValue(0, distanceFirst ? distanceBits : indexBits);
  packedResults.SetValue(8, distanceFirst ? indexBits : distanceBits);
  GlobalTensor<uint32_t> resultAddress;
  resultAddress.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(
    distanceFirst ? distanceAddress : indexAddress));
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(resultAddress, packedResults,
    {2, sizeof(uint32_t), 0, uint32_t(gapBytes - sizeof(uint32_t)), 0});
  Fence<HardEvent::MTE3_S>();
  return true;
}

// Complete current small M1 batch; no state survives this invocation.
// Use the same direct one-segment projection as FixedPoint<T,1>. Segment
// direction/length are shared across points within this one invocation only.
template <typename T, uint32_t PointCount>
__aicore__ inline void CoalescedSingleSegment(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR indices) {
  static_assert(PointCount >= 2 && PointCount <= kCoalescedSingleSegmentMaximumPoints);
  if (GetBlockIdx() != 0)
    return;
  constexpr uint32_t pointBufferBytes = (PointCount * 3 * sizeof(T) + 31) / 32 * 32;
  constexpr uint32_t segmentBufferBytes = 32;
  constexpr uint32_t outputOffsetBytes = pointBufferBytes + segmentBufferBytes;
  GlobalTensor<T> points, segments, distances;
  GlobalTensor<int32_t> nearest;
  points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p));
  segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(s));
  distances.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(o));
  nearest.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
  LocalTensor<T> pointValues(TPosition::VECCALC, 0, pointBufferBytes / sizeof(T));
  LocalTensor<T> segmentValues(TPosition::VECCALC, pointBufferBytes, segmentBufferBytes / sizeof(T));
  LocalTensor<T> output(TPosition::VECCALC, outputOffsetBytes, 32 / sizeof(T));
  LocalTensor<int32_t> indexValues(TPosition::VECCALC, outputOffsetBytes + 32, 8);
  DataCopyPad(pointValues, points, {1, PointCount * 3 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  DataCopyPad(segmentValues, segments, {1, 6 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  Fence<HardEvent::MTE2_S>();
  float a = float(segmentValues.GetValue(0));
  float b = float(segmentValues.GetValue(1));
  float c = float(segmentValues.GetValue(2));
  float u = float(segmentValues.GetValue(3)) - a;
  float v = float(segmentValues.GetValue(4)) - b;
  float w = float(segmentValues.GetValue(5)) - c;
  float squaredLength = (u * u + v * v) + w * w;
#pragma unroll
  for (uint32_t point = 0; point < PointCount; ++point) {
    float px = float(pointValues.GetValue(point * 3));
    float py = float(pointValues.GetValue(point * 3 + 1));
    float pz = float(pointValues.GetValue(point * 3 + 2));
    float projection = 0.0f;
    if (squaredLength > 0.0f)
      projection = (((px - a) * u + (py - b) * v) + (pz - c) * w) / squaredLength;
    projection = projection < 0.0f ? 0.0f : (projection > 1.0f ? 1.0f : projection);
    float ex = px - (a + projection * u);
    float ey = py - (b + projection * v);
    float ez = pz - (c + projection * w);
    float squaredDistance = (ex * ex + ey * ey) + ez * ez;
    output.SetValue(point, T(sqrt(squaredDistance)));
    // Exactly one valid candidate exists, so its index is always zero.
    indexValues.SetValue(point, int32_t(0));
  }
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(distances, output, {1, PointCount * sizeof(T), 0, 0, 0});
  DataCopyPad(nearest, indexValues, {1, PointCount * sizeof(int32_t), 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}

// Each productive block owns one point; the launch preserves the minimum eight cores.
// All coordinates are loaded and every one of the M segments is evaluated.
template <typename T, uint32_t PointsPerBatch, uint32_t SegmentCount, bool SingleBatch = false>
__aicore__ inline void StaticScalarDomain(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR indices,
                     GM_ADDR tiling) {
  // Eight launched cores preserve the existing minimum group policy. Small
  // point rows may need padding; one current metadata word bounds ownership.
  // N>=8 never pads because every selected worker owns exactly one point.
  if constexpr (SingleBatch && PointsPerBatch != 0) {
    if (GetBlockIdx() >= PointsPerBatch)
      return;
  } else if constexpr (!SingleBatch && PointsPerBatch < 8) {
    GlobalTensor<uint32_t> schedulingWords;
    schedulingWords.SetGlobalBuffer((__gm__ uint32_t *)tiling);
    uint32_t workerCount = schedulingWords.GetValue(
      offsetof(TreeLineMinDistanceTilingData, workerCount) / sizeof(uint32_t));
    if (GetBlockIdx() >= workerCount)
      return;
  }
  static_assert(SegmentCount >= 1 && SegmentCount <= 16);
  static_assert((PointsPerBatch >= 1 && PointsPerBatch <= 16) ||
                (SingleBatch && PointsPerBatch == 0 && SegmentCount == 2));
  static_assert(PointsPerBatch * SegmentCount <= kMaximumScalarPairCount);
  uint32_t pointIndex = GetBlockIdx(), batchIndex = 0;
  if constexpr (!SingleBatch)
    batchIndex = pointIndex / PointsPerBatch;
  GlobalTensor<T> points, segments, distances;
  GlobalTensor<int32_t> nearest;
  points.SetGlobalBuffer((__gm__ T *)p);
  segments.SetGlobalBuffer((__gm__ T *)s);
  distances.SetGlobalBuffer((__gm__ T *)o);
  nearest.SetGlobalBuffer((__gm__ int32_t *)indices);
  constexpr uint32_t segmentBytes = (SegmentCount * 6 * sizeof(T) + 31) / 32 * 32;
  constexpr uint32_t outputOffsetBytes = 32 + segmentBytes;
  LocalTensor<T> pointValues(TPosition::VECCALC, 0, 32 / sizeof(T));
  LocalTensor<T> segmentValues(TPosition::VECCALC, 32, segmentBytes / sizeof(T));
  DataCopyPad(pointValues, points[pointIndex * 3], {1, 3 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  DataCopyPad(segmentValues, segments[batchIndex * SegmentCount * 6],
        {1, SegmentCount * 6 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  Fence<HardEvent::MTE2_S>();
  float px = float(pointValues.GetValue(0)), py = float(pointValues.GetValue(1)),
     pz = float(pointValues.GetValue(2));
  float best = 3.402823466e38f;
  int32_t arg = 0;
#pragma unroll
  for (uint32_t segment = 0; segment < SegmentCount; ++segment) {
    uint32_t coordinate = segment * 6;
    float value = PointSegmentSquaredDistance(px, py, pz,
      float(segmentValues.GetValue(coordinate)), float(segmentValues.GetValue(coordinate + 1)),
      float(segmentValues.GetValue(coordinate + 2)), float(segmentValues.GetValue(coordinate + 3)),
      float(segmentValues.GetValue(coordinate + 4)), float(segmentValues.GetValue(coordinate + 5)));
    if (segment == 0 || value < best) {
      best = value;
      arg = int32_t(segment);
    }
  }
  if constexpr (IsSameType<T, float>::value) {
    if (TryStorePairedFloatResults(o + pointIndex * sizeof(float),
                   indices + pointIndex * sizeof(int32_t),
                   outputOffsetBytes, best, arg))
      return;
  }
  LocalTensor<T> output(TPosition::VECCALC, outputOffsetBytes, 32 / sizeof(T));
  LocalTensor<int32_t> indexValues(TPosition::VECCALC, outputOffsetBytes + 32, 8);
  output.SetValue(0, T(sqrt(best)));
  indexValues.SetValue(0, arg);
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(distances[pointIndex], output, {1, sizeof(T), 0, 0, 0});
  DataCopyPad(nearest[pointIndex], indexValues, {1, 4, 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}

template <typename T, uint32_t Algorithm>
__aicore__ inline void ExecuteNewLayout(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR indices,
                   GM_ADDR workspace, GM_ADDR tiling) {
  if constexpr (Algorithm >= kCoalescedSingleSegmentAlgorithmBase &&
         Algorithm < kCoalescedSingleSegmentAlgorithmBase + kCoalescedSingleSegmentMaximumPoints - 1) {
    CoalescedSingleSegment<T, Algorithm - kCoalescedSingleSegmentAlgorithmBase + 2>(p, s, o, indices);
  } else if constexpr (Algorithm >= kDenseScratchAlgorithmBase &&
         Algorithm < kDenseScratchAlgorithmBase + kDenseScratchPitchCount) {
    auto metadata = LoadLayoutTemplateMetadata(tiling);
    if (GetBlockIdx() >= metadata.workerCount)
      return;
    constexpr uint32_t pitchElements = 256 + (Algorithm - kDenseScratchAlgorithmBase) * 64;
    DenseScratchTiledTreeLine<T, pitchElements> op;
    op.Run(p, s, o, indices, metadata);
  } else if constexpr (Algorithm >= kWideOutputAlgorithmBase &&
         Algorithm < kWideOutputAlgorithmBase + kWideOutputWidthCount) {
    // This route has exactly one logical output. Read only its current
    // segment count; the appended tags keep the original full descriptor.
    if (GetBlockIdx() != 0)
      return;
    GlobalTensor<uint32_t> metadataWords;
    metadataWords.SetGlobalBuffer((__gm__ uint32_t *)tiling);
    uint32_t segmentCount = metadataWords.GetValue(
      offsetof(TreeLineMinDistanceTilingData, segmentCount) / sizeof(uint32_t));
    SinglePointTreeLine<T, (16U << (Algorithm - kWideOutputAlgorithmBase))>(
      p, s, o, indices, segmentCount);
    PipeBarrier<PIPE_ALL>();
  } else if constexpr (Algorithm >= kOptimizedGenericAlgorithmBase && Algorithm < 512) {
    constexpr uint32_t originalAlgorithm = Algorithm - kOptimizedGenericAlgorithmBase;
    auto metadata = LoadLayoutTemplateMetadata(tiling);
    if (GetBlockIdx() >= metadata.workerCount)
      return;
    metadata.algorithm = originalAlgorithm;
    KernelTreeLineMinDistance<T, originalAlgorithm> op;
    op.Init(p, s, o, indices, metadata);
    op.Process();
    PipeBarrier<PIPE_ALL>();
  } else if constexpr (Algorithm >= 768) {
    StaticScalarDomain<T, Algorithm - 768 + 2, 1, true>(p, s, o, indices, tiling);
  } else if constexpr (Algorithm == 608) {
    StaticScalarDomain<T, 0, 2, true>(p, s, o, indices, tiling);
  } else if constexpr (Algorithm >= 512) {
    constexpr uint32_t shape = Algorithm - 512;
    StaticScalarDomain<T, (shape >> 4) + 2, (shape & 15) + 2, true>(p, s, o, indices, tiling);
  } else if constexpr (Algorithm >= kStaticSingleSegmentAlgorithmBase) {
    constexpr uint32_t pointsPerBatch = Algorithm - kStaticSingleSegmentAlgorithmBase + 1;
    StaticScalarDomain<T, pointsPerBatch, 1>(p, s, o, indices, tiling);
  } else if constexpr (Algorithm >= kStaticScalarAlgorithmBase) {
    constexpr uint32_t shape = Algorithm - kStaticScalarAlgorithmBase;
    constexpr uint32_t pointsPerBatch = (shape >> kStaticScalarPointShift) + 1;
    constexpr uint32_t segmentCount = (shape & kStaticScalarSegmentMask) + 2;
    StaticScalarDomain<T, pointsPerBatch, segmentCount>(p, s, o, indices, tiling);
  } else {
    auto metadata = LoadLayoutTemplateMetadata(tiling);
    if (GetBlockIdx() >= metadata.workerCount)
      return;
    if constexpr (Algorithm == 57) {
      metadata.segmentCount = 1;
      metadata.algorithm = 3;
      KernelTreeLineMinDistance<T, 3> op;
      op.Init(p, s, o, indices, metadata);
      op.Process();
      PipeBarrier<PIPE_ALL>();
    } else if constexpr (Algorithm <= 51) {
      constexpr uint32_t segmentPitchElements = (Algorithm - 35) * 64;
      TiledTreeLine<T, segmentPitchElements> op;
      op.Run(p, s, o, indices, metadata);
    } else {
      constexpr uint32_t segmentLanes = Algorithm == 55 ? 2 :
        (Algorithm == 56 ? 4 : (Algorithm == 54 ? 8 : (Algorithm == 52 ? 16 : 32)));
      metadata.algorithm = Algorithm - 20;
      PackedRowsTreeLine<T, segmentLanes, true> op;
      op.Run(p, s, o, indices, metadata);
    }
  }
}

}
template <typename DT_TREE_POINTS, uint32_t ALGORITHM>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments,
  GM_ADDR min_distance, GM_ADDR nearest_segment_idx, GM_ADDR workspace, GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
  if constexpr (ALGORITHM <= 36) {

  if constexpr (ALGORITHM >= 5 && ALGORITHM <= 12) {
    FixedPoint<DT_TREE_POINTS, ALGORITHM - 4>(tree_points, line_segments, min_distance,
                         nearest_segment_idx);
    return;
  } else {
    GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, t, tiling);
    if (GetBlockIdx() >= t.workerCount)
      return;
    if constexpr (ALGORITHM == 1) {
      if (t.segmentCount == 1 && t.batchCount * t.pointsPerBatch >= 8)
        CachedSingleSegmentTiny<DT_TREE_POINTS>(tree_points, line_segments, min_distance,
                            nearest_segment_idx, t);
      else
        ScalarDmaTiny<DT_TREE_POINTS>(tree_points, line_segments, min_distance, nearest_segment_idx,
                       t);
      PipeBarrier<PIPE_ALL>();
      return;
    } else if constexpr (ALGORITHM == 4) {
      SpatialTreeLine<DT_TREE_POINTS> op;
      op.Run(tree_points, line_segments, min_distance, nearest_segment_idx,
         GetUserWorkspace(workspace), t);
      return;
    } else {
      KernelTreeLineMinDistance<DT_TREE_POINTS, ALGORITHM> op;
      op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, t);
      op.Process();
      PipeBarrier<PIPE_ALL>();
    }
  }

  } else {
    OptimizedLayout::ExecuteNewLayout<DT_TREE_POINTS, ALGORITHM>(tree_points, line_segments,
      min_distance, nearest_segment_idx, workspace, tiling);
  }
}
