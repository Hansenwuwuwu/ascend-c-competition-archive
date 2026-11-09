#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tree_line_min_distance_tiling.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  const auto *pointDescriptor = context->GetInputDesc(0);
  const auto *segmentDescriptor = context->GetInputDesc(1);
  const auto *pointShape = context->GetInputShape(0);
  const auto *segmentShape = context->GetInputShape(1);
  if (!pointDescriptor || !segmentDescriptor || !pointShape || !segmentShape)
    return ge::GRAPH_FAILED;
  const auto &pointStorageShape = pointShape->GetStorageShape();
  const auto &segmentStorageShape = segmentShape->GetStorageShape();
  if (pointStorageShape.GetDimNum() != 3 || segmentStorageShape.GetDimNum() != 4 ||
    pointStorageShape.GetDim(2) != 3 || segmentStorageShape.GetDim(2) != 2 ||
    segmentStorageShape.GetDim(3) != 3 ||
    pointStorageShape.GetDim(0) != segmentStorageShape.GetDim(0) || pointStorageShape.GetDim(0) < 1 ||
    pointStorageShape.GetDim(0) > 8 || pointStorageShape.GetDim(1) < 1 ||
    pointStorageShape.GetDim(1) > 16384 || segmentStorageShape.GetDim(1) < 1 ||
    segmentStorageShape.GetDim(1) > 1024 ||
    pointDescriptor->GetDataType() != segmentDescriptor->GetDataType() ||
    (pointDescriptor->GetDataType() != ge::DT_FLOAT &&
    pointDescriptor->GetDataType() != ge::DT_FLOAT16))
    return ge::GRAPH_FAILED;
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t DT_TREE_POINTS = uint32_t(pointDescriptor->GetDataType());
  auto *tilingData = context->GetTilingData<TreeLineMinDistanceTilingData>();
  tilingData->batchCount = pointStorageShape.GetDim(0);
  tilingData->pointsPerBatch = pointStorageShape.GetDim(1);
  tilingData->segmentCount = segmentStorageShape.GetDim(1);
  uint32_t totalPointCount = tilingData->batchCount * tilingData->pointsPerBatch,
      vectorCoreCount = platform.GetCoreNumAiv();
  if (!vectorCoreCount)
    return ge::GRAPH_FAILED;
  // Private layout descriptors encode at most64workers. Bound the launch
  // budget here so superseded unpacked vector entries are never selected.
  if (vectorCoreCount > 64)
    vectorCoreCount = 64;
  tilingData->workerCount = totalPointCount < vectorCoreCount ? totalPointCount : vectorCoreCount;
  tilingData->algorithm = totalPointCount * tilingData->segmentCount <= kMaximumScalarPairCount;
  if (totalPointCount == 1 && tilingData->segmentCount <= 8)
    tilingData->algorithm = 4 + tilingData->segmentCount;
  if (tilingData->segmentCount <= 8 && totalPointCount >= 64)
    tilingData->algorithm = 3;
  if (tilingData->algorithm == 0 && tilingData->segmentCount <= 64) {
    uint32_t requiredWorkers = (totalPointCount + 31) / 32;
    if (requiredWorkers < 8)
      requiredWorkers = 8;
    tilingData->workerCount = requiredWorkers < vectorCoreCount ? requiredWorkers : vectorCoreCount;
    if (tilingData->workerCount > totalPointCount)
      tilingData->workerCount = totalPointCount;
  }
  if (tilingData->algorithm == 3) {
    uint32_t requiredWorkers = (totalPointCount * tilingData->segmentCount + 1023) / 1024;
    tilingData->workerCount = requiredWorkers < vectorCoreCount ? requiredWorkers : vectorCoreCount;
  }
  if (tilingData->algorithm == 3 && tilingData->segmentCount == 1) {
    uint32_t requiredWorkers = (totalPointCount + 1023) / 1024;
    // One worker per batch avoids serial loads of unrelated short tables.
    // Use one launch group, with batch-aligned ownership and padding cores.
    if (requiredWorkers < tilingData->batchCount)
      requiredWorkers = tilingData->batchCount * (8 / tilingData->batchCount);
    tilingData->workerCount = requiredWorkers < vectorCoreCount ? requiredWorkers : vectorCoreCount;
  }
  uint64_t unifiedBufferBytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, unifiedBufferBytes);
  // Reserve an aligned representable UB budget rather than falling back
  // to duplicate unpacked kernels on a larger-UB target.
  constexpr uint64_t maximumDescriptorUbBytes = (1ULL << kLayoutDescriptorBufferWidthBits) - 32;
  if (unifiedBufferBytes > maximumDescriptorUbBytes)
    unifiedBufferBytes = maximumDescriptorUbBytes;
  tilingData->unifiedBufferBytes = uint32_t(unifiedBufferBytes);
  tilingData->workersPerBatch = vectorCoreCount / tilingData->batchCount;
  // Spatial ordering amortizes its per-group bounds once each core has enough points.
  if ((tilingData->pointsPerBatch / tilingData->workersPerBatch >= 1024 &&
    tilingData->segmentCount >= 512) ||
    (tilingData->pointsPerBatch / tilingData->workersPerBatch >= 512 &&
    tilingData->segmentCount >= kSatMinimumSegmentCount)) {
    tilingData->algorithm = 4;
    tilingData->workerCount = tilingData->batchCount * tilingData->workersPerBatch;
  }
  if (tilingData->algorithm == 4) {
    uint32_t pointsPerWorker = (tilingData->pointsPerBatch + tilingData->workersPerBatch - 1) /
                 tilingData->workersPerBatch,
        sortCapacity = 32, segmentCapacity = 32;
    while (sortCapacity < pointsPerWorker)
      sortCapacity *= 2;
    while (segmentCapacity < tilingData->segmentCount)
      segmentCapacity *= 2;
    uint64_t sortPrefixElements = 2 * ((pointsPerWorker + 15) / 16 * 16) + sortCapacity;
    if ((sortPrefixElements + 6 * sortCapacity + 3 * pointsPerWorker) * 4 + 8192 > unifiedBufferBytes ||
      (64 + 22 * segmentCapacity + 64 + 4 * 4160 + 2688) * 4 + 8192 > unifiedBufferBytes) {
      tilingData->algorithm = 0;
      tilingData->workerCount = totalPointCount < vectorCoreCount ? totalPointCount : vectorCoreCount;
    }
  }
  // Compare arithmetic padding costs before paying per-chunk setup for a short point slice.
  uint32_t denseSegmentLanes = 128;
  while (denseSegmentLanes < tilingData->segmentCount)
    denseSegmentLanes *= 2;
  uint32_t paddedSegmentCount = (tilingData->segmentCount + 63) / 64 * 64;
  bool hasCompactSegmentPadding = uint64_t(paddedSegmentCount) * 4 <= uint64_t(denseSegmentLanes) * 3;
  uint32_t batchAlignedWorkerCount = tilingData->batchCount * (vectorCoreCount / tilingData->batchCount);
  uint32_t tiledWorkerCount =
    totalPointCount / batchAlignedWorkerCount < 128 ? batchAlignedWorkerCount : tilingData->workerCount;
  uint32_t pointsPerWorker = totalPointCount / tiledWorkerCount;
  if (tilingData->algorithm == 0 && tilingData->segmentCount > 64 &&
    (pointsPerWorker >= 64 || (pointsPerWorker >= 8 && hasCompactSegmentPadding)) &&
    (8ULL * paddedSegmentCount + 37888) * 4 + 8192 <= unifiedBufferBytes) {
    tilingData->algorithm = 2;
    tilingData->workerCount = tiledWorkerCount;
  }
  if (tilingData->algorithm == 2) {
    // Two launch groups trade a bounded extra point row span against
    // dispatch/table traffic. Batch-aligned slices remain within8..32rows,
    // retaining the original residual arithmetic and complete tail handling.
    constexpr uint32_t tiledWorkerBudget = 16;
    constexpr uint32_t minimumPointRows = 8, maximumPointRows = 32;
    uint32_t availableWorkers = vectorCoreCount < tiledWorkerBudget ? vectorCoreCount : tiledWorkerBudget;
    uint32_t alignedWorkers = availableWorkers / tilingData->batchCount * tilingData->batchCount;
    if (alignedWorkers > 0 && alignedWorkers < tilingData->workerCount &&
        totalPointCount >= alignedWorkers * minimumPointRows &&
        totalPointCount <= alignedWorkers * maximumPointRows)
      tilingData->workerCount = alignedWorkers;
  }
  if (tilingData->algorithm == 0 && tilingData->segmentCount > 64) {
    uint32_t laneCount = 128;
    tilingData->algorithm = 13;
    while (laneCount < tilingData->segmentCount) {
      laneCount *= 2;
      ++tilingData->algorithm;
    }
  }
  if (totalPointCount == 1 && tilingData->segmentCount > 8) {
    uint32_t laneCount = 16;
    tilingData->algorithm = 17;
    while (laneCount < tilingData->segmentCount) {
      laneCount *= 2;
      ++tilingData->algorithm;
    }
  }
  if (tilingData->algorithm == 0 && tilingData->segmentCount <= 64)
    tilingData->algorithm = 23 + (tilingData->segmentCount + 7) / 8;
  // At least two point rows amortize packed setup for short per-worker slices. Keep
  // ownership batch-aligned so short tiles never reload an unrelated table.
  if (tilingData->segmentCount >= 9 && tilingData->segmentCount <= 32) {
    uint32_t packedWorkersPerBatch = tilingData->workerCount / tilingData->batchCount;
    uint32_t packedWorkerCount = tilingData->batchCount * packedWorkersPerBatch;
    uint32_t packedSegmentLanes = tilingData->segmentCount <= 16 ? 16 : 32;
    uint32_t packedPointCount = packedSegmentLanes == 16 ? 240 : 216;
    uint64_t packedBufferBytes =
      (uint64_t(packedPointCount) * (6 * packedSegmentLanes + 16) + 1920) * 4 + 8192;
    if (packedWorkersPerBatch > 0 && tilingData->pointsPerBatch / packedWorkersPerBatch >= 2 &&
      packedBufferBytes <= unifiedBufferBytes) {
      tilingData->algorithm = packedSegmentLanes == 16 ? 32 : 33;
      tilingData->workerCount = packedWorkerCount;
    }
  }
  // One launch group limits concurrent requests to a segment table shorter than 512 bytes.
  constexpr uint32_t packedSegmentLanes = 8, packedPointRows = 240, packedWorkerBudget = 8;
  // Six pair planes, thirteen state elements per point, fixed coefficient/reduction space,
  // and the same 8 KiB API scratch reservation used by the other UB paths.
  constexpr uint64_t packedUnifiedBufferBytes =
    (uint64_t(packedPointRows) * (6 * packedSegmentLanes + 16) + 1920) * sizeof(float) + 8192;
  uint32_t packedWorkers = (vectorCoreCount < packedWorkerBudget ? vectorCoreCount : packedWorkerBudget);
  packedWorkers = packedWorkers / tilingData->batchCount * tilingData->batchCount;
  // Eight 8-lane query rows fill a 64-lane FP32 vector. For shorter
  // inputs, avoid assigning multiple mostly empty vectors to each batch.
  constexpr uint32_t packedPointsPerVector = 64 / packedSegmentLanes;
  uint32_t densityWorkersPerBatch =
    (tilingData->pointsPerBatch + packedPointsPerVector - 1) / packedPointsPerVector;
  uint32_t densityWorkerCount = densityWorkersPerBatch * tilingData->batchCount;
  if (densityWorkerCount < packedWorkers)
    packedWorkers = densityWorkerCount;
  // Two or more points per worker amortize packed-row setup and avoid the
  // short vector route's per-point scalar reduction handoffs. Keep the
  // bounded scalar route for its smaller total arithmetic workloads.
  if (packedWorkers != 0 && packedUnifiedBufferBytes <= unifiedBufferBytes &&
    tilingData->algorithm != 1 && totalPointCount >= packedWorkers * 2 &&
    tilingData->segmentCount * 2 >= packedSegmentLanes &&
    tilingData->segmentCount <= packedSegmentLanes &&
    (totalPointCount + packedWorkers - 1) / packedWorkers <= packedPointRows) {
    tilingData->algorithm = 34;
    tilingData->workerCount = packedWorkers;
  }
  // Short point-vector slices can walk several batch tables serially. Eight
  // packed workers amortize those table loads; dense M=4 tiles also avoid half
  // of the original eight-lane arithmetic when each worker owns one point tile.
  if (tilingData->segmentCount >= 2 && tilingData->segmentCount <= 4) {
    // Keep each batch on its own workers. The physical launch remains eight
    // cores, including padding cores when B does not divide eight.
    const uint32_t packedWorkerCount = tilingData->batchCount * (8 / tilingData->batchCount);
    uint32_t packedLanes = tilingData->segmentCount == 2 ? 2 : 4;
    uint32_t packedRows = packedLanes == 2 ? 224 : 240;
    uint32_t maximumWorkerPoints = (totalPointCount + packedWorkerCount - 1) / packedWorkerCount;
    bool hasSerialBatchTables =
      tilingData->algorithm == 3 && tilingData->workerCount < tilingData->batchCount;
    bool hasDenseFourLaneTile = tilingData->segmentCount == 4 &&
                  totalPointCount >= packedWorkerCount * 64 &&
                  maximumWorkerPoints <= packedRows;
    uint64_t requiredUbBytes = (uint64_t(packedRows) * (6 * packedLanes + 16) + 1984) * 4 + 8192;
    if ((hasSerialBatchTables || hasDenseFourLaneTile) && maximumWorkerPoints <= packedRows * 2 &&
      packedWorkerCount <= vectorCoreCount && requiredUbBytes <= unifiedBufferBytes) {
      tilingData->algorithm = packedLanes == 2 ? 35 : 36;
      tilingData->workerCount = packedWorkerCount;
    }
  }
  // Cache a single short table while point-vector setup is still
  // dominant. The existing vector route remains for 64 or more points.
  constexpr uint32_t scalarWorkerBudget = 8;
  uint32_t scalarWorkers = vectorCoreCount < scalarWorkerBudget ? vectorCoreCount : scalarWorkerBudget;
  uint32_t scalarWorkersPerBatch = scalarWorkers / tilingData->batchCount;
  bool isShortScalarOrVector =
    tilingData->algorithm == 1 || (tilingData->algorithm >= 24 && tilingData->algorithm <= 31);
  if (isShortScalarOrVector && tilingData->segmentCount == 1 && totalPointCount >= 8 &&
    scalarWorkersPerBatch != 0) {
    if (scalarWorkersPerBatch > tilingData->pointsPerBatch)
      scalarWorkersPerBatch = tilingData->pointsPerBatch;
    tilingData->algorithm = 1;
    tilingData->workerCount = tilingData->batchCount * scalarWorkersPerBatch;
  }
  bool usesStaticScalarDomain = tilingData->algorithm == 1 &&
    tilingData->workerCount == totalPointCount && totalPointCount >= 2 &&
    ((tilingData->segmentCount == 1 &&
     tilingData->pointsPerBatch <= kStaticSingleSegmentPointCapacity) ||
    // Keep the proven two-candidate family. Other short tables use
    // ScalarDmaTiny, which evaluates every segment with runtime bounds.
    (tilingData->segmentCount == 2 &&
     tilingData->pointsPerBatch <= 16 &&
     uint64_t(tilingData->pointsPerBatch) * tilingData->segmentCount <= kMaximumScalarPairCount));
  uint32_t ALGORITHM = tilingData->algorithm;
  if (usesStaticScalarDomain) {
    if (tilingData->segmentCount == 1)
      ALGORITHM = kStaticSingleSegmentAlgorithmBase + tilingData->pointsPerBatch - 1;
    else
      ALGORITHM = kStaticScalarAlgorithmBase +
        ((tilingData->pointsPerBatch - 1) << kStaticScalarPointShift) + tilingData->segmentCount - 2;
  }
  // Every tiled input uses an aligned pitch in65..1024segments. The bounded
  // launch/UB budgets above make this private descriptor available for all routes.
  if (ALGORITHM == 2)
    ALGORITHM = 37 + (tilingData->segmentCount - 65) / 64;
  if (ALGORITHM >= 32 && ALGORITHM <= 36)
    ALGORITHM += 20;
  // A single-segment layout eliminates generic argmin and multi-table branches.
  if (ALGORITHM == 3 && tilingData->segmentCount == 1)
    ALGORITHM = 57;
  bool isGenericVectorLayout = (ALGORITHM >= 13 && ALGORITHM <= 16) ||
                (ALGORITHM >= 24 && ALGORITHM <= 31);
  if (isGenericVectorLayout)
    ALGORITHM += kOptimizedGenericAlgorithmBase;
  // A single batch has a fixed table origin and a compile-time point bound.
  // Exhaust the existing scalar pair budget; padding cores need no GM metadata.
  bool usesSingleBatchScalar = usesStaticScalarDomain && tilingData->batchCount == 1;
  if (usesSingleBatchScalar) {
    if (tilingData->segmentCount == 1)
      ALGORITHM = 768 + tilingData->pointsPerBatch - 2;
    else {
      // A single batch with at least eight points launches exactly N cores.
      // Its complete two-segment calculation needs no static N or padding check.
      ALGORITHM = tilingData->pointsPerBatch >= 8 ? 608 :
        512 + ((tilingData->pointsPerBatch - 2) << 4) + tilingData->segmentCount - 2;
    }
  }
  // Encode the compiler declaration index from exhaustive metadata domains.
  // The scalar prefix counts exactly min(16,32/N)-1 segment choices per N;
  // it is dispatch metadata, never a table or result derived from tensor data.
  // The existing complete single-batch M1 domain has2..8points. One
  // productive core reuses the current six segment coordinates and writes
  // complete contiguous output vectors, avoiding per-point small requests.
  bool usesCoalescedSingleSegment =
    ALGORITHM >= 768 + kCoalescedSingleSegmentMinimumPoints - 2 && ALGORITHM <= 774;
  if (usesCoalescedSingleSegment) {
    ALGORITHM = kCoalescedSingleSegmentAlgorithmBase + ALGORITHM - 768;
    tilingData->workerCount = 1;
  }
  // A complete worker span of at most64points fits each4096float
  // residual half. Dense tiles amortize the physical scratch selection;
  // runtime batch tails retain the complete sparse layout inside the kernel.
  uint32_t pointSpan = (totalPointCount + tilingData->workerCount - 1) / tilingData->workerCount;
  uint32_t minimumPointSpan = totalPointCount / tilingData->workerCount;
  bool usesDenseScratch = ALGORITHM >= 39 && ALGORITHM <= 51 && minimumPointSpan >= 32 && pointSpan <= 64;
  if (usesDenseScratch)
    ALGORITHM = kDenseScratchAlgorithmBase + ALGORITHM - 39;
  bool usesWideOutputTemplate = ALGORITHM >= 17 && ALGORITHM <= 23;
  if (usesWideOutputTemplate)
    ALGORITHM = kWideOutputAlgorithmBase + ALGORITHM - 17;
  uint64_t algorithmIndex;
  if (usesCoalescedSingleSegment) {
    algorithmIndex = kCoalescedSingleSegmentDeclarationBase + ALGORITHM - kCoalescedSingleSegmentAlgorithmBase;
  } else if (usesDenseScratch) {
    algorithmIndex = kDenseScratchDeclarationBase + ALGORITHM - kDenseScratchAlgorithmBase;
  } else if (usesWideOutputTemplate) {
    algorithmIndex = kWideOutputDeclarationBase + ALGORITHM - kWideOutputAlgorithmBase;
  } else if (ALGORITHM <= 57) {
    algorithmIndex = ALGORITHM;
  } else if (usesSingleBatchScalar) {
    if (tilingData->segmentCount == 1) {
      algorithmIndex = 149 + 56 + tilingData->pointsPerBatch - 2;
    } else {
      uint32_t scalarPrefix = 0;
      for (uint32_t pointCount = 2; pointCount < tilingData->pointsPerBatch; ++pointCount) {
        uint32_t segmentChoices = kMaximumScalarPairCount / pointCount;
        if (segmentChoices > 16)
          segmentChoices = 16;
        scalarPrefix += segmentChoices - 1;
      }
      algorithmIndex = tilingData->pointsPerBatch >= 8 ? 192 :
        149 + scalarPrefix + tilingData->segmentCount - 2;
    }
  } else if (usesStaticScalarDomain && tilingData->segmentCount >= 2) {
    uint32_t scalarPrefix = 0;
    for (uint32_t pointCount = 1; pointCount < tilingData->pointsPerBatch; ++pointCount) {
      uint32_t segmentChoices = kMaximumScalarPairCount / pointCount;
      if (segmentChoices > 16)
        segmentChoices = 16;
      scalarPrefix += segmentChoices - 1;
    }
    algorithmIndex = 58 + scalarPrefix + tilingData->segmentCount - 2;
  } else if (ALGORITHM >= kStaticSingleSegmentAlgorithmBase &&
       ALGORITHM < kStaticSingleSegmentAlgorithmBase + kStaticSingleSegmentPointCapacity) {
    algorithmIndex = 58 + 71 + ALGORITHM - kStaticSingleSegmentAlgorithmBase;
  } else if (ALGORITHM >= kOptimizedGenericAlgorithmBase + 13 &&
       ALGORITHM <= kOptimizedGenericAlgorithmBase + 16) {
    algorithmIndex = 58 + 71 + 8 + ALGORITHM - kOptimizedGenericAlgorithmBase - 13;
  } else if (ALGORITHM >= kOptimizedGenericAlgorithmBase + 24 &&
       ALGORITHM <= kOptimizedGenericAlgorithmBase + 31) {
    algorithmIndex = 58 + 71 + 8 + 4 + ALGORITHM - kOptimizedGenericAlgorithmBase - 24;
  } else {
    return ge::GRAPH_FAILED;
  }
  // DATATYPE_DECL reserves the SDK's8 low bits; the uint flag stores its list index.
  context->SetTilingKey((algorithmIndex << ASCENDC_TPL_8_BW) | uint64_t(DT_TREE_POINTS));
  uint32_t launchCoreCount = tilingData->workerCount < 8 ? 8 : tilingData->workerCount;
  if (tilingData->algorithm == 4 || tilingData->algorithm == 32 || tilingData->algorithm == 33) {
    uint32_t alignedCoreCount = (launchCoreCount + 7) / 8 * 8;
    if (alignedCoreCount <= vectorCoreCount)
      launchCoreCount = alignedCoreCount;
  }
  context->SetBlockDim(launchCoreCount);
  context->GetWorkspaceSizes(1)[0] =
    tilingData->algorithm == 4 ? platform.GetLibApiWorkSpaceSize() + uint64_t(totalPointCount) * 16 : 0;
  if (usesDenseScratch || (ALGORITHM >= 37 && ALGORITHM <= 57) || (ALGORITHM >= kOptimizedGenericAlgorithmBase && ALGORITHM < 512)) {
    // This descriptor format is private to the new layout templates. Values
    // describe shape and launch metadata only; no input coordinates are read.
    uint64_t descriptor = uint64_t(tilingData->batchCount - 1) |
      (uint64_t(tilingData->pointsPerBatch - 1) << kLayoutDescriptorPointShiftBits) |
      (uint64_t(tilingData->segmentCount - 1) << kLayoutDescriptorSegmentShiftBits) |
      (uint64_t(tilingData->workerCount - 1) << kLayoutDescriptorWorkerShiftBits) |
      (uint64_t(tilingData->unifiedBufferBytes) << kLayoutDescriptorBufferShiftBits);
    // Assign words separately so Host storage need only be 4-byte aligned.
    // All launch/workspace decisions above have consumed the unpacked fields.
    tilingData->batchCount = uint32_t(descriptor);
    tilingData->pointsPerBatch = uint32_t(descriptor >> 32);
  }
  return ge::GRAPH_SUCCESS;
}
} // namespace optiling
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
  const auto *pointStorageShape = c->GetInputShape(0);
  const auto *segmentStorageShape = c->GetInputShape(1);
  if (!pointStorageShape || !segmentStorageShape || pointStorageShape->GetDimNum() != 3 ||
    segmentStorageShape->GetDimNum() != 4 || pointStorageShape->GetDim(2) != 3 ||
    segmentStorageShape->GetDim(2) != 2 || segmentStorageShape->GetDim(3) != 3 ||
    pointStorageShape->GetDim(0) != segmentStorageShape->GetDim(0) ||
    pointStorageShape->GetDim(0) < 1 || pointStorageShape->GetDim(0) > 8 ||
    pointStorageShape->GetDim(1) < 1 || pointStorageShape->GetDim(1) > 16384 ||
    segmentStorageShape->GetDim(1) < 1 || segmentStorageShape->GetDim(1) > 1024)
    return GRAPH_FAILED;
  for (int k = 0; k < 2; ++k) {
    auto *o = c->GetOutputShape(k);
    o->SetDimNum(2);
    o->SetDim(0, pointStorageShape->GetDim(0));
    o->SetDim(1, pointStorageShape->GetDim(1));
  }
  return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
  c->SetOutputDataType(0, c->GetInputDataType(0));
  c->SetOutputDataType(1, DT_INT32);
  return GRAPH_SUCCESS;
}
} // namespace ge
namespace ops {
class TreeLineMinDistance : public OpDef {
 public:
  explicit TreeLineMinDistance(const char *name) : OpDef(name) {
    this->Input("tree_points")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("line_segments")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("min_distance")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("nearest_segment_idx")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b").AddConfig("ascend910_93");
  }
};
OP_ADD(TreeLineMinDistance);
} // namespace ops
