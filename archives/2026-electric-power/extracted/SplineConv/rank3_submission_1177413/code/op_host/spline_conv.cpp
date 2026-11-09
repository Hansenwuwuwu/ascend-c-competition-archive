#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/spline_conv_tiling.h"
#include "../op_kernel/tiling_key_spline_conv.h"
#include <algorithm>
#include <cstring>

namespace optiling {
// Eight reserved vector workers reduced dispatch variance on local 910_93;
// this is the grouping used by the platform-validated baseline. It is a
// scheduling parameter independent of graph values and complete shapes.
static constexpr uint32_t kPreferredVectorWorkerCount = 8;
// Two arrays of 64 int64 endpoints occupy the 1024-byte control region.
static constexpr uint32_t kSparseControlEdgeCapacity = 64;
static constexpr uint32_t kGenericHeadroomBytes = 16384;

// The platform schedules vector workers in groups of eight. Metadata keeps
// the real output-owner count; additional launch lanes return on device.
static uint32_t GroupedLaunchWorkerCount(uint32_t requestedWorkerCount, uint32_t availableWorkerCount) {
    const uint32_t groupedWorkerCount =
        (requestedWorkerCount + kPreferredVectorWorkerCount - 1) / kPreferredVectorWorkerCount *
        kPreferredVectorWorkerCount;
    // Preserve a correct launch on hardware without a complete final group.
    return groupedWorkerCount <= availableWorkerCount ? groupedWorkerCount : requestedWorkerCount;
}

static uint64_t AlignUbBytes(uint64_t bytes) {
    return (bytes + kUbAlignmentBytes - 1) / kUbAlignmentBytes * kUbAlignmentBytes;
}
static uint64_t Bf16StorageStartBytes(uint64_t bytes) {
    return (bytes + kBf16StorageBoundaryBytes - 1) / kBf16StorageBoundaryBytes * kBf16StorageBoundaryBytes;
}

// Mirrors RuntimeSparseGraph's allocations, including the zero source that
// remains live while a coowned interval computes its messages. All modes
// allocate the full zero capacity; graph/tile row counts do not change it.
static uint64_t RuntimeSparseFootprintBytes(uint64_t inputChannels, uint64_t outputChannels, bool bf16) {
    const uint64_t matrixElements = inputChannels * outputChannels;
    const uint64_t broadcastElements = inputChannels * (outputChannels <= 32 ? outputChannels : 8);
    uint64_t messageBytes = kSparseControlBytes + AlignUbBytes(inputChannels * 4) +
                            AlignUbBytes(4 * matrixElements * 4) + kResultPhasePaddingBytes +
                            AlignUbBytes(BilinearProductElementCount(inputChannels, outputChannels) * 4) + kOperandPhasePaddingBytes +
                            AlignUbBytes(broadcastElements * 4) + AlignUbBytes(outputChannels * 4) +
                            AlignUbBytes(outputChannels * 4); // Per-message support sum, separate from neighbor sum.
    if (bf16) {
        messageBytes = Bf16StorageStartBytes(messageBytes);
        messageBytes += AlignUbBytes(inputChannels * 2) + AlignUbBytes(4 * matrixElements * 2) +
                        AlignUbBytes(outputChannels * 2);
    }
    const uint64_t zeroBytes = AlignUbBytes(kWholeOutputCapacityElements * (bf16 ? 2 : 4));
    return std::max(messageBytes + zeroBytes, kSparseControlBytes + zeroBytes) +
           kSparseAllocationAllowanceBytes;
}

// UnitCubeSingleEdge uses at most 8192 knot elements; GeneralSingleEdge's
// 16384-element reserve also bounds its extra 128-element feature prefix.
static uint64_t GeneralSingleFootprintBytes(uint64_t inputChannels, uint64_t outputChannels, bool bf16) {
    const uint64_t matrixElements = inputChannels * outputChannels;
    const uint64_t broadcastElements = inputChannels * (outputChannels <= 32 ? outputChannels : 8);
    uint64_t bytes = kSparseControlBytes + AlignUbBytes(inputChannels * 4) +
                     AlignUbBytes(kSingleOwnerWeightCapacityElements * 4) + kResultPhasePaddingBytes +
                     AlignUbBytes(matrixElements * SupportBatchWidth(matrixElements, 8) * 4) + kOperandPhasePaddingBytes +
                     AlignUbBytes(broadcastElements * 4) + kWholeOutputCapacityElements * 4 +
                     AlignUbBytes(outputChannels * 4); // Independent per-message support sum.
    if (bf16) {
        bytes = Bf16StorageStartBytes(bytes);
        bytes += AlignUbBytes(inputChannels * 2) + AlignUbBytes(kSingleOwnerWeightCapacityElements * 2) +
                 kWholeOutputCapacityElements * 2;
    }
    return bytes + kSparseAllocationAllowanceBytes;
}

// This function reads descriptors/attributes only. Failure means no launch;
// tensor values (including edge endpoints and coordinates) stay on device.
static ge::graphStatus ReadValidatedMetadata(gert::TilingContext *context, SplineConvTilingData &data,
                                             uint32_t &knotCount, bool &bf16) {
    const auto x = context->GetInputShape(0);
    const auto edges = context->GetInputShape(1);
    const auto attributes = context->GetInputShape(2);
    const auto weights = context->GetInputShape(3);
    if (!x || !edges || !attributes || !weights)
        return ge::GRAPH_FAILED;
    const auto xShape = x->GetStorageShape();
    const auto edgeShape = edges->GetStorageShape();
    const auto attributeShape = attributes->GetStorageShape();
    const auto weightShape = weights->GetStorageShape();
    if (xShape.GetDimNum() != 2 || edgeShape.GetDimNum() != 2 || attributeShape.GetDimNum() != 2 ||
        weightShape.GetDimNum() != 3)
        return ge::GRAPH_FAILED;
    const auto attrs = context->GetAttrs();
    if (!attrs)
        return ge::GRAPH_FAILED;
    const auto dimensions = attrs->GetInt(0);
    const auto knotSizes = attrs->GetListInt(1);
    const auto openFlags = attrs->GetListInt(2);
    const auto degree = attrs->GetInt(3);
    const auto aggregation = attrs->GetStr(4);
    if (!dimensions || !degree || !knotSizes || !openFlags || !aggregation || *dimensions < 2 ||
        *dimensions > 3 || *degree < 1 || *degree > 3 || knotSizes->GetSize() != *dimensions ||
        openFlags->GetSize() != *dimensions)
        return ge::GRAPH_FAILED;

    const int64_t nodeCount = xShape.GetDim(0);
    const int64_t inputChannels = xShape.GetDim(1);
    const int64_t edgeCount = edgeShape.GetDim(1);
    const int64_t outputChannels = weightShape.GetDim(2);
    if (nodeCount < 1 || nodeCount > 65536 || edgeCount < 0 || edgeCount > 262144 || inputChannels < 1 ||
        inputChannels > 128 || outputChannels < 1 || outputChannels > 128 || edgeShape.GetDim(0) != 2 ||
        attributeShape.GetDim(0) != edgeCount || attributeShape.GetDim(1) != *dimensions ||
        weightShape.GetDim(1) != inputChannels)
        return ge::GRAPH_FAILED;

    data.nodeCount = nodeCount;
    data.edgeCount = edgeCount;
    data.inputChannels = inputChannels;
    data.outputChannels = outputChannels;
    data.alignedOutputChannels = (outputChannels + 7) / 8 * 8;
    data.scratchRowElements = (outputChannels + 8) / 8 * 8;
    data.dimensionCount = *dimensions;
    data.degree = *degree;
    data.supportCount = 1;
    knotCount = 1;
    for (uint32_t axis = 0; axis < data.dimensionCount; ++axis) {
        const int64_t axisKnotCount = knotSizes->GetData()[axis];
        const int64_t open = openFlags->GetData()[axis];
        if (axisKnotCount < 2 || axisKnotCount > 4 || (open != 0 && open != 1) ||
            (open && axisKnotCount < *degree + 1))
            return ge::GRAPH_FAILED;
        data.knotSizes[axis] = axisKnotCount;
        data.coordinateScales[axis] = axisKnotCount - *degree * open;
        knotCount *= axisKnotCount;
        data.supportCount *= *degree + 1;
    }
    // Retains the operator's work limit; it also bounds 32-bit device jobs.
    if (weightShape.GetDim(0) < knotCount ||
        uint64_t(edgeCount) * data.supportCount * inputChannels * outputChannels > (1ULL << 31))
        return ge::GRAPH_FAILED;
    if (std::strcmp(aggregation, "mean") == 0)
        data.applyMean = 1;
    else if (std::strcmp(aggregation, "add") != 0)
        return ge::GRAPH_FAILED;

    const auto dtype = context->GetInputDesc(0)->GetDataType();
    bf16 = dtype == ge::DT_BF16;
    if (bf16)
        data.alignedOutputChannels = (outputChannels + 15) / 16 * 16;
    if (dtype != ge::DT_FLOAT && !bf16)
        return ge::GRAPH_FAILED;
    if (context->GetInputDesc(1)->GetDataType() != ge::DT_INT64 ||
        context->GetInputDesc(2)->GetDataType() != dtype || context->GetInputDesc(3)->GetDataType() != dtype)
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

static void SelectGeneralStrategy(SplineConvTilingData &data, uint32_t knotCount, bool bf16,
                                  uint32_t vectorCoreCount, uint64_t ubBytes) {
    const uint64_t matrixElements = uint64_t(data.inputChannels) * data.alignedOutputChannels;
    data.cacheWeights =
        uint64_t(knotCount) * matrixElements * (bf16 ? 6 : 4) + matrixElements * 4 + kGenericHeadroomBytes <
        ubBytes;
    // Independent output ownership works for every valid edge count.
    data.directOutput = 1;
    data.precomputeFeatures = 0;
    if (data.directOutput && knotCount > data.supportCount) {
        // BF16 converts every cached element, hence its smaller cache budget.
        const uint64_t cachedWeightElements = uint64_t(knotCount) * matrixElements;
        const uint64_t cachedWeightBudgetElements = bf16 ? 2048 : 8192;
        if (cachedWeightElements > cachedWeightBudgetElements)
            data.cacheWeights = 0;
    }
    data.contractKnots = 0;
    data.precomputeFeatures = 0;
    data.workerCount = std::min<uint32_t>(vectorCoreCount, data.nodeCount);
}

struct KernelDispatch {
    uint32_t mode;
    uint32_t requestedBlockCount;
    uint32_t inputChannels;
    uint32_t outputChannels;
    uint32_t staticEdgeCount;
};

static uint64_t PackSparseMetadata(const SplineConvTilingData &data, bool coowned) {
    return uint64_t(data.nodeCount) | (uint64_t(data.edgeCount) << kSparseEdgeShift) |
           (uint64_t(data.zeroTileRowCount) << kZeroTileRowsShift) |
           (uint64_t(data.applyMean) << kMeanShift) | (uint64_t(coowned) << kCoownedShift) |
           (uint64_t(data.inputChannels) << kInputChannelsShift);
}
static uint64_t PackSingleMetadata(const SplineConvTilingData &data) {
    return uint64_t(data.nodeCount) | (uint64_t(data.knotSizes[2]) << kKnotSizeZShift) |
           (uint64_t(data.coordinateScales[2]) << kScaleZShift) |
           (uint64_t(data.dimensionCount == 3) << kIs3dShift) |
           (uint64_t(data.knotSizes[0]) << kKnotSizeXShift) |
           (uint64_t(data.knotSizes[1]) << kKnotSizeYShift) |
           (uint64_t(data.coordinateScales[0]) << kScaleXShift) |
           (uint64_t(data.coordinateScales[1]) << kScaleYShift);
}
static bool IsPowerOfTwo(uint32_t count) {
    return count != 0 && (count & (count - 1)) == 0;
}

// Families cover every channel pair satisfying their alignment/capacity
// bounds. If any condition fails, the owned-row MODE1 handles the full input.
// Priority is retained: runtime sparse -> static bucket -> small single ->
// general single -> unit cube. N remains runtime metadata in every family.
static KernelDispatch SelectFastStrategy(SplineConvTilingData &data, uint32_t knotCount, bool bf16,
                                         uint32_t vectorCoreCount, uint64_t ubBytes) {
    // Two-worker single-edge entries statically own the message and its
    // zero complement. A platform with fewer workers uses the generic owner.
    if (vectorCoreCount < 2)
        return {data.directOutput, data.workerCount, 0, 0, 0};
    const uint32_t inputChannels = data.inputChannels;
    const uint32_t outputChannels = data.outputChannels;
    const uint32_t edgeCount = data.edgeCount;
    const uint32_t nodeCount = data.nodeCount;
    const uint32_t dtypeBytes = bf16 ? 2 : 4;
    const uint32_t matrixElements = inputChannels * outputChannels;
    const uint64_t outputElements = uint64_t(nodeCount) * outputChannels;
    const bool bilinear = data.dimensionCount == 2 && data.degree == 1 && data.knotSizes[0] == 2 &&
                          data.knotSizes[1] == 2 && data.coordinateScales[0] == 1 &&
                          data.coordinateScales[1] == 1;
    const bool smallSingleResources =
        bilinear && edgeCount == 1 && inputChannels % 8 == 0 && outputChannels % 8 == 0 &&
        matrixElements <= 256 && outputElements <= 1024 &&
        uint64_t(128 * 1024 + 512 + 32 + kSparseAllocationAllowanceBytes) < ubBytes;
    // The legal prefix contains floor(128/CI) complete rows: CI=24 uses 120
    // elements. The new family is exactly the capacity complement, so every
    // previously eligible prefix keeps its existing entry and DMA schedule.
    const uint32_t prefixRows = kSingleOwnerFeaturePrefixElements / inputChannels;
    const bool smallSingle = smallSingleResources && nodeCount >= prefixRows;
    const bool boundedSingle = smallSingleResources && nodeCount < prefixRows;
    const uint32_t zeroTileRows =
        std::min<uint32_t>(nodeCount, kWholeOutputCapacityElements / outputChannels);
    const bool alignedChannels = inputChannels >= 8 && outputChannels >= 8 && IsPowerOfTwo(inputChannels) &&
                                 IsPowerOfTwo(outputChannels);
    bool runtimeSparse = bilinear && edgeCount > 0 && edgeCount < vectorCoreCount &&
                         edgeCount <= kSparseControlEdgeCapacity && alignedChannels && matrixElements <= 4096;
    const bool generalSingle = !smallSingle && !boundedSingle && data.degree == 1 && edgeCount == 1 &&
                               alignedChannels && matrixElements <= 1024 &&
                               uint64_t(knotCount) * matrixElements <= kSingleOwnerWeightCapacityElements &&
                               outputElements <= kWholeOutputCapacityElements &&
                               GeneralSingleFootprintBytes(inputChannels, outputChannels, bf16) < ubBytes;
    const uint32_t intervalRows = edgeCount > 0 ? (nodeCount + edgeCount - 1) / edgeCount : 0;
    // A 64-bit occupied-row mask bounds a coowned interval to 64 rows.
    bool coowned = runtimeSparse && edgeCount <= kPreferredVectorWorkerCount &&
                   outputElements <= kWholeOutputCapacityElements && intervalRows <= 64 &&
                   outputChannels * dtypeBytes >= 32;
    runtimeSparse =
        runtimeSparse && RuntimeSparseFootprintBytes(inputChannels, outputChannels, bf16) < ubBytes;
    coowned = coowned && runtimeSparse;
    const uint32_t zeroTileCount = (nodeCount + zeroTileRows - 1) / zeroTileRows;
    const uint32_t zeroWorkerCount =
        runtimeSparse ? std::min<uint32_t>(vectorCoreCount - edgeCount, zeroTileCount) : 0;
    KernelDispatch dispatch{data.directOutput, data.workerCount, 0, 0, 0};
    if (runtimeSparse) {
        dispatch.mode = coowned ? kSparseCoownedZeroMode : kSparseSeparateZeroMode;
        dispatch.inputChannels = inputChannels;
        dispatch.outputChannels = outputChannels;
        data.zeroTileRowCount = coowned ? intervalRows : zeroTileRows;
        data.packedMetadata = PackSparseMetadata(data, coowned);
        dispatch.requestedBlockCount =
            coowned ? std::min(kPreferredVectorWorkerCount, vectorCoreCount) : edgeCount + zeroWorkerCount;
    }
    // All power-of-two edge counts through the preferred worker-group size
    // are compiled. A full group owns disjoint zero intervals itself; smaller
    // buckets retain a separate zero owner. Other edge counts use runtime sparse.
    const uint32_t staticWorkerCount = (edgeCount + 1 + kPreferredVectorWorkerCount - 1) /
                                       kPreferredVectorWorkerCount * kPreferredVectorWorkerCount;
    const bool staticFullGraphResources =
        runtimeSparse && edgeCount <= kPreferredVectorWorkerCount &&
        outputElements <= kWholeOutputCapacityElements && outputChannels * dtypeBytes >= 32 &&
        staticWorkerCount <= vectorCoreCount &&
        (edgeCount != kPreferredVectorWorkerCount || coowned);
    const bool staticFullGraph = staticFullGraphResources && IsPowerOfTwo(edgeCount);
    if (staticFullGraph) {
        dispatch.mode = kStaticSparseMode;
        dispatch.inputChannels = inputChannels;
        dispatch.outputChannels = outputChannels;
        dispatch.staticEdgeCount = (edgeCount > 1 && IsPowerOfTwo(edgeCount)) ? edgeCount : 0;
        // Exactly one complete owner group can clear disjoint intervals itself.
        // Smaller static buckets retain their separate zero-complement owner.
        dispatch.requestedBlockCount = edgeCount == kPreferredVectorWorkerCount ? edgeCount : staticWorkerCount;
    }
    if (smallSingle || boundedSingle) {
        dispatch = {kSmallBilinearSingleMode,
                    std::min(kPreferredVectorWorkerCount, vectorCoreCount), inputChannels, outputChannels, 0};
        data.packedMetadata = uint64_t(nodeCount);
    }
    if (generalSingle) {
        dispatch = {kGeneralSingleMode, std::min(kPreferredVectorWorkerCount, vectorCoreCount), inputChannels,
                    outputChannels, 0};
        data.packedMetadata = PackSingleMetadata(data);
    }
    // The mathematical family is the entire open degree-one 2x2x2 grid.
    // Insufficient prefix capacity selects actual-row DMA. Otherwise actual
    // source membership in the prefetched rows is checked on device.
    const bool unitCubeSingle =
        generalSingle && data.dimensionCount == 3 && data.knotSizes[0] == 2 && data.knotSizes[1] == 2 &&
        data.knotSizes[2] == 2 && data.coordinateScales[0] == 1 && data.coordinateScales[1] == 1 &&
        data.coordinateScales[2] == 1;
    if (unitCubeSingle)
        dispatch.mode = uint64_t(nodeCount) * inputChannels < kSingleOwnerFeaturePrefixElements
                            ? kBoundedUnitCubeSingleMode : kUnitCubeSingleMode;
    if (dispatch.mode == kSparseSeparateZeroMode || dispatch.mode == kSparseCoownedZeroMode) {
        // The cache holds every source row in the input dtype. Row alignment
        // makes each actual source view legal for vector input or BF16 Cast.
        // Bound extra DMA to one CI-by-CO matrix: at most one quarter of
        // the four knot matrices each active owner already loads. Retain the
        // full existing live UB budget, including its conservative padding.
        const uint64_t featureBytes = uint64_t(nodeCount) * inputChannels * dtypeBytes;
        const uint64_t messageMatrixInputBytes = uint64_t(matrixElements) * dtypeBytes;
        const bool cacheFeatures =
            featureBytes <= messageMatrixInputBytes && inputChannels * dtypeBytes % kUbAlignmentBytes == 0 &&
            RuntimeSparseFootprintBytes(inputChannels, outputChannels, bf16) + featureBytes < ubBytes;
        if (cacheFeatures)
            dispatch.mode = dispatch.mode == kSparseSeparateZeroMode ? kCachedSparseSeparateZeroMode
                                                                     : kCachedSparseCoownedZeroMode;
    }
    // Complete the small static-count family only after preserving the
    // full-input cache choices above. Every remaining coowned owner group
    // already launches eight workers; the same group can instead use the
    // existing message owners and one independent zero-complement owner.
    // Powers of two were selected earlier, leaving E=3/5/6/7 here.
    if (dispatch.mode == kSparseCoownedZeroMode && staticFullGraphResources) {
        dispatch = {kStaticSparseMode, staticWorkerCount, inputChannels, outputChannels,
                    (edgeCount > 1 && IsPowerOfTwo(edgeCount)) ? edgeCount : 0};
    }
    // Node and partial-owner counts stay runtime metadata. Single-edge
    // kernels stage the actual source row and never require an N selector.
    // Preserve the eight-worker launch group used by the platform baseline.
    // Small independent owner groups measured slower on that platform; unused
    // lanes return without graph computation. Unaligned BF16 rows retain the
    // single whole-output owner to avoid overlapping zero stores.
    const bool singleOwner = dispatch.mode == kSmallBilinearSingleMode ||
                             dispatch.mode == kBoundedBilinearSingleMode ||
                             dispatch.mode == kCachedCoordinateSingleMode ||
                             dispatch.mode == kGeneralSingleMode ||
                             dispatch.mode == kUnitCubeSingleMode ||
                             dispatch.mode == kBoundedUnitCubeSingleMode;
    if (singleOwner && outputChannels * dtypeBytes % kUbAlignmentBytes != 0)
        dispatch.requestedBlockCount = 1;
    // On >=16 AIV workers, eligible coowned graphs already select MODE6
    // or MODE10. Smaller devices retain complete generic row ownership,
    // allowing the unused MODE4 executable entries to be omitted.
    if (dispatch.mode == kSparseCoownedZeroMode)
        dispatch = {data.directOutput, data.workerCount, 0, 0, 0};
    // Dispatch already satisfies the underlying algorithm's full resource
    // contract. Complete32-byte DMA blocks cover the whole bounded input.
    const uint64_t inputBytes = uint64_t(nodeCount) * inputChannels * dtypeBytes;
    const bool cachedFloatSingle = !bf16 && dispatch.mode == kSmallBilinearSingleMode &&
        inputChannels <= 24 && outputChannels <= 32 && matrixElements <= 256 &&
        (4 * outputChannels) % kFloatVectorElements == 0 &&
        inputBytes >= kUbAlignmentBytes && inputBytes <= kCachedFloatInputCapacityBytes &&
        inputBytes % kUbAlignmentBytes == 0 &&
        (inputBytes % kGmCacheLineBytes == 0 || vectorCoreCount >= kPreferredVectorWorkerCount);
    if (cachedFloatSingle) {
        // Both classes contain the complete input. A partial GMcache line
        // uses the measured coordinate-DMA entry; full lines retain the
        // scalar-coordinate entry. No graph or coordinate values are read.
        dispatch.mode = inputBytes % kGmCacheLineBytes != 0
            ? kCachedCoordinateSingleMode : kCachedFloatSingleMode;
    }
    // The complete owner group and packed product slab bound pair batching.
    // No graph values or destination counts are read on Host.
    const uint64_t packedProductBytes = uint64_t(4) * matrixElements * sizeof(float);
    const bool pairedIncoming = !bf16 && dispatch.mode == kStaticSparseMode &&
        dispatch.staticEdgeCount == kPairedOwnerGroupWidth &&
        outputChannels >= 16 && outputChannels <= 32 &&
        kFloatVectorElements % outputChannels == 0 &&
        packedProductBytes >= kPairedProductMinimumBytes &&
        packedProductBytes <= kPairedProductMaximumBytes;
    if (pairedIncoming)
        dispatch.mode = kPairedIncomingGraphMode;
    return dispatch;
}

// Generic nodes have disjoint owners. Dense row tiles avoid atomics and
// cross-core barriers; large index scans use private device-built edge lists.
static void ConfigureDirectRows(SplineConvTilingData &data, uint32_t knotCount, bool bf16, uint64_t ubBytes) {
    const uint64_t matrixElements = uint64_t(data.inputChannels) * data.alignedOutputChannels;
    const auto fixedBytes = [&](bool cacheWeights) {
        const uint64_t cachedKnots = cacheWeights ? knotCount : 1;
        return cachedKnots * matrixElements * (bf16 ? 6 : 4) + matrixElements * 4 +
               (bf16 ? 0 : 32) + 4096 + 512 +
               uint64_t(data.scratchRowElements + data.alignedOutputChannels) * 4 +
               kOwnedControlBufferBytes + kGenericHeadroomBytes;
    };
    const uint64_t rowBytes = uint64_t(data.alignedOutputChannels) * 4 + sizeof(int32_t);
    if (data.cacheWeights && fixedBytes(true) + rowBytes + 64 >= ubBytes)
        data.cacheWeights = 0;
    const uint64_t baseBytes = fixedBytes(data.cacheWeights != 0);
    const uint32_t rowsPerOwner = (data.nodeCount + data.workerCount - 1) / data.workerCount;
    data.rowCapacity = baseBytes + rowBytes + 64 < ubBytes
                           ? std::min<uint64_t>(rowsPerOwner, (ubBytes - baseBytes - 64) / rowBytes)
                           : 0;
    data.directRowIntervals = data.rowCapacity != 0;
    data.usePrivateEdgeBuckets = data.directRowIntervals && rowsPerOwner > data.rowCapacity &&
                                 data.edgeCount >= kOwnedDestinationBatchEntries;
    if (!data.directRowIntervals)
        data.rowCapacity = 1;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto data = context->GetTilingData<SplineConvTilingData>();
    *data = {};
    uint32_t knotCount = 0;
    bool bf16 = false;
    if (ReadValidatedMetadata(context, *data, knotCount, bf16) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t vectorCoreCount = platform.GetCoreNumAiv();
    SelectGeneralStrategy(*data, knotCount, bf16, vectorCoreCount, ubBytes);
    auto dispatch = SelectFastStrategy(*data, knotCount, bf16, vectorCoreCount, ubBytes);
    if (dispatch.mode == kDirectOutputMode)
        ConfigureDirectRows(*data, knotCount, bf16, ubBytes);
    const uint32_t launchWorkerCount = GroupedLaunchWorkerCount(dispatch.requestedBlockCount, vectorCoreCount);
    context->SetBlockDim(launchWorkerCount);
    context->SetScheduleMode(0);
    if (dispatch.mode == kDirectOutputMode && data->usePrivateEdgeBuckets) {
        const uint64_t entriesPerOwner =
            (uint64_t(data->edgeCount) + kOwnedEdgeWorkspaceAlignmentEntries - 1) /
            kOwnedEdgeWorkspaceAlignmentEntries * kOwnedEdgeWorkspaceAlignmentEntries;
        context->GetWorkspaceSizes(1)[0] = entriesPerOwner * sizeof(int64_t) * data->workerCount;
    }
    // Keep original resource/workspace/launch decisions; specialize only the key.
    const uint32_t MODE = dispatch.mode;
    // GeneralSingle carries dimensions in packed metadata. Canonical DIM=2
    // avoids duplicate entries; UnitCube has the mathematical DIM=3 key.
    const uint32_t DIM = MODE == kGeneralSingleMode ? 2 : data->dimensionCount;
    const uint32_t DEG = data->degree;
    const uint32_t CI = dispatch.inputChannels;
    const uint32_t CO = dispatch.outputChannels;
    const uint32_t ES = dispatch.staticEdgeCount;
    // Native input0 datatype binds each compiled artifact to its actual T.
    // NN carries a complete DMA-block extent only in the dedicated FP32
    // input-cache family. Existing BF16 hint encodings stay unchanged.
    const uint64_t inputBytes = uint64_t(data->nodeCount) * CI * (bf16 ? 2 : 4);
    const bool hintFamily = bf16 && MODE == kSmallBilinearSingleMode && CI <= 24 &&
                            CO <= 32 && CI * CO <= 256 && (CO * 2) % kUbAlignmentBytes != 0 &&
                            inputBytes <= 512;
    const uint32_t NN = (MODE == kCachedFloatSingleMode || MODE == kCachedCoordinateSingleMode)
        ? uint32_t(inputBytes / kUbAlignmentBytes)
        : (hintFamily ? uint32_t((inputBytes + 63) / 64) : 0);
    const uint32_t INPUT_T = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, MODE, DIM, DEG, CI, CO, ES, NN, INPUT_T);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
    auto x = c->GetInputShape(0), w = c->GetInputShape(3);
    if (!x || !w || x->GetDimNum() != 2 || w->GetDimNum() != 3)
        return GRAPH_FAILED;
    auto o = c->GetOutputShape(0);
    o->SetDimNum(2);
    o->SetDim(0, x->GetDim(0));
    o->SetDim(1, w->GetDim(2));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
    c->SetOutputDataType(0, c->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge
namespace ops {
class SplineConv : public OpDef {
  public:
    explicit SplineConv(const char *name) : OpDef(name) {
        Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Input("edge_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Input("edge_attr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Attr("dim").AttrType(OPTIONAL).Int(2);
        Attr("kernel_size").AttrType(OPTIONAL).ListInt({2, 2});
        Attr("is_open_spline").AttrType(OPTIONAL).ListInt({1, 1});
        Attr("degree").AttrType(OPTIONAL).Int(1);
        Attr("aggr").AttrType(OPTIONAL).String("mean");
        SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(SplineConv);
} // namespace ops
