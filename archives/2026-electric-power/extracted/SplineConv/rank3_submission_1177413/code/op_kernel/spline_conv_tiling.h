#pragma once
#include <cstdint>

// Counts are elements unless named Bytes. FP32 vectors process 64 elements
// per repeat; DMA/UB allocation boundaries are 32 bytes.
static constexpr uint32_t kUbAlignmentBytes = 32;
static constexpr uint32_t kFloatVectorElements = 64;
// Two 2-KiB scalar-control halves stage destinations and packed edge IDs.
static constexpr uint32_t kOwnedControlBufferBytes = 4096;
static constexpr uint32_t kOwnedDestinationBatchEntries = kOwnedControlBufferBytes / (2 * sizeof(int64_t));
// Each owner starts on a separate 64-byte GM cache line.
static constexpr uint32_t kOwnedEdgeWorkspaceAlignmentEntries = 8;
// Control regions use separate 512-byte phases. The remaining allowance is
// conservative padding retained from the validated raw-UB allocation budget.
static constexpr uint32_t kSparseControlBytes = 2048;
static constexpr uint32_t kSparseAllocationAllowanceBytes = 8448;
static constexpr uint32_t kResultPhasePaddingBytes = 256;
static constexpr uint32_t kOperandPhasePaddingBytes = 32;
static constexpr uint32_t kBf16StorageBoundaryBytes = 65536;
static constexpr uint32_t kWholeOutputCapacityElements = 4096;
// 64 KiB of FP32 knots plus BF16 input storage fit alongside a whole output;
// host tiling checks the complete UB footprint before choosing this family.
static constexpr uint32_t kSingleOwnerWeightCapacityElements = 16384;
// A 512-byte FP32 prefix fits two vector repeats. The host proves that these
// elements exist; the device checks that the actual source row is in them.
static constexpr uint32_t kSingleOwnerFeaturePrefixElements = 128;

// MODE is an algorithm/resource family, never a graph identifier.
// MODE0 remains an enum placeholder; its mixed entry is not instantiated.
static constexpr uint32_t kGlobalAccumulationMode = 0;
static constexpr uint32_t kDirectOutputMode = 1;
static constexpr uint32_t kSmallBilinearSingleMode = 2;
static constexpr uint32_t kSparseSeparateZeroMode = 3;
static constexpr uint32_t kSparseCoownedZeroMode = 4;
static constexpr uint32_t kGeneralSingleMode = 5;
static constexpr uint32_t kStaticSparseMode = 6;
static constexpr uint32_t kBoundedUnitCubeSingleMode = 7;
static constexpr uint32_t kUnitCubeSingleMode = 8;
static constexpr uint32_t kCachedSparseSeparateZeroMode = 9;
static constexpr uint32_t kCachedSparseCoownedZeroMode = 10;
static constexpr uint32_t kBoundedBilinearSingleMode = 11;
// A partial64-byte cache line still contains complete32-byte DMA blocks.
static constexpr uint32_t kCachedCoordinateSingleMode = 12;
// Metadata-only resource families with dedicated device entries.
static constexpr uint32_t kCachedFloatSingleMode = 14;
static constexpr uint32_t kPairedIncomingGraphMode = 15;
static constexpr uint32_t kGmCacheLineBytes = 64;
static constexpr uint32_t kCachedFloatInputCapacityBytes = 512;
static constexpr uint32_t kPairedProductMinimumBytes = 16 * 1024;
static constexpr uint32_t kPairedProductMaximumBytes = 32 * 1024;
static constexpr uint32_t kPairedOwnerGroupWidth = 8;

// packedMetadata is the first word so fast owners can DMA just the controls.
// Sparse modes 3/4/6/9/10: N[16:0], E[32:17], zero rows[44:33], mean[45],
// coowned[46], CI[54:47]. Single modes 5/8 reuse non-N fields for knot grid
// metadata: Kz[19:17], scaleZ[22:20], is3D[23], Kx[49:47], Ky[52:50],
// scaleX[55:53], scaleY[58:56]. Modes 2/11 need only N. Field widths include
// N=65536, E<=64, zero rows<=512, CI<=128, and knot sizes/scales<=4.
static constexpr uint64_t kNodeCountMask = 0x1ffff;
static constexpr uint32_t kSparseEdgeShift = 17;
static constexpr uint32_t kSparseEdgeMask = 0xffff;
static constexpr uint32_t kZeroTileRowsShift = 33;
static constexpr uint32_t kZeroTileRowsMask = 0xfff;
static constexpr uint32_t kMeanShift = 45;
static constexpr uint32_t kCoownedShift = 46;
static constexpr uint32_t kInputChannelsShift = 47;
// Sparse metadata has a free bit above its 8-bit CI field.
static constexpr uint32_t kFusedNeighborShift = 55;
static constexpr uint32_t kMaxFusedNeighborEdges = 8;
static constexpr uint32_t kKnotSizeZShift = 17;
static constexpr uint32_t kScaleZShift = 20;
static constexpr uint32_t kIs3dShift = 23;
static constexpr uint32_t kKnotSizeXShift = 47;
static constexpr uint32_t kKnotSizeYShift = 50;
static constexpr uint32_t kScaleXShift = 53;
static constexpr uint32_t kScaleYShift = 56;
static constexpr uint32_t kGridFieldMask = 7;

// Field order preserves the host/device ABI and the 32-byte control DMA.
struct SplineConvTilingData {
    uint64_t packedMetadata;
    uint32_t nodeCount;
    uint32_t edgeCount;
    uint32_t zeroTileRowCount;
    uint32_t usePrivateEdgeBuckets;
    uint32_t inputChannels;
    uint32_t outputChannels;
    uint32_t alignedOutputChannels;
    uint32_t scratchRowElements; // FP32 output plus one incoming-edge count.
    uint32_t dimensionCount;
    uint32_t degree;
    uint32_t supportCount;
    uint32_t applyMean;
    uint32_t workerCount;
    uint32_t cacheWeights;
    uint32_t precomputeFeatures;
    uint32_t directOutput;
    uint32_t contractKnots;
    uint32_t directRowIntervals;
    uint32_t rowCapacity;
    uint32_t knotSizes[3];
    uint32_t coordinateScales[3];
};

// Bound the concurrently live XW product scratch by 16 KiB whenever one
// matrix fits that capacity. Larger matrices retain the original one-matrix
// scratch. This is a UB capacity rule shared by Host and Kernel.
constexpr uint32_t kSupportProductScratchLimitBytes = 16384;
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr uint32_t SupportBatchWidth(uint64_t matrixElements, uint32_t supportCount) {
    return supportCount >= 4 && matrixElements * 4 * sizeof(float) <= kSupportProductScratchLimitBytes
               ? 4
               : (supportCount >= 2 && matrixElements * 2 * sizeof(float) <= kSupportProductScratchLimitBytes ? 2 : 1);
}

// The full four-knot product is also the temporary for an in-kernel UB
// layout conversion. CO>=16 gives a packed CI row at least one FP32 vector;
// CO<=32 bounds it to two vectors. 32 KiB includes the M<=2048 family while
// retaining the existing one-matrix path for larger resources.
constexpr uint32_t kPackedBilinearProductLimitBytes = 32768;
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr bool UsePackedBilinearWeights(uint64_t inputChannels, uint64_t outputChannels) {
    return outputChannels >= 16 && outputChannels <= 32 &&
           inputChannels * outputChannels * 4 * sizeof(float) <= kPackedBilinearProductLimitBytes;
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr uint64_t BilinearProductElementCount(uint64_t inputChannels, uint64_t outputChannels) {
    return inputChannels * outputChannels *
           (UsePackedBilinearWeights(inputChannels, outputChannels)
                ? 4 : SupportBatchWidth(inputChannels * outputChannels, 4));
}

// UI_LIST slot13 remains reserved so every retained key keeps its encoding.
