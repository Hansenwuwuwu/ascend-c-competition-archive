#pragma once
#include <cstdint>

// Field order is shared by Host and Device. All sizes describe one invocation.
struct TreeLineMinDistanceTilingData {
  uint32_t batchCount;
  uint32_t pointsPerBatch;
  uint32_t segmentCount;
  uint32_t workerCount;
  uint32_t algorithm;
  uint32_t workersPerBatch;
  uint32_t unifiedBufferBytes;
};

// Local density sweeps amortize the extra SAT axes on larger segment tables.
constexpr uint32_t kSatMinimumSegmentCount = 768;

// Scalar work is bounded by point/segment pairs, independent of test cases.
constexpr uint32_t kMaximumScalarPairCount = 32;

// Packed layout descriptor: B-1[2:0], N-1[16:3], M-1[26:17],
// workers-1[32:27], UB bytes[50:33]. Bounds follow input limits and core/UB guards.
constexpr uint32_t kLayoutDescriptorPointShiftBits = 3;
constexpr uint32_t kLayoutDescriptorSegmentShiftBits = 17;
constexpr uint32_t kLayoutDescriptorWorkerShiftBits = 27;
constexpr uint32_t kLayoutDescriptorBufferShiftBits = 33;
constexpr uint32_t kLayoutDescriptorBufferWidthBits = 18;

// Scalar one-point-per-worker layouts: the existing worker budget is eight.
constexpr uint32_t kStaticScalarAlgorithmBase = 64;
constexpr uint32_t kStaticScalarPointShift = 4;
constexpr uint32_t kStaticScalarSegmentMask = 15;
constexpr uint32_t kStaticSingleSegmentAlgorithmBase = 320;
constexpr uint32_t kStaticSingleSegmentPointCapacity = 8;
// Generic vector layouts use the original width flag plus this disjoint tag.
constexpr uint32_t kOptimizedGenericAlgorithmBase = 384;

// The next16-tag group after single-batch M1 tags768..774. Append seven
// vector widths16..1024 after the existing212declaration entries.
constexpr uint32_t kWideOutputAlgorithmBase = 784;
constexpr uint32_t kWideOutputDeclarationBase = 212;
constexpr uint32_t kWideOutputWidthCount = 7;

// An int32 result for each point fits one32byte output block. At least
// two points amortize the shared current segment load and contiguous stores.
constexpr uint32_t kCoalescedSingleSegmentMaximumPoints = 32 / sizeof(int32_t);
constexpr uint32_t kCoalescedSingleSegmentAlgorithmBase = 848;
constexpr uint32_t kCoalescedSingleSegmentDeclarationBase = 219;

// Dense residual rows contain32..64points and fit the4096float scratch half.
// Exhaust every64-element physical pitch from256through1024.
constexpr uint32_t kDenseScratchAlgorithmBase = 864;
constexpr uint32_t kDenseScratchDeclarationBase = 226;
constexpr uint32_t kDenseScratchPitchCount = (1024 - 256) / 64 + 1;

// Two points did not establish a gain beyond observed drift. Three points
// and the full eight-point output-block boundary had stable measured gains.
constexpr uint32_t kCoalescedSingleSegmentMinimumPoints = 3;
