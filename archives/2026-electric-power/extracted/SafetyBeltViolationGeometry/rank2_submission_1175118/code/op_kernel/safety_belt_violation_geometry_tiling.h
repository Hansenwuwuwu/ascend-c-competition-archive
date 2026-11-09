#pragma once
#include <cstdint>
// Legal batch capacity and the Float wide-family lower owner bound are shared
// by validation, tiling and the matching static device address calculation.
static constexpr uint32_t kGeometryMaxBatch=4096U;
static constexpr uint32_t kGeometryFloatWideOwners=kGeometryMaxBatch/512U;
static constexpr uint32_t kGeometryFoldMinLaunch=4U;
struct SafetyBeltViolationGeometryTilingData {
 uint32_t batch;
 uint32_t perCore;
 uint32_t hasThreshold;
 uint32_t thresholdIsHalf;
 float defaultThreshold;
 float waistRatio;
};
