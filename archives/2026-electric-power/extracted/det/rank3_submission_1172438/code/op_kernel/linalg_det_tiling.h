#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct LinalgDetConfig {
    uint32_t n;
    uint32_t batch;
    uint32_t paddedN;
    uint32_t coresPerMatrix;
    uint32_t groups;
    uint32_t columnsPerCore;
    uint32_t sharedStride;
    uint32_t cubeMode;
    uint32_t robustOffset;
};

struct LinalgDetTilingData {
    uint32_t n;
    uint32_t batch;
    uint32_t paddedN;
    uint32_t coresPerMatrix;
    uint32_t groups;
    uint32_t columnsPerCore;
    uint32_t sharedStride;
    uint32_t cubeMode;
    uint32_t robustOffset;
    AscendC::tiling::TCubeTiling cube;
};

struct LinalgDetTilingPayload {
    LinalgDetTilingData base;
    alignas(32) uint32_t gatherIndices[128+256];
};
