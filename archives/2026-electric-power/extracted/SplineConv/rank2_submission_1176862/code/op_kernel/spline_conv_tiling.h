#pragma once
#include <cstddef>
#include <cstdint>

struct SplineResidentTilingData {
    uint32_t nodes;
    uint32_t edges;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t mean;
    uint32_t owners;
    uint32_t paddedInput;
    uint32_t ownerShift;
};

struct SplineConvTilingData {
    uint32_t nodes;
    uint32_t edges;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t mean;
    uint32_t owners;
    uint32_t paddedInput;
    uint32_t ownerShift;
    uint32_t dim;
    uint32_t degree;
    uint32_t hasRoot;
    uint32_t hasBias;
    uint32_t nodeTile;
    uint32_t edgeStride;
    uint32_t blocks;
    uint32_t schedule;
    uint32_t kernelSize[3];
    uint32_t open[3];
};
static_assert(offsetof(SplineConvTilingData, dim) == sizeof(SplineResidentTilingData),
              "Resident tiling must be the full record's 32-byte prefix");
