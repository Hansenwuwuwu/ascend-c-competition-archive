#pragma once
#include <cstdint>

constexpr uint32_t SWIGLU_GATHER_CAPACITY = 256;

struct SwiGluTilingData {
    uint32_t rows;
    uint32_t dim;
    uint32_t tileRows;
    uint32_t tileCols;
    uint32_t cores;
};
using SwiGluVectorTilingData = SwiGluTilingData;
static_assert(sizeof(SwiGluTilingData) == 20);
