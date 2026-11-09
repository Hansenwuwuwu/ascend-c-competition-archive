#pragma once
#include <cstdint>

struct LinalgSolveTilingData {
    uint32_t n;
    uint32_t rows;
    uint32_t k;
    uint32_t batch;
    uint32_t bBatch;
    uint32_t cores;
    uint32_t stride;
    uint32_t parts;
    uint32_t groups;
};
