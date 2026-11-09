// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LinalgEighTilingData {
    uint32_t n;
    uint32_t stride;
    uint32_t uplo;
    uint32_t active_cores;
    uint64_t batch_size;
    uint32_t num_groups;
    uint32_t group_lanes;
    uint32_t rows_per_lane;
    uint32_t vec_rows_per_lane;
    uint32_t solve_mode;
    uint32_t wsOffsetFloats;
};
