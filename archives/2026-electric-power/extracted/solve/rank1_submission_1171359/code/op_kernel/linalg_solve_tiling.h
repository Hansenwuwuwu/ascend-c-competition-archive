// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LinalgSolveTilingData {
    uint32_t n, k, batches, broadcastB, cores, resident, grouped;
};
