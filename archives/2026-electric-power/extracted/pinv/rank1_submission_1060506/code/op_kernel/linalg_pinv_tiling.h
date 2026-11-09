#pragma once
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include "pinv_contract.h"
// WINDOW templates carry the window width in local until the fallback restores zero.
struct LinalgPinvTilingData {
    uint32_t m, n, batch, rows, cols, stride, local, parallel, localInverse, inverseTile, resident, groupCount, singleResident;
    float atol, rtol;
};

#include "kernel_tiling/kernel_tiling.h"
struct LinalgPinvCubeTilingData { LinalgPinvTilingData base; AscendC::tiling::TCubeTiling cube; AscendC::tiling::TCubeTiling correction; };

// The CANN base packet is exactly the first object in an extended Cube packet.
static_assert(std::is_standard_layout<LinalgPinvTilingData>::value, "base tiling must be standard layout");
static_assert(std::is_standard_layout<LinalgPinvCubeTilingData>::value, "Cube tiling must be standard layout");
static_assert(offsetof(LinalgPinvCubeTilingData, base) == 0, "base tiling offset is part of the ABI");
static_assert(sizeof(LinalgPinvTilingData) == 15 * sizeof(uint32_t), "base tiling ABI changed");
static_assert(offsetof(LinalgPinvCubeTilingData, cube) >= sizeof(LinalgPinvTilingData), "Cube data overlaps the base");
