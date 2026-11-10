#pragma once
#include <cstdint>

// 数值保持稳定；kernel入口保留字面量供CANN扫描分派宏。
enum SwiGluKey : uint32_t {
    SWIGLU_SCALAR = 0,
    SWIGLU_ROW4 = 1,
    SWIGLU_ROW32 = 2,
    SWIGLU_ROW64 = 3,
    SWIGLU_ROWS13_STATIC = 4,
    SWIGLU_ROW_DYNAMIC = 6,
    SWIGLU_FULL_K = 7,
    SWIGLU_SPLIT_K = 8,
    SWIGLU_ROWS33 = 9,
    SWIGLU_CUBE256 = 10,
    SWIGLU_CUBE128 = 11,
    SWIGLU_RESIDENT_STATIC = 12,
    SWIGLU_ROWS13_DYNAMIC = 13,
};

struct SwiGluTilingData {
    uint32_t rows, dim;
    // Zero groups selects the generic kernel; otherwise every row group
    // contains dim / GATE_RESIDENT_BN column partitions.
    uint32_t residentGroups, residentPanels;
};

constexpr uint32_t GATE_RESIDENT_BM = 64;
constexpr uint32_t GATE_RESIDENT_BN = 128;
constexpr uint32_t GATE_RESIDENT_VM = 32;
// Four READY flags per projection, with one notification per four panels.
constexpr uint32_t GATE_RESIDENT_MAX_PANELS = 16;

constexpr uint32_t TILE_L1_BASEM = 64;
constexpr uint32_t TILE_L1_BASEN = 256;
constexpr uint32_t TILE_L1_BASEK = 256;
constexpr uint32_t TILE_L2_BASEN = 64;
constexpr uint32_t TILE_L2_BASEK = 256;

constexpr uint32_t SMALL13_ROWS = 7;
