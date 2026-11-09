#pragma once
#include <cstdint>
struct LinalgDetTilingData {uint32_t n, rows, batches, blocks, panel, team, groups, reserved;};
struct LinalgDetMetadata {LinalgDetTilingData base;uint32_t rowOffsets[64],diagonalOffsets[64];};

static constexpr uint32_t DET_CUBE_BLOCK=64;
