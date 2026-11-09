#pragma once
#include <cstdint>
struct LinalgLstsqTilingData {
    uint32_t m, n, k, batches, rows, blocks;
    uint64_t residualElements;
    uint32_t teams;
};

constexpr uint32_t kOrthogonalWorkspaceFloats=438800;

// Fixed workspace stride for the 96/112-row panels and accurate residual pipeline.
constexpr uint32_t kNarrowOrthogonalWorkspaceFloats=587520;
