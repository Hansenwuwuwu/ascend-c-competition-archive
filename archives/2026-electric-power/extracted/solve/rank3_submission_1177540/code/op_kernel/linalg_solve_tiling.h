#pragma once
#include <cstdint>
struct LinalgSolveTilingData {
    uint32_t n, k, batches, rows, blocks, broadcastB, teams;
};

struct LinalgSolveCubeTilingData {
    LinalgSolveTilingData base;
    uint32_t cubeWords[128];
    uint32_t halfWords[128];
    uint32_t systemWorkspaceBytes;
};

// CT, inverse, negative H, normalized A; two B64 blocks, RHS buffers, reduction slots, and local-Newton scratch.
constexpr uint32_t kOrthogonalWorkspaceFloats=4*256*256+2*64*64+4*256*32+256+3*64*64+4096;

// Narrow reprojected QR tail and six signed base-128 residual weights.
constexpr uint32_t kNarrowOrthogonalWorkspaceFloats=538368;
