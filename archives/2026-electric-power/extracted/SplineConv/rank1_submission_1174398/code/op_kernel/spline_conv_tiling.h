#pragma once
#include <cstdint>
struct SplineConvTilingData {
    uint32_t n, e, mean, blocks, nodesPerBlock;
    uint32_t scale[8];
    uint32_t ci, co, dim, degree, slots, kernels;
    uint32_t ks[8];
    uint32_t ciPad, coPad, hasRoot, hasBias, cacheWeights;
    uint32_t streaming, cacheGraph, vectorDot, fusedDot, smallScalar;
    uint32_t edgeTrans, pseudoTrans, weightTrans;
};
