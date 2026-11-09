#pragma once
#include <cstdint>

namespace elec {
// Each pair is calculated in FP32. For width=32, 4 points per core (128 lanes)
// doubles multi-core parallelism on Case 3 (33 points -> 9 cores) while keeping
// Case 2 (width=8 -> 8 points per core) at 1 single 64-lane repeat per tile.
constexpr uint64_t tree_short_point_tile(uint64_t /*element_bytes*/,uint64_t width) {
    return width>16?4:8;
}
constexpr uint64_t tree_short_align(uint64_t bytes) {return (bytes+31)/32*32;}
inline uint64_t tree_short_ub_bytes(uint64_t element_bytes,uint64_t width,uint64_t tile) {
    const uint64_t pairs=width*tile,stage=6*width+3*tile;
    const uint64_t work_stride=(pairs&64)?pairs:(pairs+64);
    return tree_short_align(stage*element_bytes)+tree_short_align(stage*sizeof(float))+
        tree_short_align(12*work_stride*sizeof(float))+tree_short_align((3*pairs+tile)*sizeof(int32_t))+
        160*sizeof(float)+tree_short_align(pairs/8<32?32:pairs/8);
}
}
