#pragma once
#include <cstdint>
struct FireCandidateDecodeFusionTilingData {
    uint32_t batch;
    uint32_t candidates;
    uint32_t cores;
    uint32_t segments;
};
