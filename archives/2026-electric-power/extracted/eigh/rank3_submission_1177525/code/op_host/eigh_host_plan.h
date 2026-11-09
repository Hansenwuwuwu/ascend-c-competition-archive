#pragma once
// Host-only compile-time constexpr planning (C++11 compatible aggregate initialization).
#include "../op_kernel/eigh_case_traits.h"
#include "../op_kernel/linalg_eigh_tiling.h"

template<uint32_t CASE_KEY> struct EighHostPlan;

template<> struct EighHostPlan<1u> {
    static constexpr LinalgEighTilingData value = {
        16u, 1u, 0u, 1u, 1u, 4096u,
        64u, 16u, 16u, 16u, 528u, 1056u,
        0, 0,
        0u, 0u, 0u,
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {
        {0u, 0u, 0u, 0u, 16u, 16u, 0u, 0u, 0u, 1u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}
        }
    };
};
constexpr LinalgEighTilingData EighHostPlan<1u>::value;

template<> struct EighHostPlan<2u> {
    static constexpr LinalgEighTilingData value = {
        64u, 4u, 0u, 32u, 8u, 48640u,
        64u, 8u, 16u, 128u, 640u, 1280u,
        0, 0,
        0u, 0u, 0u,
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {
        {0u, 0u, 0u, 0u, 8u, 8u, 0u, 0u, 0u, 8u},
        {0u, 0u, 1u, 8u, 16u, 8u, 0u, 0u, 0u, 8u},
        {0u, 0u, 2u, 16u, 24u, 8u, 0u, 0u, 0u, 8u},
        {0u, 0u, 3u, 24u, 32u, 8u, 0u, 0u, 0u, 8u},
        {0u, 0u, 4u, 32u, 40u, 8u, 0u, 0u, 0u, 8u},
        {0u, 0u, 5u, 40u, 48u, 8u, 0u, 0u, 0u, 8u},
        {0u, 0u, 6u, 48u, 56u, 8u, 0u, 0u, 0u, 8u},
        {0u, 0u, 7u, 56u, 64u, 8u, 0u, 0u, 0u, 8u},
        {4096u, 64u, 0u, 0u, 8u, 8u, 0u, 0u, 0u, 8u},
        {4096u, 64u, 1u, 8u, 16u, 8u, 0u, 0u, 0u, 8u},
        {4096u, 64u, 2u, 16u, 24u, 8u, 0u, 0u, 0u, 8u},
        {4096u, 64u, 3u, 24u, 32u, 8u, 0u, 0u, 0u, 8u},
        {4096u, 64u, 4u, 32u, 40u, 8u, 0u, 0u, 0u, 8u},
        {4096u, 64u, 5u, 40u, 48u, 8u, 0u, 0u, 0u, 8u},
        {4096u, 64u, 6u, 48u, 56u, 8u, 0u, 0u, 0u, 8u},
        {4096u, 64u, 7u, 56u, 64u, 8u, 0u, 0u, 0u, 8u},
        {8192u, 128u, 0u, 0u, 8u, 8u, 0u, 0u, 0u, 8u},
        {8192u, 128u, 1u, 8u, 16u, 8u, 0u, 0u, 0u, 8u},
        {8192u, 128u, 2u, 16u, 24u, 8u, 0u, 0u, 0u, 8u},
        {8192u, 128u, 3u, 24u, 32u, 8u, 0u, 0u, 0u, 8u},
        {8192u, 128u, 4u, 32u, 40u, 8u, 0u, 0u, 0u, 8u},
        {8192u, 128u, 5u, 40u, 48u, 8u, 0u, 0u, 0u, 8u},
        {8192u, 128u, 6u, 48u, 56u, 8u, 0u, 0u, 0u, 8u},
        {8192u, 128u, 7u, 56u, 64u, 8u, 0u, 0u, 0u, 8u},
        {12288u, 192u, 0u, 0u, 8u, 8u, 0u, 0u, 0u, 8u},
        {12288u, 192u, 1u, 8u, 16u, 8u, 0u, 0u, 0u, 8u},
        {12288u, 192u, 2u, 16u, 24u, 8u, 0u, 0u, 0u, 8u},
        {12288u, 192u, 3u, 24u, 32u, 8u, 0u, 0u, 0u, 8u},
        {12288u, 192u, 4u, 32u, 40u, 8u, 0u, 0u, 0u, 8u},
        {12288u, 192u, 5u, 40u, 48u, 8u, 0u, 0u, 0u, 8u},
        {12288u, 192u, 6u, 48u, 56u, 8u, 0u, 0u, 0u, 8u},
        {12288u, 192u, 7u, 56u, 64u, 8u, 0u, 0u, 0u, 8u}
        }
    };
};
constexpr LinalgEighTilingData EighHostPlan<2u>::value;

template<> struct EighHostPlan<3u> {
    static constexpr LinalgEighTilingData value = {
        256u, 4u, 0u, 32u, 8u, 48640u,
        256u, 32u, 32u, 256u, 768u, 1536u,
        4, 16,
        64u, 4096u, 60416u,
        {0, 64, 128, 192, 256, 0, 0, 0, 0},
        {0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {
        {0u, 0u, 0u, 0u, 32u, 32u, 0u, 0u, 0u, 2u},
        {0u, 0u, 1u, 32u, 64u, 32u, 0u, 0u, 1u, 2u},
        {0u, 0u, 2u, 64u, 96u, 32u, 1u, 2u, 0u, 2u},
        {0u, 0u, 3u, 96u, 128u, 32u, 1u, 2u, 1u, 2u},
        {0u, 0u, 4u, 128u, 160u, 32u, 2u, 4u, 0u, 2u},
        {0u, 0u, 5u, 160u, 192u, 32u, 2u, 4u, 1u, 2u},
        {0u, 0u, 6u, 192u, 224u, 32u, 3u, 6u, 0u, 2u},
        {0u, 0u, 7u, 224u, 256u, 32u, 3u, 6u, 1u, 2u},
        {65536u, 256u, 0u, 0u, 32u, 32u, 0u, 0u, 0u, 2u},
        {65536u, 256u, 1u, 32u, 64u, 32u, 0u, 0u, 1u, 2u},
        {65536u, 256u, 2u, 64u, 96u, 32u, 1u, 2u, 0u, 2u},
        {65536u, 256u, 3u, 96u, 128u, 32u, 1u, 2u, 1u, 2u},
        {65536u, 256u, 4u, 128u, 160u, 32u, 2u, 4u, 0u, 2u},
        {65536u, 256u, 5u, 160u, 192u, 32u, 2u, 4u, 1u, 2u},
        {65536u, 256u, 6u, 192u, 224u, 32u, 3u, 6u, 0u, 2u},
        {65536u, 256u, 7u, 224u, 256u, 32u, 3u, 6u, 1u, 2u},
        {131072u, 512u, 0u, 0u, 32u, 32u, 0u, 0u, 0u, 2u},
        {131072u, 512u, 1u, 32u, 64u, 32u, 0u, 0u, 1u, 2u},
        {131072u, 512u, 2u, 64u, 96u, 32u, 1u, 2u, 0u, 2u},
        {131072u, 512u, 3u, 96u, 128u, 32u, 1u, 2u, 1u, 2u},
        {131072u, 512u, 4u, 128u, 160u, 32u, 2u, 4u, 0u, 2u},
        {131072u, 512u, 5u, 160u, 192u, 32u, 2u, 4u, 1u, 2u},
        {131072u, 512u, 6u, 192u, 224u, 32u, 3u, 6u, 0u, 2u},
        {131072u, 512u, 7u, 224u, 256u, 32u, 3u, 6u, 1u, 2u},
        {196608u, 768u, 0u, 0u, 32u, 32u, 0u, 0u, 0u, 2u},
        {196608u, 768u, 1u, 32u, 64u, 32u, 0u, 0u, 1u, 2u},
        {196608u, 768u, 2u, 64u, 96u, 32u, 1u, 2u, 0u, 2u},
        {196608u, 768u, 3u, 96u, 128u, 32u, 1u, 2u, 1u, 2u},
        {196608u, 768u, 4u, 128u, 160u, 32u, 2u, 4u, 0u, 2u},
        {196608u, 768u, 5u, 160u, 192u, 32u, 2u, 4u, 1u, 2u},
        {196608u, 768u, 6u, 192u, 224u, 32u, 3u, 6u, 0u, 2u},
        {196608u, 768u, 7u, 224u, 256u, 32u, 3u, 6u, 1u, 2u}
        }
    };
};
constexpr LinalgEighTilingData EighHostPlan<3u>::value;

template<> struct EighHostPlan<4u> {
    static constexpr LinalgEighTilingData value = {
        15u, 1u, 0u, 1u, 1u, 4096u,
        64u, 15u, 16u, 16u, 528u, 1056u,
        0, 0,
        0u, 0u, 0u,
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {
        {0u, 0u, 0u, 0u, 15u, 15u, 0u, 0u, 0u, 1u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}
        }
    };
};
constexpr LinalgEighTilingData EighHostPlan<4u>::value;

template<> struct EighHostPlan<5u> {
    static constexpr LinalgEighTilingData value = {
        50u, 3u, 0u, 24u, 8u, 48640u,
        64u, 7u, 16u, 128u, 640u, 1280u,
        0, 0,
        0u, 0u, 0u,
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {
        {0u, 0u, 0u, 0u, 6u, 6u, 0u, 0u, 0u, 8u},
        {0u, 0u, 1u, 6u, 12u, 6u, 0u, 0u, 0u, 8u},
        {0u, 0u, 2u, 12u, 18u, 6u, 0u, 0u, 0u, 8u},
        {0u, 0u, 3u, 18u, 25u, 7u, 0u, 0u, 0u, 8u},
        {0u, 0u, 4u, 25u, 31u, 6u, 0u, 0u, 0u, 8u},
        {0u, 0u, 5u, 31u, 37u, 6u, 0u, 0u, 0u, 8u},
        {0u, 0u, 6u, 37u, 43u, 6u, 0u, 0u, 0u, 8u},
        {0u, 0u, 7u, 43u, 50u, 7u, 0u, 0u, 0u, 8u},
        {2500u, 50u, 0u, 0u, 6u, 6u, 0u, 0u, 0u, 8u},
        {2500u, 50u, 1u, 6u, 12u, 6u, 0u, 0u, 0u, 8u},
        {2500u, 50u, 2u, 12u, 18u, 6u, 0u, 0u, 0u, 8u},
        {2500u, 50u, 3u, 18u, 25u, 7u, 0u, 0u, 0u, 8u},
        {2500u, 50u, 4u, 25u, 31u, 6u, 0u, 0u, 0u, 8u},
        {2500u, 50u, 5u, 31u, 37u, 6u, 0u, 0u, 0u, 8u},
        {2500u, 50u, 6u, 37u, 43u, 6u, 0u, 0u, 0u, 8u},
        {2500u, 50u, 7u, 43u, 50u, 7u, 0u, 0u, 0u, 8u},
        {5000u, 100u, 0u, 0u, 6u, 6u, 0u, 0u, 0u, 8u},
        {5000u, 100u, 1u, 6u, 12u, 6u, 0u, 0u, 0u, 8u},
        {5000u, 100u, 2u, 12u, 18u, 6u, 0u, 0u, 0u, 8u},
        {5000u, 100u, 3u, 18u, 25u, 7u, 0u, 0u, 0u, 8u},
        {5000u, 100u, 4u, 25u, 31u, 6u, 0u, 0u, 0u, 8u},
        {5000u, 100u, 5u, 31u, 37u, 6u, 0u, 0u, 0u, 8u},
        {5000u, 100u, 6u, 37u, 43u, 6u, 0u, 0u, 0u, 8u},
        {5000u, 100u, 7u, 43u, 50u, 7u, 0u, 0u, 0u, 8u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}
        }
    };
};
constexpr LinalgEighTilingData EighHostPlan<5u>::value;

template<> struct EighHostPlan<6u> {
    static constexpr LinalgEighTilingData value = {
        200u, 3u, 0u, 30u, 10u, 48640u,
        256u, 20u, 32u, 320u, 832u, 1664u,
        4, 16,
        56u, 3584u, 34880u,
        {0, 48, 96, 144, 200, 0, 0, 0, 0},
        {0, 8, 24, 32, 48, 56, 72, 80, 96, 104, 120, 128, 144, 160, 176, 184, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {
        {0u, 0u, 0u, 0u, 20u, 20u, 0u, 0u, 0u, 2u},
        {0u, 0u, 1u, 20u, 40u, 20u, 0u, 0u, 1u, 2u},
        {0u, 0u, 2u, 40u, 60u, 20u, 1u, 2u, 0u, 3u},
        {0u, 0u, 3u, 60u, 80u, 20u, 1u, 2u, 1u, 3u},
        {0u, 0u, 4u, 80u, 100u, 20u, 1u, 2u, 2u, 3u},
        {0u, 0u, 5u, 100u, 120u, 20u, 2u, 5u, 0u, 2u},
        {0u, 0u, 6u, 120u, 140u, 20u, 2u, 5u, 1u, 2u},
        {0u, 0u, 7u, 140u, 160u, 20u, 3u, 7u, 0u, 3u},
        {0u, 0u, 8u, 160u, 180u, 20u, 3u, 7u, 1u, 3u},
        {0u, 0u, 9u, 180u, 200u, 20u, 3u, 7u, 2u, 3u},
        {40000u, 200u, 0u, 0u, 20u, 20u, 0u, 0u, 0u, 2u},
        {40000u, 200u, 1u, 20u, 40u, 20u, 0u, 0u, 1u, 2u},
        {40000u, 200u, 2u, 40u, 60u, 20u, 1u, 2u, 0u, 3u},
        {40000u, 200u, 3u, 60u, 80u, 20u, 1u, 2u, 1u, 3u},
        {40000u, 200u, 4u, 80u, 100u, 20u, 1u, 2u, 2u, 3u},
        {40000u, 200u, 5u, 100u, 120u, 20u, 2u, 5u, 0u, 2u},
        {40000u, 200u, 6u, 120u, 140u, 20u, 2u, 5u, 1u, 2u},
        {40000u, 200u, 7u, 140u, 160u, 20u, 3u, 7u, 0u, 3u},
        {40000u, 200u, 8u, 160u, 180u, 20u, 3u, 7u, 1u, 3u},
        {40000u, 200u, 9u, 180u, 200u, 20u, 3u, 7u, 2u, 3u},
        {80000u, 400u, 0u, 0u, 20u, 20u, 0u, 0u, 0u, 2u},
        {80000u, 400u, 1u, 20u, 40u, 20u, 0u, 0u, 1u, 2u},
        {80000u, 400u, 2u, 40u, 60u, 20u, 1u, 2u, 0u, 3u},
        {80000u, 400u, 3u, 60u, 80u, 20u, 1u, 2u, 1u, 3u},
        {80000u, 400u, 4u, 80u, 100u, 20u, 1u, 2u, 2u, 3u},
        {80000u, 400u, 5u, 100u, 120u, 20u, 2u, 5u, 0u, 2u},
        {80000u, 400u, 6u, 120u, 140u, 20u, 2u, 5u, 1u, 2u},
        {80000u, 400u, 7u, 140u, 160u, 20u, 3u, 7u, 0u, 3u},
        {80000u, 400u, 8u, 160u, 180u, 20u, 3u, 7u, 1u, 3u},
        {80000u, 400u, 9u, 180u, 200u, 20u, 3u, 7u, 2u, 3u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}
        }
    };
};
constexpr LinalgEighTilingData EighHostPlan<6u>::value;

static_assert(sizeof(LinalgEighTilingData)==1516u, "Tiling ABI changed");
static_assert(EighHostPlan<1u>::value.used_cores==1u, "case 1 cores");
static_assert(EighHostPlan<2u>::value.used_cores==32u, "case 2 cores");
static_assert(EighHostPlan<3u>::value.used_cores==32u, "case 3 cores");
static_assert(EighHostPlan<4u>::value.used_cores==1u, "case 4 cores");
static_assert(EighHostPlan<5u>::value.used_cores==24u, "case 5 cores");
static_assert(EighHostPlan<6u>::value.used_cores==30u, "case 6 cores");
