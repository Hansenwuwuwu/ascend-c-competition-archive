#pragma once
#include <cstddef>
#include <cstdint>

constexpr uint32_t SBG_MAX_LOCAL_ROWS = 512U;
struct alignas(8) SafetyBeltViolationGeometryTilingData {
    struct Slot {
        uint32_t pack;
        float waistRatio;
    };
    static constexpr uint32_t SLOT_BYTES = 8U;
    static constexpr uint32_t SLOT_COUNT = 1U;
    Slot slots[SLOT_COUNT];
};
static_assert(sizeof(SafetyBeltViolationGeometryTilingData) == 8U, "tiling ABI must be eight bytes");
static_assert(offsetof(SafetyBeltViolationGeometryTilingData::Slot, waistRatio) == 4U, "ratio ABI");
