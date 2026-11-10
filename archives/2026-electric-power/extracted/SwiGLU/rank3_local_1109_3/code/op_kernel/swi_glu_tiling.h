#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 11265UL)
#pragma once
#include <cstdint>
#include <cstddef>
#include "kernel_tiling/kernel_tiling.h"
struct SwiGluTilingData {
uint32_t path;
uint32_t reserved0;
uint64_t m;
uint64_t d;
uint32_t dAlign;
uint32_t tileN;
uint32_t usedCoreNum;
uint32_t taskM;
uint32_t taskNum;
uint32_t taskN;
AscendC::tiling::TCubeTiling cubeTilingData;
alignas(32) uint32_t gatherOffsetsB[2000];
};
struct SwiGluDirectTilingData {
uint32_t path, reserved0;
uint64_t m, d;
uint32_t dAlign, tileN, usedCoreNum, taskM, taskNum, taskN;
};
static_assert(sizeof(SwiGluDirectTilingData) == offsetof(SwiGluTilingData, cubeTilingData),
"Direct tiling must match prefix");
struct SwiGluTinyTilingData { uint32_t path, reserved0; uint64_t m, d; };
static_assert(sizeof(SwiGluTinyTilingData) == 24, "Tiny tiling prefix size");
#elif defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12033UL)
#pragma once
#include <cstdint>
#include <cstddef>
#include "kernel_tiling/kernel_tiling.h"
struct SwiGluTilingData {
uint32_t path;
uint32_t reserved0;
uint64_t m;
uint64_t d;
uint32_t dAlign;
uint32_t tileN;
uint32_t usedCoreNum;
uint32_t taskM;
uint32_t taskNum;
uint32_t taskN;
AscendC::tiling::TCubeTiling cubeTilingData;
alignas(32) uint32_t gatherOffsetsA[1024];
alignas(32) uint32_t gatherOffsetsB[2000];
};
struct SwiGluDirectTilingData {
uint32_t path, reserved0;
uint64_t m, d;
uint32_t dAlign, tileN, usedCoreNum, taskM, taskNum, taskN;
};
static_assert(sizeof(SwiGluDirectTilingData) == offsetof(SwiGluTilingData, cubeTilingData),
"Direct tiling must match prefix");
struct SwiGluTinyTilingData { uint32_t path, reserved0; uint64_t m, d; };
static_assert(sizeof(SwiGluTinyTilingData) == 24, "Tiny tiling prefix size");
#elif defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)
#pragma once
#include <cstdint>
#include <cstddef>
#include "kernel_tiling/kernel_tiling.h"
struct SwiGluTilingData {
uint32_t path;
uint32_t reserved0;
uint64_t m;
uint64_t d;
uint32_t dAlign;
uint32_t tileN;
uint32_t usedCoreNum;
uint32_t taskM;
uint32_t taskNum;
uint32_t taskN;
AscendC::tiling::TCubeTiling cubeTilingData;
};
struct SwiGluDirectTilingData {
uint32_t path, reserved0;
uint64_t m, d;
uint32_t dAlign, tileN, usedCoreNum, taskM, taskNum, taskN;
};
static_assert(sizeof(SwiGluDirectTilingData) == offsetof(SwiGluTilingData, cubeTilingData),
"Direct tiling must match prefix");
struct SwiGluTinyTilingData { uint32_t path, reserved0; uint64_t m, d; };
static_assert(sizeof(SwiGluTinyTilingData) == 24, "Tiny tiling prefix size");
#else
#pragma once
#include <cstdint>
#include <cstddef>
#include "kernel_tiling/kernel_tiling.h"
struct SwiGluTilingData {
uint32_t path;
uint32_t reserved0;
uint64_t m;
uint64_t d;
uint32_t dAlign;
uint32_t tileN;
uint32_t usedCoreNum;
uint32_t taskM;
uint32_t taskNum;
uint32_t taskN;
AscendC::tiling::TCubeTiling cubeTilingData;
};
struct SwiGluDirectTilingData {
uint32_t path, reserved0;
uint64_t m, d;
uint32_t dAlign, tileN, usedCoreNum, taskM, taskNum, taskN;
};
static_assert(sizeof(SwiGluDirectTilingData) == offsetof(SwiGluTilingData, cubeTilingData),
"Direct tiling must match prefix");
struct SwiGluTinyTilingData { uint32_t path, reserved0; uint64_t m, d; };
static_assert(sizeof(SwiGluTinyTilingData) == 24, "Tiny tiling prefix size");
#endif
