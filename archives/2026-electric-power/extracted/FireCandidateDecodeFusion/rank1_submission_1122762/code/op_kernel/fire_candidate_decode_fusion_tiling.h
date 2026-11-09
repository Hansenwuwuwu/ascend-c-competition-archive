#pragma once
#include <cstdint>
#include <cstddef>
struct FireCandidateDecodeFusionTilingData { uint32_t total, n, perCore; };
struct FireSmallCompositeTilingData {
    FireCandidateDecodeFusionTilingData header;
    uint32_t reserved[5];
    uint32_t full[512], tail[512];
};
struct FireFlatCompositeTilingData {
    FireCandidateDecodeFusionTilingData header;
    uint32_t reserved[5];
    uint32_t table0[512], table1[512], tail[128];
};
static_assert(sizeof(FireCandidateDecodeFusionTilingData) == 12, "base ABI");
static_assert(sizeof(FireSmallCompositeTilingData) == 4128, "row-table ABI");
static_assert(sizeof(FireFlatCompositeTilingData) == 4640, "flat-table ABI");
static_assert(offsetof(FireFlatCompositeTilingData, table0) == 32, "table alignment");
static_assert(offsetof(FireFlatCompositeTilingData, table1) == 2080, "table alignment");
static_assert(offsetof(FireFlatCompositeTilingData, tail) == 4128, "table alignment");

struct FireFrameTilingData { FireCandidateDecodeFusionTilingData header; uint32_t geometry; };
static_assert(sizeof(FireFrameTilingData)==16, "frame schedule ABI");

struct FireSingleBlockTilingData {
    FireCandidateDecodeFusionTilingData header;
    uint32_t reserved[5], table[512];
};
static_assert(sizeof(FireSingleBlockTilingData)==2080, "single-block metadata");
static_assert(offsetof(FireSingleBlockTilingData,table)==32, "same first flat table");

struct FireReplicatedFrameSlot {
    FireFrameTilingData schedule;
    uint32_t padding[20];
};
struct FireReplicatedFrameTilingData {
    FireFrameTilingData prefix;
    uint32_t reserved[4];
    FireReplicatedFrameSlot slots[48];
};
static_assert(sizeof(FireReplicatedFrameSlot) == 96, "replicated frame slot");
static_assert(offsetof(FireReplicatedFrameSlot, schedule) == 0, "slot schedule prefix");
static_assert(offsetof(FireReplicatedFrameTilingData, slots) == 32, "replicated frame slot offset");
static_assert(sizeof(FireReplicatedFrameTilingData) == 4640, "existing metadata envelope");
