#pragma once
#include <cstdint>
#include <cstddef>
struct FireCandidateDecodeFusionTilingData { uint32_t batch, n, blocks, tile;
    uint32_t unitsPerWorker, extraUnits, denseOne, totalRows; };

// Full arithmetic state is expanded in registers after reading this packet.
struct FireCandidateDecodeFusionPackedTilingData {
    uint32_t n, totalRows, control, batch;
};
static_assert(sizeof(FireCandidateDecodeFusionPackedTilingData)==16,"packed tiling size");

// Legacy packet layouts retained for the reference Host table generator.
// Production launch metadata is read from fixed device read-only tables.
struct FireCandidateDecodeFusionAddressTilingData {
    FireCandidateDecodeFusionPackedTilingData header{};
    uint32_t reserved[124]{};
    uint64_t addressTable[128]{};
};
static_assert(offsetof(FireCandidateDecodeFusionAddressTilingData,addressTable)==512,"address table alignment");
static_assert(sizeof(FireCandidateDecodeFusionAddressTilingData)==1536,"address tiling size");

struct FireCandidateDecodeFusionLargeLowAddressTilingData {
    FireCandidateDecodeFusionPackedTilingData header{};
    uint32_t reserved[124]{};
    uint64_t addressTable[336]{};
};
static_assert(offsetof(FireCandidateDecodeFusionLargeLowAddressTilingData,addressTable)==512,"large low VA alignment");
static_assert(sizeof(FireCandidateDecodeFusionLargeLowAddressTilingData)==3200,"large low VA size");

struct FireCandidateDecodeFusionLargeHalfLowAddressTilingData {
    FireCandidateDecodeFusionPackedTilingData header{};
    uint32_t reserved[124]{};
    uint64_t addressTable[288]{};
};
static_assert(offsetof(FireCandidateDecodeFusionLargeHalfLowAddressTilingData,addressTable)==512,"large Half low VA alignment");
static_assert(sizeof(FireCandidateDecodeFusionLargeHalfLowAddressTilingData)==2816,"large Half low VA size");

struct FireCandidateDecodeFusionMidLowAddressTilingData {
    FireCandidateDecodeFusionPackedTilingData header{};
    uint32_t reserved[124]{};
    uint64_t addressTable[256]{};
};
static_assert(offsetof(FireCandidateDecodeFusionMidLowAddressTilingData,addressTable)==512,"mid VA alignment");
static_assert(sizeof(FireCandidateDecodeFusionMidLowAddressTilingData)==2560,"mid VA packet size");

// The immutable address tables are produced in UB; the launch packet now only
// carries the ordinary dynamic tiling fields and fixed-size padding.
struct FireCandidateDecodeFusionHeaderOnlyTilingData {
    FireCandidateDecodeFusionPackedTilingData header{};
    uint64_t reserved[2]{};
};
static_assert(sizeof(FireCandidateDecodeFusionHeaderOnlyTilingData)==32,"compact launch packet");
