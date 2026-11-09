#pragma once
#include <cstdint>
struct LinalgEighTilingData {
    uint32_t n;
    uint32_t batch;
    uint32_t stride;
    uint32_t upper;
    uint32_t team;
    uint32_t groups;
    uint32_t rows;
    uint32_t vectorAlgorithm;
    uint32_t vectorFlags;
};

constexpr uint32_t EIGH_MEDIUM_OUTPUT_SCRATCH = 1;
constexpr uint32_t EIGH_MEDIUM_TWO_INVERSE_ITERATIONS = 2;
// Each owner publishes a complete 64-byte DMA record; writers never share
// a partial GM transfer block. Only the first float carries its certificate.
constexpr uint32_t EIGH_DISTRIBUTED_OUTPUT_HISTORY = 4;
constexpr uint32_t EIGH_SMALL_TWISTED_FACTORIZATION = 32;
constexpr uint32_t EIGH_MEDIUM_CERTIFIED_FLOAT_RAYLEIGH = 256;
constexpr uint32_t EIGH_CERTIFICATE_SLOT_FLOAT_COUNT = 16;

// Algorithm choice is runtime tiling data, separate from required core types.
enum class EighVectorAlgorithm : uint32_t {
    Distributed = 0,
    ResidentMedium = 1,
    ResidentSmall = 2,
};

// Direct Cube Matmul owns its local buffers; reserve only the launch protocol
// prefix, rounded to 4 KiB pages. Its bound is checked against the device SDK.
constexpr uint64_t LINALG_EIGH_CUBE_PROTOCOL_BYTES = 832 * 1024;

// Independent ownership needs no shared Gram snapshot or rendezvous.
constexpr uint32_t EIGH_MEDIUM_PRIVATE_SOLVE = 512;

// A matrix leader publishes current-call Householder coefficients and history.
constexpr uint32_t EIGH_MEDIUM_SHARED_REDUCTION = 2048;

// A three-tile private tail uses96 physical columns and32-float row alignment.
constexpr uint32_t EIGH_DISTRIBUTED_TAIL_96 = 1024;

// Current-input cluster ownership chooses the established full certificate.
constexpr uint32_t EIGH_MEDIUM_BOUNDARY_CONSENSUS = 4096;

// Full two-float original-input Rayleigh quotients use vector reductions.
constexpr uint32_t EIGH_MEDIUM_VECTOR_ORIGINAL_QUOTIENT = 8192;

// Resident reflector vector pipeline for the final8-column SIMD bucket.
constexpr uint32_t EIGH_MEDIUM_UB_REFLECTOR_PIPELINE = 16384;

// Two16-row reflector tiles fit the smaller one-repeat row arenas.
constexpr uint32_t EIGH_MEDIUM_NARROW_ROW_PIPELINE = 32768;
