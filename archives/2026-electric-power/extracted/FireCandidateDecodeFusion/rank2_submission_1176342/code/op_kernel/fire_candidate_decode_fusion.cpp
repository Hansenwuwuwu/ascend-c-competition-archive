#include "kernel_operator.h"
#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && \
    !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM) && \
    (((TILING_KEY_VAR >> 19U) & 3ULL) == 2ULL) && \
    (((TILING_KEY_VAR >> 8U) & 2047ULL) < 3ULL)
#define FIRE_COMPILED_SMALL_SINGLE 1
#else
#define FIRE_COMPILED_SMALL_SINGLE 0
#endif
#include "device_fire.h"
#include "tiling_key_fire_candidate_decode_fusion.h"

#if FIRE_COMPILED_SMALL_SINGLE
template<typename T, uint32_t TILE, uint32_t MODE>
__global__ __aicore__ void fire_candidate_decode_fusion(
    GM_ADDR r, GM_ADDR p, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionHeaderOnlyTilingData);
    static_assert(MODE == 2 && TILE <= 128, "one complete small tile");
    constexpr uint32_t units = TILE / (sizeof(T) == 2 ? 16U : 8U);
    const FireCandidateDecodeFusionTilingData t{1,TILE,1,TILE,units,0,1,TILE};
    if constexpr (sizeof(T) == 4 && TILE == 32) {
        // Only logical owner zero uses this complete one-tile resource.
        // Read-only DMA can run during the original SoC state setup.
        if (static_cast<uint64_t>(get_block_idx()) == 0ULL &&
            static_cast<uint64_t>(GetSubBlockIdx()) == 0ULL) {
            PadSmallDecode<T,TILE>::PrefetchCompleteSingleInputs(r,p);
            InitSocState();
            PadSmallDecode<T,TILE> op;
            op.template Init<false,true>(r,p,o,t);
            op.ProcessSingle();
        }
    } else {
        // Preserve the original metadata prefetch on all launched cores.
        InitSocState();
        PadSmallDecode<T,TILE>::PrefetchIndexMetadata(
            nullptr);
        const uint32_t owner = GetBlockIdx();
        if (owner == 0U) {
            PadSmallDecode<T,TILE> op;
            if constexpr (sizeof(T) == 2 && TILE == 32) {
                op.template InitWithHalfRawReady<true,true>(r,p,o,t,0);
                op.ProcessSingleWithHalfRawReady(0);
            } else if constexpr (TILE == 64) {
                op.template InitWithInputPrefetch<true,true,true>(r,p,o,t,0);
                op.template ProcessSingleAfterInputPrefetch<true,true>(0);
            } else {
                op.template Init<true,true>(r,p,o,t);
                op.template ProcessSingle<true,true>();
            }
        } else {
            PipeBarrier<PIPE_ALL>();
        }
    }
}
#else
template<typename T, uint32_t TILE, bool Dual>
__aicore__ inline void FireLargeUniform(GM_ADDR r, GM_ADDR p, GM_ADDR o,
    FireCandidateDecodeFusionTilingData t, uint32_t control, uint32_t owner) {
    using Decode = BlockedTransposeDecode<T, TILE, false, false, Dual, true>;
    Decode op;
    op.Init(r, p, o, t, nullptr, 0, control & 65535u, owner);
    op.Process();
}

// Keep the output-pointer dependent alternative in its own ordinary function.
// Large UniformTiles consume batch/n/blocks and the separate tile count only.
template<typename T, uint32_t TILE>
__aicore__ __attribute__((noinline)) void FireLargeUniformDualFallback(
    GM_ADDR r, GM_ADDR p, GM_ADDR o, uint32_t batch, uint32_t n,
    uint32_t blocks, uint32_t count, uint32_t owner) {
    FireCandidateDecodeFusionTilingData t{batch,n,blocks,TILE,0,0,0,batch*n};
    FireLargeUniform<T,TILE,true>(r,p,o,t,count,owner);
}

template<typename T, uint32_t TILE, uint32_t MODE>
__global__ __aicore__ void fire_candidate_decode_fusion(
    GM_ADDR r, GM_ADDR p, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionHeaderOnlyTilingData);
    InitSocState();
    constexpr bool smallPrefetch = TILE <= 128 &&
        ((MODE != 0 && (sizeof(T) == 2 || TILE > 32)) ||
         (MODE == 0 && sizeof(T) == 2 && TILE <= 32));
    constexpr bool midPrefetch = TILE > 128 && TILE <= 1024 && MODE != 0;
    if constexpr (TILE > 1024) {
        if constexpr (sizeof(T) == 2)
            BlockedTransposeDecode<T,TILE>::PrefetchAddressMetadata(
                nullptr);
        else
            BlockedTransposeDecode<T,TILE>::PrefetchAddressMetadata(
                nullptr);
    } else if constexpr (smallPrefetch) {
        PadSmallDecode<T,TILE>::PrefetchIndexMetadata(
            nullptr);
    } else if constexpr (midPrefetch) {
        BlockedTransposeDecode<T,TILE,true,(MODE == 2)>::PrefetchAddressMetadata(
            nullptr);
    }
    uint32_t control;
    FireCandidateDecodeFusionTilingData t;
    if constexpr (TILE > 1024) {
        const uint64_t word = *reinterpret_cast<const __gm__ uint64_t *>(tiling);
        const uint32_t packed = static_cast<uint32_t>(word >> 32);
        const uint32_t n = static_cast<uint32_t>(word);
        const uint32_t batch = ((packed >> 16) & 7u) + 1u;
        const uint32_t blocks = (packed >> 22) & 63u;
        const uint32_t totalRows = batch * n;
        constexpr uint32_t rowAlign = sizeof(T) == 2 ? 16u : 8u;
        const uint32_t extra = ((totalRows + rowAlign - 1u) / rowAlign) % blocks;
        control = (packed & ~(63u << 16)) | (extra << 16);
        t = {batch, n, blocks, TILE, control & 65535u, extra,
             (control >> 28) & 1u, totalRows};
    } else {
        const uint64_t word = *reinterpret_cast<const __gm__ uint64_t *>(tiling);
        const uint32_t packed = static_cast<uint32_t>(word >> 32);
        const uint32_t n = static_cast<uint32_t>(word);
        const uint32_t batch = ((packed >> 25) & 7u) + 1u;
        const uint32_t units = packed & 8191u;
        const uint32_t extra = (packed >> 13) & 63u;
        const uint32_t blocks = (packed >> 19) & 63u;
        control = units | (extra << 16) | (blocks << 22) | (packed & 0xf0000000u);
        t = {batch, n, blocks, TILE, units, extra,
             (control >> 28) & 1u, batch * n};
    }
    const uint32_t owner = GetBlockIdx();
    if (owner >= t.blocks) {
        if constexpr (smallPrefetch || midPrefetch) PipeBarrier<PIPE_ALL>();
        return;
    }
    if constexpr (TILE <= 128) {
        if constexpr (MODE == 0 && TILE <= 64) {
            if constexpr (smallPrefetch) PipeBarrier<PIPE_ALL>();
            RunSmallMany<T,TILE>(r,p,o,t.batch,t.n,t.blocks,t.tile);
        } else if constexpr (TILE == 128) {
            if (__builtin_expect(t.denseOne != 0, true)) {
                PadSmallDecode<T,TILE> op;
                op.template Init<true,true>(r,p,o,t);
                op.template ProcessSingle<true,true>();
            } else {
                RunSmallManyIndexed<T,TILE,true>(r,p,o,t.batch,t.n,t.blocks,t.tile);
            }
        } else {
            PadSmallDecode<T,TILE> op;
            if constexpr (sizeof(T) == 2 && TILE == 32) {
                op.template InitWithHalfRawReady<true,true>(r,p,o,t,owner);
                op.ProcessSingleWithHalfRawReady(owner);
            } else if constexpr (TILE == 64) {
                op.template InitWithInputPrefetch<true,true,true>(r,p,o,t,owner);
                op.template ProcessSingleAfterInputPrefetch<true,true>(owner);
            } else {
                op.template Init<(sizeof(T) == 2 || TILE > 32),true>(r,p,o,t);
                op.ProcessSingle();
            }
        }
    } else if constexpr (TILE <= 1024) {
        if constexpr (MODE == 0) {
            RunDenseDecode<T,TILE,true>(r,p,o,t);
        } else if constexpr (MODE == 2) {
            BlockedTransposeDecode<T,TILE,true,true> op;
            op.template Init<true>(r,p,o,t,nullptr,control >> 30);
            op.ProcessFullGuardCount();
        } else {
            BlockedTransposeDecode<T,TILE,true,false> op;
            op.Init(r,p,o,t,nullptr,control >> 30);
            op.ProcessPartialGuardCount();
        }
    } else if constexpr (sizeof(T) == 4) {
        const uint32_t totalTiles = ((t.n + TILE - 1) / TILE) * t.batch;
        const uint32_t outputBatchBytes = t.n * 5 * sizeof(T);
        const bool dual = t.batch > 1 && totalTiles >= 2 * t.blocks &&
            ((reinterpret_cast<uint64_t>(o) | outputBatchBytes) & 31u) != 0;
        if constexpr (TILE == 1640 && (MODE == 1 || MODE == 2)) {
            t.unitsPerWorker = 0;
            if constexpr (MODE == 2) {
                FireLargeUniform<T,TILE,true>(r,p,o,t,control,owner);
            } else {
                if (__builtin_expect(dual, false))
                    FireLargeUniformDualFallback<T,TILE>(r,p,o,t.batch,t.n,
                        t.blocks,control & 65535u,owner);
                else FireLargeUniform<T,TILE,false>(r,p,o,t,control,owner);
            }
        } else if (dual) {
            BlockedTransposeDecode<T,TILE,false,false,true> op;
            op.Init(r,p,o,t);
            op.Process();
        } else {
            RunDenseDecode<T,TILE,true>(r,p,o,t);
        }
    } else {
        RunDenseDecode<T,TILE,true>(r,p,o,t);
    }
}
#endif

#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && \
    !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
namespace AscendC {
template<uint64_t Key> struct FireSmallWorkspacePolicy {
    static constexpr uint32_t resource = static_cast<uint32_t>((Key >> 8U) & 2047ULL);
    static constexpr uint32_t mode = static_cast<uint32_t>((Key >> 19U) & 3ULL);
    static constexpr bool unused = (mode == 1U && resource < 2U) ||
                                   (mode == 2U && resource < 3U);
};
template<uint64_t Key>
__aicore__ inline void FireSmallCompilerWorkspace(GM_ADDR workspace) {
    if constexpr (!FireSmallWorkspacePolicy<Key>::unused)
        (SetSysWorkspaceForce)(workspace);
}
template<uint64_t Key>
__aicore__ inline GM_ADDR FireSmallCompilerUserWorkspace(GM_ADDR workspace) {
    if constexpr (FireSmallWorkspacePolicy<Key>::unused) return nullptr;
    else return (GetUserWorkspace)(workspace);
}
}
#define SetSysWorkspaceForce(...) FireSmallCompilerWorkspace<static_cast<uint64_t>(TILING_KEY_VAR)>(__VA_ARGS__)
#define GetUserWorkspace(...) FireSmallCompilerUserWorkspace<static_cast<uint64_t>(TILING_KEY_VAR)>(__VA_ARGS__)
#endif

// Match entry placement to the instruction-cache line size. This declaration
// applies to the SDK-generated exported entry for every tiling resource.
#if defined(TILING_KEY_VAR) && defined(auto_gen_fire_candidate_decode_fusion_kernel) && \
    __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG
extern "C" __attribute__((cce_kernel, aiv, aligned(128)))
void auto_gen_fire_candidate_decode_fusion_kernel(
    GM_ADDR r, GM_ADDR p, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling);
#endif
