#include "pinv_contract.h"
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & (PINV_KEY_GROUP | PINV_KEY_COMPACT | PINV_KEY_LOCAL))
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;
#include "pinv_cube_residual.h"
#include "pinv_rect.h"
#include "pinv_pipeline.h"

#include "pinv_window_pipeline.h"
#include "tiling_key_linalg_pinv.h"
#include "pinv_stable.h"
#include "pinv_local.h"
#include "pinv_compact.h"
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;
template <uint32_t PAD>
class CompactQuickLight {
    TBuf<TPosition::VECCALC> mem;
    LocalTensor<float> matrix, lowMatrix, u, prod, tmp, red, gather, norms;
    GlobalTensor<float> input,result;
    float atol,rtol;
    __aicore__ inline void VS(){SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);}
    __aicore__ inline void SV(){SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);}
    __aicore__ inline float Root(float value){red.SetValue(0,value);SV();Sqrt(red,red,1);VS();return red.GetValue(0);}
public:
    __aicore__ inline void Init(GM_ADDR in, GM_ADDR dst, const LinalgPinvTilingData &data, TPipe &pipe){
        constexpr uint32_t pitch=2*PAD+8,count=PAD*pitch;
        pipe.InitBuffer(mem,(2*count+pitch+3*PAD+64)*sizeof(float));
        auto all=mem.Get<float>();uint32_t off=0;
        matrix=all;off+=count;lowMatrix=all[off];off+=count;u=all[off];off+=pitch;
        prod=all[off];off+=PAD;tmp=all[off];off+=32;red=all[off];off+=32;
        gather=all[off];off+=PAD;norms=all[off];
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(in));
        result.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
        atol=data.atol;rtol=data.rtol;
    }
    __aicore__ inline bool RunAligned(uint32_t offset) {
        constexpr uint32_t n = PAD, width = 2 * PAD, pitch = 2 * PAD + 8;
        constexpr uint32_t count = n * pitch;
        Duplicate(matrix, 0.0f, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(matrix, input[offset], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, 0, (pitch - n) / 8, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        Abs(lowMatrix, matrix, count); PipeBarrier<PIPE_V>();
        ReduceMax(red, lowMatrix, tmp, count, false); VS();
        float scale = red.GetValue(0);
        if (!(scale >= 0.1f && scale < pinv::kScaleLimit)) return false;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix, matrix, count); PipeBarrier<PIPE_V>();
        ReduceSum(red, lowMatrix, tmp, count); VS();
        float total = red.GetValue(0);
        auto values = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        for (uint32_t i = 0; i < n; ++i) values[i * pitch + n + i] = 1.0f;
        auto indices = gather.ReinterpretCast<uint32_t>();
        uint32_t order[PAD];
        for (uint32_t i = 0; i < n; ++i) indices.SetValue(i, i * pitch * sizeof(float));
        uint64_t available = (uint64_t(1) << n) - 1;
        SV();
        for (uint32_t column = 0; column < n; ++column) {
            Gather(norms, matrix, indices, column * sizeof(float), n); PipeBarrier<PIPE_V>();
            Abs(prod, norms, n); PipeBarrier<PIPE_V>();
            uint64_t mask[2] = {available, 0};
            WholeReduceMax(red, prod, mask, 1, 1, 1, 8); VS();
            float best, pivotBits;
            GetReduceRepeatMaxMinSpr<float>(best, pivotBits);
            uint32_t pivot = *reinterpret_cast<uint32_t *>(&pivotBits);
            if (!(best > 0.0f)) return false;
            available &= ~(uint64_t(1) << pivot);
            order[pivot] = column;
            Muls(matrix[pivot * pitch], matrix[pivot * pitch], -1.0f / values[pivot * pitch + column], width); PipeBarrier<PIPE_V>();
            norms.SetValue(pivot, 0.0f); SV();
            Brcb(lowMatrix, norms, n / 8, {static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch)}); PipeBarrier<PIPE_V>();
            MulAddDst(matrix, matrix[pivot * pitch], lowMatrix, static_cast<uint64_t>(width), static_cast<uint8_t>(n), {1, 1, 0, static_cast<uint8_t>(pitch / 8), 0, static_cast<uint8_t>(pitch / 8)});
            PipeBarrier<PIPE_V>();
        }
        Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix[n], matrix[n], static_cast<uint64_t>(n), static_cast<uint8_t>(n), {1, 1, 1, static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch / 8)});
        PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS();
        float inverseTotal = red.GetValue(0), cutoff = Root(total) * rtol;
        float absolute = atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < pinv::kDirectGateSquared)) return false;
        Muls(matrix, matrix, -1.0f / power, count); PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < n; ++row) {
            uint32_t target = order[row];
            DataCopy(lowMatrix[target * pitch], matrix[row * pitch], pitch);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopyPad(result[offset], lowMatrix[n], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, (pitch - n) / 8, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
        return true;
    }
    __aicore__ inline bool Run(uint32_t offset){
        return RunAligned(offset);
    }
};
template <uint32_t PAD>
__aicore__ inline bool TryCompactQuickLight(GM_ADDR in,GM_ADDR dst,const LinalgPinvTilingData &data){
    TPipe pipe;CompactQuickLight<PAD> op;op.Init(in,dst,data,pipe);
    bool valid=op.Run(GetBlockIdx()*data.m*data.n);
    PipeBarrier<PIPE_ALL>();pipe.Destroy();return valid;
}

template <typename DT_A, bool COMPACT, bool TAIL, bool WAVE, bool LOCAL, bool GROUP, bool PIPELINE, bool WINDOW, bool CUBE, bool RECT>
__aicore__ __attribute__((noinline)) void PvStageNative(GM_ADDR A, GM_ADDR A_pinv, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgPinvCubeTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData, data, tiling);
    if constexpr(RECT){
        TPipe pipe;RectPinv op;op.Init(A,A_pinv,GetUserWorkspace(workspace),data,pipe);
        op.Process(GetUserWorkspace(workspace));return;
    }
    if constexpr(CUBE){
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & PINV_KEY_CUBE)
#if (TILING_KEY_VAR & PINV_KEY_WINDOW)
        GET_TILING_DATA_WITH_STRUCT(LinalgPinvCubeTilingData,cubeData,tiling);
        if ASCEND_IS_AIC { PinvCubeService(GetUserWorkspace(workspace),A_pinv,cubeData);return; }
#else
        if ASCEND_IS_AIC {return;}
#endif
#endif
    }
    PinvCubeGuard<CUBE && WINDOW> cubeGuard((CUBE && WINDOW) ? GetUserWorkspace(workspace) : nullptr,data);
    uint32_t windowState=0;
    if constexpr (PIPELINE) {
        if constexpr (WINDOW) {
            windowState=TryWindowPipelinePinv(A,A_pinv,GetUserWorkspace(workspace),data);
            if((windowState&15)==(uint32_t(1)<<data.batch)-1)return;
            data.local=0;
        }else{
#if defined(TILING_KEY_VAR) && TILING_KEY_VAR==(PINV_KEY_GROUP | PINV_KEY_PIPELINE | PINV_KEY_CUBE)
            if(TryPeerFullPinv(A,A_pinv,GetUserWorkspace(workspace),data))return;
#else
            if(TryPipelinePinv(A,A_pinv,GetUserWorkspace(workspace),data))return;
#endif
        }
    }
    if constexpr (COMPACT || LOCAL) if (GetBlockIdx() >= data.batch) return;
    if constexpr (COMPACT && !TAIL) {
        if(data.rows>8 && data.rows<=16 && data.cols%8==0)
            if(TryCompactQuickLight<16>(A,A_pinv,data))return;
    }
    TPipe pipe;
    if constexpr (COMPACT) {
        data.parallel = 0; data.local = 1; data.localInverse = 1;
        if (data.rows > 8 && data.rows <= 16) {
            Pinv<TAIL, 16> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
        } else {
            Pinv<TAIL> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
        }

    } else {
        if constexpr (WAVE) { data.parallel = 1; data.local = 0; data.localInverse = 0; }
        if constexpr (LOCAL) { data.parallel = 0; data.groupCount = 0; data.localInverse = 1; }
        if constexpr (LOCAL) {
            LocalPinv<WAVE, LOCAL> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
        } else {
            StablePinv<WAVE, LOCAL, GROUP> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe);
            if constexpr (WINDOW) op.ProcessPrepared(windowState);
            else op.Process();
        }
    }
}

#include "pinv_dsl_compact.h"
#include "pinv_dsl_local.h"
#include "pinv_dsl_half_pipeline.h"
#include "pinv_dsl_peer_tiles.h"
#include "pinv_dsl_window.h"
template <typename DT_A, bool COMPACT, bool TAIL, bool WAVE, bool LOCAL, bool GROUP, bool PIPELINE, bool WINDOW, bool CUBE, bool RECT>
__global__ __aicore__ void linalg_pinv(GM_ADDR A, GM_ADDR A_pinv, GM_ADDR workspace, GM_ADDR tiling) {
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & PINV_KEY_WINDOW) && (TILING_KEY_VAR & PINV_KEY_CUBE)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
#endif
    REGISTER_TILING_DEFAULT(LinalgPinvCubeTilingData);
    if constexpr (COMPACT) {
        // Keep the DSL inactive job guard ahead of the CANN tiling copy.
        auto packet = reinterpret_cast<__gm__ const uint32_t *>(tiling);
        if (GetBlockIdx() >= packet[offsetof(LinalgPinvTilingData, batch) / sizeof(uint32_t)]) return;
    }
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData, data, tiling);
    if constexpr (GROUP && PIPELINE && CUBE && !WINDOW && WAVE) {
        if ASCEND_IS_AIC {return;}
        GM_ADDR work=GetUserWorkspace(workspace);
        GM_ADDR status=work+pinv_dsl::WorkFloats(data.batch,data.rows,data.cols,data.stride)*sizeof(float);
        PvStagePeerTile(A,A_pinv,work,status,GetBlockNum(),data.cols,data.batch,data.stride,0,data.atol,data.rtol,0);
        return;
    } else {
    if constexpr (COMPACT || LOCAL) {
        if (COMPACT || data.cols<=64) {
            // Complete resident paths keep matrices and admission status in
            // TileLang UB buffers, including rank/truncation continuations.
            GM_ADDR work=nullptr;
            GM_ADDR status=nullptr;
            if constexpr (COMPACT) {
                if constexpr (TAIL) PvStageCompact(A,A_pinv,work,status,GetBlockNum(),data.m,data.n,data.batch,data.atol,data.rtol,0);
                else PvStageCompactAligned(A,A_pinv,work,status,GetBlockNum(),data.m,data.n,data.batch,data.atol,data.rtol,0);
            }
            else PvStageLocal(A,A_pinv,work,status,GetBlockNum(),data.m,data.n,data.batch,data.atol,data.rtol,0);
            return;
        }
    }
    if constexpr (GROUP && PIPELINE && WINDOW && CUBE && !WAVE) {
        if(pinv_dsl::WindowSupported(data.m,data.n,data.batch,data.stride,data.local,GetBlockNum())) {
            GM_ADDR work=GetUserWorkspace(workspace);
            GM_ADDR status=work+pinv_dsl::WindowWorkFloats(data.batch,data.cols,data.stride,GetBlockNum())*sizeof(float);
            int32_t windowAdmitted=PvStageWindow(A,A_pinv,work,status,GetBlockNum(),data.cols,data.batch,data.stride,data.local,data.atol,data.rtol,0,0);
            if(windowAdmitted!=0)return;
            AscendC::GlobalTensor<float> flags;
            flags.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(status));
            bool admitted=true;
            for(uint32_t b=0;b<data.batch;++b) {
                AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(flags[b*512+16]);
                if(!(flags.GetValue(b*512+16)>0.5f))admitted=false;
            }
            if(admitted)return;
            {
                AscendC::TPipe continuation_barrier;
                AscendC::SyncAll<false>();
                continuation_barrier.Destroy();
            }
        }
    }
    if constexpr (GROUP && PIPELINE && CUBE && !WINDOW && !WAVE) {
        if ASCEND_IS_AIC {return;}
        GM_ADDR work=GetUserWorkspace(workspace);
        GM_ADDR status=work+pinv_dsl::WorkFloats(data.batch,data.rows,data.cols,data.stride)*sizeof(float);
        // CANN initializes the hardware sync base; 0 skips the JIT-only setter.
        if constexpr (TAIL) {
            PvStageHalfPipeline(A,A_pinv,work,status,GetBlockNum(),data.cols,data.batch,data.stride,0,data.atol,data.rtol,0);
        } else {
            PvStagePeerPipeline(A,A_pinv,work,status,GetBlockNum(),data.cols,data.batch,data.stride,0,data.atol,data.rtol,0);
        }
        return;
    }
    PvStageNative<DT_A,COMPACT,TAIL,WAVE,LOCAL,GROUP,PIPELINE,WINDOW,CUBE,RECT>(A,A_pinv,workspace,tiling);
    }
}

#if defined(__CCE_AICORE__)
#define cce_kernel cce_kernel, aligned(16384)
#endif
