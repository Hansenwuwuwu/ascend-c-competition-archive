#include "kernel_operator.h"
#include "swi_glu_tiling.h"
#include "swi_glu_vector.h"
#include "swi_glu_cube.h"
#include "swi_glu_resident.h"
#include "swi_glu_generic.h"

// 本算子只使用基础Mmad；若引入Matmul对象，须恢复库的工作区初始化。
namespace matmul {
__aicore__ inline void clearWorkspace(__gm__ uint8_t *workspace) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    AscendC::SetAtomicNone();
    if ASCEND_IS_AIC {
        AscendC::SetMaskNorm();
        AscendC::SetLoadDataBoundary((uint64_t)0);
        AscendC::SetLoadDataPaddingValue((uint64_t)0);
    } else {
        AscendC::SetVectorMask<uint64_t, AscendC::MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
        AscendC::SetMaskNorm();
    }
#endif
}
} // namespace matmul

using namespace AscendC;

__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                 GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                 GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if TILING_KEY_IS(0) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSingleRow<1, 1>(input_x, weight_1, bias_1, weight_2, bias_2, output);
        return;
    } else if TILING_KEY_IS(1) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSingleRow<4, 1>(input_x, weight_1, bias_1, weight_2, bias_2, output);
        return;
    } else if TILING_KEY_IS(2) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSingleRow<32, 1>(input_x, weight_1, bias_1, weight_2, bias_2, output);
        return;
    } else if TILING_KEY_IS(3) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSingleRow<64, 2>(input_x, weight_1, bias_1, weight_2, bias_2, output);
        return;
    } else if TILING_KEY_IS(4) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSmall13<21>(input_x, weight_1, bias_1, weight_2, bias_2, output, 0);
        return;
    } else if TILING_KEY_IS(13) {
        KERNEL_TASK_TYPE(13, KERNEL_TYPE_AIV_ONLY);
        REGISTER_TILING_DEFAULT(SwiGluTilingData);
        GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, plan, tiling);
        swiglu::RunSmall13<0>(input_x, weight_1, bias_1, weight_2, bias_2, output, plan.rows);
        return;
    } else if TILING_KEY_IS(6) {
        KERNEL_TASK_TYPE(6, KERNEL_TYPE_AIV_ONLY);
        REGISTER_TILING_DEFAULT(SwiGluTilingData);
        GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, plan, tiling);
        swiglu::RunDynamicRow(input_x, weight_1, bias_1, weight_2, bias_2, output, plan.dim);
        return;
    } else if TILING_KEY_IS(9) {
        KERNEL_TASK_TYPE(9, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSmall33(input_x, weight_1, bias_1, weight_2, bias_2, output);
        return;
    } else if TILING_KEY_IS(10) {
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_MIX_AIC_1_2);
        swiglu::KernelCubePipeline<256, 64, 64, 4> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
        op.Process();
        return;
    } else if TILING_KEY_IS(11) {
        KERNEL_TASK_TYPE(11, KERNEL_TYPE_MIX_AIC_1_2);
        swiglu::KernelCubePipeline<128, 128, 128, 2> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
        op.Process();
        return;
    } else if TILING_KEY_IS(12) {
        KERNEL_TASK_TYPE(12, KERNEL_TYPE_MIX_AIC_1_2);
        TPipe pipe;
        constexpr SwiGluTilingData fixedPlan{8192, 256, 10, 13};
        swiglu::KernelGateResident<256, 8192, 10> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, fixedPlan, &pipe);
        op.Process();
        return;
    } else if TILING_KEY_IS(7) {
        KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIC_1_2);
        REGISTER_TILING_DEFAULT(SwiGluTilingData);
        GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
        TPipe pipe;
        if (tiling_data.residentGroups != 0 && tiling_data.dim == 256) {
            swiglu::KernelGateResident<256> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling_data, &pipe);
            op.Process();
        } else if (tiling_data.residentGroups != 0 && tiling_data.dim == 128) {
            swiglu::KernelGateResident<128> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling_data, &pipe);
            op.Process();
        } else {
            swiglu::KernelFullK op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling_data.dim, tiling_data.rows, &pipe);
            op.Process();
        }
    } else if TILING_KEY_IS(8) {
        KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
        REGISTER_TILING_DEFAULT(SwiGluTilingData);
        GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
        TPipe pipe;
        swiglu::KernelSwiGluSplitK op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace,
                tiling_data.dim, tiling_data.rows, &pipe);
        op.Process();
    }

}
