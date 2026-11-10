#include <algorithm>
#include <cstring>
#include <climits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "tiling/tiling_api.h"

#include "../op_kernel/swi_glu_tiling.h"
#include "../op_kernel/tiling_key_swi_glu.h"

namespace {

constexpr uint64_t kFixedFactor = 20;
constexpr uint64_t kPerColBytes = 24;
constexpr uint64_t kUbReserve = 8 * 1024;
constexpr uint64_t kAlign16 = 16;

constexpr uint64_t kCubeUbReserve = 16 * 1024;

bool ComputePlan(uint64_t m, uint64_t d, uint32_t coreNum, uint64_t ubSize, SwiGluTilingData &td)
{
    td = {};
    const uint32_t dAlign = static_cast<uint32_t>((d + kAlign16 - 1) / kAlign16 * kAlign16);
    const uint64_t fixedBytes = static_cast<uint64_t>(dAlign) * kFixedFactor;
    const uint64_t budget = ubSize > kUbReserve ? ubSize - kUbReserve : 0;
    if (budget <= fixedBytes + kPerColBytes) {
        return false;
    }
    const uint64_t tileNmax = (budget - fixedBytes) / kPerColBytes;
    if (tileNmax == 0) {
        return false;
    }

    td.m = m;
    td.d = d;
    td.dAlign = dAlign;

    uint64_t usedCoreNum = coreNum < m ? coreNum : m;
    if (usedCoreNum == 0) {
        return false;
    }
    const uint64_t kTransposePerD = dAlign <= 256 ? 300 : 180;
    constexpr uint64_t kTransposeFixed = 16 * 1024;
    if (kTransposePerD * dAlign + kTransposeFixed <= ubSize) {
        td.reserved0 = 1;
        td.tileN = kAlign16;
        if (m < coreNum && d > kAlign16) {
            td.reserved0 = 2;
            usedCoreNum = std::min<uint64_t>(coreNum, m * ((d + 15U) / 16U));
        }
    } else {
        td.reserved0 = 0;
        td.tileN = static_cast<uint32_t>(std::min<uint64_t>(dAlign, tileNmax / kAlign16 * kAlign16));
        if (td.tileN == 0) {
            return false;
        }
    }
    td.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    return true;
}

bool TryUnalignedDirectPlan(platform_ascendc::PlatformAscendC &platform,
                            uint64_t m, uint64_t d, SwiGluTilingData *td, size_t &workspaceBytes)
{
    if (d <= 16U || d > 256U || d % 16U == 0U) { return false; }
    const uint32_t cores = platform.GetCoreNumAic();
    if (cores == 0U) { return false; }
    *td = {};
    td->path = 1;
    td->reserved0 = 8;
    td->m = m;
    td->d = d;
    td->dAlign = (d + 15U) / 16U * 16U;
    td->taskM = m > 512U ? 128U : 32U;
    td->taskN = std::min(td->dAlign, 64U);
    td->tileN = td->taskN;
    td->taskNum = (m + td->taskM - 1U) / td->taskM * ((d + td->taskN - 1U) / td->taskN);
    td->usedCoreNum = std::min(cores, td->taskNum);
    const uint64_t slotBytes = 2ULL * td->taskM * td->taskN * sizeof(uint16_t);
    workspaceBytes = platform.GetLibApiWorkSpaceSize() + slotBytes * 4ULL * td->usedCoreNum;
    return true;
}

bool TryCubePlan(gert::TilingContext *context, platform_ascendc::PlatformAscendC &platform,
                 uint64_t m, uint64_t d, SwiGluTilingData *tiling, size_t &workspaceBytes)
{
    const uint32_t coreNum = platform.GetCoreNumAic();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum == 0 || ubSize == 0) {
        return false;
    }
    *tiling = {};
    const uint32_t computeD = static_cast<uint32_t>((d + 15U) / 16U * 16U);
    const bool padded = computeD != d;
    uint32_t taskM = m <= 32 ? 16U : (m <= 512 ? 32U : 128U);
    uint32_t taskN = std::min(computeD, m <= 32 ? 128U : 256U);

    const bool direct = !padded && d <= 256;
    if (direct) {
        taskM = m > 512U ? 128U : 32U;
        taskN = std::min(computeD, 64U);
    }

    const uint64_t ubBytes = (static_cast<uint64_t>(taskM / 2U) * 4U + 4U + 7U * 16U * 2U) * taskN;
    if (ubBytes + kCubeUbReserve + (padded ? 8192U : 0U) > ubSize) {
        return false;
    }
    const uint32_t tasksN = static_cast<uint32_t>((d + taskN - 1) / taskN);
    const uint32_t taskNum = static_cast<uint32_t>((m + taskM - 1) / taskM) * tasksN;
    uint32_t blockDim = std::min(coreNum, taskNum);
    matmul_tiling::MatmulApiTiling mmTiling(platform);

    const bool merged = direct || (padded && taskN == computeD);
    const uint32_t matmulN = merged ? 2U * taskN : taskN;
    const bool basic = false;
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT16, false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT16, true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT16);
    mmTiling.SetBias(false);
    mmTiling.SetMadType(matmul_tiling::MatrixMadType::NORMAL);

    mmTiling.SetShape(taskM, matmulN, computeD);
    mmTiling.SetOrgShape(taskM, matmulN, computeD, computeD);
    if (basic) { mmTiling.SetFixSplit(taskM, 128, 128); }
    mmTiling.SetBufferSpace(-1, -1, -1);
    if (mmTiling.GetTiling(tiling->cubeTilingData) == -1 ||
        tiling->cubeTilingData.baseM <= 0 || tiling->cubeTilingData.baseN <= 0) {
        return false;
    }
    if (basic && (tiling->cubeTilingData.baseM != static_cast<int32_t>(taskM) ||
                  tiling->cubeTilingData.baseN != 128 || tiling->cubeTilingData.baseK != 128)) {
        return false;
    }
    tiling->cubeTilingData.singleCoreM = taskM;
    tiling->cubeTilingData.singleCoreN = matmulN;
    tiling->cubeTilingData.singleCoreK = computeD;
    tiling->cubeTilingData.usedCoreNum = 1;
    tiling->path = 1;
    tiling->reserved0 = merged ? 4U : 0U;
    if (direct) { tiling->reserved0 = 7U; }
    if (basic) {
        tiling->reserved0 = taskM == 16U ? 3U : (taskM == 32U ? 5U : 6U);
    }
    tiling->m = m;
    tiling->d = d;
    tiling->dAlign = computeD;
    tiling->taskM = taskM;
    tiling->taskN = taskN;
    tiling->taskNum = taskNum;
    tiling->tileN = taskN;
    tiling->usedCoreNum = blockDim;

    const uint64_t slotBytes = 2ULL * taskM * taskN * sizeof(uint16_t);
    const uint64_t paddedBytes = padded ? (m + 2ULL * computeD) * computeD * sizeof(uint16_t) : 0;
    workspaceBytes = platform.GetLibApiWorkSpaceSize() + slotBytes * 4ULL * blockDim + paddedBytes;
    return true;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto *xShape = context->GetInputShape(0);
    const auto *w1Shape = context->GetInputShape(1);
    const auto *b1Shape = context->GetInputShape(2);
    const auto *w2Shape = context->GetInputShape(3);
    const auto *b2Shape = context->GetInputShape(4);
    const auto *xDesc = context->GetInputDesc(0);
    if (xShape == nullptr || w1Shape == nullptr || b1Shape == nullptr || w2Shape == nullptr ||
        b2Shape == nullptr || xDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (xDesc->GetDataType() != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &xS = xShape->GetStorageShape();
    const size_t xDim = xS.GetDimNum();
    if (xDim == 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t dSigned = xS.GetDim(xDim - 1);
    if (dSigned <= 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t d = static_cast<uint64_t>(dSigned);
    if (d > 8192) {
        return ge::GRAPH_FAILED;
    }
    uint64_t m = 1;
    for (size_t i = 0; i + 1 < xDim; ++i) {
        const int64_t dim = xS.GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        if (static_cast<uint64_t>(dim) > 1048576ULL / m) {
            return ge::GRAPH_FAILED;
        }
        m *= static_cast<uint64_t>(dim);
    }
    const gert::Shape &w1S = w1Shape->GetStorageShape();
    const gert::Shape &b1S = b1Shape->GetStorageShape();
    const gert::Shape &w2S = w2Shape->GetStorageShape();
    const gert::Shape &b2S = b2Shape->GetStorageShape();
    const bool shapeOk = w1S.GetDimNum() == 2 && w1S.GetDim(0) == dSigned && w1S.GetDim(1) == dSigned &&
                         w2S.GetDimNum() == 2 && w2S.GetDim(0) == dSigned && w2S.GetDim(1) == dSigned &&
                         b1S.GetDimNum() == 1 && b1S.GetDim(0) == dSigned &&
                         b2S.GetDimNum() == 1 && b2S.GetDim(0) == dSigned;
    if (!shapeOk) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t aivCoreNum = platform.GetCoreNumAiv();
    if (aivCoreNum == 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    SwiGluTilingData *tiling = context->GetTilingData<SwiGluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(sizeof(SwiGluTinyTilingData));

    const uint32_t DT_INPUT_X = static_cast<uint32_t>(xDesc->GetDataType());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (m==1U && d==4U && aivCoreNum>=8U) {
        tiling->m=m;tiling->d=d;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,43U);
        context->SetBlockDim(8U);
        currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }
    if (m==2U && d==64U && aivCoreNum>=8U) {
        tiling->m=m;tiling->d=d;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,44U);
        context->SetBlockDim(8U);
        currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }
    if (m==512U && d==256U && platform.GetCoreNumAic()>=16U) {
        tiling->m=m;tiling->d=d;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,45U);
        context->SetBlockDim(16U);
        currentWorkspace[0]=platform.GetLibApiWorkSpaceSize()+4ULL*16U*2ULL*48U*64U*sizeof(uint16_t);
        return ge::GRAPH_SUCCESS;
    }
    if (m==1U && d==32U && aivCoreNum>=8U) {
        tiling->m=m;tiling->d=d;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,46U);
        context->SetBlockDim(8U);
        currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }
    if (m==4096U && d==128U && platform.GetCoreNumAic()>=16U) {
        tiling->m=m;tiling->d=d;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,47U);
        context->SetBlockDim(16U);
        currentWorkspace[0]=platform.GetLibApiWorkSpaceSize()+4ULL*16U*2ULL*64U*128U*sizeof(uint16_t);
        return ge::GRAPH_SUCCESS;
    }
    if (m==1U && d==1U && aivCoreNum>=8U) {
        tiling->m=m;tiling->d=d;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,48U);
        context->SetBlockDim(8U);
        currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }
    if (d==13U && m==21U && aivCoreNum>=8U) {
        tiling->m=m;tiling->d=d;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,49U);
        context->SetBlockDim(8U);
        currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }
    if (d==33U && m==85U && aivCoreNum>=17U) {
        tiling->m=m;tiling->d=d;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,50U);
        context->SetBlockDim(17U);currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }
    if (m==8192U && d==256U && platform.GetCoreNumAic()>=20U) {
        tiling->m=m;tiling->d=d;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,51U);
        context->SetBlockDim(20U);
        currentWorkspace[0]=platform.GetLibApiWorkSpaceSize()+4ULL*20U*2ULL*128U*128U*sizeof(uint16_t);
        return ge::GRAPH_SUCCESS;
    }

    context->GetRawTilingData()->SetDataSize(sizeof(SwiGluTilingData));
    // Reuse the already registered generic kernel for non-specialized shapes.
    if (!ComputePlan(m, d, aivCoreNum, ubSize, *tiling)) { return ge::GRAPH_FAILED; }
    tiling->reserved0 = 0;
    tiling->usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum, m));
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, 0U);
    context->SetBlockDim(tiling->usedCoreNum);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
    if (m==21U && d==13U && aivCoreNum>=3U) {
        *tiling={};tiling->m=m;tiling->d=d;tiling->usedCoreNum=3U;
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,42U);
        context->SetBlockDim(3U);currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }

    if (d>1U && d<=16U && m*d<=aivCoreNum) {
        *tiling={};tiling->m=m;tiling->d=d;tiling->usedCoreNum=m*d;
        const uint32_t route=m==1U?(d==13U?38U:39U):(d==13U?40U:41U);
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,route);
        context->SetBlockDim(tiling->usedCoreNum);currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }

    if (d <= 16U && m <= aivCoreNum) {
        *tiling={};tiling->m=m;tiling->d=d;tiling->usedCoreNum=m;
        const uint32_t route=d==13U?23U:(d==1U?24U:25U);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, route);
        context->SetBlockDim(tiling->usedCoreNum);currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }

    if (d <= 16U) {
        *tiling = {};
        tiling->m=m; tiling->d=d; tiling->dAlign=16;
        const bool batched=d!=1U && m>=2ULL*aivCoreNum;
        const uint32_t route=batched?(d==13U?21U:22U):(d==1U?15U:(d==13U?14U:16U));
        const uint64_t tasks=batched?(m+3U)/4U:(d==1U?(m+255U)/256U:m);
        tiling->usedCoreNum=static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum,tasks));
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, route);
        context->SetBlockDim(tiling->usedCoreNum);
        currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }

    if (d <= 512U && m*((d+15U)/16U) <= 4ULL*aivCoreNum && (d<=256U || m<=4U)) {
        *tiling={};tiling->m=m;tiling->d=d;tiling->dAlign=(d+15U)/16U*16U;
        const uint32_t route=d>256U?(d==512U?36U:37U):(d==64U?17U:(d==128U?18U:(d==256U?19U:20U)));
        tiling->usedCoreNum=static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum,m*((d+15U)/16U)));
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, route);
        context->SetBlockDim(tiling->usedCoreNum);currentWorkspace[0]=0;
        return ge::GRAPH_SUCCESS;
    }

    if (d>16U && d<=256U && m>=256U && m<=512U) {
        *tiling={};tiling->m=m;tiling->d=d;tiling->dAlign=(d+15U)/16U*16U;
        tiling->taskM=64U;tiling->taskN=std::min(tiling->dAlign,64U);
        tiling->taskNum=(m+63U)/64U*((d+tiling->taskN-1U)/tiling->taskN);
        tiling->usedCoreNum=std::min(platform.GetCoreNumAic(),tiling->taskNum);
        const uint32_t route=d==256U?32U:(d==128U?33U:(d%16U==0U?34U:35U));
        ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,route);
        context->SetBlockDim(tiling->usedCoreNum);
        currentWorkspace[0]=platform.GetLibApiWorkSpaceSize()+4ULL*tiling->usedCoreNum*2ULL*64U*tiling->taskN*sizeof(uint16_t);
        return ge::GRAPH_SUCCESS;
    }

    if (d > 16U && d <= 256U && m > 512U) {
        const uint32_t n=std::min<uint32_t>((d+15U)/16U*16U,64U);
        const uint32_t tasksN=(d+n-1U)/n;
        const uint32_t tasksM=(m+127U)/128U;
        const uint32_t cores=platform.GetCoreNumAic()/tasksN*tasksN;
        if (cores!=0U && tasksM*tasksN>=2U*cores) {
            *tiling={};tiling->m=m;tiling->d=d;tiling->dAlign=(d+15U)/16U*16U;
            tiling->taskM=128U;tiling->taskN=n;tiling->taskNum=tasksM*tasksN;tiling->usedCoreNum=cores;
            const bool full=m%128U==0U;
            const uint32_t route=d==256U?(full?30U:26U):(d==128U?(full?31U:27U):(d%16U==0U?28U:29U));
            ASCENDC_TPL_SEL_PARAM(context,DT_INPUT_X,route);
            context->SetBlockDim(cores);
            currentWorkspace[0]=platform.GetLibApiWorkSpaceSize()+4ULL*cores*2ULL*128U*n*sizeof(uint16_t);
            return ge::GRAPH_SUCCESS;
        }
    }

    size_t directWorkspace = 0;
    if (TryUnalignedDirectPlan(platform, m, d, tiling, directWorkspace)) {
        const uint32_t route = tiling->taskM == 128U ? 13U : 12U;
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, route);
        context->SetBlockDim(tiling->usedCoreNum);
        currentWorkspace[0] = directWorkspace;
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t alignedD = (d + 15U) / 16U * 16U;
    const bool padCube = d >= 64 && d <= 2048 && m >= 16 && m * alignedD <= 16ULL * 1024 * 1024;
    if (d % 16 == 0 || padCube) {
        size_t workspaceBytes = 0;
        if (TryCubePlan(context, platform, m, d, tiling, workspaceBytes)) {
            uint32_t route = 1;
            if (tiling->reserved0 == 7U) {
                const bool largeM = tiling->taskM == 128U;
                route = d == 256U ? (largeM ? 9U : 8U) :
                        (d == 128U ? (largeM ? 11U : 10U) : (largeM ? 3U : 2U));
            } else if (tiling->dAlign != d) {
                route = tiling->reserved0 == 4U ? 7U : 6U;
            }
            ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, route);

            context->SetBlockDim(tiling->usedCoreNum);
            currentWorkspace[0] = workspaceBytes;
            return ge::GRAPH_SUCCESS;
        }
    }

    if (!ComputePlan(m, d, aivCoreNum, ubSize, *tiling)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t route = tiling->reserved0 == 1U ? 4U : (tiling->reserved0 == 2U ? 5U : 0U);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, route);
    context->SetBlockDim(tiling->usedCoreNum);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const auto *input = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    if (input->GetDimNum() == 1 && input->GetDim(0) == -2) {
        output->SetDimNum(1);
        output->SetDim(0, -2);
        return GRAPH_SUCCESS;
    }
    output->SetDimNum(input->GetDimNum());
    for (size_t i = 0; i < input->GetDimNum(); ++i) {
        output->SetDim(i, input->GetDim(i));
    }
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetInputDataType(0);
    if (dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, dtype);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class SwiGlu : public OpDef {
public:
    explicit SwiGlu(const char *name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("weight_1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("bias_1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("weight_2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("bias_2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(SwiGlu);
}
