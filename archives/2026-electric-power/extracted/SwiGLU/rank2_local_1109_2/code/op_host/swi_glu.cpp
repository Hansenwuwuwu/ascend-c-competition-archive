#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/swi_glu_tiling.h"

namespace optiling {

static bool PlanGateResident(const platform_ascendc::PlatformAscendC &platform,
                             uint32_t rows, uint32_t dim, uint32_t aicNum,
                             SwiGluTilingData *tiling, uint32_t &usedAic,
                             size_t &workspace) {
    // 驻留路径支持完整K=128/256及整齐M块；其他shape保留通用尾块处理。
    if ((dim != 128 && dim != 256) || rows % GATE_RESIDENT_BM != 0) {
        return false;
    }
    const uint32_t columns = dim / GATE_RESIDENT_BN;
    const uint32_t totalPanels = rows / GATE_RESIDENT_BM;
    const uint32_t groups = std::min(aicNum / columns, totalPanels);
    if (groups == 0) {
        return false;
    }
    const uint32_t panels = (totalPanels + groups - 1) / groups;
    if (panels > GATE_RESIDENT_MAX_PANELS) {
        return false;
    }

    uint64_t l1 = 0, ub = 0, l0a = 0, l0b = 0, l0c = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    const uint64_t xTile = uint64_t(GATE_RESIDENT_BM) * dim * sizeof(uint16_t);
    const uint64_t weight = uint64_t(GATE_RESIDENT_BN) * dim * sizeof(uint16_t);
    const uint64_t vectorTile = uint64_t(GATE_RESIDENT_VM) * GATE_RESIDENT_BN * sizeof(uint16_t);
    const uint64_t ubBytes = (panels + 6) * vectorTile + 2 * GATE_RESIDENT_BN * sizeof(uint16_t) + 256;
    const uint64_t cBytes = uint64_t(4) * GATE_RESIDENT_BM * GATE_RESIDENT_BN * sizeof(float);
    if (panels * xTile + weight > l1 || ubBytes + 1024 > ub ||
        2 * xTile > l0a || weight > l0b || cBytes > l0c) {
        return false;
    }

    tiling->residentGroups = groups;
    tiling->residentPanels = panels;
    usedAic = groups * columns;
    workspace = size_t(usedAic) * panels * GATE_RESIDENT_BM * GATE_RESIDENT_BN * sizeof(uint16_t) * 2;
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto &shape = context->GetInputShape(0)->GetStorageShape();
    if (shape.GetDimNum() == 0) {
        return ge::GRAPH_FAILED;
    }
    // X视为[M,D]：M展平所有前导维，D为末维；投影的N、K都等于D。
    // W按[输出列N,K]存储，两路投影均计算X * W^T，bias长度为D。
    int64_t d = shape.GetDim(shape.GetDimNum() - 1);
    if (d < 1 || d > 8192) {
        return ge::GRAPH_FAILED;
    }
    int64_t m = shape.GetShapeSize() / d;
    if (m < 1 || m > 1048576) {
        return ge::GRAPH_FAILED;
    }

    SwiGluTilingData *tiling = context->GetTilingData<SwiGluTilingData>();
    tiling->rows = m;
    tiling->dim = d;
    tiling->residentGroups = 0;
    tiling->residentPanels = 0;

    uint32_t aicNum = platform.GetCoreNumAic();
    uint32_t aivNum = platform.GetCoreNumAiv();
    // 小M的大D和单行尾块也走通用Cube路径，避免旧Vector/尾块布局限制。
    if (d > TILE_L1_BASEK || (m <= aivNum && d >= 128) ||
        (m > aivNum && m % TILE_L1_BASEM == 1)) {
        uint32_t mTiles = (m + TILE_L1_BASEM - 1) / TILE_L1_BASEM;
        uint32_t nTiles = (d + TILE_L2_BASEN - 1) / TILE_L2_BASEN;
        uint32_t usedAic = std::min(aicNum, mTiles * nTiles);
        context->SetTilingKey(SWIGLU_SPLIT_K);
        context->SetBlockDim(usedAic);
        size_t *workspace = context->GetWorkspaceSizes(1);
        if (workspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspace[0] = platform.GetLibApiWorkSpaceSize() +
            usedAic * 2 * TILE_L1_BASEM * TILE_L2_BASEN * sizeof(uint16_t) * 2;
        return ge::GRAPH_SUCCESS;
    }
    // 两种整齐shape复用同一个Cube流水：X双缓冲，Gate/Up权重同时驻留L0B。
    if ((m == 512 && d == 256) || (m == 4096 && d == 128)) {
        const uint32_t cores = d == 256 ? 8 : 16;
        uint64_t l1 = 0, ub = 0, l0a = 0, l0b = 0, l0c = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
        const uint64_t l1Bytes = d == 256 ? 196608 : 131072;
        const uint64_t ubBytes = d == 256 ? 17152 : 66560;
        if (aicNum >= cores && l0a >= 65536 && l0b >= 65536 && l0c >= 131072 &&
            l1 >= l1Bytes && ub >= ubBytes) {
            context->SetTilingKey(d == 256 ? SWIGLU_CUBE256 : SWIGLU_CUBE128);
            context->SetBlockDim(cores);
            context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + size_t(m) * d * 4;
            return ge::GRAPH_SUCCESS;
        }
    }
    if (m == 85 && d == 33 && aivNum >= 17) {
        context->SetTilingKey(SWIGLU_ROWS33);
        context->SetBlockDim(17);
        context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    } else if ((m == 1 && (d == 1 || d == 4 || d == 32)) || (m == 2 && d == 64)) {
        const SwiGluKey key = d == 1 ? SWIGLU_SCALAR : d == 4 ? SWIGLU_ROW4 :
                             d == 32 ? SWIGLU_ROW32 : SWIGLU_ROW64;
        context->SetTilingKey(key);
        // 保留8核启动；有效计算核数由kernel编译期参数决定。
        context->SetBlockDim(8);
    } else if (m <= aivNum && d == 13) {
        // 21行使用常量路径，其余行数保留动态尾块处理。
        context->SetTilingKey(m == 21 ? SWIGLU_ROWS13_STATIC : SWIGLU_ROWS13_DYNAMIC);
        const uint32_t cores = (m + SMALL13_ROWS - 1) / SMALL13_ROWS;
        context->SetBlockDim(std::min(aivNum, (cores + 7) / 8 * 8));
    } else if (m <= aivNum) {
        context->SetTilingKey(SWIGLU_ROW_DYNAMIC);
        context->SetBlockDim(m);
    } else {
        uint32_t mTiles = (m + TILE_L1_BASEM - 1) / TILE_L1_BASEM;
        uint32_t usedAic = std::min(aicNum, mTiles);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        size_t userWorkspace = usedAic * 2 * TILE_L1_BASEM * TILE_L1_BASEN * sizeof(uint16_t) * 2;
        PlanGateResident(platform, static_cast<uint32_t>(m), static_cast<uint32_t>(d),
                         aicNum, tiling, usedAic, userWorkspace);
        context->SetBlockDim(usedAic);
        currentWorkspace[0] = platform.GetLibApiWorkSpaceSize() + userWorkspace;

        // 静态驻留计划保留常量传播；其他shape使用同一实现的动态计划。
        if (m == 8192 && d == 256 && usedAic == 20 &&
            tiling->residentGroups == 10 && tiling->residentPanels == 13) {
            context->SetTilingKey(SWIGLU_RESIDENT_STATIC);
        } else {
            // D>256已在前面的Split-K分支返回。
            context->SetTilingKey(SWIGLU_FULL_K);
        }
    }

    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}
namespace ops {
    class SwiGlu : public OpDef {
    public:
        explicit SwiGlu(const char *name) : OpDef(name) {
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
}  // namespace ops
