#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *raw = context->GetRequiredInputTensor(0);
    const auto *prior = context->GetRequiredInputTensor(1);
    if (raw == nullptr || prior == nullptr) return ge::GRAPH_FAILED;
    const auto &r = raw->GetStorageShape();
    const auto &p = prior->GetStorageShape();
    const auto dtype = raw->GetDataType();
    if (r.GetDimNum() != 3 || p.GetDimNum() != 2 ||
        r.GetDim(0) < 1 || r.GetDim(0) > 8 || r.GetDim(1) < 1 ||
        r.GetDim(1) > 1048576 || r.GetDim(2) != 6 ||
        p.GetDim(0) != r.GetDim(1) || p.GetDim(1) != 5 ||
        (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        prior->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) return ge::GRAPH_FAILED;
    auto *data = context->GetTilingData<FireCandidateDecodeFusionTilingData>();
    data->batch = r.GetDim(0);
    data->candidates = r.GetDim(1);
    const uint32_t tiles = (data->candidates + 255) / 256;
    if (cores > tiles * data->batch) cores = tiles * data->batch;
    const uint32_t total = data->batch * data->candidates;
    uint32_t PATH = total <= 128 ? (total <= platform.GetCoreNumAiv() ? 0 : 3) : (total <= 65536 ? 2 : 1);
    data->segments = 0;
    if ((PATH == 0 || PATH == 3) && dtype == ge::DT_FLOAT16 && data->candidates % 8 == 0) PATH = 5;
    if (PATH == 2) {
        cores = platform.GetCoreNumAiv();
        if (cores > 32) cores = 32;
        const uint32_t tasks = data->batch * ((data->candidates + 511) / 512);
        if (cores > tasks) cores = tasks;
        const uint32_t segments = cores / data->batch;
        if (segments > 0 && tasks > cores &&
            (data->candidates + segments - 1) / segments <= 576) {
            PATH = 4;
            data->segments = segments;
            cores = segments * data->batch;
        }
    }
    if (PATH == 5) {
        cores = platform.GetCoreNumAiv();
        const uint32_t tasks = data->batch * (data->candidates / 8);
        if (cores > tasks) cores = tasks;
    }
    if (PATH == 3) {
        cores = platform.GetCoreNumAiv();
        const uint32_t tasks = data->batch * ((data->candidates + 1) / 2);
        if (cores > tasks) cores = tasks;
    }
    if (PATH == 0) {
        cores = platform.GetCoreNumAiv();
        if (cores > data->batch * data->candidates) cores = data->batch * data->candidates;
    }
    if (PATH == 1) {
        cores = platform.GetCoreNumAiv();
        const uint32_t groups = (data->candidates + 7) / 8;
        if (cores > groups) cores = groups;
        const uint32_t span = ((groups + cores - 1) / cores) * 8;
        PATH = span <= 768 ? 6 : 7;
    }
    if ((PATH == 6 || PATH == 7) && data->batch % 2 == 0) {
        const uint32_t groups = (data->candidates + 15) / 16;
        const uint32_t span = ((groups + cores - 1) / cores) * 16;
        if (span >= 528 && span <= 704) PATH = 8;
        else if (span > 704 && span <= 832) PATH = 19;
    }
    if (total <= 128 && data->candidates >= 8) {
        PATH = data->candidates % 4 == 0 ? 14 : 10;
        cores = platform.GetCoreNumAiv();
        const uint32_t tasks = data->batch * ((data->candidates + 3) / 4);
        if (cores > tasks) cores = tasks;
    }
    if ((PATH == 2 || PATH == 4) && data->batch >= 4 && data->batch % 2 == 0) {
        const uint32_t groups = (data->candidates + 15) / 16;
        const uint32_t span = ((groups + cores - 1) / cores) * 16;
        if (span >= 96 && span <= 128) PATH = 12;
        else if (span > 128 && span <= 160) PATH = 13;
    }
    if (PATH == 14 && data->batch == 1 &&
        data->candidates / 4 <= platform.GetCoreNumAiv()) PATH = 15;
    if (total <= 128 && data->batch >= 2 && data->candidates >= 8) {
        PATH = 24;
        cores = platform.GetCoreNumAiv();
        const uint32_t tasks = data->batch * ((data->candidates + 7) / 8);
        if (cores > tasks) cores = tasks;
        if (cores == tasks) PATH = 26;
    }
    if (PATH == 13) PATH = 21;
    if (PATH == 8) PATH = 18;
    if (data->batch % 4 == 0 && PATH == 12) PATH = 31;
    if (data->batch % 4 == 0 && PATH == 21) PATH = 32;
    if (PATH == 31 || PATH == 32) {
        cores = platform.GetCoreNumAiv();
        if (cores > 16) cores = 16;
        const uint32_t groups = (data->candidates + 15) / 16;
        const uint32_t span = ((groups + cores - 1) / cores) * 16;
        if (span <= 320) PATH = 33;
    }
    if (data->batch == 4 && data->candidates % 256 == 0) {
        const uint32_t fullTileCores = data->candidates / 256;
        if (fullTileCores <= platform.GetCoreNumAiv()) {
            cores = fullTileCores;
            PATH = 34;
        }
    }
    if (data->batch == 4 && data->candidates >= 1025 && data->candidates % 256 == 1) {
        const uint32_t fullTileCores = data->candidates / 256;
        if (fullTileCores <= platform.GetCoreNumAiv()) {
            cores = fullTileCores;
            PATH = 35;
        }
    }
    if ((PATH == 18 || PATH == 19) && data->batch % 4 == 0 &&
        (data->candidates & (data->candidates - 1)) == 0) {
        PATH = (PATH == 18 ? 36 : 38) + (data->batch == 8 ? 1 : 0);
    }
    if ((PATH == 18 || PATH == 19) && data->batch % 4 == 0 &&
        ((data->candidates - 1) & (data->candidates - 2)) == 0) {
        PATH = (PATH == 18 ? 40 : 42) + (data->batch == 8 ? 1 : 0);
    }
    if (PATH == 26 && data->candidates % 8 == 1) PATH = 42 + data->batch;
    if (PATH == 15 && data->batch == 1 && data->candidates % 8 == 0 &&
        2 * (data->candidates / 8) <= platform.GetCoreNumAiv()) {
        PATH = 51;
        cores = data->candidates / 8;
    }
    if (data->batch == 1 && data->candidates >= 512) {
        const uint32_t available = platform.GetCoreNumAiv();
        const uint32_t span = (data->candidates + available - 1) / available;
        if (span <= 512) {
            const bool fineTail = data->candidates % 128 != 0 && span <= 128;
            const uint32_t tile = fineTail ? 128 : 512;
            cores = (data->candidates + tile - 1) / tile;
            PATH = fineTail ? 53 : 52;
        }
    }
    if (data->batch == 4 && data->candidates > platform.GetCoreNumAiv() * 256 && data->candidates % 256 == 0) {
        const uint32_t tileCores = (data->candidates + CandidateMediumRows - 1) / CandidateMediumRows;
        if (tileCores <= platform.GetCoreNumAiv()) {
            const bool powerTwo = (data->candidates & (data->candidates - 1)) == 0;
            PATH = powerTwo ? 65 : 66; cores = tileCores;
        }
    }
    if (PATH == 51 && dtype == ge::DT_FLOAT && data->candidates >= 32 && data->candidates % 16 == 0) {
        PATH = 54;
        cores = data->candidates / 16;
    }
    data->cores = cores;
    if (PATH == 33 || PATH == 34) {
        uint32_t shift = 0;
        while ((1u << shift) < cores) ++shift;
        data->segments = (1u << shift) == cores ? shift : 32;
    }
    context->SetBlockDim(PATH == 54 ? cores * 4u : (PATH == 51 ? cores * 2u : cores));
    uint32_t DT_RAW_PRED = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_RAW_PRED, PATH);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *raw = context->GetInputShape(0);
    const auto *prior = context->GetInputShape(1);
    auto *out = context->GetOutputShape(0);
    if (!raw || !prior || !out || raw->GetDimNum() != 3 ||
        prior->GetDimNum() != 2 || raw->GetDim(2) != 6 ||
        prior->GetDim(1) != 5 || raw->GetDim(1) != prior->GetDim(0))
        return GRAPH_FAILED;
    *out = *raw;
    out->SetDim(2, 5);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    auto dtype = context->GetInputDataType(0);
    if ((dtype != DT_FLOAT && dtype != DT_FLOAT16) ||
        context->GetInputDataType(1) != DT_FLOAT) return GRAPH_FAILED;
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class FireCandidateDecodeFusion : public OpDef {
public:
    explicit FireCandidateDecodeFusion(const char *name) : OpDef(name)
    {
        this->Input("raw_pred").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("priors").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("decoded").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(FireCandidateDecodeFusion);
}
