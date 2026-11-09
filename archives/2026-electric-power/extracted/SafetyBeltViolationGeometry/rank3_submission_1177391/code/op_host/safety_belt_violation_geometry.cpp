#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/safety_belt_violation_geometry_tiling.h"
#include "../op_kernel/tiling_key_safety_belt_violation_geometry.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

#ifndef SBG_STATIC_SMALL
#define SBG_STATIC_SMALL 1U
#endif
static_assert(SBG_STATIC_SMALL <= 1U, "invalid static-small selector");

namespace {
uint32_t AlignRows(uint32_t rows)
{
    return ((rows + 63U) / 64U) * 64U;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *person = context->GetRequiredInputTensor(0);
    const auto *anchor = context->GetRequiredInputTensor(1);
    const auto *threshold = context->GetOptionalInputTensor(2);
    if (person == nullptr || anchor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &ps = person->GetOriginShape();
    const auto &as = anchor->GetOriginShape();
    if (ps.GetDimNum() != 2U || as.GetDimNum() != 2U || ps.GetDim(1) != 4 ||
        as.GetDim(1) != 2 || ps.GetDim(0) != as.GetDim(0) ||
        ps.GetDim(0) < 1 || ps.GetDim(0) > 4096) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = person->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) || anchor->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }
    const auto *outDesc = context->GetOutputDesc(0);
    if (outDesc == nullptr) return ge::GRAPH_FAILED;
    const auto outType = outDesc->GetDataType();
    const bool legacyIO = dtype == ge::DT_FLOAT16 && outType == ge::DT_FLOAT16;
    if (outType != ge::DT_FLOAT && !legacyIO) return ge::GRAPH_FAILED;
    if (threshold != nullptr && (threshold->GetDataType() != (legacyIO ? ge::DT_FLOAT16 : ge::DT_FLOAT) ||
        threshold->GetOriginShape().GetDimNum() != 1U || threshold->GetShapeSize() != 1)) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    const float *ratio = attrs == nullptr ? nullptr : attrs->GetFloat(0);
    // Follow the problem contract: waist_ratio is in (0,1].
    if (ratio == nullptr || !std::isfinite(*ratio) || *ratio <= 0.0f || *ratio > 1.0f) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t available = 8U;
    const uint32_t batch = static_cast<uint32_t>(ps.GetDim(0));
    const uint32_t workers = std::min(available, (batch + SBG_MAX_LOCAL_ROWS - 1U) / SBG_MAX_LOCAL_ROWS);
    const bool splitMedium = dtype == ge::DT_FLOAT && batch > 128U && batch <= 384U;
    const uint32_t rowsPerCore = splitMedium ? (batch <= 256U ? 128U : 192U) : AlignRows((batch + workers - 1U) / workers);
    if (rowsPerCore > SBG_MAX_LOCAL_ROWS) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<SafetyBeltViolationGeometryTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->slots[0].pack = (batch - 1U) | ((rowsPerCore / 64U) << 12U) |
                            (static_cast<uint32_t>(threshold != nullptr) << 16U);
    tiling->slots[0].waistRatio = *ratio;
    const uint32_t used = (batch + rowsPerCore - 1U) / rowsPerCore;
    // Generic capacity classes; no batch values appear in the key.
    uint32_t gridClass = 0U;
    uint32_t capacity = 0U;
    if (used == 1U && rowsPerCore == 64U) {
        gridClass = SBG_STATIC_SMALL != 0U ? 16U : 9U;
        capacity = 64U;
    } else if (used == 1U && rowsPerCore == 256U) {
        gridClass = 10U;
        capacity = 256U;
    } else if (used == 1U && rowsPerCore == 320U) {
        gridClass = 11U;
        capacity = 320U;
    } else if (rowsPerCore == SBG_MAX_LOCAL_ROWS && used <= 8U) {
        gridClass = used;
        capacity = SBG_MAX_LOCAL_ROWS;
        if (used == 8U) {
            uint32_t ratioIndex = 0U;
            bool constantRatio = true;
            if (*ratio == 0.8f) {
                ratioIndex = 0U;
            } else if (*ratio == 0.9f) {
                ratioIndex = 1U;
            } else if (*ratio == 1.0f) {
                ratioIndex = 2U;
            } else if (*ratio == 0.5f) {
                ratioIndex = 3U;
            } else {
                constantRatio = false;
            }
            if (constantRatio) {
                gridClass = 12U + ratioIndex;
            }
        }
    }
    if (splitMedium) { gridClass = batch <= 256U ? 17U : 18U; capacity = rowsPerCore; }
    // One 256-row full tile plus a 1..64-row vector tail for Float32 coordinates.
    const bool preserveFullTile = used == 1U && batch == rowsPerCore;
    const bool vectorTailPair = dtype == ge::DT_FLOAT && batch > 256U && batch - 256U <= 64U && !preserveFullTile;
    if (vectorTailPair) { gridClass = 28U; capacity = 256U; }
    const bool staticMediumBand = gridClass == 28U && *ratio == 0.5f;
    if (staticMediumBand) { gridClass = 29U; capacity = 256U; }
    constexpr uint32_t vectorRows = 64U;
    constexpr uint32_t lastVectorStart = SBG_MAX_LOCAL_ROWS - vectorRows;
    const uint32_t tailRows = batch % SBG_MAX_LOCAL_ROWS;
    const bool staticLastVector = gridClass == 12U && tailRows >= lastVectorStart;
    if (staticLastVector) gridClass = 19U;
    const uint32_t tailPresent = gridClass == 16U ? 0U : (gridClass != 0U ?
        static_cast<uint32_t>((batch % capacity) != 0U) : 0U);
    // Static lengths for the complete Half320 capacity class at this attribute.
    // Other attributes keep the existing generic/constant-attribute routes.
    const bool staticHalf320 = dtype == ge::DT_FLOAT16 && gridClass == 11U &&
        tailPresent != 0U && *ratio == 0.5f;
    if (staticHalf320) gridClass = 30U;
    const uint32_t constantRatioMode = *ratio == 1.0f ? 1U : (*ratio == 0.8f ? 2U : (*ratio == 0.9f ? 3U : (*ratio == 0.5f ? 4U : 0U)));
    uint32_t ratioMode = (gridClass == 29U || gridClass == 28U || gridClass == 16U || gridClass == 17U || gridClass == 18U || (dtype == ge::DT_FLOAT16 && (gridClass == 10U || gridClass == 11U))) ? (*ratio == 1.0f ? 1U : (*ratio == 0.8f ? 2U : (*ratio == 0.9f ? 3U : (*ratio == 0.5f ? 4U : 0U)))) : 0U;
    if (gridClass >= 1U && gridClass <= 7U && constantRatioMode != 0U) ratioMode = constantRatioMode;
    if (staticHalf320) ratioMode = 4U;
    // Keep every small length known before input DMA, and share the existing
    // runtime attribute path for non-unit ratios. It fetches ratio after both DMAs.
    // Preserve optional-threshold specialization and every small row count.
    // Only the unit attribute uses its existing constant-ratio kernel; all other
    // valid attributes share the original runtime-ratio implementation.
    if (gridClass == 16U) ratioMode = *ratio == 1.0f ? 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context,
                          static_cast<uint32_t>(dtype),
                          static_cast<uint32_t>(outType),
                          gridClass,
                          tailPresent,
                          static_cast<uint32_t>(threshold != nullptr),
                          staticLastVector ? tailRows - lastVectorStart : gridClass == 16U ? batch - 1U : (gridClass == 29U ? batch - (256U + 1U) : (staticHalf320 ? batch - (capacity - 64U + 1U) : 0U)),
                          ratioMode);
    context->SetBlockDim(std::min(available, std::max(used, 8U)));
    context->GetWorkspaceSizes(1)[0] = 0U;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *person = context->GetInputShape(0);
    const auto *anchor = context->GetInputShape(1);
    auto *output = context->GetOutputShape(0);
    if (person == nullptr || anchor == nullptr || output == nullptr ||
        person->GetDimNum() != 2U || anchor->GetDimNum() != 2U ||
        person->GetDim(1) != 4 || anchor->GetDim(1) != 2 || person->GetDim(0) != anchor->GetDim(0)) {
        return GRAPH_FAILED;
    }
    *output = gert::Shape();
    output->AppendDim(person->GetDim(0));
    output->AppendDim(1);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // Required Float32 threshold / absent threshold infer Float32 output.
    // The additional Half-threshold protocol preserves compatibility with existing callers.
    const auto thresholdType = context->GetOptionalInputDataType(2);
    const bool legacyIO = context->GetInputDataType(0) == ge::DT_FLOAT16 && thresholdType == ge::DT_FLOAT16;
    context->SetOutputDataType(0, legacyIO ? ge::DT_FLOAT16 : ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SafetyBeltViolationGeometry : public OpDef {
public:
    explicit SafetyBeltViolationGeometry(const char *name) : OpDef(name)
    {
        this->Input("person_boxes").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("anchor_points").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("threshold").ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("violation_score").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("waist_ratio").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SafetyBeltViolationGeometry);
}
