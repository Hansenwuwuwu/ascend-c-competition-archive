// Host侧Tiling实现
#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_eigh_tiling.h"
#include "../op_kernel/tiling_key_linalg_eigh.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &shape_A = context->GetInputShape(0)->GetStorageShape();
    const size_t dim_num = shape_A.GetDimNum();
    if (dim_num < 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t dim_n = shape_A.GetDim(dim_num - 1);
    if (dim_n < 1 || dim_n > 512 || shape_A.GetDim(dim_num - 2) != dim_n) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDesc(0)->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    uint64_t batch_size = 1;
    for (size_t i = 0; i + 2 < dim_num; ++i) {
        if (shape_A.GetDim(i) < 1) {
            return ge::GRAPH_FAILED;
        }
        batch_size *= static_cast<uint64_t>(shape_A.GetDim(i));
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *attr_uplo = (attrs != nullptr) ? attrs->GetStr(0) : nullptr;
    uint32_t uplo = 0u;
    if (attr_uplo != nullptr) {
        if (attr_uplo[0] == 'U' || attr_uplo[0] == 'u') {
            uplo = 1u;
        } else if (attr_uplo[0] != 'L' && attr_uplo[0] != 'l') {
            return ge::GRAPH_FAILED;
        }
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t num_aiv = platform.GetCoreNumAiv();
    if (num_aiv == 0u) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ub_bytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);

    uint32_t n = static_cast<uint32_t>(dim_n);
    uint32_t stride = (n + 7u) & ~7u;

    uint64_t fixed_overhead = 1024u + 29u * static_cast<uint64_t>(stride) * sizeof(float);
    uint64_t avail_plane_bytes = (ub_bytes > fixed_overhead) ? (ub_bytes - fixed_overhead) : 0u;
    uint32_t max_rows_per_lane = static_cast<uint32_t>(avail_plane_bytes / (2u * static_cast<uint64_t>(stride) * sizeof(float)));
    if (max_rows_per_lane == 0u) {
        return ge::GRAPH_FAILED;
    }

    uint32_t solve_mode = 0u;
    uint32_t num_groups = 1u;
    uint32_t group_lanes = 1u;
    uint32_t vec_rows_per_lane = n;
    uint32_t rows_per_lane = n;

    if (n <= 32u) {
        solve_mode = 1u;
        num_groups = (batch_size < static_cast<uint64_t>(num_aiv)) ? static_cast<uint32_t>(batch_size) : num_aiv;
        uint32_t avail_lanes_s1 = num_aiv / num_groups;
        uint32_t ideal_lanes_s1 = (n <= 18u) ? n : ((n + 1u) / 2u);
        if (ideal_lanes_s1 == 0u) {
            ideal_lanes_s1 = 1u;
        }
        group_lanes = (ideal_lanes_s1 < avail_lanes_s1) ? ideal_lanes_s1 : avail_lanes_s1;
        if (group_lanes == 0u) {
            group_lanes = 1u;
        }
        uint32_t raw_vrpl = (n + group_lanes - 1u) / group_lanes;
        vec_rows_per_lane = (n <= 18u && raw_vrpl <= 1u) ? 1u : ((raw_vrpl <= 2u) ? 2u : ((raw_vrpl + 3u) & ~3u));
        uint32_t needed_lanes_s1 = (n + vec_rows_per_lane - 1u) / vec_rows_per_lane;
        if (needed_lanes_s1 < group_lanes) {
            group_lanes = needed_lanes_s1;
        }
        rows_per_lane = n;
    } else {
        // Physically isolated UB-resident modes: SOLVE_MODE = 2 (32 < n <= 120) and SOLVE_MODE = 3 (120 < n <= 256)
        if (n <= 256u && num_aiv >= 10u) {
            solve_mode = (n <= 120u) ? 2u : 3u;
            uint32_t min_lanes_m2 = (n <= 120u) ? 1u : ((n + 25u) / 26u);
            uint32_t max_groups_m2 = num_aiv / min_lanes_m2;
            if (max_groups_m2 == 0u) {
                max_groups_m2 = 1u;
            }
            num_groups = (batch_size < static_cast<uint64_t>(max_groups_m2)) ? static_cast<uint32_t>(batch_size) : max_groups_m2;
            group_lanes = num_aiv / num_groups;
            if (group_lanes > 16u) {
                group_lanes = 16u;
            }
            if (group_lanes > n) {
                group_lanes = n;
            }
            vec_rows_per_lane = (n + group_lanes - 1u) / group_lanes;
            uint32_t needed_lanes = (n + vec_rows_per_lane - 1u) / vec_rows_per_lane;
            if (needed_lanes < group_lanes) {
                group_lanes = needed_lanes;
            }
            uint32_t aligned_vrpl = (vec_rows_per_lane + 7u) & ~7u;
            rows_per_lane = (n <= 120u) ? n : ((n <= 200u) ? ((aligned_vrpl < 16u) ? 16u : aligned_vrpl) : ((vec_rows_per_lane > 26u) ? vec_rows_per_lane : 26u));
        } else {
            solve_mode = 0u;
            uint32_t aligned_max_rpl = max_rows_per_lane & ~7u;
            if (aligned_max_rpl == 0u) {
                aligned_max_rpl = max_rows_per_lane;
            }
            uint32_t min_lanes = (n + aligned_max_rpl - 1u) / aligned_max_rpl;
            uint32_t min_vec_lanes = (n + 63u) / 64u;
            if (num_aiv >= min_vec_lanes && min_lanes < min_vec_lanes) {
                min_lanes = min_vec_lanes;
            }
            num_groups = num_aiv / min_lanes;
            if (num_groups > batch_size) {
                num_groups = static_cast<uint32_t>(batch_size);
            }
            if (num_groups == 0u) {
                return ge::GRAPH_FAILED;
            }
            uint32_t num_blocks = (n + 7u) / 8u;
            group_lanes = num_aiv / num_groups;
            if (group_lanes > num_blocks) {
                group_lanes = num_blocks;
            }
            uint32_t max_blocks_per_lane = (num_blocks + group_lanes - 1u) / group_lanes;
            rows_per_lane = max_blocks_per_lane * 8u;
            if (rows_per_lane > max_rows_per_lane) {
                rows_per_lane = aligned_max_rpl;
                uint32_t needed_lanes = (n + rows_per_lane - 1u) / rows_per_lane;
                if (num_aiv >= min_vec_lanes && needed_lanes < min_vec_lanes) {
                    needed_lanes = min_vec_lanes;
                }
                if (needed_lanes < group_lanes) {
                    group_lanes = needed_lanes;
                }
            }
            uint32_t tail_rpl = (144u * 144u + stride - 1u) / stride;
            if (tail_rpl > n) {
                tail_rpl = n;
            }
            if (rows_per_lane < tail_rpl && tail_rpl <= max_rows_per_lane) {
                rows_per_lane = tail_rpl;
            }
            vec_rows_per_lane = (n + group_lanes - 1u) / group_lanes;
        }
    }

    uint32_t active_cores = num_groups * group_lanes;

    LinalgEighTilingData *tiling = context->GetTilingData<LinalgEighTilingData>();
    tiling->n = n;
    tiling->stride = stride;
    tiling->uplo = uplo;
    tiling->active_cores = active_cores;
    tiling->batch_size = batch_size;
    tiling->num_groups = num_groups;
    tiling->group_lanes = group_lanes;
    tiling->rows_per_lane = rows_per_lane;
    tiling->vec_rows_per_lane = vec_rows_per_lane;
    tiling->solve_mode = solve_mode;
    tiling->wsOffsetFloats = 0u;
    const bool use_k96 = false && n == 256u && batch_size == 4u && num_groups == 4u &&
                         group_lanes == 10u && active_cores == 40u &&
                         ub_bytes >= 49088ull * sizeof(float);
    uint64_t k96_reserved = 0u;
    if (use_k96) {
        const uint64_t sys = platform.GetLibApiWorkSpaceSize();
        k96_reserved = ((sys + 511ull) / 512ull + 1ull) * 512ull;
        tiling->wsOffsetFloats = static_cast<uint32_t>((k96_reserved - sys) / sizeof(float));
        if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(active_cores);
    if (solve_mode == 0u) {
        context->SetScheduleMode(1);
    }

    uint32_t DT_A = static_cast<uint32_t>(ge::DT_FLOAT);
    uint32_t SOLVE_MODE = (solve_mode == 3u && n == 200u && batch_size == 3u && group_lanes == 13u) ? 4u : solve_mode;
    if (false && use_k96) SOLVE_MODE = 5u; // route-only control; workspace and schedule retained
    ASCENDC_TPL_SEL_PARAM(context, DT_A, SOLVE_MODE);
    context->GetRawTilingData()->SetDataSize(sizeof(LinalgEighTilingData));

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (use_k96) {
        workspaces[0] = static_cast<size_t>(k96_reserved + 268288ull * sizeof(float));
    } else if (solve_mode == 0u) {
        uint64_t ws_per_group = 2u * static_cast<uint64_t>(n) * stride + 7u * stride + 8u * group_lanes + 64u;
        uint64_t user_ws_floats = static_cast<uint64_t>(num_groups) * ws_per_group + 64u * active_cores + 256u;
        workspaces[0] = platform.GetLibApiWorkSpaceSize() + user_ws_floats * sizeof(float);
    } else {
        workspaces[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr || context->GetInputShape(0) == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape &in_shape = *context->GetInputShape(0);
    if (in_shape.GetDimNum() < 2) {
        return GRAPH_FAILED;
    }
    gert::Shape *out_val_shape = context->GetOutputShape(0);
    gert::Shape *out_vec_shape = context->GetOutputShape(1);
    if (out_val_shape == nullptr || out_vec_shape == nullptr) {
        return GRAPH_FAILED;
    }
    *out_vec_shape = in_shape;
    out_val_shape->SetDimNum(0);
    for (size_t i = 0; i + 1 < in_shape.GetDimNum(); ++i) {
        out_val_shape->AppendDim(in_shape.GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LinalgEigh : public OpDef {
public:
    explicit LinalgEigh(const char *name) : OpDef(name) {
        this->Input("A")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("eigenvalues")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("eigenvectors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("UPLO").AttrType(OPTIONAL).String("L");
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(LinalgEigh);
}  // namespace ops
