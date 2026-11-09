// Host侧Tiling实现
#include "register/op_def_registry.h"
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_solve_tiling.h"
#include "../op_kernel/tiling_key_linalg_solve.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_A = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_B = context->GetRequiredInputTensor(1);
        ge::DataType dtype_A = tensor_A->GetDataType();
        uint32_t length_A = tensor_A->GetShapeSize();

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const bool *attr_left = attrs->GetBool(0);
        uint32_t DT_A = static_cast<uint32_t>(dtype_A);

        LinalgSolveTilingData *tiling = context->GetTilingData<LinalgSolveTilingData>();
        const auto &as = tensor_A->GetStorageShape();
        const auto &bs = tensor_B->GetStorageShape();
        uint32_t n = as.GetDim(as.GetDimNum() - 1);
        uint32_t k = bs.GetDimNum() == 1 ? 1 : bs.GetDim(bs.GetDimNum() - 1);
        if (bs.GetDimNum() == as.GetDimNum() - 1 && bs.GetDim(bs.GetDimNum() - 1) == n &&
            (bs.GetDimNum() < 2 || bs.GetDim(bs.GetDimNum() - 2) != n)) {
            k = 1;
        }
        if (n == 0 || n > 512 || as.GetDim(as.GetDimNum() - 2) != n || k == 0 || k > 256 ||
            (attr_left && !*attr_left)) {
            return ge::GRAPH_FAILED;
        }

        uint32_t batch = length_A / (n * n);
        uint32_t SHAPE = 0;
        if (dtype_A == ge::DT_FLOAT && n >= 8 && batch <= uint32_t(num_cores_aiv)) {
            if (n <= 16 && k == 1) SHAPE = 1;
            else if (n <= 64 && k <= 8) SHAPE = 2;
            else if (n == 200 && k == 4) SHAPE = 3;
            else if (n == 256 && k == 3) SHAPE = 4;
        }
        ASCENDC_TPL_SEL_PARAM(context, DT_A, SHAPE);

        tiling->n = n;
        tiling->k = k;
        tiling->batch = batch;
        tiling->bBatch = tensor_B->GetShapeSize() / (n * k);

        uint32_t padn = (SHAPE == 3 || SHAPE == 4) ? ((n + 15) / 16 * 16) : ((n + 7) / 8 * 8);
        if (SHAPE == 1 || SHAPE == 2) {
            tiling->parts = 1;
            tiling->stride = padn;
            tiling->groups = std::min(batch, uint32_t(num_cores_aiv));
            tiling->cores = tiling->groups;
        } else if (SHAPE == 3 || SHAPE == 4) {
            tiling->parts = 8;
            tiling->stride = ((padn / 8 + 7) / 8) * 8;
            tiling->groups = std::min(batch, uint32_t(num_cores_aiv) / tiling->parts);
            if (tiling->groups == 0) tiling->groups = 1;
            tiling->cores = tiling->groups * tiling->parts;
        } else {
            tiling->stride = padn;
            tiling->parts = 1;
            while (padn * tiling->stride > 40960 ||
                   (8 * tiling->stride + 2 * padn > 4096 - 8 * padn &&
                    8 * tiling->stride + 2 * padn > 40960 - padn * tiling->stride)) {
                tiling->parts *= 2;
                tiling->stride = ((n + tiling->parts - 1) / tiling->parts + 7) / 8 * 8;
            }
            tiling->groups = std::min(batch * k, uint32_t(num_cores_aiv) / tiling->parts);
            tiling->cores = tiling->groups * tiling->parts;
        }
        if (SHAPE != 0) {
            tiling->n = n | (batch << 9) | ((tiling->bBatch == 1 ? 1u : 0u) << 15) |
                        (tiling->cores << 16) | (k << 22) | (tiling->groups << 26);
        }
        tiling->rows = 0;

        context->SetBlockDim(tiling->cores);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        size_t wsElements = (SHAPE == 1 || SHAPE == 2) ? 0
            : (SHAPE == 3 || SHAPE == 4)
            ? (size_t(tiling->groups) * (n * 16 + 4 * padn + 128 + n * padn) + size_t(batch) * k * padn)
            : (size_t(tiling->groups) * 2 * (padn + 8) + (size_t(tiling->groups) * n + size_t(batch) * k) * padn);
        currentWorkspace[0] = platform.GetLibApiWorkSpaceSize() + wsElements * sizeof(float);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto *a = context->GetInputShape(0);
        const auto *b = context->GetInputShape(1);
        auto *x = context->GetOutputShape(0);
        if (!a || !b || !x || a->GetDimNum() < 2 || b->GetDimNum() < 1) return GRAPH_FAILED;
        int64_t n = a->GetDim(a->GetDimNum() - 1);
        bool vec = b->GetDimNum() == 1 ||
            (b->GetDimNum() == a->GetDimNum() - 1 &&
             b->GetDim(b->GetDimNum() - 1) == n &&
             b->GetDim(b->GetDimNum() - 2) != n);
        *x = *a;
        if (vec) {
            x->SetDimNum(a->GetDimNum() - 1);
            x->SetDim(a->GetDimNum() - 2, n);
        } else {
            x->SetDim(a->GetDimNum() - 1, b->GetDim(b->GetDimNum() - 1));
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class LinalgSolve : public OpDef {
    public:
        explicit LinalgSolve(const char *name) : OpDef(name) {
            this->Input("A")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("B")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("X")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("left").AttrType(OPTIONAL).Bool(true);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LinalgSolve);
}  // namespace ops
