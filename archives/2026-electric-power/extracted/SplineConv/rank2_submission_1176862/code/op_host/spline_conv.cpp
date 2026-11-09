#include <algorithm>
#include <cstring>
#include <vector>
#include <stdexcept>
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/spline_conv_tiling.h"
#include "../op_kernel/tiling_key_spline_conv.h"

namespace optiling {
static ge::graphStatus TilingImpl(gert::TilingContext* context, bool withRootBias) {
    const auto* x = context->GetInputShape(0);
    const auto* edges = context->GetInputShape(1);
    const auto* pseudo = context->GetInputShape(2);
    const auto* weight = context->GetInputShape(3);
    const auto* output = context->GetOutputShape(0);
    const auto* xd = context->GetInputDesc(0);
    const auto* ed = context->GetInputDesc(1);
    const auto* pd = context->GetInputDesc(2);
    const auto* wd = context->GetInputDesc(3);
    const auto* yd = context->GetOutputDesc(0);
    if (!x || !edges || !pseudo || !weight || !output || !xd || !ed || !pd || !wd || !yd)
        return ge::GRAPH_FAILED;
    const auto& xs = x->GetStorageShape();
    const auto& es = edges->GetStorageShape();
    const auto& ps = pseudo->GetStorageShape();
    const auto& ws = weight->GetStorageShape();
    const auto& ys = output->GetStorageShape();
    if (xs.GetDimNum() != 2 || es.GetDimNum() != 2 || ps.GetDimNum() != 2 ||
        ws.GetDimNum() == 0 || ys.GetDimNum() != 2)
        return ge::GRAPH_FAILED;
    const auto* attrs = context->GetAttrs();
    if (!attrs) return ge::GRAPH_FAILED;
    const auto* dimAttr = attrs->GetInt(0);
    const auto* sizes = attrs->GetListInt(1);
    const auto* open = attrs->GetListInt(2);
    const auto* degreeAttr = attrs->GetInt(3);
    const char* aggr = attrs->GetStr(4);
    if (!dimAttr || !sizes || !open || !degreeAttr || !aggr) return ge::GRAPH_FAILED;
    const int64_t dim = *dimAttr, degree = *degreeAttr;
    const int64_t n = xs.GetDim(0), ci = xs.GetDim(1);
    const int64_t e = es.GetDim(1), co = ys.GetDim(1);
    if (dim < 1 || dim > 3 || degree < 1 || degree > 3 ||
        n < 1 || n > INT32_MAX || e < 0 || e > INT32_MAX - 31 ||
        ci < 1 || ci > INT32_MAX || co < 1 || co > INT32_MAX ||
        es.GetDim(0) != 2 || ps.GetDim(0) != e || ps.GetDim(1) != dim || ys.GetDim(0) != n ||
        (std::strcmp(aggr, "add") != 0 && std::strcmp(aggr, "sum") != 0 &&
         std::strcmp(aggr, "mean") != 0)) return ge::GRAPH_FAILED;
    const auto dtype = xd->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_BF16) || ed->GetDataType() != ge::DT_INT64 ||
        pd->GetDataType() != dtype || wd->GetDataType() != dtype || yd->GetDataType() != dtype)
        return ge::GRAPH_FAILED;
    // DataCopyExtParams uses a 32-bit byte stride for each GM matrix row.
    if (static_cast<uint64_t>(co) * ge::GetSizeByDataType(dtype) > UINT32_MAX)
        return ge::GRAPH_FAILED;
    auto* data = context->GetTilingData<SplineConvTilingData>();
    if (!data) return ge::GRAPH_FAILED;
    *data = {};
    uint32_t kernels = 1;
    const auto* sizeValues = sizes->GetData();
    const auto* openValues = open->GetData();
    // Uniform scalar/list attributes broadcast across dimensions, including
    // the template's two-element defaults when dim=3.
    const auto uniform = [](const gert::TypedContinuousVector<int64_t>* values, int64_t dims) {
        if (values->GetSize() == static_cast<size_t>(dims)) return true;
        if (values->GetSize() == 0 || values->GetSize() > 3) return false;
        for (size_t i = 1; i < values->GetSize(); ++i)
            if (values->GetData()[i] != values->GetData()[0]) return false;
        return true;
    };
    if (!uniform(sizes, dim) || !uniform(open, dim)) return ge::GRAPH_FAILED;
    for (int64_t d = 0; d < dim; ++d) {
        const int64_t kd = sizeValues[sizes->GetSize() == static_cast<size_t>(dim) ? d : 0];
        const int64_t od = openValues[open->GetSize() == static_cast<size_t>(dim) ? d : 0];
        if (kd < 1 || kd > 16 || (od != 0 && od != 1))
            return ge::GRAPH_FAILED;
        data->kernelSize[d] = static_cast<uint32_t>(kd);
        data->open[d] = static_cast<uint32_t>(od);
        kernels *= static_cast<uint32_t>(kd);
    }
    if (static_cast<uint64_t>(ci) >
        static_cast<uint64_t>(INT64_MAX) / kernels / static_cast<uint64_t>(co))
        return ge::GRAPH_FAILED;
    // The declared kernel sizes index only the first K contiguous Ci-by-Co
    // matrices. Additional complete matrices are valid unused storage.
    if (ws.GetShapeSize() < static_cast<int64_t>(kernels) * ci * co ||
        ws.GetShapeSize() % (ci * co) != 0)
        return ge::GRAPH_FAILED;
    const auto* root = withRootBias ? context->GetOptionalInputShape(4) : nullptr;
    const auto* bias = withRootBias ? context->GetOptionalInputShape(5) : nullptr;
    const auto* rd = withRootBias ? context->GetOptionalInputDesc(4) : nullptr;
    const auto* bd = withRootBias ? context->GetOptionalInputDesc(5) : nullptr;
    if (root && (!rd || rd->GetDataType() != dtype || root->GetStorageShape().GetDimNum() != 2 ||
                 root->GetStorageShape().GetDim(0) != ci || root->GetStorageShape().GetDim(1) != co))
        return ge::GRAPH_FAILED;
    if (bias && (!bd || bd->GetDataType() != dtype || bias->GetStorageShape().GetDimNum() != 1 ||
                 bias->GetStorageShape().GetDim(0) != co)) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = std::max(1U, static_cast<uint32_t>(platform.GetCoreNumAiv()));
    data->nodes = static_cast<uint32_t>(n);
    data->edges = static_cast<uint32_t>(e);
    data->inChannels = static_cast<uint32_t>(ci);
    data->outChannels = static_cast<uint32_t>(co);
    data->dim = static_cast<uint32_t>(dim);
    data->degree = static_cast<uint32_t>(degree);
    data->mean = std::strcmp(aggr, "mean") == 0;
    data->hasRoot = root != nullptr;
    data->hasBias = bias != nullptr;
    const uint32_t nodeBlocks = std::min(cores, data->nodes);
    const uint32_t columnTiles = (data->outChannels + 127U) / 128U;
    data->blocks = static_cast<uint32_t>(std::min<uint64_t>(cores,
        static_cast<uint64_t>(data->nodes) * columnTiles));
    data->nodeTile = std::min(4096U, (data->nodes + nodeBlocks - 1) / nodeBlocks);
    data->edgeStride = (data->edges + 31U) / 32U * 32U;
    // Resident bilinear matrices, aligned vector rows, and cyclic output
    // ownership form a general small sparse graph domain.
    uint32_t paddedInput = 8;
    while (paddedInput < data->inChannels && paddedInput <= 128) paddedInput *= 2;
    uint32_t residentOwners = 1;
    uint32_t ownerShift = 0;
    while (residentOwners < data->edges && residentOwners < 8U && residentOwners * 2U <= cores) {
        residentOwners *= 2U;
        ++ownerShift;
    }
    const uint32_t residentRows = (data->nodes + residentOwners - 1U) / residentOwners;
    const uint64_t residentCounts = (static_cast<uint64_t>(residentRows) + 7U) / 8U * 8U;
    const uint64_t residentMatrix = static_cast<uint64_t>(paddedInput) * data->outChannels;
    const uint64_t residentUb = 1536U + (residentMatrix >= 512U ? 512U : 0U) +
        4U * (5U * residentMatrix +
        std::max(residentMatrix, residentCounts * data->outChannels) +
        static_cast<uint64_t>(residentRows) * data->outChannels +
        2U * residentCounts + static_cast<uint64_t>(std::max(1U, data->edges)) * paddedInput);
    const bool resident = residentUb <= 184U * 1024U && dtype == ge::DT_FLOAT && !root && !bias &&
        dim == 2 && degree == 1 && data->kernelSize[0] == 2 &&
        data->kernelSize[1] == 2 && data->open[0] == 1 && data->open[1] == 1 &&
        ci <= 128 && ci % 8 == 0 && co <= 32 && co % 8 == 0 &&
        n <= 2048 && e > 0 && e <= 64;
    if (resident) {
        data->schedule = 1;
        data->owners = residentOwners;
        data->blocks = std::min(8U, cores);
        data->ownerShift = ownerShift;
        data->paddedInput = paddedInput;
    }
    const uint32_t DT_X = static_cast<uint32_t>(dtype);
    const uint32_t SCHEDULE = data->schedule;
    // Two 32-byte blocks per output row allow fully constant vector strides.
    const uint32_t PADDED_ROWS = resident && co == 16 ? paddedInput : 0U;
    uint32_t NODE_LOG = 0U;
    if (resident && e == 1 && co == 16 && ci == paddedInput &&
        (data->nodes & (data->nodes - 1U)) == 0U) {
        uint32_t remainingRows = data->nodes;
        while (remainingRows) { ++NODE_LOG; remainingRows >>= 1U; }
        if (NODE_LOG > SPLINE_STATIC_NODE_MAX_LOG) NODE_LOG = 0U;
    }
    // A complete output tile of at most 2KiB has a bounded constant layout.
    // Preserve geometric keys; encode every remaining legal row count in
    // this whole interval, independent of graph endpoints or tensor values.
    if (resident && e == 1 && co == 16 && ci == paddedInput && NODE_LOG == 0U &&
        data->nodes <= SPLINE_TINY_SINGLE_NODE_MAX) {
        NODE_LOG = SPLINE_TINY_SINGLE_NODE_OFFSET + data->nodes;
    }
    const uint32_t EDGE_MODE = NODE_LOG ? 3U : resident && e == 1 ? 1U :
        resident && co == 16 && ci == paddedInput && e >= SPLINE_EARLY_WEIGHT_MIN_EDGES ? 2U : 0U;
    const uint32_t FULL_WORKERS = resident && data->blocks == 8U && e == data->blocks &&
        co == 16 && ci == paddedInput &&
        4U * residentMatrix * sizeof(float) >= SPLINE_STREAMED_WEIGHT_BYTES ? 1U : 0U;
    if (FULL_WORKERS && data->nodes <= SPLINE_FULL_BITMAP_NODE_MAX) {
        NODE_LOG = PADDED_ROWS == 128U &&
            data->nodes % SPLINE_FULL_VECTOR_TILE_NODE_MULTIPLE == 0U ?
            SPLINE_FULL_VECTOR_TILE_NODE_MODE +
                data->nodes / SPLINE_FULL_VECTOR_TILE_NODE_MULTIPLE :
            PADDED_ROWS == 128U && data->nodes % SPLINE_FULL_COMPACT_NODE_MULTIPLE == 0U ?
                SPLINE_FULL_COMPACT_NODE_OFFSET +
                    data->nodes / SPLINE_FULL_COMPACT_NODE_MULTIPLE : 1U;
    }
    const uint32_t HALF_WORKERS = resident && data->blocks == 8U &&
        e == data->blocks / 2U && co == 16 && ci == paddedInput &&
        paddedInput >= SPLINE_HALF_OWNER_MIN_ROWS &&
        ((data->nodes & (data->nodes - 1U)) == 0U ||
         data->nodes <= SPLINE_TINY_HALF_NODE_MAX) ? 1U : 0U;
    if (HALF_WORKERS) {
        if ((data->nodes & (data->nodes - 1U)) == 0U) {
            uint32_t remainingRows = data->nodes;
            while (remainingRows) { ++NODE_LOG; remainingRows >>= 1U; }
        } else NODE_LOG = SPLINE_TINY_HALF_NODE_OFFSET + data->nodes;
    }
    const bool staticFullHeader = FULL_WORKERS &&
        ((NODE_LOG >= SPLINE_FULL_STATIC_NODE_MODE_MIN &&
          NODE_LOG <= SPLINE_FULL_STATIC_NODE_MODE_MAX) ||
         (NODE_LOG >= SPLINE_FULL_COMPACT_NODE_MODE_MIN &&
          NODE_LOG <= SPLINE_FULL_COMPACT_NODE_MODE_MAX));
    const uint32_t HALF_MEAN_MODE =
        (HALF_WORKERS || staticFullHeader) ? data->mean + 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context, DT_X, SCHEDULE, PADDED_ROWS, EDGE_MODE, NODE_LOG, FULL_WORKERS, HALF_WORKERS, HALF_MEAN_MODE);
    context->SetBlockDim(data->blocks);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
        (data->schedule ? 0 : static_cast<size_t>(data->blocks) * data->edgeStride * sizeof(int32_t));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    return TilingImpl(context, false);
}
static ge::graphStatus TilingFullFunc(gert::TilingContext* context) {
    return TilingImpl(context, true);
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const auto* x = context->GetInputShape(0);
    const auto* w = context->GetInputShape(3);
    auto* out = context->GetOutputShape(0);
    const auto* attrs = context->GetAttrs();
    if (!x || !w || !out || !attrs || x->GetDimNum() != 2 || w->GetDimNum() == 0)
        return GRAPH_FAILED;
    const auto* dims = attrs->GetInt(0);
    const auto* sizes = attrs->GetListInt(1);
    if (!dims || !sizes || *dims < 1 || *dims > 3 || sizes->GetSize() == 0 ||
        sizes->GetSize() > 3 || x->GetDim(1) <= 0) return GRAPH_FAILED;
    if (sizes->GetSize() != static_cast<size_t>(*dims)) {
        for (size_t d = 1; d < sizes->GetSize(); ++d)
            if (sizes->GetData()[d] != sizes->GetData()[0]) return GRAPH_FAILED;
    }
    int64_t kernels = 1;
    for (int64_t d = 0; d < *dims; ++d) {
        const int64_t size = sizes->GetData()[sizes->GetSize() == static_cast<size_t>(*dims) ? d : 0];
        if (size < 1 || size > 16) return GRAPH_FAILED;
        kernels *= size;
    }
    if (x->GetDim(1) > INT64_MAX / kernels) return GRAPH_FAILED;
    const int64_t matrixRows = kernels * x->GetDim(1);
    if (w->GetShapeSize() <= 0) return GRAPH_FAILED;
    int64_t co = 0;
    if (w->GetDimNum() >= 3 && w->GetDim(w->GetDimNum() - 2) == x->GetDim(1)) {
        co = w->GetDim(w->GetDimNum() - 1);
    } else if (w->GetDimNum() == 2 && w->GetDim(1) % x->GetDim(1) == 0) {
        co = w->GetDim(1) / x->GetDim(1);
    } else if (w->GetShapeSize() % matrixRows == 0) {
        co = w->GetShapeSize() / matrixRows;
    }
    if (co < 1 || co > INT64_MAX / matrixRows || w->GetShapeSize() < matrixRows * co ||
        w->GetShapeSize() % (x->GetDim(1) * co) != 0) return GRAPH_FAILED;
    out->SetDimNum(2);
    out->SetDim(0, x->GetDim(0));
    out->SetDim(1, co);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SplineConv : public OpDef {
public:
    explicit SplineConv(const char* name, bool withRootBias = false) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("edge_index").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("edge_attr").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        if (withRootBias) {
            this->Input("root_weight").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("bias").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(2);
        this->Attr("kernel_size").AttrType(OPTIONAL).ListInt({2, 2});
        this->Attr("is_open_spline").AttrType(OPTIONAL).ListInt({1, 1});
        this->Attr("degree").AttrType(OPTIONAL).Int(1);
        this->Attr("aggr").AttrType(OPTIONAL).String("mean");
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        if (withRootBias) {
            OpAICoreConfig config;
            config.ExtendCfgInfo("opFile.value", "spline_conv")
                .ExtendCfgInfo("opInterface.value", "spline_conv_full");
            this->AICore().SetTiling(optiling::TilingFullFunc).AddConfig("ascend910b", config);
        } else {
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    }
};
class SplineConvFull : public SplineConv {
public:
    explicit SplineConvFull(const char* name) : SplineConv(name, true) {}
};
OP_ADD(SplineConv);
OP_ADD(SplineConvFull);
}

// Export callbacks for the self-contained ACLNN binary package.
IMPL_OP_OPTILING(SplineConv).Tiling(optiling::TilingFunc)
    .InferShape(ge::InferShape).InferDataType(ge::InferDataType);
IMPL_OP_OPTILING(SplineConvFull).Tiling(optiling::TilingFullFunc)
    .InferShape(ge::InferShape).InferDataType(ge::InferDataType);

#ifdef OP_TILING_LIB
#include <dlfcn.h>
#include "register/op_impl_kernel_registry.h"
#include "../pkg_inc/base/registry/opp_package_utils.h"

// CANN 9.0.0's platform CMake exposes include/, but not pkg_inc/. These
// declarations are copied unchanged from pkg_inc/base/registry/
// op_impl_space_registry_v2.h (Huawei, CANN Open Software License v2).
namespace gert {
using char_t = char;
class OpImplRegistryHolder;
class OpImplSpaceRegistryImpl;
using OpImplSpaceRegistryImplPtr = std::shared_ptr<OpImplSpaceRegistryImpl>;
class OpImplSpaceRegistryV2 {
 public:
  OpImplSpaceRegistryV2();
  ~OpImplSpaceRegistryV2() = default;
  ge::graphStatus AddSoToRegistry(const OppSoDesc& so_desc);
  const OpImplKernelRegistry::OpImplFunctionsV2 *GetOpImpl(const char_t *op_type) const;
 private:
  friend class OpImplSpaceRegistry;
  ge::graphStatus AddRegistry(const std::shared_ptr<OpImplRegistryHolder> &registry_holder);
  OpImplKernelRegistry::OpImplFunctionsV2 *CreateOrGetOpImpl(const char_t *op_type);
  OpImplSpaceRegistryImplPtr impl_;
};
class DefaultOpImplSpaceRegistryV2Impl;
using DefaultOpImplSpaceRegistryV2ImplPtr = std::shared_ptr<DefaultOpImplSpaceRegistryV2Impl>;
class DefaultOpImplSpaceRegistryV2 {
 public:
  DefaultOpImplSpaceRegistryV2(const DefaultOpImplSpaceRegistryV2 &) = delete;
  DefaultOpImplSpaceRegistryV2(const DefaultOpImplSpaceRegistryV2 &&) = delete;
  DefaultOpImplSpaceRegistryV2 &operator=(const DefaultOpImplSpaceRegistryV2 &) & = delete;
  DefaultOpImplSpaceRegistryV2 &operator=(const DefaultOpImplSpaceRegistryV2 &&) & = delete;
  static DefaultOpImplSpaceRegistryV2 &GetInstance();
  const std::shared_ptr<OpImplSpaceRegistryV2> GetSpaceRegistry(
      gert::OppImplVersionTag opp_impl_version = gert::OppImplVersionTag::kOpp) const;
  ge::graphStatus SetSpaceRegistry(const std::shared_ptr<OpImplSpaceRegistryV2> &space_registry_v2,
      gert::OppImplVersionTag version_tag = gert::OppImplVersionTag::kOpp);
  void ClearSpaceRegistry();
 private:
  DefaultOpImplSpaceRegistryV2();
  ~DefaultOpImplSpaceRegistryV2() = default;
  DefaultOpImplSpaceRegistryV2ImplPtr impl_;
};
}

namespace {
static bool RegisterSplinePackage() {
    if (gert::OppPackageUtils::LoadAllOppPackage() != ge::GRAPH_SUCCESS)
        throw std::runtime_error("SplineConv: cannot load CANN operator packages");
    auto& registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance();
    auto space = registry.GetSpaceRegistry();
    if (!space) space = std::make_shared<gert::OpImplSpaceRegistryV2>();
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(optiling::TilingFunc), &info) || !info.dli_fname)
        throw std::runtime_error("SplineConv: cannot locate its shared library");
    const gert::OppSoDesc package({ge::AscendString(info.dli_fname)}, "SplineConv");
    if (space->AddSoToRegistry(package) != ge::GRAPH_SUCCESS ||
        registry.SetSpaceRegistry(space) != ge::GRAPH_SUCCESS)
        throw std::runtime_error("SplineConv: cannot register its Host callbacks");
    const auto* base = space->GetOpImpl("SplineConv");
    const auto* full = space->GetOpImpl("SplineConvFull");
    if (!base || !base->tiling || !full || !full->tiling)
        throw std::runtime_error("SplineConv: missing registered tiling callbacks");
    return true;
}
}

namespace {
// Register while this Host library loads, including builds whose generated
// ACLNN entry does not carry embedded binary-resource calls.
[[maybe_unused]] const bool registeredSplinePackage = RegisterSplinePackage();
}
#endif
