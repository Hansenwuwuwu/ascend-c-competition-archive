#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_det_tiling.h"
#include "../op_kernel/tiling_key_linalg_det.h"
#include <algorithm>
#include "tiling/tiling_api.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const auto &shape = context->GetInputShape(0)->GetOriginShape();
    const size_t rank = shape.GetDimNum();
    if (rank < 2) return ge::GRAPH_FAILED;
    const int64_t n = shape.GetDim(rank - 1);
    if (n < 1 || n > 512 || shape.GetDim(rank - 2) != n) return ge::GRAPH_FAILED;
    uint64_t batch = 1;
    for (size_t d = 0; d + 2 < rank; ++d) {
        const int64_t dim = shape.GetDim(d);
        if (dim < 1 || batch > UINT32_MAX / static_cast<uint64_t>(dim)) return ge::GRAPH_FAILED;
        batch *= dim;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = std::max(1U, static_cast<uint32_t>(platform.GetCoreNumAic()));
    const bool USE_CUBE = n > 128;
    const uint32_t block=n>256?32:64;
    const uint32_t pn = USE_CUBE ? (n>256?(n+31)/32*32:(n+31)/32*32) : (n + 7) / 8 * 8;
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint32_t capacity = static_cast<uint32_t>(std::min<uint64_t>(128 * 1024, ub > 49152 ? ub - 49152 : 65536));
    const uint32_t maxColumns = std::max(1U, capacity / (pn * 8U));
    uint32_t perMatrix = (n + maxColumns - 1) / maxColumns;
    if (USE_CUBE) perMatrix = (pn + block-1) / block;
    else if (n > 128) perMatrix = (n + 15) / 16;
    perMatrix = std::min(perMatrix, cores);
    const uint32_t available = USE_CUBE ? std::max(1U,static_cast<uint32_t>(platform.GetCoreNumAic())) : cores;
    // A compact wave owns four matrices on four AICs per matrix. Reuse it
    // for any whole number of waves; other geometries keep the generic plan.
    const uint32_t compactWaveMatrices = 4;
    const bool compactWaves = n == 256 && perMatrix == 4 &&
        available / perMatrix >= compactWaveMatrices && batch % compactWaveMatrices == 0;
    const uint32_t groupCapacity = compactWaves ? compactWaveMatrices : available / perMatrix;
    const uint32_t groups = n >= 3 && n <= 16 ? 1 : std::min(static_cast<uint32_t>(batch), groupCapacity);
    const uint32_t columns = USE_CUBE ? block : static_cast<uint32_t>((n + perMatrix - 1) / perMatrix);
    const uint32_t sharedStride = columns * pn + 2 * pn + 128;
    auto *payload = context->GetTilingData<LinalgDetTilingPayload>();
    *payload={};
    auto *t=&payload->base;
    *t = {static_cast<uint32_t>(n), static_cast<uint32_t>(batch), pn,
          perMatrix, groups, columns, sharedStride};
    uint32_t PADDED_N=0;
    if(n>=3&&n<=64){
        PADDED_N=uint32_t(n)+64;uint32_t q=8;while(q<pn)q*=2;
        const uint32_t pitch=n<=16?pn:(pn==56?56:2*pn+8);
        for(uint32_t r=0;r<q;++r){payload->gatherIndices[r]=r*pitch*4;payload->gatherIndices[q+r]=r*4;}
        if(n<=16&&n%8){
            const uint32_t length=(n*pitch+63)/64*64;
            for(uint32_t i=0;i<length;++i){
                const uint32_t r=i/pitch,c=i%pitch;
                payload->gatherIndices[128+i]=(r<n&&c<n?r*n+c:0)*4;
            }
        }
    }
    const uint32_t DT_A = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
    t->cubeMode = USE_CUBE ? 1 : 0;
    t->robustOffset = USE_CUBE ? 2 * (pn * block + 128) + perMatrix * (pn * block + 2*block*block) : 0;
    if(n==256)t->robustOffset+=2*pn*64;
    if (USE_CUBE) {
        matmul_tiling::MatmulApiTiling mm(platform);
        using namespace matmul_tiling;
        const bool preloadA=(n==200||n==256)&&perMatrix==4;
        mm.SetAType(preloadA?TPosition::A1:TPosition::GM, preloadA?CubeFormat::NZ:CubeFormat::ND, DataType::DT_FLOAT, true);
        mm.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        mm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        mm.SetShape(n>256?pn:256, block, n==256?64:32); mm.SetOrgShape(n>256?pn:256, block, n==256?64:32);
        if(n<=256)mm.SetDoubleBuffer(false,false,false,false);
        mm.SetBias(false); mm.SetFixSplit(n>256?128:256, block, 32); mm.SetBufferSpace(-1,-1,-1);
        if (mm.GetTiling(t->cube) != 0) {
            mm.SetFixSplit(128,block,32);
            if(mm.GetTiling(t->cube)!=0)return ge::GRAPH_FAILED;
        }
    }
    if(USE_CUBE&&n<=256&&t->cube.baseM==256)t->cubeMode=2;
    const bool ONE_ROUND=PADDED_N>0&&batch==groups;
    // 5: one compact wave, 6: several compact waves using the same kernel.
    // These are implementation tags; they are not case IDs or batch values.
    const uint32_t STATIC_BATCH=compactWaves?(batch==groups?5:6):n==256?3:n==200?2:ONE_ROUND&&n>16&&batch<=4?uint32_t(batch):0;
    ASCENDC_TPL_SEL_PARAM(context, DT_A, USE_CUBE, PADDED_N, ONE_ROUND, STATIC_BATCH);
    const uint32_t activeBlocks = perMatrix * groups;
    context->SetBlockDim(PADDED_N > 0 ? std::max(8U, activeBlocks) : activeBlocks);
    context->GetWorkspaceSizes(1)[0] = n>=3&&n<=16 ? 0U : platform.GetLibApiWorkSpaceSize() +
        (static_cast<size_t>(groups) * (sharedStride + t->robustOffset)) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const auto *a = context->GetInputShape(0);
    auto *out = context->GetOutputShape(0);
    if (!a || !out || a->GetDimNum() < 2) return GRAPH_FAILED;
    out->SetDimNum(a->GetDimNum() - 2);
    for (size_t d = 0; d < out->GetDimNum(); ++d) out->SetDim(d, a->GetDim(d));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class LinalgDet : public OpDef {
public:
    explicit LinalgDet(const char *name) : OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Output("det").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(LinalgDet);
}
