#include "tiling/tiling_api.h"
#include <cstring>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_lstsq_tiling.h"
#include "../op_kernel/tiling_key_linalg_lstsq.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *c) {
    const auto &a=c->GetInputShape(0)->GetStorageShape();
    const auto &b=c->GetInputShape(1)->GetStorageShape();
    const auto rank=a.GetDimNum();
    if(rank<2 || b.GetDimNum()!=rank) return ge::GRAPH_FAILED;
    const int64_t m=a.GetDim(rank-2),n=a.GetDim(rank-1),k=b.GetDim(rank-1);
    if(m<1 || m>1024 || n<1 || n>m || n>512 || k<1 || k>64 || b.GetDim(rank-2)!=m)
        return ge::GRAPH_FAILED;
    uint64_t batches=1;
    for(size_t i=0;i+2<rank;++i) {
        const int64_t dim=a.GetDim(i);
        if(dim<1 || dim>32 || b.GetDim(i)!=dim || batches>UINT32_MAX/(uint64_t)dim)
            return ge::GRAPH_FAILED;
        batches*=dim;
    }
    if(c->GetInputDesc(0)->GetDataType()!=ge::DT_FLOAT || c->GetInputDesc(1)->GetDataType()!=ge::DT_FLOAT)
        return ge::GRAPH_FAILED;
    const char *driver=c->GetAttrs()->GetStr(0);
    if(driver && std::strcmp(driver,"gels")) return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    const uint32_t cores=platform.GetCoreNumAiv();
    if(!cores)return ge::GRAPH_FAILED;
    auto *t=c->GetTilingData<LinalgLstsqTilingData>();
    t->m=m;t->n=n;t->k=k;t->batches=batches;t->rows=8;
    const auto *residualShape=c->GetOutputShape(1);
    t->residualElements=residualShape?residualShape->GetStorageShape().GetShapeSize():0;
    while(t->rows<(uint32_t)m)t->rows*=2;
    uint32_t KERNEL_PATH=7,DT_A=ge::DT_FLOAT;
    t->teams=1;
    // Square systems need no orthogonal projection. A single Vector owns a
    // resident small matrix; column layout keeps each pivot reduction contiguous.
    if(m==n && n<=64 && k<=8) {
        KERNEL_PATH=n<=16?1:2;
        // A maximum-pivot resident initial solve has no cross-core consumer.
        // One owner per matrix retains packed loads without a shared handoff.
        if(n>32)KERNEL_PATH=6;
    } else if(m==n && n>64 && n<=256 && k<=8) {
        // Each lane owns 32 columns plus inverse columns; one lane owns RHS.
        const uint32_t capacity=n<=128?128:256;
        const uint32_t needed=capacity/32+1;
        if(cores>=needed) {t->teams=needed;KERNEL_PATH=capacity==128?3:4;}
    }
    const uint32_t groups=cores/t->teams;
    t->blocks=(batches<groups?batches:groups)*t->teams;
    if(t->teams==1 && KERNEL_PATH!=7) {
        const uint32_t grouped=(t->blocks+7)/8*8;
        t->blocks=grouped<cores?grouped:cores;
    }
    const uint32_t cubeCores=platform.GetCoreNumAic();
    const bool blockOrthogonal=m==n && n>128 && n<=256 && k<=8 &&
        cubeCores>=4 && cores==2*cubeCores && batches<=cubeCores/4;
    if(blockOrthogonal) {KERNEL_PATH=5;t->teams=0;t->blocks=batches*4;c->SetScheduleMode(1);}
    // Panel selection leaves at most 32 trailing rows in either narrow path.
    // Four Cube tiles provide eight Vector lanes for the inverse row groups.
    const bool narrowOrthogonal=blockOrthogonal && n>2*96 && n<=2*96+64 &&
        k<=4 && cubeCores>=4 && batches<=cubeCores/4;
    if(narrowOrthogonal){KERNEL_PATH=0;t->blocks=batches*4;}
    // Two 112-row panels remove the outer tail block when at most 32 rows remain.
    if(narrowOrthogonal && n>2*112 && n<=2*112+32)KERNEL_PATH=8;
    const uint32_t capacity=n<=128?128:256;
    const uint64_t words=narrowOrthogonal?(uint64_t)batches*kNarrowOrthogonalWorkspaceFloats:blockOrthogonal?(uint64_t)batches*kOrthogonalWorkspaceFloats:t->teams>1?
        (uint64_t)(t->blocks/t->teams)*(2*(capacity*16+16)+capacity*capacity+2*k*256):
        (KERNEL_PATH==7?(uint64_t)t->blocks*(2*(n+k)*t->rows+((n*k+7)/8*8)):0);
    ASCENDC_TPL_SEL_PARAM(c,DT_A,KERNEL_PATH);
    c->SetBlockDim(t->blocks);
    c->GetWorkspaceSizes(1)[0]=words*4+platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
    const auto &a=*c->GetInputShape(0),&b=*c->GetInputShape(1);
    const auto rank=a.GetDimNum();
    if(rank<2 || b.GetDimNum()!=rank)return GRAPH_FAILED;
    auto &x=*c->GetOutputShape(0);
    x=a;x.SetDim(rank-2,a.GetDim(rank-1));x.SetDim(rank-1,b.GetDim(rank-1));
    if(auto *r=c->GetOutputShape(1)) {
        if(a.GetDim(rank-2)==a.GetDim(rank-1)) {r->SetDimNum(1);r->SetDim(0,0);}
        else {
            r->SetDimNum(rank-1);
            for(size_t i=0;i+2<rank;++i)r->SetDim(i,a.GetDim(i));
            r->SetDim(rank-2,b.GetDim(rank-1));
        }
    }
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
    c->SetOutputDataType(0,ge::DT_FLOAT);c->SetOutputDataType(1,ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class LinalgLstsq:public OpDef {
public:
    explicit LinalgLstsq(const char *name):OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("B").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("solution").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("residuals").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("driver").AttrType(OPTIONAL).String("gels");
        this->Attr("rcond").AttrType(OPTIONAL).Float(-1.0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgLstsq);
}
