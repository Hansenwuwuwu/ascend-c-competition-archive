#include <cstring>
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_det_tiling.h"
#include "../op_kernel/tiling_key_linalg_det.h"
namespace {
bool ParseShape(const gert::Shape &a,gert::Shape &out,LinalgDetTilingData &t) {
    const size_t rank=a.GetDimNum();if(rank<2)return false;
    int64_t n=a.GetDim(rank-1);if(n<1||n>512||a.GetDim(rank-2)!=n)return false;
    uint64_t batches=1;
    for(size_t i=0;i+2<rank;++i) {auto d=a.GetDim(i);if(d<1||d>256||batches>UINT32_MAX/(uint64_t)d)return false;batches*=d;}
    out=a;out.SetDimNum(rank-2);t.n=n;t.batches=batches;t.rows=8;while(t.rows<t.n)t.rows*=2;
    return true;
}
}
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *c) {
    auto *metadata=c->GetTilingData<LinalgDetMetadata>();auto *t=&metadata->base;gert::Shape out;
    if(!ParseShape(c->GetInputShape(0)->GetStorageShape(),out,*t))return ge::GRAPH_FAILED;
    if(c->GetInputDesc(0)->GetDataType()!=ge::DT_FLOAT)return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    const uint32_t cores=platform.GetCoreNumAiv();if(!cores)return ge::GRAPH_FAILED;
    t->blocks=t->batches<cores?t->batches:cores;
    uint64_t ubBytes=0;platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);
    const uint32_t padded=(t->n+7)/8*8,pitch=padded+8;
    
    uint32_t DT_A=ge::DT_FLOAT,KERNEL_PATH=1;
    const uint64_t reserve=14*padded+128;
    t->panel=(ubBytes/4-reserve)/(t->n+1)/8*8;
    if(t->panel>padded)t->panel=padded;
    if(t->panel<padded){uint32_t power=8;while(power*2<=t->panel)power*=2;t->panel=power;}
    uint64_t workspace=KERNEL_PATH?(uint64_t)t->blocks*t->n*padded*4:0;
    t->team=(t->n+63)/64;
    if(t->n<=64){
        t->rows=t->n<=16?16:t->n<=32?32:64;
        KERNEL_PATH=t->n<=16?3:t->n<=32?4:(padded-32>=24?7:5);
        if(KERNEL_PATH==4)for(uint32_t j=0;j<t->rows;++j){metadata->rowOffsets[j]=j*(t->rows+8)*4;metadata->diagonalOffsets[j]=j*(t->rows+9)*4;}
    }
    if(t->n>32 && t->n<=64 && t->blocks<8){
        const uint32_t grouped=(t->blocks+7)/8*8;
        t->blocks=grouped<cores?grouped:cores;
    }
    if(t->n<=16 && t->blocks<8)t->blocks=cores<8?cores:8;
    const uint32_t aic=platform.GetCoreNumAic();
    if(t->n>64 && aic>=t->team){
        uint32_t G=t->team*64,groups=aic/t->team;if(groups>t->batches)groups=t->batches;
        while(t->batches%groups)--groups;
        if(t->n%64!=0 || groups==t->batches){KERNEL_PATH=6;t->groups=groups;
        t->blocks=t->groups*t->team;
        workspace=uint64_t(t->groups)*(G*G+128*G)*4;c->SetScheduleMode(1);}
    }
    uint32_t ORDER=((t->n<=16 && t->batches<=cores) || (t->n>32 && t->n<=64))?t->n:0;
    if(KERNEL_PATH==6){
        ORDER=0;
        if(t->groups==t->batches){
            if(t->n%64==0)ORDER=1;
            else if(t->n%t->team==0){
                const uint32_t width=t->n/t->team;
                if(width<=56){
                    ORDER=width;
                }
            }
        }
    }
    ASCENDC_TPL_SEL_PARAM(c,DT_A,KERNEL_PATH,ORDER);c->SetBlockDim(t->blocks);
    if(KERNEL_PATH!=4)c->GetRawTilingData()->SetDataSize(sizeof(LinalgDetTilingData));
    c->GetWorkspaceSizes(1)[0]=workspace+platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {LinalgDetTilingData t{};return ParseShape(*c->GetInputShape(0),*c->GetOutputShape(0),t)?GRAPH_SUCCESS:GRAPH_FAILED;}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {c->SetOutputDataType(0,ge::DT_FLOAT);return GRAPH_SUCCESS;}
}
namespace ops {
class LinalgDet:public OpDef {
public:explicit LinalgDet(const char *name):OpDef(name) {
    this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Output("det").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
}};OP_ADD(LinalgDet);
}
