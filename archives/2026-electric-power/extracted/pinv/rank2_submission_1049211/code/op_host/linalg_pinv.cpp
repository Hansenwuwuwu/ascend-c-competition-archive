#include <cstring>
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_pinv_tiling.h"
#include "../op_kernel/tiling_key_linalg_pinv.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *c) {
    const auto &shape = c->GetInputShape(0)->GetStorageShape();
    const auto rank = shape.GetDimNum();
    if (rank < 2) return ge::GRAPH_FAILED;
    const int64_t m = shape.GetDim(rank-2), n = shape.GetDim(rank-1);
    if (m < 1 || n < 1 || m > 512 || n > 512) return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    uint32_t cores=platform.GetCoreNumAiv();
    uint32_t DT_A=c->GetInputDesc(0)->GetDataType();
    uint32_t batches=shape.GetShapeSize()/(m*n),cubeCores=platform.GetCoreNumAic();
    bool useCube=DT_A==ge::DT_FLOAT && m>PINV_ROW_BLOCK_PANEL && m<=PINV_ROW_BLOCK_CAPACITY && n==m && batches>0 && batches<=cubeCores/PINV_CUBE_CORES_PER_BATCH && cores==2*cubeCores && cores<=64;
    auto *cube=useCube?c->GetTilingData<LinalgPinvCubeTilingData>():nullptr;
    if(useCube && cube==nullptr)return ge::GRAPH_FAILED;
    bool tinyPath=DT_A==ge::DT_FLOAT && m==n && m>=PINV_TINY_MIN_ORDER && m<=PINV_TINY_CAPACITY;
    auto *tiny=tinyPath?c->GetTilingData<LinalgPinvTinyTilingData>():nullptr;
    if(tinyPath && !tiny)return ge::GRAPH_FAILED;
    auto *t=useCube?&cube->base:tinyPath?&tiny->base:c->GetTilingData<LinalgPinvTilingData>();
    if(tinyPath) {
        std::memset(tiny->padding,0,sizeof(tiny->padding));
        auto encode=[](uint32_t x)->uint16_t {if(!x)return 0;float f=(float)x;uint32_t bits;std::memcpy(&bits,&f,4);return (uint16_t)((((bits>>23)-112)<<10)|((bits>>13)&1023));};
        for(uint32_t row=0;row<16;++row) {
            for(uint32_t col=0;col<8;++col) {
                auto inputOffset=[&](uint32_t c){return (row<(uint32_t)m && c<(uint32_t)m?row*m+c:m*m)*4;};
                tiny->packed[row*32+col]=(uint32_t)encode(inputOffset(2*col))|((uint32_t)encode(inputOffset(2*col+1))<<16);
                tiny->packed[row*32+8+col]=(uint32_t)encode((2*col*32+row)*4)|((uint32_t)encode(((2*col+1)*32+row)*4)<<16);
            }
            for(uint32_t col=0;col<16;++col)tiny->packed[row*32+16+col]=row==col?0x3f800000u:0u;
        }
    }
    if(t==nullptr)return ge::GRAPH_FAILED;
    t->m=m; t->n=n; t->batches=shape.GetShapeSize()/(m*n);
    t->columns=m<n?m:n; t->rows=m>n?m:n;
    t->stride=((t->rows+7)/8+(t->columns+7)/8)*8;
    // At most eight column owners per batch; use wider panels when batches
    // consume the available cores, up to the eight-column UB work tile.
    uint32_t smallBudget=batches?cores/batches:0;
    if(smallBudget>PINV_COLUMN_MAX_SHARDS)smallBudget=PINV_COLUMN_MAX_SHARDS;
    uint32_t smallWidth=smallBudget?(n+smallBudget-1)/smallBudget:0;
    uint32_t smallShards=smallWidth?(n+smallWidth-1)/smallWidth:0;
    bool smallTeam=m==n && DT_A==ge::DT_FLOAT && n>PINV_TINY_CAPACITY && n<=PINV_COLUMN_CAPACITY && smallWidth>0 && smallWidth<=PINV_COLUMN_MAX_WIDTH;
    t->parallel=smallTeam?2:(t->columns>=64 && !(m==n && m<=64));
    t->resident=t->parallel!=1 && t->columns*t->stride*4 <= 128*1024;
    const auto *attrs=c->GetAttrs();
    t->atol=*attrs->GetFloat(0); t->rtol=*attrs->GetFloat(1);
    if (t->rtol < 0) t->rtol=t->atol>0 ? 0.0f : (float)(m>n?m:n)*1.1920928955078125e-7f;
    if (*attrs->GetBool(2)) return ge::GRAPH_FAILED;
    uint32_t KERNEL_PATH=t->parallel?1:0;
    if(smallTeam)KERNEL_PATH=6;
    if(tinyPath)KERNEL_PATH=5;
    if(useCube) {
        KERNEL_PATH=4;
        cube->systemWorkspaceBytes=platform.GetLibApiWorkSpaceSize();
    }
    ASCENDC_TPL_SEL_PARAM(c,DT_A,KERNEL_PATH);
    uint32_t launchBlocks=useCube ? (cubeCores>PINV_CUBE_CORES_PER_BATCH*t->batches?PINV_CUBE_CORES_PER_BATCH*t->batches:cubeCores) : smallTeam ? t->batches*smallShards : (t->parallel ? cores : (t->batches<cores?t->batches:cores));
    if(DT_A==ge::DT_FLOAT && m==n && m<=16 && launchBlocks<PINV_TINY_MIN_BLOCKS)
        launchBlocks=cores<PINV_TINY_MIN_BLOCKS?cores:PINV_TINY_MIN_BLOCKS;
    c->SetBlockDim(launchBlocks);
    if(t->parallel) c->SetScheduleMode(1);
    uint64_t userWorkspace=smallTeam?((uint64_t)t->batches*smallShards*(smallWidth*((n+7)/8*8)+8))*4:(t->resident?0:(uint64_t)t->batches*t->columns*(t->stride+(t->parallel?8:0))*4)
        + ((t->parallel && m==n)?(uint64_t)t->batches*m*n*4:0);
    if(useCube) {
        const uint64_t blockWorkspace=(uint64_t)t->batches*PINV_ROW_BLOCK_WORKSPACE_FLOATS*sizeof(float);
        if(userWorkspace<blockWorkspace)userWorkspace=blockWorkspace;
    }
    // Reserve the framework workspace as well as the user buffers.
    c->GetWorkspaceSizes(1)[0]=userWorkspace ? userWorkspace+platform.GetLibApiWorkSpaceSize() : 0;
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
    auto shape=*c->GetInputShape(0);
    auto rank=shape.GetDimNum();
    if(rank<2) return GRAPH_FAILED;
    auto m=shape.GetDim(rank-2), n=shape.GetDim(rank-1);
    shape.SetDim(rank-2,n); shape.SetDim(rank-1,m);
    *c->GetOutputShape(0)=shape;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
    c->SetOutputDataType(0,c->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}
namespace ops {
class LinalgPinv:public OpDef {
public:
    explicit LinalgPinv(const char *name):OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("A_pinv").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Attr("atol").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("rtol").AttrType(OPTIONAL).Float(-1.0f);
        this->Attr("hermitian").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgPinv);
}
