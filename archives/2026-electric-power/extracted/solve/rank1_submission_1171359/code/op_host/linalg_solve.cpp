#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_solve_tiling.h"
#include "../op_kernel/tiling_key_linalg_solve.h"
namespace {
template<class S> bool VectorRhs(const S& a, const S& b) {
    size_t ar=a.GetDimNum(), br=b.GetDimNum();
    if(br==1) return true;
    int64_t n=a.GetDim(ar-1);
    if(br==2 && b.GetDim(0)==n) return false;
    if(br==ar-1 && b.GetDim(br-1)==n) {
        bool same=true;
        for(size_t i=0;i+1<br;++i) same &= b.GetDim(i)==a.GetDim(i);
        if(same) return true;
    }
    return false;
}
}
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto platform=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto& a=context->GetInputShape(0)->GetStorageShape();
    const auto& b=context->GetInputShape(1)->GetStorageShape();
    uint32_t n=a.GetDim(a.GetDimNum()-1);
    uint32_t k=VectorRhs(a,b)?1:b.GetDim(b.GetDimNum()-1);
    uint64_t batches=1;
    for(size_t i=0;i+2<a.GetDimNum();++i) batches*=a.GetDim(i);
    auto* t=context->GetTilingData<LinalgSolveTilingData>();
    t->n=n; t->k=k; t->batches=batches;
    t->broadcastB=b.GetShapeSize()==int64_t(n)*k;
    uint32_t cores=platform.GetCoreNumAiv();
    t->cores=batches<cores?batches:cores;
    uint32_t width=((n+7)/8+(k+7)/8)*8;
    uint64_t ub=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    uint64_t extra=(n>64 && k<=8)?uint64_t(2*n*((k+7)/8*8)+k*((n+7)/8*8)+(n+7)/8*8)*4:0;
    t->resident=uint64_t(n)*width*4+32768+extra<=ub;
    t->grouped=1;
    uint32_t dt=static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
    bool columnSolve=dt==static_cast<uint32_t>(ge::DT_FLOAT) && n<=256 && k<=8;
    uint32_t columnRows=(n+7)/8*8;
    if(columnSolve) {
        uint64_t scratch=uint64_t((n>=224?36:28)*columnRows+8*n)*4+4096;
        uint32_t capacity=ub>scratch?(ub-scratch)/(uint64_t(columnRows)*4):0;
        t->grouped=capacity<n+k?n+k-capacity:0;
        t->resident=t->grouped==0;
    }
    bool paired=columnSolve && n>128 && batches*4<=cores;
    // One 16-row capacity bucket shares the existing three-RHS kernels.
    // n/k in tiling remain the actual GM extents; only private storage is padded.
    bool capacityFamily=paired && n>240 && n<=256 && k<=3 && cores/batches>=8;
    if(paired){
        uint32_t workers=(n+15)/16,available=cores/batches;
        if(workers>available)workers=available;
        if(n==256 || capacityFamily) {
            uint32_t cap=capacityFamily && available>=10?10:8;
            if(workers>cap)workers=cap;
        }
        t->cores=batches*workers;t->resident=3;t->grouped=0;context->SetScheduleMode(1);
    }
    uint32_t smallCap=n<=16 && k==1?8:16;
    if(columnSolve && n<=64 && t->cores<smallCap)t->cores=cores<smallCap?cores:smallCap;
    uint32_t nValue=0;
    switch(n) {case 1:case 2:case 4:case 8:case 15:case 16:case 32:case 64:case 128:case 256:case 512:nValue=n;break;default:break;}
    uint32_t taskMode=columnSolve && n<=16 && k==1?3:columnSolve && n<=64?2:paired?(n>=224?5:1):0;
    if(paired && n==256 && t->cores==batches*8)taskMode=6;
    if(capacityFamily) {
        taskMode=t->cores==batches*10?8:7;
        nValue=256;
    }
    // Runtime partial tiny matrices share the full16-column capacity.
    if(taskMode==3 && n>8 && n<16)nValue=15;
    // Hardware transpose accepts complete16-row blocks in each row capacity.
    if(taskMode==2)nValue=n%16==0?(n>32?64:32):0;
    // Bind metadata only for full row tiles or full A+RHS column tiles.
    // Partial tiles retain their actual dimensions in ordinary tiling data.
    bool fullTinyTile=n%16==0 || (n+k)%8==0;
    if(columnSolve && n>8 && n<=16 && k==1 && fullTinyTile && batches<=8 && t->cores==8) {
        taskMode=64+2*static_cast<uint32_t>(batches-1)+uint32_t(t->broadcastB);
        nValue=n;
    }
    ASCENDC_TPL_SEL_PARAM(context,dt,taskMode,nValue);
    context->SetBlockDim(t->cores);
    context->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+
        (capacityFamily?uint64_t(batches)*256*(512+(taskMode==8?41:40))*4:
         paired?uint64_t(batches)*columnRows*(taskMode==1?2*n+48:n==256?2*n+40:n+40)*4:columnSolve?uint64_t(t->cores)*columnRows*(n+k)*4:(t->resident?0:uint64_t(t->cores)*n*width*4));
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const auto& a=*context->GetInputShape(0);
    const auto& b=*context->GetInputShape(1);
    auto* x=context->GetOutputShape(0);
    bool vec=VectorRhs(a,b);
    x->SetDimNum(a.GetDimNum()-(vec?1:0));
    for(size_t i=0;i+2<a.GetDimNum();++i) x->SetDim(i,a.GetDim(i));
    x->SetDim(a.GetDimNum()-2,a.GetDim(a.GetDimNum()-1));
    if(!vec) x->SetDim(a.GetDimNum()-1,b.GetDim(b.GetDimNum()-1));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0,context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}
namespace ops {
class LinalgSolve:public OpDef {
public:
    explicit LinalgSolve(const char* name):OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Input("B").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("X").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Attr("left").AttrType(OPTIONAL).Bool(true);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgSolve);
}
