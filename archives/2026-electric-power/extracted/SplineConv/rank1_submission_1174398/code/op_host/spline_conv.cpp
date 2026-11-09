#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/spline_conv_tiling.h"
#include "../op_kernel/tiling_key_spline_conv.h"
#include <algorithm>
#include <cstring>

namespace optiling {
static ge::graphStatus Tile(gert::TilingContext *c, bool full) {
    const auto *x = c->GetRequiredInputShape(0);
    const auto *edges = c->GetRequiredInputShape(1);
    const auto *p = c->GetRequiredInputShape(2);
    const auto *w = c->GetRequiredInputShape(3);
    const auto *xd=c->GetRequiredInputDesc(0), *ed=c->GetRequiredInputDesc(1);
    const auto *pd=c->GetRequiredInputDesc(2), *wd=c->GetRequiredInputDesc(3);
    if (!x || !edges || !p || !w || !xd || !ed || !pd || !wd) return ge::GRAPH_FAILED;
    const auto &xs = x->GetOriginShape();
    const auto &es = edges->GetOriginShape();
    const auto &ps = p->GetOriginShape();
    const auto &ws = w->GetOriginShape();
    if (xs.GetDimNum()!=2 || es.GetDimNum()!=2 || ps.GetDimNum()!=2 || ws.GetDimNum()!=3) return ge::GRAPH_FAILED;
    auto dt=xd->GetDataType();
    if ((dt!=ge::DT_FLOAT && dt!=ge::DT_BF16) || ed->GetDataType()!=ge::DT_INT64 || pd->GetDataType()!=dt || wd->GetDataType()!=dt) return ge::GRAPH_FAILED;
    const auto *a=c->GetAttrs();
    const auto *dim=a->GetInt(0); const auto *ks=a->GetListInt(1);
    const auto *open=a->GetListInt(2); const auto *degree=a->GetInt(3);
    const char *aggr=a->GetStr(4);
    if (!aggr || !aggr[0]) aggr="mean";
    if (!std::strcmp(aggr,"sum")) aggr="add";
    bool defaultOpen=!open || open->GetSize()==0 || (open->GetSize()==1 && open->GetData()[0]==-1);
    if (!dim || !ks || !degree || !aggr) return ge::GRAPH_FAILED;
    if (*dim<1 || *dim>8 || *degree<0 || *degree>3 || ks->GetSize()<1 || (!defaultOpen && open->GetSize()<1)) return ge::GRAPH_FAILED;
    if (std::strcmp(aggr,"mean") && std::strcmp(aggr,"add")) return ge::GRAPH_FAILED;
    bool edgeTrans=es.GetDim(0)!=2 && es.GetDim(1)==2;
    int64_t n=xs.GetDim(0), ci=xs.GetDim(1), e=es.GetDim(edgeTrans?0:1);
    bool weightTrans=ws.GetDim(1)!=ci && ws.GetDim(2)==ci;
    int64_t co=ws.GetDim(weightTrans?1:2);
    bool pseudoTrans=!(ps.GetDim(0)==e && ps.GetDim(1)==*dim) && ps.GetDim(0)==*dim && ps.GetDim(1)==e;
    if (n<1 || n>65536 || ci<0 || ci>65535 || co<1 || co>65535 || e<0 || e>262144) return ge::GRAPH_FAILED;
    if ((es.GetDim(0)!=2 && !edgeTrans) || (!pseudoTrans && (ps.GetDim(0)!=e || ps.GetDim(1)!=*dim)) || (!weightTrans && ws.GetDim(1)!=ci)) return ge::GRAPH_FAILED;
    auto *t=c->GetTilingData<SplineConvTilingData>();
    *t={}; t->n=n; t->e=e; t->ci=ci; t->co=co; t->dim=*dim; t->degree=*degree;
    t->edgeTrans=edgeTrans; t->pseudoTrans=pseudoTrans;
    t->weightTrans=weightTrans;
    t->slots=1; t->kernels=1;
    for (uint32_t d=0; d<t->dim; ++d) {
        int64_t k=ks->GetData()[std::min<uint32_t>(d,ks->GetSize()-1)], o=defaultOpen?1:int64_t(open->GetData()[std::min<uint32_t>(d,open->GetSize()-1)]!=0);
        if (k<1 || k>65535 || t->kernels>65535/k) return ge::GRAPH_FAILED;
        t->ks[d]=k; t->scale[d]=k-*degree*o; t->kernels*=k; t->slots*=*degree+1;
    }
    if (ws.GetDim(0)<t->kernels || (ci>0 && co>0 && uint64_t(e)*t->slots>(uint64_t(1)<<31)/(uint64_t(ci)*co))) return ge::GRAPH_FAILED;
    if (full) {
        const auto *r=c->GetOptionalInputShape(4), *b=c->GetOptionalInputShape(5);
        const auto *rd=c->GetOptionalInputDesc(4), *bd=c->GetOptionalInputDesc(5);
        if (r) { const auto &s=r->GetOriginShape(); if (!rd || rd->GetDataType()!=dt || s.GetDimNum()!=2 || s.GetDim(0)!=ci || s.GetDim(1)!=co) return ge::GRAPH_FAILED; t->hasRoot=1; }
        if (b) { const auto &s=b->GetOriginShape(); if (!bd || bd->GetDataType()!=dt || s.GetDimNum()!=1 || s.GetDim(0)!=co) return ge::GRAPH_FAILED; t->hasBias=1; }
    }
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    t->blocks=std::min<uint32_t>(platform.GetCoreNumAiv(), n);
    uint32_t sparseAlign=dt==ge::DT_BF16?16:8;
    if(!edgeTrans && !pseudoTrans && !weightTrans && !t->hasRoot && !t->hasBias &&
       e>0 && e<n && e<=64 && co==16 && ci==32 && ci%sparseAlign==0 &&
       *dim==2 && *degree==1 && t->ks[0]==2 && t->ks[1]==2) {
        uint32_t proposed=std::min<uint32_t>(t->blocks,std::max<uint32_t>(8,e));
        uint32_t rows=(n+proposed-1)/proposed;
        if(rows<=ci && uint64_t(rows)*e<=512) t->blocks=proposed;
    }
    if(!edgeTrans && !pseudoTrans && !weightTrans && !t->hasRoot && !t->hasBias &&
       e>0 && e<n && e<=64 && ci==128 && co==16 && *dim==2 && *degree==1 &&
       t->ks[0]==2 && t->ks[1]==2) {
        uint32_t rowBound=(n+127)/128;
        uint32_t scanBound=uint32_t((uint64_t(n)*e+511)/512);
        uint32_t useful=std::max<uint32_t>({8,uint32_t(e),rowBound,scanBound});
        t->blocks=std::min(t->blocks,useful);
    }
    bool chunkSingle=!edgeTrans && !pseudoTrans && !weightTrans && !t->hasRoot && !t->hasBias &&
        ci==8 && co==16 && e==1 && n<=64 && n%16==0 && *dim==2 && *degree==1 && t->ks[0]==2 && t->ks[1]==2 &&
        t->blocks==uint32_t(n);
    bool tailSingle=!edgeTrans && !pseudoTrans && !weightTrans && !t->hasRoot && !t->hasBias &&
        ci==8 && co==16 && e==1 && n<=64 && n%(dt==ge::DT_BF16?4:2)!=0 &&
        *dim==2 && *degree==1 && t->ks[0]==2 && t->ks[1]==2 &&
        t->blocks==uint32_t(n) && uint32_t(n)<platform.GetCoreNumAiv();
    if(tailSingle) ++t->blocks;
    t->nodesPerBlock=(n+t->blocks-1)/t->blocks;
    uint32_t align=dt==ge::DT_BF16?16:8; t->ciPad=(ci+align-1)/align*align; t->coPad=(co+align-1)/align*align; t->mean=!std::strcmp(aggr,"mean");
    uint64_t ubSize=0; platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubSize);
    uint64_t elementBytes=dt==ge::DT_BF16?2:4;
    uint64_t matrixElements=uint64_t(ci)*t->coPad, cacheElements=uint64_t(t->kernels)*matrixElements;
    uint64_t baseBytes=uint64_t((t->nodesPerBlock+7)/8*8)*4+uint64_t(t->kernels)*t->ciPad*8+
        uint64_t(t->ciPad)*12+matrixElements*(elementBytes+(dt==ge::DT_BF16?4:0))+
        uint64_t(t->ciPad+t->coPad)*(elementBytes+4);
    if (t->kernels>64) baseBytes+=uint64_t((t->kernels+7)/8*8)*4;
    if (ubSize<16384) return ge::GRAPH_FAILED;
    t->streaming=edgeTrans || pseudoTrans || weightTrans || ci==0 || baseBytes>ubSize-16384;
    uint64_t streamBytes=uint64_t((t->nodesPerBlock+7)/8*8)*4+128*(24+2*elementBytes);
    if (t->streaming && streamBytes>ubSize-16384) return ge::GRAPH_FAILED;
    t->cacheWeights=!t->streaming && cacheElements*4<=65536 && baseBytes+cacheElements*(elementBytes+(dt==ge::DT_BF16?4:0))<=ubSize-16384;
    uint64_t graphBytes=uint64_t((2*e+3)/4*4)*8+uint64_t((e*(*dim)+15)/16*16)*elementBytes+uint64_t((e+7)/8*8)*4;
    uint64_t usedBytes=baseBytes+(t->cacheWeights?cacheElements*(elementBytes+(dt==ge::DT_BF16?4:0)):0);
    t->cacheGraph=!t->streaming && e>0 && usedBytes+graphBytes<=ubSize-16384;
    usedBytes+=t->cacheGraph?graphBytes:0;
    uint64_t dotBytes=uint64_t(t->ciPad)*t->coPad*8;
    t->vectorDot=!t->streaming && ci<=128 && t->coPad<=128 && usedBytes+dotBytes<=ubSize-16384;
    uint64_t fusedBytes=uint64_t(t->kernels)*t->ciPad*(t->coPad+8)*4;
    t->fusedDot=t->cacheWeights && ci==t->ciPad && t->coPad<=128 && uint64_t(t->kernels)*t->ciPad<=2040 && usedBytes+fusedBytes<=ubSize-16384;
    uint64_t smallBytes=uint64_t(n)*8*(elementBytes+(dt==ge::DT_BF16?4:0))+uint64_t(e)*(16+2*elementBytes)+4096;
    t->smallScalar=!t->streaming && ci==8 && co==8 && *dim==2 && *degree==1 && t->ks[0]==2 && t->ks[1]==2 &&
        uint64_t(t->nodesPerBlock)*e<=512 && smallBytes<=ubSize-16384;
    uint32_t single=!t->streaming && !edgeTrans && !pseudoTrans && !weightTrans &&
        !t->hasRoot && !t->hasBias && ci==8 && co==16 && e==1 && t->nodesPerBlock<=8 &&
        *dim==2 && *degree==1 && t->ks[0]==2 && t->ks[1]==2;
    uint32_t sparse=!t->streaming && !edgeTrans && !pseudoTrans && !weightTrans &&
        !t->hasRoot && !t->hasBias && t->cacheGraph && t->cacheWeights && t->fusedDot &&
        (ci==32 || ci==128) && ci==t->ciPad && co==16 && e>0 && e<n && e<=64 && t->nodesPerBlock<=ci &&
        uint64_t(t->nodesPerBlock)*e<=512 && *dim==2 && *degree==1 && t->ks[0]==2 && t->ks[1]==2;
    if(sparse && ci==128) sparse=2;
    if(sparse==1 && dt==ge::DT_FLOAT && n<=64 && e<=t->blocks) sparse=3;
    if(sparse==2 && dt==ge::DT_FLOAT && e==t->blocks && t->nodesPerBlock<=32) sparse=4;
    if(single && tailSingle) single=4;
    else if(single && chunkSingle) single=3;
    else if(single && t->nodesPerBlock==1) single=2;
    if(dt==ge::DT_FLOAT && (single==3 || single==4)) t->blocks/=2;
    uint32_t NODEPAD=0;
    if(dt==ge::DT_FLOAT && single==4 && n>1) single=5;
    if(single==5 && n>=3 && n<=31 && n%2!=0) {single=6;NODEPAD=(31-uint32_t(n))/2;}
    if(sparse==4) sparse=8;
    if(sparse==4 && platform.GetCoreNumAiv()>uint32_t(e)) {
        uint32_t clearing=(uint32_t(n)+255)/256;
        if(clearing<=16 && clearing<=platform.GetCoreNumAiv()-uint32_t(e)) {
            sparse=7;NODEPAD=clearing-1;t->blocks=uint32_t(e)+clearing;
        }
    }
    if(sparse==3 && e==4) {
        uint32_t groups=(n+15)/16,needed=4+groups;
        if(platform.GetCoreNumAiv()>=needed) {
            if(groups==4 && n%16!=0) {sparse=5;NODEPAD=64-uint32_t(n);t->blocks=8;}
            else {sparse=6;NODEPAD=groups*16-n;t->blocks=needed;}
        }
    }
    uint32_t closed0=(single>=2 || sparse) && t->scale[0]==2,closed1=(single>=2 || sparse) && t->scale[1]==2;
    uint32_t DT_X=static_cast<uint32_t>(dt); ASCENDC_TPL_SEL_PARAM(c, DT_X, single, sparse, closed0, closed1, NODEPAD);
    c->SetBlockDim(t->blocks);
    if((single==3 || single==5 || single==6) && n<=32 && t->blocks>1) c->SetScheduleMode(1);
    c->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+uint64_t(t->blocks)*((e+15)/16*16)*4;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingFunc(gert::TilingContext *c) { return Tile(c,false); }
static ge::graphStatus FullTilingFunc(gert::TilingContext *c) { return Tile(c,true); }
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
    auto *out=c->GetOutputShape(0); const auto *x=c->GetInputShape(0), *w=c->GetInputShape(3);
    if (!out || !x || !w || x->GetDimNum()!=2 || w->GetDimNum()!=3) return GRAPH_FAILED;
    bool trans=w->GetDim(1)!=x->GetDim(1) && w->GetDim(2)==x->GetDim(1);
    out->SetDimNum(2); out->SetDim(0,x->GetDim(0)); out->SetDim(1,w->GetDim(trans?1:2)); return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) { c->SetOutputDataType(0,c->GetInputDataType(0)); return GRAPH_SUCCESS; }
}
namespace ops {
class SplineConv : public OpDef {
public:
    explicit SplineConv(const char *name):OpDef(name) {
        Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Input("edge_index").ParamType(REQUIRED).DataType({ge::DT_INT64,ge::DT_INT64}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Input("edge_attr").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Attr("dim").AttrType(OPTIONAL).Int(2);
        Attr("kernel_size").AttrType(OPTIONAL).ListInt({2,2});
        Attr("is_open_spline").AttrType(OPTIONAL).ListInt({-1});
        Attr("degree").AttrType(OPTIONAL).Int(1); Attr("aggr").AttrType(OPTIONAL).String("mean");
        SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SplineConv);
class SplineConvFull : public OpDef {
public:
    explicit SplineConvFull(const char *name):OpDef(name) {
        Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Input("edge_index").ParamType(REQUIRED).DataType({ge::DT_INT64,ge::DT_INT64}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Input("edge_attr").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Input("root_weight").ParamType(OPTIONAL).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Input("bias").ParamType(OPTIONAL).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        Attr("dim").AttrType(OPTIONAL).Int(2); Attr("kernel_size").AttrType(OPTIONAL).ListInt({2,2});
        Attr("is_open_spline").AttrType(OPTIONAL).ListInt({-1}); Attr("degree").AttrType(OPTIONAL).Int(1); Attr("aggr").AttrType(OPTIONAL).String("mean");
        SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        OpAICoreConfig config; config.ExtendCfgInfo("opFile.value","spline_conv").ExtendCfgInfo("opInterface.value","spline_conv_full");
        AICore().SetTiling(optiling::FullTilingFunc).AddConfig("ascend910b",config);
    }
};
OP_ADD(SplineConvFull);
}
