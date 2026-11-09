// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_pinv_tiling.h"
#include "../op_kernel/tiling_key_linalg_pinv.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const auto &shape = context->GetInputShape(0)->GetStorageShape();
        auto *t=context->GetTilingData<LinalgPinvTilingData>();
        t->m=shape.GetDim(shape.GetDimNum()-2);t->n=shape.GetDim(shape.GetDimNum()-1);
        t->p=t->m<t->n?t->m:t->n;t->q=t->m>t->n?t->m:t->n;
        t->pp=(t->p+7)/8*8;t->qp=(t->q+7)/8*8;
        t->batch=shape.GetShapeSize()/(t->m*t->n);
        t->cached=t->p*(t->qp+t->pp)<=32768;
        const auto *attrs=context->GetAttrs();
        const auto *at=attrs->GetFloat(0);const auto *rt=attrs->GetFloat(1);
        t->atol=at?*at:0.0f;t->rtol=rt?*rt:-1.0f;
        if(t->rtol<0)t->rtol=t->atol>0?0.0f:t->q*1.1920928955078125e-7f;
        uint32_t DT_A=static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
        if(t->m==t->n && t->p==199 && DT_A==static_cast<uint32_t>(ge::DT_FLOAT)){
            t->pp=200;t->qp=200;t->cached=t->p*(t->qp+t->pp)<=32768;
        }
        bool LARGE=!t->cached;uint32_t WIDTH=(!LARGE && t->m==t->n && t->p<=64)?t->p:0;
        uint32_t cores=platform.GetCoreNumAiv();t->group=1;t->groups=1;t->cols=2*t->pp;
        if(LARGE){
            uint32_t minGroup=1;
            while(((2*t->pp+minGroup*8-1)/(minGroup*8)*8)*(t->pp+8)>19000)minGroup++;
            for(uint32_t g=1;g<=t->batch && g<=cores/minGroup;g++)if(t->batch%g==0)t->groups=g;
            t->group=cores/t->groups;
            if(t->group>2*t->pp/8)t->group=2*t->pp/8;
            if(t->p==256 && t->m==t->n && t->batch==1 && DT_A==static_cast<uint32_t>(ge::DT_FLOAT) && t->group>16)t->group=16;
            t->cols=(2*t->pp+t->group*8-1)/(t->group*8)*8;
            if(t->pp==200 && (t->m==199 || t->m==200) && t->m==t->n && t->batch==1)t->group=(2*t->pp+t->cols-1)/t->cols;
            if(t->p==256 && t->m==t->n && DT_A==static_cast<uint32_t>(ge::DT_FLOAT)){
                if(t->group%2)t->group--;context->SetBlockDim(t->groups*t->group/2);
                matmul_tiling::MatmulApiTiling mm(platform);
                mm.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT);
                mm.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT);
                mm.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT);
                uint32_t singleM=t->group>=16?32:(t->group>=12?48:64);mm.SetShape(singleM,256,256);mm.SetOrgShape(256,256,256);mm.SetFixSplit(singleM,128,64);mm.SetBufferSpace(-1,-1,-1);
                auto *extended=context->GetTilingData<LinalgPinvCubeTilingData>();if(mm.GetTiling(extended->cube)<0)return ge::GRAPH_FAILED;
            }else context->SetBlockDim(t->groups*t->group);
        }else if(WIDTH>32 && DT_A==static_cast<uint32_t>(ge::DT_FLOAT) && t->batch*4<=cores){
            t->group=4;t->groups=t->batch;t->cols=(t->pp+31)/32*8;
            uint32_t target=t->batch*4<8?8:t->batch*4;
            context->SetBlockDim(target<cores?target:cores);
        }else{
            uint32_t target=(WIDTH>0 && DT_A==static_cast<uint32_t>(ge::DT_FLOAT) && t->batch<8)?8:t->batch;
            context->SetBlockDim(target<cores?target:cores);
        }
        if(!LARGE && WIDTH>=32 && t->group==4)WIDTH+=64;
        if(LARGE && t->p==256 && t->m==t->n && DT_A==static_cast<uint32_t>(ge::DT_FLOAT))WIDTH=2;
        if(LARGE && (t->p==199 || t->p==200) && t->m==t->n && DT_A==static_cast<uint32_t>(ge::DT_FLOAT))WIDTH=3;
        ASCENDC_TPL_SEL_PARAM(context,DT_A,LARGE,WIDTH);
        context->GetRawTilingData()->SetDataSize(WIDTH==2?sizeof(LinalgPinvCubeTilingData):sizeof(LinalgPinvTilingData));
        size_t oldSize=static_cast<size_t>(t->batch)*t->p*(2*t->qp+t->pp+8)*4+static_cast<size_t>(t->group*t->groups+t->groups+1)*32;
        size_t fastSize=static_cast<size_t>(t->groups)*2*t->cols*(t->pp+8)*4+static_cast<size_t>(t->batch)*t->pp*t->pp*4+static_cast<size_t>(t->group*t->groups+1)*32;
        context->GetWorkspaceSizes(1)[0]=(!LARGE && t->q<=64)?(t->group==4?platform.GetLibApiWorkSpaceSize()+t->batch*128+256:0):platform.GetLibApiWorkSpaceSize()+(oldSize>fastSize?oldSize:fastSize)+256+static_cast<size_t>(t->batch)*t->p*t->p*8+fastSize+static_cast<size_t>(t->batch)*t->pp*t->pp*8;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class LinalgPinv : public OpDef {
    public:
        explicit LinalgPinv(const char *name) : OpDef(name) {
            this->Input("A")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("A_pinv")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("atol").AttrType(OPTIONAL).Float();
            this->Attr("rtol").AttrType(OPTIONAL).Float(-1.0);
            this->Attr("hermitian").AttrType(OPTIONAL).Bool();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LinalgPinv);
}  // namespace ops
