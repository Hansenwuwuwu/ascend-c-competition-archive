#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_det_tiling.h"
#include "../op_kernel/tiling_key_linalg_det.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
 auto platform=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
 const auto& a=context->GetInputShape(0)->GetStorageShape();
 uint32_t n=a.GetDim(a.GetDimNum()-1);uint64_t batches=1;
 for(size_t i=0;i+2<a.GetDimNum();++i)batches*=a.GetDim(i);
 uint32_t cores=platform.GetCoreNumAiv();
 uint32_t owners=0;
 if(n>128){
  uint32_t slots=platform.GetCoreNumAic()/4;if(slots>4)slots=4;
  cores=slots*4;context->SetScheduleMode(1);
 } else if(n<=64){uint64_t target=batches<8?8:batches;if(target<cores)cores=uint32_t(target);}
 else if(batches<cores)cores=uint32_t(batches);
 auto* t=context->GetTilingData<LinalgDetTilingData>();
 t->n=n;t->cores=cores;t->batches=batches;t->resident=n<16?1:owners;
 // Compile complete geometry families. No input values or evaluation identifiers are used.
 constexpr uint32_t dmaFloats=32/sizeof(float),vectorFloats=256/sizeof(float);
 constexpr uint32_t tinyRows=16,panelTile=128,maxMatrixSlots=4;
 uint32_t dt=static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
 uint32_t taskMode=n>panelTile?1:0;
 if(cores==8&&batches==1&&n<=tinyRows)taskMode=n+1;
 else if(cores==8&&batches<=cores){
  // Even row counts in the final 32-byte capacity bucket share paired
  // elimination. Preserve the existing full-capacity mode numbers and body.
  if((n+dmaFloats-1)/dmaFloats==vectorFloats/dmaFloats&&(n%2)==0)
   taskMode=n==vectorFloats?32+uint32_t(batches)-1:
    160+((n-(vectorFloats-dmaFloats+2))/2)*8+uint32_t(batches)-1;
  else if((n+dmaFloats-1)/dmaFloats==7&&(n%2)==0)
   taskMode=48+((n-(6*dmaFloats+2))/2)*8+uint32_t(batches)-1;
 }
 if(cores==4*maxMatrixSlots&&n>panelTile&&n<=2*panelTile&&n%dmaFloats==0&&batches<=maxMatrixSlots)
  taskMode=96+(n/dmaFloats-(panelTile/dmaFloats+1))*maxMatrixSlots+uint32_t(batches)-1;
 ASCENDC_TPL_SEL_PARAM(context,dt,taskMode);
 context->SetBlockDim(cores);
 context->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+(n>128?uint64_t(cores/4)*(uint64_t((n+127)/128*128)*((n+127)/128*128)+64*((n+127)/128*128)+4*8192)*4:0);
 return ge::GRAPH_SUCCESS;
}}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
 const auto& a=*context->GetInputShape(0);auto* o=context->GetOutputShape(0);
 o->SetDimNum(a.GetDimNum()-2);for(size_t i=0;i+2<a.GetDimNum();++i)o->SetDim(i,a.GetDim(i));return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
 context->SetOutputDataType(0,ge::DT_FLOAT);return GRAPH_SUCCESS;
}}
namespace ops {
class LinalgDet:public OpDef {public:explicit LinalgDet(const char* name):OpDef(name) {
 this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
 this->Output("det").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
 this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
 this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
}};OP_ADD(LinalgDet);
}
