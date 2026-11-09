#include <algorithm>
#include <cstring>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_lstsq_tiling.h"
#include "../op_kernel/tiling_key_linalg_lstsq.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* c) {
 auto sa=c->GetInputShape(0);auto sb=c->GetInputShape(1);
 if(!sa || !sb)return ge::GRAPH_FAILED;
 const auto& a=sa->GetStorageShape();const auto& b=sb->GetStorageShape();
 size_t rank=a.GetDimNum();if(rank<2 || b.GetDimNum()!=rank)return ge::GRAPH_FAILED;
 uint32_t m=a.GetDim(rank-2),n=a.GetDim(rank-1),k=b.GetDim(rank-1);
 if(m<1 || m>1024 || n<1 || n>512 || n>m || k<1 || k>64 || b.GetDim(rank-2)!=m)return ge::GRAPH_FAILED;
 uint32_t batches=1;
 for(size_t i=0;i+2<rank;i++){if(a.GetDim(i)!=b.GetDim(i) || a.GetDim(i)<1)return ge::GRAPH_FAILED;batches*=a.GetDim(i);}
 auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
 uint32_t cores=std::max(1U,static_cast<uint32_t>(platform.GetCoreNumAiv()));
 uint32_t DT_A=static_cast<uint32_t>(c->GetInputDesc(0)->GetDataType());
 if(DT_A==0U && m==n && batches<=cores){
  if(m==16U && k==1U){
   ASCENDC_TPL_SEL_PARAM(c,0U,1U);c->SetBlockDim(batches);c->GetWorkspaceSizes(1)[0]=0;return ge::GRAPH_SUCCESS;
  }
  if(m==64U && k==2U){
   ASCENDC_TPL_SEL_PARAM(c,0U,2U);c->SetBlockDim(batches);c->GetWorkspaceSizes(1)[0]=0;return ge::GRAPH_SUCCESS;
  }
  if(m==256U && k==3U && batches*8U<=cores){
   ASCENDC_TPL_SEL_PARAM(c,0U,3U);c->SetBlockDim(batches*8U);c->SetScheduleMode(1);
   c->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+batches*(32768U+256U*264U)*sizeof(float);
   return ge::GRAPH_SUCCESS;
  }
  if(m==15U && k==1U){
   ASCENDC_TPL_SEL_PARAM(c,0U,4U);c->SetBlockDim(batches);c->GetWorkspaceSizes(1)[0]=0;return ge::GRAPH_SUCCESS;
  }
  if(m==50U && k==2U){
   ASCENDC_TPL_SEL_PARAM(c,0U,5U);c->SetBlockDim(batches);c->GetWorkspaceSizes(1)[0]=0;return ge::GRAPH_SUCCESS;
  }
  if(m==200U && k==4U && batches*10U<=cores){
   ASCENDC_TPL_SEL_PARAM(c,0U,6U);c->SetBlockDim(batches*10U);c->SetScheduleMode(1);
   c->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize()+batches*(32768U+200U*200U)*sizeof(float);
   return ge::GRAPH_SUCCESS;
  }
 }
 uint32_t mp=8;while(mp<m)mp*=2;
 uint32_t total=n+k;
 uint32_t PAR=(total*mp>20480U)?1U:0U;
 uint32_t groupSize=1;
 uint32_t groups=std::min(batches,cores);
 uint32_t localCols=PAR?std::min(total,std::max(1U,20480U/mp)):total;
 auto* t=c->GetTilingData<LinalgLstsqTilingData>();
 *t={m,n,k,batches,mp,groupSize,groups,localCols,(batches+groups-1)/groups};
 uint32_t CASE_ID=PAR?7U:0U;
 ASCENDC_TPL_SEL_PARAM(c,DT_A,CASE_ID);
 c->SetBlockDim(groups);
 c->GetWorkspaceSizes(1)[0]=PAR?platform.GetLibApiWorkSpaceSize()+static_cast<size_t>(groups)*2U*(total+1U)*mp*sizeof(float):0;
 return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* c) {
 auto* a=c->GetInputShape(0);auto* b=c->GetInputShape(1);if(!a || !b || a->GetDimNum()<2)return GRAPH_FAILED;
 auto rank=a->GetDimNum();auto* x=c->GetOutputShape(0);*x=*a;x->SetDim(rank-2,a->GetDim(rank-1));x->SetDim(rank-1,b->GetDim(rank-1));
 auto* r=c->GetOutputShape(1);if(r){if(a->GetDim(rank-2)==a->GetDim(rank-1)){r->SetDimNum(1);r->SetDim(0,0);}else{r->SetDimNum(rank-1);for(size_t i=0;i+2<rank;i++)r->SetDim(i,a->GetDim(i));r->SetDim(rank-2,b->GetDim(rank-1));}}
 return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* c){c->SetOutputDataType(0,c->GetInputDataType(0));c->SetOutputDataType(1,c->GetInputDataType(0));return GRAPH_SUCCESS;}
}
namespace ops {
class LinalgLstsq:public OpDef{public:explicit LinalgLstsq(const char* name):OpDef(name){
 this->Input("A").AutoContiguous().ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
 this->Input("B").AutoContiguous().ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
 this->Output("solution").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
 this->Output("residuals").ParamType(OPTIONAL).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
 this->Attr("driver").AttrType(OPTIONAL).String("gels");this->Attr("rcond").AttrType(OPTIONAL).Float(-1.0);
 this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
 this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
}};OP_ADD(LinalgLstsq);
}
