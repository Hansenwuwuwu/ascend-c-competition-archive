#include <cstring>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/safety_belt_violation_geometry_tiling.h"
#include "../op_kernel/tiling_key_safety_belt_violation_geometry.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
 const auto *boxes=context->GetInputShape(0);
 const auto *anchors=context->GetInputShape(1);
 if (!boxes || !anchors) return ge::GRAPH_FAILED;
 const auto &b=boxes->GetStorageShape(); const auto &a=anchors->GetStorageShape();
 if(b.GetDimNum()!=2 || a.GetDimNum()!=2 || b.GetDim(1)!=4 || a.GetDim(1)!=2 || b.GetDim(0)!=a.GetDim(0) || b.GetDim(0)<1 || b.GetDim(0)>kGeometryMaxBatch) return ge::GRAPH_FAILED;
 auto *t=context->GetTilingData<SafetyBeltViolationGeometryTilingData>();
 t->batch=b.GetDim(0); t->waistRatio=*context->GetAttrs()->GetFloat(0);
 t->hasThreshold=context->GetOptionalInputShape(2)!=nullptr;
 t->thresholdIsHalf=t->hasThreshold && context->GetInputDesc(2)->GetDataType()==ge::DT_FLOAT16;
 t->defaultThreshold=0.0f;
 auto platform=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
 uint32_t tile=256;
 if(t->batch<=512 && (t->batch+63)/64<=platform.GetCoreNumAiv())tile=64;
 // Wide tiles require enough owners and more than half occupancy in the tail.
 const uint32_t wideWorkers=(t->batch+511)/512;
 const uint32_t wideTail=t->batch%512;
 const bool halfInput=context->GetInputDesc(0)->GetDataType()==ge::DT_FLOAT16;
 if((halfInput && wideWorkers>=4 && (wideTail==0 || wideTail>256)) ||
    (!halfInput && wideWorkers>=kGeometryFloatWideOwners && (wideTail==0 || wideTail>256)))tile=512;
 if(!halfInput && t->batch<=64 && platform.GetCoreNumAiv()>=2)tile=32;
 // Attribute identity is known to tiling; tensor values remain device inputs.
 const bool floatUnitTiles=!halfInput && t->batch<=64 &&
                           context->GetOutputDesc(0)->GetDataType()==ge::DT_FLOAT;
 const bool identityTiling=(halfInput || floatUnitTiles) && t->waistRatio==1.0f && t->batch%32==0 &&
                           t->batch/32<=platform.GetCoreNumAiv();
 if(identityTiling)tile=32;
 uint32_t cores=(t->batch+tile-1)/tile;
 if(cores>platform.GetCoreNumAiv()) cores=platform.GetCoreNumAiv();
 t->perCore=tile;
 if((t->batch+tile-1)/tile>cores)t->perCore=((t->batch+cores-1)/cores+15)/16*16;
 const uint32_t workers=(t->batch+t->perCore-1)/t->perCore;
 uint32_t launchCores=workers;
 if(tile==64 || tile==32){
  launchCores=8;
  if(t->batch>64 && t->batch%64==0)launchCores=workers;
  if(launchCores<workers)launchCores=workers;
  if(launchCores>platform.GetCoreNumAiv())launchCores=platform.GetCoreNumAiv();
 }
 if(identityTiling){
  launchCores=workers<8?8:workers;
  if(launchCores>platform.GetCoreNumAiv())launchCores=platform.GetCoreNumAiv();
 }
 context->SetBlockDim(launchCores);
 uint32_t dtype=static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
 uint32_t outdtype=static_cast<uint32_t>(context->GetOutputDesc(0)->GetDataType());
 uint32_t thresholdKind=t->hasThreshold?(t->thresholdIsHalf?2:1):0;
 // Native element types retain the static kernels. Mixed interfaces share
 // the existing typed device loop instead of duplicating every resource family.
 const bool nativeSignature=dtype==outdtype &&
     ((!halfInput && thresholdKind==1U) || (halfInput && thresholdKind==2U));
 const uint32_t full64=t->batch/64U;
 const uint32_t short64=t->batch%64U;
 const bool runtimeFold=!halfInput && nativeSignature && t->waistRatio!=0.5f &&
     tile==64U && t->perCore==64U && full64>=1U && full64<=7U &&
     short64>=1U && short64<=8U && platform.GetCoreNumAiv()>=8U;
 // Keys classify tile width, full/tail status and worker count; lengths are runtime data.
 uint32_t tileClass=tile==32?23+2*(workers-1)+(t->batch%32!=0):tile==512 && t->perCore==512?(t->batch%512==0?19:20):tile==64?3+2*(workers-1)+(t->batch%64!=0):t->perCore>256?0:(t->batch%256==0?1:2);
 // Complete64 multi-owner entries have identical device work: every
 // launched core owns64 rows. Keep the actual launch count, share the key.
 if(tile==64 && t->perCore==64 && t->batch%64==0 && workers>1)tileClass=5;
 if(identityTiling)tileClass=workers<=8?26+workers:35;
 // Constant near-full tails fit in the final output DMA block.
 const uint32_t outputBlockRows=halfInput?16U:8U;
 if((!halfInput || dtype==outdtype) && tile==512 && t->perCore==512 &&
    wideTail>512U-outputBlockRows)
  tileClass=36+wideTail-449;
 // Fold short tails. Float uses an exact midpoint attribute; Half retains
 // its runtime multiplier. Other attributes and tail sizes keep prior paths.
 if(tile==64 && t->perCore==64 && workers>1 && t->batch%64>0 && t->batch%64<=8 &&
    (halfInput || t->waistRatio==0.5f || runtimeFold))
  tileClass=99+(workers-2)*8+(t->batch%64-1);
 // Matching Float folded cohorts use their actual owner budget.
 if(!halfInput && dtype==outdtype && tileClass>=99 && tileClass<=154){
  const uint32_t foldedWorkers=(tileClass-99)/8+1;
  uint32_t foldedLaunch=foldedWorkers<kGeometryFoldMinLaunch?kGeometryFoldMinLaunch:foldedWorkers;
  if(foldedLaunch>platform.GetCoreNumAiv())foldedLaunch=platform.GetCoreNumAiv();
  context->SetBlockDim(foldedLaunch);
 }
 // Coarsen complete128-row prefixes and merge a short final remainder.
 if(halfInput && dtype==outdtype && tile==64 && t->perCore==64 &&
    t->batch>128 && t->batch%128>0 && t->batch%128<=8)
  tileClass=155+(t->batch/128-1)*8+(t->batch%128-1);
 // Exact single-tile length is ordinary shape metadata.
 if(halfInput && dtype==outdtype && tile==64 && t->perCore==64 &&
    workers==1 && t->batch<64)
  tileClass=179+t->batch-1;
 // A partial32-row cohort of fewer than64 rows fits one vector repeat.
 if(!halfInput && dtype==outdtype && tile==32 && t->perCore==32 &&
    t->batch<64 && t->batch%32!=0){
  tileClass=179+t->batch-1;
  t->perCore=64;
 }
 // Propagate only actual scalar metadata for complete static resource families.
 if(!halfInput && dtype==outdtype && thresholdKind==1U &&
    ((tileClass>=36U && tileClass<=98U) || tileClass==19U) && t->waistRatio==0.8f)thresholdKind=3;
 if(halfInput && dtype==outdtype && thresholdKind==2U &&
    tileClass>=155U && tileClass<=178U && t->waistRatio==0.5f)thresholdKind=3;
 if(!halfInput && dtype==outdtype && thresholdKind==1U &&
    tileClass>=179U && tileClass<=241U && t->waistRatio==0.8f)thresholdKind=3;
 // Only actual coefficient metadata selects this complete resource family.
 if(halfInput && dtype==outdtype && thresholdKind==2U &&
    (tileClass==19U || (tileClass>=36U && tileClass<=98U)) &&
    t->waistRatio==0.8f)thresholdKind=3;
 // Propagate the actual coefficient only for complete matched Half owners.
 if(halfInput && dtype==outdtype && thresholdKind==2U && t->waistRatio==0.9f &&
    tileClass>=3U && tileClass<=17U && (tileClass&1U)!=0U)thresholdKind=3U;
 // Propagate the real coefficient for complete native Float owners.
 if(!halfInput && dtype==outdtype && thresholdKind==1U && tileClass==5U &&
    t->waistRatio==0.9f)thresholdKind=3U;
 // Plan1 reuses the logical owner and derives tensor bases once.
 uint32_t addressPlan=!halfInput && dtype==outdtype && thresholdKind==1U &&
     tileClass>=27U && tileClass<=35U?1U:0U;
 uint32_t broadcastPlan=halfInput && dtype==outdtype && (thresholdKind==2U || thresholdKind==3U) &&
     tileClass>=3U && tileClass<=17U && (tileClass&1U)!=0U?1U:0U;
 // The complete native Float coefficient family uses score-broadcast plan1.
 if(!halfInput && dtype==outdtype && thresholdKind==3U && tileClass==19U)
  broadcastPlan=1U;
 // Complete native Half owners use the shared score descriptor resource.
 if(halfInput && dtype==outdtype && thresholdKind==3U && tileClass==19U)
  broadcastPlan=1U;
 // Near-full native Float owners validate a redundant threshold Add on device.
 if(!halfInput && dtype==outdtype && thresholdKind==3U && tileClass>=92U && tileClass<=98U)
  broadcastPlan=1U;
 // Near-full Float score producers share one single-repeat descriptor.
 if(!halfInput && dtype==outdtype && thresholdKind==3U && tileClass>=92U && tileClass<=98U)
  addressPlan=1U;
 // Native Half tails in the final32B output block share score state.
 if(halfInput && dtype==outdtype && thresholdKind==3U && tileClass>=84U && tileClass<=98U)
  broadcastPlan=1U;
 auto *metadataPayload=context->GetRawTilingData();
 uint32_t floatMetadataPlan=!halfInput && dtype==outdtype && thresholdKind==1U &&
     tileClass>=3U && tileClass<=17U && (tileClass&1U)!=0U &&
     metadataPayload->GetCapacity()>=32U?1U:0U;
 // A256-row prefix and short remainder share one native owner; launch is unchanged.
 const bool folded256=halfInput && dtype==outdtype && thresholdKind==3U &&
     tileClass>=155U && tileClass<=178U && t->batch>256U &&
     t->batch%256U>0U && t->batch%256U<=8U;
 if(folded256){tileClass=155U+t->batch%256U-1U;addressPlan=1U;}
 // Near-full native Float owners retain the ordinary integer metadata.
 const bool selectScalarPitchPlan=!halfInput && dtype==outdtype && thresholdKind==3U &&
     tileClass>=92U && tileClass<=98U && addressPlan==1U && broadcastPlan==1U;
 if(selectScalarPitchPlan)floatMetadataPlan=3U;
 // A complete native Float tile uses the original integer header and owner map.
 if(!halfInput && dtype==outdtype && thresholdKind==3U && tileClass==19U && addressPlan==0U && broadcastPlan==1U)
  floatMetadataPlan=2U;
 // Fully occupied folded64 cohorts share the remainder resource.
 const bool foldedRawEntry=!halfInput && dtype==outdtype && thresholdKind==1U &&
     tileClass>=99U+(kGeometryFoldMinLaunch-1U)*8U && tileClass<=154U;
 if(foldedRawEntry){
  tileClass=99U+(tileClass-99U)%8U;
  addressPlan=1U;broadcastPlan=1U;floatMetadataPlan=3U;
 }
 // Split-Half near-full owners retain their ordinary integer metadata.
 const bool splitHalfEntryV2=halfInput && dtype==outdtype && thresholdKind==3U &&
     tileClass>=84U && tileClass<=98U && addressPlan==0U && broadcastPlan==1U;
 if(splitHalfEntryV2){
  tileClass=107U+tileClass-84U;addressPlan=1U;floatMetadataPlan=3U;
 }
 // Complete native Half64 owners retain their original byte-origin map.
 const bool halfFullEarlyEntry=halfInput && dtype==outdtype && thresholdKind==3U &&
     tileClass==5U && addressPlan==0U && broadcastPlan==1U;
 if(halfFullEarlyEntry){tileClass=255U;addressPlan=1U;floatMetadataPlan=3U;}
 // Same fully occupied folded owners; only their index arithmetic differs.
 if(!halfInput && dtype==outdtype && thresholdKind==1U && addressPlan==1U &&
    broadcastPlan==1U && floatMetadataPlan==3U && tileClass>=99U && tileClass<=106U)
  tileClass=125U+tileClass-99U;
 // Complete native Float64 owners retain their existing origin and buffers.
 const bool floatFullEntryMaskReuse=!halfInput && dtype==outdtype &&
     thresholdKind==3U && tileClass==5U && addressPlan==0U && broadcastPlan==0U;
 if(floatFullEntryMaskReuse){tileClass=250U;addressPlan=1U;broadcastPlan=1U;floatMetadataPlan=3U;}
 // Preserve the complete near-full Half family and its integer payload.
 if(splitHalfEntryV2)tileClass=136U+tileClass-107U;
 // Complete native Float32 owners retain ordinary integer metadata.
 const bool floatUnitEntry= !halfInput && dtype==outdtype && thresholdKind==1U &&
     tileClass>=27U && tileClass<=28U && addressPlan==1U && t->waistRatio==1.0f;
 if(floatUnitEntry){
  tileClass=248U+tileClass-27U;broadcastPlan=1U;floatMetadataPlan=3U;
 }
 // The minimum fully occupied folded cohort is known to the Host planner.
 // Both native Float threshold tags denote a present Float input here.
 // Two recipe tags pack the lower/upper four remainders without changing dtype.
 if(foldedRawEntry && tileClass>=125U && tileClass<=132U &&
    t->batch/64U==kGeometryFoldMinLaunch){
  const uint32_t extra=tileClass-125U+1U;
  tileClass=251U+(extra-1U)%4U;
  thresholdKind=extra<=4U?1U:3U;
  if(platform.GetCoreNumAiv()>=8U)context->SetBlockDim(8U);
 }
 if(runtimeFold){
  tileClass=251U+(short64-1U)%4U;
  thresholdKind=short64<=4U?1U:3U;
  addressPlan=1U;broadcastPlan=1U;floatMetadataPlan=3U;
  context->SetBlockDim(8U);
 }
 if(!nativeSignature){
  tileClass=0;addressPlan=0;broadcastPlan=0;floatMetadataPlan=0;
  thresholdKind=t->hasThreshold?(t->thresholdIsHalf?2U:1U):0U;
  // The generic loop uses runtime perCore and requires one real owner per block.
  const uint32_t genericWorkers=(t->batch+t->perCore-1U)/t->perCore;
  context->SetBlockDim(genericWorkers);
 }
 // The original complete64 Float family has one repeat per consumer.
 // Literal metadata is selected only by the actual attribute and native types.
 const bool halfShortLiteralBits=halfInput && dtype==outdtype && thresholdKind==2U &&
     tileClass>=211U && tileClass<=241U && addressPlan==0U && broadcastPlan==0U &&
     floatMetadataPlan==0U && t->waistRatio==0.8f;
 // The existing unit family has exactly one logical output owner.
 // The existing complete unit tile has one logical output owner.
 const bool floatShortPositiveZeroPlan=!halfInput && nativeSignature && dtype==outdtype &&
     thresholdKind==3U && tileClass>=211U && tileClass<=241U &&
     addressPlan==0U && broadcastPlan==0U && floatMetadataPlan==0U && t->waistRatio==0.8f;
 // Merge complete small native batches, keeping ordinary integer metadata.
 const bool halfBatchMerge=halfFullEarlyEntry && t->batch>=128U && t->batch<=512U &&
     t->batch%64U==0U && platform.GetCoreNumAiv()>=8U;
 if(halfBatchMerge)context->SetBlockDim(8U);
 // Complete native Float batches use one compute owner and eight launch blocks.
 const bool floatBatchMerge=floatFullEntryMaskReuse && t->batch>=128U && t->batch<=512U &&
     t->batch%64U==0U && platform.GetCoreNumAiv()>=8U;
 if(floatBatchMerge){tileClass=62U+t->batch/64U;context->SetBlockDim(8U);}
 const uint32_t repeatAddressPlan=runtimeFold?68U+full64:floatBatchMerge?66U:halfBatchMerge?(50U+t->batch/64U):floatShortPositiveZeroPlan?51U:(splitHalfEntryV2 && tileClass>=136U && tileClass<=150U)?46U:(halfInput && nativeSignature && thresholdKind==2U &&
     tileClass==27U && addressPlan==0U && broadcastPlan==0U && floatMetadataPlan==0U)?64U:(floatUnitEntry && tileClass==248U && thresholdKind==1U &&
     addressPlan==1U && broadcastPlan==1U && floatMetadataPlan==3U)?65U:halfShortLiteralBits?60U:(selectScalarPitchPlan && floatMetadataPlan==3U)?19U:(halfInput && dtype==outdtype && thresholdKind==3U &&
     tileClass==19U && addressPlan==0U && broadcastPlan==1U && floatMetadataPlan==0U)?67U:folded256?13U:(floatFullEntryMaskReuse && tileClass==250U)?45U:
     ((halfFullEarlyEntry && tileClass==255U)?49U:
      ((halfInput && dtype==outdtype && thresholdKind==2U && tileClass>=211U && tileClass<=241U &&
        addressPlan==0U && broadcastPlan==0U && floatMetadataPlan==0U)?18U:((foldedRawEntry && tileClass>=251U && tileClass<=254U)?68U:0U)));
 ASCENDC_TPL_SEL_PARAM(context,dtype,outdtype,thresholdKind,tileClass,addressPlan,broadcastPlan,floatMetadataPlan,repeatAddressPlan);
 if(floatMetadataPlan!=0U && !selectScalarPitchPlan && !foldedRawEntry && !runtimeFold && !splitHalfEntryV2 && !halfFullEarlyEntry && !floatFullEntryMaskReuse && !floatUnitEntry){
  const float metadataBlock[8]={t->waistRatio,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
  std::memcpy(metadataPayload->GetData(),metadataBlock,sizeof(metadataBlock));
  metadataPayload->SetDataSize(sizeof(metadataBlock));
 }
 context->GetWorkspaceSizes(1)[0]=0;
 return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
 *c->GetOutputShape(0)=*c->GetInputShape(0); c->GetOutputShape(0)->SetDim(1,1); return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
 c->SetOutputDataType(0,DT_FLOAT); return GRAPH_SUCCESS;
}
}
namespace ops {
class SafetyBeltViolationGeometry:public OpDef {
public:
 explicit SafetyBeltViolationGeometry(const char *n):OpDef(n) {
 this->Input("person_boxes").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
 this->Input("anchor_points").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
 this->Input("threshold").ParamType(OPTIONAL).DataType({ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
 this->Output("violation_score").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
 this->Attr("waist_ratio").AttrType(REQUIRED).Float();
 this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
 this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
 }
}; OP_ADD(SafetyBeltViolationGeometry);
}
