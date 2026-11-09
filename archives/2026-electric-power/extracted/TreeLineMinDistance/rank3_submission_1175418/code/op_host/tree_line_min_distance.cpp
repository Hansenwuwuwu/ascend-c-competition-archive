#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/tree_line_min_distance_tiling.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const auto *pointShape = context->GetInputShape(0);
        const auto *segmentShape = context->GetInputShape(1);
        const auto *pointDesc = context->GetInputDesc(0);
        const auto *segmentDesc = context->GetInputDesc(1);
        if (!pointShape || !segmentShape || !pointDesc || !segmentDesc) {
            return ge::GRAPH_FAILED;
        }
        const auto &p = pointShape->GetStorageShape();
        const auto &s = segmentShape->GetStorageShape();
        if (p.GetDimNum() != 3 || s.GetDimNum() != 4 ||
            p.GetDim(2) != 3 || s.GetDim(2) != 2 || s.GetDim(3) != 3 ||
            p.GetDim(0) != s.GetDim(0) || p.GetDim(0) < 1 || p.GetDim(0) > 8 ||
            p.GetDim(1) < 1 || p.GetDim(1) > 16384 || s.GetDim(1) < 1 || s.GetDim(1) > 1024) {
            return ge::GRAPH_FAILED;
        }
        const auto dtype = pointDesc->GetDataType();
        if (dtype != segmentDesc->GetDataType() ||
            (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16)) {
            return ge::GRAPH_FAILED;
        }
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t cores = static_cast<uint32_t>(std::max<int32_t>(1, platform.GetCoreNumAiv()));
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        auto *tiling = context->GetTilingData<TreeLineMinDistanceTilingData>();
        tiling->batches = static_cast<uint32_t>(p.GetDim(0));
        tiling->points = static_cast<uint32_t>(p.GetDim(1));
        tiling->segments = static_cast<uint32_t>(s.GetDim(1));
        tiling->schedule = 1;
        tiling->segmentLog = 3;
        tiling->paddedSegments = 8;
        while (tiling->paddedSegments < tiling->segments) {
            tiling->paddedSegments *= 2;
            ++tiling->segmentLog;
        }
        tiling->pointTile = 1;
        while (tiling->pointTile < tiling->points &&
               tiling->pointTile * tiling->paddedSegments < 1024) tiling->pointTile *= 2;
        if (tiling->paddedSegments <= 64 && tiling->pointTile > 8) tiling->pointTile = 8;
        if (tiling->segments <= 8 && tiling->pointTile == 8) tiling->schedule = 2;
        if (tiling->points <= 8 && tiling->batches * tiling->points <= cores && tiling->segments <= 8) {
            tiling->schedule = 5;
            tiling->pointTile = tiling->segments<=2 && tiling->points>1 ? 4 : 1;
        }
        if (tiling->segments == 1 && tiling->schedule != 5) {
            tiling->schedule = 3;
            tiling->segmentLog = 0;
            tiling->paddedSegments = 1;
            const uint32_t pointCores=std::min(cores,16U);
            tiling->pointTile = ((tiling->points + pointCores - 1) / pointCores + 63) / 64 * 64;
        }
        if (tiling->points >= 16 && tiling->segments >= 128) {
            tiling->schedule = tiling->segments > 512 ? 6 : 7;
            tiling->pointTile = tiling->segments > 512 ? 64 : 16;
        }
        if (tiling->paddedSegments == 64 && tiling->points >= 16) {
            tiling->pointTile = std::min(16U, (tiling->batches*tiling->points+cores-1)/cores);
            tiling->pointTile = std::max(1U, tiling->pointTile);
            if(tiling->batches*tiling->points <= cores*32U) {
                tiling->schedule = 4;
                tiling->pointTile = 32;
            }
        }
        if(tiling->points==1 && tiling->segments>8) {tiling->schedule=8;tiling->pointTile=1;}
        const uint32_t tasks = tiling->batches * ((tiling->points + tiling->pointTile - 1) / tiling->pointTile);
        if(tiling->schedule==2 && tasks<=cores)tiling->schedule=9;
        tiling->blocks = std::min(cores, tasks);
        if(tiling->schedule==6)
            tiling->blocks=std::min(cores,tiling->batches*((tiling->points+31)/32));
        if(tiling->schedule==7)
            tiling->blocks=std::min(cores,tiling->batches*((tiling->points+15)/16));
        // Includes the low parts of coordinate differences for accurate argmin comparisons.
        if (ubSize < 184U * 1024U) return ge::GRAPH_FAILED;
        const uint32_t tail=tiling->segments&63U;
        if(tail>0 && tail<=8) {
            if(tiling->schedule==7)tiling->schedule=tail==1?10:12;
            else if(tiling->schedule==6)tiling->schedule=tail==1?11:13;
        }
        // Native Half single-point batches use a static, repeat-batched Tiny entry.
        if(dtype==ge::DT_FLOAT16 && tiling->schedule==5 &&
           tiling->points==1 && tiling->segments>1)tiling->schedule=14;
        // Native FP32 single-segment four-row tiles use an independent entry.
        if(dtype==ge::DT_FLOAT && tiling->schedule==5 &&
           tiling->segments==1 && tiling->pointTile==4)tiling->schedule=15;
        // Two-segment native Half records fit a sixteen-lane packet.
        if(dtype==ge::DT_FLOAT16 && tiling->schedule==14 && tiling->segments<=2)tiling->schedule=16;
        // Native FP32 one-point two-segment records share the static packet.
        if(dtype==ge::DT_FLOAT && tiling->schedule==5 && tiling->points==1 && tiling->segments==2)
            tiling->schedule=16;
        // The native two-segment point-owner resource needs only N and owners.
        if(dtype==ge::DT_FLOAT && tiling->schedule==5 && tiling->points>1 && tiling->segments==2)
            tiling->schedule=23;
        // A bounded single-query packet needs only the live segment count.
        if(dtype==ge::DT_FLOAT && tiling->schedule==5 && tiling->points==1 &&
           tiling->segments>2U && tiling->segments<=8U)tiling->schedule=24;
        // Native FP32 single-point records pair independent differences.
        if(dtype==ge::DT_FLOAT && tiling->schedule==5 && tiling->points==1 && tiling->segments==1)
            tiling->schedule=20;
        // A complete hardware-resident owner grid and bounded per-owner UB
        // capacity admit batch-wide geometric partitioning with hard sync.
        const uint32_t owners=tiling->batches && tiling->blocks%tiling->batches==0?tiling->blocks/tiling->batches:0;
        if(dtype==ge::DT_FLOAT && tiling->schedule==6 && tiling->paddedSegments==1024 &&
           tiling->blocks==cores && owners>=2 && owners<=8 && tiling->points/owners>=2048 &&
           (tiling->points+owners-1)/owners+8*owners+64<=4096) {
            tiling->schedule=21;
            context->SetScheduleMode(1);
        }
        // The complete resident Half grid shares exact promoted inputs across buckets.
        // Each producer and restoration interval must fit the fixed UB arena.
        if(dtype==ge::DT_FLOAT16 && tiling->schedule==6 && tiling->paddedSegments==1024 &&
           tiling->blocks==cores && owners>=2 && owners<=16 && tiling->points/owners>=512 &&
           (tiling->points+owners-1)/owners+8*owners+64<=4096) {
            tiling->schedule=21;
            context->SetScheduleMode(1);
        }
        // Resident single-point packets encode every supported owner count.
        if(tiling->schedule==16 && tiling->blocks==tiling->batches &&
           tiling->blocks>=1U && tiling->blocks<=8U)
            tiling->schedule=24U+tiling->blocks;
        // A one-tail prefix in the bounded 192-coefficient cache has its own
        // entry so unrelated cache capacities do not enlarge its instruction body.
        if(dtype==ge::DT_FLOAT && tiling->schedule==10 &&
           tiling->paddedSegments==256 && tiling->segments<=192)tiling->schedule=17;
        if(dtype==ge::DT_FLOAT && tiling->schedule==17) {
            const uint32_t ownersPerBatch=cores/tiling->batches;
            if(ownersPerBatch)
                tiling->blocks=tiling->batches*std::min(ownersPerBatch,(tiling->points+15U)/16U);
        }
        // The 576-coefficient prefix arena excludes the full 1024-capacity body.
        // Separate ordinary, one-tail, and bounded-tail cache entry resources.
        if(dtype==ge::DT_FLOAT && tiling->paddedSegments==1024 && tiling->segments<=576) {
            if(tiling->schedule==6)tiling->schedule=18;
            else if(tiling->schedule==11)tiling->schedule=19;
            else if(tiling->schedule==13)tiling->schedule=22;
        }
        // The Half 8x32 and 32x64 pair arenas have independent entries.
        // A complete resident tile grid excludes the dynamic looping resource.
        if(dtype==ge::DT_FLOAT16) {
            const uint32_t residentTiles=tiling->batches*((tiling->points+tiling->pointTile-1)/tiling->pointTile);
            if(residentTiles<=tiling->blocks) {
                if(tiling->schedule==1 && tiling->pointTile==8 && tiling->paddedSegments==32)tiling->schedule=18;
                else if(tiling->schedule==4 && tiling->pointTile==32 && tiling->paddedSegments==64)tiling->schedule=19;
            }
        }
        uint32_t DT_TREE_POINTS = static_cast<uint32_t>(dtype);
        uint32_t SCHEDULE = tiling->schedule;
        // One launch group admits a compile-time active-owner budget.
        const bool residentPacket=dtype==ge::DT_FLOAT &&
            (SCHEDULE==20 || (SCHEDULE==23 && tiling->batches==1U));
        uint32_t RESIDENT_OWNERS=residentPacket && tiling->blocks<=8U?tiling->blocks:0U;
        // The two-segment four-row packet encodes its complete point budget.
        if(dtype==ge::DT_FLOAT && SCHEDULE==23 && tiling->batches==1U)
            RESIDENT_OWNERS=tiling->points;
        // Schedule24 uses this resource axis for its bounded segment budget.
        if(dtype==ge::DT_FLOAT && SCHEDULE==24 && tiling->batches==1U && tiling->blocks==1U)
            RESIDENT_OWNERS=tiling->segments;
        // Schedule15 carries the complete single-batch four-point tile budget.
        if(dtype==ge::DT_FLOAT && SCHEDULE==15 && tiling->batches==1U &&
           tiling->blocks==(tiling->points+3U)/4U)
            RESIDENT_OWNERS=tiling->points;
        // A complete four-row, 32-segment arena fixes the live row budget.
        if(dtype==ge::DT_FLOAT && SCHEDULE==1 && tiling->pointTile==4U &&
           tiling->paddedSegments==32U && tiling->points<=4U && tiling->blocks==tiling->batches)
            RESIDENT_OWNERS=tiling->points;
        ASCENDC_TPL_SEL_PARAM(context, DT_TREE_POINTS, SCHEDULE, RESIDENT_OWNERS);
        context->SetBlockDim(std::min(cores,(tiling->blocks+7U)/8U*8U));
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = dtype==ge::DT_FLOAT && (tiling->schedule==6 || tiling->schedule==21) && tiling->paddedSegments==1024 &&
            tiling->batches*tiling->points/tiling->blocks>=1024 ?
            platform.GetLibApiWorkSpaceSize()+size_t(tiling->blocks)*size_t(6*4096+14*1056+1024)*sizeof(float) : 0;
        if(dtype==ge::DT_FLOAT16 && tiling->schedule==21) {
            const size_t spatialWords=size_t(tiling->blocks)*size_t(6*4096+14*1056+1024);
            const size_t promotedWords=size_t(tiling->batches)*size_t(tiling->points)*4U+
                size_t(tiling->batches)*size_t(tiling->segments)*6U;
            currentWorkspace[0]=platform.GetLibApiWorkSpaceSize()+(spatialWords+promotedWords)*sizeof(float);
        }
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto *p = context->GetInputShape(0);
        const auto *s = context->GetInputShape(1);
        auto *distance = context->GetOutputShape(0);
        auto *index = context->GetOutputShape(1);
        if (!p || !s || !distance || !index || p->GetDimNum() != 3 || s->GetDimNum() != 4 ||
            p->GetDim(2) != 3 || s->GetDim(2) != 2 || s->GetDim(3) != 3 ||
            p->GetDim(0) != s->GetDim(0)) {
            return GRAPH_FAILED;
        }
        distance->SetDimNum(2);
        distance->SetDim(0, p->GetDim(0));
        distance->SetDim(1, p->GetDim(1));
        *index = *distance;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto dtype = context->GetInputDataType(0);
        if (dtype != context->GetInputDataType(1) ||
            (dtype != DT_FLOAT && dtype != DT_FLOAT16)) {
            return GRAPH_FAILED;
        }
        context->SetOutputDataType(0, dtype);
        context->SetOutputDataType(1, DT_INT32);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class TreeLineMinDistance : public OpDef {
    public:
        explicit TreeLineMinDistance(const char *name) : OpDef(name) {
            this->Input("tree_points")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("line_segments")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("min_distance")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("nearest_segment_idx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(TreeLineMinDistance);
}  // namespace ops
