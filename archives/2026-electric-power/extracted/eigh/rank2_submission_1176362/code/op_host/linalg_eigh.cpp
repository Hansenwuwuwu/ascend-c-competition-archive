#include <algorithm>
#include <cstring>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_eigh_tiling.h"
#include "../op_kernel/tiling_key_linalg_eigh.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *ctx) {
    const auto *shape = ctx->GetInputShape(0);
    const auto *desc = ctx->GetInputDesc(0);
    if (!shape || !desc || desc->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    const auto &s = shape->GetStorageShape();
    const auto rank = s.GetDimNum();
    if (rank < 2) return ge::GRAPH_FAILED;
    int64_t n = s.GetDim(rank - 1), batch = 1;
    if (n < 1 || n > 512 || s.GetDim(rank - 2) != n) return ge::GRAPH_FAILED;
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (s.GetDim(i) < 1 || s.GetDim(i) > 64 || batch > INT32_MAX / s.GetDim(i)) return ge::GRAPH_FAILED;
        batch *= s.GetDim(i);
    }
    const char *uplo = ctx->GetAttrs() ? ctx->GetAttrs()->GetStr(0) : nullptr;
    if (uplo && std::strcmp(uplo, "L") && std::strcmp(uplo, "U")) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint32_t cores = platform.GetCoreNumAiv();
    if (!cores) return ge::GRAPH_FAILED;
    auto *t = ctx->GetTilingData<LinalgEighTilingData>();
    t->n = n; t->batch = batch; t->stride = n>25 ? (n+63)/64*64 : (n+7)/8*8;
    t->upper = uplo && uplo[0] == 'U';
    t->vectorAlgorithm=static_cast<uint32_t>(n>=3 && n<=25?
        EighVectorAlgorithm::ResidentSmall:EighVectorAlgorithm::Distributed);
    t->vectorFlags=n>=3 && n<=25?EIGH_SMALL_TWISTED_FACTORIZATION:0;
    if(n<=2){
        // Analytic matrices are independent; one vector core owns each matrix.
        t->rows=n;t->team=1;t->groups=std::min<uint32_t>(batch,cores);
        ctx->SetBlockDim(t->groups);
        uint32_t DT_A=static_cast<uint32_t>(ge::DT_FLOAT),kernelMode=0;
        ASCENDC_TPL_SEL_PARAM(ctx,DT_A,kernelMode);
        ctx->GetWorkspaceSizes(1)[0]=0;
        return ge::GRAPH_SUCCESS;
    }
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    // Keep space for compiler spills and library scratch in addition to TBufs.
    if (ub <= 16 * 1024) return ge::GRAPH_FAILED;
    ub -= 16 * 1024;
    if(n>25 && n<=64){
        const uint32_t ownerCapacity=std::max<uint32_t>(1,cores/uint32_t(batch));
        const uint32_t desiredOwners=std::min<uint32_t>((n+7)/8,ownerCapacity);
        const uint32_t roundedOwners=(desiredOwners+7)/8*8;
        const uint32_t owners=roundedOwners<=ownerCapacity && roundedOwners<=uint32_t(n)?roundedOwners:desiredOwners;
        const uint32_t ownedRows=(n+owners-1)/owners;
        // Reserve a complete row arena for numerical fallback, while
        // normal work still owns only its original eigenvalue interval.
        const uint64_t paddedRows=(ownedRows+7)/8*8;
        // PackMatrix rounds every complete fallback to eight rows.
        const uint64_t fullPaddedRows=(uint64_t(n)+7)/8*8;
        const uint64_t dense=fullPaddedRows*t->stride;
        const uint64_t resident=std::max<uint64_t>(8192,3*dense+((dense+1)/2+7)/8*8+22*t->stride+64+256);
        const uint64_t wide=std::max<uint64_t>(2464,((t->stride+1)/2)*paddedRows);
        const uint64_t requiredBytes=(resident+wide+128+6*n*t->stride)*sizeof(float);
        if(requiredBytes<=ub){
            t->team=owners;t->rows=ownedRows;t->groups=std::min<uint32_t>(batch,cores/owners);
            const uint32_t activeCores=t->groups*owners,roundedCores=(activeCores+7)/8*8;
            ctx->SetBlockDim(roundedCores<=cores?roundedCores:activeCores);
            t->vectorAlgorithm=static_cast<uint32_t>(EighVectorAlgorithm::ResidentMedium);
            t->vectorFlags=EIGH_MEDIUM_OUTPUT_SCRATCH|EIGH_MEDIUM_CERTIFIED_FLOAT_RAYLEIGH|EIGH_MEDIUM_PRIVATE_SOLVE|EIGH_MEDIUM_TWO_INVERSE_ITERATIONS|EIGH_MEDIUM_VECTOR_ORIGINAL_QUOTIENT;
            if(owners>1){
                t->vectorFlags|=EIGH_MEDIUM_SHARED_REDUCTION|EIGH_MEDIUM_BOUNDARY_CONSENSUS;
                t->vectorFlags&=~EIGH_MEDIUM_OUTPUT_SCRATCH;
            }
            // Restrict the vector-only reflector pipeline to the final
            // eight-column bucket of one64-float physical row. All shapes
            // in this contiguous range use the same complete eigensolver.
            if(owners>1 && t->stride==64 && uint64_t(n)+8>t->stride)
                t->vectorFlags|=EIGH_MEDIUM_UB_REFLECTOR_PIPELINE;
            else if(owners>1 && t->stride==64 && fullPaddedRows>=32 &&
                    fullPaddedRows<=t->stride-8 && 2*16*t->stride<=dense)
                // Complete26..56 padding buckets: the same current-call UB
                // history and both16-row tiles fit without extra allocation.
                t->vectorFlags|=EIGH_MEDIUM_NARROW_ROW_PIPELINE;
            ASCENDC_TPL_SEL_PARAM(ctx,static_cast<uint32_t>(ge::DT_FLOAT),uint32_t(0));
            // Match the restored release's batch launch policy: wait for
            // enough idle cores before starting the complete launch.
            if(ctx->SetScheduleMode(1)!=ge::GRAPH_SUCCESS)return ge::GRAPH_FAILED;
            // Q is fully published to the output before certificate readers.
            // Only the separate owner certificate slots require workspace.
            const uint64_t matrixFloatCount=(t->vectorFlags&EIGH_MEDIUM_OUTPUT_SCRATCH)?0:uint64_t(n)*t->stride;
            const uint32_t metricFloatCount=owners*EIGH_CERTIFICATE_SLOT_FLOAT_COUNT;
            ctx->GetWorkspaceSizes(1)[0]=
                uint64_t(batch)*(matrixFloatCount+metricFloatCount)*sizeof(float);
            if(t->vectorFlags&EIGH_MEDIUM_SHARED_REDUCTION)
                ctx->GetWorkspaceSizes(1)[0]+=uint64_t(batch)*((uint64_t(n)+1)*t->stride+16)*sizeof(float);
            // Classification records outlive every reader and are disjoint
            // from the later Gram-certificate publications.
            if(t->vectorFlags&EIGH_MEDIUM_BOUNDARY_CONSENSUS)
                ctx->GetWorkspaceSizes(1)[0]+=uint64_t(batch)*metricFloatCount*sizeof(float);
            return ge::GRAPH_SUCCESS;
        }
    }

    uint32_t minTeam = 1;
    while (minTeam <= cores) {
        uint32_t rows = (n + minTeam - 1) / minTeam;
        uint64_t dense = std::max<uint64_t>(uint64_t(rows) * t->stride,
            uint64_t(t->stride) * ((rows + 7) / 8 * 8));
        if(n>=3 && n<=25)dense=uint64_t(t->stride)*t->stride;
        uint64_t wideN = n>=3 && n<=25 ? t->stride : (n<=25 ? n : 25);
        uint64_t wide = std::max<uint64_t>(3 * wideN * wideN + 21 * wideN + 64, 8 * t->stride);
        wide=std::max<uint64_t>(wide,((t->stride+1)/2)*((rows+7)/8*8));
        // Keep inverse iteration within one 64-float SIMD lane group.
        uint64_t color=n>25 && t->stride%64==0?1:0;
        uint64_t resident=3*dense+((dense+1)/2+7)/8*8+22*t->stride+64+(128+128*(t->stride%128==64))*color;
        if(n>25)resident=std::max<uint64_t>(resident,8192);
        if(n>128 && n<=256)resident=std::max<uint64_t>(resident,2*96*128);
        uint64_t small=n>=3 && n<=25?17*uint64_t(t->stride)*t->stride:0;
        // A complete one-repeat matrix can stage reflector history in UB.
        // Include that invocation-local allocation in the same UB bound.
        uint64_t localHistory=n>25 && n<=64?uint64_t(n)*t->stride:0;
        if ((n <= 25 || rows <= 64) && (resident+wide+128*color+small+localHistory)*4<=ub) break;
        minTeam *= 2;
    }
    if (minTeam > cores) return ge::GRAPH_FAILED;
    uint32_t groups = std::min<uint32_t>(batch, cores / minTeam);
    uint32_t targetRows = n <= 25 ? 1 : (n<=64 && n%64 ? 8 : 16);
    uint32_t team = std::max(minTeam, std::min<uint32_t>((n + targetRows - 1) / targetRows, cores / groups));
    t->rows = (n + team - 1) / team;
    t->team = (n + t->rows - 1) / t->rows;
    t->groups = std::min<uint32_t>(batch, cores / t->team);
    // Nonaligned matrices fitting one SIMD width use vector ownership.
    // Prefer a complete pair where that preserves the actual UB/core bound.
    if(n>25 && n<=64 && n%64 && t->team%2 && t->team>minTeam){
        for(uint32_t requested=t->team-1;requested>=minTeam && requested>=2;requested-=2){
            uint32_t otherRows=(n+requested-1)/requested,otherTeam=(n+otherRows-1)/otherRows;
            if(otherTeam>=minTeam && otherTeam%2==0){t->rows=otherRows;t->team=otherTeam;break;}
        }
        t->groups=std::min<uint32_t>(batch,cores/t->team);
    }
    uint32_t padded=(t->rows+7)/8*8,physicalTeam=(t->team+1)/2*2;
    // Retire a padded tail locally only when its actual row count fits
    // one SIMD repeat. A dense96-row serial tail lost time in the ABBA test.
    const uint32_t privateTailColumnCount=96,maximumPrivateTailRowCount=64;
    if(n>128 && n<=256 && uint64_t(n)+privateTailColumnCount<=uint64_t(t->stride)+maximumPrivateTailRowCount)
        t->vectorFlags|=EIGH_DISTRIBUTED_TAIL_96;
    uint32_t USE_CUBE=n>256 && t->stride>=64 && t->stride%64==0 && padded>=16;
    if(USE_CUBE && t->stride>=384)USE_CUBE=2;
    // Keep a valid odd vector team rather than reducing parallelism
    // merely to enable the mixed Cube entry.
    if(USE_CUBE && t->team%2)USE_CUBE=0;
    if(n>25 && n%64==0 && t->rows%8){
        for(uint32_t alignedRows=(t->rows+7)/8*8;alignedRows<=64;alignedRows+=8){
            if(n%alignedRows)continue;
            uint32_t alignedTeam=uint32_t(n)/alignedRows;
            if(alignedTeam<minTeam || alignedTeam*t->groups>cores)continue;
            if(USE_CUBE && alignedTeam%2)continue;
            t->rows=alignedRows;t->team=alignedTeam;break;
        }
    }
    padded=(t->rows+7)/8*8;physicalTeam=t->team;
    uint32_t launched=USE_CUBE?t->team/2*t->groups:t->team*t->groups;
    uint32_t launchLimit=USE_CUBE?platform.GetCoreNumAic():cores;
    // Keep logical ownership fixed while padding the physical launch where
    // the actual device has capacity. Empty groups retain all global barriers.
    if(n>=3 && (launched+7)/8*8<=launchLimit)launched=(launched+7)/8*8;
    ctx->SetBlockDim(launched);
    // In-place history needs a complete aligned output matrix and a launch
    // whose every participant owns rows at each transpose rendezvous.
    if(!USE_CUBE && n>64 && uint64_t(n)==t->stride &&
       launched==t->groups*t->team && uint32_t(batch)%t->groups==0)
        t->vectorFlags|=EIGH_DISTRIBUTED_OUTPUT_HISTORY;
    // SyncAll requires every launched participant to become runnable.
    // Request the engineering API's whole-launch resource scheduling.
    if(ctx->SetScheduleMode(1)!=ge::GRAPH_SUCCESS)return ge::GRAPH_FAILED;
    uint32_t DT_A = static_cast<uint32_t>(ge::DT_FLOAT);
    const uint32_t kernelMode=USE_CUBE;
    ASCENDC_TPL_SEL_PARAM(ctx, DT_A, kernelMode);
    uint64_t perMatrix=(t->vectorFlags&EIGH_DISTRIBUTED_OUTPUT_HISTORY)?5*uint64_t(t->stride):
        uint64_t(n<=25?2*n+3:(USE_CUBE?2*n+5:n+5))*t->stride;
    if(USE_CUBE){
        perMatrix+=uint64_t(n)*t->stride+uint64_t(physicalTeam)*padded*(t->stride+(USE_CUBE==1?16:64));
        if(USE_CUBE==2)perMatrix+=uint64_t((n-2+63)/64)*6*64*64;
    }
    ctx->GetWorkspaceSizes(1)[0]=(n>=3 && n<=25)?0:uint64_t(batch)*perMatrix*sizeof(float)+(USE_CUBE?LINALG_EIGH_CUBE_PROTOCOL_BYTES:0);
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *ctx) {
    const auto *s = ctx->GetInputShape(0);
    auto *values = ctx->GetOutputShape(0); auto *vectors = ctx->GetOutputShape(1);
    if (!s || !values || !vectors || s->GetDimNum() < 2) return GRAPH_FAILED;
    int64_t n = s->GetDim(s->GetDimNum() - 1);
    if (n < 1 || n > 512 || s->GetDim(s->GetDimNum() - 2) != n) return GRAPH_FAILED;
    const size_t rank = s->GetDimNum();
    *vectors = *s; values->SetDimNum(rank - 1);
    for (size_t i = 0; i + 1 < rank; ++i) values->SetDim(i, s->GetDim(i));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
    if (ctx->GetInputDataType(0) != DT_FLOAT) return GRAPH_FAILED;
    ctx->SetOutputDataType(0, DT_FLOAT); ctx->SetOutputDataType(1, DT_FLOAT);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class LinalgEigh : public OpDef {
public:
    explicit LinalgEigh(const char *name) : OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("eigenvalues").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("eigenvectors").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("UPLO").AttrType(OPTIONAL).String("L");
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgEigh);
}
