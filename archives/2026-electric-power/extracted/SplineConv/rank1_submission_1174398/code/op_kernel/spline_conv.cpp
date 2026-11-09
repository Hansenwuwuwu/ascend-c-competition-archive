#include "kernel_operator.h"
#include "spline_conv_tiling.h"
using namespace AscendC;

#include "spline_conv_resident.h"
#include "spline_conv_streaming.h"
#include "spline_conv_small.h"
template <typename T>
__aicore__ inline bool RunBoundedSparse(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,GM_ADDR workspace,const SplineConvTilingData &t) {
    if(t.ci!=32 || t.streaming || !t.cacheGraph || !t.cacheWeights || !t.fusedDot || t.hasRoot || t.hasBias ||
       t.e>=t.n || t.co!=16 || t.ci!=t.ciPad || t.dim!=2 || t.degree!=1 || t.kernels!=4) return false;
    SplineKernel<T,false,32> op;op.Init(x,edges,pseudo,weight,nullptr,nullptr,out,workspace,t);op.Process();
    return true;
}
#include "spline_conv_single.h"
#include "spline_conv_single_stage.h"
#include "spline_conv_single_bulk_stage.h"
#include "spline_conv_local_weights.h"
#include "spline_conv_compact.h"
#include "spline_conv_edge_owner.h"
#include "spline_conv_four_owner.h"
#include "spline_conv_wide_owner.h"
#include "spline_conv_encoded_owner.h"
#include "tiling_key_spline_conv.h"
template <typename DT_X, uint32_t SINGLE, uint32_t SPARSE, bool CLOSED0, bool CLOSED1, uint32_t NODEPAD>
__global__ __aicore__ void spline_conv(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,GM_ADDR workspace,GM_ADDR tiling) {
    if constexpr(SINGLE) {
        SplineConvTilingData unused;
        if constexpr(SINGLE==6) RunSingleClearAhead<CLOSED0,CLOSED1,true,true>(x,edges,pseudo,weight,out,31-2*NODEPAD);
        else if constexpr(SINGLE==5 && sizeof(DT_X)==4) RunSingleTailBulk<CLOSED0,CLOSED1>(x,edges,pseudo,weight,out);
        else if constexpr(SINGLE==4 && sizeof(DT_X)==4) RunSingleStaged<CLOSED0,CLOSED1,true>(x,edges,pseudo,weight,out);
        else if constexpr(SINGLE==3 && sizeof(DT_X)==4) RunSingleBulkStaged<CLOSED0,CLOSED1,false>(x,edges,pseudo,weight,out);
        else if constexpr(SINGLE==4) RunSingleStatic<DT_X,true,false,CLOSED0,CLOSED1,true,true>(x,edges,pseudo,weight,out,unused,tiling);
        else if constexpr(SINGLE==3) RunSingleStatic<DT_X,true,false,CLOSED0,CLOSED1,true>(x,edges,pseudo,weight,out,unused,tiling);
        else if constexpr(SINGLE==2) RunSingleStatic<DT_X,true,true,CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,unused,tiling);
        else RunSingleStatic<DT_X,true>(x,edges,pseudo,weight,out,unused,tiling);
        return;
    }
    else if constexpr(SPARSE) {
        const __gm__ SplineConvTilingData *td=(const __gm__ SplineConvTilingData*)tiling;
        if constexpr(SPARSE==5) {
            RunPartialEdgeOwner<NODEPAD,CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,tiling);return;
        }
        if constexpr(SPARSE==6) {
            RunFixedEdgeOwner<NODEPAD,CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,tiling);return;
        }
        if constexpr(SPARSE==9) {
            RunEncodedWideOwner<NODEPAD,CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,tiling);return;
        }
        if constexpr(SPARSE==8) {
            RunWideEarlyOwner<CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,tiling);return;
        }
        if constexpr(SPARSE==7) {
            uint32_t edgesPerTask=GetBlockNum()-(NODEPAD+1),block=GetBlockIdx();
            SplineWideEdgeOwner<CLOSED0,CLOSED1,true> op;
            op.Init(x,edges,pseudo,weight,out,0,edgesPerTask,0);
            if(block>=edgesPerTask) {
                op.PrefetchClearGraph();op.SetShape(td->n,0);op.ClearOnly(block-edgesPerTask,NODEPAD+1);return;
            }
            op.PrefetchGraph();op.SetShape(td->n,td->mean);op.Process();return;
        }
        if constexpr(SPARSE==4) {
            SplineWideEdgeOwner<CLOSED0,CLOSED1> op;
            op.Init(x,edges,pseudo,weight,out,0,GetBlockNum(),0);
            op.PrefetchGraph();op.SetShape(td->n,td->mean);op.Process();return;
        }
        SplineConvTilingData minimal;
        minimal.n=td->n; minimal.e=td->e; minimal.blocks=GetBlockNum();
        minimal.scale[0]=CLOSED0?2:1; minimal.scale[1]=CLOSED1?2:1; minimal.mean=td->mean;
        if constexpr(SPARSE==3) {
            SplineEdgeOwner<32,CLOSED0,CLOSED1> op;
            op.Init(x,edges,pseudo,weight,out,minimal.n,minimal.e,minimal.mean);op.Process();
        } else if constexpr(SPARSE==1) {
            SplineCompactSparse<DT_X,32> op;op.Init(x,edges,pseudo,weight,out,minimal,nullptr);if(minimal.blocks==8) op.template Process<8>();else op.Process();
        } else {
            SplineCompactSparse<DT_X,128> op;op.Init(x,edges,pseudo,weight,out,minimal,nullptr);
            if(minimal.blocks==8) op.template Process<8>();else op.Process();
        }
    } else {
    REGISTER_TILING_DEFAULT(SplineConvTilingData);
    GET_TILING_DATA_WITH_STRUCT(SplineConvTilingData,t,tiling);
    if(RunSingleStatic<DT_X>(x,edges,pseudo,weight,out,t)) return;
    if(RunSparseLocalWeights<DT_X>(x,edges,pseudo,weight,out,t)) return;
    if(RunCompactSparse<DT_X>(x,edges,pseudo,weight,out,t)) return;
    if(RunBoundedSparse<DT_X>(x,edges,pseudo,weight,out,workspace,t)) return;
    if(t.smallScalar) {TPipe pipe; SplineSmallKernel<DT_X> op;op.Init(x,edges,pseudo,weight,nullptr,nullptr,out,t,&pipe);op.Process();}
    else if(t.streaming) {SplineStreamingKernel<DT_X> op;op.Init(x,edges,pseudo,weight,nullptr,nullptr,out,workspace,t);op.Process();}
    else if(t.kernels>64) {SplineKernel<DT_X,true> op; op.Init(x,edges,pseudo,weight,nullptr,nullptr,out,workspace,t); op.Process();}
    else {SplineKernel<DT_X> op; op.Init(x,edges,pseudo,weight,nullptr,nullptr,out,workspace,t); op.Process();}
    }
}

template <typename DT_X, uint32_t SINGLE, uint32_t SPARSE, bool CLOSED0, bool CLOSED1, uint32_t NODEPAD>
__global__ __aicore__ void spline_conv_full(GM_ADDR x,GM_ADDR edges,GM_ADDR pseudo,GM_ADDR weight,GM_ADDR root,GM_ADDR bias,GM_ADDR out,GM_ADDR workspace,GM_ADDR tiling) {
    if constexpr(SINGLE) {
        SplineConvTilingData unused;
        if constexpr(SINGLE==6) RunSingleClearAhead<CLOSED0,CLOSED1,true,true>(x,edges,pseudo,weight,out,31-2*NODEPAD);
        else if constexpr(SINGLE==5 && sizeof(DT_X)==4) RunSingleTailBulk<CLOSED0,CLOSED1>(x,edges,pseudo,weight,out);
        else if constexpr(SINGLE==4 && sizeof(DT_X)==4) RunSingleStaged<CLOSED0,CLOSED1,true>(x,edges,pseudo,weight,out);
        else if constexpr(SINGLE==3 && sizeof(DT_X)==4) RunSingleBulkStaged<CLOSED0,CLOSED1,false>(x,edges,pseudo,weight,out);
        else if constexpr(SINGLE==4) RunSingleStatic<DT_X,true,false,CLOSED0,CLOSED1,true,true>(x,edges,pseudo,weight,out,unused,tiling);
        else if constexpr(SINGLE==3) RunSingleStatic<DT_X,true,false,CLOSED0,CLOSED1,true>(x,edges,pseudo,weight,out,unused,tiling);
        else if constexpr(SINGLE==2) RunSingleStatic<DT_X,true,true,CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,unused,tiling);
        else RunSingleStatic<DT_X,true>(x,edges,pseudo,weight,out,unused,tiling);
        return;
    }
    else if constexpr(SPARSE) {
        const __gm__ SplineConvTilingData *td=(const __gm__ SplineConvTilingData*)tiling;
        if constexpr(SPARSE==5) {
            RunPartialEdgeOwner<NODEPAD,CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,tiling);return;
        }
        if constexpr(SPARSE==6) {
            RunFixedEdgeOwner<NODEPAD,CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,tiling);return;
        }
        if constexpr(SPARSE==9) {
            RunEncodedWideOwner<NODEPAD,CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,tiling);return;
        }
        if constexpr(SPARSE==8) {
            RunWideEarlyOwner<CLOSED0,CLOSED1>(x,edges,pseudo,weight,out,tiling);return;
        }
        if constexpr(SPARSE==7) {
            uint32_t edgesPerTask=GetBlockNum()-(NODEPAD+1),block=GetBlockIdx();
            SplineWideEdgeOwner<CLOSED0,CLOSED1,true> op;
            op.Init(x,edges,pseudo,weight,out,0,edgesPerTask,0);
            if(block>=edgesPerTask) {
                op.PrefetchClearGraph();op.SetShape(td->n,0);op.ClearOnly(block-edgesPerTask,NODEPAD+1);return;
            }
            op.PrefetchGraph();op.SetShape(td->n,td->mean);op.Process();return;
        }
        if constexpr(SPARSE==4) {
            SplineWideEdgeOwner<CLOSED0,CLOSED1> op;
            op.Init(x,edges,pseudo,weight,out,0,GetBlockNum(),0);
            op.PrefetchGraph();op.SetShape(td->n,td->mean);op.Process();return;
        }
        SplineConvTilingData minimal;
        minimal.n=td->n; minimal.e=td->e; minimal.blocks=GetBlockNum();
        minimal.scale[0]=CLOSED0?2:1; minimal.scale[1]=CLOSED1?2:1; minimal.mean=td->mean;
        if constexpr(SPARSE==3) {
            SplineEdgeOwner<32,CLOSED0,CLOSED1> op;
            op.Init(x,edges,pseudo,weight,out,minimal.n,minimal.e,minimal.mean);op.Process();
        } else if constexpr(SPARSE==1) {
            SplineCompactSparse<DT_X,32> op;op.Init(x,edges,pseudo,weight,out,minimal,nullptr);if(minimal.blocks==8) op.template Process<8>();else op.Process();
        } else {
            SplineCompactSparse<DT_X,128> op;op.Init(x,edges,pseudo,weight,out,minimal,nullptr);
            if(minimal.blocks==8) op.template Process<8>();else op.Process();
        }
    } else {
    REGISTER_TILING_DEFAULT(SplineConvTilingData);
    GET_TILING_DATA_WITH_STRUCT(SplineConvTilingData,t,tiling);
    if(RunSingleStatic<DT_X>(x,edges,pseudo,weight,out,t)) return;
    if(RunSparseLocalWeights<DT_X>(x,edges,pseudo,weight,out,t)) return;
    if(RunCompactSparse<DT_X>(x,edges,pseudo,weight,out,t)) return;
    if(RunBoundedSparse<DT_X>(x,edges,pseudo,weight,out,workspace,t)) return;
    if(t.smallScalar) {TPipe pipe; SplineSmallKernel<DT_X> op;op.Init(x,edges,pseudo,weight,root,bias,out,t,&pipe);op.Process();}
    else if(t.streaming) {SplineStreamingKernel<DT_X> op;op.Init(x,edges,pseudo,weight,root,bias,out,workspace,t);op.Process();}
    else if(t.kernels>64) {SplineKernel<DT_X,true> op; op.Init(x,edges,pseudo,weight,root,bias,out,workspace,t); op.Process();}
    else {SplineKernel<DT_X> op; op.Init(x,edges,pseudo,weight,root,bias,out,workspace,t); op.Process();}
    }
}

namespace AscendC {
// The kernels use basic APIs and explicit scratch, with no system workspace consumer.
__aicore__ inline void SplineConvBasicWorkspace(GM_ADDR) {}
__aicore__ inline GM_ADDR SplineConvUserWorkspace(GM_ADDR workspace) {
    return workspace + RESERVED_WORKSPACE;
}
}

#define SetSysWorkspaceForce SplineConvBasicWorkspace
#define GetUserWorkspace SplineConvUserWorkspace
