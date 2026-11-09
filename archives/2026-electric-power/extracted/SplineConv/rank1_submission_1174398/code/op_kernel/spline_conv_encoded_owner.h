#pragma once
#include "spline_conv_wide_owner.h"

template <uint32_t Pad,bool Closed0,bool Closed1>
__aicore__ inline void RunEncodedWideOwner(GM_ADDR x,GM_ADDR edges,
        GM_ADDR pseudo,GM_ADDR weight,GM_ADDR out,GM_ADDR tiling) {
    SplineWideEdgeOwner<Closed0,Closed1,false,true,true> op;
    op.Init(x,edges,pseudo,weight,out,256-8*Pad,8,0);
    op.PrefetchGraph();op.Process(tiling);
}
