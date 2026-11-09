#pragma once
#include "spline_conv_sparse_output.h"

template <uint32_t Edges,uint32_t Blocks>
__aicore__ inline void SplineClearOwner(const GlobalTensor<int64_t>& graph,
        const GlobalTensor<float>& output,uint32_t nodes,uint32_t block,
        const LocalTensor<int64_t>& localGraph,const LocalTensor<float>& zeros) {
    uint32_t bound=((nodes+Blocks-Edges-1)/(Blocks-Edges)+7)/8*8;
    uint32_t begin=(block-Edges)*bound;
    uint32_t rows=begin<nodes?nodes-begin:0;
    if(rows>bound) rows=bound;
    if(!rows) return;
    DataCopyPad(localGraph,graph,DataCopyExtParams{1,Edges*16,0,0,0},
        DataCopyPadExtParams<int64_t>{false,0,0,0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    Duplicate(zeros,0.0f,rows*16);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint32_t occupied=0;
    for(uint32_t e=0;e<Edges;++e) {
        int64_t dst=localGraph.GetValue(Edges+e);
        if(dst>=begin && dst<int64_t(begin+rows))
            occupied|=1u<<(uint32_t(dst)-begin);
    }
    spline_output::ClearInactiveRows(output,zeros,begin,rows,occupied);
}

__aicore__ inline uint32_t SplineTargetBits4(const LocalTensor<int64_t>& edges,int64_t target) {
    return uint32_t(edges.GetValue(4)==target) |
        (uint32_t(edges.GetValue(5)==target)<<1) |
        (uint32_t(edges.GetValue(6)==target)<<2) |
        (uint32_t(edges.GetValue(7)==target)<<3);
}
