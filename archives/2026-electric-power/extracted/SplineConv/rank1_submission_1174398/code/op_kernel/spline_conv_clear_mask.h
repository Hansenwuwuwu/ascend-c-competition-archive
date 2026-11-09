#pragma once
#include "spline_conv_sparse_output.h"

__aicore__ inline void SplineClearMask128(const GlobalTensor<float>& global,
        const LocalTensor<float>& zeros,uint32_t begin,uint32_t rows,
        uint64_t low,uint64_t high) {
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    uint32_t cursor=0;
    for(uint32_t word=0;word<2;++word) {
        uint64_t mask=word?high:low;
        while(mask) {
            uint32_t bit=uint32_t(63-CountLeadingZero(mask&(uint64_t(0)-mask)));
            uint32_t next=word*64+bit;
            if(next>cursor) spline_output::CopyRows(global,zeros,begin+cursor,next-cursor);
            cursor=next+1;mask&=mask-1;
        }
    }
    if(cursor<rows) spline_output::CopyRows(global,zeros,begin+cursor,rows-cursor);
}

__aicore__ inline void SplineCopyZeroSpan128(const GlobalTensor<float>& global,
        const LocalTensor<float>& zeros,uint32_t begin,uint32_t rows) {
    while(rows) {
        uint32_t count=rows>128?128:rows;
        spline_output::CopyRows(global,zeros,begin,count);
        begin+=count;rows-=count;
    }
}
__aicore__ inline void SplineClearMask256(const GlobalTensor<float>& global,
        const LocalTensor<float>& zeros,uint32_t begin,uint32_t rows,
        uint64_t m0,uint64_t m1,uint64_t m2,uint64_t m3) {
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    uint32_t cursor=0,words=(rows+63)/64;
    for(uint32_t word=0;word<words;++word) {
        uint64_t mask=word==0?m0:word==1?m1:word==2?m2:m3;
        while(mask) {
            uint32_t bit=uint32_t(63-CountLeadingZero(mask&(uint64_t(0)-mask)));
            uint32_t next=word*64+bit;
            if(next>cursor) SplineCopyZeroSpan128(global,zeros,begin+cursor,next-cursor);
            cursor=next+1;mask&=mask-1;
        }
    }
    if(cursor<rows) SplineCopyZeroSpan128(global,zeros,begin+cursor,rows-cursor);
}
