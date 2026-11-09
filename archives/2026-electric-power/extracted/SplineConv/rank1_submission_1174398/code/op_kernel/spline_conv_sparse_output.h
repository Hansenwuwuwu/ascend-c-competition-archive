#pragma once
#include "kernel_operator.h"
namespace spline_output {
using namespace AscendC;
__aicore__ inline uint32_t FirstBit(uint32_t mask) {
    uint32_t low=mask&(0u-mask);
    return uint32_t(63-CountLeadingZero(uint64_t(low)));
}
template <typename T>
__aicore__ inline void CopyRows(const GlobalTensor<T>& global,const LocalTensor<T>& local,uint32_t node,uint32_t rows) {
    if(rows==0) return;
    if((uint64_t(global.GetPhyAddr())&31)==0) DataCopy(global[node*16],local,rows*16);
    else {
        DataCopyExtParams p{1,rows*16*uint32_t(sizeof(T)),0,0,0};
        DataCopyPad(global[node*16],local,p);
    }
}
template <typename T,bool Ready=false>
__aicore__ inline void ClearInactiveRows(const GlobalTensor<T>& global,const LocalTensor<T>& zeros,uint32_t begin,uint32_t rows,uint32_t occupied) {
    if constexpr(!Ready) SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if(occupied==0) {CopyRows(global,zeros,begin,rows);return;}
    uint64_t mask=(~uint64_t(occupied))&((uint64_t(1)<<rows)-1);
    while(mask) {
        uint32_t first=FirstBit(uint32_t(mask));
        uint64_t ended=~(mask>>first);
        uint32_t count=uint32_t(63-CountLeadingZero(ended&(uint64_t(0)-ended)));
        CopyRows(global,zeros,begin+first,count);
        mask&=~(((uint64_t(1)<<count)-1)<<first);
    }
}
}
