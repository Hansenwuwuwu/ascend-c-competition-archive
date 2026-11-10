#pragma once

#include "kernel_operator.h"
#include "swi_glu_activation.h"

namespace swiglu {
using namespace AscendC;

// X在L0A复用，Gate/Up分别写回，让SiLU与第二路投影重叠。
template<uint32_t BK, uint32_t BM, uint32_t BN, uint32_t PANELS>
class KernelCubePipeline {
    static constexpr uint32_t VM = BM / 2;
    static constexpr uint32_t C_CAPACITY = 131072 / (BM * BN * sizeof(float));
    static constexpr uint32_t C_SLOTS = 2 * PANELS < C_CAPACITY ? 2 * PANELS : C_CAPACITY;
    static constexpr uint32_t A_SLOTS = PANELS > 1 && 2 * BM * BK * sizeof(half) <= 65536 ? 2 : 1;
    static_assert(2 * BN * BK * sizeof(half) <= 65536, "Both weights must fit in L0B");
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                               GM_ADDR b2, GM_ADDR y, GM_ADDR workspace) {
        xGm.SetGlobalBuffer((__gm__ half *)x);
        wGm[0].SetGlobalBuffer((__gm__ half *)w1);
        wGm[1].SetGlobalBuffer((__gm__ half *)w2);
        bGm[0].SetGlobalBuffer((__gm__ half *)b1);
        bGm[1].SetGlobalBuffer((__gm__ half *)b2);
        yGm.SetGlobalBuffer((__gm__ half *)y);
        workGm.SetGlobalBuffer((__gm__ half *)GetUserWorkspace(workspace));
        core = GetBlockIdx();
        if ASCEND_IS_AIV { core /= 2; }
        col = (core % (BK / BN)) * BN;
        firstRow = (core / (BK / BN)) * PANELS * BM;
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { Cube(); }
        if ASCEND_IS_AIV { Vector(); }
    }
private:
    __aicore__ inline void CopyX(LocalTensor<half> x, uint32_t panel) {
        Nd2NzParams cp;
        cp.ndNum = 1; cp.nValue = BM; cp.dValue = BK;
        cp.srcNdMatrixStride = 0; cp.srcDValue = BK;
        cp.dstNzC0Stride = BM; cp.dstNzNStride = 1; cp.dstNzMatrixStride = 0;
        DataCopy(x[panel * BM * BK], xGm[(firstRow + panel * BM) * BK], cp);
        SetFlag<HardEvent::MTE2_MTE1>(2 + panel);
    }
    __aicore__ inline void Cube() {
        LocalMemAllocator<AscendC::Hardware::L1> l1;
        LocalMemAllocator<AscendC::Hardware::L0A> l0a;
        LocalMemAllocator<AscendC::Hardware::L0B> l0b;
        LocalMemAllocator<AscendC::Hardware::L0C> l0c;
        auto x = l1.template Alloc<half>(PANELS * BM * BK);
        auto w = l1.template Alloc<half>(2 * BN * BK);
        auto a = l0a.template Alloc<half>(A_SLOTS * BM * BK);
        auto b = l0b.template Alloc<half>(2 * BN * BK);
        auto c = l0c.template Alloc<float>(C_SLOTS * BM * BN);
        // 首块X先到L0A；两份权重在L0B常驻，后续X使用双缓冲。
        CopyX(x, 0);
        Nd2NzParams cp;
        cp.ndNum = 1; cp.nValue = BN; cp.dValue = BK;
        cp.srcNdMatrixStride = 0; cp.srcDValue = BK;
        cp.dstNzC0Stride = BN; cp.dstNzNStride = 1; cp.dstNzMatrixStride = 0;
        DataCopy(w, wGm[0][col * BK], cp);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        DataCopy(w[BN * BK], wGm[1][col * BK], cp);
        SetFlag<HardEvent::MTE2_MTE1>(1);
        for (uint32_t panel = 1; panel < PANELS; ++panel) CopyX(x, panel);
        LoadData2DParams ld;
        for (uint32_t panel = 0; panel < PANELS; ++panel) {
            if (panel >= A_SLOTS) WaitFlag<HardEvent::M_MTE1>(panel % A_SLOTS);
            WaitFlag<HardEvent::MTE2_MTE1>(2 + panel);
            ld.repeatTimes = BK / 16; ld.srcStride = BM / 16;
            #pragma unroll
            for (uint32_t m = 0; m < BM / 16; ++m) {
                ld.startIndex = m;
                LoadData(a[(panel % A_SLOTS) * BM * BK + m * BK * 16], x[panel * BM * BK], ld);
            }
            if (panel == 0) {
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                ld.startIndex = 0; ld.repeatTimes = BN * BK / 256; ld.srcStride = 1;
                LoadData(b, w, ld);
            }
            SetFlag<HardEvent::MTE1_M>(0);
            WaitFlag<HardEvent::MTE1_M>(0);
            #pragma unroll
            for (uint32_t p = 0; p < 2; ++p) {
                // Up权重延后装入L0B，与Gate的矩阵乘和SiLU重叠。
                if (p == 1 && panel == 0) {
                    WaitFlag<HardEvent::MTE2_MTE1>(1);
                    ld.startIndex = 0; ld.repeatTimes = BN * BK / 256; ld.srcStride = 1;
                    LoadData(b[BN * BK], w[BN * BK], ld);
                    SetFlag<HardEvent::MTE1_M>(0);
                    WaitFlag<HardEvent::MTE1_M>(0);
                }
                if (panel * 2 + p >= C_SLOTS) WaitFlag<HardEvent::FIX_M>((panel * 2 + p) % C_SLOTS);
                MmadParams mp;
                mp.m = BM; mp.n = BN; mp.k = BK;
                mp.cmatrixInitVal = true; mp.unitFlag = 3;
                Mmad(c[((panel * 2 + p) % C_SLOTS) * BM * BN], a[(panel % A_SLOTS) * BM * BK], b[p * BN * BK], mp);
                FixpipeParamsV220 fp;
                fp.mSize = BM; fp.nSize = BN; fp.srcStride = BM; fp.dstStride = BN;
                fp.ndNum = 1; fp.srcNdStride = 0; fp.dstNdStride = 0;
                fp.unitFlag = 3; fp.quantPre = QuantMode_t::F322F16;
                Fixpipe(workGm[((core * PANELS + panel) * 2 + p) * BM * BN], c[((panel * 2 + p) % C_SLOTS) * BM * BN], fp);
                SetFlag<HardEvent::FIX_M>((panel * 2 + p) % C_SLOTS);
                CrossCoreSetFlag<2, PIPE_FIX>(panel * 2 + p);
            }
            SetFlag<HardEvent::M_MTE1>(panel % A_SLOTS);
        }
        for (uint32_t i = 0; i < A_SLOTS; ++i) WaitFlag<HardEvent::M_MTE1>(i);
        for (uint32_t i = 0; i < C_SLOTS; ++i) WaitFlag<HardEvent::FIX_M>(i);
    }
    __aicore__ inline void Vector() {
        LocalMemAllocator<AscendC::Hardware::UB> ub;
        auto bias = ub.template Alloc<half>(2 * BN);
        // 保留经过测试的256B间距，错开两组缓冲的地址。
        auto input = ub.template Alloc<half>(2 * VM * BN + 128);
        auto gate = ub.template Alloc<half>(VM * BN + 128);
        auto output = ub.template Alloc<half>(VM * BN);
        DataCopy(bias, bGm[0][col], BN);
        DataCopy(bias[BN], bGm[1][col], BN);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Muls(bias, bias, half(-1), 2 * BN);
        PipeBarrier<PIPE_V>();
        for (uint32_t panel = 0; panel < PANELS; ++panel) {
            if (panel > 0) WaitFlag<HardEvent::V_MTE2>(0);
            uint32_t row = GetSubBlockIdx() * VM;
            CrossCoreWaitFlag(panel * 2);
            DataCopy(input, workGm[((core * PANELS + panel) * 2) * BM * BN + row * BN], VM * BN);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Sub(input, bias, input, (uint64_t)BN, VM, {1, 1, 1, BN / 16, 0, BN / 16});
            NegativeSilu(gate, input, VM * BN);
            CrossCoreWaitFlag(panel * 2 + 1);
            DataCopy(input[VM * BN], workGm[((core * PANELS + panel) * 2 + 1) * BM * BN + row * BN], VM * BN);
            SetFlag<HardEvent::MTE2_V>(1);
            WaitFlag<HardEvent::MTE2_V>(1);
            Sub(input[VM * BN], bias[BN], input[VM * BN], (uint64_t)BN, VM, {1, 1, 1, BN / 16, 0, BN / 16});
            if (panel > 0) WaitFlag<HardEvent::MTE3_V>(0);
            Mul(output, gate, input[VM * BN], VM * BN);
            SetFlag<HardEvent::V_MTE2>(0);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyPad(yGm[(firstRow + panel * BM + row) * BK + col], output,
                        {VM, BN * sizeof(half), 0, (BK - BN) * sizeof(half), 0});
            SetFlag<HardEvent::MTE3_V>(0);
        }
        WaitFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
    uint32_t core, col, firstRow;
    GlobalTensor<half> xGm, wGm[2], bGm[2], yGm, workGm;
};

} // namespace swiglu
