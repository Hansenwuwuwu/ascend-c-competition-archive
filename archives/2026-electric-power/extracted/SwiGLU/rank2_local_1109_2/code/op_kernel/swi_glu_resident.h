#pragma once

#include "kernel_operator.h"
#include "swi_glu_tiling.h"

namespace swiglu {
using namespace AscendC;

template<uint32_t BK, uint32_t STATIC_ROWS = 0, uint32_t STATIC_GROUPS = 0>
class KernelGateResident {
    static_assert((STATIC_ROWS == 0) == (STATIC_GROUPS == 0), "Static row/group plans must be paired");
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                               GM_ADDR b2, GM_ADDR y, GM_ADDR workspace,
                               const SwiGluTilingData &tiling, TPipe *p) {
        pipe = p;
        xGm.SetGlobalBuffer((__gm__ half *)x);
        w1Gm.SetGlobalBuffer((__gm__ half *)w1);
        w2Gm.SetGlobalBuffer((__gm__ half *)w2);
        b1Gm.SetGlobalBuffer((__gm__ half *)b1);
        b2Gm.SetGlobalBuffer((__gm__ half *)b2);
        yGm.SetGlobalBuffer((__gm__ half *)y);
        auto ws = (__gm__ half *)GetUserWorkspace(workspace);
        gateGm.SetGlobalBuffer(ws);
        // Static plans retain the specialized resident kernel's constant buffer sizes.
        const uint32_t rowGroups = STATIC_GROUPS ? STATIC_GROUPS : tiling.residentGroups;
        const uint32_t totalPanels = (STATIC_ROWS ? STATIC_ROWS : tiling.rows) / BM;
        const uint32_t maxPanels = STATIC_ROWS ? (totalPanels + rowGroups - 1) / rowGroups : tiling.residentPanels;
        const uint32_t usedCores = STATIC_GROUPS ? STATIC_GROUPS * (BK / BN) : GetBlockNum();
        maxRows = maxPanels * BM;
        upGm.SetGlobalBuffer(ws + usedCores * maxRows * BN);
        core = GetBlockIdx();
        if ASCEND_IS_AIV { core /= 2; }
        const uint32_t columns = BK / BN;
        const uint32_t group = core / columns;
        col = (core % columns) * BN;
        const uint32_t base = totalPanels / rowGroups;
        const uint32_t extra = totalPanels % rowGroups;
        panels = base + (group < extra ? 1 : 0);
        firstRow = (group * base + (group < extra ? group : extra)) * BM;
        if ASCEND_IS_AIC {
            pipe->InitBuffer(xBuf, maxPanels * BM * BK * sizeof(half));
            pipe->InitBuffer(wBuf, BN * BK * sizeof(half));
            pipe->InitBuffer(bBuf, BN * BK * sizeof(half));
            pipe->InitBuffer(qA, 2, BM * BK * sizeof(half));
            pipe->InitBuffer(cBuf, 4 * BM * BN * sizeof(float));
            weightReady = pipe->AllocEventID<HardEvent::MTE2_MTE1>();
            xReady[0] = pipe->AllocEventID<HardEvent::MTE2_MTE1>();
            xReady[1] = pipe->AllocEventID<HardEvent::MTE2_MTE1>();
            weightToM = pipe->AllocEventID<HardEvent::MTE1_M>();
            weightL1Free = pipe->AllocEventID<HardEvent::MTE1_MTE2>();
            weightL0Free = pipe->AllocEventID<HardEvent::M_MTE1>();
            cFree[0] = pipe->AllocEventID<HardEvent::FIX_M>();
            cFree[1] = pipe->AllocEventID<HardEvent::FIX_M>();
        }
        if ASCEND_IS_AIV {
            pipe->InitBuffer(qInput, 2, VM * BN * sizeof(half));
            pipe->InitBuffer(qOutput, 2, VM * BN * sizeof(half));
            pipe->InitBuffer(qBias, 1, 2 * BN * sizeof(half));
            pipe->InitBuffer(biasPad, 256);
            pipe->InitBuffer(biasBuf, 2 * VM * BN * sizeof(half));
            pipe->InitBuffer(gateBuf, maxPanels * VM * BN * sizeof(half));
        }
    }

    __aicore__ inline void Process() {
        bool cPending[2] = {false, false};
        if ASCEND_IS_AIC {
            CopyWeight(0);
            PrefetchX(0);
            if (panels > 1) PrefetchX(1);
            LoadWeight();
            // L1 staging can be overwritten once the Gate weight is in L0B.
            SetFlag<HardEvent::MTE1_MTE2>(weightL1Free);
            WaitFlag<HardEvent::MTE1_MTE2>(weightL1Free);
            CopyWeight(1);
        }
        if ASCEND_IS_AIV { LoadBias(); }
        for (uint32_t projection = 0; projection < 2; ++projection) {
            if ASCEND_IS_AIC {
                if (projection == 1) {
                    SetFlag<HardEvent::M_MTE1>(weightL0Free);
                    WaitFlag<HardEvent::M_MTE1>(weightL0Free);
                    LoadWeight();
                }
                for (uint32_t panel = 0; panel < panels; ++panel) {
                    uint32_t group = (panel / 2) & 1;
                    if ((panel & 1) == 0 && cPending[group]) {
                        WaitFlag<HardEvent::FIX_M>(cFree[group]);
                        cPending[group] = false;
                    }
                    if (projection == 0) {
                        WaitFlag<HardEvent::MTE2_MTE1>(xReady[panel & 1]);
                        if (panel + 2 < panels) PrefetchX(panel + 2);
                    }
                    auto a = qA.AllocTensor<half>();
                    LoadData2DParams ld;
                    ld.repeatTimes = BK / 16; ld.srcStride = BM / 16;
                    for (uint32_t m = 0; m < BM / 16; ++m) {
                        ld.startIndex = m;
                        LoadData(a[m * BK * 16], xBuf.Get<half>()[panel * BM * BK], ld);
                    }
                    qA.EnQue(a); a = qA.DeQue<half>();
                    MmadParams mp;
                    mp.m = BM; mp.n = BN; mp.k = BK;
                    mp.cmatrixInitVal = true; mp.unitFlag = 3;
                    Mmad(cBuf.Get<float>()[(panel & 3) * BM * BN], a, bBuf.Get<half>(), mp);
                    qA.FreeTensor(a);
                    if ((panel & 1) || panel + 1 == panels) {
                        StorePair(projection, panel & ~1U, (panel & 1) ? 2 : 1);
                        SetFlag<HardEvent::FIX_M>(cFree[group]);
                        cPending[group] = true;
                    }
                    if ((panel & 1) || panel + 1 == panels) {
                        CrossCoreSetFlag<2, PIPE_FIX>(projection * 4 + ((panel / 2) & 3));
                    }
                }
            }
            if ASCEND_IS_AIV { VectorProjection(projection); }
        }
        if ASCEND_IS_AIC {
            for (uint32_t group = 0; group < 2; ++group) {
                if (cPending[group]) WaitFlag<HardEvent::FIX_M>(cFree[group]);
            }
            pipe->ReleaseEventID<HardEvent::MTE2_MTE1>(weightReady);
            pipe->ReleaseEventID<HardEvent::MTE2_MTE1>(xReady[0]);
            pipe->ReleaseEventID<HardEvent::MTE2_MTE1>(xReady[1]);
            pipe->ReleaseEventID<HardEvent::MTE1_M>(weightToM);
            pipe->ReleaseEventID<HardEvent::MTE1_MTE2>(weightL1Free);
            pipe->ReleaseEventID<HardEvent::M_MTE1>(weightL0Free);
            pipe->ReleaseEventID<HardEvent::FIX_M>(cFree[0]);
            pipe->ReleaseEventID<HardEvent::FIX_M>(cFree[1]);
        }
    }

private:
    static constexpr uint32_t BM = GATE_RESIDENT_BM;
    static constexpr uint32_t BN = GATE_RESIDENT_BN;
    static constexpr uint32_t VM = GATE_RESIDENT_VM;

    __aicore__ inline void CopyWeight(uint32_t projection) {
        Nd2NzParams cp;
        cp.ndNum = 1; cp.nValue = BN; cp.dValue = BK;
        cp.srcNdMatrixStride = 0; cp.srcDValue = BK;
        cp.dstNzC0Stride = BN; cp.dstNzNStride = 1; cp.dstNzMatrixStride = 0;
        if (projection == 0) DataCopy(wBuf.Get<half>(), w1Gm[col * BK], cp);
        else DataCopy(wBuf.Get<half>(), w2Gm[col * BK], cp);
        SetFlag<HardEvent::MTE2_MTE1>(weightReady);
    }

    __aicore__ inline void LoadWeight() {
        WaitFlag<HardEvent::MTE2_MTE1>(weightReady);
        LoadData2DParams ld;
        ld.repeatTimes = BK * BN / 256; ld.srcStride = 1;
        LoadData(bBuf.Get<half>(), wBuf.Get<half>(), ld);
        SetFlag<HardEvent::MTE1_M>(weightToM);
        WaitFlag<HardEvent::MTE1_M>(weightToM);
    }

    __aicore__ inline void PrefetchX(uint32_t panel) {
        Nd2NzParams cp;
        cp.ndNum = 1; cp.nValue = BM; cp.dValue = BK;
        cp.srcNdMatrixStride = 0; cp.srcDValue = BK;
        cp.dstNzC0Stride = BM; cp.dstNzNStride = 1; cp.dstNzMatrixStride = 0;
        DataCopy(xBuf.Get<half>()[panel * BM * BK], xGm[(firstRow + panel * BM) * BK], cp);
        SetFlag<HardEvent::MTE2_MTE1>(xReady[panel & 1]);
    }

    __aicore__ inline void StorePair(uint32_t projection, uint32_t first, uint32_t count) {
        FixpipeParamsV220 fp;
        fp.mSize = BM; fp.nSize = BN; fp.srcStride = BM; fp.dstStride = BN;
        fp.ndNum = count;
        fp.srcNdStride = count == 1 ? 0 : BM * BN / 256;
        fp.dstNdStride = count == 1 ? 0 : BM * BN;
        fp.unitFlag = 3; fp.quantPre = QuantMode_t::F322F16;
        uint32_t offset = (core * maxRows + first * BM) * BN;
        auto source = cBuf.Get<float>()[(first & 3) * BM * BN];
        if (projection == 0) Fixpipe(gateGm[offset], source, fp);
        else Fixpipe(upGm[offset], source, fp);
    }

    __aicore__ inline void LoadBias() {
        auto input = qBias.AllocTensor<half>();
        DataCopy(input, b1Gm[col], BN);
        DataCopy(input[BN], b2Gm[col], BN);
        qBias.EnQue(input); input = qBias.DeQue<half>();
        auto bias = biasBuf.Get<half>();
        for (uint32_t row = 0; row < VM; ++row) {
            DataCopy(bias[row * BN], input, BN);
            DataCopy(bias[VM * BN + row * BN], input[BN], BN);
        }
        qBias.FreeTensor(input);
    }

    __aicore__ inline void LoadInput(uint32_t projection, uint32_t panel) {
        auto input = qInput.AllocTensor<half>();
        uint32_t offset = (core * maxRows + panel * BM + GetSubBlockIdx() * VM) * BN;
        if (projection == 0) DataCopy(input, gateGm[offset], VM * BN);
        else DataCopy(input, upGm[offset], VM * BN);
        qInput.EnQue(input);
    }

    __aicore__ inline void VectorProjection(uint32_t projection) {
        auto bias = biasBuf.Get<half>()[projection * VM * BN];
        auto gate = gateBuf.Get<half>();
        for (uint32_t first = 0; first < panels; first += 2) {
            CrossCoreWaitFlag(projection * 4 + ((first / 2) & 3));
            uint32_t count = panels - first < 2 ? panels - first : 2;
            LoadInput(projection, first);
            if (count > 1) LoadInput(projection, first + 1);
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t panel = first + i;
                auto input = qInput.DeQue<half>();
                Add(input, input, bias, VM * BN);
                if (projection == 0) {
                    Silu(gate[panel * VM * BN], input, VM * BN);
                } else {
                    auto output = qOutput.AllocTensor<half>();
                    Mul(output, gate[panel * VM * BN], input, VM * BN);
                    qOutput.EnQue(output); output = qOutput.DeQue<half>();
                    uint32_t row = firstRow + panel * BM + GetSubBlockIdx() * VM;
                    DataCopyPad(yGm[row * BK + col], output,
                                {VM, BN * sizeof(half), 0, (BK - BN) * sizeof(half), 0});
                    qOutput.FreeTensor(output);
                }
                qInput.FreeTensor(input);
                if (i + 2 < count) LoadInput(projection, panel + 2);
            }
        }
    }

    TPipe *pipe;
    uint32_t core, col, panels, firstRow, maxRows;
    TEventID weightReady, xReady[2], weightToM, weightL1Free, weightL0Free, cFree[2];
    TBuf<TPosition::A1> xBuf;
    TBuf<TPosition::B1> wBuf;
    TBuf<TPosition::B2> bBuf;
    TBuf<TPosition::CO1> cBuf;
    TQue<TPosition::A2, 2> qA;
    TQue<TPosition::VECIN, 2> qInput;
    TQue<TPosition::VECOUT, 2> qOutput;
    TQue<TPosition::VECIN, 1> qBias;
    TBuf<TPosition::VECCALC> biasPad, biasBuf, gateBuf;
    GlobalTensor<half> xGm, w1Gm, w2Gm, b1Gm, b2Gm, yGm, gateGm, upGm;
};

} // namespace swiglu
