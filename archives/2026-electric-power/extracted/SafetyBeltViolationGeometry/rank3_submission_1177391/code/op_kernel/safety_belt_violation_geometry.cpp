#define K_MAX_SHAPE_DIM 0
#include "kernel_tpipe.h"
#include "kernel_tensor.h"
#include "kernel_type.h"
#include "kernel_prof_trace_intf.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_operator_common_intf.h"
#include "kernel_operator_vec_binary_intf.h"
#include "kernel_operator_vec_binary_scalar_intf.h"
#include "kernel_operator_vec_duplicate_intf.h"
#include "kernel_operator_vec_gather_mask_intf.h"
#include "kernel_operator_vec_vconv_intf.h"
#include "kernel_operator_scalar_intf.h"
#include "kernel_operator_vec_cmpsel_intf.h"
#include "kernel_operator_vec_ternary_scalar_intf.h"
#include "kernel_operator_sys_var_intf.h"
#include "kernel_operator_cache_intf.h"
#include "kernel_operator_utils_intf.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"
#include <type_traits>

// Both variants use this identical, input-dependent FP32 arithmetic pipeline.
namespace LegacyCore {
using namespace AscendC;

template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false>
struct Layout {
    static constexpr uint32_t ROWS = PLANE_ROWS;
    static constexpr uint32_t PLANE_PAD = PACKED_CAST ? 0U : 32U;
    static constexpr uint32_t OUTPUT_ROWS = std::is_same_v<T, half> && ROWS < 128U ? 128U : ROWS;
    static_assert(ROWS >= 64U && ROWS <= SBG_MAX_LOCAL_ROWS && ROWS % 64U == 0U, "invalid plane capacity");
    static_assert(!PACKED_CAST || std::is_same_v<T, half>, "packed cast is half-only");
    static constexpr uint32_t PERSON = 0U;
    static constexpr uint32_t ANCHOR = PERSON + ROWS * 4U * sizeof(T) + 32U;
    static constexpr uint32_t HALF_Y1 = ANCHOR + ROWS * 2U * sizeof(T) + 32U;
    static constexpr uint32_t HALF_Y2 = HALF_Y1 + ROWS * sizeof(T) + PLANE_PAD;
    static constexpr uint32_t HALF_CY = HALF_Y2 + ROWS * sizeof(T) + PLANE_PAD;
    static constexpr uint32_t Y1 = HALF_CY + ROWS * sizeof(T) + 32U;
    static constexpr uint32_t Y2 = Y1 + ROWS * sizeof(float) + PLANE_PAD;
    static constexpr uint32_t CY = Y2 + ROWS * sizeof(float) + PLANE_PAD;
    static constexpr uint32_t OUTPUT = CY + ROWS * sizeof(float) + 32U;
    static constexpr uint32_t MASK = OUTPUT + OUTPUT_ROWS * sizeof(T) + 32U;
    static constexpr uint32_t ZERO = MASK + ROWS + 32U;
    static constexpr uint32_t THRESHOLD = ZERO + 32U;
    static constexpr uint32_t THRESH_FLOAT = THRESHOLD + 32U;
    static constexpr uint32_t BYTES = THRESH_FLOAT + 32U;
    static_assert(BYTES < 192U * 1024U, "UB capacity exceeded");
    static_assert(BYTES % 32U == 0U, "UB layout alignment");
};

template <typename U>
__aicore__ inline LocalTensor<U> At(uint32_t offset, uint32_t count)
{
    return LocalTensor<U>(TPosition::VECCALC, offset, count);
}

template <typename T>
__aicore__ inline void Load(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                            uint32_t elements)
{
    if ((elements * sizeof(T)) % 32U == 0U) {
        DataCopy(dst, src, elements);
    } else {
        DataCopyExtParams copy{1U, elements * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src, copy, padding);
    }
}

template <typename T>
__aicore__ inline void Store(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                             uint32_t elements)
{
    if ((elements * sizeof(T)) % 32U == 0U) {
        DataCopy(dst, src, elements);
    } else {
        DataCopyExtParams copy{1U, elements * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(dst, src, copy);
    }
}

template <typename T, uint8_t PATTERN>
__aicore__ inline void GatherColumn(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                    uint16_t repeats)
{
    static_assert(PATTERN == 2U || PATTERN == 4U || PATTERN == 6U, "invalid coordinate pattern");
    using Bits = std::conditional_t<sizeof(T) == 2U, uint16_t, uint32_t>;
    auto dstPtr = reinterpret_cast<__ubuf__ Bits *>(dst.GetPhyAddr());
    auto srcPtr = reinterpret_cast<__ubuf__ Bits *>(src.GetPhyAddr());
    auto unusedPattern = srcPtr + 256U * sizeof(Bits);
    vreducev2(dstPtr, srcPtr, unusedPattern, repeats, 1, PATTERN, 8, 0);
}

template <typename T, typename L>
__aicore__ inline void StageThresholdScalar(GM_ADDR address)
{
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address), 1U);
    auto local = At<T>(L::THRESHOLD, 32U / sizeof(T));
    DataCopyExtParams copy{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
    DataCopyPadExtParams<T> padding{true, 0U, 32U / sizeof(T) - 1U, static_cast<T>(0)};
    DataCopyPad(local, gm, copy, padding);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
}

template <typename T, typename L>
__aicore__ inline float ConsumeThresholdScalar()
{
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    return static_cast<float>(At<T>(L::THRESHOLD, 32U / sizeof(T)).GetValue(0));
}

template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false,
          bool ONE_VECTOR = false>
__aicore__ inline void VectorThresholdRunTile(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
                               GM_ADDR violationScore, uint32_t firstRow, uint32_t validRows,
                               float ratio, bool hasThreshold)
{
    using L = Layout<T, PLANE_ROWS, PACKED_CAST>;
    const uint32_t workRows = ONE_VECTOR ? 64U : ((validRows + 63U) / 64U) * 64U;
    const uint8_t mathRepeats = ONE_VECTOR ? 1U : static_cast<uint8_t>(workRows / 64U);
    const uint16_t personRepeats = static_cast<uint16_t>(
        (validRows + 256U / (4U * sizeof(T)) - 1U) / (256U / (4U * sizeof(T))));
    const uint16_t anchorRepeats = static_cast<uint16_t>(
        (validRows + 256U / (2U * sizeof(T)) - 1U) / (256U / (2U * sizeof(T))));
    GlobalTensor<T> personGm;
    GlobalTensor<T> anchorGm;
    GlobalTensor<T> outputGm;
    personGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(personBoxes) + firstRow * 4U, validRows * 4U);
    anchorGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(anchorPoints) + firstRow * 2U, validRows * 2U);
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(violationScore) + firstRow, validRows);
    auto person = At<T>(L::PERSON, L::ROWS * 4U);
    auto anchor = At<T>(L::ANCHOR, L::ROWS * 2U);
    auto output = At<T>(L::OUTPUT, L::OUTPUT_ROWS);
    auto zero = At<T>(L::ZERO, 32U / sizeof(T));
    auto y1 = At<float>(L::Y1, L::ROWS);
    auto y2 = At<float>(L::Y2, L::ROWS);
    auto cy = At<float>(L::CY, L::ROWS);
    auto mask = At<uint8_t>(L::MASK, L::ROWS);

    Load(person, personGm, validRows * 4U);
    Load(anchor, anchorGm, validRows * 2U);
    if (hasThreshold) {
        GlobalTensor<T> thresholdInput;
        thresholdInput.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(threshold), 1U);
        DataCopyExtParams thresholdCopy{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> autoPadding{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(At<T>(L::THRESHOLD, 32U / sizeof(T)), thresholdInput, thresholdCopy, autoPadding);
    }
    Duplicate(output, static_cast<T>(1.0f), workRows);
    Duplicate(zero, static_cast<T>(0), 32U / sizeof(T));
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if constexpr (std::is_same_v<T, half>) {
        if (hasThreshold) {
            Cast(At<float>(L::THRESH_FLOAT, 8U), At<half>(L::THRESHOLD, 16U), RoundMode::CAST_NONE, 8U);
        }
    }
    SetMaskNorm();
    SetVectorMask<T>(0, 0);
    if constexpr (std::is_same_v<T, half>) {
        auto halfY1 = At<T>(L::HALF_Y1, L::ROWS);
        auto halfY2 = At<T>(L::HALF_Y2, L::ROWS);
        auto halfCy = At<T>(L::HALF_CY, L::ROWS);
        GatherColumn<T, 4U>(halfY1, person, personRepeats);
        GatherColumn<T, 6U>(halfY2, person, personRepeats);
        GatherColumn<T, 2U>(halfCy, anchor, anchorRepeats);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        if constexpr (PACKED_CAST) {
            static_assert(L::HALF_Y2 - L::HALF_Y1 == PLANE_ROWS * sizeof(T), "half planes must be contiguous");
            static_assert(L::Y2 - L::Y1 == PLANE_ROWS * sizeof(float), "float planes must be contiguous");
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U,
                                  3U * (PLANE_ROWS / 64U), {1, 1, 8, 4});
        } else {
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(y2, halfY2, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(cy, halfCy, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
        }
    } else {
        GatherColumn<float, 4U>(y1, person, personRepeats);
        GatherColumn<float, 6U>(y2, person, personRepeats);
        GatherColumn<float, 2U>(cy, anchor, anchorRepeats);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
    Sub<float, false>(y2, y2, y1, 64U, mathRepeats, {1, 1, 1, 8, 8, 8});
    Axpy<float, float, false>(y1, y2, ratio, 64U, mathRepeats, {1, 1, 8, 8});
    if (hasThreshold) {
        constexpr uint32_t thresholdOffset = std::is_same_v<T, half> ? L::THRESH_FLOAT : L::THRESHOLD;
        Add<float, false>(y1, y1, At<float>(thresholdOffset, 8U), 64U, mathRepeats, {1, 1, 0, 8, 8, 0});
    }
    if constexpr (ONE_VECTOR) {
        const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
        Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Select<T, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (PLANE_ROWS < SBG_MAX_LOCAL_ROWS) {
            // Capacity classes 256/320 always hold more than one 64-row block.
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ T *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<T>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        } else if (validRows <= 64U) {
            const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
            Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            Select<T, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
        } else {
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ T *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<T>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        }
    } else {
        Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        set_cmpmask(reinterpret_cast<__ubuf__ T *>(zero.GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<T>(0, workRows);
        vsel(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
             1, 1, 1, 1, 8, 8, 8, 1);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    Store(outputGm, output, validRows);
}
template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false,
          bool ONE_VECTOR = false>
__aicore__ inline void BaselineRunTile(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
                               GM_ADDR violationScore, uint32_t firstRow, uint32_t validRows,
                               float ratio, bool hasThreshold)
{
    if (hasThreshold) {
        GlobalTensor<uint64_t> thresholdCache;
        thresholdCache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(threshold), 1U);
        DataCachePreload(thresholdCache, static_cast<int16_t>(0));
    }
    using L = Layout<T, PLANE_ROWS, PACKED_CAST>;
    const uint32_t workRows = ONE_VECTOR ? 64U : ((validRows + 63U) / 64U) * 64U;
    const uint8_t mathRepeats = ONE_VECTOR ? 1U : static_cast<uint8_t>(workRows / 64U);
    const uint16_t personRepeats = static_cast<uint16_t>(
        (validRows + 256U / (4U * sizeof(T)) - 1U) / (256U / (4U * sizeof(T))));
    const uint16_t anchorRepeats = static_cast<uint16_t>(
        (validRows + 256U / (2U * sizeof(T)) - 1U) / (256U / (2U * sizeof(T))));
    GlobalTensor<T> personGm;
    GlobalTensor<T> anchorGm;
    GlobalTensor<T> outputGm;
    personGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(personBoxes) + firstRow * 4U, validRows * 4U);
    anchorGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(anchorPoints) + firstRow * 2U, validRows * 2U);
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(violationScore) + firstRow, validRows);
    auto person = At<T>(L::PERSON, L::ROWS * 4U);
    auto anchor = At<T>(L::ANCHOR, L::ROWS * 2U);
    auto output = At<T>(L::OUTPUT, L::OUTPUT_ROWS);
    auto zero = At<T>(L::ZERO, 32U / sizeof(T));
    auto y1 = At<float>(L::Y1, L::ROWS);
    auto y2 = At<float>(L::Y2, L::ROWS);
    auto cy = At<float>(L::CY, L::ROWS);
    auto mask = At<uint8_t>(L::MASK, L::ROWS);

    Load(person, personGm, validRows * 4U);
    Load(anchor, anchorGm, validRows * 2U);
    Duplicate(output, static_cast<T>(1.0f), workRows);
    Duplicate(zero, static_cast<T>(0), 32U / sizeof(T));
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetMaskNorm();
    SetVectorMask<T>(0, 0);
    if constexpr (std::is_same_v<T, half>) {
        auto halfY1 = At<T>(L::HALF_Y1, L::ROWS);
        auto halfY2 = At<T>(L::HALF_Y2, L::ROWS);
        auto halfCy = At<T>(L::HALF_CY, L::ROWS);
        GatherColumn<T, 4U>(halfY1, person, personRepeats);
        GatherColumn<T, 6U>(halfY2, person, personRepeats);
        GatherColumn<T, 2U>(halfCy, anchor, anchorRepeats);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        if constexpr (PACKED_CAST) {
            static_assert(L::HALF_Y2 - L::HALF_Y1 == PLANE_ROWS * sizeof(T), "half planes must be contiguous");
            static_assert(L::Y2 - L::Y1 == PLANE_ROWS * sizeof(float), "float planes must be contiguous");
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U,
                                  3U * (PLANE_ROWS / 64U), {1, 1, 8, 4});
        } else {
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(y2, halfY2, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(cy, halfCy, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
        }
    } else {
        GatherColumn<float, 4U>(y1, person, personRepeats);
        GatherColumn<float, 6U>(y2, person, personRepeats);
        GatherColumn<float, 2U>(cy, anchor, anchorRepeats);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
    Sub<float, false>(y2, y2, y1, 64U, mathRepeats, {1, 1, 1, 8, 8, 8});
    Axpy<float, float, false>(y1, y2, ratio, 64U, mathRepeats, {1, 1, 8, 8});
    if (hasThreshold) {
        const float thresholdValue = static_cast<float>(*reinterpret_cast<const __gm__ T *>(threshold));
        if (thresholdValue != 0.0f) {
            Adds<float, false>(y1, y1, thresholdValue, 64U, mathRepeats, {1, 1, 8, 8});
        }
    }
    if constexpr (ONE_VECTOR) {
        const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
        Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Select<T, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (PLANE_ROWS < SBG_MAX_LOCAL_ROWS) {
            // Capacity classes 256/320 always hold more than one 64-row block.
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ T *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<T>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        } else if (validRows <= 64U) {
            const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
            Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            Select<T, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
        } else {
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ T *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<T>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        }
    } else {
        Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        set_cmpmask(reinterpret_cast<__ubuf__ T *>(zero.GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<T>(0, workRows);
        vsel(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
             1, 1, 1, 1, 8, 8, 8, 1);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    Store(outputGm, output, validRows);
}

template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false, bool ONE_VECTOR = false>
__aicore__ inline void RunTile(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
    GM_ADDR violationScore, uint32_t firstRow, uint32_t validRows, float ratio, bool hasThreshold) {
    if constexpr (std::is_same_v<T, half> && (PLANE_ROWS == 256U || PLANE_ROWS == 320U)) {
        VectorThresholdRunTile<T,PLANE_ROWS,PACKED_CAST,ONE_VECTOR>(personBoxes,anchorPoints,threshold,violationScore,firstRow,validRows,ratio,hasThreshold);
    } else {
        BaselineRunTile<T,PLANE_ROWS,PACKED_CAST,ONE_VECTOR>(personBoxes,anchorPoints,threshold,violationScore,firstRow,validRows,ratio,hasThreshold);
    }
}
}  // namespace LegacyCore



namespace LegacySmall {
using namespace AscendC;
using LegacyCore::At;
using LegacyCore::GatherColumn;

template <typename T, uint32_t N, bool HAS_THRESHOLD, uint32_t RATIO_ONE>
__aicore__ inline void Run(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR thresholdGm,
                         GM_ADDR outputGm, GM_ADDR tiling)
{
    static_assert(N >= 1U && N <= 64U, "invalid small-domain row count");
    if (get_block_idx() != 0 || get_subblockid() != 0) {
        return;
    }
    using L = LegacyCore::Layout<T, 64U, std::is_same_v<T, half>>;
    GlobalTensor<T> person, anchor, out;
    person.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(personBoxes), N * 4U);
    anchor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(anchorPoints), N * 2U);
    out.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outputGm), N);
    auto personUb = At<T>(L::PERSON, 256U);
    auto anchorUb = At<T>(L::ANCHOR, 128U);
    auto output = At<T>(L::OUTPUT, L::OUTPUT_ROWS);
    auto zero = At<T>(L::ZERO, 32U / sizeof(T));
    auto y1 = At<float>(L::Y1, 64U);
    auto y2 = At<float>(L::Y2, 64U);
    auto cy = At<float>(L::CY, 64U);
    LegacyCore::Load(personUb, person, N * 4U);
    LegacyCore::Load(anchorUb, anchor, N * 2U);
    if constexpr (HAS_THRESHOLD) {
        LegacyCore::StageThresholdScalar<T, L>(thresholdGm);
    }
    Duplicate(output, static_cast<T>(1.0f), 64U);
    Duplicate(zero, static_cast<T>(0), 32U / sizeof(T));
    float ratio = RATIO_ONE == 2U ? 0.8f : (RATIO_ONE == 3U ? 0.9f : (RATIO_ONE == 4U ? 0.5f : 1.0f));
    if constexpr (RATIO_ONE == 0U) {
        // Length is already known; issue both input DMAs before this scalar load.
        ratio = *reinterpret_cast<const __gm__ volatile float *>(tiling + 4U);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetMaskNorm();
    SetVectorMask<T>(0, 0);
    constexpr uint16_t personRepeats = (N + 256U / (4U * sizeof(T)) - 1U) / (256U / (4U * sizeof(T)));
    constexpr uint16_t anchorRepeats = (N + 256U / (2U * sizeof(T)) - 1U) / (256U / (2U * sizeof(T)));
    if constexpr (std::is_same_v<T, half>) {
        auto h1 = At<half>(L::HALF_Y1, 64U);
        auto h2 = At<half>(L::HALF_Y2, 64U);
        auto hc = At<half>(L::HALF_CY, 64U);
        GatherColumn<half, 4U>(h1, personUb, personRepeats);
        GatherColumn<half, 6U>(h2, personUb, personRepeats);
        GatherColumn<half, 2U>(hc, anchorUb, anchorRepeats);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        Cast<float, half, false>(y1, h1, RoundMode::CAST_NONE, 64U, 3U, {1, 1, 8, 4});
    } else {
        GatherColumn<float, 4U>(y1, personUb, personRepeats);
        GatherColumn<float, 6U>(y2, personUb, personRepeats);
        GatherColumn<float, 2U>(cy, anchorUb, anchorRepeats);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
    Sub<float, false>(y2, y2, y1, 64U, 1U, {1, 1, 1, 8, 8, 8});
    if constexpr (RATIO_ONE == 1U) {
        Add<float, false>(y1, y1, y2, 64U, 1U, {1, 1, 1, 8, 8, 8});
    } else {
        Axpy<float, float, false>(y1, y2, ratio, 64U, 1U, {1, 1, 8, 8});
    }
    if constexpr (HAS_THRESHOLD) {
        const float threshold = LegacyCore::ConsumeThresholdScalar<T, L>();
        if (threshold != 0.0f) {
            Adds<float, false>(y1, y1, threshold, 64U, 1U, {1, 1, 8, 8});
        }
    }
    const uint64_t mask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
    Compare(cy, y1, CMPMODE::GT, mask, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Select<T, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    LegacyCore::Store(out, output, N);
}
}

namespace StrictCore {
using namespace AscendC;

template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false>
struct Layout {
    static constexpr uint32_t ROWS = PLANE_ROWS;
    static constexpr uint32_t PLANE_PAD = PACKED_CAST ? 0U : 32U;
    static constexpr uint32_t OUTPUT_ROWS = std::is_same_v<T, half> && ROWS < 128U ? 128U : ROWS;
    static_assert(ROWS >= 64U && ROWS <= SBG_MAX_LOCAL_ROWS && ROWS % 64U == 0U, "invalid plane capacity");
    static_assert(!PACKED_CAST || std::is_same_v<T, half>, "packed cast is half-only");
    static constexpr uint32_t PERSON = 0U;
    static constexpr uint32_t ANCHOR = PERSON + ROWS * 4U * sizeof(T) + 32U;
    static constexpr uint32_t HALF_Y1 = ANCHOR + ROWS * 2U * sizeof(T) + 32U;
    static constexpr uint32_t HALF_Y2 = HALF_Y1 + ROWS * sizeof(T) + PLANE_PAD;
    static constexpr uint32_t HALF_CY = HALF_Y2 + ROWS * sizeof(T) + PLANE_PAD;
    static constexpr uint32_t Y1 = HALF_CY + ROWS * sizeof(T) + 32U;
    static constexpr uint32_t Y2 = Y1 + ROWS * sizeof(float) + PLANE_PAD;
    static constexpr uint32_t CY = Y2 + ROWS * sizeof(float) + PLANE_PAD;
    static constexpr uint32_t OUTPUT = CY + ROWS * sizeof(float) + 32U;
    static constexpr uint32_t MASK = OUTPUT + OUTPUT_ROWS * sizeof(float) + 32U;
    static constexpr uint32_t ZERO = MASK + ROWS + 32U;
    static constexpr uint32_t THRESHOLD = ZERO + 32U;
    static constexpr uint32_t THRESH_FLOAT = THRESHOLD + 32U;
    static constexpr uint32_t BYTES = THRESH_FLOAT + 32U;
    static_assert(BYTES < 192U * 1024U, "UB capacity exceeded");
    static_assert(BYTES % 32U == 0U, "UB layout alignment");
};

template <typename U>
__aicore__ inline LocalTensor<U> At(uint32_t offset, uint32_t count)
{
    return LocalTensor<U>(TPosition::VECCALC, offset, count);
}

template <typename T>
__aicore__ inline void Load(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                            uint32_t elements)
{
    if ((elements * sizeof(T)) % 32U == 0U) {
        DataCopy(dst, src, elements);
    } else {
        DataCopyExtParams copy{1U, elements * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src, copy, padding);
    }
}

template <typename T>
__aicore__ inline void Store(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                             uint32_t elements)
{
    if ((elements * sizeof(T)) % 32U == 0U) {
        DataCopy(dst, src, elements);
    } else {
        DataCopyExtParams copy{1U, elements * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(dst, src, copy);
    }
}

template <typename T, uint8_t PATTERN>
__aicore__ inline void GatherColumn(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                    uint16_t repeats)
{
    static_assert(PATTERN == 2U || PATTERN == 4U || PATTERN == 6U, "invalid coordinate pattern");
    using Bits = std::conditional_t<sizeof(T) == 2U, uint16_t, uint32_t>;
    auto dstPtr = reinterpret_cast<__ubuf__ Bits *>(dst.GetPhyAddr());
    auto srcPtr = reinterpret_cast<__ubuf__ Bits *>(src.GetPhyAddr());
    auto unusedPattern = srcPtr + 256U * sizeof(Bits);
    vreducev2(dstPtr, srcPtr, unusedPattern, repeats, 1, PATTERN, 8, 0);
}

template <typename T, typename L>
__aicore__ inline void StageThresholdScalar(GM_ADDR address)
{
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address), 1U);
    auto local = At<float>(L::THRESHOLD, 8U);
    DataCopyExtParams copy{1U, 4U, 0U, 0U, 0U};
    DataCopyPadExtParams<float> padding{true, 0U, 7U, 0.0f};
    DataCopyPad(local, gm, copy, padding);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
}

template <typename T, typename L>
__aicore__ inline float ConsumeThresholdScalar()
{
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    return At<float>(L::THRESHOLD, 8U).GetValue(0);
}

template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false,
          bool ONE_VECTOR = false>
__aicore__ inline void VectorThresholdRunTile(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
                               GM_ADDR violationScore, uint32_t firstRow, uint32_t validRows,
                               float ratio, bool hasThreshold)
{
    using L = Layout<T, PLANE_ROWS, PACKED_CAST>;
    const uint32_t workRows = ONE_VECTOR ? 64U : ((validRows + 63U) / 64U) * 64U;
    const uint8_t mathRepeats = ONE_VECTOR ? 1U : static_cast<uint8_t>(workRows / 64U);
    const uint16_t personRepeats = static_cast<uint16_t>(
        (validRows + 256U / (4U * sizeof(T)) - 1U) / (256U / (4U * sizeof(T))));
    const uint16_t anchorRepeats = static_cast<uint16_t>(
        (validRows + 256U / (2U * sizeof(T)) - 1U) / (256U / (2U * sizeof(T))));
    GlobalTensor<T> personGm;
    GlobalTensor<T> anchorGm;
    GlobalTensor<float> outputGm;
    personGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(personBoxes) + firstRow * 4U, validRows * 4U);
    anchorGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(anchorPoints) + firstRow * 2U, validRows * 2U);
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(violationScore) + firstRow, validRows);
    auto person = At<T>(L::PERSON, L::ROWS * 4U);
    auto anchor = At<T>(L::ANCHOR, L::ROWS * 2U);
    auto output = At<float>(L::OUTPUT, L::OUTPUT_ROWS);
    auto zero = At<float>(L::ZERO, 8U);
    auto y1 = At<float>(L::Y1, L::ROWS);
    auto y2 = At<float>(L::Y2, L::ROWS);
    auto cy = At<float>(L::CY, L::ROWS);
    auto mask = At<uint8_t>(L::MASK, L::ROWS);

    Load(person, personGm, validRows * 4U);
    Load(anchor, anchorGm, validRows * 2U);
    if (hasThreshold) {
        GlobalTensor<float> thresholdInput;
        thresholdInput.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(threshold), 1U);
        DataCopyExtParams thresholdCopy{1U, 4U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> autoPadding{false, 0U, 0U, 0.0f};
        DataCopyPad(At<float>(L::THRESHOLD, 8U), thresholdInput, thresholdCopy, autoPadding);
    }
    Duplicate(output, 1.0f, workRows);
    Duplicate(zero, 0.0f, 8U);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetMaskNorm();
    SetVectorMask<T>(0, 0);
    if constexpr (std::is_same_v<T, half>) {
        auto halfY1 = At<T>(L::HALF_Y1, L::ROWS);
        auto halfY2 = At<T>(L::HALF_Y2, L::ROWS);
        auto halfCy = At<T>(L::HALF_CY, L::ROWS);
        GatherColumn<T, 4U>(halfY1, person, personRepeats);
        GatherColumn<T, 6U>(halfY2, person, personRepeats);
        GatherColumn<T, 2U>(halfCy, anchor, anchorRepeats);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        if constexpr (PACKED_CAST) {
            static_assert(L::HALF_Y2 - L::HALF_Y1 == PLANE_ROWS * sizeof(T), "half planes must be contiguous");
            static_assert(L::Y2 - L::Y1 == PLANE_ROWS * sizeof(float), "float planes must be contiguous");
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U,
                                  3U * (PLANE_ROWS / 64U), {1, 1, 8, 4});
        } else {
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(y2, halfY2, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(cy, halfCy, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
        }
    } else {
        GatherColumn<float, 4U>(y1, person, personRepeats);
        GatherColumn<float, 6U>(y2, person, personRepeats);
        GatherColumn<float, 2U>(cy, anchor, anchorRepeats);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
    Sub<float, false>(y2, y2, y1, 64U, mathRepeats, {1, 1, 1, 8, 8, 8});
    Axpy<float, float, false>(y1, y2, ratio, 64U, mathRepeats, {1, 1, 8, 8});
    if (hasThreshold) {
        constexpr uint32_t thresholdOffset = L::THRESHOLD;
        Add<float, false>(y1, y1, At<float>(thresholdOffset, 8U), 64U, mathRepeats, {1, 1, 0, 8, 8, 0});
    }
    if constexpr (ONE_VECTOR) {
        const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
        Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Select<float, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (PLANE_ROWS < SBG_MAX_LOCAL_ROWS) {
            // Capacity classes 256/320 always hold more than one 64-row block.
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ float *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<float>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        } else if (validRows <= 64U) {
            const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
            Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            Select<float, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
        } else {
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ float *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<float>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        }
    } else {
        Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        set_cmpmask(reinterpret_cast<__ubuf__ float *>(zero.GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float>(0, workRows);
        vsel(reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
             1, 1, 1, 1, 8, 8, 8, 1);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    Store(outputGm, output, validRows);
}
template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false,
          bool ONE_VECTOR = false>
__aicore__ inline void BaselineRunTile(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
                               GM_ADDR violationScore, uint32_t firstRow, uint32_t validRows,
                               float ratio, bool hasThreshold)
{
    if (hasThreshold) {
        GlobalTensor<uint64_t> thresholdCache;
        thresholdCache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(threshold), 1U);
        DataCachePreload(thresholdCache, static_cast<int16_t>(0));
    }
    using L = Layout<T, PLANE_ROWS, PACKED_CAST>;
    const uint32_t workRows = ONE_VECTOR ? 64U : ((validRows + 63U) / 64U) * 64U;
    const uint8_t mathRepeats = ONE_VECTOR ? 1U : static_cast<uint8_t>(workRows / 64U);
    const uint16_t personRepeats = static_cast<uint16_t>(
        (validRows + 256U / (4U * sizeof(T)) - 1U) / (256U / (4U * sizeof(T))));
    const uint16_t anchorRepeats = static_cast<uint16_t>(
        (validRows + 256U / (2U * sizeof(T)) - 1U) / (256U / (2U * sizeof(T))));
    GlobalTensor<T> personGm;
    GlobalTensor<T> anchorGm;
    GlobalTensor<float> outputGm;
    personGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(personBoxes) + firstRow * 4U, validRows * 4U);
    anchorGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(anchorPoints) + firstRow * 2U, validRows * 2U);
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(violationScore) + firstRow, validRows);
    auto person = At<T>(L::PERSON, L::ROWS * 4U);
    auto anchor = At<T>(L::ANCHOR, L::ROWS * 2U);
    auto output = At<float>(L::OUTPUT, L::OUTPUT_ROWS);
    auto zero = At<float>(L::ZERO, 8U);
    auto y1 = At<float>(L::Y1, L::ROWS);
    auto y2 = At<float>(L::Y2, L::ROWS);
    auto cy = At<float>(L::CY, L::ROWS);
    auto mask = At<uint8_t>(L::MASK, L::ROWS);

    Load(person, personGm, validRows * 4U);
    Load(anchor, anchorGm, validRows * 2U);
    Duplicate(output, 1.0f, workRows);
    Duplicate(zero, 0.0f, 8U);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetMaskNorm();
    SetVectorMask<T>(0, 0);
    if constexpr (std::is_same_v<T, half>) {
        auto halfY1 = At<T>(L::HALF_Y1, L::ROWS);
        auto halfY2 = At<T>(L::HALF_Y2, L::ROWS);
        auto halfCy = At<T>(L::HALF_CY, L::ROWS);
        GatherColumn<T, 4U>(halfY1, person, personRepeats);
        GatherColumn<T, 6U>(halfY2, person, personRepeats);
        GatherColumn<T, 2U>(halfCy, anchor, anchorRepeats);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        if constexpr (PACKED_CAST) {
            static_assert(L::HALF_Y2 - L::HALF_Y1 == PLANE_ROWS * sizeof(T), "half planes must be contiguous");
            static_assert(L::Y2 - L::Y1 == PLANE_ROWS * sizeof(float), "float planes must be contiguous");
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U,
                                  3U * (PLANE_ROWS / 64U), {1, 1, 8, 4});
        } else {
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(y2, halfY2, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(cy, halfCy, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
        }
    } else {
        GatherColumn<float, 4U>(y1, person, personRepeats);
        GatherColumn<float, 6U>(y2, person, personRepeats);
        GatherColumn<float, 2U>(cy, anchor, anchorRepeats);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
    Sub<float, false>(y2, y2, y1, 64U, mathRepeats, {1, 1, 1, 8, 8, 8});
    Axpy<float, float, false>(y1, y2, ratio, 64U, mathRepeats, {1, 1, 8, 8});
    if (hasThreshold) {
        const float thresholdValue = *reinterpret_cast<const __gm__ float *>(threshold);
        if (thresholdValue != 0.0f) {
            Adds<float, false>(y1, y1, thresholdValue, 64U, mathRepeats, {1, 1, 8, 8});
        }
    }
    if constexpr (ONE_VECTOR) {
        const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
        Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Select<float, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (PLANE_ROWS < SBG_MAX_LOCAL_ROWS) {
            // Capacity classes 256/320 always hold more than one 64-row block.
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ float *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<float>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        } else if (validRows <= 64U) {
            const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
            Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            Select<float, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
        } else {
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ float *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<float>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        }
    } else {
        Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        set_cmpmask(reinterpret_cast<__ubuf__ float *>(zero.GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float>(0, workRows);
        vsel(reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
             1, 1, 1, 1, 8, 8, 8, 1);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    Store(outputGm, output, validRows);
}

template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false, bool ONE_VECTOR = false>
__aicore__ inline void RunTile(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
    GM_ADDR violationScore, uint32_t firstRow, uint32_t validRows, float ratio, bool hasThreshold) {
    if constexpr (std::is_same_v<T, half> && (PLANE_ROWS == 256U || PLANE_ROWS == 320U)) {
        VectorThresholdRunTile<T,PLANE_ROWS,PACKED_CAST,ONE_VECTOR>(personBoxes,anchorPoints,threshold,violationScore,firstRow,validRows,ratio,hasThreshold);
    } else {
        BaselineRunTile<T,PLANE_ROWS,PACKED_CAST,ONE_VECTOR>(personBoxes,anchorPoints,threshold,violationScore,firstRow,validRows,ratio,hasThreshold);
    }
}
}  // namespace StrictCore



namespace StrictSmall {
using namespace AscendC;
using StrictCore::At;
using StrictCore::GatherColumn;

template <typename T, uint32_t N, bool HAS_THRESHOLD, uint32_t RATIO_ONE>
__aicore__ inline void Run(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR thresholdGm,
                         GM_ADDR outputGm, GM_ADDR tiling)
{
    static_assert(N >= 1U && N <= 64U, "invalid small-domain row count");
    if (get_block_idx() != 0 || get_subblockid() != 0) {
        return;
    }
    using L = StrictCore::Layout<T, 64U, std::is_same_v<T, half>>;
    GlobalTensor<T> person, anchor;
    GlobalTensor<float> out;
    person.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(personBoxes), N * 4U);
    anchor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(anchorPoints), N * 2U);
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputGm), N);
    auto personUb = At<T>(L::PERSON, 256U);
    auto anchorUb = At<T>(L::ANCHOR, 128U);
    auto output = At<float>(L::OUTPUT, L::OUTPUT_ROWS);
    auto zero = At<float>(L::ZERO, 8U);
    auto y1 = At<float>(L::Y1, 64U);
    auto y2 = At<float>(L::Y2, 64U);
    auto cy = At<float>(L::CY, 64U);
    // For every Float32 small batch, overlap threshold cache fill with input DMAs.
    if constexpr (HAS_THRESHOLD && std::is_same_v<T, float>) {
        GlobalTensor<uint64_t> thresholdCache;
        thresholdCache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(thresholdGm), 1U);
        DataCachePreload(thresholdCache, static_cast<int16_t>(0));
    }
    StrictCore::Load(personUb, person, N * 4U);
    StrictCore::Load(anchorUb, anchor, N * 2U);
    if constexpr (HAS_THRESHOLD && std::is_same_v<T, half>) {
        StrictCore::StageThresholdScalar<T, L>(thresholdGm);
    }
    Duplicate(output, 1.0f, 64U);
    Duplicate(zero, 0.0f, 8U);
    float ratio = RATIO_ONE == 2U ? 0.8f : (RATIO_ONE == 3U ? 0.9f : (RATIO_ONE == 4U ? 0.5f : 1.0f));
    if constexpr (RATIO_ONE == 0U) {
        // Length is already known; issue both input DMAs before this scalar load.
        ratio = *reinterpret_cast<const __gm__ volatile float *>(tiling + 4U);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetMaskNorm();
    SetVectorMask<T>(0, 0);
    constexpr uint16_t personRepeats = (N + 256U / (4U * sizeof(T)) - 1U) / (256U / (4U * sizeof(T)));
    constexpr uint16_t anchorRepeats = (N + 256U / (2U * sizeof(T)) - 1U) / (256U / (2U * sizeof(T)));
    if constexpr (std::is_same_v<T, half>) {
        auto h1 = At<half>(L::HALF_Y1, 64U);
        auto h2 = At<half>(L::HALF_Y2, 64U);
        auto hc = At<half>(L::HALF_CY, 64U);
        GatherColumn<half, 4U>(h1, personUb, personRepeats);
        GatherColumn<half, 6U>(h2, personUb, personRepeats);
        GatherColumn<half, 2U>(hc, anchorUb, anchorRepeats);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        Cast<float, half, false>(y1, h1, RoundMode::CAST_NONE, 64U, 3U, {1, 1, 8, 4});
    } else {
        GatherColumn<float, 4U>(y1, personUb, personRepeats);
        GatherColumn<float, 6U>(y2, personUb, personRepeats);
        GatherColumn<float, 2U>(cy, anchorUb, anchorRepeats);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
    Sub<float, false>(y2, y2, y1, 64U, 1U, {1, 1, 1, 8, 8, 8});
    if constexpr (RATIO_ONE == 1U) {
        Add<float, false>(y1, y1, y2, 64U, 1U, {1, 1, 1, 8, 8, 8});
    } else {
        Axpy<float, float, false>(y1, y2, ratio, 64U, 1U, {1, 1, 8, 8});
    }
    if constexpr (HAS_THRESHOLD) {
        float threshold;
        if constexpr (std::is_same_v<T, float>) {
            threshold = *reinterpret_cast<const __gm__ float *>(thresholdGm);
        } else {
            threshold = StrictCore::ConsumeThresholdScalar<T, L>();
        }
        if (threshold != 0.0f) {
            Adds<float, false>(y1, y1, threshold, 64U, 1U, {1, 1, 8, 8});
        }
    }
    const uint64_t mask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
    Compare(cy, y1, CMPMODE::GT, mask, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Select<float, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    StrictCore::Store(out, output, N);
}
}


namespace OverlapLegacy {
using namespace LegacyCore;
template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false,
          bool ONE_VECTOR = false>
__aicore__ inline void BaselineRunTile(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
                               GM_ADDR violationScore, uint32_t firstRow, uint32_t validRows,
                               float ratio, bool hasThreshold)
{
    if (hasThreshold) {
        GlobalTensor<uint64_t> thresholdCache;
        thresholdCache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(threshold), 1U);
        DataCachePreload(thresholdCache, static_cast<int16_t>(0));
    }
    using L = LegacyCore::Layout<T, PLANE_ROWS, PACKED_CAST>;
    const uint32_t workRows = ONE_VECTOR ? 64U : ((validRows + 63U) / 64U) * 64U;
    const uint8_t mathRepeats = ONE_VECTOR ? 1U : static_cast<uint8_t>(workRows / 64U);
    const uint16_t personRepeats = static_cast<uint16_t>(
        (validRows + 256U / (4U * sizeof(T)) - 1U) / (256U / (4U * sizeof(T))));
    const uint16_t anchorRepeats = static_cast<uint16_t>(
        (validRows + 256U / (2U * sizeof(T)) - 1U) / (256U / (2U * sizeof(T))));
    GlobalTensor<T> personGm;
    GlobalTensor<T> anchorGm;
    GlobalTensor<T> outputGm;
    personGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(personBoxes) + firstRow * 4U, validRows * 4U);
    anchorGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(anchorPoints) + firstRow * 2U, validRows * 2U);
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(violationScore) + firstRow, validRows);
    auto person = At<T>(L::PERSON, L::ROWS * 4U);
    auto anchor = At<T>(L::ANCHOR, L::ROWS * 2U);
    auto output = At<T>(L::OUTPUT, L::OUTPUT_ROWS);
    auto zero = At<T>(L::ZERO, 32U / sizeof(T));
    auto y1 = At<float>(L::Y1, L::ROWS);
    auto y2 = At<float>(L::Y2, L::ROWS);
    auto cy = At<float>(L::CY, L::ROWS);
    auto mask = At<uint8_t>(L::MASK, L::ROWS);

    Load(person, personGm, validRows * 4U);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Load(anchor, anchorGm, validRows * 2U);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Duplicate(output, static_cast<T>(1.0f), workRows);
    Duplicate(zero, static_cast<T>(0), 32U / sizeof(T));
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetMaskNorm();
    SetVectorMask<T>(0, 0);
    if constexpr (std::is_same_v<T, half>) {
        auto halfY1 = At<T>(L::HALF_Y1, L::ROWS);
        auto halfY2 = At<T>(L::HALF_Y2, L::ROWS);
        auto halfCy = At<T>(L::HALF_CY, L::ROWS);
        GatherColumn<T, 4U>(halfY1, person, personRepeats);
        GatherColumn<T, 6U>(halfY2, person, personRepeats);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        GatherColumn<T, 2U>(halfCy, anchor, anchorRepeats);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        if constexpr (PACKED_CAST) {
            static_assert(L::HALF_Y2 - L::HALF_Y1 == PLANE_ROWS * sizeof(T), "half planes must be contiguous");
            static_assert(L::Y2 - L::Y1 == PLANE_ROWS * sizeof(float), "float planes must be contiguous");
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U,
                                  3U * (PLANE_ROWS / 64U), {1, 1, 8, 4});
        } else {
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(y2, halfY2, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(cy, halfCy, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
        }
    } else {
        GatherColumn<float, 4U>(y1, person, personRepeats);
        GatherColumn<float, 6U>(y2, person, personRepeats);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        GatherColumn<float, 2U>(cy, anchor, anchorRepeats);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
    Sub<float, false>(y2, y2, y1, 64U, mathRepeats, {1, 1, 1, 8, 8, 8});
    Axpy<float, float, false>(y1, y2, ratio, 64U, mathRepeats, {1, 1, 8, 8});
    if (hasThreshold) {
        const float thresholdValue = static_cast<float>(*reinterpret_cast<const __gm__ T *>(threshold));
        if (thresholdValue != 0.0f) {
            Adds<float, false>(y1, y1, thresholdValue, 64U, mathRepeats, {1, 1, 8, 8});
        }
    }
    if constexpr (ONE_VECTOR) {
        const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
        Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Select<T, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (PLANE_ROWS < SBG_MAX_LOCAL_ROWS) {
            // Capacity classes 256/320 always hold more than one 64-row block.
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ T *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<T>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        } else if (validRows <= 64U) {
            const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
            Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            Select<T, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
        } else {
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ T *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<T>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        }
    } else {
        Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        set_cmpmask(reinterpret_cast<__ubuf__ T *>(zero.GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<T>(0, workRows);
        vsel(reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ T *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
             1, 1, 1, 1, 8, 8, 8, 1);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    Store(outputGm, output, validRows);
}

}

namespace OverlapStrict {
using namespace StrictCore;
template <typename T, uint32_t PLANE_ROWS = SBG_MAX_LOCAL_ROWS, bool PACKED_CAST = false,
          bool ONE_VECTOR = false>
__aicore__ inline void BaselineRunTile(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
                               GM_ADDR violationScore, uint32_t firstRow, uint32_t validRows,
                               float ratio, bool hasThreshold)
{
    if (hasThreshold) {
        GlobalTensor<uint64_t> thresholdCache;
        thresholdCache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(threshold), 1U);
        DataCachePreload(thresholdCache, static_cast<int16_t>(0));
    }
    using L = StrictCore::Layout<T, PLANE_ROWS, PACKED_CAST>;
    const uint32_t workRows = ONE_VECTOR ? 64U : ((validRows + 63U) / 64U) * 64U;
    const uint8_t mathRepeats = ONE_VECTOR ? 1U : static_cast<uint8_t>(workRows / 64U);
    const uint16_t personRepeats = static_cast<uint16_t>(
        (validRows + 256U / (4U * sizeof(T)) - 1U) / (256U / (4U * sizeof(T))));
    const uint16_t anchorRepeats = static_cast<uint16_t>(
        (validRows + 256U / (2U * sizeof(T)) - 1U) / (256U / (2U * sizeof(T))));
    GlobalTensor<T> personGm;
    GlobalTensor<T> anchorGm;
    GlobalTensor<float> outputGm;
    personGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(personBoxes) + firstRow * 4U, validRows * 4U);
    anchorGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(anchorPoints) + firstRow * 2U, validRows * 2U);
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(violationScore) + firstRow, validRows);
    auto person = At<T>(L::PERSON, L::ROWS * 4U);
    auto anchor = At<T>(L::ANCHOR, L::ROWS * 2U);
    auto output = At<float>(L::OUTPUT, L::OUTPUT_ROWS);
    auto zero = At<float>(L::ZERO, 8U);
    auto y1 = At<float>(L::Y1, L::ROWS);
    auto y2 = At<float>(L::Y2, L::ROWS);
    auto cy = At<float>(L::CY, L::ROWS);
    auto mask = At<uint8_t>(L::MASK, L::ROWS);

    Load(person, personGm, validRows * 4U);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Load(anchor, anchorGm, validRows * 2U);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Duplicate(output, 1.0f, workRows);
    Duplicate(zero, 0.0f, 8U);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetMaskNorm();
    SetVectorMask<T>(0, 0);
    if constexpr (std::is_same_v<T, half>) {
        auto halfY1 = At<T>(L::HALF_Y1, L::ROWS);
        auto halfY2 = At<T>(L::HALF_Y2, L::ROWS);
        auto halfCy = At<T>(L::HALF_CY, L::ROWS);
        GatherColumn<T, 4U>(halfY1, person, personRepeats);
        GatherColumn<T, 6U>(halfY2, person, personRepeats);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        GatherColumn<T, 2U>(halfCy, anchor, anchorRepeats);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        if constexpr (PACKED_CAST) {
            static_assert(L::HALF_Y2 - L::HALF_Y1 == PLANE_ROWS * sizeof(T), "half planes must be contiguous");
            static_assert(L::Y2 - L::Y1 == PLANE_ROWS * sizeof(float), "float planes must be contiguous");
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U,
                                  3U * (PLANE_ROWS / 64U), {1, 1, 8, 4});
        } else {
            Cast<float, T, false>(y1, halfY1, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(y2, halfY2, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
            Cast<float, T, false>(cy, halfCy, RoundMode::CAST_NONE, 64U, mathRepeats, {1, 1, 8, 4});
        }
    } else {
        GatherColumn<float, 4U>(y1, person, personRepeats);
        GatherColumn<float, 6U>(y2, person, personRepeats);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        GatherColumn<float, 2U>(cy, anchor, anchorRepeats);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
    Sub<float, false>(y2, y2, y1, 64U, mathRepeats, {1, 1, 1, 8, 8, 8});
    Axpy<float, float, false>(y1, y2, ratio, 64U, mathRepeats, {1, 1, 8, 8});
    if (hasThreshold) {
        const float thresholdValue = *reinterpret_cast<const __gm__ float *>(threshold);
        if (thresholdValue != 0.0f) {
            Adds<float, false>(y1, y1, thresholdValue, 64U, mathRepeats, {1, 1, 8, 8});
        }
    }
    if constexpr (ONE_VECTOR) {
        const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
        Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Select<float, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (PLANE_ROWS < SBG_MAX_LOCAL_ROWS) {
            // Capacity classes 256/320 always hold more than one 64-row block.
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ float *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<float>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        } else if (validRows <= 64U) {
            const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
            Compare(cy, y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            Select<float, SELMODE::VSEL_CMPMASK_SPR>(output, output, zero, 1U, {1, 1, 0, 8, 8, 0});
        } else {
            Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                    {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            set_cmpmask(reinterpret_cast<__ubuf__ float *>(zero.GetPhyAddr()));
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<float>(0, workRows);
            vsel(reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, 1);
        }
    } else {
        Compare(mask, cy, y1, CMPMODE::GT, static_cast<uint64_t>(64U), mathRepeats,
                {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        set_cmpmask(reinterpret_cast<__ubuf__ float *>(zero.GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float>(0, workRows);
        vsel(reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ float *>(output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ uint8_t *>(mask.GetPhyAddr()),
             1, 1, 1, 1, 8, 8, 8, 1);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    Store(outputGm, output, validRows);
}

}

namespace DualIO {
using namespace AscendC;
template <typename In, typename Out, uint32_t Rows, bool Packed, bool One, bool EarlyInput = false>
__aicore__ inline void RunTile(GM_ADDR p, GM_ADDR a, GM_ADDR t, GM_ADDR o,
                              uint32_t first, uint32_t valid, float ratio, bool hasThreshold)
{
    if constexpr (std::is_same_v<Out, half>) {
        static_assert(std::is_same_v<In, half>, "legacy output requires half coordinates");
        if constexpr (EarlyInput) {
            static_assert(Rows == SBG_MAX_LOCAL_ROWS && !One, "full capacity overlap path");
            OverlapLegacy::BaselineRunTile<In,Rows,Packed,One>(p,a,t,o,first,valid,ratio,hasThreshold);
        } else {
            LegacyCore::RunTile<In,Rows,Packed,One>(p,a,t,o,first,valid,ratio,hasThreshold);
        }
    } else {
        if constexpr (EarlyInput) {
            static_assert(Rows == SBG_MAX_LOCAL_ROWS && !One, "full capacity overlap path");
            OverlapStrict::BaselineRunTile<In,Rows,Packed,One>(p,a,t,o,first,valid,ratio,hasThreshold);
        } else {
            StrictCore::RunTile<In,Rows,Packed,One>(p,a,t,o,first,valid,ratio,hasThreshold);
        }
    }
}
}

using namespace AscendC;

template <typename DT_PERSON_BOXES, typename DT_VIOLATION_SCORE, uint32_t GRID_CLASS, uint32_t TAIL_PRESENT,
          uint32_t HAS_THRESHOLD, uint32_t SMALL_ROWS_MINUS_ONE, uint32_t RATIO_ONE>
__global__ __aicore__ void safety_belt_violation_geometry(
    GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold, GM_ADDR violationScore,
    GM_ADDR workspace, GM_ADDR tiling)
{
    static_assert(GRID_CLASS <= 30U && TAIL_PRESENT <= 1U && HAS_THRESHOLD <= 1U &&
                  SMALL_ROWS_MINUS_ONE <= 63U && RATIO_ONE <= 4U,
                  "invalid scheduling class");
    if constexpr (GRID_CLASS == 29U) {
        static_assert(std::is_same_v<DT_PERSON_BOXES, float> && std::is_same_v<DT_VIOLATION_SCORE, float> &&
                      TAIL_PRESENT == 1U && RATIO_ONE == 4U, "Float32 half-ratio static medium band");
        constexpr uint32_t fullRows = 256U;
        constexpr uint32_t batchRows = 256U + 1U + SMALL_ROWS_MINUS_ONE;
        constexpr uint32_t tailRows = batchRows - fullRows;
        static_assert(tailRows >= 1U && tailRows <= 64U, "complete first-vector tail band");
        const uint32_t index = GetBlockIdx();
        if (index >= 2U) return;
        if (index == 1U) {
            DualIO::RunTile<DT_PERSON_BOXES, DT_VIOLATION_SCORE, 64U, false, true>(
                personBoxes, anchorPoints, threshold, violationScore, fullRows, tailRows, 0.5f, HAS_THRESHOLD != 0U);
        } else {
            DualIO::RunTile<DT_PERSON_BOXES, DT_VIOLATION_SCORE, SBG_MAX_LOCAL_ROWS, false, false>(
                personBoxes, anchorPoints, threshold, violationScore, 0U, fullRows, 0.5f, HAS_THRESHOLD != 0U);
        }
        return;
    }
    if constexpr (GRID_CLASS == 16U) {
        if constexpr (std::is_same_v<DT_VIOLATION_SCORE, half>) {
            LegacySmall::Run<DT_PERSON_BOXES, SMALL_ROWS_MINUS_ONE + 1U, HAS_THRESHOLD != 0U, RATIO_ONE>(personBoxes, anchorPoints, threshold, violationScore, tiling);
        } else {
            StrictSmall::Run<DT_PERSON_BOXES, SMALL_ROWS_MINUS_ONE + 1U, HAS_THRESHOLD != 0U, RATIO_ONE>(personBoxes, anchorPoints, threshold, violationScore, tiling);
        }
        return;
    }
    if constexpr (GRID_CLASS == 30U) {
        static_assert(std::is_same_v<DT_PERSON_BOXES, half> && TAIL_PRESENT == 1U && RATIO_ONE == 4U,
                      "invalid staticHalf320 capacity route");
        if (get_block_idx() != 0 || get_subblockid() != 0) return;
        constexpr uint32_t N = 320U - 64U + 1U + SMALL_ROWS_MINUS_ONE;
        static_assert(N <= 320U, "invalid static length");
        DualIO::RunTile<DT_PERSON_BOXES,DT_VIOLATION_SCORE,320U,true,false>(
            personBoxes,anchorPoints,threshold,violationScore,0U,N,0.5f,HAS_THRESHOLD != 0U);
        return;
    }
    if constexpr (GRID_CLASS == 28U) {
        constexpr uint32_t C = 256U;
        constexpr bool H = std::is_same_v<DT_PERSON_BOXES, half>;
        const uint32_t index = GetBlockIdx();
        if (index >= 2U) return;
        float ratio = RATIO_ONE == 1U ? 1.0f : RATIO_ONE == 2U ? 0.8f : RATIO_ONE == 3U ? 0.9f : 0.5f;
        if constexpr (RATIO_ONE == 0U) ratio = *reinterpret_cast<const __gm__ volatile float *>(tiling + 4U);
        if (index == 0U) {
            const uint32_t batch = (*reinterpret_cast<const __gm__ uint32_t *>(tiling) & 0xFFFU) + 1U;
            DualIO::RunTile<DT_PERSON_BOXES, DT_VIOLATION_SCORE, 64U, H, true>(
                personBoxes, anchorPoints, threshold, violationScore, C, batch - C, ratio, HAS_THRESHOLD != 0U);
        } else {
            DualIO::RunTile<DT_PERSON_BOXES, DT_VIOLATION_SCORE, H ? C : SBG_MAX_LOCAL_ROWS, H, false>(
                personBoxes, anchorPoints, threshold, violationScore, 0U, C, ratio, HAS_THRESHOLD != 0U);
        }
        return;
    }
    constexpr uint32_t CAPACITY =
        GRID_CLASS == 17U ? 128U : GRID_CLASS == 18U ? 192U : GRID_CLASS == 9U ? 64U : (GRID_CLASS == 10U ? 256U : (GRID_CLASS == 11U ? 320U : 512U));
    constexpr uint32_t USED = (GRID_CLASS == 17U || GRID_CLASS == 18U) ? 2U : GRID_CLASS == 0U ? 0U :
        (GRID_CLASS <= 8U ? GRID_CLASS : (GRID_CLASS <= 11U ? 1U : 8U));
    constexpr bool STATIC_LAST_VECTOR = GRID_CLASS == 19U;
    constexpr bool CONST_RATIO = (GRID_CLASS >= 12U && GRID_CLASS <= 15U) || STATIC_LAST_VECTOR || RATIO_ONE != 0U;
    constexpr uint32_t RATIO_INDEX = STATIC_LAST_VECTOR ? 0U : (CONST_RATIO ? (GRID_CLASS - 12U) : 0U);
    constexpr bool ONE_VECTOR = CAPACITY == 64U;
    constexpr bool PACKED_CAST = std::is_same_v<DT_PERSON_BOXES, half> && USED != 0U;
    constexpr uint32_t PLANE_ROWS = PACKED_CAST ? CAPACITY : SBG_MAX_LOCAL_ROWS;
    const uint32_t blockIndex = GetBlockIdx();
    constexpr bool NEED_SLOT = USED == 0U || (TAIL_PRESENT != 0U && !STATIC_LAST_VECTOR) || !CONST_RATIO;
    uint64_t slot = 0U;
    if constexpr (NEED_SLOT) {
        slot = *reinterpret_cast<const __gm__ uint64_t *>(tiling);
    }
    union RatioBits {
        uint32_t bits;
        float value;
    } ratioBits;
    ratioBits.bits = static_cast<uint32_t>(slot >> 32U);
    const uint32_t pack = static_cast<uint32_t>(slot);
    const float ratio = RATIO_ONE != 0U ?
        (RATIO_ONE == 1U ? 1.0f : (RATIO_ONE == 2U ? 0.8f : (RATIO_ONE == 3U ? 0.9f : 0.5f))) : (CONST_RATIO ?
        (RATIO_INDEX == 0U ? 0.8f : (RATIO_INDEX == 1U ? 0.9f : (RATIO_INDEX == 2U ? 1.0f : 0.5f))) :
        ratioBits.value);
    if constexpr (USED != 0U) {
        if (blockIndex >= USED) {
            return;
        }
        // With a real tail, rotate logical blocks so the partial tile lands on block0.
        uint32_t logical = blockIndex;
        if constexpr (TAIL_PRESENT != 0U && USED > 1U) {
            logical = blockIndex + USED - 1U;
            if (logical >= USED) {
                logical -= USED;
            }
        }
        const uint32_t start = logical * CAPACITY;
        if constexpr (TAIL_PRESENT != 0U) {
            if (logical + 1U == USED) {
                constexpr uint32_t lastVectorStart = SBG_MAX_LOCAL_ROWS - 64U;
                const uint32_t batch = STATIC_LAST_VECTOR ? (USED - 1U) * CAPACITY + lastVectorStart + SMALL_ROWS_MINUS_ONE : (pack & 0xFFFU) + 1U;
                DualIO::RunTile<DT_PERSON_BOXES, DT_VIOLATION_SCORE, PLANE_ROWS, PACKED_CAST, ONE_VECTOR, ((GRID_CLASS >= 12U && GRID_CLASS <= 15U) || STATIC_LAST_VECTOR)>(personBoxes, anchorPoints, threshold,
                    violationScore, start, STATIC_LAST_VECTOR ? lastVectorStart + SMALL_ROWS_MINUS_ONE : batch - start, ratio, HAS_THRESHOLD != 0U);
                return;
            }
        }
        DualIO::RunTile<DT_PERSON_BOXES, DT_VIOLATION_SCORE, PLANE_ROWS, PACKED_CAST, ONE_VECTOR, ((GRID_CLASS >= 12U && GRID_CLASS <= 15U) || STATIC_LAST_VECTOR)>(personBoxes, anchorPoints, threshold,
            violationScore, start, CAPACITY, ratio, HAS_THRESHOLD != 0U);
        return;
    }
    const uint32_t batch = (pack & 0xFFFU) + 1U;
    const uint32_t rowsPerCore = ((pack >> 12U) & 0xFU) * 64U;
    const uint32_t start = blockIndex * rowsPerCore;
    if (start >= batch) {
        return;
    }
    const uint32_t remaining = batch - start;
    const uint32_t validRows = remaining < rowsPerCore ? remaining : rowsPerCore;
    DualIO::RunTile<DT_PERSON_BOXES, DT_VIOLATION_SCORE, PLANE_ROWS, PACKED_CAST, ONE_VECTOR>(personBoxes, anchorPoints, threshold,
        violationScore, start, validRows, ratio, HAS_THRESHOLD != 0U);
}

// This operator uses direct UB tensors and never consumes the system workspace.
// Experimental generated-entry specialization for CANN 9.0; SDK files stay untouched.
// Preserve the standard entry for debug, timestamp and OOM-instrumented builds.
#include "kernel_common.h"
#if (!defined(ASCENDC_DUMP) || ASCENDC_DUMP == 0) && !defined(ASCENDC_TIME_STAMP_ON) && (!defined(ASCENDC_OOM) || ASCENDC_OOM == 0)
namespace AscendC {
__aicore__ inline void SbgUnusedWorkspaceEntry(GM_ADDR workspace)
{
    (void)workspace;
}
}
#define SetSysWorkspaceForce SbgUnusedWorkspaceEntry
#endif
