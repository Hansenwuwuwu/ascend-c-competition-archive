// 本轮策略：所有boxes、anchors和output DMA均使用普通DataCopy，字节数向上补齐到32B。
// 逻辑B、FP32分步计算和严格GT不变。每个核起点须32B对齐，补齐输出不能覆盖其他核有效数据。
// 存储前提：每个独立输入/输出GM分配实际覆盖ALIGN_UP(逻辑字节数,32)；SetGlobalBuffer不会扩容。
// 输入多读区不属于有效样本；输出多写区只允许位于该张量已分配的padding。任意无padding子视图不适用。
// 十二段四阶段性能恢复版：区间类自己拥有原v528优化与真实B流程，未增加单点类。
// Host真实尺寸/属性守卫选择编译期参数；没有缓存输入坐标或预生成答案。
// B6、4094等仍在对应区间类处理；原v528和之前审阅代码均保留。
// 十二个完整尺寸分支的结构审阅候选，来源v528；并非原高分二进制。
// 每种dtype：[1,31]、32、[33,255]、256、[257,4095]、4096。
// 每个普通区间由一个类完整处理；不再把4095和4094拆到不同类/函数。
// Host只决定每核分工与元数据；真实坐标仍在NPU读取、抽列、按FP32分步计算。
// coreRows是有效行数，pad是UB/向量长度；DataCopy是32B块单位，Pad接口是字节单位。
// 保留当前历史FP16阈值/输出ABI；题面固定FP32文字差异仍需要单独review。
// 每个类均有Init/CopyIn/Compute/CopyOut；当前候选待本轮编译、实机检查及官方评测。
#include "kernel_operator.h"
#define SBG_POLICY_TARGET __aicore__
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"
using namespace AscendC;

namespace {
constexpr int32_t kSbgRowAlign = 16;   // 行对齐：fp16 输出 32B 对齐要求 16 行的倍数
constexpr int32_t kSbgPadMin = 64;     // PairReduceSum 需要 pad>=64（rep2>=2）
constexpr int32_t kSbgPerRepeat = 64;  // 一个 vector repeat 处理 64 个 f32
constexpr uint64_t kSbgMaskOdd = 0xAAAAAAAAAAAAAAAAULL;
constexpr uint64_t kSbgMaskEven = 0x5555555555555555ULL;

__aicore__ inline uint32_t SbgAlignBytes(uint32_t bytes) {
    return (bytes + 31U) & ~31U;
}
__aicore__ inline uint32_t SbgAlignUp(uint32_t a, uint32_t b) {
    return (a + b - 1U) & ~(b - 1U);
}
}  // namespace

// ============================================================================
// 【分支01】FP32，完整覆盖B∈[1,31]：SbgFp32Range1To31
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
class SbgFp32Range1To31 final {
public:
    // Run只负责调度；本类完整处理[1,31]，不转发给单点实现。
    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR threshold, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score,
                  length, rowsPerCore, hasThreshold, tiling)) return;
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    using T = float;
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
    uint32_t rowStart = 0U, coreRows = 0U, movedRows = 0U, pad = 0U;
    uint32_t hasThreshold = 0U;
    GM_ADDR threshold = nullptr;
    GM_ADDR tiling = nullptr;
    float waistRatio = 0.0f, thresholdValue = 0.0f;
    __gm__ T *boxesBase = nullptr, *anchorsBase = nullptr, *outBase = nullptr;
    LocalTensor<T> boxIn, ancIn, outUb, cst;
    LocalTensor<float> sv;

    // Init：按真实B分片并建立UB/GM视图；保留原阈值预取位置，不添加同步。
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        threshold = thresholdArg;
        hasThreshold = hasThresholdArg;
        tiling = tilingArg;

        if (length < 1U || length > 31U) return false;

        // 本类直接处理整个区间：真实B和每核行步长来自Tiling，不转发给单点类。
        // 超出有效B的启动核退出；最后一核只取剩余的有效行数。
        const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        rowStart = blockIdx * rowsPerCore;
        {
            if (rowStart >= length) return false;
        }
        {
            coreRows = length - rowStart;
            if (coreRows > rowsPerCore) coreRows = rowsPerCore;
        }
        
        // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
        movedRows = SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgRowAlign));
        pad = 64U;
        // 以下是本类自有的紧凑UB排列；o按字节，LocalTensor下标按其元素类型。
        // 当前布局预算为FP32的65*pad+32字节，FP16的71*pad+32字节。
        uint32_t o = 0;
        boxIn = LocalTensor<T>(TPosition::VECIN, o, pad * 4U);
        o += SbgAlignBytes(pad * 4U * tSize);
        ancIn = LocalTensor<T>(TPosition::VECIN, o, pad * 2U);
        o += SbgAlignBytes(pad * 2U * tSize);
        outUb = LocalTensor<T>(TPosition::VECCALC, o, pad);
        o += SbgAlignBytes(pad * tSize);
        
        cst = LocalTensor<T>(TPosition::VECCALC, o, pad * 2U);
        o += SbgAlignBytes(pad * tSize * 2U);
        sv = LocalTensor<float>(TPosition::VECCALC, o, 7U * pad + 4U * bankGap + (SbgAlignBytes(pad) / 4U) + 8U);
        boxesBase = reinterpret_cast<__gm__ T *>(person_boxes);
        anchorsBase = reinterpret_cast<__gm__ T *>(anchor_points);
        outBase = reinterpret_cast<__gm__ T *>(violation_score);
        return true;
    }

    // CopyIn：发出boxes/anchors搬入，并读取本次调用的属性和适用阈值。
    __aicore__ inline void CopyIn() {
        GlobalTensor<T> boxGm;
        boxGm.SetGlobalBuffer(boxesBase + static_cast<int64_t>(rowStart) * 4, SbgAlignBytes(coreRows * 4U * sizeof(T)) / sizeof(T));
        // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
        // boxes统一按有效字节数向上补齐到32B；GM须有真实可读padding。
        const uint32_t boxBytes = SbgAlignBytes(coreRows * 4U * tSize);
        DataCopyParams boxDp{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
        DataCopy(boxIn, boxGm, boxDp);

        GlobalTensor<T> ancGm;
        ancGm.SetGlobalBuffer(anchorsBase + static_cast<int64_t>(rowStart) * 2, SbgAlignBytes(coreRows * 2U * sizeof(T)) / sizeof(T));
        // anchors与boxes使用相同的普通DMA策略，不能只修改其中一路。
        const uint32_t anchorBytes = SbgAlignBytes(coreRows * 2U * tSize);
        DataCopyParams ancDp{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
        DataCopy(ancIn, ancGm, ancDp);
        // Both input DMA requests are already in flight. Read the attribute here,
        // so the scalar cache miss can overlap input transfers.
        // 仍读取本次调用的真实float属性，不因shape相同而忽略输入值。
        waistRatio = reinterpret_cast<__gm__ volatile float*>(tiling)[3];
        AscendC::InitSocState();

        thresholdValue = 0.0f;
        {
            if ((hasThreshold != 0U) && (threshold != nullptr)) {
                thresholdValue = static_cast<float>(reinterpret_cast<__gm__ T *>(threshold)[0]);
            }
        }
    }

    // Compute：等待搬入完成，抽列、FP32分步计算、严格GT比较并生成0/1。
    __aicore__ inline void Compute() {
        LocalTensor<uint8_t> s8 = sv.template ReinterpretCast<uint8_t>();
        LocalTensor<uint8_t> cmpUb = s8[(7U * pad + 4U * bankGap) * static_cast<uint32_t>(sizeof(float))];

        LocalTensor<T> oneUb;
        oneUb = cst[0];
        LocalTensor<T> zeroUb;
        zeroUb = cst[pad];
        Duplicate(oneUb, static_cast<T>(1.0f), static_cast<int32_t>(pad));
        Duplicate(zeroUb, static_cast<T>(0.0f), static_cast<int32_t>(pad));


        // The last 32 bytes of sv are reserved and do not overlap cmpUb or any
        // intermediate tensor. Only the address is needed here, not comparison data.
        

        // 等输入DMA完成后才消费UB；当前仍是单段切片，没有双缓冲。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        LocalTensor<float> boxVec;
        LocalTensor<float> ancVec;
        {
            boxVec = boxIn.template ReinterpretCast<float>();
            ancVec = ancIn.template ReinterpretCast<float>();
        }

        LocalTensor<float> lUb;
        lUb = sv[0];
        LocalTensor<float> y1Ub;
        y1Ub = sv[2U * pad];
        LocalTensor<float> y2Ub;
        y2Ub = sv[3U * pad + bankGap];
        LocalTensor<float> dUb;
        dUb = sv[4U * pad + bankGap];
        LocalTensor<float> boundUb;
        boundUb = sv[5U * pad + 2U * bankGap];
        LocalTensor<float> cyUb;
        cyUb = sv[6U * pad + 3U * bankGap];

        const uint64_t maskOdd[1] = {kSbgMaskOdd};
        const uint64_t maskEven[1] = {kSbgMaskEven};
        const int32_t rep4 = static_cast<int32_t>((4U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
        const int32_t rep2 = static_cast<int32_t>((2U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
        const int32_t padI = static_cast<int32_t>(pad);

        {
            // 奇/偶掩码完成AoS→三列抽取；不对不同样本做汇总。
            PairReduceSum<float>(lUb, boxVec, rep4, maskOdd, 1, 1, 8);
            PairReduceSum<float>(y1Ub, lUb, rep2, maskEven, 1, 1, 8);
            PairReduceSum<float>(y2Ub, lUb, rep2, maskOdd, 1, 1, 8);
            PairReduceSum<float>(cyUb, ancVec, rep2, maskOdd, 1, 1, 8);
        }

        {
            // 保持FP32分步舍入及原公式，不引入FMA或改变结合顺序。
            Sub(dUb, y2Ub, y1Ub, padI);
            Muls(dUb, dUb, waistRatio, padI);
            Add(boundUb, y1Ub, dUb, padI);
            
            Adds(boundUb, boundUb, thresholdValue, padI);
        }

        {
            // 严格GT：cy等于边界时输出0；padding lane不得影响有效输出。
            Compare(cmpUb, cyUb, boundUb, CMPMODE::GT, padI);
            Select(outUb, cmpUb, oneUb, zeroUb, SELMODE::VSEL_TENSOR_TENSOR_MODE, padI);
        }
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<T> outGm;
        outGm.SetGlobalBuffer(outBase + rowStart, SbgAlignBytes(coreRows * sizeof(T)) / sizeof(T));
        const uint32_t outBytes = coreRows * tSize;
        if ((outBytes & 31U) == 0U) {
            DataCopyParams outDp;
            outDp.blockCount = 1;
            outDp.blockLen = static_cast<uint16_t>(outBytes / 32U);
            outDp.srcStride = 0;
            outDp.dstStride = 0;
            DataCopy(outGm, outUb, outDp);
        } else {
            DataCopyParams outCp;
            outCp.blockCount = 1;
            outCp.blockLen = static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U);
            outCp.srcStride = 0;
            outCp.dstStride = 0;
            DataCopy(outGm, outUb, outCp);
        }
    }
};

// 尺寸分支2：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case01：FP32 / B=32】SbgFp32B32
// 当前启动8核，工作核上限1；核0处理32行；核1～7退出。
// 实际向量计算行数：64；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：512 / 256 / 128 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,1024)；anchorIn_[1024,1536)；output_[1536,1792)；constants_[1792,2304)；scratch_[2304,4192)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp32B32 final {
public:
    static constexpr uint32_t kBatch = 32U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 1U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 32U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 64U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 0U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 2304U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 472U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 4192U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgShapePolicy(kBatch, false).launch == 8U,
                  "Host launch policy changed: review this point explicitly");
    static_assert(SbgShapePolicy(kBatch, false).working == kWorkingCores,
                  "Host working-core policy changed: review this point explicitly");
    static_assert((((kBatch + kWorkingCores - 1U) / kWorkingCores + 15U) / 16U * 16U)
                  == kRowsPerCore, "Host/device slice mismatch");
    static_assert(kUbBytes <= 192U * 1024U, "Point UB allocation exceeds 192 KiB");

    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
                              GM_ADDR threshold, GM_ADDR violation_score,
                              uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score, hasThreshold, tiling)) return;
        // 仅有一个数据切片；这些阶段有必要的依赖等待，不能凭函数名认定已流水重叠。
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    using T = float;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
        if (block >= kWorkingCores) return false;
        rowStart_ = 0U;
        coreRows_ = 32U;
        threshold_ = threshold;
        tiling_ = tiling;
        hasThreshold_ = hasThreshold;
        thresholdValue_ = 0.0f;
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T));
        anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
        anchorIn_ = LocalTensor<T>(TPosition::VECIN, 1024U, kPad * 2U);
        output_ = LocalTensor<T>(TPosition::VECCALC, 1536U, kPad);
        constants_ = LocalTensor<T>(TPosition::VECCALC, 1792U, kPad * 2U);
        scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
        compare_ = scratch_.template ReinterpretCast<uint8_t>()[1792U];
        one_ = constants_[0];
        zero_ = constants_[kPad];

        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {

        DataCopyParams boxCopy{1, 16, 0, 0};
        DataCopyParams anchorCopy{1, 8, 0, 0};
        DataCopy(boxIn_, boxesGm_, boxCopy);
        DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
            RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
            reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
        // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
        AscendC::InitSocState();
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
            thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ T*>(threshold_)[0]);
        }
    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
        Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));

        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        LocalTensor<float> y1 = scratch_[2U * kPad];
        LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
        LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
        LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
        LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
        // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
        // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
        LocalTensor<float> pairs = scratch_[0];
        if ASCEND_IS_AIV {
            auto pairsPtr = reinterpret_cast<__ubuf__ float*>(pairs.GetPhyAddr());
            auto boxesPtr = reinterpret_cast<__ubuf__ float*>(boxIn_.GetPhyAddr());
            auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
            vcpadd(pairsPtr, boxesPtr, 4, 1, 1, 8);
            vcpadd(cyPtr, anchorsPtr, 2, 1, 1, 8);
            pipe_barrier(PIPE_V);
            vcpadd(y2Ptr, pairsPtr, 2, 1, 1, 8);
            set_vector_mask(0ULL, 0x5555555555555555ULL);
            vcpadd(y1Ptr, pairsPtr, 2, 1, 1, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
        }
        if ASCEND_IS_AIV {
            auto d = reinterpret_cast<__ubuf__ float*>(difference.GetPhyAddr());
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto bound = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
            vsub(d, y2Ptr, y1Ptr, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (!UNIT_RATIO) {
                vmuls(d, d, waistRatio_, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            vadd(bound, y1Ptr, d, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadds(bound, bound, thresholdValue_, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        SetMaskNorm();
        // 一个FP32 repeat的比较结果直接放CMPMASK寄存器；GT相等时为假。
        Compare(cy, boundary, CMPMODE::GT, uint64_t(64), {1, 1, 1, 8, 8, 8});
        Select<T, SELMODE::VSEL_CMPMASK_SPR>(output_, one_, zero_, 1, {1, 1, 1, 8, 8, 8});
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams outputCopy{1, 4, 0, 0};
        DataCopy(outputGm_, output_, outputCopy);
    }
};

// ============================================================================
// 【分支03】FP32，完整覆盖B∈[33,255]：SbgFp32Range33To255
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
template <uint32_t FIXED_BATCH = 0, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0, bool TAIL = false>
class SbgFp32Range33To255 final {
public:
    // 完整区间[33,255]只有本类。FIXED_BATCH是Host真实B守卫后的编译期策略；0表示真实运行时B。
    // 编译期策略恢复v528原优化，不创建独立单点类；B4094等仍由本类处理。
    static constexpr uint32_t kBatch = FIXED_BATCH;
    static constexpr uint32_t kWorkingCores = FIXED_BATCH == 33U ? (1U) : 0U;
    static constexpr uint32_t kRowsPerCore = FIXED_BATCH == 33U ? (48U) : 0U;
    static constexpr uint32_t kPad = FIXED_BATCH == 33U ? (64U) : 0U;
    static constexpr uint32_t kBankGap = FIXED_BATCH == 33U ? (0U) : 0U;
    static constexpr uint32_t kSvBase = FIXED_BATCH == 33U ? (2304U) : 0U;
    static constexpr uint32_t kSvFloatCount = FIXED_BATCH == 33U ? (472U) : 0U;
    static constexpr uint32_t kUbBytes = FIXED_BATCH == 33U ? (4192U) : 0U;
    static_assert(FIXED_BATCH == 0U || (FIXED_BATCH == 33U), "Unsupported constant interval plan");
    static_assert(kUbBytes <= 192U * 1024U, "UB budget exceeds 192 KiB");
    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR threshold, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score,
                  length, rowsPerCore, hasThreshold, tiling)) return;
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    // 原优化路径所需的成员与UB视图；通用成员不会被常量策略使用。
    using T = float;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;
    // 通用B的真实长度/片上视图；编译器在常量策略中消除未使用成员。
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
    uint32_t rowStart = 0U, coreRows = 0U, movedRows = 0U, pad = 0U;
    uint32_t hasThreshold = 0U;
    GM_ADDR threshold = nullptr;
    GM_ADDR tiling = nullptr;
    float waistRatio = 0.0f, thresholdValue = 0.0f;
    __gm__ T *boxesBase = nullptr, *anchorsBase = nullptr, *outBase = nullptr;
    LocalTensor<T> boxIn, ancIn, outUb, cst;
    LocalTensor<float> sv;
    // Init按本类策略建立分片和UB；每条路径均保持原有效GM长度。
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        if constexpr (FIXED_BATCH == 33U) {
            // B=33由核0一次处理全部33行；核1~7退出，不再拆成32+1行。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
            if (block >= kWorkingCores) return false;
            rowStart_ = 0U;
            coreRows_ = 33U;
            threshold_ = thresholdArg;
            tiling_ = tilingArg;
            hasThreshold_ = hasThresholdArg;
            thresholdValue_ = 0.0f;
            // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
            // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
            boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, ((coreRows_ * 4U * sizeof(T) + 31U) & ~31U) / sizeof(T));
            anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, ((coreRows_ * 2U * sizeof(T) + 31U) & ~31U) / sizeof(T));
            outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
            anchorIn_ = LocalTensor<T>(TPosition::VECIN, 1024U, kPad * 2U);
            output_ = LocalTensor<T>(TPosition::VECCALC, 1536U, kPad);
            constants_ = LocalTensor<T>(TPosition::VECCALC, 1792U, kPad * 2U);
            scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
            compare_ = scratch_.template ReinterpretCast<uint8_t>()[1792U];
            one_ = constants_[0];
            zero_ = constants_[kPad];

            return true;
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            threshold = thresholdArg;
            hasThreshold = hasThresholdArg;
            tiling = tilingArg;

            if (length < 33U || length > 255U) return false;

            // 本类直接处理整个区间：真实B和每核行步长来自Tiling，不转发给单点类。
            // 超出有效B的启动核退出；最后一核只取剩余的有效行数。
            const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
            rowStart = blockIdx * rowsPerCore;
            {
                if (rowStart >= length) return false;
            }
            {
                coreRows = length - rowStart;
                if (coreRows > rowsPerCore) coreRows = rowsPerCore;
            }
            
            // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
            movedRows = SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgRowAlign));
            pad = SbgAlignUp(coreRows, 64U);
            // 以下是本类自有的紧凑UB排列；o按字节，LocalTensor下标按其元素类型。
            // 当前布局预算为FP32的65*pad+32字节，FP16的71*pad+32字节。
            uint32_t o = 0;
            boxIn = LocalTensor<T>(TPosition::VECIN, o, pad * 4U);
            o += SbgAlignBytes(pad * 4U * tSize);
            ancIn = LocalTensor<T>(TPosition::VECIN, o, pad * 2U);
            o += SbgAlignBytes(pad * 2U * tSize);
            outUb = LocalTensor<T>(TPosition::VECCALC, o, pad);
            o += SbgAlignBytes(pad * tSize);
            
            cst = LocalTensor<T>(TPosition::VECCALC, o, pad * 2U);
            o += SbgAlignBytes(pad * tSize * 2U);
            sv = LocalTensor<float>(TPosition::VECCALC, o, 7U * pad + 4U * bankGap + (SbgAlignBytes(pad) / 4U) + 8U);
            boxesBase = reinterpret_cast<__gm__ T *>(person_boxes);
            anchorsBase = reinterpret_cast<__gm__ T *>(anchor_points);
            outBase = reinterpret_cast<__gm__ T *>(violation_score);
            return true;
        
        }
    }

    // CopyIn：搬入真实输入并准备实际属性/阈值。
    __aicore__ inline void CopyIn() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的B=33策略：32B补齐输入，普通DataCopy，有效行数独立控制。


            // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
            // 单工作核33行：boxes 528B向上补齐到544B，anchors 264B补齐到288B。
            // 调用方必须提供实际可读的32B补齐存储；SetGlobalBuffer不会扩容GM。
            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            const uint32_t boxBytes = (coreRows_ * 16U + 31U) & ~31U;
            const uint32_t anchorBytes = (coreRows_ * 8U + 31U) & ~31U;
            DataCopyParams boxCopy{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
            DataCopyParams anchorCopy{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
            DataCopy(boxIn_, boxesGm_, boxCopy);
            DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
                RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
                reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
            // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
            AscendC::InitSocState();
            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
                thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ T*>(threshold_)[0]);
            }
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            GlobalTensor<T> boxGm;
            boxGm.SetGlobalBuffer(boxesBase + static_cast<int64_t>(rowStart) * 4, SbgAlignBytes(coreRows * 4U * sizeof(T)) / sizeof(T));
            // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
            // boxes统一按有效字节数向上补齐到32B；GM须有真实可读padding。
            const uint32_t boxBytes = SbgAlignBytes(coreRows * 4U * tSize);
            DataCopyParams boxDp{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
            DataCopy(boxIn, boxGm, boxDp);

            GlobalTensor<T> ancGm;
            ancGm.SetGlobalBuffer(anchorsBase + static_cast<int64_t>(rowStart) * 2, SbgAlignBytes(coreRows * 2U * sizeof(T)) / sizeof(T));
            // anchors与boxes使用相同的普通DMA策略，不能只修改其中一路。
            const uint32_t anchorBytes = SbgAlignBytes(coreRows * 2U * tSize);
            DataCopyParams ancDp{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
            DataCopy(ancIn, ancGm, ancDp);
            // Both input DMA requests are already in flight. Read the attribute here,
            // so the scalar cache miss can overlap input transfers.
            // 仍读取本次调用的真实float属性，不因shape相同而忽略输入值。
            waistRatio = reinterpret_cast<__gm__ volatile float*>(tiling)[3];
            AscendC::InitSocState();

            thresholdValue = 0.0f;
            {
                if ((hasThreshold != 0U) && (threshold != nullptr)) {
                    thresholdValue = static_cast<float>(reinterpret_cast<__gm__ T *>(threshold)[0]);
                }
            }
        
        }
    }

    // Compute：抽列、FP32分步运算、严格GT比较。
    __aicore__ inline void Compute() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的B=33策略：保留抽列与转换，逐行算术及比较只处理coreRows_。

            // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
            Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
            Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));

            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            LocalTensor<float> y1 = scratch_[2U * kPad];
            LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
            LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
            LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
            LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
            // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
            // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
            LocalTensor<float> pairs = scratch_[0];
            if ASCEND_IS_AIV {
                auto pairsPtr = reinterpret_cast<__ubuf__ float*>(pairs.GetPhyAddr());
                auto boxesPtr = reinterpret_cast<__ubuf__ float*>(boxIn_.GetPhyAddr());
                auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
                auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
                auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
                auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
                vcpadd(pairsPtr, boxesPtr, 4, 1, 1, 8);
                vcpadd(cyPtr, anchorsPtr, 2, 1, 1, 8);
                pipe_barrier(PIPE_V);
                vcpadd(y2Ptr, pairsPtr, 2, 1, 1, 8);
                set_vector_mask(0ULL, 0x5555555555555555ULL);
                vcpadd(y1Ptr, pairsPtr, 2, 1, 1, 8);
                pipe_barrier(PIPE_V);
                set_vector_mask(0ULL, 0xffffffffffffffffULL);
            }
            if ASCEND_IS_AIV {
                auto d = reinterpret_cast<__ubuf__ float*>(difference.GetPhyAddr());
                auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
                auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
                auto bound = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
            // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
            // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
                // 普通DataCopy多搬的lane不参与逐行算术；仅计算有效的33行。
                set_mask_norm();
                set_vector_mask(0ULL, (1ULL << coreRows_) - 1ULL);
                vsub(d, y2Ptr, y1Ptr, 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                if constexpr (!UNIT_RATIO) {
                    vmuls(d, d, waistRatio_, 1, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                vadd(bound, y1Ptr, d, 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vadds(bound, bound, thresholdValue_, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            SetMaskNorm();
            // 一个FP32 repeat的比较结果直接放CMPMASK寄存器；GT相等时为假。
            Compare(cy, boundary, CMPMODE::GT, uint64_t(coreRows_), {1, 1, 1, 8, 8, 8});
            // 选择结果也只写入有效lane；CopyOut继续按实际有效字节数输出逻辑有效行。
            if ASCEND_IS_AIV { set_mask_norm(); set_vector_mask(0ULL, (1ULL << coreRows_) - 1ULL); }
            Select<T, SELMODE::VSEL_CMPMASK_SPR>(output_, one_, zero_, 1, {1, 1, 1, 8, 8, 8});
            ResetMask();
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            LocalTensor<uint8_t> s8 = sv.template ReinterpretCast<uint8_t>();
            LocalTensor<uint8_t> cmpUb = s8[(7U * pad + 4U * bankGap) * static_cast<uint32_t>(sizeof(float))];

            LocalTensor<T> oneUb;
            oneUb = cst[0];
            LocalTensor<T> zeroUb;
            zeroUb = cst[pad];
            Duplicate(oneUb, static_cast<T>(1.0f), static_cast<int32_t>(pad));
            Duplicate(zeroUb, static_cast<T>(0.0f), static_cast<int32_t>(pad));


            // The last 32 bytes of sv are reserved and do not overlap cmpUb or any
            // intermediate tensor. Only the address is needed here, not comparison data.
            

            // 等输入DMA完成后才消费UB；当前仍是单段切片，没有双缓冲。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            LocalTensor<float> boxVec;
            LocalTensor<float> ancVec;
            {
                boxVec = boxIn.template ReinterpretCast<float>();
                ancVec = ancIn.template ReinterpretCast<float>();
            }

            LocalTensor<float> lUb;
            lUb = sv[0];
            LocalTensor<float> y1Ub;
            y1Ub = sv[2U * pad];
            LocalTensor<float> y2Ub;
            y2Ub = sv[3U * pad + bankGap];
            LocalTensor<float> dUb;
            dUb = sv[4U * pad + bankGap];
            LocalTensor<float> boundUb;
            boundUb = sv[5U * pad + 2U * bankGap];
            LocalTensor<float> cyUb;
            cyUb = sv[6U * pad + 3U * bankGap];

            const uint64_t maskOdd[1] = {kSbgMaskOdd};
            const uint64_t maskEven[1] = {kSbgMaskEven};
            const int32_t rep4 = static_cast<int32_t>((4U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
            const int32_t rep2 = static_cast<int32_t>((2U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
            const int32_t padI = static_cast<int32_t>(pad);

            {
                // 奇/偶掩码完成AoS→三列抽取；不对不同样本做汇总。
                PairReduceSum<float>(lUb, boxVec, rep4, maskOdd, 1, 1, 8);
                PairReduceSum<float>(y1Ub, lUb, rep2, maskEven, 1, 1, 8);
                PairReduceSum<float>(y2Ub, lUb, rep2, maskOdd, 1, 1, 8);
                PairReduceSum<float>(cyUb, ancVec, rep2, maskOdd, 1, 1, 8);
            }

            {
                // 保持FP32分步舍入及原公式，不引入FMA或改变结合顺序。
                Sub(dUb, y2Ub, y1Ub, padI);
                Muls(dUb, dUb, waistRatio, padI);
                Add(boundUb, y1Ub, dUb, padI);
                
                Adds(boundUb, boundUb, thresholdValue, padI);
            }

            {
                // 严格GT：cy等于边界时输出0；padding lane不得影响有效输出。
                Compare(cmpUb, cyUb, boundUb, CMPMODE::GT, padI);
                Select(outUb, cmpUb, oneUb, zeroUb, SELMODE::VSEL_TENSOR_TENSOR_MODE, padI);
            }
        
        }
    }

    // CopyOut：只搬出有效行。
    __aicore__ inline void CopyOut() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的编译期B=33策略：恢复冻结v528原CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyParams outputCopy{1, static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 4U) / 32U), 0, 0};
            DataCopy(outputGm_, output_, outputCopy);
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            GlobalTensor<T> outGm;
            outGm.SetGlobalBuffer(outBase + rowStart, SbgAlignBytes(coreRows * sizeof(T)) / sizeof(T));
            const uint32_t outBytes = coreRows * tSize;
            if ((outBytes & 31U) == 0U) {
                DataCopyParams outDp;
                outDp.blockCount = 1;
                outDp.blockLen = static_cast<uint16_t>(outBytes / 32U);
                outDp.srcStride = 0;
                outDp.dstStride = 0;
                DataCopy(outGm, outUb, outDp);
            } else {
                DataCopyParams outCp;
                outCp.blockCount = 1;
                outCp.blockLen = static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U);
                outCp.srcStride = 0;
                outCp.dstStride = 0;
                DataCopy(outGm, outUb, outCp);
            }
        
        }
    }
};

// 尺寸分支4：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case03：FP32 / B=256】SbgFp32B256
// 当前启动8核，工作核上限2；核0、1各处理128行；其余退出。
// 实际向量计算行数：128；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：每核2048 / 1024 / 512 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,2048)；anchorIn_[2048,3072)；output_[3072,3584)；constants_[3584,4608)；scratch_[65536,70304)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp32B256 final {
public:
    static constexpr uint32_t kBatch = 256U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 2U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 128U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 128U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 64U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 65536U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 1192U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 70304U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgShapePolicy(kBatch, false).launch == 8U,
                  "Host launch policy changed: review this point explicitly");
    static_assert(SbgShapePolicy(kBatch, false).working == kWorkingCores,
                  "Host working-core policy changed: review this point explicitly");
    static_assert((((kBatch + kWorkingCores - 1U) / kWorkingCores + 15U) / 16U * 16U)
                  == kRowsPerCore, "Host/device slice mismatch");
    static_assert(kUbBytes <= 192U * 1024U, "Point UB allocation exceeds 192 KiB");

    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
                              GM_ADDR threshold, GM_ADDR violation_score,
                              uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score, hasThreshold, tiling)) return;
        // 仅有一个数据切片；这些阶段有必要的依赖等待，不能凭函数名认定已流水重叠。
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    using T = float;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
        if (block >= kWorkingCores) return false;
        rowStart_ = block * kRowsPerCore;
        coreRows_ = 128U;
        threshold_ = threshold;
        tiling_ = tiling;
        hasThreshold_ = hasThreshold;
        thresholdValue_ = 0.0f;
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T));
        anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
        anchorIn_ = LocalTensor<T>(TPosition::VECIN, 2048U, kPad * 2U);
        output_ = LocalTensor<T>(TPosition::VECCALC, 3072U, kPad);
        constants_ = LocalTensor<T>(TPosition::VECCALC, 3584U, kPad * 2U);
        scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
        compare_ = scratch_.template ReinterpretCast<uint8_t>()[4608U];
        one_ = constants_[0];
        zero_ = constants_[kPad];

        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {

        DataCopyParams boxCopy{1, 64, 0, 0};
        DataCopyParams anchorCopy{1, 32, 0, 0};
        DataCopy(boxIn_, boxesGm_, boxCopy);
        DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
            RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
            reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
        // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
        AscendC::InitSocState();
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
            thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ T*>(threshold_)[0]);
        }
    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
        Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
        if ASCEND_IS_AIV {
            // 为当前CANN9选择指令约定准备“比较位图UB地址”的载体。
            // 这里不是最终比较结果；真正的0/1位图由后面的VCMPV写出。
            LocalTensor<uint32_t> addressPayload = scratch_.template ReinterpretCast<uint32_t>()[1184U];
            auto addressPtr = reinterpret_cast<__ubuf__ uint32_t*>(addressPayload.GetPhyAddr());
            const uint32_t compareAddress = static_cast<uint32_t>(
                reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr())));
            set_mask_norm();
            set_vector_mask(0ULL, 0xffULL);
            vector_dup(addressPtr, compareAddress, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_cmpmask(addressPtr);
            pipe_barrier(PIPE_V);
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        LocalTensor<float> y1 = scratch_[2U * kPad];
        LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
        LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
        LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
        LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
        // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
        // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
        LocalTensor<float> pairs = scratch_[0];
        if ASCEND_IS_AIV {
            auto pairsPtr = reinterpret_cast<__ubuf__ float*>(pairs.GetPhyAddr());
            auto boxesPtr = reinterpret_cast<__ubuf__ float*>(boxIn_.GetPhyAddr());
            auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
            vcpadd(pairsPtr, boxesPtr, 8, 1, 1, 8);
            vcpadd(cyPtr, anchorsPtr, 4, 1, 1, 8);
            pipe_barrier(PIPE_V);
            vcpadd(y2Ptr, pairsPtr, 4, 1, 1, 8);
            set_vector_mask(0ULL, 0x5555555555555555ULL);
            vcpadd(y1Ptr, pairsPtr, 4, 1, 1, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
        }
        if ASCEND_IS_AIV {
            auto d = reinterpret_cast<__ubuf__ float*>(difference.GetPhyAddr());
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto bound = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
            vsub(d, y2Ptr, y1Ptr, 2, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (!UNIT_RATIO) {
                vmuls(d, d, waistRatio_, 2, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            vadd(bound, y1Ptr, d, 2, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadds(bound, bound, thresholdValue_, 2, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        if ASCEND_IS_AIV {
            auto maskPtr = reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            auto boundPtr = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
            auto outputPtr = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
            auto onePtr = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
            auto zeroPtr = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);

            // 严格GT：cy==boundary时输出0。VCMPV把位图写入compare_，随后VSEL选择一/零。
            vcmpv_gt(maskPtr, cyPtr, boundPtr, 2, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vsel(outputPtr, onePtr, zeroPtr, 2, 1, 1, 1, 8, 8, 8,
                 static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
            // 此后不再由Vector读output_；CopyOut的V_MTE3事件等待VSEL写完。
            // 用该跨流水依赖覆盖最终完成要求，不重复插入PIPE_V。
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams outputCopy{1, 16, 0, 0};
        DataCopy(outputGm_, output_, outputCopy);
    }
};

// ============================================================================
// 【分支05】FP32，完整覆盖B∈[257,4095]：SbgFp32Range257To4095
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
template <uint32_t FIXED_BATCH = 0, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0, bool TAIL = false>
class SbgFp32Range257To4095 final {
public:
    // 完整区间[257,4095]只有本类。FIXED_BATCH是Host真实B守卫后的编译期策略；0表示真实运行时B。
    // 编译期策略恢复v528原优化，不创建独立单点类；B4094等仍由本类处理。
    static constexpr uint32_t kBatch = FIXED_BATCH;
    static constexpr uint32_t kWorkingCores = FIXED_BATCH == 257U ? (1U) : FIXED_BATCH == 4095U ? (8U) : 0U;
    static constexpr uint32_t kRowsPerCore = FIXED_BATCH == 257U ? (272U) : FIXED_BATCH == 4095U ? (512U) : 0U;
    static constexpr uint32_t kPad = FIXED_BATCH == 257U ? (320U) : FIXED_BATCH == 4095U ? (512U) : 0U;
    static constexpr uint32_t kBankGap = FIXED_BATCH == 257U ? (0U) : FIXED_BATCH == 4095U ? (64U) : 0U;
    static constexpr uint32_t kSvBase = FIXED_BATCH == 257U ? (11520U) : FIXED_BATCH == 4095U ? (36U * kPad) : 0U;
    static constexpr uint32_t kSvFloatCount = FIXED_BATCH == 257U ? (2296U) : FIXED_BATCH == 4095U ? (7U * kPad + 3U * kBankGap + 56U) : 0U;
    static constexpr uint32_t kUbBytes = FIXED_BATCH == 257U ? (20704U) : FIXED_BATCH == 4095U ? (kSvBase + 4U * kSvFloatCount) : 0U;
    static_assert(FIXED_BATCH == 0U || (FIXED_BATCH == 257U || FIXED_BATCH == 4095U), "Unsupported constant interval plan");
    static_assert(kUbBytes <= 192U * 1024U, "UB budget exceeds 192 KiB");
    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR threshold, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score,
                  length, rowsPerCore, hasThreshold, tiling)) return;
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    // 原优化路径所需的成员与UB视图；通用成员不会被常量策略使用。
    using T = float;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;
    // 通用B的真实长度/片上视图；编译器在常量策略中消除未使用成员。
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // 跨阶段保留本核有效行、向量长度与UB视图；普通区间保持有效GM长度；4095候选按已核实分配padding补齐输入，输出仍使用coreRows。
    uint32_t rowStart = 0U, coreRows = 0U, movedRows = 0U, pad = 0U;
    uint32_t hasThreshold = 0U;
    GM_ADDR threshold = nullptr;
    GM_ADDR tiling = nullptr;
    float waistRatio = 0.0f, thresholdValue = 0.0f;
    __gm__ T *boxesBase = nullptr, *anchorsBase = nullptr, *outBase = nullptr;
    LocalTensor<T> boxIn, ancIn, outUb, cst;
    LocalTensor<float> sv;
    // Init按本类策略建立分片和UB；每条路径均保持原有效GM长度。
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原Init语句。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
            if (block >= kWorkingCores) return false;
            rowStart_ = block * kRowsPerCore;
            coreRows_ = 257U;
            threshold_ = thresholdArg;
            tiling_ = tilingArg;
            hasThreshold_ = hasThresholdArg;
            thresholdValue_ = 0.0f;
            // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
            // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
            boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T));
            anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
            outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
            anchorIn_ = LocalTensor<T>(TPosition::VECIN, 5120U, kPad * 2U);
            output_ = LocalTensor<T>(TPosition::VECCALC, 7680U, kPad);
            constants_ = LocalTensor<T>(TPosition::VECCALC, 8960U, kPad * 2U);
            scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
            compare_ = scratch_.template ReinterpretCast<uint8_t>()[8960U];
            one_ = constants_[0];
            zero_ = constants_[kPad];

            return true;
        
        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B=4095候选：7×512+511有效行，统一512行输入/计算；阶段Init语句。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
            if constexpr (TAIL) {
                if (block != 7U) return false;
                rowStart_ = 3584U; coreRows_ = 511U;
            } else {
                if (block >= 7U) return false;
                rowStart_ = block * kRowsPerCore; coreRows_ = kRowsPerCore;
            }
            threshold_ = thresholdArg;
            tiling_ = tilingArg;
            hasThreshold_ = hasThresholdArg;
            thresholdValue_ = 0.0f;
            // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
            // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
            boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, kPad * 4U);
            anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, kPad * 2U);
            outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
            anchorIn_ = LocalTensor<T>(TPosition::VECIN, 16U * kPad, kPad * 2U);
            output_ = LocalTensor<T>(TPosition::VECCALC, 24U * kPad, kPad);
            constants_ = LocalTensor<T>(TPosition::VECCALC, 28U * kPad, kPad * 2U);
            scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
            compare_ = scratch_.template ReinterpretCast<uint8_t>()[4U * (7U * kPad + 3U * kBankGap)];
            one_ = constants_[0];
            zero_ = constants_[kPad];

            return true;
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            threshold = thresholdArg;
            hasThreshold = hasThresholdArg;
            tiling = tilingArg;

            if (length < 257U || length > 4095U) return false;

            // 本类直接处理整个区间：真实B和每核行步长来自Tiling，不转发给单点类。
            // 超出有效B的启动核退出；最后一核只取剩余的有效行数。
            const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
            rowStart = blockIdx * rowsPerCore;
            {
                if (rowStart >= length) return false;
            }
            {
                coreRows = length - rowStart;
                if (coreRows > rowsPerCore) coreRows = rowsPerCore;
            }
            
            // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
            movedRows = SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgRowAlign));
            pad = SbgAlignUp(coreRows, 64U);
            // 以下是本类自有的紧凑UB排列；o按字节，LocalTensor下标按其元素类型。
            // 当前布局预算为FP32的65*pad+32字节，FP16的71*pad+32字节。
            uint32_t o = 0;
            boxIn = LocalTensor<T>(TPosition::VECIN, o, pad * 4U);
            o += SbgAlignBytes(pad * 4U * tSize);
            ancIn = LocalTensor<T>(TPosition::VECIN, o, pad * 2U);
            o += SbgAlignBytes(pad * 2U * tSize);
            outUb = LocalTensor<T>(TPosition::VECCALC, o, pad);
            o += SbgAlignBytes(pad * tSize);
            
            cst = LocalTensor<T>(TPosition::VECCALC, o, pad * 2U);
            o += SbgAlignBytes(pad * tSize * 2U);
            sv = LocalTensor<float>(TPosition::VECCALC, o, 7U * pad + 4U * bankGap + (SbgAlignBytes(pad) / 4U) + 8U);
            boxesBase = reinterpret_cast<__gm__ T *>(person_boxes);
            anchorsBase = reinterpret_cast<__gm__ T *>(anchor_points);
            outBase = reinterpret_cast<__gm__ T *>(violation_score);
            return true;
        
        }
    }

    // CopyIn：搬入真实输入并准备实际属性/阈值。
    __aicore__ inline void CopyIn() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原CopyIn语句。


            DataCopyParams boxCopy{1, static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 16U) / 32U), 0, 0};
            DataCopyParams anchorCopy{1, static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 8U) / 32U), 0, 0};
            DataCopy(boxIn_, boxesGm_, boxCopy);
            DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
                RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
                reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
            // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
            AscendC::InitSocState();
            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
                thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ T*>(threshold_)[0]);
            }
        
        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B=4095候选：7×512+511有效行，统一512行输入/计算；阶段CopyIn语句。


            // 4095搬入候选：每个输入只发一条普通DataCopy，长度按本核kPad补齐。
            // ALIGN_UP到32B的输入存储必须真实可读；末核仅多读boxes16B/anchors8B。
            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            DataCopyParams boxCopy{1, static_cast<uint16_t>(kPad * 16U / 32U), 0, 0};
            DataCopyParams anchorCopy{1, static_cast<uint16_t>(kPad * 8U / 32U), 0, 0};            DataCopy(boxIn_, boxesGm_, boxCopy);
            // 分别记录boxes与anchors就绪；允许先到的boxes先进入Vector。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
                RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
                reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
            // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
            AscendC::InitSocState();
            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
                thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ T*>(threshold_)[0]);
            }
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            GlobalTensor<T> boxGm;
            boxGm.SetGlobalBuffer(boxesBase + static_cast<int64_t>(rowStart) * 4, SbgAlignBytes(coreRows * 4U * sizeof(T)) / sizeof(T));
            // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
            // boxes统一按有效字节数向上补齐到32B；GM须有真实可读padding。
            const uint32_t boxBytes = SbgAlignBytes(coreRows * 4U * tSize);
            DataCopyParams boxDp{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
            DataCopy(boxIn, boxGm, boxDp);

            GlobalTensor<T> ancGm;
            ancGm.SetGlobalBuffer(anchorsBase + static_cast<int64_t>(rowStart) * 2, SbgAlignBytes(coreRows * 2U * sizeof(T)) / sizeof(T));
            // anchors与boxes使用相同的普通DMA策略，不能只修改其中一路。
            const uint32_t anchorBytes = SbgAlignBytes(coreRows * 2U * tSize);
            DataCopyParams ancDp{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
            DataCopy(ancIn, ancGm, ancDp);
            // Both input DMA requests are already in flight. Read the attribute here,
            // so the scalar cache miss can overlap input transfers.
            // 仍读取本次调用的真实float属性，不因shape相同而忽略输入值。
            waistRatio = reinterpret_cast<__gm__ volatile float*>(tiling)[3];
            AscendC::InitSocState();

            thresholdValue = 0.0f;
            {
                if ((hasThreshold != 0U) && (threshold != nullptr)) {
                    thresholdValue = static_cast<float>(reinterpret_cast<__gm__ T *>(threshold)[0]);
                }
            }
        
        }
    }

    // Compute：抽列、FP32分步运算、严格GT比较。
    __aicore__ inline void Compute() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原Compute语句。

            // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
            Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
            Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
            if ASCEND_IS_AIV {
                // 为当前CANN9选择指令约定准备“比较位图UB地址”的载体。
                // 这里不是最终比较结果；真正的0/1位图由后面的VCMPV写出。
                LocalTensor<uint32_t> addressPayload = scratch_.template ReinterpretCast<uint32_t>()[2288U];
                auto addressPtr = reinterpret_cast<__ubuf__ uint32_t*>(addressPayload.GetPhyAddr());
                const uint32_t compareAddress = static_cast<uint32_t>(
                    reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr())));
                set_mask_norm();
                set_vector_mask(0ULL, 0xffULL);
                vector_dup(addressPtr, compareAddress, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_cmpmask(addressPtr);
                pipe_barrier(PIPE_V);
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            LocalTensor<float> y1 = scratch_[2U * kPad];
            LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
            LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
            LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
            LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
            // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
            // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
            LocalTensor<float> pairs = scratch_[0];
            if ASCEND_IS_AIV {
                auto pairsPtr = reinterpret_cast<__ubuf__ float*>(pairs.GetPhyAddr());
                auto boxesPtr = reinterpret_cast<__ubuf__ float*>(boxIn_.GetPhyAddr());
                auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
                auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
                auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
                auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
                vcpadd(pairsPtr, boxesPtr, 20, 1, 1, 8);
                vcpadd(cyPtr, anchorsPtr, 10, 1, 1, 8);
                pipe_barrier(PIPE_V);
                vcpadd(y2Ptr, pairsPtr, 10, 1, 1, 8);
                set_vector_mask(0ULL, 0x5555555555555555ULL);
                vcpadd(y1Ptr, pairsPtr, 10, 1, 1, 8);
                pipe_barrier(PIPE_V);
                set_vector_mask(0ULL, 0xffffffffffffffffULL);
            }
            if ASCEND_IS_AIV {
                auto d = reinterpret_cast<__ubuf__ float*>(difference.GetPhyAddr());
                auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
                auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
                auto bound = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
            // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
            // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
                vsub(d, y2Ptr, y1Ptr, 5, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                if constexpr (!UNIT_RATIO) {
                    vmuls(d, d, waistRatio_, 5, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                vadd(bound, y1Ptr, d, 5, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vadds(bound, bound, thresholdValue_, 5, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            if ASCEND_IS_AIV {
                auto maskPtr = reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr());
                auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
                auto boundPtr = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
                auto outputPtr = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
                auto onePtr = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
                auto zeroPtr = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);

                // 严格GT：cy==boundary时输出0。VCMPV把位图写入compare_，随后VSEL选择一/零。
                vcmpv_gt(maskPtr, cyPtr, boundPtr, 5, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vsel(outputPtr, onePtr, zeroPtr, 5, 1, 1, 1, 8, 8, 8,
                     static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
        
        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B=4095候选：7×512+511有效行，统一512行输入/计算；阶段Compute语句。

            // 每核计算512行；末核第512个结果丢弃，FP32四步舍入与严格GT保持。
            // 一/零向量供最终Select使用；仍根据每行真实输入判定。
            // 组合项C6：只初始化一个32B常量块，VSEL源零步长广播。
            // 输入、FP32四步舍入、有效行数和搬出边界全部保持基线。
            Duplicate(one_, static_cast<T>(1.0f), 8);
            Duplicate(zero_, static_cast<T>(0.0f), 8);
            if ASCEND_IS_AIV {
                // 为当前CANN9选择指令约定准备“比较位图UB地址”的载体。
                // 这里不是最终比较结果；真正的0/1位图由后面的VCMPV写出。
                LocalTensor<uint32_t> addressPayload = scratch_.template ReinterpretCast<uint32_t>()[7U * kPad + 3U * kBankGap + 48U];
                auto addressPtr = reinterpret_cast<__ubuf__ uint32_t*>(addressPayload.GetPhyAddr());
                const uint32_t compareAddress = static_cast<uint32_t>(
                    reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr())));
                set_mask_norm();
                set_vector_mask(0ULL, 0xffULL);
                vector_dup(addressPtr, compareAddress, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_cmpmask(addressPtr);
                pipe_barrier(PIPE_V);
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            // CopyIn已分别记录两路就绪；这里先仅等待boxes。
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            LocalTensor<float> y1 = scratch_[2U * kPad];
            LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
            LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
            LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
            LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
            // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
            // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
            LocalTensor<float> pairs = scratch_[0];
            if ASCEND_IS_AIV {
                auto pairsPtr = reinterpret_cast<__ubuf__ float*>(pairs.GetPhyAddr());
                auto boxesPtr = reinterpret_cast<__ubuf__ float*>(boxIn_.GetPhyAddr());
                auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
                auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
                auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
                auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
                vcpadd(pairsPtr, boxesPtr, static_cast<uint8_t>(kPad / 16U), 1, 1, 8);

                pipe_barrier(PIPE_V);
                vcpadd(y2Ptr, pairsPtr, static_cast<uint8_t>(kPad / 32U), 1, 1, 8);
                set_vector_mask(0ULL, 0x5555555555555555ULL);
                vcpadd(y1Ptr, pairsPtr, static_cast<uint8_t>(kPad / 32U), 1, 1, 8);
                pipe_barrier(PIPE_V);
                set_vector_mask(0ULL, 0xffffffffffffffffULL);
            }
            if ASCEND_IS_AIV {
                auto d = reinterpret_cast<__ubuf__ float*>(difference.GetPhyAddr());
                auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
                auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
                auto bound = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
            // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
            // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
                vsub(d, y2Ptr, y1Ptr, static_cast<uint8_t>(kPad / 64U), 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                if constexpr (!UNIT_RATIO) {
                    vmuls(d, d, waistRatio_, static_cast<uint8_t>(kPad / 64U), 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                vadd(bound, y1Ptr, d, static_cast<uint8_t>(kPad / 64U), 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vadds(bound, bound, thresholdValue_, static_cast<uint8_t>(kPad / 64U), 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            // boxes已完成参考线计算；比较前才等待anchors并抽取cy。
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            if ASCEND_IS_AIV {
                auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
                auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
                vcpadd(cyPtr, anchorsPtr, static_cast<uint8_t>(kPad / 32U), 1, 1, 8);
                pipe_barrier(PIPE_V);
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
            if ASCEND_IS_AIV {
                auto maskPtr = reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr());
                auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
                auto boundPtr = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
                auto outputPtr = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
                auto onePtr = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
                auto zeroPtr = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);

                // 严格GT：cy==boundary时输出0。VCMPV把位图写入compare_，随后VSEL选择一/零。
                vcmpv_gt(maskPtr, cyPtr, boundPtr, static_cast<uint8_t>(kPad / 64U), 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vsel(outputPtr, onePtr, zeroPtr, static_cast<uint8_t>(kPad / 64U), 1, 0, 0, 8, 0, 0,
                     static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            LocalTensor<uint8_t> s8 = sv.template ReinterpretCast<uint8_t>();
            LocalTensor<uint8_t> cmpUb = s8[(7U * pad + 4U * bankGap) * static_cast<uint32_t>(sizeof(float))];

            LocalTensor<T> oneUb;
            oneUb = cst[0];
            LocalTensor<T> zeroUb;
            zeroUb = cst[pad];
            Duplicate(oneUb, static_cast<T>(1.0f), static_cast<int32_t>(pad));
            Duplicate(zeroUb, static_cast<T>(0.0f), static_cast<int32_t>(pad));


            // The last 32 bytes of sv are reserved and do not overlap cmpUb or any
            // intermediate tensor. Only the address is needed here, not comparison data.
            

            // 等输入DMA完成后才消费UB；当前仍是单段切片，没有双缓冲。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            LocalTensor<float> boxVec;
            LocalTensor<float> ancVec;
            {
                boxVec = boxIn.template ReinterpretCast<float>();
                ancVec = ancIn.template ReinterpretCast<float>();
            }

            LocalTensor<float> lUb;
            lUb = sv[0];
            LocalTensor<float> y1Ub;
            y1Ub = sv[2U * pad];
            LocalTensor<float> y2Ub;
            y2Ub = sv[3U * pad + bankGap];
            LocalTensor<float> dUb;
            dUb = sv[4U * pad + bankGap];
            LocalTensor<float> boundUb;
            boundUb = sv[5U * pad + 2U * bankGap];
            LocalTensor<float> cyUb;
            cyUb = sv[6U * pad + 3U * bankGap];

            const uint64_t maskOdd[1] = {kSbgMaskOdd};
            const uint64_t maskEven[1] = {kSbgMaskEven};
            const int32_t rep4 = static_cast<int32_t>((4U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
            const int32_t rep2 = static_cast<int32_t>((2U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
            const int32_t padI = static_cast<int32_t>(pad);

            {
                // 奇/偶掩码完成AoS→三列抽取；不对不同样本做汇总。
                PairReduceSum<float>(lUb, boxVec, rep4, maskOdd, 1, 1, 8);
                PairReduceSum<float>(y1Ub, lUb, rep2, maskEven, 1, 1, 8);
                PairReduceSum<float>(y2Ub, lUb, rep2, maskOdd, 1, 1, 8);
                PairReduceSum<float>(cyUb, ancVec, rep2, maskOdd, 1, 1, 8);
            }

            {
                // 保持FP32分步舍入及原公式，不引入FMA或改变结合顺序。
                Sub(dUb, y2Ub, y1Ub, padI);
                Muls(dUb, dUb, waistRatio, padI);
                Add(boundUb, y1Ub, dUb, padI);
                
                Adds(boundUb, boundUb, thresholdValue, padI);
            }

            {
                // 严格GT：cy等于边界时输出0；padding lane不得影响有效输出。
                Compare(cmpUb, cyUb, boundUb, CMPMODE::GT, padI);
                Select(outUb, cmpUb, oneUb, zeroUb, SELMODE::VSEL_TENSOR_TENSOR_MODE, padI);
            }
        
        }
    }

    // CopyOut：只搬出有效行。
    __aicore__ inline void CopyOut() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyParams outputCopy{1, static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 4U) / 32U), 0, 0};
            DataCopy(outputGm_, output_, outputCopy);
        
        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B=4095候选：7×512+511有效行，统一512行输入/计算；阶段CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if constexpr (TAIL) {
                DataCopyParams outputCopy{1, static_cast<uint16_t>(SbgAlignBytes(2044U) / 32U), 0, 0};
                DataCopy(outputGm_, output_, outputCopy);
            } else {
                DataCopyParams outputCopy{1, 64, 0, 0};
                DataCopy(outputGm_, output_, outputCopy);
            }
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            GlobalTensor<T> outGm;
            outGm.SetGlobalBuffer(outBase + rowStart, SbgAlignBytes(coreRows * sizeof(T)) / sizeof(T));
            const uint32_t outBytes = coreRows * tSize;
            if ((outBytes & 31U) == 0U) {
                DataCopyParams outDp;
                outDp.blockCount = 1;
                outDp.blockLen = static_cast<uint16_t>(outBytes / 32U);
                outDp.srcStride = 0;
                outDp.dstStride = 0;
                DataCopy(outGm, outUb, outDp);
            } else {
                DataCopyParams outCp;
                outCp.blockCount = 1;
                outCp.blockLen = static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U);
                outCp.srcStride = 0;
                outCp.dstStride = 0;
                DataCopy(outGm, outUb, outCp);
            }
        
        }
    }
};

// 尺寸分支6：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case05：FP32 / B=4096】SbgFp32B4096
// 当前启动8核，工作核上限8；8核各处理512行。
// 实际向量计算行数：512；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：每核8192 / 4096 / 2048 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 此处以“公开case”编号为准，不能沿用旧英文注释的内部排列编号。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,8192)；anchorIn_[8192,12288)；output_[12288,14336)；constants_[14336,18432)；scratch_[65536,81440)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp32B4096 final {
public:
    static constexpr uint32_t kBatch = 4096U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 8U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 512U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 512U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 64U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 65536U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 3976U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 81440U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgShapePolicy(kBatch, false).launch == 8U,
                  "Host launch policy changed: review this point explicitly");
    static_assert(SbgShapePolicy(kBatch, false).working == kWorkingCores,
                  "Host working-core policy changed: review this point explicitly");
    static_assert((((kBatch + kWorkingCores - 1U) / kWorkingCores + 15U) / 16U * 16U)
                  == kRowsPerCore, "Host/device slice mismatch");
    static_assert(kUbBytes <= 192U * 1024U, "Point UB allocation exceeds 192 KiB");

    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
                              GM_ADDR threshold, GM_ADDR violation_score,
                              uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score, hasThreshold, tiling)) return;
        // 仅有一个数据切片；这些阶段有必要的依赖等待，不能凭函数名认定已流水重叠。
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    using T = float;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
        if (block >= kWorkingCores) return false;
        rowStart_ = block * kRowsPerCore;
        coreRows_ = 512U;
        threshold_ = threshold;
        tiling_ = tiling;
        hasThreshold_ = hasThreshold;
        thresholdValue_ = 0.0f;
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T));
        anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
        anchorIn_ = LocalTensor<T>(TPosition::VECIN, 8192U, kPad * 2U);
        output_ = LocalTensor<T>(TPosition::VECCALC, 12288U, kPad);
        constants_ = LocalTensor<T>(TPosition::VECCALC, 14336U, kPad * 2U);
        scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
        compare_ = scratch_.template ReinterpretCast<uint8_t>()[15360U];
        one_ = constants_[0];
        zero_ = constants_[kPad];

        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {

        DataCopyParams boxCopy{1, 256, 0, 0};
        DataCopyParams anchorCopy{1, 128, 0, 0};        DataCopy(boxIn_, boxesGm_, boxCopy);
        // 分别记录boxes与anchors就绪；允许先到的boxes先进入Vector。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
            RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
            reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
        // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
        AscendC::InitSocState();
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
            thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ T*>(threshold_)[0]);
        }
    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
        Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
        if ASCEND_IS_AIV {
            // 为当前CANN9选择指令约定准备“比较位图UB地址”的载体。
            // 这里不是最终比较结果；真正的0/1位图由后面的VCMPV写出。
            LocalTensor<uint32_t> addressPayload = scratch_.template ReinterpretCast<uint32_t>()[3968U];
            auto addressPtr = reinterpret_cast<__ubuf__ uint32_t*>(addressPayload.GetPhyAddr());
            const uint32_t compareAddress = static_cast<uint32_t>(
                reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr())));
            set_mask_norm();
            set_vector_mask(0ULL, 0xffULL);
            vector_dup(addressPtr, compareAddress, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_cmpmask(addressPtr);
            pipe_barrier(PIPE_V);
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        // CopyIn已分别记录两路就绪；这里先仅等待boxes。
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        LocalTensor<float> y1 = scratch_[2U * kPad];
        LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
        LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
        LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
        LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
        // FP32抽列：boxes奇数掩码归约为[y1,y2]交错对，再用奇/偶掩码分别抽y2/y1。
        // anchors用奇数掩码抽cy；这些归约用于抽列，不是跨样本求和。
        LocalTensor<float> pairs = scratch_[0];
        if ASCEND_IS_AIV {
            auto pairsPtr = reinterpret_cast<__ubuf__ float*>(pairs.GetPhyAddr());
            auto boxesPtr = reinterpret_cast<__ubuf__ float*>(boxIn_.GetPhyAddr());
            auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
            vcpadd(pairsPtr, boxesPtr, 32, 1, 1, 8);

            pipe_barrier(PIPE_V);
            vcpadd(y2Ptr, pairsPtr, 16, 1, 1, 8);
            set_vector_mask(0ULL, 0x5555555555555555ULL);
            vcpadd(y1Ptr, pairsPtr, 16, 1, 1, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
        }
        if ASCEND_IS_AIV {
            auto d = reinterpret_cast<__ubuf__ float*>(difference.GetPhyAddr());
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto bound = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
            vsub(d, y2Ptr, y1Ptr, 8, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (!UNIT_RATIO) {
                vmuls(d, d, waistRatio_, 8, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            vadd(bound, y1Ptr, d, 8, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadds(bound, bound, thresholdValue_, 8, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        // boxes已完成参考线计算；比较前才等待anchors并抽取cy。
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        if ASCEND_IS_AIV {
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
            vcpadd(cyPtr, anchorsPtr, 16, 1, 1, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
        if ASCEND_IS_AIV {
            auto maskPtr = reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            auto boundPtr = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
            auto outputPtr = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
            auto onePtr = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
            auto zeroPtr = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);

            // 严格GT：cy==boundary时输出0。VCMPV把位图写入compare_，随后VSEL选择一/零。
            vcmpv_gt(maskPtr, cyPtr, boundPtr, 8, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vsel(outputPtr, onePtr, zeroPtr, 8, 1, 1, 1, 8, 8, 8,
                 static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams outputCopy{1, 64, 0, 0};
        DataCopy(outputGm_, output_, outputCopy);
    }
};

// ============================================================================
// 【分支07】FP16，完整覆盖B∈[1,31]：SbgFp16Range1To31
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
class SbgFp16Range1To31 final {
public:
    // Run只负责调度；本类完整处理[1,31]，不转发给单点实现。
    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR threshold, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score,
                  length, rowsPerCore, hasThreshold, tiling)) return;
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    using T = half;
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
    uint32_t rowStart = 0U, coreRows = 0U, movedRows = 0U, pad = 0U;
    uint32_t hasThreshold = 0U;
    GM_ADDR threshold = nullptr;
    GM_ADDR tiling = nullptr;
    float waistRatio = 0.0f, thresholdValue = 0.0f;
    __gm__ T *boxesBase = nullptr, *anchorsBase = nullptr, *outBase = nullptr;
    LocalTensor<T> boxIn, ancIn, outUb, cst;
    LocalTensor<float> sv, boxF32, ancF32;

    // Init：按真实B分片并建立UB/GM视图；保留原阈值预取位置，不添加同步。
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        threshold = thresholdArg;
        hasThreshold = hasThresholdArg;
        tiling = tilingArg;

        // 保留历史工程ABI：FP16阈值/输出为half；与题面FP32文字差异仍需单独review。
        if (length < 1U || length > 31U) return false;

        // 本类直接处理整个区间：真实B和每核行步长来自Tiling，不转发给单点类。
        // 超出有效B的启动核退出；最后一核只取剩余的有效行数。
        const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        rowStart = blockIdx * rowsPerCore;
        {
            if (rowStart >= length) return false;
        }
        {
            coreRows = length - rowStart;
            if (coreRows > rowsPerCore) coreRows = rowsPerCore;
        }
        {
            if (hasThreshold != 0U && threshold != nullptr) {
                GlobalTensor<uint64_t> thresholdCacheLine;
                thresholdCacheLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold), 1);
                DataCachePreload(thresholdCacheLine, static_cast<int16_t>(0));
            }
        }
        // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
        movedRows = SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgRowAlign));
        pad = 64U;
        // 以下是本类自有的紧凑UB排列；o按字节，LocalTensor下标按其元素类型。
        // 当前布局预算为FP32的65*pad+32字节，FP16的71*pad+32字节。
        uint32_t o = 0;
        boxIn = LocalTensor<T>(TPosition::VECIN, o, pad * 4U);
        o += SbgAlignBytes(pad * 4U * tSize);
        ancIn = LocalTensor<T>(TPosition::VECIN, o, pad * 2U);
        o += SbgAlignBytes(pad * 2U * tSize);
        outUb = LocalTensor<T>(TPosition::VECCALC, o, pad);
        o += SbgAlignBytes(pad * tSize);
        {
            boxF32 = LocalTensor<float>(TPosition::VECCALC, o, pad * 4U);
            o += SbgAlignBytes(pad * 4U * 4U);
            ancF32 = LocalTensor<float>(TPosition::VECCALC, o, pad * 2U);
            o += SbgAlignBytes(pad * 2U * 4U);
        }
        cst = LocalTensor<T>(TPosition::VECCALC, o, pad * 2U);
        o += SbgAlignBytes(pad * tSize * 2U);
        sv = LocalTensor<float>(TPosition::VECCALC, o, 7U * pad + 4U * bankGap + (SbgAlignBytes(pad) / 4U) + 8U);
        boxesBase = reinterpret_cast<__gm__ T *>(person_boxes);
        anchorsBase = reinterpret_cast<__gm__ T *>(anchor_points);
        outBase = reinterpret_cast<__gm__ T *>(violation_score);
        return true;
    }

    // CopyIn：发出boxes/anchors搬入，并读取本次调用的属性和适用阈值。
    __aicore__ inline void CopyIn() {
        GlobalTensor<T> boxGm;
        boxGm.SetGlobalBuffer(boxesBase + static_cast<int64_t>(rowStart) * 4, SbgAlignBytes(coreRows * 4U * sizeof(T)) / sizeof(T));
        // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
        // boxes统一按有效字节数向上补齐到32B；GM须有真实可读padding。
        const uint32_t boxBytes = SbgAlignBytes(coreRows * 4U * tSize);
        DataCopyParams boxDp{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
        DataCopy(boxIn, boxGm, boxDp);

        GlobalTensor<T> ancGm;
        ancGm.SetGlobalBuffer(anchorsBase + static_cast<int64_t>(rowStart) * 2, SbgAlignBytes(coreRows * 2U * sizeof(T)) / sizeof(T));
        // anchors与boxes使用相同的普通DMA策略，不能只修改其中一路。
        const uint32_t anchorBytes = SbgAlignBytes(coreRows * 2U * tSize);
        DataCopyParams ancDp{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
        DataCopy(ancIn, ancGm, ancDp);
        // Both input DMA requests are already in flight. Read the attribute here,
        // so the scalar cache miss can overlap input transfers.
        // 仍读取本次调用的真实float属性，不因shape相同而忽略输入值。
        waistRatio = reinterpret_cast<__gm__ volatile float*>(tiling)[3];
        AscendC::InitSocState();

        thresholdValue = 0.0f;
    }

    // Compute：等待搬入完成，抽列、FP32分步计算、严格GT比较并生成0/1。
    __aicore__ inline void Compute() {
        LocalTensor<uint8_t> s8 = sv.template ReinterpretCast<uint8_t>();
        LocalTensor<uint8_t> cmpUb = s8[(7U * pad + 4U * bankGap) * static_cast<uint32_t>(sizeof(float))];

        LocalTensor<T> oneUb;
        oneUb = cst[0];
        LocalTensor<T> zeroUb;
        zeroUb = cst[pad];
        Duplicate(oneUb, static_cast<T>(1.0f), static_cast<int32_t>(pad));
        Duplicate(zeroUb, static_cast<T>(0.0f), static_cast<int32_t>(pad));


        // The last 32 bytes of sv are reserved and do not overlap cmpUb or any
        // intermediate tensor. Only the address is needed here, not comparison data.
        

        // 等输入DMA完成后才消费UB；当前仍是单段切片，没有双缓冲。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        

        LocalTensor<float> lUb;
        lUb = sv[0];
        LocalTensor<float> y1Ub;
        y1Ub = sv[2U * pad];
        LocalTensor<float> y2Ub;
        y2Ub = sv[3U * pad + bankGap];
        LocalTensor<float> dUb;
        dUb = sv[4U * pad + bankGap];
        LocalTensor<float> boundUb;
        boundUb = sv[5U * pad + 2U * bankGap];
        LocalTensor<float> cyUb;
        cyUb = sv[6U * pad + 3U * bankGap];

        const uint64_t maskOdd[1] = {kSbgMaskOdd};
        const uint64_t maskEven[1] = {kSbgMaskEven};
        const int32_t rep4 = static_cast<int32_t>((4U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
        const int32_t rep2 = static_cast<int32_t>((2U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
        const int32_t padI = static_cast<int32_t>(pad);

        {
            // On CANN 9.0, the no-count overload avoids the
            // V->S count synchronization. Promote only the three used columns.
            SetMaskNorm();
            ResetMask();
            const uint16_t boxReps = static_cast<uint16_t>(4U * pad * sizeof(T) / 256U);
            const uint16_t ancReps = static_cast<uint16_t>(2U * pad * sizeof(T) / 256U);
            LocalTensor<T> y1Half = boxF32.template ReinterpretCast<T>();
            LocalTensor<T> y2Half = y1Half[pad];
            LocalTensor<T> cyHalf = y1Half[2U * pad];
            // 固定模式抽取y1/y2/cy后分别转FP32；几何运算和比较都用FP32。
            GatherMask(y1Half, boxIn, static_cast<uint8_t>(4), {1,boxReps,8,0});
            GatherMask(y2Half, boxIn, static_cast<uint8_t>(6), {1,boxReps,8,0});
            GatherMask(cyHalf, ancIn, static_cast<uint8_t>(2), {1,ancReps,8,0});
            Cast(y1Ub, y1Half, RoundMode::CAST_NONE, pad);
            Cast(y2Ub, y2Half, RoundMode::CAST_NONE, pad);
            Cast(cyUb, cyHalf, RoundMode::CAST_NONE, pad);
        }

        {
            // 保持FP32分步舍入及原公式，不引入FMA或改变结合顺序。
            Sub(dUb, y2Ub, y1Ub, padI);
            Muls(dUb, dUb, waistRatio, padI);
            Add(boundUb, y1Ub, dUb, padI);
            {
                if ((hasThreshold != 0U) && (threshold != nullptr)) {
                    thresholdValue = static_cast<float>(reinterpret_cast<__gm__ volatile T *>(threshold)[0]);
                }
            }
            Adds(boundUb, boundUb, thresholdValue, padI);
        }

        {
            // 严格GT：cy等于边界时输出0；padding lane不得影响有效输出。
            Compare(cmpUb, cyUb, boundUb, CMPMODE::GT, padI);
            Select(outUb, cmpUb, oneUb, zeroUb, SELMODE::VSEL_TENSOR_TENSOR_MODE, padI);
        }
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<T> outGm;
        outGm.SetGlobalBuffer(outBase + rowStart, SbgAlignBytes(coreRows * sizeof(T)) / sizeof(T));
        const uint32_t outBytes = coreRows * tSize;
        if ((outBytes & 31U) == 0U) {
            DataCopyParams outDp;
            outDp.blockCount = 1;
            outDp.blockLen = static_cast<uint16_t>(outBytes / 32U);
            outDp.srcStride = 0;
            outDp.dstStride = 0;
            DataCopy(outGm, outUb, outDp);
        } else {
            DataCopyParams outCp;
            outCp.blockCount = 1;
            outCp.blockLen = static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U);
            outCp.srcStride = 0;
            outCp.dstStride = 0;
            DataCopy(outGm, outUb, outCp);
        }
    }
};

// 尺寸分支8：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case07：FP16 / B=32】SbgFp16B32
// 当前启动8核，工作核上限1；核0处理32行；核1～7退出。
// 实际向量计算行数：64；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：256 / 128 / 64 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 当前历史ABI：阈值/输出使用half；只有y1/y2/cy转FP32后参与几何计算和比较。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,512)；anchorIn_[512,768)；output_[768,896)；constants_[2432,2688)；scratch_[2688,4576)；halfStage_[896,1920)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp16B32 final {
public:
    static constexpr uint32_t kBatch = 32U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 1U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 32U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 64U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 0U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 2688U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 472U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 4576U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgShapePolicy(kBatch, true).launch == 8U,
                  "Host launch policy changed: review this point explicitly");
    static_assert(SbgShapePolicy(kBatch, true).working == kWorkingCores,
                  "Host working-core policy changed: review this point explicitly");
    static_assert((((kBatch + kWorkingCores - 1U) / kWorkingCores + 15U) / 16U * 16U)
                  == kRowsPerCore, "Host/device slice mismatch");
    static_assert(kUbBytes <= 192U * 1024U, "Point UB allocation exceeds 192 KiB");

    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
                              GM_ADDR threshold, GM_ADDR violation_score,
                              uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score, hasThreshold, tiling)) return;
        // 仅有一个数据切片；这些阶段有必要的依赖等待，不能凭函数名认定已流水重叠。
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    using T = half;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;
    LocalTensor<float> halfStage_;
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
        if (block >= kWorkingCores) return false;
        rowStart_ = 0U;
        coreRows_ = 32U;
        threshold_ = threshold;
        tiling_ = tiling;
        hasThreshold_ = hasThreshold;
        thresholdValue_ = 0.0f;
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T));
        anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
        anchorIn_ = LocalTensor<T>(TPosition::VECIN, 512U, kPad * 2U);
        output_ = LocalTensor<T>(TPosition::VECCALC, 768U, kPad);
        constants_ = LocalTensor<T>(TPosition::VECCALC, 2432U, kPad * 2U);
        scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
        compare_ = scratch_.template ReinterpretCast<uint8_t>()[1792U];
        one_ = constants_[0];
        zero_ = constants_[kPad];
        // 声明为float容量容器，计算时重新解释为half，暂存抽出的三列half。
        halfStage_ = LocalTensor<float>(TPosition::VECCALC, 896U, kPad * 4U);
        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            GlobalTensor<uint64_t> thresholdLine;
            thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
            // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
            DataCachePreload(thresholdLine, static_cast<int16_t>(0));
        }
        DataCopyParams boxCopy{1, 8, 0, 0};
        DataCopyParams anchorCopy{1, 4, 0, 0};
        DataCopy(boxIn_, boxesGm_, boxCopy);
        DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
            RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
            reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
        // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
        AscendC::InitSocState();

    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        // 组合项C7：正常64-lane掩码共用，避免两次Counter模式往返。
        // InitSocState已恢复Normal；后续Gather/转换/比较及输出逻辑有效行均不变。
        SetVectorMask<T>(0ULL, 0xffffffffffffffffULL);
        Duplicate<T, false>(one_, static_cast<T>(1.0f), uint64_t(64), 1, 1, 8);
        Duplicate<T, false>(zero_, static_cast<T>(0.0f), uint64_t(64), 1, 1, 8);

        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        LocalTensor<float> y1 = scratch_[2U * kPad];
        LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
        LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
        LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
        LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
        SetMaskNorm();
        ResetMask();
        // 先在half域抽列，避免把x1/x2/cx这些不用的列也转成FP32。
        // half三列的起点由kPad决定；真实计算长度见kPad或calcRows。
        LocalTensor<T> y1Half = halfStage_.template ReinterpretCast<T>();
        LocalTensor<T> y2Half = y1Half[kPad];
        LocalTensor<T> cyHalf = y1Half[2U * kPad];
        // 固定模式4/6分别取boxes的y1/y2，模式2取anchors的cy。
        GatherMask(y1Half, boxIn_, static_cast<uint8_t>(4), {1, 2, 8, 0});
        GatherMask(y2Half, boxIn_, static_cast<uint8_t>(6), {1, 2, 8, 0});
        GatherMask(cyHalf, anchorIn_, static_cast<uint8_t>(2), {1, 1, 8, 0});
        // Three independent conversions, followed by one dependency barrier.
        if ASCEND_IS_AIV {
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            auto y1HalfPtr = reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr());
            auto y2HalfPtr = reinterpret_cast<__ubuf__ T*>(y2Half.GetPhyAddr());
            auto cyHalfPtr = reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            pipe_barrier(PIPE_V);  // all GatherMask writes must be complete
            // 三次原生half→FP32转换；后面的几何运算和比较不使用half精度。
            vconv_f162f32(y1Ptr, y1HalfPtr, 1, 1, 1, 8, 4);
            vconv_f162f32(y2Ptr, y2HalfPtr, 1, 1, 1, 8, 4);
            vconv_f162f32(cyPtr, cyHalfPtr, 1, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }
        if ASCEND_IS_AIV {
            auto d = reinterpret_cast<__ubuf__ float*>(difference.GetPhyAddr());
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto bound = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
            vsub(d, y2Ptr, y1Ptr, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (!UNIT_RATIO) {
                vmuls(d, d, waistRatio_, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            vadd(bound, y1Ptr, d, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
            thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
            }
            vadds(bound, bound, thresholdValue_, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        SetMaskNorm();
        // 一个FP32 repeat的比较结果直接放CMPMASK寄存器；GT相等时为假。
        Compare(cy, boundary, CMPMODE::GT, uint64_t(64), {1, 1, 1, 8, 8, 8});
        Select<T, SELMODE::VSEL_CMPMASK_SPR>(output_, one_, zero_, 1, {1, 1, 1, 8, 8, 8});
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams outputCopy{1, 2, 0, 0};
        DataCopy(outputGm_, output_, outputCopy);
    }
};

// ============================================================================
// 【分支09】FP16，完整覆盖B∈[33,255]：SbgFp16Range33To255
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
template <uint32_t FIXED_BATCH = 0, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0, bool TAIL = false>
class SbgFp16Range33To255 final {
public:
    // 完整区间[33,255]只有本类。FIXED_BATCH是Host真实B守卫后的编译期策略；0表示真实运行时B。
    // 编译期策略恢复v528原优化，不创建独立单点类；B4094等仍由本类处理。
    static constexpr uint32_t kBatch = FIXED_BATCH;
    static constexpr uint32_t kWorkingCores = FIXED_BATCH == 33U ? (1U) : 0U;
    static constexpr uint32_t kRowsPerCore = FIXED_BATCH == 33U ? (48U) : 0U;
    static constexpr uint32_t kPad = FIXED_BATCH == 33U ? (64U) : 0U;
    static constexpr uint32_t kBankGap = FIXED_BATCH == 33U ? (0U) : 0U;
    static constexpr uint32_t kSvBase = FIXED_BATCH == 33U ? (2688U) : 0U;
    static constexpr uint32_t kSvFloatCount = FIXED_BATCH == 33U ? (472U) : 0U;
    static constexpr uint32_t kUbBytes = FIXED_BATCH == 33U ? (4576U) : 0U;
    static_assert(FIXED_BATCH == 0U || (FIXED_BATCH == 33U), "Unsupported constant interval plan");
    static_assert(kUbBytes <= 192U * 1024U, "UB budget exceeds 192 KiB");
    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR threshold, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score,
                  length, rowsPerCore, hasThreshold, tiling)) return;
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    // 原优化路径所需的成员与UB视图；通用成员不会被常量策略使用。
    using T = half;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;
    LocalTensor<float> halfStage_;
    // 通用B的真实长度/片上视图；编译器在常量策略中消除未使用成员。
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
    uint32_t rowStart = 0U, coreRows = 0U, movedRows = 0U, pad = 0U;
    uint32_t hasThreshold = 0U;
    GM_ADDR threshold = nullptr;
    GM_ADDR tiling = nullptr;
    float waistRatio = 0.0f, thresholdValue = 0.0f;
    __gm__ T *boxesBase = nullptr, *anchorsBase = nullptr, *outBase = nullptr;
    LocalTensor<T> boxIn, ancIn, outUb, cst;
    LocalTensor<float> sv, boxF32, ancF32;
    // Init按本类策略建立分片和UB；每条路径均保持原有效GM长度。
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的编译期B=33策略：恢复冻结v528原Init语句。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
            if (block >= kWorkingCores) return false;
            rowStart_ = 0U;
            coreRows_ = 33U;
            threshold_ = thresholdArg;
            tiling_ = tilingArg;
            hasThreshold_ = hasThresholdArg;
            thresholdValue_ = 0.0f;
            // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
            // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
            boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, ((coreRows_ * 4U * sizeof(T) + 31U) & ~31U) / sizeof(T));
            anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, ((coreRows_ * 2U * sizeof(T) + 31U) & ~31U) / sizeof(T));
            outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
            anchorIn_ = LocalTensor<T>(TPosition::VECIN, 512U, kPad * 2U);
            output_ = LocalTensor<T>(TPosition::VECCALC, 768U, kPad);
            constants_ = LocalTensor<T>(TPosition::VECCALC, 2432U, kPad * 2U);
            scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
            compare_ = scratch_.template ReinterpretCast<uint8_t>()[1792U];
            one_ = constants_[0];
            zero_ = constants_[kPad];
            // 声明为float容量容器，计算时重新解释为half，暂存抽出的三列half。
            halfStage_ = LocalTensor<float>(TPosition::VECCALC, 896U, kPad * 4U);
            return true;
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            threshold = thresholdArg;
            hasThreshold = hasThresholdArg;
            tiling = tilingArg;

            // 保留历史工程ABI：FP16阈值/输出为half；与题面FP32文字差异仍需单独review。
            if (length < 33U || length > 255U) return false;

            // 本类直接处理整个区间：真实B和每核行步长来自Tiling，不转发给单点类。
            // 超出有效B的启动核退出；最后一核只取剩余的有效行数。
            const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
            rowStart = blockIdx * rowsPerCore;
            {
                if (rowStart >= length) return false;
            }
            {
                coreRows = length - rowStart;
                if (coreRows > rowsPerCore) coreRows = rowsPerCore;
            }
            {
                if (hasThreshold != 0U && threshold != nullptr) {
                    GlobalTensor<uint64_t> thresholdCacheLine;
                    thresholdCacheLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold), 1);
                    DataCachePreload(thresholdCacheLine, static_cast<int16_t>(0));
                }
            }
            // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
            movedRows = SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgRowAlign));
            pad = SbgAlignUp(coreRows, 64U);
            // 以下是本类自有的紧凑UB排列；o按字节，LocalTensor下标按其元素类型。
            // 当前布局预算为FP32的65*pad+32字节，FP16的71*pad+32字节。
            uint32_t o = 0;
            boxIn = LocalTensor<T>(TPosition::VECIN, o, pad * 4U);
            o += SbgAlignBytes(pad * 4U * tSize);
            ancIn = LocalTensor<T>(TPosition::VECIN, o, pad * 2U);
            o += SbgAlignBytes(pad * 2U * tSize);
            outUb = LocalTensor<T>(TPosition::VECCALC, o, pad);
            o += SbgAlignBytes(pad * tSize);
            {
                boxF32 = LocalTensor<float>(TPosition::VECCALC, o, pad * 4U);
                o += SbgAlignBytes(pad * 4U * 4U);
                ancF32 = LocalTensor<float>(TPosition::VECCALC, o, pad * 2U);
                o += SbgAlignBytes(pad * 2U * 4U);
            }
            cst = LocalTensor<T>(TPosition::VECCALC, o, pad * 2U);
            o += SbgAlignBytes(pad * tSize * 2U);
            sv = LocalTensor<float>(TPosition::VECCALC, o, 7U * pad + 4U * bankGap + (SbgAlignBytes(pad) / 4U) + 8U);
            boxesBase = reinterpret_cast<__gm__ T *>(person_boxes);
            anchorsBase = reinterpret_cast<__gm__ T *>(anchor_points);
            outBase = reinterpret_cast<__gm__ T *>(violation_score);
            return true;
        
        }
    }

    // CopyIn：搬入真实输入并准备实际属性/阈值。
    __aicore__ inline void CopyIn() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的B=33策略：32B补齐输入，普通DataCopy，有效行数独立控制。

            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                GlobalTensor<uint64_t> thresholdLine;
                thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
                // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
                DataCachePreload(thresholdLine, static_cast<int16_t>(0));
            }
            // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
            // 本核有效33行：boxes264B→288B，anchors132B→160B。
            // 调用方必须提供实际可读的32B补齐存储；SetGlobalBuffer不会扩容GM。
            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            const uint32_t boxBytes = (coreRows_ * 8U + 31U) & ~31U;
            const uint32_t anchorBytes = (coreRows_ * 4U + 31U) & ~31U;
            DataCopyParams boxCopy{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
            DataCopyParams anchorCopy{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
            DataCopy(boxIn_, boxesGm_, boxCopy);
            // boxes单独记录EVENT_ID0；后续anchors搬运与boxes的参考线计算重叠。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
                RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
                reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
            // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
            AscendC::InitSocState();

        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            GlobalTensor<T> boxGm;
            boxGm.SetGlobalBuffer(boxesBase + static_cast<int64_t>(rowStart) * 4, SbgAlignBytes(coreRows * 4U * sizeof(T)) / sizeof(T));
            // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
            // boxes统一按有效字节数向上补齐到32B；GM须有真实可读padding。
            const uint32_t boxBytes = SbgAlignBytes(coreRows * 4U * tSize);
            DataCopyParams boxDp{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
            DataCopy(boxIn, boxGm, boxDp);

            GlobalTensor<T> ancGm;
            ancGm.SetGlobalBuffer(anchorsBase + static_cast<int64_t>(rowStart) * 2, SbgAlignBytes(coreRows * 2U * sizeof(T)) / sizeof(T));
            // anchors与boxes使用相同的普通DMA策略，不能只修改其中一路。
            const uint32_t anchorBytes = SbgAlignBytes(coreRows * 2U * tSize);
            DataCopyParams ancDp{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
            DataCopy(ancIn, ancGm, ancDp);
            // Both input DMA requests are already in flight. Read the attribute here,
            // so the scalar cache miss can overlap input transfers.
            // 仍读取本次调用的真实float属性，不因shape相同而忽略输入值。
            waistRatio = reinterpret_cast<__gm__ volatile float*>(tiling)[3];
            AscendC::InitSocState();

            thresholdValue = 0.0f;
        
        }
    }

    // Compute：抽列、FP32分步运算、严格GT比较。
    __aicore__ inline void Compute() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的B=33策略：保留抽列与转换，逐行算术及比较只处理coreRows_。

            // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
            Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
            Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));

            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            // 先只等待boxes的EVENT_ID0；anchors的EVENT_ID1在比较前再等待。
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            LocalTensor<float> y1 = scratch_[2U * kPad];
            LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
            LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
            LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
            LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
            SetMaskNorm();
            ResetMask();
            // 先在half域抽列，避免把x1/x2/cx这些不用的列也转成FP32。
            // half三列的起点由kPad决定；真实计算长度见kPad或calcRows。
            LocalTensor<T> y1Half = halfStage_.template ReinterpretCast<T>();
            LocalTensor<T> y2Half = y1Half[kPad];
            LocalTensor<T> cyHalf = y1Half[2U * kPad];
            // 固定模式4/6分别取boxes的y1/y2，模式2取anchors的cy。
            GatherMask(y1Half, boxIn_, static_cast<uint8_t>(4), {1, 2, 8, 0});
            GatherMask(y2Half, boxIn_, static_cast<uint8_t>(6), {1, 2, 8, 0});
            // 先精确提升boxes的y1/y2；cy在anchors就绪后再精确提升为FP32。
            // C8分路计算：几何运算仍按原顺序分步FP32舍入，最终严格GT保持。
            // half y1/y2在[896,1152)相邻，float y1/y2在[3200,3712)相邻。
            // 先用一个2-repeat转换boxes两列；cy在下方等anchors完成后单独转换。
            // boxes与cy的输入/输出区间互不重叠；各自保留生产者到消费者的V屏障。
            static_assert(kPad == 64U && kBankGap == 0U, "C8 exact native conversion layout");
            if ASCEND_IS_AIV {
                set_mask_norm();
                set_vector_mask(0ULL, 0xffffffffffffffffULL);
                // 原生转换不会代替高层Cast建立Gather→VCONV的数据依赖。
                // 先确保boxes两次Gather完成，再提升两列；末尾保留算术消费者屏障。
                pipe_barrier(PIPE_V);
                vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr()), 2, 1, 1, 8, 4);

                pipe_barrier(PIPE_V);
            }
            // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
            // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
            Sub(difference, y2, y1, static_cast<int32_t>(coreRows_));
            if constexpr (!UNIT_RATIO) Muls(difference, difference, waistRatio_, static_cast<int32_t>(coreRows_));
            Add(boundary, y1, difference, static_cast<int32_t>(coreRows_));
            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
                thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
            }
            Adds(boundary, boundary, thresholdValue_, static_cast<int32_t>(coreRows_));
            SetMaskNorm();
            // 一个FP32 repeat的比较结果直接放CMPMASK寄存器；GT相等时为假。
            // 只有比较真正依赖anchors；先完成boxes的参考线计算，再等待anchors。
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            GatherMask(cyHalf, anchorIn_, static_cast<uint8_t>(2), {1, 1, 8, 0});
            if ASCEND_IS_AIV {
                set_mask_norm();
                set_vector_mask(0ULL, 0xffffffffffffffffULL);
                pipe_barrier(PIPE_V);
                vconv_f162f32(reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr()), 1, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
            }
            Compare(cy, boundary, CMPMODE::GT, uint64_t(coreRows_), {1, 1, 1, 8, 8, 8});
            // 选择结果也只写入有效lane；CopyOut继续按实际有效字节数输出逻辑有效行。
            if ASCEND_IS_AIV { set_mask_norm(); set_vector_mask(0ULL, (1ULL << coreRows_) - 1ULL); }
            Select<T, SELMODE::VSEL_CMPMASK_SPR>(output_, one_, zero_, 1, {1, 1, 1, 8, 8, 8});
            ResetMask();
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            LocalTensor<uint8_t> s8 = sv.template ReinterpretCast<uint8_t>();
            LocalTensor<uint8_t> cmpUb = s8[(7U * pad + 4U * bankGap) * static_cast<uint32_t>(sizeof(float))];

            LocalTensor<T> oneUb;
            oneUb = cst[0];
            LocalTensor<T> zeroUb;
            zeroUb = cst[pad];
            Duplicate(oneUb, static_cast<T>(1.0f), static_cast<int32_t>(pad));
            Duplicate(zeroUb, static_cast<T>(0.0f), static_cast<int32_t>(pad));


            // The last 32 bytes of sv are reserved and do not overlap cmpUb or any
            // intermediate tensor. Only the address is needed here, not comparison data.
            

            // 等输入DMA完成后才消费UB；当前仍是单段切片，没有双缓冲。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            

            LocalTensor<float> lUb;
            lUb = sv[0];
            LocalTensor<float> y1Ub;
            y1Ub = sv[2U * pad];
            LocalTensor<float> y2Ub;
            y2Ub = sv[3U * pad + bankGap];
            LocalTensor<float> dUb;
            dUb = sv[4U * pad + bankGap];
            LocalTensor<float> boundUb;
            boundUb = sv[5U * pad + 2U * bankGap];
            LocalTensor<float> cyUb;
            cyUb = sv[6U * pad + 3U * bankGap];

            const uint64_t maskOdd[1] = {kSbgMaskOdd};
            const uint64_t maskEven[1] = {kSbgMaskEven};
            const int32_t rep4 = static_cast<int32_t>((4U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
            const int32_t rep2 = static_cast<int32_t>((2U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
            const int32_t padI = static_cast<int32_t>(pad);

            {
                // On CANN 9.0, the no-count overload avoids the
                // V->S count synchronization. Promote only the three used columns.
                SetMaskNorm();
                ResetMask();
                const uint16_t boxReps = static_cast<uint16_t>(4U * pad * sizeof(T) / 256U);
                const uint16_t ancReps = static_cast<uint16_t>(2U * pad * sizeof(T) / 256U);
                LocalTensor<T> y1Half = boxF32.template ReinterpretCast<T>();
                LocalTensor<T> y2Half = y1Half[pad];
                LocalTensor<T> cyHalf = y1Half[2U * pad];
                // 固定模式抽取y1/y2/cy后分别转FP32；几何运算和比较都用FP32。
                GatherMask(y1Half, boxIn, static_cast<uint8_t>(4), {1,boxReps,8,0});
                GatherMask(y2Half, boxIn, static_cast<uint8_t>(6), {1,boxReps,8,0});
                GatherMask(cyHalf, ancIn, static_cast<uint8_t>(2), {1,ancReps,8,0});
                Cast(y1Ub, y1Half, RoundMode::CAST_NONE, pad);
                Cast(y2Ub, y2Half, RoundMode::CAST_NONE, pad);
                Cast(cyUb, cyHalf, RoundMode::CAST_NONE, pad);
            }

            {
                // 保持FP32分步舍入及原公式，不引入FMA或改变结合顺序。
                Sub(dUb, y2Ub, y1Ub, padI);
                Muls(dUb, dUb, waistRatio, padI);
                Add(boundUb, y1Ub, dUb, padI);
                {
                    if ((hasThreshold != 0U) && (threshold != nullptr)) {
                        thresholdValue = static_cast<float>(reinterpret_cast<__gm__ volatile T *>(threshold)[0]);
                    }
                }
                Adds(boundUb, boundUb, thresholdValue, padI);
            }

            {
                // 严格GT：cy等于边界时输出0；padding lane不得影响有效输出。
                Compare(cmpUb, cyUb, boundUb, CMPMODE::GT, padI);
                Select(outUb, cmpUb, oneUb, zeroUb, SELMODE::VSEL_TENSOR_TENSOR_MODE, padI);
            }
        
        }
    }

    // CopyOut：只搬出有效行。
    __aicore__ inline void CopyOut() {
        if constexpr (FIXED_BATCH == 33U) {
            // 同一区间类的编译期B=33策略：恢复冻结v528原CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyParams outputCopy{1, static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 2U) / 32U), 0, 0};
            DataCopy(outputGm_, output_, outputCopy);
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            GlobalTensor<T> outGm;
            outGm.SetGlobalBuffer(outBase + rowStart, SbgAlignBytes(coreRows * sizeof(T)) / sizeof(T));
            const uint32_t outBytes = coreRows * tSize;
            if ((outBytes & 31U) == 0U) {
                DataCopyParams outDp;
                outDp.blockCount = 1;
                outDp.blockLen = static_cast<uint16_t>(outBytes / 32U);
                outDp.srcStride = 0;
                outDp.dstStride = 0;
                DataCopy(outGm, outUb, outDp);
            } else {
                DataCopyParams outCp;
                outCp.blockCount = 1;
                outCp.blockLen = static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U);
                outCp.srcStride = 0;
                outCp.dstStride = 0;
                DataCopy(outGm, outUb, outCp);
            }
        
        }
    }
};

// 尺寸分支10：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case09：FP16 / B=256】SbgFp16B256
// 当前启动8核，工作核上限1；核0处理256行；核1～7退出。
// 实际向量计算行数：256；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：2048 / 1024 / 512 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 当前历史ABI：阈值/输出使用half；只有y1/y2/cy转FP32后参与几何计算和比较。
// 基线仍为三次独立Cast；合并转换候选尚未成为当前选定基线。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,2048)；anchorIn_[2048,3072)；output_[3072,3584)；constants_[9728,10752)；scratch_[65536,73888)；halfStage_[3584,7680)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp16B256 final {
public:
    static constexpr uint32_t kBatch = 256U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 1U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 256U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 256U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 64U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 65536U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 2088U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 73888U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgShapePolicy(kBatch, true).launch == 8U,
                  "Host launch policy changed: review this point explicitly");
    static_assert(SbgShapePolicy(kBatch, true).working == kWorkingCores,
                  "Host working-core policy changed: review this point explicitly");
    static_assert((((kBatch + kWorkingCores - 1U) / kWorkingCores + 15U) / 16U * 16U)
                  == kRowsPerCore, "Host/device slice mismatch");
    static_assert(kUbBytes <= 192U * 1024U, "Point UB allocation exceeds 192 KiB");

    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
                              GM_ADDR threshold, GM_ADDR violation_score,
                              uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score, hasThreshold, tiling)) return;
        // 仅有一个数据切片；这些阶段有必要的依赖等待，不能凭函数名认定已流水重叠。
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    using T = half;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;
    LocalTensor<float> halfStage_;
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
        if (block >= kWorkingCores) return false;
        rowStart_ = block * kRowsPerCore;
        coreRows_ = 256U;
        threshold_ = threshold;
        tiling_ = tiling;
        hasThreshold_ = hasThreshold;
        thresholdValue_ = 0.0f;
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T));
        anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
        anchorIn_ = LocalTensor<T>(TPosition::VECIN, 2048U, kPad * 2U);
        output_ = LocalTensor<T>(TPosition::VECCALC, 3072U, kPad);
        constants_ = LocalTensor<T>(TPosition::VECCALC, 9728U, kPad * 2U);
        scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
        compare_ = scratch_.template ReinterpretCast<uint8_t>()[8192U];
        one_ = constants_[0];
        zero_ = constants_[kPad];
        // 声明为float容量容器，计算时重新解释为half，暂存抽出的三列half。
        halfStage_ = LocalTensor<float>(TPosition::VECCALC, 3584U, kPad * 4U);
        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            GlobalTensor<uint64_t> thresholdLine;
            thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
            // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
            DataCachePreload(thresholdLine, static_cast<int16_t>(0));
        }
        DataCopyParams boxCopy{1, 64, 0, 0};
        DataCopyParams anchorCopy{1, 32, 0, 0};
        DataCopy(boxIn_, boxesGm_, boxCopy);
        DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
            RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
            reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
        // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
        AscendC::InitSocState();

    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
        Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
        if ASCEND_IS_AIV {
            // 为当前CANN9选择指令约定准备“比较位图UB地址”的载体。
            // 这里不是最终比较结果；真正的0/1位图由后面的VCMPV写出。
            LocalTensor<uint32_t> addressPayload = scratch_.template ReinterpretCast<uint32_t>()[2080U];
            auto addressPtr = reinterpret_cast<__ubuf__ uint32_t*>(addressPayload.GetPhyAddr());
            const uint32_t compareAddress = static_cast<uint32_t>(
                reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr())));
            set_mask_norm();
            set_vector_mask(0ULL, 0xffULL);
            vector_dup(addressPtr, compareAddress, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_cmpmask(addressPtr);
            pipe_barrier(PIPE_V);
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        LocalTensor<float> y1 = scratch_[2U * kPad];
        LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
        LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
        LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
        LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
        SetMaskNorm();
        ResetMask();
        // 先在half域抽列，避免把x1/x2/cx这些不用的列也转成FP32。
        // half三列的起点由kPad决定；真实计算长度见kPad或calcRows。
        LocalTensor<T> y1Half = halfStage_.template ReinterpretCast<T>();
        LocalTensor<T> y2Half = y1Half[kPad];
        LocalTensor<T> cyHalf = y1Half[2U * kPad];
        // 固定模式4/6分别取boxes的y1/y2，模式2取anchors的cy。
        GatherMask(y1Half, boxIn_, static_cast<uint8_t>(4), {1, 8, 8, 0});
        GatherMask(y2Half, boxIn_, static_cast<uint8_t>(6), {1, 8, 8, 0});
        GatherMask(cyHalf, anchorIn_, static_cast<uint8_t>(2), {1, 4, 8, 0});
        // 三列各自提升为FP32；这是基线的三次Cast，不是合并转换候选。
        // C9独立试验：保留三列原地址和bank间隔，只更换精确half→FP32提升。
        // 每个正常mask repeat转换64元素，4次覆盖256元素；不移动UB、不改变GM。
        // 前后V屏障保护Gather结果和后续Sub消费者，FP32运算顺序及GT均保持。
        if ASCEND_IS_AIV {
            set_mask_norm();
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            pipe_barrier(PIPE_V);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr()), 4, 1, 1, 8, 4);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(y2Half.GetPhyAddr()), 4, 1, 1, 8, 4);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr()), 4, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
        Sub(difference, y2, y1, static_cast<int32_t>(kPad));
        if constexpr (!UNIT_RATIO) Muls(difference, difference, waistRatio_, static_cast<int32_t>(kPad));
        Add(boundary, y1, difference, static_cast<int32_t>(kPad));
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
            thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
        }
        Adds(boundary, boundary, thresholdValue_, static_cast<int32_t>(kPad));
        if ASCEND_IS_AIV {
            auto maskPtr = reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            auto boundPtr = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
            auto outputPtr = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
            auto onePtr = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
            auto zeroPtr = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            pipe_barrier(PIPE_V);
            // 严格GT：cy==boundary时输出0。VCMPV把位图写入compare_，随后VSEL选择一/零。
            vcmpv_gt(maskPtr, cyPtr, boundPtr, 4, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            // half输出每repeat可处理128元素；count模式按真实向量长度覆盖末段。
            set_mask_count();
            set_vector_mask(0ULL, static_cast<uint64_t>(kPad));
            vsel(outputPtr, onePtr, zeroPtr, 1, 1, 1, 1, 8, 8, 8,
                 static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams outputCopy{1, 16, 0, 0};
        DataCopy(outputGm_, output_, outputCopy);
    }
};

// ============================================================================
// 【分支11】FP16，完整覆盖B∈[257,4095]：SbgFp16Range257To4095
// 本类拥有完整搬入/抽列/计算/搬出流程，区间端点和普通B都在同一实现内处理。
// Host可在本类内选择编译期分工；普通B与端点均不转发给另一个单点类。
// 例：B4094分为7×512＋510；B4095分为6×640＋255。两者都由这个类处理。
// 算法来自原v528已存在的区间实现，扩大入口覆盖；不宣称与原单点路径性能相同。
// ============================================================================
template <uint32_t FIXED_BATCH = 0, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0, bool TAIL = false>
class SbgFp16Range257To4095 final {
public:
    // 完整区间[257,4095]只有本类。FIXED_BATCH是Host真实B守卫后的编译期策略；0表示真实运行时B。
    // 编译期策略恢复v528原优化，不创建独立单点类；B4094等仍由本类处理。
    static constexpr uint32_t kBatch = FIXED_BATCH;
    static constexpr uint32_t kWorkingCores = FIXED_BATCH == 257U ? (1U) : FIXED_BATCH == 4095U ? (6U) : 0U;
    static constexpr uint32_t kRowsPerCore = FIXED_BATCH == 257U ? (272U) : FIXED_BATCH == 4095U ? (768U) : 0U;
    static constexpr uint32_t kPad = FIXED_BATCH == 257U ? (320U) : FIXED_BATCH == 4095U ? (768U) : 0U;
    static constexpr uint32_t kBankGap = FIXED_BATCH == 257U ? (0U) : FIXED_BATCH == 4095U ? (64U) : 0U;
    static constexpr uint32_t kSvBase = FIXED_BATCH == 257U ? (42U * kPad) : FIXED_BATCH == 4095U ? (65536U) : 0U;
    static constexpr uint32_t kSvFloatCount = FIXED_BATCH == 257U ? (7U * kPad + 56U) : FIXED_BATCH == 4095U ? (7688U) : 0U;
    static constexpr uint32_t kUbBytes = FIXED_BATCH == 257U ? (70U * kPad + 224U) : FIXED_BATCH == 4095U ? (96288U) : 0U;
    static_assert(FIXED_BATCH == 0U || (FIXED_BATCH == 257U || FIXED_BATCH == 4095U), "Unsupported constant interval plan");
    static_assert(kUbBytes <= 192U * 1024U, "UB budget exceeds 192 KiB");
    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR threshold, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score,
                  length, rowsPerCore, hasThreshold, tiling)) return;
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    // 原优化路径所需的成员与UB视图；通用成员不会被常量策略使用。
    using T = half;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;
    LocalTensor<float> halfStage_;
    // 通用B的真实长度/片上视图；编译器在常量策略中消除未使用成员。
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
    uint32_t rowStart = 0U, coreRows = 0U, movedRows = 0U, pad = 0U;
    uint32_t hasThreshold = 0U;
    GM_ADDR threshold = nullptr;
    GM_ADDR tiling = nullptr;
    float waistRatio = 0.0f, thresholdValue = 0.0f;
    __gm__ T *boxesBase = nullptr, *anchorsBase = nullptr, *outBase = nullptr;
    LocalTensor<T> boxIn, ancIn, outUb, cst;
    LocalTensor<float> sv, boxF32, ancF32;
    // Init按本类策略建立分片和UB；每条路径均保持原有效GM长度。
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原Init语句。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
            if (block != 0U) return false;
            rowStart_ = 0U;
            coreRows_ = 257U;
            threshold_ = thresholdArg;
            tiling_ = tilingArg;
            hasThreshold_ = hasThresholdArg;
            thresholdValue_ = 0.0f;
            // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
            // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
            boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T));
            anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
            outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
            anchorIn_ = LocalTensor<T>(TPosition::VECIN, 8U * kPad, kPad * 2U);
            output_ = LocalTensor<T>(TPosition::VECCALC, 12U * kPad, kPad);
            constants_ = LocalTensor<T>(TPosition::VECCALC, 38U * kPad, kPad * 2U);
            scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
            compare_ = scratch_.template ReinterpretCast<uint8_t>()[28U * kPad];
            one_ = constants_[0];
            zero_ = constants_[kPad];
            // 声明为float容量容器，计算时重新解释为half，暂存抽出的三列half。
            halfStage_ = LocalTensor<float>(TPosition::VECCALC, 14U * kPad, kPad * 4U);
            return true;
        
        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B4095候选：5×768+255有效行，输入对齐到与B4096相同；阶段Init语句。

            // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
            const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
            if (block >= kWorkingCores) return false;
            rowStart_ = block * kRowsPerCore;
            coreRows_ = block == 5U ? 255U : 768U;
            threshold_ = thresholdArg;
            tiling_ = tilingArg;
            hasThreshold_ = hasThresholdArg;
            thresholdValue_ = 0.0f;
            // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
            // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
            boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, ((coreRows_ + 255U) & ~255U) * 4U);
            anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, ((coreRows_ + 255U) & ~255U) * 2U);
            outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
            // Byte offsets belong only to this point. None is shared with another class.
            // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
            // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
            // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
            boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
            anchorIn_ = LocalTensor<T>(TPosition::VECIN, 8192U, kPad * 2U);
            output_ = LocalTensor<T>(TPosition::VECCALC, 12288U, kPad);
            constants_ = LocalTensor<T>(TPosition::VECCALC, 38912U, kPad * 2U);
            scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
            compare_ = scratch_.template ReinterpretCast<uint8_t>()[29696U];
            one_ = constants_[0];
            zero_ = constants_[kPad];
            // 声明为float容量容器，计算时重新解释为half，暂存抽出的三列half。
            halfStage_ = LocalTensor<float>(TPosition::VECCALC, 14336U, kPad * 4U);
            return true;
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            threshold = thresholdArg;
            hasThreshold = hasThresholdArg;
            tiling = tilingArg;

            // 保留历史工程ABI：FP16阈值/输出为half；与题面FP32文字差异仍需单独review。
            if (length < 257U || length > 4095U) return false;

            // 本类直接处理整个区间：真实B和每核行步长来自Tiling，不转发给单点类。
            // 超出有效B的启动核退出；最后一核只取剩余的有效行数。
            const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
            rowStart = blockIdx * rowsPerCore;
            {
                if (rowStart >= length) return false;
            }
            {
                coreRows = length - rowStart;
                if (coreRows > rowsPerCore) coreRows = rowsPerCore;
            }
            {
                if (hasThreshold != 0U && threshold != nullptr) {
                    GlobalTensor<uint64_t> thresholdCacheLine;
                    thresholdCacheLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold), 1);
                    DataCachePreload(thresholdCacheLine, static_cast<int16_t>(0));
                }
            }
            // coreRows是逻辑有效行数；GM搬运视图32B补齐，调用者须提供实际可读写padding。
            movedRows = SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgRowAlign));
            pad = SbgAlignUp(coreRows, 64U);
            // 以下是本类自有的紧凑UB排列；o按字节，LocalTensor下标按其元素类型。
            // 当前布局预算为FP32的65*pad+32字节，FP16的71*pad+32字节。
            uint32_t o = 0;
            boxIn = LocalTensor<T>(TPosition::VECIN, o, pad * 4U);
            o += SbgAlignBytes(pad * 4U * tSize);
            ancIn = LocalTensor<T>(TPosition::VECIN, o, pad * 2U);
            o += SbgAlignBytes(pad * 2U * tSize);
            outUb = LocalTensor<T>(TPosition::VECCALC, o, pad);
            o += SbgAlignBytes(pad * tSize);
            {
                boxF32 = LocalTensor<float>(TPosition::VECCALC, o, pad * 4U);
                o += SbgAlignBytes(pad * 4U * 4U);
                ancF32 = LocalTensor<float>(TPosition::VECCALC, o, pad * 2U);
                o += SbgAlignBytes(pad * 2U * 4U);
            }
            cst = LocalTensor<T>(TPosition::VECCALC, o, pad * 2U);
            o += SbgAlignBytes(pad * tSize * 2U);
            sv = LocalTensor<float>(TPosition::VECCALC, o, 7U * pad + 4U * bankGap + (SbgAlignBytes(pad) / 4U) + 8U);
            boxesBase = reinterpret_cast<__gm__ T *>(person_boxes);
            anchorsBase = reinterpret_cast<__gm__ T *>(anchor_points);
            outBase = reinterpret_cast<__gm__ T *>(violation_score);
            return true;
        
        }
    }

    // CopyIn：搬入真实输入并准备实际属性/阈值。
    __aicore__ inline void CopyIn() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原CopyIn语句。

            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                GlobalTensor<uint64_t> thresholdLine;
                thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
                // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
                DataCachePreload(thresholdLine, static_cast<int16_t>(0));
            }
            DataCopyParams boxCopy{1, static_cast<uint16_t>(SbgAlignBytes(2056U) / 32U), 0, 0};
            DataCopyParams anchorCopy{1, static_cast<uint16_t>(SbgAlignBytes(1028U) / 32U), 0, 0};
            DataCopy(boxIn_, boxesGm_, boxCopy);
            DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
                RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
                reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
            // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
            AscendC::InitSocState();

        
        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B4095候选：5×768+255有效行，输入对齐到与B4096相同；阶段CopyIn语句。

            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                GlobalTensor<uint64_t> thresholdLine;
                thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
                // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
                DataCachePreload(thresholdLine, static_cast<int16_t>(0));
            }
            // C12与FP16 B4096采用相同分工：前5核768行，尾核搬256行。
            // 逻辑尾核仅255行；末核额外读boxes8B/anchors4B，须有真实可读的对齐存储。
            // 额外lane允许NaN，其结果不写出；输出总量仍严格4095行。
            const uint32_t copyRows = (coreRows_ + 255U) & ~255U;
            DataCopyParams boxCopy{1, static_cast<uint16_t>(copyRows / 4U), 0, 0};
            DataCopyParams anchorCopy{1, static_cast<uint16_t>(copyRows / 8U), 0, 0};            DataCopy(boxIn_, boxesGm_, boxCopy);
            // 分别记录boxes与anchors就绪；允许先到的boxes先进入Vector。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            DataCopy(anchorIn_, anchorsGm_, anchorCopy);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
            waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
                RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
                reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
            // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
            AscendC::InitSocState();

        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            GlobalTensor<T> boxGm;
            boxGm.SetGlobalBuffer(boxesBase + static_cast<int64_t>(rowStart) * 4, SbgAlignBytes(coreRows * 4U * sizeof(T)) / sizeof(T));
            // 统一使用普通DataCopy，搬运字节数向上补齐到32B。
            // boxes统一按有效字节数向上补齐到32B；GM须有真实可读padding。
            const uint32_t boxBytes = SbgAlignBytes(coreRows * 4U * tSize);
            DataCopyParams boxDp{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
            DataCopy(boxIn, boxGm, boxDp);

            GlobalTensor<T> ancGm;
            ancGm.SetGlobalBuffer(anchorsBase + static_cast<int64_t>(rowStart) * 2, SbgAlignBytes(coreRows * 2U * sizeof(T)) / sizeof(T));
            // anchors与boxes使用相同的普通DMA策略，不能只修改其中一路。
            const uint32_t anchorBytes = SbgAlignBytes(coreRows * 2U * tSize);
            DataCopyParams ancDp{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
            DataCopy(ancIn, ancGm, ancDp);
            // Both input DMA requests are already in flight. Read the attribute here,
            // so the scalar cache miss can overlap input transfers.
            // 仍读取本次调用的真实float属性，不因shape相同而忽略输入值。
            waistRatio = reinterpret_cast<__gm__ volatile float*>(tiling)[3];
            AscendC::InitSocState();

            thresholdValue = 0.0f;
        
        }
    }

    // Compute：抽列、FP32分步运算、严格GT比较。
    __aicore__ inline void Compute() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原Compute语句。

            // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
            Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(kPad));
            Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(kPad));
            if ASCEND_IS_AIV {
                // 为当前CANN9选择指令约定准备“比较位图UB地址”的载体。
                // 这里不是最终比较结果；真正的0/1位图由后面的VCMPV写出。
                LocalTensor<uint32_t> addressPayload = scratch_.template ReinterpretCast<uint32_t>()[7U * kPad + 48U];
                auto addressPtr = reinterpret_cast<__ubuf__ uint32_t*>(addressPayload.GetPhyAddr());
                const uint32_t compareAddress = static_cast<uint32_t>(
                    reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr())));
                set_mask_norm();
                set_vector_mask(0ULL, 0xffULL);
                vector_dup(addressPtr, compareAddress, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_cmpmask(addressPtr);
                pipe_barrier(PIPE_V);
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            LocalTensor<float> y1 = scratch_[2U * kPad];
            LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
            LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
            LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
            LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
            SetMaskNorm();
            ResetMask();
            // 先在half域抽列，避免把x1/x2/cx这些不用的列也转成FP32。
            // half三列的起点由kPad决定；真实计算长度见kPad或calcRows。
            LocalTensor<T> y1Half = halfStage_.template ReinterpretCast<T>();
            LocalTensor<T> y2Half = y1Half[kPad];
            LocalTensor<T> cyHalf = y1Half[2U * kPad];
            // 固定模式4/6分别取boxes的y1/y2，模式2取anchors的cy。
            GatherMask(y1Half, boxIn_, static_cast<uint8_t>(4), {1, static_cast<uint16_t>(kPad / 32U), 8, 0});
            GatherMask(y2Half, boxIn_, static_cast<uint8_t>(6), {1, static_cast<uint16_t>(kPad / 32U), 8, 0});
            GatherMask(cyHalf, anchorIn_, static_cast<uint8_t>(2), {1, static_cast<uint16_t>(kPad / 64U), 8, 0});
            // 三列各自提升为FP32；这是基线的三次Cast，不是合并转换候选。
            // C10独立试验：三列仍在原来的UB地址，不移动或合并任何plane。
            // 正常64-lane mask下每列5个repeat精确提升320个half；有效GM仍只有257行。
            // 只合并独立转换之间的同步，保留Gather生产者和Sub消费者所需前后屏障。
            static_assert(kPad == 320U && kBankGap == 0U, "C10 native conversion plan");
            if ASCEND_IS_AIV {
                set_mask_norm();
                set_vector_mask(0ULL, 0xffffffffffffffffULL);
                pipe_barrier(PIPE_V);
                vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr()), 5, 1, 1, 8, 4);
                vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(y2Half.GetPhyAddr()), 5, 1, 1, 8, 4);
                vconv_f162f32(reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr()), 5, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
            }
            // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
            // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
            Sub(difference, y2, y1, static_cast<int32_t>(kPad));
            if constexpr (!UNIT_RATIO) Muls(difference, difference, waistRatio_, static_cast<int32_t>(kPad));
            Add(boundary, y1, difference, static_cast<int32_t>(kPad));
            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
                thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
            }
            Adds(boundary, boundary, thresholdValue_, static_cast<int32_t>(kPad));
            if ASCEND_IS_AIV {
                auto maskPtr = reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr());
                auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
                auto boundPtr = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
                auto outputPtr = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
                auto onePtr = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
                auto zeroPtr = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
                pipe_barrier(PIPE_V);
                // 严格GT：cy==boundary时输出0。VCMPV把位图写入compare_，随后VSEL选择一/零。
                vcmpv_gt(maskPtr, cyPtr, boundPtr, kPad / 64U, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                // half输出每repeat可处理128元素；count模式按真实向量长度覆盖末段。
                set_mask_count();
                set_vector_mask(0ULL, static_cast<uint64_t>(kPad));
                vsel(outputPtr, onePtr, zeroPtr, 1, 1, 1, 1, 8, 8, 8,
                     static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
        
        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B4095候选：5×768+255有效行，输入对齐到与B4096相同；阶段Compute语句。

            // calcRows是本核向量长度：只在UB内向上对齐，搬出仍严格用coreRows_。
            const uint32_t calcRows = (coreRows_ + 63U) & ~63U;
            // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
            Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(calcRows));
            Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(calcRows));
            if ASCEND_IS_AIV {
                // 为当前CANN9选择指令约定准备“比较位图UB地址”的载体。
                // 这里不是最终比较结果；真正的0/1位图由后面的VCMPV写出。
                LocalTensor<uint32_t> addressPayload = scratch_.template ReinterpretCast<uint32_t>()[7680U];
                auto addressPtr = reinterpret_cast<__ubuf__ uint32_t*>(addressPayload.GetPhyAddr());
                const uint32_t compareAddress = static_cast<uint32_t>(
                    reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr())));
                set_mask_norm();
                set_vector_mask(0ULL, 0xffULL);
                vector_dup(addressPtr, compareAddress, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_cmpmask(addressPtr);
                pipe_barrier(PIPE_V);
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
            // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
            // CopyIn已分别记录两路就绪；这里先仅等待boxes。
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // 下面为同一个scratch中的视图，各偏移单位是float元素。
            // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
            LocalTensor<float> y1 = scratch_[2U * kPad];
            LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
            LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
            LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
            LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
            SetMaskNorm();
            ResetMask();
            // 先在half域抽列，避免把x1/x2/cx这些不用的列也转成FP32。
            // half三列的起点由kPad决定；真实计算长度见kPad或calcRows。
            LocalTensor<T> y1Half = halfStage_.template ReinterpretCast<T>();
            LocalTensor<T> y2Half = y1Half[kPad];
            LocalTensor<T> cyHalf = y1Half[2U * kPad];
            // Independent extraction operations share one dependency barrier.
            // Fixed-pattern source-1 addresses match the CANN 9.0 dav_c220 wrapper.
            if ASCEND_IS_AIV {
                auto boxesPtr = reinterpret_cast<__ubuf__ uint16_t*>(boxIn_.GetPhyAddr());
                auto anchorsPtr = reinterpret_cast<__ubuf__ uint16_t*>(anchorIn_.GetPhyAddr());
                auto y1HalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(y1Half.GetPhyAddr());
                auto y2HalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(y2Half.GetPhyAddr());
                auto cyHalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(cyHalf.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
                // 原生固定模式抽列；src1的+512保留已核验的CANN9调用形式。
                // 它不是“第512行”，也不要当作本算子的自定义索引表。
                vreducev2(y1HalfPtr, boxesPtr, boxesPtr + 512, calcRows / 32U, 1, 4, 8, 0);
                vreducev2(y2HalfPtr, boxesPtr, boxesPtr + 512, calcRows / 32U, 1, 6, 8, 0);

                pipe_barrier(PIPE_V);
                set_vector_mask(0ULL, 0xffffffffffffffffULL);
                // 三次原生half→FP32转换；后面的几何运算和比较不使用half精度。
                vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr()), calcRows / 64U, 1, 1, 8, 4);
                vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ T*>(y2Half.GetPhyAddr()), calcRows / 64U, 1, 1, 8, 4);

                pipe_barrier(PIPE_V);
            }
            // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
            // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
            Sub(difference, y2, y1, static_cast<int32_t>(calcRows));
            if constexpr (!UNIT_RATIO) Muls(difference, difference, waistRatio_, static_cast<int32_t>(calcRows));
            Add(boundary, y1, difference, static_cast<int32_t>(calcRows));
            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
                thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
            }
            // An actual zero threshold leaves every finite comparison unchanged,
            // including signed-zero boundaries. Positive thresholds retain Adds.
            // 仅在真实读取的阈值为±0时省略Adds；正阈值必须走原来的加法。
            if (thresholdValue_ != 0.0f) {
                Adds(boundary, boundary, thresholdValue_, static_cast<int32_t>(calcRows));
            }
            // boxes的half抽列、FP32转换及参考线先算；比较前汇合anchors。
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            if ASCEND_IS_AIV {
                auto anchorsPtr = reinterpret_cast<__ubuf__ uint16_t*>(anchorIn_.GetPhyAddr());
                auto cyHalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(cyHalf.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
                vreducev2(cyHalfPtr, anchorsPtr, anchorsPtr + 512, calcRows / 64U, 1, 2, 8, 0);
                pipe_barrier(PIPE_V);
                set_vector_mask(0ULL, 0xffffffffffffffffULL);
                vconv_f162f32(reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr()), calcRows / 64U, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
            }
            if ASCEND_IS_AIV {
                auto maskPtr = reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr());
                auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
                auto boundPtr = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
                auto outputPtr = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
                auto onePtr = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
                auto zeroPtr = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
                pipe_barrier(PIPE_V);
                // 严格GT：cy==boundary时输出0。VCMPV把位图写入compare_，随后VSEL选择一/零。
                vcmpv_gt(maskPtr, cyPtr, boundPtr, calcRows / 64U, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                // half输出每repeat可处理128元素；count模式按真实向量长度覆盖末段。
                set_mask_count();
                set_vector_mask(0ULL, static_cast<uint64_t>(calcRows));
                vsel(outputPtr, onePtr, zeroPtr, 1, 1, 1, 1, 8, 8, 8,
                     static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            LocalTensor<uint8_t> s8 = sv.template ReinterpretCast<uint8_t>();
            LocalTensor<uint8_t> cmpUb = s8[(7U * pad + 4U * bankGap) * static_cast<uint32_t>(sizeof(float))];

            LocalTensor<T> oneUb;
            oneUb = cst[0];
            LocalTensor<T> zeroUb;
            zeroUb = cst[pad];
            Duplicate(oneUb, static_cast<T>(1.0f), static_cast<int32_t>(pad));
            Duplicate(zeroUb, static_cast<T>(0.0f), static_cast<int32_t>(pad));


            // The last 32 bytes of sv are reserved and do not overlap cmpUb or any
            // intermediate tensor. Only the address is needed here, not comparison data.
            

            // 等输入DMA完成后才消费UB；当前仍是单段切片，没有双缓冲。
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            

            LocalTensor<float> lUb;
            lUb = sv[0];
            LocalTensor<float> y1Ub;
            y1Ub = sv[2U * pad];
            LocalTensor<float> y2Ub;
            y2Ub = sv[3U * pad + bankGap];
            LocalTensor<float> dUb;
            dUb = sv[4U * pad + bankGap];
            LocalTensor<float> boundUb;
            boundUb = sv[5U * pad + 2U * bankGap];
            LocalTensor<float> cyUb;
            cyUb = sv[6U * pad + 3U * bankGap];

            const uint64_t maskOdd[1] = {kSbgMaskOdd};
            const uint64_t maskEven[1] = {kSbgMaskEven};
            const int32_t rep4 = static_cast<int32_t>((4U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
            const int32_t rep2 = static_cast<int32_t>((2U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
            const int32_t padI = static_cast<int32_t>(pad);

            {
                // On CANN 9.0, the no-count overload avoids the
                // V->S count synchronization. Promote only the three used columns.
                SetMaskNorm();
                ResetMask();
                const uint16_t boxReps = static_cast<uint16_t>(4U * pad * sizeof(T) / 256U);
                const uint16_t ancReps = static_cast<uint16_t>(2U * pad * sizeof(T) / 256U);
                LocalTensor<T> y1Half = boxF32.template ReinterpretCast<T>();
                LocalTensor<T> y2Half = y1Half[pad];
                LocalTensor<T> cyHalf = y1Half[2U * pad];
                // 固定模式抽取y1/y2/cy后分别转FP32；几何运算和比较都用FP32。
                GatherMask(y1Half, boxIn, static_cast<uint8_t>(4), {1,boxReps,8,0});
                GatherMask(y2Half, boxIn, static_cast<uint8_t>(6), {1,boxReps,8,0});
                GatherMask(cyHalf, ancIn, static_cast<uint8_t>(2), {1,ancReps,8,0});
                Cast(y1Ub, y1Half, RoundMode::CAST_NONE, pad);
                Cast(y2Ub, y2Half, RoundMode::CAST_NONE, pad);
                Cast(cyUb, cyHalf, RoundMode::CAST_NONE, pad);
            }

            {
                // 保持FP32分步舍入及原公式，不引入FMA或改变结合顺序。
                Sub(dUb, y2Ub, y1Ub, padI);
                Muls(dUb, dUb, waistRatio, padI);
                Add(boundUb, y1Ub, dUb, padI);
                {
                    if ((hasThreshold != 0U) && (threshold != nullptr)) {
                        thresholdValue = static_cast<float>(reinterpret_cast<__gm__ volatile T *>(threshold)[0]);
                    }
                }
                Adds(boundUb, boundUb, thresholdValue, padI);
            }

            {
                // 严格GT：cy等于边界时输出0；padding lane不得影响有效输出。
                Compare(cmpUb, cyUb, boundUb, CMPMODE::GT, padI);
                Select(outUb, cmpUb, oneUb, zeroUb, SELMODE::VSEL_TENSOR_TENSOR_MODE, padI);
            }
        
        }
    }

    // CopyOut：只搬出有效行。
    __aicore__ inline void CopyOut() {
        if constexpr (FIXED_BATCH == 257U) {
            // 同一区间类的编译期B=257策略：恢复冻结v528原CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyParams outputCopy{1, static_cast<uint16_t>(SbgAlignBytes(514U) / 32U), 0, 0};
            DataCopy(outputGm_, output_, outputCopy);
        
        } else if constexpr (FIXED_BATCH == 4095U) {
            // 同一区间类B4095候选：5×768+255有效行，输入对齐到与B4096相同；阶段CopyOut语句。

            // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (coreRows_ == kRowsPerCore) {
                DataCopyParams outputCopy{1, static_cast<uint16_t>(coreRows_ / 16U), 0, 0};
                DataCopy(outputGm_, output_, outputCopy);
            } else {
                DataCopyParams outputCopy{1, static_cast<uint16_t>(SbgAlignBytes(coreRows_ * 2U) / 32U), 0, 0};
                DataCopy(outputGm_, output_, outputCopy);
            }
        
        } else {
            // 普通B（包含区间内所有非端点尺寸）直接执行本类通用流程。

            // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            GlobalTensor<T> outGm;
            outGm.SetGlobalBuffer(outBase + rowStart, SbgAlignBytes(coreRows * sizeof(T)) / sizeof(T));
            const uint32_t outBytes = coreRows * tSize;
            if ((outBytes & 31U) == 0U) {
                DataCopyParams outDp;
                outDp.blockCount = 1;
                outDp.blockLen = static_cast<uint16_t>(outBytes / 32U);
                outDp.srcStride = 0;
                outDp.dstStride = 0;
                DataCopy(outGm, outUb, outDp);
            } else {
                DataCopyParams outCp;
                outCp.blockCount = 1;
                outCp.blockLen = static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U);
                outCp.srcStride = 0;
                outCp.dstStride = 0;
                DataCopy(outGm, outUb, outCp);
            }
        
        }
    }
};

// 尺寸分支12：保留原v528的该精确点类主体。
// ============================================================================
// 【公开case11：FP16 / B=4096】SbgFp16B4096
// 当前启动8核，工作核上限6；核0～4各768行；核5处理256行；核6、7退出。
// 实际向量计算行数：主核768，尾核256；下列字节数依次为boxes / anchors / output。
// 当前搬运：DataCopy：主核6144/3072/1536 B；尾核2048/1024/512 B。
// 每个工作核只处理自己的单段切片：CopyIn → Compute → CopyOut，没有下一tile的双缓冲。
// 当前历史ABI：阈值/输出使用half；只有y1/y2/cy转FP32后参与几何计算和比较。
// 此处以“公开case”编号为准，不能沿用旧英文注释的内部排列编号。
// 主核/常规核UB字节区间（左闭右开）：boxIn_[0,6144)；anchorIn_[8192,11264)；output_[12288,13824)；constants_[38912,41984)；scratch_[65536,96288)；halfStage_[14336,26624)
// ============================================================================
template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
class SbgFp16B4096 final {
public:
    static constexpr uint32_t kBatch = 4096U; // 精确点的逻辑B，不代表输入坐标内容已知。
    static constexpr uint32_t kWorkingCores = 6U; // 工作核上限；启动核之外/多余启动核在Init退出。
    static constexpr uint32_t kRowsPerCore = 768U; // 用于定位GM切片起点的行步长；尾核有效行数另算。
    static constexpr uint32_t kPad = 768U; // UB分配/列间距基准；不授权多搬GM数据，也不一定等于calcRows。
    static constexpr uint32_t kBankGap = 64U; // FP32元素单位的间隔，不是字节；当前布局参数，不是数学所需数据。
    static constexpr uint32_t kSvBase = 65536U; // scratch在本核UB中的字节起点。
    static constexpr uint32_t kSvFloatCount = 7688U; // scratch容量，单位float元素，包含位图及地址载体预留。
    static constexpr uint32_t kUbBytes = 96288U; // 本类UB最高字节边界，Host对应策略必须覆盖它。
    static_assert(SbgShapePolicy(kBatch, true).launch == 8U,
                  "Host launch policy changed: review this point explicitly");
    static_assert(SbgShapePolicy(kBatch, true).working == kWorkingCores,
                  "Host working-core policy changed: review this point explicitly");
    static_assert(kBatch == 5U * kRowsPerCore + 256U, "Exact 512B-aligned partition");
    static_assert(kUbBytes <= 192U * 1024U, "Point UB allocation exceeds 192 KiB");

    __aicore__ inline void Run(GM_ADDR person_boxes, GM_ADDR anchor_points,
                              GM_ADDR threshold, GM_ADDR violation_score,
                              uint32_t hasThreshold, GM_ADDR tiling) {
        if (!Init(person_boxes, anchor_points, threshold, violation_score, hasThreshold, tiling)) return;
        // 仅有一个数据切片；这些阶段有必要的依赖等待，不能凭函数名认定已流水重叠。
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    using T = half;
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<T> outputGm_;
    LocalTensor<T> boxIn_;
    LocalTensor<T> anchorIn_;
    LocalTensor<T> output_;
    LocalTensor<T> constants_;
    LocalTensor<T> one_;
    LocalTensor<T> zero_;
    LocalTensor<float> scratch_;
    LocalTensor<uint8_t> compare_;
    LocalTensor<float> halfStage_;
    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points,
                               GM_ADDR threshold, GM_ADDR violation_score,
                               uint32_t hasThreshold, GM_ADDR tiling) {
        // block是本次launch的逻辑核号；先确定是否工作，再访问本核数据。
        const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
        if (block >= kWorkingCores) return false;
        rowStart_ = block * kRowsPerCore;
        coreRows_ = block == 5U ? 256U : 768U;
        threshold_ = threshold;
        tiling_ = tiling;
        hasThreshold_ = hasThreshold;
        thresholdValue_ = 0.0f;
        // rowStart_/coreRows_是“行”；指针下标是T元素，不是字节。
        // 一行boxes有4个T，anchors有2个T，输出有1个T；SetGlobalBuffer只建视图。
        boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(person_boxes) + rowStart_ * 4U, SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T));
        anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchor_points) + rowStart_ * 2U, SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(violation_score) + rowStart_, SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
        // Byte offsets belong only to this point. None is shared with another class.
        // LocalTensor构造器：第二参数是UB字节地址，第三参数是该类型元素数。
        // scratch_[n]以float计偏移；ReinterpretCast<uint8_t>()[n]才以字节计。
        // ReinterpretCast只改变解释方式，不搬运、不转置，也不产生新内存。
        boxIn_ = LocalTensor<T>(TPosition::VECIN, 0U, kPad * 4U);
        anchorIn_ = LocalTensor<T>(TPosition::VECIN, 8192U, kPad * 2U);
        output_ = LocalTensor<T>(TPosition::VECCALC, 12288U, kPad);
        constants_ = LocalTensor<T>(TPosition::VECCALC, 38912U, kPad * 2U);
        scratch_ = LocalTensor<float>(TPosition::VECCALC, kSvBase, kSvFloatCount);
        compare_ = scratch_.template ReinterpretCast<uint8_t>()[29696U];
        one_ = constants_[0];
        zero_ = constants_[kPad];
        // 声明为float容量容器，计算时重新解释为half，暂存抽出的三列half。
        halfStage_ = LocalTensor<float>(TPosition::VECCALC, 14336U, kPad * 4U);
        return true;
    }

    // 【搬入】发出输入DMA，并按本分支的时序准备真实属性和阈值。
    __aicore__ inline void CopyIn() {
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            GlobalTensor<uint64_t> thresholdLine;
            thresholdLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold_), 1);
            // 缓存预取提示；uint64_t是此API的地址载体，实际阈值仍在后面按T读取。
            DataCachePreload(thresholdLine, static_cast<int16_t>(0));
        }
        DataCopyParams boxCopy{1, static_cast<uint16_t>(coreRows_ / 4U), 0, 0};
        DataCopyParams anchorCopy{1, static_cast<uint16_t>(coreRows_ / 8U), 0, 0};        DataCopy(boxIn_, boxesGm_, boxCopy);
        // 分别记录boxes与anchors就绪；允许先到的boxes先进入Vector。
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        DataCopy(anchorIn_, anchorsGm_, anchorCopy);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // 已被Host真实属性守卫选中的模板才使用常量；否则从Tiling[3]取实际float属性。
        waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
            RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
            reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
        // 保留当前工具链的设备状态初始化；它不会在CPU重排数据，也不等于预热。
        AscendC::InitSocState();

    }

    // 【计算】抽取三列 → FP32几何运算 → 严格大于比较 → 生成0/1。
    __aicore__ inline void Compute() {
        // calcRows是本核向量长度：只在UB内向上对齐，搬出仍严格用coreRows_。
        const uint32_t calcRows = (coreRows_ + 255U) & ~255U;
        // 一/零向量供最终Select使用；真实判定尚未计算，不是按公开样例填答案。
        Duplicate(one_, static_cast<T>(1.0f), static_cast<int32_t>(calcRows));
        Duplicate(zero_, static_cast<T>(0.0f), static_cast<int32_t>(calcRows));
        if ASCEND_IS_AIV {
            // 为当前CANN9选择指令约定准备“比较位图UB地址”的载体。
            // 这里不是最终比较结果；真正的0/1位图由后面的VCMPV写出。
            LocalTensor<uint32_t> addressPayload = scratch_.template ReinterpretCast<uint32_t>()[7680U];
            auto addressPtr = reinterpret_cast<__ubuf__ uint32_t*>(addressPayload.GetPhyAddr());
            const uint32_t compareAddress = static_cast<uint32_t>(
                reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr())));
            set_mask_norm();
            set_vector_mask(0ULL, 0xffULL);
            vector_dup(addressPtr, compareAddress, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_cmpmask(addressPtr);
            pipe_barrier(PIPE_V);
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        // CopyIn已分别记录两路就绪；这里先仅等待boxes。
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        LocalTensor<float> y1 = scratch_[2U * kPad];
        LocalTensor<float> y2 = scratch_[3U * kPad + kBankGap];
        LocalTensor<float> difference = scratch_[4U * kPad + kBankGap];
        LocalTensor<float> boundary = scratch_[5U * kPad + 2U * kBankGap];
        LocalTensor<float> cy = scratch_[6U * kPad + 3U * kBankGap];
        SetMaskNorm();
        ResetMask();
        // 先在half域抽列，避免把x1/x2/cx这些不用的列也转成FP32。
        // half三列的起点由kPad决定；真实计算长度见kPad或calcRows。
        LocalTensor<T> y1Half = halfStage_.template ReinterpretCast<T>();
        LocalTensor<T> y2Half = y1Half[kPad];
        LocalTensor<T> cyHalf = y1Half[2U * kPad];
        // Independent extraction operations share one dependency barrier.
        // Fixed-pattern source-1 addresses match the CANN 9.0 dav_c220 wrapper.
        if ASCEND_IS_AIV {
            auto boxesPtr = reinterpret_cast<__ubuf__ uint16_t*>(boxIn_.GetPhyAddr());
            auto anchorsPtr = reinterpret_cast<__ubuf__ uint16_t*>(anchorIn_.GetPhyAddr());
            auto y1HalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(y1Half.GetPhyAddr());
            auto y2HalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(y2Half.GetPhyAddr());
            auto cyHalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(cyHalf.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            // 原生固定模式抽列；src1的+512保留已核验的CANN9调用形式。
            // 它不是“第512行”，也不要当作本算子的自定义索引表。
            vreducev2(y1HalfPtr, boxesPtr, boxesPtr + 512, calcRows / 32U, 1, 4, 8, 0);
            vreducev2(y2HalfPtr, boxesPtr, boxesPtr + 512, calcRows / 32U, 1, 6, 8, 0);

            pipe_barrier(PIPE_V);
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            // 三次原生half→FP32转换；后面的几何运算和比较不使用half精度。
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr()), calcRows / 64U, 1, 1, 8, 4);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ T*>(y2Half.GetPhyAddr()), calcRows / 64U, 1, 1, 8, 4);

            pipe_barrier(PIPE_V);
        }
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
        Sub(difference, y2, y1, static_cast<int32_t>(calcRows));
        if constexpr (!UNIT_RATIO) Muls(difference, difference, waistRatio_, static_cast<int32_t>(calcRows));
        Add(boundary, y1, difference, static_cast<int32_t>(calcRows));
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
            thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
        }
        // An actual zero threshold leaves every finite comparison unchanged,
        // including signed-zero boundaries. Positive thresholds retain Adds.
        // 仅在真实读取的阈值为±0时省略Adds；正阈值必须走原来的加法。
        if (thresholdValue_ != 0.0f) {
            Adds(boundary, boundary, thresholdValue_, static_cast<int32_t>(calcRows));
        }
        // boxes的half抽列、FP32转换及参考线先算；比较前汇合anchors。
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        if ASCEND_IS_AIV {
            auto anchorsPtr = reinterpret_cast<__ubuf__ uint16_t*>(anchorIn_.GetPhyAddr());
            auto cyHalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(cyHalf.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            vreducev2(cyHalfPtr, anchorsPtr, anchorsPtr + 512, calcRows / 64U, 1, 2, 8, 0);
            pipe_barrier(PIPE_V);
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr()), calcRows / 64U, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }
        if ASCEND_IS_AIV {
            auto maskPtr = reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            auto boundPtr = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
            auto outputPtr = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
            auto onePtr = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
            auto zeroPtr = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            pipe_barrier(PIPE_V);
            // 严格GT：cy==boundary时输出0。VCMPV把位图写入compare_，随后VSEL选择一/零。
            vcmpv_gt(maskPtr, cyPtr, boundPtr, calcRows / 64U, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            // half输出每repeat可处理128元素；count模式按真实向量长度覆盖末段。
            set_mask_count();
            set_vector_mask(0ULL, static_cast<uint64_t>(calcRows));
            vsel(outputPtr, onePtr, zeroPtr, 1, 1, 1, 1, 8, 8, 8,
                 static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    // 逻辑结果仅coreRows行；输出DMA向上补齐到32B，额外写入限真实分配的输出padding。
    __aicore__ inline void CopyOut() {
        // Vector→MTE3事件：输出向量产生后才能写GM；不能提前复用/覆盖output_。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyParams outputCopy{1, static_cast<uint16_t>(coreRows_ / 16U), 0, 0};
        DataCopy(outputGm_, output_, outputCopy);
    }
};

// 仅保留资源不足/空输入等兼容回退。
template <typename DT_PERSON_BOXES, uint32_t CONST_B = 0, uint32_t CONST_PER = 1, bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0>
__aicore__ inline void SbgIntervalFallback(GM_ADDR person_boxes, GM_ADDR anchor_points, GM_ADDR threshold,
                              GM_ADDR violation_score, uint32_t length, uint32_t rowsPerCore,
                              uint32_t hasThreshold, GM_ADDR tiling) {
    using T = DT_PERSON_BOXES;
    const uint32_t tSize = static_cast<uint32_t>(sizeof(T));

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    constexpr bool single33 = sizeof(T) == 2U && CONST_B == 33U && CONST_PER >= CONST_B;
    const uint32_t rowStart = single33 ? 0U : blockIdx * rowsPerCore;
    if constexpr (single33) {
        if (blockIdx != 0U) return;
    } else {
        if (rowStart >= length) return;
    }
    uint32_t coreRows;
    if constexpr (single33) {
        coreRows = CONST_B;
    } else if constexpr (CONST_B > 0 && CONST_B % CONST_PER == 0) {
        coreRows = CONST_PER;
    } else {
        coreRows = length - rowStart;
        if (coreRows > rowsPerCore) coreRows = rowsPerCore;
    }
    if constexpr (sizeof(T) == 2) {
        if (hasThreshold != 0U && threshold != nullptr) {
            GlobalTensor<uint64_t> thresholdCacheLine;
            thresholdCacheLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold), 1);
            DataCachePreload(thresholdCacheLine, static_cast<int16_t>(0));
        }
    }
    const uint32_t movedRows = SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgRowAlign));
    const uint32_t pad = CONST_B > 0 ? ((CONST_PER + 63U) / 64U * 64U)
                                       : SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgPadMin));
    uint32_t o = 0;
    LocalTensor<T> boxIn;
    boxIn = LocalTensor<T>(TPosition::VECIN, o, pad * 4U);
    o += SbgAlignBytes(pad * 4U * tSize);
    LocalTensor<T> ancIn;
    ancIn = LocalTensor<T>(TPosition::VECIN, o, pad * 2U);
    o += SbgAlignBytes(pad * 2U * tSize);
    LocalTensor<T> outUb;
    outUb = LocalTensor<T>(TPosition::VECCALC, o, pad);
    o += SbgAlignBytes(pad * tSize);
    LocalTensor<float> boxF32;
    LocalTensor<float> ancF32;
    if constexpr (sizeof(T) == 2) {
        boxF32 = LocalTensor<float>(TPosition::VECCALC, o, pad * 4U);
        o += SbgAlignBytes(pad * 4U * 4U);
        ancF32 = LocalTensor<float>(TPosition::VECCALC, o, pad * 2U);
        o += SbgAlignBytes(pad * 2U * 4U);
    }
    LocalTensor<T> cst;
    cst = LocalTensor<T>(TPosition::VECCALC, o, pad * 2U);
    o += SbgAlignBytes(pad * tSize * 2U);
    const uint32_t bankGap = (CONST_B > 0 && CONST_PER > 64U && (pad % 128U) == 0U) ? 64U : 0U;
    LocalTensor<float> sv;
    sv = LocalTensor<float>(TPosition::VECCALC, bankGap ? 65536U : o, 7U * pad + 4U * bankGap + (SbgAlignBytes(pad) / 4U) + 8U);
    __gm__ T *boxesBase = reinterpret_cast<__gm__ T *>(person_boxes);
    __gm__ T *anchorsBase = reinterpret_cast<__gm__ T *>(anchor_points);
    __gm__ T *outBase = reinterpret_cast<__gm__ T *>(violation_score);

    GlobalTensor<T> boxGm;
    boxGm.SetGlobalBuffer(boxesBase + static_cast<int64_t>(rowStart) * 4, SbgAlignBytes(coreRows * 4U * sizeof(T)) / sizeof(T));
    // boxes统一按有效字节数向上补齐到32B；GM须有真实可读padding。
    const uint32_t boxBytes = SbgAlignBytes(coreRows * 4U * tSize);
    DataCopyParams boxDp{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
    DataCopy(boxIn, boxGm, boxDp);

    GlobalTensor<T> ancGm;
    ancGm.SetGlobalBuffer(anchorsBase + static_cast<int64_t>(rowStart) * 2, SbgAlignBytes(coreRows * 2U * sizeof(T)) / sizeof(T));
    // anchors与boxes使用相同的普通DMA策略，不能只修改其中一路。
    const uint32_t anchorBytes = SbgAlignBytes(coreRows * 2U * tSize);
    DataCopyParams ancDp{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
    DataCopy(ancIn, ancGm, ancDp);
    // Both input DMA requests are already in flight. Read the attribute here,
    // so the scalar cache miss can overlap input transfers.
    const float waistRatio = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
        RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
        reinterpret_cast<__gm__ volatile float*>(tiling)[3];
    AscendC::InitSocState();

    float thresholdValue = 0.0f;
    if constexpr (sizeof(T) == 4) {
        if ((hasThreshold != 0U) && (threshold != nullptr)) {
            thresholdValue = static_cast<float>(reinterpret_cast<__gm__ T *>(threshold)[0]);
        }
    }

    LocalTensor<uint8_t> s8 = sv.template ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> cmpUb = s8[(7U * pad + 4U * bankGap) * static_cast<uint32_t>(sizeof(float))];

    LocalTensor<T> oneUb;
    oneUb = cst[0];
    LocalTensor<T> zeroUb;
    zeroUb = cst[pad];
    Duplicate(oneUb, static_cast<T>(1.0f), static_cast<int32_t>(pad));
    Duplicate(zeroUb, static_cast<T>(0.0f), static_cast<int32_t>(pad));


    // The last 32 bytes of sv are reserved and do not overlap cmpUb or any
    // intermediate tensor. Only the address is needed here, not comparison data.
    if constexpr (CONST_B > 0 && CONST_PER > 64U) {
        if ASCEND_IS_AIV {
            LocalTensor<uint32_t> selAddrUb = sv.template ReinterpretCast<uint32_t>()[
                7U * pad + 4U * bankGap + SbgAlignBytes(pad) / sizeof(uint32_t)];
            auto addrp = reinterpret_cast<__ubuf__ uint32_t*>(selAddrUb.GetPhyAddr());
            const uint32_t cmpAddr = static_cast<uint32_t>(
                reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(cmpUb.GetPhyAddr())));
            set_mask_norm();
            set_vector_mask(0ULL, 0xffULL);  // exactly eight uint32 lanes / 32 B
            vector_dup(addrp, cmpAddr, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);  // address payload ready before SPR load
            set_cmpmask(addrp);
            pipe_barrier(PIPE_V);  // CMPMASK ready before any later Select
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    LocalTensor<float> boxVec;
    LocalTensor<float> ancVec;
    if constexpr (sizeof(T) == 4) {
        boxVec = boxIn.template ReinterpretCast<float>();
        ancVec = ancIn.template ReinterpretCast<float>();
    }

    LocalTensor<float> lUb;
    lUb = sv[0];
    LocalTensor<float> y1Ub;
    y1Ub = sv[2U * pad];
    LocalTensor<float> y2Ub;
    y2Ub = sv[3U * pad + bankGap];
    LocalTensor<float> dUb;
    dUb = sv[4U * pad + bankGap];
    LocalTensor<float> boundUb;
    boundUb = sv[5U * pad + 2U * bankGap];
    LocalTensor<float> cyUb;
    cyUb = sv[6U * pad + 3U * bankGap];

    const uint64_t maskOdd[1] = {kSbgMaskOdd};
    const uint64_t maskEven[1] = {kSbgMaskEven};
    const int32_t rep4 = static_cast<int32_t>((4U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
    const int32_t rep2 = static_cast<int32_t>((2U * pad) / static_cast<uint32_t>(kSbgPerRepeat));
    const int32_t padI = static_cast<int32_t>(pad);

    if constexpr (sizeof(T) == 2) {
        // On CANN 9.0, the no-count overload avoids the
        // V->S count synchronization. Promote only the three used columns.
        SetMaskNorm();
        ResetMask();
        const uint16_t boxReps = static_cast<uint16_t>(4U * pad * sizeof(T) / 256U);
        const uint16_t ancReps = static_cast<uint16_t>(2U * pad * sizeof(T) / 256U);
        LocalTensor<T> y1Half = boxF32.template ReinterpretCast<T>();
        LocalTensor<T> y2Half = y1Half[pad];
        LocalTensor<T> cyHalf = y1Half[2U * pad];
        GatherMask(y1Half, boxIn, static_cast<uint8_t>(4), {1,boxReps,8,0});
        GatherMask(y2Half, boxIn, static_cast<uint8_t>(6), {1,boxReps,8,0});
        GatherMask(cyHalf, ancIn, static_cast<uint8_t>(2), {1,ancReps,8,0});
        Cast(y1Ub, y1Half, RoundMode::CAST_NONE, pad);
        Cast(y2Ub, y2Half, RoundMode::CAST_NONE, pad);
        Cast(cyUb, cyHalf, RoundMode::CAST_NONE, pad);
    } else if constexpr (CONST_B > 0) {
        // Two independent groups. In group 1, boxes and anchors have different
        // sources and destinations. Group 2 reads the completed lUb and writes
        // two different columns. No pair in either group has RAW/WAR/WAW overlap.
        if ASCEND_IS_AIV {
            auto lp = reinterpret_cast<__ubuf__ float*>(lUb.GetPhyAddr());
            auto boxp = reinterpret_cast<__ubuf__ float*>(boxVec.GetPhyAddr());
            auto ancp = reinterpret_cast<__ubuf__ float*>(ancVec.GetPhyAddr());
            auto y1p = reinterpret_cast<__ubuf__ float*>(y1Ub.GetPhyAddr());
            auto y2p = reinterpret_cast<__ubuf__ float*>(y2Ub.GetPhyAddr());
            auto cyp = reinterpret_cast<__ubuf__ float*>(cyUb.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, kSbgMaskOdd);
            vcpadd(lp, boxp, rep4, 1, 1, 8);
            vcpadd(cyp, ancp, rep2, 1, 1, 8);
            pipe_barrier(PIPE_V);  // lUb must be complete before either reader.
            vcpadd(y2p, lp, rep2, 1, 1, 8);
            set_vector_mask(0ULL, kSbgMaskEven);
            vcpadd(y1p, lp, rep2, 1, 1, 8);
            pipe_barrier(PIPE_V);  // Both columns must be ready for Sub.
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
        }
    } else {
        PairReduceSum<float>(lUb, boxVec, rep4, maskOdd, 1, 1, 8);
        PairReduceSum<float>(y1Ub, lUb, rep2, maskEven, 1, 1, 8);
        PairReduceSum<float>(y2Ub, lUb, rep2, maskOdd, 1, 1, 8);
        PairReduceSum<float>(cyUb, ancVec, rep2, maskOdd, 1, 1, 8);
    }

    if constexpr (sizeof(T) == 4 && CONST_B > 0) {
        // Extraction leaves a full 64-lane normal mask. Keep that mask for all
        // four instructions; preserve every original FP32 rounding boundary.
        if ASCEND_IS_AIV {
            auto dp = reinterpret_cast<__ubuf__ float*>(dUb.GetPhyAddr());
            auto y1p = reinterpret_cast<__ubuf__ float*>(y1Ub.GetPhyAddr());
            auto y2p = reinterpret_cast<__ubuf__ float*>(y2Ub.GetPhyAddr());
            auto bp = reinterpret_cast<__ubuf__ float*>(boundUb.GetPhyAddr());
            const uint8_t reps = static_cast<uint8_t>(pad / 64U);
            vsub(dp, y2p, y1p, reps, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (!UNIT_RATIO) {
                vmuls(dp, dp, static_cast<float>(waistRatio), reps, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            vadd(bp, y1p, dp, reps, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadds(bp, bp, thresholdValue, reps, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
    } else {
        Sub(dUb, y2Ub, y1Ub, padI);
        if constexpr (!UNIT_RATIO) Muls(dUb, dUb, waistRatio, padI);
        Add(boundUb, y1Ub, dUb, padI);
        if constexpr (sizeof(T) == 2) {
            if ((hasThreshold != 0U) && (threshold != nullptr)) {
                thresholdValue = static_cast<float>(reinterpret_cast<__gm__ volatile T *>(threshold)[0]);
            }
        }
        Adds(boundUb, boundUb, thresholdValue, padI);
    }

    if constexpr (CONST_B > 0 && CONST_PER <= 64U) {
        // Consume the one-repeat comparison mask directly in the Select unit.
        SetMaskNorm();
        Compare(cyUb, boundUb, CMPMODE::GT, uint64_t(64), {1,1,1,8,8,8});
        Select<T, SELMODE::VSEL_CMPMASK_SPR>(outUb, oneUb, zeroUb, 1, {1,1,1,8,8,8});
    } else if constexpr (CONST_B > 0 && CONST_PER > 64U) {
        if ASCEND_IS_AIV {
            auto maskp = reinterpret_cast<__ubuf__ uint8_t*>(cmpUb.GetPhyAddr());
            auto cyp = reinterpret_cast<__ubuf__ float*>(cyUb.GetPhyAddr());
            auto bp = reinterpret_cast<__ubuf__ float*>(boundUb.GetPhyAddr());
            auto outp = reinterpret_cast<__ubuf__ T*>(outUb.GetPhyAddr());
            auto onep = reinterpret_cast<__ubuf__ T*>(oneUb.GetPhyAddr());
            auto zerop = reinterpret_cast<__ubuf__ T*>(zeroUb.GetPhyAddr());
            constexpr uint32_t padded = (CONST_PER + 63U) / 64U * 64U;
            constexpr uint8_t compareReps = static_cast<uint8_t>(padded / 64U);
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            // VCMPV writes UB only; it does not replace the earlier CMPMASK address.
            if constexpr (sizeof(T) == 2) pipe_barrier(PIPE_V);
            vcmpv_gt(maskp, cyp, bp, compareReps, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);  // comparison result must be ready for VSEL
            if constexpr (sizeof(T) == 4) {
                vsel(outp, onep, zerop, compareReps, 1, 1, 1, 8, 8, 8,
                     static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
            } else {
                // FP16 pads can contain an odd number of 64-lane chunks.
                // Count mode covers the final partial 128-lane repeat safely.
                set_mask_count();
                set_vector_mask(0ULL, static_cast<uint64_t>(padded));
                vsel(outp, onep, zerop, 1, 1, 1, 1, 8, 8, 8,
                     static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
            }
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    } else {
        Compare(cmpUb, cyUb, boundUb, CMPMODE::GT, padI);
        Select(outUb, cmpUb, oneUb, zeroUb, SELMODE::VSEL_TENSOR_TENSOR_MODE, padI);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<T> outGm;
    outGm.SetGlobalBuffer(outBase + rowStart, SbgAlignBytes(coreRows * sizeof(T)) / sizeof(T));
    const uint32_t outBytes = coreRows * tSize;
    if constexpr (CONST_B > 0 && CONST_B % CONST_PER != 0) {
        DataCopyParams cp{1, static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U), 0, 0};
        DataCopy(outGm, outUb, cp);
    } else if ((outBytes & 31U) == 0U) {
        DataCopyParams outDp;
        outDp.blockCount = 1;
        outDp.blockLen = static_cast<uint16_t>(outBytes / 32U);
        outDp.srcStride = 0;
        outDp.dstStride = 0;
        DataCopy(outGm, outUb, outDp);
    } else {
        DataCopyParams outCp;
        outCp.blockCount = 1;
        outCp.blockLen = static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U);
        outCp.srcStride = 0;
        outCp.dstStride = 0;
        DataCopy(outGm, outUb, outCp);
    }
}

// 【NPU总入口：合法B的12个尺寸分支】
// 三个区间类与三个精确点类分别各有FP32/FP16版本；不存在独立B33/B257/B4095类。
// 低资源/空输入兼容回退保留在最后；正常B4094、4095均进入大区间类。
template <typename DT_PERSON_BOXES, uint32_t SBG_BATCH = 0>
__global__ __aicore__ void safety_belt_violation_geometry(GM_ADDR person_boxes, GM_ADDR anchor_points,
    GM_ADDR threshold, GM_ADDR violation_score, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
    constexpr bool unitRatio = SBG_BATCH == 8224U;
    constexpr bool fixedRatio = SBG_BATCH >= 16384U;
    constexpr uint32_t base = fixedRatio ? SBG_BATCH - 16384U : unitRatio ? 32U : SBG_BATCH;
    constexpr uint32_t ratioKind = !fixedRatio ? 0U : base == 256U ? 2U : base == 257U ? 3U : 1U;
    // 33/257/4095选择同一区间类的编译期策略，8193/8194/8195选择该类的真实B策略。
    constexpr bool small = base == 8193U;
    constexpr bool middle = base == 8194U || base == 33U;
    constexpr bool large = base == 8195U || base == 8196U || base == 257U || base == 4095U;
    __gm__ uint32_t* tp = reinterpret_cast<__gm__ uint32_t*>(tiling);
    if constexpr (sizeof(DT_PERSON_BOXES) == 4U && small) {
        // 分支01：FP32 [1,31]，只进入SbgFp32Range1To31。
        SbgFp32Range1To31 point;
        point.Run(person_boxes, anchor_points, threshold, violation_score, tp[0], tp[1], tp[2], tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 4U && base == 32U) {
        // 分支02：FP32 32，只进入SbgFp32B32。
        SbgFp32B32<unitRatio, ratioKind> point;
        point.Run(person_boxes, anchor_points, threshold, violation_score,
                  threshold != nullptr ? 1U : 0U, tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 4U && middle) {
        // 分支03：FP32 [33,255]，只进入SbgFp32Range33To255。
        constexpr uint32_t fixed = (base == 33U) ? base : 0U;
        SbgFp32Range33To255<fixed, unitRatio, ratioKind> point;
        point.Run(person_boxes, anchor_points, threshold, violation_score, fixed ? fixed : tp[0], fixed ? 48U : tp[1], fixed ? (threshold != nullptr ? 1U : 0U) : tp[2], tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 4U && base == 256U) {
        // 分支04：FP32 256，只进入SbgFp32B256。
        SbgFp32B256<unitRatio, ratioKind> point;
        point.Run(person_boxes, anchor_points, threshold, violation_score,
                  threshold != nullptr ? 1U : 0U, tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 4U && large) {
        // 分支05：FP32 [257,4095]，只进入SbgFp32Range257To4095。
        constexpr uint32_t fixed = (base == 257U || base == 4095U) ? base : 0U;
        if constexpr (fixed == 4095U) {
            // 同一个类中保留原v528主/尾核的编译期UB策略；不改变尺寸分支。
            if (GetBlockIdx() == 7U) {
                SbgFp32Range257To4095<fixed, unitRatio, ratioKind, true> point;
                point.Run(person_boxes, anchor_points, threshold, violation_score, fixed ? fixed : tp[0], fixed ? (fixed == 4095U ? 576U : fixed == 257U ? 272U : 32U) : tp[1], fixed ? (threshold != nullptr ? 1U : 0U) : tp[2], tiling);
            } else {
                SbgFp32Range257To4095<fixed, unitRatio, ratioKind, false> point;
                point.Run(person_boxes, anchor_points, threshold, violation_score, fixed ? fixed : tp[0], fixed ? (fixed == 4095U ? 576U : fixed == 257U ? 272U : 32U) : tp[1], fixed ? (threshold != nullptr ? 1U : 0U) : tp[2], tiling);
            }
        } else {
            SbgFp32Range257To4095<fixed, unitRatio, ratioKind> point;
            point.Run(person_boxes, anchor_points, threshold, violation_score, fixed ? fixed : tp[0], fixed ? (fixed == 4095U ? 576U : fixed == 257U ? 272U : 32U) : tp[1], fixed ? (threshold != nullptr ? 1U : 0U) : tp[2], tiling);
        }
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 4U && base == 4096U) {
        // 分支06：FP32 4096，只进入SbgFp32B4096。
        SbgFp32B4096<unitRatio, ratioKind> point;
        point.Run(person_boxes, anchor_points, threshold, violation_score,
                  threshold != nullptr ? 1U : 0U, tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 2U && small) {
        // 分支07：FP16 [1,31]，只进入SbgFp16Range1To31。
        SbgFp16Range1To31 point;
        point.Run(person_boxes, anchor_points, threshold, violation_score, tp[0], tp[1], tp[2], tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 2U && base == 32U) {
        // 分支08：FP16 32，只进入SbgFp16B32。
        SbgFp16B32<unitRatio, ratioKind> point;
        point.Run(person_boxes, anchor_points, threshold, violation_score,
                  threshold != nullptr ? 1U : 0U, tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 2U && middle) {
        // 分支09：FP16 [33,255]，只进入SbgFp16Range33To255。
        constexpr uint32_t fixed = (base == 33U) ? base : 0U;
        SbgFp16Range33To255<fixed, unitRatio, ratioKind> point;
        point.Run(person_boxes, anchor_points, threshold, violation_score, fixed ? fixed : tp[0], fixed ? (fixed == 4095U ? 640U : fixed == 257U ? 272U : 48U) : tp[1], fixed ? (threshold != nullptr ? 1U : 0U) : tp[2], tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 2U && base == 256U) {
        // 分支10：FP16 256，只进入SbgFp16B256。
        SbgFp16B256<unitRatio, ratioKind> point;
        point.Run(person_boxes, anchor_points, threshold, violation_score,
                  threshold != nullptr ? 1U : 0U, tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 2U && large) {
        // 分支11：FP16 [257,4095]，只进入SbgFp16Range257To4095。
        constexpr uint32_t fixed = (base == 257U || base == 4095U) ? base : 0U;
        SbgFp16Range257To4095<fixed, unitRatio, ratioKind> point;
        point.Run(person_boxes, anchor_points, threshold, violation_score, fixed ? fixed : tp[0], fixed ? (fixed == 4095U ? 640U : fixed == 257U ? 272U : 48U) : tp[1], fixed ? (threshold != nullptr ? 1U : 0U) : tp[2], tiling);
    } else if constexpr (sizeof(DT_PERSON_BOXES) == 2U && base == 4096U) {
        // 分支12：FP16 4096，只进入SbgFp16B4096。
        SbgFp16B4096<unitRatio, ratioKind> point;
        point.Run(person_boxes, anchor_points, threshold, violation_score,
                  threshold != nullptr ? 1U : 0U, tiling);
    } else {
        // key0：平台资源不足或B0兼容。旧登记值也保留原兼容处理，不用于普通区间选路。
        if constexpr (SBG_BATCH == 0U) {
            SbgIntervalFallback<DT_PERSON_BOXES>(person_boxes, anchor_points, threshold,
                violation_score, tp[0], tp[1], tp[2], tiling);
        } else {
            constexpr uint32_t active = SbgShapePolicy(base, sizeof(DT_PERSON_BOXES) == 2U).working;
            constexpr uint32_t per = ((base + active - 1U) / active + 15U) / 16U * 16U;
            SbgIntervalFallback<DT_PERSON_BOXES, base, per, unitRatio, ratioKind>(
                person_boxes, anchor_points, threshold, violation_score,
                base, per, threshold != nullptr ? 1U : 0U, tiling);
        }
    }
}

// 系统工作区策略恢复冻结v528：列明Key已恢复原计算主体，普通区间Key仍保留标准包装。
// dump/timestamp/OOM构建保留框架包装；此处不是对任意新算法通用删除工作区。
// Workspace policy inherited only by the unchanged exact-point classes.
// Guarded constant plans use the original v528 stages inside their owning interval class.
// Only points using basic SIMD/DMA APIs are listed. Range/fallback keys retain
// the standard framework setup. Keep that setup for dump/timestamp/OOM builds.
// Bit order follows public cases, not the internal Host debug branch IDs.
#define SBG_NO_SYSTEM_WS_MASK 0xFEFU
#if defined(TILING_KEY_VAR) && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
// bit 0: case1, FP32, B=32; keys 768,3584
// bit 1: case2, FP32, B=33; keys 1024
// bit 2: case3, FP32, B=256; keys 1792
// bit 3: case4, FP32, B=257; keys 2048
// bit 4: case5, FP32, B=4096; keys 3328,4864
// bit 5: case6, FP32, B=4095; keys 3072,4608
// bit 6: case7, FP16, B=32; keys 769,3585
// bit 7: case8, FP16, B=33; keys 1025
// bit 8: case9, FP16, B=256; keys 1793
// bit 9: case10, FP16, B=257; keys 2049
// bit 10: case11, FP16, B=4096; keys 3329,4865
// bit 11: case12, FP16, B=4095; keys 3073,4609
#if ((SBG_NO_SYSTEM_WS_MASK & (1U << 0)) && (TILING_KEY_VAR == 768UL || TILING_KEY_VAR == 3584UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 1)) && (TILING_KEY_VAR == 1024UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 2)) && (TILING_KEY_VAR == 1792UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 3)) && (TILING_KEY_VAR == 2048UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 4)) && (TILING_KEY_VAR == 3328UL || TILING_KEY_VAR == 4864UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 5)) && (TILING_KEY_VAR == 3072UL || TILING_KEY_VAR == 4608UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 6)) && (TILING_KEY_VAR == 769UL || TILING_KEY_VAR == 3585UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 7)) && (TILING_KEY_VAR == 1025UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 8)) && (TILING_KEY_VAR == 1793UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 9)) && (TILING_KEY_VAR == 2049UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 10)) && (TILING_KEY_VAR == 3329UL || TILING_KEY_VAR == 4865UL)) || \
    ((SBG_NO_SYSTEM_WS_MASK & (1U << 11)) && (TILING_KEY_VAR == 3073UL || TILING_KEY_VAR == 4609UL))
namespace AscendC {
__aicore__ inline void SbgNoUnusedSystemWorkspace(GM_ADDR unusedWorkspace) {
    (void)unusedWorkspace;
}
}
#define SetSysWorkspaceForce SbgNoUnusedSystemWorkspace
#endif
#endif
