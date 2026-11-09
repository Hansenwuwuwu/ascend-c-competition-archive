// 同一范围类的编译期策略可使用33/257/4095；普通B用8193/8194/8195。登记列表与冻结v528一致。
// 中文审阅：Host写入、Kernel读取的二进制协议。字段顺序/类型不能只改一侧。
// 当前7个32位字段共28B：length@0、rowsPerCore@4、hasThreshold@8、waistRatio@12、
// tileRows@16、thresholdInTiling@20、threshold@24（偏移单位为字节）。
// 最后两字段只是历史兼容保留，不是把真实threshold搬到CPU后缓存下来的值。
// launch=启动数，working=工作核上限；实际有效工作核还要看rowStart是否越界。
// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct SafetyBeltViolationGeometryTilingData {
    uint32_t length;            // B：样本行数（person_boxes 第 0 维）
    uint32_t rowsPerCore;       // 名义行分片步长，通常16对齐；精确点的有效尾长在Kernel独立决定
    uint32_t hasThreshold;      // 1：threshold 输入存在；0：按 0.0 处理
    float waistRatio;           // 判定参考线比例，范围 (0, 1]
    uint32_t tileRows;          // 当前Host写入perCore；本版本没有据此实现多tile循环或双缓冲
    uint32_t thresholdInTiling; // 兼容字段，恒 0
    float threshold;            // 兼容字段，恒 0.0f
};

// One policy entry per public dtype/shape. Tensor values remain runtime inputs.
// Shared by Host and Kernel so a changed working-core count cannot drift.
struct SbgShapePlan { uint32_t launch; uint32_t working; };
#ifndef SBG_POLICY_TARGET
#define SBG_POLICY_TARGET
#endif
// Registration source generation uses C++11; constexpr needs one return.
// 与Host分支及Kernel static_assert相互核对；改核数策略需检查三处。
// 三元表达式形式为兼容注册代码生成时使用的C++11 constexpr限制。
SBG_POLICY_TARGET constexpr SbgShapePlan SbgShapePolicyF16(uint32_t b) {
    return b==32U ? SbgShapePlan{8U,1U} :
           b==33U ? SbgShapePlan{8U,1U} :
           b==256U ? SbgShapePlan{8U,1U} :
           b==257U ? SbgShapePlan{8U,1U} :
           b==4095U ? SbgShapePlan{8U,6U} :
           b==4096U ? SbgShapePlan{8U,6U} :
           b<32U ? SbgShapePlan{8U,2U} :
           b<256U ? SbgShapePlan{8U,2U} :
           b<=1024U ? SbgShapePlan{8U,2U} : SbgShapePlan{8U,8U};
}
SBG_POLICY_TARGET constexpr SbgShapePlan SbgShapePolicyF32(uint32_t b) {
    return b==32U ? SbgShapePlan{8U,1U} :
           b==33U ? SbgShapePlan{8U,2U} :
           b==256U ? SbgShapePlan{8U,2U} :
           b==257U ? SbgShapePlan{8U,1U} :
           b==4095U ? SbgShapePlan{8U,8U} :
           b==4096U ? SbgShapePlan{8U,8U} :
           b<32U ? SbgShapePlan{8U,2U} :
           b<256U ? SbgShapePlan{8U,2U} :
           b<=1024U ? SbgShapePlan{8U,2U} : SbgShapePlan{8U,8U};
}
SBG_POLICY_TARGET constexpr SbgShapePlan SbgShapePolicy(uint32_t b, bool fp16) {
    return fp16 ? SbgShapePolicyF16(b) : SbgShapePolicyF32(b);
}
