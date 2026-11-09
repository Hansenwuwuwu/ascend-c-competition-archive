// 同一范围类的编译期策略可使用33/257/4095；普通B用8193/8194/8195。登记列表与冻结v528一致。
// 中文审阅：这里只声明可编译/可选择的模板参数值，不直接指定每核数据分工。
// SBG_BATCH的15是该整数参数的位宽，不是列表只能放15项；当前最大编码20480可放入15位。
// 0：兼容/资源不足回退；普通B：精确点或保留模板；8193～8196：四个普通区间。
// 8224：B32且ratio=1；16417/16640/16641/20479/20480对应B33/256/257/4095/4096常量比例模板。
// “已编译模板存在”不等于Host正在选择它；实际常量属性守卫请看TilingFunc。
// 增加新值时，DECL和SEL两份列表必须同步，否则会出现“SEL值不在DECL中”的错误。
// 最终日志中的768/1793/4609等是dtype与列表位置编码的TilingKey，不是这里的B本身。
#pragma once
#include "ascendc/host_api/tiling/template_argument.h"
ASCENDC_TPL_ARGS_DECL(SafetyBeltViolationGeometry,
    ASCENDC_TPL_DATATYPE_DECL(DT_PERSON_BOXES, C_DT_FLOAT16, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(SBG_BATCH, 15, ASCENDC_TPL_UI_LIST, 0,1,16,32,33,64,128,256,257,512,1024,2048,4095,4096,8224,16417,16640,16641,20479,20480,8193,8194,8195,8196),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT16, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(SBG_BATCH, ASCENDC_TPL_UI_LIST, 0,1,16,32,33,64,128,256,257,512,1024,2048,4095,4096,8224,16417,16640,16641,20479,20480,8193,8194,8195,8196),
    ),
);
