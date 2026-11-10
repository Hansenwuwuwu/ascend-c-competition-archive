// #pragma once
// #include "ascendc/host_api/tiling/template_argument.h"
// // Keys encode algorithms; tile sizes are runtime data.
// ASCENDC_TPL_ARGS_DECL(SwiGlu,
//     ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16),
//     ASCENDC_TPL_UINT_DECL(FAMILY, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
// );
// ASCENDC_TPL_SEL(
//     ASCENDC_TPL_ARGS_SEL(
//         ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
//         ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
//         ASCENDC_TPL_UINT_SEL(FAMILY, ASCENDC_TPL_UI_LIST, 0),
//     ),
//     ASCENDC_TPL_ARGS_SEL(
//         ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
//         ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
//         ASCENDC_TPL_UINT_SEL(FAMILY, ASCENDC_TPL_UI_LIST, 1, 2, 3, 4),
//     ),
// );
