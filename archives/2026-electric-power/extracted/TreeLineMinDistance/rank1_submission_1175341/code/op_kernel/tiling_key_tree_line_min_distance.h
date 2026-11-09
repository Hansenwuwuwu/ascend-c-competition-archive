// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(TreeLineMinDistance,
    ASCENDC_TPL_DATATYPE_DECL(DT_TREE_POINTS, C_DT_FLOAT, C_DT_FLOAT16),
    ASCENDC_TPL_UINT_DECL(PATH, 5, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_TREE_POINTS, C_DT_FLOAT, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
    ),
);
