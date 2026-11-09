#pragma once

static constexpr unsigned int CandidateMediumRows = 448;

static constexpr unsigned int CandidateSmallLaunchFloat = 4;
static constexpr unsigned int CandidateSmallLaunchHalf = 2;
#include "ascendc/host_api/tiling/template_argument.h"
ASCENDC_TPL_ARGS_DECL(FireCandidateDecodeFusion,
    ASCENDC_TPL_DATATYPE_DECL(DT_RAW_PRED, C_DT_FLOAT, C_DT_FLOAT16),
    ASCENDC_TPL_UINT_DECL(PATH, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18, 19, 21, 24, 26, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 65, 66),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18, 19, 21, 24, 26, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 65, 66),
    ),
);
