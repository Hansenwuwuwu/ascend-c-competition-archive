#pragma once
#include "ascendc/host_api/tiling/template_argument.h"
// Keep the full declaration order for the tiling-key ABI. Instantiate only
// reachable scalar routes: B=1 uses its own tags; B>=2 halves the32pair budget.
// The eight-worker M1 policy also makes old N>4 scalar tags unreachable.
ASCENDC_TPL_ARGS_DECL(TreeLineMinDistance,
  ASCENDC_TPL_DATATYPE_DECL(DT_TREE_POINTS, C_DT_FLOAT, C_DT_FLOAT16),
  ASCENDC_TPL_UINT_DECL(ALGORITHM, 10, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 112, 113, 114, 115, 116, 117, 118, 128, 129, 130, 131, 132, 144, 145, 146, 147, 160, 161, 162, 176, 177, 178, 192, 193, 208, 209, 224, 240, 256, 272, 288, 304, 320, 321, 322, 323, 324, 325, 326, 327, 397, 398, 399, 400, 408, 409, 410, 411, 412, 413, 414, 415, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 528, 529, 530, 531, 532, 533, 534, 535, 536, 544, 545, 546, 547, 548, 549, 550, 560, 561, 562, 563, 564, 576, 577, 578, 579, 592, 593, 594, 608, 609, 610, 624, 625, 640, 641, 656, 672, 688, 704, 720, 736, 768, 769, 770, 771, 772, 773, 774, 784, 785, 786, 787, 788, 789, 790, 848, 849, 850, 851, 852, 853, 854, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876));
// Only selectors are pruned; the declaration order and Host tiling keys stay stable.
// Old0/2/13..36 entries are superseded by bounded private layouts or wide output.
// Single-batch M1 N>=3 uses coalesced entries; its old769..774 and N=2coalesced848 are unreachable.
// Short tables with three or more segments use complete generic scalar loops.
// Single-batch two-segment N>=8 shares one exactly sized launch implementation.
ASCENDC_TPL_SEL(
  ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_TREE_POINTS, C_DT_FLOAT),
    ASCENDC_TPL_UINT_SEL(ALGORITHM, ASCENDC_TPL_UI_LIST, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY)),
  ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_TREE_POINTS, C_DT_FLOAT16),
    ASCENDC_TPL_UINT_SEL(ALGORITHM, ASCENDC_TPL_UI_LIST, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY)),
  ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_TREE_POINTS, C_DT_FLOAT, C_DT_FLOAT16),
    ASCENDC_TPL_UINT_SEL(ALGORITHM, ASCENDC_TPL_UI_LIST, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 64, 80, 96, 112, 128, 144, 160, 176, 320, 321, 322, 323, 397, 398, 399, 400, 408, 409, 410, 411, 412, 413, 414, 415, 512, 528, 544, 560, 576, 592, 608, 768, 784, 785, 786, 787, 788, 789, 790, 849, 850, 851, 852, 853, 854, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876),
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY)));
