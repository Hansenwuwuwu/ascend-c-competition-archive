#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct LinalgPinvTilingData {
 uint32_t m,n,p,q,pp,qp,batch,cached;
 float atol,rtol;
 uint32_t group,groups,cols;
};

struct LinalgPinvCubeTilingData {LinalgPinvTilingData base;AscendC::tiling::TCubeTiling cube;};
