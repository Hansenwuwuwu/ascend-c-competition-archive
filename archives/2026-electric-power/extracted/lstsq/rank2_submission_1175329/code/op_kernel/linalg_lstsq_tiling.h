#pragma once
#include <cstdint>
struct LinalgLstsqTilingData {
 uint32_t m,n,k,batches,mp,groupSize,groups,localCols,rounds;
};
