#pragma once
#include <cstdint>

namespace elec {
// A nonempty optional output explicitly requests the square-system residual.
constexpr uint32_t LSTSQ_SQUARE_RESIDUAL = uint32_t(1) << 5;
}
