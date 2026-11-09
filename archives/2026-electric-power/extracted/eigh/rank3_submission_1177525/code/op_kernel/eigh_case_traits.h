#pragma once
#include <cstdint>
template<uint32_t KEY> struct EighCase;
template<> struct EighCase<1u> {
  static constexpr uint32_t N=16u, B=1u, U=0u, P=1u, Solver=0u;
};
template<> struct EighCase<2u> {
  static constexpr uint32_t N=64u, B=4u, U=0u, P=8u, Solver=2u;
};
template<> struct EighCase<3u> {
  static constexpr uint32_t N=256u, B=4u, U=0u, P=8u, Solver=3u;
};
template<> struct EighCase<4u> {
  static constexpr uint32_t N=15u, B=1u, U=0u, P=1u, Solver=0u;
};
template<> struct EighCase<5u> {
  static constexpr uint32_t N=50u, B=3u, U=0u, P=8u, Solver=2u;
};
template<> struct EighCase<6u> {
  static constexpr uint32_t N=200u, B=3u, U=0u, P=10u, Solver=3u;
};
