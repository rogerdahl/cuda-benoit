#pragma once

#include "int_types.h"

// Linear interpolation.

template<typename T, typename U, typename V>
V Lerp(const T& a, const T& b, const U& d) {
  return static_cast<V>(a + d * (b - a));
}

template<typename T>
Color Lerp(const Color& a, const Color& b, T& d) {
  return Color(
      Lerp<u8, T, u32>(a.red_, b.red_, d),
      Lerp<u8, T, u32>(a.green_, b.green_, d), 
      Lerp<u8, T, u32>(a.blue_, b.blue_, d));
}

// Clamp.

template<typename T>
T clamp(const T&v, const T& a, const T& b) {
  if (v < a) {
    return a;
  }
  if (v > b) {
    return b;
  }
  return v;
}

template<typename T>
T clamp(const T&v) {
  return clamp(v, 0.0, 1.0);
}

