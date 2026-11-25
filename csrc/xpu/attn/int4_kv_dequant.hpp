/***************************************************************************************************
 * Copyright (C) 2025 — FA2 int4-KV greenfield (fork-local).
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Shared INT4 per-token-head dequant helpers for Xe FA2.
 * Must match vLLM Triton write/read semantics in
 * vllm/v1/attention/ops/int4_per_token_head.py:
 *   - two int4 values per uint8 (lo nibble, hi nibble)
 *   - asymmetric quant; 4-bit zero-point steganographed in low 4 bits of
 *     the float32 scale mantissa (cleared with & ~0xF before use as scale)
 *   - dequant: (q_u4 - zp) * scale
 **************************************************************************************************/

#pragma once

#include <cstdint>
#include <cstring>

namespace vllm::xpu::int4_kv {

inline constexpr int kInt4PackingFactor = 2;
inline constexpr int kScalePadU8 = 4;  // sizeof(float) / sizeof(uint8_t)

/// Bytes of packed int4 data per head for logical head_size (e.g. 128 → 64).
inline constexpr int packed_hs(int head_size) { return head_size / kInt4PackingFactor; }

/// Content bytes per (token, kv_head): [K|Ks|V|Vs] — 136 for D=128.
inline constexpr int content_bytes(int head_size) {
  return 2 * (packed_hs(head_size) + kScalePadU8);
}

struct ScaleZp {
  float scale;
  float zp;
};

/// Split packed scale float: zp in low 4 mantissa bits, scale with those bits cleared.
inline ScaleZp unpack_scale_zp(float packed_scale) {
  uint32_t bits;
  static_assert(sizeof(float) == sizeof(uint32_t));
  std::memcpy(&bits, &packed_scale, sizeof(bits));
  const uint32_t zp_u = bits & 0xFu;
  const uint32_t scale_bits = bits & ~0xFu;
  float scale;
  std::memcpy(&scale, &scale_bits, sizeof(scale));
  return ScaleZp{scale, static_cast<float>(zp_u)};
}

inline void unpack_nibbles(uint8_t packed, uint8_t& lo, uint8_t& hi) {
  lo = static_cast<uint8_t>(packed & 0x0F);
  hi = static_cast<uint8_t>((packed >> 4) & 0x0F);
}

/// Dequantize one packed byte → two float values (even, odd lanes).
inline void dequant_byte(uint8_t packed, float scale, float zp, float& even,
                         float& odd) {
  uint8_t lo, hi;
  unpack_nibbles(packed, lo, hi);
  even = (static_cast<float>(lo) - zp) * scale;
  odd = (static_cast<float>(hi) - zp) * scale;
}

/// Expand ``packed_hs`` uint8 bytes into ``head_size`` floats (D=2*packed_hs).
inline void dequant_head(const uint8_t* packed, int packed_hs_len, float packed_scale,
                         float* out /* head_size */) {
  const ScaleZp sz = unpack_scale_zp(packed_scale);
  for (int i = 0; i < packed_hs_len; ++i) {
    float even, odd;
    dequant_byte(packed[i], sz.scale, sz.zp, even, odd);
    out[2 * i] = even;
    out[2 * i + 1] = odd;
  }
}

}  // namespace vllm::xpu::int4_kv
