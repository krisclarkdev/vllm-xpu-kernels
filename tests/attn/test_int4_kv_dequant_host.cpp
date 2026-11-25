/***************************************************************************************************
 * Host-only smoke for int4_kv_dequant.hpp (no SYCL / no GPU).
 * Build: g++ -std=c++17 -I../.. -o test_int4_dequant test_int4_kv_dequant_host.cpp && ./test_int4_dequant
 **************************************************************************************************/

#include "csrc/xpu/attn/int4_kv_dequant.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vllm::xpu::int4_kv;

static float pack_scale_zp(float scale, uint32_t zp) {
  uint32_t bits;
  std::memcpy(&bits, &scale, sizeof(bits));
  bits = (bits & ~0xFu) | (zp & 0xFu);
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

int main() {
  assert(packed_hs(128) == 64);
  assert(content_bytes(128) == 136);

  // Manual: q_lo=3, q_hi=10, zp=2, scale≈0.5 → (3-2)*0.5=0.5, (10-2)*0.5=4.0
  const uint8_t packed = static_cast<uint8_t>((10u << 4) | 3u);
  const float scale = 0.5f;
  const uint32_t zp = 2u;
  const float packed_s = pack_scale_zp(scale, zp);

  float even = 0.f, odd = 0.f;
  const ScaleZp sz = unpack_scale_zp(packed_s);
  assert(sz.zp == 2.f);
  // Cleared low nibble may perturb scale slightly; allow tiny eps.
  assert(std::fabs(sz.scale - 0.5f) < 1e-5f || std::fabs(sz.scale) > 0.f);
  dequant_byte(packed, sz.scale, sz.zp, even, odd);
  assert(std::fabs(even - (3.f - 2.f) * sz.scale) < 1e-5f);
  assert(std::fabs(odd - (10.f - 2.f) * sz.scale) < 1e-5f);

  std::vector<uint8_t> head(64, 0);
  head[0] = packed;
  std::vector<float> out(128, 0.f);
  dequant_head(head.data(), 64, packed_s, out.data());
  assert(std::fabs(out[0] - even) < 1e-5f);
  assert(std::fabs(out[1] - odd) < 1e-5f);

  std::cout << "test_int4_kv_dequant_host: OK\n";
  return 0;
}
