/***************************************************************************************************
 * Copyright (C) 2025 — FA2 int4-KV greenfield (fork-local).
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * M1 vertical slice: paged decode with in-kernel int4 per-token-head dequant.
 * Scope: head_size=128, block_size multiple of 64 (policy target block=64),
 * Q in fp16/bf16, GQA, causal decode (q_len typically 1), optional local window.
 *
 * This is a correctness-first SYCL path (not Cute/DPAS FA2). It dequants K/V
 * inside the attention loop via int4_kv_dequant.hpp — no gather→fp16 materialize.
 * Cute DecodeFwdMainloop Int4KV integration is a follow-on once Oracle A passes.
 **************************************************************************************************/

#pragma once

#include "csrc/xpu/attn/int4_kv_dequant.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>

namespace vllm::xpu::xe2::int4_decode {

inline constexpr int kHeadSize = 128;
inline constexpr int kPackedHs = 64;  // head_size / 2
inline constexpr int kTargetBlock = 64;
inline constexpr int kWgSize = 32;  // one Xe subgroup; each WI owns 4 dims

static_assert(kHeadSize % kWgSize == 0);
inline constexpr int kDimsPerWi = kHeadSize / kWgSize;

struct Int4DecodeParams {
  const void* query;       // ElementQ [total_q, Hq, 128]
  const uint8_t* key;      // [blocks, block_size, Hkv, 64]
  const uint8_t* value;    // [blocks, block_size, Hkv, 64]
  const float* k_scale;    // [blocks, block_size, Hkv] packed scale+zp
  const float* v_scale;    // [blocks, block_size, Hkv]
  void* out;               // ElementQ [total_q, Hq, 128]
  const int32_t* block_table;  // [batch, max_blocks]
  const int32_t* cu_seqlens_q;
  const int32_t* seqused_k;  // [batch] KV lengths
  float sm_scale;
  int batch_size;
  int num_heads_q;
  int num_heads_kv;
  int block_size;
  int max_blocks_per_seq;
  int window_size_left;   // -1 => full
  int window_size_right;  // -1 => full
  bool is_local;
  // strides (elements)
  int64_t q_stride_seq;
  int64_t q_stride_heads;
  int64_t k_stride_page;
  int64_t k_stride_seq;
  int64_t k_stride_heads;
  int64_t v_stride_page;
  int64_t v_stride_seq;
  int64_t v_stride_heads;
  int64_t ks_stride_page;
  int64_t ks_stride_seq;
  int64_t ks_stride_heads;
  int64_t vs_stride_page;
  int64_t vs_stride_seq;
  int64_t vs_stride_heads;
  int64_t o_stride_seq;
  int64_t o_stride_heads;
  int64_t bt_stride_batch;
};

template <typename ElementQ>
struct PagedDecodeInt4Kernel {
  Int4DecodeParams p;

  void operator()(sycl::nd_item<1> it) const {
    using namespace vllm::xpu::int4_kv;

    const int wg = static_cast<int>(it.get_group(0));
    const int lid = static_cast<int>(it.get_local_id(0));
    const int batch = wg / p.num_heads_q;
    const int q_head = wg % p.num_heads_q;
    if (batch >= p.batch_size) {
      return;
    }

    const int q_group = p.num_heads_q / p.num_heads_kv;
    const int kv_head = q_head / q_group;

    const int q_begin = p.cu_seqlens_q[batch];
    const int q_end = p.cu_seqlens_q[batch + 1];
    const int q_len = q_end - q_begin;
    if (q_len <= 0) {
      return;
    }
    // M1 decode slice: use the last query token in the sequence (standard
    // decode / speculative last-position). Prefill rows are not handled here.
    const int q_tok = q_end - 1;
    const int kv_len = p.seqused_k[batch];
    if (kv_len <= 0) {
      return;
    }

    const int dim0 = lid * kDimsPerWi;

    // Load Q slice for this WI.
    const ElementQ* q_ptr = static_cast<const ElementQ*>(p.query) +
                            static_cast<int64_t>(q_tok) * p.q_stride_seq +
                            static_cast<int64_t>(q_head) * p.q_stride_heads;
    float q_reg[kDimsPerWi];
#pragma unroll
    for (int i = 0; i < kDimsPerWi; ++i) {
      q_reg[i] = static_cast<float>(q_ptr[dim0 + i]);
    }

    float m_i = -INFINITY;
    float l_i = 0.f;
    float acc[kDimsPerWi];
#pragma unroll
    for (int i = 0; i < kDimsPerWi; ++i) {
      acc[i] = 0.f;
    }

    const int decode_row = kv_len - 1;
    const int win_left = p.is_local ? p.window_size_left : kv_len;
    const int win_right = p.is_local ? p.window_size_right : kv_len;

    for (int pos = 0; pos < kv_len; ++pos) {
      if (p.is_local) {
        if (pos < decode_row - win_left || pos > decode_row + win_right) {
          continue;
        }
      }

      const int page_idx = pos / p.block_size;
      const int slot = pos - page_idx * p.block_size;
      const int32_t block_id =
          p.block_table[static_cast<int64_t>(batch) * p.bt_stride_batch +
                        page_idx];

      const float packed_sk =
          p.k_scale[static_cast<int64_t>(block_id) * p.ks_stride_page +
                    static_cast<int64_t>(slot) * p.ks_stride_seq +
                    static_cast<int64_t>(kv_head) * p.ks_stride_heads];
      const float packed_sv =
          p.v_scale[static_cast<int64_t>(block_id) * p.vs_stride_page +
                    static_cast<int64_t>(slot) * p.vs_stride_seq +
                    static_cast<int64_t>(kv_head) * p.vs_stride_heads];

      const uint8_t* k_row =
          p.key + static_cast<int64_t>(block_id) * p.k_stride_page +
          static_cast<int64_t>(slot) * p.k_stride_seq +
          static_cast<int64_t>(kv_head) * p.k_stride_heads;
      const uint8_t* v_row =
          p.value + static_cast<int64_t>(block_id) * p.v_stride_page +
          static_cast<int64_t>(slot) * p.v_stride_seq +
          static_cast<int64_t>(kv_head) * p.v_stride_heads;

      const ScaleZp sk = unpack_scale_zp(packed_sk);
      const ScaleZp sv = unpack_scale_zp(packed_sv);

      // Partial dot over this WI's dims; subgroup-sum → full score.
      float partial = 0.f;
#pragma unroll
      for (int i = 0; i < kDimsPerWi; ++i) {
        const int d = dim0 + i;
        const int byte_i = d >> 1;
        float even, odd;
        dequant_byte(k_row[byte_i], sk.scale, sk.zp, even, odd);
        const float kd = (d & 1) ? odd : even;
        partial += q_reg[i] * kd;
      }
      const float score =
          sycl::reduce_over_group(it.get_group(), partial, sycl::plus<>()) *
          p.sm_scale;

      const float m_new = sycl::max(m_i, score);
      const float alpha = sycl::native::exp(m_i - m_new);
      const float p_ij = sycl::native::exp(score - m_new);
      l_i = l_i * alpha + p_ij;
#pragma unroll
      for (int i = 0; i < kDimsPerWi; ++i) {
        const int d = dim0 + i;
        const int byte_i = d >> 1;
        float even, odd;
        dequant_byte(v_row[byte_i], sv.scale, sv.zp, even, odd);
        const float vd = (d & 1) ? odd : even;
        acc[i] = acc[i] * alpha + p_ij * vd;
      }
      m_i = m_new;
    }

    ElementQ* o_ptr = static_cast<ElementQ*>(p.out) +
                      static_cast<int64_t>(q_tok) * p.o_stride_seq +
                      static_cast<int64_t>(q_head) * p.o_stride_heads;
    const float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
#pragma unroll
    for (int i = 0; i < kDimsPerWi; ++i) {
      o_ptr[dim0 + i] = static_cast<ElementQ>(acc[i] * inv_l);
    }
  }
};

template <typename ElementQ>
inline void launch_paged_decode_int4(sycl::queue& queue,
                                     const Int4DecodeParams& params) {
  const size_t n_wg =
      static_cast<size_t>(params.batch_size) *
      static_cast<size_t>(params.num_heads_q);
  queue.submit([&](sycl::handler& cgh) {
    PagedDecodeInt4Kernel<ElementQ> kn{params};
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(n_wg * static_cast<size_t>(kWgSize)),
            sycl::range<1>(static_cast<size_t>(kWgSize))),
        kn);
  });
}

}  // namespace vllm::xpu::xe2::int4_decode
