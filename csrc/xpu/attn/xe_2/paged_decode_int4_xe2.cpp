/***************************************************************************************************
 * Copyright (C) 2025 — FA2 int4-KV greenfield (fork-local).
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Host launcher for M1 xe_2 int4 paged_decode (hd=128, block=64 policy).
 **************************************************************************************************/

#include "paged_decode_int4.hpp"
#include "paged_decode_xe2.h"

#include <c10/core/ScalarType.h>

namespace vllm::xpu::xe2 {
namespace {

using int4_decode::Int4DecodeParams;
using int4_decode::kHeadSize;
using int4_decode::kPackedHs;
using int4_decode::kTargetBlock;

bool is_per_token_head_scale(const at::Tensor& t) {
  if (t.scalar_type() != at::ScalarType::Float || t.dim() != 3) {
    return false;
  }
  // [num_blocks, block_size, num_kv_heads] — reject single-value fp8 broadcast.
  return t.numel() > 1;
}

}  // namespace

bool is_int4_paged_decode_request(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const std::optional<const at::Tensor>& k_scale,
    const std::optional<const at::Tensor>& v_scale) {
  (void)query;
  if (key_cache.scalar_type() != at::ScalarType::Byte ||
      value_cache.scalar_type() != at::ScalarType::Byte) {
    return false;
  }
  if (!k_scale.has_value() || !v_scale.has_value()) {
    return false;
  }
  return is_per_token_head_scale(*k_scale) && is_per_token_head_scale(*v_scale);
}

void cutlass_paged_decode_int4_xe2(
    sycl::queue& queue,
    const at::Tensor& query,      // [total_q, Hq, 128]
    const at::Tensor& key_cache,  // [blocks, block_size, Hkv, 64] uint8
    const at::Tensor& value_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,  // used as seqused_k when paged decode
    int /*max_seqlen_q*/,
    int /*max_seqlen_k*/,
    const at::Tensor& k_scale,  // [blocks, block_size, Hkv] float32
    const at::Tensor& v_scale,
    double sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_local) {
  TORCH_CHECK(query.dim() == 3, "int4 paged_decode: query must be [T, Hq, D]");
  TORCH_CHECK(
      key_cache.dim() == 4 && value_cache.dim() == 4,
      "int4 paged_decode: K/V must be [blocks, block_size, Hkv, packed_hs]");
  TORCH_CHECK(
      key_cache.scalar_type() == at::ScalarType::Byte &&
          value_cache.scalar_type() == at::ScalarType::Byte,
      "int4 paged_decode: K/V must be uint8 packed");

  const int head_size = static_cast<int>(query.size(2));
  const int packed_hs = static_cast<int>(key_cache.size(3));
  const int block_size = static_cast<int>(key_cache.size(1));
  const int num_heads_q = static_cast<int>(query.size(1));
  const int num_heads_kv = static_cast<int>(key_cache.size(2));
  const int batch_size = static_cast<int>(cu_seqlens_q.numel()) - 1;

  TORCH_CHECK(
      head_size == kHeadSize,
      "int4 paged_decode M1 only supports head_size=",
      kHeadSize,
      " got ",
      head_size);
  TORCH_CHECK(
      packed_hs == kPackedHs,
      "int4 paged_decode: packed head dim must be ",
      kPackedHs,
      " got ",
      packed_hs);
  TORCH_CHECK(
      value_cache.size(3) == packed_hs,
      "int4 paged_decode: V packed head dim mismatch");
  TORCH_CHECK(
      block_size % kTargetBlock == 0,
      "int4 paged_decode M1 expects block_size multiple of ",
      kTargetBlock,
      " got ",
      block_size);
  TORCH_CHECK(
      num_heads_q % num_heads_kv == 0,
      "int4 paged_decode: Hq must be divisible by Hkv");
  TORCH_CHECK(
      query.stride(-1) == 1 && out.stride(-1) == 1 &&
          key_cache.stride(-1) == 1 && value_cache.stride(-1) == 1,
      "int4 paged_decode: last dims must be contiguous");
  TORCH_CHECK(
      k_scale.sizes() ==
              at::IntArrayRef(
                  {key_cache.size(0), key_cache.size(1), key_cache.size(2)}) &&
          v_scale.sizes() == k_scale.sizes(),
      "int4 paged_decode: scales must be [blocks, block_size, Hkv]");
  TORCH_CHECK(
      cu_seqlens_k.dim() == 1 && cu_seqlens_k.numel() == batch_size,
      "int4 paged_decode: seqused_k / cu_seqlens_k must be [batch]");

  Int4DecodeParams params{};
  params.query = query.data_ptr();
  params.key = key_cache.data_ptr<uint8_t>();
  params.value = value_cache.data_ptr<uint8_t>();
  params.k_scale = k_scale.data_ptr<float>();
  params.v_scale = v_scale.data_ptr<float>();
  params.out = out.data_ptr();
  params.block_table = block_table.data_ptr<int32_t>();
  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int32_t>();
  params.seqused_k = cu_seqlens_k.data_ptr<int32_t>();
  params.sm_scale = static_cast<float>(sm_scale);
  params.batch_size = batch_size;
  params.num_heads_q = num_heads_q;
  params.num_heads_kv = num_heads_kv;
  params.block_size = block_size;
  params.max_blocks_per_seq = static_cast<int>(block_table.size(1));
  params.window_size_left = window_size_left < 0 ? 0 : window_size_left;
  params.window_size_right = window_size_right < 0 ? 0 : window_size_right;
  params.is_local = is_local;
  params.q_stride_seq = query.stride(0);
  params.q_stride_heads = query.stride(1);
  params.k_stride_page = key_cache.stride(0);
  params.k_stride_seq = key_cache.stride(1);
  params.k_stride_heads = key_cache.stride(2);
  params.v_stride_page = value_cache.stride(0);
  params.v_stride_seq = value_cache.stride(1);
  params.v_stride_heads = value_cache.stride(2);
  params.ks_stride_page = k_scale.stride(0);
  params.ks_stride_seq = k_scale.stride(1);
  params.ks_stride_heads = k_scale.stride(2);
  params.vs_stride_page = v_scale.stride(0);
  params.vs_stride_seq = v_scale.stride(1);
  params.vs_stride_heads = v_scale.stride(2);
  params.o_stride_seq = out.stride(0);
  params.o_stride_heads = out.stride(1);
  params.bt_stride_batch = block_table.stride(0);

  if (query.scalar_type() == at::ScalarType::Half) {
    int4_decode::launch_paged_decode_int4<sycl::half>(queue, params);
  } else if (query.scalar_type() == at::ScalarType::BFloat16) {
    int4_decode::launch_paged_decode_int4<sycl::ext::oneapi::bfloat16>(
        queue, params);
  } else {
    TORCH_CHECK(
        false,
        "int4 paged_decode: Q must be fp16 or bf16, got ",
        query.scalar_type());
  }
}

}  // namespace vllm::xpu::xe2
