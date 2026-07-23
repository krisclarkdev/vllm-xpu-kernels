# SPDX-License-Identifier: Apache-2.0
"""Fused attention prologue helpers for decode.

Composes existing XPU ops (fused QK-RMSNorm+RoPE and KV cache write) into a
single Python entrypoint so callers avoid extra host round-trips between
norm/RoPE and reshape_and_cache. Device-side fusion of cache write into the
norm/RoPE kernel remains a follow-up.
"""
from __future__ import annotations

from typing import Optional

import torch


def fused_qk_norm_rope_and_cache(
    qkv: torch.Tensor,
    *,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    is_neox_style: bool = True,
    forced_token_heads_per_warp: int = 0,
) -> torch.Tensor:
    """In-place QK norm+RoPE on ``qkv``, then write K/V into paged cache.

    Returns query as ``[num_tokens, num_heads_q, head_dim]`` for attention.
    """
    if qkv.dim() != 2:
        raise ValueError("qkv must be [num_tokens, (nq+nk+nv)*head_dim]")
    num_tokens = qkv.size(0)

    torch.ops._C.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        float(eps),
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox_style,
        position_ids,
        forced_token_heads_per_warp,
    )

    q_end = num_heads_q * head_dim
    k_end = q_end + num_heads_k * head_dim
    q = qkv[:, :q_end].view(num_tokens, num_heads_q, head_dim)
    k = qkv[:, q_end:k_end].view(num_tokens, num_heads_k, head_dim)
    v = qkv[:, k_end:].view(num_tokens, num_heads_v, head_dim)

    if k_scale is None:
        k_scale = torch.ones((), device=qkv.device, dtype=torch.float32)
    if v_scale is None:
        v_scale = torch.ones((), device=qkv.device, dtype=torch.float32)

    torch.ops._C_cache_ops.reshape_and_cache_flash(
        k,
        v,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    return q
