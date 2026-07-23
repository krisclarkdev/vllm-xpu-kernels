# SPDX-License-Identifier: Apache-2.0
"""Cascade / shared-prefix attention (FlashInfer-inspired).

Runs attention over a shared KV prefix once, then per-request suffixes, and
merges partial outputs with ``merge_attn_states`` (LSE-aware). This is a
host-orchestrated cascade on top of existing XPU kernels — not a new device
kernel — and is intended for multi-tenant RAG / system-prompt batches.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .flash_attn_interface import flash_attn_varlen_func


def cascade_varlen_attention(
    q: torch.Tensor,
    *,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    suffix_k: torch.Tensor,
    suffix_v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    cu_seqlens_suffix: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_prefix: int,
    max_seqlen_suffix: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    return_softmax_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute attention as merge(prefix_attn, suffix_attn).

    ``q`` is ragged ``[total_q, H, D]``. Prefix K/V are shared across the
    batch (same layout as a single varlen sequence expanded per request, or
    a broadcastable shared block). Suffix K/V hold per-request unique tokens.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # Prefix pass (usually non-causal vs shared prompt).
    prefix_out, prefix_lse = flash_attn_varlen_func(
        q,
        prefix_k,
        prefix_v,
        cu_seqlens_q,
        cu_seqlens_prefix,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_prefix,
        softmax_scale=softmax_scale,
        causal=False,
        return_softmax_lse=True,
    )

    # Suffix pass (causal against per-request unique KV).
    suffix_out, suffix_lse = flash_attn_varlen_func(
        q,
        suffix_k,
        suffix_v,
        cu_seqlens_q,
        cu_seqlens_suffix,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_suffix,
        softmax_scale=softmax_scale,
        causal=causal,
        return_softmax_lse=True,
    )

    out = torch.empty_like(prefix_out)
    out_lse = torch.empty_like(prefix_lse) if return_softmax_lse else None
    torch.ops._C.merge_attn_states(
        out,
        out_lse,
        prefix_out,
        prefix_lse,
        suffix_out,
        suffix_lse,
    )
    if return_softmax_lse:
        return out, out_lse
    return out


def plan_shared_prefix_lengths(
    prefix_len: int,
    suffix_lens: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build cu_seqlens for a uniform shared prefix + per-req suffixes."""
    n = len(suffix_lens)
    cu_prefix = torch.zeros(n + 1, dtype=torch.int32)
    cu_suffix = torch.zeros(n + 1, dtype=torch.int32)
    for i, sl in enumerate(suffix_lens):
        cu_prefix[i + 1] = cu_prefix[i] + prefix_len
        cu_suffix[i + 1] = cu_suffix[i] + int(sl)
    return cu_prefix, cu_suffix, prefix_len, max(suffix_lens) if suffix_lens else 0
