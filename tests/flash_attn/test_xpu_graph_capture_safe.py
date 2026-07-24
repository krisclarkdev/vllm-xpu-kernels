# SPDX-License-Identifier: Apache-2.0
"""Fail-closed + capture auto-on for FA2 / flash_attn_varlen_func."""
from __future__ import annotations

import logging
import os

import pytest
import torch

import vllm_xpu_kernels.flash_attn_interface as fai
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


def _truthy_env(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)


def _clear_fail_env(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_XPU_ATTN_FAIL_ON_FALLBACK", raising=False)


@pytest.mark.parametrize("val,expect", [
    ("1", True),
    ("ON", True),
    ("true", True),
    ("Yes", True),
    ("0", False),
    ("", False),
])
def test_env_fail_on_fallback(monkeypatch, val, expect):
    if val == "":
        _clear_fail_env(monkeypatch)
    else:
        _truthy_env(monkeypatch, "VLLM_XPU_ATTN_FAIL_ON_FALLBACK", val)
    assert fai._env_fail_on_fallback() is expect


def test_should_fail_closed_when_capturing(monkeypatch):
    _clear_fail_env(monkeypatch)
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: True)
    assert fai._should_fail_closed() is True


def test_should_fail_closed_env_without_capturing(monkeypatch):
    _truthy_env(monkeypatch, "VLLM_XPU_ATTN_FAIL_ON_FALLBACK", "1")
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: False)
    assert fai._should_fail_closed() is True


def _minimal_varlen_args(device: torch.device):
    headdim = 64
    nq, nkv = 8, 2
    block = 64
    batch = 1
    kv_len = 128
    q = torch.randn(batch, nq, headdim, dtype=torch.float16, device=device)
    k = torch.randn(
        2, block, nkv, headdim, dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused = torch.tensor([kv_len], dtype=torch.int32, device=device)
    block_table = torch.zeros(batch, 2, dtype=torch.int32, device=device)
    block_table[0, 0] = 0
    block_table[0, 1] = 1
    return dict(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=1,
        cu_seqlens_q=cu_q,
        max_seqlen_k=kv_len,
        seqused_k=seqused,
        block_table=block_table,
        softmax_scale=headdim**-0.5,
        causal=True,
    )


def test_fail_closed_env_raises_on_not_compiled(monkeypatch):
    if not fai.FA2_AVAILABLE:
        pytest.skip("FA2 extension not available")
    if not torch.xpu.is_available():
        pytest.skip("XPU required for FA2 dispatch path")

    _truthy_env(monkeypatch, "VLLM_XPU_ATTN_FAIL_ON_FALLBACK", "1")
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: False)

    def boom(*_a, **_k):
        raise RuntimeError("Paged decode kernel not compiled for this config")

    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", boom)
    args = _minimal_varlen_args(torch.device("xpu"))
    with pytest.raises(RuntimeError, match="FAIL_ON_FALLBACK|refusing"):
        flash_attn_varlen_func(**args)


def test_fail_closed_capturing_raises_without_env(monkeypatch):
    if not fai.FA2_AVAILABLE:
        pytest.skip("FA2 extension not available")
    if not torch.xpu.is_available():
        pytest.skip("XPU required for FA2 dispatch path")

    _clear_fail_env(monkeypatch)
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: True)

    def boom(*_a, **_k):
        raise RuntimeError("Paged decode kernel not compiled for this config")

    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", boom)
    args = _minimal_varlen_args(torch.device("xpu"))
    with pytest.raises(RuntimeError, match="refusing|capture"):
        flash_attn_varlen_func(**args)


def test_fallback_allowed_when_env_off(monkeypatch, caplog):
    if not fai.FA2_AVAILABLE:
        pytest.skip("FA2 extension not available")
    if not torch.xpu.is_available():
        pytest.skip("XPU required for FA2 dispatch path")

    _clear_fail_env(monkeypatch)
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: False)

    def boom(*_a, **_k):
        raise RuntimeError("Paged decode kernel not compiled for this config")

    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", boom)
    args = _minimal_varlen_args(torch.device("xpu"))
    with caplog.at_level(logging.WARNING, logger=fai.__name__):
        out = flash_attn_varlen_func(**args)
    assert out is not None
    assert any("falling back" in r.message for r in caplog.records)
