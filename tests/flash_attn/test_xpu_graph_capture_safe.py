# SPDX-License-Identifier: Apache-2.0
"""Capture-safety and fail-closed behavior for FA2 / flash_attn_varlen_func."""
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


def test_host_kv_lens_for_plan_cpu_ok_under_capture(monkeypatch):
    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: True)
    cpu = torch.tensor([128, 256], dtype=torch.int32)
    assert fai._host_kv_lens_for_plan(cpu) is cpu
    assert fai._host_kv_lens_for_plan([128, 256]) == [128, 256]


def _minimal_varlen_args(device: torch.device):
    """Tiny tensors sufficient to reach the FA2 try/except path."""
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


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")
def test_host_plan_device_kv_lens_rejected_under_capture(monkeypatch):
    if not fai.FA2_AVAILABLE:
        pytest.skip("FA2 extension not available")

    monkeypatch.setattr(fai, "_is_xpu_capturing", lambda: True)
    device_lens = torch.tensor([512, 1024], dtype=torch.int32, device="xpu")
    with pytest.raises(RuntimeError, match="host_kv_lens must be a CPU"):
        fai._host_kv_lens_for_plan(device_lens)

    # End-to-end through flash_attn_varlen_func planning branch.
    args = _minimal_varlen_args(torch.device("xpu"))
    args["host_kv_lens"] = torch.tensor([128], dtype=torch.int32, device="xpu")
    args["seqused_k"] = None
    args["num_splits_kv"] = 4
    with pytest.raises(RuntimeError, match="host_kv_lens must be a CPU"):
        flash_attn_varlen_func(**args)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")
@pytest.mark.skipif(
    not hasattr(torch.xpu, "XPUGraph"),
    reason="torch.xpu.XPUGraph not available")
def test_xpugraph_capture_replay_paged_decode():
    """Warmup → capture → replay on a default.conf-friendly shape."""
    if not fai.FA2_AVAILABLE:
        pytest.skip("FA2 extension not available")

    torch.set_default_device("xpu")
    headdim = 128
    nq, nkv = 8, 2
    block = 64
    batch = 2
    kv_lens = [256, 512]
    max_k = max(kv_lens)
    max_blocks = (max_k + block - 1) // block
    total_blocks = sum((L + block - 1) // block for L in kv_lens)

    k = torch.randn(
        total_blocks, block, nkv, headdim, dtype=torch.float16, device="xpu")
    v = torch.randn_like(k)
    q = torch.randn(batch, nq, headdim, dtype=torch.float16, device="xpu")
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, 1, 2], dtype=torch.int32, device="xpu")
    seqused = torch.tensor(kv_lens, dtype=torch.int32, device="xpu")
    bt = torch.zeros(batch, max_blocks, dtype=torch.int32, device="xpu")
    cursor = 0
    for i, L in enumerate(kv_lens):
        nb = (L + block - 1) // block
        bt[i, :nb] = torch.arange(cursor, cursor + nb, dtype=torch.int32,
                                  device="xpu")
        cursor += nb
    scale = headdim**-0.5

    def run():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            1,
            cu_q,
            max_k,
            seqused_k=seqused,
            softmax_scale=scale,
            causal=True,
            block_table=bt,
            out=out,
            num_splits_kv=None,
            is_mix_batch=True,
        )

    # Eager warmup (JIT / first-touch) outside capture.
    for _ in range(3):
        run()
    torch.xpu.synchronize()

    g = torch.xpu.XPUGraph()
    # Side-stream warmup pattern recommended for XPUGraph.
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(2):
            run()
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    with torch.xpu.graph(g):
        captured = run()
    torch.xpu.synchronize()

    for _ in range(5):
        g.replay()
        torch.xpu.synchronize()
        result = out if captured is None else (
            captured[0] if isinstance(captured, tuple) else captured)
        assert torch.isfinite(result).all(), "non-finite output after replay"
