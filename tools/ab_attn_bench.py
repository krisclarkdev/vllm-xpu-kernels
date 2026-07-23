#!/usr/bin/env python3
"""A/B microbench for flash_attn varlen prefill + paged decode on XPU.

Fails hard if the XPU kernel falls back to the PyTorch reference path.
Run with PYTHONPATH=<repo_root> so tools/ does not shadow the package.
"""
from __future__ import annotations

import argparse
import logging
import statistics
import time
import warnings

import torch

warnings.filterwarnings("ignore")

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

# Head D-tile count for Xe2 chunk_policy_* (ShapeQK K-atom = 32).
# Prefetch windowing in fmha_mainloop_detail runs only when N_D > 4.
HEAD_ND = {64: 2, 96: 3, 128: 4, 192: 6, 256: 8, 512: 16}
LARGE_HEAD_ND_THRESHOLD = 4


class FallbackError(RuntimeError):
    """Raised when attention falls back to the PyTorch reference path."""


class _FallbackLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "falling back" in msg or "not compiled" in msg:
            self.messages.append(msg)


def _median_ms(times: list[float]) -> float:
    return statistics.median(times) * 1e3


def _call_with_fallback_guard(fn, *args, **kwargs):
    logger = logging.getLogger("vllm_xpu_kernels.flash_attn_interface")
    handler = _FallbackLogHandler()
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        out = fn(*args, **kwargs)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    if handler.messages:
        raise FallbackError(handler.messages[-1].split("\n", 1)[0])
    return out


def bench_prefill(
    headdim: int,
    *,
    seqlen: int = 1024,
    nheads: int = 16,
    batch: int = 1,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    dtype = torch.float16
    total = seqlen * batch
    q = torch.randn(total, nheads, headdim, dtype=dtype)
    k = torch.randn(total, nheads, headdim, dtype=dtype)
    v = torch.randn(total, nheads, headdim, dtype=dtype)
    cu = torch.arange(0, batch + 1, dtype=torch.int32) * seqlen
    scale = headdim**-0.5

    def run():
        return _call_with_fallback_guard(
            flash_attn_varlen_func,
            q,
            k,
            v,
            seqlen,
            cu,
            seqlen,
            cu,
            softmax_scale=scale,
            causal=True,
        )

    for _ in range(warmup):
        run()
    torch.xpu.synchronize()
    times: list[float] = []
    for _ in range(iters):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        run()
        torch.xpu.synchronize()
        times.append(time.perf_counter() - t0)
    return _median_ms(times)


def bench_decode(
    headdim: int,
    *,
    kv_len: int = 4096,
    nq: int = 16,
    nkv: int = 2,
    block_size: int = 64,
    causal: bool = True,
    warmup: int = 10,
    iters: int = 50,
) -> float:
    """Paged decode. Default causal=True matches paged_decode_default.conf."""
    dtype = torch.float16
    q = torch.randn(1, nq, headdim, dtype=dtype)
    num_blocks = (kv_len + block_size - 1) // block_size
    k_cache = torch.randn(num_blocks, block_size, nkv, headdim, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    cu_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_k = torch.tensor([kv_len], dtype=torch.int32)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(1, -1)
    scale = headdim**-0.5

    def run():
        return _call_with_fallback_guard(
            flash_attn_varlen_func,
            q,
            k_cache,
            v_cache,
            1,
            cu_q,
            kv_len,
            seqused_k=seq_k,
            softmax_scale=scale,
            causal=causal,
            block_table=block_tables,
        )

    for _ in range(warmup):
        run()
    torch.xpu.synchronize()
    times: list[float] = []
    for _ in range(iters):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        run()
        torch.xpu.synchronize()
        times.append(time.perf_counter() - t0)
    return _median_ms(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--heads", default="64,128,192")
    parser.add_argument(
        "--seqlens",
        default="4096,8192",
        help="Prefill sequence lengths (per sequence)",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--kv-lens", default="4096,8192")
    parser.add_argument(
        "--decode-causal",
        type=int,
        default=1,
        help="1=causal decode (default.conf), 0=non-causal",
    )
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    torch.set_default_device("xpu")
    assert torch.xpu.is_available(), "XPU required"
    print(f"LABEL={args.label}")
    print(f"DEVICE={torch.xpu.get_device_name(0)}")
    print(f"BATCH={args.batch}")
    print(f"DECODE_CAUSAL={bool(args.decode_causal)}")
    print("head,N_D,prefetch_window")
    heads = [int(x) for x in args.heads.split(",") if x]
    for h in heads:
        n_d = HEAD_ND.get(h, headdim_nd_guess(h))
        window = n_d > LARGE_HEAD_ND_THRESHOLD
        print(f"{h},{n_d},{int(window)}")

    seqlens = [int(x) for x in args.seqlens.split(",") if x]
    kv_lens = [int(x) for x in args.kv_lens.split(",") if x]

    print("op,head,len,paged,causal,N_D,median_ms")
    for h in heads:
        n_d = HEAD_ND.get(h, headdim_nd_guess(h))
        if not args.skip_prefill:
            for s in seqlens:
                try:
                    ms = bench_prefill(
                        h,
                        seqlen=s,
                        batch=args.batch,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    print(
                        f"prefill,{h},{s},0,1,{n_d},{ms:.4f}",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}:{e}".replace(",", ";")
                    print(
                        f"prefill,{h},{s},0,1,{n_d},ERROR:{err}",
                        flush=True,
                    )
        if not args.skip_decode:
            for kv in kv_lens:
                try:
                    ms = bench_decode(
                        h,
                        kv_len=kv,
                        causal=bool(args.decode_causal),
                        warmup=max(args.warmup, 10),
                        iters=max(args.iters, 40),
                    )
                    print(
                        f"decode,{h},{kv},1,{int(bool(args.decode_causal))},"
                        f"{n_d},{ms:.4f}",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}:{e}".replace(",", ";")
                    print(
                        f"decode,{h},{kv},1,{int(bool(args.decode_causal))},"
                        f"{n_d},ERROR:{err}",
                        flush=True,
                    )


def headdim_nd_guess(headdim: int) -> int:
    return max(1, (headdim + 31) // 32)


if __name__ == "__main__":
    main()
