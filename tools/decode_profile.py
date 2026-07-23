#!/usr/bin/env python3
"""Decode-step op-time share on XPU (paged attn + GEMM + norms + cache).

Intended for B70 / Xe2. Run with:
  PYTHONPATH=<repo_root> ZE_AFFINITY_MASK=0 python tools/decode_profile.py

Writes JSON + markdown-friendly summary to stdout and optional --out path.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass
class TimedOp:
    name: str
    median_ms: float
    notes: str = ""


def _median_ms(times: list[float]) -> float:
    return statistics.median(times) * 1e3


def _bench(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    times: list[float] = []
    for _ in range(iters):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        times.append(time.perf_counter() - t0)
    return _median_ms(times)


def bench_paged_decode(
    headdim: int,
    *,
    kv_len: int,
    nq: int = 16,
    nkv: int = 2,
    block_size: int = 64,
    causal: bool = False,
    warmup: int = 10,
    iters: int = 40,
) -> TimedOp:
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

    dtype = torch.float16
    q = torch.randn(1, nq, headdim, dtype=dtype, device="xpu")
    num_blocks = (kv_len + block_size - 1) // block_size
    k_cache = torch.randn(
        num_blocks, block_size, nkv, headdim, dtype=dtype, device="xpu")
    v_cache = torch.randn_like(k_cache)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    seq_k = torch.tensor([kv_len], dtype=torch.int32, device="xpu")
    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device="xpu").view(1, -1)
    scale = headdim**-0.5

    def run():
        return flash_attn_varlen_func(
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

    ms = _bench(run, warmup=warmup, iters=iters)
    return TimedOp(
        name=f"paged_decode_h{headdim}_kv{kv_len}",
        median_ms=ms,
        notes=f"nq={nq},nkv={nkv},bs={block_size},causal={causal}",
    )


def bench_dense_gemm(
    m: int,
    n: int,
    k: int,
    *,
    warmup: int,
    iters: int,
) -> TimedOp:
    dtype = torch.float16
    a = torch.randn(m, k, dtype=dtype, device="xpu")
    b = torch.randn(k, n, dtype=dtype, device="xpu")

    def run():
        return torch.mm(a, b)

    ms = _bench(run, warmup=warmup, iters=iters)
    flops = 2.0 * m * n * k
    tflops = (flops / (ms * 1e-3)) / 1e12 if ms > 0 else 0.0
    return TimedOp(
        name=f"dense_gemm_m{m}_n{n}_k{k}",
        median_ms=ms,
        notes=f"fp16,{tflops:.2f} TFLOPS",
    )


def bench_rms_norm(hidden: int, *, batch: int, warmup: int,
                   iters: int) -> TimedOp:
    dtype = torch.float16
    x = torch.randn(batch, hidden, dtype=dtype, device="xpu")
    w = torch.ones(hidden, dtype=dtype, device="xpu")

    def run():
        # Reference RMSNorm; custom op may not be on PYTHONPATH path.
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + 1e-6) * w

    ms = _bench(run, warmup=warmup, iters=iters)
    return TimedOp(name=f"rms_norm_h{hidden}_b{batch}", median_ms=ms)


def bench_silu_mul(hidden: int, *, batch: int, warmup: int,
                   iters: int) -> TimedOp:
    dtype = torch.float16
    x = torch.randn(batch, hidden * 2, dtype=dtype, device="xpu")

    def run():
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

    ms = _bench(run, warmup=warmup, iters=iters)
    return TimedOp(name=f"silu_mul_h{hidden}_b{batch}", median_ms=ms)


def synthetic_layer_share(ops: list[TimedOp], *, n_layers: int = 32) -> dict:
    """Approximate a dense transformer decode step from measured kernels.

    Per layer: attn + 4 GEMMs (QKV fused~3, O, gate/up, down) + 2 norms + silu.
    """
    by = {o.name: o.median_ms for o in ops}

    def pick(prefix: str) -> float:
        for k, v in by.items():
            if k.startswith(prefix):
                return v
        return 0.0

    attn = pick("paged_decode_h128")
    # Prefer mid-size decode GEMM as stand-in for projection weight BW.
    gemm_candidates = [v for k, v in by.items() if k.startswith("dense_gemm_m")]
    gemm = statistics.median(gemm_candidates) if gemm_candidates else 0.0
    norm = pick("rms_norm")
    silu = pick("silu_mul")

    per_layer = attn + 4 * gemm + 2 * norm + silu
    step = per_layer * n_layers
    shares = {
        "paged_decode_pct": 100.0 * (attn * n_layers) / step if step else 0.0,
        "dense_gemm_pct": 100.0 * (4 * gemm * n_layers) / step if step else 0.0,
        "norm_act_pct": 100.0 * ((2 * norm + silu) * n_layers) / step
                        if step else 0.0,
        "per_layer_ms": per_layer,
        "step_ms_est": step,
        "tok_per_s_est": 1000.0 / step if step else 0.0,
        "n_layers": n_layers,
        "attn_ms": attn,
        "gemm_ms_each": gemm,
    }
    return shares


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--kv-lens", default="4096,8192")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--n-layers", type=int, default=32)
    args = parser.parse_args()

    torch.set_default_device("xpu")
    assert torch.xpu.is_available(), "XPU required"

    kv_lens = [int(x) for x in args.kv_lens.split(",") if x]
    results: list[TimedOp] = []

    print(f"DEVICE={torch.xpu.get_device_name(0)}")
    print(f"BATCH={args.batch} HIDDEN={args.hidden}")

    for h in (128, 192):
        for kv in kv_lens:
            try:
                results.append(
                    bench_paged_decode(
                        h,
                        kv_len=kv,
                        causal=False,
                        warmup=args.warmup,
                        iters=max(args.iters, 40),
                    ))
                print(
                    f"OK {results[-1].name} {results[-1].median_ms:.4f} ms",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                results.append(
                    TimedOp(
                        name=f"paged_decode_h{h}_kv{kv}",
                        median_ms=-1.0,
                        notes=f"ERROR:{type(e).__name__}:{e}",
                    ))
                print(f"ERR {results[-1].name} {results[-1].notes}", flush=True)

    # Decode-shaped GEMMs: M=batch, N=hidden or 4*hidden, K=hidden
    gemm_shapes = [
        (args.batch, args.hidden, args.hidden),
        (args.batch, args.hidden * 4, args.hidden),
        (args.batch, args.hidden, args.hidden * 4),
        (1, args.hidden, args.hidden),
        (32, args.hidden, args.hidden),
    ]
    for m, n, k in gemm_shapes:
        try:
            results.append(
                bench_dense_gemm(
                    m, n, k, warmup=args.warmup, iters=args.iters))
            print(
                f"OK {results[-1].name} {results[-1].median_ms:.4f} ms "
                f"({results[-1].notes})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            results.append(
                TimedOp(
                    name=f"dense_gemm_m{m}_n{n}_k{k}",
                    median_ms=-1.0,
                    notes=f"ERROR:{type(e).__name__}:{e}",
                ))
            print(f"ERR {results[-1].name} {results[-1].notes}", flush=True)

    for fn in (bench_rms_norm, bench_silu_mul):
        try:
            results.append(
                fn(args.hidden,
                   batch=args.batch,
                   warmup=args.warmup,
                   iters=args.iters))
            print(
                f"OK {results[-1].name} {results[-1].median_ms:.4f} ms",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            results.append(
                TimedOp(
                    name=fn.__name__,
                    median_ms=-1.0,
                    notes=f"ERROR:{type(e).__name__}:{e}",
                ))
            print(f"ERR {results[-1].name} {results[-1].notes}", flush=True)

    ok_ops = [o for o in results if o.median_ms > 0]
    share = synthetic_layer_share(ok_ops, n_layers=args.n_layers)

    # Rank attack list
    ranked = []
    if share["paged_decode_pct"] >= 25:
        ranked.append("attn:compact-grid / mixbatch-splitk / kv_tile=_128")
    if share["dense_gemm_pct"] >= 25:
        ranked.append("gemm: quantized oneDNN + small-M recipes")
    ranked.append("gate: no FA2 fallback (Pivot #6 h192 non-causal)")

    payload = {
        "device": torch.xpu.get_device_name(0),
        "ops": [asdict(o) for o in results],
        "synthetic_dense_layer_share": share,
        "ranked_attacks": ranked,
        "primary_bucket": (
            "paged_decode" if share["paged_decode_pct"]
            >= share["dense_gemm_pct"] else "dense_gemm"),
    }

    print("\n=== SYNTHETIC DENSE DECODE SHARE ===")
    print(json.dumps(share, indent=2))
    print("PRIMARY=", payload["primary_bucket"])
    print("RANKED=", ranked)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
