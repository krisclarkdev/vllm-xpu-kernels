# Decode profile (B70)

Host: `kclark@hal`, device `Intel(R) Graphics [0xe223]` (Arc Pro B70).
Measured inside `vllm-kernel-ab` with `ZE_AFFINITY_MASK=0`, oneAPI 2025.3.
Date: 2026-07-23.

## Method

Synthetic decode-step share from [`tools/decode_profile.py`](../tools/decode_profile.py):

- Paged decode (h128 / h192, KV 4096/8192, non-causal)
- Dense FP16 GEMM at batch 1/8/32, hidden 4096
- Eager RMSNorm + SiLU-mul (upper bound on elementwise cost)

Raw JSON: [`decode_profile.json`](decode_profile.json).

## Results (median ms)

| Op | ms |
|----|-----|
| paged_decode h128 kv4096 | 0.086 |
| paged_decode h128 kv8192 | 0.083 |
| paged_decode h192 kv4096 | 0.084 |
| paged_decode h192 kv8192 | 0.082 |
| dense_gemm M8×4096×4096 | 0.105 |
| dense_gemm M8×16384×4096 | 0.270 |
| dense_gemm M1×4096×4096 | 0.104 |
| dense_gemm M32×4096×4096 | 0.098 |
| rms_norm (eager) | 0.155 |
| silu_mul (eager) | 0.080 |

## Synthetic dense layer share (32 layers)

| Bucket | % of step |
|--------|-----------|
| **dense GEMM (×4/layer)** | **~47%** |
| norm/act (eager, inflated) | ~44% |
| paged_decode | ~10% |

Estimated step ~28.6 ms → ~35 tok/s single-stream (order-of-magnitude only).

**Primary bucket: dense_gemm** (weight bandwidth at small M).

Note: eager norm/act overstates that share vs fused XPU kernels; even if norms drop 5×, GEMM remains first-order and attention stays secondary for this short-KV single-seq shape. Attention share rises for long KV / mixed continuous batching.

## Ranked attacks for this hardware

1. **GEMM / MoE small-M** — use `*_policy_m_8` for decode-avg tokens/expert; keep quantized oneDNN paths hot.
2. **Pivot #6 gate** — non-causal h192 in `paged_decode_default.conf` (no FA2→PyTorch fallback).
3. **Attn occupancy (secondary here, still land)** — compact-grid from `seqused_k`, mix-batch Split-K, `kv_tile=_128` with ReduceK=4-safe layout.
4. **Stop rule** — keep only ≥10% E2E tok/s or ≥10–15% micro on the target op; drop prefill FMHA knobs (already dead).

## Gate checks

- AOT target: `VLLM_XPU_XE2_AOT_DEVICES=bmg`
- DaemonSet `inference/vllm-xpu` paused during exclusive A/B
- Fail-on-fallback decode benches after cfg6 rebuild
