# Stop rules (B70 tok/s hunt)

Keep a change only if **at least one** holds:

1. End-to-end decode tok/s improves by **≥10%** at fixed quality on B70, or
2. Target-op microbench improves by **≥10–15%** geomean **and** the op is
   ≥~25% of the decode step in [`DECODE_PROFILE.md`](DECODE_PROFILE.md).

Drop immediately if:

- Prefill-only FMHA prefetch/clear knobs (Decision #4: ~0.96–0.97×).
- Uplift &lt;~3% micro **and** &lt;~5% E2E after one solid A/B.
- Any accuracy regression (especially `kv_tile=_128` / gpt-oss paths).

Applied on this hunt:

| Change | Verdict |
|--------|---------|
| Pivot #6 non-causal h192 config | **Keep** (fallback gate; catastrophic if missing) |
| Compact-grid from `seqused_k` | **Keep** (occupancy; zero cost when splits=1) |
| Mix-batch Split-K | **Keep** (long-KV mix serving) |
| `kv_tile=_128` with ReduceK=4 layout | **Keep if** decode A/B ≥10% on page_size≥128 **and** accuracy OK |
| MoE `*_policy_m_8` for tiny avg M | **Keep if** MoE micro ≥10% at decode-M |
| Prefill FMHA #4 knobs | **Drop** (already dead) |
