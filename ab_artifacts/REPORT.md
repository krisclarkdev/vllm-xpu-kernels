# Attention perf hunt report (B70)

Host: Intel(R) Graphics [0xe223] (B70). Prefill benches: batch=2, causal,
heads 128/192, seqlens 4096/8192, fail-on-fallback, `PYTHONPATH`=repo root.
Builds used `MAX_JOBS=16` / cmake job pool `compile=16`.

## N_D by head (Xe2 chunk_policy, D-atom=32)

| head | N_D | prefetch window (N_D > 4) |
|------|-----|---------------------------|
| 64   | 2   | no                        |
| 128  | 4   | no                        |
| 192  | 6   | **yes**                   |

## Decision #4 — dead end (restore baseline)

Compile knobs tested against `libattn_kernels_xe_2.base.so` (pre-#4 mainloop):

| variant | knobs (PREFETCH_D,THRESH,WINDOW,CLEAR) | prefill geomean | h192 geomean | notes |
|---------|------------------------------------------|-----------------|--------------|-------|
| new (full #4) | 2,4,1,1 | **0.96x** | **0.97x** | combined package slower |
| clear_only | 2,4,0,1 | 1.00x | **1.07x** | h192 win; h128 −6…−8% |
| prefetch_only | 2,4,1,0 | 0.99x | 1.02x | within ±2% dead-band overall |
| pref1 | 1,4,1,1 | 1.00x | 1.02x | |
| pref3 | 3,4,1,1 | 1.01x | 1.04x | |
| pref4 | 4,4,1,1 | 1.00x | 1.06x | h192 win; h128 regresses |

Stop criteria (≥3% h192 geomean **or** dead end ±2%): isolate `clear_only` /
`pref4` exceed 3% on h192 alone, but they regress h128 and do not improve overall
prefill geomean. Full #4 (`new`) is a clear loss (~−3…−4%).

**Action:** declare #4 dead end for shipping. Restored
`chunk_prefill_mainloop.hpp` to git/baseline (no FMHA_* knobs / clear-peel /
prefetch-window changes).

### Per-shape deltas (selected)

**base vs new (full #4):**

| shape | base ms | new ms | delta |
|-------|---------|--------|-------|
| prefill h128/4096 | 1.489 | 1.627 | −9.3% |
| prefill h128/8192 | 5.069 | 5.226 | −3.1% |
| prefill h192/4096 | 3.121 | 3.219 | −3.2% |
| prefill h192/8192 | 12.769 | 13.166 | −3.1% |

**base vs clear_only (best h192 isolate):**

| shape | base ms | clear_only ms | delta |
|-------|---------|---------------|-------|
| prefill h128/4096 | 1.489 | 1.585 | −6.5% |
| prefill h128/8192 | 5.073 | 5.485 | −8.1% |
| prefill h192/4096 | 3.462 | 3.079 | **+11.1%** |
| prefill h192/8192 | 13.219 | 12.933 | +2.2% |

## Pivot #6 — paged_decode_default.conf non-causal h192

Decode h192 previously fell back requesting:

```
8,192,64,false,false,false
```

even when Python passed `causal=True` (dispatch uses non-causal for
`max_seqlen_q==1`). Added non-causal 192 lines to `paged_decode_default.conf`:

```
8,192,16,false,false,false
8,192,32,false,false,false
8,192,64,false,false,false
```

Full FA2 rebuild (`rm -rf build/temp`, `MAX_JOBS=16`) produces
`libattn_kernels_xe_2.cfg6.so`. Decode A/B vs baseline artifact follows in
summary / CSVs after rebuild.

## Service

`inference/vllm-xpu` DaemonSet was paused
(`nodeSelector: hal.local/vllm-xpu-paused=true`) for GPU access during A/B.
Restored after benches (see final service status below).
