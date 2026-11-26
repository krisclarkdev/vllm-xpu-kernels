# REPORT-WS1 — Cute/DPAS Int4KV paged_decode

Date: 2026-11-26  
Branch: `fa2-int4-xpu`  
Tree: `/home/herdr/work/coder/vllm-xpu-kernels`

## What landed

- Enabled `DecodeFwdMainloop::Int4KV` (`can_implement` + per-token-head dequant FE into ElementQ DPAS fragments).
- Host hot path: uint8 KV + per-token-head scales → Cute `decode_int4_dispatch` (q8/q16 × h128 × p64 only).
- M1 scalar SYCL retired from hot path; restore with `VLLM_XPU_FA_INT4_M1_SYCL=1`.
- `num_kv_splits` forced to 1 for int4 (unchanged; design risk mitigation).
- `FA_INT4.md` updated.

## Scope locks

- hd=128, packed_hs=64, block_size % 64 == 0, decode-only, xe_2
- No WS2/WS3 (Hadamard / fused write)

## Tester queue — wheel build

```bash
source /opt/intel/oneapi/setvars.sh
cd /home/herdr/work/coder/vllm-xpu-kernels
# match image torch XPU + uv env per AGENTS.md
uv venv --python 3.12   # if needed
uv pip install --no-build-isolation -e .
# or produce wheel into dist/ for trellis-serve overlay
```

A/B: default Cute path vs `VLLM_XPU_FA_INT4_M1_SYCL=1`.  
Oracle A + ctx=8k GB/s vs M1 ~1.5 / fp16 ~445 (exit: ≫ M1, stretch ≥20% fp16).

## Push

Remote fork only: `fork` → `krisclarkdev/vllm-xpu-kernels` (never `origin` / vllm-project).
