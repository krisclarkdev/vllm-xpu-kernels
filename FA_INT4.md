# FA2 int4-KV (greenfield) — local fork notes

Pinned upstream clone SHA at clone time: see `PINNED_SHA.txt`.

## Status (WS1 Cute/DPAS)

| Piece | Status |
|-------|--------|
| `csrc/xpu/attn/int4_kv_dequant.hpp` | **Landed** — nibble unpack + zp steganography matching Triton |
| `DecodeFwdMainloop::Int4KV` | **Landed** — `can_implement=true`; per-token-head dequant FE → ElementQ DPAS fragments |
| Host dispatch (`paged_decode_xe2.cpp`) | **Cute hot path** for uint8 KV + per-token-head scales (hd=128, block%64) |
| M1 SYCL `paged_decode_int4.hpp` | **Rollback only** — `VLLM_XPU_FA_INT4_M1_SYCL=1` |
| `flash_api.cpp` | Byte KV allowed; decode-only; splits=1 for int4 |
| `FA_INT4_AVAILABLE` in `flash_attn_interface.py` | **True** (wheel links int4 decode; still needs env) |
| M1 target (kept) | decode, hd=128, block=64, xe_2 |
| Write path | Reuse Triton `reshape_and_cache_int4` (vLLM) |

## Build

```bash
# oneAPI SYCL env matching trellis-serve
source /opt/intel/oneapi/setvars.sh   # or image env
cd /home/herdr/work/coder/vllm-xpu-kernels
# follow Dockerfile.xpu / build_script for wheel
uv pip install --no-build-isolation -e .
# or wheel:
# python -m build --wheel  (with oneAPI + torch XPU env)
```

Host dequant smoke (no GPU):

```bash
g++ -std=c++17 -I. -o /tmp/test_int4_dequant tests/attn/test_int4_kv_dequant_host.cpp
/tmp/test_int4_dequant
```

## Runtime

| Env | Effect |
|-----|--------|
| `VLLM_XPU_FA_INT4=1` | vLLM routes int4 FA (Python) |
| `VLLM_XPU_FA_INT4_M1_SYCL=1` | Force M1 scalar SYCL decode (oracle / A/B) |
| (default kernels) | Cute/DPAS Int4KV dequant FE |

`FA_INT4_AVAILABLE = True` when the wheel links int4 decode.
vLLM still needs `VLLM_XPU_FA_INT4=1`. Oracle A/B on B70 before prod.

## Test image

`192.168.1.228:5000/hal/trellis-serve:fa2-int4-m1f` — prior M1 image; rebuild wheel for Cute path.
Notes: `DESIGN-fa2-impl.md`, `DIAGNOSE-WS1.md`.
