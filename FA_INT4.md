# FA2 int4-KV (greenfield) — local fork notes

Pinned upstream clone SHA at clone time: see `PINNED_SHA.txt`.

## Status (M0 / M1)

| Piece | Status |
|-------|--------|
| `csrc/xpu/attn/int4_kv_dequant.hpp` | **Landed** — nibble unpack + zp steganography matching Triton |
| `csrc/xpu/attn/xe_2/paged_decode_int4.hpp` | **Landed** — SYCL decode, in-loop dequant, hd=128 / block%64 |
| `paged_decode_int4_xe2.cpp` + host detect | **Landed** — uint8 KV + per-token-head scales → int4 path |
| `flash_api.cpp` | Byte KV allowed for int4; decode-only; splits=1 |
| `DecodeFwdMainloop::Int4KV` | Scaffold only (`can_implement=false`); Cute follow-on |
| `FA_INT4_AVAILABLE` in `flash_attn_interface.py` | **True** (wheel links int4 decode; still needs env) |
| M1 target | decode, hd=128, block=64, xe_2 |
| Write path | Reuse Triton `reshape_and_cache_int4` (vLLM) |

## Build

```bash
# oneAPI SYCL env matching trellis-serve
source /opt/intel/oneapi/setvars.sh   # or image env
cd /home/herdr/work/coder/vllm-xpu-kernels
# follow Dockerfile.xpu / build_script for wheel
```

Host dequant smoke (no GPU):

```bash
g++ -std=c++17 -I. -o /tmp/test_int4_dequant tests/attn/test_int4_kv_dequant_host.cpp
/tmp/test_int4_dequant
```

`FA_INT4_AVAILABLE = True` in the M1 wheel (`dist/vllm_xpu_kernels-0.1.0+fa2int4m1-*.whl`).
vLLM still needs `VLLM_XPU_FA_INT4=1`. Oracle A/B on B70 before prod.
