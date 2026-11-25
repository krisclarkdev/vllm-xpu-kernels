# FA2 int4-KV (greenfield) — local fork notes

Pinned upstream clone SHA at clone time: see `PINNED_SHA.txt`.

## Status (M0 / M1)

| Piece | Status |
|-------|--------|
| `csrc/xpu/attn/int4_kv_dequant.hpp` | **Landed** — nibble unpack + zp steganography matching Triton |
| `FA_INT4_AVAILABLE` in `flash_attn_interface.py` | **False** until xe_2 paged_decode int4 policy ships |
| M1 target | decode, hd=128, block=64, xe_2 |
| Write path | Reuse Triton `reshape_and_cache_int4` (vLLM) |

## Build

```bash
# oneAPI SYCL env matching trellis-serve
source /opt/intel/oneapi/setvars.sh   # or image env
cd /home/herdr/work/coder/vllm-xpu-kernels
# follow Dockerfile.xpu / build_script for wheel
```

Do **not** set `FA_INT4_AVAILABLE = True` until an int4 paged_decode policy is linked and Oracle A/B tests pass.
