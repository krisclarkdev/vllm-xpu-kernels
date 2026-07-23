#!/usr/bin/env bash
# Self-contained B70 attention A/B + #6 pipeline. MAX_JOBS=16 by default.
set -euo pipefail
ROOT=/work/vllm-xpu-kernels
cd "$ROOT"
# Prefer an already-configured oneAPI env. Avoid sourcing setvars in
# docker-exec -d (it can hang when stdout is redirected).
if ! command -v icpx >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
fi
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:${PATH:-/usr/bin}"

export PYTHONPATH="$ROOT"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.1.8.2+localtest}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_TYPE=Release
export FA2_KERNELS_ENABLED=ON
export BASIC_KERNELS_ENABLED=OFF
export MOE_KERNELS_ENABLED=OFF
export GDN_KERNELS_ENABLED=OFF
export MQA_LOGITS_KERNELS_ENABLED=OFF
export XPU_SPECIFIC_KERNELS_ENABLED=OFF
export XPUMEM_ALLOCATOR_ENABLED=OFF
export VLLM_XPU_ENABLE_XE2=ON
export VLLM_XPU_ENABLE_XE_DEFAULT=OFF
export VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf
export VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf
export VLLM_XPU_XE2_AOT_DEVICES=bmg
export VLLM_XPU_AOT_DEVICES=
export FETCHCONTENT_BASE_DIR=/tmp/vllm-xpu-kernels-deps

ART=ab_artifacts
HDR=csrc/xpu/attn/xe_2/collective/chunk_prefill_mainloop.hpp
mkdir -p "$ART"
LOG="$ART/pipeline.log"
exec >>"$LOG" 2>&1

echo "PIPELINE_START $(date -Is) MAX_JOBS=$MAX_JOBS"

set_knobs() {
  local pd=$1 th=$2 win=$3 clear=$4
  python - "$HDR" "$pd" "$th" "$win" "$clear" <<'PY'
import pathlib, re, sys
path, pd, th, win, clear = sys.argv[1:6]
text = pathlib.Path(path).read_text()
def sub_def(name, val, s):
    return re.sub(
        rf"(#ifndef {name}\n#define {name} )[^\n]+",
        rf"\g<1>{val}",
        s,
        count=1,
    )
text = sub_def("FMHA_PREFETCH_D", pd, text)
text = sub_def("FMHA_LARGE_HEAD_ND_THRESHOLD", th, text)
text = sub_def("FMHA_ENABLE_PREFETCH_WINDOW", win, text)
text = sub_def("FMHA_ENABLE_CLEAR_PEEL", clear, text)
pathlib.Path(path).write_text(text)
print(f"knobs PREFETCH_D={pd} THRESH={th} WINDOW={win} CLEAR={clear}")
PY
}

rebuild_current() {
  local name=$1
  echo "REBUILD_START $name $(date -Is)"
  touch "$HDR"
  if [[ -f build/temp/build.ninja ]]; then
    cmake --build build/temp -j"${MAX_JOBS}" --target attn_kernels_xe_2 _vllm_fa2_C
    find build/temp -name libattn_kernels_xe_2.so -exec cp -f {} vllm_xpu_kernels/ \;
    if [[ -f build/temp/vllm_xpu_kernels/_vllm_fa2_C.abi3.so ]]; then
      cp -f build/temp/vllm_xpu_kernels/_vllm_fa2_C.abi3.so vllm_xpu_kernels/
    fi
  else
    pip install --no-build-isolation -e .
  fi
  cp -f vllm_xpu_kernels/libattn_kernels_xe_2.so "$ART/libattn_kernels_xe_2.${name}.so"
  cp -f vllm_xpu_kernels/_vllm_fa2_C.abi3.so "$ART/_vllm_fa2_C.${name}.abi3.so"
  echo "REBUILD_DONE $name $(date -Is) size=$(stat -c%s "$ART/libattn_kernels_xe_2.${name}.so")"
}

install_libs() {
  cp -f "$1" vllm_xpu_kernels/libattn_kernels_xe_2.so
  cp -f "$2" vllm_xpu_kernels/_vllm_fa2_C.abi3.so
}

run_bench_prefill() {
  local label=$1
  python tools/ab_attn_bench.py --label "$label" \
    --heads 128,192 --seqlens 4096,8192 --batch 2 \
    --skip-decode --warmup 5 --iters 15 \
    >"$ART/bench_${label}.csv" 2>"$ART/bench_${label}.err"
  if ! grep -qE ',[0-9]+\.[0-9]+$' "$ART/bench_${label}.csv"; then
    echo "BENCH_FAIL $label"
    cat "$ART/bench_${label}.csv" || true
    cat "$ART/bench_${label}.err" || true
    exit 1
  fi
  echo "BENCH_OK $label"
}

compare_csv() {
  local base_l=$1 new_l=$2
  python - "$base_l" "$new_l" <<'PY'
import math, sys
from pathlib import Path

def load(label):
    rows = {}
    for line in Path(f"ab_artifacts/bench_{label}.csv").read_text().splitlines():
        if not line or line.startswith(("LABEL", "DEVICE", "BATCH", "DECODE", "head,", "op,")):
            continue
        if ",ERROR:" in line:
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        try:
            rows[(parts[0], int(parts[1]), int(parts[2]))] = float(parts[-1])
        except ValueError:
            continue
    return rows

base_l, new_l = sys.argv[1], sys.argv[2]
base, new = load(base_l), load(new_l)
keys = sorted(set(base) & set(new))
print(f"COMPARE {base_l} vs {new_l}")
print("op,head,len,base_ms,new_ms,delta_pct")
prefs, big = [], []
for k in keys:
    b, n = base[k], new[k]
    d = (b - n) / b * 100 if b else 0.0
    print(f"{k[0]},{k[1]},{k[2]},{b:.4f},{n:.4f},{d:+.2f}")
    if k[0] == "prefill":
        prefs.append((b, n))
        if k[1] == 192:
            big.append((b, n))

def geo(pairs):
    rs = [b / n for b, n in pairs if n > 0]
    return math.prod(rs) ** (1 / len(rs)) if rs else float("nan")

print(f"prefill_geomean_speedup={geo(prefs):.4f}x")
print(f"prefill_h192_geomean_speedup={geo(big):.4f}x")
PY
}

ab_pair() {
  local base_name=$1 new_name=$2
  install_libs "$ART/libattn_kernels_xe_2.${base_name}.so" "$ART/_vllm_fa2_C.${base_name}.abi3.so"
  run_bench_prefill "${base_name}_vs_${new_name}_A"
  # reuse label scheme expected by compare: write aliases
  cp -f "$ART/bench_${base_name}_vs_${new_name}_A.csv" "$ART/bench_${base_name}.csv"
  install_libs "$ART/libattn_kernels_xe_2.${new_name}.so" "$ART/_vllm_fa2_C.${new_name}.abi3.so"
  run_bench_prefill "${new_name}"
  compare_csv "$base_name" "$new_name" | tee -a "$ART/summary.txt"
}

# --- Phase 2: #4 isolate sweep (skip rebuilding 'new'; already have it) ---
: >"$ART/summary.txt"
echo "=== #4 SWEEP BUILDS ===" | tee -a "$ART/summary.txt"
set_knobs 2 4 0 1
rebuild_current clear_only
set_knobs 2 4 1 0
rebuild_current prefetch_only
set_knobs 1 4 1 1
rebuild_current pref1
set_knobs 3 4 1 1
rebuild_current pref3
set_knobs 4 4 1 1
rebuild_current pref4
set_knobs 2 4 1 1
echo "SWEEP_BUILDS_DONE $(date -Is)" | tee -a "$ART/summary.txt"

echo "=== #4 SWEEP BENCH ===" | tee -a "$ART/summary.txt"
ab_pair base new
ab_pair base clear_only
ab_pair base prefetch_only
ab_pair base pref1
ab_pair base pref3
ab_pair base pref4
echo "SWEEP_BENCH_DONE $(date -Is)" | tee -a "$ART/summary.txt"

# --- Phase 3: #6 rebuild with extended paged_decode_default.conf ---
echo "=== #6 CFG REBUILD ===" | tee -a "$ART/summary.txt"
set_knobs 2 4 1 1
echo "CFG6_BUILD_START $(date -Is)"
# Force regenerate decode templates from updated conf
rm -rf build/temp
pip install --no-build-isolation -e .
cp -f vllm_xpu_kernels/libattn_kernels_xe_2.so "$ART/libattn_kernels_xe_2.cfg6.so"
cp -f vllm_xpu_kernels/_vllm_fa2_C.abi3.so "$ART/_vllm_fa2_C.cfg6.abi3.so"
echo "CFG6_BUILD_DONE $(date -Is)" | tee -a "$ART/summary.txt"

install_libs "$ART/libattn_kernels_xe_2.base.so" "$ART/_vllm_fa2_C.base.abi3.so"
python tools/ab_attn_bench.py --label decode_base --heads 128,192 --skip-prefill \
  --kv-lens 4096,8192 --warmup 10 --iters 40 \
  >"$ART/bench_decode_base.csv" 2>"$ART/bench_decode_base.err" || true

install_libs "$ART/libattn_kernels_xe_2.cfg6.so" "$ART/_vllm_fa2_C.cfg6.abi3.so"
python tools/ab_attn_bench.py --label decode_cfg6 --heads 128,192 --skip-prefill \
  --kv-lens 4096,8192 --warmup 10 --iters 40 \
  >"$ART/bench_decode_cfg6.csv" 2>"$ART/bench_decode_cfg6.err"

echo "DECODE_COMPARE" | tee -a "$ART/summary.txt"
python - <<'PY' | tee -a ab_artifacts/summary.txt
from pathlib import Path

def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith(("LABEL", "DEVICE", "BATCH", "DECODE", "head,", "op,")):
            continue
        parts = line.split(",")
        key = f"{parts[0]},{parts[1]},{parts[2]}"
        rows[key] = parts[-1]
    return rows

b = load("ab_artifacts/bench_decode_base.csv")
n = load("ab_artifacts/bench_decode_cfg6.csv")
print("shape,base,cfg6")
for k in sorted(set(b) | set(n)):
    print(f"{k},{b.get(k,'MISSING')},{n.get(k,'MISSING')}")
PY

# Leave tree at default knobs (full #4) for cleanliness; decision recorded in report.
set_knobs 2 4 1 1
echo "ALL_DONE $(date -Is)" | tee -a "$ART/summary.txt"
