#!/usr/bin/env bash
# Measure the native engine against llama.cpp through LLMario's own benchmark harness
# (docs/BENCHMARK_PLAN.md rules: cold prefix cache, temperature 0, quiet machine).
#
# Usage: LLMARIO_HOME=<home> scripts/engine/bench_m1.sh <model-id> [out-dir] [extra bench args]
#   The home's registry must contain <model-id> (a GGUF) and its config must enable the native
#   backend ([backends.native] enabled = true, engine_path = ...). Needs the release binaries
#   (cargo build --release -p llmario -p llmario-engine; $LLMARIO_BIN overrides the path)
#   and llama-server on PATH.
#
# Three targets, concurrency 1 (the native engine serves one request at a time in M1):
#   native        — LLMario's own engine
#   llamacpp-cpu  — llama-server with gpu_layers = 0 (the CPU-to-CPU comparison), via a copy of
#                   the home whose config pins gpu_layers
#   llamacpp-gpu  — llama-server with its default Metal/CUDA offload (the ceiling M2 aims at)
set -euo pipefail
MODEL="${1:?model id}"
OUT="${2:-benchmarks/results/$(date +%Y-%m-%d)-engine-M1-$(uname -m)}"
shift $(( $# >= 2 ? 2 : $# ))
TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
BIN="${LLMARIO_BIN:-$TARGET_DIR/release/llmario}"
[ -x "$BIN" ] || { echo "missing $BIN (cargo build --release -p llmario)"; exit 1; }
: "${LLMARIO_HOME:?set LLMARIO_HOME}"
mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"

echo "== native"
"$BIN" bench -m "$MODEL" --concurrency 1 --runs 3 --out "$OUT_ABS" --label "native-$MODEL" \
  --backend native "$@"

CPU_HOME="$(mktemp -d "${LLMARIO_HOME}/bench-cpu-home.XXXX")"
trap 'rm -rf "$CPU_HOME"' EXIT
cp "$LLMARIO_HOME/registry.toml" "$CPU_HOME/"
printf '[backends.llamacpp]\ngpu_layers = 0\n' > "$CPU_HOME/config.toml"
echo "== llama.cpp, CPU only (gpu_layers = 0)"
LLMARIO_HOME="$CPU_HOME" "$BIN" bench -m "$MODEL" --concurrency 1 --runs 3 --out "$OUT_ABS" \
  --label "llamacpp-cpu-$MODEL" --backend llamacpp "$@"

echo "== llama.cpp, default GPU offload"
"$BIN" bench -m "$MODEL" --concurrency 1 --runs 3 --out "$OUT_ABS" --label "llamacpp-gpu-$MODEL" \
  --backend llamacpp "$@"

echo "reports in $OUT_ABS"
ls -1 "$OUT_ABS"
