#!/usr/bin/env bash
# Phase 1 baseline (docs/PHASES_16GB_AND_SPEED.md): benchmark every reference model
# (benchmarks/reference-16gb.toml) that is installed, in LLMario Beta and optionally in
# production LLMario, with the same suite and settings. Models that are refused (for example
# "does not fit") are recorded too: on 16 GB hardware that is part of the baseline.
#
#   scripts/bench-baseline.sh                                  # beta only
#   scripts/bench-baseline.sh --production "$(command -v llmario)"
#
# Options:
#   --beta PATH        llmario-beta binary (default: on PATH, else target/release/llmario-beta)
#   --production PATH  production llmario binary to benchmark the same models with
#   --models a,b       model ids instead of the reference set
#   --concurrency L    comma-separated levels (default 1)
#   --runs N           runs per prompt (default 3)
#   --out DIR          default benchmarks/results/<date>-<cpu>-<ram>gb
#   --allow-busy       run even when the machine is busy (reports are marked busy)
#
# Protocol: AC power and a quiet machine (1-minute load average at most half the CPU cores).
# Nothing written here contains file paths from your home directory.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BETA="" PROD="" MODELS="" CONC=1 RUNS=3 OUT="" ALLOW_BUSY=0
while (( $# )); do
  case "$1" in
    --beta) BETA="$2"; shift ;;
    --production) PROD="$2"; shift ;;
    --models) MODELS="$2"; shift ;;
    --concurrency) CONC="$2"; shift ;;
    --runs) RUNS="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --allow-busy) ALLOW_BUSY=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
# Keep home-directory paths out of anything written (results may be committed).
scrub() { sed -e "s#$HOME#~#g"; }

if [[ -z "$BETA" ]]; then
  BETA="$(command -v llmario-beta || true)"
  [[ -n "$BETA" ]] || BETA="$ROOT/target/release/llmario-beta"
fi
[[ -x "$BETA" ]] || die "llmario-beta not found (build it: cargo build --release -p llmario-beta, or pass --beta)"
[[ -z "$PROD" || -x "$PROD" ]] || die "--production $PROD is not executable"
command -v python3 >/dev/null || die "python3 is required"

say "Checking the protocol"
if [[ "$(uname -s)" == "Darwin" ]]; then
  pmset -g batt | head -1 | grep -q "AC Power" || die "not on AC power; plug in and retry"
fi
read -r LOAD CORES <<<"$(python3 -c 'import os; print(f"{os.getloadavg()[0]:.1f}", os.cpu_count())')"
if python3 -c "import sys; l, c = float(sys.argv[1]), int(sys.argv[2]); sys.exit(0 if l > max(2.0, c * 0.5) else 1)" "$LOAD" "$CORES"; then
  (( ALLOW_BUSY )) || die "machine is busy (load average $LOAD on $CORES cores); close other work and retry, or pass --allow-busy"
  echo "  ! busy (load average $LOAD on $CORES cores); reports will be marked busy"
else
  echo "  ✓ quiet (load average $LOAD on $CORES cores), AC power"
fi

if [[ -z "$MODELS" ]]; then
  # Plain parsing (no tomllib): the stock macOS python3 is 3.9.
  MODELS="$(python3 -c '
import re, sys
text = open(sys.argv[1]).read()
block = text[text.index("models = ["):]
block = block[:block.index("]")]
print(",".join(re.findall(r"\"([^\"]+)\"", block)))' "$ROOT/benchmarks/reference-16gb.toml")"
fi
IFS=',' read -r -a IDS <<<"$MODELS"

ENV_JSON="$("$BETA" doctor --json)" || die "llmario-beta doctor failed"
if [[ -z "$OUT" ]]; then
  TAG="$(python3 -c '
import json, re, sys
h = json.loads(sys.stdin.read())["hardware"]
cpu = re.sub(r"[^a-z0-9]+", "-", h["cpu_brand"].lower()).strip("-")
print(cpu + "-" + str(round(h["total_memory_bytes"] / 2**30)) + "gb")' <<<"$ENV_JSON")"
  OUT="$ROOT/benchmarks/results/$(date +%Y-%m-%d)-$TAG"
fi
mkdir -p "$OUT"
# Hardware and engine versions only: engine paths and details can contain home paths.
python3 -c '
import json, sys
d = json.loads(sys.stdin.read())
keep = {k: d[k] for k in ("version", "hardware", "hardware_fingerprint", "profile")}
keep["backends"] = [{k: b.get(k) for k in ("kind", "available", "version", "tested_version")} for b in d["backends"]]
print(json.dumps(keep, indent=2))' <<<"$ENV_JSON" >"$OUT/environment.json"

installed() { "$1" model list --json 2>/dev/null | python3 -c 'import json, sys; print("\n".join(m["id"] for m in json.load(sys.stdin)))'; }

SUMMARY="$OUT/SUMMARY.md"
{
  echo "# Baseline: $(python3 -c 'import json,sys; h=json.load(open(sys.argv[1]))["hardware"]; print(h["cpu_brand"], "·", round(h["total_memory_bytes"]/2**30), "GB")' "$OUT/environment.json")"
  echo
  echo "Date $(date -u +%Y-%m-%dT%H:%MZ) · load average $LOAD on $CORES cores at start · concurrency $CONC · $RUNS run(s) per prompt"
  echo
  echo "| Edition | Model | Result |"
  echo "|---|---|---|"
} >"$SUMMARY"

bench_all() { # edition-name binary [extra args]
  local name="$1" bin="$2"; shift 2
  local have; have="$(installed "$bin")"
  mkdir -p "$OUT/$name"
  for id in "${IDS[@]}"; do
    if ! grep -qx "$id" <<<"$have"; then
      echo "| $name | \`$id\` | not installed |" >>"$SUMMARY"; continue
    fi
    say "$name: $id"
    local log="$OUT/$name/.$id.log"
    if "$bin" bench -m "$id" --concurrency "$CONC" --runs "$RUNS" --out "$OUT/$name" "$@" >/dev/null 2>"$log"; then
      echo "| $name | \`$id\` | measured |" >>"$SUMMARY"
    else
      local why; why="$(grep -E "error|refus|fit" "$log" | tail -1 | scrub | tr '|' '/' | cut -c1-300)"
      echo "| $name | \`$id\` | **failed**: ${why:-see log} |" >>"$SUMMARY"
      echo "  ✗ ${why:-failed}"
    fi
    rm -f "$log"
  done
}

BUSY_ARGS=(); (( ALLOW_BUSY )) && BUSY_ARGS=(--allow-busy)
bench_all beta "$BETA" "${BUSY_ARGS[@]+"${BUSY_ARGS[@]}"}"
[[ -n "$PROD" ]] && bench_all production "$PROD"

for name in beta production; do
  if compgen -G "$OUT/$name/*.json" >/dev/null; then
    { echo; echo "## $name"; python3 "$ROOT/scripts/bench-table.py" "$OUT/$name" | sed "s#](\([^)]*\.md\))#]($name/\1)#g"; } >>"$SUMMARY"
  fi
done
say "Done: $SUMMARY"
cat "$SUMMARY"
