#!/usr/bin/env bash
# Local CI for LLMario: runs the checks of .github/workflows/ci.yml and engine.yml on this Mac.
#
# Owner rule (2026-10-30): CI runs on the local machine, never on GitHub. GitHub stays the code
# host. ci.yml and engine.yml are disabled on GitHub and remain the job specification: change them
# together with this script. windows.yml still runs on GitHub (owner's choice).
#
# Usage: scripts/ci-local.sh [<commit>] [--only job,job] [--keep]
#
#   <commit>  commit to check (default HEAD), resolved to a full sha
#   --only    run only these jobs (a partial run is never recorded as a pass)
#   --keep    leave this run's containers and throwaway directory in place for debugging
#
# Jobs (workflow job / GitHub runner it stands in for):
#   linux-test     ci.yml test, ubuntu-latest     Docker (rust image, arm64)
#   macos-test     ci.yml test, macos-14          native, throwaway dir
#   macos-desktop  ci.yml desktop, macos-14       native, throwaway dir
#   licenses       ci.yml licenses, ubuntu-latest Docker (cargo-deny)
#   linux-engine   engine.yml serve-e2e, ubuntu   Docker
#   macos-engine   engine.yml serve-e2e, macos-14 native, throwaway dir
# The windows-latest legs of ci.yml (test, desktop) cannot be reproduced on a Mac. windows.yml still
# runs on GitHub, but it only builds the CLI and installer and smoke-tests them: it has no cargo
# test, clippy, fmt or node test, and it builds no crates/engine crate.
# Linux runs as arm64 here (ubuntu-latest is x86_64), so x86_64-only code, such as
# crates/engine/cpu/src/simd/avx2.rs, is not compiled or tested by any job.
#
# How it works
#   - Source is always `git archive <sha>`: piped into each Linux job's container, or extracted into
#     a throwaway directory for the macOS legs. The working tree is never built. Files are extracted
#     with fresh mtimes so cargo can never mistake changed sources for fresh ones; caches hold
#     dependencies only, as Swatinem/rust-cache does on GitHub.
#   - Toolchain: dtolnay/rust-toolchain@stable = the latest stable Rust. The runner pulls
#     rust:1-bookworm (latest stable), adds rustfmt, clippy and jq, and tags the result
#     llmario-ci-rust:<version>. The macOS legs use the same version through rustup
#     (`rustup toolchain install <version>` if missing; the default toolchain is not changed).
#   - Each Linux job is one long-lived container (`sleep infinity`) and each workflow `run:` step
#     is one `docker exec ... bash -eo pipefail -c`. No host ports are published. Caches are named
#     volumes: llmario-ci-cargo-home-cache (registry, git, cargo-deny binary and advisory DB),
#     llmario-ci-<job>-target-cache, llmario-ci-engine-home-cache (smoke model).
#   - The macOS legs run with HOME pointed at the throwaway dir, so tests never touch the real
#     ~/.llmario*; targets are cached in ~/Library/Caches/llmario-ci/target/<job>, the smoke model
#     in ~/Library/Caches/llmario-ci/engine-home.
#   - Every container carries the label llmario-ci.run=<runId>; the exit trap removes exactly
#     those and the run's throwaway directory (unless --keep). Nothing else is touched.
#   - At most LLMARIO_CI_PARALLEL (default 2) jobs run at once; containers build with
#     CARGO_BUILD_JOBS=LLMARIO_CI_JOBS (default 4), macOS legs with LLMARIO_CI_MAC_JOBS (default 6).
#   - Logs: <git common dir>/ci-local/<sha>.<job>.log. A full run where every job passes writes
#     <sha>.pass there (commit, UTC date, per-job result and test counts).
#   - This runs commits that are already in the local repository. It never fetches and runs pull
#     request code by itself (this is a public repository).
#
# Environment: LLMARIO_CI_PARALLEL, LLMARIO_CI_JOBS, LLMARIO_CI_MAC_JOBS, LLMARIO_CI_NO_PULL=1
# (do not pull a newer rust:1-bookworm), LLMARIO_CI_NODE (node binary for the desktop UI test).
#
# Needs: Docker Desktop running; the repository's drive mounted; rustup on the Mac; node; jq.
# Written for macOS's /bin/bash 3.2.
set -euo pipefail

ALL_JOBS=(linux-test macos-test linux-engine macos-engine macos-desktop licenses)
# cargo-deny version bundled by EmbarkStudios/cargo-deny-action (its Dockerfile, checked 2026-10-30).
DENY_VERSION="0.20.2"
BASE_IMAGE="rust:1-bookworm"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { echo "ci-local: $*" >&2; exit 2; }
fmt_dur() { local s=$1; printf '%dm%02ds' $((s / 60)) $((s % 60)); }

job_spec() {
  case $1 in
    linux-test) echo "ci.yml test (ubuntu-latest)" ;;
    macos-test) echo "ci.yml test (macos-14)" ;;
    macos-desktop) echo "ci.yml desktop (macos-14)" ;;
    licenses) echo "ci.yml licenses (ubuntu-latest)" ;;
    linux-engine) echo "engine.yml serve-e2e (ubuntu-latest)" ;;
    macos-engine) echo "engine.yml serve-e2e (macos-14)" ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------------------------------------
# Step helpers (run inside a job process)

banner() { printf '\n==> [%s] %s\n' "$(date -u +%H:%M:%S)" "$1"; }

# lstep <container> <script>: one workflow `run:` step inside the job's container. /tmp/ci.env
# stands in for $GITHUB_ENV / step outputs between steps.
lstep() {
  banner "$2"
  docker exec "$1" bash -eo pipefail -c "cd /src; if [ -f \"\$CI_ENV\" ]; then . \"\$CI_ENV\"; fi
$2"
}

# mstep <script>: one workflow `run:` step on the Mac, in the throwaway source dir.
mstep() {
  banner "$1"
  (cd "$SRC" && bash -eo pipefail -c "if [ -f \"\$CI_ENV\" ]; then . \"\$CI_ENV\"; fi
$1")
}

start_container() {
  local job=$1
  shift
  C="llmario-ci-${RUN_ID}-${job}"
  banner "container $C from $IMAGE (source: git archive $SHA)"
  docker run -d --name "$C" --label "$LABEL" \
    -v llmario-ci-cargo-home-cache:/cargo-home \
    -v "llmario-ci-${job}-target-cache:/target" \
    "$@" \
    -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/target -e CI_ENV=/tmp/ci.env \
    -e CARGO_BUILD_JOBS="$CI_JOBS" -e CARGO_TERM_COLOR=never -e CARGO_INCREMENTAL=0 -e CI=true \
    -e PATH=/cargo-home/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    "$IMAGE" sleep infinity >/dev/null
  docker exec "$C" mkdir -p /src
  git -C "$REPO" archive "$SHA" | docker exec -i "$C" tar -xm -C /src
  docker exec "$C" sh -c 'rustc --version; cargo --version'
}

setup_mac() {
  local job=$1
  WORK="$RUNDIR/$job"
  SRC="$WORK/src"
  mkdir -p "$SRC" "$WORK/home" "$CACHE/target/$job"
  banner "throwaway dir $SRC (source: git archive $SHA)"
  git -C "$REPO" archive "$SHA" | tar -xm -C "$SRC"
  export CARGO_HOME="$REAL_CARGO_HOME" RUSTUP_HOME="$REAL_RUSTUP_HOME"
  export PATH="$REAL_CARGO_HOME/bin:$PATH"
  export HOME="$WORK/home"
  export RUSTUP_TOOLCHAIN="$RUST_VERSION" CARGO_TARGET_DIR="$CACHE/target/$job"
  export CARGO_BUILD_JOBS="$MAC_JOBS" CARGO_TERM_COLOR=never CARGO_INCREMENTAL=0 CI=true
  export CI_ENV="$WORK/ci.env"
  rustc --version
  cargo --version
}

# ---------------------------------------------------------------------------------------------
# Jobs. Steps mirror the workflows' `run:` steps in order; `uses:` steps are noted.

ci_test_steps() { # $1 = lstep container or empty for mac
  local run=(mstep)
  if [ -n "${1:-}" ]; then run=(lstep "$1"); fi
  # git config --global core.autocrlf false: Windows-only concern; archives keep LF.
  # actions/checkout -> git archive; dtolnay/rust-toolchain (rustfmt, clippy) -> image/rustup;
  # Swatinem/rust-cache -> cache volumes / cache dir.
  "${run[@]}" 'cargo fmt --all -- --check'
  "${run[@]}" 'cargo clippy --all-targets -- -D warnings'
  "${run[@]}" 'cargo test'
}

job_linux-test() { start_container linux-test; ci_test_steps "$C"; }
job_macos-test() { setup_mac macos-test; ci_test_steps; }

job_macos-desktop() {
  setup_mac macos-desktop
  # actions/setup-node@v4 node-version 22 -> the Mac's node (LLMARIO_CI_NODE to override).
  banner "node: $("$NODE_BIN" --version) ($NODE_BIN; the workflow pins 22)"
  export NODE_BIN
  mstep '"$NODE_BIN" --test apps/desktop/ui/markdown.test.mjs apps/desktop/ui/memory-state.test.mjs'
  mstep 'cargo clippy -p llmario-desktop -- -D warnings'
  # if: runner.os == 'macOS'
  mstep 'cargo build --release -p llmario-desktop'
}

job_licenses() {
  start_container licenses
  # EmbarkStudios/cargo-deny-action@v2 runs its bundled cargo-deny with its default inputs
  # (log-level warn, manifest-path ./Cargo.toml, arguments --all-features).
  setenv "$C" DENY_VERSION "$DENY_VERSION"
  lstep "$C" 'if [ "$(cargo-deny --version 2>/dev/null)" != "cargo-deny $DENY_VERSION" ]; then
  cargo install --locked cargo-deny --version "$DENY_VERSION" --target-dir /target/tools
fi
cargo-deny --version'

  lstep "$C" 'cargo deny --log-level warn --manifest-path ./Cargo.toml --all-features check licenses advisories sources'
}

# setenv <container-or-empty> NAME VALUE: stands in for writing to $GITHUB_ENV.
setenv() {
  local line="export $2='$3'"
  if [ -n "$1" ]; then docker exec "$1" sh -c "echo \"\$0\" >> /tmp/ci.env" "$line"; else echo "$line" >>"$CI_ENV"; fi
}

ENGINE_REV_STEP=$(cat <<'EOF'
rev=$(awk -v id="\"$SMOKE_MODEL\"" '$1=="id" && $3==id {f=1} f && $1=="revision" {gsub(/"/,"",$3); print $3; exit}' crates/model_registry/catalog.toml)
test -n "$rev"
echo "rev=$rev"
mkdir -p "$ENGINE_HOME_ROOT/$SMOKE_MODEL-$rev"
echo "export LLMARIO_HOME='$ENGINE_HOME_ROOT/$SMOKE_MODEL-$rev'" >> "$CI_ENV"
EOF
)
# The workflow's exe is target/release/llmario-beta; the target dir is $CARGO_TARGET_DIR here.
ENGINE_PULL_STEP=$(cat <<'EOF'
echo "LLMARIO_HOME=$LLMARIO_HOME"
exe="$CARGO_TARGET_DIR/release/llmario-beta"
if ! "$exe" model list --json | jq -e --arg id "$SMOKE_MODEL" '.[] | select(.id == $id)' > /dev/null; then
  "$exe" model pull "$SMOKE_MODEL"
fi
gguf=$("$exe" model list --json | jq -r --arg id "$SMOKE_MODEL" '.[] | select(.id == $id) | .path')
test -f "$gguf"
echo "export LLMARIO_TEST_GGUF='$gguf'" >> "$CI_ENV"
EOF
)

engine_steps() { # $1 = container or empty for mac; $2 = model home root
  local run=(mstep)
  if [ -n "$1" ]; then run=(lstep "$1"); fi
  # job env: SMOKE_MODEL (read from engine.yml). LLMARIO_HOME is set per model revision below.
  # timeout-minutes: 60 is not enforced locally; concurrency: n/a.
  setenv "$1" SMOKE_MODEL "$SMOKE_MODEL"
  setenv "$1" ENGINE_HOME_ROOT "$2"
  "${run[@]}" 'cargo build --release --locked -p llmario-beta -p llmario-engine'
  # "Model revision pinned in the catalog". "Restore the test model" (actions/cache keyed by
  # os + model + revision) -> a persistent home per model and revision under the root.
  "${run[@]}" "$ENGINE_REV_STEP"
  # "Pull the test model (verified against the Hub's SHA-256)": `llmario model pull` verifies.
  "${run[@]}" "$ENGINE_PULL_STEP"
  "${run[@]}" 'cargo test --release --locked -p llmario-engine --test serve_e2e -- --nocapture'
}

job_linux-engine() { start_container linux-engine -v llmario-ci-engine-home-cache:/engine-home; engine_steps "$C" /engine-home; }
job_macos-engine() {
  setup_mac macos-engine
  mkdir -p "$CACHE/engine-home"
  engine_steps "" "$CACHE/engine-home"
}

# Internal entry point: one job in its own process, so `set -e` applies to every step.
if [ "${1:-}" = "__job" ]; then
  job=$2
  echo "ci-local job $job: $(job_spec "$job") at $SHA (run $RUN_ID)"
  "job_$job"
  echo
  echo "ci-local job $job: PASS"
  exit 0
fi

# ---------------------------------------------------------------------------------------------
# Main

COMMIT=HEAD
ONLY=""
KEEP=0
while [ $# -gt 0 ]; do
  case $1 in
    --only) [ $# -ge 2 ] || die "--only needs a job list"; ONLY=$2; shift 2 ;;
    --only=*) ONLY=${1#--only=}; shift ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) COMMIT=$1; shift ;;
  esac
done

# The repository is the one this script lives in, whatever the current directory is.
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
SHA=$(git -C "$REPO" rev-parse --verify --quiet "${COMMIT}^{commit}") || die "not a commit: $COMMIT"
OUT="$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)/ci-local"
mkdir -p "$OUT"

JOBS=()
if [ -n "$ONLY" ]; then
  IFS=',' read -r -a wanted <<<"$ONLY"
  for j in "${ALL_JOBS[@]}"; do
    for w in "${wanted[@]}"; do [ "$w" = "$j" ] && JOBS+=("$j"); done
  done
  for w in "${wanted[@]}"; do job_spec "$w" >/dev/null || die "unknown job $w (jobs: ${ALL_JOBS[*]})"; done
else
  JOBS=("${ALL_JOBS[@]}")
fi
[ ${#JOBS[@]} -gt 0 ] || die "no jobs selected"

RUN_ID="$(date -u +%Y%m%d%H%M%S)-$$"
LABEL="llmario-ci.run=$RUN_ID"
CACHE="$HOME/Library/Caches/llmario-ci"
RUNDIR="$CACHE/runs/$RUN_ID"
CI_JOBS=${LLMARIO_CI_JOBS:-4}
MAC_JOBS=${LLMARIO_CI_MAC_JOBS:-6}
PARALLEL=${LLMARIO_CI_PARALLEL:-2}
REAL_HOME=$HOME
REAL_CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}
REAL_RUSTUP_HOME=${RUSTUP_HOME:-$HOME/.rustup}
NODE_BIN=${LLMARIO_CI_NODE:-$(command -v node || true)}
mkdir -p "$RUNDIR"
cp "$0" "$RUNDIR/ci-local.sh"   # job processes run this copy, not a file that may change mid-run

PIDS=()
killtree() {
  local p
  for p in $(pgrep -P "$1" 2>/dev/null || true); do killtree "$p"; done
  kill -TERM "$1" 2>/dev/null || true
}
cleanup() {
  local rc=$? p ids nets
  trap - EXIT INT TERM
  for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -0 "$p" 2>/dev/null && killtree "$p"; done
  ids=$(docker ps -aq --filter "label=$LABEL" 2>/dev/null || true)
  nets=$(docker network ls -q --filter "label=$LABEL" 2>/dev/null || true)
  if [ "$KEEP" = 1 ]; then
    [ -n "$ids" ] && echo "ci-local: --keep: containers left: $(echo $ids)"
    echo "ci-local: --keep: throwaway dir left: $RUNDIR"
  else
    # shellcheck disable=SC2086 # ids are docker ids, split on purpose
    [ -z "$ids" ] || docker rm -f $ids >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    [ -z "$nets" ] || docker network rm $nets >/dev/null 2>&1 || true
    rm -rf "$RUNDIR"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

T0=$(date +%s)
echo "ci-local: $SHA ($(git -C "$REPO" log -1 --format=%s "$SHA"))"
echo "ci-local: run $RUN_ID, jobs: ${JOBS[*]}, logs in $OUT"
[ -z "$ONLY" ] && rm -f "$OUT/$SHA.pass"   # a full run replaces the commit's pass

needs_linux=0; needs_mac=0
for j in "${JOBS[@]}"; do case $j in macos-*) needs_mac=1 ;; *) needs_linux=1 ;; esac; done

# -- Toolchain (always resolved from the image so Linux and macOS legs use the same Rust)
PREP="$OUT/$SHA.prep.log"
: >"$PREP"
docker info >/dev/null 2>&1 || die "Docker is not running (start Docker Desktop)"
if [ "${LLMARIO_CI_NO_PULL:-0}" != 1 ] || ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  log "pulling $BASE_IMAGE (latest stable Rust, as dtolnay/rust-toolchain@stable)"
  docker pull -q "$BASE_IMAGE" >>"$PREP" 2>&1 || die "docker pull $BASE_IMAGE failed (see $PREP)"
fi
RUST_VERSION=$(docker run --rm --label "$LABEL" "$BASE_IMAGE" rustc --version | awk '{print $2}')
[ -n "$RUST_VERSION" ] || die "could not read the Rust version from $BASE_IMAGE"
IMAGE="llmario-ci-rust:$RUST_VERSION"
log "Rust $RUST_VERSION"
if [ "$needs_linux" = 1 ] && ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "building $IMAGE (rustfmt, clippy, jq)"
  docker build -q --label llmario-ci.image=1 -t "$IMAGE" - >>"$PREP" 2>&1 <<EOF || die "image build failed (see $PREP)"
FROM $BASE_IMAGE
RUN rustup component add rustfmt clippy \\
 && apt-get update && apt-get install -y --no-install-recommends jq \\
 && rm -rf /var/lib/apt/lists/*
EOF
fi
if [ "$needs_mac" = 1 ]; then
  command -v rustup >/dev/null || die "rustup not found (needed for the macOS legs)"
  if ! rustup run "$RUST_VERSION" cargo clippy --version >/dev/null 2>&1 \
    || ! rustup run "$RUST_VERSION" cargo fmt --version >/dev/null 2>&1; then
    log "installing Rust $RUST_VERSION on the Mac via rustup (default toolchain unchanged)"
    rustup toolchain install "$RUST_VERSION" --profile minimal --component clippy,rustfmt \
      --no-self-update >>"$PREP" 2>&1 || die "rustup toolchain install failed (see $PREP)"
  fi
  [ -n "$NODE_BIN" ] || die "node not found (needed for macos-desktop)"
fi
SMOKE_MODEL=$(git -C "$REPO" show "$SHA:.github/workflows/engine.yml" | awk '$1=="SMOKE_MODEL:" {print $2; exit}')
[ -n "$SMOKE_MODEL" ] || die "SMOKE_MODEL not found in engine.yml at $SHA"

export SHA REPO RUN_ID LABEL RUNDIR CACHE IMAGE RUST_VERSION CI_JOBS MAC_JOBS NODE_BIN OUT \
  REAL_HOME REAL_CARGO_HOME REAL_RUSTUP_HOME SMOKE_MODEL DENY_VERSION

# -- Run jobs, at most $PARALLEL at once
queue=("${JOBS[@]}")
running=()
start_job() {
  local j=$1
  date +%s >"$RUNDIR/$j.start"
  log "start $j: $(job_spec "$j")"
  (
    rc=0
    bash "$RUNDIR/ci-local.sh" __job "$j" >"$OUT/$SHA.$j.log" 2>&1 || rc=$?
    date +%s >"$RUNDIR/$j.end"
    echo "$rc" >"$RUNDIR/$j.rc"
  ) &
  PIDS+=($!)
  running+=("$j:$!")
}
while [ ${#queue[@]} -gt 0 ] || [ ${#running[@]} -gt 0 ]; do
  while [ ${#queue[@]} -gt 0 ] && [ ${#running[@]} -lt "$PARALLEL" ]; do
    start_job "${queue[0]}"
    queue=("${queue[@]:1}")
  done
  still=()
  for r in ${running[@]+"${running[@]}"}; do
    j=${r%%:*}; p=${r##*:}
    if kill -0 "$p" 2>/dev/null; then
      still+=("$r")
    else
      wait "$p" 2>/dev/null || true
      rc=$(cat "$RUNDIR/$j.rc" 2>/dev/null || echo 1)
      d=$(($(cat "$RUNDIR/$j.end" 2>/dev/null || date +%s) - $(cat "$RUNDIR/$j.start")))
      echo "$d" >"$RUNDIR/$j.dur"
      if [ "$rc" = 0 ]; then log "pass  $j ($(fmt_dur "$d"))"; else log "FAIL  $j ($(fmt_dur "$d")), log: $OUT/$SHA.$j.log"; fi
    fi
  done
  running=(${still[@]+"${still[@]}"})
  [ ${#running[@]} -eq 0 ] && [ ${#queue[@]} -eq 0 ] && break
  sleep 5
done

# -- Summary
counts() {
  awk '
    /^test result:/ { for (i = 1; i < NF; i++) {
        if ($(i+1) ~ /^passed;/) p += $i; if ($(i+1) ~ /^failed;/) f += $i; if ($(i+1) ~ /^ignored;/) ig += $i } t++ }
    /^(# |ℹ )pass [0-9]+$/ { np += $NF; n = 1 }
    /^(# |ℹ )fail [0-9]+$/ { nf += $NF }
    END {
      s = ""
      if (t) s = sprintf("cargo test: %d passed, %d failed, %d ignored (%d suites)", p, f, ig, t)
      if (n) s = s (s ? "; " : "") sprintf("node test: %d passed, %d failed", np, nf)
      print s
    }' "$1"
}
failed=0
lines=()
printf '\n%-14s %-38s %-6s %-8s %s\n' JOB WORKFLOW RESULT TIME DETAIL
for j in "${JOBS[@]}"; do
  rc=$(cat "$RUNDIR/$j.rc" 2>/dev/null || echo 1)
  d=$(cat "$RUNDIR/$j.dur" 2>/dev/null || echo 0)
  if [ "$rc" = 0 ]; then res=pass; else res=FAIL; failed=1; fi
  detail=$(counts "$OUT/$SHA.$j.log")
  if [ "$j" = licenses ]; then
    detail="cargo-deny $DENY_VERSION: $(grep -E '^advisories .*licenses .*sources ' "$OUT/$SHA.$j.log" | tail -1 || true)"
    n=$(grep -c '^error\[' "$OUT/$SHA.$j.log" || true)
    [ "$n" = 0 ] || detail="$detail ($n errors)"
  fi
  case $j in linux-*|licenses) detail="${detail:+$detail; }linux/arm64 container" ;; esac
  printf '%-14s %-38s %-6s %-8s %s\n' "$j" "$(job_spec "$j")" "$res" "$(fmt_dur "$d")" "$detail"
  lines+=("$j	$res	$(fmt_dur "$d")	$detail")
done
NOTE="not reproducible on a Mac (windows.yml still runs on GitHub)"
printf '%-14s %-38s %-6s %-8s %s\n' - "ci.yml test (windows-latest)" "-" "-" "$NOTE"
printf '%-14s %-38s %-6s %-8s %s\n' - "ci.yml desktop (windows-latest)" "-" "-" "$NOTE"

TOTAL=$(($(date +%s) - T0))
echo
if [ "$failed" = 0 ]; then
  echo "ci-local: PASS $SHA in $(fmt_dur "$TOTAL")"
  if [ -n "$ONLY" ]; then
    echo "ci-local: partial run (--only), not recorded as a pass"
  else
    {
      echo "commit $SHA"
      echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "runner scripts/ci-local.sh run $RUN_ID, $(fmt_dur "$TOTAL")"
      echo "rust $RUST_VERSION; node $("$NODE_BIN" --version 2>/dev/null || echo n/a); cargo-deny $DENY_VERSION"
      for l in "${lines[@]}"; do echo "$l"; done
      echo "windows-latest legs (ci.yml test, desktop): $NOTE"
    } >"$OUT/$SHA.pass"
    echo "ci-local: recorded $OUT/$SHA.pass"
  fi
  exit 0
fi
echo "ci-local: FAIL $SHA in $(fmt_dur "$TOTAL") (logs: $OUT/$SHA.<job>.log)"
exit 1
