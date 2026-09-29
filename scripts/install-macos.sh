#!/usr/bin/env bash
# Build and install LLMario from source on macOS:
#   - the `llmario` command-line tool (into ~/.cargo/bin via `cargo install`)
#   - LLMario.app (into /Applications, or ~/Applications if /Applications is not writable)
#
#   git clone https://github.com/mannysinghx/LLMario.git && cd LLMario && ./scripts/install-macos.sh
#
# Options: --cli-only (skip the app) · --check (only report prerequisites)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI_ONLY=0
CHECK=0
for a in "$@"; do
  case "$a" in
    --cli-only) CLI_ONLY=1 ;;
    --check) CHECK=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
miss() { printf '  \033[31m✗\033[0m %s\n' "$*"; }

[[ "$(uname -s)" == "Darwin" ]] || { echo "This installer is for macOS. On Linux: cargo install --path crates/cli --locked"; exit 1; }

say "Checking prerequisites"
missing=0
if xcode-select -p >/dev/null 2>&1; then ok "Xcode Command Line Tools"; else miss "Xcode Command Line Tools — run: xcode-select --install"; missing=1; fi
if command -v cargo >/dev/null 2>&1; then ok "Rust $(rustc --version | cut -d' ' -f2)"; else miss "Rust — install from https://rustup.rs (then open a new terminal)"; missing=1; fi
if (( ! CLI_ONLY )); then
  if cargo tauri --version >/dev/null 2>&1; then ok "Tauri CLI"; else echo "  • Tauri CLI not installed yet (will be installed: cargo install tauri-cli, a few minutes)"; fi
fi
engines=0
if command -v llama-server >/dev/null 2>&1; then ok "llama.cpp engine"; engines=1; else echo "  • llama.cpp engine not found — install with: brew install llama.cpp"; fi
if command -v mlx_lm.server >/dev/null 2>&1 || [[ -x "$HOME/.llmario/venvs/mlx/bin/python" ]]; then ok "MLX-LM engine"; engines=1
elif [[ "$(uname -m)" == "arm64" ]]; then echo "  • MLX-LM engine not found (Apple Silicon, optional, often fastest) — install with: pip3 install mlx-lm"; fi
(( engines )) || echo "  ! LLMario needs at least one engine to run models. Install one of the above (you can do it after this script)."
(( missing )) && { echo; echo "Install the missing prerequisites above, then run this script again."; exit 1; }
(( CHECK )) && exit 0

say "Installing the llmario command (cargo install)"
cargo install --path "$ROOT/crates/cli" --locked --force
ok "installed $(command -v llmario || echo "$HOME/.cargo/bin/llmario")"

if (( ! CLI_ONLY )); then
  if ! cargo tauri --version >/dev/null 2>&1; then
    say "Installing the Tauri CLI (one time)"
    cargo install tauri-cli --version "^2" --locked
  fi
  if pgrep -f "LLMario.app/Contents/MacOS/llmario-desktop" >/dev/null; then
    echo "LLMario is running. Quit it (⌘Q), then run this script again."; exit 1
  fi
  say "Building LLMario.app"
  (cd "$ROOT/apps/desktop/src-tauri" && cargo tauri build --bundles app)
  APP="$ROOT/target/release/bundle/macos/LLMario.app"
  DEST=/Applications
  [[ -w "$DEST" ]] || { DEST="$HOME/Applications"; mkdir -p "$DEST"; }
  rm -rf "$DEST/LLMario.app"
  ditto "$APP" "$DEST/LLMario.app"
  codesign --verify --deep --strict "$DEST/LLMario.app"
  ok "installed $DEST/LLMario.app"
fi

say "Done"
echo "  • Open LLMario from Launchpad or Spotlight, or run: llmario doctor"
echo "  • In the app: Models → Library → Download a model marked 'recommended', then start chatting."
(( engines )) || echo "  • Remember to install an engine first: brew install llama.cpp   (or: pip3 install mlx-lm)"
