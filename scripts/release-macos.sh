#!/usr/bin/env bash
# Build a signed, notarized, stapled LLMario DMG that opens on any Mac without warnings.
#
#   scripts/release-macos.sh                  # universal (Apple Silicon + Intel), notarized
#   scripts/release-macos.sh --skip-notarize  # sign + DMG only (dry run; others will see warnings)
#
# One-time setup (see docs/RELEASING.md):
#   1. A "Developer ID Application" certificate in your login keychain (Apple Developer Program).
#   2. Notary credentials stored in the keychain (you type the password, this script never sees it):
#        xcrun notarytool store-credentials llmario-notary --apple-id YOU@EXAMPLE.COM --team-id TEAMID
#
# Environment overrides:
#   APPLE_SIGNING_IDENTITY  certificate name (default: the first "Developer ID Application" found)
#   NOTARY_PROFILE          keychain profile name (default: llmario-notary)
#   TARGET                  universal-apple-darwin (default) | aarch64-apple-darwin | x86_64-apple-darwin
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAURI_DIR="$ROOT/apps/desktop/src-tauri"
PROFILE="${NOTARY_PROFILE:-llmario-notary}"
TARGET="${TARGET:-universal-apple-darwin}"
NOTARIZE=1
[[ "${1:-}" == "--skip-notarize" ]] && NOTARIZE=0

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "macOS only"
command -v cargo >/dev/null || die "Rust/cargo not found"
cargo tauri --version >/dev/null 2>&1 || die 'tauri-cli missing: cargo install tauri-cli --version "^2" --locked'

IDENTITY="${APPLE_SIGNING_IDENTITY:-$(security find-identity -v -p codesigning \
  | sed -n 's/.*"\(Developer ID Application: [^"]*\)"/\1/p' | head -1)}"
[[ -n "$IDENTITY" ]] || die 'no "Developer ID Application" certificate in your keychain (see docs/RELEASING.md)'
TEAM_ID="$(sed -n 's/.*(\([A-Z0-9]\{10\}\))$/\1/p' <<<"$IDENTITY")"

if (( NOTARIZE )); then
  xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1 || die "notary profile '$PROFILE' not found or invalid.
Create it once (you will be asked for an app-specific password from appleid.apple.com):
  xcrun notarytool store-credentials $PROFILE --apple-id YOUR_APPLE_ID --team-id ${TEAM_ID:-TEAMID}"
fi

VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$TAURI_DIR/tauri.conf.json")"
case "$TARGET" in
  universal-apple-darwin) ARCH=universal ;;
  aarch64-apple-darwin) ARCH=arm64 ;;
  x86_64-apple-darwin) ARCH=x86_64 ;;
  *) die "unsupported TARGET $TARGET" ;;
esac
for t in $([[ $TARGET == universal-apple-darwin ]] && echo aarch64-apple-darwin x86_64-apple-darwin || echo "$TARGET"); do
  rustup target list --installed | grep -qx "$t" || die "missing Rust target $t: rustup target add $t"
done

say "Building LLMario $VERSION ($ARCH), signing as: $IDENTITY"
say "(macOS may ask once for permission to use the signing key: choose Always Allow)"
(cd "$TAURI_DIR" && APPLE_SIGNING_IDENTITY="$IDENTITY" cargo tauri build --target "$TARGET" --bundles app)
APP="$ROOT/target/$TARGET/release/bundle/macos/LLMario.app"
[[ -d "$APP" ]] || die "bundle not found at $APP"

say "Verifying app signature"
codesign --verify --deep --strict --verbose=2 "$APP"
SIGINFO="$(codesign -d --verbose=4 "$APP" 2>&1)"
grep -q "Authority=Developer ID Application" <<<"$SIGINFO" || die "app is not signed with a Developer ID certificate"
grep -Eq "flags=.*runtime" <<<"$SIGINFO" || die "hardened runtime is not enabled (required for notarization)"
grep -q "^Timestamp=" <<<"$SIGINFO" || die "signature has no secure timestamp (required for notarization)"
lipo -archs "$APP/Contents/MacOS/llmario-desktop"

OUT="$ROOT/dist"
mkdir -p "$OUT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if (( NOTARIZE )); then
  say "Notarizing the app (usually 1–5 minutes)"
  ditto -c -k --keepParent "$APP" "$WORK/LLMario.zip"
  xcrun notarytool submit "$WORK/LLMario.zip" --keychain-profile "$PROFILE" --wait
  xcrun stapler staple "$APP"
fi

say "Creating DMG"
DMG="$OUT/LLMario-$VERSION-macos-$ARCH.dmg"
mkdir "$WORK/dmg"
ditto "$APP" "$WORK/dmg/LLMario.app"
ln -s /Applications "$WORK/dmg/Applications"
rm -f "$DMG"
hdiutil create -volname "LLMario $VERSION" -srcfolder "$WORK/dmg" -fs HFS+ -format UDZO -ov "$DMG" >/dev/null
codesign --sign "$IDENTITY" --timestamp "$DMG"

if (( NOTARIZE )); then
  say "Notarizing the DMG"
  xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
  xcrun stapler staple "$DMG"
  xcrun stapler validate "$DMG"
  say "Gatekeeper assessment"
  spctl --assess --type execute --verbose=2 "$APP"
  spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
fi

(cd "$OUT" && shasum -a 256 "$(basename "$DMG")" > "$(basename "$DMG").sha256")
say "Done: $DMG"
cat "$DMG.sha256"
(( NOTARIZE )) || echo "NOTE: not notarized; other Macs will block it. Re-run without --skip-notarize to share."
