# Releasing the macOS app

A shareable LLMario build is **signed** with a Developer ID certificate, uses the **hardened
runtime**, is **notarized** by Apple, and has the notarization ticket **stapled** to both the app
and the DMG, so it opens on other Macs without Gatekeeper warnings, even offline.

`scripts/release-macos.sh` does all of this. Signing identities and credentials stay on the
release machine; nothing secret is in the repository.

## One-time setup

1. **Apple Developer Program membership** (paid) for the account that will publish the app.
2. **Developer ID Application certificate** in your login keychain.
   Xcode → Settings → Accounts → *your team* → Manage Certificates → **+** → *Developer ID
   Application*. Check with:

   ```bash
   security find-identity -v -p codesigning | grep "Developer ID Application"
   ```

3. **Notary credentials in the keychain.** Create an *app-specific password* at
   [appleid.apple.com](https://appleid.apple.com) (Sign-In and Security → App-Specific Passwords),
   then run the following yourself. It prompts for the password and stores it in your keychain:

   ```bash
   xcrun notarytool store-credentials llmario-notary --apple-id YOUR_APPLE_ID --team-id YOUR_TEAM_ID
   ```

   Your team ID is the 10-character code in parentheses at the end of the certificate name.
4. Tooling: Xcode (for `notarytool`/`stapler`), Rust with both Mac targets, and the Tauri CLI:

   ```bash
   rustup target add aarch64-apple-darwin x86_64-apple-darwin
   cargo install tauri-cli --version "^2" --locked
   ```

## Cutting a release

1. Bump `version` in `apps/desktop/src-tauri/tauri.conf.json` (and `Cargo.toml`
   `[workspace.package]`).
2. Run:

   ```bash
   scripts/release-macos.sh
   ```

   The script:
   - builds a **universal** app (Apple Silicon + Intel); set `TARGET=aarch64-apple-darwin` for
     Apple Silicon only
   - signs it with your Developer ID (hardened runtime, secure timestamp) and verifies the signature
   - notarizes the app and staples it
   - builds `dist/LLMario-<version>-macos-universal.dmg` (app + Applications shortcut), then signs,
     notarizes and staples the DMG
   - runs Gatekeeper's own assessment (`spctl`) and writes a `.sha256` file next to the DMG

   `--skip-notarize` does everything except notarization. Use it as a dry run; that output is not
   shareable.
3. Share the DMG, and publish its SHA-256 so recipients can verify the download:

   ```bash
   shasum -a 256 -c LLMario-<version>-macos-universal.dmg.sha256
   ```

   To publish on GitHub:
   `gh release create v<version> dist/LLMario-*.dmg dist/LLMario-*.dmg.sha256 --notes "…"`.

## What recipients need

- macOS 13 or later.
- **An inference engine.** The app does not bundle one yet. Recipients install either:
  - llama.cpp: `brew install llama.cpp` (any Mac), or
  - MLX-LM on Apple Silicon: `pip install mlx-lm`.

  The app's **Models → Engines** panel shows what it found. Models that need a missing engine
  are shown as unavailable instead of failing.

## Checking a build by hand

```bash
codesign -dv --verbose=4 LLMario.app 2>&1 | grep -E "Authority|flags|Timestamp|TeamIdentifier"
spctl --assess --type execute --verbose=2 LLMario.app          # "accepted source=Notarized Developer ID"
xcrun stapler validate LLMario-<version>-macos-universal.dmg   # "The validate action worked!"
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no "Developer ID Application" certificate` | Create it in Xcode (step 2). An *Apple Development* certificate cannot be used for distribution. |
| `notary profile 'llmario-notary' not found` | Run the `store-credentials` command (step 3). |
| Build waits with no output | A Keychain dialog is asking to use the signing key. Choose **Always Allow**. |
| Notarization `Invalid` | Run `xcrun notarytool log <submission-id> --keychain-profile llmario-notary`. The log lists each rejected file. |
| Recipient sees "damaged" or "unidentified developer" | They got an unnotarized build (`--skip-notarize`) or a file modified after signing. Re-run the full script. |

# Releasing for Windows

Windows builds are made on GitHub's Windows machines by
[`.github/workflows/windows.yml`](../.github/workflows/windows.yml), which runs on every pull
request and every push to `main`. Nothing Windows-specific is needed on the release machine.

Each run:

- builds `llmario.exe` and runs smoke tests on real Windows with llama.cpp (pinned build,
  checksum-verified). It downloads and checks the smallest library model, chats with it, and
  confirms that engines stop with llmario, including when llmario is killed (job object).
- builds the desktop installer (NSIS, per-user, no admin rights), then installs it, launches the
  app for 20 s and uninstalls it.
- uploads the artifact `llmario-windows-<version>` with:
  - `LLMario-<version>-windows-x64-setup.exe`: the desktop app installer
  - `llmario-<version>-windows-x64.zip`: the command-line tool (`llmario.exe`, LICENSE, NOTICE)
  - a `.sha256` file for each

## Attaching the Windows files to a release

1. Check that the `windows` workflow passed for the release commit.
2. Download its artifact and attach the files, together with the macOS DMG:

   ```bash
   gh run download <run-id> -n llmario-windows-<version> -D dist/windows
   gh release create v<version> dist/LLMario-*.dmg dist/LLMario-*.dmg.sha256 dist/windows/* --notes-file notes.md
   ```

3. Update the download links in `website/index.html` and `README.md`. They point at the
   versioned files, so a one-click download never lands on an empty page. Pushing
   `website/` redeploys llmario.com.

## Signing

The Windows files are **not code-signed** yet, so Windows SmartScreen shows "Windows protected
your PC" the first time the installer runs. Users click **More info → Run anyway**. The website and
release notes say so. Signing (for example with Azure Trusted Signing) can be added to the
workflow later without other changes.

## What Windows users need

- Windows 10 or 11, x64. The installer adds the Microsoft Edge WebView2 runtime if it is missing
  (it is built into Windows 11).
- llama.cpp: `winget install ggml.llamacpp` (its Vulkan build), or a zip from
  [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) on `PATH`. MLX is
  macOS-only. LLMario detects NVIDIA GPUs (via `nvidia-smi`) and offloads to them; on other
  GPUs it plans for CPU-only execution for now.
- LLMario keeps its files in `%USERPROFILE%\.llmario` (override with `LLMARIO_HOME`).
