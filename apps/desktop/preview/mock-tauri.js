// Simulated Tauri bridge for previewing the UI in a normal browser. Library data comes from
// `llmario model catalog --json > apps/desktop/preview/catalog.json` (real data for this machine);
// everything else is canned. Never loaded by the desktop app.
(() => {
  const GiB = 2 ** 30;
  let catalog = [];
  const ready = fetch("/preview/catalog.json").then((r) => (r.ok ? r.json() : [])).then((c) => (catalog = c)).catch(() => {});
  const overview = {
    version: "0.1.0-preview", cpu: "Preview CPU", os: "macOS", memoryTotalBytes: 64 * GiB, memoryAvailableBytes: 30 * GiB,
    gpu: "Preview GPU", gpuMemoryBytes: 48 * GiB, unifiedMemory: true,
    backends: [
      { kind: "llamacpp", available: true, version: "0.5.0 (build 11146)", detail: "preview" },
      { kind: "mlx", available: true, version: "mlx-lm 0.31.3 / mlx 0.32.2", detail: "preview" },
    ],
    profile: { kind: "latency", parallel: 1, ctx_per_slot: 8192 }, home: "~/.llmario-beta", loaded: [],
  };
  const installed = () => catalog.filter((c) => c.installed).map((c) => ({
    id: c.id, family: c.family, format: c.format, backend: c.backend, backendAvailable: c.backendAvailable,
    quantization: c.quantization, sizeBytes: c.approxBytes, license: c.license, source: c.repo, contextMax: c.contextMax,
    managed: true, fits: c.fits, tight: c.tight, needsBytes: c.needsBytes, budgetBytes: c.budgetBytes, loaded: false,
  }));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  class Channel { constructor() { this.onmessage = () => {}; } }
  async function invoke(cmd, args = {}) {
    await ready;
    switch (cmd) {
      case "start": case "overview": case "apply_settings": return overview;
      case "list_models": return installed();
      case "list_catalog": return catalog;
      case "load_model": await sleep(600); return { model: args.name, backend: "mlx", readySeconds: 1.3, footprintBytes: 2 * GiB, estimateBytes: 3 * GiB, budgetBytes: 48 * GiB, profile: overview.profile, notes: [] };
      case "unload_model": return true;
      case "pull_model": {
        const c = catalog.find((x) => x.id === args.id);
        for (let i = 1; i <= 20; i++) { await sleep(120); args.onProgress.onmessage({ file: "model", doneBytes: (c.approxBytes * i) / 20, totalBytes: c.approxBytes, reused: false }); }
        c.installed = true;
        return installed().find((m) => m.id === c.id);
      }
      case "chat": {
        args.onEvent.onmessage({ type: "start", model: args.model, backend: "mlx" });
        for (const w of "Let me think about that.".split(" ")) { await sleep(40); args.onEvent.onmessage({ type: "reasoning", text: w + " " }); }
        for (const w of "Here is a **preview** reply with `code` and a list:\n\n- one\n- two".split(" ")) { await sleep(40); args.onEvent.onmessage({ type: "content", text: w + " " }); }
        return { ttftMs: 120, tokensPerSecond: 220, completionTokens: 24, finishReason: "stop", cancelled: false };
      }
      case "cancel_chat": return true;
      default: throw { code: "preview", message: `${cmd} is not simulated in the preview` };
    }
  }
  window.__TAURI__ = { core: { invoke, Channel }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => () => {} }) } };
})();
