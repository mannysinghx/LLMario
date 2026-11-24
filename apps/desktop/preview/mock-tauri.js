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
    profile: { kind: "latency", parallel: 1, ctx_per_slot: 8192 }, home: "~/.llmario", loaded: [],
  };
  const installed = () => catalog.filter((c) => c.installed).map((c) => ({
    id: c.id, family: c.family, format: c.format, backend: c.backend, backendAvailable: c.backendAvailable,
    quantization: c.quantization, sizeBytes: c.approxBytes, license: c.license, source: c.repo, contextMax: c.contextMax,
    managed: true, fits: c.fits, tight: c.tight, speed: c.speed, needsBytes: c.needsBytes, budgetBytes: c.budgetBytes, loaded: false,
  }));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // Canned runtime-monitor snapshots (the gateway's `GET /llmario/monitor` shape) that wobble.
  let tick = 0;
  function monitorSnapshot() {
    tick++;
    const now = Date.now();
    const gen = tick % 12 < 8;
    const w = (base, amp) => Math.max(0, base + amp * Math.sin(tick / 2));
    return {
      version: overview.version, at_unix_ms: now, uptime_s: 120 + tick, state: gen ? "generating" : "idle", waiting: 0,
      speed_tok_s: gen ? w(88, 8) : 0,
      live: gen ? [{ model: "qwen3-8b-mlx-4bit", phase: "generating", elapsed_s: tick % 12, ttft_ms: 310, prompt_tokens_estimate: 2800, generated: 40 * (tick % 12), tok_s: w(88, 8) }] : [],
      engines: [{ model: "qwen3-8b-mlx-4bit", backend: "mlx", pid: 4242, family: "qwen3-8b", format: "mlx", architecture: "qwen3", quantization: "4-bit (group 64)",
        file_bytes: 4.31 * GiB, layers: 36, context_max: 40960, context_per_slot: 8192, parallel: 1, batch: 2048, ubatch: 512, profile: "latency",
        active_requests: gen ? 1 : 0, context_used: 2800 + 40 * (tick % 12), load_seconds: 2.4, idle_seconds: gen ? 0 : tick % 12,
        estimated_bytes: 6.2 * GiB, memory_bytes: 5.1 * GiB, cpu_percent: gen ? w(35, 10) : 2, disk_read_bps: gen ? w(2e6, 1e6) : 0, disk_write_bps: 0,
        disk_read_total: 4.4 * GiB, engine_stats: null }],
      recent: [
        { at_unix_ms: now - 15000, model: "qwen3-8b-mlx-4bit", outcome: "ok", prompt_tokens: 2793, cached_tokens: 2610, completion_tokens: 346, ttft_ms: 290, duration_ms: 4200, decode_tok_s: 87.3 },
        { at_unix_ms: now - 42000, model: "qwen3-8b-mlx-4bit", outcome: "cancelled", prompt_tokens: 2100, cached_tokens: 1900, completion_tokens: 120, ttft_ms: 310, duration_ms: 1700, decode_tok_s: 85.0 },
        { at_unix_ms: now - 80000, model: "qwen3-8b-mlx-4bit", outcome: "context_overflow", prompt_tokens: 0, cached_tokens: null, completion_tokens: 0, ttft_ms: null, duration_ms: 12, decode_tok_s: null },
      ],
      totals: { requests: 3, ok: 1, cancelled: 1, failed: 1, prompt_tokens: 4893, cached_tokens: 4510, completion_tokens: 466 },
      machine: { cpu: overview.cpu, os: "macOS", arch: "aarch64", performance_cores: 12, efficiency_cores: 4, unified_memory: true, memory_bandwidth_gbs: 546, gpu: "Preview GPU", gpu_api: "metal", profile: "latency" },
      gateway: { pid: 1, cpu_percent: 1, memory_bytes: 80e6, disk_read_bps: 0, disk_write_bps: 0, disk_read_total: 0, disk_write_total: 0 },
      system: { cpu_percent: gen ? w(40, 12) : 6, logical_cores: 16, physical_cores: 16, load_average_1m: 3.2, memory_total_bytes: 64 * GiB,
        memory_used_bytes: 41 * GiB, memory_available_bytes: 23 * GiB, swap_total_bytes: 2 * GiB, swap_used_bytes: 0.3 * GiB,
        gpus: [{ name: "Preview GPU", source: "ioreg", utilization_percent: gen ? w(85, 10) : 3, memory_used_bytes: 5.4 * GiB, memory_total_bytes: null,
          temperature_c: null, power_w: null, power_limit_w: null, pcie: null }], processes: [] },
    };
  }
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
      case "engine_memory": return null;
      case "runtime_monitor": return monitorSnapshot();
      default: throw { code: "preview", message: `${cmd} is not simulated in the preview` };
    }
  }
  window.__TAURI__ = { core: { invoke, Channel }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => () => {} }) } };
})();
