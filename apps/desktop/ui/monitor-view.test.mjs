// Run: node --test apps/desktop/ui/monitor-view.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { History, monitorView, sparkPath, fmt, statusOf, HISTORY } from "./monitor-view.js";

const GiB = 2 ** 30;

// Field names exactly as `llmario_api::monitor::Snapshot` serializes them.
function snapshot(over = {}) {
  return {
    version: "0.2.2",
    at_unix_ms: 1_790_000_000_000,
    uptime_s: 75,
    state: "generating",
    waiting: 0,
    speed_tok_s: 92.34,
    live: [{ model: "m", phase: "generating", elapsed_s: 2, ttft_ms: 310, prompt_tokens_estimate: 28000, generated: 120, tok_s: 92.34 }],
    engines: [
      {
        model: "m", backend: "native", pid: 42, family: "qwen3", format: "gguf", architecture: "qwen3", quantization: "Q4_K_M",
        file_bytes: 5 * GiB, layers: 36, context_max: 131072, context_per_slot: 131072, parallel: 1, batch: 2048, ubatch: 512,
        profile: "latency", active_requests: 1, context_used: 29000, load_seconds: 3.2, idle_seconds: 0,
        estimated_bytes: 7 * GiB, memory_bytes: 6 * GiB, cpu_percent: 61, disk_read_bps: 62_600_000, disk_write_bps: 0,
        disk_read_total: 9 * GiB,
        engine_stats: {
          device: "metal", threads: 12, kernels: "neon", prefill_tok_s: 850.5, decode_tok_s: 88.1, prompt_tokens: 1000,
          cached_prompt_tokens: 600, completion_tokens: 400, kv_disk: { enabled: true, saves: 2, restores: 1 },
          memory_pressure: { level: "normal" }, paging: { model_does_not_fit_memory: false },
        },
      },
    ],
    recent: [
      { at_unix_ms: 1_790_000_000_000, model: "m", outcome: "ok", prompt_tokens: 28693, cached_tokens: 28140, completion_tokens: 346, ttft_ms: 400, duration_ms: 7000, decode_tok_s: 73.5 },
      { at_unix_ms: 1_789_999_990_000, model: "m", outcome: "context_overflow", prompt_tokens: 0, cached_tokens: null, completion_tokens: 0, ttft_ms: null, duration_ms: 12, decode_tok_s: null },
    ],
    totals: { requests: 2, ok: 1, cancelled: 0, failed: 1, prompt_tokens: 28693, cached_tokens: 28140, completion_tokens: 346 },
    machine: { cpu: "Apple M4 Max", os: "macOS", arch: "aarch64", unified_memory: true, memory_bandwidth_gbs: 546, gpu: "Apple M4 Max", gpu_api: "metal", profile: "latency" },
    gateway: null,
    system: {
      cpu_percent: 61, logical_cores: 16, physical_cores: 16, load_average_1m: 3, memory_total_bytes: 64 * GiB,
      memory_used_bytes: 40 * GiB, memory_available_bytes: 24 * GiB, swap_total_bytes: 4 * GiB, swap_used_bytes: 2 * GiB,
      gpus: [{ name: "Apple M4 Max", source: "ioreg", utilization_percent: 97, memory_used_bytes: 1.5 * GiB, memory_total_bytes: null, temperature_c: null, power_w: null, power_limit_w: null, pcie: null }],
      processes: [],
    },
    ...over,
  };
}

const tile = (v, key) => v.tiles.find((t) => t.key === key);

test("a generating snapshot fills tiles, bars, engine details and recent requests", () => {
  const s = snapshot();
  const h = new History();
  h.push(s);
  const v = monitorView(s, h);
  assert.equal(v.state, "generating");
  assert.deepEqual(v.states.filter((x) => x.active).map((x) => x.key), ["generating"]);
  assert.equal(tile(v, "speed").value, "92.3 tok/s");
  assert.equal(tile(v, "speed").sub, "now");
  assert.equal(tile(v, "gpu").value, "97 %");
  assert.equal(tile(v, "vram").title, "GPU memory", "unified memory has no VRAM total");
  assert.equal(tile(v, "disk").value, "59.7 MB/s");
  assert.equal(tile(v, "reuse").value, "98 %");
  const ctx = v.bars.find((b) => b.key === "context");
  assert.equal(ctx.value, "28K / 128K · 22%");
  assert.equal(v.bars.find((b) => b.key === "swap").tone, "warn", "over 1 GB of swap is flagged");
  const kv = Object.fromEntries(v.engine);
  assert.equal(kv.Engine, "native · metal · pid 42");
  assert.equal(kv["Prefill / decode"], "851 / 88.1 tok/s avg");
  assert.equal(kv["KV disk tier"], "on · 2 saves · 1 restores");
  assert.equal(kv.Paging, "no");
  assert.equal(v.recent.length, 2);
  assert.deepEqual([v.recent[0].status, v.recent[0].prompt, v.recent[0].tokS, v.recent[0].duration], ["Done", "28,693", "73.5", "7.0 s"]);
  assert.deepEqual([v.recent[1].status, v.recent[1].title, v.recent[1].reused], ["Failed", "context_overflow", "n/a"]);
  assert.match(v.machine, /Apple M4 Max · 546 GB\/s · up 1m15s/);
});

test("a discrete GPU shows VRAM and power; idle shows the last speed", () => {
  const s = snapshot({ state: "idle", live: [], speed_tok_s: 0 });
  s.system.gpus = [{ name: "RTX 3060", source: "nvidia-smi", utilization_percent: 100, memory_used_bytes: 11.4 * GiB, memory_total_bytes: 12 * GiB, temperature_c: 68, power_w: 159, power_limit_w: 170, pcie: "Gen3 x16" }];
  const v = monitorView(s, new History());
  assert.equal(tile(v, "vram").title, "VRAM");
  assert.equal(tile(v, "vram").value, "11.4 GB / 12.0 GB");
  assert.ok(Math.abs(tile(v, "vram").ratio - 0.95) < 0.001);
  assert.equal(tile(v, "thermal").value, "159 W · 68 °C");
  assert.equal(tile(v, "thermal").sub, "of 170 W limit");
  assert.equal(tile(v, "speed").sub, "last 73.5 tok/s");
});

test("missing readings show n/a, never a made-up zero", () => {
  const s = snapshot({ state: "idle", live: [], engines: [], recent: [], speed_tok_s: 0 });
  s.system.gpus = [];
  s.system.cpu_percent = null;
  const v = monitorView(s, new History());
  assert.equal(tile(v, "gpu").value, "n/a");
  assert.equal(tile(v, "cpu").value, "n/a");
  assert.equal(tile(v, "vram").title, "Engine memory");
  assert.equal(tile(v, "vram").value, "n/a");
  assert.equal(tile(v, "reuse").value, "n/a");
  assert.equal(v.engine, null);
  assert.deepEqual(v.recent, []);
  assert.ok(!v.bars.some((b) => b.key === "context"), "no context bar without a model");
});

test("history keeps the newest points and sparkline paths stay inside the box", () => {
  const h = new History();
  for (let i = 0; i < HISTORY + 30; i++) h.push(snapshot({ speed_tok_s: i }));
  assert.equal(h.series.speed.length, HISTORY);
  assert.equal(h.series.speed.at(-1), HISTORY + 29);
  const d = sparkPath([0, 5, 10], 120, 26);
  assert.match(d, /^M0\.0,25\.5 L60\.0,13\.0 L120\.0,0\.5$/);
  assert.equal(sparkPath([], 120, 26), "");
  assert.match(sparkPath([0, 0], 120, 26), /^M0\.0,25\.5 L120\.0,25\.5$/, "all-zero series is a flat line");
  h.clear();
  assert.equal(h.series.cpu.length, 0);
});

test("non-streaming requests show a running state, listed only while active", () => {
  const live = [{ model: "m", phase: "running", streaming: false, elapsed_s: 3, ttft_ms: null, prompt_tokens_estimate: 50, generated: 0, tok_s: null }];
  const v = monitorView(snapshot({ state: "running", live, speed_tok_s: 0 }), new History());
  assert.deepEqual(v.states.filter((x) => x.active).map((x) => x.key), ["running"]);
  assert.match(tile(v, "speed").sub, /not streamed/);
  const idle = monitorView(snapshot({ state: "idle", live: [] }), new History());
  assert.ok(!idle.states.some((x) => x.key === "running"));
});

test("context past the planned window says how far over, not a percentage above 100", () => {
  const s = snapshot();
  s.engines[0].context_per_slot = 4096;
  s.engines[0].context_used = 5728;
  const ctx = monitorView(s, new History()).bars.find((b) => b.key === "context");
  assert.equal(ctx.value, "5.6K / 4K · 1.6K over plan");
  assert.equal(ctx.tone, "warn");
  assert.equal(ctx.ratio, 1);
  assert.match(ctx.title, /5,728 tokens.*4,096 per request/);
  s.engines[0].context_used = 2048;
  const ok = monitorView(s, new History()).bars.find((b) => b.key === "context");
  assert.equal(ok.value, "2K / 4K · 50%");
  assert.equal(ok.tone, null);
  assert.equal(ok.title, "");
});

test("formatting", () => {
  assert.equal(fmt.ktok(131072), "128K");
  assert.equal(fmt.ktok(5728), "5.6K");
  assert.equal(fmt.ktok(4096), "4K");
  assert.equal(fmt.int(28693), "28,693");
  assert.equal(fmt.ms(169900), "2m50s");
  assert.equal(fmt.ms(119700), "2m00s");
  assert.equal(fmt.ms(59700), "1m00s");
  assert.equal(fmt.ms(4800), "4.8 s");
  assert.equal(fmt.rate(500), "0 KB/s");
  assert.equal(fmt.gb(undefined), "n/a");
  assert.deepEqual(statusOf("cancelled"), { text: "Stopped", tone: "warn" });
});
