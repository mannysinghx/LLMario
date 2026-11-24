// What the Monitor panel shows, computed from one `runtime_monitor` snapshot (the gateway's
// `GET /llmario/monitor`, field names in snake_case) and the history of earlier ones.
//
// Pure functions only, so they run under `node --test`; `monitor-panel.js` draws the result.
// A value the machine or engine does not report is shown as "n/a", never as a made-up zero.

const GB = 2 ** 30;
const MB = 2 ** 20;
export const HISTORY = 120;
export const STATES = ["idle", "reading", "generating", "queued", "error"];

const na = "n/a";
const titleCase = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");
const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export const fmt = {
  gb: (b) => (isNum(b) ? `${(b / GB).toFixed(1)} GB` : na),
  rate: (bps) => {
    if (!isNum(bps)) return na;
    if (bps >= GB) return `${(bps / GB).toFixed(2)} GB/s`;
    if (bps >= MB) return `${(bps / MB).toFixed(1)} MB/s`;
    return `${Math.round(bps / 1024)} KB/s`;
  },
  // Context sizes: 131072 → "128K".
  ktok: (n) => (n >= 1024 ? `${Math.round(n / 1024)}K` : String(n ?? 0)),
  int: (n) => (isNum(n) ? Math.round(n).toLocaleString("en-US") : na),
  pct: (v) => (isNum(v) ? `${Math.round(v)} %` : na),
  tps: (v) => (isNum(v) ? `${v.toFixed(1)} tok/s` : na),
  ms: (ms) => {
    if (!isNum(ms)) return na;
    if (ms >= 60000) return `${Math.floor(ms / 60000)}m${String(Math.round((ms % 60000) / 1000)).padStart(2, "0")}s`;
    if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`;
    return `${Math.round(ms)} ms`;
  },
  clock: (unixMs) => {
    const d = new Date(unixMs);
    return Number.isNaN(d.getTime()) ? "--:--:--" : d.toTimeString().slice(0, 8);
  },
};

/** Rolling series for the sparklines; one point per snapshot. */
export class History {
  constructor(max = HISTORY) {
    this.max = max;
    this.series = { speed: [], gpu: [], cpu: [], disk: [] };
  }
  push(s) {
    const g = s?.system?.gpus?.[0];
    const add = (k, v) => {
      const a = this.series[k];
      a.push(isNum(v) ? v : 0);
      if (a.length > this.max) a.shift();
    };
    add("speed", s?.speed_tok_s);
    add("gpu", g?.utilization_percent);
    add("cpu", s?.system?.cpu_percent);
    add("disk", engineDiskRead(s));
  }
  clear() {
    for (const k of Object.keys(this.series)) this.series[k] = [];
  }
}

/** SVG path for a sparkline of `values` in a `w`×`h` box, scaled to the largest value. */
export function sparkPath(values, w, h) {
  if (!values || values.length === 0) return "";
  const max = Math.max(1e-9, ...values);
  const step = values.length > 1 ? w / (values.length - 1) : 0;
  return values
    .map((v, i) => {
      const x = values.length > 1 ? i * step : w;
      const y = h - (Math.max(0, v) / max) * (h - 1) - 0.5;
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function engineDiskRead(s) {
  return (s?.engines || []).reduce((a, e) => a + (isNum(e.disk_read_bps) ? e.disk_read_bps : 0), 0);
}

const reuse = (r) => (r && isNum(r.cached_tokens) && r.prompt_tokens > 0 ? (r.cached_tokens * 100) / r.prompt_tokens : null);
const ratio = (a, b) => (isNum(a) && isNum(b) && b > 0 ? Math.min(1, Math.max(0, a / b)) : null);

export function statusOf(outcome) {
  if (outcome === "ok") return { text: "Done", tone: "ok" };
  if (outcome === "cancelled") return { text: "Stopped", tone: "warn" };
  return { text: "Failed", tone: "bad", title: outcome };
}

/**
 * The panel's content. `hist` is a History that has already seen `s`.
 * Returns { state, states, waiting, tiles, bars, engine, recent, machine }.
 */
export function monitorView(s, hist) {
  const gpu = s.system?.gpus?.[0] || null;
  const eng = s.engines?.[0] || null;
  const es = eng?.engine_stats || null;
  const last = s.recent?.[0] || null;
  const sys = s.system || {};
  const generating = (s.live || []).some((l) => l.phase === "generating");
  const series = hist?.series || {};

  const tiles = [
    {
      key: "speed",
      title: "Speed",
      value: fmt.tps(s.speed_tok_s ?? 0),
      sub: generating ? "now" : last && isNum(last.decode_tok_s) ? `last ${last.decode_tok_s.toFixed(1)} tok/s` : "idle",
      spark: series.speed,
    },
    {
      key: "gpu",
      title: "GPU load",
      value: fmt.pct(gpu?.utilization_percent),
      sub: gpu?.name || s.machine?.gpu || "no GPU reported",
      spark: series.gpu,
    },
    gpu && isNum(gpu.memory_total_bytes)
      ? {
          key: "vram",
          title: "VRAM",
          value: `${fmt.gb(gpu.memory_used_bytes)} / ${fmt.gb(gpu.memory_total_bytes)}`,
          sub: "dedicated GPU memory",
          ratio: ratio(gpu.memory_used_bytes, gpu.memory_total_bytes),
        }
      : gpu
        ? {
            key: "vram",
            title: "GPU memory",
            value: fmt.gb(gpu.memory_used_bytes),
            sub: `unified · of ${fmt.gb(sys.memory_total_bytes)}`,
            ratio: ratio(gpu.memory_used_bytes, sys.memory_total_bytes),
          }
        : {
            key: "vram",
            title: "Engine memory",
            value: fmt.gb(eng?.memory_bytes),
            sub: eng ? `planned ${fmt.gb(eng.estimated_bytes)}` : "no model loaded",
            ratio: ratio(eng?.memory_bytes, sys.memory_total_bytes),
          },
    {
      key: "cpu",
      title: "CPU",
      value: fmt.pct(sys.cpu_percent),
      sub: `${sys.physical_cores ?? "?"} cores · ${sys.logical_cores ?? "?"} threads`,
      spark: series.cpu,
    },
    {
      key: "disk",
      title: "Disk read",
      value: fmt.rate(engineDiskRead(s)),
      sub: "by the engines",
      spark: series.disk,
    },
    isNum(gpu?.temperature_c) || isNum(gpu?.power_w)
      ? {
          key: "thermal",
          title: "GPU power",
          value: [isNum(gpu.power_w) ? `${Math.round(gpu.power_w)} W` : null, isNum(gpu.temperature_c) ? `${Math.round(gpu.temperature_c)} °C` : null]
            .filter(Boolean)
            .join(" · "),
          sub: isNum(gpu.power_limit_w) ? `of ${Math.round(gpu.power_limit_w)} W limit` : "",
          ratio: ratio(gpu.power_w, gpu.power_limit_w),
        }
      : {
          key: "reuse",
          title: "Cache reuse",
          value: fmt.pct(reuse(last)),
          sub: last && isNum(last.cached_tokens) ? `${fmt.int(last.cached_tokens)} prompt tokens reused` : "not reported",
          ratio: reuse(last) == null ? null : reuse(last) / 100,
        },
  ];

  const bars = [];
  if (eng) {
    const used = eng.context_used ?? 0;
    const total = eng.context_per_slot || 0;
    bars.push({
      key: "context",
      label: "Context fill",
      value: `${fmt.ktok(used)} / ${fmt.ktok(total)} · ${total ? Math.round((used * 100) / total) : 0}%`,
      ratio: ratio(used, total) ?? 0,
    });
    bars.push({
      key: "engine-mem",
      label: "Engine memory",
      value: `${fmt.gb(eng.memory_bytes)} · planned ${fmt.gb(eng.estimated_bytes)}`,
      ratio: ratio(eng.memory_bytes, sys.memory_total_bytes),
    });
  }
  bars.push({
    key: "ram",
    label: "System RAM",
    value: `${fmt.gb(sys.memory_used_bytes)} / ${fmt.gb(sys.memory_total_bytes)}`,
    ratio: ratio(sys.memory_used_bytes, sys.memory_total_bytes),
  });
  bars.push({
    key: "swap",
    label: "Swap",
    value: `${fmt.gb(sys.swap_used_bytes)} / ${fmt.gb(sys.swap_total_bytes)}`,
    ratio: ratio(sys.swap_used_bytes, sys.swap_total_bytes),
    tone: sys.swap_used_bytes > GB ? "warn" : null,
  });

  let engine = null;
  if (eng) {
    engine = [
      ["Model", eng.model],
      ["Engine", `${eng.backend}${es?.device ? ` · ${es.device}` : ""} · pid ${eng.pid}`],
      ["Format", [eng.format, eng.quantization, eng.architecture].filter(Boolean).join(" · ") || na],
      ["Model file", fmt.gb(eng.file_bytes)],
      ["Layers", eng.layers ?? na],
      ["Context", `${fmt.int(eng.context_per_slot)} × ${eng.parallel} slot${eng.parallel === 1 ? "" : "s"} (max ${eng.context_max ? fmt.int(eng.context_max) : na})`],
      ["Batch", `${eng.batch} / ${eng.ubatch}`],
      ["Active", `${eng.active_requests} of ${eng.parallel}`],
      ["Load time", fmt.ms(eng.load_seconds * 1000)],
      ["Engine CPU", isNum(eng.cpu_percent) ? `${Math.round(eng.cpu_percent)} % of one core` : na],
    ];
    if (es) {
      engine.push(
        ["Threads", `${es.threads ?? na}${es.kernels ? ` · ${es.kernels}` : ""}`],
        ["Prefill / decode", `${isNum(es.prefill_tok_s) ? es.prefill_tok_s.toFixed(0) : na} / ${isNum(es.decode_tok_s) ? es.decode_tok_s.toFixed(1) : na} tok/s avg`],
        ["Tokens", `${fmt.int(es.prompt_tokens)} in · ${fmt.int(es.cached_prompt_tokens)} cached · ${fmt.int(es.completion_tokens)} out`],
        ["KV disk tier", es.kv_disk?.enabled ? `on · ${es.kv_disk.saves} saves · ${es.kv_disk.restores} restores` : "off"],
        ["Memory pressure", es.memory_pressure?.level ?? na],
        ["Paging", es.paging?.model_does_not_fit_memory ? "yes: weights re-read every token" : "no"]
      );
    }
  }

  const recent = (s.recent || []).slice(0, 12).map((r) => {
    const st = statusOf(r.outcome);
    return {
      time: fmt.clock(r.at_unix_ms),
      status: st.text,
      tone: st.tone,
      title: st.title || "",
      prompt: fmt.int(r.prompt_tokens),
      reused: isNum(r.cached_tokens) ? fmt.int(r.cached_tokens) : na,
      output: fmt.int(r.completion_tokens),
      tokS: isNum(r.decode_tok_s) ? r.decode_tok_s.toFixed(1) : na,
      duration: fmt.ms(r.duration_ms),
    };
  });

  const m = s.machine || {};
  return {
    state: s.state,
    stateLabel: titleCase(s.state),
    states: STATES.map((k) => ({ key: k, label: titleCase(k), active: k === s.state })),
    waiting: s.waiting || 0,
    tiles,
    bars,
    engine,
    recent,
    machine: [m.cpu, m.memory_bandwidth_gbs ? `${Math.round(m.memory_bandwidth_gbs)} GB/s` : null, `up ${fmt.ms((s.uptime_s || 0) * 1000)}`]
      .filter(Boolean)
      .join(" · "),
  };
}
