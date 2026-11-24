// The Monitor panel: live technical readings of the running model, switched on and off with the
// topbar button or ⌘⇧M / Ctrl+Shift+M. While it is off (or the window is hidden) nothing is
// polled, so the runtime reads nothing about the machine. All text goes in through
// `textContent`; nothing from the snapshot is parsed as HTML.

import { History, monitorView, sparkPath } from "./monitor-view.js";

const POLL_MS = 1000;
const SVG = "http://www.w3.org/2000/svg";

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = String(text);
  return n;
};

function spark(values) {
  const w = 120;
  const h = 26;
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("class", "mon-spark");
  svg.setAttribute("aria-hidden", "true");
  const d = sparkPath(values, w, h);
  if (d) {
    const area = document.createElementNS(SVG, "path");
    area.setAttribute("d", `${d} L${w},${h} L0,${h} Z`);
    area.setAttribute("class", "mon-spark-area");
    const line = document.createElementNS(SVG, "path");
    line.setAttribute("d", d);
    line.setAttribute("class", "mon-spark-line");
    svg.append(area, line);
  }
  return svg;
}

function meter(ratio, tone) {
  const m = el("div", `mon-meter${tone ? ` ${tone}` : ""}`);
  const f = el("div", "mon-meter-fill");
  f.style.width = `${Math.round((ratio ?? 0) * 100)}%`;
  m.append(f);
  if (ratio == null) m.classList.add("empty");
  return m;
}

function render(body, pill, v) {
  pill.className = `pill mon-${v.state}`;
  pill.textContent = v.stateLabel;

  const states = el("div", "mon-states");
  for (const s of v.states) states.append(el("span", s.active ? `active mon-${s.key}` : "", s.label));
  if (v.waiting) states.append(el("span", "mon-waiting", `${v.waiting} waiting`));

  const tiles = el("div", "mon-tiles");
  for (const t of v.tiles) {
    const tile = el("div", "mon-tile");
    tile.dataset.key = t.key;
    tile.append(el("div", "mon-tile-title", t.title), el("div", "mon-tile-value", t.value), el("div", "mon-tile-sub", t.sub));
    tile.append(t.spark ? spark(t.spark) : meter(t.ratio));
    tiles.append(tile);
  }

  const bars = el("div", "mon-bars");
  for (const b of v.bars) {
    const row = el("div", "mon-bar");
    const head = el("div", "mon-bar-head");
    head.append(el("span", null, b.label), el("span", "mon-bar-value", b.value));
    row.append(head, meter(b.ratio, b.tone));
    bars.append(row);
  }

  const parts = [states, tiles, bars];

  if (v.engine) {
    const sec = el("section", "mon-section");
    sec.append(el("h3", null, "Engine"));
    const dl = el("dl", "mon-kv");
    for (const [k, val] of v.engine) dl.append(el("dt", null, k), el("dd", null, val));
    sec.append(dl);
    parts.push(sec);
  } else {
    parts.push(el("p", "mon-empty", "No model loaded. Engine details appear once a model is loaded."));
  }

  const sec = el("section", "mon-section");
  sec.append(el("h3", null, "Recent requests"));
  if (v.recent.length) {
    const table = el("table", "mon-table");
    const thead = el("tr");
    const heads = [
      ["Time", "When it finished"],
      ["Status", ""],
      ["In", "Prompt tokens"],
      ["Cached", "Prompt tokens reused from the engine's cache"],
      ["Out", "Generated tokens"],
      ["Tok/s", "Generation speed after the first token"],
      ["Took", "Whole request, start to finish"],
    ];
    for (const [h, title] of heads) {
      const th = el("th", null, h);
      if (title) th.title = title;
      thead.append(th);
    }
    table.append(thead);
    for (const r of v.recent) {
      const tr = el("tr");
      const st = el("td", `mon-${r.tone}`, r.status);
      if (r.title) st.title = r.title;
      tr.append(el("td", null, r.time), st, el("td", null, r.prompt), el("td", null, r.reused), el("td", null, r.output), el("td", null, r.tokS), el("td", null, r.duration));
      table.append(tr);
    }
    sec.append(table);
  } else {
    sec.append(el("p", "mon-empty", "None yet. Send a message to see its timings here."));
  }
  parts.push(sec);
  parts.push(el("p", "mon-foot", v.machine));

  body.replaceChildren(...parts);
}

/**
 * Wires the panel. `invoke` is Tauri's; `store` is app.js's localStorage wrapper.
 * Returns { set(on), toggle(), isOn() }.
 */
export function createMonitor({ invoke, store }) {
  const app = document.querySelector(".app");
  const panel = document.querySelector("#monitor-panel");
  const body = document.querySelector("#mon-body");
  const pill = document.querySelector("#mon-state");
  const button = document.querySelector("#toggle-monitor");
  const hist = new History();
  let on = false;
  let timer = null;
  let busy = false;

  async function tick() {
    if (busy || !on || document.hidden) return;
    busy = true;
    try {
      const s = await invoke("runtime_monitor");
      if (!on) return;
      hist.push(s);
      render(body, pill, monitorView(s, hist));
    } catch (e) {
      if (!on) return;
      pill.className = "pill error";
      pill.textContent = "Unavailable";
      body.replaceChildren(el("p", "mon-empty", `The runtime did not answer: ${typeof e === "string" ? e : e?.message || "unknown error"}`));
    } finally {
      busy = false;
    }
  }

  function set(next) {
    on = !!next;
    store.set("monitor", on);
    panel.hidden = !on;
    app.classList.toggle("monitor-on", on);
    button.setAttribute("aria-pressed", String(on));
    button.title = on ? "Turn the monitor off (⌘⇧M)" : "Show live model metrics (⌘⇧M)";
    clearInterval(timer);
    timer = null;
    if (on) {
      hist.clear();
      tick();
      timer = setInterval(tick, POLL_MS);
    }
  }

  button.addEventListener("click", () => set(!on));
  document.querySelector("#mon-close").addEventListener("click", () => set(false));
  // A hidden window polls nothing; catch up as soon as it is visible again.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && on) tick();
  });
  set(store.get("monitor", false));
  return { set, toggle: () => set(!on), isOn: () => on };
}
