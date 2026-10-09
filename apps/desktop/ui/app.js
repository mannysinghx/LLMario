// LLMario desktop UI. Talks to the Rust side only through Tauri IPC (`invoke` + channels).
import { renderMarkdown } from "./markdown.js";

const tauri = window.__TAURI__;
const invoke = tauri?.core?.invoke;
const Channel = tauri?.core?.Channel;
const $ = (s) => document.querySelector(s);

const DEFAULT_SETTINGS = {
  profile: "latency",
  context: "",
  maxTokens: 2048,
  temperature: 0.6,
  system: "",
  showThinking: false,
  saveHistory: true,
};
const BACKEND = { llamacpp: "llama.cpp", mlx: "MLX", native: "LLMario engine", mock: "mock" };
// "mac" | "windows" | "linux". Drives platform-specific hints (`data-os` in index.html) and
// hides MLX, which runs only on Apple Silicon Macs.
const PLATFORM = /Windows/.test(navigator.userAgent) ? "windows" : /Macintosh|Mac OS X/.test(navigator.userAgent) ? "mac" : "linux";
document.documentElement.dataset.platform = PLATFORM;
const SUGGESTIONS = [
  ["Explain an idea", "How does a transformer model generate text, in simple terms?"],
  ["Write code", "Write a Python function that removes duplicates from a list but keeps the order."],
  ["Brainstorm", "Give me three weekend project ideas that use a local AI model."],
  ["Summarize", "What are the pros and cons of running language models locally instead of in the cloud?"],
];

// localStorage can be unavailable; the app must work without it.
const store = {
  get(k, d) {
    try {
      const v = localStorage.getItem(`llmario.${k}`);
      return v == null ? d : JSON.parse(v);
    } catch {
      return d;
    }
  },
  set(k, v) {
    try {
      localStorage.setItem(`llmario.${k}`, JSON.stringify(v));
    } catch {}
  },
  del(k) {
    try {
      localStorage.removeItem(`llmario.${k}`);
    } catch {}
  },
};

const state = {
  overview: null,
  models: [],
  catalog: [],
  chats: [],
  currentId: null,
  model: store.get("model", null),
  loadedModel: null,
  loading: null,
  generating: null,
  pulls: {},
  settings: { ...DEFAULT_SETTINGS, ...store.get("settings", {}) },
};

// ---------- formatting ----------
const fmtDisk = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
const fmtMem = (b) => `${(b / 2 ** 30).toFixed(b >= 10 * 2 ** 30 ? 0 : 1)} GB`;
const errMsg = (e) => (typeof e === "string" ? e : e?.message || JSON.stringify(e));
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

function toast(title, body = "", isError = false, ms = 6000) {
  const t = el("div", `toast${isError ? " error" : ""}`);
  t.append(el("strong", null, title));
  if (body) t.append(el("div", null, body));
  $("#toasts").append(t);
  setTimeout(() => t.remove(), ms);
}

function setStatus(kind, text) {
  const p = $("#model-status");
  p.className = `pill ${kind}`;
  p.textContent = text;
}

// ---------- chats ----------
function loadChats() {
  state.chats = state.settings.saveHistory ? store.get("chats", []) : [];
  state.chats.sort((a, b) => b.updated - a.updated);
  state.currentId = state.chats[0]?.id ?? null;
}

function saveChats() {
  if (!state.settings.saveHistory) return store.del("chats");
  const clean = state.chats
    .filter((c) => c.messages.length)
    .map((c) => ({ ...c, messages: c.messages.map(({ streaming, _el, ...m }) => m) }));
  store.set("chats", clean);
}

const currentChat = () => state.chats.find((c) => c.id === state.currentId) || null;

function newChat() {
  if (state.generating) return;
  const empty = state.chats.find((c) => !c.messages.length);
  if (empty) {
    state.currentId = empty.id;
  } else {
    const c = { id: crypto.randomUUID(), title: "New chat", created: Date.now(), updated: Date.now(), messages: [] };
    state.chats.unshift(c);
    state.currentId = c.id;
  }
  renderChatList();
  renderTranscript();
  $("#input").focus();
}

function renderChatList() {
  const list = $("#chat-list");
  list.replaceChildren();
  const chats = state.chats.filter((c) => c.messages.length || c.id === state.currentId);
  if (!chats.length) list.append(el("div", "empty-note", "No chats yet"));
  for (const c of chats) {
    const item = el("button", `chat-item${c.id === state.currentId ? " active" : ""}`);
    item.type = "button";
    item.append(el("span", "title", c.title || "New chat"));
    const del = el("span", "del", "✕");
    del.title = "Delete chat";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      if (state.generating?.chatId === c.id) return;
      state.chats = state.chats.filter((x) => x.id !== c.id);
      if (state.currentId === c.id) state.currentId = state.chats[0]?.id ?? null;
      saveChats();
      renderChatList();
      renderTranscript();
    });
    item.append(del);
    item.addEventListener("click", () => {
      if (state.generating) return;
      state.currentId = c.id;
      renderChatList();
      renderTranscript();
    });
    list.append(item);
  }
}

// ---------- transcript ----------
function renderTranscript() {
  const box = $("#transcript");
  box.replaceChildren();
  const chat = currentChat();
  if (!chat || !chat.messages.length) {
    box.append(renderWelcome());
    return;
  }
  const thread = el("div", "thread");
  for (const m of chat.messages) thread.append(renderMessage(m));
  box.append(thread);
  box.scrollTop = box.scrollHeight;
}

function renderWelcome() {
  const w = el("div", "welcome");
  const img = el("img");
  img.src = "icon.png";
  img.alt = "";
  w.append(img);
  if (!state.models.length) {
    w.append(el("h1", null, "Download a model to start"));
    w.append(el("p", null, `Models run entirely on this computer. Pick one sized for your machine, or drag your own .gguf file${PLATFORM === "mac" ? " or MLX model folder" : ""} onto this window.`));
    const picks = state.catalog
      .filter((c) => c.recommended && !c.installed && c.tasks.includes("chat"))
      .sort((a, b) => (a.approxBytes || 0) - (b.approxBytes || 0))
      .slice(0, 3);
    const s = el("div", "starter");
    for (const c of picks) s.append(variantRow(c, { compact: true }));
    if (!picks.length) s.append(el("p", "muted", "Open Models to see what can run here."));
    const more = el("button", "btn small ghost", `Browse all ${new Set(state.catalog.map((c) => c.family)).size} model families →`);
    more.type = "button";
    more.addEventListener("click", () => openModels("library"));
    s.append(more);
    w.append(s);
    return w;
  }
  w.append(el("h1", null, "What can I help with?"));
  w.append(el("p", null, "Private by default: your messages never leave this computer."));
  const grid = el("div", "suggestions");
  for (const [title, prompt] of SUGGESTIONS) {
    const b = el("button", "suggestion");
    b.type = "button";
    b.append(el("strong", null, title), el("span", null, prompt));
    b.addEventListener("click", () => send(prompt));
    grid.append(b);
  }
  w.append(grid);
  return w;
}

function renderMessage(m) {
  if (m.role === "user") {
    const row = el("div", "msg user");
    row.append(el("div", "bubble", m.content));
    return row;
  }
  const row = el("div", "msg assistant");
  const av = el("img", "avatar");
  av.src = "icon.png";
  av.alt = "";
  const body = el("div", "body");
  row.append(av, body);
  m._el = body;
  updateAssistant(m);
  return row;
}

function updateAssistant(m) {
  const body = m._el;
  if (!body) return;
  const box = $("#transcript");
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;

  let think = body.querySelector(".thinking");
  if (m.reasoning) {
    if (!think) {
      think = el("details", "thinking");
      think.open = state.settings.showThinking;
      think.append(el("summary"), el("div", "thinking-body"));
      body.prepend(think);
    }
    const words = m.reasoning.trim().split(/\s+/).length;
    think.querySelector("summary").textContent =
      m.streaming && !m.content ? "Thinking…" : `Thought for ~${words.toLocaleString()} words`;
    think.querySelector(".thinking-body").textContent = m.reasoning;
  }

  let answer = body.querySelector(".answer");
  if (!answer) {
    answer = el("div", "answer");
    body.append(answer);
  }
  answer.innerHTML = renderMarkdown(m.content); // renderer escapes all model text
  answer.classList.toggle("cursor", !!m.streaming && (!m.reasoning || !!m.content));
  if (m.streaming && !m.content && !m.reasoning) answer.textContent = "";

  body.querySelector(".error-box")?.remove();
  if (m.error) body.append(el("div", "error-box", m.error));

  body.querySelector(".stats")?.remove();
  if (!m.streaming && (m.stats || m.backend)) {
    const s = el("div", "stats");
    const st = m.stats || {};
    if (st.ttftMs != null) s.append(el("span", null, `first token ${Math.round(st.ttftMs)} ms`));
    if (st.tokensPerSecond != null) s.append(el("span", null, `${st.tokensPerSecond.toFixed(1)} tok/s`));
    if (st.completionTokens) s.append(el("span", null, `${st.completionTokens} tokens`));
    if (m.model) s.append(el("span", null, `${m.model} · ${BACKEND[m.backend] || m.backend}`));
    if (st.cancelled) s.append(el("span", null, "stopped"));
    if (st.finishReason === "length") s.append(el("span", null, "hit max length"));
    body.append(s);
  }
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

let pendingFrame = null;
function scheduleRender(m) {
  if (pendingFrame) return;
  pendingFrame = requestAnimationFrame(() => {
    pendingFrame = null;
    updateAssistant(m);
  });
}

// ---------- models ----------
async function refreshModels() {
  state.models = await invoke("list_models");
  const loaded = state.models.find((m) => m.loaded);
  if (loaded) state.loadedModel = loaded.id;
  const usable = (m) => m && m.fits && m.backendAvailable;
  if (!usable(state.models.find((m) => m.id === state.model))) {
    state.model = (state.models.find((m) => m.loaded) || state.models.find(usable))?.id ?? null;
  }
  renderModelSelect();
  renderInstalled();
  if (!currentChat()?.messages.length) renderTranscript();
}

async function refreshCatalog() {
  const catalog = await invoke("list_catalog");
  state.catalog = PLATFORM === "mac" ? catalog : catalog.filter((c) => c.backend !== "mlx");
  renderCatalog();
  if (!currentChat()?.messages.length) renderTranscript();
}

function renderModelSelect() {
  const sel = $("#model-select");
  sel.replaceChildren();
  if (!state.models.length) {
    const o = el("option", null, "No models installed");
    o.value = "";
    sel.append(o);
    sel.disabled = true;
    setStatus("", "Open Models to download one");
    return;
  }
  sel.disabled = false;
  for (const m of state.models) {
    const why = !m.backendAvailable ? " — engine missing" : !m.fits ? " — too large" : "";
    const o = el("option", null, `${m.id} · ${BACKEND[m.backend]}${why}`);
    o.value = m.id;
    o.disabled = !!why;
    sel.append(o);
  }
  sel.value = state.model || "";
  if (state.loadedModel && state.loadedModel === state.model) {
    setStatus("ready", "Ready");
    $("#unload").hidden = false;
  } else if (!state.loading) {
    setStatus("", "Loads on first message");
    $("#unload").hidden = true;
  }
}

function ensureLoaded(id) {
  if (state.loadedModel === id) return Promise.resolve();
  if (state.loading?.id === id) return state.loading.promise;
  setStatus("loading", `Loading ${id}…`);
  $("#unload").hidden = true;
  const promise = invoke("load_model", { name: id })
    .then((info) => {
      state.loadedModel = info.model;
      if (state.model === info.model) {
        setStatus("ready", `Ready · ${BACKEND[info.backend]} · loaded in ${info.readySeconds.toFixed(1)} s`);
        $("#unload").hidden = false;
      }
      if (info.warning) toast("Memory is shared with another app", info.warning, false, 12000);
      return info;
    })
    .catch((e) => {
      setStatus("error", "Could not load");
      throw e;
    })
    .finally(() => {
      if (state.loading?.promise === promise) state.loading = null;
    });
  state.loading = { id, promise };
  return promise;
}

async function selectModel(id) {
  state.model = id;
  store.set("model", id);
  renderModelSelect();
  if (!id || state.loadedModel === id) return;
  try {
    await ensureLoaded(id);
    await refreshModels();
  } catch (e) {
    toast("Could not load the model", errMsg(e), true, 10000);
  }
}

function renderInstalled() {
  const list = $("#installed-list");
  list.replaceChildren();
  if (!state.models.length) list.append(el("p", "muted small", "No models yet. Download one below."));
  for (const m of state.models) {
    const row = el("div", "row");
    const info = el("div", "info");
    const name = el("div", "name", m.id);
    name.append(el("span", "badge accent", BACKEND[m.backend]));
    if (m.loaded) name.append(el("span", "badge ok", "loaded"));
    if (!m.backendAvailable) name.append(el("span", "badge bad", `needs ${BACKEND[m.backend]}`));
    else if (!m.fits) name.append(el("span", "badge bad", "too large"));
    else if (m.tight) {
      const b = el("span", "badge warn", "tight");
      b.title = "Fits, but uses most of this computer's memory. Close other apps before loading it.";
      name.append(b);
    }
    info.append(name);
    const bits = [m.quantization, fmtDisk(m.sizeBytes), `needs ~${fmtMem(m.needsBytes)} of ${fmtMem(m.budgetBytes)} memory`, fmtSpeed(m.speed)];
    if (m.license) bits.push(m.license);
    info.append(el("div", "meta", bits.filter(Boolean).join(" · ")));
    const actions = el("div", "actions");
    const use = el("button", "btn small", m.id === state.model ? "Selected" : "Use");
    use.type = "button";
    use.disabled = m.id === state.model || !m.fits || !m.backendAvailable;
    use.addEventListener("click", () => {
      closeOverlays();
      selectModel(m.id);
    });
    actions.append(use);
    const rm = el("button", "btn small ghost danger", m.managed ? "Remove" : "Unregister");
    rm.type = "button";
    rm.title = m.managed ? "Delete the downloaded files" : "Forget this model; its files stay where they are";
    rm.addEventListener("click", () => confirmThen(rm, () => removeModel(m.id)));
    actions.append(rm);
    row.append(info, actions);
    list.append(row);
  }
}

// Two-step confirm without native dialogs.
function confirmThen(btn, action) {
  if (btn.dataset.armed) return action();
  const label = btn.textContent;
  btn.dataset.armed = "1";
  btn.textContent = "Click to confirm";
  setTimeout(() => {
    delete btn.dataset.armed;
    btn.textContent = label;
  }, 3000);
}

async function removeModel(id) {
  try {
    await invoke("remove_model", { id });
    if (state.loadedModel === id) state.loadedModel = null;
    if (state.model === id) state.model = null;
    await Promise.all([refreshModels(), refreshCatalog()]);
    toast("Model removed", id);
  } catch (e) {
    toast("Could not remove the model", errMsg(e), true);
  }
}

// ---------- model library ----------
const TASK_LABELS = {
  chat: "Chat",
  reasoning: "Reasoning",
  coding: "Coding",
  agents: "Agents",
  multilingual: "Multilingual",
  "long-context": "Long context",
  small: "Small & fast",
};
// "~N tok/s" (measured or predicted decode speed on this computer), or null.
function fmtSpeed(s) {
  if (!s || !s.tokens_per_second) return null;
  return `~${Math.round(s.tokens_per_second)} tok/s${s.measured ? " (measured)" : ""}`;
}

const lib = { query: "", task: null, fits: false, comfortable: false, works: false };
const usable = (c) => c.backendAvailable && c.supported !== false;
const fmtCtx = (n) => (n % 1024 === 0 ? `${n >= 1048576 ? n / 1048576 + "M" : n / 1024 + "K"}` : `${Math.round(n / 1000)}K`);
const pullPct = (p) => (p.totalBytes ? Math.min(100, (100 * p.doneBytes) / p.totalBytes) : 0);

function variantRow(c, { compact = false } = {}) {
  const row = el("div", `variant${compact ? " compact" : ""}`);
  row.dataset.catalog = c.id;
  const main = el("div", "variant-main");
  const line = el("div", "variant-line");
  line.append(el("span", "badge accent", BACKEND[c.backend]));
  if (c.recommended) line.append(el("span", "badge ok", "recommended"));
  if (compact) line.append(el("strong", null, c.name));
  const bits = [
    c.quantization,
    c.approxBytes ? `${fmtDisk(c.approxBytes)} download` : null,
    c.contextMax ? `${fmtCtx(c.contextMax)} context` : null,
    fmtSpeed(c.speed),
  ];
  line.append(el("span", null, bits.filter(Boolean).join(" · ")));
  const fit = el(
    "span",
    !c.fits ? "fit-bad" : c.tight ? "fit-tight" : "fit-ok",
    !c.fits
      ? `needs ~${fmtMem(c.needsBytes)} · too large (${fmtMem(c.budgetBytes)} available)`
      : `needs ~${fmtMem(c.needsBytes)} · ${c.tight ? "fits, tight" : "fits"}`
  );
  if (c.fits && c.tight) fit.title = "Fits, but uses most of this computer's memory. Close other apps before loading it.";
  line.append(fit);
  if (c.mtp) {
    const b = el("span", "badge accent", "MTP");
    b.title = "Includes multi-token-prediction layers: faster with speculative decoding (backends.llamacpp.speculative = \"mtp\").";
    line.append(b);
  }
  if (!c.backendAvailable) line.append(el("span", "badge bad", `${BACKEND[c.backend]} not installed`));
  else if (c.supported === false) {
    const b = el("span", "badge bad", "needs a newer engine");
    b.title = `Architecture "${c.architecture}" is not supported by your ${BACKEND[c.backend]} (${c.engineVersion || "unknown version"}).`;
    line.append(b);
  }
  if (c.gated) line.append(el("span", "badge", "requires accepting terms on Hugging Face"));
  main.append(line);
  if (!compact) {
    const name = c.files.length
      ? `${c.repo} / ${c.files[0]}${c.files.length > 1 ? ` (+${c.files.length - 1} more parts)` : ""}`
      : `${c.repo} (MLX model folder)`;
    const exact = el("div", "exact", name);
    exact.title = `Pinned to commit ${c.revision}`;
    const copy = el("button", "linkish", "copy");
    copy.type = "button";
    copy.title = "Copy the exact Hugging Face name";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(c.files.length ? `${c.repo}/${c.files[0]}` : c.repo);
        copy.textContent = "copied";
        setTimeout(() => (copy.textContent = "copy"), 1500);
      } catch {
        toast("Copy failed", "Clipboard access was denied.", true);
      }
    });
    exact.append(" ", copy);
    main.append(exact);
  }
  const actions = el("div", "actions");
  const btn = el("button", "btn small primary", "Download");
  btn.type = "button";
  const pull = state.pulls[c.id];
  if (c.installed) {
    btn.className = "btn small";
    btn.textContent = state.model === c.id ? "In use" : "Use";
    btn.disabled = state.model === c.id || !c.fits || !usable(c);
    btn.addEventListener("click", () => {
      closeOverlays();
      selectModel(c.id);
    });
  } else if (!c.backendAvailable) {
    btn.className = "btn small";
    btn.textContent = "Engine missing";
    btn.disabled = true;
  } else if (c.supported === false) {
    btn.className = "btn small";
    btn.textContent = "Not supported";
    btn.disabled = true;
  } else if (pull) {
    btn.textContent = "Downloading…";
    btn.disabled = true;
  } else if (!c.fits) {
    // Estimates are conservative, so allow it, but never as a single accidental click.
    btn.className = "btn small";
    btn.textContent = "Download anyway";
    btn.title = `Needs ~${fmtMem(c.needsBytes)}; ${fmtMem(c.budgetBytes)} is available. It will probably not load with the current settings.`;
    btn.addEventListener("click", () => confirmThen(btn, () => pullModel(c)));
  } else {
    btn.addEventListener("click", () => pullModel(c));
  }
  actions.append(btn);
  row.append(main, actions);
  if (pull) {
    const bar = el("div", "progress");
    const fill = el("div");
    fill.style.width = `${pullPct(pull)}%`;
    bar.append(fill);
    const text = `${fmtDisk(pull.doneBytes)} of ${fmtDisk(pull.totalBytes || c.approxBytes || 0)}${pull.reused ? " (from local cache)" : ""}`;
    row.append(bar, el("div", "progress-text", text));
  }
  return row;
}

function familyVariants(variants) {
  const q = lib.query.trim().toLowerCase();
  const v0 = variants[0];
  if (q) {
    const hay = [
      v0.name, v0.family, v0.publisher, v0.description, v0.params, v0.released,
      ...v0.tasks.map((t) => TASK_LABELS[t] || t),
      ...variants.flatMap((v) => [v.id, v.repo, BACKEND[v.backend], v.architecture]),
    ].join(" ").toLowerCase();
    if (!q.split(/\s+/).every((w) => hay.includes(w))) return null;
  }
  if (lib.task && !v0.tasks.includes(lib.task)) return null;
  let vs = variants;
  if (lib.fits) vs = vs.filter((v) => v.fits);
  if (lib.comfortable) vs = vs.filter((v) => v.fits && !v.tight);
  if (lib.works) vs = vs.filter(usable);
  return vs.length ? vs : null;
}

function renderCatalog() {
  const list = $("#catalog-list");
  if (!list) return;
  list.replaceChildren();
  const fams = new Map();
  for (const c of state.catalog) {
    if (!fams.has(c.family)) fams.set(c.family, []);
    fams.get(c.family).push(c);
  }
  let shown = 0;
  for (const variants of fams.values()) {
    const vs = familyVariants(variants);
    if (!vs) continue;
    shown++;
    const v0 = variants[0];
    const card = el("article", "family");
    const head = el("div", "family-head");
    head.append(el("h4", null, v0.name));
    head.append(
      el("span", "family-meta", [v0.publisher, v0.released, v0.params ? `${v0.params} parameters` : null, v0.license].filter(Boolean).join(" · "))
    );
    card.append(head, el("p", "family-desc", v0.description));
    if (v0.notes) card.append(el("p", "family-notes", v0.notes));
    const tags = el("div", "tags");
    for (const t of v0.tasks) tags.append(el("span", "tag", TASK_LABELS[t] || t));
    card.append(tags);
    for (const v of [...vs].sort((a, b) => b.recommended - a.recommended)) card.append(variantRow(v));
    list.append(card);
  }
  $("#lib-count").textContent = `Showing ${shown} of ${fams.size} model families · ${state.catalog.length} downloads, each verified on Hugging Face and pinned to an exact version.`;
  if (!shown) list.append(el("div", "empty-lib", "No models match. Try another search or filter."));
}

function renderTaskChips() {
  const box = $("#lib-tasks");
  box.replaceChildren();
  for (const [key, label] of [[null, "All"], ...Object.entries(TASK_LABELS)]) {
    const b = el("button", `chip-btn${lib.task === key ? " active" : ""}`, label);
    b.type = "button";
    b.addEventListener("click", () => {
      lib.task = key;
      renderTaskChips();
      renderCatalog();
    });
    box.append(b);
  }
}

// Re-render just the rows of one model (download progress) instead of the whole library.
function refreshVariantRows(id) {
  const c = state.catalog.find((x) => x.id === id);
  if (!c) return;
  document.querySelectorAll(".variant").forEach((r) => {
    if (r.dataset.catalog === id) r.replaceWith(variantRow(c, { compact: r.classList.contains("compact") }));
  });
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
}

async function pullModel(c) {
  if (state.pulls[c.id]) return;
  state.pulls[c.id] = { doneBytes: 0, totalBytes: c.approxBytes || 0 };
  refreshVariantRows(c.id);
  const ch = new Channel();
  let frame = null;
  ch.onmessage = (p) => {
    state.pulls[c.id] = p;
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      refreshVariantRows(c.id);
    });
  };
  try {
    const m = await invoke("pull_model", { id: c.id, onProgress: ch });
    toast("Download complete", `${c.name} (${m.id}) is ready to use.`);
    delete state.pulls[c.id];
    await Promise.all([refreshCatalog(), refreshModels()]);
    if (!state.model || state.model === m.id) {
      state.model = m.id;
      store.set("model", m.id);
      renderModelSelect();
    }
  } catch (e) {
    delete state.pulls[c.id];
    refreshVariantRows(c.id);
    toast(`Could not download ${c.name}`, errMsg(e), true, 10000);
  }
  if (!currentChat()?.messages.length) renderTranscript();
}

function renderOverview() {
  const o = state.overview;
  $("#version").textContent = `v${o.version} · local models`;
  const gpu = o.gpu ? `${o.gpu}` : "CPU only";
  $("#hw-line").textContent = `${gpu} · ${fmtMem(o.memoryTotalBytes)} ${o.unifiedMemory ? "unified memory" : "RAM"}`;
  $("#profile-chip").textContent = `${o.profile.kind} · ${o.profile.ctx_per_slot.toLocaleString()} ctx`;
  const list = $("#backend-list");
  list.replaceChildren();
  for (const b of o.backends) {
    const row = el("div", "row");
    const info = el("div", "info");
    const name = el("div", "name", BACKEND[b.kind]);
    name.append(el("span", `badge ${b.available ? "ok" : "bad"}`, b.available ? "available" : "not found"));
    info.append(name, el("div", "meta", [b.version, b.detail].filter(Boolean).join(" · ")));
    row.append(info);
    list.append(row);
  }
}

// ---------- chatting ----------
function historyFor(chat, reply) {
  const msgs = [];
  if (state.settings.system.trim()) msgs.push({ role: "system", content: state.settings.system.trim() });
  for (const m of chat.messages) {
    if (m === reply) break;
    if (m.role === "assistant" && (m.error || !m.content)) continue;
    msgs.push({ role: m.role, content: m.content });
  }
  return msgs;
}

function setBusy(busy) {
  const b = $("#send");
  b.classList.toggle("stop", busy);
  b.setAttribute("aria-label", busy ? "Stop" : "Send");
  updateSendEnabled();
}

function updateSendEnabled() {
  $("#send").disabled = !state.generating && !$("#input").value.trim();
}

async function send(text) {
  text = (text ?? $("#input").value).trim();
  if (!text || state.generating) return;
  if (!state.model) {
    openModels();
    toast("Choose a model first", "Download or select a model to chat with.");
    return;
  }
  if (!currentChat()) newChat();
  const chat = currentChat();
  const modelId = state.model;
  $("#input").value = "";
  autoGrow();
  chat.messages.push({ role: "user", content: text });
  if (chat.title === "New chat") chat.title = text.length > 48 ? `${text.slice(0, 48)}…` : text;
  const reply = { role: "assistant", content: "", reasoning: "", streaming: true };
  chat.messages.push(reply);
  chat.updated = Date.now();
  state.chats.sort((a, b) => b.updated - a.updated);
  renderChatList();
  renderTranscript();

  const requestId = crypto.randomUUID();
  state.generating = { requestId, chatId: chat.id };
  setBusy(true);
  try {
    await ensureLoaded(modelId);
    const ch = new Channel();
    ch.onmessage = (ev) => {
      if (ev.type === "start") {
        reply.model = ev.model;
        reply.backend = ev.backend;
      } else if (ev.type === "reasoning") reply.reasoning += ev.text;
      else if (ev.type === "content") reply.content += ev.text;
      scheduleRender(reply);
    };
    reply.stats = await invoke("chat", {
      requestId,
      model: modelId,
      messages: historyFor(chat, reply),
      maxTokens: Number(state.settings.maxTokens) || null,
      temperature: Number(state.settings.temperature),
      onEvent: ch,
    });
  } catch (e) {
    reply.error = errMsg(e);
    if (/engine_crashed|crashed/.test(reply.error)) state.loadedModel = null;
  } finally {
    reply.streaming = false;
    state.generating = null;
    setBusy(false);
    chat.updated = Date.now();
    saveChats();
    // Channel messages can land just after the command resolves; settle the final render.
    updateAssistant(reply);
    setTimeout(() => updateAssistant(reply), 80);
  }
}

async function stop() {
  if (!state.generating) return;
  await invoke("cancel_chat", { requestId: state.generating.requestId });
}

// ---------- overlays & settings ----------
function openModels(tab = "library") {
  renderInstalled();
  renderTaskChips();
  renderCatalog();
  showTab(tab);
  $("#models-panel").hidden = false;
  if (tab === "library") $("#lib-search").focus();
}
function openSettings() {
  const f = $("#settings-form");
  const s = state.settings;
  f.profile.value = s.profile;
  f.context.value = s.context || "";
  f.maxTokens.value = s.maxTokens;
  f.temperature.value = s.temperature;
  $("#temp-out").textContent = Number(s.temperature).toFixed(2);
  f.system.value = s.system;
  f.showThinking.checked = s.showThinking;
  f.saveHistory.checked = s.saveHistory;
  $("#settings-panel").hidden = false;
}
function closeOverlays() {
  document.querySelectorAll(".overlay").forEach((o) => (o.hidden = true));
}

async function saveSettings(e) {
  e.preventDefault();
  const f = e.target;
  const next = {
    profile: f.profile.value || "latency",
    context: f.context.value,
    maxTokens: Math.max(16, Number(f.maxTokens.value) || DEFAULT_SETTINGS.maxTokens),
    temperature: Number(f.temperature.value),
    system: f.system.value,
    showThinking: f.showThinking.checked,
    saveHistory: f.saveHistory.checked,
  };
  const runtimeChanged = next.profile !== state.settings.profile || next.context !== state.settings.context;
  if (runtimeChanged && state.generating) {
    toast("Wait for the reply to finish", "Profile changes restart the model.", true);
    return;
  }
  state.settings = next;
  store.set("settings", next);
  saveChats();
  closeOverlays();
  if (runtimeChanged) {
    setStatus("loading", "Applying settings…");
    try {
      state.overview = await invoke("apply_settings", { settings: runtimeSettings() });
      state.loadedModel = null;
      renderOverview();
      await refreshModels();
      toast("Settings applied", `Profile ${state.overview.profile.kind}; the model reloads on the next message.`);
    } catch (e) {
      setStatus("error", "Settings failed");
      toast("Could not apply settings", errMsg(e), true);
    }
  }
}

function runtimeSettings() {
  const s = state.settings;
  return { profile: s.profile, context: s.context ? Number(s.context) : null };
}

function autoGrow() {
  const t = $("#input");
  t.style.height = "auto";
  t.style.height = `${Math.min(t.scrollHeight, 220)}px`;
  updateSendEnabled();
}

// ---------- wiring ----------
function wire() {
  $("#new-chat").addEventListener("click", newChat);
  $("#open-models").addEventListener("click", () => openModels("library"));
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  $("#lib-search").addEventListener("input", (e) => {
    lib.query = e.target.value;
    renderCatalog();
  });
  $("#lib-fits").addEventListener("change", (e) => {
    lib.fits = e.target.checked;
    renderCatalog();
  });
  $("#lib-comfortable").addEventListener("change", (e) => {
    lib.comfortable = e.target.checked;
    renderCatalog();
  });
  $("#lib-works").addEventListener("change", (e) => {
    lib.works = e.target.checked;
    renderCatalog();
  });
  $("#open-settings").addEventListener("click", openSettings);
  document.querySelectorAll(".overlay").forEach((o) =>
    o.addEventListener("click", (e) => {
      if (e.target === o || e.target.closest(".close")) closeOverlays();
    })
  );
  $("#settings-form").addEventListener("submit", saveSettings);
  $("#settings-form").temperature.addEventListener("input", (e) => ($("#temp-out").textContent = Number(e.target.value).toFixed(2)));
  $("#clear-history").addEventListener("click", (e) =>
    confirmThen(e.currentTarget, () => {
      if (state.generating) return;
      state.chats = [];
      state.currentId = null;
      store.del("chats");
      renderChatList();
      renderTranscript();
      toast("Chat history cleared");
    })
  );
  $("#model-select").addEventListener("change", (e) => selectModel(e.target.value));
  $("#unload").addEventListener("click", async () => {
    if (state.generating || !state.loadedModel) return;
    await invoke("unload_model", { id: state.loadedModel });
    state.loadedModel = null;
    await refreshModels();
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    state.generating ? stop() : send();
  });
  $("#input").addEventListener("input", autoGrow);
  $("#input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!state.generating) send();
    }
  });
  $("#transcript").addEventListener("click", async (e) => {
    const b = e.target.closest(".md-copy");
    if (!b) return;
    const code = b.closest(".md-code").querySelector("code").textContent;
    try {
      await navigator.clipboard.writeText(code);
      b.textContent = "Copied";
      setTimeout(() => (b.textContent = "Copy"), 1500);
    } catch {
      toast("Copy failed", "Clipboard access was denied.", true);
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeOverlays();
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") {
      e.preventDefault();
      newChat();
    }
    // Windows/Linux webviews reload on F5 / Ctrl+R, which would drop a reply mid-stream.
    if (PLATFORM !== "mac" && (e.key === "F5" || (e.ctrlKey && e.key.toLowerCase() === "r"))) {
      e.preventDefault();
    }
  });
}

// ---------- drag and drop your own model files ----------
async function wireDragDrop() {
  const webview = tauri?.webview?.getCurrentWebview?.();
  if (!webview?.onDragDropEvent) return;
  const zone = $("#drop-zone");
  await webview.onDragDropEvent((event) => {
    const p = event.payload || {};
    if (p.type === "enter") zone.hidden = !(p.paths && p.paths.length);
    else if (p.type === "leave") zone.hidden = true;
    else if (p.type === "drop") {
      zone.hidden = true;
      addDropped(p.paths || []);
    }
  });
}

async function addDropped(paths) {
  for (const path of paths) {
    const name = path.split(/[\\/]/).filter(Boolean).pop() || path;
    setStatus("loading", `Adding ${name}…`);
    toast(`Adding ${name}`, "Reading the model and computing its checksum. Large files take a few seconds.", false, 4000);
    try {
      const m = await invoke("add_model", { path });
      await refreshModels();
      const what = [BACKEND[m.backend], m.quantization, fmtDisk(m.sizeBytes)].filter(Boolean).join(" · ");
      if (!m.backendAvailable) {
        toast("Added, but its engine is not installed", `${m.id} needs ${BACKEND[m.backend]}. See Models → Engines.`, true, 10000);
      } else if (!m.fits) {
        toast("Added, but it is too large for this computer",
          `${m.id} needs ~${fmtMem(m.needsBytes)} of memory; ${fmtMem(m.budgetBytes)} is available. A smaller context (Settings) may help.`, true, 12000);
      } else if (state.generating) {
        toast("Model added", `${m.id} (${what}). Select it when the current reply finishes.`);
      } else {
        toast("Model added", `${m.id} (${what}). Loading it now.`);
        await selectModel(m.id);
      }
    } catch (e) {
      toast(`Could not add ${name}`, errMsg(e), true, 12000);
    }
  }
  renderModelSelect();
  renderInstalled();
}

async function boot() {
  if (!invoke) {
    document.body.textContent = "LLMario's interface must run inside the desktop app.";
    return;
  }
  wire();
  wireDragDrop().catch(() => {});
  loadChats();
  renderChatList();
  renderTranscript();
  updateSendEnabled();
  setStatus("loading", "Starting…");
  try {
    state.overview = await invoke("start", { settings: runtimeSettings() });
    renderOverview();
    await Promise.all([refreshModels(), refreshCatalog()]);
  } catch (e) {
    setStatus("error", "Startup failed");
    toast("LLMario could not start", errMsg(e), true, 15000);
  }
  $("#input").focus();
}

boot();
