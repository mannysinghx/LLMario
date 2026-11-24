//! `llmario top`: a live terminal dashboard of the running models.
//!
//! It attaches to a running gateway (`llmario serve`, or any LLMario API) and polls
//! `GET /llmario/monitor` once a second. Starting it switches the monitor on; quitting switches
//! it off again, and the server reads nothing about the machine while nobody watches. The
//! server and the models keep running either way.

use anyhow::Context;
use clap::Args;
use llmario_api::monitor::{EngineView, RecentRequest, Snapshot};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Cell, Paragraph, Row, Sparkline, Table, Wrap};
use ratatui::Frame;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Args, Clone)]
pub struct TopArgs {
    /// Base URL of a running LLMario server (default: the configured `[server]` address).
    #[arg(long)]
    pub url: Option<String>,
    /// API key, when the server requires one (prefer the LLMARIO_TOP_API_KEY env var). Without
    /// --url, the configured server's own key (`[server] api_key` or LLMARIO_API_KEY) is used; it
    /// is never sent to a host named with --url.
    #[arg(long, env = "LLMARIO_TOP_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Refresh interval in milliseconds (at least 250).
    #[arg(long, default_value_t = 1000)]
    pub interval_ms: u64,
    /// Print one snapshot as JSON and exit (for scripts).
    #[arg(long)]
    pub json: bool,
}

const HISTORY: usize = 240;
const GIB: f64 = (1u64 << 30) as f64;
const MIB: f64 = (1u64 << 20) as f64;
const ACCENT: Color = Color::Green;
const DIM: Color = Color::DarkGray;

pub async fn run(a: TopArgs) -> anyhow::Result<()> {
    let (_, cfg) = crate::util::load_config(&crate::RuntimeArgs::default())?;
    let url = a.url.clone().unwrap_or_else(|| {
        let host = match cfg.server.host.as_str() {
            "0.0.0.0" | "" => "127.0.0.1".to_string(),
            "::" => "[::1]".to_string(),
            h if h.contains(':') && !h.starts_with('[') => format!("[{h}]"),
            h => h.to_string(),
        };
        format!("http://{host}:{}", cfg.server.port)
    });
    let url = url.trim_end_matches('/').to_string();
    let key = top_key(&a, cfg.server.api_key.as_deref());
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()?;
    let fetch = Fetcher {
        http,
        endpoint: format!("{url}/llmario/monitor"),
        key,
    };

    if a.json {
        let v = fetch.raw().await?;
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }

    let shared = Arc::new(Mutex::new(App::new(url.clone())));
    let refresh = Arc::new(tokio::sync::Notify::new());
    let interval = Duration::from_millis(a.interval_ms.max(250));
    let poller = {
        let shared = shared.clone();
        let refresh = refresh.clone();
        tokio::spawn(async move {
            loop {
                let r = fetch.snapshot().await;
                shared.lock().unwrap().apply(r);
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {}
                    _ = refresh.notified() => {}
                }
            }
        })
    };
    // SIGTERM / SIGHUP (and Ctrl-C outside raw mode) end the loop through its normal exit, so
    // the terminal is always restored.
    let quit = Arc::new(AtomicBool::new(false));
    let signals = {
        let quit = quit.clone();
        tokio::spawn(async move {
            stop_signal().await;
            quit.store(true, Ordering::Relaxed);
        })
    };
    let ui = tokio::task::spawn_blocking(move || ui_loop(shared, refresh, quit));
    let res = ui.await.context("dashboard thread")?;
    poller.abort();
    signals.abort();
    res
}

async fn stop_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let other = async {
        use tokio::signal::unix::{signal, SignalKind};
        match (
            signal(SignalKind::terminate()),
            signal(SignalKind::hangup()),
        ) {
            (Ok(mut t), Ok(mut h)) => {
                tokio::select! { _ = t.recv() => {}, _ = h.recv() => {} }
            }
            _ => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let other = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = other => {} }
}

/// The key to send: the one given for this command, else the local server's own key, but only
/// when the address also comes from the local configuration.
fn top_key(a: &TopArgs, configured: Option<&str>) -> Option<String> {
    a.api_key
        .clone()
        .or_else(|| {
            a.url
                .is_none()
                .then(|| configured.map(str::to_string))
                .flatten()
        })
        .filter(|k| !k.is_empty())
}

struct Fetcher {
    http: reqwest::Client,
    endpoint: String,
    key: Option<String>,
}

impl Fetcher {
    async fn raw(&self) -> anyhow::Result<Value> {
        let mut req = self.http.get(&self.endpoint);
        if let Some(k) = &self.key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().await.with_context(|| {
            format!(
                "no LLMario server answered at {} (start one with `llmario serve`, or pass --url)",
                self.endpoint
            )
        })?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            anyhow::bail!(
                "{} has no runtime monitor (an older LLMario?)",
                self.endpoint
            );
        }
        if status == reqwest::StatusCode::UNAUTHORIZED {
            anyhow::bail!("the server needs an API key: set LLMARIO_TOP_API_KEY or pass --api-key");
        }
        if !status.is_success() {
            anyhow::bail!("{} answered HTTP {status}", self.endpoint);
        }
        Ok(resp.json().await?)
    }

    async fn snapshot(&self) -> Result<Snapshot, String> {
        let v = self.raw().await.map_err(|e| format!("{e:#}"))?;
        serde_json::from_value(v).map_err(|e| format!("unexpected monitor data: {e}"))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    Overview,
    Requests,
    Engine,
    Hardware,
}

impl View {
    const ALL: [View; 4] = [View::Overview, View::Requests, View::Engine, View::Hardware];
    fn name(self) -> &'static str {
        match self {
            View::Overview => "Overview",
            View::Requests => "Requests",
            View::Engine => "Engine",
            View::Hardware => "Hardware",
        }
    }
}

/// A rolling series for a sparkline.
#[derive(Default)]
pub struct Series(VecDeque<u64>);

impl Series {
    fn push(&mut self, v: Option<f64>) {
        if self.0.len() == HISTORY {
            self.0.pop_front();
        }
        // Tenths, so small values still draw.
        self.0.push_back(
            v.filter(|v| v.is_finite())
                .map_or(0, |v| (v * 10.0).max(0.0) as u64),
        );
    }
    fn tail(&self, width: u16) -> Vec<u64> {
        let n = width as usize;
        self.0
            .iter()
            .skip(self.0.len().saturating_sub(n))
            .copied()
            .collect()
    }
}

pub struct App {
    pub url: String,
    pub view: View,
    pub frozen: bool,
    pub help: bool,
    /// Which group of three engines the Engine view shows.
    pub engine_page: usize,
    pub snap: Option<Snapshot>,
    pub error: Option<String>,
    pub last_ok: Option<Instant>,
    speed: Series,
    gpu: Series,
    gpu_mem: Series,
    cpu: Series,
    disk: Series,
    ram: Series,
    power: Series,
    reuse: Series,
}

impl App {
    pub fn new(url: String) -> Self {
        Self {
            url,
            view: View::Overview,
            frozen: false,
            help: false,
            engine_page: 0,
            snap: None,
            error: None,
            last_ok: None,
            speed: Series::default(),
            gpu: Series::default(),
            gpu_mem: Series::default(),
            cpu: Series::default(),
            disk: Series::default(),
            ram: Series::default(),
            power: Series::default(),
            reuse: Series::default(),
        }
    }

    pub fn apply(&mut self, r: Result<Snapshot, String>) {
        match r {
            Err(e) => self.error = Some(e),
            Ok(_) if self.frozen => self.error = None,
            Ok(s) => {
                self.error = None;
                self.last_ok = Some(Instant::now());
                let g = s.system.gpus.first();
                self.speed.push(Some(s.speed_tok_s));
                self.gpu
                    .push(g.and_then(|g| g.utilization_percent).map(f64::from));
                self.gpu_mem
                    .push(g.and_then(|g| g.memory_used_bytes).map(|b| b as f64 / GIB));
                self.cpu.push(s.system.cpu_percent.map(f64::from));
                self.disk.push(Some(engine_disk_read(&s) as f64 / MIB));
                self.ram.push(Some(s.system.memory_used_bytes as f64 / GIB));
                self.power.push(g.and_then(|g| g.power_w).map(f64::from));
                self.reuse.push(s.recent.first().map(reuse_percent));
                self.snap = Some(s);
            }
        }
    }

    /// Handles one key; returns false to quit.
    pub fn key(&mut self, code: KeyCode, mods: KeyModifiers) -> KeyAction {
        match code {
            // Esc closes the help first; a second Esc quits.
            KeyCode::Esc if self.help => self.help = false,
            KeyCode::Char('q') | KeyCode::Esc => return KeyAction::Quit,
            KeyCode::Char('c') if mods.contains(KeyModifiers::CONTROL) => return KeyAction::Quit,
            KeyCode::Char(c @ '1'..='4') => self.view = View::ALL[(c as u8 - b'1') as usize],
            KeyCode::Tab | KeyCode::Right => {
                let i = View::ALL.iter().position(|v| *v == self.view).unwrap_or(0);
                self.view = View::ALL[(i + 1) % View::ALL.len()];
            }
            KeyCode::BackTab | KeyCode::Left => {
                let i = View::ALL.iter().position(|v| *v == self.view).unwrap_or(0);
                self.view = View::ALL[(i + View::ALL.len() - 1) % View::ALL.len()];
            }
            KeyCode::Char(' ') => self.frozen = !self.frozen,
            KeyCode::Char('?') | KeyCode::Char('h') => self.help = !self.help,
            KeyCode::Char('r') => return KeyAction::Refresh,
            KeyCode::PageDown | KeyCode::Char(']') => self.engine_page += 1,
            KeyCode::PageUp | KeyCode::Char('[') => {
                self.engine_page = self.engine_page.saturating_sub(1)
            }
            _ => {}
        }
        KeyAction::None
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum KeyAction {
    None,
    Quit,
    Refresh,
}

fn ui_loop(
    shared: Arc<Mutex<App>>,
    refresh: Arc<tokio::sync::Notify>,
    quit: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let mut terminal = ratatui::try_init().context("cannot open the terminal (is this a TTY?)")?;
    let result = (|| -> anyhow::Result<()> {
        loop {
            if quit.load(Ordering::Relaxed) {
                return Ok(());
            }
            terminal.draw(|f| draw(f, &shared.lock().unwrap()))?;
            if event::poll(Duration::from_millis(200))? {
                if let Event::Key(k) = event::read()? {
                    if k.kind != KeyEventKind::Press {
                        continue;
                    }
                    match shared.lock().unwrap().key(k.code, k.modifiers) {
                        KeyAction::Quit => return Ok(()),
                        KeyAction::Refresh => refresh.notify_one(),
                        KeyAction::None => {}
                    }
                }
            }
        }
    })();
    ratatui::restore();
    result
}

// ---------- formatting ----------

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / GIB)
}

fn rate(bps: u64) -> String {
    let b = bps as f64;
    if b >= GIB {
        format!("{:.2} GB/s", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MB/s", b / MIB)
    } else {
        format!("{:.0} KB/s", b / 1024.0)
    }
}

/// Context sizes: 131072 → "128K", 5728 → "5.6K", 4096 → "4K".
fn ktok(n: u64) -> String {
    let k = n as f64 / 1024.0;
    if n >= 10 * 1024 {
        format!("{}K", k.round() as u64)
    } else if n >= 1024 {
        format!("{k:.1}K").replace(".0K", "K")
    } else {
        n.to_string()
    }
}

/// Tokens used past the planned per-request window, when any. Engines that do not enforce the
/// window (MLX) can run past it, because the gateway checks prompts by an estimate.
fn over_plan(used: u64, planned: u64) -> Option<u64> {
    (planned > 0 && used > planned).then(|| used - planned)
}

/// 28693 → "28,693".
fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn opt<T>(v: Option<T>, f: impl Fn(T) -> String) -> String {
    v.map(f).unwrap_or_else(|| "n/a".into())
}

fn duration(ms: f64) -> String {
    // Round to whole seconds before splitting, so 119.7 s is "2m00s", not "1m60s".
    let secs = (ms / 1000.0).round() as u64;
    if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else if ms >= 1000.0 {
        format!("{:.1} s", ms / 1000.0)
    } else {
        format!("{ms:.0} ms")
    }
}

fn clock(unix_ms: u64) -> String {
    use chrono::TimeZone;
    chrono::Local
        .timestamp_millis_opt(unix_ms as i64)
        .single()
        .map(|t| t.format("%H:%M:%S").to_string())
        .unwrap_or_else(|| "--:--:--".into())
}

fn reuse_percent(r: &RecentRequest) -> f64 {
    match r.cached_tokens {
        Some(c) if r.prompt_tokens > 0 => c as f64 * 100.0 / r.prompt_tokens as f64,
        _ => 0.0,
    }
}

fn engine_disk_read(s: &Snapshot) -> u64 {
    s.engines.iter().filter_map(|e| e.disk_read_bps).sum()
}

fn status_style(outcome: &str) -> Style {
    match outcome {
        "ok" => Style::new().fg(ACCENT),
        "cancelled" => Style::new().fg(Color::Yellow),
        _ => Style::new().fg(Color::Red),
    }
}

/// Compact tables say "Failed"; the Requests view shows the error code.
fn status_word(outcome: &str, full: bool) -> &str {
    match outcome {
        "ok" => "Done",
        "cancelled" => "Stopped",
        other if full => other,
        _ => "Failed",
    }
}

fn stat<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    v.pointer(path)
}

fn stat_f(v: &Value, path: &str) -> Option<f64> {
    stat(v, path).and_then(Value::as_f64)
}

// ---------- drawing ----------

pub fn draw(f: &mut Frame, app: &App) {
    let [head, tabs, body, foot] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(2),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(f.area());
    draw_header(f, head, app);
    draw_tabs(f, tabs, app);
    match (&app.snap, app.view) {
        (None, _) => {
            let msg = app
                .error
                .clone()
                .unwrap_or_else(|| format!("Connecting to {}…", app.url));
            f.render_widget(
                Paragraph::new(msg)
                    .wrap(Wrap { trim: true })
                    .block(Block::bordered().title(" LLMario runtime monitor ")),
                body,
            );
        }
        (Some(s), View::Overview) => draw_overview(f, body, app, s),
        (Some(s), View::Requests) => draw_requests(f, body, s),
        (Some(s), View::Engine) => draw_engines(f, body, s, app.engine_page),
        (Some(s), View::Hardware) => draw_hardware(f, body, s),
    }
    draw_footer(f, foot, app);
    if app.help {
        draw_help(f, body);
    }
}

fn first_engine(s: &Snapshot) -> Option<&EngineView> {
    s.engines.first()
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let mut spans = vec![
        Span::styled(
            "LLMario",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(app.view.name(), Style::new().fg(ACCENT)),
        Span::raw("  "),
        Span::styled(app.url.clone(), Style::new().fg(ACCENT)),
    ];
    if let Some(e) = app.snap.as_ref().and_then(first_engine) {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            e.model.clone(),
            Style::new().add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw(format!(" {}", ktok(e.context_per_slot as u64))));
        if let Some(n) = app
            .snap
            .as_ref()
            .map(|s| s.engines.len())
            .filter(|n| *n > 1)
        {
            spans.push(Span::styled(
                format!(" (+{} more)", n - 1),
                Style::new().fg(DIM),
            ));
        }
    }
    let right = match (&app.error, app.frozen, &app.snap) {
        (Some(_), _, _) => Span::styled("● Disconnected", Style::new().fg(Color::Red)),
        (_, true, _) => Span::styled("❚❚ Frozen", Style::new().fg(Color::Yellow)),
        (_, _, Some(s)) => Span::styled(
            format!("● {}", title_case(&s.state)),
            Style::new().fg(state_color(&s.state)),
        ),
        _ => Span::styled("○ Connecting", Style::new().fg(DIM)),
    };
    let [l, r] = Layout::horizontal([Constraint::Min(0), Constraint::Length(18)]).areas(area);
    f.render_widget(Paragraph::new(Line::from(spans)), l);
    f.render_widget(Paragraph::new(Line::from(right).right_aligned()), r);
}

fn title_case(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
        .unwrap_or_default()
}

fn state_color(state: &str) -> Color {
    match state {
        "generating" | "reading" | "running" => ACCENT,
        "queued" => Color::Yellow,
        "error" => Color::Red,
        _ => DIM,
    }
}

fn draw_tabs(f: &mut Frame, area: Rect, app: &App) {
    let mut spans = Vec::new();
    for (i, v) in View::ALL.iter().enumerate() {
        let label = format!("{} {}", i + 1, v.name());
        if *v == app.view {
            spans.push(Span::styled(
                format!("[{label}]"),
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::raw(format!(" {label} ")));
        }
        spans.push(Span::raw(" "));
    }
    f.render_widget(
        Paragraph::new(Line::from(spans)).block(
            Block::new()
                .borders(ratatui::widgets::Borders::BOTTOM)
                .border_style(Style::new().fg(DIM)),
        ),
        area,
    );
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let mut text =
        "1-4 views   ←/→ switch   Space freeze   r refresh   ? help   q quit".to_string();
    if let Some(e) = &app.error {
        if app.snap.is_some() {
            text = format!("{e}  (showing the last data)");
        }
    }
    f.render_widget(Paragraph::new(text).style(Style::new().fg(DIM)), area);
}

fn draw_help(f: &mut Frame, body: Rect) {
    let w = body.width.min(66);
    let h = body.height.min(14);
    let area = Rect::new(
        body.x + (body.width - w) / 2,
        body.y + (body.height - h) / 2,
        w,
        h,
    );
    let text = vec![
        Line::from("LLMario runtime monitor"),
        Line::from(""),
        Line::from("1 Overview   live speed, GPU, CPU, memory, disk, recent requests"),
        Line::from("2 Requests   requests in flight and the last 64, with timings"),
        Line::from("3 Engine     each loaded model: plan, process, counters (PgUp/PgDn)"),
        Line::from("4 Hardware   machine, memory, swap, GPUs, processes"),
        Line::from(""),
        Line::from("Space freezes the display; r fetches now; q quits."),
        Line::from("Quitting switches the monitor off; the models keep running."),
        Line::from("n/a: this machine or engine does not report the value."),
    ];
    f.render_widget(ratatui::widgets::Clear, area);
    f.render_widget(
        Paragraph::new(text).block(
            Block::bordered()
                .title(" Help ")
                .border_style(Style::new().fg(ACCENT)),
        ),
        area,
    );
}

/// A tile: title in the border, a bold value, a dim subtitle, and a sparkline or bar.
struct Tile<'a> {
    title: &'a str,
    value: String,
    sub: String,
    graph: TileGraph<'a>,
}

enum TileGraph<'a> {
    Spark(&'a Series),
    Bar(Option<f64>, Color),
}

fn draw_tile(f: &mut Frame, area: Rect, t: Tile<'_>) {
    let block = Block::bordered()
        .title(format!(" {} ", t.title))
        .border_style(Style::new().fg(DIM));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let [v, s, g] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
    ])
    .areas(inner);
    f.render_widget(
        Paragraph::new(t.value).style(Style::new().add_modifier(Modifier::BOLD)),
        v,
    );
    f.render_widget(Paragraph::new(t.sub).style(Style::new().fg(DIM)), s);
    match t.graph {
        TileGraph::Spark(series) => f.render_widget(
            Sparkline::default()
                .data(series.tail(g.width))
                .style(Style::new().fg(ACCENT)),
            g,
        ),
        TileGraph::Bar(ratio, color) => {
            if let Some(r) = ratio {
                f.render_widget(Paragraph::new(bar_line(r, g.width, color)), g);
            }
        }
    }
}

fn draw_overview(f: &mut Frame, area: Rect, app: &App, s: &Snapshot) {
    let [state, tiles, bottom] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(10),
        Constraint::Min(0),
    ])
    .areas(area);
    draw_state_bar(f, state, s);

    let gpu = s.system.gpus.first();
    let eng = first_engine(s);
    let rows = Layout::vertical([Constraint::Length(5), Constraint::Length(5)]).split(tiles);
    let cols = |r: Rect| Layout::horizontal([Constraint::Ratio(1, 4); 4]).split(r);
    let (r1, r2) = (cols(rows[0]), cols(rows[1]));

    let last = s.recent.first();
    let speed_sub = if s.live.iter().any(|l| l.phase == "generating") {
        "now".to_string()
    } else {
        opt(last.and_then(|r| r.decode_tok_s), |t| {
            format!("last {t:.1} tok/s")
        })
    };
    draw_tile(
        f,
        r1[0],
        Tile {
            title: "Speed",
            value: format!("{:.1} tok/s", s.speed_tok_s),
            sub: speed_sub,
            graph: TileGraph::Spark(&app.speed),
        },
    );
    draw_tile(
        f,
        r1[1],
        Tile {
            title: "GPU load",
            value: opt(gpu.and_then(|g| g.utilization_percent), |u| {
                format!("{u:.0} %")
            }),
            sub: gpu
                .and_then(|g| g.name.clone())
                .or(s.machine.gpu.clone())
                .unwrap_or_else(|| "no GPU reported".into()),
            graph: TileGraph::Spark(&app.gpu),
        },
    );
    let (mem_title, mem_value, mem_sub, mem_ratio) = match gpu {
        Some(g) if g.memory_total_bytes.is_some() => (
            "VRAM",
            format!(
                "{} / {}",
                gb(g.memory_used_bytes.unwrap_or(0)),
                gb(g.memory_total_bytes.unwrap_or(0))
            ),
            "dedicated GPU memory".to_string(),
            g.memory_used_bytes
                .zip(g.memory_total_bytes)
                .map(|(u, t)| u as f64 / t.max(1) as f64),
        ),
        Some(g) => (
            "GPU memory",
            opt(g.memory_used_bytes, gb),
            format!("unified · of {}", gb(s.system.memory_total_bytes)),
            g.memory_used_bytes
                .map(|u| u as f64 / s.system.memory_total_bytes.max(1) as f64),
        ),
        None => (
            "Engine memory",
            opt(eng.and_then(|e| e.memory_bytes), gb),
            opt(eng.map(|e| e.estimated_bytes), |b| {
                format!("planned {}", gb(b))
            }),
            eng.and_then(|e| e.memory_bytes)
                .map(|m| m as f64 / s.system.memory_total_bytes.max(1) as f64),
        ),
    };
    draw_tile(
        f,
        r1[2],
        Tile {
            title: mem_title,
            value: mem_value,
            sub: mem_sub,
            graph: TileGraph::Bar(mem_ratio, ACCENT),
        },
    );
    match gpu.and_then(|g| g.temperature_c) {
        Some(t) => draw_tile(
            f,
            r1[3],
            Tile {
                title: "GPU temp",
                value: format!("{t:.0} °C"),
                sub: String::new(),
                graph: TileGraph::Bar(
                    Some(t as f64 / 100.0),
                    if t >= 80.0 { Color::Red } else { Color::Yellow },
                ),
            },
        ),
        None => {
            let prefill = eng
                .and_then(|e| e.engine_stats.as_ref())
                .and_then(|v| stat_f(v, "/prefill_tok_s"));
            draw_tile(
                f,
                r1[3],
                Tile {
                    title: "Prompt reading",
                    value: opt(prefill, |p| format!("{p:.0} tok/s")),
                    sub: opt(last.and_then(|r| r.ttft_ms), |t| {
                        format!("first token {}", duration(t))
                    }),
                    graph: TileGraph::Bar(None, ACCENT),
                },
            );
        }
    }
    match gpu.and_then(|g| g.power_w) {
        Some(p) => draw_tile(
            f,
            r2[0],
            Tile {
                title: "Power",
                value: format!("{p:.0} W"),
                sub: opt(gpu.and_then(|g| g.power_limit_w), |l| {
                    format!("of {l:.0} W limit")
                }),
                graph: TileGraph::Spark(&app.power),
            },
        ),
        None => draw_tile(
            f,
            r2[0],
            Tile {
                title: "Cache reuse",
                value: opt(
                    last.and_then(|r| r.cached_tokens.map(|_| reuse_percent(r))),
                    |p| format!("{p:.0} %"),
                ),
                sub: opt(last.and_then(|r| r.cached_tokens), |c| {
                    format!("{} prompt tokens reused", thousands(c))
                }),
                graph: TileGraph::Spark(&app.reuse),
            },
        ),
    }
    draw_tile(
        f,
        r2[1],
        Tile {
            title: "Engine",
            value: eng.map_or("no model loaded".into(), |e| {
                let dev = e
                    .engine_stats
                    .as_ref()
                    .and_then(|v| stat(v, "/device"))
                    .and_then(Value::as_str)
                    .map(|d| format!(" · {d}"))
                    .unwrap_or_default();
                format!("{}{dev}", e.backend)
            }),
            sub: eng.map_or(String::new(), |e| {
                format!(
                    "{} · {} slot{}",
                    e.quantization
                        .clone()
                        .unwrap_or_else(|| e.format.clone().unwrap_or_default()),
                    e.parallel,
                    if e.parallel == 1 { "" } else { "s" }
                )
            }),
            graph: TileGraph::Bar(
                eng.map(|e| e.active_requests as f64 / e.parallel.max(1) as f64),
                ACCENT,
            ),
        },
    );
    draw_tile(
        f,
        r2[2],
        Tile {
            title: "CPU",
            value: opt(s.system.cpu_percent, |c| format!("{c:.0} %")),
            sub: format!(
                "{} cores · {} threads",
                s.system.physical_cores, s.system.logical_cores
            ),
            graph: TileGraph::Spark(&app.cpu),
        },
    );
    let write: u64 = s.engines.iter().filter_map(|e| e.disk_write_bps).sum();
    draw_tile(
        f,
        r2[3],
        Tile {
            title: "Disk read",
            value: rate(engine_disk_read(s)),
            sub: format!("engines · write {}", rate(write)),
            graph: TileGraph::Spark(&app.disk),
        },
    );

    let [left, right] =
        Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)])
            .spacing(3)
            .areas(bottom);
    draw_bars(f, left, s);
    draw_recent_table(f, right, s, false);
}

fn draw_state_bar(f: &mut Frame, area: Rect, s: &Snapshot) {
    let mut spans = vec![Span::styled(
        "Model state ",
        Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
    )];
    // `running` (non-streaming requests, phase unknown) is listed only while it is the state.
    let running = s.state == "running";
    for st in [
        "idle",
        "reading",
        "generating",
        "running",
        "queued",
        "error",
    ] {
        if st == "running" && !running {
            continue;
        }
        let word = title_case(st);
        if s.state == st {
            spans.push(Span::styled(
                format!("[{word}]"),
                Style::new()
                    .fg(state_color(st))
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::styled(format!(" {word} "), Style::new().fg(DIM)));
        }
        spans.push(Span::raw(" "));
    }
    if s.waiting > 0 {
        spans.push(Span::styled(
            format!("{} waiting", s.waiting),
            Style::new().fg(Color::Yellow),
        ));
    }
    let ctx = first_engine(s).map(|e| {
        let used = e.context_used.unwrap_or(0);
        let total = e.context_per_slot as u64;
        match over_plan(used, total) {
            Some(over) => (
                format!(
                    "{} / {} tokens · {} over plan",
                    thousands(used),
                    thousands(total),
                    thousands(over)
                ),
                Style::new().fg(Color::Yellow),
            ),
            None => (
                format!(
                    "{} / {} tokens · {:.0}%",
                    thousands(used),
                    thousands(total),
                    used as f64 * 100.0 / total.max(1) as f64
                ),
                Style::new().fg(DIM),
            ),
        }
    });
    let [l, r] = Layout::horizontal([Constraint::Min(0), Constraint::Length(44)])
        .areas(Rect { height: 1, ..area });
    f.render_widget(Paragraph::new(Line::from(spans)), l);
    if let Some((c, style)) = ctx {
        f.render_widget(
            Paragraph::new(Line::from(c).right_aligned()).style(style),
            r,
        );
    }
}

/// A solid bar over a dotted track.
fn bar_line(ratio: f64, width: u16, color: Color) -> Line<'static> {
    let w = width as usize;
    let filled = ((ratio.clamp(0.0, 1.0) * w as f64).round() as usize).min(w);
    Line::from(vec![
        Span::styled("█".repeat(filled), Style::new().fg(color)),
        Span::styled("░".repeat(w - filled), Style::new().fg(DIM)),
    ])
}

fn bar(f: &mut Frame, area: Rect, label: String, ratio: Option<f64>, color: Color) {
    let [l, g] = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);
    f.render_widget(Paragraph::new(label), l);
    if let Some(r) = ratio {
        f.render_widget(Paragraph::new(bar_line(r, g.width, color)), g);
    }
}

fn draw_bars(f: &mut Frame, area: Rect, s: &Snapshot) {
    let rows = Layout::vertical([Constraint::Length(3); 4]).split(area);
    let e = first_engine(s);
    if let Some(e) = e {
        let used = e.context_used.unwrap_or(0);
        let planned = e.context_per_slot as u64;
        let (label, color) = match over_plan(used, planned) {
            Some(over) => (
                format!(
                    "Context fill   {} / {}  {} over plan",
                    ktok(used),
                    ktok(planned),
                    ktok(over)
                ),
                Color::Yellow,
            ),
            None => (
                format!(
                    "Context fill   {} / {}  {:.0}%",
                    ktok(used),
                    ktok(planned),
                    used as f64 * 100.0 / planned.max(1) as f64
                ),
                ACCENT,
            ),
        };
        bar(
            f,
            rows[0],
            label,
            Some(used as f64 / planned.max(1) as f64),
            color,
        );
        bar(
            f,
            rows[1],
            format!(
                "Engine memory   {} · planned {}",
                opt(e.memory_bytes, gb),
                gb(e.estimated_bytes)
            ),
            e.memory_bytes
                .map(|m| m as f64 / s.system.memory_total_bytes.max(1) as f64),
            ACCENT,
        );
    } else {
        f.render_widget(
            Paragraph::new(
                "No model loaded. Send a request or run `llmario serve --preload <model>`.",
            )
            .style(Style::new().fg(DIM))
            .wrap(Wrap { trim: true }),
            rows[0],
        );
    }
    let m = &s.system;
    bar(
        f,
        rows[2],
        format!(
            "System RAM   {} / {}",
            gb(m.memory_used_bytes),
            gb(m.memory_total_bytes)
        ),
        Some(m.memory_used_bytes as f64 / m.memory_total_bytes.max(1) as f64),
        ACCENT,
    );
    let swap_ratio =
        (m.swap_total_bytes > 0).then(|| m.swap_used_bytes as f64 / m.swap_total_bytes as f64);
    bar(
        f,
        rows[3],
        format!(
            "Swap   {} / {}",
            gb(m.swap_used_bytes),
            gb(m.swap_total_bytes)
        ),
        swap_ratio,
        if m.swap_used_bytes > (1 << 30) {
            Color::Yellow
        } else {
            ACCENT
        },
    );
}

fn draw_recent_table(f: &mut Frame, area: Rect, s: &Snapshot, full: bool) {
    let mut header = vec![
        "TIME", "STATUS", "PROMPT", "REUSED", "OUTPUT", "TOK/S", "DURATION",
    ];
    if full {
        header.insert(6, "TTFT");
        header.push("MODEL");
    }
    let rows = s.recent.iter().map(|r| {
        let mut cells = vec![
            Cell::from(clock(r.at_unix_ms)),
            Cell::from(status_word(&r.outcome, full).to_string()).style(status_style(&r.outcome)),
            Cell::from(thousands(r.prompt_tokens)),
            Cell::from(opt(r.cached_tokens, thousands)),
            Cell::from(thousands(r.completion_tokens)),
            Cell::from(opt(r.decode_tok_s, |t| format!("{t:.1}"))),
            Cell::from(duration(r.duration_ms)),
        ];
        if full {
            cells.insert(6, Cell::from(opt(r.ttft_ms, duration)));
            cells.push(Cell::from(r.model.clone()));
        }
        Row::new(cells)
    });
    let mut widths = vec![
        Constraint::Length(9),
        Constraint::Length(if full { 18 } else { 8 }),
        Constraint::Length(9),
        Constraint::Length(9),
        Constraint::Length(8),
        Constraint::Length(7),
        Constraint::Length(9),
    ];
    if full {
        widths.insert(6, Constraint::Length(9));
        widths.push(Constraint::Min(10));
    }
    let title = if s.recent.is_empty() {
        " Recent requests (none yet) "
    } else {
        " Recent requests "
    };
    f.render_widget(
        Table::new(rows, widths)
            .header(Row::new(header).style(Style::new().fg(DIM)))
            .block(Block::new().title(Span::styled(
                title,
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
            ))),
        area,
    );
}

fn draw_requests(f: &mut Frame, area: Rect, s: &Snapshot) {
    let live_h = (s.live.len() as u16 + 3).min(area.height / 2).max(4);
    let [live, totals, recent] = Layout::vertical([
        Constraint::Length(live_h),
        Constraint::Length(2),
        Constraint::Min(0),
    ])
    .areas(area);
    let rows = s.live.iter().map(|l| {
        Row::new(vec![
            Cell::from(title_case(&l.phase)).style(Style::new().fg(ACCENT)),
            Cell::from(l.model.clone()),
            Cell::from(format!("≈{}", thousands(l.prompt_tokens_estimate))),
            Cell::from(thousands(l.generated)),
            Cell::from(opt(l.tok_s, |t| format!("{t:.1}"))),
            Cell::from(opt(l.ttft_ms, duration)),
            Cell::from(duration(l.elapsed_s * 1000.0)),
        ])
    });
    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(11),
                Constraint::Min(16),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(7),
                Constraint::Length(9),
                Constraint::Length(9),
            ],
        )
        .header(
            Row::new([
                "PHASE", "MODEL", "PROMPT", "OUTPUT", "TOK/S", "TTFT", "ELAPSED",
            ])
            .style(Style::new().fg(DIM)),
        )
        .block(Block::bordered().title(format!(
            " In flight: {}  ·  waiting: {} ",
            s.live.len(),
            s.waiting
        ))),
        live,
    );
    let t = &s.totals;
    f.render_widget(
        Paragraph::new(format!(
            "Since start: {} requests ({} done, {} stopped, {} failed) · prompt {} · reused {} · output {} tokens · up {}",
            t.requests,
            t.ok,
            t.cancelled,
            t.failed,
            thousands(t.prompt_tokens),
            thousands(t.cached_tokens),
            thousands(t.completion_tokens),
            duration(s.uptime_s * 1000.0)
        ))
        .style(Style::new().fg(DIM))
        .wrap(Wrap { trim: true }),
        totals,
    );
    draw_recent_table(f, recent, s, true);
}

fn kv(k: &str, v: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{k:<22}"), Style::new().fg(DIM)),
        Span::raw(v),
    ])
}

fn engine_lines(e: &EngineView, total_mem: u64) -> Vec<Line<'static>> {
    let mut l = vec![
        Line::from(Span::styled(
            e.model.clone(),
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        kv("Backend · pid", format!("{} · {}", e.backend, e.pid)),
        kv(
            "Family · architecture",
            format!(
                "{} · {}",
                e.family.clone().unwrap_or_else(|| "?".into()),
                e.architecture.clone().unwrap_or_else(|| "?".into())
            ),
        ),
        kv(
            "Format · quantization",
            format!(
                "{} · {}",
                e.format.clone().unwrap_or_else(|| "?".into()),
                e.quantization.clone().unwrap_or_else(|| "n/a".into())
            ),
        ),
        kv("Model file", opt(e.file_bytes, gb)),
        kv("Layers", opt(e.layers, |n| n.to_string())),
        kv(
            "Context",
            format!(
                "{} per slot × {} slot(s) · model max {}",
                thousands(e.context_per_slot as u64),
                e.parallel,
                opt(e.context_max, |c| thousands(c as u64))
            ),
        ),
        kv("Batch · micro-batch", format!("{} · {}", e.batch, e.ubatch)),
        kv("Profile", e.profile.clone()),
        kv(
            "Active requests",
            format!("{} of {}", e.active_requests, e.parallel),
        ),
        kv(
            "Load time · idle",
            format!(
                "{} · {}",
                duration(e.load_seconds * 1000.0),
                duration(e.idle_seconds * 1000.0)
            ),
        ),
        kv(
            "Memory (used · planned)",
            format!(
                "{} · {} ({})",
                opt(e.memory_bytes, gb),
                gb(e.estimated_bytes),
                opt(e.memory_bytes, |m| format!(
                    "{:.0}% of RAM",
                    m as f64 * 100.0 / total_mem.max(1) as f64
                ))
            ),
        ),
        kv(
            "CPU",
            opt(e.cpu_percent, |c| format!("{c:.0} % of one core")),
        ),
        kv(
            "Disk read · write",
            format!(
                "{} · {} (read {} since start)",
                opt(e.disk_read_bps, rate),
                opt(e.disk_write_bps, rate),
                opt(e.disk_read_total, gb)
            ),
        ),
    ];
    match &e.engine_stats {
        None => l.push(kv(
            "Engine counters",
            "not reported by this engine (built-in engine only)".into(),
        )),
        Some(v) => {
            let s = |p: &str| {
                stat(v, p)
                    .map(|x| x.to_string().trim_matches('"').to_string())
                    .unwrap_or_else(|| "n/a".into())
            };
            let f1 = |p: &str| opt(stat_f(v, p), |x| format!("{x:.1}"));
            l.push(Line::from(""));
            l.push(Line::from(Span::styled(
                "Built-in engine",
                Style::new().fg(ACCENT),
            )));
            l.push(kv(
                "Device · threads",
                format!("{} · {}", s("/device"), s("/threads")),
            ));
            l.push(kv("Kernels", s("/kernels")));
            l.push(kv(
                "Prefill · decode avg",
                format!("{} · {} tok/s", f1("/prefill_tok_s"), f1("/decode_tok_s")),
            ));
            l.push(kv(
                "Requests · cancelled",
                format!("{} · {}", s("/requests"), s("/cancelled")),
            ));
            l.push(kv(
                "Prompt · cached · output",
                format!(
                    "{} · {} · {}",
                    s("/prompt_tokens"),
                    s("/cached_prompt_tokens"),
                    s("/completion_tokens")
                ),
            ));
            l.push(kv(
                "Steps · max batch",
                format!("{} · {}", s("/steps"), s("/max_seqs_in_step")),
            ));
            l.push(kv(
                "Bytes read / token",
                opt(stat(v, "/bytes_per_token").and_then(Value::as_u64), |b| {
                    format!("{:.1} MB", b as f64 / MIB)
                }),
            ));
            l.push(kv(
                "KV disk tier",
                if stat(v, "/kv_disk/enabled").and_then(Value::as_bool) == Some(true) {
                    format!(
                        "on · {} saves · {} restores · {} in {} files",
                        s("/kv_disk/saves"),
                        s("/kv_disk/restores"),
                        opt(stat(v, "/kv_disk/bytes").and_then(Value::as_u64), gb),
                        s("/kv_disk/files")
                    )
                } else {
                    "off".into()
                },
            ));
            l.push(kv(
                "Memory pressure",
                format!(
                    "{} · relief {} · {} sheds",
                    s("/memory_pressure/level"),
                    s("/memory_pressure/relief"),
                    s("/memory_pressure/sheds")
                ),
            ));
            let paging =
                stat(v, "/paging/model_does_not_fit_memory").and_then(Value::as_bool) == Some(true);
            l.push(Line::from(vec![
                Span::styled(format!("{:<22}", "Paging"), Style::new().fg(DIM)),
                if paging {
                    Span::styled(
                        "YES: weights re-read from storage every token",
                        Style::new().fg(Color::Red),
                    )
                } else {
                    Span::raw("no")
                },
            ]));
            l.push(kv("Sleeping", s("/sleeping")));
        }
    }
    l
}

/// Engines shown side by side in the Engine view; more are paged with PgUp/PgDn.
const ENGINES_PER_PAGE: usize = 3;

fn draw_engines(f: &mut Frame, area: Rect, s: &Snapshot, page: usize) {
    if s.engines.is_empty() {
        f.render_widget(
            Paragraph::new("No model loaded. The engine view fills in when a model is loaded.")
                .block(Block::bordered().title(" Engine ")),
            area,
        );
        return;
    }
    let pages = s.engines.len().div_ceil(ENGINES_PER_PAGE);
    let page = page.min(pages - 1);
    let first = page * ENGINES_PER_PAGE;
    let shown = &s.engines[first..(first + ENGINES_PER_PAGE).min(s.engines.len())];
    let n = shown.len() as u32;
    let cols = Layout::horizontal((0..n).map(|_| Constraint::Ratio(1, n))).split(area);
    for (i, (e, c)) in shown.iter().zip(cols.iter()).enumerate() {
        let title = if s.engines.len() > 1 {
            let more = if pages > 1 {
                " · PgUp/PgDn for more"
            } else {
                ""
            };
            format!(" Engine {} of {}{more} ", first + i + 1, s.engines.len())
        } else {
            " Engine ".to_string()
        };
        f.render_widget(
            Paragraph::new(engine_lines(e, s.system.memory_total_bytes))
                .wrap(Wrap { trim: false })
                .block(
                    Block::bordered()
                        .title(title)
                        .border_style(Style::new().fg(DIM)),
                ),
            *c,
        );
    }
}

fn draw_hardware(f: &mut Frame, area: Rect, s: &Snapshot) {
    let m = &s.machine;
    let sys = &s.system;
    let mut l = vec![
        Line::from(Span::styled(
            "Machine",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        kv("CPU", m.cpu.clone()),
        kv(
            "Cores",
            format!(
                "{} physical · {} logical{}",
                sys.physical_cores,
                sys.logical_cores,
                match (m.performance_cores, m.efficiency_cores) {
                    (Some(p), Some(e)) => format!(" · {p} performance + {e} efficiency"),
                    _ => String::new(),
                }
            ),
        ),
        kv("OS · arch", format!("{} · {}", m.os, m.arch)),
        kv(
            "Memory",
            format!(
                "{}{}",
                gb(sys.memory_total_bytes),
                if m.unified_memory {
                    " unified (shared by CPU and GPU)"
                } else {
                    ""
                }
            ),
        ),
        kv(
            "Memory bandwidth",
            opt(m.memory_bandwidth_gbs, |b| {
                format!("{b:.0} GB/s (published)")
            }),
        ),
        kv("Runtime profile", m.profile.clone()),
        kv(
            "LLMario",
            format!("{} · up {}", s.version, duration(s.uptime_s * 1000.0)),
        ),
        Line::from(""),
        Line::from(Span::styled(
            "Now",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        kv("CPU use", opt(sys.cpu_percent, |c| format!("{c:.0} %"))),
        kv(
            "Load average (1 min)",
            opt(sys.load_average_1m, |l| format!("{l:.2}")),
        ),
        kv(
            "RAM used · available",
            format!(
                "{} · {}",
                gb(sys.memory_used_bytes),
                gb(sys.memory_available_bytes)
            ),
        ),
        kv(
            "Swap used",
            format!(
                "{} of {}",
                gb(sys.swap_used_bytes),
                gb(sys.swap_total_bytes)
            ),
        ),
    ];
    if sys.gpus.is_empty() {
        l.push(kv(
            "GPU",
            m.gpu.clone().unwrap_or_else(|| "none reported".into()),
        ));
    }
    for g in &sys.gpus {
        l.push(Line::from(""));
        l.push(Line::from(Span::styled(
            format!(
                "GPU: {}",
                g.name
                    .clone()
                    .or(m.gpu.clone())
                    .unwrap_or_else(|| "?".into())
            ),
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        )));
        l.push(kv(
            "Utilisation",
            opt(g.utilization_percent, |u| format!("{u:.0} %")),
        ));
        l.push(kv(
            "Memory in use",
            format!(
                "{}{}",
                opt(g.memory_used_bytes, gb),
                opt(g.memory_total_bytes, |t| format!(" of {}", gb(t)))
            ),
        ));
        l.push(kv(
            "Temperature",
            opt(g.temperature_c, |t| format!("{t:.0} °C")),
        ));
        l.push(kv(
            "Power",
            format!(
                "{}{}",
                opt(g.power_w, |p| format!("{p:.0} W")),
                opt(g.power_limit_w, |p| format!(" of {p:.0} W"))
            ),
        ));
        if let Some(p) = &g.pcie {
            l.push(kv("PCIe link", p.clone()));
        }
        l.push(kv("Read with", g.source.clone()));
        if g.source == "ioreg" {
            l.push(Line::from(Span::styled(
                "Temperature and power need `sudo powermetrics` on macOS; not read.",
                Style::new().fg(DIM),
            )));
        }
    }
    if let Some(p) = &s.gateway {
        l.push(Line::from(""));
        l.push(kv(
            "Gateway process",
            format!(
                "pid {} · {} · CPU {}",
                p.pid,
                opt(p.memory_bytes, gb),
                opt(p.cpu_percent, |c| format!("{c:.0} %"))
            ),
        ));
    }
    f.render_widget(
        Paragraph::new(l).wrap(Wrap { trim: false }).block(
            Block::bordered()
                .title(" Hardware ")
                .border_style(Style::new().fg(DIM)),
        ),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    fn sample() -> Snapshot {
        let engine_stats = json!({"device": "metal", "threads": 12, "kernels": "neon", "prefill_tok_s": 850.5,
                    "decode_tok_s": 88.1, "requests": 6, "cancelled": 0, "prompt_tokens": 1000,
                    "cached_prompt_tokens": 600, "completion_tokens": 400, "steps": 900, "max_seqs_in_step": 1,
                    "bytes_per_token": 4_000_000u64,
                    "kv_disk": {"enabled": true, "saves": 2, "restores": 1, "bytes": 1000, "files": 2},
                    "memory_pressure": {"level": "normal", "relief": true, "sheds": 0},
                    "paging": {"model_does_not_fit_memory": false}, "sleeping": false});
        let system = json!({"cpu_percent": 61.0, "logical_cores": 16, "physical_cores": 16, "load_average_1m": 3.5,
                "memory_total_bytes": 68_719_476_736u64, "memory_used_bytes": 40_000_000_000u64,
                "memory_available_bytes": 28_000_000_000u64, "swap_total_bytes": 0, "swap_used_bytes": 0,
                "gpus": [{"name": "Apple M4 Max", "source": "ioreg", "utilization_percent": 97.0,
                    "memory_used_bytes": 1_517_174_784u64, "memory_total_bytes": null, "temperature_c": null,
                    "power_w": null, "power_limit_w": null, "pcie": null}],
                "processes": []});
        serde_json::from_value(json!({
            "version": "0.2.2", "at_unix_ms": 1_790_000_000_000u64, "uptime_s": 75.0,
            "state": "generating", "waiting": 0, "speed_tok_s": 92.3,
            "live": [{"model": "qwen3-8b", "phase": "generating", "elapsed_s": 2.0, "ttft_ms": 310.0,
                      "prompt_tokens_estimate": 28000, "generated": 120, "tok_s": 92.3}],
            "engines": [{"model": "qwen3-8b", "backend": "native", "pid": 4242, "family": "qwen3",
                "format": "gguf", "architecture": "qwen3", "quantization": "Q4_K_M", "file_bytes": 5_000_000_000u64,
                "layers": 36, "context_max": 131072, "context_per_slot": 131072, "parallel": 1,
                "batch": 2048, "ubatch": 512, "profile": "latency", "active_requests": 1,
                "context_used": 28120, "load_seconds": 3.2, "idle_seconds": 0.0,
                "estimated_bytes": 7_000_000_000u64, "memory_bytes": 6_500_000_000u64, "cpu_percent": 61.0,
                "disk_read_bps": 62_600_000u64, "disk_write_bps": 100_000u64, "disk_read_total": 9_000_000_000u64,
                "engine_stats": engine_stats}],
            "recent": [{"at_unix_ms": 1_790_000_000_000u64, "model": "qwen3-8b", "outcome": "ok",
                "prompt_tokens": 28693, "cached_tokens": 28140, "completion_tokens": 346, "ttft_ms": 400.0,
                "duration_ms": 7000.0, "decode_tok_s": 73.5},
                {"at_unix_ms": 1_789_999_990_000u64, "model": "qwen3-8b", "outcome": "context_overflow",
                "prompt_tokens": 0, "cached_tokens": null, "completion_tokens": 0, "ttft_ms": null,
                "duration_ms": 12.0, "decode_tok_s": null}],
            "totals": {"requests": 2, "ok": 1, "cancelled": 0, "failed": 1, "prompt_tokens": 28693,
                "cached_tokens": 28140, "completion_tokens": 346},
            "machine": {"cpu": "Apple M4 Max", "os": "macOS 27", "arch": "aarch64", "performance_cores": 12,
                "efficiency_cores": 4, "unified_memory": true, "memory_bandwidth_gbs": 546.0,
                "gpu": "Apple M4 Max", "gpu_api": "metal", "profile": "latency"},
            "gateway": {"pid": 1, "cpu_percent": 1.0, "memory_bytes": 50_000_000u64, "disk_read_bps": 0,
                "disk_write_bps": 0, "disk_read_total": 0, "disk_write_total": 0},
            "system": system
        }))
        .unwrap()
    }

    fn render(app: &App, w: u16, h: u16) -> String {
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| draw(f, app)).unwrap();
        let buf = t.backend().buffer().clone();
        let mut s = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                s.push_str(buf[(x, y)].symbol());
            }
            s.push('\n');
        }
        s
    }

    #[test]
    fn every_view_renders_the_snapshot() {
        let mut app = App::new("http://127.0.0.1:11500".into());
        app.apply(Ok(sample()));
        app.apply(Ok(sample()));
        let o = render(&app, 120, 34);
        for want in [
            "LLMario",
            "[Generating]",
            "92.3 tok/s",
            "97 %",
            "Recent requests",
            "28,693",
            "Done",
            "Failed",
            "qwen3-8b 128K",
            "Context fill",
        ] {
            assert!(o.contains(want), "overview lacks {want:?}:\n{o}");
        }
        app.key(KeyCode::Char('2'), KeyModifiers::NONE);
        let r = render(&app, 120, 34);
        assert!(
            r.contains("In flight: 1") && r.contains("≈28,000") && r.contains("TTFT"),
            "{r}"
        );
        assert!(r.contains("context_overflow"), "{r}");
        app.key(KeyCode::Char('3'), KeyModifiers::NONE);
        let e = render(&app, 120, 50);
        for want in [
            "Q4_K_M",
            "Built-in engine",
            "metal · 12",
            "850.5",
            "on · 2 saves",
        ] {
            assert!(e.contains(want), "engine view lacks {want:?}:\n{e}");
        }
        app.key(KeyCode::Char('4'), KeyModifiers::NONE);
        let h = render(&app, 120, 50);
        assert!(
            h.contains("Apple M4 Max") && h.contains("546 GB/s") && h.contains("ioreg"),
            "{h}"
        );
    }

    #[test]
    fn small_terminals_and_missing_data_do_not_panic() {
        let mut app = App::new("http://x".into());
        render(&app, 20, 5);
        app.apply(Err("no LLMario server answered".into()));
        assert!(render(&app, 80, 24).contains("no LLMario server answered"));
        let mut s = sample();
        s.engines.clear();
        s.system.gpus.clear();
        s.recent.clear();
        s.live.clear();
        s.state = "idle".into();
        app.apply(Ok(s));
        for v in ['1', '2', '3', '4'] {
            app.key(KeyCode::Char(v), KeyModifiers::NONE);
            render(&app, 60, 20);
            render(&app, 10, 4);
        }
        app.help = true;
        render(&app, 30, 8);
    }

    #[test]
    fn keys_switch_views_freeze_and_quit() {
        let mut app = App::new("u".into());
        assert_eq!(app.key(KeyCode::Right, KeyModifiers::NONE), KeyAction::None);
        assert_eq!(app.view, View::Requests);
        app.key(KeyCode::Left, KeyModifiers::NONE);
        app.key(KeyCode::Left, KeyModifiers::NONE);
        assert_eq!(app.view, View::Hardware);
        app.key(KeyCode::Char(' '), KeyModifiers::NONE);
        app.apply(Ok(sample()));
        assert!(app.snap.is_none(), "frozen: new data is not shown");
        app.key(KeyCode::Char(' '), KeyModifiers::NONE);
        app.apply(Ok(sample()));
        assert!(app.snap.is_some());
        assert_eq!(
            app.key(KeyCode::Char('r'), KeyModifiers::NONE),
            KeyAction::Refresh
        );
        assert_eq!(
            app.key(KeyCode::Char('q'), KeyModifiers::NONE),
            KeyAction::Quit
        );
        assert_eq!(
            app.key(KeyCode::Char('c'), KeyModifiers::CONTROL),
            KeyAction::Quit
        );
    }

    #[test]
    fn engines_past_three_are_paged_and_esc_closes_help_first() {
        let mut app = App::new("u".into());
        let mut s = sample();
        for i in 2..=5 {
            let mut e = s.engines[0].clone();
            e.model = format!("model-{i}");
            s.engines.push(e);
        }
        app.apply(Ok(s));
        app.key(KeyCode::Char('3'), KeyModifiers::NONE);
        let p1 = render(&app, 150, 50);
        assert!(
            p1.contains("Engine 1 of 5") && !p1.contains("model-4"),
            "{p1}"
        );
        app.key(KeyCode::PageDown, KeyModifiers::NONE);
        let p2 = render(&app, 150, 50);
        assert!(
            p2.contains("Engine 4 of 5") && p2.contains("model-5"),
            "{p2}"
        );
        app.key(KeyCode::PageDown, KeyModifiers::NONE);
        assert!(
            render(&app, 150, 50).contains("model-5"),
            "past the end stays on the last page"
        );
        app.key(KeyCode::Char('?'), KeyModifiers::NONE);
        assert_eq!(app.key(KeyCode::Esc, KeyModifiers::NONE), KeyAction::None);
        assert!(!app.help);
        assert_eq!(app.key(KeyCode::Esc, KeyModifiers::NONE), KeyAction::Quit);
    }

    #[test]
    fn running_state_and_api_key_scope() {
        let mut app = App::new("u".into());
        let mut s = sample();
        s.state = "running".into();
        app.apply(Ok(s));
        assert!(render(&app, 120, 34).contains("[Running]"));
        let mut s = sample();
        s.state = "idle".into();
        app.apply(Ok(s));
        assert!(!render(&app, 120, 34).contains("Running"));

        let args = |url: Option<&str>, key: Option<&str>| TopArgs {
            url: url.map(String::from),
            api_key: key.map(String::from),
            interval_ms: 1000,
            json: false,
        };
        assert_eq!(
            top_key(&args(None, None), Some("local")).as_deref(),
            Some("local")
        );
        assert_eq!(
            top_key(&args(Some("http://10.0.0.7:11500"), None), Some("local")),
            None
        );
        assert_eq!(
            top_key(
                &args(Some("http://10.0.0.7:11500"), Some("theirs")),
                Some("local")
            )
            .as_deref(),
            Some("theirs")
        );
        assert_eq!(top_key(&args(None, Some("")), None), None);
    }

    #[test]
    fn context_past_the_planned_window_says_over_plan_not_a_percentage() {
        let mut app = App::new("u".into());
        let mut s = sample();
        s.engines[0].context_per_slot = 4096;
        s.engines[0].context_used = Some(5728);
        app.apply(Ok(s));
        let o = render(&app, 120, 34);
        assert!(o.contains("5,728 / 4,096 tokens · 1,632 over plan"), "{o}");
        assert!(
            o.contains("Context fill   5.6K / 4K  1.6K over plan"),
            "{o}"
        );
        assert!(!o.contains("140%"), "{o}");
    }

    #[test]
    fn formatting() {
        assert_eq!(thousands(28693), "28,693");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000_000), "1,000,000");
        assert_eq!(ktok(131072), "128K");
        assert_eq!(ktok(5728), "5.6K");
        assert_eq!(ktok(4096), "4K");
        assert_eq!(ktok(1632), "1.6K");
        assert_eq!(ktok(800), "800");
        assert_eq!(over_plan(5728, 4096), Some(1632));
        assert_eq!(over_plan(4096, 4096), None);
        assert_eq!(over_plan(10, 0), None);
        assert_eq!(rate(62_600_000), "59.7 MB/s");
        assert_eq!(duration(169_900.0), "2m50s");
        assert_eq!(duration(119_700.0), "2m00s");
        assert_eq!(duration(59_700.0), "1m00s");
        assert_eq!(duration(4800.0), "4.8 s");
    }
}
