//! End-to-end checks of a running `llmario-engine serve` process (macOS and Linux).
//!
//! Runs only when `LLMARIO_TEST_GGUF` is set; the first path in it is used (a small model keeps
//! this fast: CI uses Qwen3.5-0.8B Q4_0). `LLMARIO_E2E_DEVICE` picks the backend (default `cpu`;
//! `auto` uses Metal where available).
//!
//! Covers: readiness, plan and ledger endpoints, non-streaming and streaming chat with usage,
//! JSON-schema output, a forced tool call, cancellation on client disconnect, two concurrent
//! requests, a prompt longer than the context, and shutdown through `/engine/control`.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Engine {
    child: Child,
    port: u16,
    log_path: std::path::PathBuf,
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.log_path);
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Minimal HTTP/1.1 client (keeps the test free of extra dependencies). Returns the status and
/// the body, or reads the body line by line with `on_line` for SSE (return `false` to hang up).
fn http(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&Value>,
    mut on_line: Option<&mut dyn FnMut(&str) -> bool>,
) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(600))).unwrap();
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    )
    .unwrap();
    let mut r = BufReader::new(s);
    let mut status_line = String::new();
    r.read_line(&mut status_line).unwrap();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let mut chunked = false;
    loop {
        let mut h = String::new();
        r.read_line(&mut h).unwrap();
        if h == "\r\n" || h.is_empty() {
            break;
        }
        if h.to_ascii_lowercase()
            .starts_with("transfer-encoding: chunked")
        {
            chunked = true;
        }
    }
    let mut out = String::new();
    if chunked {
        // Decode chunks and feed complete lines to `on_line`.
        let mut pending = String::new();
        loop {
            let mut size_line = String::new();
            if r.read_line(&mut size_line).unwrap_or(0) == 0 {
                break;
            }
            let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
            if size == 0 {
                break;
            }
            let mut buf = vec![0u8; size + 2];
            r.read_exact(&mut buf).unwrap();
            let text = String::from_utf8_lossy(&buf[..size]).to_string();
            out.push_str(&text);
            pending.push_str(&text);
            while let Some(i) = pending.find('\n') {
                let line = pending[..i].trim_end_matches('\r').to_string();
                pending.drain(..=i);
                if let Some(f) = on_line.as_deref_mut() {
                    if !f(&line) {
                        return (status, out); // hang up mid-stream
                    }
                }
            }
        }
    } else {
        r.read_to_string(&mut out).unwrap();
    }
    (status, out)
}

fn get_json(port: u16, path: &str) -> Value {
    let (st, b) = http(port, "GET", path, None, None);
    assert_eq!(st, 200, "{path}: {b}");
    serde_json::from_str(&b).unwrap()
}

fn chat(port: u16, body: Value) -> (u16, Value) {
    let (st, b) = http(port, "POST", "/v1/chat/completions", Some(&body), None);
    (st, serde_json::from_str(&b).unwrap_or(Value::String(b)))
}

fn start() -> Option<Engine> {
    let model = std::env::var("LLMARIO_TEST_GGUF").ok()?;
    let model = model.split(',').next()?.to_string();
    let device = std::env::var("LLMARIO_E2E_DEVICE").unwrap_or_else(|_| "cpu".into());
    let port = free_port();
    // Engine logs go to a file so a startup failure can be shown in the panic message.
    let log_path = std::env::temp_dir().join(format!("llmario-serve-e2e-{port}.log"));
    let log = std::fs::File::create(&log_path).expect("create engine log");
    let child = Command::new(env!("CARGO_BIN_EXE_llmario-engine"))
        .args([
            "serve",
            "--model",
            &model,
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--ctx",
            "2048",
            "--threads",
            "4",
            "--device",
            &device,
            "--model-id",
            "e2e",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::from(log.try_clone().unwrap()))
        .spawn()
        .expect("spawn llmario-engine");
    let mut e = Engine {
        child,
        port,
        log_path: log_path.clone(),
    };
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(180) {
        if let Some(status) = e.child.try_wait().unwrap() {
            let text = std::fs::read_to_string(&log_path).unwrap_or_default();
            let tail: Vec<&str> = text.lines().rev().take(20).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            panic!(
                "engine exited during startup ({status}):\n{}",
                tail.join("\n")
            );
        }
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            let (st, _) = http(port, "GET", "/health", None, None);
            if st == 200 {
                return Some(e);
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    panic!("engine did not become ready within 180 s");
}

fn user(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

#[test]
fn serve_end_to_end() {
    let Some(mut e) = start() else {
        eprintln!("LLMARIO_TEST_GGUF not set; skipping");
        return;
    };
    let port = e.port;

    // Plan and ledger.
    let plan = get_json(port, "/engine/plan");
    eprintln!("plan: {}", plan["plan"]);
    assert_eq!(plan["plan"]["fits"], true);
    assert_eq!(plan["plan"]["kv_dtype"], "f16");
    let ledger = get_json(port, "/engine/ledger");
    assert!(ledger["devices"].as_array().is_some_and(|d| !d.is_empty()));
    let models = get_json(port, "/v1/models");
    let backend = models["data"][0]["engine"]["device"]
        .as_str()
        .unwrap_or("")
        .to_string();
    eprintln!(
        "backend: {backend}, kernels: {}",
        models["data"][0]["engine"]["kernels"]
    );
    let requested = std::env::var("LLMARIO_E2E_DEVICE").unwrap_or_else(|_| "cpu".into());
    if requested != "auto" {
        assert_eq!(backend, requested, "the engine ignored --device");
    }
    assert_eq!(models["data"][0]["engine"]["capabilities"]["tools"], true);

    // Non-streaming chat.
    let question = json!({"messages": user("What is the capital of France? One word."),
                          "temperature": 0, "max_tokens": 64, "enable_thinking": false});
    let (st, r) = chat(port, question.clone());
    assert_eq!(st, 200, "{r}");
    let content = r["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(
        content.to_lowercase().contains("paris"),
        "answer: {content:?}"
    );
    assert!(r["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(r["usage"]["completion_tokens"].as_u64().unwrap() > 0);

    // Streaming chat with usage: the same greedy request must stream exactly the same text.
    let mut streamed_req = question.clone();
    streamed_req["stream"] = json!(true);
    streamed_req["stream_options"] = json!({"include_usage": true});
    let mut text = String::new();
    let (mut usage, mut done, mut finish) = (false, false, String::new());
    let (st, _) = http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(&streamed_req),
        Some(&mut |line: &str| {
            if line == "data: [DONE]" {
                done = true;
            } else if let Some(d) = line.strip_prefix("data: ") {
                let v: Value = serde_json::from_str(d).unwrap();
                if v.get("usage").is_some() {
                    usage = true;
                }
                if let Some(c) = v
                    .pointer("/choices/0/delta/content")
                    .and_then(Value::as_str)
                {
                    text.push_str(c);
                }
                if let Some(f) = v
                    .pointer("/choices/0/finish_reason")
                    .and_then(Value::as_str)
                {
                    finish = f.to_string();
                }
            }
            true
        }),
    );
    assert_eq!(st, 200);
    assert!(
        done && usage,
        "stream must end with a usage chunk and [DONE]"
    );
    assert_eq!(
        text, content,
        "streamed and non-streamed greedy answers differ"
    );
    assert!(finish == "stop" || finish == "length", "finish: {finish}");

    // JSON-schema output.
    let schema = json!({"type": "object", "properties": {"city": {"type": "string"},
        "country": {"type": "string"}}, "required": ["city", "country"], "additionalProperties": false});
    let (st, r) = chat(
        port,
        json!({"messages": user("Name a capital city and its country."), "temperature": 0, "max_tokens": 96,
               "enable_thinking": false,
               "response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": schema}}}),
    );
    assert_eq!(st, 200, "{r}");
    let out: Value =
        serde_json::from_str(r["choices"][0]["message"]["content"].as_str().unwrap()).unwrap();
    assert!(
        out["city"].is_string() && out["country"].is_string(),
        "{out}"
    );

    // Forced tool call.
    let tools = json!([{"type": "function", "function": {"name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]);
    let (st, r) = chat(
        port,
        json!({"messages": user("What's the weather in Oslo?"), "tools": tools, "tool_choice": "required",
               "temperature": 0, "max_tokens": 128, "enable_thinking": false}),
    );
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["choices"][0]["finish_reason"], "tool_calls", "{r}");
    let call = &r["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    let args: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert!(args["city"].is_string(), "{args}");

    // Cancellation: hang up after three content chunks; the engine must notice.
    let before = get_json(port, "/engine/stats")["cancelled"]
        .as_u64()
        .unwrap();
    let mut chunks = 0;
    http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(
            &json!({"messages": user("Write a long story about the sea."), "temperature": 0,
                     "max_tokens": 1000, "stream": true, "enable_thinking": false}),
        ),
        Some(&mut |line: &str| {
            if line.contains("\"content\"") {
                chunks += 1;
            }
            chunks < 3
        }),
    );
    let t0 = Instant::now();
    loop {
        let now = get_json(port, "/engine/stats")["cancelled"]
            .as_u64()
            .unwrap();
        if now > before {
            break;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "generation was not cancelled after the client hung up"
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    // Two concurrent requests both complete.
    let a = std::thread::spawn(move || {
        chat(
            port,
            json!({"messages": user("Say hello."), "temperature": 0, "max_tokens": 24, "enable_thinking": false}),
        )
    });
    let b = std::thread::spawn(move || {
        chat(
            port,
            json!({"messages": user("Say goodbye."), "temperature": 0, "max_tokens": 24, "enable_thinking": false}),
        )
    });
    assert_eq!(a.join().unwrap().0, 200);
    assert_eq!(b.join().unwrap().0, 200);

    // Features the engine does not implement are refused, not silently ignored.
    let (st, r) = chat(
        port,
        json!({"messages": user("Hi"), "max_tokens": 4, "logprobs": true}),
    );
    assert_eq!(st, 400, "{r}");
    assert!(r.to_string().contains("logprobs"), "{r}");

    // A prompt longer than the context is refused with a clear error, and the engine stays up.
    let (st, r) = chat(
        port,
        json!({"messages": user(&"word ".repeat(3000)), "max_tokens": 8}),
    );
    assert_eq!(st, 400, "{r}");
    assert!(r.to_string().contains("context"), "{r}");
    assert_eq!(http(port, "GET", "/health", None, None).0, 200);

    // Shutdown through the control endpoint.
    let (st, _) = http(
        port,
        "POST",
        "/engine/control",
        Some(&json!({"action": "exit"})),
        None,
    );
    assert_eq!(st, 200);
    let t0 = Instant::now();
    loop {
        if let Some(status) = e.child.try_wait().unwrap() {
            assert!(status.success(), "engine exit status {status}");
            break;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "engine did not exit"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}
