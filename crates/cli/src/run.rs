//! `llmario run`: terminal chat through the same gateway path as `serve`.

use crate::{util, RuntimeArgs};
use futures::StreamExt;
use serde_json::{json, Value};
use std::io::{BufRead, IsTerminal, Write};
use std::time::Instant;

pub struct RunOpts {
    pub model: String,
    pub prompt: String,
    pub system: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub hide_reasoning: bool,
    pub rt: RuntimeArgs,
}

pub async fn run(o: RunOpts) -> anyhow::Result<()> {
    let (paths, cfg) = util::load_config(&o.rt)?;
    let sup = util::supervisor(cfg, paths).await?;
    let result = session(&sup, &o).await;
    sup.shutdown().await;
    result
}

async fn session(
    sup: &std::sync::Arc<llmario_supervisor::Supervisor>,
    o: &RunOpts,
) -> anyhow::Result<()> {
    util::load_with_report(sup, &o.model).await?;
    let (addr, _server) = llmario_api::spawn_ephemeral(sup.clone()).await?;
    let url = format!("http://{addr}/v1/chat/completions");
    let http = reqwest::Client::builder().no_proxy().build()?;
    let mut history: Vec<Value> = Vec::new();
    if let Some(s) = &o.system {
        history.push(json!({"role": "system", "content": s}));
    }

    if !o.prompt.is_empty() {
        history.push(json!({"role": "user", "content": o.prompt}));
        turn(&http, &url, &o.model, &history, o).await?;
        return Ok(());
    }

    let interactive = std::io::stdin().is_terminal();
    if interactive {
        eprintln!(
            "\nChat with {} — /reset clears history, /exit or Ctrl-D quits.",
            o.model
        );
    }
    let stdin = std::io::stdin();
    loop {
        if interactive {
            eprint!("\n› ");
            std::io::stderr().flush().ok();
        }
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                history.retain(|m| m["role"] == "system");
                eprintln!("(history cleared)");
                continue;
            }
            _ => {}
        }
        history.push(json!({"role": "user", "content": line}));
        match turn(&http, &url, &o.model, &history, o).await {
            Ok(answer) => history.push(json!({"role": "assistant", "content": answer})),
            Err(e) => {
                history.pop();
                eprintln!("error: {e:#}");
            }
        }
    }
    Ok(())
}

async fn turn(
    http: &reqwest::Client,
    url: &str,
    model: &str,
    history: &[Value],
    o: &RunOpts,
) -> anyhow::Result<String> {
    let mut body = json!({
        "model": model,
        "messages": history,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(m) = o.max_tokens {
        body["max_tokens"] = json!(m);
    }
    if let Some(t) = o.temperature {
        body["temperature"] = json!(t);
    }
    let start = Instant::now();
    let resp = http.post(url).json(&body).send().await?;
    if !resp.status().is_success() {
        let v: Value = resp.json().await.unwrap_or_default();
        anyhow::bail!(
            "{}",
            v.pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("request failed")
        );
    }
    let color = std::io::stdout().is_terminal();
    let mut out = std::io::stdout();
    let mut answer = String::new();
    let (mut first, mut last, mut n) = (None::<Instant>, None::<Instant>, 0u64);
    let mut usage = None;
    let mut in_reasoning = false;
    let mut buf = Vec::new();
    let mut stream = resp.bytes_stream();
    'outer: while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk?);
        while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if data == "[DONE]" {
                break 'outer;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(err) = v.pointer("/error/message").and_then(Value::as_str) {
                anyhow::bail!("{err}");
            }
            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                usage = u.get("completion_tokens").and_then(Value::as_u64);
            }
            let Some(delta) = v.pointer("/choices/0/delta") else {
                continue;
            };
            let reasoning = delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let content = delta.get("content").and_then(Value::as_str).unwrap_or("");
            if reasoning.is_empty() && content.is_empty() {
                continue;
            }
            let now = Instant::now();
            first.get_or_insert(now);
            last = Some(now);
            n += 1;
            if !reasoning.is_empty() && !o.hide_reasoning {
                if !in_reasoning && color {
                    write!(out, "\x1b[2m")?;
                }
                in_reasoning = true;
                write!(out, "{reasoning}")?;
            }
            if !content.is_empty() {
                if in_reasoning {
                    if color {
                        write!(out, "\x1b[0m")?;
                    }
                    writeln!(out)?;
                    in_reasoning = false;
                }
                write!(out, "{content}")?;
                answer.push_str(content);
            }
            out.flush()?;
        }
    }
    if in_reasoning && color {
        write!(out, "\x1b[0m")?;
    }
    writeln!(out)?;
    let tokens = usage.unwrap_or(n);
    if let (Some(f), Some(l)) = (first, last) {
        let dt = (l - f).as_secs_f64();
        let tps = if tokens > 1 && dt > 0.0 {
            (tokens - 1) as f64 / dt
        } else {
            0.0
        };
        eprintln!(
            "[TTFT {:.0} ms · {tps:.1} tok/s · {tokens} tokens]",
            (f - start).as_secs_f64() * 1000.0
        );
    }
    Ok(answer)
}
