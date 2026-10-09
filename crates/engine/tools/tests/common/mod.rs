//! A minimal HTTP/1.1 server on `std::net::TcpListener` for hermetic fetch tests.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What a path returns.
#[derive(Debug, Clone)]
pub enum Route {
    /// Fixed response.
    Static {
        status: u16,
        content_type: &'static str,
        body: Vec<u8>,
    },
    /// 302 to `location`.
    Redirect(String),
    /// Sleep, then 200 text.
    Slow(Duration),
}

impl Route {
    pub fn html(body: &str) -> Self {
        Route::Static {
            status: 200,
            content_type: "text/html; charset=utf-8",
            body: body.as_bytes().to_vec(),
        }
    }
    pub fn text(body: &str) -> Self {
        Route::Static {
            status: 200,
            content_type: "text/plain",
            body: body.as_bytes().to_vec(),
        }
    }
    pub fn json(body: &str) -> Self {
        Route::Static {
            status: 200,
            content_type: "application/json",
            body: body.as_bytes().to_vec(),
        }
    }
    pub fn bytes(content_type: &'static str, body: Vec<u8>) -> Self {
        Route::Static {
            status: 200,
            content_type,
            body,
        }
    }
    pub fn status(status: u16) -> Self {
        Route::Static {
            status,
            content_type: "text/plain",
            body: b"status".to_vec(),
        }
    }
}

/// A running server.
pub struct TestServer {
    pub addr: SocketAddr,
    pub hits: Arc<Mutex<Vec<String>>>,
}

impl TestServer {
    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn url(&self, host: &str, path: &str) -> String {
        format!("http://{host}:{}{path}", self.port())
    }

    pub fn hits(&self) -> Vec<String> {
        self.hits.lock().unwrap().clone()
    }
}

/// Start serving `routes` on 127.0.0.1 (ephemeral port). Unknown paths get 404.
pub fn spawn(routes: HashMap<String, Route>) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(Mutex::new(Vec::new()));
    let routes = Arc::new(routes);
    let hits2 = hits.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let routes = routes.clone();
            let hits = hits2.clone();
            std::thread::spawn(move || handle(stream, &routes, &hits));
        }
    });
    TestServer { addr, hits }
}

fn handle(mut stream: TcpStream, routes: &HashMap<String, Route>, hits: &Mutex<Vec<String>>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    let head = String::from_utf8_lossy(&buf);
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or("");
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();
    let host = lines
        .find_map(|l| {
            l.strip_prefix("Host: ")
                .or_else(|| l.strip_prefix("host: "))
        })
        .unwrap_or("")
        .to_string();
    let ua = head
        .lines()
        .find_map(|l| {
            l.strip_prefix("user-agent: ")
                .or_else(|| l.strip_prefix("User-Agent: "))
        })
        .unwrap_or("")
        .to_string();
    hits.lock().unwrap().push(format!("{host} {path} ua={ua}"));
    let path_only = path.split('?').next().unwrap_or("/").to_string();
    let response = match routes.get(&path_only) {
        Some(Route::Static {
            status,
            content_type,
            body,
        }) => build(*status, content_type, body),
        Some(Route::Redirect(loc)) => {
            let port = stream.local_addr().map(|a| a.port()).unwrap_or(0);
            let loc = loc.replace("{PORT}", &port.to_string());
            format!("HTTP/1.1 302 Found\r\nLocation: {loc}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").into_bytes()
        }
        Some(Route::Slow(d)) => {
            std::thread::sleep(*d);
            build(200, "text/plain", b"slow")
        }
        None => build(404, "text/plain", b"not found"),
    };
    let _ = stream.write_all(&response);
    let _ = stream.flush();
}

fn build(status: u16, content_type: &str, body: &[u8]) -> Vec<u8> {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Status",
    };
    let mut out = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}
