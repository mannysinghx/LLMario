//! SSRF guard: scheme and port allowlist, blocked address ranges, DNS resolution with every
//! address validated, so the fetcher can pin the connection to validated addresses only.
//!
//! The `url` crate canonicalises numeric hosts (decimal `2130706433`, octal `0177.0.0.1`,
//! hex `0x7f.1`, IPv6 literals), so encoding tricks reach [`classify_ip`] as plain addresses.
//! IPv4-mapped (`::ffff:a.b.c.d`), IPv4-compatible (`::a.b.c.d`), NAT64 (`64:ff9b::/96`) and
//! 6to4 (`2002::/16`) forms are unwrapped to the embedded IPv4 address before classification.

use std::collections::{BTreeSet, HashMap};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use futures::future::BoxFuture;
use ipnet::IpNet;
use url::{Host, Url};

use crate::ToolsError;

/// Guard configuration.
#[derive(Debug, Clone)]
pub struct SsrfConfig {
    /// Ports the fetcher may connect to (default 80 and 443).
    pub allowed_ports: BTreeSet<u16>,
    /// Address ranges that are never contacted (default: [`default_blocked_ranges`]).
    pub blocked_ranges: Vec<IpNet>,
    /// Hosts (lowercase hostname or IP literal) exempt from the range check — used for a
    /// self-hosted search instance on a private address. Scheme/port checks and DNS pinning
    /// still apply.
    pub exempt_hosts: BTreeSet<String>,
    /// Allow plain `http://` (default true; `https://` is always allowed).
    pub allow_http: bool,
}

impl Default for SsrfConfig {
    fn default() -> Self {
        Self {
            allowed_ports: [80, 443].into_iter().collect(),
            blocked_ranges: default_blocked_ranges(),
            exempt_hosts: BTreeSet::new(),
            allow_http: true,
        }
    }
}

impl SsrfConfig {
    /// Allow an extra port.
    pub fn with_port(mut self, port: u16) -> Self {
        self.allowed_ports.insert(port);
        self
    }

    /// Exempt a host from the range check (see [`SsrfConfig::exempt_hosts`]).
    pub fn with_exempt_host(mut self, host: &str) -> Self {
        self.exempt_hosts
            .insert(host.trim().trim_matches(['[', ']']).to_ascii_lowercase());
        self
    }
}

/// The default blocklist: RFC 1918, loopback, link-local (incl. cloud metadata 169.254.169.254),
/// CGNAT 100.64/10, unspecified, multicast, reserved, benchmarking, IETF protocol assignments,
/// `::1`, `fc00::/7`, `fe80::/10`, multicast `ff00::/8`, and the IPv4-embedding IPv6 prefixes
/// (whose embedded addresses are also checked individually).
pub fn default_blocked_ranges() -> Vec<IpNet> {
    [
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "::ffff:0:0/96",
        "::/96",
        "64:ff9b::/96",
        "64:ff9b:1::/48",
        "2002::/16",
        "fc00::/7",
        "fe80::/10",
        "fec0::/10",
        "ff00::/8",
    ]
    .iter()
    .map(|s| s.parse().expect("static CIDR"))
    .collect()
}

/// Unwrap IPv6 forms that embed an IPv4 address.
pub fn embedded_ipv4(ip: &Ipv6Addr) -> Option<Ipv4Addr> {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return Some(v4);
    }
    let seg = ip.segments();
    // IPv4-compatible ::a.b.c.d (deprecated) — but not :: or ::1 themselves.
    if seg[..6] == [0, 0, 0, 0, 0, 0] && (seg[6] != 0 || seg[7] > 1) {
        return Some(Ipv4Addr::new(
            (seg[6] >> 8) as u8,
            seg[6] as u8,
            (seg[7] >> 8) as u8,
            seg[7] as u8,
        ));
    }
    // NAT64 64:ff9b::/96
    if seg[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        return Some(Ipv4Addr::new(
            (seg[6] >> 8) as u8,
            seg[6] as u8,
            (seg[7] >> 8) as u8,
            seg[7] as u8,
        ));
    }
    // 6to4 2002:a.b.c.d::/48
    if seg[0] == 0x2002 {
        return Some(Ipv4Addr::new(
            (seg[1] >> 8) as u8,
            seg[1] as u8,
            (seg[2] >> 8) as u8,
            seg[2] as u8,
        ));
    }
    None
}

/// Why an address is blocked, or `None` if it is allowed under `ranges`.
pub fn classify_ip_with(ip: IpAddr, ranges: &[IpNet]) -> Option<String> {
    if let IpAddr::V6(v6) = ip {
        if let Some(v4) = embedded_ipv4(&v6) {
            if let Some(reason) = classify_ip_with(IpAddr::V4(v4), ranges) {
                return Some(format!("{ip} embeds {v4}: {reason}"));
            }
        }
    }
    ranges
        .iter()
        .find(|net| net.contains(&ip))
        .map(|net| format!("{ip} is in blocked range {net}"))
}

/// Why an address is blocked under the default ranges.
pub fn classify_ip(ip: IpAddr) -> Option<String> {
    classify_ip_with(ip, &default_blocked_ranges())
}

/// Resolves hostnames. Pluggable so tests stay hermetic.
pub trait DnsResolver: Send + Sync {
    /// All addresses for `host`.
    fn resolve<'a>(
        &'a self,
        host: &'a str,
        port: u16,
    ) -> BoxFuture<'a, Result<Vec<IpAddr>, ToolsError>>;
}

/// The system resolver (`getaddrinfo` through tokio).
#[derive(Debug, Default, Clone)]
pub struct SystemResolver;

impl DnsResolver for SystemResolver {
    fn resolve<'a>(
        &'a self,
        host: &'a str,
        port: u16,
    ) -> BoxFuture<'a, Result<Vec<IpAddr>, ToolsError>> {
        Box::pin(async move {
            let addrs = tokio::net::lookup_host((host, port))
                .await
                .map_err(|e| ToolsError::Fetch(format!("dns lookup of {host} failed: {e}")))?;
            Ok(addrs.map(|a| a.ip()).collect())
        })
    }
}

/// A fixed hostname → addresses table (tests).
#[derive(Debug, Default, Clone)]
pub struct StaticResolver {
    table: HashMap<String, Vec<IpAddr>>,
}

impl StaticResolver {
    /// Empty table: every lookup fails.
    pub fn new() -> Self {
        Self::default()
    }

    /// Map a host to addresses.
    pub fn with(mut self, host: &str, addrs: &[IpAddr]) -> Self {
        self.table.insert(host.to_ascii_lowercase(), addrs.to_vec());
        self
    }
}

impl DnsResolver for StaticResolver {
    fn resolve<'a>(
        &'a self,
        host: &'a str,
        _port: u16,
    ) -> BoxFuture<'a, Result<Vec<IpAddr>, ToolsError>> {
        Box::pin(async move {
            self.table
                .get(&host.to_ascii_lowercase())
                .cloned()
                .ok_or_else(|| {
                    ToolsError::Fetch(format!("dns lookup of {host} failed: unknown host"))
                })
        })
    }
}

/// A URL that passed every check, with the addresses the fetcher must pin to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedTarget {
    /// Canonical URL.
    pub url: Url,
    /// Hostname as it appears in the URL (for SNI / certificate validation).
    pub host: String,
    /// Effective port.
    pub port: u16,
    /// Validated socket addresses (non-empty).
    pub addrs: Vec<SocketAddr>,
}

/// The guard.
#[derive(Clone)]
pub struct SsrfGuard {
    cfg: SsrfConfig,
    resolver: Arc<dyn DnsResolver>,
}

impl std::fmt::Debug for SsrfGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SsrfGuard")
            .field("cfg", &self.cfg)
            .finish_non_exhaustive()
    }
}

impl SsrfGuard {
    /// Guard with the system resolver.
    pub fn new(cfg: SsrfConfig) -> Self {
        Self {
            cfg,
            resolver: Arc::new(SystemResolver),
        }
    }

    /// Guard with a custom resolver.
    pub fn with_resolver(cfg: SsrfConfig, resolver: Arc<dyn DnsResolver>) -> Self {
        Self { cfg, resolver }
    }

    /// The configuration.
    pub fn config(&self) -> &SsrfConfig {
        &self.cfg
    }

    /// Syntax-level checks: parse, scheme, port, no userinfo. No network.
    pub fn check_syntax(&self, raw: &str) -> Result<Url, ToolsError> {
        let url = Url::parse(raw.trim()).map_err(|e| ToolsError::Url(format!("{raw:?}: {e}")))?;
        match url.scheme() {
            "https" => {}
            "http" if self.cfg.allow_http => {}
            s => return Err(ToolsError::Ssrf(format!("scheme {s:?} is not allowed"))),
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(ToolsError::Ssrf("userinfo in URL is not allowed".into()));
        }
        let port = url
            .port_or_known_default()
            .ok_or_else(|| ToolsError::Url("no port".into()))?;
        if !self.cfg.allowed_ports.contains(&port) {
            return Err(ToolsError::Ssrf(format!("port {port} is not allowed")));
        }
        if url.host().is_none() {
            return Err(ToolsError::Url("URL has no host".into()));
        }
        Ok(url)
    }

    /// Why `ip` may not be contacted for `host`, or `None`.
    pub fn blocked_reason(&self, host: &str, ip: IpAddr) -> Option<String> {
        if self.cfg.exempt_hosts.contains(&host.to_ascii_lowercase()) {
            return None;
        }
        classify_ip_with(ip, &self.cfg.blocked_ranges)
    }

    /// Full validation: syntax, then resolution, then every address checked. Fails if *any*
    /// resolved address is blocked (an attacker controls which one the OS would pick).
    pub async fn validate(&self, raw: &str) -> Result<ValidatedTarget, ToolsError> {
        let url = self.check_syntax(raw)?;
        let port = url.port_or_known_default().unwrap_or(80);
        let (host, ips): (String, Vec<IpAddr>) = match url.host().expect("checked") {
            Host::Ipv4(v4) => (v4.to_string(), vec![IpAddr::V4(v4)]),
            Host::Ipv6(v6) => (v6.to_string(), vec![IpAddr::V6(v6)]),
            Host::Domain(d) => {
                let d = d.to_ascii_lowercase();
                let ips = self.resolver.resolve(&d, port).await?;
                (d, ips)
            }
        };
        if ips.is_empty() {
            return Err(ToolsError::Fetch(format!(
                "dns lookup of {host} returned no addresses"
            )));
        }
        for ip in &ips {
            if let Some(reason) = self.blocked_reason(&host, *ip) {
                return Err(ToolsError::Ssrf(format!("{host}: {reason}")));
            }
        }
        Ok(ValidatedTarget {
            host,
            port,
            addrs: ips
                .into_iter()
                .map(|ip| SocketAddr::new(ip, port))
                .collect(),
            url,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> SsrfGuard {
        let resolver = StaticResolver::new()
            .with("public.example", &["93.184.216.34".parse().unwrap()])
            .with(
                "rebind.example",
                &[
                    "93.184.216.34".parse().unwrap(),
                    "10.0.0.5".parse().unwrap(),
                ],
            )
            .with(
                "v6.example",
                &["2606:2800:220:1:248:1893:25c8:1946".parse().unwrap()],
            )
            .with(
                "mapped.example",
                &["::ffff:169.254.169.254".parse().unwrap()],
            )
            .with("empty.example", &[]);
        SsrfGuard::with_resolver(SsrfConfig::default(), Arc::new(resolver))
    }

    #[tokio::test]
    async fn blocked_corpus() {
        let g = guard();
        let blocked = [
            "http://127.0.0.1/",
            "http://127.1/",
            "http://2130706433/",
            "http://0x7f000001/",
            "http://0177.0.0.1/",
            "http://0x7f.1/",
            "http://10.0.0.1/",
            "http://10.255.255.255/",
            "http://172.16.0.1/",
            "http://172.31.255.254/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://100.64.0.1/",
            "http://100.127.255.255/",
            "http://0.0.0.0/",
            "http://192.0.0.1/",
            "http://198.18.0.1/",
            "http://224.0.0.1/",
            "http://255.255.255.255/",
            "http://[::1]/",
            "http://[::]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:7f00:1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://[::127.0.0.1]/",
            "http://[64:ff9b::7f00:1]/",
            "http://[64:ff9b::192.168.0.1]/",
            "http://[2002:7f00:1::]/",
            "http://[2002:a00:1::1]/",
            "http://[fc00::1]/",
            "http://[fd12:3456::1]/",
            "http://[fe80::1]/",
            "http://[ff02::1]/",
            "http://rebind.example/",
            "http://mapped.example/",
        ];
        for u in blocked {
            let r = g.validate(u).await;
            assert!(
                matches!(r, Err(ToolsError::Ssrf(_))),
                "{u} should be blocked, got {r:?}"
            );
        }
    }

    #[tokio::test]
    async fn allowed_and_pinned() {
        let g = guard();
        let t = g.validate("https://PUBLIC.example/path?q=1").await.unwrap();
        assert_eq!(t.host, "public.example");
        assert_eq!(t.port, 443);
        assert_eq!(
            t.addrs,
            vec!["93.184.216.34:443".parse::<SocketAddr>().unwrap()]
        );
        let t = g.validate("http://v6.example:80/").await.unwrap();
        assert_eq!(
            t.addrs[0].ip(),
            "2606:2800:220:1:248:1893:25c8:1946"
                .parse::<IpAddr>()
                .unwrap()
        );
        let t = g.validate("http://93.184.216.34/").await.unwrap();
        assert_eq!(t.host, "93.184.216.34");
        let t = g
            .validate("http://[2606:2800:220:1:248:1893:25c8:1946]/")
            .await
            .unwrap();
        assert_eq!(t.port, 80);
    }

    #[tokio::test]
    async fn syntax_rules() {
        let g = guard();
        for u in [
            "ftp://public.example/",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,hi",
            "gopher://public.example/",
            "https://public.example:8443/",
            "http://public.example:22/",
            "https://user:pw@public.example/",
            "https://user@public.example/",
        ] {
            let r = g.validate(u).await;
            assert!(matches!(r, Err(ToolsError::Ssrf(_))), "{u}: {r:?}");
        }
        assert!(matches!(
            g.validate("not a url").await,
            Err(ToolsError::Url(_))
        ));
        assert!(matches!(
            g.validate("http://unknown.example/").await,
            Err(ToolsError::Fetch(_))
        ));
        assert!(matches!(
            g.validate("http://empty.example/").await,
            Err(ToolsError::Fetch(_))
        ));
        let no_http = SsrfGuard::with_resolver(
            SsrfConfig {
                allow_http: false,
                ..Default::default()
            },
            Arc::new(
                StaticResolver::new().with("public.example", &["93.184.216.34".parse().unwrap()]),
            ),
        );
        assert!(matches!(
            no_http.validate("http://public.example/").await,
            Err(ToolsError::Ssrf(_))
        ));
        assert!(no_http.validate("https://public.example/").await.is_ok());
    }

    #[tokio::test]
    async fn exempt_host_and_extra_port() {
        let cfg = SsrfConfig::default()
            .with_port(8080)
            .with_exempt_host("127.0.0.1")
            .with_exempt_host("searx.local");
        let resolver = StaticResolver::new().with("searx.local", &["10.0.0.9".parse().unwrap()]);
        let g = SsrfGuard::with_resolver(cfg, Arc::new(resolver));
        assert!(g.validate("http://127.0.0.1:8080/search").await.is_ok());
        assert!(g.validate("http://searx.local:8080/search").await.is_ok());
        // Exemption is by host string, not by address: the same address via another name is blocked.
        assert!(matches!(
            g.validate("http://127.0.0.2:8080/").await,
            Err(ToolsError::Ssrf(_))
        ));
        assert!(matches!(
            g.validate("http://[::1]:8080/").await,
            Err(ToolsError::Ssrf(_))
        ));
        assert!(matches!(
            g.validate("http://127.0.0.1:9090/").await,
            Err(ToolsError::Ssrf(_))
        ));
    }

    #[test]
    fn embedded_forms() {
        assert_eq!(
            embedded_ipv4(&"::ffff:1.2.3.4".parse().unwrap()),
            Some(Ipv4Addr::new(1, 2, 3, 4))
        );
        assert_eq!(
            embedded_ipv4(&"::1.2.3.4".parse().unwrap()),
            Some(Ipv4Addr::new(1, 2, 3, 4))
        );
        assert_eq!(
            embedded_ipv4(&"64:ff9b::1.2.3.4".parse().unwrap()),
            Some(Ipv4Addr::new(1, 2, 3, 4))
        );
        assert_eq!(
            embedded_ipv4(&"2002:102:304::".parse().unwrap()),
            Some(Ipv4Addr::new(1, 2, 3, 4))
        );
        assert_eq!(embedded_ipv4(&"::1".parse().unwrap()), None);
        assert_eq!(embedded_ipv4(&"::".parse().unwrap()), None);
        assert_eq!(embedded_ipv4(&"2606::1".parse().unwrap()), None);
        assert!(classify_ip("8.8.8.8".parse().unwrap()).is_none());
        assert!(classify_ip("2606:2800::1".parse().unwrap()).is_none());
        assert!(
            classify_ip("::ffff:8.8.8.8".parse().unwrap()).is_some(),
            "mapped forms are blocked outright"
        );
    }
}
