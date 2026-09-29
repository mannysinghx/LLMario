# LLMario website

Single static page (`index.html`, `styles.css`, `script.js`, `icon.png`) served by Caddy with strict
security headers (`Caddyfile`). No external scripts, fonts, analytics or trackers.

- Preview locally: `python3 -m http.server 8766 --bind 127.0.0.1 --directory website`
- Deploy: Railway builds `website/Dockerfile` (service root directory `website`).
