# UI preview

Preview the desktop interface in any browser, without building the Tauri app. The real `ui/`
files are loaded with a simulated Tauri bridge (`mock-tauri.js`). Library data is real for your
machine; chat, downloads and model loading are simulated.

```bash
llmario model catalog --json > apps/desktop/preview/catalog.json
python3 -m http.server 8765 --bind 127.0.0.1 --directory apps/desktop
open http://127.0.0.1:8765/preview/index.html
```
