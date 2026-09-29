#!/usr/bin/env python3
"""Build crates/model_registry/catalog.toml from scripts/catalog/sources.toml.

For every family and format it tries the candidate repos in order and, for the first one that
exists on the Hugging Face Hub, records:
  - the commit SHA (revision is pinned, so downloads are reproducible)
  - the exact files to download (4-bit GGUF file or shard set; MLX directory files), sizes
  - license (from the model card), gated flag
  - architecture, context length and attention shape, read from the GGUF header (HTTP range
    requests; never the whole file) or from config.json, for memory estimates before download
Variants that cannot be verified are dropped and reported. Standard library only.

Usage: python3 scripts/catalog/build.py [--only FAMILY_ID ...]
"""
import json
import re
import struct
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "scripts/catalog/sources.toml"
OUT = ROOT / "crates/model_registry/catalog.toml"
DOCS = ROOT / "docs/MODELS.md"
HUB = "https://huggingface.co"
UA = {"User-Agent": "llmario-catalog-builder/0.1"}
GGUF_QUANT_PREFERENCE = ["Q4_K_M", "Q4_0", "MXFP4", "Q4_K_S", "IQ4_XS", "Q4_K_L", "Q5_K_M", "Q8_0"]
GGUF_EXCLUDE = re.compile(r"mmproj|imatrix|draft|(^|/)mtp[-_/]|[-_]mtp[-_.]", re.I)
MLX_ALLOWED_EXT = {"json", "safetensors", "txt", "model", "jinja", "tiktoken"}
LICENSES = {
    "apache-2.0": "Apache-2.0", "mit": "MIT", "llama3.1": "Llama 3.1 Community License",
    "llama3.2": "Llama 3.2 Community License", "llama3.3": "Llama 3.3 Community License",
    "gemma": "Gemma Terms of Use", "bsd-3-clause": "BSD-3-Clause", "cc-by-4.0": "CC-BY-4.0",
}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get_range(url, n):
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes=0-{n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def repo_info(repo):
    q = urllib.parse.quote(repo, safe="/")
    try:
        info = get_json(f"{HUB}/api/models/{q}/revision/main?blobs=true")
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    try:
        card = get_json(f"{HUB}/api/models/{q}")
    except urllib.error.HTTPError:
        card = {}
    info["_card"] = card.get("cardData") or info.get("cardData") or {}
    info["_gated"] = bool(card.get("gated", info.get("gated", False)))
    return info, None


def license_label(card):
    lic = (card or {}).get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if not lic:
        return None
    if lic == "other":
        return card.get("license_name") or "Custom license (see model card)"
    return LICENSES.get(lic, lic)


# ---------- GGUF header parsing (bounded, from a byte prefix) ----------
class Truncated(Exception):
    pass


def parse_gguf(buf):
    pos = 0

    def take(n):
        nonlocal pos
        if pos + n > len(buf):
            raise Truncated()
        b = buf[pos:pos + n]
        pos += n
        return b

    u32 = lambda: struct.unpack("<I", take(4))[0]
    u64 = lambda: struct.unpack("<Q", take(8))[0]

    def string():
        n = u64()
        if n > 64 * 1024 * 1024:
            raise ValueError("string too long")
        return take(n).decode("utf-8", "replace")

    scalar = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
              6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

    def value(t, top=True):
        if t in scalar:
            fmt, n = scalar[t]
            return struct.unpack(fmt, take(n))[0]
        if t == 8:
            return string()
        if t == 9 and top:
            et, n = u32(), u64()
            if et == 8:
                for _ in range(n):
                    string()
                return {"len": n}
            vals = [value(et, False) for _ in range(n)]
            return vals if n <= 4096 else {"len": n}
        raise ValueError(f"bad gguf type {t}")

    if take(4) != b"GGUF":
        raise ValueError("not GGUF")
    version = u32()
    if version not in (2, 3):
        raise ValueError(f"GGUF v{version}")
    u64()  # tensor count
    kv = {}
    for _ in range(u64()):
        k = string()
        kv[k] = value(u32())
    return kv


def gguf_metadata(repo, sha, fname):
    url = f"{HUB}/{urllib.parse.quote(repo, safe='/')}/resolve/{sha}/{urllib.parse.quote(fname)}"
    for n in (4, 16, 48):
        try:
            return parse_gguf(get_range(url, n * 1024 * 1024))
        except Truncated:
            continue
    raise ValueError("GGUF header larger than 48 MB")


def as_int(v):
    if isinstance(v, list):
        nums = [x for x in v if isinstance(x, (int, float))]
        return int(max(nums)) if nums else None
    return int(v) if isinstance(v, (int, float)) else None


def gguf_facts(kv):
    arch = kv.get("general.architecture")
    a = lambda k: as_int(kv.get(f"{arch}.{k}"))
    heads, hidden = a("attention.head_count"), a("embedding_length")
    shape = None
    if a("block_count") and heads and hidden:
        shape = {
            "n_layers": a("block_count"),
            "n_heads": heads,
            "n_kv_heads": a("attention.head_count_kv") or heads,
            "head_dim": a("attention.key_length") or hidden // max(heads, 1),
            "hidden_size": hidden,
            "context_max": a("context_length"),
        }
    return arch, a("context_length"), shape


def mlx_facts(repo, sha):
    cfg = get_json(f"{HUB}/{urllib.parse.quote(repo, safe='/')}/resolve/{sha}/config.json")
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    g = lambda k: tc.get(k, cfg.get(k))
    heads, hidden, layers = g("num_attention_heads"), g("hidden_size"), g("num_hidden_layers")
    ctx = g("max_position_embeddings")
    shape = None
    if heads and hidden and layers:
        shape = {
            "n_layers": layers,
            "n_heads": heads,
            "n_kv_heads": g("num_key_value_heads") or heads,
            "head_dim": g("head_dim") or hidden // heads,
            "hidden_size": hidden,
            "context_max": ctx,
        }
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    quant = f"{q['bits']}-bit" + (f" (group {q['group_size']})" if q.get("group_size") else "") if q.get("bits") else None
    return cfg.get("model_type"), ctx, shape, quant


# ---------- file selection ----------
SHARD = re.compile(r"^(?P<prefix>.*)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")


def pick_gguf(siblings):
    files = [s for s in siblings if s["rfilename"].lower().endswith(".gguf") and not GGUF_EXCLUDE.search(s["rfilename"])]
    for quant in GGUF_QUANT_PREFERENCE:
        tok = re.compile(rf"(^|[-_./]){re.escape(quant)}([-_.]|$)", re.I)
        cands = [s for s in files if tok.search(s["rfilename"])]
        if not cands:
            continue
        # Group shards; prefer plain names over "UD-"/dynamic variants, then fewer path parts.
        groups = {}
        for s in cands:
            m = SHARD.match(s["rfilename"])
            key = m.group("prefix") if m else s["rfilename"]
            groups.setdefault(key, []).append(s)
        def rank(item):
            key, group = item
            return ("ud-" in key.lower() or "dynamic" in key.lower(), key.count("/"), len(key))
        key, group = sorted(groups.items(), key=rank)[0]
        group.sort(key=lambda s: s["rfilename"])
        m = SHARD.match(group[0]["rfilename"])
        if m and len(group) != int(m.group("n")):
            continue  # incomplete shard set
        return quant, group
    return None, None


def size_of(s):
    return (s.get("lfs") or {}).get("size") or s.get("size") or 0


def quant_slug(q):
    s = q.lower()
    return s.replace("_", "") if "_k" in s else s


def mlx_suffix(repo):
    m = re.search(r"(mxfp4-q8|mxfp4|\d+bit(?:-dwq)?)$", repo.split("/")[-1], re.I)
    return (m.group(1) if m else "4bit").lower()


# ---------- main ----------
def build_variant(fam, fmt, repos):
    errors = []
    for repo in repos:
        info, err = repo_info(repo)
        if not info:
            errors.append(f"{repo}: {err}")
            continue
        sha, sibs, card = info["sha"], info.get("siblings", []), info["_card"]
        if fmt == "gguf":
            quant, group = pick_gguf(sibs)
            if not group:
                errors.append(f"{repo}: no 4-bit GGUF file")
                continue
            try:
                arch, ctx, shape = gguf_facts(gguf_metadata(repo, sha, group[0]["rfilename"]))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{repo}: header unreadable ({e})")
                continue
            files = [s["rfilename"] for s in group]
            vid = f"{fam['id']}-gguf-{quant_slug(quant)}"
            quant_label = quant
        else:
            chosen = [s for s in sibs if "/" not in s["rfilename"]
                      and s["rfilename"].rsplit(".", 1)[-1] in MLX_ALLOWED_EXT]
            if not any(s["rfilename"].endswith(".safetensors") for s in chosen):
                errors.append(f"{repo}: no safetensors")
                continue
            try:
                arch, ctx, shape, quant_label = mlx_facts(repo, sha)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{repo}: config.json unreadable ({e})")
                continue
            group, files = chosen, []  # MLX: whole directory (allowlisted files)
            vid = f"{fam['id']}-mlx-{mlx_suffix(repo)}"
        return {
            "id": vid, "family": fam["id"], "format": fmt, "repo": repo, "revision": sha,
            "files": files, "approx_bytes": sum(size_of(s) for s in group),
            "license": license_label(card), "gated": info["_gated"],
            "architecture": arch, "context_max": ctx, "quantization": quant_label, "shape": shape,
        }, errors
    return None, errors


def toml_str(v):
    return json.dumps(v, ensure_ascii=False)


def emit(fams, variants):
    lines = [
        "# GENERATED by scripts/catalog/build.py from scripts/catalog/sources.toml. Do not edit by hand.",
        "# Every entry was verified against the Hugging Face Hub: revision is a pinned commit, files and",
        "# sizes are exact, architecture/context/shape come from the GGUF header or config.json.",
        "# License labels come from the model cards and are not legal advice.",
        "",
    ]
    for v in variants:
        f = fams[v["family"]]
        lines.append("[[models]]")
        fields = [
            ("id", v["id"]), ("family", f["id"]), ("name", f["name"]), ("publisher", f["publisher"]),
            ("released", f["released"]), ("params", f["params"]), ("tasks", f["tasks"]),
            ("format", v["format"]), ("repo", v["repo"]), ("revision", v["revision"]),
            ("files", v["files"]), ("approx_bytes", v["approx_bytes"]),
            ("license", v["license"] or f.get("license") or "See model card"),
            ("gated", v["gated"]), ("architecture", v["architecture"]),
            ("quantization", v["quantization"]), ("context_max", v["context_max"]),
            ("description", f["summary"]), ("notes", f.get("notes")),
        ]
        for k, val in fields:
            if val is None or val == []:
                if k == "files":
                    lines.append("files = []")
                continue
            lines.append(f"{k} = {toml_str(val) if not isinstance(val, bool) else str(val).lower()}")
        if v["shape"]:
            s = v["shape"]
            inner = ", ".join(f"{k} = {s[k]}" for k in ("n_layers", "n_heads", "n_kv_heads", "head_dim", "hidden_size") if s.get(k) is not None)
            if s.get("context_max"):
                inner += f", context_max = {s['context_max']}"
            lines.append(f"shape = {{ {inner} }}")
        lines.append("")
    OUT.write_text("\n".join(lines))


def emit_docs(fams, variants):
    """Human-readable library for GitHub readers (same verified data as the app)."""
    engines = {"gguf": "llama.cpp", "mlx": "MLX"}
    fam_order, by_fam = [], {}
    for v in variants:
        if v["family"] not in by_fam:
            fam_order.append(v["family"])
        by_fam.setdefault(v["family"], []).append(v)
    out = [
        "# Model library",
        "",
        "Generated by `scripts/catalog/build.py`. Every download below was verified on the Hugging Face",
        "Hub and is pinned to an exact commit. Sizes are download sizes. The desktop app and",
        "`llmario model catalog` also show, for **your** computer, how much memory each model needs and",
        "whether your installed engines can run it.",
        "",
        f"{len(fam_order)} model families · {len(variants)} downloads.",
        "",
        "| Model | Publisher | Released | Parameters | Good at | License |",
        "|---|---|---|---|---|---|",
    ]
    for fid in fam_order:
        f = fams[fid]
        lic = by_fam[fid][0]["license"] or "See model card"
        out.append(f"| [{f['name']}](#{fid.replace('.', '')}) | {f['publisher']} | {f['released']} | {f['params']} | {', '.join(f['tasks'])} | {lic} |")
    for fid in fam_order:
        f = fams[fid]
        out += ["", f'<a id="{fid.replace(".", "")}"></a>', f"## {f['name']}", "", f["summary"], ""]
        if f.get("notes"):
            out += [f"*{f['notes']}*", ""]
        out += ["| Engine | Download id | Exact Hugging Face files | Quantization | Size | Context |", "|---|---|---|---|---|---|"]
        for v in by_fam[fid]:
            files = (v["files"][0] + (f" (+{len(v['files']) - 1} parts)" if len(v["files"]) > 1 else "")) if v["files"] else "MLX folder"
            ctx = f"{v['context_max'] // 1024}K" if v.get("context_max") else "?"
            out.append(f"| {engines[v['format']]} | `{v['id']}` | [{v['repo']}](https://huggingface.co/{v['repo']}/tree/{v['revision']}) · `{files}` | {v['quantization'] or '?'} | {v['approx_bytes'] / 1e9:.1f} GB | {ctx} |")
        out += ["", f"Download: `llmario model pull {fid}` (best variant for your machine) or pick an id above."]
    DOCS.write_text("\n".join(out) + "\n")


def main():
    only = set(sys.argv[sys.argv.index("--only") + 1:]) if "--only" in sys.argv else None
    fams_list = tomllib.loads(SOURCES.read_text())["family"]
    fams = {f["id"]: f for f in fams_list}
    variants, problems = [], []
    for f in fams_list:
        if only and f["id"] not in only:
            continue
        for fmt in ("mlx", "gguf"):
            if not f.get(fmt):
                continue
            v, errs = build_variant(f, fmt, f[fmt])
            if v:
                # License from a converter repo is often missing; share within the family.
                variants.append(v)
                print(f"✓ {v['id']:<44} {v['repo']:<58} {v['approx_bytes']/1e9:6.2f} GB  {v['architecture']}  ctx={v['context_max']}")
            else:
                problems.append((f["id"], fmt, errs))
                print(f"✗ {f['id']} [{fmt}]  " + "; ".join(errs))
    # Fill missing licenses from a sibling variant of the same family.
    by_fam = {}
    for v in variants:
        if v["license"]:
            by_fam.setdefault(v["family"], v["license"])
    for v in variants:
        v["license"] = v["license"] or by_fam.get(v["family"])
    if not only:
        emit(fams, variants)
        emit_docs(fams, variants)
        print(f"\nwrote {OUT.relative_to(ROOT)}: {len(variants)} variants from {len({v['family'] for v in variants})} families; {len(problems)} variant(s) dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
