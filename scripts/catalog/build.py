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
# 3-bit builds (Phase 5): best quality first, then smaller.
GGUF_3BIT_PREFERENCE = ["Q3_K_M", "Q3_K_XL", "IQ3_M", "Q3_K_S", "IQ3_XXS"]
QUALITY_3BIT = "3-bit: smaller and faster than 4-bit, with some quality loss"
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
        shape["kv_groups"], shape["state_bytes_per_seq"] = layout_from_gguf(
            kv, shape["n_layers"], shape["n_kv_heads"], shape["head_dim"])
        shape["mtp_layers"] = a("nextn_predict_layers") or 0
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
        shape["kv_groups"], shape["state_bytes_per_seq"] = layout_from_config(tc)
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    quant = f"{q['bits']}-bit" + (f" (group {q['group_size']})" if q.get("group_size") else "") if q.get("bits") else None
    return cfg.get("model_type"), ctx, shape, quant


# ---------- per-layer KV layout (mirrors crates/model_registry/src/layout.rs) ----------
# Recorded only when it differs from "every layer full attention" and can be computed exactly;
# anything uncertain records nothing and the planner counts every layer (an overestimate).

def _finish(layers, state):
    groups = {}
    for l in layers:
        if l != "rec":
            groups[l] = groups.get(l, 0) + 1
    plain = state == 0 and len(groups) == 1 and next(iter(groups))[2] is None and "rec" not in layers
    if plain or not groups:
        return [], 0
    order = sorted(groups, key=lambda g: (g[2] is not None, g[2] or 0, g[0], g[1]))
    out = []
    for heads, dim, window in order:
        g = {"layers": groups[(heads, dim, window)], "n_kv_heads": heads, "head_dim": dim}
        if window is not None:
            g["window"] = window
        out.append(g)
    return out, state


def layout_from_gguf(kv, n_layers, default_heads, default_dim):
    arch = kv.get("general.architecture")
    if not arch:
        return [], 0
    n = n_layers

    def per_layer(key):
        v = kv.get(f"{arch}.{key}")
        if isinstance(v, list) and len(v) == n:
            return [int(x) for x in v]
        x = as_int(v)
        return None if x is None else [x] * n

    u = lambda k: as_int(kv.get(f"{arch}.{k}"))

    def mean(k, v):
        if k is not None and v is not None:
            return -(-(k + v) // 2)
        return k if k is not None else (v if v is not None else default_dim)

    heads = per_layer("attention.head_count_kv") or [default_heads] * n
    main_layers = n - (u("nextn_predict_layers") or 0)  # MTP draft layers come last
    dim = mean(u("attention.key_length"), u("attention.value_length"))
    ks, vs = u("attention.key_length_swa"), u("attention.value_length_swa")
    dim_swa = dim if ks is None and vs is None else mean(ks, vs)
    window = u("attention.sliding_window") or None
    pattern = per_layer("attention.sliding_window_pattern")
    interval = u("full_attention_interval") or None
    layers = []
    for i, h in enumerate(heads[:main_layers]):
        if not ((interval is None or (i + 1) % interval == 0) and h > 0):
            layers.append("rec")
            continue
        if window is None:
            swa = False
        elif pattern is not None:
            swa = pattern[i] != 0
        elif arch == "gpt-oss":  # llama.cpp hard-codes set_swa_pattern(2): even layers slide
            swa = i % 2 == 0
        else:
            return [], 0
        layers.append((h, dim_swa, window) if swa else (h, dim, None))
    rec = layers.count("rec")
    state = 0
    if rec:
        inner, st, kernel = u("ssm.inner_size"), u("ssm.state_size"), u("ssm.conv_kernel")
        if None in (inner, st, kernel):
            return [], 0
        groups = u("ssm.group_count") or 1
        state = rec * 4 * (inner * st + max(kernel - 1, 0) * (inner + 2 * groups * st))
    return _finish(layers, state)


def layout_from_config(tc):
    num = lambda k: tc.get(k) if isinstance(tc.get(k), int) and not isinstance(tc.get(k), bool) else None
    n = num("num_hidden_layers")
    if n is None:
        return [], 0
    types = tc.get("layer_types")
    if isinstance(types, list):
        if len(types) != n:
            return [], 0
    elif num("full_attention_interval"):
        iv = num("full_attention_interval")
        types = ["full_attention" if (i + 1) % iv == 0 else "linear_attention" for i in range(n)]
    else:
        return [], 0
    heads = num("num_key_value_heads") or num("num_attention_heads")
    dim = num("head_dim") or (num("hidden_size") // num("num_attention_heads")
                              if num("hidden_size") and num("num_attention_heads") else None)
    if heads is None or dim is None:
        return [], 0
    g_heads, g_dim = num("num_global_key_value_heads") or heads, num("global_head_dim") or dim
    window = num("sliding_window") or None
    layers = []
    for t in types:
        if t == "full_attention":
            layers.append((g_heads, g_dim, None))
        elif t == "sliding_attention" and window:
            layers.append((heads, dim, window))
        elif t == "linear_attention":
            layers.append("rec")
        else:
            return [], 0
    rec = layers.count("rec")
    state = 0
    if rec:
        vals = [num(k) for k in ("linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
                                 "linear_value_head_dim", "linear_conv_kernel_dim")]
        if None in vals:
            return [], 0
        vh, kh, kd, vd, kernel = vals
        state = rec * (vh * kd * vd * 4 + max(kernel - 1, 0) * (2 * kh * kd + vh * vd) * 2)
    return _finish(layers, state)


def self_test():
    """Same fixtures and expected values as the Rust tests in layout.rs."""
    q = {"general.architecture": "qwen35", "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256,
         "qwen35.attention.value_length": 256, "qwen35.full_attention_interval": 4, "qwen35.ssm.conv_kernel": 4,
         "qwen35.ssm.state_size": 128, "qwen35.ssm.group_count": 16, "qwen35.ssm.inner_size": 4096}
    assert layout_from_gguf(q, 32, 4, 256) == ([{"layers": 8, "n_kv_heads": 4, "head_dim": 256}], 52_690_944)
    gm = {"general.architecture": "gemma4", "gemma4.attention.head_count_kv": [1 if i % 6 == 5 else 8 for i in range(48)],
          "gemma4.attention.key_length": 512, "gemma4.attention.value_length": 512,
          "gemma4.attention.key_length_swa": 256, "gemma4.attention.value_length_swa": 256,
          "gemma4.attention.sliding_window": 1024,
          "gemma4.attention.sliding_window_pattern": [i % 6 != 5 for i in range(48)]}
    assert layout_from_gguf(gm, 48, 8, 512) == ([{"layers": 8, "n_kv_heads": 1, "head_dim": 512},
                                                 {"layers": 40, "n_kv_heads": 8, "head_dim": 256, "window": 1024}], 0)
    oss = {"general.architecture": "gpt-oss", "gpt-oss.attention.head_count_kv": 8, "gpt-oss.attention.key_length": 64,
           "gpt-oss.attention.value_length": 64, "gpt-oss.attention.sliding_window": 128}
    assert layout_from_gguf(oss, 24, 8, 64) == ([{"layers": 12, "n_kv_heads": 8, "head_dim": 64},
                                                 {"layers": 12, "n_kv_heads": 8, "head_dim": 64, "window": 128}], 0)
    assert layout_from_gguf(dict(q, **{"qwen35.nextn_predict_layers": 1}), 33, 4, 256) == layout_from_gguf(q, 32, 4, 256)
    assert layout_from_gguf({"general.architecture": "qwen3", "qwen3.attention.head_count_kv": 8}, 36, 8, 128) == ([], 0)
    assert layout_from_gguf({"general.architecture": "olmo2", "olmo2.attention.sliding_window": 4096}, 32, 32, 128) == ([], 0)
    assert layout_from_gguf({"general.architecture": "lfm2", "lfm2.attention.head_count_kv": [0, 0, 8, 0, 0, 8]}, 6, 8, 64) == ([], 0)
    qm = {"num_hidden_layers": 64, "full_attention_interval": 4, "num_attention_heads": 24, "num_key_value_heads": 4,
          "head_dim": 256, "linear_num_value_heads": 48, "linear_num_key_heads": 16, "linear_key_head_dim": 128,
          "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}
    assert layout_from_config(qm) == ([{"layers": 16, "n_kv_heads": 4, "head_dim": 256}], 153_944_064)
    gc = {"num_hidden_layers": 48, "layer_types": ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(48)],
          "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256, "num_global_key_value_heads": 1,
          "global_head_dim": 512, "sliding_window": 1024}
    assert layout_from_config(gc) == ([{"layers": 8, "n_kv_heads": 1, "head_dim": 512},
                                       {"layers": 40, "n_kv_heads": 8, "head_dim": 256, "window": 1024}], 0)
    assert layout_from_config({"num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}) == ([], 0)
    assert layout_from_config({"num_hidden_layers": 2, "layer_types": ["conv", "full_attention"], "num_attention_heads": 2,
                               "num_key_value_heads": 1, "head_dim": 8}) == ([], 0)
    print("layout self-test: ok")


def shape_toml(s):
    inner = ", ".join(f"{k} = {s[k]}" for k in ("n_layers", "n_heads", "n_kv_heads", "head_dim", "hidden_size") if s.get(k) is not None)
    if s.get("context_max"):
        inner += f", context_max = {s['context_max']}"
    if s.get("mtp_layers"):
        inner += f", mtp_layers = {s['mtp_layers']}"
    if s.get("kv_groups"):
        gs = ", ".join("{ " + ", ".join(f"{k} = {g[k]}" for k in ("layers", "n_kv_heads", "head_dim", "window") if k in g) + " }"
                       for g in s["kv_groups"])
        inner += f", kv_groups = [{gs}]"
    if s.get("state_bytes_per_seq"):
        inner += f", state_bytes_per_seq = {s['state_bytes_per_seq']}"
    return f"shape = {{ {inner} }}"


def add_layouts():
    """Add per-layer KV layouts to the existing catalog at each entry's pinned revision, changing
    nothing else in catalog.toml."""
    text = OUT.read_text()
    cat = tomllib.loads(text)["models"]
    blocks = text.split("\n[[models]]\n")
    head, entries = blocks[0], blocks[1:]
    assert len(entries) == len(cat), "catalog layout changed; refusing to rewrite"
    changed = 0
    for i, m in enumerate(cat):
        s = m.get("shape")
        if not s:
            continue
        try:
            if m["format"] == "gguf":
                groups, state = layout_from_gguf(gguf_metadata(m["repo"], m["revision"], m["files"][0]),
                                                 s["n_layers"], s["n_kv_heads"], s["head_dim"])
            else:
                cfg = get_json(f"{HUB}/{urllib.parse.quote(m['repo'], safe='/')}/resolve/{m['revision']}/config.json")
                tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
                groups, state = layout_from_config(tc)
        except Exception as e:  # keep the entry as it was
            print(f"✗ {m['id']}: {e}")
            continue
        new = dict(s, kv_groups=groups, state_bytes_per_seq=state)
        old_line = next(l for l in entries[i].splitlines() if l.startswith("shape = "))
        new_line = shape_toml(new)
        if new_line != old_line:
            entries[i] = entries[i].replace(old_line, new_line)
            changed += 1
        full = sum(g["layers"] for g in groups if "window" not in g)
        print(f"{'✓' if groups else ' '} {m['id']:<44} " + (f"{full}/{s['n_layers']} full, {len(groups)} group(s), state {state}" if groups else "plain or unknown: unchanged"))
    OUT.write_text("\n[[models]]\n".join([head] + entries))
    print(f"\n{changed} catalog entr(ies) gained a layout; nothing else changed")


# ---------- file selection ----------
SHARD = re.compile(r"^(?P<prefix>.*)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")


def pick_gguf(siblings, preference=None):
    files = [s for s in siblings if s["rfilename"].lower().endswith(".gguf") and not GGUF_EXCLUDE.search(s["rfilename"])]
    for quant in preference or GGUF_QUANT_PREFERENCE:
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
        if fmt in ("gguf", "gguf_mtp", "gguf_3bit"):
            quant, group = pick_gguf(sibs, GGUF_3BIT_PREFERENCE if fmt == "gguf_3bit" else None)
            if not group:
                errors.append(f"{repo}: no {'3' if fmt == 'gguf_3bit' else '4'}-bit GGUF file")
                continue
            try:
                arch, ctx, shape = gguf_facts(gguf_metadata(repo, sha, group[0]["rfilename"]))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{repo}: header unreadable ({e})")
                continue
            files = [s["rfilename"] for s in group]
            vid = f"{fam['id']}-gguf-{quant_slug(quant)}"
            quant_label = quant
            if fmt == "gguf_mtp":
                if not (shape or {}).get("mtp_layers"):
                    errors.append(f"{repo}: no MTP (nextn) layers in the header")
                    continue
                vid += "-mtp"
            if fmt == "gguf_3bit":
                quant_label = f"{quant} ({QUALITY_3BIT})"
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
            if fmt == "mlx_3bit":
                if not mlx_suffix(repo).startswith("3bit"):
                    errors.append(f"{repo}: not a 3-bit build")
                    continue
                quant_label = f"{quant_label}: {QUALITY_3BIT.split(': ', 1)[1]}"
        return {
            "id": vid, "family": fam["id"], "format": {"gguf_mtp": "gguf", "gguf_3bit": "gguf", "mlx_3bit": "mlx"}.get(fmt, fmt), "repo": repo, "revision": sha,
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
        lines.extend(variant_lines(fams[v["family"]], v))
    OUT.write_text("\n".join(lines))


def variant_lines(f, v):
    """One `[[models]]` block (with its trailing blank line) as catalog.toml lines."""
    lines = ["[[models]]"]
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
        lines.append(shape_toml(v["shape"]))
    lines.append("")
    return lines


def doc_row(v):
    """One download row of docs/MODELS.md (catalog entry or freshly built variant)."""
    engine = {"gguf": "llama.cpp", "mlx": "MLX"}[v["format"]]
    if (v.get("shape") or {}).get("mtp_layers"):
        engine += " (MTP)"
    files = (v["files"][0] + (f" (+{len(v['files']) - 1} parts)" if len(v["files"]) > 1 else "")) if v["files"] else "MLX folder"
    ctx = f"{v['context_max'] // 1024}K" if v.get("context_max") else "?"
    return f"| {engine} | `{v['id']}` | [{v['repo']}](https://huggingface.co/{v['repo']}/tree/{v['revision']}) · `{files}` | {v.get('quantization') or '?'} | {v['approx_bytes'] / 1e9:.1f} GB | {ctx} |"


def sync_docs():
    """Add a docs/MODELS.md row for every catalog entry that lacks one (after its family's other
    rows), and refresh the download count. Offline: uses the pinned catalog data."""
    cat = tomllib.loads(OUT.read_text())["models"]
    lines = DOCS.read_text().split("\n")
    added = 0
    for m in cat:
        if any(f"`{m['id']}`" in l for l in lines):
            continue
        fam_rows = [i for i, l in enumerate(lines) if l.startswith("| ") and f"| `{m['family']}-" in l]
        assert fam_rows, f"no docs rows for family {m['family']}"
        lines.insert(fam_rows[-1] + 1, doc_row(m))
        added += 1
    n = len(cat)
    lines = [re.sub(r"· \d+ downloads\.", f"· {n} downloads.", l) for l in lines]
    DOCS.write_text("\n".join(lines))
    print(f"docs/MODELS.md: {added} row(s) added; {n} downloads")


def add_variants(kinds):
    """Append the variants of the given kinds (`gguf_mtp`, `gguf_3bit`, `mlx_3bit`) that the
    catalog lacks, right after their family's last entry, pinned at the repo's current commit.
    Nothing else in catalog.toml changes."""
    fams_list = tomllib.loads(SOURCES.read_text())["family"]
    text = OUT.read_text()
    have = {m["id"] for m in tomllib.loads(text)["models"]}
    added = 0
    for f in fams_list:
        for kind in kinds:
            if not f.get(kind):
                continue
            v, errs = build_variant(f, kind, f[kind])
            if not v:
                print(f"✗ {f['id']} [{kind}]  " + "; ".join(errs))
                continue
            if v["id"] in have:
                print(f"  {v['id']:44} already in the catalog")
                continue
            v["license"] = v["license"] or f.get("license")
            blocks = text.split("\n[[models]]\n")
            fam_line = 'family = "' + f["id"] + '"'
            last = max((i for i, b in enumerate(blocks) if fam_line in b.splitlines()), default=None)
            assert last is not None, f"family {f['id']} has no entry to insert after"
            new = "\n".join(variant_lines(f, v)[1:])
            blocks.insert(last + 1, new.rstrip("\n") + "\n")
            text = "\n[[models]]\n".join(blocks)
            have.add(v["id"])
            added += 1
            print(f"✓ {v['id']:44} {v['repo']:40} {v['approx_bytes']/1e9:6.2f} GB  {v['quantization']}")
    OUT.write_text(text)
    print(f"\n{added} variant(s) added; nothing else changed")


def emit_docs(fams, variants):
    """Human-readable library for GitHub readers (same verified data as the app)."""
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
            out.append(doc_row(v))
        out += ["", f"Download: `llmario model pull {fid}` (best variant for your machine) or pick an id above."]
    DOCS.write_text("\n".join(out) + "\n")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return 0
    if "--layouts" in sys.argv:
        self_test()
        add_layouts()
        return 0
    if "--add-mtp" in sys.argv or "--add-3bit" in sys.argv:
        self_test()
        add_variants(["gguf_mtp"] if "--add-mtp" in sys.argv else ["gguf_3bit", "mlx_3bit"])
        sync_docs()
        return 0
    if "--sync-docs" in sys.argv:
        sync_docs()
        return 0
    only = set(sys.argv[sys.argv.index("--only") + 1:]) if "--only" in sys.argv else None
    fams_list = tomllib.loads(SOURCES.read_text())["family"]
    fams = {f["id"]: f for f in fams_list}
    variants, problems = [], []
    for f in fams_list:
        if only and f["id"] not in only:
            continue
        for fmt in ("mlx", "gguf", "gguf_mtp", "gguf_3bit", "mlx_3bit"):
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
