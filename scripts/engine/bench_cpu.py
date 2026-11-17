#!/usr/bin/env python3
"""Reproducible CPU benchmark of the native engine (and, optionally, llama.cpp on the CPU).

Standard library only. Measures, per engine binary and model:

  raw     prefill and decode tok/s from `llmario-engine raw-run` over a thread sweep
  serve   load time (cold page cache and warm), memory after load and after prompts, disk bytes
          read, CPU time, time to first token and decode tok/s over HTTP at several prompt
          lengths, and a small fixed correctness set (temperature 0)
  conc    aggregate and per-request decode tok/s with 1, 2, 4 ... concurrent requests
  llama   `llama-bench` strictly on the CPU for the same model, when llama-bench is on PATH

Several `--engine label=path` binaries are run interleaved, round by round, so a shared machine
disturbs them equally. Nothing is invented: a step that cannot run is recorded as `"untested"`
with the reason.

Examples:
  scripts/engine/bench_cpu.py --engine base=target/baseline/llmario-engine \
      --engine new=target/release/llmario-engine --model /path/a.gguf --model /path/b.gguf \
      --modes raw,serve,llama --threads 8,12,16 --out benchmarks/results/<dir>

Process figures come from `proc_pid_rusage` on macOS and `/proc/<pid>` on Linux.
"""

import argparse
import concurrent.futures
import ctypes
import ctypes.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

PASSAGE = (
    "The lighthouse keeper climbed the spiral stairs every evening before dusk, carrying a small "
    "brass lamp and a notebook in which he recorded the weather, the tides and the ships that "
    "passed the headland. In forty years he had missed only three nights, each of them during the "
    "great storm that tore the roof from the boathouse and scattered the fishing fleet along the "
    "coast. When the supply boat failed to arrive in the second winter of the war, he rationed "
    "the oil, trimmed the wick shorter and kept the light burning for half of each night, "
    "choosing the hours when the channel was most dangerous. The harbour master later wrote that "
    "no vessel was lost on the rocks in those months, although several captains complained that "
    "the light seemed to them fainter than the charts had promised. "
)

QUALITY = [
    ("What is the capital of France? Answer with one word.", ["paris"]),
    ("What is 17 + 26? Answer with the number only.", ["43"]),
    ("What is the chemical symbol for gold? Answer with the symbol only.", ["au"]),
    ("How many days are in a leap year? Answer with the number only.", ["366"]),
    ("Which planet is known as the Red Planet? Answer with one word.", ["mars"]),
    ("What is the opposite of 'hot'? Answer with one word.", ["cold"]),
    ("What is 9 times 8? Answer with the number only.", ["72"]),
    ("Who wrote the play 'Romeo and Juliet'? Answer with the surname only.", ["shakespeare"]),
    ("What is the largest ocean on Earth? Answer with one word.", ["pacific"]),
    ("What is the square root of 144? Answer with the number only.", ["12"]),
    ("Spell the word 'cat' backwards. Answer with the three letters only.", ["tac"]),
    ("What gas do plants absorb from the air for photosynthesis? Answer briefly.",
     ["carbon dioxide", "co2", "co₂"]),
]


# ---------------------------------------------------------------- process figures

class RusageV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
            "proc_exit_abstime", "child_user_time", "child_system_time", "child_pkg_idle_wkups",
            "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
            "diskio_bytesread", "diskio_byteswritten", "cpu_time_qos_default",
            "cpu_time_qos_maintenance", "cpu_time_qos_background", "cpu_time_qos_utility",
            "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
            "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time",
            "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles",
            "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time",
        )
    ]


class Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_TB = None


def proc_figures(pid):
    """Memory, CPU time and disk reads of a live process, or {} when unavailable."""
    if sys.platform == "darwin":
        global _TB
        if _TB is None:
            tb = Timebase()
            _LIBC.mach_timebase_info(ctypes.byref(tb))
            _TB = tb.numer / tb.denom / 1e9
        ru = RusageV4()
        if _LIBC.proc_pid_rusage(pid, 4, ctypes.byref(ru)) != 0:
            return {}
        qos = {
            k: round(getattr(ru, "cpu_time_qos_" + k) * _TB, 3)
            for k in ("default", "utility", "legacy", "user_initiated", "user_interactive",
                      "background")
        }
        return {
            "footprint": ru.phys_footprint,
            "footprint_peak": ru.lifetime_max_phys_footprint,
            "resident": ru.resident_size,
            "cpu_s": (ru.user_time + ru.system_time) * _TB,
            "disk_read": ru.diskio_bytesread,
            "pageins": ru.pageins,
            "cpu_s_by_qos": {k: v for k, v in qos.items() if v > 0},
        }
    try:
        out = {}
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "VmHWM", "RssAnon", "Threads"):
                    out[k] = int(v.split()[0]) * (1 if k == "Threads" else 1024)
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        tick = os.sysconf("SC_CLK_TCK")
        cpu = (int(parts[11]) + int(parts[12])) / tick
        disk = None
        try:
            with open(f"/proc/{pid}/io") as f:
                for line in f:
                    if line.startswith("read_bytes:"):
                        disk = int(line.split()[1])
        except OSError:
            pass
        return {
            "footprint": out.get("RssAnon"),
            "footprint_peak": None,
            "resident": out.get("VmRSS"),
            "resident_peak": out.get("VmHWM"),
            "cpu_s": cpu,
            "disk_read": disk,
            "threads": out.get("Threads"),
        }
    except OSError:
        return {}


def thread_count(pid):
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["ps", "-M", "-p", str(pid)], capture_output=True, text=True)
            return max(0, len(out.stdout.strip().splitlines()) - 1)
        return proc_figures(pid).get("threads")
    except OSError:
        return None


def evict(path):
    """Drop `path` from the page cache without privileges; returns the share still resident."""
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        if sys.platform == "darwin":
            # Python's mmap gives no address, so map through libc for msync and mincore.
            _LIBC.mmap.restype = ctypes.c_void_p
            _LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_longlong]
            p = _LIBC.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
            if p in (None, ctypes.c_void_p(-1).value):
                return None
            _LIBC.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            _LIBC.msync(p, size, 2)  # MS_INVALIDATE
            resident = _resident_share(p, size)
            _LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            _LIBC.munmap(p, size)
            return resident
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return None
    finally:
        os.close(fd)


def _resident_share(p, size):
    page = os.sysconf("SC_PAGE_SIZE")
    n = (size + page - 1) // page
    vec = (ctypes.c_char * n)()
    _LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    if _LIBC.mincore(p, size, vec) != 0:
        return None
    return sum(1 for b in vec.raw if b & 1) / n


# ---------------------------------------------------------------- helpers

QUIET = {"idle_pct": 0.0, "wait_s": 0}


def idle_pct():
    """System-wide idle CPU share over about a second (None when it cannot be read)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["top", "-l", "2", "-n", "0", "-s", "1"], capture_output=True,
                                 text=True).stdout
            m = re.findall(r"CPU usage:.*?([\d.]+)% idle", out)
            return float(m[-1]) if m else None

        def snap():
            with open("/proc/stat") as f:
                v = [int(x) for x in f.readline().split()[1:]]
            return v[3] + v[4], sum(v)
        i0, t0 = snap()
        time.sleep(1)
        i1, t1 = snap()
        return 100.0 * (i1 - i0) / max(1, t1 - t0)
    except (OSError, ValueError):
        return None


def wait_quiet():
    """Wait (up to --quiet-wait) for the machine to be at least --quiet-idle percent idle;
    returns the idle share the measurement started with."""
    deadline = time.time() + QUIET["wait_s"]
    while True:
        idle = idle_pct()
        if idle is None or idle >= QUIET["idle_pct"] or time.time() >= deadline:
            return idle
        time.sleep(3)

def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def environment():
    env = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "load_average": list(os.getloadavg()),
    }
    if sys.platform == "darwin":
        for key, name in (("machdep.cpu.brand_string", "cpu"), ("hw.memsize", "ram_bytes"),
                          ("hw.perflevel0.physicalcpu", "performance_cores"),
                          ("hw.perflevel1.physicalcpu", "efficiency_cores")):
            v = sh(["sysctl", "-n", key]).stdout.strip()
            env[name] = int(v) if v.isdigit() else v
        vm = sh(["vm_stat"]).stdout
        page = int(re.search(r"page size of (\d+)", vm).group(1))
        stat = {m.group(1): int(m.group(2)) * page
                for m in re.finditer(r"Pages ([\w ]+):\s+(\d+)\.", vm)}
        env["ram_free_bytes"] = stat.get("free", 0) + stat.get("inactive", 0) + stat.get(
            "speculative", 0)
        env["ram_wired_bytes"] = stat.get("wired down")
    else:
        try:
            with open("/proc/meminfo") as f:
                mem = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in f}
            env["ram_bytes"] = mem.get("MemTotal")
            env["ram_free_bytes"] = mem.get("MemAvailable")
            with open("/proc/cpuinfo") as f:
                m = re.search(r"model name\s*:\s*(.+)", f.read())
            env["cpu"] = m.group(1) if m else None
        except OSError:
            pass
    return env


def storage_of(path):
    info = {"path_volume": None}
    if sys.platform == "darwin":
        df = sh(["df", path]).stdout.strip().splitlines()
        if len(df) > 1:
            dev = df[1].split()[0]
            info["path_volume"] = df[1].split()[-1]
            d = sh(["diskutil", "info", dev]).stdout
            for key in ("Solid State", "Protocol", "File System Personality", "Device Location"):
                m = re.search(rf"{key}:\s+(.+)", d)
                if m:
                    info[key.lower().replace(" ", "_")] = m.group(1).strip()
    return info


def model_info(engine, path):
    out = sh([engine, "inspect", path]).stdout
    info = {"file": os.path.basename(path), "bytes": os.path.getsize(path)}
    for key in ("general.name", "general.architecture", "general.file_type",
                "general.size_label", "general.quantization_version"):
        m = re.search(rf"^{re.escape(key)} = (.+)$", out, re.M)
        if m:
            info[key] = m.group(1).strip()
    m = re.search(r"context_length = (\d+)", out)
    if m:
        info["context_length"] = int(m.group(1))
    types = re.findall(r"^\s*(\w+):\s+([\d.]+) MiB \(([\d.]+)%\)", out, re.M)
    info["bytes_by_type_pct"] = {t: float(p) for t, _, p in types}
    return info


def tokens_for(engine, model, n):
    text = PASSAGE * (n // 100 + 2)
    out = sh([engine, "tokenize", "--model", model, "--text", text]).stdout.splitlines()
    ids = [t for t in out[0].split(",") if t.strip()]
    if len(ids) < n:
        raise RuntimeError(f"only {len(ids)} tokens for a {n}-token prompt")
    return ",".join(ids[:n])


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def http_json(url, body=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat_stream(base, prompt, max_tokens, timeout=1800):
    """One streamed chat request; client-side TTFT plus the server's own timings."""
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "stream": True,
            "temperature": 0, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    text, reasoning, usage, timings, pieces = "", "", None, None, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    pieces += 1
                text += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
    end = time.perf_counter()
    out = {"ttft_s": (first - t0) if first else None, "total_s": end - t0, "text": text,
           "reasoning": reasoning, "usage": usage, "timings": timings}
    if timings and timings.get("predicted_ms"):
        out["decode_tok_s"] = timings["predicted_n"] / (timings["predicted_ms"] / 1000)
        if timings.get("prompt_ms"):
            out["prefill_tok_s"] = timings["prompt_n"] / (timings["prompt_ms"] / 1000)
    elif first and last and pieces > 1 and last > first:
        out["decode_tok_s"] = (pieces - 1) / (last - first)
    return out


# ---------------------------------------------------------------- modes

RAW_RE = {
    "load_s": re.compile(r"loaded .* in ([\d.]+)s"),
    "prefill_tok_s": re.compile(r"prefill \d+ tokens in [\d.]+s \(([\d.]+) tok/s\)"),
    "decode_tok_s": re.compile(r"decode \d+ tokens in [\d.]+s \(([\d.]+) tok/s\)"),
}


def run_raw(engine, model, tokens, threads, n_decode, ctx, extra_env=None):
    cmd = [engine, "raw-run", "--model", model, "--tokens", tokens, "--n", str(n_decode + 1),
           "--ctx", str(ctx), "--device", "cpu"]
    if threads:
        cmd += ["--threads", str(threads)]
    env = dict(os.environ, **(extra_env or {}))
    idle = wait_quiet()
    t0 = time.perf_counter()
    p = sh(cmd, env=env)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        return {"error": p.stderr.strip()[-400:]}
    out = {"wall_s": round(wall, 3), "idle_before_pct": idle,
           "output_ids": p.stdout.strip().splitlines()[0][:400]}
    for k, rx in RAW_RE.items():
        m = rx.search(p.stderr)
        if m:
            out[k] = float(m.group(1))
    m = re.search(r"threads (\d+)", p.stderr)
    if m:
        out["threads_used"] = int(m.group(1))
    return out


def run_llama_bench(model, threads, prompt_tokens, n_decode, reps):
    exe = shutil.which("llama-bench")
    if not exe:
        return {"untested": "llama-bench not on PATH"}
    base = [exe, "-m", model, "-ngl", "0", "-dev", "none", "-nopo", "1", "-nkvo", "1",
            "-t", str(threads), "-r", str(reps), "-o", "json"]
    res = {"version": sh([exe, "--version"]).stderr.strip().splitlines()[-2:] or None}
    for name, args in (("prefill_tok_s", ["-p", str(prompt_tokens), "-n", "0"]),
                       ("decode_tok_s", ["-p", "0", "-n", str(n_decode), "-d",
                                         str(prompt_tokens)])):
        p = sh(base + args)
        try:
            rows = json.loads(p.stdout[p.stdout.index("["):])
            res[name] = round(rows[0]["avg_ts"], 2)
            res[name + "_stddev"] = round(rows[0].get("stddev_ts", 0.0), 2)
        except (ValueError, KeyError, IndexError):
            res[name] = None
            res[name + "_error"] = (p.stderr or p.stdout).strip()[-300:]
    return res


class Server:
    def __init__(self, engine, model, threads, ctx, parallel, extra, log_path, extra_env=None):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        cmd = [engine, "serve", "--model", model, "--listen", f"127.0.0.1:{self.port}",
               "--ctx", str(ctx), "--parallel", str(parallel), "--device", "cpu"] + extra
        if threads:
            cmd += ["--threads", str(threads)]
        self.cmd = cmd
        self.log = open(log_path, "w")
        self.t0 = time.perf_counter()
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, **(extra_env or {})))
        self.ready_s = None

    def wait_ready(self, timeout=1800):
        while time.perf_counter() - self.t0 < timeout:
            if self.p.poll() is not None:
                raise RuntimeError(f"engine exited with {self.p.returncode}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        self.ready_s = time.perf_counter() - self.t0
                        return self.ready_s
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("engine did not become ready")

    def figures(self):
        return proc_figures(self.p.pid)

    def stop(self):
        try:
            http_json(self.base + "/engine/control", {"action": "exit"}, timeout=5)
        except Exception:  # noqa: BLE001 - the process is stopped below either way
            pass
        try:
            self.p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.p.terminate()
            self.p.wait(timeout=15)
        self.log.close()


def prompt_of(words):
    reps = max(1, words // len(PASSAGE.split()))
    return (PASSAGE * reps) + "\nSummarize the passage above in one sentence."


def run_serve(engine, model, args, out_dir, tag, cold, extra_env=None):
    res = {"cold": cold, "idle_before_pct": wait_quiet()}
    if cold:
        res["resident_after_evict"] = evict(model)
    log_path = os.path.join(out_dir, f"{tag}.log")
    srv = Server(engine, model, args.serve_threads, args.ctx, 1, args.serve_arg, log_path,
                 extra_env)
    try:
        res["load_s"] = round(srv.wait_ready(), 3)
        res["cmd"] = " ".join(srv.cmd)
        f = srv.figures()
        res["after_load"] = f
        res["threads_os"] = thread_count(srv.p.pid)
        try:
            res["engine_model"] = http_json(srv.base + "/v1/models")["data"][0]["engine"]
            plan = http_json(srv.base + "/engine/plan")["plan"]
            res["plan"] = {k: plan.get(k) for k in ("planned_peak", "ctx_per_slot", "kv_dtype",
                                                    "slots", "degradations", "fits")}
        except Exception as e:  # noqa: BLE001
            res["plan_error"] = str(e)
        # First request after load (after a cold load the weights were paged in by the warm-up).
        res["requests"] = []
        for words in args.prompt_words:
            idle = wait_quiet()
            before = srv.figures()
            t0 = time.perf_counter()
            r = chat_stream(srv.base, prompt_of(words), args.max_tokens)
            wall = time.perf_counter() - t0
            r["idle_before_pct"] = idle
            after = srv.figures()
            r.pop("text"), r.pop("reasoning")
            r["prompt_words"] = words
            r["avg_cores"] = round((after.get("cpu_s", 0) - before.get("cpu_s", 0)) / wall, 2)
            r["footprint_after"] = after.get("footprint")
            r["resident_after"] = after.get("resident")
            res["requests"].append(r)
        if not args.no_quality:
            hits, answers = 0, []
            for q, accept in QUALITY:
                r = chat_stream(srv.base, q, 400)
                ans = r["text"].strip()
                ok = any(a in ans.lower() for a in accept)
                hits += ok
                answers.append({"q": q, "answer": ans[:120], "ok": ok})
            res["quality"] = {"correct": hits, "of": len(QUALITY), "answers": answers}
        end = srv.figures()
        res["at_end"] = end
        res["stats"] = http_json(srv.base + "/engine/stats")
    except Exception as e:  # noqa: BLE001
        res["error"] = str(e)
    finally:
        srv.stop()
    return res


def run_concurrency(engine, model, args, out_dir, tag, parallel, extra_env=None):
    res = {"parallel": parallel, "idle_before_pct": wait_quiet()}
    srv = Server(engine, model, args.serve_threads, args.ctx, parallel, args.serve_arg,
                 os.path.join(out_dir, f"{tag}.log"), extra_env)
    try:
        res["load_s"] = round(srv.wait_ready(), 3)
        res["after_load"] = srv.figures()
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
            rs = list(ex.map(lambda i: chat_stream(srv.base, f"Request {i}. " + prompt_of(150),
                                                   args.max_tokens), range(parallel)))
        wall = time.perf_counter() - t0
        toks = sum((r["usage"] or {}).get("completion_tokens", 0) for r in rs)
        res.update({
            "wall_s": round(wall, 3),
            "aggregate_tok_s": round(toks / wall, 2),
            "ttft_s": sorted(round(r["ttft_s"], 3) for r in rs if r["ttft_s"]),
            "decode_tok_s": sorted(round(r.get("decode_tok_s", 0), 2) for r in rs),
            "at_end": srv.figures(),
        })
    except Exception as e:  # noqa: BLE001
        res["error"] = str(e)
    finally:
        srv.stop()
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", action="append", required=True, metavar="LABEL=PATH[,KEY=VAL...]",
                    help="engine binary; KEY=VAL pairs after the path are extra environment")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--modes", default="raw,serve", help="raw,serve,conc,llama")
    ap.add_argument("--threads", default="12", help="thread sweep for raw (0 = engine default)")
    ap.add_argument("--serve-threads", type=int, default=0, help="0 = the engine's default")
    ap.add_argument("--serve-arg", action="append", default=[], help="extra `serve` argument")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-words", default="40,700,2800",
                    type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--concurrency", default="1,2,4", help="parallel requests (conc mode)")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--no-cold", action="store_true")
    ap.add_argument("--quiet-idle", type=float, default=0.0,
                    help="wait until the machine is at least this percent idle before each run")
    ap.add_argument("--quiet-wait", type=int, default=120, help="longest such wait, seconds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    QUIET.update(idle_pct=args.quiet_idle, wait_s=args.quiet_wait)

    engines = []
    for spec in args.engine:
        label, _, rest = spec.partition("=")
        path, *pairs = rest.split(",")
        env = dict(p.split("=", 1) for p in pairs)
        engines.append((label, os.path.abspath(path), env))
    modes = args.modes.split(",")
    threads = [int(t) for t in args.threads.split(",")]
    os.makedirs(args.out, exist_ok=True)
    report = {"schema": 1, "environment": environment(), "args": vars(args) | {"engine": [
        {"label": l, "path": p, "env": e, "version": sh([p, "--version"]).stdout.strip()}
        for l, p, e in engines]}, "models": []}

    for model in args.model:
        model = os.path.abspath(model)
        m = {"info": model_info(engines[0][1], model), "storage": storage_of(model)}
        name = os.path.splitext(os.path.basename(model))[0]
        print(f"== {name}", flush=True)
        if "raw" in modes:
            toks = tokens_for(engines[0][1], model, args.prompt_tokens)
            raw = {label: {str(t): [] for t in threads} for label, _, _ in engines}
            for rnd in range(args.rounds):
                for t in threads:
                    for label, path, env in engines:
                        r = run_raw(path, model, toks, t, args.decode_tokens,
                                    args.prompt_tokens + args.decode_tokens + 64, env)
                        raw[label][str(t)].append(r)
                        print(f"  raw r{rnd} {label} t={t}: prefill {r.get('prefill_tok_s')} "
                              f"decode {r.get('decode_tok_s')} idle {r.get('idle_before_pct')} "
                              f"{r.get('error', '')}", flush=True)
            m["raw"] = raw
        if "llama" in modes:
            m["llama_bench_cpu"] = {str(t): run_llama_bench(model, t, args.prompt_tokens,
                                                            args.decode_tokens, args.rounds)
                                    for t in threads if t > 0}
            print(f"  llama-bench: {json.dumps(m['llama_bench_cpu'])[:300]}", flush=True)
        if "serve" in modes:
            serve = {label: [] for label, _, _ in engines}
            plans = ([True] if not args.no_cold else []) + [False]
            for rnd in range(args.rounds):
                for cold in plans:
                    for label, path, env in engines:
                        tag = f"{name}.{label}.{'cold' if cold else 'warm'}.r{rnd}"
                        r = run_serve(path, model, args, args.out, tag, cold, env)
                        serve[label].append(r)
                        q = r.get("quality", {})
                        print(f"  serve r{rnd} {label} {'cold' if cold else 'warm'}: load "
                              f"{r.get('load_s')}s footprint "
                              f"{(r.get('after_load') or {}).get('footprint')} first-ttft "
                              f"{[round(x['ttft_s'], 3) for x in r.get('requests', []) if x.get('ttft_s')]} "
                              f"quality {q.get('correct')}/{q.get('of')} {r.get('error', '')}",
                              flush=True)
            m["serve"] = serve
        if "conc" in modes:
            conc = {label: [] for label, _, _ in engines}
            for rnd in range(args.rounds):
                for par in [int(c) for c in args.concurrency.split(",")]:
                    for label, path, env in engines:
                        r = run_concurrency(path, model, args, args.out,
                                            f"{name}.{label}.par{par}.r{rnd}", par, env)
                        conc[label].append(r)
                        print(f"  concurrency {par} r{rnd} {label}: "
                              f"{r.get('aggregate_tok_s')} tok/s, per request "
                              f"{r.get('decode_tok_s')}, ttft {r.get('ttft_s')} "
                              f"{r.get('error', '')}", flush=True)
            m["concurrency"] = conc
        report["models"].append(m)
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
    report["environment"]["load_average_end"] = list(os.getloadavg())
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"report: {os.path.join(args.out, 'report.json')}")


if __name__ == "__main__":
    main()
