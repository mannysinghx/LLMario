#!/usr/bin/env python3
"""Measure memory-pressure relief: what an idle conversation's KV cache costs and what the next
turn costs after it was freed. Run from the repository root.

  scripts/engine/bench_pressure.py <model.gguf> <prompt words> <rounds> [old-engine] [new-engine]

Six configurations per round, each a fresh engine on the CPU with a simulated pressure level
(LLMARIO_SIM_PRESSURE): the old engine, then the new one with no pressure, a warning with and
without the KV disk tier, critical pressure without the tier, and a static 8-bit KV cache with
relief off. Prints the physical footprint after load, after the first turn and after idling
under pressure, and the second turn's time and cached tokens.
"""
import sys, os, time, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_cpu as b
model = sys.argv[1]; words = int(sys.argv[2])
MiB = 2**20
def chat(base, msgs, max_tokens=24):
    body = {"model": "m", "messages": msgs, "stream": False, "temperature": 0, "max_tokens": max_tokens}
    t0 = time.perf_counter()
    r = b.http_json(base + "/v1/chat/completions", body, timeout=3600)
    return time.perf_counter() - t0, r
def run(label, path, extra, sim_level):
    d = tempfile.mkdtemp(prefix="relief-")
    sim = os.path.join(d, "level"); open(sim, "w").write("normal")
    env = {"LLMARIO_SIM_PRESSURE": sim}
    args = [a.replace("@KV", os.path.join(d, "kv")) for a in extra]
    srv = b.Server(os.path.abspath(path), model, 12, 8192, 1, args, os.path.join(d, "log"), env)
    try:
        srv.wait_ready()
        f_load = srv.figures()["footprint"]
        msgs = [{"role": "user", "content": b.prompt_of(words)}]
        t1, r1 = chat(srv.base, msgs)
        u1 = r1["usage"]
        f_after = srv.figures()["footprint"]
        if sim_level:
            open(sim, "w").write(sim_level)
        time.sleep(9)   # idle: relief polls every 2 s; --pressure-idle-secs 3 in these runs
        f_idle = srv.figures()["footprint"]
        st = b.http_json(srv.base + "/engine/stats")
        open(sim, "w").write("normal")
        msgs += [{"role": "assistant", "content": r1["choices"][0]["message"].get("content") or ""},
                 {"role": "user", "content": "Now say it in five words."}]
        t2, r2 = chat(srv.base, msgs)
        u2 = r2["usage"]
        st2 = b.http_json(srv.base + "/engine/stats")
        print(f"{label:28s} load {f_load/MiB:5.0f}M | turn1 {u1['prompt_tokens']} tok {t1:6.2f}s -> {f_after/MiB:5.0f}M | idle+pressure -> {f_idle/MiB:5.0f}M "
              f"(sheds {st.get('memory_pressure',{}).get('sheds','-')}, released {st.get('memory_pressure',{}).get('released_bytes',0)/MiB:.0f}M) | "
              f"turn2 {t2:6.2f}s cached {u2['prompt_tokens_details']['cached_tokens']}/{u2['prompt_tokens']} disk restores {st2['kv_disk']['restores']} "
              f"answer {r2['choices'][0]['message'].get('content','')[:40]!r}", flush=True)
    finally:
        srv.stop(); shutil.rmtree(d, ignore_errors=True)
base = sys.argv[4] if len(sys.argv) > 4 else "target/baseline/llmario-engine"
new = sys.argv[5] if len(sys.argv) > 5 else "target/release/llmario-engine"
tier = ["--kv-cache-dir", "@KV"]
fast = ["--pressure-idle-secs", "3"]
for rnd in range(int(sys.argv[3])):
    run("before (no relief), tier", base, tier, "warning")
    run("relief, no pressure", new, tier + fast, None)
    run("relief, warning, tier", new, tier + fast, "warning")
    run("relief, warning, no tier", new, fast, "warning")
    run("relief, critical, no tier", new, fast, "critical")
    run("static q8_0 KV, no relief", new, tier + ["--kv-type", "q8_0", "--pressure-relief", "off"], "warning")
