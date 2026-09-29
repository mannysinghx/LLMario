#!/usr/bin/env python3
"""Print Markdown comparison tables (one per concurrency level) for the llmario bench JSON
reports in a directory. Usage: scripts/bench-table.py benchmarks/results/<date>"""
import glob, json, os, sys

def rows(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        q = r["quality"]
        for l in r["levels"]:
            out.append(dict(
                file=os.path.basename(f)[:-5] + ".md", label=r["label"], conc=l["concurrency"],
                ttft50=l["ttft_s"]["p50"] * 1000, ttft95=l["ttft_s"]["p95"] * 1000,
                dec=l["decode_tps"]["mean"], agg=l["aggregate_decode_tps"],
                peak=(l["peak_memory_bytes"] or 0) / 2**30,
                est=(r.get("estimated_memory_bytes") or 0) / 2**30,
                quality=f"{sum(x['passed'] for x in q)}/{len(q)}" if q else "–"))
    return out

def table(rs):
    t = ["| Target | TTFT p50 | TTFT p95 | Decode tok/s / req | Aggregate tok/s | Peak footprint | Estimate | Quality | Report |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rs:
        est = f"{r['est']:.2f} GiB" if r["est"] else "–"
        t.append(f"| {r['label']} | {r['ttft50']:.0f} ms | {r['ttft95']:.0f} ms | {r['dec']:.1f} | {r['agg']:.1f} "
                 f"| {r['peak']:.2f} GiB | {est} | {r['quality']} | [md]({r['file']}) |")
    return "\n".join(t)

if __name__ == "__main__":
    rs = rows(sys.argv[1])
    for c in sorted({r["conc"] for r in rs}):
        print(f"### Concurrency {c}\n\n{table([r for r in rs if r['conc'] == c])}\n")
