"""Dump reference dequantized values from a GGUF via gguf-py for the engine's tests.
Usage: dequant_ref.py <model.gguf> <out.json> [n_values]"""
import sys, json, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
path, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
r = GGUFReader(path)
seen = {}
tensors = []
for t in r.tensors:
    tname = t.tensor_type.name
    if tname in seen and seen[tname] >= 2:
        continue
    seen[tname] = seen.get(tname, 0) + 1
    raw = np.asarray(t.data)
    try:
        deq = dequantize(raw, t.tensor_type) if tname not in ("F32","F16","BF16") else raw.astype(np.float32)
    except Exception as e:
        print("skip", t.name, tname, e); continue
    flat = np.asarray(deq, dtype=np.float32).reshape(-1)
    # also the last n values so block tails are covered
    tensors.append({"name": t.name, "type": tname, "numel": int(flat.size),
                    "head": [float(x) for x in flat[:n]], "tail": [float(x) for x in flat[-n:]]})
json.dump({"gguf": path, "n": n, "tensors": tensors}, open(out, "w"))
print("wrote", out, len(tensors), "tensors:", sorted(seen))
