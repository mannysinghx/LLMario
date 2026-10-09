"""Reference values for the engine's safetensors/MLX reader and MLX affine dequantiser.

Reads an MLX (or plain HF) safetensors folder with its own 60-line parser plus numpy, dequantises
rows of a few quantised tensors with an independent numpy implementation of `w = scale * q + bias`
(q = `bits`-wide unsigned ints read LSB-first from the little-endian u32 words), cross-checks them
against `mlx.core.dequantize` when `mlx` is importable, and writes a JSON the Rust test
`crates/engine/formats/tests/mlx_ref.rs` compares against (`LLMARIO_TEST_MLX_REF=<out.json>`).

Usage:
  mlx_ref.py <folder> <out.json> [--rows N] [--values N] [--no-mlx] [--modules a,b,...]
  mlx_ref.py --fixtures            # print the Rust MLX_FIXTURES table (needs mlx)

Dependencies: numpy; optional mlx (MIT). No safetensors package needed.
"""
import json
import os
import struct
import sys

import numpy as np

ELEM = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U32": 4}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def to_f32(raw, dtype):
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4")
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32)
    if dtype == "F64":
        return np.frombuffer(raw, dtype="<f8").astype(np.float32)
    raise ValueError(dtype)


def unpack_bits(words_u32, bits):
    """Read `bits`-wide unsigned ints from a row of little-endian u32 words, LSB first."""
    b = np.asarray(words_u32, dtype="<u4").tobytes()
    stream = np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little")
    n = stream.size // bits
    stream = stream[: n * bits].reshape(n, bits).astype(np.uint32)
    return (stream * (1 << np.arange(bits, dtype=np.uint32))).sum(axis=1)


def dequant_rows(wq_rows, scales_rows, biases_rows, bits, group_size):
    """numpy reference: f32 multiply then f32 add, like the engine (no fused multiply-add)."""
    rows = wq_rows.shape[0]
    cols = wq_rows.shape[1] * 32 // bits
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        q = unpack_bits(wq_rows[r], bits).astype(np.float32)
        s = np.repeat(scales_rows[r], group_size)
        b = np.repeat(biases_rows[r], group_size)
        out[r] = (s * q) + b
    return out


class Folder:
    def __init__(self, d):
        self.dir = d
        self.config = json.load(open(os.path.join(d, "config.json")))
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.isfile(idx):
            self.index = json.load(open(idx))
            files = sorted(set(self.index["weight_map"].values()))
        else:
            self.index = None
            files = ["model.safetensors"] if os.path.isfile(os.path.join(d, "model.safetensors")) \
                else sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
        self.shards = []
        self.tensors = {}
        for i, f in enumerate(files):
            p = os.path.join(d, f)
            h, data_off = read_header(p)
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            self.shards.append((f, mm, data_off, h.get("__metadata__")))
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                assert k not in self.tensors, f"duplicate {k}"
                self.tensors[k] = (i, v["dtype"], v["shape"], v["data_offsets"])
        q = self.config.get("quantization") or self.config.get("quantization_config")
        self.quant = q if isinstance(q, dict) and "quant_method" not in q and "bits" in q else None

    def raw(self, name):
        i, dtype, shape, (b, e) = self.tensors[name]
        _, mm, off, _ = self.shards[i]
        return bytes(mm[off + b: off + e]), dtype, shape

    def params(self, module):
        o = self.quant.get(module)
        if o is False:
            return None
        if isinstance(o, dict):
            return o.get("bits", self.quant["bits"]), o.get("group_size", self.quant["group_size"])
        return self.quant["bits"], self.quant["group_size"]

    def quant_rows(self, module, rows_idx):
        """(wq rows [n, packed], scales rows, biases rows, bits, gs, scale dtype, hf shapes)."""
        bits, gs = self.params(module)
        wraw, wdt, wshape = self.raw(module + ".weight")
        sraw, sdt, sshape = self.raw(module + ".scales")
        braw, bdt, bshape = self.raw(module + ".biases")
        assert wdt == "U32" and sdt == bdt and sshape == bshape, module
        wq = np.frombuffer(wraw, dtype="<u4").reshape(wshape)
        sc = to_f32(sraw, sdt).reshape(sshape)
        bi = to_f32(braw, bdt).reshape(bshape)
        cols = wshape[1] * 32 // bits
        assert sshape == [wshape[0], cols // gs], (module, wshape, sshape, bits, gs)
        return wq[rows_idx], sc[rows_idx], bi[rows_idx], bits, gs, sdt, (wshape, sshape), (sraw, braw, rows_idx)


def mlx_check(wq_rows, sc_rows, bi_rows, bits, gs, scale_dtype, raw_sb, ours):
    """Cross-check with mlx.core.dequantize: f32-cast scales/biases (expect bit-exact against the
    numpy f32 path up to fma rounding) and the native dtype (bf16/f16 output rounding)."""
    try:
        import mlx.core as mx
    except ImportError:
        return {"available": False}
    w = mx.array(wq_rows.astype(np.uint32))
    ref32 = np.array(mx.dequantize(w, mx.array(sc_rows), mx.array(bi_rows), group_size=gs, bits=bits))
    d32 = float(np.abs(ref32 - ours).max())
    res = {"available": True, "max_abs_diff_f32": d32, "exact_f32": bool(d32 == 0.0),
           "mismatch_frac_f32": float((ref32 != ours).mean())}
    sraw, braw, rows_idx = raw_sb
    if scale_dtype in ("F16", "BF16"):
        mdt = mx.float16 if scale_dtype == "F16" else mx.bfloat16
        s_native = mx.array(np.frombuffer(sraw, dtype="<u2").reshape(-1)).view(mdt).reshape(-1, sc_rows.shape[1])[rows_idx]
        b_native = mx.array(np.frombuffer(braw, dtype="<u2").reshape(-1)).view(mdt).reshape(-1, bi_rows.shape[1])[rows_idx]
        refn = np.array(mx.dequantize(w, s_native, b_native, group_size=gs, bits=bits).astype(mx.float32))
        res["max_abs_diff_native"] = float(np.abs(refn - ours).max())
        res["native_dtype"] = scale_dtype
    return res


def pick_modules(folder, limit):
    mods = [k[: -len(".scales")] for k in folder.tensors if k.endswith(".scales")]
    want = []
    for m in sorted(mods):
        if "embed_tokens" in m or m.endswith("lm_head") or ".layers.0." in m or ".layers.3." in m:
            want.append(m)
    if not want:
        want = sorted(mods)[:limit]
    return want[:limit]


def main():
    args = sys.argv[1:]
    if args and args[0] == "--fixtures":
        return fixtures()
    folder, out = args[0], args[1]
    rows = int(opt(args, "--rows", 4))
    values = int(opt(args, "--values", 2048))
    use_mlx = "--no-mlx" not in args
    f = Folder(folder)
    modules = opt(args, "--modules", None)
    modules = modules.split(",") if modules else pick_modules(f, 12)
    total = sum(e - b for (_, _, _, (b, e)) in f.tensors.values())
    report = {"folder": os.path.abspath(folder), "rows": rows, "values": values,
              "quantization": f.quant, "tensor_count": len(f.tensors), "total_bytes": total,
              "index_total_size": (f.index or {}).get("metadata", {}).get("total_size"),
              "shards": [s[0] for s in f.shards], "shard_metadata": [s[3] for s in f.shards],
              "tensors": []}
    for m in modules:
        if f.params(m) is None:
            continue
        wshape = f.tensors[m + ".weight"][2]
        n = min(rows, wshape[0])
        head_idx = list(range(n))
        tail_idx = list(range(wshape[0] - n, wshape[0]))
        entry = {"module": m, "kind": "mlx"}
        for label, idx in (("head", head_idx), ("tail", tail_idx)):
            wq, sc, bi, bits, gs, sdt, shapes, raw_sb = f.quant_rows(m, idx)
            ours = dequant_rows(wq, sc, bi, bits, gs)
            entry.update({"bits": bits, "group_size": gs, "scale_dtype": sdt,
                          "weight_shape": shapes[0], "scales_shape": shapes[1],
                          "rows": shapes[0][0], "cols": ours.shape[1]})
            entry[label + "_rows"] = [idx[0], len(idx)]
            entry[label] = [float(x) for x in ours.reshape(-1)[:values]]
            if use_mlx:
                entry[label + "_mlx"] = mlx_check(wq, sc, bi, bits, gs, sdt, raw_sb, ours)
        report["tensors"].append(entry)
        print(m, entry["bits"], "bit g", entry["group_size"], entry["scale_dtype"], entry["weight_shape"],
              "mlx:", entry.get("head_mlx"))
    # A few plain float tensors (norms) too.
    floats = [k for k, v in f.tensors.items() if v[1] in ("F32", "F16", "BF16") and not k.endswith((".scales", ".biases"))]
    for name in sorted(floats)[:3]:
        raw, dtype, shape = f.raw(name)
        vals = to_f32(raw, dtype)
        report["tensors"].append({"name": name, "kind": "float", "dtype": dtype, "hf_shape": shape,
                                  "numel": int(vals.size),
                                  "head": [float(x) for x in vals[:values]],
                                  "tail": [float(x) for x in vals[-values:]]})
        print(name, dtype, shape)
    json.dump(report, open(out, "w"))
    print("wrote", out, len(report["tensors"]), "tensors;", len(f.tensors), "tensors in", len(f.shards), "shards;", total, "bytes")


def opt(args, key, default):
    return args[args.index(key) + 1] if key in args else default


def fixtures():
    """MLX-produced packed rows with unit scales, for core's dequant tests (mlx required)."""
    import mlx.core as mx
    print("const MLX_FIXTURES: &[(u32, &[u32], &[u32])] = &[")
    for bits in (2, 3, 4, 5, 6, 8):
        cols = 96 if bits != 2 else 64
        q = (np.arange(cols) * 7 + 3) % (1 << bits)
        w = q.astype(np.float32)[None, :]
        w[0, ::32] = 0
        w[0, 1::32] = (1 << bits) - 1
        wq, sc, bi = mx.quantize(mx.array(w), group_size=32, bits=bits)
        wq = np.array(wq)[0]
        dec = unpack_bits(wq, bits)
        ref = np.array(mx.dequantize(mx.array(wq[None, :]), sc, bi, group_size=32, bits=bits))[0]
        assert np.array_equal(np.repeat(np.array(sc)[0], 32) * dec + np.repeat(np.array(bi)[0], 32), ref)
        assert np.array_equal(ref, w[0]) and np.all(np.array(sc) == -1.0) and np.all(np.array(bi) == (1 << bits) - 1)
        print(f"    ({bits}, &[{', '.join(f'0x{x:08x}' for x in wq)}], &[{', '.join(str(int(x)) for x in dec)}]),")
    print("];")


if __name__ == "__main__":
    main()
