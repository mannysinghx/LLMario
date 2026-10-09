import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
out = sys.argv[2]

def block(name):
    m = re.search(name + r"\s*=\s*\{(.*?)\n\};", src, re.S)
    assert m, name
    return m.group(1)

pairs = re.findall(r"\{\s*(0x[0-9A-Fa-f]+),\s*(0x[0-9A-Fa-f]+)\s*\}", block("unicode_ranges_flags"))
ws = re.findall(r"(0x[0-9A-Fa-f]+)", block("unicode_set_whitespace"))
lower = re.findall(r"\{\s*(0x[0-9A-Fa-f]+),\s*(0x[0-9A-Fa-f]+)\s*\}", block("unicode_map_lowercase"))

with open(out, "w", encoding="utf-8") as f:
    f.write("//! Generated from llama.cpp `src/unicode-data.cpp` (commit 7fe450e19, MIT) by\n")
    f.write("//! `crates/engine/tokenizer/tools/gen_unicode_data.py`. Do not edit by hand.\n//!\n")
    f.write("//! `RANGES_FLAGS` lists `(first codepoint, flags)` with each entry valid until the next entry's\n")
    f.write("//! start; the last entry starts at `0x110000`. Flag bits match `unicode_cpt_flags`.\n\n")
    f.write("/// `(start, flags)` sorted by start; see module docs.\n")
    f.write("pub const RANGES_FLAGS: &[(u32, u16)] = &[\n")
    for a, b in pairs:
        f.write(f"    ({a}, {b}),\n")
    f.write("];\n\n/// Codepoints llama.cpp treats as `\\s`.\npub const WHITESPACE: &[u32] = &[\n")
    for w in ws:
        f.write(f"    {w},\n")
    f.write("];\n\n/// `(codepoint, lowercase)` sorted by codepoint.\npub const LOWERCASE: &[(u32, u32)] = &[\n")
    for a, b in lower:
        f.write(f"    ({a}, {b}),\n")
    f.write("];\n")
print(len(pairs), len(ws), len(lower))
