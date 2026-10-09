# llmario-engine-tokenizer

Tokenizer of the LLMario native inference engine. It is built from GGUF metadata alone (no
`tokenizer.json`, no Python, no external files) and matches llama.cpp token for token.

## Responsibility

- Read `tokenizer.ggml.*` from a `GgufFile` (`llmario-engine-formats`): vocabulary, scores,
  token types, merges, special ids, `add_bos_token` / `add_eos_token` / `add_space_prefix` /
  `remove_extra_whitespaces`, and `tokenizer.chat_template`.
- Encode text with
  - byte-level BPE (`tokenizer.ggml.model = "gpt2"`, and Gemma 4's `"gemma4"` variant that
    works on raw UTF-8 with `▁` escaping and `<0xXX>` byte fallback), using the pre-tokenizer
    family selected by `tokenizer.ggml.pre` exactly as llama.cpp's `llama-vocab.cpp` does;
  - SentencePiece (`"llama"`): llama.cpp's score-ordered bigram merge with byte fallback and
    space-prefix handling.
- Split literal special tokens out of the text (`parse_special`), add BOS/EOS (`add_special`).
- Decode ids back to text, render single tokens, and stream text incrementally without ever
  emitting a partial UTF-8 sequence.
- Expose end-of-generation knowledge (`is_eog`) with llama.cpp's text-based heuristics.

Unknown `tokenizer.ggml.pre` ids fall back to `"default"` with a `tracing` warning. Families
llama.cpp only handles through bespoke splitters that are not ported here (`kimi-k2`, `afmoe`,
`tiny_aya`/`cohere2moe`, `superbpe`) are refused with `TokenizerError::UnsupportedPre`.

## Public types

| Item | Purpose |
| --- | --- |
| `Tokenizer::from_gguf(&GgufFile) -> Result<Tokenizer, TokenizerError>` | Build from metadata. |
| `Tokenizer::encode(text, add_special, parse_special) -> Vec<u32>` | Text to ids. |
| `Tokenizer::decode(&[u32]) -> String` / `decode_with_special` | Ids to text (control tokens hidden / rendered). |
| `Tokenizer::token_to_piece(id) -> Vec<u8>` | Rendered bytes of one token (byte tokens give one byte). |
| `token_type`, `is_control`, `token_text`, `token_score`, `token_to_id`, `byte_to_token` | Vocabulary lookups. |
| `n_vocab`, `bos`, `eos`, `eot`, `eom`, `pad`, `unk`, `sep`, `add_bos`, `add_eos`, `add_space_prefix`, `remove_extra_whitespaces`, `kind`, `pre_name`, `pre_type`, `chat_template` | Metadata accessors. |
| `is_eog(id)`, `eog_ids()` | End-of-generation set: eos/eot/eom plus `<\|im_end\|>`, `<\|eot_id\|>`, `<end_of_turn>`, `<\|endoftext\|>`, ... |
| `Detokenizer::new(&Tokenizer)` / `with_special`, `push(id) -> String`, `flush()`, `pending()` | Streaming decode holding back incomplete multi-byte sequences. |
| `TokenType`, `VocabKind`, `PreType`, `PreConfig`, `TokenizerError` | Supporting types. |

Modules: `pretok` (pre-tokenizer table and splitters), `bpe`, `spm`, `unicode` (llama.cpp's
codepoint categories and the GPT-2 byte map), `unicode_data` (generated tables), `detok`.

## How parity with llama.cpp is achieved

llama.cpp hand-implements five split patterns as state machines (`gpt2`, `llama3`, `qwen2`,
`qwen35`, newline runs); those are ported literally in `pretok.rs`. Every other pattern
(`gpt-4o`, `tekken`, `default`, `smollm`'s `\p{N}` pass, ...) runs in llama.cpp through
`std::regex` over a *collapsed* text in which each non-ASCII codepoint is replaced by one marker
for its category. The crate reproduces that haystack and rewrites the pattern for `fancy-regex`
(MIT) so lookaheads behave identically. Codepoint categories come from llama.cpp's own
`unicode-data.cpp` tables, regenerated with `tools/gen_unicode_data.py` (pass the path to that
file and the output path `src/unicode_data.rs`).

Deliberate deviations are listed in the crate docs (`src/lib.rs`): no `clean_spaces` post-pass
in `decode`, deterministic EOT choice when the id is not in the file, `unk` instead of a throw
when a SentencePiece byte has no token.

## Tests

```sh
cargo test -p llmario-engine-tokenizer            # unit tests on synthetic vocabularies
cargo clippy -p llmario-engine-tokenizer --all-targets --no-deps -- -D warnings
```

### llama.cpp parity (needs `llama-tokenize`, default `/opt/homebrew/bin/llama-tokenize`,
override with `LLMARIO_LLAMA_TOKENIZE`)

`tests/llama_cpp_parity.rs` runs when `LLMARIO_TEST_GGUF` lists GGUF paths (comma-separated).
For each model and each of 59 corpus strings (English, code with tabs, long digit runs,
punctuation, emoji and other 4-byte UTF-8, CJK, Arabic/Hebrew, mixed scripts, leading/trailing/
multiple spaces and newlines, the empty string, literal special tokens, very long words) it
compares `encode(text, true, true)` with

```text
llama-tokenize -m <gguf> -f <tmpfile> --no-escape --ids --log-disable
```

(`-f --no-escape` feeds exact bytes; `llama-tokenize` uses `add_special = add_bos_token`,
`parse_special = true`, which equals `encode(.., true, true)` for models without
`add_eos_token`). It also checks `decode(encode(text)) == text`, that the streaming
`Detokenizer` yields the same text, that BOS/EOS/PAD and the EOG set equal llama.cpp's
`print_info` lines, and prints the time to encode 100 KB.

```sh
LLMARIO_TEST_GGUF="/Volumes/Extreme Pro/AIProjects/LLMario-models/qwen3-1.7b-gguf-q4km/Qwen3-1.7B-Q4_K_M.gguf,\
/Volumes/Extreme Pro/AIProjects/LLMario-beta-models/qwen3.5-0.8b-gguf-q4_0/Qwen3.5-0.8B-Q4_0.gguf,\
/Volumes/Extreme Pro/AIProjects/LLMario-beta-models/gemma-4-12b-gguf-q4_0/gemma-4-12b-it-qat-q4_0.gguf,\
/Volumes/Extreme Pro/AIProjects/LLMario-beta-models/gpt-oss-20b-gguf-mxfp4/gpt-oss-20b-MXFP4.gguf,\
/Volumes/Extreme Pro/AIProjects/LoopForgeProject/platform/.models/smollm3-3b-q4_k_m.gguf,\
/Volumes/Extreme Pro/AIProjects/LoopForgeProject/platform/.models/olmo-3-7b-instruct-q4_k_m.gguf,\
/Volumes/Extreme Pro/AIProjects/LoopForgeProject/platform/.models/smollm2-1.7b-instruct-q4_k_m.gguf" \
  cargo test -p llmario-engine-tokenizer --test llama_cpp_parity -- --nocapture
```

`tests/spm_synthetic_parity.rs` covers SentencePiece the same way on a generated vocabulary
(no `"llama"`-model GGUF is in the local catalog; `llama-tokenize` loads a vocab-only file):

```sh
LLMARIO_TEST_GGUF=1 cargo test -p llmario-engine-tokenizer --test spm_synthetic_parity -- --nocapture
```

Last run (llama.cpp 0.5.0 build 11146, release build): 59/59 on each of the seven models above
(pre ids `qwen2`, `qwen35`, `gemma4`, `gpt-4o`, `smaug-bpe`, `dbrx`, `smollm`), 36/36 on the
synthetic SentencePiece vocabulary; 100 KB encodes in 7–14 ms, vocabularies load in 9–61 ms.
