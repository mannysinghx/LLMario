# Test fixture templates

Used by `tests/tool_call_round_trip.rs` to render assistant turns that the output parser must
read back. Only the assistant-turn syntax matters; the round-trip test additionally asserts,
when `LLMARIO_TEST_GGUF` names the corresponding model file, that the vendored or minimal
fixture renders the assistant turn byte-identically to the real `tokenizer.chat_template`.

| File | Origin | Licence | Form |
|---|---|---|---|
| `qwen3.jinja` | `tokenizer.chat_template` of Qwen3-1.7B-Q4_K_M.gguf | Apache-2.0 | verbatim |
| `qwen35.jinja` | `tokenizer.chat_template` of Qwen3.5-0.8B-Q4_0.gguf | Apache-2.0 | verbatim |
| `gptoss.jinja` | `tokenizer.chat_template` of gpt-oss-20b-MXFP4.gguf | Apache-2.0 | verbatim |
| `olmo3.jinja` | `tokenizer.chat_template` of olmo-3-7b-instruct-q4_k_m.gguf | Apache-2.0 | verbatim |
| `glm47.jinja` | huggingface.co/zai-org/GLM-4.7-Flash/blob/main/chat_template.jinja | MIT | verbatim |
| `gemma4.jinja` | written here after gemma-4-12b-it-qat-q4_0.gguf (Gemma Terms of Use: not vendored) | this repo | minimal |
| `mistral.jinja` | assistant-turn lines of huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/blob/main/chat_template.jinja | Apache-2.0 | minimal |
| `lfm2.jinja` | written here after huggingface.co/LiquidAI/LFM2-1.2B/blob/main/chat_template.jinja (LFM Open License: not vendored) | this repo | minimal |
| `llama31.jinja` | written here after github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md (HF repo gated) | this repo | minimal |
| `llama4.jinja` | written here after github.com/meta-llama/llama-models/blob/main/models/llama4/prompt_format.md (HF repo gated) | this repo | minimal |

The `{#- … -#}` header comment on each file is the only change to the verbatim ones.
