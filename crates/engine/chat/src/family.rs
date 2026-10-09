//! Template family detection.
//!
//! A family names the tool-call wire format a template teaches the model (ARCHITECTURE.md
//! Appendix D). Detection is deliberately conservative: first an exact SHA-256 match against
//! templates we have verified by hand, then structural rules that each require at least two
//! independent markers. A single substring is never enough (a MiMo distill was once
//! mis-detected as Qwen3-Coder by a lone substring and its tool calls never completed).

/// Tool-call wire family a chat template renders (see ARCHITECTURE.md Appendix D).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TemplateFamily {
    /// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (Qwen3, SmolLM3, Granite 4).
    Hermes,
    /// `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
    /// (Qwen3.5, Qwen3-Coder).
    QwenXml,
    /// `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>` with `<|observation|>` (GLM-4.x).
    Glm,
    /// `<|tool_call>call:name{k:<|"|>v<|"|>}<tool_call|>` inside `<|turn>` turns (Gemma 4).
    Gemma4,
    /// `<|python_tag|>`/JSON calls between `<|start_header_id|>`/`<|eot_id|>` (Llama 3.x / 4).
    Llama3,
    /// `[TOOL_CALLS]`/`[AVAILABLE_TOOLS]` control tokens (Mistral Small / Ministral / Devstral).
    Mistral,
    /// `<|start|>`/`<|channel|>`/`<|message|>` channels (gpt-oss Harmony).
    Harmony,
    /// `<|tool_call_start|>[f(a='b')]<|tool_call_end|>` pythonic calls (LFM2).
    Lfm2,
    /// ChatML turn markers (`<|im_start|>`/`<|im_end|>`) without a recognised tool protocol.
    ChatMl,
    /// Nothing matched.
    Unknown,
}

impl TemplateFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            TemplateFamily::Hermes => "hermes",
            TemplateFamily::QwenXml => "qwen_xml",
            TemplateFamily::Glm => "glm",
            TemplateFamily::Gemma4 => "gemma4",
            TemplateFamily::Llama3 => "llama3",
            TemplateFamily::Mistral => "mistral",
            TemplateFamily::Harmony => "harmony",
            TemplateFamily::Lfm2 => "lfm2",
            TemplateFamily::ChatMl => "chatml",
            TemplateFamily::Unknown => "unknown",
        }
    }
}

/// Templates verified by hand (SHA-256 of the exact `tokenizer.chat_template` string).
/// Hashes come from the local GGUF files named in the crate README and are re-checked by the
/// `jinja2_parity` integration test, which asserts the family per `general.architecture`.
const KNOWN_HASHES: &[(&str, TemplateFamily, &str)] = &[
    (
        "8428c815ac94d82064e35ff1e841dcbe260e7e53a8d0bd3b94afa2eefa9bccab",
        TemplateFamily::Hermes,
        "Qwen3-1.7B (Qwen3-1.7B-Q4_K_M.gguf)",
    ),
    (
        "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        TemplateFamily::QwenXml,
        "Qwen3.5-0.8B (Qwen3.5-0.8B-Q4_0.gguf)",
    ),
    (
        "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        TemplateFamily::Gemma4,
        "Gemma 4 12B it (gemma-4-12b-it-qat-q4_0.gguf)",
    ),
    (
        "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146",
        TemplateFamily::Harmony,
        "gpt-oss-20b (gpt-oss-20b-MXFP4.gguf)",
    ),
    (
        "b9b66f04c64fbb8695cf5b35c37780efd0b8e0829fbfe3e30fafb9f469b7d30e",
        TemplateFamily::Hermes,
        "SmolLM3-3B (smollm3-3b-q4_k_m.gguf): Hermes JSON via xml_tools",
    ),
    (
        "f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b",
        TemplateFamily::ChatMl,
        "OLMo-3-7B-Instruct (olmo-3-7b-instruct-q4_k_m.gguf): ChatML turns, own <function_calls> syntax",
    ),
];

/// Detect the family of `source` whose SHA-256 hex digest is `hash_hex`.
pub fn detect(source: &str, hash_hex: &str) -> TemplateFamily {
    for (hash, family, _label) in KNOWN_HASHES {
        if hash.eq_ignore_ascii_case(hash_hex) {
            return *family;
        }
    }
    detect_structural(source)
}

/// Structural rules, most specific first. Each arm needs every listed marker.
///
/// | Family  | Markers (all required)                                                   |
/// |---------|--------------------------------------------------------------------------|
/// | Gemma4  | `<\|turn>` turn opener, `<\|tool_call>` opener, `<tool_call\|>` closer    |
/// | Harmony | `<\|start\|>`, `<\|channel\|>`, `<\|message\|>`                           |
/// | QwenXml | `<function=` and `<parameter=` tags (inside `<tool_call>`)               |
/// | Glm     | `<arg_key>` and `<arg_value>` tags (plus `<\|observation\|>` result role) |
/// | Lfm2    | `<\|tool_call_start\|>` and `<\|tool_call_end\|>` wrappers                |
/// | Mistral | `[TOOL_CALLS]` and `[AVAILABLE_TOOLS]` (or `[INST]`) control tokens       |
/// | Llama3  | `<\|start_header_id\|>` and `<\|eot_id\|>` plus `<\|python_tag\|>`/`ipython` |
/// | Hermes  | `<tool_call>`/`</tool_call>` call wrappers and `<tools>`/`</tools>` list wrappers, without the QwenXml tags |
/// | ChatMl  | `<\|im_start\|>` and `<\|im_end\|>` with no tool family above            |
fn detect_structural(src: &str) -> TemplateFamily {
    let has = |m: &str| src.contains(m);
    if has("<|turn>") && has("<|tool_call>") && has("<tool_call|>") {
        return TemplateFamily::Gemma4;
    }
    if has("<|start|>") && has("<|channel|>") && has("<|message|>") {
        return TemplateFamily::Harmony;
    }
    if has("<function=") && has("<parameter=") {
        return TemplateFamily::QwenXml;
    }
    if has("<arg_key>") && has("<arg_value>") {
        return TemplateFamily::Glm;
    }
    if has("<|tool_call_start|>") && has("<|tool_call_end|>") {
        return TemplateFamily::Lfm2;
    }
    if has("[TOOL_CALLS]") && (has("[AVAILABLE_TOOLS]") || has("[INST]")) {
        return TemplateFamily::Mistral;
    }
    if has("<|start_header_id|>") && has("<|eot_id|>") && (has("<|python_tag|>") || has("ipython"))
    {
        return TemplateFamily::Llama3;
    }
    if has("<tool_call>") && has("</tool_call>") && has("<tools>") && has("</tools>") {
        return TemplateFamily::Hermes;
    }
    if has("<|im_start|>") && has("<|im_end|>") {
        return TemplateFamily::ChatMl;
    }
    TemplateFamily::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_markers_are_not_enough() {
        assert_eq!(detect_structural("<tool_call>"), TemplateFamily::Unknown);
        assert_eq!(detect_structural("<|turn>user"), TemplateFamily::Unknown);
        assert_eq!(detect_structural("[TOOL_CALLS]"), TemplateFamily::Unknown);
        assert_eq!(detect_structural("<|start|>"), TemplateFamily::Unknown);
        assert_eq!(detect_structural("<|im_start|>"), TemplateFamily::Unknown);
    }

    #[test]
    fn marker_pairs_classify() {
        assert_eq!(
            detect_structural("{{ '<tool_call><function=' ~ n ~ '><parameter=' }}"),
            TemplateFamily::QwenXml
        );
        assert_eq!(
            detect_structural(
                "<tool_call>x<arg_key>k</arg_key><arg_value>v</arg_value><|observation|>"
            ),
            TemplateFamily::Glm
        );
        assert_eq!(
            detect_structural("<|turn>model\n<|tool_call>call:f{}<tool_call|>"),
            TemplateFamily::Gemma4
        );
        assert_eq!(
            detect_structural("<|start|>assistant<|channel|>final<|message|>"),
            TemplateFamily::Harmony
        );
        assert_eq!(
            detect_structural("<|tool_call_start|>[f()]<|tool_call_end|>"),
            TemplateFamily::Lfm2
        );
        assert_eq!(
            detect_structural("[AVAILABLE_TOOLS]{}[/AVAILABLE_TOOLS][INST]x[/INST][TOOL_CALLS]"),
            TemplateFamily::Mistral
        );
        assert_eq!(
            detect_structural("<|start_header_id|>ipython<|end_header_id|><|eot_id|>"),
            TemplateFamily::Llama3
        );
        assert_eq!(
            detect_structural("<tools>{{ tool | tojson }}</tools><tool_call>{}</tool_call>"),
            TemplateFamily::Hermes
        );
        assert_eq!(
            detect_structural(
                "<tools>\n{{ tool | string }}</tools><tool_call>{}</tool_call><|im_start|>"
            ),
            TemplateFamily::Hermes
        );
        assert_eq!(
            detect_structural("<|im_start|>user\n{{ c }}<|im_end|>"),
            TemplateFamily::ChatMl
        );
    }
}
