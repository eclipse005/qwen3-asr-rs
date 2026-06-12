//! Prompt construction and result parsing for Qwen3-ASR.

use crate::inference::TranscribeResult;

// ─── Token constants ──────────────────────────────────────────────

pub(crate) const IM_END_TOKEN_ID: i64 = 151645;
pub(crate) const ENDOFTEXT_TOKEN_ID: i64 = 151643;
pub(crate) const ASR_TEXT_SEP_TOKEN_ID: u32 = 151704;

pub(crate) const TOK_IM_START: i64 = 151644;
pub(crate) const TOK_SYSTEM: i64 = 8948;
pub(crate) const TOK_NEWLINE: i64 = 198;
pub(crate) const TOK_IM_END: i64 = IM_END_TOKEN_ID;
pub(crate) const TOK_USER: i64 = 872;
pub(crate) const TOK_ASSISTANT: i64 = 77091;

// ─── Prompt building ──────────────────────────────────────────────

pub(crate) fn build_prompt(
    tokenizer: &tokenizers::Tokenizer,
    audio_start_token_id: i64,
    audio_token_id: i64,
    audio_end_token_id: i64,
    nat: usize,
    language: Option<&str>,
    prefix_text: Option<&str>,
) -> anyhow::Result<(Vec<i64>, usize)> {
    let mut tokens: Vec<i64> = vec![
        TOK_IM_START, TOK_SYSTEM, TOK_NEWLINE, TOK_IM_END, TOK_NEWLINE,
        TOK_IM_START, TOK_USER, TOK_NEWLINE, audio_start_token_id,
    ];
    let asp = tokens.len();
    tokens.extend(std::iter::repeat_n(audio_token_id, nat));
    tokens.extend_from_slice(&[audio_end_token_id, TOK_IM_END, TOK_NEWLINE, TOK_IM_START]);
    if let Some(lang) = language {
        tokens.push(TOK_ASSISTANT); tokens.push(TOK_NEWLINE);
        let lang_str = format!("language {}", capitalize_first(lang));
        let enc = tokenizer.encode(lang_str.as_str(), false).map_err(|e| anyhow::anyhow!("encode: {}", e))?;
        tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
    } else {
        tokens.push(TOK_ASSISTANT); tokens.push(TOK_NEWLINE);
    }
    if let Some(prefix) = prefix_text {
        if !prefix.is_empty() {
            let enc = tokenizer.encode(prefix, false).map_err(|e| anyhow::anyhow!("encode prefix: {}", e))?;
            tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
        }
    }
    Ok((tokens, asp))
}

// ─── Result parsing ───────────────────────────────────────────────

pub(crate) fn decode_result(
    tokenizer: &tokenizers::Tokenizer,
    generated_ids: &[u32],
    language: Option<&str>,
) -> anyhow::Result<TranscribeResult> {
    let raw_text = tokenizer.decode(generated_ids, true).map_err(|e| anyhow::anyhow!("decode: {}", e))?;
    let (lang, text) = if language.is_some() {
        ("forced".to_string(), raw_text.trim().to_string())
    } else if let Some(sep_pos) = generated_ids.iter().position(|&id| id == ASR_TEXT_SEP_TOKEN_ID) {
        let lang_ids: Vec<u32> = generated_ids[..sep_pos].to_vec();
        let text_ids: Vec<u32> = generated_ids[sep_pos + 1..].to_vec();
        let lang_raw = tokenizer.decode(&lang_ids, true).map_err(|e| anyhow::anyhow!("decode lang: {}", e))?;
        let text_raw = tokenizer.decode(&text_ids, true).map_err(|e| anyhow::anyhow!("decode text: {}", e))?;
        (lang_raw.strip_prefix("language ").unwrap_or(&lang_raw).trim().to_string(), text_raw.trim().to_string())
    } else {
        parse_asr_output(&raw_text, false)
    };
    Ok(TranscribeResult { text, language: lang, raw_output: raw_text })
}

fn parse_asr_output(raw: &str, forced: bool) -> (String, String) {
    if forced { return ("forced".to_string(), raw.trim().to_string()); }
    let raw = raw.trim();
    if let Some(rest) = raw.strip_prefix("language ") {
        if let Some(pos) = rest.find("<asr_text>") {
            return (rest[..pos].trim().to_string(), rest[pos + "<asr_text>".len()..].trim().to_string());
        }
        let mut le = rest.len();
        for (i, c) in rest.char_indices() { if c.is_whitespace() || !c.is_alphabetic() { le = i; break; } }
        if le > 0 && le < rest.len() { return (rest[..le].to_string(), rest[le..].trim().to_string()); }
    }
    ("unknown".to_string(), raw.to_string())
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}
