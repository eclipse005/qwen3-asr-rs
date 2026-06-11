//! Burn CUDA f16 benchmark & correctness tests.
//!
//! Run: cargo test --release --test transcribe_burn -- --ignored --nocapture
//!
//! Models and fixtures resolve in this order:
//!   1. Env var (`QWEN3_ASR_MODEL_06_DIR`, `QWEN3_ASR_FIXTURES_DIR`, etc.) — if set, used as-is.
//!   2. Repo-relative defaults (`models/Qwen3-ASR-0.6B`, `tests/fixtures/...`) resolved against
//!      `CARGO_MANIFEST_DIR` so it works no matter where you invoke `cargo test` from.

use std::path::PathBuf;
use std::time::Instant;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir_06() -> String {
    std::env::var("QWEN3_ASR_MODEL_06_DIR")
        .unwrap_or_else(|_| repo_root().join("models/Qwen3-ASR-0.6B").to_string_lossy().into_owned())
}

fn model_dir_17() -> String {
    std::env::var("QWEN3_ASR_MODEL_17_DIR")
        .unwrap_or_else(|_| repo_root().join("models/Qwen3-ASR-1.7B").to_string_lossy().into_owned())
}

fn fixture(name: &str) -> String {
    let base = std::env::var("QWEN3_ASR_FIXTURES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("tests/fixtures"));
    base.join(name).to_string_lossy().into_owned()
}

#[test]
#[ignore]
fn test_q06_sample1() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();

    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_06()), device,
    ).expect("load 0.6B");

    let result = engine.transcribe(
        &fixture("sample1.wav"),
        qwen3_asr_burn::TranscribeOptions::default(),
    ).expect("transcribe");

    println!("Language : {}", result.language);
    println!("Text     : {}", result.text);
    println!("Raw      : {}", result.raw_output);

    let expected = "The quick brown fox jumps over the lazy dog.";
    let text_lower = result.text.to_lowercase();
    let expected_lower = expected.to_lowercase();
    assert!(
        text_lower.contains(&expected_lower[..expected_lower.len().min(20)]),
        "Expected text containing '{}', got '{}'", &expected[..20], result.text
    );
}

#[test]
#[ignore]
fn test_q06_15s() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();

    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_06()), device,
    ).expect("load 0.6B");

    let t0 = Instant::now();
    let result = engine.transcribe(
        &fixture("15s.wav"),
        qwen3_asr_burn::TranscribeOptions::default(),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    let rtfx = 15.0 / elapsed;

    println!("0.6B-15s | {:.3}s elapsed | RTFx {:.2}x", elapsed, rtfx);
    println!("Language : {}", result.language);
    println!("Text     : {}", result.text);
    assert!(!result.text.is_empty(), "Transcription should not be empty");
}

#[test]
#[ignore]
fn test_q06_30s() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();

    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_06()), device,
    ).expect("load 0.6B");

    let t0 = Instant::now();
    let result = engine.transcribe(
        &fixture("30s.wav"),
        qwen3_asr_burn::TranscribeOptions::default(),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    let rtfx = 30.0 / elapsed;

    println!("0.6B-30s | {:.3}s elapsed | RTFx {:.2}x", elapsed, rtfx);
    println!("Language : {}", result.language);
    println!("Text     : {}", result.text);
    assert!(!result.text.is_empty(), "Transcription should not be empty");
}

#[test]
#[ignore]
fn test_q17_15s() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();

    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_17()), device,
    ).expect("load 1.7B");

    let t0 = Instant::now();
    let result = engine.transcribe(
        &fixture("15s.wav"),
        qwen3_asr_burn::TranscribeOptions::default(),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    let rtfx = 15.0 / elapsed;

    println!("1.7B-15s | {:.3}s elapsed | RTFx {:.2}x", elapsed, rtfx);
    println!("Language : {}", result.language);
    println!("Text     : {}", result.text);
    assert!(!result.text.is_empty(), "Transcription should not be empty");
}

#[test]
#[ignore]
fn test_q17_30s() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();

    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_17()), device,
    ).expect("load 1.7B");

    let t0 = Instant::now();
    let result = engine.transcribe(
        &fixture("30s.wav"),
        qwen3_asr_burn::TranscribeOptions::default(),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    let rtfx = 30.0 / elapsed;

    println!("1.7B-30s | {:.3}s elapsed | RTFx {:.2}x", elapsed, rtfx);
    println!("Language : {}", result.language);
    println!("Text     : {}", result.text);
    assert!(!result.text.is_empty(), "Transcription should not be empty");
}

#[test]
#[ignore]
fn test_q06_90s() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();
    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_06()), device,
    ).expect("load 0.6B");
    let t0 = std::time::Instant::now();
    let result = engine.transcribe(
        &fixture("90s.wav"),
        qwen3_asr_burn::TranscribeOptions::default(),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    println!("0.6B-90s | {:.3}s elapsed | RTFx {:.2}x", elapsed, 90.0 / elapsed);
    println!("Text: {}", result.text);
    assert!(!result.text.is_empty());
}

#[test]
#[ignore]
fn test_q06_180s() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();
    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_06()), device,
    ).expect("load 0.6B");
    let t0 = std::time::Instant::now();
    let result = engine.transcribe(
        &fixture("180s.wav"),
        qwen3_asr_burn::TranscribeOptions::default().with_max_new_tokens(1024),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    println!("0.6B-180s | {:.3}s elapsed | RTFx {:.2}x", elapsed, 180.0 / elapsed);
    println!("Text: {}", result.text);
    assert!(!result.text.is_empty());
}

#[test]
#[ignore]
fn test_q06_180s_en() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();
    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_06()), device,
    ).expect("load 0.6B");
    let t0 = std::time::Instant::now();
    let result = engine.transcribe(
        &fixture("180s_en.wav"),
        qwen3_asr_burn::TranscribeOptions::default().with_max_new_tokens(1024),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    println!("0.6B-180s_en | {:.3}s elapsed | RTFx {:.2}x", elapsed, 180.0 / elapsed);
    println!("Text: {}", result.text);
    assert!(!result.text.is_empty());
}

#[test]
#[ignore]
fn test_q17_90s() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();
    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_17()), device,
    ).expect("load 1.7B");
    let t0 = std::time::Instant::now();
    let result = engine.transcribe(
        &fixture("90s.wav"),
        qwen3_asr_burn::TranscribeOptions::default().with_max_new_tokens(1024),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    println!("1.7B-90s | {:.3}s elapsed | RTFx {:.2}x", elapsed, 90.0 / elapsed);
    println!("Text: {}", result.text);
    assert!(!result.text.is_empty());
}

#[test]
#[ignore]
fn test_q17_180s() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();
    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_17()), device,
    ).expect("load 1.7B");
    let t0 = std::time::Instant::now();
    let result = engine.transcribe(
        &fixture("180s.wav"),
        qwen3_asr_burn::TranscribeOptions::default().with_max_new_tokens(1024),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    println!("1.7B-180s | {:.3}s elapsed | RTFx {:.2}x", elapsed, 180.0 / elapsed);
    println!("Text: {}", result.text);
    assert!(!result.text.is_empty());
}

#[test]
#[ignore]
fn test_q17_180s_en() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();
    let device = qwen3_asr_burn::best_device();
    let engine = qwen3_asr_burn::AsrInference::load(
        std::path::Path::new(&model_dir_17()), device,
    ).expect("load 1.7B");
    let t0 = std::time::Instant::now();
    let result = engine.transcribe(
        &fixture("180s_en.wav"),
        qwen3_asr_burn::TranscribeOptions::default().with_max_new_tokens(1024),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    println!("1.7B-180s_en | {:.3}s elapsed | RTFx {:.2}x", elapsed, 180.0 / elapsed);
    println!("Text: {}", result.text);
    assert!(!result.text.is_empty());
}
