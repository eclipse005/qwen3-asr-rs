//! CPU f32 transcribe benchmarks — all fixtures, RTFx timing.
//!
//! Run: cargo test --release --no-default-features --features cpu --test cpu_transcribe -- --nocapture --test-threads=1

use std::path::PathBuf;
use std::time::Instant;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir_06() -> String {
    std::env::var("QWEN3_ASR_MODEL_06_DIR")
        .unwrap_or_else(|_| repo_root().join("models/Qwen3-ASR-0.6B").to_string_lossy().into_owned())
}

fn fixture(name: &str) -> String {
    let base = std::env::var("QWEN3_ASR_FIXTURES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("tests/fixtures"));
    base.join(name).to_string_lossy().into_owned()
}

fn run_cpu(name: &str, wav: &str, duration_s: f32) {
    let backend = qwen3_asr::Backend::Cpu;
    let engine = qwen3_asr::AsrInference::load(
        std::path::Path::new(&model_dir_06()), backend,
    ).expect("load 0.6B CPU");

    let t0 = Instant::now();
    let result = engine.transcribe(
        &fixture(wav),
        qwen3_asr::TranscribeOptions::default(),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    let rtfx = if duration_s > 0.0 { format!("{:.2}x", duration_s / elapsed) } else { "—".to_string() };

    println!("CPU {} | {:.3}s elapsed | RTFx {} | {} | {}", name, elapsed, rtfx, result.language, result.text);
    assert!(!result.text.is_empty(), "Transcription should not be empty");
}

#[test] fn test_cpu_sample1()   { run_cpu("sample1",   "sample1.wav",   0.0); }
#[test] fn test_cpu_15s()      { run_cpu("15s",       "15s.wav",      15.0); }
#[test] fn test_cpu_30s()      { run_cpu("30s",       "30s.wav",      30.0); }
#[test] fn test_cpu_90s()      { run_cpu("90s",       "90s.wav",      90.0); }
#[test] fn test_cpu_89s_ja()   { run_cpu("89s_ja",    "ja_89s.wav",   89.0); }
#[test] fn test_cpu_180s()     { run_cpu("180s",      "180s.wav",    180.0); }
#[test] fn test_cpu_180s_en()  { run_cpu("180s_en",   "180s_en.wav", 180.0); }
