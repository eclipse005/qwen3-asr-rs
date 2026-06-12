//! CPU f32 transcribe test — 90s audio, memory + timing.
//!
//! Run: cargo test --release --no-default-features --features cpu --test cpu_90s -- --nocapture

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

#[test]
fn test_cpu_90s() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();

    let backend = qwen3_asr::Backend::Cpu;
    let engine = qwen3_asr::AsrInference::load(
        std::path::Path::new(&model_dir_06()), backend,
    ).expect("load 0.6B CPU");

    let t0 = Instant::now();
    let result = engine.transcribe(
        &fixture("90s.wav"),
        qwen3_asr::TranscribeOptions::default(),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();
    let rtfx = 90.0 / elapsed;

    println!("CPU 0.6B-90s | {:.3}s elapsed | RTFx {:.2}x", elapsed, rtfx);
    println!("Language : {}", result.language);
    println!("Text     : {}", result.text);
}
