//! CUDA end-to-end benchmark: timing + peak GPU memory for both Qwen3-ASR models.
//!
//! Run all benches:
//!   cargo test --release --test bench_cuda -- --ignored --nocapture --test-threads=1
//!
//! Run a single bench:
//!   cargo test --release --test bench_cuda test_q06_15s -- --ignored --nocapture
//!
//! Output is printed as tab-separated lines so it can be copied into a spreadsheet.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

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

/// Returns (free, total) GPU memory in bytes via nvidia-smi.
/// Uses a subprocess so it works from any thread without a CUDA context.
fn gpu_mem_info() -> Option<(usize, usize)> {
    use std::process::Command;
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let line = text.lines().next()?;
    let mut parts = line.split(',').map(|s| s.trim());
    let free: usize = parts.next()?.parse().ok()?;
    let total: usize = parts.next()?.parse().ok()?;
    // nvidia-smi reports MiB; convert to bytes.
    Some((free * 1024 * 1024, total * 1024 * 1024))
}

/// Spawn a background thread that samples GPU memory every `interval` and returns
/// the peak used bytes observed. Stop by dropping the returned closure or calling it.
fn spawn_gpu_mem_sampler(interval: Duration) -> (thread::JoinHandle<usize>, Arc<AtomicBool>) {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();
    let handle = thread::spawn(move || {
        let mut peak_used = 0usize;
        while !stop_clone.load(Ordering::Relaxed) {
            if let Some((free, total)) = gpu_mem_info() {
                let used = total.saturating_sub(free);
                if used > peak_used {
                    peak_used = used;
                }
            }
            thread::sleep(interval);
        }
        // One last sample after the workload finishes so we don't miss a peak right at the end.
        if let Some((free, total)) = gpu_mem_info() {
            let used = total.saturating_sub(free);
            if used > peak_used {
                peak_used = used;
            }
        }
        peak_used
    });
    (handle, stop)
}

fn stop_sampler(stop: Arc<AtomicBool>, handle: thread::JoinHandle<usize>) -> usize {
    stop.store(true, Ordering::Relaxed);
    handle.join().unwrap_or(0)
}

fn bytes_to_mib(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn run_bench(model_name: &str, model_dir: &str, wav: &str, duration_s: f32, max_new_tokens: usize) {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();

    let baseline = gpu_mem_info().map(|(free, total)| (bytes_to_mib(total - free), bytes_to_mib(total), bytes_to_mib(free)));

    let backend = qwen3_asr::Backend::best().expect("best backend");
    let engine = qwen3_asr::AsrInference::load(
        std::path::Path::new(model_dir), backend,
    ).expect(&format!("load {}", model_name));

    let after_load = gpu_mem_info().map(|(free, total)| bytes_to_mib(total - free));

    // Start sampling GPU memory right before transcribe.
    let (sampler_handle, stop_flag) = spawn_gpu_mem_sampler(Duration::from_millis(50));

    let t0 = Instant::now();
    let result = engine.transcribe(
        &fixture(wav),
        qwen3_asr::TranscribeOptions::default().with_max_new_tokens(max_new_tokens),
    ).expect("transcribe");
    let elapsed = t0.elapsed().as_secs_f32();

    let peak_used = stop_sampler(stop_flag, sampler_handle);

    let after_transcribe = gpu_mem_info().map(|(free, total)| bytes_to_mib(total - free));
    let rtfx = if duration_s > 0.0 { duration_s / elapsed } else { 0.0 };

    println!(
        "RUST_CUDA\t{}\t{}\t{:.3}s\t{:.2}x\tpeak_used={:.1}MiB\tbaseline={}\tafter_load={}\tafter_transcribe={}\tlang={}\ttext={}",
        model_name,
        wav,
        elapsed,
        rtfx,
        bytes_to_mib(peak_used),
        match baseline { Some((used, total, free)) => format!("{:.1}MiB/{:.1}MiB free={:.1}MiB", used, total, free), None => "n/a".to_string() },
        match after_load { Some(used) => format!("{:.1}MiB", used), None => "n/a".to_string() },
        match after_transcribe { Some(used) => format!("{:.1}MiB", used), None => "n/a".to_string() },
        result.language,
        result.text.replace('\t', " ").replace('\n', " "),
    );

    assert!(!result.text.is_empty(), "Transcription should not be empty");
}

// 0.6B model benches
#[test]
#[ignore]
fn test_q06_15s()      { run_bench("0.6B", &model_dir_06(), "15s.wav",      15.0,  512); }
#[test]
#[ignore]
fn test_q06_30s()      { run_bench("0.6B", &model_dir_06(), "30s.wav",      30.0,  512); }
#[test]
#[ignore]
fn test_q06_90s()      { run_bench("0.6B", &model_dir_06(), "90s.wav",      90.0,  1024); }
#[test]
#[ignore]
fn test_q06_89s_ja()   { run_bench("0.6B", &model_dir_06(), "ja_89s.wav",   89.0,  1024); }
#[test]
#[ignore]
fn test_q06_180s()     { run_bench("0.6B", &model_dir_06(), "180s.wav",    180.0,  1024); }
#[test]
#[ignore]
fn test_q06_180s_en()  { run_bench("0.6B", &model_dir_06(), "180s_en.wav", 180.0,  1024); }

// 1.7B model benches
#[test]
#[ignore]
fn test_q17_15s()      { run_bench("1.7B", &model_dir_17(), "15s.wav",      15.0,  512); }
#[test]
#[ignore]
fn test_q17_30s()      { run_bench("1.7B", &model_dir_17(), "30s.wav",      30.0,  512); }
#[test]
#[ignore]
fn test_q17_90s()      { run_bench("1.7B", &model_dir_17(), "90s.wav",      90.0,  1024); }
#[test]
#[ignore]
fn test_q17_89s_ja()   { run_bench("1.7B", &model_dir_17(), "ja_89s.wav",   89.0,  1024); }
#[test]
#[ignore]
fn test_q17_180s()     { run_bench("1.7B", &model_dir_17(), "180s.wav",    180.0,  1024); }
#[test]
#[ignore]
fn test_q17_180s_en()  { run_bench("1.7B", &model_dir_17(), "180s_en.wav", 180.0,  1024); }
