//! 独立 ASR 测试:复现管线调用方式 (transcribe_samples),验证 #292-313 的"あああ..."幻觉。
//!
//! 用法: cargo run --release --example test_hallucination -- <wav路径> [语言] [模型目录]

use qwen3_asr::{AsrInference, Backend, TranscribeOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "test_hallucination.wav".to_string());
    let lang = args.get(2).cloned().unwrap_or_else(|| "ja".to_string());
    let model_dir = std::path::PathBuf::from(
        args.get(3).cloned().unwrap_or_else(|| "models/Qwen3-ASR-1.7B".to_string())
    );
    if !model_dir.is_dir() {
        eprintln!("未找到 ASR 模型目录: {}", model_dir.display());
        std::process::exit(1);
    }
    eprintln!("模型: {}", model_dir.display());

    // 读取 wav 为 16kHz mono f32 samples —— 完全复现管线的行为
    let mut reader = hound::WavReader::open(&wav_path).unwrap_or_else(|e| {
        panic!("无法打开 {wav_path}: {e}");
    });
    let spec = reader.spec();
    eprintln!(
        "wav: {}Hz {}ch {:?}, duration: {}",
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample,
        reader.duration()
    );

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / 32768.0)
        .collect();
    eprintln!("音频文件: {wav_path}");
    eprintln!("采样数: {} ({:.1}s @ 16kHz)", samples.len(), samples.len() as f64 / 16000.0);
    eprintln!("语言: {lang}");

    let started = std::time::Instant::now();
    let asr = AsrInference::load(&model_dir, Backend::Cuda).unwrap_or_else(|e| {
        eprintln!("CUDA 加载失败,尝试 CPU: {e}");
        AsrInference::load(&model_dir, Backend::Cpu).unwrap_or_else(|e| {
            panic!("CPU 加载也失败: {e}");
        })
    });
    eprintln!("模型加载耗时: {:.2}s", started.elapsed().as_secs_f64());

    // 方式1: transcribe_samples (管线用的方式)
    let t1 = std::time::Instant::now();
    let options = TranscribeOptions::default().with_language(&lang);
    let result1 = asr.transcribe_samples(&samples, options).unwrap_or_else(|e| {
        panic!("transcribe_samples 失败: {e}");
    });
    let elapsed1 = t1.elapsed().as_secs_f64();

    println!("========== transcribe_samples (管线方式) ==========");
    println!("{}", result1.text);
    println!("字符数: {}", result1.text.chars().count());
    eprintln!("耗时: {:.2}s", elapsed1);

    // 方式2: transcribe (文件方式)
    let t2 = std::time::Instant::now();
    let options2 = TranscribeOptions::default().with_language(&lang);
    let result2 = asr.transcribe(&wav_path, options2).unwrap_or_else(|e| {
        panic!("transcribe(文件) 失败: {e}");
    });
    let elapsed2 = t2.elapsed().as_secs_f64();

    println!("========== transcribe (文件方式) ==========");
    println!("{}", result2.text);
    println!("字符数: {}", result2.text.chars().count());
    eprintln!("耗时: {:.2}s", elapsed2);
}
