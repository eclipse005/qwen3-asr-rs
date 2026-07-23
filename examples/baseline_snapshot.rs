//! Freeze pre-refactor baselines: transcription text + RTFx for all fixtures.
//!
//! Models: 0.6B + 1.7B. Backends: CPU + CUDA.
//! Fixtures (zh / en / ja): see `tests/fixtures/*_{zh,en,ja}.wav`.
//!
//! Run:
//!   cargo run --release --example baseline_snapshot
//!
//! Writes:
//!   docs/baseline/pre-refactor.md
//!   docs/baseline/pre-refactor.tsv
//!   docs/baseline/texts/{backend}_{model}_{wav}.txt

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use qwen3_asr::{AsrInference, Backend, TranscribeOptions};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir(size: &str) -> PathBuf {
    let key = if size == "0.6B" {
        "QWEN3_ASR_MODEL_06_DIR"
    } else {
        "QWEN3_ASR_MODEL_17_DIR"
    };
    let default = repo_root().join(format!("models/Qwen3-ASR-{size}"));
    std::env::var(key)
        .map(PathBuf::from)
        .unwrap_or(default)
}

fn fixtures_dir() -> PathBuf {
    std::env::var("QWEN3_ASR_FIXTURES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("tests/fixtures"))
}

fn wav_duration_s(path: &Path) -> f64 {
    let r = hound::WavReader::open(path).expect("open wav");
    let sr = r.spec().sample_rate as f64;
    r.duration() as f64 / sr
}

/// (filename, language_tag, max_new_tokens)
fn fixtures() -> Vec<(&'static str, &'static str, usize)> {
    vec![
        // English
        ("15s_en.wav", "en", 512),
        ("90s_en.wav", "en", 1024),
        ("180s_en.wav", "en", 1024),
        // Chinese
        ("30s_zh.wav", "zh", 512),
        ("180s_zh.wav", "zh", 1024),
        // Japanese
        ("90s_ja.wav", "ja", 1024),
    ]
}

struct Row {
    backend: String,
    model: String,
    wav: String,
    lang_tag: String,
    audio_s: f64,
    elapsed_s: f64,
    rtfx: f64,
    detected_lang: String,
    text: String,
}

fn stem(wav: &str) -> &str {
    wav.trim_end_matches(".wav")
}

fn run_one(
    backend_name: &str,
    backend: Backend,
    model: &str,
    model_path: &Path,
    wav_name: &str,
    lang_tag: &str,
    max_new: usize,
    out_texts: &Path,
) -> Row {
    let wav_path = fixtures_dir().join(wav_name);
    let audio_s = wav_duration_s(&wav_path);

    eprintln!(
        ">>> {backend_name} | {model} | {wav_name} ({audio_s:.1}s audio, max_new={max_new})"
    );

    let engine = AsrInference::load(model_path, backend).unwrap_or_else(|e| {
        panic!("load {model} {backend_name}: {e:#}");
    });

    let t0 = Instant::now();
    let result = engine
        .transcribe(
            wav_path.to_str().unwrap(),
            TranscribeOptions::default().with_max_new_tokens(max_new),
        )
        .unwrap_or_else(|e| panic!("transcribe {wav_name}: {e:#}"));
    let elapsed_s = t0.elapsed().as_secs_f64();
    let rtfx = if elapsed_s > 0.0 {
        audio_s / elapsed_s
    } else {
        0.0
    };

    let text_path = out_texts.join(format!(
        "{backend_name}_{model}_{}.txt",
        stem(wav_name)
    ));
    let body = format!(
        "backend: {backend_name}\nmodel: {model}\nwav: {wav_name}\naudio_s: {audio_s:.3}\nelapsed_s: {elapsed_s:.3}\nrtfx: {rtfx:.3}\ndetected_language: {}\n\n{}\n",
        result.language, result.text
    );
    fs::write(&text_path, &body).expect("write text");

    eprintln!(
        "    elapsed={elapsed_s:.2}s  RTFx={rtfx:.2}x  lang={}  text_len={}",
        result.language,
        result.text.chars().count()
    );

    Row {
        backend: backend_name.to_string(),
        model: model.to_string(),
        wav: wav_name.to_string(),
        lang_tag: lang_tag.to_string(),
        audio_s,
        elapsed_s,
        rtfx,
        detected_lang: result.language,
        text: result.text,
    }
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .try_init();

    let out_dir = repo_root().join("docs/baseline");
    let texts_dir = out_dir.join("texts");
    fs::create_dir_all(&texts_dir).expect("mkdir baseline");

    let mut rows: Vec<Row> = Vec::new();

    let plans: Vec<(&str, Backend, &str)> = vec![
        ("cuda", Backend::Cuda, "0.6B"),
        ("cuda", Backend::Cuda, "1.7B"),
        ("cpu", Backend::Cpu, "0.6B"),
        ("cpu", Backend::Cpu, "1.7B"),
    ];

    for (backend_name, backend, model) in plans {
        let mdir = model_dir(model);
        if !mdir.join("config.json").is_file() {
            eprintln!("SKIP {backend_name}/{model}: missing {}", mdir.display());
            continue;
        }
        for (wav, lang_tag, max_new) in fixtures() {
            let wav_path = fixtures_dir().join(wav);
            if !wav_path.is_file() {
                eprintln!("SKIP missing fixture {}", wav_path.display());
                continue;
            }
            let row = run_one(
                backend_name,
                backend,
                model,
                &mdir,
                wav,
                lang_tag,
                max_new,
                &texts_dir,
            );
            rows.push(row);
        }
    }

    // TSV
    let mut tsv = String::from(
        "backend\tmodel\twav\tlang_tag\taudio_s\telapsed_s\trtfx\tdetected_lang\ttext\n",
    );
    for r in &rows {
        let text = r
            .text
            .replace('\t', " ")
            .replace('\n', " ")
            .replace('\r', "");
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\n",
            r.backend, r.model, r.wav, r.lang_tag, r.audio_s, r.elapsed_s, r.rtfx, r.detected_lang, text
        ));
    }
    fs::write(out_dir.join("pre-refactor.tsv"), &tsv).expect("write tsv");

    // Markdown summary
    let device = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.lines().next().unwrap_or("unknown").trim().to_string())
        .unwrap_or_else(|| "n/a".into());

    let mut md = String::new();
    md.push_str("# Pre-refactor baseline (fixed snapshot)\n\n");
    md.push_str(&format!("- Date: {} (file mtime is authoritative)\n", chrono_like_now()));
    md.push_str(&format!("- GPU: `{device}`\n"));
    md.push_str("- Host: Windows, release build (`cargo run --release --example baseline_snapshot`)\n");
    md.push_str("- Timing: wall time of `transcribe` only (model already loaded)\n");
    md.push_str("- RTFx = audio_duration / elapsed  (>1 faster than realtime)\n");
    md.push_str("- Fixtures: `tests/fixtures/*_{zh,en,ja}.wav`\n");
    md.push_str("- Per-run text dumps: `docs/baseline/texts/`\n");
    md.push_str("- Machine-readable: `docs/baseline/pre-refactor.tsv`\n\n");

    md.push_str("## Summary table\n\n");
    md.push_str("| Backend | Model | Wav | Lang | Audio (s) | Elapsed (s) | RTFx | Detected |\n");
    md.push_str("|---------|-------|-----|------|-----------|-------------|------|----------|\n");
    for r in &rows {
        md.push_str(&format!(
            "| {} | {} | `{}` | {} | {:.1} | {:.2} | **{:.2}x** | {} |\n",
            r.backend, r.model, r.wav, r.lang_tag, r.audio_s, r.elapsed_s, r.rtfx, r.detected_lang
        ));
    }

    md.push_str("\n## Transcription texts\n\n");
    for r in &rows {
        md.push_str(&format!(
            "### {} / {} / `{}`\n\n",
            r.backend, r.model, r.wav
        ));
        md.push_str(&format!(
            "- audio: {:.2}s · elapsed: {:.2}s · RTFx: **{:.2}x** · lang: {}\n\n",
            r.audio_s, r.elapsed_s, r.rtfx, r.detected_lang
        ));
        md.push_str("```\n");
        md.push_str(&r.text);
        md.push_str("\n```\n\n");
    }

    fs::write(out_dir.join("pre-refactor.md"), &md).expect("write md");
    eprintln!("\nWrote {} rows → {}", rows.len(), out_dir.join("pre-refactor.md").display());
}

/// Avoid chrono dependency: local-ish ISO-ish stamp from system time.
fn chrono_like_now() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix-{secs} (run host local; see file mtime)")
}
