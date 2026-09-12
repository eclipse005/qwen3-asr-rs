//! Full-fidelity baseline check: transcribe every fixture and compare the text
//! byte-for-byte against the frozen reference in `docs/baseline/texts/`.
//!
//! `smoke_baseline` covers a fast subset; this covers everything, which is what you
//! want to run after any change that touches numerics (kernel rewrites, fused ops,
//! different accumulation order, …).  A transcription that merely *looks* right is
//! not good enough — a single flipped token in a borderline argmax is a regression.
//!
//! Usage:
//!   cargo run --release --example verify_baseline -- [backend] [model] [filter]
//!   cargo run --release --example verify_baseline -- cuda 1.7B 15s_en
//!
//!   backend : cuda (default) | cpu
//!   model   : 0.6B (default) | 1.7B | all
//!   filter  : substring of the wav name
//!
//! Freezing new references (use the *known-good* build only):
//!   cargo run --release --example verify_baseline -- cuda all --freeze
//!
//! Writes `target/verify_report_<backend>_<model>.tsv` and dumps every transcript to
//! `target/verify_got_<model>_<stem>.txt` so run-to-run determinism can be checked by
//! hashing.  Exits non-zero if any fixture diverges (outside freeze mode).

use std::path::Path;
use std::time::Instant;

use qwen3_asr::{AsrInference, Backend, TranscribeOptions};

/// (wav, stem, max_new_tokens) — must mirror `examples/baseline_snapshot.rs`.
const CASES: &[(&str, &str, usize)] = &[
    ("15s_en.wav", "15s_en", 512),
    ("90s_en.wav", "90s_en", 1024),
    ("180s_en.wav", "180s_en", 1024),
    ("30s_zh.wav", "30s_zh", 512),
    ("180s_zh.wav", "180s_zh", 1024),
    ("90s_ja.wav", "90s_ja", 1024),
];

struct Args {
    backend: String,
    models: Vec<String>,
    filter: Option<String>,
    freeze: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        backend: "cuda".into(),
        models: Vec::new(),
        filter: None,
        freeze: false,
    };
    let mut positional = 0usize;
    for arg in std::env::args().skip(1) {
        if arg == "--freeze" {
            a.freeze = true;
            continue;
        }
        match positional {
            0 => a.backend = arg,
            1 => a.models = if arg == "all" {
                vec!["0.6B".into(), "1.7B".into()]
            } else {
                vec![arg]
            },
            2 => a.filter = Some(arg),
            _ => {}
        }
        positional += 1;
    }
    if a.models.is_empty() {
        a.models = vec!["0.6B".into()];
    }
    a
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("tests/fixtures");
    let texts = root.join("docs/baseline/texts");

    let args = parse_args();
    let (backend, tag) = match args.backend.as_str() {
        "cpu" => (Backend::Cpu, "cpu"),
        _ => (Backend::Cuda, "cuda"),
    };

    let mut total_fail = 0usize;

    for model in &args.models {
        let mut report = String::from(
            "backend\tmodel\twav\taudio_s\telapsed_s\trtfx\tmatch\ttext_len\n",
        );
        let model_dir = root.join(format!("models/Qwen3-ASR-{model}"));
        if !model_dir.join("config.json").is_file() {
            println!("-- SKIP model {model}: {} not found", model_dir.display());
            continue;
        }

        let engine = match AsrInference::load(&model_dir, backend) {
            Ok(e) => e,
            Err(e) => {
                println!("-- FAIL load {model}: {e:#}");
                total_fail += 1;
                continue;
            }
        };

        println!(
            "\n=== verify_baseline | backend={tag} | model={model} | {} ===\n",
            if args.freeze { "FREEZE" } else { "compare" }
        );
        println!(
            "{:<14} {:>9} {:>9} {:>8} {:>9}  {}",
            "wav", "audio", "elapsed", "RTFx", "match", "baseline"
        );

        let mut fail = 0usize;
        for &(wav, stem, max_new) in CASES {
            if let Some(f) = &args.filter {
                if !wav.contains(f.as_str()) {
                    continue;
                }
            }
            let base_file = texts.join(format!("{tag}_{model}_{stem}.txt"));
            let want = match std::fs::read_to_string(&base_file) {
                Ok(s) => s.split("\n\n").nth(1).unwrap_or("").trim().to_string(),
                Err(e) => {
                    println!("{wav:<14} -- no baseline ({e})");
                    continue;
                }
            };

            // Deterministic-but-meaningful timing: model is already loaded, so this is
            // the cost of prefill + decode only.
            let t0 = Instant::now();
            let r = match engine.transcribe(
                fixtures.join(wav).to_str().unwrap(),
                TranscribeOptions::default().with_max_new_tokens(max_new),
            ) {
                Ok(r) => r,
                Err(e) => {
                    println!("{wav:<14} -- FAIL transcribe: {e:#}");
                    fail += 1;
                    continue;
                }
            };
            let elapsed = t0.elapsed().as_secs_f64();
            let audio_s: f64 = wav
                .trim_end_matches(".wav")
                .split('s')
                .next()
                .unwrap()
                .parse()
                .unwrap_or(0.0);
            let got = r.text.trim();

            let got_file = root
                .join("target")
                .join(format!("verify_got_{model}_{stem}.txt"));
            let _ = std::fs::write(&got_file, got);

            let ok = got == want;
            let shown = if args.freeze {
                "FROZEN".to_string()
            } else if ok {
                "OK".to_string()
            } else {
                fail += 1;
                "MISMATCH".to_string()
            };

            println!(
                "{wav:<14} {:>8.1}s {:>8.2}s {:>7.1}x {:>9}  {}",
                audio_s,
                elapsed,
                if elapsed > 0.0 { audio_s / elapsed } else { 0.0 },
                shown,
                base_file.file_name().unwrap().to_string_lossy()
            );
            report.push_str(&format!(
                "{tag}\t{model}\t{wav}\t{audio_s:.3}\t{elapsed:.3}\t{:.3}\t{}\t{}\n",
                if elapsed > 0.0 { audio_s / elapsed } else { 0.0 },
                if args.freeze { "FREEZE" } else if ok { "OK" } else { "MISMATCH" },
                got.chars().count()
            ));

            if args.freeze {
                // Mirror the on-disk format written by `baseline_snapshot.rs`.
                let body = format!(
                    "backend: {tag}\nmodel: {model}\nwav: {wav}\naudio_s: {audio_s:.3}\nelapsed_s: {elapsed:.3}\nrtfx: {:.3}\ndetected_language: {}\n\n{}\n",
                    if elapsed > 0.0 { audio_s / elapsed } else { 0.0 },
                    r.language,
                    got
                );
                std::fs::write(&base_file, &body).expect("write baseline");
            } else if !ok {
                let out = root.join("target");
                let _ = std::fs::write(out.join(format!("verify_want_{model}_{stem}.txt")), &want);
                println!("  wrote target/verify_want_{model}_{stem}.txt + verify_got_{model}_{stem}.txt");

                let a = want.chars().collect::<Vec<_>>();
                let b = got.chars().collect::<Vec<_>>();
                match (0..a.len().min(b.len())).find(|&i| a[i] != b[i]) {
                    Some(i) => {
                        let lo = i.saturating_sub(30);
                        let hi_a = (i + 30).min(a.len());
                        let hi_b = (i + 30).min(b.len());
                        println!("  first diff at char {i}");
                        println!("  want…{}", a[lo..hi_a].iter().collect::<String>());
                        println!("  got …{}", b[lo..hi_b].iter().collect::<String>());
                    }
                    None => println!(
                        "  (one string is a prefix of the other: chars want={} got={})",
                        a.len(),
                        b.len()
                    ),
                }
            }
        }

        if !args.freeze {
            if fail == 0 {
                println!("  -> {model}: all fixtures byte-identical to the frozen baseline");
            } else {
                println!("  -> {model}: {fail} fixture(s) DIVERGED");
            }
        }
        total_fail += fail;

        let report_file = root
            .join("target")
            .join(format!("verify_report_{tag}_{model}.tsv"));
        let _ = std::fs::write(&report_file, &report);
        println!("  report: {}", report_file.display());
    }

    if total_fail > 0 && !args.freeze {
        println!("\n{total_fail} fixture(s) DIVERGED from the frozen baseline");
        std::process::exit(1);
    }
}
