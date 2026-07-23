use std::time::Instant;
use qwen3_asr::{AsrInference, Backend, TranscribeOptions};

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = root.join("models/Qwen3-ASR-0.6B");
    let fixtures = root.join("tests/fixtures");
    let base = root.join("docs/baseline/texts");
    let cases = [
        ("cuda", Backend::Cuda, "15s_en.wav", "cuda_0.6B_15s_en.txt"),
        ("cuda", Backend::Cuda, "30s_zh.wav", "cuda_0.6B_30s_zh.txt"),
        ("cpu", Backend::Cpu, "15s_en.wav", "cpu_0.6B_15s_en.txt"),
        ("cpu", Backend::Cpu, "30s_zh.wav", "cpu_0.6B_30s_zh.txt"),
    ];
    for (tag, backend, wav, base_name) in cases {
        let eng = AsrInference::load(&model, backend).expect("load");
        let path = fixtures.join(wav);
        let t0 = Instant::now();
        let r = eng.transcribe(path.to_str().unwrap(), TranscribeOptions::default().with_max_new_tokens(512)).expect("tx");
        let elapsed = t0.elapsed().as_secs_f64();
        let base_txt = std::fs::read_to_string(base.join(base_name)).expect("baseline");
        // baseline file has metadata then blank line then text
        let want = base_txt.split("\n\n").nth(1).unwrap_or("").trim();
        let got = r.text.trim();
        let ok = got == want;
        println!("{tag} {wav}: elapsed={elapsed:.2}s match={ok} lang={}", r.language);
        if !ok {
            println!("  WANT: {want}");
            println!("  GOT:  {got}");
            std::process::exit(1);
        }
    }
    println!("all smoke checks passed");
}
