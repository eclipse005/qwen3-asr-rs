//! Compare two `cpu_transcribe` stdout logs (f16 vs INT8) and print per-fixture CER + RTFx.
//!
//! Parses lines of the form:
//!   `... CPU <name> | <e>s elapsed | RTFx <x>|— | <lang> | <transcript>`
//! and reports per-fixture RTFx, speedup, and character-error-rate (CER) of the int8
//! transcript against the f16 baseline.
//!
//! Usage: `cargo run --example cer_compare --no-default-features --features cpu -- <f16.log> <int8.log>`

use std::collections::BTreeMap;
use std::env;
use std::fs;

/// name -> (rtfx or NaN, transcript)
fn parse_log(path: &str) -> BTreeMap<String, (f64, String)> {
    let s = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {}", path, e);
        std::process::exit(1);
    });
    let mut out = BTreeMap::new();
    for line in s.lines() {
        let Some(cpu) = line.find("CPU ") else { continue };
        let rest = &line[cpu + 4..]; // "<name> | <e>s elapsed | RTFx <x> | <lang> | <text>"
        let parts: Vec<&str> = rest.splitn(5, " | ").collect();
        if parts.len() < 5 {
            continue;
        }
        let name = parts[0].trim().to_string();
        let rtfx = parts[2]
            .trim_start_matches("RTFx ")
            .trim_end_matches('x')
            .parse::<f64>()
            .unwrap_or(f64::NAN);
        let text = parts[4].trim().to_string();
        out.insert(name, (rtfx, text));
    }
    out
}

/// Strip whitespace + punctuation (CJK and ASCII) so CER reflects only
/// "word" characters — punctuation/spacing/paraphrase-noise doesn't inflate it.
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !c.is_whitespace()
                && !matches!(
                    c,
                    '。' | '、' | '！' | '？' | '，' | '．' | '「' | '」' | '『' | '』'
                        | '（' | '）' | '《' | '》' | '“' | '”' | '‘' | '’' | '…' | '―' | 'ー'
                        | ',' | '.' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')'
                        | '-' | '—' | '・'
                )
        })
        .collect()
}

/// Character error rate = Levenshtein(a, b) / max(|a|, |b|).
fn cer(a: &str, b: &str) -> (usize, usize) {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    let denom = m.max(n);
    if denom == 0 {
        return (0, 0);
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut cur = vec![0usize; n + 1];
    for i in 1..=m {
        cur[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    (prev[n], denom)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: cer_compare <f16.log> <int8.log>");
        std::process::exit(1);
    }
    let base = parse_log(&args[1]);
    let i8l = parse_log(&args[2]);

    println!(
        "\n{:<10} {:>9} {:>9} {:>10} {:>8} {:>8}",
        "fixture", "RTFx_f16", "RTFx_i8", "speedup", "CER%", "CER%nrm"
    );
    println!("{}", "-".repeat(62));
    let mut tot_ed = 0usize;
    let mut tot_den = 0usize;
    let mut tot_ed_n = 0usize;
    let mut tot_den_n = 0usize;
    for (name, (rb, tb)) in &base {
        if let Some((ri, ti)) = i8l.get(name) {
            let (ed, den) = cer(tb, ti);
            let (edn, denn) = cer(&normalize(tb), &normalize(ti));
            let sp = if ri.is_finite() && rb.is_finite() && *rb > 0.0 {
                ri / rb
            } else {
                f64::NAN
            };
            println!(
                "{:<10} {:>8.2}x {:>8.2}x {:>9.2}x {:>7.2}% {:>7.2}%",
                name,
                rb,
                ri,
                sp,
                100.0 * ed as f64 / den as f64,
                100.0 * edn as f64 / denn as f64,
            );
            tot_ed += ed;
            tot_den += den;
            tot_ed_n += edn;
            tot_den_n += denn;
        } else {
            println!("{:<10}  (int8 log missing this fixture)", name);
        }
    }
    if tot_den > 0 {
        println!("{}", "-".repeat(62));
        println!(
            "overall CER: {:.2}%   normalized: {:.2}%  (raw {} / {}, nrm {} / {})",
            100.0 * tot_ed as f64 / tot_den as f64,
            100.0 * tot_ed_n as f64 / tot_den_n as f64,
            tot_ed,
            tot_den,
            tot_ed_n,
            tot_den_n
        );
    }
}
