# Pre-refactor baselines

Frozen transcription text + RTFx for regression after architecture refactors.

- **Do not edit texts by hand.** Regenerate with:

  ```bash
  cargo run --release --example baseline_snapshot
  ```

- Summary: `pre-refactor.md` / `pre-refactor.tsv`
- Per-run dumps: `texts/{backend}_{model}_{wav}.txt`
- Smoke vs short texts: `cargo run --release --example smoke_baseline`
