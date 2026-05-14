---
license: mit
pretty_name: SteerBoost — Steered Generations, GPT Judgments, and Feature Caches
size_categories:
  - 100K<n<1M
task_categories:
  - text-generation
language:
  - en
tags:
  - steering
  - representation-engineering
  - alpaca-eval
  - interpretability
configs:
  - config_name: cache_features
    data_files:
      - split: train
        path: "cache/features__*.npz"
  - config_name: cache_judgments
    data_files:
      - split: train
        path: "cache/judgments__*.json"
---

# SteerBoost Data

Companion artifacts for the SteerBoost paper. The associated code is at
[`Fcr09/SteerBoost`](https://github.com/Fcr09/SteerBoost).

This dataset distributes the **expensive intermediates** needed to reproduce the
XGBoost early-predictor and AlphaSearch results without burning GPU hours:

| Asset | What it is | Compressed size |
|---|---|---|
| `cache/features__*.npz` | Hand-crafted features per (concept, prompt, alpha), used by `xgb/train_weighted_search.py` and `alphasearch.py` | ~1.9 GB |
| `cache/judgments__*.json` | GPT labels in `{0: under-steer, 1: success, 2: over-steer}` plus average response length per (cid, alpha, sample) | ~25 MB |
| `result__{method}__{model}.tar.zst` | Per-(method, model) tarball of `result/{method}/{model}/{cid}/{layer}-{alpha}.json` — every steered generation with its GPT label | ~1–2 GB total |
| `training/{cid}.json` | GPT-generated positive/negative example bank per concept (used by `pipeline.py` to fit DiffMean / LinearProbe) | ~10 MB |
| `concepts.json` | The 150 target concepts (id 0–149) referenced throughout the paper | 14 KB |
| `alpaca_eval.json` | 805 AlpacaEval prompts (input distribution) | 620 KB |

**Not included:** raw / steered hidden-state `.pt` files (>500 GB on disk).
Regenerating them requires running `scripts/01_run_steering.sh` and
`scripts/02_extract_raw_hidden_states.sh` from the code repo.

## Coverage

- **Methods:** `diffmean`, `probe_l2-0.01`
- **Models:** `Qwen3-1.7B`, `gemma-2-2b-it`, `Llama-3.2-3B-Instruct`
- **Layer:** 10 (the paper setting; see code README for the per-model L)
- **Alphas:** 0.2 → 9.0, step 0.2 (45 values per concept)
- **Concepts:** 150 stratified across three abstraction levels
- **Prompts:** 50 AlpacaEval prompts per (concept, alpha)

## Quick start

```bash
pip install huggingface_hub
git clone https://github.com/Fcr09/SteerBoost.git
cd SteerBoost
python scripts/download_data.py            # populates ./data/

# Train the XGBoost predictor (CPU, minutes):
bash scripts/05_train_xgb.sh

# Run AlphaSearch (CPU, minutes):
bash scripts/06_run_alphasearch.sh
```

## File-name convention

```
features__{method}__{model}__L{layer}.npz       # feature matrix + meta sidecar
judgments__{method}__{model}__L{layer}.json     # cid -> alpha -> sample -> label
result__{method}__{model}.tar.zst               # extract under ./data/
training/{cid}.json                             # pos/neg example bank
```

`{method}` ∈ {`diffmean`, `probe_l2-0.01`}, `{model}` ∈ {`Qwen3-1.7B`,
`gemma-2-2b-it`, `Llama-3.2-3B-Instruct`}, `{layer}` = `10`.

## License

Released under MIT, except `data/alpaca_eval.json` which inherits the
[AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) license (Apache-2.0).

## Citation

```bibtex
@article{steerboost2026,
  title   = {SteerBoost: TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```
