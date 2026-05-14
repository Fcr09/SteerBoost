---
license: mit
pretty_name: SteerBoost Data
viewer: false
---

# SteerBoost Data

Companion artifacts for the [SteerBoost](https://github.com/Fcr09/SteerBoost)
codebase: GPT-judged steered generations and pre-built feature caches.
Hidden-state `.pt` files are **not** included (>500 GB).

## Contents

- `cache/features__{method}__{model}__L10.npz` — feature matrices
- `cache/judgments__{method}__{model}__L10.json` — GPT labels
- `result__{method}__{model}.tar.zst` — steered generations (extract under `data/`)
- `training/{cid}.json` — pos/neg example banks
- `concepts.json`, `alpaca_eval.json`

`method` ∈ {`diffmean`, `probe_l2-0.01`}, `model` ∈ {`Qwen3-1.7B`,
`gemma-2-2b-it`, `Llama-3.2-3B-Instruct`}.

## Usage

```bash
git clone https://github.com/Fcr09/SteerBoost.git && cd SteerBoost
pip install -r requirements.txt
python scripts/download_data.py
```
