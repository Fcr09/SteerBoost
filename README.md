# SteerPred — Supplementary Code

Implementation  covers the three
stages reported in the experimental section:

1. **Data curation** — for each of 150 concepts, generate steered model
   rollouts on AlpacaEval prompts, capture paired *raw* and *steered* hidden
   states, and label every (concept, prompt, alpha) triple with GPT.
2. **XGBoost early predictor** — train a steerability classifier on
   handcrafted features computed from the paired hidden states.
3. **AlphaSearch** — compare alpha-selection strategies under a budget
   *K* (training-set CGS, oracle CGS, item-level GS, ascending /
   descending IGS, and our XGBoost-ranked search).

The code is self-contained: nothing outside this repository is required at
runtime. Hidden-state `.pt` files (>500 GB) are **not** shipped — reproducing
them on a new LLM means executing `scripts/01_run_steering.sh` and
`scripts/02_extract_raw_hidden_states.sh` yourself.

To skip the expensive steps, you can pull our pre-computed **steered
generations + GPT judgments + feature caches** for the three paper models
(Qwen3-1.7B, gemma-2-2b-it, Llama-3.2-3B-Instruct, DiffMean & LinearProbe,
Layer 10) from the companion Hugging Face dataset:

```bash
pip install huggingface_hub
python scripts/download_data.py        # ~9 GB, populates ./data/
# Now you can jump straight to step 5 (train_xgb) or step 6 (alphasearch).
```

Dataset:
[`Fcr09/SteerBoost-data`](https://huggingface.co/datasets/Fcr09/SteerBoost-data).

---

## Layout

```
supplementary/
├── README.md                       — this file
├── requirements.txt
├── seed_instructions.json          — used when GPT generates training examples
├── data/
│   ├── concepts.json               — 150 target concepts (id 0..149) used in the paper
│   └── alpaca_eval.json            — 805 AlpacaEval prompts (input distribution)
│
├── steering.py                     — DiffMean / LinearProbe + steered generation
├── pipeline.py                     — concept-level wrapper around a SteeringMethod
├── utils.py                        — I/O helpers, judgment loading, paired-data loader
├── dataset_steering.py             — entry point: steered rollouts + hidden states
├── raw_hidden_state.py             — entry point: paired *unsteered* hidden states
├── submit_gpt_judgment.py          — entry point: queue OpenAI batch judgments
├── process_judgment_results.py     — entry point: poll & write judgments back
├── alphasearch.py                  — entry point: alpha-selection comparison + plots
│
├── xgb/
│   ├── utils.py                    — handcrafted-feature extractor
│   ├── concept_partition.py        — stratified ID/OOD split over the 150 concepts
│   ├── feature_cache.py            — .npz feature-matrix loader/saver
│   ├── train.py                    — feature/label helpers shared by the trainer
│   ├── train_weighted_search.py    — entry point: weighted XGB hyperparameter search
│   ├── cache_judgments.py          — entry point: collapse result JSONs into one cache
│   └── build_feature_cache.py      — entry point: collapse hidden-state .pt files into .npz
│
└── scripts/
    ├── 01_run_steering.sh          — sweep alphas: steered rollouts + hidden states
    ├── 02_extract_raw_hidden_states.sh
    ├── 03_judge_with_gpt.sh        — submit GPT judgments (then poll)
    ├── 04_build_caches.sh          — judgments + features
    ├── 05_train_xgb.sh             — train the early predictor
    └── 06_run_alphasearch.sh       — produce the AlphaSearch tables / Pareto figures
```

After a full run the directory will also contain:

```
data/training/{concept_id}.json         cached GPT-generated pos/neg examples
data/result/{method}/{model}/{cid}/{layer}-{alpha}.json
data/hidden_states/{method}/{model}/{cid}/{layer}-{alpha}.pt        steered HS
data/hidden_states_raw/{method}/{model}/{cid}/{layer}-{alpha}.pt    raw HS
data/cache/judgments__{method}__{model}__L{layer}.json
data/cache/features__{method}__{model}__L{layer}.npz  (+ _meta.json)
saved_predictors_xgb/{model}/{method}/(model.json|splits.json|...)
alphasearch_out/{model}/{method}/(results.json|pareto.pdf|pareto_front.pdf)
```

---

## Setup

```bash
conda create -n steerpred python=3.12 -y
conda activate steerpred
pip install -r requirements.txt

# Required for steps 1, 3 (concept example generation + GPT judgment).
export OPENAI_API_KEY="sk-..."
# Optional: alternate base URL.
# export OPENAI_BASE_URL="https://us.api.openai.com/v1"
```

A single GPU is sufficient for the rollout / hidden-state steps; the XGB
trainer and AlphaSearch run on CPU.

---

## End-to-end pipeline

All scripts read configuration from environment variables — defaults match
the Qwen3-1.7B / DiffMean / Layer-10 setting reported in the paper. Override
as needed:

```bash
export METHOD=diffmean              # or probe
export MODEL=Qwen/Qwen3-1.7B        # any HF causal-LM
export MODEL_SHORT=Qwen3-1.7B       # last path component used in folder names
export LLM=Qwen3-1.7B               # short name for downstream scripts
export LAYER=10
export ALPHA_START=0.2 ALPHA_STOP=9.0 ALPHA_STEP=0.2
export K_LIST="1,2,3,5,10,15"
export N_LIST="1,3,5"
export NUM_SAMPLES=50               # AlpacaEval prompts per concept
export PARTITION=stratified_shuffle # stratified ID/OOD over abstraction levels
```

### 1. Steered rollouts + steered hidden states (GPU, ~hours per alpha)

```bash
bash scripts/01_run_steering.sh
```

Iterates over `data/concepts.json` and the alpha grid. For each concept it
(a) ensures `data/training/{cid}.json` exists, calling GPT to generate the
positive/negative example bank if not, (b) computes the steering vector
(DiffMean of the two banks, or LinearProbe weights), and (c) generates
steered responses on `--num_samples` AlpacaEval prompts while a forward hook
captures hidden states at layer L and L+k for the requested
`target_token_n_list` positions.

### 2. Paired *raw* hidden states (GPU)

```bash
bash scripts/02_extract_raw_hidden_states.sh
```

Replays the **unsteered** model on the same `(prompt, generated response)`
text pairs and captures hidden states at the matching positions / layers,
giving us aligned `(raw, steered)` pairs that the predictor's features are
defined on. This step is technically optional — `utils.load_paired_data`
will lazy-generate any missing raw files — but doing it once up front avoids
spinning up the model repeatedly inside training scripts.

### 3. GPT judgment of every (concept, prompt, alpha) triple

```bash
bash scripts/03_judge_with_gpt.sh
# wait for OpenAI batches to finish, then:
python process_judgment_results.py --last_k <num_batches> --watch
```

The judge gives each response a label in `{0: under-steer, 1: successful
steer, 2: over-steer}` according to the rubric in `submit_gpt_judgment.py`.
Labels are written back into the corresponding
`data/result/{method}/{model}/{cid}/{layer}-{alpha}.json`.

### 4. Build judgment + feature caches

```bash
bash scripts/04_build_caches.sh
```

* `data/cache/judgments__*.json` — every (cid, alpha, sample) -> label, plus
  the average response length used by AlphaSearch's token-cost accounting.
* `data/cache/features__*.npz` — the full ID+OOD feature matrix
  (≈4-6k features per row in the default config), with a sidecar
  `_meta.json` describing the build configuration. Re-running the trainer
  or AlphaSearch with the same configuration is now ~50× faster.

### 5. Train the XGBoost early predictor

```bash
bash scripts/05_train_xgb.sh
```

* 120 ID concepts vs 30 OOD concepts (`stratified_shuffle`), 70/10/20
  prompt-level train/val/test split per concept (seeded).
* Class-balanced sample weights, `n_trials` random hyperparameter samples;
  the validation-best model is saved to
  `saved_predictors_xgb/{llm}/{method}/model.json` together with
  `splits.json` (for AlphaSearch), `result.json` (metrics + class-wise confusion
  matrices), `feature_importance.json`, and a `top_features.png` heatmap.

### 6. AlphaSearch

```bash
bash scripts/06_run_alphasearch.sh
```

Reports, on both the held-out ID test prompts and the OOD concepts, the
budget-vs-success-rate trade-off for every alpha-selection strategy and
writes `pareto.pdf` (baselines + Ours at each *K*) and `pareto_front.pdf`
(non-dominated frontier of Ours over (*K*, threshold)) to
`alphasearch_out/{llm}/{method}/`.

To regenerate the figures from a previously saved `results.json` without
recomputing anything, use the plot-only mode:

```bash
python alphasearch.py --from-results alphasearch_out/Qwen3-1.7B/diffmean/results.json
```

---

## Concept list

`data/concepts.json` contains the 150 concepts referenced as concept_id
0..149 throughout the paper. Concepts are stratified into three abstraction
levels by `xgb/concept_partition.py`:

| Level | Concept IDs                            |
|-------|----------------------------------------|
| low   | 0–39 + 120–129                         |
| mid   | 40–79 + 130–139                        |
| high  | 80–119 + 140–149                       |

The default `stratified_shuffle` partition draws 40 ID + 10 OOD concepts
**per level** (seeded), giving 120 ID / 30 OOD overall while preserving the
abstraction-level mix in both halves. The legacy contiguous split (ID
0–119, OOD 120–149) is available as `--id_ood_partition static_ranges`.

---

## Reproducing the paper's main numbers

Default flags in every shell script reproduce the Qwen3-1.7B + DiffMean +
Layer 10 row of the main results table. To produce the LinearProbe row,
set `METHOD=probe`. To produce the gemma-2-2b-it / Llama-3.2-3B-Instruct
rows, set `MODEL=google/gemma-2-2b-it MODEL_SHORT=gemma-2-2b-it
LLM=gemma-2-2b-it` (resp. `meta-llama/Llama-3.2-3B-Instruct
MODEL_SHORT=Llama-3.2-3B-Instruct LLM=Llama-3.2-3B-Instruct`).

The *layer* used for steering is model-specific. For the models in the
paper we used:

| Model                    | Layer (L) |
|--------------------------|-----------|
| google/gemma-2-2b-it     | 10        |
| Qwen/Qwen3-1.7B          | 10        |
| meta-llama/Llama-3.2-3B  | 10        |

`K_LIST` (offsets `L+k` to capture) and `N_LIST` (generated-token positions)
are model-agnostic in our experiments and default to the values listed at
the top of this file.

---