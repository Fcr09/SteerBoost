#!/usr/bin/env bash
# Step 4: build cached judgment + feature matrices so XGB training and
# AlphaSearch don't have to re-scan thousands of JSONs / re-load .pt files.
#
# Outputs:
#   data/cache/judgments__{method}__{llm}__L{layer}.json
#   data/cache/features__{method}__{llm}__L{layer}.npz  (+ _meta.json)

set -euo pipefail

SUPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUPP_ROOT"

METHOD="${METHOD:-diffmean}"
LLM="${LLM:-Qwen3-1.7B}"
LAYER="${LAYER:-10}"
SEED="${SEED:-42}"
PARTITION="${PARTITION:-stratified_shuffle}"   # or static_ranges
K_LIST="${K_LIST:-1,2,3,5,10,15}"
N_LIST="${N_LIST:-1,3,5}"

mkdir -p data/cache

JUDGMENTS_CACHE="${JUDGMENTS_CACHE:-data/cache/judgments__${METHOD}__${LLM}__L${LAYER}.json}"
FEATURES_CACHE="${FEATURES_CACHE:-data/cache/features__${METHOD}__${LLM}__L${LAYER}.npz}"

echo "=== Judgment cache -> $JUDGMENTS_CACHE ==="
python xgb/cache_judgments.py \
    --steering_method "$METHOD" \
    --llm "$LLM" \
    --layer "$LAYER" \
    --include_token_stats \
    --output "$JUDGMENTS_CACHE" \
    --force

echo "=== Feature cache -> $FEATURES_CACHE ==="
python xgb/build_feature_cache.py \
    --steering_method "$METHOD" \
    --llm "$LLM" \
    --curr_layer "$LAYER" \
    --k_list "$K_LIST" \
    --target_token_n_list "$N_LIST" \
    --id_ood_partition "$PARTITION" \
    --seed "$SEED" \
    --output "$FEATURES_CACHE" \
    --force
