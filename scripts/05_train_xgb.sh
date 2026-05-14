#!/usr/bin/env bash
# Step 5: train the XGBoost early-prediction classifier.
#
# Uses random hyperparameter search (n_trials) on a held-out validation
# split, then evaluates the best model on (a) ID test prompts and (b) OOD
# concepts.  Saves model.json, splits.json (used by AlphaSearch to recover
# the same train/test split), normalization stats, and feature importances.
#
# Output goes under saved_predictors_xgb/{llm}/{method}/.

set -euo pipefail

SUPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUPP_ROOT"

METHOD="${METHOD:-diffmean}"
LLM="${LLM:-Qwen3-1.7B}"
LAYER="${LAYER:-10}"
SEED="${SEED:-42}"
PARTITION="${PARTITION:-stratified_shuffle}"
N_TRIALS="${N_TRIALS:-20}"
K_LIST="${K_LIST:-1,2,3,5,10,15}"
N_LIST="${N_LIST:-1,3,5}"
FEATURES_CACHE="${FEATURES_CACHE:-data/cache/features__${METHOD}__${LLM}__L${LAYER}.npz}"

python xgb/train_weighted_search.py \
    --llm "$LLM" \
    --steering_method "$METHOD" \
    --curr_layer "$LAYER" \
    --k_list "$K_LIST" \
    --target_token_n_list "$N_LIST" \
    --id_ood_partition "$PARTITION" \
    --seed "$SEED" \
    --n_trials "$N_TRIALS" \
    --features_cache "$FEATURES_CACHE"
