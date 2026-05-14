#!/usr/bin/env bash
# Step 6: Early-prediction vs grid-search comparison (Section "AlphaSearch"
# in the paper).  Evaluates several alpha-selection strategies under a
# budget K:
#   TCGS  - training-set concept-level grid search (single alpha per concept)
#   CGS   - concept-level grid search oracle
#   IGS   - item-level grid search upper bound
#   IGS-A - item-level grid search, ascending alpha
#   IGS-D - item-level grid search, descending alpha
#   Ours  - alpha ranking from XGBoost P(success), with optional confidence threshold
#
# Pareto plots are written under alphasearch_out/{llm}/{method}/.

set -euo pipefail

SUPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUPP_ROOT"

METHOD="${METHOD:-diffmean}"
LLM="${LLM:-Qwen3-1.7B}"
LAYER="${LAYER:-10}"
TOP_K_VALUES="${TOP_K_VALUES:-5,10,15,20}"
JUDGMENTS_CACHE="${JUDGMENTS_CACHE:-data/cache/judgments__${METHOD}__${LLM}__L${LAYER}.json}"
FEATURES_CACHE="${FEATURES_CACHE:-data/cache/features__${METHOD}__${LLM}__L${LAYER}.npz}"

python alphasearch.py \
    --llm "$LLM" \
    --steering_method "$METHOD" \
    --layer "$LAYER" \
    --top_k_values "$TOP_K_VALUES" \
    --judgments_cache "$JUDGMENTS_CACHE" \
    --feature_cache "$FEATURES_CACHE" \
    --threshold_sweep
