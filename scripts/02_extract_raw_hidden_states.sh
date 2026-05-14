#!/usr/bin/env bash
# Step 2 of data curation:
#   For every existing steered-rollout file, replay the *unsteered* model on the
#   same (prompt, generated response) text and capture the *raw* hidden states
#   at the same token positions / layers.  These paired (raw, steered) tensors
#   are what the XGBoost predictor reads at training and inference time.
#
# Outputs:
#   data/hidden_states_raw/{method}/{model_short}/{concept_id}/{layer}-{alpha}.pt
#
# Note: utils.load_paired_data also lazily generates these if they are
# missing, so this step is technically optional but pre-computing them once
# makes downstream loops much faster.

set -euo pipefail

SUPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUPP_ROOT"

METHOD="${METHOD:-diffmean}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
LAYER="${LAYER:-10}"
K_LIST="${K_LIST:-1,2,3,5,10,15}"
N_LIST="${N_LIST:-1,3,5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ALPHA_START="${ALPHA_START:-0.2}"
ALPHA_STOP="${ALPHA_STOP:-9.0}"
ALPHA_STEP="${ALPHA_STEP:-0.2}"

mapfile -t ALPHAS < <(python - <<PY
import numpy as np
for a in np.arange(${ALPHA_START}, ${ALPHA_STOP} + 1e-9, ${ALPHA_STEP}):
    print(f"{round(float(a), 2)}")
PY
)

for alpha in "${ALPHAS[@]}"; do
    echo "=== alpha=${alpha} ==="
    python raw_hidden_state.py \
        --method "$METHOD" \
        --model "$MODEL" \
        --layer "$LAYER" \
        --alpha "$alpha" \
        --k_list "$K_LIST" \
        --target_token_n_list "$N_LIST" \
        --batch_size "$BATCH_SIZE"
done
