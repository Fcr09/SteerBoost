#!/usr/bin/env bash
# Step 1 of data curation:
#   For every (concept, alpha) pair, generate a steered response on the AlpacaEval
#   prompts AND collect the hidden states needed for early prediction.
#
# Outputs (under SUPP_ROOT/data):
#   data/training/{concept_id}.json
#       per-concept positive / negative training examples (cached so repeated
#       runs don't re-call GPT)
#   data/result/{method}/{model_short}/{concept_id}/{layer}-{alpha}.json
#       generation results for each (concept, alpha)
#   data/hidden_states/{method}/{model_short}/{concept_id}/{layer}-{alpha}.pt
#       steered hidden states at layer L and L+k for the captured tokens
#
# Run from the supplementary/ directory.

set -euo pipefail

SUPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUPP_ROOT"

METHOD="${METHOD:-diffmean}"          # diffmean | probe
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"     # any HF causal-LM repo id
LAYER="${LAYER:-10}"                  # steering layer L
NUM_SAMPLES="${NUM_SAMPLES:-50}"      # AlpacaEval prompts per concept
NUM_EXAMPLES="${NUM_EXAMPLES:-50}"    # GPT-generated pos/neg examples per concept
BATCH_SIZE="${BATCH_SIZE:-8}"
K_LIST="${K_LIST:-1,2,3,5,10,15}"     # L+k offsets to capture
N_LIST="${N_LIST:-1,3,5}"             # generated token positions to capture
ALPHA_START="${ALPHA_START:-0.2}"
ALPHA_STOP="${ALPHA_STOP:-9.0}"
ALPHA_STEP="${ALPHA_STEP:-0.2}"
CONCEPT_FILE="${CONCEPT_FILE:-data/concepts.json}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "WARNING: OPENAI_API_KEY is unset; concept example generation will fail "
    echo "the first time a concept is seen.  Set it before running this script."
fi

mapfile -t ALPHAS < <(python - <<PY
import numpy as np
for a in np.arange(${ALPHA_START}, ${ALPHA_STOP} + 1e-9, ${ALPHA_STEP}):
    print(f"{round(float(a), 2)}")
PY
)

echo "Sweeping ${#ALPHAS[@]} alphas: ${ALPHAS[0]} ... ${ALPHAS[-1]}"

for alpha in "${ALPHAS[@]}"; do
    echo "=== alpha=${alpha} ==="
    python dataset_steering.py \
        --method "$METHOD" \
        --model "$MODEL" \
        --layer "$LAYER" \
        --alpha "$alpha" \
        --num_samples "$NUM_SAMPLES" \
        --num_examples "$NUM_EXAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --concept_file "$CONCEPT_FILE" \
        --collect_hidden_states \
        --k_list "$K_LIST" \
        --target_token_n_list "$N_LIST"
done

echo "Done.  Steered rollouts and hidden states are under data/result/ and data/hidden_states/."
