#!/usr/bin/env bash
# Step 3 of data curation: GPT-judged labels.
#
#   3a) submit_gpt_judgment.py builds a JSONL file of OpenAI batch requests for
#       every (concept, alpha, prompt) without a judgment yet and submits it.
#       It prints one or more BATCH_IDs.  Each batch can take up to 24h.
#
#   3b) process_judgment_results.py polls those BATCH_IDs and, once they are
#       complete, writes `judgment` and `explanation` fields back into the
#       data/result/.../*.json files.  Re-running submit_gpt_judgment.py after
#       this step is a no-op for already-judged samples.
#
# After this script is done, data/result/ contains the final 0/1/2 labels
# (under-steer / successful-steer / over-steer) used by the XGBoost predictor.

set -euo pipefail

SUPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUPP_ROOT"

METHOD="${METHOD:-diffmean}"
MODEL_SHORT="${MODEL_SHORT:-Qwen3-1.7B}"
CONCEPT_FILE="${CONCEPT_FILE:-data/concepts.json}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set."
    exit 1
fi

echo "=== 3a) Submitting GPT batch judgment requests ==="
python submit_gpt_judgment.py \
    --method "$METHOD" \
    --model "$MODEL_SHORT" \
    --concept_file "$CONCEPT_FILE"

cat <<'MSG'

Batches have been queued with OpenAI.  After they finish (typically a few
hours, up to 24h), run step 3b to download the results and merge judgments
back into data/result/:

    python process_judgment_results.py --last_k <number_of_batches_just_submitted>

or, if you saved the batch IDs printed above:

    python process_judgment_results.py --batch_ids batch_xxx batch_yyy ...

Add --watch to poll until every batch finishes.
MSG
