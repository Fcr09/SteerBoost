import json
import os
from openai import OpenAI
import argparse
from pathlib import Path
from utils import load_concepts
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="diffmean")
parser.add_argument("--model", type=str, default="gemma-2-2b-it")
parser.add_argument("--concept_file", type=str, default="data/concepts.json")
args = parser.parse_args()


evaluation_prompt = """
You are an expert evaluator assessing the effectiveness of AI model steering. You will evaluate whether a model's response successfully incorporates a target concept while maintaining coherence and relevance to the original prompt.

## Task
You are given:
1. An original prompt/instruction
2. A target concept that should be steered towards
3. A model's response after steering

## Evaluation Criteria
Rate the response on a scale of 0-2:

**0 - Under-steering (Concept Missing)**
- The response does NOT incorporate the target concept
- The response is coherent and relevant to the prompt, but lacks the desired steering
- Example: Target "response with emojis" but response contains no emojis

**1 - Successful Steering (Concept Present & Coherent)**
- The response successfully incorporates the target concept
- The response remains coherent, relevant, and appropriate to the original prompt
- The steering feels natural and doesn't break the response quality
- Example: Target "response with humor" and response includes appropriate humor while answering the question

**2 - Over-steering (Incoherent Answer)**
- The response does not answer the original question and becomes nonsensical, irrelevant, or inappropriate to the original prompt
- The steering is so strong that it breaks the response quality
- When response is empty, it should be rated as over-steering
- When response is gibberish and does not answer the original question, it should be rated as over-steering rather than under-steering
- Example: Target "response with emojis" but response contains emojis throughout while completely ignoring the original question

## Guidelines
- Consider the appropriateness of the concept for the given prompt
- Evaluate whether the steering enhances or detracts from the response quality
- Be strict about coherence - a response that doesn't make sense should be rated 2
- Consider cultural and linguistic appropriateness

## Input Data
**Original Prompt:** {prompt}
**Target Concept:** {concept}
**Steered Response:** {response_after}

## Output Format
Return your evaluation in JSON format:
{{
    "explanation": "Brief explanation of your judgment, including specific evidence from the response",
    "judgment": <0, 1, or 2>
}}
"""


concepts = load_concepts(args.concept_file)
batch_tasks = []

print("Preparing batch tasks...")
for concept in tqdm.tqdm(concepts):
    concept_id = concept["concept_id"]
    concept_name = concept["concept_name"]
    result_dir = Path(f"data/result/{args.method}/{args.model}/{concept_id}")
    
    if not result_dir.exists():
        continue

    for file in result_dir.glob("*.json"):
        alpha = float(file.name.split("-")[1].replace(".json", ""))

        with open(file, "r") as f:
            file_result = json.load(f)
            
        if "under_steering" in file_result["metadata"].keys():
            continue
            
        results = file_result["results"]
        for i, result in enumerate(results):
            # if result["judgment"]!=0:
            #     continue
            prompt = result["prompt"]
            response_after = result["response_after"]
            evaluation_prompt_formatted = evaluation_prompt.format(prompt=prompt, concept=concept_name, response_after=response_after)
            
            # Create a custom_id that allows us to map back to the exact record
            # Format: method::model::concept_id::filename::index
            custom_id = f"{args.method}::{args.model}::{concept_id}::{file.name}::{i}"
            
            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                # Batch uses /v1/chat/completions; it does not accept `reasoning` (use Responses API for that).
                "body": {
                    "model": "gpt-5-mini",
                    "messages": [{"role": "user", "content": evaluation_prompt_formatted}],
                    "response_format": {"type": "json_object"},
                }
            }
            batch_tasks.append(task)

if not batch_tasks:
    print("No tasks to process.")
    exit(0)

# OpenAI batch: max 50k lines per batch, but gpt-5-mini also enforces ~200MB input file size.
MAX_BATCH_REQUESTS = 15_000
n_tasks = len(batch_tasks)
chunks = [
    batch_tasks[i : i + MAX_BATCH_REQUESTS]
    for i in range(0, n_tasks, MAX_BATCH_REQUESTS)
]

print(f"Found {n_tasks} tasks in {len(chunks)} batch(es) (max {MAX_BATCH_REQUESTS} requests each).")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://us.api.openai.com/v1"
)

for chunk_idx, chunk in enumerate(chunks):
    if len(chunks) == 1:
        batch_filename = "batch_input.jsonl"
    else:
        batch_filename = f"batch_input_{chunk_idx}.jsonl"

    print(f"Creating batch file '{batch_filename}' ({len(chunk)} tasks)...")
    with open(batch_filename, "w") as f:
        for task in chunk:
            f.write(json.dumps(task) + "\n")

    print(f"Submitting to OpenAI Batch API ({batch_filename})...")
    try:
        with open(batch_filename, "rb") as fh:
            batch_file = client.files.create(file=fh, purpose="batch")

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f"Batch job created successfully!")
        print(f"Batch ID: {batch_job.id}")
        print(f"Input File ID: {batch_file.id}")
        print(f"Status: {batch_job.status}")
        if len(chunks) > 1:
            print()
    except Exception as e:
        print(f"Error submitting batch ({batch_filename}): {e}")

if len(chunks) > 1:
    print(
        "\nTo retrieve results later, you will need to check each batch above:"
        "\n1. Check status: client.batches.retrieve('BATCH_ID')"
        "\n2. Once completed, download each output file."
        "\n3. Parse the output and map back using the custom_id (method::model::concept_id::filename::index)."
    )
else:
    print("\nTo retrieve results later, you will need to:")
    print("1. Check status: client.batches.retrieve('BATCH_ID')")
    print("2. Once completed, download the output file.")
    print("3. Parse the output and map back using the custom_id (method::model::concept_id::filename::index).")
