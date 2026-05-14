#!/usr/bin/env python3
"""
Shared utility functions for steering pipelines.
"""

import torch
import json
import os
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional, Iterable
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import process_file
try:
    from raw_hidden_state import process_file
except ImportError:
    pass

def load_seed_instructions(filepath="seed_instructions.json"):
    """Load seed instructions from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Flatten all categories into a single list
            instructions = []
            for category, items in data['categories'].items():
                instructions.extend(items)
            return instructions
    except FileNotFoundError:
        print(f"Warning: {filepath} not found, using default seed instructions")
        # Fallback to a small default set
        return [
            "Explain how photosynthesis works.",
            "What is artificial intelligence?",
            "How do airplanes fly?",
            "What is quantum computing?",
            "Explain how the internet works.",
            "What is climate change?",
            "How does a car engine work?",
            "What is machine learning?",
            "Explain the theory of evolution.",
            "How do you learn a new language effectively?",
        ]

SEED_INSTRUCTIONS = load_seed_instructions()


# ============================================================================
# GPT Generation Functions
# ============================================================================

def generate_examples_with_gpt(concept, concept_id, num_examples=5, model="gpt-4o-mini", data_dir=None):
    """
    Generate training examples using OpenAI's API.
    
    Args:
        concept: The concept to generate examples for
        num_examples: Number of examples per category
        model: OpenAI model to use
        
    Returns:
        Tuple of (positive_examples, negative_examples)
    """
    


    data_file = os.path.join(data_dir, f"training/{concept_id}.json")
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            data = json.load(f)
        positive_examples = data["positive"]
        negative_examples = data["negative"]
    else:
        positive_examples = []
        negative_examples = []

        
    if len(positive_examples) >= num_examples:
        positive_examples = positive_examples[:num_examples]
        positive_examples_to_generate = 0
    else:
        positive_examples_to_generate = num_examples - len(positive_examples)
    
    if len(negative_examples) >= num_examples:
        negative_examples = negative_examples[:num_examples]
        negative_examples_to_generate = 0
    else:
        negative_examples_to_generate = num_examples - len(negative_examples)
    
    if positive_examples_to_generate == 0 and negative_examples_to_generate == 0:
        return positive_examples, negative_examples

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Install with: pip install openai")
        print("Falling back to hardcoded examples...")
        return None, None
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Falling back to hardcoded examples...")
        return None, None
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://us.api.openai.com/v1"
    )
    
    # Templates
    response_with_concept = """Given the following instruction:

{instruction}

Your task is to:
1. Provide a response that incorporates elements related to '{concept}'.
2. Ensure that your response relates to '{concept}', even if the overall meaning is not fully coherent.

Return only the response to the instruction in plain text without any additional explanations."""

    response_without_concept = """Given the following instruction:

{instruction}

Your task is to:
1. Provide a response that continues or addresses the instruction naturally.
2. Avoid any mention of '{concept}' in the continuation.

Return only the response to the instruction in plain text without any additional explanations."""
    
    # Shuffle and select instructions
    random.shuffle(SEED_INSTRUCTIONS)
    selected_instructions = SEED_INSTRUCTIONS[:positive_examples_to_generate + negative_examples_to_generate]
    
    print(f"\nGenerating {positive_examples_to_generate} positive examples using GPT...")
    for i, instruction in enumerate(selected_instructions[:positive_examples_to_generate]):
        try:
            prompt = response_with_concept.format(instruction=instruction, concept=concept)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=150,
            )
            text = response.choices[0].message.content.strip().strip('"').strip("'")
            positive_examples.append(text)
            print(f"  [{i+1}/{positive_examples_to_generate}] ✓ {text[:60]}...")
        except Exception as e:
            print(f"  [{i+1}/{positive_examples_to_generate}] ✗ Error: {e}")
    
    print(f"\nGenerating {negative_examples_to_generate} negative examples using GPT...")
    for i, instruction in enumerate(selected_instructions[positive_examples_to_generate:positive_examples_to_generate + negative_examples_to_generate]):
        # try:
            prompt = response_without_concept.format(instruction=instruction, concept=concept)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=150,
            )
            text = response.choices[0].message.content.strip().strip('"').strip("'")
            negative_examples.append(text)
            print(f"  [{i+1}/{negative_examples_to_generate}] ✓ {text[:60]}...")
        # except Exception as e:
        #     print(f"  [{i+1}/{negative_examples_to_generate}] ✗ Error: {e}")
    
    with open(data_file, "w") as f:
        json.dump({"positive": positive_examples, "negative": negative_examples}, f, indent=4)
    
    return positive_examples, negative_examples


# ============================================================================
# Helper Functions
# ============================================================================
def gather_residual_activations(model, target_layer, inputs):
    """Extract hidden states from a specific layer."""
    target_act = None
    
    def hook(mod, inputs, outputs):
        nonlocal target_act
        if isinstance(outputs, tuple):
            target_act = outputs[0]
        else:
            target_act = outputs
        return outputs
    
    handle = model.model.layers[target_layer].register_forward_hook(hook, always_call=True)
    _ = model.forward(**inputs)
    handle.remove()
    return target_act

def load_alpaca_eval(alpaca_path: str, num_samples: int = 10, seed: int = 42) -> List[Dict[str, str]]:
    """
    Load AlpacaEval dataset and sample a subset.
    
    Args:
        alpaca_path: Path to alpaca_eval.json
        num_samples: Number of samples to load
        seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries with 'instruction' and 'dataset' keys
    """
    print(f"\n{'='*80}")
    print("LOADING ALPACAEVAL DATA")
    print(f"{'='*80}")
    print(f"Path: {alpaca_path}")
    
    with open(alpaca_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples available: {len(data)}")
    
    # Sample subset
    import random
    random.seed(seed)
    
    if num_samples > len(data):
        print(f"⚠ Requested {num_samples} samples but only {len(data)} available")
        num_samples = len(data)
    
    samples = random.sample(data, num_samples)
    print(f"Selected {len(samples)} samples")
    
    return samples

def save_results(
    results: List[Dict[str, Any]],
    method: str,
    model_name: str,
    layer: int,
    alpha: float,
    concept_id: int,
    method_stats: Dict[str, Any] = None,
    output_dir: str = 'data/result'
):
    """
    Save results to structured path.
    
    Args:
        results: List of result dictionaries
        method: Steering method name (probe/diffmean)
        model_name: Model name
        layer: Layer number
        alpha: Alpha value
        concept: Concept name
        output_dir: Base output directory
    """
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # Create path: data/result/{method}/{model}/{layer}-{alpha}.json
    model_short = model_name.split('/')[-1]  # e.g., "gemma-2-2b-it"
    output_path = Path(output_dir) / method / model_short / str(concept_id) / f"{layer}-{alpha}.json"
    
    # Create directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'method': method,
        'model': model_name,
        'layer': layer,
        'alpha': alpha,
        'concept_id': concept_id,
        'num_samples': len(results),
        'num_successful': len([r for r in results if 'error' not in r]),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add method statistics to metadata
    if method_stats:
        metadata.update(method_stats)
    
    # Combine metadata and results
    output_data = {
        'metadata': metadata,
        'results': results
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")
    print(f"  Total samples: {metadata['num_samples']}")
    print(f"  Successful: {metadata['num_successful']}")
    
    return output_path

def load_concepts(concept_file: str) -> List[Dict[str, Any]]:
    """
    Load concepts from JSON file.
    """
    with open(concept_file, "r") as f:
        return json.load(f)


def save_hidden_states_dataset(
    hidden_states_data: Dict[str, Any],
    method: str,
    model_name: str,
    concept_id: int,
    layer: int,
    alpha: float,
    k_list: list = [5],
    target_token_n_list: list = [5],
    output_dir: str = 'data/hidden_states'
) -> Path:
    """
    Save hidden states for dataset samples in a structured format.
    
    Data structure:
    data/hidden_states/
    ├── {method}/
    │   └── {model_short}/
    │       └── {concept_id}/
    │           └── {layer}-{alpha}.pt
    
    Args:
        hidden_states_data: Dictionary containing hidden states for all samples
        method: Steering method name (probe/diffmean)
        model_name: Model name (e.g., "google/gemma-2-2b-it")
        concept_id: Concept ID
        layer: Steering layer number
        alpha: Steering strength
        k_list: List of k values used
        target_token_n_list: List of target token n values used
        output_dir: Base output directory
        
    Returns:
        Path to saved file
    """
    print(f"\nSaving hidden states dataset...")
    
    # Create path: data/hidden_states/{method}/{model}/{concept_id}/{layer}-{alpha}.pt
    model_short = model_name.split('/')[-1]  # e.g., "gemma-2-2b-it"
    filename = f"{layer}-{alpha}.pt"
    output_path = Path(output_dir) / method / model_short / str(concept_id) / filename
    
    # Create directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'method': method,
        'model': model_name,
        'concept_id': concept_id,
        'layer': layer,
        'alpha': alpha,
        'k_list': k_list,
        'target_token_n_list': target_token_n_list,
        'num_samples': len(hidden_states_data),
        'timestamp': datetime.now().isoformat(),
        'data_structure': {
            'sample_id': 'Hidden states for each sample',
            'token_0': 'Hidden states for first generated token',
            'token_N': 'Hidden states for N-th generated token (where N is in target_token_n_list)',
            'before_steering': 'States before steering is applied',
            'after_steering': 'States after steering is applied (only for layer L)',
            'layer_L': f'Hidden states at steering layer L',
            'layer_L+k': f'Hidden states at layer L+k (where k is in k_list)'
        }
    }
    
    # Combine metadata and data
    output_data = {
        'metadata': metadata,
        'hidden_states': hidden_states_data
    }
    
    # Save to PyTorch file
    torch.save(output_data, output_path)
    
    print(f"✓ Hidden states saved: {output_path} ({len(hidden_states_data)} samples)")
    
    return output_path


def load_hidden_states_dataset(
    method: str,
    model_name: str,
    concept_id: int,
    layer: int,
    alpha: float,
    k_list: list = [5],
    target_token_n_list: list = [5],
    output_dir: str = 'data/hidden_states'
) -> Dict[str, Any]:
    """
    Load hidden states for dataset samples.
    
    Args:
        method: Steering method name (probe/diffmean)
        model_name: Model name (e.g., "google/gemma-2-2b-it")
        concept_id: Concept ID
        layer: Steering layer number
        alpha: Steering strength
        k_list: List of k values used
        target_token_n_list: List of target token n values used
        output_dir: Base output directory
        
    Returns:
        Dictionary containing hidden states data
    """
    model_short = model_name.split('/')[-1]
    k_str = "_".join(map(str, k_list))
    n_str = "_".join(map(str, target_token_n_list))
    filename = f"{layer}-{alpha}.pt"
    file_path = Path(output_dir) / method / model_short / str(concept_id) / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Hidden states file not found: {file_path}")
    
    data = torch.load(file_path, map_location='cpu')
    print(f"✓ Loaded hidden states: {data['metadata']['num_samples']} samples")
    
    return data

def load_all_data(method, model, layer=None, alpha=None, required_n=None):
    hidden_states_dir = f'data/hidden_states/{method}/{model}'
    result_dir = f'data/result/{method}/{model}'

    hidden_states = []
    result_labels = []

    for concept_id in sorted(os.listdir(hidden_states_dir)):
        for file in os.listdir(os.path.join(hidden_states_dir, f'{concept_id}')):
            if layer is not None:
                if layer != int(file.split('-')[0]):
                    continue
            if alpha is not None:
                if alpha != float(file.split('-')[1]):
                    continue
            result_data = json.load(open(os.path.join(result_dir, f'{concept_id}', file.replace('.pt', '.json'))))
            result_data = result_data['results']
            hidden_states_data = torch.load(os.path.join(hidden_states_dir, f'{concept_id}', file), map_location='cuda' if torch.cuda.is_available() else 'cpu')
            for sample_idx in range(len(result_data)):
                if not check_hidden_states_data(hidden_states_data['hidden_states'][f'sample_{sample_idx}'], required_n):
                    continue
                result_labels.append(result_data[sample_idx]['judgment'])
                hidden_states.append(hidden_states_data['hidden_states'][f'sample_{sample_idx}'])

    # Shuffle hidden_states and result_labels consistently
    combined = list(zip(hidden_states, result_labels))
    random.shuffle(combined)
    hidden_states, result_labels = zip(*combined) if combined else ([], [])
    hidden_states = list(hidden_states)
    result_labels = list(result_labels)
    return hidden_states, result_labels

def check_hidden_states_data(hidden_states, n=None):
    """
    Checking if the hidden states data is valid.
    """
    if 'token_0' not in hidden_states:
        return False  
    if 'before_steering' not in hidden_states['token_0']:
        return False
    if 'after_steering' not in hidden_states['token_0']:
        return False
    if list(hidden_states['token_0']['before_steering'].keys()) == []:
        return False
    if list(hidden_states['token_0']['after_steering'].keys()) == []:
        return False
    if n is not None:
        if f'token_{n}' not in hidden_states:
            return False
        if 'before_steering' not in hidden_states[f'token_{n}']:
            return False
        if 'after_steering' not in hidden_states[f'token_{n}']:
            return False
        if list(hidden_states[f'token_{n}']['before_steering'].keys()) == []:
            return False
        if list(hidden_states[f'token_{n}']['after_steering'].keys())  == []:
            return False
    return True


# --- Judgment aggregation (result JSONs) and cache ---------------------------------

JUDGMENT_CACHE_VERSION = 1

# For optional tokenizer stats when building judgment caches
JUDGMENT_CACHE_LLM_TO_TOKENIZER = {
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
}


def iter_concept_ids_in_result_dir(result_dir: str) -> List[int]:
    """Numeric subdirectory names under result_dir, sorted."""
    if not os.path.isdir(result_dir):
        return []
    out = []
    for name in os.listdir(result_dir):
        path = os.path.join(result_dir, name)
        if os.path.isdir(path) and name.isdigit():
            out.append(int(name))
    return sorted(out)


def load_all_judgments(
    result_dir: str,
    concept_ids: Iterable[int],
    layer: int,
) -> Tuple[Dict[int, Dict[float, Dict[int, int]]], List[float]]:
    """
    Load judgment data from per-file result JSONs under result_dir.

    Returns:
        data[cid][alpha][sample_id] = judgment (int)
        alphas: sorted unique alpha values (excluding 0).
    """
    data: Dict[int, Dict[float, Dict[int, int]]] = {}
    all_alphas_set = set()

    for cid in concept_ids:
        data[cid] = {}
        cdir = os.path.join(result_dir, str(cid))
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(os.listdir(cdir)):
            if not fname.endswith(".json"):
                continue
            parts = fname.replace(".json", "").split("-")
            if len(parts) < 2:
                continue
            try:
                file_layer = int(parts[0])
                file_alpha = round(float(parts[1]), 2)
            except ValueError:
                continue
            if file_layer != layer:
                continue
            if file_alpha == 0.0:
                continue
            all_alphas_set.add(file_alpha)
            fpath = os.path.join(cdir, fname)
            with open(fpath) as fp:
                result_data = json.load(fp)
            judgments: Dict[int, int] = {}
            for item in result_data["results"]:
                sid = item["sample_id"]
                j = item.get("judgment")
                if j is not None:
                    judgments[int(sid)] = int(j)
            data[cid][file_alpha] = judgments

    return data, sorted(all_alphas_set)


def build_judgment_cache_payload(
    result_dir: str,
    layer: int,
    steering_method: str,
    llm: str,
    concept_ids: Optional[Iterable[int]] = None,
    include_token_stats: bool = False,
) -> Dict[str, Any]:
    """
    Scan result JSONs once and build a JSON-serializable cache dict.

    If concept_ids is None, all numeric concept subdirectories are used.
    When include_token_stats is True, loads HF tokenizer and records
    mean token length of response_after (same files/layer filter as judgments).
    """
    if concept_ids is None:
        concept_ids = iter_concept_ids_in_result_dir(result_dir)
    else:
        concept_ids = list(concept_ids)

    tokenizer = None
    if include_token_stats:
        from transformers import AutoTokenizer

        tname = JUDGMENT_CACHE_LLM_TO_TOKENIZER.get(llm, llm)
        tokenizer = AutoTokenizer.from_pretrained(tname)

    data, alphas = load_all_judgments(result_dir, concept_ids, layer)

    # JSON keys must be strings; nested judgments[cid][alpha][sid]
    judgments_out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for cid, by_alpha in data.items():
        judgments_out[str(cid)] = {}
        for alpha, sid_map in by_alpha.items():
            judgments_out[str(cid)][str(alpha)] = {
                str(sid): int(j) for sid, j in sid_map.items()
            }

    payload: Dict[str, Any] = {
        "format_version": JUDGMENT_CACHE_VERSION,
        "layer": layer,
        "steering_method": steering_method,
        "llm": llm,
        "result_dir": os.path.abspath(result_dir),
        "alphas": alphas,
        "judgments": judgments_out,
    }

    if include_token_stats and tokenizer is not None:
        lengths: List[int] = []
        for cid in concept_ids:
            cdir = os.path.join(result_dir, str(cid))
            if not os.path.isdir(cdir):
                continue
            for fname in sorted(os.listdir(cdir)):
                if not fname.endswith(".json"):
                    continue
                parts = fname.replace(".json", "").split("-")
                if len(parts) < 2:
                    continue
                try:
                    file_layer = int(parts[0])
                    file_alpha = round(float(parts[1]), 2)
                except ValueError:
                    continue
                if file_layer != layer or file_alpha == 0.0:
                    continue
                fpath = os.path.join(cdir, fname)
                with open(fpath) as fp:
                    result_data = json.load(fp)
                for item in result_data["results"]:
                    resp = item.get("response_after", "")
                    if resp:
                        toks = tokenizer.encode(resp, add_special_tokens=False)
                        lengths.append(len(toks))
        if lengths:
            payload["avg_response_tokens"] = float(np.mean(lengths))
            payload["response_token_count"] = len(lengths)
        else:
            payload["avg_response_tokens"] = None
            payload["response_token_count"] = 0

    return payload


def load_judgments_from_cache_file(
    cache_path: str,
    concept_ids: Iterable[int],
    layer: int,
    steering_method: Optional[str] = None,
    llm: Optional[str] = None,
) -> Tuple[Dict[int, Dict[float, Dict[int, int]]], List[float], Dict[str, Any]]:
    """
    Load judgments from a JSON file written by build_judgment_cache_payload
    or script/cache_judgments.py.

    Filters to the requested concept_ids. Validates layer and optionally
    steering_method / llm when provided.

    Returns:
        data, alphas, extras (e.g. avg_response_tokens if present)
    """
    with open(cache_path) as f:
        blob = json.load(f)

    if blob.get("format_version") != JUDGMENT_CACHE_VERSION:
        raise ValueError(
            f"Unsupported judgment cache version: {blob.get('format_version')}"
        )
    if int(blob["layer"]) != int(layer):
        raise ValueError(
            f"Cache layer {blob['layer']} != requested layer {layer}"
        )
    if steering_method is not None and blob.get("steering_method") != steering_method:
        raise ValueError(
            f"Cache steering_method {blob.get('steering_method')} != {steering_method}"
        )
    if llm is not None and blob.get("llm") != llm:
        raise ValueError(
            f"Cache llm {blob.get('llm')} != {llm}"
        )

    alphas = [float(a) for a in blob["alphas"]]
    raw_j = blob["judgments"]
    want = set(int(x) for x in concept_ids)
    data: Dict[int, Dict[float, Dict[int, int]]] = {c: {} for c in want}

    for cid_str, by_alpha in raw_j.items():
        cid = int(cid_str)
        if cid not in want:
            continue
        for alpha_str, sid_map in by_alpha.items():
            alpha = round(float(alpha_str), 2)
            data[cid][alpha] = {
                int(sid): int(j) for sid, j in sid_map.items()
            }

    extras: Dict[str, Any] = {}
    if "avg_response_tokens" in blob:
        extras["avg_response_tokens"] = blob["avg_response_tokens"]
    if "response_token_count" in blob:
        extras["response_token_count"] = blob["response_token_count"]

    return data, alphas, extras


def load_paired_data(
    method,
    model,
    layer=None,
    alpha=None,
    n_list=None,
    k_list=None,
    alpha_min=None,
    alpha_max=None,
    concept_ids=None,
    force_regenerate_raw=False,
    return_metadata=False,
    exclude_baseline_success=False,
    batch_size_raw=1,
):
    """
    Load both steered and raw hidden states along with labels.
    If raw states are missing, extract them on the fly.
    
    Args:
        method: steering method
        model: model name
        layer: steering layer
        alpha: alpha value
        n_list: list of token offsets (target_token_n) to ensure are present/generated
        k_list: list of layer offsets (after_l_layers) to ensure are present/generated
        alpha_min: minimum alpha to include (inclusive)
        alpha_max: maximum alpha to include (inclusive)
        concept_ids: set/list of concept IDs to include (None = all)
        force_regenerate_raw: if True, regenerate raw hidden states even if cached files exist
        return_metadata: if True, also return list of (concept_id, alpha, sample_id) per sample
    """
    hidden_states_dir = f'data/hidden_states/{method}/{model}'
    raw_states_dir = f'data/hidden_states_raw/{method}/{model}'
    result_dir = f'data/result/{method}/{model}'

    pairs = [] # List of tuples (steered_sample, raw_sample)
    result_labels = []
    metadata_list = []  # (concept_id, alpha, sample_id) per sample
    
    if not os.path.exists(hidden_states_dir):
        print(f"Directory not found: {hidden_states_dir}")
        if return_metadata:
            return [], [], []
        return [], []
        
    # Model/Tokenizer lazy loading
    model_obj = None
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize lists
    if n_list is None:
        n_list = [5]
    if k_list is None:
        k_list = [5]

    allowed_concept_ids = set(concept_ids) if concept_ids is not None else None

    baseline_exclude = set()
    if exclude_baseline_success and os.path.exists(result_dir):
        for _cid_dir in os.listdir(result_dir):
            _cdir = os.path.join(result_dir, _cid_dir)
            if not os.path.isdir(_cdir):
                continue
            try:
                _cid = int(_cid_dir)
            except ValueError:
                continue
            if allowed_concept_ids is not None and _cid not in allowed_concept_ids:
                continue
            for _fn in os.listdir(_cdir):
                if not _fn.endswith('.json'):
                    continue
                _parts = _fn.replace('.json', '').split('-')
                if len(_parts) < 2:
                    continue
                try:
                    if float(_parts[1]) != 0.0:
                        continue
                except ValueError:
                    continue
                try:
                    with open(os.path.join(_cdir, _fn)) as _f:
                        _bl = json.load(_f)
                    for _item in _bl.get('results', []):
                        if _item.get('judgment') == 1:
                            baseline_exclude.add((_cid, _item.get('sample_id', -1)))
                except Exception:
                    continue
        if baseline_exclude:
            print(f"Baseline-success exclusion: {len(baseline_exclude)} (concept, sample) pairs")

    for concept_id in os.listdir(hidden_states_dir):
        concept_path = os.path.join(hidden_states_dir, concept_id)
        if not os.path.isdir(concept_path):
            continue
        
        if allowed_concept_ids is not None:
            try:
                if int(concept_id) not in allowed_concept_ids:
                    continue
            except ValueError:
                continue
            
        for file in sorted(os.listdir(concept_path)):
            if not file.endswith('.pt'):
                continue
                
            # Filter by layer/alpha if specified
            try:
                file_parts = file.split('-')
                if len(file_parts) >= 2:
                    file_layer = int(file_parts[0])
                    file_alpha = float(file_parts[1].replace('.pt', ''))
                    
                    if layer is not None and layer != file_layer:
                        continue
                    if alpha is not None and alpha != file_alpha:
                        continue
                        
                    if file_alpha == 0.0:
                        continue
                    # Range filtering
                    if alpha_min is not None and file_alpha < alpha_min:
                        continue
                    if alpha_max is not None and file_alpha > alpha_max:
                        continue
            except ValueError:
                continue
            
            # Paths
            steered_path = os.path.join(concept_path, file)
            raw_path = os.path.join(raw_states_dir, concept_id, file) # Same filename structure
            result_path = os.path.join(result_dir, concept_id, file.replace('.pt', '.json'))
            
            if not os.path.exists(result_path):
                print(f"Skipping {file}: Result file not found.")
                continue

            # Check if raw data exists and if it covers all requirements
            raw_data_exists = os.path.exists(raw_path)
            needs_generation = force_regenerate_raw or (not raw_data_exists)
                
            if needs_generation:
                if force_regenerate_raw and raw_data_exists:
                    print(f"Force-regenerating raw hidden states for {file}...")
                    os.remove(raw_path)
                else:
                    print(f"Raw hidden states missing for {file}. Generating...")
                
                # Load model if not loaded
                if model_obj is None:
                    print(f"Loading model {model} for extraction...")
                    model_to_model_name = {
                        "gemma-2-2b-it": "google/gemma-2-2b-it",
                        "Qwen3-4B": "Qwen/Qwen3-4B",
                        "Qwen3-8B": "Qwen/Qwen3-8B",
                        "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
                        "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",

                    }
                    model_name_path = model_to_model_name.get(model, model)
                    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
                    model_obj = AutoModelForCausalLM.from_pretrained(
                        model_name_path,
                        torch_dtype=torch.bfloat16,
                        device_map=device
                    )
                    model_obj.eval()
                
                # Determine layers to capture
                # We need layer L (curr_layer) and L+k for all k in k_list
                curr_layer = int(file.split('-')[0])
                layers_to_capture = {curr_layer}
                
                for k in k_list:
                    l_plus_k = curr_layer + k
                    # Check if layer exists in model (simple check against model config if available, or just try)
                    if l_plus_k < len(model_obj.model.layers):
                        layers_to_capture.add(l_plus_k)
                        
                # Ensure output directory exists
                os.makedirs(os.path.join(raw_states_dir, concept_id), exist_ok=True)
                
                # Verify process_file is available
                if 'process_file' not in globals():
                     print("Error: process_file not imported. Cannot generate raw states.")
                     continue
                
                base_raw_output_dir = os.path.dirname(os.path.dirname(os.path.dirname(raw_states_dir)))
                
                process_file(
                    result_file=result_path,
                    model=model_obj,
                    tokenizer=tokenizer,
                    device=device,
                    k_list=k_list,
                    target_token_n_list=n_list,
                    output_dir="data/hidden_states_raw",
                    layers_to_capture=layers_to_capture,
                    batch_size=batch_size_raw,
                )
                
                # Verify it was created
                if not os.path.exists(raw_path):
                    print(f"Raw hidden states already exists at {raw_path}")
                    continue
                
            try:
                # Load data
                # Use CPU to avoid OOM during loading, move to device later if needed
                steered_data = torch.load(steered_path, map_location='cpu')
                raw_data = torch.load(raw_path, map_location='cpu')
                result_json = json.load(open(result_path))
                result_list = result_json['results']
                
                steered_samples = steered_data['hidden_states']
                raw_samples = raw_data['hidden_states']
                
                # Iterate samples
                for i, result_item in enumerate(result_list):
                    sample_key = f'sample_{i}'
                    if sample_key not in steered_samples:
                        continue
                        
                    s_sample = steered_samples[sample_key]
                    
                    if sample_key not in raw_samples:
                        # Sometimes raw extraction might miss a sample if it errored?
                        continue
                        
                    r_sample = raw_samples[sample_key]
                    
                    # Check if we have required n
                    valid = True
                    for n in n_list:
                         if not check_hidden_states_data(s_sample, n):
                             valid = False
                             break
                    if not valid:
                        continue
                    
                    judgment = result_item.get('judgment')
                    if judgment is None:
                        continue

                    if exclude_baseline_success:
                        _sid = result_item.get('sample_id', i)
                        if (int(concept_id), _sid) in baseline_exclude:
                            continue

                    pairs.append((s_sample, r_sample))
                    result_labels.append(judgment)
                    metadata_list.append((int(concept_id), file_alpha, i))
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

    if return_metadata:
        return list(pairs), list(result_labels), metadata_list
    return list(pairs), list(result_labels)
