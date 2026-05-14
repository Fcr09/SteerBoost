#!/usr/bin/env python3
"""
Run steering pipeline on AlpacaEval dataset samples.

This script loads AlpacaEval data, applies steering to each sample,
and saves the results in a structured format.

Usage:
    # DiffMean on 10 samples
    python dataset_steering.py --method diffmean --concept "response with emojis" --num_samples 10 --alpha 5
    
    # LinearProbe on 20 samples
    python dataset_steering.py --method probe --concept "scientific language" --num_samples 20 --alpha 10
    
    # Custom AlpacaEval path
    python dataset_steering.py --method diffmean --alpaca_path ./my_alpaca.json --num_samples 5
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from datetime import datetime

from steering import LinearProbe, DiffMean
from pipeline import SteeringPipeline
from utils import load_alpaca_eval, save_results, save_hidden_states_dataset



def run_steering_on_dataset(
    pipeline: SteeringPipeline,
    samples: List[Dict[str, str]],
    alpha: float,
    max_tokens: int = 256,
    generate_unsteered: bool = False,
    collect_hidden_states: bool = False,
    k_list: list = [5],
    target_token_n_list: list = [5],
    batch_size: int = 8,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Run steering pipeline on all samples using batched generation.

    Both unsteered and steered generations are fully batched. When
    collect_hidden_states is True, batch-aware hooks capture
    [batch, hidden_dim] tensors that are split per-sample afterwards.

    Args:
        pipeline: Configured SteeringPipeline instance
        samples: List of AlpacaEval samples
        alpha: Steering strength
        max_tokens: Maximum tokens to generate
        generate_unsteered: Whether to run baseline (alpha=0.0) generation
        collect_hidden_states: Whether to collect hidden states for each sample
        k_list: List of offsets for L+k layers
        target_token_n_list: List of generated tokens to capture
        batch_size: Number of prompts per generation batch

    Returns:
        Tuple of (results, method_stats, hidden_states_data)
    """
    instructions = [s.get('instruction', '') for s in samples]
    datasets = [s.get('dataset', 'unknown') for s in samples]
    method = pipeline.method
    method.load_model()

    # -- Method statistics (available once the vector is computed) ----------
    method_stats = {}
    if hasattr(method, 'get_training_accuracy'):
        method_stats['classification_accuracy'] = method.get_training_accuracy()
    elif hasattr(method, 'get_steering_vector_norm'):
        method_stats['steering_vector_norm'] = method.get_steering_vector_norm()

    # -- Step 1: Optional batch unsteered generation -----------------------
    responses_before = [None] * len(instructions)
    if generate_unsteered:
        print(f"\n--- Unsteered generation ({len(instructions)} samples, batch_size={batch_size}) ---")
        responses_before = method.generate_batch(
            instructions, alpha=0.0, max_new_tokens=max_tokens, batch_size=batch_size
        )
    else:
        print("\n--- Unsteered generation skipped (use --generate_unsteered to enable) ---")

    # -- Step 2: Steered generation ----------------------------------------
    hs_label = " + hidden states" if collect_hidden_states else ""
    print(f"\n--- Steered generation{hs_label} ({len(instructions)} samples, batch_size={batch_size}) ---")
    responses_after = method.generate_batch(
        instructions, alpha=alpha, max_new_tokens=max_tokens, batch_size=batch_size,
        collect_hidden_states=collect_hidden_states,
        k_list=k_list, target_token_n_list=target_token_n_list,
    )
    hidden_states_data = method.get_batch_hidden_states() if collect_hidden_states else {}

    # -- Assemble results --------------------------------------------------
    results = []
    for idx in range(len(samples)):
        resp_before = responses_before[idx] if idx < len(responses_before) else None
        resp_after = responses_after[idx] if idx < len(responses_after) else None

        if resp_before is None and resp_after is None:
            results.append({
                'sample_id': idx,
                'instruction': instructions[idx],
                'dataset': datasets[idx],
                'error': 'Generation failed',
                'response_before': None,
                'response_after': None,
            })
        else:
            results.append({
                'sample_id': idx,
                'dataset': datasets[idx],
                'instruction': instructions[idx],
                'prompt': instructions[idx],
                'response_before': resp_before,
                'response_after': resp_after,
                'alpha': alpha,
            })

    successful = len([r for r in results if 'error' not in r])
    print(f"\n✓ Completed {successful}/{len(samples)} samples successfully")
    if collect_hidden_states:
        print(f"✓ Collected hidden states for {len(hidden_states_data)} samples")

    return results, method_stats, hidden_states_data



def main():
    parser = argparse.ArgumentParser(
        description="Run steering pipeline on AlpacaEval dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DiffMean on 10 samples
  python dataset_steering.py --method diffmean --concept "response with emojis" --num_samples 10
  
  # LinearProbe with custom parameters
  python dataset_steering.py --method probe --concept "scientific language" --num_samples 20 --epochs 15
  
  # Custom AlpacaEval path and output directory
  python dataset_steering.py --method diffmean --alpaca_path ./my_alpaca.json --output_dir ./results
  
  # Collect hidden states for analysis (single values)
  python dataset_steering.py --method diffmean --collect_hidden_states --k_list 5 --target_token_n_list 3 --num_samples 5
  
  # Collect hidden states for multiple k and target_token_n values
  python dataset_steering.py --method diffmean --collect_hidden_states --k_list "3,5,10" --target_token_n_list "1,3,5" --num_samples 5
        """
    )
    
    # Dataset parameters
    parser.add_argument(
        "--alpaca_path",
        type=str,
        default="data/alpaca_eval.json",
        help="Path to alpaca_eval.json (default: data/alpaca_eval.json)"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to process (default: 10)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    
    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        choices=["probe", "diffmean"],
        default="diffmean",
        help="Steering method to use (default: diffmean)"
    )
    
    # Steering parameters
    parser.add_argument(
        "--concept_file",
        type=str,
        default="data/concepts.json",
        help="Concept file with the 150 target concepts (default: 'data/concepts.json')"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: google/gemma-2-2b-it)"
    )
    
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Layer to extract activations from (default: 20)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Steering strength (default: 5.0)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 1024)"
    )
    
    parser.add_argument(
        "--num_examples",
        type=int,
        default=50,
        help="Number of training examples per category (default: 10)"
    )
    
    # LinearProbe specific
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for LinearProbe (default: 10)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate for LinearProbe (default: 0.005)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay for LinearProbe (default: 0.001)"
    )
    
    parser.add_argument(
        "--l2_lambda",
        type=float,
        default=0.0,
        help="L2 regularization on probe weights to discourage sparsity (default: 0.0)"
    )
    
    # DiffMean specific
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize DiffMean vector to unit norm"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/result",
        help="Base output directory (default: data/result)"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory for training data (default: data)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    
    # Hidden state collection arguments
    parser.add_argument(
        "--collect_hidden_states",
        action="store_true",
        help="Collect hidden states for analysis during generation"
    )
    
    parser.add_argument(
        "--k_list",
        type=str,
        default="5",
        help="Comma-separated list of offsets for L+k layers (default: 5)"
    )
    
    parser.add_argument(
        "--target_token_n_list",
        type=str,
        default="5",
        help="Comma-separated list of generated tokens to capture (default: 5)"
    )
    
    parser.add_argument(
        "--hidden_states_dir",
        type=str,
        default="data/hidden_states",
        help="Directory to save hidden states (default: data/hidden_states)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)"
    )

    parser.add_argument(
        "--generate_unsteered",
        action="store_true",
        help="Also run unsteered generation (alpha=0.0). Disabled by default."
    )
    
    args = parser.parse_args()
    
    try:
        # Load AlpacaEval data
        samples = load_alpaca_eval(args.alpaca_path, args.num_samples, args.seed)

        with open(args.concept_file, "r") as f:
            concepts = json.load(f)

        # Create method once so the model is loaded only once across all concepts
        if args.method == "probe":
            method = LinearProbe(
                model_name=args.model,
                layer=args.layer,
                device=args.device,
                learning_rate=args.lr,
                num_epochs=args.epochs,
                weight_decay=args.weight_decay,
                l2_lambda=args.l2_lambda,
            )
        else:  # diffmean
            method = DiffMean(
                model_name=args.model,
                layer=args.layer,
                device=args.device,
                normalize=args.normalize,
            )
        method.load_model()

        # Build method directory name (e.g. "probe_l2-0.01" when l2_lambda is set)
        method_dir = args.method
        if args.method == "probe" and args.l2_lambda != 0:
            method_dir = f"probe_l2-{args.l2_lambda}"

        # Parse comma-separated lists once
        k_list = [int(x.strip()) for x in args.k_list.split(',')]
        target_token_n_list = [int(x.strip()) for x in args.target_token_n_list.split(',')]

        for item in concepts:
            concept_name = item["concept_name"]
            concept_id = item["concept_id"]
            model_short = args.model.split('/')[-1]
            if os.path.exists(os.path.join(args.output_dir, method_dir, model_short, str(concept_id), f"{args.layer}-{args.alpha}.json")):
                print(f"Concept: {concept_name} (ID: {concept_id}) already processed")
                continue
            print(f"\nConcept: {concept_name} (ID: {concept_id}) Alpha: {args.alpha}")

            try:
                # Reset per-concept state while keeping the loaded model
                method.steering_vector = None
                method.stored_hidden_states = {}
                method.captured_tokens = set()
                method.hook_call_count = 0
                method.current_generation_step = 0
                method.generation_step = 0
                method.collect_hidden_states = False
                if hasattr(method, "probe"):
                    method.probe = None
                    method.training_stats = {}
                if hasattr(method, "mean_positive"):
                    method.mean_positive = None
                    method.mean_negative = None

                pipeline = SteeringPipeline(
                    method=method,
                    concept=concept_name,
                    concept_id=concept_id,
                    data_dir=args.data_dir,
                )

                pos, neg = pipeline.generate_examples(
                    num_examples=args.num_examples,
                    gpt_model="gpt-4o-mini",
                )
                if pos is None or neg is None:
                    print(f"⚠ Skipping concept {concept_name} (ID: {concept_id}): example generation failed")
                    continue

                vector = pipeline.compute_vector()
                print(f"✓ Steering vector computed")

                results, method_stats, hidden_states_data = run_steering_on_dataset(
                    pipeline=pipeline,
                    samples=samples,
                    alpha=args.alpha,
                    max_tokens=args.max_tokens,
                    generate_unsteered=args.generate_unsteered,
                    collect_hidden_states=args.collect_hidden_states,
                    k_list=k_list,
                    target_token_n_list=target_token_n_list,
                    batch_size=args.batch_size,
                )

                output_path = save_results(
                    results=results,
                    method=method_dir,
                    model_name=args.model,
                    layer=args.layer,
                    alpha=args.alpha,
                    concept_id=concept_id,
                    method_stats=method_stats,
                    output_dir=args.output_dir,
                )

                if args.collect_hidden_states and hidden_states_data:
                    hidden_states_path = save_hidden_states_dataset(
                        hidden_states_data=hidden_states_data,
                        method=method_dir,
                        model_name=args.model,
                        concept_id=concept_id,
                        layer=args.layer,
                        alpha=args.alpha,
                        k_list=k_list,
                        target_token_n_list=target_token_n_list,
                        output_dir=args.hidden_states_dir,
                    )
                    print(f"✓ Hidden states saved")

            except Exception as e:
                print(f"\n⚠ Error processing concept {concept_name} (ID: {concept_id}): {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\nSummary: {args.method} | {args.model} | Layer {args.layer} | Alpha {args.alpha}")
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("Make sure the alpaca_eval.json path is correct.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

