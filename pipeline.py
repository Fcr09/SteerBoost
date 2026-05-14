#!/usr/bin/env python3
"""
Unified steering pipeline that works with any steering method.
"""

import torch
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import os
from steering import SteeringMethod, LinearProbe, DiffMean
from utils import generate_examples_with_gpt


class SteeringPipeline:
    """
    Unified pipeline for steering workflows.
    
    Supports both LinearProbe and DiffMean methods with a consistent interface.
    """
    
    def __init__(
        self,
        method: SteeringMethod,
        concept: str,
        concept_id: int,
        data_dir: str,
        positive_examples: Optional[List[str]] = None,
        negative_examples: Optional[List[str]] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            method: Steering method instance (LinearProbe or DiffMean)
            concept: The concept to steer towards
            data_dir: The directory to save the data
            positive_examples: Optional pre-defined positive examples
            negative_examples: Optional pre-defined negative examples
        """
        self.method = method
        self.concept = concept
        self.concept_id = concept_id
        self.data_dir = data_dir
        self.positive_examples = positive_examples or []
        self.negative_examples = negative_examples or []
        self.vector_computed = False
    
    def generate_examples(
        self,
        num_examples: int = 10,
        gpt_model: str = "gpt-4o-mini"
    ) -> tuple:
        """
        Generate training examples using GPT.
        
        Args:
            num_examples: Number of examples per category
            gpt_model: OpenAI model to use
            
        Returns:
            Tuple of (positive_examples, negative_examples)
        """
        print(f"\nGenerating examples for concept: {self.concept} ({num_examples} per category)")
        
        pos, neg = generate_examples_with_gpt(
            self.concept,
            self.concept_id,
            num_examples=num_examples,
            model=gpt_model,
            data_dir=self.data_dir
        )
        
        if pos and neg:
            self.positive_examples = pos
            self.negative_examples = neg
            print(f"\n✓ Generated {len(pos)} positive and {len(neg)} negative examples")
            return pos, neg
        else:
            print("\n⚠ Failed to generate examples")
            return None, None
    
    def use_examples(
        self,
        positive_examples: List[str],
        negative_examples: List[str]
    ):
        """Set examples manually."""
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        print(f"✓ Using {len(positive_examples)} positive and {len(negative_examples)} negative examples")
    
    def compute_vector(self) -> torch.Tensor:
        """Compute the steering vector using the configured method."""
        if not self.positive_examples or not self.negative_examples:
            raise ValueError("No examples provided. Call generate_examples() or use_examples() first.")
        
        vector = self.method.compute_steering_vector(
            self.positive_examples,
            self.negative_examples
        )
        self.vector_computed = True
        return vector
    
    def test_steering(
        self,
        test_prompt: str,
        alpha: float = 5.0,
        show_comparison: bool = True,
        max_new_tokens: int = 20480,
        collect_hidden_states: bool = False,
        k_list: list = [5],
        target_token_n_list: list = [5]
    ) -> Dict[str, Any]:
        """
        Test steering on a prompt.
        
        Args:
            test_prompt: Prompt to test
            alpha: Steering strength
            show_comparison: Whether to print comparison
            max_new_tokens: Maximum tokens to generate
            collect_hidden_states: Whether to collect hidden states for analysis
            k_list: List of offsets for L+k layers
            target_token_n_list: List of generated tokens to capture
            
        Returns:
            Dictionary with results
        """
        if not self.vector_computed:
            raise ValueError("Steering vector not computed. Call compute_vector() first.")
        
        if show_comparison:
            print(f"\nTesting steering (alpha={alpha})")
            print(f"Prompt: {test_prompt[:100]}...")
        
        # Generate without steering
        if show_comparison:
            print(f"\nWithout steering:")
        
        response_before = self.method.generate(
            test_prompt,
            alpha=0.0,
            max_new_tokens=max_new_tokens
        )
        
        if show_comparison:
            print(f"\nResponse:")
            print("-" * 80)
            print(response_before)
            print("-" * 80)
        
        # Generate with steering
        if show_comparison:
            print(f"\n{'='*80}")
            print("WITH STEERING")
            print(f"{'='*80}")
        
        response_after = self.method.generate(
            test_prompt,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            collect_hidden_states=collect_hidden_states,
            k_list=k_list,
            target_token_n_list=target_token_n_list,
            clear_after_generation=False  # Don't clear if we might save later
        )
        
        if show_comparison:
            print(f"\nResponse: {response_after[:200]}...")
        
        # Get method-specific statistics
        method_stats = {}
        if hasattr(self.method, 'get_training_accuracy'):
            # LinearProbe
            method_stats['classification_accuracy'] = self.method.get_training_accuracy()
        elif hasattr(self.method, 'get_steering_vector_norm'):
            # DiffMean
            method_stats['steering_vector_norm'] = self.method.get_steering_vector_norm()
        
        results = {
            "prompt": test_prompt,
            "response_before": response_before,
            "response_after": response_after,
            "alpha": alpha,
            "method_stats": method_stats
        }
        
        # Add hidden states if collected
        if collect_hidden_states:
            results["hidden_states"] = self.method.get_collected_hidden_states()
        
        if show_comparison:
            print(f"\n{'='*80}")
            print("METHOD STATISTICS")
            print(f"{'='*80}")
            for key, value in method_stats.items():
                print(f"{key}: {value:.4f}")
        
        return results
    
    def run_full_pipeline(
        self,
        test_prompt: str,
        num_examples: int = 10,
        alpha: float = 5.0,
        save_dir: Optional[str] = None,
        collect_hidden_states: bool = False,
        k_list: list = [5],
        target_token_n_list: list = [5],
        save_hidden_states_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.
        
        Args:
            test_prompt: Prompt to test steering on
            num_examples: Number of examples to generate
            alpha: Steering strength
            save_dir: Optional directory to save steering vector
            collect_hidden_states: Whether to collect hidden states for analysis
            k_list: List of offsets for L+k layers
            target_token_n_list: List of generated tokens to capture
            save_hidden_states_path: Optional path to save collected hidden states
            
        Returns:
            Dictionary with all results
        """
        print(f"\nSteering Pipeline: {self.method.__class__.__name__} | {self.concept} | Layer {self.method.layer}")
        
        # Step 1: Get examples

        self.generate_examples(num_examples=num_examples)
        
        # Step 2: Compute steering vector
        self.compute_vector()
        
        # Step 3: Test steering
        results = self.test_steering(
            test_prompt, 
            alpha=alpha, 
            show_comparison=True,
            collect_hidden_states=collect_hidden_states,
            k_list=k_list,
            target_token_n_list=target_token_n_list
        )
        
        # Step 4: Save if requested
        if save_dir:
            metadata = {
                "concept": self.concept,
                "num_positive_examples": len(self.positive_examples),
                "num_negative_examples": len(self.negative_examples),
                "test_prompt": test_prompt,
                "alpha": alpha,
            }
            
            # Add method-specific metadata
            if isinstance(self.method, LinearProbe):
                metadata.update(self.method.training_stats)
            
            self.method.save_vector(save_dir, metadata)
        
        # Step 5: Save hidden states if requested
        if collect_hidden_states and save_hidden_states_path:
            self.method.save_hidden_states(save_hidden_states_path)
            # Clear the states after saving
            self.method.clear_hidden_state_collection()
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETE")
        print(f"{'='*80}")
        
        return results
    
    def compare_responses(self, results: Dict[str, Any]):
        """Print a side-by-side comparison of responses."""
        print(f"\nResponse Comparison:")
        print(f"Without steering: {results['response_before'][:100]}...")
        print(f"With steering: {results['response_after'][:100]}...")

