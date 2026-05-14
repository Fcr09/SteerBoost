#!/usr/bin/env python3
"""
Steering methods for language model activation steering.

This module provides classes for different steering methods:
- LinearProbe: Train a binary classifier and use its weights as steering vector
- DiffMean: Compute difference of means between positive and negative examples
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import gather_residual_activations, generate_examples_with_gpt


class SteeringMethod(ABC):
    """Base class for steering methods."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        layer: int = 20,
        device: str = None
    ):
        """
        Initialize steering method.
        
        Args:
            model_name: HuggingFace model name
            layer: Layer to extract activations from
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.steering_vector = None
        
        # Hidden state storage for analysis
        self.stored_hidden_states = {}
        self.collect_hidden_states = False
        self.generation_step = 0
        self.captured_tokens = set()  # Track which tokens we've already captured
        self.current_generation_step = 0  # Track current generation step
        self.hook_call_count = 0  # Track how many times the hook is called
        
    def load_model(self):
        """Load the language model and tokenizer."""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"✓ Model loaded on {self.device}")
    
    @abstractmethod
    def compute_steering_vector(
        self,
        positive_examples: List[str],
        negative_examples: List[str]
    ) -> torch.Tensor:
        """
        Compute the steering vector from examples.
        
        Args:
            positive_examples: List of positive example texts
            negative_examples: List of negative example texts
            
        Returns:
            Steering vector tensor
        """
        pass
    
    def extract_hidden_states(
        self,
        examples: List[str],
        average: bool = True
    ) -> List[torch.Tensor]:
        """
        Extract hidden states from examples.
        
        Args:
            examples: List of text examples
            average: Whether to average over sequence length
            
        Returns:
            List of hidden state tensors
        """
        self.load_model()
        hidden_states = []
        
        self.model.eval()
        with torch.no_grad():
            for text in examples:
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    add_special_tokens=True
                ).to(self.device)
                
                hidden = gather_residual_activations(self.model, self.layer, inputs)
                
                if average:
                    if hidden.dim() == 3:
                        hidden_states.append(hidden[0, 1:, :].mean(dim=0))
                    else:
                        hidden_states.append(hidden[1:, :].mean(dim=0))
                else:
                    if hidden.dim() == 3:
                        hidden_states.append(hidden[0, 1:, :])
                    else:
                        hidden_states.append(hidden[1:, :])
        
        return hidden_states
    
    def setup_hidden_state_collection(self, k_list: list = [5], target_token_n_list: list = [5]):
        """
        Setup hidden state collection for analysis with multiple k and target_token_n values.
        
        Args:
            k_list: List of offsets for L+k layers (default: [5])
            target_token_n_list: List of generated tokens to capture (default: [5])
        """
        self.collect_hidden_states = True
        self.k_list = k_list if isinstance(k_list, list) else [k_list]
        self.target_token_n_list = target_token_n_list if isinstance(target_token_n_list, list) else [target_token_n_list]
        
        # Initialize storage structure for all token positions
        self.stored_hidden_states = {
            'token_0': {
                'before_steering': {},
                'after_steering': {}
            }
        }
        
        # Add storage for each target token
        for target_token_n in self.target_token_n_list:
            self.stored_hidden_states[f'token_{target_token_n}'] = {
                'before_steering': {},
                'after_steering': {}
            }
        
        self.token_count = 0
        self.prompt_length = 0
        
    def clear_hidden_state_collection(self):
        """Clear stored hidden states and disable collection."""
        self.collect_hidden_states = False
        self.stored_hidden_states = {}
        self.token_count = 0
        self.prompt_length = 0
        self.generation_step = 0
        self.current_generation_step = 0
        self.hook_call_count = 0
        self.captured_tokens = set()
    
    def apply_steering_hook(self, alpha: float = 1.0):
        """
        Create a hook that applies steering to the model.
        
        Args:
            alpha: Steering strength
            
        Returns:
            Hook function and handle
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            self.hook_call_count += 1
            
            # Collect hidden states for both prompt processing and generation
            if self.collect_hidden_states:
                if self.hook_call_count == 1:
                    # Hook call #1: Prompt processing - capture as token_0 (hidden state used to generate first token)
                    self.current_generation_step = 0
                    self._store_hidden_state_before_steering(hidden_states)
                elif self.hook_call_count > 1:
                    # Hook call #2+: Generation phase
                    # Hook call #2 = generation step 1 (first generated token)
                    # Hook call #3 = generation step 2 (second generated token), etc.
                    self.current_generation_step = self.hook_call_count - 1
                    self._store_hidden_state_before_steering(hidden_states)
            
            # Convert steering vector to same dtype and device
            steering = self.steering_vector.to(
                device=hidden_states.device, 
                dtype=hidden_states.dtype
            )
            steered = hidden_states + alpha * steering
            
            # Store hidden states after steering if collection is enabled
            if self.collect_hidden_states:
                self._store_hidden_state_after_steering(steered)
            
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            else:
                return steered
        
        handle = self.model.model.layers[self.layer].register_forward_hook(hook)
        return hook, handle
    
    def _get_last_token_hidden(self, hidden_states):
        """Extract last token hidden state robustly (handles 2D and 3D tensors)."""
        if hidden_states.dim() == 3:
            return hidden_states[0, -1, :]
        elif hidden_states.dim() == 2:
            return hidden_states[-1, :]
        else:
            raise ValueError(f"Unexpected hidden states dimension: {hidden_states.shape}")

    def _store_hidden_state_before_steering(self, hidden_states):
        """Store hidden states before steering is applied."""
        if not self.collect_hidden_states:
            return
            
        # Use hook call count to determine token position (0-indexed)
        # Hook call #1 = token 0 (prompt processing), hook call #2 = token 1, hook call #3 = token 2, etc.
        current_generated_token = self.hook_call_count - 1
        
        # Check if this is the first generated token (index 0)
        if current_generated_token == 0:
            # Create unique key for token_0
            token_key = f"token_0_layer_{self.layer}_before"
            if token_key not in self.captured_tokens:
                self.stored_hidden_states['token_0']['before_steering'][f'layer_{self.layer}'] = self._get_last_token_hidden(hidden_states).clone()
                # print(f"Captured first token before steering at layer {self.layer}")
                self.captured_tokens.add(token_key)
                
        # Check if this is one of the target generated tokens
        # current_generated_token directly corresponds to the token_N we want to capture
        if current_generated_token in self.target_token_n_list:
            token_key_name = f'token_{current_generated_token}'
            # Create unique key for this specific token
            token_key = f"{token_key_name}_layer_{self.layer}_before"
            if token_key not in self.captured_tokens:
                self.stored_hidden_states[token_key_name]['before_steering'][f'layer_{self.layer}'] = self._get_last_token_hidden(hidden_states).clone()
                # print(f"Captured token {current_generated_token} before steering at layer {self.layer}")
                self.captured_tokens.add(token_key)
    
    def _store_hidden_state_after_steering(self, steered_hidden_states):
        """Store hidden states after steering is applied."""
        if not self.collect_hidden_states:
            return
            
        # Use hook call count to determine token position (0-indexed)
        # Hook call #1 = token 0 (prompt processing), hook call #2 = token 1, hook call #3 = token 2, etc.
        current_generated_token = self.hook_call_count - 1
        
        # Check if this is the first generated token (index 0)
        if current_generated_token == 0:
            # Create unique key for token_0
            token_key = f"token_0_layer_{self.layer}_after"
            if token_key not in self.captured_tokens:
                self.stored_hidden_states['token_0']['after_steering'][f'layer_{self.layer}'] = self._get_last_token_hidden(steered_hidden_states).clone()
                # print(f"Captured first token after steering at layer {self.layer}")
                self.captured_tokens.add(token_key)
                
        # Check if this is one of the target generated tokens
        # current_generated_token directly corresponds to the token_N we want to capture
        if current_generated_token in self.target_token_n_list:
            token_key_name = f'token_{current_generated_token}'
            # Create unique key for this specific token
            token_key = f"{token_key_name}_layer_{self.layer}_after"
            if token_key not in self.captured_tokens:
                self.stored_hidden_states[token_key_name]['after_steering'][f'layer_{self.layer}'] = self._get_last_token_hidden(steered_hidden_states).clone()
                # print(f"Captured token {current_generated_token} after steering at layer {self.layer}")
                self.captured_tokens.add(token_key)
    
    def _setup_additional_layer_hooks(self):
        """
        Setup hooks for L+k layer to capture hidden states there too.
        
        Note: L+k layers only have 'before_steering' states because steering
        is only applied at layer L. The L+k layers receive the steered output
        from layer L, but no additional steering is applied at L+k.
        """
        if not self.collect_hidden_states:
            return []
            
        handles = []
        
        def create_hook_for_layer(layer_idx, layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Collect for both prompt processing and generation phase
                # Use hook call count to determine token position (0-indexed)
                # Hook call #1 = token 0 (prompt processing), hook call #2 = token 1, hook call #3 = token 2, etc.
                if self.hook_call_count >= 1:
                    current_generated_token = self.hook_call_count - 1
                    
                    # Store for first token (only once) - L+k layers only have "before_steering" (no steering applied)
                    if current_generated_token == 0:
                        before_key = f"token_0_{layer_name}_before"
                        if before_key not in self.captured_tokens:
                            self.stored_hidden_states['token_0']['before_steering'][layer_name] = self._get_last_token_hidden(hidden_states).clone()
                            # print(f"Captured first token at {layer_name}")
                            self.captured_tokens.add(before_key)
                        
                    # Store for N-th tokens (only once) - L+k layers only have "before_steering" (no steering applied)
                    # current_generated_token directly corresponds to the token_N we want to capture
                    if current_generated_token in self.target_token_n_list:
                        token_key = f'token_{current_generated_token}'
                        before_key = f"{token_key}_{layer_name}_before"
                        if before_key not in self.captured_tokens:
                            self.stored_hidden_states[token_key]['before_steering'][layer_name] = self._get_last_token_hidden(hidden_states).clone()
                            # print(f"Captured token {current_generated_token} at {layer_name}")
                            self.captured_tokens.add(before_key)
                    
                return output
            return hook
        
        # Setup hooks for all k values
        for k in self.k_list:
            l_plus_k = min(self.layer + k, len(self.model.model.layers) - 1)
            if l_plus_k == self.layer:
                continue  # Skip if no additional layer needed
                
            layer_name = f"layer_{l_plus_k}"
            hook = create_hook_for_layer(l_plus_k, layer_name)
            handle = self.model.model.layers[l_plus_k].register_forward_hook(hook)
            handles.append(handle)
        
        return handles
    
    def generate(
        self,
        prompt: str,
        alpha: float = 0.0,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        collect_hidden_states: bool = False,
        k_list: list = [5],
        target_token_n_list: list = [5],
        clear_after_generation: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with optional steering and hidden state collection.
        
        Args:
            prompt: Input prompt
            alpha: Steering strength (0 = no steering)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            collect_hidden_states: Whether to collect hidden states for analysis
            k_list: List of offsets for L+k layers
            target_token_n_list: List of generated tokens to capture
            clear_after_generation: Whether to clear collected states after generation
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        self.load_model()

        is_qwen3 = "qwen3" in self.model_name.lower()
        use_chat_template = hasattr(self.tokenizer, "apply_chat_template")
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            chat_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }
            # Qwen3 models default to thinking mode; disable it explicitly.
            if is_qwen3:
                chat_kwargs["enable_thinking"] = False

            inputs = self.tokenizer.apply_chat_template(messages, **chat_kwargs)
            if isinstance(inputs, torch.Tensor):
                inputs = {"input_ids": inputs}
            else:
                inputs = dict(inputs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)

        # Some chat-template paths return only input_ids. Always provide mask
        # to avoid generation ambiguity when pad_token_id == eos_token_id.
        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        prompt_length = inputs['input_ids'].shape[1]
        self.prompt_length = prompt_length
        self.current_generation_step = 0  # Reset generation step
        self.hook_call_count = 0  # Reset hook call count
        
        if collect_hidden_states:
            print(f"Collecting hidden states (k_list={k_list}, target_token_n_list={target_token_n_list})")
        
        # Setup hidden state collection if requested
        additional_handles = []
        if collect_hidden_states:
            self.setup_hidden_state_collection(k_list=k_list, target_token_n_list=target_token_n_list)
            additional_handles = self._setup_additional_layer_hooks()
        
        # Apply steering if alpha > 0
        handle = None
        if alpha != 0.0 and self.steering_vector is not None:
            _, handle = self.apply_steering_hook(alpha)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
        finally:
            # Remove all hooks
            if handle:
                handle.remove()
            for h in additional_handles:
                h.remove()
            
            # Clear hidden state collection if requested
            if collect_hidden_states and clear_after_generation:
                self.clear_hidden_state_collection()
        
        generated_ids = outputs[0][prompt_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._clean_response(response)

    def generate_batch(
        self,
        prompts: List[str],
        alpha: float = 0.0,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        batch_size: int = 8,
        collect_hidden_states: bool = False,
        k_list: list = None,
        target_token_n_list: list = None,
        **kwargs
    ) -> List[str]:
        """
        Batch generate text with optional steering and hidden state collection.

        Processes prompts in mini-batches with left-padding. When
        collect_hidden_states is True, batch-aware hooks capture
        ``[batch, hidden_dim]`` tensors that are split per-sample afterwards.
        Collected states are available via :meth:`get_batch_hidden_states`.

        Args:
            prompts: List of input prompts
            alpha: Steering strength (0 = no steering)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            batch_size: Number of prompts per mini-batch
            collect_hidden_states: Whether to collect hidden states
            k_list: Layer offsets for L+k capture (default: [5])
            target_token_n_list: Generation steps to capture (default: [5])
            **kwargs: Additional generation arguments

        Returns:
            List of generated texts, one per prompt
        """
        self.load_model()

        k_list = k_list if k_list is not None else [5]
        target_token_n_list = target_token_n_list if target_token_n_list is not None else [5]

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        is_qwen3 = "qwen3" in self.model_name.lower()
        use_chat_template = hasattr(self.tokenizer, "apply_chat_template")

        all_responses = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        self._batch_hidden_states = {}

        try:
            for batch_start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[batch_start:batch_start + batch_size]
                batch_num = batch_start // batch_size + 1

                # ---- Tokenize ------------------------------------------------
                if use_chat_template:
                    batch_ids = []
                    for prompt in batch_prompts:
                        messages = [{"role": "user", "content": prompt}]
                        chat_kwargs = {
                            "tokenize": True,
                            "add_generation_prompt": True,
                            "return_tensors": "pt",
                        }
                        if is_qwen3:
                            chat_kwargs["enable_thinking"] = False

                        ids = self.tokenizer.apply_chat_template(messages, **chat_kwargs)
                        if isinstance(ids, torch.Tensor):
                            batch_ids.append(ids.squeeze(0))
                        else:
                            batch_ids.append(dict(ids)["input_ids"].squeeze(0))

                    max_len = max(ids.shape[0] for ids in batch_ids)
                    pad_id = self.tokenizer.pad_token_id
                    padded_ids = []
                    masks = []
                    for ids in batch_ids:
                        pad_len = max_len - ids.shape[0]
                        padded_ids.append(torch.cat([
                            torch.full((pad_len,), pad_id, dtype=ids.dtype),
                            ids,
                        ]))
                        masks.append(torch.cat([
                            torch.zeros(pad_len, dtype=torch.long),
                            torch.ones(ids.shape[0], dtype=torch.long),
                        ]))

                    inputs = {
                        "input_ids": torch.stack(padded_ids).to(self.device),
                        "attention_mask": torch.stack(masks).to(self.device),
                    }
                else:
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=True,
                    ).to(self.device)
                    if "attention_mask" not in inputs:
                        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

                input_length = inputs["input_ids"].shape[1]

                # ---- Setup hooks ---------------------------------------------
                handles = []
                mini_batch_states = None

                if collect_hidden_states:
                    step_counter = [0]
                    captured = set()
                    apply_steer = alpha != 0.0 and self.steering_vector is not None

                    mini_batch_states = {
                        'token_0': {'before_steering': {}, 'after_steering': {}}
                    }
                    for n in target_token_n_list:
                        mini_batch_states[f'token_{n}'] = {
                            'before_steering': {}, 'after_steering': {}
                        }

                    def _make_steering_hook(a, li, st, ctr, cap, ttn, do_steer):
                        def hook(module, inp, output):
                            hs = output[0] if isinstance(output, tuple) else output
                            ctr[0] += 1
                            step = ctr[0] - 1
                            last = hs[:, -1, :].clone()

                            if step == 0:
                                k = f"t0_L{li}_b"
                                if k not in cap:
                                    st['token_0']['before_steering'][f'layer_{li}'] = last
                                    cap.add(k)
                            for n in ttn:
                                if step == n:
                                    k = f"t{n}_L{li}_b"
                                    if k not in cap:
                                        st[f'token_{n}']['before_steering'][f'layer_{li}'] = last
                                        cap.add(k)

                            if do_steer:
                                sv = self.steering_vector.to(
                                    device=hs.device, dtype=hs.dtype
                                )
                                steered = hs + a * sv
                            else:
                                steered = hs

                            steered_last = steered[:, -1, :].clone()
                            if step == 0:
                                k = f"t0_L{li}_a"
                                if k not in cap:
                                    st['token_0']['after_steering'][f'layer_{li}'] = steered_last
                                    cap.add(k)
                            for n in ttn:
                                if step == n:
                                    k = f"t{n}_L{li}_a"
                                    if k not in cap:
                                        st[f'token_{n}']['after_steering'][f'layer_{li}'] = steered_last
                                        cap.add(k)

                            if isinstance(output, tuple):
                                return (steered,) + output[1:]
                            return steered
                        return hook

                    h = self.model.model.layers[self.layer].register_forward_hook(
                        _make_steering_hook(
                            alpha, self.layer, mini_batch_states,
                            step_counter, captured, target_token_n_list,
                            apply_steer,
                        )
                    )
                    handles.append(h)

                    def _make_layer_hook(ln, st, ctr, cap, ttn):
                        def hook(module, inp, output):
                            hs = output[0] if isinstance(output, tuple) else output
                            step = ctr[0] - 1
                            if step < 0:
                                return output
                            last = hs[:, -1, :].clone()
                            if step == 0:
                                k = f"t0_{ln}_b"
                                if k not in cap:
                                    st['token_0']['before_steering'][ln] = last
                                    cap.add(k)
                            for n in ttn:
                                if step == n:
                                    k = f"t{n}_{ln}_b"
                                    if k not in cap:
                                        st[f'token_{n}']['before_steering'][ln] = last
                                        cap.add(k)
                            return output
                        return hook

                    for kv in k_list:
                        lpk = min(self.layer + kv, len(self.model.model.layers) - 1)
                        if lpk == self.layer:
                            continue
                        layer_name = f"layer_{lpk}"
                        h = self.model.model.layers[lpk].register_forward_hook(
                            _make_layer_hook(
                                layer_name, mini_batch_states,
                                step_counter, captured, target_token_n_list,
                            )
                        )
                        handles.append(h)
                else:
                    self.hook_call_count = 0
                    self.collect_hidden_states = False
                    if alpha != 0.0 and self.steering_vector is not None:
                        _, h = self.apply_steering_hook(alpha)
                        handles.append(h)

                # ---- Generate ------------------------------------------------
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            pad_token_id=self.tokenizer.eos_token_id,
                            **kwargs
                        )
                finally:
                    for h in handles:
                        h.remove()

                # ---- Decode --------------------------------------------------
                for output in outputs:
                    generated_ids = output[input_length:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    all_responses.append(self._clean_response(response))

                # ---- Split batch hidden states into per-sample ---------------
                if collect_hidden_states and mini_batch_states is not None:
                    for i in range(len(batch_prompts)):
                        sample_idx = batch_start + i
                        sample_states = {}
                        for token_key, token_data in mini_batch_states.items():
                            sample_states[token_key] = {}
                            for timing_key, timing_data in token_data.items():
                                sample_states[token_key][timing_key] = {}
                                for layer_key, tensor in timing_data.items():
                                    sample_states[token_key][timing_key][layer_key] = (
                                        tensor[i].clone()
                                    )
                        self._batch_hidden_states[f'sample_{sample_idx}'] = sample_states

                print(f"  Batch [{batch_num}/{total_batches}]: {len(batch_prompts)} samples done")
        finally:
            self.tokenizer.padding_side = original_padding_side

        return all_responses

    def get_batch_hidden_states(self) -> Dict[str, Any]:
        """Return hidden states collected by the last :meth:`generate_batch` call."""
        return getattr(self, '_batch_hidden_states', {})

    @staticmethod
    def _clean_response(response: str) -> str:
        """Strip chat template role markers if the model echoed the full conversation."""
        for marker in ["\nmodel\n", "\nassistant\n"]:
            idx = response.find(marker)
            if idx != -1:
                response = response[idx + len(marker):]
                break
        return response.strip()

    def get_hidden_states_for_prompt(self, prompt: str, with_steering: bool = False, alpha: float = 1.0) -> torch.Tensor:
        """Get hidden states for a prompt, optionally with steering applied."""
        self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        
        handle = None
        if with_steering and self.steering_vector is not None:
            _, handle = self.apply_steering_hook(alpha)
        
        with torch.no_grad():
            hidden = gather_residual_activations(self.model, self.layer, inputs)
        
        if handle:
            handle.remove()
            
        return hidden
    
    def get_collected_hidden_states(self) -> Dict[str, Any]:
        """
        Get the collected hidden states for analysis.
        
        Returns:
            Dictionary containing hidden states at different layers and token positions
        """
        return self.stored_hidden_states.copy()
    
    def save_hidden_states(self, save_path: str):
        """
        Save collected hidden states to disk.
        
        Args:
            save_path: Path to save the hidden states
        """
        # print(f"DEBUG: Attempting to save hidden states. Keys: {list(self.stored_hidden_states.keys())}")
        
        if not self.stored_hidden_states:
            print("No hidden states to save")
            return
        
        # Check if we have any actual data
        has_data = False
        for token_key, token_data in self.stored_hidden_states.items():
            for timing_key, timing_data in token_data.items():
                if timing_data:  # If this timing has data
                    has_data = True
                    break
            if has_data:
                break
        
        if not has_data:
            print("No hidden states data to save (empty dictionaries)")
            return
            
        # Convert tensors to CPU and detach for saving
        states_to_save = {}
        for token_key, token_data in self.stored_hidden_states.items():
            states_to_save[token_key] = {}
            for timing_key, timing_data in token_data.items():
                states_to_save[token_key][timing_key] = {}
                for layer_key, tensor in timing_data.items():
                    states_to_save[token_key][timing_key][layer_key] = tensor.cpu().detach()
        
        torch.save(states_to_save, save_path)
        print(f"✓ Hidden states saved to {save_path}")
        # print(f"  Saved data structure: {list(states_to_save.keys())}")
    
    def save_vector(self, save_dir: str, metadata: Dict[str, Any] = None):
        """Save steering vector and metadata."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vector
        torch.save(self.steering_vector, save_path / f"{self.__class__.__name__}_vector.pt")
        
        # Save metadata
        meta = {
            "method": self.__class__.__name__,
            "model_name": self.model_name,
            "layer": self.layer,
            "vector_shape": list(self.steering_vector.shape),
            "vector_norm": torch.norm(self.steering_vector).item(),
            **(metadata or {})
        }
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✓ Saved to {save_path}")
    
    def load_vector(self, load_dir: str):
        """Load steering vector from disk."""
        load_path = Path(load_dir)
        vector_file = load_path / f"{self.__class__.__name__}_vector.pt"
        
        self.steering_vector = torch.load(vector_file, map_location=self.device)
        print(f"✓ Loaded steering vector from {vector_file}")


class LinearProbe(SteeringMethod):
    """
    Linear probe steering method.
    
    Trains a binary classifier on hidden states and uses the learned
    weight vector as the steering direction.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        layer: int = 20,
        device: str = None,
        learning_rate: float = 0.005,
        num_epochs: int = 10,
        weight_decay: float = 0.001,
        l2_lambda: float = 0.01
    ):
        super().__init__(model_name, layer, device)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.l2_lambda = l2_lambda
        self.probe = None
        self.training_stats = {}
    
    def compute_steering_vector(
        self,
        positive_examples: List[str],
        negative_examples: List[str]
    ) -> torch.Tensor:
        """
        Train linear probe and extract steering vector.
        
        Args:
            positive_examples: Texts containing the concept
            negative_examples: Texts not containing the concept
            
        Returns:
            Steering vector (probe weights)
        """
        print(f"\n{'='*80}")
        print("TRAINING LINEAR PROBE")
        print(f"{'='*80}")
        
        # Extract hidden states
        print(f"\nExtracting hidden states from {len(positive_examples)} positive examples...")
        pos_hidden = self.extract_hidden_states(positive_examples, average=True)
        
        print(f"Extracting hidden states from {len(negative_examples)} negative examples...")
        neg_hidden = self.extract_hidden_states(negative_examples, average=True)
        
        # Prepare training data
        X = torch.stack(pos_hidden + neg_hidden).float()  # [num_examples, hidden_size]
        y = torch.tensor([1.0] * len(pos_hidden) + [0.0] * len(neg_hidden))  # [num_examples]
        
        # Initialize probe
        hidden_size = X.shape[1]
        self.probe = nn.Linear(hidden_size, 1, bias=False).to(self.device)
        
        # Training
        optimizer = optim.AdamW(
            self.probe.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"\nTraining for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            logits = self.probe(X.to(self.device)).squeeze(-1)
            loss = criterion(logits, y.to(self.device)) + self.l2_lambda * self.probe.weight.norm(2)
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == y.to(self.device)).float().mean()
            
            if (epoch + 1) % max(1, self.num_epochs // 5) == 0:
                print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}, Acc = {acc.item():.3f}")
        
        # Extract steering vector
        self.steering_vector = self.probe.weight.detach().clone()  # [1, hidden_size]
        
        # Save training stats
        self.training_stats = {
            "final_loss": loss.item(),
            "final_accuracy": acc.item(),
            "num_positive": len(pos_hidden),
            "num_negative": len(neg_hidden),
        }
        
        print(f"\n✓ Training complete!")
        print(f"  Final loss: {loss.item():.4f}")
        print(f"  Final accuracy: {acc.item():.3f}")
        print(f"  Steering vector shape: {self.steering_vector.shape}")
        print(f"  Steering vector norm: {torch.norm(self.steering_vector).item():.4f}")
        
        return self.steering_vector
    
    def get_training_accuracy(self) -> float:
        """Get the final training accuracy."""
        return self.training_stats.get("final_accuracy", 0.0)


class DiffMean(SteeringMethod):
    """
    Difference of Means steering method.
    
    Computes steering vector as the difference between the mean of positive
    and negative example representations. No training required!
    
    Reference:
    - https://arxiv.org/abs/2310.06824
    - https://blog.eleuther.ai/diff-in-means/
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        layer: int = 20,
        device: str = None,
        normalize: bool = False
    ):
        super().__init__(model_name, layer, device)
        self.normalize = normalize
        self.mean_positive = None
        self.mean_negative = None
    
    def compute_steering_vector(
        self,
        positive_examples: List[str],
        negative_examples: List[str]
    ) -> torch.Tensor:
        """
        Compute steering vector as difference of means.
        
        Args:
            positive_examples: Texts containing the concept
            negative_examples: Texts not containing the concept
            
        Returns:
            Steering vector (mean_positive - mean_negative)
        """
        print(f"\n{'='*80}")
        print("COMPUTING DIFFMEAN STEERING VECTOR")
        print(f"{'='*80}")
        
        # Extract hidden states
        print(f"\nExtracting hidden states from {len(positive_examples)} positive examples...")
        pos_hidden = self.extract_hidden_states(positive_examples, average=True)
        
        print(f"Extracting hidden states from {len(negative_examples)} negative examples...")
        neg_hidden = self.extract_hidden_states(negative_examples, average=True)
        
        # Compute means
        print("\nComputing means...")
        self.mean_positive = torch.stack(pos_hidden).mean(dim=0)  # [hidden_size]
        self.mean_negative = torch.stack(neg_hidden).mean(dim=0)  # [hidden_size]
        
        # Compute difference
        self.steering_vector = (self.mean_positive - self.mean_negative).unsqueeze(0)  # [1, hidden_size]
        
        # Optional normalization
        if self.normalize:
            self.steering_vector = self.steering_vector / torch.norm(self.steering_vector)
            print("  Normalized to unit norm")
        
        print(f"\n✓ Steering vector computed!")
        print(f"  Shape: {self.steering_vector.shape}")
        print(f"  Norm: {torch.norm(self.steering_vector).item():.4f}")
        print(f"  Mean positive norm: {torch.norm(self.mean_positive).item():.4f}")
        print(f"  Mean negative norm: {torch.norm(self.mean_negative).item():.4f}")
        
        return self.steering_vector
    
    def get_steering_vector_norm(self) -> float:
        """Get the norm of the steering vector."""
        return torch.norm(self.steering_vector).item()

