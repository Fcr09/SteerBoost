import argparse
import json
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def process_file(result_file, model, tokenizer, device, k_list, target_token_n_list, output_dir, layers_to_capture, batch_size=1):
    print(f"Processing: {result_file}")
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
        return

    metadata = data['metadata']
    results = data['results']
    
    method = metadata['method']
    model_name = metadata['model']
    layer_idx = metadata['layer']
    alpha = metadata['alpha'] 
    concept_id = metadata['concept_id']
    
    model_short = model_name.split('/')[-1]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    valid_samples = []
    for item in results:
        sample_id = item['sample_id']
        instruction = item.get('instruction', item.get('prompt'))
        response_after = item.get('response_after')
        if response_after is None:
            continue
        full_text = instruction + response_after
        prompt_inputs = tokenizer(instruction, return_tensors="pt", add_special_tokens=True)
        prompt_len = prompt_inputs.input_ids.shape[1]
        valid_samples.append({
            'sample_id': sample_id,
            'full_text': full_text,
            'prompt_len': prompt_len,
        })

    hidden_states_data = {}
    n_batches = (len(valid_samples) + batch_size - 1) // batch_size

    for batch_start in tqdm(range(0, len(valid_samples), batch_size),
                            desc=f"Concept {concept_id} (bs={batch_size})",
                            total=n_batches, leave=False):
        batch = valid_samples[batch_start:batch_start + batch_size]
        batch_texts = [s['full_text'] for s in batch]

        batch_inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, add_special_tokens=True,
        ).to(device)

        captured_states = {l: None for l in layers_to_capture}

        def get_hook(layer_num):
            def hook(module, input, output):
                val = output[0] if isinstance(output, tuple) else output
                captured_states[layer_num] = val.detach().cpu()
            return hook

        hooks = []
        for l in layers_to_capture:
            hooks.append(model.model.layers[l].register_forward_hook(get_hook(l)))

        with torch.no_grad():
            model(**batch_inputs)

        for h in hooks:
            h.remove()

        for batch_idx, sample_info in enumerate(batch):
            sample_id = sample_info['sample_id']
            idx_0 = sample_info['prompt_len'] - 1
            seq_len = int(batch_inputs.attention_mask[batch_idx].sum().item())

            sample_data = {'token_0': {'before_steering': {}}}
            for n in target_token_n_list:
                sample_data[f'token_{n}'] = {'before_steering': {}}

            if idx_0 < seq_len:
                for l in layers_to_capture:
                    layer_key = f"layer_{l}"
                    if captured_states[l] is not None:
                        sample_data['token_0']['before_steering'][layer_key] = captured_states[l][batch_idx, idx_0, :].clone()

            for n in target_token_n_list:
                idx_n = idx_0 + n
                if idx_n < seq_len:
                    for l in layers_to_capture:
                        layer_key = f"layer_{l}"
                        if captured_states[l] is not None:
                            sample_data[f'token_{n}']['before_steering'][layer_key] = captured_states[l][batch_idx, idx_n, :].clone()

            hidden_states_data[f'sample_{sample_id}'] = sample_data

    tokenizer.padding_side = orig_padding_side

    # Construct output path
    filename = f"{layer_idx}-{alpha}.pt"
    output_path = Path(output_dir) / method / model_short / str(concept_id) / filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_data = {
        'metadata': metadata.copy(),
        'hidden_states': hidden_states_data
    }
    
    final_data['metadata']['k_list'] = k_list
    final_data['metadata']['target_token_n_list'] = target_token_n_list
    final_data['metadata']['is_raw_extraction'] = True
    final_data['metadata']['source_result_file'] = str(result_file)
    final_data['metadata']['timestamp'] = str(data['metadata'].get('timestamp')) + "_raw_extracted"
    
    print(f"Saving raw hidden states to: {output_path}")
    torch.save(final_data, output_path)


def main():
    parser = argparse.ArgumentParser(description="Extract raw hidden states from model inference on existing results")
    parser.add_argument("--result_dir", type=str, default="data/result", help="Base directory for results")
    parser.add_argument("--method", type=str, required=True, help="Steering method (e.g., diffmean)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., google/gemma-2-2b-it)")
    parser.add_argument("--layer", type=int, required=True, help="Layer index")
    parser.add_argument("--alpha", type=float, required=True, help="Steering strength alpha")
    parser.add_argument("--output_dir", type=str, default="data/hidden_states_raw", help="Directory to save raw hidden states")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--k_list", type=str, default="5", help="Comma-separated list of offsets")
    parser.add_argument("--target_token_n_list", type=str, default="5", help="Comma-separated list of tokens")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for model inference")
    
    args = parser.parse_args()
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Parse lists
    k_list = [int(x.strip()) for x in args.k_list.split(',')]
    target_token_n_list = [int(x.strip()) for x in args.target_token_n_list.split(',')]
    
    # Identify layers to capture
    layer_idx = args.layer
    layers_to_capture = {layer_idx}
    
    # Load model first
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    model.eval()
    
    # Add L+k layers
    # We need to check model layer count to be safe, though usually fixed for a model
    num_layers = len(model.model.layers)
    for k in k_list:
        l_plus_k = layer_idx + k
        if l_plus_k < num_layers:
            layers_to_capture.add(l_plus_k)
            
    print(f"Collecting hidden states for layer {layer_idx}, k={k_list}, target_tokens={target_token_n_list}")
    print(f"Capturing layers: {sorted(list(layers_to_capture))}")

    # Find matching files
    model_short = args.model.split('/')[-1]
    search_path = Path(args.result_dir) / args.method / model_short
    
    if not search_path.exists():
        print(f"Search path does not exist: {search_path}")
        return

    # Pattern: {concept_id}/{layer}-{alpha}.json
    # We'll glob for all subdirectories
    found_files = []
    target_filename = f"{args.layer}-{args.alpha}.json"
    
    print(f"Searching in: {search_path} for files named {target_filename}")
    
    for concept_dir in search_path.iterdir():
        if concept_dir.is_dir():
            target_file = concept_dir / target_filename
            if target_file.exists():
                found_files.append(target_file)
                
    print(f"Found {len(found_files)} files to process.")
    
    for result_file in found_files:
        process_file(
            result_file, 
            model, 
            tokenizer, 
            device, 
            k_list, 
            target_token_n_list, 
            args.output_dir,
            layers_to_capture,
            batch_size=args.batch_size,
        )

    print("All done.")

if __name__ == "__main__":
    main()
