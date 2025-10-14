# ABOUTME: Demo notebook for calculating KL divergence between two models' full probability distributions.
# ABOUTME: Loads both models, extracts full vocabulary probabilities, and calculates KL divergence.

# %%
import glob
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
# Parameters
model_1_name = "google/gemma-2-9b-it"
model_2_name = "bcywinski/gemma-2-9b-it-taboo-cloud"
responses_dir_model1 = "/workspace/projects/diffing-prompts/experiments/results/responses_openrouter/gemma-2-9b-it"
responses_dir_model2 = "/workspace/projects/diffing-prompts/experiments/results/responses_openrouter/gemma-2-9b-it-taboo-cloud"
output_dir = "/workspace/projects/diffing-prompts/experiments/results/kl"  # Where to save the KL results
max_prompts = None  # Set to a number to limit, or None to process all
os.makedirs(output_dir, exist_ok=True)
# %%
# Load both models and tokenizers
print(f"Loading model 1: {model_1_name}")
tokenizer_1 = AutoTokenizer.from_pretrained(model_1_name, trust_remote_code=True)
model_1 = AutoModelForCausalLM.from_pretrained(
    model_1_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("Model 1 loaded successfully")

print(f"\nLoading model 2: {model_2_name}")
tokenizer_2 = AutoTokenizer.from_pretrained(model_2_name, trust_remote_code=True)
model_2 = AutoModelForCausalLM.from_pretrained(
    model_2_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("Model 2 loaded successfully")

# %%
# Find all response files for both models
responses_files_model1 = sorted(glob.glob(os.path.join(responses_dir_model1, "*.json")))
responses_files_model2 = sorted(glob.glob(os.path.join(responses_dir_model2, "*.json")))

print(f"\nFound {len(responses_files_model1)} response files for model 1")
print(f"Found {len(responses_files_model2)} response files for model 2")

# Create a mapping from prompt index to files
# Extract prompt indices from filenames
def get_prompt_index(filepath):
    """Extract prompt index from filename like 'model_prompt_123.json'"""
    basename = os.path.basename(filepath)
    # Find the last occurrence of 'prompt_' and extract number
    parts = basename.split('prompt_')
    if len(parts) > 1:
        idx = parts[-1].split('.')[0]
        try:
            return int(idx)
        except ValueError:
            return None
    return None

# Build mappings
model1_files_by_idx = {}
for f in responses_files_model1:
    idx = get_prompt_index(f)
    if idx is not None:
        model1_files_by_idx[idx] = f

model2_files_by_idx = {}
for f in responses_files_model2:
    idx = get_prompt_index(f)
    if idx is not None:
        model2_files_by_idx[idx] = f

# Find common prompt indices
common_indices = sorted(set(model1_files_by_idx.keys()) & set(model2_files_by_idx.keys()))
print(f"\nFound {len(common_indices)} prompts with responses from both models")

if max_prompts is not None:
    common_indices = common_indices[:max_prompts]
    print(f"Processing first {len(common_indices)} prompts")

# %%
def calculate_kl_divergence_full_vocab(
    prompt: str,
    response_tokens: list,
    model_1,
    tokenizer_1,
    model_2,
    tokenizer_2,
):
    """Calculate KL divergence between two models' full vocabulary distributions.

    Calculates KL divergence for responses from model 1: KL(model_1 || model_2)

    Args:
        prompt: The input prompt text
        response_tokens: List of response token strings
        model_1: First model (P distribution)
        tokenizer_1: Tokenizer for first model
        model_2: Second model (Q distribution)
        tokenizer_2: Tokenizer for second model

    Returns:
        Average KL divergence per token (float), list of per-token KLs
    """
    # Prepare prompt with chat template
    user_prompt = tokenizer_1.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
        add_bos=False,
    )
    user_prompt_tokens = tokenizer_1.encode(
        user_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )[0, :]

    # Convert response tokens to ids
    response_token_ids = []
    for token in response_tokens:
        token_id = tokenizer_1.encode(token, add_special_tokens=False)[0]
        response_token_ids.append(token_id)
    response_token_ids = torch.tensor(response_token_ids)

    # Combine prompt and response tokens
    tokens = torch.cat([user_prompt_tokens, response_token_ids])
    tokens = tokens.to(model_1.device)

    # Get full vocabulary log probabilities from model 1
    with torch.no_grad():
        outputs_1 = model_1(tokens.unsqueeze(0))
        logits_1 = outputs_1.logits
        log_probs_1 = torch.log_softmax(logits_1, dim=-1)
        # Extract log probs for the response tokens
        log_probs_1 = log_probs_1[0, len(user_prompt_tokens) - 1 : -1].cpu()

    # Get full vocabulary log probabilities from model 2
    # Assuming same tokenizer, so same tokens work
    tokens_2 = tokens.to(model_2.device)
    with torch.no_grad():
        outputs_2 = model_2(tokens_2.unsqueeze(0))
        logits_2 = outputs_2.logits
        log_probs_2 = torch.log_softmax(logits_2, dim=-1)
        # Extract log probs for the response tokens
        log_probs_2 = log_probs_2[0, len(user_prompt_tokens) - 1 : -1].cpu()

    # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
    # In log space: KL(P||Q) = sum(exp(log_P) * (log_P - log_Q))
    # Using PyTorch's kl_div: expects input=log(Q), target=log(P)
    # F.kl_div computes sum(exp(target) * (target - input))
    kl_per_token = torch.nn.functional.kl_div(
        log_probs_2,  # Q distribution (model 2)
        log_probs_1,  # P distribution (model 1)
        reduction="none",
        log_target=True,
    ).sum(dim=-1)  # Sum over vocabulary dimension

    # Average over tokens
    avg_kl = kl_per_token.mean().item()

    return avg_kl, kl_per_token.tolist()

# %%
# Calculate KL divergence for each prompt
results = []

for prompt_idx in common_indices:
    file_model1 = model1_files_by_idx[prompt_idx]
    file_model2 = model2_files_by_idx[prompt_idx]

    # Load responses from model 1
    with open(file_model1, "r") as fp:
        data_model1 = json.load(fp)

    # Load responses from model 2
    with open(file_model2, "r") as fp:
        data_model2 = json.load(fp)

    print(f"\nProcessing prompt {prompt_idx}:")
    print(f"  Model 1: {os.path.basename(file_model1)}")
    print(f"  Model 2: {os.path.basename(file_model2)}")

    prompt = data_model1["prompt"]

    # Verify prompts match
    if data_model1["prompt"] != data_model2["prompt"]:
        print("  WARNING: Prompts don't match! Skipping.")
        continue

    # Process responses from model 1: calculate KL(M1||M2)
    print("  Model 1 responses:")
    response_kls_model1 = []
    for response_data in data_model1["responses"]:
        # Extract tokens from response
        logprobs_data = response_data["choices"][0]["logprobs"]["content"]
        response_tokens = [token_data["token"] for token_data in logprobs_data]

        try:
            # KL(model_1 || model_2) using model 1's response
            avg_kl, per_token_kls = calculate_kl_divergence_full_vocab(
                prompt=prompt,
                response_tokens=response_tokens,
                model_1=model_1,
                tokenizer_1=tokenizer_1,
                model_2=model_2,
                tokenizer_2=tokenizer_2,
            )
            response_kls_model1.append(avg_kl)
            print(f"    KL(M1||M2): {avg_kl:.6f}")
        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Process responses from model 2: calculate KL(M2||M1)
    print("  Model 2 responses:")
    response_kls_model2 = []
    for response_data in data_model2["responses"]:
        # Extract tokens from response
        logprobs_data = response_data["choices"][0]["logprobs"]["content"]
        response_tokens = [token_data["token"] for token_data in logprobs_data]

        try:
            # KL(model_2 || model_1) using model 2's response
            avg_kl, per_token_kls = calculate_kl_divergence_full_vocab(
                prompt=prompt,
                response_tokens=response_tokens,
                model_1=model_2,  # Swapped: model_2 is P
                tokenizer_1=tokenizer_2,
                model_2=model_1,  # Swapped: model_1 is Q
                tokenizer_2=tokenizer_1,
            )
            response_kls_model2.append(avg_kl)
            print(f"    KL(M2||M1): {avg_kl:.6f}")
        except Exception as e:
            print(f"    Error: {e}")
            continue

    if len(response_kls_model1) == 0 and len(response_kls_model2) == 0:
        print("  Skipped (no valid responses)")
        continue

    # Calculate averages
    avg_kl_model1 = sum(response_kls_model1) / len(response_kls_model1) if response_kls_model1 else 0
    avg_kl_model2 = sum(response_kls_model2) / len(response_kls_model2) if response_kls_model2 else 0

    # Average of both directions (symmetric KL divergence)
    average_kl = (avg_kl_model1 + avg_kl_model2) / 2

    print(f"  Average KL(M1||M2): {avg_kl_model1:.6f}")
    print(f"  Average KL(M2||M1): {avg_kl_model2:.6f}")
    print(f"  Symmetric Average: {average_kl:.6f}")

    results.append({
        "prompt_idx": prompt_idx,
        "file_model1": file_model1,
        "file_model2": file_model2,
        "prompt": prompt,
        "average_kl": average_kl,
        "avg_kl_model1_to_model2": avg_kl_model1,
        "avg_kl_model2_to_model1": avg_kl_model2,
        "response_kls_model1": response_kls_model1,
        "response_kls_model2": response_kls_model2,
    })

# %%
# Sort results by average KL divergence (highest first) and save to JSON
import datetime

results.sort(key=lambda x: x["average_kl"], reverse=True)

# Create detailed results with prompts, average KL and per-response KL values
kl_results = [
    {
        "prompt_idx": r["prompt_idx"],
        "prompt": r["prompt"],
        "average_kl_symmetric": r["average_kl"],
        "avg_kl_model1_to_model2": r["avg_kl_model1_to_model2"],
        "avg_kl_model2_to_model1": r["avg_kl_model2_to_model1"],
        "response_kls_model1": r["response_kls_model1"],
        "response_kls_model2": r["response_kls_model2"],
        "file_model1": r["file_model1"],
        "file_model2": r["file_model2"],
    }
    for i, r in enumerate(results)
]

# Save KL results to JSON file
output_file = os.path.join(output_dir, "prompt_kl_values_full_vocab.json")
with open(output_file, "w") as f:
    json.dump({
        "prompt_kl_values": kl_results,
        "metadata": {
            "model_1": model_1_name,
            "model_2": model_2_name,
            "n_prompts": len(kl_results),
            "calculation_method": "symmetric_full_vocabulary_kl_divergence",
            "description": "Average of KL(M1||M2) on M1 responses and KL(M2||M1) on M2 responses",
            "timestamp": datetime.datetime.now().isoformat()
        }
    }, f, indent=2)

print("\n" + "="*80)
print("SORTED RESULTS (highest KL divergence first):")
print("="*80)
print(f"Results saved to: {output_file}")

# %%
for i, result in enumerate(results, 1):
    print(f"\n{i}. Prompt {result['prompt_idx']}: Symmetric Average KL: {result['average_kl']:.6f}")
    print(f"   KL(M1||M2): {result['avg_kl_model1_to_model2']:.6f}")
    print(f"   KL(M2||M1): {result['avg_kl_model2_to_model1']:.6f}")
    print(f"   Prompt: {result['prompt'][:100]}...")
    if i >= 10:
        break

# %%
