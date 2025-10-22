# ABOUTME: Script to calculate KL divergence between two models' probability distributions on a set of prompts.
# ABOUTME: Processes response files, computes full-vocabulary KL divergence, and saves sorted results to JSON.

import json
import os
from typing import Optional

import fire
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

from kl_divergence import (
    calculate_kl_divergence_full_vocab,
    load_model_and_tokenizer,
    load_response_files,
)


def save_stage1_intermediate(
    intermediate_dir: str,
    prompt_idx: int,
    response_idx: int,
    prompt: str,
    response_token_ids: list,
    user_prompt_tokens: list,
    log_probs_1: torch.Tensor,
):
    """Save intermediate results from stage 1 (model_1 log probabilities).

    Args:
        intermediate_dir: Directory to save intermediate results
        prompt_idx: Index of the prompt
        response_idx: Index of the response
        prompt: The prompt text
        response_token_ids: List of response token IDs
        user_prompt_tokens: List of user prompt token IDs
        log_probs_1: Log probabilities from model_1 [num_tokens, vocab_size]
    """
    os.makedirs(intermediate_dir, exist_ok=True)

    data = {
        "prompt_idx": prompt_idx,
        "response_idx": response_idx,
        "prompt": prompt,
        "response_token_ids": response_token_ids,
        "user_prompt_tokens": user_prompt_tokens,
        "log_probs_1": log_probs_1,  # Keep as tensor
    }

    filename = f"stage1_prompt_{prompt_idx}_response_{response_idx}.pt"
    filepath = os.path.join(intermediate_dir, filename)

    torch.save(data, filepath)


def load_stage1_intermediate(intermediate_dir: str, prompt_idx: int, response_idx: int):
    """Load intermediate results from stage 1.

    Args:
        intermediate_dir: Directory containing intermediate results
        prompt_idx: Index of the prompt
        response_idx: Index of the response

    Returns:
        Dictionary containing stage 1 data
    """
    filename = f"stage1_prompt_{prompt_idx}_response_{response_idx}.pt"
    filepath = os.path.join(intermediate_dir, filename)

    data = torch.load(filepath, map_location="cpu", weights_only=False)

    return data


def compute_model_log_probs(
    prompt: str,
    response_token_ids: list,
    model,
    tokenizer,
):
    """Compute log probabilities for a model given prompt and response.

    Args:
        prompt: The input prompt text
        response_token_ids: List of response token IDs
        model: The model to compute log probs with
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (user_prompt_tokens as list, log_probs as tensor [num_tokens, vocab_size])
    """
    # Prepare prompt with chat template
    user_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
        add_bos=False,
    )
    user_prompt_tokens = tokenizer.encode(
        user_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )[0, :]

    # Combine prompt and response tokens
    tokens = torch.cat([user_prompt_tokens, torch.tensor(response_token_ids)])
    tokens = tokens.to(model.device)

    # Get full vocabulary log probabilities
    with torch.no_grad():
        outputs = model(tokens.unsqueeze(0))
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        # Extract log probs for the response tokens
        log_probs = log_probs[0, len(user_prompt_tokens) - 1 : -1].cpu()

    return user_prompt_tokens.tolist(), log_probs


def main(config_path: str, max_prompts: Optional[int] = None, stage: str = "all"):
    """Calculate KL divergence between two models using saved responses.

    Args:
        config_path: Path to YAML config file
        max_prompts: Maximum number of prompts to process (None for all)
        stage: Execution stage - "all" (default), "stage1" (only model_1), or "stage2" (only model_2)
    """
    if stage not in ["all", "stage1", "stage2"]:
        raise ValueError(f"Invalid stage: {stage}. Must be 'all', 'stage1', or 'stage2'")
    load_dotenv()

    # Load configuration
    config = OmegaConf.load(config_path)
    print(f"Loaded config from: {config_path}")
    print(f"Stage: {stage}")
    print(OmegaConf.to_yaml(config))

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Set intermediate directory (used for stage1 and stage2)
    intermediate_dir = config.get("intermediate_dir", os.path.join(config.output_dir, "intermediate"))

    # Load models based on stage
    if stage in ["all", "stage1"]:
        model_1, tokenizer_1 = load_model_and_tokenizer(
            config.model_1_name, config.device_map
        )
    else:
        model_1, tokenizer_1 = None, None

    if stage in ["all", "stage2"]:
        model_2, tokenizer_2 = load_model_and_tokenizer(
            config.model_2_name, config.device_map
        )
    else:
        model_2, tokenizer_2 = None, None

    # Load response files from model 2
    model2_files_by_idx = load_response_files(config.responses_dir_model2)

    # Limit number of prompts if specified
    prompt_indices = sorted(model2_files_by_idx.keys())
    if max_prompts is not None:
        prompt_indices = prompt_indices[:max_prompts]
        print(f"Limiting to {max_prompts} prompts")

    if stage == "stage1":
        # Stage 1: Compute and save model_1 log probabilities
        print("\nRunning Stage 1: Computing model_1 log probabilities")
        print(f"Intermediate results will be saved to: {intermediate_dir}")

        for prompt_idx in tqdm(prompt_indices, desc="Stage 1: Processing prompts"):
            file_model2 = model2_files_by_idx[prompt_idx]

            # Load responses from model 2
            with open(file_model2, "r") as fp:
                data_model2 = json.load(fp)

            prompt = data_model2["prompt"]

            for response_idx, response_data in enumerate(data_model2["responses"]):
                response_text = response_data["choices"][0]["message"]["content"]
                token_ids = tokenizer_1.encode(response_text, add_special_tokens=False)

                try:
                    user_prompt_tokens, log_probs_1 = compute_model_log_probs(
                        prompt=prompt,
                        response_token_ids=token_ids,
                        model=model_1,
                        tokenizer=tokenizer_1,
                    )

                    save_stage1_intermediate(
                        intermediate_dir=intermediate_dir,
                        prompt_idx=prompt_idx,
                        response_idx=response_idx,
                        prompt=prompt,
                        response_token_ids=token_ids,
                        user_prompt_tokens=user_prompt_tokens,
                        log_probs_1=log_probs_1,
                    )
                except Exception as e:
                    print(f"Error processing prompt {prompt_idx}, response {response_idx}: {e}")
                    continue

        print(f"\nStage 1 complete. Intermediate results saved to: {intermediate_dir}")
        print("Run stage2 to compute KL divergence using model_2")

    elif stage == "stage2":
        # Stage 2: Load stage 1 data and compute KL divergence with model_2
        print("\nRunning Stage 2: Computing KL divergence with model_2")
        print(f"Loading intermediate results from: {intermediate_dir}")

        results = []

        for prompt_idx in tqdm(prompt_indices, desc="Stage 2: Processing prompts"):
            file_model2 = model2_files_by_idx[prompt_idx]

            # Load responses from model 2 to get count
            with open(file_model2, "r") as fp:
                data_model2 = json.load(fp)

            response_kls_model2_avg = []
            response_kls_model2_per_token = []
            response_texts = []

            for response_idx in range(len(data_model2["responses"])):
                try:
                    # Load stage 1 intermediate data
                    stage1_data = load_stage1_intermediate(
                        intermediate_dir, prompt_idx, response_idx
                    )

                    prompt = stage1_data["prompt"]
                    response_text = stage1_data["response_text"]
                    response_token_ids = stage1_data["response_token_ids"]
                    log_probs_1 = stage1_data["log_probs_1"]

                    # Compute model_2 log probabilities
                    _, log_probs_2 = compute_model_log_probs(
                        prompt=prompt,
                        response_token_ids=response_token_ids,
                        model=model_2,
                        tokenizer=tokenizer_2,  # Use same tokenizer for consistency
                    )

                    # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
                    kl_per_token = torch.nn.functional.kl_div(
                        log_probs_2,  # Q distribution (model 2)
                        log_probs_1,  # P distribution (model 1)
                        reduction="none",
                        log_target=True,
                    ).sum(dim=-1)

                    response_texts.append(response_text)

                    avg_kl = kl_per_token.mean().item()

                    response_kls_model2_avg.append(avg_kl)
                    response_kls_model2_per_token.append(kl_per_token.tolist())

                except Exception as e:
                    print(f"Error processing prompt {prompt_idx}, response {response_idx}: {e}")
                    continue

            if len(response_kls_model2_avg) == 0:
                print(f"Skipped prompt {prompt_idx} (no valid responses)")
                continue

            # Average over responses
            avg_kl_model2 = sum(response_kls_model2_avg) / len(response_kls_model2_avg)

            results.append(
                {
                    "prompt_idx": prompt_idx,
                    "file_model2": file_model2,
                    "prompt": prompt,
                    "response_texts": response_texts,
                    "average_kl": avg_kl_model2,
                    "response_kls_model2_avg": response_kls_model2_avg,
                    "response_kls_model2_per_token": response_kls_model2_per_token,
                }
            )

        # Sort and save results
        results.sort(key=lambda x: x["average_kl"], reverse=True)
        print(f"\nSorted {len(results)} results by average KL divergence")

        output_file = os.path.join(config.output_dir, "kl_divergence_results_sorted.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        print(f"Total prompts processed: {len(results)}")

        # Print top 5 prompts with highest KL
        print("\nTop 5 prompts with highest KL divergence:")
        print("-" * 80)
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. Prompt {result['prompt_idx']}: {result['average_kl']:.6f}")
            print(f"   {result['prompt'][:100]}...")
            print()

    else:  # stage == "all"
        # Original logic: Load both models and compute KL in one pass
        print("\nRunning all stages in one pass (both models loaded)")

        results = []

        for prompt_idx in tqdm(prompt_indices, desc="Processing prompts"):
            file_model2 = model2_files_by_idx[prompt_idx]

            # Load responses from model 2
            with open(file_model2, "r") as fp:
                data_model2 = json.load(fp)

            prompt = data_model2["prompt"]

            # Process responses from model 2: calculate KL(M1||M2)
            response_kls_model2_avg = []
            response_kls_model2_per_token = []
            response_texts = []
            for response_data in data_model2["responses"]:
                response_text = response_data["choices"][0]["message"]["content"]
                token_ids = tokenizer_1.encode(response_text, add_special_tokens=False)

                try:
                    # KL(model_1 || model_2) using model 2's response
                    avg_kl, per_token_kls = calculate_kl_divergence_full_vocab(
                        prompt=prompt,
                        response_token_ids=token_ids,
                        model_1=model_1,
                        tokenizer_1=tokenizer_1,
                        model_2=model_2,
                    )
                    response_texts.append(response_text)
                    response_kls_model2_avg.append(avg_kl)
                    response_kls_model2_per_token.append(per_token_kls)
                except Exception as e:
                    print(f"Error processing prompt {prompt_idx}, response: {e}")
                    continue

            if len(response_kls_model2_avg) == 0:
                print(f"Skipped prompt {prompt_idx} (no valid responses)")
                continue

            # Average over model 2 responses
            avg_kl_model2 = sum(response_kls_model2_avg) / len(response_kls_model2_avg)

            results.append(
                {
                    "prompt_idx": prompt_idx,
                    "file_model2": file_model2,
                    "prompt": prompt,
                    "response_texts": response_texts,
                    "average_kl": avg_kl_model2,
                    "response_kls_model2_avg": response_kls_model2_avg,
                    "response_kls_model2_per_token": response_kls_model2_per_token,
                }
            )

        # Sort results by average KL divergence (highest first)
        results.sort(key=lambda x: x["average_kl"], reverse=True)
        print(f"\nSorted {len(results)} results by average KL divergence")

        # Save results to JSON file
        output_file = os.path.join(config.output_dir, "kl_divergence_results_sorted.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        print(f"Total prompts processed: {len(results)}")

        # Print top 5 prompts with highest KL
        print("\nTop 5 prompts with highest KL divergence:")
        print("-" * 80)
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. Prompt {result['prompt_idx']}: {result['average_kl']:.6f}")
            print(f"   {result['prompt'][:100]}...")
            print()


if __name__ == "__main__":
    fire.Fire(main)
