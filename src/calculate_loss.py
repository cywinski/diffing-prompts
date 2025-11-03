# ABOUTME: Script to calculate cross-entropy loss per token for a model given ground truth responses from another model.
# ABOUTME: Loads responses as ground truth, computes loss with evaluation model, and saves sorted results to JSON.

import json
import os
from typing import Optional

import fire
import torch
import transformers
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

from kl_divergence import (
    load_model_and_tokenizer,
    load_response_files,
)


def compute_loss_per_token(
    prompt: str,
    response_text: str,
    model,
    tokenizer: transformers.PreTrainedTokenizerFast,
    reasoning_text: str = None,
    thinking_token_start: str = "<think>",
    thinking_token_end: str = "</think>",
):
    """Compute cross-entropy loss per token for a model given ground truth response.

    Args:
        prompt: The input prompt text
        response_text: The ground truth response text (content)
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        reasoning_text: Optional reasoning trace to include before response
        thinking_token_start: Start token for reasoning (default: "<think>")
        thinking_token_end: End token for reasoning (default: "</think>")

    Returns:
        Tuple of (loss_per_token as list, average_loss as float, entropy_per_token as list)
    """
    # Format response with reasoning if provided
    if reasoning_text is not None:
        full_response_text = f"{reasoning_text}{thinking_token_end}{response_text}"
    else:
        full_response_text = response_text

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

    # Tokenize response
    response_token_ids = tokenizer.encode(
        full_response_text,
        add_special_tokens=False,
    )

    # Combine prompt and response tokens
    tokens = torch.cat([user_prompt_tokens, torch.tensor(response_token_ids)])
    tokens = tokens.to(model.device)

    # Get log probabilities from model
    with torch.no_grad():
        outputs = model(tokens.unsqueeze(0))
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Extract log probs for the response tokens
        # log_probs[0, i] corresponds to prediction for token i+1
        # We want log_probs for tokens at positions len(user_prompt_tokens) to len(tokens)-1
        log_probs = log_probs[0, len(user_prompt_tokens) - 1 : -1].cpu()

    # Calculate loss per token (negative log probability of ground truth token)
    response_tokens = torch.tensor(response_token_ids)
    loss_per_token = -log_probs[range(len(response_tokens)), response_tokens]

    # Calculate entropy per token
    probs = torch.exp(log_probs)
    entropy_per_token = -(probs * log_probs).sum(dim=-1)

    return (
        loss_per_token.tolist(),
        loss_per_token.mean().item(),
        entropy_per_token.tolist(),
    )


def main(config_path: str, max_prompts: Optional[int] = None):
    """Calculate cross-entropy loss for an evaluation model using ground truth responses.

    Args:
        config_path: Path to YAML config file
        max_prompts: Maximum number of prompts to process (None for all)
    """
    load_dotenv()

    # Load configuration
    config = OmegaConf.load(config_path)
    print(f"Loaded config from: {config_path}")
    print(OmegaConf.to_yaml(config))

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Get reasoning config
    include_reasoning = config.get("include_reasoning", False)
    thinking_token_start = config.get("thinking_token_start", "<think>")
    thinking_token_end = config.get("thinking_token_end", "</think>")

    # Get tokenizer name (default to model name if not specified)
    tokenizer_eval_name = config.get("tokenizer_eval_name", None)

    # Get device map
    device_map = config.get("device_map", "auto")
    print(f"Device map for evaluation model: {device_map}")

    # Load evaluation model (the model we're evaluating)
    model_eval, tokenizer_eval = load_model_and_tokenizer(
        config.model_eval_name, device_map, tokenizer_name=tokenizer_eval_name
    )

    # Load response files from ground truth model
    gt_files_by_idx = load_response_files(config.responses_dir_ground_truth)

    # Limit number of prompts if specified
    prompt_indices = sorted(gt_files_by_idx.keys())
    if max_prompts is not None:
        prompt_indices = prompt_indices[:max_prompts]
        print(f"Limiting to {max_prompts} prompts")

    results = []

    for prompt_idx in tqdm(prompt_indices, desc="Processing prompts"):
        file_gt = gt_files_by_idx[prompt_idx]

        # Load ground truth responses
        with open(file_gt, "r") as fp:
            data_gt = json.load(fp)

        prompt = data_gt["prompt"]

        # Process responses: calculate loss for evaluation model
        response_losses_avg = []
        response_losses_per_token = []
        response_entropy_per_token = []
        response_texts = []

        for response_data in data_gt["responses"]:
            response_text = response_data["choices"][0]["message"]["content"]
            reasoning_text = None
            if include_reasoning:
                reasoning_text = response_data["choices"][0]["message"].get("reasoning")

            try:
                loss_per_token, avg_loss, entropy_per_token = compute_loss_per_token(
                    prompt=prompt,
                    response_text=response_text,
                    model=model_eval,
                    tokenizer=tokenizer_eval,
                    reasoning_text=reasoning_text,
                    thinking_token_start=thinking_token_start,
                    thinking_token_end=thinking_token_end,
                )
                response_texts.append(response_text)
                response_losses_avg.append(avg_loss)
                response_losses_per_token.append(loss_per_token)
                response_entropy_per_token.append(entropy_per_token)
            except Exception as e:
                print(f"Error processing prompt {prompt_idx}, response: {e}")
                continue

        if len(response_losses_avg) == 0:
            print(f"Skipped prompt {prompt_idx} (no valid responses)")
            continue

        # Average over responses
        avg_loss = sum(response_losses_avg) / len(response_losses_avg)

        results.append(
            {
                "prompt_idx": prompt_idx,
                "file_ground_truth": file_gt,
                "prompt": prompt,
                "response_texts": response_texts,
                "average_loss": avg_loss,
                "response_losses_avg": response_losses_avg,
                "response_losses_per_token": response_losses_per_token,
                "response_entropy_per_token": response_entropy_per_token,
            }
        )

    # Sort results by average loss (highest first)
    results.sort(key=lambda x: x["average_loss"], reverse=True)
    print(f"\nSorted {len(results)} results by average loss")

    # Save results to JSON file
    output_file = os.path.join(config.output_dir, "loss_results_sorted.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Total prompts processed: {len(results)}")

    # Print top 5 prompts with highest loss
    print("\nTop 5 prompts with highest average loss:")
    print("-" * 80)
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. Prompt {result['prompt_idx']}: Loss={result['average_loss']:.6f}")
        print(f"   {result['prompt'][:100]}...")
        print()


if __name__ == "__main__":
    fire.Fire(main)
