# ABOUTME: Script to calculate perplexity of responses from two models using a third evaluation model.
# ABOUTME: Loads responses from both models and evaluates them with an independent model to measure quality/fluency.

import json
import os
from typing import Optional

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

from kl_divergence import calculate_perplexity, load_model_and_tokenizer, load_response_files


def main(config_path: str, max_prompts: Optional[int] = None):
    """Calculate perplexity of responses using an evaluation model.

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

    # Load evaluation model
    eval_model, eval_tokenizer = load_model_and_tokenizer(
        config.eval_model_name, config.device_map
    )

    # Load response files from both models
    print("\nLoading response files from model 1...")
    model1_files_by_idx = load_response_files(config.responses_dir_model1)
    print(f"Found {len(model1_files_by_idx)} files for model 1")

    print("\nLoading response files from model 2...")
    model2_files_by_idx = load_response_files(config.responses_dir_model2)
    print(f"Found {len(model2_files_by_idx)} files for model 2")

    # Get common prompt indices
    common_indices = set(model1_files_by_idx.keys()) & set(model2_files_by_idx.keys())
    prompt_indices = sorted(common_indices)

    print(f"\nFound {len(prompt_indices)} prompts with responses from both models")

    # Limit number of prompts if specified
    if max_prompts is not None:
        prompt_indices = prompt_indices[:max_prompts]
        print(f"Limiting to {max_prompts} prompts")

    # Calculate perplexity for each prompt
    results = []

    for prompt_idx in tqdm(prompt_indices, desc="Processing prompts"):
        model1_file = model1_files_by_idx[prompt_idx]
        model2_file = model2_files_by_idx[prompt_idx]

        # Load response data from both models
        with open(model1_file, "r") as fp:
            data_model1 = json.load(fp)

        with open(model2_file, "r") as fp:
            data_model2 = json.load(fp)

        prompt = data_model1["prompt"]

        # Verify prompts match
        if data_model1["prompt"] != data_model2["prompt"]:
            print(f"Warning: Prompts don't match for index {prompt_idx}, skipping")
            continue

        # Process responses from model 1
        model1_perplexities = []
        model1_log_probs_per_token = []
        model1_response_texts = []

        for response_data in data_model1["responses"]:
            # Extract text response and tokenize using eval tokenizer
            response_text = response_data["choices"][0]["message"]["content"]
            model1_response_texts.append(response_text)

            # Tokenize the response text
            token_ids = eval_tokenizer.encode(response_text, add_special_tokens=False)

            try:
                # Calculate perplexity using eval model
                perplexity, log_probs = calculate_perplexity(
                    prompt=prompt,
                    response_token_ids=token_ids,
                    model=eval_model,
                    tokenizer=eval_tokenizer,
                )
                model1_perplexities.append(perplexity)
                model1_log_probs_per_token.append(log_probs)
            except Exception as e:
                print(f"Error processing model 1, prompt {prompt_idx}, response: {e}")
                continue

        # Process responses from model 2
        model2_perplexities = []
        model2_log_probs_per_token = []
        model2_response_texts = []

        for response_data in data_model2["responses"]:
            # Extract text response and tokenize using eval tokenizer
            response_text = response_data["choices"][0]["message"]["content"]
            model2_response_texts.append(response_text)

            # Tokenize the response text
            token_ids = eval_tokenizer.encode(response_text, add_special_tokens=False)

            try:
                # Calculate perplexity using eval model
                perplexity, log_probs = calculate_perplexity(
                    prompt=prompt,
                    response_token_ids=token_ids,
                    model=eval_model,
                    tokenizer=eval_tokenizer,
                )
                model2_perplexities.append(perplexity)
                model2_log_probs_per_token.append(log_probs)
            except Exception as e:
                print(f"Error processing model 2, prompt {prompt_idx}, response: {e}")
                continue

        if len(model1_perplexities) == 0 and len(model2_perplexities) == 0:
            print(f"Skipped prompt {prompt_idx} (no valid responses)")
            continue

        # Calculate average perplexities
        avg_perplexity_model1 = (
            sum(model1_perplexities) / len(model1_perplexities)
            if model1_perplexities
            else None
        )
        avg_perplexity_model2 = (
            sum(model2_perplexities) / len(model2_perplexities)
            if model2_perplexities
            else None
        )

        results.append(
            {
                "prompt_idx": prompt_idx,
                "file_model1": model1_file,
                "file_model2": model2_file,
                "prompt": prompt,
                "model1_response_texts": model1_response_texts,
                "model2_response_texts": model2_response_texts,
                "model1_perplexities": model1_perplexities,
                "model2_perplexities": model2_perplexities,
                "model1_log_probs_per_token": model1_log_probs_per_token,
                "model2_log_probs_per_token": model2_log_probs_per_token,
                "avg_perplexity_model1": avg_perplexity_model1,
                "avg_perplexity_model2": avg_perplexity_model2,
            }
        )

    # Sort results by average perplexity difference (model1 - model2)
    # Negative values mean model1 has lower perplexity (better)
    results_with_diff = []
    for r in results:
        if r["avg_perplexity_model1"] is not None and r["avg_perplexity_model2"] is not None:
            perplexity_diff = r["avg_perplexity_model1"] - r["avg_perplexity_model2"]
            r["perplexity_diff"] = perplexity_diff
            results_with_diff.append(r)

    results_with_diff.sort(key=lambda x: abs(x["perplexity_diff"]), reverse=True)
    print(f"\nProcessed {len(results)} prompts")

    # Save results to JSON file
    output_file = os.path.join(config.output_dir, "perplexity_results_sorted.json")
    with open(output_file, "w") as f:
        json.dump(results_with_diff, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Total prompts processed: {len(results_with_diff)}")

    # Print top 5 prompts with largest perplexity differences
    print("\nTop 5 prompts with largest perplexity differences:")
    print("-" * 80)
    for i, result in enumerate(results_with_diff[:5], 1):
        print(f"{i}. Prompt {result['prompt_idx']}")
        print(f"   Model 1 avg perplexity: {result['avg_perplexity_model1']:.4f}")
        print(f"   Model 2 avg perplexity: {result['avg_perplexity_model2']:.4f}")
        print(f"   Difference (M1-M2): {result['perplexity_diff']:.4f}")
        print(f"   Prompt: {result['prompt'][:100]}...")
        print()


if __name__ == "__main__":
    fire.Fire(main)
