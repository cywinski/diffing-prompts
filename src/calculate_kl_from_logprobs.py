# ABOUTME: Script to calculate KL divergence from stored top-k logprobs data in response files.
# ABOUTME: No model loading required - processes pre-computed logprobs for both models and calculates KL divergence.

import json
import os
from typing import Optional

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

from kl_divergence import calculate_kl_divergence_from_logprobs, load_response_files


def main(config_path: str, max_prompts: Optional[int] = None):
    """Calculate KL divergence from stored logprobs data.

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

    # Calculate KL divergence for each prompt
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

        # Process each response pair
        # Assume we're comparing response i from model1 with response i from model2
        num_responses = min(
            len(data_model1["responses"]), len(data_model2["responses"])
        )

        response_kls_avg = []
        response_kls_per_token = []
        response_texts_model1 = []
        response_texts_model2 = []

        for response_idx in range(num_responses):
            response_data_model1 = data_model1["responses"][response_idx]
            response_data_model2 = data_model2["responses"][response_idx]

            # Extract response texts
            response_text_model1 = response_data_model1["choices"][0]["message"][
                "content"
            ]
            response_text_model2 = response_data_model2["choices"][0]["message"][
                "content"
            ]
            response_texts_model1.append(response_text_model1)
            response_texts_model2.append(response_text_model2)

            # Extract logprobs data from model 1
            logprobs_content_model1 = response_data_model1["choices"][0]["logprobs"][
                "content"
            ]

            # Extract logprobs data from model 2
            logprobs_content_model2 = response_data_model2["choices"][0]["logprobs"][
                "content"
            ]

            # Verify token counts match (they should since both models tokenize the same way)
            if len(logprobs_content_model1) != len(logprobs_content_model2):
                print(
                    f"Warning: Token count mismatch for prompt {prompt_idx}, response {response_idx}, skipping"
                )
                continue

            # Parse logprobs for both models
            model_1_logprobs_list = []
            model_2_logprobs_list = []

            for token_data_m1, token_data_m2 in zip(
                logprobs_content_model1, logprobs_content_model2
            ):
                # Extract top-k logprobs for model 1
                model_1_top = token_data_m1.get("top_logprobs", [])
                model_1_logprobs = {
                    item["token"]: item["logprob"] for item in model_1_top
                }
                model_1_logprobs_list.append(model_1_logprobs)

                # Extract top-k logprobs for model 2
                model_2_top = token_data_m2.get("top_logprobs", [])
                model_2_logprobs = {
                    item["token"]: item["logprob"] for item in model_2_top
                }
                model_2_logprobs_list.append(model_2_logprobs)

            try:
                # Calculate KL(model_1 || model_2)
                avg_kl, per_token_kls = calculate_kl_divergence_from_logprobs(
                    model_1_logprobs_list=model_1_logprobs_list,
                    model_2_logprobs_list=model_2_logprobs_list,
                )
                response_kls_avg.append(avg_kl)
                response_kls_per_token.append(per_token_kls)
            except Exception as e:
                print(f"Error processing prompt {prompt_idx}, response {response_idx}: {e}")
                continue

        if len(response_kls_avg) == 0:
            print(f"Skipped prompt {prompt_idx} (no valid responses)")
            continue

        # Average over all responses for this prompt
        avg_kl = sum(response_kls_avg) / len(response_kls_avg)

        results.append(
            {
                "prompt_idx": prompt_idx,
                "file_model1": model1_file,
                "file_model2": model2_file,
                "prompt": prompt,
                "response_texts_model1": response_texts_model1,
                "response_texts_model2": response_texts_model2,
                "average_kl": avg_kl,
                "response_kls_avg": response_kls_avg,
                "response_kls_per_token": response_kls_per_token,
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
