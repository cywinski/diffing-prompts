# ABOUTME: Main script for sampling multiple LLM responses from JSON prompts via OpenRouter API.
# ABOUTME: Loads prompts from JSON files and samples responses with configurable parameters.

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from openrouter_client import OpenRouterClient


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_prompts_from_json(
    json_path: str,
    prompt_field: Optional[str] = None,
    num_samples: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> list[str]:
    """Load prompts from a JSON file.

    Args:
        json_path: Path to JSON file containing prompts.
        prompt_field: Field name containing the prompt text. If None, assumes data is a list of strings.
        num_samples: Number of prompts to load. If None, loads all.
        min_length: Minimum prompt length in characters.
        max_length: Maximum prompt length in characters.

    Returns:
        List of prompt strings.
    """
    print(f"Loading prompts from: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    prompts = []

    # Handle direct list of strings
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        for prompt in data:
            # Apply length filters
            if min_length and len(prompt) < min_length:
                continue
            if max_length and len(prompt) > max_length:
                continue

            prompts.append(prompt)

            # Stop when we have enough prompts
            if num_samples is not None and len(prompts) >= num_samples:
                break
    else:
        # Handle list of objects or single object
        if isinstance(data, dict):
            data = [data]

        if prompt_field is None:
            raise ValueError(
                "prompt_field must be specified when data is not a list of strings"
            )

        for item in data:
            if prompt_field not in item:
                continue

            prompt = item[prompt_field]

            # Apply length filters
            if min_length and len(prompt) < min_length:
                continue
            if max_length and len(prompt) > max_length:
                continue

            prompts.append(prompt)

            # Stop when we have enough prompts
            if num_samples is not None and len(prompts) >= num_samples:
                break

    print(f"Loaded {len(prompts)} prompts from {json_path}")
    return prompts


async def sample_responses_for_model(
    prompts: list[str],
    model: str,
    num_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logprobs: bool,
    top_logprobs: int,
    output_dir: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    reasoning: bool = False,
    max_retries: int = 5,
) -> None:
    """Sample responses for a single model.

    Args:
        prompts: List of prompts to sample.
        model: Model identifier.
        num_samples: Number of samples per prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        logprobs: Whether to return log probabilities.
        top_logprobs: Number of top logprobs to return per token.
        output_dir: Directory to save JSON output files.
        api_key: Optional API key override.
        base_url: Optional API base URL override.
        reasoning: Whether to enable reasoning.
        max_retries: Maximum number of retry attempts for failed requests.
    """
    print(f"\n{'=' * 60}")
    print(f"Sampling responses for model: {model}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Samples per prompt: {num_samples}")
    print(f"{'=' * 60}\n")

    client = OpenRouterClient(api_key=api_key, base_url=base_url)

    samples = await client.sample_prompts_batch(
        prompts=prompts,
        model=model,
        num_samples_per_prompt=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        reasoning=reasoning,
        output_dir=output_dir,
        max_retries=max_retries,
    )

    print(f"✓ Completed sampling for {model}")


async def main():
    """Main function for sampling responses."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sample LLM responses from JSON prompts via OpenRouter API"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load prompts from JSON
    dataset_config = config["dataset"]
    prompts = load_prompts_from_json(
        json_path=dataset_config["json_path"],
        prompt_field=dataset_config.get("prompt_field"),
        num_samples=dataset_config.get("num_prompts"),
        min_length=dataset_config.get("min_length"),
        max_length=dataset_config.get("max_length"),
    )

    if not prompts:
        raise ValueError("No prompts loaded!")

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(config["output"]["base_dir"]) / config["output"]["experiment_name"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample responses for each model concurrently
    sampling_config = config["sampling"]
    models = config["models"]
    api_config = config.get("api", {})
    base_url = api_config.get("base_url")

    # Create tasks for all models to run concurrently
    tasks = []
    for model_id in models:
        task = sample_responses_for_model(
            prompts=prompts,
            model=model_id,
            num_samples=sampling_config["num_samples_per_prompt"],
            max_tokens=sampling_config["max_tokens"],
            temperature=sampling_config["temperature"],
            top_p=sampling_config["top_p"],
            logprobs=sampling_config["logprobs"],
            top_logprobs=sampling_config["top_logprobs"],
            output_dir=str(output_dir),
            base_url=base_url,
            reasoning=sampling_config["reasoning"],
            max_retries=sampling_config.get("max_retries", 5),
        )
        tasks.append(task)

    # Run all model sampling concurrently
    print(f"\n{'=' * 60}")
    print(f"Sampling {len(models)} models concurrently...")
    print(f"{'=' * 60}\n")

    await asyncio.gather(*tasks)

    print(f"\n{'=' * 60}")
    print("✓ All sampling complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
