# ABOUTME: Main script for sampling multiple LLM responses via OpenRouter API.
# ABOUTME: Loads prompts from datasets and samples responses with configurable parameters.

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from openrouter_client import OpenRouterClient, save_samples_to_json
from prompts_loader import (
    PromptLoader,
    wildchat_language_filter,
    wildchat_prompt_extractor,
)


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
    """
    print(f"\n{'='*60}")
    print(f"Sampling responses for model: {model}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Samples per prompt: {num_samples}")
    print(f"{'='*60}\n")

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
    )

    save_samples_to_json(samples, output_dir, model)
    print(f"✓ Completed sampling for {model}")


async def main():
    """Main function for sampling responses."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sample LLM responses via OpenRouter API"
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

    # Load prompts
    dataset_config = config["dataset"]
    source = dataset_config["source"]

    if source == "huggingface":
        # Build filter function if language is specified
        filter_fn = None
        language = dataset_config.get("language")
        if language:
            filter_fn = wildchat_language_filter(language)

        # Determine if we need a custom prompt extractor
        prompt_extractor = None
        if dataset_config.get("use_wildchat_format"):
            prompt_extractor = wildchat_prompt_extractor

        prompts = PromptLoader.load_from_huggingface(
            dataset_name=dataset_config["dataset_name"],
            split=dataset_config["split"],
            prompt_field=dataset_config.get("prompt_field") if not prompt_extractor else None,
            num_samples=dataset_config.get("num_prompts"),
            seed=dataset_config.get("seed"),
            min_length=dataset_config.get("min_length"),
            max_length=dataset_config.get("max_length"),
            filter_fn=filter_fn,
            prompt_extractor=prompt_extractor,
        )
    elif source == "file":
        prompts = PromptLoader.load_from_file(
            file_path=dataset_config["file_path"],
            num_samples=dataset_config.get("num_prompts"),
            seed=dataset_config.get("seed"),
            min_length=dataset_config.get("min_length"),
            max_length=dataset_config.get("max_length"),
        )
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    if not prompts:
        raise ValueError("No prompts loaded!")

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["output"]["base_dir"]) / config['output']['experiment_name']

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
        )
        tasks.append(task)

    # Run all model sampling concurrently
    print(f"\n{'='*60}")
    print(f"Sampling {len(models)} models concurrently...")
    print(f"{'='*60}\n")

    await asyncio.gather(*tasks)

    print(f"\n{'='*60}")
    print("✓ All sampling complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
