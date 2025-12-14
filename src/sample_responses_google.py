# ABOUTME: Main script for sampling multiple LLM responses via Google Gemini API.
# ABOUTME: Loads prompts from datasets and samples responses with configurable parameters.

import argparse
import asyncio
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from google_client import GoogleClient
from prompts_loader import (
    PromptLoader,
    gpqa_prompt_extractor,
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
    response_logprobs: bool,
    logprobs: int,
    output_dir: str,
    project_id: Optional[str] = None,
    location: str = "global",
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
        response_logprobs: Whether to return log probabilities.
        logprobs: Number of top logprobs to return per token.
        output_dir: Directory to save JSON output files.
        project_id: Optional Google Cloud project ID override.
        location: Google Cloud region.
        max_retries: Maximum number of retry attempts for failed requests.
    """
    print(f"\n{'=' * 60}")
    print(f"Sampling responses for model: {model}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Samples per prompt: {num_samples}")
    print(f"{'=' * 60}\n")

    client = GoogleClient(project_id=project_id, location=location)

    samples = await client.sample_prompts_batch(
        prompts=prompts,
        model=model,
        num_samples_per_prompt=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        response_logprobs=response_logprobs,
        logprobs=logprobs,
        output_dir=output_dir,
        max_retries=max_retries,
    )

    print(f"✓ Completed sampling for {model}")


async def main():
    """Main function for sampling responses."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sample LLM responses via Google Gemini API"
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
        elif dataset_config.get("use_gpqa_format"):
            prompt_extractor = gpqa_prompt_extractor

        prompts = PromptLoader.load_from_huggingface(
            dataset_name=dataset_config["dataset_name"],
            split=dataset_config["split"],
            prompt_field=dataset_config.get("prompt_field")
            if not prompt_extractor
            else None,
            num_samples=dataset_config.get("num_prompts"),
            seed=dataset_config.get("seed"),
            min_length=dataset_config.get("min_length"),
            max_length=dataset_config.get("max_length"),
            filter_fn=filter_fn,
            prompt_extractor=prompt_extractor,
            sampling_mode=dataset_config.get("sampling_mode", "first"),
        )
    elif source == "file":
        prompts = PromptLoader.load_from_file(
            file_path=dataset_config["file_path"],
            num_samples=dataset_config.get("num_prompts"),
            seed=dataset_config.get("seed"),
            min_length=dataset_config.get("min_length"),
            max_length=dataset_config.get("max_length"),
            sampling_mode=dataset_config.get("sampling_mode", "first"),
        )
    else:
        raise ValueError(f"Unknown dataset source: {source}")

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
    project_id = api_config.get("project_id")
    location = api_config.get("location", "global")

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
            response_logprobs=sampling_config["response_logprobs"],
            logprobs=sampling_config["logprobs"],
            output_dir=str(output_dir),
            project_id=project_id,
            location=location,
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
