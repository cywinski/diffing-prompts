# ABOUTME: Sample multiple LLM responses for high KL divergence tokens using vLLM-served models.
# ABOUTME: Uses extra_body params for proper assistant prefill with vLLM's continue_final_message.

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

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


async def sample_for_entry(
    client: OpenRouterClient,
    entry: dict,
    entry_idx: int,
    model: str,
    num_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logprobs: bool,
    top_logprobs: int,
    output_dir: Path,
    reasoning: bool,
    max_retries: int,
) -> None:
    """Sample responses for a single high KL entry.

    Args:
        client: OpenRouter client instance.
        entry: High KL divergence entry dictionary.
        entry_idx: Index of the entry.
        model: Model identifier.
        num_samples: Number of samples to generate.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        logprobs: Whether to return log probabilities.
        top_logprobs: Number of top logprobs to return per token.
        output_dir: Directory to save output files.
        reasoning: Whether to enable reasoning.
        max_retries: Maximum number of retry attempts.
    """
    # Use prompt and prefix_response separately for assistant prefill
    prompt = entry["prompt"]
    assistant_prefill = entry["prefix_response"]

    # vLLM-specific extra_body for proper assistant prefill continuation
    extra_body = {
        "add_generation_prompt": False,
        "continue_final_message": True,
    }

    # Sample responses with assistant prefill
    responses = await client.sample_multiple_concurrent(
        prompt=prompt,
        model=model,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        reasoning=reasoning,
        max_retries=max_retries,
        assistant_prefill=assistant_prefill,
        extra_body=extra_body,
    )

    # Prepare output data
    output_data = {
        "entry_metadata": {
            "original_prompt": entry["prompt"],
            "prefix_response": entry["prefix_response"],
            "high_kl_token": entry["token"],
            "kl_divergence": entry["kl_divergence"],
            "source_file": entry["source_file"],
            "response_idx": entry["response_idx"],
            "token_position": entry["token_position"],
        },
        "sampling_config": {
            "model": model,
            "num_samples": num_samples,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "prompt": prompt,
        "assistant_prefill": assistant_prefill,
        "responses": responses,
    }

    # Save to file
    model_name = model.replace("/", "_").replace(":", "_")
    filename = f"{model_name}_entry_{entry_idx}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


async def main():
    """Main function for sampling responses for high KL divergence entries."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sample LLM responses for high KL divergence tokens (vLLM version)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to high_kl_tokens.json file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to process (for testing)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load high KL tokens data
    input_path = Path(args.input)
    with open(input_path, "r") as f:
        high_kl_data = json.load(f)

    print(f"Loaded {high_kl_data['num_entries']} entries from {input_path}")

    # Filter out infinite KL divergence entries
    entries = [
        entry
        for entry in high_kl_data["entries"]
        if entry["kl_divergence"] != float("inf")
    ]

    print(f"Filtered to {len(entries)} entries (removed infinite KL values)")

    # Limit entries if requested (command line overrides config)
    max_entries = args.max_entries
    if max_entries is None and "entries" in config:
        max_entries = config["entries"].get("max_entries")

    if max_entries is not None:
        entries = entries[:max_entries]
        print(f"Limited to {len(entries)} entries for processing")

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(config["output"]["base_dir"]) / config["output"]["experiment_name"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get configuration
    sampling_config = config["sampling"]
    model = config["model"]
    api_config = config.get("api", {})
    base_url = api_config.get("base_url")

    # Create OpenRouter client
    client = OpenRouterClient(base_url=base_url)

    print(f"\n{'=' * 60}")
    print(f"Sampling responses for model: {model}")
    print(f"Number of entries: {len(entries)}")
    print(f"Samples per entry: {sampling_config['num_samples_per_entry']}")
    print(
        "Using vLLM extra_body: add_generation_prompt=False, continue_final_message=True"
    )
    print(f"{'=' * 60}\n")

    # Process entries sequentially with progress bar
    for entry_idx, entry in enumerate(
        tqdm(entries, desc="Processing entries", unit="entry")
    ):
        await sample_for_entry(
            client=client,
            entry=entry,
            entry_idx=entry_idx,
            model=model,
            num_samples=sampling_config["num_samples_per_entry"],
            max_tokens=sampling_config["max_tokens"],
            temperature=sampling_config["temperature"],
            top_p=sampling_config["top_p"],
            logprobs=sampling_config["logprobs"],
            top_logprobs=sampling_config["top_logprobs"],
            output_dir=output_dir,
            reasoning=sampling_config["reasoning"],
            max_retries=sampling_config.get("max_retries", 5),
        )

    print(f"\n{'=' * 60}")
    print("All sampling complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
