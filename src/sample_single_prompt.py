# ABOUTME: Script for sampling multiple responses from a single prompt.
# ABOUTME: Saves all responses to a single JSON file.

import asyncio
import json
from pathlib import Path
from typing import Optional

import fire
import yaml
from dotenv import load_dotenv

from openrouter_client import OpenRouterClient


async def sample_single_prompt(
    config_path: str,
    output_path: Optional[str] = None,
) -> None:
    """Sample responses for a single prompt.

    Args:
        config_path: Path to YAML config file containing prompt and sampling params.
        output_path: Optional override for output file path.
    """
    load_dotenv()

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    prompt = config["prompt"]
    model = config["model"]
    num_samples = config["num_samples"]
    max_tokens = config["max_tokens"]
    temperature = config["temperature"]
    top_p = config["top_p"]
    logprobs = config.get("logprobs", False)
    top_logprobs = config.get("top_logprobs", 0)
    reasoning = config.get("reasoning", False)
    max_retries = config.get("max_retries", 5)

    # Set up output path
    if output_path is None:
        output_path = config.get("output_path", "output.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Sampling responses for single prompt")
    print(f"Model: {model}")
    print(f"Number of samples: {num_samples}")
    print(f"{'=' * 60}\n")

    # Create client and sample
    client = OpenRouterClient()

    samples = await client.sample_prompts_batch(
        prompts=[prompt],
        model=model,
        num_samples_per_prompt=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        reasoning=reasoning,
        output_dir=None,  # Don't save per-prompt files
        max_retries=max_retries,
    )

    # Save all responses to single file
    result = {
        "prompt": prompt,
        "model": model,
        "config": {
            "num_samples": num_samples,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "reasoning": reasoning,
        },
        "responses": samples[0],  # All samples for the single prompt
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Sampling complete!")
    print(f"Saved {len(samples[0])} responses to: {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: asyncio.run(sample_single_prompt(**kwargs)))
