# ABOUTME: Script to run LLM judge comparing responses from two models and generating hypotheses.
# ABOUTME: Loads paired JSON files, formats comparison prompts, and outputs hypothesis analysis.

import asyncio
import json
from pathlib import Path
from typing import Optional

import fire
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


def load_prompt_template(template_path: str) -> str:
    """Load prompt template from file.

    Args:
        template_path: Path to prompt template file.

    Returns:
        Prompt template string.
    """
    with open(template_path, "r") as f:
        return f.read()


def format_judge_prompt(
    template: str,
    user_prompt: str,
    assistant_prefill: str,
    model_a_responses: list[str],
    model_b_responses: list[str],
    num_hypotheses: int,
) -> str:
    """Format the judge prompt using template and data.

    Args:
        template: Prompt template string.
        user_prompt: Original user prompt.
        assistant_prefill: Assistant prefill text.
        model_a_responses: List of Model A responses.
        model_b_responses: List of Model B responses.
        num_hypotheses: Number of hypotheses to generate.

    Returns:
        Formatted prompt string.
    """
    # Format Model A responses
    model_a_text = "\n".join(
        [f"{i + 1}. {resp}" for i, resp in enumerate(model_a_responses)]
    )

    # Format Model B responses
    model_b_text = "\n".join(
        [f"{i + 1}. {resp}" for i, resp in enumerate(model_b_responses)]
    )

    # Replace template variables
    prompt = template.replace("{USER_PROMPT}", user_prompt)
    prompt = prompt.replace("{ASSISTANT_PREFILL}", assistant_prefill)
    prompt = prompt.replace("{MODEL_A_RESPONSES}", model_a_text)
    prompt = prompt.replace("{MODEL_B_RESPONSES}", model_b_text)
    prompt = prompt.replace("{NUM_HYPOTHESES}", str(num_hypotheses))

    return prompt


def load_json_file(file_path: Path) -> dict:
    """Load JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def extract_responses(data: dict) -> list[str]:
    """Extract response texts from JSON data.

    Args:
        data: Parsed JSON data.

    Returns:
        List of response texts.
    """
    responses = []
    for response in data["responses"]:
        content = response["choices"][0]["message"]["content"]
        # Replace llama tokenizer special characters with actual characters
        content = content.replace("Ġ", " ").replace("Ċ", "\n")
        responses.append(content)
    return responses


async def judge_response_pair(
    model_a_file: Path,
    model_b_file: Path,
    prompt_template: str,
    num_hypotheses: int,
    judge_model: str,
    judge_max_tokens: int,
    judge_temperature: float,
    client: OpenRouterClient,
    output_dir: Path,
) -> dict:
    """Run LLM judge on a pair of response files.

    Args:
        model_a_file: Path to Model A responses JSON.
        model_b_file: Path to Model B responses JSON.
        prompt_template: Prompt template string.
        num_hypotheses: Number of hypotheses to generate.
        judge_model: Model to use for judging.
        judge_max_tokens: Maximum tokens for judge response.
        judge_temperature: Temperature for judge model.
        client: OpenRouter client instance.
        output_dir: Directory to save output.

    Returns:
        Dictionary with judgment results.
    """
    # Load data from both files
    model_a_data = load_json_file(model_a_file)
    model_b_data = load_json_file(model_b_file)

    # Extract responses
    model_a_responses = extract_responses(model_a_data)
    model_b_responses = extract_responses(model_b_data)

    # Get prompt and prefill from Model A data
    user_prompt = model_a_data["prompt"]
    assistant_prefill = model_a_data["assistant_prefill"]
    # Replace llama tokenizer special characters
    assistant_prefill = assistant_prefill.replace("Ġ", " ").replace("Ċ", "\n")

    # Format the judge prompt
    judge_prompt = format_judge_prompt(
        template=prompt_template,
        user_prompt=user_prompt,
        assistant_prefill=assistant_prefill,
        model_a_responses=model_a_responses,
        model_b_responses=model_b_responses,
        num_hypotheses=num_hypotheses,
    )

    # Call judge model
    response = await client.sample_response(
        prompt=judge_prompt,
        model=judge_model,
        max_tokens=judge_max_tokens,
        temperature=judge_temperature,
        top_p=1.0,
        logprobs=False,
        top_logprobs=0,
    )

    judge_response_raw = response["choices"][0]["message"]["content"]

    # Parse JSON from judge response
    parsed_hypotheses = None
    parse_error = None
    try:
        # Extract JSON from response (handle markdown code blocks)
        response_text = judge_response_raw.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith("```"):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove trailing ```
        response_text = response_text.strip()

        parsed_hypotheses = json.loads(response_text)
    except json.JSONDecodeError as e:
        parse_error = str(e)
        print(f"⚠ Warning: Failed to parse JSON response: {e}")

    # Prepare output data in readable format
    output_data = {
        "comparison_metadata": {
            "model_a_file": str(model_a_file),
            "model_b_file": str(model_b_file),
            "judge_model": judge_model,
            "num_hypotheses_requested": num_hypotheses,
        },
        "input": {
            "user_prompt": user_prompt,
            "assistant_prefill": assistant_prefill,
            "model_a_metadata": model_a_data["entry_metadata"],
            "model_b_metadata": model_b_data["entry_metadata"],
        },
        "responses": {
            "model_a": model_a_responses,
            "model_b": model_b_responses,
        },
        "hypotheses": parsed_hypotheses if parsed_hypotheses else None,
        "raw_judge_response": judge_response_raw,
        "parse_error": parse_error,
    }

    # Save output
    output_file = output_dir / f"{model_a_file.stem}_vs_{model_b_file.stem}_judge.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Completed: {model_a_file.name} vs {model_b_file.name}")

    return output_data


async def run_judge(
    model_a_dir: Optional[str] = None,
    model_b_dir: Optional[str] = None,
    config_path: str = None,
    output_dir: Optional[str] = None,
) -> None:
    """Run LLM judge on all pairs of response files.

    Args:
        model_a_dir: Directory containing Model A responses (overrides config).
        model_b_dir: Directory containing Model B responses (overrides config).
        config_path: Path to YAML configuration file.
        output_dir: Optional output directory (overrides config).
    """
    # Load environment variables
    load_dotenv()

    # Load configuration
    config = load_config(config_path)

    # Load prompt template
    prompt_template = load_prompt_template(config["judge"]["prompt_template"])

    # Set up output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = (
            Path(config["output"]["base_dir"]) / config["output"]["experiment_name"]
        )
    output_path.mkdir(parents=True, exist_ok=True)

    # Get model directories from config or args
    if model_a_dir is None:
        model_a_dir = config["input"]["model_a_dir"]
    if model_b_dir is None:
        model_b_dir = config["input"]["model_b_dir"]

    # Get file lists
    model_a_path = Path(model_a_dir)
    model_b_path = Path(model_b_dir)

    model_a_files = sorted(model_a_path.glob("*.json"))
    model_b_files = sorted(model_b_path.glob("*.json"))

    print(f"\n{'=' * 60}")
    print(f"Found {len(model_a_files)} files in Model A directory")
    print(f"Found {len(model_b_files)} files in Model B directory")
    print(f"{'=' * 60}\n")

    # Match files by entry number
    file_pairs = []
    for a_file in model_a_files:
        # Extract entry number from filename
        entry_num = a_file.stem.split("_entry_")[-1]

        # Find matching file in Model B
        matching_b_files = [f for f in model_b_files if f"_entry_{entry_num}" in f.name]

        if matching_b_files:
            file_pairs.append((a_file, matching_b_files[0]))
        else:
            print(f"⚠ No matching file for {a_file.name}")

    print(f"Found {len(file_pairs)} matching pairs\n")

    # Create OpenRouter client
    client = OpenRouterClient(
        api_key=config.get("api", {}).get("api_key"),
        base_url=config.get("api", {}).get("base_url"),
    )

    # Process all pairs
    judge_config = config["judge"]
    tasks = []

    for model_a_file, model_b_file in file_pairs:
        task = judge_response_pair(
            model_a_file=model_a_file,
            model_b_file=model_b_file,
            prompt_template=prompt_template,
            num_hypotheses=judge_config["num_hypotheses"],
            judge_model=judge_config["model"],
            judge_max_tokens=judge_config["max_tokens"],
            judge_temperature=judge_config["temperature"],
            client=client,
            output_dir=output_path,
        )
        tasks.append(task)

    # Run all judgments concurrently
    print(f"\n{'=' * 60}")
    print(f"Running judge on {len(tasks)} pairs...")
    print(f"{'=' * 60}\n")

    await asyncio.gather(*tasks)

    print(f"\n{'=' * 60}")
    print("✓ All judgments complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}\n")


def main(
    config: str,
    model_a_dir: Optional[str] = None,
    model_b_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Main entry point for LLM judge script.

    Args:
        config: Path to YAML configuration file.
        model_a_dir: Directory containing Model A responses (overrides config).
        model_b_dir: Directory containing Model B responses (overrides config).
        output_dir: Optional output directory (overrides config).
    """
    asyncio.run(run_judge(model_a_dir, model_b_dir, config, output_dir))


if __name__ == "__main__":
    fire.Fire(main)
