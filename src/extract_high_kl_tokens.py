# ABOUTME: Extract tokens with KL divergence above a threshold from saved JSON files
# ABOUTME: Outputs sorted JSON file with prompt, prefix, token, KL value, and source file path

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import fire


def extract_high_kl_tokens(
    results_dir: str, kl_threshold: float, output_path: str = None
) -> None:
    """
    Extract tokens with KL divergence exceeding threshold from JSON files.

    Args:
        results_dir: Path to directory containing KL divergence JSON files
        kl_threshold: Minimum KL divergence value to include
        output_path: Path to save output JSON file (default: results_dir/high_kl_tokens.json)
    """
    results_dir = Path(results_dir)

    if output_path is None:
        output_path = results_dir / "high_kl_tokens.json"
    else:
        output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    print(f"Searching for tokens with KL > {kl_threshold} in {results_dir}")

    # Collect all high-KL tokens
    high_kl_entries: List[Dict[str, Any]] = []

    # Process all JSON files in directory
    json_files = sorted(results_dir.glob("*_kl.json"))
    print(f"Found {len(json_files)} JSON files to process")

    for json_file in json_files:
        print(f"Processing {json_file.name}...")

        with open(json_file, "r") as f:
            data = json.load(f)

        prompt = data["prompt"]

        # Process each response
        for response in data["responses"]:
            tokens = response["token_details"]

            # Build prefix incrementally and check each token
            prefix_tokens = []

            for i, token_info in enumerate(tokens):
                kl_value = token_info["kl_divergence"]

                # Check if this token exceeds threshold
                if kl_value >= kl_threshold:
                    # Prefix is all tokens up to (but not including) this one
                    prefix = "".join(prefix_tokens)

                    entry = {
                        "prompt": prompt,
                        "prefix_response": prefix,
                        "token": token_info["token"],
                        "kl_divergence": kl_value,
                        "source_file": str(json_file),
                        "response_idx": response["response_idx"],
                        "token_position": token_info["position"],
                    }
                    high_kl_entries.append(entry)

                # Add current token to prefix for next iteration
                prefix_tokens.append(token_info["token"])

    print(f"\nFound {len(high_kl_entries)} tokens with KL > {kl_threshold}")

    # Sort by KL divergence (descending)
    high_kl_entries.sort(key=lambda x: x["kl_divergence"], reverse=True)

    # Save to output file
    output_data = {
        "kl_threshold": kl_threshold,
        "num_entries": len(high_kl_entries),
        "entries": high_kl_entries,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {output_path}")

    # Print top 10 examples
    print("\nTop 10 highest KL tokens:")
    for i, entry in enumerate(high_kl_entries[:10], 1):
        print(f"\n{i}. KL = {entry['kl_divergence']:.4f}")
        print(f"   Token: '{entry['token']}'")
        print(f"   Prompt: {entry['prompt'][:80]}...")
        print(f"   Prefix: ...{entry['prefix_response'][-60:]}")


if __name__ == "__main__":
    fire.Fire(extract_high_kl_tokens)
