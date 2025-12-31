# ABOUTME: Extract tokens with score above a threshold from saved JSON files
# ABOUTME: Score can be KL divergence or normalized KL (KL/(H1+H2))

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal

import fire


def extract_high_kl_tokens(
    results_dir: str,
    min_threshold: float,
    max_threshold: float = None,
    output_path: str = None,
    score_type: Literal["kl", "normalized"] = "kl",
) -> None:
    """
    Extract tokens with score within threshold interval from JSON files.

    Args:
        results_dir: Path to directory containing KL divergence JSON files
        min_threshold: Minimum score value to include
        max_threshold: Maximum score value to include (None for no upper limit)
        output_path: Path to save output JSON file (default: results_dir/high_kl_tokens.json)
        score_type: "kl" for raw KL divergence, "normalized" for KL/(H1+H2)
    """
    results_dir = Path(results_dir)

    if output_path is None:
        output_path = results_dir / "high_kl_tokens.json"
    else:
        output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    score_label = "KL/(H1+H2)" if score_type == "normalized" else "KL"
    if max_threshold is not None:
        print(f"Searching for tokens with {min_threshold} <= {score_label} <= {max_threshold} in {results_dir}")
    else:
        print(f"Searching for tokens with {score_label} >= {min_threshold} in {results_dir}")

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

                # Calculate score based on score_type
                if score_type == "normalized":
                    h1 = token_info.get("entropy1", 0)
                    h2 = token_info.get("entropy2", 0)
                    denominator = h1 + h2
                    if denominator <= 0:
                        score = 0.0
                    else:
                        score = kl_value / denominator
                else:
                    score = kl_value

                # Check if this token is within threshold interval
                if score >= min_threshold and (max_threshold is None or score <= max_threshold):
                    # Prefix is all tokens up to (but not including) this one
                    prefix = "".join(prefix_tokens)

                    entry = {
                        "prompt": prompt,
                        "prefix_response": prefix,
                        "token": token_info["token"],
                        "score": score,
                        "kl_divergence": kl_value,
                        "source_file": str(json_file),
                        "response_idx": response["response_idx"],
                        "token_position": token_info["position"],
                    }
                    if score_type == "normalized":
                        entry["entropy1"] = h1
                        entry["entropy2"] = h2
                    high_kl_entries.append(entry)

                # Add current token to prefix for next iteration
                prefix_tokens.append(token_info["token"])

    if max_threshold is not None:
        print(f"\nFound {len(high_kl_entries)} tokens with {min_threshold} <= {score_label} <= {max_threshold}")
    else:
        print(f"\nFound {len(high_kl_entries)} tokens with {score_label} >= {min_threshold}")

    # Sort by score (descending)
    high_kl_entries.sort(key=lambda x: x["score"], reverse=True)

    # Save to output file
    output_data = {
        "score_type": score_type,
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "num_entries": len(high_kl_entries),
        "entries": high_kl_entries,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {output_path}")

    # Print top 10 examples
    print(f"\nTop 10 highest {score_label} tokens:")
    for i, entry in enumerate(high_kl_entries[:10], 1):
        print(f"\n{i}. {score_label} = {entry['score']:.4f}")
        if score_type == "normalized":
            print(f"   KL = {entry['kl_divergence']:.4f}, H1 = {entry['entropy1']:.4f}, H2 = {entry['entropy2']:.4f}")
        print(f"   Token: '{entry['token']}'")
        print(f"   Prompt: {entry['prompt'][:80]}...")
        print(f"   Prefix: ...{entry['prefix_response'][-60:]}")


if __name__ == "__main__":
    fire.Fire(extract_high_kl_tokens)
