# ABOUTME: Example script demonstrating KL divergence calculation workflow.
# ABOUTME: Shows how to load results and analyze KL divergence statistics.

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def load_kl_results(result_dir: str) -> List[Dict[str, Any]]:
    """Load all KL divergence result files from a directory.

    Args:
        result_dir: Directory containing KL divergence result files.

    Returns:
        List of result dictionaries.
    """
    result_path = Path(result_dir)
    result_files = sorted(result_path.glob("*_kl.json"))

    results = []
    for file_path in result_files:
        with open(file_path, "r") as f:
            results.append(json.load(f))

    return results


def analyze_kl_statistics(results: List[Dict[str, Any]]) -> None:
    """Analyze and print KL divergence statistics.

    Args:
        results: List of KL divergence result dictionaries.
    """
    all_mean_kls = []
    all_total_kls = []
    all_max_kls = []

    for result in results:
        for response in result.get("responses", []):
            if "statistics" in response:
                stats = response["statistics"]
                all_mean_kls.append(stats["mean_kl"])
                all_total_kls.append(stats["total_kl"])
                all_max_kls.append(stats["max_kl"])

    print("=" * 60)
    print("KL Divergence Statistics Summary")
    print("=" * 60)
    print(f"Model 1: {results[0]['model1']}")
    print(f"Model 2: {results[0]['model2']}")
    print(f"Total responses: {len(all_mean_kls)}")
    print()

    print("Mean KL per token (averaged across responses):")
    print(f"  Mean:   {np.mean(all_mean_kls):.4f}")
    print(f"  Median: {np.median(all_mean_kls):.4f}")
    print(f"  Std:    {np.std(all_mean_kls):.4f}")
    print(f"  Min:    {np.min(all_mean_kls):.4f}")
    print(f"  Max:    {np.max(all_mean_kls):.4f}")
    print()

    print("Total KL per sequence:")
    print(f"  Mean:   {np.mean(all_total_kls):.4f}")
    print(f"  Median: {np.median(all_total_kls):.4f}")
    print(f"  Std:    {np.std(all_total_kls):.4f}")
    print(f"  Min:    {np.min(all_total_kls):.4f}")
    print(f"  Max:    {np.max(all_total_kls):.4f}")
    print()

    print("Max KL per token (across sequence):")
    print(f"  Mean:   {np.mean(all_max_kls):.4f}")
    print(f"  Median: {np.median(all_max_kls):.4f}")
    print(f"  Std:    {np.std(all_max_kls):.4f}")
    print(f"  Min:    {np.min(all_max_kls):.4f}")
    print(f"  Max:    {np.max(all_max_kls):.4f}")
    print("=" * 60)


def find_most_divergent_responses(
    results: List[Dict[str, Any]], top_k: int = 5, metric: str = "total_kl"
) -> List[Dict[str, Any]]:
    """Find responses with highest KL divergence.

    Args:
        results: List of KL divergence result dictionaries.
        top_k: Number of top responses to return.
        metric: Metric to use for ranking ("total_kl", "mean_kl", or "max_kl").

    Returns:
        List of top-k most divergent responses with metadata.
    """
    all_responses = []

    for result in results:
        prompt = result.get("prompt", "")
        for response in result.get("responses", []):
            if "statistics" in response:
                all_responses.append(
                    {
                        "prompt": prompt,
                        "text": response.get("text", ""),
                        "statistics": response["statistics"],
                        "num_tokens": response.get("num_tokens", 0),
                    }
                )

    # Sort by selected metric
    sorted_responses = sorted(
        all_responses,
        key=lambda x: x["statistics"][metric],
        reverse=True,
    )

    return sorted_responses[:top_k]


def plot_kl_distributions(
    results: List[Dict[str, Any]], output_path: str = None
) -> None:
    """Plot distributions of KL divergence statistics.

    Args:
        results: List of KL divergence result dictionaries.
        output_path: Optional path to save plot.
    """
    all_mean_kls = []
    all_total_kls = []

    for result in results:
        for response in result.get("responses", []):
            if "statistics" in response:
                stats = response["statistics"]
                all_mean_kls.append(stats["mean_kl"])
                all_total_kls.append(stats["total_kl"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Mean KL distribution
    axes[0].hist(all_mean_kls, bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Mean KL per token")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Mean KL Divergence")
    axes[0].axvline(
        np.median(all_mean_kls),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(all_mean_kls):.3f}",
    )
    axes[0].legend()

    # Total KL distribution
    axes[1].hist(all_total_kls, bins=50, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Total KL per sequence")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Total KL Divergence")
    axes[1].axvline(
        np.median(all_total_kls),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(all_total_kls):.3f}",
    )
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    """Main function demonstrating KL divergence analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze KL divergence results")
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory containing KL divergence result files",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save plot (optional)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of most divergent responses to show",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.result_dir}...")
    results = load_kl_results(args.result_dir)
    print(f"Loaded {len(results)} result files\n")

    # Analyze statistics
    analyze_kl_statistics(results)
    print()

    # Find most divergent responses
    print(f"Top {args.top_k} most divergent responses (by total KL):")
    print("=" * 60)
    top_responses = find_most_divergent_responses(results, top_k=args.top_k)
    for i, response in enumerate(top_responses, 1):
        print(f"\n{i}. Total KL: {response['statistics']['total_kl']:.4f}")
        print(f"   Mean KL: {response['statistics']['mean_kl']:.4f}")
        print(f"   Tokens: {response['num_tokens']}")
        print(f"   Prompt: {response['prompt'][:100]}...")
        print(f"   Response: {response['text'][:100]}...")
    print("=" * 60)

    # Plot distributions
    if args.plot or True:  # Always show plot
        plot_path = args.plot if args.plot else None
        plot_kl_distributions(results, output_path=plot_path)


if __name__ == "__main__":
    main()
