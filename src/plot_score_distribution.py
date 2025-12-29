# ABOUTME: Plots the distribution of normalized KL divergence scores (KL/(H1+H2)) from JSON files
# ABOUTME: containing token-level KL divergence and entropy data across multiple model responses

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def collect_scores(kl_dir):
    """
    Collect all normalized scores (KL / (H1 + H2)) from JSON files in directory.

    Args:
        kl_dir: Path to directory containing *_kl.json files

    Returns:
        List of normalized scores for all tokens across all files
    """
    kl_path = Path(kl_dir)
    scores = []

    json_files = list(kl_path.glob("*_kl.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Process each response in the file
        for response in data.get("responses", []):
            token_details = response.get("token_details", [])

            for token_info in token_details:
                kl = token_info.get("kl_divergence")
                h1 = token_info.get("entropy1")
                h2 = token_info.get("entropy2")

                # Calculate normalized score: KL / (H1 + H2)
                if kl is not None and h1 is not None and h2 is not None:
                    if h1 + h2 > 0:  # Avoid division by zero
                        score = kl / (h1 + h2)
                        scores.append(score)

    return scores


def plot_distribution(scores, output_path=None):
    """
    Plot the distribution of normalized scores.

    Args:
        scores: List of normalized scores
        output_path: Optional path to save the plot
    """
    scores = np.array(scores)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    axes[0, 0].hist(scores, bins=100, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Score (KL / (H1 + H2))")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Normalized KL Scores")
    axes[0, 0].grid(True, alpha=0.3)

    # Log-scale histogram
    axes[0, 1].hist(scores, bins=100, edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Score (KL / (H1 + H2))")
    axes[0, 1].set_ylabel("Frequency (log scale)")
    axes[0, 1].set_title("Distribution of Normalized KL Scores (Log Scale)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1, 0].plot(sorted_scores, cumulative)
    axes[1, 0].set_xlabel("Score (KL / (H1 + H2))")
    axes[1, 0].set_ylabel("Cumulative Probability")
    axes[1, 0].set_title("Cumulative Distribution of Normalized KL Scores")
    axes[1, 0].grid(True, alpha=0.3)

    # Box plot
    axes[1, 1].boxplot(scores, vert=True)
    axes[1, 1].set_ylabel("Score (KL / (H1 + H2))")
    axes[1, 1].set_title("Box Plot of Normalized KL Scores")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Print statistics
    print(f"\nStatistics:")
    print(f"Total tokens: {len(scores):,}")
    print(f"Mean: {np.mean(scores):.6f}")
    print(f"Median: {np.median(scores):.6f}")
    print(f"Std Dev: {np.std(scores):.6f}")
    print(f"Min: {np.min(scores):.6f}")
    print(f"Max: {np.max(scores):.6f}")
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(scores, p):.6f}")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")

    plt.show()


def main(kl_dir, output_path=None):
    """
    Main function to collect scores and plot distribution.

    Args:
        kl_dir: Path to directory containing *_kl.json files
        output_path: Optional path to save the plot (default: None, shows plot only)
    """
    print(f"Collecting scores from: {kl_dir}")
    scores = collect_scores(kl_dir)

    if not scores:
        print("No scores found!")
        return

    plot_distribution(scores, output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
