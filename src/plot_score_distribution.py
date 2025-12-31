# ABOUTME: Plots the distribution of KL divergence and normalized KL (KL/(H1+H2)) from JSON files
# ABOUTME: Shows side-by-side histograms and box plots for easy comparison of both scores

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def collect_scores(kl_dir):
    """
    Collect KL and normalized scores (KL / (H1 + H2)) from JSON files in directory.

    Args:
        kl_dir: Path to directory containing *_kl.json files

    Returns:
        Tuple of (kl_scores, normalized_scores) lists
    """
    kl_path = Path(kl_dir)
    kl_scores = []
    normalized_scores = []

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

                if kl is not None:
                    kl_scores.append(kl)

                    # Calculate normalized score: KL / (H1 + H2)
                    if h1 is not None and h2 is not None and h1 + h2 > 0:
                        score = kl / (h1 + h2)
                        normalized_scores.append(score)

    return kl_scores, normalized_scores


def print_stats(name, scores):
    """Print statistics for a score array."""
    print(f"\n{name}:")
    print(f"  Total tokens: {len(scores):,}")
    print(f"  Mean: {np.mean(scores):.6f}")
    print(f"  Median: {np.median(scores):.6f}")
    print(f"  Std Dev: {np.std(scores):.6f}")
    print(f"  Min: {np.min(scores):.6f}")
    print(f"  Max: {np.max(scores):.6f}")
    print(f"  Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"    {p}th: {np.percentile(scores, p):.6f}")


def plot_distribution(kl_scores, normalized_scores, output_path=None):
    """
    Plot the distribution of KL and normalized scores side by side.

    Args:
        kl_scores: List of raw KL divergence scores
        normalized_scores: List of normalized scores (KL / (H1 + H2))
        output_path: Optional path to save the plot
    """
    kl_scores = np.array(kl_scores)
    normalized_scores = np.array(normalized_scores)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Row 1: Histograms
    axes[0, 0].hist(kl_scores, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].set_xlabel("KL Divergence")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of KL Divergence")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(normalized_scores, bins=100, edgecolor="black", alpha=0.7, color="coral")
    axes[0, 1].set_xlabel("KL / (H1 + H2)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Normalized KL")
    axes[0, 1].grid(True, alpha=0.3)

    # Row 2: Box plots
    bp1 = axes[1, 0].boxplot(kl_scores, vert=True, patch_artist=True)
    bp1["boxes"][0].set_facecolor("steelblue")
    axes[1, 0].set_ylabel("KL Divergence")
    axes[1, 0].set_title("Box Plot of KL Divergence")
    axes[1, 0].set_xticklabels(["KL"])
    axes[1, 0].grid(True, alpha=0.3)

    bp2 = axes[1, 1].boxplot(normalized_scores, vert=True, patch_artist=True)
    bp2["boxes"][0].set_facecolor("coral")
    axes[1, 1].set_ylabel("KL / (H1 + H2)")
    axes[1, 1].set_title("Box Plot of Normalized KL")
    axes[1, 1].set_xticklabels(["Normalized"])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Print statistics
    print_stats("KL Divergence", kl_scores)
    print_stats("Normalized KL (KL / (H1 + H2))", normalized_scores)

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
    kl_scores, normalized_scores = collect_scores(kl_dir)

    if not kl_scores:
        print("No scores found!")
        return

    plot_distribution(kl_scores, normalized_scores, output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
