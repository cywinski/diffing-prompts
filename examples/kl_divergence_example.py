# ABOUTME: Example script demonstrating KL divergence calculation from test logprobs files.
# ABOUTME: Calculates and visualizes KL divergence between two token probability distributions.

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Add src directory to path to import kl_divergence module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kl_divergence import calculate_kl_divergence_from_logprobs


def load_logprobs(file_path: str) -> List[Dict[str, float]]:
    """Load logprobs from a JSON file.

    Args:
        file_path: Path to JSON file containing logprobs.

    Returns:
        List of dictionaries with token and logprob.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def convert_to_logprobs_dict(logprobs_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Convert list of logprob dicts to a single dict mapping token -> logprob.

    Args:
        logprobs_list: List of dicts with 'token' and 'logprob' keys.

    Returns:
        Dictionary mapping token -> logprob.
    """
    return {item["token"]: item["logprob"] for item in logprobs_list}


def calculate_entropy(logprobs_dict: Dict[str, float]) -> float:
    """Calculate entropy from logprobs distribution.

    Args:
        logprobs_dict: Dictionary mapping token -> logprob.

    Returns:
        Entropy in nats.
    """
    probs = [math.exp(logprob) for logprob in logprobs_dict.values()]
    total = sum(probs)

    if total <= 0:
        return 0.0

    entropy = 0.0
    for prob in probs:
        p_norm = prob / total
        if p_norm > 0:
            entropy -= p_norm * math.log(p_norm)

    return entropy


def analyze_logprobs_comparison(
    file1: str, file2: str, model1_name: str = "Model 1", model2_name: str = "Model 2"
) -> Dict:
    """Analyze and compare logprobs from two files.

    Args:
        file1: Path to first logprobs file.
        file2: Path to second logprobs file.
        model1_name: Name for first model.
        model2_name: Name for second model.

    Returns:
        Dictionary containing analysis results.
    """
    # Load logprobs
    logprobs1_list = load_logprobs(file1)
    logprobs2_list = load_logprobs(file2)

    # Convert to dict format
    logprobs1_dict = convert_to_logprobs_dict(logprobs1_list)
    logprobs2_dict = convert_to_logprobs_dict(logprobs2_list)

    # Calculate KL divergence using the function from kl_divergence module
    # The function expects a list of dicts (one per token position), so we wrap in a list
    avg_kl, kl_per_token = calculate_kl_divergence_from_logprobs(
        model_1_logprobs_list=[logprobs1_dict],
        model_2_logprobs_list=[logprobs2_dict],
    )

    # Calculate entropies
    entropy1 = calculate_entropy(logprobs1_dict)
    entropy2 = calculate_entropy(logprobs2_dict)

    return {
        "file1": file1,
        "file2": file2,
        "model1_name": model1_name,
        "model2_name": model2_name,
        "logprobs1_list": logprobs1_list,
        "logprobs2_list": logprobs2_list,
        "logprobs1_dict": logprobs1_dict,
        "logprobs2_dict": logprobs2_dict,
        "kl_divergence": avg_kl,
        "entropy1": entropy1,
        "entropy2": entropy2,
    }


def print_analysis(results: Dict) -> None:
    """Print KL divergence analysis results.

    Args:
        results: Dictionary containing analysis results.
    """
    print("=" * 80)
    print("KL DIVERGENCE ANALYSIS")
    print("=" * 80)
    print(f"\n{results['model1_name']}: {results['file1']}")
    print(f"  Tokens in top-k: {len(results['logprobs1_list'])}")
    print(f"  Entropy: {results['entropy1']:.6f} nats")
    if results["logprobs1_list"]:
        top_token1 = results["logprobs1_list"][0]
        print(
            f"  Top token: '{top_token1['token']}' "
            f"(logprob: {top_token1['logprob']:.6f})"
        )

    print(f"\n{results['model2_name']}: {results['file2']}")
    print(f"  Tokens in top-k: {len(results['logprobs2_list'])}")
    print(f"  Entropy: {results['entropy2']:.6f} nats")
    if results["logprobs2_list"]:
        top_token2 = results["logprobs2_list"][0]
        print(
            f"  Top token: '{top_token2['token']}' "
            f"(logprob: {top_token2['logprob']:.6f})"
        )

    print("\n" + "-" * 80)
    print("KL DIVERGENCE")
    print("-" * 80)
    print(
        f"KL({results['model1_name']} || {results['model2_name']}): "
        f"{results['kl_divergence']:.6f} nats"
    )

    # Show top tokens comparison
    print("\n" + "-" * 80)
    print("TOP TOKENS COMPARISON")
    print("-" * 80)
    print(
        f"{'Rank':<6} {results['model1_name']:<20} {'Prob':<12} "
        f"{results['model2_name']:<20} {'Prob':<12}"
    )
    print("-" * 80)

    max_len = max(len(results["logprobs1_list"]), len(results["logprobs2_list"]))
    for i in range(min(10, max_len)):
        # Get token and prob from file 1
        if i < len(results["logprobs1_list"]):
            token1 = results["logprobs1_list"][i]["token"]
            logprob1 = results["logprobs1_list"][i]["logprob"]
            prob1 = math.exp(logprob1)
            # Normalize
            total1 = sum(
                math.exp(item["logprob"]) for item in results["logprobs1_list"]
            )
            prob1_norm = prob1 / total1 if total1 > 0 else 0.0
        else:
            token1 = "-"
            prob1_norm = 0.0

        # Get token and prob from file 2
        if i < len(results["logprobs2_list"]):
            token2 = results["logprobs2_list"][i]["token"]
            logprob2 = results["logprobs2_list"][i]["logprob"]
            prob2 = math.exp(logprob2)
            # Normalize
            total2 = sum(
                math.exp(item["logprob"]) for item in results["logprobs2_list"]
            )
            prob2_norm = prob2 / total2 if total2 > 0 else 0.0
        else:
            token2 = "-"
            prob2_norm = 0.0

        print(
            f"{i + 1:<6} {token1:<20} {prob1_norm:<12.6f} "
            f"{token2:<20} {prob2_norm:<12.6f}"
        )

    print("=" * 80)


def plot_distributions(results: Dict, output_path: str = None) -> None:
    """Plot probability distributions comparison.

    Args:
        results: Dictionary containing analysis results.
        output_path: Optional path to save plot.
    """
    logprobs1_list = results["logprobs1_list"]
    logprobs2_list = results["logprobs2_list"]

    # Get all unique tokens
    all_tokens = set(item["token"] for item in logprobs1_list) | set(
        item["token"] for item in logprobs2_list
    )
    all_tokens = sorted(all_tokens)

    # Calculate normalized probabilities
    total1 = sum(math.exp(item["logprob"]) for item in logprobs1_list)
    total2 = sum(math.exp(item["logprob"]) for item in logprobs2_list)

    probs1 = {
        item["token"]: math.exp(item["logprob"]) / total1 for item in logprobs1_list
    }
    probs2 = {
        item["token"]: math.exp(item["logprob"]) / total2 for item in logprobs2_list
    }

    # Get top 15 tokens by average probability
    avg_probs = {
        token: (probs1.get(token, 0) + probs2.get(token, 0)) / 2 for token in all_tokens
    }
    top_tokens = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)[:15]
    top_tokens = [token for token, _ in top_tokens]

    # Prepare data for plotting
    tokens_plot = top_tokens
    probs1_plot = [probs1.get(token, 0) for token in tokens_plot]
    probs2_plot = [probs2.get(token, 0) for token in tokens_plot]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tokens_plot))
    width = 0.35

    ax.bar(x - width / 2, probs1_plot, width, label=results["model1_name"], alpha=0.7)
    ax.bar(x + width / 2, probs2_plot, width, label=results["model2_name"], alpha=0.7)

    ax.set_xlabel("Token")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Probability Distributions Comparison\n"
        f"KL Divergence: {results['kl_divergence']:.6f} nats"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tokens_plot, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {output_path}")
    else:
        plt.show()


def main():
    """Main function demonstrating KL divergence analysis from test logprobs files."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze KL divergence from test logprobs files"
    )
    parser.add_argument(
        "--file1",
        type=str,
        default="test_logprobs1.json",
        help="Path to first logprobs JSON file",
    )
    parser.add_argument(
        "--file2",
        type=str,
        default="test_logprobs2.json",
        help="Path to second logprobs JSON file",
    )
    parser.add_argument(
        "--model1-name",
        type=str,
        default="Model 1",
        help="Name for first model",
    )
    parser.add_argument(
        "--model2-name",
        type=str,
        default="Model 2",
        help="Name for second model",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save plot (optional, shows plot if not provided)",
    )

    args = parser.parse_args()

    # Analyze logprobs
    print(f"Loading logprobs from {args.file1} and {args.file2}...")
    results = analyze_logprobs_comparison(
        args.file1, args.file2, args.model1_name, args.model2_name
    )

    # Print analysis
    print_analysis(results)

    # Plot distributions
    plot_distributions(results, output_path=args.plot)


if __name__ == "__main__":
    main()
