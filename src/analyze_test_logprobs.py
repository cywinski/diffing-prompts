# ABOUTME: Calculate and print KL divergence and entropy for test logprobs files.
# ABOUTME: Compares token probability distributions between two models/prompts.

import json
import math
from pathlib import Path
from typing import Any, Dict, List


def normalize_logprob_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize different logprob formats to consistent structure."""
    normalized = []
    for item in data:
        # Handle both logProbability and logprob field names
        logprob = item.get("logprob") or item.get("logProbability")
        token_id = item.get("token_id") or item.get("tokenId")

        normalized.append(
            {"token": item["token"], "logprob": logprob, "token_id": token_id}
        )
    return normalized


def calculate_entropy(top_logprobs: List[Dict[str, Any]]) -> float:
    """Calculate entropy from top-k logprobs.

    Computes entropy over normalized top-k distribution:
        H = -Σ p_i log(p_i)
    """
    if not top_logprobs:
        return 0.0

    probs = []
    total = 0.0
    for item in top_logprobs:
        logprob = item.get("logprob")
        if logprob is None:
            continue
        p = math.exp(logprob)
        probs.append(p)
        total += p

    if total <= 0.0 or not probs:
        return 0.0

    entropy = 0.0
    for p in probs:
        pn = p / total
        if pn > 0.0:
            entropy -= pn * math.log(pn)

    return entropy


def build_distribution(top_logprobs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Build normalized probability distribution from top logprobs."""
    if not top_logprobs:
        return {}

    floor_logprob = min(item["logprob"] for item in top_logprobs)

    dist = {}
    for item in top_logprobs:
        token = item["token"]
        logprob = item["logprob"]
        prob = math.exp(logprob)
        dist[token] = prob

    total_prob = sum(dist.values())
    if total_prob > 0:
        dist = {token: prob / total_prob for token, prob in dist.items()}

    floor_prob = math.exp(floor_logprob) / total_prob if total_prob > 0 else 1e-10
    dist["__floor__"] = floor_prob

    return dist


def calculate_kl_divergence(
    p_dist: Dict[str, float], q_dist: Dict[str, float]
) -> float:
    """Calculate KL divergence KL(P||Q)."""
    if not p_dist or not q_dist:
        return float("inf")

    p_floor = p_dist.get("__floor__", 1e-10)
    q_floor = q_dist.get("__floor__", 1e-10)

    all_tokens = set(p_dist.keys()) | set(q_dist.keys())
    all_tokens.discard("__floor__")

    kl_div = 0.0
    for token in all_tokens:
        p = p_dist.get(token, p_floor)
        q = q_dist.get(token, q_floor)

        if p > 0 and q > 0:
            kl_div += p * math.log(p / q)

    return kl_div


def analyze_logprobs(file1: str, file2: str):
    """Analyze and compare logprobs from two files."""
    # Load files
    with open(file1, "r") as f:
        data1 = json.load(f)

    with open(file2, "r") as f:
        data2 = json.load(f)

    # Normalize formats
    logprobs1 = normalize_logprob_format(data1)
    logprobs2 = normalize_logprob_format(data2)

    # Calculate entropy for each distribution
    entropy1 = calculate_entropy(logprobs1)
    entropy2 = calculate_entropy(logprobs2)

    # Build probability distributions
    dist1 = build_distribution(logprobs1)
    dist2 = build_distribution(logprobs2)

    # Calculate KL divergences (both directions)
    kl_1_to_2 = calculate_kl_divergence(dist1, dist2)
    kl_2_to_1 = calculate_kl_divergence(dist2, dist1)

    # Print results
    print("=" * 80)
    print("LOGPROBS ANALYSIS")
    print("=" * 80)
    print(f"\nFile 1: {file1}")
    print(f"  Tokens: {len(logprobs1)}")
    print(f"  Entropy: {entropy1:.4f} nats")
    print(
        f"  Top token: '{logprobs1[0]['token']}' (logprob: {logprobs1[0]['logprob']:.4f})"
    )

    print(f"\nFile 2: {file2}")
    print(f"  Tokens: {len(logprobs2)}")
    print(f"  Entropy: {entropy2:.4f} nats")
    print(
        f"  Top token: '{logprobs2[0]['token']}' (logprob: {logprobs2[0]['logprob']:.4f})"
    )

    print("\n" + "-" * 80)
    print("KL DIVERGENCE")
    print("-" * 80)
    print(f"KL(File1 || File2): {kl_1_to_2:.6f} nats")
    print(f"KL(File2 || File1): {kl_2_to_1:.6f} nats")
    print(f"Symmetric KL:       {(kl_1_to_2 + kl_2_to_1) / 2:.6f} nats")

    # Show top tokens comparison
    print("\n" + "-" * 80)
    print("TOP TOKENS COMPARISON")
    print("-" * 80)
    print(
        f"{'Rank':<6} {'File 1 Token':<20} {'Prob':<12} {'File 2 Token':<20} {'Prob':<12}"
    )
    print("-" * 80)

    max_len = max(len(logprobs1), len(logprobs2))
    for i in range(min(10, max_len)):  # Show top 10
        # Get token and prob from file 1
        if i < len(logprobs1):
            token1 = logprobs1[i]["token"]
            # Get normalized prob
            prob1 = dist1.get(token1, 0.0)
        else:
            token1 = "-"
            prob1 = 0.0

        # Get token and prob from file 2
        if i < len(logprobs2):
            token2 = logprobs2[i]["token"]
            prob2 = dist2.get(token2, 0.0)
        else:
            token2 = "-"
            prob2 = 0.0

        print(f"{i + 1:<6} {token1:<20} {prob1:<12.6f} {token2:<20} {prob2:<12.6f}")

    print("=" * 80)


def main(file1: str = "test_logprobs1.json", file2: str = "test_logprobs2.json"):
    """Calculate KL divergence and entropy between two logprob files.

    Args:
        file1: Path to first logprobs JSON file
        file2: Path to second logprobs JSON file
    """
    analyze_logprobs(file1, file2)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
