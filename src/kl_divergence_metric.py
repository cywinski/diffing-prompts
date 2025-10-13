# ABOUTME: Engine for calculating KL divergence between model responses.
# ABOUTME: Measures token-level distribution differences between two models' outputs.

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class KLMetricResult:
    """Result of KL divergence calculation for a single prompt."""

    prompt: str
    prompt_index: int
    kl_divergence: float
    num_comparisons: int
    avg_response_length: float


def load_samples_from_pickle(file_path: str) -> List[Dict[str, Any]]:
    """Load samples from a pickle file.

    Args:
        file_path: Path to pickle file.

    Returns:
        List of sample dictionaries.
    """
    with open(file_path, "rb") as f:
        samples = pickle.load(f)
    return samples


def extract_logprobs_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract logprobs from a single response.

    Args:
        response: Response dictionary from OpenRouter API.

    Returns:
        List of token logprob dictionaries, each containing:
        - token: str
        - logprob: float
        - top_logprobs: List[Dict[str, Any]] (optional)
    """
    if "choices" not in response or not response["choices"]:
        return []

    choice = response["choices"][0]
    if "logprobs" not in choice or not choice["logprobs"]:
        return []

    if "content" not in choice["logprobs"]:
        return []

    return choice["logprobs"]["content"]


def calculate_kl_divergence_for_tokens(
    tokens_model1: List[Dict[str, Any]],
    tokens_model2: List[Dict[str, Any]],
) -> float:
    """Calculate KL divergence between two sequences of token logprobs.

    This calculates D_KL(P || Q) where:
    - P is the distribution from model1 (the response being evaluated)
    - Q is the distribution from model2 (the reference distribution)

    Args:
        tokens_model1: Token logprobs from model 1.
        tokens_model2: Token logprobs from model 2.

    Returns:
        Average KL divergence per token.
    """
    if not tokens_model1 or not tokens_model2:
        return 0.0

    # For each token in the sequence from model1, we need to find its logprob
    # in model2's distribution
    total_kl = 0.0
    num_tokens = 0

    for token_data_1, token_data_2 in zip(tokens_model1, tokens_model2):
        # Get the token and its logprob from model1
        token = token_data_1["token"]
        logprob_p = token_data_1["logprob"]

        # Find this token's logprob in model2's distribution
        # First check if it's the top token in model2
        logprob_q = None
        if token_data_2["token"] == token:
            logprob_q = token_data_2["logprob"]
        else:
            # Search in top_logprobs from model2
            if "top_logprobs" in token_data_2:
                for top_token in token_data_2["top_logprobs"]:
                    if top_token["token"] == token:
                        logprob_q = top_token["logprob"]
                        break

        # If token not found in model2's distribution, skip
        # (This is a limitation of only having top-k logprobs)
        if logprob_q is None:
            continue

        # Convert logprobs to probabilities
        p = math.exp(logprob_p)
        q = math.exp(logprob_q)

        # Calculate KL divergence for this token: p * log(p/q)
        if p > 0 and q > 0:
            kl_token = p * (math.log(p) - math.log(q))
            total_kl += kl_token
            num_tokens += 1

    if num_tokens == 0:
        return 0.0

    return total_kl / num_tokens


def calculate_kl_for_response_pair(
    response1: Dict[str, Any],
    response2: Dict[str, Any],
) -> float:
    """Calculate KL divergence for a pair of responses.

    Takes response from model1 and autoregressively evaluates it
    against model2's distribution.

    Args:
        response1: Response from model 1.
        response2: Response from model 2 (same prompt).

    Returns:
        Average KL divergence per token, normalized by response length.
    """
    tokens1 = extract_logprobs_from_response(response1)
    tokens2 = extract_logprobs_from_response(response2)

    if not tokens1 or not tokens2:
        return 0.0

    kl = calculate_kl_divergence_for_tokens(tokens1, tokens2)

    # Normalize by length
    response_length = len(tokens1)
    if response_length == 0:
        return 0.0

    return kl / response_length


def calculate_bidirectional_kl_for_response_pair(
    response1: Dict[str, Any],
    response2: Dict[str, Any],
) -> float:
    """Calculate bidirectional KL divergence for a pair of responses.

    Calculates KL in both directions and averages them.

    Args:
        response1: Response from model 1.
        response2: Response from model 2.

    Returns:
        Average of KL(model1 || model2) and KL(model2 || model1).
    """
    kl_1_to_2 = calculate_kl_for_response_pair(response1, response2)
    kl_2_to_1 = calculate_kl_for_response_pair(response2, response1)

    return (kl_1_to_2 + kl_2_to_1) / 2.0


def calculate_kl_metric_for_prompt(
    samples_model1: Dict[str, Any],
    samples_model2: Dict[str, Any],
) -> float:
    """Calculate KL divergence metric for a single prompt.

    Calculates KL divergence for all combinations of responses between
    two models and returns the average.

    Args:
        samples_model1: Sample dictionary for model 1 (contains prompt and responses).
        samples_model2: Sample dictionary for model 2 (same prompt).

    Returns:
        Average KL divergence across all response combinations.
    """
    responses1 = samples_model1["responses"]
    responses2 = samples_model2["responses"]

    if not responses1 or not responses2:
        return 0.0

    total_kl = 0.0
    num_comparisons = 0

    # Compare all combinations
    for resp1 in responses1:
        for resp2 in responses2:
            kl = calculate_bidirectional_kl_for_response_pair(resp1, resp2)
            total_kl += kl
            num_comparisons += 1

    if num_comparisons == 0:
        return 0.0

    return total_kl / num_comparisons


def calculate_kl_metrics(
    samples_file1: str,
    samples_file2: str,
    max_prompts: Optional[int] = None,
) -> List[KLMetricResult]:
    """Calculate KL divergence metrics for all prompts.

    Args:
        samples_file1: Path to pickle file with samples from model 1.
        samples_file2: Path to pickle file with samples from model 2.
        max_prompts: Maximum number of prompts to process (None for all).

    Returns:
        List of KLMetricResult objects, one per prompt.
    """
    samples1 = load_samples_from_pickle(samples_file1)
    samples2 = load_samples_from_pickle(samples_file2)

    if len(samples1) != len(samples2):
        raise ValueError(
            f"Number of prompts mismatch: {len(samples1)} vs {len(samples2)}"
        )

    # Verify prompts match
    for i, (s1, s2) in enumerate(zip(samples1, samples2)):
        if s1["prompt"] != s2["prompt"]:
            raise ValueError(f"Prompt mismatch at index {i}")

    # Limit number of prompts if requested
    if max_prompts is not None:
        samples1 = samples1[:max_prompts]
        samples2 = samples2[:max_prompts]

    results = []
    for i, (s1, s2) in enumerate(zip(samples1, samples2)):
        kl = calculate_kl_metric_for_prompt(s1, s2)

        # Calculate average response length
        all_responses = s1["responses"] + s2["responses"]
        avg_length = 0.0
        if all_responses:
            total_tokens = sum(
                len(extract_logprobs_from_response(resp)) for resp in all_responses
            )
            avg_length = total_tokens / len(all_responses)

        num_comparisons = len(s1["responses"]) * len(s2["responses"])

        results.append(
            KLMetricResult(
                prompt=s1["prompt"],
                prompt_index=i,
                kl_divergence=kl,
                num_comparisons=num_comparisons,
                avg_response_length=avg_length,
            )
        )

    return results
