# ABOUTME: Test script to verify KL divergence calculation works correctly.
# ABOUTME: Creates synthetic test data and runs the KL metric calculation.

import math
import pickle
import tempfile
from pathlib import Path

from src.kl_divergence_metric import (
    calculate_kl_divergence_for_tokens,
    calculate_kl_metrics,
    extract_logprobs_from_response,
)


def create_test_response(tokens_with_logprobs):
    """Create a test response with specified tokens and logprobs."""
    content = []
    for token, logprob, top_logprobs in tokens_with_logprobs:
        token_data = {
            "token": token,
            "logprob": logprob,
        }
        if top_logprobs:
            token_data["top_logprobs"] = [
                {"token": t, "logprob": lp} for t, lp in top_logprobs
            ]
        content.append(token_data)

    return {
        "model": "test-model",
        "choices": [
            {
                "message": {"content": "Test response"},
                "finish_reason": "stop",
                "logprobs": {"content": content},
            }
        ],
    }


def test_kl_calculation():
    """Test KL divergence calculation with synthetic data."""
    print("=" * 80)
    print("Testing KL Divergence Calculation")
    print("=" * 80)

    # Create test tokens
    # Model 1: "Hello world" with high confidence
    tokens1 = [
        ("Hello", math.log(0.9), [("Hello", math.log(0.9)), ("Hi", math.log(0.05))]),
        (" world", math.log(0.85), [(" world", math.log(0.85)), (" there", math.log(0.1))]),
    ]

    # Model 2: Same tokens but different probabilities
    tokens2 = [
        ("Hello", math.log(0.5), [("Hello", math.log(0.5)), ("Hi", math.log(0.3))]),
        (" world", math.log(0.6), [(" world", math.log(0.6)), (" there", math.log(0.25))]),
    ]

    response1 = create_test_response(tokens1)
    response2 = create_test_response(tokens2)

    # Extract logprobs
    logprobs1 = extract_logprobs_from_response(response1)
    logprobs2 = extract_logprobs_from_response(response2)

    print(f"✓ Created test responses with {len(logprobs1)} tokens")

    # Calculate KL divergence
    kl = calculate_kl_divergence_for_tokens(logprobs1, logprobs2)
    print(f"✓ KL Divergence: {kl:.6f}")

    # Since model 1 has higher confidence (more certain distributions),
    # and model 2 has more spread out probabilities, we expect positive KL
    assert kl > 0, f"Expected positive KL divergence, got {kl}"
    print("✓ KL divergence is positive (as expected)")

    return True


def test_full_pipeline():
    """Test the full pipeline with pickle files."""
    print("\n" + "=" * 80)
    print("Testing Full Pipeline with Pickle Files")
    print("=" * 80)

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data for 3 prompts with 2 responses each
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        samples1 = []
        samples2 = []

        for prompt in prompts:
            # Create responses for model 1
            responses1 = [
                create_test_response([
                    ("Token", math.log(0.9), [("Token", math.log(0.9)), ("Word", math.log(0.05))]),
                    (" A", math.log(0.85), [(" A", math.log(0.85)), (" B", math.log(0.1))]),
                ]),
                create_test_response([
                    ("Token", math.log(0.88), [("Token", math.log(0.88)), ("Word", math.log(0.07))]),
                    (" A", math.log(0.82), [(" A", math.log(0.82)), (" B", math.log(0.12))]),
                ]),
            ]

            # Create responses for model 2 (different distributions)
            responses2 = [
                create_test_response([
                    ("Token", math.log(0.6), [("Token", math.log(0.6)), ("Word", math.log(0.25))]),
                    (" A", math.log(0.55), [(" A", math.log(0.55)), (" B", math.log(0.3))]),
                ]),
                create_test_response([
                    ("Token", math.log(0.65), [("Token", math.log(0.65)), ("Word", math.log(0.2))]),
                    (" A", math.log(0.58), [(" A", math.log(0.58)), (" B", math.log(0.28))]),
                ]),
            ]

            samples1.append({
                "prompt": prompt,
                "model": "model1",
                "responses": responses1,
            })

            samples2.append({
                "prompt": prompt,
                "model": "model2",
                "responses": responses2,
            })

        # Save to pickle files
        file1 = Path(tmpdir) / "model1.pkl"
        file2 = Path(tmpdir) / "model2.pkl"

        with open(file1, "wb") as f:
            pickle.dump(samples1, f)
        with open(file2, "wb") as f:
            pickle.dump(samples2, f)

        print(f"✓ Created test pickle files with {len(prompts)} prompts")

        # Run the full calculation
        results = calculate_kl_metrics(str(file1), str(file2), max_prompts=None)

        print(f"✓ Calculated metrics for {len(results)} prompts")

        # Verify results
        assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"
        print("✓ Correct number of results")

        # Check that all KL values are positive
        for result in results:
            assert result.kl_divergence > 0, f"Expected positive KL for prompt {result.prompt_index}"
        print("✓ All KL divergences are positive")

        # Display results
        print("\n" + "-" * 80)
        print("Results:")
        print("-" * 80)
        for result in results:
            print(f"Prompt {result.prompt_index}: KL = {result.kl_divergence:.6f}")

        return True


if __name__ == "__main__":
    print("\nRunning KL Divergence Tests\n")

    try:
        test_kl_calculation()
        test_full_pipeline()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 80)
        raise
