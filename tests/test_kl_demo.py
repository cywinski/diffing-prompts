# ABOUTME: Test script for KL divergence calculation (DEPRECATED - see kl_demo.py).
# ABOUTME: This test file is for the old pickle-based approach, no longer in use.

# NOTE: This test file is deprecated. The current implementation in src/kl_metric.py
# uses a different approach that requires model inference and works with JSON files
# rather than pickle files. See kl_demo.py for the current implementation.
#
# The old approach (tested here) compared API logprobs directly without model inference.
# The new approach (in src/kl_metric.py) recalculates logprobs using local model inference
# and compares them with saved API logprobs.
#
# To test the new implementation, run kl_demo.py or notebooks/kl_demo_notebook.py
# with actual model inference.

import math
import pickle
import tempfile
from pathlib import Path


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


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEPRECATED TEST FILE")
    print("=" * 80)
    print("\nThis test file is for the old pickle-based KL divergence approach.")
    print("The current implementation uses model inference (see src/kl_metric.py).")
    print("\nTo test the current implementation, run:")
    print("  - kl_demo.py")
    print("  - notebooks/kl_demo_notebook.py")
    print("=" * 80 + "\n")
