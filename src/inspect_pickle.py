#!/usr/bin/env python3
# ABOUTME: Simple utility to display the first N entries from a pickle file.
# ABOUTME: Supports pretty-printing of nested data structures with configurable depth.

import json
import pickle
from pathlib import Path

import fire


def inspect_pickle(
    pickle_path: str,
    num_entries: int = 5,
    max_depth: int = 3,
    show_responses: bool = True,
    show_logprobs: bool = False,
) -> None:
    """Display the first N entries from a pickle file.

    Args:
        pickle_path: Path to the pickle file.
        num_entries: Number of entries to display (default: 5).
        max_depth: Maximum depth for nested structure display (default: 3).
        show_responses: Whether to show full response content (default: True).
        show_logprobs: Whether to show logprobs data (default: False).
    """
    pickle_path = Path(pickle_path)

    if not pickle_path.exists():
        print(f"Error: File not found: {pickle_path}")
        return

    # Load pickle file
    print(f"Loading pickle file: {pickle_path}")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        print(f"Error: Expected list, got {type(data)}")
        return

    print(f"\nTotal entries in file: {len(data)}")
    print(f"Displaying first {min(num_entries, len(data))} entries\n")
    print("=" * 80)

    # Display first N entries
    for i, entry in enumerate(data[:num_entries]):
        print(f"\n[Entry {i+1}]")
        print("-" * 80)

        # Show prompt
        if "prompt" in entry:
            prompt_preview = entry["prompt"][:200]
            if len(entry["prompt"]) > 200:
                prompt_preview += "..."
            print(f"Prompt: {prompt_preview}")

        # Show model
        if "model" in entry:
            print(f"Model: {entry['model']}")

        # Show number of responses
        if "responses" in entry:
            responses = entry["responses"]
            print(f"Number of responses: {len(responses)}")

            if show_responses and responses:
                print("\nResponses:")
                for j, response in enumerate(responses[:3]):  # Show first 3 responses
                    print(f"\n  Response {j+1}:")

                    # Extract content from response
                    if "choices" in response and response["choices"]:
                        choice = response["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                            content_preview = content[:150]
                            if len(content) > 150:
                                content_preview += "..."
                            print(f"    Content: {content_preview}")

                        if "finish_reason" in choice:
                            print(f"    Finish reason: {choice['finish_reason']}")

                        # Show logprobs if requested
                        if show_logprobs and "logprobs" in choice and choice["logprobs"]:
                            if "content" in choice["logprobs"]:
                                num_tokens = len(choice["logprobs"]["content"])
                                print(f"    Logprobs: {num_tokens} tokens")
                                if num_tokens > 0:
                                    first_token = choice["logprobs"]["content"][0]
                                    print(f"      First token: {first_token.get('token', 'N/A')} "
                                          f"(logprob: {first_token.get('logprob', 'N/A'):.4f})")

                    # Show usage stats if present
                    if "usage" in response:
                        usage = response["usage"]
                        print(f"    Usage: {usage}")

                if len(responses) > 3:
                    print(f"\n  ... and {len(responses) - 3} more responses")

        print()

    print("=" * 80)
    print(f"\nShowing {min(num_entries, len(data))} of {len(data)} total entries")


if __name__ == "__main__":
    fire.Fire(inspect_pickle)
