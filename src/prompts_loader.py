# ABOUTME: Generic prompt loader for loading prompts from various sources.
# ABOUTME: Supports HuggingFace datasets, text files, and Python lists with filtering/sampling.

import random
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset


class PromptLoader:
    """Generic loader for prompt datasets from various sources."""

    @staticmethod
    def load_from_huggingface(
        dataset_name: str,
        split: str = "train",
        prompt_field: str = "text",
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        prompt_extractor: Optional[Callable[[Dict[str, Any]], str]] = None,
        sampling_mode: str = "first",
    ) -> List[str]:
        """Load prompts from a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset identifier.
            split: Dataset split to load ('train', 'test', etc).
            prompt_field: Field name containing the prompt text.
                For nested fields, use dot notation (e.g., "conversation.0.content").
            num_samples: Number of prompts to sample. If None, returns all.
            seed: Random seed for shuffling/sampling (only used in 'random' mode).
            min_length: Minimum prompt length in characters.
            max_length: Maximum prompt length in characters.
            filter_fn: Optional custom filter function that takes an item and returns bool.
            prompt_extractor: Optional custom function to extract prompt from item.
                If provided, overrides prompt_field.
            sampling_mode: How to sample prompts - 'first' takes first N, 'random' samples randomly.

        Returns:
            List of prompt strings.
        """
        print(f"Loading dataset: {dataset_name} (split: {split})...")
        dataset = load_dataset(dataset_name, split=split)

        # Shuffle dataset if using random sampling mode
        if sampling_mode == "random" and seed is not None:
            print(f"Shuffling dataset with seed {seed}...")
            dataset = dataset.shuffle(seed=seed)
        elif sampling_mode == "random" and seed is None:
            print("Warning: 'random' sampling mode without seed - results may not be reproducible")

        prompts = []
        for item in dataset:
            # Extract prompt text
            if prompt_extractor:
                # Use custom extraction function
                try:
                    content = prompt_extractor(item)
                except Exception as e:
                    continue
            else:
                # Use field name (supports nested fields with dot notation)
                try:
                    content = item
                    for field in prompt_field.split("."):
                        if field.isdigit():
                            content = content[int(field)]
                        else:
                            content = content[field]
                    content = str(content)
                except (KeyError, IndexError, TypeError):
                    continue

            # Apply custom filter
            if filter_fn and not filter_fn(item):
                continue

            # Apply length filters
            if min_length and len(content) < min_length:
                continue
            if max_length and len(content) > max_length:
                continue

            prompts.append(content)

            # Stop when we have enough prompts
            if num_samples is not None and len(prompts) >= num_samples:
                break

        print(f"Loaded {len(prompts)} prompts from {dataset_name}")
        return prompts

    @staticmethod
    def load_from_file(
        file_path: str,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        sampling_mode: str = "first",
    ) -> List[str]:
        """Load prompts from a text file.

        Args:
            file_path: Path to text file.
            num_samples: Number of prompts to sample. If None, returns all.
            seed: Random seed for sampling (only used in 'random' mode).
            min_length: Minimum prompt length in characters.
            max_length: Maximum prompt length in characters.
            sampling_mode: How to sample prompts - 'first' takes first N, 'random' samples randomly.

        Returns:
            List of prompt strings.
        """
        with open(file_path, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        # Apply length filters
        if min_length or max_length:
            original_count = len(prompts)
            prompts = [
                p for p in prompts
                if (not min_length or len(p) >= min_length)
                and (not max_length or len(p) <= max_length)
            ]
            if len(prompts) < original_count:
                print(f"Filtered {original_count - len(prompts)} prompts by length")

        print(f"Total {len(prompts)} prompts from {file_path}")

        # Sample prompts based on mode
        if num_samples is not None and num_samples < len(prompts):
            if sampling_mode == "first":
                prompts = prompts[:num_samples]
                print(f"Took first {num_samples} prompts")
            elif sampling_mode == "random":
                if seed is not None:
                    random.seed(seed)
                else:
                    print("Warning: 'random' sampling mode without seed - results may not be reproducible")
                prompts = random.sample(prompts, num_samples)
                print(f"Randomly sampled {num_samples} prompts")

        return prompts

    @staticmethod
    def load_from_list(
        prompts: List[str],
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        sampling_mode: str = "first",
    ) -> List[str]:
        """Load prompts from a Python list.

        Args:
            prompts: List of prompt strings.
            num_samples: Number of prompts to sample. If None, returns all.
            seed: Random seed for sampling (only used in 'random' mode).
            sampling_mode: How to sample prompts - 'first' takes first N, 'random' samples randomly.

        Returns:
            List of prompt strings.
        """
        print(f"Total {len(prompts)} prompts from list")

        # Sample prompts based on mode
        if num_samples is not None and num_samples < len(prompts):
            if sampling_mode == "first":
                prompts = prompts[:num_samples]
                print(f"Took first {num_samples} prompts")
            elif sampling_mode == "random":
                if seed is not None:
                    random.seed(seed)
                else:
                    print("Warning: 'random' sampling mode without seed - results may not be reproducible")
                prompts = random.sample(prompts, num_samples)
                print(f"Randomly sampled {num_samples} prompts")

        return prompts


def wildchat_prompt_extractor(item: Dict[str, Any]) -> str:
    """Extract first user message from WildChat conversation format.

    Args:
        item: WildChat dataset item.

    Returns:
        First user message content.

    Raises:
        ValueError: If conversation format is invalid.
    """
    if "conversation" not in item or len(item["conversation"]) == 0:
        raise ValueError("No conversation found")

    conversation = item["conversation"][0]
    if conversation.get("role") == "user":
        return conversation.get("content")

    raise ValueError("No user message found")


def wildchat_language_filter(language: str) -> Callable[[Dict[str, Any]], bool]:
    """Create a filter function for WildChat language field.

    Args:
        language: Language code (e.g., 'en' for English).

    Returns:
        Filter function that returns True if item matches language.
    """
    def filter_fn(item: Dict[str, Any]) -> bool:
        return item.get("language") == language and item.get("turn") == 1
    return filter_fn
