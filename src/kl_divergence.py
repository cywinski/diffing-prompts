# ABOUTME: Core functions for calculating KL divergence between two models' probability distributions.
# ABOUTME: Contains utilities for loading models, processing responses, and computing full-vocabulary KL divergence.

import glob
import math
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, device_map: str = "auto", attn_implementation: str = "eager"):
    """Load a model and its tokenizer.

    Args:
        model_name: HuggingFace model name or path
        device_map: Device mapping strategy (default: "auto")

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    print(f"Model {model_name} loaded successfully")
    return model, tokenizer


def calculate_kl_divergence_full_vocab(
    prompt: str,
    response_text: str,
    model_1,
    tokenizer_1,
    model_2,
    reasoning_text: str = None,
    thinking_token_start: str = "<think>",
    thinking_token_end: str = "</think>",
) -> Tuple[float, List[float]]:
    """Calculate KL divergence between two models' full vocabulary distributions.

    Calculates KL divergence for responses: KL(model_1 || model_2)

    Args:
        prompt: The input prompt text
        response_text: The response text (content)
        model_1: First model (P distribution)
        tokenizer_1: Tokenizer for first model
        model_2: Second model (Q distribution)
        reasoning_text: Optional reasoning trace to include before response
        thinking_token_start: Start token for reasoning (default: "<think>")
        thinking_token_end: End token for reasoning (default: "</think>")

    Returns:
        Tuple of (average KL divergence per token, list of per-token KLs)
    """
    # Format response with reasoning if provided
    if reasoning_text is not None:
        full_response_text = (
            f"{thinking_token_start}{reasoning_text}{thinking_token_end}{response_text}"
        )
    else:
        full_response_text = response_text

    # Prepare prompt with chat template
    user_prompt = tokenizer_1.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
        add_bos=False,
    )
    print(f"User prompt: {user_prompt}")
    user_prompt_tokens = tokenizer_1.encode(
        user_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )[0, :]

    # Tokenize response
    response_token_ids = tokenizer_1.encode(
        full_response_text,
        add_special_tokens=False,
    )

    # Combine prompt and response tokens
    tokens = torch.cat([user_prompt_tokens, torch.tensor(response_token_ids)])
    tokens = tokens.to(model_1.device)

    # Get full vocabulary log probabilities from model 1
    with torch.no_grad():
        outputs_1 = model_1(tokens.unsqueeze(0))
        logits_1 = outputs_1.logits
        log_probs_1 = torch.log_softmax(logits_1, dim=-1)
        # Extract log probs for the response tokens
        log_probs_1 = log_probs_1[0, len(user_prompt_tokens) - 1 : -1].cpu()

    # Get full vocabulary log probabilities from model 2
    with torch.no_grad():
        outputs_2 = model_2(tokens.unsqueeze(0))
        logits_2 = outputs_2.logits
        log_probs_2 = torch.log_softmax(logits_2, dim=-1)
        # Extract log probs for the response tokens
        log_probs_2 = log_probs_2[0, len(user_prompt_tokens) - 1 : -1].cpu()

    # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
    # In log space: KL(P||Q) = sum(exp(log_P) * (log_P - log_Q))
    # Using PyTorch's kl_div: expects input=log(Q), target=log(P)
    # F.kl_div computes sum(exp(target) * (target - input))
    kl_per_token = torch.nn.functional.kl_div(
        log_probs_2,  # Q distribution (model 2)
        log_probs_1,  # P distribution (model 1)
        reduction="none",
        log_target=True,
    ).sum(dim=-1)  # Sum over vocabulary dimension

    # Average over tokens
    avg_kl = kl_per_token.mean().item()

    return avg_kl, kl_per_token.tolist()


def calculate_kl_divergence_from_logprobs(
    model_1_logprobs_list: List[Dict[str, float]],
    model_2_logprobs_list: List[Dict[str, float]],
) -> Tuple[float, List[float]]:
    """Calculate KL divergence from stored top-k logprobs data.

    Calculates KL divergence: KL(model_1 || model_2)
    Uses only the top-k tokens available in the logprobs data.

    Args:
        model_1_logprobs_list: List of dicts mapping tokens to logprobs for model 1 (P distribution)
        model_2_logprobs_list: List of dicts mapping tokens to logprobs for model 2 (Q distribution)

    Returns:
        Tuple of (average KL divergence per token, list of per-token KLs)
    """
    if len(model_1_logprobs_list) != len(model_2_logprobs_list):
        raise ValueError(
            f"Mismatched lengths: model_1 has {len(model_1_logprobs_list)} tokens, "
            f"model_2 has {len(model_2_logprobs_list)} tokens"
        )

    kl_per_token = []

    for model_1_logprobs, model_2_logprobs in zip(
        model_1_logprobs_list, model_2_logprobs_list
    ):
        # Get union of all tokens from both models' top-k
        all_tokens = set(model_1_logprobs.keys()) | set(model_2_logprobs.keys())

        # Calculate KL divergence for this token position
        # KL(P||Q) = sum_x P(x) * log(P(x)/Q(x))
        # In log space: KL(P||Q) = sum_x exp(log_P(x)) * (log_P(x) - log_Q(x))
        kl = 0.0

        for token in all_tokens:
            # Get logprobs, use very negative value if token not in top-k
            # (represents very low probability)
            log_p = model_1_logprobs.get(token, -30.0)  # P distribution
            log_q = model_2_logprobs.get(token, -30.0)  # Q distribution

            # Convert log prob to prob
            p = math.exp(log_p)

            # KL contribution: P(x) * log(P(x)/Q(x)) = P(x) * (log_P(x) - log_Q(x))
            kl += p * (log_p - log_q)

        kl_per_token.append(kl)

    # Average over tokens
    avg_kl = sum(kl_per_token) / len(kl_per_token) if kl_per_token else 0.0

    return avg_kl, kl_per_token


def calculate_perplexity(
    prompt: str,
    response_text: str,
    model,
    tokenizer,
    reasoning_text: str = None,
    thinking_token_start: str = "<think>",
    thinking_token_end: str = "</think>",
) -> Tuple[float, List[float]]:
    """Calculate perplexity of a response using a given model.

    Args:
        prompt: The input prompt text
        response_text: The response text (content)
        model: Model to evaluate perplexity with
        tokenizer: Tokenizer for the model
        reasoning_text: Optional reasoning trace to include before response
        thinking_token_start: Start token for reasoning (default: "<think>")
        thinking_token_end: End token for reasoning (default: "</think>")

    Returns:
        Tuple of (perplexity, list of per-token log probabilities)
    """
    # Format response with reasoning if provided
    if reasoning_text is not None:
        full_response_text = (
            f"{thinking_token_start}{reasoning_text}{thinking_token_end}{response_text}"
        )
    else:
        full_response_text = response_text

    # Prepare prompt with chat template
    user_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
        add_bos=False,
    )
    user_prompt_tokens = tokenizer.encode(
        user_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )[0, :]

    # Tokenize response
    response_token_ids = tokenizer.encode(
        full_response_text,
        add_special_tokens=False,
    )

    # Combine prompt and response tokens
    tokens = torch.cat([user_prompt_tokens, torch.tensor(response_token_ids)])
    tokens = tokens.to(model.device)

    # Get log probabilities
    with torch.no_grad():
        outputs = model(tokens.unsqueeze(0))
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Vectorized extraction of log probs for the response tokens
        # Calculate position indices for response tokens in the full sequence
        start_pos = len(user_prompt_tokens) - 1
        pos_indices = torch.arange(start_pos, start_pos + len(response_token_ids), device=tokens.device)
        response_token_ids_tensor = torch.tensor(response_token_ids, device=tokens.device)

        # log_probs[0, pos_indices, response_token_ids_tensor] -> log prob at each token position
        response_log_probs = log_probs[0, pos_indices, response_token_ids_tensor].tolist()

    # Calculate perplexity: exp(-mean(log_probs))
    mean_log_prob = sum(response_log_probs) / len(response_log_probs)
    perplexity = math.exp(-mean_log_prob)

    return perplexity, response_log_probs


def get_prompt_index(filepath: str) -> int:
    """Extract prompt index from filename like 'model_prompt_123.json'"""
    basename = os.path.basename(filepath)
    # Find the last occurrence of 'prompt_' and extract number
    parts = basename.split('prompt_')
    if len(parts) > 1:
        idx = parts[-1].split('.')[0]
        try:
            return int(idx)
        except ValueError:
            return None
    return None


def load_response_files(responses_dir: str) -> dict:
    """Load all response files from a directory and map by prompt index.

    Args:
        responses_dir: Directory containing response JSON files

    Returns:
        Dictionary mapping prompt index to file path
    """
    response_files = sorted(glob.glob(os.path.join(responses_dir, "*.json")))
    print(f"Found {len(response_files)} response files")

    files_by_idx = {}
    for f in response_files:
        idx = get_prompt_index(f)
        if idx is not None:
            files_by_idx[idx] = f

    return files_by_idx
