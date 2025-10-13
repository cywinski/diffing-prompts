# ABOUTME: Calculate KL divergence metric between model responses and saved logprobs.
# ABOUTME: Used to find prompts with maximally different responses between models.

import torch
import torch.nn.functional as F


def average_kl_from_logprobs(p_log: torch.Tensor, q_log: torch.Tensor) -> float:
    """Calculate average KL divergence from log probabilities.

    Args:
        p_log: Log probabilities from model inference (shape: [seq_len, n_top_logprobs])
        q_log: Log probabilities from saved API response (shape: [seq_len, n_top_logprobs])

    Returns:
        Average KL divergence per token
    """
    per_elem = F.kl_div(p_log, q_log, reduction="mean", log_target=True)
    return per_elem.item()


def calculate_prompt_kl(
    prompt: str,
    responses_data: list,
    model,
    tokenizer,
    n_top_logprobs: int = 5,
) -> dict:
    """Calculate KL divergence for all responses to a given prompt.

    Args:
        prompt: The input prompt text
        responses_data: List of response objects with logprobs from API
        model: The model to calculate logprobs with
        tokenizer: Tokenizer for the model
        n_top_logprobs: Number of top logprobs to use (default: 5)

    Returns:
        Dictionary with 'response_kls' list and 'average_kl' float
    """
    prompt_kls = []

    for response_data in responses_data:
        # Get the saved tokens from logprobs data
        logprobs_data = response_data["choices"][0]["logprobs"]["content"]
        response_tokens = []
        for token_data in logprobs_data:
            response_tokens.append(token_data["token"])

        # Get prompt tokens
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

        # Convert response tokens to ids
        response_token_ids = []
        for token in response_tokens:
            token_id = tokenizer.encode(token, add_special_tokens=False)[0]
            response_token_ids.append(token_id)
        response_token_ids = torch.tensor(response_token_ids)

        # Combine prompt and response tokens
        tokens = torch.cat([user_prompt_tokens, response_token_ids])
        tokens = tokens.to(model.device)

        with torch.no_grad():
            outputs = model(tokens.unsqueeze(0))
            logits = outputs.logits
            log_probs2 = torch.log_softmax(logits, dim=-1)
            top_logprobs2, _ = torch.topk(log_probs2, k=n_top_logprobs, dim=-1)
            top_logprobs2 = top_logprobs2[0, len(user_prompt_tokens) - 1 : -1].cpu()

        orig_logprobs = []
        for logprob_data in logprobs_data:
            logs = []
            for top_logprob_data in logprob_data["top_logprobs"]:
                logs.append(top_logprob_data["logprob"])
            logs.sort(reverse=True)
            orig_logprobs.append(logs)
        orig_logprobs = torch.tensor(orig_logprobs)

        if top_logprobs2.shape != orig_logprobs.shape:
            print(f"Shape mismatch: {top_logprobs2.shape=} {orig_logprobs.shape=}")
            continue

        avg_kl_top5 = average_kl_from_logprobs(top_logprobs2, orig_logprobs)
        prompt_kls.append(avg_kl_top5)

    if len(prompt_kls) == 0:
        return {"response_kls": [], "average_kl": None}

    return {
        "response_kls": prompt_kls,
        "average_kl": sum(prompt_kls) / len(prompt_kls)
    }
