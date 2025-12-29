# ABOUTME: Simple example of sampling a single response from OpenRouter.
# ABOUTME: Shows basic usage with logprobs extraction and assistant prefill.

# %%
# Parameters
model = "qwen/qwen3-32b"
prompt = "Quality assurance job role qualifications"

# Assistant prefill: forces the assistant to start its response with this text
# This is useful for:
# 1. Constraining response format
# 2. Getting logprobs for a specific completion
# 3. KL divergence calculations (comparing how different models complete the same text)
assistant_prefill = "The"  # Set to None to disable

max_tokens = 5
temperature = 1.0
top_logprobs = 5

# %%
# Setup
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from openrouter_client import OpenRouterClient

load_dotenv()

# %%
# Sample a single response


async def sample_single():
    client = OpenRouterClient()

    print(f"Sampling from {model}...")
    print(f"Prompt: {prompt}")
    if assistant_prefill:
        print(f"Assistant prefill: '{assistant_prefill}'")
        print(
            "\nChat messages format:\n"
            "  [{role: 'user', content: prompt},\n"
            f"   {{role: 'assistant', content: '{assistant_prefill}'}}]\n"
        )
    else:
        print("No assistant prefill (normal sampling)\n")

    response = await client.sample_response(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        reasoning=False,
        top_logprobs=top_logprobs,
        assistant_prefill=assistant_prefill,
        provider={"only": ["fireworks"]},
    )

    # Extract the text
    text = response["choices"][0]["message"]["content"]
    if assistant_prefill:
        print(f"Full response: {assistant_prefill}{text}\n")
        print(f"Generated continuation: {text}\n")
    else:
        print(f"Response: {text}\n")

    # Extract logprobs
    # NOTE: With assistant_prefill, logprobs will include ALL tokens (prefill + generated)
    logprobs_data = response["choices"][0]["logprobs"]["content"]

    if assistant_prefill:
        print("Token logprobs (includes prefill + generated tokens):")
    else:
        print("Token logprobs:")
    print(f"Total tokens: {len(logprobs_data)}\n")

    # Show first 15 tokens
    for i, token_data in enumerate(logprobs_data[:15]):
        token = token_data["token"]
        logprob = token_data["logprob"]
        prob = 2.718281828459045**logprob  # e^logprob
        print(f"  {i}: '{token}' | logprob={logprob:.4f} | prob={prob:.4f}")

        # Show top alternatives for first 5 tokens
        if "top_logprobs" in token_data and i < 5:
            print(f"     Top {top_logprobs} alternatives:")
            for alt in token_data["top_logprobs"]:
                alt_token = alt["token"]
                alt_logprob = alt["logprob"]
                alt_prob = 2.718281828459045**alt_logprob
                print(f"       '{alt_token}': {alt_prob:.4f}")

    return response


# %%
# Run
response = asyncio.run(sample_single())
