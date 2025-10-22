# ABOUTME: Minimal demo for sampling Gemini 2.5 Flash Lite and accessing logprobs.
# ABOUTME: Shows basic usage of Google Gen AI SDK with logprob output.

# %%
import os

from dotenv import load_dotenv

# Note: Requires google-genai package (official GA SDK as of May 2025)
# Install with: uv add google-genai
try:
    from google import genai
    from google.genai.types import GenerateContentConfig
except ImportError:
    print("ERROR: google-genai not installed")
    print("Run: uv add google-genai")
    exit(1)

# %%
# Parameters
model_name = "gemini-2.5-flash-lite-preview-09-2025"  # Gemini 2.5 Flash Lite model
prompt = "What is the capital of France?"
temperature = 1.0
max_output_tokens = 100
top_p = 1.0

# %%
# Load API key from .env and create client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

# %%
# Generate response with logprobs enabled
print(f"Prompt: {prompt}\n")

response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        response_logprobs=True,  # Enable logprobs in response
        logprobs=5,  # Number of alternative tokens to return (1-20)
    ),
)

# %%
# Display response text
print(f"Response: {response.text}\n")

# %%
# Access logprobs from the response
# logprobs are in response.candidates[0].logprobs_result
candidate = response.candidates[0]

if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
    logprobs_result = candidate.logprobs_result

    print("=" * 80)
    print("Token-level logprobs:")
    print("=" * 80)

    # Iterate through chosen tokens
    chosen = logprobs_result.chosen_candidates
    top_candidates = logprobs_result.top_candidates

    for i, token_data in enumerate(chosen):
        token = token_data.token
        logprob = token_data.log_probability

        print(f"\n{i:3d}. Token: {repr(token):20s} | Log Prob: {logprob:7.4f}")

        # Show top alternatives for this position
        if i < len(top_candidates) and top_candidates[i].candidates:
            print(f"     Top alternatives:")
            for alt in top_candidates[i].candidates[:3]:  # Show top 3
                print(f"       - {repr(alt.token):20s} | Log Prob: {alt.log_probability:7.4f}")
else:
    print("No logprobs available in response")
    print(f"Available attributes: {dir(candidate)}")

# %%
# Calculate perplexity from logprobs (optional)
if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
    import math

    logprobs = [t.log_probability for t in logprobs_result.chosen_candidates]
    avg_logprob = sum(logprobs) / len(logprobs)
    perplexity = math.exp(-avg_logprob)

    print(f"\n{'=' * 80}")
    print("Statistics:")
    print(f"{'=' * 80}")
    print(f"Number of tokens: {len(logprobs)}")
    print(f"Average log probability: {avg_logprob:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

# %%
