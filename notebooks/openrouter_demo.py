# ABOUTME: Minimal demo for sampling a single response from OpenRouter API.
# ABOUTME: Shows basic usage of OpenRouterClient with a simple prompt.

# %%
import os
import sys
from dotenv import load_dotenv

sys.path.append('/workspace/projects/diffing-prompts/src')
from openrouter_client import OpenRouterClient

# %%
# Parameters
model = "meta-llama/llama-3.2-3b-instruct"
prompt = "What is the capital of France?"
max_tokens = 100
temperature = 1.0
top_p = 1.0
logprobs = True
top_logprobs = 5
reasoning = False

# %%
# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# %%
# Create OpenRouter client
client = OpenRouterClient(api_key=api_key)

# %%
# Sample a single response
print(f"Prompt: {prompt}\n")
print(f"Model: {model}\n")

response = client.sample(
    prompt=prompt,
    model=model,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    logprobs=logprobs,
    top_logprobs=top_logprobs,
)

# %%
# Display response text
if response.get("choices") and len(response["choices"]) > 0:
    message = response["choices"][0]["message"]
    content = message.get("content", "")
    print(f"Response: {content}\n")
else:
    print("No response generated")

# %%
# Display logprobs if available
if logprobs and response.get("choices"):
    choice = response["choices"][0]
    if "logprobs" in choice and choice["logprobs"]:
        logprobs_data = choice["logprobs"]
        if "content" in logprobs_data:
            print("=" * 80)
            print("Token-level logprobs:")
            print("=" * 80)

            for i, token_data in enumerate(logprobs_data["content"]):
                token = token_data["token"]
                logprob = token_data["logprob"]
                print(f"\n{i:3d}. Token: {repr(token):20s} | Log Prob: {logprob:7.4f}")

                # Show top alternatives
                if "top_logprobs" in token_data and token_data["top_logprobs"]:
                    print(f"     Top alternatives:")
                    for alt in token_data["top_logprobs"][:3]:
                        print(f"       - {repr(alt['token']):20s} | Log Prob: {alt['logprob']:7.4f}")

# %%
# Display usage statistics
if "usage" in response:
    usage = response["usage"]
    print(f"\n{'=' * 80}")
    print("Usage Statistics:")
    print(f"{'=' * 80}")
    print(f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
    print(f"Completion tokens: {usage.get('completion_tokens', 'N/A')}")
    print(f"Total tokens: {usage.get('total_tokens', 'N/A')}")

# %%
