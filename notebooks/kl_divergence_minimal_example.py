# %%
# Minimal example: Sample from Model 1 and calculate KL divergence with Model 2
# This script demonstrates the complete workflow for a single prompt

import math
import os

import google.auth
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

# %%
# Parameters - Modify these as needed

# Google Cloud settings
PROJECT_ID = "adroit-solstice-480816-f5"  # Your Google Cloud project ID
LOCATION = "global"

# Models to compare
MODEL_1 = "gemini-2.5-flash-lite"
MODEL_2 = "gemini-2.5-flash-lite-preview-09-2025"

# Prompt to test
PROMPT = "I am not sure if I really like this restaurant a lot."

# Sampling parameters
MAX_TOKENS = 100
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_LOGPROBS = 20  # Number of top logprobs to get (1-20)

# %%
# Setup authentication and client

load_dotenv()

# Get credentials
credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Create client
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)

print(f"✓ Client initialized")
print(f"  Project: {PROJECT_ID}")
print(f"  Location: {LOCATION}")

# %%
# Step 1: Sample from Model 1 with logprobs

print(f"\nSampling from {MODEL_1}...")
print(f"Prompt: {PROMPT}\n")

config_model1 = GenerateContentConfig(
    max_output_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    response_logprobs=True,
    logprobs=TOP_LOGPROBS,
)

response_model1 = client.models.generate_content(
    model=MODEL_1,
    contents=PROMPT,
    config=config_model1,
)

# Extract generated text
generated_text = response_model1.text
print(f"Generated text:\n{generated_text}\n")

# Extract logprobs from Model 1
model1_tokens = []
if (
    response_model1.candidates
    and response_model1.candidates[0].logprobs_result
    and response_model1.candidates[0].logprobs_result.chosen_candidates
):
    logprobs_result = response_model1.candidates[0].logprobs_result

    for i, chosen_candidate in enumerate(logprobs_result.chosen_candidates):
        token_data = {
            "token": chosen_candidate.token,
            "logprob": chosen_candidate.log_probability,
            "top_logprobs": [],
        }

        # Add top alternatives
        if i < len(logprobs_result.top_candidates):
            top_alternatives = logprobs_result.top_candidates[i].candidates
            token_data["top_logprobs"] = [
                {
                    "token": alt.token,
                    "logprob": alt.log_probability,
                }
                for alt in top_alternatives
            ]

        model1_tokens.append(token_data)

print(f"✓ Got {len(model1_tokens)} tokens with logprobs from {MODEL_1}")

# %%
# Step 2: Get logprobs from Model 2 using iterative prefilling with Model 1's tokens

print(f"\nGetting logprobs from {MODEL_2} for the same text...")
print(
    f"Note: API doesn't return logprobs for prefilled text, so prefilling iteratively with Model 1's tokens"
)

config_model2 = GenerateContentConfig(
    max_output_tokens=1,  # Generate only 1 token at a time
    response_logprobs=True,
    logprobs=TOP_LOGPROBS,
)

model2_tokens = []

# Iteratively prefill with Model 1's tokens to collect logprobs from Model 2
for i, target_token_data in enumerate(model1_tokens):
    # Build prefill from Model 1's tokens up to position i
    prefill_text = "".join([t["token"] for t in model1_tokens[:i]])

    # Build contents with Model 1's tokens as prefill
    contents = [{"role": "user", "parts": [{"text": PROMPT}]}]

    if prefill_text:
        contents.append({"role": "model", "parts": [{"text": prefill_text}]})

    # Generate next token to get logprobs at position i
    response = client.models.generate_content(
        model=MODEL_2,
        contents=contents,
        config=config_model2,
    )

    # Extract logprobs for this token position
    if (
        response.candidates
        and response.candidates[0].logprobs_result
        and response.candidates[0].logprobs_result.chosen_candidates
    ):
        logprobs_result = response.candidates[0].logprobs_result
        chosen_candidate = logprobs_result.chosen_candidates[0]

        # Store the token from Model 1 (not what Model 2 generated)
        # We only care about Model 2's probability distribution
        token_data = {
            "token": target_token_data["token"],  # Use Model 1's token
            "logprob": None,  # Will find this in top_logprobs
            "top_logprobs": [],
        }

        # Add top alternatives and find Model 1's token logprob
        if len(logprobs_result.top_candidates) > 0:
            top_alternatives = logprobs_result.top_candidates[0].candidates
            token_data["top_logprobs"] = [
                {
                    "token": alt.token,
                    "logprob": alt.log_probability,
                }
                for alt in top_alternatives
            ]

            # Find the logprob for Model 1's token in Model 2's distribution
            for alt in top_alternatives:
                if alt.token == target_token_data["token"]:
                    token_data["logprob"] = alt.log_probability
                    break

            # If Model 1's token not in top-k, use floor probability
            if token_data["logprob"] is None:
                min_logprob = min(alt.log_probability for alt in top_alternatives)
                token_data["logprob"] = min_logprob
                print(
                    f"  Warning: M1 token '{target_token_data['token']}' not in M2's top {TOP_LOGPROBS} at position {i}, using floor"
                )

        model2_tokens.append(token_data)
    else:
        print(f"  Error: Failed to get logprobs at position {i}")
        break

    # Progress indicator
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(model1_tokens)} tokens...")

print(f"✓ Got {len(model2_tokens)} tokens with logprobs from {MODEL_2}")

# %%
# Step 3: Build probability distributions for each token


def build_distribution(top_logprobs):
    """Build probability distribution from top logprobs.

    Uses floor probability (worst logprob) for tokens not in top K.
    """
    if not top_logprobs:
        return {}, 1e-10

    # Get floor logprob (worst/lowest probability in the list)
    floor_logprob = min(item["logprob"] for item in top_logprobs)

    # Convert logprobs to probabilities
    dist = {}
    for item in top_logprobs:
        token = item["token"]
        logprob = item["logprob"]
        prob = math.exp(logprob)
        dist[token] = prob

    # Normalize (since we only have top K, probabilities won't sum to 1)
    total_prob = sum(dist.values())
    if total_prob > 0:
        dist = {token: prob / total_prob for token, prob in dist.items()}
        floor_prob = math.exp(floor_logprob) / total_prob
    else:
        floor_prob = 1e-10

    return dist, floor_prob


print("\nBuilding probability distributions...")

# Build distributions for each token position
model1_distributions = []
model2_distributions = []

for token1_data, token2_data in zip(model1_tokens, model2_tokens):
    dist1, floor1 = build_distribution(token1_data["top_logprobs"])
    dist2, floor2 = build_distribution(token2_data["top_logprobs"])

    model1_distributions.append((dist1, floor1))
    model2_distributions.append((dist2, floor2))

print(f"✓ Built distributions for {len(model1_distributions)} token positions")

# %%
# Step 4: Calculate KL divergence per token


def calculate_kl_divergence(p_dist, p_floor, q_dist, q_floor):
    """Calculate KL divergence KL(P||Q).

    For tokens not in top K, uses floor probability.
    """
    # Get all unique tokens
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())

    kl_div = 0.0
    for token in all_tokens:
        # Get probabilities, using floor if token not in top K
        p = p_dist.get(token, p_floor)
        q = q_dist.get(token, q_floor)

        # KL divergence formula: P(x) * log(P(x) / Q(x))
        if p > 0 and q > 0:
            kl_div += p * math.log(p / q)

    return kl_div


print("\nCalculating KL divergence per token...")

kl_per_token = []
token_details = []

for i, ((dist1, floor1), (dist2, floor2)) in enumerate(
    zip(model1_distributions, model2_distributions)
):
    kl_div = calculate_kl_divergence(dist1, floor1, dist2, floor2)
    kl_per_token.append(kl_div)

    token_details.append(
        {
            "position": i,
            "token": model1_tokens[i]["token"],
            "kl_divergence": kl_div,
            "model1_logprob": model1_tokens[i]["logprob"],
            "model2_logprob": model2_tokens[i]["logprob"],
        }
    )

print(f"✓ Calculated KL divergence for {len(kl_per_token)} tokens")

# %%
# Step 5: Analyze results

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Model 1: {MODEL_1}")
print(f"Model 2: {MODEL_2}")
print(f"Prompt: {PROMPT}")
print(f"\nGenerated text:\n{generated_text}")
print("\n" + "=" * 60)

# Calculate statistics
kl_array = np.array(kl_per_token)
print("\nKL Divergence Statistics:")
print(f"  Mean:   {np.mean(kl_array):.4f}")
print(f"  Median: {np.median(kl_array):.4f}")
print(f"  Std:    {np.std(kl_array):.4f}")
print(f"  Min:    {np.min(kl_array):.4f}")
print(f"  Max:    {np.max(kl_array):.4f}")
print(f"  Total:  {np.sum(kl_array):.4f}")

# Show top 10 tokens with highest KL divergence
print("\n" + "=" * 60)
print("Top 10 tokens with highest KL divergence:")
print("=" * 60)

sorted_tokens = sorted(token_details, key=lambda x: x["kl_divergence"], reverse=True)
for i, token_info in enumerate(sorted_tokens[:10], 1):
    print(f"\n{i}. Token: '{token_info['token']}' (position {token_info['position']})")
    print(f"   KL divergence: {token_info['kl_divergence']:.4f}")
    print(f"   {MODEL_1} logprob: {token_info['model1_logprob']:.4f}")
    print(f"   {MODEL_2} logprob: {token_info['model2_logprob']:.4f}")

# %%
# Step 6: Visualize KL divergence per token position

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: KL divergence per token position
axes[0].plot(kl_per_token, marker="o", linestyle="-", linewidth=1, markersize=3)
axes[0].axhline(
    np.mean(kl_array),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(kl_array):.4f}",
)
axes[0].set_xlabel("Token Position")
axes[0].set_ylabel("KL Divergence")
axes[0].set_title(f"KL Divergence per Token Position ({MODEL_1} vs {MODEL_2})")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Histogram of KL divergences
axes[1].hist(kl_per_token, bins=30, alpha=0.7, edgecolor="black")
axes[1].axvline(
    np.median(kl_array),
    color="red",
    linestyle="--",
    label=f"Median: {np.median(kl_array):.4f}",
)
axes[1].set_xlabel("KL Divergence")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of KL Divergence Values")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Step 7: Show per-token breakdown (first 20 tokens)

print("\n" + "=" * 60)
print("Per-token KL divergence (first 20 tokens):")
print("=" * 60)

for i, token_info in enumerate(token_details[:20]):
    print(
        f"{token_info['position']:3d}. '{token_info['token']:15s}' "
        f"KL: {token_info['kl_divergence']:6.4f}  "
        f"M1: {token_info['model1_logprob']:6.3f}  "
        f"M2: {token_info['model2_logprob']:6.3f}"
    )

print("\n✓ Done!")
