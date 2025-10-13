# ABOUTME: Demo script for calculating KL divergence between two models' responses.
# ABOUTME: Uses jupyter-style cells to interactively explore prompt differences.

# %%
# Parameters - Modify these to use different files or limits
# IMPORTANT: Both files must contain responses for the SAME prompts
# (generated from the same config but with different models)

# Example: Use two files from the same experiment run with different models
MODEL1_FILE = "experiments/results/YOUR_EXPERIMENT/model1_timestamp.pkl"
MODEL2_FILE = "experiments/results/YOUR_EXPERIMENT/model2_timestamp.pkl"

# For demo purposes with available data (note: these may have different prompts)
# Uncomment these lines to use actual available files:
# MODEL1_FILE = "experiments/results/sample_responses_openrouter_20251013_095245/cloud_20251013_095250.pkl"
# MODEL2_FILE = "experiments/results/sample_responses_openrouter_20251013_094756/cloud_20251013_094801.pkl"

MAX_PROMPTS = 5  # Set to None to process all prompts
TOP_N_DISPLAY = 10  # Number of top prompts to display

# %%
# Import the KL divergence engine
from src.kl_divergence_metric import calculate_kl_metrics, load_samples_from_pickle

# %%
# Load and inspect sample data
print("Loading samples...")
samples1 = load_samples_from_pickle(MODEL1_FILE)
samples2 = load_samples_from_pickle(MODEL2_FILE)

print(f"\nModel 1: {samples1[0]['model']}")
print(f"Model 2: {samples2[0]['model']}")
print(f"Total prompts: {len(samples1)}")
print(f"Responses per prompt (model 1): {len(samples1[0]['responses'])}")
print(f"Responses per prompt (model 2): {len(samples2[0]['responses'])}")

# %%
# Example: Inspect first prompt and response
print("\n" + "=" * 80)
print("EXAMPLE PROMPT:")
print("=" * 80)
print(samples1[0]["prompt"][:500] + "..." if len(samples1[0]["prompt"]) > 500 else samples1[0]["prompt"])

print("\n" + "=" * 80)
print("EXAMPLE RESPONSE (Model 1):")
print("=" * 80)
if samples1[0]["responses"] and "choices" in samples1[0]["responses"][0]:
    response_text = samples1[0]["responses"][0]["choices"][0]["message"]["content"]
    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)

# %%
# Calculate KL divergence metrics for all prompts
print("\n" + "=" * 80)
print(f"Calculating KL divergence for first {MAX_PROMPTS} prompts...")
print("=" * 80)

results = calculate_kl_metrics(
    MODEL1_FILE,
    MODEL2_FILE,
    max_prompts=MAX_PROMPTS,
)

print(f"✓ Calculated metrics for {len(results)} prompts")

# %%
# Sort prompts by KL divergence (highest first)
sorted_results = sorted(results, key=lambda x: x.kl_divergence, reverse=True)

# %%
# Display results summary
print("\n" + "=" * 80)
print(f"TOP {min(TOP_N_DISPLAY, len(sorted_results))} PROMPTS BY KL DIVERGENCE")
print("=" * 80)

for i, result in enumerate(sorted_results[:TOP_N_DISPLAY], 1):
    print(f"\n{i}. Prompt Index: {result.prompt_index}")
    print(f"   KL Divergence: {result.kl_divergence:.6f}")
    print(f"   Avg Response Length: {result.avg_response_length:.1f} tokens")
    print(f"   Num Comparisons: {result.num_comparisons}")
    print(f"   Prompt Preview: {result.prompt[:150]}...")

# %%
# Display detailed analysis of the most different prompt
print("\n" + "=" * 80)
print("MOST DIFFERENT PROMPT (Highest KL Divergence)")
print("=" * 80)

most_different = sorted_results[0]
print(f"\nKL Divergence: {most_different.kl_divergence:.6f}")
print(f"Prompt Index: {most_different.prompt_index}")
print(f"\nFull Prompt:")
print("-" * 80)
print(most_different.prompt)

# Display sample responses from both models
prompt_idx = most_different.prompt_index
print(f"\n{'=' * 80}")
print(f"SAMPLE RESPONSES FOR THIS PROMPT")
print("=" * 80)

print(f"\nModel 1 ({samples1[0]['model']}) - First Response:")
print("-" * 80)
if samples1[prompt_idx]["responses"] and "choices" in samples1[prompt_idx]["responses"][0]:
    response_text = samples1[prompt_idx]["responses"][0]["choices"][0]["message"]["content"]
    print(response_text)

print(f"\nModel 2 ({samples2[0]['model']}) - First Response:")
print("-" * 80)
if samples2[prompt_idx]["responses"] and "choices" in samples2[prompt_idx]["responses"][0]:
    response_text = samples2[prompt_idx]["responses"][0]["choices"][0]["message"]["content"]
    print(response_text)

# %%
# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

kl_values = [r.kl_divergence for r in results]
avg_kl = sum(kl_values) / len(kl_values)
max_kl = max(kl_values)
min_kl = min(kl_values)

print(f"Average KL Divergence: {avg_kl:.6f}")
print(f"Maximum KL Divergence: {max_kl:.6f}")
print(f"Minimum KL Divergence: {min_kl:.6f}")
print(f"Range: {max_kl - min_kl:.6f}")

# %%
# Done!
print("\n" + "=" * 80)
print("✓ DEMO COMPLETE")
print("=" * 80)
print(f"Processed {len(results)} prompts")
print(f"Found prompts with KL divergence ranging from {min_kl:.6f} to {max_kl:.6f}")
