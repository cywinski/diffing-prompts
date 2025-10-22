# ABOUTME: Demo notebook for visualizing and analyzing KL divergence results between two models.
# ABOUTME: Loads pre-calculated KL divergence results and creates various plots and visualizations.

# %%
import json
import os
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# %%
# Parameters
results_file = "/workspace/projects/diffing-prompts/experiments/results/kl_llama-3.2-1b-instruct/kl_divergence_results_sorted.json"
tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"

# %%
# Load tokenizer (same one used for KL calculation)
print(f"Loading tokenizer: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
print("Tokenizer loaded successfully")

# %%
# Load results from saved file
with open(results_file, "r") as f:
    results = json.load(f)

print(f"Loaded {len(results)} results from: {results_file}")

# %%
# Helper function to load response tokens from file
def load_response_tokens_from_file(response_file: str, response_idx: int = 0):
    """Load response tokens from a saved response JSON file.

    Args:
        response_file: Path to the response JSON file
        response_idx: Index of the response to load (default: 0)

    Returns:
        List of token strings for the specified response
    """
    with open(response_file, "r") as fp:
        data = json.load(fp)

    if response_idx >= len(data["responses"]):
        raise IndexError(
            f"Response index {response_idx} out of range. File has {len(data['responses'])} responses."
        )

    # Extract text response and tokenize using the tokenizer
    response_text = data["responses"][response_idx]["choices"][0]["message"]["content"]

    # Tokenize the response text to get token IDs
    token_ids = tokenizer.encode(response_text, add_special_tokens=False)

    # Convert token IDs back to token strings
    response_tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return response_tokens


# %%
def visualize_response_with_kl(
    prompt: str,
    response_tokens: list,
    kl_per_token: list,
    title: str = "Response Tokens with KL Divergence",
    cmap_name: str = "Reds",
    figsize: tuple = (14, 5),
    min_value: float = None,
    max_value: float = None,
):
    """Visualize response tokens with background heatmap based on KL divergence.

    Args:
        prompt: The input prompt text
        response_tokens: List of response token strings
        kl_per_token: List of KL divergence values per token
        title: Plot title
        cmap_name: Colormap name (single-color maps: 'Reds', 'Blues', 'Greens', 'Purples', 'Oranges')
        figsize: Figure size (width, height)
        min_value: Minimum value for the colormap (overrides KL values' min for color scaling if not None)
        max_value: Maximum value for the colormap (overrides KL values' max for color scaling if not None)
    """
    assert len(kl_per_token) == len(response_tokens)
    fig, (ax_prompt, ax_response) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [0.4, 3], "hspace": 0.15}
    )

    # Display prompt
    ax_prompt.text(
        0.05,
        0.5,
        f"Prompt: {prompt}",
        fontsize=10,
        verticalalignment="center",
        wrap=True,
    )
    ax_prompt.axis("off")
    ax_prompt.set_xlim(0, 1)
    ax_prompt.set_ylim(0, 1)
    ax_prompt.margins(0)

    # Prepare response visualization
    ax_response.axis("off")
    ax_response.set_xlim(0, 1)
    ax_response.set_ylim(0, 1)

    # Normalize KL values for colormap
    kl_array = np.array(kl_per_token)
    vmin = min_value if min_value is not None else kl_array.min()
    vmax = max_value if max_value is not None else kl_array.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(cmap_name)

    # Layout parameters
    x_pos = 0.02
    y_pos = 0.93
    line_height = 0.065
    max_width = 0.96

    # Draw tokens with colored backgrounds
    for token, kl_val in zip(response_tokens, kl_per_token):
        # Estimate token width (rough approximation)
        token_width = len(token) * 0.012

        # Check if we need to wrap to next line
        if x_pos + token_width > max_width:
            x_pos = 0.02
            y_pos -= line_height

        # Get color for this KL value
        color = cmap(norm(kl_val))

        # Draw background rectangle
        rect = mpatches.Rectangle(
            (x_pos, y_pos - 0.05),
            token_width,
            0.06,
            facecolor=color,
            edgecolor="none",
            transform=ax_response.transAxes,
        )
        ax_response.add_patch(rect)

        # Draw token text
        ax_response.text(
            x_pos + token_width / 2,
            y_pos - 0.02,
            token.replace("\n", "\\n"),
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            family="monospace",
        )

        x_pos += token_width + 0.005

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax_response, orientation="horizontal", pad=0.01, fraction=0.04, aspect=50
    )
    cbar.set_label("KL Divergence per Token", fontsize=9)

    plt.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


# %%
# Print summary statistics
print(f"\nSummary Statistics:")
print(f"Total prompts: {len(results)}")
kl_values = [r["average_kl"] for r in results]
print(f"Average KL: {np.mean(kl_values):.6f}")
print(f"Median KL: {np.median(kl_values):.6f}")
print(f"Min KL: {np.min(kl_values):.6f}")
print(f"Max KL: {np.max(kl_values):.6f}")

# %%
# Distribution of KL per token values
kl_per_token_values = [
    y for x in results[:100] for y in x["response_kls_model2_per_token"]
]
plt.figure(figsize=(10, 6))
plt.hist(kl_per_token_values, bins=50, alpha=0.7, edgecolor="black")
plt.xlabel("KL Divergence per Token", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Distribution of KL Divergence per Token Values")
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Plot: Response length vs KL divergence
responses_len = [len(y) for x in results for y in x["response_kls_model2_per_token"]]
kls = [y for x in results for y in x["response_kls_model2_avg"]]

plt.figure(figsize=(10, 6))
plt.scatter(responses_len, kls, alpha=0.5)
plt.xlabel("Length of response", fontsize=18)
plt.ylabel("KL", fontsize=16)
plt.grid(True, alpha=0.3)
plt.title("KL Divergence vs Response Length")
plt.show()

# %%
# Plot: Distribution of KL values
plt.figure(figsize=(10, 6))
plt.hist(kl_values, bins=50, alpha=0.7, edgecolor="black")
plt.xlabel("KL Divergence", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Distribution of KL Divergence Values")
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Print top 10 prompts with highest KL
print("\nTop 10 prompts with highest KL divergence:")
print("-" * 80)
for i, result in enumerate(results[:10], 1):
    print(f"{i:2d}. Prompt {result['prompt_idx']}: {result['average_kl']:.6f}")
    print(f"    {result['prompt'][:100]}...")
    print()

# %%
ind = 400
visualize_n = 3
# Visualize all responses for the selected prompt, sorted by average KL divergence
if len(results) > 0:
    top_result = results[ind]
    num_responses = len(top_result["response_kls_model2_per_token"])

    # Sort response indices by average KL (descending)
    sorted_indices = sorted(
        range(num_responses),
        key=lambda idx: top_result["response_kls_model2_avg"][idx],
        reverse=True,
    )

    for rank, response_idx in enumerate(sorted_indices[:visualize_n], 1):
        response_tokens = load_response_tokens_from_file(
            top_result["file_model2"], response_idx
        )

        fig = visualize_response_with_kl(
            prompt=top_result["prompt"],
            response_tokens=response_tokens,
            kl_per_token=top_result["response_kls_model2_per_token"][response_idx],
            title=(
                f"Prompt {top_result['prompt_idx']} - Response {response_idx} "
                f"(Sorted Rank: {rank}, Avg KL: {top_result['response_kls_model2_avg'][response_idx]:.4f})"
            ),
            cmap_name="Reds",
            min_value=0,
            max_value=10,
        )
        plt.show()


# %%
# Calculate average KL per token across all results
token_kl_values = defaultdict(list)

print("\nAggregating KL values per token across all results...")
for result in tqdm(results, desc="Processing results"):
    # Load response tokens from file for each response
    for response_idx in range(len(result["response_kls_model2_per_token"])):
        response_tokens = load_response_tokens_from_file(
            result["file_model2"], response_idx
        )
        kl_values = result["response_kls_model2_per_token"][response_idx]

        # Aggregate KL values by token
        for token, kl in zip(response_tokens, kl_values):
            token_kl_values[token].append(kl)

print(f"Found {len(token_kl_values)} unique tokens across all responses")

# %%
# Calculate average KL per token
token_avg_kl = {}
for token, kl_list in tqdm(token_kl_values.items(), desc="Calculating averages"):
    token_avg_kl[token] = sum(kl_list) / len(kl_list)

# Sort by average KL (highest first)
sorted_tokens = sorted(token_avg_kl.items(), key=lambda x: x[1], reverse=True)

# %%
# Print top 30 tokens with highest average KL (filtered by minimum count)
min_count = 10
print(f"\nTop 30 tokens with highest average KL divergence (min count: {min_count}):")
print("-" * 60)
filtered_tokens = [
    (token, avg_kl)
    for token, avg_kl in sorted_tokens
    if len(token_kl_values[token]) >= min_count
]
for i, (token, avg_kl) in enumerate(filtered_tokens[:30], 1):
    token_repr = repr(token)
    count = len(token_kl_values[token])
    print(f"{i:2d}. {token_repr:20s} | Avg KL: {avg_kl:.6f} | Count: {count}")

# %%
