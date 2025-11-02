# ABOUTME: Demo notebook for visualizing and analyzing KL divergence results between two models.
# ABOUTME: Loads pre-calculated KL divergence results and creates various plots and visualizations.

# %%
import json
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# %%
# Parameters
results_file1 = "/workspace/projects/diffing-prompts/experiments/results/kl/kl_llama-3.1-70b_llama-3.3-70b/kl_divergence_results_sorted.json"
results_file2 = "/workspace/projects/diffing-prompts/experiments/results/kl/kl_llama-3.3-70b_llama-3.1-70b/kl_divergence_results_sorted.json"
tokenizer_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

# %%
# Load tokenizer (same one used for KL calculation)
print(f"Loading tokenizer: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
# # %%
# Load results from saved file
with open(results_file1, "r") as f:
    results1 = json.load(f)

print(f"Loaded {len(results1)} results from: {results_file1}")
with open(results_file2, "r") as f:
    results2 = json.load(f)
print(f"Loaded {len(results2)} results from: {results_file2}")


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
    response_tokens_1: list,
    kl_per_token_1: list,
    response_tokens_2: list = None,
    kl_per_token_2: list = None,
    title: str = "Response Tokens with KL Divergence",
    cmap_name_1: str = "Reds",
    cmap_name_2: str = "Blues",
    figsize: tuple = (14, 5.0),
    min_value: float = None,
    max_value: float = None,
):
    """
    Visualize response tokens with KL divergence as background colors.
    Can display responses from two models for comparison.
    Model 1 uses cmap_name_1 and Model 2 uses cmap_name_2.

    Args:
        prompt: The prompt text
        response_tokens_1: List of lists, where each inner list contains tokens for a response from model 1
        kl_per_token_1: List of lists, where each inner list contains KL values for a response from model 1
        response_tokens_2: List of lists, where each inner list contains tokens for a response from model 2
        kl_per_token_2: List of lists, where each inner list contains KL values for a response from model 2
        title: Title for the visualization
        cmap_name_1: Colormap name for model 1 responses
        cmap_name_2: Colormap name for model 2 responses
        figsize: Figure size tuple
        min_value: Minimum value for colormap normalization
        max_value: Maximum value for colormap normalization
    """
    import html

    from matplotlib.colors import Normalize

    # Decode HTML entities and ensure proper Unicode handling
    def decode_text(text):
        """Decode HTML entities and handle Unicode properly"""
        if isinstance(text, str):
            # Decode HTML entities
            text = html.unescape(text)
            # Handle any byte strings that might be encoded
            try:
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
            except Exception:
                pass
        return text

    # Decode prompt
    prompt = decode_text(prompt)

    # Prepare all responses
    all_responses = []

    # Add model 1 responses
    if isinstance(response_tokens_1[0], str):
        # Single response case - wrap in list
        decoded_tokens_1 = [decode_text(token) for token in response_tokens_1]
        all_responses.append((decoded_tokens_1, kl_per_token_1, cmap_name_1))
    else:
        # Multiple responses case
        for tokens, kls in zip(response_tokens_1, kl_per_token_1):
            decoded_tokens = [decode_text(token) for token in tokens]
            all_responses.append((decoded_tokens, kls, cmap_name_1))

    # Add model 2 responses if provided
    if response_tokens_2 is not None and kl_per_token_2 is not None:
        if isinstance(response_tokens_2[0], str):
            # Single response case - wrap in list
            decoded_tokens_2 = [decode_text(token) for token in response_tokens_2]
            all_responses.append((decoded_tokens_2, kl_per_token_2, cmap_name_2))
        else:
            # Multiple responses case
            for tokens, kls in zip(response_tokens_2, kl_per_token_2):
                decoded_tokens = [decode_text(token) for token in tokens]
                all_responses.append((decoded_tokens, kls, cmap_name_2))

    # Normalize KL values across all responses
    all_kl_values = []
    for _, kls, _ in all_responses:
        if isinstance(kls[0], (int, float)):
            all_kl_values.extend(kls)
        else:
            for kl_list in kls:
                all_kl_values.extend(kl_list)

    if min_value is None:
        min_value = min(all_kl_values)
    if max_value is None:
        max_value = max(all_kl_values)

    norm = Normalize(vmin=min_value, vmax=max_value)

    # Dynamic figure sizing based on token count
    max_tokens = max(len(tokens) for tokens, _, _ in all_responses)
    tokens_per_line = 12
    n_lines = max(5, (max_tokens // tokens_per_line) + 1)

    # Add extra height for prompt display and multiple responses
    prompt_lines = len(prompt) // 300 + 2
    num_responses = len(all_responses)
    dynamic_height = max(figsize[1], n_lines * 0.3 * num_responses + prompt_lines * 0.2)

    fig, ax = plt.subplots(figsize=(figsize[0], dynamic_height))

    # Configuration
    line_width = figsize[0] * 0.95 / 8
    char_width = 0.02
    line_height = 0.3
    x_margin = 0.1
    y_start = dynamic_height - 1.0
    padding = 0.02

    # Display prompt at the top
    prompt_y = y_start + 0.5
    ax.text(
        x_margin - 0.3,
        prompt_y,
        "Prompt:",
        fontsize=11,
        style="italic",
        weight="bold",
        verticalalignment="top",
    )

    # Wrap and display prompt text
    prompt_text = prompt.replace("\n", " ")
    max_chars_per_line = 140
    prompt_lines_list = []
    for i in range(0, len(prompt_text), max_chars_per_line):
        prompt_lines_list.append(prompt_text[i : i + max_chars_per_line])

    current_prompt_y = prompt_y - 0.35
    for line in prompt_lines_list:
        ax.text(
            x_margin,
            current_prompt_y,
            line,
            fontsize=10,
            verticalalignment="top",
            fontfamily="sans-serif",
            color="#333333",
            wrap=True,
        )
        current_prompt_y -= 0.3

    # Add separator after prompt
    separator_y = current_prompt_y - 0.1
    ax.axhline(y=separator_y, color="gray", linewidth=1, linestyle="-", alpha=0.5)

    # Start tokens below the prompt
    current_y = separator_y - 0.5

    # Render each response
    for response_idx, (tokens, kls, response_cmap_name) in enumerate(all_responses):
        cmap = plt.get_cmap(response_cmap_name)
        current_x = x_margin

        for i, (token, kl_value) in enumerate(zip(tokens, kls)):
            # Clean token for display
            display_token = token.replace("\t", "\\t")
            # Keep actual newlines as \n for display
            if "\n" in display_token:
                display_token = display_token.replace("\n", "\\n")

            token_width = len(display_token) * char_width

            # Check if we need to wrap to next line
            if current_x + token_width > line_width and current_x > x_margin:
                current_y -= line_height
                current_x = x_margin

            # Get color based on KL value
            color = cmap(norm(kl_value))

            # Draw background rectangle
            rect = mpatches.FancyBboxPatch(
                (current_x, current_y - 0.05),
                token_width + 2 * padding,
                line_height * 0.85,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
            )
            ax.add_patch(rect)

            # Draw token text (centered in the padded box)
            text_color = "black" if norm(kl_value) < 0.5 else "white"
            ax.text(
                current_x + padding,
                current_y + line_height * 0.25,
                display_token,
                fontsize=11,
                verticalalignment="center",
                fontfamily="monospace",
                color=text_color,
                weight="normal",
            )

            # Move to next position
            current_x += token_width + 2 * padding + 0.05

        # Add spacing between responses
        if response_idx < len(all_responses) - 1:
            current_y -= line_height * 1.5

    # Add single colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name_1), norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8
    )
    cbar.set_label("KL Divergence per Token", fontsize=11, weight="bold")
    cbar.ax.tick_params(labelsize=10)

    # Set title
    ax.set_title(title, fontsize=14, weight="bold", pad=15)

    # Configure axes
    ax.set_xlim(0, line_width + x_margin)
    ax.set_ylim(current_y - 1, prompt_y + 0.5)
    ax.axis("off")

    plt.tight_layout()
    return fig


# %%
ind = 6
display_num = 2
top_result = results1[ind]
prompt = top_result["prompt"]
prompt_idx = top_result["prompt_idx"]
# search for prompt in results2
top_result2 = None
for ind2, result in enumerate(results2):
    if result["prompt_idx"] == prompt_idx:
        top_result2 = result
        break
num_responses = len(top_result["response_kls_model2_per_token"])
# Sort response indices by average KL (descending)
sorted_indices = sorted(
    range(num_responses),
    key=lambda idx: top_result["response_kls_model2_avg"][idx],
    reverse=True,
)

# Collect all responses from both models
response_tokens_list = []
kl_per_token_list = []
for response_idx in range(display_num):
    response_tokens = load_response_tokens_from_file(
        top_result["file_model2"].replace(
            "/workspace/projects/diffing-prompts/experiments/results/openrouter_samples/",
            "../experiments/results/responses_openrouter/",
        ),
        int(response_idx),
    )
    response_tokens_list.append(response_tokens)
    kl_per_token_list.append(top_result["response_kls_model2_per_token"][response_idx])

for response_idx in range(display_num):
    response_tokens2 = load_response_tokens_from_file(
        top_result2["file_model2"].replace(
            "/workspace/projects/diffing-prompts/experiments/results/openrouter_samples/",
            "../experiments/results/responses_openrouter/",
        ),
        int(response_idx),
    )
    response_tokens_list.append(response_tokens2)
    kl_per_token_list.append(top_result2["response_kls_model2_per_token"][response_idx])

# Visualize all responses at once
model_1_name = "Llama-3.3-70B"
model_2_name = "Llama-3.1-70B"
fig = visualize_response_with_kl(
    prompt=top_result["prompt"],
    response_tokens_1=response_tokens_list[:display_num],
    kl_per_token_1=kl_per_token_list[:display_num],
    response_tokens_2=response_tokens_list[display_num:],
    kl_per_token_2=kl_per_token_list[display_num:],
    title=(
        f"Prompt {top_result['prompt_idx']}, {ind, ind2} Average KL: {top_result['response_kls_model2_avg'][0]:.4f} (Red: {model_1_name}, Blue: {model_2_name})"
    ),
    cmap_name_1="Reds",
    cmap_name_2="Blues",
    min_value=0,
)
plt.show()

# %%
# Calculate average KL per token across all responses


def calculate_avg_kl_per_token(results):
    """
    Calculate the average KL divergence per token across all responses.

    Args:
        results: List of result dictionaries from KL divergence calculation

    Returns:
        Tuple of (avg_kl_per_token dict, token_kl_map dict, token_prompts_map dict, token_prompts_kl_list dict)
        - avg_kl_per_token: Dictionary mapping token strings to their average KL divergence
        - token_kl_map: Dictionary mapping tokens to list of all KL values encountered
        - token_prompts_map: Dictionary mapping tokens to set of prompt indices where they appear
        - token_prompts_kl_list: Dictionary mapping tokens to list of (prompt_idx, kl_value) tuples
    """
    token_kl_map = defaultdict(list)
    token_prompts_map = defaultdict(set)
    token_prompts_kl_list = defaultdict(list)

    # Iterate through all results (all prompts)
    for result in tqdm(results, desc="Processing results"):
        prompt_idx = result["prompt_idx"]
        num_responses = len(result["response_kls_model2_per_token"])

        # Get the response file path
        response_file = result["file_model2"].replace(
            "/workspace/projects/diffing-prompts/experiments/results/openrouter_samples/",
            "../experiments/results/responses_openrouter/",
        )

        # Iterate through all responses for this prompt
        for response_idx in range(num_responses):
            try:
                # Load response tokens
                response_tokens = load_response_tokens_from_file(
                    response_file, response_idx
                )

                # Get KL values for this response
                kl_values = result["response_kls_model2_per_token"][response_idx]

                # Map each token to its KL value
                if len(response_tokens) == len(kl_values):
                    for token, kl_value in zip(response_tokens, kl_values):
                        token_kl_map[token].append(kl_value)
                        token_prompts_map[token].add(prompt_idx)
                        token_prompts_kl_list[token].append((prompt_idx, kl_value))
                else:
                    print(
                        f"Warning: Token count mismatch for result {result['prompt_idx']}, response {response_idx}"
                    )
            except Exception as e:
                print(
                    f"Error processing result {result['prompt_idx']}, response {response_idx}: {e}"
                )

    # Calculate average KL for each token
    avg_kl_per_token = {}
    for token, kl_values in token_kl_map.items():
        avg_kl_per_token[token] = np.mean(kl_values)

    return avg_kl_per_token, token_kl_map, token_prompts_map, token_prompts_kl_list


def print_top_k_tokens(
    avg_kl_per_token,
    token_kl_map=None,
    token_prompts_map=None,
    token_prompts_kl_list=None,
    k=20,
    min_count=1,
):
    """
    Sort and print the top k tokens with highest average KL divergence.

    Args:
        avg_kl_per_token: Dictionary mapping tokens to average KL values
        token_kl_map: Optional dictionary mapping tokens to list of KL values (for count)
        token_prompts_map: Optional dictionary mapping tokens to set of prompt indices
        token_prompts_kl_list: Optional dictionary mapping tokens to list of (prompt_idx, kl_value) tuples
        k: Number of top tokens to display
        min_count: Minimum number of occurrences to include a token (default: 1)

    Returns:
        List of tuples (token, avg_kl) sorted by average KL (descending)
    """
    # Sort tokens by average KL (descending)
    sorted_tokens = sorted(avg_kl_per_token.items(), key=lambda x: x[1], reverse=True)

    # Filter tokens by minimum count if token_kl_map is available
    if token_kl_map is not None:
        sorted_tokens = [
            (token, avg_kl)
            for token, avg_kl in sorted_tokens
            if len(token_kl_map.get(token, [])) >= min_count
        ]

    print(f"\n{'=' * 70}")
    print(f"Top {k} Tokens with Highest Average KL Divergence (min_count={min_count})")
    print(f"{'=' * 70}")

    for rank, (token, avg_kl) in enumerate(sorted_tokens[:k], 1):
        # Get count of occurrences for this token (if token_kl_map is available)
        if token_kl_map is not None:
            token_count = len(token_kl_map.get(token, []))
        else:
            token_count = "N/A"

        # Create readable representation
        token_display = repr(token)[:60]  # Limit display width

        print(f"\n{rank}. Token: {token_display}")
        print(f"   Average KL: {avg_kl:.6f} | Count: {token_count}")

        # Display all prompt indices where this token appears
        if token_prompts_map is not None:
            prompt_indices = sorted(token_prompts_map.get(token, set()))
            prompts_str = ", ".join(map(str, prompt_indices[:20]))  # Show first 20
            if len(prompt_indices) > 20:
                prompts_str += f", ... (+{len(prompt_indices) - 20} more)"
            print(f"   All Prompt Indices: {prompts_str}")

        # Display top 5 prompts by KL value for this token
        if token_prompts_kl_list is not None:
            prompt_kl_pairs = token_prompts_kl_list.get(token, [])
            if prompt_kl_pairs:
                # Sort by KL value (descending) and get top 5
                top_5_pairs = sorted(prompt_kl_pairs, key=lambda x: x[1], reverse=True)[
                    :5
                ]
                top_5_str = ", ".join(
                    [f"prompt {p_idx} (KL: {kl:.4f})" for p_idx, kl in top_5_pairs]
                )
                print(f"   Top 5 Prompts by KL: {top_5_str}")

    return sorted_tokens


# %%
# Calculate average KL per token for results1
print("\n" + "=" * 70)
print("CALCULATING AVERAGE KL PER TOKEN ACROSS ALL RESPONSES")
print("=" * 70)
print("\nCalculating average KL per token for all responses...")
(
    avg_kl_per_token1,
    token_kl_map1,
    token_prompts_map1,
    token_prompts_kl_list1,
) = calculate_avg_kl_per_token(results1)

print(f"\nTotal unique tokens found: {len(avg_kl_per_token1)}")
# %%
# Print top 50 tokens with minimum count of 20
top_k = 100
min_count = 20
sorted_tokens1 = print_top_k_tokens(
    avg_kl_per_token1,
    token_kl_map=token_kl_map1,
    token_prompts_map=token_prompts_map1,
    token_prompts_kl_list=token_prompts_kl_list1,
    k=top_k,
    min_count=min_count,
)

# %%
unsorted_ind = 417
display_num = 5
for ind in range(len(results1)):
    top_result = results1[ind]
    if top_result["prompt_idx"] == unsorted_ind:
        break

prompt = top_result["prompt"]
prompt_idx = top_result["prompt_idx"]
# search for prompt in results2
top_result2 = None
for ind2, result in enumerate(results2):
    if result["prompt_idx"] == prompt_idx:
        top_result2 = result
        break
num_responses = len(top_result["response_kls_model2_per_token"])
# Sort response indices by average KL (descending)
sorted_indices = sorted(
    range(num_responses),
    key=lambda idx: top_result["response_kls_model2_avg"][idx],
    reverse=True,
)

# Collect all responses from both models
response_tokens_list = []
kl_per_token_list = []
for response_idx in range(display_num):
    response_tokens = load_response_tokens_from_file(
        top_result["file_model2"].replace(
            "/workspace/projects/diffing-prompts/experiments/results/openrouter_samples/",
            "../experiments/results/responses_openrouter/",
        ),
        int(response_idx),
    )
    response_tokens_list.append(response_tokens)
    kl_per_token_list.append(top_result["response_kls_model2_per_token"][response_idx])

if top_result2 is not None:
    for response_idx in range(display_num):
        response_tokens2 = load_response_tokens_from_file(
            top_result2["file_model2"].replace(
                "/workspace/projects/diffing-prompts/experiments/results/openrouter_samples/",
                "../experiments/results/responses_openrouter/",
            ),
            int(response_idx),
        )
        response_tokens_list.append(response_tokens2)
        kl_per_token_list.append(
            top_result2["response_kls_model2_per_token"][response_idx]
        )

# Visualize all responses at once
model_1_name = "Llama-3.3-70B"
model_2_name = "Llama-3.1-70B"
fig = visualize_response_with_kl(
    prompt=top_result["prompt"],
    response_tokens_1=response_tokens_list[:display_num],
    kl_per_token_1=kl_per_token_list[:display_num],
    response_tokens_2=response_tokens_list[display_num:]
    if top_result2 is not None
    else None,
    kl_per_token_2=kl_per_token_list[display_num:] if top_result2 is not None else None,
    title=(
        f"Prompt {top_result['prompt_idx']}, \n 3.1 vs 3.3: index: {ind}, KL: {top_result['response_kls_model2_avg'][0]:.4f} \n 3.3 vs 3.1: index: {ind2}, KL: {top_result2['response_kls_model2_avg'][0]:.4f} \n (Red: {model_1_name}, Blue: {model_2_name})"
        if top_result2 is not None
        else f"Prompt {top_result['prompt_idx']}, \n 3.1 vs 3.3: index: {ind}, KL: {top_result['response_kls_model2_avg'][0]:.4f} \n (Red: {model_1_name}, Blue: {model_2_name})"
    ),
    cmap_name_1="Reds",
    cmap_name_2="Blues",
    min_value=0,
)
plt.savefig(f"../plots/prompt_{unsorted_ind}.png")
# plt.show()
