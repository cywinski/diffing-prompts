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
results_file = "/workspace/projects/diffing-prompts/experiments/results/kl/kl_llama-3.3-70b-instruct_distill/kl_divergence_results_sorted.json"
tokenizer_name = "unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit"

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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np
import html


def visualize_response_with_kl_split_thinking(
    prompt: str,
    response_tokens: list,
    kl_per_token: list,
    title: str = "Response Tokens with KL Divergence",
    cmap_name: str = "Reds",
    figsize: tuple = (14, 5.0),
    min_value: float = None,
    max_value: float = None,
    thinking_token_end: str = "</think>",
):
    """
    Visualize response tokens with KL divergence as background colors.
    """

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
            except:
                pass
        return text

    # Decode prompt and tokens
    prompt = decode_text(prompt)
    response_tokens = [decode_text(token) for token in response_tokens]

    # Find thinking section boundary (after the thinking_token_end)
    thinking_end_idx = None
    for i, token in enumerate(response_tokens):
        if thinking_token_end in token:
            thinking_end_idx = i
            break

    # Normalize KL values
    if min_value is None:
        min_value = min(kl_per_token)
    if max_value is None:
        max_value = max(kl_per_token)

    norm = Normalize(vmin=min_value, vmax=max_value)
    cmap = plt.get_cmap(cmap_name)

    # Dynamic figure sizing based on token count
    n_tokens = len(response_tokens)
    tokens_per_line = 12
    n_lines = max(5, (n_tokens // tokens_per_line) + 1)

    # Add extra height for prompt display
    prompt_lines = len(prompt) // 100 + 2
    dynamic_height = max(figsize[1], n_lines * 0.4 + prompt_lines * 0.3)

    fig, ax = plt.subplots(figsize=(figsize[0], dynamic_height))

    # Configuration
    line_width = figsize[0] * 0.95
    char_width = 0.09
    line_height = 0.35
    x_margin = 0.5
    y_start = dynamic_height - 1.0
    padding = 0.03

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
    current_x = x_margin
    current_y = separator_y - 0.5

    # Track sections
    in_thinking = True

    for i, (token, kl_value) in enumerate(zip(response_tokens, kl_per_token)):
        # Check if we're leaving thinking section (after displaying the </think> token)
        if thinking_end_idx is not None and i == thinking_end_idx + 1:
            in_thinking = False
            # Add section separator
            current_y -= line_height * 0.8
            ax.axhline(
                y=current_y + line_height / 2,
                color="black",
                linewidth=1.5,
                linestyle="--",
                alpha=0.3,
            )
            current_y -= line_height * 0.5
            current_x = x_margin

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

    # Add section labels
    if thinking_end_idx is not None:
        ax.text(
            x_margin - 0.3,
            separator_y,
            "Thinking Trace",
            fontsize=11,
            style="italic",
            weight="bold",
            verticalalignment="top",
        )

        ax.text(
            x_margin - 0.3,
            current_y - line_height,
            "Normal Response",
            fontsize=11,
            style="italic",
            weight="bold",
            verticalalignment="top",
        )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
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
ind = -2
visualize_n = 1
# Visualize all responses for the selected prompt, showing thinking trace and response separately
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
        prompt_file_path = top_result["file_model2"]
        with open(prompt_file_path, "r") as f:
            prompt_data = json.load(f)
        response_data = prompt_data["responses"][response_idx]["choices"][0]["message"][
            "content"
        ]
        reasoning_data = prompt_data["responses"][response_idx]["choices"][0][
            "message"
        ].get("reasoning")
        print(f"Response {response_idx}:")
        print(response_data)
        print(f"Reasoning {response_idx}:")
        print(reasoning_data)
        print(
            "Number of response tokens:",
            len(top_result["response_kls_model2_per_token"][response_idx]),
        )

        concat_response = f"{reasoning_data}</think>{response_data}"
        response_tokens = tokenizer.encode(concat_response, add_special_tokens=False)
        response_tokens = [tokenizer.decode([tid]) for tid in response_tokens]
        print("Number of response tokens:", len(response_tokens))

        fig = visualize_response_with_kl_split_thinking(
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
            thinking_token_end="</think>",
        )
        plt.show()

# %%
