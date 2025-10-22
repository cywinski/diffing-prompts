# ABOUTME: Simple demo notebook for calculating and visualizing perplexity for a single prompt and response.
# ABOUTME: Displays per-token log probabilities as a heatmap to understand model confidence across tokens.

# %%
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append('/workspace/projects/diffing-prompts/src')
from kl_divergence import calculate_perplexity

# %%
# Parameters
model_name = "openai/gpt-oss-20b"

# %%
# Load model and tokenizer
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
print("Model loaded successfully")

# %%
prompt = "lets make a story i start you continue"
response = (
    "Great! Go ahead and start the story\u2014I'm ready to continue when you're done."
)
# %%
# Tokenize the response
token_ids = tokenizer.encode(response, add_special_tokens=False)
tokens = [tokenizer.decode([tid]) for tid in token_ids]

print(f"\nPrompt: {prompt}")
print(f"Response: {response}")
print(f"\nNumber of tokens: {len(tokens)}")
print(f"Tokens: {tokens}")
# %%
# Calculate perplexity
perplexity, log_probs = calculate_perplexity(
    prompt=prompt,
    response_token_ids=token_ids,
    model=model,
    tokenizer=tokenizer,
)

print(f"\nPerplexity: {perplexity:.4f}")
print(f"Mean log probability: {np.mean(log_probs):.4f}")

# %%
def visualize_tokens_with_logprobs(
    prompt: str,
    tokens: list,
    log_probs: list,
    title: str = "Response Tokens with Log Probabilities",
    figsize: tuple = (14, 5),
):
    """Visualize response tokens with background heatmap based on log probabilities.

    Args:
        prompt: The input prompt text
        tokens: List of token strings
        log_probs: List of log probability values per token
        title: Plot title
        figsize: Figure size (width, height)
    """
    assert len(log_probs) == len(tokens)

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

    # Normalize log probs for colormap (more negative = worse = redder)
    log_probs_array = np.array(log_probs)
    vmin, vmax = log_probs_array.min(), log_probs_array.max()

    # Invert colormap so lower log prob (worse) is redder
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap("RdYlGn")  # Red (bad) to Green (good)

    # Layout parameters
    x_pos = 0.02
    y_pos = 0.93
    line_height = 0.065
    max_width = 0.96

    # Draw tokens with colored backgrounds
    for token, log_prob in zip(tokens, log_probs):
        # Estimate token width (rough approximation)
        token_width = len(token) * 0.012

        # Check if we need to wrap to next line
        if x_pos + token_width > max_width:
            x_pos = 0.02
            y_pos -= line_height

        # Get color for this log prob value
        color = cmap(norm(log_prob))

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
    cbar.set_label("Log Probability (Green = High Confidence, Red = Low Confidence)", fontsize=9)

    plt.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


# %%
# Visualize the response with log probabilities
fig = visualize_tokens_with_logprobs(
    prompt=prompt,
    tokens=tokens,
    log_probs=log_probs,
    title=f"Perplexity: {perplexity:.4f} | Model: {model_name.split('/')[-1]}",
)
plt.show()

# %%
# Print token-level statistics
print("\nPer-token statistics:")
print("-" * 80)
for i, (token, log_prob) in enumerate(zip(tokens, log_probs)):
    prob = np.exp(log_prob)
    print(f"{i:3d}. {repr(token):20s} | Log Prob: {log_prob:7.4f} | Prob: {prob:6.4f}")

# %%
# Find tokens with lowest confidence (most negative log prob)
sorted_indices = np.argsort(log_probs)
print("\nTokens with LOWEST confidence (5 worst):")
print("-" * 80)
for idx in sorted_indices[:5]:
    print(f"{repr(tokens[idx]):20s} | Log Prob: {log_probs[idx]:7.4f} | Prob: {np.exp(log_probs[idx]):6.4f}")

print("\nTokens with HIGHEST confidence (5 best):")
print("-" * 80)
for idx in sorted_indices[-5:]:
    print(f"{repr(tokens[idx]):20s} | Log Prob: {log_probs[idx]:7.4f} | Prob: {np.exp(log_probs[idx]):6.4f}")

# %%
