# Notebooks

This directory contains Jupyter-style Python scripts for interactive exploration and examples.

## Files

### `kl_divergence_minimal_example.py`

Complete minimal example demonstrating the full KL divergence workflow for a single prompt.

**What it does:**
1. Samples a response from Model 1 with logprobs
2. Gets logprobs from Model 2 by iteratively prefilling with Model 1's tokens
3. Builds probability distributions for each token position
4. Calculates KL divergence per token
5. Analyzes and visualizes results

**Note:** Since Gemini API doesn't return logprobs for prefilled text, the script iteratively prefills Model 2 with Model 1's tokens, making one API call per token position to get Model 2's probability distribution at each step. This is slower but necessary to get accurate logprobs for comparison.

**Usage:**

Open in VS Code and run interactively using the `# %%` cell markers:
```bash
code notebooks/kl_divergence_minimal_example.py
```

Or run as a complete script:
```bash
python notebooks/kl_divergence_minimal_example.py
```

**Configuration:**

Edit the parameters at the top of the file:
```python
# Models to compare
MODEL_1 = "gemini-2.5-flash-lite"
MODEL_2 = "gemini-2.0-flash-exp"

# Prompt to test
PROMPT = "I am not sure if I really like this restaurant a lot."

# Sampling parameters
MAX_TOKENS = 100
TEMPERATURE = 1.0
TOP_LOGPROBS = 20
```

**Output:**
- Generated text from Model 1
- KL divergence statistics (mean, median, std, min, max, total)
- Top 10 tokens with highest KL divergence
- Visualization plots:
  - KL divergence per token position
  - Distribution of KL divergence values
- Per-token breakdown table

## Requirements

All dependencies are in `pyproject.toml`. Install with:
```bash
uv sync
```

Ensure Google Cloud authentication is configured (see `docs/guides/google_api_sampling.md`).
