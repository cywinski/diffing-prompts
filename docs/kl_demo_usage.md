# KL Divergence Demo - Usage Guide

## Overview

This demo calculates KL divergence between two models' responses to find prompts that produce maximally different outputs.

## What's Included

1. **KL Divergence Engine** ([src/kl_divergence_metric.py](../src/kl_divergence_metric.py))
   - Core algorithm for calculating KL divergence between model responses
   - Handles logprobs from OpenRouter API format
   - Bidirectional KL calculation (averaged)
   - Per-prompt metric aggregation

2. **Demo Script** ([kl_demo.py](../kl_demo.py))
   - Jupyter-style Python script with `# %%` cell markers
   - Interactive exploration of prompt differences
   - Configurable parameters at the top
   - Shows top N most different prompts

3. **Tests** ([tests/test_kl_demo.py](../tests/test_kl_demo.py))
   - Synthetic data tests to verify correctness
   - Full pipeline test with pickle files

## Requirements

- Two pickle files with responses from **different models** for the **same prompts**
- Each file must be in the format produced by `sample_responses.py`
- Responses must include logprobs (with `top_logprobs` for accurate KL calculation)

## How to Use

### Step 1: Generate Sample Responses

First, you need two pickle files with responses from two different models for the same set of prompts.

Example config (`experiments/configs/kl_demo.yaml`):

```yaml
models:
  - "openai/gpt-4"
  - "anthropic/claude-3-opus"

dataset:
  source: "huggingface"
  dataset_name: "allenai/WildChat"
  split: "train"
  num_prompts: 100
  seed: 42

sampling:
  num_samples_per_prompt: 5
  max_tokens: 200
  temperature: 1.0
  top_p: 1.0
  logprobs: true
  top_logprobs: 10  # Higher is better for KL calculation

output:
  base_dir: "experiments/results"
  experiment_name: "kl_demo"
```

Run sampling:

```bash
python src/sample_responses.py --config experiments/configs/kl_demo.yaml
```

This will create a directory like `experiments/results/kl_demo_20231013_123456/` with two pickle files:
- `openai_gpt-4_20231013_123456.pkl`
- `anthropic_claude-3-opus_20231013_123456.pkl`

### Step 2: Configure the Demo Script

Edit [kl_demo.py](../kl_demo.py) and update the parameters at the top:

```python
# %%
# Parameters
MODEL1_FILE = "experiments/results/kl_demo_20231013_123456/openai_gpt-4_20231013_123456.pkl"
MODEL2_FILE = "experiments/results/kl_demo_20231013_123456/anthropic_claude-3-opus_20231013_123456.pkl"
MAX_PROMPTS = 10  # Start with a small number for testing
TOP_N_DISPLAY = 5
```

### Step 3: Run the Demo

You can run the demo in two ways:

**Option A: Run all cells at once**

```bash
python kl_demo.py
```

**Option B: Run interactively in VS Code**

1. Open `kl_demo.py` in VS Code
2. Click "Run Cell" buttons or use Ctrl+Enter to run cells individually
3. Explore results interactively

### Step 4: Interpret Results

The demo will output:

1. **Data summary**: Number of prompts, responses per prompt, model names
2. **Example prompt and response**: Shows the first prompt/response
3. **Top N prompts by KL divergence**: Sorted list of most different prompts
4. **Detailed analysis**: Full prompt and responses for the most different prompt
5. **Summary statistics**: Average, max, min KL divergence values

**What does the KL divergence value mean?**

- **Higher KL divergence** = Models assign very different probabilities to tokens
- **Lower KL divergence** = Models have similar probability distributions
- The metric is normalized by response length and averaged across all response pairs

## Algorithm Details

The KL divergence calculation follows this process:

1. For each prompt, take one response from Model 1
2. For each token in that response:
   - Look up the token's logprob in Model 1's distribution (p)
   - Look up the same token's logprob in Model 2's distribution (q)
   - Calculate KL contribution: p × log(p/q)
3. Average KL across all tokens in the response
4. Normalize by response length
5. Repeat for all response pairs (Model 1 → Model 2 and Model 2 → Model 1)
6. Average the bidirectional KL values
7. Average across all response combinations for the prompt

## Limitations

- **Top-k logprobs only**: If a token doesn't appear in the top-k logprobs from the other model, it's skipped. Use higher `top_logprobs` values (e.g., 10-20) for better accuracy.
- **Token alignment**: Assumes token sequences align between models. Different tokenizers may cause misalignment.
- **Computational cost**: Calculating KL for all response pairs can be expensive for large numbers of samples.

## Testing

Run the tests to verify the implementation:

```bash
PYTHONPATH=. python tests/test_kl_demo.py
```

This runs synthetic tests with known distributions to ensure the KL calculation is correct.

## Next Steps

After identifying prompts with high KL divergence, you can:

1. Manually inspect the responses to understand why they differ
2. Use these prompts as a dataset for further analysis
3. Investigate model internals (if available) for these prompts
4. Design new prompts based on patterns in high-KL examples
