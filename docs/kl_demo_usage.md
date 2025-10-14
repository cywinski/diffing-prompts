# KL Divergence Demo - Usage Guide

## Overview

This demo calculates KL divergence between model-generated logprobs and saved API logprobs to validate that a local model produces the same distributions as the API version.

## What's Included

1. **KL Divergence Engine** ([src/kl_metric.py](../src/kl_metric.py))
   - Core algorithm for calculating KL divergence between local model inference and saved API logprobs
   - Requires model inference with transformers
   - Handles logprobs from OpenRouter API format (JSON files)
   - Per-response and per-prompt metric aggregation

2. **Demo Script** ([kl_demo.py](../kl_demo.py))
   - Jupyter-style Python script with `# %%` cell markers
   - Interactive exploration of KL divergence between local and API inference
   - Configurable parameters at the top
   - Shows KL divergence for each prompt

3. **Notebook** ([notebooks/kl_demo_notebook.py](../notebooks/kl_demo_notebook.py))
   - Enhanced version with sorting and analysis
   - Shows prompts sorted by KL divergence
   - Configurable max_prompts parameter for quick demos

## Requirements

- JSON files with responses from OpenRouter API (with logprobs)
- Local model accessible via transformers
- Each JSON file must contain:
  - `prompt`: The input prompt
  - `responses`: List of API responses with logprobs

## How to Use

### Step 1: Generate Sample Responses via API

First, you need JSON files with responses from the API. Use the OpenRouter API sampling script:

```bash
bash experiments/scripts/sample_responses_openrouter.sh
```

This will create a directory like `experiments/results/responses_openrouter/gemma-2-9b-it/` with JSON files, one per prompt.

### Step 2: Configure the Demo Script

Edit [kl_demo.py](../kl_demo.py) and update the parameters at the top:

```python
# %%
# model_name = "bcywinski/gemma-2-9b-it-taboo-cloud"
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# %%
responses_dir = "/workspace/projects/diffing-prompts/experiments/results/responses_openrouter/gemma-2-9b-it"
responses_files = glob.glob(os.path.join(responses_dir, "*.json"))
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

**Option C: Use the notebook with sorting**

```bash
python notebooks/kl_demo_notebook.py
```

This version will sort prompts by KL divergence and show you which prompts have the highest divergence between local and API inference.

### Step 4: Interpret Results

The demo will output for each prompt:

1. **Average response KL**: KL divergence for each response
2. **Average prompt KL**: Average KL across all responses for this prompt

**What does the KL divergence value mean?**

- **Higher KL divergence** = Local model assigns different probabilities than the API (possible model mismatch or version difference)
- **Lower KL divergence** = Local model matches the API closely (good validation)
- The metric is averaged across top-5 logprobs for each token

## Algorithm Details

The KL divergence calculation follows this process:

1. For each prompt, load the saved API responses with logprobs
2. Extract the response tokens from the API logprobs
3. Apply the chat template to the prompt and concatenate with response tokens
4. Run the local model on the full sequence to get logprobs
5. Extract top-5 logprobs for each token position
6. Compare with the saved API top-5 logprobs using KL divergence: `F.kl_div(p_log, q_log, reduction="mean", log_target=True)`
7. Average across all tokens in the response
8. Repeat for all responses for the prompt
9. Calculate the final average KL for the prompt

## Limitations

- **Top-k logprobs only**: Compares only the top-5 logprobs from both model and API
- **Requires exact model match**: The local model should be the same as the API version
- **Tokenizer differences**: Different tokenizer versions may cause issues
- **Shape mismatches**: If the shapes don't match (different number of tokens), the response is skipped

## Use Cases

This implementation is useful for:

1. **Model validation**: Verify that your local model matches the API version
2. **Version checking**: Detect if the API model has been updated
3. **Debugging**: Find prompts where local and API inference diverge
4. **Quality control**: Ensure consistent behavior between local and API deployment

## Next Steps

After calculating KL divergence:

1. Low KL values indicate good model match
2. High KL values may indicate:
   - Model version mismatch
   - Tokenizer differences
   - Numerical precision issues
   - Different sampling parameters
3. Use this to validate your local model setup before running expensive experiments
