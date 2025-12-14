# KL Divergence Calculation Guide

This guide explains how to calculate KL divergence per token between two models using logprobs.

## Two Versions Available

| Version | Script | Best For | Speed | Cost |
|---------|--------|----------|-------|------|
| **Standard** | `calculate_kl_divergence.py` | Small datasets, quick testing | Fast (parallel API calls) | Standard pricing |
| **Batch** | `calculate_kl_divergence_batch.py` | Large datasets, production | Moderate (async batch jobs) | 50% cheaper |

**This guide covers the standard version.** For batch prediction, see [kl_divergence_batch.md](kl_divergence_batch.md).

## Overview

The standard version calculates KL divergence by:
1. Loading responses from Model 1 (which already have logprobs saved)
2. Making parallel API calls to get logprobs from Model 2 for each token position
3. Calculating KL divergence per token between the two probability distributions
4. Handling truncated distributions (only top 20 logprobs available)

## Setup

Install dependencies:
```bash
uv sync
```

Ensure Google Cloud authentication is configured (see [google_api_sampling.md](google_api_sampling.md)).

## Usage

### Basic Usage

```bash
python src/calculate_kl_divergence.py \
  --input-dir experiments/results/gemini-2.5-flash-lite \
  --output-dir experiments/results/kl_divergence \
  --model2 gemini-2.0-flash-exp \
  --max-concurrent 20
```

### Parameters

- **--input-dir**: Directory containing JSON files from Model 1 (with logprobs)
- **--output-dir**: Directory to save KL divergence results
- **--model2**: Model identifier for comparison (Model 2)
- **--project-id**: Google Cloud project ID (optional, uses env var if not provided)
- **--location**: Google Cloud region (default: "global")
- **--top-logprobs**: Number of top logprobs to request (1-20, default: 20)
- **--pattern**: Glob pattern for JSON files (default: "*.json")
- **--max-concurrent**: Maximum number of concurrent API calls per response (default: 10)
- **--max-files**: Maximum number of files to process (default: all)

### Example with Custom Pattern

```bash
python src/calculate_kl_divergence.py \
  --input-dir experiments/results/gemini-2.5-flash-lite \
  --output-dir experiments/results/kl_divergence_flash_vs_pro \
  --model2 gemini-2.5-pro-preview-03-2025 \
  --top-logprobs 20 \
  --pattern "gemini-2.5-flash-lite_prompt_*.json"
```

## Input Format

The script expects JSON files in the following format (from `sample_responses_google.py`):

```json
{
  "prompt": "user prompt text",
  "model": "gemini-2.5-flash-lite",
  "responses": [
    {
      "model": "gemini-2.5-flash-lite",
      "text": "generated response",
      "logprobs": {
        "content": [
          {
            "token": "Hello",
            "logprob": -0.123,
            "token_id": 1234,
            "top_logprobs": [
              {"token": "Hello", "logprob": -0.123, "token_id": 1234},
              {"token": "Hi", "logprob": -2.456, "token_id": 5678}
            ]
          }
        ]
      }
    }
  ]
}
```

## Output Format

The script generates one output file per input file with the suffix `_kl.json`:

```json
{
  "prompt": "user prompt text",
  "model1": "gemini-2.5-flash-lite",
  "model2": "gemini-2.0-flash-exp",
  "num_responses": 5,
  "responses": [
    {
      "text": "generated response",
      "num_tokens": 42,
      "kl_per_token": [0.123, 0.456, 0.789, ...],
      "token_details": [
        {
          "position": 0,
          "token": "Hello",
          "kl_divergence": 0.123,
          "model1_chosen_logprob": -0.123,
          "model2_chosen_logprob": -0.456
        }
      ],
      "statistics": {
        "mean_kl": 0.345,
        "median_kl": 0.234,
        "std_kl": 0.123,
        "min_kl": 0.001,
        "max_kl": 1.234,
        "total_kl": 14.49
      }
    }
  ]
}
```

## How It Works

### 1. Parallel Prefilling with Model 1's Tokens

Since the Gemini API doesn't return logprobs for prefilled text (only for generated text), the script uses an iterative prefilling approach **with parallel processing for speed**:

For each response from Model 1, the script:

1. Creates API calls for all token positions (0 to N)
2. Processes them in parallel with a configurable concurrency limit
3. Each call prefills with Model 1's tokens up to position i, then generates 1 token to get Model 2's logprobs at position i

```python
config = GenerateContentConfig(
    max_output_tokens=1,
    response_logprobs=True,
    logprobs=20,
)

# Create semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(max_concurrent)  # e.g., 10 or 20

async def get_logprobs_for_position(position):
    async with semaphore:
        # Build prefill from Model 1's tokens up to position
        prefill_text = "".join([t["token"] for t in model1_tokens[:position]])

        # Build contents with Model 1's tokens as prefill
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        if prefill_text:
            contents.append({"role": "model", "parts": [{"text": prefill_text}]})

        # Generate next token to get logprobs at position
        return await client.models.generate_content(
            model=model2_id, contents=contents, config=config
        )

# Process all positions in parallel
tasks = [get_logprobs_for_position(i) for i in range(len(model1_tokens))]
results = await asyncio.gather(*tasks)
```

**Key insight:** We always use Model 1's tokens in the prefill. We don't care what token Model 2 would generate - we only want Model 2's probability distribution (logprobs) at each position, given the same context (Model 1's tokens so far).

**Performance:** Instead of N sequential API calls, we make N/max_concurrent batches of parallel calls. With `--max-concurrent 20`, a 100-token response takes ~5 batches instead of 100 sequential calls, significantly speeding up processing.

### 2. Handling Truncated Distributions

Since the API only returns top 20 logprobs, we use a floor probability for missing tokens:

1. For each token position, get top 20 logprobs from both models
2. Convert logprobs to probabilities: `prob = exp(logprob)`
3. For tokens not in top 20, use floor probability (20th logprob value)
4. Normalize probabilities to sum to 1
5. Calculate KL divergence: `KL(P||Q) = sum(P(i) * log(P(i) / Q(i)))`

### 3. Handling Missing Tokens in Top-K

Since we only get top-20 logprobs from the API, Model 1's token might not appear in Model 2's top-20 at a given position. When this happens:

1. A warning is printed
2. We use the floor probability (the 20th logprob value) as an approximation
3. The KL divergence calculation proceeds

This is conservative - it assumes Model 2 assigned very low probability to Model 1's chosen token.

**Note:** Both models should use the same tokenizer (which is true for different Gemini model variants). If you compare models with different tokenizers, the results may not be meaningful.

## Statistics Provided

For each response, the script calculates:

- **mean_kl**: Average KL divergence across all tokens
- **median_kl**: Median KL divergence
- **std_kl**: Standard deviation of KL divergence
- **min_kl**: Minimum KL divergence across tokens
- **max_kl**: Maximum KL divergence across tokens
- **total_kl**: Sum of KL divergences (total sequence-level divergence)

## Example Workflow

### Step 1: Sample responses from Model 1 with logprobs

```bash
python src/sample_responses_google.py \
  --config experiments/configs/sample_responses_google.yaml
```

### Step 2: Calculate KL divergence against Model 2

```bash
python src/calculate_kl_divergence.py \
  --input-dir experiments/results/gemini-2.5-flash-lite \
  --output-dir experiments/results/kl_divergence \
  --model2 gemini-2.0-flash-exp
```

### Step 3: Analyze results

Load and analyze the KL divergence results:

```python
import json
import numpy as np

# Load results
with open("experiments/results/kl_divergence/model_prompt_0_kl.json") as f:
    data = json.load(f)

# Analyze statistics
for i, response in enumerate(data["responses"]):
    stats = response["statistics"]
    print(f"Response {i}:")
    print(f"  Mean KL: {stats['mean_kl']:.4f}")
    print(f"  Total KL: {stats['total_kl']:.4f}")
    print(f"  Max KL: {stats['max_kl']:.4f}")
```

## Interpretation

### KL Divergence Values

- **Low KL (< 0.1)**: Models have very similar probability distributions for this token
- **Medium KL (0.1-1.0)**: Moderate difference in distributions
- **High KL (> 1.0)**: Large difference in distributions

### Total KL

The `total_kl` statistic (sum of per-token KL) represents the overall divergence for the entire sequence. Higher values indicate:
- Models disagree more about what tokens should come next
- Different generation strategies or capabilities

### Use Cases

1. **Model Comparison**: Identify which prompts produce most different responses
2. **Fine-tuning Evaluation**: Measure how much a fine-tuned model differs from base
3. **Model Selection**: Choose models with appropriate diversity for your task
4. **Anomaly Detection**: Find responses where models strongly disagree

## Performance Considerations

The script processes multiple token positions in parallel for significant speedup:

### Speed

- **Parallel processing**: Uses `asyncio.gather()` with semaphore for controlled concurrency
- **Configurable concurrency**: Use `--max-concurrent` to control parallel API calls (default: 10)
- **Example timing**:
  - Sequential (old): 100 tokens × 1 second = 100 seconds
  - Parallel with `--max-concurrent 20`: 100 tokens / 20 = ~5 batches × 1 second = ~5 seconds
  - **~20x speedup**!

### Cost

- **Same number of API calls**: Still requires N calls for N tokens (unavoidable)
- **Higher API costs** than single-response approaches (but no alternative given API limitations)

### Tuning `--max-concurrent`

- **Too low (e.g., 5)**: Slower processing
- **Too high (e.g., 50)**: May hit rate limits or overwhelm API
- **Recommended**: Start with 10-20, increase if no rate limit errors

### Other Optimizations

- Use shorter responses (reduce `max_tokens` in sampling config)
- Process overnight for very large datasets
- Filter to most important prompts first

## Troubleshooting

### Token Not in Top-K Warnings

If you see warnings like "M1 token 'X' not in M2's top 20 at position i", this means:
- Model 2 assigned very low probability to the token that Model 1 chose
- This is actually interesting - it indicates the models strongly disagree at this position
- The script uses the floor probability (20th logprob) as an approximation
- The KL divergence at this position will be higher, which is correct

### API Rate Limits

If you hit rate limits with the error `429 Too Many Requests`:
- **Reduce `--max-concurrent`**: Try 5 or 10 instead of 20
- **Use a higher tier Google Cloud quota**: Increase your API quota in GCP console
- **Process fewer responses per prompt**: Reduce `num_samples_per_prompt` in sampling config
- **Add delays** (if needed): The semaphore already provides throttling

### Memory Issues

For large datasets:
- Process files one at a time instead of in parallel
- Filter input files using the `--pattern` argument
- Process in chunks and combine results later
