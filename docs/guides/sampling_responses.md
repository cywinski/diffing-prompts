# Sampling LLM Responses via OpenRouter API

This guide explains how to sample multiple responses from LLMs using the OpenRouter API.

## Overview

The implementation supports:
- Concurrent sampling of multiple responses for the same prompt
- Multiple LLM models via OpenRouter API
- Generic HuggingFace dataset loader (including WildChat)
- Configurable sampling parameters
- JSON output with logprobs for analysis

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up API key:**

   Create a `.env` file or export the environment variable:
   ```bash
   export OPENAI_API_KEY=your_openrouter_api_key
   ```

   Note: OpenRouter accepts OpenAI API key format for authentication.

3. **Obtain OpenRouter API key:**
   - Sign up at https://openrouter.ai/
   - Get your API key from the dashboard
   - OpenRouter provides access to multiple LLM providers

## Configuration

Edit [experiments/configs/2025-10-12_sample_responses.yaml](../../experiments/configs/2025-10-12_sample_responses.yaml):

```yaml
# Dataset configuration
dataset:
  source: "huggingface"              # Options: "huggingface" or "file"
  dataset_name: "allenai/WildChat"   # HuggingFace dataset identifier
  use_wildchat_format: true          # Use custom WildChat extractor
  # prompt_field: "text"             # Or specify field name directly
  num_prompts: 10                    # Number of prompts to sample
  seed: 42                           # For reproducibility

# Models to sample from (list of arbitrary length)
models:
  - "openai/gpt-4"
  - "anthropic/claude-3-5-sonnet"
  # Add more models as needed

# Sampling parameters
sampling:
  num_samples_per_prompt: 5  # Samples per prompt
  max_tokens: 100
  temperature: 1.0
```

### Using Different Datasets

The loader is generic and works with any HuggingFace dataset:

**Simple text dataset:**
```yaml
dataset:
  source: "huggingface"
  dataset_name: "your-org/your-dataset"
  prompt_field: "text"  # Field containing prompts
```

**Nested field access:**
```yaml
dataset:
  source: "huggingface"
  dataset_name: "some-dataset"
  prompt_field: "data.0.content"  # Use dot notation for nested fields
```

**WildChat with conversation format:**
```yaml
dataset:
  source: "huggingface"
  dataset_name: "allenai/WildChat"
  use_wildchat_format: true  # Uses custom extractor for conversations
  language: "en"             # Optional language filter
```

### Available Model IDs

OpenRouter supports many models. Common options:
- `openai/gpt-4`
- `openai/gpt-3.5-turbo`
- `anthropic/claude-3-5-sonnet`
- `anthropic/claude-3-opus`
- `meta-llama/llama-3-70b-instruct`

See full list at: https://openrouter.ai/models

## Running the Experiment

Execute the bash script:

```bash
./experiments/scripts/run_2025-10-12_sample_responses.sh
```

This will:
1. Load prompts from the configured HuggingFace dataset
2. Sample responses from both models concurrently
3. Save results to JSON files with timestamp
4. Create logs in `experiments/logs/`

## Output Format

Results are saved as JSON files in `experiments/results/`:

```json
[
  {
    "prompt": "What is machine learning?",
    "model": "openai/gpt-4",
    "responses": [
      {
        "choices": [
          {
            "message": {"content": "..."},
            "logprobs": {
              "content": [
                {
                  "token": "Machine",
                  "logprob": -0.123,
                  "top_logprobs": [...]
                }
              ]
            }
          }
        ]
      }
    ]
  }
]
```

Each response includes:
- The generated text
- Log probabilities for each token
- Top-k alternative tokens with their logprobs

## Using Custom Prompts

Instead of WildChat, you can use a custom prompt file:

1. Create a text file with one prompt per line:
   ```
   What is the capital of France?
   Explain quantum computing.
   Write a haiku about nature.
   ```

2. Update config:
   ```yaml
   dataset:
     source: "file"
     file_path: "data/my_prompts.txt"
   ```

## Troubleshooting

**API Key Issues:**
- Verify `OPENAI_API_KEY` is set: `echo $OPENAI_API_KEY`
- Check your OpenRouter dashboard for key validity

**Rate Limits:**
- OpenRouter enforces rate limits per provider
- Adjust `num_samples_per_prompt` or add delays if needed

**Model Not Found:**
- Verify model ID at https://openrouter.ai/models
- Some models require additional permissions

## Next Steps

After sampling, you can:
1. Calculate KL divergence between model responses (Step 2 of spec)
2. Find prompts with maximally different responses
3. Analyze token-level differences using logprobs
