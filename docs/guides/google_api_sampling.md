# Google Gemini API Sampling Guide

This guide explains how to sample LLM responses through the Google Gemini API with logprobs.

## Setup

### 1. Install Dependencies

The required dependencies are already in `pyproject.toml`:
```bash
uv sync
```

### 2. Configure Google Cloud Authentication

You have several options for authentication:

**Option A: Service Account JSON (Recommended for production)**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Option B: Application Default Credentials (Recommended for development)**
```bash
gcloud auth application-default login
```

**Option C: Set in .env file**
```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### 3. Set Google Cloud Project ID

Set your project ID in the environment:
```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
```

Or add it to your `.env` file:
```
GOOGLE_CLOUD_PROJECT=your-project-id
```

## Usage

### Basic Usage

Run the sampling script with a config file:

```bash
python src/sample_responses_google.py --config experiments/configs/sample_responses_google.yaml
```

### Configuration

Edit `experiments/configs/sample_responses_google.yaml` to customize:

#### Dataset Configuration
- **source**: "huggingface" or "file"
- **dataset_name**: HuggingFace dataset identifier
- **num_prompts**: Number of prompts to sample
- **sampling_mode**: "random" or "first"
- **language**: Language filter (for WildChat dataset)

#### Models
Add any Gemini model IDs from [Google's model list](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models):
```yaml
models:
  - "gemini-2.0-flash-exp"
  - "gemini-2.5-flash-preview-04-2025"
  - "gemini-2.5-pro-preview-03-2025"
```

#### Sampling Parameters
- **num_samples_per_prompt**: Number of samples per prompt per model
- **max_tokens**: Maximum tokens to generate (default: 8192)
- **temperature**: Sampling temperature (0.0-2.0, default: 1.0)
- **top_p**: Nucleus sampling parameter (0.0-1.0, default: 1.0)
- **response_logprobs**: Whether to return log probabilities (true/false)
- **logprobs**: Number of top logprobs per token (1-20)

## Output Format

The script saves one JSON file per prompt with the following structure:

```json
{
  "prompt": "the input prompt text",
  "model": "gemini-2.0-flash-exp",
  "responses": [
    {
      "model": "gemini-2.0-flash-exp",
      "text": "generated response text",
      "finish_reason": "STOP",
      "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 174,
        "total_tokens": 189
      },
      "logprobs": {
        "content": [
          {
            "token": "That",
            "logprob": -0.0053292112,
            "token_id": 6372,
            "top_logprobs": [
              {
                "token": "That",
                "logprob": -0.0053292112,
                "token_id": 6372
              },
              {
                "token": "I",
                "logprob": -5.1234,
                "token_id": 235
              }
            ]
          }
        ]
      }
    }
  ]
}
```

## Features

- **Parallel Sampling**: Samples multiple responses concurrently for faster execution
- **Batch Processing**: Processes multiple prompts in batches
- **Retry Logic**: Automatically retries failed requests (configurable)
- **Skip Existing**: Skips prompts whose output files already exist
- **Logprobs Support**: Returns log probabilities for each token with top alternatives
- **Flexible Prompt Loading**: Supports HuggingFace datasets and text files

## Example: Sampling from WildChat

```bash
# 1. Edit config to set dataset parameters
# 2. Run sampling
python src/sample_responses_google.py --config experiments/configs/sample_responses_google.yaml

# Results will be saved to:
# experiments/results/sample_responses_google/gemini-2.0-flash-exp_prompt_0.json
# experiments/results/sample_responses_google/gemini-2.0-flash-exp_prompt_1.json
# ...
```

## Example: Custom Prompts from File

Create a config with file source:

```yaml
dataset:
  source: "file"
  file_path: "data/my_prompts.txt"
  num_prompts: 10
  sampling_mode: "first"
```

## Troubleshooting

### Authentication Error
If you get authentication errors, ensure:
1. You've set up Google Cloud credentials properly
2. Your project ID is correct
3. The Vertex AI API is enabled in your Google Cloud project

### Quota Exceeded
If you hit quota limits, reduce:
- `num_samples_per_prompt`
- `num_prompts`
- Number of models in the config

Or increase the concurrency delay in the code if needed.
