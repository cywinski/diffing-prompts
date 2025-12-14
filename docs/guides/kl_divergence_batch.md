# KL Divergence Calculation with Batch Prediction API

This guide explains how to calculate KL divergence using Google Cloud's Batch Prediction API for maximum efficiency.

## Overview

The batch version submits **one batch job per response**, where each job contains all the prefill requests for that response. This is much faster and more cost-effective than making individual API calls.

## Key Differences from Standard Version

| Feature | Standard (`calculate_kl_divergence.py`) | Batch (`calculate_kl_divergence_batch.py`) |
|---------|----------------------------------------|-------------------------------------------|
| **API Calls** | N individual calls per response (parallelized) | 1 batch job per response |
| **Speed** | ~5-10 seconds per 100-token response | ~30-60 seconds per batch job (but processes all tokens) |
| **Cost** | Standard API pricing per call | Batch API pricing (50% discount) |
| **Complexity** | Simple, immediate results | Requires GCS bucket, async job waiting |
| **Best For** | Small datasets, quick testing | Large datasets, production use |

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Enable APIs

Ensure these APIs are enabled in your Google Cloud project:
- Vertex AI API
- Cloud Storage API

```bash
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
```

### 3. Create GCS Bucket (Optional)

The script will automatically create a bucket named `{project_id}-kl-batch` if it doesn't exist. You can also specify a custom bucket with `--bucket-name`.

### 4. Set Permissions

Ensure your service account has:
- `Vertex AI User` role
- `Storage Object Admin` role

## Usage

### Basic Usage

```bash
python src/calculate_kl_divergence_batch.py \
  --input-dir experiments/results/gemini-2.5-flash-lite \
  --output-dir experiments/results/kl_divergence_batch \
  --model2 gemini-2.0-flash-exp
```

### Parameters

- **--input-dir**: Directory containing JSON files from Model 1 (with logprobs)
- **--output-dir**: Directory to save KL divergence results
- **--model2**: Model identifier for comparison (Model 2)
- **--project-id**: Google Cloud project ID (optional, uses env var if not provided)
- **--location**: Google Cloud region (default: "us-central1")
- **--bucket-name**: GCS bucket name (default: "{project_id}-kl-batch")
- **--top-logprobs**: Number of top logprobs to request (1-20, default: 20)
- **--pattern**: Glob pattern for JSON files (default: "*.json")

### Example with Custom Bucket

```bash
python src/calculate_kl_divergence_batch.py \
  --input-dir experiments/results/gemini-2.5-flash-lite \
  --output-dir experiments/results/kl_divergence_batch \
  --model2 gemini-2.5-pro-preview-03-2025 \
  --bucket-name my-custom-bucket \
  --location us-central1
```

## How It Works

### 1. Create Batch Requests

For each response from Model 1 with N tokens, create N requests:

```python
requests = []
for i in range(len(model1_tokens)):
    # Build prefill from Model 1's tokens up to position i
    prefill_text = "".join([t["token"] for t in model1_tokens[:i]])

    requests.append({
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "model", "parts": [{"text": prefill_text}]} if prefill_text else None
            ],
            "generationConfig": {
                "maxOutputTokens": 1,
                "responseLogprobs": True,
                "logprobs": 20,
            }
        },
        "position": i  # Track for sorting later
    })
```

### 2. Upload to GCS

Write requests as JSONL (one request per line) to GCS:

```
gs://{bucket}/batch_input/{job_prefix}_{timestamp}.jsonl
```

### 3. Submit Batch Job

```python
batch_job = client.batches.create(
    model=model_id,
    src="gs://{bucket}/batch_input/...",
    dest="gs://{bucket}/batch_output/..."
)
```

### 4. Wait for Completion

Poll the batch job status every 10 seconds until completion:

```python
while True:
    batch_job = client.batches.get(name=job_name)
    if batch_job.state == "JOB_STATE_SUCCEEDED":
        break
    await asyncio.sleep(10)
```

### 5. Download Results

Read output JSONL files from GCS and parse into logprobs.

## Output Format

Same as the standard version - see [kl_divergence_calculation.md](kl_divergence_calculation.md#output-format).

## Performance

### Speed

- **Batch job creation**: ~1 second
- **Job execution**: 30-60 seconds (regardless of token count)
- **Total per response**: ~1 minute

For a 100-token response:
- Standard version with `--max-concurrent 20`: ~5-10 seconds
- Batch version: ~60 seconds

**When is batch faster?**
- Large datasets (100+ responses)
- Long responses (>200 tokens)
- When processing overnight (cost savings outweigh speed)

### Cost

Batch prediction API offers **50% discount** compared to online prediction:

| Version | 100-token response | Cost Savings |
|---------|-------------------|--------------|
| Standard | 100 API calls × $X | - |
| Batch | 100 batch requests × $X/2 | **50%** |

## Advantages

1. **Cost Effective**: 50% discount on API calls
2. **Scalable**: Handles large datasets efficiently
3. **No Rate Limits**: Batch jobs don't count against online quota
4. **Automatic Retry**: Google handles retries internally

## Disadvantages

1. **Slower for Small Jobs**: Batch overhead (~30-60s) not worth it for <10 responses
2. **More Complex**: Requires GCS bucket, async job management
3. **Delayed Results**: Can't see results until batch job completes
4. **Debugging Harder**: Errors are in batch output files, not immediate

## When to Use Each Version

### Use Standard Version (`calculate_kl_divergence.py`)

- ✅ Quick testing with <10 responses
- ✅ Short responses (<50 tokens)
- ✅ Need results immediately
- ✅ Iterative development/debugging

### Use Batch Version (`calculate_kl_divergence_batch.py`)

- ✅ Large datasets (>50 responses)
- ✅ Long responses (>200 tokens)
- ✅ Production workloads
- ✅ Cost optimization is priority
- ✅ Can wait ~1 hour for results

## Troubleshooting

### Batch Job Stuck in PENDING

- Check Vertex AI quotas in GCP console
- Ensure region supports batch prediction (use `us-central1`)
- Verify model ID is correct

### GCS Permission Errors

```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member=serviceAccount:SERVICE_ACCOUNT \
  --role=roles/storage.objectAdmin
```

### Batch Job Failed

Check job details:
```python
batch_job = client.batches.get(name=job_name)
print(batch_job.error)
```

Download error files from GCS:
```
gs://{bucket}/batch_output/{job_prefix}_{timestamp}/error/
```

## Example Workflow

```bash
# 1. Sample responses from Model 1 with logprobs
python src/sample_responses_google.py \
  --config experiments/configs/sample_responses_google.yaml

# 2. Calculate KL divergence using batch API
python src/calculate_kl_divergence_batch.py \
  --input-dir experiments/results/gemini-2.5-flash-lite \
  --output-dir experiments/results/kl_divergence_batch \
  --model2 gemini-2.0-flash-exp

# 3. Analyze results (same as standard version)
python examples/kl_divergence_example.py \
  --result-dir experiments/results/kl_divergence_batch \
  --top-k 10
```

## Cleanup

Batch I/O files are stored in GCS. To save storage costs, delete old files:

```bash
# Delete files older than 7 days
gsutil -m rm "gs://{bucket}/batch_input/**"
gsutil -m rm "gs://{bucket}/batch_output/**"
```

Or set lifecycle rules on the bucket:
```bash
gcloud storage buckets update gs://{bucket} \
  --lifecycle-file=lifecycle.json
```

Where `lifecycle.json`:
```json
{
  "lifecycle": {
    "rule": [{
      "action": {"type": "Delete"},
      "condition": {"age": 7}
    }]
  }
}
```
