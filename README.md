# Model diffing via finding prompts that yield maximally different responses

Simple investigation based on the per token KL divergence

## Gemini revisions diff pipeline

### 1. Sample responses from Preview

```bash
python src/sample_responses_google.py \
  --config experiments/configs/sample_responses_google.yaml
```

### 2. Calculate KL divergence between Preview and Stable

```bash
python src/calculate_kl_divergence_batch.py --input-dir experiments/results/gemini-2.5-flash-lite-preview-09-2025-v2/ --output-dir experiments/results/kl/gemini-2.5-flash-lite-preview-09-2025_v2 --model2 gemini-2.5-flash-lite --top-logprobs 20 --bucket-name my-kl-batch-bucket
```

### 3. Get all high KL tokens

```bash
python src/extract_high_kl_tokens.py --results_dir experiments/results/kl/gemini-2.5-flash-lite-preview-09-2025/ --kl_threshold 10 --output_path experiments/results/high_kl_preview/kl.json
```

### 4. Sample responses for high KL tokens from **both models**

```bash
python src/sample_high_kl_responses.py --config experiments/configs/sample_high_kl_responses.yaml --input experiments/results/high_kl_preview/kl.json
```

### 5. Judge the differences between Preview and Stable

```bash
python src/llm_judge.py --config experiments/configs/llm_judge.yaml
```
