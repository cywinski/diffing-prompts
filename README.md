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

## Sampling from models served locally via VLLM

Serve the models via VLLM:
```bash
vllm serve unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --max-model-len 8192
```

Sample:

```bash
python src/sample_responses.py --config experiments/configs/sample_responses_openrouter_vllm_served.yaml
 ```


Calculate KL divergence (in 2 stages):

```bash
python src/calculate_kl_divergence_vllm.py --input_dir experiments/results/llama-3-70b-diff/responses_Meta-Llama-3.1-70B/ --output_dir experiments/results/llama-3-70b-diff/kl --model_name unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --mode model1
```

```bash
python src/calculate_kl_divergence_vllm.py --input_dir experiments/results/llama-3-70b-diff/kl/ --output_dir experiments/results/llama-3-70b-diff/kl_final_final --model_name unsloth/Llama-3.3-70B-Instruct-bnb-4bit --mode model2
```

```bash
python src/extract_high_kl_tokens.py --results_dir experiments/results/llama-3-70b-diff/kl_model2/ --threshold 20 --output_path experiments/results/llama-3-70b-diff/kl_extracted_score.json --score_type normalized
```

```bash
python src/sample_high_kl_responses_vllm.py --config experiments/configs/sample_high_kl_responses_vllm_served.yaml --input experiments/results/llama-3-70b-diff/kl_extracted_score.json
```
