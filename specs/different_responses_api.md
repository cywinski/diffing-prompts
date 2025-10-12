# Experiment: Find maximally different responses between two models for the same prompt via API

**Date**: [2025-10-12]
**Goal**: Find maximally different responses between two models for the same prompt via API using KL Divergence metric.
Sample via API multiple responses for the same prompt for selected two LLMs. For each combination calculate KL Divergence based on logprobs between the first N tokens of each response.
Implementation should easily support:

- different LLMs
- sample via OpenRouter API
- sample multiple responses concurrently
- save sampled results to json file
- support easily adding different similarity metrics to compare responses

---

## Data & Setup

**Prompts Dataset**: https://huggingface.co/datasets/allenai/WildChat
**API Provider**: OpenRouter

## What to Implement

1. Sampling via OpenRouter API multiple responses for selected LLM concurrently. Use OpenAI API key for this. Save sampled results to json file. Add config file.
2. Calculate KL Divergence based on logprobs between the first N tokens of each response for the same prompt given two JSON files with sampled responses.
