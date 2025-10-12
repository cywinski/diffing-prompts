#!/bin/bash
# Experiment: Sample multiple LLM responses via OpenRouter API
# Date: 2025-10-12
# Goal: Sample responses from two models for the same prompts to compare differences

set -e  # Exit on error

# Configuration
CONFIG_FILE="experiments/configs/sample_responses_openrouter.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/results/sample_responses_openrouter_${TIMESTAMP}"
LOG_FILE="experiments/logs/sample_responses_openrouter_${TIMESTAMP}.log"
# Create directories
mkdir -p experiments/logs
mkdir -p experiments/results

# Run sampling
echo "Starting LLM response sampling..."
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo ""

python src/sample_responses.py \
    --config "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "✓ Sampling complete!"
