#!/bin/bash
# Experiment: Sample multiple LLM responses via OpenRouter API
# Date: 2025-10-12
# Goal: Sample responses from two models for the same prompts to compare differences

set -e  # Exit on error

# Configuration
CONFIG_FILE="experiments/configs/2025-10-12_sample_responses.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/results/2025-10-12_sample_responses_${TIMESTAMP}"
LOG_FILE="experiments/logs/2025-10-12_sample_responses_${TIMESTAMP}.log"

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY=your_api_key"
    exit 1
fi

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
echo "Results saved to: ${OUTPUT_DIR}"
echo "Log saved to: ${LOG_FILE}"
