#!/bin/bash

set -e  # Exit on error

# Configuration
CONFIG_FILE="experiments/configs/sample_responses_openrouter_vllm_served.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/results/responses_openrouter/gemma-2-9b-it-taboo-cloud"
LOG_FILE="experiments/logs/responses_openrouter_gemma-2-9b-it-taboo-cloud_${TIMESTAMP}.log"
# Create directories
mkdir -p experiments/logs
mkdir -p experiments/results/responses_openrouter/gemma-2-9b-it-taboo-cloud

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
