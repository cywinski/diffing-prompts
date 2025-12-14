# Examples

This directory contains example scripts demonstrating various analyses and workflows.

## Available Examples

### KL Divergence Analysis (`kl_divergence_example.py`)

Demonstrates how to:
- Load KL divergence results from multiple files
- Calculate summary statistics across all responses
- Find the most divergent responses
- Visualize KL divergence distributions

**Usage:**
```bash
python examples/kl_divergence_example.py \
  --result-dir experiments/results/kl_divergence \
  --top-k 10 \
  --plot plots/kl_distribution.png
```

**Output:**
- Summary statistics printed to console
- List of top-k most divergent responses
- Histogram plots of KL divergence distributions
