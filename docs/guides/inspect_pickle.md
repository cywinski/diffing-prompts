# Inspecting Pickle Files

A utility to quickly inspect the contents of pickle files containing LLM responses.

## Quick Start

```bash
# Show first 5 entries (default)
./scripts/inspect_pickle.sh experiments/results/myfile.pkl

# Show first 10 entries
./scripts/inspect_pickle.sh experiments/results/myfile.pkl 10

# Show logprobs information
./scripts/inspect_pickle.sh experiments/results/myfile.pkl 5 --show_logprobs

# Hide response content (just show metadata)
./scripts/inspect_pickle.sh experiments/results/myfile.pkl 5 --show_responses=0
```

## Using the Python Script Directly

For more control, use the Python script directly:

```bash
.venv/bin/python src/inspect_pickle.py <pickle_file> [options]
```

### Options

- `--num_entries`: Number of entries to display (default: 5)
- `--max_depth`: Maximum depth for nested structure display (default: 3)
- `--show_responses`: Whether to show full response content (default: True)
- `--show_logprobs`: Whether to show logprobs data (default: False)

### Examples

```bash
# Show first 3 entries with logprobs
.venv/bin/python src/inspect_pickle.py results.pkl --num_entries 3 --show_logprobs

# Show 10 entries without response content
.venv/bin/python src/inspect_pickle.py results.pkl --num_entries 10 --show_responses=False
```

## Output Format

The script displays:
- **Prompt**: First 200 characters of each prompt
- **Model**: Model identifier used
- **Number of responses**: Total responses per prompt
- **Response content**: First 150 characters of each response (up to 3 shown)
- **Finish reason**: How the model ended generation (e.g., "stop", "length")
- **Usage stats**: Token counts (prompt, completion, total)
- **Logprobs** (optional): Token-level log probabilities

## File Structure

Pickle files contain a list of dictionaries with this structure:

```python
[
    {
        "prompt": "User's prompt text",
        "model": "model/identifier",
        "responses": [
            {
                "model": "model/identifier",
                "choices": [
                    {
                        "message": {"content": "Generated text..."},
                        "finish_reason": "stop",
                        "logprobs": {...}  # Optional
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 50,
                    "total_tokens": 60
                }
            },
            # ... more responses
        ]
    },
    # ... more prompts
]
```
