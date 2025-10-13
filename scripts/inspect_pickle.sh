#!/bin/bash
# Simple wrapper script to inspect pickle files
# Usage: ./scripts/inspect_pickle.sh <pickle_file> [num_entries]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PICKLE_FILE="$1"
NUM_ENTRIES="${2:-5}"  # Default to 5 if not specified

if [ -z "$PICKLE_FILE" ]; then
    echo "Usage: $0 <pickle_file> [num_entries]"
    echo ""
    echo "Options:"
    echo "  pickle_file    Path to the pickle file to inspect"
    echo "  num_entries    Number of entries to display (default: 5)"
    echo ""
    echo "Additional options can be passed using --flag syntax:"
    echo "  --show_logprobs      Show logprobs data"
    echo "  --show_responses=0   Hide response content"
    echo ""
    echo "Example:"
    echo "  $0 experiments/results/myfile.pkl 3"
    echo "  $0 experiments/results/myfile.pkl --num_entries 3 --show_logprobs"
    exit 1
fi

# Run the Python script using the virtual environment
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/inspect_pickle.py" "$PICKLE_FILE" --num_entries "$NUM_ENTRIES" "${@:3}"
