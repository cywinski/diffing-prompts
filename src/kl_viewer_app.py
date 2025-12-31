# ABOUTME: Simple Flask app to visualize KL divergence results from JSON files
# ABOUTME: Displays prompts and responses with token-level heatmaps based on KL divergence

import json
import os
from pathlib import Path
from flask import Flask, render_template, jsonify, request

# Set template folder relative to this file
template_dir = Path(__file__).parent / "templates"
app = Flask(__name__, template_folder=str(template_dir))


def get_json_files(directory: str):
    """Get all JSON files in the specified directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return sorted([f.name for f in dir_path.glob("*.json")])


def replace_llama_special_chars(text: str) -> str:
    """Replace Llama tokenizer special characters with actual characters.

    Llama uses Ġ (\u0120) for spaces and Ċ (\u010a) for newlines.
    """
    return text.replace("\u0120", " ").replace("\u010a", "\n")


def process_tokens_in_data(data):
    """Recursively process data to replace Llama special chars in token strings."""
    if isinstance(data, str):
        return replace_llama_special_chars(data)
    elif isinstance(data, list):
        return [process_tokens_in_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: process_tokens_in_data(value) for key, value in data.items()}
    return data


def load_json_file(directory: str, filename: str):
    """Load a JSON file from the directory."""
    file_path = Path(directory) / filename
    if not file_path.exists():
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    return process_tokens_in_data(data)


@app.route("/")
def index():
    """Render the main page."""
    return render_template(
        "index.html",
        default_directory=app.config.get("DEFAULT_DIRECTORY", ""),
        normalize_by_entropy_default=app.config.get(
            "NORMALIZE_BY_ENTROPY_DEFAULT", False
        ),
    )


@app.route("/api/files")
def list_files():
    """API endpoint to list all JSON files in the directory."""
    directory = request.args.get("directory", "")
    if not directory:
        return jsonify({"error": "directory parameter required"}), 400

    files = get_json_files(directory)
    return jsonify({"files": files})


@app.route("/api/file")
def get_file():
    """API endpoint to get a specific JSON file."""
    directory = request.args.get("directory", "")
    filename = request.args.get("filename", "")

    if not directory or not filename:
        return jsonify({"error": "directory and filename parameters required"}), 400

    data = load_json_file(directory, filename)
    if data is None:
        return jsonify({"error": "File not found"}), 404

    return jsonify(data)


def main(
    directory: str = "/Users/bcywinski/work/code/diffing-prompts/experiments/results/kl/gemini-2.5-flash-lite-preview-09-2025",
    port: int = 5000,
    normalize_by_entropy: bool = False,
):
    """Run the KL Divergence Viewer Flask app.

    Args:
        directory: Directory path containing JSON files to display
        port: Port number to run the server on (default: 5000)
        normalize_by_entropy: Whether to display KL divided by entropy by default
    """
    print(f"KL Divergence Viewer")
    print(f"Serving files from: {directory}")
    print(f"Open http://localhost:{port} in your browser")

    app.config["DEFAULT_DIRECTORY"] = directory
    app.config["NORMALIZE_BY_ENTROPY_DEFAULT"] = normalize_by_entropy
    app.run(debug=True, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
