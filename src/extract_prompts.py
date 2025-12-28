# ABOUTME: Script to extract user prompts from JSONL files
# ABOUTME: Reads JSONL and saves prompts one per line to a text file

import json
import fire


def extract_prompts(input_file: str, output_file: str = "prompts.json"):
    """Extract user prompts from JSONL file and save to JSON file.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSON file (default: prompts.json)
    """
    prompts = []

    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            # The data is a dict with hash as key, containing conversation and response
            for key, value in data.items():
                if "conversation" in value:
                    # Extract user message from conversation
                    for msg in value["conversation"]:
                        if msg.get("role") == "user":
                            prompts.append(msg["content"])
                            break

    # Save prompts to JSON file
    with open(output_file, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(prompts)} prompts to {output_file}")


if __name__ == "__main__":
    fire.Fire(extract_prompts)
