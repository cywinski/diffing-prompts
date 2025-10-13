# ABOUTME: Demo script for calculating KL divergence between two models' responses.
# ABOUTME: Uses jupyter-style cells to interactively explore prompt differences.

# %%
import glob
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.kl_metric import calculate_prompt_kl

# %%
# model_name = "bcywinski/gemma-2-9b-it-taboo-cloud"
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# %%
responses_dir = "/workspace/projects/diffing-prompts/experiments/results/responses_openrouter/gemma-2-9b-it"
responses_files = glob.glob(os.path.join(responses_dir, "*.json"))
print(responses_files)

# %%
n_top_logprobs = 5
for f in responses_files:
    with open(f, "r") as fp:
        data = json.load(fp)
    print(f)

    prompt = data["prompt"]
    responses_data = data["responses"]

    kl_result = calculate_prompt_kl(
        prompt=prompt,
        responses_data=responses_data,
        model=model,
        tokenizer=tokenizer,
        n_top_logprobs=n_top_logprobs,
    )

    if kl_result["average_kl"] is None:
        print("Skipping prompt because no responses")
        continue

    for kl in kl_result["response_kls"]:
        print(f"Average response KL: {kl:.6f}")
    print(f"Average prompt KL: {kl_result['average_kl']:.6f}")

# %%
