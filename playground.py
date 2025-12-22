# %%
import json
import math
import os
from typing import Optional, Dict, List
from src.calculate_kl_divergence_batch import KLDivergenceBatchCalculator

# %%
dir_path = "/Users/bcywinski/work/code/diffing-prompts/experiments/results/kl/gemini-2.5-flash-lite-preview-09-2025"
# %%
files = sorted(os.listdir(dir_path))


# %%
response_idx = 0
n_tokens = 100
text = ""
stop_token = " small"
for file in files:
    with open(os.path.join(dir_path, file), "r") as f:
        data = json.load(f)
        token_data = data["responses"][response_idx]["token_details"]
        for t in token_data:
            if t["token"] == stop_token:
                break
            text += t["token"]
        break
# %%
text
# %%
import os

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

api_url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
# %%
model1 = "google/gemini-2.5-flash-lite-preview-09-2025"
model2 = "google/gemini-2.5-flash-lite"
payload = {
    "model": model1,
    "temperature": 1.0,
    "max_tokens": 20,
    "stream": False,
    "messages": [
        {
            "role": "user",
            "content": "what is the average salary in egypt per month for Data Science, Data Analyst, Business Analyst, Cyber Security, Cloud Engineering ?",
        },
        {
            "role": "assistant",
            "content": "It's challenging to provide one single, definitive \"average salary\" for these roles in Egypt for a few key reasons:\n\n1. **Rapidly Evolving Market:** Tech salaries in Egypt are increasing quickly, especially in high-demand fields like Data Science and Cloud Engineering.\n2. **Company Type & Location:** Salaries vary drastically between international tech hubs (like Cairo's tech parks), local Egyptian companies, startups, and multinational corporations (MNCs). MNCs almost always pay significantly more.\n3. **Experience Level:** An entry-level analyst will earn a fraction of what a Senior or Lead professional earns.\n4. **Currency Fluctuations:** Salaries are often negotiated based on an equivalent USD rate, which impacts the L.E. (Egyptian Pound) figures.\n\nHowever, I can provide **realistic salary ranges in Egyptian Pounds (EGP) per month** for different experience levels, based on recent market data (late 2023/early 2024 data trends).\n\n---\n\n## Estimated Monthly Salary Ranges in Egypt (EGP)\n\nThe figures below are **gross salary estimates** (before taxes and deductions). They aim to represent a broad market average, favoring roles in major tech hubs in Cairo (New Cairo, 6th of October City) or Alexandria.\n\n| Role | Entry-Level (0–2 Years) | Mid-Level (3–6 Years) | Senior/Lead (7+ Years) |\n| :--- | :--- | :--- | :--- |\n| **Data Analyst** | 8,000 – 15,000 EGP | 16,000 – ",
        },
    ],
}
response = requests.post(api_url, json=payload, headers=headers)
print(response.json()["choices"][0]["message"]["content"])
# %%
