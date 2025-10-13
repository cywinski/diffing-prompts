# Experiment: Find maximally different responses between two models for the same prompt using KL Divergence metric

**Goal**: I have two JSON files with sampled responses from two models for the same prompt along with logprobs. I want to search for such prompt that gives maximally different responses between the two models.
**KL Divergence approach:** Algorithm for calculating difference metric value between two prompts:
Assume having 2 prompts, 2 models, and for each prompt having 5 responses from each model.
For each prompt calculate metric as follows:
- Take response from one model
- Autoregressively put this response into the second model, calculating KL divergence per token for each next token
- Calculate average KL divergence per token for the whole response
- Normalize by the length of the response
- Calculate this for each combination of responses for this prompt
- Calculate total average: this is a final metric value for this prompt
- At the end, each prompt should have assigned a single metric value, so that I can sort them accordingly

---

## Data & Setup

**Saved 2 JSON files**

## What to Implement

1. Simple demo in a jupyter-style python script which will load two JSON files, calculate the metric value for each prompt, and sort them accordingly. Enable calculating it only for the first N prompts from the file to only show the demo. The engine for metric calculation should already be implemented in the code as normal python file and it should be imported into the notebook.
