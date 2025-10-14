# ABOUTME: Demo notebook for calculating cosine similarity between embeddings of two models' responses.
# ABOUTME: Uses OpenAI embeddings API to compute embeddings and calculates cosine similarity between them.

# %%
import glob
import json
import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# %%
# Parameters
embedding_model = "text-embedding-3-small"  # Options: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
responses_dir_model1 = "/workspace/projects/diffing-prompts/experiments/results/responses_openrouter/gemma-2-9b-it"
responses_dir_model2 = "/workspace/projects/diffing-prompts/experiments/results/responses_openrouter/gemma-2-9b-it-taboo-cloud"
output_dir = "/workspace/projects/diffing-prompts/experiments/results/embeddings_similarity"
max_prompts = None  # Set to a number to limit, or None to process all
os.makedirs(output_dir, exist_ok=True)

# %%
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print(f"Using embedding model: {embedding_model}")


# %%
def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI API.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model to use

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    # OpenAI API can handle batches, but has limits
    # Process in batches of 100 to be safe
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


def cosine_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Calculate pairwise cosine similarity between two sets of embeddings.

    Args:
        embeddings1: numpy array of shape (n1, embedding_dim)
        embeddings2: numpy array of shape (n2, embedding_dim)

    Returns:
        numpy array of shape (n1, n2) with cosine similarities
    """
    # Normalize embeddings to unit vectors
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Cosine similarity is just the dot product of normalized vectors
    return np.dot(embeddings1_norm, embeddings2_norm.T)


def calculate_response_similarity(
    responses_model1: List[str],
    responses_model2: List[str],
    model: str = "text-embedding-3-small"
) -> dict:
    """Calculate cosine similarity between responses from two models.

    Args:
        responses_model1: List of response texts from model 1
        responses_model2: List of response texts from model 2
        model: OpenAI embedding model to use

    Returns:
        Dictionary containing similarity scores and statistics
    """
    # Get embeddings for all responses
    all_texts = responses_model1 + responses_model2
    embeddings = get_embeddings(all_texts, model=model)

    # Split embeddings back into model1 and model2
    n1 = len(responses_model1)
    embeddings_model1 = embeddings[:n1]
    embeddings_model2 = embeddings[n1:]

    # Calculate pairwise cosine similarities
    # For each response from model1, compare with all responses from model2
    similarities = cosine_similarity_matrix(embeddings_model1, embeddings_model2)

    # Calculate various statistics
    # Average similarity: mean of all pairwise similarities
    avg_similarity = similarities.mean()

    # Max similarity: highest similarity between any pair
    max_similarity = similarities.max()

    # Min similarity: lowest similarity between any pair
    min_similarity = similarities.min()

    # Average best match: for each model1 response, find best model2 match, then average
    avg_best_match_m1_to_m2 = similarities.max(axis=1).mean()

    # Average best match: for each model2 response, find best model1 match, then average
    avg_best_match_m2_to_m1 = similarities.max(axis=0).mean()

    return {
        "avg_similarity": float(avg_similarity),
        "max_similarity": float(max_similarity),
        "min_similarity": float(min_similarity),
        "avg_best_match_m1_to_m2": float(avg_best_match_m1_to_m2),
        "avg_best_match_m2_to_m1": float(avg_best_match_m2_to_m1),
        "pairwise_similarities": similarities.tolist(),
    }


# %%
# Find all response files for both models
responses_files_model1 = sorted(glob.glob(os.path.join(responses_dir_model1, "*.json")))
responses_files_model2 = sorted(glob.glob(os.path.join(responses_dir_model2, "*.json")))

print(f"\nFound {len(responses_files_model1)} response files for model 1")
print(f"Found {len(responses_files_model2)} response files for model 2")


# %%
# Create a mapping from prompt index to files
def get_prompt_index(filepath):
    """Extract prompt index from filename like 'model_prompt_123.json'"""
    basename = os.path.basename(filepath)
    # Find the last occurrence of 'prompt_' and extract number
    parts = basename.split('prompt_')
    if len(parts) > 1:
        idx = parts[-1].split('.')[0]
        try:
            return int(idx)
        except ValueError:
            return None
    return None


# Build mappings
model1_files_by_idx = {}
for f in responses_files_model1:
    idx = get_prompt_index(f)
    if idx is not None:
        model1_files_by_idx[idx] = f

model2_files_by_idx = {}
for f in responses_files_model2:
    idx = get_prompt_index(f)
    if idx is not None:
        model2_files_by_idx[idx] = f

# Find common prompt indices
common_indices = sorted(set(model1_files_by_idx.keys()) & set(model2_files_by_idx.keys()))
print(f"\nFound {len(common_indices)} prompts with responses from both models")

if max_prompts is not None:
    common_indices = common_indices[:max_prompts]
    print(f"Processing first {len(common_indices)} prompts")


# %%
# Calculate embedding similarity for each prompt
results = []

for prompt_idx in common_indices:
    file_model1 = model1_files_by_idx[prompt_idx]
    file_model2 = model2_files_by_idx[prompt_idx]

    # Load responses from model 1
    with open(file_model1, "r") as fp:
        data_model1 = json.load(fp)

    # Load responses from model 2
    with open(file_model2, "r") as fp:
        data_model2 = json.load(fp)

    print(f"\nProcessing prompt {prompt_idx}:")
    print(f"  Model 1: {os.path.basename(file_model1)}")
    print(f"  Model 2: {os.path.basename(file_model2)}")

    prompt = data_model1["prompt"]

    # Verify prompts match
    if data_model1["prompt"] != data_model2["prompt"]:
        print("  WARNING: Prompts don't match! Skipping.")
        continue

    # Extract response texts
    responses_model1 = []
    for response_data in data_model1["responses"]:
        text = response_data["choices"][0]["message"]["content"]
        responses_model1.append(text)

    responses_model2 = []
    for response_data in data_model2["responses"]:
        text = response_data["choices"][0]["message"]["content"]
        responses_model2.append(text)

    if not responses_model1 or not responses_model2:
        print("  WARNING: No responses found! Skipping.")
        continue

    print(f"  Model 1: {len(responses_model1)} responses")
    print(f"  Model 2: {len(responses_model2)} responses")

    try:
        # Calculate similarity
        similarity_results = calculate_response_similarity(
            responses_model1,
            responses_model2,
            model=embedding_model
        )

        print(f"  Average similarity: {similarity_results['avg_similarity']:.4f}")
        print(f"  Max similarity: {similarity_results['max_similarity']:.4f}")
        print(f"  Min similarity: {similarity_results['min_similarity']:.4f}")
        print(f"  Avg best match M1→M2: {similarity_results['avg_best_match_m1_to_m2']:.4f}")
        print(f"  Avg best match M2→M1: {similarity_results['avg_best_match_m2_to_m1']:.4f}")

        results.append({
            "prompt_idx": prompt_idx,
            "file_model1": file_model1,
            "file_model2": file_model2,
            "prompt": prompt,
            "n_responses_model1": len(responses_model1),
            "n_responses_model2": len(responses_model2),
            **similarity_results
        })

    except Exception as e:
        print(f"  Error: {e}")
        continue


# %%
# Sort results by average similarity (lowest first, since low similarity = high difference)
import datetime

results.sort(key=lambda x: x["avg_similarity"])

# Create detailed results
similarity_results = [
    {
        "prompt_idx": r["prompt_idx"],
        "prompt": r["prompt"],
        "avg_similarity": r["avg_similarity"],
        "max_similarity": r["max_similarity"],
        "min_similarity": r["min_similarity"],
        "avg_best_match_m1_to_m2": r["avg_best_match_m1_to_m2"],
        "avg_best_match_m2_to_m1": r["avg_best_match_m2_to_m1"],
        "n_responses_model1": r["n_responses_model1"],
        "n_responses_model2": r["n_responses_model2"],
        "pairwise_similarities": r["pairwise_similarities"],
        "file_model1": r["file_model1"],
        "file_model2": r["file_model2"],
    }
    for r in results
]

# Save results to JSON file
output_file = os.path.join(output_dir, "prompt_embedding_similarities.json")
with open(output_file, "w") as f:
    json.dump({
        "prompt_similarities": similarity_results,
        "metadata": {
            "embedding_model": embedding_model,
            "responses_dir_model1": responses_dir_model1,
            "responses_dir_model2": responses_dir_model2,
            "n_prompts": len(similarity_results),
            "calculation_method": "cosine_similarity_of_embeddings",
            "description": "Cosine similarity between embeddings of responses from two models",
            "timestamp": datetime.datetime.now().isoformat()
        }
    }, f, indent=2)

print("\n" + "="*80)
print("SORTED RESULTS (lowest similarity = highest difference):")
print("="*80)
print(f"Results saved to: {output_file}")


# %%
# Display top results (most different responses)
for i, result in enumerate(results[:10], 1):
    print(f"\n{i}. Prompt {result['prompt_idx']}: Avg Similarity: {result['avg_similarity']:.4f}")
    print(f"   Max: {result['max_similarity']:.4f}, Min: {result['min_similarity']:.4f}")
    print(f"   Best match M1→M2: {result['avg_best_match_m1_to_m2']:.4f}")
    print(f"   Best match M2→M1: {result['avg_best_match_m2_to_m1']:.4f}")
    print(f"   Prompt: {result['prompt'][:100]}...")

# %%
# load results
results = json.load(open(os.path.join(output_dir, "prompt_embedding_similarities.json")))
# %%
# Plot average similarities
import matplotlib.pyplot as plt

# Extract avg similarities from results
avg_similarities = [r['avg_similarity'] for r in results['prompt_similarities']]

plt.figure(figsize=(12, 6))
plt.hist(avg_similarities, bins=50, edgecolor='black')
plt.title('Distribution of Average Response Similarities Between Models')
plt.xlabel('Average Cosine Similarity')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

# Add mean and median lines
plt.axvline(np.mean(avg_similarities), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(avg_similarities):.3f}')
plt.axvline(np.median(avg_similarities), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(avg_similarities):.3f}')
plt.legend()

plt.tight_layout()
plt.show()
