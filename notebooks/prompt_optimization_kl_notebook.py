# ABOUTME: Demo notebook for optimizing prompts to maximize KL divergence between two models' responses.
# ABOUTME: Uses gradient-based optimization on prompt embeddings, sampling via OpenRouter API.

# %%
import json
import os
import sys

import httpx
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
# %%
# =============================================================================
# PARAMETERS - Modify these to configure the optimization
# =============================================================================

# Target models to compare (sampled via OpenRouter)
target_model_1 = "google/gemma-3-4b-it"
target_model_2 = "google/gemma-3-12b-it"

# Local model for KL divergence calculation (needs gradients)
scoring_model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Optimization parameters
num_iterations = 1
num_samples_per_model = 1  # Number of response samples per iteration
learning_rate = 0.1
max_response_tokens = 100
temperature = 0.0
use_seed = True  # Use same seed for both models to ensure comparable sampling

# Prompt optimization settings
prompt_length = 1  # Number of tokens in optimized prompt
seed_prompt = None  # Optional: seed prompt to start from (or None for random init)

# OpenRouter API settings

api_key = os.getenv("OPENROUTER_API_KEY")

# Embeddings API settings (for cosine similarity scoring)
embedding_model = "text-embedding-3-small"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Output settings
output_dir = "/workspace/projects/diffing-prompts/experiments/results/prompt_optimization"
os.makedirs(output_dir, exist_ok=True)

# %%
# Load scoring model (with gradients enabled)
print(f"\nLoading scoring model: {scoring_model_name}")
scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model_name, trust_remote_code=True)
scoring_model = AutoModelForCausalLM.from_pretrained(
    scoring_model_name,
    torch_dtype=torch.bfloat16,  # Use float32 for gradient computation
    device_map="auto",
    trust_remote_code=True,
)
scoring_model.eval()
print("Scoring model loaded successfully")

# %%
# OpenRouter API setup
openrouter_base_url = "https://openrouter.ai/api/v1"
openrouter_headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# %%
# =============================================================================
# PROMPT INITIALIZATION
# =============================================================================

print("\n" + "="*80)
print("INITIALIZING PROMPT")
print("="*80)

# Initialize prompt embeddings
if seed_prompt:
    # Start from seed prompt
    seed_tokens = scoring_tokenizer.encode(seed_prompt, add_special_tokens=False)
    # Pad or truncate to desired length
    if len(seed_tokens) < prompt_length:
        # Pad with random tokens
        vocab_size = scoring_tokenizer.vocab_size
        random_tokens = torch.randint(0, vocab_size, (prompt_length - len(seed_tokens),))
        init_token_ids = torch.cat([torch.tensor(seed_tokens), random_tokens])
    else:
        init_token_ids = torch.tensor(seed_tokens[:prompt_length])
    print(f"Initialized from seed prompt: '{seed_prompt}'")
else:
    # Random initialization
    vocab_size = scoring_tokenizer.vocab_size
    init_token_ids = torch.randint(0, vocab_size, (prompt_length,))
    print("Randomly initialized prompt tokens")

# Get initial embeddings
embedding_layer = scoring_model.get_input_embeddings()
prompt_embeddings = torch.nn.Parameter(
    embedding_layer(init_token_ids.to(scoring_model.device)).detach().clone(),
    requires_grad=True,
)

# Optimizer for standard gradient descent on the prompt embeddings
optimizer = torch.optim.SGD([prompt_embeddings], lr=learning_rate)

print(f"Initial prompt tokens: {init_token_ids.tolist()}")
print(f"Initial prompt tokens: {scoring_tokenizer.convert_ids_to_tokens(init_token_ids.tolist())}")
print(f"Initial prompt text: {scoring_tokenizer.decode(init_token_ids)}")

# %%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def embeddings_to_tokens(embeddings, embedding_layer):
    """Project continuous embeddings back to discrete tokens."""
    # Get all token embeddings
    all_embeddings = embedding_layer.weight  # [vocab_size, embed_dim]

    # Find nearest tokens (cosine similarity)
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=-1)

    similarities = torch.matmul(embeddings_norm, all_embeddings_norm.T)
    token_ids = similarities.argmax(dim=-1)

    return token_ids


def extract_response_text(response_json):
    """Extract assistant text from OpenRouter chat completion JSON."""
    try:
        choice = response_json.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")
        if content is None:
            # Some providers return 'text' at the top level of choice
            content = choice.get("text")
        return content or ""
    except Exception:
        return ""


def get_embeddings(texts, model: str = "text-embedding-3-small"):
    """Get embeddings for a list of texts using OpenAI API (batched)."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = openai_client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Calculate pairwise cosine similarity between two sets of embeddings."""
    if embeddings1.size == 0 or embeddings2.size == 0:
        return np.zeros((embeddings1.shape[0], embeddings2.shape[0]), dtype=np.float32)
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    return np.dot(embeddings1_norm, embeddings2_norm.T)


def calculate_embedding_similarity_score(prompt_text, responses_model1, responses_model2):
    """Compute score as negative average cosine similarity of response embeddings.

    Returns a torch scalar on the scoring model device. Higher score => more different.
    """
    texts_1 = [t for t in responses_model1 if isinstance(t, str) and t]
    texts_2 = [t for t in responses_model2 if isinstance(t, str) and t]
    if not texts_1 or not texts_2:
        return torch.tensor(0.0, device=scoring_model.device)

    # Fetch embeddings via OpenAI API
    all_texts = texts_1 + texts_2
    embeddings = get_embeddings(all_texts, model=embedding_model)
    n1 = len(texts_1)
    emb_1 = embeddings[:n1]
    emb_2 = embeddings[n1:]

    # Compute average cosine similarity
    sim_matrix = cosine_similarity_matrix(emb_1, emb_2)
    avg_similarity = float(sim_matrix.mean())

    # Score is negative similarity (maximize difference)
    score_value = -avg_similarity
    return torch.tensor(score_value, device=scoring_model.device)


def sample_model_responses_httpx(
    *,
    prompt_text: str,
    model_id: str,
    num_samples: int,
    seed: int | None,
    base_url: str,
    headers: dict,
    max_tokens: int,
    temperature: float,
    timeout_s: float = 60.0,
):
    """Sample multiple responses using httpx and the OpenRouter Chat Completions API."""
    responses = []
    with httpx.Client(timeout=timeout_s) as client:
        for _ in range(num_samples):
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if seed is not None:
                payload["seed"] = seed
            resp = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            responses.append(resp.json())
    return responses

def calculate_kl_divergence_score(prompt_text, responses_model1, responses_model2):
    # Deprecated: Use embedding similarity score instead
    return calculate_embedding_similarity_score(prompt_text, responses_model1, responses_model2)


# %%
# =============================================================================
# OPTIMIZATION LOOP
# =============================================================================

print("\n" + "="*80)
print("STARTING OPTIMIZATION")
print("="*80)

optimization_history = []

for iteration in range(num_iterations):
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration + 1}/{num_iterations}")
    print(f"{'='*80}")

    # Project embeddings to tokens for sampling
    current_token_ids = embeddings_to_tokens(prompt_embeddings, embedding_layer)
    current_prompt_text = scoring_tokenizer.decode(current_token_ids)

    print(f"\nCurrent prompt: {current_prompt_text}")
    print(f"Token IDs: {current_token_ids.tolist()}")

    # Sample responses from both target models via OpenRouter
    # Use same seed for both models to ensure comparable sampling
    current_seed = iteration if use_seed else None
    print(f"\nSampling {num_samples_per_model} responses from each model (seed={current_seed})...")

    try:
        responses_1_json = sample_model_responses_httpx(
            prompt_text=current_prompt_text,
            model_id=target_model_1,
            num_samples=num_samples_per_model,
            seed=current_seed,
            base_url=openrouter_base_url,
            headers=openrouter_headers,
            max_tokens=max_response_tokens,
            temperature=temperature,
        )
        responses_2_json = sample_model_responses_httpx(
            prompt_text=current_prompt_text,
            model_id=target_model_2,
            num_samples=num_samples_per_model,
            seed=current_seed,
            base_url=openrouter_base_url,
            headers=openrouter_headers,
            max_tokens=max_response_tokens,
            temperature=temperature,
        )

        print(f"✓ Got {len(responses_1_json)} responses from {target_model_1}")
        print(f"✓ Got {len(responses_2_json)} responses from {target_model_2}")

        # Extract text from responses
        response_texts_1 = [extract_response_text(r) for r in responses_1_json]
        response_texts_2 = [extract_response_text(r) for r in responses_2_json]

        # Display sample responses
        print("\nSample responses:")
        if response_texts_1 and response_texts_1[0]:
            print(f"  Model 1: {response_texts_1[0]}...")
        if response_texts_2 and response_texts_2[0]:
            print(f"  Model 2: {response_texts_2[0]}...")

    except Exception as e:
        print(f"✗ Error sampling responses: {e}")
        print("  Skipping this iteration...")
        continue

    # Calculate embedding similarity score (negative avg cosine similarity)
    print("\nCalculating embedding similarity score...")
    kl_score = calculate_embedding_similarity_score(
        current_prompt_text, response_texts_1, response_texts_2
    )

    print(f"Score: {kl_score.item():.6f}")

    # Standard gradient descent on loss = -score (equivalent to ascent on score)
    print("\nComputing gradients (SGD)...")
    if kl_score.requires_grad:
        optimizer.zero_grad()
        loss = -kl_score
        loss.backward()
        if prompt_embeddings.grad is not None:
            grad_norm = prompt_embeddings.grad.norm().item()
            print(f"Gradient norm: {grad_norm:.6f}")
        else:
            print("Warning: No gradient computed")
        optimizer.step()
    else:
        print("Score is non-differentiable w.r.t. prompt embeddings; skipping optimization step.")

    # Save iteration results
    iteration_result = {
        "iteration": iteration + 1,
        "prompt_text": current_prompt_text,
        "token_ids": current_token_ids.tolist(),
        "kl_score": kl_score.item(),
        "num_responses_model1": len(response_texts_1),
        "num_responses_model2": len(response_texts_2),
        "seed": current_seed,
    }
    optimization_history.append(iteration_result)

    print(f"\n{'─'*80}")

# %%
# =============================================================================
# FINAL RESULTS
# =============================================================================

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)

# Get final prompt
final_token_ids = embeddings_to_tokens(prompt_embeddings, embedding_layer)
final_prompt_text = scoring_tokenizer.decode(final_token_ids)

print(f"\nFinal optimized prompt: {final_prompt_text}")
print(f"Final token IDs: {final_token_ids.tolist()}")

# Print score progression
print("\n" + "="*80)
print("SCORE PROGRESSION")
print("="*80)
for result in optimization_history:
    print(f"Iteration {result['iteration']:2d}: KL Score = {result['kl_score']:.6f} | Prompt: {result['prompt_text'][:60]}...")

# Find best iteration
best_iteration = max(optimization_history, key=lambda x: x['kl_score'])
print(f"\nBest iteration: {best_iteration['iteration']}")
print(f"Best KL score: {best_iteration['kl_score']:.6f}")
print(f"Best prompt: {best_iteration['prompt_text']}")

# %%
# =============================================================================
# SAVE RESULTS
# =============================================================================

output_file = os.path.join(output_dir, "prompt_optimization_results.json")
with open(output_file, "w") as f:
    json.dump({
        "config": {
            "target_model_1": target_model_1,
            "target_model_2": target_model_2,
            "scoring_model": scoring_model_name,
            "num_iterations": num_iterations,
            "learning_rate": learning_rate,
            "prompt_length": prompt_length,
            "num_samples_per_model": num_samples_per_model,
            "max_response_tokens": max_response_tokens,
            "temperature": temperature,
            "seed_prompt": seed_prompt,
            "use_seed": use_seed,
        },
        "final_result": {
            "prompt_text": final_prompt_text,
            "token_ids": final_token_ids.tolist(),
        },
        "best_iteration": best_iteration,
        "optimization_history": optimization_history,
    }, f, indent=2)

print(f"\nResults saved to: {output_file}")
