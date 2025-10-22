# ABOUTME: Demo script for loading tokenizers for two models and applying chat templates
# ABOUTME: Shows how different models tokenize the same prompt/response with their chat formats

# %%
# Parameters
model_1 = "meta-llama/Llama-3.2-1B-Instruct"
model_2 = "meta-llama/Llama-3.2-3B-Instruct"

sample_prompt = "What is the capital of France?"
sample_response = "The capital of France is Paris."

# %%
# Load tokenizers
from transformers import AutoTokenizer

print(f"Loading tokenizer for {model_1}...")
tokenizer_1 = AutoTokenizer.from_pretrained(model_1)

print(f"Loading tokenizer for {model_2}...")
tokenizer_2 = AutoTokenizer.from_pretrained(model_2)

# %%
# Create chat messages
messages = [
    {"role": "user", "content": sample_prompt},
    {"role": "assistant", "content": sample_response}
]

# %%
# Apply chat templates and tokenize
print("\n" + "="*80)
print(f"MODEL 1: {model_1}")
print("="*80)

# Apply chat template
formatted_1 = tokenizer_1.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
print("\nFormatted prompt:")
print(formatted_1)

# Tokenize
tokens_1 = tokenizer_1.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False
)
print(f"\nNumber of tokens: {len(tokens_1)}")
print(f"Token IDs: {tokens_1[:20]}...")  # Show first 20 tokens

# Decode to show what each token represents
print("\nFirst 10 decoded tokens:")
for i, token_id in enumerate(tokens_1[:10]):
    token_str = tokenizer_1.decode([token_id])
    print(f"  {i}: {token_id} -> '{token_str}'")

# %%
print("\n" + "="*80)
print(f"MODEL 2: {model_2}")
print("="*80)

# Apply chat template
formatted_2 = tokenizer_2.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
print("\nFormatted prompt:")
print(formatted_2)

# Tokenize
tokens_2 = tokenizer_2.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False
)
print(f"\nNumber of tokens: {len(tokens_2)}")
print(f"Token IDs: {tokens_2[:20]}...")  # Show first 20 tokens

# Decode to show what each token represents
print("\nFirst 10 decoded tokens:")
for i, token_id in enumerate(tokens_2[:10]):
    token_str = tokenizer_2.decode([token_id])
    print(f"  {i}: {token_id} -> '{token_str}'")

# %%
print(tokenizer_1.convert_ids_to_tokens(tokens_1))
print(tokenizer_2.convert_ids_to_tokens(tokens_2))
# %%
print(tokenizer_1.vocab_size)
print(tokenizer_2.vocab_size)
# %%
# print first 100 tokens of each model
print(tokenizer_1.convert_ids_to_tokens(list(range(100))))
print(tokenizer_2.convert_ids_to_tokens(list(range(100))))
# %%
