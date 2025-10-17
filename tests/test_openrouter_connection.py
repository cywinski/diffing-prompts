# ABOUTME: Quick test script to verify OpenRouter API connectivity and configuration.
# ABOUTME: Helps diagnose 404 and other API errors.
# %%
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

load_dotenv()

print("=" * 80)
print("OPENROUTER API TEST")
print("=" * 80)

# Check 1: API Key
api_key = os.getenv("OPENROUTER_API_KEY")
if api_key:
    print(f"✓ API key found: {api_key[:10]}...{api_key[-4:]}")
else:
    print("✗ No API key found in environment")
    print("  Set OPENROUTER_API_KEY in your .env file")
    sys.exit(1)

# Check 2: Direct API test with httpx
print("\n" + "-" * 80)
print("Testing direct API call with httpx...")
print("-" * 80)
# %%
import httpx

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# Test with a simple model
test_model = "openai/gpt-3.5-turbo"
payload = {
    "model": test_model,
    "messages": [{"role": "user", "content": "What's up"}],
    "max_tokens": 10,
}

try:
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            print("✓ API call successful!")
            result = response.json()
            print(f"Response: {result['choices'][0]['message']['content']}")
        elif response.status_code == 404:
            print("✗ 404 Error: Endpoint not found")
            print(f"Response body: {response.text}")
            print("\nPossible causes:")
            print("  - Model name might be incorrect")
            print("  - API endpoint might have changed")
        elif response.status_code == 401:
            print("✗ 401 Error: Unauthorized")
            print("  Your API key might be invalid")
        else:
            print(f"✗ Error {response.status_code}")
            print(f"Response: {response.text}")

except Exception as e:
    print(f"✗ Exception occurred: {e}")
    import traceback
    traceback.print_exc()
# %%
# Check 3: Test with OpenRouterClient
print("\n" + "-" * 80)
print("Testing with OpenRouterClient...")
print("-" * 80)

try:
    from openrouter_client import OpenRouterClient

    client = OpenRouterClient(api_key=api_key)
    print("✓ Client initialized")

    print(f"\nTesting model: {test_model}")
    response = client.sample(
        prompt="Say 'Hello'",
        model=test_model,
        max_tokens=10,
        temperature=1.0,
    )

    print("✓ Sample call successful!")
    print(f"Response: {response['choices'][0]['message']['content']}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Check 4: Test alternative models
print("\n" + "-" * 80)
print("Testing alternative models...")
print("-" * 80)

test_models = [
    "google/gemma-2-9b-it",
    "mistralai/mistral-7b-instruct",
    "anthropic/claude-3-haiku",
]

for model in test_models:
    try:
        print(f"\nTesting {model}...")
        response = client.sample(
            prompt="What's up",
            model=model,
            max_tokens=5,
            temperature=1.0,
        )
        print(f"  ✓ {model} works!")
    except Exception as e:
        print(f"  ✗ {model} failed: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

# %%
