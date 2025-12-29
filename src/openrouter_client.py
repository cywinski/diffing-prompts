# ABOUTME: OpenRouter API client for sampling multiple LLM responses concurrently.
# ABOUTME: Supports concurrent requests and extracting logprobs from responses.

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import httpx
from tqdm import tqdm


class OpenRouterClient:
    """Client for interacting with OpenRouter API to sample LLM responses."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.
            base_url: Base URL for OpenRouter API.
            timeout: Request timeout in seconds.
            site_url: Optional site URL for rankings on openrouter.ai.
            site_name: Optional site name for rankings on openrouter.ai.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in OPENROUTER_API_KEY env var"
            )

        self.base_url = base_url
        self.timeout = timeout

        # Build headers for OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if site_url:
            self.headers["HTTP-Referer"] = site_url
        if site_name:
            self.headers["X-Title"] = site_name

    async def sample_response(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: bool = True,
        top_logprobs: int = 5,
        seed: Optional[int] = None,
        reasoning: bool = False,
        assistant_prefill: Optional[str] = None,
        provider: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sample a single response from the model.

        Args:
            prompt: Input prompt text.
            model: Model identifier (e.g., "openai/gpt-4").
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of top logprobs to return per token.
            seed: Optional seed for deterministic sampling (if supported by model).
            reasoning: Whether to enable reasoning.
            assistant_prefill: Optional text to prefill the assistant response.
            provider: Optional provider configuration (e.g., {"only": ["provider_name"]}).
        Returns:
            Dictionary containing the response and metadata.
        """
        # Build messages with optional assistant prefill
        messages = [{"role": "user", "content": prompt}]
        if assistant_prefill is not None:
            messages.append({"role": "assistant", "content": assistant_prefill})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning": {
                "enabled": reasoning,
            },
        }

        # Only add logprobs if requested
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs

        # Add seed if provided
        if seed is not None:
            payload["seed"] = seed

        # Add provider if provided
        if provider is not None:
            payload["provider"] = provider

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            print(payload)
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            print(response.json())
            return response.json()

    def sample(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        seed: Optional[int] = None,
        reasoning: bool = False,
        provider: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sample a single response from the model (synchronous version).

        Args:
            prompt: Input prompt text.
            model: Model identifier (e.g., "openai/gpt-4").
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of top logprobs to return per token.
            seed: Optional seed for deterministic sampling (if supported by model).
            provider: Optional provider configuration (e.g., {"only": ["provider_name"]}).
        Returns:
            Dictionary containing the response and metadata.
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning": {
                "enabled": reasoning,
            },
        }

        # Only add logprobs if requested
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs

        # Add seed if provided
        if seed is not None:
            payload["seed"] = seed

        # Add provider if provided
        if provider is not None:
            payload["provider"] = provider

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def sample_multiple_concurrent(
        self,
        prompt: str,
        model: str,
        num_samples: int,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: bool = True,
        top_logprobs: int = 5,
        reasoning: bool = False,
        max_retries: int = 5,
        assistant_prefill: Optional[str] = None,
        provider: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Sample multiple responses concurrently for the same prompt with retries.

        Args:
            prompt: Input prompt text.
            model: Model identifier.
            num_samples: Number of responses to sample.
            max_tokens: Maximum tokens to generate per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of top logprobs to return per token.
            reasoning: Whether to enable reasoning.
            max_retries: Maximum number of retry attempts for failed requests.
            assistant_prefill: Optional text to prefill the assistant response.
            provider: Optional provider configuration (e.g., {"only": ["provider_name"]}).
        Returns:
            List of response dictionaries.
        """
        valid_responses = []
        retry_count = 0

        # Start with initial number of samples needed
        remaining = num_samples

        while len(valid_responses) < num_samples and retry_count <= max_retries:
            tasks = [
                self.sample_response(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    reasoning=reasoning,
                    assistant_prefill=assistant_prefill,
                    provider=provider,
                )
                for _ in range(remaining)
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Process responses and validate
            failed_count = 0
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    print(f"Warning: Sample failed with error: {resp}")
                    failed_count += 1
                else:
                    # Validate logprobs if requested
                    if logprobs:
                        has_logprobs = self._validate_logprobs(resp)
                        if not has_logprobs:
                            print("Warning: Response missing logprobs, will retry")
                            failed_count += 1
                            continue

                    valid_responses.append(resp)

            # Update remaining count
            remaining = num_samples - len(valid_responses)

            if remaining > 0:
                retry_count += 1
                print(
                    f"Retrying {remaining} failed requests (attempt {retry_count}/{max_retries})..."
                )
            else:
                break

        if len(valid_responses) < num_samples:
            print(
                f"Warning: Only got {len(valid_responses)}/{num_samples} successful responses after {max_retries} retries"
            )

        return valid_responses

    def _validate_logprobs(self, response: Dict[str, Any]) -> bool:
        """Check if response contains valid logprobs.

        Note: This validates the raw API response before filtering.

        Args:
            response: Raw API response dictionary.

        Returns:
            True if response contains logprobs, False otherwise.
        """
        if "choices" not in response or not response["choices"]:
            return False

        # Check first choice for logprobs
        choice = response["choices"][0]
        if "logprobs" not in choice:
            return False

        # Check that logprobs is not None/empty
        logprobs_data = choice["logprobs"]
        if logprobs_data is None:
            return False

        # Check that logprobs contains content
        if isinstance(logprobs_data, dict) and "content" in logprobs_data:
            return len(logprobs_data["content"]) > 0

        return False

    async def sample_prompts_batch(
        self,
        prompts: List[str],
        model: str,
        num_samples_per_prompt: int,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: bool = True,
        top_logprobs: int = 5,
        reasoning: bool = False,
        output_dir: Optional[str] = None,
        filter_fields: bool = True,
        skip_existing: bool = True,
        max_retries: int = 5,
        provider: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Sample multiple responses for multiple prompts.

        Args:
            prompts: List of input prompts.
            model: Model identifier.
            num_samples_per_prompt: Number of responses per prompt.
            max_tokens: Maximum tokens to generate per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of top logprobs to return per token.
            reasoning: Whether to enable reasoning.
            output_dir: Optional directory to save results after each prompt.
            filter_fields: If True, filter out unnecessary fields from responses.
            skip_existing: If True, skip prompts whose output files already exist.
            max_retries: Maximum number of retry attempts for failed requests.
            provider: Optional provider configuration (e.g., {"only": ["provider_name"]}).
        Returns:
            List of dictionaries with structure:
            {
                "prompt": str,
                "model": str,
                "responses": List[Dict],
            }
        """
        results = []

        for prompt_idx, prompt in enumerate(
            tqdm(prompts, desc=f"Sampling prompts ({model})", unit="prompt")
        ):
            # Check if output file already exists
            if skip_existing and output_dir:
                model_name = model.replace("/", "_").replace(":", "_")
                filename = f"{model_name}_prompt_{prompt_idx}.json"
                filepath = os.path.join(output_dir, filename)

                if os.path.exists(filepath):
                    print(
                        f"Skipping prompt {prompt_idx} - file already exists: {filename}"
                    )
                    continue

            responses = await self.sample_multiple_concurrent(
                prompt=prompt,
                model=model,
                num_samples=num_samples_per_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                reasoning=reasoning,
                max_retries=max_retries,
                provider=provider,
            )

            sample = {
                "prompt": prompt,
                "model": model,
                "responses": responses,
            }
            results.append(sample)

            # Save immediately after sampling this prompt
            if output_dir:
                save_sample_to_json(
                    sample, output_dir, model, prompt_idx, filter_fields, logprobs
                )

        return results


def filter_response_fields(
    response: Dict[str, Any], include_logprobs: bool = True
) -> Dict[str, Any]:
    """Filter OpenRouter API responses to match Google client format.

    Flattens the choices array and extracts content to create a simpler structure.

    Args:
        response: Raw response dictionary from OpenRouter API.
        include_logprobs: Whether to include logprobs in filtered output.

    Returns:
        Filtered response dictionary with flattened structure.
    """
    filtered = {}

    # Keep model field at top level
    if "model" in response:
        filtered["model"] = response["model"]

    # Extract content from first choice
    if "choices" in response and response["choices"]:
        choice = response["choices"][0]

        # Extract text from message
        if "message" in choice and "content" in choice["message"]:
            filtered["text"] = choice["message"]["content"]

        # Keep finish_reason
        if "finish_reason" in choice:
            filtered["finish_reason"] = choice["finish_reason"]

        # Keep logprobs but filter bytes (only if include_logprobs=True)
        if include_logprobs and "logprobs" in choice and choice["logprobs"]:
            # Defensive check: ensure logprobs is a dict before processing
            if isinstance(choice["logprobs"], dict):
                filtered["logprobs"] = {}
                if "content" in choice["logprobs"]:
                    filtered_content = []
                    for token_data in choice["logprobs"]["content"]:
                        filtered_token = {
                            "token": token_data["token"],
                            "logprob": token_data["logprob"],
                        }
                        # Filter top_logprobs
                        if "top_logprobs" in token_data:
                            filtered_token["top_logprobs"] = [
                                {
                                    "token": top_token["token"],
                                    "logprob": top_token["logprob"],
                                }
                                for top_token in token_data["top_logprobs"]
                            ]
                        filtered_content.append(filtered_token)
                    filtered["logprobs"]["content"] = filtered_content

    # Keep usage stats if present
    if "usage" in response:
        filtered["usage"] = response["usage"]

    return filtered


def save_sample_to_json(
    sample: Dict[str, Any],
    output_dir: str,
    model_id: str,
    prompt_idx: int,
    filter_fields: bool = True,
    include_logprobs: bool = True,
) -> None:
    """Save a single sampled response to JSON file.

    Args:
        sample: Sample dictionary from sample_prompts_batch.
        output_dir: Directory to save JSON files.
        model_id: Model identifier (e.g., "openai/gpt-4").
        prompt_idx: Index of the prompt.
        filter_fields: If True, filter out unnecessary fields from responses.
        include_logprobs: If True, include logprobs in filtered output.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create filename-safe model name
    model_name = model_id.replace("/", "_").replace(":", "_")

    # Filter sample if requested
    if filter_fields:
        filtered_sample = {
            "prompt": sample["prompt"],
            "model": sample["model"],
            "responses": [
                filter_response_fields(resp, include_logprobs)
                for resp in sample["responses"]
            ],
        }
    else:
        filtered_sample = sample

    # Create filename: {model_name}_prompt_{idx}.json
    filename = f"{model_name}_prompt_{prompt_idx}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(filtered_sample, f, indent=2)


def save_samples_to_json(
    samples: List[Dict[str, Any]],
    output_dir: str,
    model_id: str,
    filter_fields: bool = True,
    include_logprobs: bool = True,
) -> None:
    """Save sampled responses to JSON files, one file per prompt.

    Args:
        samples: List of sample dictionaries from sample_prompts_batch.
        output_dir: Directory to save JSON files.
        model_id: Model identifier (e.g., "openai/gpt-4").
        filter_fields: If True, filter out unnecessary fields from responses.
        include_logprobs: If True, include logprobs in filtered output.
    """
    for prompt_idx, sample in enumerate(samples):
        save_sample_to_json(
            sample, output_dir, model_id, prompt_idx, filter_fields, include_logprobs
        )

    print(f"Saved {len(samples)} prompts to {output_dir} as individual JSON files")
