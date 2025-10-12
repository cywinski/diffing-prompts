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
            raise ValueError("API key must be provided or set in OPENROUTER_API_KEY env var")

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

        Returns:
            Dictionary containing the response and metadata.
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs if logprobs else None,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
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
    ) -> List[Dict[str, Any]]:
        """Sample multiple responses concurrently for the same prompt.

        Args:
            prompt: Input prompt text.
            model: Model identifier.
            num_samples: Number of responses to sample.
            max_tokens: Maximum tokens to generate per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of top logprobs to return per token.

        Returns:
            List of response dictionaries.
        """
        tasks = [
            self.sample_response(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )
            for _ in range(num_samples)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any exceptions and convert to proper format
        valid_responses = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                print(f"Warning: Sample {i} failed with error: {resp}")
            else:
                valid_responses.append(resp)

        return valid_responses

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

        Returns:
            List of dictionaries with structure:
            {
                "prompt": str,
                "model": str,
                "responses": List[Dict],
            }
        """
        results = []

        for prompt in tqdm(prompts, desc=f"Sampling prompts ({model})", unit="prompt"):
            responses = await self.sample_multiple_concurrent(
                prompt=prompt,
                model=model,
                num_samples=num_samples_per_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )

            results.append({
                "prompt": prompt,
                "model": model,
                "responses": responses,
            })

        return results


def save_samples_to_json(
    samples: List[Dict[str, Any]],
    output_path: str,
    indent: int = 2,
) -> None:
    """Save sampled responses to a JSON file.

    Args:
        samples: List of sample dictionaries from sample_prompts_batch.
        output_path: Path to output JSON file.
        indent: JSON indentation level.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=indent)

    print(f"Saved {len(samples)} prompts with samples to {output_path}")
