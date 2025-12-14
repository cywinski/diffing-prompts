# ABOUTME: Google Gemini API client for sampling multiple LLM responses concurrently.
# ABOUTME: Supports concurrent requests and extracting logprobs from responses.

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.auth
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from tqdm import tqdm


class GoogleClient:
    """Client for interacting with Google Gemini API to sample LLM responses."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "global",
        credentials: Optional[Any] = None,
    ):
        """Initialize Google Gemini client.

        Args:
            project_id: Google Cloud project ID. If None, reads from GOOGLE_CLOUD_PROJECT env var.
            location: Google Cloud region (default: "global").
            credentials: Optional credentials object. If None, uses Application Default Credentials.
        """
        # Load environment variables
        load_dotenv()

        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError(
                "Project ID must be provided or set in GOOGLE_CLOUD_PROJECT env var"
            )

        self.location = location

        # Set up credentials
        if credentials is None:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        self.credentials = credentials

        # Create client
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
        )

    async def sample_response(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        response_logprobs: bool = True,
        logprobs: int = 5,
    ) -> Dict[str, Any]:
        """Sample a single response from the model.

        Args:
            prompt: Input prompt text.
            model: Model identifier (e.g., "gemini-2.0-flash").
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            response_logprobs: Whether to return log probabilities.
            logprobs: Number of top logprobs to return per token.

        Returns:
            Dictionary containing the response and metadata.
        """
        config = GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_logprobs=response_logprobs,
            logprobs=logprobs,
        )

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            ),
        )

        # Convert response to dictionary format
        return self._convert_response_to_dict(response)

    def _convert_response_to_dict(self, response) -> Dict[str, Any]:
        """Convert Google API response to dictionary format.

        Args:
            response: Google API response object.

        Returns:
            Dictionary with response data and logprobs.
        """
        result = {
            "model": response.model_version,
            "text": response.text,
            "finish_reason": str(response.candidates[0].finish_reason),
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            },
        }

        # Extract logprobs if available
        if (
            response.candidates
            and response.candidates[0].logprobs_result
            and response.candidates[0].logprobs_result.chosen_candidates
        ):
            logprobs_result = response.candidates[0].logprobs_result
            result["logprobs"] = {
                "content": [],
            }

            # Process each token
            for i, chosen_candidate in enumerate(logprobs_result.chosen_candidates):
                token_data = {
                    "token": chosen_candidate.token,
                    "logprob": chosen_candidate.log_probability,
                    "token_id": chosen_candidate.token_id,
                }

                # Add top alternatives if available
                if i < len(logprobs_result.top_candidates):
                    top_alternatives = logprobs_result.top_candidates[i].candidates
                    token_data["top_logprobs"] = [
                        {
                            "token": alt.token,
                            "logprob": alt.log_probability,
                            "token_id": alt.token_id,
                        }
                        for alt in top_alternatives
                    ]

                result["logprobs"]["content"].append(token_data)

        return result

    async def sample_multiple_concurrent(
        self,
        prompt: str,
        model: str,
        num_samples: int,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        response_logprobs: bool = True,
        logprobs: int = 5,
        max_retries: int = 5,
    ) -> List[Dict[str, Any]]:
        """Sample multiple responses concurrently for the same prompt with retries.

        Args:
            prompt: Input prompt text.
            model: Model identifier.
            num_samples: Number of responses to sample.
            max_tokens: Maximum tokens to generate per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            response_logprobs: Whether to return log probabilities.
            logprobs: Number of top logprobs to return per token.
            max_retries: Maximum number of retry attempts for failed requests.

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
                    response_logprobs=response_logprobs,
                    logprobs=logprobs,
                )
                for _ in range(remaining)
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Process responses and count failures
            failed_count = 0
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    print(f"Warning: Sample failed with error: {resp}")
                    failed_count += 1
                else:
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

    async def sample_prompts_batch(
        self,
        prompts: List[str],
        model: str,
        num_samples_per_prompt: int,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        response_logprobs: bool = True,
        logprobs: int = 5,
        output_dir: Optional[str] = None,
        skip_existing: bool = True,
        max_retries: int = 5,
    ) -> List[Dict[str, Any]]:
        """Sample multiple responses for multiple prompts.

        Args:
            prompts: List of input prompts.
            model: Model identifier.
            num_samples_per_prompt: Number of responses per prompt.
            max_tokens: Maximum tokens to generate per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            response_logprobs: Whether to return log probabilities.
            logprobs: Number of top logprobs to return per token.
            output_dir: Optional directory to save results after each prompt.
            skip_existing: If True, skip prompts whose output files already exist.
            max_retries: Maximum number of retry attempts for failed requests.

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
                response_logprobs=response_logprobs,
                logprobs=logprobs,
                max_retries=max_retries,
            )

            sample = {
                "prompt": prompt,
                "model": model,
                "responses": responses,
            }
            results.append(sample)

            # Save immediately after sampling this prompt
            if output_dir:
                save_sample_to_json(sample, output_dir, model, prompt_idx)

        return results


def save_sample_to_json(
    sample: Dict[str, Any],
    output_dir: str,
    model_id: str,
    prompt_idx: int,
) -> None:
    """Save a single sampled response to JSON file.

    Args:
        sample: Sample dictionary from sample_prompts_batch.
        output_dir: Directory to save JSON files.
        model_id: Model identifier (e.g., "gemini-2.0-flash").
        prompt_idx: Index of the prompt.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create filename-safe model name
    model_name = model_id.replace("/", "_").replace(":", "_")

    # Create filename: {model_name}_prompt_{idx}.json
    filename = f"{model_name}_prompt_{prompt_idx}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(sample, f, indent=2)
