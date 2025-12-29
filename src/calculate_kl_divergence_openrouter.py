# ABOUTME: Calculate KL divergence using OpenRouter API.
# ABOUTME: Uses assistant prefill to get all token logprobs in a single request.

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from openrouter_client import OpenRouterClient


class KLDivergenceOpenRouterCalculator:
    """Calculate KL divergence using OpenRouter API."""

    def __init__(
        self,
        model1_id: str,
        model2_id: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        provider: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OpenRouter calculator.

        Args:
            model1_id: Model identifier for reference model (model 1).
            model2_id: Model identifier for comparison model (model 2).
            api_key: OpenRouter API key.
            timeout: Request timeout in seconds.
            provider: Optional provider configuration (e.g., {"only": ["provider_name"]}).
        """
        load_dotenv()

        self.model1_id = model1_id
        self.model2_id = model2_id
        self.provider = provider
        self.client = OpenRouterClient(api_key=api_key, timeout=timeout)

    async def _get_all_token_logprobs(
        self,
        prompt: str,
        prefill_text: str,
        model_id: str,
        top_logprobs: int = 20,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get logprobs for all tokens in the prefilled text.

        Args:
            prompt: User prompt.
            prefill_text: Text to prefill the assistant response.
            model_id: Model identifier to use for sampling.
            top_logprobs: Number of top logprobs to request.

        Returns:
            List of token dictionaries with logprobs, or None if request failed.
        """
        try:
            # Estimate max_tokens needed (conservative: ~2 chars per token)
            estimated_tokens = max(len(prefill_text) // 2, 1)
            # Cap at reasonable maximum to avoid excessive costs
            max_tokens = min(estimated_tokens, 4096)

            response = await self.client.sample_response(
                prompt=prompt,
                model=model_id,
                max_tokens=max_tokens,
                temperature=0.0,
                logprobs=True,
                top_logprobs=top_logprobs,
                assistant_prefill=prefill_text,
                provider=self.provider,
            )

            # Extract logprobs from response
            if "choices" not in response or not response["choices"]:
                return None

            choice = response["choices"][0]
            if "logprobs" not in choice or not choice["logprobs"]:
                return None

            logprobs_data = choice["logprobs"]
            if "content" not in logprobs_data or not logprobs_data["content"]:
                return None

            # Get all tokens' logprobs from the prefill
            all_tokens = []
            for token_data in logprobs_data["content"]:
                all_tokens.append(
                    {
                        "token": token_data["token"],
                        "logprob": token_data["logprob"],
                        "top_logprobs": [
                            {"token": t["token"], "logprob": t["logprob"]}
                            for t in token_data.get("top_logprobs", [])
                        ],
                    }
                )

            return all_tokens

        except Exception as e:
            print(f"    Warning: Failed to get logprobs: {e}")
            return None

    async def get_logprobs_from_both_models(
        self,
        prompt: str,
        response_text: str,
        top_logprobs: int = 20,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        """Get logprobs from both models for the same response text.

        Args:
            prompt: User prompt.
            response_text: Response text to prefill.
            top_logprobs: Number of top logprobs to request.

        Returns:
            Tuple of (model1_logprobs, model2_logprobs).
        """
        print(f"    Sampling logprobs from both models for response...")

        # Get logprobs from both models concurrently
        model1_task = self._get_all_token_logprobs(
            prompt=prompt,
            prefill_text=response_text,
            model_id=self.model1_id,
            top_logprobs=top_logprobs,
        )

        model2_task = self._get_all_token_logprobs(
            prompt=prompt,
            prefill_text=response_text,
            model_id=self.model2_id,
            top_logprobs=top_logprobs,
        )

        model1_logprobs, model2_logprobs = await asyncio.gather(
            model1_task, model2_task
        )

        if model1_logprobs and model2_logprobs:
            print(
                f"    Got {len(model1_logprobs)} tokens from model1, {len(model2_logprobs)} tokens from model2"
            )

        return model1_logprobs, model2_logprobs

    def calculate_kl_per_token(
        self,
        model1_logprobs: List[Dict[str, Any]],
        model2_logprobs: List[Dict[str, Any]],
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Calculate KL divergence per token between two models.

        Args:
            model1_logprobs: List of token logprobs from model 1 (reference).
            model2_logprobs: List of token logprobs from model 2 (comparison).

        Returns:
            Tuple of (list of KL divergences per token, list of token details).
        """
        if len(model1_logprobs) != len(model2_logprobs):
            raise ValueError(
                f"Token count mismatch: model1={len(model1_logprobs)}, model2={len(model2_logprobs)}"
            )

        kl_divergences = []
        token_details = []

        for i, (token1_data, token2_data) in enumerate(
            zip(model1_logprobs, model2_logprobs)
        ):
            # Get top logprobs for each model
            model1_top = token1_data.get("top_logprobs", [])
            model2_top = token2_data.get("top_logprobs", [])

            # Build probability distributions
            model1_dist = self._build_distribution(model1_top)
            model2_dist = self._build_distribution(model2_top)

            # Calculate KL divergence for this token position
            kl_div = self._calculate_kl_divergence(model1_dist, model2_dist)
            kl_divergences.append(kl_div)

            entropy1 = self._calculate_entropy_from_top_logprobs(model1_top)
            entropy2 = self._calculate_entropy_from_top_logprobs(model2_top)

            # Store token details
            token_details.append(
                {
                    "position": i,
                    "token": token1_data["token"],
                    "kl_divergence": kl_div,
                    "entropy1": entropy1,
                    "entropy2": entropy2,
                    "model1_chosen_logprob": token1_data.get("logprob"),
                    "model2_chosen_logprob": token2_data.get("logprob"),
                }
            )

        return kl_divergences, token_details

    def _calculate_entropy_from_top_logprobs(
        self, top_logprobs: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate entropy from a top-k (truncated) logprob list.

        This computes entropy over the normalized top-k distribution:
          H = -Σ p_i log(p_i)
        where p_i ∝ exp(logprob_i).
        """
        if not top_logprobs:
            return None

        probs = []
        total = 0.0
        for item in top_logprobs:
            logprob = item.get("logprob")
            if logprob is None:
                continue
            p = math.exp(logprob)
            probs.append(p)
            total += p

        if total <= 0.0 or not probs:
            return None

        entropy = 0.0
        for p in probs:
            pn = p / total
            if pn > 0.0:
                entropy -= pn * math.log(pn)

        return entropy

    def _build_distribution(
        self, top_logprobs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Build probability distribution from top logprobs."""
        if not top_logprobs:
            return {}

        floor_logprob = min(item["logprob"] for item in top_logprobs)

        dist = {}
        for item in top_logprobs:
            token = item["token"]
            logprob = item["logprob"]
            prob = math.exp(logprob)
            dist[token] = prob

        total_prob = sum(dist.values())
        if total_prob > 0:
            dist = {token: prob / total_prob for token, prob in dist.items()}

        floor_prob = math.exp(floor_logprob) / total_prob if total_prob > 0 else 1e-10
        dist["__floor__"] = floor_prob

        return dist

    def _calculate_kl_divergence(
        self, p_dist: Dict[str, float], q_dist: Dict[str, float]
    ) -> float:
        """Calculate KL divergence KL(P||Q)."""
        if not p_dist or not q_dist:
            return float("inf")

        p_floor = p_dist.get("__floor__", 1e-10)
        q_floor = q_dist.get("__floor__", 1e-10)

        all_tokens = set(p_dist.keys()) | set(q_dist.keys())
        all_tokens.discard("__floor__")

        kl_div = 0.0
        for token in all_tokens:
            p = p_dist.get(token, p_floor)
            q = q_dist.get(token, q_floor)

            if p > 0 and q > 0:
                kl_div += p * math.log(p / q)

        return kl_div

    async def process_response(
        self,
        prompt: str,
        response_data: Dict[str, Any],
        top_logprobs: int = 20,
        response_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process a single response to calculate KL divergence.

        Args:
            prompt: Original user prompt.
            response_data: Response data containing text field (logprobs are disregarded).
            top_logprobs: Number of top logprobs to request.
            response_idx: Optional response index for progress messages.

        Returns:
            Dictionary with KL divergence results.
        """
        text = response_data.get("text", "")

        if not text:
            return {
                "error": "Missing text in response data",
                "text": text,
            }

        if response_idx is not None:
            print(f"  Processing response {response_idx + 1}")

        # Get logprobs from both models
        try:
            model1_logprobs, model2_logprobs = await self.get_logprobs_from_both_models(
                prompt=prompt,
                response_text=text,
                top_logprobs=top_logprobs,
            )
        except Exception as e:
            return {
                "error": f"Failed to get logprobs: {str(e)}",
                "text": text,
            }

        if model1_logprobs is None or model2_logprobs is None:
            return {
                "error": "Failed to get logprobs from one or both models",
                "text": text,
            }

        # Check if tokenizations match
        if len(model1_logprobs) != len(model2_logprobs):
            return {
                "error": f"Tokenization mismatch: model1={len(model1_logprobs)} tokens, model2={len(model2_logprobs)} tokens",
                "text": text,
                "model1_tokens": len(model1_logprobs),
                "model2_tokens": len(model2_logprobs),
            }

        # Calculate KL divergence per token
        try:
            kl_divergences, token_details = self.calculate_kl_per_token(
                model1_logprobs, model2_logprobs
            )
        except Exception as e:
            return {
                "error": f"Failed to calculate KL divergence: {str(e)}",
                "text": text,
            }

        # Calculate statistics
        kl_array = np.array(kl_divergences)
        result = {
            "text": text,
            "num_tokens": len(kl_divergences),
            "kl_per_token": kl_divergences,
            "token_details": token_details,
            "statistics": {
                "mean_kl": float(np.mean(kl_array)),
                "median_kl": float(np.median(kl_array)),
                "std_kl": float(np.std(kl_array)),
                "min_kl": float(np.min(kl_array)),
                "max_kl": float(np.max(kl_array)),
                "total_kl": float(np.sum(kl_array)),
            },
            "raw_logprobs": {
                "model1": model1_logprobs,
                "model2": model2_logprobs,
            },
        }

        return result


async def process_json_file(
    json_path: Path,
    calculator: KLDivergenceOpenRouterCalculator,
    output_dir: Path,
    top_logprobs: int = 20,
) -> None:
    """Process a single JSON file to calculate KL divergences."""
    with open(json_path, "r") as f:
        data = json.load(f)

    prompt = data.get("prompt", "")
    model1_name = data.get("model", "unknown")
    responses = data.get("responses", [])

    if not prompt or not responses:
        print(f"Skipping {json_path.name}: missing prompt or responses")
        return

    print(f"\nProcessing {len(responses)} responses from {json_path.name}...")

    # Prepare output path
    output_filename = json_path.stem + "_kl.json"
    output_path = output_dir / output_filename

    # Prepare base output data structure
    output_data = {
        "prompt": prompt,
        "model1": calculator.model1_id,
        "model2": calculator.model2_id,
        "original_model": model1_name,
        "num_responses": len(responses),
        "responses": [],
    }

    # Process all responses concurrently
    tasks = [
        calculator.process_response(
            prompt=prompt,
            response_data=response,
            top_logprobs=top_logprobs,
            response_idx=i,
        )
        for i, response in enumerate(responses)
    ]
    results = await asyncio.gather(*tasks)

    # Save results
    output_data["responses"] = results
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved results to {output_filename}")


async def process_directory(
    input_dir: str,
    output_dir: str,
    model1_id: str,
    model2_id: str,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
    top_logprobs: int = 20,
    pattern: str = "*.json",
    max_files: Optional[int] = None,
    provider: Optional[Dict[str, Any]] = None,
) -> None:
    """Process all JSON files in a directory.

    Args:
        input_dir: Directory containing JSON files with responses.
        output_dir: Directory to save KL divergence results.
        model1_id: Model identifier for reference model (model 1).
        model2_id: Model identifier for comparison model (model 2).
        api_key: OpenRouter API key.
        timeout: Request timeout in seconds.
        top_logprobs: Number of top logprobs to request.
        pattern: Glob pattern for JSON files.
        max_files: Maximum number of files to process (None for all).
        provider: Optional provider configuration (e.g., {"only": ["provider_name"]}).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob(pattern))

    # Limit number of files if max_files is specified
    if max_files is not None and max_files > 0:
        json_files = json_files[:max_files]
        print(
            f"Processing {len(json_files)} files (limited by --max-files {max_files})"
        )
    else:
        print(f"Found {len(json_files)} JSON files in {input_dir}")

    calculator = KLDivergenceOpenRouterCalculator(
        model1_id=model1_id,
        model2_id=model2_id,
        api_key=api_key,
        timeout=timeout,
        provider=provider,
    )

    print(f"Comparing models: {model1_id} vs {model2_id}")
    print(f"Output directory: {output_dir}")
    print()

    # Process all files
    for json_file in json_files:
        await process_json_file(json_file, calculator, output_path, top_logprobs)

    print("\n✓ All processing complete!")
    print(f"Results saved to: {output_dir}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate KL divergence using OpenRouter API"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSON files with responses",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save KL divergence results",
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Model identifier for reference model (model 1)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Model identifier for comparison model (model 2)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (default: from env)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="Number of top logprobs to request (default: 20)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files (default: *.json)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help='Provider configuration as JSON (e.g., \'{"only": ["OpenAI"]}\')',
    )

    args = parser.parse_args()

    # Parse provider JSON if provided
    provider = None
    if args.provider:
        try:
            provider = json.loads(args.provider)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for --provider: {e}")
            return

    asyncio.run(
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model1_id=args.model1,
            model2_id=args.model2,
            api_key=args.api_key,
            timeout=args.timeout,
            top_logprobs=args.top_logprobs,
            pattern=args.pattern,
            max_files=args.max_files,
            provider=provider,
        )
    )


if __name__ == "__main__":
    main()
