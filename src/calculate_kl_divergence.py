# ABOUTME: Calculate KL divergence per token between two models' logprobs.
# ABOUTME: Uses API prefilling to get logprobs from second model for generated text.

import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.auth
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from tqdm import tqdm


class KLDivergenceCalculator:
    """Calculate KL divergence between two models' logprobs."""

    def __init__(
        self,
        model_id: str,
        project_id: Optional[str] = None,
        location: str = "global",
        credentials: Optional[Any] = None,
    ):
        """Initialize KL divergence calculator.

        Args:
            model_id: Model identifier for the comparison model.
            project_id: Google Cloud project ID.
            location: Google Cloud region.
            credentials: Optional credentials object.
        """
        load_dotenv()

        self.model_id = model_id
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError(
                "Project ID must be provided or set in GOOGLE_CLOUD_PROJECT env var"
            )

        self.location = location

        if credentials is None:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        self.credentials = credentials
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
        )

    async def _get_logprobs_for_position(
        self,
        prompt: str,
        target_tokens: List[Dict[str, Any]],
        position: int,
        config: GenerateContentConfig,
        top_logprobs: int,
    ) -> Dict[str, Any]:
        """Get logprobs for a single token position.

        Args:
            prompt: User prompt.
            target_tokens: List of token data from Model 1.
            position: Token position to get logprobs for.
            config: GenerateContentConfig.
            top_logprobs: Number of top logprobs.

        Returns:
            Token data dictionary with logprobs.
        """
        target_token_data = target_tokens[position]

        # Build prefill from Model 1's tokens up to position
        prefill_text = "".join([t["token"] for t in target_tokens[:position]])

        # Build contents with Model 1's tokens as prefill
        contents = [{"role": "user", "parts": [{"text": prompt}]}]

        if prefill_text:
            contents.append({"role": "model", "parts": [{"text": prefill_text}]})

        # Generate next token to get logprobs at this position
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            ),
        )

        # Extract logprobs for this token position
        if (
            response.candidates
            and response.candidates[0].logprobs_result
            and response.candidates[0].logprobs_result.chosen_candidates
        ):
            logprobs_result = response.candidates[0].logprobs_result

            # Store the token from Model 1
            token_data = {
                "token": target_token_data["token"],
                "logprob": None,
                "token_id": target_token_data.get("token_id"),
                "top_logprobs": [],
                "position": position,
            }

            # Add top alternatives and find Model 1's token logprob
            if len(logprobs_result.top_candidates) > 0:
                top_alternatives = logprobs_result.top_candidates[0].candidates
                token_data["top_logprobs"] = [
                    {
                        "token": alt.token,
                        "logprob": alt.log_probability,
                        "token_id": alt.token_id,
                    }
                    for alt in top_alternatives
                ]

                # Find the logprob for Model 1's token in Model 2's distribution
                for alt in top_alternatives:
                    if alt.token == target_token_data["token"]:
                        token_data["logprob"] = alt.log_probability
                        break

                # If Model 1's token not in top-k, use floor probability
                if token_data["logprob"] is None:
                    min_logprob = min(alt.log_probability for alt in top_alternatives)
                    token_data["logprob"] = min_logprob
                    print(
                        f"    Warning: M1 token '{target_token_data['token']}' not in M2's top {top_logprobs} at position {position}, using floor"
                    )

            return token_data
        else:
            raise Exception(f"Failed to get logprobs at position {position}")

    async def get_iterative_logprobs(
        self,
        prompt: str,
        target_tokens: List[Dict[str, Any]],
        top_logprobs: int = 20,
        max_concurrent: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get logprobs by prefilling with Model 1's tokens in parallel batches.

        API doesn't return logprobs for prefilled text, so we prefill
        iteratively with Model 1's tokens and collect Model 2's probability
        distribution at each position. Processes multiple positions in parallel
        for speed.

        Args:
            prompt: User prompt.
            target_tokens: List of token data from Model 1 (with "token" field).
            top_logprobs: Number of top logprobs to return per token.
            max_concurrent: Maximum number of concurrent API calls.

        Returns:
            List of token dictionaries with logprobs from Model 2.
        """
        config = GenerateContentConfig(
            max_output_tokens=1,
            response_logprobs=True,
            logprobs=top_logprobs,
        )

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def get_with_semaphore(position):
            async with semaphore:
                return await self._get_logprobs_for_position(
                    prompt, target_tokens, position, config, top_logprobs
                )

        # Create tasks for all positions
        tasks = [get_with_semaphore(i) for i in range(len(target_tokens))]

        # Process all positions in parallel (with concurrency limit)
        print(
            f"    Processing {len(target_tokens)} tokens with max {max_concurrent} concurrent requests..."
        )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors and sort by position
        model2_tokens = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"    Error at position {i}: {result}")
                # Create error entry with None logprob
                model2_tokens.append(
                    {
                        "token": target_tokens[i]["token"],
                        "logprob": None,
                        "token_id": target_tokens[i].get("token_id"),
                        "top_logprobs": [],
                        "position": i,
                        "error": str(result),
                    }
                )
            else:
                model2_tokens.append(result)

        # Sort by position to ensure correct order
        model2_tokens.sort(key=lambda x: x["position"])

        # Remove position field (was only for sorting)
        for token in model2_tokens:
            token.pop("position", None)

        return model2_tokens

    def calculate_kl_per_token(
        self,
        model1_logprobs: List[Dict[str, Any]],
        model2_logprobs: List[Dict[str, Any]],
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Calculate KL divergence per token between two models.

        For tokens not in top K, uses floor probability (Kth logprob value).

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
            # Verify tokens match
            if token1_data["token"] != token2_data["token"]:
                print(
                    f"Warning: Token mismatch at position {i}: '{token1_data['token']}' vs '{token2_data['token']}'"
                )

            # Get top logprobs for each model
            model1_top = token1_data.get("top_logprobs", [])
            model2_top = token2_data.get("top_logprobs", [])

            # Build probability distributions
            model1_dist = self._build_distribution(model1_top)
            model2_dist = self._build_distribution(model2_top)

            # Calculate KL divergence for this token position
            kl_div = self._calculate_kl_divergence(model1_dist, model2_dist)
            kl_divergences.append(kl_div)

            # Store token details
            token_details.append(
                {
                    "position": i,
                    "token": token1_data["token"],
                    "kl_divergence": kl_div,
                    "model1_chosen_logprob": token1_data.get("logprob"),
                    "model2_chosen_logprob": token2_data.get("logprob"),
                }
            )

        return kl_divergences, token_details

    def _build_distribution(
        self, top_logprobs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Build probability distribution from top logprobs.

        Args:
            top_logprobs: List of top logprob entries.

        Returns:
            Dictionary mapping token to probability.
        """
        if not top_logprobs:
            return {}

        # Get floor logprob (last/worst logprob in the list)
        floor_logprob = min(item["logprob"] for item in top_logprobs)

        # Convert logprobs to probabilities
        dist = {}
        for item in top_logprobs:
            token = item["token"]
            logprob = item["logprob"]
            prob = math.exp(logprob)
            dist[token] = prob

        # Normalize (since we only have top K, probabilities won't sum to 1)
        total_prob = sum(dist.values())
        if total_prob > 0:
            dist = {token: prob / total_prob for token, prob in dist.items()}

        # Store floor probability for later use
        floor_prob = math.exp(floor_logprob) / total_prob if total_prob > 0 else 1e-10
        dist["__floor__"] = floor_prob

        return dist

    def _calculate_kl_divergence(
        self, p_dist: Dict[str, float], q_dist: Dict[str, float]
    ) -> float:
        """Calculate KL divergence KL(P||Q).

        For tokens not in top K, uses floor probability.

        Args:
            p_dist: Reference distribution (model 1).
            q_dist: Comparison distribution (model 2).

        Returns:
            KL divergence value.
        """
        if not p_dist or not q_dist:
            return float("inf")

        # Get floor probabilities
        p_floor = p_dist.get("__floor__", 1e-10)
        q_floor = q_dist.get("__floor__", 1e-10)

        # Get all unique tokens (excluding floor marker)
        all_tokens = set(p_dist.keys()) | set(q_dist.keys())
        all_tokens.discard("__floor__")

        kl_div = 0.0
        for token in all_tokens:
            # Get probabilities, using floor if token not in top K
            p = p_dist.get(token, p_floor)
            q = q_dist.get(token, q_floor)

            # KL divergence formula: P(x) * log(P(x) / Q(x))
            if p > 0 and q > 0:
                kl_div += p * math.log(p / q)

        return kl_div

    async def process_response(
        self,
        prompt: str,
        response_data: Dict[str, Any],
        top_logprobs: int = 20,
        response_idx: Optional[int] = None,
        max_concurrent: int = 10,
    ) -> Dict[str, Any]:
        """Process a single response to calculate KL divergence.

        Args:
            prompt: Original user prompt.
            response_data: Response data from model 1 with logprobs.
            top_logprobs: Number of top logprobs to request.
            response_idx: Optional response index for progress messages.
            max_concurrent: Maximum number of concurrent API calls.

        Returns:
            Dictionary with KL divergence results.
        """
        # Extract text and logprobs from model 1
        text = response_data.get("text", "")
        model1_logprobs = response_data.get("logprobs", {}).get("content", [])

        if not text or not model1_logprobs:
            return {
                "error": "Missing text or logprobs in response data",
                "text": text,
            }

        # Show progress
        if response_idx is not None:
            print(
                f"  Processing response {response_idx + 1}: {len(model1_logprobs)} tokens"
            )

        # Get logprobs from model 2 using iterative prefilling with Model 1's tokens
        try:
            model2_logprobs = await self.get_iterative_logprobs(
                prompt=prompt,
                target_tokens=model1_logprobs,
                top_logprobs=top_logprobs,
                max_concurrent=max_concurrent,
            )
        except Exception as e:
            return {
                "error": f"Failed to get model 2 logprobs: {str(e)}",
                "text": text,
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
                "model1_tokens": len(model1_logprobs),
                "model2_tokens": len(model2_logprobs),
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
        }

        return result


async def process_json_file(
    json_path: Path,
    calculator: KLDivergenceCalculator,
    output_dir: Path,
    top_logprobs: int = 20,
    max_concurrent: int = 10,
) -> None:
    """Process a single JSON file to calculate KL divergences.

    Args:
        json_path: Path to input JSON file.
        calculator: KLDivergenceCalculator instance.
        output_dir: Output directory for results.
        top_logprobs: Number of top logprobs to request.
        max_concurrent: Maximum number of concurrent API calls per response.
    """
    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    prompt = data.get("prompt", "")
    model1_name = data.get("model", "unknown")
    responses = data.get("responses", [])

    if not prompt or not responses:
        print(f"Skipping {json_path.name}: missing prompt or responses")
        return

    # Process each response
    print(f"\nProcessing {len(responses)} responses from {json_path.name}...")
    results = []
    for i, response in enumerate(responses):
        result = await calculator.process_response(
            prompt=prompt,
            response_data=response,
            top_logprobs=top_logprobs,
            response_idx=i,
            max_concurrent=max_concurrent,
        )
        results.append(result)

    # Save results
    output_data = {
        "prompt": prompt,
        "model1": model1_name,
        "model2": calculator.model_id,
        "num_responses": len(results),
        "responses": results,
    }

    # Create output filename
    output_filename = json_path.stem + "_kl.json"
    output_path = output_dir / output_filename

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved results to {output_filename}")


async def process_directory(
    input_dir: str,
    output_dir: str,
    model2_id: str,
    project_id: Optional[str] = None,
    location: str = "global",
    top_logprobs: int = 20,
    pattern: str = "*.json",
    max_concurrent: int = 10,
    max_files: Optional[int] = None,
) -> None:
    """Process all JSON files in a directory to calculate KL divergences.

    Args:
        input_dir: Directory containing JSON files from model 1.
        output_dir: Directory to save KL divergence results.
        model2_id: Model identifier for comparison model.
        project_id: Google Cloud project ID.
        location: Google Cloud region.
        top_logprobs: Number of top logprobs to request.
        pattern: Glob pattern for JSON files.
        max_concurrent: Maximum number of concurrent API calls per response.
        max_files: Maximum number of files to process (None for all).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(input_path.glob(pattern))

    # Limit number of files if max_files is specified
    if max_files is not None and max_files > 0:
        json_files = json_files[:max_files]
        print(
            f"Processing {len(json_files)} files (limited by --max-files {max_files})"
        )
    else:
        print(f"Found {len(json_files)} JSON files in {input_dir}")

    # Create calculator
    calculator = KLDivergenceCalculator(
        model_id=model2_id,
        project_id=project_id,
        location=location,
    )

    print(f"Comparing against model: {model2_id}")
    print(f"Output directory: {output_dir}\n")

    # Process each file
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            await process_json_file(
                json_path=json_file,
                calculator=calculator,
                output_dir=output_path,
                top_logprobs=top_logprobs,
                max_concurrent=max_concurrent,
            )
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    print(f"\n✓ Completed! Results saved to {output_dir}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate KL divergence between two models' logprobs"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSON files from model 1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save KL divergence results",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Model identifier for comparison model (model 2)",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="Google Cloud project ID (default: from env)",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="global",
        help="Google Cloud region (default: global)",
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
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent API calls per response (default: 10)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )

    args = parser.parse_args()

    asyncio.run(
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model2_id=args.model2,
            project_id=args.project_id,
            location=args.location,
            top_logprobs=args.top_logprobs,
            pattern=args.pattern,
            max_concurrent=args.max_concurrent,
            max_files=args.max_files,
        )
    )


if __name__ == "__main__":
    main()
