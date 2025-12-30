# ABOUTME: Calculate KL divergence using locally hosted VLLM models.
# ABOUTME: Run twice - once to sample model1 logprobs, once for model2 and KL calculation.

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from openrouter_client import OpenRouterClient


class KLDivergenceVLLMCalculator:
    """Calculate KL divergence using locally hosted VLLM model."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ):
        """Initialize VLLM calculator.

        Args:
            model_name: Model name/identifier for the hosted model.
            base_url: Base URL for the VLLM server.
            api_key: API key (VLLM typically doesn't require one).
            timeout: Request timeout in seconds.
        """
        load_dotenv()

        self.model_name = model_name
        self.client = OpenRouterClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def _get_logprobs_for_position(
        self,
        prompt: str,
        prefill_text: str,
        position: int,
        top_logprobs: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Get logprobs for the next token given a prefill.

        Args:
            prompt: User prompt.
            prefill_text: Text to prefill the assistant response.
            position: Token position (0 for first token).
            top_logprobs: Number of top logprobs to request.

        Returns:
            Dictionary with token and logprobs data, or None if failed.
        """
        # For position 0, use generation prompt; for others, continue the message
        if position == 0:
            add_generation_prompt = True
            continue_final_message = False
        else:
            add_generation_prompt = False
            continue_final_message = True

        try:
            response = await self.client.sample_response(
                prompt=prompt,
                model=self.model_name,
                max_tokens=1,
                temperature=1.0,
                logprobs=True,
                top_logprobs=top_logprobs,
                assistant_prefill=prefill_text if prefill_text else None,
                extra_body={
                    "add_generation_prompt": add_generation_prompt,
                    "continue_final_message": continue_final_message,
                },
            )

            if "choices" not in response or not response["choices"]:
                return None

            choice = response["choices"][0]
            if "logprobs" not in choice or not choice["logprobs"]:
                return None

            logprobs_data = choice["logprobs"]
            if "content" not in logprobs_data or not logprobs_data["content"]:
                return None

            token_data = logprobs_data["content"][0]

            return {
                "token": token_data["token"],
                "logprob": token_data["logprob"],
                "top_logprobs": [
                    {"token": t["token"], "logprob": t["logprob"]}
                    for t in token_data.get("top_logprobs", [])
                ],
            }

        except Exception as e:
            print(f"    Warning: Failed to get logprobs: {e}")
            return None

    async def get_iterative_logprobs(
        self,
        prompt: str,
        target_tokens: List[Dict[str, Any]],
        top_logprobs: int = 20,
        max_concurrent: int = 10,
        max_retries: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get logprobs by prefilling with target tokens iteratively.

        Args:
            prompt: User prompt.
            target_tokens: List of target token data (must have 'token' field).
            top_logprobs: Number of top logprobs to request.
            max_concurrent: Maximum number of concurrent API calls.
            max_retries: Maximum number of retries for failed requests.

        Returns:
            List of token dictionaries with logprobs.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def get_with_semaphore(position: int) -> Dict[str, Any]:
            async with semaphore:
                # Build prefill from tokens up to this position
                prefill_text = "".join([t["token"] for t in target_tokens[:position]])
                if position == 0 or position == 1:
                    print(f"Prefill text: {prefill_text}")

                for retry in range(max_retries):
                    try:
                        result = await self._get_logprobs_for_position(
                            prompt=prompt,
                            prefill_text=prefill_text,
                            position=position,
                            top_logprobs=top_logprobs,
                        )

                        if result and result.get("top_logprobs"):
                            result["position"] = position
                            # Always use the original token from input, not the API response
                            result["token"] = target_tokens[position]["token"]
                            result["prefill_text"] = prefill_text
                            return result

                        if retry < max_retries - 1:
                            print(
                                f"    Position {position}: Invalid logprobs, retrying ({retry + 1}/{max_retries})..."
                            )
                            await asyncio.sleep(0.5)
                        else:
                            print(
                                f"    Position {position}: Failed after {max_retries} retries"
                            )
                            return {
                                "token": target_tokens[position]["token"],
                                "logprob": None,
                                "top_logprobs": [],
                                "position": position,
                                "prefill_text": prefill_text,
                                "error": "Failed to get valid logprobs after retries",
                            }
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(
                                f"    Position {position}: Error, retrying ({retry + 1}/{max_retries}): {str(e)}"
                            )
                            await asyncio.sleep(0.5)
                        else:
                            print(
                                f"    Position {position}: Failed after {max_retries} retries: {str(e)}"
                            )
                            return {
                                "token": target_tokens[position]["token"],
                                "logprob": None,
                                "top_logprobs": [],
                                "position": position,
                                "prefill_text": prefill_text,
                                "error": str(e),
                            }

                return {
                    "token": target_tokens[position]["token"],
                    "logprob": None,
                    "top_logprobs": [],
                    "position": position,
                    "prefill_text": prefill_text,
                    "error": "Unknown error",
                }

        # Create tasks for all positions
        tasks = [get_with_semaphore(i) for i in range(len(target_tokens))]

        print(
            f"    Processing {len(target_tokens)} tokens with max {max_concurrent} concurrent requests..."
        )

        # Process all positions in parallel with progress bar
        results = []
        with tqdm(total=len(target_tokens), desc="    Tokens", leave=False) as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if "error" in result:
                    print(
                        f"    Error at position {result['position']}: {result['error']}"
                    )
                results.append(result)
                pbar.update(1)

        # Sort by position
        results.sort(key=lambda x: x["position"])

        # Check for errors
        errors = [t for t in results if "error" in t]
        if errors:
            print(
                f"    Warning: {len(errors)} tokens failed to get valid logprobs after retries"
            )

        # Remove position field
        for token in results:
            token.pop("position", None)

        return results

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
            # Verify tokens match
            if token1_data["token"] != token2_data["token"]:
                print(
                    f"Warning: Token mismatch at position {i}: '{token1_data['token']}' vs '{token2_data['token']}'"
                )

            model1_top = token1_data.get("top_logprobs", [])
            model2_top = token2_data.get("top_logprobs", [])

            model1_dist = self._build_distribution(model1_top)
            model2_dist = self._build_distribution(model2_top)

            kl_div = self._calculate_kl_divergence(model1_dist, model2_dist)
            kl_divergences.append(kl_div)

            entropy1 = self._calculate_entropy_from_top_logprobs(model1_top)
            entropy2 = self._calculate_entropy_from_top_logprobs(model2_top)

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
        """Calculate entropy from a top-k (truncated) logprob list."""
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

    async def process_response_model1(
        self,
        prompt: str,
        response_data: Dict[str, Any],
        top_logprobs: int = 20,
        response_idx: Optional[int] = None,
        max_concurrent: int = 10,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        """Process a response to sample logprobs from model1.

        Args:
            prompt: Original user prompt.
            response_data: Response data containing text and token sequence.
            top_logprobs: Number of top logprobs to request.
            response_idx: Optional response index for progress messages.
            max_concurrent: Maximum number of concurrent API calls.
            max_retries: Maximum number of retries for failed API calls.

        Returns:
            Dictionary with model1 logprobs.
        """
        text = response_data.get("text", "")
        original_logprobs = response_data.get("logprobs", {}).get("content", [])

        if not text:
            return {
                "response_idx": response_idx,
                "text": text,
                "error": "Missing text in response data",
            }

        if not original_logprobs:
            return {
                "response_idx": response_idx,
                "text": text,
                "error": "Missing logprobs in response data - need token sequence",
            }

        num_tokens = len(original_logprobs)

        if response_idx is not None:
            print(f"  Processing response {response_idx + 1}: {num_tokens} tokens")
            print("    Sampling logprobs from model 1...")

        try:
            model1_logprobs = await self.get_iterative_logprobs(
                prompt=prompt,
                target_tokens=original_logprobs,
                top_logprobs=top_logprobs,
                max_concurrent=max_concurrent,
                max_retries=max_retries,
            )
        except Exception as e:
            return {
                "response_idx": response_idx,
                "text": text,
                "error": f"Failed to get model 1 logprobs: {str(e)}",
            }

        return {
            "response_idx": response_idx,
            "text": text,
            "num_tokens": len(model1_logprobs),
            "model1_logprobs": model1_logprobs,
        }

    async def process_response_model2(
        self,
        prompt: str,
        response_data: Dict[str, Any],
        top_logprobs: int = 20,
        response_idx: Optional[int] = None,
        max_concurrent: int = 10,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        """Process a response to sample logprobs from model2 and calculate KL.

        Args:
            prompt: Original user prompt.
            response_data: Response data with model1_logprobs already present.
            top_logprobs: Number of top logprobs to request.
            response_idx: Optional response index for progress messages.
            max_concurrent: Maximum number of concurrent API calls.
            max_retries: Maximum number of retries for failed API calls.

        Returns:
            Dictionary with KL divergence results.
        """
        text = response_data.get("text", "")
        model1_logprobs = response_data.get("model1_logprobs", [])

        if not text:
            return {
                "response_idx": response_idx,
                "text": text,
                "error": "Missing text in response data",
            }

        if not model1_logprobs:
            return {
                "response_idx": response_idx,
                "text": text,
                "error": "Missing model1_logprobs in response data",
            }

        num_tokens = len(model1_logprobs)

        if response_idx is not None:
            print(f"  Processing response {response_idx + 1}: {num_tokens} tokens")
            print("    Sampling logprobs from model 2...")

        try:
            model2_logprobs = await self.get_iterative_logprobs(
                prompt=prompt,
                target_tokens=model1_logprobs,
                top_logprobs=top_logprobs,
                max_concurrent=max_concurrent,
                max_retries=max_retries,
            )
        except Exception as e:
            return {
                "response_idx": response_idx,
                "text": text,
                "error": f"Failed to get model 2 logprobs: {str(e)}",
            }

        # Calculate KL divergence
        try:
            kl_divergences, token_details = self.calculate_kl_per_token(
                model1_logprobs, model2_logprobs
            )
        except Exception as e:
            return {
                "response_idx": response_idx,
                "text": text,
                "error": f"Failed to calculate KL divergence: {str(e)}",
                "model1_tokens": len(model1_logprobs),
                "model2_tokens": len(model2_logprobs),
            }

        # Check for infinity values
        inf_positions = [i for i, kl in enumerate(kl_divergences) if math.isinf(kl)]
        if inf_positions:
            failed_tokens = [
                (i, model1_logprobs[i]["token"]) for i in inf_positions[:10]
            ]
            error_msg = f"KL divergence contains {len(inf_positions)} infinity values. Failed tokens: {failed_tokens}"
            if len(inf_positions) > 10:
                error_msg += f"... and {len(inf_positions) - 10} more"
            print(f"    Error: {error_msg}")
            return {
                "response_idx": response_idx,
                "text": text,
                "error": error_msg,
                "model1_tokens": len(model1_logprobs),
                "model2_tokens": len(model2_logprobs),
                "failed_positions": inf_positions,
            }

        # Calculate statistics
        kl_array = np.array(kl_divergences)
        result = {
            "response_idx": response_idx,
            "text": text,
            "num_tokens": len(kl_divergences),
            "kl_per_token": kl_divergences,
            "token_details": token_details,
            "model1_logprobs": model1_logprobs,
            "model2_logprobs": model2_logprobs,
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


async def process_json_file_model1(
    json_path: Path,
    calculator: KLDivergenceVLLMCalculator,
    output_dir: Path,
    top_logprobs: int = 20,
    max_concurrent: int = 10,
    max_retries: int = 5,
) -> None:
    """Process a JSON file to sample model1 logprobs.

    Args:
        json_path: Path to input JSON file.
        calculator: KLDivergenceVLLMCalculator instance.
        output_dir: Output directory for intermediate results.
        top_logprobs: Number of top logprobs to request.
        max_concurrent: Maximum number of concurrent API calls.
        max_retries: Maximum number of retries for failed API calls.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    prompt = data.get("prompt", "")
    model1_name = data.get("model", "unknown")
    responses = data.get("responses", [])

    if not prompt or not responses:
        print(f"Skipping {json_path.name}: missing prompt or responses")
        return

    print(f"\nProcessing {len(responses)} responses from {json_path.name}...")

    results = []
    for i, response in enumerate(responses):
        result = await calculator.process_response_model1(
            prompt=prompt,
            response_data=response,
            top_logprobs=top_logprobs,
            response_idx=i,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
        )
        results.append(result)

    # Sort by response_idx
    results.sort(key=lambda x: x.get("response_idx", -1))

    # Save intermediate results
    output_data = {
        "prompt": prompt,
        "model1": calculator.model_name,
        "model1_resampled": True,
        "original_model": model1_name,
        "num_responses": len(results),
        "responses": results,
    }

    output_filename = json_path.stem + "_model1.json"
    output_path = output_dir / output_filename

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved model1 logprobs to {output_filename}")


async def process_json_file_model2(
    json_path: Path,
    calculator: KLDivergenceVLLMCalculator,
    output_dir: Path,
    top_logprobs: int = 20,
    max_concurrent: int = 10,
    max_retries: int = 5,
) -> None:
    """Process a JSON file with model1 logprobs to calculate KL divergence.

    Args:
        json_path: Path to intermediate JSON file with model1 logprobs.
        calculator: KLDivergenceVLLMCalculator instance.
        output_dir: Output directory for final results.
        top_logprobs: Number of top logprobs to request.
        max_concurrent: Maximum number of concurrent API calls.
        max_retries: Maximum number of retries for failed API calls.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    prompt = data.get("prompt", "")
    model1_name = data.get("model1", "unknown")
    responses = data.get("responses", [])

    if not prompt or not responses:
        print(f"Skipping {json_path.name}: missing prompt or responses")
        return

    print(f"\nProcessing {len(responses)} responses from {json_path.name}...")

    results = []
    for i, response in enumerate(responses):
        result = await calculator.process_response_model2(
            prompt=prompt,
            response_data=response,
            top_logprobs=top_logprobs,
            response_idx=i,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
        )
        results.append(result)

    # Sort by response_idx
    results.sort(key=lambda x: x.get("response_idx", -1))

    # Save final results (same format as calculate_kl_divergence.py)
    output_data = {
        "prompt": prompt,
        "model1": model1_name,
        "model1_resampled": True,
        "model2": calculator.model_name,
        "num_responses": len(results),
        "responses": results,
    }

    # Remove _model1 suffix and add _kl suffix
    base_name = json_path.stem
    if base_name.endswith("_model1"):
        base_name = base_name[:-7]
    output_filename = base_name + "_kl.json"
    output_path = output_dir / output_filename

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved KL divergence results to {output_filename}")


async def process_directory(
    input_dir: str,
    output_dir: str,
    model_name: str,
    mode: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    timeout: float = 120.0,
    top_logprobs: int = 20,
    pattern: str = "*.json",
    max_concurrent: int = 10,
    max_files: Optional[int] = None,
    max_retries: int = 5,
) -> None:
    """Process all JSON files in a directory.

    Args:
        input_dir: Directory containing JSON files.
        output_dir: Directory to save results.
        model_name: Model name for the currently hosted model.
        mode: "model1" to sample model1 logprobs, "model2" to calculate KL.
        base_url: Base URL for the VLLM server.
        api_key: API key for the VLLM server.
        timeout: Request timeout in seconds.
        top_logprobs: Number of top logprobs to request.
        pattern: Glob pattern for JSON files.
        max_concurrent: Maximum number of concurrent API calls.
        max_files: Maximum number of files to process.
        max_retries: Maximum number of retries for failed API calls.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob(pattern))

    if max_files is not None and max_files > 0:
        json_files = json_files[:max_files]
        print(
            f"Processing {len(json_files)} files (limited by --max-files {max_files})"
        )
    else:
        print(f"Found {len(json_files)} JSON files in {input_dir}")

    calculator = KLDivergenceVLLMCalculator(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )

    print(f"Mode: {mode}")
    print(f"Model: {model_name}")
    print(f"VLLM server: {base_url}")
    print(f"Output directory: {output_dir}\n")

    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            if mode == "model1":
                await process_json_file_model1(
                    json_path=json_file,
                    calculator=calculator,
                    output_dir=output_path,
                    top_logprobs=top_logprobs,
                    max_concurrent=max_concurrent,
                    max_retries=max_retries,
                )
            else:  # mode == "model2"
                await process_json_file_model2(
                    json_path=json_file,
                    calculator=calculator,
                    output_dir=output_path,
                    top_logprobs=top_logprobs,
                    max_concurrent=max_concurrent,
                    max_retries=max_retries,
                )
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    print(f"\n✓ Completed! Results saved to {output_dir}")


def main(
    input_dir: str,
    output_dir: str,
    model_name: str,
    mode: str = "model1",
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    timeout: float = 120.0,
    top_logprobs: int = 5,
    pattern: str = "*.json",
    max_concurrent: int = 10,
    max_files: Optional[int] = None,
    max_retries: int = 5,
):
    """Calculate KL divergence using locally hosted VLLM model.

    Run this script twice:
    1. First with mode="model1" to sample logprobs from model1
    2. Then with mode="model2" to sample from model2 and calculate KL divergence

    Args:
        input_dir: Directory containing JSON files.
            - For mode="model1": original response files with token sequences
            - For mode="model2": intermediate files with model1 logprobs (*_model1.json)
        output_dir: Directory to save results.
        model_name: Model name for the currently hosted VLLM model.
        mode: "model1" or "model2" (default: "model1").
        base_url: Base URL for the VLLM server (default: http://localhost:8000/v1).
        api_key: API key for the VLLM server (default: EMPTY).
        timeout: Request timeout in seconds (default: 120.0).
        top_logprobs: Number of top logprobs to request (default: 20).
        pattern: Glob pattern for JSON files (default: *.json).
        max_concurrent: Maximum concurrent API calls (default: 10).
        max_files: Maximum number of files to process (default: all).
        max_retries: Maximum retries for failed API calls (default: 5).
    """
    if mode not in ("model1", "model2"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'model1' or 'model2'")

    asyncio.run(
        process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            model_name=model_name,
            mode=mode,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            top_logprobs=top_logprobs,
            pattern=pattern,
            max_concurrent=max_concurrent,
            max_files=max_files,
            max_retries=max_retries,
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
