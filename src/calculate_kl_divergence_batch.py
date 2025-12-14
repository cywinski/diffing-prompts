# ABOUTME: Calculate KL divergence using Google Cloud Batch Prediction API.
# ABOUTME: Much faster than individual API calls - processes all token positions in one batch job.

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.auth
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig
from tqdm import tqdm


class KLDivergenceBatchCalculator:
    """Calculate KL divergence using batch prediction API."""

    def __init__(
        self,
        model_id: str,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        bucket_name: Optional[str] = None,
        credentials: Optional[Any] = None,
    ):
        """Initialize batch calculator.

        Args:
            model_id: Model identifier for comparison model.
            project_id: Google Cloud project ID.
            location: Google Cloud region (must support batch prediction).
            bucket_name: GCS bucket for batch input/output.
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
        self.bucket_name = bucket_name or f"{self.project_id}-kl-batch"

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
        self.storage_client = storage.Client(
            project=self.project_id, credentials=self.credentials
        )

        # Ensure bucket exists
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Ensure GCS bucket exists."""
        try:
            self.storage_client.get_bucket(self.bucket_name)
            print(f"Using existing bucket: {self.bucket_name}")
        except Exception:
            print(f"Creating bucket: {self.bucket_name}")
            bucket = self.storage_client.create_bucket(
                self.bucket_name, location=self.location
            )
            print(f"✓ Created bucket: {bucket.name}")

    def _create_batch_requests(
        self,
        prompt: str,
        target_tokens: List[Dict[str, Any]],
        top_logprobs: int = 20,
    ) -> List[Dict[str, Any]]:
        """Create batch prediction requests for all token positions.

        Args:
            prompt: User prompt.
            target_tokens: List of token data from Model 1.
            top_logprobs: Number of top logprobs to request.

        Returns:
            List of request dictionaries for batch API.
        """
        requests = []

        for i, target_token_data in enumerate(target_tokens):
            # Build prefill from Model 1's tokens up to position i
            prefill_text = "".join([t["token"] for t in target_tokens[:i]])

            # Build contents
            contents = [{"role": "user", "parts": [{"text": prompt}]}]
            if prefill_text:
                contents.append({"role": "model", "parts": [{"text": prefill_text}]})

            # Create request
            request = {
                "request": {
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": 1,
                        "responseLogprobs": True,
                        "logprobs": top_logprobs,
                    },
                },
                "position": i,  # Track position for sorting later
            }
            requests.append(request)

        return requests

    def _upload_batch_input(
        self, requests: List[Dict[str, Any]], blob_name: str
    ) -> str:
        """Upload batch requests to GCS.

        Args:
            requests: List of request dictionaries.
            blob_name: Name for the blob in GCS.

        Returns:
            GCS URI of uploaded file.
        """
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)

        # Write as JSONL (one request per line)
        jsonl_content = "\n".join([json.dumps(req) for req in requests])
        blob.upload_from_string(jsonl_content, content_type="application/jsonl")

        return f"gs://{self.bucket_name}/{blob_name}"

    async def _submit_batch_job(
        self,
        input_uri: str,
        output_uri: str,
    ) -> str:
        """Submit batch prediction job.

        Args:
            input_uri: GCS URI of input JSONL file.
            output_uri: GCS URI prefix for output.

        Returns:
            Job name.
        """
        loop = asyncio.get_event_loop()

        def submit_job():
            return self.client.batches.create(
                model=self.model_id,
                src=input_uri,
                config=CreateBatchJobConfig(dest=output_uri),
            )

        batch_job = await loop.run_in_executor(None, submit_job)
        return batch_job.name

    async def _wait_for_batch_job(
        self, job_name: str, poll_interval: int = 10, verbose: bool = False
    ) -> None:
        """Wait for batch job to complete.

        Args:
            job_name: Batch job name.
            poll_interval: Seconds between status checks.
            verbose: If True, print individual job states.
        """
        loop = asyncio.get_event_loop()

        while True:

            def get_job():
                return self.client.batches.get(name=job_name)

            batch_job = await loop.run_in_executor(None, get_job)

            state = batch_job.state
            if verbose:
                print(f"    Job state: {state}")

            if state == "JOB_STATE_SUCCEEDED":
                if verbose:
                    print("    ✓ Batch job completed successfully")
                break
            elif state == "JOB_STATE_FAILED":
                raise Exception(f"Batch job failed: {batch_job}")
            elif state in ["JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
                raise Exception(f"Batch job stopped: {state}")

            await asyncio.sleep(poll_interval)

    async def _wait_for_multiple_jobs(
        self, job_names: List[str], poll_interval: int = 10
    ) -> None:
        """Wait for multiple batch jobs to complete, showing progress.

        Args:
            job_names: List of batch job names.
            poll_interval: Seconds between status checks.
        """
        if not job_names:
            return

        loop = asyncio.get_event_loop()
        total_jobs = len(job_names)
        completed_jobs = set()

        while len(completed_jobs) < total_jobs:

            async def check_job(job_name: str) -> Optional[str]:
                """Check a single job's state, return job_name if completed."""
                try:

                    def get_job():
                        return self.client.batches.get(name=job_name)

                    batch_job = await loop.run_in_executor(None, get_job)
                    state = batch_job.state

                    if state == "JOB_STATE_SUCCEEDED":
                        return job_name
                    elif state == "JOB_STATE_FAILED":
                        raise Exception(f"Batch job {job_name} failed: {batch_job}")
                    elif state in ["JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
                        raise Exception(f"Batch job {job_name} stopped: {state}")
                    return None
                except Exception:
                    raise

            # Check all jobs concurrently
            results = await asyncio.gather(
                *[
                    check_job(job_name)
                    for job_name in job_names
                    if job_name not in completed_jobs
                ],
                return_exceptions=True,
            )

            # Update completed jobs
            for result in results:
                if isinstance(result, str):
                    completed_jobs.add(result)

            # Print progress
            completed_count = len(completed_jobs)
            print(
                f"    Waiting for batch jobs: {completed_count}/{total_jobs} completed",
                end="\r",
            )

            if completed_count < total_jobs:
                await asyncio.sleep(poll_interval)

        print(f"    ✓ All {total_jobs} batch jobs completed successfully")

    def _download_batch_output(self, output_uri_prefix: str) -> List[Dict[str, Any]]:
        """Download and parse batch output from GCS.

        Args:
            output_uri_prefix: GCS URI prefix where outputs were written.

        Returns:
            List of response dictionaries.
        """
        # Parse GCS URI
        uri_parts = output_uri_prefix.replace("gs://", "").split("/", 1)
        bucket_name = uri_parts[0]
        prefix = uri_parts[1] if len(uri_parts) > 1 else ""

        # List all output files
        bucket = self.storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        # Download and parse all output files
        results = []
        for blob in blobs:
            if not blob.name.endswith(".jsonl"):
                continue

            content = blob.download_as_text()
            for line in content.strip().split("\n"):
                if line:
                    results.append(json.loads(line))

        return results

    def _parse_batch_results(
        self,
        results: List[Dict[str, Any]],
        target_tokens: List[Dict[str, Any]],
        top_logprobs: int,
    ) -> List[Dict[str, Any]]:
        """Parse batch prediction results into token logprobs.

        Args:
            results: Batch prediction results.
            target_tokens: Target tokens from Model 1.
            top_logprobs: Number of top logprobs.

        Returns:
            List of token dictionaries with logprobs.
        """
        model2_tokens = []

        # Sort by position
        results_by_position = {}
        for result in results:
            position = int(result.get("position", -1))
            results_by_position[position] = result

        # Process each position in order
        for i, target_token_data in enumerate(target_tokens):
            if i not in results_by_position:
                print(f"    Warning: Missing result for position {i}")
                model2_tokens.append(
                    {
                        "token": target_token_data["token"],
                        "logprob": None,
                        "token_id": target_token_data.get("token_id"),
                        "top_logprobs": [],
                        "error": "Missing batch result",
                    }
                )
                continue

            result = results_by_position[i]
            response = result.get("response", {})

            # Extract logprobs from response
            candidates = response.get("candidates", [])
            if not candidates:
                print(f"    Warning: No candidates for position {i}")
                model2_tokens.append(
                    {
                        "token": target_token_data["token"],
                        "logprob": None,
                        "token_id": target_token_data.get("token_id"),
                        "top_logprobs": [],
                        "error": "No candidates in response",
                    }
                )
                continue

            candidate = candidates[0]
            logprobs_result = candidate.get("logprobsResult", {})
            top_candidates_list = logprobs_result.get("topCandidates", [])

            if not top_candidates_list:
                print(f"    Warning: No logprobs for position {i}")
                model2_tokens.append(
                    {
                        "token": target_token_data["token"],
                        "logprob": None,
                        "token_id": target_token_data.get("token_id"),
                        "top_logprobs": [],
                        "error": "No logprobs in response",
                    }
                )
                continue

            # Get first token's top candidates
            top_candidates = top_candidates_list[0].get("candidates", [])

            # Build token data
            token_data = {
                "token": target_token_data["token"],
                "logprob": None,
                "token_id": target_token_data.get("token_id"),
                "top_logprobs": [],
            }

            # Add top alternatives
            token_data["top_logprobs"] = [
                {
                    "token": cand.get("token", ""),
                    "logprob": cand.get("logProbability", 0.0),
                    "token_id": cand.get("tokenId"),
                }
                for cand in top_candidates
            ]

            # Find Model 1's token in Model 2's distribution
            for alt in token_data["top_logprobs"]:
                if alt["token"] == target_token_data["token"]:
                    token_data["logprob"] = alt["logprob"]
                    break

            # If not found, use floor probability
            if token_data["logprob"] is None and token_data["top_logprobs"]:
                min_logprob = min(alt["logprob"] for alt in token_data["top_logprobs"])
                token_data["logprob"] = min_logprob
                print(
                    f"    Warning: M1 token '{target_token_data['token']}' not in M2's top {top_logprobs} at position {i}, using floor"
                )

            model2_tokens.append(token_data)

        return model2_tokens

    async def get_batch_logprobs(
        self,
        prompt: str,
        target_tokens: List[Dict[str, Any]],
        top_logprobs: int = 20,
        job_prefix: str = "kl_job",
    ) -> List[Dict[str, Any]]:
        """Get logprobs using batch prediction API.

        Args:
            prompt: User prompt.
            target_tokens: List of token data from Model 1.
            top_logprobs: Number of top logprobs to return per token.
            job_prefix: Prefix for batch job names.

        Returns:
            List of token dictionaries with logprobs from Model 2.
        """
        # Create batch requests
        requests = self._create_batch_requests(prompt, target_tokens, top_logprobs)
        print(f"    Created {len(requests)} batch requests")

        # Upload input to GCS
        timestamp = int(time.time())
        input_blob_name = f"batch_input/{job_prefix}_{timestamp}.jsonl"
        input_uri = self._upload_batch_input(requests, input_blob_name)
        print(f"    Uploaded input to {input_uri}")

        # Submit batch job
        output_uri_prefix = (
            f"gs://{self.bucket_name}/batch_output/{job_prefix}_{timestamp}/"
        )
        print("    Submitting batch job...")
        job_name = await self._submit_batch_job(input_uri, output_uri_prefix)
        print(f"    Job submitted: {job_name}")

        # Wait for completion (silent - no verbose state printing)
        await self._wait_for_batch_job(job_name, verbose=False)

        # Download results
        print(f"    Downloading results from {output_uri_prefix}")
        results = self._download_batch_output(output_uri_prefix)
        print(f"    Downloaded {len(results)} results")

        # Parse results
        model2_tokens = self._parse_batch_results(results, target_tokens, top_logprobs)

        return model2_tokens

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
        job_prefix: str = "kl_job",
    ) -> Dict[str, Any]:
        """Process a single response to calculate KL divergence using batch API.

        Args:
            prompt: Original user prompt.
            response_data: Response data from model 1 with logprobs.
            top_logprobs: Number of top logprobs to request.
            response_idx: Optional response index for progress messages.
            job_prefix: Prefix for batch job name.

        Returns:
            Dictionary with KL divergence results.
        """
        text = response_data.get("text", "")
        model1_logprobs = response_data.get("logprobs", {}).get("content", [])

        if not text or not model1_logprobs:
            return {
                "error": "Missing text or logprobs in response data",
                "text": text,
            }

        if response_idx is not None:
            print(
                f"  Processing response {response_idx + 1}: {len(model1_logprobs)} tokens (using batch API)"
            )

        # Get logprobs from model 2 using batch prediction
        try:
            model2_logprobs = await self.get_batch_logprobs(
                prompt=prompt,
                target_tokens=model1_logprobs,
                top_logprobs=top_logprobs,
                job_prefix=f"{job_prefix}_resp{response_idx}",
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
    calculator: KLDivergenceBatchCalculator,
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
    results = []

    # Use filename stem as job prefix
    job_prefix = json_path.stem

    for i, response in enumerate(responses):
        result = await calculator.process_response(
            prompt=prompt,
            response_data=response,
            top_logprobs=top_logprobs,
            response_idx=i,
            job_prefix=job_prefix,
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
    location: str = "us-central1",
    bucket_name: Optional[str] = None,
    top_logprobs: int = 20,
    pattern: str = "*.json",
    max_files: Optional[int] = None,
) -> None:
    """Process all JSON files in a directory using batch prediction.

    Args:
        input_dir: Directory containing JSON files from model 1.
        output_dir: Directory to save KL divergence results.
        model2_id: Model identifier for comparison model.
        project_id: Google Cloud project ID.
        location: Google Cloud region.
        bucket_name: GCS bucket name for batch I/O.
        top_logprobs: Number of top logprobs to request.
        pattern: Glob pattern for JSON files.
        max_files: Maximum number of files to process (None for all).
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

    calculator = KLDivergenceBatchCalculator(
        model_id=model2_id,
        project_id=project_id,
        location=location,
        bucket_name=bucket_name,
    )

    print(f"Comparing against model: {model2_id}")
    print("Using batch prediction API")
    print(f"GCS bucket: {calculator.bucket_name}")
    print(f"Output directory: {output_dir}\n")

    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            await process_json_file(
                json_path=json_file,
                calculator=calculator,
                output_dir=output_path,
                top_logprobs=top_logprobs,
            )
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    print("\n✓ All processing complete!")
    print(f"Results saved to: {output_dir}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate KL divergence using batch prediction API"
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
        help="Google Cloud region for batch jobs (default: us-central1)",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default=None,
        help="GCS bucket name for batch I/O (default: {project_id}-kl-batch)",
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

    args = parser.parse_args()

    asyncio.run(
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model2_id=args.model2,
            project_id=args.project_id,
            location=args.location,
            bucket_name=args.bucket_name,
            top_logprobs=args.top_logprobs,
            pattern=args.pattern,
            max_files=args.max_files,
        )
    )


if __name__ == "__main__":
    main()
