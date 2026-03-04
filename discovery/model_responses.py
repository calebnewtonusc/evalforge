"""
discovery/model_responses.py — Probe candidate questions against multiple models
to detect contamination by consistency analysis.

If multiple models answer a question too consistently (especially with high
confidence), that is evidence the question appears in training data.

Contamination detection logic:
  1. Submit candidate question to N models
  2. If >80% of models answer identically on first token → likely contaminated
  3. If answer matches known-correct answer AND consistency > threshold → flag

Usage:
    python discovery/model_responses.py \
        --questions data/synthesized/candidate_questions.jsonl \
        --output data/raw/model_responses \
        --models gpt2 facebook/opt-125m
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
import requests
from loguru import logger

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    def retry(*args, **kwargs):
        def decorator(fn): return fn
        return decorator
    def stop_after_attempt(n): return None
    def wait_exponential(**kwargs): return None

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VLLM_URLS = os.environ.get(
    "VLLM_URLS",
    "http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004",
).split(",")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "evalforge-secret")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

HF_INFERENCE_API_BASE = "https://api-inference.huggingface.co/models"

# Open-source models that are lightweight enough to call via HF Inference API
PROBE_MODELS: list[dict[str, str]] = [
    {"id": "gpt2", "type": "hf_inference", "label": "GPT-2"},
    {"id": "facebook/opt-125m", "type": "hf_inference", "label": "OPT-125M"},
    {"id": "EleutherAI/gpt-neo-125m", "type": "hf_inference", "label": "GPT-Neo-125M"},
    {"id": "distilgpt2", "type": "hf_inference", "label": "DistilGPT-2"},
]

# Consistency threshold — if >= this fraction of models give same answer, flag contaminated
CONSISTENCY_THRESHOLD = 0.75
# Min models that must respond for a valid contamination judgment
MIN_RESPONDING_MODELS = 2


def _build_prompt(question: str, choices: list[str] | None = None) -> str:
    """Build a question prompt for probing models."""
    if choices:
        choices_str = "\n".join(f"  ({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        return f"Question: {question}\nChoices:\n{choices_str}\nAnswer:"
    return f"Question: {question}\nAnswer:"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _call_hf_inference(model_id: str, prompt: str, max_tokens: int = 5) -> str | None:
    """Call HuggingFace Inference API for text generation. Returns generated text or None."""
    url = f"{HF_INFERENCE_API_BASE}/{model_id}"
    headers = {"Accept": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        },
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 503:
            # Model loading, skip
            return None
        if resp.status_code == 429:
            time.sleep(30)
            resp.raise_for_status()
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()
        return None
    except Exception as exc:
        logger.debug(f"HF inference failed for {model_id}: {exc}")
        return None


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def _call_vllm(url: str, prompt: str, max_tokens: int = 5) -> str | None:
    """Call vLLM OpenAI-compatible API."""
    try:
        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
        resp = requests.post(
            f"{url}/v1/completions",
            json=payload,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"].strip()
    except Exception as exc:
        logger.debug(f"vLLM call failed ({url}): {exc}")
        return None


def _normalize_answer(text: str) -> str:
    """Normalize answer text for comparison: lowercase, strip whitespace and punctuation."""
    return text.lower().strip().rstrip(".,;:!?").split("\n")[0].split(".")[0].strip()


def probe_question(
    question: str,
    choices: list[str] | None = None,
    correct_answer: str | None = None,
) -> dict[str, Any]:
    """
    Probe a single question across all models.

    Returns:
        {
            "question": ...,
            "responses": {model_label: response_text},
            "normalized_responses": {model_label: normalized},
            "consistency": float,          # fraction of models giving same top answer
            "top_answer": str,
            "is_contaminated": bool,
            "contamination_confidence": float,
            "correct_answer_match": bool,   # if correct_answer provided
        }
    """
    prompt = _build_prompt(question, choices)
    responses: dict[str, str] = {}
    normalized: dict[str, str] = {}

    # Try vLLM first if available
    for vllm_url in VLLM_URLS[:2]:
        model_label = f"vllm_{vllm_url.split(':')[-1]}"
        resp = _call_vllm(vllm_url, prompt, max_tokens=8)
        if resp is not None:
            responses[model_label] = resp
            normalized[model_label] = _normalize_answer(resp)

    # HuggingFace Inference API models
    for model_def in PROBE_MODELS:
        resp = _call_hf_inference(model_def["id"], prompt, max_tokens=8)
        if resp is not None:
            responses[model_def["label"]] = resp
            normalized[model_def["label"]] = _normalize_answer(resp)

    if not normalized:
        return {
            "question": question,
            "responses": {},
            "normalized_responses": {},
            "consistency": 0.0,
            "top_answer": "",
            "is_contaminated": False,
            "contamination_confidence": 0.0,
            "correct_answer_match": False,
            "responding_models": 0,
        }

    # Compute answer consistency
    answer_counts: dict[str, int] = {}
    for ans in normalized.values():
        answer_counts[ans] = answer_counts.get(ans, 0) + 1

    top_answer = max(answer_counts, key=lambda k: answer_counts[k])
    top_count = answer_counts[top_answer]
    n_responding = len(normalized)
    consistency = top_count / n_responding if n_responding > 0 else 0.0

    # Contamination flag
    is_contaminated = (
        n_responding >= MIN_RESPONDING_MODELS
        and consistency >= CONSISTENCY_THRESHOLD
    )

    # Check if top answer matches correct answer
    correct_match = False
    if correct_answer:
        correct_norm = _normalize_answer(str(correct_answer))
        correct_match = top_answer == correct_norm

    contamination_confidence = consistency if is_contaminated else 0.0

    return {
        "question": question,
        "responses": responses,
        "normalized_responses": normalized,
        "consistency": round(consistency, 3),
        "top_answer": top_answer,
        "is_contaminated": is_contaminated,
        "contamination_confidence": round(contamination_confidence, 3),
        "correct_answer_match": correct_match,
        "responding_models": n_responding,
    }


async def _probe_async(
    session: aiohttp.ClientSession,
    question_data: dict,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Async wrapper for probe_question (runs in thread pool)."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: probe_question(
                question=question_data.get("question", ""),
                choices=question_data.get("choices"),
                correct_answer=question_data.get("answer"),
            ),
        )
        result["source"] = question_data.get("source", "")
        result["category"] = question_data.get("category", "")
        result["question_id"] = question_data.get("id", "")
        return result


class ModelResponseProber:
    """
    Batch probe questions against multiple models for contamination detection.

    Input:  JSONL file with fields: question, choices (optional), answer (optional)
    Output: JSONL file with contamination analysis per question
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def probe_file(
        self,
        questions_path: str | Path,
        output_filename: str = "model_responses.jsonl",
        concurrency: int = 4,
        max_questions: int = 5000,
    ) -> dict[str, int]:
        """
        Probe all questions in a JSONL file.

        Returns summary statistics.
        """
        questions_path = Path(questions_path)
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")

        questions: list[dict] = []
        with questions_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
                if len(questions) >= max_questions:
                    break

        logger.info(f"Probing {len(questions)} questions for contamination...")
        # asyncio.run() raises RuntimeError when called inside an existing event loop
        # (e.g. when invoked from a Jupyter cell or another async context).
        # Use the loop-safe helper instead.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            future = concurrent.futures.Future()
            def _run():
                import asyncio as _asyncio
                result = _asyncio.run(self._probe_batch_async(questions, concurrency=concurrency))
                future.set_result(result)
            import threading
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join()
            results = future.result()
        else:
            results = asyncio.run(
                self._probe_batch_async(questions, concurrency=concurrency)
            )

        # Save results
        output_path = self.output_dir / output_filename
        with output_path.open("w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")

        # Save contamination-flagged subset
        flagged = [r for r in results if r["is_contaminated"]]
        flagged_path = self.output_dir / output_filename.replace(".jsonl", "_contaminated.jsonl")
        with flagged_path.open("w") as fh:
            for r in flagged:
                fh.write(json.dumps(r) + "\n")

        stats = {
            "total_questions": len(questions),
            "probed": len(results),
            "contaminated": len(flagged),
            "contamination_rate": round(len(flagged) / max(len(results), 1), 3),
            "avg_consistency": round(
                sum(r["consistency"] for r in results) / max(len(results), 1), 3
            ),
        }
        logger.info(f"Probe complete: {stats}")
        return stats

    async def _probe_batch_async(
        self,
        questions: list[dict],
        concurrency: int = 4,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [
                _probe_async(session, q, semaphore) for q in questions
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        processed: list[dict] = []
        for r in results:
            if isinstance(r, Exception):
                logger.debug(f"Probe task failed: {r}")
            else:
                processed.append(r)
        return processed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Probe questions for contamination via model consistency")
    parser.add_argument("--questions", required=True, help="JSONL file with questions to probe")
    parser.add_argument("--output", default="data/raw/model_responses", help="Output directory")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel probes")
    parser.add_argument("--max-questions", type=int, default=5000)
    args = parser.parse_args()

    prober = ModelResponseProber(output_dir=args.output)
    stats = prober.probe_file(
        questions_path=args.questions,
        concurrency=args.concurrency,
        max_questions=args.max_questions,
    )

    logger.info("=== Contamination Probe Summary ===")
    for key, val in stats.items():
        logger.info(f"  {key:<30} {val}")
