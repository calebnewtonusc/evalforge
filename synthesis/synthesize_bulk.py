"""
synthesis/synthesize_bulk.py — Parallel synthesis of EvalForge training pairs.

Generates training pairs across all 5 data streams:
  1. Benchmark audit pairs (critique paper → structured audit)
  2. Shortcut detection pairs (items → shortcut analysis)
  3. IRT calibration pairs (response matrix → IRT parameters)
  4. Goodhart pattern pairs (case study → pattern analysis)
  5. Item generation pairs (constraints → novel items)

Usage:
    python synthesis/synthesize_bulk.py --backend vllm --workers 20
    python synthesis/synthesize_bulk.py --backend claude --workers 5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from loguru import logger

try:
    from tenacity import retry, stop_after_attempt, wait_exponential

    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False

    def retry(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def stop_after_attempt(n):
        return None

    def wait_exponential(**kwargs):
        return None


from synthesis.prompts import (
    AUDIT_PAIR_PROMPT,
    GOODHART_PATTERN_PROMPT,
    ITEM_GENERATION_PROMPT,
    SHORTCUT_DETECTION_PROMPT,
    SYSTEM_AUDIT,
    SYSTEM_GOODHART,
    SYSTEM_ITEM_GENERATOR,
    SYSTEM_SHORTCUT_DETECTOR,
)


class BulkSynthesizer:
    """
    Orchestrates bulk synthesis across all EvalForge data streams.

    Supports two backends:
    - 'vllm': uses local vLLM servers (fast, requires GPU)
    - 'claude': uses Anthropic API (slower, no GPU required)
    """

    def __init__(
        self,
        raw_dir: str | Path,
        output_dir: str | Path,
        backend: str = "vllm",
        vllm_urls: list[str] | None = None,
        workers: int = 20,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.vllm_urls = vllm_urls or ["http://localhost:8001"]
        self.workers = workers

        if backend == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Set it before using the claude backend."
                )
            self.claude_client = anthropic.Anthropic(api_key=api_key)
        else:
            import openai

            self.vllm_clients = [
                openai.OpenAI(
                    base_url=f"{url}/v1",
                    api_key=os.environ.get("VLLM_API_KEY", "dummy"),
                )
                for url in self.vllm_urls
            ]
            self._client_idx = 0
            self._client_lock = threading.Lock()

    def _next_client(self):
        """Round-robin across vLLM instances (thread-safe)."""
        with self._client_lock:
            client = self.vllm_clients[self._client_idx % len(self.vllm_clients)]
            self._client_idx += 1
        return client

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
    def _call_llm(
        self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 2048
    ) -> str:
        """Call LLM backend with retry logic."""
        if self.backend == "claude":
            msg = self.claude_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        else:
            client = self._next_client()
            resp = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

    def _parse_json_response(self, text: str) -> dict | None:
        """Extract and parse JSON from LLM response."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON block
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return None

    def _make_audit_pair(self, paper: dict) -> dict | None:
        """Generate an audit pair from a benchmark critique paper."""
        # Extract a plausible benchmark item from the paper context
        item_json = json.dumps(
            {
                "question": paper.get("title", "")[:200],
                "abstract_excerpt": paper.get("abstract", "")[:300],
                "reviews_excerpt": [
                    r["text"][:200] for r in paper.get("reviews", [])[:2]
                ],
            },
            indent=2,
        )

        user = AUDIT_PAIR_PROMPT.format(
            benchmark_name=paper.get("venue", "Unknown"),
            category="methodology",
            item_json=item_json,
            model_scores_json=json.dumps({"hypothetical_model": 0.85}, indent=2),
        )

        raw = self._call_llm(system=SYSTEM_AUDIT, user=user)
        parsed = self._parse_json_response(raw)
        if parsed is None:
            return None

        return {
            "id": str(uuid.uuid4()),
            "type": "audit_pair",
            "source_paper": paper.get("id", ""),
            "conversations": [
                {"from": "system", "value": SYSTEM_AUDIT},
                {"from": "human", "value": user},
                {"from": "gpt", "value": json.dumps(parsed, indent=2)},
            ],
        }

    def _make_shortcut_pair(self, benchmark_items: list[dict]) -> dict | None:
        """Generate a shortcut detection pair from benchmark items."""
        items_json = json.dumps(benchmark_items[:10], indent=2)
        user = SHORTCUT_DETECTION_PROMPT.format(items_json=items_json)
        raw = self._call_llm(system=SYSTEM_SHORTCUT_DETECTOR, user=user)
        parsed = self._parse_json_response(raw)
        if parsed is None:
            return None

        return {
            "id": str(uuid.uuid4()),
            "type": "shortcut_detection",
            "conversations": [
                {"from": "system", "value": SYSTEM_SHORTCUT_DETECTOR},
                {"from": "human", "value": user},
                {"from": "gpt", "value": json.dumps(parsed, indent=2)},
            ],
        }

    def _make_goodhart_pair(self, case_study: str, domain: str) -> dict | None:
        """Generate a Goodhart pattern analysis pair."""
        user = GOODHART_PATTERN_PROMPT.format(case_study=case_study, domain=domain)
        raw = self._call_llm(system=SYSTEM_GOODHART, user=user)
        parsed = self._parse_json_response(raw)
        if parsed is None:
            return None

        return {
            "id": str(uuid.uuid4()),
            "type": "goodhart_pattern",
            "conversations": [
                {"from": "system", "value": SYSTEM_GOODHART},
                {"from": "human", "value": user},
                {"from": "gpt", "value": json.dumps(parsed, indent=2)},
            ],
        }

    def _make_item_gen_pair(
        self, construct: str, difficulty_range: tuple
    ) -> dict | None:
        """Generate a novel item generation pair."""
        user = ITEM_GENERATION_PROMPT.format(
            n_items=5,
            construct=construct,
            difficulty_min=difficulty_range[0],
            difficulty_max=difficulty_range[1],
            shortcuts_to_avoid="length_bias, lexical_overlap, negation_artifact",
            benchmark_style="multiple_choice_4_option",
            existing_items_sample="[]",
        )
        raw = self._call_llm(system=SYSTEM_ITEM_GENERATOR, user=user, max_tokens=3000)
        parsed = self._parse_json_response(raw)
        if parsed is None:
            return None

        return {
            "id": str(uuid.uuid4()),
            "type": "item_generation",
            "construct": construct,
            "conversations": [
                {"from": "system", "value": SYSTEM_ITEM_GENERATOR},
                {"from": "human", "value": user},
                {"from": "gpt", "value": json.dumps(parsed, indent=2)},
            ],
        }

    def run(self) -> dict[str, int]:
        """Run all synthesis streams. Returns stats dict."""
        stats: dict[str, int] = {
            "audit_pairs": 0,
            "shortcut_pairs": 0,
            "irt_pairs": 0,
            "goodhart_pairs": 0,
            "item_gen_pairs": 0,
            "total_pairs": 0,
        }

        # Stream 1: Audit pairs from OpenReview papers
        openreview_dir = self.raw_dir / "openreview"
        paper_files = (
            list(openreview_dir.glob("*.json")) if openreview_dir.exists() else []
        )
        if paper_files:
            logger.info(
                f"Stream 1: Generating audit pairs from {len(paper_files):,} papers..."
            )
            n = self._synthesize_stream(
                items=paper_files,
                fn=lambda p: self._make_audit_pair(json.loads(p.read_text())),
                output_file=self.output_dir / "audit_pairs.jsonl",
                label="audit",
            )
            stats["audit_pairs"] = n

        # Stream 2: Shortcut pairs from benchmark items
        bench_dir = self.raw_dir / "benchmarks"
        bench_files = (
            list(bench_dir.glob("*/items.jsonl")) if bench_dir.exists() else []
        )
        if bench_files:
            logger.info(
                f"Stream 2: Generating shortcut pairs from {len(bench_files)} benchmarks..."
            )
            all_items: list[list[dict]] = []
            for f in bench_files:
                chunk = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
                # Create batches of 10 items
                for i in range(0, len(chunk), 10):
                    all_items.append(chunk[i : i + 10])

            n = self._synthesize_stream(
                items=all_items,
                fn=self._make_shortcut_pair,
                output_file=self.output_dir / "shortcut_pairs.jsonl",
                label="shortcut",
            )
            stats["shortcut_pairs"] = n

        # Stream 3: Goodhart pattern pairs (synthetic case studies)
        goodhart_cases = self._load_goodhart_cases()
        if goodhart_cases:
            logger.info(
                f"Stream 3: Generating {len(goodhart_cases)} Goodhart pattern pairs..."
            )
            n = self._synthesize_stream(
                items=goodhart_cases,
                fn=lambda c: self._make_goodhart_pair(c["case_study"], c["domain"]),
                output_file=self.output_dir / "goodhart_pairs.jsonl",
                label="goodhart",
            )
            stats["goodhart_pairs"] = n

        # Stream 4: Item generation pairs
        constructs = self._load_construct_list()
        if constructs:
            logger.info(
                f"Stream 4: Generating item pairs for {len(constructs)} constructs..."
            )
            items_with_ranges = [
                (c, (round(-2.0 + i * 0.1, 1), round(-1.0 + i * 0.1, 1)))
                for i, c in enumerate(constructs)
            ]
            n = self._synthesize_stream(
                items=items_with_ranges,
                fn=lambda t: self._make_item_gen_pair(t[0], t[1]),
                output_file=self.output_dir / "item_gen_pairs.jsonl",
                label="item_gen",
            )
            stats["item_gen_pairs"] = n

        stats["total_pairs"] = sum(v for k, v in stats.items() if k != "total_pairs")
        logger.info(f"Synthesis complete. Total pairs: {stats['total_pairs']:,}")
        return stats

    def _synthesize_stream(
        self,
        items: list,
        fn,
        output_file: Path,
        label: str,
    ) -> int:
        """Run parallel synthesis for one stream."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        saved = 0
        failed = 0

        with open(output_file, "w") as f_out:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(fn, item): item for item in items}
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)
                        if result is not None:
                            f_out.write(json.dumps(result) + "\n")
                            saved += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        logger.debug(f"  {label} synthesis failed: {e}")

                    if (saved + failed) % 100 == 0:
                        logger.info(f"  {label}: {saved} saved, {failed} failed")

        logger.info(f"  {label}: {saved} pairs saved to {output_file}")
        return saved

    def _load_goodhart_cases(self) -> list[dict]:
        """Load built-in Goodhart's Law case studies."""
        return [
            {
                "domain": "corporate",
                "case_study": (
                    "Wells Fargo was pressured to meet cross-selling quotas. "
                    "The metric: number of accounts per customer. Branch employees opened "
                    "unauthorized accounts to hit targets. The proxy (account count) became "
                    "detached from the underlying construct (customer relationship depth)."
                ),
            },
            {
                "domain": "academic",
                "case_study": (
                    "Citation count became the dominant metric for academic impact. "
                    "Researchers began citing each other extensively to boost counts, "
                    "forming citation cartels. High citation count no longer reliably "
                    "signals scientific importance."
                ),
            },
            {
                "domain": "ai_benchmarks",
                "case_study": (
                    "BLEU score was established as the standard MT quality metric. "
                    "Systems were optimized to maximize BLEU rather than translation quality. "
                    "Models learned to produce n-gram overlapping text that scores highly on BLEU "
                    "but is judged lower quality by humans."
                ),
            },
            {
                "domain": "social_media",
                "case_study": (
                    "Engagement (likes, shares, comments) became the primary optimization target "
                    "for recommendation algorithms. Content that maximizes engagement was found "
                    "to be systematically more outrage-inducing, divisive, and emotionally "
                    "manipulative — the proxy detached from the construct of 'valuable content'."
                ),
            },
        ]

    def _load_construct_list(self) -> list[str]:
        """Load list of cognitive constructs for item generation."""
        return [
            "multi-step logical deduction",
            "causal reasoning",
            "temporal reasoning",
            "counterfactual reasoning",
            "mathematical word problem solving",
            "reading comprehension — inference",
            "reading comprehension — main idea",
            "analogical reasoning",
            "spatial reasoning",
            "scientific reasoning — hypothesis generation",
            "commonsense reasoning",
            "ethical reasoning under uncertainty",
        ]
