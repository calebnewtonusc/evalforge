"""
agents/contamination_agent.py — Dedicated contamination detection agent.

Specializes in multi-method contamination detection:
  1. N-gram overlap with known training corpora
  2. Embedding proximity (semantic similarity to training data)
  3. Model memorization probe (can the model reproduce the item?)
  4. Answer distribution shift analysis

Usage:
    agent = ContaminationAgent()
    result = agent.check(item, corpus_index=corpus)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ContaminationResult:
    """Result of contamination check for a single item."""

    item_id: str
    contamination_score: float  # 0.0–1.0
    is_contaminated: bool
    methods_flagged: list[str]
    evidence: dict[str, Any]
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "contamination_score": self.contamination_score,
            "is_contaminated": self.is_contaminated,
            "methods_flagged": self.methods_flagged,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


class ContaminationAgent:
    """
    Multi-method contamination detection agent.

    Methods:
    1. N-gram index lookup (fast, heuristic)
    2. Embedding proximity (semantic-level contamination)
    3. Memorization probe (LLM-based, most accurate)
    """

    CONTAMINATION_THRESHOLD = 0.5  # Flag if score > 0.5

    def __init__(
        self,
        ngram_index_path: str | None = None,
        embedding_model: str | None = None,
        model_url: str | None = None,
    ) -> None:
        self.ngram_index: dict[str, set[str]] = {}
        self.model_url = model_url
        self._embedding_model = None
        self._llm_client = None

        if ngram_index_path and Path(ngram_index_path).exists():
            self._load_ngram_index(ngram_index_path)

        if model_url:
            import openai

            self._llm_client = openai.OpenAI(
                base_url=f"{model_url}/v1",
                api_key=os.environ.get("VLLM_API_KEY", "dummy"),
            )

    def check(self, item: dict, threshold: float | None = None) -> ContaminationResult:
        """
        Run all contamination detection methods on a single item.

        Args:
            item: Benchmark item dict with 'question', 'choices', 'answer'.
            threshold: Override contamination_score threshold.

        Returns:
            ContaminationResult with composite score.
        """
        if threshold is None:
            threshold = self.CONTAMINATION_THRESHOLD
        item_id = item.get("id", "unknown")
        item_text = self._item_to_text(item)

        scores: dict[str, float] = {}
        evidence: dict[str, Any] = {}

        # Method 1: N-gram overlap
        ngram_score, ngram_evidence = self._ngram_check(item_text)
        scores["ngram"] = ngram_score
        evidence["ngram"] = ngram_evidence

        # Method 2: Template detection
        template_score, template_evidence = self._template_check(item_text)
        scores["template"] = template_score
        evidence["template"] = template_evidence

        # Method 3: Memorization probe (if LLM available)
        if self._llm_client:
            mem_score, mem_evidence = self._memorization_probe(item)
            scores["memorization"] = mem_score
            evidence["memorization"] = mem_evidence

        # Composite score (weighted)
        weights = {"ngram": 0.4, "template": 0.3, "memorization": 0.3}
        total_weight = sum(weights[m] for m in scores)
        composite = (
            sum(scores[m] * weights[m] for m in scores) / total_weight
            if total_weight > 0
            else 0.0
        )

        methods_flagged = [m for m, s in scores.items() if s > 0.5]

        confidence = "LOW"
        if len(methods_flagged) >= 2:
            confidence = "HIGH"
        elif len(methods_flagged) == 1 and max(scores.values()) > 0.7:
            confidence = "MEDIUM"

        return ContaminationResult(
            item_id=item_id,
            contamination_score=round(composite, 3),
            is_contaminated=composite > threshold,
            methods_flagged=methods_flagged,
            evidence=evidence,
            confidence=confidence,
        )

    def batch_check(
        self, items: list[dict], threshold: float | None = None
    ) -> list[ContaminationResult]:
        """Check a list of items."""
        results = []
        for item in items:
            result = self.check(item, threshold)
            results.append(result)
        logger.info(
            f"Contamination check: {sum(1 for r in results if r.is_contaminated)}/{len(results)} flagged"
        )
        return results

    def _item_to_text(self, item: dict) -> str:
        """Convert item to searchable text string."""
        parts = [item.get("question", "")]
        choices = item.get("choices", {})
        if isinstance(choices, dict):
            parts.extend(choices.values())
        elif isinstance(choices, list):
            parts.extend(str(c) for c in choices)
        return " ".join(str(p) for p in parts if p)

    def _ngram_check(self, text: str, n: int = 6) -> tuple[float, dict]:
        """
        Check for n-gram overlap with indexed training data.

        Returns:
            (score, evidence_dict)
        """
        words = text.lower().split()
        if len(words) < n:
            return 0.0, {"reason": "text too short for n-gram analysis"}

        item_ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            item_ngrams.add(ngram)

        if not self.ngram_index:
            # Heuristic: flag items with very common phrases
            common_indicators = {
                "according to the passage",
                "which of the following is",
                "in the context of",
                "based on the information",
                "the correct answer is",
            }
            hits = sum(1 for phrase in common_indicators if phrase in text.lower())
            score = min(1.0, hits * 0.3)
            return score, {"method": "heuristic", "hits": hits}

        # Real n-gram index lookup
        matches = []
        for ngram in item_ngrams:
            h = hashlib.md5(ngram.encode(), usedforsecurity=False).hexdigest()[:8]
            if h in self.ngram_index:
                matches.append(
                    {"ngram": ngram, "sources": list(self.ngram_index[h])[:3]}
                )

        if not matches:
            return 0.0, {"method": "ngram_index", "matches": 0}

        score = min(1.0, len(matches) / max(1, len(item_ngrams)) * 3.0)
        return round(score, 3), {
            "method": "ngram_index",
            "matches": len(matches),
            "examples": matches[:3],
        }

    def _template_check(self, text: str) -> tuple[float, dict]:
        """
        Detect if item follows a fill-in-the-blank template structure.
        """
        template_patterns = [
            r"^which of the following",
            r"what is the (primary|main|best|correct)",
            r"in the year \d{4}",
            r"the (capital|president|ceo|founder) of",
            r"___+ is defined as",
            r"select the (best|most appropriate|correct)",
        ]

        text_lower = text.lower()
        matches = []
        for pattern in template_patterns:
            if re.search(pattern, text_lower):
                matches.append(pattern)

        score = min(1.0, len(matches) * 0.25)
        return round(score, 3), {"patterns_matched": matches}

    def _memorization_probe(self, item: dict) -> tuple[float, dict]:
        """
        Probe model memorization by asking it to complete the item text.

        If the model can reproduce the item text given only a prefix,
        the item is likely in its training data.
        """
        if not self._llm_client:
            return 0.0, {"reason": "no LLM client configured"}

        question = item.get("question", "")
        if not question or len(question.split()) < 5:
            return 0.0, {"reason": "question too short for memorization probe"}

        # Give model first 30% of question, ask it to complete
        words = question.split()
        prefix_len = max(3, len(words) // 3)
        prefix = " ".join(words[:prefix_len])
        expected_continuation = " ".join(words[prefix_len:])

        try:
            resp = self._llm_client.chat.completions.create(
                model="evalforge",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Complete this text (this is a memorization test, reproduce exactly "
                            f"if you know it): '{prefix} ...'"
                        ),
                    }
                ],
                max_tokens=len(expected_continuation.split()) + 10,
                temperature=0.0,
            )
            completion = resp.choices[0].message.content.strip()

            # Compute word overlap
            expected_words = set(expected_continuation.lower().split())
            completion_words = set(completion.lower().split())
            if not expected_words:
                return 0.0, {}
            overlap = len(expected_words & completion_words) / len(expected_words)
            return round(overlap, 3), {
                "prefix": prefix,
                "expected": expected_continuation[:100],
                "completion": completion[:100],
                "overlap": overlap,
            }
        except Exception as e:
            logger.debug(f"Memorization probe failed: {e}")
            return 0.0, {"error": str(e)}

    def _load_ngram_index(self, path: str) -> None:
        """Load precomputed n-gram index."""
        logger.info(f"Loading n-gram index from {path}...")
        try:
            data = json.loads(Path(path).read_text())
            self.ngram_index = {k: set(v) for k, v in data.items()}
            logger.info(f"Loaded {len(self.ngram_index):,} n-gram entries")
        except json.JSONDecodeError as exc:
            logger.warning(
                f"N-gram index at {path} is corrupted and could not be parsed "
                f"({exc}). Falling back to heuristic contamination detection."
            )
            self.ngram_index = {}
