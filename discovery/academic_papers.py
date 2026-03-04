"""
discovery/academic_papers.py — Crawl academic papers about AI evaluation, benchmarks,
and contamination from Semantic Scholar.

Extracts:
  - Papers citing "benchmark", "evaluation", "contamination"
  - Evaluation methodologies and their known failure modes
  - Critique papers that identify shortcuts in existing benchmarks
  - Survey papers on benchmark saturation

Usage:
    python discovery/academic_papers.py \
        --output data/raw/papers \
        --max-papers 5000
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

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
VLLM_URLS = os.environ.get("VLLM_URLS", "http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004").split(",")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

# Search queries for finding evaluation-critical papers
PAPER_QUERIES: list[dict[str, Any]] = [
    {
        "query": "benchmark contamination language model evaluation",
        "category": "contamination",
        "limit": 500,
    },
    {
        "query": "shortcut learning NLP benchmark dataset bias",
        "category": "shortcuts",
        "limit": 500,
    },
    {
        "query": "benchmark saturation leaderboard overfitting evaluation",
        "category": "saturation",
        "limit": 400,
    },
    {
        "query": "evaluation methodology language model failure modes",
        "category": "methodology",
        "limit": 400,
    },
    {
        "query": "dataset artifacts annotation bias NLP evaluation",
        "category": "artifacts",
        "limit": 400,
    },
    {
        "query": "train test contamination data leakage large language model",
        "category": "contamination",
        "limit": 500,
    },
    {
        "query": "robust evaluation adversarial benchmark LLM",
        "category": "adversarial",
        "limit": 300,
    },
    {
        "query": "MMLU benchmark critique limitations failure",
        "category": "critique",
        "limit": 300,
    },
    {
        "query": "HumanEval benchmark code generation limitations",
        "category": "critique",
        "limit": 200,
    },
    {
        "query": "math benchmark GSM8K MATH reasoning evaluation",
        "category": "math_eval",
        "limit": 300,
    },
    {
        "query": "dynamic benchmark adversarial data collection evaluation",
        "category": "dynamic",
        "limit": 300,
    },
    {
        "query": "capability evaluation AI safety benchmark",
        "category": "safety",
        "limit": 300,
    },
]

FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "citationCount",
    "authors",
    "venue",
    "fieldsOfStudy",
    "externalIds",
    "tldr",
]


def _s2_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return headers


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def _s2_search(
    query: str,
    offset: int = 0,
    limit: int = 100,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Call Semantic Scholar paper search endpoint."""
    if fields is None:
        fields = FIELDS

    resp = requests.get(
        f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
        params={
            "query": query,
            "offset": offset,
            "limit": min(limit, 100),
            "fields": ",".join(fields),
        },
        headers=_s2_headers(),
        timeout=30,
    )
    # Let raise_for_status() fire on 429; the @retry decorator with
    # wait_exponential handles backoff. A manual sleep here would execute
    # and then raise anyway, wasting time without giving the retry a chance.
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _s2_paper_details(paper_id: str) -> dict[str, Any]:
    """Fetch full paper details including references."""
    resp = requests.get(
        f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
        params={
            "fields": ",".join(FIELDS + ["references", "citations"]),
        },
        headers=_s2_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_methodology_patterns(abstract: str) -> list[str]:
    """
    Extract evaluation methodology descriptions from a paper abstract.
    Returns a list of identified methodology snippets.
    """
    if not abstract:
        return []

    patterns: list[str] = []

    # Find sentences describing evaluation approaches
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    eval_keywords = [
        "evaluat", "benchmark", "measur", "assess", "test", "metric",
        "performance", "accuracy", "contamin", "shortcut", "artifact",
        "bias", "generaliz", "robustness",
    ]
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(kw in sentence_lower for kw in eval_keywords):
            patterns.append(sentence.strip())

    return patterns[:5]  # Cap at 5 patterns per abstract


def _extract_failure_modes(abstract: str) -> list[str]:
    """
    Extract described failure modes / critique points from an abstract.
    """
    if not abstract:
        return []

    failure_keywords = [
        "fail", "limitation", "problem", "issue", "bias", "shortcut",
        "artifact", "contamin", "overfit", "saturate", "incorrect",
        "mislead", "flawed", "invalid", "wrong",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    failure_sentences: list[str] = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(kw in sentence_lower for kw in failure_keywords):
            failure_sentences.append(sentence.strip())

    return failure_sentences[:3]


def _score_paper_relevance(paper: dict[str, Any]) -> float:
    """
    Score 0.0–1.0 relevance for EvalForge training data extraction.
    Higher = more relevant to evaluation critique.
    """
    score = 0.0
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    text = title + " " + abstract

    high_value_terms = [
        "contamination", "shortcut", "artifact", "benchmark critique",
        "evaluation failure", "spurious", "overfitting benchmark",
        "data leakage", "annotation artifact", "saturation",
    ]
    medium_value_terms = [
        "evaluation", "benchmark", "assessment", "performance measurement",
        "metric", "leaderboard", "robustness", "generalization",
    ]

    for term in high_value_terms:
        if term in text:
            score += 0.15
    for term in medium_value_terms:
        if term in text:
            score += 0.05

    citations = paper.get("citationCount", 0) or 0
    if citations > 500:
        score += 0.3
    elif citations > 100:
        score += 0.2
    elif citations > 20:
        score += 0.1

    year = paper.get("year", 0) or 0
    if year >= 2022:
        score += 0.1
    elif year >= 2020:
        score += 0.05

    return min(score, 1.0)


class AcademicPaperCrawler:
    """
    Crawls Semantic Scholar for papers about AI evaluation and benchmarks.

    Output:
      data/raw/papers/papers.jsonl          — all papers with metadata
      data/raw/papers/methodologies.jsonl   — extracted evaluation methodology patterns
      data/raw/papers/failure_modes.jsonl   — extracted failure mode descriptions
      data/raw/papers/high_value.jsonl      — top-scored papers for training data
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seen_ids: set[str] = set()

    def run(self, max_papers: int = 5000) -> dict[str, int]:
        """Crawl all query categories and return paper counts per category."""
        all_papers: list[dict] = []
        methodologies: list[dict] = []
        failure_modes: list[dict] = []

        per_query_limit = max_papers // len(PAPER_QUERIES)

        for query_def in PAPER_QUERIES:
            query = query_def["query"]
            category = query_def["category"]
            limit = min(query_def["limit"], per_query_limit)

            logger.info(f"Querying: '{query}' (limit={limit})")
            papers = self._crawl_query(query, category, limit)
            all_papers.extend(papers)
            logger.info(f"  {category}: {len(papers)} papers")

            # Extract patterns
            for paper in papers:
                abstract = paper.get("abstract") or ""
                paper_id = paper.get("paperId", "")
                title = paper.get("title", "")

                meth = _extract_methodology_patterns(abstract)
                if meth:
                    methodologies.append({
                        "paper_id": paper_id,
                        "title": title,
                        "category": category,
                        "patterns": meth,
                        "year": paper.get("year"),
                        "citations": paper.get("citationCount", 0),
                    })

                fails = _extract_failure_modes(abstract)
                if fails:
                    failure_modes.append({
                        "paper_id": paper_id,
                        "title": title,
                        "category": category,
                        "failure_modes": fails,
                        "year": paper.get("year"),
                        "citations": paper.get("citationCount", 0),
                    })

            time.sleep(1.0)  # Polite delay between queries

        # Score and sort for high-value papers
        for paper in all_papers:
            paper["relevance_score"] = _score_paper_relevance(paper)
        high_value = [p for p in all_papers if p["relevance_score"] >= 0.4]
        high_value.sort(key=lambda p: p["relevance_score"], reverse=True)

        self._save_jsonl("papers.jsonl", all_papers)
        self._save_jsonl("methodologies.jsonl", methodologies)
        self._save_jsonl("failure_modes.jsonl", failure_modes)
        self._save_jsonl("high_value.jsonl", high_value[:2000])

        logger.info(
            f"Papers total: {len(all_papers)}, "
            f"high-value: {len(high_value)}, "
            f"methodologies: {len(methodologies)}, "
            f"failure_modes: {len(failure_modes)}"
        )

        return {
            "total_papers": len(all_papers),
            "high_value": len(high_value),
            "methodology_papers": len(methodologies),
            "failure_mode_papers": len(failure_modes),
        }

    def _crawl_query(
        self, query: str, category: str, limit: int
    ) -> list[dict]:
        """Fetch papers for a single search query, paginating as needed."""
        papers: list[dict] = []
        offset = 0
        page_size = 100

        while len(papers) < limit:
            batch_limit = min(page_size, limit - len(papers))
            try:
                result = _s2_search(query, offset=offset, limit=batch_limit)
                batch = result.get("data", [])
                if not batch:
                    break

                for paper in batch:
                    pid = paper.get("paperId")
                    if not pid or pid in self._seen_ids:
                        continue
                    self._seen_ids.add(pid)

                    # Normalize
                    papers.append({
                        "paperId": pid,
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", ""),
                        "year": paper.get("year"),
                        "citationCount": paper.get("citationCount", 0),
                        "authors": [a.get("name", "") for a in (paper.get("authors") or [])[:5]],
                        "venue": paper.get("venue", ""),
                        "fieldsOfStudy": paper.get("fieldsOfStudy") or [],
                        "tldr": (paper.get("tldr") or {}).get("text", ""),
                        "category": category,
                        "query": query,
                    })

                offset += len(batch)
                total = result.get("total", 0)
                if offset >= total:
                    break

                time.sleep(0.3)

            except Exception as exc:
                logger.warning(f"  Query failed at offset {offset}: {exc}")
                break

        return papers

    def _save_jsonl(self, filename: str, records: list[dict]) -> None:
        path = self.output_dir / filename
        with path.open("w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        logger.debug(f"  Saved {len(records)} records → {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl academic papers about AI evaluation")
    parser.add_argument("--output", default="data/raw/papers", help="Output directory")
    parser.add_argument("--max-papers", type=int, default=5000, help="Max papers to collect")
    args = parser.parse_args()

    crawler = AcademicPaperCrawler(output_dir=args.output)
    stats = crawler.run(max_papers=args.max_papers)

    logger.info("=== Paper Crawl Summary ===")
    for key, val in stats.items():
        logger.info(f"  {key:<30} {val:>5}")
