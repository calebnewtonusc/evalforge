"""
discovery/openreview_crawler.py — Crawl OpenReview for evaluation methodology papers.

Fetches papers + reviews + rebuttals from NeurIPS, ICLR, ICML, ACL, EMNLP
where the content is relevant to benchmark design, contamination, or shortcuts.

Usage:
    python discovery/openreview_crawler.py \
        --venues NeurIPS ICLR ICML \
        --query "benchmark contamination shortcut" \
        --max-papers 50000 \
        --output data/raw/openreview
"""

from __future__ import annotations

import json
import time
from datetime import datetime
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


OPENREVIEW_API_V2 = "https://api2.openreview.net"

VENUE_IDS: dict[str, list[str]] = {
    "NeurIPS": [f"NeurIPS.cc/{y}" for y in range(2018, 2027)],
    "ICLR": [f"ICLR.cc/{y}/Conference" for y in range(2018, 2027)],
    "ICML": [f"ICML.cc/{y}/Conference" for y in range(2018, 2027)],
    "ACL": [f"aclweb.org/ACL/{y}/Conference" for y in range(2020, 2027)],
    "EMNLP": [f"aclweb.org/EMNLP/{y}/Conference" for y in range(2020, 2027)],
}

EVAL_QUERY_TERMS = [
    "benchmark", "evaluation", "contamination", "shortcut", "construct validity",
    "annotation artifact", "Goodhart", "leakage", "data contamination",
    "benchmark gaming", "memorization", "item response theory",
]


class OpenReviewCrawler:
    """Crawl OpenReview for evaluation-relevant papers and their reviews."""

    def __init__(self, output_dir: str | Path, rate_limit: float = 1.0) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit  # seconds between requests
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "EvalForge-Crawler/1.0 (research)"})

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict:
        """Make a rate-limited GET request to OpenReview API."""
        time.sleep(self.rate_limit)
        resp = self.session.get(f"{OPENREVIEW_API_V2}{endpoint}", params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def run(
        self,
        venues: list[str],
        query_terms: list[str] | None = None,
        max_papers: int = 50_000,
        since_year: int = 2018,
    ) -> int:
        """
        Crawl papers from specified venues, filter by query terms.

        Returns:
            Number of papers saved.
        """
        if query_terms is None:
            query_terms = EVAL_QUERY_TERMS

        total_saved = 0
        seen_ids: set[str] = set()

        # Load existing IDs to allow resumable crawl
        existing_ids_path = self.output_dir / "crawled_ids.txt"
        if existing_ids_path.exists():
            seen_ids = set(existing_ids_path.read_text().splitlines())
            logger.info(f"Resuming from {len(seen_ids):,} already-crawled papers")

        for venue_name in venues:
            if venue_name not in VENUE_IDS:
                logger.warning(f"Unknown venue: {venue_name}")
                continue

            for venue_id in VENUE_IDS[venue_name]:
                year = int(venue_id.split("/")[1]) if "/" in venue_id else 0
                if year < since_year:
                    continue
                logger.info(f"Crawling {venue_id}...")
                n = self._crawl_venue(
                    venue_id=venue_id,
                    query_terms=query_terms,
                    max_papers=max_papers - total_saved,
                    seen_ids=seen_ids,
                )
                total_saved += n
                logger.info(f"  {venue_id}: {n} papers saved (total: {total_saved:,})")

                if total_saved >= max_papers:
                    logger.info(f"Reached max_papers={max_papers}, stopping")
                    break

        # Save crawled IDs for resumption
        existing_ids_path.write_text("\n".join(sorted(seen_ids)))
        logger.info(f"Crawl complete. Total papers: {total_saved:,}")
        return total_saved

    def _crawl_venue(
        self,
        venue_id: str,
        query_terms: list[str],
        max_papers: int,
        seen_ids: set[str],
    ) -> int:
        """Crawl one venue, returning number of saved papers."""
        saved = 0
        offset = 0
        page_size = 100

        while saved < max_papers:
            try:
                result = self._get(
                    "/notes",
                    params={
                        "venue": venue_id,
                        "offset": offset,
                        "limit": page_size,
                        "details": "replyCount,invitation",
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to fetch {venue_id} offset={offset}: {e}")
                break

            notes = result.get("notes", [])
            if not notes:
                break

            for note in notes:
                note_id = note.get("id", "")
                if note_id in seen_ids:
                    continue

                # Filter by relevance to evaluation methodology
                if not self._is_eval_relevant(note, query_terms):
                    continue

                # Fetch reviews for this paper
                paper_data = self._enrich_with_reviews(note)
                if paper_data is None:
                    continue

                # Save
                out_path = self.output_dir / f"{note_id}.json"
                out_path.write_text(json.dumps(paper_data, indent=2))
                seen_ids.add(note_id)
                saved += 1

                if saved >= max_papers:
                    break

            offset += page_size
            if len(notes) < page_size:
                break

        return saved

    def _is_eval_relevant(self, note: dict, query_terms: list[str]) -> bool:
        """Check if a paper is relevant to evaluation methodology."""
        content = note.get("content", {})
        title = content.get("title", {})
        if isinstance(title, dict):
            title = title.get("value", "")
        abstract = content.get("abstract", {})
        if isinstance(abstract, dict):
            abstract = abstract.get("value", "")

        text = f"{title} {abstract}".lower()
        return any(term.lower() in text for term in query_terms)

    def _enrich_with_reviews(self, note: dict) -> dict | None:
        """Fetch reviews and rebuttals for a paper."""
        note_id = note.get("id", "")
        content = note.get("content", {})

        def _val(field: Any) -> str:
            return field.get("value", "") if isinstance(field, dict) else (field or "")

        paper_data: dict[str, Any] = {
            "id": note_id,
            "title": _val(content.get("title", "")),
            "abstract": _val(content.get("abstract", "")),
            "keywords": _val(content.get("keywords", "")),
            "venue": note.get("venue", ""),
            "year": datetime.utcfromtimestamp(note.get("cdate", 0) / 1000).year if note.get("cdate") else 0,
            "reviews": [],
            "rebuttals": [],
        }

        try:
            reviews_result = self._get("/notes", params={"replyto": note_id, "limit": 50})
            for review_note in reviews_result.get("notes", []):
                review_content = review_note.get("content", {})
                review_text = _val(review_content.get("review", "")) or _val(
                    review_content.get("comment", "")
                )
                if review_text and len(review_text) > 100:
                    paper_data["reviews"].append(
                        {
                            "id": review_note.get("id", ""),
                            "text": review_text,
                            "rating": _val(review_content.get("rating", "")),
                            "confidence": _val(review_content.get("confidence", "")),
                        }
                    )
        except Exception as e:
            logger.debug(f"Could not fetch reviews for {note_id}: {e}")

        # Only keep papers that have at least one substantive review
        if not paper_data["reviews"]:
            return None

        return paper_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl OpenReview evaluation papers")
    parser.add_argument("--venues", nargs="+", default=["NeurIPS", "ICLR", "ICML"])
    parser.add_argument("--max-papers", type=int, default=50_000)
    parser.add_argument("--since-year", type=int, default=2018)
    parser.add_argument("--output", default="data/raw/openreview")
    args = parser.parse_args()

    crawler = OpenReviewCrawler(output_dir=args.output)
    n = crawler.run(
        venues=args.venues,
        max_papers=args.max_papers,
        since_year=args.since_year,
    )
    logger.info(f"Done. Saved {n:,} papers to {args.output}/")
