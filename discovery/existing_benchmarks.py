"""
discovery/existing_benchmarks.py — Download and catalog existing AI benchmarks.

Downloads benchmark datasets from HuggingFace Hub, analyzes question structure,
difficulty distribution, and category distribution. Builds a catalog of known
benchmark items used for contamination detection during synthesis.

Usage:
    python discovery/existing_benchmarks.py \
        --output data/raw/benchmarks \
        --catalog-output data/raw/contamination_catalog.jsonl
"""

from __future__ import annotations

import hashlib
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

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VLLM_URLS = os.environ.get("VLLM_URLS", "http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004").split(",")

# Benchmarks to download — (hf_dataset_id, subset, split, item_limit)
BENCHMARK_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "mmlu",
        "hf_id": "cais/mmlu",
        "subsets": ["abstract_algebra", "anatomy", "astronomy", "business_ethics",
                    "clinical_knowledge", "college_biology", "college_chemistry",
                    "college_computer_science", "college_mathematics", "college_medicine",
                    "college_physics", "computer_security", "conceptual_physics",
                    "econometrics", "electrical_engineering", "elementary_mathematics",
                    "formal_logic", "global_facts", "high_school_biology",
                    "high_school_chemistry", "high_school_computer_science",
                    "high_school_european_history", "high_school_geography",
                    "high_school_government_and_politics", "high_school_macroeconomics",
                    "high_school_mathematics", "high_school_microeconomics",
                    "high_school_physics", "high_school_psychology",
                    "high_school_statistics", "high_school_us_history",
                    "high_school_world_history", "human_aging", "human_sexuality",
                    "international_law", "jurisprudence", "logical_fallacies",
                    "machine_learning", "management", "marketing", "medical_genetics",
                    "miscellaneous", "moral_disputes", "moral_scenarios",
                    "nutrition", "philosophy", "prehistory", "professional_accounting",
                    "professional_law", "professional_medicine", "professional_psychology",
                    "public_relations", "security_studies", "sociology",
                    "us_foreign_policy", "virology", "world_religions"],
        "split": "test",
        "item_limit": 100,
        "question_field": "question",
        "answer_field": "answer",
        "choices_field": "choices",
        "category": "knowledge",
    },
    {
        "name": "hellaswag",
        "hf_id": "Rowan/hellaswag",
        "subsets": ["default"],
        "split": "validation",
        "item_limit": 2000,
        "question_field": "ctx",
        "answer_field": "label",
        "choices_field": "endings",
        "category": "commonsense",
    },
    {
        "name": "arc_challenge",
        "hf_id": "allenai/ai2_arc",
        "subsets": ["ARC-Challenge"],
        "split": "test",
        "item_limit": 1000,
        "question_field": "question",
        "answer_field": "answerKey",
        "choices_field": "choices",
        "category": "science",
    },
    {
        "name": "arc_easy",
        "hf_id": "allenai/ai2_arc",
        "subsets": ["ARC-Easy"],
        "split": "test",
        "item_limit": 1000,
        "question_field": "question",
        "answer_field": "answerKey",
        "choices_field": "choices",
        "category": "science",
    },
    {
        "name": "winogrande",
        "hf_id": "winogrande",
        "subsets": ["winogrande_xl"],
        "split": "validation",
        "item_limit": 1000,
        "question_field": "sentence",
        "answer_field": "answer",
        "choices_field": None,
        "category": "commonsense",
    },
    {
        "name": "gsm8k",
        "hf_id": "openai/gsm8k",
        "subsets": ["main"],
        "split": "test",
        "item_limit": 1000,
        "question_field": "question",
        "answer_field": "answer",
        "choices_field": None,
        "category": "math",
    },
    {
        "name": "humaneval",
        "hf_id": "openai_humaneval",
        "subsets": ["openai_humaneval"],
        "split": "test",
        "item_limit": 500,
        "question_field": "prompt",
        "answer_field": "canonical_solution",
        "choices_field": None,
        "category": "coding",
    },
    {
        "name": "mbpp",
        "hf_id": "google-research-datasets/mbpp",
        "subsets": ["sanitized"],
        "split": "test",
        "item_limit": 500,
        "question_field": "text",
        "answer_field": "code",
        "choices_field": None,
        "category": "coding",
    },
    {
        "name": "bbh",
        "hf_id": "lukaemon/bbh",
        "subsets": ["boolean_expressions", "causal_judgement", "date_understanding",
                    "disambiguation_qa", "dyck_languages", "formal_fallacies",
                    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
                    "logical_deduction_seven_objects", "logical_deduction_three_objects",
                    "movie_recommendation", "multistep_arithmetic_two",
                    "navigate", "object_counting", "penguins_in_a_table",
                    "reasoning_about_colored_objects", "ruin_names",
                    "salient_translation_error_detection", "snarks",
                    "sports_understanding", "temporal_sequences",
                    "tracking_shuffled_objects_five_objects",
                    "tracking_shuffled_objects_seven_objects",
                    "tracking_shuffled_objects_three_objects",
                    "web_of_lies", "word_sorting"],
        "split": "test",
        "item_limit": 100,
        "question_field": "input",
        "answer_field": "target",
        "choices_field": None,
        "category": "reasoning",
    },
    {
        "name": "math",
        "hf_id": "hendrycks/competition_math",
        "subsets": ["default"],
        "split": "test",
        "item_limit": 2000,
        "question_field": "problem",
        "answer_field": "solution",
        "choices_field": None,
        "category": "math",
    },
]


def _item_fingerprint(question_text: str) -> str:
    """SHA-256 fingerprint of a normalized question string (for contamination lookup)."""
    normalized = re.sub(r"\s+", " ", question_text.lower().strip())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _ngram_fingerprints(text: str, n: int = 8) -> list[str]:
    """Word n-gram fingerprints for fuzzy contamination detection."""
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return [hashlib.md5(ng.encode()).hexdigest() for ng in ngrams]


class BenchmarkDownloader:
    """
    Downloads benchmark datasets from HuggingFace and builds a contamination catalog.

    Output per benchmark:
      data/raw/benchmarks/{name}/items.jsonl    — all downloaded items
      data/raw/benchmarks/{name}/catalog.json   — metadata + stats

    Unified contamination catalog:
      data/raw/contamination_catalog.jsonl      — all fingerprints (exact + fuzzy)
    """

    def __init__(self, output_dir: str | Path, catalog_output: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.catalog_output = Path(catalog_output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_output.parent.mkdir(parents=True, exist_ok=True)

    def run(self, benchmarks: list[str] | None = None) -> dict[str, int]:
        """Download all benchmarks and build contamination catalog. Returns item counts."""
        target = set(benchmarks) if benchmarks else {b["name"] for b in BENCHMARK_REGISTRY}
        results: dict[str, int] = {}
        catalog_entries: list[dict] = []

        for bdef in BENCHMARK_REGISTRY:
            name = bdef["name"]
            if name not in target:
                continue

            logger.info(f"Downloading {name}...")
            try:
                items, catalog = self._download_benchmark(bdef)
                results[name] = len(items)
                catalog_entries.extend(catalog)
                logger.success(f"  {name}: {len(items)} items, {len(catalog)} catalog entries")
            except Exception as exc:
                logger.error(f"  {name}: failed — {exc}")
                results[name] = 0

        # Write unified catalog
        if catalog_entries:
            with self.catalog_output.open("w") as fh:
                for entry in catalog_entries:
                    fh.write(json.dumps(entry) + "\n")
            logger.info(f"Contamination catalog: {len(catalog_entries)} entries → {self.catalog_output}")

        return results

    def _download_benchmark(self, bdef: dict) -> tuple[list[dict], list[dict]]:
        """Download one benchmark. Returns (items, catalog_entries)."""
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise RuntimeError("pip install datasets")

        bench_dir = self.output_dir / bdef["name"]
        bench_dir.mkdir(exist_ok=True)

        all_items: list[dict] = []
        catalog_entries: list[dict] = []
        category_counts: dict[str, int] = {}

        for subset in bdef["subsets"]:
            try:
                ds_kwargs: dict[str, Any] = {
                    "path": bdef["hf_id"],
                    "split": bdef["split"],
                    "trust_remote_code": True,
                }
                if subset != "default":
                    ds_kwargs["name"] = subset

                ds = load_dataset(**ds_kwargs)
                limit = bdef.get("item_limit", 500)

                items_loaded = 0
                for i, raw in enumerate(ds):
                    if i >= limit:
                        break

                    q_field = bdef["question_field"]
                    question_text = str(raw.get(q_field, ""))
                    answer_text = str(raw.get(bdef["answer_field"], ""))

                    # Choices
                    choices_field = bdef.get("choices_field")
                    choices = raw.get(choices_field, []) if choices_field else []

                    # Difficulty heuristic: longer questions tend to be harder
                    word_count = len(question_text.split())
                    if word_count < 20:
                        difficulty = "easy"
                    elif word_count < 60:
                        difficulty = "medium"
                    else:
                        difficulty = "hard"

                    category = bdef["category"]
                    category_counts[category] = category_counts.get(category, 0) + 1

                    item = {
                        "benchmark": bdef["name"],
                        "subset": subset,
                        "index": i,
                        "question": question_text,
                        "answer": answer_text,
                        "choices": choices if isinstance(choices, list) else [],
                        "category": category,
                        "difficulty": difficulty,
                        "word_count": word_count,
                        "fingerprint": _item_fingerprint(question_text),
                    }
                    all_items.append(item)
                    items_loaded += 1

                    # Catalog entry for contamination detection
                    catalog_entries.append({
                        "benchmark": bdef["name"],
                        "subset": subset,
                        "exact_fingerprint": item["fingerprint"],
                        "ngram_fingerprints": _ngram_fingerprints(question_text),
                        "category": category,
                        "question_preview": question_text[:120],
                    })

                logger.debug(f"  {bdef['name']}/{subset}: {items_loaded} items")

            except Exception as exc:
                logger.warning(f"  Could not load {bdef['hf_id']}/{subset}: {exc}")

        # Save items
        if all_items:
            items_path = bench_dir / "items.jsonl"
            with items_path.open("w") as fh:
                for item in all_items:
                    fh.write(json.dumps(item) + "\n")

        # Save catalog metadata
        catalog_meta = {
            "benchmark": bdef["name"],
            "hf_id": bdef["hf_id"],
            "total_items": len(all_items),
            "category": bdef["category"],
            "category_counts": category_counts,
            "difficulty_distribution": _difficulty_distribution(all_items),
            "subsets": bdef["subsets"],
        }
        (bench_dir / "catalog.json").write_text(json.dumps(catalog_meta, indent=2))

        return all_items, catalog_entries


def _difficulty_distribution(items: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for item in items:
        d = item.get("difficulty", "unknown")
        dist[d] = dist.get(d, 0) + 1
    return dist


class ContaminationChecker:
    """
    Checks whether a candidate question is likely contaminated by known benchmarks.

    Usage:
        checker = ContaminationChecker("data/raw/contamination_catalog.jsonl")
        result = checker.check("What is the capital of France?")
        if result.is_contaminated:
            print(f"Matches {result.benchmark} ({result.match_type})")
    """

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self._exact: dict[str, str] = {}   # fingerprint → benchmark
        self._ngrams: dict[str, str] = {}  # ngram_fp → benchmark
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.catalog_path.exists():
            logger.warning(f"Catalog not found: {self.catalog_path} — contamination checking disabled")
            self._loaded = True
            return

        with self.catalog_path.open() as fh:
            for line in fh:
                entry = json.loads(line.strip())
                bench = entry["benchmark"]
                self._exact[entry["exact_fingerprint"]] = bench
                for ng_fp in entry.get("ngram_fingerprints", []):
                    self._ngrams[ng_fp] = bench

        self._loaded = True
        logger.info(f"Loaded contamination catalog: {len(self._exact)} exact, {len(self._ngrams)} ngram entries")

    def check(self, question_text: str) -> dict[str, Any]:
        """
        Returns dict with keys: is_contaminated, benchmark, match_type, confidence.
        match_type is 'exact' or 'fuzzy' or None.
        """
        self._load()

        exact_fp = _item_fingerprint(question_text)
        if exact_fp in self._exact:
            return {
                "is_contaminated": True,
                "benchmark": self._exact[exact_fp],
                "match_type": "exact",
                "confidence": 1.0,
            }

        ngram_fps = _ngram_fingerprints(question_text)
        if ngram_fps:
            match_counts: dict[str, int] = {}
            for fp in ngram_fps:
                if fp in self._ngrams:
                    b = self._ngrams[fp]
                    match_counts[b] = match_counts.get(b, 0) + 1

            if match_counts:
                best_bench = max(match_counts, key=lambda k: match_counts[k])
                hit_ratio = match_counts[best_bench] / max(len(ngram_fps), 1)
                if hit_ratio >= 0.3:
                    return {
                        "is_contaminated": True,
                        "benchmark": best_bench,
                        "match_type": "fuzzy",
                        "confidence": round(hit_ratio, 3),
                    }

        return {"is_contaminated": False, "benchmark": None, "match_type": None, "confidence": 0.0}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _list_hf_benchmark_datasets(
    page: int = 0,
    page_size: int = 100,
) -> list[dict]:
    """List datasets tagged 'benchmark' on HuggingFace Hub."""
    url = "https://huggingface.co/api/datasets"
    params = {
        "filter": "benchmark",
        "sort": "downloads",
        "direction": -1,
        "limit": page_size,
        "offset": page * page_size,
        "full": False,
    }
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def discover_hf_benchmarks(output_path: str | Path, max_pages: int = 5) -> int:
    """
    Discover additional benchmarks on HuggingFace Hub tagged as 'benchmark'.
    Saves a registry to output_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_datasets: list[dict] = []
    for page in range(max_pages):
        try:
            datasets = _list_hf_benchmark_datasets(page=page)
            if not datasets:
                break
            all_datasets.extend(datasets)
            logger.debug(f"  HF Hub page {page}: {len(datasets)} datasets")
            time.sleep(0.5)
        except Exception as exc:
            logger.warning(f"  HF Hub page {page} failed: {exc}")
            break

    registry = [
        {
            "hf_id": d.get("id", ""),
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "tags": d.get("tags", []),
        }
        for d in all_datasets
    ]
    output_path.write_text(json.dumps(registry, indent=2))
    logger.info(f"Discovered {len(registry)} benchmark datasets on HuggingFace Hub → {output_path}")
    return len(registry)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download benchmark corpora for EvalForge")
    parser.add_argument("--output", default="data/raw/benchmarks", help="Output directory")
    parser.add_argument(
        "--catalog-output",
        default="data/raw/contamination_catalog.jsonl",
        help="Path for contamination catalog",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Specific benchmarks to download (default: all)",
    )
    parser.add_argument(
        "--discover-hf",
        action="store_true",
        help="Also discover additional benchmark datasets on HuggingFace Hub",
    )
    args = parser.parse_args()

    if args.discover_hf:
        discover_hf_benchmarks(Path(args.output) / "hf_registry.json")

    downloader = BenchmarkDownloader(
        output_dir=args.output,
        catalog_output=args.catalog_output,
    )
    results = downloader.run(benchmarks=args.benchmarks)

    logger.info("=== Download Summary ===")
    total = 0
    for name, count in sorted(results.items()):
        logger.info(f"  {name:<25} {count:>5} items")
        total += count
    logger.info(f"  {'TOTAL':<25} {total:>5} items")
    logger.info(f"Catalog: {args.catalog_output}")
