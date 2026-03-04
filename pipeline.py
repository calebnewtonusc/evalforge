"""
pipeline.py — EvalForge end-to-end orchestration.

Stages:
  discovery  — crawl OpenReview papers + benchmark corpora
  synthesis  — generate training pairs (audit, shortcut, IRT)
  train      — 3-stage training (SFT → GRPO → DPO)
  eval       — ForgeQualityBench evaluation

Usage:
    python pipeline.py                          # full run
    python pipeline.py --stage discovery        # crawl only
    python pipeline.py --stage synthesis        # synthesis only
    python pipeline.py --stage train            # training only (SFT + RL + DPO)
    python pipeline.py --stage eval             # evaluation only
    python pipeline.py --stats                  # dataset statistics
    python pipeline.py --stage synthesis --backend claude  # use Claude API instead of vLLM
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DIR = DATA_DIR / "train"
CHECKPOINTS_DIR = ROOT / "checkpoints"

for d in [RAW_DIR, PROCESSED_DIR, TRAIN_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def stage_discovery(args: argparse.Namespace) -> None:
    """Crawl OpenReview papers and benchmark corpora."""
    logger.info("=== STAGE: DISCOVERY ===")

    logger.info("Crawling OpenReview evaluation papers...")
    from discovery.openreview_crawler import OpenReviewCrawler

    crawler = OpenReviewCrawler(output_dir=RAW_DIR / "openreview")
    n_papers = crawler.run(
        venues=["NeurIPS", "ICLR", "ICML", "ACL", "EMNLP"],
        query_terms=[
            "benchmark",
            "evaluation",
            "contamination",
            "shortcut",
            "construct validity",
        ],
        max_papers=50_000,
        since_year=2018,
    )
    logger.info(f"Collected {n_papers:,} papers from OpenReview")

    logger.info("Indexing benchmark corpora...")
    from discovery.benchmark_corpus import BenchmarkCorpusIndexer

    indexer = BenchmarkCorpusIndexer(output_dir=RAW_DIR / "benchmarks")
    indexer.run(
        benchmarks=[
            "bigbench",
            "helm",
            "mmlu",
            "superglue",
            "gsm8k",
            "humaneval",
            "math",
        ]
    )
    logger.info("Benchmark corpus indexed")


def stage_synthesis(args: argparse.Namespace) -> None:
    """Generate training pairs from crawled data."""
    logger.info("=== STAGE: SYNTHESIS ===")

    backend = getattr(args, "backend", "vllm")
    vllm_urls = None
    if backend == "vllm":
        import os

        urls_str = os.environ.get(
            "VLLM_URLS", "http://localhost:8001,http://localhost:8002"
        )
        vllm_urls = [u.strip() for u in urls_str.split(",")]
        logger.info(f"Using vLLM backend: {vllm_urls}")
    else:
        logger.info("Using Claude API backend (slower, no GPU required)")

    from synthesis.synthesize_bulk import BulkSynthesizer

    synthesizer = BulkSynthesizer(
        raw_dir=RAW_DIR,
        output_dir=PROCESSED_DIR,
        backend=backend,
        vllm_urls=vllm_urls,
        workers=30,
    )
    stats = synthesizer.run()
    logger.info(f"Synthesis complete: {stats['total_pairs']:,} pairs generated")
    logger.info(f"  audit pairs:      {stats.get('audit_pairs', 0):,}")
    logger.info(f"  shortcut pairs:   {stats.get('shortcut_pairs', 0):,}")
    logger.info(f"  irt pairs:        {stats.get('irt_pairs', 0):,}")
    logger.info(f"  goodhart pairs:   {stats.get('goodhart_pairs', 0):,}")
    logger.info(f"  item gen pairs:   {stats.get('item_gen_pairs', 0):,}")

    # Merge, dedup, split
    logger.info("Merging and splitting dataset...")
    _merge_and_split()


def _merge_and_split() -> None:
    """Merge all processed JSONL files, deduplicate, and create train/val/test splits."""
    try:
        from datasketch import MinHash, MinHashLSH

        HAS_DATASKETCH = True
    except ImportError:
        HAS_DATASKETCH = False
    import random

    all_pairs: list[dict] = []
    for f in sorted(PROCESSED_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    all_pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    logger.info(f"Total pairs before dedup: {len(all_pairs):,}")

    if HAS_DATASKETCH:
        # MinHash LSH dedup at 0.9 similarity
        lsh = MinHashLSH(threshold=0.9, num_perm=128)
        deduped: list[dict] = []
        for i, pair in enumerate(all_pairs):
            text = json.dumps(pair, sort_keys=True)
            m = MinHash(num_perm=128)
            for word in text.split():
                m.update(word.encode())
            key = f"pair_{i}"
            if not lsh.query(m):
                lsh.insert(key, m)
                deduped.append(pair)
    else:
        logger.warning(
            "datasketch not available — skipping MinHash dedup, using all pairs"
        )
        deduped = all_pairs

    logger.info(
        f"Pairs after dedup: {len(deduped):,} ({len(all_pairs) - len(deduped):,} removed)"
    )

    # Shuffle and split 90/5/5 with a fixed seed for reproducibility.
    random.seed(42)
    random.shuffle(deduped)
    n = len(deduped)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)

    if n_val == 0 and n > 0:
        logger.warning(
            f"Dataset has only {n} pairs; the 5% val split is 0 examples. "
            "Eval will be disabled — consider using a larger dataset."
        )

    splits = {
        "train": deduped[:n_train],
        "val": deduped[n_train : n_train + n_val],
        "test": deduped[n_train + n_val :],
    }

    for split_name, pairs in splits.items():
        out_path = TRAIN_DIR / f"evalforge_{split_name}.jsonl"
        if not pairs:
            out_path.write_text("")
        else:
            out_path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
        logger.info(f"  {split_name}: {len(pairs):,} pairs → {out_path}")


def stage_train(args: argparse.Namespace) -> None:
    """Run 3-stage training: SFT → GRPO → DPO."""
    logger.info("=== STAGE: TRAINING ===")

    sft_checkpoint = CHECKPOINTS_DIR / "evalforge-sft"
    # The SFT trainer merges LoRA weights and saves the final model under
    # sft_checkpoint/final — use that path for existence checks and for
    # passing to downstream stages.
    sft_final = sft_checkpoint / "final"
    rl_checkpoint = CHECKPOINTS_DIR / "evalforge-rl"
    final_checkpoint = CHECKPOINTS_DIR / "evalforge-final"

    # Stage 1: SFT
    if not (sft_final / "config.json").exists():
        logger.info("--- Stage 1: SFT ---")
        _run_deepspeed(
            script="training/train.py",
            extra_args=[
                "--model",
                "Qwen/Qwen2.5-7B-Coder-Instruct",
                "--data-dir",
                str(TRAIN_DIR),
                "--output-dir",
                str(sft_checkpoint),
                "--epochs",
                "3",
                "--batch-size",
                "4",
                "--grad-accum",
                "4",
                "--lr",
                "2e-4",
                "--lora-r",
                "64",
                "--max-length",
                "4096",
                "--deepspeed",
                "training/configs/deepspeed_zero3.json",
            ],
        )
    else:
        logger.info(f"SFT checkpoint found at {sft_final}, skipping Stage 1")

    # Stage 2: GRPO
    if not (rl_checkpoint / "config.json").exists():
        logger.info("--- Stage 2: GRPO ---")
        _run_deepspeed(
            script="training/train_rl.py",
            extra_args=[
                "--model",
                str(sft_final),
                "--data-dir",
                str(TRAIN_DIR),
                "--output-dir",
                str(rl_checkpoint),
                "--deepspeed",
                "training/configs/rl_config.yaml",
            ],
        )
    else:
        logger.info(f"RL checkpoint found at {rl_checkpoint}, skipping Stage 2")

    # Stage 3: DPO
    if not (final_checkpoint / "config.json").exists():
        logger.info("--- Stage 3: DPO ---")
        _run_deepspeed(
            script="training/train_dpo.py",
            extra_args=[
                "--model",
                str(rl_checkpoint),
                "--dpo-data",
                str(TRAIN_DIR / "dpo_pairs.jsonl"),
                "--output-dir",
                str(final_checkpoint),
                "--deepspeed",
                "training/configs/deepspeed_zero3.json",
            ],
        )
    else:
        logger.info(f"Final checkpoint found at {final_checkpoint}, skipping Stage 3")

    logger.info(f"Training complete. Final model: {final_checkpoint}")


def _run_deepspeed(script: str, extra_args: list[str]) -> None:
    """Launch a DeepSpeed training job."""
    import os
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        n_gpus = len(visible.split(","))
    else:
        n_gpus = torch.cuda.device_count() or 1
    cmd = ["deepspeed", f"--num_gpus={n_gpus}", script] + extra_args
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def stage_eval(args: argparse.Namespace) -> None:
    """Run ForgeQualityBench evaluation."""
    logger.info("=== STAGE: EVALUATION ===")
    final_checkpoint = CHECKPOINTS_DIR / "evalforge-final"
    if not final_checkpoint.exists():
        logger.error(f"No final checkpoint at {final_checkpoint}. Run training first.")
        sys.exit(1)

    from evaluation.forgequality_bench import ForgeQualityBench

    bench = ForgeQualityBench(model_path=str(final_checkpoint))
    results = bench.run_all()

    logger.info("=== ForgeQualityBench Results ===")
    for metric, value in results.items():
        logger.info(f"  {metric:<45} {value:.4f}")

    results_path = ROOT / "results" / "forgequality_bench_results.json"
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {results_path}")


def print_stats() -> None:
    """Print dataset statistics."""
    logger.info("=== DATASET STATS ===")

    raw_papers = (
        list((RAW_DIR / "openreview").glob("*.json"))
        if (RAW_DIR / "openreview").exists()
        else []
    )
    logger.info(f"Raw papers (OpenReview):    {len(raw_papers):,}")

    processed_files = list(PROCESSED_DIR.glob("*.jsonl"))
    total_pairs = 0
    for f in processed_files:
        lines = [line for line in f.read_text().splitlines() if line.strip()]
        total_pairs += len(lines)
    logger.info(f"Processed JSONL files:      {len(processed_files)}")
    logger.info(f"Total pairs (pre-dedup):    {total_pairs:,}")

    for split in ["train", "val", "test"]:
        p = TRAIN_DIR / f"evalforge_{split}.jsonl"
        if p.exists():
            n = len([line for line in p.read_text().splitlines() if line.strip()])
            logger.info(f"  {split} split:             {n:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvalForge — evaluation design pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["discovery", "synthesis", "train", "eval"],
        help="Run a specific pipeline stage (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "claude"],
        default="vllm",
        help="Synthesis backend (default: vllm)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print dataset statistics and exit"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run stages even if outputs exist"
    )
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.stage is None:
        # Full pipeline
        stage_discovery(args)
        stage_synthesis(args)
        stage_train(args)
        stage_eval(args)
    elif args.stage == "discovery":
        stage_discovery(args)
    elif args.stage == "synthesis":
        stage_synthesis(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "eval":
        stage_eval(args)


if __name__ == "__main__":
    main()
