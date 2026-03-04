#!/bin/bash
# EvalForge — Full pipeline: discovery → synthesis → train SFT → train RL → train DPO
# Runtime: ~35 hours on 18× A6000
#
# Resume from a stage: FROM_STAGE=3 ./scripts/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

FROM_STAGE="${FROM_STAGE:-1}"

# Load environment
if [ -f .env ]; then
	set -a
	source .env
	set +a
else
	echo "ERROR: .env file not found. Copy .env.example and fill in your keys."
	exit 1
fi

echo "=== EvalForge Full Pipeline ==="
echo "Started: $(date)"
echo "Resuming from stage: $FROM_STAGE"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 DISCOVER — Crawl benchmarks, academic papers, question templates
# ─────────────────────────────────────────────────────────────────────────────
if [ "$FROM_STAGE" -le 1 ]; then
	echo "━━━ STEP 1 DISCOVER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	echo "  [1/5] Downloading benchmark corpora and building contamination catalog..."
	python discovery/existing_benchmarks.py \
		--output data/raw/benchmarks \
		--catalog-output data/raw/contamination_catalog.jsonl \
		--discover-hf

	echo "  [2/5] Crawling academic papers on evaluation and contamination..."
	python discovery/academic_papers.py \
		--output data/raw/papers \
		--max-papers 5000

	echo "  [3/5] Collecting question templates from AoPS, Euler, Rosetta, HackerRank..."
	python discovery/question_templates.py \
		--output data/raw/question_templates \
		--sources aops euler rosetta hackerrank \
		--max-per-source 500

	echo "  [4/5] Indexing existing benchmark corpus (BigBench, HELM metadata)..."
	python discovery/benchmark_corpus.py \
		--benchmarks mmlu bigbench superglue gsm8k \
		--output data/raw/benchmark_meta

	echo "  [5/5] Crawling OpenReview papers..."
	python discovery/openreview_crawler.py \
		--output data/raw/openreview \
		2>/dev/null || echo "    [WARN] OpenReview crawler failed — skipping"

	echo ""
	echo "  STEP 1 DISCOVER complete: $(date)"
	echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 SYNTHESIZE — Generate contamination-resistant benchmark questions
# ─────────────────────────────────────────────────────────────────────────────
if [ "$FROM_STAGE" -le 2 ]; then
	echo "━━━ STEP 2 SYNTHESIZE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

	echo "  Starting vLLM synthesis cluster (4 instances, GPUs 0-15)..."
	bash scripts/start_vllm.sh
	export VLLM_URLS="http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004"

	echo "  Generating benchmark questions (target: 50,000)..."
	python synthesis/benchmark_generator.py \
		--templates data/raw/question_templates/all_templates.jsonl \
		--catalog data/raw/contamination_catalog.jsonl \
		--output data/synthesized/benchmark_questions.jsonl \
		--count 50000 \
		--backend vllm

	echo "  Running bulk synthesis (EvalForge training pairs)..."
	python synthesis/synthesize_bulk.py \
		--backend vllm \
		--output data/train/evalforge_train.jsonl

	echo "  Killing vLLM synthesis cluster..."
	pkill -f "vllm serve" 2>/dev/null || pkill -f "vllm.entrypoints" 2>/dev/null || true

	echo ""
	echo "  STEP 2 SYNTHESIZE complete: $(date)"
	echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 TRAIN SFT — Supervised fine-tuning on benchmark generation corpus
# ─────────────────────────────────────────────────────────────────────────────
if [ "$FROM_STAGE" -le 3 ]; then
	echo "━━━ STEP 3 TRAIN SFT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
		deepspeed --num_gpus=18 training/train.py \
		--config training/configs/sft_config.yaml \
		--deepspeed training/configs/deepspeed_zero3.json

	echo ""
	echo "  STEP 3 TRAIN SFT complete: $(date)"
	echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 TRAIN RL — GRPO with shortcut-detection reward
# ─────────────────────────────────────────────────────────────────────────────
if [ "$FROM_STAGE" -le 4 ]; then
	echo "━━━ STEP 4 TRAIN RL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
		deepspeed --num_gpus=18 training/train_rl.py \
		--config training/configs/rl_config.yaml \
		--deepspeed training/configs/deepspeed_zero3.json

	echo ""
	echo "  STEP 4 TRAIN RL complete: $(date)"
	echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 TRAIN DPO — Direct preference optimization
# ─────────────────────────────────────────────────────────────────────────────
if [ "$FROM_STAGE" -le 5 ]; then
	echo "━━━ STEP 5 TRAIN DPO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
		deepspeed --num_gpus=18 training/train_dpo.py \
		--config training/configs/dpo_config.yaml \
		--deepspeed training/configs/deepspeed_zero3.json

	echo ""
	echo "  STEP 5 TRAIN DPO complete: $(date)"
	echo ""
fi

echo "=== Pipeline complete: $(date) ==="
echo "Final model: checkpoints/evalforge-final/"
echo "Results: results/forgequality_bench_results.json"
