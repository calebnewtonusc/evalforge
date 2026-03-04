# EvalForge — Evaluations that models can't game.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: Qwen2.5-7B-Coder](https://img.shields.io/badge/base_model-Qwen2.5--7B--Coder-purple.svg)](https://huggingface.co/Qwen)
[![GPUs: 18x A6000](https://img.shields.io/badge/training-18×_A6000-red.svg)](https://www.nvidia.com)
[![Stage: Active Build](https://img.shields.io/badge/stage-active_build-orange.svg)]()

> **"Evaluations that models can't game."**

EvalForge is the first AI trained on *evaluation quality* as a first-class objective. It does not run benchmarks — it *designs* them. Given a saturated benchmark, EvalForge detects the exact memorization shortcuts and contamination patterns making it non-discriminative, then generates contamination-resistant replacement items with provably higher construct validity.

The core insight: every existing benchmark eventually dies by Goodhart's Law. EvalForge is the antidote — a continuous evaluation design engine trained on the full literature of psychometrics, benchmark critique, contamination analysis, and shortcut detection.

---

## What Makes EvalForge Different

| Capability | HELM | BIG-Bench | MMLU | LM-Eval Harness | **EvalForge** |
|---|---|---|---|---|---|
| Runs existing benchmarks | Yes | Yes | Yes | Yes | **Yes (also)** |
| Detects contamination in a benchmark | — | — | — | — | **Yes — item-level** |
| Detects shortcut exploitation | — | — | — | — | **Yes — per-pattern taxonomy** |
| Generates new eval items | — | Partial | — | — | **Yes — adversarially robust** |
| Tracks real-world correlation | — | — | — | — | **Yes — Pearson + Spearman over time** |
| IRT-calibrated item difficulty | — | — | — | — | **Yes — 2PL and 3PL models** |
| Annotation artifact detection | — | — | — | — | **Yes — Hans patterns, length bias, lexical overlap** |
| Goodhart pattern taxonomy | — | — | — | — | **Yes — 23 failure modes catalogued** |
| Trained reward signal | Rules-based | Rules-based | Rules-based | Rules-based | **Correlation stability + shortcut detection** |

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
  Benchmark Input ─►│              EvalForge Model                    │
  (items + scores)  │  (Qwen2.5-7B-Coder + LoRA, 3-stage trained)    │
                    └──────────────────┬──────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────────┐
                    │           Contamination Prober                   │
                    │  Detects: n-gram overlap, training data leakage, │
                    │  template memorization, answer-set distribution  │
                    └──────────────────┬──────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────────┐
                    │            Shortcut Detector                     │
                    │  Patterns: length bias, lexical overlap, negation│
                    │  artifacts, annotation agreement shortcuts, etc  │
                    └──────────────────┬──────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────────┐
                    │            IRT Calibrator                        │
                    │  2PL/3PL Item Response Theory — discrimination,  │
                    │  difficulty, guessing parameter estimation       │
                    └──────────────────┬──────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────────┐
                    │           Item Generator                         │
                    │  Produces new items: removes detected shortcuts, │
                    │  targets underrepresented construct facets,      │
                    │  maintains calibrated difficulty distribution    │
                    └──────────────────┬──────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────────┐
                    │         Correlation Tracker                      │
                    │  Tracks benchmark score vs. downstream perf over │
                    │  time — Pearson r, Spearman rho, drift alerts    │
                    └─────────────────────────────────────────────────┘

Training data streams (5 streams, 400k+ pairs):
  Stream 1: OpenReview eval methodology critique papers + reviews (30%)
  Stream 2: BIG-Bench/HELM/MMLU/SuperGLUE design documents + critique (20%)
  Stream 3: Contamination analysis studies + shortcut detection research (25%)
  Stream 4: Psychometrics literature — IRT, construct validity, test design (15%)
  Stream 5: Annotation artifact studies + Goodhart's Law failure cases (10%)
```

---

## Quick Start

```bash
git clone https://github.com/calebnewtonusc/evalforge
cd evalforge
pip install -r requirements.txt
cp .env.example .env  # Fill in your API keys

# Validate environment
bash scripts/check_env.sh

# Run full pipeline (data → training → eval), ~28 hours on 18× A6000
bash scripts/run_all.sh

# Or step by step:
python pipeline.py --stage discovery    # ~8h, crawl OpenReview + benchmark corpora
python pipeline.py --stage synthesis    # ~12h, generate training pairs
python pipeline.py --stage train        # ~8h, 3-stage training (SFT 4h + GRPO 3h + DPO 1h)
python pipeline.py --stage eval         # ~2h, ForgeQualityBench evaluation
```

---

## Use EvalForge on Your Benchmark

```python
from evalforge import EvalForgeClient

client = EvalForgeClient(model_url="http://localhost:9000")

# Audit an existing benchmark for gaming vulnerabilities
report = client.audit_benchmark(
    benchmark_path="data/my_benchmark.jsonl",
    model_scores={"gpt-4": 0.92, "llama-70b": 0.87, "qwen-7b": 0.76},
)

print(report.contamination_score)   # 0.0–1.0
print(report.shortcuts_detected)    # list of Goodhart patterns found
print(report.irt_analysis)          # item discrimination, difficulty params
print(report.downstream_correlation)  # Pearson r vs downstream task set

# Generate replacement items that remove detected shortcuts
new_items = client.forge_items(
    existing_items=report.flagged_items,
    target_difficulty_range=(0.3, 0.7),
    avoid_shortcuts=report.shortcuts_detected,
    n_items=100,
)
```

---

## ForgeQualityBench

ForgeQualityBench is our meta-evaluation: a benchmark for evaluating benchmark quality. It measures:

- **Shortcut detection recall** — what fraction of planted shortcuts does EvalForge catch?
- **Contamination precision** — when EvalForge flags contamination, how often is it real?
- **Downstream correlation** — do EvalForge-designed items predict real capability better than MMLU?
- **Stability over time** — does benchmark ranking stay stable across model releases?
- **Item diversity** — are generated items lexically and structurally distinct from training data?

```bash
python evaluation/forgequality_bench.py --model checkpoints/evalforge-final
```

---

## Performance Targets (v1)

| Metric | Target | MMLU baseline | HELM baseline |
|--------|--------|---------------|---------------|
| Shortcut detection recall | >85% | — | — |
| Contamination precision | >90% | — | — |
| Downstream Pearson r | >0.75 | 0.42 | 0.51 |
| Ranking stability (Kendall tau) | >0.80 | 0.61 | 0.68 |
| Item generation diversity (self-BLEU) | >0.65 | — | — |

---

## Hardware Requirements

### Data Collection
Any CPU machine — OpenReview and benchmark corpus crawling is I/O bound. Expect 4–8 hours for full corpus (100k+ papers + metadata).

### Synthesis (vLLM)
Qwen2.5-72B synthesis requires 4× A6000 per instance. With 2 synthesis instances on 8 cards, generates ~10k pairs/hour.

### Training
| Resource | Specification |
|---|---|
| GPUs | 18× NVIDIA A6000 (48GB each) |
| Total VRAM | 864GB |
| Strategy | DeepSpeed ZeRO-3 + CPU optimizer offload |
| RAM | 512GB+ |
| Expected time | 4–8 hours (SFT on 400k pairs, 3 epochs) |

### Inference
| Config | Latency | Throughput |
|---|---|---|
| 2× A100 80GB | <100ms | 40 req/s |
| 1× A6000 48GB | ~200ms | 18 req/s |
| 1× RTX 4090 | ~600ms | 6 req/s |

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full system architecture, 7 differentiators
- [DATA_SOURCES.md](DATA_SOURCES.md) — 5 training streams with sources and volumes
- [MODEL_CARD.md](MODEL_CARD.md) — Model specification, capabilities, limitations
- [ROADMAP.md](ROADMAP.md) — v1 through v3 roadmap
- [SETUP_GPU.md](SETUP_GPU.md) — 18× A6000 cluster configuration

---

## Citation

```bibtex
@inproceedings{newton2026evalforge,
  title     = {EvalForge: Training a Model to Design Contamination-Resistant Evaluations},
  author    = {Newton, Caleb and others},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026},
}
```

---

## License

Code: MIT License. Model weights: Apache 2.0. Training data: CC BY 4.0 for community contributions.

*Target: 864GB VRAM, 400k+ training pairs. Training in progress — USC IYA Innovation Quest 2026.*
