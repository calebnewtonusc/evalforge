# EvalForge — Full System Architecture
## "Evaluations that models can't game."

---

## THE CORE INSIGHT

Every benchmark is a proxy for capability. Every proxy degrades over time — models learn to exploit the proxy without gaining the underlying capability. This is Goodhart's Law applied to AI evaluation, and it has made the entire evaluation field unreliable.

Existing solutions are reactive: when contamination is detected, humans manually update benchmarks. This is too slow. By the time a benchmark is patched, the next generation of models has already been trained on it.

EvalForge is the first *proactive* system: a model trained to continuously generate benchmarks that are robust against the known taxonomy of gaming strategies. It doesn't react to contamination — it architects evaluations that resist contamination structurally.

```
Phase 1 (v1):   AUDIT            detect shortcuts, contamination, IRT calibration   ← CURRENT BUILD
Phase 2 (v1.5): FORGE            generate replacement items, populate gap regions
Phase 3 (v2):   TRACK            longitudinal correlation monitoring, drift alerts
Phase 4 (v3):   CERTIFY          machine-readable evaluation certificates for models
```

---

## 7 TECHNICAL DIFFERENTIATORS

### 1. Evaluation Quality as a Verifiable Reward Signal

EvalForge's RL training reward is not human preference — it is *verifiable measurement quality*. The reward function has three components:

```
R_total = w1 * R_correlation + w2 * R_shortcut + w3 * R_stability

where:
  R_correlation = Pearson r between forge-designed benchmark score and downstream task perf
  R_shortcut    = 1 - fraction of planted shortcuts undetected (planted = ground truth)
  R_stability   = Kendall tau of model rankings before/after benchmark replacement
```

This is a free, ground-truth signal that does not require human labelers. Ground-truth shortcuts are planted during synthesis; detection is binary. Downstream task correlation is computed against a held-out real-world task set.

### 2. The Goodhart Pattern Taxonomy

EvalForge is trained on an explicit taxonomy of 23 benchmark gaming strategies, catalogued in `core/goodhart_patterns.py`. This taxonomy is derived from the evaluation critique literature and covers:

- **Memorization shortcuts**: n-gram overlap with training data, template fill-in
- **Length bias**: models prefer longer / shorter answer options regardless of content
- **Lexical overlap**: entailment resolved by keyword matching instead of reasoning
- **Negation artifacts**: presence/absence of "not" as a sufficient signal
- **Annotation agreement shortcuts**: items where majority-vote annotation is itself a bias
- **Format artifacts**: answer identifiers (A/B/C/D position) as signals
- **Semantic collapse**: multiple choices with near-identical meaning
- **Distractor quality failure**: distractors so implausible they're trivially rejected

### 3. IRT-Calibrated Item Generation

EvalForge uses Item Response Theory (2-parameter and 3-parameter logistic models) to characterize every eval item along three axes:

- **Discrimination (a)**: does the item separate capable from incapable models?
- **Difficulty (b)**: what model ability level does this item target?
- **Guessing (c)**: is random guessing a viable strategy on this item?

Items with low discrimination (a < 0.3) are flagged as non-informative. Items with difficulty outside [−3σ, +3σ] of the target distribution are discarded. The item generator is constrained to produce items that fill discrimination-difficulty gaps in the existing test blueprint.

```python
# From core/irt_models.py
def p_correct(theta: float, a: float, b: float, c: float = 0.0) -> float:
    """3PL IRT probability of correct response."""
    return c + (1 - c) / (1 + math.exp(-a * (theta - b)))
```

### 4. Contamination Detection at the Item Level

Most contamination detection works at the dataset level (did any training set item overlap with this benchmark?). EvalForge works at the *item level*: for each benchmark item, it produces a contamination confidence score and identifies the specific training data source most likely responsible.

Detection methods:
- **N-gram overlap**: sliding window n-gram comparison against training corpus index
- **Embedding proximity**: cosine similarity in embedding space to identify paraphrased contamination
- **Answer distribution shift**: if model accuracy on an item is 40+ points above its expected difficulty, flag for review
- **Template detection**: identifies fill-in-the-blank memorization via structural similarity

### 5. Downstream Correlation Tracking

A benchmark's purpose is to predict real-world capability. EvalForge tracks whether it succeeds at this purpose by computing correlation between benchmark rankings and a held-out downstream task set across model releases.

```
Correlation signal:
  - Pearson r: linear correlation of normalized scores
  - Spearman rho: rank correlation (more robust to outliers)
  - Kendall tau: pairwise rank agreement
  - Temporal drift: how correlation changes across model generations

Downstream task set (20 tasks, held out from training):
  - Code generation (HumanEval variants)
  - Mathematical reasoning (GSM8K variants)
  - Scientific QA (real exam questions, not MMLU)
  - Long-context retrieval
  - Instruction following (real user queries)
```

### 6. Cross-Benchmark Diagnostic Reports

EvalForge produces structured audit reports for any benchmark, enabling AI teams to understand exactly why their benchmark has saturated:

```json
{
  "benchmark": "MMLU-Pro",
  "audit_date": "2026-03-01",
  "contamination_score": 0.34,
  "shortcuts_detected": [
    {"pattern": "length_bias", "severity": 0.72, "affected_fraction": 0.41},
    {"pattern": "lexical_overlap", "severity": 0.58, "affected_fraction": 0.29},
    {"pattern": "negation_artifact", "severity": 0.44, "affected_fraction": 0.18}
  ],
  "irt_analysis": {
    "low_discrimination_fraction": 0.23,
    "difficulty_distribution_skew": -0.8,
    "floor_items": 0.12,
    "ceiling_items": 0.07
  },
  "downstream_correlation": {
    "pearson_r": 0.42,
    "spearman_rho": 0.39,
    "trend": "declining",
    "last_5_model_generations_tau": 0.31
  },
  "recommendation": "Replace 41% of items. Priority: length-biased items in STEM categories."
}
```

### 7. ForgeQualityBench — Measuring the Measurer

The only way to know if an evaluation designer works is to evaluate it. ForgeQualityBench provides:

- Ground-truth shortcut detection (planted shortcuts with known patterns)
- Contamination recall/precision against known-contaminated items
- Downstream correlation improvement after forge-redesigned benchmarks
- Longitudinal stability measurement across 50+ model releases

---

## 3-STAGE TRAINING PIPELINE

### Stage 1 — Supervised Fine-Tuning (SFT): `training/train.py`

Teaches the model what evaluation quality analysis looks like via expert demonstration.

```
Base model: Qwen/Qwen2.5-7B-Coder-Instruct
Data: 400k+ pairs across 5 streams (see DATA_SOURCES.md)
Training: DeepSpeed ZeRO-3, LoRA rank 64, alpha 128, 3 epochs
Hardware: 18× A6000, estimated 4–5 hours

Input format:
  {"role": "user", "content": "Audit this benchmark item for Goodhart patterns:\n[item JSON]"}
  {"role": "assistant", "content": "[structured audit report JSON]"}

Output: checkpoints/evalforge-sft/
```

### Stage 2 — GRPO Reinforcement Learning: `training/train_rl.py`

Optimizes the model on the verifiable reward signal: shortcut detection + downstream correlation.

```
Algorithm: GRPO (Group Relative Policy Optimization)
Reward: R_correlation (0.4) + R_shortcut (0.4) + R_stability (0.2)
Reference model: evalforge-sft checkpoint
KL penalty coefficient: 0.01
Mini-batch size: 8 completions per prompt
Hardware: 12× A6000 (training) + 6× A6000 (reward eval)

Reward computation:
  1. Generate benchmark from prompt
  2. Run planted-shortcut test suite → R_shortcut
  3. Evaluate on downstream task set → R_correlation
  4. Score ranking stability vs. reference ranker → R_stability

Output: checkpoints/evalforge-rl/
```

### Stage 3 — DPO Preference Optimization: `training/train_dpo.py`

Fine-tunes on human expert preferences for audit report quality and item diversity.

```
Data: 20k DPO pairs (chosen/rejected audit reports + item sets)
Beta: 0.1
Training: 1 epoch, 1× A6000 per GPU, 18 GPUs

Preferred behaviors:
  - Specific, actionable shortcut descriptions (not generic)
  - IRT-grounded difficulty estimates (not vibes)
  - Diverse generated items (not paraphrases of same template)
  - Conservative contamination flags (high precision over high recall)

Output: checkpoints/evalforge-final/
```

---

## SYSTEM COMPONENTS

### Core Library (`core/`)

```
core/
├── goodhart_patterns.py    # Taxonomy of 23 benchmark gaming strategies
├── irt_models.py           # 2PL/3PL IRT, ability estimation, item calibration
├── audit_report.py         # Structured audit report dataclass + serialization
└── correlation_metrics.py  # Pearson, Spearman, Kendall tau + temporal drift
```

### Agents (`agents/`)

```
agents/
├── eval_designer_agent.py      # Orchestrates full evaluation design workflow
├── contamination_agent.py      # Item-level contamination detection
└── correlation_tracker_agent.py # Longitudinal benchmark quality monitoring
```

### Discovery (`discovery/`)

```
discovery/
├── openreview_crawler.py   # Fetch NeurIPS/ICLR/ICML evaluation papers + reviews
└── benchmark_corpus.py     # BIG-Bench, HELM, MMLU, SuperGLUE metadata indexer
```

### Synthesis (`synthesis/`)

```
synthesis/
├── prompts.py              # Prompt templates for all synthesis tasks
├── synthesize_bulk.py      # Parallel synthesis runner (vLLM or Claude API)
├── item_generator.py       # Novel eval item generation
├── contamination_prober.py # Synthetic contamination injection for training
└── shortcut_detector.py    # Shortcut detection pair synthesis
```

### Evaluation (`evaluation/`)

```
evaluation/
└── forgequality_bench.py   # ForgeQualityBench — evaluates evaluation quality
```

---

## DATA PIPELINE

```
OpenReview API ──────────────────────────────────────► papers.jsonl
arXiv eval papers ──────────────────────────────────► papers.jsonl
                                                           │
                              ┌────────────────────────────┘
                              ▼
                   Synthesis (Qwen2.5-72B or Claude)
                              │
              ┌───────────────┼────────────────────────────────┐
              ▼               ▼                                ▼
       audit_pairs.jsonl  item_pairs.jsonl          shortcut_pairs.jsonl
              │               │                                │
              └───────────────┴────────────────────────────────┘
                              │
                    Quality Filter (IRT + dedup)
                              │
                    train/ val/ test splits
                              │
                 ┌────────────┴───────────┐
                 ▼                        ▼
           SFT Training            DPO Pair Mining
                 │                        │
                 └────────────┬───────────┘
                              ▼
                        GRPO Training
                              │
                              ▼
                   evalforge-final checkpoint
```

---

## INFERENCE API

```
POST /audit
  body: {"benchmark": [...items], "model_scores": {...}}
  returns: AuditReport JSON

POST /forge
  body: {"existing_items": [...], "n_items": 100, "constraints": {...}}
  returns: {"new_items": [...], "irt_calibration": {...}}

GET /track/{benchmark_id}
  returns: LongitudinalReport with correlation history

POST /certify
  body: {"model_name": "...", "benchmark_results": {...}}
  returns: EvalCertificate (signed JSON artifact)
```

---

## TARGET METRICS

| Version | Task | Target | Baseline |
|---------|------|--------|----------|
| v1 | Shortcut detection recall | >85% | 0% (no baseline) |
| v1 | Contamination precision | >90% | 0% |
| v1 | Downstream Pearson r | >0.75 | MMLU: 0.42 |
| v1.5 | Ranking stability (Kendall tau) | >0.82 | 0.61 |
| v2 | Longitudinal drift detection | >0.90 AUC | — |
| v3 | EvalCertificate adoption | 10 AI labs | — |
