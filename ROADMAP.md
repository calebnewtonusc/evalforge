# EvalForge Roadmap

---

## v1 — Audit (Current Build)

**Theme:** Detect what's wrong with existing benchmarks.

**Capabilities:**
- Contamination detection at the item level (n-gram + embedding proximity)
- Goodhart pattern classification across 23 known gaming strategies
- IRT calibration (2PL/3PL) — discrimination, difficulty, guessing
- Annotation artifact detection (length bias, lexical overlap, negation)
- Structured audit report generation (JSON + human-readable)
- ForgeQualityBench v1 — shortcut detection + contamination recall/precision

**Training:**
- 400k+ pairs across 5 data streams
- SFT on Qwen2.5-7B-Coder-Instruct + LoRA rank 64
- GRPO with verifiable reward (shortcut detection + downstream correlation)
- DPO on expert-curated audit quality preferences

**Hardware:** 18× A6000, ~8 hours total training

**Deliverable:** Open-source audit API. Run `POST /audit` on any benchmark, receive structured contamination + shortcut + IRT report.

---

## v1.5 — Forge (Next Release)

**Theme:** Generate new benchmark items that eliminate detected weaknesses.

**New Capabilities:**
- Item generator conditioned on: difficulty target, construct facet, shortcut blacklist
- IRT-aware gap filling — generate items in underrepresented discrimination-difficulty cells
- Cross-benchmark transfer — forge items in the style of MMLU while eliminating MMLU's shortcuts
- Diversity verification — self-BLEU + embedding deduplication against existing items
- Difficulty calibration loop — iterate generation until IRT parameters converge

**New Data:**
- 50k item generation pairs (existing item → improved replacement)
- 20k constraint-conditioned generation pairs (difficulty/facet/shortcut constraints)

**New Training:**
- Stage 2 reward extended: item diversity (self-BLEU ≥ 0.65) as additional reward component
- Item quality DPO pairs from human expert review

**Deliverable:** `POST /forge` API. Give EvalForge a broken benchmark, receive a replacement item set.

---

## v2 — Track (Q3 2026)

**Theme:** Longitudinal correlation monitoring — know when a benchmark is dying before everyone else does.

**New Capabilities:**
- Benchmark registration system — register any benchmark for longitudinal monitoring
- Drift alerting — automatic notification when ranking stability drops below threshold
- Model generation tracking — separate discrimination curves per model family
- Correlation dashboard — live Pearson/Spearman/Kendall tracking across model releases
- Temporal contamination detection — items that were clean at release but contaminated after training cutoff

**New Data:**
- 100k longitudinal pairs (benchmark at time T → correlation shift at T+1)
- Temporal contamination event corpus (cases where clean benchmarks became contaminated)

**Target Customers:** AI evaluation teams at major labs (Anthropic, OpenAI, Google DeepMind, Cohere)

---

## v3 — Certify (Q1 2027)

**Theme:** Machine-readable, cryptographically-signed evaluation certificates.

**New Capabilities:**
- EvalCertificate: signed JSON artifact attesting benchmark quality scores
- Certificate versioning — track how a benchmark's certificate degrades over time
- Third-party verification — any party can verify a certificate against the EvalForge API
- Regulatory readiness — format designed for AI Act (EU) compliance audit workflows
- Enterprise API — white-label evaluation certification for AI labs

**Vision:** The way SSL certificates became the standard for web trust, EvalCertificates become the standard for model evaluation trust. Every model card links its benchmark results to a machine-verifiable EvalCertificate.

**Business model:** SaaS API for AI teams + enterprise licensing for evaluation certification.
