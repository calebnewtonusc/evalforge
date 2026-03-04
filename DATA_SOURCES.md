# EvalForge — Training Data Sources

## Overview

EvalForge is trained on the full literature of evaluation methodology, benchmark critique, contamination analysis, and psychometrics. The goal is to internalize expert knowledge about *what makes a good benchmark* into model weights — not to scrape capability-testing data.

Total target: **400,000+ training pairs** across 5 streams.

---

## Stream 1 — OpenReview Evaluation Critique Papers (30% — ~120k pairs)

**Source:** OpenReview.net — NeurIPS, ICLR, ICML, ACL, EMNLP (2018–2026)

**What we collect:**
- Papers with "benchmark", "evaluation", "contamination", "shortcut" in title/abstract
- Peer reviews critiquing evaluation methodology
- Author rebuttals defending or conceding evaluation choices
- Meta-review assessments of evaluation quality

**Why this works:** OpenReview reviews are adversarial critiques by expert reviewers. A reviewer who writes "the accuracy on MMLU is inflated because the training set overlaps with benchmark items" has produced a ground-truth contamination analysis. EvalForge learns to reason like this reviewer.

**Collection:** `discovery/openreview_crawler.py`
- API endpoint: `https://api2.openreview.net/notes`
- Rate limit: 1 req/s, respectful crawling
- Target: 50k papers with methodology critique, ~200k review comments

**Synthesis to pairs:**
- Paper abstract + claimed evaluation results → `CRITIQUE` pair (what's wrong with this eval)
- Review comment → `SHORTCUT_DETECTION` pair (here is the specific flaw)
- Author rebuttal → `DEFENSE_ANALYSIS` pair (why this critique does/doesn't apply)

---

## Stream 2 — Benchmark Design Documents (20% — ~80k pairs)

**Source:** Official documentation, design papers, and critique analyses for major benchmarks

**Benchmarks covered:**
- **BIG-Bench**: 204 tasks, design rationale documents, failure case reports
- **HELM**: benchmark design papers, holistic evaluation framework documentation
- **MMLU**: original paper, critique papers (Gururangan et al., Alzahrani et al.)
- **SuperGLUE / GLUE**: design documents, annotation artifact papers
- **GSM8K**: contamination analysis studies
- **HumanEval**: shortcut exploitation reports
- **MATH**: distribution analysis, difficulty calibration papers
- **BoolQ, COPA, MultiRC, ReCoRD**: annotation process documentation

**Synthesis to pairs:**
- Benchmark design choice → `DESIGN_RATIONALE` pair
- Benchmark limitation → `LIMITATION_ANALYSIS` pair
- Critique paper → `BENCHMARK_AUDIT` pair (full audit in EvalForge format)

---

## Stream 3 — Contamination Analysis Studies (25% — ~100k pairs)

**Source:** Papers specifically studying training data contamination and benchmark gaming

**Key papers and datasets:**
- Golchin & Surdeanu (2023): "Time Travel in LLMs: Tracing Data Contamination in Large Language Models"
- Shi et al. (2023): "Detecting Pretraining Data from Large Language Models"
- Deng et al. (2023): "Don't Make Your LLM an Evaluation Benchmark Cheater"
- Zhou et al. (2023): "Don't Trust ChatGPT When Your Question Is Not in English"
- Magar & Schwartz (2022): "Data Contamination: From Memorization to Exploitation"
- Shortcut learning survey: Geirhos et al. (2020), Pezeshki et al. (2021)
- HANS dataset: annotation artifacts in NLI benchmarks

**Synthesis to pairs:**
- Contaminated item + detection method → `CONTAMINATION_DETECTION` pair
- Planted shortcut + identification → `SHORTCUT_IDENTIFICATION` pair
- Clean/contaminated item pairs → `COMPARATIVE_AUDIT` pair

**Key contribution:** We *plant* synthetic contamination during synthesis — known n-gram overlaps, template fills, answer position artifacts — creating ground-truth labeled training data for the shortcut detection reward signal.

---

## Stream 4 — Psychometrics Literature (15% — ~60k pairs)

**Source:** Educational measurement, cognitive psychology, and test design research

**Topics covered:**
- Item Response Theory (IRT): 1PL Rasch, 2PL, 3PL models — parameter estimation, item fit
- Construct validity: convergent, discriminant, content, criterion-related validity
- Item difficulty and discrimination: how to design items that separate ability levels
- Classical Test Theory (CTT): reliability, item-total correlation, split-half reliability
- Differential Item Functioning (DIF): detecting items that function differently across groups
- Anchor items and equating: linking scores across different test versions
- Standard error of measurement, confidence intervals for ability estimates

**Key textbooks synthesized:**
- Hambleton, Swaminathan & Rogers: "Fundamentals of Item Response Theory"
- Lord: "Applications of Item Response Theory to Practical Testing Problems"
- Cronbach: "Coefficient Alpha and the Internal Structure of Tests"
- Standards for Educational and Psychological Testing (AERA/APA/NCME, 2014)

**Synthesis to pairs:**
- Test design principle → `IRT_CALIBRATION` pair (apply IRT to a given item set)
- Construct validity question → `VALIDITY_ANALYSIS` pair
- Item parameter estimation → `IRT_FIT` pair

---

## Stream 5 — Goodhart's Law Failure Cases (10% — ~40k pairs)

**Source:** Case studies of metrics being optimized into meaninglessness

**Coverage:**
- **AI benchmarks**: BLEU score gaming, perplexity minimization artifacts, reward hacking in RLHF
- **Academic**: h-index manipulation, citation gaming, grade inflation
- **Corporate**: Wells Fargo account fraud (sales metric gaming), VW emissions scandal (measurement gaming)
- **Economics**: Campbell's Law, Cobra effect, teaching to the test
- **Sports**: Moneyball era stat gaming, PED-era home run records
- **Social media**: engagement metric optimization → outrage content

**Why this matters for EvalForge:** Every Goodhart's Law case follows the same structure — a proxy metric is established, agents find the gap between the proxy and the underlying construct, the proxy becomes detached from the construct. Recognizing this pattern across domains makes EvalForge robust to novel evaluation gaming strategies not yet observed in AI.

**Synthesis to pairs:**
- Case study description → `GOODHART_PATTERN` pair (which of the 23 patterns does this exemplify?)
- Novel scenario → `PATTERN_TRANSFER` pair (apply Goodhart pattern taxonomy to new domain)

---

## Data Quality Filters

All pairs pass through a multi-stage quality filter before training:

1. **Length filter**: discard pairs with system prompt <50 tokens or response <100 tokens
2. **Deduplication**: MinHash LSH dedup at 0.9 similarity threshold
3. **IRT quality**: for item-level pairs, IRT discrimination parameter must be >0.3
4. **Contamination self-check**: pairs must not contain verbatim chunks from known benchmark items
5. **Format validation**: structured audit reports must parse as valid JSON

**Expected yield after filtering:** ~75% of raw synthesis output

---

## Dataset Statistics (Target)

| Stream | Raw pairs | After filter | Format |
|--------|-----------|--------------|--------|
| OpenReview critique | 160k | 120k | JSONL (ShareGPT) |
| Benchmark design | 107k | 80k | JSONL (ShareGPT) |
| Contamination analysis | 133k | 100k | JSONL (ShareGPT) |
| Psychometrics | 80k | 60k | JSONL (ShareGPT) |
| Goodhart cases | 53k | 40k | JSONL (ShareGPT) |
| **Total** | **533k** | **400k** | — |

Train/val/test split: 90% / 5% / 5%
