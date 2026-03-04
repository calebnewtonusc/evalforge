# Contributing to EvalForge

EvalForge improves with community contributions — especially training data, benchmark audit reports, and IRT calibration expertise.

---

## Ways to Contribute

### 1. Training Data — Benchmark Audit Cases

The most valuable contribution is a high-quality benchmark audit case: a (benchmark_item, audit_report) pair where the audit is grounded in published evidence.

Format:
```json
{
  "benchmark": "MMLU",
  "category": "professional_medicine",
  "item": {
    "question": "...",
    "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "C"
  },
  "audit": {
    "contamination_evidence": "This item appears verbatim in Kaplan USMLE Step 1 prep materials, which are in Common Crawl.",
    "shortcuts_detected": ["length_bias"],
    "irt_notes": "All incorrect options are implausibly short — trivially rejectable.",
    "sources": ["doi:10.xxxx/xxxxx"]
  }
}
```

Submit via PR to `data/community/audit_cases.jsonl`.

### 2. Goodhart Pattern Taxonomy Additions

Found a benchmark gaming strategy not in our 23-pattern taxonomy? Submit a PR to `core/goodhart_patterns.py` with:
- Pattern name and category
- Formal definition
- At least one documented example from the literature
- Detection heuristic (how to identify this pattern)

### 3. ForgeQualityBench Test Cases

Add test cases for the meta-evaluation benchmark:
```bash
python evaluation/forgequality_bench.py --add-test \
  --shortcut-type length_bias \
  --item-file my_test_item.json \
  --expected-flag True
```

### 4. IRT Validation Studies

If you have access to model response data on an existing benchmark (the full response matrix, not just aggregate accuracy), we can use this to validate IRT calibration. Reach out via GitHub Issues with the `[irt-data]` tag.

---

## Code Guidelines

- Python 3.11+, type annotations required on all public functions
- Docstrings required (Google style)
- `loguru` for logging — no `print()` in library code
- `pytest` for tests — 90%+ coverage on core/ modules
- Format: `black --line-length 100`
- Lint: `ruff check`

Run before submitting:
```bash
black --check --line-length 100 .
ruff check .
pytest tests/ -v
```

---

## Community Standards

- Cite your sources — every audit case should reference published evidence
- Do not submit benchmark items verbatim from copyrighted sources
- Flag uncertainty — if an audit claim is speculative, mark it `"confidence": "low"`
- Peer review — all community audit cases go through two-reviewer process before merging

---

## Issue Templates

**Bug report**: Use the `[bug]` tag. Include Python version, platform, full traceback.

**Feature request**: Use the `[feature]` tag. Describe the use case before the implementation.

**Benchmark audit request**: Use the `[audit-request]` tag. Specify the benchmark and the specific concern (contamination suspicion, shortcut type, etc.).

---

## Contact

- GitHub Issues: preferred for all technical discussion
- Maintainer: Caleb Newton ([@calebnewtonusc](https://github.com/calebnewtonusc))
