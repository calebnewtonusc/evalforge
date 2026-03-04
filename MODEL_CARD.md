# EvalForge Model Card

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | EvalForge-7B-v1 |
| **Base model** | Qwen/Qwen2.5-7B-Coder-Instruct |
| **Fine-tuning method** | LoRA (rank 64, alpha 128) → merged |
| **Training stages** | 3 (SFT → GRPO → DPO) |
| **Training data** | 400k+ pairs (5 streams, see DATA_SOURCES.md) |
| **Training hardware** | 18× NVIDIA A6000 48GB |
| **Training duration** | ~8 hours total |
| **Context length** | 16,384 tokens |
| **License** | Apache 2.0 |
| **Developer** | Caleb Newton (USC) |

---

## What This Model Does

EvalForge-7B is a specialist model trained to analyze, critique, and improve AI evaluation benchmarks. It takes a benchmark (a set of items with model scores) as input and produces:

1. **Contamination analysis** — which items overlap with likely training data
2. **Shortcut detection** — which items can be solved without the target capability
3. **IRT calibration** — discrimination and difficulty parameters for each item
4. **Replacement item generation** — novel items that address detected weaknesses
5. **Downstream correlation** — does this benchmark predict real capability?

---

## Intended Use

**Primary use cases:**
- AI research teams auditing their benchmark suite before publication
- ML practitioners evaluating whether model rankings on a benchmark are trustworthy
- Benchmark designers generating adversarially robust evaluation items
- AI policy teams assessing benchmark quality for regulatory purposes

**Secondary use cases:**
- Educational testing organizations applying AI-era contamination detection to standardized tests
- Data scientists designing internal evaluation sets for production ML systems

---

## Out-of-Scope Uses

- **Not an evaluation runner** — EvalForge does not score models on benchmarks; it audits benchmarks themselves
- **Not for personal assessment** — this model is not designed for or tested on human ability assessment (LSAT, GRE, etc.) despite psychometrics training data
- **Not for adversarial benchmark poisoning** — the model's item generation capability should not be used to introduce misleading items into public benchmarks

---

## Training Data Summary

| Stream | Description | Volume |
|--------|-------------|--------|
| OpenReview critique papers | Expert peer review of evaluation methodology | 120k pairs |
| Benchmark design documents | BIG-Bench, HELM, MMLU, SuperGLUE documentation | 80k pairs |
| Contamination analysis studies | Papers studying training data contamination | 100k pairs |
| Psychometrics literature | IRT, construct validity, test design | 60k pairs |
| Goodhart's Law cases | Cross-domain metric gaming case studies | 40k pairs |

---

## Limitations

**Known limitations:**

- **Domain coverage**: training data is weighted toward NLP/reasoning benchmarks. Performance on computer vision or code execution benchmark auditing is less validated.
- **Novel gaming strategies**: the Goodhart pattern taxonomy covers 23 known strategies. Models trained after EvalForge's training cutoff may develop novel gaming strategies not in the taxonomy.
- **IRT assumptions**: IRT assumes unidimensionality (one latent trait). Many AI benchmarks test multiple capabilities; IRT estimates on multi-dimensional data are approximate.
- **Contamination detection scope**: n-gram and embedding detection can identify known contamination patterns. Paraphrastic contamination (same content, different surface form) may not be detected at v1.
- **7B model size**: some audit tasks (cross-benchmark correlation analysis, complex IRT estimation on large item sets) may exceed 7B model capacity. Larger-scale versions planned for v2.

**Evaluation coverage at v1:**
- English-language benchmarks only
- Text-based benchmarks (not multimodal at v1)
- Classification and generation tasks (not interactive/agentic benchmarks at v1)

---

## Evaluation Results (ForgeQualityBench v1)

*Results to be populated after training run completes.*

| Metric | EvalForge-7B-v1 | No-model baseline |
|--------|-----------------|-------------------|
| Shortcut detection recall | TBD | 0% |
| Contamination precision | TBD | 0% |
| Downstream Pearson r (vs. MMLU) | TBD | MMLU: 0.42 |
| Ranking stability (Kendall tau) | TBD | 0.61 |
| Item generation diversity (self-BLEU) | TBD | — |

---

## Ethical Considerations

**Dual-use risk:** EvalForge can generate evaluation items and detect shortcuts. A bad actor could theoretically use this to understand *what shortcuts work*, then train models to exploit them more effectively.

**Mitigation:** The shortcut detection capability is more valuable than the gaming capability — most shortcuts are already known in the literature. EvalForge's contribution is systematizing detection and making it accessible to evaluation designers, not revealing new attack surface.

**Bias in psychometrics data:** Historical psychometrics literature contains work produced in contexts with known measurement bias (differential item functioning against certain demographic groups). EvalForge is trained on this literature for its statistical methodology, not its historical bias patterns. The IRT implementation explicitly includes DIF analysis as a quality check.

---

## Citation

```bibtex
@inproceedings{newton2026evalforge,
  title     = {EvalForge: Training a Model to Design Contamination-Resistant Evaluations},
  author    = {Newton, Caleb},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026},
}
```
