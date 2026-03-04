"""
core/goodhart_patterns.py — Taxonomy of benchmark gaming strategies.

23 Goodhart patterns catalogued from the evaluation critique literature.
Each pattern describes how optimizing a proxy metric diverges from the
true underlying construct.

Reference: Campbell's Law, Goodhart's Law, Strathern (1997),
           Geirhos et al. (2020), Ribeiro et al. (2020), and others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PatternCategory(str, Enum):
    SURFACE = "surface"              # Exploits surface-level features
    DISTRIBUTIONAL = "distributional"  # Exploits statistical distribution of training data
    STRUCTURAL = "structural"        # Exploits test format/structure
    SEMANTIC = "semantic"            # Exploits semantic shortcuts
    TEMPORAL = "temporal"            # Exploits temporal leakage
    SYSTEMIC = "systemic"           # Exploits system-level measurement failures


@dataclass
class GoodhartPattern:
    """A single Goodhart pattern with detection heuristic."""

    id: str
    name: str
    category: PatternCategory
    description: str
    ai_benchmark_example: str
    real_world_analog: str
    detection_heuristic: str
    severity_indicators: list[str] = field(default_factory=list)
    mitigation: str = ""
    references: list[str] = field(default_factory=list)


PATTERNS: dict[str, GoodhartPattern] = {
    # --- SURFACE PATTERNS ---
    "G01_length_bias": GoodhartPattern(
        id="G01_length_bias",
        category=PatternCategory.SURFACE,
        name="Length Bias",
        description=(
            "Models learn that correct answers are systematically longer (or shorter) "
            "than distractors, and exploit this as a shortcut."
        ),
        ai_benchmark_example=(
            "In RACE, DREAM, and other reading comprehension datasets, the correct answer "
            "is on average 2.3× longer than distractors. Models trained on these datasets "
            "learn to pick the longest option without reading the passage."
        ),
        real_world_analog=(
            "Multiple-choice test design: teachers often write more hedged, qualified "
            "correct answers ('Under most conditions, X...') while incorrect options "
            "are shorter and bolder."
        ),
        detection_heuristic=(
            "Compute mean length of correct choices vs. mean length of distractors "
            "per category. Flag if ratio > 1.5 or < 0.67."
        ),
        severity_indicators=[
            "Correct answer length / mean distractor length > 2.0",
            "Correlation between answer choice length rank and correctness > 0.4",
        ],
        mitigation=(
            "Length-match answer choices during item construction. "
            "Constrain item generator to produce choices within ±20% of mean length."
        ),
        references=["Ko et al. 2020: Length Bias in Question Answering"],
    ),

    "G02_lexical_overlap": GoodhartPattern(
        id="G02_lexical_overlap",
        category=PatternCategory.SURFACE,
        name="Lexical Overlap Shortcut",
        description=(
            "Correct answers contain more words from the question/premise "
            "than distractors, enabling resolution without comprehension."
        ),
        ai_benchmark_example=(
            "In NLI datasets (SNLI, MNLI), entailment hypotheses share more "
            "vocabulary with the premise than contradiction hypotheses. "
            "A bag-of-words classifier achieves 67% accuracy on SNLI."
        ),
        real_world_analog=(
            "In standardized testing, students learn that the correct answer "
            "often contains keywords from the question stem — a surface heuristic "
            "that works even without domain understanding."
        ),
        detection_heuristic=(
            "Compute Jaccard similarity between question stem tokens and each "
            "answer choice. Flag if correct choice has Jaccard ≥ 0.5."
        ),
        severity_indicators=[
            "Mean Jaccard(question, correct) > 2× mean Jaccard(question, distractors)",
            "A TF-IDF classifier trained only on choice text achieves > 60% accuracy",
        ],
        mitigation="Screen generated items with `ShortcutDetector._check_lexical_overlap()`",
        references=["Gururangan et al. 2018: Annotation Artifacts in Natural Language Inference Data"],
    ),

    "G03_negation_artifact": GoodhartPattern(
        id="G03_negation_artifact",
        category=PatternCategory.SURFACE,
        name="Negation Artifact",
        description=(
            "Presence or absence of negation words (not, never, no) in question "
            "or answer choice is a sufficient signal for correct answer selection."
        ),
        ai_benchmark_example=(
            "In WinoGrande, questions containing 'NOT' were solved at 74% accuracy "
            "by a model that only attended to negation words — 15 points above baseline."
        ),
        real_world_analog="True/False exams where 'always' and 'never' are usually false.",
        detection_heuristic=(
            "Strip all negation words from question and choices. "
            "If correct answer ranking changes, negation is a shortcut signal."
        ),
        severity_indicators=[
            "Model accuracy on negation-containing items > 10pp above non-negation items",
            "Correct answer contains negation at 2× the rate of distractors",
        ],
        mitigation="Balance negation distribution across correct and distractor choices.",
        references=["Kassner & Schütze 2020: Negated and Misprimed Probes for Pretrained LMs"],
    ),

    "G04_position_bias": GoodhartPattern(
        id="G04_position_bias",
        category=PatternCategory.STRUCTURAL,
        name="Answer Position Bias",
        description=(
            "Correct answers cluster at certain positions (A/B/C/D) in the test, "
            "allowing models to exploit position as a signal."
        ),
        ai_benchmark_example=(
            "Analysis of MMLU showed that across professional law questions, "
            "option D is correct 29% of the time (expected: 25%). Calibration-aware "
            "models learn this distribution and bias toward D."
        ),
        real_world_analog=(
            "Standardized test lore: 'When in doubt, choose C.' This reflects real "
            "position bias in historical SAT and ACT question construction."
        ),
        detection_heuristic=(
            "Compute frequency distribution of correct answers across positions. "
            "χ² test for uniform distribution. Flag if max position > 35% for 4-option MC."
        ),
        severity_indicators=[
            "Any single position has >30% correct answers in a 4-option benchmark",
            "χ² p-value < 0.01 against uniform distribution",
        ],
        mitigation="Shuffle answer position after generation; ensure uniform correct-answer distribution.",
        references=["Ko et al. 2020: Position Bias in Question Answering"],
    ),

    # --- DISTRIBUTIONAL PATTERNS ---
    "G05_ngram_contamination": GoodhartPattern(
        id="G05_ngram_contamination",
        category=PatternCategory.TEMPORAL,
        name="N-gram Training Contamination",
        description=(
            "Benchmark items appear verbatim or near-verbatim in model training data, "
            "allowing memorization rather than generalization."
        ),
        ai_benchmark_example=(
            "Shi et al. (2023) showed that GPT-3 could reproduce held-out MMLU questions "
            "when prompted with the first few words — evidence of direct memorization."
        ),
        real_world_analog="Teaching to the test: curriculum is aligned to exact exam questions.",
        detection_heuristic=(
            "Compute 8-gram overlap between benchmark items and Common Crawl snapshots. "
            "Flag items with > 2 matching 8-grams in any training corpus snapshot."
        ),
        severity_indicators=[
            "Model achieves > 15pp higher accuracy on items with training overlap",
            "Model can reproduce item text with > 5-gram accuracy given 3-gram prompt",
        ],
        mitigation=(
            "Filter item generation to avoid n-grams present in any known pretraining corpus. "
            "Use EvalForge's contamination prober during item construction."
        ),
        references=["Shi et al. 2023: Detecting Pretraining Data from Large Language Models"],
    ),

    "G06_frequency_prior": GoodhartPattern(
        id="G06_frequency_prior",
        category=PatternCategory.DISTRIBUTIONAL,
        name="Frequency Prior",
        description=(
            "The correct answer is the most statistically frequent answer in the training "
            "distribution, resolvable without the question."
        ),
        ai_benchmark_example=(
            "On CommonsenseQA, a prior-only baseline that always predicts 'yes' achieves 58% "
            "on BoolQ — substantially above the random baseline of 50%."
        ),
        real_world_analog="Market research: surveying only frequent customers biases results.",
        detection_heuristic=(
            "Train a label-only classifier (no input features) on the training split. "
            "Flag if it achieves > 5pp above random baseline on the test split."
        ),
        severity_indicators=[
            "Label-only baseline accuracy > 60% for binary, > 30% for 4-option MC",
            "Answer choice text contains domain-specific keywords that are always correct",
        ],
        mitigation="Balance answer choice label frequencies during dataset construction.",
        references=["Gururangan et al. 2018: Annotation Artifacts in Natural Language Inference Data"],
    ),

    # --- SEMANTIC PATTERNS ---
    "G07_semantic_collapse": GoodhartPattern(
        id="G07_semantic_collapse",
        category=PatternCategory.SEMANTIC,
        name="Semantic Collapse of Distractors",
        description=(
            "Multiple distractor options convey nearly identical meanings, "
            "allowing elimination by identifying the one semantically distinct choice."
        ),
        ai_benchmark_example=(
            "In some medical knowledge MCQ benchmarks, three of four options describe "
            "the same underlying mechanism in different phrasings — the correct answer "
            "is trivially isolated as the only semantically distinct option."
        ),
        real_world_analog="Survey design: redundant options reduce effective choice set size.",
        detection_heuristic=(
            "Compute pairwise semantic similarity (cosine in sentence embedding space) "
            "among distractor options. Flag if any pair exceeds 0.85 similarity."
        ),
        severity_indicators=[
            "Pairwise cosine similarity between any two distractors > 0.85",
            "Removing one distractor leaves a 3-option set with < 5pp accuracy change",
        ],
        mitigation="Enforce semantic diversity: all choices must have pairwise cosine < 0.70.",
        references=["Swaminathan et al. 2020: Perturbation CheckLists for NLI"],
    ),

    "G08_distractor_implausibility": GoodhartPattern(
        id="G08_distractor_implausibility",
        category=PatternCategory.SEMANTIC,
        name="Implausible Distractor",
        description=(
            "One or more incorrect choices are so obviously wrong that the effective "
            "choice set is smaller than designed, inflating accuracy estimates."
        ),
        ai_benchmark_example=(
            "In early MMLU versions, some professional medicine questions included "
            "distractors that were anatomically impossible (e.g., 'the liver is located "
            "in the skull') — eliminatable without domain knowledge."
        ),
        real_world_analog=(
            "Multiple-choice test with one obviously wrong option reduces the guessing "
            "floor from 25% to 33%, inflating scores for partially-informed test takers."
        ),
        detection_heuristic=(
            "Measure model confidence for each choice in isolation (without the question). "
            "Flag distractors with model confidence < 5% unconditionally."
        ),
        severity_indicators=[
            "Any distractor achieves < 5% selection rate across multiple model evaluations",
            "Distractor contains obvious factual impossibilities detectable by pattern matching",
        ],
        mitigation="Require each distractor to be selected by at least 10% of human test-takers.",
        references=["Haley & Stasio 2005: Test Construction Guidelines (adapted)"],
    ),

    # --- STRUCTURAL PATTERNS ---
    "G09_template_fill": GoodhartPattern(
        id="G09_template_fill",
        category=PatternCategory.STRUCTURAL,
        name="Template Fill-in",
        description=(
            "Items are constructed from templates where the answer can be recovered "
            "by slot-filling without understanding the underlying concept."
        ),
        ai_benchmark_example=(
            "LAMA probes (Petroni et al.) — 'The Eiffel Tower is in [MASK]' — test "
            "factual knowledge but are solvable by pattern completion. Models achieve "
            "high accuracy by matching training-set templates, not retrieving facts."
        ),
        real_world_analog="Mad Libs: fill-in-the-blank without reading context.",
        detection_heuristic=(
            "Compute structural similarity across items using parse tree edit distance. "
            "Flag batches of items with > 80% structural overlap (template families)."
        ),
        severity_indicators=[
            "More than 20% of items share the same parse tree structure",
            "Accuracy on template-filled items > 20pp above paraphrase equivalents",
        ],
        mitigation="Constrain item generation to avoid repeated syntactic templates.",
        references=["Petroni et al. 2019: Language Models as Knowledge Bases?"],
    ),

    "G10_annotation_agreement": GoodhartPattern(
        id="G10_annotation_agreement",
        category=PatternCategory.DISTRIBUTIONAL,
        name="Annotation Agreement Artifact",
        description=(
            "Items where annotators had near-perfect agreement are trivially easy, "
            "producing ceiling effects that prevent discrimination among capable models."
        ),
        ai_benchmark_example=(
            "In SNLI, items with 5/5 annotator agreement are solved at > 95% by models "
            "that score only 75% on 3/5 agreement items. The easy items don't discriminate."
        ),
        real_world_analog="Exam questions that every student gets right don't measure ability.",
        detection_heuristic=(
            "Filter benchmark by annotation agreement score. Compare model accuracy "
            "on high-agreement vs. low-agreement subsets."
        ),
        severity_indicators=[
            "Accuracy on high-agreement items > 15pp above low-agreement items",
            "More than 40% of items have > 90% annotator agreement",
        ],
        mitigation=(
            "Include items across the full agreement spectrum. "
            "Weight low-agreement items more heavily in scoring."
        ),
        references=["Nie et al. 2020: Adversarial NLI: A New Benchmark for Natural Language Understanding"],
    ),

    # --- SYSTEMIC PATTERNS ---
    "G11_benchmark_saturation": GoodhartPattern(
        id="G11_benchmark_saturation",
        category=PatternCategory.SYSTEMIC,
        name="Benchmark Saturation",
        description=(
            "As models approach ceiling performance on a benchmark, ranking among "
            "models becomes noise-dominated — the benchmark has lost discriminative power."
        ),
        ai_benchmark_example=(
            "GPT-3 achieved 43.9% on MMLU (2020). By 2024, multiple models exceeded 87%. "
            "Ranking among models in the 85-90% range is dominated by measurement noise."
        ),
        real_world_analog="IQ test given to PhD students — ceiling effect removes discriminative power.",
        detection_heuristic=(
            "Compute model ranking stability across two evaluation runs with different "
            "item samplings. Low Kendall tau indicates saturation-induced noise."
        ),
        severity_indicators=[
            "Top 5 models within 2 standard errors of each other on all items",
            "Score variance across models < 5pp",
            "Kendall tau of model rankings across two benchmark variants < 0.6",
        ],
        mitigation="Trigger automatic benchmark refresh when ranking stability drops below threshold.",
        references=["Raji et al. 2021: AI and the Everything in the Whole Wide World Benchmark"],
    ),

    "G12_measurement_drift": GoodhartPattern(
        id="G12_measurement_drift",
        category=PatternCategory.TEMPORAL,
        name="Measurement Drift",
        description=(
            "Benchmark performance improves over time not because the measured capability "
            "improves, but because the benchmark is increasingly incorporated into training."
        ),
        ai_benchmark_example=(
            "BIG-Bench tasks showed rapid performance improvement across model generations "
            "uncorrelated with the model's performance on held-out real-world tasks, "
            "suggesting the improvement was task-specific rather than capability-general."
        ),
        real_world_analog=(
            "Teaching to the test: NAEP scores rise after high-stakes testing policies, "
            "but don't predict improved college readiness."
        ),
        detection_heuristic=(
            "Track correlation between benchmark scores and downstream task performance "
            "across model generations. Declining Pearson r signals measurement drift."
        ),
        severity_indicators=[
            "Pearson r between benchmark and downstream tasks declines > 0.1 per model generation",
            "Models with high benchmark scores perform worse than expected on real-world tasks",
        ],
        mitigation=(
            "Continuously refresh benchmark items. "
            "Track downstream correlation as a first-class metric."
        ),
        references=["Raji et al. 2021: AI and the Everything in the Whole Wide World Benchmark"],
    ),
}

# Add remaining patterns (13–23) with abbreviated entries
_ADDITIONAL_PATTERNS = [
    ("G13_cherry_picking", "Cherry-Picking", PatternCategory.SYSTEMIC,
     "Selective reporting of benchmark subsets where model performs best."),
    ("G14_calibration_gaming", "Calibration Gaming", PatternCategory.DISTRIBUTIONAL,
     "Optimizing for calibration metrics (ECE) without improving accuracy."),
    ("G15_contamination_paraphrase", "Paraphrastic Contamination", PatternCategory.TEMPORAL,
     "Training data contains paraphrases of benchmark items — embedding-level overlap."),
    ("G16_metric_hacking", "Metric Hacking", PatternCategory.STRUCTURAL,
     "Direct optimization of benchmark-specific patterns (e.g., BLEU-optimized generation)."),
    ("G17_context_length_shortcut", "Context Length Shortcut", PatternCategory.SURFACE,
     "Long-context tasks solvable by attending to first/last sentence only."),
    ("G18_format_memorization", "Format Memorization", PatternCategory.STRUCTURAL,
     "Models memorize the output format expected by evaluator, not the underlying task."),
    ("G19_distractor_generation_bias", "Distractor Generation Bias", PatternCategory.DISTRIBUTIONAL,
     "LLM-generated distractors share distributional properties of LLM training data."),
    ("G20_adversarial_blindspot", "Adversarial Blind Spot", PatternCategory.SEMANTIC,
     "Benchmark items are drawn from a narrow adversarial distribution not representative of real use."),
    ("G21_language_style_bias", "Language Style Bias", PatternCategory.SURFACE,
     "Models prefer certain writing styles (formal, bullet-pointed) regardless of correctness."),
    ("G22_multihop_shortcut", "Multi-hop Shortcut", PatternCategory.SEMANTIC,
     "Multi-hop reasoning tasks solvable by single-hop patterns on bridge entities."),
    ("G23_evaluation_reward_hacking", "Evaluation Reward Hacking", PatternCategory.SYSTEMIC,
     "RLHF models learn to produce outputs that score well on the specific evaluator, not on the task."),
]

for pattern_id, name, category, desc in _ADDITIONAL_PATTERNS:
    PATTERNS[pattern_id] = GoodhartPattern(
        id=pattern_id,
        name=name,
        category=category,
        description=desc,
        ai_benchmark_example="See literature review in ARCHITECTURE.md.",
        real_world_analog="See Goodhart's Law literature.",
        detection_heuristic="Domain-specific — see individual pattern documentation.",
    )


def get_pattern(pattern_id: str) -> GoodhartPattern | None:
    """Retrieve a pattern by ID."""
    return PATTERNS.get(pattern_id)


def get_patterns_by_category(category: PatternCategory) -> list[GoodhartPattern]:
    """Get all patterns in a category."""
    return [p for p in PATTERNS.values() if p.category == category]


def pattern_ids() -> list[str]:
    """Return all pattern IDs."""
    return list(PATTERNS.keys())


def pattern_summary() -> dict[str, str]:
    """Return ID → name mapping for all patterns."""
    return {pid: p.name for pid, p in PATTERNS.items()}
