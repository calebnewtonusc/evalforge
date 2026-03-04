"""
synthesis/shortcut_detector.py — Programmatic shortcut detection for benchmark items.

Detects exploitable patterns without needing an LLM call.
Used both for:
  1. Data quality filtering (remove shortcuts from training data)
  2. Generating ground-truth labels for the GRPO reward signal

Detectable patterns:
  - length_bias: correct choice is ≥2× length of average distractor
  - lexical_overlap: all content words in correct choice appear in question stem
  - negation_artifact: negation word is the only differentiator between choices
  - position_bias: correct answer at position A or D in >70% of items in a set
  - distractor_implausibility: distractor is <3 words or obviously nonsensical
  - answer_choice_asymmetry: choices differ wildly in specificity/granularity
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

NEGATION_WORDS = {"not", "never", "no", "none", "neither", "nor", "nothing", "nowhere"}
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "and", "or", "but", "if", "then", "that", "this", "it", "its",
}


@dataclass
class ShortcutReport:
    """Result of shortcut analysis for a single item."""

    item_id: str
    shortcuts: list[dict[str, Any]] = field(default_factory=list)
    overall_quality_score: float = 1.0
    recommendation: str = "KEEP"

    def add_shortcut(
        self, pattern: str, severity: float, description: str, evidence: str = ""
    ) -> None:
        self.shortcuts.append(
            {
                "pattern": pattern,
                "severity": severity,
                "description": description,
                "evidence": evidence,
            }
        )
        # Degrade quality score
        self.overall_quality_score = max(0.0, self.overall_quality_score - severity * 0.3)
        if self.overall_quality_score < 0.4:
            self.recommendation = "REPLACE"
        elif self.overall_quality_score < 0.7:
            self.recommendation = "REVISE"


class ShortcutDetector:
    """Programmatic shortcut detection for multiple-choice benchmark items."""

    def analyze_item(self, item: dict) -> ShortcutReport:
        """
        Analyze a single item for exploitable shortcuts.

        Args:
            item: Dict with keys: question, choices (dict A/B/C/D → text), answer.

        Returns:
            ShortcutReport with detected patterns.
        """
        item_id = item.get("id", "unknown")
        question = item.get("question", "")
        choices = item.get("choices", {})
        answer_key = item.get("answer", "")

        report = ShortcutReport(item_id=item_id)

        if not choices or answer_key not in choices:
            return report

        correct_text = choices[answer_key]
        distractors = {k: v for k, v in choices.items() if k != answer_key}

        # 1. Length bias check
        self._check_length_bias(report, correct_text, distractors)

        # 2. Lexical overlap check
        self._check_lexical_overlap(report, question, correct_text)

        # 3. Negation artifact check
        self._check_negation_artifact(report, question, choices, answer_key)

        # 4. Distractor implausibility check
        self._check_distractor_implausibility(report, distractors)

        # 5. Answer choice asymmetry
        self._check_choice_asymmetry(report, choices)

        return report

    def analyze_item_set(self, items: list[dict]) -> dict[str, Any]:
        """
        Analyze a set of items for position bias and distribution-level shortcuts.

        Args:
            items: List of benchmark item dicts.

        Returns:
            Dict with set-level shortcut analysis.
        """
        item_reports = [self.analyze_item(item) for item in items]

        # Position bias: check if correct answer clusters at one position
        answer_positions = Counter()
        for item in items:
            answer_key = item.get("answer", "")
            if answer_key:
                answer_positions[answer_key] += 1

        position_bias_score = 0.0
        if answer_positions and len(items) > 0:
            max_count = max(answer_positions.values())
            expected = len(items) / max(1, len(answer_positions))
            position_bias_score = max(0.0, (max_count / expected - 1.0) / 2.0)
            position_bias_score = min(1.0, position_bias_score)

        # Aggregate item-level reports
        all_shortcuts: list[dict] = []
        for rep in item_reports:
            all_shortcuts.extend(rep.shortcuts)

        shortcut_counts = Counter(s["pattern"] for s in all_shortcuts)
        n_items = len(items)
        affected_fractions = {
            pattern: count / n_items for pattern, count in shortcut_counts.items()
        }

        return {
            "n_items": n_items,
            "position_bias_score": round(position_bias_score, 3),
            "position_distribution": dict(answer_positions),
            "item_level_shortcuts": affected_fractions,
            "flagged_item_fraction": sum(
                1 for r in item_reports if r.shortcuts
            ) / max(1, n_items),
            "replace_recommended_fraction": sum(
                1 for r in item_reports if r.recommendation == "REPLACE"
            ) / max(1, n_items),
        }

    def _check_length_bias(
        self, report: ShortcutReport, correct: str, distractors: dict[str, str]
    ) -> None:
        """Flag if correct answer is ≥2× length of average distractor."""
        if not distractors:
            return
        correct_len = len(correct.split())
        avg_distractor = sum(len(v.split()) for v in distractors.values()) / len(distractors)
        if avg_distractor > 0 and correct_len / avg_distractor >= 2.0:
            severity = min(1.0, (correct_len / avg_distractor - 2.0) * 0.5 + 0.5)
            report.add_shortcut(
                pattern="length_bias",
                severity=round(severity, 2),
                description="Correct answer is significantly longer than average distractor",
                evidence=f"correct={correct_len} tokens, avg_distractor={avg_distractor:.1f} tokens",
            )

    def _check_lexical_overlap(
        self, report: ShortcutReport, question: str, correct: str
    ) -> None:
        """Flag if content words in correct answer are all present in the question."""
        def content_words(text: str) -> set[str]:
            words = re.sub(r"[^\w\s]", "", text.lower()).split()
            return {w for w in words if w not in STOP_WORDS and len(w) > 2}

        q_words = content_words(question)
        a_words = content_words(correct)

        if len(a_words) < 2:
            return

        overlap = a_words & q_words
        overlap_fraction = len(overlap) / len(a_words)

        if overlap_fraction >= 0.75:
            report.add_shortcut(
                pattern="lexical_overlap",
                severity=round(overlap_fraction, 2),
                description="Most content words in correct answer appear in question stem",
                evidence=f"{len(overlap)}/{len(a_words)} answer content words overlap with question",
            )

    def _check_negation_artifact(
        self,
        report: ShortcutReport,
        question: str,
        choices: dict[str, str],
        answer_key: str,
    ) -> None:
        """Flag if negation word is the sole differentiator."""
        q_lower = question.lower()
        has_negation_in_q = any(neg in q_lower.split() for neg in NEGATION_WORDS)
        if not has_negation_in_q:
            return

        correct = choices.get(answer_key, "")
        other_choices = [v for k, v in choices.items() if k != answer_key]

        # Check if removing negations makes choices near-identical
        def strip_negation(text: str) -> str:
            return " ".join(w for w in text.lower().split() if w not in NEGATION_WORDS)

        correct_stripped = strip_negation(correct)
        for other in other_choices:
            other_stripped = strip_negation(other)
            # If stripped versions are nearly identical, negation is the key signal
            if _jaccard(correct_stripped.split(), other_stripped.split()) > 0.8:
                report.add_shortcut(
                    pattern="negation_artifact",
                    severity=0.7,
                    description="Question contains negation; correct/distractor differ only by negation word",
                    evidence=f"Q negation detected; choice similarity after stripping negation > 0.8",
                )
                return

    def _check_distractor_implausibility(
        self, report: ShortcutReport, distractors: dict[str, str]
    ) -> None:
        """Flag if any distractor is trivially rejectable (too short or nonsensical)."""
        short_distractors = [k for k, v in distractors.items() if len(v.split()) < 3]
        if short_distractors:
            report.add_shortcut(
                pattern="distractor_implausibility",
                severity=0.5,
                description=f"Distractor(s) {short_distractors} are implausibly short",
                evidence=f"Choices {short_distractors} have <3 words",
            )

    def _check_choice_asymmetry(
        self, report: ShortcutReport, choices: dict[str, str]
    ) -> None:
        """Flag large variance in choice specificity (granularity mismatch)."""
        if len(choices) < 2:
            return
        lengths = [len(v.split()) for v in choices.values()]
        max_len, min_len = max(lengths), min(lengths)
        if min_len > 0 and max_len / min_len >= 5:
            report.add_shortcut(
                pattern="answer_choice_asymmetry",
                severity=0.4,
                description="Extreme length variance across answer choices",
                evidence=f"max={max_len} tokens, min={min_len} tokens (ratio {max_len/min_len:.1f}x)",
            )


def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two word lists."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)
