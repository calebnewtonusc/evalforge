"""
agents/eval_designer_agent.py — Main EvalForge orchestration agent.

The EvalDesignerAgent coordinates the full evaluation design workflow:
  1. Receive benchmark (items + model scores)
  2. Run contamination detection
  3. Run shortcut detection
  4. Compute IRT calibration
  5. Check downstream correlation
  6. Generate audit report
  7. Optionally forge replacement items

Usage:
    from agents.eval_designer_agent import EvalDesignerAgent

    agent = EvalDesignerAgent(model_url="http://localhost:9000")
    report = agent.audit(benchmark_path="my_benchmark.jsonl", model_scores={...})
    print(report.to_json())
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from core.irt_models import IRTCalibrator
from synthesis.shortcut_detector import ShortcutDetector


@dataclass
class AuditReport:
    """Structured output from EvalDesignerAgent.audit()."""

    benchmark_name: str
    n_items: int
    contamination_score: float  # 0.0–1.0
    contaminated_items: list[str]  # item IDs flagged
    shortcuts_detected: list[dict]  # list of shortcut pattern summaries
    irt_analysis: dict[str, Any]  # discrimination, difficulty, quality flags
    downstream_correlation: dict[str, float]  # pearson r, spearman rho, etc.
    flagged_item_fraction: float
    recommendation: str
    replacement_priority_items: list[str]  # top items to replace

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.__dict__, indent=indent, default=str)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== EvalForge Audit Report: {self.benchmark_name} ===",
            f"Items: {self.n_items}",
            f"Contamination score: {self.contamination_score:.2f} (0=clean, 1=contaminated)",
            f"Contaminated items flagged: {len(self.contaminated_items)}",
            f"Shortcuts detected: {len(self.shortcuts_detected)} patterns",
            f"Items with shortcuts: {self.flagged_item_fraction:.1%}",
            f"IRT: low-discrimination items = {self.irt_analysis.get('low_discrimination_count', 0)}",
            f"Recommendation: {self.recommendation}",
            f"Priority replacements: {self.replacement_priority_items[:5]}",
        ]
        return "\n".join(lines)


class EvalDesignerAgent:
    """
    Orchestrates the full benchmark audit and redesign workflow.

    Combines:
    - Programmatic shortcut detection (ShortcutDetector)
    - LLM-based contamination analysis
    - IRT calibration (IRTCalibrator)
    - Downstream correlation tracking
    """

    def __init__(
        self,
        model_url: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self.model_url = model_url
        self.model_path = model_path
        self._client = None
        self.shortcut_detector = ShortcutDetector()
        self.irt_calibrator = IRTCalibrator(model="2pl")

        if model_url:
            import openai

            self._client = openai.OpenAI(
                base_url=f"{model_url}/v1",
                api_key=os.environ.get("VLLM_API_KEY", "dummy"),
            )

    def audit(
        self,
        items: list[dict],
        model_scores: dict[str, dict[str, float]] | None = None,
        benchmark_name: str = "unnamed_benchmark",
        response_matrix: np.ndarray | None = None,
        model_names: list[str] | None = None,
    ) -> AuditReport:
        """
        Full benchmark audit.

        Args:
            items: List of benchmark item dicts (id, question, choices, answer).
            model_scores: Dict of model_name → {item_id: 0/1 score}
            benchmark_name: Name for the report.
            response_matrix: (n_models, n_items) response matrix (alternative to model_scores).
            model_names: Model names corresponding to response_matrix rows.

        Returns:
            AuditReport with full analysis.
        """
        logger.info(f"Auditing benchmark: {benchmark_name} ({len(items)} items)")

        # 1. Programmatic shortcut detection
        logger.info("  Running shortcut detection...")
        set_analysis = self.shortcut_detector.analyze_item_set(items)
        item_reports = [self.shortcut_detector.analyze_item(item) for item in items]

        # Aggregate shortcuts
        all_shortcuts: dict[str, dict] = {}
        for report in item_reports:
            for shortcut in report.shortcuts:
                pattern = shortcut["pattern"]
                if pattern not in all_shortcuts:
                    all_shortcuts[pattern] = {
                        "pattern": pattern,
                        "n_items_affected": 0,
                        "mean_severity": 0.0,
                        "affected_fraction": 0.0,
                    }
                all_shortcuts[pattern]["n_items_affected"] += 1

        n_items = len(items)
        for pattern, data in all_shortcuts.items():
            data["affected_fraction"] = round(
                data["n_items_affected"] / max(1, n_items), 3
            )

        shortcuts_detected = list(all_shortcuts.values())

        # 2. LLM-based contamination analysis
        logger.info("  Running contamination analysis...")
        contaminated_items = self._run_contamination_analysis(items)
        # Denominator must match the number of items actually checked (capped at 50),
        # not n_items, otherwise the score is diluted for large benchmarks.
        contamination_score = len(contaminated_items) / max(1, min(n_items, 50))

        # 3. IRT calibration
        logger.info("  Running IRT calibration...")
        irt_result = None
        irt_analysis: dict[str, Any] = {}

        if response_matrix is not None and model_names is not None:
            item_ids = [item.get("id", f"item_{i}") for i, item in enumerate(items)]
            irt_result = self.irt_calibrator.calibrate(
                response_matrix, item_ids, model_names
            )
            irt_analysis = {
                "n_items": n_items,
                "low_discrimination_count": sum(
                    1
                    for ip in irt_result.item_parameters
                    if "LOW_DISCRIMINATION" in ip.quality_flags
                ),
                "ceiling_items": sum(
                    1
                    for ip in irt_result.item_parameters
                    if "CEILING" in ip.quality_flags
                ),
                "floor_items": sum(
                    1
                    for ip in irt_result.item_parameters
                    if "FLOOR" in ip.quality_flags
                ),
                "reliability_estimate": irt_result.test_information.reliability_estimate,
                "effective_n_items": irt_result.test_information.effective_n_items,
            }
        elif model_scores:
            # Estimate from accuracy data
            item_ids = [item.get("id", f"item_{i}") for i, item in enumerate(items)]
            model_list = list(model_scores.keys())
            matrix = np.zeros((len(model_list), n_items))
            for i, model in enumerate(model_list):
                for j, item_id in enumerate(item_ids):
                    matrix[i, j] = model_scores[model].get(item_id, 0.5)
            irt_result = self.irt_calibrator.calibrate(matrix, item_ids, model_list)
            irt_analysis = {
                "n_items": n_items,
                "low_discrimination_count": sum(
                    1
                    for ip in irt_result.item_parameters
                    if "LOW_DISCRIMINATION" in ip.quality_flags
                ),
                "reliability_estimate": irt_result.test_information.reliability_estimate,
            }

        # 4. Downstream correlation (stub — real version requires downstream task scores)
        downstream_correlation = {
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "note": "Provide downstream_task_scores to compute correlation.",
        }

        # 5. Priority items to replace
        flagged_items: list[str] = []
        if irt_result:
            flagged_items.extend(
                self.irt_calibrator.get_items_to_replace(irt_result, max_to_replace=10)
            )
        flagged_items.extend(contaminated_items[:10])
        flagged_items = list(
            dict.fromkeys(flagged_items)
        )  # deduplicate, preserve order

        # 6. Overall recommendation
        critical_fraction = len(contaminated_items) / max(1, n_items)
        shortcut_fraction = set_analysis.get("flagged_item_fraction", 0.0)
        if critical_fraction > 0.3 or shortcut_fraction > 0.5:
            recommendation = "MAJOR_REVISION_REQUIRED"
        elif critical_fraction > 0.1 or shortcut_fraction > 0.2:
            recommendation = "PARTIAL_REVISION_RECOMMENDED"
        else:
            recommendation = "ACCEPTABLE_WITH_MONITORING"

        report = AuditReport(
            benchmark_name=benchmark_name,
            n_items=n_items,
            contamination_score=round(contamination_score, 3),
            contaminated_items=contaminated_items,
            shortcuts_detected=shortcuts_detected,
            irt_analysis=irt_analysis,
            downstream_correlation=downstream_correlation,
            flagged_item_fraction=round(shortcut_fraction, 3),
            recommendation=recommendation,
            replacement_priority_items=flagged_items[:20],
        )

        logger.info(f"Audit complete. Recommendation: {recommendation}")
        return report

    def forge_replacement_items(
        self,
        items_to_replace: list[dict],
        constraints: dict | None = None,
        n_replacements_per_item: int = 1,
    ) -> list[dict]:
        """
        Generate replacement items for flagged items.

        Args:
            items_to_replace: Items that need replacement.
            constraints: Optional constraints (difficulty_range, shortcuts_blacklist).
            n_replacements_per_item: How many replacements to generate per item.

        Returns:
            List of replacement item dicts.
        """
        from synthesis.item_generator import ItemGenerator

        gen = ItemGenerator()
        constraints = constraints or {}

        all_replacements = []
        for item in items_to_replace:
            construct = item.get("construct", "general reasoning")
            difficulty_range = constraints.get("difficulty_range", (-1.0, 1.0))
            replacements = gen.generate(
                construct=construct,
                n_items=n_replacements_per_item,
                difficulty_range=difficulty_range,
                shortcuts_to_avoid=constraints.get(
                    "shortcuts_blacklist",
                    ["length_bias", "lexical_overlap", "negation_artifact"],
                ),
                existing_items=[item],
            )
            all_replacements.extend(replacements)

        logger.info(f"Generated {len(all_replacements)} replacement items")
        return all_replacements

    def _run_contamination_analysis(self, items: list[dict]) -> list[str]:
        """
        LLM-based contamination detection.

        Returns list of item IDs flagged as potentially contaminated.
        """
        if self._client is None:
            # Fall back to heuristic n-gram detection
            return self._heuristic_contamination(items)

        contaminated: list[str] = []
        for item in items[:50]:  # Limit for cost
            item_id = item.get("id", "unknown")
            try:
                prompt = (
                    f"Analyze this benchmark item for training data contamination. "
                    f"Does the question appear verbatim or near-verbatim in Common Crawl, "
                    f"Wikipedia, or other common pretraining sources?\n\n"
                    f"Item: {json.dumps(item, indent=2)}\n\n"
                    'Respond with JSON: {"contaminated": true/false, "evidence": "..."}'
                )
                resp = self._client.chat.completions.create(
                    model="evalforge",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                try:
                    data = json.loads(text)
                    if data.get("contaminated", False):
                        contaminated.append(item_id)
                except json.JSONDecodeError:
                    if "true" in text.lower():
                        contaminated.append(item_id)
            except Exception as e:
                logger.debug(f"Contamination check failed for {item_id}: {e}")

        return contaminated

    def _heuristic_contamination(self, items: list[dict]) -> list[str]:
        """
        Simple heuristic contamination detection without LLM.
        Flags items with very common n-grams as potentially contaminated.
        """
        common_phrases = {
            "according to",
            "as stated in",
            "the following",
            "which of the following",
            "based on the",
            "in accordance with",
        }
        flagged = []
        for item in items:
            question = item.get("question", "").lower()
            if any(phrase in question for phrase in common_phrases):
                if (
                    len(question) < 100
                ):  # Short items with template phrases are suspicious
                    flagged.append(item.get("id", "unknown"))
        return flagged
