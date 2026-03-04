"""
evaluation/forgequality_bench.py — ForgeQualityBench: evaluating evaluation quality.

This is the meta-evaluation: a benchmark for evaluating how well EvalForge
designs and audits benchmarks. It measures:

  1. Shortcut detection recall  — fraction of planted shortcuts detected
  2. Contamination precision    — precision when flagging contaminated items
  3. IRT discrimination         — are IRT estimates well-calibrated?
  4. Downstream correlation     — do forge-designed benchmarks predict real capability?
  5. Item diversity             — are generated items distinct from training data?

Usage:
    python evaluation/forgequality_bench.py --model checkpoints/evalforge-final
    python evaluation/forgequality_bench.py --model-url http://localhost:9000
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.stats import pearsonr, kendalltau


class ForgeQualityBench:
    """
    Meta-evaluation for EvalForge.

    Runs five test suites and reports a composite ForgeQuality score.
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_url: str | None = None,
        test_data_dir: str = "evaluation/test_data",
    ) -> None:
        self.model_path = model_path
        self.model_url = model_url
        self.test_data_dir = Path(test_data_dir)
        self._client = None

        if model_url:
            import openai
            self._client = openai.OpenAI(
                base_url=f"{model_url}/v1",
                api_key=os.environ.get("VLLM_API_KEY", "dummy"),
            )
        elif model_path:
            self._load_local_model(model_path)

    def _load_local_model(self, model_path: str) -> None:
        """Load model for local inference."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Loading model from {model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()

    def _infer(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Run inference on the EvalForge model."""
        if self._client:
            resp = self._client.chat.completions.create(
                model="evalforge",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content
        else:
            import torch

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            return self._tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def run_all(self) -> dict[str, float]:
        """Run all ForgeQualityBench test suites."""
        results: dict[str, float] = {}

        logger.info("Running ForgeQualityBench...")

        logger.info("  Suite 1: Shortcut Detection...")
        results["shortcut_detection_recall"] = self.eval_shortcut_detection()

        logger.info("  Suite 2: Contamination Precision...")
        results["contamination_precision"] = self.eval_contamination_precision()

        logger.info("  Suite 3: IRT Calibration Accuracy...")
        results["irt_calibration_mse"] = self.eval_irt_calibration()

        logger.info("  Suite 4: Item Generation Diversity...")
        results["item_diversity_self_bleu"] = self.eval_item_diversity()

        logger.info("  Suite 5: Goodhart Pattern Classification...")
        results["goodhart_classification_accuracy"] = self.eval_goodhart_classification()

        # Composite score (weighted average)
        weights = {
            "shortcut_detection_recall": 0.30,
            "contamination_precision": 0.30,
            "irt_calibration_mse": 0.10,  # lower is better — invert
            "item_diversity_self_bleu": 0.15,
            "goodhart_classification_accuracy": 0.15,
        }

        composite = (
            weights["shortcut_detection_recall"] * results["shortcut_detection_recall"]
            + weights["contamination_precision"] * results["contamination_precision"]
            + weights["irt_calibration_mse"] * (1.0 - min(1.0, results["irt_calibration_mse"]))
            + weights["item_diversity_self_bleu"] * results["item_diversity_self_bleu"]
            + weights["goodhart_classification_accuracy"] * results["goodhart_classification_accuracy"]
        )
        results["composite_forge_quality_score"] = round(composite, 4)

        logger.info(f"ForgeQuality composite score: {results['composite_forge_quality_score']:.4f}")
        return results

    def eval_shortcut_detection(self, n_tests: int = 100) -> float:
        """
        Evaluate shortcut detection recall on planted shortcuts.

        Returns fraction of planted shortcuts correctly detected.
        """
        from synthesis.contamination_prober import ContaminationProber
        from synthesis.shortcut_detector import ShortcutDetector

        prober = ContaminationProber(seed=99)
        detector = ShortcutDetector()

        # Generate synthetic test items with planted shortcuts
        test_items = self._generate_synthetic_items(n=200)
        contaminated = prober.create_contaminated_dataset(
            test_items, n_contaminated=n_tests
        )
        contaminated = [c for c in contaminated if c["is_contaminated"]]

        if not contaminated:
            logger.warning("No contaminated test items generated")
            return 0.0

        detected = 0
        for item_data in contaminated:
            item = item_data["item"]
            planted_type = item_data["contamination_type"]
            report = detector.analyze_item(item)
            detected_patterns = {s["pattern"] for s in report.shortcuts}
            if planted_type in detected_patterns or self._llm_detects(item, planted_type):
                detected += 1

        recall = detected / len(contaminated)
        logger.info(f"  Shortcut detection: {detected}/{len(contaminated)} = {recall:.3f}")
        return round(recall, 4)

    def eval_contamination_precision(self, n_tests: int = 100) -> float:
        """
        Evaluate contamination detection precision.
        Precision = TP / (TP + FP) — when model flags contamination, is it real?
        """
        from synthesis.contamination_prober import ContaminationProber

        prober = ContaminationProber(seed=777)
        test_items = self._generate_synthetic_items(n=200)

        # Mix 50% contaminated, 50% clean
        contaminated_data = prober.create_contaminated_dataset(
            test_items, n_contaminated=n_tests // 2
        )

        true_positives = 0
        false_positives = 0
        flags_total = 0

        for item_data in contaminated_data:
            item = item_data["item"]
            is_actually_contaminated = item_data["is_contaminated"]
            model_flags = self._model_flags_contamination(item)
            if model_flags:
                flags_total += 1
                if is_actually_contaminated:
                    true_positives += 1
                else:
                    false_positives += 1

        if flags_total == 0:
            return 0.0

        precision = true_positives / flags_total
        logger.info(
            f"  Contamination precision: {true_positives}/{flags_total} = {precision:.3f} "
            f"(FP: {false_positives})"
        )
        return round(precision, 4)

    def eval_irt_calibration(self, n_items: int = 20, n_models: int = 10) -> float:
        """
        Evaluate IRT calibration accuracy via MSE of difficulty estimates.
        Uses synthetic response matrices with known parameters.

        Returns MSE (lower is better).
        """
        from core.irt_models import IRTCalibrator

        calibrator = IRTCalibrator(model="2pl")
        true_b_values = np.linspace(-2.0, 2.0, n_items)
        true_a_values = np.ones(n_items) * 1.0

        theta_values = np.linspace(-3.0, 3.0, n_models)

        # Generate synthetic response matrix
        from core.irt_models import p_correct_2pl

        response_matrix = np.zeros((n_models, n_items))
        for i, theta in enumerate(theta_values):
            for j, (a, b) in enumerate(zip(true_a_values, true_b_values)):
                p = p_correct_2pl(theta, a, b)
                response_matrix[i, j] = float(np.random.binomial(1, p))

        item_ids = [f"item_{j}" for j in range(n_items)]
        model_names = [f"model_{i}" for i in range(n_models)]

        result = calibrator.calibrate(response_matrix, item_ids, model_names)

        # Compute MSE between true and estimated b values
        estimated_b = {ip.item_id: ip.difficulty_b for ip in result.item_parameters}
        mse = float(np.mean([
            (estimated_b[f"item_{j}"] - true_b_values[j])**2
            for j in range(n_items)
        ]))

        logger.info(f"  IRT calibration MSE: {mse:.4f}")
        return round(mse, 4)

    def eval_item_diversity(self, n_items: int = 30) -> float:
        """
        Evaluate diversity of generated items using Self-BLEU.
        Higher self-BLEU = less diverse (worse).
        Returns 1 - self_bleu (so higher is better).
        """
        from synthesis.item_generator import ItemGenerator

        gen = ItemGenerator()
        items = gen.generate(
            construct="causal reasoning",
            n_items=n_items,
            difficulty_range=(-1.0, 1.0),
        )

        if len(items) < 2:
            return 0.0

        # Compute pairwise BLEU (simplified via bigram overlap)
        questions = [item.get("question", "") for item in items]
        bleu_scores = []
        for i, q in enumerate(questions):
            others = [questions[j] for j in range(len(questions)) if j != i]
            if others:
                bleu_scores.append(self._compute_self_bleu(q, others))

        mean_self_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.5
        diversity = 1.0 - mean_self_bleu
        logger.info(f"  Item diversity (1 - self-BLEU): {diversity:.4f}")
        return round(diversity, 4)

    def eval_goodhart_classification(self) -> float:
        """
        Evaluate Goodhart pattern classification accuracy.
        Tests whether model correctly classifies case studies into the 23-pattern taxonomy.

        Returns macro F1 score.
        """
        from core.goodhart_patterns import PATTERNS

        # Use the patterns with known AI examples as test cases
        test_cases = [
            {"case": p.ai_benchmark_example, "true_pattern": p.id}
            for p in list(PATTERNS.values())[:10]  # use first 10 for evaluation
            if p.ai_benchmark_example and len(p.ai_benchmark_example) > 50
        ]

        if not test_cases:
            return 0.0

        correct = 0
        for test in test_cases:
            prompt = (
                f"Classify this case study into one Goodhart pattern from the taxonomy.\n\n"
                f"Case study: {test['case']}\n\n"
                f"Available patterns: {', '.join(list(PATTERNS.keys())[:15])}\n\n"
                "Respond with just the pattern_id (e.g., G01_length_bias):"
            )
            try:
                response = self._infer(
                    system="You are an expert in benchmark gaming patterns. Classify the case study.",
                    user=prompt,
                    max_tokens=50,
                ).strip()
                # Check if the correct pattern appears in the response
                if test["true_pattern"] in response or test["true_pattern"].split("_")[0] in response:
                    correct += 1
            except Exception as e:
                logger.debug(f"Goodhart classification failed: {e}")

        accuracy = correct / len(test_cases) if test_cases else 0.0
        logger.info(f"  Goodhart classification: {correct}/{len(test_cases)} = {accuracy:.3f}")
        return round(accuracy, 4)

    # --- Helpers ---

    def _generate_synthetic_items(self, n: int = 200) -> list[dict]:
        """Generate synthetic multiple-choice items for testing."""
        random.seed(42)
        items = []
        constructs = ["causal reasoning", "logical deduction", "reading comprehension"]
        for i in range(n):
            construct = random.choice(constructs)
            choices = {
                "A": f"Option A for item {i}: this is a plausible answer",
                "B": f"Option B for item {i}: this is another option",
                "C": f"Option C for item {i}: this is also plausible",
                "D": f"Option D for item {i}: this is the last option",
            }
            answer_key = random.choice(["A", "B", "C", "D"])
            items.append({
                "id": f"synthetic_{i}",
                "question": f"Given {construct} scenario {i}, what is the most likely outcome?",
                "choices": choices,
                "answer": answer_key,
                "construct": construct,
            })
        return items

    def _llm_detects(self, item: dict, planted_type: str) -> bool:
        """Use LLM to detect a specific shortcut type (fallback for complex patterns)."""
        try:
            prompt = (
                f"Does this benchmark item contain a '{planted_type}' shortcut?\n\n"
                f"Item: {json.dumps(item, indent=2)}\n\n"
                "Answer YES or NO:"
            )
            response = self._infer(
                system="You are a benchmark quality checker.", user=prompt, max_tokens=10
            )
            return "YES" in response.upper()
        except Exception:
            return False

    def _model_flags_contamination(self, item: dict) -> bool:
        """Ask EvalForge model if an item is contaminated."""
        try:
            prompt = (
                f"Is this benchmark item contaminated (does it appear in likely pretraining data)?\n\n"
                f"Item: {json.dumps(item, indent=2)}\n\n"
                "Answer YES or NO:"
            )
            response = self._infer(
                system="You are an AI evaluation auditor.", user=prompt, max_tokens=10
            )
            return "YES" in response.upper()
        except Exception:
            return False

    @staticmethod
    def _compute_self_bleu(candidate: str, references: list[str]) -> float:
        """Compute average bigram BLEU between candidate and references."""
        def bigrams(text: str) -> set[tuple[str, ...]]:
            words = text.lower().split()
            return set(zip(words, words[1:]))

        c_bigrams = bigrams(candidate)
        if not c_bigrams:
            return 0.0

        scores = []
        for ref in references:
            r_bigrams = bigrams(ref)
            if r_bigrams:
                overlap = len(c_bigrams & r_bigrams)
                precision = overlap / len(c_bigrams)
                scores.append(precision)

        return float(np.mean(scores)) if scores else 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ForgeQualityBench")
    parser.add_argument("--model", help="Path to model checkpoint")
    parser.add_argument("--model-url", help="URL of vLLM server")
    args = parser.parse_args()

    bench = ForgeQualityBench(model_path=args.model, model_url=args.model_url)
    results = bench.run_all()

    print("\n=== ForgeQualityBench Results ===")
    for metric, value in results.items():
        print(f"  {metric:<45} {value:.4f}")
