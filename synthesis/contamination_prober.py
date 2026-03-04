"""
synthesis/contamination_prober.py — Plant synthetic contamination for training data.

Creates ground-truth labeled (item, contamination_type) pairs by:
  1. Taking clean benchmark items
  2. Injecting known contamination patterns (n-gram overlap, template fill, paraphrase)
  3. Labeling the contamination type and severity

This produces the ground-truth signal needed for the GRPO reward:
  R_shortcut = fraction of planted contamination detected by EvalForge

Usage:
    python synthesis/contamination_prober.py \
        --input data/raw/benchmarks/mmlu/items.jsonl \
        --output data/processed/contamination_pairs.jsonl \
        --n-contaminated 5000
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

from loguru import logger


CONTAMINATION_TYPES = [
    "ngram_overlap",  # verbatim n-gram from training data in item
    "template_fill",  # fill-in-blank structure resolvable by pattern
    "answer_leak",  # correct answer is recoverable from common web text
]

SHORTCUT_TYPES = [
    # Only types that _inject_contamination() actually handles — other types
    # cause it to fall through and return (None, 0.0), producing false
    # shortcuts_planted=[] even when a shortcut was intended.
    "length_bias",  # correct answer is systematically longer
    "position_bias",  # correct answer always at position A or D
    "ngram_overlap",  # verbatim n-gram from training data in item
    "template_fill",  # fill-in-blank structure resolvable by pattern
    "answer_leak",  # correct answer is recoverable from common web text
]


class ContaminationProber:
    """Plants and labels contamination in benchmark items for training data."""

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)

    def create_contaminated_dataset(
        self,
        clean_items: list[dict],
        n_contaminated: int = 5000,
        contamination_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Create a labeled dataset of clean and contaminated items.

        Args:
            clean_items: Original clean benchmark items.
            n_contaminated: Number of contaminated items to generate.
            contamination_types: Types to inject (default: all types).

        Returns:
            List of dicts with 'item', 'contamination_label', 'is_contaminated'.
        """
        if contamination_types is None:
            contamination_types = CONTAMINATION_TYPES

        results: list[dict] = []

        # Add clean items (labeled as clean)
        for item in random.sample(
            clean_items, min(len(clean_items), n_contaminated // 2)
        ):
            results.append(
                {
                    "id": str(uuid.uuid4()),
                    "item": item,
                    "is_contaminated": False,
                    "contamination_type": None,
                    "contamination_severity": 0.0,
                    "planted": False,
                }
            )

        # Generate contaminated items
        for i in range(n_contaminated):
            source_item = random.choice(clean_items)
            c_type = random.choice(contamination_types)
            contaminated, severity = self._inject_contamination(source_item, c_type)
            if contaminated is not None:
                results.append(
                    {
                        "id": str(uuid.uuid4()),
                        "item": contaminated,
                        "is_contaminated": True,
                        "contamination_type": c_type,
                        "contamination_severity": severity,
                        "planted": True,  # ground-truth label
                    }
                )

        random.shuffle(results)
        logger.info(
            f"Created {len(results)} items: "
            f"{sum(1 for r in results if r['is_contaminated'])} contaminated, "
            f"{sum(1 for r in results if not r['is_contaminated'])} clean"
        )
        return results

    def _inject_contamination(
        self, item: dict, contamination_type: str
    ) -> tuple[dict | None, float]:
        """
        Inject a specific contamination pattern into an item.

        Returns:
            (contaminated_item, severity) or (None, 0) if injection failed.
        """
        item = dict(item)
        item_content = item.get("item", item)

        if contamination_type == "ngram_overlap":
            return self._inject_ngram_overlap(item_content), 0.8

        elif contamination_type == "template_fill":
            return self._inject_template_fill(item_content), 0.7

        elif contamination_type == "answer_leak":
            return self._inject_answer_leak(item_content), 0.9

        elif contamination_type == "length_bias":
            return self._inject_length_bias(item_content), 0.6

        elif contamination_type == "position_bias":
            return self._inject_position_bias(item_content), 0.5

        return None, 0.0

    def _inject_ngram_overlap(self, item: dict) -> dict:
        """Inject verbatim text that would appear in training data."""
        item = dict(item)
        question = item.get("question", "")
        # Inject a verbatim Wikipedia-style phrase into the question
        common_phrases = [
            "According to widely available sources,",
            "As documented in standard references,",
            "The established consensus is that",
        ]
        phrase = random.choice(common_phrases)
        item["question"] = f"{phrase} {question}"
        item["_contamination_note"] = "ngram_overlap: verbatim common phrase injected"
        return item

    def _inject_template_fill(self, item: dict) -> dict:
        """Make item resolvable as a template fill-in."""
        item = dict(item)
        question = item.get("question", "")
        # Simplify to a fill-in structure
        choice_values = list(item.get("choices", {}).values())
        choice_values[0] if choice_values else ""
        item["question"] = f"The answer to '{question[:50]}...' is: ___"
        item["_contamination_note"] = "template_fill: question restructured as fill-in"
        return item

    def _inject_answer_leak(self, item: dict) -> dict:
        """Make the answer recoverable from the question text."""
        item = dict(item)
        choices = item.get("choices", {})
        answer_key = item.get("answer", "A")
        correct = choices.get(answer_key, "")
        # Embed the answer keyword in the question
        if correct:
            words = correct.split()[:3]
            embed = " ".join(words)
            item["question"] = (
                item.get("question", "") + f" (Note: {embed} is relevant.)"
            )
            item["_contamination_note"] = (
                "answer_leak: correct answer keywords embedded in question"
            )
        return item

    def _inject_length_bias(self, item: dict) -> dict:
        """Make correct answer systematically longer than distractors."""
        item = dict(item)
        choices = dict(item.get("choices", {}))
        answer_key = item.get("answer", "A")
        if answer_key in choices:
            choices[answer_key] = choices[answer_key] + (
                " This is the correct and complete answer providing full detail and explanation."
            )
            item["choices"] = choices
            item["_contamination_note"] = (
                "length_bias: correct answer padded to be longest"
            )
        return item

    def _inject_position_bias(self, item: dict) -> dict:
        """Move correct answer to a predictable position."""
        item = dict(item)
        choices = dict(item.get("choices", {}))
        answer_key = item.get("answer", "A")
        if len(choices) == 4 and answer_key in choices:
            correct_text = choices[answer_key]
            # Force correct answer to position A
            keys = ["A", "B", "C", "D"]
            other_texts = [v for k, v in choices.items() if k != answer_key]
            new_choices = {"A": correct_text}
            for k, text in zip(keys[1:], other_texts):
                new_choices[k] = text
            item["choices"] = new_choices
            item["answer"] = "A"
            item["_contamination_note"] = (
                "position_bias: correct answer moved to position A"
            )
        return item

    def create_shortcut_detection_pairs(
        self, items: list[dict], n_pairs: int = 2000
    ) -> list[dict]:
        """
        Create (items_with_shortcuts, detection_labels) pairs.

        Returns pairs where the ground truth is which shortcut patterns
        are present in the item set.
        """
        pairs = []
        for _ in range(n_pairs):
            # Sample 5-15 items, inject 0-3 shortcuts
            sample_size = random.randint(5, 15)
            sample = random.sample(items, min(sample_size, len(items)))

            shortcuts_planted = []
            modified_sample = []
            for item in sample:
                item_content = item.get("item", item)
                if random.random() < 0.3:  # 30% chance of shortcut
                    shortcut = random.choice(SHORTCUT_TYPES)
                    contaminated, _ = self._inject_contamination(item_content, shortcut)
                    if contaminated is not None:
                        shortcuts_planted.append(shortcut)
                        modified_sample.append(contaminated)
                        continue
                modified_sample.append(item_content)

            pairs.append(
                {
                    "id": str(uuid.uuid4()),
                    "items": modified_sample,
                    "shortcuts_planted": list(set(shortcuts_planted)),
                    "n_shortcuts": len(set(shortcuts_planted)),
                }
            )

        return pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-contaminated", type=int, default=5000)
    args = parser.parse_args()

    items = [
        json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()
    ]
    prober = ContaminationProber()
    contaminated = prober.create_contaminated_dataset(
        items, n_contaminated=args.n_contaminated
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(c) for c in contaminated) + "\n")
    logger.info(f"Saved {len(contaminated)} labeled items to {out}")
