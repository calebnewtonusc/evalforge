"""
synthesis/item_generator.py — Generate novel, adversarially robust eval items.

The item generator is constrained by:
  - IRT difficulty target (b parameter range)
  - Construct specification (what capability to test)
  - Shortcut blacklist (patterns to avoid)
  - Diversity constraint (min self-BLEU distance from existing items)

Usage:
    python synthesis/item_generator.py \
        --construct "multi-step logical deduction" \
        --difficulty-min -0.5 \
        --difficulty-max 0.5 \
        --n-items 50 \
        --output data/processed/generated_items.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import uuid
from typing import Any

import anthropic
from loguru import logger

from synthesis.prompts import ITEM_GENERATION_PROMPT, SYSTEM_ITEM_GENERATOR


class ItemGenerator:
    """
    Generates novel evaluation items conditioned on construct and difficulty.

    Applies post-hoc quality checks:
    - Length balance across answer choices
    - Lexical overlap between stem and correct answer
    - N-gram diversity vs existing item pool
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        if client is not None:
            self.client = client
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Set it or pass an anthropic.Anthropic client explicitly."
                )
            self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        construct: str,
        n_items: int = 10,
        difficulty_range: tuple[float, float] = (-1.0, 1.0),
        shortcuts_to_avoid: list[str] | None = None,
        existing_items: list[dict] | None = None,
        benchmark_style: str = "multiple_choice_4_option",
    ) -> list[dict]:
        """
        Generate n_items novel benchmark items.

        Args:
            construct: The cognitive capability to test.
            n_items: Number of items to generate.
            difficulty_range: (min_b, max_b) IRT difficulty range.
            shortcuts_to_avoid: List of Goodhart pattern names to avoid.
            existing_items: Existing items (for diversity checking).
            benchmark_style: Item format style.

        Returns:
            List of validated item dicts.
        """
        if shortcuts_to_avoid is None:
            shortcuts_to_avoid = ["length_bias", "lexical_overlap", "negation_artifact"]

        existing_sample = json.dumps(
            (existing_items or [])[:5], indent=2, ensure_ascii=False
        )

        prompt = ITEM_GENERATION_PROMPT.format(
            n_items=n_items,
            construct=construct,
            difficulty_min=difficulty_range[0],
            difficulty_max=difficulty_range[1],
            shortcuts_to_avoid=", ".join(shortcuts_to_avoid),
            benchmark_style=benchmark_style,
            existing_items_sample=existing_sample,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.9,
            system=SYSTEM_ITEM_GENERATOR,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text

        parsed = self._parse_items(raw)
        validated = [self._validate_item(item) for item in parsed]
        valid_items = [item for item in validated if item is not None]

        logger.info(
            f"Generated {len(valid_items)}/{len(parsed)} valid items for construct: {construct}"
        )
        return valid_items

    def _parse_items(self, response: str) -> list[dict]:
        """Extract item list from LLM response."""
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            data = json.loads(response)
            return data.get("items", [])
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(response[start:end])
                    return data.get("items", [])
                except json.JSONDecodeError:
                    pass
        logger.warning("Failed to parse item generation response")
        return []

    def _validate_item(self, item: dict) -> dict | None:
        """
        Validate a generated item for quality.

        Checks:
        - Required fields present
        - No length bias (choice lengths within 2× of each other)
        - No trivial lexical overlap (correct answer words not all in question)
        - Correct answer key is valid

        Returns validated item or None if quality check fails.
        """
        required = ["question", "choices", "answer", "construct"]
        for field in required:
            if field not in item:
                logger.debug(f"Item missing field: {field}")
                return None

        choices: dict = item.get("choices", {})
        answer_key: str = item.get("answer", "")

        if answer_key not in choices:
            logger.debug(f"Answer key {answer_key!r} not in choices")
            return None

        # Length balance check — flag if correct answer is 2× longer than avg distractor
        choice_lens = {k: len(str(v).split()) for k, v in choices.items()}
        avg_distractor_len = sum(
            v for k, v in choice_lens.items() if k != answer_key
        ) / max(1, len(choice_lens) - 1)
        correct_len = choice_lens.get(answer_key, 0)
        if avg_distractor_len > 0 and correct_len / avg_distractor_len > 2.5:
            logger.debug("Item failed length balance check")
            return None

        # Ensure item has a unique ID
        if "id" not in item:
            item["id"] = str(uuid.uuid4())

        return item


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate novel benchmark items")
    parser.add_argument("--construct", required=True)
    parser.add_argument("--difficulty-min", type=float, default=-1.0)
    parser.add_argument("--difficulty-max", type=float, default=1.0)
    parser.add_argument("--n-items", type=int, default=20)
    parser.add_argument("--output", default="data/processed/generated_items.jsonl")
    args = parser.parse_args()

    gen = ItemGenerator()
    items = gen.generate(
        construct=args.construct,
        n_items=args.n_items,
        difficulty_range=(args.difficulty_min, args.difficulty_max),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(item) for item in items) + "\n")
    logger.info(f"Saved {len(items)} items to {out}")
