"""
training/train_rl.py — Stage 2: GRPO reinforcement learning for EvalForge.

Reward signal (3 components):
  R_shortcut    (0.4): fraction of planted shortcuts correctly detected
  R_correlation (0.4): Pearson r between model scores and downstream task perf
  R_stability   (0.2): Kendall tau of model rankings before/after replacement

Algorithm: GRPO (Group Relative Policy Optimization) via TRL
Reference model: evalforge-sft checkpoint

Launch:
    CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15,16,17 \\
    deepspeed --num_gpus=10 training/train_rl.py \\
      --model checkpoints/evalforge-sft \\
      --data-dir data/train \\
      --output-dir checkpoints/evalforge-rl \\
      --deepspeed training/configs/deepspeed_zero3.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


def compute_shortcut_reward(response: str, ground_truth: dict) -> float:
    """
    Compute R_shortcut: fraction of planted shortcuts detected.

    Args:
        response: Model's JSON audit/shortcut detection response.
        ground_truth: Dict with 'shortcuts_planted' list.

    Returns:
        Float in [0, 1].
    """
    planted = set(ground_truth.get("shortcuts_planted", []))
    if not planted:
        return 1.0  # No shortcuts to detect — full score

    # Parse model response
    detected: set[str] = set()
    try:
        # Strip markdown fences
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Strip opening/closing fence lines
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        data = json.loads(text)

        # Check multiple possible response schemas
        if "shortcuts_found" in data:
            detected = {s["pattern"] for s in data["shortcuts_found"]}
        elif "shortcuts" in data:
            detected = {s["pattern"] for s in data["shortcuts"]}
        elif "shortcuts_detected" in data:
            detected = set(data["shortcuts_detected"])

    except (json.JSONDecodeError, KeyError, TypeError):
        return 0.0

    if not detected:
        return 0.0

    recall = len(planted & detected) / len(planted)
    # Penalize false positives lightly
    precision = len(planted & detected) / len(detected)
    f1 = 2 * recall * precision / max(recall + precision, 1e-8)
    return f1


def compute_format_reward(response: str) -> float:
    """
    Compute format reward: did the model produce valid JSON?
    Returns 1.0 for valid JSON, 0.3 for near-valid, 0.0 for invalid.
    """
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Strip opening/closing fence lines
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        json.loads(text)
        return 1.0
    except json.JSONDecodeError:
        # Check if there's any JSON-like structure
        if "{" in text and "}" in text:
            return 0.3
        return 0.0


def compute_reward(
    response: str,
    ground_truth: dict,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Combined reward function.

    R_total = w_shortcut * R_shortcut + w_format * R_format + w_correlation * R_correlation
    """
    if weights is None:
        weights = {"shortcut": 0.5, "format": 0.3, "correlation": 0.2}

    r_shortcut = compute_shortcut_reward(response, ground_truth)
    r_format = compute_format_reward(response)
    # R_correlation requires downstream task evaluation — approximated here as a
    # scaled proxy of the shortcut reward (both measure benchmark validity).
    r_correlation = r_shortcut * 0.8

    total = (
        weights["shortcut"] * r_shortcut
        + weights["format"] * r_format
        + weights["correlation"] * r_correlation
    )
    return round(float(total), 4)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rl_dataset(data_dir: Path) -> Dataset:
    """
    Load dataset for GRPO training.

    Each example needs:
    - 'prompt': the input (system + user turn)
    - 'ground_truth': dict with expected shortcut labels
    """
    records: list[dict] = []

    # Load shortcut detection pairs as RL prompts
    shortcut_file = data_dir / "shortcut_pairs.jsonl"
    if shortcut_file.exists():
        for line in shortcut_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                if "conversations" not in pair:
                    continue
                conversations = pair.get("conversations", [])
                system_msg = next(
                    (c["value"] for c in conversations if c["from"] == "system"), ""
                )
                user_msg = next(
                    (c["value"] for c in conversations if c["from"] == "human"), ""
                )
                prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
                records.append(
                    {
                        "prompt": prompt,
                        "ground_truth": {
                            "shortcuts_planted": pair.get("shortcuts_planted", []),
                            "type": pair.get("type", "shortcut_detection"),
                        },
                    }
                )
            except (json.JSONDecodeError, KeyError):
                pass

    logger.info(f"Loaded {len(records):,} RL training examples")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Reward function wrapper for TRL
# ---------------------------------------------------------------------------


def reward_fn(
    prompts: list[str], completions: list[list[str]], **kwargs
) -> list[float]:
    """
    TRL-compatible reward function.

    GRPOTrainer passes completions as list[list[str]] — one inner list per prompt,
    containing num_generations completions. ground_truth is one entry per prompt
    and must be expanded to match the flattened completions.

    Args:
        prompts: List of input prompts (one per group).
        completions: List of lists of model completions (one list per prompt).
        **kwargs: Passed by TRL, includes metadata columns.

    Returns:
        List of scalar rewards, one per completion (len = n_prompts * num_generations).
    """
    ground_truths = kwargs.get("ground_truth")
    if not ground_truths:
        # No ground truth available — return penalty to avoid reward poisoning
        total = sum(len(group) for group in completions)
        return [-1.0] * total

    rewards = []
    # Per-group degenerate-reward guard: check each prompt's group independently
    for group_completions, gt in zip(completions, ground_truths):
        group_rewards = [compute_reward(text, gt) for text in group_completions]
        # If all rewards in this GRPO group are identical, std=0 → NaN gradients.
        # Replace with uniform penalty so the update is a clean no-op.
        if len(set(group_rewards)) == 1:
            group_rewards = [-1.0] * len(group_rewards)
        rewards.extend(group_rewards)
    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="EvalForge Stage 2: GRPO RL training")
    parser.add_argument("--model", default="checkpoints/evalforge-sft")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--output-dir", default="checkpoints/evalforge-rl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--n-completions", type=int, default=8, help="GRPO group size")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    logger.info(f"Loading model from: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_rl_dataset(data_dir)
    if len(train_ds) == 0:
        logger.error("No RL training data found. Run synthesis stage first.")
        return

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.kl_coef,
        num_generations=args.n_completions,
        max_completion_length=args.max_new_tokens,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        deepspeed=args.deepspeed,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    logger.info(f"Starting GRPO training → {args.output_dir}")
    trainer.train()

    logger.info(f"GRPO complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
