"""
training/train_dpo.py — Stage 3: DPO preference optimization for EvalForge.

Trains on expert-curated preferences for:
  - Audit report quality (specific vs. vague)
  - Item generation diversity (novel vs. paraphrase)
  - Contamination detection precision (cautious vs. trigger-happy)

Launch:
    CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15,16,17 \\
    deepspeed --num_gpus=10 training/train_dpo.py \\
      --model checkpoints/evalforge-rl \\
      --dpo-data data/train/dpo_pairs.jsonl \\
      --output-dir checkpoints/evalforge-final \\
      --deepspeed training/configs/deepspeed_zero3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_dpo_dataset(dpo_file: str | Path) -> Dataset:
    """
    Load DPO preference pairs.

    Expected format:
    {
        "prompt": "<system + user turn>",
        "chosen": "<preferred response>",
        "rejected": "<rejected response>",
        "preference_reason": "<human annotation rationale>"
    }
    """
    records: list[dict] = []
    for line in Path(dpo_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            # Validate required fields
            if all(k in record for k in ("prompt", "chosen", "rejected")):
                records.append(record)
        except json.JSONDecodeError:
            pass

    logger.info(f"Loaded {len(records):,} DPO pairs from {dpo_file}")
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="EvalForge Stage 3: DPO training")
    parser.add_argument("--model", default="checkpoints/evalforge-rl")
    parser.add_argument("--dpo-data", default="data/train/dpo_pairs.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/evalforge-final")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--deepspeed", type=str)
    args = parser.parse_args()

    dpo_file = Path(args.dpo_data)
    if not dpo_file.exists():
        logger.error(
            f"DPO data not found at {dpo_file}. "
            "Generate DPO pairs via human review or automated preference mining."
        )
        return

    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)  # nosec B615
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_ds = load_dpo_dataset(dpo_file)

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        report_to="none",
        deepspeed=args.deepspeed,
    )

    # Load both policy and reference models explicitly as objects so that
    # DeepSpeed can manage them consistently (passing a string path for the
    # policy while the ref is already an object causes device placement
    # conflicts under ZeRO-3).
    logger.info("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.model, torch_dtype=torch.bfloat16, device_map=None
    )

    # Load a frozen reference model so DPO has a proper KL anchor.
    # Using ref_model=None would make the policy serve as its own reference,
    # eliminating the KL constraint and destabilizing training.
    logger.info("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.model, torch_dtype=torch.bfloat16, device_map=None
    )
    ref_model.eval()

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    logger.info(f"Starting DPO training → {args.output_dir}")
    trainer.train()

    logger.info(f"Saving DPO model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info(f"DPO complete. Final model saved to {args.output_dir}")
    logger.info("EvalForge training pipeline complete.")
    logger.info(
        f"To serve: vllm serve {args.output_dir} --port 9000 --gpu-memory-utilization 0.9"
    )


if __name__ == "__main__":
    main()
