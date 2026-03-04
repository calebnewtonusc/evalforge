"""
training/train.py — Stage 1: SFT fine-tuning of EvalForge on Qwen2.5-7B-Coder-Instruct.

Hardware target: 18× A6000 (48GB) — use GPUs 8–17 for training
Strategy: LoRA rank 64, DeepSpeed ZeRO-3, Flash Attention 2

Launch (single node, 10 GPUs):
    CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15,16,17 \\
    deepspeed --num_gpus=10 training/train.py \\
      --data-dir data/train \\
      --output-dir checkpoints/evalforge-sft \\
      --deepspeed training/configs/deepspeed_zero3.json

Or via pipeline:
    python pipeline.py --stage train
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


def load_sharegpt_dataset(jsonl_path: str) -> Dataset:
    """Load ShareGPT-format JSONL file as HuggingFace Dataset."""
    records: list[dict] = []
    for line in Path(jsonl_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    logger.info(f"Loaded {len(records):,} records from {jsonl_path}")
    return Dataset.from_list(records)


def format_to_text(example: dict, tokenizer) -> dict:
    """
    Convert ShareGPT conversation format to a single text string
    using the model's chat template.
    """
    conversations = example.get("conversations", [])
    system_msgs = [c for c in conversations if c["from"] == "system"]
    user_msgs = [
        {"role": "user" if c["from"] == "human" else "assistant", "content": c["value"]}
        for c in conversations
        if c["from"] in ("human", "gpt")
    ]
    messages = []
    if system_msgs:
        messages.append({"role": "system", "content": system_msgs[0]["value"]})
    messages.extend(user_msgs)

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


class LogMetricsCallback(TrainerCallback):
    """Log training metrics at each logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            loss = logs.get("loss", "—")
            lr = logs.get("learning_rate", "—")
            grad_norm = logs.get("grad_norm", "—")
            if isinstance(loss, float):
                logger.info(
                    f"step {step:>6} | loss {loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm}"
                )


def build_lora_config(lora_r: int = 64) -> LoraConfig:
    """Build LoRA config targeting all attention + MLP projection layers."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_r * 2,  # alpha/r = 2x
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EvalForge Stage 1: SFT training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Coder-Instruct")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--output-dir", default="checkpoints/evalforge-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per-device batch size"
    )
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config JSON")
    parser.add_argument("--flash-attn", action="store_true", default=True)
    parser.add_argument("--wandb-project", default="evalforge")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_file = data_dir / "evalforge_train.jsonl"
    val_file = data_dir / "evalforge_val.jsonl"

    if not train_file.exists():
        logger.error(
            f"Training file not found: {train_file}. Run pipeline.py --stage synthesis first."
        )
        return

    logger.info(f"Loading base model: {args.model}")
    model_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": None}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)  # nosec B615
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.model, trust_remote_code=True, **model_kwargs
    )
    model.enable_input_require_grads()

    # Apply LoRA
    lora_cfg = build_lora_config(args.lora_r)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load datasets
    logger.info("Loading datasets...")
    train_ds = load_sharegpt_dataset(str(train_file))
    val_ds_raw = load_sharegpt_dataset(str(val_file)) if val_file.exists() else None
    # Treat a 0-byte / empty JSONL as no validation set to avoid trainer errors
    val_ds = val_ds_raw if (val_ds_raw is not None and len(val_ds_raw) > 0) else None

    train_ds = train_ds.map(lambda ex: format_to_text(ex, tokenizer))
    if val_ds:
        val_ds = val_ds.map(lambda ex: format_to_text(ex, tokenizer))

    n_gpus = torch.cuda.device_count() or 1
    effective_batch = args.batch_size * args.grad_accum * n_gpus
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    total_steps = steps_per_epoch * args.epochs

    logger.info(
        f"GPUs: {n_gpus} | Effective batch: {effective_batch} | Total steps: {total_steps:,}"
    )
    logger.info(
        f"Train size: {len(train_ds):,} | Val size: {len(val_ds) if val_ds else 0:,}"
    )

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    # Use SFTConfig (subclass of TrainingArguments) so that SFT-specific
    # parameters like max_seq_length and dataset_text_field are passed through
    # the config object rather than the SFTTrainer constructor (which no longer
    # accepts them as top-level kwargs in recent TRL versions).
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=max(1, steps_per_epoch // 4) if val_ds else None,
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch // 2),
        save_total_limit=3,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss" if val_ds else None,
        report_to=report_to,
        run_name="evalforge-sft",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        max_seq_length=args.max_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[LogMetricsCallback()],
    )

    logger.info(f"Starting SFT training → {args.output_dir}")
    trainer.train()

    # Merge LoRA and save
    final_dir = Path(args.output_dir) / "final"
    logger.info(f"Merging LoRA weights → {final_dir}")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"SFT complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
