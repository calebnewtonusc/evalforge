# EvalForge — 18× A6000 GPU Setup

This guide covers setting up EvalForge training on an 18× NVIDIA A6000 (48GB) cluster.

---

## Hardware Configuration

```
Total VRAM:     18 × 48GB = 864GB
GPU allocation:
  GPUs 0-7:    vLLM synthesis (2 instances × 4 GPUs, Qwen2.5-72B)
  GPUs 8-17:   Training (10 GPUs, DeepSpeed ZeRO-3)

Host RAM:       512GB+ required (ZeRO-3 offloads optimizer states to CPU)
Storage:        2TB NVMe SSD (raw corpus ~200GB, processed ~50GB, model ~30GB)
Interconnect:   NVLink or PCIe Gen4 x16 (NVLink strongly preferred)
```

---

## Driver and CUDA Setup

```bash
# Verify GPU visibility
nvidia-smi

# Required: CUDA 12.1+
nvcc --version

# Install CUDA 12.2 if needed (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-12-2
```

---

## Python Environment

```bash
# Python 3.11
conda create -n evalforge python=3.11 -y
conda activate evalforge

# Core dependencies
pip install -r requirements.txt

# Flash Attention 2 (significant speedup, requires CUDA 11.6+)
pip install flash-attn --no-build-isolation

# Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); print(f'CUDA: {torch.version.cuda}')"
```

---

## DeepSpeed ZeRO-3 Configuration

The training config at `training/configs/deepspeed_zero3.json` is tuned for 18× A6000:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e6
  }
}
```

Key tuning notes:
- `offload_optimizer: cpu` with `pin_memory: true` — required for 7B on A6000 with ZeRO-3
- `reduce_bucket_size: 5e8` — large bucket size reduces communication overhead on NVLink
- Increase `stage3_prefetch_bucket_size` if you have headroom (not memory bound)

---

## vLLM Synthesis Setup (GPUs 0–7)

Run synthesis before training to generate the full 400k+ pair dataset.

```bash
# Instance 1 — GPUs 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --port 8001 \
  --api-key $VLLM_API_KEY \
  --max-model-len 8192 &

# Instance 2 — GPUs 4,5,6,7
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 \
  --port 8002 \
  --api-key $VLLM_API_KEY \
  --max-model-len 8192 &

# Or use the convenience script
bash scripts/start_vllm.sh
```

---

## Training Launch

### Stage 1 — SFT

```bash
# 10 GPUs (8-17), ZeRO-3, LoRA rank 64
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15,16,17 \
deepspeed --num_gpus=10 training/train.py \
  --model Qwen/Qwen2.5-7B-Coder-Instruct \
  --data-dir data/train \
  --output-dir checkpoints/evalforge-sft \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --lora-r 64 \
  --max-length 4096 \
  --deepspeed training/configs/deepspeed_zero3.json \
  --flash-attn

# Expected: ~4-5 hours, ~3GB/GPU memory with ZeRO-3
```

### Stage 2 — GRPO

```bash
# Reward model needs 4 GPUs for evaluation; training on remaining 14
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15,16,17 \
deepspeed --num_gpus=10 training/train_rl.py \
  --model checkpoints/evalforge-sft \
  --output-dir checkpoints/evalforge-rl \
  --reward-gpus 0,1,2,3 \
  --deepspeed training/configs/deepspeed_zero3.json

# Expected: ~3 hours
```

### Stage 3 — DPO

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15,16,17 \
deepspeed --num_gpus=10 training/train_dpo.py \
  --model checkpoints/evalforge-rl \
  --dpo-data data/train/dpo_pairs.jsonl \
  --output-dir checkpoints/evalforge-final \
  --deepspeed training/configs/deepspeed_zero3.json

# Expected: ~1 hour
```

---

## Environment Validation

```bash
bash scripts/check_env.sh
```

Checks:
- Python 3.11+
- CUDA 12.1+
- All required env vars (.env loaded)
- GPU count >= 10
- RAM >= 256GB
- Disk space >= 500GB
- All Python packages importable
- vLLM instances reachable (if running)

---

## Memory Estimates

| Stage | GPUs | VRAM/GPU | CPU RAM |
|-------|------|----------|---------|
| SFT (LoRA rank 64) | 10× A6000 | ~28GB | ~120GB |
| GRPO | 10× A6000 | ~32GB | ~140GB |
| DPO | 10× A6000 | ~28GB | ~120GB |
| vLLM synthesis (72B, TP=4) | 4× A6000 | ~46GB | ~40GB |

---

## Monitoring

```bash
# Live GPU utilization
watch -n 1 nvidia-smi

# DeepSpeed logs
tail -f checkpoints/evalforge-sft/training.log

# WandB (if configured)
wandb online  # set WANDB_API_KEY in .env
```
