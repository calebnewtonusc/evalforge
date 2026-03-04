#!/bin/bash
# check_env.sh — Validate EvalForge environment before running pipeline

set -euo pipefail

ERRORS=0
WARNINGS=0

check() {
    local label="$1"
    local condition="$2"
    local msg="$3"

    if eval "$condition" > /dev/null 2>&1; then
        echo "  [OK] $label"
    else
        echo "  [FAIL] $label — $msg"
        ERRORS=$((ERRORS + 1))
    fi
}

warn() {
    local label="$1"
    local condition="$2"
    local msg="$3"

    if eval "$condition" > /dev/null 2>&1; then
        echo "  [OK] $label"
    else
        echo "  [WARN] $label — $msg"
        WARNINGS=$((WARNINGS + 1))
    fi
}

echo "=== EvalForge Environment Check ==="
echo ""

# Python version
echo "[Python]"
check "Python 3.11+" \
    "python3 -c \"import sys; assert sys.version_info >= (3, 11)\"" \
    "Python 3.11+ required"

# Required packages
echo ""
echo "[Python packages]"
for pkg in torch transformers peft trl deepspeed datasets loguru scipy numpy; do
    check "$pkg" "python3 -c 'import $pkg'" "pip install $pkg"
done

# CUDA
echo ""
echo "[GPU/CUDA]"
check "CUDA available" \
    "python3 -c \"import torch; assert torch.cuda.is_available()\"" \
    "No CUDA detected — training requires GPUs"

warn "CUDA 12.1+" \
    "python3 -c \"import torch; v=torch.version.cuda; assert v and int(v.split('.')[0]) >= 12\"" \
    "CUDA 12.1+ recommended for Flash Attention 2"

warn "10+ GPUs" \
    "python3 -c \"import torch; assert torch.cuda.device_count() >= 10\"" \
    "Full training needs 10+ GPUs; you have $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"

warn "Flash Attention 2" \
    "python3 -c 'import flash_attn'" \
    "Recommended: pip install flash-attn --no-build-isolation"

# Environment variables
echo ""
echo "[Environment variables]"
check "ANTHROPIC_API_KEY" \
    "[ -n \"\${ANTHROPIC_API_KEY:-}\" ]" \
    "Set ANTHROPIC_API_KEY in .env"

warn "VLLM_API_KEY" \
    "[ -n \"\${VLLM_API_KEY:-}\" ]" \
    "Set VLLM_API_KEY in .env for vLLM synthesis"

warn "WANDB_API_KEY" \
    "[ -n \"\${WANDB_API_KEY:-}\" ]" \
    "Optional: set WANDB_API_KEY for training monitoring"

# Disk space
echo ""
echo "[Storage]"
AVAILABLE_GB=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo 0)
if [ "$AVAILABLE_GB" -ge 500 ] 2>/dev/null; then
    echo "  [OK] Disk space (${AVAILABLE_GB}GB available)"
else
    echo "  [WARN] Disk space — ${AVAILABLE_GB}GB available, recommend 500GB+"
    WARNINGS=$((WARNINGS + 1))
fi

# RAM
echo ""
echo "[Memory]"
AVAILABLE_RAM_GB=$(awk '/MemTotal/ {printf "%d\n", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$AVAILABLE_RAM_GB" -ge 256 ] 2>/dev/null; then
    echo "  [OK] RAM (${AVAILABLE_RAM_GB}GB)"
else
    echo "  [WARN] RAM — ${AVAILABLE_RAM_GB}GB detected, ZeRO-3 recommends 512GB+"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""
echo "==================================="
if [ $ERRORS -eq 0 ]; then
    echo "Environment OK ($WARNINGS warnings)"
    exit 0
else
    echo "FAILED: $ERRORS error(s), $WARNINGS warning(s)"
    echo "Fix errors before running pipeline."
    exit 1
fi
