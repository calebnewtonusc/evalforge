#!/bin/bash
# start_vllm.sh — Start 4 vLLM synthesis servers on 16 GPUs
# Instance 1: GPUs 0-3   → port 8001
# Instance 2: GPUs 4-7   → port 8002
# Instance 3: GPUs 8-11  → port 8003
# Instance 4: GPUs 12-15 → port 8004

set -euo pipefail

VLLM_MODEL="${VLLM_SYNTHESIS_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
VLLM_API_KEY="${VLLM_API_KEY:-evalforge-secret}"
mkdir -p logs

echo "Starting EvalForge vLLM synthesis servers..."
echo "Model: $VLLM_MODEL"

# Instance 1 — GPUs 0,1,2,3
echo "Starting instance 1 on GPUs 0-3 → port 8001"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
	vllm serve "$VLLM_MODEL" \
	--tensor-parallel-size 4 \
	--port 8001 \
	--api-key "$VLLM_API_KEY" \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.92 \
	--disable-log-requests \
	>logs/vllm_instance1.log 2>&1 &
echo "  PID: $!"

# Instance 2 — GPUs 4,5,6,7
echo "Starting instance 2 on GPUs 4-7 → port 8002"
CUDA_VISIBLE_DEVICES=4,5,6,7 \
	vllm serve "$VLLM_MODEL" \
	--tensor-parallel-size 4 \
	--port 8002 \
	--api-key "$VLLM_API_KEY" \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.92 \
	--disable-log-requests \
	>logs/vllm_instance2.log 2>&1 &
echo "  PID: $!"

# Instance 3 — GPUs 8,9,10,11
echo "Starting instance 3 on GPUs 8-11 → port 8003"
CUDA_VISIBLE_DEVICES=8,9,10,11 \
	vllm serve "$VLLM_MODEL" \
	--tensor-parallel-size 4 \
	--port 8003 \
	--api-key "$VLLM_API_KEY" \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.92 \
	--disable-log-requests \
	>logs/vllm_instance3.log 2>&1 &
echo "  PID: $!"

# Instance 4 — GPUs 12,13,14,15
echo "Starting instance 4 on GPUs 12-15 → port 8004"
CUDA_VISIBLE_DEVICES=12,13,14,15 \
	vllm serve "$VLLM_MODEL" \
	--tensor-parallel-size 4 \
	--port 8004 \
	--api-key "$VLLM_API_KEY" \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.92 \
	--disable-log-requests \
	>logs/vllm_instance4.log 2>&1 &
echo "  PID: $!"

echo ""
echo "Waiting 60s for servers to initialize..."
sleep 60

echo "Checking health..."
for port in 8001 8002 8003 8004; do
	if curl -sf "http://localhost:$port/health" >/dev/null; then
		echo "  [OK] vLLM instance on port $port is healthy"
	else
		echo "  [WARN] vLLM instance on port $port not yet responding"
	fi
done

echo ""
echo "vLLM servers started. Logs: logs/vllm_instance{1..4}.log"
echo "VLLM_URLS=http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004"
