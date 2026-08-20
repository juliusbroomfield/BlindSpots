#!/bin/bash
# Mixture of Standpoints against a locally served Llama 3.1 8B.
#
# vLLM speaks the OpenAI protocol, so LiteLLM reaches it with the hosted_vllm/
# prefix and no batch endpoint is involved — the three MoS stages run as
# concurrent live calls against the local server.
#
#   sbatch scripts/run_mos_vllm.slurm

#SBATCH --job-name=mos-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mos_vllm_%j.out
#SBATCH --error=logs/mos_vllm_%j.err

set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8765
OUTPUT="mos_llama_responses.json"

module load cuda/12.1             
source ~/envs/blindspot/bin/activate

mkdir -p logs

echo "[$(date)] starting vLLM on port ${PORT}"
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype auto &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[$(date)] server up after ${i} attempts"
        break
    fi
    sleep 5
    if [ "$i" -eq 60 ]; then
        echo "ERROR: vLLM did not start within 5 minutes" >&2
        exit 1
    fi
done

echo "[$(date)] running MoS"
blindspot generate \
    --model "hosted_vllm/${MODEL}" \
    --api-base "http://localhost:${PORT}/v1" \
    --condition mos \
    --levels A \
    --max-workers 32 \
    --output "${OUTPUT}"

echo "[$(date)] judging"
blindspot judge \
    --input  "${OUTPUT}" \
    --output "mos_llama_evaluated.json" \
    --response-field final_response

echo "[$(date)] done"
