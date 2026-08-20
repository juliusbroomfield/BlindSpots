#!/bin/bash
# LoRA distillation of GPT-5's base-condition responses into Llama 3.1 8B.
#
# Trains on 8 domains and holds out healthcare interactions and events &
# gatherings, so held-out recall measures transfer rather than memorisation.
#
#   export HF_TOKEN=...        # Llama 3.1 is gated
#   export WANDB_API_KEY=...   # optional
#   sbatch scripts/run_finetune.slurm

#SBATCH --job-name=llama-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1           # an A100 40GB fits 8B with LoRA + bf16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

set -euo pipefail

module load cuda/12.1         
source ~/envs/blindspot/bin/activate

mkdir -p logs checkpoints finetune_data

: "${HF_TOKEN:?export HF_TOKEN — Llama 3.1 8B is a gated model}"
export WANDB_PROJECT="${WANDB_PROJECT:-blindspot-sft}"
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
    echo "WANDB_API_KEY unset — logging offline"
fi

echo "[$(date)] preparing splits"
python mitigations/sft/prepare_finetune_data.py

echo "[$(date)] fine-tuning"
python mitigations/sft/finetune_lora.py \
    --train    finetune_data/train.jsonl \
    --val      finetune_data/val.jsonl   \
    --output   checkpoints               \
    --run-name "llama-3.1-8b-lora-$(date +%Y%m%d)"

echo "[$(date)] evaluating on held-out domains"
python mitigations/sft/finetune_lora.py \
    --eval-only \
    --checkpoint checkpoints/best \
    --held-out   finetune_data/held_out.jsonl

echo "[$(date)] done"
