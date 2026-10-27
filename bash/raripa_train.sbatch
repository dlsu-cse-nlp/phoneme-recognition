#!/bin/bash
#SBATCH --account=dlsu-ilao-group-2
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu-a100_default
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=3-00:00:00
#SBATCH --job-name=raripa
#SBATCH --output=logs/%x.%j.out

set -euo pipefail

# Environment variables
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/scratch1/lorenz.marqueses/uv_cache
export HF_HOME=/scratch1/lorenz.marqueses/hf_cache
export LD_LIBRARY_PATH=/scratch1/lorenz.marqueses/envs/ffmpeg/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
[ "${OFFLINE:-0}" = "1" ] && export HF_HUB_OFFLINE=1

# This is the repository clone in the scratch1/ directory, since we can't run
# jobs from home/ directly
cd ~/scratch1/phoneme-recognition

# Training config
MODE=${MODE:-combined}
DESC=${DESC:-job${SLURM_JOB_ID}}
TRAIN=${TRAIN:-data/nexdata/filipino_822/train_manifest.json}
VALID=${VALID:-data/nexdata/filipino_822/valid_manifest.json}
TEST=${TEST:-data/nexdata/filipino_822/test_manifest.json}
# TODO: Add these eventually...?
# CKPT=<ctc checkpoint to start from>
# RESUME=<checkpoint to resume>

echo "Training '$DESC' ($MODE) on job ${SLURM_JOB_ID}, node $(hostname)"
echo "Started: $(date)"

# Check for NVIDIA GPUs
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run --frozen python -c "import torch; print('cuda available:', torch.cuda.is_available())"

# Call the training script
uv run --frozen python -m src.scripts.raripa_train \
  --mode "$MODE" \
  --description "$DESC" \
  --train-manifest-path "$TRAIN" \
  --valid-manifest-path "$VALID" \
  --test-manifest-path "$TEST" \
  ${CKPT:+--ctc-checkpoint-path "$CKPT"} \
  ${RESUME:+--resume-from-checkpoint "$RESUME"}

echo "Finished: $(date)"
