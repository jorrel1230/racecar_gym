#!/bin/bash
#SBATCH --job-name=f1-rl-train
#SBATCH --output=/home/jr8867/racecar_gym/logs/slurm_%j.out
#SBATCH --error=/home/jr8867/racecar_gym/logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jorrel@princeton.edu

set -euo pipefail

REPO_DIR="/home/jr8867/racecar_gym"

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"
echo "Repo: $REPO_DIR"

module load anaconda3/2025.12
conda activate racecar

# PyBullet requires a display or DIRECT mode — force headless
export DISPLAY=""
export PYBULLET_EGL=1   # use EGL for headless GPU rendering if available

mkdir -p "$REPO_DIR/training/logs"
mkdir -p "$REPO_DIR/training/checkpoints"

cd "$REPO_DIR/training"

# Resolve resume checkpoint if continuing previous run
RESUME_ARG=""
LATEST=$(ls -t checkpoints/sac_step_*.zip 2>/dev/null | head -1)
if [[ -n "$LATEST" ]]; then
    echo "Resuming from: $LATEST"
    RESUME_ARG="--resume $LATEST"
fi

# Resolve demos arg if demos exist
DEMOS_ARG=""
if [[ -d "./demos" && "$(ls demos/ep_*.npz 2>/dev/null | wc -l)" -gt 0 ]]; then
    echo "Using demos from ./demos/"
    DEMOS_ARG="--demos ./demos"
fi

python train.py \
    --track SingleAgentAustria_train-v0 \
    --num-cpu "$SLURM_CPUS_PER_TASK" \
    --total-timesteps 10000000 \
    --checkpoint-freq 25000 \
    --buffer-size 500000 \
    $RESUME_ARG \
    $DEMOS_ARG

echo "Done: $(date)"
