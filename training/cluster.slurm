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

module load anaconda3/2025.12
conda activate racecar

python /home/jr8867/racecar_gym/training/train.py \
    --track SingleAgentAustria_train-v0 \
    --num-cpu 8 \
    --total-timesteps 10000000 \
    --checkpoint-freq 25000 \
    --buffer-size 500000
