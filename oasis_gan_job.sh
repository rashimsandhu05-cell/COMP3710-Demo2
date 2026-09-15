#!/bin/bash
#SBATCH --job-name=oasis-gan
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=oasis_gan_%j.out
#SBATCH --error=oasis_gan_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

echo "=== OASIS GAN JOB ==="
nvidia-smi

python -u oasis_gan.py
