#!/bin/bash
#SBATCH --job-name=oasis-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=oasis_unet_%j.out
#SBATCH --error=oasis_unet_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

echo "=== OASIS U-NET JOB ==="
nvidia-smi

python -u oasis_unet.py
