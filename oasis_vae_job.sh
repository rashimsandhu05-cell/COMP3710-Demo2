#!/bin/bash
#SBATCH --job-name=oasis-vae
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=oasis_vae_%j.out
#SBATCH --error=oasis_vae_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

echo "=== OASIS VAE JOB ==="
nvidia-smi

python -u oasis_vae.py
