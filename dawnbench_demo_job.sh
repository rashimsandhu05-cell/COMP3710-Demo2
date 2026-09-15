#!/bin/bash
#SBATCH --job-name=dawnbench
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=dawnbench_%j.out
#SBATCH --error=dawnbench_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

echo "=========================================="
echo "DAWNBench ResNet-18 Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

echo "GPU information:"
nvidia-smi

echo ""
echo "Starting training..."
echo ""

python -u dawnbench_demo.py
