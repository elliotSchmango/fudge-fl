#!/bin/bash
#SBATCH --job-name=fudge_suite
#SBATCH --account=<YOUR_UVA_ALLOCATION> #replace with UVA proj name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1               #request nvidia a100 gpu 
#SBATCH --nodes=1                       #fit on single node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fudge_hpc_%j.out
#SBATCH --error=logs/fudge_hpc_%j.err

echo "Starting FUDGE-Suite Benchmark on UVA HPC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"

module purge
module load miniforge

echo "Linking UV dependencies..."
uv sync

echo "Installing CUDA PyTorch inside UV environment..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Launching benchmark runner..."
uv run python runner.py

echo "Evaluation complete, check results in /results folder."