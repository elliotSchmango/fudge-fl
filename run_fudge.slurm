#!/bin/bash
#SBATCH --job-name=fudge_suite
#SBATCH --account=<YOUR_UVA_ALLOCATION> #replace with UVA proj name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a6000|a40|v100"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/fudge_hpc_%A_%a.out
#SBATCH --error=logs/fudge_hpc_%A_%a.err

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

echo "Calculating config chunk for array task ${SLURM_ARRAY_TASK_ID}..."
START_IDX=$(( SLURM_ARRAY_TASK_ID * 17 + 1 ))
END_IDX=$(( START_IDX + 16 ))

echo "Launching benchmark runner for configs ${START_IDX} to ${END_IDX}..."
uv run python runner.py --start-idx ${START_IDX} --end-idx ${END_IDX}

echo "Evaluation complete, check results in /results folder."