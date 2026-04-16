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
#SBATCH --time=9:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/fudge_hpc_%A_%a.out
#SBATCH --error=logs/fudge_hpc_%A_%a.err

mkdir -p results logs

echo "Starting FUDGE-Suite Benchmark on UVA HPC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"

module purge
module load miniforge
conda activate

#explicitly point uv at Python 3.12 to avoid 3.14 incompatibility with torch
export UV_PYTHON=$(which python3.12 2>/dev/null || conda run python3.12 -c 'import sys; print(sys.executable)' 2>/dev/null || echo "python3.12")
echo "Using Python: $UV_PYTHON"

if [ "$SLURM_ARRAY_TASK_ID" == "0" ]; then
    echo "Task 0: Wiping corrupted environment and rebuilding (Tasks 1 and 2 will wait)..."
    rm -rf .venv .env_ready
    
    echo "Linking UV dependencies..."
    uv sync --python "$UV_PYTHON"
    
    echo "Installing CUDA PyTorch inside UV environment..."
    uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
    
    touch .env_ready
else
    echo "Task $SLURM_ARRAY_TASK_ID: Waiting for Task 0 to finish building the environment..."
    while [ ! -f .env_ready ]; do
        sleep 5
    done
    echo "Environment is ready! Proceeding..."
fi

echo "Calculating config chunk for array task ${SLURM_ARRAY_TASK_ID}..."
START_IDX=$(( SLURM_ARRAY_TASK_ID * 14 + 1 ))
END_IDX=$(( START_IDX + 13 ))

echo "Verifying torch sees GPU..."
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "Launching benchmark runner for configs ${START_IDX} to ${END_IDX}..."
.venv/bin/python runner.py --start-idx ${START_IDX} --end-idx ${END_IDX}

echo "Evaluation complete, check results in /results folder."