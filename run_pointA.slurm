#!/bin/bash
#SBATCH --job-name=fudge_ptA
#SBATCH --account=<YOUR_UVA_ALLOCATION> #replace with UVA proj name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a6000|a40|v100"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=logs/fudge_pointA_%j.out
#SBATCH --error=logs/fudge_pointA_%j.err

echo "Starting FUDGE-Suite Point A Validation Run"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"

module purge
module load miniforge
conda activate

#explicitly point uv at Python 3.12 to avoid 3.14 incompatibility with torch
export UV_PYTHON=$(which python3.12 2>/dev/null || conda run python3.12 -c 'import sys; print(sys.executable)' 2>/dev/null || echo "python3.12")
echo "Using Python: $UV_PYTHON"

#only rebuild env if .env_ready is missing (reuse from previous run)
if [ ! -f .env_ready ]; then
    echo "Building environment from scratch..."
    rm -rf .venv
    uv sync --python "$UV_PYTHON"
    uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
    touch .env_ready
else
    echo "Reusing existing environment."
fi

echo "Verifying torch sees GPU..."
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

#point-a-only: 10 configs (5 agg x 2 threat x 1 unlearn), 20 rounds with per-aggregator poison rates
echo "Launching Point A validation (10 configs, 20 rounds, per-agg poison rates)..."
.venv/bin/python runner.py --point-a-only --num-rounds 20 --results-dir results_pointA

echo "Point A validation complete! Check results_pointA/ for trajectory data."
