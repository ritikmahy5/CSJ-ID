#!/bin/bash
#SBATCH --job-name=csjid_beauty
#SBATCH --partition=sharing
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/csjid_beauty_%j.out
#SBATCH --error=logs/csjid_beauty_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@northeastern.edu

# ============================================
# CSJ-ID: Beauty Dataset Experiment
# ============================================
# GPU Options (uncomment ONE):
#   --gres=gpu:1              # Any available GPU (fastest queue)
#   --gres=gpu:v100-sxm2:1    # V100 32GB (reliable)
#   --gres=gpu:a100:1         # A100 40/80GB (faster)
#   --gres=gpu:h100:1         # H100 80GB (fastest, may wait longer)
# ============================================

echo "============================================"
echo "CSJ-ID Experiment - Beauty Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

# Create logs directory
mkdir -p logs

# Load modules
module load anaconda3/2024.06
module load cuda/11.8

# Activate environment
source activate csjid

# Set environment variables
export PYTHONUNBUFFERED=1

# Navigate to project directory
cd /scratch/$USER/research/Generative_Recommendation

# Print environment info
echo ""
echo "Environment Info:"
python -c "from config import print_environment_info; print_environment_info()"
echo ""

# Run experiment
echo "Starting training..."
python src/run_experiments.py \
    --dataset beauty \
    --seed 42 \
    --lambda_sem 0.5

echo ""
echo "============================================"
echo "Experiment Complete"
echo "End Time: $(date)"
echo "============================================"
