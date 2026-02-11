#!/bin/bash
#SBATCH --job-name=csjid_sports
#SBATCH --partition=sharing
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/csjid_sports_%j.out
#SBATCH --error=logs/csjid_sports_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@northeastern.edu

# ============================================
# CSJ-ID: Sports Dataset Experiment
# (Larger dataset - needs more memory/time)
# ============================================

echo "============================================"
echo "CSJ-ID Experiment - Sports Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

mkdir -p logs

module load anaconda3/2024.06
module load cuda/11.8
source activate csjid

export PYTHONUNBUFFERED=1
cd /scratch/$USER/research/Generative_Recommendation

echo ""
python -c "from config import print_environment_info; print_environment_info()"
echo ""

# Sports is larger, sample 30k users for reasonable runtime
# Remove --max_users for full dataset (will take much longer)
echo "Starting training..."
python src/run_experiments.py \
    --dataset sports \
    --seed 42 \
    --lambda_sem 0.5 \
    --max_users 30000

echo ""
echo "============================================"
echo "Experiment Complete - $(date)"
echo "============================================"
