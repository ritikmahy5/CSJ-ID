#!/bin/bash
#SBATCH --job-name=csjid_both
#SBATCH --partition=sharing
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/csjid_both_%j.out
#SBATCH --error=logs/csjid_both_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@northeastern.edu

# ============================================
# CSJ-ID: Run Both Datasets
# Full experiments on Beauty and Sports
# ============================================

echo "============================================"
echo "CSJ-ID Full Experiments - Both Datasets"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

mkdir -p logs

# Load modules
module load anaconda3/2024.06
module load cuda/11.8

source activate csjid

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd /scratch/$USER/research/Generative_Recommendation

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# ============================================
# Beauty Dataset
# ============================================
echo ""
echo "============================================"
echo "Starting Beauty Dataset Experiment"
echo "Time: $(date)"
echo "============================================"

python src/run_experiments.py \
    --dataset beauty \
    --seed 42 \
    --lambda_sem 0.5

# ============================================
# Sports Dataset
# ============================================
echo ""
echo "============================================"
echo "Starting Sports Dataset Experiment"
echo "Time: $(date)"
echo "============================================"

python src/run_experiments.py \
    --dataset sports \
    --seed 42 \
    --lambda_sem 0.5 \
    --max_users 30000

# ============================================
# Summary
# ============================================
echo ""
echo "============================================"
echo "All Experiments Complete!"
echo "End Time: $(date)"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  - outputs_beauty/"
echo "  - outputs_sports/"
