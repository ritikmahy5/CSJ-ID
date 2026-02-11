#!/bin/bash
#SBATCH --job-name=csjid_significance
#SBATCH --partition=sharing
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/csjid_significance_%j.out
#SBATCH --error=logs/csjid_significance_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@northeastern.edu

# ============================================
# CSJ-ID: Multi-Seed Significance Testing
# Runs experiments with multiple seeds for
# statistical significance analysis
# ============================================

echo "============================================"
echo "CSJ-ID Significance Testing"
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
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd /scratch/$USER/research/Generative_Recommendation

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Seeds for significance testing
SEEDS=(42 123 456 789 1024)
DATASET="beauty"

echo ""
echo "Running significance tests with seeds: ${SEEDS[*]}"
echo ""

# Run experiments with different seeds
for SEED in "${SEEDS[@]}"; do
    echo "============================================"
    echo "Running with seed: $SEED"
    echo "Time: $(date)"
    echo "============================================"
    
    python src/run_experiments.py \
        --dataset $DATASET \
        --seed $SEED \
        --lambda_sem 0.5
    
    echo "Completed seed $SEED"
    echo ""
done

# Run significance analysis
echo "============================================"
echo "Running significance analysis..."
echo "============================================"
python src/run_significance.py --dataset $DATASET

echo ""
echo "============================================"
echo "All Experiments Complete"
echo "End Time: $(date)"
echo "============================================"
