#!/bin/bash
#SBATCH --job-name=csjid_test
#SBATCH --partition=sharing
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/csjid_test_%j.out
#SBATCH --error=logs/csjid_test_%j.err

# ============================================
# CSJ-ID: Quick Test Run
# Use this to verify everything works
# Expected runtime: 15-30 minutes
# ============================================

echo "============================================"
echo "CSJ-ID Quick Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "============================================"

mkdir -p logs

module load anaconda3/2024.06
module load cuda/11.8
source activate csjid

export PYTHONUNBUFFERED=1
cd /scratch/$USER/research/Generative_Recommendation

# Print environment info
echo ""
echo "Environment Check:"
python -c "
import sys
sys.path.insert(0, 'src')
from config import print_environment_info
print_environment_info()
"

# Verify data files exist
echo ""
echo "Data Files:"
ls -lh *.json.gz 2>/dev/null || echo "No data files found!"

# Run quick test
echo ""
echo "Running quick test (reduced epochs)..."
python src/run_experiments.py \
    --dataset beauty \
    --seed 42 \
    --quick

echo ""
echo "============================================"
echo "Quick Test Complete!"
echo "End Time: $(date)"
echo "============================================"
echo ""
echo "If this succeeded, you can run full experiments:"
echo "  sbatch cluster/run_beauty.slurm"
echo "  sbatch cluster/run_sports.slurm"
