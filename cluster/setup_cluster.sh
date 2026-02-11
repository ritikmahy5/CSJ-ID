#!/bin/bash
# ============================================
# CSJ-ID Setup Script for NEU Discovery Cluster
# Run this script ONCE after uploading your code
# ============================================

set -e  # Exit on error

echo "============================================"
echo "CSJ-ID Cluster Setup"
echo "============================================"

# Check if we're on the cluster
if [[ ! -d "/scratch" ]]; then
    echo "❌ Error: This script should be run on the Discovery cluster"
    echo "   Please ssh to login.discovery.neu.edu first"
    exit 1
fi

# Get paths
USER_DIR="/scratch/$USER"
PROJECT_DIR="$USER_DIR/research/Generative_Recommendation"

echo "User: $USER"
echo "Project directory: $PROJECT_DIR"

# Check if project directory exists
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "❌ Error: Project directory not found at $PROJECT_DIR"
    echo "   Please upload your files first using:"
    echo "   scp -r /path/to/ICMLFinal $USER@xfer.discovery.neu.edu:$USER_DIR/research/"
    exit 1
fi

cd "$PROJECT_DIR"

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p outputs_beauty
mkdir -p outputs_sports

# Load modules
echo ""
echo "📦 Loading modules..."
module load anaconda3/2024.06
module load cuda/11.8

# Check if environment exists
echo ""
echo "🐍 Setting up conda environment..."
if conda env list | grep -q "^csjid "; then
    echo "   Environment 'csjid' already exists"
    read -p "   Recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing old environment..."
        conda env remove -n csjid -y
        CREATE_ENV=true
    else
        CREATE_ENV=false
    fi
else
    CREATE_ENV=true
fi

if [ "$CREATE_ENV" = true ]; then
    echo "   Creating conda environment 'csjid'..."
    conda create -n csjid python=3.10 -y
fi

# Activate environment
echo ""
echo "   Activating environment..."
source activate csjid

# Install dependencies
echo ""
echo "📥 Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

echo ""
echo "📥 Installing other dependencies..."
pip install -r requirements.txt --quiet

# Make scripts executable
echo ""
echo "🔧 Making scripts executable..."
chmod +x cluster/*.slurm 2>/dev/null || true
chmod +x cluster/*.sh 2>/dev/null || true
chmod +x run.sh 2>/dev/null || true

# Verify setup
echo ""
echo "✅ Verifying setup..."

# Check Python
python --version

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check if sentence-transformers is installed
python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformer: OK')" 2>/dev/null || {
    echo "⚠️  Installing sentence-transformers..."
    pip install sentence-transformers --quiet
}

# Check data files
echo ""
echo "📊 Checking data files..."
for file in Beauty.json.gz Sports.json.gz; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        SIZE=$(du -h "$PROJECT_DIR/$file" | cut -f1)
        echo "   ✅ $file ($SIZE)"
    else
        echo "   ❌ $file NOT FOUND"
    fi
done

# Print config info
echo ""
echo "🔧 Testing configuration..."
cd "$PROJECT_DIR"
python -c "
import sys
sys.path.insert(0, 'src')
from config import print_environment_info
print_environment_info()
"

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Test with a quick run:"
echo "     sbatch cluster/run_quick_test.slurm"
echo ""
echo "  2. Check job status:"
echo "     squeue -u \$USER"
echo ""
echo "  3. Run full experiment:"
echo "     sbatch cluster/run_beauty.slurm"
echo ""
echo "  4. View logs:"
echo "     tail -f logs/csjid_*.out"
echo ""
