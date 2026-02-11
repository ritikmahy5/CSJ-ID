#!/bin/bash
# =============================================================================
# CSJ-ID: Run All Tasks
# =============================================================================
# This script runs:
# 1. Generate paper figures
# 2. Run additional seeds (optional)
# 3. Codebook visualization
# =============================================================================

set -e

cd /Users/ritik/Desktop/Research/ICMLFinal

echo "=============================================="
echo "CSJ-ID: Running All Tasks"
echo "=============================================="

# Create necessary directories
mkdir -p figures
mkdir -p paper

# Task 1: Generate Paper Figures
echo ""
echo "=============================================="
echo "Task 1: Generating Paper Figures"
echo "=============================================="
python src/generate_figures.py

# Task 4: Codebook Visualization  
echo ""
echo "=============================================="
echo "Task 4: Generating Codebook Visualizations"
echo "=============================================="
python src/codebook_visualization.py

echo ""
echo "=============================================="
echo "Completed Tasks 1, 3, 4!"
echo "=============================================="
echo ""
echo "Files created:"
echo "  - figures/*.pdf - Publication figures"
echo "  - paper/csj_id_paper.tex - Paper draft"
echo "  - paper/references.bib - Bibliography"
echo ""
echo "=============================================="
echo "Task 2: Additional Seeds (OPTIONAL)"
echo "=============================================="
echo ""
echo "To run 2 more seeds on Beauty for statistical significance:"
echo "  python src/run_additional_seeds.py --dataset beauty --seeds 789 1011"
echo ""
echo "Estimated time: ~14 hours"
echo ""
