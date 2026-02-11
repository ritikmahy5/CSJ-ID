#!/bin/bash
# ============================================
# Quick Transfer Commands for CSJ-ID
# Folder: Generative_Recommendation
# ============================================
# 
# INSTRUCTIONS:
# 1. Replace YOUR_USERNAME with your NEU username below
# 2. Copy and paste the commands into your terminal
#
# ============================================

# ========== STEP 1: Set your username ==========
export NEU_USER="YOUR_USERNAME"  # <-- CHANGE THIS!

# ========== STEP 2: Create directory on cluster ==========
# Run this first (creates the destination folder)
ssh ${NEU_USER}@login.discovery.neu.edu "mkdir -p /scratch/${NEU_USER}/research/Generative_Recommendation"

# ========== STEP 3: Transfer files ==========
# Option A: Transfer everything at once (recommended)
scp -r /Users/ritik/Desktop/Research/ICMLFinal/{src,cluster,requirements.txt,run.sh,README.md,Beauty.json.gz,Sports.json.gz} ${NEU_USER}@xfer.discovery.neu.edu:/scratch/${NEU_USER}/research/Generative_Recommendation/

# Option B: Transfer in parts (if connection is unstable)
# First, transfer code (small, fast):
scp -r /Users/ritik/Desktop/Research/ICMLFinal/src ${NEU_USER}@xfer.discovery.neu.edu:/scratch/${NEU_USER}/research/Generative_Recommendation/
scp -r /Users/ritik/Desktop/Research/ICMLFinal/cluster ${NEU_USER}@xfer.discovery.neu.edu:/scratch/${NEU_USER}/research/Generative_Recommendation/
scp /Users/ritik/Desktop/Research/ICMLFinal/{requirements.txt,run.sh,README.md} ${NEU_USER}@xfer.discovery.neu.edu:/scratch/${NEU_USER}/research/Generative_Recommendation/

# Then, transfer datasets (large, may take a few minutes):
scp /Users/ritik/Desktop/Research/ICMLFinal/Beauty.json.gz ${NEU_USER}@xfer.discovery.neu.edu:/scratch/${NEU_USER}/research/Generative_Recommendation/
scp /Users/ritik/Desktop/Research/ICMLFinal/Sports.json.gz ${NEU_USER}@xfer.discovery.neu.edu:/scratch/${NEU_USER}/research/Generative_Recommendation/

# ========== STEP 4: Verify transfer ==========
ssh ${NEU_USER}@login.discovery.neu.edu "ls -la /scratch/${NEU_USER}/research/Generative_Recommendation/"

# ========== STEP 5: Setup on cluster ==========
# SSH to cluster and run setup
ssh ${NEU_USER}@login.discovery.neu.edu
# Then on the cluster:
# cd /scratch/$USER/research/Generative_Recommendation
# bash cluster/setup_cluster.sh
