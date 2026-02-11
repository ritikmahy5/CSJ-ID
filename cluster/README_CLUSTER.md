# Running CSJ-ID Experiments on NEU Discovery Cluster

## Quick Start Guide for https://ood.explorer.northeastern.edu/

**Project Folder:** `Generative_Recommendation`

### 1. Login & Access
1. Go to https://ood.explorer.northeastern.edu/
2. Login with your NEU credentials
3. Navigate to **Clusters → Discovery Shell Access** or use the **Files** app

### 2. Setup (First Time Only)

```bash
# Connect to Discovery cluster
ssh <your_username>@login.discovery.neu.edu

# Navigate to your scratch space (recommended for large files)
cd /scratch/$USER
mkdir -p research
cd research

# Clone/copy your project (or use SFTP via OOD Files app)
# Option A: Upload via OOD Files interface to /scratch/$USER/research/
# Option B: scp from your local machine:
# scp -r ~/Desktop/Research/ICMLFinal/* $USER@xfer.discovery.neu.edu:/scratch/$USER/research/Generative_Recommendation/

# Create conda environment
module load anaconda3/2023.09
conda create -n csjid python=3.10 -y
conda activate csjid

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
cd Generative_Recommendation
pip install -r requirements.txt
```

### 3. Running Experiments

#### Option A: Interactive Session (for debugging/small runs)
1. Go to OOD → Interactive Apps → Jupyter Notebook (or VSCode)
2. Select:
   - Partition: `gpu`
   - Time: 4-8 hours
   - GPUs: 1
   - Memory: 32GB
   - Cores: 4

#### Option B: Batch Jobs (recommended for full experiments)

```bash
# Make scripts executable
chmod +x cluster/*.slurm

# Submit a single experiment (Beauty dataset)
sbatch cluster/run_beauty.slurm

# Submit Sports dataset
sbatch cluster/run_sports.slurm

# Submit all experiments with multiple seeds
sbatch cluster/run_significance.slurm

# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>
```

### 4. Monitor Jobs

```bash
# View job output in real-time
tail -f logs/csjid_beauty_*.out

# Check GPU utilization (while job is running)
srun --jobid=<job_id> --pty nvidia-smi

# View all your running jobs
squeue -u $USER
```

### 5. Available Partitions on Discovery

| Partition | GPUs | Max Time | Best For |
|-----------|------|----------|----------|
| `gpu` | V100/A100 | 8h | Full experiments |
| `multigpu` | Multiple GPUs | 24h | Large models (need approval) |
| `short` | CPU only | 4h | Quick preprocessing |

### 6. GPU Options

Request specific GPU types in SLURM scripts:
```bash
#SBATCH --gres=gpu:1              # Any GPU (fastest queue)
#SBATCH --gres=gpu:v100-sxm2:1    # V100 32GB
#SBATCH --gres=gpu:a100:1         # A100 40/80GB
#SBATCH --gres=gpu:h100:1         # H100 80GB
```

### 7. Output Files
Results will be saved to:
- `/scratch/$USER/research/Generative_Recommendation/outputs_beauty/`
- `/scratch/$USER/research/Generative_Recommendation/outputs_sports/`

Log files:
- `/scratch/$USER/research/Generative_Recommendation/logs/`

### 8. Troubleshooting

**CUDA out of memory:**
- Reduce batch size: Edit `src/config.py`, set `genrec_batch_size = 128`
- Use user sampling: `python src/run_experiments.py --max_users 30000`

**Module not found:**
```bash
module load anaconda3/2023.09
conda activate csjid
```

**Job pending too long:**
- Try generic `gpu` partition instead of specific GPU types
- Reduce requested time/resources
- Check cluster load: `squeue -p gpu`

### 9. Copying Results Back to Local Machine

```bash
# From your local machine
scp -r $USER@xfer.discovery.neu.edu:/scratch/$USER/research/Generative_Recommendation/outputs_* ~/Desktop/Research/ICMLFinal/
```

### 10. Email Notifications
All SLURM scripts are configured to email you at your NEU email when jobs start/end/fail.
Update `#SBATCH --mail-user=` in the scripts with your actual email.
