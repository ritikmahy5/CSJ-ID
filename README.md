# CSJ-ID: Collaborative-Semantic Joint IDs for Generative Recommendation

**Target**: ICML 2026 (Deadline: January 29, 2026)

## Overview

This repository implements **Collaborative-Semantic Joint IDs (CSJ-IDs)**, a novel approach to learning discrete item representations that capture both content semantics AND collaborative filtering signals for generative recommendation systems.

### Key Innovation
Unlike TIGER and other semantic ID approaches that only use content embeddings, CSJ-IDs jointly optimize for:
- **Semantic reconstruction**: Preserving content similarity
- **CF reconstruction**: Preserving interaction patterns

## Project Structure

```
ICMLFinal/
├── src/                          # Main source code (Python scripts)
│   ├── __init__.py
│   ├── config.py                 # Configuration dataclasses
│   ├── data.py                   # Data loading and preprocessing
│   ├── models.py                 # Model implementations (RQ-VAE, LightGCN, GenRec)
│   ├── train.py                  # Training functions
│   ├── evaluate.py               # Evaluation metrics
│   ├── utils.py                  # Logging and utilities
│   ├── run_experiments.py        # Main experiment runner
│   └── requirements.txt          # Python dependencies
│
├── outputs/                      # Results, models, and logs
│   ├── experiment_*.txt          # Experiment logs
│   ├── results.json              # Final results
│   ├── csj_codes.pt              # CSJ-ID codes
│   ├── sem_codes.pt              # Semantic-only codes
│   └── *.pt                      # Model checkpoints
│
├── paper/                        # LaTeX paper
│   ├── main.tex
│   └── references.bib
│
├── Beauty.json.gz                # Amazon Beauty dataset
├── run.sh                        # Shell script runner
├── README.md                     # This file
└── paper_draft.md                # Markdown draft
```

## Quick Start

### 1. Install Dependencies
```bash
cd src
pip install -r requirements.txt
```

### 2. Run Full Experiment
```bash
# Full run (takes ~2-3 hours)
python src/run_experiments.py

# Quick test run (fewer epochs, ~30 min)
python src/run_experiments.py --quick

# Or use the shell script
chmod +x run.sh
./run.sh --quick
```

### 3. Run Specific Stage
```bash
# Just train LightGCN
python src/run_experiments.py --stage lightgcn

# Just evaluate
python src/run_experiments.py --stage eval
```

## Configuration

Edit `src/config.py` to modify:
- Training hyperparameters
- Model architecture
- Data paths
- Evaluation settings

Key hyperparameters:
```python
lambda_sem = 0.5      # Balance semantic vs CF loss
num_levels = 4        # Quantization levels  
codebook_size = 256   # Codes per level
hidden_dim = 256      # Latent dimension
```

## Expected Results

| Method | R@1 | R@5 | R@10 | R@20 |
|--------|-----|-----|------|------|
| Semantic-only | 0.0055 | 0.0065 | 0.0090 | 0.0095 |
| **CSJ-ID (Ours)** | **0.0090** | **0.0190** | **0.0220** | **0.0220** |
| Improvement | +63.6% | +192.3% | +144.4% | +131.6% |

## Output Files

After running experiments, find results in `outputs/`:
- `experiment_YYYYMMDD_HHMMSS.txt` - Detailed log with all training progress
- `results.json` - All metrics and ablation results
- `csj_codes.pt`, `sem_codes.pt` - Learned item codes
- `genrec_csj.pt`, `genrec_sem.pt` - Trained GenRec models

## Experiment Stages

1. **Data Loading** - Load and preprocess Amazon Beauty dataset
2. **Semantic Embeddings** - Extract embeddings with SentenceTransformer
3. **LightGCN Training** - Train LightGCN for CF embeddings (BPR loss)
4. **RQ-VAE Training** - Train CSJ-ID and Semantic-only baselines
5. **GenRec Training** - Train generative recommenders with learned IDs
6. **Evaluation** - Compute Recall@K, NDCG@K, MRR
7. **Ablations** - Lambda sensitivity analysis
8. **Cold-Start Analysis** - Evaluate on cold vs warm users

## Baselines

1. **Semantic-only (TIGER)** - Content-only RQ-VAE
2. **CF-only** - Interaction-only RQ-VAE  
3. **CSJ-ID (Ours)** - Joint optimization

## Citation

```bibtex
@inproceedings{csjid2026,
  title={Collaborative-Semantic Joint IDs: Bridging Interaction Patterns and Content Semantics for Generative Recommendation},
  author={...},
  booktitle={ICML},
  year={2026}
}
```

## References

- TIGER: [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (NeurIPS 2023)
- LightGCN: [Simplifying and Powering Graph Convolution Network](https://arxiv.org/abs/2002.02126) (SIGIR 2020)
