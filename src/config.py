"""
Configuration for CSJ-ID Experiments
ICML 2026 Submission

Automatically detects environment (local Mac vs NEU cluster) and sets paths accordingly.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch


def _detect_base_dir():
    """
    Automatically detect the base directory based on environment.
    Works on both local Mac and NEU Discovery cluster.
    """
    # Check if on NEU Discovery/Explorer cluster
    if os.path.exists('/scratch'):
        user = os.environ.get('USER', os.environ.get('SLURM_JOB_USER', 'unknown'))
        cluster_path = f'/scratch/{user}/research/Generative_Recommendation'
        if os.path.exists(cluster_path):
            return cluster_path
        # Try to create if parent exists
        parent = f'/scratch/{user}/research'
        if os.path.exists(parent):
            os.makedirs(cluster_path, exist_ok=True)
            return cluster_path
    
    # Check common local paths
    local_paths = [
        '/Users/ritik/Desktop/Research/ICMLFinal',
        os.path.expanduser('~/Desktop/Research/ICMLFinal'),
        os.path.expanduser('~/research/ICMLFinal'),
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            return path
    
    # Fallback: use directory containing this config file
    config_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(config_dir)  # Parent of src/


# Auto-detect base directory at import time
_BASE_DIR = _detect_base_dir()
_IS_CLUSTER = '/scratch' in _BASE_DIR


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "beauty"  # beauty, sports, toys
    data_path: str = field(default_factory=lambda: f"{_BASE_DIR}/Beauty.json.gz")
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    max_seq_len: int = 50
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    max_users: int = 0  # 0 = no limit, otherwise sample this many users
    
    def __post_init__(self):
        # Set dataset paths based on detected base directory
        self.DATASET_PATHS = {
            'beauty': f'{_BASE_DIR}/Beauty.json.gz',
            'sports': f'{_BASE_DIR}/Sports.json.gz',
            'toys': f'{_BASE_DIR}/Toys.json.gz',
        }
        # Update data_path if it's using old hardcoded path
        if '/Users/ritik' in self.data_path and _IS_CLUSTER:
            self.data_path = self.DATASET_PATHS.get(self.dataset, self.data_path)
    
    def set_dataset(self, name: str):
        """Set dataset by name."""
        if name in self.DATASET_PATHS:
            self.dataset = name
            self.data_path = self.DATASET_PATHS[name]
        else:
            raise ValueError(f"Unknown dataset: {name}. Choose from {list(self.DATASET_PATHS.keys())}")
        return self
    

@dataclass
class ModelConfig:
    """Model configuration."""
    # Embedding dimensions
    semantic_dim: int = 384  # SentenceTransformer output
    cf_dim: int = 64  # LightGCN embedding dim
    hidden_dim: int = 256
    
    # RQ-VAE settings
    num_levels: int = 4
    codebook_size: int = 256
    commitment_weight: float = 0.25
    
    # LightGCN settings
    lightgcn_layers: int = 3
    lightgcn_dropout: float = 0.1
    
    # Transformer (GenRec) settings
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    

@dataclass
class TrainingConfig:
    """Training configuration."""
    # General
    seed: int = 42
    device: str = "auto"  # auto, cuda, mps, cpu
    
    # RQ-VAE training
    rqvae_epochs: int = 100
    rqvae_batch_size: int = 256
    rqvae_lr: float = 1e-4
    rqvae_weight_decay: float = 1e-5
    
    # LightGCN training
    lightgcn_epochs: int = 100
    lightgcn_batch_size: int = 2048
    lightgcn_lr: float = 0.01  # Increased from 1e-3 - BPR needs higher LR
    lightgcn_weight_decay: float = 1e-5  # Reduced regularization
    
    # GenRec training
    genrec_epochs: int = 30
    genrec_batch_size: int = 256  # Increased for efficient single-pass training
    genrec_lr: float = 1e-4
    genrec_weight_decay: float = 1e-5
    
    # Lambda for CSJ-ID
    lambda_sem: float = 0.5
    

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    ks: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    num_neg_samples: int = 99  # For ranking evaluation


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Paths - auto-configured based on environment
    base_dir: str = field(default_factory=lambda: _BASE_DIR)
    output_dir: str = field(default_factory=lambda: f"{_BASE_DIR}/outputs")
    log_file: str = "experiment_log.txt"
    
    # Experiment settings
    run_baselines: bool = True
    run_ablations: bool = True
    run_cold_start: bool = True
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Optimize batch sizes for cluster GPUs
        if _IS_CLUSTER:
            self._optimize_for_cluster()
    
    def _optimize_for_cluster(self):
        """Optimize settings for cluster environment."""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                
                if 'a100' in gpu_name:
                    self.training.genrec_batch_size = 512
                    self.training.rqvae_batch_size = 512
                    self.training.lightgcn_batch_size = 4096
                elif 'h100' in gpu_name or 'h200' in gpu_name:
                    self.training.genrec_batch_size = 512
                    self.training.rqvae_batch_size = 1024
                    self.training.lightgcn_batch_size = 8192
                elif 'v100' in gpu_name:
                    self.training.genrec_batch_size = 256
                    self.training.rqvae_batch_size = 512
                    self.training.lightgcn_batch_size = 4096
        except:
            pass  # If detection fails, use defaults
        
    def get_device(self) -> torch.device:
        if self.training.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.training.device)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def print_environment_info():
    """Print detected environment information."""
    print(f"{'='*50}")
    print("CSJ-ID Environment Configuration")
    print(f"{'='*50}")
    print(f"Base directory: {_BASE_DIR}")
    print(f"Running on cluster: {_IS_CLUSTER}")
    
    config = get_default_config()
    print(f"Device: {config.get_device()}")
    print(f"Output directory: {config.output_dir}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"{'='*50}")


if __name__ == "__main__":
    print_environment_info()
