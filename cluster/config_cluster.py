"""
Cluster-specific configuration for NEU Discovery.
This file overrides paths for the cluster environment.

Usage:
    from cluster.config_cluster import get_cluster_config
    config = get_cluster_config()
"""

import os
from dataclasses import dataclass


def get_scratch_dir():
    """Get the scratch directory for the current user."""
    user = os.environ.get('USER', os.environ.get('SLURM_JOB_USER', 'unknown'))
    return f"/scratch/{user}/research/Generative_Recommendation"


def update_config_for_cluster(config):
    """
    Update an ExperimentConfig object for cluster paths.
    
    Args:
        config: ExperimentConfig from src/config.py
    
    Returns:
        Updated config with cluster-appropriate paths
    """
    base_dir = get_scratch_dir()
    
    # Update paths
    config.base_dir = base_dir
    config.output_dir = os.path.join(base_dir, f'outputs_{config.data.dataset}')
    
    # Update dataset paths
    config.data.DATASET_PATHS = {
        'beauty': os.path.join(base_dir, 'Beauty.json.gz'),
        'sports': os.path.join(base_dir, 'Sports.json.gz'),
        'toys': os.path.join(base_dir, 'Toys.json.gz'),
    }
    
    # Update data path if dataset is set
    if config.data.dataset in config.data.DATASET_PATHS:
        config.data.data_path = config.data.DATASET_PATHS[config.data.dataset]
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    return config


def get_cluster_config():
    """
    Get a default cluster configuration.
    
    Returns:
        ExperimentConfig configured for cluster environment
    """
    import sys
    sys.path.insert(0, os.path.join(get_scratch_dir(), 'src'))
    from config import get_default_config
    
    config = get_default_config()
    return update_config_for_cluster(config)


# Cluster-specific optimizations
CLUSTER_OPTIMIZATIONS = {
    'v100': {
        'genrec_batch_size': 256,
        'rqvae_batch_size': 512,
        'lightgcn_batch_size': 4096,
    },
    'a100': {
        'genrec_batch_size': 512,
        'rqvae_batch_size': 1024,
        'lightgcn_batch_size': 8192,
    },
}


def detect_gpu_and_optimize(config):
    """
    Detect GPU type and optimize batch sizes accordingly.
    """
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        if 'a100' in gpu_name:
            opts = CLUSTER_OPTIMIZATIONS['a100']
        elif 'v100' in gpu_name:
            opts = CLUSTER_OPTIMIZATIONS['v100']
        else:
            return config
        
        config.training.genrec_batch_size = opts['genrec_batch_size']
        config.training.rqvae_batch_size = opts['rqvae_batch_size']
        config.training.lightgcn_batch_size = opts['lightgcn_batch_size']
        
        print(f"Detected {gpu_name}, optimized batch sizes: {opts}")
    
    return config


if __name__ == '__main__':
    # Test configuration
    print(f"Scratch directory: {get_scratch_dir()}")
    
    try:
        config = get_cluster_config()
        print(f"Base dir: {config.base_dir}")
        print(f"Output dir: {config.output_dir}")
        print(f"Data path: {config.data.data_path}")
    except ImportError as e:
        print(f"Could not import config (run from project root): {e}")
