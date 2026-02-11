"""
Environment-aware configuration wrapper for CSJ-ID.

This module provides automatic environment detection and path configuration
for running experiments both locally and on the NEU Discovery cluster.

Usage:
    from config_auto import get_config
    config = get_config()
    # Paths are automatically configured based on environment
"""

import os
import sys


def detect_environment():
    """
    Detect the runtime environment.
    
    Returns:
        tuple: (base_dir, is_cluster, env_name)
    """
    # Check if on NEU Discovery cluster
    if os.path.exists('/scratch') and 'USER' in os.environ:
        user = os.environ['USER']
        cluster_path = f'/scratch/{user}/research/ICMLFinal'
        if os.path.exists(cluster_path) or os.path.exists(f'/scratch/{user}'):
            return cluster_path, True, 'NEU Discovery'
    
    # Check for common local paths
    local_paths = [
        '/Users/ritik/Desktop/Research/ICMLFinal',
        os.path.expanduser('~/Desktop/Research/ICMLFinal'),
        os.path.expanduser('~/research/ICMLFinal'),
        os.getcwd(),
    ]
    
    for path in local_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'src')):
            return path, False, 'Local'
    
    # Fallback to current directory
    return os.getcwd(), False, 'Unknown'


def get_config(dataset='beauty'):
    """
    Get experiment configuration with environment-aware paths.
    
    Args:
        dataset: Dataset name ('beauty', 'sports', 'toys')
    
    Returns:
        ExperimentConfig with correct paths for current environment
    """
    base_dir, is_cluster, env_name = detect_environment()
    
    # Import config module
    sys.path.insert(0, os.path.join(base_dir, 'src'))
    from config import get_default_config
    
    config = get_default_config()
    
    # Update paths for current environment
    config.base_dir = base_dir
    config.output_dir = os.path.join(base_dir, f'outputs_{dataset}')
    
    # Update dataset paths
    config.data.DATASET_PATHS = {
        'beauty': os.path.join(base_dir, 'Beauty.json.gz'),
        'sports': os.path.join(base_dir, 'Sports.json.gz'),
        'toys': os.path.join(base_dir, 'Toys.json.gz'),
    }
    
    # Set the requested dataset
    config.data.set_dataset(dataset)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Print environment info
    print(f"Environment: {env_name}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Dataset: {dataset} -> {config.data.data_path}")
    
    # Cluster-specific optimizations
    if is_cluster:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            print(f"GPU: {gpu_name}")
            
            # Optimize batch sizes for cluster GPUs
            if 'a100' in gpu_name:
                config.training.genrec_batch_size = 512
                config.training.rqvae_batch_size = 1024
                config.training.lightgcn_batch_size = 8192
                print("Optimized for A100 GPU")
            elif 'v100' in gpu_name:
                config.training.genrec_batch_size = 256
                config.training.rqvae_batch_size = 512
                config.training.lightgcn_batch_size = 4096
                print("Optimized for V100 GPU")
    
    return config


if __name__ == '__main__':
    # Test configuration
    base_dir, is_cluster, env_name = detect_environment()
    print(f"Detected environment: {env_name}")
    print(f"Base directory: {base_dir}")
    print(f"Is cluster: {is_cluster}")
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = 'beauty'
    
    print(f"\nGetting config for dataset: {dataset}")
    config = get_config(dataset)
    print(f"\nDevice: {config.get_device()}")
