#!/usr/bin/env python3
"""
Quick test to verify CSJ-ID code works correctly.
Run this before the full experiment to catch any issues.

Usage:
    python src/test_setup.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

def test_imports():
    """Test all imports work."""
    print("Testing imports...", end=" ")
    try:
        from config import ExperimentConfig, get_default_config
        from utils import Logger, Timer, set_seed
        from models import CSJ_RQVAE, SemanticOnlyRQVAE, LightGCN, GenRec
        from data import ProcessedData, ItemEmbeddingDataset, BPRDataset
        from train import train_rqvae, train_lightgcn
        from evaluate import evaluate_recommendations, recall_at_k, ndcg_at_k
        print("✓ PASSED")
        return True
    except ImportError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_config():
    """Test configuration."""
    print("Testing config...", end=" ")
    try:
        from config import get_default_config
        config = get_default_config()
        device = config.get_device()
        assert config.model.hidden_dim == 256
        assert config.model.num_levels == 4
        print(f"✓ PASSED (device: {device})")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_csj_rqvae():
    """Test CSJ-RQVAE model."""
    print("Testing CSJ-RQVAE...", end=" ")
    try:
        from models import CSJ_RQVAE
        
        torch.manual_seed(42)
        B, D = 32, 384
        z_sem = torch.randn(B, D)
        z_cf = torch.randn(B, D)
        
        model = CSJ_RQVAE(input_dim=D, hidden_dim=256, num_levels=4, codebook_size=256)
        
        # Forward pass
        outputs = model(z_sem, z_cf, lambda_sem=0.5)
        assert 'loss' in outputs
        assert 'loss_sem' in outputs
        assert 'loss_cf' in outputs
        assert outputs['loss'].requires_grad
        
        # Get codes
        codes = model.get_codes(z_sem, z_cf)
        assert codes.shape == (B, 4)  # B x num_levels
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_semantic_rqvae():
    """Test Semantic-only RQ-VAE."""
    print("Testing Semantic-only RQ-VAE...", end=" ")
    try:
        from models import SemanticOnlyRQVAE
        
        torch.manual_seed(42)
        B, D = 32, 384
        z = torch.randn(B, D)
        
        model = SemanticOnlyRQVAE(input_dim=D)
        outputs = model(z)
        
        assert 'loss' in outputs
        assert outputs['loss'].requires_grad
        
        codes = model.get_codes(z)
        assert codes.shape == (B, 4)
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_lightgcn():
    """Test LightGCN model."""
    print("Testing LightGCN...", end=" ")
    try:
        from models import LightGCN
        from data import create_adj_matrix
        
        torch.manual_seed(42)
        num_users, num_items = 100, 50
        
        # Create fake interactions
        interactions = [(i % num_users, i % num_items) for i in range(500)]
        
        # Create adjacency matrix
        adj = create_adj_matrix(interactions, num_users, num_items)
        
        # Create model
        model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3)
        model.set_adj_matrix(adj)
        
        # Forward pass
        user_emb, item_emb = model()
        assert user_emb.shape == (num_users, 64)
        assert item_emb.shape == (num_items, 64)
        
        # BPR loss
        users = torch.randint(0, num_users, (16,))
        pos = torch.randint(0, num_items, (16,))
        neg = torch.randint(0, num_items, (16,))
        loss = model.bpr_loss(users, pos, neg)
        assert loss.requires_grad
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_genrec():
    """Test GenRec model."""
    print("Testing GenRec...", end=" ")
    try:
        from models import GenRec
        
        torch.manual_seed(42)
        B = 16
        num_codes, num_levels = 256, 4
        seq_len = 10
        
        model = GenRec(num_codes=num_codes, num_levels=num_levels, hidden_dim=256)
        
        # Create input
        input_codes = torch.randint(0, num_codes, (B, seq_len, num_levels))
        target_codes = torch.randint(0, num_codes, (B, num_levels))
        
        # Forward pass
        outputs = model(input_codes, target_codes)
        assert 'loss' in outputs
        assert outputs['loss'].requires_grad
        
        # Generation
        pred = model.generate(input_codes)
        assert pred.shape == (B, num_levels)
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_evaluation():
    """Test evaluation metrics."""
    print("Testing evaluation...", end=" ")
    try:
        from evaluate import evaluate_recommendations, recall_at_k, ndcg_at_k, mrr
        
        # Test individual metrics
        pred = [3, 1, 4, 1, 5]
        assert recall_at_k(pred, 3, k=1) == 1.0
        assert recall_at_k(pred, 3, k=5) == 1.0
        assert recall_at_k(pred, 9, k=5) == 0.0
        
        assert ndcg_at_k(pred, 3, k=1) == 1.0
        assert ndcg_at_k(pred, 1, k=5) > 0
        
        # Test batch evaluation
        predictions = [[1, 2, 3], [3, 2, 1], [5, 6, 7]]
        ground_truth = [1, 3, 10]
        
        metrics = evaluate_recommendations(predictions, ground_truth, ks=[1, 3])
        assert 'Recall@1' in metrics
        assert 'NDCG@1' in metrics
        assert 'MRR' in metrics
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_data_loading():
    """Test data loading (if data file exists)."""
    print("Testing data loading...", end=" ")
    try:
        data_path = '/Users/ritik/Desktop/Research/ICMLFinal/Beauty.json.gz'
        
        if not os.path.exists(data_path):
            print("⊘ SKIPPED (data file not found)")
            return True
        
        from data import load_amazon_data
        from config import DataConfig
        
        config = DataConfig()
        config.data_path = data_path
        
        # Load just a few lines to test
        import gzip
        import json
        
        with gzip.open(data_path, 'rt') as f:
            first_line = f.readline()
            review = json.loads(first_line)
        
        assert 'reviewerID' in review
        assert 'asin' in review
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_logger():
    """Test logging utilities."""
    print("Testing logger...", end=" ")
    try:
        from utils import Logger, Timer
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            log_path = f.name
        
        logger = Logger(log_path, name="test")
        logger.info("Test message")
        logger.section("Test Section")
        logger.metrics({'loss': 0.5, 'acc': 0.9})
        
        # Check file was written
        with open(log_path, 'r') as f:
            content = f.read()
        assert "Test message" in content
        
        os.remove(log_path)
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("CSJ-ID Setup Test")
    print("=" * 60)
    print()
    
    tests = [
        test_imports,
        test_config,
        test_csj_rqvae,
        test_semantic_rqvae,
        test_lightgcn,
        test_genrec,
        test_evaluation,
        test_data_loading,
        test_logger,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests passed! Ready to run experiments.")
        print("\nRun full experiment:")
        print("  python src/run_experiments.py")
        print("\nRun quick test:")
        print("  python src/run_experiments.py --quick")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before running experiments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
