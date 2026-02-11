#!/usr/bin/env python3
"""
Baseline experiments for CSJ-ID comparison.

Baselines:
- SASRec: Self-Attentive Sequential Recommendation
- GRU4Rec: GRU-based Sequential Recommendation  
- BPR-MF: Bayesian Personalized Ranking with Matrix Factorization
- Pop: Popularity-based recommendation

Usage:
    python run_baselines.py --dataset beauty
    python run_baselines.py --dataset sports --max_users 50000
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_default_config
from utils import Logger, set_seed, count_parameters
from data import load_amazon_data, BPRDataset
from models import SASRec, GRU4Rec, BPRMF, PopularityBaseline


# =============================================================================
# Datasets for Sequential Models
# =============================================================================

class SequentialDataset(Dataset):
    """Dataset for SASRec/GRU4Rec training."""
    
    def __init__(
        self,
        sequences: Dict[int, List[int]],
        num_items: int,
        max_seq_len: int = 50,
    ):
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.pad_token = num_items
        
        # Create training samples: for each sequence, predict each item from previous items
        self.samples = []
        for user_idx, seq in sequences.items():
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                input_seq = seq[max(0, i - max_seq_len):i]
                target = seq[i]
                self.samples.append((user_idx, input_seq, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user_idx, input_seq, target = self.samples[idx]
        
        # Pad sequence
        seq_len = len(input_seq)
        if seq_len < self.max_seq_len:
            padding = [self.pad_token] * (self.max_seq_len - seq_len)
            input_seq = padding + input_seq
        else:
            input_seq = input_seq[-self.max_seq_len:]
        
        return {
            'user_idx': user_idx,
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'target': target,
        }


class TestDataset(Dataset):
    """Dataset for evaluation."""
    
    def __init__(
        self,
        test_data: Dict[int, Tuple[List[int], int]],
        num_items: int,
        max_seq_len: int = 50,
    ):
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.pad_token = num_items
        
        self.samples = []
        for user_idx, (history, target) in test_data.items():
            if len(history) > 0:
                self.samples.append((user_idx, history, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user_idx, history, target = self.samples[idx]
        
        # Pad/truncate history
        if len(history) < self.max_seq_len:
            padding = [self.pad_token] * (self.max_seq_len - len(history))
            input_seq = padding + history
        else:
            input_seq = history[-self.max_seq_len:]
        
        return {
            'user_idx': user_idx,
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'target': target,
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_sequential_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    logger: Logger,
    model_name: str = "Model",
) -> Dict:
    """Train SASRec or GRU4Rec."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, targets)
            loss = outputs['loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 5 == 0 and logger:
            logger.info(f"{model_name} Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
    
    return history


def train_bprmf(
    model: BPRMF,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    logger: Logger,
) -> Dict:
    """Train BPR-MF."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"BPR-MF Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            users = batch['user'].to(device)
            pos_items = batch['pos_item'].to(device)
            neg_items = batch['neg_item'].to(device)
            
            optimizer.zero_grad()
            outputs = model(users, pos_items, neg_items)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0 and logger:
            logger.info(f"BPR-MF Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
    
    return history


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_sequential_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    ks: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """Evaluate SASRec or GRU4Rec."""
    model.eval()
    
    metrics = {f'Recall@{k}': 0.0 for k in ks}
    metrics.update({f'NDCG@{k}': 0.0 for k in ks})
    metrics['MRR'] = 0.0
    
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].numpy()
            
            outputs = model(input_ids)
            logits = outputs['logits'].cpu().numpy()
            
            for i, target in enumerate(targets):
                # Get ranking
                scores = logits[i]
                ranked_items = np.argsort(-scores)
                rank = np.where(ranked_items == target)[0]
                
                if len(rank) > 0:
                    rank = rank[0] + 1  # 1-indexed
                    
                    # MRR
                    metrics['MRR'] += 1.0 / rank
                    
                    # Recall and NDCG at k
                    for k in ks:
                        if rank <= k:
                            metrics[f'Recall@{k}'] += 1.0
                            metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 1)
                
                total += 1
    
    # Average
    for key in metrics:
        metrics[key] /= total
    
    return metrics


def evaluate_bprmf(
    model: BPRMF,
    test_data: Dict[int, Tuple[List[int], int]],
    device: torch.device,
    ks: List[int] = [1, 5, 10, 20],
    batch_size: int = 256,
) -> Dict[str, float]:
    """Evaluate BPR-MF."""
    model.eval()
    
    metrics = {f'Recall@{k}': 0.0 for k in ks}
    metrics.update({f'NDCG@{k}': 0.0 for k in ks})
    metrics['MRR'] = 0.0
    
    users = list(test_data.keys())
    targets = [test_data[u][1] for u in users]
    total = len(users)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(users), batch_size), desc="Evaluating BPR-MF", leave=False):
            batch_users = users[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            user_tensor = torch.tensor(batch_users, device=device)
            scores = model.get_all_scores(user_tensor).cpu().numpy()
            
            for j, target in enumerate(batch_targets):
                ranked_items = np.argsort(-scores[j])
                rank = np.where(ranked_items == target)[0]
                
                if len(rank) > 0:
                    rank = rank[0] + 1
                    
                    metrics['MRR'] += 1.0 / rank
                    
                    for k in ks:
                        if rank <= k:
                            metrics[f'Recall@{k}'] += 1.0
                            metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 1)
    
    for key in metrics:
        metrics[key] /= total
    
    return metrics


def evaluate_popularity(
    pop_baseline: PopularityBaseline,
    test_data: Dict[int, Tuple[List[int], int]],
    ks: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """Evaluate popularity baseline."""
    metrics = {f'Recall@{k}': 0.0 for k in ks}
    metrics.update({f'NDCG@{k}': 0.0 for k in ks})
    metrics['MRR'] = 0.0
    
    max_k = max(ks)
    top_items = pop_baseline.recommend(k=max_k)
    top_items_set = set(top_items)
    
    total = len(test_data)
    
    for user_idx, (history, target) in test_data.items():
        if target in top_items_set:
            rank = top_items.index(target) + 1
            
            metrics['MRR'] += 1.0 / rank
            
            for k in ks:
                if rank <= k:
                    metrics[f'Recall@{k}'] += 1.0
                    metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 1)
    
    for key in metrics:
        metrics[key] /= total
    
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline Experiments")
    parser.add_argument('--dataset', type=str, default='beauty',
                        choices=['beauty', 'sports', 'toys'])
    parser.add_argument('--max_users', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    # Config
    config = get_default_config()
    config.data.set_dataset(args.dataset)
    config.data.max_users = args.max_users
    config.output_dir = os.path.join(config.base_dir, f'outputs_{args.dataset}')
    os.makedirs(config.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    device = config.get_device()
    
    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.output_dir, f"baselines_{timestamp}.txt")
    logger = Logger(log_path)
    
    logger.info("=" * 70)
    logger.info("Baseline Experiments")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")
    if args.max_users > 0:
        logger.info(f"Max users: {args.max_users}")
    
    # Load data
    logger.info("\n--- Loading Data ---")
    data = load_amazon_data(config.data.data_path, config.data, logger)
    
    # Prepare test data
    test_data = {}
    for user_idx, test_seq in data.test_sequences.items():
        if test_seq:
            history = data.train_sequences.get(user_idx, [])
            test_data[user_idx] = (history, test_seq[0])
    
    logger.info(f"Test users: {len(test_data)}")
    
    results = {}
    
    # ==========================================================================
    # 1. Popularity Baseline
    # ==========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("1. Popularity Baseline")
    logger.info("=" * 70)
    
    item_counts = defaultdict(int)
    for user, items in data.train_sequences.items():
        for item in items:
            item_counts[item] += 1
    
    pop_baseline = PopularityBaseline(dict(item_counts))
    pop_metrics = evaluate_popularity(pop_baseline, test_data)
    results['popularity'] = pop_metrics
    
    logger.info("Popularity Results:")
    for k, v in pop_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # ==========================================================================
    # 2. BPR-MF
    # ==========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("2. BPR-MF")
    logger.info("=" * 70)
    
    bprmf = BPRMF(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=64,
    )
    logger.info(f"BPR-MF parameters: {count_parameters(bprmf):,}")
    
    bpr_dataset = BPRDataset(
        data.train_interactions,
        data.num_users,
        data.num_items,
    )
    bpr_loader = DataLoader(bpr_dataset, batch_size=2048, shuffle=True, num_workers=0)
    
    bpr_history = train_bprmf(bprmf, bpr_loader, args.epochs, lr=0.01, device=device, logger=logger)
    bpr_metrics = evaluate_bprmf(bprmf, test_data, device)
    results['bprmf'] = bpr_metrics
    results['bprmf_history'] = bpr_history
    
    logger.info("BPR-MF Results:")
    for k, v in bpr_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    torch.save(bprmf.state_dict(), os.path.join(config.output_dir, 'bprmf.pt'))
    
    # ==========================================================================
    # 3. GRU4Rec
    # ==========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("3. GRU4Rec")
    logger.info("=" * 70)
    
    gru4rec = GRU4Rec(
        num_items=data.num_items,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        max_seq_len=config.data.max_seq_len,
    )
    logger.info(f"GRU4Rec parameters: {count_parameters(gru4rec):,}")
    
    seq_dataset = SequentialDataset(data.train_sequences, data.num_items, config.data.max_seq_len)
    seq_loader = DataLoader(seq_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset = TestDataset(test_data, data.num_items, config.data.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    gru_history = train_sequential_model(
        gru4rec, seq_loader, args.epochs, lr=1e-3, device=device, logger=logger, model_name="GRU4Rec"
    )
    gru_metrics = evaluate_sequential_model(gru4rec, test_loader, device)
    results['gru4rec'] = gru_metrics
    results['gru4rec_history'] = gru_history
    
    logger.info("GRU4Rec Results:")
    for k, v in gru_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    torch.save(gru4rec.state_dict(), os.path.join(config.output_dir, 'gru4rec.pt'))
    
    # ==========================================================================
    # 4. SASRec
    # ==========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("4. SASRec")
    logger.info("=" * 70)
    
    sasrec = SASRec(
        num_items=data.num_items,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_seq_len=config.data.max_seq_len,
    )
    logger.info(f"SASRec parameters: {count_parameters(sasrec):,}")
    
    sas_history = train_sequential_model(
        sasrec, seq_loader, args.epochs, lr=1e-3, device=device, logger=logger, model_name="SASRec"
    )
    sas_metrics = evaluate_sequential_model(sasrec, test_loader, device)
    results['sasrec'] = sas_metrics
    results['sasrec_history'] = sas_history
    
    logger.info("SASRec Results:")
    for k, v in sas_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    torch.save(sasrec.state_dict(), os.path.join(config.output_dir, 'sasrec.pt'))
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 70)
    
    # Load CSJ-ID results if available
    csj_results_path = os.path.join(config.output_dir, 'results.json')
    if os.path.exists(csj_results_path):
        with open(csj_results_path, 'r') as f:
            csj_results = json.load(f)
        results['csj_id'] = csj_results.get('csj_metrics', {})
        results['semantic_only'] = csj_results.get('sem_metrics', {})
    
    # Print comparison table
    methods = ['popularity', 'bprmf', 'gru4rec', 'sasrec']
    if 'semantic_only' in results:
        methods.append('semantic_only')
    if 'csj_id' in results:
        methods.append('csj_id')
    
    metrics_to_show = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'NDCG@10', 'MRR']
    
    logger.info("\n" + "-" * 100)
    header = f"{'Method':<15}" + "".join([f"{m:<12}" for m in metrics_to_show])
    logger.info(header)
    logger.info("-" * 100)
    
    for method in methods:
        if method in results:
            row = f"{method:<15}"
            for metric in metrics_to_show:
                val = results[method].get(metric, 0)
                row += f"{val:<12.4f}"
            logger.info(row)
    
    logger.info("-" * 100)
    
    # Save results
    results_path = os.path.join(config.output_dir, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
