#!/usr/bin/env python3
"""
Statistical Significance Testing for CSJ-ID Experiments.

Runs experiments with multiple seeds and computes:
- Mean and standard deviation for all metrics
- Statistical significance (paired t-test) vs baselines

Usage:
    python run_significance.py --dataset beauty --seeds 42 123 456
    python run_significance.py --dataset sports --max_users 50000 --seeds 42 123 456
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_default_config
from utils import Logger, set_seed, count_parameters, Timer
from data import (
    load_amazon_data, ProcessedData,
    ItemEmbeddingDataset, BPRDataset, create_adj_matrix,
    get_semantic_embeddings
)
from models import (
    CSJ_RQVAE, SemanticOnlyRQVAE, LightGCN, GenRec,
    SASRec, GRU4Rec, BPRMF, PopularityBaseline,
    compute_codebook_usage
)
from train import train_lightgcn, train_rqvae, train_genrec, GenRecDataset
from evaluate import (
    evaluate_genrec, evaluate_cold_warm_split,
    build_code_to_item_mapping
)
from run_baselines import (
    SequentialDataset, TestDataset,
    train_sequential_model, train_bprmf,
    evaluate_sequential_model, evaluate_bprmf, evaluate_popularity
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm


def run_single_experiment(
    config,
    data: ProcessedData,
    seed: int,
    device: torch.device,
    logger: Logger,
) -> Dict:
    """Run a single experiment with given seed and return all metrics."""
    
    set_seed(seed)
    logger.info(f"\n{'='*70}")
    logger.info(f"Running experiment with seed={seed}")
    logger.info(f"{'='*70}")
    
    results = {'seed': seed}
    
    # ==========================================================================
    # Stage 1: Semantic Embeddings (deterministic, no need to recompute)
    # ==========================================================================
    cache_path = os.path.join(config.output_dir, f'semantic_embeddings_{data.num_items}.pt')
    if os.path.exists(cache_path):
        semantic_emb = torch.load(cache_path)
    else:
        semantic_emb = get_semantic_embeddings(data.item_texts, device=device, batch_size=64)
        torch.save(semantic_emb, cache_path)
    
    if semantic_emb.shape[1] != config.model.semantic_dim:
        semantic_emb = semantic_emb[:, :config.model.semantic_dim]
    
    # ==========================================================================
    # Stage 2: LightGCN for CF Embeddings
    # ==========================================================================
    logger.info(f"[Seed {seed}] Training LightGCN...")
    
    adj_matrix = create_adj_matrix(
        data.train_interactions, data.num_users, data.num_items
    )
    
    lightgcn = LightGCN(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=config.model.cf_dim,
        num_layers=config.model.lightgcn_layers,
        dropout=config.model.lightgcn_dropout,
    )
    lightgcn.set_adj_matrix(adj_matrix.to(device))
    
    bpr_dataset = BPRDataset(data.train_interactions, data.num_users, data.num_items)
    bpr_loader = DataLoader(bpr_dataset, batch_size=config.training.lightgcn_batch_size, shuffle=True)
    
    train_lightgcn(
        lightgcn, bpr_loader,
        num_epochs=config.training.lightgcn_epochs,
        lr=config.training.lightgcn_lr,
        weight_decay=config.training.lightgcn_weight_decay,
        device=device, logger=None  # Suppress verbose logging
    )
    
    lightgcn.eval()
    with torch.no_grad():
        _, item_emb = lightgcn.forward()
        cf_emb = item_emb.cpu()
    
    if cf_emb.shape[1] != semantic_emb.shape[1]:
        projection = nn.Linear(cf_emb.shape[1], semantic_emb.shape[1])
        nn.init.orthogonal_(projection.weight)
        cf_emb = projection(cf_emb).detach()
    
    # ==========================================================================
    # Stage 3: RQ-VAE Training
    # ==========================================================================
    logger.info(f"[Seed {seed}] Training RQ-VAE models...")
    
    emb_dataset = ItemEmbeddingDataset(semantic_emb, cf_emb)
    emb_loader = DataLoader(emb_dataset, batch_size=config.training.rqvae_batch_size, shuffle=True)
    
    input_dim = semantic_emb.shape[1]
    
    # CSJ-ID
    csj_model = CSJ_RQVAE(
        input_dim=input_dim,
        hidden_dim=config.model.hidden_dim,
        num_levels=config.model.num_levels,
        codebook_size=config.model.codebook_size,
        commitment_weight=config.model.commitment_weight,
    )
    train_rqvae(
        csj_model, emb_loader,
        num_epochs=config.training.rqvae_epochs,
        lr=config.training.rqvae_lr,
        weight_decay=config.training.rqvae_weight_decay,
        device=device, model_type='csj',
        lambda_sem=config.training.lambda_sem,
        logger=None
    )
    
    csj_model.eval()
    with torch.no_grad():
        csj_codes = csj_model.get_codes(
            semantic_emb.to(device), cf_emb.to(device), config.training.lambda_sem
        ).cpu()
    
    # Semantic-only
    sem_model = SemanticOnlyRQVAE(
        input_dim=input_dim,
        hidden_dim=config.model.hidden_dim,
        num_levels=config.model.num_levels,
        codebook_size=config.model.codebook_size,
    )
    train_rqvae(
        sem_model, emb_loader,
        num_epochs=config.training.rqvae_epochs,
        lr=config.training.rqvae_lr,
        weight_decay=config.training.rqvae_weight_decay,
        device=device, model_type='semantic',
        logger=None
    )
    
    sem_model.eval()
    with torch.no_grad():
        sem_codes = sem_model.get_codes(semantic_emb.to(device)).cpu()
    
    # ==========================================================================
    # Stage 4: GenRec Training
    # ==========================================================================
    logger.info(f"[Seed {seed}] Training GenRec models...")
    
    # CSJ-ID GenRec
    csj_train_dataset = GenRecDataset(
        data.train_sequences, csj_codes,
        max_seq_len=config.data.max_seq_len,
        num_levels=config.model.num_levels,
    )
    csj_train_loader = DataLoader(
        csj_train_dataset, batch_size=config.training.genrec_batch_size,
        shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    genrec_csj = GenRec(
        num_codes=config.model.codebook_size,
        num_levels=config.model.num_levels,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.transformer_layers,
        num_heads=config.model.transformer_heads,
        dropout=config.model.transformer_dropout,
        max_seq_len=config.data.max_seq_len,
    )
    train_genrec(
        genrec_csj, csj_train_loader,
        val_loader=None,
        num_epochs=config.training.genrec_epochs,
        lr=config.training.genrec_lr,
        weight_decay=config.training.genrec_weight_decay,
        device=device, logger=None
    )
    
    # Semantic GenRec
    sem_train_dataset = GenRecDataset(
        data.train_sequences, sem_codes,
        max_seq_len=config.data.max_seq_len,
        num_levels=config.model.num_levels,
    )
    sem_train_loader = DataLoader(
        sem_train_dataset, batch_size=config.training.genrec_batch_size,
        shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    genrec_sem = GenRec(
        num_codes=config.model.codebook_size,
        num_levels=config.model.num_levels,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.transformer_layers,
        num_heads=config.model.transformer_heads,
        dropout=config.model.transformer_dropout,
        max_seq_len=config.data.max_seq_len,
    )
    train_genrec(
        genrec_sem, sem_train_loader,
        val_loader=None,
        num_epochs=config.training.genrec_epochs,
        lr=config.training.genrec_lr,
        weight_decay=config.training.genrec_weight_decay,
        device=device, logger=None
    )
    
    # ==========================================================================
    # Stage 5: Evaluation
    # ==========================================================================
    logger.info(f"[Seed {seed}] Evaluating...")
    
    test_data = {}
    for user_idx, test_seq in data.test_sequences.items():
        if test_seq:
            history = data.train_sequences.get(user_idx, [])
            test_data[user_idx] = (history, test_seq[0])
    
    csj_code_to_item = build_code_to_item_mapping(csj_codes)
    sem_code_to_item = build_code_to_item_mapping(sem_codes)
    
    # CSJ-ID metrics
    csj_metrics = evaluate_genrec(
        genrec_csj, test_data, csj_codes, csj_code_to_item,
        device, ks=[1, 5, 10, 20], logger=None
    )
    results['csj_id'] = csj_metrics
    
    # Semantic-only metrics
    sem_metrics = evaluate_genrec(
        genrec_sem, test_data, sem_codes, sem_code_to_item,
        device, ks=[1, 5, 10, 20], logger=None
    )
    results['semantic_only'] = sem_metrics
    
    # Cold/Warm analysis
    csj_cold_warm = evaluate_cold_warm_split(
        genrec_csj, data.cold_users, data.warm_users,
        test_data, csj_codes, csj_code_to_item,
        device, ks=[1, 5, 10, 20], logger=None
    )
    results['csj_cold_warm'] = csj_cold_warm
    
    sem_cold_warm = evaluate_cold_warm_split(
        genrec_sem, data.cold_users, data.warm_users,
        test_data, sem_codes, sem_code_to_item,
        device, ks=[1, 5, 10, 20], logger=None
    )
    results['sem_cold_warm'] = sem_cold_warm
    
    # ==========================================================================
    # Stage 6: Baselines (with same seed)
    # ==========================================================================
    logger.info(f"[Seed {seed}] Training baselines...")
    
    # Popularity (deterministic)
    item_counts = defaultdict(int)
    for user, items in data.train_sequences.items():
        for item in items:
            item_counts[item] += 1
    pop_baseline = PopularityBaseline(dict(item_counts))
    results['popularity'] = evaluate_popularity(pop_baseline, test_data)
    
    # BPR-MF
    bprmf = BPRMF(num_users=data.num_users, num_items=data.num_items, embedding_dim=64)
    bpr_loader2 = DataLoader(bpr_dataset, batch_size=2048, shuffle=True)
    train_bprmf(bprmf, bpr_loader2, num_epochs=30, lr=0.01, device=device, logger=None)
    results['bprmf'] = evaluate_bprmf(bprmf, test_data, device)
    
    # Sequential models
    seq_dataset = SequentialDataset(data.train_sequences, data.num_items, config.data.max_seq_len)
    seq_loader = DataLoader(seq_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset = TestDataset(test_data, data.num_items, config.data.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # GRU4Rec
    gru4rec = GRU4Rec(
        num_items=data.num_items, hidden_dim=256, num_layers=2,
        dropout=0.1, max_seq_len=config.data.max_seq_len
    )
    train_sequential_model(gru4rec, seq_loader, num_epochs=30, lr=1e-3, device=device, logger=None, model_name="GRU4Rec")
    results['gru4rec'] = evaluate_sequential_model(gru4rec, test_loader, device)
    
    # SASRec
    sasrec = SASRec(
        num_items=data.num_items, hidden_dim=256, num_layers=2,
        num_heads=4, dropout=0.1, max_seq_len=config.data.max_seq_len
    )
    train_sequential_model(sasrec, seq_loader, num_epochs=30, lr=1e-3, device=device, logger=None, model_name="SASRec")
    results['sasrec'] = evaluate_sequential_model(sasrec, test_loader, device)
    
    logger.info(f"[Seed {seed}] Complete!")
    
    return results


def compute_statistics(all_results: List[Dict], methods: List[str], metrics: List[str]) -> Dict:
    """Compute mean, std, and significance tests."""
    
    stats_results = {}
    
    for method in methods:
        stats_results[method] = {}
        for metric in metrics:
            values = []
            for run in all_results:
                if method in run and metric in run[method]:
                    values.append(run[method][metric])
            
            if values:
                stats_results[method][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values,
                }
    
    # Compute p-values (CSJ-ID vs each baseline)
    if 'csj_id' in stats_results:
        stats_results['significance'] = {}
        for method in methods:
            if method != 'csj_id' and method in stats_results:
                stats_results['significance'][f'csj_vs_{method}'] = {}
                for metric in metrics:
                    if metric in stats_results['csj_id'] and metric in stats_results[method]:
                        csj_vals = stats_results['csj_id'][metric]['values']
                        other_vals = stats_results[method][metric]['values']
                        
                        if len(csj_vals) >= 2 and len(other_vals) >= 2:
                            # Paired t-test
                            t_stat, p_value = stats.ttest_rel(csj_vals, other_vals)
                            stats_results['significance'][f'csj_vs_{method}'][metric] = {
                                't_stat': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                            }
    
    return stats_results


def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Testing")
    parser.add_argument('--dataset', type=str, default='beauty', choices=['beauty', 'sports', 'toys'])
    parser.add_argument('--max_users', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()
    
    # Config
    config = get_default_config()
    config.data.set_dataset(args.dataset)
    config.data.max_users = args.max_users
    config.output_dir = os.path.join(config.base_dir, f'outputs_{args.dataset}')
    os.makedirs(config.output_dir, exist_ok=True)
    
    device = config.get_device()
    
    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.output_dir, f"significance_{timestamp}.txt")
    logger = Logger(log_path)
    
    logger.info("=" * 70)
    logger.info("Statistical Significance Testing")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Device: {device}")
    
    # Load data once (use first seed for data split)
    set_seed(args.seeds[0])
    logger.info("\nLoading data...")
    data = load_amazon_data(config.data.data_path, config.data, logger)
    
    # Run experiments for each seed
    all_results = []
    for seed in args.seeds:
        with Timer(f"Seed {seed}", logger):
            results = run_single_experiment(config, data, seed, device, logger)
            all_results.append(results)
    
    # Compute statistics
    methods = ['popularity', 'bprmf', 'gru4rec', 'sasrec', 'semantic_only', 'csj_id']
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'NDCG@10', 'MRR']
    
    stats_results = compute_statistics(all_results, methods, metrics)
    
    # Print results table
    logger.info("\n" + "=" * 100)
    logger.info("RESULTS WITH STATISTICAL SIGNIFICANCE")
    logger.info("=" * 100)
    
    # Header
    header = f"{'Method':<15}"
    for metric in metrics:
        header += f"{metric:<18}"
    logger.info(header)
    logger.info("-" * 100)
    
    # Results rows
    for method in methods:
        if method in stats_results:
            row = f"{method:<15}"
            for metric in metrics:
                if metric in stats_results[method]:
                    mean = stats_results[method][metric]['mean']
                    std = stats_results[method][metric]['std']
                    row += f"{mean:.4f}±{std:.4f}  "
                else:
                    row += f"{'N/A':<18}"
            logger.info(row)
    
    logger.info("-" * 100)
    
    # Significance tests
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL SIGNIFICANCE (CSJ-ID vs Baselines)")
    logger.info("=" * 70)
    
    if 'significance' in stats_results:
        for comparison, metrics_dict in stats_results['significance'].items():
            logger.info(f"\n{comparison}:")
            for metric, sig_results in metrics_dict.items():
                sig_marker = "✓" if sig_results['significant'] else "✗"
                logger.info(f"  {metric}: p={sig_results['p_value']:.4f} {sig_marker}")
    
    # Cold-start analysis
    logger.info("\n" + "=" * 70)
    logger.info("COLD-START ANALYSIS (Mean ± Std)")
    logger.info("=" * 70)
    
    cold_metrics = {'csj_cold': [], 'sem_cold': [], 'csj_warm': [], 'sem_warm': []}
    for run in all_results:
        if 'csj_cold_warm' in run and 'cold' in run['csj_cold_warm']:
            cold_metrics['csj_cold'].append(run['csj_cold_warm']['cold'].get('Recall@10', 0))
            cold_metrics['csj_warm'].append(run['csj_cold_warm']['warm'].get('Recall@10', 0))
        if 'sem_cold_warm' in run and 'cold' in run['sem_cold_warm']:
            cold_metrics['sem_cold'].append(run['sem_cold_warm']['cold'].get('Recall@10', 0))
            cold_metrics['sem_warm'].append(run['sem_cold_warm']['warm'].get('Recall@10', 0))
    
    logger.info(f"\nCold Users Recall@10:")
    logger.info(f"  Semantic-only: {np.mean(cold_metrics['sem_cold']):.4f} ± {np.std(cold_metrics['sem_cold']):.4f}")
    logger.info(f"  CSJ-ID:        {np.mean(cold_metrics['csj_cold']):.4f} ± {np.std(cold_metrics['csj_cold']):.4f}")
    if len(cold_metrics['sem_cold']) >= 2:
        _, p = stats.ttest_rel(cold_metrics['csj_cold'], cold_metrics['sem_cold'])
        logger.info(f"  Improvement:   {(np.mean(cold_metrics['csj_cold'])/np.mean(cold_metrics['sem_cold'])-1)*100:.1f}% (p={p:.4f})")
    
    logger.info(f"\nWarm Users Recall@10:")
    logger.info(f"  Semantic-only: {np.mean(cold_metrics['sem_warm']):.4f} ± {np.std(cold_metrics['sem_warm']):.4f}")
    logger.info(f"  CSJ-ID:        {np.mean(cold_metrics['csj_warm']):.4f} ± {np.std(cold_metrics['csj_warm']):.4f}")
    if len(cold_metrics['sem_warm']) >= 2:
        _, p = stats.ttest_rel(cold_metrics['csj_warm'], cold_metrics['sem_warm'])
        logger.info(f"  Improvement:   {(np.mean(cold_metrics['csj_warm'])/np.mean(cold_metrics['sem_warm'])-1)*100:.1f}% (p={p:.4f})")
    
    # Save all results
    save_results = {
        'seeds': args.seeds,
        'all_runs': all_results,
        'statistics': stats_results,
        'cold_start': {
            'csj_cold': cold_metrics['csj_cold'],
            'sem_cold': cold_metrics['sem_cold'],
            'csj_warm': cold_metrics['csj_warm'],
            'sem_warm': cold_metrics['sem_warm'],
        }
    }
    
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    save_path = os.path.join(config.output_dir, 'significance_results.json')
    with open(save_path, 'w') as f:
        json.dump(convert_types(save_results), f, indent=2)
    
    logger.info(f"\nResults saved to {save_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
