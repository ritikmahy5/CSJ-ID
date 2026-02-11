#!/usr/bin/env python3
"""
Run additional seeds for Beauty dataset to increase statistical power.

Usage:
    python run_additional_seeds.py --seeds 789 1011
"""

import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_significance import run_single_experiment, compute_statistics
from config import get_default_config
from utils import Logger, set_seed
from data import load_amazon_data


def merge_results(existing_results, new_results):
    """Merge new seed results with existing ones."""
    # Add new seeds to the list
    existing_results['seeds'].extend([r['seed'] for r in new_results])
    existing_results['all_runs'].extend(new_results)
    
    # Recompute statistics
    methods = ['popularity', 'bprmf', 'gru4rec', 'sasrec', 'semantic_only', 'csj_id']
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'NDCG@10', 'MRR']
    
    new_stats = compute_statistics(existing_results['all_runs'], methods, metrics)
    existing_results['statistics'] = new_stats
    
    # Update cold-start analysis
    csj_cold = []
    sem_cold = []
    csj_warm = []
    sem_warm = []
    
    for run in existing_results['all_runs']:
        if 'csj_cold_warm' in run and 'cold' in run['csj_cold_warm']:
            csj_cold.append(run['csj_cold_warm']['cold'].get('Recall@10', 0))
            csj_warm.append(run['csj_cold_warm']['warm'].get('Recall@10', 0))
        if 'sem_cold_warm' in run and 'cold' in run['sem_cold_warm']:
            sem_cold.append(run['sem_cold_warm']['cold'].get('Recall@10', 0))
            sem_warm.append(run['sem_cold_warm']['warm'].get('Recall@10', 0))
    
    existing_results['cold_start'] = {
        'csj_cold': csj_cold,
        'sem_cold': sem_cold,
        'csj_warm': csj_warm,
        'sem_warm': sem_warm,
    }
    
    return existing_results


def print_updated_results(results, logger):
    """Print the updated results with all seeds."""
    methods = ['popularity', 'bprmf', 'gru4rec', 'sasrec', 'semantic_only', 'csj_id']
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'NDCG@10', 'MRR']
    
    stats = results['statistics']
    
    logger.info("\n" + "=" * 100)
    logger.info(f"UPDATED RESULTS WITH {len(results['seeds'])} SEEDS")
    logger.info("=" * 100)
    
    # Header
    header = f"{'Method':<15}"
    for metric in metrics:
        header += f"{metric:<18}"
    logger.info(header)
    logger.info("-" * 100)
    
    # Results rows
    for method in methods:
        if method in stats:
            row = f"{method:<15}"
            for metric in metrics:
                if metric in stats[method]:
                    mean = stats[method][metric]['mean']
                    std = stats[method][metric]['std']
                    row += f"{mean:.4f}±{std:.4f}  "
                else:
                    row += f"{'N/A':<18}"
            logger.info(row)
    
    # Significance tests
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL SIGNIFICANCE (CSJ-ID vs Semantic-only)")
    logger.info("=" * 70)
    
    if 'significance' in stats and 'csj_vs_semantic_only' in stats['significance']:
        sig_data = stats['significance']['csj_vs_semantic_only']
        for metric in metrics:
            if metric in sig_data:
                p = sig_data[metric]['p_value']
                sig = "✓ SIGNIFICANT" if sig_data[metric]['significant'] else "✗ not significant"
                logger.info(f"  {metric}: p={p:.4f} {sig}")
    
    # Cold-start
    logger.info("\n" + "=" * 70)
    logger.info("COLD-START ANALYSIS (Mean ± Std)")
    logger.info("=" * 70)
    
    cold_data = results['cold_start']
    
    csj_cold_mean = np.mean(cold_data['csj_cold'])
    csj_cold_std = np.std(cold_data['csj_cold'])
    sem_cold_mean = np.mean(cold_data['sem_cold'])
    sem_cold_std = np.std(cold_data['sem_cold'])
    
    logger.info(f"\nCold Users Recall@10:")
    logger.info(f"  Semantic-only: {sem_cold_mean:.4f} ± {sem_cold_std:.4f}")
    logger.info(f"  CSJ-ID:        {csj_cold_mean:.4f} ± {csj_cold_std:.4f}")
    
    if len(cold_data['csj_cold']) >= 2:
        _, p = stats['ttest_rel'](cold_data['csj_cold'], cold_data['sem_cold']) if hasattr(stats, 'ttest_rel') else (0, 1)
        try:
            _, p = stats.ttest_rel(cold_data['csj_cold'], cold_data['sem_cold'])
        except:
            from scipy.stats import ttest_rel
            _, p = ttest_rel(cold_data['csj_cold'], cold_data['sem_cold'])
        imp = (csj_cold_mean / sem_cold_mean - 1) * 100 if sem_cold_mean > 0 else 0
        sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
        logger.info(f"  Improvement: {imp:+.1f}% (p={p:.4f}) {sig}")


def main():
    parser = argparse.ArgumentParser(description="Run additional seeds")
    parser.add_argument('--dataset', type=str, default='beauty', choices=['beauty', 'sports'])
    parser.add_argument('--max_users', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[789, 1011])
    args = parser.parse_args()
    
    # Config
    config = get_default_config()
    config.data.set_dataset(args.dataset)
    config.data.max_users = args.max_users
    
    if args.dataset == 'sports':
        config.output_dir = os.path.join(config.base_dir, 'outputs_sports')
    else:
        config.output_dir = os.path.join(config.base_dir, 'outputs')
    
    device = config.get_device()
    
    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.output_dir, f"additional_seeds_{timestamp}.txt")
    logger = Logger(log_path)
    
    logger.info("=" * 70)
    logger.info("Running Additional Seeds")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"New Seeds: {args.seeds}")
    logger.info(f"Device: {device}")
    
    # Load existing results
    existing_path = os.path.join(config.output_dir, 'significance_results.json')
    if os.path.exists(existing_path):
        with open(existing_path, 'r') as f:
            existing_results = json.load(f)
        logger.info(f"Loaded existing results with seeds: {existing_results['seeds']}")
    else:
        logger.info("No existing results found. Starting fresh.")
        existing_results = {'seeds': [], 'all_runs': [], 'statistics': {}, 'cold_start': {}}
    
    # Skip seeds that already exist
    new_seeds = [s for s in args.seeds if s not in existing_results['seeds']]
    if not new_seeds:
        logger.info("All requested seeds already exist. Nothing to do.")
        return
    
    logger.info(f"Running new seeds: {new_seeds}")
    
    # Load data (use first existing seed or first new seed)
    first_seed = existing_results['seeds'][0] if existing_results['seeds'] else new_seeds[0]
    set_seed(first_seed)
    logger.info("\nLoading data...")
    data = load_amazon_data(config.data.data_path, config.data, logger)
    
    # Run experiments for new seeds
    new_results = []
    for seed in new_seeds:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running seed {seed}")
        logger.info(f"{'='*70}")
        
        results = run_single_experiment(config, data, seed, device, logger)
        new_results.append(results)
        
        # Save intermediate results
        merged = merge_results(existing_results.copy(), new_results.copy())
        
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
            json.dump(convert_types(merged), f, indent=2)
        logger.info(f"Intermediate results saved to {save_path}")
    
    # Final merge and save
    final_results = merge_results(existing_results, new_results)
    
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
        json.dump(convert_types(final_results), f, indent=2)
    
    # Print updated results
    print_updated_results(final_results, logger)
    
    logger.info(f"\nFinal results saved to {save_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
