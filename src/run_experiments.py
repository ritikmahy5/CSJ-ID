#!/usr/bin/env python3
"""
CSJ-ID: Collaborative-Semantic Joint IDs for Generative Recommendation
Main Experiment Runner

ICML 2026 Submission

Usage:
    python run_experiments.py                    # Run full experiment
    python run_experiments.py --quick            # Quick test run
    python run_experiments.py --stage lightgcn   # Run specific stage
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, get_default_config
from utils import Logger, Timer, set_seed, count_parameters
from data import (
    load_amazon_data, ProcessedData,
    ItemEmbeddingDataset, BPRDataset, create_adj_matrix,
    get_semantic_embeddings
)
from models import (
    CSJ_RQVAE, SemanticOnlyRQVAE, CFOnlyRQVAE,
    LightGCN, GenRec,
    compute_codebook_usage
)
from train import train_lightgcn, train_rqvae, train_genrec, GenRecDataset
from evaluate import (
    evaluate_genrec, evaluate_cold_warm_split,
    build_code_to_item_mapping, evaluate_recommendations
)


class ExperimentRunner:
    """Main experiment runner for CSJ-ID."""
    
    def __init__(self, config: ExperimentConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.device = config.get_device()
        
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Output directory: {config.output_dir}")
        
        # Data and models
        self.data: Optional[ProcessedData] = None
        self.semantic_emb: Optional[torch.Tensor] = None
        self.cf_emb: Optional[torch.Tensor] = None
        self.lightgcn: Optional[LightGCN] = None
        self.csj_model: Optional[CSJ_RQVAE] = None
        self.sem_model: Optional[SemanticOnlyRQVAE] = None
        self.cf_model: Optional[CFOnlyRQVAE] = None
        self.genrec_csj: Optional[GenRec] = None
        self.genrec_sem: Optional[GenRec] = None
        
        # Results
        self.results = {}
    
    def run_all(self):
        """Run the complete experiment pipeline."""
        stages = [
            ("Data Loading", self.stage_load_data),
            ("Semantic Embeddings", self.stage_semantic_embeddings),
            ("LightGCN Training", self.stage_train_lightgcn),
            ("RQ-VAE Training", self.stage_train_rqvae),
            ("GenRec Training", self.stage_train_genrec),
            ("Evaluation", self.stage_evaluate),
        ]
        
        if self.config.run_ablations:
            stages.append(("Ablation Studies", self.stage_ablations))
        
        if self.config.run_cold_start:
            stages.append(("Cold-Start Analysis", self.stage_cold_start))
        
        for stage_name, stage_fn in stages:
            self.logger.section(stage_name)
            with Timer(stage_name, self.logger):
                stage_fn()
        
        # Save final results
        self.save_results()
        self.logger.section("Experiment Complete")
        self.print_summary()
    
    def stage_load_data(self):
        """Stage 1: Load and preprocess data."""
        self.data = load_amazon_data(
            self.config.data.data_path,
            self.config.data,
            self.logger
        )
        
        # Save data stats
        self.results['data_stats'] = {
            'num_users': self.data.num_users,
            'num_items': self.data.num_items,
            'num_interactions': self.data.num_interactions,
            'num_cold_users': len(self.data.cold_users),
            'num_warm_users': len(self.data.warm_users),
        }
    
    def stage_semantic_embeddings(self):
        """Stage 2: Extract semantic embeddings."""
        self.logger.info("Extracting semantic embeddings with SentenceTransformer...")
        
        # Cache path includes num_items to handle different samplings
        cache_path = os.path.join(
            self.config.output_dir, 
            f'semantic_embeddings_{self.data.num_items}.pt'
        )
        
        need_regenerate = True
        if os.path.exists(cache_path):
            self.logger.info(f"Loading cached embeddings from {cache_path}")
            self.semantic_emb = torch.load(cache_path)
            # Verify size matches
            if len(self.semantic_emb) == self.data.num_items:
                need_regenerate = False
            else:
                self.logger.info(f"Cache size mismatch ({len(self.semantic_emb)} vs {self.data.num_items}), regenerating...")
        
        if need_regenerate:
            self.semantic_emb = get_semantic_embeddings(
                self.data.item_texts,
                device=self.device,
                batch_size=64,
            )
            torch.save(self.semantic_emb, cache_path)
            self.logger.info(f"Saved embeddings to {cache_path}")
        
        self.logger.info(f"Semantic embeddings shape: {self.semantic_emb.shape}")
        
        # Project to common dimension if needed
        if self.semantic_emb.shape[1] != self.config.model.semantic_dim:
            self.logger.info(f"Projecting from {self.semantic_emb.shape[1]} to {self.config.model.semantic_dim}")
            # Simple linear projection (in practice, could train this)
            self.semantic_emb = self.semantic_emb[:, :self.config.model.semantic_dim]
    
    def stage_train_lightgcn(self):
        """Stage 3: Train LightGCN for CF embeddings."""
        self.logger.info("Building adjacency matrix...")
        
        adj_matrix = create_adj_matrix(
            self.data.train_interactions,
            self.data.num_users,
            self.data.num_items,
        )
        
        self.logger.info(f"Adjacency matrix shape: {adj_matrix.shape}")
        
        # Create model
        self.lightgcn = LightGCN(
            num_users=self.data.num_users,
            num_items=self.data.num_items,
            embedding_dim=self.config.model.cf_dim,
            num_layers=self.config.model.lightgcn_layers,
            dropout=self.config.model.lightgcn_dropout,
        )
        self.lightgcn.set_adj_matrix(adj_matrix.to(self.device))
        
        self.logger.info(f"LightGCN parameters: {count_parameters(self.lightgcn):,}")
        
        # Create data loader
        train_dataset = BPRDataset(
            self.data.train_interactions,
            self.data.num_users,
            self.data.num_items,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.lightgcn_batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        # Train
        self.logger.info(f"Training LightGCN for {self.config.training.lightgcn_epochs} epochs...")
        history = train_lightgcn(
            self.lightgcn,
            train_loader,
            num_epochs=self.config.training.lightgcn_epochs,
            lr=self.config.training.lightgcn_lr,
            weight_decay=self.config.training.lightgcn_weight_decay,
            device=self.device,
            logger=self.logger,
        )
        
        self.results['lightgcn_history'] = history
        
        # Extract CF embeddings
        self.lightgcn.eval()
        with torch.no_grad():
            _, item_emb = self.lightgcn.forward()
            self.cf_emb = item_emb.cpu()
        
        self.logger.info(f"CF embeddings shape: {self.cf_emb.shape}")
        
        # Project to same dimension as semantic
        if self.cf_emb.shape[1] != self.semantic_emb.shape[1]:
            self.logger.info(f"Projecting CF embeddings from {self.cf_emb.shape[1]} to {self.semantic_emb.shape[1]}")
            projection = nn.Linear(self.cf_emb.shape[1], self.semantic_emb.shape[1])
            nn.init.orthogonal_(projection.weight)
            self.cf_emb = projection(self.cf_emb).detach()
        
        # Save
        torch.save(self.cf_emb, os.path.join(self.config.output_dir, 'cf_embeddings.pt'))
        torch.save(self.lightgcn.state_dict(), os.path.join(self.config.output_dir, 'lightgcn.pt'))
    
    def stage_train_rqvae(self):
        """Stage 4: Train RQ-VAE models."""
        # Create dataset
        dataset = ItemEmbeddingDataset(
            self.semantic_emb,
            self.cf_emb,
        )
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.rqvae_batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        input_dim = self.semantic_emb.shape[1]
        
        # ========== Train CSJ-ID Model ==========
        self.logger.info("\n--- Training CSJ-ID RQ-VAE ---")
        self.csj_model = CSJ_RQVAE(
            input_dim=input_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_levels=self.config.model.num_levels,
            codebook_size=self.config.model.codebook_size,
            commitment_weight=self.config.model.commitment_weight,
        )
        self.logger.info(f"CSJ-ID parameters: {count_parameters(self.csj_model):,}")
        
        csj_history = train_rqvae(
            self.csj_model,
            train_loader,
            num_epochs=self.config.training.rqvae_epochs,
            lr=self.config.training.rqvae_lr,
            weight_decay=self.config.training.rqvae_weight_decay,
            device=self.device,
            model_type='csj',
            lambda_sem=self.config.training.lambda_sem,
            logger=self.logger,
        )
        self.results['csj_rqvae_history'] = csj_history
        
        # Get CSJ codes
        self.csj_model.eval()
        with torch.no_grad():
            csj_codes = self.csj_model.get_codes(
                self.semantic_emb.to(self.device),
                self.cf_emb.to(self.device),
                self.config.training.lambda_sem,
            ).cpu()
        
        usage = compute_codebook_usage(csj_codes, self.config.model.codebook_size)
        self.logger.info(f"CSJ-ID codebook usage: {usage['overall_usage']:.2%}")
        self.results['csj_codebook_usage'] = usage
        
        torch.save(csj_codes, os.path.join(self.config.output_dir, 'csj_codes.pt'))
        torch.save(self.csj_model.state_dict(), os.path.join(self.config.output_dir, 'csj_model.pt'))
        
        # ========== Train Semantic-only Baseline ==========
        self.logger.info("\n--- Training Semantic-only RQ-VAE (Baseline) ---")
        self.sem_model = SemanticOnlyRQVAE(
            input_dim=input_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_levels=self.config.model.num_levels,
            codebook_size=self.config.model.codebook_size,
        )
        
        sem_history = train_rqvae(
            self.sem_model,
            train_loader,
            num_epochs=self.config.training.rqvae_epochs,
            lr=self.config.training.rqvae_lr,
            weight_decay=self.config.training.rqvae_weight_decay,
            device=self.device,
            model_type='semantic',
            logger=self.logger,
        )
        self.results['sem_rqvae_history'] = sem_history
        
        # Get semantic-only codes
        self.sem_model.eval()
        with torch.no_grad():
            sem_codes = self.sem_model.get_codes(
                self.semantic_emb.to(self.device)
            ).cpu()
        
        usage = compute_codebook_usage(sem_codes, self.config.model.codebook_size)
        self.logger.info(f"Semantic-only codebook usage: {usage['overall_usage']:.2%}")
        self.results['sem_codebook_usage'] = usage
        
        torch.save(sem_codes, os.path.join(self.config.output_dir, 'sem_codes.pt'))
        torch.save(self.sem_model.state_dict(), os.path.join(self.config.output_dir, 'sem_model.pt'))
        
        # Store codes for later use
        self.csj_codes = csj_codes
        self.sem_codes = sem_codes
    
    def stage_train_genrec(self):
        """Stage 5: Train Generative Recommenders."""
        # ========== Train GenRec with CSJ-ID ==========
        self.logger.info("\n--- Training GenRec with CSJ-ID ---")
        
        csj_train_dataset = GenRecDataset(
            self.data.train_sequences,
            self.csj_codes,
            max_seq_len=self.config.data.max_seq_len,
            num_levels=self.config.model.num_levels,
        )
        csj_train_loader = DataLoader(
            csj_train_dataset,
            batch_size=self.config.training.genrec_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type != 'cpu' else False,
            persistent_workers=True,
        )
        
        self.genrec_csj = GenRec(
            num_codes=self.config.model.codebook_size,
            num_levels=self.config.model.num_levels,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.transformer_layers,
            num_heads=self.config.model.transformer_heads,
            dropout=self.config.model.transformer_dropout,
            max_seq_len=self.config.data.max_seq_len,
        )
        self.logger.info(f"GenRec parameters: {count_parameters(self.genrec_csj):,}")
        
        csj_genrec_history = train_genrec(
            self.genrec_csj,
            csj_train_loader,
            val_loader=None,
            num_epochs=self.config.training.genrec_epochs,
            lr=self.config.training.genrec_lr,
            weight_decay=self.config.training.genrec_weight_decay,
            device=self.device,
            logger=self.logger,
        )
        self.results['csj_genrec_history'] = csj_genrec_history
        
        torch.save(self.genrec_csj.state_dict(), os.path.join(self.config.output_dir, 'genrec_csj.pt'))
        
        # ========== Train GenRec with Semantic-only IDs ==========
        self.logger.info("\n--- Training GenRec with Semantic-only IDs ---")
        
        sem_train_dataset = GenRecDataset(
            self.data.train_sequences,
            self.sem_codes,
            max_seq_len=self.config.data.max_seq_len,
            num_levels=self.config.model.num_levels,
        )
        sem_train_loader = DataLoader(
            sem_train_dataset,
            batch_size=self.config.training.genrec_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type != 'cpu' else False,
            persistent_workers=True,
        )
        
        self.genrec_sem = GenRec(
            num_codes=self.config.model.codebook_size,
            num_levels=self.config.model.num_levels,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.transformer_layers,
            num_heads=self.config.model.transformer_heads,
            dropout=self.config.model.transformer_dropout,
            max_seq_len=self.config.data.max_seq_len,
        )
        
        sem_genrec_history = train_genrec(
            self.genrec_sem,
            sem_train_loader,
            val_loader=None,
            num_epochs=self.config.training.genrec_epochs,
            lr=self.config.training.genrec_lr,
            weight_decay=self.config.training.genrec_weight_decay,
            device=self.device,
            logger=self.logger,
        )
        self.results['sem_genrec_history'] = sem_genrec_history
        
        torch.save(self.genrec_sem.state_dict(), os.path.join(self.config.output_dir, 'genrec_sem.pt'))
    
    def stage_evaluate(self):
        """Stage 6: Evaluate models."""
        self.logger.info("Preparing test data...")
        
        # Create test data dict: user -> (history, target)
        test_data = {}
        for user_idx, test_seq in self.data.test_sequences.items():
            if test_seq:
                history = self.data.train_sequences.get(user_idx, [])
                test_data[user_idx] = (history, test_seq[0])
        
        self.logger.info(f"Test users: {len(test_data)}")
        
        # Build code-to-item mappings
        csj_code_to_item = build_code_to_item_mapping(self.csj_codes)
        sem_code_to_item = build_code_to_item_mapping(self.sem_codes)
        
        # ========== Evaluate CSJ-ID ==========
        self.logger.info("\n--- Evaluating CSJ-ID GenRec ---")
        csj_metrics = evaluate_genrec(
            self.genrec_csj,
            test_data,
            self.csj_codes,
            csj_code_to_item,
            self.device,
            ks=self.config.eval.ks,
            logger=self.logger,
        )
        self.results['csj_metrics'] = csj_metrics
        self.logger.info("CSJ-ID Results:")
        for k, v in csj_metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
        
        # ========== Evaluate Semantic-only ==========
        self.logger.info("\n--- Evaluating Semantic-only GenRec ---")
        sem_metrics = evaluate_genrec(
            self.genrec_sem,
            test_data,
            self.sem_codes,
            sem_code_to_item,
            self.device,
            ks=self.config.eval.ks,
            logger=self.logger,
        )
        self.results['sem_metrics'] = sem_metrics
        self.logger.info("Semantic-only Results:")
        for k, v in sem_metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
        
        # ========== Compute Improvements ==========
        self.logger.info("\n--- Improvements ---")
        improvements = {}
        for k in csj_metrics.keys():
            if sem_metrics[k] > 0:
                imp = (csj_metrics[k] - sem_metrics[k]) / sem_metrics[k] * 100
                improvements[k] = imp
                self.logger.info(f"  {k}: {imp:+.1f}%")
        self.results['improvements'] = improvements
    
    def stage_ablations(self):
        """Run ablation studies."""
        self.logger.info("Running lambda sensitivity analysis...")
        
        lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
        ablation_results = {}
        
        for lambda_val in tqdm(lambdas, desc="Lambda ablation"):
            self.csj_model.eval()
            with torch.no_grad():
                outputs = self.csj_model(
                    self.semantic_emb.to(self.device),
                    self.cf_emb.to(self.device),
                    lambda_sem=lambda_val,
                )
            
            ablation_results[lambda_val] = {
                'loss_sem': outputs['loss_sem'].item(),
                'loss_cf': outputs['loss_cf'].item(),
                'total_loss': outputs['loss'].item(),
            }
            
            self.logger.info(
                f"λ={lambda_val:.1f}: Sem={outputs['loss_sem'].item():.4f}, "
                f"CF={outputs['loss_cf'].item():.4f}, Total={outputs['loss'].item():.4f}"
            )
        
        self.results['lambda_ablation'] = ablation_results
    
    def stage_cold_start(self):
        """Run cold-start analysis."""
        self.logger.info("Running cold-start analysis...")
        
        test_data = {}
        for user_idx, test_seq in self.data.test_sequences.items():
            if test_seq:
                history = self.data.train_sequences.get(user_idx, [])
                test_data[user_idx] = (history, test_seq[0])
        
        csj_code_to_item = build_code_to_item_mapping(self.csj_codes)
        sem_code_to_item = build_code_to_item_mapping(self.sem_codes)
        
        # CSJ-ID cold/warm
        self.logger.info("\nCSJ-ID Cold/Warm Analysis:")
        csj_cold_warm = evaluate_cold_warm_split(
            self.genrec_csj,
            self.data.cold_users,
            self.data.warm_users,
            test_data,
            self.csj_codes,
            csj_code_to_item,
            self.device,
            ks=self.config.eval.ks,
            logger=self.logger,
        )
        self.results['csj_cold_warm'] = csj_cold_warm
        
        if 'cold' in csj_cold_warm:
            self.logger.info("Cold users:")
            for k, v in csj_cold_warm['cold'].items():
                self.logger.info(f"  {k}: {v:.4f}")
        
        if 'warm' in csj_cold_warm:
            self.logger.info("Warm users:")
            for k, v in csj_cold_warm['warm'].items():
                self.logger.info(f"  {k}: {v:.4f}")
        
        # Semantic-only cold/warm
        self.logger.info("\nSemantic-only Cold/Warm Analysis:")
        sem_cold_warm = evaluate_cold_warm_split(
            self.genrec_sem,
            self.data.cold_users,
            self.data.warm_users,
            test_data,
            self.sem_codes,
            sem_code_to_item,
            self.device,
            ks=self.config.eval.ks,
            logger=self.logger,
        )
        self.results['sem_cold_warm'] = sem_cold_warm
    
    def save_results(self):
        """Save all results to file."""
        # Convert tensors to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            else:
                return obj
        
        results_json = convert_for_json(self.results)
        
        results_path = os.path.join(self.config.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
    
    def print_summary(self):
        """Print final summary."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FINAL RESULTS SUMMARY")
        self.logger.info("=" * 70)
        
        # Print comparison table
        headers = ["Metric", "Semantic-only", "CSJ-ID (Ours)", "Improvement"]
        rows = []
        
        for metric in ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'NDCG@10', 'MRR']:
            if metric in self.results.get('csj_metrics', {}):
                sem_val = self.results['sem_metrics'].get(metric, 0)
                csj_val = self.results['csj_metrics'].get(metric, 0)
                imp = self.results['improvements'].get(metric, 0)
                rows.append([
                    metric,
                    f"{sem_val:.4f}",
                    f"{csj_val:.4f}",
                    f"{imp:+.1f}%"
                ])
        
        self.logger.table(headers, rows, "Main Results")
        
        self.logger.info("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CSJ-ID Experiments")
    parser.add_argument('--quick', action='store_true', help='Quick test run with fewer epochs')
    parser.add_argument('--dataset', type=str, default='beauty', 
                        choices=['beauty', 'sports', 'toys'],
                        help='Dataset to use (beauty, sports, toys)')
    parser.add_argument('--max_users', type=int, default=0,
                        help='Maximum users to sample (0=no limit, recommended: 30000 for large datasets)')
    parser.add_argument('--stage', type=str, default=None, 
                        help='Run specific stage (data, semantic, lightgcn, rqvae, genrec, eval)')
    parser.add_argument('--lambda_sem', type=float, default=0.5, help='Lambda for semantic weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Get config
    config = get_default_config()
    config.training.seed = args.seed
    config.training.lambda_sem = args.lambda_sem
    
    # Set dataset
    config.data.set_dataset(args.dataset)
    config.data.max_users = args.max_users
    
    # Reduce LightGCN epochs for large datasets (converges faster)
    if args.dataset in ['sports', 'toys'] and args.max_users == 0:
        config.training.lightgcn_epochs = 30  # Full dataset converges faster
    
    # Update output directory for dataset
    config.output_dir = os.path.join(config.base_dir, f'outputs_{args.dataset}')
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Quick mode settings
    if args.quick:
        config.training.lightgcn_epochs = 10
        config.training.rqvae_epochs = 20
        config.training.genrec_epochs = 5
    
    # Set seed
    set_seed(config.training.seed)
    
    # Create logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.output_dir, f"experiment_{timestamp}.txt")
    logger = Logger(log_path)
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Data path: {config.data.data_path}")
    if args.max_users > 0:
        logger.info(f"Max users: {args.max_users}")
    
    # Run experiments
    runner = ExperimentRunner(config, logger)
    
    if args.stage:
        # Run specific stage
        stage_map = {
            'data': runner.stage_load_data,
            'semantic': runner.stage_semantic_embeddings,
            'lightgcn': runner.stage_train_lightgcn,
            'rqvae': runner.stage_train_rqvae,
            'genrec': runner.stage_train_genrec,
            'eval': runner.stage_evaluate,
        }
        if args.stage in stage_map:
            # Load necessary data for the stage
            runner.stage_load_data()
            if args.stage != 'data':
                runner.stage_semantic_embeddings()
            stage_map[args.stage]()
        else:
            logger.error(f"Unknown stage: {args.stage}")
    else:
        # Run all stages
        runner.run_all()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
