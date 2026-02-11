#!/usr/bin/env python3
"""
Generate publication-ready figures for CSJ-ID paper.

Generates:
1. Main results comparison bar chart
2. Lambda sensitivity plot
3. Cold vs Warm user analysis
4. Training curves
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colors - colorblind friendly palette
COLORS = {
    'csj_id': '#2ecc71',       # Green (ours)
    'semantic_only': '#3498db', # Blue
    'sasrec': '#e74c3c',        # Red
    'gru4rec': '#9b59b6',       # Purple
    'bprmf': '#f39c12',         # Orange
    'popularity': '#95a5a6',    # Gray
}

def load_results(output_dir):
    """Load all results from output directory."""
    results = {}
    
    # Load main results
    results_path = os.path.join(output_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results['main'] = json.load(f)
    
    # Load baseline results
    baseline_path = os.path.join(output_dir, 'baseline_results.json')
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            results['baselines'] = json.load(f)
    
    # Load significance results
    sig_path = os.path.join(output_dir, 'significance_results.json')
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            results['significance'] = json.load(f)
    
    return results


def plot_main_comparison(results, dataset_name, save_dir):
    """Plot main comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    methods = ['popularity', 'bprmf', 'gru4rec', 'sasrec', 'semantic_only', 'csj_id']
    method_labels = ['Pop', 'BPR-MF', 'GRU4Rec', 'SASRec', 'Sem-only', 'CSJ-ID\n(Ours)']
    metrics = ['Recall@10', 'NDCG@10', 'MRR']
    
    # Get data from significance results if available, else from baselines
    if 'significance' in results and 'statistics' in results['significance']:
        stats = results['significance']['statistics']
        data = {}
        errors = {}
        for method in methods:
            if method in stats:
                data[method] = {m: stats[method].get(m, {}).get('mean', 0) for m in metrics}
                errors[method] = {m: stats[method].get(m, {}).get('std', 0) for m in metrics}
            else:
                data[method] = {m: 0 for m in metrics}
                errors[method] = {m: 0 for m in metrics}
    else:
        # Fallback to baseline results
        data = {}
        errors = {}
        for method in methods:
            if 'baselines' in results and method in results['baselines']:
                data[method] = {m: results['baselines'][method].get(m, 0) for m in metrics}
            elif 'main' in results:
                if method == 'csj_id' and 'csj_metrics' in results['main']:
                    data[method] = {m: results['main']['csj_metrics'].get(m, 0) for m in metrics}
                elif method == 'semantic_only' and 'sem_metrics' in results['main']:
                    data[method] = {m: results['main']['sem_metrics'].get(m, 0) for m in metrics}
                else:
                    data[method] = {m: 0 for m in metrics}
            else:
                data[method] = {m: 0 for m in metrics}
            errors[method] = {m: 0 for m in metrics}
    
    x = np.arange(len(methods))
    width = 0.7
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [data[m][metric] for m in methods]
        errs = [errors[m][metric] for m in methods]
        colors = [COLORS[m] for m in methods]
        
        bars = ax.bar(x, values, width, yerr=errs, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
        
        # Highlight CSJ-ID bar
        bars[-1].set_edgecolor('#1a5f2a')
        bars[-1].set_linewidth(2)
        
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=0)
        ax.set_title(metric)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f'{val:.4f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=90)
    
    fig.suptitle(f'{dataset_name} Dataset - Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{dataset_name.lower()}_comparison.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def plot_generative_comparison(results, dataset_name, save_dir):
    """Plot comparison between generative methods only (CSJ-ID vs Semantic-only)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'NDCG@10', 'MRR']
    
    # Get data
    if 'significance' in results and 'statistics' in results['significance']:
        stats = results['significance']['statistics']
        csj_means = [stats['csj_id'].get(m, {}).get('mean', 0) for m in metrics]
        csj_stds = [stats['csj_id'].get(m, {}).get('std', 0) for m in metrics]
        sem_means = [stats['semantic_only'].get(m, {}).get('mean', 0) for m in metrics]
        sem_stds = [stats['semantic_only'].get(m, {}).get('std', 0) for m in metrics]
    else:
        csj_means = [results['main']['csj_metrics'].get(m, 0) for m in metrics]
        sem_means = [results['main']['sem_metrics'].get(m, 0) for m in metrics]
        csj_stds = [0] * len(metrics)
        sem_stds = [0] * len(metrics)
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sem_means, width, yerr=sem_stds, capsize=3,
                   label='Semantic-only', color=COLORS['semantic_only'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, csj_means, width, yerr=csj_stds, capsize=3,
                   label='CSJ-ID (Ours)', color=COLORS['csj_id'], edgecolor='#1a5f2a', linewidth=1.5)
    
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title(f'{dataset_name} - Generative Recommendation Comparison')
    
    # Add improvement percentages
    for i, (sem, csj) in enumerate(zip(sem_means, csj_means)):
        if sem > 0:
            imp = (csj - sem) / sem * 100
            color = 'green' if imp > 0 else 'red'
            ax.annotate(f'{imp:+.0f}%', xy=(i + width/2, csj + csj_stds[i] if csj_stds[i] else csj),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', fontsize=9, color=color, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{dataset_name.lower()}_generative_comparison.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def plot_lambda_ablation(results, dataset_name, save_dir):
    """Plot lambda sensitivity analysis."""
    if 'main' not in results or 'lambda_ablation' not in results['main']:
        print(f"No lambda ablation data for {dataset_name}")
        return
    
    ablation = results['main']['lambda_ablation']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    lambdas = sorted([float(k) for k in ablation.keys()])
    total_loss = [ablation[str(l)]['total_loss'] for l in lambdas]
    sem_loss = [ablation[str(l)]['loss_sem'] for l in lambdas]
    cf_loss = [ablation[str(l)]['loss_cf'] for l in lambdas]
    
    ax.plot(lambdas, total_loss, 'o-', label='Total Loss', color='#2c3e50', linewidth=2, markersize=8)
    ax.plot(lambdas, sem_loss, 's--', label='Semantic Loss', color=COLORS['semantic_only'], linewidth=2, markersize=8)
    ax.plot(lambdas, cf_loss, '^--', label='CF Loss', color=COLORS['bprmf'], linewidth=2, markersize=8)
    
    # Mark optimal lambda
    optimal_idx = np.argmin(total_loss)
    ax.axvline(x=lambdas[optimal_idx], color='green', linestyle=':', alpha=0.7, linewidth=2)
    ax.annotate(f'Optimal λ={lambdas[optimal_idx]}', 
                xy=(lambdas[optimal_idx], total_loss[optimal_idx]),
                xytext=(10, 30), textcoords="offset points",
                fontsize=11, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    ax.set_xlabel('λ (Semantic Weight)')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title(f'{dataset_name} - Lambda Sensitivity Analysis')
    ax.legend()
    ax.set_xticks(lambdas)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{dataset_name.lower()}_lambda_ablation.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def plot_cold_warm_analysis(results, dataset_name, save_dir):
    """Plot cold vs warm user analysis."""
    # Try significance results first
    if 'significance' in results and 'cold_start' in results['significance']:
        cold_data = results['significance']['cold_start']
        csj_cold = np.mean(cold_data['csj_cold'])
        csj_cold_std = np.std(cold_data['csj_cold'])
        sem_cold = np.mean(cold_data['sem_cold'])
        sem_cold_std = np.std(cold_data['sem_cold'])
        csj_warm = np.mean(cold_data['csj_warm'])
        csj_warm_std = np.std(cold_data['csj_warm'])
        sem_warm = np.mean(cold_data['sem_warm'])
        sem_warm_std = np.std(cold_data['sem_warm'])
    elif 'main' in results and 'csj_cold_warm' in results['main']:
        csj_cold = results['main']['csj_cold_warm']['cold'].get('Recall@10', 0)
        sem_cold = results['main']['sem_cold_warm']['cold'].get('Recall@10', 0)
        csj_warm = results['main']['csj_cold_warm']['warm'].get('Recall@10', 0)
        sem_warm = results['main']['sem_cold_warm']['warm'].get('Recall@10', 0)
        csj_cold_std = csj_warm_std = sem_cold_std = sem_warm_std = 0
    else:
        print(f"No cold/warm data for {dataset_name}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(2)
    width = 0.35
    
    sem_values = [sem_cold, sem_warm]
    sem_errors = [sem_cold_std, sem_warm_std]
    csj_values = [csj_cold, csj_warm]
    csj_errors = [csj_cold_std, csj_warm_std]
    
    bars1 = ax.bar(x - width/2, sem_values, width, yerr=sem_errors, capsize=5,
                   label='Semantic-only', color=COLORS['semantic_only'], edgecolor='black')
    bars2 = ax.bar(x + width/2, csj_values, width, yerr=csj_errors, capsize=5,
                   label='CSJ-ID (Ours)', color=COLORS['csj_id'], edgecolor='#1a5f2a', linewidth=2)
    
    ax.set_ylabel('Recall@10')
    ax.set_xticks(x)
    ax.set_xticklabels(['Cold Users', 'Warm Users'])
    ax.legend()
    ax.set_title(f'{dataset_name} - Cold-Start Analysis')
    
    # Add improvement percentages
    for i, (sem, csj) in enumerate(zip(sem_values, csj_values)):
        if sem > 0:
            imp = (csj - sem) / sem * 100
            color = 'green' if imp > 0 else 'red'
            y_pos = max(sem, csj) + max(sem_errors[i], csj_errors[i]) if any([sem_errors[i], csj_errors[i]]) else max(sem, csj)
            ax.annotate(f'{imp:+.1f}%', xy=(i, y_pos),
                       xytext=(0, 10), textcoords="offset points",
                       ha='center', fontsize=12, color=color, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{dataset_name.lower()}_cold_warm.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def plot_training_curves(results, dataset_name, save_dir):
    """Plot training loss curves."""
    if 'main' not in results:
        return
    
    main = results['main']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # LightGCN
    if 'lightgcn_history' in main:
        ax = axes[0]
        loss = main['lightgcn_history']['loss']
        ax.plot(loss, color='#2c3e50', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BPR Loss')
        ax.set_title('LightGCN Training')
    
    # RQ-VAE
    if 'csj_rqvae_history' in main:
        ax = axes[1]
        loss = main['csj_rqvae_history']['loss']
        ax.plot(loss, label='CSJ-ID', color=COLORS['csj_id'], linewidth=2)
        if 'sem_rqvae_history' in main:
            sem_loss = main['sem_rqvae_history']['loss']
            ax.plot(sem_loss, label='Semantic-only', color=COLORS['semantic_only'], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('RQ-VAE Training')
        ax.legend()
    
    # GenRec
    if 'csj_genrec_history' in main:
        ax = axes[2]
        loss = main['csj_genrec_history']['train_loss']
        ax.plot(loss, label='CSJ-ID', color=COLORS['csj_id'], linewidth=2)
        if 'sem_genrec_history' in main:
            sem_loss = main['sem_genrec_history']['train_loss']
            ax.plot(sem_loss, label='Semantic-only', color=COLORS['semantic_only'], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross-Entropy Loss')
        ax.set_title('GenRec Training')
        ax.legend()
    
    fig.suptitle(f'{dataset_name} - Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{dataset_name.lower()}_training_curves.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def create_summary_table(beauty_results, sports_results, save_dir):
    """Create a summary comparison table as a figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Table data
    methods = ['Popularity', 'BPR-MF', 'GRU4Rec', 'SASRec', 'Semantic-only', 'CSJ-ID (Ours)']
    method_keys = ['popularity', 'bprmf', 'gru4rec', 'sasrec', 'semantic_only', 'csj_id']
    
    # Get Beauty data
    beauty_data = []
    if beauty_results and 'significance' in beauty_results and 'statistics' in beauty_results['significance']:
        stats = beauty_results['significance']['statistics']
        for m in method_keys:
            if m in stats and 'Recall@10' in stats[m]:
                r10 = stats[m]['Recall@10']
                beauty_data.append(f"{r10.get('mean', 0):.4f}±{r10.get('std', 0):.4f}")
            else:
                beauty_data.append("-")
    else:
        beauty_data = ["-"] * len(method_keys)
    
    # Get Sports data
    sports_data = []
    if sports_results and 'significance' in sports_results and 'statistics' in sports_results['significance']:
        stats = sports_results['significance']['statistics']
        for m in method_keys:
            if m in stats and 'Recall@10' in stats[m]:
                r10 = stats[m]['Recall@10']
                sports_data.append(f"{r10.get('mean', 0):.4f}±{r10.get('std', 0):.4f}")
            else:
                sports_data.append("-")
    else:
        sports_data = ["-"] * len(method_keys)
    
    table_data = list(zip(methods, beauty_data, sports_data))
    
    if not table_data:
        print("No data available for summary table")
        plt.close()
        return
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'Beauty Recall@10', 'Sports Recall@10'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight CSJ-ID row
    for i in range(3):
        table[(6, i)].set_facecolor('#d5f5e3')
    
    ax.set_title('Recall@10 Comparison Across Datasets (Mean ± Std)', fontsize=14, fontweight='bold', pad=20)
    
    save_path = os.path.join(save_dir, 'summary_table.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def main():
    base_dir = '/Users/ritik/Desktop/Research/ICMLFinal'
    
    # Create figures directory
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)
    
    # Load Beauty results
    print("\nProcessing Beauty dataset...")
    beauty_dir = os.path.join(base_dir, 'outputs')
    beauty_results = load_results(beauty_dir)
    
    if beauty_results:
        plot_main_comparison(beauty_results, 'Beauty', figures_dir)
        plot_generative_comparison(beauty_results, 'Beauty', figures_dir)
        plot_lambda_ablation(beauty_results, 'Beauty', figures_dir)
        plot_cold_warm_analysis(beauty_results, 'Beauty', figures_dir)
        plot_training_curves(beauty_results, 'Beauty', figures_dir)
    
    # Load Sports results
    print("\nProcessing Sports dataset...")
    sports_dir = os.path.join(base_dir, 'outputs_sports')
    sports_results = load_results(sports_dir)
    
    if sports_results:
        plot_main_comparison(sports_results, 'Sports', figures_dir)
        plot_generative_comparison(sports_results, 'Sports', figures_dir)
        plot_lambda_ablation(sports_results, 'Sports', figures_dir)
        plot_cold_warm_analysis(sports_results, 'Sports', figures_dir)
        plot_training_curves(sports_results, 'Sports', figures_dir)
    
    # Create summary table
    if beauty_results and sports_results:
        create_summary_table(beauty_results, sports_results, figures_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
