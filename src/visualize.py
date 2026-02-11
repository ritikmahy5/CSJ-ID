"""
Visualization utilities for CSJ-ID experiments.
Generates publication-quality figures for ICML submission.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.manifold import TSNE

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'csj': '#2ecc71',      # Green
    'semantic': '#3498db',  # Blue
    'cf': '#e74c3c',       # Red
    'random': '#95a5a6',   # Gray
}


def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: str,
    title: str = "Training Loss",
):
    """
    Plot training loss curves for multiple models.
    
    Args:
        histories: Dict of model_name -> {'loss': [...], ...}
        save_path: Path to save figure
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, history in histories.items():
        if 'loss' in history:
            epochs = range(1, len(history['loss']) + 1)
            color = COLORS.get(name.lower().split('_')[0], '#333333')
            ax.plot(epochs, history['loss'], label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_recall_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    ks: List[int] = [1, 5, 10, 20],
):
    """
    Plot bar chart comparing Recall@K for different methods.
    
    Args:
        results: Dict of method_name -> {metric: value}
        save_path: Path to save figure
        ks: List of K values
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    x = np.arange(len(ks))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        recalls = [results[method].get(f'Recall@{k}', 0) for k in ks]
        color = COLORS.get(method.lower().split('_')[0], '#333333')
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=method, color=color, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, recalls):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('K')
    ax.set_ylabel('Recall@K')
    ax.set_title('Recall Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in ks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_lambda_sensitivity(
    ablation_results: Dict[float, Dict[str, float]],
    save_path: str,
):
    """
    Plot lambda sensitivity analysis.
    
    Args:
        ablation_results: Dict of lambda -> {loss_sem, loss_cf, total_loss}
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    lambdas = sorted(ablation_results.keys())
    sem_losses = [ablation_results[l]['loss_sem'] for l in lambdas]
    cf_losses = [ablation_results[l]['loss_cf'] for l in lambdas]
    total_losses = [ablation_results[l]['total_loss'] for l in lambdas]
    
    # Left plot: Individual losses
    ax1.plot(lambdas, sem_losses, 'o-', label='Semantic Loss', color=COLORS['semantic'], linewidth=2, markersize=8)
    ax1.plot(lambdas, cf_losses, 's-', label='CF Loss', color=COLORS['cf'], linewidth=2, markersize=8)
    ax1.set_xlabel('λ (Semantic Weight)')
    ax1.set_ylabel('Reconstruction Loss')
    ax1.set_title('Individual Losses vs λ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Total loss
    ax2.plot(lambdas, total_losses, 'D-', label='Total Loss', color=COLORS['csj'], linewidth=2, markersize=8)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='λ=0.5')
    ax2.set_xlabel('λ (Semantic Weight)')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Total Loss vs λ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_cold_warm_comparison(
    csj_cold_warm: Dict[str, Dict[str, float]],
    sem_cold_warm: Dict[str, Dict[str, float]],
    save_path: str,
    metric: str = 'Recall@10',
):
    """
    Plot cold vs warm user performance comparison.
    
    Args:
        csj_cold_warm: CSJ-ID results for cold/warm users
        sem_cold_warm: Semantic-only results for cold/warm users
        save_path: Path to save figure
        metric: Metric to plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Cold Users', 'Warm Users']
    x = np.arange(len(categories))
    width = 0.35
    
    csj_values = [
        csj_cold_warm.get('cold', {}).get(metric, 0),
        csj_cold_warm.get('warm', {}).get(metric, 0),
    ]
    sem_values = [
        sem_cold_warm.get('cold', {}).get(metric, 0),
        sem_cold_warm.get('warm', {}).get(metric, 0),
    ]
    
    bars1 = ax.bar(x - width/2, sem_values, width, label='Semantic-only', color=COLORS['semantic'], alpha=0.8)
    bars2 = ax.bar(x + width/2, csj_values, width, label='CSJ-ID (Ours)', color=COLORS['csj'], alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('User Type')
    ax.set_ylabel(metric)
    ax.set_title(f'Cold-Start Analysis: {metric}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_code_tsne(
    codes: torch.Tensor,
    save_path: str,
    title: str = "Code Space Visualization",
    labels: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    perplexity: int = 30,
):
    """
    Plot t-SNE visualization of learned codes.
    
    Args:
        codes: Item codes tensor [num_items, num_levels]
        save_path: Path to save figure
        title: Figure title
        labels: Optional labels for coloring
        n_samples: Number of samples to visualize
        perplexity: t-SNE perplexity
    """
    # Sample if too many points
    if len(codes) > n_samples:
        indices = np.random.choice(len(codes), n_samples, replace=False)
        codes = codes[indices]
        if labels is not None:
            labels = labels[indices]
    
    # Convert codes to features
    codes_np = codes.numpy().astype(float)
    
    # Run t-SNE
    print(f"Running t-SNE on {len(codes_np)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings = tsne.fit_transform(codes_np)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if labels is not None:
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, 
                           cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Category')
    else:
        ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=20, 
                  color=COLORS['csj'])
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_table(
    results: Dict[str, Dict[str, float]],
    save_path: str,
):
    """
    Create a visual table of metrics.
    
    Args:
        results: Dict of method_name -> {metric: value}
        save_path: Path to save figure
    """
    methods = list(results.keys())
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'NDCG@10', 'MRR']
    
    # Filter to available metrics
    metrics = [m for m in metrics if any(m in results[method] for method in methods)]
    
    # Build data matrix
    data = []
    for method in methods:
        row = [results[method].get(m, 0) for m in metrics]
        data.append(row)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=[[f'{v:.4f}' for v in row] for row in data],
        rowLabels=methods,
        colLabels=metrics,
        cellLoc='center',
        loc='center',
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Highlight best values
    for j, metric in enumerate(metrics):
        values = [data[i][j] for i in range(len(methods))]
        best_idx = np.argmax(values)
        table[(best_idx + 1, j)].set_facecolor('#d5f5e3')
    
    plt.title('Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_figures(output_dir: str):
    """
    Generate all figures from saved results.
    
    Args:
        output_dir: Directory containing results.json and model files
    """
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)
    
    # Load results
    results_path = os.path.join(output_dir, 'results.json')
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Training curves
    if 'csj_rqvae_history' in results and 'sem_rqvae_history' in results:
        print("\n1. Generating training curves...")
        plot_training_curves(
            {
                'CSJ-ID': results['csj_rqvae_history'],
                'Semantic-only': results['sem_rqvae_history'],
            },
            os.path.join(figures_dir, 'training_curves.png'),
            title='RQ-VAE Training Loss'
        )
    
    # 2. Recall comparison
    if 'csj_metrics' in results and 'sem_metrics' in results:
        print("\n2. Generating recall comparison...")
        plot_recall_comparison(
            {
                'Semantic-only': results['sem_metrics'],
                'CSJ-ID (Ours)': results['csj_metrics'],
            },
            os.path.join(figures_dir, 'recall_comparison.png'),
        )
    
    # 3. Lambda sensitivity
    if 'lambda_ablation' in results:
        print("\n3. Generating lambda sensitivity plot...")
        # Convert string keys to floats
        ablation = {float(k): v for k, v in results['lambda_ablation'].items()}
        plot_lambda_sensitivity(
            ablation,
            os.path.join(figures_dir, 'lambda_sensitivity.png'),
        )
    
    # 4. Cold-warm comparison
    if 'csj_cold_warm' in results and 'sem_cold_warm' in results:
        print("\n4. Generating cold-start analysis plot...")
        plot_cold_warm_comparison(
            results['csj_cold_warm'],
            results['sem_cold_warm'],
            os.path.join(figures_dir, 'cold_warm_comparison.png'),
        )
    
    # 5. t-SNE visualizations
    csj_codes_path = os.path.join(output_dir, 'csj_codes.pt')
    sem_codes_path = os.path.join(output_dir, 'sem_codes.pt')
    
    if os.path.exists(csj_codes_path):
        print("\n5. Generating CSJ-ID t-SNE...")
        csj_codes = torch.load(csj_codes_path)
        plot_code_tsne(
            csj_codes,
            os.path.join(figures_dir, 'csj_tsne.png'),
            title='CSJ-ID Code Space (t-SNE)',
        )
    
    if os.path.exists(sem_codes_path):
        print("\n6. Generating Semantic-only t-SNE...")
        sem_codes = torch.load(sem_codes_path)
        plot_code_tsne(
            sem_codes,
            os.path.join(figures_dir, 'sem_tsne.png'),
            title='Semantic-only Code Space (t-SNE)',
        )
    
    # 6. Metrics table
    if 'csj_metrics' in results and 'sem_metrics' in results:
        print("\n7. Generating metrics table...")
        plot_metrics_table(
            {
                'Semantic-only': results['sem_metrics'],
                'CSJ-ID (Ours)': results['csj_metrics'],
            },
            os.path.join(figures_dir, 'metrics_table.png'),
        )
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate figures for CSJ-ID paper")
    parser.add_argument('--output_dir', type=str, 
                       default='/Users/ritik/Desktop/Research/ICMLFinal/outputs',
                       help='Directory containing results')
    args = parser.parse_args()
    
    generate_all_figures(args.output_dir)
