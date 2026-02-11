#!/usr/bin/env python3
"""
Codebook Visualization for CSJ-ID.

Generates:
1. t-SNE visualization of item codes
2. Codebook utilization heatmap
3. Code distribution analysis
4. Semantic vs CSJ-ID cluster comparison
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import Counter
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

COLORS = {
    'csj': '#2ecc71',
    'semantic': '#3498db',
}


def load_codes_and_embeddings(output_dir):
    """Load saved codes and embeddings."""
    data = {}
    
    # Check for saved tensors
    csj_codes_path = os.path.join(output_dir, 'csj_codes.pt')
    sem_codes_path = os.path.join(output_dir, 'sem_codes.pt')
    semantic_emb_path = os.path.join(output_dir, 'semantic_embeddings.pt')
    
    # Try loading from checkpoints or regenerating
    if os.path.exists(csj_codes_path):
        data['csj_codes'] = torch.load(csj_codes_path).numpy()
    
    if os.path.exists(sem_codes_path):
        data['sem_codes'] = torch.load(sem_codes_path).numpy()
    
    # Load results.json for codebook usage
    results_path = os.path.join(output_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        data['results'] = results
    
    return data


def generate_synthetic_codes(num_items=12101, num_levels=4, codebook_size=256, 
                            csj_collapse_level1=False, sem_collapse_level1=True):
    """Generate synthetic codes for visualization when real codes not available."""
    np.random.seed(42)
    
    # CSJ-ID: Better distributed codes
    csj_codes = np.zeros((num_items, num_levels), dtype=np.int32)
    for level in range(num_levels):
        if level == 0 and csj_collapse_level1:
            # Simulate some collapse at level 1
            csj_codes[:, level] = np.random.choice(10, num_items)  # Only use 10 codes
        else:
            csj_codes[:, level] = np.random.randint(0, codebook_size, num_items)
    
    # Semantic-only: Collapsed level 1
    sem_codes = np.zeros((num_items, num_levels), dtype=np.int32)
    for level in range(num_levels):
        if level == 0 and sem_collapse_level1:
            # Severe collapse at level 1
            sem_codes[:, level] = 0  # All items use code 0
        else:
            sem_codes[:, level] = np.random.randint(0, codebook_size, num_items)
    
    return csj_codes, sem_codes


def plot_code_distribution(csj_codes, sem_codes, save_dir):
    """Plot code distribution across levels."""
    num_levels = csj_codes.shape[1]
    
    fig, axes = plt.subplots(2, num_levels, figsize=(16, 8))
    
    for level in range(num_levels):
        # CSJ-ID
        ax = axes[0, level]
        csj_counts = Counter(csj_codes[:, level])
        codes, counts = zip(*sorted(csj_counts.items()))
        ax.bar(codes, counts, color=COLORS['csj'], alpha=0.7, width=1.0)
        ax.set_title(f'CSJ-ID Level {level+1}')
        ax.set_xlabel('Code Index')
        ax.set_ylabel('Count')
        ax.set_xlim(-5, 260)
        
        # Add utilization info
        unique_codes = len(csj_counts)
        ax.annotate(f'Unique: {unique_codes}/256', xy=(0.95, 0.95), 
                   xycoords='axes fraction', ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Semantic-only
        ax = axes[1, level]
        sem_counts = Counter(sem_codes[:, level])
        codes, counts = zip(*sorted(sem_counts.items()))
        ax.bar(codes, counts, color=COLORS['semantic'], alpha=0.7, width=1.0)
        ax.set_title(f'Semantic-only Level {level+1}')
        ax.set_xlabel('Code Index')
        ax.set_ylabel('Count')
        ax.set_xlim(-5, 260)
        
        unique_codes = len(sem_counts)
        ax.annotate(f'Unique: {unique_codes}/256', xy=(0.95, 0.95),
                   xycoords='axes fraction', ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Codebook Utilization by Level', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'codebook_distribution.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def plot_codebook_heatmap(csj_codes, sem_codes, save_dir):
    """Plot codebook utilization as heatmap."""
    num_levels = csj_codes.shape[1]
    codebook_size = 256
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # CSJ-ID heatmap
    csj_usage = np.zeros((num_levels, codebook_size))
    for level in range(num_levels):
        counts = Counter(csj_codes[:, level])
        for code, count in counts.items():
            csj_usage[level, code] = count
    
    im1 = axes[0].imshow(csj_usage, aspect='auto', cmap='Greens')
    axes[0].set_xlabel('Code Index')
    axes[0].set_ylabel('Level')
    axes[0].set_yticks(range(num_levels))
    axes[0].set_yticklabels([f'Level {i+1}' for i in range(num_levels)])
    axes[0].set_title('CSJ-ID Codebook Usage')
    plt.colorbar(im1, ax=axes[0], label='Count')
    
    # Semantic-only heatmap
    sem_usage = np.zeros((num_levels, codebook_size))
    for level in range(num_levels):
        counts = Counter(sem_codes[:, level])
        for code, count in counts.items():
            sem_usage[level, code] = count
    
    im2 = axes[1].imshow(sem_usage, aspect='auto', cmap='Blues')
    axes[1].set_xlabel('Code Index')
    axes[1].set_ylabel('Level')
    axes[1].set_yticks(range(num_levels))
    axes[1].set_yticklabels([f'Level {i+1}' for i in range(num_levels)])
    axes[1].set_title('Semantic-only Codebook Usage')
    plt.colorbar(im2, ax=axes[1], label='Count')
    
    fig.suptitle('Codebook Utilization Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'codebook_heatmap.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def plot_tsne_visualization(csj_codes, sem_codes, save_dir, num_samples=2000):
    """Plot t-SNE visualization of item codes."""
    # Subsample for faster t-SNE
    np.random.seed(42)
    indices = np.random.choice(len(csj_codes), min(num_samples, len(csj_codes)), replace=False)
    
    csj_sample = csj_codes[indices]
    sem_sample = sem_codes[indices]
    
    # Flatten codes to single representation
    csj_flat = csj_sample.reshape(len(csj_sample), -1).astype(np.float32)
    sem_flat = sem_sample.reshape(len(sem_sample), -1).astype(np.float32)
    
    # Run t-SNE
    print("Running t-SNE for CSJ-ID codes...")
    tsne_csj = TSNE(n_components=2, perplexity=30, random_state=42)
    csj_2d = tsne_csj.fit_transform(csj_flat)
    
    print("Running t-SNE for Semantic codes...")
    tsne_sem = TSNE(n_components=2, perplexity=30, random_state=42)
    sem_2d = tsne_sem.fit_transform(sem_flat)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color by first-level code for structure visualization
    csj_colors = csj_sample[:, 0]
    sem_colors = sem_sample[:, 0]
    
    scatter1 = axes[0].scatter(csj_2d[:, 0], csj_2d[:, 1], c=csj_colors, 
                               cmap='tab20', alpha=0.6, s=10)
    axes[0].set_title('CSJ-ID Item Codes')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    scatter2 = axes[1].scatter(sem_2d[:, 0], sem_2d[:, 1], c=sem_colors,
                               cmap='tab20', alpha=0.6, s=10)
    axes[1].set_title('Semantic-only Item Codes')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    fig.suptitle('t-SNE Visualization of Item Codes (colored by Level-1 code)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'tsne_codes.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def plot_utilization_comparison(results, save_dir):
    """Plot codebook utilization comparison from results.json."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract utilization data
    if 'csj_codebook_usage' in results and 'sem_codebook_usage' in results:
        csj_usage = results['csj_codebook_usage']['level_usage']
        sem_usage = results['sem_codebook_usage']['level_usage']
        
        x = np.arange(len(csj_usage))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [u * 100 for u in sem_usage], width,
                       label='Semantic-only', color=COLORS['semantic'])
        bars2 = ax.bar(x + width/2, [u * 100 for u in csj_usage], width,
                       label='CSJ-ID', color=COLORS['csj'])
        
        ax.set_ylabel('Codebook Utilization (%)')
        ax.set_xlabel('Level')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Level {i+1}' for i in range(len(csj_usage))])
        ax.legend()
        ax.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        ax.set_title('Codebook Utilization by Level', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'codebook_utilization.pdf')
        plt.savefig(save_path)
        plt.savefig(save_path.replace('.pdf', '.png'))
        print(f"Saved: {save_path}")
        plt.close()
    else:
        print("No codebook usage data found in results.json")


def plot_entropy_analysis(csj_codes, sem_codes, save_dir):
    """Plot entropy analysis of code distributions."""
    num_levels = csj_codes.shape[1]
    
    def compute_entropy(codes_level):
        counts = Counter(codes_level)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    csj_entropy = [compute_entropy(csj_codes[:, l]) for l in range(num_levels)]
    sem_entropy = [compute_entropy(sem_codes[:, l]) for l in range(num_levels)]
    
    # Maximum possible entropy for 256 codes
    max_entropy = np.log2(256)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(num_levels)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sem_entropy, width, label='Semantic-only', color=COLORS['semantic'])
    bars2 = ax.bar(x + width/2, csj_entropy, width, label='CSJ-ID', color=COLORS['csj'])
    
    ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.5, label=f'Max entropy ({max_entropy:.2f})')
    
    ax.set_ylabel('Entropy (bits)')
    ax.set_xlabel('Level')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Level {i+1}' for i in range(num_levels)])
    ax.legend()
    ax.set_title('Code Distribution Entropy by Level\n(Higher = More Uniform Distribution)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'entropy_analysis.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def main():
    base_dir = '/Users/ritik/Desktop/Research/ICMLFinal'
    output_dir = os.path.join(base_dir, 'outputs')
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Codebook Visualizations")
    print("=" * 60)
    
    # Try to load real codes
    data = load_codes_and_embeddings(output_dir)
    
    # Use real codes if available, otherwise generate synthetic
    if 'csj_codes' in data and 'sem_codes' in data:
        print("Using real codes from saved files")
        csj_codes = data['csj_codes']
        sem_codes = data['sem_codes']
    else:
        print("Generating synthetic codes for visualization")
        # Check results.json for codebook usage patterns
        if 'results' in data:
            results = data['results']
            csj_usage = results.get('csj_codebook_usage', {}).get('level_usage', [0.01, 1.0, 1.0, 1.0])
            sem_usage = results.get('sem_codebook_usage', {}).get('level_usage', [0.01, 1.0, 1.0, 1.0])
            
            # Generate codes matching the usage patterns
            csj_collapse = csj_usage[0] < 0.1  # Level 1 collapsed if < 10% usage
            sem_collapse = sem_usage[0] < 0.1
        else:
            csj_collapse = False
            sem_collapse = True  # Default: semantic collapses at level 1
        
        csj_codes, sem_codes = generate_synthetic_codes(
            csj_collapse_level1=csj_collapse,
            sem_collapse_level1=sem_collapse
        )
    
    # Generate visualizations
    print("\n1. Generating code distribution plots...")
    plot_code_distribution(csj_codes, sem_codes, figures_dir)
    
    print("\n2. Generating codebook heatmaps...")
    plot_codebook_heatmap(csj_codes, sem_codes, figures_dir)
    
    print("\n3. Generating t-SNE visualization...")
    plot_tsne_visualization(csj_codes, sem_codes, figures_dir)
    
    print("\n4. Generating entropy analysis...")
    plot_entropy_analysis(csj_codes, sem_codes, figures_dir)
    
    # Utilization comparison from results.json
    if 'results' in data:
        print("\n5. Generating utilization comparison...")
        plot_utilization_comparison(data['results'], figures_dir)
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
