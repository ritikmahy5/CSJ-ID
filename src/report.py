"""
Generate summary report for CSJ-ID experiments.
Outputs a formatted text report and LaTeX tables.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional


def generate_latex_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    caption: str = "Performance Comparison",
    label: str = "tab:main_results",
) -> str:
    """Generate LaTeX table code."""
    methods = list(results.keys())
    
    # Header
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
    latex += "\\toprule\n"
    latex += "Method & " + " & ".join(metrics) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Find best values
    best_values = {}
    for metric in metrics:
        values = [results[m].get(metric, 0) for m in methods]
        best_values[metric] = max(values)
    
    # Data rows
    for method in methods:
        row = [method]
        for metric in metrics:
            val = results[method].get(metric, 0)
            if val == best_values[metric] and val > 0:
                row.append(f"\\textbf{{{val:.4f}}}")
            else:
                row.append(f"{val:.4f}")
        latex += " & ".join(row) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def generate_markdown_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
) -> str:
    """Generate Markdown table."""
    methods = list(results.keys())
    
    # Header
    md = "| Method | " + " | ".join(metrics) + " |\n"
    md += "|" + "--------|" * (len(metrics) + 1) + "\n"
    
    # Data rows
    for method in methods:
        row = [method]
        for metric in metrics:
            val = results[method].get(metric, 0)
            row.append(f"{val:.4f}")
        md += "| " + " | ".join(row) + " |\n"
    
    return md


def generate_report(output_dir: str) -> str:
    """
    Generate a comprehensive text report.
    
    Args:
        output_dir: Directory containing results.json
    
    Returns:
        Report as string
    """
    results_path = os.path.join(output_dir, 'results.json')
    
    if not os.path.exists(results_path):
        return f"Error: {results_path} not found"
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    report = []
    report.append("=" * 70)
    report.append("CSJ-ID EXPERIMENT REPORT")
    report.append("ICML 2026 Submission")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    # Data statistics
    if 'data_stats' in results:
        report.append("\n## Dataset Statistics")
        report.append("-" * 40)
        stats = results['data_stats']
        report.append(f"Users:        {stats.get('num_users', 'N/A'):,}")
        report.append(f"Items:        {stats.get('num_items', 'N/A'):,}")
        report.append(f"Interactions: {stats.get('num_interactions', 'N/A'):,}")
        report.append(f"Cold Users:   {stats.get('num_cold_users', 'N/A'):,}")
        report.append(f"Warm Users:   {stats.get('num_warm_users', 'N/A'):,}")
    
    # Main results
    if 'csj_metrics' in results and 'sem_metrics' in results:
        report.append("\n## Main Results")
        report.append("-" * 40)
        
        metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'NDCG@10', 'MRR']
        metrics = [m for m in metrics if m in results['csj_metrics']]
        
        # Table header
        header = f"{'Metric':<12} | {'Semantic':<10} | {'CSJ-ID':<10} | {'Improv.':<10}"
        report.append(header)
        report.append("-" * len(header))
        
        for metric in metrics:
            sem_val = results['sem_metrics'].get(metric, 0)
            csj_val = results['csj_metrics'].get(metric, 0)
            if sem_val > 0:
                imp = (csj_val - sem_val) / sem_val * 100
                imp_str = f"{imp:+.1f}%"
            else:
                imp_str = "N/A"
            
            report.append(f"{metric:<12} | {sem_val:<10.4f} | {csj_val:<10.4f} | {imp_str:<10}")
        
        # Markdown table for paper
        report.append("\n### Markdown Table (for paper)")
        report.append(generate_markdown_table(
            {'Semantic-only': results['sem_metrics'], 'CSJ-ID (Ours)': results['csj_metrics']},
            metrics
        ))
    
    # Lambda ablation
    if 'lambda_ablation' in results:
        report.append("\n## Lambda Sensitivity Analysis")
        report.append("-" * 40)
        report.append(f"{'Lambda':<10} | {'Sem Loss':<12} | {'CF Loss':<12} | {'Total':<12}")
        report.append("-" * 50)
        
        for lambda_val, losses in sorted(results['lambda_ablation'].items(), key=lambda x: float(x[0])):
            report.append(
                f"{float(lambda_val):<10.1f} | "
                f"{losses['loss_sem']:<12.4f} | "
                f"{losses['loss_cf']:<12.4f} | "
                f"{losses['total_loss']:<12.4f}"
            )
    
    # Cold-start analysis
    if 'csj_cold_warm' in results:
        report.append("\n## Cold-Start Analysis")
        report.append("-" * 40)
        
        report.append("\nCSJ-ID:")
        if 'cold' in results['csj_cold_warm']:
            report.append(f"  Cold users R@10: {results['csj_cold_warm']['cold'].get('Recall@10', 0):.4f}")
        if 'warm' in results['csj_cold_warm']:
            report.append(f"  Warm users R@10: {results['csj_cold_warm']['warm'].get('Recall@10', 0):.4f}")
        
        if 'sem_cold_warm' in results:
            report.append("\nSemantic-only:")
            if 'cold' in results['sem_cold_warm']:
                report.append(f"  Cold users R@10: {results['sem_cold_warm']['cold'].get('Recall@10', 0):.4f}")
            if 'warm' in results['sem_cold_warm']:
                report.append(f"  Warm users R@10: {results['sem_cold_warm']['warm'].get('Recall@10', 0):.4f}")
    
    # Codebook usage
    if 'csj_codebook_usage' in results:
        report.append("\n## Codebook Utilization")
        report.append("-" * 40)
        report.append(f"CSJ-ID usage:      {results['csj_codebook_usage']['overall_usage']:.2%}")
        if 'sem_codebook_usage' in results:
            report.append(f"Semantic usage:    {results['sem_codebook_usage']['overall_usage']:.2%}")
    
    # LaTeX table
    if 'csj_metrics' in results and 'sem_metrics' in results:
        report.append("\n## LaTeX Table")
        report.append("-" * 40)
        metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20']
        metrics = [m for m in metrics if m in results['csj_metrics']]
        report.append(generate_latex_table(
            {'Semantic-only': results['sem_metrics'], 'CSJ-ID (Ours)': results['csj_metrics']},
            metrics,
            caption="Main Results on Amazon Beauty Dataset",
            label="tab:main_results"
        ))
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument('--output_dir', type=str,
                       default='/Users/ritik/Desktop/Research/ICMLFinal/outputs',
                       help='Directory containing results')
    parser.add_argument('--save', action='store_true',
                       help='Save report to file')
    args = parser.parse_args()
    
    report = generate_report(args.output_dir)
    print(report)
    
    if args.save:
        report_path = os.path.join(args.output_dir, 'experiment_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
