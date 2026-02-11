"""
Evaluation functions for CSJ-ID experiments.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from models import GenRec


def recall_at_k(pred_items: List[int], true_item: int, k: int) -> float:
    """Compute Recall@K for a single prediction."""
    return 1.0 if true_item in pred_items[:k] else 0.0


def ndcg_at_k(pred_items: List[int], true_item: int, k: int) -> float:
    """Compute NDCG@K for a single prediction."""
    if true_item not in pred_items[:k]:
        return 0.0
    rank = pred_items[:k].index(true_item) + 1
    return 1.0 / np.log2(rank + 1)


def mrr(pred_items: List[int], true_item: int) -> float:
    """Compute Mean Reciprocal Rank for a single prediction."""
    if true_item not in pred_items:
        return 0.0
    rank = pred_items.index(true_item) + 1
    return 1.0 / rank


def hit_rate(pred_items: List[int], true_item: int, k: int) -> float:
    """Compute Hit Rate@K for a single prediction."""
    return 1.0 if true_item in pred_items[:k] else 0.0


def evaluate_recommendations(
    predictions: List[List[int]],  # List of predicted item lists
    ground_truth: List[int],       # List of true items
    ks: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate recommendation predictions.
    
    Args:
        predictions: List of predicted item lists for each user
        ground_truth: List of true items for each user
        ks: List of K values for metrics
    
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    for k in ks:
        recalls = []
        ndcgs = []
        hits = []
        
        for pred, true in zip(predictions, ground_truth):
            recalls.append(recall_at_k(pred, true, k))
            ndcgs.append(ndcg_at_k(pred, true, k))
            hits.append(hit_rate(pred, true, k))
        
        metrics[f'Recall@{k}'] = np.mean(recalls)
        metrics[f'NDCG@{k}'] = np.mean(ndcgs)
        metrics[f'HR@{k}'] = np.mean(hits)
    
    # MRR (computed once)
    mrrs = [mrr(pred, true) for pred, true in zip(predictions, ground_truth)]
    metrics['MRR'] = np.mean(mrrs)
    
    return metrics


def evaluate_genrec(
    model: GenRec,
    test_data: Dict[int, Tuple[List[int], int]],  # user -> (history, target)
    item_codes: torch.Tensor,
    code_to_item: Dict[Tuple, int],
    device: torch.device,
    ks: List[int] = [1, 5, 10, 20],
    max_seq_len: int = 50,
    batch_size: int = 64,
    logger = None,
) -> Dict[str, float]:
    """
    Evaluate GenRec model on test data.
    
    Args:
        model: Trained GenRec model
        test_data: Dictionary of user -> (history items, target item)
        item_codes: Item codes tensor [num_items, num_levels]
        code_to_item: Mapping from code tuple to item index
        device: Device to run on
        ks: K values for metrics
        max_seq_len: Maximum sequence length
        batch_size: Batch size for evaluation
        logger: Optional logger
    
    Returns:
        Dictionary with metrics
    """
    log = logger.info if logger else print
    
    model.eval()
    model.to(device)
    num_levels = item_codes.shape[1]
    num_codes = model.num_codes
    
    predictions = []
    ground_truth = []
    
    users = list(test_data.keys())
    
    with torch.no_grad():
        for i in tqdm(range(0, len(users), batch_size), desc="Evaluating", leave=False):
            batch_users = users[i:i+batch_size]
            batch_histories = []
            batch_targets = []
            
            for user in batch_users:
                history, target = test_data[user]
                batch_histories.append(history)
                batch_targets.append(target)
            
            # Prepare input codes
            max_hist_len = min(max(len(h) for h in batch_histories), max_seq_len)
            
            input_codes_batch = []
            for history in batch_histories:
                if len(history) == 0:
                    # No history - use padding
                    codes = torch.full((max_hist_len, num_levels), num_codes, dtype=torch.long)
                else:
                    history_truncated = history[-max_hist_len:]
                    codes = item_codes[history_truncated]
                    # Pad if needed
                    if len(codes) < max_hist_len:
                        padding = torch.full(
                            (max_hist_len - len(codes), num_levels),
                            num_codes,
                            dtype=torch.long
                        )
                        codes = torch.cat([padding, codes], dim=0)
                input_codes_batch.append(codes)
            
            input_codes = torch.stack(input_codes_batch).to(device)
            
            # Generate predictions
            pred_codes = model.generate(input_codes)  # [B, num_levels]
            
            # Convert codes to items
            for j, pred_code in enumerate(pred_codes):
                code_tuple = tuple(pred_code.cpu().tolist())
                
                # Find matching item or nearest
                if code_tuple in code_to_item:
                    pred_item = code_to_item[code_tuple]
                    predictions.append([pred_item])
                else:
                    # Find items with similar codes (approximate matching)
                    pred_items = find_similar_items(pred_code.cpu(), item_codes, k=20)
                    predictions.append(pred_items)
                
                ground_truth.append(batch_targets[j])
    
    # Compute metrics
    metrics = evaluate_recommendations(predictions, ground_truth, ks)
    
    return metrics


def find_similar_items(
    pred_code: torch.Tensor,  # [num_levels]
    item_codes: torch.Tensor,  # [num_items, num_levels]
    k: int = 20,
) -> List[int]:
    """
    Find items with codes most similar to predicted codes.
    Uses Hamming distance (number of matching code levels).
    """
    # Compute match counts
    matches = (item_codes == pred_code.unsqueeze(0)).sum(dim=1)  # [num_items]
    
    # Get top-k items by match count
    _, top_indices = torch.topk(matches, k=min(k, len(matches)))
    
    return top_indices.tolist()


def build_code_to_item_mapping(item_codes: torch.Tensor) -> Dict[Tuple, int]:
    """Build mapping from code tuples to item indices."""
    code_to_item = {}
    for item_idx, codes in enumerate(item_codes):
        code_tuple = tuple(codes.tolist())
        if code_tuple not in code_to_item:
            code_to_item[code_tuple] = item_idx
    return code_to_item


def evaluate_cold_warm_split(
    model: GenRec,
    cold_users: List[int],
    warm_users: List[int],
    test_data: Dict[int, Tuple[List[int], int]],
    item_codes: torch.Tensor,
    code_to_item: Dict[Tuple, int],
    device: torch.device,
    ks: List[int] = [1, 5, 10, 20],
    logger = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate separately on cold-start and warm users.
    
    Returns:
        Dictionary with 'cold' and 'warm' metrics
    """
    log = logger.info if logger else print
    
    # Filter test data
    cold_test = {u: test_data[u] for u in cold_users if u in test_data}
    warm_test = {u: test_data[u] for u in warm_users if u in test_data}
    
    log(f"Evaluating on {len(cold_test)} cold users and {len(warm_test)} warm users")
    
    results = {}
    
    if cold_test:
        log("Evaluating cold users...")
        results['cold'] = evaluate_genrec(
            model, cold_test, item_codes, code_to_item, device, ks, logger=logger
        )
    
    if warm_test:
        log("Evaluating warm users...")
        results['warm'] = evaluate_genrec(
            model, warm_test, item_codes, code_to_item, device, ks, logger=logger
        )
    
    return results


def run_statistical_tests(
    results_a: List[float],
    results_b: List[float],
    test_type: str = 'paired_t',
) -> Dict[str, float]:
    """
    Run statistical significance tests.
    
    Args:
        results_a: Per-sample results for method A
        results_b: Per-sample results for method B
        test_type: Type of test ('paired_t', 'wilcoxon')
    
    Returns:
        Dictionary with test statistics and p-value
    """
    from scipy import stats
    
    if test_type == 'paired_t':
        stat, pvalue = stats.ttest_rel(results_a, results_b)
        return {'t_statistic': stat, 'p_value': pvalue}
    elif test_type == 'wilcoxon':
        stat, pvalue = stats.wilcoxon(results_a, results_b)
        return {'w_statistic': stat, 'p_value': pvalue}
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def bootstrap_confidence_interval(
    data: List[float],
    statistic=np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return statistic(data), lower, upper


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation functions...")
    
    # Mock predictions
    predictions = [
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 3, 5, 7, 9],
    ]
    ground_truth = [1, 3, 10]
    
    metrics = evaluate_recommendations(predictions, ground_truth, ks=[1, 3, 5])
    print("Metrics:", metrics)
    
    # Test bootstrap
    data = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    mean, lower, upper = bootstrap_confidence_interval(data)
    print(f"Bootstrap: {mean:.3f} ({lower:.3f}, {upper:.3f})")
    
    print("Evaluation tests passed!")
