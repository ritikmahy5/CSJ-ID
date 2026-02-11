"""
Data loading and preprocessing for CSJ-ID experiments.
"""

import os
import json
import gzip
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from config import DataConfig


@dataclass
class ProcessedData:
    """Container for processed dataset."""
    # Mappings
    user2idx: Dict[str, int] = None
    idx2user: Dict[int, str] = None
    item2idx: Dict[str, int] = None
    idx2item: Dict[int, str] = None
    
    # Item metadata
    item_texts: Dict[int, str] = None  # item_idx -> text description
    
    # Interactions
    train_sequences: Dict[int, List[int]] = None  # user_idx -> [item_idx, ...]
    val_sequences: Dict[int, List[int]] = None
    test_sequences: Dict[int, List[int]] = None
    
    # For LightGCN
    train_interactions: List[Tuple[int, int]] = None  # [(user_idx, item_idx), ...]
    
    # Statistics
    num_users: int = 0
    num_items: int = 0
    num_interactions: int = 0
    
    # Cold-start splits
    cold_users: List[int] = None  # Users with few interactions
    warm_users: List[int] = None  # Users with many interactions


def load_amazon_data(
    data_path: str,
    config: DataConfig,
    logger = None
) -> ProcessedData:
    """
    Load and preprocess Amazon review data.
    
    Args:
        data_path: Path to gzipped JSON file
        config: Data configuration
        logger: Optional logger for progress
    
    Returns:
        ProcessedData object with all necessary data
    """
    log = logger.info if logger else print
    
    log(f"Loading data from {data_path}")
    
    # Load raw reviews
    reviews = []
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading reviews", leave=False):
            reviews.append(json.loads(line))
    
    log(f"Loaded {len(reviews)} raw reviews")
    
    # Build user-item interactions with timestamps
    user_items = defaultdict(list)  # user -> [(item, timestamp), ...]
    item_texts = defaultdict(list)  # item -> [text descriptions]
    
    for review in tqdm(reviews, desc="Processing reviews", leave=False):
        user_id = review['reviewerID']
        item_id = review['asin']
        timestamp = review.get('unixReviewTime', 0)
        
        user_items[user_id].append((item_id, timestamp))
        
        # Collect item text
        text_parts = []
        if 'summary' in review and review['summary']:
            text_parts.append(review['summary'])
        if 'reviewText' in review and review['reviewText']:
            text_parts.append(review['reviewText'][:200])  # Truncate long reviews
        if text_parts:
            item_texts[item_id].append(' '.join(text_parts))
    
    # Filter users and items by minimum interactions
    log(f"Filtering: min_user={config.min_user_interactions}, min_item={config.min_item_interactions}")
    
    # Count item interactions
    item_counts = defaultdict(int)
    for user, items in user_items.items():
        for item, _ in items:
            item_counts[item] += 1
    
    # Filter items
    valid_items = {item for item, count in item_counts.items() 
                   if count >= config.min_item_interactions}
    
    # Filter user interactions to valid items
    filtered_user_items = {}
    for user, items in user_items.items():
        filtered = [(item, ts) for item, ts in items if item in valid_items]
        if len(filtered) >= config.min_user_interactions:
            filtered_user_items[user] = filtered
    
    log(f"After filtering: {len(filtered_user_items)} users, {len(valid_items)} items")
    
    # Sample users if max_users is set
    if config.max_users > 0 and len(filtered_user_items) > config.max_users:
        log(f"Sampling {config.max_users} users from {len(filtered_user_items)}")
        sampled_users = np.random.choice(
            list(filtered_user_items.keys()), 
            size=config.max_users, 
            replace=False
        )
        filtered_user_items = {u: filtered_user_items[u] for u in sampled_users}
        
        # Re-filter items to only those interacted with by sampled users
        valid_items = set()
        for user, items in filtered_user_items.items():
            for item, _ in items:
                valid_items.add(item)
        log(f"After user sampling: {len(filtered_user_items)} users, {len(valid_items)} items")
    
    # Create mappings
    user2idx = {user: idx for idx, user in enumerate(sorted(filtered_user_items.keys()))}
    idx2user = {idx: user for user, idx in user2idx.items()}
    item2idx = {item: idx for idx, item in enumerate(sorted(valid_items))}
    idx2item = {idx: item for item, idx in item2idx.items()}
    
    # Create item text descriptions (use most common/longest)
    final_item_texts = {}
    for item_id, idx in item2idx.items():
        texts = item_texts.get(item_id, [])
        if texts:
            # Use the longest text description
            final_item_texts[idx] = max(texts, key=len)
        else:
            final_item_texts[idx] = f"Item {item_id}"
    
    # Sort interactions by timestamp and create sequences
    train_sequences = {}
    val_sequences = {}
    test_sequences = {}
    train_interactions = []
    
    for user_id, items in tqdm(filtered_user_items.items(), desc="Creating sequences", leave=False):
        user_idx = user2idx[user_id]
        
        # Sort by timestamp
        sorted_items = sorted(items, key=lambda x: x[1])
        item_indices = [item2idx[item] for item, _ in sorted_items]
        
        # Split: last item for test, second-to-last for val, rest for train
        if len(item_indices) >= 3:
            train_sequences[user_idx] = item_indices[:-2]
            val_sequences[user_idx] = item_indices[-2:-1]
            test_sequences[user_idx] = item_indices[-1:]
        elif len(item_indices) == 2:
            train_sequences[user_idx] = item_indices[:-1]
            val_sequences[user_idx] = []
            test_sequences[user_idx] = item_indices[-1:]
        else:
            train_sequences[user_idx] = item_indices
            val_sequences[user_idx] = []
            test_sequences[user_idx] = []
        
        # Train interactions for LightGCN
        for item_idx in train_sequences[user_idx]:
            train_interactions.append((user_idx, item_idx))
    
    # Identify cold-start users (fewer than median interactions)
    user_interaction_counts = [len(seq) for seq in train_sequences.values()]
    median_interactions = np.median(user_interaction_counts)
    
    cold_users = [u for u, seq in train_sequences.items() if len(seq) <= median_interactions]
    warm_users = [u for u, seq in train_sequences.items() if len(seq) > median_interactions]
    
    # Create ProcessedData object
    data = ProcessedData(
        user2idx=user2idx,
        idx2user=idx2user,
        item2idx=item2idx,
        idx2item=idx2item,
        item_texts=final_item_texts,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        train_interactions=train_interactions,
        num_users=len(user2idx),
        num_items=len(item2idx),
        num_interactions=len(train_interactions),
        cold_users=cold_users,
        warm_users=warm_users,
    )
    
    log(f"Final dataset: {data.num_users} users, {data.num_items} items, {data.num_interactions} interactions")
    log(f"Cold users: {len(cold_users)}, Warm users: {len(warm_users)}")
    
    return data


class SequenceDataset(Dataset):
    """Dataset for sequential recommendation training."""
    
    def __init__(
        self,
        sequences: Dict[int, List[int]],
        num_items: int,
        max_seq_len: int = 50,
        mode: str = 'train'
    ):
        self.sequences = sequences
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.mode = mode
        
        # Create samples
        self.samples = []
        for user_idx, seq in sequences.items():
            if len(seq) < 2:
                continue
            if mode == 'train':
                # Create multiple samples per user for training
                for i in range(1, len(seq)):
                    input_seq = seq[max(0, i-max_seq_len):i]
                    target = seq[i]
                    self.samples.append((user_idx, input_seq, target))
            else:
                # For eval, use full history to predict last item
                self.samples.append((user_idx, seq[:-1], seq[-1]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user_idx, input_seq, target = self.samples[idx]
        
        # Pad sequence
        seq_len = len(input_seq)
        if seq_len < self.max_seq_len:
            padding = [self.num_items] * (self.max_seq_len - seq_len)  # Use num_items as padding token
            input_seq = padding + input_seq
        else:
            input_seq = input_seq[-self.max_seq_len:]
        
        return {
            'user_idx': user_idx,
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target': target,
            'seq_len': seq_len,
        }


class ItemEmbeddingDataset(Dataset):
    """Dataset for training RQ-VAE on item embeddings."""
    
    def __init__(
        self,
        semantic_embeddings: torch.Tensor,
        cf_embeddings: torch.Tensor,
    ):
        assert len(semantic_embeddings) == len(cf_embeddings)
        self.semantic_embeddings = semantic_embeddings
        self.cf_embeddings = cf_embeddings
    
    def __len__(self):
        return len(self.semantic_embeddings)
    
    def __getitem__(self, idx):
        return {
            'item_idx': idx,
            'semantic': self.semantic_embeddings[idx],
            'cf': self.cf_embeddings[idx],
        }


class BPRDataset(Dataset):
    """Dataset for BPR training of LightGCN."""
    
    def __init__(
        self,
        interactions: List[Tuple[int, int]],
        num_users: int,
        num_items: int,
        num_negatives: int = 1,
    ):
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Build user->items dict for negative sampling
        self.user_items = defaultdict(set)
        for user, item in interactions:
            self.user_items[user].add(item)
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]
        
        # Sample negative item
        neg_item = np.random.randint(0, self.num_items)
        while neg_item in self.user_items[user]:
            neg_item = np.random.randint(0, self.num_items)
        
        return {
            'user': user,
            'pos_item': pos_item,
            'neg_item': neg_item,
        }


def create_adj_matrix(
    interactions: List[Tuple[int, int]],
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    """Create normalized adjacency matrix for LightGCN (sparse version)."""
    # Build sparse adjacency matrix
    rows = []
    cols = []
    
    for user, item in interactions:
        # User -> Item edge
        rows.append(user)
        cols.append(num_users + item)
        # Item -> User edge
        rows.append(num_users + item)
        cols.append(user)
    
    # Create sparse tensor
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    size = num_users + num_items
    adj = torch.sparse_coo_tensor(indices, values, size=(size, size))
    adj = adj.coalesce()
    
    # Compute degree for normalization (D^(-1/2))
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    # For sparse: multiply each edge value by d_inv_sqrt[row] * d_inv_sqrt[col]
    row_indices = adj.indices()[0]
    col_indices = adj.indices()[1]
    edge_weights = d_inv_sqrt[row_indices] * d_inv_sqrt[col_indices]
    
    norm_adj = torch.sparse_coo_tensor(
        adj.indices(), 
        edge_weights, 
        size=(size, size)
    ).coalesce()
    
    return norm_adj


def get_semantic_embeddings(
    item_texts: Dict[int, str],
    model_name: str = 'all-MiniLM-L6-v2',
    device: torch.device = torch.device('cpu'),
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Extract semantic embeddings using SentenceTransformer.
    
    Args:
        item_texts: Dictionary mapping item_idx to text description
        model_name: SentenceTransformer model name
        device: Device to use
        batch_size: Batch size for encoding
    
    Returns:
        Tensor of shape [num_items, embedding_dim]
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name, device=str(device))
    
    # Sort by index to ensure correct order
    num_items = max(item_texts.keys()) + 1
    texts = [item_texts.get(i, f"Item {i}") for i in range(num_items)]
    
    # Encode in batches
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=str(device),
    )
    
    return embeddings.cpu()


if __name__ == "__main__":
    # Test data loading
    from config import get_default_config
    
    config = get_default_config()
    data = load_amazon_data(config.data.data_path, config.data)
    
    print(f"\nDataset loaded:")
    print(f"  Users: {data.num_users}")
    print(f"  Items: {data.num_items}")
    print(f"  Interactions: {data.num_interactions}")
    
    # Test datasets
    train_dataset = SequenceDataset(data.train_sequences, data.num_items)
    print(f"\nTrain samples: {len(train_dataset)}")
    
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input seq shape: {sample['input_seq'].shape}")
