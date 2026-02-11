"""
Training functions for CSJ-ID experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np

from models import CSJ_RQVAE, SemanticOnlyRQVAE, CFOnlyRQVAE, LightGCN, GenRec
from data import ItemEmbeddingDataset, BPRDataset, create_adj_matrix
from utils import Logger, Timer, EarlyStopping


def train_lightgcn(
    model: LightGCN,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    logger: Optional[Logger] = None,
    early_stopping: Optional[EarlyStopping] = None,
) -> Dict[str, list]:
    """
    Train LightGCN using BPR loss.
    
    Returns:
        Dictionary with training history
    """
    log = logger.info if logger else print
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"LightGCN Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            users = batch['user'].to(device)
            pos_items = batch['pos_item'].to(device)
            neg_items = batch['neg_item'].to(device)
            
            optimizer.zero_grad()
            loss = model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            log(f"LightGCN Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        if early_stopping and early_stopping(avg_loss):
            log(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


def train_rqvae(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    model_type: str = 'csj',  # 'csj', 'semantic', 'cf'
    lambda_sem: float = 0.5,
    logger: Optional[Logger] = None,
    early_stopping: Optional[EarlyStopping] = None,
) -> Dict[str, list]:
    """
    Train RQ-VAE model.
    
    Args:
        model: RQ-VAE model (CSJ, Semantic-only, or CF-only)
        train_loader: DataLoader with item embeddings
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        model_type: Type of model ('csj', 'semantic', 'cf')
        lambda_sem: Lambda for CSJ model
        logger: Optional logger
        early_stopping: Optional early stopping handler
    
    Returns:
        Dictionary with training history
    """
    log = logger.info if logger else print
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    history = {
        'loss': [],
        'recon_loss': [],
        'commitment_loss': [],
    }
    if model_type == 'csj':
        history['loss_sem'] = []
        history['loss_cf'] = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {k: [] for k in history.keys()}
        
        pbar = tqdm(train_loader, desc=f"RQ-VAE Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            
            if model_type == 'csj':
                z_sem = batch['semantic'].to(device)
                z_cf = batch['cf'].to(device)
                outputs = model(z_sem, z_cf, lambda_sem)
                epoch_losses['loss_sem'].append(outputs['loss_sem'].item())
                epoch_losses['loss_cf'].append(outputs['loss_cf'].item())
            elif model_type == 'semantic':
                z = batch['semantic'].to(device)
                outputs = model(z)
            else:  # cf
                z = batch['cf'].to(device)
                outputs = model(z)
            
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses['loss'].append(loss.item())
            epoch_losses['recon_loss'].append(outputs['recon_loss'].item())
            epoch_losses['commitment_loss'].append(outputs['commitment_loss'].item())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # Record average losses
        for k, v in epoch_losses.items():
            if v:
                history[k].append(np.mean(v))
        
        if (epoch + 1) % 10 == 0:
            loss_str = f"Loss: {history['loss'][-1]:.4f}"
            if model_type == 'csj':
                loss_str += f" | Sem: {history['loss_sem'][-1]:.4f} | CF: {history['loss_cf'][-1]:.4f}"
            log(f"RQ-VAE Epoch {epoch+1}/{num_epochs} | {loss_str}")
        
        if early_stopping and early_stopping(history['loss'][-1]):
            log(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


def train_genrec(
    model: GenRec,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    logger: Optional[Logger] = None,
    early_stopping: Optional[EarlyStopping] = None,
) -> Dict[str, list]:
    """
    Train Generative Recommender.
    
    Args:
        model: GenRec model
        train_loader: Training DataLoader
        val_loader: Optional validation DataLoader
        num_epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        logger: Optional logger
        early_stopping: Optional early stopping
    
    Returns:
        Dictionary with training history
    """
    log = logger.info if logger else print
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"GenRec Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            input_codes = batch['input_codes'].to(device)
            target_codes = batch['target_codes'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_codes, target_codes)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    input_codes = batch['input_codes'].to(device)
                    target_codes = batch['target_codes'].to(device)
                    outputs = model(input_codes, target_codes)
                    val_losses.append(outputs['loss'].item())
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            if (epoch + 1) % 5 == 0:
                log(f"GenRec Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
            
            if early_stopping and early_stopping(avg_val_loss):
                log(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if (epoch + 1) % 5 == 0:
                log(f"GenRec Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")
    
    return history


class GenRecDataset(torch.utils.data.Dataset):
    """Dataset for training GenRec with item codes."""
    
    def __init__(
        self,
        sequences: Dict[int, list],  # user -> [item_idx, ...]
        item_codes: torch.Tensor,    # [num_items, num_levels]
        max_seq_len: int = 50,
        num_levels: int = 4,
    ):
        self.item_codes = item_codes
        self.max_seq_len = max_seq_len
        self.num_levels = num_levels
        self.num_codes = item_codes.max().item() + 1
        
        # Create training samples
        self.samples = []
        for user_idx, seq in sequences.items():
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                input_items = seq[max(0, i-max_seq_len):i]
                target_item = seq[i]
                self.samples.append((input_items, target_item))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_items, target_item = self.samples[idx]
        
        # Get codes for input items
        input_codes = self.item_codes[input_items]  # [seq_len, num_levels]
        
        # Pad if necessary
        seq_len = len(input_items)
        if seq_len < self.max_seq_len:
            padding = torch.full(
                (self.max_seq_len - seq_len, self.num_levels),
                self.num_codes,  # padding token
                dtype=torch.long
            )
            input_codes = torch.cat([padding, input_codes], dim=0)
        
        target_codes = self.item_codes[target_item]  # [num_levels]
        
        return {
            'input_codes': input_codes,
            'target_codes': target_codes,
            'target_item': target_item,
        }


if __name__ == "__main__":
    # Test training functions
    print("Training module loaded successfully!")
