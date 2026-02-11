"""
Model implementations for CSJ-ID experiments.
Includes: RQ-VAE variants, LightGCN, and Generative Recommender.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math


# =============================================================================
# Vector Quantization Components
# =============================================================================

class VectorQuantizer(nn.Module):
    """Single-level vector quantization with EMA updates."""
    
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        
        self.register_buffer('codebook', torch.randn(num_codes, code_dim))
        self.register_buffer('cluster_size', torch.zeros(num_codes))
        self.register_buffer('ema_w', torch.randn(num_codes, code_dim))
        self._initialized = False
        
    def _init_codebook(self, z: torch.Tensor):
        """Initialize codebook from data."""
        if not self._initialized and self.training:
            batch_size = z.size(0)
            if batch_size >= self.num_codes:
                # Sample without replacement
                indices = torch.randperm(batch_size, device=z.device)[:self.num_codes]
                self.codebook.data = z[indices].clone()
            else:
                # Sample with replacement when batch is smaller than codebook
                indices = torch.randint(0, batch_size, (self.num_codes,), device=z.device)
                self.codebook.data = z[indices].clone()
            self.ema_w.data = self.codebook.data.clone()
            self._initialized = True
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._init_codebook(z)
        
        distances = torch.cdist(z, self.codebook)
        indices = distances.argmin(dim=-1)
        quantized = self.codebook[indices]
        
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_codes).float()
                self.cluster_size.data = (
                    self.decay * self.cluster_size + 
                    (1 - self.decay) * encodings.sum(0)
                )
                dw = encodings.T @ z
                self.ema_w.data = self.decay * self.ema_w + (1 - self.decay) * dw
                
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps) / 
                    (n + self.num_codes * self.eps) * n
                )
                self.codebook.data = self.ema_w / cluster_size.unsqueeze(1)
        
        commitment_loss = F.mse_loss(z, quantized.detach())
        quantized = z + (quantized - z).detach()  # Straight-through estimator
        
        return quantized, indices, commitment_loss


class ResidualQuantizer(nn.Module):
    """Multi-level residual quantization."""
    
    def __init__(
        self,
        num_levels: int,
        num_codes: int,
        code_dim: int,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_codes, code_dim, commitment_weight)
            for _ in range(num_levels)
        ])
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        codes = []
        quantized = torch.zeros_like(z)
        residual = z
        total_loss = 0
        
        for quantizer in self.quantizers:
            q, idx, loss = quantizer(residual)
            codes.append(idx)
            quantized = quantized + q
            residual = residual - q.detach()
            total_loss = total_loss + loss
        
        return quantized, codes, total_loss / self.num_levels


# =============================================================================
# RQ-VAE Models
# =============================================================================

class CSJ_RQVAE(nn.Module):
    """
    Collaborative-Semantic Joint RQ-VAE.
    
    Key innovation: Multi-objective loss optimizes for BOTH
    semantic and CF signal reconstruction simultaneously.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_levels: int = 4,
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Residual quantizer
        self.quantizer = ResidualQuantizer(
            num_levels=num_levels,
            num_codes=codebook_size,
            code_dim=hidden_dim,
            commitment_weight=commitment_weight,
        )
        
        # Separate decoders for each modality
        self.decoder_sem = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self.decoder_cf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, z_sem: torch.Tensor, z_cf: torch.Tensor, 
               lambda_sem: float = 0.5) -> torch.Tensor:
        """Encode and mix both modalities."""
        h_sem = self.encoder(z_sem)
        h_cf = self.encoder(z_cf)
        h_joint = lambda_sem * h_sem + (1 - lambda_sem) * h_cf
        return h_joint
    
    def forward(
        self, 
        z_sem: torch.Tensor, 
        z_cf: torch.Tensor,
        lambda_sem: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        h_joint = self.encode(z_sem, z_cf, lambda_sem)
        quantized, codes, commitment_loss = self.quantizer(h_joint)
        
        z_sem_recon = self.decoder_sem(quantized)
        z_cf_recon = self.decoder_cf(quantized)
        
        loss_sem = F.mse_loss(z_sem_recon, z_sem)
        loss_cf = F.mse_loss(z_cf_recon, z_cf)
        
        recon_loss = lambda_sem * loss_sem + (1 - lambda_sem) * loss_cf
        total_loss = recon_loss + commitment_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'loss_sem': loss_sem,
            'loss_cf': loss_cf,
            'commitment_loss': commitment_loss,
            'codes': codes,
            'z_sem_recon': z_sem_recon,
            'z_cf_recon': z_cf_recon,
        }
    
    @torch.no_grad()
    def get_codes(
        self, 
        z_sem: torch.Tensor, 
        z_cf: torch.Tensor,
        lambda_sem: float = 0.5,
    ) -> torch.Tensor:
        """Get CSJ-ID codes for items. Returns [B, num_levels]."""
        self.eval()
        h_joint = self.encode(z_sem, z_cf, lambda_sem)
        _, codes, _ = self.quantizer(h_joint)
        return torch.stack(codes, dim=1)


class SemanticOnlyRQVAE(nn.Module):
    """Baseline: Standard semantic-only RQ-VAE (TIGER-style)."""
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_levels: int = 4,
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.quantizer = ResidualQuantizer(
            num_levels=num_levels,
            num_codes=codebook_size,
            code_dim=hidden_dim,
            commitment_weight=commitment_weight,
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(z)
        quantized, codes, commitment_loss = self.quantizer(h)
        z_recon = self.decoder(quantized)
        
        recon_loss = F.mse_loss(z_recon, z)
        total_loss = recon_loss + commitment_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'commitment_loss': commitment_loss,
            'codes': codes,
            'z_recon': z_recon,
        }
    
    @torch.no_grad()
    def get_codes(self, z: torch.Tensor) -> torch.Tensor:
        self.eval()
        h = self.encoder(z)
        _, codes, _ = self.quantizer(h)
        return torch.stack(codes, dim=1)


class CFOnlyRQVAE(SemanticOnlyRQVAE):
    """Baseline: CF-only RQ-VAE."""
    pass


# =============================================================================
# LightGCN for CF Embeddings
# =============================================================================

class LightGCN(nn.Module):
    """
    LightGCN for collaborative filtering embeddings.
    
    Simplified GCN that only uses neighborhood aggregation without
    feature transformation or non-linear activation.
    
    Note: Sparse operations run on CPU, embeddings transferred to target device.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Adjacency matrix stored on CPU for sparse ops
        self.register_buffer('adj_matrix', None)
        self._adj_on_cpu = None
    
    def set_adj_matrix(self, adj_matrix: torch.Tensor):
        """Set the normalized adjacency matrix (keeps on CPU for sparse ops)."""
        # Store on CPU for sparse matrix multiplication
        self._adj_on_cpu = adj_matrix.cpu()
        self.adj_matrix = adj_matrix
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute final user and item embeddings.
        
        Returns:
            user_embeddings: [num_users, embedding_dim]
            item_embeddings: [num_items, embedding_dim]
        """
        device = self.user_embedding.weight.device
        
        # Concatenate user and item embeddings
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        embeddings_list = [all_embeddings]
        
        # Graph convolution layers - use CPU for sparse ops
        if self._adj_on_cpu is not None and self._adj_on_cpu.is_sparse:
            # Move to CPU for sparse matmul
            all_embeddings_cpu = all_embeddings.cpu()
            for layer in range(self.num_layers):
                all_embeddings_cpu = torch.sparse.mm(self._adj_on_cpu, all_embeddings_cpu)
                if self.training:
                    all_embeddings_cpu = F.dropout(all_embeddings_cpu, p=self.dropout)
                embeddings_list.append(all_embeddings_cpu.to(device))
        else:
            # Dense path (original)
            for layer in range(self.num_layers):
                all_embeddings = self.adj_matrix @ all_embeddings
                if self.training:
                    all_embeddings = F.dropout(all_embeddings, p=self.dropout)
                embeddings_list.append(all_embeddings)
        
        # Mean pooling across layers
        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        
        user_embeddings = final_embeddings[:self.num_users]
        item_embeddings = final_embeddings[self.num_users:]
        
        return user_embeddings, item_embeddings
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get item embeddings only."""
        _, item_emb = self.forward()
        return item_emb
    
    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BPR loss for training."""
        user_emb, item_emb = self.forward()
        
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        pos_scores = (user_e * pos_e).sum(dim=1)
        neg_scores = (user_e * neg_e).sum(dim=1)
        
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization
        reg_loss = (
            self.user_embedding.weight[users].norm(2).pow(2) +
            self.item_embedding.weight[pos_items].norm(2).pow(2) +
            self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / users.size(0)
        
        return loss + 1e-5 * reg_loss


# =============================================================================
# Generative Recommender (Transformer)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GenRec(nn.Module):
    """
    Generative Recommender using Transformer (GPT-style decoder-only).
    
    Efficient single-pass training with causal masking.
    """
    
    def __init__(
        self,
        num_codes: int,
        num_levels: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()
        
        self.num_codes = num_codes
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = num_codes + 2  # +1 for padding, +1 for BOS
        self.pad_token = num_codes
        self.bos_token = num_codes + 1
        
        # Embeddings
        self.code_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=self.pad_token)
        self.level_embedding = nn.Embedding(num_levels, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len * num_levels + num_levels + 10, hidden_dim)
        
        # GPT-style Transformer (using encoder with causal mask = decoder-only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.output_head = nn.Linear(hidden_dim, num_codes)
        
        # Layer norm
        self.ln_f = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_codes: torch.Tensor,  # [B, seq_len, num_levels]
        target_codes: torch.Tensor,  # [B, num_levels] - next item's codes
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient single-pass forward for training.
        
        Predicts all L target levels in one transformer pass.
        """
        B, S, L = input_codes.shape
        device = input_codes.device
        
        # Flatten input: [B, S, L] -> [B, S*L]
        flat_input = input_codes.view(B, S * L)
        
        # Build full sequence: [BOS, history..., target_0, target_1, ..., target_{L-2}]
        # We predict: [target_0, target_1, ..., target_{L-1}]
        bos = torch.full((B, 1), self.bos_token, device=device, dtype=torch.long)
        
        # Teacher forcing: include target codes (except last) for parallel training
        target_input = target_codes[:, :-1]  # [B, L-1]
        
        # Full input sequence
        full_seq = torch.cat([bos, flat_input, target_input], dim=1)  # [B, 1 + S*L + L-1]
        seq_len = full_seq.size(1)
        
        # Embeddings
        code_emb = self.code_embedding(full_seq)  # [B, seq_len, hidden]
        
        # Level embeddings (cycling)
        level_ids = torch.arange(seq_len, device=device) % L
        level_emb = self.level_embedding(level_ids)  # [seq_len, hidden]
        
        # Position embeddings
        pos_ids = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embedding(pos_ids)  # [seq_len, hidden]
        
        # Combine embeddings
        x = code_emb + level_emb.unsqueeze(0) + pos_emb.unsqueeze(0)
        
        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Transformer forward
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        
        # Get logits for prediction positions
        # We predict at positions: [S*L, S*L+1, ..., S*L+L-1] (after history, predict each target level)
        pred_start = 1 + S * L - 1  # Position before first target
        pred_positions = x[:, pred_start:pred_start + L]  # [B, L, hidden]
        
        logits = self.output_head(pred_positions)  # [B, L, num_codes]
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, self.num_codes),
            target_codes.view(-1),
            ignore_index=self.pad_token,
        )
        
        return {
            'loss': loss,
            'logits': logits,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_codes: torch.Tensor,  # [B, seq_len, num_levels]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate next item's codes autoregressively.
        """
        self.eval()
        B, S, L = input_codes.shape
        device = input_codes.device
        
        # Flatten input
        flat_input = input_codes.view(B, S * L)
        bos = torch.full((B, 1), self.bos_token, device=device, dtype=torch.long)
        context = torch.cat([bos, flat_input], dim=1)  # [B, 1 + S*L]
        
        predicted = []
        
        for level in range(L):
            seq_len = context.size(1)
            
            # Embeddings
            code_emb = self.code_embedding(context)
            level_ids = torch.arange(seq_len, device=device) % L
            level_emb = self.level_embedding(level_ids)
            pos_ids = torch.arange(seq_len, device=device)
            pos_emb = self.pos_embedding(pos_ids)
            
            x = code_emb + level_emb.unsqueeze(0) + pos_emb.unsqueeze(0)
            
            # Causal mask
            causal_mask = self._generate_causal_mask(seq_len, device)
            
            # Forward
            x = self.transformer(x, mask=causal_mask, is_causal=True)
            x = self.ln_f(x)
            
            # Predict next
            logits = self.output_head(x[:, -1]) / temperature  # [B, num_codes]
            next_code = torch.argmax(logits, dim=-1)  # Greedy decoding
            
            predicted.append(next_code)
            context = torch.cat([context, next_code.unsqueeze(1)], dim=1)
        
        return torch.stack(predicted, dim=1)  # [B, L]


# =============================================================================
# Baseline Models
# =============================================================================

class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).
    
    Transformer-based sequential recommender that predicts next item
    directly from item IDs (no quantization).
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()
        
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.pad_token = num_items  # Use num_items as padding
        
        # Item embedding (+1 for padding)
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=self.pad_token)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_items)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor = None):
        """
        Args:
            input_ids: [B, seq_len] item indices
            target_ids: [B] target item indices (for training)
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        item_emb = self.item_embedding(input_ids)
        pos_ids = torch.arange(S, device=device)
        pos_emb = self.pos_embedding(pos_ids)
        x = self.dropout(item_emb + pos_emb)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        # Padding mask
        padding_mask = (input_ids == self.pad_token)
        
        # Transformer
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)
        x = self.ln_f(x)
        
        # Get last position output
        logits = self.output_layer(x[:, -1])  # [B, num_items]
        
        if target_ids is not None:
            loss = F.cross_entropy(logits, target_ids)
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}
    
    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Predict top-k items."""
        self.eval()
        outputs = self.forward(input_ids)
        logits = outputs['logits']
        _, top_k = torch.topk(logits, k, dim=-1)
        return top_k


class BPRMF(nn.Module):
    """
    Bayesian Personalized Ranking with Matrix Factorization.
    
    Classic collaborative filtering baseline.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        """
        BPR loss computation.
        """
        user_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        
        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_emb).sum(dim=-1)
        
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization
        reg_loss = (
            user_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / users.size(0)
        
        return {'loss': loss + 1e-5 * reg_loss}
    
    @torch.no_grad()
    def predict(self, user_id: int, k: int = 10) -> torch.Tensor:
        """Predict top-k items for a user."""
        self.eval()
        user_emb = self.user_embedding.weight[user_id]  # [dim]
        scores = self.item_embedding.weight @ user_emb  # [num_items]
        _, top_k = torch.topk(scores, k)
        return top_k
    
    @torch.no_grad()
    def get_all_scores(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get scores for all items for given users."""
        self.eval()
        user_emb = self.user_embedding(user_ids)  # [B, dim]
        scores = user_emb @ self.item_embedding.weight.T  # [B, num_items]
        return scores


class GRU4Rec(nn.Module):
    """
    GRU-based Sequential Recommendation (Hidasi et al., 2016).
    
    RNN-based sequential recommender baseline.
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()
        
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.pad_token = num_items
        
        # Item embedding (+1 for padding)
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=self.pad_token)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_items)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor = None):
        """
        Args:
            input_ids: [B, seq_len] item indices
            target_ids: [B] target item indices (for training)
        """
        # Embeddings
        x = self.item_embedding(input_ids)
        x = self.dropout(x)
        
        # GRU forward
        output, _ = self.gru(x)
        
        # Get last output
        logits = self.output_layer(output[:, -1])  # [B, num_items]
        
        if target_ids is not None:
            loss = F.cross_entropy(logits, target_ids)
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}
    
    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Predict top-k items."""
        self.eval()
        outputs = self.forward(input_ids)
        logits = outputs['logits']
        _, top_k = torch.topk(logits, k, dim=-1)
        return top_k


class PopularityBaseline:
    """Recommend most popular items."""
    
    def __init__(self, item_counts: Dict[int, int]):
        sorted_items = sorted(item_counts.items(), key=lambda x: -x[1])
        self.popular_items = [item for item, _ in sorted_items]
    
    def recommend(self, k: int = 10) -> List[int]:
        return self.popular_items[:k]


class RandomBaseline:
    """Recommend random items."""
    
    def __init__(self, num_items: int, seed: int = 42):
        self.num_items = num_items
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
    
    def recommend(self, k: int = 10) -> torch.Tensor:
        return torch.randint(0, self.num_items, (k,), generator=self.rng)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_codebook_usage(codes: torch.Tensor, codebook_size: int) -> Dict[str, float]:
    """Compute codebook utilization metrics."""
    unique_codes = set()
    for level_codes in codes.T:
        unique_codes.update(level_codes.unique().tolist())
    
    usage = len(unique_codes) / (codebook_size * codes.shape[1])
    
    level_usage = []
    for level_codes in codes.T:
        level_usage.append(level_codes.unique().numel() / codebook_size)
    
    return {
        'overall_usage': usage,
        'level_usage': level_usage,
        'unique_codes': len(unique_codes),
    }


if __name__ == '__main__':
    # Quick test
    torch.manual_seed(42)
    device = torch.device('cpu')
    
    B, D = 32, 384
    z_sem = torch.randn(B, D)
    z_cf = torch.randn(B, D)
    
    print("Testing CSJ-RQVAE...")
    model = CSJ_RQVAE(input_dim=D)
    outputs = model(z_sem, z_cf)
    print(f"  Total loss: {outputs['loss'].item():.4f}")
    print(f"  Semantic loss: {outputs['loss_sem'].item():.4f}")
    print(f"  CF loss: {outputs['loss_cf'].item():.4f}")
    
    codes = model.get_codes(z_sem, z_cf)
    print(f"  Codes shape: {codes.shape}")
    
    print("\nTesting GenRec...")
    genrec = GenRec(num_codes=256, num_levels=4)
    input_codes = torch.randint(0, 256, (B, 10, 4))  # 10 items in history
    target_codes = torch.randint(0, 256, (B, 4))
    
    outputs = genrec(input_codes, target_codes)
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    pred = genrec.generate(input_codes)
    print(f"  Generated codes shape: {pred.shape}")
    
    print("\nAll tests passed!")
