"""
Utility functions for CSJ-ID experiments.
Includes logging, timing, and helper functions.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager

import torch
import numpy as np
import random


class Logger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_path: str, name: str = "CSJ-ID"):
        self.log_path = log_path
        self.name = name
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Write header
        self.logger.info("=" * 70)
        self.logger.info(f"CSJ-ID Experiment Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 70)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def section(self, title: str):
        """Print a section header."""
        self.logger.info("")
        self.logger.info("-" * 70)
        self.logger.info(f" {title}")
        self.logger.info("-" * 70)
    
    def metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics dictionary."""
        msg = prefix + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(msg)
    
    def table(self, headers: list, rows: list, title: str = ""):
        """Print a formatted table."""
        if title:
            self.logger.info(f"\n{title}")
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Print header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        self.logger.info(header_line)
        self.logger.info("-" * len(header_line))
        
        # Print rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            self.logger.info(row_line)


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "", logger: Optional[Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.logger:
            self.logger.info(f"{self.name} completed in {self.elapsed:.2f}s")
    
    @property
    def elapsed_str(self) -> str:
        if self.elapsed is None:
            return "N/A"
        if self.elapsed < 60:
            return f"{self.elapsed:.2f}s"
        elif self.elapsed < 3600:
            return f"{self.elapsed/60:.1f}m"
        else:
            return f"{self.elapsed/3600:.1f}h"


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    extra: Optional[Dict] = None
):
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary as string."""
    return " | ".join([f"{k}: {v:.{precision}f}" for k, v in metrics.items()])


if __name__ == "__main__":
    # Test logger
    logger = Logger("test_log.txt")
    logger.section("Test Section")
    logger.info("This is an info message")
    logger.metrics({"loss": 0.1234, "acc": 0.9876})
    
    # Test timer
    with Timer("Test operation", logger):
        time.sleep(0.5)
    
    print("Utilities test complete!")
