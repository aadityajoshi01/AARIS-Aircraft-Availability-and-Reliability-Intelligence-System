"""
PyTorch Dataset classes for NASA C-MAPSS RUL prediction.

Provides DataLoader-compatible datasets for training, validation, and testing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class RULDataset(Dataset):
    """
    PyTorch Dataset for RUL prediction.
    
    Wraps numpy arrays of sequences and targets for use with DataLoader.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences of shape [num_samples, seq_length, num_features].
            y: Target RUL values of shape [num_samples].
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class RULTestDataset(Dataset):
    """
    PyTorch Dataset for test set (no targets).
    """
    
    def __init__(self, X: np.ndarray, engine_ids: np.ndarray):
        """
        Initialize test dataset.
        
        Args:
            X: Input sequences.
            engine_ids: Engine IDs corresponding to each sequence.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.engine_ids = torch.tensor(engine_ids, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.engine_ids[idx]


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[Config] = None,
    val_split: Optional[float] = None,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        X: Full training sequences.
        y: Full training targets.
        config: Configuration object.
        val_split: Validation split ratio (uses config default if None).
        shuffle_train: Whether to shuffle training data.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    config = config or Config()
    val_split = val_split or config.VALIDATION_SPLIT
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_split,
        random_state=config.RANDOM_SEED
    )
    
    # Create datasets
    train_dataset = RULDataset(X_train, y_train)
    val_dataset = RULDataset(X_val, y_val)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle_train,
        num_workers=0,  # Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def create_test_loader(
    X: np.ndarray,
    engine_ids: np.ndarray,
    config: Optional[Config] = None
) -> DataLoader:
    """
    Create test DataLoader.
    
    Args:
        X: Test sequences.
        engine_ids: Engine IDs.
        config: Configuration object.
        
    Returns:
        Test DataLoader.
    """
    config = config or Config()
    
    test_dataset = RULTestDataset(X, engine_ids)
    
    return DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
