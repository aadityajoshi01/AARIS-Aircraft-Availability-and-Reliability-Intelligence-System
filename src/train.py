"""
Training pipeline for Transformer RUL predictor.

Handles the complete training loop with:
- Early stopping
- Model checkpointing
- Training history logging
- Validation monitoring
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models.transformer import TransformerRULPredictor, create_model
from models.loss import AsymmetricMSELoss, compute_overestimation_rate
from data.preprocessing import CMAPSSPreprocessor, create_sequences
from data.dataset import create_data_loaders


class EarlyStopping:
    """
    Early stopping handler.
    
    Monitors validation loss and stops training if no improvement
    is seen for a specified number of epochs.
    """
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' (lower is better) or 'max' (higher is better).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score.
            
        Returns:
            True if training should stop.
        """
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
                self.early_stop = True
                return True
        
        return False


class Trainer:
    """
    Training manager for Transformer RUL predictor.
    
    Handles the complete training process including:
    - Training loop with gradient updates
    - Validation monitoring
    - Early stopping
    - Model checkpointing
    - Training history logging
    """
    
    def __init__(
        self,
        model: TransformerRULPredictor,
        config: Optional[Config] = None,
        save_dir: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: The Transformer model to train.
            config: Configuration object.
            save_dir: Directory to save checkpoints and logs.
        """
        self.model = model
        self.config = config or Config()
        self.save_dir = save_dir or self.config.OUTPUT_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Loss function
        self.criterion = AsymmetricMSELoss(lambda_over=self.config.LAMBDA_OVER)
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.EARLY_STOPPING_PATIENCE
        )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_over_rate': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.config.DEVICE)
            y_batch = y_batch.to(self.config.DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, _ = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader) -> Tuple[float, float, float]:
        """
        Validate model performance.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Tuple of (val_loss, rmse, overestimation_rate).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.config.DEVICE)
                y_batch = y_batch.to(self.config.DEVICE)
                
                predictions, _ = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_trues.append(y_batch.cpu().numpy())
        
        # Aggregate predictions
        all_preds = np.concatenate(all_preds).flatten()
        all_trues = np.concatenate(all_trues).flatten()
        
        # Compute metrics (labels are normalized by MAX_RUL; report RMSE in cycles)
        val_loss = total_loss / len(val_loader)
        rmse = np.sqrt(np.mean((all_preds - all_trues) ** 2)) * self.config.MAX_RUL
        over_rate = float(np.mean(all_preds > all_trues))
        
        return val_loss, rmse, over_rate
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': {
                'd_model': self.config.D_MODEL,
                'nhead': self.config.N_HEADS,
                'num_layers': self.config.N_ENCODER_LAYERS
            }
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_latest.pt'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_best.pt'))
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        # Convert numpy floats to Python floats for JSON serialization
        serializable_history = {
            key: [float(v) for v in values]
            for key, values in self.history.items()
        }
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def train(
        self, 
        train_loader, 
        val_loader, 
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs (uses config default if None).
            verbose: Whether to print progress.
            
        Returns:
            Training history dictionary.
        """
        epochs = epochs or self.config.EPOCHS
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Transformer RUL Predictor")
            print(f"Device: {self.config.DEVICE}")
            print(f"Epochs: {epochs}")
            print(f"Early Stopping Patience: {self.config.EARLY_STOPPING_PATIENCE}")
            print(f"{'='*60}\n")
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_rmse, val_over_rate = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_over_rate'].append(val_over_rate)
            self.history['learning_rate'].append(current_lr)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"RMSE: {val_rmse:.2f} | "
                    f"Over-Rate: {val_over_rate:.2%} | "
                    f"LR: {current_lr:.2e}"
                    + (" *" if is_best else "")
                )
            
            # Early stopping check
            if self.early_stopping(val_loss):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Save final history
        self.save_history()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"Best Validation Loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
            print(f"Model saved to: {self.save_dir}")
            print(f"{'='*60}\n")
        
        return self.history
    
    def load_best_model(self):
        """Load the best checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint_best.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            print("No best checkpoint found")


def train_model(config: Optional[Config] = None) -> Tuple[TransformerRULPredictor, Dict]:
    """
    Complete training pipeline.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (trained model, training history).
    """
    config = config or Config()
    
    print("=" * 60)
    print("NASA C-MAPSS Transformer RUL Prediction")
    print("=" * 60)
    
    # Prepare data
    print("\n[1/4] Loading and preprocessing data...")
    preprocessor = CMAPSSPreprocessor(config)
    train_df = preprocessor.prepare_train_data()
    
    stats = preprocessor.get_statistics(train_df)
    print(f"      Engines: {stats['num_engines']}")
    print(f"      Total records: {stats['total_records']}")
    print(f"      Avg cycles/engine: {stats['avg_cycles_per_engine']:.1f}")
    
    # Create sequences
    print("\n[2/4] Creating training sequences...")
    X, y = create_sequences(train_df, config)
    print(f"      Sequences: {len(X)}")
    print(f"      Shape: {X.shape}")
    
    # Create data loaders
    print("\n[3/4] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(X, y, config)
    print(f"      Train batches: {len(train_loader)}")
    print(f"      Val batches: {len(val_loader)}")
    
    # Create model
    print("\n[4/4] Initializing model...")
    model = create_model(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        nhead=config.N_HEADS,
        num_layers=config.N_ENCODER_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        device=config.DEVICE
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total parameters: {total_params:,}")
    
    # Train
    trainer = Trainer(model, config)
    history = trainer.train(train_loader, val_loader)
    
    # Load best model
    trainer.load_best_model()
    
    return model, history


if __name__ == "__main__":
    model, history = train_model()
