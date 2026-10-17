"""
Transformer Encoder model for Remaining Useful Life prediction.

Implements a pure Transformer-based architecture with multi-head self-attention
and provides methods to extract attention weights for model interpretability.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.positional_encoding import PositionalEncoding


class TransformerEncoderLayerWithAttention(nn.Module):
    """
    Custom Transformer Encoder Layer that exposes attention weights.
    
    Identical to nn.TransformerEncoderLayer but returns attention weights
    for visualization and interpretability.
    """
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        d_ff: int = 256, 
        dropout: float = 0.1
    ):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Model dimension.
            nhead: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weight return.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model].
            return_attention: Whether to return attention weights.
            
        Returns:
            Tuple of (output tensor, attention weights or None).
        """
        # Self-attention with residual
        attn_output, attn_weights = self.self_attn(
            x, x, x, 
            need_weights=return_attention,
            average_attn_weights=False  # Get per-head attention
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class TransformerRULPredictor(nn.Module):
    """
    Transformer Encoder for Remaining Useful Life Prediction.
    
    Architecture:
    1. Input projection: [input_dim] -> [d_model]
    2. Positional encoding
    3. N x Transformer Encoder layers (multi-head self-attention + FFN)
    4. Global average pooling over time steps
    5. Regression head for RUL prediction
    
    Features:
    - Multi-head self-attention captures long-range temporal dependencies
    - Attention weight extraction for model interpretability
    - Suitable for variable-length time series
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Initialize Transformer model.
        
        Args:
            input_dim: Number of input features (sensors + op_settings).
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of encoder layers.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout rate.
            max_seq_len: Maximum sequence length for positional encoding.
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Input projection: map features to model dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )
        
        # Transformer encoder layers with attention extraction
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Storage for attention weights (populated during forward pass)
        self._attention_weights: List[torch.Tensor] = []
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through the Transformer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
            return_attention: If True, return attention weights from all layers.
            
        Returns:
            If return_attention is False:
                RUL predictions of shape [batch_size, 1].
            If return_attention is True:
                Tuple of (predictions, list of attention weight tensors).
                Each attention tensor has shape [batch_size, nhead, seq_len, seq_len].
        """
        # Input projection
        x = self.input_projection(x)  # [B, T, d_model]
        
        # Add positional encoding
        x = self.positional_encoding(x)  # [B, T, d_model]
        
        # Pass through encoder layers
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_weights.append(attn)
        
        # Global average pooling over time dimension
        x = x.mean(dim=1)  # [B, d_model]
        
        # Regression head
        output = self.regression_head(x)  # [B, 1]
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def get_attention_maps(
        self, 
        x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get attention weight maps for visualization.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
            
        Returns:
            List of attention tensors, one per layer.
            Each has shape [batch_size, nhead, seq_len, seq_len].
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights
    
    def get_aggregated_attention(
        self, 
        x: torch.Tensor, 
        aggregation: str = 'mean'
    ) -> torch.Tensor:
        """
        Get aggregated attention weights across heads and layers.
        
        Args:
            x: Input tensor.
            aggregation: How to aggregate ('mean', 'max', 'last_layer').
            
        Returns:
            Aggregated attention of shape [batch_size, seq_len, seq_len].
        """
        attention_maps = self.get_attention_maps(x)
        
        if aggregation == 'last_layer':
            # Use only the last layer, average over heads
            return attention_maps[-1].mean(dim=1)
        
        # Stack all layers: [num_layers, batch, heads, seq, seq]
        stacked = torch.stack(attention_maps, dim=0)
        
        if aggregation == 'mean':
            # Average over layers and heads
            return stacked.mean(dim=(0, 2))
        elif aggregation == 'max':
            # Max over layers and heads
            return stacked.max(dim=0)[0].max(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


def create_model(
    input_dim: int,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 4,
    d_ff: int = 256,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
) -> TransformerRULPredictor:
    """
    Factory function to create a Transformer RUL predictor.
    
    Args:
        input_dim: Number of input features.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of encoder layers.
        d_ff: Feed-forward dimension.
        dropout: Dropout rate.
        device: Device to place model on.
        
    Returns:
        Initialized TransformerRULPredictor model.
    """
    model = TransformerRULPredictor(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    if device is not None:
        model = model.to(device)
    
    return model
