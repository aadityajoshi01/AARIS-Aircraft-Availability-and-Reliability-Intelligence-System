"""
Positional Encoding for Transformer models.

Adds positional information to input embeddings to help the model
understand sequence order.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as described in 'Attention is All You Need'.
    
    Adds fixed sinusoidal positional embeddings to input tensors to provide
    the model with information about token positions in the sequence.
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 500, 
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension (must match model dimension).
            max_len: Maximum sequence length to support.
            dropout: Dropout rate to apply after adding positional encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for sin/cos frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but persists in state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding.
    
    Uses learnable embeddings instead of fixed sinusoidal patterns.
    Can sometimes work better for specific domains.
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 500, 
        dropout: float = 0.1
    ):
        """
        Initialize learned positional encoding.
        
        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            
        Returns:
            Tensor with positional encoding added.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)
