"""
Attention weight extraction and analysis.

Provides utilities for extracting and analyzing attention weights
from the Transformer model for interpretability.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer import TransformerRULPredictor
from config import Config


def extract_attention_weights(
    model: TransformerRULPredictor,
    x: torch.Tensor
) -> List[torch.Tensor]:
    """
    Extract attention weights from all layers.
    
    Args:
        model: Trained Transformer model.
        x: Input tensor of shape [batch_size, seq_len, input_dim].
        
    Returns:
        List of attention tensors, one per layer.
        Each has shape [batch_size, nhead, seq_len, seq_len].
    """
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(x, return_attention=True)
    return attention_weights


def aggregate_attention(
    attention_weights: List[torch.Tensor],
    method: str = 'mean'
) -> torch.Tensor:
    """
    Aggregate attention weights across layers and heads.
    
    Args:
        attention_weights: List of attention tensors from each layer.
        method: Aggregation method ('mean', 'max', 'last').
        
    Returns:
        Aggregated attention of shape [batch_size, seq_len, seq_len].
    """
    if method == 'last':
        # Use only last layer, average over heads
        return attention_weights[-1].mean(dim=1)
    
    # Stack: [num_layers, batch, heads, seq, seq]
    stacked = torch.stack(attention_weights, dim=0)
    
    if method == 'mean':
        return stacked.mean(dim=(0, 2))
    elif method == 'max':
        return stacked.max(dim=0)[0].max(dim=1)[0]
    else:
        raise ValueError(f"Unknown method: {method}")


def get_temporal_attention(
    attention_weights: List[torch.Tensor]
) -> torch.Tensor:
    """
    Get attention to each time step (how much attention each position receives).
    
    This shows which time steps the model considers most important.
    
    Args:
        attention_weights: List of attention tensors.
        
    Returns:
        Temporal importance scores of shape [batch_size, seq_len].
    """
    # Aggregate attention
    agg_attn = aggregate_attention(attention_weights, method='mean')
    
    # Sum attention received by each position (column sum)
    temporal_importance = agg_attn.sum(dim=1)
    
    # Normalize
    temporal_importance = temporal_importance / temporal_importance.sum(dim=1, keepdim=True)
    
    return temporal_importance


def get_feature_importance_from_attention(
    model: TransformerRULPredictor,
    x: torch.Tensor,
    feature_names: List[str],
    config: Optional[Config] = None
) -> Dict[str, float]:
    """
    Estimate feature importance based on attention patterns.
    
    Uses gradient-weighted attention to determine feature importance.
    
    Args:
        model: Trained model.
        x: Single input sample [1, seq_len, input_dim].
        feature_names: List of feature names.
        config: Configuration object.
        
    Returns:
        Dictionary mapping feature names to importance scores.
    """
    config = config or Config()
    model.eval()
    
    x = x.clone().requires_grad_(True)
    output, attention_weights = model(x, return_attention=True)
    
    # Compute gradients w.r.t. input
    output.backward()
    
    # Get absolute gradients
    gradients = x.grad.abs().detach().cpu().numpy()[0]  # [seq_len, input_dim]
    
    # Average across time
    feature_importance = gradients.mean(axis=0)
    
    # Normalize
    feature_importance = feature_importance / (feature_importance.sum() + 1e-10)
    
    return {name: float(imp) for name, imp in zip(feature_names, feature_importance)}


def analyze_attention_per_layer(
    attention_weights: List[torch.Tensor]
) -> Dict[str, np.ndarray]:
    """
    Analyze attention patterns per layer.
    
    Args:
        attention_weights: List of attention tensors.
        
    Returns:
        Dictionary with per-layer analysis.
    """
    analysis = {}
    
    for layer_idx, attn in enumerate(attention_weights):
        # attn shape: [batch, heads, seq, seq]
        attn_np = attn.cpu().numpy()
        
        analysis[f'layer_{layer_idx}'] = {
            'mean': float(attn_np.mean()),
            'std': float(attn_np.std()),
            'max': float(attn_np.max()),
            'diagonal_ratio': float(np.trace(attn_np[0, 0]) / attn_np.shape[-1])
        }
    
    return analysis


def get_head_diversity(
    attention_weights: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Measure diversity across attention heads.
    
    Higher diversity indicates heads are learning different patterns.
    
    Args:
        attention_weights: List of attention tensors.
        
    Returns:
        Diversity metrics per layer.
    """
    diversity = {}
    
    for layer_idx, attn in enumerate(attention_weights):
        # attn: [batch, heads, seq, seq]
        heads = attn[0]  # First sample: [heads, seq, seq]
        
        # Compute pairwise cosine similarity between heads
        heads_flat = heads.reshape(heads.shape[0], -1)  # [heads, seq*seq]
        heads_norm = heads_flat / (heads_flat.norm(dim=1, keepdim=True) + 1e-10)
        similarity = torch.mm(heads_norm, heads_norm.t())  # [heads, heads]
        
        # Average off-diagonal similarity (lower = more diverse)
        n_heads = similarity.shape[0]
        mask = ~torch.eye(n_heads, dtype=torch.bool, device=similarity.device)
        avg_similarity = similarity[mask].mean().item()
        
        diversity[f'layer_{layer_idx}'] = {
            'avg_head_similarity': avg_similarity,
            'diversity_score': 1 - avg_similarity
        }
    
    return diversity
