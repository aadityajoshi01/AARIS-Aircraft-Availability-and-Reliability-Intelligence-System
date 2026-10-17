"""
Loss functions for RUL prediction.

Includes standard MSE and asymmetric loss that penalizes overestimation
(critical for safety applications like aircraft maintenance).
"""

import torch
import torch.nn as nn
from typing import Optional


class AsymmetricMSELoss(nn.Module):
    """
    Asymmetric Mean Squared Error Loss.
    
    Penalizes overestimation of RUL more heavily than underestimation.
    This is important for safety-critical applications where predicting
    a longer remaining life than actual could lead to missed maintenance.
    
    Loss = MSE + lambda_over * mean(max(0, pred - true)^2)
    """
    
    def __init__(self, lambda_over: float = 10.0):
        """
        Initialize asymmetric loss.
        
        Args:
            lambda_over: Weight for overestimation penalty.
                        Higher values = stronger penalty for overestimation.
        """
        super().__init__()
        self.lambda_over = lambda_over
        self.mse = nn.MSELoss()
        
    def forward(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            pred: Predicted RUL values.
            true: True RUL values.
            
        Returns:
            Scalar loss value.
        """
        # Base MSE loss
        base_loss = self.mse(pred, true)
        
        # Overestimation penalty (only when pred > true)
        overestimation = torch.relu(pred - true)
        over_penalty = (overestimation ** 2).mean()
        
        return base_loss + self.lambda_over * over_penalty


class WeightedAsymmetricLoss(nn.Module):
    """
    Weighted Asymmetric Loss.
    
    Similar to AsymmetricMSELoss but also weights samples by their true RUL.
    This gives more importance to engines with higher RUL (early degradation).
    """
    
    def __init__(self, lambda_over: float = 10.0, max_rul: float = 125.0):
        """
        Initialize weighted asymmetric loss.
        
        Args:
            lambda_over: Weight for overestimation penalty.
            max_rul: Maximum RUL for normalization.
        """
        super().__init__()
        self.lambda_over = lambda_over
        self.max_rul = max_rul
        
    def forward(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted asymmetric loss.
        
        Args:
            pred: Predicted RUL values.
            true: True RUL values.
            
        Returns:
            Scalar loss value.
        """
        # Sample weights based on true RUL
        weights = true / self.max_rul
        
        # Weighted MSE
        squared_error = (pred - true) ** 2
        weighted_mse = (weights * squared_error).mean()
        
        # Overestimation penalty
        overestimation = torch.relu(pred - true)
        over_penalty = (overestimation ** 2).mean()
        
        return weighted_mse + self.lambda_over * over_penalty


class NASAScore(nn.Module):
    """
    NASA Prognostics Score Function.
    
    Standard metric from the PHM08 challenge that uses exponential
    penalties for early/late predictions.
    """
    
    def forward(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NASA prognostics score (lower is better).
        
        Standard PHM08 convention (Saxena et al. 2008), with d = true - pred:
            d >= 0 (early prediction / underestimation): exp(d/13) - 1
            d <  0 (late prediction / overestimation):   exp(-d/10) - 1
        so overestimation uses the steeper /10 scale.
        
        Args:
            pred: Predicted RUL values.
            true: True RUL values.
            
        Returns:
            Total score.
        """
        d = true - pred  # Positive = underestimation (early, safe), Negative = overestimation (late, dangerous)
        
        # Late predictions (overestimation, d < 0): steeper /10 penalty
        late_mask = d < 0
        late_score = torch.sum(torch.exp(-d[late_mask] / 10) - 1)
        
        # Early predictions (underestimation, d >= 0): milder /13 penalty
        early_mask = d >= 0
        early_score = torch.sum(torch.exp(d[early_mask] / 13) - 1)
        
        return late_score + early_score


def compute_overestimation_rate(
    pred: torch.Tensor, 
    true: torch.Tensor
) -> float:
    """
    Compute the fraction of predictions that overestimate RUL.
    
    Args:
        pred: Predicted RUL values.
        true: True RUL values.
        
    Returns:
        Overestimation rate as a float between 0 and 1.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
        
    overestimated = (pred - true) > 0
    return float(overestimated.sum()) / len(overestimated)
