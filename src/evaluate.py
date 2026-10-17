"""
Evaluation module for Transformer RUL predictor.

Provides functions for:
- Model evaluation on test set
- Metric computation (RMSE, MAE, NASA Score)
- Prediction generation and analysis
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models.transformer import TransformerRULPredictor, create_model
from models.loss import compute_overestimation_rate, NASAScore
from data.preprocessing import CMAPSSPreprocessor, create_test_sequences


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted RUL values.
        targets: True RUL values.
        
    Returns:
        Dictionary of metrics.
    """
    # Basic metrics
    rmse = math.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    # Overestimation rate
    over_rate = compute_overestimation_rate(
        torch.tensor(predictions),
        torch.tensor(targets)
    )
    
    # NASA Score
    nasa_scorer = NASAScore()
    nasa_score = nasa_scorer(
        torch.tensor(predictions),
        torch.tensor(targets)
    ).item()
    
    # Error analysis
    errors = predictions - targets
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'nasa_score': nasa_score,
        'overestimation_rate': over_rate,
        'mean_error': mean_error,
        'std_error': std_error,
        'max_overestimation': float(np.max(errors)),
        'max_underestimation': float(np.min(errors))
    }


def evaluate_model(
    model: TransformerRULPredictor,
    X_test: np.ndarray,
    true_rul: np.ndarray,
    config: Optional[Config] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model.
        X_test: Test sequences.
        true_rul: Ground truth RUL values.
        config: Configuration object.
        verbose: Print results.
        
    Returns:
        Tuple of (predictions, metrics dictionary).
    """
    config = config or Config()
    
    # Set model to eval mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.DEVICE)
        predictions, _ = model(X_tensor)
        predictions = predictions.cpu().numpy().flatten()
    
    # Model outputs are normalized by MAX_RUL; convert back to cycles
    predictions = np.clip(predictions * config.MAX_RUL, 0.0, config.MAX_RUL)
    
    # Compute metrics
    metrics = compute_metrics(predictions, true_rul)
    
    if verbose:
        print("\n" + "=" * 50)
        print("TEST SET EVALUATION RESULTS")
        print("=" * 50)
        print(f"  RMSE:                  {metrics['rmse']:.2f} cycles")
        print(f"  MAE:                   {metrics['mae']:.2f} cycles")
        print(f"  NASA Score:            {metrics['nasa_score']:.2f}")
        print(f"  Overestimation Rate:   {metrics['overestimation_rate']:.1%}")
        print("-" * 50)
        print(f"  Mean Error:            {metrics['mean_error']:.2f}")
        print(f"  Std Error:             {metrics['std_error']:.2f}")
        print(f"  Max Overestimation:    {metrics['max_overestimation']:.2f}")
        print(f"  Max Underestimation:   {metrics['max_underestimation']:.2f}")
        print("=" * 50 + "\n")
    
    return predictions, metrics


def generate_prediction_report(
    predictions: np.ndarray,
    true_rul: np.ndarray,
    engine_ids: np.ndarray,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate detailed prediction report.
    
    Args:
        predictions: Predicted RUL values.
        true_rul: True RUL values.
        engine_ids: Engine IDs.
        save_path: Path to save CSV report.
        
    Returns:
        DataFrame with predictions and analysis.
    """
    df = pd.DataFrame({
        'engine_id': engine_ids,
        'true_rul': true_rul,
        'predicted_rul': predictions.round(1),
        'error': (predictions - true_rul).round(2),
        'abs_error': np.abs(predictions - true_rul).round(2),
        'overestimated': predictions > true_rul
    })
    
    # Add urgency flag
    df['urgent'] = df['predicted_rul'] <= 30
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to: {save_path}")
    
    return df


def load_and_evaluate(
    checkpoint_path: str,
    config: Optional[Config] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load model from checkpoint and evaluate on test set.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config: Configuration object.
        
    Returns:
        Tuple of (predictions DataFrame, metrics dictionary).
    """
    config = config or Config()
    
    print("Loading model from checkpoint...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    
    # Create model
    model = create_model(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        nhead=config.N_HEADS,
        num_layers=config.N_ENCODER_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        device=config.DEVICE
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Prepare test data
    print("Preparing test data...")
    preprocessor = CMAPSSPreprocessor(config)
    
    # Need to fit on training data first
    train_df = preprocessor.prepare_train_data()
    test_df, true_rul = preprocessor.prepare_test_data()
    
    # Create test sequences
    X_test, engine_ids = create_test_sequences(test_df, config)
    true_rul = true_rul[:len(engine_ids)]
    
    print(f"Test engines: {len(engine_ids)}")
    
    # Evaluate
    predictions, metrics = evaluate_model(
        model, X_test, true_rul, config, verbose=True
    )
    
    # Generate report
    report_path = os.path.join(config.OUTPUT_DIR, 'predictions.csv')
    predictions_df = generate_prediction_report(
        predictions, true_rul, engine_ids, report_path
    )
    
    return predictions_df, metrics


def run_evaluation(config: Optional[Config] = None):
    """
    Complete evaluation pipeline.
    
    Args:
        config: Configuration object.
    """
    config = config or Config()
    
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'checkpoint_best.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}")
        print("Please train the model first using: python main.py --mode train")
        return
    
    predictions_df, metrics = load_and_evaluate(checkpoint_path, config)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"  Total engines evaluated: {len(predictions_df)}")
    print(f"  Urgent attention needed: {predictions_df['urgent'].sum()}")
    print(f"  Overestimated count: {predictions_df['overestimated'].sum()}")


if __name__ == "__main__":
    run_evaluation()
