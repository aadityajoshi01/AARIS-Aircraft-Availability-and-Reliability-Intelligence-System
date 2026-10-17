"""
NASA C-MAPSS Transformer-Based RUL Prediction

Main entry point for training, evaluation, and visualization.

Usage:
    python main.py --mode train          # Train the model
    python main.py --mode evaluate       # Evaluate on test set
    python main.py --mode visualize      # Generate visualizations
    python main.py --mode all            # Full pipeline
"""

import argparse
import os
import sys
import json
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from train import train_model, Trainer
from evaluate import run_evaluation, load_and_evaluate
from models.transformer import create_model
from data.preprocessing import CMAPSSPreprocessor, create_sequences, create_test_sequences
from data.dataset import create_data_loaders
from visualization.plots import create_all_visualizations

import random
import numpy as np

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NASA C-MAPSS Transformer RUL Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode train --epochs 50
    python main.py --mode evaluate
    python main.py --mode visualize
    python main.py --mode all
        """
    )
    
    parser.add_argument(
        '--mode', type=str, default='all',
        choices=['train', 'evaluate', 'visualize', 'all'],
        help='Operation mode (default: all)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (default: from config)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (default: from config)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate (default: from config)'
    )
    parser.add_argument(
        '--dataset', type=str, default='FD001',
        choices=['FD001', 'FD002', 'FD003', 'FD004'],
        help='Dataset to use (default: FD001)'
    )
    
    return parser.parse_args()


def print_header():
    """Print application header."""
    header = """
╔══════════════════════════════════════════════════════════════╗
║     NASA C-MAPSS Transformer-Based RUL Prediction            ║
║     Remaining Useful Life Prediction for Jet Engines         ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(header)


def run_training(config: Config, epochs: int = None):
    """Run training pipeline."""
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    if epochs:
        config.EPOCHS = epochs
    
    model, history = train_model(config)
    return model, history


def run_visualization(config: Config):
    """Generate all visualizations."""
    print("\n" + "=" * 60)
    print("VISUALIZATION PHASE")
    print("=" * 60)
    
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'checkpoint_best.pt')
    history_path = os.path.join(config.OUTPUT_DIR, 'training_history.json')
    
    if not os.path.exists(checkpoint_path):
        print("Error: No trained model found. Please run training first.")
        return
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
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
    
    # Load history
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            print("Warning: Could not load training history (corrupted). Skipping training curves.")
            history = {}
    
    # Prepare test data
    preprocessor = CMAPSSPreprocessor(config)
    train_df = preprocessor.prepare_train_data()
    test_df, true_rul = preprocessor.prepare_test_data()
    X_test, engine_ids = create_test_sequences(test_df, config)
    true_rul = true_rul[:len(engine_ids)]
    
    # Get predictions (model outputs normalized RUL; convert to cycles)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.DEVICE)
        predictions, _ = model(X_tensor)
        predictions = np.clip(
            predictions.cpu().numpy().flatten() * config.MAX_RUL,
            0.0, config.MAX_RUL
        )
    
    # Create visualizations
    create_all_visualizations(
        model, X_test, predictions, true_rul, history, config,
        train_df=train_df, engine_ids=engine_ids
    )


def main():
    """Main entry point."""
    args = parse_args()
    print_header()
    
    # Initialize config
    config = Config()
    
    # Apply command line overrides
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # Set dataset
    if args.dataset != 'FD001':
        config.TRAIN_FILE = f"train_{args.dataset}.txt"
        config.TEST_FILE = f"test_{args.dataset}.txt"
        config.RUL_FILE = f"RUL_{args.dataset}.txt"
    
    print(f"Device: {config.DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    
    # Execute based on mode
    if args.mode in ['train', 'all']:
        run_training(config, args.epochs)
    
    if args.mode in ['evaluate', 'all']:
        run_evaluation(config)
    
    if args.mode in ['visualize', 'all']:
        run_visualization(config)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
