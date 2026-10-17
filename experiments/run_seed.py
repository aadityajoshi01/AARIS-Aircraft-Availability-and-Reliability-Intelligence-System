"""Repeatability run: train + evaluate FD001 with a given seed, into an isolated output dir.

Usage (from repo root):
    .venv/Scripts/python.exe experiments/run_seed.py --seed 43
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import Config
from train import Trainer
from models.transformer import create_model
from data.preprocessing import CMAPSSPreprocessor, create_sequences, create_test_sequences
from data.dataset import create_data_loaders
from evaluate import evaluate_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)

    config = Config()
    config.EPOCHS = args.epochs
    run_dir = os.path.join(ROOT, 'results', f'seed_{args.seed}')
    os.makedirs(run_dir, exist_ok=True)

    preprocessor = CMAPSSPreprocessor(config)
    train_df = preprocessor.prepare_train_data()
    X, y = create_sequences(train_df, config)
    train_loader, val_loader = create_data_loaders(X, y, config)

    model = create_model(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        nhead=config.N_HEADS,
        num_layers=config.N_ENCODER_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        device=config.DEVICE,
    )

    trainer = Trainer(model, config, save_dir=run_dir)
    trainer.train(train_loader, val_loader, epochs=args.epochs, verbose=False)
    trainer.load_best_model()

    test_df, true_rul = preprocessor.prepare_test_data()
    X_test, engine_ids = create_test_sequences(test_df, config)
    true_rul = true_rul[:len(engine_ids)]
    predictions, metrics = evaluate_model(model, X_test, true_rul, config, verbose=False)

    result = {
        'seed': args.seed,
        'epochs': args.epochs,
        'best_val_loss': trainer.best_val_loss,
        'best_epoch': trainer.best_epoch,
        'test': {k: float(v) for k, v in metrics.items()},
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
