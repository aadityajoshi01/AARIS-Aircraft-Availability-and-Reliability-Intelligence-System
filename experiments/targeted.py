"""Targeted experiments vs the Ridge baseline; selection on VALIDATION only.

Variants (seed 42, 30 epochs, identical 80/20 split random_state=42):
  no_pe    - positional encoding replaced by Identity (does temporal order help?)
  sensors  - drop the 6 zero-variance sensors (+ their dead scaled channels)
  seq50    - window length 50 instead of 30

Control: the frozen seed-42 run's own validation curve, truncated to 30 epochs
(results/run_seed42_main/training_history.json).

The test set is NOT touched here; it is reserved for one final evaluation of
whichever configuration validation selects.

Usage: .venv/Scripts/python.exe experiments/targeted.py --variant sensors
"""

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import Config
from train import Trainer
from models.transformer import create_model
from data.preprocessing import CMAPSSPreprocessor, create_sequences
from data.dataset import create_data_loaders

DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']


def build_config(variant):
    cfg = Config()
    cfg.EPOCHS = 30
    if variant == 'seq50':
        cfg.SEQUENCE_LENGTH = 50
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=['no_pe', 'sensors', 'seq50'])
    args = parser.parse_args()

    import random, numpy as np
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    cfg = build_config(args.variant)

    prep = CMAPSSPreprocessor(cfg)
    train_df = prep.prepare_train_data()
    if args.variant == 'sensors':
        # load/scaling use the full column set (load_data names all 26 raw columns);
        # drop dead sensors only for sequence building. MinMax is per-feature,
        # so remaining features' scaled values are identical to fitting without them.
        train_df = train_df.drop(columns=DROP_SENSORS)
        cfg.SENSOR_COLS = [s for s in cfg.SENSOR_COLS if s not in DROP_SENSORS]
    X, y = create_sequences(train_df, cfg)
    train_loader, val_loader = create_data_loaders(X, y, cfg)

    model = create_model(
        input_dim=cfg.INPUT_DIM, d_model=cfg.D_MODEL, nhead=cfg.N_HEADS,
        num_layers=cfg.N_ENCODER_LAYERS, d_ff=cfg.D_FF, dropout=cfg.DROPOUT,
        device=cfg.DEVICE,
    )
    if args.variant == 'no_pe':
        model.positional_encoding = torch.nn.Identity()

    save_dir = os.path.join(ROOT, 'results', 'targeted', args.variant)
    trainer = Trainer(model, cfg, save_dir=save_dir)
    trainer.train(train_loader, val_loader, epochs=30, verbose=False)
    trainer.load_best_model()

    best_val_rmse = min(trainer.history['val_rmse'])
    result = {
        'variant': args.variant,
        'epochs': 30,
        'input_dim': cfg.INPUT_DIM,
        'seq_len': cfg.SEQUENCE_LENGTH,
        'best_val_loss': float(trainer.best_val_loss),
        'best_val_rmse_cycles': float(best_val_rmse),
        'best_epoch': int(trainer.best_epoch),
        'note': 'validation-only selection; test untouched',
    }
    with open(os.path.join(save_dir, 'val_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
