"""NASA score convention audit.

Standard PHM08 score (Saxena et al. 2008): with e = pred - true,
  e < 0 (early/underestimation): exp(-e/13) - 1
  e >= 0 (late/overestimation):  exp(e/10) - 1
i.e. overestimates are penalized with the steeper /10 scale.

This script recomputes both the repo's implemented score and the standard
score for every model on the frozen FD001 test split.

Usage: .venv/Scripts/python.exe experiments/nasa_audit.py
"""

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import Config
from models.loss import NASAScore
from models.transformer import create_model
from data.preprocessing import CMAPSSPreprocessor, create_sequences, create_test_sequences
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from baselines import LSTMBaseline, run_lstm


def standard_nasa(pred, true):
    e = np.asarray(pred) - np.asarray(true)
    early = e < 0
    s = np.zeros_like(e, dtype=float)
    s[early] = np.exp(-e[early] / 13.0) - 1.0
    s[~early] = np.exp(e[~early] / 10.0) - 1.0
    return float(s.sum())


def repo_nasa(pred, true):
    return NASAScore()(torch.tensor(pred), torch.tensor(true)).item()


if __name__ == '__main__':
    config = Config()
    prep = CMAPSSPreprocessor(config)
    train_df = prep.prepare_train_data()
    X, y = create_sequences(train_df, config)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED)
    test_df, true_rul = prep.prepare_test_data()
    X_test, ids = create_test_sequences(test_df, config)
    true_rul = true_rul[:len(ids)]

    preds = {}

    preds['constant_mean'] = np.full(len(true_rul), float(y_tr.mean() * config.MAX_RUL))
    preds['constant_38'] = np.full(len(true_rul), 38.0)

    ridge = Ridge(alpha=10.0).fit(X_tr.reshape(len(X_tr), -1), y_tr)
    preds['ridge'] = np.clip(ridge.predict(X_test.reshape(len(X_test), -1)) * config.MAX_RUL, 0, config.MAX_RUL)

    lstm_norm = run_lstm(X_tr, y_tr, X_test, config, epochs=30, seed=42)
    preds['lstm'] = np.clip(lstm_norm.flatten() * config.MAX_RUL, 0, config.MAX_RUL)
    lstm_rmse_check = float(np.sqrt(np.mean((preds['lstm'] - true_rul) ** 2)))

    for seed, path in [(42, 'outputs/checkpoint_best.pt'),
                       (43, 'results/seed_43/checkpoint_best.pt'),
                       (44, 'results/seed_44/checkpoint_best.pt')]:
        ck = torch.load(os.path.join(ROOT, path), map_location='cpu', weights_only=False)
        m = create_model(input_dim=config.INPUT_DIM, d_model=config.D_MODEL, nhead=config.N_HEADS,
                        num_layers=config.N_ENCODER_LAYERS, d_ff=config.D_FF, dropout=config.DROPOUT)
        m.load_state_dict(ck['model_state_dict']); m.eval()
        with torch.no_grad():
            out, _ = m(torch.tensor(X_test))
        preds[f'transformer_seed{seed}'] = np.clip(out.numpy().flatten() * config.MAX_RUL, 0, config.MAX_RUL)

    table = {}
    for name, p in preds.items():
        rmse = float(np.sqrt(np.mean((p - true_rul) ** 2)))
        table[name] = {
            'rmse': round(rmse, 2),
            'overest_rate': round(float(np.mean(p > true_rul)), 2),
            'repo_nasa_as_implemented': round(repo_nasa(p, true_rul), 1),
            'nasa_standard_correct': round(standard_nasa(p, true_rul), 1),
        }

    print('LSTM reproducibility check (RMSE, expect ~17.65):', round(lstm_rmse_check, 2))
    print(json.dumps(table, indent=2))
    out_path = os.path.join(ROOT, 'results', 'nasa_audit.json')
    with open(out_path, 'w') as f:
        json.dump(table, f, indent=2)
    print('saved', out_path)
