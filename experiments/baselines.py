"""Baselines on the exact FD001 split used by the Transformer pipeline.

- constant mean predictor (over training windows)
- constant asymmetric-loss optimum (~38 cycles, the collapsed model's value)
- ridge regression on flattened window features
- 1-layer LSTM (hidden 64), same normalized targets, same asymmetric loss,
  same optimizer settings, 30 epochs

Usage (from repo root):
    .venv/Scripts/python.exe experiments/baselines.py
"""

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import Config
from data.preprocessing import CMAPSSPreprocessor, create_sequences, create_test_sequences
from data.dataset import create_data_loaders
from models.loss import AsymmetricMSELoss
from evaluate import compute_metrics


def to_cycles(pred_norm):
    return np.clip(np.asarray(pred_norm).flatten() * MAX_RUL, 0.0, MAX_RUL)


class LSTMBaseline(torch.nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


def run_lstm(X_train, y_train, X_test, config, epochs=30, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config.DEVICE
    model = LSTMBaseline(X_train.shape[2]).to(device)
    criterion = AsymmetricMSELoss(config.LAMBDA_OVER)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    Xs = torch.tensor(X_train, device=device)
    ys = torch.tensor(y_train, device=device).view(-1, 1)
    n = len(Xs)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            opt.zero_grad()
            loss = criterion(model(Xs[idx]), ys[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, device=device)).cpu().numpy()
    return pred


if __name__ == '__main__':
    config = Config()
    MAX_RUL = config.MAX_RUL

    preprocessor = CMAPSSPreprocessor(config)
    train_df = preprocessor.prepare_train_data()
    X, y = create_sequences(train_df, config)  # y normalized [0,1]

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED
    )

    test_df, true_rul = preprocessor.prepare_test_data()
    X_test, engine_ids = create_test_sequences(test_df, config)
    true_rul = true_rul[:len(engine_ids)]

    results = {}

    # 1) constant mean of training labels (raw cycles)
    mean_cycles = float(y_train.mean() * MAX_RUL)
    pred = np.full(len(true_rul), mean_cycles)
    results['constant_mean'] = compute_metrics(pred, true_rul)

    # 2) constant asymmetric-loss optimum (~38 cycles, the collapsed value)
    pred = np.full(len(true_rul), 38.0)
    results['constant_asym_opt_38'] = compute_metrics(pred, true_rul)

    # 3) ridge regression on flattened window features
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train.reshape(len(X_train), -1), y_train)
    results['ridge_flattened_window'] = compute_metrics(
        to_cycles(ridge.predict(X_test.reshape(len(X_test), -1))), true_rul
    )

    # 4) 1-layer LSTM, same targets/loss/optimizer, 30 epochs
    pred_norm = run_lstm(X_train, y_train, X_test, config, epochs=30, seed=42)
    results['lstm_1layer_h64_30ep'] = compute_metrics(to_cycles(pred_norm), true_rul)

    out = {k: {m: float(v) for m, v in r.items()} for k, r in results.items()}
    path = os.path.join(ROOT, 'results', 'baselines.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: {'rmse': round(v['rmse'], 2), 'mae': round(v['mae'], 2),
                          'nasa_score': round(v['nasa_score'], 1)}
                      for k, v in out.items()}, indent=2))
