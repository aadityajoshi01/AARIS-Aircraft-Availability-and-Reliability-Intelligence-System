"""Leak-free validation protocol: ENGINE-WISE split re-run of the targeted set.

Why: src/data/dataset.py:create_data_loaders splits *windows* with
train_test_split, but create_sequences makes stride-1 sliding windows per
engine - consecutive windows share 29/30 rows - so near-duplicate windows
landed in both train and validation. Every previous validation number
(control 10.48, no_pe 4.92, seq50 7.71, sensors 13.77) is inflated by this
leak (docs/AUDIT_AND_IMPROVEMENTS.md section 0). The test set is unaffected
(test engines are disjoint; final-window construction).

This script re-runs the four configs (control, no_pe, seq50, sensors) on an
ENGINE-WISE split: 20% of the training *engines* are held out whole, so no
engine contributes windows to both sides. Seeds 42/43/44 (the seed also
re-draws the engine split, which is the honest treatment of run variance).
Everything else is identical to experiments/targeted.py: same preprocessor,
same normalized labels, same model, same asymmetric loss, same Trainer,
same 30-epoch budget (early stopping patience 15).

Validation-only. The test set is NOT touched. No frozen artifact is read
with write intent; outputs go to results/engine_split/ (new directory).

Usage:
  .venv/Scripts/python.exe experiments/engine_split.py --probe        # 1 epoch timing
  .venv/Scripts/python.exe experiments/engine_split.py                # full 12-run matrix
  .venv/Scripts/python.exe experiments/engine_split.py --runs control:42,no_pe:42
                                                                      # subset (for parallel
                                                                      # processes; OMP_NUM_THREADS=4)

Each run writes results/engine_split/{variant}_seed{seed}/val_result.json.
summary.json is rebuilt by scanning those files, so parallel processes never
race on it.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import Config
from train import Trainer
from models.transformer import create_model
from data.preprocessing import CMAPSSPreprocessor
from data.dataset import RULDataset
from torch.utils.data import DataLoader

DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
VARIANTS = ['control', 'no_pe', 'seq50', 'sensors']
SEEDS = [42, 43, 44]
EPOCHS = 30
OUT_ROOT = os.path.join(ROOT, 'results', 'engine_split')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_sequences_with_ids(df, cfg):
    """Byte-identical window recipe to preprocessing.create_sequences
    (same iteration order, same stride, same target), but also returns the
    engine id of each window so the split can be engine-wise."""
    seq_length = cfg.SEQUENCE_LENGTH
    X, y, ids = [], [], []
    for engine_id in df[cfg.ID_COL].unique():
        engine_df = df[df[cfg.ID_COL] == engine_id].reset_index(drop=True)
        for i in range(len(engine_df) - seq_length + 1):
            X.append(engine_df.iloc[i:i + seq_length][cfg.FEATURE_COLS].values)
            y.append(engine_df.iloc[i + seq_length - 1]['RUL'] / cfg.MAX_RUL)
            ids.append(engine_id)
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(ids))


def engine_wise_split(ids, seed, val_frac=0.2):
    """Split by engine, never by window. Returns (train_mask, val_mask,
    n_val_engines)."""
    engines = np.unique(ids)
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(engines))
    n_val = max(1, int(round(val_frac * len(engines))))
    val_engines = set(engines[perm[:n_val]].tolist())
    val_mask = np.array([e in val_engines for e in ids])
    return ~val_mask, val_mask, n_val


def build_config(variant):
    cfg = Config()
    cfg.EPOCHS = EPOCHS
    if variant == 'seq50':
        cfg.SEQUENCE_LENGTH = 50
    return cfg


def prepare_windows(train_df_full, variant):
    """Scaled df -> (X, y, engine_ids, cfg) for the variant, mirroring
    targeted.py's sensors trick (drop after scaling; MinMax is per-feature).
    The returned cfg is the one the model must be built with: for 'sensors'
    its SENSOR_COLS/INPUT_DIM reflect the dropped columns."""
    df = train_df_full
    cfg = build_config(variant)
    if variant == 'sensors':
        df = df.drop(columns=DROP_SENSORS)
        cfg.SENSOR_COLS = [s for s in cfg.SENSOR_COLS if s not in DROP_SENSORS]
    X, y, ids = create_sequences_with_ids(df, cfg)
    return X, y, ids, cfg


def run_one(variant, seed, train_df_full, epochs=EPOCHS):
    set_seed(seed)
    X, y, ids, cfg = prepare_windows(train_df_full, variant)
    tr_mask, va_mask, n_val_eng = engine_wise_split(ids, seed)

    train_loader = DataLoader(RULDataset(X[tr_mask], y[tr_mask]),
                              batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(RULDataset(X[va_mask], y[va_mask]),
                            batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    model = create_model(
        input_dim=cfg.INPUT_DIM, d_model=cfg.D_MODEL, nhead=cfg.N_HEADS,
        num_layers=cfg.N_ENCODER_LAYERS, d_ff=cfg.D_FF, dropout=cfg.DROPOUT,
        device=cfg.DEVICE,
    )
    if variant == 'no_pe':
        model.positional_encoding = torch.nn.Identity()

    save_dir = os.path.join(OUT_ROOT, f'{variant}_seed{seed}')
    trainer = Trainer(model, cfg, save_dir=save_dir)
    trainer.train(train_loader, val_loader, epochs=epochs, verbose=False)

    rmse = trainer.history['val_rmse']
    result = {
        'variant': variant,
        'seed': seed,
        'split': 'engine-wise',
        'epochs_run': len(rmse),
        'epoch_budget': epochs,
        'n_train_windows': int(tr_mask.sum()),
        'n_val_windows': int(va_mask.sum()),
        'n_val_engines': n_val_eng,
        'leak_free': True,
        'best_val_rmse_cycles': float(min(rmse)),
        'mean_last5_val_rmse_cycles': float(np.mean(rmse[-5:])),
        'best_epoch': int(trainer.best_epoch),
        'note': 'validation-only; test untouched; engine-wise split (no window leak)',
    }
    with open(os.path.join(save_dir, 'val_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def rebuild_summary():
    """Scan per-run val_result.json files and rewrite summary.json.
    Safe with parallel processes: each run owns its directory, and the
    summary is a pure function of the files on disk."""
    all_results = []
    for variant in VARIANTS:
        for seed in SEEDS:
            p = os.path.join(OUT_ROOT, f'{variant}_seed{seed}', 'val_result.json')
            if os.path.exists(p):
                with open(p, encoding='utf-8') as f:
                    all_results.append(json.load(f))
    with open(os.path.join(OUT_ROOT, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true', help='1-epoch timing probe (control, seed 42)')
    ap.add_argument('--runs', type=str, default=None,
                    help='comma-separated variant:seed subset, e.g. control:42,no_pe:42')
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    prep = CMAPSSPreprocessor(Config())
    t0 = time.time()
    train_df_full = prep.prepare_train_data()
    n_engines = train_df_full[Config().ID_COL].nunique()
    print(f'preprocessed train df: {len(train_df_full)} rows, {n_engines} engines '
          f'({time.time() - t0:.0f}s)', flush=True)

    if args.probe:
        t1 = time.time()
        r = run_one('control', 42, train_df_full, epochs=1)
        dt = time.time() - t1
        print(json.dumps(r, indent=2))
        est = dt * sum(EPOCHS for _ in range(len(VARIANTS) * len(SEEDS)))
        print(f'PROBE done: 1 epoch = {dt:.0f}s  -> full {len(VARIANTS) * len(SEEDS)}-run '
              f'matrix at {EPOCHS} epochs ~ {est / 3600:.1f} h sequential '
              f'(seq50 runs a bit longer; early stopping may cut some)')
        return

    if args.runs:
        pairs = []
        for token in args.runs.split(','):
            variant, seed = token.strip().split(':')
            assert variant in VARIANTS and int(seed) in SEEDS, f'bad run token {token!r}'
            pairs.append((variant, int(seed)))
    else:
        pairs = [(v, s) for s in SEEDS for v in VARIANTS]

    for variant, seed in pairs:
        t1 = time.time()
        r = run_one(variant, seed, train_df_full)
        r['wall_seconds'] = round(time.time() - t1, 1)
        rebuild_summary()
        print(f"[done] {variant} seed{seed}: best val RMSE {r['best_val_rmse_cycles']:.2f} "
              f"(last-5 mean {r['mean_last5_val_rmse_cycles']:.2f}) "
              f"in {r['wall_seconds']:.0f}s", flush=True)

    print(f'\n=== ran {len(pairs)} run(s); {len(rebuild_summary())}/12 val_result.json present ===')


if __name__ == '__main__':
    main()
