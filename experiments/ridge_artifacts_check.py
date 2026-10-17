"""Verify the frozen Ridge artifacts reproduce the pipeline end-to-end.

Checks, using ONLY files in results/ridge_model/:
1. scaler.pkl reproduces the pipeline's scaled test windows (float32 tolerance).
2. ridge_model.pkl + scaler.pkl recompute the full FD001 test RMSE (~16.22 uncapped).
3. Writes sample_input.json (engine 1's last 30 raw cycles) for standalone testing.

Usage (repo root):
    .venv/Scripts/python.exe experiments/ridge_artifacts_check.py
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import Config
from data.preprocessing import CMAPSSPreprocessor

ART = os.path.join(ROOT, 'results', 'ridge_model')


def main():
    cfg = Config()
    cols = cfg.FEATURE_COLS

    # --- load frozen artifacts ---
    with open(os.path.join(ART, 'ridge_model.pkl'), 'rb') as f:
        ridge = pickle.load(f)
    with open(os.path.join(ART, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(ART, 'feature_order.json'), 'r') as f:
        meta = json.load(f)
    assert meta['feature_order'] == cols, "feature order mismatch"
    seq = meta['sequence_length']
    max_rul = float(meta['max_rul'])

    # --- pipeline: fit scaler on train, produce scaled test windows ---
    prep = CMAPSSPreprocessor(cfg)
    prep.prepare_train_data()
    test_df_scaled, true_rul = prep.prepare_test_data()

    # raw (unscaled) test file, same loader
    raw_df = prep.load_data(cfg.TEST_PATH)

    # per-engine last-30 windows, raw and scaled, in pipeline engine order
    engines = test_df_scaled[cfg.ID_COL].unique()
    raw_windows, scaled_pipeline = [], []
    for eid in engines:
        e_raw = raw_df[raw_df[cfg.ID_COL] == eid]
        e_sc = test_df_scaled[test_df_scaled[cfg.ID_COL] == eid]
        if len(e_sc) >= seq:
            raw_windows.append(e_raw[cols].values[-seq:])
            scaled_pipeline.append(e_sc[cols].values[-seq:])
    raw_windows = np.array(raw_windows, dtype=np.float64)
    scaled_pipeline = np.array(scaled_pipeline, dtype=np.float32)
    true_rul = np.asarray(true_rul)[:len(raw_windows)]
    assert len(raw_windows) == 100, f"expected 100 test engines, got {len(raw_windows)}"

    # --- check 1: scaler.pkl on raw windows == pipeline scaled windows ---
    X_scaled_standalone = np.empty_like(scaled_pipeline)
    for i, w in enumerate(raw_windows):
        X_scaled_standalone[i] = scaler.transform(pd.DataFrame(w, columns=cols))
    max_diff = float(np.abs(X_scaled_standalone - scaled_pipeline).max())
    print(f"[1] scaler.pkl round-trip max abs diff vs pipeline: {max_diff:.2e}")
    assert max_diff < 1e-5, "scaler.pkl does not reproduce pipeline scaling"

    # --- check 2: full test RMSE from artifacts alone ---
    preds_norm = ridge.predict(X_scaled_standalone.reshape(len(X_scaled_standalone), -1))
    preds = np.clip(preds_norm * max_rul, 0.0, max_rul)
    rmse_uncapped = float(np.sqrt(np.mean((preds - true_rul) ** 2)))
    truth_capped = np.minimum(true_rul, max_rul)
    rmse_capped = float(np.sqrt(np.mean((preds - truth_capped) ** 2)))
    print(f"[2] engines scored: {len(preds)}")
    print(f"    RMSE (uncapped truth): {rmse_uncapped:.2f}  (expected ~{meta['expected_test_rmse_uncapped']})")
    print(f"    RMSE (capped truth):   {rmse_capped:.2f}  (expected ~{meta['expected_test_rmse_capped_truth']})")
    assert abs(rmse_uncapped - meta['expected_test_rmse_uncapped']) < 0.05
    assert abs(rmse_capped - meta['expected_test_rmse_capped_truth']) < 0.05

    # --- check 3: sample_input.json (engine 1, last 30 raw cycles) ---
    first_engine = int(engines[0])
    sample = {
        'metadata': {
            'description': 'Raw (unscaled) last 30 cycles of test engine 1; input to predict_rul(window).',
            'engine_id': first_engine,
            'sequence_length': seq,
            'feature_order': cols,
            'expected_rul_cycles': float(preds[0]),
            'true_rul_cycles': float(true_rul[0]),
        },
        'window': np.round(raw_windows[0], 6).tolist(),
    }
    sample_path = os.path.join(ART, 'sample_input.json')
    with open(sample_path, 'w') as f:
        json.dump(sample, f, indent=2)
    print(f"[3] wrote {sample_path} (engine {first_engine}: predicted {preds[0]:.2f}, true {true_rul[0]:.0f})")

    print("\nAll artifact checks passed.")


if __name__ == '__main__':
    main()
