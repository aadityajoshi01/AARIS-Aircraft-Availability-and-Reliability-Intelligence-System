"""Standalone Ridge RUL predictor — no training code or dataset needed.

Usage:
    from predict_rul import predict_rul
    rul_cycles = predict_rul(window)   # window: list of 30 rows x 24 raw feature values

CLI test:
    python predict_rul.py sample_input.json
"""

import json
import os

import numpy as np
import pandas as pd
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_artifacts():
    with open(os.path.join(HERE, 'ridge_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(HERE, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(HERE, 'feature_order.json'), 'r') as f:
        meta = json.load(f)
    return model, scaler, meta


def predict_rul(window):
    """Predict RUL in cycles from one raw (un-normalized) 30-cycle window.

    window: array-like of shape [sequence_length, 24], columns ordered per
    feature_order.json (op_setting_1..3, sensor_1..21).
    """
    model, scaler, meta = _load_artifacts()
    cols = meta['feature_order']
    seq_len = meta['sequence_length']
    max_rul = meta['max_rul']

    arr = np.asarray(window, dtype=np.float64)
    if arr.shape != (seq_len, len(cols)):
        raise ValueError(f"Expected window shape ({seq_len}, {len(cols)}), got {arr.shape}")

    scaled = scaler.transform(pd.DataFrame(arr, columns=cols))
    pred_norm = model.predict(scaled.reshape(1, -1))[0]
    return float(np.clip(pred_norm * max_rul, 0.0, max_rul))


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'sample_input.json')
    with open(path) as f:
        sample = json.load(f)
    meta = sample.get('metadata', {})
    print(f"Engine: {meta.get('engine_id')}  true RUL: {meta.get('true_rul_cycles')} cycles  "
          f"(expected prediction: {meta.get('expected_rul_cycles')})")
    print(f"Predicted RUL: {predict_rul(sample['window']):.2f} cycles")
