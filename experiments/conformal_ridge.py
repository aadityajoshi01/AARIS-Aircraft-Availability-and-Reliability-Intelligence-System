"""P1: Split-conformal uncertainty calibration for the frozen Ridge RUL model.

What this produces
------------------
`conformal/risk_calibration.json` — a sidecar with 80/90/95% half-widths
(quantiles of absolute out-of-fold errors, in final clipped cycles) for the
FROZEN Ridge recipe: MinMaxScaler -> flatten 30x24 -> Ridge(alpha=10) ->
clip(pred * MAX_RUL, 0, MAX_RUL). The frozen artifacts themselves are never
read with write intent and never retrained.

Why cross-fit (and not the 20% window holdout)
----------------------------------------------
The frozen model was fitted on 80% of the *windows* of the training engines,
and windows are stride-1 overlapping (29/30 rows shared with neighbours), so
the remaining 20% are near-duplicates of training rows — calibrating there
would produce optimistically narrow bands. Instead we cross-fit: engine-wise
GroupKFold (k=5) over the 100 training engines; per fold we refit the scaler
on the fit engines' raw rows only (so held-out engines never influence
scaling stats), refit the same-recipe Ridge, and score the held-out engines'
windows. Residuals therefore come from *unseen engines*, matching deployment
(test engines / real aircraft are never in training).

Honesty notes (also written into the JSON)
------------------------------------------
- Cross-conformal coverage is a heuristic guarantee, not an exact finite-
  sample guarantee: it needs exchangeability across engine folds, which is
  an approximation (engines differ in life length and degradation rate).
- Test set (test_FD001/RUL_FD001) is NOT touched here.
- Point-estimate risk bands (LOW/MEDIUM/HIGH/CRITICAL) are unchanged; this
  sidecar only adds interval support around them.

Usage (repo root):
    .venv/Scripts/python.exe experiments/conformal_ridge.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import Config  # noqa: E402
from data.preprocessing import CMAPSSPreprocessor  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'conformal')
OUT_PATH = os.path.join(OUT_DIR, 'risk_calibration.json')

K_FOLDS = 5
SPLIT_SEED = 42          # deterministic fold assignment
LEVELS = [0.80, 0.90, 0.95]


def build_windows(df, cfg, cols):
    """Same recipe as preprocessing.create_sequences (stride-1, target = last
    row's capped RUL normalized by MAX_RUL), but also returns the engine id of
    each window. df rows must already be in per-engine cycle order."""
    seq = cfg.SEQUENCE_LENGTH
    X, y, ids = [], [], []
    for eid in df[cfg.ID_COL].unique():
        e = df[df[cfg.ID_COL] == eid].reset_index(drop=True)
        vals = e[cols].values
        rul = e['RUL'].values
        for i in range(len(e) - seq + 1):
            X.append(vals[i:i + seq])
            y.append(rul[i + seq - 1] / cfg.MAX_RUL)
            ids.append(eid)
    return (np.array(X, dtype=np.float64), np.array(y, dtype=np.float64),
            np.array(ids))


def conformal_q(abs_res: np.ndarray, coverage: float) -> float:
    """Split-conformal quantile level min(1, ceil((n+1)*coverage)/n) with the
    'higher' interpolation — the standard finite-sample-safe choice."""
    n = len(abs_res)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(abs_res, level, method="higher"))


def main():
    t0 = time.time()
    cfg = Config()
    cols = cfg.FEATURE_COLS
    max_rul = float(cfg.MAX_RUL)
    alpha = 10.0

    # Raw (UNSCALED) train df with capped RUL — scaling happens per fold.
    prep = CMAPSSPreprocessor(cfg)
    raw = prep.load_data(cfg.TRAIN_PATH)
    raw = prep.compute_rul(raw)
    engines = np.sort(raw[cfg.ID_COL].unique())
    assert len(engines) == 100, f"expected 100 training engines, got {len(engines)}"

    # Deterministic engine-wise folds (seeded permutation, round-robin).
    rs = np.random.RandomState(SPLIT_SEED)
    perm = rs.permutation(len(engines))
    fold_of = {}
    for rank, idx in enumerate(perm):
        fold_of[engines[idx]] = rank % K_FOLDS
    fold_id = raw[cfg.ID_COL].map(fold_of).values

    print(f"[conformal] {len(raw)} raw rows, {len(engines)} engines, "
          f"k={K_FOLDS} engine-wise folds, seed={SPLIT_SEED}", flush=True)

    all_abs, all_signed, all_pred, all_true = [], [], [], []
    fold_stats = []
    for k in range(K_FOLDS):
        fit_mask = fold_id != k
        hold_mask = fold_id == k
        # Refit the scaler on fit engines' ROWS only (no holdout influence).
        scaler = MinMaxScaler().fit(raw.loc[fit_mask, cols])
        df_fit = raw.loc[fit_mask].copy()
        df_hold = raw.loc[hold_mask].copy()
        df_fit[cols] = scaler.transform(df_fit[cols])
        df_hold[cols] = scaler.transform(df_hold[cols])

        X_fit, y_fit, _ = build_windows(df_fit, cfg, cols)
        X_hold, y_hold, _ = build_windows(df_hold, cfg, cols)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_fit.reshape(len(X_fit), -1), y_fit)
        pred_norm = ridge.predict(X_hold.reshape(len(X_hold), -1))
        pred_cy = np.clip(pred_norm * max_rul, 0.0, max_rul)   # deployment scale
        true_cy = y_hold * max_rul                              # capped <= 125

        res = pred_cy - true_cy
        all_abs.append(np.abs(res))
        all_signed.append(res)
        all_pred.append(pred_cy)
        all_true.append(true_cy)
        fold_stats.append({
            'fold': k,
            'n_holdout_engines': int(df_hold[cfg.ID_COL].nunique()),
            'n_holdout_windows': int(len(pred_cy)),
            'fold_rmse_cycles': float(np.sqrt(np.mean(res ** 2))),
            'fold_mae_cycles': float(np.mean(np.abs(res))),
        })
        print(f"[conformal] fold {k}: {fold_stats[-1]['n_holdout_windows']} OOF windows, "
              f"RMSE {fold_stats[-1]['fold_rmse_cycles']:.2f} cycles", flush=True)

    abs_res = np.concatenate(all_abs)
    signed_res = np.concatenate(all_signed)
    pred_all = np.concatenate(all_pred)
    true_all = np.concatenate(all_true)
    n = len(abs_res)

    halfwidths = {}
    coverage = {}
    for lvl in LEVELS:
        q = conformal_q(abs_res, lvl)
        halfwidths[f"{int(lvl * 100)}"] = round(q, 2)
        coverage[f"{int(lvl * 100)}"] = round(float(np.mean(abs_res <= q)), 4)

    # Per-prediction-band diagnostics (honesty: does one global width behave
    # differently where the risk bands cut?). Band = point-estimate band.
    def band(p):
        return ('CRITICAL' if p < 30 else 'HIGH' if p < 60 else
                'MEDIUM' if p < 100 else 'LOW')

    band_diag = {}
    for name in ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'):
        m = np.array([band(p) == name for p in pred_all])
        if m.sum() == 0:
            continue
        hw90 = halfwidths['90']
        lower = np.clip(pred_all[m] - hw90, 0.0, max_rul)
        upper = np.clip(pred_all[m] + hw90, 0.0, max_rul)
        cov = float(np.mean((true_all[m] >= lower) & (true_all[m] <= upper)))
        band_diag[name] = {
            'n_windows': int(m.sum()),
            'mean_abs_resid_cycles': round(float(np.mean(abs_res[m])), 2),
            'empirical_coverage_90_clipped': round(cov, 4),
        }

    # Grounding-decision diagnostic: among windows whose point estimate says
    # CRITICAL (<30), how often does the 90% upper bound still exceed 30?
    crit = pred_all < 30.0
    hw90 = halfwidths['90']
    crit_upper_above_30 = float(np.mean((pred_all[crit] + hw90) >= 30.0)) if crit.any() else None
    crit_true_below_30 = float(np.mean(true_all[crit] < 30.0)) if crit.any() else None

    result = {
        'model': 'Ridge(alpha=10) on flattened 30x24 MinMax windows (frozen recipe)',
        'target_scale': 'final clipped cycles: clip(pred_norm*125, 0, 125); truth = piecewise-capped RUL (<=125)',
        'method': 'cross-conformal regression: engine-wise GroupKFold cross-fit, absolute out-of-fold residuals',
        'k_folds': K_FOLDS,
        'fold_split_seed': SPLIT_SEED,
        'per_fold_scaler_refit': True,
        'n_calibration_windows': n,
        'n_calibration_engines': 100,
        'oof_rmse_cycles': round(float(np.sqrt(np.mean(signed_res ** 2))), 2),
        'oof_mae_cycles': round(float(np.mean(abs_res)), 2),
        'residual_quantiles_abs_cycles': {
            'median': round(float(np.median(abs_res)), 2),
            'p75': round(float(np.quantile(abs_res, 0.75, method='higher')), 2),
            'p90': round(float(np.quantile(abs_res, 0.90, method='higher')), 2),
            'p95': round(float(np.quantile(abs_res, 0.95, method='higher')), 2),
            'max': round(float(abs_res.max()), 2),
        },
        'halfwidths_cycles': halfwidths,
        'empirical_coverage_oof': coverage,
        'band_diagnostics_90': band_diag,
        'grounding_diagnostics': {
            'n_windows_point_critical': int(crit.sum()),
            'frac_90pct_upper_bound_still_ge_30': crit_upper_above_30,
            'frac_true_below_30_among_point_critical': crit_true_below_30,
        },
        'folds': fold_stats,
        'test_set_touched': False,
        'caveats': [
            'cross-conformal coverage is a heuristic (needs approximate exchangeability across engine folds), not an exact finite-sample guarantee',
            'calibrated on training engines (FD001); applies to same-distribution data only',
            'half-widths are global; band_diagnostics_90 shows where that is conservative or thin',
            'point-estimate risk bands (LOW>=100, MEDIUM 60-99, HIGH 30-59, CRITICAL<30) are unchanged; this adds intervals around them',
        ],
        'calibrated_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'source_script': 'experiments/conformal_ridge.py',
        'elapsed_seconds': round(time.time() - t0, 1),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\n[conformal] wrote {OUT_PATH}", flush=True)
    print(json.dumps({k: result[k] for k in
                      ('oof_rmse_cycles', 'oof_mae_cycles', 'halfwidths_cycles',
                       'empirical_coverage_oof')}, indent=2))


if __name__ == '__main__':
    main()
