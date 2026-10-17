# Conformal risk calibration (sidecar)

`risk_calibration.json` gives honest uncertainty intervals for the **frozen**
Ridge RUL model (`Ridge(alpha=10)` on flattened 30×24 MinMax windows,
clip `[0, 125]`). It is produced by:

```bash
# from the repo root
.venv/Scripts/python.exe experiments/conformal_ridge.py   # ~15 s, CPU-only
```

## How it is calibrated

- **Cross-conformal regression**: engine-wise GroupKFold (k=5) over the 100
  FD001 training engines. Per fold the scaler is refit on the fold-training
  engines only, the same-recipe Ridge is refit, and the held-out engines'
  windows are scored — so every residual comes from a *never-seen engine*,
  matching deployment (test/real aircraft are never in training).
- Residuals live in the **final clipped-cycles scale** (`clip(pred·125, 0, 125)`),
  the same scale as `/predict` output and the risk bands.
- Half-widths are split-conformal quantiles of `|residual|` at
  `min(1, ceil((n+1)·coverage)/n)` with `method='higher'`, for 80/90/95%.

Current values (n=17,731 out-of-fold windows): half-widths
**21.81 / 27.19 / 31.32 cycles**, empirical OOF coverage 0.800 / 0.900 / 0.950.
OOF RMSE 16.41 cycles sits next to the frozen model's 16.22 test RMSE — the
calibration error scale tracks held-out test error closely.

## Who consumes it

- `edge/edge_node.py` `/predict` adds an `uncertainty` block (90% interval +
  grounding-threshold set-membership statement). Point estimates and risk
  bands are unchanged; the block degrades gracefully if this file is absent.
- `dashboard/app.py` *Aircraft Detail* renders the interval from the same file.

## Honesty caveats (also inside the JSON)

- Cross-conformal coverage is a **heuristic** (needs approximate
  exchangeability across engine folds), not an exact finite-sample guarantee.
- Global half-widths behave unevenly by band (see `band_diagnostics_90`):
  coverage is ≈97–100% for LOW/HIGH/CRITICAL but only **76% for MEDIUM** —
  the dashboard surface states this per band.
- Calibrated on FD001 training engines only; applies to same-distribution
  data. The test set (`test_FD001/RUL_FD001`) was not touched by calibration.

## Regenerating

Re-running the script overwrites `risk_calibration.json` deterministically
(fold seed 42, `Ridge(alpha=10)`, no sampling). Nothing else in the repo
writes this file; consumers treat it as read-only.
