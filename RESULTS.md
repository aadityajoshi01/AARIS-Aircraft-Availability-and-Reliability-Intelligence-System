# NASA C-MAPSS Transformer RUL — Failure Analysis, Fixes, and Verified Results

Date: 2026-10-10/11 (all runs on this machine, CPU, `.venv` Python 3.14)

---

## 1. Original failure

- Symptom: FD001 test RMSE **55.30**, predictions constant at **39.03 cycles** for all 100 test engines (`outputs/predictions.csv` of the broken run, preserved in `outputs_backup/`).
- `checkpoint_best.pt` was saved at epoch 3 of a 5-epoch run whose training history shows loss **rising** (4250 -> 4660) and flatlining; validation RMSE never went below ~57.

### Root cause (verified, not assumed)

1. **RUL labels were unscaled** (0-125 cycles) while features were MinMax-scaled. Initial gradient norms measured ~1600, so `clip_grad_norm_(..., max_norm=1.0)` rescaled every gradient by ~1/1600 each step.
2. **Asymmetric loss with `LAMBDA_OVER = 10`** makes a low constant the optimal degenerate solution: numerically, `argmin_c E[(c - y)^2 + 10*relu(c - y)^2] = 38` over the FD001 training-window label distribution (mean 80.4). The collapsed model output 39.03 - exactly this basin. Inside the basin, encoder gradients measured ~0 (`input_projection` grad norm effectively zero), so the 5-epoch run could not escape.
3. **Window/target misalignment**: training windows (rows i..i+29) were paired with the RUL of row **i+30**, so no training window ever contained the engine's final cycle, while every test window ends on it (train/test distribution mismatch).
4. Checkpoint loading, scaler fitting order (fit on train, transform test), and NaN state were checked and found correct - the model was genuinely collapsed, not mis-loaded.

## 2. Fixes applied

| File | Change |
|---|---|
| `src/data/preprocessing.py` | Sequence labels normalized by `MAX_RUL` (targets in [0, 1]); target = RUL of the **last row inside** the window (`i + seq_length - 1`), window count fixed to `n - seq_length + 1` |
| `src/train.py` | Validation RMSE reported in cycle units (x `MAX_RUL`) |
| `src/evaluate.py` | Model output denormalized (x `MAX_RUL`) and clipped to [0, 125] before metrics |
| `main.py` | Same denormalization for the visualization path |

No changes to model architecture, loss, optimizer, or checkpointing.

## 3. Evaluation integrity audit (post-fix)

Verified by an independent script (rebuilding sequences and recomputing metrics from `checkpoint_best.pt`):

- Training targets in [0.0, 1.0]; window alignment holds for **all** engines (X rows i..i+29, target = normalized RUL of row i+29); 17,731 windows total.
- Test: engine IDs 1..100 (file order) pair 1:1 with the 100 rows of `RUL_FD001.txt`; all 100 test sequences end exactly on their engine's final cycle (checked in raw units via scaler inverse-transform).
- Denormalization: raw outputs in [0.034, 0.955] -> [4.3, 119.4] cycles; the [0, 125] clip was active on **0** predictions for the seed-42 model.
- Recomputed test RMSE = 18.76, identical to `outputs/predictions.csv`.

## 4. Dataset caveat (non-standard FD001)

The local `Dataset/` is **not** the standard NASA release:

- `test_FD001.txt`: 100 engines, 13,096 records (standard public FD001 test = 21 engines)
- `RUL_FD001.txt`: 100 rows, values up to **145** - i.e. ground-truth RULs are **uncapped**, while training targets are capped at `MAX_RUL = 125`
- Consequence: 11/100 test engines have truth > 125 which a [0, 125]-clipped predictor can never fit; this adds a floor to achievable RMSE (~small but nonzero) and affects all models equally. The old "21 vs 100" confusion in reports was purely this difference.

## 5. Verified performance

### FD001 test (100 engines, same split, same metric code for every row)

| Model | Test RMSE | MAE | NASA Score | Over-rate |
|---|---|---|---|---|
| Constant-38 predictor (old collapsed model's behavior) | 55.99 | 47.76 | 257,838 | - |
| Constant-mean predictor | 41.89 | 36.14 | 7,391 | - |
| Ridge on flattened 30-cycle window | **16.22** | 13.21 | 444 | - |
| 1-layer LSTM (h=64, 30 epochs) | 17.65 | 12.54 | 973 | - |
| Transformer, seed 42 (main run, best epoch 48/50) | 18.76 | 13.50 | 1,130 | 37% |
| Transformer, seed 43 (best epoch 50/50) | 19.21 | 14.69 | 819 | 42% |
| Transformer, seed 44 (best epoch 49/50) | **17.42** | 12.94 | 622 | 34% |

> Convention note: the "NASA Score" column above uses the repo's **as-shipped** metric, which §8 shows had the 13/10 penalty scales swapped. The audited PHM08-standard scores are in §8 (Ridge 444 -> **413 corrected**; seed 42 1,130 -> 544). Use the corrected scores for any external claim.

### Repeatability (three Transformer seeds, identical split `random_state=42`, 50 epochs each)

- Test RMSE: 18.76, 19.21, 17.42 -> **mean 18.46, std 0.76** (range 17.42-19.21)
- Test MAE: mean 13.71, std 0.73
- NASA score: mean 856.9, std 209.1
- No seed collapsed; all three show monotonically decreasing train/val loss (final train loss 0.013-0.015 in normalized units vs the broken run's flatlined 4600+)

### RUL cap: consistent capped-truth evaluation

Predictions are capped at `MAX_RUL = 125` but this dataset's ground truth is uncapped (up to 145; 11/100 engines above 125), so the primary numbers above mix a capped predictor with uncapped labels. Re-scoring the same three checkpoints against **consistently capped truth** (`min(RUL, 125)`, the standard PHM08 convention) - stored in `results/capped_truth_metrics.json`:

| Seed | RMSE (uncapped truth) | RMSE (capped truth) | MAE (capped) | NASA (capped) |
|---|---|---|---|---|
| 42 | 18.76 | 17.00 | 12.43 | 708 |
| 43 | 19.21 | 18.06 | 13.62 | 661 |
| 44 | 17.42 | 16.00 | 11.87 | 487 |
| **mean +/- std** | **18.46 +/- 0.76** | **17.02 +/- 0.84** | 12.64 | 619 |

Choice documented: keep `MAX_RUL = 125` with capped-truth as the headline-consistent metric; the uncapped-truth numbers are reported alongside because that is what the shipped `main.py --mode evaluate` prints. The ~1.4-cycle gap is the structural penalty of the label/truth cap mismatch, not model error. Under the same capped-truth rescore, Ridge gets **14.88** (MAE 12.14) - the Transformer does not beat it under either convention.

### What improved

- Test RMSE **55.30 -> 18.46 +/- 0.76** (mean over 3 seeds): 3.0x reduction, verified on the same 100 test engines.
- Predictions went from 1 unique value (39.03) to full spread (mean ~71, std ~38, 94 unique values across 100 engines; corr with truth 0.899 for the seed-42 model).

## 6. Reproduce / retrain

```bash
# from repo root
.venv/Scripts/python.exe main.py --mode train --epochs 50
.venv/Scripts/python.exe main.py --mode evaluate
.venv/Scripts/python.exe main.py --mode all          # + plots
OMP_NUM_THREADS=6 .venv/Scripts/python.exe experiments/run_seed.py --seed 43
OMP_NUM_THREADS=3 .venv/Scripts/python.exe experiments/baselines.py
```

On Windows with piped stdout, set `PYTHONIOENCODING=utf-8` (the box-drawing header otherwise crashes cp1252).

## 7. Artifacts

- `results/frozen_baseline_2026-10-11/` - **frozen reproducible baseline** (`frozen_baseline_2026-10-11.zip`, sha256 `7643a1a0b8139ae9...`, 116 files): full source at freeze time, Dataset/, all run artifacts, RESULTS.md, `environment.json` (python/torch/numpy/sklearn/pandas versions + headline metrics), `SHA256MANIFEST.txt`. Frozen **before** the NASA-score fix, so it preserves the code as-experimented-on.
- `results/run_seed42_main/` - best checkpoint, config.py, requirements.txt, training_history.json, predictions.csv, logs, regenerated plots
- `results/seed_43/`, `results/seed_44/` - per-seed checkpoints, histories, `metrics.json`
- `results/baselines.json`, `results/capped_truth_metrics.json`, `results/nasa_audit.json`
- `outputs_backup/` - the original broken artifacts (deletable once satisfied)

## 8. NASA score audit (post-freeze fix)

The shipped `NASAScore` (src/models/loss.py) **had the penalty scales swapped** relative to the PHM08 standard (Saxena et al. 2008): it applied the mild /13 scale to late/overestimating errors and the steep /10 scale to early/underestimating ones, punishing safety-conservative models instead of dangerous optimism. Fixed after the freeze; verified two ways: unit check (equal-magnitude over- vs under-error: over now scores higher) and end-to-end re-evaluation (seed-42 score 1,130 -> 543.96, matching an independent standard-convention recompute; `experiments/nasa_audit.py`, `results/nasa_audit.json`).

Both scores side by side (uncapped truth, same predictions):

| Model | RMSE | over-rate | score (old, wrong) | score (standard) |
|---|---|---|---|---|
| Constant-mean | 41.89 | 45% | 7,391 | 19,054 |
| Constant-38 | 55.99 | 27% | 257,838 | 30,201 |
| Ridge | 16.22 | 54% | 444 | 413 |
| LSTM (30 ep) | 17.65* | 29% | 973 | 425 |
| Transformer seed 42 | 18.76 | 37% | 1,130 | 544 |
| Transformer seed 43 | 19.21 | 42% | 819 | 800 |
| Transformer seed 44 | 17.42 | 34% | 622 | 608 |

Implications: the old score distorted comparisons (it inflated conservative models ~2-8x and made the constant-38 model look worst of all when under the standard it beats constant-mean). With the corrected metric, standard-score ranking is Ridge (413) ~ LSTM (425) < Transformer (544-800). RMSE and NASA agree on Ridge leadership; the Transformer remains clearly better than constants but has no evidence of superiority over simple baselines on this dataset variant.

## 9. Limitations and honest claims

1. **The Transformer does not currently beat a simple baseline.** Ridge regression on flattened windows gets RMSE 16.22 vs the Transformer's 18.46 +/- 0.76 (3 seeds). The Transformer's demonstrable win is over the *collapsed state* (55.3 -> 18.5) and it matches/exceeds the LSTM (17.65), but no evidence supports "Transformer > regression" on this dataset variant. Tuning knobs if pursued: larger `MAX_RUL`, target dropout, warmup + cosine LR, pooling on the last timestep instead of mean, or feature selection (5 near-constant sensors are standard removals in C-MAPSS work).
2. **RUL cap policy (125) was inconsistent with this dataset's ground truth.** Training labels are piece-wise capped at 125 and predictions are clipped to [0, 125], but 11 of 100 test engines have uncapped true RUL up to 145. Resolved by evaluation, not retraining: both conventions are now reported (Section 5) - RMSE 18.46 +/- 0.76 against the shipped uncapped truth, 17.02 +/- 0.84 against consistently capped truth. The training-side setup (`MAX_RUL = 125` piece-wise cap) was left unchanged for comparability; retraining with a raised cap (e.g. 145) is a reasonable follow-up but is not covered by any measurement here. Ridge was also rescored under capped truth (14.88), so the head-to-head is fair under both conventions.
3. **Non-standard dataset** (Section 4): results on this 100-engine FD001 variant are not comparable to published 21-engine FD001 numbers (SOTA RMSE ~11-14 on the standard split).
4. **Baseline fairness:** Ridge and LSTM use the identical split, scaler, test windows, and metric code; the LSTM had fewer epochs (30 vs 50), so its RMSE 17.65 is a mild handicap; the constant predictors are exact by construction.
5. NASA score is exponential and high-variance by nature; corrected-convention per-seed spread (544-800) is dominated by a few engines, not systematic model differences.
6. The mechanism of the original collapse (why the old 5-epoch run diverged from epoch 1) could not be fully reproduced with the fixed pipeline even under the old unscaled-labels configuration in a 6-epoch test - the historical artifacts show loss *rising* 4250 -> 4660, i.e. the run fell into the flat constant-39 basin early and had no epochs left to escape. The fixes remove the basin's attractor strength (normalized labels, O(1) gradients, clipping now meaningful) and the measured escape behavior confirms it.
