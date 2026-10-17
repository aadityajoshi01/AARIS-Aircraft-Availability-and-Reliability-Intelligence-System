# AARIS — Project Audit & Improvement Opportunities

Date: 2026-10-11. Scope: read-only audit of this repo (services, ML protocol, ops) plus a web survey of open-source C-MAPSS RUL projects and tooling. **Nothing in this file has been implemented yet** — no code, models, artifacts, or services were changed by this audit. Severity is graded honestly for what this is: a hackathon-grade verified prototype, not production software.

Method: three independent passes — (A) engineering audit of `edge/`, `ground/`, `dashboard/`, `demo/`, launchers, requirements; (B) ML-protocol audit of `RESULTS.md`, `src/`, `experiments/`, `results/*.json`; (C) external survey of GitHub projects and libraries. Every internal finding cites file:line; every external claim links its source and is labeled *reported by authors, not independently verified*. Critical findings were re-verified first-hand before inclusion.

---

## 0. Headline finding (fix before spending the reserved test-set evaluation)

**The validation split leaks, and the `no_pe` decision currently rests on it.**

Evidence, verified first-hand:

- `src/data/preprocessing.py:250` builds **stride-1 sliding windows** per engine — consecutive windows from one engine share 29 of 30 rows.
- `src/data/dataset.py:91-95` splits **windows** (not engines) with plain `train_test_split(..., random_state=42)`, so near-duplicate windows from the same engine land in both train and validation.
- Consequence: every validation number is inflated — consistent with val RMSE 4.92–10.48 vs test RMSE 16–19 for the same model family.
- `experiments/targeted.py:50` pins a single seed (`random/np/torch = 42`), so `no_pe` 4.92 vs control 10.48 is **one seed on a leaky metric**. The delta is as plausibly leakage noise as a real effect (removing positional encoding *helping* is already counter-intuitive).

What this does NOT affect: the test-set numbers in `RESULTS.md` §5 — the test set is built from each *test* engine's final window (`preprocessing.py:285-286`), and test engines are fully disjoint from train engines. Ridge 16.22/14.88, Transformer 17.42–19.21, the NASA audit, and the deployed frozen model all stand.

**Protocol implication:** the single reserved `no_pe` test evaluation should wait. The honest order is: (1) rebuild the split engine-wise (GroupShuffleSplit on engine ID), (2) re-run control + `no_pe` + `seq50` + `sensors` × 3 seeds under the matched 30-epoch budget (validation-only, no test touch), (3) only then spend the one test evaluation on whatever survives. This converts a protocol flaw into a demonstrated methodology fix — a stronger story than the original no_pe claim either way.

> **STATUS 2026-10-11: RESOLVED — steps (1) and (2) are done.** `experiments/engine_split.py` re-ran all 4 configs × 3 seeds on an engine-wise split (20% of engines held out whole; the seed also re-draws the split; 12/12 runs, every `val_result.json` marked `"leak_free": true`; outputs in `results/engine_split/`). Outcome: **`no_pe` is worse than control at every seed** (best val RMSE 16.94/19.43/21.17 vs 13.30/18.19/13.63) — the 4.92 was pure leak artifact, and step (3) must **not** spend the reserved test run on it. **`seq50` beats control at every seed** (12.59/15.01/11.46) and is the only surviving candidate. `sensors` is borderline (better on last-5 mean at all seeds, loses seed-44 best). Full table and wording in `docs/PROJECT_REPORT.md` §5. All figures remain validation-only; the test set is still untouched.

---

## 1. Engineering audit — services stack

### 1.1 Robustness / error handling

| # | Sev | Finding | Evidence | Fix | Effort |
|---|---|---|---|---|---|
| R1 | M | SQLite has no WAL or busy timeout; concurrent `POST /ingest_report` + `POST /schedule` can hit `database is locked` (new connection per request, default 5 s timeout) | `ground_station.py:74-78` | `PRAGMA journal_mode=WAL` + `timeout=` in `_connect()` | S |
| R2 | M | `/schedule` catches only `SchedulerError`; a sqlite/read error inside `load_fleet_state` returns an unhandled 500 with traceback to client | `ground_station.py:422-429`, `scheduler.py:72-79` | broad try/except → 500 envelope | S |
| R3 | M | Dashboard re-hits the API on every Streamlit rerun (any widget click); `refresh_nonce` + `st.cache_data.clear()` exist but no fetch function is ever decorated — dead cache code | `app.py:114-127,151` | `@st.cache_data(ttl=5)` keyed on nonce, or drop the nonce | S |
| R4 | L | `_parse(stored)` raises → 500 if `latest_report_at` is ever hand-edited/corrupted | `ground_station.py:330` | wrap in try, treat unparseable as oldest | S |
| R5 | L | Edge `_STORE` is unbounded and in-process (lost on restart) | `edge_node.py:45,318-325` | acceptable for demo; cap size if kept | S |

### 1.2 Security / posture (framed for a local demo)

All three services bind `127.0.0.1` only in the launchers (`start.bat:21-27`, `run_demo.ps1:66-74`, `ground_station.py:512`) — genuinely good. Remaining items:

- **M (demo: L)** — no auth and no body-size cap; `/predict` reads `await request.json()` fully and validates cell-by-cell in Python loops (quadratic on attacker-controlled payload size). `edge_node.py:281,147-158`. Fix: reject `Content-Length > ~256 KB`. (S)
- **L** — pickle artifacts: SHA256s are *displayed* in `/health` but never *verified against a pinned manifest* at startup — the checksum is cosmetic. `edge_node.py:64-67,84-91`. Fix: expected-hashes file + startup abort. (S)
- **L** — `.gitignore` contains one notebook entry; `fleet.db`, `__pycache__`, `.pyc`, logs get committed. (S)
- No CORS config anywhere is actually fine (dashboard proxies server-side, no browser→API calls). No secrets in code found.

### 1.3 Testing / CI

- **M** — no pytest suite, no CI (confirmed: no `pytest.ini`, `pyproject.toml`, `.github/`, Dockerfile). Both smoke tests are well-designed but script-shaped (`assert` + `main()`, manual). Converting them verbatim to `test_*.py` + one GitHub Actions workflow (smoke + `verify_pipeline` against in-process TestClients) earns a real CI badge in about an hour. (M)
- **M** — untested paths: scheduler phase-2/3 fallbacks (only phase 1 exercised — force deferrals via `POST /schedule` body with tight `max_daily`), risk-band boundaries (99.9/100), `_sanitize_non_finite` handler (`ground_station.py:189-194`). (S)

### 1.4 Packaging / ops

- **M** — requirements fragmentation: root `requirements.txt` lacks FastAPI/uvicorn/ortools/streamlit/plotly/requests entirely; `ground/requirements.txt` covers ground only; edge and dashboard have none (dashboard deps exist only in README prose). `ground/requirements.txt:2` claims "Python 3.11+" while the venv is 3.14. Fix: per-service files or one root + `pip freeze` lockfile. (S)
- **M** — no logging/observability: only `print()` (`ground_station.py:153`, `edge_node.py:101`); no request IDs, nothing persisted for post-mortem. Fix: stdlib `logging` + request-ID middleware. (S/M)
- **L** — schema evolution: single `CREATE TABLE IF NOT EXISTS` script (`ground_station.py:81-121`); added columns would silently break an existing `fleet.db` (scheduler defensively probes columns via PRAGMA — good instinct, not a migration). `start.bat`'s fixed `ping /n 7` sleep races slow starts, unlike `run_demo.ps1`'s proper readiness poll. Ports/URLs hardcoded in 5 files.

### 1.5 Data contracts

- **M** — the edge response is hand-built dicts (`edge_node.py:296-315`) and the ground station re-declares a *mirror* Pydantic model (`ground_station.py:200-238`); drift is caught only by smoke tests. `MAX_RUL=125.0` duplicated in 3 files (`ground_station.py:44`, `scheduler.py:41`, `seed_demo.py:68`); risk-band thresholds duplicated (`edge_node.py:178-185` vs `seed_demo.py:56-63`). Reports carry **no schema/version field**. Fix: shared `contracts.py`, or emit `schema_version` + `max_rul` in the report and validate on ingest. (M)

### 1.6 Code smells (small, real)

Unused `Response` import (`ground_station.py:29`); `hashlib` imported inside a function (`edge_node.py:52-53`); `_sched_reason` prints "None risk" when risk is NULL but data present (`scheduler.py:180` — `load_fleet_state` doesn't exclude risk-NULL from candidates); if every aircraft has `base=NULL`, the dashboard's empty `base_filter` hides the whole fleet (`app.py:130-148`); `stop.bat:10-12` kills whatever owns 8000/8001/8501, including unrelated processes; POST /schedule's INSERT-then-UPDATE `plan_id` double-write is non-atomic (`ground_station.py:434-445` — use `RETURNING`).

### Already solid (do not "fix")

Edge 422 validation is unusually thorough (bool rejection, per-cell offending positions, non-finite scan — `edge_node.py:135-172`); the NaN-safe 422 handler is a genuine subtle catch; strictly-newer timestamp guard + full history retention is correct; CP-SAT determinism (1 worker, seed 7, verified by re-solve in `ground/smoke_test.py:133-137`) works; dashboard is strictly read-only with 5 s timeouts; `run_demo.ps1` readiness polling and `-Reset` are solid; `verify_pipeline.py` checks the right invariants (capacity, `sum(daily_load) == scheduled_count`).

### Quick wins (cheap, demo-visible)

1. WAL + busy timeout in `_connect()` — 2 lines, kills the flakiest real concurrency failure.
2. Content-Length cap on `/predict` — 3 lines.
3. Activate `@st.cache_data(ttl=5)` (or delete the dead nonce) — dashboard feels instant.
4. Smoke tests → pytest + one CI workflow — "CI" claim becomes true.
5. `schema_version` + `max_rul` in the report payload, validated at ingest — makes the edge↔ground contract a demo talking point instead of an implicit convention.

---

## 2. ML audit — prioritized opportunities (all additive; frozen artifacts untouched)

| Pri | Opportunity | Evidence | Expected benefit | Cost | Risk to frozen discipline |
|---|---|---|---|---|---|
| **P0** | **Engine-wise validation split + 3-seed re-run of the targeted experiments** (see §0) — ✅ **DONE 2026-10-11** (12/12 runs; `no_pe` eliminated, `seq50` the only survivor; §0 status note) | `dataset.py:91-95`, `preprocessing.py:250`, `targeted.py:50` | Protocol integrity; determines whether no_pe deserves its reserved test run at all | M | none — new experiment dirs only |
| **P1** | **Calibrated uncertainty via split-conformal prediction** on Ridge residuals — ✅ **DONE 2026-10-11** as additive sidecar (`conformal/risk_calibration.json`, ±27.19 c @90%; bands stay point-estimate; MEDIUM-band under-coverage disclosed) | zero conformal code in repo (`grep`); bands are fixed cut-offs on a point estimate — 29.9 vs 30.1 flips grounding (`edge_node.py:178-185`) | biggest operational-credibility upgrade; scheduler and dashboard could act on interval, not point | S–M | none — sidecar calibration artifact |
| **P2** | **Raised-cap (145) Ridge variant** — linear, retrains in seconds | `src/config.py:33` MAX_RUL=125 vs truth to 145 (`RUL_FD001.txt` tail 137/142/145); flagged-but-unrun follow-up in RESULTS.md §9.2 | removes the ~structural penalty on 11/100 engines; closes an acknowledged gap; expected RMSE below current capped-equivalent | S | additive experiment only |
| **P3** | **Feature engineering** (moving averages, first differences, degradation-rate features) for the Ridge class | `src/data/preprocessing.py` uses raw MinMax'd sensors only | cheapest path to a headline accuracy gain, and Ridge *is* the deployed model class | M | additive |
| **P4** | **Multi-subset generalization (FD002–FD004)** — files already ship in `Dataset/` (verified: train/test/RUL for all four; FD002 has the standard 259 test engines) | `src/config.py:25-27` hardcodes FD001 filenames; nothing else uses the others | claims become general, not single-variant; highest external-validity payoff | L | config-parameterization only |
| **P5** | **Statistical rigor on test results**: bootstrap CIs on RMSE, paired per-engine resampling tests (Ridge vs Transformer) | Ridge is one fit, no CV/seeds (`results/baselines.json`); per-engine predictions already saved | lets claims say "Ridge < Transformer with p<…" instead of point comparisons | S | pure post-processing of saved artifacts |
| **P6** | **Reproducibility debt**: requirements pin nothing (torch>=2.0.0 vs actual 2.14.1+cpu per frozen `environment.json`); minor LSTM inconsistency to reconcile (17.65 in `baselines.json` vs 17.68 in `nasa_audit.json`) | `requirements.txt`, results JSONs | a clean rebuild cannot currently reproduce the environment | S | none |

Seeds in the *training* code are properly pinned (`main.py:33-40`, `run_seed.py`), and metric code is shared via `src/evaluate.compute_metrics` with no duplication — both good.

**Judge-needle ranking: P0, then P1.** Together they turn "Transformer loses to Ridge and val numbers are unexplained" into "found and fixed a subtle split leak, then shipped calibrated risk bands" — the strongest available story given everything already verified.

---

## 3. External survey — open-source approaches worth integrating (or not)

Honesty note: all performance numbers below are **reported by their authors, not independently verified**, and most are on the *standard public* FD001 (21 test engines, capped labels) — not comparable 1:1 with this repo's non-standard 100-engine uncapped variant. Negative results included: the widely-cited TARL repo (`charanvk/TARL`) and AT-BiLSTM (`moazelsayed/AT-BiLSTM`) are **404 — effectively no public code**; thuml's Time-Series-Library, darts, and sktime have **no C-MAPSS/RUL task** despite being general time-series libraries.

### Adopt (real integration candidates)

1. **MAPIE** (scikit-learn-contrib, BSD-3, ~1.6k★, active) — model-agnostic **conformal prediction intervals for regression**, wraps any sklearn estimator. Fits P1 exactly: calibrate on a held-out split of Ridge residuals, no refit → frozen-artifact-safe, CPU-native. → [repo](https://github.com/scikit-learn-contrib/MAPIE), [docs](https://mapie.readthedocs.io/en/latest/). *Verdict: adopt for calibrated risk bands.*
2. **XGBoost as a protocol-gated candidate model** — supported by a 2026 matched-pipeline comparison ([arXiv 2604.27234](https://arxiv.org/html/2604.27234v1)) and by [alels001/rul-prediction-cmapss-fd001](https://github.com/alels001/rul-prediction-cmapss-fd001) (MIT, created Oct 2026), which unusually evaluates on **100 test engines** — same shape as this variant — reporting XGBoost RMSE 12.03 / frozen Chronos-2 12.08 / LSTM 13.49 / LR 17.48 (authors' numbers, unverified). CPU-friendly (~23 s fit on 16-core), and feature-importance explainability matches the Ridge `top_issues` story. *Verdict: adopt as an experiment under the existing protocol (validation-only selection first).*
3. **PyOD** (BSD-2, ~10k★, very active) — not an RUL method; a per-engine anomaly/degradation flag layer for edge reports *before* RUL trust is invoked. CPU, sklearn-style. → [repo](https://github.com/yzhao062/pyod). *Verdict: adopt as an optional monitoring layer.*

### Borrow idea / methodology (do not vendor code)

4. **alels001's methodology** — Wilcoxon + Holm paired tests, and cleanly separating *label capping* from *prediction clipping* — directly implements P5 for this repo. (Same link as #2.)
5. **biswajitsahoo1111/rul_codes_open** (MIT, ~221★, Jan 2025) — reproducible classical-ML C-MAPSS baselines (SVR, GP, ensembles) with explicit protocols; closest OSS analog to this repo's Ridge-audit workflow; useful cross-check. → [repo](https://github.com/biswajitsahoo1111/rul_codes_open).
6. **tilman151/rul-datasets** (~72★, active Mar 2026, CI-tested) — drop-in PyTorch-Lightning data modules for C-MAPSS / N-CMAPSS / FEMTO / XJTU; makes P4 (FD002–FD004) cheap. ⚠ API shows **no license** → use the loader pattern, or go to the canonical NASA source directly. → [repo](https://github.com/tilman151/rul-datasets).
7. **Frank-Wang-oss/GNN_RUL_Benchmarking** (~59★, paper [arXiv:2409.19629](https://arxiv.org/pdf/2409.19629)) — benchmarking protocol with preprocessed FD001–FD004; also confirms the canonical dataset lives at NASA: [data.nasa.gov CMAPSS ff5v-kuh6](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6). ⚠ no license → protocol only.
8. **Conformal-for-RUL prior art** — [alireza-javanmardi/conformal-RUL-intervals](https://github.com/alireza-javanmardi/conformal-RUL-intervals) (IJPHM 2023; validates the design of #1) and [arthurviens/bayesrul](https://github.com/arthurviens/bayesrul) (BNN uncertainty on N-CMAPSS). ⚠ first has no license → implement via MAPIE instead.

### Skip

9. **jiaxiang-cheng/PyTorch-Transformer-for-RUL-Prediction** (~311★, Apache-2.0) — last push **2021**; reported FD001 RMSE ≈12.16 (authors, unverified) is on the capped-label public split, not comparable; GCU-Transformer derivatives add fragility (PyTorch 1.8, no license).
10. **Graph methods (SAGCN etc.)** — [uta-smile/Adaptive-Graph-Convolutional-Network](https://github.com/uta-smile/Adaptive-Graph-Convolutional-Network) is TF1/2020/no-license; not worth the stack risk on a CPU Windows demo.
11. **Foundation models for RUL** — nothing verified exists beyond the GPU-only Chronos comparison in #2; premature here.
12. **Fleet/PHM OSS ecosystem** — searches found only 0–6★ one-off repos (e.g. [aero_phm_kgqa_manager](https://github.com/JeanChowRan/aero_phm_kgqa_manager)); OR-Tools' official examples include **no preventive-maintenance-scheduling model** — the repo's own CP-SAT phase logic is not behind the OSS state. Keep it.

---

## 4. Consolidated roadmap

**Before submission (hours, additive, no artifact risk):** P0 split fix + 3-seed re-run (decides the no_pe question honestly) · conformal sidecar via MAPIE (P1) · bootstrap CI on the existing test predictions (P5, free post-processing) · engineering quick wins 1–3 + 5 · pin one lockfile (P6).

**If time allows (days):** raised-cap Ridge variant (P2) · XGBoost under protocol (#2) · feature engineering (P3).

**After submission (scope decisions):** FD002–FD004 generalization (P4) · pytest suite + CI (1.3) · shared contracts package (1.5) · logging/migrations (1.4).

**Standing rule:** the frozen Ridge artifacts, `results/`, `edge/models/ridge_model/`, and the deployed services stay byte-identical; every item above lands in new experiment dirs or a separate calibration artifact, and no superiority language changes anywhere until a head-to-head under the fixed protocol says so.
