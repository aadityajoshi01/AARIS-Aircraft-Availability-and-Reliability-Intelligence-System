# AARIS — Detailed Work Report (for presentation extraction)

Date: 2026-10-11 (ML evidence), with the demo fleet scaled to 90 aircraft (6 synthetic model
blocks × 15) and the whole stack re-verified end to end on 2026-10-17. Covers the full arc from
the inherited broken Transformer repo to the verified, secured, three-service fleet-readiness
platform, including dashboard authentication, the completed leak-free experiment matrix, and
the 90-aircraft fleet/scheduler/dashboard expansion. Every number below was produced by scripts
in this repo and is re-checkable (§11). The "headline number sheet" (§12) and the slide map (§13)
are provided specifically so a deck can be assembled from this file.

Claim discipline used throughout (do not let slides break these):
- The only Ridge NASA score that circulates in headlines is **413** (audited, PHM08-correct).
  444/443.9 is the pre-audit scale-swapped value and exists only as audit trail.
- All Transformer variant findings (`no_pe`, `seq50`, `sensors`) are **validation-only**;
  no test-performance or deployment-superiority claim is made for any of them.
- Jetson status is "field-node-ready, verified on dev machine" — never "Jetson-verified".
- No artifact claims the Transformer beats a simple baseline: Ridge wins on RMSE under both
  truth conventions.

---

## 1. What we inherited

`NASA-C-MAPSS_Transformer_Based_Analysis-main` — a C-MAPSS FD001 RUL Transformer whose
published-style training had **collapsed**: the shipped model predicted a single constant
(~39.03 cycles) for every engine. Test behavior of that collapsed state: **RMSE 55.99**,
NASA score 257,838 (i.e. a constant predictor dressed as a neural net).

Root cause (verified, not assumed): regression targets of 0–125 cycles fed raw into an MSE
loss with `LAMBDA_OVER = 10` asymmetric penalty and gradient clipping — the loss landscape
had a flat "predict the mean" basin the model fell into and never escaped. The original
artifacts show loss *rising* 4250 → 4660 in a 5-epoch run.

## 2. The fix and the evidence protocol

Fixes applied to the pipeline: **labels normalized by `MAX_RUL`** (0–1 targets), window-target
alignment corrected, gradient clipping made meaningful, and the training loop hardened. This
removed the basin's attractor strength; measured escape behavior confirmed it.

The user dictated a standing validation protocol, which shaped everything after:
1. Re-verify evaluation integrity (label scaling, engine-ID ↔ RUL pairing, denorm/clipping,
   independent RMSE recomputation from predictions.csv).
2. Archive every successful run as a bundle (checkpoint + config + logs + plots).
3. Compare against baselines on the identical split (constant-mean, Ridge, simple LSTM).
4. Repeatability across ≥3 seeds; report mean ± variation, never a lucky single run.
5. Freeze the reproducible baseline before further iteration (archive + `environment.json`
   + SHA256 manifest — `results/frozen_baseline_2026-10-11/`).
6. Audit metric *implementations*, not just values.
7. Run targeted single-factor experiments, select on validation only, keep the test set
   untouched for one final reserved evaluation.
8. Document failure, fix, and verified improvement honestly (`RESULTS.md` style).

## 3. Verified performance (FD001 test, 100 engines, one code path for every row)

Dataset caveat: this is a **non-standard FD001 variant** (100 test engines, uncapped ground
truth up to 145, 6 zero-variance sensors), so numbers are not comparable to published
21-engine FD001 SOTA (~11–14 RMSE).

| Model | Test RMSE (uncapped truth) | MAE | NASA (audited) |
|---|---|---|---|
| Constant-38 (the inherited collapse) | 55.99 | 47.76 | 257,838 |
| Constant-mean | 41.89 | 36.14 | 7,391 |
| **Ridge on flattened 30×24 window** | **16.22** | 13.21 | **413** |
| 1-layer LSTM (30 epochs) | 17.65 | 12.54 | 973 |
| Transformer, 3 seeds | **18.46 ± 0.76** (17.42–19.21) | 13.71 ± 0.73 | — |
| Transformer, capped-truth rescore | 17.02 ± 0.84 (Ridge rescores 14.88) | 12.64 | — |

- Headline improvement: **test RMSE 55.3 → 18.46 ± 0.76 (3.0×)**, same 100 engines.
- Predictions went from 1 unique value to full spread (94 unique, corr with truth 0.899).
- Both truth conventions reported (capped at 125 vs shipped uncapped); the ~1.4-cycle gap is
  structural cap mismatch (11/100 engines have truth > 125), not model error.
- Honest finding kept: **Ridge beats the Transformer on this variant under both conventions**.
  The Transformer's demonstrable win is over the collapsed state, and it matches/exceeds the
  LSTM. Nothing claims "Transformer > regression".

## 4. Bugs we caught in our own work (audit findings → resolutions)

**(a) NASA score convention bug.** The repo's scoring had the PHM08 late/early penalty scales
(13/10) swapped. Audited and corrected (`experiments/nasa_audit.py`, `results/nasa_audit.json`):
Ridge 444 → **413**. Rule adopted: only 413 circulates.

**(b) Validation split leak (the big one).** `src/data/dataset.py` splits *windows* randomly
while windows are stride-1 slides sharing 29 of 30 rows — near-duplicates landed on both sides
of every train/val split. Consequence: all earlier validation numbers were inflated (absurd
4.92 val vs 17–19 test), and the flagship `no_pe` observation was one seed on a leaky metric.
Test-set numbers were never affected (test engines disjoint, final-window construction).

**Resolution — leak-free engine-wise matrix (completed 2026-10-11).**
`experiments/engine_split.py` holds out 20% of *engines whole* (20 val engines per run; the
seed also re-draws the split so seed spread honestly includes split variance); everything else
identical to the original protocol (same preprocessor, normalized labels, model, asymmetric
loss, 30-epoch budget, patience 15). 4 configs × 3 seeds = **12/12 runs, all `leak_free: true`**
(`results/engine_split/`).

Best val RMSE (cycles), last-5-epoch mean in parentheses:

| Variant | seed 42 | seed 43 | seed 44 | mean best | Verdict |
|---|---|---|---|---|---|
| control | 13.30 (14.88) | 18.19 (20.89) | 13.63 (17.26) | 15.04 | reference |
| `no_pe` (PE → Identity) | 16.94 (18.83) | 19.43 (22.76) | 21.17 (22.31) | 19.18 | **worse than control at all 3 seeds** — old 4.92 was pure leak artifact |
| `seq50` (window 30→50) | 12.59 (14.61) | 15.01 (17.33) | 11.46 (12.24) | 13.02 | **beats control at all 3 seeds** — only surviving candidate |
| `sensors` (drop 6 zero-variance) | 12.84 (14.74) | 16.37 (20.78) | 15.27 (16.09) | 14.83 | borderline — last-5 mean better at all seeds, best-epoch loses seed 44 |

Protocol story: the leak-free validation band (11.5–21.2) now sits where it should relative to
test RMSE (17.4–19.2), instead of the leaky split's impossible 4.9. The single reserved
test-set evaluation remains **unspent**; it no longer applies to `no_pe` (eliminated) and would
only make sense for `seq50`, and only on explicit owner go-ahead.

## 5. Calibrated uncertainty (conformal sidecar)

`experiments/conformal_ridge.py` → `conformal/risk_calibration.json`: cross-conformal on the
frozen Ridge recipe (engine-wise GroupKFold k=5, per-fold scaler + Ridge(alpha=10) refit,
absolute out-of-fold residuals in final clipped-cycles scale; deterministic, ~15 s).

- 90% interval half-width **±27.19 cycles** (80%: 21.81, 95%: 31.32).
- Empirical OOF coverage 0.800 / 0.900 / 0.950; OOF RMSE 16.41 ≈ frozen test RMSE 16.22
  (calibration scale tracks test error).
- Disclosed caveat (kept in all docs): global half-widths **under-cover the MEDIUM band**
  (90% interval coverage only 76% there); LOW/HIGH/CRITICAL ≈ 97–100%.
- Edge `/predict` gained an additive `uncertainty` block (interval clipped to [0,125], a
  set-membership statement against the 30-cycle grounding threshold, per-band coverage).
  Point estimates and the 112.76/90.21/LOW fixture are byte-unchanged; graceful
  `available:false` if the sidecar is absent.
- Dashboard Aircraft Detail renders the interval live from the sidecar.
- Rule the deck must keep: **the interval never re-labels the risk band — bands stay
  point-estimate**; it informs borderline calls (e.g. AC-001 demo: [0.0, 29.6] → "robust ground").

## 6. What we built: the AARIS system

Three services + demo kit, each layer verified before the next was built on it, real outputs
(not mocks) at every integration step. Ridge (not the Transformer) was packaged because the
evidence said so: simpler, verified, and it wins on RMSE.

```
Aircraft Edge Node                Ground Station                 Dashboard
FastAPI :8000                     FastAPI :8001 + SQLite         Streamlit :8501
30×24 window → Ridge RUL  ──POST /ingest_report──▶  fleet state + report history     GET /fleet
health/risk/conformal report      CP-SAT 30-day planner         GET /fleet/{id}
                                  + security layer (JWT/RBAC/   GET /schedule/latest
                                    audit)          ◀──Bearer──── (commander service token)
```

### 6.1 Edge node (`edge/edge_node.py`)
Frozen Ridge artifacts (`ridge_model.pkl`, `scaler.pkl`, `feature_order.json`), byte-identical
to the validated run, SHA256-verified, never retrained. Inference = one matrix multiply:
MinMaxScaler → flatten 30×24 → Ridge → clip [0, 125]. Response: rul_cycles, engine_health
(clip(rul/125·100, 0, 100)), risk band (LOW ≥100 / MEDIUM 60–99 / HIGH 30–59 / CRITICAL <30),
**top_issues** (Ridge coefficient × scaled contribution — model-explainable degradation
drivers), recommended_action, and the conformal uncertainty block. Strict input validation
(non-numeric/non-finite/wrong-shape → 422 with offending positions; a real NaN-crash bug in
FastAPI's default handler was found and fixed). **Verified fixture: engine-1 window →
112.76 cycles / 90.21% health / LOW**, matching the offline predictor to the cent.
Field-node story: Jetson-class deployment path documented with pickle-compatibility pins
(numpy/scikit-learn/pandas), stdlib-only smoke script; dev-machine verified, hardware run pending.

### 6.2 Ground station (`ground/ground_station.py`)
FastAPI + SQLite (`ground/fleet.db`), roster of 90 aircraft (AC-001…AC-090, 6 models × 15
airframes; synthetic bases ASN/BBI/BLR/… with mission priorities and spares). `POST /ingest_report` accepts the exact
edge response; latest state updates **only on strictly newer timestamps** (out-of-order reports
kept as history); NaN-safe 422s. `GET /fleet`, `GET /fleet/{id}` (+history), `GET /models`
(6 model blocks × 15 airframes), `GET /alerts`, `POST /schedule`,
`GET /schedule/latest`, `GET /health`. All timestamps UTC ISO-8601.

### 6.3 CP-SAT scheduler (`ground/scheduler.py`)
Google OR-Tools CP-SAT, rules as specified: 30-day horizon, ≤1 task per aircraft,
**max 8 maintenance/day**, deadline `max(1, min(30, floor(RUL×0.8)))`, urgent aircraft never
late, spares a hard constraint only when all deadlines stay feasible (3-phase fallback),
LOW-risk/no-telemetry aircraft DEFERRED with explicit reasons. Deterministic
(num_search_workers=1, random_seed=7) — two solves byte-identical. Plan persisted, so
`/schedule/latest` survives restarts. **Demo fleet plan: OPTIMAL, 54 scheduled / 36 deferred;
peak load 8/day (8 on days 1–6, 6 on day 7, nothing after day 7); AC-001 (RUL 2.4) on day 1 =
its deadline; the no-spare AC-003 (RUL 7.0) held to day 5 by the spare-lead rule. The 36
deferrals = 24 healthy LOW aircraft + 12 airframes still lacking telemetry.**

### 6.4 Dashboard (`dashboard/app.py`) — "Fleet Readiness Decision Support"
Plain Streamlit (no custom CSS), read-only, answers five operator questions:
available now / at risk / who must enter maintenance and by when / daily load / deferrals
+ reasons. Headline **Fleet Availability KPI** = (not CRITICAL AND not in maintenance today
UTC) / full roster, **computed live from the API on every render, never hardcoded** — a
freshly reset demo fleet reads **80.0% (72/90)** (it moves with the plan day, e.g. **71.1%
(64/90)** six days later); unreported aircraft counted as not-CRITICAL are surfaced in the
caption; a stale/absent plan means "no plan for today", never a silent zero. Aircraft Detail
adds the conformal interval; Maintenance Plan shows Gantt + full task table + solver status.
**DEMO MODE** button (sidebar, directly under the banner logo) serves everything from the
bundled `demo_data.json` snapshot with no API and no credentials. Sections are switched by a
state-driven `st.segmented_control` nav bar (Fleet Readiness · Aircraft Models ·
Aircraft Detail · Maintenance Plan · About) rather than `st.tabs`, specifically so the app can change section from code:
**one click on an aircraft opens its profile** — any row of the readiness, next-7-days,
deferred or plan-task tables, plus a **Profile** button on each critical alert. Verified in a
real browser (fleet row → AC-005, deferred → AC-017, plan tasks → AC-008, alert → AC-001) and
headlessly as `app_tests.py` T10 (DEMO MODE alert → profile, zero network).

## 7. Security layer (ground station)

Additive; did not touch model artifacts, prediction logic, scheduler logic, or response schemas.

- **Humans**: HS256 JWTs from `POST /auth/login` (OAuth2 form). Demo users:
  maintainer/maintainer123 → `maintenance_officer`, commander/commander123 → `commander`,
  admin/admin123 → `admin`. TTL default 60 min (`AEROGUARD_JWT_TTL_MIN`); signing key
  `AEROGUARD_SECRET_KEY` with loud dev-default warning.
- **Edge devices**: static bearer tokens `EDGE_<AIRCRAFT>_TOKEN` (dev default
  `EDGE_AC017_TOKEN` → AC-017 only), constant-time compared.
- **Deny-by-default RBAC**: ingest = edge device on its own aircraft (+admin); fleet reads =
  maintenance_officer/commander/admin; `POST /schedule` = maintenance_officer/admin (commander
  may read but not schedule); 401 missing/invalid/expired, 403 wrong role. `/health` open.
- **Audit trail** (`audit_logs` table): logins, auth denials, permission denials, ingestion
  success/rejection, fleet/schedule views, scheduling runs — **never tokens, passwords, or raw
  payloads** (smoke asserts the hygiene).
- Edge node gained an optional authenticated push (`GROUND_PUSH_URL`, off by default); local
  `/predict` stays credential-free for the field-node demo.

## 8. Dashboard LIVE-mode authentication (this session)

The security layer broke the dashboard's LIVE mode (protected GETs → 401). Fixed with a
commander **service account**:

- `DASHBOARD_SERVICE_TOKEN` (env var, prefills a masked sidebar field; empty default) is sent
  as `Authorization: Bearer <token>` on exactly `GET /fleet`, `GET /fleet/{id}`,
  `GET /schedule/latest`; 5-second timeouts kept. The token is **never displayed or logged**
  (masked input; error messages quote only HTTP statuses).
- No token → LIVE mode makes **zero network calls** and shows setup guidance (login command
  + "click the DEMO MODE button" suggestion) instead of a wall of 401s.
- 401 → "Ground station rejected the dashboard credentials…" (absent/malformed/expired;
  60-min TTL called out). 403 → "Dashboard service account does not have permission for this
  endpoint…" (wrong role). Identical fleet/schedule errors deduped.
- DEMO MODE path untouched (bundled data, layout). Its widget later became a
  prominent sidebar button under the banner logo; `app_tests.py` T8 clicks it
  and re-verifies the path.
- Mint a token: `curl -X POST http://127.0.0.1:8001/auth/login -d "username=commander&password=commander123"`,
  run: `export DASHBOARD_SERVICE_TOKEN=<jwt>` then streamlit; re-mint when the 401 message appears.

## 9. Verification ledger (what was actually run, and passed)

| Check | Result |
|---|---|
| Edge smoke (`edge/smoke_test.py`) | PASS — 6 steps incl. fixture + uncertainty block |
| Ground+security smoke (`ground/smoke_test.py`) | PASS — full [0]–[14] + [M1] (GET /models) + [S1]–[S6] + [A1]–[A5b]: RBAC matrix, 401/403 negatives, device-token binding, audit capture + zero-secret hygiene, and the critical-alert invariant (ACTIVE alerts == exactly the 18 CRITICAL airframes, one each; ack/clear semantics) |
| Demo rehearsal (`demo/verify_pipeline.py`) | **13/13 PASS** live and after full `-Reset` (incl. commander login, unauthenticated-401, plan 54/36, /alerts == CRITICAL aircraft) |
| Dashboard suite (`dashboard/app_tests.py`, Streamlit AppTest) | **10/10 PASS** vs the secured live station (login form with zero network; Bearer header on every LIVE GET incl. `/alerts`; token absent from UI; 401/403 messages; DEMO MODE offline incl. bundled alert banner, Acknowledge click 18->17, and the Profile click that switches to Aircraft Detail); graceful live-test skip with station down |
| Real-browser LIVE session (verified 2026-10-17) | 90 aircraft, KPI 80.0% on a fresh reset (the value is date-dependent — **71.1% = 64/90** six days into the same plan), AC-017 112.76/90.21/LOW + interval [85.6, 125.0], plan OPTIMAL 54/36 — then DEMO MODE re-verified **with the ground station stopped** |
| Real-browser one-click profile | fleet row → AC-005, deferred row → AC-017, plan-task row → AC-008, alert **Profile** → AC-001 (LIVE) and → AC-079 (DEMO MODE — the first bundled critical alert): each single click moved the nav to Aircraft Detail with that aircraft selected |
| Engine-wise matrix | 12/12 runs `leak_free: true`, deterministic protocol |
| `start.bat` / `stop.bat` round-trip | tested after every service change |

## 10. Limitations kept honest (no slide should overstate these)

- Dataset is a non-standard FD001 variant; not comparable to published FD001 SOTA numbers.
- The Transformer does not beat Ridge here; deployment chose Ridge on evidence.
- All variant experiments are validation-only; the one reserved test evaluation is unspent and
  owner-gated (`no_pe` is now unqualified for it).
- 11/100 test engines have truth > 125 — a structural equal-floor penalty for every clipped model.
- Fleet, bases, priorities, spares are a synthetic demo rig.
- Conformal coverage is FD001-distribution heuristic; MEDIUM band under-covers (76%) and is
  disclosed everywhere it renders.
- Jetson run on real hardware pending; NASA headline score = 413 only.
- Dashboard: read-only. Only the LIVE critical-alerts banner auto-refreshes (~20 s); the rest
  of the page updates on interaction or via **Refresh**.

## 11. Reproduce everything

```bash
# ML evidence
.venv/Scripts/python.exe experiments/engine_split.py          # 12-run leak-free matrix
.venv/Scripts/python.exe experiments/conformal_ridge.py       # rebuild risk_calibration.json
# full stack
start.bat                     # edge 8000 + secured ground 8001 + dashboard 8501
start.bat --live-feed         # ... plus the AC-017 synthetic reporter (live alerts)
start.bat --fleet-feed        # ... plus one synthetic reporter per aircraft (AC-001..AC-090)
stop.bat                      # shuts all of the above down, reporter included
powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset   # scripted demo fleet + plan
.venv/Scripts/python.exe demo/verify_pipeline.py              # expect 13/13 PASS
.venv/Scripts/python.exe dashboard/app_tests.py               # expect 10/10 (with station up)
# dashboard LIVE mode
curl -X POST http://127.0.0.1:8001/auth/login -d "username=commander&password=commander123"
export DASHBOARD_SERVICE_TOKEN=<jwt>   # then streamlit run dashboard/app.py
```

## 12. Headline number sheet (one line each — drop straight into slides)

| Fact | Value | Source |
|---|---|---|
| Inherited collapse, constant-38 predictor | test RMSE 55.99 | RESULTS.md §5 |
| Fixed pipeline, Transformer 3 seeds | 18.46 ± 0.76 (uncapped) / 17.02 ± 0.84 (capped) | RESULTS.md §5 |
| Deployed Ridge model | **16.22 RMSE / NASA 413** | RESULTS.md §5, §8 (corrected) |
| Headline improvement | 3.0× RMSE reduction, same 100 test engines | RESULTS.md §5 |
| Edge verified fixture | 112.76 cycles / 90.21% / LOW | edge/models/ridge_model/ |
| Conformal 90% interval | ±27.19 cycles; OOF coverage 0.900 | conformal/risk_calibration.json |
| MEDIUM-band caveat | 76% coverage (disclosed) | conformal sidecar band_diagnostics |
| Leak-free matrix | 12/12 runs, engine-wise, 3 seeds | results/engine_split/ |
| no_pe after fix | loses at all 3 seeds (16.94/19.43/21.17 vs 13.30/18.19/13.63) | results/engine_split/ |
| seq50 after fix | wins at all 3 seeds (12.59/15.01/11.46) | results/engine_split/ |
| CP-SAT plan | OPTIMAL, 54 scheduled / 36 deferred, peak 8/day | ground scheduler, deterministic seed 7 |
| Fleet Availability KPI | 80.0% (72/90) on a freshly reset demo fleet; date-dependent (71.1% = 64/90 six days later) | dashboard/app.py::compute_readiness |
| Fleet risk spread (demo) | 18 CRITICAL / 18 HIGH / 18 MEDIUM / 24 LOW (+12 unreported) | ground/seed_demo.py |
| Security | JWT + device tokens, deny-by-default RBAC, audit w/ no secrets | ground/security.py, smoke S1–S6 |
| Dashboard LIVE auth | commander service token via DASHBOARD_SERVICE_TOKEN | dashboard/app.py, README |
| Rehearsal checks | verify_pipeline 13/13 · app_tests 10/10 · ground smoke (incl. alerts A1–A5b + [M1] /models) · edge smoke | §9 |

## 13. Suggested slide map

1. **Title** — AARIS: Aircraft Availability & Reliability Intelligence System (logo: `dashboard/assets/aaris_heading.png`, wide banner — square emblem `aaris_logo.png` as fallback; subtitle "Onboard AI & Fleet Readiness Platform"; C-MAPSS RUL → secured fleet stack).
2. **The problem we inherited** — collapsed Transformer, constant-39 outputs, RMSE 55.99 (§1).
3. **Root cause + fix** — unscaled labels + asymmetric loss basin; normalize + harden (§2).
4. **Evidence protocol** — integrity audit, baselines, ≥3 seeds, freeze, metric-code audit (§2).
5. **Head-to-head results** — table of §3; honest line: Ridge wins, Transformer's win is over the collapse.
6. **Auditing ourselves #1: NASA score** — 13/10 scales swapped → 413 (§4a).
7. **Auditing ourselves #2: the split leak** — what it inflated; the engine-wise rebuild (§4b).
8. **Leak-free matrix slide** — the 4×3 table + verdict arrows: no_pe eliminated, seq50 survives (§4b).
9. **Calibrated uncertainty** — ±27.19 c, coverage table, MEDIUM caveat, grounding phrase (§5).
10. **System architecture** — diagram from §6 (edge → secured ground → dashboard + CP-SAT).
11. **Edge node** — frozen artifacts, fixture 112.76/90.21/LOW, explainable top_issues (§6.1).
12. **Scheduler** — CP-SAT rules; OPTIMAL 54/36, peak 8/day, deterministic (§6.3).
13. **Dashboard decision support** — five operator questions + KPI formula (80.0% = 72/90 on a fresh demo) (§6.4).
14. **Security** — JWT/RBAC/device tokens/audit; what was NOT touched (§7).
15. **Dashboard LIVE auth + test pyramid** — service token (§8) + ledger table (§9).
16. **Limitations, stated up front** — §10 (keep this slide; it is the credibility slide).
17. **Reproduce in one command** — §11.
18. **Close** — reserved test evaluation decision gated on the owner; only seq50 qualifies.
