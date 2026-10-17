# AARIS — Aircraft Availability & Reliability Intelligence System

*Onboard AI & Fleet Readiness Platform*

A verified three-service system that turns onboard engine telemetry into RUL forecasts, fleet health visibility, and a deterministic 30-day maintenance plan — built on a frozen, independently validated Ridge RUL model after root-causing and repairing a collapsed Transformer baseline.

## Executive summary

AARIS is a fleet-readiness prototype in which each aircraft runs an onboard edge node that predicts remaining useful life (RUL, in cycles) from a 30-cycle sensor window. The edge node sends a compact health report to a ground station, which maintains fleet state and uses Google OR-Tools CP-SAT to generate a capacity-constrained 30-day maintenance plan. A plain Streamlit dashboard presents fleet health, aircraft diagnostics, and the resulting maintenance schedule.

The deployed model is a frozen Ridge regression achieving 16.22 RMSE uncapped and 14.88 RMSE capped (cycles) on the available FD001 test set. The system was verified end to end using real service outputs, deterministic scheduling, persisted plans, and repeatable smoke tests.

**System status**

| Component | Status |
|---|---|
| Edge node — frozen Ridge inference (`:8000`) | ✅ live, fixture-verified (112.76 cycles / 90.21% / LOW) |
| Ground station — SQLite ingest + fleet state (`:8001`) | ✅ live, full ground+security+scheduler smoke test (incl. a `/models` check) |
| CP-SAT 30-day maintenance scheduler | ✅ live, deterministic, plan persists across restart |
| Dashboard — plain Streamlit (`:8501`) | ✅ verified live and in DEMO MODE |
| Demo rehearsal (`start.bat`, `run_demo.ps1`, `verify_pipeline.py`) | ✅ 13/13 PASS live and after full reset |
| `no_pe` test-set evaluation | ⏸ reserved test run unused — after the leak-free 12-run re-run (§5), `no_pe` loses to control at all 3 seeds, so it no longer qualifies; only `seq50` survives, still awaiting an explicit go-ahead |

The sections below tell the full story: what we inherited (a collapsed repo), how we root-caused and fixed it, what the evidence says, and what was built and verified on top of it.

Dates: 2026-10-10 → 2026-10-11 for the ML evidence, with the demo fleet scaled to 90 aircraft (6 model blocks × 15) and the whole stack re-verified end to end on 2026-10-17. All work run on this machine (Windows, CPU-only 16-core, `.venv` Python 3.14). Every number below was produced by scripts in this repo and is re-checkable with the commands in §10.

---

## 1. Where we started

The starting point was `NASA-C-MAPSS_Transformer_Based_Analysis-main` — a repo claiming Transformer-based Remaining Useful Life (RUL) prediction on NASA C-MAPSS FD001. Reality check found:

- The delivered model was **collapsed**: test RMSE 55.30 cycles with a constant prediction of **39.03 cycles** for all 100 test engines.
- `checkpoint_best.pt` came from epoch 3 of a 5-epoch run whose loss was **rising** (4250 → 4660).
- The dataset shipped in `Dataset/` is a **non-standard FD001 variant**: 100 test engines (public FD001 has 21), uncapped ground-truth RULs up to 145 while training targets are capped at MAX_RUL=125, and 6 zero-variance sensor columns.
- The repo's NASA score implementation had the 13/10 penalty scales **swapped** (an early false "superiority by NASA score" almost slipped through).

No working pipeline, no trustworthy metrics, no deployment. That was the "what we had."

## 2. Root-causing the collapse (verified, not assumed)

Four causes, each confirmed by measurement:

1. **Unscaled labels.** Targets 0–125 while features were MinMax-scaled → gradient norms ~1600, and `clip_grad_norm_(max_norm=1.0)` crushed every step by ~1/1600.
2. **Degenerate optimum of the asymmetric loss.** With `LAMBDA_OVER=10`, the constant minimizing `E[(c−y)² + 10·relu(c−y)²]` over the training windows is **38** — the collapsed model sat at 39.03. Inside that basin, encoder gradients measured ~0.
3. **Window/target misalignment.** Training windows paired row-range i..i+29 with the RUL of row **i+30**, so no training window ever contained the final cycle — but every test window does.
4. Loading/scaler-order/NaN were explicitly ruled out — the model was genuinely collapsed, not mis-loaded.

**Fixes** (labels normalized by MAX_RUL; target = last row inside the window; denormalize+clip [0,125] at evaluation; validation RMSE reported in cycles). No change to architecture, loss, or optimizer — the same recipe now trains honestly.

## 3. The validation protocol we built and followed

Before accepting any number, we established (and it governed everything after):

- **Independent integrity audit** — rebuild sequences and recompute metrics from checkpoint; confirmed 17,731 windows, 1:1 engine-ID↔RUL pairing, clip active on 0 predictions, recomputed RMSE identical to `predictions.csv`.
- **Fixed baselines on the identical split** — constant-mean, constant-38, Ridge, 1-layer LSTM (`results/baselines.json`).
- **Repeatability across 3 seeds** (42/43/44), reported as a range, never a lucky run.
- **Metric-convention audit** — NASA score corrected and re-scored both ways (`results/nasa_audit.json`); results reported under capped-truth and uncapped-truth conventions.
- **Frozen baseline archive** — `results/frozen_baseline_2026-10-11.zip` with source + checkpoints + RESULTS.md + environment + SHA256 manifest.
- **Validation-only selection**, test set reserved for a single final evaluation.

## 4. Head-to-head results (test set, 100 engines, same code path for every row)

| Model | Test RMSE, cycles (uncapped truth) | Test RMSE, cycles (capped truth) | NASA score, corrected (uncapped truth) |
|---|---|---|---|
| Constant-38 (old collapsed behavior) | 55.99 | — | 30,201 |
| Constant-mean | 41.89 | — | 19,054 |
| **Ridge on flattened 30×24 window** | **16.22** | **14.88** | **413** |
| 1-layer LSTM (h=64, 30 ep) | 17.65 | — | 425 |
| Transformer, seed 42 (best epoch 48/50) | 18.76 | 17.00 | 544 |
| Transformer, seed 43 (best epoch 50/50) | 19.21 | 18.06 | 800 |
| Transformer, seed 44 (best epoch 49/50) | **17.42** | 16.00 | 608 |

(Sources: `results/baselines.json`, `results/nasa_audit.json`, `results/capped_truth_metrics.json`; under the capped-truth convention the Transformer seeds score 708/661/487.)

**The 413-versus-444 note (one number circulates):** the table uses Ridge's **audited** NASA score, **413** (413.3 in `results/nasa_audit.json`, `nasa_standard_correct`) under the PHM08-standard convention. The 444 figure you may see elsewhere is `nasa_audit.json`'s `repo_nasa_as_implemented` value (443.9) — Ridge's score under the repo's original, scale-swapped metric before the audit fix. It is kept in `RESULTS.md` §8 purely as the audit trail. Every headline, slide, README, and demo talking point uses **413**.

**The honest headline:** the fix is real (55.3 → ~18 RMSE is a genuine escape from collapse), but **a 2-parameter-tuned Ridge regression on the flattened window beats the Transformer on this dataset variant** under both metric conventions (14.88 vs ~17.02 capped truth). We documented that instead of hiding it — and it drove the deployment decision below.

Reading the seed rows: **seed 44 is the best Transformer on RMSE under both conventions (17.42 / 16.00), while seed 42 scores best on the NASA metric (544).** That divergence is expected — RMSE weights all errors equally, while the NASA score's asymmetric exponential penalty rewards a different error profile — and it is exactly why the protocol requires both. No single seed dominates on both columns; the mean across seeds is what we quote in claims.

**Deployment choice, stated explicitly:** AARIS runs the frozen Ridge model because it is the most accurate model in this repo *and* the cheapest to run — it beats the best Transformer seed on RMSE under both conventions (16.22 vs 17.42 uncapped; 14.88 vs 16.00 capped), has no seed variance at all (a closed-form fit gives byte-identical results every time), runs in milliseconds as one matrix multiply on CPU, and its coefficients are what make the edge node's `top_issues` explainable. Ridge wins on accuracy per unit of compute and memory, with explainability as a bonus — not as a trade-off against accuracy.

## 5. Targeted experiments (validation-only; test still untouched)

Three single-factor variants were first screened at seed 42 on the original window-level split (control bar 10.48): `no_pe` 4.92, `seq50` 7.71, `sensors` 13.77. Our own audit (`docs/AUDIT_AND_IMPROVEMENTS.md` §0) then caught a protocol flaw: `dataset.py` splits **windows**, not engines, while windows are stride-1 slides sharing 29 of 30 rows — so near-duplicate windows sat on both sides of the split and **every one of those validation numbers was leak-inflated**. They are superseded, not quoted, below.

**Leak-free re-run (completed 2026-10-11).** `experiments/engine_split.py` holds out 20% of the training **engines whole** (20 validation engines per run; the seed also re-draws the engine split, so seed spread honestly includes split variance). Everything else is identical to the first screen: same preprocessor, normalized labels, model, asymmetric loss, `Trainer`, 30-epoch budget with patience 15. 4 configs × 3 seeds = **12 runs, 12/12** `val_result.json` written, all `"leak_free": true`; outputs in `results/engine_split/`.

Best val RMSE (cycles), last-5-epoch mean in parentheses:

| Variant | seed 42 | seed 43 | seed 44 | mean best | vs control (best RMSE, per seed) |
|---|---|---|---|---|---|
| `control` | 13.30 (14.88) | 18.19 (20.89) | 13.63 (17.26) | 15.04 | — |
| `no_pe` (PE → Identity) | 16.94 (18.83) | 19.43 (22.76) | 21.17 (22.31) | 19.18 | **worse at all 3 seeds** (+3.6 / +1.2 / +7.5) |
| `seq50` (window 30 → 50) | 12.59 (14.61) | 15.01 (17.33) | 11.46 (12.24) | 13.02 | **better at all 3 seeds** (−0.7 / −3.2 / −2.2) |
| `sensors` (drop 6 zero-variance) | 12.84 (14.74) | 16.37 (20.78) | 15.27 (16.09) | 14.83 | better at 2/3 seeds; seed 44 best is worse (last-5 mean better at all 3) |

What the leak-free protocol changes, plainly:

- **`no_pe` does not survive its own gate.** Its headline 4.92 was pure leak artifact — under an honest split, removing positional encoding hurts at every seed. The one reserved test-set evaluation should **not** be spent on it.
- **`seq50` is the only variant that beats control at every seed** — the sole surviving candidate if a test evaluation is ever commissioned.
- **`sensors` is borderline**: consistently good on the stability metric (last-5 mean), inconsistent on best-epoch — not a clean win.
- Coherence check the leak could not pass: leak-free validation values (11.5–21.2) now sit in the same range as test RMSE (17.4–19.2), instead of the absurd 4.9-vs-17 gap the leaky split produced.

All of the above remains **validation-only. No test-set evaluation has been run, so no test-performance or deployment-superiority claim is made** (deliberately — the one reserved test run awaits the go-ahead). The "PE hurts" reading is now seed-replicated but still validation-scoped, not a general claim about positional encoding. Until a test run happens, no artifact claims superiority: `demo_script.md`, the dashboard, and the edge/ground services were grep-checked and reference only the frozen Ridge model and its validated numbers. The §4 test-set head-to-head is unaffected by the leak (test engines were already disjoint from train engines; final-window construction).

## 6. What we built: the AARIS system

A three-service fleet-availability stack, each layer verified before the next was built on it, using **real outputs** (not mocks) at every integration step:

```text
┌─────────────────────┐         ┌───────────────────────┐         ┌────────────────────┐
│ Aircraft Edge Node  │         │    Ground Station     │         │     Dashboard      │
│  FastAPI :8000      │         │  FastAPI :8001        │         │  Streamlit :8501   │
│                     │         │  SQLite fleet state   │         │                    │
│ 30×24 sensor window │──POST──▶│ /ingest_report        │         │ GET /fleet         │
│ Ridge RUL inference │         │ fleet + history views │◀──GET───│ GET /schedule/...  │
│ health/risk report  │         │ CP-SAT 30-day planner │         │ fleet + plan views │
└─────────────────────┘         └───────────────────────┘         └────────────────────┘
```

Wire-level contract: edge `--POST /ingest_report-->` ground; dashboard `--GET /fleet, /fleet/{id}, /models, /alerts, /schedule/latest-->` ground. The dashboard never talks to the edge; the ground station is the single source of fleet state.

### 6.1 Edge node — `edge/edge_node.py`
FastAPI service loading the frozen artifacts **exactly once at startup** (`ridge_model.pkl`, `scaler.pkl`, `feature_order.json`, byte-identical copies in `edge/models/ridge_model/`, SHA256-verified, checksums exposed in `/health`). `POST /predict` takes a raw 30×24 window → aircraft_id, engine_id, timestamp, rul_cycles, engine_health (clip(rul/125·100,0,100)), risk band (LOW ≥100 / MEDIUM 60–99 / HIGH 30–59 / CRITICAL <30), **top_issues** (Ridge coefficients × scaled inputs — model-explainable degradation drivers), recommended_action (band playbook). Strict input validation: non-numeric, non-finite, wrong shape → 422 with exact offending positions. Verified fixture: engine-1 test window → **112.76 cycles / 90.21% / LOW**, matching the offline `predict_rul.py` to the cent. Found and fixed a real production bug along the way: FastAPI's default 422 handler crashes when echoing NaN inputs — replaced with a NaN-safe handler.

**Field-node story:** the aircraft-side profile is a Jetson-class AHMS edge processor — the whole node is the frozen artifact bundle plus one CPU-only FastAPI service (inference = one matrix multiply, no GPU, no torch). `docs/JETSON_FIELD_NODE.md` gives the exact aarch64 deployment path with pickle-compatibility version pins (sklearn 1.9.1 / numpy 2.5.3), and `demo/jetson_smoke.py` (stdlib-only, runs on a bare Jetson venv) proves the fixture on whatever device boots the service. Status, stated honestly: the path and script are verified on this Windows CPU machine (PASS, 112.76/90.21/LOW); **hardware execution on an actual Jetson is pending** — the claim is "field-node-ready", not "Jetson-verified".

### 6.2 Ground station — `ground/ground_station.py`
Separate FastAPI service, SQLite (`ground/fleet.db`), three tables: `aircraft` (roster of 90 seeded from `fleet_config.json` + latest state), `reports` (full history, every payload retained verbatim), `maintenance_plans` (persisted solver outputs). Contract: `POST /ingest_report` (201; latest state updates **only on strictly newer timestamps**, out-of-order reports kept as history), `GET /fleet`, `GET /fleet/{id}` (+history), `GET /models` (per-aircraft-model profile rollup of the 6 synthetic model blocks), `GET /alerts`, `POST /schedule`, `GET /schedule/latest`, `GET /health`. All timestamps UTC ISO-8601. A full ground+security+scheduler smoke test (checks [0]–[14], a [M1] `/models` check and [S1]–[S6]) on an isolated temp DB (`ground/smoke_test.py`), including an AC-017 report generated by the *real edge node* inside the same test.

### 6.3 CP-SAT maintenance scheduler — `ground/scheduler.py`
Google OR-Tools CP-SAT replaces the earlier deliberate 501 placeholder. Rules exactly as specified: 30-day horizon, ≤1 task per aircraft, **max 8 maintenance/day**, deadline `floor(RUL×0.8)` clamped to [1,30], urgent aircraft never late, objective = urgency + risk-weighted mission priority × scheduled day + deferral penalties, spare-lead (5 days) treated as a **hard constraint only when every aircraft can still be served on time**, otherwise soft with a documented trade-off. Every aircraft ends SCHEDULED (day, deadline, reason) or DEFERRED (explicit reason — healthy LOW-risk, or no telemetry). Deterministic (single worker, seed 7): two solves are byte-identical. Plan persisted so `/schedule/latest` returns the same plan after a cold service restart (proven; a persistence inconsistency caught during testing was fixed). On the seeded demo fleet: **OPTIMAL, 54 scheduled / 36 deferred; peak load 8/day (8 on days 1–6, 6 on day 7, nothing after day 7); AC-001 (RUL 2.4) scheduled day 1 = its deadline; the no-spare AC-003 (RUL 7.0) held to day 5 by the spare-lead rule.** The 36 deferrals are 24 healthy LOW aircraft (no need inside the 30-day horizon) plus 12 airframes still lacking telemetry.

### 6.4 Dashboard — `dashboard/app.py`
Plain Streamlit UI — no custom CSS, standard components, default Plotly styling, muted colors only for risk. Left navigation bar: AARIS banner logo on top with a prominent **DEMO MODE** button directly under it, then URL, risk/base filters, Refresh. DEMO MODE is backed by `dashboard/demo_data.json` (a snapshot captured from the live station, so the demo survives an API outage). Sections are switched with a state-driven nav bar (`st.segmented_control`: **Fleet Readiness · Aircraft Models · Aircraft Detail · Maintenance Plan · About**) rather than `st.tabs`, specifically so the app can move to a section from code: **one click on an aircraft opens its profile** — any row in the readiness, next-7-days, deferred or plan-task tables, and the **Profile** button on a critical alert, all select that aircraft in the Aircraft Detail picker and show it. The headline section is framed as **Fleet Readiness Decision Support**, answering the five operator questions explicitly: *Fleet Availability* KPI (aircraft not CRITICAL **and** not scheduled for maintenance today, over the full roster — recomputed live from `GET /fleet` + `GET /schedule/latest` every render, never hardcoded; **80.0% = 72/90** on a freshly reset demo fleet — date-dependent, since plan days are offsets from the plan's `generated_at`; e.g. **71.1% = 64/90** six days later), an AVAILABLE / GROUNDED / IN-MAINTENANCE-TODAY status column per aircraft, risk counts + RUL-vs-health scatter, a next-7-days "who enters maintenance by when" table with calendar dates, the daily-load chart, and the deferred-with-reasons table. Other sections: Aircraft Models (one synthetic profile card per type — 6 models × 15 airframes — with airframe count, ready/grounded, avg RUL, in-plan split and risk mix, above a 15-row airframe table), Aircraft Detail (incl. honest "no data" handling), Maintenance Plan (Gantt + full tasks table), About (states explicitly: RUL/health = model output; top_issues/recommended_action = operational heuristics). All HTTP calls timeout-bounded with readable errors. Verified live **and** offline via `streamlit.testing.v1.AppTest` + real browser render.

### 6.5 Operations kit
- `start.bat` / `stop.bat` (repo root) — one command up/down for all three services; round-trip tested; survive the space-in-path and Git-Bash-`timeout`-shadowing gotchas. `start.bat --live-feed` additionally runs the AC-017 synthetic reporter (15-min cadence, compressed for rehearsal) so the live profile + critical-alert path can be demoed in minutes; `start.bat --fleet-feed` runs one reporter per roster aircraft (AC-001…AC-090, each with its own device token, staggered) so all 90 profiles update live; `stop.bat` kills those reporters' python children too.
- `demo/run_demo.ps1` — rehearsal launcher; `-Reset` wipes the DB, reseeds the scripted demo fleet, and regenerates the plan so the demo always starts from known state.
- `demo/verify_pipeline.py` — thirteen independent HTTP checks (edge health → predict fixture → ground fleet → login + 401-deny → plan shape → capacity → /alerts vs CRITICAL aircraft → /models profiles + 401-deny), PASS/FAIL summary, nonzero exit on failure: **13/13 PASS live, 13/13 PASS after full reset, 0/13 FAIL with clean hints when services are down.**
- `demo/demo_script.md` — timed 3-minute presentation narrative with fallback instructions.
- `demo/jetson_smoke.py` + `docs/JETSON_FIELD_NODE.md` — stdlib-only fixture verifier and the Jetson-class field-node deployment path (see §6.1).
- `docs/screenshots/` — capture checklist for slides.

## 7. Operational impact

AARIS converts raw engine telemetry into three operational decisions:

0. **One headline number for decision-makers** — the dashboard's **Fleet Availability** KPI (not CRITICAL and not in maintenance today / full roster). On a freshly reset demo fleet it reads **80.0% (72/90)**, computed live from fleet state + the plan's day-1 shop bookings, with the caption surfacing exactly how it was derived (18 grounded, the 8 airframes booked into hangars on plan day 1 — all already CRITICAL, 12 unreported treated as not-CRITICAL). The value moves with the plan day (e.g. **71.1% = 64/90** six days later, when 8 non-CRITICAL airframes are in the hangar). This is the fleet-availability language the MoD audience plans in; the number moves the moment a real report or plan changes.

1. **Continue operations** for healthy aircraft — LOW-band aircraft are explicitly deferred by the planner with a stated reason rather than silently dropped.
2. **Monitor or inspect** aircraft entering elevated-risk bands — MEDIUM/HIGH bands map to recommended actions on the edge report, visible per aircraft in the dashboard.
3. **Prioritize limited maintenance capacity** for aircraft approaching RUL thresholds — CRITICAL/HIGH aircraft are scheduled before their derived deadlines under a hard daily-capacity constraint.

In the demonstration fleet, the scheduler produced a feasible 30-day plan for all 90 aircraft while respecting a maximum of eight concurrent maintenance tasks per day: **solver OPTIMAL, 54 scheduled / 36 deferred, peak load exactly 8/day (day 7 has 6)**. AC-001 (RUL 2.4 cycles) was scheduled on day 1 — exactly its `floor(RUL×0.8)` deadline; the no-spare aircraft AC-003 was held to day 5 with the trade-off documented in its deferral reason. Every one of the 90 aircraft ends with a status, a day (or a reason there isn't one), and a human-readable rationale — an operator can audit *why* the plan looks the way it does.

## 8. Bugs we caught in our own work (kept, not hidden)

- NaN-echo crash path in FastAPI's default 422 encoder → custom sanitizer (ground + edge).
- `plan_id` stamped only on the POST response, not in the persisted plan → re-persisted; POST and `/schedule/latest` now byte-identical.
- First CP-SAT attempt used a non-existent `AddExactlyTrue` API; a too-eager "fix" (`sum == len(days)`) would have forced every day true — caught in review, correct `AddExactlyOne` in the fallback phase.
- Streamlit `use_container_width` deprecation replaced mid-build after the runtime warning surfaced.
- Broken README one-liner (TestClient without lifespan) caught by the smoke test that then existed.
- A stale verification path (`run_seed42_main` history) nearly mis-validated the no_pe result — the matched-budget comparison was rebuilt from the actual history file.

## 9. Honest limitations

- Dataset is a non-standard FD001 variant; absolute RMSE is tied to this copy, and 11/100 engines have truth > 125 that no clipped model can fit (equal floor for all models).
- Ridge overestimation rate on the test set is 0.54 — it guesses "healthy too long" slightly more than half the time; the asymmetric penalty convention is documented, and the risk bands were chosen to be conservative around this.
- The fleet, bases, priorities, and spares in `fleet_config.json`/`seed_demo.py` are synthetic — a demo rig for the pipeline, not real airline data.
- All validation-only figures now come from the leak-free engine-wise re-run (§5); the pre-audit window-level numbers (4.92 etc.) were leak-inflated and are cited only as the finding being corrected. Nothing has been evaluated on test, and nothing downstream claims any Transformer variant.
- Dashboard is read-only by design (scheduling stays on the ground station). In
  LIVE mode only the critical-alerts banner auto-refreshes (~20 s); the rest of
  the page updates on interaction or via **Refresh**.
- Demo mode serves a snapshot; after any real state change, untick it.

## 10. Reproduce everything

```bash
# ML evidence
.venv/Scripts/python.exe experiments/baselines.py
.venv/Scripts/python.exe ground/smoke_test.py          # full ground+security+scheduler smoke, temp DB
.venv/Scripts/python.exe results/ridge_model/predict_rul.py   # 112.76 standalone

# full stack + rehearsal
powershell -ExecutionPolicy Bypass -File demo/run_demo.ps1 -Reset
.venv/Scripts/python.exe demo/verify_pipeline.py       # expect 13 PASS
stop.bat
```

## 11. What's next (user-side decisions)

1. **Final presentation and demo video** using `demo/demo_script.md` (system is rehearsal-complete today; screenshots for slides per `docs/screenshots/README.md`).
2. One reserved **test-set evaluation of `no_pe`** — say the word and it runs, RESULTS.md updates, and only then does any superiority language change.
3. Optional directions after that: real telemetry ingestion at the edge (replace the fixture), multi-fleet scheduling, or plan-sensitivity views (what-if for `max_daily`).

---

*Artifacts: `RESULTS.md` (ML detail), `edge/README.md`, `ground/README.md`, `dashboard/README.md`, `demo/demo_script.md`, frozen archive `results/frozen_baseline_2026-10-11.zip`. The ML numbers in §4–§5 are quoted from those files and were re-verified on 2026-10-11; the system figures in §6–§7 (90-aircraft fleet, plan, KPI, section list) were re-verified on 2026-10-17 against the shipped 90-aircraft build.*
