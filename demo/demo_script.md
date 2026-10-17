# AARIS — 3-Minute Demo Rehearsal Script

A rehearsed narrative for presenting the live system. Everything below was
verified on 2026-10-17 with `demo/verify_pipeline.py` (all checks PASS, also
after a full database reset + reseed).

## Before you present (5 minutes earlier)

```powershell
# Windows, from the repo root — fresh, deterministic demo state:
powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset

# confirm the whole pipeline is healthy:
.venv\Scripts\python.exe demo\verify_pipeline.py
```

Expected: 13 PASS, exit 0. Have http://localhost:8501 open on the **Fleet
Readiness** section (signed in as `commander` / `commander123`).

Know the fallback: if anything about the API breaks mid-demo, tick
**DEMO MODE** in the dashboard sidebar — it renders from
`dashboard/demo_data.json`, a snapshot captured from this exact live system,
so every number you quote stays true even with all services stopped.

---

## The 3 minutes

### 0:00 – 0:30 · Problem

> "An airline's fleet is only worth something when it's in the air.
> Unplanned engine maintenance grounds aircraft with no warning: a shop slot
> is missed, a spare engine isn't staged, a high-priority route loses its
> plane. We built **AARIS** to turn raw engine telemetry into a
> day-by-day maintenance plan a planner can actually trust."

Show: the three minimized service windows (or just say they're running):
edge node on 8000, ground station on 8001, dashboard on 8501.

### 0:30 – 1:00 · Edge prediction (live)

From the repo root:

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data @edge/models/ridge_model/sample_input.json
```

> "This is one aircraft's edge node. It just took a 30-cycle window of real
> sensor data from a C-MAPSS engine run — 30 rows by 24 sensor/operating
> columns — and returned **112.76 remaining useful cycles**, **90.21% engine
> health**, risk band **LOW**. The model is a frozen Ridge regression —
> deliberately simple, validated with a test RMSE of about 16.2 cycles, and
> it runs in milliseconds on cheap hardware. One number we can defend beats
> a fancy number we can't. And the response carries an `uncertainty` block
> too: a 90% conformal interval — ±27.19 cycles, calibrated engine-wise on
> the frozen recipe — here **[85.6, 125] cycles**, comfortably clear of the
> 30-cycle grounding threshold. The point number still drives the band; the
> interval says how much to trust it."

Jetson framing (say when asked, or when a field node is live on the bench):

> "The aircraft-side profile is a Jetson-class AHMS edge processor. The whole
> edge node is the frozen artifact bundle plus one FastAPI service — pure
> CPU, one matrix multiply. `docs/JETSON_FIELD_NODE.md` is the exact aarch64
> deployment path, and `demo/jetson_smoke.py` (stdlib-only) proves the fixture
> on whatever device boots it: same 112.76 on the Jetson, or anywhere else."
>
> "For the real-time side: `start.bat --live-feed` makes AC-017 report on a
> compressed 15-minute cadence and push urgent CRITICAL alerts to the banner;
> `start.bat --fleet-feed` runs one feed **per aircraft** (AC-001..AC-090), each
> with its own device token and staggered, so all 90 profiles move live at once.
> Reset after with `run_demo.ps1 -Reset`."

Honesty guard: only claim "verified on Jetson hardware" after
`jetson_smoke.py` prints PASS on actual Jetson hardware. Until then say
"field-node deployment path prepared and verified on this machine."

### 1:00 – 1:30 · Ground station + fleet state

> "Each aircraft ships a compact JSON report like that one to the ground
> station, which stores every report in SQLite and keeps exactly one latest
> state per aircraft — 90 aircraft in this fleet."

Show: Dashboard → **Fleet Readiness**. Lead with the headline metric —
**Fleet Availability: 80.0% (72 of 90)** on a freshly reset demo — "aircraft
not in CRITICAL risk and not on the shop table today, computed live from the
same two endpoints the plan uses, never hardcoded; it moves with the plan day
(e.g. 71.1% = 64/90 six days later)." Point at the status column: every
aircraft reads AVAILABLE / GROUNDED (CRITICAL) / IN MAINTENANCE TODAY; the
8 airframes IN MAINTENANCE TODAY (plan day 1) are all already GROUNDED
(CRITICAL). Then **click the AC-001 row** — one click
opens that aircraft's profile in **Aircraft Detail** (no dropdown hunting):
RUL ≈ 2.4 cycles, risk CRITICAL, and its *Uncertainty (conformal 90% interval)*
reads **[0.0, 29.6] cycles** — even the optimistic end is below the 30-cycle
threshold, so grounding it is not a borderline call. The same click works on any
table on the page: the next-7-days list, the deferred list, the plan tasks, and
the **Profile** button on an alert. Now click **AC-014** in the deferred table:
"no data", an aircraft whose edge unit hasn't reported — the dashboard jumps to
it and says so plainly instead of guessing. The nav is a five-section control —
Fleet Readiness · Aircraft Models · Aircraft Detail · Maintenance Plan · About —
and **Aircraft Models** rolls the fleet up per type: 6 synthetic models × 15
airframes (Rafale EH, Su-30MKI, Tejas Mk1A, MiG-29UPG, Mirage 2000I/TI, Jaguar
IS/IM), each block the same health shape — 4 LOW / 3 MEDIUM / 3 HIGH / 3
CRITICAL plus 2 airframes with no telemetry, 12 ready to fly, 9 of 15 in the
plan — and clicking any airframe row there opens that aircraft's profile in one
click, exactly like the fleet tables.

### 1:30 – 2:15 · The 30-day plan

> "Now the operations question: with a shop that can handle at most **eight
> aircraft a day**, who goes when? A CP-SAT optimizer reads the fleet state
> and solves a 30-day plan. Every aircraft appears exactly once, scheduled
> **or** deferred with a written reason."

Show: Dashboard → **Maintenance Plan**.
- **54 scheduled, 36 deferred, solver status OPTIMAL** — the optimizer proves
  optimality, no black-box ranking.
- The Gantt: AC-001 sits on **day 1**, because its deadline is computed as
  `floor(RUL × 0.8)` = day 1. Urgent, high-mission-priority aircraft get the
  earliest slots.
- AC-003 has **no spare engine available** — it is kept from day 5 onward;
  the spare lead-time is a hard constraint, but only when every aircraft can
  still meet its deadline.
- The DEFERRED rows carry reasons: the 24 healthy LOW-risk aircraft ("no maintenance
  needed inside 30 days") and the 12 aircraft without telemetry.

### 2:15 – 3:00 · Honesty about what is what

> "Two kinds of numbers live here, and we label them where you see them.
> **RUL and engine health are model outputs** from the frozen Ridge model —
> retrained never, versioned, checksummed. **Top issues and recommended
> actions are operational heuristics**: the top-issues list comes from the
> Ridge coefficients weighted by the sensor window, and the action text is a
> standing playbook keyed to the risk band. Useful for triage — not a
> diagnosis. That distinction is stated in the About section on purpose."

Close: "One pipeline, three services, every number reproducible — you can run
the same rehearsal check we did: `demo\verify_pipeline.py`."

---

## Fallback if the API is offline mid-demo

1. Tick **DEMO MODE** in the dashboard sidebar. Every section renders from the
   bundled `demo_data.json` snapshot — same fleet, same 54/36 plan, same
   AC-017 values (112.76 / 90.21 / LOW). The snapshot also carries the
   critical-alerts list, so the alert banner, **Acknowledge** and the one-click
   **Profile** jump all still demo offline — acknowledgment is session-local and
   touches no station.
   (Note: dashboard **LIVE** mode signs in with an ID + password — use
   `commander` / `commander123`, which is read-only; the sidebar **Advanced**
   expander still accepts a pasted `DASHBOARD_SERVICE_TOKEN` JWT as a fallback.
   Without either, LIVE mode shows the login form instead of data, so
   **DEMO MODE stays the zero-credential dashboard path** for this demo. Ground
   curl calls authenticate via `POST /auth/login`, and the edge `curl` segment
   stays credential-free.)
2. The edge `curl` segment is the only part that needs the live edge node;
   if it is down, show the captured response in `ground/report_AC-017.json`
   instead — it is a real edge output, filed during verification.
3. After the demo: `powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset`
   restores the scripted demo state for the next rehearsal.

## Reproducing the verified numbers

| Fact | Source |
|---|---|
| 112.76 cycles / 90.21% / LOW for the sample window | frozen Ridge artifacts (`edge/models/ridge_model/`) |
| ±27.19-cycle 90% conformal interval (e.g. AC-001 → [0.0, 29.6]) | `conformal/risk_calibration.json` — regenerate with `experiments/conformal_ridge.py` (~15 s); dashboard recomputes interval = RUL ± halfwidth live |
| 90 aircraft roster (6 models × 15), 18/18/18/24 risk spread (+12 unreported) | `ground/fleet_config.json` + `ground/seed_demo.py` |
| 54 scheduled / 36 deferred, peak 8/day, OPTIMAL | CP-SAT via `POST /schedule` (deterministic: seed 7, 1 worker) — requires maintainer/admin JWT |
| 401 unauthenticated, 403 wrong role, per-aircraft device tokens (`edge-ac###-dev-token`, AC-001..AC-090) bound to their own airframe; audit trail with no secrets | `ground/security.py`, asserted by `ground/smoke_test.py` |
| Fleet Availability 80.0% (72/90) on a freshly reset demo (moves with the plan day, e.g. 71.1% = 64/90 six days later) | computed live in `dashboard/app.py::compute_readiness` from `GET /fleet` + `GET /schedule/latest` — recheck by running `demo/verify_pipeline.py` then opening the Fleet Readiness section |
| All of the above re-checked automatically | `demo/verify_pipeline.py` (13 PASS expected, incl. login, 401-deny, /alerts and /models checks) |
| Screenshots for slides | `docs/screenshots/` (capture list inside README) |

```powershell
# start / stop for rehearsal
powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1        # live mode
powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset # scripted demo fleet
stop.bat
```
