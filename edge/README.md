# AARIS — C-MAPSS Ridge RUL Edge Node

A small FastAPI service that scores a raw turbofan sensor window into a
Remaining-Useful-Life (RUL) estimate plus fleet-ops fields. It runs **inference
only** against the frozen Ridge artifacts — it never retrains or mutates them.

Inference reuses the exact logic from `../results/ridge_model/predict_rul.py`:

```
MinMaxScaler.transform(window) -> flatten 30x24 -> Ridge.predict -> clip to [0, 125]
```

## Layout

```
edge/
  edge_node.py                 # FastAPI app
  live_reporter.py             # SYNTHETIC live feed: streams one aircraft's
                               #   health reports to the ground on a cadence
                               #   (15-min routine + immediate on CRITICAL);
                               #   --fleet does that for every roster aircraft
  models/ridge_model/          # FROZEN artifacts (loaded once at startup)
    ridge_model.pkl            # Ridge(alpha=10.0) on flattened 30x24 window
    scaler.pkl                 # pipeline MinMaxScaler (fit on FD001 train)
    feature_order.json         # 24 columns, sequence_length=30, max_rul=125
    sample_input.json          # engine 1 last-30-cycle window (test fixture)
```

All four artifacts are byte-identical to the validated copies in
`../results/ridge_model/` (verified by SHA-256).

## Run

From the `edge/` directory (uses the repo virtualenv):

```bash
../.venv/Scripts/python.exe -m uvicorn edge_node:app --reload
# -> http://127.0.0.1:8000  (docs at /docs)
```

The startup log prints `Application startup complete` then
`[edge] loaded Ridge artifacts ... (seq_len=30, features=24, max_rul=125.0)`.

## Live demo feed — `live_reporter.py`

For the Jetson/real-time test, `live_reporter.py` treats one aircraft
(**AC-017**) as a live edge device that keeps the ground station's view of it
updating and raises urgent critical alerts. It streams a **SYNTHETIC degradation
timeline** (it does NOT call `/predict` or touch the frozen artifacts) so the
whole low → critical → alert path is visible on cue. It authenticates with the
device token (`EDGE_AC017_TOKEN`) and posts to `POST /ingest_report`.

```bash
# from edge/, ground running on :8001
../.venv/Scripts/python.exe live_reporter.py                 # 15-min cadence (900s)
../.venv/Scripts/python.exe live_reporter.py --fast --ramp   # watchable ~2-3 min cycle
../.venv/Scripts/python.exe live_reporter.py --once --level CRITICAL   # fire one urgent alert
# point at a base station on another machine (LAN):
../.venv/Scripts/python.exe live_reporter.py --ground-url http://GROUND_IP:8001
```

A CRITICAL report is sent immediately (not held for the next tick); the ground
then auto-raises one active alert per aircraft, which the dashboard shows in a
live banner. When a later, strictly-newer reading is no longer CRITICAL, that
aircraft's open alert is marked **CLEARED**, so the banner always lists exactly
the aircraft that are critical right now. See `../docs/JETSON_FIELD_NODE.md` §7
for the full bench procedure.

### Whole-fleet feed — `--fleet`

To put a live profile on **every** aircraft instead of only AC-017, `--fleet`
runs one independent synthetic feed per roster tail number
(`ground/fleet_config.json`, AC-001..AC-090: 6 aircraft models x 15 airframes).
Each aircraft authenticates with its **own** device token
(`EDGE_AC002_TOKEN` → `edge-ac002-dev-token` dev
default), starts at a different point of the degradation timeline, and reports
on a staggered clock, so the ground sees a continuous stream of per-aircraft
updates and critical alerts arrive one at a time rather than as one burst.

```bash
../.venv/Scripts/python.exe live_reporter.py --fast --ramp --fleet          # all 90
../.venv/Scripts/python.exe live_reporter.py --fast --ramp --fleet --fleet-count 6
../.venv/Scripts/python.exe live_reporter.py --interval 900 --fleet        # true 15-min cadence
```

It is one reporter process streaming all aircraft (not 90 processes).
`--fleet-count` must be within `[1, 90]`; `--fleet` + `--once` sends nothing
(`--once` wins, single aircraft). **This overwrites the canonical AC-017 demo
reading** (112.76 / 90.21 / LOW) and moves the plan/KPI numbers — restore the
demo state afterwards with
`powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Reset -NoBrowser`.
`start.bat --fleet-feed` launches the same feed next to the three services.

## Endpoints

| Method | Path       | Purpose |
|--------|------------|---------|
| GET    | `/health`  | Model, scaler, feature-order, and expected-input-shape status + artifact SHA-256 integrity. |
| POST   | `/predict` | Score a raw 30×24 window → RUL + fleet-ops fields. |
| GET    | `/report?aircraft_id=AC-017` | Latest stored prediction for an aircraft + artifact/diagnostic status. |

**POST /predict body** — accepts the `sample_input.json` shape directly:

```json
{ "window": [[...24 numbers...], ... 30 rows ...],
  "aircraft_id": "AC-017", "engine_id": 1, "timestamp": "optional ISO-8601" }
```

`aircraft_id` / `engine_id` may also be nested under `metadata`. Defaults:
`aircraft_id = "UNKNOWN"`, `timestamp = now(UTC)`.

**Response fields:** `aircraft_id, engine_id, timestamp, rul_cycles,
engine_health, risk, top_issues, recommended_action, uncertainty, model,
input_window_shape`.

- `engine_health = clip((rul_cycles / 125) * 100, 0, 100)`
- `risk`: `RUL ≥ 100 = LOW`, `60–99 = MEDIUM`, `30–59 = HIGH`, `< 30 = CRITICAL`
- `top_issues`: up to 5 sensors that most **reduce** predicted RUL, ranked by the
  Ridge model's own coefficient × scaled-value contribution (in cycles).

## Uncertainty (conformal, additive)

`/predict` also returns an `uncertainty` block built from the calibration
sidecar at `../conformal/risk_calibration.json` (produced by
`experiments/conformal_ridge.py` — cross-conformal, engine-wise 5-fold on the
frozen Ridge recipe):

```json
"uncertainty": { "available": true, "level": 90, "halfwidth_cycles": 27.19,
                 "interval_cycles": [85.57, 125.0],
                 "empirical_coverage_oof": 0.9001,
                 "grounding_assessment": "interval_above_grounding_threshold", ... }
```

- The interval is `point estimate ± halfwidth`, clipped to `[0, 125]`.
- `grounding_assessment` is a set-membership statement against the 30-cycle
  CRITICAL boundary: `interval_below_...` (robust ground),
  `interval_above_...` (robust fly), `interval_crosses_...` (borderline).
- **Point-estimate fields (`rul_cycles`, `engine_health`, `risk`) are unchanged
  by this block** — the fixture stays `112.76 / 90.21 / LOW`.
- If the sidecar is missing or malformed the node still serves predictions;
  `uncertainty.available = false` with the expected path, and `/health`
  reports `conformal.loaded = false`.

## Security (edge → ground reporting)

The edge node now carries a per-device credential for the ground station's
protected `POST /ingest_report` (see `../ground/README.md` → Security):

| Env var | Default | Meaning |
|---------|---------|---------|
| `EDGE_AC017_TOKEN` | `edge-ac017-dev-token` | Static bearer token bound to **AC-017**. Change it per real device (ground registers `EDGE_<AIRCRAFT>_TOKEN`). |
| `EDGE_<AIRCRAFT>_TOKEN` | `edge-ac###-dev-token` | One token per aircraft, bound to exactly that tail number. The ground registers all 90 roster entries this way (AC-001…AC-090), so `--fleet` needs no extra config on localhost. Set a real value **on both sides** for anything beyond the demo — the defaults are guessable and the ground logs a startup warning while they are in use. |
| `GROUND_PUSH_URL` | *(unset — push disabled)* | If set (e.g. `http://127.0.0.1:8001/ingest_report`), every `/predict` response is also POSTed to the ground station best-effort, with `Authorization: Bearer <EDGE_AC017_TOKEN>`. |

- The local `/predict` endpoint is **unchanged and needs no credentials** —
  the Jetson demo keeps working offline.
- A push failure never breaks `/predict` (best-effort, logged as HTTP status
  only). The token itself is **never printed or logged**.
- Nothing under `models/ridge_model/` or the prediction logic is touched by
  this layer.

## Input validation → 422

`/predict` returns **422** with a clear message when the window is: missing,
not a 30-row list, has any row that is not a 24-value list, contains a
non-numeric/boolean value, or contains NaN/Inf. Shape must be exactly `[30, 24]`.

## Test commands

**curl** — POST the sample fixture and check the two required numbers:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @models/ridge_model/sample_input.json
# expect "rul_cycles": 112.76, "engine_health": 90.21, "risk": "LOW"
```

**Python** — one-liner that asserts the verified values (works while the server
is running):

```bash
../.venv/Scripts/python.exe -c "import json,urllib.request as u; d=open('models/ridge_model/sample_input.json','rb').read(); r=json.load(u.urlopen(u.Request('http://127.0.0.1:8000/predict',data=d,headers={'Content-Type':'application/json'}))); print(r['rul_cycles'],r['engine_health']); assert r['rul_cycles']==112.76 and r['engine_health']==90.21; print('OK')"
```

**No-server smoke test** — boots the app in-process (the `with` block fires the
startup lifespan that loads the artifacts exactly once), then asserts — in
checks `[1]`–`[6]`, plus a `[5b]` NaN case — `GET /health` (200,
`expected_input_shape [30, 24]`, `conformal.loaded true`), the required
`/predict` response keys, the two required numbers and their risk band,
`GET /report?aircraft_id=AC-017` (`available true`, `diagnostics.integrity_ok
true`), the 422 shape/non-numeric/missing-window guards, and the conformal
`uncertainty` block; then exits:

```bash
../.venv/Scripts/python.exe smoke_test.py
# -> PASS: edge node verified — rul_cycles=112.76, engine_health=90.21,
#          all endpoints + 422 guards + conformal uncertainty OK
```

## Verified behavior

- `uvicorn edge_node:app --reload` starts clean; artifacts load exactly once.
- `POST /predict` with `sample_input.json` → `rul_cycles = 112.76`, `engine_health = 90.21`.
- `GET /report?aircraft_id=AC-017` → latest prediction + `diagnostics.integrity_ok = true`.
- Invalid windows → HTTP 422 with specific messages.

> Note: `top_issues` and `recommended_action` are derived heuristics for
> fleet-ops triage (Ridge coefficient contributions and risk-band mapping).
> `rul_cycles` / `engine_health` come strictly from the frozen model.
