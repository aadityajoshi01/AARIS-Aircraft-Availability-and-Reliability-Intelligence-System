# AARIS on a Jetson — aircraft-side AHMS field node

**Story (one line):** aircraft sensors → **Jetson edge node** → Ridge RUL inference → compact
health report → ground station → fleet state + CP-SAT maintenance plan → dashboard.

The Jetson stands in for the aircraft-side Aircraft Health Management System (AHMS) edge
processor. This is credible for a specific engineering reason: the deployed model is a frozen
Ridge regression — inference is **one matrix multiply on CPU** (no GPU, no torch, model +
scaler + feature list ≈ a few hundred KB). A Jetson (Nano/Orin class), a Raspberry Pi, or any
small x86 box runs the exact same `edge/edge_node.py` service. The whole point of the Ridge
deployment choice (see `docs/PROJECT_REPORT.md` §4) was that the accuracy winner is also the
cheapest to run at the edge.

> **Honest status (re-verified 2026-10-17 against the 90-aircraft build):** the edge node is
> verified on this Windows CPU dev machine —
> `demo/verify_pipeline.py` passes 13/13 against it, and `demo/jetson_smoke.py` passes locally.
> It has **not yet been run on Jetson hardware**. This document is the complete, ready-to-run
> deployment path; once one Jetson boots the service and `jetson_smoke.py` prints PASS with
> 112.76 cycles, the field-node claim is substantiated with zero changes to the model.

## 1. What to copy to the Jetson

From this repo, two things are needed — nothing else:

```
edge/edge_node.py                    # the service
edge/models/ridge_model/             # frozen artifacts (pkl + json), SHA256-fixed
  ridge_model.pkl  scaler.pkl  feature_order.json  sample_input.json
conformal/risk_calibration.json      # optional uncertainty sidecar — keep it as <repo>/conformal/
                                     # next to edge/ so the node finds it (predictions still
                                     # work without it; /predict just omits the uncertainty block)
demo/jetson_smoke.py                 # verification (stdlib-only, no pip needed for it)
```

Copy with `scp -r` or a USB stick. Do **not** retrain or regenerate anything: the artifacts are
byte-identical frozen files (`/health` publishes their SHA256s — they must match the values in
`edge/README.md`).

## 2. Environment (Ubuntu aarch64, JetPack)

**Version pinning is not optional.** `ridge_model.pkl` and `scaler.pkl` were pickled with
scikit-learn 1.9.1 / numpy 2.5.3 (see `results/frozen_baseline_2026-10-11/stage/environment.json`).
Unpickling sklearn objects with a different major version raises compatibility errors or silently
misbehaves. Install exactly:

```bash
sudo apt-get install -y python3-venv python3-dev build-essential
python3 -m venv ~/aeroguard-venv
source ~/aeroguard-venv/bin/activate
pip install numpy==2.5.3 scikit-learn==1.9.1 pandas==3.0.6 fastapi "uvicorn[standard]" requests
# Pickle-critical pins: scikit-learn + numpy (the .pkl files). pandas is imported by
# edge_node.py for the in-flight DataFrame only — any recent version works, pinned for
# fidelity. numpy/scikit-learn/pandas all ship manylinux_aarch64 wheels; on JetPack 5/6
# (aarch64) pip installs them as binaries. If a wheel ever misses, pip builds from
# source (slow but works).
```

## 3. Run the edge node

```bash
cd <wherever>/edge                    # same layout as start.bat: run from inside edge/
~/aeroguard-venv/bin/python -m uvicorn edge_node:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` lets the ground station/laptop reach it (on the dev machine the launchers bind
127.0.0.1 only; on the bench, keep it on a trusted network). Check from any machine:

```bash
curl http://JETSON_IP:8000/health     # must list model/scaler SHA256s matching the manifest
```

## 4. Prove the fixture (the demo money-shot)

On the Jetson itself (or from the laptop with `--url http://JETSON_IP:8000/predict`):

```bash
~/aeroguard-venv/bin/python demo/jetson_smoke.py
# expect: PASS  /predict on this device returned the frozen fixture:
#         rul_cycles=112.76  engine_health=90.21  risk=LOW  (N ms round trip)
```

Tolerance is ±0.01 cycles: BLAS floating-point paths differ slightly across CPU
architectures, so the third decimal may move — the prediction, the risk band, and the
`top_issues` ranking are the same. Same artifacts + same math = same decision, which is
exactly what an edge deployment must guarantee.

## 5. Close the loop to the ground station

From the Jetson, send a real report the way an aircraft would (this is the `--POST
/ingest_report-->` arrow in the architecture diagram):

```bash
# run these from the directory holding the copied edge/ folder
SAMPLE=edge/models/ridge_model/sample_input.json
# The ground station is access-controlled: this device's edge token
# (EDGE_AC017_TOKEN, dev default edge-ac017-dev-token) is bound to AC-017 and
# is the ONLY credential that may ingest for it. The token must match the
# ground's environment; never print or log it.
DEVICE_TOKEN=edge-ac017-dev-token
# 1) edge predicts on the frozen window          2) ground ingests it
curl -s http://127.0.0.1:8000/predict \
  -X POST -H "Content-Type: application/json" \
  --data @$SAMPLE \
  | curl -s -X POST http://GROUND_IP:8001/ingest_report \
      -H "Authorization: Bearer $DEVICE_TOKEN" \
      -H "Content-Type: application/json" --data @-
```

Then `POST /schedule` on the ground from the laptop (needs a `maintenance_officer`
JWT — `POST /auth/login` with the maintainer demo user first; see
`../ground/README.md` → Security; or just use `run_demo.ps1 -Reset`, which logs
in and schedules for you) and watch the aircraft's state/plan update in the
Streamlit dashboard — an inference made on Jetson hardware, landing in the
fleet plan.

## 6. Presentation framing

- Say "**field-node architecture**": the edge service is a self-contained CPU-only artifact
  bundle, identical on every platform; the Jetson profile documents the aarch64 path.
- If the Jetson is live at demo time: run §4 + §5 on stage and say the loop closed on the
  device. If it is not: show `demo/jetson_smoke.py` PASS on the dev machine and describe §1–§5
  as the deployment path — do **not** claim hardware-verified results you don't have.
- One-line answer to "why not the Transformer on the Jetson?": Ridge won the head-to-head on
  RMSE under both metric conventions *and* runs as a single matrix multiply with inspectable
  coefficients — the Transformer costs more compute for worse accuracy on this dataset variant.

## 7. Real-time single-aircraft demo (15-min feed + urgent critical alerts)

For the live bench test the Jetson is treated as **one aircraft (AC-017)** that keeps the
ground station's view of it updating in real time and raises an **urgent alert** the moment a
reading crosses into CRITICAL. This uses `edge/live_reporter.py`, which streams a **SYNTHETIC
degradation timeline** (not the frozen model) so the whole low → critical → alert path is
visible on cue — nothing here touches the Ridge artifacts or `/predict`.

The three moving parts, all on local ports:

1. **Base station (ground)** holds the fleet, mints login JWTs, and now keeps an `alerts`
   table: any CRITICAL report ingested auto-raises **one active alert per aircraft**
   (`GET /alerts`, `POST /alerts/{id}/ack`).
2. **The aircraft (reporter)** posts a health report on a routine cadence (default **15 min**)
   and, when a reading is CRITICAL, sends it **immediately** and keeps re-reporting on a short
   cadence until it recovers.
3. **The dashboard** signs in with an **ID + password** (`POST /auth/login` → commander JWT held
   only in the browser session), and auto-refreshes so a new report updates the profile and a
   new CRITICAL alert pops into the banner without a manual refresh.

### 7.1 Start the base station so the Jetson can reach it (LAN)

On the laptop/base-station machine, bind `0.0.0.0` so the Jetson on the same network can post:

```bash
cd ground
# local demo secret; GROUND_HOST=0.0.0.0 exposes :8001 on the LAN (demo/local only)
AEROGUARD_SECRET_KEY=demo-secret-for-local-testing GROUND_HOST=0.0.0.0 \
  ../.venv/Scripts/python.exe -m uvicorn ground_station:app --host 0.0.0.0 --port 8001
```

Find this machine's LAN IP (`ipconfig` on Windows / `hostname -I` on Linux) — call it
`GROUND_IP`. Confirm the device token for AC-017 matches the ground's env
(`EDGE_AC017_TOKEN`, dev default `edge-ac017-dev-token`). Every roster aircraft has
its own slot the same way — `EDGE_AC002_TOKEN` for AC-002, and so on — which is what
the whole-fleet feed in §7.2b relies on.

### 7.2 Run the reporter on the Jetson (or any one machine as the aircraft)

```bash
# from the directory holding edge/
python live_reporter.py --ground-url http://GROUND_IP:8001
#   -> posts a report every 15 min; sends CRITICAL immediately + re-checks every 60s
```

Watchable rehearsal (compress the cadence so a full low→critical→alert cycle plays in ~2–3 min):

```bash
python live_reporter.py --ground-url http://GROUND_IP:8001 --fast --ramp
```

Force a single urgent alert right now to demo the banner on cue:

```bash
python live_reporter.py --ground-url http://GROUND_IP:8001 --once --level CRITICAL
```

On the Windows dev machine the same rehearsal is one command — `start.bat --live-feed`
starts edge + ground + dashboard **and** the AC-017 reporter (`--fast --ramp`, pointed at
`http://127.0.0.1:8001`) in a minimized window; `stop.bat` shuts everything down including
the reporter. Use it to rehearse the whole flow before the Jetson is on the bench.

### 7.2b Give every aircraft its own live profile (`--fleet`)

One Jetson = one aircraft, which is the real deployment shape. To rehearse the whole
fleet at once on the dev machine, the reporter can stream a separate profile for every
roster tail number:

```bash
python live_reporter.py --ground-url http://GROUND_IP:8001 --fast --ramp --fleet
#   -> AC-001..AC-090, each with its OWN device token, staggered in time
#      and started at a different point of the synthetic timeline
python live_reporter.py --fast --ramp --fleet --fleet-count 6   # a smaller subset
```

Each aircraft authenticates with `EDGE_<AIRCRAFT>_TOKEN` (dev default
`edge-ac###-dev-token`), and the ground registers a token for every entry in
`ground/fleet_config.json`, so 90 feeds work with no extra config on localhost — while
the per-aircraft binding still holds (AC-001's token cannot report for AC-002, → 403).
It is one reporter process streaming all aircraft, not 90 processes.

`start.bat --fleet-feed` does the same in a minimized window on the dev machine.
Expect the canonical demo numbers to move: the fleet feed overwrites AC-017's
112.76 / LOW reading and reseeding (`demo\run_demo.ps1 -Reset`) restores it.

### 7.3 Sign in to the dashboard and watch it update

Open the dashboard, leave **DEMO MODE** off, and sign in with a demo user
(`commander` / `commander123`, or `maintainer` / `maintainer123`). Above the
section nav (**Fleet Readiness · Aircraft Models · Aircraft Detail · Maintenance Plan · About**)
sits a live critical-alerts banner; **Live auto-refresh** (on by default) pulls
new alerts and reflects a new aircraft report automatically.

Each alert row has **Acknowledge** and **Profile**, and clicking any table row —
readiness, next-7-days, deferred, or the plan tasks — opens that aircraft's
profile directly in the Aircraft Detail section. One click, no dropdown
hunting: on the Jetson demo this is how you jump from a CRITICAL alert to the
aircraft the commander has to decide about. Click **Acknowledge** on an alert
to clear it (`POST /alerts/{id}/ack`). An alert also closes itself when that aircraft's next
newer reading is no longer CRITICAL (`status=CLEARED`), so the banner lists exactly the aircraft
that are critical right now rather than every aircraft that was ever critical. Toggle
**Live auto-refresh** off to drive updates with the **Refresh** button instead.

> All reporter data is synthetic demonstration data — a scripted RUL timeline, not real IAF
> telemetry, and not an output of the frozen model. It exists to exercise the ingest → alert →
> dashboard path end to end.
