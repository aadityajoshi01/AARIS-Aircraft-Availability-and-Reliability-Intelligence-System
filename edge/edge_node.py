"""AARIS — C-MAPSS Ridge RUL edge node.

A small FastAPI service that turns a raw 30-cycle x 24-feature turbofan sensor
window into a Remaining-Useful-Life (RUL) estimate plus fleet-ops fields
(engine health, risk band, top issues, recommended action).

Inference reuses the exact logic from `predict_rul.py`:
    MinMaxScaler.transform -> flatten 30x24 -> Ridge.predict -> clip to [0, max_rul]

Model artifacts under `edge/models/ridge_model/` are FROZEN. This node never
retrains or mutates them; it only loads them once at startup.

Run (from the `edge/` directory):
    uvicorn edge_node:app --reload
"""

from __future__ import annotations

import json
import os
import pickle
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

# --------------------------------------------------------------------------
# Artifact locations (resolved relative to this file so --reload & cwd vary OK)
# --------------------------------------------------------------------------
EDGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EDGE_DIR)
ARTIFACT_DIR = os.path.join(EDGE_DIR, "models", "ridge_model")
RIDGE_PKL = os.path.join(ARTIFACT_DIR, "ridge_model.pkl")
SCALER_PKL = os.path.join(ARTIFACT_DIR, "scaler.pkl")
FEATURE_ORDER_JSON = os.path.join(ARTIFACT_DIR, "feature_order.json")
SAMPLE_INPUT_JSON = os.path.join(ARTIFACT_DIR, "sample_input.json")

# Conformal calibration sidecar (see experiments/conformal_ridge.py). Read-only,
# optional: predictions work without it; when present, /predict gains an
# additive "uncertainty" block. Point-estimate risk bands are unchanged.
CONFORMAL_PATH = os.path.join(REPO_DIR, "conformal", "risk_calibration.json")
CONFORMAL_LEVEL = "90"  # headline interval level

# Expected input contract (authoritative values come from feature_order.json)
EXPECTED_SEQ_LEN = 30
EXPECTED_N_FEATURES = 24

# In-memory store of the latest prediction per aircraft_id (for GET /report)
_STORE: dict[str, dict] = {}
_STORE_LOCK = threading.Lock()

# Loaded artifacts live here, populated exactly once at startup by the lifespan.
_ARTIFACTS: dict | None = None

# ---------------------------------------------------------------------------
# Optional ground forwarding (OFF unless GROUND_PUSH_URL is set, so the local
# demo and jetson_smoke stay unaffected). The device bearer token is only ever
# placed in the Authorization header — it is never written to logs.
# ---------------------------------------------------------------------------
EDGE_DEVICE_TOKEN = os.environ.get("EDGE_AC017_TOKEN", "edge-ac017-dev-token")
GROUND_PUSH_URL = os.environ.get("GROUND_PUSH_URL", "").strip()


def _push_report_to_ground(report: dict) -> None:
    """Best-effort POST of one /predict response to the ground station."""
    import urllib.request

    try:
        req = urllib.request.Request(
            GROUND_PUSH_URL,
            data=json.dumps(report).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {EDGE_DEVICE_TOKEN}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            code = resp.status
        print(f"[edge] ground push -> HTTP {code} (aircraft={report.get('aircraft_id')})")
    except Exception as exc:  # noqa: BLE001 — push failure must never break /predict
        print(f"[edge] ground push failed: {type(exc).__name__}: {str(exc)[:120]}")

# Conformal calibration dict, or None when the sidecar file is absent/invalid.
_CONFORMAL: dict | None = None


def _load_conformal() -> dict | None:
    """Load the conformal sidecar once at startup; absence is tolerated."""
    try:
        with open(CONFORMAL_PATH, "r", encoding="utf-8") as f:
            cal = json.load(f)
        if "halfwidths_cycles" not in cal or CONFORMAL_LEVEL not in cal["halfwidths_cycles"]:
            print(f"[edge] conformal sidecar lacks level {CONFORMAL_LEVEL}; disabling uncertainty block")
            return None
        return cal
    except FileNotFoundError:
        print(f"[edge] conformal sidecar not found at {CONFORMAL_PATH}; predictions will omit 'uncertainty'")
        return None
    except (ValueError, OSError) as exc:
        print(f"[edge] conformal sidecar unreadable ({exc}); disabling uncertainty block")
        return None


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_artifacts_once() -> dict:
    """Load model + scaler + feature-order once; record integrity/diagnostic status."""
    with open(RIDGE_PKL, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_ORDER_JSON, "r") as f:
        meta = json.load(f)

    cols = meta["feature_order"]
    seq_len = int(meta["sequence_length"])
    max_rul = float(meta["max_rul"])

    return {
        "model": model,
        "scaler": scaler,
        "meta": meta,
        "cols": cols,
        "n_features": len(cols),
        "seq_len": seq_len,
        "max_rul": max_rul,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "integrity": {
            "ridge_model.pkl": {"path": RIDGE_PKL, "exists": os.path.exists(RIDGE_PKL),
                                 "sha256": _sha256(RIDGE_PKL) if os.path.exists(RIDGE_PKL) else None},
            "scaler.pkl": {"path": SCALER_PKL, "exists": os.path.exists(SCALER_PKL),
                            "sha256": _sha256(SCALER_PKL) if os.path.exists(SCALER_PKL) else None},
            "feature_order.json": {"path": FEATURE_ORDER_JSON, "exists": os.path.exists(FEATURE_ORDER_JSON),
                                    "sha256": _sha256(FEATURE_ORDER_JSON) if os.path.exists(FEATURE_ORDER_JSON) else None},
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts exactly once at startup."""
    global _ARTIFACTS, _CONFORMAL
    _ARTIFACTS = _load_artifacts_once()
    app.state.artifacts = _ARTIFACTS
    _CONFORMAL = _load_conformal()
    app.state.conformal = _CONFORMAL
    print(f"[edge] loaded Ridge artifacts from {ARTIFACT_DIR} "
          f"(seq_len={_ARTIFACTS['seq_len']}, features={_ARTIFACTS['n_features']}, "
          f"max_rul={_ARTIFACTS['max_rul']}; conformal={'loaded' if _CONFORMAL else 'absent'})")
    yield
    print("[edge] shutting down")


app = FastAPI(
    title="AARIS RUL Edge Node",
    description="Edge inference for the frozen C-MAPSS Ridge RUL model.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_artifacts() -> dict:
    if _ARTIFACTS is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded yet; node is warming up.")
    return _ARTIFACTS


# --------------------------------------------------------------------------
# Input parsing / validation
# --------------------------------------------------------------------------
def _parse_ids(payload: dict) -> tuple[object, object]:
    """Pull aircraft_id / engine_id from top level, else metadata block."""
    md = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    aircraft_id = payload.get("aircraft_id", md.get("aircraft_id"))
    engine_id = payload.get("engine_id", md.get("engine_id"))
    if aircraft_id is None:
        aircraft_id = "UNKNOWN"
    return aircraft_id, engine_id


def _validate_window(raw_window) -> np.ndarray:
    """Return a float64 (30, 24) ndarray or raise a clear 422."""
    if raw_window is None:
        raise HTTPException(status_code=422, detail="Missing required field 'window'.")
    if not isinstance(raw_window, list):
        raise HTTPException(status_code=422,
                            detail="'window' must be a list of 30 rows, each a list of 24 numbers.")

    if len(raw_window) != EXPECTED_SEQ_LEN:
        raise HTTPException(status_code=422,
                            detail=f"'window' must have exactly {EXPECTED_SEQ_LEN} rows (cycles); got {len(raw_window)}.")

    for r, row in enumerate(raw_window):
        if not isinstance(row, list):
            raise HTTPException(status_code=422,
                                detail=f"'window' row {r} must be a list of {EXPECTED_N_FEATURES} numbers.")
        if len(row) != EXPECTED_N_FEATURES:
            raise HTTPException(status_code=422,
                                detail=f"'window' row {r} must have exactly {EXPECTED_N_FEATURES} values; got {len(row)}.")
        for c, v in enumerate(row):
            # bool is a subclass of int in Python — reject it explicitly.
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise HTTPException(status_code=422,
                                    detail=f"'window[{r}][{c}]' = {v!r} is not numeric; all values must be int/float.")

    try:
        arr = np.asarray(raw_window, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded by checks above
        raise HTTPException(status_code=422, detail=f"'window' is not fully numeric: {exc}")

    if arr.shape != (EXPECTED_SEQ_LEN, EXPECTED_N_FEATURES):
        raise HTTPException(status_code=422,
                            detail=f"'window' must be shape [{EXPECTED_SEQ_LEN}, {EXPECTED_N_FEATURES}]; got {list(arr.shape)}.")
    if not np.isfinite(arr).all():
        bad = [[int(r), int(c)] for r, c in np.argwhere(~np.isfinite(arr))][:10]
        raise HTTPException(status_code=422,
                            detail=f"'window' contains non-finite values (NaN/Inf) at positions {bad}.")
    return arr


# --------------------------------------------------------------------------
# Derived fleet-ops fields
# --------------------------------------------------------------------------
def _risk_band(rul_cycles: float) -> str:
    if rul_cycles >= 100:
        return "LOW"
    if rul_cycles >= 60:
        return "MEDIUM"
    if rul_cycles >= 30:
        return "HIGH"
    return "CRITICAL"


_ACTION_BY_RISK = {
    "LOW": "Routine monitoring; no action. Fly per scheduled maintenance program.",
    "MEDIUM": "Add to planned-maintenance queue; inspect at next convenient turnaround.",
    "HIGH": "Book maintenance intervention within the current cycle window; stage parts.",
    "CRITICAL": "Ground the engine. Do not dispatch until inspected and cleared by maintenance.",
}


def _top_issues(art: dict, scaled: np.ndarray, limit: int = 5) -> list[dict]:
    """Rank sensors by how much they reduce predicted RUL (Ridge coefficient x scaled value).

    coef_ is ordered as the flattened (30, 24) window, so reshape back to
    (timesteps, features) and sum each feature's contribution across the window.
    Multiplying by max_rul expresses the contribution in cycles. Negative =
    the sensor pushes predicted life DOWN, i.e. it is a degradation driver.
    """
    model = art["model"]
    cols = art["cols"]
    max_rul = art["max_rul"]
    coef = np.asarray(model.coef_).reshape(art["seq_len"], art["n_features"])
    contrib_cycles = (scaled * coef).sum(axis=0) * max_rul  # (24,)

    order = np.argsort(contrib_cycles)  # most negative first
    issues = []
    for f in order:
        if contrib_cycles[f] >= 0:
            break
        if len(issues) >= limit:
            break
        issues.append({
            "sensor": cols[f],
            "impact_cycles": round(float(contrib_cycles[f]), 2),
            "note": "reduces predicted RUL (degradation driver)",
        })
    return issues


# --------------------------------------------------------------------------
# Conformal uncertainty block (additive; point-estimate fields unchanged)
# --------------------------------------------------------------------------
_GROUNDING_THRESHOLD_CYCLES = 30.0  # CRITICAL band boundary: rul < 30 => ground


def _uncertainty_block(rul_cycles: float, risk: str, max_rul: float) -> dict:
    """90% conformal interval + set-membership statement vs the grounding
    threshold, from the calibration sidecar. Never alters the point estimate."""
    if _CONFORMAL is None:
        return {"available": False,
                "reason": f"conformal sidecar not loaded (expected {CONFORMAL_PATH})"}
    hw = float(_CONFORMAL["halfwidths_cycles"][CONFORMAL_LEVEL])
    lo = round(max(0.0, rul_cycles - hw), 2)
    hi = round(min(max_rul, rul_cycles + hw), 2)

    if hi < _GROUNDING_THRESHOLD_CYCLES:
        grounding = "interval_below_grounding_threshold"
    elif lo >= _GROUNDING_THRESHOLD_CYCLES:
        grounding = "interval_above_grounding_threshold"
    else:
        grounding = "interval_crosses_grounding_threshold"

    band_diag = (_CONFORMAL.get("band_diagnostics_90") or {}).get(risk) or {}
    return {
        "available": True,
        "method": _CONFORMAL.get("method"),
        "level": int(CONFORMAL_LEVEL),
        "halfwidth_cycles": hw,
        "interval_cycles": [lo, hi],
        "empirical_coverage_oof": _CONFORMAL.get("empirical_coverage_oof", {}).get(CONFORMAL_LEVEL),
        "grounding_assessment": grounding,
        "grounding_threshold_cycles": _GROUNDING_THRESHOLD_CYCLES,
        "point_band_coverage_90": band_diag.get("empirical_coverage_90_clipped"),
        "source": "conformal/risk_calibration.json",
        "caveat": ("cross-conformal heuristic calibrated on FD001 training engines; "
                   "interval does not change the point-estimate risk band"),
    }


# --------------------------------------------------------------------------
# Core inference (mirrors predict_rul.py)
# --------------------------------------------------------------------------
def _infer_rul(art: dict, window: np.ndarray) -> dict:
    cols = art["cols"]
    max_rul = art["max_rul"]

    scaled = art["scaler"].transform(pd.DataFrame(window, columns=cols))
    pred_norm = art["model"].predict(scaled.reshape(1, -1))[0]
    rul_cycles = float(np.clip(pred_norm * max_rul, 0.0, max_rul))

    engine_health = float(np.clip((rul_cycles / max_rul) * 100.0, 0.0, 100.0))
    risk = _risk_band(rul_cycles)

    return {
        "rul_cycles": round(rul_cycles, 2),
        "engine_health": round(engine_health, 2),
        "risk": risk,
        "recommended_action": _ACTION_BY_RISK[risk],
        "top_issues": _top_issues(art, scaled),
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    """Artifact + expected-input-shape status."""
    art = _require_artifacts()
    integrity = art["integrity"]
    expected_ok = (art["seq_len"] == EXPECTED_SEQ_LEN and art["n_features"] == EXPECTED_N_FEATURES)
    return {
        "status": "ok" if all(v["exists"] for v in integrity.values()) else "degraded",
        "model": {"loaded": art["model"] is not None,
                  "type": type(art["model"]).__name__,
                  "alpha": float(getattr(art["model"], "alpha", float("nan")))},
        "scaler": {"loaded": art["scaler"] is not None,
                   "type": type(art["scaler"]).__name__,
                   "n_features_in": int(getattr(art["scaler"], "n_features_in_", art["n_features"]))},
        "feature_order": {"loaded": bool(art["cols"]), "count": art["n_features"],
                          "first_3": art["cols"][:3]},
        "expected_input_shape": [EXPECTED_SEQ_LEN, EXPECTED_N_FEATURES],
        "artifacts_consistent_with_contract": expected_ok,
        "artifacts_dir": ARTIFACT_DIR,
        "integrity": integrity,
        "conformal": {
            "loaded": _CONFORMAL is not None,
            "path": CONFORMAL_PATH,
            "level": int(CONFORMAL_LEVEL),
            "halfwidths_cycles": _CONFORMAL["halfwidths_cycles"] if _CONFORMAL else None,
            "empirical_coverage_oof": _CONFORMAL.get("empirical_coverage_oof") if _CONFORMAL else None,
        },
        "loaded_at": art["loaded_at"],
    }


@app.post("/predict")
async def predict(request: Request):
    """Predict RUL + fleet-ops fields from a raw 30x24 sensor window."""
    art = _require_artifacts()

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Request body must be valid JSON: {exc}")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422,
                            detail="Body must be a JSON object with a 'window' key "
                                   "(e.g. {\"window\": [[...24...], x30], \"aircraft_id\": \"AC-017\", \"engine_id\": 1}).")

    window_raw = payload.get("window", payload.get("sensor_window"))
    arr = _validate_window(window_raw)
    aircraft_id, engine_id = _parse_ids(payload)

    result = _infer_rul(art, arr)
    timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()

    response = {
        "aircraft_id": aircraft_id,
        "engine_id": engine_id,
        "timestamp": timestamp,
        "rul_cycles": result["rul_cycles"],
        "engine_health": result["engine_health"],
        "risk": result["risk"],
        "top_issues": result["top_issues"],
        "recommended_action": result["recommended_action"],
        "uncertainty": _uncertainty_block(result["rul_cycles"], result["risk"], art["max_rul"]),
        "model": {
            "name": "C-MAPSS Ridge RUL",
            "type": type(art["model"]).__name__,
            "alpha": float(getattr(art["model"], "alpha", float("nan"))),
            "sequence_length": art["seq_len"],
            "feature_count": art["n_features"],
            "max_rul": art["max_rul"],
            "artifact": "ridge_model.pkl",
        },
        "input_window_shape": [EXPECTED_SEQ_LEN, EXPECTED_N_FEATURES],
    }

    # Store as the latest reading for this aircraft (used by GET /report).
    with _STORE_LOCK:
        _STORE[str(aircraft_id)] = {
            "aircraft_id": aircraft_id,
            "engine_id": engine_id,
            "timestamp": timestamp,
            "prediction": response,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }

    # Optional authenticated forward to the ground station (off by default).
    if GROUND_PUSH_URL:
        threading.Thread(target=_push_report_to_ground, args=(response,),
                         daemon=True).start()

    return response


@app.get("/report")
def report(aircraft_id: str):
    """Latest stored prediction for an aircraft + artifact/diagnostic status."""
    art = _require_artifacts()
    with _STORE_LOCK:
        rec = _STORE.get(str(aircraft_id))

    diagnostics = {
        "model_loaded": art["model"] is not None,
        "scaler_loaded": art["scaler"] is not None,
        "feature_order_loaded": bool(art["cols"]),
        "expected_input_shape": [EXPECTED_SEQ_LEN, EXPECTED_N_FEATURES],
        "artifacts_dir": ARTIFACT_DIR,
        "integrity_ok": all(v["exists"] for v in art["integrity"].values()),
        "loaded_at": art["loaded_at"],
        "stored_predictions": len(_STORE),
    }

    if rec is None:
        return {
            "aircraft_id": aircraft_id,
            "available": False,
            "message": (f"No stored prediction for aircraft_id='{aircraft_id}'. "
                        "POST /predict with this aircraft_id first."),
            "diagnostics": diagnostics,
        }

    return {
        "aircraft_id": aircraft_id,
        "available": True,
        "latest": rec,
        "diagnostics": diagnostics,
    }
