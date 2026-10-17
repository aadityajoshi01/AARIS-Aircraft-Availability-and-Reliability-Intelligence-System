"""AARIS edge node — no-server smoke test.

Boots the FastAPI app in-process (the TestClient context manager runs the
startup lifespan that loads the frozen artifacts exactly once), then asserts:
  * GET /health is 200 and reports the expected [30, 24] input shape
  * POST /predict on sample_input.json -> rul_cycles == 112.76, engine_health == 90.21
  * risk band, shape, and required response keys
  * GET /report returns the latest stored prediction
  * invalid windows -> HTTP 422

Run (from edge/):  ../.venv/Scripts/python.exe smoke_test.py
"""

import json
import os

from fastapi.testclient import TestClient

import edge_node

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "models", "ridge_model", "sample_input.json")

REQUIRED_KEYS = {
    "aircraft_id", "engine_id", "timestamp", "rul_cycles", "engine_health",
    "risk", "top_issues", "recommended_action", "model", "input_window_shape",
}


def main() -> int:
    with open(SAMPLE) as f:
        sample = json.load(f)

    with TestClient(edge_node.app) as c:  # `with` fires the startup lifespan
        # 1) health
        h = c.get("/health")
        assert h.status_code == 200, h.text
        assert h.json()["expected_input_shape"] == [30, 24]
        print(f"[1] /health 200 | status={h.json()['status']} | shape={[30,24]}")

        # 2) predict sample
        r = c.post("/predict", json=sample)
        assert r.status_code == 200, r.text
        j = r.json()
        assert REQUIRED_KEYS.issubset(j.keys()), sorted(REQUIRED_KEYS - j.keys())
        print(f"[2] /predict {r.status_code} | keys OK ({len(j)} fields)")

        # 3) the two required numbers + band
        assert j["rul_cycles"] == 112.76, j["rul_cycles"]
        assert j["engine_health"] == 90.21, j["engine_health"]
        assert j["risk"] == "LOW" and j["input_window_shape"] == [30, 24]
        print(f"[3] rul_cycles={j['rul_cycles']} engine_health={j['engine_health']} risk={j['risk']}")

        # 4) report (store under AC-017 first)
        payload = dict(sample, aircraft_id="AC-017", engine_id=1)
        c.post("/predict", json=payload)
        rep = c.get("/report?aircraft_id=AC-017")
        assert rep.status_code == 200 and rep.json()["available"] is True
        assert rep.json()["latest"]["prediction"]["rul_cycles"] == 112.76
        assert rep.json()["diagnostics"]["integrity_ok"] is True
        print(f"[4] /report?AC-017 available=True integrity_ok=True")

        # 5) invalid inputs -> 422
        cases = {}
        cases["wrong_rows"] = dict(sample, window=sample["window"][:29])
        w2 = [list(map(float, row)) for row in sample["window"]]; w2[0] = w2[0][:23]
        cases["wrong_cols"] = dict(sample, window=w2)
        w3 = [list(map(float, row)) for row in sample["window"]]; w3[0][0] = "x"
        cases["non_numeric"] = dict(sample, window=w3)
        cases["missing_window"] = {"aircraft_id": "X"}

        for name, body in cases.items():
            bad = c.post("/predict", json=body)
            assert bad.status_code == 422, (name, bad.status_code, bad.text)
        print(f"[5] 422 guards OK for: {', '.join(cases)}")

        # 5b) non-finite (NaN) via raw JSON body
        w = [list(map(float, row)) for row in sample["window"]]; w[0][0] = float("nan")
        bad = c.post("/predict", content=json.dumps({"window": w}, allow_nan=True).encode(),
                     headers={"content-type": "application/json"})
        assert bad.status_code == 422, bad.text
        assert "[" in bad.json()["detail"]
        print(f"[5b] 422 NaN guard OK -> {bad.json()['detail'][:60]}...")

        # 6) conformal uncertainty block (additive; recalibration-safe checks)
        cal_path = os.path.join(HERE, os.pardir, "conformal", "risk_calibration.json")
        assert h.json()["conformal"]["loaded"] is True, "conformal sidecar not loaded in /health"
        with open(cal_path) as f:
            cal = json.load(f)
        hw = float(cal["halfwidths_cycles"]["90"])
        unc = j["uncertainty"]
        assert unc["available"] is True and unc["level"] == 90
        assert unc["halfwidth_cycles"] == round(hw, 2) or abs(unc["halfwidth_cycles"] - hw) < 1e-9
        lo, hi = unc["interval_cycles"]
        assert abs(lo - max(0.0, 112.76 - hw)) < 0.011, (lo, hw)
        assert abs(hi - min(125.0, 112.76 + hw)) < 0.011, (hi, hw)
        assert unc["grounding_assessment"] == "interval_above_grounding_threshold"
        # point-estimate fields must remain exactly the fixture values
        assert j["rul_cycles"] == 112.76 and j["risk"] == "LOW"
        print(f"[6] uncertainty OK | 90% interval [{lo}, {hi}] hw={hw} | "
              f"grounding={unc['grounding_assessment']}")

    print("\nPASS: edge node verified — rul_cycles=112.76, engine_health=90.21, "
          "all endpoints + 422 guards + conformal uncertainty OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
