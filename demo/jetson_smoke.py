"""AARIS field-node smoke test — run this ON THE EDGE DEVICE (e.g. a Jetson).

Stdlib-only (urllib), so it runs in a bare venv with no extra packages.
Posts the frozen 30x24 sample window to a running edge node's /predict and
checks the response against the verified fixture (112.76 cycles / 90.21% /
LOW). The same window, the same frozen Ridge artifacts — this proves the
edge service behaves identically on the aircraft-side hardware.

Usage:
  # on the Jetson itself, against a local edge node:
  python3 demo/jetson_smoke.py
  # from any machine, against a Jetson on the LAN:
  python demo/jetson_smoke.py --url http://JETSON_IP:8000/predict
  # window file defaults to edge/models/ridge_model/sample_input.json (repo run);
  # on a device without the repo, point --input at a copy of it.

Exit code: 0 = PASS, 1 = FAIL (prints what mismatched).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(REPO_ROOT, "edge", "models", "ridge_model", "sample_input.json")

# Fixture verified offline (predict_rul.py) and on the Windows edge node.
EXPECTED = {"rul_cycles": 112.76, "engine_health": 90.21}
EXPECTED_RISK = "LOW"
# Cross-platform BLAS can move the 3rd decimal; 0.01 is the honest tolerance.
TOL = 0.01


def main() -> int:
    ap = argparse.ArgumentParser(description="Edge-node /predict fixture check.")
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict",
                    help="edge node /predict URL (default: local)")
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="path to sample_input.json (30x24 window payload)")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL  cannot read input window: {exc}")
        return 1

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.url, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"FAIL  HTTP {exc.code} from {args.url}: {exc.read()[:200]!r}")
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"FAIL  cannot reach edge node at {args.url}: {exc}")
        return 1

    problems = []
    for key, want in EXPECTED.items():
        got = out.get(key)
        if not isinstance(got, (int, float)) or abs(got - want) > TOL:
            problems.append(f"{key}: got {got!r}, want {want} +/- {TOL}")
    if out.get("risk") != EXPECTED_RISK:
        problems.append(f"risk: got {out.get('risk')!r}, want {EXPECTED_RISK!r}")

    if problems:
        print("FAIL  edge node returned unexpected values:")
        for p in problems:
            print(f"      - {p}")
        print(f"      (model={out.get('model', {}).get('name', '?')}, device-side response above)")
        return 1

    print("PASS  /predict on this device returned the frozen fixture:")
    print(f"      rul_cycles={out['rul_cycles']}  engine_health={out['engine_health']}  "
          f"risk={out['risk']}  ({elapsed_ms:.0f} ms round trip)")
    print(f"      model: {out.get('model', {}).get('name', '?')} {out.get('model', {}).get('version', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
