"""AARIS — capture docs/screenshots/*.png from a running stack (headless Chrome + CDP).

The dashboard is a live Streamlit app: it needs a signed-in session, a rendered
WebSocket frame, and a click to change section. This script drives a headless
Chrome over the Chrome DevTools Protocol so the pictures in the README are real
captures of the running system, not mock-ups.

    .venv\\Scripts\\python.exe demo\\capture_screenshots.py

Prerequisite: the stack is up (demo\\run_demo.ps1 or start.bat). Chrome/Edge is
found automatically on Windows; override with --chrome.

Signing in: the script types the demo credentials into the dashboard's own
login form (see --user/--password). If the dashboard was started with
DASHBOARD_SERVICE_TOKEN set it is already signed in and the form is skipped.

What it writes (all in docs/screenshots/):

    01_login.png                  the sign-in screen (no credentials yet)
    02_fleet_readiness.png        KPI row + availability table
    03_critical_alerts.png        the ACTIVE critical-alert banner + rows
    04_aircraft_models.png        one model profile card (type photo + metrics)
    05_model_airframes.png        that model's airframe table (plan status/day)
    06_aircraft_detail.png        one aircraft profile (AC-017 fixture)
    07_maintenance_plan.png       CP-SAT plan: Gantt, daily load, deferrals
    08_about.png                  how the demo works (data provenance)
    09_edge_docs.png              edge node OpenAPI contract (:8000/docs)
    10_ground_docs.png            ground station OpenAPI contract (:8001/docs)

Read-only against the services: it only GETs pages and clicks navigation
widgets. It never posts a report, never acks an alert, never touches the DB.
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import requests

try:
    import websockets  # ships with uvicorn[standard]; no extra install needed
except ImportError:  # pragma: no cover
    sys.exit("websockets is required (pip install 'uvicorn[standard]')")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "screenshots")

SECTIONS = ["Fleet Readiness", "Aircraft Models", "Aircraft Detail",
            "Maintenance Plan", "About"]


def find_chrome(explicit: str | None) -> str:
    """Locate a Chromium-family browser binary on Windows."""
    if explicit:
        if os.path.exists(explicit):
            return explicit
        sys.exit(f"--chrome not found: {explicit}")
    candidates = [
        os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES (X86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    sys.exit("no Chrome/Edge found — pass --chrome <path to chrome.exe>")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CDP:
    """Minimal async Chrome DevTools Protocol client (one page target)."""

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self._id = 0
        self.ws = None

    async def __aenter__(self):
        self.ws = await websockets.connect(self.ws_url, max_size=None)
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def call(self, method: str, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method,
                                       "params": params}))
        while True:  # skip event notifications until our reply arrives
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expression: str):
        expr = expression.strip()
        if expr.startswith("() =>") or expr.startswith("async () =>"):
            expr = f"({expr})()"   # evaluate the arrow function, not just define it
        res = await self.call("Runtime.evaluate", expression=expr,
                              returnByValue=True, awaitPromise=True)
        exc = res.get("exceptionDetails")
        if exc:
            detail = (exc.get("exception") or {}).get("description") or exc.get("text")
            raise RuntimeError(f"JS error: {detail}\n--- expression ---\n{expr}")
        return res.get("result", {}).get("value")

    async def goto(self, url: str):
        await self.call("Page.navigate", url=url)

    async def wait_for_text(self, needle: str, timeout: float = 30.0) -> bool:
        """Poll until `needle` appears in the rendered body text."""
        deadline = time.time() + timeout
        probe = ("() => document.body.innerText.indexOf(%s) !== -1"
                 % json.dumps(needle))
        while time.time() < deadline:
            if await self.js(probe):
                return True
            await asyncio.sleep(0.4)
        return False

    async def wait_for_selector(self, css: str, timeout: float = 30.0) -> bool:
        """Poll until at least one element matches `css`.

        Streamlit paints skeleton placeholders first and swaps in the real
        widgets a round-trip later, so waiting for body text alone is not
        enough before typing into a field.
        """
        deadline = time.time() + timeout
        probe = ("() => document.querySelectorAll(%s).length > 0"
                 % json.dumps(css))
        while time.time() < deadline:
            if await self.js(probe):
                return True
            await asyncio.sleep(0.4)
        return False

    async def click_text(self, needle: str, timeout: float = 12.0) -> bool:
        """Click the label/radio/button whose text equals `needle`."""
        deadline = time.time() + timeout
        probe = """
        () => {
          const want = %s;
          const nodes = [...document.querySelectorAll(
              'label, button, [role="radio"], [role="tab"]')];
          const hit = nodes.find(n => (n.textContent || '').trim() === want);
          if (!hit) return false;
          const input = hit.querySelector('input') ||
                        (hit.previousElementSibling &&
                         hit.previousElementSibling.tagName === 'INPUT'
                         ? hit.previousElementSibling : null);
          (input || hit).click();
          return true;
        }
        """ % json.dumps(needle)
        while time.time() < deadline:
            if await self.js(probe):
                return True
            await asyncio.sleep(0.4)
        return False

    FIELD_JS = """
    () => {
      const want = %s;
      const norm = (s) => (s || '').trim().replace(/\\s+/g, ' ');
      let hit = null;
      // 1) aria-label / placeholder, when Streamlit sets one
      for (const i of document.querySelectorAll('input')) {
        if (norm(i.getAttribute('aria-label')) === want ||
            norm(i.placeholder) === want) { hit = i; break; }
      }
      // 2) the visible <label> of the matching stTextInput container
      if (!hit) {
        for (const box of document.querySelectorAll('[data-testid="stTextInput"]')) {
          const label = box.querySelector('label');
          if (!label || norm(label.textContent) !== want) continue;
          const input = box.querySelector('input');
          if (input) { hit = input; break; }
        }
      }
      if (!hit) return null;
      hit.scrollIntoView({block: 'center'});
      const r = hit.getBoundingClientRect();
      return {x: r.x + r.width / 2, y: r.y + r.height / 2,
              value: hit.value, type: hit.type};
    }
    """

    async def field_rect(self, label: str):
        return await self.js(self.FIELD_JS % json.dumps(label))

    async def type_into(self, label: str, value: str) -> bool:
        """Focus the input labelled `label` and type `value` into it."""
        rect = await self.field_rect(label)
        if not rect:
            return False
        await self.call("Input.dispatchMouseEvent", type="mousePressed",
                        x=rect["x"], y=rect["y"], button="left", clickCount=1)
        await self.call("Input.dispatchMouseEvent", type="mouseReleased",
                        x=rect["x"], y=rect["y"], button="left", clickCount=1)
        await asyncio.sleep(0.3)
        await self.call("Input.insertText", text=value)
        await asyncio.sleep(0.8)  # let Streamlit's onChange debounce fire
        check = await self.field_rect(label)
        if check and check.get("value") == value:
            return True
        # insertText can miss React's change tracking — fall back to key events
        for ch in value:
            await self.call("Input.dispatchKeyEvent", type="keyDown", text=ch)
            await self.call("Input.dispatchKeyEvent", type="keyUp", text=ch)
        await asyncio.sleep(0.8)
        check = await self.field_rect(label)
        if not check or check.get("value") != value:
            print(f"[shots] WARN field {label!r} holds "
                  f"{(check or {}).get('value')!r}, expected {value!r}")
        return True

    async def scroll(self, top: int | None = None, into_view: str | None = None,
                     tags: str = "h1,h2,h3,label,[data-testid=\"stMetric\"],"
                                "[data-testid=\"stCaptionContainer\"],"
                                "[data-testid=\"stMarkdown\"]"):
        """Scroll the main pane.

        `tags` narrows which elements `into_view` may match: the section names
        also appear in the nav bar (which sits above the content), so a header
        shot passes tags="h2,h3" to land on the section header instead.
        """
        if into_view:
            expr = """
            () => {
              const want = %s;
              const els = [...document.querySelectorAll(%s)];
              const hit = els.find(e => (e.textContent || '').includes(want));
              if (!hit) return false;
              hit.scrollIntoView({block: 'start'});
              return true;
            }
            """ % (json.dumps(into_view), json.dumps(tags))
            if not await self.js(expr):
                print(f"[shots] WARN could not find {into_view!r}; "
                      "capturing at the current scroll position")
                await asyncio.sleep(0.6)
                return
        elif top is not None:
            await self.js("() => { const el = document.querySelector("
                          "'[data-testid=\"stMain\"]') || document.scrollingElement;"
                          "el.scrollTop = %d; return el.scrollTop; }" % top)
        await asyncio.sleep(0.6)

    async def shot(self, path: str):
        res = await self.call("Page.captureScreenshot", format="png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(res["data"]))
        print(f"[shots] wrote {os.path.relpath(path, ROOT)}")


def launch_chrome(exe: str, port: int, width: int, height: int) -> subprocess.Popen:
    profile = tempfile.mkdtemp(prefix="aaris_shots_")
    cmd = [exe, f"--headless=new", "--disable-gpu", "--hide-scrollbars",
           f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
           f"--window-size={width},{height}", "--no-first-run",
           "--no-default-browser-check", "about:blank"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    deadline = time.time() + 25
    while time.time() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{port}/json/version",
                            timeout=1).ok:
                return proc
        except requests.exceptions.RequestException:
            time.sleep(0.4)
    proc.kill()
    sys.exit("headless Chrome did not expose a DevTools port")


async def capture(args) -> list[str]:
    port = free_port()
    proc = launch_chrome(find_chrome(args.chrome), port, args.width, args.height)
    missing: list[str] = []
    try:
        targets = requests.get(f"http://127.0.0.1:{port}/json/list",
                               timeout=10).json()
        page = next(t for t in targets if t.get("type") == "page")
        async with CDP(page["webSocketDebuggerUrl"]) as cdp:
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Emulation.setDeviceMetricsOverride",
                           width=args.width, height=args.height,
                           deviceScaleFactor=1, mobile=False)

            # ---- 1) sign-in screen ------------------------------------------
            await cdp.goto(args.dashboard)
            signed_in = await cdp.wait_for_text("Fleet Availability", timeout=20)
            if not signed_in:
                if await cdp.wait_for_text("Sign in to the ground station", 10):
                    await cdp.shot(os.path.join(OUT_DIR, "01_login.png"))
                    # the form's real inputs mount a round-trip after the frame
                    if not await cdp.wait_for_selector(
                            '[data-testid="stTextInput"] input', 20):
                        raise RuntimeError("login inputs never mounted")
                    if not await cdp.type_into("ID", args.user):
                        raise RuntimeError("ID field not found")
                    if not await cdp.type_into("Password", args.password):
                        raise RuntimeError("Password field not found")
                    if not await cdp.click_text("Log in"):
                        raise RuntimeError("Log in button not found")
                    signed_in = await cdp.wait_for_text("Fleet Availability", 30)
            if not signed_in:
                raise RuntimeError(
                    "dashboard never rendered the fleet — is the ground station "
                    "up and are the demo credentials correct?")

            # ---- 2) Fleet Readiness: alerts first, then the KPI row ---------
            await cdp.scroll(top=0)
            if await cdp.wait_for_text("ACTIVE CRITICAL ALERT", 8):
                await cdp.shot(os.path.join(OUT_DIR, "03_critical_alerts.png"))
            else:
                missing.append("03_critical_alerts.png (no ACTIVE alerts right now)")
            await cdp.scroll(into_view="Fleet Readiness Decision Support",
                             tags="h1,h2,h3")
            await cdp.shot(os.path.join(OUT_DIR, "02_fleet_readiness.png"))

            # ---- 3) Aircraft Models: profile card, then an airframe table ---
            # Each section is screenshotted right after it is opened: the nav is
            # a single-page state switch, so anything captured after the next
            # click would show the wrong section.
            await cdp.click_text("Aircraft Models")
            if not await cdp.wait_for_text("Airframes", 30):
                raise RuntimeError("Aircraft Models never rendered")
            await asyncio.sleep(1.5)          # images + charts settle
            await cdp.scroll(into_view="Aircraft Models", tags="h1,h2,h3")
            await cdp.shot(os.path.join(OUT_DIR, "04_aircraft_models.png"))
            await cdp.scroll(into_view="Click a row to open that aircraft")
            await cdp.shot(os.path.join(OUT_DIR, "05_model_airframes.png"))

            # ---- 4) Aircraft Detail (AC-017, the edge-node fixture) ---------
            await cdp.click_text("Aircraft Detail")
            if not await cdp.wait_for_text("Aircraft profile (synthetic)", 30):
                raise RuntimeError("Aircraft Detail never rendered")
            if not await cdp.wait_for_text("112.76", 10):
                missing.append("06_aircraft_detail.png (AC-017 fixture RUL "
                               "112.76 not on screen — check the picker)")
            await cdp.scroll(into_view="Aircraft Detail", tags="h1,h2,h3")
            await cdp.shot(os.path.join(OUT_DIR, "06_aircraft_detail.png"))

            # ---- 5) Maintenance Plan ----------------------------------------
            await cdp.click_text("Maintenance Plan")
            if not await cdp.wait_for_text("Peak daily load", 30):
                raise RuntimeError("Maintenance Plan never rendered")
            await asyncio.sleep(1.5)          # the Gantt draws after the metrics
            await cdp.scroll(into_view="Maintenance Plan", tags="h1,h2,h3")
            await cdp.shot(os.path.join(OUT_DIR, "07_maintenance_plan.png"))

            # ---- 6) About ----------------------------------------------------
            await cdp.click_text("About")
            if not await cdp.wait_for_text("How this system works", 30):
                raise RuntimeError("About never rendered")
            await cdp.scroll(into_view="About", tags="h1,h2,h3")
            await cdp.shot(os.path.join(OUT_DIR, "08_about.png"))

            # ---- 9/10) OpenAPI contracts ------------------------------------
            # Swagger UI is rendered client-side from a CDN bundle, so wait for
            # the endpoint rows themselves rather than any body text.
            for url, fname in ((args.edge + "/docs", "09_edge_docs.png"),
                               (args.ground + "/docs", "10_ground_docs.png")):
                await cdp.goto(url)
                if await cdp.wait_for_selector(".swagger-ui .opblock", 30):
                    await asyncio.sleep(1.5)
                    await cdp.shot(os.path.join(OUT_DIR, fname))
                else:
                    missing.append(f"{fname} ({url} never rendered — is the "
                                   "Swagger UI CDN reachable from this machine?)")
    finally:
        proc.kill()
        try:
            requests.get(f"http://127.0.0.1:{port}/json/version", timeout=1)
        except requests.exceptions.RequestException:
            pass
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture AARIS README screenshots")
    ap.add_argument("--dashboard", default="http://localhost:8501")
    ap.add_argument("--edge", default="http://127.0.0.1:8000")
    ap.add_argument("--ground", default="http://127.0.0.1:8001")
    ap.add_argument("--user", default="commander")
    ap.add_argument("--password", default="commander123")
    ap.add_argument("--chrome", default=None, help="chrome.exe / msedge.exe path")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1100)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if not shutil.which("curl") and not requests:  # keep requests import honest
        sys.exit("requests is required")
    try:
        requests.get(args.dashboard, timeout=3)
    except requests.exceptions.RequestException:
        sys.exit(f"dashboard not reachable at {args.dashboard} — "
                 "start the stack first (demo\\run_demo.ps1)")

    missing = asyncio.run(capture(args))
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".png"))
    print(f"[shots] {len(files)} PNGs in docs/screenshots")
    for m in missing:
        print(f"[shots] SKIPPED {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
