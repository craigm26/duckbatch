#!/usr/bin/env python3
"""The plain-language router over HTTP, for Duck Studio's plan editor.

    python scripts/route_server.py                      # 0.0.0.0:8771, GLiNER2.5-Decide
    python scripts/route_server.py --port 9000 --host 127.0.0.1

    POST /route   {"text": "walk slowly, then turn left"}  ->  duck-intent-plan/0
    GET  /health  ->  {"ok": true, "model": "decide", ...}

WHY A SERVER AT ALL. Decide is 340M parameters and a phone app that bundled it would be a
different app. Until an on-device model can do the job, the router runs on a machine the person
owns and the app reaches it at an address the person types, which is how Duck Studio already
reaches a model server. Duck Studio's GATES.md rules out any endpoint the project runs, so there
is deliberately no hosted default here.

WHY THE STANDARD LIBRARY AND NOT FASTAPI. One route, one model, one person at a time. Model
loading is the slow part (tens of seconds) and happens once at start; after that each clause is
about 1.4 s on an x86 CPU and slower on a Pi. `ThreadingHTTPServer` with a lock around the model
is all that needs.

NOTHING IS STORED. Requests are not logged beyond the line http.server prints, and the text is
not written anywhere. Corrections come back to duckbatch only as `duck-feedback/0` records a
person chose to share.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duckbatch.router import FORMAT, route  # noqa: E402

MAX_TEXT = 500  # characters; a request is a sentence or three, not a document


def make_handler(model, lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                return self._send(200, {"ok": True, "model": model.name, "format": FORMAT})
            self._send(404, {"error": "GET /health or POST /route"})

        def do_POST(self):  # noqa: N802
            if self.path != "/route":
                return self._send(404, {"error": "POST /route"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(min(length, 16_384)) or b"{}")
                text = str(body.get("text", "")).strip()
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"error": "the body must be JSON: {\"text\": \"...\"}"})
            if not text:
                return self._send(400, {"error": "there is no text to route"})
            if len(text) > MAX_TEXT:
                return self._send(413, {"error": f"at most {MAX_TEXT} characters"})
            started = time.monotonic()
            with lock:
                plan = route(text, model)
            plan["seconds"] = round(time.monotonic() - started, 2)
            self._send(200, plan)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (default: one per core)")
    args = ap.parse_args()

    from duckbatch.router import DecideRouter

    print("loading GLiNER2.5-Decide ...", flush=True)
    started = time.monotonic()
    model = DecideRouter()
    # ONE THREAD PER CORE. DecideRouter asks for eight, which suits the x86 box it was measured
    # on; on a four-core Pi, eight threads fight over four cores and every clause gets slower.
    import os
    import torch

    torch.set_num_threads(args.threads or os.cpu_count() or 4)
    print(f"loaded in {time.monotonic() - started:.0f} s; serving on {args.host}:{args.port}",
          flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, threading.Lock()))
    server.serve_forever()


if __name__ == "__main__":
    main()
