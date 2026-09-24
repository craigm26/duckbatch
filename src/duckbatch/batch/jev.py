"""Minimal TypeSafe System One (Jev) client, stdlib only.

Ported from craigm26/bounded-answer-lab `harness/jev.py` (MIT), same API shape (jev-1.13.0):

  POST https://api.typesafe.ai/v1/systemone
  {"model": "jev-latest", "state": <str|dict>, "questions": {id: {"type", "instructions", "criteria"}}}

  choice -> {"choice", "confidence", "probabilities"}   criteria: {option_id: description}
  score  -> {"score", "confidence", "probabilities"}    criteria: [ordered level descriptions]
  noul   -> {"noul": p_yes}                             no criteria

Jev answers over a fixed set of options with calibrated probabilities and never generates text.
That is the property duckbatch leans on: a judgment that can be thresholded, logged and replayed.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = os.environ.get("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone")


def load_dotenv(path: str | Path = ".env") -> None:
    """KEY=VALUE lines into os.environ, never overwriting what is already set."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def available() -> bool:
    load_dotenv()
    return bool(os.environ.get("TYPESAFE_API_KEY"))


class JevClient:
    def __init__(self, api_key: str | None = None, model: str = "jev-latest", timeout: float = 30.0):
        load_dotenv()
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY not set (put it in .env; see .env.example)")
        self.model = model
        self.timeout = timeout
        self.log: list[dict[str, Any]] = []

    def ask(self, state: Any, questions: dict[str, dict], retries: int = 3) -> dict[str, Any]:
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        last = None
        t0 = time.perf_counter()
        for attempt in range(retries):
            req = urllib.request.Request(
                API_URL, data=body, method="POST",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
            except urllib.error.HTTPError as e:
                last = f"{e.code}: {e.read()[:300]!r}"
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            except (urllib.error.URLError, TimeoutError) as e:
                last = str(e)
                time.sleep(1.5 * (attempt + 1))
                continue
            self.log.append({"state": state, "questions": questions, "answers": data["answers"],
                             "model": data.get("model"), "usage": data.get("usage"),
                             "seconds": round(time.perf_counter() - t0, 3)})
            return data["answers"]
        raise RuntimeError(f"Jev request failed: {last}")
