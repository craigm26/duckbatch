"""GLiNER2.5-Decide as a judge tier: local, open (Apache-2.0), 340M, no key, CPU is fine.

fastino/GLiNER2.5-Decide scores a fixed label set (with descriptions, or an ordinal scale)
against a text in one forward pass and generates nothing, which is the same "search the answer
space" shape as Jev, but running on the reproducer's own machine. It is loaded on CPU on
purpose: the simulator needs ~3.2 of the 4 GB of VRAM, and a rung is a handful of calls.

It reads TEXT, so every case is rendered by `judge.render_case`; Jev is given the same string,
so the two models are compared on identical input.

Caveat carried over from craigm26/rtlab-private (record/rtlab-cells/TESTING.md): base GLiNER2
collapsed on cases whose answer is a numeric threshold (52/58 misses). Judge cases here are
mostly numbers too, so Decide starts in shadow and is measured before it may act.
"""

from __future__ import annotations

import time
from typing import Any

MODEL_ID = "fastino/GLiNER2.5-Decide"
# Pinned: a later push to the repo must not silently change a judge that decisions rest on.
REVISION = "65624f1a0265b3f612bae66a2685a06b94a68a9d"

SCHEMA: dict[str, Any] = {
    "decision": {
        "labels": {
            "extend": "Worth more training budget: the gap to the teacher is small, or the loss "
                      "is still falling fast enough that more training plausibly closes it",
            "kill": "Not worth more budget: the gap to the teacher is material and the loss has "
                    "flattened, or it falls or drifts where the teacher does not",
        },
    },
    "gap": {
        "labels": {
            "0": "No gap: indistinguishable from the teacher within eval noise",
            "1": "Small gap a user would not notice while driving the duck",
            "2": "Material gap: visibly worse tracking or stability than the teacher",
            "3": "Severe: falls, drifts or ignores commands where the teacher does not",
        },
    },
    "still_learning": {
        "labels": ["yes", "no"],
        "prompt": "Is the loss still falling enough that more training would change the verdict?",
    },
}


def available() -> bool:
    try:
        import gliner2  # noqa: F401
    except ImportError:
        return False
    return True


class DecideClient:
    def __init__(self, device: str = "cpu", threads: int = 8):
        import torch
        from gliner2 import AutoExtractor

        torch.set_num_threads(threads)
        self.model = AutoExtractor.from_pretrained(MODEL_ID, revision=REVISION)
        if device != "cpu":
            self.model = self.model.to(device)
        self.log: list[dict[str, Any]] = []

    def ask(self, text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        out = self.model.classify_text(text, SCHEMA, include_confidence=True)
        ans = {
            "decision": {"choice": out["decision"]["label"],
                         "confidence": float(out["decision"]["confidence"])},
            "gap": {"score": int(out["gap"]["label"]),
                    "confidence": float(out["gap"]["confidence"])},
            "still_learning": {"noul": float(out["still_learning"]["confidence"])
                               if out["still_learning"]["label"] == "yes"
                               else 1.0 - float(out["still_learning"]["confidence"])},
        }
        self.log.append({"model": f"{MODEL_ID}@{REVISION[:7]}", "text": text, "answers": ans,
                         "seconds": round(time.perf_counter() - t0, 3)})
        return ans
