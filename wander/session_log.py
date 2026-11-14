"""The wander session log: one JSON line per tick, the same schema wherever the brain runs.

The practice duck, the Space driving a real duck, duck-sim and the Swift port all write this, so
any run is a replay fixture (wander plan, W0 item 3). This module is the schema; `replay.py` reads
a log back through the senses and the brain.

A log is a header line, then one line per tick:

    {"kind": "session", "schema": "microduck-wander-session", "v": 1, "venue": "practice", ...}
    {"kind": "tick", "v": 1, "i": 0, "t": 0.25, "in": {...}, "senses": {...}, "brain": {...},
     "command": {...}, "sent": {...}, "gate": {"verdict": "none", "rule": null}, "runner": "Wander"}

- `in` is what the tick read: the depth frame it used (or null when it had none fresh), the pose
  and fall flags from `robot.state` (null when stale), the head yaw and pitch, whether the duck
  was holding still (`steady`), and whether the frame came after the head stopped moving
  (`settled`). Feeding `in` to `LiveSenses` gives `senses` again.
- `senses` is what the brain was given. Feeding it to a fresh brain gives `command` again.
- `brain` and `command` are null on ticks where the runner held the duck without asking the brain
  (fallen, no fresh state, recovering).
- `sent` is what went to the robot: the twist every tick, the head only on ticks it changed.
- `gate` is the envelope gate's verdict for this tick's twist. Its names are DuckEnvelopeGate's
  (duck-studio, brains-layer.md 6.2): pass, clamped, slowed, held, cut, invalid. Nothing in this
  folder runs a gate yet, so it writes "none".

Non-finite numbers are the strings "inf", "-inf" and "nan", as in the gate's golden cases, so every
line is strict JSON.

**The depth frame is packed, losslessly.** `distance_mm` (64 values, 0 to 4000) as little-endian
uint16, then `status` (64 values, 0 to 255) as uint8: 192 bytes, base64, 256 characters. As JSON
arrays the same frame is about 450 characters, and the frame is most of a line, so packing roughly
halves a log. It keeps every value `senses.py` reads, exactly; a frame with values outside those
ranges is refused rather than squeezed. Base64 decodes in one call in Python and in Swift
(`Data(base64Encoded:)`). Other fields a sensor sends (sigma, reflectance) are not read by the
brain and are not kept.
"""

from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import asdict
from typing import IO, Any, Iterable

from brain import Command, Senses

SCHEMA = "microduck-wander-session"
VERSION = 1
VENUES = ("practice", "duck-sim", "real")
VERDICTS = ("none", "pass", "clamped", "slowed", "held", "cut", "invalid")
ZONES = 64


class LogError(ValueError):
    """A line that is not this schema, or not this version of it."""


# -- numbers ---------------------------------------------------------------------------------------
def _enc(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    if isinstance(x, dict):
        return {k: _enc(v) for k, v in x.items()}
    return x


def num(x: Any) -> float:
    """A logged number back to a float, the non-finite strings included."""
    return float(x)


# -- the depth frame -------------------------------------------------------------------------------
def pack_depth(frame: dict | None) -> dict | None:
    if frame is None:
        return None
    d, s = list(frame.get("distance_mm") or []), list(frame.get("status") or [])
    if len(d) != ZONES or len(s) != ZONES:
        raise LogError(f"a depth frame has {ZONES} zones, this one {len(d)} and {len(s)}")
    if not all(isinstance(v, int) and 0 <= v <= 0xFFFF for v in d) or \
            not all(isinstance(v, int) and 0 <= v <= 0xFF for v in s):
        raise LogError("depth values outside uint16 distances and uint8 statuses")
    raw = struct.pack("<64H64B", *d, *s)
    return {"seq": frame.get("seq"), "rows": frame.get("rows", 8), "cols": frame.get("cols", 8),
            "zones": base64.b64encode(raw).decode("ascii")}


def unpack_depth(packed: dict | None) -> dict | None:
    if packed is None:
        return None
    vals = struct.unpack("<64H64B", base64.b64decode(packed["zones"], validate=True))
    return {"seq": packed.get("seq"), "rows": packed.get("rows", 8), "cols": packed.get("cols", 8),
            "distance_mm": list(vals[:ZONES]), "status": list(vals[ZONES:])}


# -- senses and commands ---------------------------------------------------------------------------
def senses_json(s: Senses) -> dict:
    return _enc(asdict(s))


def senses_from(d: dict) -> Senses:
    return Senses(t=num(d["t"]), clear_ahead_m=num(d["clear_ahead_m"]), near_id=d["near_id"],
                  near_m=num(d["near_m"]), near_bearing=num(d["near_bearing"]), company=d["company"],
                  fallen=d["fallen"], edge_ahead_m=num(d["edge_ahead_m"]),
                  edge_bearing=num(d["edge_bearing"]),
                  pose=None if d["pose"] is None else tuple(num(v) for v in d["pose"]),
                  sent=tuple(num(v) for v in d["sent"]))


def command_json(c: Command) -> dict:
    return _enc(asdict(c))


def command_from(d: dict) -> Command:
    return Command(twist=tuple(num(v) for v in d["twist"]), head=tuple(num(v) for v in d["head"]),
                   quack=d["quack"], wheee=d["wheee"], sit=d["sit"], emote=d["emote"], state=d["state"])


# -- lines -----------------------------------------------------------------------------------------
def header(venue: str, brain_name: str, seed: int, tick_hz: float, max_speed: float, *,
           host: str = "local", room: str | None = None, world_seed: int | None = None,
           started_utc: str | None = None, emotions: Iterable[str] = ()) -> dict:
    if venue not in VENUES:
        raise LogError(f"venue {venue!r} is not one of {VENUES}")
    return {"kind": "session", "schema": SCHEMA, "v": VERSION, "venue": venue, "host": host,
            "brain": {"name": brain_name, "seed": seed, "emotions": sorted(emotions)},
            "tick_hz": tick_hz, "max_speed": max_speed, "room": room, "world_seed": world_seed,
            "started_utc": started_utc}


def tick(i: int, t: float, *, depth: dict | None, depth_age_s: float | None, pose, safety: dict | None,
         head_yaw: float, pitch: float, steady: bool, settled: bool, senses: Senses, brain: dict | None,
         command: Command | None, twist: tuple, head: tuple | None, emote: str | None, wheee: bool,
         runner: str, verdict: str = "none", rule: str | None = None) -> dict:
    if verdict not in VERDICTS:
        raise LogError(f"gate verdict {verdict!r} is not one of {VERDICTS}")
    return _enc({
        "kind": "tick", "v": VERSION, "i": i, "t": t,
        "in": {"depth": pack_depth(depth), "depth_age_s": depth_age_s,
               "pose": None if pose is None else list(pose), "safety": safety,
               "head_yaw": head_yaw, "pitch": pitch, "steady": steady, "settled": settled},
        "senses": senses_json(senses),
        "brain": brain,
        "command": None if command is None else command_json(command),
        "sent": {"twist": list(twist), "head": None if head is None else list(head),
                 "emote": emote, "wheee": wheee},
        "gate": {"verdict": verdict, "rule": rule},
        "runner": runner,
    })


def dumps(record: dict) -> str:
    return json.dumps(record, separators=(",", ":"), allow_nan=False)


def loads(line: str) -> dict:
    rec = json.loads(line)
    if rec.get("v") != VERSION:
        raise LogError(f"schema version {rec.get('v')!r}, this reader is version {VERSION}")
    if rec.get("kind") == "session":
        if rec.get("schema") != SCHEMA:
            raise LogError(f"schema {rec.get('schema')!r} is not {SCHEMA}")
    elif rec.get("kind") == "tick":
        if rec["gate"]["verdict"] not in VERDICTS:
            raise LogError(f"gate verdict {rec['gate']['verdict']!r} is not one of {VERDICTS}")
    else:
        raise LogError(f"line kind {rec.get('kind')!r} is neither session nor tick")
    return rec


def read(lines: Iterable[str]) -> tuple[dict, list[dict]]:
    """A whole log: its header and its ticks, in order."""
    recs = [loads(line) for line in lines if line.strip()]
    if not recs or recs[0]["kind"] != "session":
        raise LogError("a log starts with its session line")
    if any(r["kind"] != "tick" for r in recs[1:]):
        raise LogError("one session line per log, at the top")
    return recs[0], recs[1:]


class Writer:
    """Appends lines to an open text file, or to a list when `out` is None (for tests)."""

    def __init__(self, out: IO[str] | None = None):
        self.out = out
        self.lines: list[str] = []

    def write(self, record: dict) -> None:
        line = dumps(record)
        if self.out is None:
            self.lines.append(line)
        else:
            self.out.write(line + "\n")
            self.out.flush()
