"""Read a session log back and check it reproduces itself.

Two checks, both exact:

- `replay_brain`: the logged senses, fed to a fresh brain with the logged seed, give the logged
  commands. This is the conformance check for the Swift port: same senses in, same commands out.
- `replay_full`: the logged inputs (depth frame, pose, fall flags, head) fed to fresh `LiveSenses`
  give the logged senses, and then the brain and the runner's rules give the logged commands and
  what was sent.

Each returns a list of mismatches, empty when the log reproduces. Run it on a file:

    python3 replay.py tests/golden/table-seed1.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

import session_log as sl
from brain import Brain
from runner import head_for, twist_for
from senses import LiveSenses


def _brain(header: dict) -> Brain:
    if header["brain"]["emotions"]:
        raise sl.LogError("this log's brain played emotion head paths, which the log does not carry")
    return Brain(header["brain"]["name"], header["brain"]["seed"])


def _brain_state(b: Brain) -> dict:
    return {"state": b.state, "since": b.since, "energy": b.energy, "turn_dir": b.turn_dir,
            "reason": b.reason, "target": b.target}


def replay_brain(header: dict, ticks: list[dict]) -> list[tuple]:
    brain, dt, out = _brain(header), 1.0 / header["tick_hz"], []
    for rec in ticks:
        if rec["command"] is None:
            continue
        cmd = brain.tick(sl.senses_from(rec["senses"]), dt)
        if sl.command_json(cmd) != rec["command"]:
            out.append((rec["i"], "command", rec["command"], sl.command_json(cmd)))
    return out


def replay_full(header: dict, ticks: list[dict]) -> list[tuple]:
    brain, dt, out = _brain(header), 1.0 / header["tick_hz"], []
    senses, sent, last_head = LiveSenses(), (0.0, 0.0, 0.0), None
    for rec in ticks:
        got = rec["in"]
        senses.pitch = sl.num(got["pitch"])
        senses.head_z = sl.num(got.get("head_z", sl.LEGACY_HEAD_Z_M))
        pose = None if got["pose"] is None else tuple(sl.num(v) for v in got["pose"])
        s = senses.senses(sl.num(rec["t"]), None if got["safety"] is None else {"safety": got["safety"]},
                          sl.unpack_depth(got["depth"]), sl.num(got["head_yaw"]), got["steady"], pose, sent,
                          got["settled"])
        if sl.senses_json(s) != rec["senses"]:
            out.append((rec["i"], "senses", rec["senses"], sl.senses_json(s)))
        if rec["command"] is None:
            twist, head = (0.0, 0.0, 0.0), None
        else:
            cmd = brain.tick(s, dt)
            if sl.command_json(cmd) != rec["command"]:
                out.append((rec["i"], "command", rec["command"], sl.command_json(cmd)))
            if _brain_state(brain) != rec["brain"]:
                out.append((rec["i"], "brain", rec["brain"], _brain_state(brain)))
            twist = twist_for(cmd, header["max_speed"], got["depth"] is not None)
            head = head_for(cmd)
            head, last_head = (head if head != last_head else None), head
        if [sl.num(v) for v in rec["sent"]["twist"]] != list(twist):
            out.append((rec["i"], "sent.twist", rec["sent"]["twist"], list(twist)))
        logged_head = rec["sent"]["head"]
        if (None if logged_head is None else tuple(sl.num(v) for v in logged_head)) != head:
            out.append((rec["i"], "sent.head", logged_head, head))
        sent = twist
    return out


def main(paths: list[str]) -> int:
    bad = 0
    for p in paths:
        header, ticks = sl.read(Path(p).read_text().splitlines())
        b, f = replay_brain(header, ticks), replay_full(header, ticks)
        print(f"{p}: {len(ticks)} ticks, brain mismatches {len(b)}, full mismatches {len(f)}")
        for m in (b + f)[:5]:
            print("  ", m)
        bad += len(b) + len(f)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
