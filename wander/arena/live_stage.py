"""Run the brains IN the Arena's own simulator, closed loop, and write down what they chose.

stage.py guessed where the ducks would be from a model of their walk, and the guess drifted: an
open-loop timeline played on the real physics put two ducks into each other. Here each brain
senses the simulated stage itself (every duck's true position and heading, ten times a second) and
its choices drive the next steps of the same simulation the Arena runs (arena.runner.run_stage's
loop, seed 1, the event's own settle). Its choices are recorded as keyframes on its own ticks, so
`arena score --timeline` replays exactly this run: the timeline is a record of decisions, made by a
brain that was watching.

    venv/bin/python behavior/live_stage.py b003e.onnx behavior/live-timeline.json
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from arena.cast import cast_of, load_borrowed
from arena.constants import CTRL_DT, TRACE_WIDTH
from arena.course import compile_stage
from arena.hub_fetch import fetch_references
from arena.manifest import load_event
from arena.policy import load_policy
from arena.runner import _reset_by_mujoco, _start
from arena.timeline import load_troupe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain import Senses  # noqa: E402
from stage import DURATION, cast as make_cast, load_emotions  # noqa: E402

TICK_STEPS = 5          # a brain decides every 5 control steps: 10 Hz
SEED = 1                # got-a-talent plays seed 1


def pose(frame: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = frame[3:7]
    return float(frame[0]), float(frame[1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def senses_of(i: int, poses, ducks, t: float, fallen: bool) -> Senses:
    me = ducks[i]
    me.x, me.y, me.yaw = poses[i]
    others = []
    for j, d in enumerate(ducks):
        if j != i:
            d.x, d.y, d.yaw = poses[j]
            others.append(d)
    s = me.senses(t, others)
    s.fallen = fallen
    return s


def command_vector(duck, cmd) -> np.ndarray:
    twist = list(cmd.twist) if duck.brain.can_walk else ([1.0, 0.0, 0.0] if cmd.sit else [0.0, 0.0, 0.0])
    return np.asarray(twist + list(cmd.head) + [0.0] * 6, dtype=np.float64)


def main(onnx: str, out: str) -> None:
    ducks = make_cast(load_emotions())
    manifest = load_event("got-a-talent")
    stub = {"timeline_version": 2, "duration_s": DURATION,
            "ducks": [{"policy": d.policy, "start": d.start, "keyframes": [{"t": 0.0}]} for d in ducks]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(stub, f)
    troupe = load_troupe(Path(f.name), manifest)
    cast = cast_of(manifest, troupe, load_policy(Path(onnx)), None,
                   load_borrowed(troupe, fetch_references(troupe, manifest)))
    performers = cast.performers

    model = compile_stage(manifest, [(p.rig, p.start) for p in performers])
    zeros = [np.zeros(13) for _ in performers]
    sim, settle, _ = _start(manifest, model, [p.policy for p in performers],
                            [p.start.yaw_deg for p in performers], SEED, zeros)
    steps = round(DURATION / CTRL_DT)
    frame = settle[:, -1].copy()
    commands = zeros
    trace = np.empty((len(performers), steps, TRACE_WIDTH), dtype=np.float32)
    for step in range(steps):
        t = round(step * CTRL_DT, 4)
        if step % TICK_STEPS == 0:
            poses = [pose(frame[i]) for i in range(len(ducks))]
            tilts = [math.degrees(math.acos(max(-1, min(1, 1 - 2 * (frame[i][4] ** 2 + frame[i][5] ** 2)))))
                     for i in range(len(ducks))]
            commands = []
            for i, d in enumerate(ducks):
                cmd = d.brain.tick(senses_of(i, poses, ducks, t, tilts[i] > 60), TICK_STEPS * CTRL_DT)
                d.record(cmd, t)
                commands.append(command_vector(d, cmd))
        sim.step(commands)
        if _reset_by_mujoco(sim.data):
            print("the simulation reset at", t)
            break
        sim.record(trace[:, step])
        frame = trace[:, step]

    tl = {"timeline_version": 2, "duration_s": DURATION, "ducks": [d.entry() for d in ducks]}
    Path(out).write_text(json.dumps(tl, indent=1))
    np.save(Path(out).with_suffix(".trace.npy"), trace)
    for d in ducks:
        print(f"{d.brain.name}: {len(d.keyframes)} keyframes, met {d.brain.met}")
        print("   ", " → ".join(f"{s}@{t}" for t, s in d.states))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
