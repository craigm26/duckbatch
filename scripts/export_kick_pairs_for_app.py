#!/usr/bin/env python3
"""A `duckbatch kick-pairs` run -> one `duck-preference-pairs/0` file Duck Studio shows as kick duels.

    python scripts/export_kick_pairs_for_app.py records/kp001-right

THE WALKING PACK'S FORMAT, WITH THREE ADDITIONS a reader that does not know them ignores:
  - `skill`: the robotd slot these networks fill (`kick_right`, `kick_left`). The app keeps a
    separate taste per skill, so a pick between kicks never moves the walking taste.
  - `ball`: per clip key, per env, one [x, y, z] per frame, re-zeroed with the duck's root so
    the two are drawn in one frame.
  - `features`: the seven kick features measured in the simulator (`kick_pairs.FEATURES`), per
    clip key and env. They are NOT shown to the person; they ride in each pick record so the
    taste is fitted over what the sim measured, not over what a phone could estimate without a
    ball. The walking pack leaves its features out; a kick pack cannot, because nothing on the
    phone can measure a kick.

`commands` holds the conditions (plain, backlash) as zero twists, because a kick has no command
and the app's deck is keyed by them. Clips stop at each env's first episode end
(`lengths.npz`), so no clip shows the reset teleport.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duckbatch.policy import fingerprint  # noqa: E402
from export_pairs_for_app import DECIMATE, DUCKKIT_JOINTS, MOUTH  # noqa: E402

FORMAT = "duck-preference-pairs/0"

# Where each network can be fetched, so a pick names a network anybody can get.
PUBLISHED = {
    "pollen_kick_right": {"repo": "pollen-robotics/microduck-policies", "file": "ball_kick_right.onnx"},
    "pollen_kick_left": {"repo": "pollen-robotics/microduck-policies", "file": "ball_kick_left.onnx"},
    "k001": {"repo": "craigm26/duckbatch-records",
             "file": "jobs/6ac0aaf1404719ba3762a7bb/k001-kick-straight/policies/finetuned/policy.onnx"},
    "k002": {"repo": "craigm26/duckbatch-records",
             "file": "jobs/6ac12bebfbc85ba68237dd86/k002-kick-left-straight/policies/finetuned/policy.onnx"},
    "k002b": {"repo": "craigm26/duckbatch-records",
              "file": "jobs/6ac13433fbc85ba68237ed3d/k002b-kick-left-straighter/policies/finetuned/policy.onnx"},
}


def clips_for(raw: np.ndarray, ball: np.ndarray, lengths: np.ndarray):
    """[steps, keep, 21] + [steps, keep, 3] -> per env: frames [n][22] and ball [n][3]."""
    steps, keep, cols = raw.shape
    assert cols == 3 + 4 + 14, f"expected 21 columns, found {cols}"
    frames_out, ball_out = [], []
    for e in range(keep):
        end = int(lengths[e])
        rows = raw[:end:DECIMATE, e, :].astype(np.float64)
        brows = ball[:end:DECIMATE, e, :].astype(np.float64)
        origin = rows[0, :2].copy()
        frames, bs = [], []
        for r, b in zip(rows, brows):
            joints = list(r[7:])
            joints.insert(MOUTH, 0.0)
            frames.append([round(float(v), 3) for v in
                           [r[0] - origin[0], r[1] - origin[1], r[2], *r[3:7], *joints]])
            bs.append([round(float(b[0] - origin[0]), 3), round(float(b[1] - origin[1]), 3),
                       round(float(b[2]), 3)])
        frames_out.append(frames)
        ball_out.append(bs)
    return frames_out, ball_out


def main(run: str) -> None:
    run = Path(run)
    meta = json.loads((run / "pairs.json").read_text())
    feats = np.load(run / "features.npz")["features"]
    traj, balls, lengths = (np.load(run / f) for f in ("trajectories.npz", "balls.npz", "lengths.npz"))
    policies, conditions = list(meta["policies"]), list(meta["conditions"])
    keep = meta["kept_envs"]

    clips, ball, values = {}, {}, {}
    for pi, p in enumerate(policies):
        for ci, c in enumerate(conditions):
            key = f"{p}__{c}"
            clips[key], ball[key] = clips_for(traj[key], balls[key], lengths[key])
            values[key] = [[round(float(v), 5) for v in feats[pi, ci, e]] for e in range(keep)]

    # Close pairs first, by standardised distance between per-condition feature means.
    flat = feats.reshape(-1, feats.shape[-1])
    scale = flat.std(axis=0)
    scale[scale == 0] = 1.0
    close = {}
    for ci, c in enumerate(conditions):
        means = feats[:, ci].mean(axis=1) / scale
        rows = sorted((float(np.linalg.norm(means[i] - means[j])), policies[i], policies[j])
                      for i, j in combinations(range(len(policies)), 2))
        close[c] = [{"a": a, "b": b, "distance": round(d, 4)} for d, a, b in rows]

    doc = {
        "format": FORMAT,
        "skill": meta["skill"],
        "source": {"batch": run.name, "task": meta["conditions"], "seed": meta["seed"],
                   "num_envs": meta["num_envs"], "simulated": True,
                   "note": "MuJoCo via mjlab; Pollen's BallKick task (left: duckbatch's mirror); "
                           "every frame a recorded sim frame, decimated, cut at the first episode end"},
        "hz": 50 / DECIMATE,
        "seconds": 5.0,
        "envs": keep,
        "frame": ["x", "y", "z", "qw", "qx", "qy", "qz", *DUCKKIT_JOINTS],
        "policies": {p: {**PUBLISHED[p], "fingerprint": fingerprint(ROOT / meta["policies"][p])}
                     for p in policies},
        "commands": {c: [0.0, 0.0, 0.0] for c in conditions},
        "close_first": close,
        "clips": clips,
        "ball": ball,
        "features": {"names": meta["features"], "measured": "simulator", "values": values},
    }
    out = run / "app-pairs.json"
    out.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {out}: {out.stat().st_size / 1e6:.2f} MB, {len(clips)} clips x {keep} envs")


if __name__ == "__main__":
    main(sys.argv[1])
