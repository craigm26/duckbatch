#!/usr/bin/env python3
"""p001's paired rollouts -> one JSON file Duck Studio bundles for its A/B preference screen.

    python scripts/export_pairs_for_app.py            # writes records/p001-pairs/app-pairs.json

WHAT THE PERSON SEES IS WHAT WAS SIMULATED. Every frame here is a frame from
`records/p001-pairs/trajectories.npz`: the same seed, the same command, the same env on both
sides, so the only difference between the two ducks on screen is the network. Nothing is
re-simulated or smoothed; frames are decimated (50 Hz -> 25 Hz) and rounded to a millimetre and
a milliradian, both well under anything a phone screen can show.

THE JOINT ORDER IS CONVERTED HERE, ONCE. mjlab's `joint_pos` for the MicroDuck asset has 14
columns in the order each policy's own `joint_names` metadata states (duckkit's
`DuckModel.jointNames` with the mouth left out); the first frame of every clip equals the
policy's `default_joint_pos`, which is how that was checked. The app's renderer wants all 15 in
duckkit's order, so the mouth goes back in at index 9 as 0 (no alpha policy drives it).

THE ROOT IS RE-ZEROED PER ENV. mjlab lays envs out on a grid, so raw root positions carry a
grid offset of tens of metres. Each clip starts at x = y = 0; height is kept as simulated.

CLOSE PAIRS FIRST. p001's close note says a preference model learns most from pairs a person
finds hard, so each command lists its six policy pairs ordered by how alike the two policies'
behaviour features are (standardised distance between their per-command means). The app draws
from the front of that list. The features themselves are NOT exported: a rater shown "fell: 3%"
is rating the number, not the duck.

FINGERPRINTS ARE duckkit's. `duckbatch.policy.fingerprint` equals `DuckPolicy.fingerprint`; the
official alpha_walking value (da820b71...) is asserted below so a drift fails loudly here rather
than as records the reader cannot match.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duckbatch.policy import fingerprint  # noqa: E402

PAIRS = ROOT / "records" / "p001-pairs"
OUT = PAIRS / "app-pairs.json"
FORMAT = "duck-preference-pairs/0"
DECIMATE = 2  # 50 Hz -> 25 Hz
ALPHA_OFFICIAL = "sha256:da820b718aa8bdb3317c018afba3ad3f461e0cf42256811c204dc005546ec4a3"

DUCKKIT_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
MOUTH = DUCKKIT_JOINTS.index("mouth")

# Where each network lives for anyone else, so a preference names a network people can fetch.
PUBLISHED = {
    "velstand": {"repo": "pollen-robotics/microduck-policies", "file": "velstand.onnx"},
    "alpha_walking": {"repo": "pollen-robotics/microduck-policies", "file": "alpha_walking.onnx"},
    "student_256x128x64": {"repo": "craigm26/microduck-duckbatch-b002-256x128x64"},
    "student_128x128": {"repo": "craigm26/microduck-duckbatch-b002-128x128"},
}


def clip(raw: np.ndarray) -> list[list[list[float]]]:
    """[steps, envs, 21] mjlab rows -> [envs][frames][22]: x, y, z, qw, qx, qy, qz, 15 joints."""
    steps, envs, cols = raw.shape
    assert cols == 3 + 4 + 14, f"expected 21 columns, found {cols}"
    out = []
    for e in range(envs):
        rows = raw[::DECIMATE, e, :].astype(np.float64)
        origin = rows[0, :2].copy()
        frames = []
        for r in rows:
            joints = list(r[7:])
            joints.insert(MOUTH, 0.0)
            frame = [r[0] - origin[0], r[1] - origin[1], r[2], *r[3:7], *joints]
            frames.append([round(float(v), 3) for v in frame])
        out.append(frames)
    return out


def closeness(features: np.ndarray, policies: list[str], commands: list[str]) -> dict:
    """Per command, the six policy pairs ordered closest first (standardised feature means)."""
    # features: [policy, command, env, feature]
    flat = features.reshape(-1, features.shape[-1])
    scale = flat.std(axis=0)
    scale[scale == 0] = 1.0
    order = {}
    for c, cname in enumerate(commands):
        means = features[:, c].mean(axis=1) / scale
        rows = []
        for i, j in combinations(range(len(policies)), 2):
            rows.append((float(np.linalg.norm(means[i] - means[j])), policies[i], policies[j]))
        rows.sort()
        order[cname] = [{"a": a, "b": b, "distance": round(d, 4)} for d, a, b in rows]
    return order


def main() -> None:
    meta = json.loads((PAIRS / "pairs.json").read_text())
    traj = np.load(PAIRS / "trajectories.npz")
    feats = np.load(PAIRS / "features.npz")["features"]
    policies = list(meta["policies"])
    commands = list(meta["commands"])
    assert feats.shape[:2] == (len(policies), len(commands)), feats.shape

    prints = {}
    for name, rel in meta["policies"].items():
        prints[name] = fingerprint(ROOT / rel)
    assert prints["alpha_walking"] == ALPHA_OFFICIAL, (
        f"alpha_walking fingerprint {prints['alpha_walking']} is not duckkit's official value; "
        "run scripts/fetch_teachers.sh and check duckbatch.policy against duckkit")

    envs = None
    clips = {}
    for p in policies:
        for c in commands:
            key = f"{p}__{c}"
            clips[key] = clip(traj[key])
            envs = len(clips[key])

    doc = {
        "format": FORMAT,
        "source": {"batch": "p001", "task": meta["task"], "seed": meta["seed"],
                   "num_envs": meta["num_envs"], "simulated": True,
                   "note": "MuJoCo via mjlab; every frame is a recorded sim frame, decimated"},
        "hz": 50 / DECIMATE,
        "seconds": meta["seconds"],
        "envs": envs,
        "frame": ["x", "y", "z", "qw", "qx", "qy", "qz", *DUCKKIT_JOINTS],
        "policies": {p: {**PUBLISHED[p], "fingerprint": prints[p]} for p in policies},
        "commands": meta["commands"],
        "close_first": closeness(feats, policies, commands),
        "clips": clips,
    }
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB, "
          f"{len(clips)} clips x {envs} envs, {len(next(iter(clips.values()))[0])} frames each")


if __name__ == "__main__":
    main()
