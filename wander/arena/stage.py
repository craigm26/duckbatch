"""Run brains on a model of the Arena's talent stage and write down what they chose, as a timeline.

The stage's floor is 2.6 m deep along x (towards the audience) by 3.8 m wide along y. A walker's
motion is modelled from b003e measured on that stage with the Arena's own simulator
(calibration in README): 0.20 m/s at a 0.45 forward command, 0.25 at 0.55, 0.18 m/s turning at
0.52 rad/s for (0.45, 0, 0.6), 0.09 m/s at 1.15 rad/s for (0.4, 0, 1.0), nothing below about 0.35,
and roughly 0.8 s to get going. The brains see this model; `arena score` then plays the timeline
on the real simulator, and verify.py checks that the two agree.

    python3 stage.py out/timeline.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain import Brain, Senses  # noqa: E402

HALF_X, HALF_Y = 1.3, 1.9
DT = 0.1
DURATION = 20.0
DUCK_R = 0.15
SIT = "pollen-robotics/microduck-got-a-talent-reference@0454fe8a848526338ba6edf72b8358fbf51d6b21"
EMOTIONS = Path(__file__).with_name("emotions")


def walker_motion(vx: float, wz: float) -> tuple[float, float]:
    """Forward speed (m/s) and turn rate (rad/s) a command gives once the duck is going."""
    if vx < 0.35:
        return 0.0, 0.0
    v = 0.20 * vx / 0.45 * (1.0 - 0.55 * min(1.0, abs(wz)))
    k = 0.87 + 0.28 * max(0.0, abs(wz) - 0.6) / 0.4
    return v, k * wz


def load_emotions() -> dict:
    out = {}
    for f in EMOTIONS.glob("*.json"):
        kf = json.loads(f.read_text())["keyframes"]
        path = [(round(k["t"], 2), (k["neck"], k["head_pitch"], k["head_yaw"], k["head_roll"]))
                for k in kf if round(k["t"] * 10) % 2 == 0]           # every 0.2 s is plenty
        out[f.stem] = path
    return out


class Duck:
    def __init__(self, brain: Brain, x: float, y: float, yaw_deg: float, policy: str = "self"):
        self.brain, self.policy = brain, policy
        self.x, self.y, self.yaw = x, y, math.radians(yaw_deg)
        self.start = {"x": x, "y": y, "yaw": yaw_deg}
        self.v = self.w = 0.0
        self.keyframes: list[dict] = []
        self.states: list[tuple[float, str]] = []
        self._last = None

    def clear_ahead(self, others: list["Duck"]) -> float:
        """Free distance straight ahead: to the stage's edge, or to a duck in a 70° cone."""
        hx, hy = math.cos(self.yaw), math.sin(self.yaw)
        best = math.inf
        for edge, d in ((HALF_X, hx), (-HALF_X, hx), (HALF_Y, hy), (-HALF_Y, hy)):
            pos = self.x if abs(edge) == HALF_X else self.y
            if d * edge > 0:
                best = min(best, (edge - pos) / d)
        for o in others:
            dx, dy = o.x - self.x, o.y - self.y
            along = dx * hx + dy * hy
            if along > 0 and abs(math.atan2(dx * -hy + dy * hx, along)) < math.radians(45):
                best = min(best, math.hypot(dx, dy) - 2 * DUCK_R)
        return max(0.0, best)

    def senses(self, t: float, others: list["Duck"]) -> Senses:
        s = Senses(t=t, clear_ahead_m=self.clear_ahead(others))
        for o in others:
            d = math.hypot(o.x - self.x, o.y - self.y)
            if d < s.near_m:
                bearing = math.atan2(o.y - self.y, o.x - self.x) - self.yaw
                s.near_id, s.near_m = o.brain.name, d
                s.near_bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if d < 0.75:
                s.company += 1
        return s

    def step(self, cmd, t: float) -> None:
        if self.brain.can_walk:
            v, w = walker_motion(cmd.twist[0], cmd.twist[2])
            a = 1 - math.exp(-DT / 0.6)                                  # getting going takes a moment
            self.v += a * (v - self.v)
            self.w += a * (w - self.w)
            self.yaw += self.w * DT
            self.x += self.v * math.cos(self.yaw) * DT
            self.y += self.v * math.sin(self.yaw) * DT
        self.record(cmd, t)

    def record(self, cmd, t: float) -> None:
        if self.brain.can_walk:
            twist = [round(c, 2) for c in cmd.twist]
        else:
            twist = [1, 0, 0] if cmd.sit else [0, 0, 0]
        head = [round(h, 2) for h in cmd.head]
        trig = [1, 0] if cmd.quack else ([0, 1] if cmd.wheee else None)
        key = (tuple(twist), tuple(head), tuple(trig) if trig else None)
        if not self.states or self.states[-1][1] != cmd.state:
            self.states.append((round(t, 1), cmd.state))
        if key == self._last:
            return
        self._last = key
        kf = {"t": round(t, 2)}
        if any(twist):
            kf["twist"] = twist
        if any(head):
            kf["head"] = head
        if trig:
            kf["triggers"] = trig
        self.keyframes.append(kf)

    def entry(self) -> dict:
        return {"policy": self.policy, "start": self.start, "keyframes": self.keyframes or [{"t": 0.0}]}


def cast(emotions: dict) -> list[Duck]:
    """Three walkers who have never met, and a sit-and-stand duck who is about to nap."""
    return [
        Duck(Brain("anna", 11, emotions=emotions), 0.1, -0.3, -100),
        Duck(Brain("bo", 23, emotions=emotions), 0.6, 1.4, -95),
        Duck(Brain("cy", 37, emotions=emotions), -0.6, 1.2, -70),
        Duck(Brain("dot", 41, can_walk=False, emotions=emotions, energy=0.5), -0.3, -1.5, 80, policy=SIT),
    ]


def run(ducks: list[Duck]) -> dict:
    t = 0.0
    while t < DURATION - 1e-9:
        cmds = [d.brain.tick(d.senses(t, [o for o in ducks if o is not d]), DT) for d in ducks]
        for d, c in zip(ducks, cmds):
            d.step(c, t)
        t = round(t + DT, 2)
    return {"timeline_version": 2, "duration_s": DURATION, "ducks": [d.entry() for d in ducks]}


if __name__ == "__main__":
    ducks = cast(load_emotions())
    tl = run(ducks)
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "timeline.json")
    out.write_text(json.dumps(tl, indent=1))
    for d in ducks:
        print(d.brain.name, f"end ({d.x:.2f}, {d.y:.2f})", len(d.keyframes), "keyframes")
        print("   ", " → ".join(f"{s}@{t}" for t, s in d.states))
        print("    met:", d.brain.met)
    json.dump({d.brain.name: [round(d.x, 3), round(d.y, 3)] for d in ducks}, open(out.with_suffix(".model.json"), "w"))
