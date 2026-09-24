"""Plain language -> proposed Microduck steps. Code splits and maps; a decision model picks.

    "walk slowly while looking right, then say hello"
      -> clauses (code):   ["walk slowly while looking right", "say hello"]
      -> labels (model):   {action: walk_forward, speed: slow, head: look_right}, {action: make_sound, sound: greet}
      -> steps (code):     twist (0.08, 0, 0) + head yaw -0.4 rad, 2 s; sound "greet"

WHY THE SPLIT IS NOT A MODEL. Where a sentence breaks into steps is a string question with a
checkable answer, and a model that got it wrong would be wrong silently. Where the model earns
its place is the part code cannot do: "boot the ball with your right leg" is `kick_right`.

WHY THE MAPPING IS NOT A MODEL. Velocities and skill tags are the robot's contract (Pollen's
command limits, duckkit's `DuckSkill` / `DuckSound` tags). A model choosing numbers is how a
duck gets commanded 3 m/s. The model picks a label; code owns what the label means.

A PROPOSAL, NOT A COMMAND. The output is `duck-intent-plan/0`: steps plus the model's label and
confidence per head, for a person to edit in Duck Studio's plan editor. Nothing here moves a robot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FORMAT = "duck-intent-plan/0"

ACTIONS: dict[str, str] = {
    "walk_forward": "Walk or move forwards, ahead, towards something",
    "walk_backward": "Walk or move backwards, back up, reverse",
    "turn_left": "Turn, rotate or spin to the left (anticlockwise) on the spot",
    "turn_right": "Turn, rotate or spin to the right (clockwise) on the spot",
    "stop": "Stop moving and stand still, freeze, halt",
    "kick_left": "Kick with the left foot or leg",
    "kick_right": "Kick with the right foot or leg",
    "sit_or_stand": "Sit down, take a seat, or stand back up from sitting",
    "roll": "Do a roll, tumble or somersault forwards and recover",
    "peck_ground": "Peck at the ground or floor, pick at food with the beak",
    "make_sound": "Make a sound or noise: quack, say hello, chirp, coo, an alarm, a question",
    "unsupported": "Something this small duck robot cannot do: flying, climbing, grasping or "
                   "carrying objects, swimming, flips, opening things, or a question, not a command",
}
SPEEDS: dict[str, str] = {
    "slow": "Slowly, gently, carefully, a little",
    "normal": "At a normal pace, or no speed mentioned",
    "fast": "Quickly, fast, hurry, run",
}
HEADS: dict[str, str] = {
    "straight": "Head facing forward, or no head direction mentioned",
    "look_left": "Look or turn the head to the left",
    "look_right": "Look or turn the head to the right",
    "look_up": "Look up, raise the head",
    "look_down": "Look down, at the feet or the floor",
}
SOUNDS: dict[str, str] = {
    "greet": "Say hello, greet someone",
    "alarm": "Sound an alarm, a warning",
    "inquire": "Ask a question, a curious questioning noise",
    "peck": "A pecking sound",
    "chirp": "A happy chirp or quack",
    "coo": "A soft, gentle cooing noise",
    "wheee": "Wheee! An excited, joyful noise",
}

# Pollen's command limits (pollen-robotics/microduck-simulator constants.js: VEL_FWD 0.25,
# VEL_BACK -0.2, VEL_ANG 1.0), and a third of, two thirds of, or all of each for slow/normal/fast.
VX_FORWARD, VX_BACK, WZ = 0.25, 0.20, 1.0
SPEED_SHARE = {"slow": 1 / 3, "normal": 2 / 3, "fast": 1.0}
HEAD_OFFSETS = {  # (neck_pitch, head_pitch, head_yaw, head_roll) offsets, rad: modest, in range
    "straight": (0.0, 0.0, 0.0, 0.0), "look_left": (0.0, 0.0, 0.4, 0.0),
    "look_right": (0.0, 0.0, -0.4, 0.0), "look_up": (0.0, -0.3, 0.0, 0.0),
    "look_down": (0.0, 0.3, 0.0, 0.0),
}
SKILL_TAGS = {"kick_left": "kick_left", "kick_right": "kick_right", "sit_or_stand": "sit_toggle",
              "roll": "roulade", "peck_ground": "ground_pick"}
DEFAULT_MOVE_S = 2.0

_SPLIT = re.compile(r"\s*(?:,\s*then\s+|;\s*|\s+and then\s+|\s+then\s+|\s+after that\s+)\s*",
                    re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    """Where a request breaks into steps. Code, so it is checkable and never silently wrong."""
    return [c.strip(" .!?,") for c in _SPLIT.split(text.strip()) if c.strip(" .!?,")]


@dataclass
class Step:
    clause: str
    labels: dict[str, str]
    confidence: dict[str, float]
    command: dict[str, Any] = field(default_factory=dict)


def to_command(labels: dict[str, str]) -> dict[str, Any]:
    """A label set -> what the robot is asked to do. Code owns every number."""
    a = labels["action"]
    head = list(HEAD_OFFSETS[labels.get("head", "straight")])
    share = SPEED_SHARE[labels.get("speed", "normal")]
    if a == "unsupported":
        return {"kind": "refused", "reason": "not something this robot can do"}
    if a == "make_sound":
        return {"kind": "sound", "tag": labels.get("sound", "chirp")}
    if a in SKILL_TAGS:
        return {"kind": "skill", "tag": SKILL_TAGS[a]}
    twist = {"walk_forward": (VX_FORWARD * share, 0.0, 0.0),
             "walk_backward": (-VX_BACK * share, 0.0, 0.0),
             "turn_left": (0.0, 0.0, WZ * share), "turn_right": (0.0, 0.0, -WZ * share),
             "stop": (0.0, 0.0, 0.0)}[a]
    return {"kind": "twist", "twist": [round(v, 4) for v in twist], "head": head,
            "duration_s": None if a == "stop" else DEFAULT_MOVE_S}


# ---- decision models ------------------------------------------------------------------------


class DecideRouter:
    name = "decide"

    def __init__(self):
        from .batch.decide import MODEL_ID, REVISION
        import torch
        from gliner2 import AutoExtractor

        torch.set_num_threads(8)
        self.model = AutoExtractor.from_pretrained(MODEL_ID, revision=REVISION)
        self.schema = {
            "action": {"labels": ACTIONS},
            "speed": {"labels": SPEEDS},
            "head": {"labels": HEADS},
            "sound": {"labels": SOUNDS},
        }

    def label(self, clause: str) -> tuple[dict[str, str], dict[str, float]]:
        out = self.model.classify_text(clause, self.schema, include_confidence=True)
        return ({k: v["label"] for k, v in out.items()},
                {k: float(v["confidence"]) for k, v in out.items()})


class JevRouter:
    name = "jev"

    def __init__(self):
        from .batch import jev as jevmod

        self.jev = jevmod
        self.client = jevmod.JevClient()
        instr = "This is one instruction given to a small toy duck robot. {q}"
        self.questions = {
            "action": jevmod.choice(instr.format(q="What should the robot do?"), ACTIONS),
            "speed": jevmod.choice(instr.format(q="How fast?"), SPEEDS),
            "head": jevmod.choice(instr.format(q="Where should its head point?"), HEADS),
            "sound": jevmod.choice(instr.format(q="If it makes a sound, which one?"), SOUNDS),
        }

    def label(self, clause: str) -> tuple[dict[str, str], dict[str, float]]:
        ans = self.client.ask(clause, self.questions)
        return ({k: v["choice"] for k, v in ans.items()},
                {k: float(v.get("confidence", 0.0)) for k, v in ans.items()})


def route(text: str, model) -> dict[str, Any]:
    steps = []
    for clause in split_clauses(text):
        labels, conf = model.label(clause)
        steps.append(Step(clause, labels, conf, to_command(labels)))
    return {"format": FORMAT, "request": text, "model": model.name,
            "steps": [{"clause": s.clause, "labels": s.labels, "confidence": s.confidence,
                       "command": s.command} for s in steps]}


# ---- measurement -----------------------------------------------------------------------------


def evaluate(requests: list[dict], model) -> dict[str, Any]:
    """Score one model on the pre-registered set. Every clause is scored on `action`;
    `speed` and `head` on every clause whose gold action is a movement (walk, turn, stop);
    `sound` only where the gold action is `make_sound`."""
    heads = {"action": [0, 0], "speed": [0, 0], "head": [0, 0], "sound": [0, 0]}
    exact, refused, wrong_conf, rows = 0, 0, [], []
    movement = {"walk_forward", "walk_backward", "turn_left", "turn_right", "stop"}
    oos = [r for r in requests if r["steps"][0]["action"] == "unsupported"]
    for r in requests:
        plan = route(r["text"], model)
        all_ok = len(plan["steps"]) == len(r["steps"])
        for got, gold in zip(plan["steps"], r["steps"]):
            checks = ["action"]
            if gold["action"] in movement:
                checks += ["speed", "head"]
            if gold["action"] == "make_sound":
                checks += ["sound"]
            for h in checks:
                ok = got["labels"].get(h) == gold.get(h, "normal" if h == "speed" else "straight")
                heads[h][0] += ok
                heads[h][1] += 1
                if not ok:
                    all_ok = False
                    wrong_conf.append(got["confidence"].get(h, 0.0))
            rows.append({"request": r["text"], "clause": got["clause"], "gold": gold,
                         "got": got["labels"], "confidence": got["confidence"]})
        exact += all_ok
        if r in oos and plan["steps"][0]["labels"]["action"] == "unsupported":
            refused += 1
    return {
        "model": model.name,
        "accuracy": {h: round(c / n, 4) if n else None for h, (c, n) in heads.items()},
        "counts": {h: f"{c}/{n}" for h, (c, n) in heads.items()},
        "exact_requests": f"{exact}/{len(requests)}",
        "out_of_scope_refused": f"{refused}/{len(oos)}",
        "mean_confidence_when_wrong": round(sum(wrong_conf) / len(wrong_conf), 4) if wrong_conf else None,
        "rows": rows,
    }
