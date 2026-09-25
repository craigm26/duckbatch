"""A language model that proposes whole sequences, constrained so it cannot invent a skill.

    duckbatch plan "walk forward for three seconds, then peck the ground twice"

WHY A PLANNER AS WELL AS THE ROUTER. The router (r001) labels one clause at a time: it cannot say
"twice", "for three seconds", or "then wait". A plan needs order, duration and repetition, and
that is generation, which a decision model does not do. So a small local LLM writes the plan,
and three things keep it honest:

  - its output is constrained by a JSON schema whose `skill` is an enum of the router's own
    vocabulary (llama.cpp grammar-constrained decoding), so an invented action cannot be
    emitted, only a wrong choice among real ones;
  - code maps every label to the robot (router.to_command), so the model never picks a number
    the robot runs;
  - plans can be checked in simulation before a person runs them (p002's verifier).

The model is Gemma 4 E4B (Google's QAT q4_0 GGUF, open weights) served by llama.cpp's llama-server on CPU; any
OpenAI-compatible server with json_schema support works (`--url`).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from .router import ACTIONS, HEADS, SOUNDS, SPEEDS, to_command

FORMAT = "duck-intent-plan/1"
DEFAULT_URL = "http://127.0.0.1:8081/v1/chat/completions"

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array", "minItems": 1, "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "enum": list(ACTIONS)},
                    "speed": {"type": "string", "enum": list(SPEEDS)},
                    "head": {"type": "string", "enum": list(HEADS)},
                    "sound": {"type": "string", "enum": [*SOUNDS, "none"]},
                    "seconds": {"type": "number", "minimum": 0.5, "maximum": 10},
                    "repeat": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["skill", "speed", "head", "sound", "seconds", "repeat"],
            },
        },
    },
    "required": ["steps"],
}

SYSTEM = f"""You turn one instruction for a small toy duck robot into a short plan.
Output JSON only, matching the schema. Each step is one skill:
{chr(10).join(f"- {k}: {v}" for k, v in ACTIONS.items())}
speed: {", ".join(SPEEDS)} (normal unless the instruction says otherwise).
head: {", ".join(HEADS)} (straight unless the instruction says where to look).
sound: which noise for make_sound ({", ".join(SOUNDS)}), otherwise "none".
seconds: how long a walk, turn or stop lasts; use 2 if not stated. For other skills use 1.
repeat: how many times in a row (1 unless it says twice, three times...).
Keep the order the instruction gives. If a part asks for something the robot cannot do
(flying, climbing, grasping, swimming, flips, opening things, or it is a question), use the
single skill "unsupported" for that part. Do not add steps that were not asked for."""


def ask(request: str, url: str = DEFAULT_URL, temperature: float = 0.0, seed: int = 0,
        timeout: float = 120.0) -> dict[str, Any]:
    body = {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}],
        "temperature": temperature, "seed": seed, "max_tokens": 600,
        "response_format": {"type": "json_schema", "json_schema": {"name": "plan", "schema": SCHEMA}},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return json.loads(out["choices"][0]["message"]["content"])


def plan(request: str, url: str = DEFAULT_URL, **kw) -> dict[str, Any]:
    """A `duck-intent-plan/1`: the model's steps, each with the command code maps it to."""
    raw = ask(request, url, **kw)
    steps = []
    for s in raw["steps"]:
        labels = {"action": s["skill"], "speed": s["speed"], "head": s["head"]}
        if s["skill"] == "make_sound":
            labels["sound"] = s["sound"] if s["sound"] != "none" else "chirp"
        cmd = to_command(labels)
        if cmd.get("kind") == "twist" and cmd.get("duration_s") is not None:
            cmd["duration_s"] = float(s["seconds"])
        steps.append({"labels": labels, "seconds": float(s["seconds"]), "repeat": int(s["repeat"]),
                      "command": cmd})
    return {"format": FORMAT, "request": request, "model": "google/gemma-4-E4B-it-qat-q4_0-gguf (llama.cpp, json_schema)",
            "steps": steps}


def expand(steps: list[dict]) -> list[str]:
    """The skill sequence a plan executes, repeats unrolled: what p002 scores against."""
    out = []
    for s in steps:
        out += [s["labels"]["action"] if "labels" in s else s["action"]] * int(s.get("repeat", 1))
    return out
