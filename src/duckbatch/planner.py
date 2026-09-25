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
from pathlib import Path
import urllib.request
from typing import Any

from .router import ACTIONS, HEADS, SOUNDS, SPEEDS, to_command

FORMAT = "duck-intent-plan/1"
DEFAULT_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "google/gemma-4-E4B-it-qat-q4_0-gguf (llama.cpp, json_schema)"

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
        timeout: float = 300.0, constrained: bool = True, thinking: bool = True,
        timings: dict | None = None) -> dict[str, Any]:
    """The model's raw plan. `constrained=False` is how a phone runs it: Apple's on-device model
    and MLX in Duck Studio have no grammar, so the reply is prose that should contain JSON, and a
    strict reader (`read_reply`) decides whether it does. p002 measures both ways."""
    system = SYSTEM if constrained else SYSTEM + UNCONSTRAINED_TAIL
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": request}],
        "temperature": temperature, "seed": seed, "max_tokens": 600,
        # Gemma 4's template turns thinking ON unless told otherwise: ~300 hidden reasoning tokens
        # before ~75 of plan, which was most of p002's first latency. Duck Studio's phone runtime
        # turns it off, so the phone-comparable runs do too.
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if constrained:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "plan", "schema": SCHEMA}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    if timings is not None and "timings" in out:
        t = out["timings"]
        timings.update({"prompt_tokens": t.get("prompt_n"), "prompt_s": round(t.get("prompt_ms", 0) / 1000, 2),
                        "gen_tokens": t.get("predicted_n"), "gen_s": round(t.get("predicted_ms", 0) / 1000, 2)})
    return read_reply(out["choices"][0]["message"]["content"])


UNCONSTRAINED_TAIL = """
Reply with one JSON object and nothing else, shaped like:
{"steps": [{"skill": "walk_forward", "speed": "normal", "head": "straight", "sound": "none", "seconds": 2, "repeat": 1}]}"""


class PlanReadError(ValueError):
    pass


def read_reply(text: str) -> dict[str, Any]:
    """Strict, like the app's reader: the first JSON object in the reply, every label in the
    vocabulary, numbers clamped to the schema's ranges. Anything else is a refusal, not a guess."""
    # As Duck Studio's ChatWire.firstJSONObject reads a reply: reasoning blocks dropped, a fenced
    # block preferred, then the first object in what is left.
    import re

    for tag in ("think", "thinking", "reasoning"):
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.S)
    fence = re.search(r"```[^\n]*\n(.*?)(```|$)", text, flags=re.S)
    if fence and "{" in fence.group(1):
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise PlanReadError("no JSON object in the reply")
    try:
        raw, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise PlanReadError(f"not JSON: {e}") from None
    steps = raw.get("steps") if isinstance(raw, dict) else None
    if not isinstance(steps, list) or not steps:
        raise PlanReadError("no steps")
    enums = {"skill": ACTIONS, "speed": SPEEDS, "head": HEADS, "sound": [*SOUNDS, "none"]}
    defaults = {"speed": "normal", "head": "straight", "sound": "none"}
    clean = []
    for s in steps:
        if not isinstance(s, dict):
            raise PlanReadError("a step is not an object")
        step = {}
        for key, allowed in enums.items():
            v = s.get(key, defaults.get(key))
            if v not in allowed:
                raise PlanReadError(f"{key} {v!r} is not in the vocabulary")
            step[key] = v
        sec, rep = s.get("seconds", 2), s.get("repeat", 1)
        is_num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        if not is_num(sec) or not is_num(rep) or float(rep) != int(rep):
            raise PlanReadError("seconds is not a number, or repeat is not a whole number")
        step["seconds"] = min(10.0, max(0.5, float(sec)))
        step["repeat"] = min(5, max(1, int(rep)))
        clean.append(step)
    return {"steps": clean}


def plan(request: str, url: str = DEFAULT_URL, model: str = MODEL, **kw) -> dict[str, Any]:
    """A `duck-intent-plan/1`: the model's steps, each with the command code maps it to."""
    timings: dict = {}
    raw = ask(request, url, timings=timings, **kw)
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
    return {"format": FORMAT, "request": request, "model": model,
            "steps": steps, "timings": timings}


def expand(steps: list[dict]) -> list[str]:
    """The skill sequence a plan executes, repeats unrolled: what p002 scores against."""
    out = []
    for s in steps:
        out += [s["labels"]["action"] if "labels" in s else s["action"]] * int(s.get("repeat", 1))
    return out


# ---- measurement (p002) ----------------------------------------------------------------------

MOVEMENT = {"walk_forward", "walk_backward", "turn_left", "turn_right", "stop"}


def _collapse(seq: list[dict]) -> list[dict]:
    out = []
    for s in seq:
        if not (out and s["action"] == "unsupported" and out[-1]["action"] == "unsupported"):
            out.append(s)
    return out


def _unroll_gold(steps: list[dict]) -> list[dict]:
    out = []
    for s in steps:
        out += [s] * int(s.get("repeat", 1))
    return _collapse(out)


def _unroll_plan(p: dict) -> list[dict]:
    """Planner or router output as one dict per executed step: action, labels, seconds."""
    out = []
    for s in p["steps"]:
        step = {**s["labels"], "seconds": s.get("seconds")}
        out += [step] * int(s.get("repeat", 1))
    return _collapse(out)


def _timing_summary(ts: list[dict]) -> dict[str, Any]:
    if not ts:
        return {}
    med = lambda xs: sorted(xs)[len(xs) // 2]
    return {"prompt_s_median": med([t["prompt_s"] for t in ts]),
            "gen_tokens_median": med([t["gen_tokens"] for t in ts]),
            "gen_tokens_per_s": round(sum(t["gen_tokens"] for t in ts) / max(sum(t["gen_s"] for t in ts), 1e-9), 1)}


def evaluate(requests: list[dict], make_plan, name: str, log=print) -> dict[str, Any]:
    import time

    heads = {"speed": [0, 0], "head": [0, 0], "sound": [0, 0], "seconds": [0, 0]}
    seq_ok = full_ok = refused = 0
    whole_oos = [r for r in requests if all(s["action"] == "unsupported" for s in r["steps"])]
    rows, lat = [], []
    for r in requests:
        t0 = time.perf_counter()
        try:
            p = make_plan(r["text"])
            err = None
        except Exception as e:  # a failed plan is a wrong plan, recorded
            p, err = {"steps": []}, repr(e)
        lat.append(time.perf_counter() - t0)
        got, gold = _unroll_plan(p), _unroll_gold(r["steps"])
        s_ok = [g["action"] for g in got] == [g["action"] for g in gold]
        f_ok = s_ok
        if s_ok:
            for g, want in zip(got, gold):
                checks = []
                if want["action"] in MOVEMENT:
                    checks += ["speed", "head"]
                if want["action"] == "make_sound":
                    checks += ["sound"]
                for h in checks:
                    ok = g.get(h) == want.get(h, "normal" if h == "speed" else "straight")
                    heads[h][0] += ok
                    heads[h][1] += 1
                    f_ok &= ok
                if "seconds" in want:
                    ok = g.get("seconds") is not None and abs(float(g["seconds"]) - want["seconds"]) < 0.25
                    heads["seconds"][0] += ok
                    heads["seconds"][1] += 1
                    f_ok &= ok
        seq_ok += s_ok
        full_ok += f_ok
        if r in whole_oos and got and all(g["action"] == "unsupported" for g in got):
            refused += 1
        log(f"[{name}] {'OK ' if f_ok else ('seq' if s_ok else 'BAD')} {lat[-1]:5.1f}s  {r['text']}  ->  "
            f"{[g['action'] for g in got]}{'  ERR ' + err if err else ''}")
        rows.append({"request": r["text"], "gold": gold, "got": got, "sequence_exact": s_ok,
                     "fully_exact": f_ok, "seconds": round(lat[-1], 2), "error": err,
                     "timings": p.get("timings")})
    n = len(requests)
    lat_sorted = sorted(lat)
    return {
        "model": name,
        "sequence_exact": f"{seq_ok}/{n}", "fully_exact": f"{full_ok}/{n}",
        "sequence_exact_rate": round(seq_ok / n, 4), "fully_exact_rate": round(full_ok / n, 4),
        "labels_given_sequence": {h: f"{c}/{t}" for h, (c, t) in heads.items()},
        "whole_out_of_scope_refused": f"{refused}/{len(whole_oos)}",
        "latency_s": {"median": round(lat_sorted[n // 2], 2), "max": round(lat_sorted[-1], 2)},
        "read_failures": sum(1 for x in rows if x["error"]),
        **_timing_summary([x["timings"] for x in rows if x.get("timings")]),
        "rows": rows,
    }


# ---- what the phone needs (Duck Studio) ------------------------------------------------------

READER_CASES = [  # raw replies a phone model might give, each read by `read_reply` for the answer
    ('{"steps": [{"skill": "walk_forward", "speed": "slow", "head": "look_left", "sound": "none", '
     '"seconds": 3, "repeat": 1}]}'),
    'Here is the plan:\n```json\n{"steps": [{"skill": "peck_ground", "repeat": 2}]}\n```',
    '{"steps": [{"skill": "make_sound", "sound": "chirp", "seconds": 0.1, "repeat": 9}]}',
    '{"steps": [{"skill": "walk_forward", "seconds": 60}, {"skill": "unsupported"}]}',
    '{"steps": [{"skill": "fly"}]}',
    '{"steps": [{"skill": "walk_forward", "speed": "very fast"}]}',
    '{"steps": []}',
    'I cannot help with that.',
    '{"steps": [{"skill": "turn_left", "seconds": "two"}]}',
    '{"steps": [{"skill": "turn_left", "repeat": 2.5}]}',
    '<think>maybe {"steps": [{"skill": "roll"}]}</think>{"steps": [{"skill": "stop", "seconds": 1}]}',
    '{"steps": [{"skill": "walk_backward", "speed": "fast"}, {"skill": "make_sound"}]}',
]


def export_for_phone(out: str | Path) -> Path:
    """The files Duck Studio bundles so the phone asks exactly what p002 measured:
    the instructions (no-grammar form, which is how every phone runtime runs), the p002 set with
    its gold, and reader cases with Python's answers, so the Swift reader is tested to agree."""
    import yaml

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "instructions.txt").write_text(SYSTEM + UNCONSTRAINED_TAIL)
    menu = yaml.safe_load((Path(__file__).parents[2] / "menus" / "p002-plan-requests.yaml").read_text())
    (out / "p002.json").write_text(json.dumps(
        {"id": menu["id"], "format": FORMAT, "requests": menu["requests"]}, indent=1))
    cases = []
    for text in READER_CASES:
        try:
            cases.append({"reply": text, "steps": read_reply(text)["steps"]})
        except PlanReadError as e:
            cases.append({"reply": text, "refused": str(e)})
    (out / "reader-cases.json").write_text(json.dumps(cases, indent=1))
    return out
