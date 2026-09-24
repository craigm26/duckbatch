"""The judge: which attempts in a batch die, which get more budget. Rule -> Jev -> person.

WHERE THE DECISION MODEL SITS. Jev never touches the control loop and never proposes a config.
It answers one bounded question at the edge of the search: for an attempt whose measured
numbers land in the band the pre-registered rules cannot call, "extend or kill?". The code owns
the search (rungs, budgets, ranking); Jev owns only the calls a threshold cannot make; a person
owns whatever Jev is not sure about. Same cascade as rtlab's `judge_cell.py`.

WHAT JEV SEES. Observed numbers only: this arm's eval row, the teacher's row from the SAME eval
(the noise floor), the loss trail, and the lab conventions (the thresholds, in words). Never a
prior like "bigger nets are better", which is exactly the kind of belief the batch is testing.

Every decision (whichever tier made it) is a row in the batch record, and every Jev call is
logged verbatim so the thresholds can be recalibrated later against what actually survived.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from . import jev as jevmod

# Pre-registered defaults (2026-09-24, before batch b001 ran). A menu may override them, and
# the override is part of the committed menu, so it is pre-registered too.
DEFAULT_GATES: dict[str, float] = {
    # falls per env-minute above the teacher's, from the same eval
    "falls_keep_abs": 0.25,
    "falls_kill_abs": 1.5,
    # velocity tracking error as a ratio to the teacher's
    "lin_keep_ratio": 1.15,
    "lin_kill_ratio": 1.6,
    "ang_keep_ratio": 1.15,
    "ang_kill_ratio": 1.6,
    # fraction of time spent tilted past 60 degrees (walk profile)
    "down_keep": 0.02,
    "down_kill": 0.10,
    # share of prone spawns back up within the window, in points BELOW the teacher's
    "recover_keep_drop": 0.10,
    "recover_kill_drop": 0.30,
    # Jev acts alone at or above this confidence; below it the case goes to a person
    "jev_act_alone": 0.90,
    # share of machine-closed decisions sampled for a person to audit (FNV-1a of the arm id)
    "audit_share": 0.10,
}

LAB_CONVENTIONS = {
    "task": "Microduck biped velocity tracking with standing (mjlab, domain randomization and "
            "pushes on). The teacher is Pollen's shipped default walker; attempts are smaller "
            "student networks distilled from it.",
    "falls_per_min": "upright to tilted-past-60-degrees transitions per env-minute; lower is better",
    "down_frac": "fraction of time tilted past 60 degrees; lower is better",
    "recovered_frac": "share of prone spawns upright within 6 s; higher is better",
    "lin_err": "mean |commanded - actual| planar velocity, m/s; lower is better",
    "ang_err": "mean |commanded - actual| yaw rate, rad/s; lower is better",
    "teacher_mse": "mean squared action gap to the teacher on the student's own states",
    "teacher_row": "the teacher's numbers from the same eval: the floor any student is held to, "
                   "and a reading of eval noise (the teacher is not retrained)",
    "budget": "later rungs give a surviving attempt more training and more envs; an attempt "
              "that is still improving quickly can close a gap a rung later",
}


@dataclass
class Decision:
    arm_id: str
    verdict: str  # keep | kill | pending
    tier: str  # rule | jev | person
    reasons: list[str] = field(default_factory=list)
    confidence: float | None = None
    audit: bool = False
    # every consulted model's answer, whether or not it acted: {name: {choice, confidence, ...}}
    shadow: dict[str, Any] = field(default_factory=dict)
    case_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def rule_verdict(row: dict[str, float], teacher: dict[str, float], g: dict[str, float]):
    """`kill` if any kill line is crossed, `keep` if every keep line holds, else `uncertain`."""
    core = ("falls_per_min", "down_frac", "lin_err", "ang_err")
    if not all(math.isfinite(float(row[k])) for k in core):
        return "kill", ["non-finite metric"]
    kill, keep_fail = [], []
    df = row["falls_per_min"] - teacher["falls_per_min"]
    lin = row["lin_err"] / max(teacher["lin_err"], 1e-6)
    ang = row["ang_err"] / max(teacher["ang_err"], 1e-6)
    down = row["down_frac"]
    if df > g["falls_kill_abs"]:
        kill.append(f"falls +{df:.2f}/min over teacher > {g['falls_kill_abs']}")
    elif df > g["falls_keep_abs"]:
        keep_fail.append(f"falls +{df:.2f}/min over teacher")
    if lin > g["lin_kill_ratio"]:
        kill.append(f"lin_err {lin:.2f}x teacher > {g['lin_kill_ratio']}")
    elif lin > g["lin_keep_ratio"]:
        keep_fail.append(f"lin_err {lin:.2f}x teacher")
    if ang > g["ang_kill_ratio"]:
        kill.append(f"ang_err {ang:.2f}x teacher > {g['ang_kill_ratio']}")
    elif ang > g["ang_keep_ratio"]:
        keep_fail.append(f"ang_err {ang:.2f}x teacher")
    if down > g["down_kill"]:
        kill.append(f"down {down:.1%} > {g['down_kill']:.0%}")
    elif down > g["down_keep"]:
        keep_fail.append(f"down {down:.1%}")
    if "recovered_frac" in row and "recovered_frac" in teacher:
        drop = teacher["recovered_frac"] - row["recovered_frac"]
        if drop > g["recover_kill_drop"]:
            kill.append(f"gets up {row['recovered_frac']:.0%} vs teacher "
                        f"{teacher['recovered_frac']:.0%}")
        elif drop > g["recover_keep_drop"]:
            keep_fail.append(f"gets up {row['recovered_frac']:.0%} vs teacher "
                             f"{teacher['recovered_frac']:.0%}")
    if kill:
        return "kill", kill
    if keep_fail:
        return "uncertain", keep_fail
    return "keep", ["within every keep line"]


def jev_questions() -> dict[str, dict]:
    return {
        "decision": jevmod.choice(
            "Given only the observed numbers for this attempt and the teacher row from the same "
            "evaluation, should this attempt get the next rung's training budget?",
            {
                "extend": "Worth more budget: its gap to the teacher is small, or its loss is "
                          "still falling fast enough that more training plausibly closes it.",
                "kill": "Not worth more budget: its gap to the teacher is material and its loss "
                        "has flattened, or it falls or drifts where the teacher does not.",
            },
        ),
        "gap": jevmod.score(
            "How large is this attempt's real deficit against the teacher, reading the teacher "
            "row as the eval's noise floor?",
            [
                "None: indistinguishable from the teacher within eval noise",
                "Small: a gap, but one a user would not notice while driving the duck",
                "Material: visibly worse tracking or stability than the teacher",
                "Severe: falls, drifts or ignores commands where the teacher does not",
            ],
        ),
        "still_learning": jevmod.noul(
            "Is the loss trail still falling enough that the next rung's budget would likely "
            "change the verdict?"
        ),
    }


def render_case(arm_id: str, row: dict, teacher: dict, loss_trail: list[float], rung: int,
                rungs_left: int, params: int, gates: dict[str, float]) -> str:
    """The one text every decision model sees for a case. The code does the arithmetic (ratios
    to the teacher); the model makes the call. Which rule lines were crossed is NOT included:
    on the cases the rules settle, that would hand the answer over and make agreement free."""
    tr = [round(x, 4) for x in loss_trail[-6:]]
    lin_r = row["lin_err"] / max(teacher["lin_err"], 1e-6)
    ang_r = row["ang_err"] / max(teacher["ang_err"], 1e-6)
    return (
        f"Microduck walking policy attempt {arm_id}: a {params:,}-parameter student distilled "
        f"from Pollen's 197,774-parameter default walker, after training rung {rung + 1} "
        f"({rungs_left} rung(s) left). Simulator eval with pushes and domain randomization; "
        f"the teacher ran in the same eval.\n"
        f"Falls per minute: attempt {row['falls_per_min']:.2f}, teacher "
        f"{teacher['falls_per_min']:.2f}.\n"
        f"Time spent fallen: attempt {row['down_frac']:.1%}, teacher {teacher['down_frac']:.1%}.\n"
        f"Planar velocity error: attempt {row['lin_err']:.3f} m/s, teacher "
        f"{teacher['lin_err']:.3f} m/s ({lin_r:.2f}x the teacher).\n"
        f"Yaw rate error: attempt {row['ang_err']:.3f} rad/s, teacher {teacher['ang_err']:.3f} "
        f"rad/s ({ang_r:.2f}x the teacher).\n"
        + (f"Gets up from a prone spawn within 6 s: attempt {row['recovered_frac']:.0%}, "
           f"teacher {teacher['recovered_frac']:.0%}.\n" if "recovered_frac" in row else "")
        + f"Action gap to the teacher (mean squared): {row['teacher_mse']:.4f}.\n"
        f"Training loss, last checkpoints: {', '.join(str(x) for x in tr) or 'none yet'}.\n"
        f"Lab conventions: an attempt that falls about {gates['falls_keep_abs']} more times per "
        f"minute than the teacher, or tracks velocity more than "
        f"{(gates['lin_keep_ratio'] - 1):.0%} worse, is noticeably worse; "
        f"{gates['falls_kill_abs']} more falls per minute, or "
        f"{(gates['lin_kill_ratio'] - 1):.0%} worse tracking, is clearly unacceptable."
    )


class JevJudge:
    """Jev behind the same `ask(text)` shape as Decide (the text is sent as the state)."""

    name = "jev"

    def __init__(self, client: "jevmod.JevClient"):
        self.client = client

    @property
    def log(self):
        return self.client.log

    def ask(self, text: str) -> dict:
        return self.client.ask(text, jev_questions())


class DecideJudge:
    name = "decide"

    def __init__(self, client):
        self.client = client

    @property
    def log(self):
        return self.client.log

    def ask(self, text: str) -> dict:
        return self.client.ask(text)


def _brief(ans: dict) -> dict:
    return {"choice": ans["decision"]["choice"],
            "confidence": round(float(ans["decision"].get("confidence", 0.0)), 4),
            "gap": ans["gap"].get("score"),
            "still_learning": round(float(ans["still_learning"].get("noul", 0.0)), 4)}


def judge_rung(rows: dict[str, dict], teacher_id: str, meta: dict[str, dict], rung: int,
               rungs_left: int, gates: dict[str, float] | None = None,
               models: dict | None = None, actors: list[str] | None = None) -> list[Decision]:
    """One decision per student arm. `meta[arm_id]` carries `loss_trail` and `params`.

    `models` ({name: judge}) are ALL consulted on EVERY arm (shadow), so each model's agreement
    with the rules on the cases the rules settle can be measured. Only the names in `actors` may
    decide an uncertain case, in order, at confidence >= `<name>_act_alone` (default
    `jev_act_alone`); otherwise it goes to a person.
    """
    g = {**DEFAULT_GATES, **(gates or {})}
    models = models or {}
    actors = [a for a in (actors or []) if a in models]
    teacher = rows[teacher_id]
    out = []
    for arm_id, row in rows.items():
        if arm_id == teacher_id:
            continue
        verdict, reasons = rule_verdict(row, teacher, g)
        m = meta.get(arm_id, {})
        text = render_case(arm_id, row, teacher, m.get("loss_trail", []), rung, rungs_left,
                           m.get("params", 0), g)
        shadow: dict[str, Any] = {}
        for name, model in models.items():
            try:
                shadow[name] = _brief(model.ask(text))
            except Exception as e:  # a judge that cannot answer is a person's case, not a crash
                shadow[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        if verdict != "uncertain":
            d = Decision(arm_id, verdict, "rule", reasons)
        else:
            d = None
            for name in actors:
                a = shadow.get(name, {})
                if "error" in a:
                    reasons = reasons + [f"{name} could_not_check"]
                    continue
                thr = g.get(f"{name}_act_alone", g["jev_act_alone"])
                if a["confidence"] >= thr:
                    d = Decision(arm_id, "keep" if a["choice"] == "extend" else "kill", name,
                                 reasons + [f"{name} {a['choice']} @ {a['confidence']:.2f} "
                                            f"gap={a['gap']}"], a["confidence"])
                    break
                reasons = reasons + [f"{name} {a['choice']} @ {a['confidence']:.2f} < {thr}"]
            if d is None:
                why = [] if actors else ["no acting decision model: to a person"]
                d = Decision(arm_id, "pending", "person", reasons + why)
        d.shadow = shadow
        d.case_text = text
        if d.tier != "person":
            d.audit = (fnv1a(f"{arm_id}:r{rung}") % 1000) < g["audit_share"] * 1000
        out.append(d)
    return out


def rank_score(row: dict, teacher: dict) -> float:
    """Lower is better. Tracking as teacher-ratios, plus a stability term. Pre-registered."""
    return (row["lin_err"] / max(teacher["lin_err"], 1e-6)
            + row["ang_err"] / max(teacher["ang_err"], 1e-6)
            + max(0.0, row["falls_per_min"] - teacher["falls_per_min"])
            + 10.0 * row["down_frac"]
            + 3.0 * max(0.0, teacher.get("recovered_frac", 0.0) - row.get("recovered_frac", 0.0)))


def survivors(decisions: list[Decision], rows: dict[str, dict], teacher_id: str,
              quota: int, mode: str = "quality", params: dict[str, int] | None = None) -> list[str]:
    """Kills go. Of the rest (keep and pending: never kill what could not be judged), `quota`
    advance and the others are closed as `outranked`.

    quality     best rank score first.
    efficiency  arms that passed every keep line first, SMALLEST first (the question is "how
                small can it get and still hold the teacher's line"), then the undecided ones by
                rank score.
    """
    t = rows[teacher_id]
    alive = [d for d in decisions if d.verdict != "kill"]
    if mode == "efficiency":
        p = params or {}
        alive.sort(key=lambda d: (d.verdict != "keep",
                                  p.get(d.arm_id, 0) if d.verdict == "keep" else 0,
                                  rank_score(rows[d.arm_id], t)))
    else:
        alive.sort(key=lambda d: rank_score(rows[d.arm_id], t))
    return [d.arm_id for d in alive[:quota]]
