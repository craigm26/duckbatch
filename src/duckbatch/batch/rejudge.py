"""Replay decision models over a finished batch's stored cases, and measure them.

Every rung file keeps each case's rendered text, the rule verdict, and the shadow answers. The
cases the pre-registered rules settled (keep = extend, kill = kill) are labels no model saw
the answer to, so agreement on them is a clean measurement. Uncertain cases have no label; the
models' answers there are reported side by side, with what the next rung showed where there is
one.

    duckbatch rejudge records/b001-student-size                    # report on the stored shadows
    duckbatch rejudge records/b001-student-size --models jev       # ask Jev now, same texts

Writes `<batch>/judges.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import judge as judgemod

RULE_TO_CHOICE = {"keep": "extend", "kill": "kill"}


def _cases(batch_dir: Path) -> list[dict]:
    cases = []
    for f in sorted(batch_dir.glob("rung-*.json")):
        rung = json.loads(f.read_text())
        rows = rung["eval"]["rows"]
        gates = {**judgemod.DEFAULT_GATES}
        for d in rung["decisions"]:
            rule, _ = judgemod.rule_verdict(rows[d["arm_id"]], rows["teacher"], gates)
            cases.append({"rung": rung["rung"], "arm_id": d["arm_id"], "rule": rule,
                          "final": d["verdict"], "tier": d["tier"],
                          "text": d.get("case_text", ""), "shadow": d.get("shadow", {})})
    return cases


def _load_model(name: str):
    if name == "jev":
        from . import jev as jevmod

        return judgemod.JevJudge(jevmod.JevClient())
    if name == "decide":
        from .decide import DecideClient

        return judgemod.DecideJudge(DecideClient())
    raise ValueError(f"unknown model {name!r}")


def rejudge(batch_dir: str | Path, models: list[str] | None = None) -> dict:
    batch_dir = Path(batch_dir)
    cases = _cases(batch_dir)
    for name in models or []:
        m = _load_model(name)
        for c in cases:
            try:
                c["shadow"][name] = judgemod._brief(m.ask(c["text"]))
            except Exception as e:
                c["shadow"][name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        c_log = getattr(m, "log", [])
        (batch_dir / f"rejudge-{name}-calls.json").write_text(json.dumps(c_log, indent=1))
    names = sorted({k for c in cases for k in c["shadow"]})
    report: dict = {"batch": batch_dir.name, "cases": len(cases), "models": {}}
    for name in names:
        settled = [c for c in cases if c["rule"] in RULE_TO_CHOICE
                   and "choice" in c["shadow"].get(name, {})]
        agree = [c for c in settled if c["shadow"][name]["choice"] == RULE_TO_CHOICE[c["rule"]]]
        by_rule = {}
        for r in ("keep", "kill"):
            sub = [c for c in settled if c["rule"] == r]
            by_rule[r] = {"n": len(sub),
                          "agree": sum(c["shadow"][name]["choice"] == RULE_TO_CHOICE[r]
                                       for c in sub)}
        # confident-and-wrong is what would hurt if the model were allowed to act
        conf_wrong = [c for c in settled if c not in agree
                      and c["shadow"][name]["confidence"] >= judgemod.DEFAULT_GATES["jev_act_alone"]]
        report["models"][name] = {
            "rule_settled": len(settled),
            "agreement": round(len(agree) / len(settled), 4) if settled else None,
            "by_rule": by_rule,
            "confident_and_wrong": len(conf_wrong),
            "mean_confidence": round(sum(c["shadow"][name]["confidence"] for c in settled)
                                     / len(settled), 4) if settled else None,
            "errors": sum("error" in c["shadow"].get(name, {}) for c in cases),
        }
    report["uncertain"] = [
        {"rung": c["rung"], "arm_id": c["arm_id"], "final": c["final"], "tier": c["tier"],
         **{n: c["shadow"].get(n, {}).get("choice") for n in names}}
        for c in cases if c["rule"] == "uncertain"
    ]
    report["cases_detail"] = [{k: c[k] for k in ("rung", "arm_id", "rule", "shadow")}
                              for c in cases]
    (batch_dir / "judges.json").write_text(json.dumps(report, indent=1))
    for name, r in report["models"].items():
        print(f"[rejudge] {name:>7}: agrees with the rules on {r['agreement']:.0%} of "
              f"{r['rule_settled']} settled cases (keep {r['by_rule']['keep']['agree']}/"
              f"{r['by_rule']['keep']['n']}, kill {r['by_rule']['kill']['agree']}/"
              f"{r['by_rule']['kill']['n']}); confident-and-wrong {r['confident_and_wrong']}; "
              f"mean confidence {r['mean_confidence']}")
    for u in report["uncertain"]:
        print(f"[rejudge] uncertain r{u['rung']} {u['arm_id']}: final={u['final']} via "
              f"{u['tier']}; " + " ".join(f"{n}={u.get(n)}" for n in names))
    return report
