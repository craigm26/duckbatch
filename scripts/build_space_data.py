"""Collect a batch's record, bench, judge report and Hub repo ids into space/data.json.

    uv run python scripts/build_space_data.py records/b001-student-size [--namespace craigm26]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev

METRICS = ("falls_per_min", "down_frac", "lin_err", "ang_err", "teacher_mse", "recovered_frac",
           "t_up_s")


def _finite(x):
    """NaN/inf -> null: browsers' JSON.parse rejects NaN (an arm that never got up has no t_up)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir")
    ap.add_argument("--namespace", default="craigm26")
    ap.add_argument("--out", default="space/data.json")
    a = ap.parse_args()
    bd = Path(a.batch_dir)
    rec = json.loads((bd / "record.json").read_text())
    bench = {}
    if (bd / "bench.json").is_file():
        for r in json.loads((bd / "bench.json").read_text()):
            bench[Path(r["file"]).parent.name] = r
    judges = json.loads((bd / "judges.json").read_text()) if (bd / "judges.json").is_file() else None

    def stats(who):
        rows = [s[who] for s in rec["final_eval"]["by_seed"].values() if who in s]
        if not rows:
            return None
        out = {}
        for m in METRICS:
            v = [r[m] for r in rows if m in r and math.isfinite(r[m])]
            out[m] = ({"mean": mean(v), "sd": pstdev(v), "min": min(v), "max": max(v)} if v
                      else {"mean": None, "sd": None, "min": None, "max": None})
        return out

    from duckbatch.publish import repo_name  # noqa: E402  (same naming as the publisher)

    arms = []
    for arm_id, arm in rec["arms"].items():
        row = {"id": arm_id, "hidden": arm["hidden"], "params": arm["params"],
               "flops": arm["flops"], "status": arm["status"],
               "closed_at_rung": arm.get("closed_at_rung"), "verdict": arm.get("verdict"),
               "tier": arm.get("tier"), "reasons": arm.get("reasons", []),
               "final": stats(arm_id)}
        if arm["status"] == "finalist":
            row["repo"] = f"{a.namespace}/{repo_name(rec, arm_id)}"
        if arm_id in bench:
            b = bench[arm_id]
            row["bench"] = {"p50_us": b["fp32"]["p50_us"], "p99_us": b["fp32"]["p99_us"],
                            "bytes": b["bytes"],
                            "int8_p50_us": b.get("int8", {}).get("p50_us"),
                            "int8_drift": b.get("int8", {}).get("drift_vs_fp32", {}).get("max_abs")}
        arms.append(row)
    rungs = []
    for f in rec["rungs"]:
        r = json.loads((bd / f).read_text())
        rungs.append({"rung": r["rung"], "iters": r["iters"], "train_seconds": r["train_seconds"],
                      "advance": r["advance"],
                      "decisions": [{k: d.get(k, {} if k == "shadow" else None)
                                     for k in ("arm_id", "verdict", "tier", "reasons", "shadow")}
                                    for d in r["decisions"]]})
    teacher = {"params": rec["teacher"]["params"], "flops": rec["teacher"]["flops"],
               "source": rec["teacher"].get("source"), "final": stats("teacher")}
    if "teacher" in bench:
        b = bench["teacher"]
        teacher["bench"] = {"p50_us": b["fp32"]["p50_us"], "p99_us": b["fp32"]["p99_us"],
                            "bytes": b["bytes"],
                            "int8_p50_us": b.get("int8", {}).get("p50_us"),
                            "int8_drift": b.get("int8", {}).get("drift_vs_fp32", {}).get("max_abs")}
    data = {"batch_id": rec["batch_id"], "created": rec["created"], "task": rec["task"],
            "executed": rec["executed"], "final_eval": {k: rec["final_eval"][k]
                                                        for k in ("seconds", "seeds")},
            "teacher": teacher, "arms": arms, "rungs": rungs,
            "judges": {k: judges[k] for k in ("models", "uncertain")} if judges else None,
            "gates": rec["gates"]}
    Path(a.out).write_text(json.dumps(_finite(data), indent=1, allow_nan=False))
    print(f"wrote {a.out}: {len(arms)} arms, {len(rungs)} rungs")


if __name__ == "__main__":
    main()
