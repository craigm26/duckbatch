"""Run a batch: a menu of attempts, trained together, judged and halved rung by rung.

    rung 0: all K students share the env batch     -> eval (with teacher) -> judge -> keep K/eta
    rung 1: survivors inherit the dead arms' envs  -> eval -> judge -> keep K/eta^2
    ...
    final:  every arm that reached the last rung is exported to ONNX and re-evaluated on
            several held-out seeds, teacher alongside, for the numbers that get reported.

Everything lands in `records/<batch_id>/`: the menu as run, one `rung-N.json` per rung (eval rows,
decisions, Jev log), the exported policies, and `record.json` (schema `duckbatch.batch.v1`).
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..policy import mlp_flops, mlp_params
from . import judge as judgemod
from . import jev as jevmod

SCHEMA = "duckbatch.batch.v1"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def evaluate_both(pop, walk_s: float, recover_s: float, seed: int, record_obs: int = 0) -> dict:
    """Walk profile (no prone spawns, no deliberate topples) + recovery profile, one row per arm.
    The walk eval runs first so `recorded_obs` holds walking observations."""
    rows = pop.evaluate(seconds=walk_s, seed=seed, record_obs=record_obs, profile="walk")
    rec = pop.evaluate(seconds=recover_s, seed=seed + 500_000, profile="recover")
    for k, r in rec.items():
        rows[k].update({"recovered_frac": r["recovered_frac"], "t_up_s": r["t_up_s"]})
    return rows


def load_menu(path: str | Path) -> dict[str, Any]:
    menu = yaml.safe_load(Path(path).read_text())
    for key in ("batch_id", "teacher", "arms", "rungs"):
        if key not in menu:
            raise ValueError(f"menu {path} is missing `{key}`")
    ids = [a["id"] for a in menu["arms"]]
    if len(set(ids)) != len(ids) or "teacher" in ids:
        raise ValueError("arm ids must be unique and not 'teacher'")
    return menu


def run_batch(menu_path: str | Path, out_root: str | Path = "records", device: str = "cuda:0",
              use_jev: bool | None = None, log=print) -> Path:
    import torch

    from ..policy import export_onnx
    from ..sim import DEFAULT_TASK, Arm, Population, make_env

    menu = load_menu(menu_path)
    out = Path(out_root) / menu["batch_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "menu.yaml").write_text(Path(menu_path).read_text())

    task = menu.get("task", DEFAULT_TASK)
    num_envs = int(menu.get("num_envs", 512))
    eta = int(menu.get("eta", 2))
    seed = int(menu.get("seed", 0))
    gates = menu.get("gates", {})
    # Early rungs (all but the last) may carry looser kill lines: successive halving prunes
    # early by RANK, and absolute bars belong on the final, full-budget rung (b001 lesson).
    gates_early = {**gates, **menu.get("gates_early", {})}
    ev = menu.get("eval", {})
    rung_eval_s = float(ev.get("rung_seconds", 20.0))
    recover_s = float(ev.get("recover_seconds", 6.0))
    rung_eval_seed = int(ev.get("rung_seed", 1000))
    final_seeds = [int(s) for s in ev.get("final_seeds", [2001, 2002, 2003])]
    final_s = float(ev.get("final_seconds", 30.0))
    train = menu.get("train", {})

    # Decision models. Every available one runs in shadow on every case; only `judge.act`
    # (in order) may decide an uncertain case. Pre-registered in the menu.
    jcfg = menu.get("judge", {})
    want = jcfg.get("models", ["decide", "jev"])
    models: dict[str, Any] = {}
    if "jev" in want and (use_jev if use_jev is not None else jevmod.available()):
        models["jev"] = judgemod.JevJudge(jevmod.JevClient())
    if "decide" in want:
        from . import decide as decidemod

        if decidemod.available():
            models["decide"] = judgemod.DecideJudge(decidemod.DecideClient())
    actors = [a for a in jcfg.get("act", ["jev"]) if a in models]
    chain = "->".join(["rule", *actors, "person"])
    log(f"[batch] {menu['batch_id']}: {len(menu['arms'])} arms, {num_envs} envs, "
        f"{len(menu['rungs'])} rungs, eta={eta}, judge={chain}, "
        f"shadow={sorted(models) or 'none'}")

    t_start = time.perf_counter()
    env = make_env(task, num_envs, device=device, seed=seed)
    arms = [Arm("teacher", "teacher")] + [
        Arm(a["id"], "student", tuple(a["hidden"]), float(a.get("lr", 1e-3)),
            a.get("activation", "Elu"))
        for a in menu["arms"]
    ]
    pop = Population(env, menu["teacher"], arms, device=device, seed=seed,
                     buffer_size=int(train.get("buffer_size", 262_144)))

    closed: dict[str, dict] = {}
    it = 0
    rung_files = []
    for r, rung in enumerate(menu["rungs"]):
        iters = int(rung["iters"])
        live = [a.arm_id for a in pop.arms if a.kind == "student"]
        log(f"[batch] rung {r}: {len(live)} students x ~{num_envs // (len(live) + 1)} envs, "
            f"{iters} iters")
        t0 = time.perf_counter()
        pop.train(iters, steps_per_iter=int(train.get("steps_per_iter", 24)),
                  grad_steps=int(train.get("grad_steps", 20)),
                  beta0=float(train.get("beta0", 1.0)),
                  beta_decay_iters=int(train.get("beta_decay_iters", 40)),
                  start_iter=it, log=log)
        it += iters
        train_s = time.perf_counter() - t0
        rows = evaluate_both(pop, rung_eval_s, recover_s, rung_eval_seed)
        meta = {a.arm_id: {"loss_trail": [h["loss"] for h in a.history],
                           "params": mlp_params(a.hidden)}
                for a in pop.arms if a.kind == "student"}
        last = r == len(menu["rungs"]) - 1
        n_log = {k: len(m.log) for k, m in models.items()}
        decisions = judgemod.judge_rung(rows, "teacher", meta, r, len(menu["rungs"]) - r - 1,
                                        gates if last else gates_early, models, actors)
        quota = len(live) if last else max(1, math.ceil(len(live) / eta))
        keep = judgemod.survivors(decisions, rows, "teacher", quota,
                                  mode=menu.get("rank", "quality"),
                                  params={k: v["params"] for k, v in meta.items()})
        for d in decisions:
            if d.arm_id not in keep and d.verdict != "kill":
                d.reasons.append(f"outranked (quota {quota})")
        rung_rec = {
            "rung": r, "iters": iters, "train_seconds": round(train_s, 1),
            "gates": {**judgemod.DEFAULT_GATES, **(gates if last else gates_early)},
            "eval": {"seconds": rung_eval_s, "recover_seconds": recover_s,
                     "seed": rung_eval_seed, "profile": "walk+recover", "rows": rows},
            "decisions": [d.as_dict() for d in decisions],
            "advance": keep,
            "model_calls": {k: m.log[n_log[k]:] for k, m in models.items()},
        }
        f = out / f"rung-{r}.json"
        f.write_text(json.dumps(rung_rec, indent=1))
        rung_files.append(f.name)
        for d in decisions:
            sh = " ".join(f"{k}:{v.get('choice', 'ERR')}@{v.get('confidence', 0):.2f}"
                          for k, v in d.shadow.items())
            log(f"[judge] r{r} {d.arm_id:>10} {d.verdict:>7} via {d.tier:<6} "
                f"{'ADVANCE' if d.arm_id in keep else 'closed '} {'; '.join(d.reasons)}"
                f"{'  | shadow ' + sh if sh else ''}")
        # Close the losers now: their envs go to the survivors from the next rung on.
        for a in list(pop.arms):
            if a.kind == "student" and a.arm_id not in keep:
                dead = pop.drop_arm(a.arm_id)
                dec = next(d for d in decisions if d.arm_id == a.arm_id)
                closed[a.arm_id] = {"closed_at_rung": r, "verdict": dec.verdict,
                                    "tier": dec.tier, "reasons": dec.reasons,
                                    "samples_seen": dead.samples_seen}
        if not any(a.kind == "student" for a in pop.arms):
            log("[batch] every attempt closed; nothing to export")
            break

    # Export the finalists, then the reported numbers: held-out seeds, teacher alongside.
    finalists = [a for a in pop.arms if a.kind == "student"]
    teacher_meta = dict(pop.teacher.metadata)
    policies = {}
    for a in finalists:
        meta = {k: v for k, v in teacher_meta.items()
                if k in ("joint_names", "default_joint_pos", "command_names",
                         "observation_names", "action_scale")}
        meta.update({"duckbatch_batch": menu["batch_id"], "duckbatch_arm": a.arm_id,
                     "duckbatch_teacher": Path(menu["teacher"]).name,
                     "run_path": f"duckbatch/{menu['batch_id']}/{a.arm_id}"})
        p = export_onnx(a.net, out / "policies" / a.arm_id / "policy.onnx", meta)
        policies[a.arm_id] = str(p.relative_to(out))
    final = {}
    if finalists:
        for i, s in enumerate(final_seeds):
            final[str(s)] = evaluate_both(pop, final_s, recover_s, s,
                                          record_obs=4096 if i == 0 else 0)
            if i == 0:
                import numpy as np
                np.save(out / "obs_sample.npy", torch.cat(pop.recorded_obs).numpy())
    total_s = time.perf_counter() - t_start

    arm_rows = {}
    for a in menu["arms"]:
        h = tuple(a["hidden"])
        arm_rows[a["id"]] = {
            "hidden": list(h), "lr": a.get("lr", 1e-3), "params": mlp_params(h),
            "flops": mlp_flops(h), "status": "finalist" if a["id"] in policies else "closed",
            **closed.get(a["id"], {}),
        }
        if a["id"] in policies:
            arm_rows[a["id"]]["policy"] = policies[a["id"]]
    record = {
        "schema": SCHEMA,
        "batch_id": menu["batch_id"],
        "created": _now(),
        "task": task,
        "teacher": {"file": Path(menu["teacher"]).name,
                    "params": mlp_params((512, 256, 128)), "flops": mlp_flops((512, 256, 128)),
                    "source": menu.get("teacher_source")},
        "executed": {
            "host": platform.node(), "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "num_envs": num_envs, "wall_seconds": round(total_s, 1),
            "duckbatch_commit": _git_sha(),
            "judge": chain,
            "shadow_models": sorted(models),
            "rank": menu.get("rank", "quality"),
        },
        "gates": {**judgemod.DEFAULT_GATES, **gates},
        "rungs": rung_files,
        "arms": arm_rows,
        "final_eval": {"seconds": final_s, "recover_seconds": recover_s, "seeds": final_seeds,
                       "profile": "walk+recover", "by_seed": final},
        "model_calls_total": {k: len(m.log) for k, m in models.items()},
    }
    (out / "record.json").write_text(json.dumps(record, indent=1))
    log(f"[batch] done in {total_s / 60:.1f} min -> {out}/record.json")
    env.close()
    return out
