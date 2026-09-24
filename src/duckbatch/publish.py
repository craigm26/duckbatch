"""Push a batch to the Hub: one model repo per finalist, one dataset for every record.

A policy repo is exactly what Pollen's tooling reads (`pollen-robotics/microduck_rl`
`publish/manifest.py`, schema 2): the sole `policy.onnx` plus `manifest.json` at the root. So a
finalist is immediately:

  * playable in the browser: pollen-robotics/microduck-simulator `?move=<repo>`
  * loadable on a robot:     `robotctl policy load walk <repo>`
  * listed by duck-studio's community catalogue (`microduck` / `microduck-policy` tags)

`--dry-run` writes the repo folders under `<batch>/hub/` and uploads nothing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from statistics import mean, pstdev

SIM_URL = "https://pollen-robotics-microduck-simulator.hf.space/?move="
REPO_URL = "https://github.com/craigm26/duckbatch"
ROBOT = {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50}
METRICS = ("falls_per_min", "down_frac", "lin_err", "ang_err", "teacher_mse", "recovered_frac",
           "t_up_s")


def summarize(record: dict, arm_id: str) -> dict[str, dict[str, float]]:
    """Mean and spread over the final held-out seeds, for an arm and for the teacher."""
    by_seed = record["final_eval"]["by_seed"]
    out = {}
    for who in (arm_id, "teacher"):
        rows = [s[who] for s in by_seed.values() if who in s]
        out[who] = {}
        for m in METRICS:
            v = [r[m] for r in rows if m in r and r[m] == r[m]]  # r[m] == r[m] drops NaN
            out[who][m] = {"mean": mean(v) if v else float("nan"),
                           "sd": pstdev(v) if v else float("nan")}
    return out


def repo_name(record: dict, arm_id: str) -> str:
    hidden = "x".join(str(h) for h in record["arms"][arm_id]["hidden"])
    return f"microduck-duckbatch-{record['batch_id'].split('-')[0]}-{hidden}"


def build_manifest(record: dict, arm_id: str, summary: dict) -> dict:
    arm = record["arms"][arm_id]
    hidden = "-".join(str(h) for h in arm["hidden"])
    m = {
        "schema_version": 2,
        "model_api": 1,
        "obs_len": 61,
        "action_len": 14,
        "robot": dict(ROBOT),
        "name": f"duckbatch-{hidden}",
        "kind": "perpetual",
        "slot": "walk",
        "entry_pose": "standing",
        "description": (
            f"A {arm['params']:,}-parameter walking gait (61-{hidden}-14) distilled from Pollen's "
            f"default walker ({record['teacher']['params']:,} params) in simulation. Sim only: "
            f"never run on hardware."
        ),
        # `twist` as a list: duck-studio's PolicyManifest decodes [String] (a string decodes as
        # empty there); Pollen's validator only checks `encoding` and `idle`; the community
        # flamingo-cycle manifest uses the same list form.
        "command": {"encoding": "constant", "idle": [0.0, 0.0, 0.0],
                    "twist": ["vx (m/s)", "vy (m/s)", "wz (rad/s)"],
                    "head": "neck/head pose", "body": "unused (zeros)"},
        "duration_s": None,
        "training": {
            "repo": "craigm26/duckbatch",
            "method": "population DAgger distillation",
            "task_id": record["task"],
            "run": f"{record['batch_id']}/{arm_id}",
            "teacher": record["teacher"].get("source") or record["teacher"]["file"],
            "commit": record["executed"].get("duckbatch_commit"),
            "exported": record["created"],
        },
        "eval": {
            "where": f"mjlab {record['task']}, domain randomization and pushes on",
            "seconds_per_seed": record["final_eval"]["seconds"],
            "seeds": record["final_eval"]["seeds"],
            "student": {k: round(v["mean"], 4) for k, v in summary[arm_id].items()},
            "teacher_same_eval": {k: round(v["mean"], 4) for k, v in summary["teacher"].items()},
        },
    }
    try:  # Pollen's own validator, when the sim extra is installed
        from mjlab_microduck.publish.manifest import validate_manifest
    except ImportError:
        pass
    else:
        validate_manifest(m)
    return m


def model_card(record: dict, arm_id: str, repo_id: str, summary: dict, bench: dict | None) -> str:
    arm = record["arms"][arm_id]
    s, t = summary[arm_id], summary["teacher"]

    def cell(d, m, fmt):
        return f"{fmt.format(d[m]['mean'])} ± {fmt.format(d[m]['sd'])}"

    rows = [
        ("Falls per minute", "falls_per_min", "{:.2f}"),
        ("Time fallen", "down_frac", "{:.2%}"),
        ("Planar velocity error (m/s)", "lin_err", "{:.3f}"),
        ("Yaw rate error (rad/s)", "ang_err", "{:.3f}"),
        ("Gets up from prone within 6 s", "recovered_frac", "{:.1%}"),
        ("Time to get up (s)", "t_up_s", "{:.2f}"),
    ]
    table = "\n".join(f"| {name} | {cell(s, m, f)} | {cell(t, m, f)} |" for name, m, f in rows)
    lat = ""
    if bench:
        lat = (f"\n| 1-thread latency, p50 (x86 laptop) | {bench['student_p50_us']:.1f} us | "
               f"{bench['teacher_p50_us']:.1f} us |")
    return f"""---
license: apache-2.0
library_name: onnx
tags: [microduck, microduck-policy, robotics, reinforcement-learning, distillation, duckbatch]
---

# {repo_id}

A walking policy for Pollen Robotics' [Microduck](https://github.com/pollen-robotics/microduck):
{arm['params']:,} parameters (61-{'-'.join(map(str, arm['hidden']))}-14), distilled from Pollen's
default walker (`velstand.onnx`, {record['teacher']['params']:,} parameters) by
[duckbatch]({REPO_URL}), batch `{record['batch_id']}`, attempt `{arm_id}`.

**[Drive it in your browser]({SIM_URL}{repo_id})** (Pollen's simulator loads it from this repo).

**Simulation only.** It has never run on a real robot. A student that matches the teacher in
mjlab is a candidate for a hardware test, not a result about hardware.

## Measured

`mjlab` `{record['task']}` with domain randomization, stumble pushes and observation noise on;
the task's deliberate prone spawns and topple pushes are off for the walking numbers and on
(every episode prone) for the get-up numbers.
{record['final_eval']['seconds']:.0f} s per seed on held-out seeds
{', '.join(map(str, record['final_eval']['seeds']))}; mean ± spread over seeds. The teacher ran
in the same eval.

| | This policy | Teacher |
|---|---|---|
{table}
| Parameters | {arm['params']:,} | {record['teacher']['params']:,} |
| FLOPs per step | {arm['flops']:,} | {record['teacher']['flops']:,} |{lat}

## Use

- Browser: `{SIM_URL}{repo_id}`
- Robot: `robotctl policy load walk {repo_id}` (manifest schema 2, walk slot)
- Contract: `obs[1,61] -> actions[1,14]`, the normalizer baked in, the same as every Pollen policy.

Full record (training trace, every judge decision, the decision-model answers):
[{REPO_URL}]({REPO_URL}), `records/{record['batch_id']}/`.
"""


def publish_batch(batch_dir: str | Path, namespace: str, arms: list[str] | None = None,
                  dataset: str | None = None, private: bool = False, dry_run: bool = False,
                  bench_file: str | Path | None = None) -> list[str]:
    batch_dir = Path(batch_dir)
    record = json.loads((batch_dir / "record.json").read_text())
    arms = arms or [a for a, v in record["arms"].items() if v.get("status") == "finalist"]
    bench = {}
    bf = Path(bench_file) if bench_file else batch_dir / "bench.json"
    if bf.is_file():
        for row in json.loads(bf.read_text()):
            bench[Path(row["file"]).parent.name] = row
    out_root = batch_dir / "hub"
    api = None
    if not dry_run:
        from huggingface_hub import HfApi

        from .batch.jev import load_dotenv

        load_dotenv()  # HF_TOKEN from .env (gitignored), if it is not already in the environment

        api = HfApi()
    published = []
    for arm_id in arms:
        repo_id = f"{namespace}/{repo_name(record, arm_id)}"
        summary = summarize(record, arm_id)
        folder = out_root / repo_name(record, arm_id)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(batch_dir / record["arms"][arm_id]["policy"], folder / "policy.onnx")
        try:  # Pollen's pre-upload smoke run: shapes, NaN/inf, a network that never changes
            from mjlab_microduck.publish.manifest import check_onnx, smoke_run_onnx
        except ImportError:
            pass
        else:
            check_onnx(folder / "policy.onnx")
            smoke_run_onnx(folder / "policy.onnx")
        (folder / "manifest.json").write_text(
            json.dumps(build_manifest(record, arm_id, summary), indent=2) + "\n")
        b = None
        if arm_id in bench and "teacher" in bench:
            b = {"student_p50_us": bench[arm_id]["fp32"]["p50_us"],
                 "teacher_p50_us": bench["teacher"]["fp32"]["p50_us"]}
        (folder / "README.md").write_text(model_card(record, arm_id, repo_id, summary, b))
        print(f"[publish] {repo_id} <- {folder}")
        if api:
            api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
            api.upload_folder(repo_id=repo_id, folder_path=str(folder),
                              commit_message=f"duckbatch {record['batch_id']}/{arm_id}")
        published.append(repo_id)
    if dataset and api:
        api.create_repo(dataset, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(repo_id=dataset, repo_type="dataset", folder_path=str(batch_dir),
                          path_in_repo=f"records/{record['batch_id']}",
                          ignore_patterns=["hub/*", "*.int8.onnx"],
                          commit_message=f"duckbatch {record['batch_id']}")
        print(f"[publish] records -> datasets/{dataset}")
    return published


def publish_space(space_dir: str | Path, repo_id: str, private: bool = False) -> str:
    """Upload the static results page (index.html + data.json + README) as an HF Space."""
    from huggingface_hub import HfApi

    from .batch.jev import load_dotenv

    load_dotenv()
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="static", private=private,
                    exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(space_dir),
                      allow_patterns=["index.html", "data.json", "README.md"],
                      commit_message="duckbatch results page")
    print(f"[publish] space -> https://huggingface.co/spaces/{repo_id}")
    return repo_id
