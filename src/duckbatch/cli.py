"""`duckbatch <command>`: probe, batch, bench, publish."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="duckbatch", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="env-steps/s and VRAM for a task at several env counts")
    pr.add_argument("--task", default=None)
    pr.add_argument("--envs", default="128,256,512", help="comma-separated env counts")

    b = sub.add_parser("batch", help="run a menu: train, judge and halve a batch of attempts")
    b.add_argument("menu")
    b.add_argument("--out", default="records")
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--no-jev", action="store_true", help="rule -> person only")

    be = sub.add_parser("bench", help="params, FLOPs, bytes and 1-thread latency of ONNX policies")
    be.add_argument("onnx", nargs="+")
    be.add_argument("--int8", action="store_true", help="also quantize and measure drift")
    be.add_argument("--obs", default=None, help="recorded observations .npy for inputs/drift")
    be.add_argument("--runs", type=int, default=5000)
    be.add_argument("--out", default=None)

    rj = sub.add_parser("rejudge", help="replay/measure decision models on a batch's cases")
    rj.add_argument("batch_dir")
    rj.add_argument("--models", default="", help="comma-separated: jev,decide (ask them now)")

    pu = sub.add_parser("publish", help="push a batch's finalists and records to the HF Hub")
    pu.add_argument("batch_dir")
    pu.add_argument("--namespace", required=True, help="HF user or org")
    pu.add_argument("--arms", default=None, help="comma-separated arm ids (default: finalists)")
    pu.add_argument("--dataset", default=None, help="dataset repo for the records")
    pu.add_argument("--private", action="store_true")
    pu.add_argument("--dry-run", action="store_true", help="write the repo folders, upload nothing")
    pu.add_argument("--bench", default=None, help="bench.json (default: <batch>/bench.json)")
    pu.add_argument("--space", default=None, help="also upload space/ to this Space repo id")

    hj = sub.add_parser("hf-job", help="run a menu on Hugging Face Jobs (billed to you)")
    hj.add_argument("menu")
    hj.add_argument("--run", default="batch", choices=["batch", "finetune", "skills"],
                    help="which duckbatch command runs the menu in the job")
    hj.add_argument("--dataset", required=True, help="dataset repo that receives the records")
    hj.add_argument("--flavor", default="l4x1")
    hj.add_argument("--timeout", default="4h")
    hj.add_argument("--commit", default=None, help="default: HEAD (must be pushed)")
    hj.add_argument("--jev", action="store_true", help="pass TYPESAFE_API_KEY to the job")
    hj.add_argument("--dry-run", action="store_true")

    ro = sub.add_parser("route", help="plain language -> proposed duck steps")
    ro.add_argument("text")
    ro.add_argument("--model", default="decide", choices=["decide", "jev"])

    re_ = sub.add_parser("route-eval", help="score the router on a pre-registered request set")
    re_.add_argument("menu")
    re_.add_argument("--models", default="decide,jev")
    re_.add_argument("--out", default="records")

    fb = sub.add_parser("feedback", help="duck-feedback/0: validate, report, export")
    fb.add_argument("action", choices=["validate", "report", "export-decide", "pull"])
    fb.add_argument("paths", nargs="*", help=".jsonl files or directories of them (not for pull)")
    fb.add_argument("--out", default=None, help="export-decide: the JSONL; pull: the directory")
    fb.add_argument("--require-share", default=None, choices=["research", "public"],
                    help="refuse records whose consent.share is not this (public dataset: public)")
    fb.add_argument("--pr", type=int, default=None, help="pull: one open pull request instead of main")

    pa = sub.add_parser("pairs", help="paired rollouts of several policies under identical conditions")
    pa.add_argument("--out", default="records/p001-pairs")
    pa.add_argument("--envs", type=int, default=512)
    pa.add_argument("--seconds", type=float, default=8.0)

    pl = sub.add_parser("plan", help="plan a multi-step request with the local LLM planner")
    pl.add_argument("text")
    pl.add_argument("--url", default=None)
    pe = sub.add_parser("plan-eval", help="score the LLM planner and the router on p002")
    pe.add_argument("menu", nargs="?", default="menus/p002-plan-requests.yaml")
    pe.add_argument("--models", default="gemma,decide")
    pe.add_argument("--url", default=None)
    pe.add_argument("--out", default="records/p002-plan-requests")
    sk = sub.add_parser("skills", help="distil several of Pollen's skills into one student")
    sk.add_argument("menu")
    sk.add_argument("--out", default="records")

    fi = sub.add_parser("finetune", help="PPO fine-tune a distilled student, anchored to its teacher")
    fi.add_argument("menu")
    fi.add_argument("--out", default="records")

    sc = sub.add_parser("speed-curve", help="achieved speed vs commanded speed, per walker (mjlab)")
    sc.add_argument("--out", default="records/s000-speed-curve")
    sc.add_argument("--policy", action="append", default=None, help="name=path.onnx (repeatable)")

    pf = sub.add_parser("prefer", help="preference model: size it with simulated raters, or fit real records")
    pf.add_argument("action", choices=["sizing", "fit"])
    pf.add_argument("pairs", help="a `duckbatch pairs` output directory")
    pf.add_argument("feedback", nargs="*", help="fit: duck-feedback/0 .jsonl files or dirs")
    pf.add_argument("--out", default=None)

    a = p.parse_args(argv)
    if a.cmd == "probe":
        from .probe import probe

        probe(a.task, [int(x) for x in a.envs.split(",")])
    elif a.cmd == "batch":
        from .batch.runner import run_batch

        run_batch(a.menu, a.out, a.device, use_jev=False if a.no_jev else None)
    elif a.cmd == "bench":
        from .bench import main as bench_main

        bench_main(a.onnx, a.int8, a.out, a.runs, a.obs)
    elif a.cmd == "rejudge":
        from .batch.rejudge import rejudge

        rejudge(a.batch_dir, [m for m in a.models.split(",") if m])
    elif a.cmd == "hf-job":
        from .hf_job import launch

        launch(a.menu, a.dataset, a.flavor, a.timeout, a.commit, a.jev, a.dry_run, run=a.run)
    elif a.cmd in ("route", "route-eval"):
        import json

        from . import router

        make = {"decide": router.DecideRouter, "jev": router.JevRouter}
        if a.cmd == "route":
            print(json.dumps(router.route(a.text, make[a.model]()), indent=1))
        else:
            import yaml
            from pathlib import Path

            menu = yaml.safe_load(Path(a.menu).read_text())
            out = Path(a.out) / menu["id"]
            out.mkdir(parents=True, exist_ok=True)
            for name in a.models.split(","):
                res = router.evaluate(menu["requests"], make[name]())
                (out / f"{name}.json").write_text(json.dumps(res, indent=1))
                print(f"[route-eval] {name:>6}: action {res['counts']['action']} "
                      f"speed {res['counts']['speed']} head {res['counts']['head']} "
                      f"sound {res['counts']['sound']} | exact {res['exact_requests']} "
                      f"| out-of-scope refused {res['out_of_scope_refused']} "
                      f"| mean conf when wrong {res['mean_confidence_when_wrong']}")
    elif a.cmd == "feedback":
        import json

        from . import feedback

        if a.action == "pull":
            out = feedback.pull(a.out or "records/community-feedback", pr=a.pr)
            a.paths = [str(out)]
            a.require_share = a.require_share or "public"
            print(f"[feedback] pulled {feedback.COMMUNITY_DATASET}"
                  f"{f' PR #{a.pr}' if a.pr is not None else ' main'} -> {out}")
        records = feedback.read(a.paths, a.require_share)
        kinds = {k: sum(r["kind"] == k for r in records) for k in feedback.KINDS}
        held = sum(feedback.is_held_out(r["id"]) for r in records)
        print(f"[feedback] {len(records)} valid records {kinds}; {held} held out")
        if a.action == "report":
            print(json.dumps(feedback.calibration(records), indent=1))
        elif a.action == "export-decide":
            if not a.out:
                raise SystemExit("export-decide needs --out")
            print(json.dumps(feedback.export_decide(records, a.out)))
    elif a.cmd == "pairs":
        from .pairs import generate

        generate(a.out, num_envs=a.envs, seconds=a.seconds)
    elif a.cmd == "plan":
        import json

        from . import planner

        print(json.dumps(planner.plan(a.text, a.url or planner.DEFAULT_URL), indent=1))
    elif a.cmd == "plan-eval":
        import json
        from pathlib import Path

        import yaml

        from . import planner, router

        menu = yaml.safe_load(Path(a.menu).read_text())
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for name in a.models.split(","):
            if name == "gemma":
                res = planner.evaluate(menu["requests"],
                                       lambda t: planner.plan(t, a.url or planner.DEFAULT_URL), "gemma")
            else:
                model = {"decide": router.DecideRouter, "jev": router.JevRouter}[name]()
                res = planner.evaluate(menu["requests"], lambda t: router.route(t, model), name)
            (out / f"{name}.json").write_text(json.dumps(res, indent=1))
            print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    elif a.cmd == "skills":
        from .skills import run_skills

        run_skills(a.menu, a.out)
    elif a.cmd == "finetune":
        from .finetune import run_finetune

        run_finetune(a.menu, a.out)
    elif a.cmd == "speed-curve":
        from .pairs import speed_curve

        pols = dict(p.split("=", 1) for p in a.policy) if a.policy else None
        speed_curve(a.out, pols)
    elif a.cmd == "prefer":
        import json

        from . import feedback, preference

        ps = preference.PairSet.load(a.pairs)
        if a.action == "sizing":
            res = preference.sizing(ps)
            for temp, curve in res["curves"].items():
                print(f"[prefer] SIMULATED raters, {temp} (noise ceiling {curve['ceiling']}):")
                for r in curve["rows"]:
                    print(f"   N={r['n']:>4}  held-out acc {r['accuracy']:.3f}  tau {r['kendall_tau']:+.2f} "
                          f"(min {r['tau_min']:+.2f})  cos(w,w*) {r['cos_w']:.2f}  "
                          f"beta {r['beta']:+.2f}±{r['beta_sd']:.2f} (true +0.30)")
            print("[prefer] true ranking:", " > ".join(res["true_ranking"]))
        else:
            res = preference.fit_records(ps, feedback.read(a.feedback))
            print(json.dumps(res, indent=1))
        if a.out:
            from pathlib import Path

            Path(a.out).write_text(json.dumps(res, indent=1))
    elif a.cmd == "publish":
        from .publish import publish_batch

        publish_batch(a.batch_dir, a.namespace, arms=a.arms.split(",") if a.arms else None,
                      dataset=a.dataset, private=a.private, dry_run=a.dry_run,
                      bench_file=a.bench)
        if a.space and not a.dry_run:
            from .publish import publish_space

            publish_space("space", a.space, private=a.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
