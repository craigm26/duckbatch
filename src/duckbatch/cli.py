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

    hj = sub.add_parser("hf-job", help="re-run a menu on Hugging Face Jobs (billed to you)")
    hj.add_argument("menu")
    hj.add_argument("--dataset", required=True, help="dataset repo that receives the records")
    hj.add_argument("--flavor", default="l4x1")
    hj.add_argument("--timeout", default="4h")
    hj.add_argument("--commit", default=None, help="default: HEAD (must be pushed)")
    hj.add_argument("--jev", action="store_true", help="pass TYPESAFE_API_KEY to the job")
    hj.add_argument("--dry-run", action="store_true")

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

        launch(a.menu, a.dataset, a.flavor, a.timeout, a.commit, a.jev, a.dry_run)
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
