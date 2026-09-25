"""Re-run a batch on Hugging Face Jobs: no local GPU needed to reproduce a result.

    duckbatch hf-job menus/b002-student-size-longer.yaml --dataset <you>/duckbatch-records
    duckbatch hf-job menus/b002-student-size-longer.yaml --dataset <you>/duckbatch-records --dry-run

The job clones this repo at an exact commit (the one your checkout is on, which must be pushed),
installs the same locked environment, fetches Pollen's teachers by pinned revision and sha256,
runs the menu, and uploads `records/<batch_id>/` to the dataset under
`jobs/<job-id>/`. It never overwrites the records that were run locally.

Jobs are billed to the account that launches them (l4x1 by default; the batch fits a 4 GB
laptop GPU, so the cheapest CUDA flavor is plenty). The launcher pattern (pinned uv,
UV_LINK_MODE=copy) follows pollen-robotics/microduck_rl `hf_jobs.py`, for the same reasons.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = "https://github.com/craigm26/duckbatch.git"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
FLAVOR = "l4x1"
UV_VERSION = "0.12.18"

BOOTSTRAP = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -qq -y --no-install-recommends git curl ca-certificates >/dev/null
curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | sh >/dev/null
export PATH="/root/.local/bin:$PATH"
export UV_LINK_MODE=copy
git clone -q "$REPO" /work && cd /work && git checkout -q "$COMMIT"
uv sync --no-progress --extra sim --extra decide
./scripts/fetch_teachers.sh
set +e
uv run python -u -m duckbatch.cli $RUN "$MENU" --out /work/job-records $JUDGE_FLAG
RC=$?
set -e
export BATCH_DIR=$(ls -d /work/job-records/*/ | head -1)
uv run python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(os.environ["DATASET"], repo_type="dataset", exist_ok=True)
batch = os.environ["BATCH_DIR"].rstrip("/").split("/")[-1]
api.upload_folder(repo_id=os.environ["DATASET"], repo_type="dataset",
                  folder_path=os.environ["BATCH_DIR"],
                  path_in_repo=f"jobs/{os.environ.get('JOB_ID', 'job')}/{batch}",
                  commit_message=f"duckbatch HF Job {batch} @ {os.environ['COMMIT'][:7]}")
print("[job] records uploaded")
PY
exit $RC
"""


def _head_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    pushed = subprocess.run(["git", "branch", "-r", "--contains", sha], capture_output=True,
                            text=True).stdout.strip()
    if not pushed:
        raise SystemExit(f"commit {sha[:7]} is not on any remote branch: push it first, the job "
                         f"clones {REPO} and cannot see local commits")
    return sha


def launch(menu: str, dataset: str, flavor: str = FLAVOR, timeout: str = "4h",
           commit: str | None = None, jev: bool = False, dry_run: bool = False,
           run: str = "batch") -> str | None:
    from .batch.jev import load_dotenv

    load_dotenv()
    import os

    if not Path(menu).is_file():
        raise SystemExit(f"no menu at {menu}")
    commit = commit or _head_commit()
    env = {"REPO": REPO, "COMMIT": commit, "MENU": menu, "DATASET": dataset,
           "UV_VERSION": UV_VERSION, "RUN": run,
           "JUDGE_FLAG": ("" if jev else "--no-jev") if run == "batch" else ""}
    secrets = {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}
    if jev:
        secrets["TYPESAFE_API_KEY"] = os.environ.get("TYPESAFE_API_KEY", "")
    print(f"[hf-job] duckbatch {run} {menu} @ {commit[:7]} on {flavor} (timeout {timeout}) -> datasets/{dataset}")
    print(f"[hf-job] image {IMAGE}; secrets {sorted(k for k, v in secrets.items() if v)}")
    if dry_run:
        print("[hf-job] dry run: nothing submitted")
        return None
    if not secrets["HF_TOKEN"]:
        raise SystemExit("HF_TOKEN is not set (environment or .env): the job needs it to upload")
    from huggingface_hub import HfApi

    job = HfApi().run_job(image=IMAGE, command=["bash", "-c", BOOTSTRAP], env=env,
                          secrets=secrets, flavor=flavor, timeout=timeout)
    print(f"[hf-job] submitted {job.id}: {job.url}")
    return job.id
