#!/bin/bash
# Generated from duckbatch.hf_job.BOOTSTRAP; do not edit.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -qq -y --no-install-recommends git curl ca-certificates >/dev/null
curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | sh >/dev/null
export PATH="/root/.local/bin:$PATH"
export UV_LINK_MODE=copy
git clone -q "$REPO" /work && cd /work && git checkout -q "$COMMIT"
# A MENU CAN ARRIVE AS TEXT. The code always comes from the pinned commit; the menu may be
# written somewhere that cannot push a commit (Microduck Studio on a Mac), so it travels in
# MENU_YAML and is written here. The record still carries it: finetune copies the menu in.
if [ -n "${MENU_YAML:-}" ]; then printf '%s\n' "$MENU_YAML" > /work/menu-inline.yaml; MENU=/work/menu-inline.yaml; fi
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
