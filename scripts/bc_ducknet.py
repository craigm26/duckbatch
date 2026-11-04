# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "numpy>=1.26",
#   "safetensors>=0.4",
#   "huggingface_hub>=0.25",
#   "pyarrow>=15",
#   "av>=12",
#   "onnx>=1.16",
#   "onnxscript>=0.1",
# ]
# ///
"""Train Duck Driver S1, the phone driver of Microduck Studio build 90, or write its test fixture.

WHAT IT IS. Behaviour cloning: two head-camera pictures (now and 0.4 s ago) in, the next 48 walk
commands [vx, vy, vyaw] out. It reads the app's own LeRobot v2.1 export directly (no lerobot
install) and writes what the app installs: model.safetensors, policy.json, golden.json, plus
ducknet-s1.onnx (Pollen's deployment format, so the same driver could later run on the robot) and
a training bundle with provenance (after pollen-robotics/microduck_rl 3b8759e).

THE PREP MATCHES THE APP, LINE FOR LINE. StudioKit's DriverPrep: RGB 320x176, 2x2 average to
160x88, /255, per-channel (x - mean) / std, the newest picture's three channels first. The network
is DuckNetS1.swift's; tensor names and shapes are the contract between the two.

GOLDEN INPUTS NEED NO FILE. Each golden input is generated from a seed by SplitMix64 (the same
generator as StudioKit's), so golden.json holds three seeds and three outputs, not megabytes of
pixels. The phone regenerates the inputs, runs its own forward pass and compares.

    uv run scripts/bc_ducknet.py --golden-only --out-dir OUT            # the fixture, CPU, seconds
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN <raw url of this file> \
        --dataset craigm26/duck-drive-walk-to-ball --dataset-revision SHA \
        --task "walk to the ball" --steps 20000 --out craigm26/duck-driver-walk-to-ball
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

ARCH = "ducknet-s1"
CHUNK = 48
H, W = 88, 160
FRAME_OFFSETS = [0, -4]          # dataset frames at 10 fps: now and 0.4 s ago
GOLDEN_SEEDS = [1, 2, 3]
MASK64 = (1 << 64) - 1


class DuckNetS1(nn.Module):
    """Must stay identical to StudioKit/Sources/StudioKit/DuckNetS1.swift."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(6, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 10, 256)
        self.fc2 = nn.Linear(256, CHUNK * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)          # C, H, W order: what Swift flattens
        x = F.relu(self.fc1(x))
        return self.fc2(x)               # (N, 144) = 48 rows x [vx, vy, vyaw], row-major


def splitmix_inputs(seed: int, count: int) -> np.ndarray:
    """StudioKit's SplitMix64(seed:).next() mapped to [-1, 1), as float32."""
    state = seed & MASK64
    out = np.empty(count, dtype=np.float32)
    for i in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z ^= z >> 31
        out[i] = np.float32((z >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0)
    return out


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def golden(model: DuckNetS1) -> dict:
    model.eval()
    outputs = []
    with torch.no_grad():
        for seed in GOLDEN_SEEDS:
            x = torch.from_numpy(splitmix_inputs(seed, 6 * H * W)).reshape(1, 6, H, W)
            outputs.append([float(v) for v in model(x)[0].tolist()])
    return {"arch": ARCH, "input": "splitmix64(seed) * 2 - 1, float32, shape [6, 88, 160]",
            "seeds": GOLDEN_SEEDS, "outputs": outputs, "tolerance": 1e-3}


def write_driver(model: DuckNetS1, out: pathlib.Path, *, task: str, label: str, stats: dict,
                 extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous().float() for k, v in model.state_dict().items()}
    save_file(tensors, str(out / "model.safetensors"), metadata={"arch": ARCH})
    (out / "golden.json").write_text(json.dumps(golden(model), indent=1))
    try:
        torch.onnx.export(model.eval(), torch.zeros(1, 6, H, W), str(out / "ducknet-s1.onnx"),
                          input_names=["pictures"], output_names=["commands"], opset_version=17)
    except Exception as exc:  # ONNX is a bonus for the robot later, never a reason to fail
        print(f"onnx export skipped: {exc}", file=sys.stderr)
    files = {name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
             for name in ("model.safetensors", "golden.json") }
    manifest = {"arch": ARCH, "format": "safetensors", "task": task, "label": label,
                "input": [6, H, W], "frame_offsets": FRAME_OFFSETS, "chunk": CHUNK,
                "uses_state": False, "trained_on": "sim", **stats, "files": files, **extra}
    (out / "policy.json").write_text(json.dumps(manifest, indent=1))


def self_check(out: pathlib.Path) -> None:
    """Reload what was written and rerun the golden inputs: the file is the model."""
    from safetensors.torch import load_file
    model = DuckNetS1()
    model.load_state_dict(load_file(str(out / "model.safetensors")))
    again = golden(model)["outputs"]
    want = json.loads((out / "golden.json").read_text())["outputs"]
    worst = max(abs(a - b) for ra, rb in zip(again, want) for a, b in zip(ra, rb))
    if worst > 1e-6:
        sys.exit(f"self-check failed: reloaded weights differ by {worst}")
    print(f"self-check ok (max diff {worst:g})")


# --------------------------------------------------------------------------- training

def decode_video(path: pathlib.Path) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24", width=320, height=176) for f in c.decode(video=0)]
    return np.stack(frames)                                            # (T, 176, 320, 3) uint8


def pool(pictures: np.ndarray) -> np.ndarray:
    """(T, 176, 320, 3) uint8 -> (T, 3, 88, 160) float32 in [0, 1], 2x2 averaged like DriverPrep."""
    p = pictures.astype(np.float32).reshape(-1, 88, 2, 160, 2, 3).sum(axis=(2, 4)) / (4 * 255)
    return p.transpose(0, 3, 1, 2)


def load_episodes(root: pathlib.Path, task: str) -> list[tuple[np.ndarray, np.ndarray]]:
    import pyarrow.parquet as pq
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        sys.exit("expected a LeRobot v2.1 dataset")
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in (root / "meta/tasks.jsonl").read_text().splitlines() if l.strip()}
    episodes = []
    for line in (root / "meta/episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if task not in e["tasks"]:
            continue
        i = e["episode_index"]
        table = pq.read_table(root / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i))
        actions = np.array(table.column("action").to_pylist(), dtype=np.float32)
        video = root / info["video_path"].format(episode_chunk=i // info["chunks_size"],
                                                 video_key="observation.images.head", episode_index=i)
        frames = pool(decode_video(video))
        if len(frames) != len(actions):
            sys.exit(f"episode {i}: {len(frames)} video frames but {len(actions)} rows")
        episodes.append((frames, actions))
    if not episodes:
        sys.exit(f"no episode has the task {task!r} (tasks here: {sorted(set(tasks.values()))})")
    return episodes


def samples(episodes, img_mean, img_std, act_mean, act_std):
    xs, ys, masks = [], [], []
    for frames, actions in episodes:
        norm = (frames - img_mean[None, :, None, None]) / img_std[None, :, None, None]
        n = len(actions)
        for t in range(n):
            old = max(t + FRAME_OFFSETS[1], 0)
            xs.append(np.concatenate([norm[t], norm[old]], axis=0))
            idx = np.minimum(np.arange(t, t + CHUNK), n - 1)
            ys.append(((actions[idx] - act_mean) / act_std).reshape(-1))
            masks.append(np.repeat((np.arange(t, t + CHUNK) < n).astype(np.float32), 3))
    return np.stack(xs), np.stack(ys), np.stack(masks)


def train(args) -> None:
    from huggingface_hub import HfApi, snapshot_download
    root = pathlib.Path(snapshot_download(args.dataset, repo_type="dataset", revision=args.dataset_revision))
    episodes = load_episodes(root, args.task)
    all_frames = np.concatenate([f for f, _ in episodes])
    all_actions = np.concatenate([a for _, a in episodes])
    img_mean = all_frames.mean(axis=(0, 2, 3)); img_std = all_frames.std(axis=(0, 2, 3)) + 1e-3
    act_mean = all_actions.mean(axis=0); act_std = all_actions.std(axis=0) + 1e-3
    rng = np.random.default_rng(0)
    order = rng.permutation(len(episodes))
    n_val = max(1, len(episodes) // 10)
    val = [episodes[i] for i in order[:n_val]]; tr = [episodes[i] for i in order[n_val:]]
    xt, yt, mt = samples(tr, img_mean, img_std, act_mean, act_std)
    xv, yv, mv = samples(val, img_mean, img_std, act_mean, act_std)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = DuckNetS1().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    xt_t, yt_t, mt_t = map(torch.from_numpy, (xt, yt, mt))
    for step in range(args.steps):
        b = torch.from_numpy(rng.integers(0, len(xt), args.batch))
        x = xt_t[b].to(device); y = yt_t[b].to(device); m = mt_t[b].to(device)
        # Augment: brightness/contrast +-10% and shifts of up to 4 px (2 px pooled); no flips.
        x = x * (1 + 0.1 * (2 * torch.rand(len(x), 1, 1, 1, device=device) - 1))
        x = torch.roll(x, shifts=(int(rng.integers(-2, 3)), int(rng.integers(-2, 3))), dims=(2, 3))
        loss = ((model(x) - y).abs() * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    model = model.cpu().eval()

    def l1(pred: np.ndarray) -> float:
        return float((np.abs(pred - yv) * mv).sum() / mv.sum())
    with torch.no_grad():
        val_l1 = l1(model(torch.from_numpy(xv)).numpy())
    zero_l1 = l1(np.tile(((0 - act_mean) / act_std), CHUNK)[None, :].repeat(len(yv), 0))
    repeat_l1 = l1(np.repeat(yv[:, :3], CHUNK, axis=0).reshape(len(yv), -1))
    print(f"val L1 {val_l1:.4f}  zero baseline {zero_l1:.4f}  repeat-first baseline {repeat_l1:.4f}")
    out = pathlib.Path(args.out_dir or "driver-out")
    stats = {"img_mean": img_mean.tolist(), "img_std": img_std.tolist(),
             "act_mean": act_mean.tolist(), "act_std": act_std.tolist()}
    commit = os.environ.get("DUCKBATCH_COMMIT", "unknown")
    write_driver(model, out, task=args.task, label=args.label or args.task, stats=stats, extra={
        "dataset": {"repo": args.dataset, "commit": args.dataset_revision},
        "val_l1": val_l1, "baseline_zero_l1": zero_l1, "baseline_repeat_l1": repeat_l1,
        "duckbatch_commit": commit})
    self_check(out)
    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "training-checkpoint.pt")
    (out / "record.json").write_text(json.dumps({
        "command": sys.argv, "duckbatch_commit": commit, "dataset": args.dataset,
        "dataset_revision": args.dataset_revision, "steps": args.steps, "batch": args.batch,
        "episodes": {"train": len(tr), "val": len(val)},
        "sha256": {n: sha256(out / n) for n in ("model.safetensors", "training-checkpoint.pt")},
        "val_l1": val_l1, "baseline_zero_l1": zero_l1, "baseline_repeat_l1": repeat_l1}, indent=1))
    if args.out:
        api = HfApi()
        api.create_repo(args.out, exist_ok=True, private=False)
        api.upload_folder(repo_id=args.out, folder_path=str(out))
        print(f"pushed to https://huggingface.co/{args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--golden-only", action="store_true", help="random tiny weights + fixture files, no data")
    p.add_argument("--out-dir")
    p.add_argument("--dataset"); p.add_argument("--dataset-revision")
    p.add_argument("--task", default="walk to the ball"); p.add_argument("--label")
    p.add_argument("--steps", type=int, default=20000); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", help="HF model repo to push to (public)")
    args = p.parse_args()
    if args.golden_only:
        if not args.out_dir:
            sys.exit("--golden-only needs --out-dir")
        torch.manual_seed(1234)
        model = DuckNetS1()
        out = pathlib.Path(args.out_dir)
        stats = {"img_mean": [0.5, 0.5, 0.5], "img_std": [0.25, 0.25, 0.25],
                 "act_mean": [0.1, 0.0, 0.0], "act_std": [0.1, 0.05, 0.4]}
        write_driver(model, out, task="walk to the ball", label="fixture (random weights)", stats=stats,
                     extra={"fixture": True})
        self_check(out)
        return
    if not (args.dataset and args.dataset_revision):
        sys.exit("training needs --dataset and --dataset-revision (a pinned commit)")
    train(args)


if __name__ == "__main__":
    main()
