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

# THE COLOUR MATRIX IS NAMED, NEVER LEFT TO THE DECODER. The app's videos are H.264 4:2:0, video
# range; the phone at run time decodes its JPEG straight to RGB. Decoding the video with a different
# YCbCr matrix than the encoder used shifts strong colours (the red ball most) between training and
# run time. Duck Studio's DreamTakeWriter tags BT.709 (duck-studio feat/driver-v2). Exports from before
# that carry no tag at all (color_space unknown), and for those the encoder measurably used BT.601:
# decoded frames match the app's own PNG renders at 1.1 to 1.4 mean abs error with BT.601 against
# 1.6 to 1.9 with BT.709 (dataset-v2, episodes 0, 57, 140). So: the stream's tag when it has one,
# BT.601 when it has none, and anything else refuses.
_MATRIX_BY_TAG = {1: "ITU709", 5: "ITU601", 6: "ITU601"}   # AVCOL_SPC_BT709, _BT470BG, _SMPTE170M
UNTAGGED = 2                                                 # AVCOL_SPC_UNSPECIFIED


def video_matrix(tag: int) -> str:
    if tag == UNTAGGED:
        return "ITU601"
    if tag not in _MATRIX_BY_TAG:
        raise SystemExit(f"video colour matrix tag {tag} is not one this script decodes (BT.709 or BT.601)")
    return _MATRIX_BY_TAG[tag]


def decode_video(path: pathlib.Path) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        matrix = video_matrix(int(stream.codec_context.colorspace))
        full = int(stream.codec_context.color_range) == 2           # AVCOL_RANGE_JPEG
        frames = [f.to_ndarray(format="rgb24", width=320, height=176, src_colorspace=matrix,
                               dst_colorspace=matrix, src_color_range="JPEG" if full else "MPEG",
                               dst_color_range="JPEG")
                  for f in c.decode(video=0)]
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
        if len(episodes) % 20 == 0:
            print(f"read {len(episodes)} episodes", flush=True)
    if not episodes:
        sys.exit(f"no episode has the task {task!r} (tasks here: {sorted(set(tasks.values()))})")
    return episodes


def samples(episodes, act_mean, act_std, offset: int):
    """Indices, not pictures: (now, then) rows into the stacked frames, plus targets and masks.

    Stacking every two-picture input up front costs 340 KB a sample (6.5 GB for 19,314 rows,
    twice during np.stack), more than a t4-small holds; batches are gathered from these instead."""
    now, then, ys, masks = [], [], [], []
    for frames, actions in episodes:
        n = len(actions)
        for t in range(n):
            now.append(offset + t)
            then.append(offset + max(t + FRAME_OFFSETS[1], 0))
            idx = np.minimum(np.arange(t, t + CHUNK), n - 1)
            ys.append(((actions[idx] - act_mean) / act_std).reshape(-1))
            masks.append(np.repeat((np.arange(t, t + CHUNK) < n).astype(np.float32), 3))
        offset += n
    return (np.array(now), np.array(then), np.stack(ys).astype(np.float32),
            np.stack(masks).astype(np.float32))


def gather(frames: torch.Tensor, now, then) -> torch.Tensor:
    """(B, 6, 88, 160): the newest picture's three channels first, as DriverPrep stacks them."""
    return torch.cat([frames[torch.as_tensor(now)], frames[torch.as_tensor(then)]], dim=1)


def train(args) -> None:
    from huggingface_hub import HfApi, snapshot_download
    root = pathlib.Path(snapshot_download(args.dataset, repo_type="dataset", revision=args.dataset_revision))
    episodes = load_episodes(root, args.task)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(episodes))
    n_val = max(1, len(episodes) // 10)
    val = [episodes[i] for i in order[:n_val]]; tr = [episodes[i] for i in order[n_val:]]
    all_actions = np.concatenate([a for _, a in episodes])
    act_mean = all_actions.mean(axis=0); act_std = all_actions.std(axis=0) + 1e-3
    # One stacked copy of the pictures, train episodes first, normalised in place.
    frames = np.concatenate([f for f, _ in tr] + [f for f, _ in val])
    n_train_rows = sum(len(a) for _, a in tr)
    episodes = [(None, a) for _, a in episodes]
    tr = [(None, a) for _, a in tr]; val = [(None, a) for _, a in val]
    img_mean = frames.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    img_std = (frames.std(axis=(0, 2, 3), dtype=np.float64) + 1e-3).astype(np.float32)
    frames -= img_mean[None, :, None, None]; frames /= img_std[None, :, None, None]
    nt, ot, yt, mt = samples(tr, act_mean, act_std, 0)
    nv, ov, yv, mv = samples(val, act_mean, act_std, n_train_rows)
    print(f"{len(tr)} train episodes ({len(nt)} rows), {len(val)} val episodes ({len(nv)} rows), "
          f"pictures {frames.nbytes / 1e9:.2f} GB", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = DuckNetS1().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    frames_t = torch.from_numpy(frames)
    yt_t, mt_t = torch.from_numpy(yt), torch.from_numpy(mt)
    for step in range(args.steps):
        b = rng.integers(0, len(nt), args.batch)
        x = gather(frames_t, nt[b], ot[b]).to(device)
        bt = torch.as_tensor(b)
        y = yt_t[bt].to(device); m = mt_t[bt].to(device)
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
        pred = np.concatenate([model(gather(frames_t, nv[i:i + 256], ov[i:i + 256])).numpy()
                               for i in range(0, len(nv), 256)])
    val_l1 = l1(pred)
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
