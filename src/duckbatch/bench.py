"""On-device cost of a policy: parameters, FLOPs, bytes, and single-thread latency (core install).

Latency is measured the way the robot runs it: onnxruntime, batch 1, ONE intra-op thread, graph
optimization level 3 (pollen-robotics/microduck `duck-control/src/policy.rs`). Absolute numbers
are for the host they ran on, not the robot's RK3566 (4x Cortex-A55); the ratios between
policies are the portable part, and the host is written into every result.

`--int8` also writes a dynamically quantized copy (int8 weights; activations quantized per call)
and reports how far its actions drift from the fp32 policy. Drift MUST be measured on real
observations (`--obs`, an [N,61] .npy recorded from the sim): synthetic Gaussians divided by the
normalizer's tiny per-dim std blow up far past anything the robot sees and wreck the per-tensor
activation range, which exaggerates the drift by orders of magnitude.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .policy import OBS_LEN, mlp_flops, onnx_mlp_weights


def _session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


_OBS: np.ndarray | None = None


def set_observations(path: str | Path | None) -> None:
    """Use recorded observations ([N,61] .npy) for latency inputs and drift."""
    global _OBS
    _OBS = None if path is None else np.load(path).astype(np.float32).reshape(-1, 1, OBS_LEN)


def _inputs(n: int, seed: int = 0) -> np.ndarray:
    if _OBS is not None:
        idx = np.random.default_rng(seed).integers(0, len(_OBS), n)
        return _OBS[idx]
    return np.random.default_rng(seed).normal(0.0, 0.5, size=(n, 1, OBS_LEN)).astype(np.float32)


def latency_us(path: Path, runs: int = 5000, warmup: int = 500) -> dict[str, float]:
    s = _session(path)
    name = s.get_inputs()[0].name
    xs = _inputs(64)
    for i in range(warmup):
        s.run(None, {name: xs[i % 64]})
    t = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter_ns()
        s.run(None, {name: xs[i % 64]})
        t[i] = time.perf_counter_ns() - t0
    t /= 1000.0
    return {"p50_us": float(np.percentile(t, 50)), "p99_us": float(np.percentile(t, 99)),
            "mean_us": float(t.mean())}


def quantize_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def action_drift(a: Path, b: Path, n: int = 2048) -> dict[str, float]:
    sa, sb = _session(a), _session(b)
    na, nb = sa.get_inputs()[0].name, sb.get_inputs()[0].name
    xs = _inputs(n, seed=1)
    d = np.stack([np.abs(sa.run(None, {na: x})[0] - sb.run(None, {nb: x})[0])[0] for x in xs])
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean())}


def bench(path: str | Path, int8: bool = False, runs: int = 5000) -> dict:
    path = Path(path)
    w = onnx_mlp_weights(path)
    row = {
        "file": str(path),
        "hidden": list(w.hidden),
        "params": w.n_params,
        "flops": mlp_flops(w.hidden),
        "bytes": path.stat().st_size,
        "fp32": latency_us(path, runs),
        "inputs": "recorded" if _OBS is not None else "synthetic",
        "host": {"machine": platform.machine(), "processor": platform.processor() or None,
                 "system": platform.system(), "cpus": os.cpu_count(), "ort": ort.__version__,
                 "threads": 1},
    }
    if int8:
        q = quantize_int8(path, path.with_name(path.stem + ".int8.onnx"))
        row["int8"] = {"file": str(q), "bytes": q.stat().st_size, **latency_us(q, runs),
                       "drift_vs_fp32": action_drift(path, q)}
    return row


def bench_many(paths: list[str | Path], int8: bool = False, runs: int = 5000) -> list[dict]:
    return [bench(p, int8, runs) for p in paths]


def main(paths: list[str], int8: bool, out: str | None, runs: int,
         obs: str | None = None) -> None:
    set_observations(obs)
    rows = bench_many(paths, int8, runs)
    for r in rows:
        line = (f"{Path(r['file']).parent.name + '/' + Path(r['file']).name:<44} "
                f"params={r['params']:>7,} flops={r['flops']:>8,} "
                f"p50={r['fp32']['p50_us']:6.1f}us p99={r['fp32']['p99_us']:6.1f}us")
        if "int8" in r:
            line += (f" | int8 p50={r['int8']['p50_us']:6.1f}us "
                     f"drift={r['int8']['drift_vs_fp32']['max_abs']:.4f}")
        print(line)
    if out:
        Path(out).write_text(json.dumps(rows, indent=1))
