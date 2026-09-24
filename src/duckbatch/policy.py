"""Policies as torch modules: load a Pollen-shaped ONNX MLP, build small students, export back.

Every Microduck policy in the wild (Pollen's `microduck-policies`, the simulator Space, duckbench,
duck-studio) is the same graph: `(obs - mean) / std` then an ELU MLP, `obs[1,61] -> actions[1,14]`.
This module reads that graph into torch so a teacher can label a whole batch of envs on the GPU,
and writes students in the same shape so everything downstream loads them unchanged.

Only torch is imported lazily: `onnx_mlp_weights` and `count_params` work on the core install.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
from onnx import numpy_helper

OBS_LEN = 61
ACTION_LEN = 14


@dataclass
class MlpWeights:
    """The parameters of a normalizer + MLP policy, in forward order (W is [out, in])."""

    mean: np.ndarray
    std: np.ndarray
    layers: list[tuple[np.ndarray, np.ndarray]]
    activation: str
    metadata: dict[str, str]

    @property
    def hidden(self) -> tuple[int, ...]:
        return tuple(w.shape[0] for w, _ in self.layers[:-1])

    @property
    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in self.layers)


def onnx_mlp_weights(path: str | Path) -> MlpWeights:
    """Read a `Sub -> Div -> (Gemm -> act)* -> Gemm` graph. Refuses anything else, with the reason."""
    model = onnx.load(str(path))
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    # Constants can also live in Constant nodes depending on the exporter.
    for node in model.graph.node:
        if node.op_type == "Constant":
            inits[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    ops = [n for n in model.graph.node if n.op_type != "Constant"]
    kinds = [n.op_type for n in ops]
    if kinds[:2] != ["Sub", "Div"] or kinds[-1] != "Gemm":
        raise ValueError(f"{Path(path).name}: not a normalizer+MLP graph: {kinds}")
    mean = inits[ops[0].input[1]].reshape(-1).astype(np.float32)
    std = inits[ops[1].input[1]].reshape(-1).astype(np.float32)
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    acts = set()
    for node in ops[2:]:
        if node.op_type == "Gemm":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            w = inits[node.input[1]].astype(np.float32)
            if not attrs.get("transB", 0):
                w = w.T
            layers.append((w, inits[node.input[2]].reshape(-1).astype(np.float32)))
        elif node.op_type in ("Elu", "Relu", "Tanh"):
            acts.add(node.op_type)
        else:
            raise ValueError(f"{Path(path).name}: unsupported op {node.op_type} in {kinds}")
    if len(acts) > 1:
        raise ValueError(f"{Path(path).name}: mixed activations {acts}")
    meta = {p.key: p.value for p in model.metadata_props}
    return MlpWeights(mean, std, layers, acts.pop() if acts else "Elu", meta)


def mlp_flops(hidden: Sequence[int], obs_len: int = OBS_LEN, action_len: int = ACTION_LEN) -> int:
    """Multiply-accumulates x2 for one forward pass (the Gemms; activations are noise)."""
    dims = [obs_len, *hidden, action_len]
    return sum(2 * a * b for a, b in zip(dims[:-1], dims[1:]))


def mlp_params(hidden: Sequence[int], obs_len: int = OBS_LEN, action_len: int = ACTION_LEN) -> int:
    dims = [obs_len, *hidden, action_len]
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))


# ---------------------------------------------------------------------------------------------
# torch side (sim extra).


def _torch():
    import torch

    return torch


def build_mlp(hidden: Sequence[int], activation: str = "Elu", obs_len: int = OBS_LEN,
              action_len: int = ACTION_LEN):
    """A normalizer + MLP with the exact parameter names mjlab exports (`obs_normalizer`, `mlp.N`)."""
    torch = _torch()
    nn = torch.nn
    act = {"Elu": nn.ELU, "Relu": nn.ReLU, "Tanh": nn.Tanh}[activation]

    class NormalizedMlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("obs_mean", torch.zeros(1, obs_len))
            self.register_buffer("obs_std", torch.ones(1, obs_len))
            dims = [obs_len, *hidden, action_len]
            mods: list[Any] = []
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
                mods.append(nn.Linear(a, b))
                if i < len(dims) - 2:
                    mods.append(act())
            self.mlp = nn.Sequential(*mods)
            self.hidden = tuple(hidden)
            self.activation = activation

        def forward(self, obs):
            return self.mlp((obs - self.obs_mean) / self.obs_std)

    return NormalizedMlp()


def load_teacher(path: str | Path, device: str = "cpu"):
    """The ONNX policy as a frozen torch module, numerically identical to onnxruntime (checked)."""
    torch = _torch()
    w = onnx_mlp_weights(path)
    net = build_mlp(w.hidden, w.activation)
    with torch.no_grad():
        net.obs_mean.copy_(torch.from_numpy(w.mean).view(1, -1))
        net.obs_std.copy_(torch.from_numpy(w.std).view(1, -1))
        linears = [m for m in net.mlp if isinstance(m, torch.nn.Linear)]
        for lin, (wt, b) in zip(linears, w.layers):
            lin.weight.copy_(torch.from_numpy(wt))
            lin.bias.copy_(torch.from_numpy(b))
    net.metadata = w.metadata
    return net.to(device).eval().requires_grad_(False)


def export_onnx(net, path: str | Path, metadata: dict[str, str] | None = None) -> Path:
    """Write `obs[1,61] -> actions[1,14]` with the normalizer folded into the graph, like mjlab does."""
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    was_training = net.training
    net = net.eval()
    dummy = torch.zeros(1, OBS_LEN, device=next(net.parameters()).device)
    torch.onnx.export(
        net, (dummy,), str(path), input_names=["obs"], output_names=["actions"],
        opset_version=17, dynamo=False,
    )
    model = onnx.load(str(path))
    for k, v in (metadata or {}).items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, str(v)
    onnx.save(model, str(path))
    net.train(was_training)
    return path


ALPHA_HIDDEN = (512, 256, 128)


def identity_bytes(path: str | Path) -> tuple[str, bytes]:
    """duckkit's `DuckPolicy.canonicalIdentityBytes`, in Python: (scheme, bytes).

    v1 for the alpha shape (mean, std, then each layer's weights and biases, little-endian
    float32); v2 for any other shape (`DPv2`, the layer count and each layer's widths as
    little-endian uint32, then the v1 bytes). Checked against duckkit's recorded official
    fingerprints in the tests, so the two implementations cannot drift apart silently.
    """
    import struct

    w = onnx_mlp_weights(path)
    body = [w.mean.astype("<f4").tobytes(), w.std.astype("<f4").tobytes()]
    for W, b in w.layers:
        body += [W.astype("<f4").tobytes(), b.astype("<f4").tobytes()]
    v1 = b"".join(body)
    if w.hidden == ALPHA_HIDDEN:
        return "canonical-parameter-bytes-v1", v1
    widths = [OBS_LEN, *w.hidden, ACTION_LEN]
    header = b"DPv2" + struct.pack("<I", len(widths) - 1)
    for a, o in zip(widths[:-1], widths[1:]):
        header += struct.pack("<II", a, o)
    return "canonical-parameter-bytes-v2", header + v1


def fingerprint(path: str | Path) -> str:
    """`sha256:<hex>` over the identity bytes: duckkit's `DuckPolicy.fingerprint`."""
    import hashlib

    return "sha256:" + hashlib.sha256(identity_bytes(path)[1]).hexdigest()
