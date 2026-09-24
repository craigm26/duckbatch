"""Pure-logic tests: no simulator, no GPU, no network (core install + onnx)."""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from duckbatch.batch import judge
from duckbatch.policy import mlp_flops, mlp_params, onnx_mlp_weights

T = {"falls_per_min": 0.2, "down_frac": 0.002, "lin_err": 0.10, "ang_err": 0.30,
     "teacher_mse": 0.0, "reward_per_s": 5.0}


def row(**kw):
    return {**T, "teacher_mse": 0.01, **kw}


def test_rules_keep_kill_uncertain():
    g = judge.DEFAULT_GATES
    assert judge.rule_verdict(row(), T, g)[0] == "keep"
    assert judge.rule_verdict(row(lin_err=0.13), T, g)[0] == "uncertain"
    assert judge.rule_verdict(row(lin_err=0.20), T, g)[0] == "kill"
    assert judge.rule_verdict(row(falls_per_min=5.0), T, g)[0] == "kill"
    assert judge.rule_verdict(row(down_frac=0.05), T, g)[0] == "uncertain"
    assert judge.rule_verdict(row(lin_err=float("nan")), T, g)[0] == "kill"


def test_efficiency_rank_prefers_smallest_passing():
    rows = {"teacher": T, "big": row(), "small": row(), "unsure": row(lin_err=0.13)}
    ds = judge.judge_rung(rows, "teacher", {}, 0, 2)
    assert {d.arm_id: d.verdict for d in ds} == {"big": "keep", "small": "keep",
                                                  "unsure": "pending"}
    params = {"big": 50_000, "small": 5_000, "unsure": 1_000}
    assert judge.survivors(ds, rows, "teacher", 2, "efficiency", params) == ["small", "big"]
    assert judge.survivors(ds, rows, "teacher", 3, "efficiency", params)[-1] == "unsure"


class FakeModel:
    def __init__(self, choice, conf):
        self.choice, self.conf, self.log = choice, conf, []

    def ask(self, text):
        self.log.append(text)
        return {"decision": {"choice": self.choice, "confidence": self.conf},
                "gap": {"score": 1}, "still_learning": {"noul": 0.5}}


def test_shadow_everywhere_actor_only_on_uncertain():
    rows = {"teacher": T, "ok": row(), "unsure": row(lin_err=0.13)}
    models = {"jev": FakeModel("kill", 0.95), "decide": FakeModel("extend", 0.99)}
    ds = {d.arm_id: d for d in judge.judge_rung(rows, "teacher", {}, 0, 2, None, models, ["jev"])}
    assert ds["ok"].tier == "rule" and ds["ok"].shadow["decide"]["choice"] == "extend"
    assert ds["unsure"].tier == "jev" and ds["unsure"].verdict == "kill"
    assert len(models["decide"].log) == 2  # consulted on both, acted on neither
    assert "line" not in ds["ok"].case_text.split("Lab conventions")[0]


def test_low_confidence_goes_to_person():
    rows = {"teacher": T, "unsure": row(lin_err=0.13)}
    ds = judge.judge_rung(rows, "teacher", {}, 0, 2, None, {"jev": FakeModel("kill", 0.6)},
                          ["jev"])
    assert ds[0].verdict == "pending" and ds[0].tier == "person"


def _mlp_onnx(path, hidden=(8, 4)):
    rng = np.random.default_rng(0)
    dims = [61, *hidden, 14]
    inits = [numpy_helper.from_array(rng.normal(size=(1, 61)).astype(np.float32), "mean"),
             numpy_helper.from_array(np.full((1, 61), 2.0, np.float32), "std")]
    nodes = [helper.make_node("Sub", ["obs", "mean"], ["x0"]),
             helper.make_node("Div", ["x0", "std"], ["h0"])]
    cur = "h0"
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        inits += [numpy_helper.from_array(rng.normal(size=(b, a)).astype(np.float32), f"W{i}"),
                  numpy_helper.from_array(rng.normal(size=(b,)).astype(np.float32), f"b{i}")]
        out = "actions" if i == len(dims) - 2 else f"g{i}"
        nodes.append(helper.make_node("Gemm", [cur, f"W{i}", f"b{i}"], [out], transB=1))
        if out != "actions":
            nodes.append(helper.make_node("Elu", [out], [f"e{i}"]))
            cur = f"e{i}"
    graph = helper.make_graph(nodes, "p",
                              [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
                              [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
                              inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), path)


def test_onnx_mlp_weights_matches_ort(tmp_path):
    p = tmp_path / "p.onnx"
    _mlp_onnx(p)
    w = onnx_mlp_weights(p)
    assert w.hidden == (8, 4) and w.n_params == mlp_params((8, 4))
    x = np.random.default_rng(1).normal(size=(1, 61)).astype(np.float32)
    h = (x - w.mean) / w.std
    for i, (W, b) in enumerate(w.layers):
        h = h @ W.T + b
        if i < len(w.layers) - 1:
            h = np.where(h > 0, h, np.expm1(h))
    ref = ort.InferenceSession(str(p)).run(None, {"obs": x})[0]
    assert np.allclose(h, ref, atol=1e-5)


def test_teacher_size():
    assert mlp_params((512, 256, 128)) == 197_774
    assert mlp_flops((64, 64)) == 2 * (61 * 64 + 64 * 64 + 64 * 14)
