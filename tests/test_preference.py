"""The preference model on a synthetic pair set whose answer is known by construction."""

import json

import numpy as np

from duckbatch import feedback, preference
from duckbatch.pairs import FEATURES


def synthetic(tmp_path, P=4, C=3, E=64, seed=0):
    rng = np.random.default_rng(seed)
    # Policy p is worse than p-1 on every feature by a margin, plus per-env noise.
    base = np.arange(P)[:, None, None, None] * 0.5
    f = base + rng.normal(0, 0.3, size=(P, C, E, len(FEATURES)))
    np.savez_compressed(tmp_path / "features.npz", features=f.astype(np.float32))
    (tmp_path / "pairs.json").write_text(json.dumps({
        "policies": {f"p{i}": "" for i in range(P)}, "commands": {f"c{i}": [0, 0, 0] for i in range(C)},
        "features": list(FEATURES)}))
    return preference.PairSet.load(tmp_path)


def test_bradley_terry_recovers_a_known_weight_and_side_bias():
    rng = np.random.default_rng(1)
    w_true, beta_true = np.array([1.5, -1.0, 0.5]), 0.4
    X = rng.normal(size=(20000, 3))
    s = rng.choice([-1.0, 1.0], size=20000)
    p = 1 / (1 + np.exp(-(X @ w_true + beta_true * s)))
    y = (rng.random(20000) < p).astype(float)
    w, beta = preference.fit(X, s, y, l2=1e-4)
    assert np.allclose(w, w_true, atol=0.08) and abs(beta - beta_true) < 0.06


def test_simulated_raters_rank_policies_from_best_to_worst(tmp_path):
    ps = synthetic(tmp_path)
    res = preference.sizing(ps, sizes=(200,), temperatures=(1.0,), draws=3, test_pairs=500)
    assert res["true_ranking"] == ["p0", "p1", "p2", "p3"]
    assert res["curves"]["temperature_1.0"]["rows"][0]["kendall_tau"] > 0.9


def test_simulated_choices_are_valid_feedback_records_and_fit_back(tmp_path):
    ps = synthetic(tmp_path)
    recs = [feedback.validate(r) for r in preference.simulated_records(ps, 400, 1.0)]
    assert all(r["source"]["client"] == "duckbatch simulated rater" for r in recs)
    fit = preference.fit_records(ps, recs)
    assert fit["n"] == 400 and fit["ranking"][0] == "p0" and fit["ranking"][-1] == "p3"


def test_the_python_fingerprint_is_duckkits():
    from duckbatch.policy import fingerprint
    # DuckOfficialPolicies (duckkit) records alpha_walking.onnx as da820b71…
    assert fingerprint("teachers/alpha_walking.onnx") == (
        "sha256:da820b718aa8bdb3317c018afba3ad3f461e0cf42256811c204dc005546ec4a3")
