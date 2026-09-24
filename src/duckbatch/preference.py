"""Learning what people prefer from pairwise choices (Bradley–Terry), and sizing how many it takes.

    P(a ≻ b) = σ( w · (φ_a − φ_b) + β · [a was shown on the left] )

φ are the paired-rollout features (`pairs.py`), standardised; w says what the person cares about;
β is the side bias `duck-feedback/0` records `shown.order` for, because people favour a side and
a model that cannot see that learns it as taste.

SIMULATED RATERS ARE A SIZING TOOL, NOT A FINDING. With no human preferences yet, `simulate`
draws choices from a rater whose taste (w*) and side bias (β*) are known, writes them as real
`duck-feedback/0` records, and asks: after N choices, does the fitted model rank the policies the
way the rater would, and does it find the side bias? That says how many choices the app should
collect. It says nothing about what people actually like; that is what real records are for.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pairs import FEATURES

# The simulated rater's taste over STANDARDISED features: falls matter most, then smoothness,
# then tracking. Negative = less is better. Pre-registered in notes/2026-09-24-p001-design.md.
HIDDEN_TASTE = {"fell": -3.0, "down_frac": -1.0, "jitter": -1.0, "wobble": -1.0,
                "action_rate": -0.5, "lin_err": -0.8, "ang_err": -0.8}
HIDDEN_SIDE_BIAS = 0.3


@dataclass
class PairSet:
    features: np.ndarray  # [P, C, E, K] raw
    policies: list[str]
    commands: list[str]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def load(cls, root: str | Path) -> "PairSet":
        root = Path(root)
        meta = json.loads((root / "pairs.json").read_text())
        f = np.load(root / "features.npz")["features"].astype(np.float64)
        flat = f.reshape(-1, f.shape[-1])
        return cls(f, list(meta["policies"]), list(meta["commands"]),
                   flat.mean(axis=0), flat.std(axis=0) + 1e-9)

    def phi(self, p: int, c: int, e: int) -> np.ndarray:
        return (self.features[p, c, e] - self.mean) / self.std


def fit(diffs: np.ndarray, sides: np.ndarray, y: np.ndarray, l2: float = 1e-2,
        iters: int = 50) -> tuple[np.ndarray, float]:
    """Regularised logistic regression by Newton's method. y is 1, 0 or 0.5 (a tie)."""
    X = np.hstack([diffs, sides[:, None]])
    theta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ theta))
        g = X.T @ (p - y) + l2 * theta
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-8:
            break
    return theta[:-1], float(theta[-1])


def policy_utilities(ps: PairSet, w: np.ndarray) -> np.ndarray:
    """Mean utility per policy over every command and env: the ranking a model implies."""
    z = (ps.features - ps.mean) / ps.std
    return (z @ w).mean(axis=(1, 2))


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float(s / (n * (n - 1) / 2))


def draw_pairs(ps: PairSet, n: int, rng: np.random.Generator):
    P, C, E, _ = ps.features.shape
    out = []
    for _ in range(n):
        a, b = rng.choice(P, 2, replace=False)
        out.append((int(a), int(b), int(rng.integers(C)), int(rng.integers(E)),
                    bool(rng.integers(2))))
    return out


def rater_probability(ps: PairSet, pair, w_star: np.ndarray, beta_star: float,
                      temperature: float) -> float:
    a, b, c, e, a_left = pair
    u = (ps.phi(a, c, e) - ps.phi(b, c, e)) @ w_star / temperature
    return 1 / (1 + math.exp(-(u + beta_star * (1 if a_left else -1))))


def design(ps: PairSet, pairs):
    diffs = np.array([ps.phi(a, c, e) - ps.phi(b, c, e) for a, b, c, e, _ in pairs])
    sides = np.array([1.0 if left else -1.0 for *_, left in pairs])
    return diffs, sides


def sizing(ps: PairSet, sizes=(25, 50, 100, 200, 400, 800), temperatures=(0.5, 2.0),
           draws: int = 10, test_pairs: int = 4000, seed: int = 0) -> dict[str, Any]:
    """The learning curve under simulated raters: held-out accuracy (and its ceiling), Kendall τ
    of the policy ranking, cos(w, w*), and the recovered side bias, for each N and noise level."""
    w_star = np.array([HIDDEN_TASTE[k] for k in FEATURES])
    true_util = policy_utilities(ps, w_star)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"true_ranking": [ps.policies[i] for i in np.argsort(-true_util)],
                           "true_utility": dict(zip(ps.policies, map(float, true_util))),
                           "curves": {}}
    test = draw_pairs(ps, test_pairs, rng)
    dt, st = design(ps, test)
    for temp in temperatures:
        pt = np.array([rater_probability(ps, pr, w_star, HIDDEN_SIDE_BIAS, temp) for pr in test])
        ceiling = float(np.mean(np.maximum(pt, 1 - pt)))
        yt = (rng.random(len(pt)) < pt).astype(float)
        rows = []
        for n in sizes:
            acc, tau, cos, beta = [], [], [], []
            for _ in range(draws):
                train = draw_pairs(ps, n, rng)
                p = np.array([rater_probability(ps, pr, w_star, HIDDEN_SIDE_BIAS, temp) for pr in train])
                y = (rng.random(n) < p).astype(float)
                d, s = design(ps, train)
                w, b = fit(d, s, y)
                pred = 1 / (1 + np.exp(-(dt @ w + b * st)))
                acc.append(float(np.mean((pred > 0.5) == (yt > 0.5))))
                tau.append(kendall_tau(policy_utilities(ps, w), true_util))
                cos.append(float(w @ w_star / (np.linalg.norm(w) * np.linalg.norm(w_star) + 1e-12)))
                beta.append(b)
            rows.append({"n": n, "accuracy": round(float(np.mean(acc)), 4),
                         "kendall_tau": round(float(np.mean(tau)), 4),
                         "tau_min": round(float(np.min(tau)), 4),
                         "cos_w": round(float(np.mean(cos)), 4),
                         "beta": round(float(np.mean(beta)), 4),
                         "beta_sd": round(float(np.std(beta)), 4)})
        out["curves"][f"temperature_{temp}"] = {"ceiling": round(ceiling, 4), "rows": rows}
    return out


def simulated_records(ps: PairSet, n: int, temperature: float, seed: int = 0,
                      fingerprints: dict[str, str] | None = None) -> list[dict]:
    """Simulated choices as real `duck-feedback/0` records, so the reader-to-fit path is exercised.
    Marked `client: duckbatch simulated rater` so nothing downstream can mistake them for people."""
    import uuid

    rng = np.random.default_rng(seed)
    w_star = np.array([HIDDEN_TASTE[k] for k in FEATURES])
    fps = fingerprints or {p: f"sha256:{p}" for p in ps.policies}
    recs = []
    for pr in draw_pairs(ps, n, rng):
        a, b, c, e, a_left = pr
        p = rater_probability(ps, pr, w_star, HIDDEN_SIDE_BIAS, temperature)
        choice = "a" if rng.random() < p else "b"
        recs.append({
            "format": "duck-feedback/0", "id": str(uuid.UUID(int=int(rng.integers(2**63)) << 64)),
            "created": "2026-09-24T00:00:00Z", "kind": "policy_preference",
            "consent": {"opt_in": True, "share": "research"},
            "source": {"client": "duckbatch simulated rater"},
            "policy_preference": {
                "a": {"repo": ps.policies[a], "fingerprint": fps[ps.policies[a]]},
                "b": {"repo": ps.policies[b], "fingerprint": fps[ps.policies[b]]},
                "shown": {"where": "sim", "order": "a_left" if a_left else "b_left",
                          "pair_id": f"p001/{ps.commands[c]}/{e}"},
                "choice": choice, "reasons": [],
            },
        })
    return recs


def fit_records(ps: PairSet, records: list[dict]) -> dict[str, Any]:
    """Fit on `policy_preference` records whose `pair_id` points into this pair set."""
    idx = {p: i for i, p in enumerate(ps.policies)}
    cidx = {c: i for i, c in enumerate(ps.commands)}
    pairs, y = [], []
    for r in records:
        if r["kind"] != "policy_preference":
            continue
        b = r["policy_preference"]
        if b["choice"] == "both_bad":
            continue
        _, cmd, env = b["shown"]["pair_id"].split("/")
        pairs.append((idx[b["a"]["repo"]], idx[b["b"]["repo"]], cidx[cmd], int(env),
                      b["shown"]["order"] == "a_left"))
        y.append({"a": 1.0, "b": 0.0, "tie": 0.5}[b["choice"]])
    d, s = design(ps, pairs)
    w, beta = fit(d, s, np.array(y))
    util = policy_utilities(ps, w)
    return {"n": len(y), "weights": dict(zip(FEATURES, map(float, w))), "side_bias": beta,
            "ranking": [ps.policies[i] for i in np.argsort(-util)]}
