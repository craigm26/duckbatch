"""Paired rollouts: several policies, the same conditions, features a preference can be about.

    duckbatch pairs --out records/p001

WHY PAIRED. A person choosing between two ducks is comparing the POLICIES, so everything else
has to be the same: the reset seed (so env i has the same mass, friction and push schedule for
every policy, up to MuJoCo-Warp's small non-determinism), the command, the walk profile. Each
policy therefore runs on the whole env batch in turn, from the same seed, rather than on a slice
of it as the population does.

WHY THESE FEATURES. They are what the sim can measure that a person watching might care about:
does it do what it was told (tracking), does it fall, and does it look smooth or twitchy (action
rate, jitter, wobble). Whether "looks natural" lives in them at all is exactly what real
preferences will reveal; simulated ones cannot.
"""

from __future__ import annotations

import copy
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

FEATURES = ("lin_err", "ang_err", "fell", "down_frac", "action_rate", "jitter", "wobble")
COMMANDS = {  # name -> (vx, vy, wz); within Pollen's limits
    "fwd_015": (0.15, 0.0, 0.0),
    "fwd_025": (0.25, 0.0, 0.0),
    "turn_06": (0.0, 0.0, 0.6),
    "arc": (0.15, 0.0, 0.4),
    "stand": (0.0, 0.0, 0.0),
}
DEFAULT_POLICIES = {
    "velstand": "teachers/velstand.onnx",
    "alpha_walking": "teachers/alpha_walking.onnx",
    "student_256x128x64": "records/b002-student-size-longer/policies/a01/policy.onnx",
    "student_128x128": "records/b002-student-size-longer/policies/a02/policy.onnx",
}


@contextmanager
def fixed_command(env, twist: tuple[float, float, float]):
    """Every env gets exactly `twist`, head and body poses zero, for as long as this is open.

    Pins the live command term cfgs AND the curricula that would re-widen them at the next reset
    (`standing_envs`, `head_pose_range`, `body_pose_range`), then restores all of it.
    """
    cm, cur = env.command_manager, env.curriculum_manager
    saved = []

    def pin(obj, **values):
        saved.append((obj, {k: copy.deepcopy(getattr(obj, k)) for k in values}))
        for k, v in values.items():
            setattr(obj, k, v)

    tw = cm.get_term("twist").cfg
    vx, vy, wz = twist
    pin(tw.ranges, lin_vel_x=(vx, vx), lin_vel_y=(vy, vy), ang_vel_z=(wz, wz))
    pin(tw, rel_standing_envs=0.0, rel_forward_envs=0.0, rel_heading_envs=0.0,
        heading_command=False, resampling_time_range=(1e9, 1e9))
    for name, dims in (("head_pose", 4), ("body_pose", 6)):
        if name in cm.active_terms:
            c = cm.get_term(name).cfg
            pin(c, ranges=tuple((0.0, 0.0) for _ in range(dims)), resampling_time_range=(1e9, 1e9))
    saved_params = []
    for term, key, value in (("standing_envs", "standing_stages",
                              [{"step": 0, "rel_standing_envs": 0.0}]),
                             ("head_pose_range", "range_stages",
                              [{"step": 0, "ranges": tuple((0.0, 0.0) for _ in range(4))}]),
                             ("body_pose_range", "range_stages",
                              [{"step": 0, "ranges": tuple((0.0, 0.0) for _ in range(6))}])):
        if term in cur.active_terms:
            params = cur.get_term_cfg(term).params
            saved_params.append((params, key, params[key]))
            params[key] = value
    try:
        yield
    finally:
        for params, key, old in saved_params:
            params[key] = old
        for obj, old in reversed(saved):
            for k, v in old.items():
                setattr(obj, k, v)


def rollout(env, policy, seconds: float, seed: int, keep_traj: int = 4):
    """One policy on every env from `seed`. Returns per-env features and a few trajectories."""
    dt = env.step_dt
    steps = int(math.ceil(seconds / dt))
    robot = env.scene["robot"]
    obs, _ = env.reset(seed=seed)
    obs = obs["actor"]
    n = env.num_envs
    dev = obs.device
    z = lambda: torch.zeros(n, device=dev)
    lin, ang, up_n, fell, down, arate, jit, wob = z(), z(), z(), z(), z(), z(), z(), z()
    a1 = torch.zeros(n, 14, device=dev)
    a2 = torch.zeros(n, 14, device=dev)
    traj = []
    cmd = env.command_manager.get_command("twist")
    for t in range(steps):
        with torch.no_grad():
            act = policy(obs)
        obs_d, _, terminated, truncated, _ = env.step(act)
        obs = obs_d["actor"]
        g = robot.data.projected_gravity_b[:, 2]
        is_down = g > -0.5
        up = (~is_down).float()
        v = robot.data.root_link_lin_vel_b
        w = robot.data.root_link_ang_vel_b
        lin += torch.linalg.norm(cmd[:, :2] - v[:, :2], dim=1) * up
        ang += (cmd[:, 2] - w[:, 2]).abs() * up
        up_n += up
        down += is_down.float()
        fell = torch.maximum(fell, is_down.float())
        if t >= 1:
            arate += (act - a1).abs().mean(dim=1)
        if t >= 2:
            jit += (act - 2 * a1 + a2).abs().mean(dim=1)
        wob += (w[:, 0] ** 2 + w[:, 1] ** 2)
        a2, a1 = a1, act
        if keep_traj:
            traj.append(torch.cat([robot.data.root_link_pos_w[:keep_traj],
                                   robot.data.root_link_quat_w[:keep_traj],
                                   robot.data.joint_pos[:keep_traj]], dim=1).cpu())
    feats = torch.stack([
        lin / up_n.clamp(min=1), ang / up_n.clamp(min=1), fell, down / steps,
        arate / max(steps - 1, 1), jit / max(steps - 2, 1), (wob / steps).sqrt(),
    ], dim=1).cpu().numpy()
    return feats, (torch.stack(traj).numpy() if traj else None)


def generate(out: str | Path, policies: dict[str, str] | None = None,
             commands: dict[str, tuple] | None = None, num_envs: int = 512, seconds: float = 8.0,
             seed: int = 3001, device: str = "cuda:0", log=print) -> Path:
    from .policy import load_teacher
    from .sim import DEFAULT_TASK, Population, make_env

    policies = policies or DEFAULT_POLICIES
    commands = commands or COMMANDS
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(DEFAULT_TASK, num_envs, device=device, seed=seed)
    # Borrow the population's walk profile (no prone spawns, no topples).
    prof = Population(env, next(iter(policies.values())), [], device=device)
    feats = np.zeros((len(policies), len(commands), num_envs, len(FEATURES)), np.float32)
    trajs = {}
    for pi, (pname, path) in enumerate(policies.items()):
        net = load_teacher(path, device)
        for ci, (cname, twist) in enumerate(commands.items()):
            with prof.profile("walk"), fixed_command(env, twist):
                f, tr = rollout(env, net, seconds, seed)
            feats[pi, ci] = f
            trajs[f"{pname}__{cname}"] = tr
            m = f.mean(axis=0)
            log(f"[pairs] {pname:>20} {cname:>8}: " + " ".join(
                f"{k}={v:.3f}" for k, v in zip(FEATURES, m)))
    np.savez_compressed(out / "features.npz", features=feats)
    np.savez_compressed(out / "trajectories.npz", **trajs)
    (out / "pairs.json").write_text(json.dumps({
        "format": "duckbatch.pairs.v1", "task": DEFAULT_TASK, "seed": seed, "seconds": seconds,
        "num_envs": num_envs, "profile": "walk", "features": list(FEATURES),
        "policies": policies, "commands": {k: list(v) for k, v in commands.items()},
        "trajectories": {"envs": 4, "columns": "root pos (3), root quat wxyz (4), joint pos"},
    }, indent=1))
    env.close()
    log(f"[pairs] wrote {out}")
    return out


def speed_curve(out: str | Path, policies: dict[str, str] | None = None,
                vx_levels=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4), wz_levels=(0.5, 1.0),
                num_envs: int = 512, seconds: float = 8.0, seed: int = 4001,
                device: str = "cuda:0", log=print) -> Path:
    """What each walker DOES for a range of commands: mean forward speed and yaw rate while
    upright, over the last half of the episode (after it has got going), plus the share of envs
    that fell. The dead band shows up as a command the walker answers by standing still."""
    from .policy import load_teacher
    from .sim import DEFAULT_TASK, Population, make_env

    policies = policies or {k: v for k, v in DEFAULT_POLICIES.items() if k != "alpha_walking"}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(DEFAULT_TASK, num_envs, device=device, seed=seed)
    prof = Population(env, next(iter(policies.values())), [], device=device)
    robot = env.scene["robot"]
    dt = env.step_dt
    steps = int(math.ceil(seconds / dt))
    commands = [(v, 0.0, 0.0) for v in vx_levels] + [(0.0, 0.0, w) for w in wz_levels]
    rows = []
    for pname, path in policies.items():
        net = load_teacher(path, device)
        for twist in commands:
            with prof.profile("walk"), fixed_command(env, twist):
                obs, _ = env.reset(seed=seed)
                obs = obs["actor"]
                vx_sum = wz_sum = n_up = 0.0
                fell = torch.zeros(num_envs, device=obs.device)
                for t in range(steps):
                    with torch.no_grad():
                        obs = env.step(net(obs))[0]["actor"]
                    down = robot.data.projected_gravity_b[:, 2] > -0.5
                    fell = torch.maximum(fell, down.float())
                    if t >= steps // 2:
                        up = (~down).float()
                        vx_sum += float((robot.data.root_link_lin_vel_b[:, 0] * up).sum())
                        wz_sum += float((robot.data.root_link_ang_vel_b[:, 2] * up).sum())
                        n_up += float(up.sum())
            row = {"policy": pname, "command": list(twist),
                   "vx": vx_sum / max(n_up, 1), "wz": wz_sum / max(n_up, 1),
                   "fell_share": float(fell.mean())}
            rows.append(row)
            log(f"[speed] {pname:>20} cmd vx {twist[0]:.2f} wz {twist[2]:.2f} -> "
                f"vx {row['vx']:+.3f} wz {row['wz']:+.3f} fell {row['fell_share']:.1%}")
    (out / "speed_curve.json").write_text(json.dumps(
        {"task": DEFAULT_TASK, "profile": "walk", "seconds": seconds, "seed": seed,
         "measured_over": "last half of each episode, upright envs", "rows": rows}, indent=1))
    env.close()
    return out
