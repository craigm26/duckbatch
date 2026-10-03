"""Paired kick rollouts for a person to choose between: per-skill RLHF, starting with the kicks.

    duckbatch kick-pairs right --out records/kp001-right
    duckbatch kick-pairs left  --out records/kp001-left

WHY A KICK NEEDS ITS OWN PAIRS. p001's pairs are walkers under a twist command, and their seven
features (tracking error, falls, smoothness) say nothing about a kick: a kick has no command, and
what a person watching one judges is where the ball went and how the duck looked doing it. So each
kick is rolled out in its own task (Pollen's BallKick, or duckbatch's left-foot mirror), from one
seed for every policy, so env i has the same ball, mass and pushes on both sides of a pair.

THE FEATURES ARE MEASURED IN THE SIMULATOR, NOT ON THE PHONE. A phone clip carries no ball, so
the phone cannot measure a kick. These numbers travel in the exported pack beside the frames and
go into each pick record, so the phone fits a taste over the same values the sim measured. Each is
something less of is better by Pollen's reward; a person who prefers more of one is told so
rather than trained toward it (the app's `PreferenceModel.rewardPlan`).

    off_angle    radians off the line the env froze at reset, |atan2(side, travel)|
    speed_err    |peak forward ball speed - BALL_TARGET_SPEED| (Pollen's target, m/s)
    fell         1 if the duck went down before its first episode ended
    foot_lift    share of steps the support foot was NOT down (Pollen's support term, inverted)
    action_rate  mean |a_t - a_t-1| of the network's own output
    jitter       mean |a_t - 2a_t-1 + a_t-2|
    wobble       RMS trunk roll/pitch rate

Each env is read only until its first episode ends, as `skill_eval.kick` reads it: mjlab resets an
ended env inside `step`, and reading past that would credit a fall with the next ball.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

# The 14 actuated joints in the policy's order. The backlash twin adds a passive joint per servo
# (28 in `joint_pos`), so frames select these by name rather than taking every column.
ACTUATED = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
            "neck_pitch", "head_pitch", "head_yaw", "head_roll",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")

FEATURES = ("off_angle", "speed_err", "fell", "foot_lift", "action_rate", "jitter", "wobble")

# Which policies a person compares, per foot. Pollen's own kick beside the fine-tunes that passed
# or nearly passed their lines; one seed, so the only difference across a pair is the network.
SETS = {
    "right": {
        "tasks": {"plain": "Mjlab-BallKick-Flat-MicroDuck",
                  "backlash": "Mjlab-BallKick-Flat-Backlash-MicroDuck"},
        "slot": "kick_right",
        "policies": {
            "pollen_kick_right": "teachers/ball_kick_right.onnx",
            "k001": "records/k001-kick-straight/policies/finetuned/policy.onnx",
        },
    },
    "left": {
        "tasks": {"plain": "Duckbatch-BallKick-Left-Flat-MicroDuck",
                  "backlash": "Duckbatch-BallKick-Left-Flat-Backlash-MicroDuck"},
        "slot": "kick_left",
        "policies": {
            "pollen_kick_left": "teachers/ball_kick_left.onnx",
            "k002": "records/k002-kick-left-straight/policies/finetuned/policy.onnx",
            "k002b": "records/k002b-kick-left-straighter/policies/finetuned/policy.onnx",
        },
    },
}


def rollout(env, policy, seed: int, keep: int):
    """One policy on every env from `seed`: per-env features, and frames + ball for `keep` envs."""
    from mjlab_microduck.tasks import mdp as microduck_mdp
    from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import BALL_TARGET_SPEED

    from .skill_eval import robot_down

    n = env.num_envs
    obs = env.reset(seed=seed)[0]["actor"]
    dev = obs.device
    robot, ball = env.scene["robot"], env.scene["ball"]
    start = ball.data.root_link_pos_w[:, :2].clone()
    direction = microduck_mdp._ball_kick_dir(env).clone()
    steps = int(round(env.cfg.episode_length_s / env.step_dt))
    z = lambda: torch.zeros(n, device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    travel, side, peak, fell, lifted, counted = z(), z(), z(), z(), z(), z()
    arate, jit, wob, read = z(), z(), z(), z()
    a1 = torch.zeros(n, 14, device=dev)
    a2 = torch.zeros(n, 14, device=dev)
    frames, balls = [], []
    cols = torch.tensor([list(robot.joint_names).index(j) for j in ACTUATED], device=dev)
    # Frames kept per env stop at the first end: the frame after it is already the next episode.
    length = torch.full((n,), steps, dtype=torch.long, device=dev)
    for t in range(steps):
        # Recorded BEFORE the step, so a frame is never a freshly-reset env.
        if keep:
            frames.append(torch.cat([robot.data.root_link_pos_w[:keep], robot.data.root_link_quat_w[:keep],
                                     robot.data.joint_pos[:keep][:, cols]], dim=1).cpu())
            balls.append(ball.data.root_link_pos_w[:keep].cpu())
        with torch.no_grad():
            act = policy(obs)
        obs_d, _, terminated, truncated, _ = env.step(act)
        obs = obs_d["actor"]
        ended = terminated | truncated
        reading = alive & ~ended
        live = reading.float()
        pos = ball.data.root_link_pos_w[:, :2]
        d = pos - start
        travel = torch.where(reading, (d * direction).sum(dim=1), travel)
        side = torch.where(reading, (d[:, 0] * -direction[:, 1] + d[:, 1] * direction[:, 0]).abs(), side)
        fwd_vel = (ball.data.root_link_lin_vel_w[:, :2] * direction).sum(dim=1)
        peak = torch.where(reading, torch.maximum(peak, fwd_vel), peak)
        foot = microduck_mdp.single_foot_grounded_reward(env, sensor_name="support_foot_ground_contact")
        lifted += (1 - foot) * live
        counted += live
        fell = torch.maximum(fell, ((robot_down(env) & reading) | (alive & terminated & ~truncated)).float())
        if t >= 1:
            arate += (act - a1).abs().mean(dim=1) * live
        if t >= 2:
            jit += (act - 2 * a1 + a2).abs().mean(dim=1) * live
        w = robot.data.root_link_ang_vel_b
        wob += (w[:, 0] ** 2 + w[:, 1] ** 2) * live
        read += live
        a2, a1 = a1, act
        length = torch.where(alive & ended, torch.full_like(length, t + 1), length)
        alive = alive & ~ended
    r = read.clamp(min=1)
    feats = torch.stack([
        torch.atan2(side, travel.clamp(min=1e-3)).abs(),
        (peak - BALL_TARGET_SPEED).abs(),
        fell,
        lifted / counted.clamp(min=1),
        arate / r, jit / r, (wob / r).sqrt(),
    ], dim=1).cpu().numpy()
    traj = torch.stack(frames).numpy() if frames else None
    ball_traj = torch.stack(balls).numpy() if balls else None
    return feats, traj, ball_traj, direction[:keep].cpu().numpy(), length[:keep].cpu().numpy()


def generate(foot: str, out: str | Path, num_envs: int = 32, keep: int = 8, seed: int = 4001,
             device: str = "cpu", log=print) -> Path:
    from . import tasks as duckbatch_tasks
    from .policy import load_teacher
    from .sim import make_env

    duckbatch_tasks.register()
    spec = SETS[foot]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    feats = np.zeros((len(spec["policies"]), len(spec["tasks"]), num_envs, len(FEATURES)), np.float32)
    trajs, balls, dirs, lengths = {}, {}, {}, {}
    for ci, (cname, task) in enumerate(spec["tasks"].items()):
        env = make_env(task, num_envs, device=device, seed=seed)
        for pi, (pname, path) in enumerate(spec["policies"].items()):
            net = load_teacher(path, device)
            f, tr, bl, dr, ln = rollout(env, net, seed, keep)
            feats[pi, ci] = f
            trajs[f"{pname}__{cname}"] = tr
            balls[f"{pname}__{cname}"] = bl
            dirs[f"{pname}__{cname}"] = dr
            lengths[f"{pname}__{cname}"] = ln
            m = f.mean(axis=0)
            log(f"[kick-pairs] {pname:>18} {cname:>8}: " + " ".join(
                f"{k}={v:.3f}" for k, v in zip(FEATURES, m)), flush=True)
        env.close()
    np.savez_compressed(out / "features.npz", features=feats)
    np.savez_compressed(out / "trajectories.npz", **trajs)
    np.savez_compressed(out / "balls.npz", **balls)
    np.savez_compressed(out / "directions.npz", **dirs)
    np.savez_compressed(out / "lengths.npz", **lengths)
    (out / "pairs.json").write_text(json.dumps({
        "format": "duckbatch.kick-pairs.v1", "skill": spec["slot"], "foot": foot, "seed": seed,
        "num_envs": num_envs, "kept_envs": keep, "features": list(FEATURES),
        "policies": spec["policies"], "conditions": spec["tasks"],
        "trajectories": {"columns": "root pos (3), root quat wxyz (4), joint pos (14)",
                         "ball": "ball pos (3)", "recorded": "before each step"},
    }, indent=1))
    log(f"[kick-pairs] wrote {out}", flush=True)
    return out
