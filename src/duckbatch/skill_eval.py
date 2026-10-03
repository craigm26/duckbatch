"""Judging a skill that is not walking: what it did, measured in the task it was trained in.

    duckbatch kick-eval <task> name=policy.onnx [name=policy.onnx ...] [--envs 256] [--seed 0]

THE SPEED CURVE IS A WALKING INSTRUMENT. `pairs.speed_curve` and the walk + recovery evaluation
answer "does it go where it is told"; a kick has no command and no gait. So each skill gets the
numbers its own task is about, read from the env exactly as it trained — domain randomisation,
pushes and noise on, curricula at their final stage (`sim.make_env`) — on the same seed for every
policy, so the arms differ only in the network.

A KICK (Mjlab-BallKick-*, Pollen's `microduck_ball_kick_env_cfg`): the ball's travel along the
kick direction the env froze at reset, its peak forward speed, whether the robot fell, and the
share of steps the support foot is down (Pollen's own `single_foot_grounded_reward`, the term the
task trains on). Each env is read only until its episode first ends: after a termination the env
resets itself and puts a new ball down, and counting past that would credit a fall with a kick.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def kick(task: str, policies: dict[str, str], num_envs: int = 256, seed: int = 0,
         device: str = "cuda:0") -> dict[str, dict[str, float]]:
    from mjlab_microduck.tasks import mdp as microduck_mdp

    from .policy import load_teacher
    from .sim import make_env

    out: dict[str, dict[str, float]] = {}
    for name, onnx in policies.items():
        env = make_env(task, num_envs, device=device, seed=seed)
        net = load_teacher(onnx, device)
        obs = env.reset(seed=seed)[0]["actor"]
        ball = env.scene["ball"]
        start = ball.data.root_link_pos_w[:, :2].clone()
        direction = microduck_mdp._ball_kick_dir(env).clone()
        steps = int(round(env.cfg.episode_length_s / env.step_dt))
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        travel = torch.zeros(num_envs, device=device)
        lateral = torch.zeros(num_envs, device=device)
        peak = torch.zeros(num_envs, device=device)
        fell = torch.zeros(num_envs, device=device)
        grounded = torch.zeros(num_envs, device=device)
        counted = torch.zeros(num_envs, device=device)
        for _ in range(steps):
            with torch.no_grad():
                obs_d, _, terminated, truncated, _ = env.step(net(obs))
            obs = obs_d["actor"]
            # THE STEP AN EPISODE ENDS ON IS ALREADY RESET. mjlab resets a terminated or timed-out
            # env inside `step`, so what is read after it is a NEW ball at its start — the first
            # draft read it and reported a kick that sent the ball backwards. Only envs still in
            # their first episode after this step are read.
            ended = terminated | truncated
            reading = alive & ~ended
            live = reading.float()
            pos = ball.data.root_link_pos_w[:, :2]
            fwd_pos = ((pos - start) * direction).sum(dim=1)
            fwd_vel = (ball.data.root_link_lin_vel_w[:, :2] * direction).sum(dim=1)
            travel = torch.where(reading, fwd_pos, travel)
            # How far off the line it went: the component across the kick direction.
            side = (pos - start)[:, 0] * -direction[:, 1] + (pos - start)[:, 1] * direction[:, 0]
            lateral = torch.where(reading, side.abs(), lateral)
            peak = torch.where(reading, torch.maximum(peak, fwd_vel), peak)
            foot = microduck_mdp.single_foot_grounded_reward(env, sensor_name="support_foot_ground_contact")
            grounded += foot * live
            counted += live
            down = robot_down(env)
            # A fall TERMINATES a kick episode, so the fall shows as `terminated` (not time-out).
            fell = torch.maximum(fell, ((down & reading) | (alive & terminated & ~truncated)).float())
            alive = alive & ~ended
        env.close()
        n = float(num_envs)
        out[name] = {"ball_travel_m": float(travel.mean()), "ball_peak_mps": float(peak.mean()),
                     "ball_off_line_m": float(lateral.mean()),
                     "ball_angle_deg": float(torch.rad2deg(torch.atan2(lateral, travel.clamp(min=1e-3))).mean()),
                     "fell_share": float(fell.sum() / n),
                     "support_foot_down": float(grounded.sum() / counted.clamp(min=1).sum()),
                     "envs": num_envs, "seed": seed, "steps": steps}
        print(f"[kick] {name:12s} {task}: travel {out[name]['ball_travel_m']:.3f} m, "
              f"peak {out[name]['ball_peak_mps']:.2f} m/s, off line {out[name]['ball_off_line_m']:.2f} m "
              f"({out[name]['ball_angle_deg']:.1f} deg), fell {out[name]['fell_share']:.1%}, "
              f"support foot down {out[name]['support_foot_down']:.1%}", flush=True)
    return out


def robot_down(env) -> torch.Tensor:
    """Tilted past the fall gate pairs.py uses: projected gravity z above -0.5."""
    return env.scene["robot"].data.projected_gravity_b[:, 2] > -0.5


def write(path: Path, result: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return path


def profile(task: str, policy: str, num_envs: int = 64, seed: int = 0,
            device: str = "cuda:0") -> dict[str, Any]:
    """Any skill, in its own task: what ended each episode and what each of Pollen's terms paid.

    A FIRST LOOK, NOT A PASS LINE. The kick's weakness (aim) was not the one guessed (backlash);
    this is the cheap instrument for finding the next one before anybody pays for a GPU. For every
    env, until its first episode ends:
      - which termination ended it (fell_over, time_out, …), or that it ran to the end;
      - each reward term's weighted sum, divided by the time the env was read, so terms compare
        as reward per second the way mjlab logs `Episode_Reward/<term>`.
    The sums are read BEFORE each step, because mjlab zeroes them inside `step` for an env that
    ends (`RewardManager.reset`); the last step of each episode is the one value lost.
    """
    from .policy import load_teacher
    from .sim import make_env

    env = make_env(task, num_envs, device=device, seed=seed)
    net = load_teacher(policy, device)
    obs = env.reset(seed=seed)[0]["actor"]
    rm, tm = env.reward_manager, env.termination_manager
    names = list(rm._episode_sums.keys())
    final = {k: torch.zeros(num_envs, device=device) for k in names}
    seconds = torch.zeros(num_envs, device=device)
    ended_by: dict[str, torch.Tensor] = {t: torch.zeros(num_envs, dtype=torch.bool, device=device)
                                          for t in tm.active_terms}
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    steps = int(round(env.cfg.episode_length_s / env.step_dt))
    for t in range(steps):
        before = {k: v.clone() for k, v in rm._episode_sums.items()}
        with torch.no_grad():
            obs_d, _, terminated, truncated, _ = env.step(net(obs))
        obs = obs_d["actor"]
        ended = (terminated | truncated) & alive
        still = alive & ~ended
        for k in names:
            final[k] = torch.where(still, rm._episode_sums[k], torch.where(ended, before[k], final[k]))
        seconds = torch.where(alive, torch.full_like(seconds, (t + 1) * env.step_dt), seconds)
        for term in ended_by:
            ended_by[term] |= ended & tm._term_dones[term]
        alive = still
    env.close()
    per_second = {k: float((final[k] / seconds.clamp(min=env.step_dt)).mean()) for k in names}
    reasons = {t: float(v.float().mean()) for t, v in ended_by.items()}
    reasons["ran_to_end"] = float(alive.float().mean())
    return {"task": task, "policy": Path(policy).name, "envs": num_envs, "seed": seed,
            "episode_s": float(env.cfg.episode_length_s), "ended_by": reasons,
            "reward_per_s": per_second}
