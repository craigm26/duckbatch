"""Reward terms duckbatch adds to Pollen's tasks. Plain functions in mjlab's reward signature.

WHY THESE EXIST (b003 close, notes/2026-09-25-b003-close.md). mjlab's tracking terms are
`exp(-error² / std²)`. Lenient, standing still at a slow command pays ~90%; tight, partial
progress pays next to nothing. Either way there is no gradient from standing toward walking
slowly, and b003's fine-tune followed the reward into standing more.

A PROGRESS TERM PAYS FOR GOING, LINEARLY, AND NOTHING FOR STANDING. Speed along the commanded
direction, clipped to [0, |command|] and divided by |command|: 0 for standing still, 0.5 for
half the commanded speed, 1 at the command, and no more for overshooting. Below `min_command`
the command counts as "stand" and the term pays 0, leaving standing to the terms that already
pay for it. Same body-frame velocities and the same command as mjlab's tracking terms, so the
two agree on what "the commanded velocity" is.

TRACKING RELATIVE TO THE COMMAND (b003c close, notes/2026-10-01-b003c-close.md). mjlab's linear
tracking kernel has one width for every command, so standing still at 0.10 m/s pays exp(−0.1) ≈
0.90 of it and walking at 0.05 m/s only 0.97. Going from standing to half the command earns
+0.07, and PPO has stayed standing through three fine-tunes. `track_linear_velocity_relative`
narrows the width in proportion to the command below `full_width_speed`, so the planar error is
judged relative to the command: standing at 0.10 m/s pays what standing at 0.4 m/s pays, and
half the command pays what half of 0.4 m/s pays. At and above `full_width_speed` it is mjlab's
term exactly; a planar command below `stand_below` is "stand" and also gets mjlab's width,
so standing envs are judged as before. The vertical part keeps mjlab's width at every command:
bobbing is a property of the gait, not of the command.

JUDGE THE STRIDE, NOT THE INSTANT (b003d close, notes/2026-10-01-b003d-close.md). Within a
stride the body sways 0.10-0.15 m/s, more than a slow command, and an instantaneous error
counts the sway: under mjlab's width a walk at exactly 0.10 m/s earns what standing earns, and
under b003d's relative width stepping in place pays less than standing. On the planar velocity
averaged over half a second the sway falls to 0.03-0.05 m/s. `track_linear_velocity_stride`
is b003d's term on that average; it keeps a short history per env, refilled when an episode
starts so nothing carries over from the last one.
"""

from __future__ import annotations

import torch


def _robot_and_command(env, command_name: str, asset_name: str):
    asset = env.scene[asset_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    return asset, command


def command_progress_linear(env, command_name: str = "twist", min_command: float = 0.02,
                            asset_name: str = "robot") -> torch.Tensor:
    """Fraction of the commanded planar speed achieved along the commanded direction, in [0, 1]."""
    asset, command = _robot_and_command(env, command_name, asset_name)
    cmd = command[:, :2]
    speed = torch.linalg.norm(cmd, dim=1)
    along = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * cmd, dim=1) / speed.clamp_min(1e-6)
    fraction = torch.minimum(along.clamp_min(0.0), speed) / speed.clamp_min(1e-6)
    return torch.where(speed > min_command, fraction, torch.zeros_like(fraction))


def command_progress_angular(env, command_name: str = "twist", min_command: float = 0.1,
                             asset_name: str = "robot") -> torch.Tensor:
    """Fraction of the commanded yaw rate achieved in the commanded sense, in [0, 1]."""
    asset, command = _robot_and_command(env, command_name, asset_name)
    want = command[:, 2]
    rate = want.abs()
    along = asset.data.root_link_ang_vel_b[:, 2] * torch.sign(want)
    fraction = torch.minimum(along.clamp_min(0.0), rate) / rate.clamp_min(1e-6)
    return torch.where(rate > min_command, fraction, torch.zeros_like(fraction))


def track_linear_velocity_relative(env, std: float, command_name: str = "twist",
                                   full_width_speed: float = 0.4, min_width_speed: float = 0.05,
                                   stand_below: float = 0.02, asset_name: str = "robot") -> torch.Tensor:
    """mjlab's `track_linear_velocity` with the planar width scaled by the command, in [0, 1].

    Width `std · clamp(|cmd| / full_width_speed, min_width_speed / full_width_speed, 1)` for
    commands from `stand_below` up, mjlab's `std` below it. With full_width_speed 0.4 and mjlab's
    std √0.1, standing still pays exp(−1.6) ≈ 0.20 at any command from 0.05 to 0.4 m/s.
    """
    asset, command = _robot_and_command(env, command_name, asset_name)
    actual = asset.data.root_link_lin_vel_b
    return _relative_tracking(command[:, :2], actual[:, :2], actual[:, 2], std, full_width_speed,
                              min_width_speed, stand_below)


def _relative_tracking(cmd, planar, vertical, std, full_width_speed, min_width_speed, stand_below):
    speed = torch.linalg.norm(cmd, dim=1)
    scale = (speed / full_width_speed).clamp(min_width_speed / full_width_speed, 1.0)
    scale = torch.where(speed < stand_below, torch.ones_like(scale), scale)
    xy_error = torch.sum(torch.square(cmd - planar), dim=1)
    return torch.exp(-xy_error / (std * scale) ** 2 - torch.square(vertical) / std**2)


class track_linear_velocity_stride:
    """`track_linear_velocity_relative` on the planar velocity averaged over `window_s`, in [0, 1].

    A class term (mjlab builds it with `(cfg, env)` and calls `reset` when episodes end): it keeps
    the last `window_s` of planar body velocity per env. An env's history is refilled with its
    current velocity on the first call of a new episode, and a second call in the same env step
    does not push twice. The vertical part stays instantaneous, at mjlab's width.
    """

    def __init__(self, cfg, env):
        window_s = float(cfg.params.get("window_s", 0.5))
        self.window = max(1, round(window_s / env.step_dt))
        self.history = torch.zeros(env.num_envs, self.window, 2, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.slot = 0
        self.last_step = None

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.fresh[:] = True
        else:
            self.fresh[env_ids] = True

    def __call__(self, env, std: float, command_name: str = "twist", full_width_speed: float = 0.4,
                 min_width_speed: float = 0.05, stand_below: float = 0.02, window_s: float = 0.5,
                 asset_name: str = "robot") -> torch.Tensor:
        asset, command = _robot_and_command(env, command_name, asset_name)
        actual = asset.data.root_link_lin_vel_b
        planar = actual[:, :2]
        if env.common_step_counter != self.last_step:   # once per env step, every env
            self.history[:, self.slot] = planar
            self.slot = (self.slot + 1) % self.window
            self.last_step = env.common_step_counter
        # Then new episodes start clean (torch.where: no host sync on the GPU).
        self.history = torch.where(self.fresh[:, None, None], planar[:, None, :], self.history)
        self.fresh.zero_()
        return _relative_tracking(command[:, :2], self.history.mean(dim=1), actual[:, 2], std,
                                  full_width_speed, min_width_speed, stand_below)


def ball_lateral_speed(env, asset_name: str = "ball") -> torch.Tensor:
    """How fast the ball is going ACROSS the kick line, |v · perp(kick direction)|, in m/s.

    k001 (notes/2026-10-02-k001-design.md). Pollen's kick pays for speed along the line the env
    froze at reset (`ball_forward_velocity`) and says nothing about speed across it; Pollen's right
    kick goes 7.3° off line on average. Weighted like the forward term, this charges a crooked kick
    in proportion to how crooked it is, every step the ball keeps rolling — and slowing the kick
    down does not escape it, because the sideways speed falls with the forward speed.
    """
    from mjlab_microduck.tasks import mdp as microduck_mdp

    ball = env.scene[asset_name]
    d = microduck_mdp._ball_kick_dir(env)
    v = ball.data.root_link_lin_vel_w[:, :2]
    return (v[:, 0] * -d[:, 1] + v[:, 1] * d[:, 0]).abs()
