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
    cmd = command[:, :2]
    actual = asset.data.root_link_lin_vel_b
    speed = torch.linalg.norm(cmd, dim=1)
    scale = (speed / full_width_speed).clamp(min_width_speed / full_width_speed, 1.0)
    scale = torch.where(speed < stand_below, torch.ones_like(scale), scale)
    xy_error = torch.sum(torch.square(cmd - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    return torch.exp(-xy_error / (std * scale) ** 2 - z_error / std**2)
