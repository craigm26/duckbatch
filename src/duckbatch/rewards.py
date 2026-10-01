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
