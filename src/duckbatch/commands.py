"""Command samplers duckbatch puts in place of Pollen's, for one experiment at a time.

WHY (b003b close, notes/2026-10-01-b003b-close.md). With a reward that pays for going slowly,
PPO still did not find a slow gait: 0.008 m/s at a 0.10 m/s command, the student's own number.
VelStand samples planar commands uniformly in ±0.4 × ±0.3 m/s, so only about 15% of them fall
under 0.15 m/s. That is not much practice in the band nobody can walk in.

`DeadBandCommandCfg` keeps Pollen's sampler exactly — ranges, standing envs, its own
turn-in-place practice — and then sends a further share of the envs that are neither standing
nor turning to a planar speed drawn from `slow_speed_range` in a uniformly random direction,
with no yaw. It is the same move Pollen made for turning in place
(`VelocityCommandCommandOnly._resample_command`), applied to the dead band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly, VelocityCommandCommandOnlyCfg


def slow_walk_commands(n: int, lo: float, hi: float, device="cpu") -> torch.Tensor:
    """n planar commands, speed uniform in [lo, hi], direction uniform on the circle."""
    speed = torch.empty(n, device=device).uniform_(lo, hi)
    angle = torch.empty(n, device=device).uniform_(0.0, 2.0 * math.pi)
    return torch.stack((speed * torch.cos(angle), speed * torch.sin(angle)), dim=1)


class DeadBandCommand(VelocityCommandCommandOnly):
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        p = getattr(self.cfg, "rel_slow_envs", 0.0)
        if p <= 0.0 or len(env_ids) == 0:
            return
        # Neither standing nor turning in place: Pollen's two practices keep their envs.
        cmd = self.vel_command_b[env_ids]
        turning = (cmd[:, :2].abs().sum(dim=1) == 0) & (cmd[:, 2] != 0)
        free = env_ids[~self.is_standing_env[env_ids] & ~turning]
        if len(free) == 0:
            return
        pick = free[torch.empty(len(free), device=self.device).uniform_(0.0, 1.0) < p]
        if len(pick) == 0:
            return
        lo, hi = self.cfg.slow_speed_range
        self.vel_command_b[pick, :2] = slow_walk_commands(len(pick), lo, hi, self.device)
        self.vel_command_b[pick, 2] = 0.0
        self.vel_command_w[pick] = self.vel_command_b[pick]


@dataclass(kw_only=True)
class DeadBandCommandCfg(VelocityCommandCommandOnlyCfg):
    # Share of the envs that are neither standing nor turning in place that get a slow walk.
    rel_slow_envs: float = 0.0
    slow_speed_range: tuple[float, float] = (0.05, 0.25)

    def build(self, env) -> DeadBandCommand:
        return DeadBandCommand(self, env)

    @classmethod
    def from_upstream(cls, upstream: VelocityCommandCommandOnlyCfg, **extra) -> "DeadBandCommandCfg":
        """The same command, every upstream field carried over, plus the dead-band knobs."""
        return cls(**{f.name: getattr(upstream, f.name) for f in fields(upstream)}, **extra)
