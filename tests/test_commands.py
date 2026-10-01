"""b003c's slow-walk commands stay in the band, in every direction, with no yaw."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab_microduck")

from duckbatch.commands import DeadBandCommandCfg, slow_walk_commands


def test_slow_walk_speeds_stay_in_the_band():
    torch.manual_seed(0)
    c = slow_walk_commands(20000, 0.05, 0.25)
    speed = torch.linalg.norm(c, dim=1)
    assert speed.min() >= 0.05 - 1e-6 and speed.max() <= 0.25 + 1e-6


def test_slow_walk_goes_every_way():
    torch.manual_seed(1)
    c = slow_walk_commands(20000, 0.05, 0.25)
    angle = torch.atan2(c[:, 1], c[:, 0])
    counts = torch.histc(angle, bins=8, min=-math.pi, max=math.pi)
    assert counts.min() > 0.8 * counts.mean()      # roughly uniform on the circle


def test_the_upstream_command_is_carried_over_whole():
    from mjlab.tasks.registry import load_env_cfg
    import mjlab.tasks  # noqa: F401
    up = load_env_cfg("Mjlab-VelStand-Flat-MicroDuck").commands["twist"]
    db = DeadBandCommandCfg.from_upstream(up, rel_slow_envs=0.35, slow_speed_range=(0.05, 0.25))
    assert db.ranges == up.ranges and db.rel_turn_in_place_envs == up.rel_turn_in_place_envs
    assert db.rel_standing_envs == up.rel_standing_envs and db.rel_slow_envs == 0.35
