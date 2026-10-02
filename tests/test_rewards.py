"""The b003b progress terms pay for going, linearly, and nothing for standing; b003d's tracking
term judges the planar error relative to the command."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from duckbatch.rewards import (command_progress_angular, command_progress_linear,
                               track_linear_velocity_relative)


def _env(cmd, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0)):
    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor([lin]),
                           root_link_ang_vel_b=torch.tensor([ang]))
    command = torch.tensor([cmd])
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                           command_manager=SimpleNamespace(get_command=lambda name: command))


@pytest.mark.parametrize("vx, want", [(0.0, 0.0), (0.05, 0.5), (0.1, 1.0), (0.3, 1.0), (-0.1, 0.0)])
def test_linear_pays_the_fraction_of_the_command_achieved(vx, want):
    got = command_progress_linear(_env((0.1, 0.0, 0.0), lin=(vx, 0.0, 0.0))).item()
    assert got == pytest.approx(want)


def test_linear_measures_along_the_command_not_any_motion():
    # Commanded sideways; walking forward is no progress.
    assert command_progress_linear(_env((0.0, 0.2, 0.0), lin=(0.2, 0.0, 0.0))).item() == 0.0


def test_linear_pays_nothing_below_the_stand_threshold():
    # A 0.01 m/s command counts as "stand": the term stays out of it.
    assert command_progress_linear(_env((0.01, 0.0, 0.0), lin=(0.01, 0.0, 0.0))).item() == 0.0


@pytest.mark.parametrize("wz, want", [(0.0, 0.0), (-0.5, 0.5), (-1.5, 1.0), (0.5, 0.0)])
def test_angular_pays_in_the_commanded_sense(wz, want):
    got = command_progress_angular(_env((0.0, 0.0, -1.0), ang=(0.0, 0.0, wz))).item()
    assert got == pytest.approx(want)


def test_angular_pays_nothing_below_the_stand_threshold():
    assert command_progress_angular(_env((0.0, 0.0, 0.05), ang=(0.0, 0.0, 0.05))).item() == 0.0


# b003d: linear tracking judged relative to the command.

STD = math.sqrt(0.1)   # mjlab's linear width in VelStand


def _batch(cmds, lins):
    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor(lins, dtype=torch.float32),
                           root_link_ang_vel_b=torch.zeros(len(lins), 3))
    command = torch.tensor(cmds, dtype=torch.float32)
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                           command_manager=SimpleNamespace(get_command=lambda name: command))


def _rel(cmds, lins, **kw):
    return track_linear_velocity_relative(_batch(cmds, lins), std=STD, **kw)


def test_relative_is_mjlabs_term_at_and_above_full_width():
    mdp = pytest.importorskip("mjlab.tasks.velocity.mdp")
    cmds = [(0.4, 0.0, 0.3), (0.3, 0.3, 0.0), (-0.45, 0.1, -1.0)]
    lins = [(0.25, 0.05, 0.03), (0.1, 0.35, -0.02), (0.0, 0.0, 0.0)]
    env = _batch(cmds, lins)
    want = mdp.track_linear_velocity(env, std=STD, command_name="twist")
    got = track_linear_velocity_relative(env, std=STD)
    assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("speed", [0.05, 0.1, 0.2, 0.3, 0.4])
def test_standing_pays_the_same_at_every_command_in_the_band(speed):
    # Any direction: what matters is the error relative to the command.
    a = 0.7
    got = _rel([(speed * math.cos(a), speed * math.sin(a), 0.0)], [(0.0, 0.0, 0.0)]).item()
    assert got == pytest.approx(math.exp(-1.6), rel=1e-5)


def test_half_the_command_pays_the_same_at_every_speed_in_the_band():
    speeds = [0.05, 0.1, 0.2, 0.4]
    got = _rel([(s, 0.0, 0.0) for s in speeds], [(s / 2, 0.0, 0.0) for s in speeds])
    assert torch.allclose(got, torch.full_like(got, math.exp(-0.4)), atol=1e-6)


def test_going_pays_more_than_standing_by_far_more_than_mjlabs_term():
    # The point of b003d: at 0.10 m/s, half the command is worth +0.47 of the term, not +0.07.
    rel = _rel([(0.1, 0.0, 0.0)] * 2, [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)])
    assert rel[1] - rel[0] == pytest.approx(math.exp(-0.4) - math.exp(-1.6), rel=1e-5)
    assert rel[1] - rel[0] > 0.45


def test_a_stand_command_keeps_mjlabs_width():
    # Standing envs (command 0) and anything under 0.02 m/s are judged exactly as before.
    got = _rel([(0.0, 0.0, 0.0), (0.01, 0.0, 0.0)], [(0.05, 0.0, 0.0), (0.0, 0.03, 0.0)])
    want = torch.tensor([math.exp(-0.0025 / 0.1), math.exp(-(0.0001 + 0.0009) / 0.1)])
    assert torch.allclose(got, want, atol=1e-6)


def test_the_width_has_a_floor_under_min_width_speed():
    # 0.03 m/s commanded: the width is 0.05/0.4 of mjlab's, not 0.03/0.4.
    got = _rel([(0.03, 0.0, 0.0)], [(0.0, 0.0, 0.0)]).item()
    assert got == pytest.approx(math.exp(-0.0009 / (0.1 * (0.05 / 0.4) ** 2)), rel=1e-5)


def test_bobbing_keeps_mjlabs_width_at_a_slow_command():
    # Vertical speed is the gait's, not the command's: 0.1 m/s up costs what it costs in mjlab.
    got = _rel([(0.1, 0.0, 0.0)], [(0.1, 0.0, 0.1)]).item()
    assert got == pytest.approx(math.exp(-0.01 / 0.1), rel=1e-5)


def test_b003d_menu_is_b003b_but_for_the_tracking_term():
    """One change, as pre-registered: the relative term replaces mjlab's at mjlab's weight and
    width, and nothing else in the recipe moves (the standing line states what runs anyway)."""
    import yaml
    b = yaml.safe_load(open("menus/b003b-progress.yaml"))["finetune"]
    d = yaml.safe_load(open("menus/b003d-relative-tracking.yaml"))["finetune"]
    assert d["rewards"] == {"track_linear_velocity": {"weight": 0.0}}
    rel = d["add_rewards"].pop("track_linear_velocity_relative")
    assert d["add_rewards"] == b["add_rewards"]
    assert d["command"] == {**b["command"], "rel_standing_envs": 0.25}
    rest = {k: v for k, v in d.items() if k not in ("rewards", "add_rewards", "command")}
    assert rest == {k: v for k, v in b.items() if k not in ("add_rewards", "command")}
    pytest.importorskip("mjlab_microduck")
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    up = load_env_cfg("Mjlab-VelStand-Flat-MicroDuck").rewards["track_linear_velocity"]
    assert rel["weight"] == up.weight and rel["params"]["std"] == pytest.approx(up.params["std"])
    assert rel["params"]["command_name"] == up.params["command_name"]
