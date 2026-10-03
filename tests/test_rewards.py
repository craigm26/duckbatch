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


# b003e: the same relative term on the planar velocity averaged over half a second.

from duckbatch.rewards import track_linear_velocity_stride  # noqa: E402

PARAMS = {"command_name": "twist", "std": STD, "full_width_speed": 0.4, "min_width_speed": 0.05,
          "stand_below": 0.02, "window_s": 0.5}


class _Walk:
    """A stand-in env: set velocities, step, and call the stride term the way mjlab does."""

    def __init__(self, cmds):
        self.n = len(cmds)
        self.command = torch.tensor(cmds, dtype=torch.float32)
        self.data = SimpleNamespace(root_link_lin_vel_b=torch.zeros(self.n, 3),
                                    root_link_ang_vel_b=torch.zeros(self.n, 3))
        self.env = SimpleNamespace(num_envs=self.n, device="cpu", step_dt=0.02, common_step_counter=0,
                                   scene={"robot": SimpleNamespace(data=self.data)},
                                   command_manager=SimpleNamespace(get_command=lambda name: self.command))
        self.term = track_linear_velocity_stride(SimpleNamespace(params=PARAMS), self.env)

    def step(self, lin):
        self.data.root_link_lin_vel_b = torch.tensor(lin, dtype=torch.float32)
        self.env.common_step_counter += 1
        return self.term(self.env, **PARAMS)


def test_stride_window_is_half_a_second_of_steps():
    assert _Walk([(0.1, 0.0, 0.0)]).term.window == 25


def test_stride_at_a_steady_velocity_is_b003ds_term():
    cmds = [(0.1, 0.0, 0.0), (0.3, 0.1, 0.5), (0.45, 0.0, 0.0), (0.0, 0.0, 0.0), (0.01, 0.0, 0.0)]
    lins = [(0.05, 0.01, 0.02), (0.2, 0.0, 0.0), (0.3, -0.05, 0.01), (0.02, 0.0, 0.0), (0.0, 0.03, 0.0)]
    w = _Walk(cmds)
    for _ in range(30):
        got = w.step(lins)
    assert torch.allclose(got, _rel(cmds, lins), atol=1e-6)


def test_stride_does_not_count_sway_that_averages_out():
    # Walking 0.10 m/s exactly with 0.12 m/s of sideways sway, one sway period per window.
    w = _Walk([(0.1, 0.0, 0.0)])
    stride, instant = [], []
    for k in range(50):
        lin = [(0.1, 0.12 * math.sin(2 * math.pi * k / 25), 0.0)]
        stride.append(w.step(lin).item())
        instant.append(_rel([(0.1, 0.0, 0.0)], lin).item())
    assert min(stride[25:]) > 0.99                   # the average is the command
    assert sum(instant[25:]) / 25 < 0.6              # b003d's instantaneous term charges the sway


def test_a_new_episode_starts_with_no_history():
    w = _Walk([(0.1, 0.0, 0.0)])
    for _ in range(30):
        w.step([(0.3, 0.0, 0.0)])
    w.term.reset(env_ids=torch.tensor([0]))
    got = w.step([(0.0, 0.0, 0.0)]).item()
    assert got == pytest.approx(math.exp(-1.6), rel=1e-5)   # standing, not a blend with the walk


def test_other_envs_keep_their_history_while_one_resets_every_step():
    # With thousands of envs some episode ends nearly every step; the rest must still update.
    w = _Walk([(0.2, 0.0, 0.0), (0.2, 0.0, 0.0)])
    w.step([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])     # both windows start at a standstill
    for _ in range(30):                              # then env 0 walks; env 1 restarts every step
        w.term.reset(env_ids=torch.tensor([1]))
        got = w.step([(0.2, 0.0, 0.0), (0.0, 0.0, 0.0)])
    assert got[0].item() == pytest.approx(1.0, abs=1e-6)
    assert got[1].item() == pytest.approx(math.exp(-1.6), rel=1e-5)


def test_a_second_call_in_the_same_step_does_not_push_twice():
    w = _Walk([(0.1, 0.0, 0.0)])
    w.step([(0.0, 0.0, 0.0)])                        # first call fills the window with 0
    once = w.step([(0.25, 0.0, 0.0)]).item()         # one push: the average is 0.25 / 25
    again = w.term(w.env, **PARAMS).item()           # same step, called again
    assert once == again == pytest.approx(_rel([(0.1, 0.0, 0.0)], [(0.01, 0.0, 0.0)]).item(), rel=1e-5)


def test_b003e_menu_is_b003d_but_for_the_stride_average():
    import yaml
    d = yaml.safe_load(open("menus/b003d-relative-tracking.yaml"))
    e = yaml.safe_load(open("menus/b003e-stride-average.yaml"))
    rel = d["finetune"]["add_rewards"].pop("track_linear_velocity_relative")
    stride = e["finetune"]["add_rewards"].pop("track_linear_velocity_stride")
    assert stride["func"] == "duckbatch.rewards.track_linear_velocity_stride"
    assert stride["weight"] == rel["weight"]
    assert stride["params"] == {**rel["params"], "window_s": 0.5}
    assert e["finetune"] == d["finetune"]
    assert {k: v for k, v in e.items() if k not in ("batch_id", "finetune")} == \
        {k: v for k, v in d.items() if k not in ("batch_id", "finetune")}


def _ball_env(vel, direction=(1.0, 0.0)):
    data = SimpleNamespace(root_link_lin_vel_w=torch.tensor([vel]))
    env = SimpleNamespace(scene={"ball": SimpleNamespace(data=data)}, num_envs=1, device="cpu")
    env._ball_kick_dir_w = torch.tensor([direction])
    return env


def test_a_ball_going_straight_is_not_charged():
    pytest.importorskip("mjlab_microduck")
    from duckbatch.rewards import ball_lateral_speed
    assert ball_lateral_speed(_ball_env((1.3, 0.0, 0.0))).item() == pytest.approx(0.0)


def test_the_sideways_speed_is_charged_either_side_and_along_any_line():
    pytest.importorskip("mjlab_microduck")
    from duckbatch.rewards import ball_lateral_speed
    assert ball_lateral_speed(_ball_env((1.0, 0.2, 0.0))).item() == pytest.approx(0.2)
    assert ball_lateral_speed(_ball_env((1.0, -0.2, 0.0))).item() == pytest.approx(0.2)
    # Kick line along +y: speed along x is now the sideways part.
    assert ball_lateral_speed(_ball_env((0.3, 1.0, 0.0), direction=(0.0, 1.0))).item() == pytest.approx(0.3)
