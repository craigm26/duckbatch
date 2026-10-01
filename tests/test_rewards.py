"""The b003b progress terms pay for going, linearly, and nothing for standing."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from duckbatch.rewards import command_progress_angular, command_progress_linear


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
