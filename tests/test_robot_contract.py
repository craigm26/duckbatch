"""Pollen's own publish checks, applied here: a gait manifest passes, the app's old `kind` does not."""

import pytest

pytest.importorskip("mjlab_microduck")

from duckbatch.robot_contract import check, gait_manifest


def test_a_walking_fine_tune_manifest_is_one_the_daemon_loads():
    m = gait_manifest("b003b-progress", "test", {"task_id": "Mjlab-VelStand-Flat-MicroDuck"})
    assert m["kind"] == "perpetual" and m["slot"] == "walk" and m["obs_len"] == 61
    rows = check("teachers/velstand.onnx", m)
    assert all(ok for _, ok, _ in rows), rows


def test_the_apps_old_manifest_kind_is_refused():
    m = gait_manifest("x", "test", {})
    m["kind"] = "alpha_walking"          # what PolicyPublication wrote before 2026-10-02
    rows = dict((n, ok) for n, ok, _ in check("teachers/velstand.onnx", m))
    assert rows["manifest (daemon would load it)"] is False
