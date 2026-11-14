"""Wander plan W0: edge stop, stuck recovery, the session log, and the golden traces."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import brain as brain_mod
import session_log as sl
import sim_check
from brain import BACKOFF_VX, EDGE_M, HALT_S, STUCK_S, Brain, Senses
from replay import replay_brain, replay_full
from senses import EDGE_ROWS, HEAD_Z_M, edge_points, read_depth, zone_elevation
from test_wander import floor_frame

GOLDEN = Path(__file__).with_name("golden")


def drop_frame(drop_m=None, cols=range(8), rows=EDGE_ROWS):
    """A level floor whose bottom rows look past an edge: a lower floor, or nothing (drop_m None)."""
    f = floor_frame()
    for row in rows:
        e = -zone_elevation(row)
        for col in cols:
            i = row * 8 + col
            if drop_m is None:
                f["distance_mm"][i], f["status"][i] = 0, 255
            else:
                f["distance_mm"][i] = int((HEAD_Z_M + drop_m) / math.sin(e) * 1000)
    return f


def walking(b: Brain, t: float = 10.0) -> Brain:
    b.state, b.since = "Wander", t - 1.0
    return b


# -- 1. edge stop ----------------------------------------------------------------------------------
def test_a_level_floor_has_no_edge():
    assert edge_points(floor_frame()) == []
    assert read_depth(floor_frame()).clear_ahead_m > 1.0


@pytest.mark.parametrize("drop_m", [0.72, 0.17, 0.06])     # a table, a stair, a kerb just past the limit
def test_a_floor_that_falls_away_is_an_edge(drop_m):
    pts = edge_points(drop_frame(drop_m))
    assert pts and min(a for a, _ in pts) < EDGE_M


def test_a_floor_that_vanishes_is_an_edge():
    assert edge_points(drop_frame(None))


def test_a_small_dip_is_floor_not_an_edge():
    assert edge_points(drop_frame(0.03)) == []


def test_a_covered_sensor_has_no_edges():
    """It already reads as no room and the duck stands still; it must not back off blind."""
    f = {"rows": 8, "cols": 8, "distance_mm": [0] * 64, "status": [255] * 64}
    assert edge_points(f) == []


def test_an_edge_to_the_left_turns_the_duck_right():
    from senses import LiveSenses
    s = LiveSenses().senses(0.0, None, drop_frame(0.72, cols=(0, 1, 2, 3)), 0.0)
    assert s.edge_ahead_m < EDGE_M and s.edge_bearing > 0
    b = walking(Brain("d", 1))
    assert b.tick(Senses(t=10.0, clear_ahead_m=2.0, edge_ahead_m=s.edge_ahead_m,
                         edge_bearing=s.edge_bearing), 0.25).state == "BackOff"
    assert b.turn_dir == -1.0


def test_at_an_edge_the_brain_stops_backs_off_then_turns_away():
    b = walking(Brain("d", 1))
    edge = dict(clear_ahead_m=2.0, edge_ahead_m=0.5, edge_bearing=-0.3)
    seen = [b.tick(Senses(t=10.0 + 0.25 * k, **edge), 0.25) for k in range(10)]
    assert [c.state for c in seen[:6]] == ["BackOff"] * 6
    assert all(c.twist == (0.0, 0.0, 0.0) for c in seen[:int(HALT_S / 0.25)])        # stop
    assert seen[int(HALT_S / 0.25)].twist[0] == BACKOFF_VX                              # back off
    assert seen[6].state == "TurnInPlace" and b.turn_dir == 1.0                         # turn away
    assert all(c.state == "TurnInPlace" for c in seen[6:])     # and not walk on while it is still there


def test_the_brain_does_not_set_off_towards_an_edge():
    b = Brain("d", 1)
    b.state, b.since, b.energy = "Chill", 0.0, 0.9
    cmd = b.tick(Senses(t=5.0, clear_ahead_m=2.0, edge_ahead_m=0.7), 0.25)
    assert cmd.state == "TurnInPlace"


@pytest.mark.parametrize("room", ["table", "stair"])
@pytest.mark.parametrize("budget", [None, 400])
def test_no_practice_session_crosses_an_edge(room, budget):
    """The W0 gate: every seeded session in a room with an edge, and none goes over."""
    stops = 0
    for brain_seed in (1, 4, 7, 10):
        _, duck, r = sim_check.run(seconds=120, budget=budget, room=room, brain_seed=brain_seed, quiet=True)
        assert duck.edge_crossings == 0 and not duck.fallen, (room, brain_seed, duck.notes)
        assert duck.closest_edge > 0.09, (room, brain_seed, duck.closest_edge)     # not even a foot over
        stops += r.edge_stops
    assert stops > 0                                   # it met the edge, so the zero means something


def test_without_the_edge_rule_the_table_duck_goes_over(monkeypatch):
    """The gate test can fail: switch the rule off and the practice duck walks off the table."""
    monkeypatch.setattr(brain_mod, "EDGE_M", -1.0)
    monkeypatch.setattr(brain_mod, "EDGE_CLEAR_M", -1.0)
    _, duck, _ = sim_check.run(seconds=120, budget=None, room="table", brain_seed=7, quiet=True)
    assert duck.edge_crossings == 1 and duck.fallen


# -- 2. stuck recovery -----------------------------------------------------------------------------
def test_stuck_after_a_still_odometry_for_long_enough():
    b = walking(Brain("d", 1), t=0.0)
    still = dict(clear_ahead_m=2.0, pose=(1.0, 1.0, 0.0), sent=(0.12, 0.0, 0.0))
    states = [b.tick(Senses(t=0.25 * k, **still), 0.25).state for k in range(int(STUCK_S / 0.25) + 2)]
    assert "BackOff" not in states[:int(STUCK_S / 0.25)]
    assert states[-1] == "BackOff" and b.reason == "stuck"


@pytest.mark.parametrize("pose_of, sent", [
    (lambda k: (1.0 + 0.03 * k, 1.0, 0.0), (0.12, 0.0, 0.0)),    # moving: 3 cm a tick
    (lambda k: (1.0, 1.0, 0.0), (0.0, 0.0, 0.0)),                # nothing was sent
    (lambda k: None, (0.12, 0.0, 0.0)),                          # no odometry at all
])
def test_not_stuck(pose_of, sent):
    b = walking(Brain("d", 1), t=0.0)
    for k in range(12):
        b.tick(Senses(t=0.25 * k, clear_ahead_m=2.0, pose=pose_of(k), sent=sent), 0.25)
        b.since = 0.25 * k - 1.0                                  # keep it walking, not off to look around
        assert b.state == "Wander"


def test_a_duck_held_by_the_sill_backs_off_and_moves_on():
    _, duck, r = sim_check.run(seconds=15, budget=None, room="threshold", brain_seed=4, quiet=True, t0=0.0)
    assert r.stuck_events == 1 and not duck.fallen
    after = [p for p in duck.path if p[0] > 9.0]                  # BackOff at 6.5, turned by 9
    assert max(math.hypot(x - after[0][1], y - after[0][2]) for _, x, y in after) > 0.1


# -- 3. session log --------------------------------------------------------------------------------
def test_the_depth_frame_packs_losslessly():
    f = floor_frame()
    f["distance_mm"][5], f["status"][5] = 4000, 9
    back = sl.unpack_depth(sl.pack_depth(f))
    assert back["distance_mm"] == f["distance_mm"] and back["status"] == f["status"]
    assert len(sl.pack_depth(f)["zones"]) == 256


def test_out_of_range_depth_is_refused_not_squeezed():
    f = floor_frame()
    f["distance_mm"][0] = 70000
    with pytest.raises(sl.LogError):
        sl.pack_depth(f)


def test_senses_and_commands_round_trip_through_a_line():
    s = Senses(t=1.5, clear_ahead_m=math.inf, near_id="visitor-1", near_m=0.4, near_bearing=-0.2,
               company=1, edge_ahead_m=0.51, edge_bearing=0.3, pose=(1.0, 2.0, 0.5), sent=(0.1, 0.0, -0.8))
    line = sl.dumps(sl.tick(0, 1.5, depth=floor_frame(), depth_age_s=0.02, pose=s.pose, safety=None,
                            head_yaw=0.0, pitch=0.0, steady=True, settled=True, senses=s, brain=None,
                            command=Brain("d", 1).tick(s, 0.25), twist=(0.1, 0.0, 0.0), head=None,
                            emote=None, wheee=False, runner="Wander"))
    json.loads(line, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    rec = sl.loads(line)
    assert sl.senses_from(rec["senses"]) == s
    assert rec["gate"] == {"verdict": "none", "rule": None}


def test_the_schema_version_is_checked():
    h = sl.header("practice", "duck", 7, 4.0, 0.12)
    assert h["schema"] == sl.SCHEMA and h["v"] == sl.VERSION == 1
    with pytest.raises(sl.LogError):
        sl.loads(sl.dumps({**h, "v": 2}))
    with pytest.raises(sl.LogError):
        sl.header("arena", "duck", 7, 4.0, 0.12)


def test_only_the_gate_verdicts_are_accepted():
    assert sl.VERDICTS == ("none", "pass", "clamped", "slowed", "held", "cut", "invalid")
    with pytest.raises(sl.LogError):
        sl.tick(0, 0.0, depth=None, depth_age_s=None, pose=None, safety=None, head_yaw=0.0, pitch=0.0,
                steady=False, settled=False, senses=Senses(t=0.0, clear_ahead_m=0.0), brain=None,
                command=None, twist=(0.0, 0.0, 0.0), head=None, emote=None, wheee=False, runner="x",
                verdict="safe")


@pytest.mark.parametrize("room", ["living", "table"])
def test_a_practice_run_replays_to_the_same_commands(room):
    """With the lane limit on, so the log has blind ticks and held ticks as well as walking."""
    w = sl.Writer()
    sim_check.run(seconds=45, budget=400, room=room, brain_seed=3, log=w, quiet=True)
    header, ticks = sl.read(w.lines)
    assert header["venue"] == "practice" and len(ticks) == 180
    assert any(t["in"]["depth"] is None for t in ticks) and any(t["command"] for t in ticks)
    assert replay_brain(header, ticks) == []
    assert replay_full(header, ticks) == []


def test_a_tampered_log_does_not_replay():
    w = sl.Writer()
    sim_check.run(seconds=20, budget=None, room="living", log=w, quiet=True)
    header, ticks = sl.read(w.lines)
    walk = next(t for t in ticks if t["command"] and t["command"]["twist"][0] > 0)
    walk["senses"]["clear_ahead_m"] = 0.1
    assert replay_brain(header, ticks) and replay_full(header, ticks)


# -- 5. golden traces ------------------------------------------------------------------------------
GOLDENS = sorted(GOLDEN.glob("*.jsonl"))


def test_the_golden_traces_are_there_and_small():
    names = {p.stem for p in GOLDENS}
    assert {f"{room}-brain{b}" for room, _, b, _ in sim_check.GOLDEN_RUNS} == names
    assert sum(p.stat().st_size for p in GOLDENS) < 400_000


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.stem)
def test_the_brain_reproduces_a_golden_trace(path):
    header, ticks = sl.read(path.read_text().splitlines())
    assert replay_brain(header, ticks) == []
    assert replay_full(header, ticks) == []


def test_the_golden_traces_cover_each_rule():
    def events(stem):
        _, ticks = sl.read((GOLDEN / f"{stem}.jsonl").read_text().splitlines())
        return {(t["brain"] or {}).get("reason") for t in ticks if t["runner"] == "BackOff"}, \
               {t["command"]["emote"] for t in ticks if t["command"]}
    assert "edge" in events("table-brain10")[0]
    assert "edge" in events("stair-brain4")[0]
    assert "stuck" in events("threshold-brain4")[0]
    assert "curious" in events("living-brain11")[1]
