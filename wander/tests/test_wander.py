"""The parts that keep a duck safe, pinned: blind means still, the floor is not a wall, a corner
beside the path counts, and the practice duck keeps the lane's budget."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain import Command
from practice import PracticeDuck
from runner import FRESH_TOF_S, Runner, scaled
from senses import HEAD_Z_M, LiveSenses, read_depth, zone_bearing, zone_elevation


def frame(fill_mm=None, status=5, overrides=None):
    d = [fill_mm or 0] * 64
    s = [status if fill_mm else 255] * 64
    for (row, col), mm in (overrides or {}).items():
        d[row * 8 + col], s[row * 8 + col] = mm, 5
    return {"rows": 8, "cols": 8, "distance_mm": d, "status": s}


def floor_frame():
    """Nothing in range but the floor, as a level head sees it."""
    over = {}
    for row in range(8):
        e = zone_elevation(row)
        if e < -0.05:
            for col in range(8):
                over[(row, col)] = int(HEAD_Z_M / math.sin(-e) * 1000)
    f = frame(overrides=over)
    # Zones with no target are "no measurement"; give the strip a far wall so it is seeing.
    for row in range(2, 6):
        for col in range(2, 6):
            if f["status"][row * 8 + col] != 5:
                f["distance_mm"][row * 8 + col], f["status"][row * 8 + col] = 3500, 5
    return f


def test_column_zero_is_left():
    assert zone_bearing(0) > 0 > zone_bearing(7)


def test_floor_is_not_an_obstacle():
    assert read_depth(floor_frame()).clear_ahead_m > 1.0


def test_a_covered_sensor_reads_as_no_room():
    assert read_depth(frame()).clear_ahead_m == 0.0


def test_a_corner_beside_the_path_counts():
    f = floor_frame()
    f["distance_mm"][3 * 8 + 1], f["status"][3 * 8 + 1] = 300, 5   # column 1, 0.3 m: 0.11 m to the left
    assert read_depth(f).clear_ahead_m < 0.35


def test_a_visitor_is_something_that_arrives():
    s = LiveSenses()
    far, near = floor_frame(), floor_frame()
    for col in (3, 4):
        for row in (3, 4):
            near["distance_mm"][row * 8 + col] = 450
    assert s.senses(0.0, None, far, 0.0).near_id is None
    assert s.senses(0.5, None, near, 0.0).near_id is not None


def test_a_wall_the_duck_turned_to_is_not_a_visitor():
    s = LiveSenses()
    far, near = floor_frame(), floor_frame()
    for col in range(8):
        near["distance_mm"][3 * 8 + col] = 450
    s.senses(0.0, None, far, 0.0)
    assert s.senses(0.5, None, near, 0.0, steady=False).near_id is None


def test_turn_in_place_does_not_creep_and_speed_is_capped():
    assert scaled(Command(twist=(0.45, 0.0, 0.0)), 0.12)[0] == 0.12
    assert scaled(Command(twist=(9.0, 0.0, 9.0)), 0.12) == (0.12, 0.0, 0.8)


def test_blind_means_standing_still():
    t = [0.0]
    duck = PracticeDuck(budget=None, clock=lambda: t[0], auto=False)
    r = Runner(duck, clock=lambda: t[0])
    r._emote = lambda name: None
    assert r.start(limit_s=1e6) == "running"
    r._stop.set(); r._thread.join()
    for _ in range(200):                     # walk for a while with fresh frames
        duck.step(t[0]); r.tick(t[0]); t[0] += 0.25
    duck.streaming = False                   # frames stop arriving
    t[0] += FRESH_TOF_S + 0.1
    duck.step(t[0]); r.tick(t[0])
    assert duck.twist[0] == 0.0 and duck.twist[2] == 0.0


def test_the_practice_duck_keeps_the_lane_budget():
    t = [0.0]
    duck = PracticeDuck(budget=400, clock=lambda: t[0], auto=False)
    duck.call("robot.subscribe", {"hz": 2}); duck.call("tof.stream", {})
    for _ in range(int(59 / 0.02)):          # inside one 60 s window
        duck.step(t[0]); t[0] += 0.02
    assert duck._spent == 400 and duck.dropped > 0
