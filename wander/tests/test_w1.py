"""W1 bench plumbing that runs without a simulator: rooms, seeded starts, ground-truth metrics."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import sim_rooms  # noqa: E402
import w1  # noqa: E402


def test_starts_are_clear_and_seeded():
    for name in sim_rooms.SIM_ROOMS:
        room = sim_rooms.room_of(name)
        for seed in w1.TRAIN + w1.HOLDOUT:
            st = sim_rooms.start_for(name, seed)
            assert st == sim_rooms.start_for(name, seed)
            assert room.drop_at(st.x, st.y) == 0
            assert room.edge_margin(st.x, st.y) >= min(sim_rooms.TABLE_MARGIN, sim_rooms.DROP_MARGIN) - 1e-9
            for b in room.boxes + room.snags:
                assert not (b.x0 - sim_rooms.CLEAR < st.x < b.x1 + sim_rooms.CLEAR
                            and b.y0 - sim_rooms.CLEAR < st.y < b.y1 + sim_rooms.CLEAR)
        assert len({(s.x, s.y) for s in (sim_rooms.start_for(name, k) for k in (1, 2, 3, 101, 102))}) == 5


def test_to_room_inverts_the_scene_transform():
    st = sim_rooms.Start("living", 9, 1.2, 0.8, 2.1)
    # The scene puts room point p at world R(-yaw)(p - start); to_room must undo that.
    for px, py in ((1.2, 0.8), (0.0, 0.0), (3.0, 2.4), (2.0, 1.0)):
        c, s = math.cos(-st.yaw), math.sin(-st.yaw)
        wx, wy = c * (px - st.x) - s * (py - st.y), s * (px - st.x) + c * (py - st.y)
        rx, ry = st.to_room(wx, wy)
        assert abs(rx - px) < 1e-9 and abs(ry - py) < 1e-9
    # Facing +x in the world is facing `yaw` in the room.
    rx, ry = st.to_room(1.0, 0.0)
    assert abs(math.atan2(ry - st.y, rx - st.x) - st.yaw) < 1e-9


def test_scene_has_the_drops_and_the_sill():
    table = sim_rooms.scene_xml(sim_rooms.start_for("table", 1), "robot.xml", "")
    assert 'name="table"' in table and "floor0" in table
    stair = sim_rooms.scene_xml(sim_rooms.start_for("stair", 1), "robot.xml", "")
    assert stair.count('name="floor') == 6          # the stairwell splits the floor into a 3 x 2 grid
    assert 'name="sill0"' in sim_rooms.scene_xml(sim_rooms.start_for("threshold", 1), "robot.xml", "")
    chairs = sim_rooms.scene_xml(sim_rooms.start_for("chairs", 1), "robot.xml", "")
    assert chairs.count("leg") == 0 and 'name="seat1"' in chairs   # legs are boxes named thingN


def test_truth_counts_a_crossing_and_a_fall():
    st = sim_rooms.Start("table", 0, 1.0, 1.2, 0.0)     # facing +x on the table, edge at x = 2.4
    samples = []
    for k in range(40):                                  # walk 0.05 m per 0.25 s off the far edge
        x = 0.05 * k
        gz = -1.0 if x < 1.5 else -0.2
        samples.append((k * 0.25, k * 0.25, [x, 0.0, 0.116], gz))
    t = w1.truth(st, "table", samples, [], 0.0)
    assert t["edge_crossings"] == 1 and t["falls"] == 1
    assert t["closest_edge_m"] < 0
    assert 1.8 < t["distance_m"] < 2.0


def test_nothing_baseline_sends_zero():
    b = w1.Fixed("nothing", 1, 0.0)
    from brain import Senses
    assert b.tick(Senses(t=0, clear_ahead_m=2.0), 0.25).twist == (0.0, 0.0, 0.0)
    assert w1.Fixed("straight", 1, 0.45).tick(Senses(t=0, clear_ahead_m=0.0), 0.25).twist == (0.45, 0.0, 0.0)
