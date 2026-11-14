"""Where the depth sensor is, and which pitch a frame was taken at (BrainBench W1 findings 2 and 3).

Frames here are cast the way duck-sim's `tof.py` casts them: 8x8 rays over 45 degrees, row 0 at
the top and column 0 on the left, turned by the sensor's pitch, against a floor that may stop at
a straight edge D metres ahead and drop away by a given depth (or not come back at all).
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain import CLEAR_M, EDGE_M, STOP_M
from runner import Runner
from senses import (HEAD_Z_M, MOUNT_PITCH_RAD, MOUNT_Z_M, LiveSenses, edge_ahead, edge_points,
                    frame_pitch, mount, read_depth)

DEG = math.radians(1.0)
STAND = MOUNT_PITCH_RAD          # 10.6 degrees nose down, at 0.233 m: duck-sim's standing duck


def cast(head_z=MOUNT_Z_M, pitch=STAND, edge_m=math.inf, drop_m=None, wall_m=3.5):
    """One frame: floor out to `edge_m` ahead of the sensor, `drop_m` below it after (None: no
    return), and a wall `wall_m` ahead for the rays that never come down."""
    half = math.radians(22.5)
    centres = [-half + (i + 0.5) * 2 * half / 8 for i in range(8)]
    dist, status = [], []
    for row in range(8):
        el = -centres[row]
        for col in range(8):
            az = -centres[col]
            x, y, z = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
            dx, dz = x * math.cos(pitch) + z * math.sin(pitch), -x * math.sin(pitch) + z * math.cos(pitch)
            t = None
            if dz < -1e-6:
                t = head_z / -dz
                if t * dx >= edge_m:
                    t = None if drop_m is None else (head_z + drop_m) / -dz
            elif dx > 1e-3:
                t = wall_m / dx
            if t is not None and t <= 4.0:
                dist.append(int(t * 1000)); status.append(5)
            else:
                dist.append(0); status.append(255)
    return {"rows": 8, "cols": 8, "distance_mm": dist, "status": status}


def stops(frame, said_pitch, head_z=MOUNT_Z_M):
    """Would a walking duck stop on this frame, given the pitch robot.state said? The edge rule,
    or, where the floor gives no return at all, the frame is not seeing and there is no room."""
    s = LiveSenses(pitch=said_pitch, head_z=head_z).senses(0.0, None, frame, 0.0)
    return s.edge_ahead_m < EDGE_M or s.clear_ahead_m < STOP_M


# -- 1. the mount comes from robot.state --------------------------------------------------------------
def state(pitch=STAND, sensor_z=0.117, trunk_z=0.116, trunk_pitch=0.0):
    q = lambda a: [math.cos(a / 2), 0.0, math.sin(a / 2), 0.0]       # nose down about +y
    return {"odom": {"position": [0.0, 0.0, trunk_z], "yaw": 0.0},
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": q(trunk_pitch)},
            "frames": {"tof": {"pos": [0.078, 0.02, sensor_z], "quat": q(pitch)}}}


def test_the_mount_is_read_from_robot_state():
    pitch, z = mount(state(pitch=0.2, sensor_z=0.12, trunk_z=0.11))
    assert pitch == pytest.approx(0.2) and z == pytest.approx(0.23)


def test_the_trunk_tilt_adds_to_the_head_pitch():
    """frames.tof is in the trunk; a trunk leaning forward tips the sensor with it."""
    pitch, _ = mount(state(pitch=0.1, trunk_pitch=0.05))
    assert pitch == pytest.approx(0.15)


def test_without_frames_the_standing_mount_stands_in():
    """robotd's kinematics, not the old level sensor at 0.21 m, tipped by the head pitch sent."""
    assert mount(None) == (MOUNT_PITCH_RAD, MOUNT_Z_M)
    assert mount({"odom": {"position": [0, 0, 0.1], "yaw": 0}}, head_pitch=-0.1) == (MOUNT_PITCH_RAD - 0.1, MOUNT_Z_M)
    assert MOUNT_Z_M == pytest.approx(0.233) and math.degrees(MOUNT_PITCH_RAD) == pytest.approx(10.6)


def test_an_implausible_odometry_height_falls_back_to_standing():
    _, z = mount(state(sensor_z=0.117, trunk_z=0.0))
    assert z == pytest.approx(0.116 + 0.117)


class OneFrameLane:
    """A lane that always has fresh robot.state and one depth frame."""

    def __init__(self, st, frame):
        self.st, self.frame, self.now, self.sent = st, frame, 0.0, []

    def call(self, method, params=None):
        return {"sensor": "vl53l5cx"} if method == "tof.stream" else {"policy": "walk"}

    def notify(self, method, params):
        self.sent.append((method, params))
        return True

    def latest(self, method):
        return (self.st if method == "robot.state" else self.frame), self.now - 0.05


def test_a_duck_at_robotds_mount_sees_its_floor_as_floor():
    """W1: read as a level sensor at 0.21 m, the floor of a standing duck is an obstacle 0.4 to
    0.6 m ahead and an edge, and the brain turns instead of walking."""
    lane = OneFrameLane(state(), cast())
    r = Runner(lane, clock=lambda: lane.now)
    assert r.start(limit_s=1e6, thread=False) == "running"
    for k in range(4):
        lane.now = 10.0 + k * 0.25
        r.tick(lane.now)
    assert r.senses.head_z == pytest.approx(MOUNT_Z_M, abs=1e-3)
    assert r.senses.pitch == pytest.approx(STAND, abs=1e-6)
    depth = read_depth(lane.frame, r.senses.pitch, r.senses.head_z)
    assert depth.clear_ahead_m > CLEAR_M + 0.25
    assert edge_points(lane.frame, r.senses.pitch, r.senses.head_z) == []


def test_the_floor_test_takes_the_height_it_is_given():
    assert read_depth(cast(), STAND, MOUNT_Z_M).clear_ahead_m > 1.0
    assert read_depth(cast(), 0.0, 0.21).clear_ahead_m < CLEAR_M      # the old guess, for contrast
    assert HEAD_Z_M == MOUNT_Z_M


# -- 2. phantom edges: a head tipped up between robot.state reports -----------------------------------
@pytest.mark.parametrize("true_deg", [10.6, 4.0, 0.0])
@pytest.mark.parametrize("err_deg", [6, 9])
def test_a_floor_seen_with_the_head_higher_than_reported_is_no_edge(true_deg, err_deg):
    """robot.state comes at 2 Hz and the head bobs with the gait: W1's living room frames were
    taken up to 10 degrees higher than the latest state said, and read the floor as a drop."""
    floor = cast(pitch=math.radians(true_deg))
    said = math.radians(true_deg + err_deg)
    assert edge_points(floor, said, MOUNT_Z_M)            # read at the pitch it was told, a drop
    assert not stops(floor, said)                         # read at its own, the floor it is


@pytest.mark.parametrize("true_deg", [10.6, 4.0, 0.0])
@pytest.mark.parametrize("err_deg", [4, 6, 9])
def test_the_edge_test_reads_a_frame_at_its_own_pitch(true_deg, err_deg):
    floor = cast(pitch=math.radians(true_deg))
    said = math.radians(true_deg + err_deg)
    assert frame_pitch(floor, said, MOUNT_Z_M) == pytest.approx(math.radians(true_deg), abs=1.0 * DEG)
    assert edge_points(floor, frame_pitch(floor, said, MOUNT_Z_M), MOUNT_Z_M) == []


def test_a_pitch_robot_state_has_right_is_left_alone():
    floor = cast()
    assert frame_pitch(floor, STAND, MOUNT_Z_M) == STAND
    assert frame_pitch(floor, STAND + 1.5 * DEG, MOUNT_Z_M) == STAND + 1.5 * DEG


# -- 3. real drops still stop the duck ------------------------------------------------------------------
@pytest.mark.parametrize("drop_m", [0.72, 0.17, 0.08, None])        # table, stair, kerb, nothing
@pytest.mark.parametrize("err_deg", [-3, 0, 3, 6, 9])
def test_a_drop_filling_both_bottom_rows_is_still_an_edge(drop_m, err_deg):
    """Edge 0.3 m ahead of the sensor: rows 6 and 7 both look past it. A drop shifts the two
    rows' floors by different amounts, so it is not taken for a tipped head."""
    f = cast(edge_m=0.30, drop_m=drop_m)
    assert stops(f, STAND + err_deg * DEG)
    if drop_m is not None:
        assert frame_pitch(f, STAND + err_deg * DEG, MOUNT_Z_M) == STAND + err_deg * DEG
        assert edge_ahead(edge_points(f, frame_pitch(f, STAND + err_deg * DEG, MOUNT_Z_M), MOUNT_Z_M))[0] < EDGE_M


@pytest.mark.parametrize("drop_m", [0.72, 0.17, 0.08, None])
@pytest.mark.parametrize("err_deg", [0, 3, 6, 9])
def test_a_drop_in_the_far_row_is_an_edge_at_the_pitch_the_near_row_shows(drop_m, err_deg):
    """Edge 0.45 m ahead: row 6 looks past it, row 7 still sees floor."""
    assert stops(cast(edge_m=0.45, drop_m=drop_m), STAND + err_deg * DEG)


@pytest.mark.parametrize("drop_m", [0.17, 0.72, None])
@pytest.mark.parametrize("true_deg", [10.6, 4.0])
def test_walking_up_to_a_drop_it_is_an_edge_all_the_way_in(drop_m, true_deg):
    """From the first frame that stops the duck to the edge itself, no frame lets it go on."""
    pitch = math.radians(true_deg)
    seen = [stops(cast(pitch=pitch, edge_m=d / 100, drop_m=drop_m), pitch + 6 * DEG)
            for d in range(60, 4, -1)]
    first = seen.index(True)
    assert all(seen[first:]) and first < 25
