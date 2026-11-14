"""What a live duck can sense, as the brain reads it: robot.state and tof.frame into `Senses`.

The Arena gave the brain every duck's true pose. A real duck has none of that. It has its own
odometry and safety flags (`robot.state`), and an 8x8 depth sensor in its head (`tof.frame`):
row 0 the top of the frame, column 0 the duck's left, 45 degrees across, out to 4 m. That is the
whole of what this module works from.

Two readings come out of the depth frame:

- **Room ahead**: the nearest valid range in the middle columns and middle rows, the strip a duck
  walking straight would run into. The floor sits in the bottom rows and the ceiling, if it is near,
  in the top ones, so neither counts.
- **A visitor**: something that was not there a moment ago and now is, close. A wall is close too,
  but it was close a second ago as well; a hand, a cat or another duck arrives. Telling the two
  apart by change rather than by shape is crude, and it is the reading an 8x8 frame affords.

A third reading, an **edge**, comes from the bottom rows. On a level floor their returns sit at a
distance the head's height and the zone's angle fix. A return much farther than that, or none at
all, means the floor drops away there: a table edge or a stair. The duck stops before it.

**Where the sensor is.** The floor test needs the sensor's height and pitch. `mount` reads both
from `robot.state`: `frames.tof` is the sensor's pose in the trunk from robotd's own kinematics
(measured head joints), `imu.quat` turns the trunk into the world, and `odom.position[2]` is the
trunk's height. A duck standing on duck-sim's velstand holds its sensor at 0.233 m, nose down
10.6 degrees, not level at 0.21 m. Without a state that says, the same standing values stand in.

**Which pitch a frame was taken at.** `robot.state` comes at 2 Hz and depth at 15 Hz, and the
head bobs with the gait, so the state's pitch is up to half a second old when a frame is read.
In W1's duck-sim logs the two differed by 3 to 4.5 degrees (s.d.) and by up to 17 while the duck
moved (0.3 at rest), and a frame taken with the head higher than the state says reads the floor
as below the floor: an edge in a room with no drop. So the edge test takes the frame's own pitch
from the floor in its two bottom rows (`frame_pitch`), and keeps the state's when the frame cannot
say. The obstacle test still reads at the state's pitch. A real drop is not explained
away by it: a drop shifts the two rows' floors by different amounts, a tipped head by the same.

Zones with a status other than 5 or 9 are "no measurement", per `tof/src/lib.rs`: not empty space
and not an obstacle. A frame with too few valid zones in the strip reads as no room at all, so a
duck whose sensor is covered or failing stops rather than walks blind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from brain import Senses

ROWS = COLS = 8
FOV_DEG = 45.0
STATUS_VALID = (5, 9)
MIDDLE_COLS = range(2, 6)     # about 22 degrees wide
MIDDLE_ROWS = range(2, 6)     # above the floor's rows, below the ceiling's
MIN_VALID_IN_STRIP = 4        # fewer than this in the strip and the sensor is not seeing
HALF_WIDTH_M = 0.13           # half a duck and a margin: anything inside this corridor is in the way
# Where robotd's kinematics put the sensor on a standing duck (duck-sim, velstand, head command at
# zero: robot.state frames.tof plus odom height, W1). Used only when robot.state does not say.
MOUNT_Z_M = 0.233             # the sensor's height above the floor
MOUNT_PITCH_RAD = math.radians(10.6)   # nose down, at a zero head command
TRUNK_Z_M = 0.116             # the trunk's height standing, for an odometry height that is not plausible
HEAD_Z_M = MOUNT_Z_M          # the default height for the floor test
VISITOR_M = 0.6               # a visitor is this close or closer
ARRIVAL_M = 0.25              # and came at least this much closer than it was
ARRIVAL_WINDOW_S = 1.5        # within this long


# Edges. From the standing mount (0.233 m, 10.6 degrees down) rows 6 and 7 see the floor 0.51 and
# 0.40 m ahead; from a level head at that height, 0.93 and 0.65 m, and row 5 at 1.6 m, at a grazing
# 8 degrees, where a real sensor's floor returns are unreliable.
EDGE_ROWS = (6, 7)
EDGE_LOOK_M = 1.0             # a row whose floor would be farther than this is not used for edges
EDGE_DROP_M = 0.05            # a return this much below the floor is a drop (the floor test allows 0.04)
HEAD_SETTLE_S = 0.3           # edges only from frames that arrived this long after the head last moved
# The frame's own pitch, for the edge test (see the module docstring and `frame_pitch`).
FIT_ROWS = (6, 7)             # the floor nearest the duck, where an edge it walks to shows first
FIT_MIN = 5                   # zones of a row that must agree on one floor
FIT_AGREE_RAD = math.radians(1.5)    # within this of the row's median
ROWS_AGREE_RAD = math.radians(1.0)   # and the two rows' floors within this of each other
PITCH_SLACK_RAD = math.radians(12.0) # the most a frame may differ from robot.state's pitch
PITCH_TRUST_RAD = math.radians(2.0)  # closer than this to robot.state, and robot.state's pitch stands


def zone_bearing(col: int) -> float:
    """Radians, positive to the duck's left, of a column's centre."""
    half = math.radians(FOV_DEG) / 2
    width = 2 * half / COLS
    return half - (col + 0.5) * width


def ranges(frame: dict) -> list[float | None]:
    """Metres per zone, row-major, or None where the sensor could not measure."""
    out: list[float | None] = []
    for d, s in zip(frame.get("distance_mm") or [], frame.get("status") or []):
        out.append(d / 1000.0 if s in STATUS_VALID and d > 0 else None)
    return out


MEMORY_S = 8.0                # remember an obstacle point this long after last seeing it
MEMORY_M = 1.0                # and only within this much of the duck


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _rotate(q, v) -> tuple[float, float, float]:
    w, x, y, z = q
    _, rx, ry, rz = _qmul(_qmul(q, (0.0, *v)), (w, -x, -y, -z))
    return rx, ry, rz


def mount(state: dict | None, head_pitch: float = 0.0) -> tuple[float, float]:
    """The depth sensor's pitch (rad, nose down) and height (m) above the floor, from robot.state.

    `frames.tof` is the sensor in the trunk (+x forward, +y left, +z up, quat w-first), `imu.quat`
    the trunk in the world, `odom.position[2]` the trunk's height. Without `frames.tof` the
    standing mount stands in, tipped by the head pitch the brain last sent."""
    tof = ((state or {}).get("frames") or {}).get("tof") or {}
    q, pos = tof.get("quat"), tof.get("pos")
    if not q or not pos or len(q) != 4 or len(pos) != 3:
        return MOUNT_PITCH_RAD + head_pitch, MOUNT_Z_M
    q = tuple(float(v) for v in q)
    imu = ((state or {}).get("imu") or {}).get("quat")
    trunk = tuple(float(v) for v in imu) if imu and len(imu) == 4 else (1.0, 0.0, 0.0, 0.0)
    forward = _rotate(_qmul(trunk, q), (1.0, 0.0, 0.0))
    pitch = math.asin(max(-1.0, min(1.0, -forward[2])))
    odom = ((state or {}).get("odom") or {}).get("position") or []
    trunk_z = float(odom[2]) if len(odom) == 3 and 0.05 <= float(odom[2]) <= 0.4 else TRUNK_Z_M
    return pitch, trunk_z + _rotate(trunk, tuple(float(v) for v in pos))[2]


def _row_pitch(r: list, row: int, head_z: float, lo: float, hi: float) -> float | None:
    """The pitch that puts most of a row's returns on the floor, if FIT_MIN of them agree."""
    e = zone_elevation(row)
    cands = sorted(p for p in (e + math.asin(head_z / x) for x in r[row * COLS:(row + 1) * COLS]
                               if x is not None and head_z < x) if lo < p < hi)
    n = len(cands)
    if n < FIT_MIN:
        return None
    mid = cands[n // 2] if n % 2 else (cands[n // 2 - 1] + cands[n // 2]) / 2
    return mid if sum(abs(p - mid) <= FIT_AGREE_RAD for p in cands) >= FIT_MIN else None


def frame_pitch(frame: dict, pitch: float, head_z: float = HEAD_Z_M) -> float:
    """The pitch this frame was taken at, from its own floor, or `pitch` when it cannot say.

    Each zone of FIT_ROWS names the pitch that would put its return on the floor. When each row
    has FIT_MIN zones agreeing on one, within PITCH_SLACK_RAD of `pitch`, and the two rows agree
    with each other, the frame's pitch is their mean, unless that is within PITCH_TRUST_RAD of
    `pitch` already. A gait that tipped the head between robot.state reports moves the whole floor
    together, the same pitch in both rows, and that is what this takes back out.

    A drop can look like a head tipped up, but not in both rows alike: the steeper row 7 needs a
    bigger tip than row 6 to put the same drop on the floor (an 8 cm drop 1.5 to 2 degrees more,
    a 17 cm stair 2.5 or more). So a drop that fills both rows keeps `pitch` and still reads as a
    drop, and one that fills row 6 alone leaves row 7's floor to set the pitch it is read at.
    """
    r = ranges(frame)
    if len(r) != ROWS * COLS:
        return pitch
    lo, hi = pitch - PITCH_SLACK_RAD, pitch + PITCH_SLACK_RAD
    fits = [_row_pitch(r, row, head_z, lo, hi) for row in FIT_ROWS]
    if None in fits or max(fits) - min(fits) > ROWS_AGREE_RAD:
        return pitch
    mean = sum(fits) / len(fits)
    return pitch if abs(mean - pitch) <= PITCH_TRUST_RAD else mean


def obstacle_points(frame: dict, pitch: float = 0.0, head_z: float = HEAD_Z_M) -> list[tuple[float, float]]:
    """Non-floor returns as (ahead, left) metres in the head's frame, projected to the ground."""
    r = ranges(frame)
    out = []
    if len(r) != ROWS * COLS:
        return out
    for row in range(1, ROWS):
        for col in range(COLS):
            x = r[row * COLS + col]
            if x is None or is_floor(row, x, pitch, head_z) or x > MEMORY_M + 0.3:
                continue
            b, e = zone_bearing(col), zone_elevation(row)
            out.append((x * math.cos(e) * math.cos(b), x * math.cos(e) * math.sin(b)))
    return out


def edge_points(frame: dict, pitch: float = 0.0, head_z: float = HEAD_Z_M) -> list[tuple[float, float]]:
    """Where the floor stops, as (ahead, left) metres on the ground in the head's frame.

    A bottom-row zone whose return is more than EDGE_DROP_M below the floor, or that has no return
    at all, looked past an edge. The edge is somewhere between the duck and that zone's floor
    point, so the point kept is the nearest floor the zone could have seen: its lower boundary.

    A frame that is not seeing (the strip check in `read_depth`) has no edges: a covered sensor
    already reads as no room, and the duck stands still rather than backing off blind.
    """
    r = ranges(frame)
    out: list[tuple[float, float]] = []
    if len(r) != ROWS * COLS:
        return out
    if sum(r[row * COLS + col] is not None for row in MIDDLE_ROWS for col in MIDDLE_COLS) < MIN_VALID_IN_STRIP:
        return out
    half_row = math.radians(FOV_DEG) / ROWS / 2
    for row in EDGE_ROWS:
        elev = zone_elevation(row) - pitch
        if elev >= -0.05 or head_z / math.tan(-elev) > EDGE_LOOK_M:
            continue
        near = head_z / math.tan(-(elev - half_row))
        for col in range(COLS):
            x = r[row * COLS + col]
            if x is not None and x * math.sin(-elev) <= head_z + EDGE_DROP_M:
                continue                      # the floor, or something standing on it
            b = zone_bearing(col)
            out.append((near * math.cos(b), near * math.sin(b)))
    return out


def edge_ahead(points: list[tuple[float, float]]) -> tuple[float, float]:
    """The nearest edge point inside the duck's corridor, and the mean bearing of those in front."""
    ahead = [a for a, l in points if a > -0.05 and abs(l) <= HALF_WIDTH_M]
    front = [math.atan2(l, a) for a, l in points if a > 0]
    return (max(0.0, min(ahead)) if ahead else math.inf,
            sum(front) / len(front) if front else 0.0)


@dataclass
class DepthReading:
    clear_ahead_m: float              # room straight ahead; 0 when the sensor is not seeing
    nearest_m: float                  # the nearest thing anywhere in the middle rows
    nearest_bearing: float            # and its direction, radians, + left
    valid: int                        # valid zones in the strip


def zone_elevation(row: int) -> float:
    half = math.radians(FOV_DEG) / 2
    return half - (row + 0.5) * 2 * half / ROWS


def is_floor(row: int, slant: float, pitch: float = 0.0, head_z: float = HEAD_Z_M) -> bool:
    """A return where the floor would be, seen from a head `head_z` up and `pitch` down: not an obstacle."""
    elev = zone_elevation(row) - pitch
    return elev < -0.05 and abs(slant * math.sin(-elev) - head_z) < 0.04


def read_depth(frame: dict, pitch: float = 0.0, head_z: float = HEAD_Z_M) -> DepthReading:
    """Room ahead is the nearest non-floor return inside a corridor a duck wide, any column.

    The middle strip alone missed a table corner beside the duck's path: at 0.3 m a column two
    off-centre is 0.11 m to the side, inside the duck's own width. So every column counts, by
    where its return actually sits, and the floor is set aside by its geometry instead.
    """
    r = ranges(frame)
    if len(r) != ROWS * COLS:
        return DepthReading(0.0, math.inf, 0.0, 0)
    strip = [r[row * COLS + col] for row in MIDDLE_ROWS for col in MIDDLE_COLS]
    seen = sum(x is not None for x in strip)
    if seen < MIN_VALID_IN_STRIP:
        return DepthReading(0.0, math.inf, 0.0, seen)
    clear, nearest, bearing = math.inf, math.inf, 0.0
    for row in range(1, ROWS):
        for col in range(COLS):
            x = r[row * COLS + col]
            if x is None or is_floor(row, x, pitch, head_z):
                continue
            b = zone_bearing(col)
            ahead = x * math.cos(b) * math.cos(zone_elevation(row))
            if abs(x * math.sin(b)) <= HALF_WIDTH_M and ahead < clear:
                clear = ahead
            if row in MIDDLE_ROWS and x < nearest:
                nearest, bearing = x, b
    return DepthReading(min(clear, 4.0), nearest, bearing, seen)


@dataclass
class LiveSenses:
    """Folds the latest notifications into `Senses`, one brain tick at a time."""

    history: list[tuple[float, float]] = field(default_factory=list)   # (t, nearest_m)
    visitors: int = 0
    pitch: float = 0.0            # the sensor's pitch and height, set by the caller every tick (`mount`)
    head_z: float = HEAD_Z_M
    memory: list[tuple[float, float, float]] = field(default_factory=list)   # (t, x, y) in odom
    edges: list[tuple[float, float, float]] = field(default_factory=list)    # edge points, the same way

    def remember(self, t: float, pose: tuple[float, float, float] | None, frame: dict, head_yaw: float,
                 settled: bool = True) -> None:
        """Keep the frame's obstacle points, in odometry coordinates, for MEMORY_S.

        The sensor sees 45 degrees. A table leg the duck walked past is out of view and still
        beside it, and weaving past it is how the practice duck first fell over. A pose is needed
        to keep points still while the duck moves; without one, nothing is remembered.
        """
        if pose is None:
            return
        px, py, pyaw = pose
        h = pyaw + head_yaw
        for a, l in obstacle_points(frame, self.pitch, self.head_z):
            self.memory.append((t, px + a * math.cos(h) - l * math.sin(h), py + a * math.sin(h) + l * math.cos(h)))
        self.memory = [m for m in self.memory if t - m[0] <= MEMORY_S
                       and math.hypot(m[1] - px, m[2] - py) <= MEMORY_M][-600:]
        # Edges are kept the same way: once the duck turns from a table edge, it is out of view.
        # A frame taken while the head was still moving is read at the wrong pitch, and a floor
        # seen from a head tipped up looks like a drop, so those frames add no edges.
        for a, l in (self._edges(frame) if settled else []):
            self.edges.append((t, px + a * math.cos(h) - l * math.sin(h), py + a * math.sin(h) + l * math.cos(h)))
        self.edges = [m for m in self.edges if t - m[0] <= MEMORY_S
                      and math.hypot(m[1] - px, m[2] - py) <= MEMORY_M][-600:]

    def _edges(self, frame: dict) -> list[tuple[float, float]]:
        """Edge points, read at the pitch the frame itself was taken at."""
        return edge_points(frame, frame_pitch(frame, self.pitch, self.head_z), self.head_z)

    def edge_reading(self, frame: dict, head_yaw: float, pose: tuple[float, float, float] | None,
                     settled: bool = True) -> tuple[float, float]:
        """Edge ahead of the body and its side: from memory with a pose, from this frame without."""
        if pose is None:
            c, s_ = math.cos(head_yaw), math.sin(head_yaw)
            pts = [(a * c - l * s_, a * s_ + l * c) for a, l in (self._edges(frame) if settled else [])]
        else:
            px, py, pyaw = pose
            c, s_ = math.cos(pyaw), math.sin(pyaw)
            pts = [((mx - px) * c + (my - py) * s_, -(mx - px) * s_ + (my - py) * c) for _, mx, my in self.edges]
        return edge_ahead(pts)

    def remembered_clear(self, pose: tuple[float, float, float] | None) -> float:
        """Room ahead of the body (not the head) against every remembered point."""
        if pose is None or not self.memory:
            return math.inf
        px, py, pyaw = pose
        c, s_ = math.cos(pyaw), math.sin(pyaw)
        best = math.inf
        for _, mx, my in self.memory:
            dx, dy = mx - px, my - py
            ahead, left = dx * c + dy * s_, -dx * s_ + dy * c
            if ahead > -0.05 and abs(left) <= HALF_WIDTH_M and ahead < best:
                best = max(0.0, ahead)
        return best

    def senses(self, t: float, state: dict | None, tof: dict | None, head_yaw: float,
               steady: bool = True, pose: tuple[float, float, float] | None = None,
               sent: tuple[float, float, float] = (0.0, 0.0, 0.0), settled: bool = True) -> Senses:
        safety = (state or {}).get("safety") or {}
        fallen = bool(safety.get("fallen") or safety.get("picked_up") or safety.get("limp"))
        if tof is None:
            return Senses(t=t, clear_ahead_m=0.0, fallen=fallen, pose=pose, sent=sent)
        depth = read_depth(tof, self.pitch, self.head_z)
        self.remember(t, pose, tof, head_yaw, settled)
        # The sensor turns with the head. Room ahead is only room ahead while the head looks ahead;
        # what the duck remembers of the room counts whichever way the head is turned.
        clear = depth.clear_ahead_m if abs(head_yaw) < 0.25 else 0.0
        clear = min(clear, self.remembered_clear(pose))
        edge_m, edge_b = self.edge_reading(tof, head_yaw, pose, settled)
        s = Senses(t=t, clear_ahead_m=clear, fallen=fallen, edge_ahead_m=edge_m, edge_bearing=edge_b,
                   pose=pose, sent=sent)

        self.history = [(ht, hm) for ht, hm in self.history if t - ht <= ARRIVAL_WINDOW_S]
        before = max((hm for _, hm in self.history), default=math.inf)
        self.history.append((t, depth.nearest_m))
        # Only a duck holding still can tell an arrival from its own turn: a wall swinging into a
        # turning head's view closes 0.35 m in a frame and is no visitor. `steady` is the caller's
        # word that neither the body nor the head turned since the last tick.
        arrived = steady and depth.nearest_m <= VISITOR_M and before - depth.nearest_m >= ARRIVAL_M
        if not steady:
            self.history = [(t, depth.nearest_m)]
        if arrived:
            self.visitors += 1
        if depth.nearest_m <= VISITOR_M and (arrived or self._still_here(depth)):
            s.near_id = f"visitor-{self.visitors}"
            s.near_m = depth.nearest_m
            s.near_bearing = depth.nearest_bearing - head_yaw
            s.company = 1
        return s

    def _still_here(self, depth: DepthReading) -> bool:
        """A visitor stays a visitor while it stays close, so a greeting is not cut short."""
        recent = [hm for _, hm in self.history[-6:]]
        return self.visitors > 0 and len(recent) >= 2 and max(recent) <= VISITOR_M + 0.1
