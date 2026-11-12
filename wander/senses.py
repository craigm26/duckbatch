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
HEAD_Z_M = 0.21               # the sensor's height standing, for telling the floor from a thing
VISITOR_M = 0.6               # a visitor is this close or closer
ARRIVAL_M = 0.25              # and came at least this much closer than it was
ARRIVAL_WINDOW_S = 1.5        # within this long


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


def obstacle_points(frame: dict, pitch: float = 0.0) -> list[tuple[float, float]]:
    """Non-floor returns as (ahead, left) metres in the head's frame, projected to the ground."""
    r = ranges(frame)
    out = []
    if len(r) != ROWS * COLS:
        return out
    for row in range(1, ROWS):
        for col in range(COLS):
            x = r[row * COLS + col]
            if x is None or is_floor(row, x, pitch) or x > MEMORY_M + 0.3:
                continue
            b, e = zone_bearing(col), zone_elevation(row)
            out.append((x * math.cos(e) * math.cos(b), x * math.cos(e) * math.sin(b)))
    return out


@dataclass
class DepthReading:
    clear_ahead_m: float              # room straight ahead; 0 when the sensor is not seeing
    nearest_m: float                  # the nearest thing anywhere in the middle rows
    nearest_bearing: float            # and its direction, radians, + left
    valid: int                        # valid zones in the strip


def zone_elevation(row: int) -> float:
    half = math.radians(FOV_DEG) / 2
    return half - (row + 0.5) * 2 * half / ROWS


def is_floor(row: int, slant: float, pitch: float = 0.0) -> bool:
    """A return where the floor would be, seen from a level head: not an obstacle."""
    elev = zone_elevation(row) - pitch
    return elev < -0.05 and abs(slant * math.sin(-elev) - HEAD_Z_M) < 0.04


def read_depth(frame: dict, pitch: float = 0.0) -> DepthReading:
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
            if x is None or is_floor(row, x, pitch):
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
    pitch: float = 0.0            # the head's pitch, set by the caller, for the floor test
    memory: list[tuple[float, float, float]] = field(default_factory=list)   # (t, x, y) in odom

    def remember(self, t: float, pose: tuple[float, float, float] | None, frame: dict, head_yaw: float) -> None:
        """Keep the frame's obstacle points, in odometry coordinates, for MEMORY_S.

        The sensor sees 45 degrees. A table leg the duck walked past is out of view and still
        beside it, and weaving past it is how the practice duck first fell over. A pose is needed
        to keep points still while the duck moves; without one, nothing is remembered.
        """
        if pose is None:
            return
        px, py, pyaw = pose
        h = pyaw + head_yaw
        for a, l in obstacle_points(frame, self.pitch):
            self.memory.append((t, px + a * math.cos(h) - l * math.sin(h), py + a * math.sin(h) + l * math.cos(h)))
        self.memory = [m for m in self.memory if t - m[0] <= MEMORY_S
                       and math.hypot(m[1] - px, m[2] - py) <= MEMORY_M][-600:]

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
               steady: bool = True, pose: tuple[float, float, float] | None = None) -> Senses:
        safety = (state or {}).get("safety") or {}
        fallen = bool(safety.get("fallen") or safety.get("picked_up") or safety.get("limp"))
        if tof is None:
            return Senses(t=t, clear_ahead_m=0.0, fallen=fallen)
        depth = read_depth(tof, pitch=self.pitch)
        self.remember(t, pose, tof, head_yaw)
        # The sensor turns with the head. Room ahead is only room ahead while the head looks ahead;
        # what the duck remembers of the room counts whichever way the head is turned.
        clear = depth.clear_ahead_m if abs(head_yaw) < 0.25 else 0.0
        clear = min(clear, self.remembered_clear(pose))
        s = Senses(t=t, clear_ahead_m=clear, fallen=fallen)

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
