"""A practice duck: a room, a duck and a cat, in this process, answering what the runner asks.

For anyone without a duck online, and for testing the runner without one. It is a kinematic model,
not a physics simulator: a duck that goes where its twist says (at the rate a real one manages, and
not at all once a twist is 500 ms stale, as robotd's deadman has it), a head that turns, and a depth
sensor that casts the real sensor's 8x8 zones into the room: 45 degrees square, row 0 the top,
column 0 the left, out to 4 m, with the floor in the bottom rows as a real one sees it.

**It keeps the lane's budget.** Notifications past 400 in a fixed 60 s window are dropped, as
`mediad`'s control lane drops them, so the practice duck goes blind for part of each minute exactly
as a real one reached through the rendezvous does. Set `budget=None` for a duck on a LAN.

**Rooms.** `living` is the room it has always been. `table` puts the duck on a dining table with
the floor 0.72 m below every side. `stair` is the living room with a stairwell one step (0.17 m)
down. `threshold` is the living room with a 3 cm door sill across the duck's way: too low for the
depth sensor to tell from the floor, and enough to stop a small duck's feet. A duck whose centre
goes over a drop falls, and the crossing is counted in `edge_crossings`.
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

ROOM_X, ROOM_Y = 3.0, 2.4          # a living-room corner, metres
HEAD_Z = 0.21                      # the sensor's height when standing
FOV = math.radians(45.0)
ROWS = COLS = 8
MAX_RANGE = 4.0
DEADMAN_S = 0.5
BUDGET = 400
WINDOW_S = 60.0
IDLE_S = 120.0                     # an unsubscribed duck stops simulating after this


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    height: float
    name: str


@dataclass
class Room:
    """What the practice duck stands in. Drops are measured down from the surface it starts on."""
    name: str
    boxes: list[Box]
    start: tuple[float, float, float] = (0.6, 1.2, 0.0)
    cat: bool = True
    surface: Box | None = None          # if set, the duck stands on this and every side drops away
    pits: list[Box] = field(default_factory=list)   # lower floor, `height` is how far down
    snags: list[Box] = field(default_factory=list)  # low sills: the duck cannot walk onto them

    def drop_at(self, x: float, y: float) -> float:
        """How far below the duck's surface the floor is at (x, y). 0 on the surface itself."""
        if self.surface is not None:
            b = self.surface
            if not (b.x0 <= x <= b.x1 and b.y0 <= y <= b.y1):
                return b.height
        for p in self.pits:
            if p.x0 <= x <= p.x1 and p.y0 <= y <= p.y1:
                return p.height
        return 0.0

    def edge_margin(self, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest drop; negative over one."""
        best = math.inf
        if self.surface is not None:
            b = self.surface
            inside = min(x - b.x0, b.x1 - x, y - b.y0, b.y1 - y)
            best = inside if inside < 0 else min(best, inside)
        for p in self.pits:
            dx, dy = max(p.x0 - x, 0.0, x - p.x1), max(p.y0 - y, 0.0, y - p.y1)
            out = math.hypot(dx, dy)
            best = min(best, out if out > 0 else -min(x - p.x0, p.x1 - x, y - p.y0, p.y1 - y))
        return best


def _living() -> list[Box]:
    return [Box(1.6, 1.7, 2.2, 2.4, 0.45, "sofa end"), Box(2.5, 1.0, 3.0, 1.3, 0.4, "toy box")]


ROOMS = {
    "living": lambda: Room("living", _living()),
    "table": lambda: Room("table", [], start=(1.0, 1.2, 0.0), cat=False,
                          surface=Box(0.6, 0.7, 2.4, 1.7, 0.72, "dining table")),
    "stair": lambda: Room("stair", _living(), pits=[Box(1.2, 0.0, 1.8, 0.8, 0.17, "stairwell")]),
    "threshold": lambda: Room("threshold", _living(), snags=[Box(0.95, 0.0, 1.01, ROOM_Y, 0.03, "door sill")]),
}


def ray_box(ox: float, oy: float, dx: float, dy: float, b: Box) -> float:
    """Distance along a 2D ray to an axis-aligned box, or inf."""
    tmin, tmax = -math.inf, math.inf
    for o, d, lo, hi in ((ox, dx, b.x0, b.x1), (oy, dy, b.y0, b.y1)):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return math.inf
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
    return tmin if tmax >= max(tmin, 0.0) and tmin > 0 else math.inf


def _ray_exit(ox: float, oy: float, dx: float, dy: float, b: Box) -> float:
    """Distance along a 2D ray to where it leaves an axis-aligned box, or inf."""
    tmax = math.inf
    for o, d, lo, hi in ((ox, dx, b.x0, b.x1), (oy, dy, b.y0, b.y1)):
        if abs(d) > 1e-9:
            tmax = min(tmax, max((lo - o) / d, (hi - o) / d))
    return tmax


class PracticeDuck:
    """Implements the runner's `Lane`: call, notify, latest."""

    def __init__(self, seed: int = 1, budget: int | None = BUDGET, clock=time.monotonic,
                 auto: bool = True, room: str = "living"):
        self.rng = random.Random(seed)
        self.clock = clock
        self.budget = budget
        self.room = ROOMS[room]()
        self.x, self.y, self.yaw = self.room.start
        self.head = [0.0, 0.0, 0.0, 0.0]
        self.twist = (0.0, 0.0, 0.0)
        self.twist_at = -math.inf
        self.v = self.w = 0.0
        self.fallen = False
        self.cat = [2.6, 0.4, 0.0] if self.room.cat else [-9.0, -9.0, 0.0]   # x, y, heading
        self.cat_mode = "away"
        self.cat_until = 0.0
        self.boxes = self.room.boxes
        self.edge_crossings = 0
        self.closest_edge = self.room.edge_margin(self.x, self.y)
        self.subscribed = self.streaming = False
        self.notes: list[str] = []
        self._latest: dict[str, tuple[Any, float]] = {}
        self._spent, self._window = 0, clock()
        self.dropped = 0
        self.path: list[tuple[float, float, float]] = []
        self._lock = threading.Lock()
        self._last = clock()
        self._next_state = self._next_tof = self._last
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        if auto:
            self._thread = threading.Thread(target=self._run, daemon=True, name="practice-duck")
            self._thread.start()

    # -- Lane -------------------------------------------------------------------------------------
    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        if method == "robot.subscribe":
            self.subscribed = True
            return {"policy": "velstand (practice)"}
        if method == "tof.stream":
            self.streaming = True
            return {"accepted": True, "sensor": "VL53L8CX (practice)", "unavailable": None,
                    "rows": ROWS, "cols": COLS, "hz": 15}
        if method == "robot.stop":
            self.twist, self.twist_at = (0.0, 0.0, 0.0), self.clock()
            return None
        if method == "robot.do":
            self.notes.append(f"played {params.get('skill')}")
            return None
        raise RuntimeError(f"{method} is not something the practice duck does")

    def notify(self, method: str, params: dict[str, Any]) -> bool:
        if method == "robot.move":
            self.twist = (float(params.get("vx", 0)), float(params.get("vy", 0)), float(params.get("vyaw", 0)))
            self.twist_at = self.clock()
        elif method == "robot.head":
            self.head = [params.get(k, 0.0) for k in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
        elif method == "robot.sound":
            pass
        return True

    def latest(self, method: str) -> tuple[Any, float | None]:
        with self._lock:
            got = self._latest.get(method)
        return (got[0], got[1]) if got else (None, None)

    def close(self) -> None:
        """Stop simulating. A page that starts again gets a new duck; this one's thread ends."""
        self.subscribed = self.streaming = False
        self._closed.set()

    # -- the world --------------------------------------------------------------------------------
    def _run(self) -> None:
        # Ends on close(), or after IDLE_S with nobody subscribed: a tab that vanished without
        # closing its session must not leave a room simulating forever.
        idle_since = None
        while not self._closed.wait(0.02):
            now = self.clock()
            if self.subscribed or self.streaming:
                idle_since = None
            elif idle_since is None:
                idle_since = now
            elif now - idle_since > IDLE_S:
                return
            self.step(now)

    def step(self, now: float) -> None:
        dt = min(0.1, max(0.0, now - self._last))
        self._last = now
        vx, _, wz = self.twist if now - self.twist_at <= DEADMAN_S else (0.0, 0.0, 0.0)
        a = 1 - math.exp(-dt / 0.4)                       # getting going takes a moment
        self.v += a * (vx - self.v)
        self.w += a * (wz - self.w)
        self.yaw += self.w * dt
        nx = self.x + self.v * math.cos(self.yaw) * dt
        ny = self.y + self.v * math.sin(self.yaw) * dt
        if self._snagged(nx, ny):
            pass                                          # the sill holds its feet: no fall, no progress
        elif self._free(nx, ny):
            self.x, self.y = nx, ny
            self.closest_edge = min(self.closest_edge, self.room.edge_margin(nx, ny))
            if not self.fallen and self.room.drop_at(nx, ny) > 0:
                self.edge_crossings += 1
                self.fallen = True                        # over the edge: down it goes
                self.notes.append("went over an edge and fell")
                self.v = 0.0
        elif self.v > 0.05:
            self.fallen = True                            # walked into something: down it goes
            self.notes.append("bumped and fell")
            self.v = 0.0
        if self.room.cat:
            self._cat(now, dt)
        self.path.append((round(now, 2), round(self.x, 3), round(self.y, 3)))
        del self.path[:-3000]
        if self.subscribed and now >= self._next_state:
            self._next_state = now + 0.5
            self._publish("robot.state", self.state(), now)
        if self.streaming and now >= self._next_tof:
            self._next_tof = now + 1 / 15
            self._publish("tof.frame", self.tof(), now)

    def _publish(self, method: str, params: dict, now: float) -> None:
        if self.budget is not None:
            if now - self._window >= WINDOW_S:
                self._spent, self._window = 0, now
            if self._spent >= self.budget:
                self.dropped += 1
                return
            self._spent += 1
        with self._lock:
            self._latest[method] = (params, now)

    def _snagged(self, x: float, y: float, r: float = 0.09) -> bool:
        """Walking onto a sill is blocked; walking off one, or away from it, is not."""
        for b in self.room.snags:
            def gap(px: float, py: float) -> float:
                return math.hypot(max(b.x0 - px, 0.0, px - b.x1), max(b.y0 - py, 0.0, py - b.y1))
            if gap(x, y) < r and gap(x, y) < gap(self.x, self.y):
                return True
        return False

    def _free(self, x: float, y: float, r: float = 0.09) -> bool:
        if not (r <= x <= ROOM_X - r and r <= y <= ROOM_Y - r):
            return False
        if any(b.x0 - r <= x <= b.x1 + r and b.y0 - r <= y <= b.y1 + r for b in self.boxes):
            return False
        return math.hypot(x - self.cat[0], y - self.cat[1]) > 0.17

    def _cat(self, now: float, dt: float) -> None:
        """The visitor: wanders, and every so often walks up to the duck, sits, and leaves."""
        cx, cy, ch = self.cat
        if now >= self.cat_until:
            self.cat_mode = {"away": "visit", "visit": "sit", "sit": "away"}[self.cat_mode]
            if self.cat_mode == "visit":       # a spot beside where the duck is now, then stay put
                self.cat_goal = (self.x + 0.4 * math.cos(self.yaw + 0.3), self.y + 0.4 * math.sin(self.yaw + 0.3))
            self.cat_until = now + {"visit": 6.0, "sit": 4.0, "away": 10.0 + 8 * self.rng.random()}[self.cat_mode]
        if self.cat_mode == "visit":
            tx, ty = self.cat_goal
        elif self.cat_mode == "away":
            tx, ty = 2.7, 0.3
        else:
            return
        d = math.hypot(tx - cx, ty - cy)
        if d > 0.05:
            ch = math.atan2(ty - cy, tx - cx)
            step = min(d, 0.35 * dt)
            nx, ny = cx + step * math.cos(ch), cy + step * math.sin(ch)
            if math.hypot(nx - self.x, ny - self.y) >= 0.26:   # a polite cat: it stops short
                cx, cy = nx, ny
        self.cat = [cx, cy, ch]

    def _floor_along(self, dx: float, dy: float, elev: float, level: float) -> float:
        """Horizontal distance at which a downward ray meets the floor, drops included."""
        if level == math.inf:
            return level
        drop = self.room.drop_at(self.x + level * dx, self.y + level * dy)
        if drop <= 0:
            return level
        lower = (HEAD_Z + drop) / math.tan(-elev)
        for p in self.room.pits:                   # the far side of a pit is a step back up
            exit_t = _ray_exit(self.x, self.y, dx, dy, p)
            if exit_t < lower and p.x0 <= self.x + level * dx <= p.x1 and p.y0 <= self.y + level * dy <= p.y1:
                lower = exit_t
        return lower

    def state(self) -> dict:
        return {"t": self.clock(), "movement": {"requested": list(self.twist), "applied": [self.v, 0, self.w],
                                                "limited_by": []},
                "head": list(self.head), "policy": "walk",
                "safety": {"fallen": self.fallen, "limp": False, "picked_up": False, "gravity": [0, 0, -1]},
                "odom": {"position": [self.x, self.y, 0.0], "yaw": self.yaw}}

    def tof(self) -> dict:
        """One 8x8 frame from the head: walls, furniture and the cat in 2D, the floor by elevation."""
        half = FOV / 2
        width = FOV / COLS
        distance, status = [], []
        heading = self.yaw + self.head[2]
        pitch = self.head[1]
        objects = self.boxes + self.room.snags + [Box(self.cat[0] - 0.12, self.cat[1] - 0.12,
                                                      self.cat[0] + 0.12, self.cat[1] + 0.12, 0.25, "cat")]
        for row in range(ROWS):
            elev = half - (row + 0.5) * width - pitch
            level = HEAD_Z / math.tan(-elev) if elev < -1e-3 else math.inf
            for col in range(COLS):
                az = heading + half - (col + 0.5) * width
                dx, dy = math.cos(az), math.sin(az)
                floor = self._floor_along(dx, dy, elev, level)
                walls = []
                for edge, o, d in ((0.0, self.x, dx), (ROOM_X, self.x, dx), (0.0, self.y, dy), (ROOM_Y, self.y, dy)):
                    if abs(d) > 1e-9 and (edge - o) / d > 0:
                        walls.append((edge - o) / d)
                flat = min(walls) if walls else math.inf
                for b in objects:
                    t = ray_box(self.x, self.y, dx, dy, b)
                    # A box only stops the ray if the ray is still below its top when it gets there.
                    if t < flat and HEAD_Z + t * math.tan(elev) <= b.height:
                        flat = t
                slant = min(flat / max(math.cos(elev), 1e-3), floor)
                if slant <= MAX_RANGE:
                    noise = self.rng.gauss(0, 0.004 + 0.006 * slant)
                    distance.append(int(max(20, (slant + noise) * 1000)))
                    status.append(5)
                else:
                    distance.append(0)
                    status.append(255)
        return {"seq": len(self.path), "rows": ROWS, "cols": COLS, "distance_mm": distance, "status": status}
