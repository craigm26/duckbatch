"""A practice duck: a room, a duck and a cat, in this process, answering what the runner asks.

For anyone without a duck online, and for testing the runner without one. It is a kinematic model,
not a physics simulator: a duck that goes where its twist says (at the rate a real one manages, and
not at all once a twist is 500 ms stale, as robotd's deadman has it), a head that turns, and a depth
sensor that casts the real sensor's 8x8 zones into the room: 45 degrees square, row 0 the top,
column 0 the left, out to 4 m, with the floor in the bottom rows as a real one sees it.

**It keeps the lane's budget.** Notifications past 400 in a fixed 60 s window are dropped, as
`mediad`'s control lane drops them, so the practice duck goes blind for part of each minute exactly
as a real one reached through the rendezvous does. Set `budget=None` for a duck on a LAN.
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
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


class PracticeDuck:
    """Implements the runner's `Lane`: call, notify, latest."""

    def __init__(self, seed: int = 1, budget: int | None = BUDGET, clock=time.monotonic,
                 auto: bool = True):
        self.rng = random.Random(seed)
        self.clock = clock
        self.budget = budget
        self.x, self.y, self.yaw = 0.6, 1.2, 0.0
        self.head = [0.0, 0.0, 0.0, 0.0]
        self.twist = (0.0, 0.0, 0.0)
        self.twist_at = -math.inf
        self.v = self.w = 0.0
        self.fallen = False
        self.cat = [2.6, 0.4, 0.0]           # x, y, heading
        self.cat_mode = "away"
        self.cat_until = 0.0
        self.boxes = [Box(1.6, 1.7, 2.2, 2.4, 0.45, "sofa end"), Box(2.5, 1.0, 3.0, 1.3, 0.4, "toy box")]
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
        if self._free(nx, ny):
            self.x, self.y = nx, ny
        elif self.v > 0.05:
            self.fallen = True                            # walked into something: down it goes
            self.notes.append("bumped and fell")
            self.v = 0.0
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
        objects = self.boxes + [Box(self.cat[0] - 0.12, self.cat[1] - 0.12,
                                    self.cat[0] + 0.12, self.cat[1] + 0.12, 0.25, "cat")]
        for row in range(ROWS):
            elev = half - (row + 0.5) * width - pitch
            floor = HEAD_Z / math.tan(-elev) if elev < -1e-3 else math.inf
            for col in range(COLS):
                az = heading + half - (col + 0.5) * width
                dx, dy = math.cos(az), math.sin(az)
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
