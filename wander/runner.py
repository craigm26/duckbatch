"""The brain, driving a live duck over a JSON-RPC lane.

One thread, ticking at TICK_HZ. Each tick folds the latest `robot.state` and `tof.frame` into
`Senses`, asks the brain what to do, and says it to the robot:

- **walking** is `robot.move`, a notification sent every tick while there is anything to say,
  because robotd stops a twist that is more than 500 ms old (`[safety] deadman_ms`). A lane that
  drops, a Space that crashes or a browser tab that closes all end the same way: the duck stops.
- **looking** is `robot.head`, sent when the brain's head target changes.
- **greeting and dancing** are `robot.do <emotion>`, the robot's own emotions (`curious`, `yes`,
  `excited`), one request each.
- **wheee** is `robot.sound wheee` with `hold: true`, while the brain holds it.

## Stopping is the default, and walking needs fresh eyes

The duck walks only while its depth frames are fresh (newer than FRESH_TOF_S). Over Pollen's
rendezvous that is not always: the robot forwards at most 400 notifications a minute on a control
lane (`mediad/src/relay.rs`, `NOTIFICATIONS_PER_WINDOW`), in a fixed window, and the ToF streams at
15 Hz. So for part of every minute no frames arrive at all. The runner reads that as blindness and
stands the duck still until they return. The limit protects the robot's lease on the rendezvous,
and this runs inside it rather than around it.

The same rule covers falls and pickups: `safety.fallen`, `picked_up` or `limp` stop the brain
until the robot has reported itself fine for RECOVER_S.

## Every tick is logged

Given a `session_log.Writer`, the runner writes one line per tick in the session log schema: what
it read, the senses, the brain's state and command, and what it sent. `replay.py` checks that a log
reproduces itself.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import session_log
from brain import Brain, Command
from senses import HEAD_SETTLE_S, LiveSenses

logger = logging.getLogger(__name__)

TICK_HZ = 4.0
FRESH_TOF_S = 0.6            # older depth than this and the duck is blind
FRESH_STATE_S = 3.0          # older state than this and the brain does not trust its safety flags
RECOVER_S = 2.0              # fine for this long after a fall before walking again
BRAIN_TOP_VX = 0.45          # the brain's walking speed, in the Arena's units
DEFAULT_MAX_SPEED = 0.12     # m/s: a living-room pace
MAX_YAW_RATE = 0.8           # rad/s


class Lane(Protocol):
    """What the runner needs from a connection to a duck."""

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any: ...
    def notify(self, method: str, params: dict[str, Any]) -> bool: ...
    def latest(self, method: str) -> tuple[Any, float | None]: ...   # (params, monotonic arrival)


@dataclass
class Status:
    running: bool = False
    state: str = "stopped"
    why: str = ""
    ticks: int = 0
    walking_ticks: int = 0
    blind_ticks: int = 0
    greetings: int = 0
    emotes: list[str] = field(default_factory=list)
    clear_ahead_m: float = 0.0
    tof_age_s: float | None = None
    started: float = 0.0
    sensor: str | None = None
    log: list[str] = field(default_factory=list)

    def note(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"{stamp} {line}")
        del self.log[:-60]


def scaled(cmd: Command, max_speed: float) -> tuple[float, float, float]:
    """The brain's twist in the Arena's units, as a live duck's m/s and rad/s, capped."""
    vx, vy, wz = cmd.twist
    vx = max(-1.0, min(1.0, vx / BRAIN_TOP_VX)) * max_speed     # backwards only when backing off
    wz = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, wz * MAX_YAW_RATE))
    return round(vx, 3), 0.0, round(wz, 3)


def twist_for(cmd: Command, max_speed: float, fresh_tof: bool) -> tuple[float, float, float]:
    """What the runner sends for a brain command: scaled, no creeping in a turn, still when blind."""
    if not fresh_tof:
        return (0.0, 0.0, 0.0)
    twist = scaled(cmd, max_speed)
    if cmd.state == "TurnInPlace":
        # The Arena's walker needed forward speed to turn at all; a robot's walking policy
        # turns on the spot, and creeping forward while turning is how corners get clipped.
        twist = (0.0, 0.0, twist[2])
    return twist


def head_for(cmd: Command) -> tuple[float, ...]:
    return tuple(round(h, 2) for h in cmd.head)


class Runner:
    """Starts, ticks and stops one brain on one lane. Thread-safe to start and stop."""

    def __init__(self, lane: Lane, max_speed: float = DEFAULT_MAX_SPEED, seed: int = 7,
                 emotions: dict | None = None, clock=time.monotonic,
                 log: session_log.Writer | None = None, venue: str = "practice", **log_meta: Any):
        self.lane = lane
        self.seed = seed
        self.log = log
        self.venue = venue
        self.log_meta = log_meta          # host, room, world_seed, started_utc: header fields
        self.max_speed = max_speed
        self.brain = Brain("duck", seed, emotions=emotions or {})
        self.senses = LiveSenses()
        self.status = Status()
        self.clock = clock
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_head: tuple[float, ...] | None = None
        self._fine_since: float | None = None
        self._last_yaw = 0.0
        self._last_twist = (0.0, 0.0, 0.0)
        self._head_held = False
        self._head_at = -float("inf")     # when the head was last sent somewhere new
        self.edge_stops = 0
        self.stuck_events = 0

    # -- lifecycle --------------------------------------------------------------------------------
    def start(self, limit_s: float = 300.0, thread: bool = True) -> str:
        """Subscribe and start ticking. `thread=False` subscribes only, for a caller (a test, a
        recording) that calls `tick` itself on its own clock, so no tick runs on another thread."""
        if self.status.running:
            return "already running"
        try:
            sub = self.lane.call("robot.subscribe", {"hz": 2}) or {}
            tof = self.lane.call("tof.stream", {}) or {}
        except Exception as e:  # noqa: BLE001 - a refusal is worth showing as it came
            return f"the duck refused to start streaming: {e}"
        if not tof.get("sensor"):
            return ("this duck has no depth sensor to see with "
                    f"({tof.get('unavailable') or 'none reported'}); autonomy needs one to walk.")
        self.status = Status(running=True, state="Chill", started=self.clock(), sensor=tof.get("sensor"))
        self.status.note(f"subscribed: policy {sub.get('policy', '?')}, depth {tof.get('sensor')}")
        if self.log is not None:
            self.log.write(session_log.header(self.venue, self.brain.name, self.seed, TICK_HZ,
                                              self.max_speed, emotions=self.brain.emotions,
                                              **self.log_meta))
        self._stop.clear()
        if not thread:
            return "running"
        self._thread = threading.Thread(target=self._loop, args=(limit_s,), name="autonomy", daemon=True)
        self._thread.start()
        return "running"

    def stop(self, why: str = "stopped") -> str:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._halt()
        self.status.running = False
        self.status.state = "stopped"
        self.status.why = why
        self.status.note(f"stopped: {why}")
        return why

    def _halt(self) -> None:
        """Zero the twist, centre the head, release any wheee. Best effort: the deadman backs it."""
        for method, params in (("robot.move", {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}),
                               ("robot.head", {"neck_pitch": 0.0, "head_pitch": 0.0,
                                               "head_yaw": 0.0, "head_roll": 0.0}),
                               ("robot.sound", {"tag": "wheee", "hold": False})):
            try:
                self.lane.notify(method, params)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.lane.call("robot.stop", {})
        except Exception:  # noqa: BLE001
            pass

    # -- the loop ---------------------------------------------------------------------------------
    def _loop(self, limit_s: float) -> None:
        period = 1.0 / TICK_HZ
        try:
            while not self._stop.is_set():
                started = self.clock()
                if started - self.status.started > limit_s:
                    self.stop(f"the {limit_s / 60:.0f}-minute session ended")
                    return
                self.tick(started)
                self._stop.wait(max(0.0, period - (self.clock() - started)))
        except Exception as e:  # noqa: BLE001 - whatever broke, the duck must stop
            logger.exception("autonomy loop failed")
            self.stop(f"the loop failed: {e}")

    def tick(self, now: float) -> None:
        st = self.status
        st.ticks += 1
        t = now - st.started
        state, state_at = self.lane.latest("robot.state")
        tof, tof_at = self.lane.latest("tof.frame")
        tof_age = None if tof_at is None else now - tof_at
        st.tof_age_s = tof_age
        fresh_tof = tof_age is not None and tof_age <= FRESH_TOF_S
        if not fresh_tof:
            st.blind_ticks += 1
        state_ok = state_at is not None and now - state_at <= FRESH_STATE_S

        head_yaw = (self._last_head or (0, 0, 0, 0))[2]
        self.senses.pitch = (self._last_head or (0, 0, 0, 0))[1]
        # Still body and a head that has held its target since the last tick: only then can a
        # change in the depth frame be something arriving rather than the duck's own turn.
        steady = abs(self._last_yaw) < 0.05 and self._head_held
        pose = None
        if state_ok:
            odom = (state or {}).get("odom") or {}
            pos = odom.get("position")
            if pos and odom.get("yaw") is not None:
                pose = (float(pos[0]), float(pos[1]), float(odom["yaw"]))
        safety = ((state or {}).get("safety") or {}) if state_ok else None
        frame = tof if fresh_tof else None
        settled = tof_at is not None and tof_at - self._head_at >= HEAD_SETTLE_S
        senses = self.senses.senses(t, {"safety": safety} if state_ok else None, frame,
                                    head_yaw, steady, pose, self._last_twist, settled)
        st.clear_ahead_m = senses.clear_ahead_m
        seen = dict(depth=frame, depth_age_s=tof_age, pose=pose,
                    safety=None if safety is None else {k: bool(safety.get(k)) for k in ("fallen", "picked_up", "limp")},
                    head_yaw=head_yaw, pitch=self.senses.pitch, steady=steady, settled=settled, senses=senses)

        if senses.fallen or not state_ok:
            # Not knowing is not the same as fine. A stale or missing `robot.state` gives the brain
            # no safety flags at all, so it holds still exactly as if the robot had said "fallen",
            # and the recovery clock starts again only from a fresh report that says otherwise.
            self._fine_since = None
            label = "resting" if senses.fallen else "waiting for the robot's state"
            if st.state != label:
                st.note("fallen or picked up: holding still" if senses.fallen
                        else "no fresh robot.state: holding still until it reports")
            st.state = label
            self._send_twist((0.0, 0.0, 0.0))
            self._write(seen, None, None, label)
            return
        if self._fine_since is None:
            self._fine_since = now
        if now - self._fine_since < RECOVER_S and st.ticks > 1:
            self._send_twist((0.0, 0.0, 0.0))
            self._write(seen, None, None, "recovering")
            return

        before = self.brain.state
        cmd = self.brain.tick(senses, 1.0 / TICK_HZ)
        if cmd.state != before:
            st.note(f"{before} -> {cmd.state}" + (f" ({senses.near_id} at {senses.near_m:.2f} m)"
                                                    if cmd.state == "Greet" and senses.near_id else "")
                    + (f" ({self.brain.reason})" if cmd.state == "BackOff" else ""))
            if cmd.state == "BackOff":
                if self.brain.reason == "edge":
                    self.edge_stops += 1
                else:
                    self.stuck_events += 1
        st.state = cmd.state

        twist = twist_for(cmd, self.max_speed, fresh_tof)
        if not fresh_tof:
            if st.blind_ticks == 1 or st.blind_ticks % 40 == 0:
                st.note("no fresh depth frames: standing still until they come back")
        elif twist[0] > 0:
            st.walking_ticks += 1
        self._send_twist(twist)

        head = head_for(cmd)
        self._head_held = head == self._last_head
        head_sent = None
        if head != self._last_head:
            self.lane.notify("robot.head", dict(zip(("neck_pitch", "head_pitch", "head_yaw", "head_roll"), head)))
            self._last_head = head
            self._head_at = now
            head_sent = head
        if cmd.emote:
            self._emote(cmd.emote)
            if cmd.state == "Greet":
                st.greetings += 1
        if cmd.wheee:
            self.lane.notify("robot.sound", {"tag": "wheee", "hold": True})
        b = self.brain
        self._write(seen, {"state": b.state, "since": b.since, "energy": b.energy, "turn_dir": b.turn_dir,
                           "reason": b.reason, "target": b.target}, cmd, cmd.state,
                    head_sent=head_sent, emote=cmd.emote, wheee=cmd.wheee)

    def _write(self, seen: dict, brain: dict | None, cmd: Command | None, label: str,
               head_sent: tuple | None = None, emote: str | None = None, wheee: bool = False) -> None:
        if self.log is None:
            return
        self.log.write(session_log.tick(self.status.ticks - 1, seen["senses"].t, brain=brain, command=cmd,
                                        twist=self._last_twist, head=head_sent, emote=emote, wheee=wheee,
                                        runner=label, **seen))

    def _send_twist(self, twist: tuple[float, float, float]) -> None:
        self._last_yaw = twist[2]
        self._last_twist = twist
        self.lane.notify("robot.move", {"vx": twist[0], "vy": twist[1], "vyaw": twist[2]})

    def _emote(self, name: str) -> None:
        self.status.emotes.append(name)
        threading.Thread(target=self._do, args=(name,), daemon=True).start()

    def _do(self, name: str) -> None:
        try:
            self.lane.call("robot.do", {"skill": name})
            self.status.note(f"played {name}")
        except Exception as e:  # noqa: BLE001
            self.status.note(f"robot.do {name} refused: {e}")
