"""A small behavior engine for a Microduck: one brain per duck, ticked at a fixed rate.

It is pure: a tick takes what the duck can sense and returns what it does next, and nothing else.
The same Brain drives a duck on the Arena's talent stage (stage.py writes its choices down as a
timeline) and, later, a live duck through Pollen's rendezvous control lane, where the senses come
from robot.state and the ToF instead of from a model of the stage.

States follow the names in Pollen's docs/ideas/autonomous_behavior.md (Chill, LookAround, Wander,
TurnInPlace, Dance, Nap) plus Greet, the behavior that file calls the right first one: a duck
greets a duck it has not met, more warmly one it has, and dances when it has company.

Two rules from the wander plan's W0 sit above the rest. **Edge:** walking, with a drop in its path
nearer than EDGE_M, the duck stops, backs off and turns away from the side the drop is on, and
does not set off towards one. **Stuck:** a twist was sent and odometry has not moved for STUCK_S,
so it backs off and turns the other way. Both go through BackOff. A brain given no edge reading, no
pose and no sent twist (the Arena) never meets either rule and behaves exactly as before.

`Command.emote` names one of the robot's own emotions (`robot.do curious`, `yes`, `excited`): a
live duck plays those itself, so a greeting costs one call over the lane rather than a head path
streamed through it. The head path stays for the Arena, which has no emotions to call.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

BEAT_S = 0.5            # the shared beat: every brain reads the same clock, so no sync is needed
GREET_RANGE_M = 0.75    # a duck this close, roughly ahead, is noticed
CLEAR_M = 0.5           # less room than this ahead, and it turns away
STOP_M = 0.25           # less than this, and it stops: a turning walker still creeps forward
QUACK_GAP_S = 1.5       # never two quacks to the same duck inside this (Kith's rule)

# Edges (wander plan, W0 item 1). `Senses.edge_ahead_m` is how far ahead, inside the duck's path,
# the floor stops where it should be (senses.py reads it off the depth sensor's bottom rows).
EDGE_M = 0.6            # an edge this close ahead while walking: stop, back off, turn away
EDGE_CLEAR_M = 0.8      # and walk on only once nothing in the path is nearer than this
# Stuck (W0 item 2): a twist was sent, and odometry has not moved for STUCK_S.
STUCK_S = 1.5
STUCK_MOVE_M = 0.03     # less travel than this counts as not moving
STUCK_TURN_RAD = 0.15   # and less turn than this
# Backing off, for either reason: stand still a moment, then walk backwards, then turn.
HALT_S = 0.5
BACKOFF_S = 1.0
BACKOFF_VX = -0.3       # in the brain's units: a third of the way to full speed, backwards


@dataclass
class Senses:
    t: float                         # seconds since the start
    clear_ahead_m: float             # free distance ahead, the ToF's middle rows on a robot
    near_id: str | None = None       # the nearest other duck, by a stable id (a BLE beacon on a robot)
    near_m: float = math.inf
    near_bearing: float = 0.0        # radians, + to the duck's left
    company: int = 0                 # ducks within greeting range
    fallen: bool = False
    edge_ahead_m: float = math.inf   # the nearest drop in the duck's path (a table edge, a stair)
    edge_bearing: float = 0.0        # which side the drops are on, radians, + left
    pose: tuple[float, float, float] | None = None   # odometry x, y (m) and yaw (rad), if known
    sent: tuple[float, float, float] = (0.0, 0.0, 0.0)   # the twist the duck was last sent, m/s and rad/s


@dataclass
class Command:
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)   # vx, vy, wz as the walk policy reads them
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # neck, head pitch, yaw, roll (rad)
    quack: bool = False              # one quack (the pad's RT)
    wheee: bool = False              # held wheee (the pad's LT)
    sit: bool | None = None          # for a duck whose policy reads twist[0] as a sit flag
    emote: str | None = None         # an emotion the robot plays itself (`robot.do`), once
    state: str = "Chill"


@dataclass
class Brain:
    name: str
    seed: int
    can_walk: bool = True
    emotions: dict = field(default_factory=dict)   # name -> [(t, head tuple)], from microduck-emotions
    energy: float = 0.8
    state: str = "Chill"
    since: float = 0.0
    met: dict = field(default_factory=dict)        # duck id -> times greeted
    last_quack: dict = field(default_factory=dict)
    expr: list = field(default_factory=list)       # an emotion being played: [(t_abs, head)]
    turn_dir: float = 1.0
    target: str | None = None
    reason: str | None = None                      # why it is backing off: "edge" or "stuck"
    anchor: tuple | None = None                    # (t, x, y, yaw) where the stuck clock started

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # -- the tick -------------------------------------------------------------------------------
    def tick(self, s: Senses, dt: float) -> Command:
        if s.fallen:
            self._go("Chill", s.t)
            return Command(state="Fallen")
        self.energy += dt * (-0.02 if self.state == "Wander" else 0.01)
        self.energy = min(1.0, max(0.0, self.energy))
        stuck = self._stuck(s)
        if not self._back_off(s, stuck):
            self._transition(s)
        cmd = getattr(self, "_" + self.state.lower())(s)
        cmd.state = self.state
        if self.expr:                                   # an emotion overrides the head while it plays
            while len(self.expr) > 1 and self.expr[1][0] <= s.t:
                self.expr.pop(0)
            cmd.head = self.expr[0][1]
            if self.expr[-1][0] < s.t:
                self.expr = []
        return cmd

    # -- choosing what to do -----------------------------------------------------------------------
    def _go(self, state: str, t: float) -> None:
        if state != self.state:
            self.state, self.since = state, t
            self.anchor = None

    def _stuck(self, s: Senses) -> bool:
        """A twist was sent and odometry has not moved for STUCK_S. Without a pose, never."""
        vx, _, wz = s.sent
        if s.pose is None or (abs(vx) < 0.01 and abs(wz) < 0.05):
            self.anchor = None
            return False
        x, y, yaw = s.pose
        if self.anchor is None:
            self.anchor = (s.t, x, y, yaw)
            return False
        t0, x0, y0, yaw0 = self.anchor
        turned = abs(math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0)))
        if math.hypot(x - x0, y - y0) > STUCK_MOVE_M or turned > STUCK_TURN_RAD:
            self.anchor = (s.t, x, y, yaw)
            return False
        return s.t - t0 >= STUCK_S

    def _away(self, s: Senses) -> float:
        """Turn away from the side the drops are on; straight ahead turns left."""
        return -1.0 if s.edge_bearing > 0 else 1.0

    def _back_off(self, s: Senses, stuck: bool) -> bool:
        """The edge and stuck rules. True when they decided this tick's state."""
        if self.state == "BackOff":
            if s.t - self.since >= HALT_S + BACKOFF_S:
                self._go("TurnInPlace", s.t)
            return True
        if self.state == "Wander" and s.edge_ahead_m < EDGE_M:
            self.reason, self.turn_dir = "edge", self._away(s)
            self._go("BackOff", s.t)
            return True
        if self.state in ("Wander", "TurnInPlace") and stuck:
            self.reason, self.turn_dir = "stuck", -self.turn_dir
            self._go("BackOff", s.t)
            return True
        return False

    def _transition(self, s: Senses) -> None:
        held = s.t - self.since
        stranger = s.near_id is not None and s.near_m < GREET_RANGE_M and abs(s.near_bearing) < 1.2
        if self.state in ("Chill", "LookAround", "Wander", "TurnInPlace", "Nap") and stranger \
                and s.t - self.last_quack.get(s.near_id, -9) > QUACK_GAP_S and self.met.get(s.near_id, 0) < 2:
            self.target = s.near_id
            self._go("Greet", s.t)
            return
        if self.state == "Greet" and held > 1.8:
            friend = self.met.get(self.target, 0) >= 2
            self._go("Dance" if s.company >= 2 or (friend and s.company >= 1) else "LookAround", s.t)
            return
        if self.state == "Dance" and held > 4.0:
            self._go("Chill", s.t)
            return
        if not self.can_walk:
            if self.state in ("Chill", "LookAround") and self.energy < 0.55 and held > 2.0:
                self._go("Nap", s.t)
            elif self.state == "Nap" and self.energy > 0.95:
                self._go("Chill", s.t)
            elif self.state == "Chill" and held > 2.0:
                self._go("LookAround", s.t)
            elif self.state == "LookAround" and held > 2.5:
                self._go("Chill", s.t)
            return
        if self.state in ("Wander", "TurnInPlace") and s.clear_ahead_m < STOP_M:
            self._go("Chill", s.t)                      # too close to turn safely: stand still
            return
        if self.state == "Wander" and s.clear_ahead_m < CLEAR_M:
            self.turn_dir = 1.0 if self.rng.random() < 0.5 else -1.0
            self._go("TurnInPlace", s.t)
        elif self.state == "TurnInPlace" and s.clear_ahead_m > CLEAR_M + 0.25 and held > 0.8 \
                and s.edge_ahead_m >= EDGE_CLEAR_M:
            self._go("Wander", s.t)
        elif self.state == "Wander" and (held > 4.0 + 3.0 * self.rng.random() or self.energy < 0.3):
            self._go("LookAround", s.t)
        elif self.state == "LookAround" and held > 2.0:
            self._go(self._moving(s) if self.energy > 0.35 else "Chill", s.t)
        elif self.state == "Chill" and held > 1.0 + 2.0 * self.rng.random():
            self._go(self._moving(s) if self.energy > 0.4 else "LookAround", s.t)

    def _moving(self, s: Senses) -> str:
        """Set off: straight on if there is room, otherwise turn away first."""
        if s.edge_ahead_m < EDGE_CLEAR_M:
            self.turn_dir = self._away(s)
            return "TurnInPlace" if s.clear_ahead_m >= STOP_M else "LookAround"
        if s.clear_ahead_m >= CLEAR_M:
            return "Wander"
        self.turn_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        return "TurnInPlace" if s.clear_ahead_m >= STOP_M else "LookAround"

    # -- doing it ----------------------------------------------------------------------------------
    def _chill(self, s: Senses) -> Command:
        return Command(sit=False)

    def _nap(self, s: Senses) -> Command:
        self.energy = min(1.0, self.energy + 0.003)
        return Command(sit=True, head=(0.0, 0.3, 0.0, 0.0))

    def _lookaround(self, s: Senses) -> Command:
        """Glances: left, ahead, right, ahead, each held a moment. A head that never stops cannot
        tell something arriving from its own sweep, so the holds are where it notices."""
        glance = int((s.t - self.since) / 0.8 + self.seed) % 4
        yaw = (0.5, 0.0, -0.5, 0.0)[glance]
        return Command(head=(0.0, -0.1, yaw, 0.0), sit=False)

    def _wander(self, s: Senses) -> Command:
        drift = 0.6 * math.sin((s.t - self.since) * 0.7 + self.seed)   # a lazy weave, not a line
        return Command(twist=(0.45, 0.0, round(drift, 1)))

    def _backoff(self, s: Senses) -> Command:
        """Stand still for HALT_S, then walk backwards for BACKOFF_S. The turn comes after."""
        if s.t - self.since < HALT_S:
            return Command(sit=False)
        return Command(twist=(BACKOFF_VX, 0.0, 0.0), sit=False)

    def _turninplace(self, s: Senses) -> Command:
        return Command(twist=(0.4, 0.0, self.turn_dir))

    def _greet(self, s: Senses) -> Command:
        held = s.t - self.since
        cmd = Command(head=(0.0, 0.0, round(max(-0.6, min(0.6, s.near_bearing)), 2), 0.0), sit=False)
        if self.target is not None and getattr(self, "_greeted", None) != self.since:
            self._greeted = self.since
            friend = self.met.get(self.target, 0) > 0
            cmd.emote = "yes" if friend else "curious"
            self._play(cmd.emote, s.t + 0.3)
            self.met[self.target] = self.met.get(self.target, 0) + 1
            self.last_quack[self.target] = s.t
            cmd.quack = True
        return cmd

    def _dance(self, s: Senses) -> Command:
        beat = int(s.t / BEAT_S)
        first = getattr(self, "_danced", None) != self.since
        self._danced = self.since
        bob = 0.35 if beat % 2 == 0 else -0.05                    # head pitch on the shared beat
        sway = 0.4 if (beat // 2) % 2 == 0 else -0.4
        return Command(head=(0.0, bob, sway, 0.0), wheee=(beat % 4 == 0), sit=False,
                       emote="excited" if first else None)

    def _play(self, emotion: str, t0: float) -> None:
        path = self.emotions.get(emotion)
        if path:
            self.expr = [(t0 + t, h) for t, h in path]
