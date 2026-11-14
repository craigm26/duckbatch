"""BrainBench task W1, wander, on duck-sim: the W0 runner against the real robotd and `tofd --sim`.

    python3 bench/w1.py run                 # every room, mode and seed below, then the tables
    python3 bench/w1.py run --rooms table --modes wander --seeds 1 --seconds 20   # a smoke run
    python3 bench/w1.py tables              # rebuild w1-results.{md,json} from the session files

Each session is one cold duck-sim: the room is written as a MuJoCo scene with the seeded start on
the origin (`sim_rooms.py`), `scripts/duck-sim up` brings up the body, `tofd --sim`, `robotd` and
the rest and stands the duck, the runner drives it for the session, and `scripts/duck-sim down`
stops everything by pidfile. The runner talks to robotd's and tofd's unix sockets directly: the
lane a Space would have through `mediad` is not in the path, so there is no 400-a-minute budget.

**Ground truth comes from the body server**, on a second connection to its port: the trunk's true
position and tilt at 10 Hz. That is what counts edge crossings (the trunk's centre over a drop, as
`practice.py` counts them), falls (trunk tilted past 45 degrees), distance, floor covered and time
moving. The brain never sees any of it.

Modes:
- `wander`: the W0 brain and runner exactly as shipped on feat/wander-w0.
- `wander-mount`: the same brain, with the floor geometry taken from where robotd's own kinematics
  put the depth sensor (`robot.state` frames.tof) instead of senses.py's level head at 0.21 m. A
  diagnostic, not the shipped brain; see w1-results.md.
- `straight`: walks straight ahead at the wander speed and never turns. The coverage baseline.
- `nothing`: zero twist every tick. The floor.

No scripted oracle exists for wander: there is no task to finish, so no expert to script.

Session logs go to ~/scratch/wander-w1/logs (not to git): one W0 session log (venue "duck-sim")
and one `.json` of metrics per session.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import replay  # noqa: E402
import senses as senses_mod  # noqa: E402
import session_log  # noqa: E402
import sim_rooms  # noqa: E402
from brain import Command  # noqa: E402
from runner import TICK_HZ, Runner  # noqa: E402
from senses import LiveSenses  # noqa: E402

HOME = Path.home()
SCRATCH = Path(os.environ.get("W1_SCRATCH", HOME / "scratch/wander-w1"))
STATE = SCRATCH / "st"
LOGS = SCRATCH / "logs"
SCENES = SCRATCH / "scenes"
MICRODUCK = Path(os.environ.get("MICRODUCK", HOME / "scratch/ducksim/microduck"))
RL = Path(os.environ.get("MICRODUCK_RL", HOME / "scratch/ducksim/microduck_rl"))
MODEL_DIR = RL / "src/mjlab_microduck/robot/microduck"
BODY_PORT = 7801

TRAIN = (1, 2, 3)
HOLDOUT = (101, 102)
FAST = 0.3                # m/s: above the sim walker's dead band (see w1-results.md)
SECONDS = {"wander-0.12": 45, "wander": 50, "wander-mount": 50, "straight": 30, "nothing": 30}
SPEED = {"wander-0.12": 0.12, "wander": FAST, "wander-mount": FAST, "straight": FAST, "nothing": FAST}
SEEDS = {"wander-0.12": (1, 101), "wander": TRAIN + HOLDOUT, "wander-mount": TRAIN + HOLDOUT,
         "straight": TRAIN + HOLDOUT, "nothing": (1, 101)}
MODES = ("wander-0.12", "wander", "wander-mount", "straight", "nothing")

TILT_FALL = -0.7          # gravity z in the trunk frame above this: tilted past 45 degrees, fallen
MOVING_MPS = 0.03         # ground-truth speed over half a second above this counts as moving
CELL = 0.1                # floor covered, in cells this size
RECOVER_M = 0.15          # a stuck event is recovered if the duck gets this far from it
RECOVER_S = 10.0          # within this long


# -- the lane: robotd and tofd over their unix sockets ---------------------------------------------
class Conn:
    """One JSON-RPC connection: calls wait for their answer, notifications are kept as latest."""

    def __init__(self, path: Path, latest: dict, on_note=None):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.settimeout(10)
        self.sock.connect(str(path))
        self.sock.settimeout(None)
        self.rf = self.sock.makefile("r")
        self.wlock = threading.Lock()
        self.latest = latest
        self.on_note = on_note
        self.pending: dict[int, list] = {}
        self.next_id = 1
        self.closed = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        try:
            for line in self.rf:
                msg = json.loads(line)
                if "id" in msg and ("result" in msg or "error" in msg):
                    slot = self.pending.pop(msg["id"], None)
                    if slot is not None:
                        slot[1] = msg
                        slot[0].set()
                elif "method" in msg:
                    self.latest[msg["method"]] = (msg.get("params"), time.monotonic())
                    if self.on_note:
                        self.on_note(msg["method"], msg.get("params"))
        except (OSError, ValueError):
            pass
        self.closed = True

    def _send(self, obj: dict) -> bool:
        data = (json.dumps(obj) + "\n").encode()
        try:
            with self.wlock:
                self.sock.sendall(data)
            return True
        except OSError:
            return False

    def call(self, method: str, params: dict | None, timeout: float = 5.0) -> Any:
        with self.wlock:
            i = self.next_id
            self.next_id += 1
        slot = [threading.Event(), None]
        self.pending[i] = slot
        if not self._send({"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}):
            raise RuntimeError(f"{method}: the socket is closed")
        if not slot[0].wait(timeout):
            raise RuntimeError(f"{method}: no answer in {timeout} s")
        if "error" in slot[1]:
            raise RuntimeError(f"{method}: {slot[1]['error']}")
        return slot[1]["result"]

    def notify(self, method: str, params: dict) -> bool:
        return self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class SimLane:
    """The runner's Lane over duck-sim's sockets: robot.state on its own connection (a subscribed
    connection is a stream), commands on a second, depth from tofd on a third."""

    def __init__(self, robot_sock: Path, tof_sock: Path):
        self.latest: dict[str, tuple[Any, float]] = {}
        self.states: list[tuple[float, dict]] = []
        self.state = Conn(robot_sock, self.latest, self._on_state)
        self.cmd = Conn(robot_sock, self.latest)
        self.tof = Conn(tof_sock, self.latest)
        self.sent: dict[str, int] = {}
        self.failed: dict[str, int] = {}
        self.calls: list[tuple[str, Any]] = []
        self.last_move_at: float | None = None

    def _on_state(self, method: str, params: Any) -> None:
        if method == "robot.state" and isinstance(params, dict):
            mv = params.get("move") or {}
            self.states.append((time.monotonic(), {"move": mv, "policy": params.get("policy"),
                                                   "safety": params.get("safety"),
                                                   "tof": (params.get("frames") or {}).get("tof"),
                                                   "odom": params.get("odom")}))

    def call(self, method: str, params: dict | None = None) -> Any:
        conn = self.state if method == "robot.subscribe" else self.tof if method.startswith("tof.") else self.cmd
        result = conn.call(method, params)
        self.calls.append((method, result))
        return result

    def notify(self, method: str, params: dict) -> bool:
        ok = self.cmd.notify(method, params)
        (self.sent if ok else self.failed)[method] = (self.sent if ok else self.failed).get(method, 0) + 1
        if method == "robot.move":
            self.last_move_at = time.monotonic()
        return ok

    def latest_of(self, method: str):
        return self.latest.get(method, (None, None))

    def close(self) -> None:
        for c in (self.state, self.cmd, self.tof):
            c.close()


class LaneView:
    """What the Runner sees: call, notify, latest."""

    def __init__(self, lane: SimLane):
        self.lane = lane

    def call(self, method, params=None):
        return self.lane.call(method, params)

    def notify(self, method, params):
        return self.lane.notify(method, params)

    def latest(self, method):
        return self.lane.latest_of(method)


# -- the baselines and the mount variant -------------------------------------------------------------
class Fixed:
    """A brain that always says the same thing: `straight` walks, `nothing` stands."""

    def __init__(self, name: str, seed: int, vx: float):
        self.name, self.seed, self.vx = name, seed, vx
        self.emotions: dict = {}
        self.state = "Wander" if vx else "Chill"
        self.since = 0.0
        self.energy = 0.0
        self.turn_dir = 1.0
        self.reason = None
        self.target = None

    def tick(self, s, dt_s) -> Command:
        if s.fallen:
            return Command(state="Fallen")
        return Command(twist=(self.vx, 0.0, 0.0), state=self.state)


def quat_pitch(q: list[float]) -> float:
    """Pitch (rad, positive nose down) of a w, x, y, z quaternion."""
    w, x, y, z = q
    return math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))


class MountSenses(LiveSenses):
    """LiveSenses whose pitch is the sensor's actual pitch from robot.state, not the head command.

    The runner sets `pitch` once at the top of every tick; that is when the latest measured pitch
    is taken, so one tick reads one pitch throughout and the log's `pitch` is the one it used."""

    lane: SimLane | None = None

    @property
    def pitch(self) -> float:  # type: ignore[override]
        return self._pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        st, _ = self.lane.latest_of("robot.state") if self.lane else (None, None)
        tof = ((st or {}).get("frames") or {}).get("tof")
        self._pitch = quat_pitch(tof["quat"]) if tof else value


# -- ground truth from the body server ----------------------------------------------------------------
class Observer:
    def __init__(self, port: int = BODY_PORT, hz: float = 10.0):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.f = self.sock.makefile("rw")
        self._rpc({"op": "hello", "protocol": 1, "joints": 15})
        self.samples: list[tuple[float, float, list[float], float]] = []   # mono, sim_time, trunk, gz
        self.hz = hz
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _rpc(self, obj: dict) -> dict:
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()
        return json.loads(self.f.readline())

    def read(self) -> tuple[float, float, list[float], float]:
        r = self._rpc({"op": "read"})
        return time.monotonic(), r["sim_time"], r["trunk"], r["imu"]["gravity"][2]

    def start(self) -> None:
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self.read())
            except (OSError, ValueError, KeyError):
                return
            self._stop.wait(1.0 / self.hz)

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)
        try:
            self.sock.close()
        except OSError:
            pass


# -- duck-sim -------------------------------------------------------------------------------------------
def duck_sim(cmd: str, scene: Path | None, out: Path, timeout: float = 300) -> str:
    env = dict(os.environ, DUCK_SIM_STATE=str(STATE), DUCK_SIM_RL=str(RL), DUCK_SIM_VIEWER="0")
    if scene is not None:
        env["DUCK_SIM_SCENE"] = str(scene)
    with out.open("a") as f:
        f.write(f"\n### duck-sim {cmd} {scene or ''} {dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.flush()
        # `duck-sim-v8` is duck-sim with one change: robotd's walk policy is the seeded v8 set's
        # velstand, as a robot resolves it, instead of alpha_walking + alpha_stand (which take no
        # steps below about 0.25 m/s). `down` is the same in both.
        script = "scripts/duck-sim-v8" if cmd == "up" else "scripts/duck-sim"
        p = subprocess.run(["sh", script, cmd], cwd=MICRODUCK, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
        f.write(p.stdout)
    return p.stdout


def body_listening() -> bool:
    s = socket.socket()
    s.settimeout(0.3)
    try:
        return s.connect_ex(("127.0.0.1", BODY_PORT)) == 0
    finally:
        s.close()


# -- one session ------------------------------------------------------------------------------------------
def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def run_session(room: str, mode: str, seed: int, seconds: float, simlog: Path) -> dict:
    start = sim_rooms.start_for(room, seed)
    scene = sim_rooms.write_scene(start, SCENES, MODEL_DIR)
    name = f"{room}-{mode}-s{seed}"
    rec: dict[str, Any] = {"room": room, "mode": mode, "seed": seed, "max_speed": SPEED[mode], "split": "holdout" if seed >= 100 else "train",
                           "seconds": seconds, "start": [start.x, start.y, start.yaw],
                           "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    t_up = time.monotonic()
    up = duck_sim("up", scene, simlog)
    rec["up_s"] = round(time.monotonic() - t_up, 1)
    rec["stood"] = "standing (trunk" in up
    if not rec["stood"]:
        rec["error"] = "the duck did not stand: " + " | ".join(l for l in up.splitlines() if "stand" in l)[-300:]
        return rec

    lane = SimLane(STATE / "duck-a.sock", STATE / "duck-a-tof.sock")
    obs = Observer()
    head_z_default = senses_mod.HEAD_Z_M
    logf = (LOGS / f"{name}.jsonl").open("w")
    try:
        writer = session_log.Writer(logf)
        r = Runner(LaneView(lane), max_speed=SPEED[mode], seed=seed, log=writer, venue="duck-sim", host=socket.gethostname(),
                   room=room, world_seed=seed,
                   started_utc=rec["utc"])
        if mode in ("straight", "nothing"):
            r.brain = Fixed(mode, seed, 0.45 if mode == "straight" else 0.0)
        if mode == "wander-mount":
            ms = MountSenses()
            ms.lane = lane
            r.senses = ms
        status = r.start(limit_s=seconds + 5, thread=False)
        if status != "running":
            rec["error"] = f"runner did not start: {status}"
            return rec
        if mode == "wander-mount":
            # Wait for a state with the sensor's frame, then set the floor height from it.
            for _ in range(40):
                st, _ = lane.latest_of("robot.state")
                if st and (st.get("frames") or {}).get("tof"):
                    break
                time.sleep(0.1)
            st, _ = lane.latest_of("robot.state")
            senses_mod.HEAD_Z_M = float(st["odom"]["position"][2]) + float(st["frames"]["tof"]["pos"][2])
            rec["head_z"] = senses_mod.HEAD_Z_M          # exact: replay needs the same float
            rec["mount_pitch_deg"] = round(math.degrees(quat_pitch(st["frames"]["tof"]["quat"])), 1)
        obs.start()
        period = 1.0 / TICK_HZ
        t0 = time.monotonic()
        compute, sense_to_act, stuck_at = [], [], []
        stuck_before = 0
        while time.monotonic() - t0 < seconds:
            ts = time.monotonic()
            _, tof_at = lane.latest_of("tof.frame")
            r.tick(ts)
            if lane.last_move_at is not None and lane.last_move_at >= ts:
                compute.append(lane.last_move_at - ts)
                if tof_at is not None:
                    sense_to_act.append(lane.last_move_at - tof_at)
            if r.stuck_events > stuck_before:
                stuck_at.append(ts)
                stuck_before = r.stuck_events
            time.sleep(max(0.0, period - (time.monotonic() - ts)))
        elapsed = time.monotonic() - t0
        r.stop("session ended")
        time.sleep(0.5)
    finally:
        obs.stop()
        lane.close()
        logf.close()
        senses_mod.HEAD_Z_M = head_z_default

    s = r.status
    rec.update(elapsed_s=round(elapsed, 1), ticks=s.ticks, walking_s=round(s.walking_ticks / TICK_HZ, 1),
               blind_ticks=s.blind_ticks, edge_stops=r.edge_stops, stuck_events=r.stuck_events,
               moves_sent=lane.sent.get("robot.move", 0), moves_failed=lane.failed.get("robot.move", 0),
               heads_sent=lane.sent.get("robot.head", 0), emotes=list(s.emotes))
    rec["moves_per_s"] = round(rec["moves_sent"] / elapsed, 2)
    rec["latency_ms"] = {"compute_p50": _ms(pct(compute, .5)), "compute_p95": _ms(pct(compute, .95)),
                         "sense_to_act_p50": _ms(pct(sense_to_act, .5)), "sense_to_act_p95": _ms(pct(sense_to_act, .95))}
    # What robotd did with the twists: the share of states, while walking was asked for, in which the
    # deadman had cut the twist (a command did not arrive in time).
    asked = [m for _, m in lane.states if any(abs(v) > 1e-6 for v in (m["move"].get("requested") or [0]))]
    rec["states_walking"] = len(asked)
    rec["robotd_fallen_states"] = sum(1 for _, m in lane.states if (m.get("safety") or {}).get("fallen"))
    rec["deadman_states"] = sum(1 for _, m in lane.states if "deadman" in (m["move"].get("limited_by") or [])
                                and any(abs(v) > 1e-6 for v in (m["move"].get("requested") or [0])))
    rec.update(truth(start, room, obs.samples, stuck_at, t0))
    rec["replay_mismatches"] = replay_check(LOGS / f"{name}.jsonl", rec.get("head_z")) if mode.startswith("wander") else None
    rec["states"] = len(lane.states)
    return rec


def _ms(x: float | None) -> float | None:
    return None if x is None else round(1000 * x, 1)


def truth(start: sim_rooms.Start, room_name: str, samples: list, stuck_at: list[float], t0: float) -> dict:
    room = sim_rooms.room_of(room_name)
    samples = [s for s in samples if s[0] >= t0]
    if not samples:
        return {"truth_samples": 0}
    crossings, over, falls, fallen = 0, False, 0, False
    cells: set[tuple[int, int]] = set()
    closest = math.inf
    for _, _, trunk, gz in samples:
        rx, ry = start.to_room(trunk[0], trunk[1])
        is_over = room.drop_at(rx, ry) > 0
        if is_over and not over:
            crossings += 1
        over = is_over
        tipped = gz > TILT_FALL
        if tipped and not fallen:
            falls += 1
        fallen = tipped
        closest = min(closest, room.edge_margin(rx, ry))
        cells.add((math.floor(rx / CELL), math.floor(ry / CELL)))
    # Distance and moving time on half-second steps, so the gait's sway does not add up as travel.
    step = [samples[0]]
    for smp in samples[1:]:
        if smp[0] - step[-1][0] >= 0.5:
            step.append(smp)
    dist, moving = 0.0, 0.0
    for a, b in zip(step, step[1:]):
        d = math.hypot(b[2][0] - a[2][0], b[2][1] - a[2][1])
        dist += d
        if d / (b[0] - a[0]) >= MOVING_MPS:
            moving += b[0] - a[0]
    span = samples[-1][0] - samples[0][0]
    recovered = 0
    for ts in stuck_at:
        here = min(samples, key=lambda smp: abs(smp[0] - ts))[2]
        if any(abs(smp[0] - ts) <= RECOVER_S and smp[0] > ts
               and math.hypot(smp[2][0] - here[0], smp[2][1] - here[1]) >= RECOVER_M for smp in samples):
            recovered += 1
    sim_span = samples[-1][1] - samples[0][1]
    return {"truth_samples": len(samples), "edge_crossings": crossings, "falls": falls,
            "closest_edge_m": None if closest == math.inf else round(closest, 3),
            "distance_m": round(dist, 2), "floor_m2": round(len(cells) * CELL * CELL, 2),
            "moving_share": round(moving / span, 3) if span else 0.0, "stuck_recovered": recovered,
            "realtime": round(sim_span / span, 3) if span else None}


def replay_check(path: Path, head_z: float | None) -> int | None:
    """The W0 replay over a duck-sim log: brain mismatches plus full mismatches (0 means it replays)."""
    default = senses_mod.HEAD_Z_M
    try:
        if head_z is not None:
            senses_mod.HEAD_Z_M = head_z
        header, ticks = session_log.read(path.read_text().splitlines())
        return len(replay.replay_brain(header, ticks)) + len(replay.replay_full(header, ticks))
    except Exception as e:  # noqa: BLE001
        print(f"replay of {path.name} failed: {e}", flush=True)
        return None
    finally:
        senses_mod.HEAD_Z_M = default


# -- tables ------------------------------------------------------------------------------------------------
def load_sessions() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(LOGS.glob("*.json"))]


def aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("truth_samples")]
    n = len(ok)
    if not n:
        return {"n": 0, "not_run": len(rows)}

    def tot(k):
        return sum(r.get(k) or 0 for r in ok)

    lat = lambda k: round(statistics.median([r["latency_ms"][k] for r in ok if r["latency_ms"][k] is not None] or [0]), 1)
    secs = sum(r["elapsed_s"] for r in ok)
    return {
        "n": n, "did_not_stand": len(rows) - n,
        "no_crossing": f"{sum(1 for r in ok if r['edge_crossings'] == 0)}/{n}",
        "edge_crossings": tot("edge_crossings"), "falls": tot("falls"),
        "edge_stops": tot("edge_stops"), "stuck_events": tot("stuck_events"), "stuck_recovered": tot("stuck_recovered"),
        "distance_m": round(tot("distance_m") / n, 2), "floor_m2": round(tot("floor_m2") / n, 2),
        "moving_share": round(sum(r["moving_share"] for r in ok) / n, 3),
        "walking_share": round(tot("walking_s") / secs, 3) if secs else 0.0,
        "moves_per_s": round(tot("moves_sent") / secs, 2) if secs else 0.0,
        "deadman_share": round(tot("deadman_states") / max(1, tot("states_walking")), 3),
        "closest_edge_m": min((r["closest_edge_m"] for r in ok if r.get("closest_edge_m") is not None), default=None),
        "compute_p50_ms": lat("compute_p50"), "compute_p95_ms": lat("compute_p95"),
        "sense_to_act_p50_ms": lat("sense_to_act_p50"), "sense_to_act_p95_ms": lat("sense_to_act_p95"),
        "sim_seconds": round(secs, 1),
        "replay_mismatches": None if ok[0]["replay_mismatches"] is None else sum(r["replay_mismatches"] or 0 for r in ok),
        "realtime_min": min((r.get("realtime") or 1) for r in ok),
    }


def tables(out_md: Path, out_json: Path) -> None:
    sessions = load_sessions()
    rooms = list(sim_rooms.SIM_ROOMS)
    modes = [m for m in MODES if any(s["mode"] == m for s in sessions)]
    rows = []
    for room in rooms + ["all"]:
        for mode in modes:
            for split in ("train", "holdout", "all"):
                sel = [s for s in sessions if s["mode"] == mode and (room == "all" or s["room"] == room)
                       and (split == "all" or s["split"] == split)]
                if sel:
                    rows.append({"room": room, "mode": mode, "split": split, **aggregate(sel)})
    out_json.write_text(json.dumps({"task": "W1 wander", "venue": "duck-sim", "host": socket.gethostname(),
                                    "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                    "rows": rows, "sessions": sessions}, indent=1))
    print(f"wrote {out_json} ({len(rows)} rows, {len(sessions)} sessions)")
    print(md_table(rows))


def md_table(rows: list[dict]) -> str:
    """The results as a markdown table: one row per room, mode and split."""
    cols = [("room", "room"), ("mode", "mode"), ("split", "seeds"), ("n", "n"), ("no_crossing", "no crossing"),
            ("edge_crossings", "edge crossings"), ("falls", "falls"), ("edge_stops", "edge stops"),
            ("stuck_events", "stuck"), ("stuck_recovered", "recovered"), ("distance_m", "distance m"),
            ("floor_m2", "floor m2"), ("moving_share", "moving"), ("walking_share", "walk asked"),
            ("moves_per_s", "moves/s"), ("compute_p50_ms", "decide p50 ms"), ("compute_p95_ms", "decide p95 ms"),
            ("sense_to_act_p50_ms", "sense-act p50 ms"), ("sense_to_act_p95_ms", "sense-act p95 ms")]
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(out)


# -- main ----------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "tables"))
    ap.add_argument("--rooms", default=",".join(sim_rooms.SIM_ROOMS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--seeds", default=None, help="comma list; default is each mode's own set")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    out_json = HERE / "w1-results.json"
    if a.what == "tables":
        tables(HERE / "w1-results.md", out_json)
        return 0
    simlog = SCRATCH / "duck-sim.log"
    t_all = time.monotonic()
    try:
        for mode in a.modes.split(","):
            for room in a.rooms.split(","):
                seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else SEEDS[mode]
                for seed in seeds:
                    path = LOGS / f"{room}-{mode}-s{seed}.json"
                    if a.skip_done and path.exists():
                        continue
                    secs = a.seconds or SECONDS[mode]
                    try:
                        rec = run_session(room, mode, seed, secs, simlog)
                    except Exception as e:  # noqa: BLE001 - one bad session must not lose the rest
                        rec = {"room": room, "mode": mode, "seed": seed, "error": repr(e),
                               "split": "holdout" if seed >= 100 else "train"}
                    finally:
                        duck_sim("down", None, simlog, timeout=120)
                    path.write_text(json.dumps(rec, indent=1))
                    print(f"{time.monotonic() - t_all:7.0f}s {room:9s} {mode:12s} s{seed:<3d} "
                          + (f"error {rec['error']}" if "error" in rec else
                             f"cross {rec.get('edge_crossings')} falls {rec.get('falls')} dist {rec.get('distance_m')} m "
                             f"floor {rec.get('floor_m2')} m2 walk {rec.get('walking_s')} s edge-stops {rec.get('edge_stops')} "
                             f"stuck {rec.get('stuck_events')}/{rec.get('stuck_recovered')} moves/s {rec.get('moves_per_s')} "
                             f"lat {rec.get('latency_ms')} rt {rec.get('realtime')} replay {rec.get('replay_mismatches')}"),
                          flush=True)
    finally:
        duck_sim("down", None, simlog, timeout=120)
        print(f"duck-sim down; body still listening: {body_listening()}; total {time.monotonic() - t_all:.0f} s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
