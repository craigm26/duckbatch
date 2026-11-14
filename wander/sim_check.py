"""Drive the practice duck with the runner on a fake clock and report what happened.

    python3 sim_check.py             # 3-minute sessions in every room, with and without the lane limit
    python3 sim_check.py --golden    # rewrite the golden traces in tests/golden/
    python3 sim_check.py --scan      # which brain seeds meet a visitor, an edge or the sill early
"""
import math
import sys
from pathlib import Path

import session_log
from practice import ROOMS, PracticeDuck
from runner import Runner, TICK_HZ

GOLDEN = Path(__file__).with_name("tests") / "golden"
# (room, world seed, brain seed, seconds): short, so the fixtures stay small. The brain seeds are
# ones whose first walk meets the thing the room is for: a visitor, the table edge, the stair, the
# sill (sim_check.py --scan lists them).
GOLDEN_RUNS = (("living", 1, 11, 20), ("table", 1, 10, 15), ("stair", 1, 4, 15), ("threshold", 1, 4, 15))


def run(seconds=180, budget=400, seed=1, room="living", log=None, quiet=False, t0=1000.0, brain_seed=7):
    k = [0]                             # steps of 20 ms, so the clock does not drift by adding
    clock = lambda: t0 + k[0] / 50
    duck = PracticeDuck(seed=seed, budget=budget, clock=clock, auto=False, room=room)
    r = Runner(duck, clock=clock, seed=brain_seed, log=log, venue="practice", room=room, world_seed=seed)
    r._emote = lambda name: (r.status.emotes.append(name), duck.notes.append(f"played {name}"))
    assert r.start(limit_s=10_000, thread=False) == "running", r.status
    next_tick = 0.0                     # in steps: 12.5 to a tick, so 12 and 13 in turn
    travelled, last = 0.0, (duck.x, duck.y)
    closest_cat = 9
    while k[0] < seconds * 50:
        duck.step(clock())
        if k[0] >= next_tick:
            r.tick(clock()); next_tick += 50 / TICK_HZ
        travelled += math.hypot(duck.x - last[0], duck.y - last[1]); last = (duck.x, duck.y)
        closest_cat = min(closest_cat, math.hypot(duck.x - duck.cat[0], duck.y - duck.cat[1]))
        k[0] += 1
    s = r.status
    if not quiet:
        print(f"{room} budget={budget} seed={seed}: walking {s.walking_ticks}/{s.ticks}, blind {s.blind_ticks}, "
              f"greetings {s.greetings}, travelled {travelled:.1f} m, fallen {duck.fallen}, "
              f"edge crossings {duck.edge_crossings}, closest edge {duck.closest_edge:.2f} m, "
              f"edge stops {r.edge_stops}, stuck {r.stuck_events}, closest to cat {closest_cat:.2f} m")
    duck.travelled = travelled
    return s, duck, r


def golden() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for room, seed, brain_seed, seconds in GOLDEN_RUNS:
        path = GOLDEN / f"{room}-brain{brain_seed}.jsonl"
        with path.open("w") as f:
            _, duck, r = run(seconds=seconds, budget=None, seed=seed, room=room, quiet=True, t0=0.0,
                             brain_seed=brain_seed, log=session_log.Writer(f))
        print(f"wrote {path.name}: {path.stat().st_size // 1024} KB, edge stops {r.edge_stops}, "
              f"stuck {r.stuck_events}, fallen {duck.fallen}")


def scan(seconds=30) -> None:
    for room in ROOMS:
        for brain_seed in range(1, 13):
            _, duck, r = run(seconds=seconds, budget=None, room=room, quiet=True, t0=0.0, brain_seed=brain_seed)
            print(f"{room} brain {brain_seed}: edge stops {r.edge_stops}, stuck {r.stuck_events}, "
                  f"greetings {r.status.greetings}, fallen {duck.fallen}")


if __name__ == "__main__":
    if "--golden" in sys.argv:
        golden()
        sys.exit(0)
    if "--scan" in sys.argv:
        scan()
        sys.exit(0)
    for room in ROOMS:
        for b in (400, None):
            for seed in range(1, 5):
                run(budget=b, seed=seed, room=room)
