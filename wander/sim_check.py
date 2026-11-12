"""Drive the practice duck with the runner on a fake clock and report what happened."""
import math, sys
from practice import PracticeDuck
from runner import Runner, TICK_HZ

def run(seconds=180, budget=400, seed=1):
    t = [1000.0]
    clock = lambda: t[0]
    duck = PracticeDuck(seed=seed, budget=budget, clock=clock, auto=False)
    r = Runner(duck, clock=clock)
    r._emote = lambda name: (r.status.emotes.append(name), duck.notes.append(f"played {name}"))
    assert r.start(limit_s=10_000) == "running", r.status
    r._stop.set()                       # we tick by hand; the thread exits at once
    r._thread.join()
    next_tick, travelled, last = t[0], 0.0, (duck.x, duck.y)
    closest_cat = 9
    while t[0] < 1000 + seconds:
        duck.step(t[0])
        if t[0] >= next_tick:
            r.tick(t[0]); next_tick += 1 / TICK_HZ
        travelled += math.hypot(duck.x - last[0], duck.y - last[1]); last = (duck.x, duck.y)
        closest_cat = min(closest_cat, math.hypot(duck.x - duck.cat[0], duck.y - duck.cat[1]))
        t[0] += 0.02
    s = r.status
    print(f"budget={budget} seed={seed}: ticks {s.ticks}, walking {s.walking_ticks}, blind {s.blind_ticks}, "
          f"greetings {s.greetings}, emotes {s.emotes[:6]}, travelled {travelled:.1f} m, "
          f"fallen {duck.fallen}, dropped {duck.dropped}, closest to cat {closest_cat:.2f} m, "
          f"notes {[n for n in duck.notes if 'bump' in n]}")
    return s, duck

if __name__ == "__main__":
    for b in (400, None):
        for seed in range(1, 9):
            s, d = run(budget=b, seed=seed)
    print("\n".join(s.log[-12:]))
