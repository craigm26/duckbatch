"""How many envs fit, and how fast: run once per machine before choosing `num_envs` in a menu.

Each env count runs in a fresh subprocess (MuJoCo-Warp's allocations are not all returned on
`env.close()`), so an out-of-memory at one size does not poison the next.
"""

from __future__ import annotations

import json
import subprocess
import sys

_CHILD = r"""
import json, sys, time, torch
from duckbatch.sim import make_env, DEFAULT_TASK
task = sys.argv[1] or DEFAULT_TASK
n = int(sys.argv[2])
try:
    env = make_env(task, n)
    env.reset()
    a = torch.zeros(n, env.action_manager.total_action_dim, device="cuda:0")
    for _ in range(5):
        env.step(a)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(48):
        env.step(a)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    free, total = torch.cuda.mem_get_info()
    print("RESULT" + json.dumps({"envs": n, "env_steps_per_s": round(n * 48 / dt),
          "vram_used_mib": round((total - free) / 2**20), "vram_total_mib": round(total / 2**20)}))
except Exception as e:
    print("RESULT" + json.dumps({"envs": n, "error": f"{type(e).__name__}: {str(e)[:160]}"}))
"""


def probe(task: str | None, sizes: list[int]) -> list[dict]:
    rows = []
    for n in sizes:
        out = subprocess.run([sys.executable, "-c", _CHILD, task or "", str(n)],
                             capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
        row = json.loads(line[6:]) if line else {"envs": n, "error": out.stderr[-300:]}
        rows.append(row)
        if "error" in row:
            print(f"envs={n:>5}  FAILED  {row['error']}")
        else:
            print(f"envs={n:>5}  {row['env_steps_per_s']:>8,} env-steps/s  "
                  f"VRAM {row['vram_used_mib']:,}/{row['vram_total_mib']:,} MiB")
    return rows
