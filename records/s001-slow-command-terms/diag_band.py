"""What does each walker do at a slow command, and what does each reward term pay for it?

Scratch diagnostic for b003d's design (not part of duckbatch). CPU, small env count.
For each policy and command: mean forward speed, feet air-time stats and the per-term reward
(weight x value, per second, as mjlab's RewardManager keeps it) over the last half of the
episode, upright envs only. Plus the live standing-env share under the training command mix.
"""

import json
import sys
import time

import torch

torch.set_num_threads(2)
sys.path.insert(0, "/home/craigm26/projects/craigm26/duckbatch/src")
DB = "/home/craigm26/projects/craigm26/duckbatch"

from duckbatch.pairs import fixed_command  # noqa: E402
from duckbatch.policy import load_teacher  # noqa: E402
from duckbatch.sim import Population, make_env  # noqa: E402

TASK = "Mjlab-VelStand-Flat-MicroDuck"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
POLICIES = {
    "teacher": f"{DB}/teachers/velstand.onnx",
    "student": f"{DB}/records/b002-student-size-longer/policies/a02/policy.onnx",
    "b003b": "/home/craigm26/scratch/b003d/b003b_policy.onnx",
}
COMMANDS = [(0.10, 0.0, 0.0), (0.30, 0.0, 0.0), (0.0, 0.0, 0.0)]

env = make_env(TASK, N, device="cpu", seed=4001)
robot = env.scene["robot"]
rm = env.reward_manager
names = list(rm.active_terms)
feet = env.scene["feet_ground_contact"]
steps = int(round(SECONDS / env.step_dt))
prof = Population(env, POLICIES["teacher"], [], device="cpu")

# 1) The standing share the training command mix really runs with (curriculum at its final stage).
obs, _ = env.reset(seed=1)
for _ in range(3):
    env.command_manager.get_term("twist")._resample(torch.arange(N))
    env.reset(seed=2)
tw = env.command_manager.get_term("twist")
print(f"[mix] live cfg rel_standing_envs={tw.cfg.rel_standing_envs} "
      f"rel_turn_in_place_envs={tw.cfg.rel_turn_in_place_envs} rel_forward_envs={tw.cfg.rel_forward_envs}",
      flush=True)

rows = []
for pname, path in POLICIES.items():
    net = load_teacher(path, "cpu")
    for twist in COMMANDS:
        t0 = time.perf_counter()
        with prof.profile("walk"), fixed_command(env, twist):
            obs = env.reset(seed=4001)[0]["actor"]
            acc = torch.zeros(len(names))
            n_up = 0.0
            vx_sum = air_frac = in_range = swing_sum = swing_n = 0.0
            for t in range(steps):
                with torch.no_grad():
                    obs = env.step(net(obs))[0]["actor"]
                up = (robot.data.projected_gravity_b[:, 2] < -0.5).float()
                if t >= steps // 2:
                    acc += (rm._step_reward * up[:, None]).sum(0)
                    n_up += float(up.sum())
                    vx_sum += float((robot.data.root_link_lin_vel_b[:, 0] * up).sum())
                    cur = feet.data.current_air_time  # [N, 2]
                    air_frac += float(((cur > 0).float().mean(1) * up).sum())
                    in_range += float((((cur > 0.125) & (cur < 0.3)).float().mean(1) * up).sum())
                    last = feet.data.last_air_time
                    landed = (feet.data.current_contact_time > 0) & (feet.data.current_contact_time <= env.step_dt + 1e-6)
                    if landed.any():
                        swing_sum += float(last[landed].sum())
                        swing_n += float(landed.sum())
        n = max(n_up, 1.0)
        row = {"policy": pname, "cmd": twist, "vx": vx_sum / n, "air_frac": air_frac / n,
               "air_in_window": in_range / n, "mean_swing_s": swing_sum / max(swing_n, 1),
               "steps_landed_per_s": swing_n / max(n_up, 1) / env.step_dt,
               "terms": {k: round(float(v) / n, 4) for k, v in zip(names, acc)}}
        rows.append(row)
        top = sorted(row["terms"].items(), key=lambda kv: -abs(kv[1]))[:9]
        print(f"[diag] {pname:8s} cmd {twist} vx {row['vx']:+.3f} air {row['air_frac']:.2f} "
              f"inwin {row['air_in_window']:.2f} swing {row['mean_swing_s']:.3f}s "
              f"landings/s/env {row['steps_landed_per_s']:.2f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        print("        " + "  ".join(f"{k}={v:+.3f}" for k, v in top), flush=True)

json.dump({"N": N, "seconds": SECONDS, "terms": names, "rows": rows},
          open("/home/craigm26/scratch/b003d/diag_band.json", "w"), indent=1)
print("[diag] wrote diag_band.json", flush=True)
