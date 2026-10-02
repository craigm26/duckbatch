"""How much does a walking gait's body velocity wobble within a stride? (CPU, scratch)

mjlab's tracking error is on the instantaneous body velocity, so E|c - v|^2 = |c - mean v|^2 +
var(v): the stride's surge and sway count as error even when the mean speed is exact. Under b003d's
relative width at a 0.10 m/s command (width^2 = 0.1 * (0.1/0.4)^2 = 0.00625), var(v) alone costs
exp(-var / 0.00625). Measured per env over the last half of the episode, upright envs, walk profile.
"""

import json
import math
import sys

import torch

torch.set_num_threads(2)
sys.path.insert(0, "/home/craigm26/projects/craigm26/duckbatch/src")
DB = "/home/craigm26/projects/craigm26/duckbatch"
S = "/home/craigm26/scratch/b003d"

from duckbatch.pairs import fixed_command  # noqa: E402
from duckbatch.policy import load_teacher  # noqa: E402
from duckbatch.sim import Population, make_env  # noqa: E402

POLICIES = {"teacher": f"{DB}/teachers/velstand.onnx",
            "student": f"{DB}/records/b002-student-size-longer/policies/a02/policy.onnx",
            "b003b": f"{S}/b003b_policy.onnx"}
N, SECONDS = 24, 6.0
env = make_env("Mjlab-VelStand-Flat-MicroDuck", N, device="cpu", seed=4001)
robot = env.scene["robot"]
steps = int(round(SECONDS / env.step_dt))
win = int(round(0.5 / env.step_dt))  # a stride is ~0.5 s at 2.6-3.3 landings/s
prof = Population(env, POLICIES["teacher"], [], device="cpu")
rows = []
for pname, path in POLICIES.items():
    net = load_teacher(path, "cpu")
    for twist in [(0.30, 0.0, 0.0), (0.20, 0.0, 0.0)]:
        with prof.profile("walk"), fixed_command(env, twist):
            obs = env.reset(seed=4001)[0]["actor"]
            vs, ups = [], []
            for t in range(steps):
                with torch.no_grad():
                    obs = env.step(net(obs))[0]["actor"]
                if t >= steps // 2:
                    vs.append(robot.data.root_link_lin_vel_b[:, :2].clone())
                    ups.append(robot.data.projected_gravity_b[:, 2] < -0.5)
        v = torch.stack(vs)                      # [T, N, 2]
        ok = torch.stack(ups).all(0)             # envs upright for the whole half
        v = v[:, ok]
        mean = v.mean(0)                         # [n, 2]
        var = ((v - mean) ** 2).sum(-1).mean(0)  # per-env planar variance (m/s)^2
        smooth = v.unfold(0, win, 1).mean(-1)    # 0.5 s moving average
        var_s = ((smooth - smooth.mean(0)) ** 2).sum(-1).mean(0)
        # What the term itself would pay a walk with this stride at exactly its own mean speed:
        # the average over the stride of exp(-|v(t) - mean v|^2 / width^2), not exp(-var / width^2).
        dev = ((v - mean) ** 2).sum(-1)                       # [T, n]
        dev_s = ((smooth - smooth.mean(0)) ** 2).sum(-1)
        pay = {"rel_0p10_instant": float(torch.exp(-dev / 0.00625).mean()),
               "rel_0p10_avg0p5s": float(torch.exp(-dev_s / 0.00625).mean()),
               "mjlab_instant": float(torch.exp(-dev / 0.1).mean())}
        pay["mjlab_crossover_mps"] = math.sqrt(-0.1 * math.log(pay["mjlab_instant"]))
        # The same stride moved to a mean of f x a 0.10 m/s forward command (f = 0: stepping in
        # place; 1: walking it), judged against that command. Standing still pays exp(-1.6) at
        # b003d's width and exp(-0.1) at mjlab's.
        c = torch.tensor([0.10, 0.0])
        for f in (0.0, 0.5, 1.0):
            for name, trace, m in (("instant", v, mean), ("avg0p5s", smooth, smooth.mean(0))):
                e2 = ((c - (trace - m + f * c)) ** 2).sum(-1)
                pay[f"f{f:.1f}_{name}_rel"] = float(torch.exp(-e2 / 0.00625).mean())
                pay[f"f{f:.1f}_{name}_mjlab"] = float(torch.exp(-e2 / 0.1).mean())
        row = {"policy": pname, "cmd": twist, "envs": int(ok.sum()), "pays_at_own_mean": pay,
               "mean_vx": float(mean[:, 0].mean()), "std_planar": float(var.mean().sqrt()),
               "std_x": float(((v[..., 0] - mean[:, 0]) ** 2).mean().sqrt()),
               "std_y": float(((v[..., 1] - mean[:, 1]) ** 2).mean().sqrt()),
               "std_planar_0p5s_avg": float(var_s.mean().sqrt()),
               "relative_term_cost_at_0p10": math.exp(-float(var.mean()) / 0.00625)}
        rows.append(row)
        print(f"[wobble] {pname:8s} cmd {twist[0]:.2f}: mean vx {row['mean_vx']:+.3f}  stride wobble "
              f"{row['std_planar']:.3f} m/s (x {row['std_x']:.3f}, y {row['std_y']:.3f}); "
              f"0.5 s average {row['std_planar_0p5s_avg']:.3f}; exp(-var/0.00625) = "
              f"{row['relative_term_cost_at_0p10']:.2f}  [{row['envs']} envs]", flush=True)
        print(f"         pays at its own mean: relative@0.10 instant {pay['rel_0p10_instant']:.2f}, "
              f"0.5 s average {pay['rel_0p10_avg0p5s']:.2f}; mjlab instant {pay['mjlab_instant']:.3f} "
              f"(= standing at {pay['mjlab_crossover_mps']:.3f} m/s)", flush=True)
        for name in ("instant", "avg0p5s"):
            print(f"         at 0.10, {name:7s} f=0/0.5/1: relative " + " / ".join(
                f"{pay[f'f{f:.1f}_{name}_rel']:.2f}" for f in (0.0, 0.5, 1.0)) + "  (stand 0.20); mjlab " +
                " / ".join(f"{pay[f'f{f:.1f}_{name}_mjlab']:.3f}" for f in (0.0, 0.5, 1.0)) + "  (stand 0.905)",
                flush=True)
json.dump(rows, open(f"{S}/gait_wobble.json", "w"), indent=1)
print("[wobble] wrote gait_wobble.json", flush=True)
