"""b003d's 250-iteration checkpoints at slow commands, beside the student and b003b (CPU).

Pre-registered in notes/2026-10-01-b003d-design.md: "Before blaming the reward again, I'll read
the 250-iteration checkpoints." Each checkpoint's actor is rebuilt exactly as finetune's
export_actor builds the ONNX (normaliser std + eps 0.01). Measured like speed_curve: walk profile,
fixed command, upright envs, last half of the episode. Two slow commands: (0.10, 0, 0) is the
pass line's; (0.10, 0, 0.5) is the kind b003d trained on (slow planar with yaw, VelStand's box).
"""

import json
import sys
import time

import torch

torch.set_num_threads(2)
sys.path.insert(0, "/home/craigm26/projects/craigm26/duckbatch/src")
DB = "/home/craigm26/projects/craigm26/duckbatch"
S = "/home/craigm26/scratch/b003d"

from duckbatch.pairs import fixed_command  # noqa: E402
from duckbatch.policy import build_mlp, load_teacher  # noqa: E402
from duckbatch.sim import Population, make_env  # noqa: E402


def from_checkpoint(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
    net = build_mlp([128, 128])
    with torch.no_grad():
        net.obs_mean.copy_(sd["obs_normalizer._mean"])
        net.obs_std.copy_(sd["obs_normalizer._std"] + 0.01)
        for i in (0, 2, 4):
            net.mlp[i].weight.copy_(sd[f"mlp.{i}.weight"])
            net.mlp[i].bias.copy_(sd[f"mlp.{i}.bias"])
    return net.eval()


# The rebuilt 1499 checkpoint must compute what the job's exported ONNX computes.
x = torch.randn(256, 61)
gap = (from_checkpoint(f"{S}/b003d/ckpt/model_1499.pt")(x) - load_teacher(f"{S}/b003d/policy.onnx")(x)).abs().max()
print(f"[check] model_1499.pt vs exported policy.onnx: max |diff| {gap:.2e}", flush=True)
assert gap < 1e-4

POLICIES = {"student": load_teacher(f"{DB}/records/b002-student-size-longer/policies/a02/policy.onnx"),
            "b003b": load_teacher(f"{S}/b003b_policy.onnx")}
for it in (250, 500, 750, 1000, 1250, 1499):
    POLICIES[f"b003d@{it}"] = from_checkpoint(f"{S}/b003d/ckpt/model_{it}.pt")
COMMANDS = [(0.10, 0.0, 0.0), (0.10, 0.0, 0.5), (0.30, 0.0, 0.0)]
N, SECONDS = 24, 6.0

env = make_env("Mjlab-VelStand-Flat-MicroDuck", N, device="cpu", seed=4001)
robot = env.scene["robot"]
feet = env.scene["feet_ground_contact"]
steps = int(round(SECONDS / env.step_dt))
prof = Population(env, f"{DB}/teachers/velstand.onnx", [], device="cpu")
rows = []
for pname, net in POLICIES.items():
    for twist in COMMANDS:
        t0 = time.perf_counter()
        with prof.profile("walk"), fixed_command(env, twist):
            obs = env.reset(seed=4001)[0]["actor"]
            vx = wz = air = n_up = 0.0
            fell = torch.zeros(N)
            for t in range(steps):
                with torch.no_grad():
                    obs = env.step(net(obs))[0]["actor"]
                down = robot.data.projected_gravity_b[:, 2] > -0.5
                fell = torch.maximum(fell, down.float())
                if t >= steps // 2:
                    up = (~down).float()
                    vx += float((robot.data.root_link_lin_vel_b[:, 0] * up).sum())
                    wz += float((robot.data.root_link_ang_vel_b[:, 2] * up).sum())
                    air += float(((feet.data.current_air_time > 0).float().mean(1) * up).sum())
                    n_up += float(up.sum())
        n = max(n_up, 1.0)
        row = {"policy": pname, "cmd": twist, "vx": vx / n, "wz": wz / n, "air_frac": air / n,
               "fell_share": float(fell.mean())}
        rows.append(row)
        print(f"[ckpt] {pname:11s} cmd {twist} vx {row['vx']:+.3f} wz {row['wz']:+.3f} "
              f"air {row['air_frac']:.2f} fell {row['fell_share']:.0%} ({time.perf_counter() - t0:.0f}s)", flush=True)
json.dump({"N": N, "seconds": SECONDS, "rows": rows}, open(f"{S}/ckpt_band.json", "w"), indent=1)
print("[ckpt] wrote ckpt_band.json", flush=True)
