"""Real VelStand env: the relative term next to mjlab's, both at weight 2, student policy."""

import sys

import torch

torch.set_num_threads(2)
sys.path.insert(0, "/home/craigm26/projects/craigm26/duckbatch/src")
DB = "/home/craigm26/projects/craigm26/duckbatch"

import mjlab.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import RewardTermCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from duckbatch.pairs import fixed_command  # noqa: E402
from duckbatch.policy import load_teacher  # noqa: E402
from duckbatch.rewards import track_linear_velocity_relative  # noqa: E402
from duckbatch.sim import FINAL_CURRICULUM_STEP, Population  # noqa: E402

N = 16
cfg = load_env_cfg("Mjlab-VelStand-Flat-MicroDuck")
cfg.scene.num_envs = N
cfg.seed = 4001
std = cfg.rewards["track_linear_velocity"].params["std"]
cfg.rewards["track_linear_velocity_relative"] = RewardTermCfg(
    func=track_linear_velocity_relative, weight=2.0,
    params={"command_name": "twist", "std": std, "full_width_speed": 0.4,
            "min_width_speed": 0.05, "stand_below": 0.02})
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
env.common_step_counter = FINAL_CURRICULUM_STEP
names = list(env.reward_manager.active_terms)
i_abs, i_rel = names.index("track_linear_velocity"), names.index("track_linear_velocity_relative")
robot = env.scene["robot"]
prof = Population(env, f"{DB}/teachers/velstand.onnx", [], device="cpu")
net = load_teacher(f"{DB}/records/b002-student-size-longer/policies/a02/policy.onnx", "cpu")
steps = int(round(4.0 / env.step_dt))
for twist in [(0.1, 0.0, 0.0), (0.4, 0.0, 0.0), (0.0, 0.0, 0.0)]:
    with prof.profile("walk"), fixed_command(env, twist):
        obs = env.reset(seed=4001)[0]["actor"]
        a = r = n = vx = 0.0
        for t in range(steps):
            with torch.no_grad():
                obs = env.step(net(obs))[0]["actor"]
            if t >= steps // 2:
                up = (robot.data.projected_gravity_b[:, 2] < -0.5).float()
                sr = env.reward_manager._step_reward
                a += float((sr[:, i_abs] * up).sum()); r += float((sr[:, i_rel] * up).sum())
                vx += float((robot.data.root_link_lin_vel_b[:, 0] * up).sum()); n += float(up.sum())
    n = max(n, 1.0)
    print(f"[check] student cmd {twist}: vx {vx / n:+.3f}  mjlab {a / n:.3f}/s  relative {r / n:.3f}/s", flush=True)
