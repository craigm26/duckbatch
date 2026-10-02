"""Real VelStand env: mjlab's term, b003d's relative term and b003e's stride term, all at weight 2."""

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
from duckbatch.rewards import track_linear_velocity_relative, track_linear_velocity_stride  # noqa: E402
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
cfg.rewards["track_linear_velocity_stride"] = RewardTermCfg(
    func=track_linear_velocity_stride, weight=2.0,
    params={"command_name": "twist", "std": std, "full_width_speed": 0.4,
            "min_width_speed": 0.05, "stand_below": 0.02, "window_s": 0.5})
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
env.common_step_counter = FINAL_CURRICULUM_STEP
names = list(env.reward_manager.active_terms)
i_abs, i_rel = names.index("track_linear_velocity"), names.index("track_linear_velocity_relative")
i_str = names.index("track_linear_velocity_stride")
print("[check] stride term built as", type(env.reward_manager.get_term_cfg("track_linear_velocity_stride").func).__name__,
      "window", env.reward_manager.get_term_cfg("track_linear_velocity_stride").func.window, flush=True)
robot = env.scene["robot"]
prof = Population(env, f"{DB}/teachers/velstand.onnx", [], device="cpu")
nets = {"student": load_teacher(f"{DB}/records/b002-student-size-longer/policies/a02/policy.onnx", "cpu"),
        "b003b": load_teacher("/home/craigm26/scratch/b003d/b003b_policy.onnx", "cpu")}
steps = int(round(4.0 / env.step_dt))
for pname, twist in [("student", (0.1, 0.0, 0.0)), ("student", (0.4, 0.0, 0.0)), ("student", (0.0, 0.0, 0.0)),
                     ("b003b", (0.3, 0.0, 0.0))]:
    net = nets[pname]
    with prof.profile("walk"), fixed_command(env, twist):
        obs = env.reset(seed=4001)[0]["actor"]
        a = r = st = n = vx = 0.0
        for t in range(steps):
            with torch.no_grad():
                obs = env.step(net(obs))[0]["actor"]
            if t >= steps // 2:
                up = (robot.data.projected_gravity_b[:, 2] < -0.5).float()
                sr = env.reward_manager._step_reward
                a += float((sr[:, i_abs] * up).sum()); r += float((sr[:, i_rel] * up).sum())
                st += float((sr[:, i_str] * up).sum())
                vx += float((robot.data.root_link_lin_vel_b[:, 0] * up).sum()); n += float(up.sum())
    n = max(n, 1.0)
    print(f"[check] {pname} cmd {twist}: vx {vx / n:+.3f}  mjlab {a / n:.3f}/s  relative {r / n:.3f}/s  "
          f"stride {st / n:.3f}/s", flush=True)
