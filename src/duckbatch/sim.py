"""One vectorized mjlab env, many attempts: population DAgger and population evaluation.

WHY A POPULATION. On a 4 GB GPU the simulator is the expensive, shared thing, and a batch of
2048 envs is far more than one small student needs. So the env batch is split into K
contiguous partitions, one per attempt ("arm"), and every step advances all K at once: one
physics step, one teacher forward pass over the whole batch, K tiny student forwards. A batch of
attempts costs about what one attempt costs. Arms are killed between rungs (see `batch/`) and
their envs are handed to the survivors, so the env budget stays fixed while the field narrows.

WHY DAGGER. Behaviour cloning on teacher rollouts alone never sees the states a student's own
mistakes lead to. Each arm drives its own envs (with a teacher-override probability `beta`
that decays to 0), and the teacher labels every state the arm visits. The student is then
trained on its own state distribution, which is what makes small students hold up.

The teacher can itself be an arm (`kind="teacher"`): it is the reference row in every eval.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .policy import build_mlp, load_teacher

# The velocity-with-standing task Pollen's default walker (velstand.onnx) was trained on.
DEFAULT_TASK = "Mjlab-VelStand-Flat-MicroDuck"
# Past every curriculum stage in microduck_rl (the last ones sit at 2000 * 24 steps).
FINAL_CURRICULUM_STEP = 10**9


def make_env(task: str, num_envs: int, device: str = "cuda:0", seed: int = 0,
             play: bool = False, final_curriculum: bool = True):
    """The task's training env (domain randomization, pushes and noise left on), curricula at their
    final stage so an attempt is judged on the full problem rather than the warm-up."""
    import mjlab.tasks  # noqa: F401  registers the mjlab_microduck plugin tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=play)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    if final_curriculum:
        env.common_step_counter = FINAL_CURRICULUM_STEP
    return env


@dataclass
class Arm:
    """One attempt in the population: a student being distilled, or the teacher as reference."""

    arm_id: str
    kind: str  # "student" | "teacher"
    hidden: tuple[int, ...] = ()
    lr: float = 1e-3
    activation: str = "Elu"
    net: Any = None
    opt: Any = None
    buf_obs: Any = None
    buf_act: Any = None
    buf_n: int = 0
    buf_ptr: int = 0
    updates: int = 0
    samples_seen: int = 0
    history: list[dict] = field(default_factory=list)


class Population:
    """K arms sharing one env batch. `partition()` hands out contiguous env slices to live arms."""

    def __init__(self, env, teacher_path: str, arms: list[Arm], device: str = "cuda:0",
                 buffer_size: int = 262_144, seed: int = 0):
        self.env = env
        self.device = device
        self.teacher = load_teacher(teacher_path, device)
        self.buffer_size = buffer_size
        self.gen = torch.Generator(device=device).manual_seed(seed)
        torch.manual_seed(seed)
        self.arms: list[Arm] = []
        for arm in arms:
            self.add_arm(arm)
        self.obs = None

    # -- membership ------------------------------------------------------------------------

    def add_arm(self, arm: Arm) -> None:
        if arm.kind == "student" and arm.net is None:
            arm.net = build_mlp(arm.hidden, arm.activation).to(self.device)
            # Start from the teacher's normalizer: the student sees the same scaled inputs, and
            # the export stays a plain `(obs-mean)/std -> MLP` graph every Microduck tool loads.
            with torch.no_grad():
                arm.net.obs_mean.copy_(self.teacher.obs_mean)
                arm.net.obs_std.copy_(self.teacher.obs_std)
            arm.opt = torch.optim.Adam(arm.net.parameters(), lr=arm.lr)
            arm.buf_obs = torch.zeros(self.buffer_size, 61, device=self.device)
            arm.buf_act = torch.zeros(self.buffer_size, 14, device=self.device)
        self.arms.append(arm)

    def drop_arm(self, arm_id: str) -> Arm:
        arm = next(a for a in self.arms if a.arm_id == arm_id)
        self.arms.remove(arm)
        # Free its replay buffer now; on 4 GB it is the difference between fitting or not.
        arm.buf_obs = arm.buf_act = arm.opt = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return arm

    def partition(self) -> list[tuple[Arm, slice]]:
        n = self.env.num_envs
        k = len(self.arms)
        edges = [round(i * n / k) for i in range(k + 1)]
        return [(a, slice(edges[i], edges[i + 1])) for i, a in enumerate(self.arms)]

    # -- stepping ----------------------------------------------------------------------------

    def _actions(self, obs: torch.Tensor, beta: float, record: bool):
        """Every arm's action on its own slice; the teacher labels the whole batch in one pass."""
        with torch.no_grad():
            teacher_act = self.teacher(obs)
        act = teacher_act.clone()
        for arm, sl in self.partition():
            if arm.kind == "teacher":
                continue
            with torch.no_grad():
                student_act = arm.net(obs[sl])
            if beta > 0:
                use_teacher = torch.rand(student_act.shape[0], 1, device=self.device,
                                         generator=self.gen) < beta
                act[sl] = torch.where(use_teacher, teacher_act[sl], student_act)
            else:
                act[sl] = student_act
            if record:
                self._push(arm, obs[sl], teacher_act[sl])
        return act, teacher_act

    def _push(self, arm: Arm, obs: torch.Tensor, label: torch.Tensor) -> None:
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + arm.buf_ptr) % self.buffer_size
        arm.buf_obs[idx] = obs
        arm.buf_act[idx] = label
        arm.buf_ptr = (arm.buf_ptr + n) % self.buffer_size
        arm.buf_n = min(arm.buf_n + n, self.buffer_size)
        arm.samples_seen += n

    def reset(self, seed: int | None = None):
        obs, _ = self.env.reset(seed=seed)
        self.obs = obs["actor"]
        return self.obs

    def step(self, beta: float, record: bool = True):
        if self.obs is None:
            self.reset()
        act, teacher_act = self._actions(self.obs, beta, record)
        obs, rew, terminated, truncated, _ = self.env.step(act)
        self.obs = obs["actor"]
        return act, teacher_act, rew, terminated, truncated

    # -- learning ------------------------------------------------------------------------------

    def fit(self, grad_steps: int, batch_size: int = 4096) -> dict[str, float]:
        """`grad_steps` Adam steps per student on its aggregated DAgger buffer (MSE to teacher)."""
        losses = {}
        for arm in self.arms:
            if arm.kind != "student" or arm.buf_n < batch_size:
                continue
            total = 0.0
            for _ in range(grad_steps):
                idx = torch.randint(0, arm.buf_n, (batch_size,), device=self.device,
                                    generator=self.gen)
                loss = torch.nn.functional.mse_loss(arm.net(arm.buf_obs[idx]), arm.buf_act[idx])
                arm.opt.zero_grad(set_to_none=True)
                loss.backward()
                arm.opt.step()
                total += float(loss)
            arm.updates += grad_steps
            losses[arm.arm_id] = total / grad_steps
        return losses

    def train(self, iterations: int, steps_per_iter: int = 24, grad_steps: int = 20,
              beta0: float = 1.0, beta_decay_iters: int = 20, start_iter: int = 0,
              log=print) -> None:
        """DAgger loop. `beta` (teacher-override probability) decays linearly from `beta0` to 0
        over `beta_decay_iters` iterations counted from the arm population's first iteration."""
        for it in range(start_iter, start_iter + iterations):
            beta = max(0.0, beta0 * (1 - it / max(beta_decay_iters, 1)))
            t0 = time.perf_counter()
            for _ in range(steps_per_iter):
                self.step(beta, record=True)
            sim_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            losses = self.fit(grad_steps)
            fit_s = time.perf_counter() - t0
            for arm in self.arms:
                if arm.arm_id in losses:
                    arm.history.append({"iter": it, "loss": losses[arm.arm_id], "beta": beta})
            if it % 10 == 0 or it == start_iter + iterations - 1:
                sps = self.env.num_envs * steps_per_iter / max(sim_s, 1e-9)
                shown = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
                log(f"[distill] it={it} beta={beta:.2f} env_steps/s={sps:,.0f} "
                    f"sim={sim_s:.2f}s fit={fit_s:.2f}s  {shown}")

    # -- evaluation ------------------------------------------------------------------------------

    def evaluate(self, seconds: float = 20.0, seed: int = 12345,
                 record_obs: int = 0) -> dict[str, dict[str, float]]:
        """Every live arm drives its own envs with no teacher help, from a fixed reset seed.

        Metrics per arm (all on the arm's own state distribution):
          falls_per_min     upright -> tilted-past-60-degrees transitions, per env-minute. Not
                            terminations: at its final curriculum stage VelStand turns
                            `fell_over` off (falls are recovery practice), so a fallen duck
                            keeps its episode and only `fallen_too_long` ends it.
          down_frac         fraction of env-steps spent tilted past 60 degrees
          lin_err / ang_err mean |commanded - actual| planar / yaw velocity, m/s and rad/s
          teacher_mse       mean squared gap to the teacher's action on the same observation
          reward_per_s      the task's own reward, per simulated second
        """
        env = self.env
        dt = env.step_dt
        steps = int(math.ceil(seconds / dt))
        robot = env.scene["robot"]
        parts = self.partition()
        acc = {a.arm_id: dict(falls=0.0, down=0.0, lin=0.0, ang=0.0, mse=0.0, rew=0.0, n=0.0)
               for a, _ in parts}
        self.reset(seed=seed)
        self.recorded_obs = []
        was_down = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        for i in range(steps):
            if record_obs and i % max(1, steps // 32) == 0:
                # Observations as the policies saw them, sampled across the whole eval.
                self.recorded_obs.append(self.obs[:: max(1, self.env.num_envs * 32 // record_obs)]
                                         .cpu())
            act, teacher_act, rew, terminated, truncated = self.step(beta=0.0, record=False)
            cmd = env.command_manager.get_command("twist")
            v = robot.data.root_link_lin_vel_b
            w = robot.data.root_link_ang_vel_b
            lin = torch.linalg.norm(cmd[:, :2] - v[:, :2], dim=1)
            ang = (cmd[:, 2] - w[:, 2]).abs()
            mse = ((act - teacher_act) ** 2).mean(dim=1)
            # Gravity in the body frame: z = -1 upright, -0.5 at 60 degrees of tilt.
            down = robot.data.projected_gravity_b[:, 2] > -0.5
            fell = down & ~was_down
            was_down = down
            for arm, sl in parts:
                a = acc[arm.arm_id]
                a["falls"] += float(fell[sl].sum())
                a["down"] += float(down[sl].sum())
                a["lin"] += float(lin[sl].sum())
                a["ang"] += float(ang[sl].sum())
                a["mse"] += float(mse[sl].sum())
                a["rew"] += float(rew[sl].sum())
                a["n"] += sl.stop - sl.start
        out = {}
        for arm, sl in parts:
            a = acc[arm.arm_id]
            n = max(a["n"], 1.0)
            out[arm.arm_id] = {
                "envs": sl.stop - sl.start,
                "sim_seconds": seconds,
                "falls_per_min": a["falls"] / (n * dt / 60.0),
                "down_frac": a["down"] / n,
                "lin_err": a["lin"] / n,
                "ang_err": a["ang"] / n,
                "teacher_mse": a["mse"] / n,
                "reward_per_s": a["rew"] / (n * dt),
            }
        self.obs = None  # training resumes from a fresh reset
        return out
