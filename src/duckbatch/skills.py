"""One student, many skills: distil several of Pollen's policies into one small network.

WHY. A plain-language sequence ("walk forward, sit, then peck the ground") is several of Pollen's
policies in a row, and the robot runtime swaps networks at each boundary. The new network starts
from states the old one left behind, which is where sequences stumble. One student that has
seen every skill, and the switches between them, learns the transitions instead of inheriting
them.

HOW THE SKILL GETS IN WITHOUT BREAKING THE CONTRACT. The observation stays 61 floats. Three
command slots — 55, 56 and 60, the body-pose x, y and yaw that every teacher saw as zero —
carry a skill code, and the walker's code is all zeros, so the multi-skill student is still a
plain walker to any client that knows nothing about skills. Those three slots get mean 0,
std 1 in the student's normaliser (the teachers' were ~0.01: a ±1 code would otherwise arrive
as ±77σ). The export is the same `Sub, Div, (Gemm, Elu)×k, Gemm` graph.

EACH TEACHER GETS ITS OWN NATIVE COMMAND, exactly as Pollen's runtime drives it
(microduck-policies manifest.json):
  walk       velstand.onnx          twist (vx, vy, wz)
  stand      alpha_stand.onnx       zeros
  sitstand   alpha_sitstand.onnx    posture flag in twist.vx: 1 = sit, 0 = stand back up
  pick       alpha_ground_pick.onnx phase: [cos 2πφ, sin 2πφ] in twist.vx/vy, φ over 4 s, ends at 0.7
  kick_l/r   ball_kick_*.onnx       zeros, for 0.5 s

THE TEACHER ARM IS THE BASELINE. In the population the teacher executes the active skill's
network and swaps at each boundary — that is Pollen's runtime. The student is measured against
it on the same schedule: falls in the 2 s after a switch, and the action gap per skill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .policy import load_teacher
from .sim import Arm, Population

CODE_SLOTS = (55, 56, 60)
CMD = slice(48, 61)


@dataclass(frozen=True)
class Skill:
    name: str
    teacher: str
    code: tuple[float, float, float]
    seconds: tuple[float, float]   # how long an episode of it lasts, min/max
    weight: float                  # how often the schedule picks it


SKILLS = [
    Skill("walk", "teachers/velstand.onnx", (0, 0, 0), (2.0, 5.0), 0.35),
    Skill("stand", "teachers/alpha_stand.onnx", (1, 0, 0), (1.0, 3.0), 0.15),
    Skill("sit", "teachers/alpha_sitstand.onnx", (-1, 0, 0), (2.5, 4.0), 0.10),
    Skill("unsit", "teachers/alpha_sitstand.onnx", (-1, 0, 0), (2.0, 2.0), 0.0),   # follows sit
    Skill("pick", "teachers/alpha_ground_pick.onnx", (0, 1, 0), (2.8, 2.8), 0.15),
    Skill("kick_left", "teachers/ball_kick_left.onnx", (0, 0, 1), (0.5, 0.5), 0.125),
    Skill("kick_right", "teachers/ball_kick_right.onnx", (0, 0, -1), (0.5, 0.5), 0.125),
]
NAMES = [s.name for s in SKILLS]
SIT, UNSIT, PICK, WALK = NAMES.index("sit"), NAMES.index("unsit"), NAMES.index("pick"), NAMES.index("walk")
PICK_PERIOD, PICK_END = 4.0, 0.7


class SkillSchedule:
    """Per env: the active skill, how long it has run, how long it will, and the walk twist."""

    def __init__(self, n: int, dt: float, device: str, gen: torch.Generator):
        self.n, self.dt, self.dev, self.gen = n, dt, device, gen
        self.skill = torch.zeros(n, dtype=torch.long, device=device)
        self.t = torch.zeros(n, device=device)
        self.until = torch.zeros(n, device=device)
        self.twist = torch.zeros(n, 3, device=device)
        self.since_switch = torch.full((n,), 1e9, device=device)
        w = torch.tensor([s.weight for s in SKILLS], device=device)
        self.pick_probs = w / w.sum()
        self.lo = torch.tensor([s.seconds[0] for s in SKILLS], device=device)
        self.hi = torch.tensor([s.seconds[1] for s in SKILLS], device=device)
        self.restart(torch.arange(n, device=device), first=True)

    def _rand(self, k: int) -> torch.Tensor:
        return torch.rand(k, device=self.dev, generator=self.gen)

    def restart(self, ids: torch.Tensor, first: bool = False) -> None:
        """Pick the next skill for `ids`. An unfinished sit always hands to unsit (stand back up)."""
        if ids.numel() == 0:
            return
        nxt = torch.multinomial(self.pick_probs, ids.numel(), replacement=True, generator=self.gen)
        if not first:
            nxt = torch.where(self.skill[ids] == SIT, torch.full_like(nxt, UNSIT), nxt)
        self.skill[ids] = nxt
        self.t[ids] = 0
        self.until[ids] = self.lo[nxt] + (self.hi[nxt] - self.lo[nxt]) * self._rand(ids.numel())
        # Walk twists across VelStand's training range, a fifth of them turning on the spot.
        tw = (self._rand(ids.numel() * 3).view(-1, 3) * 2 - 1) * torch.tensor([0.4, 0.3, 1.0], device=self.dev)
        spin = self._rand(ids.numel()) < 0.2
        tw[spin, :2] = 0
        self.twist[ids] = tw
        if not first:
            self.since_switch[ids] = 0

    def step(self) -> None:
        self.t += self.dt
        self.since_switch += self.dt
        self.restart((self.t >= self.until).nonzero().flatten())

    def commands(self):
        """(teacher_cmd, code) per env: each teacher's native 13-wide command, and the skill code."""
        cmd = torch.zeros(self.n, 13, device=self.dev)
        walk = self.skill == WALK
        cmd[walk, :3] = self.twist[walk]
        cmd[self.skill == SIT, 0] = 1.0
        pick = self.skill == PICK
        phase = (self.t / PICK_PERIOD).clamp(max=PICK_END)
        cmd[pick, 0] = torch.cos(2 * math.pi * phase[pick])
        cmd[pick, 1] = torch.sin(2 * math.pi * phase[pick])
        codes = torch.tensor([s.code for s in SKILLS], device=self.dev, dtype=torch.float)
        return cmd, codes[self.skill]


class SkillPopulation(Population):
    """Population DAgger where the teacher is whichever of Pollen's skills is active per env."""

    def __init__(self, env, arms: list[Arm], device: str = "cuda:0", seed: int = 0, **kw):
        super().__init__(env, SKILLS[0].teacher, arms, device=device, seed=seed, **kw)
        self.teachers = [load_teacher(s.teacher, device) for s in SKILLS]
        self.schedule = SkillSchedule(env.num_envs, env.step_dt, device, self.gen)
        for arm in self.arms:
            if arm.kind == "student":
                with torch.no_grad():
                    for i in CODE_SLOTS:
                        arm.net.obs_mean[0, i] = 0.0
                        arm.net.obs_std[0, i] = 1.0

    def _actions(self, obs: torch.Tensor, beta: float, record: bool):
        cmd, code = self.schedule.commands()
        t_obs = obs.clone()
        t_obs[:, CMD] = cmd                     # what each teacher was trained to read
        s_obs = t_obs.clone()
        s_obs[:, list(CODE_SLOTS)] = code       # the student also reads which skill
        teacher_act = torch.zeros(obs.shape[0], 14, device=self.device)
        with torch.no_grad():
            for k, net in enumerate(self.teachers):
                m = self.schedule.skill == k
                if m.any():
                    teacher_act[m] = net(t_obs[m])
        act = teacher_act.clone()
        for arm, sl in self.partition():
            if arm.kind == "teacher":
                continue
            with torch.no_grad():
                sa = arm.net(s_obs[sl])
            if arm.kind == "student" and beta > 0:
                use_t = torch.rand(sa.shape[0], 1, device=self.device, generator=self.gen) < beta
                act[sl] = torch.where(use_t, teacher_act[sl], sa)
            else:
                act[sl] = sa
            if record and arm.kind == "student":
                self._push(arm, s_obs[sl], teacher_act[sl])
        self._last_s_obs = s_obs
        return act, teacher_act

    def step(self, beta: float, record: bool = True):
        out = super().step(beta, record)
        self.schedule.step()
        return out

    def evaluate_switches(self, seconds: float = 30.0, seed: int = 5000, window: float = 2.0):
        """Every arm on the same schedule: falls within `window` s after a switch (per 100
        switches), falls anywhere (per env-minute), and the action gap to the active teacher per
        skill. The teacher arm swaps networks at each switch: that is the runtime baseline."""
        env = self.env
        dt = env.step_dt
        steps = int(math.ceil(seconds / dt))
        robot = env.scene["robot"]
        parts = self.partition()
        self.schedule = SkillSchedule(env.num_envs, dt, self.device,
                                      torch.Generator(device=self.device).manual_seed(seed))
        with self.profile("walk"):
            self.reset(seed=seed)
            was_down = robot.data.projected_gravity_b[:, 2] > -0.5
            acc = {a.arm_id: {"falls": 0.0, "falls_after_switch": 0.0, "switches": 0.0, "n": 0.0,
                              "gap": torch.zeros(len(SKILLS), device=self.device),
                              "gap_n": torch.zeros(len(SKILLS), device=self.device)} for a, _ in parts}
            for _ in range(steps):
                prev_since = self.schedule.since_switch.clone()
                act, teacher_act, _, terminated, truncated = self.step(beta=0.0, record=False)
                switched = self.schedule.since_switch < prev_since
                down = robot.data.projected_gravity_b[:, 2] > -0.5
                fell = down & ~was_down & ~(terminated | truncated).bool()
                was_down = down
                recent = self.schedule.since_switch <= window
                gap = ((act - teacher_act) ** 2).mean(dim=1)
                onehot = torch.nn.functional.one_hot(self.schedule.skill, len(SKILLS)).float()
                for arm, sl in parts:
                    a = acc[arm.arm_id]
                    a["falls"] += float(fell[sl].sum())
                    a["falls_after_switch"] += float((fell & recent)[sl].sum())
                    a["switches"] += float(switched[sl].sum())
                    a["n"] += sl.stop - sl.start
                    a["gap"] += (onehot[sl] * gap[sl, None]).sum(dim=0)
                    a["gap_n"] += onehot[sl].sum(dim=0)
        out = {}
        for arm, sl in parts:
            a = acc[arm.arm_id]
            out[arm.arm_id] = {
                "envs": sl.stop - sl.start,
                "falls_per_min": a["falls"] / (a["n"] * dt / 60.0),
                "falls_after_switch_per_100": 100.0 * a["falls_after_switch"] / max(a["switches"], 1.0),
                "switches": a["switches"],
                "gap_by_skill": {s.name: float(a["gap"][k] / a["gap_n"][k].clamp(min=1))
                                 for k, s in enumerate(SKILLS)},
            }
        self.obs = None
        return out


def run_skills(menu_path, out_root="records", device: str = "cuda:0", log=print):
    """Train the menu's multi-skill students together, then measure them against network
    swapping on a held-out schedule. Writes records/<batch_id>/record.json and the ONNX files."""
    import json
    import time
    from pathlib import Path

    import yaml

    from .policy import export_onnx, mlp_params
    from .sim import DEFAULT_TASK, make_env

    menu = yaml.safe_load(Path(menu_path).read_text())
    out = Path(out_root) / menu["batch_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "menu.yaml").write_text(Path(menu_path).read_text())
    tr = menu["train"]
    env = make_env(menu.get("task", DEFAULT_TASK), int(menu.get("num_envs", 512)), device=device,
                   seed=int(menu.get("seed", 0)))
    arms = [Arm("teacher", "teacher")] + [
        Arm(a["id"], "student", tuple(a["hidden"]), float(a.get("lr", 1e-3))) for a in menu["arms"]]
    pop = SkillPopulation(env, arms, device=device, seed=int(menu.get("seed", 0)),
                          buffer_size=int(tr.get("buffer_size", 262_144)))
    t0 = time.perf_counter()
    # Walk profile throughout: no prone spawns or topples. Pollen's skill teachers were trained
    # from standing starts; asking the sit or pick teacher to label a prone duck labels nothing.
    with pop.profile("walk"):
        pop.train(int(tr["iterations"]), steps_per_iter=int(tr.get("steps_per_iter", 24)),
                  grad_steps=int(tr.get("grad_steps", 20)), beta0=float(tr.get("beta0", 1.0)),
                  beta_decay_iters=int(tr.get("beta_decay_iters", 60)), log=log)
    train_s = time.perf_counter() - t0
    policies = {}
    meta_src = pop.teachers[0].metadata
    for a in pop.arms:
        if a.kind != "student":
            continue
        meta = {k: v for k, v in meta_src.items()
                if k in ("joint_names", "default_joint_pos", "command_names", "observation_names", "action_scale")}
        meta.update({"duckbatch_batch": menu["batch_id"], "duckbatch_arm": a.arm_id,
                     "duckbatch_skill_slots": "55,56,60",
                     "duckbatch_skill_codes": json.dumps({s.name: s.code for s in SKILLS})})
        policies[a.arm_id] = str(export_onnx(a.net, out / "policies" / a.arm_id / "policy.onnx", meta).relative_to(out))
    evals = {str(s): pop.evaluate_switches(seconds=float(menu.get("eval_seconds", 30)), seed=int(s))
             for s in menu.get("eval_seeds", [6001, 6002, 6003])}
    record = {"schema": "duckbatch.skills.v1", "batch_id": menu["batch_id"],
              "skills": [{"name": s.name, "teacher": s.teacher, "code": s.code} for s in SKILLS],
              "arms": {a["id"]: {"hidden": a["hidden"], "params": mlp_params(a["hidden"]),
                                 "policy": policies.get(a["id"])} for a in menu["arms"]},
              "train_seconds": round(train_s, 1), "eval": evals}
    (out / "record.json").write_text(json.dumps(record, indent=1))
    env.close()
    log(f"[skills] done in {train_s / 60:.1f} min -> {out}/record.json")
    return out
