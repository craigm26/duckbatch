"""PPO fine-tuning of a distilled student: free where the teacher fails, anchored where it works.

    duckbatch finetune menus/b003-dead-band.yaml [--out records]

WHY. b002's students inherited their teacher's dead band: Pollen's velstand answers a command
below about 0.15 m/s by standing still (mjlab, s000), so "walk slowly" in plain language maps
to a duck that does not move. Distillation copies that faithfully; only RL can remove it.

THE LEVER IS THE REWARD'S WIDTH, NOT MORE PRACTICE. VelStand already samples small commands and
turns-in-place (15% of envs). But its tracking reward has width √0.1 ≈ 0.32 m/s, so standing
still when asked for 0.1 m/s earns exp(−0.01/0.1) ≈ 90% of the reward: standing is nearly
free. The fine-tune narrows the tracking widths, so a slow command is worth walking for.

WHY NOT FROM SCRATCH, AND WHY THE ANCHOR. The student already walks at 0.3-0.4 m/s and gets up
from a fall; PPO with a wide exploration noise would unlearn both before it learned anything
new. So: the actor is warm-started from the student (weights and frozen normaliser, exactly as
its ONNX computes), exploration starts small, and after each PPO update a behaviour-cloning
pass pulls the actor toward the frozen teacher on the frames where the teacher is good:
  - fallen frames (tilt above the gate): the teacher's recovery;
  - upright frames commanded at or above `walk_min_speed`, or at a standstill: its walk and stand;
  - everything in the dead band (slow, or turn-in-place) is left to PPO alone.
This is microduck_rl's `PpoWithExpertBc` pattern (tasks/distill.py), with the expert loaded from
Pollen's published velstand.onnx instead of a private W&B checkpoint, and gated on the command
as well as the tilt.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from rsl_rl.algorithms import PPO

from .policy import build_mlp, export_onnx, load_teacher, mlp_params, onnx_mlp_weights

GRAVITY = (3, 6)
TWIST = (48, 51)


class PpoWithTeacherAnchor(PPO):
    """PPO, then a behaviour-cloning pass toward a frozen teacher on the frames it is good at."""

    def __init__(self, actor, critic, storage, *args, anchor_cfg: dict | None = None, **kwargs) -> None:
        kwargs.pop("symmetry_cfg", None)
        kwargs.pop("bc_cfg", None)
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.anchor_cfg = anchor_cfg
        self.teacher = None
        if anchor_cfg:
            self.teacher = load_teacher(anchor_cfg["teacher_onnx"], str(self.device))
            self.anchor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=anchor_cfg["learning_rate"])
            print(f"[finetune] teacher anchor ON: {anchor_cfg['teacher_onnx']} fallen×{anchor_cfg['coef_fallen']} "
                  f"walk/stand×{anchor_cfg['coef_walk']} (|cmd|≥{anchor_cfg['walk_min_speed']} or ≈0); "
                  f"dead band free")

    def _masks(self, flat: torch.Tensor):
        c = self.anchor_cfg
        g = flat[:, GRAVITY[0]:GRAVITY[1]]
        g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)
        fallen = -g[:, 2] < math.cos(math.radians(c["gate_tilt_deg"]))
        tw = flat[:, TWIST[0]:TWIST[1]]
        speed = tw[:, :2].norm(dim=1)
        still = (speed < 0.02) & (tw[:, 2].abs() < 0.05)
        walking = speed >= c["walk_min_speed"]
        anchored_upright = (~fallen) & (walking | still)
        return fallen, anchored_upright

    def _anchor_update(self) -> dict[str, float]:
        c = self.anchor_cfg
        obs_td = self.storage.observations.flatten(0, 1)
        groups = list(self.actor.obs_groups)
        flat = torch.cat([obs_td[g] for g in groups], dim=-1)
        fallen, upright = self._masks(flat)
        use = fallen | upright
        stats = {"anchor_fallen_frac": fallen.float().mean().item(),
                 "anchor_upright_frac": upright.float().mean().item(),
                 "anchor_free_frac": (~use).float().mean().item()}
        idx = use.nonzero().flatten()
        if idx.numel() == 0:
            return stats
        obs_sel = obs_td[idx]
        flat_sel = flat[idx]
        weight = torch.where(fallen[idx], torch.full((idx.numel(),), float(c["coef_fallen"]), device=self.device),
                             torch.full((idx.numel(),), float(c["coef_walk"]), device=self.device))
        with torch.no_grad():
            target = self.teacher(flat_sel)
        n = idx.numel()
        mb = max(1024, n // c["mini_batches"])
        total, steps = 0.0, 0
        for _ in range(c["epochs"]):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                sel = perm[s:s + mb]
                pred = self.actor(obs_sel[sel])
                loss = (weight[sel] * (pred - target[sel]).pow(2).mean(dim=1)).mean()
                self.anchor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.anchor_optimizer.step()
                total += loss.item()
                steps += 1
        stats["anchor_bc"] = total / max(steps, 1)
        return stats

    def update(self) -> dict[str, float]:
        loss = super().update()
        if self.teacher is not None:
            loss.update(self._anchor_update())
        return loss


def warm_start_actor(actor, student_onnx: str | Path) -> dict[str, Any]:
    """Load the student's ONNX into rsl_rl's MLPModel so it computes exactly what the ONNX does,
    and freeze the observation normaliser so fine-tuning cannot rescale the inputs under it."""
    w = onnx_mlp_weights(student_onnx)
    norm = actor.obs_normalizer
    linears = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)]
    if len(linears) != len(w.layers):
        raise ValueError(f"actor has {len(linears)} layers, student {len(w.layers)}")
    with torch.no_grad():
        dev = norm._mean.device
        norm._mean.copy_(torch.from_numpy(w.mean).view(1, -1).to(dev))
        # EmpiricalNormalization divides by (_std + eps); the ONNX divides by the exported std.
        std = torch.from_numpy(w.std).view(1, -1).to(dev) - norm.eps
        norm._std.copy_(std)
        norm._var.copy_(std ** 2)
        norm.count.fill_(10**9)
        norm.until = 0
        for lin, (W, b) in zip(linears, w.layers):
            if tuple(lin.weight.shape) != W.shape:
                raise ValueError(f"layer {tuple(lin.weight.shape)} vs student {W.shape}")
            lin.weight.copy_(torch.from_numpy(W).to(dev))
            lin.bias.copy_(torch.from_numpy(b).to(dev))
    return {"hidden": list(w.hidden), "params": w.n_params}


def export_actor(actor, path: Path, metadata: dict[str, str]) -> Path:
    """The fine-tuned actor as the same `Sub, Div, (Gemm, Elu)×k, Gemm` graph every tool loads."""
    linears = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)]
    net = build_mlp([l.out_features for l in linears[:-1]])
    with torch.no_grad():
        net.obs_mean.copy_(actor.obs_normalizer._mean.cpu())
        net.obs_std.copy_((actor.obs_normalizer._std + actor.obs_normalizer.eps).cpu())
        for dst, src in zip([m for m in net.mlp if isinstance(m, torch.nn.Linear)], linears):
            dst.weight.copy_(src.weight.cpu())
            dst.bias.copy_(src.bias.cpu())
    return export_onnx(net, path, metadata)


def run_finetune(menu_path: str | Path, out_root: str | Path = "records", log=print) -> Path:
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    from .sim import FINAL_CURRICULUM_STEP

    menu = yaml.safe_load(Path(menu_path).read_text())
    out = Path(out_root) / menu["batch_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "menu.yaml").write_text(Path(menu_path).read_text())
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    task = menu["task"]
    student = menu["student"]
    ft = menu["finetune"]

    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = int(ft["num_envs"])
    env_cfg.seed = int(menu.get("seed", 0))
    tw = env_cfg.commands["twist"]
    for k, v in ft.get("command", {}).items():
        if k in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            setattr(tw.ranges, k, tuple(v))
        else:
            setattr(tw, k, v)
    for term, params in ft.get("rewards", {}).items():
        for k, v in params.items():
            if k == "weight":
                env_cfg.rewards[term].weight = float(v)
            else:
                env_cfg.rewards[term].params[k] = float(v)

    agent = load_rl_cfg(task)
    sw = onnx_mlp_weights(student)
    agent.actor.hidden_dims = tuple(sw.hidden)
    agent.actor.distribution_cfg = {**agent.actor.distribution_cfg, "init_std": float(ft["init_std"])}
    agent.max_iterations = int(ft["iterations"])
    agent.save_interval = int(ft.get("save_interval", 250))
    agent.experiment_name = menu["batch_id"]
    agent.logger = "tensorboard"
    alg = asdict(agent.algorithm)
    agent_dict = asdict(agent)
    agent_dict["algorithm"] = {k: v for k, v in alg.items() if k not in ("bc_cfg", "symmetry_cfg")}
    agent_dict["algorithm"].update({
        "class_name": "duckbatch.finetune.PpoWithTeacherAnchor",
        "learning_rate": float(ft["learning_rate"]),
        "entropy_coef": float(ft.get("entropy_coef", 0.002)),
        "anchor_cfg": {**ft["anchor"], "teacher_onnx": menu["teacher"]},
    })

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.common_step_counter = FINAL_CURRICULUM_STEP
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner_cls = load_runner_cls(task)
    log_dir = out / "train"
    runner = runner_cls(wrapped, agent_dict, str(log_dir), device)
    info = warm_start_actor(runner.alg.actor, student)
    log(f"[finetune] warm start from {student}: {info['hidden']} ({info['params']:,} params); "
        f"{ft['iterations']} iters x {ft['num_envs']} envs on {device}")
    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=int(ft["iterations"]), init_at_random_ep_len=True)
    train_s = time.perf_counter() - t0
    teacher_meta = load_teacher(menu["teacher"]).metadata
    meta = {k: v for k, v in teacher_meta.items()
            if k in ("joint_names", "default_joint_pos", "command_names", "observation_names", "action_scale")}
    meta.update({"duckbatch_batch": menu["batch_id"], "duckbatch_from": Path(student).name,
                 "run_path": f"duckbatch/{menu['batch_id']}"})
    onnx_path = export_actor(runner.alg.actor, out / "policies" / "finetuned" / "policy.onnx", meta)
    env.close()
    del runner, wrapped, env
    torch.cuda.empty_cache()
    log(f"[finetune] trained in {train_s / 60:.1f} min -> {onnx_path}")

    # The measurements the design pre-registers: speed curves for teacher, student and the
    # fine-tuned student, then the b002 walk + recovery evaluation on held-out seeds.
    from .pairs import speed_curve
    from .sim import Arm, Population, make_env

    pols = {"velstand": menu["teacher"], "student": student, "finetuned": str(onnx_path)}
    speed_curve(out / "speed", pols, num_envs=int(menu.get("eval_envs", 512)))
    env = make_env(task, int(menu.get("eval_envs", 512)), device=device, seed=0)
    arms = [Arm("teacher", "teacher"),
            Arm("student", "fixed", net=load_teacher(student, device)),
            Arm("finetuned", "fixed", net=load_teacher(str(onnx_path), device))]
    pop = Population(env, menu["teacher"], arms, device=device)
    final = {}
    from .batch.runner import evaluate_both
    for s in menu.get("final_seeds", [2001, 2002, 2003]):
        final[str(s)] = evaluate_both(pop, 30.0, 6.0, int(s))
    env.close()
    record = {"schema": "duckbatch.finetune.v1", "batch_id": menu["batch_id"], "task": task,
              "student": student, "teacher": menu["teacher"], "train_seconds": round(train_s, 1),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "params": mlp_params(sw.hidden), "policy": str(onnx_path.relative_to(out)),
              "final_eval": {"seconds": 30.0, "recover_seconds": 6.0, "by_seed": final},
              "speed_curve": json.loads((out / "speed" / "speed_curve.json").read_text())}
    (out / "record.json").write_text(json.dumps(record, indent=1))
    log("[record] " + json.dumps(record))  # a second copy in the log, in case the upload fails
    log(f"[finetune] record -> {out}/record.json")
    return out
