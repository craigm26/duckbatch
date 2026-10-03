# s003: where Pollen's other skills are weakest, measured before anyone pays for a GPU (2026-10-03)

`skill_eval.profile` ran each skill in its own task. It records which termination ended each
episode and what each of Pollen's reward terms paid per second, read before each step because
mjlab zeroes the sums at reset. Setup: CPU on the Pi, 32 envs, seed 2001, domain randomisation and
final curricula on, plain task and backlash twin. Networks are Pollen's (sit-stand is v6, the
file sha256 a6f102d9…).

| task | network | ended by | three largest penalties (reward per s) |
|---|---|---|---|
| SitStand-Flat | alpha_sitstand.onnx | time_out 1.00 | posture_pose_l1 -0.100; action_rate_l2 -0.051; descent_speed -0.042 |
| SitStand-Flat-Backlash | alpha_sitstand.onnx | time_out 1.00 | posture_pose_l1 -0.095; action_rate_l2 -0.056; descent_speed -0.042 |
| GroundPick-Flat | alpha_ground_pick.onnx | time_out 1.00 | neck_vel_descent -0.038; feet_flat -0.026; action_rate_l2 -0.020 |
| GroundPick-Flat-Backlash | alpha_ground_pick.onnx | time_out 1.00 | neck_vel_descent -0.035; feet_flat -0.026; action_rate_l2 -0.025 |
| StandUp-Flat | alpha_stand.onnx | time_out 1.00 | head_pose_bias -0.215; body_ang_vel -0.112; action_rate_l2 -0.094 |
| StandUp-Flat-Backlash | alpha_stand.onnx | time_out 1.00 | head_pose_bias -0.212; pose_stand_l1 -0.100; body_ang_vel -0.095 |
| Roulade-Flat | roulade.onnx | time_out 1.00 | self_collisions -0.139; action_rate_l2 -0.112; dof_pos_limits -0.093 |

## What it shows
- **Nothing falls.** Every episode of every skill ran to its time-out.
- **Backlash barely matters.** Where a twin exists the numbers are near-identical, as they were
  for the kick (k001 design). On this evidence, backlash robustness is not where these skills
  are weak.
- **Penalties are small everywhere.** The largest per-second costs are:
  - roulade: self-collision −0.14 and joint limits −0.09;
  - stand-up: head-pose bias −0.21;
  - sit-stand: posture L1 −0.10.

## Candidates, earmarked (not run)
- **A gentler roulade.** Its self-collision and joint-limit costs are the largest hardware-
  flavoured penalties here. On a real duck those are parts and servos hitting things. A k001-style
  fine-tune would charge those harder and require `roulade_landing_composite` to be no worse.
- **Skill-specific success measures,** the way `skill_eval.kick` measures a kick: did the roll
  land, did the mouth touch, did the sit hold. The reward profile says what is penalised, not
  whether the job got done.

## Not claimed
32 envs and one seed is a look, not a result. Simulation only.
