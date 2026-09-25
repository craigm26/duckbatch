---
title: Microduck walker arena
emoji: 🦆
colorFrom: yellow
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: Drive every Microduck walker at once, one set of controls.
tags: [microduck, robotics, reinforcement-learning, distillation, mujoco]
---

Every valid Microduck walking policy on one field, driven by the same keyboard, gamepad or
touch input: Pollen's teacher beside the smaller students
[duckbatch](https://github.com/craigm26/duckbatch) distilled from it. Each duck has its own
MuJoCo world, reset identically, so the only difference between lanes is the network. Push them
all at once and see which one goes down.

- `?move=org/repo[,org/repo]` adds Hub policies (Pollen's `manifest.json` + `policy.onnx`
  format); `?only=1` shows only those.
- A policy has to pass the same checks Pollen's simulator makes before it drives a duck.
- The scene is plain MuJoCo from [duckbench](https://github.com/craigm26/duckbench), not the
  mjlab training environment. The measured comparison is in duckbatch's notes. Simulation only.

The code (duckbatch's `arena/` and duckbench's runtime) is Apache-2.0. The robot model and
policies are [Pollen Robotics](https://github.com/pollen-robotics/microduck)'. Not affiliated
with Pollen Robotics.
