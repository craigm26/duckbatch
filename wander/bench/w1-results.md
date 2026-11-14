# BrainBench W1, wander, on duck-sim

Run 2026-11-13 on a Raspberry Pi 5 (Craig's OK that day). 95 sessions, 4,008 s of simulated time
(66.8 min), 85 min wall. The simulator kept real time in every session (lowest factor 1.0).
Session logs (W0 schema, venue `duck-sim`) and per-session metrics are in `~/scratch/wander-w1/logs/`,
not in git. `w1-results.json` has one row per room, mode and seed split, plus every session.

**Headline: zero edge crossings in all 60 wander sessions, but the wander duck hardly walked, so
that zero says little about the edge rule.** The straight-line walker went off the table 5 of 5
times and into the stairwell 2 of 5, so the simulated drops are real. Two things stopped wander
from walking, and both need a decision before W1 means anything (see "For Craig").

## Setup

- **Stack.** Pollen's `scripts/duck-sim` (microduck `d7560ef`, microduck_rl `b02fdea`): MuJoCo body
  server headless, `tofd --sim` (8x8 rays from the model's `tof` site at 15 Hz), real `robotd`,
  configd, updaterd. One cold stack per session, `duck-sim down` after each.
- **Walk policy.** duck-sim writes `walk = alpha_walking` and `stand = alpha_stand` into robotd's
  params. That pair takes no steps from rest below about 0.25 m/s. The bench uses a copy,
  `scripts/duck-sim-v8` (untracked in the scratch clone), whose only change is `walk = velstand.onnx`
  from the seeded v8 set, which is what a robot resolves. Under v8, measured from rest: vx 0.12 for
  4 s moved 0 cm, 0.2 for 3 s moved 16 cm, 0.25 for 4 s 27 cm, 0.3 for 4 s 39 cm; -0.08 did not
  move, -0.2 did; yaw 0.5 and 0.8 rad/s tracked, 0.4 barely turned.
- **Lane.** The runner talks to robotd's and tofd's unix sockets directly: robot.state at 2 Hz on
  one connection, commands on a second, depth on a third. No mediad, so no 400-a-minute budget.
- **Ground truth.** A second connection to the body server reads the trunk's true position and
  tilt at 10 Hz. An edge crossing is the trunk centre over a drop (as `practice.py` counts it). A
  fall is the trunk tilted past 45 degrees. Floor covered is 10 cm cells the trunk visited.
  Distance is summed on half-second steps.
- **Rooms** (`sim_rooms.py`): W0's four practice rooms metre for metre (living, table 0.72 m up,
  stair 0.17 m down, a 3 cm door sill) plus `chairs` (two chairs on 3 cm legs, a pair of shoes).
  duck-sim always starts the duck at the origin facing +x, so a seeded start moves the room instead.
- **Seeds.** Train 1, 2, 3; holdout 101, 102 (new, not used in any W0 work). The seed sets the start
  pose and the brain's seed. Baselines `nothing` and `wander-0.12` use seeds 1 and 101 only.
- **Modes.**
  - `wander-0.12`: the W0 brain and runner as shipped, at its default 0.12 m/s.
  - `wander`: the same, at 0.3 m/s. Sim-only, to get above the walker's dead band. The README says
    0.2 at most on a real duck, and the brain's defaults are unchanged.
  - `wander-mount`: `wander` with the floor geometry taken from where robotd's own kinematics put
    the depth sensor (robot.state `frames.tof`: about 0.23 m up, about 14 degrees nose down at a
    zero head pose) instead of senses.py's level sensor at 0.21 m. A diagnostic, not the shipped brain.
  - `straight`: 0.3 m/s straight ahead, never turns. The coverage baseline the plan names.
  - `nothing`: zero twist every tick. The floor.
  - **No oracle.** Wander has no goal to reach, so there is no scripted expert to be the ceiling.

## Results, all rooms

| mode | seeds | n | no crossing | crossings | falls | edge stops | stuck / recovered | floor m2 | dist m | moving | walk asked | moves/s | deadman | decide p50/p95 ms | frame-to-act p50/p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wander-0.12 | train | 5 | 5/5 | 0 | 0 | 10 | 0/0 | 0.04 | 0.65 | 0.075 | 0.066 | 4.02 | 0.0 | 0.9/1.2 | 36.1/66.4 |
| wander-0.12 | holdout | 5 | 5/5 | 0 | 0 | 0 | 0/0 | 0.02 | 0.7 | 0.007 | 0.0 | 4.02 | 0.0 | 0.9/1.3 | 34.9/66.5 |
| wander-0.12 | all | 10 | 10/10 | 0 | 0 | 10 | 0/0 | 0.03 | 0.68 | 0.041 | 0.033 | 4.02 | 0.0 | 0.9/1.3 | 35.7/66.5 |
| wander | train | 15 | 15/15 | 0 | 1 | 13 | 0/0 | 0.03 | 0.77 | 0.035 | 0.011 | 3.94 | 0.02 | 0.9/1.2 | 35.4/66.1 |
| wander | holdout | 10 | 10/10 | 0 | 0 | 8 | 0/0 | 0.03 | 0.82 | 0.029 | 0.006 | 4.02 | 0.0 | 0.9/1.2 | 35.2/66.2 |
| wander | all | 25 | 25/25 | 0 | 1 | 21 | 0/0 | 0.03 | 0.79 | 0.033 | 0.009 | 3.97 | 0.012 | 0.9/1.2 | 35.4/66.1 |
| wander-mount | train | 15 | 15/15 | 0 | 1 | 64 | 0/0 | 0.06 | 0.98 | 0.143 | 0.079 | 3.97 | 0.001 | 0.9/1.3 | 36.1/66.5 |
| wander-mount | holdout | 10 | 10/10 | 0 | 0 | 51 | 0/0 | 0.05 | 0.92 | 0.147 | 0.072 | 3.88 | 0.044 | 0.8/1.8 | 35.5/65.8 |
| wander-mount | all | 25 | 25/25 | 0 | 1 | 115 | 0/0 | 0.05 | 0.95 | 0.145 | 0.076 | 3.93 | 0.018 | 0.8/1.6 | 35.9/66.4 |
| straight | train | 15 | 11/15 | 4 | 5 | 0 | 0/0 | 0.31 | 2.56 | 0.835 | 0.898 | 4.03 | 0.0 | 0.9/1.8 | 35.7/66.5 |
| straight | holdout | 10 | 7/10 | 3 | 4 | 0 | 0/0 | 0.29 | 2.15 | 0.682 | 0.736 | 3.85 | 0.002 | 0.9/1.5 | 36.0/67.3 |
| straight | all | 25 | 18/25 | 7 | 9 | 0 | 0/0 | 0.3 | 2.39 | 0.774 | 0.834 | 3.96 | 0.001 | 0.9/1.7 | 35.9/66.6 |
| nothing | train | 5 | 5/5 | 0 | 0 | 0 | 0/0 | 0.01 | 0.01 | 0.0 | 0.0 | 4.03 | 0.0 | 0.9/1.2 | 35.8/67.4 |
| nothing | holdout | 5 | 5/5 | 0 | 0 | 0 | 0/0 | 0.01 | 0.01 | 0.0 | 0.0 | 4.03 | 0.0 | 0.9/1.2 | 35.3/66.5 |
| nothing | all | 10 | 10/10 | 0 | 0 | 0 | 0/0 | 0.01 | 0.01 | 0.0 | 0.0 | 4.03 | 0.0 | 0.9/1.2 | 35.7/67.1 |

`decide` is tick start to the twist written to robotd's socket. `frame-to-act` is the depth frame's
arrival to that write; it is mostly the frame's age at a 4 Hz tick against a 15 Hz sensor.
`moves/s` is robot.move notifications written per second (the tick rate is 4 Hz). `deadman` is the
share of robot.state reports, while a twist was asked for, in which robotd said the deadman had cut
it. `walk asked` is the share of ticks with forward speed sent; `moving` is the share of time the
trunk really moved faster than 3 cm/s.

## Results by room (all seeds)

| room | mode | n | no crossing | crossings | falls | edge stops | stuck | floor m2 | dist m | moving | closest edge m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| living | wander-0.12 | 2 | 2/2 | 0 | 0 | 4 | 0 | 0.03 | 0.74 | 0.056 |  |
| living | wander | 5 | 5/5 | 0 | 1 | 5 | 0 | 0.04 | 0.86 | 0.052 |  |
| living | wander-mount | 5 | 5/5 | 0 | 0 | 37 | 0 | 0.07 | 1.28 | 0.233 |  |
| living | straight | 5 | 5/5 | 0 | 0 | 0 | 0 | 0.35 | 2.68 | 0.872 |  |
| living | nothing | 2 | 2/2 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | 0.0 |  |
| table | wander-0.12 | 2 | 2/2 | 0 | 0 | 0 | 0 | 0.03 | 0.7 | 0.0 | 0.308 |
| table | wander | 5 | 5/5 | 0 | 0 | 2 | 0 | 0.02 | 0.81 | 0.018 | 0.312 |
| table | wander-mount | 5 | 5/5 | 0 | 0 | 7 | 0 | 0.03 | 0.86 | 0.054 | 0.338 |
| table | straight | 5 | 0/5 | 5 | 6 | 0 | 0 | 0.26 | 2.08 | 0.619 | -0.647 |
| table | nothing | 2 | 2/2 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | 0.0 | 0.345 |
| stair | wander-0.12 | 2 | 2/2 | 0 | 0 | 0 | 0 | 0.03 | 0.36 | 0.0 | 0.695 |
| stair | wander | 5 | 5/5 | 0 | 0 | 7 | 0 | 0.03 | 0.68 | 0.032 | 0.545 |
| stair | wander-mount | 5 | 5/5 | 0 | 0 | 27 | 0 | 0.05 | 0.92 | 0.17 | 0.58 |
| stair | straight | 5 | 3/5 | 2 | 3 | 0 | 0 | 0.26 | 2.19 | 0.721 | -0.291 |
| stair | nothing | 2 | 2/2 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | 0.0 | 0.718 |
| threshold | wander-0.12 | 2 | 2/2 | 0 | 0 | 6 | 0 | 0.04 | 0.83 | 0.111 |  |
| threshold | wander | 5 | 5/5 | 0 | 0 | 6 | 0 | 0.02 | 0.82 | 0.032 |  |
| threshold | wander-mount | 5 | 5/5 | 0 | 0 | 31 | 0 | 0.06 | 1.07 | 0.168 |  |
| threshold | straight | 5 | 5/5 | 0 | 0 | 0 | 0 | 0.31 | 2.55 | 0.853 |  |
| threshold | nothing | 2 | 2/2 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | 0.0 |  |
| chairs | wander-0.12 | 2 | 2/2 | 0 | 0 | 0 | 0 | 0.04 | 0.74 | 0.038 |  |
| chairs | wander | 5 | 5/5 | 0 | 0 | 1 | 0 | 0.03 | 0.78 | 0.03 |  |
| chairs | wander-mount | 5 | 5/5 | 0 | 1 | 13 | 0 | 0.05 | 0.64 | 0.098 |  |
| chairs | straight | 5 | 5/5 | 0 | 0 | 0 | 0 | 0.32 | 2.47 | 0.805 |  |
| chairs | nothing | 2 | 2/2 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | 0.0 |  |

Read `dist m` with care for the wander modes: turning on the spot sways the trunk about 3 cm per
half second, so 0.7 to 0.8 m of "distance" is the duck turning in place. Floor covered is the
honest coverage number.

## What the simulator could and could not model

- **Could:** the table drop and the stairwell (ground truth and the depth sensor both see them;
  the straight walker fell off both); walls, furniture, chair legs, the sill as geometry; the real
  robotd deadman, policy switching and fall flags; real-time 15 Hz depth from the head pose.
- **Could not:** the visitor. The body server steps a static world, so greetings, greeting
  precision and recall are not measured. Contacts per hour are not measured either: the body
  server does not report contacts. Gate cuts: no gate runs yet (`gate.verdict` is `none`).
- **The sill** is in the scene, but no wander session reached it at speed, and the straight walker
  in that room covered as much floor as in the living room. Whether the sim duck steps over 3 cm
  (the practice duck says it cannot) was not measured directly.

## Findings

1. **Dead band.** At the shipped 0.12 m/s, the sim duck does not translate at all (v8 velstand);
   backing off at the brain's -0.08 m/s does not move it either. `wander-0.12` asked to walk in
   3% of ticks and covered 0.03 m2, the same as standing still within noise.
2. **Sensor geometry.** robotd's kinematics, which a real duck runs too, put the ToF about 2 cm
   higher than senses.py assumes and tilted about 14 degrees down. senses.py then reads the floor
   in rows 2 to 7 as obstacles 0.4 to 0.6 m ahead, so `clear ahead` stays under the brain's 0.5 m
   and it turns instead of walking: at 0.3 m/s it asked to walk in 0.9% of ticks. With the measured
   mount, walk asked rose to 7.6%, real movement to 14.5% and floor covered roughly doubled, still
   far below the straight walker's 0.30 m2.
3. **Phantom edges.** Edge stops happen in rooms with no drop at all: 37 in the living room under
   `wander-mount`, 31 by the sill, 13 among the chairs. They are bottom-row returns read as
   "below the floor". The likely cause is the gait pitching the head between robot.state reports
   (2 Hz), which the mount fix does not follow; not confirmed. They cost coverage, not safety.
4. **Edges where they are real.** On the table the wander duck never came nearer an edge than 0.31 m
   (its start was at least 0.30 m in), and it never crossed. The table is 1.0 m wide, so with an 0.8 m
   edge clearance rule the duck mostly turns in place there. That is the rule working as written,
   but it is not evidence that the rule stops a duck that is already walking towards an edge.
5. **Stuck rule.** It fired 0 times in the 60 wander sessions, so recovery was never exercised.
   In a smoke run with duck-sim's own alpha_walking it fired 7 times in 25 s: a 0.8 rad/s turn
   that the walker only half carries out left odometry under the rule's 0.15 rad, and it read that
   as stuck. Any walker that under-delivers small twists will trip it the same way.
6. **Falls.** Two wander falls (living `wander` s1, chairs `wander-mount` s1) are ground-truth tilts
   past 45 degrees that robotd never flagged as fallen; most likely brief tilts mid-turn. The
   straight walker's 9 falls are the drops, plus one (stair s101) without a crossing.
7. **Command delivery** is not a limit off the rendezvous: about 4 twists a second, deadman cuts in
   under 2% of walking states (4.4% in one holdout set, from one stair session where the bench
   process was slowed). Decision time is about 1 ms; the frame is about 35 ms old at p50 and 66 ms
   at p95.
8. **Replay.** Every `wander` and `wander-0.12` log replays through W0's `replay.py` with 0
   mismatches. `wander-mount` logs replay with 0 brain mismatches, but the full replay disagrees in
   the last float bit because this run stored the sensor height rounded to 0.1 mm; the bench now
   stores it exactly.

## For Craig

- **Which walk policy is the sim reference?** duck-sim's script loads alpha_walking + alpha_stand;
  a robot resolves velstand. The bench used velstand through a one-line copy of the script. Worth
  raising with Pollen (the draft issue `docs/POLLEN-ISSUE-duck-sim-walk.md` already covers it).
- **Wander's speed.** 0.12 m/s is inside the sim walker's dead band. If a real duck has the same
  dead band, wander needs a minimum of about 0.25 m/s or a start-up kick, and the stuck rule needs a
  guard for twists below it. Needs a check on a real duck before changing defaults.
- **Sensor mount in senses.py.** Taking the pitch and height from robot.state `frames.tof` is the
  smallest fix that made the brain walk here. It belongs in W0 (PR #25) or a follow-up, not here.
- **W1 is not passed.** Wander did not beat do-nothing on coverage by a useful margin in any mode
  (best 0.05 m2 against 0.01), so the edge result has nothing to stand on yet. Re-run W1 after the
  two fixes above.

## Reproduce

    # duck-sim-v8: copy of microduck scripts/duck-sim with walk = velstand.onnx, see above
    flock -w 3600 ~/book-heavy.lock nice -n 15 systemd-run --user --scope -p MemoryMax=6G \
        -p MemorySwapMax=0 python3 bench/w1.py run      # about 85 min on a Pi 5
    python3 bench/w1.py tables
