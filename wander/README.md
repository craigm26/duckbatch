---
title: microduck wander
emoji: 🦆
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
hf_oauth: true
license: apache-2.0
tags:
  - microduck
  - robotics
  - autonomy
short_description: A Microduck that wanders, keeps clear and says hello.
---

# microduck wander

A Microduck that walks about on its own, keeps clear of what is in its way, and greets whoever
comes close. It runs two ways: against a **practice duck** in a simulated room, right here, or
against **your duck**, signed in with Hugging Face and reached through Pollen's rendezvous.

This is a community app. It complements Pollen's microduck Spaces and the official app; it does
not replace any of them, and everything it asks the robot to do goes through the same control lane
and the same methods their own Spaces use.

## How it works

```
robot.state (2 Hz) ─┐                         ┌─► robot.move   every tick, 4 Hz
tof.frame  (15 Hz) ─┼─► senses ─► brain ─► runner ─► robot.head   when the gaze changes
                    │                         ├─► robot.do     curious / yes / excited
                    │                         └─► robot.sound  wheee, held
```

- **`senses.py`** turns the 8x8 depth frame into two readings. *Room ahead* is the nearest thing
  inside a corridor one duck wide, from any column, with the floor set aside by its geometry. A
  *visitor* is something close that was not close a moment ago, seen while the duck holds still.
  It also keeps obstacle points for 8 s in odometry coordinates, because a table leg the duck has
  walked past is out of the sensor's 45 degree view and still beside it.
- **`brain.py`** is a pure state machine: Chill, LookAround, Wander, TurnInPlace, Greet, Dance,
  named after Pollen's `docs/ideas/autonomous_behavior.md`. It first ran inside Pollen's Arena,
  where it choreographed a four-duck Got a Talent act closed loop in the Arena's own simulator.
- **`runner.py`** drives a live duck: twist every tick (robotd stops a twist older than 500 ms),
  the robot's own emotions for greetings, and nothing when it cannot see.
- **`session_log.py`** is the session log's schema: one JSON line per tick, with what the duck read,
  what the brain made of it, and what was sent. `replay.py` plays a log back and checks it.

## Edges and getting stuck

- **Where the sensor is.** The floor test takes the depth sensor's height and pitch from
  `robot.state`: `frames.tof` (robotd's kinematics, from the measured head joints) turned by the
  trunk's `imu.quat`, on top of `odom.position[2]`. A duck standing on duck-sim holds it at 0.233 m,
  10.6 degrees nose down. A state without `frames.tof` gets those standing values, tipped by the
  head pitch last sent. (Before this, the floor test assumed a level sensor at 0.21 m, and read a
  standing duck's own floor as obstacles 0.4 to 0.6 m ahead; BrainBench W1.)
- **Edge.** The depth sensor's two bottom rows see the floor 0.40 and 0.51 m ahead from the standing
  mount. A return there more than 5 cm below the floor, or no return, is a drop: a table edge or a
  stair. With one in its path nearer than 0.6 m, the duck stands still for half a second, walks
  backwards for a second, and turns away from the drop. It does not set off towards one nearer than
  0.8 m. Edges are remembered for 8 s, like obstacles, so one it turned from still counts.
- **Which pitch a frame was taken at.** `robot.state` arrives at 2 Hz and depth at 15 Hz, and the
  head bobs with the gait, so the state's pitch can be several degrees off the frame's (in W1, up
  to 10 degrees, worst when it halts). A frame taken with the head higher than the state says reads
  the floor as a drop: that is what W1's phantom edge stops in rooms with no drop were. So the edge
  test reads each frame at its own pitch, taken from the floor in its two bottom rows, when both
  rows agree on one within 1 degree (and it is within 12 of the state's). A tipped head moves both
  rows' floors by the same angle; a drop does not (row 7 looks down more steeply and needs 1.5 to
  3 degrees more to explain away), so a drop keeps the state's pitch and still stops the duck.
- **Stuck.** A twist sent and odometry not moving (3 cm, or 0.15 rad) for 1.5 s: it backs off the
  same way and turns the other way from its last turn.
- These are tested on the practice duck only. The thresholds come from the sensor's geometry and
  the floor test's 4 cm tolerance, not from a real duck. A dark or shiny floor that returns nothing
  will read as an edge: the duck stops more, it does not walk on.

## Session logs

Each line is one tick: the depth frame packed losslessly (base64 of 64 uint16 distances and 64
uint8 statuses), the pose and fall flags, the senses, the brain's state and command, the twist and
head sent, and the envelope gate's verdict (`none` here, as no gate runs in this folder yet). The
first line names the schema, its version, the venue (`practice`, `duck-sim`, `real`) and the brain's
seed. Set `WANDER_LOG_DIR` and the Space writes one file per run. `tests/golden/` holds four short
practice sessions (a visitor, a table, a stair, a door sill) that the brain must reproduce exactly,
command for command: the conformance fixtures for a Swift port.

    python3 replay.py tests/golden/*.jsonl
    python3 sim_check.py --golden          # rewrite them after a deliberate brain change

## Why it stops now and then

Driven through this Space, the duck reaches the page over Pollen's rendezvous, and the robot
forwards at most 400 notifications a minute on a control lane (`mediad/src/relay.rs`), in a fixed
window. The limit keeps the robot on its owner's listing: the rendezvous rate-limits every peer.
The depth sensor sends 15 frames a second with no setting to slow it, so it spends the allowance in
about 25 s and then nothing arrives until the window turns over.

The runner treats that as blindness: no fresh depth frame in 0.6 s, no walking. So the duck walks
for about half of each minute and stands for the rest. The practice duck keeps the same limit, so
what you see there is what a real one does.

A `hz` on `tof.stream`, as `robot.subscribe` already has, would let a remote client ask for 2
frames a second and walk the whole minute. That is a proposal for Pollen, not something this Space
works around.

## Safety

- The duck walks at 0.12 m/s by default (0.2 at most) and turns at 0.8 rad/s at most.
- It stops when you press stop, when the page closes, when the lane drops, when the session time
  runs out (5 minutes by default), and whenever its depth frames go stale. robotd's own 500 ms
  deadman backs every one of those.
- Fallen, picked up or limp, it holds still until the robot reports itself fine for 2 s.
- Clear a patch of floor, keep it away from stairs and table edges, and stay in the room.

## Tested

- 45 tests (`tests/`): blind means still, the floor is not a wall, a corner beside the path
  counts, a wall the duck turned to is not a visitor, the practice duck keeps the lane budget; and
  for W0, a drop or a missing floor is an edge, the brain stops, backs off and turns away, no
  practice session on the table or by the stair goes over (16 two-minute sessions), and with the
  edge rule switched off the duck does walk off the table; a duck held by a sill backs off and
  moves on; logs round-trip, and a practice run and the golden traces replay to the same commands.
- `sim_check.py` runs three-minute sessions in every practice room (`living`, `table`, `stair`,
  `threshold`), with the lane limit and without. In the living room the brain's choices are the
  same as before the edge and stuck rules.
- Not yet on a physical duck at the time of writing.

## Credits

`control.py`, `rendezvous.py` and `wire.py` are Pollen Robotics' (`spaces/shared` in
[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck), Apache-2.0), unchanged.
The Dockerfile follows their vision-demo Space's. The brain, the senses, the runner and the practice
duck are by [craigm26](https://huggingface.co/craigm26).
