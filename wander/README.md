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

- 9 unit tests (`tests/`): blind means still, the floor is not a wall, a corner beside the path
  counts, a wall the duck turned to is not a visitor, the practice duck keeps the lane budget.
- 16 three-minute sessions against the practice duck, half with the lane limit and half without:
  no falls, and greetings in half of them. `sim_check.py` reruns them.
- Not yet on a physical duck at the time of writing.

## Credits

`control.py`, `rendezvous.py` and `wire.py` are Pollen Robotics' (`spaces/shared` in
[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck), Apache-2.0), unchanged.
The Dockerfile follows their vision-demo Space's. The brain, the senses, the runner and the practice
duck are by [craigm26](https://huggingface.co/craigm26).
