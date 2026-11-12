# Wander: where it goes next

Wander runs today in this Space against a practice duck, and against a real duck through Pollen's
rendezvous. **Nothing has run on a physical duck yet.** This is the plan to change that, in order.
Each step lands here first, in public, and is measured before the next.

## 1. Edge stop (before anyone puts a duck on a table)

The senses set the floor aside but do not yet notice it *missing*. On a table the duck would walk
off. The bottom rows of the depth frame see the floor about 0.6 m ahead; when that return jumps far
or vanishes, the duck stops, backs off and turns away. A table and a stair go into the practice
room, with a test for each. Until this lands: **floor only, never a table.**

## 2. Stuck recovery

Twist commanded, odometry not moving for 1.5 s: back off and turn. The classic wander failure,
caught cheaply.

## 3. A session log

One JSON line per tick: what the duck sensed, the brain's state, the twist and head sent, and the
raw 8x8 depth frame. Same format from the practice duck, a simulator and a real duck, so every run
is a replay fixture and every change is judged on the same runs.

## 4. Five rooms in a simulator

Pollen's duck-sim runs the real robot daemon with a simulated depth sensor. Five fixed rooms (open
floor, chair legs, a corridor, a table top, a visitor walking up), ten seeded starts each, three
minutes each, scored on:

- edge crossings and falls (must be zero)
- contacts per hour
- share of time moving, and area covered
- stuck events
- greetings that were real visitors

A do-nothing duck and a straight-line walker are scored beside it, so "better" means something.

## 5. Real ducks, measured by anyone who has one

The measurement can be ours or yours; the protocol is the same, and a tester's run counts.

1. Clear about 2 m by 2 m of floor, away from stairs and table edges. Stay in the room.
2. In the Space: sign in, find your duck, connect, tick "record this session", top speed 0.12.
3. Five sessions of three minutes:
   - open floor
   - two chairs in the patch
   - along a wall
   - a person walking up twice
   - a rug or dark floor
4. Each log goes to a dataset in *your* Hugging Face account, private unless you share it. Depth
   frames are 8x8 distances, not pictures.
5. Tell us what the log cannot show: a bump, a near-miss, a greeting that looked wrong.

Those logs retune what the practice room guessed: head height for the floor geometry, the speed
the walk really reaches at 0.12 m/s, odometry drift, and false visitors.

## 6. Presets

Once the base holds on real ducks:

- **Explore:** heads for floor it has not covered.
- **Patrol:** a loop.
- **Nap spot:** goes back to where it started.
- **Companion:** keeps a person in view. Camera, with consent.
- **Social:** knows a duck it has met by a stable id, not by distance.

Each preset is scored on the five rooms before it is offered.

## 7. Asks to Pollen

- A `hz` on `tof.stream`, as `robot.subscribe` has. Over the rendezvous the depth sensor alone
  spends the robot's 400-a-minute allowance in about 25 s; at 2 Hz a remote app could see all
  minute. The session logs are the evidence.
- A way for an app to say "a brain is driving", so a pad, a Space and an app do not fight over the
  same twist.
- Later, once the real-duck numbers hold: wander as an on-robot skill (`robot.do wander`).
