# Step 4 design: grounding the duck with the iPhone camera and a vision model

Written 2026-09-25. Design only: nothing here has been measured yet.

## The idea
The walking policies are blind: they read gyro, gravity, joints and a velocity command, nothing
else. The planner (p002) turns words into skills, but "walk to the red ball" needs to know where
the ball is. The iPhone in Duck Studio already has a camera, ARKit and on-device vision. It
becomes the duck's eyes, and the duck's policy stays exactly as it is: a command stream.

```
 camera frame ──► perception (on phone) ──► facts: {target, bearing, distance, confidence}
                                                   │
 "walk to the red ball, then peck it" ──► planner ──► plan with `target` steps
                                                   │
                                  closed loop at ~10 Hz: bearing → wz, distance → vx
                                                   │
                                        duck policy (unchanged) ◄── twist command
```

## Three layers, cheapest first
1. **Apple Vision and ARKit (no LLM, fast, on device).** Object detection and saliency,
   `VNDetectHumanBodyPose3DRequest` for people, ARKit planes and anchors for distance. That gives
   structured facts, which are what a controller needs.
2. **A vision-language model for open vocabulary ("the red ball", "the blue mug").** Gemma 4 E4B
   takes images through its mmproj (`gemma-4-E4B-it-mmproj.gguf`, already in ~/models), with the
   answer schema-constrained like the planner's: `{found, bearing_deg, distance_m, confidence}`.
   On the phone, the equivalents are Apple's on-device Foundation Models and MLX VLMs. The VLM
   runs at a few Hz at most, so it *finds* the target and ARKit *tracks* it between VLM calls.
3. **Mimicry.** The same body-pose stream, read from a person or a video:
   - first, **pose → skill sequence** (classification into the existing skills: a person
     crouching is `sit_or_stand`, bobbing their head is `peck_ground`). That is a Decide/router
     job, cheap, and it produces a plan the user can edit in the flow-chart editor;
   - later, **pose → reference motion → new skill**. Retarget the human keypoints to the duck's
     joints as a reference trajectory, train a tracking policy on it (DeepMimic-style reward) with
     `duckbatch hf-job`, and distil it into the multi-skill student (b004's mechanism).

## Changes this implies
- Planner schema: an optional `target` (a free-text noun phrase) on walk, turn and peck steps,
  plus a new `approach` skill that means "close the distance to the target". The skill enum stays
  closed, while the target is open vocabulary, grounded by perception rather than by the LLM.
- A `duck-percept/0` record `{t, target, bearing_deg, distance_m, confidence, source}`, logged
  beside duck-feedback, so grounding failures become training data, with consent as before.
- Controller: a proportional law with dead-zone compensation. b003 matters here: a policy that
  ignores 0.1 m/s commands cannot creep up on a target, so the dead band must go first.

## First measurable test (g001, to pre-register before running)
In simulation, not on the phone: render the arena scene with a coloured ball at a random
position, send the frame to Gemma 4 through the mmproj, and score the bearing error and
found/not-found against the true position. That scores the VLM layer on its own, with ground
truth, on this laptop, before any iOS work.
