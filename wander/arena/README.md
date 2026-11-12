# The talent act's choreography tools

How `craigm26/microduck-duckbatch-social` was made for the Arena's Got a Talent stage.

- `stage.py`: four ducks on a kinematic model of the 2.6 x 3.8 m stage, and the cast.
- `live_stage.py`: the same brains run closed loop inside the Arena's own simulator
  (`microduck-arena` 0.6.0, seed 1), sensing every duck's true pose ten times a second.
  Its keyframes are the published timeline, `live-timeline.json`.

`arena score policies/b003e.onnx --event got-a-talent --timeline live-timeline.json` replays the
run to within 3.9 cm of the closed-loop trace, with no falls and no two ducks closer than 0.53 m.
Both scripts import `brain.py` from the folder above.

The greetings' head moves came from Pollen's `pollen-robotics/microduck-emotions` dataset
(revision `b9b49d9fa06c73b7a1948f2bf7a31f5ad0e42e67`): to rerun the act exactly, put its
`curious`, `yes`, `excited` and `laugh` JSON in `arena/emotions/`.
