# Microduck Studio (iOS) integration: what blocks duckbatch students, and a plan

Read-only survey of `craigm26/duckkit`, `craigm26/duck-studio` and `craigm26/microduck-com`,
2026-09-24. Nothing here has been changed in those repos yet.

## Today

- **Listing works.** The catalogue (`StudioKit/PolicyCatalogue.swift` 144-150) queries
  `huggingface.co/api/models?filter=microduck`, and duckbatch repos carry that tag, so they
  appear. The card shows the manifest `description`, which already says "N-parameter …
  distilled from Pollen's default walker".
- **Installing fails.** `DuckPolicy.load` (`duckkit/Sources/DuckKit/DuckPolicy.swift` 280-351)
  hard-codes the 9-op graph `Sub, Div, Gemm, Elu ×3, Gemm` (line 284) and
  `expectedWidths = 61-512-256-128-14` (line 166). A 2-hidden-layer student fails the op check;
  a 3-layer 256-128-64 fails the width check. It gets filed as a refused entry.
- **The phone bench is fixed-width too.** `phonebench/assets/policyforward.mjs` 29-33 has the
  same hard-coded widths; it is copied from duckbench and digest-checked.
- **Manifest fields are dropped.** `eval.student`, `eval.teacher_same_eval`, `training.teacher`
  and `training.method` are ignored (`PolicyManifest.swift` 88-142). `command.twist` must be a
  list of strings: duckbatch now writes it that way (it was a string, which decoded as empty).
- **Nothing times inference on the phone.** No code measures `DuckPolicy.infer` on the iPhone,
  and `PhoneBenchReport.speedIsUnmeasuredOnAPhone` says as much.

## Plan (proposed, in order)

| # | Where | Change | Size |
|---|---|---|---|
| 1 | duckkit | `load`: accept `Sub, Div, (Gemm, Elu)×k, Gemm`, k in 1..4, first in = 61, last out = 14, chained widths, caps on width and total params. Keep `expectedWidths` as the alpha shape. Update the writer to check chained widths. Add a duckbatch student fixture and golden vectors against onnxruntime. | medium |
| 2 | duckkit | Fingerprint v2: prefix layer widths to the canonical bytes for non-alpha shapes (v1 bytes carry no shapes, so two architectures could collide). Official fingerprints stay v1. | small |
| 3 | StudioKit | `PolicyBlend.mix` and the fold writer refuse mismatched shapes (today they rely on the loader). Rewrite the `PolicyReport` refusal text. The `hidden_narrowed.onnx` refusal fixture would now load, so pick another defect. | small |
| 4 | StudioKit | Decode a distillation block (teacher, method, student vs teacher metrics). Show the loaded parameter count, and "distilled from velstand, 26k params, walks within teacher noise". | small–medium |
| 5 | StudioKit | On-device latency: a `ContinuousClock` loop over `infer`, reporting p50/p95 µs for the student next to the teacher. This is the iPhone number that duckbatch's laptop bench cannot give. | small |
| 6 | duckbench → phone bench | `policyforward.mjs` takes widths from the v2 header, then re-vendor with fresh digests. | medium |
| 7 | docs | PLAN.md 305-322 (the App Review argument) promises "fixed widths (61→512→256→128→14)". Reword to "one bounded op pattern, capped widths", and update RISKS, GATES, README and skills before the next submission. | small |

Guards to update: duckkit `DuckPolicyTests`, `DuckPolicyIntrospectionTests`,
`DuckPolicyBytesTests`, `DuckPolicyFingerprintTests`, `DuckPolicyWriter*Tests`,
`golden_policies.json`. duck-studio `make_refusal_corpus.py`, `PolicyReportTests`,
`PolicyBlendTests`, `WeightSearchTests` ("197,774"), `PhoneBenchReportTests`, and
`check_phonebench_fresh.sh`. `check_no_studio_math.sh` bans the literal `61` in the app target,
so the shape rules belong in the kit.

## Progress (2026-09-24)

- duckkit **v1.36.0** (craigm26/duckkit#1, merged): loader and writer take students of the alpha
  graph; identity v2 carries the shape. 385 tests, 0 failures, Linux x86_64, Swift 6.4.
- craigm26/duck-studio#1 (draft; the SwiftUI screens need Xcode): blend shape refusal, refusal
  corpus updated, the "Against its teacher" card from the manifest's comparison and Jev
  verdict, and `PolicyTiming` (on-device inference p50/p95/p99, same-device comparisons only).
  StudioKit: 2,552 tests; the only failures are 2 pre-existing `WeightSearchTests` that read a
  fixture from an absolute path on the Pi.

## Future direction: a plan editor (Craig, 2026-09-24)

When someone makes a new plan, sequence or motion, it should be customisable in a flow-chart
style visual editor and viewer, in the control and editor tabs. That is the place to get
creative with training new motions, plans and sequences:

- **Plain-language requests.** "Walk to the ball and kick it" becomes a proposed graph the user
  then edits. The router (GLiNER2.5-Decide, whose strength is intents) proposes; it does not
  execute.
- **Mimicry** from YouTube or other videos, and from real life through AR on iOS.
- **On-device models.** New MLX and Apple models in iOS 27 and on newer hardware may be powerful
  enough to run the router or planner on the phone.

What is already in duck-studio to build on:

- `StudioKit/DuckPlanFile.swift` (`duck-plan/1`). Its rule is that the file stores what was
  **measured** and the plan is recomputed every time it is read. The same rule suits an
  editable graph: nodes hold intents and parameters, and feasibility is re-derived.
- `DuckSequence.swift`, `Choreography.swift`, `DuckMove`/`.duckmove`.
- The `microduck-tree.json` and `microduck-rl-tree.json` test fixtures.
- In duckkit, the documented-but-empty steering and deliberation tiers
  (`docs/adr/0001-three-loops.md`).

The connection to duckbatch: each node is a skill policy (walk, stand, get up, kick …) or a
trained student. The planner is a tree over those nodes, and a decision model scores branches.
In duckbatch that was the one role Jev's gap score earned, not acting alone.
