# `duck-feedback/0`: what a person told us, in a form we can learn from

Records written by Duck Studio (or any client) when a person corrects a proposal or prefers one
policy over another. The format is one JSON object per line (JSONL), one record per human act.
duckbatch reads it strictly: `duckbatch feedback validate` refuses a record it does not
understand rather than guessing.

## Why these two kinds

In b001 and b002 every judge was graded against rules I wrote, and in r001 the router was graded
against labels I wrote, one of which turned out to be ambiguous (the head during a turn). People
using the app are the ground truth those grades were missing:

- **`route_correction`**: the router proposed labels for a clause, and the person kept or edited
  them. Each record is a labelled example with the model's confidence attached. That is what
  recalibrates the auto-accept threshold, settles ambiguities by what users want, and fine-tunes
  GLiNER2.5-Decide.
- **`policy_preference`**: the person watched two policies do the same thing and chose one. These
  are the pairwise comparisons a reward model learns from (RLHF), and they measure what no rule
  here can: whether a gait looks right.

## Consent comes first, and is checked

Every record carries `consent`. A record without `"opt_in": true` is **refused**, not dropped
quietly. `share` says how far it may travel:

- `"local"`: stays on the device that wrote it. duckbatch refuses to ingest it.
- `"research"`: may be uploaded to a private dataset for training and evaluation.
- `"public"`: may appear in a public dataset.

A record carries no name, no account, no device identifier and no location. `source.client` is
the app and version, nothing more. The only free text is the clause the person typed, and a
client should show it back before sharing.

## The record

Common fields:

```json
{
  "format": "duck-feedback/0",
  "id": "7f3c…",
  "created": "2026-09-24T21:04:11Z",
  "kind": "route_correction",
  "consent": {"opt_in": true, "share": "research"},
  "source": {"client": "Microduck Studio 1.54"}
}
```

- `id` is a random identifier per record (a UUID), never derived from the person or the device.
- `created` is UTC, ISO-8601, with a trailing `Z`.

### `route_correction`

```json
"route_correction": {
  "request": "turn left then say hello",
  "clause": "turn left",
  "router": {"model": "fastino/GLiNER2.5-Decide", "revision": "65624f1", "vocabulary": "r001"},
  "proposed": {"labels": {"action": "turn_left", "speed": "normal", "head": "look_left"},
               "confidence": {"action": 0.71, "speed": 0.66, "head": 0.56}},
  "final": {"labels": {"action": "turn_left", "speed": "normal", "head": "straight"}},
  "outcome": "edited"
}
```

- `outcome` is one of:
  - `accepted`: the person kept every label. `final` equals `proposed`.
  - `edited`: at least one label changed.
  - `rejected`: the person discarded the step entirely. `final` may be absent.
- Labels must come from the vocabulary named in `router.vocabulary`. For `r001` that is the
  `ACTIONS`, `SPEEDS`, `HEADS` and `SOUNDS` in `src/duckbatch/router.py`. An unknown label refuses
  the record.

### `policy_preference`

```json
"policy_preference": {
  "a": {"repo": "craigm26/microduck-duckbatch-b002-128x128", "fingerprint": "sha256:…"},
  "b": {"repo": "pollen-robotics/microduck-policies", "file": "velstand.onnx", "fingerprint": "sha256:…"},
  "shown": {"where": "sim", "command": [0.15, 0.0, 0.0], "seconds": 6, "seed": 2001,
            "order": "a_left", "pair_id": "p001/0042"},
  "choice": "b",
  "reasons": ["steadier", "more natural"]
}
```

- `choice` is one of `a`, `b`, `tie` or `both_bad`.
- `shown.where` is one of `sim`, `phone_bench`, `ar` or `robot`.
- `shown.order` records which side each policy appeared on, because people favour a side and the
  reward model has to be able to see that.
- `pair_id`, when present, points at the pair duckbatch generated (`duckbatch pairs`), so the
  trajectory features the person saw can be looked up rather than re-simulated.
- `reasons` are drawn from a fixed list (`steadier`, `more natural`, `faster`, `follows the
  command`, `fell`, `jittery`, `other`) so they can be counted. There is no free text.
- `fingerprint` is duckkit's `DuckPolicy.fingerprint`, over `canonicalIdentityBytes`, so a
  preference is about a network, not a filename.

## Held out, from the start

`duckbatch feedback` splits by a hash of `id`: 80% train and 20% held out, deterministic, and never
re-drawn. Any threshold, fine-tune or reward model is reported on the held-out part only.
