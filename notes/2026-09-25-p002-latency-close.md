# p002 addendum: thinking off takes 25 s down to 7–9 s; E2B and the phone path

Same set, same scoring, same laptop (CPU, 16 threads, llama.cpp b11188). Records:
`records/p002-plan-requests/*.json`, each with the served model, grammar and thinking recorded.

**Correction to the p002 close:** Gemma 4's chat template turns *thinking* on unless told
otherwise, and the first p002 run did not say otherwise. Its 30/30 and 28 s median are with
thinking on. That hidden reasoning (about 300 tokens before about 75 tokens of plan) was most of
the latency. `planner.ask(thinking=...)` / `plan-eval --no-think` now makes it explicit and records it.

| model | grammar | thinking | sequences | fully right | out of scope refused | unreadable | median s | output tokens |
|---|---|---|---|---|---|---|---|---|
| E4B | json_schema | on | 30/30 | 30/30 | 3/3 | 0 | 27.8 | ~375 |
| E4B | json_schema | **off** | 29/30 | 27/30 | 3/3 | 0 | 12.8 | 127 |
| E4B | none (phone-like) | off | 28/30 | 27/30 | 3/3 | 1 | 8.0 | 79 |
| E2B | json_schema | on | 25/30 | 22/30 | 3/3 | 1 | 25.3 | 364 |
| **E2B** | **json_schema** | **off** | **28/30** | 25/30 | 3/3 | 0 | **8.5** | 75 |
| E2B | none (phone-like) | off | 24/30 | 22/30 | 2/3 | 4 | 6.5 | 75 |
| Decide router | – | – | 17/30 | 9/30 | 3/3 | – | 3.0 | – |

## What it shows
- **Latency is output length, not model size.** Prompt processing is cached (0.3 s). E2B
  generates about 13 tokens/s and E4B about 9.5, so thinking (about 5× the tokens) mattered far
  more than size (1.4×). With thinking off, E2B plans in about 8 s and E4B in 8–13 s.
- **For E2B, thinking hurt.** 25/30 with thinking on against 28/30 with it off: reasoning about
  "walk while looking left" talked it into two steps.
- **Without a grammar, the failures are almost all one field, `sound`.** E2B wrote "quack" and
  "say hello" where the vocabulary says `chirp` and `greet` (4 of its 6 failed plans); E4B did it
  once. The reader refuses rather than guesses, as it should. The fix is the prompt: the no-grammar
  prompt lists the sound labels bare, while the actions get meanings. That is a **new prompt**, so
  it is scored on a new set (p003), not on p002 again.
- **Best laptop setting now: E2B, grammar, thinking off**: 28/30 sequences in 8.5 s, and 3.2 GB
  instead of 5.1 GB.

## The phone (Duck Studio, branch `phone-planner`)
The phone runtimes (Apple's on-device model, MLX) cannot constrain decoding, and the app already
turns thinking off, so the "none / off" rows are the laptop's prediction for the phone: about 24–28
of 30 with this prompt. The app now carries the same prompt and set, pinned by sha256, and
"Check this model" in the plan editor runs p002 on the phone itself. That measurement replaces the
prediction.
