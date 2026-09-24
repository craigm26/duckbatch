# f000: SIMULATED feedback. Not from any person.

`corrections.jsonl` holds 165 `duck-feedback/0` `route_correction` records, built from r001's
recorded GLiNER2.5-Decide proposals (55 clauses, three times over). In each one the "person"
simply corrects to the pre-registered gold label. They exist to exercise the pipeline
(`duckbatch feedback validate|report|export-decide`) and to give Duck Studio a file to test its
writer against. Nothing learned from them is a finding about users: the ground truth is still
my labels, including the head-during-turn ambiguity.
