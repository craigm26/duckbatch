"""duck-feedback/0: the refusals, the fixed split, calibration and the GLiNER2 export."""

import copy
import json
import uuid

import pytest

from duckbatch import feedback


def route(outcome="edited", **over):
    rec = {
        "format": "duck-feedback/0", "id": str(uuid.uuid4()), "created": "2026-09-24T21:04:11Z",
        "kind": "route_correction", "consent": {"opt_in": True, "share": "research"},
        "source": {"client": "Microduck Studio test"},
        "route_correction": {
            "request": "turn left", "clause": "turn left",
            "router": {"model": "fastino/GLiNER2.5-Decide", "revision": "65624f1", "vocabulary": "r001"},
            "proposed": {"labels": {"action": "turn_left", "speed": "normal", "head": "look_left"},
                         "confidence": {"action": 0.7, "speed": 0.6, "head": 0.5}},
            "final": {"labels": {"action": "turn_left", "speed": "normal", "head": "straight"}},
            "outcome": outcome,
        },
    }
    rec.update(over)
    return rec


def test_a_well_formed_correction_is_accepted():
    assert feedback.validate(route())["kind"] == "route_correction"


@pytest.mark.parametrize("mutate, why", [
    (lambda r: r["consent"].update(opt_in=False), "no opt-in"),
    (lambda r: r["consent"].update(share="local"), "may not leave the device"),
    (lambda r: r.update(kind="vibes"), "kind"),
    (lambda r: r.update(created="yesterday"), "UTC"),
    (lambda r: r["route_correction"]["final"]["labels"].update(head="look_sideways"), "not in the vocabulary"),
    (lambda r: r["route_correction"]["router"].update(vocabulary="r999"), "vocabulary"),
    (lambda r: r["route_correction"].update(outcome="accepted"), "accepted but labels changed"),
])
def test_bad_records_are_refused_with_a_reason(mutate, why):
    r = route()
    mutate(r)
    with pytest.raises(feedback.FeedbackError, match=why):
        feedback.validate(r)


def test_an_edit_that_changes_nothing_is_refused():
    r = route()
    r["route_correction"]["final"]["labels"] = dict(r["route_correction"]["proposed"]["labels"])
    with pytest.raises(feedback.FeedbackError, match="no label changed"):
        feedback.validate(r)


def test_a_preference_needs_two_different_networks_and_a_known_choice():
    pref = {"format": "duck-feedback/0", "id": "pref-0001", "created": "2026-09-24T21:04:11Z",
            "kind": "policy_preference", "consent": {"opt_in": True, "share": "public"},
            "policy_preference": {"a": {"repo": "x/a", "fingerprint": "sha256:aa"},
                                  "b": {"repo": "x/b", "fingerprint": "sha256:bb"},
                                  "shown": {"where": "sim", "order": "a_left"},
                                  "choice": "b", "reasons": ["steadier"]}}
    feedback.validate(pref)
    same = copy.deepcopy(pref)
    same["policy_preference"]["b"]["fingerprint"] = "sha256:aa"
    with pytest.raises(feedback.FeedbackError, match="same network"):
        feedback.validate(same)
    free = copy.deepcopy(pref)
    free["policy_preference"]["reasons"] = ["it had a certain je ne sais quoi"]
    with pytest.raises(feedback.FeedbackError, match="reasons"):
        feedback.validate(free)


def test_the_split_is_fixed_by_id_and_near_one_fifth():
    ids = [f"record-{i:05d}" for i in range(5000)]
    held = [feedback.is_held_out(i) for i in ids]
    assert held == [feedback.is_held_out(i) for i in ids], "deterministic"
    assert 0.17 < sum(held) / len(held) < 0.23


def simulated_corrections():
    """r001's recorded Decide proposals, 'corrected' to the gold labels: a SIMULATED person,
    for exercising the pipeline on realistic data. Not human feedback."""
    rows = json.load(open("records/r001-router-requests/decide.json"))["rows"]
    out = []
    for i, row in enumerate(rows * 3):
        proposed = {k: row["got"][k] for k in ("action", "speed", "head")}
        gold = {"action": row["gold"]["action"], "speed": row["gold"].get("speed", "normal"),
                "head": row["gold"].get("head", "straight")}
        out.append(route(outcome="accepted" if proposed == gold else "edited",
                         id=f"sim-r001-{i:04d}",
                         route_correction={
                             "request": row["request"], "clause": row["clause"],
                             "router": {"model": "fastino/GLiNER2.5-Decide", "revision": "65624f1",
                                        "vocabulary": "r001"},
                             "proposed": {"labels": proposed,
                                          "confidence": {k: row["confidence"][k] for k in proposed}},
                             "final": {"labels": gold},
                             "outcome": "accepted" if proposed == gold else "edited"}))
    return [feedback.validate(r) for r in out]


def test_calibration_is_measured_on_the_held_out_split_only():
    recs = simulated_corrections()
    cal = feedback.calibration(recs, min_support=5)
    assert cal["held_out_records"] == sum(feedback.is_held_out(r["id"]) for r in recs)
    rows = {r["threshold"]: r for r in cal["by_threshold"]}
    assert rows[0.0]["auto_accepted"] >= rows[0.9]["auto_accepted"]


def test_the_decide_export_is_accepted_by_gliner2s_own_validator(tmp_path):
    pytest.importorskip("gliner2")
    recs = simulated_corrections()
    res = feedback.export_decide(recs, tmp_path / "train.jsonl")
    train = [r for r in recs if not feedback.is_held_out(r["id"])]
    assert res["examples"] == len(train)
    from gliner2.training.data import TrainingDataset
    assert len(TrainingDataset.load(str(tmp_path / "train.jsonl"))) == len(train)


def test_the_public_dataset_accepts_only_public_consent():
    r = route()  # share: research
    with pytest.raises(feedback.FeedbackError, match="accepts only 'public'"):
        feedback.validate(r, require_share="public")
    r["consent"]["share"] = "public"
    assert feedback.validate(r, require_share="public")


def test_contributions_are_found_in_dated_subdirectories(tmp_path):
    day = tmp_path / "contributions" / "2026-09-25"
    day.mkdir(parents=True)
    r = route()
    r["consent"]["share"] = "public"
    (day / "abc.jsonl").write_text(json.dumps(r) + "\n")
    assert len(feedback.read([tmp_path], require_share="public")) == 1
