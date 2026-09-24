"""The router's code parts: splitting and mapping. No model, no network."""

import yaml

from duckbatch import router


def test_the_splitter_matches_every_pre_registered_request():
    reqs = yaml.safe_load(open("menus/r001-router-requests.yaml"))["requests"]
    for r in reqs:
        assert len(router.split_clauses(r["text"])) == len(r["steps"]), r["text"]


def test_walking_and_looking_stays_one_step():
    assert router.split_clauses("walk forward and look left") == ["walk forward and look left"]


def test_code_owns_every_number_within_pollens_limits():
    fast = router.to_command({"action": "walk_forward", "speed": "fast", "head": "straight"})
    assert fast["twist"] == [0.25, 0.0, 0.0]
    back = router.to_command({"action": "walk_backward", "speed": "fast", "head": "straight"})
    assert back["twist"][0] == -0.2
    spin = router.to_command({"action": "turn_left", "speed": "fast", "head": "look_left"})
    assert spin["twist"][2] == 1.0 and spin["head"][2] == 0.4


def test_skills_sounds_and_refusals_map_to_duckkit_tags():
    assert router.to_command({"action": "sit_or_stand"}) == {"kind": "skill", "tag": "sit_toggle"}
    assert router.to_command({"action": "roll"})["tag"] == "roulade"
    assert router.to_command({"action": "make_sound", "sound": "greet"}) == {"kind": "sound", "tag": "greet"}
    assert router.to_command({"action": "unsupported"})["kind"] == "refused"
