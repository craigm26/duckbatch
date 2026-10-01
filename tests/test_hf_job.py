"""The job recipe is one script, whichever launcher runs it."""

from duckbatch.hf_job import BOOTSTRAP, BOOTSTRAP_FILE


def test_the_published_bootstrap_is_the_one_the_cli_runs():
    body = BOOTSTRAP_FILE.read_text().split("\n", 2)[2]
    assert body == BOOTSTRAP.lstrip("\n"), "run: python -c 'from duckbatch.hf_job import write_bootstrap_file; write_bootstrap_file()'"


def test_a_menu_can_arrive_as_text():
    assert 'MENU_YAML' in BOOTSTRAP and "MENU=/work/menu-inline.yaml" in BOOTSTRAP
