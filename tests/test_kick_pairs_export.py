"""The kick pack exporter: clips stop at the episode end and the ball shares the duck's origin."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_kick_pairs_for_app import clips_for  # noqa: E402


def test_clips_stop_at_the_first_end_and_the_ball_is_rezeroed_with_the_duck():
    steps, keep = 10, 2
    raw = np.zeros((steps, keep, 21))
    raw[:, :, 0] = 30.0 + np.arange(steps)[:, None] * 0.01   # grid offset in x
    raw[:, :, 3] = 1.0                                        # identity quaternion
    ball = np.zeros((steps, keep, 3))
    ball[:, :, 0] = 30.1 + np.arange(steps)[:, None] * 0.1
    ball[:, :, 2] = 0.035
    frames, balls = clips_for(raw, ball, lengths=np.array([10, 4]))
    assert [len(f) for f in frames] == [5, 2], "decimated by 2, env 1 cut at its end (4 steps)"
    assert all(len(r) == 22 for r in frames[0]), "15 joints with the mouth put back"
    assert frames[0][0][0] == 0.0
    assert balls[0][0] == [0.1, 0.0, 0.035], "the ball 10 cm ahead of the duck, not 30 m"
