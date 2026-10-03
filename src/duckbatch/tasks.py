"""Tasks duckbatch registers beside Pollen's, built from Pollen's own factories.

THE LEFT KICK HAS NO TASK UPSTREAM. Pollen ships `ball_kick_left.onnx` but registers only the
right-foot BallKick (`KICK_FOOT = "right"`); the factory already takes `kick_foot`, which flips
the ball's spawn side and the support-foot sensor and changes nothing else. So the left-foot task
and its backlash twin are registered here, by the same calls Pollen uses for the right
(`mjlab_microduck/tasks/__init__.py`), under duckbatch's own ids so they can never be mistaken
for Pollen's. Importing this module registers them; `finetune`, `sim` and `skill_eval` import it.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab_microduck.robot.microduck_constants import MICRODUCK_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
)

LEFT_KICK = "Duckbatch-BallKick-Left-Flat-MicroDuck"
LEFT_KICK_BACKLASH = "Duckbatch-BallKick-Left-Flat-Backlash-MicroDuck"

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    register_mjlab_task(
        task_id=LEFT_KICK,
        env_cfg=make_microduck_ball_kick_env_cfg(kick_foot="left"),
        play_env_cfg=make_microduck_ball_kick_env_cfg(play=True, kick_foot="left"),
        rl_cfg=MicroduckBallKickRlCfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
    # The same robot model Pollen pairs with its right-kick backlash twin (_BL_GROUNDCONTACT).
    register_mjlab_task(
        task_id=LEFT_KICK_BACKLASH,
        env_cfg=make_backlash_variant(make_microduck_ball_kick_env_cfg(kick_foot="left"),
                                      MICRODUCK_BACKLASH_ROBOT_CFG),
        play_env_cfg=make_backlash_variant(make_microduck_ball_kick_env_cfg(play=True, kick_foot="left"),
                                           MICRODUCK_BACKLASH_ROBOT_CFG),
        rl_cfg=MicroduckBallKickRlCfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
    _registered = True
