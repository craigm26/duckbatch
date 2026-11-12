"""microduck wander: a duck that walks about, notices who comes close, and says hello.

Two ways in. **Practice duck** runs the whole thing in this Space against a simulated room, for
anyone without a duck online. **Your duck** signs in with Hugging Face, finds your ducks on Pollen's
rendezvous, and drives one over the control lane (`wire.py`, `control.py` and `rendezvous.py` are
Pollen's, from `spaces/shared` in `pollen-robotics/microduck`, Apache-2.0).

The brain (`brain.py`) is the one that choreographed a Got a Talent act in the Arena, with its
states named after Pollen's `docs/ideas/autonomous_behavior.md`. `runner.py` says how it drives a
live duck and why it stands still whenever its depth frames go stale.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import gradio as gr
from fastapi import FastAPI

import rendezvous
from lane import RpcLane, TimedRpc
from practice import PracticeDuck
from rendezvous import RendezvousError
from runner import DEFAULT_MAX_SPEED, Runner
from view import depth_svg, log_md, room_svg, status_md
from wire import WireError, WsConsumer

logging.basicConfig(level=os.environ.get("DUCK_LOG", "INFO"), format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("wander")

ON_A_SPACE = bool(os.environ.get("SPACE_ID"))
OAUTH_AVAILABLE = bool(os.environ.get("OAUTH_CLIENT_ID"))
LABEL = f"microduck-wander/{os.environ.get('SPACE_ID', 'local')}"
SESSION_LIMIT_S = float(os.environ.get("WANDER_LIMIT_S", "300"))


class Visit:
    """One visitor's page: a practice duck or a live link, and the runner driving it."""

    def __init__(self) -> None:
        self.practice: PracticeDuck | None = None
        self.consumer: WsConsumer | None = None
        self.rpc: TimedRpc | None = None
        self.runner: Runner | None = None
        self.robot: str | None = None

    def end(self) -> None:
        if self.runner and self.runner.status.running:
            self.runner.stop("the page closed")
        if self.practice:
            self.practice.close()
        if self.consumer:
            try:
                self.consumer.stop()
            except Exception:  # noqa: BLE001 - leaving is leaving
                pass
        self.practice = self.consumer = self.rpc = self.runner = None


def token_of(typed: str | None, oauth: gr.OAuthToken | None) -> str:
    """The visitor's token: typed, signed in, or (running locally) the stored one. Never a mock."""
    if typed and typed.strip():
        return typed.strip()
    if oauth is not None and oauth.token and oauth.token != "mock-oauth-token-for-local-dev":
        return oauth.token
    if ON_A_SPACE:
        return ""
    from huggingface_hub import get_token
    return (os.environ.get("HF_TOKEN") or get_token() or "").strip()


# -- practice -------------------------------------------------------------------------------------
def practice_start(visit: Visit, budget: bool, speed: float) -> tuple[Visit, str]:
    visit.end()
    visit.practice = PracticeDuck(seed=int(time.time()) % 1000, budget=400 if budget else None)
    visit.runner = Runner(visit.practice, max_speed=speed)
    said = visit.runner.start(limit_s=SESSION_LIMIT_S)
    return visit, f"**{said}.**"


def practice_stop(visit: Visit) -> tuple[Visit, str]:
    if visit.runner:
        visit.runner.stop("you pressed stop")
    return visit, "**stopped.**"


def practice_view(visit: Visit) -> tuple[str, str, str, str]:
    duck, runner = visit.practice, visit.runner
    if duck is None or runner is None:
        return "", "", "Press **start** to put a duck in the room.", ""
    tof, at = duck.latest("tof.frame")
    age = None if at is None else time.monotonic() - at
    return room_svg(duck), depth_svg(tof, age), status_md(runner.status), log_md(runner.status, duck.notes)


# -- your duck ------------------------------------------------------------------------------------
def find(typed: str, oauth: gr.OAuthToken | None) -> tuple[Any, str]:
    token = token_of(typed, oauth)
    if not token:
        return gr.update(choices=[]), "Sign in with Hugging Face first, or paste a token."
    try:
        ducks, others = rendezvous.ducks(token)
    except RendezvousError as e:
        return gr.update(choices=[]), f"**{e}**"
    if not ducks:
        extra = f" Also listed: {', '.join(others)}." if others else ""
        return gr.update(choices=[]), ("No ducks online for this account. A duck appears once it is "
                                       f"signed in (`robotctl account login`) and on a network.{extra}")
    return (gr.update(choices=[(d.label(), d.peer_id) for d in ducks], value=ducks[0].peer_id),
            f"{len(ducks)} duck(s) online. Connect, then start.")


def connect(visit: Visit, peer_id: str | None, typed: str, oauth: gr.OAuthToken | None) -> tuple[Visit, str]:
    if not peer_id:
        return visit, "Pick a duck first: press **find my ducks**."
    token = token_of(typed, oauth)
    if not token:
        return visit, "Sign in with Hugging Face first."
    visit.end()
    rpc = TimedRpc(timeout=20)
    consumer = WsConsumer(token, peer_id, rpc, label=LABEL)
    try:
        consumer.start()
    except WireError as e:
        return visit, f"**Could not reach that duck:** {e}"
    visit.consumer, visit.rpc = consumer, rpc
    return visit, "**Connected.** Clear a space on the floor, then press start."


def live_start(visit: Visit, speed: float, minutes: float) -> tuple[Visit, str]:
    if visit.rpc is None:
        return visit, "Connect to a duck first."
    if visit.runner and visit.runner.status.running:
        return visit, "Already running."
    visit.runner = Runner(RpcLane(visit.rpc), max_speed=speed)
    said = visit.runner.start(limit_s=minutes * 60)
    return visit, f"**{said}.**"


def live_stop(visit: Visit) -> tuple[Visit, str]:
    if visit.runner:
        visit.runner.stop("you pressed stop")
        return visit, "**Stopped.** The duck is standing still."
    return visit, "Nothing is running."


def live_view(visit: Visit) -> tuple[str, str, str]:
    if visit.rpc is None or visit.runner is None:
        return "", "Not running.", ""
    tof = visit.rpc.notifications.get("tof.frame")
    at = visit.rpc.arrived.get("tof.frame")
    age = None if at is None else time.monotonic() - at
    if visit.consumer is not None and visit.consumer.error and visit.runner.status.running:
        visit.runner.stop(visit.consumer.error)
    return depth_svg(tof, age), status_md(visit.runner.status), log_md(visit.runner.status)


# -- the page -------------------------------------------------------------------------------------
INTRO = """
# microduck wander

A duck that walks about on its own, keeps clear of what is in its way, and greets whoever comes
close: a hand, a cat, another duck. It plays the robot's own `curious` the first time and `yes`
when you come back.

**What it sees with.** The 8x8 depth sensor in its head. **What it decides with.** A small state
machine (Chill, LookAround, Wander, TurnInPlace, Greet, Dance), the same one that choreographed a
[Got a Talent act](https://huggingface.co/spaces/pollen-robotics/microduck-arena) in Pollen's Arena.

**Why it stops now and then.** Driven through this Space, the duck reaches you over Pollen's
rendezvous, which forwards at most 400 messages a minute from the robot. The depth sensor alone
sends 15 a second, so for part of every minute no frames arrive. The duck stands still until they
come back: it never walks blind.

This complements Pollen's own microduck apps; it does not replace any of them.
"""

SAFETY = """
**Before you press start:** clear a patch of floor, keep the duck away from edges and stairs, and
stay in the room. The duck stops on its own when you press **stop**, when this page closes, when the
lane drops, and after the session time runs out. Picked up or fallen, it holds still.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="microduck wander") as demo:
        visit = gr.State(Visit(), delete_callback=lambda v: v.end())
        gr.Markdown(INTRO)
        with gr.Tab("Practice duck"):
            gr.Markdown("A duck and a cat in a simulated room, run here. Nothing to sign in to.")
            with gr.Row():
                p_budget = gr.Checkbox(value=True, label="Keep the rendezvous limit, as a real duck would")
                p_speed = gr.Slider(0.06, 0.2, value=DEFAULT_MAX_SPEED, step=0.02, label="Top speed (m/s)")
                p_start = gr.Button("Start", variant="primary")
                p_stop = gr.Button("Stop", variant="stop")
            p_said = gr.Markdown()
            p_room = gr.HTML()
            with gr.Row(equal_height=False):
                p_depth = gr.HTML()
                p_status = gr.Markdown()
            p_log = gr.Markdown()
            p_tick = gr.Timer(0.5)
            p_start.click(practice_start, [visit, p_budget, p_speed], [visit, p_said])
            p_stop.click(practice_stop, visit, [visit, p_said])
            p_tick.tick(practice_view, visit, [p_room, p_depth, p_status, p_log], show_progress="hidden")

        with gr.Tab("Your duck"):
            gr.Markdown(SAFETY)
            with gr.Row():
                if OAUTH_AVAILABLE:
                    gr.LoginButton()
                typed = gr.Textbox(label="or a Hugging Face token", type="password", scale=2)
            with gr.Row():
                finding = gr.Button("find my ducks")
                chosen = gr.Dropdown(choices=[], label="your ducks", scale=2)
                connecting = gr.Button("connect")
            l_said = gr.Markdown()
            with gr.Row():
                l_speed = gr.Slider(0.06, 0.2, value=DEFAULT_MAX_SPEED, step=0.02, label="Top speed (m/s)")
                l_minutes = gr.Slider(1, 10, value=5, step=1, label="Session length (minutes)")
                l_start = gr.Button("Start", variant="primary")
                l_stop = gr.Button("STOP", variant="stop")
            with gr.Row():
                l_depth = gr.HTML()
                l_status = gr.Markdown()
            l_log = gr.Markdown()
            l_tick = gr.Timer(0.5)
            finding.click(find, typed, [chosen, l_said])
            connecting.click(connect, [visit, chosen, typed], [visit, l_said])
            l_start.click(live_start, [visit, l_speed, l_minutes], [visit, l_said])
            l_stop.click(live_stop, visit, [visit, l_said])
            l_tick.tick(live_view, visit, [l_depth, l_status, l_log], show_progress="hidden")
    return demo


demo = build()
app = FastAPI()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"ok": "yes", "space": os.environ.get("SPACE_ID", "local"),
            "sign_in": "available" if OAUTH_AVAILABLE else "token box"}


app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
