"""A live duck as the runner sees it: Pollen's control lane, plus the two things the runner needs.

`control.Rpc` keeps the last of each notification. The runner also needs to know *when* each one
arrived, because a depth frame from two seconds ago is not a reason to walk, and it needs to send
continuous intents (`robot.move`, `robot.head`) as notifications, which have no id and get no answer
(`duck-ipc-proto`: "Continuous; send as a notification").
"""

from __future__ import annotations

import json
import time
from typing import Any

from control import Rpc


class TimedRpc(Rpc):
    """`Rpc`, stamping each notification with its monotonic arrival time."""

    def __init__(self, timeout: float = 20.0):
        super().__init__(timeout=timeout)
        self.arrived: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def on_message(self, raw: Any) -> None:
        super().on_message(raw)
        try:
            message = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "replace"))
        except (TypeError, ValueError, AttributeError):
            return
        if message.get("id") is None and message.get("method"):
            method = message["method"]
            self.arrived[method] = time.monotonic()
            self.counts[method] = self.counts.get(method, 0) + 1


class RpcLane:
    """The runner's `Lane` over one `TimedRpc`, whatever transport carries it."""

    def __init__(self, rpc: TimedRpc):
        self.rpc = rpc

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return self.rpc.call(method, params)

    def notify(self, method: str, params: dict[str, Any]) -> bool:
        send = self.rpc._send
        return bool(send and send({"jsonrpc": "2.0", "method": method, "params": params}))

    def latest(self, method: str) -> tuple[Any, float | None]:
        return self.rpc.notifications.get(method), self.rpc.arrived.get(method)
