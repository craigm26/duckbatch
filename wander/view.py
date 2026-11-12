"""What the page draws: the depth frame, the practice room, and a status line. Plain SVG strings."""

from __future__ import annotations

import html
import math

from practice import ROOM_X, ROOM_Y, PracticeDuck
from runner import Status
from senses import ranges

INK = "#2b2a28"
PAPER = "#fbf7ef"
DUCK = "#f2b705"
CAT = "#7a5c3e"
WALL = "#c9bfae"
NEAR = (214, 84, 46)
FAR = (238, 233, 222)


def _mix(t: float) -> str:
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(NEAR, FAR))


def depth_svg(frame: dict | None, age_s: float | None, size: int = 224) -> str:
    """The 8x8 frame as the duck sees it: near is warm, far is pale, no measurement is hatched."""
    cell = size // 8
    parts = [f'<svg viewBox="0 0 {size} {size + 22}" style="width:100%;max-width:{size}px;display:block" '
             f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="depth frame">',
             '<defs><pattern id="h" width="6" height="6" patternUnits="userSpaceOnUse" '
             'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#bbb" '
             'stroke-width="2"/></pattern></defs>']
    zones = ranges(frame) if frame else [None] * 64
    for i, metres in enumerate(zones or [None] * 64):
        row, col = divmod(i, 8)
        fill = "url(#h)" if metres is None else _mix(metres / 1.5)
        parts.append(f'<rect x="{col * cell}" y="{row * cell}" width="{cell - 1}" height="{cell - 1}" fill="{fill}"/>')
        if metres is not None and metres < 1.0:
            parts.append(f'<text x="{col * cell + cell / 2}" y="{row * cell + cell / 2 + 4}" '
                         f'font-size="10" text-anchor="middle" fill="{INK}">{metres:.2f}</text>')
    label = ("no frame yet" if age_s is None else
             f"fresh, {age_s:.1f} s old" if age_s <= 0.6 else f"blind: last frame {age_s:.0f} s ago")
    parts.append(f'<text x="0" y="{size + 16}" font-size="12" fill="{INK}">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def room_svg(duck: PracticeDuck, scale: int = 120) -> str:
    """The practice room from above: walls, furniture, the cat, the duck and where it has been."""
    w, h = ROOM_X * scale, ROOM_Y * scale
    sx = lambda x: x * scale
    sy = lambda y: h - y * scale
    parts = [f'<svg viewBox="-6 -6 {w + 12} {h + 12}" style="width:100%;max-width:{w + 12:.0f}px;display:block" '
             f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="practice room">',
             f'<rect x="0" y="0" width="{w}" height="{h}" fill="{PAPER}" stroke="{WALL}" stroke-width="6"/>']
    for b in duck.boxes:
        parts.append(f'<rect x="{sx(b.x0)}" y="{sy(b.y1)}" width="{(b.x1 - b.x0) * scale}" '
                     f'height="{(b.y1 - b.y0) * scale}" fill="{WALL}"/>'
                     f'<text x="{sx(b.x0) + 4}" y="{sy(b.y1) + 14}" font-size="11" fill="{INK}">{b.name}</text>')
    trail = duck.path[-900::6]
    if len(trail) > 1:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for _, x, y in trail)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{DUCK}" stroke-width="2" opacity="0.5"/>')
    cx, cy, _ = duck.cat
    parts.append(f'<circle cx="{sx(cx)}" cy="{sy(cy)}" r="{0.12 * scale}" fill="{CAT}"/>'
                 f'<text x="{sx(cx)}" y="{sy(cy) + 4}" font-size="10" text-anchor="middle" fill="#fff">cat</text>')
    # The sensor's 45 degree view, from the head.
    heading = duck.yaw + duck.head[2]
    for side in (-1, 1):
        a = heading + side * math.radians(22.5)
        parts.append(f'<line x1="{sx(duck.x)}" y1="{sy(duck.y)}" x2="{sx(duck.x + 0.8 * math.cos(a))}" '
                     f'y2="{sy(duck.y + 0.8 * math.sin(a))}" stroke="{DUCK}" stroke-dasharray="3 3"/>')
    nose = (duck.x + 0.13 * math.cos(duck.yaw), duck.y + 0.13 * math.sin(duck.yaw))
    parts.append(f'<circle cx="{sx(duck.x)}" cy="{sy(duck.y)}" r="{0.09 * scale}" fill="{DUCK}" stroke="{INK}"/>'
                 f'<line x1="{sx(duck.x)}" y1="{sy(duck.y)}" x2="{sx(nose[0])}" y2="{sy(nose[1])}" '
                 f'stroke="{INK}" stroke-width="3"/>')
    if duck.fallen:
        parts.append(f'<text x="{sx(duck.x)}" y="{sy(duck.y) - 14}" font-size="12" text-anchor="middle" '
                     f'fill="#b3261e">fell</text>')
    parts.append("</svg>")
    return "".join(parts)


def status_md(st: Status) -> str:
    if not st.ticks:
        return f"**{st.state}.** {st.why}"
    blind = 100 * st.blind_ticks / max(1, st.ticks)
    walked = 100 * st.walking_ticks / max(1, st.ticks)
    lines = [
        f"**{st.state}**" + (f": {st.why}" if not st.running and st.why else ""),
        f"room ahead **{st.clear_ahead_m:.2f} m** · walking {walked:.0f}% of the time · "
        f"blind {blind:.0f}% · greetings {st.greetings}",
    ]
    return "  \n".join(lines)


def log_md(st: Status, extra: list[str] | None = None) -> str:
    lines = st.log[-14:] + (extra or [])[-4:]
    return "```\n" + "\n".join(lines) + "\n```" if lines else ""
