"""The W0 practice rooms as MuJoCo scenes for duck-sim, plus one more (`chairs`).

The geometry is `practice.py`'s, metre for metre: a 3.0 x 2.4 m corner with a sofa end and a toy
box, the dining table with the floor 0.72 m below, the stairwell one step (0.17 m) down, the 3 cm
door sill. `chairs` is the living room with two chairs (four 3 cm legs and a seat a duck walks
under) and a pair of shoes: the chair-legs room in the wander plan's W1 list.

**The duck always starts at the world origin, facing +x.** duck-sim's body server places it there
and has no way to put it anywhere else. So a seeded start is made by moving the room instead: the
whole room sits in one static body, rotated and shifted so that the chosen start (x, y, yaw) in
room coordinates lands on the origin. `to_room` turns the simulator's ground-truth trunk position
back into room coordinates, where `practice.Room.drop_at` and `edge_margin` answer "is the duck
over a drop".

The surface the duck stands on is always at z = 0. Lower floors (the room under the table, the
stairwell) are below zero, so the simulator's SIT keyframe puts the duck on the right level.

What the simulator cannot have here: the visitor. The body server steps a static world, and a cat
that walks up needs a moving body that nothing in duck-sim drives. Greetings are not measured.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from practice import ROOM_X, ROOM_Y, ROOMS, Box, Room, _living  # noqa: E402

WALL_H = 1.0            # walls stand this tall above the lowest floor
WALL_T = 0.05
SLAB = 0.3              # floors are slabs this thick under their top surface
TABLE_MARGIN = 0.3      # a start on the table is at least this far from its edge
DROP_MARGIN = 0.5       # a start is at least this far from a stairwell
CLEAR = 0.35            # and this far from walls, furniture and the sill


def _chairs() -> Room:
    boxes = list(_living())
    legs: list[Box] = []
    for name, (cx, cy) in (("chair A", (1.1, 0.7)), ("chair B", (2.0, 0.9))):
        for dx in (-0.2, 0.2):
            for dy in (-0.2, 0.2):
                x, y = cx + dx, cy + dy
                legs.append(Box(x - 0.015, y - 0.015, x + 0.015, y + 0.015, 0.45, f"{name} leg"))
    shoes = [Box(0.4, 1.9, 0.5, 2.15, 0.08, "shoe"), Box(0.55, 1.95, 0.65, 2.2, 0.08, "shoe")]
    return Room("chairs", boxes + legs + shoes, cat=False)


SIM_ROOMS = {
    "living": ROOMS["living"],
    "table": ROOMS["table"],
    "stair": ROOMS["stair"],
    "threshold": ROOMS["threshold"],
    "chairs": _chairs,
}

# Chair seats: drawn, but not boxes in the room's 2D sense, since a duck walks under them.
SEATS = {"chairs": [(1.1, 0.7), (2.0, 0.9)]}


@dataclass
class Start:
    room: str
    seed: int
    x: float
    y: float
    yaw: float

    def to_room(self, wx: float, wy: float) -> tuple[float, float]:
        """World (simulator) coordinates to room coordinates."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * wx - s * wy, self.y + s * wx + c * wy


def room_of(name: str) -> Room:
    return SIM_ROOMS[name]()


def lowest(room: Room) -> float:
    if room.surface is not None:
        return -room.surface.height
    return -max([p.height for p in room.pits], default=0.0)


def start_for(name: str, seed: int) -> Start:
    """A seeded start in room coordinates, clear of everything by CLEAR and of drops by a margin."""
    room = room_of(name)
    rng = random.Random(f"w1/{name}/{seed}")
    for _ in range(10_000):
        if room.surface is not None:
            b = room.surface
            x = rng.uniform(b.x0 + TABLE_MARGIN, b.x1 - TABLE_MARGIN)
            y = rng.uniform(b.y0 + TABLE_MARGIN, b.y1 - TABLE_MARGIN)
        else:
            x, y = rng.uniform(CLEAR, ROOM_X - CLEAR), rng.uniform(CLEAR, ROOM_Y - CLEAR)
        if any(b.x0 - CLEAR <= x <= b.x1 + CLEAR and b.y0 - CLEAR <= y <= b.y1 + CLEAR
               for b in room.boxes + room.snags):
            continue
        if room.pits and room.edge_margin(x, y) < DROP_MARGIN:
            continue
        return Start(name, seed, round(x, 3), round(y, 3), round(rng.uniform(-math.pi, math.pi), 3))
    raise RuntimeError(f"no start found in {name}")


def _box(name: str, x0: float, y0: float, x1: float, y1: float, z0: float, z1: float, rgba: str) -> str:
    return (f'<geom name="{name}" type="box" pos="{(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {(z0 + z1) / 2:.4f}" '
            f'size="{(x1 - x0) / 2:.4f} {(y1 - y0) / 2:.4f} {(z1 - z0) / 2:.4f}" rgba="{rgba}"/>')


def _floor_cells(room: Room) -> list[tuple[float, float, float, float, float]]:
    """(x0, y0, x1, y1, top) rectangles tiling the room's floor, the pits at their depth."""
    if room.surface is not None:
        return [(0.0, 0.0, ROOM_X, ROOM_Y, -room.surface.height)]
    xs = sorted({0.0, ROOM_X, *[v for p in room.pits for v in (p.x0, p.x1)]})
    ys = sorted({0.0, ROOM_Y, *[v for p in room.pits for v in (p.y0, p.y1)]})
    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            cells.append((xs[i], ys[j], xs[i + 1], ys[j + 1], -room.drop_at(cx, cy)))
    return cells


def scene_xml(start: Start, robot_xml: str, keyframes: str) -> str:
    room = room_of(start.room)
    low = lowest(room)
    g: list[str] = []
    for k, (x0, y0, x1, y1, top) in enumerate(_floor_cells(room)):
        g.append(_box(f"floor{k}", x0, y0, x1, y1, top - SLAB, top, "0.55 0.5 0.45 1"))
    if room.surface is not None:
        b = room.surface
        g.append(_box("table", b.x0, b.y0, b.x1, b.y1, low, 0.0, "0.45 0.3 0.2 1"))
    for k, (x0, y0, x1, y1) in enumerate(((-WALL_T, -WALL_T, 0.0, ROOM_Y + WALL_T),
                                         (ROOM_X, -WALL_T, ROOM_X + WALL_T, ROOM_Y + WALL_T),
                                         (0.0, -WALL_T, ROOM_X, 0.0),
                                         (0.0, ROOM_Y, ROOM_X, ROOM_Y + WALL_T))):
        g.append(_box(f"wall{k}", x0, y0, x1, y1, low - SLAB, low + WALL_H, "0.85 0.85 0.8 1"))
    for k, b in enumerate(room.boxes):
        base = -room.drop_at((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2)
        g.append(_box(f"thing{k}", b.x0, b.y0, b.x1, b.y1, base, base + b.height, "0.3 0.45 0.6 1"))
    for k, (cx, cy) in enumerate(SEATS.get(start.room, [])):
        g.append(_box(f"seat{k}", cx - 0.23, cy - 0.23, cx + 0.23, cy + 0.23, 0.43, 0.46, "0.5 0.35 0.25 1"))
    for k, b in enumerate(room.snags):
        g.append(_box(f"sill{k}", b.x0, b.y0, b.x1, b.y1, 0.0, b.height, "0.6 0.4 0.2 1"))
    # The room moves so that the start lands on the origin facing +x: world = R(-yaw) (p - start).
    c, s = math.cos(-start.yaw), math.sin(-start.yaw)
    tx, ty = -(c * start.x - s * start.y), -(s * start.x + c * start.y)
    quat = f"{math.cos(-start.yaw / 2):.6f} 0 0 {math.sin(-start.yaw / 2):.6f}"
    body = "\n      ".join(g)
    return f"""<mujoco model="w1-{start.room}-{start.seed}">
  <include file="{robot_xml}" />
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" /></visual>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
    <body name="room" pos="{tx:.4f} {ty:.4f} 0" quat="{quat}">
      {body}
    </body>
  </worldbody>
{keyframes}
</mujoco>
"""


def keyframes_from(scene: Path) -> str:
    """The <keyframe> block of a stock scene, so the body server's SIT start is the stock one."""
    text = scene.read_text()
    a, b = text.index("<keyframe>"), text.index("</keyframe>") + len("</keyframe>")
    return "  " + text[a:b]


def write_scene(start: Start, out_dir: Path, model_dir: Path) -> Path:
    """Write the scene beside links to the robot model, since MuJoCo resolves includes and meshes
    against the scene's own directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in model_dir.iterdir():
        if item.name.startswith("scene") or item.suffix not in (".xml", "") and not item.is_dir():
            continue
        link = out_dir / item.name
        if not link.exists():
            link.symlink_to(item)
    path = out_dir / f"w1-{start.room}-{start.seed}.xml"
    path.write_text(scene_xml(start, "robot_allcollisions.xml",
                              keyframes_from(model_dir / "scene_apartment.xml")))
    return path


if __name__ == "__main__":
    for name in SIM_ROOMS:
        for seed in (1, 2, 3, 4, 101, 102):
            print(start_for(name, seed))
