"""Geometry and calibration types with no I/O."""

import json
from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from pathlib import Path


class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    center: Point
    area: float
    confidence: float
    box: Box
    kind: str = "apple"


@dataclass
class Track:
    id: int
    kind: str
    center: Point
    first_seen: float
    last_seen: float
    observations: int = 1
    fired: bool = False
    rejected: bool = False
    assigned_key: Direction | None = None
    samples: list[tuple[float, Point]] = field(default_factory=list)
    predicted_arrival: float | None = None
    jev_choice: str | None = None
    jev_confidence: float | None = None
    jev_sent_at: float | None = None
    jev_received_at: float | None = None
    jev_asked: bool = False

    def __post_init__(self) -> None:
        if not self.samples:
            self.samples = [(self.first_seen, self.center)]

    @property
    def velocity_px_s(self) -> Point:
        history = self.samples[-5:]
        if len(history) < 2:
            return Point(0.0, 0.0)
        vx = vy = 0.0
        n = 0
        for (t0, p0), (t1, p1) in zip(history, history[1:]):
            dt = t1 - t0
            if dt <= 1e-6:
                continue
            vx += (p1.x - p0.x) / dt
            vy += (p1.y - p0.y) / dt
            n += 1
        if n == 0:
            return Point(0.0, 0.0)
        return Point(vx / n, vy / n)

    @property
    def speed(self) -> float:
        velocity = self.velocity_px_s
        return hypot(velocity.x, velocity.y)


def distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def closing_speed(track: Track, target: Point) -> float:
    """Positive when the track is moving toward the target, in px/s."""
    offset_x = target.x - track.center.x
    offset_y = target.y - track.center.y
    span = hypot(offset_x, offset_y)
    if span < 1e-6:
        return track.speed
    velocity = track.velocity_px_s
    return (velocity.x * offset_x + velocity.y * offset_y) / span


REFERENCE_RING_SPAN = 335.0
REFERENCE_RING_RADIUS = 35.0
REFERENCE_CHARACTER_RX = 70.0
REFERENCE_CHARACTER_RY = 90.0
REFERENCE_MIN_AREA = 400
REFERENCE_MAX_AREA = 3000


def ring_span(rings: dict[Direction, Point]) -> float:
    """Horizontal distance between the left and right rings, in pixels."""
    return abs(rings[Direction.RIGHT].x - rings[Direction.LEFT].x)


def sizes_from_rings(rings: dict[Direction, Point]) -> dict[str, float | int]:
    """Scale the 750×400 tuned sizes by how far apart these rings sit."""
    scale = ring_span(rings) / REFERENCE_RING_SPAN
    return {
        "ring_radius": round(REFERENCE_RING_RADIUS * scale, 2),
        "character_rx": round(REFERENCE_CHARACTER_RX * scale, 2),
        "character_ry": round(REFERENCE_CHARACTER_RY * scale, 2),
        "min_area": max(1, round(REFERENCE_MIN_AREA * scale * scale)),
        "max_area": max(1, round(REFERENCE_MAX_AREA * scale * scale)),
    }


@dataclass(frozen=True)
class Calibration:
    version: int
    name: str
    window_size: tuple[int, int]
    roi: tuple[int, int, int, int]
    rings: dict[Direction, Point]
    character: Point
    ring_radius: float = 35.0
    input_delay: float = 0.04
    min_area: int = 400
    max_area: int = 3000
    min_confidence: float = 0.7
    character_rx: float = 70.0
    character_ry: float = 90.0
    title: str = "Swords & Souls"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "window_size": list(self.window_size),
            "roi": list(self.roi),
            "targets": {direction.value: [point.x, point.y] for direction, point in self.rings.items()},
            "character": [self.character.x, self.character.y],
            "ring_radius": self.ring_radius,
            "input_delay": self.input_delay,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_confidence": self.min_confidence,
            "character_rx": self.character_rx,
            "character_ry": self.character_ry,
            "title": self.title,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        data = json.loads(path.read_text())
        rings = {
            Direction(name): Point(float(coords[0]), float(coords[1]))
            for name, coords in data["targets"].items()
        }
        character = data.get("character")
        if character is None:
            xs = [point.x for point in rings.values()]
            ys = [point.y for point in rings.values()]
            character = [sum(xs) / len(xs), sum(ys) / len(ys)]
        roi = data["roi"]
        window = data["window_size"]
        return cls(
            version=int(data.get("version", 3)),
            name=str(data.get("name", path.stem)),
            window_size=(int(window[0]), int(window[1])),
            roi=(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])),
            rings=rings,
            character=Point(float(character[0]), float(character[1])),
            ring_radius=float(data.get("ring_radius", 35)),
            input_delay=float(data.get("input_delay", 0.04)),
            min_area=int(data.get("min_area", 400)),
            max_area=int(data.get("max_area", 3000)),
            min_confidence=float(data.get("min_confidence", 0.7)),
            character_rx=float(data.get("character_rx", 70)),
            character_ry=float(data.get("character_ry", 90)),
            title=str(data.get("title", "Swords & Souls")),
        )


PLAY_750X400 = Calibration(
    version=3,
    name="play-750x400",
    window_size=(750, 400),
    roi=(0, 0, 750, 400),
    rings={
        Direction.LEFT: Point(221, 284),
        Direction.RIGHT: Point(556, 283),
        Direction.UP: Point(389, 116),
    },
    character=Point(390, 282),
    ring_radius=35.0,
    input_delay=0.04,
    character_rx=70.0,
    character_ry=90.0,
)

# Ring centers in the 800×500 Proton window, measured from the 2026-10-06 crop.
# The old crop put the left ring ~35px from the left edge and the up ring ~25px
# from the top, so a fast apple was already inside the ring on the first frame.
LIVE_WINDOW_RINGS = {
    Direction.LEFT: Point(320, 420),
    Direction.RIGHT: Point(479, 420),
    Direction.UP: Point(400, 340),
}


def scaled_playfield_roi(
    window_rings: dict[Direction, Point],
    window_size: tuple[int, int],
    reference: Calibration,
) -> tuple[int, int, int, int]:
    """Crop the whole reference playfield, scaled so the left ring stays put.

    The reference tape shows the approach before each ring. A tight crop that
    starts at the ring does not.
    """
    scale = ring_span(window_rings) / ring_span(reference.rings)
    ref_left = reference.rings[Direction.LEFT]
    origin_x = round(window_rings[Direction.LEFT].x - ref_left.x * scale)
    origin_y = round(window_rings[Direction.LEFT].y - ref_left.y * scale)
    width = round(reference.window_size[0] * scale)
    height = round(reference.window_size[1] * scale)
    origin_x = min(max(0, origin_x), max(0, window_size[0] - width))
    origin_y = min(max(0, origin_y), max(0, window_size[1] - height))
    return origin_x, origin_y, width, height


def _live_calibration() -> Calibration:
    window_size = (800, 500)
    roi = scaled_playfield_roi(LIVE_WINDOW_RINGS, window_size, PLAY_750X400)
    rings = {
        direction: Point(point.x - roi[0], point.y - roi[1])
        for direction, point in LIVE_WINDOW_RINGS.items()
    }
    left = rings[Direction.LEFT]
    right = rings[Direction.RIGHT]
    return Calibration(
        version=3,
        name="live",
        window_size=window_size,
        roi=roi,
        rings=rings,
        character=Point((left.x + right.x) / 2, (left.y + right.y) / 2),
        input_delay=0.04,
        title="Swords & Souls",
        **sizes_from_rings(rings),
    )


LIVE_800X500 = _live_calibration()
