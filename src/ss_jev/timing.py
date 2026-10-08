"""Assign a key when a cue appears and fire from predicted time-to-arrival."""

from .domain import REFERENCE_RING_RADIUS, Calibration, Direction, Track, closing_speed, distance


MIN_SPEED_PX_S = 40.0
GROUNDED_OBSERVATIONS = 5
ASSIGN_RADIUS_SCALE = 1.2


def time_to_arrival(track: Track, target, ring_radius: float) -> float | None:
    remaining = max(0.0, distance(track.center, target) - ring_radius * 0.25)
    closing = closing_speed(track, target)
    if closing <= 1.0:
        return None
    return remaining / closing


def assign_key(track: Track, calibration: Calibration) -> Direction | None:
    """Choose the ring on the side the cue appeared, then require it is incoming."""
    if track.fired or track.rejected or track.assigned_key is not None:
        return track.assigned_key
    rings = calibration.rings
    if not rings:
        return None
    origin = track.samples[0][1]
    nearest = min(rings.items(), key=lambda item: distance(origin, item[1]))
    direction, ring = nearest
    if track.observations == 1:
        if distance(track.center, ring) <= calibration.ring_radius * ASSIGN_RADIUS_SCALE:
            track.assigned_key = direction
            return direction
        return None
    if track.speed < MIN_SPEED_PX_S * calibration.ring_radius / REFERENCE_RING_RADIUS:
        if track.observations >= GROUNDED_OBSERVATIONS:
            track.rejected = True
        return None
    closing = closing_speed(track, ring)
    dist = distance(track.center, ring)
    origin_dx = (origin.x - ring.x) * (track.center.x - ring.x)
    origin_dy = (origin.y - ring.y) * (track.center.y - ring.y)
    crossed = origin_dx < 0 or origin_dy < 0
    if dist <= calibration.ring_radius * ASSIGN_RADIUS_SCALE or crossed:
        track.assigned_key = direction
        return direction
    if closing < 20.0 * calibration.ring_radius / REFERENCE_RING_RADIUS:
        if track.observations >= GROUNDED_OBSERVATIONS:
            track.rejected = True
        return None
    track.assigned_key = direction
    return direction


def _unique_first_frame(track: Track, calibration: Calibration) -> bool:
    """A single sighting inside the assign radius of exactly one ring."""
    if track.observations != 1 or track.assigned_key is None:
        return False
    limit = calibration.ring_radius * ASSIGN_RADIUS_SCALE
    distances = sorted(distance(track.center, point) for point in calibration.rings.values())
    if not distances or distances[0] > limit:
        return False
    if len(distances) > 1 and distances[1] <= limit:
        return False
    return distance(track.center, calibration.rings[track.assigned_key]) <= limit


def should_fire(
    track: Track,
    calibration: Calibration,
    frame_period: float,
) -> tuple[bool, str, float | None]:
    """Return whether to press, a reason, and the current TTA."""
    if track.fired or track.rejected or track.assigned_key is None:
        return False, "not_ready", None
    ring = calibration.rings[track.assigned_key]
    tta = time_to_arrival(track, ring, calibration.ring_radius)
    dist = distance(track.center, ring)
    if track.observations == 1:
        if _unique_first_frame(track, calibration):
            return True, "first_frame", tta
        return False, "ambiguous_first_frame", tta
    # One full frame of lead, plus input delay. Half a frame was late for fast cues.
    deadline = calibration.input_delay + frame_period
    if tta is not None and tta <= deadline:
        return True, "predicted", tta
    if dist <= calibration.ring_radius:
        return True, "in_ring", tta
    origin = track.samples[0][1]
    crossed = ((origin.x - ring.x) * (track.center.x - ring.x) < 0) or (
        (origin.y - ring.y) * (track.center.y - ring.y) < 0
    )
    if crossed and dist <= calibration.ring_radius * 2:
        return True, "closest_pass", tta
    return False, "waiting", tta
