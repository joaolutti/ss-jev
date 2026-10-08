from ss_jev.domain import Box, Detection, Direction, PLAY_750X400, Point, Track
from ss_jev.timing import assign_key, should_fire, time_to_arrival
from ss_jev.tracking import Tracker


def test_slow_cue_assigns_left_and_fires_on_tta():
    start = Point(80, 284)
    near = Point(140, 284)
    track = Track(
        1,
        "apple",
        near,
        0.0,
        0.4,
        observations=5,
        samples=[
            (0.0, start),
            (0.1, Point(95, 284)),
            (0.2, Point(110, 284)),
            (0.3, Point(125, 284)),
            (0.4, near),
        ],
    )
    key = assign_key(track, PLAY_750X400)
    track.assigned_key = key
    assert key is Direction.LEFT
    tta = time_to_arrival(track, PLAY_750X400.rings[Direction.LEFT], PLAY_750X400.ring_radius)
    assert tta is not None and tta > 0
    fire, reason, _ = should_fire(track, PLAY_750X400, 1 / 27)
    assert fire is False
    track.samples.append((0.5, Point(215, 284)))
    track.center = Point(215, 284)
    track.last_seen = 0.5
    fire, reason, tta = should_fire(track, PLAY_750X400, 1 / 27)
    assert fire is True
    assert reason in {"predicted", "closest_pass", "in_ring"}


def test_fast_cue_that_jumps_past_the_ring_still_fires():
    before = Point(180, 284)
    after = Point(280, 284)
    track = Track(
        2,
        "apple",
        after,
        0.0,
        0.04,
        observations=2,
        samples=[(0.0, before), (0.04, after)],
    )
    track.assigned_key = assign_key(track, PLAY_750X400)
    assert track.assigned_key is Direction.LEFT
    fire, reason, _ = should_fire(track, PLAY_750X400, 1 / 27)
    assert fire is True
    assert reason in {"predicted", "closest_pass", "in_ring"}


def test_two_cues_fire_independently():
    left = Track(
        1,
        "spike_disk",
        Point(200, 284),
        0.0,
        0.1,
        observations=2,
        samples=[(0.0, Point(140, 284)), (0.1, Point(200, 284))],
    )
    right = Track(
        2,
        "spike_disk",
        Point(520, 283),
        0.0,
        0.1,
        observations=2,
        samples=[(0.0, Point(600, 283)), (0.1, Point(520, 283))],
    )
    left.assigned_key = assign_key(left, PLAY_750X400)
    right.assigned_key = assign_key(right, PLAY_750X400)
    assert left.assigned_key is Direction.LEFT
    assert right.assigned_key is Direction.RIGHT
    left.samples.append((0.2, Point(218, 284)))
    left.center = Point(218, 284)
    right.samples.append((0.2, Point(548, 283)))
    right.center = Point(548, 283)
    fire_l, _, _ = should_fire(left, PLAY_750X400, 1 / 27)
    fire_r, _, _ = should_fire(right, PLAY_750X400, 1 / 27)
    assert fire_l and fire_r


def test_stationary_blob_is_rejected():
    track = Track(
        3,
        "apple",
        Point(40, 360),
        0.0,
        0.5,
        observations=5,
        samples=[(t * 0.1, Point(40, 360)) for t in range(6)],
    )
    assert assign_key(track, PLAY_750X400) is None
    assert track.rejected is True
    fire, _, _ = should_fire(track, PLAY_750X400, 1 / 27)
    assert fire is False


def test_cue_moving_away_from_ring_does_not_assign_that_ring():
    track = Track(
        4,
        "apple",
        Point(100, 284),
        0.0,
        0.2,
        observations=3,
        samples=[(0.0, Point(180, 284)), (0.1, Point(140, 284)), (0.2, Point(100, 284))],
    )
    key = assign_key(track, PLAY_750X400)
    assert key is not Direction.LEFT


def test_fast_apple_fires_when_tta_is_about_one_frame():
    ring = PLAY_750X400.rings[Direction.UP]
    current = Point(ring.x, ring.y - 50)
    previous = Point(ring.x, current.y - 31.7)
    track = Track(
        9,
        "apple",
        current,
        0.0,
        0.05,
        observations=2,
        samples=[(0.0, previous), (0.05, current)],
    )
    track.assigned_key = assign_key(track, PLAY_750X400)
    assert track.assigned_key is Direction.UP
    tta = time_to_arrival(track, ring, PLAY_750X400.ring_radius)
    assert tta is not None
    assert tta > PLAY_750X400.input_delay + (1 / 27) / 2
    fire, reason, _ = should_fire(track, PLAY_750X400, 1 / 27)
    assert fire is True
    assert reason == "predicted"


def test_one_frame_disk_inside_assign_radius_fires():
    ring = PLAY_750X400.rings[Direction.LEFT]
    center = Point(ring.x - 38, ring.y)
    track = Track(8, "spike_disk", center, 0.0, 0.0, observations=1, samples=[(0.0, center)])
    track.assigned_key = assign_key(track, PLAY_750X400)
    assert track.assigned_key is Direction.LEFT
    fire, reason, _ = should_fire(track, PLAY_750X400, 1 / 27)
    assert fire is True
    assert reason == "first_frame"


def test_one_frame_disk_near_two_rings_does_not_fire():
    from ss_jev.domain import Calibration

    calibration = Calibration(
        version=3,
        name="close-rings",
        window_size=(200, 200),
        roi=(0, 0, 200, 200),
        rings={Direction.LEFT: Point(100, 100), Direction.RIGHT: Point(120, 100)},
        character=Point(110, 160),
        ring_radius=35,
    )
    center = Point(110, 100)
    track = Track(7, "spike_disk", center, 0.0, 0.0, observations=1, samples=[(0.0, center)])
    track.assigned_key = assign_key(track, calibration)
    fire, _, _ = should_fire(track, calibration, 1 / 27)
    assert fire is False


def test_tracker_match_radius_grows_with_speed():
    tracker = Tracker(min_step=20, cold_step=80, lookahead=1.5, expiry_seconds=0.3)
    first = Detection(Point(10, 10), 100, 1, Box(5, 5, 10, 10), "apple")
    tracker.update([first], 0.0)
    second = Detection(Point(40, 10), 100, 1, Box(35, 5, 10, 10), "apple")
    moved = tracker.update([second], 0.04)[0]
    assert moved.id == 1
    radius = tracker.match_radius(moved, 0.04)
    assert radius >= 20
    tracker.update([], 0.4)
    assert tracker.tracks == {}
    assert tracker.expired
