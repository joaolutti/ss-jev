from pathlib import Path

import cv2
import numpy as np
import pytest

from ss_jev.domain import LIVE_800X500, PLAY_750X400, Direction, Point, sizes_from_rings
from ss_jev.perception import (
    count_rings,
    detect_cues,
    detect_red_objects,
    detect_spike_disks,
    hit_flash_active,
    inside_character,
    pause_menu_visible,
)
from ss_jev.timing import assign_key
from ss_jev.domain import Track

ROOT = Path("/home/jl/Downloads/ss-automation")


def test_red_apple_is_found_but_blue_object_is_ignored():
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    cv2.circle(frame, (50, 80), 12, (0, 0, 255), -1)
    cv2.circle(frame, (180, 80), 12, (255, 0, 0), -1)
    found = detect_red_objects(frame)
    assert len(found) == 1
    assert abs(found[0].center.x - 50) < 2
    assert found[0].confidence > 0.65


def test_live_window_sizes_follow_ring_span():
    sizes = sizes_from_rings(LIVE_800X500.rings)
    assert LIVE_800X500.window_size == (800, 500)
    assert LIVE_800X500.ring_radius == sizes["ring_radius"]
    assert LIVE_800X500.character_rx == sizes["character_rx"]
    assert LIVE_800X500.character_ry == sizes["character_ry"]
    assert LIVE_800X500.min_area == sizes["min_area"]
    assert LIVE_800X500.max_area == sizes["max_area"]
    assert LIVE_800X500.ring_radius == pytest.approx(16.61, abs=0.01)
    assert LIVE_800X500.min_area == 90


def test_ring_count_follows_ring_radius():
    _x, _y, width, height = LIVE_800X500.roi
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for target in LIVE_800X500.rings.values():
        cv2.circle(frame, (round(target.x), round(target.y)), 16, (255, 255, 255), 2)
    assert count_rings(frame, LIVE_800X500.rings, LIVE_800X500.ring_radius) == 3
    assert count_rings(frame, LIVE_800X500.rings, PLAY_750X400.ring_radius) == 0


def test_live_crop_shows_the_approach_before_each_ring():
    roi_x, roi_y, width, height = LIVE_800X500.roi
    assert roi_x >= 0 and roi_y >= 0
    assert roi_x + width <= LIVE_800X500.window_size[0]
    assert roi_y + height <= LIVE_800X500.window_size[1]
    window = {
        Direction.LEFT: Point(320, 420),
        Direction.RIGHT: Point(479, 420),
        Direction.UP: Point(400, 340),
    }
    for direction, point in window.items():
        assert LIVE_800X500.rings[direction] == Point(point.x - roi_x, point.y - roi_y)
    radius = LIVE_800X500.ring_radius
    rings = LIVE_800X500.rings
    approaches = {
        Direction.LEFT: rings[Direction.LEFT].x - radius,
        Direction.RIGHT: width - (rings[Direction.RIGHT].x + radius),
        Direction.UP: rings[Direction.UP].y - radius,
    }
    # A 380 px/s apple at 30 fps moves this far between frames. Two sightings
    # before the ring need two of those gaps outside the circle.
    frame_travel = 380 / 30
    for direction, approach in approaches.items():
        assert approach >= 2 * frame_travel, direction


def test_character_mask_rejects_center_blob():
    assert inside_character(Point(390, 282), PLAY_750X400)
    assert not inside_character(Point(221, 284), PLAY_750X400)


@pytest.mark.skipif(not (ROOT / "runs/sample-analysis/frame-49.png").is_file(), reason="legacy fixture missing")
def test_frame49_finds_disks_on_left_and_up():
    frame = cv2.imread(str(ROOT / "runs/sample-analysis/frame-49.png"))
    disks = detect_spike_disks(frame, PLAY_750X400.rings)
    kinds = {(round(d.center.x / 50), round(d.center.y / 50)) for d in disks}
    assert disks, "expected spike disks on the annotated frame"
    assert count_rings(frame, PLAY_750X400.rings) >= 2


@pytest.mark.skipif(not (ROOT / "runs/20260925-133246/ambiguous-160782.png").is_file(), reason="legacy fixture missing")
def test_two_disks_are_both_visible_on_ambiguous_frame():
    frame = cv2.imread(str(ROOT / "runs/20260925-133246/ambiguous-160782.png"))
    disks = detect_spike_disks(frame, PLAY_750X400.rings)
    assert len(disks) >= 2


@pytest.mark.skipif(not (ROOT / "runs/20260925-174035/snapshot-030s.png").is_file(), reason="legacy fixture missing")
def test_grounded_apples_are_not_assigned_a_key():
    frame = cv2.imread(str(ROOT / "runs/20260925-174035/snapshot-030s.png"))
    # This is a full-window snapshot. Inspect the raw red blobs, and treat a corner blob as grounded.
    apples = detect_red_objects(frame)
    assert apples
    # A blob in the grass stays put.
    blob = min(apples, key=lambda item: item.center.x)
    track = Track(
        9,
        "apple",
        blob.center,
        0.0,
        0.5,
        observations=5,
        samples=[(i * 0.1, blob.center) for i in range(6)],
    )
    assert assign_key(track, PLAY_750X400) is None
    assert track.rejected


def test_hit_flash_synthetic():
    frame = np.zeros((400, 750, 3), dtype=np.uint8)
    cv2.ellipse(frame, (390, 282), (60, 80), 0, 0, 360, (0, 0, 220), -1)
    assert hit_flash_active(frame, PLAY_750X400)
    cues = detect_cues(frame, PLAY_750X400)
    assert all(not (abs(c.center.x - 390) < 50 and abs(c.center.y - 282) < 50) or c.kind != "apple" for c in cues)


def _red_hat(frame: np.ndarray, calibration) -> None:
    center = (
        round(calibration.character.x),
        round(calibration.character.y - calibration.character_ry * 0.35),
    )
    cv2.circle(frame, center, 12, (0, 0, 220), -1)


def test_red_hat_is_not_a_hit_and_a_body_flash_is():
    cases = (
        (PLAY_750X400, (400, 750)),
        (LIVE_800X500, (LIVE_800X500.roi[3], LIVE_800X500.roi[2])),
    )
    for calibration, (height, width) in cases:
        standing = np.zeros((height, width, 3), dtype=np.uint8)
        _red_hat(standing, calibration)
        assert not hit_flash_active(standing, calibration)
        flash = standing.copy()
        cv2.ellipse(
            flash,
            (round(calibration.character.x), round(calibration.character.y)),
            (round(calibration.character_rx), round(calibration.character_ry)),
            0,
            0,
            360,
            (0, 0, 220),
            -1,
        )
        assert hit_flash_active(flash, calibration)
