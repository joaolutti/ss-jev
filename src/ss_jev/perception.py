"""Recognize apples, spike disks, rings, and the pause button."""

from functools import lru_cache
from math import pi
from pathlib import Path

import cv2
import numpy as np

from .domain import (
    REFERENCE_RING_RADIUS,
    Box,
    Calibration,
    Detection,
    Direction,
    Point,
)

_DISK_IMAGE = Path(__file__).parent / "assets" / "spike-disk.png"
_PAUSE_IMAGE = Path(__file__).parent / "assets" / "pause-menu.png"


def inside_character(point: Point, calibration: Calibration) -> bool:
    dx = (point.x - calibration.character.x) / max(calibration.character_rx, 1.0)
    dy = (point.y - calibration.character.y) / max(calibration.character_ry, 1.0)
    return dx * dx + dy * dy <= 1.0


# A standing hero's red hat is a few hundred pixels. A taken hit turns most of
# the sprite red, one connected blob well above the hat. 0.15 of the ellipse
# sits in the gap measured on the 2026-10-06 live crop (hat p95 ~340, flashes
# above ~800) and on the 750×400 tape (hat p95 ~1000, flashes above ~4000).
HIT_FLASH_FRACTION = 0.15


def hit_flash_active(frame: np.ndarray, calibration: Calibration) -> bool:
    """A taken hit turns the hero red. The red hat alone does not."""
    height, width = frame.shape[:2]
    cx, cy = calibration.character.x, calibration.character.y
    rx, ry = max(calibration.character_rx, 1.0), max(calibration.character_ry, 1.0)
    x0, y0 = max(0, int(cx - rx)), max(0, int(cy - ry))
    x1, y1 = min(width, int(cx + rx) + 1), min(height, int(cy + ry) + 1)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, (0, 90, 65), (13, 255, 255))
    high = cv2.inRange(hsv, (168, 90, 65), (179, 255, 255))
    mask = cv2.bitwise_or(low, high) > 0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    red = (mask & ellipse).astype(np.uint8)
    if not np.any(red):
        return False
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(red, 8)
    if count <= 1:
        return False
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    needed = max(8, round(HIT_FLASH_FRACTION * pi * rx * ry))
    return largest >= needed


def detect_red_objects(
    frame: np.ndarray, min_area: int = 400, max_area: int = 3000
) -> list[Detection]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, (0, 90, 65), (13, 255, 255))
    high = cv2.inRange(hsv, (168, 90, 65), (179, 255, 255))
    mask = cv2.bitwise_or(low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        ratio = width / height
        if not 0.55 <= ratio <= 1.8:
            continue
        bounds_area = width * height
        fill = area / bounds_area
        if fill < 0.35:
            continue
        confidence = min(1.0, fill / 0.75) * min(ratio, 1 / ratio)
        found.append(
            Detection(Point(x + width / 2, y + height / 2), area, confidence, Box(x, y, width, height))
        )
    return found


@lru_cache(maxsize=8)
def _disk_templates(scale_key: int):
    template = cv2.imread(str(_DISK_IMAGE))
    if template is None:
        raise FileNotFoundError(f"Spike disk template missing: {_DISK_IMAGE}")
    mask = np.zeros(template.shape[:2], np.uint8)
    center = (template.shape[1] // 2, template.shape[0] // 2)
    cv2.circle(mask, center, min(center) - 2, 255, -1)
    result = []
    scale = scale_key / 1000
    for factor in (0.75, 0.9, 1.0, 1.15):
        sized = factor * scale
        resized = cv2.resize(template, None, fx=sized, fy=sized)
        resized_mask = cv2.resize(mask, (resized.shape[1], resized.shape[0]))
        result.append((resized, cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), resized_mask))
    return result


def detect_spike_disks(
    frame: np.ndarray,
    targets: dict[Direction, Point],
    minimum_score: float = 0.865,
    search_radius: int | None = None,
    ring_radius: float = REFERENCE_RING_RADIUS,
) -> list[Detection]:
    scale = ring_radius / REFERENCE_RING_RADIUS
    if search_radius is None:
        search_radius = max(8, round(85 * scale))
    brown_pad = max(4, round(65 * scale))
    brown_min = max(4, round(120 * scale * scale))
    slack = max(1, round(10 * scale))
    found = []
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for target in targets.values():
        bx0 = max(0, round(target.x) - brown_pad)
        by0 = max(0, round(target.y) - brown_pad)
        bx1 = min(width, round(target.x) + brown_pad)
        by1 = min(height, round(target.y) + brown_pad)
        near = hsv[by0:by1, bx0:bx1]
        if near.size == 0:
            continue
        brown = (
            (near[:, :, 0] >= 5)
            & (near[:, :, 0] <= 27)
            & (near[:, :, 1] >= 75)
            & (near[:, :, 2] >= 45)
            & (near[:, :, 2] <= 190)
        )
        if np.count_nonzero(brown) < brown_min:
            continue
        x0 = max(0, round(target.x) - search_radius)
        y0 = max(0, round(target.y) - search_radius)
        x1 = min(width, round(target.x) + search_radius)
        y1 = min(height, round(target.y) + search_radius)
        patch = frame[y0:y1, x0:x1]
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        best = None
        for template, gray_template, mask in _disk_templates(round(scale * 1000)):
            th, tw = template.shape[:2]
            if patch.shape[0] < th or patch.shape[1] < tw:
                continue
            response = cv2.matchTemplate(gray_patch, gray_template, cv2.TM_CCORR_NORMED, mask=mask)
            _, _, _, position = cv2.minMaxLoc(response)
            candidate = patch[position[1] : position[1] + th, position[0] : position[0] + tw]
            score = float(cv2.matchTemplate(candidate, template, cv2.TM_CCORR_NORMED, mask=mask)[0, 0])
            if best is None or score > best[0]:
                best = (score, position[0] + x0, position[1] + y0, tw, th)
        if best and best[0] >= minimum_score:
            score, x, y, tw, th = best
            center = Point(x + tw / 2, y + th / 2)
            if np.hypot(center.x - target.x, center.y - target.y) <= search_radius - slack:
                found.append(Detection(center, tw * th, score, Box(x, y, tw, th), "spike_disk"))
    return found


def detect_cues(frame: np.ndarray, calibration: Calibration) -> list[Detection]:
    apples = [
        detection
        for detection in detect_red_objects(frame, calibration.min_area, calibration.max_area)
        if detection.confidence >= calibration.min_confidence
        and not inside_character(detection.center, calibration)
    ]
    disks = detect_spike_disks(frame, calibration.rings, ring_radius=calibration.ring_radius)
    return apples + disks


def count_rings(
    frame: np.ndarray,
    targets: dict[Direction, Point],
    ring_radius: float = REFERENCE_RING_RADIUS,
) -> int:
    if len(targets) < 2:
        return 0
    scale = ring_radius / REFERENCE_RING_RADIUS
    inner = 27 * scale
    outer = 38 * scale
    pad_before = max(1, round(39 * scale))
    pad_after = max(1, round(40 * scale))
    minimum = max(8, round(300 * scale * scale))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 150)
    height, width = white.shape
    visible = 0
    for target in targets.values():
        x0 = max(0, round(target.x) - pad_before)
        y0 = max(0, round(target.y) - pad_before)
        x1 = min(width, round(target.x) + pad_after)
        y1 = min(height, round(target.y) + pad_after)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        radius_squared = (xx - target.x) ** 2 + (yy - target.y) ** 2
        annulus = (radius_squared >= inner**2) & (radius_squared <= outer**2)
        if np.count_nonzero(annulus & white[y0:y1, x0:x1]) > minimum:
            visible += 1
    return visible


@lru_cache(maxsize=1)
def _pause_template():
    template = cv2.imread(str(_PAUSE_IMAGE))
    if template is None:
        raise FileNotFoundError(f"Pause menu template missing: {_PAUSE_IMAGE}")
    return template


def pause_menu_visible(frame: np.ndarray, minimum_score: float = 0.85) -> bool:
    template = _pause_template()
    if frame.shape[0] < template.shape[0] or frame.shape[1] < template.shape[1]:
        return False
    response = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    return float(response.max()) >= minimum_score


def find_rings(
    frame: np.ndarray, ring_radius: float = REFERENCE_RING_RADIUS
) -> dict[Direction, Point]:
    """Guess left, right, and up ring centers from white annuli."""
    scale = ring_radius / REFERENCE_RING_RADIUS
    margin = max(8, round(40 * scale))
    step = max(2, round(4 * scale))
    inner = 27 * scale
    outer = 38 * scale
    minimum = max(8, round(280 * scale * scale))
    separation = max(8, round(45 * scale))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
    height, width = white.shape
    peaks: list[tuple[int, int, int]] = []
    for y in range(margin, height - margin, step):
        for x in range(margin, width - margin, step):
            yy, xx = np.ogrid[y - margin : y + margin, x - margin : x + margin]
            radius_squared = (xx - x) ** 2 + (yy - y) ** 2
            annulus = (radius_squared >= inner**2) & (radius_squared <= outer**2)
            count = int(np.count_nonzero(annulus & (white[y - margin : y + margin, x - margin : x + margin] > 0)))
            if count > minimum:
                peaks.append((count, x, y))
    peaks.sort(reverse=True)
    kept: list[tuple[int, int, int]] = []
    for count, x, y in peaks:
        if all((x - kx) ** 2 + (y - ky) ** 2 > separation**2 for _, kx, ky in kept):
            kept.append((count, x, y))
        if len(kept) >= 4:
            break
    if len(kept) < 3:
        raise RuntimeError("Could not find three training rings in this frame.")
    points = [Point(x, y) for _, x, y in kept[:3]]
    leftmost = min(points, key=lambda point: point.x)
    rightmost = max(points, key=lambda point: point.x)
    upper = min(points, key=lambda point: point.y)
    return {Direction.LEFT: leftmost, Direction.RIGHT: rightmost, Direction.UP: upper}


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    calibration: Calibration,
    tracks: list | None = None,
) -> np.ndarray:
    result = frame.copy()
    for direction, target in calibration.rings.items():
        center = (round(target.x), round(target.y))
        cv2.circle(result, center, round(calibration.ring_radius), (255, 255, 0), 1)
        cv2.putText(
            result,
            direction.value,
            (center[0] - 20, center[1] - round(calibration.ring_radius) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
    cv2.ellipse(
        result,
        (round(calibration.character.x), round(calibration.character.y)),
        (round(calibration.character_rx), round(calibration.character_ry)),
        0,
        0,
        360,
        (80, 80, 255),
        1,
    )
    for detection in detections:
        box = detection.box
        color = (0, 255, 0) if detection.kind == "apple" else (0, 165, 255)
        cv2.rectangle(result, (box.x, box.y), (box.x + box.width, box.y + box.height), color, 1)
        cv2.putText(
            result,
            f"{detection.kind} {detection.confidence:.2f}",
            (box.x, box.y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
    if tracks:
        for track in tracks:
            if track.assigned_key:
                cv2.putText(
                    result,
                    f"#{track.id} {track.assigned_key.value}",
                    (round(track.center.x) + 8, round(track.center.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
    return result
