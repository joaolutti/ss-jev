"""Frame sources: a video file, or a held-open Wayland screencopy."""

import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoFileSource:
    def __init__(
        self,
        path: Path,
        realtime: bool = False,
        roi: tuple[int, int, int, int] | None = None,
        expected_size: tuple[int, int] | None = None,
    ):
        self.path = Path(path)
        self.realtime = realtime
        self.roi = roi
        self.expected_size = expected_size

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot read {self.path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / fps
        index = 0
        start = time.monotonic()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = _crop_if_needed(frame, self.roi, self.expected_size)
                timestamp = index * period
                if self.realtime:
                    target = start + timestamp
                    delay = target - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                yield frame, timestamp, index
                index += 1
        finally:
            capture.release()


def _crop_if_needed(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    expected_size: tuple[int, int] | None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if expected_size and (width, height) == expected_size:
        return frame
    if roi and (width, height) != (roi[2], roi[3]):
        x, y, crop_w, crop_h = roi
        return frame[y : y + crop_h, x : x + crop_w]
    return frame


class ScreencopySource:
    """Copy one window region through a Wayland connection that stays open."""

    def __init__(self, geometry: tuple[int, int, int, int], fps: float = 60.0):
        from .screencopy import RegionCapture

        self.geometry = geometry
        self.fps = fps
        x, y, width, height = geometry
        self._capture = RegionCapture(x, y, width, height)

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.close()

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        if self._capture is None:
            raise RuntimeError("Screencopy source is closed.")
        period = 1.0 / self.fps
        index = 0
        start = time.monotonic()
        try:
            while True:
                frame = self._capture.grab()
                now = time.monotonic() - start
                yield frame, now, index
                index += 1
                sleep_for = (start + index * period) - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self.close()
