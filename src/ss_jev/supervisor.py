"""Named scene signals and a hysteresis policy. Jev classifies in the background."""

from dataclasses import dataclass, field
from collections import deque

from .domain import Calibration
from .perception import count_rings, hit_flash_active, pause_menu_visible


SCREENS = (
    "training_active",
    "paused",
    "transition",
    "popup_or_results",
    "unknown",
)

SCREEN_CRITERIA = {
    "training_active": "The three training rings are visible and the hero is in the apple drill.",
    "paused": "The pause menu or pause button overlay is showing.",
    "transition": "The playfield is changing scenes, fading, or the rings just disappeared mid-run.",
    "popup_or_results": "A mission complete, stats, or results popup covers the drill.",
    "unknown": "The signals do not match a known training screen.",
}


@dataclass(frozen=True)
class SceneSignals:
    rings_visible: int
    rings_total: int
    pause_button: str
    cues_last_2s: str
    hit_flash: str
    previous_screen: str

    def as_state(self) -> dict:
        return {
            "rings_visible": f"{self.rings_visible} of {self.rings_total}",
            "pause_button": self.pause_button,
            "cues_last_2s": self.cues_last_2s,
            "hit_flash": self.hit_flash,
            "previous_screen": self.previous_screen,
        }


def collect_signals(
    frame,
    calibration: Calibration,
    cue_times: deque[float],
    now: float,
    previous_screen: str,
) -> SceneSignals:
    recent = sum(1 for stamp in cue_times if now - stamp <= 2.0)
    return SceneSignals(
        rings_visible=count_rings(frame, calibration.rings, calibration.ring_radius),
        rings_total=len(calibration.rings),
        pause_button="seen" if pause_menu_visible(frame) else "not seen",
        cues_last_2s="none" if recent == 0 else str(recent),
        hit_flash="active" if hit_flash_active(frame, calibration) else "not active",
        previous_screen=previous_screen,
    )


@dataclass
class InputPolicy:
    """Two consistent non-active answers pause input; training_active resumes it."""

    paused: bool = False
    pending: int = 0
    last_screen: str = "unknown"
    min_confidence: float = 0.6

    def apply(self, screen: str, confidence: float) -> str:
        self.last_screen = screen
        if screen == "training_active" and confidence >= self.min_confidence:
            self.pending = 0
            if self.paused:
                self.paused = False
                return "resume"
            return "active"
        if screen != "training_active" and confidence >= self.min_confidence:
            self.pending += 1
            if self.pending >= 2 and not self.paused:
                self.paused = True
                return "pause"
            return "pending_pause" if not self.paused else "paused"
        return "uncertain"


@dataclass
class Supervisor:
    policy: InputPolicy = field(default_factory=InputPolicy)
    last_ask: float = -1.0
    period: float = 0.5
    previous_screen: str = "unknown"
    shadow: bool = True
    pending_result: dict | None = None

    def due(self, now: float) -> bool:
        return now - self.last_ask >= self.period
