"""Jev per-cue Choice in shadow mode. Code still owns the key."""

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any, Callable

from .domain import Track

RULES = "Hit apples and spike disks with the arrow matching their ring. Ignore anything that is not an incoming cue."

KEY_CRITERIA = {
    "left": "The cue is approaching the left ring.",
    "right": "The cue is approaching the right ring.",
    "up": "The cue is approaching the upper ring.",
    "none": "This is not an incoming cue. Examples: a grounded apple, a hit flash, or a decorative object.",
}

CUE_INSTRUCTIONS = (
    "Which arrow should the hero press for `cue`? "
    "Use `cue.approach_side` and `cue.moving`. Choose none if it is not an incoming cue."
)


def approach_side(track: Track) -> str:
    if track.assigned_key:
        return track.assigned_key.value
    velocity = track.velocity_px_s
    if abs(velocity.x) < 1 and abs(velocity.y) < 1:
        return "unknown"
    if abs(velocity.x) >= abs(velocity.y):
        return "left" if velocity.x < 0 else "right"
    return "up" if velocity.y < 0 else "down"


def motion_label(track: Track) -> str:
    if track.rejected:
        return "not toward a ring"
    if track.speed < 40:
        return "stationary"
    if track.assigned_key:
        return "toward ring"
    return "moving"


def confidence_bucket(value: float) -> str:
    if value >= 0.85:
        return "high"
    if value >= 0.7:
        return "medium"
    return "low"


def cue_state(track: Track, other_in_flight: int, detector_confidence: float) -> dict[str, Any]:
    return {
        "cue": {
            "kind": track.kind,
            "approach_side": approach_side(track),
            "moving": motion_label(track),
            "detector_confidence": confidence_bucket(detector_confidence),
        },
        "other_cues_in_flight": other_in_flight,
        "rules": RULES,
    }


def choice_questions(option_order: tuple[str, ...] = ("left", "right", "up", "none")) -> dict:
    from typesafe_sdk import Choice

    criteria = {key: KEY_CRITERIA[key] for key in option_order}
    return {
        "key": Choice(instructions=CUE_INSTRUCTIONS, criteria=criteria),
    }


def screen_questions() -> dict:
    from typesafe_sdk import Choice

    from .supervisor import SCREEN_CRITERIA, SCREENS

    return {
        "screen": Choice(
            instructions="Which screen is showing, given the named signals in state?",
            criteria={name: SCREEN_CRITERIA[name] for name in SCREENS},
        )
    }


@dataclass
class ShadowAnswer:
    track_id: int
    choice: str
    confidence: float
    sent_at: float
    received_at: float
    tokens: int = 0
    raw: dict[str, float] = field(default_factory=dict)


class CueJudge:
    """Background Jev worker. Harvest answers without blocking the frame loop."""

    def __init__(self, enabled: bool = True, ask: Callable[..., Any] | None = None):
        self.enabled = enabled and (ask is not None or bool(os.environ.get("TYPESAFE_API_KEY")))
        self._ask = ask
        self._jobs: SimpleQueue = SimpleQueue()
        self._answers: dict[int, ShadowAnswer] = {}
        self._lock = threading.Lock()
        self.tokens = 0
        self.calls = 0
        self.errors = 0
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="jev-cue", daemon=True)
            self._thread.start()

    def submit(self, track: Track, state: dict[str, Any], sent_at: float) -> None:
        if not self.enabled or track.jev_asked:
            return
        track.jev_asked = True
        track.jev_sent_at = sent_at
        self._jobs.put((track.id, state, sent_at, ("left", "right", "up", "none")))

    def harvest(self) -> list[ShadowAnswer]:
        with self._lock:
            ready = list(self._answers.values())
            self._answers.clear()
        return ready

    def attach(self, tracks: dict[int, Track], answers: list[ShadowAnswer]) -> None:
        for answer in answers:
            track = tracks.get(answer.track_id)
            if track is None or track.fired:
                continue
            track.jev_choice = answer.choice
            track.jev_confidence = answer.confidence
            track.jev_received_at = answer.received_at

    def close(self) -> None:
        if self.enabled:
            self._jobs.put(None)
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            track_id, state, sent_at, order = job
            try:
                choice, confidence, tokens, dist = self._evaluate(state, order)
            except Exception:
                self.errors += 1
                continue
            answer = ShadowAnswer(
                track_id=track_id,
                choice=choice,
                confidence=confidence,
                sent_at=sent_at,
                received_at=time.monotonic(),
                tokens=tokens,
                raw=dist,
            )
            self.tokens += tokens
            self.calls += 1
            with self._lock:
                self._answers[track_id] = answer

    def _evaluate(
        self, state: dict[str, Any], order: tuple[str, ...]
    ) -> tuple[str, float, int, dict[str, float]]:
        if self._ask is not None:
            return self._ask(state, order)
        return _ask_jev(state, order)


def _ask_jev(
    state: dict[str, Any], order: tuple[str, ...]
) -> tuple[str, float, int, dict[str, float]]:
    from typesafe_sdk import AsyncTypeSafeClient

    async def _call():
        async with AsyncTypeSafeClient() as client:
            return await client.system_one(state, choice_questions(order))

    result = asyncio.run(_call())
    answer = result.choices["key"]
    tokens = 0
    usage = getattr(result, "usage", None)
    if usage is not None:
        tokens = int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
    dist = {}
    probs = getattr(answer, "probabilities", None) or {}
    if isinstance(probs, dict):
        dist = {str(key): float(value) for key, value in probs.items()}
    return str(answer.choice), float(getattr(answer, "confidence", 0.0) or 0.0), tokens, dist


class ScreenJudge:
    def __init__(self, enabled: bool = True, ask: Callable[..., Any] | None = None):
        self.enabled = enabled and (ask is not None or bool(os.environ.get("TYPESAFE_API_KEY")))
        self._ask = ask

    def ask(self, state: dict[str, Any]) -> tuple[str, float, int]:
        if self._ask is not None:
            return self._ask(state)
        if not self.enabled:
            return "unknown", 0.0, 0
        from typesafe_sdk import AsyncTypeSafeClient

        async def _call():
            async with AsyncTypeSafeClient() as client:
                return await client.system_one(state, screen_questions())

        result = asyncio.run(_call())
        answer = result.choices["screen"]
        usage = getattr(result, "usage", None)
        tokens = 0
        if usage is not None:
            tokens = int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
        return str(answer.choice), float(getattr(answer, "confidence", 0.0) or 0.0), tokens
