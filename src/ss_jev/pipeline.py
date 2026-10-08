"""Frame loop: perceive, track, time, optionally log Jev, never wait on it."""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .domain import REFERENCE_RING_RADIUS, Calibration, Track
from .events import EventLog
from .jev import CueJudge, ScreenJudge, cue_state
from .perception import annotate, detect_cues
from .sinks import InputSink, NullSink
from .supervisor import Supervisor, collect_signals
from .timing import assign_key, should_fire, time_to_arrival
from .tracking import Tracker


@dataclass
class Pipeline:
    calibration: Calibration
    events: EventLog
    sink: InputSink
    tracker: Tracker | None = None
    cue_judge: CueJudge | None = None
    screen_judge: ScreenJudge | None = None
    supervisor: Supervisor | None = None
    fps: float = 27.0

    def __post_init__(self) -> None:
        scale = self.calibration.ring_radius / REFERENCE_RING_RADIUS
        self.tracker = self.tracker or Tracker(min_step=40.0 * scale, cold_step=80.0 * scale)
        self.supervisor = self.supervisor or Supervisor()
        self._cue_times: deque[float] = deque(maxlen=64)
        self._previous_t: float | None = None
        self._screen_pool: ThreadPoolExecutor | None = None
        self._screen_future = None
        if self.screen_judge and self.screen_judge.enabled:
            self._screen_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-screen")

    def close(self) -> None:
        if self.cue_judge:
            self.cue_judge.close()
        if self._screen_pool:
            self._screen_pool.shutdown(wait=False)
        self.sink.close()
        self.events.close()

    def step(self, frame, now: float, index: int) -> list[Track]:
        if isinstance(self.sink, NullSink):
            self.sink.now = now
        period = self._frame_period(now)
        detections = detect_cues(frame, self.calibration)
        confidence_by_center = {(round(d.center.x), round(d.center.y)): d.confidence for d in detections}
        known_ids = set(self.tracker.tracks)
        tracks = self.tracker.update(detections, now)
        for expired in self.tracker.expired:
            self.events.write(
                "track_lost",
                frame=index,
                t=round(now, 4),
                track=expired.id,
                cue=expired.kind,
                x=round(expired.center.x, 1),
                y=round(expired.center.y, 1),
                speed=round(expired.speed, 1),
                fired=expired.fired,
                rejected=expired.rejected,
            )
        if self.cue_judge:
            answers = self.cue_judge.harvest()
            self.cue_judge.attach(self.tracker.tracks, answers)
            for answer in answers:
                track = self.tracker.tracks.get(answer.track_id)
                code_key = track.assigned_key.value if track and track.assigned_key else None
                self.events.write(
                    "jev_cue",
                    frame=index,
                    t=round(now, 4),
                    track=answer.track_id,
                    choice=answer.choice,
                    confidence=round(answer.confidence, 4),
                    latency=round(answer.received_at - answer.sent_at, 4),
                    code_key=code_key,
                    agree=answer.choice == code_key if code_key else answer.choice == "none",
                    tokens=answer.tokens,
                    fired=bool(track.fired) if track else None,
                )
        in_flight = sum(1 for track in tracks if not track.fired and not track.rejected)
        for track in tracks:
            spawned = track.id not in known_ids
            if spawned:
                self._cue_times.append(now)
                self.events.write(
                    "cue_spawned",
                    frame=index,
                    t=round(now, 4),
                    track=track.id,
                    cue=track.kind,
                    x=round(track.center.x, 1),
                    y=round(track.center.y, 1),
                    speed=round(track.speed, 1),
                )
            previous_key = track.assigned_key
            key = assign_key(track, self.calibration)
            if key is not None and previous_key is None:
                tta = time_to_arrival(track, self.calibration.rings[key], self.calibration.ring_radius)
                track.predicted_arrival = now + tta if tta is not None else None
                if self.cue_judge and not track.jev_asked:
                    det_conf = confidence_by_center.get((round(track.center.x), round(track.center.y)), 0.7)
                    self.cue_judge.submit(
                        track,
                        cue_state(track, max(0, in_flight - 1), det_conf),
                        sent_at=time.monotonic(),
                    )
                self.events.write(
                    "key_assigned",
                    frame=index,
                    t=round(now, 4),
                    track=track.id,
                    cue=track.kind,
                    key=key.value,
                    x=round(track.center.x, 1),
                    y=round(track.center.y, 1),
                    speed=round(track.speed, 1),
                    tta=None if tta is None else round(tta, 4),
                )
                if tta is not None:
                    self.events.write(
                        "arrival_predicted",
                        frame=index,
                        t=round(now, 4),
                        track=track.id,
                        cue=track.kind,
                        key=key.value,
                        tta=round(tta, 4),
                        predicted_at=round(now + tta, 4),
                        speed=round(track.speed, 1),
                    )
            fire, reason, tta = should_fire(track, self.calibration, period)
            if (
                track.assigned_key
                and tta is not None
                and track.predicted_arrival is not None
                and abs((now + tta) - track.predicted_arrival) > 0.05
            ):
                track.predicted_arrival = now + tta
                self.events.write(
                    "arrival_predicted",
                    frame=index,
                    t=round(now, 4),
                    track=track.id,
                    cue=track.kind,
                    key=track.assigned_key.value,
                    tta=round(tta, 4),
                    predicted_at=round(now + tta, 4),
                    speed=round(track.speed, 1),
                )
            if fire and track.assigned_key:
                track.fired = True
                paused = bool(self.supervisor and self.supervisor.policy.paused)
                self.events.write(
                    "fired",
                    frame=index,
                    t=round(now, 4),
                    track=track.id,
                    cue=track.kind,
                    key=track.assigned_key.value,
                    reason=reason,
                    x=round(track.center.x, 1),
                    y=round(track.center.y, 1),
                    speed=round(track.speed, 1),
                    tta=None if tta is None else round(tta, 4),
                    sent=not paused,
                    rejected=track.rejected,
                )
                if not paused:
                    self.sink.press(track.assigned_key)
            self.events.write(
                "cue_observed",
                frame=index,
                t=round(now, 4),
                track=track.id,
                cue=track.kind,
                x=round(track.center.x, 1),
                y=round(track.center.y, 1),
                speed=round(track.speed, 1),
                assigned=None if not track.assigned_key else track.assigned_key.value,
                fired=track.fired,
                rejected=track.rejected,
            )
        self._step_supervisor(frame, now, index)
        return tracks

    def _frame_period(self, now: float) -> float:
        """One measured frame of lead. The configured fps is only the first guess."""
        previous = self._previous_t
        self._previous_t = now
        if previous is None:
            return 1.0 / self.fps
        return min(0.1, max(1.0 / 120.0, now - previous))

    def _step_supervisor(self, frame, now: float, index: int) -> None:
        supervisor = self.supervisor
        if supervisor is None:
            return
        if self._screen_future is not None and self._screen_future.done():
            try:
                screen, confidence, tokens = self._screen_future.result()
            except Exception:
                screen, confidence, tokens = "unknown", 0.0, 0
            action = supervisor.policy.apply(screen, confidence)
            supervisor.previous_screen = screen
            self.events.write(
                "screen_judgment",
                frame=index,
                t=round(now, 4),
                screen=screen,
                confidence=round(confidence, 4),
                action=action,
                paused=supervisor.policy.paused,
                tokens=tokens,
                shadow=supervisor.shadow,
            )
            self._screen_future = None
        if not supervisor.due(now):
            return
        supervisor.last_ask = now
        signals = collect_signals(
            frame,
            self.calibration,
            self._cue_times,
            now,
            supervisor.previous_screen,
        )
        self.events.write("scene_signals", frame=index, t=round(now, 4), **signals.as_state())
        if not self.screen_judge or not self.screen_judge.enabled or self._screen_pool is None:
            screen = (
                "paused"
                if signals.pause_button == "seen"
                else "training_active"
                if signals.rings_visible >= 2
                else "transition"
            )
            action = supervisor.policy.apply(screen, 0.9)
            supervisor.previous_screen = screen
            self.events.write(
                "screen_judgment",
                frame=index,
                t=round(now, 4),
                screen=screen,
                confidence=0.9,
                action=action,
                paused=supervisor.policy.paused,
                tokens=0,
                shadow=supervisor.shadow,
                source="code",
            )
            return
        self._screen_future = self._screen_pool.submit(self.screen_judge.ask, signals.as_state())


def run_source(
    pipeline: Pipeline,
    source,
    annotate_path=None,
    fps: float = 27.0,
    max_seconds: float | None = None,
) -> int:
    import cv2

    writer = None
    frames = 0
    try:
        for frame, now, index in source:
            if max_seconds is not None and max_seconds > 0 and now >= max_seconds:
                break
            tracks = pipeline.step(frame, now, index)
            frames += 1
            if annotate_path:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(annotate_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                detections = detect_cues(frame, pipeline.calibration)
                writer.write(annotate(frame, detections, pipeline.calibration, tracks))
    finally:
        if writer:
            writer.release()
        pipeline.close()
    return frames
