from ss_jev.domain import Direction, PLAY_750X400, Point, Track
from ss_jev.jev import CueJudge, cue_state
from ss_jev.timing import assign_key


def test_jev_is_asked_only_after_a_key_is_assigned():
    import numpy as np
    import cv2

    from ss_jev.events import EventLog
    from ss_jev.pipeline import Pipeline
    from ss_jev.sinks import NullSink

    class RecordingJudge:
        def __init__(self):
            self.states = []
            self.enabled = True

        def submit(self, track, state, sent_at):
            self.states.append(state)
            track.jev_asked = True

        def harvest(self):
            return []

        def attach(self, tracks, answers):
            return None

        def close(self):
            return None

    def apple_frame(x):
        frame = np.zeros((400, 750, 3), dtype=np.uint8)
        frame[:] = (40, 90, 40)
        for target in PLAY_750X400.rings.values():
            cv2.circle(frame, (int(target.x), int(target.y)), 32, (220, 220, 220), 3)
        cv2.circle(frame, (int(x), 284), 14, (0, 0, 255), -1)
        return frame

    judge = RecordingJudge()
    pipeline = Pipeline(PLAY_750X400, EventLog(), NullSink(), cue_judge=judge, fps=10)
    pipeline.step(apple_frame(80), 0.0, 0)
    assert judge.states == []
    pipeline.step(apple_frame(140), 0.1, 1)
    pipeline.close()
    assert judge.states
    cue = judge.states[0]["cue"]
    assert cue["approach_side"] == "left"
    assert cue["moving"] == "toward ring"


def test_option_order_is_passed_through_for_stability_checks():
    seen = []

    def fake_ask(state, order):
        seen.append(list(order))
        return "left", 0.8, 3, {"left": 0.8}

    judge = CueJudge(enabled=True, ask=fake_ask)
    track = Track(
        1,
        "apple",
        Point(140, 284),
        0.0,
        0.1,
        observations=2,
        samples=[(0.0, Point(80, 284)), (0.1, Point(140, 284))],
    )
    assign_key(track, PLAY_750X400)
    judge.submit(track, cue_state(track, 0, 0.9), sent_at=0.0)
    import time

    answers = []
    for _ in range(50):
        answers = judge.harvest()
        if answers:
            break
        time.sleep(0.01)
    judge.close()
    assert seen
    assert seen[0] == ["left", "right", "up", "none"]
    assert answers[0].choice == "left"
    assert track.assigned_key is Direction.LEFT
