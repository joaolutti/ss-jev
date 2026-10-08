from pathlib import Path

import cv2
import numpy as np

from ss_jev.domain import PLAY_750X400, Direction, Point
from ss_jev.events import EventLog
from ss_jev.jev import CueJudge, ScreenJudge, cue_state
from ss_jev.pipeline import Pipeline, run_source
from ss_jev.replay import grade
from ss_jev.sinks import NullSink
from ss_jev.sources import VideoFileSource
from ss_jev.supervisor import InputPolicy
from ss_jev.domain import Track


def test_grader_records_prediction_error_and_independent_fires(tmp_path):
    events = EventLog()
    # The left cue arrives at t=0.4.
    for i, x in enumerate([80, 120, 160, 200, 221]):
        t = i * 0.1
        events.write("cue_observed", frame=i, t=t, track=1, cue="apple", x=x, y=284, speed=400, assigned="left", fired=i >= 4, rejected=False)
    events.write("cue_spawned", frame=0, t=0.0, track=1, cue="apple", x=80, y=284, speed=400)
    events.write("key_assigned", frame=1, t=0.1, track=1, cue="apple", key="left", speed=400)
    events.write("arrival_predicted", frame=1, t=0.1, track=1, cue="apple", key="left", tta=0.28, predicted_at=0.38, speed=400)
    events.write("fired", frame=4, t=0.4, track=1, cue="apple", key="left", speed=400, rejected=False)
    events.write("cue_spawned", frame=0, t=0.0, track=2, cue="spike_disk", x=620, y=283, speed=400)
    events.write("key_assigned", frame=1, t=0.1, track=2, cue="spike_disk", key="right", speed=400)
    events.write("arrival_predicted", frame=1, t=0.1, track=2, cue="spike_disk", key="right", tta=0.25, predicted_at=0.35, speed=400)
    for i, x in enumerate([620, 600, 580, 560, 556]):
        t = i * 0.1
        events.write("cue_observed", frame=i, t=t, track=2, cue="spike_disk", x=x, y=283, speed=400, assigned="right", fired=i >= 4, rejected=False)
    events.write("fired", frame=4, t=0.4, track=2, cue="spike_disk", key="right", speed=400, rejected=False)
    result = grade(events.events, PLAY_750X400)
    assert result["fired"] == 2
    assert result["misses"] == 0
    assert result["double_presses"] == 0
    errors = [cue["error"] for cue in result["per_cue"] if cue["error"] is not None]
    assert errors
    assert all(abs(err) < 0.2 for err in errors)


def test_press_after_the_cue_passed_the_ring_is_a_miss():
    def grade_press(key: str, samples: list[tuple[float, float]], fire_at: float):
        events = EventLog()
        x0, y0 = samples[0]
        events.write("cue_spawned", frame=0, t=0.0, track=1, cue="apple", x=x0, y=y0, speed=400)
        events.write("key_assigned", frame=0, t=0.0, track=1, cue="apple", key=key, speed=400)
        for index, (x, y) in enumerate(samples):
            events.write(
                "cue_observed",
                frame=index,
                t=index * 0.1,
                track=1,
                cue="apple",
                x=x,
                y=y,
                speed=400,
                assigned=key,
                fired=False,
                rejected=False,
            )
        events.write("fired", frame=1, t=fire_at, track=1, cue="apple", key=key, speed=400, rejected=False)
        return grade(events.events, PLAY_750X400)

    still_in_front = grade_press("left", [(160, 284), (200, 284)], 0.1)
    assert still_in_front["late_presses"] == 0
    assert still_in_front["misses"] == 0
    past_left = grade_press("left", [(160, 284), (250, 284)], 0.1)
    assert past_left["late_presses"] == 1
    assert past_left["misses"] == 1
    assert past_left["per_cue"][0]["late"] is True
    past_up = grade_press("up", [(389, 40), (389, 140)], 0.1)
    assert past_up["late_presses"] == 1
    assert past_up["misses"] == 1


def test_live_send_does_not_offer_a_wtype_fallback():
    import ss_jev.cli as cli
    import ss_jev.sinks as sinks
    import ss_jev.sources as sources

    assert not hasattr(sinks, "WtypeSink")
    assert not hasattr(sources, "GrimSource")
    text = Path(cli.__file__).read_text()
    assert "WtypeSink" not in text
    assert "Falling back to wtype" not in text


def test_xrgb_memory_is_read_as_bgr():
    from ss_jev.screencopy import bgr_from_xrgb

    frame = bgr_from_xrgb(bytes([10, 20, 30, 255, 1, 2, 3, 0]), 2, 1, 8)
    assert frame.shape == (1, 2, 3)
    assert frame[0, 0].tolist() == [10, 20, 30]
    assert frame[0, 1].tolist() == [1, 2, 3]


def test_one_frame_near_ring_is_a_flicker_not_a_miss():
    events = EventLog()
    events.write("cue_spawned", frame=0, t=0.0, track=1, cue="spike_disk", x=183, y=284, speed=0)
    events.write("cue_observed", frame=0, t=0.0, track=1, cue="spike_disk", x=183, y=284, speed=0, assigned="left", fired=False, rejected=False)
    events.write("key_assigned", frame=0, t=0.0, track=1, cue="spike_disk", key="left", speed=0)
    events.write("cue_spawned", frame=1, t=0.1, track=2, cue="spike_disk", x=596, y=280, speed=200)
    events.write("key_assigned", frame=1, t=0.1, track=2, cue="spike_disk", key="right", speed=200)
    events.write("cue_observed", frame=1, t=0.1, track=2, cue="spike_disk", x=596, y=280, speed=200, assigned="right", fired=False, rejected=False)
    events.write("cue_observed", frame=2, t=0.14, track=2, cue="spike_disk", x=580, y=280, speed=370, assigned="right", fired=False, rejected=False)
    result = grade(events.events, PLAY_750X400)
    assert result["flickers"] == 1
    assert result["misses"] == 1
    assert result["per_cue"][0]["flicker"] is True
    assert result["per_cue"][1]["missed"] is True


def test_video_source_and_pipeline_on_synthetic_clip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (750, 400))
    ring = PLAY_750X400.rings[Direction.LEFT]
    for i in range(12):
        frame = np.zeros((400, 750, 3), dtype=np.uint8)
        frame[:] = (40, 90, 40)
        for target in PLAY_750X400.rings.values():
            cv2.circle(frame, (int(target.x), int(target.y)), 32, (220, 220, 220), 3)
        x = int(40 + i * 18)
        cv2.circle(frame, (x, int(ring.y)), 14, (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    events = EventLog()
    pipeline = Pipeline(PLAY_750X400, events, NullSink(), fps=10)
    frames = run_source(pipeline, VideoFileSource(path, expected_size=(750, 400)), fps=10)
    assert frames == 12
    fired = [e for e in events.events if e["event"] == "fired"]
    spawned = [e for e in events.events if e["event"] == "cue_spawned"]
    assert spawned
    # The red circle moves toward the left ring. Expect a press, or at least a key assignment.
    assigned = [e for e in events.events if e["event"] == "key_assigned"]
    assert assigned
    assert assigned[0]["key"] == "left"


def test_shadow_jev_logs_without_changing_the_key():
    def fake_ask(state, order):
        return "none", 0.9, 12, {"none": 0.9}

    judge = CueJudge(enabled=True, ask=fake_ask)
    track = Track(1, "apple", Point(200, 284), 0.0, 0.1, observations=2, assigned_key=Direction.LEFT)
    judge.submit(track, cue_state(track, 0, 0.9), sent_at=0.0)
    import time
    for _ in range(50):
        answers = judge.harvest()
        if answers:
            break
        time.sleep(0.01)
    judge.close()
    assert answers
    assert answers[0].choice == "none"
    assert track.assigned_key is Direction.LEFT


def test_supervisor_hysteresis_needs_two_non_active():
    policy = InputPolicy()
    assert policy.apply("transition", 0.9) == "pending_pause"
    assert policy.paused is False
    assert policy.apply("transition", 0.9) == "pause"
    assert policy.paused is True
    assert policy.apply("training_active", 0.9) == "resume"
    assert policy.paused is False


def test_linux_window_parser():
    from ss_jev.linux import parse_clients

    payload = """[{"title": "Swords & Souls Neverseen", "at": [10, 20], "size": [750, 400], "address": "0x1", "focusHistoryID": 0}]"""
    windows = parse_clients(payload, "Swords")
    assert len(windows) == 1
    assert windows[0].width == 750
    assert windows[0].focused
