"""CLI: calibrate, replay, live, and a small Jev probe."""

import argparse
import json
import tomllib
from datetime import datetime
from pathlib import Path

from .domain import LIVE_800X500, PLAY_750X400, Calibration, Direction, Point, sizes_from_rings
from .events import EventLog, load_events
from .jev import CueJudge, ScreenJudge, choice_questions, cue_state
from .pipeline import Pipeline, run_source
from .replay import grade, write_report
from .sinks import NullSink, UInputSink
from .sources import ScreencopySource, VideoFileSource
from .supervisor import Supervisor

ROOT = Path.cwd()
for _candidate in (Path(__file__).resolve().parents[2], Path.cwd()):
    if (_candidate / "fixtures.toml").is_file():
        ROOT = _candidate
        break
DEFAULT_CALIBRATION = ROOT / "calibrations" / "live.json"
FIXTURES = ROOT / "fixtures.toml"


def load_fixtures() -> dict:
    return tomllib.loads(FIXTURES.read_text())["fixtures"]


def load_calibration(path: Path) -> Calibration:
    if path.is_file():
        return Calibration.load(path)
    if path.stem == "play-750x400":
        return PLAY_750X400
    if path.stem == "live":
        return LIVE_800X500
    raise SystemExit(f"No calibration at {path}")


def cmd_calibrate(args) -> None:
    import cv2

    if args.defaults:
        LIVE_800X500.save(args.output)
        print(f"Wrote {args.output} from the built-in 800×500 window.")
        return
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_MSEC, args.at_seconds * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"No frame at {args.at_seconds}s in {args.video}")
        window_size = (frame.shape[1], frame.shape[0])
    else:
        raise SystemExit("Pass --video PATH or --defaults.")
    from .perception import find_rings

    if args.interactive:
        x, y, width, height = map(int, cv2.selectROI("Select play area", frame, showCrosshair=True))
        cv2.destroyAllWindows()
        crop = frame[y : y + height, x : x + width]
        targets = {}
        for direction in (Direction.LEFT, Direction.RIGHT, Direction.UP):
            clicked: list[tuple[int, int]] = []

            def on_mouse(event, px, py, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked[:] = [(px, py)]

            title = f"Click {direction.value} ring"
            cv2.namedWindow(title)
            cv2.setMouseCallback(title, on_mouse)
            while True:
                preview = crop.copy()
                if clicked:
                    cv2.drawMarker(preview, clicked[0], (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
                cv2.imshow(title, preview)
                key = cv2.waitKey(30) & 0xFF
                if key in (13, 10) and clicked:
                    targets[direction] = Point(*clicked[0])
                    break
                if key == 27:
                    cv2.destroyAllWindows()
                    raise SystemExit("Cancelled.")
            cv2.destroyWindow(title)
        cv2.destroyAllWindows()
        roi = (x, y, width, height)
        rings = targets
    else:
        rings = find_rings(frame)
        roi = (0, 0, window_size[0], window_size[1])
    character = Point(
        (rings[Direction.LEFT].x + rings[Direction.RIGHT].x) / 2,
        (rings[Direction.LEFT].y + rings[Direction.RIGHT].y) / 2,
    )
    calibration = Calibration(
        version=3,
        name=args.output.stem,
        window_size=window_size,
        roi=roi,
        rings=rings,
        character=character,
        **sizes_from_rings(rings),
    )
    calibration.save(args.output)
    print(f"Saved {args.output}: rings { {d.value: [p.x, p.y] for d, p in rings.items()} }")


def cmd_replay_run(args) -> None:
    fixtures = load_fixtures()
    if args.fixture not in fixtures:
        raise SystemExit(f"Unknown fixture {args.fixture}. Have: {', '.join(fixtures)}")
    meta = fixtures[args.fixture]
    video = Path(meta["path"])
    calibration = load_calibration(ROOT / meta["calibration"] if not Path(meta["calibration"]).is_absolute() else Path(meta["calibration"]))
    out = args.output / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    events = EventLog(out / "events.jsonl")
    sink = NullSink()
    cue_judge = CueJudge(enabled=args.jev) if args.jev else CueJudge(enabled=False)
    screen_judge = ScreenJudge(enabled=args.jev) if args.jev else ScreenJudge(enabled=False)
    pipeline = Pipeline(
        calibration=calibration,
        events=events,
        sink=sink,
        cue_judge=cue_judge,
        screen_judge=screen_judge,
        supervisor=Supervisor(shadow=True),
        fps=args.fps,
    )
    source = VideoFileSource(
        video,
        realtime=args.realtime,
        roi=calibration.roi,
        expected_size=(calibration.roi[2], calibration.roi[3]),
    )
    annotated = out / "annotated.mp4" if args.annotate else None
    frames = run_source(
        pipeline,
        source,
        annotated,
        fps=args.fps,
        max_seconds=args.max_seconds or None,
    )
    result = grade(load_events(out / "events.jsonl"), calibration)
    write_report(result, out)
    print(
        f"Processed {frames} frames into {out}. "
        f"Fired {result['fired']} of {result['cues']} cues, "
        f"misses {result['misses']}, flickers {result['flickers']}."
    )


def cmd_replay_grade(args) -> None:
    calibration = load_calibration(args.calibration)
    result = grade(load_events(args.run_dir / "events.jsonl"), calibration)
    write_report(result, args.run_dir)
    print(f"Wrote {args.run_dir / 'report.md'}")


def cmd_live(args) -> None:
    from .linux import HYPRLAND_BIND, StopFlag, find_game_window, is_focused

    calibration = load_calibration(args.calibration)
    window = find_game_window(calibration.title)
    if [window.width, window.height] != list(calibration.window_size):
        raise SystemExit(
            f"Window size {window.width}x{window.height} != {calibration.window_size}. Recalibrate."
        )
    x = window.x + calibration.roi[0]
    y = window.y + calibration.roi[1]
    width, height = calibration.roi[2], calibration.roi[3]
    try:
        source = ScreencopySource((x, y, width, height), fps=args.fps)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    if args.send_keys:
        try:
            sink = UInputSink()
        except RuntimeError as error:
            source.close()
            raise SystemExit(
                f"{error} Live keys stay on the held-open uinput device. "
                "wtype is not used: each wtype press is a new keyboard and opens the game menu."
            ) from error
    else:
        sink = NullSink()
        print("Dry run: no keys. Pass --send-keys after a dry run looks right.")
    out = args.output / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(HYPRLAND_BIND)
    stop = StopFlag()
    stop.arm()
    events = EventLog(out / "events.jsonl")
    pipeline = Pipeline(
        calibration=calibration,
        events=events,
        sink=sink,
        cue_judge=CueJudge(enabled=args.jev),
        screen_judge=ScreenJudge(enabled=args.jev),
        fps=args.fps,
    )
    import cv2
    import time

    writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    started = time.monotonic()
    frames = 0
    try:
        for frame, now, index in source:
            if stop.tripped():
                events.write("stop", reason="f8_or_signal")
                break
            if not is_focused(window):
                events.write("stop", reason="focus_lost")
                break
            if frame.max() < 5:
                events.write("stop", reason="blank_capture")
                break
            pipeline.step(frame, now, index)
            writer.write(frame)
            frames += 1
            if time.monotonic() - started > args.seconds:
                events.write("stop", reason="time_limit")
                break
    finally:
        elapsed = time.monotonic() - started
        rate = (frames / elapsed) if elapsed > 0 else 0.0
        events.write("capture_rate", frames=frames, seconds=round(elapsed, 4), fps=round(rate, 2))
        writer.release()
        pipeline.close()
        source.close()
    print(f"Live run wrote {out} at {rate:.1f} fps. Each live recording is a new replay fixture.")


def cmd_jev_probe(args) -> None:
    from .domain import Track
    from .timing import assign_key

    calibration = load_calibration(args.calibration)
    track = Track(1, "apple", Point(120, 284), 0.0, 0.1, observations=2, samples=[(0.0, Point(80, 284)), (0.1, Point(120, 284))])
    assign_key(track, calibration)
    state = cue_state(track, 0, 0.9)
    orders = [
        ("left", "right", "up", "none"),
        ("none", "up", "right", "left"),
    ]
    if args.fake:
        print(json.dumps({"state": state, "would_ask": [list(order) for order in orders]}, indent=2))
        return
    import asyncio
    from typesafe_sdk import AsyncTypeSafeClient

    async def run():
        async with AsyncTypeSafeClient() as client:
            results = []
            for order in orders:
                result = await client.system_one(state, choice_questions(order))
                answer = result.choices["key"]
                results.append({"order": list(order), "choice": answer.choice, "confidence": answer.confidence})
            print(json.dumps(results, indent=2))
            if results[0]["choice"] != results[1]["choice"]:
                print("Option order moved the answer. Keep code as the source of truth.")

    asyncio.run(run())


def main() -> None:
    parser = argparse.ArgumentParser(description="Melee training bot with replay grading and Jev in shadow mode.")
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate", help="Write a calibration JSON")
    calibrate.add_argument("--video", type=Path)
    calibrate.add_argument("--at-seconds", type=float, default=2.0)
    calibrate.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION)
    calibrate.add_argument("--defaults", action="store_true")
    calibrate.add_argument("--interactive", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate)

    replay = commands.add_parser("replay")
    replay_commands = replay.add_subparsers(dest="replay_command", required=True)
    run = replay_commands.add_parser("run")
    run.add_argument("fixture")
    run.add_argument("--output", type=Path, default=ROOT / "runs")
    run.add_argument("--fps", type=float, default=27.0)
    run.add_argument("--realtime", action="store_true")
    run.add_argument("--annotate", action="store_true")
    run.add_argument("--jev", action="store_true", help="Call Jev in shadow mode (needs TYPESAFE_API_KEY)")
    run.add_argument(
        "--max-seconds",
        type=float,
        default=0,
        help="Stop after this many seconds of video. 0 replays the whole file.",
    )
    run.set_defaults(func=cmd_replay_run)
    grade_cmd = replay_commands.add_parser("grade")
    grade_cmd.add_argument("run_dir", type=Path)
    grade_cmd.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "calibrations" / "play-750x400.json",
    )
    grade_cmd.set_defaults(func=cmd_replay_grade)

    live = commands.add_parser("live", help="Capture the Proton window on Hyprland")
    live.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    live.add_argument("--send-keys", action="store_true")
    live.add_argument("--output", type=Path, default=ROOT / "runs")
    live.add_argument("--seconds", type=int, default=300)
    live.add_argument("--fps", type=float, default=60.0)
    live.add_argument("--jev", action="store_true")
    live.set_defaults(func=cmd_live)

    probe = commands.add_parser("jev-probe", help="Ask Jev the per-cue question twice with swapped option order")
    probe.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    probe.add_argument("--fake", action="store_true")
    probe.set_defaults(func=cmd_jev_probe)

    args = parser.parse_args()
    args.func(args)
