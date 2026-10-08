"""Grade a replay run from the event log and write a report."""

import json
from collections import defaultdict
from math import inf
from pathlib import Path
from statistics import median
from typing import Any

from .domain import Calibration, Direction, Point, distance
from .events import load_events, of_kind


SPEED_BANDS = (
    (0, 200, "0-200"),
    (200, 400, "200-400"),
    (400, 700, "400-700"),
    (700, inf, "700+"),
)

JEV_DOLLARS_PER_MTOK = 0.042


def speed_band(speed: float) -> str:
    for low, high, name in SPEED_BANDS:
        if low <= speed < high:
            return name


def minute_bucket(timestamp: float) -> str:
    start = int(timestamp // 60)
    return f"{start}-{start + 1} min"


def _track_rows(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for event in events:
        track_id = event.get("track")
        if track_id is None:
            continue
        row = rows.setdefault(
            track_id,
            {
                "track": track_id,
                "cue": event.get("cue"),
                "samples": [],
                "fired": [],
                "predictions": [],
                "assigned": None,
                "rejected": False,
                "spawn_t": None,
                "speed": 0.0,
            },
        )
        if event["event"] == "cue_spawned":
            row["spawn_t"] = event["t"]
            row["cue"] = event.get("cue")
            row["speed"] = event.get("speed", 0.0)
        if event["event"] == "key_assigned":
            row["assigned"] = event.get("key")
            row["speed"] = event.get("speed", row["speed"])
        if event["event"] == "arrival_predicted":
            row["predictions"].append(event)
            row["speed"] = event.get("speed", row["speed"])
        if event["event"] == "fired":
            row["fired"].append(event)
            row["speed"] = event.get("speed", row["speed"])
        if event["event"] == "cue_observed":
            row["samples"].append(event)
            row["speed"] = event.get("speed", row["speed"])
            if event.get("rejected"):
                row["rejected"] = True
        if event["event"] == "track_lost" and event.get("rejected"):
            row["rejected"] = True
    return rows


def _position_at_press(samples: list[dict[str, Any]], fire: dict[str, Any]) -> tuple[float, float] | None:
    when = fire.get("t")
    if when is not None and samples:
        prior = [sample for sample in samples if sample.get("t", 0) <= when + 1e-6]
        chosen = prior[-1] if prior else samples[0]
        if "x" in chosen and "y" in chosen:
            return float(chosen["x"]), float(chosen["y"])
    if "x" in fire and "y" in fire:
        return float(fire["x"]), float(fire["y"])
    return None


def _passed_ring(x: float, y: float, key: str, ring: Point) -> bool:
    """The hero stands inside the three rings, so past means toward the hero."""
    if key == Direction.LEFT.value:
        return x > ring.x
    if key == Direction.RIGHT.value:
        return x < ring.x
    if key == Direction.UP.value:
        return y > ring.y
    return False


def _press_after_ring(row: dict[str, Any], fire: dict[str, Any], ring: Point) -> bool:
    position = _position_at_press(row["samples"], fire)
    if position is None or not row.get("assigned"):
        return False
    return _passed_ring(position[0], position[1], row["assigned"], ring)


def closest_approach(samples: list[dict[str, Any]], ring: Point) -> tuple[float, float] | None:
    if not samples:
        return None
    best = min(samples, key=lambda item: distance(Point(item["x"], item["y"]), ring))
    return best["t"], distance(Point(best["x"], best["y"]), ring)


def grade(events: list[dict[str, Any]], calibration: Calibration) -> dict[str, Any]:
    rows = _track_rows(events)
    cues: list[dict[str, Any]] = []
    misses = 0
    late_presses = 0
    double_presses = 0
    rejected_presses = 0
    for row in rows.values():
        assigned = row["assigned"]
        ring = calibration.rings.get(Direction(assigned)) if assigned else None
        actual = closest_approach(row["samples"], ring) if ring else None
        if actual is None and row["samples"] and calibration.rings:
            actual = min(
                (
                    closest_approach(row["samples"], point)
                    for point in calibration.rings.values()
                ),
                key=lambda item: inf if item is None else item[1],
            )
        predicted_at = row["predictions"][0]["predicted_at"] if row["predictions"] else None
        error = None
        arrived = False
        if actual:
            arrived = actual[1] <= calibration.ring_radius * 1.35
            if predicted_at is not None:
                error = predicted_at - actual[0]
        fired = row["fired"]
        observations = len(row["samples"])
        if len(fired) > 1:
            double_presses += 1
        if fired and row["rejected"]:
            rejected_presses += 1
        late = _press_after_ring(row, fired[0], ring) if fired and ring is not None and assigned else False
        flicker = observations < 2 and arrived and not fired and not row["rejected"]
        missed = late or (
            observations >= 2
            and assigned is not None
            and arrived
            and not fired
            and not row["rejected"]
        )
        if late:
            late_presses += 1
        if missed:
            misses += 1
        cues.append(
            {
                "track": row["track"],
                "cue": row["cue"],
                "assigned": assigned,
                "speed": row["speed"],
                "band": speed_band(row["speed"]),
                "minute": minute_bucket(row["spawn_t"] or 0.0),
                "predicted_at": predicted_at,
                "actual_at": None if not actual else actual[0],
                "actual_distance": None if not actual else actual[1],
                "error": error,
                "fired": bool(fired),
                "rejected": row["rejected"],
                "missed": missed,
                "late": late,
                "flicker": flicker,
                "arrived": arrived,
                "observations": observations,
            }
        )
    by_band: dict[str, list[float]] = defaultdict(list)
    misses_by_band: dict[str, int] = defaultdict(int)
    arrived_by_band: dict[str, int] = defaultdict(int)
    for cue in cues:
        if cue["error"] is not None:
            by_band[cue["band"]].append(cue["error"])
        if cue["arrived"]:
            arrived_by_band[cue["band"]] += 1
        if cue["missed"]:
            misses_by_band[cue["band"]] += 1
    jev_events = of_kind(events, "jev_cue")
    jev_latencies = [event["latency"] for event in jev_events if "latency" in event]
    late = 0
    agree = 0
    none_on_rejected = 0
    rejected_ids = {row["track"] for row in rows.values() if row["rejected"]}
    for event in jev_events:
        track = rows.get(event["track"])
        tta = None
        if track and track["predictions"]:
            tta = track["predictions"][0].get("tta")
        if tta is not None and event["latency"] > tta:
            late += 1
        if event.get("agree"):
            agree += 1
        if event.get("choice") == "none" and event["track"] in rejected_ids:
            none_on_rejected += 1
    tokens = sum(event.get("tokens", 0) or 0 for event in jev_events)
    tokens += sum(event.get("tokens", 0) or 0 for event in of_kind(events, "screen_judgment"))
    screens = of_kind(events, "screen_judgment")
    transition_window = [
        event for event in screens if 165 <= event.get("t", 0) <= 180
    ]
    return {
        "cues": len(cues),
        "fired": sum(1 for cue in cues if cue["fired"]),
        "misses": misses,
        "late_presses": late_presses,
        "flickers": sum(1 for cue in cues if cue["flicker"]),
        "double_presses": double_presses,
        "rejected_presses": rejected_presses,
        "mean_error_by_band": {
            band: (sum(values) / len(values) if values else None) for band, values in by_band.items()
        },
        "median_error_by_band": {band: (median(values) if values else None) for band, values in by_band.items()},
        "misses_by_band": dict(misses_by_band),
        "arrived_by_band": dict(arrived_by_band),
        "minute_fired": _count_by(cues, "minute", lambda cue: cue["fired"]),
        "jev": {
            "calls": len(jev_events),
            "median_latency": median(jev_latencies) if jev_latencies else None,
            "p95_latency": _percentile(jev_latencies, 95) if jev_latencies else None,
            "late_vs_tta": late,
            "agree": agree,
            "agree_rate": (agree / len(jev_events) if jev_events else None),
            "none_on_rejected": none_on_rejected,
            "tokens": tokens,
            "cost_usd": tokens / 1_000_000 * JEV_DOLLARS_PER_MTOK,
        },
        "supervisor": {
            "judgments": len(screens),
            "around_170s": [
                {"t": event["t"], "screen": event["screen"], "action": event["action"]}
                for event in transition_window
            ],
        },
        "per_cue": cues,
    }


def _count_by(cues: list[dict[str, Any]], key: str, predicate) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for cue in cues:
        if predicate(cue):
            counts[cue[key]] += 1
    return dict(counts)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]


def write_report(result: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    slim = {key: value for key, value in result.items() if key != "per_cue"}
    (directory / "report.json").write_text(json.dumps(slim, indent=2) + "\n")
    lines = [
        "# Replay report",
        "",
        f"- Cues tracked: {result['cues']}",
        f"- Fired: {result['fired']}",
        f"- Misses (no press near a ring, or a press after the cue passed the ring): {result['misses']}",
        f"- Late presses (cue already past the ring): {result['late_presses']}",
        f"- Flickers (one sample near a ring, no press): {result['flickers']}",
        f"- Double presses: {result['double_presses']}",
        f"- Presses on rejected blobs: {result['rejected_presses']}",
        "",
        "## Prediction error by speed band (seconds, predicted minus actual)",
        "",
        "| Band | Mean error | Median error | Arrived | Misses |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, _, name in SPEED_BANDS:
        mean = result["mean_error_by_band"].get(name)
        med = result["median_error_by_band"].get(name)
        lines.append(
            f"| {name} | {_fmt(mean)} | {_fmt(med)} | {result['arrived_by_band'].get(name, 0)} | {result['misses_by_band'].get(name, 0)} |"
        )
    lines += ["", "## Fires by elapsed minute", ""]
    for name, count in sorted(result["minute_fired"].items()):
        lines.append(f"- {name}: {count}")
    jev = result["jev"]
    lines += [
        "",
        "## Jev (shadow)",
        "",
        f"- Calls: {jev['calls']}",
        f"- Median latency: {_fmt(jev['median_latency'])} s",
        f"- 95th percentile latency: {_fmt(jev['p95_latency'])} s",
        f"- Late vs time-to-arrival at assign: {jev['late_vs_tta']}",
        f"- Agree with code: {jev['agree_rate']}",
        f"- `none` on rejected tracks: {jev['none_on_rejected']}",
        f"- Tokens: {jev['tokens']} (about ${jev['cost_usd']:.6f} at $0.042/MTok)",
        "",
        "## Supervisor around 170s",
        "",
    ]
    if result["supervisor"]["around_170s"]:
        for event in result["supervisor"]["around_170s"]:
            lines.append(f"- t={event['t']:.2f}s screen={event['screen']} action={event['action']}")
    else:
        lines.append("- No judgments in 165–180s.")
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"
