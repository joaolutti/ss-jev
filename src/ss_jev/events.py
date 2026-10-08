"""Append-only JSONL event log."""

import json
from pathlib import Path
from typing import Any, Iterable


class EventLog:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._file = path.open("w") if path else None

    def write(self, kind: str, **data: Any) -> dict[str, Any]:
        event = {"event": kind, **data}
        self.events.append(event)
        if self._file:
            self._file.write(json.dumps(event) + "\n")
            self._file.flush()
        return event

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def of_kind(events: Iterable[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == kind]
