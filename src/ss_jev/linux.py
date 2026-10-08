"""Hyprland window lookup and a Wayland-safe stop signal."""

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STOP_FILE = Path(os.environ.get("SS_JEV_STOP_FILE", "/tmp/ss-jev-stop"))


@dataclass(frozen=True)
class WindowGeom:
    address: str
    title: str
    x: int
    y: int
    width: int
    height: int
    focused: bool


def parse_clients(payload: str, title_part: str = "Swords") -> list[WindowGeom]:
    clients = json.loads(payload)
    needle = title_part.casefold()
    found = []
    for client in clients:
        title = str(client.get("title") or "")
        if needle not in title.casefold():
            continue
        at = client.get("at") or [0, 0]
        size = client.get("size") or [0, 0]
        found.append(
            WindowGeom(
                address=str(client.get("address") or ""),
                title=title,
                x=int(at[0]),
                y=int(at[1]),
                width=int(size[0]),
                height=int(size[1]),
                focused=bool(client.get("focusHistoryID") == 0),
            )
        )
    return found


def hypr_clients() -> str:
    return subprocess.check_output(["hyprctl", "clients", "-j"], text=True)


def hypr_active() -> dict:
    return json.loads(subprocess.check_output(["hyprctl", "activewindow", "-j"], text=True))


def find_game_window(title_part: str = "Swords") -> WindowGeom:
    matches = parse_clients(hypr_clients(), title_part)
    if not matches:
        raise RuntimeError(f"No Hyprland window title contains {title_part!r}. Launch the game under Proton first.")
    if len(matches) > 1:
        titles = ", ".join(repr(item.title) for item in matches)
        raise RuntimeError(f"Several windows match: {titles}")
    return matches[0]


def is_focused(window: WindowGeom) -> bool:
    try:
        active = hypr_active()
    except Exception:
        return False
    return str(active.get("address") or "") == window.address


class StopFlag:
    def __init__(self, path: Path = DEFAULT_STOP_FILE):
        self.path = path
        self._armed = False

    def arm(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self._armed = True
        signal.signal(signal.SIGUSR1, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        self.path.touch()

    def tripped(self) -> bool:
        return self._armed and self.path.exists()


HYPRLAND_BIND = (
    "# Add to hyprland.conf to stop a live run with F8:\n"
    "bind = , F8, exec, sh -c 'touch /tmp/ss-jev-stop; pkill -USR1 -f ss-jev || true'\n"
)
