"""Key sinks. Replay uses NullSink. Live holds one uinput keyboard open."""

from dataclasses import dataclass, field
from typing import Protocol

from .domain import Direction


class InputSink(Protocol):
    def press(self, key: Direction) -> None: ...
    def close(self) -> None: ...


@dataclass
class NullSink:
    presses: list[tuple[Direction, float]] = field(default_factory=list)
    now: float = 0.0

    def press(self, key: Direction) -> None:
        self.presses.append((key, self.now))

    def close(self) -> None:
        return None


class UInputSink:
    _CODES = {
        Direction.LEFT: "KEY_LEFT",
        Direction.RIGHT: "KEY_RIGHT",
        Direction.UP: "KEY_UP",
        Direction.DOWN: "KEY_DOWN",
    }

    def __init__(self):
        try:
            from evdev import UInput, ecodes
        except ImportError as error:
            raise RuntimeError("Install ss-jev[live] (python-evdev) for uinput keys.") from error
        self._ecodes = ecodes
        capabilities = {ecodes.EV_KEY: [getattr(ecodes, name) for name in self._CODES.values()]}
        try:
            self._ui = UInput(capabilities, name="ss-jev-keys")
        except OSError as error:
            raise RuntimeError(
                "Cannot open /dev/uinput. Add a udev rule for the device and your user, then replug or reboot."
            ) from error

    def press(self, key: Direction) -> None:
        code = getattr(self._ecodes, self._CODES[key])
        self._ui.write(self._ecodes.EV_KEY, code, 1)
        self._ui.syn()
        self._ui.write(self._ecodes.EV_KEY, code, 0)
        self._ui.syn()

    def close(self) -> None:
        self._ui.close()
