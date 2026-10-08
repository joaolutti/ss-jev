"""Held-open Wayland region capture via wlr-screencopy.

grim's current build talks to ext-image-copy-capture, asks for the whole
output, then waits on a ready event that does not arrive from this session.
The older screencopy protocol copies one region into a reused shm buffer, so
the live loop does not spawn a process or encode a PNG per frame.
"""

import array
import mmap
import os
import socket
import struct
import time

import numpy as np

_WL_DISPLAY = 1
_WL_DISPLAY_GET_REGISTRY = 1
_WL_DISPLAY_SYNC = 0
_WL_DISPLAY_ERROR = 0
_WL_REGISTRY_BIND = 0
_WL_REGISTRY_GLOBAL = 0
_WL_SHM_CREATE_POOL = 0
_WL_SHM_POOL_CREATE_BUFFER = 0
_WL_SHM_POOL_DESTROY = 1
_WL_CALLBACK_DONE = 0
_SCREENCOPY_CAPTURE_REGION = 1
_FRAME_COPY = 0
_FRAME_DESTROY = 1
_FRAME_BUFFER = 0
_FRAME_READY = 2
_FRAME_FAILED = 3
_FRAME_BUFFER_DONE = 6
_XRGB8888 = 1
_ARGB8888 = 0


def _pad(size: int) -> int:
    return (size + 3) & ~3


def _encode_string(text: str) -> bytes:
    raw = text.encode() + b"\x00"
    return struct.pack("<I", len(raw)) + raw + b"\x00" * (_pad(len(raw)) - len(raw))


def _decode_string(payload: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<I", payload, offset)
    start = offset + 4
    text = payload[start : start + length].split(b"\x00", 1)[0].decode()
    return text, start + _pad(length)


def bgr_from_xrgb(buffer: memoryview | bytes, width: int, height: int, stride: int) -> np.ndarray:
    """XRGB8888 and ARGB8888 are B,G,R,X in memory on little-endian."""
    rows = np.frombuffer(buffer, dtype=np.uint8, count=height * stride).reshape(height, stride)
    pixels = rows[:, : width * 4].reshape(height, width, 4)
    return np.ascontiguousarray(pixels[:, :, :3])


class _Wayland:
    def __init__(self) -> None:
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        display = os.environ.get("WAYLAND_DISPLAY", "wayland-0")
        if not runtime:
            raise RuntimeError("WAYLAND_DISPLAY is unset; screencopy needs a Wayland session.")
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(2.0)
        self.socket.connect(os.path.join(runtime, display))
        self._buffer = bytearray()
        self._next_id = 2
        self.registry = self._alloc()
        self.output = 0
        self.shm = 0
        self.screencopy = 0
        self.output_x = 0
        self.output_y = 0
        self.output_width = 0
        self.output_height = 0
        self._send(_WL_DISPLAY, _WL_DISPLAY_GET_REGISTRY, struct.pack("<I", self.registry))
        self.globals: dict[str, tuple[int, int]] = {}
        self._roundtrip()
        self.shm = self._bind("wl_shm", 1)
        self.output = self._bind("wl_output", 3)
        self.screencopy = self._bind("zwlr_screencopy_manager_v1", 3)
        self._roundtrip()

    def close(self) -> None:
        try:
            self.socket.close()
        except OSError:
            return

    def _alloc(self) -> int:
        ident = self._next_id
        self._next_id += 1
        return ident

    def _bind(self, interface: str, version: int) -> int:
        if interface not in self.globals:
            raise RuntimeError(f"This compositor does not offer {interface}.")
        name, advertised = self.globals[interface]
        ident = self._alloc()
        payload = struct.pack("<I", name) + _encode_string(interface) + struct.pack("<II", min(version, advertised), ident)
        self._send(self.registry, _WL_REGISTRY_BIND, payload)
        return ident

    def _send(self, obj: int, opcode: int, payload: bytes = b"", fds: list[int] | None = None) -> None:
        header = struct.pack("<IHH", obj, opcode, 8 + len(payload))
        if fds:
            ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
            self.socket.sendmsg([header + payload], ancillary)
        else:
            self.socket.sendall(header + payload)

    def _recv(self) -> tuple[int, int, bytes]:
        while len(self._buffer) < 8:
            self._read_more()
        obj, opcode, size = struct.unpack_from("<IHH", self._buffer)
        while len(self._buffer) < size:
            self._read_more()
        payload = bytes(self._buffer[8:size])
        del self._buffer[:size]
        return obj, opcode, payload

    def _read_more(self) -> None:
        try:
            data, ancillary, _flags, _addr = self.socket.recvmsg(65536, 256)
        except TimeoutError as error:
            raise RuntimeError("Wayland screencopy timed out waiting for the compositor.") from error
        if not data and not ancillary:
            raise RuntimeError("Wayland display closed the screencopy connection.")
        for _level, kind, raw in ancillary:
            if kind == socket.SCM_RIGHTS:
                count = len(raw) // 4
                for fd in struct.unpack(f"{count}i", raw[: count * 4]):
                    os.close(fd)
        self._buffer += data

    def _roundtrip(self) -> None:
        callback = self._alloc()
        self._send(_WL_DISPLAY, _WL_DISPLAY_SYNC, struct.pack("<I", callback))
        while True:
            obj, opcode, payload = self._recv()
            if obj == callback and opcode == _WL_CALLBACK_DONE:
                return
            self._dispatch(obj, opcode, payload)

    def _dispatch(self, obj: int, opcode: int, payload: bytes) -> None:
        if obj == _WL_DISPLAY and opcode == _WL_DISPLAY_ERROR:
            _object_id, _code = struct.unpack_from("<II", payload)
            message, _offset = _decode_string(payload, 8)
            raise RuntimeError(f"Wayland error: {message}")
        if obj == self.registry and opcode == _WL_REGISTRY_GLOBAL:
            name = struct.unpack_from("<I", payload)[0]
            interface, offset = _decode_string(payload, 4)
            version = struct.unpack_from("<I", payload, offset)[0]
            self.globals[interface] = (name, version)
            return
        if obj == self.output and opcode == 0 and len(payload) >= 8:
            self.output_x, self.output_y = struct.unpack_from("<ii", payload)
            return
        if obj == self.output and opcode == 1 and len(payload) >= 16:
            _flags, self.output_width, self.output_height, _refresh = struct.unpack_from("<Iiii", payload)

    def capture_region(self, x: int, y: int, width: int, height: int, shm: "_ShmBuffer") -> tuple[int, int, int, int]:
        """Copy one output-local region. Return format, width, height, and stride."""
        local_x = x - self.output_x
        local_y = y - self.output_y
        frame = self._alloc()
        self._send(
            self.screencopy,
            _SCREENCOPY_CAPTURE_REGION,
            struct.pack("<IiIiiii", frame, 0, self.output, local_x, local_y, width, height),
        )
        # The buffer advertisement is queued in the capture request and flushed
        # on the next roundtrip. A bare recv times out with those events unsent.
        offered = self._await_frame(frame, _FRAME_BUFFER_DONE, "buffer advertisement")
        if offered is None:
            raise RuntimeError("wlr-screencopy offered no XRGB8888 buffer.")
        fmt, buf_w, buf_h, stride = offered
        if (buf_w, buf_h, stride) != (shm.width, shm.height, shm.stride) or shm.format != fmt:
            shm.resize(fmt, buf_w, buf_h, stride, self)
            self._roundtrip()
        self._send(frame, _FRAME_COPY, struct.pack("<I", shm.buffer_id))
        # The ready event is sent on the next output commit, after this request.
        self._await_frame(frame, _FRAME_READY, "frame ready")
        self._send(frame, _FRAME_DESTROY)
        return offered

    def _await_frame(self, frame: int, until: int, what: str) -> tuple[int, int, int, int] | None:
        """Read events until the given opcode. One sync flushes the compositor; the ready event may arrive later."""
        offered: tuple[int, int, int, int] | None = None
        seen = False
        synced = False
        callback = self._alloc()
        self._send(_WL_DISPLAY, _WL_DISPLAY_SYNC, struct.pack("<I", callback))
        deadline = time.monotonic() + 2.0
        while not seen or not synced:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"Wayland screencopy timed out waiting for {what}.")
            self.socket.settimeout(remaining)
            try:
                obj, opcode, payload = self._recv()
            except RuntimeError as error:
                raise RuntimeError(f"Wayland screencopy timed out waiting for {what}.") from error
            if obj == callback and opcode == _WL_CALLBACK_DONE:
                synced = True
                continue
            if obj != frame:
                self._dispatch(obj, opcode, payload)
                continue
            if opcode == _FRAME_BUFFER:
                fmt, buf_w, buf_h, stride = struct.unpack("<IIII", payload)
                if fmt in (_XRGB8888, _ARGB8888):
                    offered = (fmt, buf_w, buf_h, stride)
            elif opcode == _FRAME_FAILED:
                raise RuntimeError(f"wlr-screencopy failed while waiting for {what}.")
            elif opcode == until:
                seen = True
        self.socket.settimeout(2.0)
        return offered


class _ShmBuffer:
    def __init__(self) -> None:
        self.format = 0
        self.width = 0
        self.height = 0
        self.stride = 0
        self.buffer_id = 0
        self._fd = -1
        self._pool = 0
        self._map: mmap.mmap | None = None

    def resize(self, fmt: int, width: int, height: int, stride: int, wayland: _Wayland) -> None:
        self.close_pool(wayland)
        size = stride * height
        fd = os.memfd_create("ss-jev-capture")
        os.ftruncate(fd, size)
        pool = wayland._alloc()
        wayland._send(wayland.shm, _WL_SHM_CREATE_POOL, struct.pack("<Ii", pool, size), fds=[fd])
        buffer = wayland._alloc()
        wayland._send(
            pool,
            _WL_SHM_POOL_CREATE_BUFFER,
            struct.pack("<IiiiiI", buffer, 0, width, height, stride, fmt),
        )
        self._fd = fd
        self._pool = pool
        self.buffer_id = buffer
        self.format = fmt
        self.width = width
        self.height = height
        self.stride = stride
        self._map = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def view(self) -> memoryview:
        if self._map is None:
            raise RuntimeError("Screencopy buffer is not mapped.")
        return memoryview(self._map)

    def close_pool(self, wayland: _Wayland | None) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        if wayland is not None and self._pool:
            wayland._send(self._pool, _WL_SHM_POOL_DESTROY)
            self._pool = 0
        self.buffer_id = 0


class RegionCapture:
    """One Wayland connection and one reused shm buffer."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self._wayland = _Wayland()
        self._shm = _ShmBuffer()
        self._closed = False

    def grab(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Screencopy connection is closed.")
        _fmt, width, height, stride = self._wayland.capture_region(
            self.x, self.y, self.width, self.height, self._shm
        )
        frame = bgr_from_xrgb(self._shm.view(), width, height, stride)
        if (width, height) == (self.width, self.height):
            return frame
        local_x = self.x - self._wayland.output_x
        local_y = self.y - self._wayland.output_y
        if frame.shape[1] >= local_x + self.width and frame.shape[0] >= local_y + self.height:
            return np.ascontiguousarray(frame[local_y : local_y + self.height, local_x : local_x + self.width])
        raise RuntimeError(
            f"Screencopy returned {width}x{height}, not the {self.width}x{self.height} playfield."
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._shm.close_pool(self._wayland)
        finally:
            self._wayland.close()


def measure_region_rate(x: int, y: int, width: int, height: int, frames: int = 60) -> float:
    """Grab the requested number of frames as fast as the compositor answers, and return frames per second."""
    capture = RegionCapture(x, y, width, height)
    try:
        started = time.perf_counter()
        for _ in range(frames):
            capture.grab()
        elapsed = time.perf_counter() - started
    finally:
        capture.close()
    if elapsed <= 0:
        return 0.0
    return frames / elapsed
