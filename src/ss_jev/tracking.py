"""Associate detections across frames with a speed-scaled match radius."""

from .domain import Detection, Point, Track, distance


class Tracker:
    def __init__(
        self,
        min_step: float = 40.0,
        cold_step: float = 80.0,
        lookahead: float = 1.5,
        expiry_seconds: float = 0.25,
        default_dt: float = 1 / 27,
    ):
        self.min_step = min_step
        self.cold_step = cold_step
        self.lookahead = lookahead
        self.expiry_seconds = expiry_seconds
        self.default_dt = default_dt
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.expired: list[Track] = []

    def match_radius(self, track: Track, dt: float) -> float:
        if track.observations < 2 or track.speed < 1:
            return max(self.cold_step, self.min_step)
        return max(self.min_step, track.speed * max(dt, 1e-3) * self.lookahead)

    def update(self, detections: list[Detection], now: float) -> list[Track]:
        self.expired = [
            track
            for track in self.tracks.values()
            if now - track.last_seen > self.expiry_seconds
        ]
        self.tracks = {
            key: track
            for key, track in self.tracks.items()
            if now - track.last_seen <= self.expiry_seconds
        }
        available = set(self.tracks)
        updated: list[Track] = []
        for detection in sorted(detections, key=lambda item: -item.confidence):
            matches: list[tuple[float, int]] = []
            for key in available:
                track = self.tracks[key]
                if track.kind != detection.kind:
                    continue
                dt = now - track.last_seen if track.last_seen < now else self.default_dt
                radius = self.match_radius(track, dt)
                gap = distance(track.center, detection.center)
                if gap <= radius:
                    matches.append((gap, key))
            if matches:
                _, key = min(matches)
                available.remove(key)
                track = self.tracks[key]
                track.center = detection.center
                track.last_seen = now
                track.observations += 1
                track.samples.append((now, detection.center))
                if len(track.samples) > 8:
                    track.samples = track.samples[-8:]
            else:
                key = self._next_id
                self._next_id += 1
                track = Track(
                    id=key,
                    kind=detection.kind,
                    center=detection.center,
                    first_seen=now,
                    last_seen=now,
                    samples=[(now, detection.center)],
                )
                self.tracks[key] = track
            updated.append(track)
        return updated
