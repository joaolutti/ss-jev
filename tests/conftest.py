from pathlib import Path

import pytest

from ss_jev.domain import PLAY_750X400, Calibration, Direction, Point

FIXTURE_ROOT = Path("/home/jl/Downloads/ss-automation")


@pytest.fixture
def calibration() -> Calibration:
    return PLAY_750X400


@pytest.fixture
def rings():
    return PLAY_750X400.rings


def moving_track(kind, start: Point, end: Point, dt: float = 0.1, track_id: int = 1):
    from ss_jev.domain import Track

    return Track(
        id=track_id,
        kind=kind,
        center=end,
        first_seen=0.0,
        last_seen=dt,
        observations=2,
        samples=[(0.0, start), (dt, end)],
    )
