from __future__ import annotations

import math

import numpy as np
import pytest

WIDTH, HEIGHT = 640, 360
HFOV_RAD = 2.0
CAMERA_HEIGHT_M = 0.75
FAR_M = 10.0
_FOCAL = (WIDTH / 2) / math.tan(HFOV_RAD / 2)


def depth_scene(walls: list[tuple[float, float, float]] = ()) -> np.ndarray:
    """Z-depth `(360, 640)` of a flat floor under the training camera plus walls `(distance_m, left_m, right_m)`."""
    u = (np.arange(WIDTH, dtype=np.float32) - WIDTH / 2 + 0.5) / _FOCAL
    v = (np.arange(HEIGHT, dtype=np.float32) - HEIGHT / 2 + 0.5) / _FOCAL
    dx, dy = np.meshgrid(u, v)
    with np.errstate(divide="ignore"):
        depth = np.where(dy > 0.0, CAMERA_HEIGHT_M / np.where(dy > 0.0, dy, 1.0), np.inf).astype(np.float32)
    for distance, left, right in walls:
        lateral = dx * distance
        height = CAMERA_HEIGHT_M - dy * distance
        hit = (lateral >= -left) & (lateral <= -right) & (height >= 0.0) & (height <= 2.0)
        depth = np.where(hit & (distance < depth), np.float32(distance), depth)
    depth[depth > FAR_M] = np.inf
    return depth


@pytest.fixture(scope="session")
def clear_floor() -> np.ndarray:
    return depth_scene()


@pytest.fixture(scope="session")
def wall_ahead() -> np.ndarray:
    return depth_scene([(1.0, 3.0, -3.0)])
