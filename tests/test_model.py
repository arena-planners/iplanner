from __future__ import annotations

import dataclasses
import resource
import time

import numpy as np
import pytest
import torch
from arena_planners.sdk import Step
from conftest import depth_scene

import planner

_GOAL_AHEAD = np.array([4.0, 0.0, 0.0], dtype=np.float32)
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@pytest.fixture(scope="module", params=_DEVICES)
def policy(request: pytest.FixtureRequest) -> planner.Policy:
    if not planner.WEIGHTS.is_file():
        pytest.skip(f"weights missing at {planner.WEIGHTS}, run `arena feature planners add iplanner`")
    device = torch.device(request.param)
    rss_before = _rss_mib()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loaded = planner.Policy(planner.WEIGHTS, device)
    elapsed = time.perf_counter() - start
    gpu = f", {torch.cuda.memory_allocated() / 2**20:.0f} MiB allocated on cuda" if device.type == "cuda" else ""
    print(f"\niplanner load on {device}: {elapsed:.2f} s, rss +{_rss_mib() - rss_before:.0f} MiB{gpu}")
    return loaded


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _plan(policy: planner.Policy, depth: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    return policy.plan(planner.preprocess_depth(depth), goal)


def test_clear_floor_path_heads_to_the_goal(policy: planner.Policy, clear_floor: np.ndarray):
    keypoints, waypoints, fear = _plan(policy, clear_floor, _GOAL_AHEAD)
    np.set_printoptions(precision=3, suppress=True)
    print(f"clear floor, goal 4 m ahead: fear {fear:.3f}, keypoints\n{keypoints}")
    assert waypoints.shape == (51, 3)
    assert np.isfinite(waypoints).all()
    assert keypoints[-1, 0] > 2.0
    assert abs(keypoints[-1, 1]) < 1.0
    assert fear < planner.FEAR_THRESHOLD


@pytest.mark.parametrize(("side", "goal_y"), [("left", 3.0), ("right", -3.0)])
def test_goal_side_bends_the_path(policy: planner.Policy, clear_floor: np.ndarray, side: str, goal_y: float):
    keypoints, _, fear = _plan(policy, clear_floor, np.array([3.0, goal_y, 0.0], dtype=np.float32))
    print(f"goal {side}: fear {fear:.3f}, last keypoint {keypoints[-1]}")
    assert np.sign(keypoints[-1, 1]) == np.sign(goal_y)
    assert abs(keypoints[-1, 1]) > 0.5


def test_wall_ahead_raises_fear(policy: planner.Policy, clear_floor: np.ndarray, wall_ahead: np.ndarray):
    _, _, clear = _plan(policy, clear_floor, _GOAL_AHEAD)
    _, _, blocked = _plan(policy, wall_ahead, _GOAL_AHEAD)
    print(f"fear clear {clear:.3f}, wall at 1 m {blocked:.3f}")
    assert blocked > clear
    assert blocked > planner.FEAR_THRESHOLD


def test_gap_in_the_wall_steers_through_it(policy: planner.Policy):
    gap = depth_scene([(1.5, 3.0, 0.8), (1.5, -0.8, -3.0)])
    keypoints, _, fear = _plan(policy, gap, _GOAL_AHEAD)
    print(f"gap ahead: fear {fear:.3f}, keypoints\n{keypoints}")
    assert keypoints[-1, 0] > 1.0
    assert abs(keypoints[-1, 1]) < 0.8


def test_same_observation_gives_same_plan(policy: planner.Policy, clear_floor: np.ndarray):
    first = _plan(policy, clear_floor, _GOAL_AHEAD)
    second = _plan(policy, clear_floor, _GOAL_AHEAD)
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_fear_reaction_holds_the_robot_at_a_wall(policy: planner.Policy, wall_ahead: np.ndarray):
    config = dataclasses.replace(planner.Config(), device=str(policy.device), is_fear_act=True)
    iplanner = planner.IPlanner(policy, config)
    iplanner.reset({"goal_pose": {"x": 4.0, "y": 0.0, "theta": 0.0}})
    features = {"depth": wall_ahead, "robot_pose": np.zeros(3)}
    steps = [iplanner.step(features) for _ in range(config.buffer_size + 2)]
    assert all(len(step.chunk) == 50 for step in steps[: config.buffer_size])
    assert steps[-1] == Step()
    assert iplanner.fear.active


def test_step_latency(policy: planner.Policy, clear_floor: np.ndarray):
    features = {"depth": clear_floor, "robot_pose": np.zeros(3)}
    iplanner = planner.IPlanner(policy, planner.Config())
    iplanner.reset({"goal_pose": {"x": 4.0, "y": 0.0, "theta": 0.0}})
    iplanner.step(features)
    latencies = []
    for _ in range(10):
        _sync(policy.device)
        start = time.perf_counter()
        result = iplanner.step(features)
        _sync(policy.device)
        latencies.append(1000 * (time.perf_counter() - start))
    assert len(result.chunk) == 50
    peak = f", peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB" if policy.device.type == "cuda" else ""
    print(f"step on {policy.device}: mean {np.mean(latencies):.1f} ms, min {min(latencies):.1f} ms{peak}")
