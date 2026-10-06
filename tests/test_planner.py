from __future__ import annotations

import dataclasses
import math
import pathlib

import numpy as np
import pytest
import torch
import yaml
from arena_planners import goal, weights
from arena_planners.bridge.chunk import check_chunk, chunk_config_from_manifest
from arena_planners.sdk import Step

import planner
from iplanner_model.planner_net import PlannerNet
from iplanner_model.traj_opt import TrajOpt

_PLANNER_DIR = pathlib.Path(planner.__file__).parent
_CPU = torch.device("cpu")
_GOAL_AHEAD = {"x": 4.0, "y": 0.0, "theta": 0.0}


def _manifest() -> dict:
    with open(_PLANNER_DIR / "planner.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def random_policy(tmp_path_factory: pytest.TempPathFactory) -> planner.Policy:
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("weights") / "plannernet_state_dict.pt"
    torch.save(PlannerNet(planner.ENCODER_CHANNEL, planner.KEYPOINTS).state_dict(), path)
    return planner.Policy(path, _CPU)


@pytest.fixture
def iplanner(random_policy: planner.Policy) -> planner.IPlanner:
    return planner.IPlanner(random_policy, planner.Config())


def test_config_defaults_match_upstream_vehicle_sim_with_fear_off():
    config = planner.config_from({})
    assert config == planner.Config()
    assert (config.conv_dist, config.buffer_size, config.angular_thred, config.track_dist) == (0.25, 3, 0.5, 0.5)
    assert config.depth_max == 10.0 and not config.is_fear_act and not config.image_flip


def test_config_casts_launch_override_strings():
    config = planner.config_from({"is_fear_act": "false", "buffer_size": "10", "depth_max": "8", "device": "cpu"})
    assert config.is_fear_act is False
    assert config.buffer_size == 10 and isinstance(config.buffer_size, int)
    assert config.depth_max == 8.0 and isinstance(config.depth_max, float)
    assert config.device == "cpu"


def test_config_refuses_unknown_keys():
    with pytest.raises(ValueError, match="unknown config keys"):
        planner.config_from({"fear": 1})
    with pytest.raises(ValueError, match="boolean"):
        planner.config_from({"image_flip": "maybe"})


def test_manifest_config_keys_are_the_config_fields():
    assert set(_manifest()["config"]) == {field.name for field in dataclasses.fields(planner.Config)}


def test_depth_preprocess_zeroes_invalid_and_far_pixels():
    depth = np.full((360, 640), 3.0, dtype=np.float32)
    depth[0, :4] = [np.nan, np.inf, -np.inf, 10.5]
    depth[1, 0] = 10.0
    out = planner.preprocess_depth(depth)
    assert out.shape == (1, 3, 360, 640) and out.dtype == torch.float32
    np.testing.assert_array_equal(out[0, 0, 0, :4].numpy(), [0.0, 0.0, 0.0, 0.0])
    assert out[0, 0, 1, 0].item() == 10.0
    assert out[0, 0, 5, 5].item() == 3.0
    assert torch.equal(out[0, 0], out[0, 1]) and torch.equal(out[0, 0], out[0, 2])
    assert np.isnan(depth[0, 0])


def test_depth_preprocess_scales_millimeters():
    depth = np.full((360, 640), 2500, dtype=np.uint16)
    depth[0, 0] = 12000
    out = planner.preprocess_depth(depth)
    assert out[0, 0, 5, 5].item() == pytest.approx(2.5)
    assert out[0, 0, 0, 0].item() == 0.0


def test_depth_preprocess_resizes_to_crop_size():
    depth = np.full((256, 256, 1), 4.0, dtype=np.float32)
    assert planner.preprocess_depth(depth).shape == (1, 3, 360, 640)


def test_depth_preprocess_flip_rotates_by_180_degrees():
    depth = np.zeros((360, 640), dtype=np.float32)
    depth[0, 0] = 5.0
    assert planner.preprocess_depth(depth, image_flip=True)[0, 0, -1, -1].item() == 5.0
    assert planner.preprocess_depth(depth)[0, 0, 0, 0].item() == 5.0


def test_goal_in_robot_frame_rotates_by_heading():
    robot = np.array([1.0, 1.0, math.pi / 2])
    encoded = planner.goal_in_robot_frame(robot, {"x": 1.0, "y": 3.0, "theta": 0.0})
    assert encoded.dtype == np.float32
    np.testing.assert_allclose(encoded, [2.0, 0.0, 0.0], atol=1e-6)


def test_goal_in_robot_frame_left_is_positive_y():
    np.testing.assert_allclose(planner.goal_in_robot_frame(np.zeros(3), {"x": 0.0, "y": 2.0}), [0.0, 2.0, 0.0])
    np.testing.assert_allclose(planner.goal_in_robot_frame(np.zeros(3), {"x": 3.0, "y": -1.0}), [3.0, -1.0, 0.0])


def test_forward_tracking_follows_upstream_cone():
    straight = np.array([[0.1, 0.0, 0.0], [0.6, 0.0, 0.0], [1.2, 0.0, 0.0]])
    assert planner.is_forward_tracking(straight, 0.5, 0.5)
    diagonal = np.array([[0.1, 0.0, 0.0], [0.4, 0.4, 0.0], [0.5, 0.9, 0.0]])
    assert planner.is_forward_tracking(diagonal, 0.5, 0.5)
    sideways = np.array([[0.1, 0.1, 0.0], [0.2, 0.7, 0.0], [0.2, 1.5, 0.0]])
    assert not planner.is_forward_tracking(sideways, 0.5, 0.5)
    short = np.array([[0.1, 0.3, 0.0], [0.2, 0.4, 0.0]])
    assert planner.is_forward_tracking(short, 0.5, 0.5)


def test_fear_reaction_counts_forward_high_fear_only():
    reaction = planner.FearReaction(buffer_size=3)
    assert not any(reaction.update(0.9, True) for _ in range(3))
    assert reaction.update(0.9, True)
    assert reaction.update(0.9, True) and reaction.buffer == 4
    assert all(reaction.update(0.1, True) for _ in range(3))
    assert not reaction.update(0.1, True)
    assert not any(reaction.update(0.9, False) for _ in range(10))
    assert not any(reaction.update(0.4, True) for _ in range(10))


def test_spline_passes_through_keypoints_from_the_origin():
    keypoints = torch.tensor([[[0.5, 0.1, 0.0], [1.0, 0.3, 0.0], [1.5, 0.6, 0.0], [2.0, 1.0, 0.0], [2.5, 1.5, 0.0]]])
    waypoints = TrajOpt().TrajGeneratorFromPFreeRot(keypoints, step=planner.SPLINE_STEP)
    assert waypoints.shape == (1, 51, 3)
    torch.testing.assert_close(waypoints[0, 0], torch.zeros(3))
    torch.testing.assert_close(waypoints[0, 10::10], keypoints[0], atol=1e-5, rtol=0)


def test_chunk_drops_the_origin_and_keeps_x_forward_y_left():
    waypoints = np.zeros((51, 3))
    waypoints[:, 0] = np.linspace(0.0, 5.0, 51)
    waypoints[:, 1] = np.linspace(0.0, 1.0, 51)
    waypoints[:, 2] = 0.3
    chunk = planner.to_chunk(waypoints)
    assert len(chunk) == 50 and all(len(point) == 2 for point in chunk)
    np.testing.assert_allclose(chunk[0], [0.1, 0.02])
    np.testing.assert_allclose(chunk[-1], [5.0, 1.0])
    assert check_chunk("waypoints", chunk) == chunk


def test_policy_outputs_keypoints_spline_and_fear(random_policy: planner.Policy, clear_floor: np.ndarray):
    image = planner.preprocess_depth(clear_floor)
    keypoints, waypoints, fear = random_policy.plan(image, np.array([4.0, 0.0, 0.0], dtype=np.float32))
    assert keypoints.shape == (5, 3) and waypoints.shape == (51, 3)
    assert 0.0 <= fear <= 1.0
    np.testing.assert_allclose(waypoints[0], np.zeros(3))
    np.testing.assert_allclose(waypoints[10::10], keypoints, atol=1e-4)


def test_no_goal_holds_still(iplanner: planner.IPlanner, clear_floor: np.ndarray):
    iplanner.reset(None)
    assert iplanner.step({"depth": clear_floor, "robot_pose": np.zeros(3)}) == Step()


def test_missing_depth_or_pose_holds_still(iplanner: planner.IPlanner, clear_floor: np.ndarray):
    iplanner.reset({"goal_pose": _GOAL_AHEAD})
    assert iplanner.step({"robot_pose": np.zeros(3)}) == Step()
    assert iplanner.step({"depth": np.empty((0,), dtype=np.uint8), "robot_pose": np.zeros(3)}) == Step()
    assert iplanner.step({"depth": clear_floor, "robot_pose": None}) == Step()


def test_step_returns_a_fifty_point_waypoint_chunk(iplanner: planner.IPlanner, clear_floor: np.ndarray):
    iplanner.config = dataclasses.replace(iplanner.config, is_fear_act=False)
    iplanner.reset({"goal_pose": _GOAL_AHEAD})
    result = iplanner.step({"depth": clear_floor, "robot_pose": np.zeros(3)})
    assert not result.action and not result.command and not result.signal
    assert len(result.chunk) == 50
    assert check_chunk("waypoints", result.chunk) == result.chunk


def test_arrival_within_conv_dist_holds_still_and_signals(iplanner: planner.IPlanner, clear_floor: np.ndarray):
    iplanner.reset({"goal_pose": {"x": 0.2, "y": 0.1, "theta": 0.0}, "signal": "arrived"})
    features = {"depth": clear_floor, "robot_pose": np.zeros(3)}
    assert iplanner.step(features) == Step(signal="arrived")
    assert iplanner.step({**features, "robot_pose": np.array([-3.0, 0.0, 0.0])}) == Step(signal="arrived")
    iplanner.reset({"goal_pose": {"x": 0.2, "y": 0.1, "theta": 0.0}})
    assert iplanner.step(features) == Step()


def test_reset_clears_the_fear_reaction(iplanner: planner.IPlanner):
    iplanner.fear.update(0.9, True)
    iplanner.reset({"goal_pose": _GOAL_AHEAD})
    assert iplanner.fear.buffer == 0 and not iplanner.fear.active and not iplanner.arrived


def test_manifest_is_a_point_goal_waypoint_planner():
    manifest = _manifest()
    assert manifest["action_type"] == "waypoints"
    assert goal.declared(manifest) == frozenset({"pose"})
    assert manifest["signals"] == ["arrived"]
    assert manifest["sensor_needs"] == ["depth"]
    config = chunk_config_from_manifest(manifest)
    assert config.execute_s == pytest.approx(1.0 / manifest["rate_hz"], abs=1e-3)
    depth = manifest["observations"]["datasources"]["depth"]["params"]
    assert depth["sensor"] == "depth"
    assert tuple(depth["output_size"]) == (planner.CROP_SIZE[1], planner.CROP_SIZE[0])


def test_weights_manifest_points_at_the_mirror():
    entries = weights.read(_PLANNER_DIR)
    assert [entry.dest for entry in entries] == [str(planner.WEIGHTS.relative_to(_PLANNER_DIR))]
    assert entries[0].repo == "arena-rosnav/iplanner"
    assert len(entries[0].sha256) == 64


def test_steering_goal_prefers_the_plan_subgoal():
    goal_pose = {"x": 9.0, "y": 0.0, "theta": 0.0}
    assert planner.steering_goal({}, goal_pose) == goal_pose
    assert planner.steering_goal({"subgoal": None}, goal_pose) == goal_pose
    subgoal = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    assert planner.steering_goal({"subgoal": subgoal}, goal_pose) == {"x": 1.0, "y": 2.0, "theta": 0.5}


def test_manifest_steers_by_the_global_plan_subgoal():
    manifest = _manifest()
    assert manifest["depends"]["global_plan"] is True
    with pytest.raises(ValueError):
        goal.withhold_pose(manifest)
    sources = manifest["observations"]["datasources"]
    assert sources["subgoal"] == {"type": "SubgoalGenerator", "params": {"lookahead": 6.0}}
    assert sources["robot_pose_from_tf"]["type"] == "RobotPoseTFGenerator"
    assert sources["goal_pose"] == {"type": "geometry_msgs/PoseStamped", "params": {"topic": "goal_pose"}}
    assert sources["global_plan"] == {"type": "nav_msgs/Path", "params": {"topic": "plan"}}
    assert sources["costmap"] == {"type": "nav_msgs/OccupancyGrid", "params": {"topic": "global_costmap/costmap"}}
    names = list(sources)
    assert names.index("robot_pose_from_tf") < names.index("subgoal")


def test_arrival_is_judged_on_the_goal_not_the_subgoal(iplanner: planner.IPlanner, clear_floor: np.ndarray):
    iplanner.config = dataclasses.replace(iplanner.config, is_fear_act=False)
    iplanner.reset({"goal_pose": {"x": 0.2, "y": 0.1, "theta": 0.0}, "signal": "arrived"})
    far = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    assert iplanner.step({"depth": clear_floor, "robot_pose": np.zeros(3), "subgoal": far}) == Step(signal="arrived")
    iplanner.reset({"goal_pose": _GOAL_AHEAD})
    near = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    assert len(iplanner.step({"depth": clear_floor, "robot_pose": np.zeros(3), "subgoal": near}).chunk) == 50
