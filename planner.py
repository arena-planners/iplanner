"""iPlanner wrapper for the arena_planners bridge."""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib

import numpy as np
import torch
from arena_planners.sdk import Step, load_manifest, main_loop
from PIL import Image
from torchvision import transforms

from iplanner_model.planner_net import PlannerNet
from iplanner_model.traj_opt import TrajOpt

_DIR = pathlib.Path(__file__).parent
WEIGHTS = _DIR / "model" / "plannernet_state_dict.pt"
_FETCH_HINT = "run `arena feature planners add iplanner` (or `python -m arena_planners fetch iplanner`)"

CROP_SIZE = (360, 640)
ENCODER_CHANNEL = 16
KEYPOINTS = 5
SPLINE_STEP = 0.1
FEAR_THRESHOLD = 0.5
_DEPTH_MM_PER_M = 1000.0
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})

_log = logging.getLogger("iplanner")


@dataclasses.dataclass(frozen=True)
class Config:
    """Upstream's `vehicle_sim.yaml` planning and fear-reaction keys, the fear reaction off, plus the torch device."""

    device: str = "auto"
    depth_max: float = 10.0
    image_flip: bool = False
    conv_dist: float = 0.25
    is_fear_act: bool = False
    buffer_size: int = 3
    angular_thred: float = 0.5
    track_dist: float = 0.5


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"iplanner: expected a boolean, got {value!r}")


def config_from(raw: dict) -> Config:
    """`Config` from the manifest's `config` mapping, values cast to the field types, unknown keys refused."""
    fields = {field.name: field for field in dataclasses.fields(Config)}
    if unknown := sorted(set(raw) - set(fields)):
        raise ValueError(f"iplanner: unknown config keys {unknown}, accepted: {sorted(fields)}")
    casts = {"str": str, "float": float, "int": int, "bool": _as_bool}
    return Config(**{key: casts[fields[key].type](value) for key, value in raw.items()})


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


_DEPTH_TRANSFORM = transforms.Compose([transforms.Resize(CROP_SIZE), transforms.ToTensor()])


def steering_goal(features: dict, goal: dict) -> dict:
    """The global plan's lookahead point when the bridge sends one as `subgoal`, else the episode goal."""
    subgoal = features.get("subgoal")
    if subgoal is None:
        return goal
    x, y, theta = (float(v) for v in np.asarray(subgoal, dtype=np.float64)[:3])
    return {"x": x, "y": y, "theta": theta}


def preprocess_depth(depth: np.ndarray, depth_max: float = 10.0, image_flip: bool = False) -> torch.Tensor:
    """Upstream's node and `IPlannerAlgo` input `(1, 3, 360, 640)` in meters from `32FC1` meters or `16UC1` mm."""
    arr = np.asarray(depth)
    if arr.ndim == 3:
        arr = arr[..., 0]
    frame = arr.astype(np.float32) / _DEPTH_MM_PER_M if arr.dtype == np.uint16 else arr.astype(np.float32)
    frame[~np.isfinite(frame)] = 0.0
    frame[frame > depth_max] = 0.0
    if image_flip:
        frame = np.ascontiguousarray(np.rot90(frame, 2))
    return _DEPTH_TRANSFORM(Image.fromarray(frame)).expand(1, 3, -1, -1)


def goal_in_robot_frame(robot: np.ndarray, goal: dict) -> np.ndarray:
    """Model goal input `[fwd, left, 0]` in meters, the goal point in the frame of the robot pose `[x, y, theta]`."""
    rx, ry, rtheta = (float(v) for v in robot[:3])
    dx, dy = float(goal["x"]) - rx, float(goal["y"]) - ry
    fwd = dx * math.cos(rtheta) + dy * math.sin(rtheta)
    left = -dx * math.sin(rtheta) + dy * math.cos(rtheta)
    return np.array([fwd, left, 0.0], dtype=np.float32)


def is_forward_tracking(waypoints: np.ndarray, track_dist: float, angular_thred: float) -> bool:
    """Upstream's `isForwardTraking`: the first waypoint past `track_dist` lies within the heading cone."""
    for point in np.asarray(waypoints):
        norm = math.hypot(float(point[0]), float(point[1]))
        if norm > track_dist:
            return float(point[0]) / norm > 1.0 - angular_thred
    return True


def to_chunk(waypoints: np.ndarray) -> list[list[float]]:
    """Waypoints `[x, y]` in meters (x forward, y left), the spline samples after the origin."""
    return np.asarray(waypoints, dtype=np.float64)[1:, :2].tolist()


class FearReaction:
    """Upstream's `fearPathDetection` counter: holds the robot once fear stays high on a forward path."""

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.reset()

    def reset(self) -> None:
        self.buffer = 0
        self.active = False

    def update(self, fear: float, forward: bool) -> bool:
        if fear > FEAR_THRESHOLD and forward:
            if not self.active:
                self.buffer += 1
        elif self.active:
            self.buffer -= 1
        if self.buffer > self.buffer_size:
            self.active = True
        elif self.buffer <= 0:
            self.active = False
        return self.active


class Policy:
    """PlannerNet with the released weights and upstream's cubic-spline trajectory generator."""

    def __init__(self, path: pathlib.Path, device: torch.device) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"iplanner weights missing at {path}, {_FETCH_HINT}")
        self.device = device
        self.net = PlannerNet(ENCODER_CHANNEL, KEYPOINTS)
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.net.to(device).eval()
        self.traj_opt = TrajOpt()

    @torch.inference_mode()
    def plan(self, depth: torch.Tensor, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Keypoints `(5, 3)`, spline waypoints `(51, 3)` from the origin, and the fear score in [0, 1]."""
        goal_t = torch.from_numpy(np.asarray(goal, dtype=np.float32)).unsqueeze(0).to(self.device)
        keypoints, fear = self.net(depth.to(self.device), goal_t)
        waypoints = self.traj_opt.TrajGeneratorFromPFreeRot(keypoints, step=SPLINE_STEP)
        return keypoints[0].cpu().numpy(), waypoints[0].cpu().numpy(), float(fear.item())


class IPlanner:
    def __init__(self, policy: Policy, config: Config) -> None:
        self.policy = policy
        self.config = config
        self.fear = FearReaction(config.buffer_size)
        self._warned: set[str] = set()
        self.reset(None)

    def reset(self, initial_state: dict | None) -> None:
        state = initial_state or {}
        self.goal: dict | None = state.get("goal_pose")
        self.signal = str(state.get("signal", ""))
        self.arrived = False
        self.fear.reset()

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _log.warning(msg)

    def _arrived_step(self) -> Step:
        return Step(signal=self.signal) if self.signal else Step()

    def step(self, features: dict) -> Step:
        if self.goal is None:
            return Step()
        if self.arrived:
            return self._arrived_step()
        depth, robot_pose = features.get("depth"), features.get("robot_pose")
        if depth is None or np.asarray(depth).ndim < 2 or robot_pose is None:
            self._warn_once("inputs", "iplanner: depth or robot_pose missing, standing still")
            return Step()
        goal = goal_in_robot_frame(np.asarray(robot_pose), self.goal)
        if math.hypot(float(goal[0]), float(goal[1])) < self.config.conv_dist:
            self.arrived = True
            _log.info("iplanner: goal arrived")
            return self._arrived_step()
        image = preprocess_depth(depth, self.config.depth_max, self.config.image_flip)
        _, waypoints, fear = self.policy.plan(
            image, goal_in_robot_frame(np.asarray(robot_pose), steering_goal(features, self.goal))
        )
        if self.config.is_fear_act:
            forward = is_forward_tracking(waypoints, self.config.track_dist, self.config.angular_thred)
            if self.fear.update(fear, forward):
                self._warn_once("fear", "iplanner: current path prediction is invalid, holding still")
                return Step()
        return Step(chunk=to_chunk(waypoints))


_planner: IPlanner | None = None


def on_init(config: dict) -> None:
    global _planner
    settings = config_from(config)
    device = select_device(settings.device)
    _planner = IPlanner(Policy(WEIGHTS, device), settings)
    _log.warning("iplanner: loaded on %s", device)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    assert _planner is not None
    _planner.reset(initial_state)


def step(features: dict) -> Step:
    assert _planner is not None
    return _planner.step(features)


if __name__ == "__main__":
    logging.basicConfig()
    manifest = load_manifest(_DIR / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset, on_init=on_init)
